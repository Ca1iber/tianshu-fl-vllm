# SPDX-License-Identifier: Apache-2.0
"""Independent FP32 checks through actual native builder/forward wrappers."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import logging
import os
from pathlib import Path
import signal
import statistics
import sys
from types import SimpleNamespace

import torch
import vllm_fl
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import KVQuantMode
import vllm.v1.attention.backends.triton_attn as backend


def load_hybrid(path):
    spec=importlib.util.spec_from_file_location('tianshu_hybrid_validation',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def reference(q,k,v,table,qlens,lengths):
    outputs=[];begin=0
    for i,(ql,kl) in enumerate(zip(qlens,lengths)):
        pages=table[i,:(kl+15)//16].long()
        keys=k[pages].reshape(-1,2,128)[:kl].float().repeat_interleave(8,dim=1)
        values=v[pages].reshape(-1,2,128)[:kl].float().repeat_interleave(8,dim=1)
        query=q[begin:begin+ql].float().transpose(0,1)
        parts=[]
        for offset in range(0,ql,64):
            chunk=query[:,offset:offset+64]
            score=chunk@keys.permute(1,2,0)*128**-.5
            positions=torch.arange(offset,offset+chunk.shape[1],device=q.device)
            mask=torch.arange(kl,device=q.device)[None,:] <= (kl-ql+positions[:,None])
            score.masked_fill_(~mask[None],float('-inf'))
            parts.append(score.softmax(-1)@values.transpose(0,1))
        outputs.append(torch.cat(parts,dim=1).transpose(0,1));begin+=ql
    return torch.cat(outputs)


def error(a,b):
    a,b=a.float(),b.float();diff=(a-b).abs()
    return {'passed':bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not (diff>.02+.01*b.abs()).any()),
        'max_abs':diff.max().item(),'mean_abs':diff.mean().item(),
        'relative_rms':(diff.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-8)).item(),
        'atol':.02,'rtol':.01}


def run_case(hybrid,original_forward,name,qlens,lengths,strided,seed):
    torch.manual_seed(seed);n=len(qlens);total=sum(qlens);width=(max(lengths)+15)//16;pages=n*width
    table=torch.randperm(pages,device='cuda').int().reshape(n,width)
    q=torch.randn(total,20 if strided else 16,128,device='cuda',dtype=torch.bfloat16)[:,:16]
    k=torch.randn(pages*(2 if strided else 1),16,2,128,device='cuda',dtype=q.dtype)
    v=torch.randn_like(k)
    if strided:k,v=k[::2],v[::2]
    kv=torch.stack((k,v),dim=1)
    # This is the actual interleaved cache layout passed to native forward.
    k,v=kv.unbind(1)
    cpu=torch.tensor([0]+list(itertools.accumulate(qlens)),dtype=torch.int32)
    cu=cpu.cuda();used=torch.tensor(lengths,device='cuda',dtype=torch.int32)
    common=SimpleNamespace(num_reqs=n,num_actual_tokens=total,max_query_len=max(qlens),max_seq_len=max(lengths),
        query_start_loc=cu,query_start_loc_cpu=cpu,seq_lens=used,block_table_tensor=table,slot_mapping=None,
        causal=True,mm_req_doc_ranges=None)
    builder=object.__new__(backend.TritonAttentionMetadataBuilder)
    builder.seq_threshold_3D=64;builder.num_par_softmax_segments=16;builder.device=q.device
    builder.softmax_segm_output=torch.empty((64,16,16,128),device='cuda',dtype=torch.float32)
    builder.softmax_segm_max=torch.empty((64,16,16),device='cuda',dtype=torch.float32)
    builder.softmax_segm_expsum=torch.empty_like(builder.softmax_segm_max)
    metadata=builder.build(0,common)
    impl=object.__new__(backend.TritonAttentionImpl)
    impl.attn_type=AttentionType.DECODER;impl._is_per_token_head_quant=False;impl.kv_cache_dtype='auto'
    impl.fp8_dtype=torch.float8_e4m3fn;impl._kv_quant_mode=KVQuantMode.NONE
    impl.scale=128**-.5;impl.alibi_slopes=None;impl.use_alibi_sqrt=False;impl.sliding_window=(-1,-1)
    impl.logits_soft_cap=0.;impl.sinks=None;impl.chunk_lookback=-1;impl.use_td=False
    layer=SimpleNamespace(_q_scale=torch.ones(1,device='cuda'),_k_scale=torch.ones(1,device='cuda'),_v_scale=torch.ones(1,device='cuda'))
    base=torch.full((total+13,16,128),17.,device='cuda',dtype=q.dtype)
    candidate=torch.full_like(base,17.)
    # Prevent the original method from observing the wrapper context.
    original_forward(impl,layer,q,None,None,kv,metadata,base)
    impl.forward(layer,q,None,None,kv,metadata,candidate)
    expected=reference(q,k,v,table,qlens,lengths)
    route=getattr(metadata,'_fl_hybrid_route',None)
    native_check=error(base[:total],expected);candidate_check=error(candidate[:total],expected)
    if route is not None and not candidate_check['passed']:raise RuntimeError(f'{name}: candidate FP32 check failed: {candidate_check}')
    if route is None and not torch.equal(base,candidate):raise RuntimeError(f'{name}: native fallback changed')
    if not bool(torch.all(candidate[total:]==17.)):raise RuntimeError('Padded output tail overwritten')
    functions={'original':lambda:original_forward(impl,layer,q,None,None,kv,metadata,base),
               'hybrid':lambda:impl.forward(layer,q,None,None,kv,metadata,candidate)}
    graphs={};timings={key:[] for key in functions}
    for key,fn in functions.items():
        for _ in range(3):fn()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        graph.replay();torch.cuda.synchronize();graphs[key]=graph
    for trial in range(7):
        for key in ('original','hybrid') if trial%2==0 else ('hybrid','original'):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):graphs[key].replay()
            end.record();end.synchronize();timings[key].append(start.elapsed_time(end)/10)
    result={'name':name,'qlens':qlens,'kv_lengths':lengths,'strided':strided,'native_vs_fp32':native_check,
        'candidate_vs_fp32':candidate_check,'native_preserved_bitwise':bool(torch.equal(base,candidate)) if route is None else None,
        'num_decodes':route.num_decodes if route else None,'median_ms':{k:statistics.median(v) for k,v in timings.items()},'samples_ms':timings}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    signal.alarm(1800);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    original_forward=backend.TritonAttentionImpl.forward
    hybrid=load_hybrid(args.source)
    os.environ['VLLM_FL_ILUVATAR_HYBRID_ATTENTION']='1';os.environ['VLLM_FL_USE_FLAGGEMS_ATTN']='0'
    os.environ['FLAGGEMS_ILUVATAR_COMPACT_VARLEN']='1'
    config=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=2048,num_hidden_layers=42,
        num_attention_heads=16,num_key_value_heads=2),dtype=torch.bfloat16,quantization=None,is_multimodal_model=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,decode_context_parallel_size=1),
        lora_config=None,speculative_config=None)
    if not hybrid.configure(config,logging.getLogger('hybrid_validation')):raise RuntimeError('Hybrid not configured')
    data={'official_benchmark':False,'hybrid_source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),'cases':[]}
    cases=[('prefill2048',[2048],[2048],False),('chunk2048_prefix2048',[2048],[4096],True),
        ('decode1',[1],[4096],False),('decode64',[1]*64,[4096]*64,True),
        ('mixed64',[1]*63+[1985],[4096]*64,True),
        ('mixed_ragged',[1,1,1,7,129,511],[4096,4200,5120,2048,4096,5120],True),
        ('unreordered_fallback',[129,1],[4096,4096],False)]
    with torch.inference_mode():
        for i,(name,qlens,lengths,strided) in enumerate(cases):
            case=run_case(hybrid,original_forward,name,qlens,lengths,strided,2026101170+i)
            data['cases'].append(case);(args.output/'result.json').write_text(json.dumps(data,indent=2)+'\n')
            print(json.dumps(case),flush=True);torch.cuda.empty_cache()
    data['counts']=hybrid.COUNTS;data['complete']=True
    (args.output/'result.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
