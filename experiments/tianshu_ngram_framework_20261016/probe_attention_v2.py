# SPDX-License-Identifier: Apache-2.0
"""Native paged causal Q1/Q2 verification with independent FP32 and CPU checks."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import statistics
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True)
    args=p.parse_args();root=args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir(parents=True)
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>300;signal.alarm(min(1200,remaining))
    d=dict(phase='preparing',complete=False,runner_pid=os.getpid(),source_snapshots={},cases=[],events=[],routes=[],official_benchmark=False,adopted=False)
    def save():
        tmp=root/'result.tmp';tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(root/'result.json')
        print(json.dumps(dict(phase=d['phase'],cases=len(d['cases']),error=d.get('error'))),flush=True)
    def snapshot(path):
        path=Path(path).resolve();raw=path.read_bytes();d['source_snapshots'][str(path)]=hashlib.sha256(raw).hexdigest()
        dest=root/'source_snapshot'/path.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    save()
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tianshu_gemm_topp_geo_20261016'))
        from validation_profile_new import prepare
        _,environment,profile,sources=prepare(root,Path('/workspace/benchmark-results/20261016_gemm_topp_geo_service_tianshu'))
        os.environ.clear();os.environ.update(environment);runpy.run_path(str(Path(profile['PYTHONPATH'])/'sitecustomize.py'))
        import torch
        import vllm_fl
        import vllm.v1.attention.ops.triton_unified_attention as native
        import vllm.v1.attention.backends.triton_attn as backend
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.kv_cache_interface import KVQuantMode
        from types import SimpleNamespace
        import native_verify
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        cache=sys.modules['tianshu_native_cache_policy']
        fixture_path=Path(__file__).resolve().parents[1]/'tianshu_native_attention/validate_hybrid.py'
        assert hashlib.sha256(fixture_path.read_bytes()).hexdigest()==FIXTURE_SHA
        spec=importlib.util.spec_from_file_location('_ngram_fp32_fixture',fixture_path);fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        for path in sources:snapshot(path)
        import vllm.v1.attention.ops.triton_attention_helpers as helpers
        snapshot(helpers.__file__)
        for path in (Path(__file__),Path(native_verify.__file__),Path(native.__file__),Path(backend.__file__),Path(cache.__file__),fixture_path):snapshot(path)
        original=native.kernel_unified_attention;original_backend=backend.unified_attention;keepalive=[]
        def select(mode):
            native.kernel_unified_attention=original;backend.unified_attention=original_backend
            keepalive.append(native_verify.install(native,cache,backend,root/'split_generated',enable_split=(mode=='split'),record=d['routes'].append))
        def checked(a,b):
            result=fixture.error(a,b);assert result['passed'],result;return result
        plans=[(1,33,False,False),(8,257,True,False),(64,4096,False,False),(64,4097,True,True),(64,5120,False,False),(64,113,False,False),(64,225,True,True)]
        d['phase']='numerical';save()
        with torch.inference_mode():
            for case,(n,context,strided,ragged) in enumerate(plans):
                torch.manual_seed(2026102900+case)
                qlens=([1,2]*(n//2) if ragged else [2]*n);total=sum(qlens)
                lengths=[context+(i%3) for i in range(n)];width=(max(lengths)+47)//16;pages=n*width
                table=torch.randperm(pages,device='cuda').int().reshape(n,width)
                q=torch.randn(total,20 if strided else 16,128,device='cuda',dtype=torch.bfloat16)[:,:16]
                storage=torch.randn(pages*(2 if strided else 1),2,16,2,128,device='cuda',dtype=q.dtype)
                kv=storage[::2] if strided else storage;k,v=kv.unbind(1)
                cu=torch.tensor([0]+list(__import__('itertools').accumulate(qlens)),device='cuda',dtype=torch.int32)
                used=torch.tensor(lengths,device='cuda',dtype=torch.int32)
                # Actual native function and interleaved cache; no fake model or attention result.
                impl=object.__new__(backend.TritonAttentionImpl)
                impl.attn_type=AttentionType.DECODER;impl._is_per_token_head_quant=False;impl.kv_cache_dtype='auto'
                impl.fp8_dtype=torch.float8_e4m3fn;impl._kv_quant_mode=KVQuantMode.NONE
                impl.scale=128**-.5;impl.alibi_slopes=None;impl.use_alibi_sqrt=False;impl.sliding_window=(-1,-1)
                impl.logits_soft_cap=0.;impl.sinks=None;impl.chunk_lookback=-1;impl.use_td=False
                layer=SimpleNamespace(_q_scale=torch.ones(1,device='cuda'),_k_scale=torch.ones(1,device='cuda'),_v_scale=torch.ones(1,device='cuda'))
                builder=object.__new__(backend.TritonAttentionMetadataBuilder)
                builder.seq_threshold_3D=64;builder.num_par_softmax_segments=16;builder.device=q.device
                builder.softmax_segm_output=torch.empty((64,16,16,128),device='cuda',dtype=torch.float32)
                builder.softmax_segm_max=torch.empty((64,16,16),device='cuda',dtype=torch.float32);builder.softmax_segm_expsum=torch.empty_like(builder.softmax_segm_max)
                common=SimpleNamespace(num_reqs=n,num_actual_tokens=total,max_query_len=2,max_seq_len=max(lengths),query_start_loc=cu,query_start_loc_cpu=cu.cpu(),seq_lens=used,block_table_tensor=table,slot_mapping=None,causal=True,mm_req_doc_ranges=None)
                meta=builder.build(0,common)
                outputs={m:torch.full((total+13,16,128),17.,device='cuda',dtype=q.dtype) for m in ('original','split')}
                forward=backend.TritonAttentionImpl.forward
                functions={m:(lambda mode=m:forward(impl,layer,q,None,None,kv,meta,outputs[mode])) for m in outputs}
                graphs={}
                for mode in outputs:
                    select(mode)
                    for _ in range(3):functions[mode]()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):functions[mode]()
                    graphs[mode]=graph
                for phase in range(2):
                    if phase:
                        q.add_(.03125);k.add_(.015625);v.sub_(.015625);lengths=[x+1 for x in lengths];used.add_(1)
                        if ragged:qlens=list(reversed(qlens));cu.copy_(torch.tensor([0]+list(__import__('itertools').accumulate(qlens)),device='cuda',dtype=torch.int32))
                    expected=fixture.reference(q,k,v,table,qlens,lengths)
                    cpu=[];offset=0
                    for req,(ql,kl) in enumerate(zip(qlens,lengths)):
                        if req in (0,n//2,n-1):
                            ids=table[req,:(kl+15)//16].long();kc=k[ids].reshape(-1,2,128).cpu().float();vc=v[ids].reshape(-1,2,128).cpu().float()
                            for pos in range(ql):
                                limit=kl-ql+pos+1
                                for head in (0,7,8,15):
                                    query=q[offset+pos,head].cpu().float();exact=((kc[:limit,head//8]@query)*128**-.5).softmax(0)@vc[:limit,head//8]
                                    cpu.append(checked(expected[offset+pos,head].cpu(),exact))
                        offset+=ql
                    checks={}
                    for mode,out in outputs.items():
                        out.fill_(17.);graphs[mode].replay();torch.cuda.synchronize();checks[mode]=checked(out[:total],expected)
                        assert bool((out[total:]==17.).all())
                    records=[]
                    for trial in range(7):
                        for mode in (('original','split') if trial%2==0 else ('split','original')):
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                            for _ in range(10):graphs[mode].replay()
                            b.record();b.synchronize();row=dict(case=case,phase=phase,mode=mode,trial=trial,ms=a.elapsed_time(b)/10)
                            records.append(row);d['events'].append(row)
                    medians={mode:statistics.median(r['ms'] for r in records if r['mode']==mode) for mode in outputs}
                    d['cases'].append(dict(case=case,phase=phase,n=n,qlens=qlens[:],lengths=lengths[:],strided=strided,checks=checks,CPU_checks=cpu,output_tail_preserved=True,changed_Graph_inputs=bool(phase),medians_ms=medians,time_delta_percent=100*(medians['split']/medians['original']-1)));save()
        for path,sha in d['source_snapshots'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
        d.update(phase='complete',complete=True,scope='Correct-L0 native 2D versus native Q2 split-K dispatch; unchanged attention/reducer arithmetic, resident128-query scratch; no model or quality or service throughput.')
    except Exception as e:
        import traceback
        d.update(phase='failed',error=repr(e),traceback=traceback.format_exc())
    finally:save()
    return 0 if d['complete'] else 1


FIXTURE_SHA='47687cd2f7adbfc811d81d18f73d09970134b97c00df880911f7d1804fe865c5'
if __name__=='__main__':raise SystemExit(main())
