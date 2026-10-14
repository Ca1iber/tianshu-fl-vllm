# SPDX-License-Identifier: Apache-2.0
"""Actual builder/KV writer/forward for interleaved versus separated K/V planes, page16."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import inspect
import json
import logging
import os
from pathlib import Path
import runpy
import signal
import statistics
import subprocess
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base-profile',choices=['S4','S8'],required=True)
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    deadline=datetime.fromisoformat(json.loads(Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    remaining=int((deadline-datetime.now(timezone.utc)).total_seconds());assert remaining>0;signal.alarm(min(1800,remaining))
    before=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in before
    (root/'hardware_before.txt').write_text(before)
    base=Path(__file__).resolve().parents[1]
    os.environ.update(VLLM_PLUGINS='fl',VLLM_FL_ILUVATAR_HYBRID_ATTENTION='1',
        VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS='1',VLLM_FL_USE_FLAGGEMS_ATTN='0',
        FLAGGEMS_ILUVATAR_COMPACT_VARLEN='1',VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN='0',
        TIANSHU_KV_PAGE_PROBE='1',TIANSHU_KV_PAGE_BASE_PROFILE=args.base_profile,
        TIANSHU_KV_PAGE_RECORD_DIR=str(root/'geometry_records'),TIANSHU_KV16_SERVICE_MODE='candidate',
        TIANSHU_SEGMENT_POLICY_PROBE='1' if args.base_profile=='S8' else '0',
        TIANSHU_SEGMENT_POLICY_RECORD_DIR=str(root/'segment_records'))
    for prefix,folder in [('TIANSHU_FA2_TILE','fa2'),('TIANSHU_NATIVE_PREFILL','prefill'),
                          ('TIANSHU_DECODE_LAUNCH','decode'),('TIANSHU_DECODE_COHERENT','coherent'),
                          ('TIANSHU_TOPP_LAUNCH','topp')]:
        os.environ[prefix+'_PROBE']='1';os.environ[prefix+'_RECORD_DIR']=str(root/(folder+'_records'))
    runpy.run_path(str(base/'tianshu_kv_page_20261014/hooks/sitecustomize.py'))
    import torch
    import vllm_fl
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.kv_cache_interface import KVQuantMode
    import vllm.v1.attention.backends.triton_attn as backend
    import vllm.v1.attention.ops.triton_unified_attention as native
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    traces=[];real_unified=native.unified_attention

    def traced_unified(*a,**kw):
        assert not a
        kernel=native.kernel_unified_attention
        trace=dict(page_size=kw['k'].shape[1],query_tokens=kw['q'].shape[0],max_query=kw['max_seqlen_q'],
                   sequences=kw['cu_seqlens_q'].numel()-1,segments=kw['num_par_softmax_segments'],
                   kernel_class=type(kernel).__qualname__)
        trace['kernel_router_source']=kernel.__class__.__getitem__.__code__.co_filename
        answer=real_unified(**kw);traces.append(trace);return answer
    native.unified_attention=traced_unified;backend.unified_attention=traced_unified
    config=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=2048,num_hidden_layers=42,
        num_attention_heads=16,num_key_value_heads=2),dtype=torch.bfloat16,quantization=None,is_multimodal_model=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,decode_context_parallel_size=1),
        lora_config=None,speculative_config=None)
    # Same initialization order as WorkerFL: adaptive Decode before hybrid routing.
    from vllm_fl.worker import iluvatar_adaptive_decode as adaptive
    assert adaptive.configure(config,logging.getLogger('page32_probe'))
    from vllm_fl.worker import iluvatar_hybrid_attention as hybrid
    assert hybrid.configure(config,logging.getLogger('page32_probe'))
    sources=[Path(__file__).resolve(),base/'tianshu_kv_page_20261014/geometry_guard.py',
        base/'tianshu_kv_page_20261014/hooks/sitecustomize.py',Path(hybrid.__file__),Path(native.__file__),Path(backend.__file__),Path(adaptive.__file__),
        Path(hybrid.__file__).with_name("worker.py"),Path(inspect.getsourcefile(backend.triton_reshape_and_cache_flash)),
        base/'tianshu_native_prefill_20261012/hooks/sitecustomize.py',base/'tianshu_native_prefill_20261012/zero_initialization.py',
        base/'tianshu_decode_kv16_w2_20261014/hooks/sitecustomize.py',base/'tianshu_decode_kv16_w2_20261014/hooks/decode_sitecustomize.py']
    if args.base_profile=='S8':sources.append(base/'tianshu_kv16_segments_20261014/hooks/sitecustomize.py')
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    snapshots=root/'source_snapshot'
    for i,p in enumerate(sources):
        dest=snapshots/f'{i:02d}_{p.name}';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    data=dict(complete=False,official_benchmark=False,adoption_allowed=False,base_profile=args.base_profile,
              frozen_sources=frozen,cases=[],started_utc=datetime.now(timezone.utc).isoformat(),
              scope='Same page16 actual attention: interleaved vs separate K/V planes; not full model or quality')
    def save():
        p=root/'result.tmp';p.write_text(json.dumps(data,indent=2)+'\n');p.replace(root/'result.json')
    save()
    def error(a,b):
        a,b=a.float(),b.float();d=(a-b).abs();bad=d>.02+.01*b.abs()
        return dict(passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not bad.any()),
                    max_abs=d.max().item(),failed_elements=bad.sum().item(),atol=.02,rtol=.01)
    def reference(q,k,v,qlens,lengths):
        outputs=[];begin=0
        for seq,(ql,kl) in enumerate(zip(qlens,lengths)):
            if not ql:continue
            keys=k[seq,:kl].float().repeat_interleave(8,1)
            values=v[seq,:kl].float().repeat_interleave(8,1)
            query=q[begin:begin+ql].float().transpose(0,1);parts=[]
            for offset in range(0,ql,64):
                chunk=query[:,offset:offset+64];score=(chunk@keys.permute(1,2,0))*128**-.5
                pos=torch.arange(offset,offset+chunk.shape[1],device=q.device)
                mask=torch.arange(kl,device=q.device)[None,:]<=kl-ql+pos[:,None]
                score.masked_fill_(~mask[None],-float('inf'));parts.append(score.softmax(-1)@values.transpose(0,1))
            outputs.append(torch.cat(parts,1).transpose(0,1));begin+=ql
        return torch.cat(outputs)
    def cpu_reference(q,k,v,qlens,lengths,expected):
        checks=[];begin=0
        for seq,(ql,kl) in enumerate(zip(qlens,lengths)):
            if not ql:continue
            if seq in (0,len(qlens)//2,len(qlens)-1):
                kc=k[seq,:kl].cpu().float();vc=v[seq,:kl].cpu().float()
                for pos in sorted({0,ql-1}):
                    for head in (0,7,8,15):
                        qc=q[begin+pos,head].cpu().float();scores=(kc[:,head//8]@qc)*128**-.5
                        scores[kl-ql+pos+1:]=-float('inf');answer=scores.softmax(0)@vc[:,head//8]
                        checks.append(error(expected[begin+pos,head].cpu(),answer))
            begin+=ql
        assert checks and all(x['passed'] for x in checks)
        return checks
    def make_impl():
        impl=object.__new__(backend.TritonAttentionImpl)
        impl.attn_type=AttentionType.DECODER;impl._is_per_token_head_quant=False;impl.kv_cache_dtype='auto'
        impl.fp8_dtype=torch.float8_e4m3fn;impl._kv_quant_mode=KVQuantMode.NONE
        impl.scale=128**-.5;impl.alibi_slopes=None;impl.use_alibi_sqrt=False;impl.sliding_window=(-1,-1)
        impl.logits_soft_cap=0.;impl.sinks=None;impl.chunk_lookback=-1;impl.use_td=False
        return impl
    def updates(logical_k,logical_v,qlens,lengths,table,page):
        rows=[];keys=[];values=[]
        for seq,(ql,kl) in enumerate(zip(qlens,lengths)):
            positions=torch.arange(kl-ql,kl,device='cuda',dtype=torch.int64)
            rows.append(table[seq,positions//page].long()*page+positions%page)
            keys.append(logical_k[seq,kl-ql:kl]);values.append(logical_v[seq,kl-ql:kl])
        slots=torch.cat(rows);ki=torch.cat(keys);vi=torch.cat(values)
        return (torch.cat((slots,torch.full((3,),-1,device='cuda',dtype=torch.int64))),
                torch.cat((ki,torch.full((3,2,128),19.,device='cuda',dtype=ki.dtype))),
                torch.cat((vi,torch.full((3,2,128),23.,device='cuda',dtype=vi.dtype))))
    def poison(k,v,slots,page):
        valid=slots[slots>=0];pages=valid//page;positions=valid%page
        k[pages,positions]=-13.;v[pages,positions]=-13.
    cases=[('prefill32',[32],[32],False),('prefill33',[33],[65],True),
        ('prefill2048',[2048],[2048],False),('chunk2048',[2048],[4096],True),
        ('ragged',[31,32,33],[33,65,4096],True),
        ('mixed64',[1]*63+[1985],[4096]*64,True),
        ('mixed_ragged',[1,1,1,7,129,511],[4096,4200,5120,2048,4096,5120],True),
        ('decode64',[1]*64,[4096]*64,False),
        *[(f'decode{n}',[1]*n,[4096]*n,False) for n in (8,16,24,32)],('unreordered',[129,1],[4096,4096],False)]
    try:
        with torch.inference_mode():
            for case_id,(name,qlens,lengths,strided) in enumerate(cases):
                torch.manual_seed(2026101600+case_id);n=len(qlens);total=sum(qlens)
                logical_width=(max(lengths)+31)//32*32
                logical_k=torch.randn(n,logical_width,2,128,device='cuda',dtype=torch.bfloat16)
                logical_v=torch.randn_like(logical_k)
                q=torch.randn(total,20 if strided else 16,128,device='cuda',dtype=torch.bfloat16)[:,:16]
                cu_cpu=torch.tensor([0]+list(itertools.accumulate(qlens)),dtype=torch.int32);cu=cu_cpu.cuda()
                layer=SimpleNamespace(_q_scale=torch.ones(1,device='cuda'),_k_scale=torch.ones(1,device='cuda'),_v_scale=torch.ones(1,device='cuda'))
                fixtures=[];trace_begin=len(traces)
                for layout in ('interleaved','planes'):
                    page=16
                    width=(max(lengths)+page-1)//page;pages=n*width
                    table=torch.randperm(pages,device='cuda').int().reshape(n,width)
                    physical_pages=(pages+2)*(2 if strided else 1)
                    if layout=='interleaved':
                        storage=torch.full((physical_pages,2,page,2,128),17.,device='cuda',dtype=q.dtype)
                        cache=storage
                    else:
                        storage=torch.full((2,physical_pages,page,2,128),17.,device='cuda',dtype=q.dtype)
                        cache=storage.permute(1,0,2,3,4)
                    kv=cache[::2] if strided else cache;k,v=kv.unbind(1)
                    k[table.flatten().long()]=logical_k[:,:width*page].reshape(pages,page,2,128)
                    v[table.flatten().long()]=logical_v[:,:width*page].reshape(pages,page,2,128)
                    expected_storage=storage.clone()
                    slots,ki,vi=updates(logical_k,logical_v,qlens,lengths,table,page)
                    poison(k,v,slots,page)
                    impl=make_impl();impl.do_kv_cache_update(layer,ki,vi,kv,slots);torch.cuda.synchronize()
                    assert torch.equal(storage,expected_storage), 'KV writer changed wrong cells or failed padded slots'
                    used=torch.tensor(lengths,device='cuda',dtype=torch.int32)
                    common=SimpleNamespace(num_reqs=n,num_actual_tokens=total,max_query_len=max(qlens),max_seq_len=max(lengths),
                        query_start_loc=cu,query_start_loc_cpu=cu_cpu,seq_lens=used,block_table_tensor=table,slot_mapping=slots,
                        causal=True,mm_req_doc_ranges=None)
                    builder=object.__new__(backend.TritonAttentionMetadataBuilder)
                    builder.seq_threshold_3D=64;builder.num_par_softmax_segments=64;builder.device=q.device
                    builder.softmax_segm_output=torch.full((64,16,64,128),17.,device='cuda',dtype=torch.float32)
                    builder.softmax_segm_max=torch.full((64,16,64),17.,device='cuda',dtype=torch.float32)
                    builder.softmax_segm_expsum=torch.full_like(builder.softmax_segm_max,17.)
                    metadata=builder.build(0,common)
                    out_storage=torch.full((total+13,20 if strided else 16,128),17.,device='cuda',dtype=q.dtype);out=out_storage[:,:16]
                    fixtures.append(dict(layout=layout,page=page,cache_strides=list(kv.stride()),table=table,pages=pages,storage=storage,kv=kv,k=k,v=v,
                        expected_storage=expected_storage,slots=slots,ki=ki,vi=vi,impl=impl,used=used,
                        builder=builder,metadata=metadata,out_storage=out_storage,out=out))
                # Verify the existing geometry guard accepts actual page16 GPU tensors.
                assert hybrid.supported(dict(q=q,k=fixtures[1]['k'],v=fixtures[1]['v'],out=fixtures[1]['out'],
                    causal=True,window_size=(-1,-1)))
                def invoke(f):
                    f['impl'].do_kv_cache_update(layer,f['ki'],f['vi'],f['kv'],f['slots'])
                    f['impl'].forward(layer,q,None,None,f['kv'],f['metadata'],f['out'])
                phase_results=[]
                for phase in range(2):
                    if phase:
                        logical_k.mul_(.5);logical_v.neg_();q.neg_()
                        lengths=[kl-min(7,max(0,kl-ql)) for ql,kl in zip(qlens,lengths)]
                        for f in fixtures:
                            f['k'][:f['pages']].mul_(.5);f['v'][:f['pages']].neg_()
                            f['expected_storage'].copy_(f['storage'])
                            slots,ki,vi=updates(logical_k,logical_v,qlens,lengths,f['table'],f['page'])
                            f['slots'].copy_(slots);f['ki'].copy_(ki);f['vi'].copy_(vi)
                            f['used'].copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32))
                            poison(f['k'],f['v'],f['slots'],f['page'])
                    for f in fixtures:
                        if not phase:
                            for _ in range(3):invoke(f)
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):invoke(f)
                            f['graph']=graph
                        f['graph'].replay();torch.cuda.synchronize()
                        assert torch.equal(f['storage'],f['expected_storage']), 'Graph KV writer failed changed data/slots'
                        assert bool((f['out_storage'][total:]==17.).all())
                        if strided:assert bool((f['out_storage'][:total,16:]==17.).all())
                        route=getattr(f['metadata'],'_fl_hybrid_route',None)
                        decodes=n if max(qlens)==1 else route.num_decodes if route is not None else 0
                        segs=adaptive.choose_segments(decodes) if decodes else 0
                        for field,scale in (('softmax_segm_output',128),('softmax_segm_max',1),('softmax_segm_expsum',1)):
                            buffer=getattr(f['builder'],field);count=decodes*16*segs*scale
                            assert bool((buffer.flatten()[count:]==17.).all()), 'Unused segment scratch overwritten'
                    expected=reference(q,logical_k,logical_v,qlens,lengths)
                    cpu=cpu_reference(q,logical_k,logical_v,qlens,lengths,expected)
                    checks=[error(f['out'][:total],expected) for f in fixtures]
                    phase_results.append(dict(phase=phase,kv_lengths=list(lengths),reference_checks=checks,cpu_checks=cpu,
                        interleaved_vs_planes=error(fixtures[1]['out'][:total],fixtures[0]['out'][:total]),
                        layouts_bitwise_equal=torch.equal(fixtures[0]['out'],fixtures[1]['out']),
                        KV_writer_all_storage_bitwise_exact=True,output_padding_preserved=True,segment_scratch_tail_preserved=True))
                    assert all(x['passed'] for x in checks) and phase_results[-1]['interleaved_vs_planes']['passed']
                measurements=[[],[]]
                for trial in range(7):
                    for index in ((0,1) if trial%2==0 else (1,0)):
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):fixtures[index]['graph'].replay()
                        end.record();end.synchronize();measurements[index].append(start.elapsed_time(end)/20)
                medians=list(map(statistics.median,measurements))
                current_traces=traces[trace_begin:];assert {x['page_size'] for x in current_traces}=={16}
                if not name.startswith('decode') and name!='unreordered':
                    assert all(any(x['page_size']==page and x['max_query']>1 and x['kernel_router_source'].endswith('tianshu_native_prefill_20261012/hooks/sitecustomize.py') for x in current_traces) for page in (16,))
                result=dict(name=name,query_lengths=qlens,strided=strided,phases=phase_results,
                    raw_ms=measurements,median_ms=medians,time_change_percent=100*(medians[1]/medians[0]-1),
                    actual_native_calls=current_traces,cache_layouts=[dict(layout=f['layout'],strides=f['cache_strides']) for f in fixtures],query_and_KV_and_slots_and_lengths_changed_in_place=True)
                data['cases'].append(result);save();print(json.dumps({k:result[k] for k in ('name','median_ms','time_change_percent')}),flush=True)
                del fixtures;torch.cuda.empty_cache()
        assert len(data['cases'])==len(cases)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in frozen.items())
        data.update(complete=True,numerical_checks_passed=True,actual_native_prefill_composition_verified=True,
                    actual_builder_and_KV_writer_checked=True,changed_Graph_replays_verified=True,
                    finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        data.update(error=repr(exc),finished_utc=datetime.now(timezone.utc).isoformat());raise
    finally:
        save()


if __name__=='__main__':main()
