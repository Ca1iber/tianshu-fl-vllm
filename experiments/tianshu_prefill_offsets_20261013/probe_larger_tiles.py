# SPDX-License-Identifier: Apache-2.0
"""New L0 Prefill tile combinations versus the validated M256/KV128/w16/s1."""
import argparse
from datetime import datetime,timezone
import importlib.util
import itertools
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys


class Launch:
    def __init__(self,kernel,plan):self.kernel=kernel;self.plan=plan
    def __getitem__(self,grid):
        def invoke(*args,**kwargs):
            assert not args and kwargs['IS_3D'] is False and kwargs['num_queries_per_kv']==8
            bm,bn,warps,stages=self.plan;bq=bm//8
            kwargs.update(BLOCK_M=bm,BLOCK_Q=bq,TILE_SIZE=bn,num_warps=warps,num_stages=stages)
            new_grid=(kwargs['query_ptr'].shape[0]//bq+kwargs['num_seqs'],grid[1])
            return self.kernel[new_grid](**kwargs)
        return invoke


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    stop=datetime.fromisoformat(json.loads(Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    remaining=int((stop-datetime.now(timezone.utc)).total_seconds());assert remaining>1800;signal.alarm(min(3600,remaining))
    hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(root/'hardware_before.txt').write_text(hw)
    import torch
    import vllm.v1.attention.ops.triton_unified_attention as native
    from prefill32 import load_pair
    base=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('_tianshu_larger_tile_reference',base/'tianshu_native_attention/validate_hybrid.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    modules,sources=load_pair(native,root/'generated');module=modules[0];kernel=module.kernel_unified_attention
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for p in [Path(__file__),Path(__file__).with_name('prefill32.py'),base/'tianshu_native_attention/validate_hybrid.py',Path(native.__file__)]:
        (snapshot/p.name).write_bytes(p.read_bytes())
    data=dict(complete=False,official_benchmark=False,adopted=False,installed_sources_modified=False,
              baseline=[256,128,16,1],source=sources[0],cases=[])
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    def save():
        p=root/'result.tmp';p.write_text(json.dumps(data,indent=2)+'\n');p.replace(root/'result.json')
    save()
    plans=[(256,128,16,1),(512,128,16,1),(512,64,16,1),(256,256,16,1),(256,128,8,1),(256,128,16,2)]
    cases=[([2048],[2048]),([2048],[4096]),([2048],[16384]),([1537,511],[16384,4096])]
    try:
        with torch.inference_mode():
            for ci,(qlens,lengths) in enumerate(cases):
                torch.manual_seed(20261013000+ci);batch=len(qlens);total=sum(qlens);width=(max(lengths)+15)//16;pages=batch*width
                table=torch.randperm(pages,device='cuda').int().view(batch,width)
                q=torch.randn(total,20,128,device='cuda',dtype=torch.bfloat16)[:,:16]
                cache=torch.randn(pages,2,16,2,128,device='cuda',dtype=q.dtype);k,v=cache.unbind(1)
                cu=torch.tensor([0]+list(itertools.accumulate(qlens)),device='cuda',dtype=torch.int32)
                used=torch.tensor(lengths,device='cuda',dtype=torch.int32);scale=torch.ones((),device='cuda').expand(batch,2)
                expected=fixture.reference(q,k,v,table,qlens,lengths)
                row=dict(query_lengths=qlens,kv_lengths=lengths,variants=[]);graphs=[]
                for plan in plans:
                    module.kernel_unified_attention=Launch(kernel,plan)
                    output=torch.full((total+13,16,128),17.,device='cuda',dtype=q.dtype)
                    label='m%d_n%d_w%d_s%d'%plan
                    entry=dict(label=label,configuration=list(plan))
                    def invoke():
                        module.unified_attention(q=q,k=k,v=v,out=output[:total],cu_seqlens_q=cu,
                            max_seqlen_q=max(qlens),seqused_k=used,max_seqlen_k=max(lengths),softmax_scale=128**-.5,
                            causal=True,window_size=(-1,-1),block_table=table,softcap=0.,q_descale=None,
                            k_descale=scale,v_descale=scale,seq_threshold_3D=None,num_par_softmax_segments=None,
                            softmax_segm_output=None,softmax_segm_max=None,softmax_segm_expsum=None)
                    try:
                        for _ in range(3):invoke()
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):invoke()
                        graph.replay();torch.cuda.synchronize()
                        check=fixture.error(output[:total],expected);entry['vs_fp32']=check
                        entry['padded_tail_preserved']=bool((output[total:]==17.).all())
                        entry['numerical_passed']=check['passed'] and entry['padded_tail_preserved']
                        if not entry['numerical_passed']:entry['error']='FP32 or output-tail check failed'
                        else:graphs.append((entry,graph,output))
                    except Exception as error:
                        entry.update(numerical_passed=False,error=repr(error))
                        if any(t in str(error).lower() for t in ['illegal memory','device-side assert','capture']):raise
                    row['variants'].append(entry)
                timing={entry['label']:[] for entry,*_ in graphs}
                for repeat in range(7):
                    order=graphs[repeat%len(graphs):]+graphs[:repeat%len(graphs)] if graphs else []
                    for entry,graph,output in order:
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):graph.replay()
                        end.record();end.synchronize();timing[entry['label']].append(start.elapsed_time(end)/20)
                for entry,*_ in graphs:entry.update(raw_ms=timing[entry['label']],median_ms=statistics.median(timing[entry['label']]))
                data['cases'].append(row);save();print(json.dumps(row),flush=True)
                del graphs,q,k,v,cache,expected;torch.cuda.empty_cache()
        passed=[p for p in plans if all(any(v['configuration']==list(p) and v.get('numerical_passed') for v in c['variants']) for c in data['cases'])]
        data.update(complete=True,all_scenario_valid_configurations=passed,baseline_passed=plans[0] in passed);save()
    finally:module.kernel_unified_attention=kernel


if __name__=='__main__':main()
