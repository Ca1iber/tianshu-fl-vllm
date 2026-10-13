# SPDX-License-Identifier: Apache-2.0
"""Bitwise/FP32 checks and rotated graph timings for bounded L0 Prefill offsets."""
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    stop=datetime.fromisoformat(json.loads(Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    remaining=int((stop-datetime.now(timezone.utc)).total_seconds());assert remaining>1800;signal.alarm(min(3600,remaining))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    (root/'hardware_before.txt').write_text(hardware)
    import torch
    import vllm.v1.attention.ops.triton_unified_attention as native
    from prefill32 import load_pair,PrefillKernel
    base=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('_tianshu_prefill_reference',base/'tianshu_native_attention/validate_hybrid.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    spec=importlib.util.spec_from_file_location('_tianshu_prefill_offset_bounds',base/'tianshu_attention_offsets_20261013/address32.py')
    bounds_helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(bounds_helper)
    modules,sources=load_pair(native,root/'generated')
    for module in modules:module.kernel_unified_attention=PrefillKernel(module.kernel_unified_attention)
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for path in [Path(__file__),Path(__file__).with_name('prefill32.py'),Path(native.__file__),base/'tianshu_attention_offsets_20261013/address32.py',base/'tianshu_native_attention/validate_hybrid.py']:
        (snapshot/path.name).write_bytes(path.read_bytes())
    data=dict(complete=False,numerical_checks_passed=False,all_cases_bitwise_equal=False,
              official_benchmark=False,adopted=False,installed_sources_modified=False,sources=sources,cases=[])
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    def save():
        p=root/'result.tmp';p.write_text(json.dumps(data,indent=2)+'\n');p.replace(root/'result.json')
    save()
    cases=[([2048],[2048]),([2048],[4096]),([2048],[16384]),([128],[4096]),
           ([565],[16384]),([1537],[16384]),([1537,511],[16384,4096]),([128]*16,[4096]*16)]
    with torch.inference_mode():
        for ci,(qlens,lengths) in enumerate(cases):
            torch.manual_seed(2026101390+ci);batch=len(qlens);total=sum(qlens);width=(max(lengths)+15)//16;pages=batch*width
            table=torch.randperm(pages,device='cuda').int().view(batch,width)
            q=torch.randn(total,20,128,device='cuda',dtype=torch.bfloat16)[:,:16]
            cache=torch.randn(pages,2,16,2,128,device='cuda',dtype=q.dtype);k,v=cache.unbind(1)
            cu=torch.tensor([0]+list(itertools.accumulate(qlens)),device='cuda',dtype=torch.int32)
            used=torch.tensor(lengths,device='cuda',dtype=torch.int32)
            expected=fixture.reference(q,k,v,table,qlens,lengths)
            descale=torch.ones((),device='cuda',dtype=torch.float32).expand(batch,2)
            row=dict(query_lengths=qlens,kv_lengths=lengths,q_stride=list(q.stride()),kv_stride=list(k.stride()),
                     variants=[],cpu_oracle_checks=[])
            # Independent CPU FP32 checks at sequence boundaries and within a tile.
            begin=0
            for seq,(ql,kl) in enumerate(zip(qlens,lengths)):
                ids=table[seq,:(kl+15)//16].long()
                kc=k[ids].cpu().reshape(-1,2,128)[:kl].float();vc=v[ids].cpu().reshape(-1,2,128)[:kl].float()
                selected=sorted({0,min(127,ql-1),ql-1})
                for head in [0,7,8,15]:
                    qq=q[begin+torch.tensor(selected,device=q.device),head].cpu().float()
                    scores=qq@kc[:,head//8].T*128**-.5
                    scores.masked_fill_(torch.arange(kl)[None,:]>kl-ql+torch.tensor(selected)[:,None],float('-inf'))
                    cpu=scores.softmax(-1)@vc[:,head//8]
                    check=fixture.error(expected[begin+torch.tensor(selected,device=q.device),head].cpu(),cpu)
                    assert check['passed'];row['cpu_oracle_checks'].append(check)
                begin+=ql
            graphs=[]
            for bits,module in zip([64,32],modules):
                output=torch.full((total+13,16,128),17.,device='cuda',dtype=q.dtype)
                bounds=bounds_helper.assert_safe([q,k,v,output,cu,used,table])
                def invoke():
                    module.unified_attention(q=q,k=k,v=v,out=output[:total],cu_seqlens_q=cu,
                        max_seqlen_q=max(qlens),seqused_k=used,max_seqlen_k=max(lengths),softmax_scale=128**-.5,
                        causal=True,window_size=(-1,-1),block_table=table,softcap=0.,q_descale=None,
                        k_descale=descale,v_descale=descale,seq_threshold_3D=None,num_par_softmax_segments=None,
                        softmax_segm_output=None,softmax_segm_max=None,softmax_segm_expsum=None)
                for _ in range(3):invoke()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):invoke()
                graph.replay();torch.cuda.synchronize()
                check=fixture.error(output[:total],expected);assert check['passed']
                tail=bool((output[total:]==17.).all());assert tail
                entry=dict(offset_bits=bits,bounds=bounds,vs_fp32=check,padded_tail_preserved=tail)
                row['variants'].append(entry);graphs.append((entry,graph,output))
            row['candidate_bitwise_equal']=torch.equal(graphs[0][2],graphs[1][2]);assert row['candidate_bitwise_equal']
            timings={64:[],32:[]}
            for repeat in range(7):
                order=graphs[repeat%2:]+graphs[:repeat%2]
                for entry,graph,output in order:
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):graph.replay()
                    end.record();end.synchronize();timings[entry['offset_bits']].append(start.elapsed_time(end)/20)
            for entry,graph,output in graphs:entry.update(raw_ms=timings[entry['offset_bits']],median_ms=statistics.median(timings[entry['offset_bits']]))
            data['cases'].append(row);save();print(json.dumps(row),flush=True)
            del graphs,q,k,v,cache,expected;torch.cuda.empty_cache()
    data.update(complete=True,numerical_checks_passed=True,all_cases_bitwise_equal=True);save()


if __name__=='__main__':main()
