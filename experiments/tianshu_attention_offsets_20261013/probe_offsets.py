# SPDX-License-Identifier: Apache-2.0
"""Compare bounded native decode offsets without changing installed sources."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--real-layout', action='store_true')
    parser.add_argument('--bf16-descales', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    stop = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    remaining = int((stop-datetime.now(timezone.utc)).total_seconds())
    assert remaining >= 1800
    signal.alarm(min(3600, remaining))
    hardware = subprocess.check_output(['ixsmi'], text=True)
    assert 'No running processes found' in hardware
    (root/'hardware_before.txt').write_text(hardware)
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tianshu_native_attention'))
    from probe_decode_segments import native, reference, check
    from address32 import load_variant, assert_safe, Stage1Kernel
    candidate, provenance = load_variant(native, root/'generated')
    base_kernel = native.kernel_unified_attention
    native.kernel_unified_attention = Stage1Kernel(base_kernel)
    candidate.kernel_unified_attention = Stage1Kernel(candidate.kernel_unified_attention)
    data = dict(complete=False, numerical_checks_passed=False, official_benchmark=False,
                adopted=False, installed_sources_modified=False, provenance=provenance, cases=[],
                actual_interleaved_cache_layout=args.real_layout)
    data['bf16_unused_kv_descales_passed']=args.bf16_descales
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for path in (Path(__file__),Path(__file__).with_name('address32.py')):
        (snapshot/path.name).write_bytes(path.read_bytes())
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False

    def save():
        p=root/'result.tmp';p.write_text(json.dumps(data,indent=2)+'\n');p.replace(root/'result.json')

    save()
    cases=[(1,4096,False),(8,5120,True),(32,4096,False),(64,5120,True),
           (1,16384,False),(30,17408,True),(64,16384,False),(8,32768,True)]
    if args.real_layout:
        cases=[(1,5120,True),(4,5120,True),(8,5120,True),(16,5120,True),
               (24,5120,True),(30,17408,True),(32,17408,True),
               (48,5120,True),(64,5120,True),(8,32768,True)]
    try:
        with torch.inference_mode():
            for case,(batch,length,strided) in enumerate(cases):
                torch.manual_seed(2026101370+case)
                lengths=[length-(i*37%257) if strided else length for i in range(batch)]
                width=(length+15)//16;pages=batch*width
                table=torch.randperm(pages,device='cuda').int().reshape(batch,width)
                q=torch.randn(batch,20 if strided else 16,128,device='cuda',dtype=torch.bfloat16)[:,:16]
                k=torch.randn(pages*(2 if strided and not args.real_layout else 1),16,2,128,device='cuda',dtype=q.dtype)
                v=torch.randn_like(k)
                if args.real_layout:
                    cache=torch.stack((k,v),dim=1)
                    k,v=cache.unbind(1)
                elif strided:k,v=k[::2],v[::2]
                cu=torch.arange(batch+1,device='cuda',dtype=torch.int32)
                used=torch.tensor(lengths,device='cuda',dtype=torch.int32)
                expected=reference(q,k,v,table,lengths)
                segments=64 if batch==1 else 4 if batch>=16 else 16
                row=dict(batch=batch,lengths=lengths,strided=strided,segments=segments,variants=[],
                         q_stride=list(q.stride()),kv_stride=list(k.stride()))
                graphs=[]
                k_descale=torch.ones((),device='cuda',dtype=torch.float32).expand(batch,2) if args.bf16_descales else None
                v_descale=torch.ones((),device='cuda',dtype=torch.float32).expand(batch,2) if args.bf16_descales else None
                for label,module in [('original64',native),('address32',candidate)]:
                    output=torch.empty_like(q)
                    partial=torch.empty((batch,16,segments,128),device='cuda',dtype=torch.float32)
                    maximum=torch.empty((batch,16,segments),device='cuda',dtype=torch.float32)
                    expsum=torch.empty_like(maximum)
                    bounds=assert_safe([q,k,v,output,partial,maximum,expsum,table,cu,used])
                    def invoke():
                        module.unified_attention(q=q,k=k,v=v,out=output,cu_seqlens_q=cu,
                            max_seqlen_q=1,seqused_k=used,max_seqlen_k=max(lengths),
                            softmax_scale=128**-.5,causal=True,window_size=(-1,-1),
                            block_table=table,softcap=0.,q_descale=None,k_descale=k_descale,v_descale=v_descale,
                            seq_threshold_3D=64,num_par_softmax_segments=segments,
                            softmax_segm_output=partial,softmax_segm_max=maximum,softmax_segm_expsum=expsum)
                    for _ in range(3):invoke()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):invoke()
                    graph.replay();torch.cuda.synchronize()
                    entry=dict(label=label,bounded_element_offsets=bounds,vs_fp32=check(output,expected))
                    row['variants'].append(entry);graphs.append((entry,graph,output,partial,maximum,expsum))
                row['candidate_bitwise_equal']=torch.equal(graphs[0][2],graphs[1][2])
                timings={id(entry):[] for entry,*_ in graphs}
                for repeat in range(7):
                    order=graphs[repeat%2:]+graphs[:repeat%2]
                    for entry,graph,*_ in order:
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(30):graph.replay()
                        end.record();end.synchronize();timings[id(entry)].append(start.elapsed_time(end)/30)
                for entry,*_ in graphs:
                    entry['raw_ms']=timings[id(entry)];entry['median_ms']=statistics.median(entry['raw_ms'])
                data['cases'].append(row);save();print(json.dumps(row),flush=True)
                del graphs,q,k,v,expected;torch.cuda.empty_cache()
        data.update(complete=True,numerical_checks_passed=all(
            all(v['vs_fp32']['passed'] for v in c['variants']) for c in data['cases']),
            all_cases_bitwise_equal=all(c['candidate_bitwise_equal'] for c in data['cases']))
        save()
    finally:native.kernel_unified_attention=base_kernel


if __name__=='__main__':main()
