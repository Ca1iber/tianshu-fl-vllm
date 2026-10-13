# SPDX-License-Identifier: Apache-2.0
"""Separate KV address width from launch tuning on actual QKV/cache strides."""
import argparse
from datetime import datetime,timezone
import importlib
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
    remaining=int((stop-datetime.now(timezone.utc)).total_seconds());assert remaining>900;signal.alarm(min(1800,remaining))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    (root/'hardware_before.txt').write_text(hardware)
    import torch
    from kv32 import load_variant,LaunchOverride,assert_safe
    native=importlib.import_module('vllm.v1.attention.ops.triton_reshape_and_cache_flash')
    candidate,source=load_variant(native,root/'generated')
    original=native.reshape_and_cache_kernel_flash;generated=candidate.reshape_and_cache_kernel_flash
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for path in [Path(__file__),Path(__file__).with_name('kv32.py'),Path(native.__file__)]:
        (snapshot/path.name).write_bytes(path.read_bytes())
    data=dict(complete=False,numerical_checks_passed=False,official_benchmark=False,adopted=False,
              installed_sources_modified=False,source=source,cases=[])
    torch.set_num_threads(8)

    def save():
        t=root/'result.tmp';t.write_text(json.dumps(data,indent=2)+'\n');t.replace(root/'result.json')

    save()
    variants=[('original64_w16s10',native,original,16,10),('address32_w16s10',candidate,generated,16,10),
              ('original64_w4s1',native,original,4,1),('address32_w4s1',candidate,generated,4,1),
              ('original64_w1s1',native,original,1,1),('address32_w1s1',candidate,generated,1,1)]
    try:
        with torch.inference_mode():
            for ci,(tokens,pages,padded) in enumerate([(1,128,False),(8,128,True),(32,128,True),(64,128,True),
                     (256,128,True),(2048,256,True),(32,32682,True),(2048,32682,True)]):
                torch.manual_seed(2026101380+ci)
                qkv=torch.randn(tokens,2560,device='cuda',dtype=torch.bfloat16)
                key=qkv[:,2048:2304].view(tokens,2,128);value=qkv[:,2304:2560].view(tokens,2,128)
                slots=torch.randperm(pages*16,device='cuda')[:tokens].to(torch.int64)
                if padded:slots[::7]=-1
                valid=slots>=0
                assert slots.dtype==torch.int64 and bool(((slots>=-1)&(slots<pages*16)).all())
                initial=torch.full((pages,2,16,2,128),-7,device='cuda',dtype=key.dtype)
                expected=initial.clone()
                block=slots[valid]//16;offset=slots[valid]%16
                expected[block,0,offset]=key[valid];expected[block,1,offset]=value[valid]
                scale=torch.ones((),device='cuda',dtype=torch.float32)
                row=dict(tokens=tokens,pages=pages,padded=padded,key_stride=list(key.stride()),
                         public_slot_dtype=str(slots.dtype),variants=[])
                # Reuse one cache allocation; replay changes only valid destinations.
                cache=initial.clone();kc,vc=cache.unbind(1);bounds=assert_safe(key,value,kc,vc,slots)
                graphs=[]
                for label,module,kernel,warps,stages in variants:
                    module.reshape_and_cache_kernel_flash=LaunchOverride(kernel,warps,stages)
                    cache.copy_(initial)
                    def invoke():module.triton_reshape_and_cache_flash(key,value,kc,vc,slots,'auto',scale,scale)
                    for _ in range(3):invoke()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):invoke()
                    graph.replay();torch.cuda.synchronize()
                    exact=torch.equal(cache,expected);assert exact,label
                    entry=dict(label=label,bitwise_scatter_and_untouched_cache_passed=exact,bounds=bounds,
                               warps=warps,stages=stages)
                    row['variants'].append(entry);graphs.append((entry,graph))
                times={e['label']:[] for e,g in graphs}
                for repeat in range(7):
                    order=graphs[repeat%len(graphs):]+graphs[:repeat%len(graphs)]
                    for entry,graph in order:
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(100):graph.replay()
                        end.record();end.synchronize();times[entry['label']].append(start.elapsed_time(end)/100)
                for e,g in graphs:e.update(raw_ms=times[e['label']],median_ms=statistics.median(times[e['label']]))
                data['cases'].append(row);save();print(json.dumps(row),flush=True)
                del graphs,cache,kc,vc,initial,expected,qkv,key,value,slots;torch.cuda.empty_cache()
        data.update(complete=True,numerical_checks_passed=all(v['bitwise_scatter_and_untouched_cache_passed'] for c in data['cases'] for v in c['variants']))
        save()
    finally:
        native.reshape_and_cache_kernel_flash=original;candidate.reshape_and_cache_kernel_flash=generated


if __name__=='__main__':main()
