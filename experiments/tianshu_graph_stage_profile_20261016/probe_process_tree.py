# SPDX-License-Identifier: Apache-2.0
"""Spawned GPU child with known Graph/eager counts and outside-Graph events."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def child(root):
    import torch
    import triton
    import triton.language as tl
    root.mkdir(parents=True, exist_ok=False)
    signal.alarm(180)

    @triton.jit
    def tianshu_stage_graph_add(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
        i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        tl.store(y+i,tl.load(x+i,i<N,other=0)+1,i<N)

    @triton.jit
    def tianshu_stage_graph_mul(y, z, N: tl.constexpr, BLOCK: tl.constexpr):
        i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        tl.store(z+i,tl.load(y+i,i<N,other=0)*2,i<N)

    @triton.jit
    def tianshu_stage_outside(z, out, N: tl.constexpr, BLOCK: tl.constexpr):
        i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        tl.store(out+i,tl.load(z+i,i<N,other=0)*2,i<N)

    with torch.inference_mode():
        n=65536;x=torch.arange(n,device='cuda',dtype=torch.float32)
        y=torch.empty_like(x);z=torch.empty_like(x);out=torch.empty_like(x)
        def pair():
            tianshu_stage_graph_add[(256,)](x,y,n,256,num_warps=4)
            tianshu_stage_graph_mul[(256,)](y,z,n,256,num_warps=4)
        pair();tianshu_stage_outside[(256,)](z,out,n,256,num_warps=4)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(25):pair()
        for _ in range(3):graph.replay()
        torch.cuda.synchronize()
        expected=(x.cpu()+1)*4
        tianshu_stage_outside[(256,)](z,out,n,256,num_warps=4)
        torch.cuda.synchronize();assert torch.equal(out.cpu(),expected)
        # All events are recorded OUTSIDE the graph, on its launch stream.
        events=[[torch.cuda.Event(enable_timing=True) for _ in range(4)] for _ in range(20)]
        for row in events:
            for event in row:event.record()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for g0,g1,e0,e1 in events:
            g0.record();graph.replay();g1.record()
            e0.record();tianshu_stage_outside[(256,)](z,out,n,256,num_warps=4);e1.record()
        torch.cuda.synchronize();torch.cuda.profiler.stop()
        assert torch.equal(out.cpu(),expected)
        graph_ms=[g0.elapsed_time(g1) for g0,g1,_,_ in events]
        eager_ms=[e0.elapsed_time(e1) for _,_,e0,e1 in events]
        assert all(t>0 for t in graph_ms+eager_ms)
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):graph.replay()
        end.record();end.synchronize();bulk=start.elapsed_time(end)
        # Fresh timestamp sanity: ten replays must differ substantially from one.
        import statistics
        ratio=bulk/statistics.median(graph_ms)
        assert 5 < ratio < 20,(bulk,graph_ms)
        nspid=next((line.split(':',1)[1].split() for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('NSpid:')),[])
        result=dict(complete=True,child_pid=os.getpid(),parent_pid=os.getppid(),namespace_pids=list(map(int,nspid)),
                    expected_graph_calls=20,expected_inside_calls=dict(tianshu_stage_graph_add=500,tianshu_stage_graph_mul=500),
                    expected_outside_calls=20,graph_output_exact=True,graph_events_ms=graph_ms,eager_events_ms=eager_ms,
                    bulk10_graph_ms=bulk,bulk10_to_single_ratio=ratio,events_recorded_inside_graph=False)
        (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--child',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    if args.child:return child(root)
    assert not root.exists();root.mkdir(parents=True)
    signal.alarm(240)
    # The parent deliberately has no CUDA context or torch import.
    assert 'torch' not in sys.modules
    with (root/'child.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--child','--output',str(root/'child')],
                                 stdout=log,stderr=subprocess.STDOUT)
        code=process.wait(timeout=210)
    assert code==0,code
    result=json.loads((root/'child/result.json').read_text());assert result['complete'] and result['child_pid']==process.pid
    (root/'result.json').write_text(json.dumps(dict(complete=True,parent_pid=os.getpid(),child_pid=process.pid,
        parent_has_no_torch_context=True,child=result),indent=2)+'\n')


if __name__=='__main__':main()
