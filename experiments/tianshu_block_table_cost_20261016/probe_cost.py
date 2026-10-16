# SPDX-License-Identifier: Apache-2.0
"""Measure original BlockTable H2D/slot mapping; do not shrink model context."""
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import traceback


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True)
    args=parser.parse_args();root=args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir(parents=True)
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>60
    signal.alarm(min(1200,remaining))
    data=dict(phase='preparing',complete=False,runner_pid=os.getpid(),official_benchmark=False,adopted=False,
        source_snapshots={},cases=[],scope='Controlled original BlockTable instances. 320 columns is a transfer-cost comparison, not a proposed context/model change.')
    def save():
        p=root/'result.tmp';p.write_text(json.dumps(data,indent=2)+'\n');p.replace(root/'result.json')
        print(json.dumps(dict(phase=data['phase'],cases=len(data['cases']),error=data.get('error'))),flush=True)
    save()
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
        (root/'hardware_before.txt').write_text(hardware)
        import torch
        import numpy as np
        import flag_gems
        import vllm_fl
        import vllm.v1.worker.block_table as module
        from vllm.v1.utils import CpuGpuBuffer
        torch.set_num_threads(8)
        for path in (Path(__file__),Path(module.__file__),Path(inspect.getsourcefile(CpuGpuBuffer)),Path(flag_gems.__file__)):
            path=path.resolve();raw=path.read_bytes();data['source_snapshots'][str(path)]=hashlib.sha256(raw).hexdigest()
            dest=root/'source_snapshot'/path.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        assert data['source_snapshots'][str(Path(module.__file__).resolve())]=='b3a6649b968fba0b48c333d7ac96e65cc713a86133cc7eef0a8399f5339ec600'
        data.update(torch_version=torch.__version__,record_oplist=True,record_once=True,phase='numerical')
        save()
        with torch.inference_mode(),flag_gems.use_gems(record=True,once=True,path=str(root/'oplist.txt')):
            for columns in (8192,320):
                for n in (1,8,64):
                    table=module.BlockTable(block_size=16,max_num_reqs=256,max_num_blocks_per_req=columns,
                        max_num_batched_tokens=2048,pin_memory=True,device=torch.device('cuda:0'),
                        kernel_block_size=16,cp_kv_cache_interleave_size=1)
                    assert table.pcp_world_size==table.dcp_world_size==1
                    for i in range(n):table.add_row([1+i*320+j for j in range(320)],i)
                    positions_cpu=np.array([4096+i%16 for i in range(n)],dtype=np.int64)
                    positions=torch.from_numpy(positions_cpu).cuda()
                    query=torch.arange(n+1,device='cuda',dtype=torch.int32)
                    table.commit_block_table(n);table.compute_slot_mapping(n,query,positions);torch.cuda.synchronize()
                    expected=np.full(2048,-1,dtype=np.int64)
                    expected[:n]=np.array([table.block_table.np[i,positions_cpu[i]//16]*16+positions_cpu[i]%16 for i in range(n)])
                    assert np.array_equal(table.slot_mapping.gpu.cpu().numpy(),expected)
                    assert torch.equal(table.block_table.gpu[:n].cpu(),table.block_table.cpu[:n])
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):table.compute_slot_mapping(n,query,positions)
                    changed=[]
                    for step in range(2):
                        for i in range(n):
                            table.block_table.np[i,positions_cpu[i]//16]+=3
                        positions_cpu+=1;positions.copy_(torch.from_numpy(positions_cpu))
                        table.commit_block_table(n);table.slot_mapping.gpu.fill_(-333)
                        graph.replay();torch.cuda.synchronize()
                        expected[:n]=np.array([table.block_table.np[i,positions_cpu[i]//16]*16+positions_cpu[i]%16 for i in range(n)])
                        checks=dict(slots_and_padding=np.array_equal(table.slot_mapping.gpu.cpu().numpy(),expected),
                            full_active_rows_and_tail=torch.equal(table.block_table.gpu[:n].cpu(),table.block_table.cpu[:n]))
                        assert all(checks.values());changed.append(dict(step=step,outputs_cleared=True,checks=checks))
                    functions={
                        'commit':lambda:table.commit_block_table(n),
                        'slots':lambda:table.compute_slot_mapping(n,query,positions),
                        'commit_and_slots':lambda:(table.commit_block_table(n),table.compute_slot_mapping(n,query,positions)),
                    }
                    for function in functions.values():
                        for _ in range(5):function()
                    timings=[]
                    for repeat in range(7):
                        names=list(functions);offset=repeat%len(names)
                        for name in names[offset:]+names[:offset]:
                            torch.cuda.synchronize();started=time.perf_counter()
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                            for _ in range(20):functions[name]()
                            b.record();b.synchronize()
                            timings.append(dict(repeat=repeat,name=name,iterations=20,host_wall_ms=1000*(time.perf_counter()-started)/20,gpu_stream_ms=a.elapsed_time(b)/20))
                    medians={name:{key:statistics.median(t[key] for t in timings if t['name']==name)
                        for key in ('host_wall_ms','gpu_stream_ms')} for name in functions}
                    data['cases'].append(dict(requests=n,table_shape=list(table.block_table.gpu.shape),active_columns=320,
                        columns=columns,transfer_bytes=n*columns*4,pinned=table.block_table.cpu.is_pinned(),
                        positions_dtype=str(positions.dtype),slot_dtype=str(table.slot_mapping.gpu.dtype),
                        changed_graph_checks=changed,timings=timings,medians=medians));save()
        for name,sha in data['source_snapshots'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
        data.update(phase='complete',complete=True,original_methods_unmodified=True,full_model_not_run=True)
    except BaseException as error:data.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
    finally:signal.alarm(0);save()
    return 0 if data['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
