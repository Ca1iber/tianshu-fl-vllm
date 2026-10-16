# SPDX-License-Identifier: Apache-2.0
"""Actual ModelRunner method comparison plus changed-input graph and event timing."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import textwrap
import time
from types import MethodType, SimpleNamespace


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True)
    args=p.parse_args();root=args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir(parents=True)
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds())
    assert remaining>300;signal.alarm(min(1200,remaining))
    data=dict(phase='preparing',complete=False,runner_pid=os.getpid(),source_snapshots={},cases=[],graphs=[],timings=[],cold=[],official_benchmark=False,adopted=False)
    def save():
        tmp=root/'result.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(root/'result.json')
        print(json.dumps(dict(phase=data['phase'],cases=len(data['cases']),error=data.get('error'))),flush=True)
    def snapshot(path):
        path=Path(path).resolve();raw=path.read_bytes();data['source_snapshots'][str(path)]=hashlib.sha256(raw).hexdigest()
        dest=root/'source_snapshot'/path.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    save()
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
        import torch
        import numpy as np
        import vllm_fl
        import flag_gems
        from vllm.v1.utils import CpuGpuBuffer
        import positions
        from render import render
        torch.set_num_threads(8)
        source=Path('/usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py');raw=source.read_bytes()
        old_probe=Path(__file__).resolve().parents[1]/'tianshu_decode_batch_meta_20261016/probe_metadata.py'
        assert hashlib.sha256(old_probe.read_bytes()).hexdigest()==OLD_PROBE_SHA
        for path in (Path(__file__),Path(positions.__file__),Path(__file__).with_name('render.py'),source,old_probe,Path(inspect.getsourcefile(CpuGpuBuffer))):snapshot(path)
        syntax=ast.parse(raw.decode());cls=next(c for c in syntax.body if isinstance(c,ast.ClassDef) and c.name=='ModelRunnerFL')
        names=('_prepare_inputs','_prepare_input_ids','_get_cumsum_and_arange','_compute_prev_positions')
        namespace=dict(torch=torch,np=np,_tianshu_positions=positions,SimpleNamespace=SimpleNamespace,CpuGpuBuffer=CpuGpuBuffer,MethodType=MethodType,names=names,namespace=None)
        namespace['namespace']=namespace
        for name in names:
            node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
            text=textwrap.dedent('\n'.join(raw.decode().splitlines()[node.lineno-1:node.end_lineno]))
            exec(compile('from __future__ import annotations\n'+text,str(source),'exec'),namespace)
        old_tree=ast.parse(old_probe.read_text());old_main=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        # Reuse only the previously frozen controlled owner/slot observer, not its candidate or oracle.
        for name in ('Slots','owner'):
            node=next(n for n in old_main.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
            exec(compile(textwrap.dedent('\n'.join(old_probe.read_text().splitlines()[node.lineno-1:node.end_lineno])),str(old_probe),'exec'),namespace)
        owner=namespace['owner'];original=namespace['_prepare_inputs'];methods={}
        data['derivations']={}
        for mode in ('control','int32','fused'):
            rendered,derivation=render(raw,mode);data['derivations'][mode]=derivation
            for name,text in rendered.items():(root/(mode+'_'+name+'.py')).write_text(text)
            space=dict(namespace);exec(compile('from __future__ import annotations\n'+rendered['_prepare_inputs'],str(root/(mode+'__prepare_inputs.py')),'exec'),space)
            methods[mode]=space['_prepare_inputs']
        def make(n,seed,mode,counts):
            o=owner(n,seed)
            # Leave room for mixed prefill query offsets at the high-context edge.
            o.input_batch.num_computed_tokens_cpu[:n]%=120000
            for i,r in enumerate(o.input_batch.req_ids):o.requests[r].num_tokens=int(o.input_batch.num_computed_tokens_cpu[i])+1
            if mode!='control':
                o.req_indices=CpuGpuBuffer(2048,dtype=torch.int32,device=o.device);o.req_indices.cpu.fill_(-333);o.req_indices.gpu.fill_(-333)
            return o
        def compare(a,b,ra,rb,n):
            torch.cuda.synchronize();checks={}
            for name in ('positions','seq_lens','num_computed_tokens'):
                checks[name]=torch.equal(getattr(a,name).cpu(),getattr(b,name).cpu())
            for name in ('input_ids','query_start_loc','req_indices','query_pos','num_scheduled_tokens','discard_request_mask','num_accepted_tokens','prev_positions'):
                checks[name+'_gpu']=torch.equal(getattr(a,name).gpu.cpu(),getattr(b,name).gpu.cpu())
                checks[name+'_cpu']=torch.equal(getattr(a,name).cpu,getattr(b,name).cpu)
            checks['logits']=torch.equal(ra[0].cpu(),rb[0].cpu()) and ra[0].dtype==rb[0].dtype
            checks['no_spec']=ra[1] is rb[1] is None
            aa,bb=a.input_batch.block_table.captured,b.input_batch.block_table.captured
            checks['real_slot_arguments']=aa[0]==bb[0] and torch.equal(aa[1].cpu(),bb[1].cpu()) and torch.equal(aa[2].cpu(),bb[2].cpu())
            checks['int64_positions']=a.positions.dtype==b.positions.dtype==torch.int64
            assert all(checks.values()),checks
            # Independent integer CPU oracle, not either implementation's output.
            expected=[]
            for i,count in enumerate(n):expected.extend(range(int(b.input_batch.num_computed_tokens_cpu[i]),int(b.input_batch.num_computed_tokens_cpu[i])+int(count)))
            assert b.positions[:len(expected)].cpu().tolist()==expected
            return checks
        data['phase']='numerical';save()
        with torch.inference_mode(),flag_gems.use_gems():
            for case,(n,m) in enumerate(((1,1),(8,8),(29,29),(30,30),(32,32),(64,64),(64,290),(64,565),(64,1323),(64,1345),(64,1537),(64,2048),(128,2048),(256,2048))):
                counts=np.ones(n,dtype=np.int32);counts[0]+=m-n
                owners={mode:make(n,2026101600+case,mode,counts) for mode in methods}
                schedule=SimpleNamespace(total_num_scheduled_tokens=m,scheduled_spec_decode_tokens={})
                for iteration in range(2):
                    returns={}
                    for mode,o in owners.items():
                        torch.cuda.synchronize();start=time.perf_counter();returns[mode]=methods[mode](o,schedule,counts);torch.cuda.synchronize()
                        if iteration==0:data['cold'].append(dict(n=n,m=m,mode=mode,host_ms=1000*(time.perf_counter()-start)))
                    for mode in ('int32','fused'):
                        checks=compare(owners['control'],owners[mode],returns['control'],returns[mode],counts)
                        data['cases'].append(dict(n=n,m=m,phase=iteration,mode=mode,checks=checks,independent_CPU_positions=True))
                    for o in owners.values():
                        o.input_batch.num_computed_tokens_cpu[:n]+=1
                        o.input_batch.prev_sampled_token_ids.add_(17).remainder_(130560)
                        for r in o.requests.values():r.num_tokens+=1
                if case in (5,6,10,13):
                    o=owners['fused'];positions.launch(o,m);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):positions.launch(o,m)
                    for turn in range(2):
                        computed=torch.arange(256,device='cuda',dtype=torch.int32)*97+turn*101
                        o.num_computed_tokens.copy_(computed);o.query_pos.gpu[:m].add_(31)
                        o.positions.fill_(-333);graph.replay();torch.cuda.synchronize()
                        idx=o.req_indices.gpu[:m].cpu().tolist();q=o.query_pos.gpu[:m].cpu().tolist();c=computed.cpu().tolist()
                        assert o.positions[:m].cpu().tolist()==[c[r]+v for r,v in zip(idx,q)]
                        assert o.positions[m:].cpu().tolist()==[-333]*(2048-m)
                        data['graphs'].append(dict(n=n,m=m,turn=turn,changed_computed_and_query=True,cleared_outputs=True,CPU_equal=True))
                if case in (5,6,10):
                    # Actual complete prepare method; the unchanged slot mapping is an observer stub.
                    for o in owners.values():o.input_batch.block_table.record=False
                    for mode,o in owners.items():
                        for _ in range(3):methods[mode](o,schedule,counts)
                    for repeat in range(7):
                        order=list(methods) if repeat%2==0 else list(reversed(methods))
                        for mode in order:
                            o=owners[mode];torch.cuda.synchronize();start=time.perf_counter();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                            for _ in range(20):methods[mode](o,schedule,counts)
                            b.record();b.synchronize()
                            data['timings'].append(dict(n=n,m=m,mode=mode,repeat=repeat,iterations=20,host_ms=1000*(time.perf_counter()-start)/20,stream_ms=a.elapsed_time(b)/20))
                save()
            data['medians']={str(m):{mode:{key:statistics.median(r[key] for r in data['timings'] if r['m']==m and r['mode']==mode) for key in ('host_ms','stream_ms')} for mode in methods} for m in (64,290,1537)}
        for path,sha in data['source_snapshots'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
        data.update(phase='complete',complete=True,scope='Real prepare methods/controlled buffers/observed slot arguments; no model throughput, actual KV writer or quality claim.')
    except Exception as e:
        import traceback
        data.update(phase='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True));save()
    return 0 if data['complete'] else 1


OLD_PROBE_SHA='a34c5ef4f5a8ef2753126648827cf1fdd70f760569c068dbae201b0f83e542da'
if __name__=='__main__':raise SystemExit(main())
