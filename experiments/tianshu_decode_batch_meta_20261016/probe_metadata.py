# SPDX-License-Identifier: Apache-2.0
"""Execute original and rendered prepare methods on controlled real GPU buffers."""
import argparse
import ast
from datetime import datetime,timezone
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
from types import MethodType,SimpleNamespace


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    root.mkdir(parents=True);signal.alarm(1200)
    data=dict(phase='preparing',complete=False,official_benchmark=False,adopted=False,
        runner_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        cases=[],guards=[],graph_checks=[],timings=[],source_snapshots={})
    def save():
        (root/'result.json').write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps(dict(phase=data['phase'],cases=len(data['cases']),error=data.get('error'))),flush=True)
    def snapshot(p):
        p=Path(p).resolve();raw=p.read_bytes();data['source_snapshots'][str(p)]=hashlib.sha256(raw).hexdigest()
        target=root/'source_snapshot'/p.relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
        (root/'hardware_before.txt').write_text(hardware)
        import torch
        import numpy as np
        import vllm_fl
        import flag_gems
        from vllm.v1.utils import CpuGpuBuffer
        import metadata as candidate
        from render import render_source
        torch.set_num_threads(8)
        source=Path('/usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py')
        raw=source.read_bytes();generated,derivation=render_source(raw)
        (root/'generated_prepare.py').write_text(generated);data['derivation']=derivation
        for path in (Path(__file__),Path(candidate.__file__),Path(__file__).with_name('render.py'),source,
                     Path(inspect.getsourcefile(CpuGpuBuffer))):snapshot(path)
        syntax=ast.parse(raw.decode());cls=next(c for c in syntax.body if isinstance(c,ast.ClassDef) and c.name=='ModelRunnerFL')
        names=('_prepare_inputs','_prepare_input_ids','_get_cumsum_and_arange','_compute_prev_positions')
        namespace=dict(torch=torch,np=np,_tianshu_batch_meta=candidate)
        for name in names:
            method=next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name==name)
            text=textwrap.dedent('\n'.join(raw.decode().splitlines()[method.lineno-1:method.end_lineno]))
            exec(compile('from __future__ import annotations\n'+text,str(source),'exec'),namespace)
        original=namespace['_prepare_inputs']
        exec(compile('from __future__ import annotations\n'+generated,str(root/'generated_prepare.py'),'exec'),namespace)
        changed=namespace['_prepare_inputs']

        class Slots:
            def __init__(self):self.record=True;self.captured=None;self.commits=[]
            def commit_block_table(self,n):
                if self.record:self.commits.append(n)
            def compute_slot_mapping(self,n,query,positions):
                if self.record:self.captured=(n,query.clone(),positions.clone())

        def owner(n,seed):
            rng=np.random.default_rng(seed);capacity=256;tokens=2048
            o=SimpleNamespace(device=torch.device('cuda:0'),max_model_len=131072,use_async_scheduling=True,
                use_async_spec_decode=False,speculative_config=None,enable_prompt_embeds=False,
                uses_mrope=False,uses_xdrope_dim=0,num_accepted_tokens_event=None,
                mamba_prev_last_scheduled_idx=None,lora_config=None,prev_num_spec_tokens=0)
            for name,size,dtype in [('input_ids',tokens,torch.int32),('query_start_loc',257,torch.int32),
                ('req_indices',tokens,torch.int64),('query_pos',tokens,torch.int64),
                ('num_scheduled_tokens',capacity,torch.int32),('discard_request_mask',capacity,torch.bool),
                ('num_accepted_tokens',capacity,torch.int32),('prev_positions',capacity,torch.int64)]:
                value=CpuGpuBuffer(size,dtype=dtype,device=o.device);value.cpu.fill_(-333);value.gpu.fill_(-333);setattr(o,name,value)
            o.positions=torch.full((tokens,),-333,dtype=torch.int64,device=o.device)
            o.seq_lens=torch.full((capacity,),-333,dtype=torch.int32,device=o.device)
            o.num_computed_tokens=torch.full((capacity,),-333,dtype=torch.int32,device=o.device)
            o.optimistic_seq_lens_cpu=torch.full((capacity,),-333,dtype=torch.int32,pin_memory=True)
            o.arange_np=np.arange(tokens,dtype=np.int64)
            computed_cpu=torch.zeros(capacity,dtype=torch.int32,pin_memory=True)
            values=np.resize(np.array([1,15,16,4095,4096,131070],dtype=np.int32),n) if seed%2 else rng.integers(4096,131071,n,dtype=np.int32)
            computed_cpu[:n]=torch.from_numpy(values.copy())
            token_cpu=torch.zeros((n,131072),dtype=torch.int32)
            for i in range(n):token_cpu[i,int(values[i])]=int(rng.integers(512,120000))
            ids=[f'r{i}' for i in range(n)]
            previous=torch.from_numpy(rng.integers(0,130560,(n,1),dtype=np.int32)).to(o.device)
            batch=SimpleNamespace(num_reqs=n,req_ids=ids,prev_req_id_to_index={s:i for i,s in enumerate(ids)},
                prev_sampled_token_ids=previous,req_prompt_embeds={},num_computed_tokens_cpu=computed_cpu.numpy(),
                num_computed_tokens_cpu_tensor=computed_cpu,num_prompt_tokens=np.ones(capacity,dtype=np.int32),
                token_ids_cpu=token_cpu.numpy(),token_ids_cpu_tensor=token_cpu,block_table=Slots())
            o.input_batch=batch;o.requests={s:SimpleNamespace(num_tokens=int(values[i])+1) for i,s in enumerate(ids)}
            for name in names[1:]:setattr(o,name,MethodType(namespace[name],o))
            return o

        tensors=('positions','seq_lens','num_computed_tokens')
        buffers=('input_ids','query_start_loc','req_indices','query_pos','num_scheduled_tokens','discard_request_mask','num_accepted_tokens','prev_positions')
        def compare(a,b,ra,rb):
            torch.cuda.synchronize();checks={}
            for name in tensors:checks[name]=torch.equal(getattr(a,name).cpu(),getattr(b,name).cpu())
            for name in buffers:
                checks[name+'_gpu']=torch.equal(getattr(a,name).gpu.cpu(),getattr(b,name).gpu.cpu())
                checks[name+'_cpu']=torch.equal(getattr(a,name).cpu,getattr(b,name).cpu)
            checks['optimistic_cpu']=torch.equal(a.optimistic_seq_lens_cpu,b.optimistic_seq_lens_cpu)
            checks['logits_indices']=torch.equal(ra[0].cpu(),rb[0].cpu()) and ra[0].dtype==rb[0].dtype==torch.int32
            checks['no_spec']=ra[1] is rb[1] is None
            ca,cb=a.input_batch.block_table.captured,b.input_batch.block_table.captured
            checks['slot_mapping_arguments']=ca[0]==cb[0] and torch.equal(ca[1].cpu(),cb[1].cpu()) and torch.equal(ca[2].cpu(),cb[2].cpu())
            assert all(checks.values()),checks
            return checks

        data['phase']='numerical';save()
        with torch.inference_mode(),flag_gems.use_gems():
            for n in (1,7,8,16,24,32,40,48,56,64):
                for seed in (20261016,20261017):
                    a,b=owner(n,seed),owner(n,seed)
                    schedule=SimpleNamespace(total_num_scheduled_tokens=n,scheduled_spec_decode_tokens={})
                    counts=np.ones(n,dtype=np.int32);assert candidate.eligible(b,schedule,counts)
                    ra=original(a,schedule,counts);rb=changed(b,schedule,counts)
                    checks=compare(a,b,ra,rb)
                    # Change both context lengths and previous sampled values;
                    # execute the real source method again, not a copied oracle.
                    for o in (a,b):
                        o.input_batch.num_computed_tokens_cpu[:n]+=1
                        o.input_batch.prev_sampled_token_ids.add_(17).remainder_(130560)
                        for request in o.requests.values():request.num_tokens+=1
                    ra=original(a,schedule,counts);rb=changed(b,schedule,counts)
                    again=compare(a,b,ra,rb)
                    data['cases'].append(dict(batch=n,seed=seed,all_buffers_and_padding_equal=checks,
                        changed_inputs_equal=again,positions_dtype=str(b.positions.dtype),req_indices_dtype=str(b.req_indices.gpu.dtype)))
                    save()
                    del a,b
            a,b=owner(64,20261016),owner(64,20261016)
            counts=np.ones(64,dtype=np.int32);schedule=SimpleNamespace(total_num_scheduled_tokens=64,scheduled_spec_decode_tokens={})
            for label,change,undo in (
                ('mixed_prefill',lambda:counts.__setitem__(0,2),lambda:counts.__setitem__(0,1)),
                ('no_previous',lambda:setattr(b.input_batch,'prev_sampled_token_ids',None),lambda:setattr(b.input_batch,'prev_sampled_token_ids',a.input_batch.prev_sampled_token_ids.clone())),
                ('reordered',lambda:b.input_batch.prev_req_id_to_index.__setitem__('r0',1),lambda:b.input_batch.prev_req_id_to_index.__setitem__('r0',0)),
                ('async_off',lambda:setattr(b,'use_async_scheduling',False),lambda:setattr(b,'use_async_scheduling',True)),
                ('speculation',lambda:setattr(b,'speculative_config',object()),lambda:setattr(b,'speculative_config',None)),
                ('Mrope',lambda:setattr(b,'uses_mrope',True),lambda:setattr(b,'uses_mrope',False))):
                change();rejected=not candidate.eligible(b,schedule,counts);assert rejected,label
                data['guards'].append(dict(name=label,rejected=rejected));undo()
            # Capture the kernel and change both live input arrays. Clear all
            # output buffers before replay so a stale/no-op graph cannot pass.
            ra=original(a,schedule,counts);rb=changed(b,schedule,counts)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):candidate.launch(b,64)
            for iteration in range(2):
                torch.cuda.synchronize()
                for o in (a,b):
                    o.input_batch.num_computed_tokens_cpu[:64]+=1
                    o.input_batch.prev_sampled_token_ids.add_(19).remainder_(130560)
                    for request in o.requests.values():request.num_tokens+=1
                ra=original(a,schedule,counts);rb=changed(b,schedule,counts)
                for name in buffers:
                    getattr(b,name).gpu.fill_(-333)
                b.positions.fill_(-333);b.seq_lens.fill_(-333)
                b._tianshu_decode_logit_indices.fill_(-333)
                graph.replay()
                checks=compare(a,b,ra,rb)
                data['graph_checks'].append(dict(iteration=iteration,cleared_outputs=True,all_checks_equal=checks))
            # Preserve true method costs, including H2D and input preparation.
            # Slot table is a deliberately unchanged observer stub in this probe.
            a.input_batch.block_table.record=b.input_batch.block_table.record=False
            funcs=[lambda:original(a,schedule,counts),lambda:changed(b,schedule,counts)]
            for f in funcs:
                for _ in range(5):f()
            for repeat in range(7):
                for index in ((0,1) if repeat%2==0 else (1,0)):
                    torch.cuda.synchronize();start=time.perf_counter()
                    ev0,ev1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    ev0.record()
                    for _ in range(20):funcs[index]()
                    ev1.record();ev1.synchronize()
                    data['timings'].append(dict(repeat=repeat,mode=('original','fused')[index],iterations=20,
                        host_wall_ms=1000*(time.perf_counter()-start)/20,gpu_stream_ms=ev0.elapsed_time(ev1)/20))
            data['timing_medians']={mode:{key:statistics.median(v[key] for v in data['timings'] if v['mode']==mode)
                for key in ('host_wall_ms','gpu_stream_ms')} for mode in ('original','fused')}
        for path,sha in data['source_snapshots'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
        data.update(phase='complete',complete=True,
            scope='Actual original/rendered prepare methods on controlled buffers; unchanged slot-table calls observed via stub; no model throughput claim')
    except Exception as error:
        import traceback
        data.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
    finally:save()
    return 0 if data['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
