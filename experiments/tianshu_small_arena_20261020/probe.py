# SPDX-License-Identifier: Apache-2.0
"""GPU staging-capacity/bitwise/guard/reused-Graph boundary proof; no model score."""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent;PARENT=BASE.parent/'tianshu_k8v16_pipeline_20261019'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);a=p.parse_args()
remaining=int((datetime.fromisoformat(a.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>0;signal.alarm(min(900,remaining))
before=subprocess.check_output(['ixsmi'],text=True);assert 'Iluvatar BI-V150' in before and 'No running processes found' in before
root=a.output.resolve();assert root.is_relative_to('/workspace/benchmark-results');root.mkdir(parents=True,exist_ok=False)
(root/'hardware_before.txt').write_text(before)
import torch
import policy
sys.path.append(str(PARENT))
gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['CPU_only']
for path,h in gate['sources'].items():assert sha(path)==h,path
assert torch.cuda.device_count()==1;torch.set_num_threads(8)
sources={**gate['sources'],str(BASE/'CPU_GATE_32g.json'):sha(BASE/'CPU_GATE_32g.json'),str(Path(__file__).resolve()):sha(__file__)}
for path,h in sources.items():
    out=root/'source_snapshot'/Path(path).relative_to('/');out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(Path(path).read_bytes());assert sha(out)==h
result=dict(complete=False,phase='staging GPU',runner_pid=os.getpid(),frozen_sources=sources,cases=[],mode_reservations={},
            official_benchmark=False,service_executed=False,model_executed=False,adopted=False,
            scope='Capacity-only staging helper, native byte fallback identity preserved by inheritance, no full attention/model performance claim')
def save():
    temp=root/'result.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(root/'result.json')
save()
try:
    models=[]
    for mode in ('control','candidate'):
        module=load('_capacity_GPU_'+mode,PARENT/'bounded_prefill.py');proof=policy.install(module,mode,root/mode)
        obj=module.BoundedDensePrefill();obj.reserve();pointer=obj.storage.data_ptr();table_pointer=obj.dense_ids.data_ptr();obj.reserve()
        assert pointer==obj.storage.data_ptr() and table_pointer==obj.dense_ids.data_ptr()
        record=dict(obj.reservation,arena_ptr=pointer,table_ptr=table_pointer,proof=proof)
        assert record['allocated_after']-record['allocated_before']>=record['scratch_bytes']+record['table_bytes']
        result['mode_reservations'][mode]=record;models.append((mode,obj))
    values=[((((i&127)<<4)+14848) if (i&127)>=8 else 0)|((i&128)<<8) for i in range(256)]
    lookup=torch.tensor(values,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    def dequantize(x):return lookup[x.long()]
    cases=[('max_context',1,131072,False),('two_contexts',2,65536,False),('eight_contexts',8,16384,False),
           ('many_prefills',32,4096,False),('over_boundary',8,16385,False),('strided_pages',8,16384,True)]
    for case_id,(name,n,length,strided) in enumerate(cases):
        torch.manual_seed(2026102000+case_id)
        blocks=(length+15)//16;pages=n*blocks;stride=24576 if strided else 12288
        raw=torch.full(((pages+2)*stride+256,),83,device='cuda',dtype=torch.uint8)
        k=torch.as_strided(raw,(pages+2,16,2,128),(stride,256,128,1))
        v=torch.as_strided(raw[:-256].view(torch.bfloat16),(pages+2,16,2,128),(stride//2,256,128,1),2048)
        k.copy_(torch.randint(0,256,k.shape,device='cuda',dtype=torch.uint8))
        v.copy_(torch.randn(v.shape,device='cuda',dtype=torch.bfloat16)*.25)
        table=torch.randperm(pages+2,device='cuda',dtype=torch.int64)[:pages].to(torch.int32).reshape(n,blocks)
        used=torch.tensor([length-i%7 for i in range(n)],device='cuda',dtype=torch.int32)
        kw=dict(k=k,v=v,block_table=table,seqused_k=used,max_seqlen_k=length)
        graphs={};checks=[]
        for phase in (0,1):
            if phase:
                k.bitwise_xor_(128);v.neg_();table.copy_(table.roll(1,1));used.sub_(7)
            snapshot=raw.clone();rng=torch.cuda.get_rng_state().clone()
            for mode,obj in models:
                obj.storage.fill_(17.)
                if pages>obj.limit_pages:
                    assert obj(kw) is None
                    assert bool((obj.storage==17.).all()) and obj.storage.data_ptr()==result['mode_reservations'][mode]['arena_ptr']
                    checks.append(dict(mode=mode,phase=phase,fallback=True,arena_unchanged=True))
                    continue
                if not phase:
                    obj(kw);graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):obj(kw)
                    graphs[mode]=graph
                graphs[mode].replay();torch.cuda.synchronize()
                proof=obj.verify_last(dequantize)
                assert bool((obj.storage[pages:]==17.).all())
                for seq,valid in enumerate(used.cpu().tolist()):
                    begin=seq*blocks
                    for tensor in (obj.last['dense_k'],obj.last['dense_v']):
                        assert bool((tensor[begin:begin+blocks].reshape(-1,2,128)[valid:]==17.).all())
                assert obj.storage.data_ptr()==result['mode_reservations'][mode]['arena_ptr'] and obj.dense_ids.data_ptr()==result['mode_reservations'][mode]['table_ptr']
                checks.append(dict(mode=mode,phase=phase,fallback=False,**proof,unused_pages_and_rows_guarded=True,reused_pointers=True))
            assert torch.equal(raw,snapshot) and torch.equal(torch.cuda.get_rng_state(),rng)
            if pages<=models[1][1].limit_pages:
                assert torch.equal(models[0][1].storage[:pages],models[1][1].storage[:pages])
        result['cases'].append(dict(name=name,sequences=n,max_length=length,pages=pages,strided=strided,checks=checks,
                                   same_Graph_two_phases=True,raw_cache_unchanged=True,RNG_unchanged=True))
        save();print(json.dumps(dict(case=name,complete=True,fallbacks=sum(x['fallback'] for x in checks))),flush=True)
        del raw,k,v,table,used,snapshot,graphs
    assert all(sha(path)==h for path,h in sources.items())
    result.update(complete=True,phase='complete',all_staged_values_bitwise=True,all_guards_and_native_fallback_verified=True,
                  scratch_saved_bytes=result['mode_reservations']['control']['scratch_bytes']-result['mode_reservations']['candidate']['scratch_bytes'],
                  table_saved_bytes=result['mode_reservations']['control']['table_bytes']-result['mode_reservations']['candidate']['table_bytes'])
    save()
except BaseException as error:
    import traceback
    result.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
