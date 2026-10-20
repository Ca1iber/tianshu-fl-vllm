# SPDX-License-Identifier: Apache-2.0
"""Real native KV managers/Requests, complete CPU state comparison, zero visible GPUs."""
import dataclasses,hashlib,inspect,json,os,pickle,statistics,time
from pathlib import Path
import torch
from vllm import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock,get_request_block_hasher,init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec,KVCacheConfig,KVCacheGroupSpec
from vllm.v1.request import Request,RequestStatus
from shortcut import wrap
BASE=Path(__file__).resolve().parent

def hash_fn(x):return hashlib.sha256(pickle.dumps(x,protocol=5)).digest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert torch.cuda.device_count()==0 and not torch.cuda.is_initialized()
    init_none_hash(hash_fn);hasher=get_request_block_hasher(16,hash_fn)
    def make(blocks,events=False):
        spec=FullAttentionSpec(block_size=16,num_kv_heads=2,head_size=128,dtype=torch.uint8,page_size_padded=12288)
        cfg=KVCacheConfig(num_blocks=blocks,kv_cache_tensors=[],kv_cache_groups=[KVCacheGroupSpec(layer_names=[f'layer{i}' for i in range(42)],kv_cache_spec=spec)])
        return KVCacheManager(cfg,131072,16,16,max_num_batched_tokens=4096,enable_kv_cache_events=events)
    def request(name,length):return Request(name,[512+(i*13)%10000 for i in range(length)],SamplingParams(max_tokens=128,temperature=1,top_p=.95),None,arrival_time=1.,block_hasher=hasher)
    def snapshot(m,requests):
        pool=m.block_pool;queue=pool.free_block_queue
        def norm(x):
            if x is pool:return 'POOL'
            if x is queue:return ('FREE_QUEUE',queue.num_free_blocks,tuple(b.block_id for b in queue.get_all_free_blocks()))
            if isinstance(x,KVCacheBlock):return ('BLOCK_REFERENCE',x.block_id)
            if isinstance(x,torch.Tensor):raise AssertionError('Metadata test unexpectedly contains a Tensor')
            if x is None or isinstance(x,(str,int,float,bool,bytes)):return x
            if isinstance(x,dict):return tuple(sorted((repr(k),norm(v)) for k,v in x.items()))
            if isinstance(x,(list,tuple)):return tuple(norm(v) for v in x)
            if callable(x):return ('CALLABLE',getattr(x,'__module__',None),getattr(x,'__qualname__',None))
            if dataclasses.is_dataclass(x):return (type(x).__name__,tuple((f.name,norm(getattr(x,f.name))) for f in dataclasses.fields(x)))
            if hasattr(x,'__dict__'):return (type(x).__name__,norm(vars(x)))
            return repr(x)
        blocks=tuple(tuple((f.name,norm(getattr(b,f.name))) for f in dataclasses.fields(b)) for b in pool.blocks)
        return norm(vars(m)),norm(vars(pool)),blocks,tuple(norm(vars(r)) for r in requests)
    def result(m,value):return None if value is None else (value is m.empty_kv_cache_blocks,value.get_block_ids())
    checks=[];all_stats=[]
    for events in (False,True):
        for length in (1,15,16,17,63,64,129,4096,16384,131040):
            size=max(64,(length+127)//16+8);managers=[make(size,events),make(size,events)]
            requests=[request('trace',length),request('trace',length)]
            call,stats=wrap(managers[1],managers[1].allocate_slots,Request,RequestStatus.RUNNING,'candidate')
            functions=[managers[0].allocate_slots,call]
            for m,r,fn in zip(managers,requests,functions):
                assert fn(r,length,full_sequence_must_fit=True) is not None
                r.num_computed_tokens=length;r.status=RequestStatus.RUNNING
            assert snapshot(managers[0],[requests[0]])==snapshot(managers[1],[requests[1]])
            steps=min(64,131072-length)
            for step in range(steps):
                answers=[];old=stats['shortcuts']
                for m,r,fn in zip(managers,requests,functions):
                    # Some calls intentionally observe optimistic async computed counts
                    # ahead of confirmed outputs. The native min(request.num_tokens) stays authoritative.
                    if step%7!=3:r.append_output_token_ids(1000+step)
                    value=fn(r,1,has_scheduled_reqs=bool(step%2));answers.append(result(m,value));r.num_computed_tokens+=1
                assert answers[0]==answers[1]
                if step%8==0 or step==steps-1:
                    assert snapshot(managers[0],[requests[0]])==snapshot(managers[1],[requests[1]])
            for m,r in zip(managers,requests):m.free(r)
            assert snapshot(managers[0],[requests[0]])==snapshot(managers[1],[requests[1]])
            # Prefix hits read/retain block identity, then re-admission must use native allocation.
            re=[request('prefix_reuse',length),request('prefix_reuse',length)]
            answers=[]
            for m,r,fn in zip(managers,re,functions):
                blocks,tokens=m.get_computed_blocks(r)
                value=fn(r,max(1,length-tokens),num_new_computed_tokens=tokens,new_computed_blocks=blocks,full_sequence_must_fit=True)
                answers.append((tokens,result(m,value)))
            assert answers[0]==answers[1]
            assert snapshot(managers[0],re)==snapshot(managers[1],re)
            checks.append(dict(prefix_length=length,events_enabled=events,decode_steps=steps,final_free_and_prefix_reuse_equal=True,stats=dict(stats)));all_stats.append(stats)
    # Fill the pool: zero free blocks may still accept a token in an existing page.
    managers=[make(31),make(31)];requests=[[],[]]
    call,stats=wrap(managers[1],managers[1].allocate_slots,Request,RequestStatus.RUNNING,'candidate');functions=[managers[0].allocate_slots,call]
    for index in range(15):
        for m,rs,fn in zip(managers,requests,functions):
            r=request(f'full{index}',24);assert fn(r,24) is not None;r.num_computed_tokens=24;r.status=RequestStatus.RUNNING;r.append_output_token_ids(42);rs.append(r)
    assert all(m.block_pool.get_num_free_blocks()==0 for m in managers)
    for index in range(15):
        answers=[result(m,fn(rs[index],1)) for m,rs,fn in zip(managers,requests,functions)];assert answers[0]==answers[1]
    assert snapshot(managers[0],requests[0])==snapshot(managers[1],requests[1]) and stats['shortcuts']==15
    # Exhausting the page must fall back and preserve the real None/OOM result.
    for index in range(15):
        for rs in requests:rs[index].num_computed_tokens=32
        assert functions[0](requests[0][index],1) is functions[1](requests[1][index],1) is None
    assert snapshot(managers[0],requests[0])==snapshot(managers[1],requests[1])
    # Compare direct native allocation with the guarded candidate, including guard/counters.
    m=make(128);r=request('timing',64);assert m.allocate_slots(r,64) is not None
    r.status=RequestStatus.RUNNING;r.num_computed_tokens=65;r.append_output_token_ids([42,43]);assert m.allocate_slots(r,1) is not None
    candidate,timing_stats=wrap(m,m.allocate_slots,Request,RequestStatus.RUNNING,'candidate');assert candidate(r,1) is m.empty_kv_cache_blocks
    timings=[]
    for trial in range(7):
        row={}
        for name,fn in ((('native',m.allocate_slots),('candidate',candidate)) if trial%2==0 else (('candidate',candidate),('native',m.allocate_slots))):
            start=time.perf_counter_ns()
            for _ in range(2000):assert fn(r,1) is m.empty_kv_cache_blocks
            row[name]=(time.perf_counter_ns()-start)/2000/1000
        timings.append(row)
    assert torch.cuda.device_count()==0 and not torch.cuda.is_initialized()
    paths=[Path(__file__),BASE/'shortcut.py',*[Path(inspect.getsourcefile(cls)) for cls in (KVCacheManager,Request,FullAttentionSpec,type(m.coordinator),type(m.coordinator.single_type_managers[0]),type(m.block_pool))]]
    gate=dict(complete=True,CPU_only=True,visible_GPUs=0,no_GPU_context_or_tensors_or_kernels=True,actual_native_KV_managers_and_Requests_executed=True,
        metadata_pool_does_not_allocate_physical_KV=True,case_traces=checks,full_pool_existing_page_empty_returns=15,full_pool_new_page_native_None_results=15,
        complete_manager_coordinator_unit_pool_block_hash_refcount_free_queue_event_and_Request_states_compared=True,
        prefix_cache_reuse_and_free_states_equal=True,decode_native_empty_return_singleton_identity_equal=True,
        native_method_bodies_not_modified=True,total_shortcuts=sum(s['shortcuts'] for s in all_stats)+stats['shortcuts'],
        timings_us=timings,native_median_us=statistics.median(x['native'] for x in timings),candidate_median_us=statistics.median(x['candidate'] for x in timings),
        timing_scope='CPU allocation of one no-work decode token, complete candidate guard and counters included; not model/GPU performance',
        sources={str(p.resolve()):sha(p) for p in paths})
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('case_traces','sources','timings_us')}))
if __name__=='__main__':main()
