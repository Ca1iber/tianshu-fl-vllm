# SPDX-License-Identifier: Apache-2.0
"""Observe original scheduler and KV calls; do not change decisions or outputs."""
import atexit
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys
import time

BASE=Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE.parent/'tianshu_gemm_topp_geo_20261016/hooks/sitecustomize.py'))
MANIFEST=json.loads((BASE/'SOURCE_MANIFEST_32g.json').read_text())['files']
DIRECTORY=Path(os.environ['TIANSHU_SCHEDULER_TRACE_DIR'])
active=None
rows=[]
outside={}
step_index=0
observed_sources={}


def add(name,elapsed,**counts):
    target=outside if active is None else active
    row=target.setdefault(name,dict(calls=0,wall_ns=0))
    row['calls']+=1;row['wall_ns']+=elapsed
    for key,value in counts.items():row[key]=row.get(key,0)+value


def flush():
    if not observed_sources:return
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    if rows:
        with (DIRECTORY/f'steps_{os.getpid()}.jsonl').open('a') as f:
            for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
        rows.clear()
    target=DIRECTORY/f'summary_{os.getpid()}.json'
    tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(pid=os.getpid(),steps=step_index,outside_schedule=outside,
        observed_sources=observed_sources,observer_only=True,policy_changed=False,
        timings_include_nested_observer_overhead=True),indent=2)+'\n');tmp.replace(target)


atexit.register(flush)


def scheduler_hook(module):
    original=module.Scheduler.schedule
    def schedule(owner,*args,**kwargs):
        global active,step_index
        # Flush before a new step, not in the measured original schedule call.
        if rows and (len(rows)>=128 or (DIRECTORY/'flush_requested').exists()):flush()
        before=dict(waiting=len(owner.waiting),running=len(owner.running),
                    free_blocks=owner.kv_cache_manager.block_pool.get_num_free_blocks())
        calls={};active=calls;started=time.perf_counter_ns()
        try:result=original(owner,*args,**kwargs)
        finally:
            elapsed=time.perf_counter_ns()-started;active=None
        computed={r.req_id:r.num_computed_tokens for r in result.scheduled_new_reqs}
        cached=result.scheduled_cached_reqs
        computed.update(zip(cached.req_ids,cached.num_computed_tokens))
        counts=result.num_scheduled_tokens
        assert sum(counts.values())==result.total_num_scheduled_tokens
        assert set(counts).issubset(computed)
        prefill_tokens=0;decode_tokens=0;prefill_reqs=0;decode_reqs=0
        for rid,count in counts.items():
            prompt=owner.requests[rid].num_prompt_tokens
            prefill=min(count,max(0,prompt-computed[rid]));decode=count-prefill
            prefill_tokens+=prefill;decode_tokens+=decode
            prefill_reqs+=int(prefill>0);decode_reqs+=int(decode>0)
        step_index+=1
        rows.append(dict(step=step_index,wall_ns=elapsed,before=before,
            throttle_prefills=bool(args[0] if args else kwargs.get('throttle_prefills',False)),
            after=dict(waiting=len(owner.waiting),running=len(owner.running),
                free_blocks=owner.kv_cache_manager.block_pool.get_num_free_blocks()),
            scheduled_reqs=len(counts),scheduled_tokens=result.total_num_scheduled_tokens,
            prefill_tokens=prefill_tokens,decode_tokens=decode_tokens,prefill_reqs=prefill_reqs,decode_reqs=decode_reqs,
            preempted=len(result.preempted_req_ids or ()),finished=len(result.finished_req_ids),
            fresh_zero_blocks=len(result.new_block_ids_to_zero or ()),calls=calls))
        if (DIRECTORY/'flush_requested').exists():flush()
        return result
    module.Scheduler.schedule=schedule


def manager_hook(module):
    original=module.KVCacheManager.allocate_slots
    def allocate(owner,*args,**kwargs):
        started=time.perf_counter_ns();result=original(owner,*args,**kwargs)
        elapsed=time.perf_counter_ns()-started
        blocks=0 if result is None else sum(len(g) for g in result.blocks)
        add('allocate_slots',elapsed,failed=int(result is None),empty=int(result is not None and blocks==0),new_blocks=blocks)
        return result
    module.KVCacheManager.allocate_slots=allocate
    original_cache=module.KVCacheManager.cache_blocks
    def cache(owner,*args,**kwargs):
        started=time.perf_counter_ns();result=original_cache(owner,*args,**kwargs)
        add('manager_cache_blocks',time.perf_counter_ns()-started)
        return result
    module.KVCacheManager.cache_blocks=cache


def single_hook(module):
    original=module.SingleTypeKVCacheManager.cache_blocks
    def cache(owner,request,num_tokens,*args,**kwargs):
        early=owner.num_cached_block.get(request.request_id,0)>=num_tokens//owner.block_size
        started=time.perf_counter_ns();result=original(owner,request,num_tokens,*args,**kwargs)
        add('single_cache_blocks',time.perf_counter_ns()-started,already_full=int(early))
        return result
    module.SingleTypeKVCacheManager.cache_blocks=cache


TARGETS={
    'vllm.v1.core.sched.scheduler':scheduler_hook,
    'vllm.v1.core.kv_cache_manager':manager_hook,
    'vllm.v1.core.single_type_kv_cache_manager':single_hook,
}


class TraceLoader(importlib.abc.Loader):
    def __init__(self,original,name):self.original=original;self.name=name
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        relative=self.name.replace('.','/')+'.py'
        sha=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        assert sha==MANIFEST[relative]
        observed_sources[module.__file__]=sha
        TARGETS[self.name](module);flush()


class TraceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in TARGETS:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None or spec.loader is None:raise RuntimeError('Trace target missing')
        spec.loader=TraceLoader(spec.loader,fullname);return spec


sys.meta_path.insert(0,TraceFinder())
