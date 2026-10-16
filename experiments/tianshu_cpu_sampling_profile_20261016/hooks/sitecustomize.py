# SPDX-License-Identifier: Apache-2.0
"""Intrusive CPU call profiling on 16 armed FULL/C64 decode frames; never a score."""
import cProfile
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import pstats
import runpy
import sys
import threading
import time

BASE=Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE/'tianshu_max_seqs_20261014/hooks/sitecustomize.py'))
RUNNER_SHA='b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'


def stats_rows(profiler):
    rows=[]
    for (filename,line,name),(primitive,total,local,cumulative,callers) in pstats.Stats(profiler).stats.items():
        rows.append(dict(file=filename,line=line,name=name,primitive_calls=primitive,total_calls=total,
            local_wall_seconds=local,cumulative_wall_seconds=cumulative,
            callers=[dict(file=k[0],line=k[1],name=k[2],stats=v) for k,v in sorted(callers.items())]))
    return sorted(rows,key=lambda r:(-r['cumulative_wall_seconds'],r['file'],r['line'],r['name']))


def tensor_description(tensor):
    if tensor is None:return None
    return dict(shape=list(tensor.shape),stride=list(tensor.stride()),dtype=str(tensor.dtype),
        device=str(tensor.device),data_ptr=int(tensor.data_ptr()),
        storage_ptr=int(tensor.untyped_storage().data_ptr()),storage_offset=int(tensor.storage_offset()),
        contiguous=bool(tensor.is_contiguous()))


class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==RUNNER_SHA
        import torch
        cls=module.ModelRunnerFL
        original_execute=cls.execute_model;original_sample=cls.sample_tokens
        original_dispatch=cls._determine_batch_execution_and_padding
        root=Path(os.environ['TIANSHU_CPU_PROFILE_RECORD_DIR']);root.mkdir(parents=True,exist_ok=True)
        marker=Path(os.environ['TIANSHU_CPU_PROFILE_ARM_FILE'])
        assert os.environ['TIANSHU_CPU_PROFILE_MODE']=='cProfile'

        def state(owner):
            if not hasattr(owner,'_tianshu_cpu_profile'):
                owner._tianshu_cpu_profile=dict(warm=0,active=None,frames=[],done=False)
            return owner._tianshu_cpu_profile

        def call(entry,name,fn,*args,**kwargs):
            assert not torch.cuda.is_current_stream_capturing()
            profiler=cProfile.Profile();wall=time.perf_counter_ns();cpu=time.thread_time_ns();thread=threading.get_ident()
            profiler.enable()
            try:return fn(*args,**kwargs)
            finally:
                profiler.disable()
                entry[name]=dict(wall_ms=(time.perf_counter_ns()-wall)/1e6,thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6,
                    thread_id=thread,rows=stats_rows(profiler))
                assert threading.get_ident()==thread

        def finish(coordinator):
            sources={}
            for frame in coordinator['frames']:
                assert frame['dispatch']==dict(mode='FULL',padded_tokens=64)
                for phase in ('execute_model','sample_tokens'):
                    assert frame[phase]['rows'] and frame[phase]['wall_ms']>0
                    for row in frame[phase]['rows']:
                        p=Path(row['file'])
                        allowed=p.is_absolute() and (p.is_relative_to('/usr/local/lib/python3.12') or p.is_relative_to('/workspace/vllm-plugin-FL'))
                        if allowed and p.is_file() and str(p) not in sources:
                            raw=p.read_bytes();dest=root/'source_snapshot'/p.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
                            sources[str(p)]=dict(sha256=hashlib.sha256(raw).hexdigest(),snapshot=str(dest.relative_to(root)))
            out=dict(complete=True,pid=os.getpid(),frames=coordinator['frames'],frame_count=16,
                runner_source_sha256=RUNNER_SHA,hook_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),extra_sources=sources,
                profiler='cProfile default host wall clock; C-call spans can include blocked waits',GPU_events_used=False,
                interpretation='Intrusive CPU diagnostic only. Cumulative times overlap; do not sum ancestors/descendants or infer kernel-active GPU time or benchmark throughput.')
            (root/f'cpu_{os.getpid()}.json').write_text(json.dumps(out,indent=2)+'\n');coordinator['done']=True

        def dispatch(owner,*args,**kwargs):
            result=original_dispatch(owner,*args,**kwargs);s=state(owner)
            if s['active'] is not None:s['active']['dispatch']=dict(mode=result[0].name,padded_tokens=int(result[1].num_tokens))
            return result

        def execute(owner,scheduler_output,*args,**kwargs):
            s=state(owner)
            if s['done'] or not marker.exists():return original_execute(owner,scheduler_output,*args,**kwargs)
            tokens=scheduler_output.num_scheduled_tokens
            if len(tokens)!=64 or any(n!=1 for n in tokens.values()):
                assert s['active'] is None;s['warm']=0
                return original_execute(owner,scheduler_output,*args,**kwargs)
            s['warm']+=1
            if s['warm']<=32:return original_execute(owner,scheduler_output,*args,**kwargs)
            assert s['active'] is None and owner.speculative_config is None
            count=owner.input_batch.num_reqs
            assert count==64
            computed=owner.input_batch.num_computed_tokens_cpu[:count];prompts=owner.input_batch.num_prompt_tokens[:count]
            entry=dict(index=len(s['frames']),context=dict(requests=int(count),min_computed=int(min(computed)),max_computed=int(max(computed)),min_prompt=int(min(prompts)),max_prompt=int(max(prompts))))
            assert entry['context']['min_computed']>=4096 and entry['context']['min_prompt']==entry['context']['max_prompt']==4096
            s['active']=entry
            result=call(entry,'execute_model',original_execute,owner,scheduler_output,*args,**kwargs)
            assert result is None
            return result

        def sample(owner,*args,**kwargs):
            s=state(owner);entry=s['active']
            if entry is None:return original_sample(owner,*args,**kwargs)
            ephemeral=owner.execute_model_state
            assert ephemeral is not None and len(ephemeral)==10
            metadata=owner.input_batch.sampling_metadata
            entry['buffers']={name:tensor_description(tensor) for name,tensor in (
                ('logits',ephemeral[1]),('hidden_states',ephemeral[4]),('sample_hidden_states',ephemeral[5]),
                ('temperature',metadata.temperature),('top_p',metadata.top_p),('top_k',metadata.top_k))}
            entry['sampling_flags']=dict(all_greedy=metadata.all_greedy,all_random=metadata.all_random,
                no_penalties=metadata.no_penalties,max_num_logprobs=metadata.max_num_logprobs,
                generators=len(metadata.generators),bad_words_requests=len(metadata.bad_words_token_ids),
                allowed_token_ids_mask=tensor_description(metadata.allowed_token_ids_mask),
                has_thinking_budget_holder=metadata.thinking_budget_state_holder is not None)
            result=call(entry,'sample_tokens',original_sample,owner,*args,**kwargs)
            s['frames'].append(entry);s['active']=None
            if len(s['frames'])==16:finish(s)
            return result
        cls.execute_model=execute;cls.sample_tokens=sample;cls._determine_batch_execution_and_padding=dispatch


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm_fl.worker.model_runner':return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        assert spec is not None and spec.loader is not None
        spec.loader=Loader(spec.loader);return spec


if os.environ.get('TIANSHU_CPU_PROFILE_PROBE')=='1':sys.meta_path.insert(0,Finder())
