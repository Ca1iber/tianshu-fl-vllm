# SPDX-License-Identifier: Apache-2.0
"""Unprofiled, outside-Graph phase events on the actual ModelRunnerFL path."""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

BASE=Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE/'tianshu_prefill_int8_service_20261021/hooks_candidate/sitecustomize.py'))
RUNNER_SHA='b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'


class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==RUNNER_SHA
        import torch
        import inspect
        cls=module.ModelRunnerFL
        original_execute=cls.execute_model;original_sample_tokens=cls.sample_tokens
        original_forward=cls._model_forward;original_sample=cls._sample
        original_dispatch=cls._determine_batch_execution_and_padding
        directory=Path(os.environ['TIANSHU_DENSE_INT8_RECORD_DIR']).parent/'stage_records'
        marker=directory.parent/'profile_armed'
        directory.mkdir(parents=True,exist_ok=True)

        def coordinator(owner):
            if not hasattr(owner,'_tianshu_stage'):
                owner._tianshu_stage=dict(warm=0,active=None,frames=[],done=False,extra_sources={})
            return owner._tianshu_stage

        def timed(owner,name,function,args,kwargs):
            state=coordinator(owner);entry=state['active']
            if entry is None:return function(*args,**kwargs)
            assert not torch.cuda.is_current_stream_capturing(),name+' event inside graph capture'
            assert name not in entry['cpu'],name+' called more than once'
            begin,end=entry['events'][name];begin.record()
            wall=time.perf_counter_ns();cpu=time.thread_time_ns();thread=threading.get_ident()
            try:return function(*args,**kwargs)
            finally:
                entry['cpu'][name]=dict(wall_ms=(time.perf_counter_ns()-wall)/1e6,
                    thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6,thread_id=thread)
                assert threading.get_ident()==thread
                end.record()

        def forward(owner,*args,**kwargs):
            state=coordinator(owner)
            if state['active'] is not None:
                count=owner.input_batch.num_reqs
                computed=owner.input_batch.num_computed_tokens_cpu[:count]
                prompts=owner.input_batch.num_prompt_tokens[:count]
                state['active']['context']=dict(requests=int(count),min_computed=int(min(computed)),
                    max_computed=int(max(computed)),min_prompt=int(min(prompts)),max_prompt=int(max(prompts)))
            return timed(owner,'forward',lambda *a,**k:original_forward(owner,*a,**k),args,kwargs)

        def sample(owner,*args,**kwargs):
            return timed(owner,'sample',lambda *a,**k:original_sample(owner,*a,**k),args,kwargs)

        def dispatch(owner,*args,**kwargs):
            result=original_dispatch(owner,*args,**kwargs);state=coordinator(owner)
            if state['active'] is not None:
                state['active']['dispatch']=dict(mode=result[0].name,padded_tokens=int(result[1].num_tokens))
            return result

        def start(owner,state):
            assert owner.speculative_config is None
            assert not torch.cuda.is_current_stream_capturing()
            state['pool']=[{name:(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                for name in ('total','forward','logits','sample')} for _ in range(40)]
            for events in state['pool']:
                for pair in events.values():
                    for event in pair:event.record()
            # Only boundary synchronization, never one synchronization per step.
            torch.cuda.synchronize()
            old_logits=owner.model.compute_logits
            state['original_logits']=old_logits
            source=Path(inspect.getsourcefile(inspect.unwrap(old_logits)))
            data=source.read_bytes();sha=hashlib.sha256(data).hexdigest()
            target=directory/'source_snapshot'/source.relative_to('/')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            state['extra_sources'][str(source)]=dict(sha256=sha,snapshot=str(target.relative_to(directory)))
            state['model_type']=type(owner.model).__module__+'.'+type(owner.model).__qualname__
            state['logits_method']=getattr(old_logits,'__qualname__',repr(old_logits))
            def logits(*args,**kwargs):return timed(owner,'logits',old_logits,args,kwargs)
            owner.model.compute_logits=logits

        def finish(owner,state):
            torch.cuda.synchronize()
            frames=[]
            for entry in state['frames']:
                assert set(entry['cpu'])=={'forward','logits','sample','execute','sample_tokens'}
                assert entry['dispatch']==dict(mode='FULL',padded_tokens=64)
                context=entry['context'];assert context['requests']==64 and context['min_computed']>=4096 and context['min_prompt']==context['max_prompt']==4096
                gpu={name:pair[0].elapsed_time(pair[1]) for name,pair in entry['events'].items()}
                assert all(value>0 for value in gpu.values())
                assert gpu['total']+.1>=sum(gpu[name] for name in ('forward','logits','sample'))
                frames.append(dict(index=entry['index'],gpu_stream_intervals_ms=gpu,cpu_call_spans=entry['cpu'],
                    dispatch=entry['dispatch'],context=context))
            for path,metadata in state['extra_sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==metadata['sha256']
            owner.model.compute_logits=state['original_logits']
            state['done']=True;state['active']=None
            output=dict(complete=True,pid=os.getpid(),frames=frames,frame_count=40,
                model_type=state['model_type'],logits_method=state['logits_method'],extra_sources=state['extra_sources'],
                runner_source_sha256=RUNNER_SHA,hook_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                events_inside_graph=False,ixsys_not_used=True,boundary_synchronizations=2,
                interpretation='GPU stream intervals may contain host submission gaps; CPU wall spans include waits. Thread CPU excludes blocked time.',
                torch_version=torch.__version__)
            (directory/f'stages_{os.getpid()}.json').write_text(json.dumps(output,indent=2)+'\n')

        def execute(owner,scheduler_output,*args,**kwargs):
            state=coordinator(owner)
            if state['done'] or not marker.exists():return original_execute(owner,scheduler_output,*args,**kwargs)
            tokens=scheduler_output.num_scheduled_tokens
            eligible=len(tokens)==64 and all(n==1 for n in tokens.values())
            if not eligible:
                assert state['active'] is None,'Lost C64 inside measured window'
                state['warm']=0
                return original_execute(owner,scheduler_output,*args,**kwargs)
            state['warm']+=1
            if state['warm']<=32:return original_execute(owner,scheduler_output,*args,**kwargs)
            assert state['active'] is None
            if not state['frames']:start(owner,state)
            index=len(state['frames']);assert index<40
            entry=dict(index=index,events=state['pool'][index],cpu={});state['active']=entry
            entry['events']['total'][0].record()
            wall=time.perf_counter_ns();cpu=time.thread_time_ns();thread=threading.get_ident()
            result=original_execute(owner,scheduler_output,*args,**kwargs)
            entry['cpu']['execute']=dict(wall_ms=(time.perf_counter_ns()-wall)/1e6,
                thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6,thread_id=thread)
            assert result is None
            return result

        def sample_tokens(owner,*args,**kwargs):
            state=coordinator(owner);entry=state['active']
            if entry is None:return original_sample_tokens(owner,*args,**kwargs)
            wall=time.perf_counter_ns();cpu=time.thread_time_ns();thread=threading.get_ident()
            result=original_sample_tokens(owner,*args,**kwargs)
            entry['cpu']['sample_tokens']=dict(wall_ms=(time.perf_counter_ns()-wall)/1e6,
                thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6,thread_id=thread)
            entry['events']['total'][1].record();state['frames'].append(entry);state['active']=None
            if len(state['frames'])==40:finish(owner,state)
            return result

        cls.execute_model=execute;cls.sample_tokens=sample_tokens
        cls._model_forward=forward;cls._sample=sample;cls._determine_batch_execution_and_padding=dispatch


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm_fl.worker.model_runner':
            return None
        index = sys.meta_path.index(self)
        sys.meta_path.pop(index)
        try:
            spec = None
            for finder in tuple(sys.meta_path):
                method = getattr(finder, 'find_spec', None)
                if method is not None:
                    spec = method(fullname, path, target)
                    if spec is not None:
                        break
        finally:
            sys.meta_path.insert(index, self)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec

if True:
    sys.meta_path.insert(0, Finder())
