# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.abc,importlib.machinery,json,os,runpy,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE/'tianshu_W4_LMpad_service_v2_20261023/hooks_candidate/sitecustomize.py'))
RUNNER_SHA='b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'
class Loader(importlib.abc.Loader):
 def __init__(self,original):self.original=original
 def create_module(self,spec):return self.original.create_module(spec)
 def exec_module(self,module):
  self.original.exec_module(module);assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==RUNNER_SHA
  import torch
  cls=module.ModelRunnerFL;execute0,sample0,dispatch0=cls.execute_model,cls.sample_tokens,cls._determine_batch_execution_and_padding
  root=Path(os.environ['TIANSHU_GPU_PROFILE_RECORD_DIR']);root.mkdir(parents=True,exist_ok=True);marker=Path(os.environ['TIANSHU_GPU_PROFILE_ARM_FILE'])
  def state(owner):
   if not hasattr(owner,'_tianshu_gpu_profile'):owner._tianshu_gpu_profile=dict(warm=0,active=None,frames=[],done=False,profiler=None)
   return owner._tianshu_gpu_profile
  def call(entry,name,fn,*a,**k):
   assert not torch.cuda.is_current_stream_capturing();wall=time.perf_counter_ns();cpu=time.thread_time_ns()
   try:return fn(*a,**k)
   finally:entry[name]=dict(wall_ms=(time.perf_counter_ns()-wall)/1.e6,thread_cpu_ms=(time.thread_time_ns()-cpu)/1.e6)
  def finish(s):
   s['profiler'].__exit__(None,None,None);trace=root/f'cuda_trace_{os.getpid()}.json';s['profiler'].export_chrome_trace(str(trace));raw=trace.read_bytes();events=json.loads(raw)['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel'];assert kernels
   assert len(s['frames'])==16 and all(f['dispatch']==dict(mode='FULL',padded_tokens=64) for f in s['frames'])
   result=dict(complete=True,pid=os.getpid(),frames=s['frames'],frame_count=16,GPU_events_used=True,trace_file=str(trace),trace_sha256=hashlib.sha256(raw).hexdigest(),GPU_kernel_events=len(kernels),runner_source_sha256=RUNNER_SHA,hook_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),diagnostic_only=True,no_benchmark_score=True)
   (root/f'gpu_{os.getpid()}.json').write_text(json.dumps(result,indent=2)+'\n');s['done']=True
  def dispatch(owner,*a,**k):
   result=dispatch0(owner,*a,**k);s=state(owner)
   if s['active'] is not None:s['active']['dispatch']=dict(mode=result[0].name,padded_tokens=int(result[1].num_tokens))
   return result
  def execute(owner,scheduler_output,*a,**k):
   s=state(owner)
   if s['done'] or not marker.exists():return execute0(owner,scheduler_output,*a,**k)
   work=scheduler_output.num_scheduled_tokens
   if len(work)!=64 or any(v!=1 for v in work.values()):s['warm']=0;return execute0(owner,scheduler_output,*a,**k)
   s['warm']+=1
   if s['warm']<=32:return execute0(owner,scheduler_output,*a,**k)
   assert s['active'] is None and owner.speculative_config is None and owner.input_batch.num_reqs==64
   computed=owner.input_batch.num_computed_tokens_cpu[:64];prompts=owner.input_batch.num_prompt_tokens[:64];assert min(computed)>=4096 and min(prompts)==max(prompts)==4096
   if s['profiler'] is None:
    s['profiler']=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False);s['profiler'].__enter__()
   entry=dict(index=len(s['frames']),min_computed=int(min(computed)),max_computed=int(max(computed)));s['active']=entry;return call(entry,'execute_model',execute0,owner,scheduler_output,*a,**k)
  def sample(owner,*a,**k):
   s=state(owner);entry=s['active']
   if entry is None:return sample0(owner,*a,**k)
   result=call(entry,'sample_tokens',sample0,owner,*a,**k);s['frames'].append(entry);s['active']=None
   if len(s['frames'])==16:finish(s)
   return result
  cls.execute_model,cls.sample_tokens,cls._determine_batch_execution_and_padding=execute,sample,dispatch
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname!='vllm_fl.worker.model_runner':return None
  s=importlib.machinery.PathFinder.find_spec(fullname,path);assert s and s.loader;s.loader=Loader(s.loader);return s
if os.environ.get('TIANSHU_GPU_PROFILE_PROBE')=='1':sys.meta_path.insert(0,Finder())
