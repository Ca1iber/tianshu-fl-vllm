# SPDX-License-Identifier: Apache-2.0
import functools,hashlib,importlib.abc,importlib.util,json,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
MODE=Path(__file__).resolve().parent.name.removeprefix('hooks_');assert MODE in ('control','candidate')
runpy.run_path(str(BASE.parent/'tianshu_prefill_int8_service_20261021/hooks_candidate/sitecustomize.py'))
spec=importlib.util.spec_from_file_location('_current_grid_service_policy',BASE/'policy.py')
policy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=policy;spec.loader.exec_module(policy)
class Loader(importlib.abc.Loader):
 def __init__(self,original,kind):self.original,self.kind=original,kind
 def create_module(self,spec):return self.original.create_module(spec)
 def exec_module(self,module):
  self.original.exec_module(module)
  if self.kind=='worker':
   original=module.WorkerFL.__init__
   @functools.wraps(original)
   def init(owner,*args,**kwargs):
    answer=original(owner,*args,**kwargs);assert answer is None
    policy.install(owner,MODE,Path(os.environ['TIANSHU_DENSE_INT8_RECORD_DIR']).parent/'topp_grid_service_records')
    return answer
   module.WorkerFL.__init__=init
  else:
   geo=sys.modules['_tianshu_topp_geometric_geo']
   spec=importlib.util.spec_from_file_location('_current_grid_service_route',BASE/'grid_route.py')
   helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
   root=Path(os.environ['TIANSHU_DENSE_INT8_RECORD_DIR']).parent/'topp_grid_service_records';root.mkdir(parents=True,exist_ok=True)
   observed=dict(python_calls=0,routes={},scope='Python launches, not Graph replay counts')
   def flush():
    target=root/f'calls_{os.getpid()}.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(observed,indent=2)+'\n');tmp.replace(target)
   def observe(row):
    state=policy.STATE
    if state.get('capture_returned') and not state.get('ready'):state['ready']=(root.parent/'observed_server_environment.json').exists()
    stage='runtime' if state.get('ready') else state.get('stage','before_install')
    key=str((stage,row['batch'],row['topk'],row['topp'],row['active'],row['input_row_stride']))
    fresh=key not in observed['routes'];observed['python_calls']+=1
    if fresh:observed['routes'][key]=dict(row,stage=stage,python_calls=0)
    observed['routes'][key]['python_calls']+=1
    if fresh or observed['python_calls']%256==0:flush()
   helper.install(module,geo,False,observe)
   module._tianshu_grid_observed=observed;module._tianshu_grid_flush=flush
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  kinds={'vllm_fl.worker.worker':'worker','vllm.v1.sample.ops.topk_topp_triton':'sampler'}
  if fullname not in kinds:return None
  index=sys.meta_path.index(self);sys.meta_path.pop(index)
  try:
   spec=None
   for finder in tuple(sys.meta_path):
    method=getattr(finder,'find_spec',None)
    if method is not None:
     spec=method(fullname,path,target)
     if spec is not None:break
  finally:sys.meta_path.insert(index,self)
  assert spec is not None and spec.loader is not None;spec.loader=Loader(spec.loader,kinds[fullname]);return spec
assert 'vllm_fl.worker.worker' not in sys.modules and 'vllm.v1.sample.ops.topk_topp_triton' not in sys.modules
sys.meta_path.insert(0,Finder())
