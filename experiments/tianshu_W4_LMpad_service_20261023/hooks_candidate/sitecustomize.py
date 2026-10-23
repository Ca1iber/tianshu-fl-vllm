# SPDX-License-Identifier: Apache-2.0
import functools,importlib.abc,importlib.util,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];runpy.run_path(str(BASE.parent/'tianshu_dotsaddz_Down64_service_v2_20261023/hooks_candidate/sitecustomize.py'))
spec=importlib.util.spec_from_file_location('_tianshu_LMpad_live',BASE/'policy.py');policy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=policy;spec.loader.exec_module(policy)
class Loader(importlib.abc.Loader):
 def __init__(self,original):self.original=original
 def create_module(self,spec):return self.original.create_module(spec)
 def exec_module(self,module):
  self.original.exec_module(module);original=module.WorkerFL.__init__
  @functools.wraps(original)
  def init(owner,*args,**kwargs):
   r=original(owner,*args,**kwargs);policy.install(owner,Path(os.environ['TIANSHU_DENSE_INT8_RECORD_DIR']).parent/'dots_LM64_records');return r
  module.WorkerFL.__init__=init
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname!='vllm_fl.worker.worker':return None
  index=sys.meta_path.index(self);sys.meta_path.pop(index)
  try:
   found=None
   for finder in tuple(sys.meta_path):
    method=getattr(finder,'find_spec',None)
    if method is not None:
     found=method(fullname,path,target)
     if found is not None:break
  finally:sys.meta_path.insert(index,self)
  assert found is not None and found.loader is not None;found.loader=Loader(found.loader);return found
assert 'vllm_fl.worker.worker' not in sys.modules;sys.meta_path.insert(0,Finder())
