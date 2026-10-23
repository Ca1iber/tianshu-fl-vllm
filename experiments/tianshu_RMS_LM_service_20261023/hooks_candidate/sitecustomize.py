# SPDX-License-Identifier: Apache-2.0
import functools,importlib.abc,importlib.util,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];MODE=Path(__file__).resolve().parent.name.removeprefix('hooks_');assert MODE in ('candidate','control')
runpy.run_path(str(BASE.parent/'tianshu_RMS_A8_GU_service_v2_20261023'/f'hooks_{MODE}'/'sitecustomize.py'))
s=importlib.util.spec_from_file_location('_RMS_LM_combo_original_LM_policy',BASE.parent/'tianshu_LM_pad64_service_20261022/policy.py');lm=importlib.util.module_from_spec(s);sys.modules[s.name]=lm;s.loader.exec_module(lm)
class Loader(importlib.abc.Loader):
 def __init__(self,original):self.original=original
 def create_module(self,spec):return self.original.create_module(spec)
 def exec_module(self,module):
  self.original.exec_module(module);original=module.WorkerFL.__init__
  @functools.wraps(original)
  def init(owner,*args,**kwargs):
   result=original(owner,*args,**kwargs);lm.install(owner,MODE,Path(os.environ['TIANSHU_DENSE_INT8_RECORD_DIR']).parent/'lm_pad64_records');return result
  module.WorkerFL.__init__=init
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname!='vllm_fl.worker.worker':return None
  index=sys.meta_path.index(self);sys.meta_path.pop(index)
  try:
   found=None
   for finder in tuple(sys.meta_path):
    fn=getattr(finder,'find_spec',None)
    if fn is not None:
     found=fn(fullname,path,target)
     if found is not None:break
  finally:sys.meta_path.insert(index,self)
  assert found is not None and found.loader is not None;found.loader=Loader(found.loader);return found
assert 'vllm_fl.worker.worker' not in sys.modules;sys.meta_path.insert(0,Finder())
