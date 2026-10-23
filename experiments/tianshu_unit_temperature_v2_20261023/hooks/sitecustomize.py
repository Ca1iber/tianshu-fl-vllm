# SPDX-License-Identifier: Apache-2.0
import importlib.abc,importlib.machinery,importlib.util,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE.parent/'tianshu_W4_LMpad_service_v2_20261023/hooks_candidate/sitecustomize.py'))
class Loader(importlib.abc.Loader):
 def __init__(self,original):self.original=original
 def create_module(self,spec):return self.original.create_module(spec)
 def exec_module(self,module):
  self.original.exec_module(module)
  s=importlib.util.spec_from_file_location('_tianshu_actual_unit_temperature',BASE/'policy.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);m.install(os.environ['TIANSHU_UNIT_TEMP_MODE'],os.environ['TIANSHU_UNIT_TEMP_RECORD_DIR'])
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname!='vllm_fl.worker.model_runner':return None
  # Delegate through the existing r4 finders rather than bypassing them.
  pos=sys.meta_path.index(self)
  for finder in sys.meta_path[pos+1:]:
   method=getattr(finder,'find_spec',None)
   if method is None:continue
   spec=method(fullname,path,target)
   if spec is not None:assert spec.loader;spec.loader=Loader(spec.loader);return spec
  raise ImportError(fullname)
if os.environ.get('TIANSHU_UNIT_TEMP_MODE') in ('control','candidate'):sys.meta_path.insert(0,Finder())
