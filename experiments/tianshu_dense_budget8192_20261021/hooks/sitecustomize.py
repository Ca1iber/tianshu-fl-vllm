# SPDX-License-Identifier: Apache-2.0
import importlib.abc,importlib.util,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE.parent/'tianshu_lm_precision_service_20261020/hooks/sitecustomize.py'))
spec=importlib.util.spec_from_file_location('_tianshu_dense_budget_policy',BASE/'policy.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        policy.install(module,os.environ['TIANSHU_DENSE_BUDGET_MODE'],Path(os.environ['TIANSHU_DENSE_BUDGET_RECORD_DIR']))
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm.engine.arg_utils':return None
        index=sys.meta_path.index(self);sys.meta_path.pop(index)
        try:
            spec=None
            for finder in tuple(sys.meta_path):
                method=getattr(finder,'find_spec',None)
                if method is not None:
                    spec=method(fullname,path,target)
                    if spec is not None:break
        finally:sys.meta_path.insert(index,self)
        assert spec is not None and spec.loader is not None;spec.loader=Loader(spec.loader);return spec
assert 'vllm.engine.arg_utils' not in sys.modules
sys.meta_path.insert(0,Finder())
