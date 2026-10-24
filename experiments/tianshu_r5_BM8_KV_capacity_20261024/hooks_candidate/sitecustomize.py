# SPDX-License-Identifier: Apache-2.0
import importlib.abc,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        import importlib.util
        spec=importlib.util.spec_from_file_location('_BM8_capacity_policy',BASE/'capacity.py')
        policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
        assert os.environ['TIANSHU_KV_CAPACITY_QUANTUM']=='256'
        policy.install(module,os.environ['TIANSHU_KV_CAPACITY_RECORD_DIR'])
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm.v1.core.kv_cache_utils':return None
        index=sys.meta_path.index(self);sys.meta_path.pop(index)
        try:
            found=None
            for finder in tuple(sys.meta_path):
                method=getattr(finder,'find_spec',None)
                if method is not None:
                    found=method(fullname,path,target)
                    if found is not None:break
        finally:sys.meta_path.insert(index,self)
        assert found is not None and found.loader is not None
        found.loader=Loader(found.loader)
        return found
assert 'vllm.v1.core.kv_cache_utils' not in sys.modules
sys.meta_path.insert(0,Finder())
runpy.run_path(str(BASE.parent/'tianshu_r5_BM8_KV_20261024/hooks_candidate/sitecustomize.py'))
