# SPDX-License-Identifier: Apache-2.0
import functools,importlib.abc,importlib.util,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
MODE=Path(__file__).resolve().parent.name.removeprefix('hooks_');assert MODE in ('control','candidate')
runpy.run_path(str(BASE.parent/'tianshu_lm_precision_service_20261020/hooks/sitecustomize.py'))
spec=importlib.util.spec_from_file_location('_kv_empty_service_policy',BASE/'policy.py')
policy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=policy;spec.loader.exec_module(policy)
class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module);original=module.Scheduler.__init__
        @functools.wraps(original)
        def init(owner,*args,**kwargs):
            answer=original(owner,*args,**kwargs)
            policy.install(owner,MODE,Path(os.environ['TIANSHU_FUTURE_KV_RECORD_DIR']).parent/'kv_empty_allocate_records')
            return answer
        module.Scheduler.__init__=init
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm.v1.core.sched.scheduler':return None
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
assert 'vllm.v1.core.sched.scheduler' not in sys.modules
sys.meta_path.insert(0,Finder())
