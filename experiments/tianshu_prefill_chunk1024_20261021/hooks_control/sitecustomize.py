# SPDX-License-Identifier: Apache-2.0
import importlib.abc,importlib.util,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
MODE=Path(__file__).resolve().parent.name.removeprefix('hooks_');assert MODE in ('control','candidate')
runpy.run_path(str(BASE.parent/'tianshu_sampler_exp_service_20261021/hooks_candidate/sitecustomize.py'))
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('_prefill_chunk_policy',BASE/'policy.py')
policy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=policy;spec.loader.exec_module(policy)
DIRECTORY=Path(os.environ['TIANSHU_DENSE_INT8_RECORD_DIR']).parent/'prefill_chunk_records'
TARGETS={'vllm.engine.arg_utils':policy.install_defaults,'vllm.v1.core.sched.scheduler':policy.install_scheduler}
class Loader(importlib.abc.Loader):
    def __init__(self,original,install):self.original=original;self.install=install
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):self.original.exec_module(module);self.install(module,MODE,DIRECTORY)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in TARGETS:return None
        index=sys.meta_path.index(self);sys.meta_path.pop(index)
        try:
            spec=None
            for finder in tuple(sys.meta_path):
                method=getattr(finder,'find_spec',None)
                if method is not None:
                    spec=method(fullname,path,target)
                    if spec is not None:break
        finally:sys.meta_path.insert(index,self)
        assert spec is not None and spec.loader is not None;spec.loader=Loader(spec.loader,TARGETS[fullname]);return spec
for name,install in TARGETS.items():
    if name in sys.modules:install(sys.modules[name],MODE,DIRECTORY)
sys.meta_path.insert(0,Finder())
