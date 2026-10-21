# SPDX-License-Identifier: Apache-2.0
import functools,importlib.abc,importlib.util,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];MODE=Path(__file__).resolve().parent.name.removeprefix('hooks_');assert MODE in ('control','candidate')
runpy.run_path(str(BASE.parent/'tianshu_prefill_int8_service_20261021/hooks_candidate/sitecustomize.py'))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
policy=load('_static_compile_default_policy',BASE/'policy.py');trace=load('_static_compile_native_range_trace',BASE/'trace.py')
DIRECTORY=Path(os.environ['TIANSHU_DENSE_INT8_RECORD_DIR']).parent/'static_compile_records'
class Loader(importlib.abc.Loader):
    def __init__(self,original,target):self.original,self.target=original,target
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        if self.target=='vllm.engine.arg_utils':policy.install(module,MODE,DIRECTORY)
        else:
            original=module.WorkerFL.__init__
            @functools.wraps(original)
            def init(owner,*args,**kwargs):
                answer=original(owner,*args,**kwargs);assert answer is None
                trace.install(owner,MODE,DIRECTORY);return answer
            module.WorkerFL.__init__=init
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in ('vllm.engine.arg_utils','vllm_fl.worker.worker'):return None
        index=sys.meta_path.index(self);sys.meta_path.pop(index)
        try:
            spec=None
            for finder in tuple(sys.meta_path):
                method=getattr(finder,'find_spec',None)
                if method is not None:
                    spec=method(fullname,path,target)
                    if spec is not None:break
        finally:sys.meta_path.insert(index,self)
        assert spec is not None and spec.loader is not None;spec.loader=Loader(spec.loader,fullname);return spec
assert 'vllm.engine.arg_utils' not in sys.modules and 'vllm_fl.worker.worker' not in sys.modules
sys.meta_path.insert(0,Finder())
