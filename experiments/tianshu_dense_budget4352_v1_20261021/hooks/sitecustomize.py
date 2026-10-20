# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.abc,importlib.util,json,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
ORIGINAL_RUN=runpy.run_path
TARGET=(BASE.parent/'tianshu_long_segments_service_20261020/hooks/sitecustomize.py').resolve()
REPLACEMENTS=[]
def route_run(path,*args,**kwargs):
    if Path(path).resolve()==TARGET:
        REPLACEMENTS.append(str(TARGET));path=str(BASE/'long_hooks/sitecustomize.py')
    return ORIGINAL_RUN(path,*args,**kwargs)
runpy.run_path=route_run
try:ORIGINAL_RUN(str(BASE.parent/'tianshu_lm_precision_service_20261020/hooks/sitecustomize.py'))
finally:runpy.run_path=ORIGINAL_RUN
assert REPLACEMENTS==[str(TARGET)]
directory=Path(os.environ['TIANSHU_DENSE_BUDGET_RECORD_DIR']);directory.mkdir(parents=True,exist_ok=True)
paths=[TARGET,BASE/'long_hooks/sitecustomize.py',BASE.parent/'tianshu_long_segments_service_20261020/policy.py',BASE/'long_policy.py']
(directory/f'hook_route_{os.getpid()}.json').write_text(json.dumps(dict(exact_one_hook_redirect=True,runpy_restored=runpy.run_path is ORIGINAL_RUN,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))
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
ORIGINAL_RUN(str(BASE/'sampler_hook.py'))
