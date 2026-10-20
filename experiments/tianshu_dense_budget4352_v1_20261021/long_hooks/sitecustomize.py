# SPDX-License-Identifier: Apache-2.0
import functools,importlib.abc,importlib.util,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE.parent/'tianshu_k8v16_budget_default_20261019/hooks/sitecustomize.py'))
spec=importlib.util.spec_from_file_location('_tianshu_long_segments_service_policy',BASE/'long_policy.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        original_init=module.WorkerFL.__init__
        @functools.wraps(original_init)
        def init(owner,*args,**kwargs):
            original_init(owner,*args,**kwargs)
            policy.install(owner,os.environ['TIANSHU_LONG_SEGMENT_MODE'],Path(os.environ['TIANSHU_LONG_SEGMENT_RECORD_DIR']))
            import hashlib,json
            source=Path(policy.__file__).resolve();directory=Path(os.environ['TIANSHU_DENSE_BUDGET_RECORD_DIR']);directory.mkdir(parents=True,exist_ok=True)
            (directory/f'guard_worker_{os.getpid()}.json').write_text(json.dumps(dict(policy_source=str(source),policy_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),actual_budget=owner.vllm_config.scheduler_config.max_num_batched_tokens,actual_max_num_seqs=owner.vllm_config.scheduler_config.max_num_seqs,actual_long_mode=os.environ['TIANSHU_LONG_SEGMENT_MODE']),indent=2))
        module.WorkerFL.__init__=init
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm_fl.worker.worker':return None
        index=sys.meta_path.index(self);sys.meta_path.pop(index)
        try:
            spec=None
            for finder in tuple(sys.meta_path):
                method=getattr(finder,'find_spec',None)
                if method is not None:
                    spec=method(fullname,path,target)
                    if spec is not None:break
        finally:sys.meta_path.insert(index,self)
        assert spec is not None and spec.loader is not None
        spec.loader=Loader(spec.loader);return spec
assert 'vllm_fl.worker.worker' not in sys.modules
sys.meta_path.insert(0,Finder())
