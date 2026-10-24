# SPDX-License-Identifier: Apache-2.0
"""Wrap the complete r5 worker, then change only M64 quantized-KV reading."""
import functools,hashlib,importlib.abc,importlib.util,json,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE.parent/'tianshu_W4_QKVO64_fitted_service_20261024/hooks_candidate/sitecustomize.py'))
class Loader(importlib.abc.Loader):
    def __init__(self, original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        original=module.WorkerFL.__init__
        @functools.wraps(original)
        def init(owner,*args,**kwargs):
            value=original(owner,*args,**kwargs)
            mode=os.environ['TIANSHU_R5_BM8_MODE']
            assert mode in ('candidate','control') and os.environ['TIANSHU_QKVO64_FITTED_MODE']=='candidate'
            folder=Path(os.environ['TIANSHU_R5_BM8_RECORD_DIR'])
            if mode=='candidate':
                source=BASE.parent/'tianshu_k8v16_bm8_service_v3_20261021/policy.py'
                spec=importlib.util.spec_from_file_location('_r5_BM8_exact_old_policy',source)
                policy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=policy;spec.loader.exec_module(policy)
                policy.install(owner,'candidate',str(folder))
            proof=dict(mode=mode,parent_QKVO_mode='candidate',parent_full_r5=True,BM8_installed=(mode=='candidate'),
                       source=str(Path(__file__).resolve()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            target=folder.parent/'bm8_integration.json'
            target.write_text(json.dumps(proof,indent=2)+'\n')
            return value
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
        assert found is not None and found.loader is not None
        found.loader=Loader(found.loader)
        return found
assert 'vllm_fl.worker.worker' not in sys.modules
sys.meta_path.insert(0,Finder())
