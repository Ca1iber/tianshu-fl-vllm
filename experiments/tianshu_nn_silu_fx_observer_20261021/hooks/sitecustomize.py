# SPDX-License-Identifier: Apache-2.0
import functools,importlib.abc,importlib.util,json,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
runpy.run_path(str(BASE.parent/'tianshu_prefill_int8_service_20261021/hooks_candidate/sitecustomize.py'))
class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module);original=module.WorkerFL.__init__
        @functools.wraps(original)
        def init(owner,*args,**kwargs):
            value=original(owner,*args,**kwargs)
            from observer import Observer
            cc=owner.vllm_config.compilation_config
            root=Path(os.environ['TIANSHU_DENSE_INT8_RECORD_DIR']).parent/'fx_observer'
            assert 'post_grad_custom_post_pass' not in cc.inductor_compile_config
            original_sizes=list(cc.compile_sizes or [])
            assert all(isinstance(n,int) for n in original_sizes)
            cc.compile_sizes=sorted(set(original_sizes)|{64})
            cc.inductor_compile_config['post_grad_custom_post_pass']=Observer(root)
            (root/'installed.json').write_text(json.dumps(dict(original_compile_sizes=original_sizes,compile_sizes=cc.compile_sizes,read_only_pass=True,original_pre_grad_pass_retained=True,model_numerics_replacement=False),indent=2)+'\n')
            return value
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
