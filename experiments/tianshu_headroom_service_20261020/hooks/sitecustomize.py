# SPDX-License-Identifier: Apache-2.0
"""Compose the existing long-segment hook exactly once, then native KV watermark."""
import functools,hashlib,importlib.abc,importlib.util,inspect,json,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE.parent/'tianshu_long_segments_service_20261020/hooks/sitecustomize.py'))
s=importlib.util.spec_from_file_location('_tianshu_service_kv_headroom_policy',BASE.parent/'tianshu_kv_headroom_20261020/policy.py')
policy=importlib.util.module_from_spec(s);s.loader.exec_module(policy)
class Loader(importlib.abc.Loader):
    def __init__(self,original,fullname):self.original=original;self.fullname=fullname
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        directory=Path(os.environ['TIANSHU_KV_HEADROOM_RECORD_DIR'])
        if self.fullname=='vllm.engine.arg_utils':
            policy.install(module,os.environ['TIANSHU_KV_HEADROOM_MODE'],directory)
        else:
            cls=module.KVCacheManager;original=cls.__init__;signature=inspect.signature(original)
            @functools.wraps(original)
            def init(owner,*args,**kwargs):
                bound=signature.bind(owner,*args,**kwargs);bound.apply_defaults()
                original(owner,*args,**kwargs)
                directory.mkdir(parents=True,exist_ok=True)
                row=dict(source=str(module.__file__),source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                         watermark_argument=bound.arguments['watermark'],watermark_blocks=owner.watermark_blocks,
                         num_blocks=owner.kv_cache_config.num_blocks,num_free_blocks=owner.block_pool.get_num_free_blocks(),
                         max_num_batched_tokens=bound.arguments['max_num_batched_tokens'],max_model_len=owner.max_model_len,
                         native_allocate_slots_unmodified=True,arguments_and_return_values_forwarded=True,RNG_not_read_or_modified=True)
                (directory/f'manager_{os.getpid()}.json').write_text(json.dumps(row,indent=2)+'\n')
            cls.__init__=init
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in ('vllm.engine.arg_utils','vllm.v1.core.kv_cache_manager'):return None
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
        spec.loader=Loader(spec.loader,fullname);return spec
assert 'vllm.engine.arg_utils' not in sys.modules and 'vllm.v1.core.kv_cache_manager' not in sys.modules
sys.meta_path.insert(0,Finder())
