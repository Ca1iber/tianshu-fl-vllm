# SPDX-License-Identifier: Apache-2.0
"""Current S16 + arena policy, native seq-default composition, read-only scheduler record."""
import functools,hashlib,importlib.abc,importlib.util,inspect,json,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE.parent/'tianshu_small_arena_service_20261020/hooks/sitecustomize.py'))
s=importlib.util.spec_from_file_location('_memory_combo_native_seq_policy',BASE.parent/'tianshu_seq_limit_default_20261020/policy.py')
policy=importlib.util.module_from_spec(s);s.loader.exec_module(policy)
class Loader(importlib.abc.Loader):
    def __init__(self,original,name):self.original=original;self.name=name
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        directory=Path(os.environ['TIANSHU_SEQ_LIMIT_RECORD_DIR'])
        if self.name=='vllm.engine.arg_utils':
            # Existing budget4096 selector has already been installed by the inner loader.
            policy.install(module,os.environ['TIANSHU_SEQ_LIMIT_MODE'],directory)
            return
        original=module.Scheduler.__init__
        @functools.wraps(original)
        def initialize(owner,*args,**kwargs):
            original(owner,*args,**kwargs)
            allocator=Path(inspect.getsourcefile(owner.kv_cache_manager.allocate_slots))
            row=dict(native_source=str(module.__file__),native_source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                class_name=type(owner).__name__,max_num_running_reqs=owner.max_num_running_reqs,
                config_max_num_seqs=owner.scheduler_config.max_num_seqs,max_num_scheduled_tokens=owner.max_num_scheduled_tokens,
                max_num_batched_tokens=owner.scheduler_config.max_num_batched_tokens,async_scheduling=owner.scheduler_config.async_scheduling,
                max_model_len=owner.max_model_len,KV_watermark_blocks=owner.kv_cache_manager.watermark_blocks,
                KV_num_blocks=owner.kv_cache_manager.kv_cache_config.num_blocks,
                allocate_slots_source=str(allocator),allocate_slots_source_sha256=hashlib.sha256(allocator.read_bytes()).hexdigest(),
                original_scheduler_and_allocation_methods_unchanged=True,arguments_forwarded=True,RNG_not_read_or_modified=True)
            directory.mkdir(parents=True,exist_ok=True)
            (directory/f'scheduler_{os.getpid()}.json').write_text(json.dumps(row,indent=2)+'\n')
        module.Scheduler.__init__=initialize
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in ('vllm.engine.arg_utils','vllm.v1.core.sched.scheduler'):return None
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
assert 'vllm.engine.arg_utils' not in sys.modules and 'vllm.v1.core.sched.scheduler' not in sys.modules
sys.meta_path.insert(0,Finder())
