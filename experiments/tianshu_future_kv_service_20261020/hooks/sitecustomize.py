# SPDX-License-Identifier: Apache-2.0
import functools
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE.parent / 'tianshu_memory_combo_service_20261020/hooks/sitecustomize.py'))
spec = importlib.util.spec_from_file_location('_future_kv_policy', BASE.parent / 'tianshu_future_kv_reservation_20261020/policy.py')
policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)


class Loader(importlib.abc.Loader):
    def __init__(self, original): self.original = original
    def create_module(self, spec): return self.original.create_module(spec)
    def exec_module(self, module):
        self.original.exec_module(module)
        original = module.Scheduler.__init__
        @functools.wraps(original)
        def initialize(owner, *args, **kwargs):
            original(owner, *args, **kwargs)
            controller = policy.install(owner, os.environ['TIANSHU_FUTURE_KV_MODE'], Path(os.environ['TIANSHU_FUTURE_KV_RECORD_DIR']))
            owner._tianshu_future_kv_reservation = controller
        module.Scheduler.__init__ = initialize


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm.v1.core.sched.scheduler': return None
        index = sys.meta_path.index(self); sys.meta_path.pop(index)
        try:
            spec = None
            for finder in tuple(sys.meta_path):
                method = getattr(finder, 'find_spec', None)
                if method is not None:
                    spec = method(fullname, path, target)
                    if spec is not None: break
        finally:
            sys.meta_path.insert(index, self)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


assert 'vllm.v1.core.sched.scheduler' not in sys.modules
sys.meta_path.insert(0, Finder())
