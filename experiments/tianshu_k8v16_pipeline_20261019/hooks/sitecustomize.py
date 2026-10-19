# SPDX-License-Identifier: Apache-2.0
"""Compose qualified hooks, then attach byte storage after WorkerFL initialization."""
import functools
import importlib.abc
import importlib.util
from pathlib import Path
import runpy
import sys
import os

BASE = Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE.parent / 'tianshu_gemm_topp_geo_20261016/hooks/sitecustomize.py'))


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        original_init = module.WorkerFL.__init__

        @functools.wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            mode = os.environ['TIANSHU_K8V16_SERVICE_MODE']
            assert mode in ('control', 'candidate')
            target = BASE if mode == 'candidate' else BASE.parent / 'tianshu_soft8_engine_v2_20261019'
            sys.path.insert(0, str(target))
            spec = importlib.util.spec_from_file_location('_tianshu_selected_byte_engine', target / 'engine_hooks.py')
            helper = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = helper
            spec.loader.exec_module(helper)
            helper.install(self)

        module.WorkerFL.__init__ = init


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm_fl.worker.worker':
            return None
        index = sys.meta_path.index(self)
        sys.meta_path.pop(index)
        try:
            spec = None
            for finder in tuple(sys.meta_path):
                method = getattr(finder, 'find_spec', None)
                if method is not None:
                    spec = method(fullname, path, target)
                    if spec is not None:
                        break
        finally:
            sys.meta_path.insert(index, self)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


assert 'vllm_fl.worker.worker' not in sys.modules
sys.meta_path.insert(0, Finder())
