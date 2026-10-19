# SPDX-License-Identifier: Apache-2.0
"""Keep qualified K8V16 behavior and observe graph capture/replay plus native producer metadata."""
import importlib.abc
import importlib.util
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_tianshu_k8v16_graph_trace', BASE/'trace.py')
trace = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trace
spec.loader.exec_module(trace)
runpy.run_path(str(BASE.parent/'tianshu_k8v16_pipeline_20261019/hooks/sitecustomize.py'))
TARGETS = {'vllm_fl.compilation.graph': trace.install_graph,
    'vllm_fl.worker.model_runner': trace.install_runner, 'legacy_routing': trace.install_routing}


class Loader(importlib.abc.Loader):
    def __init__(self, original, name):
        self.original, self.name = original, name

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        TARGETS[self.name](module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
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
        spec.loader = Loader(spec.loader, fullname)
        return spec


assert all(name not in sys.modules for name in TARGETS)
sys.meta_path.insert(0, Finder())
