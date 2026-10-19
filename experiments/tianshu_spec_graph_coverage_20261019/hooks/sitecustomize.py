# SPDX-License-Identifier: Apache-2.0
"""Keep the validated native ngram stack; add request-scoped host observation."""
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE.parent / 'tianshu_ngram_fixed_index_20261016/hooks/sitecustomize.py'))
RUNNER_SHA = 'b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'
GRAPH_SHA = 'ba78ed0a736bb21a4f2c307fc9775f15e3742b94533eff333b0c364eb8213fc2'


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == RUNNER_SHA
        graph_module = sys.modules['vllm_fl.compilation.graph']
        assert hashlib.sha256(Path(graph_module.__file__).read_bytes()).hexdigest() == GRAPH_SHA
        spec = importlib.util.spec_from_file_location('_tianshu_spec_graph_observer', BASE / 'route_observer.py')
        observer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = observer
        spec.loader.exec_module(observer)
        from vllm.forward_context import get_forward_context, is_forward_context_available

        def context():
            return get_forward_context() if is_forward_context_available() else None

        directory = Path(os.environ['TIANSHU_SPEC_COVERAGE_RECORD_DIR'])
        marker = Path(os.environ['TIANSHU_SPEC_COVERAGE_MARKER'])
        observer.install(module.ModelRunnerFL, graph_module.GraphWrapper, context, directory, marker)
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(before_profile_capture_and_requests=True, runner_source_sha256=RUNNER_SHA,
                      graph_source_sha256=GRAPH_SHA, hook_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      observer_source_sha256=hashlib.sha256((BASE / 'route_observer.py').read_bytes()).hexdigest(),
                      source_scope='Actual FL GraphWrapper; installed sources and dispatch behavior unchanged',
                      GPU_events_or_synchronizations_added=False)
        (directory / f'installed_{os.getpid()}.json').write_text(json.dumps(record, indent=2) + '\n')


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm_fl.worker.model_runner':
            return None
        position = sys.meta_path.index(self)
        sys.meta_path.pop(position)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(position, self)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


sys.meta_path.insert(0, Finder())
