# SPDX-License-Identifier: Apache-2.0
"""Same native ngram operators; only candidate routes to qualified mixed graphs."""
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
        spec = importlib.util.spec_from_file_location('_tianshu_spec_graph_observer', BASE.parent / 'tianshu_spec_graph_coverage_20261019/route_observer.py')
        observer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = observer
        spec.loader.exec_module(observer)
        from vllm.forward_context import get_forward_context, is_forward_context_available

        def context():
            return get_forward_context() if is_forward_context_available() else None

        directory = Path(os.environ['TIANSHU_RAGGED_RECORD_DIR'])
        directory.mkdir(parents=True, exist_ok=True)
        import vllm.forward_context as fc
        import vllm.v1.cudagraph_dispatcher as dispatcher
        from vllm.config import CUDAGraphMode
        assert hashlib.sha256(Path(fc.__file__).read_bytes()).hexdigest() == '1c6fa51e9a02a120fed5e7357afb1190516cbdfefd1c644dfcd860603f1527c5'
        assert hashlib.sha256(Path(dispatcher.__file__).read_bytes()).hexdigest() == '51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63'
        def load(name, filename):
            spec = importlib.util.spec_from_file_location(name, BASE.parent / 'tianshu_spec_exact_requests_20261019' / filename)
            value = importlib.util.module_from_spec(spec); sys.modules[spec.name] = value
            spec.loader.exec_module(value)
            return value
        def record(value):
            target = directory / f'policy_{os.getpid()}.json'; tmp = target.with_suffix('.tmp')
            tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(target)
        service_mode = os.environ['TIANSHU_RAGGED_SERVICE_MODE']
        assert service_mode in ('control','candidate')
        if service_mode == 'candidate':
            policy = load('_tianshu_ragged_service_policy', 'ragged_policy.py')
            policy.install(module, CUDAGraphMode, fc.BatchDescriptor, directory/'generated'/str(os.getpid()), record=record)
        coverage = directory/'coverage'
        observer.install(module.ModelRunnerFL, graph_module.GraphWrapper, context,
                         coverage, directory.parent/'coverage_phase.txt')
        coverage.mkdir(parents=True, exist_ok=True)
        installed_record = dict(before_profile_capture_and_requests=True, runner_source_sha256=RUNNER_SHA,
                      graph_source_sha256=GRAPH_SHA, hook_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      observer_source_sha256=hashlib.sha256((BASE.parent/'tianshu_spec_graph_coverage_20261019/route_observer.py').read_bytes()).hexdigest(),
                      policy_source_sha256=hashlib.sha256((BASE.parent/'tianshu_spec_exact_requests_20261019/ragged_policy.py').read_bytes()).hexdigest(),
                      source_scope='Owned mixed graphs; native keys and sampling unchanged',
                      service_mode=service_mode, numeric_callbacks_installed=False)
        (coverage / f'installed_{os.getpid()}.json').write_text(json.dumps(installed_record, indent=2) + '\n')



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
