# SPDX-License-Identifier: Apache-2.0
"""Compose the verified attention profile with process-only sort counters."""
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
BASE = EXPERIMENT.parent
runpy.run_path(str(BASE / 'tianshu_buckets75_recheck_20261012/joint_hooks/sitecustomize.py'))


def install(module):
    requested = int(os.environ['TIANSHU_SORT_COUNT_BITS'])
    assert requested in (8, 16, 32, 64)
    if getattr(module, '_tianshu_count_router_installed', False):
        return
    helpers = runpy.run_path(str(EXPERIMENT / 'histogram_width.py'))
    root = Path(os.environ['TIANSHU_SORT_COUNT_RECORD_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    kernels, sources = helpers['create_kernels'](module, root / f'generated_{os.getpid()}')
    (root / f'sources_{os.getpid()}.json').write_text(json.dumps(sources, indent=2) + '\n')
    seen = set()

    def record(info):
        key = tuple(sorted(info.items()))
        if key in seen:
            return
        seen.add(key)
        info = dict(info, source=sources[info['actual_bits']])
        (root / f'route_{os.getpid()}_{len(seen)}.json').write_text(json.dumps(info, indent=2) + '\n')

    module.compute_global_hist_kernel = helpers['HistogramRouter'](
        module.compute_global_hist_kernel, kernels, requested, record)
    module._tianshu_count_router_installed = True


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        install(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'flag_gems.ops.sort':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


if os.environ.get('TIANSHU_SORT_COUNT_BITS'):
    if 'flag_gems.ops.sort' in sys.modules:
        install(sys.modules['flag_gems.ops.sort'])
    else:
        sys.meta_path.insert(0, Finder())
