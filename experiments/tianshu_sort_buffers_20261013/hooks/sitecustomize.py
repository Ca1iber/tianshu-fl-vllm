# SPDX-License-Identifier: Apache-2.0
"""Process-only private radix-buffer variants with public int64 indices."""
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / 'tianshu_buckets75_recheck_20261012/joint_hooks/sitecustomize.py'))


def install(module):
    mode = os.environ['TIANSHU_SORT_BUFFER_MODE']
    assert mode in ('index32', 'scan32', 'both32')
    helper = Path('/workspace/FlagGems/experiments/tianshu_sort_buffers_20261013/buffer_variants.py')
    exports = runpy.run_path(str(helper))
    root = Path(os.environ['TIANSHU_SORT_BUFFER_RECORD_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    variants, sources = exports['create_variants'](module, root / f'generated_{os.getpid()}')
    selected = variants[mode]
    seen = set()

    def radix(arr, k_bits=8, descending=False):
        values, indices = selected(arr, k_bits, descending)
        assert str(indices.dtype) == 'torch.int64'
        key = (arr.numel() // arr.shape[-1], arr.shape[-1], str(arr.dtype), descending)
        if key not in seen:
            seen.add(key)
            (root / f'route_{os.getpid()}_{len(seen)}.json').write_text(json.dumps(dict(
                mode=mode, m=key[0], n=key[1], value_dtype=key[2], descending=key[3],
                public_index_dtype=str(indices.dtype), source=sources[mode]), indent=2) + '\n')
        return values, indices

    module.radix_sort = radix


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


if os.environ.get('TIANSHU_SORT_BUFFER_MODE'):
    if 'flag_gems.ops.sort' in sys.modules:
        install(sys.modules['flag_gems.ops.sort'])
    else:
        sys.meta_path.insert(0, Finder())
