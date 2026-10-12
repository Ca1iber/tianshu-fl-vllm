# SPDX-License-Identifier: Apache-2.0
"""Compose the bucket records with actual fixed-gather route evidence."""
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / 'tianshu_buckets75_recheck_20261012/hooks/sitecustomize.py'))


class GatherLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        original = module.gather_rows
        seen = set()

        def gather(hidden, indices):
            result = original(hidden, indices)
            key = (hidden.shape[0], indices.numel(), str(indices.dtype))
            if key not in seen:
                seen.add(key)
                root = Path(os.environ['TIANSHU_BUCKET_TRACE_DIR'])
                root.mkdir(parents=True, exist_ok=True)
                (root / f'gather_{os.getpid()}_{len(seen)}.json').write_text(json.dumps(dict(
                    source_rows=key[0], rows=key[1], index_dtype=key[2],
                    original_gather_function_unchanged=True, autotune=False), indent=2) + '\n')
            return result

        module.gather_rows = gather


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm_fl.worker.iluvatar_logits_gather':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = GatherLoader(spec.loader)
        return spec


if os.getenv('TIANSHU_BUCKET_RECHECK_TRACE') == '1':
    sys.meta_path.insert(0, Finder())
