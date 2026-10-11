# SPDX-License-Identifier: Apache-2.0
"""Explicit, process-only FA2 heuristic override for a single-model trial."""
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys


class TileLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        from flag_gems import runtime
        original = runtime.get_heuristic_config
        recorded = False

        def lookup(name):
            nonlocal recorded
            if name != 'mha_block_128':
                return original(name)
            config = dict(BLOCK_M=128, BLOCK_N=128, num_warps=8, num_stages=2)
            directory = os.environ.get('TIANSHU_FA2_TILE_RECORD_DIR')
            if directory and not recorded:
                recorded = True
                output = Path(directory)
                output.mkdir(parents=True, exist_ok=True)
                (output / f'tile_{os.getpid()}.json').write_text(json.dumps(
                    {'pid': os.getpid(), 'config_key': name, 'configuration': config,
                     'module': module.__name__, 'module_file': module.__file__,
                     'process_only_override': True}, indent=2) + '\n')
            return {key: (lambda args, value=value: value) for key, value in config.items()}

        runtime.get_heuristic_config = lookup


class TileFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'flag_gems.ops.flash_api':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise RuntimeError('FA2 API module unavailable for explicit tile experiment')
        spec.loader = TileLoader(spec.loader)
        return spec


if os.environ.get('TIANSHU_FA2_TILE_PROBE') == '1':
    sys.meta_path.insert(0, TileFinder())
