# SPDX-License-Identifier: Apache-2.0
"""Keep 75 prefill warmups while bounding independent decode graph sizes."""
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / 'tianshu_buckets75_recheck_20261012/joint_hooks/sitecustomize.py'))


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        configure = module.configure

        def configure_bounded(config, platform, logger):
            result = configure(config, platform, logger)
            if not module.enabled(config, platform):
                return result
            limit = int(os.environ['TIANSHU_DECODE_GRAPH_MAX_ROWS'])
            assert limit == 64
            cfg = config.compilation_config
            before = list(cfg.cudagraph_capture_sizes)
            sizes = [size for size in before if size <= limit]
            assert sizes and max(sizes) == limit
            cfg.cudagraph_capture_sizes = sizes
            cfg.max_cudagraph_capture_size = limit
            root = Path(os.environ['TIANSHU_DECODE_GRAPH_RECORD_DIR'])
            root.mkdir(parents=True, exist_ok=True)
            (root / f'configured_{os.getpid()}.json').write_text(json.dumps(dict(
                original_decode_graph_sizes=before, decode_graph_sizes=sizes,
                prefill_warmup_sizes=list(module.buckets_for_budget(
                    config.scheduler_config.max_num_batched_tokens)),
                prefill_warmup_function_unmodified=True,
                mode=cfg.cudagraph_mode.name), indent=2) + '\n')
            logger.warning('Experimental decode graph cap=%d, independent prefill warmups remain75', limit)
            return result

        module.configure = configure_bounded


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm_fl.worker.iluvatar_prefill_buckets':
            return None
        # Preserve the existing loader that records all actual prefill warmups.
        for finder in tuple(sys.meta_path):
            if finder is self:
                continue
            method = getattr(finder, 'find_spec', None)
            spec = method(fullname, path, target) if method else None
            if spec is not None:
                assert spec.loader is not None
                spec.loader = Loader(spec.loader)
                return spec
        return None


if os.environ.get('TIANSHU_DECODE_GRAPH_MAX_ROWS'):
    assert 'vllm_fl.worker.iluvatar_prefill_buckets' not in sys.modules
    sys.meta_path.insert(0, Finder())
