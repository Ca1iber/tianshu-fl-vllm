# SPDX-License-Identifier: Apache-2.0
"""Preserve 75 Prefill warmups; remove Decode sizes24/40/48/56 only."""
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / 'tianshu_kv16_segments_20261014/hooks/sitecustomize.py'))


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
            cfg = config.compilation_config
            before = list(cfg.cudagraph_capture_sizes)
            removed = [24,40,48,56] if os.environ.get('TIANSHU_DECODE_PADDING_PROBE') == '1' else []
            assert all(size in before for size in removed)
            sizes = [size for size in before if size not in removed]
            assert len(before)==75 and len(sizes)==75-len(removed) and max(sizes)==max(before)==2048
            cfg.cudagraph_capture_sizes = sizes
            assert cfg.max_cudagraph_capture_size == 2048
            root = Path(os.environ['TIANSHU_DECODE_PADDING_RECORD_DIR'])
            root.mkdir(parents=True, exist_ok=True)
            (root / f'configured_{os.getpid()}.json').write_text(json.dumps(dict(
                original_decode_graph_sizes=before, decode_graph_sizes=sizes, removed_decode_sizes=removed,
                padding_map={str(n):next(x for x in sizes if x>=n) for n in range(1,65)},
                prefill_warmup_sizes=list(module.buckets_for_budget(
                    config.scheduler_config.max_num_batched_tokens)),
                prefill_warmup_function_unmodified=True,
                mode=cfg.cudagraph_mode.name), indent=2) + '\n')
            logger.warning('Experimental Decode padding removed=%s; Prefill warmups remain75', removed)
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


class DispatchLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        expected = '51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63'
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected
        original = module.CudagraphDispatcher.dispatch
        root = Path(os.environ['TIANSHU_DECODE_PADDING_RECORD_DIR'])
        marker = Path(os.environ['TIANSHU_DECODE_PADDING_REQUEST_MARKER'])
        seen = set()

        @functools.wraps(original)
        def dispatch(self, num_tokens, uniform_decode=False, *args, **kwargs):
            result = original(self, num_tokens, uniform_decode, *args, **kwargs)
            mode, descriptor = result
            if uniform_decode:
                key = num_tokens, mode.name, descriptor.num_tokens
                if key not in seen and marker.exists():
                    seen.add(key)
                    root.mkdir(parents=True, exist_ok=True)
                    (root / f'dispatch_{os.getpid()}_{len(seen)}.json').write_text(json.dumps(dict(
                        actual_tokens=num_tokens, padded_tokens=descriptor.num_tokens,
                        mode=mode.name, uniform_decode=True, throughput_phase_marker_present=True,
                        dispatcher_source_sha256=expected), indent=2) + '\n')
            return result

        module.CudagraphDispatcher.dispatch = dispatch


class DispatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm.v1.cudagraph_dispatcher':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = DispatchLoader(spec.loader)
        return spec


if os.environ.get('TIANSHU_DECODE_PADDING_TRACE') == '1':
    assert os.environ.get('TIANSHU_KV16_SERVICE_MODE') == 'candidate'
    assert os.environ.get('TIANSHU_SEGMENT_POLICY_PROBE') == '1'
    assert 'vllm_fl.worker.iluvatar_prefill_buckets' not in sys.modules
    sys.meta_path.insert(0, Finder())
    assert 'vllm.v1.cudagraph_dispatcher' not in sys.modules
    sys.meta_path.insert(0, DispatchFinder())
