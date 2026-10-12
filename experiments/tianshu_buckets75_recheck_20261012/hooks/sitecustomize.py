# SPDX-License-Identifier: Apache-2.0
"""Record real bucket configuration/warmup/padding without altering arithmetic."""
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys
import time

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / 'tianshu_decode_launch_20261012/hooks/sitecustomize.py'))


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        root = Path(os.environ['TIANSHU_BUCKET_TRACE_DIR'])

        def write(name, data):
            root.mkdir(parents=True, exist_ok=True)
            p = root / f'{name}_{os.getpid()}.json'
            p.write_text(json.dumps(data, indent=2) + '\n')

        configure, warmup, pad = module.configure, module.warmup, module.padded_tokens
        seen = set()

        def configure_traced(config, platform, logger):
            result = configure(config, platform, logger)
            write('configured', dict(enabled=module.enabled(config, platform),
                                    points=config.compilation_config.cudagraph_capture_sizes,
                                    max_capture=config.compilation_config.max_cudagraph_capture_size,
                                    mode=config.compilation_config.cudagraph_mode.name))
            return result

        def warmup_traced(worker, logger):
            original = worker.model_runner._dummy_run
            calls = []

            def dummy(*args, **kwargs):
                calls.append(dict(size=args[0], mode=str(kwargs.get('cudagraph_runtime_mode'))))
                write('warmup_progress', dict(complete=False, calls=calls))
                return original(*args, **kwargs)

            start = time.monotonic()
            worker.model_runner._dummy_run = dummy
            write('warmup_started', dict(started=True))
            try:
                result = warmup(worker, logger)
            finally:
                worker.model_runner._dummy_run = original
            write('warmup_completed', dict(complete=True, calls=calls,
                                           seconds=time.monotonic() - start))
            return result

        def pad_traced(num_tokens, max_tokens):
            result = pad(num_tokens, max_tokens)
            key = (num_tokens, max_tokens, result)
            if result != num_tokens and key not in seen:
                seen.add(key)
                write(f'padding_{num_tokens}_{result}', dict(actual=num_tokens, padded=result,
                                                            budget=max_tokens))
            return result

        module.configure = configure_traced
        module.warmup = warmup_traced
        module.padded_tokens = pad_traced


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm_fl.worker.iluvatar_prefill_buckets':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


if os.getenv('TIANSHU_BUCKET_RECHECK_TRACE') == '1':
    sys.meta_path.insert(0, Finder())
