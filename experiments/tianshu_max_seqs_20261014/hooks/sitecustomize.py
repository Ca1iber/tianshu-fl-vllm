# SPDX-License-Identifier: Apache-2.0
"""Keep the validated .cv profile and record actual WorkerFL scheduling config."""
from functools import wraps
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / "tianshu_decode_cache_service_20261014/hooks/sitecustomize.py"))


class Loader(importlib.abc.Loader):
    def __init__(self, original): self.original = original
    def create_module(self, spec): return self.original.create_module(spec)
    def exec_module(self, module):
        self.original.exec_module(module)
        original = module.WorkerFL.__init__

        @wraps(original)
        def initialize(worker, *args, **kwargs):
            original(worker, *args, **kwargs)
            config = worker.vllm_config
            scheduler = config.scheduler_config
            mode = os.environ["TIANSHU_MAX_SEQS_MODE"]
            assert mode in ("control", "candidate")
            if mode == "candidate": assert scheduler.max_num_seqs == 64
            record = dict(mode=mode, pid=os.getpid(), worker_class="WorkerFL",
                max_num_seqs=scheduler.max_num_seqs,
                max_num_batched_tokens=scheduler.max_num_batched_tokens,
                async_scheduling=scheduler.async_scheduling,
                model_max_len=config.model_config.max_model_len,
                cudagraph_capture_sizes=list(config.compilation_config.cudagraph_capture_sizes or []),
                worker_source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                hook_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                installed_sources_modified=False)
            root = Path(os.environ["TIANSHU_MAX_SEQS_RECORD_DIR"])
            root.mkdir(parents=True, exist_ok=True)
            (root / ("worker_" + str(os.getpid()) + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        module.WorkerFL.__init__ = initialize


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm_fl.worker.worker": return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


if os.getenv("TIANSHU_MAX_SEQS_MODE") in ("control", "candidate"):
    sys.meta_path.insert(0, Finder())
