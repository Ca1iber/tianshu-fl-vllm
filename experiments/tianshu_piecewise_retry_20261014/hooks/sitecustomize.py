# SPDX-License-Identifier: Apache-2.0
"""Compose current CV profile with piecewise eligibility and actual graph configuration records."""
from functools import wraps
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / "tianshu_decode_cache_service_20261014/hooks/sitecustomize.py"))


def save(name, record):
    root = Path(os.environ["TIANSHU_PIECEWISE_RECORD_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    record.update(pid=os.getpid(), mode=os.environ["TIANSHU_PIECEWISE_MODE"],
                  hook_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root / (name + "_" + str(os.getpid()) + ".json")).write_text(json.dumps(record, indent=2) + "\n")


class Loader(importlib.abc.Loader):
    def __init__(self, original, kind): self.original, self.kind = original, kind
    def create_module(self, spec): return self.original.create_module(spec)
    def exec_module(self, module):
        self.original.exec_module(module)
        digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        mode = os.environ["TIANSHU_PIECEWISE_MODE"]
        if self.kind == "buckets":
            assert digest == "966aad18aad80ef488928cafc8672cbfa2d547a832f9b151346f719ea3abc6db"
            assert len(module.PREFILL_BUCKETS) == 99
            proof = None
            if mode == "candidate":
                helper = Path(__file__).resolve().parents[1] / "allow_piecewise.py"
                proof = runpy.run_path(str(helper))["extend"](module)
                proof["helper_sha256"] = hashlib.sha256(helper.read_bytes()).hexdigest()
            save("guard", dict(native_source_sha256=digest, extension=proof))
        elif self.kind == "worker":
            assert digest == "6cf62f60ca8ea12b7c494f0009fa733f33270a73e014cdbd4a45cada787c032a"
            original = module.WorkerFL.compile_or_warm_up_model

            @wraps(original)
            def warmup(worker, *args, **kwargs):
                result = original(worker, *args, **kwargs)
                config = worker.vllm_config
                compilation = config.compilation_config
                expected = "FULL_AND_PIECEWISE" if mode == "candidate" else "FULL_DECODE_ONLY"
                assert compilation.cudagraph_mode.name == expected
                save("worker", dict(actual_cudagraph_mode=compilation.cudagraph_mode.name,
                    compilation_mode=str(compilation.mode), splitting_ops=compilation.splitting_ops,
                    capture_points=list(compilation.cudagraph_capture_sizes),
                    max_num_seqs=config.scheduler_config.max_num_seqs,
                    max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
                    model_max_len=config.model_config.max_model_len, worker_source_sha256=digest))
                return result
            module.WorkerFL.compile_or_warm_up_model = warmup


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        kinds = {"vllm_fl.worker.iluvatar_prefill_buckets": "buckets", "vllm_fl.worker.worker": "worker"}
        if fullname not in kinds: return None
        index = sys.meta_path.index(self)
        sys.meta_path.remove(self)
        try:
            for finder in tuple(sys.meta_path):
                method = getattr(finder, "find_spec", None)
                spec = method(fullname, path, target) if method else None
                if spec is not None:
                    spec.loader = Loader(spec.loader, kinds[fullname])
                    return spec
            raise RuntimeError("Underlying loader unavailable: " + fullname)
        finally:
            sys.meta_path.insert(min(index, len(sys.meta_path)), self)


if os.environ.get("TIANSHU_PIECEWISE_MODE") in ("control", "candidate"):
    sys.meta_path.insert(0, Finder())
