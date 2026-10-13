# SPDX-License-Identifier: Apache-2.0
"""Guarded greedy replacement with request-phase route evidence in both groups."""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / "tianshu_decode_low_warps_20261014/hooks/sitecustomize.py"))
SOURCE_SHA = "1a2f18ddc91c89f3744ab5a93624be056bf20b7a8b2e6e13d0236182718a2253"


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == SOURCE_SHA
        implementation = BASE / "tianshu_greedy_argmax_20261014/implementation.py"
        spec = importlib.util.spec_from_file_location("_tianshu_greedy_argmax_impl", implementation)
        code = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = code
        spec.loader.exec_module(code)
        original = module.Sampler.greedy_sample
        mode = os.environ["TIANSHU_GREEDY_ARGMAX_MODE"]
        assert mode in ("control", "candidate")
        marker = Path(os.environ["TIANSHU_GREEDY_REQUEST_MARKER"])
        root = Path(os.environ["TIANSHU_GREEDY_ARGMAX_RECORD_DIR"])
        seen = set()

        def greedy(logits):
            supported = (str(logits.dtype) == "torch.float32" and logits.is_cuda
                         and logits.ndim == 2 and 1 <= logits.shape[0] <= 64
                         and logits.shape[1] == 130560 and logits.stride(1) == 1)
            selected = supported and mode == "candidate"
            config = dict(tile=1024 if logits.shape[0] == 1 else 2048, warps=4) if supported else None
            output = code.launch(logits, config) if selected else original(logits)
            phase = "request" if marker.exists() else "warmup"
            key = phase, tuple(logits.shape), tuple(logits.stride()), selected
            if key not in seen:
                seen.add(key)
                root.mkdir(parents=True, exist_ok=True)
                registration = module.torch._C._dispatch_dump_table("aten::argmax")
                registration = [x for x in registration.splitlines() if x.startswith("CUDA:")]
                record = dict(phase=phase, batch=logits.shape[0], vocab=logits.shape[-1],
                              dtype=str(logits.dtype), stride=list(logits.stride()), selected=selected,
                              mode=mode, configuration=config if selected else None,
                              output_dtype=str(output.dtype), output_shape=list(output.shape),
                              actual_sampler_sha256=SOURCE_SHA,
                              implementation_sha256=hashlib.sha256(implementation.read_bytes()).hexdigest(),
                              argmax_cuda_registration=registration)
                (root / f"route_{os.getpid()}_{len(seen)}.json").write_text(json.dumps(record, indent=2) + "\n")
            return output

        module.Sampler.greedy_sample = staticmethod(greedy)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm.v1.sample.sampler":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


if os.environ.get("TIANSHU_GREEDY_ARGMAX_PROBE") == "1":
    sys.meta_path.insert(0, Finder())
