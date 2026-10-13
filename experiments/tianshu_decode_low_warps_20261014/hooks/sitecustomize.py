# SPDX-License-Identifier: Apache-2.0
"""Native top-p launch only, composed with the verified L0 DecodeKV32 profile."""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / "tianshu_decode_low_warps_20261014/hooks/decode_sitecustomize.py"))
SOURCE_SHA = "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8"


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == SOURCE_SHA
        kernel = module._topk_topp_kernel
        seen = set()

        class Router:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    supported = (bool(args) and str(args[0].dtype) == "torch.float32"
                                 and args[0].stride(1) == 1 and kwargs.get("VOCAB_SIZE") == 130560
                                 and 8 <= kwargs.get("BATCH_SIZE", 0) <= 64
                                 and kwargs.get("BLOCK_SIZE") == 8192
                                 and kwargs.get("BLOCK_SIZE_TRUNC") == 4096)
                    if supported:
                        kwargs.update(num_warps=16, num_stages=1)
                    result = kernel[grid](*args, **kwargs)
                    if supported:
                        key = kwargs["BATCH_SIZE"], kwargs["TOPK_ENABLED"], kwargs["TOPP_ENABLED"]
                        if key not in seen:
                            seen.add(key)
                            root = Path(os.environ["TIANSHU_TOPP_LAUNCH_RECORD_DIR"])
                            root.mkdir(parents=True, exist_ok=True)
                            (root / f"route_{os.getpid()}_{len(seen)}.json").write_text(json.dumps(dict(
                                batch=key[0], topk=key[1], topp=key[2], vocab=130560,
                                configuration=dict(num_warps=16, num_stages=1), source_sha256=SOURCE_SHA,
                                body_grid_blocks_RNG_and_sampling_parameters_unchanged=True,
                                compiled_metadata={k: getattr(result.metadata, k, None)
                                                   for k in ("num_warps", "num_stages", "shared")}), indent=2) + "\n")
                    return result
                return launch

        module._topk_topp_kernel = Router()


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm.v1.sample.ops.topk_topp_triton":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


if os.environ.get("TIANSHU_TOPP_LAUNCH_PROBE") == "1":
    sys.meta_path.insert(0, Finder())
