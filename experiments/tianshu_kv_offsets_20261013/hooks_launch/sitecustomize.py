# SPDX-License-Identifier: Apache-2.0
"""Native BF16 KV writer launch only; source, grid and integer widths unchanged."""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / "tianshu_buckets75_recheck_20261012/joint_hooks/sitecustomize.py"))
SOURCE_SHA = "f18224e51162e37c14a2e2b3045bca55baf0b207b1d203ad6ee7d8ed417139eb"


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == SOURCE_SHA
        kernel = module.reshape_and_cache_kernel_flash
        seen = set()

        class Router:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    supported = (
                        not args and not kwargs["FP8_KV_CACHE"]
                        and not kwargs["USE_HEAD_MAJOR_LAYOUT"]
                        and kwargs["num_heads"] == 2 and kwargs["head_size"] == 128
                        and kwargs["block_size"] == 16 and kwargs["TILE_SIZE"] == 256
                        and str(kwargs["key_ptr"].dtype) == "torch.bfloat16"
                        and str(kwargs["value_ptr"].dtype) == "torch.bfloat16"
                        and str(kwargs["key_cache_ptr"].dtype) == "torch.bfloat16"
                        and str(kwargs["value_cache_ptr"].dtype) == "torch.bfloat16"
                    )
                    if not supported:
                        return kernel[grid](*args, **kwargs)
                    original = dict(num_warps=kwargs["num_warps"], num_stages=kwargs["num_stages"])
                    kwargs.update(num_warps=4, num_stages=1)
                    compiled = kernel[grid](*args, **kwargs)
                    key = (kwargs["slot_mapping_ptr"].numel(), kwargs["key_stride"], kwargs["page_stride"])
                    if key not in seen:
                        seen.add(key)
                        root = Path(os.environ["TIANSHU_KV_LAUNCH_RECORD_DIR"])
                        root.mkdir(parents=True, exist_ok=True)
                        metadata = getattr(compiled, "metadata", None)
                        record = dict(tokens=key[0], key_stride=key[1], page_stride=key[2],
                                      source_sha256=SOURCE_SHA, original_configuration=original,
                                      configuration=dict(num_warps=4, num_stages=1),
                                      slot_dtype=str(kwargs["slot_mapping_ptr"].dtype),
                                      body_grid_pointer_types_and_integer_widths_unchanged=True,
                                      compiled_metadata={k: getattr(metadata, k, None)
                                                         for k in ("num_warps", "num_stages", "shared")})
                        (root / f"route_{os.getpid()}_{len(seen)}.json").write_text(json.dumps(record, indent=2) + "\n")
                    return compiled
                return launch

        module.reshape_and_cache_kernel_flash = Router()


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm.v1.attention.ops.triton_reshape_and_cache_flash":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


if os.environ.get("TIANSHU_KV_LAUNCH_PROBE") == "1":
    sys.meta_path.insert(0, Finder())
