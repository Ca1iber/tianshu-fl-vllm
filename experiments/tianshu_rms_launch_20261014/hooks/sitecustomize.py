# SPDX-License-Identifier: Apache-2.0
"""RMSNorm launch only for BF16 hidden2048 small batches, atop verified TopP."""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / "tianshu_topp_launch_20261013/hooks/sitecustomize.py"))
SOURCES = {
    "flag_gems.ops.rms_norm": ("rms_norm_kernel", "normal", "58407466be597c05442b3288c95abb2c33e47f0f6b98cff624b2f2e8a00823e3"),
    "flag_gems.fused.fused_add_rms_norm": ("fused_add_rms_norm_kernel", "fused", "192cbddc2283a564a9eaf90bfae1641a8a052398c0d7dd14f83f6a67f76ec3b9"),
}


class Loader(importlib.abc.Loader):
    def __init__(self, original, name):
        self.original, self.name = original, name

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        attribute, kind, sha = SOURCES[self.name]
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == sha
        original = getattr(module, attribute)
        jit = original.fn
        assert isinstance(jit, module.triton.runtime.JITFunction)
        seen = set()

        class Router:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    x = args[2] if kind == "normal" else args[0]
                    weight = args[3] if kind == "normal" else args[2]
                    n = args[8] if kind == "normal" else args[7]
                    strides = args[4:8] if kind == "normal" else args[3:7]
                    rows = x.numel() // 2048
                    supported = (str(x.dtype) == str(weight.dtype) == "torch.bfloat16"
                                 and x.is_contiguous() and weight.is_contiguous()
                                 and n == 2048 and x.shape[-1] == 2048 and 1 <= rows <= 64
                                 and tuple(strides) == (2048, 1, 2048, 1))
                    if kind == "fused":
                        supported = supported and args[1].is_contiguous() and args[1].dtype == x.dtype
                    else:
                        supported = supported and args[0].is_contiguous() and args[0].dtype == x.dtype
                    if not supported:
                        return original[grid](*args, **kwargs)
                    kwargs.update(num_warps=8, num_stages=1)
                    # Raw identical JIT avoids LibEntry reusing a launch-agnostic key.
                    compiled = jit[grid](*args, **kwargs)
                    if rows not in seen:
                        assert compiled.metadata.num_warps == 8 and compiled.metadata.num_stages == 1
                        seen.add(rows)
                        root = Path(os.environ["TIANSHU_RMS_LAUNCH_RECORD_DIR"])
                        root.mkdir(parents=True, exist_ok=True)
                        data = dict(kind=kind, rows=rows, columns=2048, dtype=str(x.dtype),
                                    configuration=dict(num_warps=8, num_stages=1),
                                    source_sha256=sha, same_jit_body=True, grid_unchanged=True,
                                    compiled_metadata={k: getattr(compiled.metadata, k, None)
                                                       for k in ("num_warps", "num_stages", "shared")})
                        (root / f"{kind}_{os.getpid()}_{rows}.json").write_text(json.dumps(data, indent=2) + "\n")
                    return compiled
                return launch

        setattr(module, "_tianshu_rms_original_kernel", original)
        setattr(module, attribute, Router())


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in SOURCES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader, fullname)
        return spec


if os.environ.get("TIANSHU_RMS_LAUNCH_PROBE") == "1":
    sys.meta_path.insert(0, Finder())
