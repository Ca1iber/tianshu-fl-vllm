# SPDX-License-Identifier: Apache-2.0
"""Compose the verified 4096/83 profile with sampler-local sweep replacement."""
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / "tianshu_budget8192_buckets_20261014/hooks/sitecustomize.py"))


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        helper = runpy.run_path(str(BASE / "tianshu_sweep_service_20261014/sampling_sort.py"))
        helper["install"](module, os.environ["TIANSHU_SWEEP_SERVICE_MODE"],
                          Path(os.environ["TIANSHU_SWEEP_RECORD_DIR"]),
                          BASE / "tianshu_sort_sweep_20261014/reference_sources/metax_repeat_digit.py")


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm.v1.sample.ops.topk_topp_sampler":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


if os.getenv("TIANSHU_SWEEP_SERVICE_MODE") in ("control", "candidate"):
    sys.meta_path.insert(0, Finder())
