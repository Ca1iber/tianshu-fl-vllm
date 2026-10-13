# SPDX-License-Identifier: Apache-2.0
"""Patch the actual native model RoPE static method, preserving other shapes."""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / "tianshu_topp_launch_20261013/hooks/sitecustomize.py"))
SOURCE_SHA = "c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5"


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == SOURCE_SHA
        source = Path(__file__).resolve().parents[1] / "implementation.py"
        spec = importlib.util.spec_from_file_location("_tianshu_native_rope_impl", source)
        impl = importlib.util.module_from_spec(spec); sys.modules[spec.name] = impl
        spec.loader.exec_module(impl); impl.install(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm.model_executor.layers.rotary_embedding.base":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


if os.environ.get("TIANSHU_NATIVE_ROPE_PROBE") == "1":
    sys.meta_path.insert(0, Finder())
