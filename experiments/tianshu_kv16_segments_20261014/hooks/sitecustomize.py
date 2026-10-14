# SPDX-License-Identifier: Apache-2.0
"""Process-only B64 segment4 to8 on the validated guardedKV16/warp1 profile."""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / "tianshu_decode_kv16_w2_20261014/hooks/sitecustomize.py"))
SOURCE_SHA = "e5d983e43c3b343afb558440a9114255cebc5d01569cc192191290b0bb0c84c8"


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == SOURCE_SHA
        original = module.choose_segments
        module.COUNTS.setdefault(8, 0)
        seen = set()

        def choose(num_sequences):
            baseline = original(num_sequences)
            selected = 8 if num_sequences == 64 else baseline
            key = num_sequences, baseline, selected
            if key not in seen:
                seen.add(key)
                root = Path(os.environ["TIANSHU_SEGMENT_POLICY_RECORD_DIR"])
                root.mkdir(parents=True, exist_ok=True)
                (root / f"choose_{os.getpid()}_{len(seen)}.json").write_text(json.dumps(dict(
                    batch=num_sequences, baseline_segments=baseline, selected_segments=selected,
                    source_sha256=SOURCE_SHA, policy_hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    supported_geometry_scratch_and_native_fallback_guards_preserved=True,
                    grid_segment_axis_and_FP32_reduction_order_changed=selected != baseline), indent=2) + "\n")
            return selected

        module.choose_segments = choose


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm_fl.worker.iluvatar_adaptive_decode":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


if os.environ.get("TIANSHU_SEGMENT_POLICY_PROBE") == "1":
    sys.meta_path.insert(0, Finder())
