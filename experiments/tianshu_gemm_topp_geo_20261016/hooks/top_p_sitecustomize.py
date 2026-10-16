# SPDX-License-Identifier: Apache-2.0
"""Standalone TopP repair/16K tile; native TopK and unsupported routes retained."""
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
runpy.run_path(str(BASE / "tianshu_decode_kv16_w2_20261014/hooks/decode_sitecustomize.py"))
SOURCE_SHA = "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8"
GENERATED_SHA = "09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d"


class Loader(importlib.abc.Loader):
    def __init__(self, original): self.original = original
    def create_module(self, spec): return self.original.create_module(spec)
    def exec_module(self, module):
        self.original.exec_module(module)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == SOURCE_SHA
        kernel = module._topk_topp_kernel
        mode = os.environ["TIANSHU_TOPP_PRECISION_SERVICE_MODE"]
        assert mode in ("control", "candidate")
        root = Path(os.environ["TIANSHU_TOPP_LAUNCH_RECORD_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        corrected, generation = None, None
        if mode == "candidate":
            helper = BASE / "tianshu_topp_geometric_20261016/precision.py"
            spec = importlib.util.spec_from_file_location("_precision_helper", helper)
            repair = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(repair)
            kernels, all_generation = repair.load_kernel(module.__file__, root / ("precision_generated_" + str(os.getpid())))
            corrected = kernels['geo']
            record = all_generation['modes']['geo']
            generation = dict(generated_source=record['source'], generated_sha256=record['sha256'],
                              strict_CPU_tolerance=1e-6, p_unchanged=True, actual_global_max_normalization=True, geometric_interval_midpoint=True)
            assert generation["generated_sha256"] == GENERATED_SHA
        seen = set()

        class Router:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    supported = (bool(args) and str(args[0].dtype) == "torch.float32"
                                 and args[0].stride(1) == 1 and kwargs.get("VOCAB_SIZE") == 130560
                                 and 8 <= kwargs.get("BATCH_SIZE", 0) <= 64
                                 and kwargs.get("BLOCK_SIZE") == 8192
                                 and kwargs.get("BLOCK_SIZE_TRUNC") == 4096)
                    active = bool(supported and mode == "candidate" and kwargs.get("TOPP_ENABLED") and not kwargs.get("TOPK_ENABLED"))
                    if supported: kwargs.update(num_warps=16, num_stages=1)
                    if active: kwargs.update(BLOCK_SIZE=32768, BLOCK_SIZE_TRUNC=16384)
                    selected = corrected if active else kernel
                    result = selected[grid](*args, **kwargs)
                    if supported:
                        key = kwargs["BATCH_SIZE"], kwargs["TOPK_ENABLED"], kwargs["TOPP_ENABLED"]
                        if key not in seen:
                            seen.add(key)
                            record = dict(batch=key[0], topk=key[1], topp=key[2], vocab=130560,
                                configuration=dict(num_warps=16, num_stages=1), source_sha256=SOURCE_SHA,
                                precision_service_mode=mode, corrected_source_active=active,
                                actual_BLOCK_SIZE=kwargs["BLOCK_SIZE"], actual_BLOCK_SIZE_TRUNC=kwargs["BLOCK_SIZE_TRUNC"],
                                actual_source_sha256=GENERATED_SHA if active else SOURCE_SHA,
                                source_module="_tianshu_topp_geometric_geo" if active else "vllm.v1.sample.ops.topk_topp_triton",
                                generation=generation, sampling_parameters_and_RNG_unchanged=True,
                                body_grid_blocks_RNG_and_sampling_parameters_unchanged=not active,
                                body_changed=active, grid_unchanged=True,
                                hook_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                compiled_metadata={k: getattr(result.metadata, k, None) for k in ("num_warps", "num_stages", "shared")})
                            (root / f"route_{os.getpid()}_{len(seen)}.json").write_text(json.dumps(record, indent=2) + "\n")
                    return result
                return launch
        module._topk_topp_kernel = Router()


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm.v1.sample.ops.topk_topp_triton": return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


if os.environ.get("TIANSHU_TOPP_LAUNCH_PROBE") == "1":
    sys.meta_path.insert(0, Finder())
