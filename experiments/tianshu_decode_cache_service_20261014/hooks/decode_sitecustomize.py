# SPDX-License-Identifier: Apache-2.0
"""Preserve current KV16/S8 routing; change only supported Decode K/V cache hint."""
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
runpy.run_path(str(BASE / "tianshu_buckets75_recheck_20261012/joint_hooks/sitecustomize.py"))
SOURCE_SHA = "bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0"


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == SOURCE_SHA
        root = Path(os.environ["TIANSHU_DECODE_COHERENT_RECORD_DIR"])
        helper = BASE / "tianshu_native_prefill_20261012/zero_initialization.py"
        spec = importlib.util.spec_from_file_location("_tianshu_decode_coherent_zero", helper)
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        candidate, provenance = generator.load_kernel(module, root / "generated" / str(os.getpid()))
        assert provenance["generated_sha256"] == "4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e"
        cache_helper = BASE / "tianshu_decode_cache_policy_20261014/experimental_cache.py"
        cache_generator = runpy.run_path(str(cache_helper))
        cache_candidate, cache_provenance = cache_generator["load_kernel"](
            provenance["generated_source"], root / "cache_generated" / str(os.getpid()))
        assert cache_provenance["generated_sha256"] == "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
        cache_mode = os.environ["TIANSHU_CACHE_SERVICE_MODE"]
        assert cache_mode in ("control", "candidate")
        actual_provenance = provenance if cache_mode == "control" else dict(
            provenance, generated_sha256=cache_provenance["generated_sha256"],
            generated_source=cache_provenance["generated_source"],
            original_L0_generation=provenance, cache_policy_generation=cache_provenance)
        producer = module.kernel_unified_attention
        reducer = module.reduce_segments
        mode = os.environ["TIANSHU_KV16_SERVICE_MODE"]
        assert mode in ("control", "candidate")
        selected = {"tile": None}
        seen = set()
        reduce_seen = set()

        class Producer:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    supported = (
                        not args and kwargs.get("IS_3D") is True
                        and kwargs.get("BLOCK_M") == 16 and kwargs.get("BLOCK_Q") == 2
                        and kwargs.get("num_query_heads") == 16 and kwargs.get("num_queries_per_kv") == 8
                        and kwargs.get("HEAD_SIZE") == 128 and kwargs.get("KV_QUANT_MODE") == 0
                        and kwargs.get("num_seqs", 65) <= 64
                        and not kwargs.get("USE_SINKS") and not kwargs.get("USE_TD")
                        and not kwargs.get("USE_TD_QO") and not kwargs.get("USE_FP8")
                        and not kwargs.get("USE_ALIBI_SLOPES") and not kwargs.get("USE_QQ_BIAS")
                        and not kwargs.get("USE_SOFTCAP") and not kwargs.get("USE_MM_PREFIX")
                        and not kwargs.get("USE_PER_SEQ_CAUSAL") and kwargs.get("USE_CAUSAL")
                        and not kwargs.get("SLIDING_WINDOW") and kwargs.get("CHUNK_LOOKBACK", -1) < 0
                        and str(kwargs["query_ptr"].dtype) == "torch.bfloat16"
                        and str(kwargs["key_cache_ptr"].dtype) == "torch.bfloat16"
                        and str(kwargs["value_cache_ptr"].dtype) == "torch.bfloat16"
                    )
                    selected["tile"] = 32 if supported else None
                    if not supported:
                        return producer[grid](*args, **kwargs)
                    warps = 2 if 8 <= kwargs["num_seqs"] <= 64 else 4
                    changed = mode == "candidate" and kwargs["num_seqs"] in (8, 16, 32, 40, 48, 56, 64)
                    tile = 16 if changed else 32
                    warps = 1 if changed else warps
                    selected["tile"] = tile
                    kwargs.update(TILE_SIZE=tile, num_warps=warps, num_stages=1)
                    if cache_mode == "candidate":
                        result = cache_candidate[grid](KV_CACHE_MODIFIER=".cv", **kwargs)
                    else:
                        result = candidate[grid](**kwargs)
                    key = kwargs["num_seqs"], kwargs["NUM_SEGMENTS_PER_SEQ"]
                    if key not in seen:
                        seen.add(key)
                        root.mkdir(parents=True, exist_ok=True)
                        record = dict(batch=key[0], segments=key[1],
                                      configuration=dict(TILE_SIZE=tile, num_warps=warps, num_stages=1),
                                      generated_source_sha256=actual_provenance["generated_sha256"],
                                      zero_initialization=actual_provenance, producer_and_reducer_tile=tile,
                                      cache_service_mode=cache_mode,
                                      actual_cache_modifier=".cv" if cache_mode == "candidate" else None,
                                      cache_policy_generation=cache_provenance,
                                      cache_helper_sha256=hashlib.sha256(cache_helper.read_bytes()).hexdigest(),
                                      mode=mode, candidate_selected=changed,
                                      grid_unchanged=True, integer_widths_and_pointer_types_unchanged=True,
                                      compiled_metadata={k: getattr(result.metadata, k, None)
                                                         for k in ("num_warps", "num_stages", "shared")})
                        (root / f"producer_{os.getpid()}_{len(seen)}.json").write_text(json.dumps(record, indent=2) + "\n")
                    return result
                return launch

        class Reducer:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    if selected["tile"] is not None:
                        kwargs["TILE_SIZE"] = selected["tile"]
                    result = reducer[grid](*args, **kwargs)
                    key = kwargs["num_seqs"], kwargs["NUM_SEGMENTS_PER_SEQ"]
                    if selected["tile"] is not None and key not in reduce_seen:
                        reduce_seen.add(key)
                        (root / f"reducer_{os.getpid()}_{len(reduce_seen)}.json").write_text(json.dumps(dict(
                            batch=key[0], segments=key[1], actual_tile=kwargs["TILE_SIZE"],
                            original_reducer_source_sha256=SOURCE_SHA, grid_unchanged=True), indent=2) + "\n")
                    return result
                return launch

        module.kernel_unified_attention = Producer()
        module.reduce_segments = Reducer()


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm.v1.attention.ops.triton_unified_attention":
            return None
        for finder in tuple(sys.meta_path):
            if finder is self:
                continue
            method = getattr(finder, "find_spec", None)
            spec = method(fullname, path, target) if method else None
            if spec is not None:
                spec.loader = Loader(spec.loader)
                return spec
        return None


if os.environ.get("TIANSHU_DECODE_COHERENT_PROBE") == "1":
    sys.meta_path.insert(0, Finder())
