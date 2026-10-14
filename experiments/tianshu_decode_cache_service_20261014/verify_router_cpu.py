# SPDX-License-Identifier: Apache-2.0
"""Exercise the real Decode producer/reducer and hook composition without GPU."""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace as NS


def main():
    base = Path(__file__).resolve().parent
    source = base / "hooks/decode_sitecustomize.py"
    original = base.parent / "tianshu_decode_kv16_w2_20261014/hooks/decode_sitecustomize.py"
    a, b = ast.parse(original.read_text()), ast.parse(source.read_text())
    find_class = lambda tree, name: next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)
    guard = lambda tree: next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "supported" for t in n.targets))
    assert ast.dump(guard(a)) == ast.dump(guard(b))
    assert ast.dump(find_class(a, "Reducer")) == ast.dump(find_class(b, "Reducer"))
    classes = ast.Module(body=[find_class(b, "Producer"), find_class(b, "Reducer")], type_ignores=[])

    class Kernel:
        def __init__(self): self.calls = []
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.calls.append((grid, args, kwargs))
                return NS(metadata=NS(num_warps=kwargs.get("num_warps"), num_stages=kwargs.get("num_stages"), shared=0))
            return launch

    count, fallbacks = 0, 0
    with tempfile.TemporaryDirectory(prefix="tianshu_cache_router_cpu_") as temp:
        for mode in ("control", "candidate"):
            kernels = {name: Kernel() for name in ("producer", "candidate", "cache_candidate", "reducer")}
            actual_sha = ("4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e" if mode == "control"
                          else "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289")
            env = dict(kernels, selected={"tile": None}, seen=set(), reduce_seen=set(), mode="candidate", cache_mode=mode,
                root=Path(temp) / mode, os=os, json=json, hashlib=hashlib,
                cache_helper=base.parent / "tianshu_decode_cache_policy_20261014/experimental_cache.py",
                cache_provenance={"generated_sha256": "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"},
                actual_provenance={"generated_sha256": actual_sha},
                SOURCE_SHA="bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0")
            exec(compile(classes, str(source), "exec"), env)
            producer, reducer = env["Producer"](), env["Reducer"]()
            for batch in range(1,65):
                kw = dict(IS_3D=True, BLOCK_M=16, BLOCK_Q=2, num_query_heads=16, num_queries_per_kv=8,
                    HEAD_SIZE=128, KV_QUANT_MODE=0, num_seqs=batch, USE_CAUSAL=True,
                    query_ptr=NS(dtype="torch.bfloat16"), key_cache_ptr=NS(dtype="torch.bfloat16"),
                    value_cache_ptr=NS(dtype="torch.bfloat16"), TILE_SIZE=99,
                    NUM_SEGMENTS_PER_SEQ=8 if batch == 64 else 64 if batch == 1 else 4 if batch >= 16 else 16)
                producer[(123,2,8)](**kw)
                kernel = kernels["candidate" if mode == "control" else "cache_candidate"]
                grid, args, actual = kernel.calls[-1]
                tile = 16 if batch in (8,16,32,40,48,56,64) else 32
                warps = 1 if tile == 16 else 2 if 8 <= batch <= 64 else 4
                assert grid == (123,2,8) and not args
                assert actual["TILE_SIZE"] == tile and actual["num_warps"] == warps and actual["num_stages"] == 1
                assert actual.get("KV_CACHE_MODIFIER") == (None if mode == "control" else ".cv")
                reducer[(456,)](num_seqs=batch, NUM_SEGMENTS_PER_SEQ=kw["NUM_SEGMENTS_PER_SEQ"], TILE_SIZE=99)
                assert kernels["reducer"].calls[-1][2]["TILE_SIZE"] == tile
                count += 1
            changes = [dict(IS_3D=False), dict(num_seqs=65), dict(KV_QUANT_MODE=1), dict(USE_CAUSAL=False),
                dict(BLOCK_M=32), dict(BLOCK_Q=4), dict(HEAD_SIZE=64), dict(num_query_heads=8),
                dict(query_ptr=NS(dtype="torch.float32")), dict(key_cache_ptr=NS(dtype="torch.float32")),
                dict(value_cache_ptr=NS(dtype="torch.float32")), dict(CHUNK_LOOKBACK=0)]
            changes += [{name: True} for name in ("USE_SINKS","USE_TD","USE_TD_QO","USE_FP8",
                "USE_ALIBI_SLOPES","USE_QQ_BIAS","USE_SOFTCAP","USE_MM_PREFIX","USE_PER_SEQ_CAUSAL","SLIDING_WINDOW")]
            for change in changes:
                changed = dict(kw, **change)
                producer[(123,2,8)](**changed)
                assert kernels["producer"].calls[-1] == ((123,2,8), (), changed)
                reducer[(456,)](num_seqs=changed["num_seqs"], NUM_SEGMENTS_PER_SEQ=8, TILE_SIZE=99)
                assert kernels["reducer"].calls[-1][2]["TILE_SIZE"] == 99
                fallbacks += 1
            assert all(json.loads(p.read_text())["generated_source_sha256"] == actual_sha
                       for p in env["root"].glob("producer_*.json"))
    composition = base / "hooks/sitecustomize.py"
    before = runpy.run_path
    for fail in (False, True):
        calls = []
        def fake(path, *args, **kwargs):
            calls.append(Path(path).resolve())
            if len(calls) == 1:
                return runpy.run_path(str(original.resolve()))
            if fail: raise RuntimeError("intentional composition failure")
            return {}
        runpy.run_path = fake
        try:
            try: exec(compile(composition.read_text(), str(composition), "exec"), {"__file__": str(composition.resolve())})
            except RuntimeError:
                assert fail
            else:
                assert not fail
            assert calls == [(base.parent / "tianshu_budget8192_buckets_20261014/hooks/sitecustomize.py").resolve(), source.resolve()]
            assert runpy.run_path is fake
        finally:
            runpy.run_path = before
    result = dict(complete=True, cpu_only=True, producer_reducer_cases=count, fallback_cases=fallbacks,
        actual_copied_router_AST_exercised=True, supported_guard_AST_unchanged=True, entire_reducer_AST_unchanged=True,
        actual_composition_dispatch_and_finally_restore_verified=True, actual_generated_provenance_verified=True,
        GPU_service_not_run=True, sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), source, original, composition)})
    out = base / "results/20261014/service_prepared_32g"
    out.mkdir(parents=True, exist_ok=True)
    (out / "CPU_ROUTING_AUDIT_32g.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
