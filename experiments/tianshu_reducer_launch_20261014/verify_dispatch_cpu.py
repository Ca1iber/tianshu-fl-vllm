# SPDX-License-Identifier: Apache-2.0
"""Exercise actual producer/tile routers and probe adapter with CPU fake JITs."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

from reducer_router import ReducerLaunchRouter


def main():
    base = Path(__file__).resolve().parent
    router_source = base.parent / "tianshu_decode_low_warps_20261014/launch_router.py"
    spec = importlib.util.spec_from_file_location("_cpu_actual_launch_router", router_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name = "vllm.v1.attention.ops.triton_unified_attention"
    for i in range(1, len(name.split(".")) + 1):
        part = ".".join(name.split(".")[:i])
        assert part not in sys.modules
        fake = ModuleType(part)
        fake.__path__ = []
        sys.modules[part] = fake
        if i > 1:
            parent, attr = part.rsplit(".", 1)
            setattr(sys.modules[parent], attr, fake)
    native = sys.modules[name]
    calls = []

    class Kernel:
        def __init__(self, kind): self.kind = kind
        def __getitem__(self, grid):
            def launch(**kwargs):
                calls.append((self.kind, tuple(grid), dict(kwargs)))
                return NS(metadata=NS(num_warps=kwargs.get("num_warps", 4),
                                      num_stages=kwargs.get("num_stages", 3), shared=0),
                          n_regs=85, n_spills=0)
            return launch

    policy = ReducerLaunchRouter(Kernel("reducer"))
    native.reduce_segments = policy
    source = base / "probe_reducer.py"
    cls = next(n for n in ast.walk(ast.parse(source.read_text()))
               if isinstance(n, ast.ClassDef) and n.name == "PageRouter")
    scope = dict(LaunchRouter=module.LaunchRouter, reduction_policy=policy,
                 case_id=0, asm_recorded=set())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), scope)
    rows = []
    for warps in (None, 1, 2, 4, 8):
        producer = scope["PageRouter"](Kernel("producer"),
            dict(TILE_SIZE=16, num_warps=1, num_stages=1, KV_CACHE_MODIFIER=".cv"), warps)
        producer[(96, 2, 8)](IS_3D=True, BLOCK_M=16, BLOCK_Q=2, num_query_heads=16,
                            num_queries_per_kv=8, HEAD_SIZE=128, KV_QUANT_MODE=0,
                            query_ptr=NS(shape=(64, 16, 128)), num_seqs=64, BLOCK_SIZE=16)
        native.reduce_segments[(64, 16)](TILE_SIZE=32, NUM_SEGMENTS_PER_SEQ=8,
                                         HEAD_SIZE=128, num_query_heads=16)
        assert calls[-2][0] == "producer" and calls[-2][2]["num_warps"] == 1
        kind, grid, kwargs = calls[-1]
        assert kind == "reducer" and grid == (64, 16) and kwargs["TILE_SIZE"] == 16
        assert kwargs.get("num_warps") == warps
        assert policy.metadata["actual_num_warps"] == (4 if warps is None else warps)
        assert policy.metadata["actual_num_stages"] == 3
        rows.append(dict(requested=warps, producer=producer.metadata, reducer=dict(policy.metadata)))
    output = dict(complete=True, cpu_only=True, gpu_execution=False,
                  actual_probe_adapter_and_existing_LaunchRouter_executed=True,
                  fake_native_and_JIT_not_runtime_evidence=True, configurations=rows,
                  sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), source, base / "reducer_router.py", router_source)})
    dest = base / "results/20261014/prepared_32g/CPU_DISPATCH_AUDIT_32g.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "configurations"}))


if __name__ == "__main__":
    main()
