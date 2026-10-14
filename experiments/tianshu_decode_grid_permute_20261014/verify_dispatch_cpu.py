# SPDX-License-Identifier: Apache-2.0
"""Execute the actual probe's grid adapter without importing any GPU packages."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from grid_permute import permute_grid


def main():
    base = Path(__file__).resolve().parent
    probe = base / "probe_grid.py"
    tree = ast.parse(probe.read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "PageRouter")

    class BaseRouter:
        def __init__(self, original, configuration):
            self.original, self.configuration, self.metadata = original, configuration, {}

        def __getitem__(self, grid):
            def launch(**kwargs):
                kwargs.update(self.configuration)
                compiled = self.original[grid](**kwargs)
                self.metadata = dict(num_warps=1, num_stages=1, producer_and_reducer_tile=16)
                return compiled
            return launch

    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(**kwargs):
                calls.append(tuple(grid))
                return SimpleNamespace(n_regs=85, n_spills=0)
            return launch

    scope = dict(LaunchRouter=BaseRouter, permute_grid=permute_grid, case_id=0,
                 asm_recorded=set(), Path=Path, json=json, hashlib=hashlib)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(probe), "exec"), scope)
    configurations = []
    for axes in [None] + list(itertools.permutations(range(3))):
        router = scope["PageRouter"](Kernel(), dict(KV_CACHE_MODIFIER=".cv"), axes)
        router[(96, 2, 8)](query_ptr=SimpleNamespace(shape=(64, 16, 128)), num_seqs=64,
                           BLOCK_Q=2, BLOCK_M=16, BLOCK_SIZE=16)
        expected = (96, 2, 8) if axes is None else permute_grid((96, 2, 8), axes)
        assert calls[-1] == expected
        assert router.metadata["actual_grid"] == list(expected)
        assert router.metadata["logical_grid"] == [96, 2, 8]
        assert router.metadata["logical_to_physical_axes"] == axes
        assert router.metadata["requested_cache_modifier"] == ".cv"
        configurations.append(dict(axes=axes, physical_grid=expected))
    output = dict(complete=True, cpu_only=True, gpu_execution=False,
                  actual_probe_adapter_executed=True, stub_parent_not_actual_LaunchRouter=True,
                  configurations=configurations,
                  sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), probe, base / "grid_permute.py")})
    out = base / "results/20261014/prepared_32g/CPU_DISPATCH_AUDIT_32g.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
