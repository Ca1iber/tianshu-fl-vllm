# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual outer Finder against a delegating inner Finder; restore on errors."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS


def main():
    base = Path(__file__).resolve().parent
    source = base / "hooks/sitecustomize.py"
    tree = ast.parse(source.read_text())
    finder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Finder")
    import importlib.abc
    scope = dict(importlib=importlib, sys=sys, Loader=lambda loader, kind: (loader, kind))
    exec(compile(ast.Module(body=[finder], type_ignores=[]), str(source), "exec"), scope)
    outer = scope["Finder"]()
    count = 0

    class Delegate:
        def find_spec(self, fullname, path=None, target=None):
            nonlocal count
            count += 1
            assert count < 5, "Import finders recurse"
            for entry in tuple(sys.meta_path):
                if entry is self: continue
                result = entry.find_spec(fullname, path, target)
                if result is not None: return result

    class Leaf:
        def __init__(self, fail): self.fail = fail
        def find_spec(self, fullname, path=None, target=None):
            if self.fail: raise RuntimeError("explicit leaf failure fixture")
            return NS(loader="native_loader")

    original = sys.meta_path
    checks = []
    try:
        for name, kind in [("vllm_fl.worker.iluvatar_prefill_buckets", "buckets"), ("vllm_fl.worker.worker", "worker")]:
            for fail in (False, True):
                count = 0
                sys.meta_path = [outer, Delegate(), Leaf(fail)]
                before = list(sys.meta_path)
                try:
                    spec = outer.find_spec(name)
                    assert not fail and spec.loader == ("native_loader", kind)
                except RuntimeError as error:
                    assert fail and str(error) == "explicit leaf failure fixture"
                assert sys.meta_path == before and count == 1
                checks.append(dict(module=name, leaf_failure=fail, inner_delegate_calls=count, restored=True))
        result = dict(complete=True, cpu_only=True, actual_Finder_AST_executed=True, checks=checks,
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        out = base / "results/20261014/service_prepared_32g"
        out.mkdir(parents=True, exist_ok=True)
        (out / "CPU_IMPORT_CHAIN_AUDIT_32g.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        sys.meta_path = original


if __name__ == "__main__": main()
