# SPDX-License-Identifier: Apache-2.0
"""Execute actual loader/Finder AST: no recursion, trace preservation, correct deferred helper."""
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch


def main():
    base = Path(__file__).resolve().parent
    source = base / "hooks/sitecustomize.py"
    namespace = dict(__file__=str(source), BASE=base.parent, importlib=importlib,
        hashlib=hashlib, json=json, os=os, Path=Path, runpy=runpy, sys=sys)
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),namespace)
    finder = namespace["Finder"]()
    old_path = base.parent / "tianshu_budget8192_buckets_20261014/hooks/sitecustomize.py"
    old_namespace = {}
    exec(compile("class OldFinder:\n def find_spec(self,*args): raise AssertionError('old extender must be excluded')",str(old_path),"exec"),old_namespace)
    old = old_namespace["OldFinder"]()
    original_path = base.parents[1] / "vllm_fl/worker/iluvatar_prefill_buckets.py"
    original_meta = list(sys.meta_path)
    outcomes = []
    try:
        for mode in ("success", "missing", "error"):
            spec = importlib.util.spec_from_file_location("_budget16384_real_native",original_path)
            native_loader = spec.loader
            class Tracer:
                def find_spec(self, fullname, path=None, target=None):
                    assert finder not in sys.meta_path and old not in sys.meta_path
                    if mode == "error": raise ValueError("fixture failure")
                    return spec if mode == "success" else None
            trace = Tracer()
            previous = [finder, old, trace]
            sys.meta_path[:] = previous
            try:
                wrapped = finder.find_spec("vllm_fl.worker.iluvatar_prefill_buckets")
                assert mode == "success" and wrapped.loader.original is native_loader
            except (ValueError, RuntimeError):
                assert mode != "success"
            assert sys.meta_path == previous
            if mode == "success":
                sys.meta_path[:] = original_meta
                with tempfile.TemporaryDirectory(prefix="budget16384_loader_") as temp:
                    with patch.dict(os.environ, TIANSHU_EXTENDED_BUCKET_RECORD_DIR=temp):
                        module = importlib.util.module_from_spec(spec)
                        wrapped.loader.exec_module(module)
                    record = json.loads(next(Path(temp).glob("extended_*.json")).read_text())
                    assert len(module.PREFILL_BUCKETS) == 131 and module.PREFILL_BUCKETS[-1] == 16384
                    assert record["changed_constants"] == [[2048,16384]]
                    assert record["helper_sha256"] == hashlib.sha256((base/"extend_buckets.py").read_bytes()).hexdigest()
                    assert module.padded_tokens(15001,16384) == 15104
            outcomes.append(mode)
        sys.meta_path[:] = [finder,old]
        assert finder.find_spec("unrelated.module") is None
        assert sys.meta_path == [finder,old]
        outcomes.append("unrelated")
    finally:
        sys.meta_path[:] = original_meta
    result = dict(complete=True,cpu_only=True,actual_Loader_and_Finder_AST_executed=True,
        old_extender_excluded=True,tracing_loader_preserved=True,meta_path_restored_all_branches=True,
        actual_deferred_helper_generates131_points=True,actual_helper_SHA_verified=True,runs=outcomes,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),GPU_started=False)
    out = base / "results/20261016/prepared_32g";out.mkdir(parents=True,exist_ok=True)
    (out/"CPU_DEFERRED_LOADER_AUDIT_32g.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__": main()
