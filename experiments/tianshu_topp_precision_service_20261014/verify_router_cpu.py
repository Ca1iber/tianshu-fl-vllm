# SPDX-License-Identifier: Apache-2.0
"""Run real hook dispatch and Loader routing with explicit CPU kernel fixtures."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace as NS


def main():
    base = Path(__file__).resolve().parent
    hook = base / "hooks/top_p_sitecustomize.py"
    native = base.parent.parent.parent / "service-results/20261014_tianshu/topp_precision/20261014_topp_precision_tianshu/micro/source_snapshot/native_topk_topp_triton.py"
    assert native.exists()
    original_run, original_meta, original_env = runpy.run_path, list(sys.meta_path), dict(os.environ)
    calls = []

    def fake_run(path, *args, **kwargs):
        calls.append(str(path))
        if str(path).endswith("tianshu_decode_cache_service_20261014/hooks/decode_sitecustomize.py"):
            return {}
        if str(path).endswith("tianshu_budget8192_buckets_20261014/hooks/sitecustomize.py"):
            return runpy.run_path(str(base.parent / "tianshu_decode_kv16_w2_20261014/hooks/sitecustomize.py"))
        return original_run(path, *args, **kwargs)

    try:
        runpy.run_path = fake_run
        original_run(str(base / "hooks/sitecustomize.py"))
        assert runpy.run_path is fake_run
        assert str(hook) in calls
        assert any(p.endswith("tianshu_decode_cache_service_20261014/hooks/decode_sitecustomize.py") for p in calls)
        tree = ast.parse(hook.read_text())
        tree.body = [n for n in tree.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Call)
                     or not isinstance(n.value.func, ast.Attribute) or not isinstance(n.value.func.value, ast.Name)
                     or n.value.func.value.id != "runpy"]
        scope = dict(__file__=str(hook))
        exec(compile(tree, str(hook), "exec"), scope)
        calls.clear()

        class Kernel:
            def __init__(self, name): self.name = name
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    calls.append(dict(kernel=self.name, grid=grid, kwargs=kwargs))
                    return NS(metadata=NS(num_warps=kwargs.get("num_warps"), num_stages=kwargs.get("num_stages"), shared=0))
                return launch

        class Original:
            def exec_module(self, module): module._topk_topp_kernel = Kernel("native")

        tensor = NS(dtype="torch.float32", stride=lambda dim: 1)
        real_spec = importlib.util.spec_from_file_location
        helper_spec = real_spec("_cpu_precision_repair", base.parent / "tianshu_topp_precision_20261014/precision.py")
        repair = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(repair)

        def fake_load(native_path, directory):
            directory.mkdir(parents=True)
            generated = directory / "generated_topp_precision.py"
            generated.write_text(repair.render_source(Path(native_path).read_bytes()))
            return Kernel("corrected"), dict(generated_sha256=hashlib.sha256(generated.read_bytes()).hexdigest())

        class FakeHelperLoader:
            def create_module(self, spec): return None
            def exec_module(self, module): module.load_kernel = fake_load

        def fake_spec(name, path, *args, **kwargs):
            if name == "_precision_helper":
                from importlib.machinery import ModuleSpec
                return ModuleSpec(name, FakeHelperLoader())
            return real_spec(name, path, *args, **kwargs)

        importlib.util.spec_from_file_location = fake_spec
        try:
            with tempfile.TemporaryDirectory(prefix="tianshu_precision_router_") as temp:
                records = []
                for mode in ("control", "candidate"):
                    os.environ.update(TIANSHU_TOPP_PRECISION_SERVICE_MODE=mode,
                                      TIANSHU_TOPP_LAUNCH_RECORD_DIR=str(Path(temp) / mode))
                    module = NS(__file__=str(native))
                    scope["Loader"](Original()).exec_module(module)
                    for batch, topk, topp in [(64, False, True), (32, True, True), (7, False, True), (8, False, False)]:
                        kwargs = dict(VOCAB_SIZE=130560, BATCH_SIZE=batch, BLOCK_SIZE=8192,
                                      BLOCK_SIZE_TRUNC=4096, TOPK_ENABLED=topk, TOPP_ENABLED=topp)
                        module._topk_topp_kernel[(1,)](tensor, **kwargs)
                        active = mode == "candidate" and batch >= 8 and topp and not topk
                        actual = calls[-1]
                        assert actual["kernel"] == ("corrected" if active else "native")
                        assert actual["kwargs"]["BLOCK_SIZE"] == (16384 if active else 8192)
                        assert actual["kwargs"]["BLOCK_SIZE_TRUNC"] == (8192 if active else 4096)
                        if batch >= 8: assert actual["kwargs"]["num_warps"] == 16 and actual["kwargs"]["num_stages"] == 1
                        records.append(dict(mode=mode, batch=batch, topk=topk, topp=topp, corrected=active))
                    emitted = [json.loads(p.read_text()) for p in (Path(temp) / mode).glob("route_*.json")]
                    assert len(emitted) == 3
                    assert all(r["body_changed"] == r["corrected_source_active"] for r in emitted)
        finally:
            importlib.util.spec_from_file_location = real_spec
        result = dict(complete=True, cpu_only=True, fixtures_are_not_GPU_evidence=True,
                      actual_nested_hook_dispatch_and_restoration_verified=True,
                      actual_loader_routes_and_records_verified=True, routes=records)
        out = base / "results/20261014/service_prepared_32g"
        out.mkdir(parents=True, exist_ok=True)
        (out / "CPU_ROUTER_AUDIT_32g.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        runpy.run_path = original_run
        sys.meta_path[:] = original_meta
        os.environ.clear()
        os.environ.update(original_env)


if __name__ == "__main__": main()
