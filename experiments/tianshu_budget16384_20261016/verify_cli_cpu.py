# SPDX-License-Identifier: Apache-2.0
"""Execute real nested launch interception and inherited pre-I/O guards."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS


def main():
    base = Path(__file__).resolve().parent
    trial = base / "run_service_trial.py"
    inherited = base / "budget_trial.py"
    foundation = base / "foundation_service_trial.py"
    outer = next(n for n in ast.parse(trial.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    split = next(i for i, n in enumerate(outer.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "spec" for t in n.targets))
    outer.body = outer.body[:split] + [ast.Return(value=ast.Name(id="popen", ctx=ast.Load()))]
    outer_module = ast.fix_missing_locations(ast.Module(body=[outer], type_ignores=[]))
    inherited_tree = ast.parse(inherited.read_text())
    inherited_argv = next(n for n in ast.walk(inherited_tree) if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                 and t.value.id == "sys" and t.attr == "argv" for t in n.targets))
    fn = next(n for n in ast.parse(foundation.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    split = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "out" for t in n.targets))
    fn.body = fn.body[:split] + [ast.Return(value=ast.Call(func=ast.Name(id="vars", ctx=ast.Load()),
                                  args=[ast.Name(id="args", ctx=ast.Load())], keywords=[]))]
    foundation_module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    common = dict(argparse=argparse, Path=Path)
    foundation_scope = dict(common)
    exec(compile(foundation_module, str(foundation), "exec"), foundation_scope)
    runs = []
    for mode in ("control", "candidate"):
        root = Path("/tmp/unused_budget16384_cli_" + mode)
        assert not root.exists()
        sys.argv = [str(trial), "--output", str(root), "--mode", mode]
        scope = dict(common, __file__=str(trial), sys=sys,
                     subprocess=NS(Popen=lambda command, *a, **kw: (command, kw)))
        exec(compile(outer_module, str(trial), "exec"), scope)
        popen = scope["main"]()
        command = ["vllm", "serve", "/workspace/MiniCPM5-2B", "--max-num-batched-tokens", "4096"]
        actual, settings = popen(command, env=dict(VLLM_PLUGINS="fl"))
        assert actual == command and "--max-num-seqs" not in actual
        env = settings["env"]
        assert env["PYTHONPATH"] == str(base / "hooks")
        assert env["TIANSHU_CACHE_SERVICE_MODE"] == "candidate"
        assert env["TIANSHU_MAX_SEQS_MODE"] == "control"
        outer_argv = next(n for n in ast.walk(ast.parse(trial.read_text())) if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                 and t.value.id == "sys" and t.attr == "argv" for t in n.targets))
        argv_scope = dict(sys=NS(argv=[]), foundation=inherited, root=root, args=NS(mode=mode))
        exec(compile(ast.Module(body=[outer_argv], type_ignores=[]), str(trial), "exec"), argv_scope)
        assert argv_scope["sys"].argv[-2:] == ["--mode", mode]
        inherited_scope = dict(sys=NS(argv=[]), trial=foundation, root=root, budget=4096 if mode == "control" else 16384)
        exec(compile(ast.Module(body=[inherited_argv], type_ignores=[]), str(inherited), "exec"), inherited_scope)
        sys.argv = inherited_scope["sys"].argv
        args = foundation_scope["main"]()
        assert args["max_batched_tokens"] == (4096 if mode == "control" else 16384) and args["requests"] == args["concurrency"] == 64
        assert args["rounds"] == 3 and args["input_tokens"] == 4096 and args["output_tokens"] == 1024
        assert args["buckets"] == args["gather"] == args["native_prefill"] == "1"
        runs.append(dict(mode=mode, actual_wrapper_environment=env, actual_foundation_arguments=args))
    data = dict(complete=True, cpu_only=True, actual_nested_argv_and_launch_interception_checked=True,
                budgets4096_and16384=True, runs=runs,
                sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), trial, inherited, foundation)})
    out = base / "results/20261016/service_prepared_32g"
    out.mkdir(parents=True, exist_ok=True)
    (out / "CPU_CLI_AUDIT_32g.json").write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(json.dumps({key: value for key, value in data.items() if key != "runs"}))


if __name__ == "__main__":
    main()
