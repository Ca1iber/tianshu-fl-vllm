# SPDX-License-Identifier: Apache-2.0
"""Exercise the real metadata loader against worker fixtures without GPU imports."""
import ast
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch


def main():
    base = Path(__file__).resolve().parent
    source = base / "hooks/sitecustomize.py"
    loader_class = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.ClassDef) and node.name == "Loader")
    scope = dict(__file__=str(source), wraps=wraps, hashlib=hashlib, json=json,
                 os=os, Path=Path, importlib=NS(abc=NS(Loader=object)))
    exec(compile(ast.Module(body=[loader_class], type_ignores=[]), str(source), "exec"), scope)
    results = []
    class OriginalLoader:
        def exec_module(self, module):
            class WorkerFL:
                def __init__(self, vllm_config, *args, **kwargs):
                    self.vllm_config = vllm_config
                    self.initialized = True
            module.WorkerFL = WorkerFL
    with tempfile.TemporaryDirectory(prefix="tianshu_maxseq_worker_cpu_") as temporary:
        for mode, actual, should_pass in (("control",256,True),("control",128,True),
                                         ("candidate",64,True),("candidate",128,False)):
            root = Path(temporary) / (mode + str(actual))
            module = ModuleType("_worker_fixture")
            module.__file__ = str(base.parent.parent / "vllm_fl/worker/worker.py")
            loader = scope["Loader"](OriginalLoader())
            config = NS(scheduler_config=NS(max_num_seqs=actual, max_num_batched_tokens=4096, async_scheduling=True),
                        model_config=NS(max_model_len=131072),
                        compilation_config=NS(cudagraph_capture_sizes=[1,2,4,8,16,32,64]))
            with patch.dict(os.environ, TIANSHU_MAX_SEQS_MODE=mode, TIANSHU_MAX_SEQS_RECORD_DIR=str(root)):
                loader.exec_module(module)
                try: worker = module.WorkerFL(config, 0, rank=0)
                except AssertionError:
                    assert not should_pass and not root.exists()
                else:
                    assert should_pass and worker.initialized
                    record = json.loads(next(root.glob("worker_*.json")).read_text())
                    assert record["max_num_seqs"] == actual and record["mode"] == mode
                    assert record["max_num_batched_tokens"] == 4096 and record["model_max_len"] == 131072
                    assert record["async_scheduling"] is True and record["cudagraph_capture_sizes"] == [1,2,4,8,16,32,64]
                    assert record["hook_source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
                    assert record["worker_source_sha256"] == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                    assert hasattr(module.WorkerFL.__init__, "__wrapped__")
            results.append(dict(mode=mode, actual_max_num_seqs=actual, expected_pass=should_pass))
    output = dict(complete=True, cpu_only=True, actual_loader_AST_exercised=True,
                  original_worker_constructor_preserved=True, wrong_candidate_limit_rejected=True,
                  numerical_or_performance_claim=False, runs=results,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), source)})
    root = base / "results/20261014/prepared_32g"
    root.mkdir(parents=True, exist_ok=True)
    (root / "CPU_WORKER_AUDIT_32g.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
