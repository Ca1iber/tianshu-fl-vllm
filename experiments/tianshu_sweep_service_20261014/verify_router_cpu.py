# SPDX-License-Identifier: Apache-2.0
"""Check exact source transformation and exercise the installed routing AST."""
import ast
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch


def main():
    base = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("_sweep_cpu_helper", base / "sampling_sort.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source_path = base / "reference_sources/topk_topp_sampler.py"
    raw = source_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == helper.SAMPLER_SHA
    tree = ast.parse(raw)
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "apply_top_k_top_p_pytorch")
    transformed = helper.transform(raw.decode())

    class Undo(ast.NodeTransformer):
        def visit_Call(self, call):
            self.generic_visit(call)
            if isinstance(call.func, ast.Name) and call.func.id == "_tianshu_sampler_sort":
                assert isinstance(call.args[0], ast.Name) and call.args[0].id == "logits"
                call.func = ast.Attribute(value=call.args.pop(0), attr="sort", ctx=ast.Load())
            return call

    Undo().visit(transformed)
    assert ast.dump(original) == ast.dump(transformed.body[0])
    batches = (1, 4, 7, 8, 64)
    sentinel_dtype = object()
    results = []

    class Tensor:
        def __init__(self, batch=1, dtype=sentinel_dtype, contiguous=True, cuda=True, n=130560):
            self.shape = (batch, n)
            self.dtype, self.is_cuda, self.ndim = dtype, cuda, 2
            self.contiguous = contiguous

        def stride(self, dim): return 1
        def is_contiguous(self): return self.contiguous
        def unsqueeze(self, dim): return self
        def softmax(self, dim): return self
        def __le__(self, other): return self
        def __rsub__(self, other): return self
        def __setitem__(self, item, value): pass
        def masked_fill_(self, mask, value): return self
        def scatter_(self, dim, index, src): return self

    class CandidateLoader:
        def create_module(self, spec): return None
        def exec_module(self, module):
            module.sort_stable = lambda logits, **kwargs: (logits, Tensor())

    apply_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_top_k_top_p")
    candidate_path = base.parent / "tianshu_sort_sweep_20261014/reference_sources/metax_repeat_digit.py"
    with tempfile.TemporaryDirectory(prefix="tianshu_sweep_cpu_") as temp:
        for mode in ("control", "candidate"):
            module = ModuleType("_fake_sampler_" + mode)
            module.__file__ = str(source_path)
            module.torch = NS(Tensor=Tensor, float32=sentinel_dtype, cumsum=lambda values, *a, **kw: values)
            module.current_platform = NS(is_cpu=lambda: False)
            module.HAS_TRITON = True
            module.apply_top_k_top_p_triton = lambda logits, k, p: "native"
            native = module.apply_top_k_top_p_triton
            module.apply_top_k_top_p_pytorch = lambda *a, **kw: "original"
            exec(compile(ast.Module(body=[apply_node], type_ignores=[]), str(source_path), "exec"), module.__dict__)
            output = Path(temp) / mode
            with patch.object(helper.importlib.util, "spec_from_file_location",
                              side_effect=lambda name, path: importlib.machinery.ModuleSpec(name, CandidateLoader())):
                helper.install(module, mode, output, candidate_path)
            assert module.apply_top_k_top_p_triton is native
            inputs = [Tensor(b) for b in batches]
            inputs += [Tensor(1, dtype="wrong"), Tensor(1, contiguous=False), Tensor(1, cuda=False), Tensor(1, n=128)]
            expected_candidates = 0
            for tensor in inputs:
                value = module.apply_top_k_top_p(tensor, None, Tensor())
                if tensor.shape[0] >= 8:
                    assert value == "native"
                elif mode == "candidate" and helper.supported(tensor, module.torch):
                    assert value is tensor
                    expected_candidates += 1
                else:
                    assert value == "original"
            counter = next(output.glob("counts_*.bin")).read_bytes()
            values = struct.unpack("<768Q", counter)
            assert sum(values[:256]) == len(inputs)
            assert sum(values[256:512]) == len(inputs) - 2
            assert sum(values[512:]) == expected_candidates
            assert sum(values[520:]) == 0
            results.append(dict(mode=mode, real_routing_AST_calls=len(inputs), candidate_sort_calls=expected_candidates,
                                native_function_identity_preserved=True, mmap_counts_verified=True))
    data = dict(complete=True, cpu_only=True, numeric_equivalence_not_claimed=True,
                exact_installed_function_AST_preserved_except_one_sort_call=True, runs=results,
                sources={str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__), base / "sampling_sort.py", source_path,
                                   base / "run_service_trial.py", base / "run_service_pair.py", base / "hooks/sitecustomize.py")})
    out = base / "results/20261014/service_prepared_32g"
    out.mkdir(parents=True, exist_ok=True)
    (out / "CPU_ROUTING_AUDIT_32g.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
