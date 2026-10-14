# SPDX-License-Identifier: Apache-2.0
"""Guarded, sampler-local sort replacement with mmap route counters."""
import ast
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import struct
import sys

SAMPLER_SHA = "35b9a1ffeb1e50b73b0ddd1ffc0f968910ee21cf9cc5455dd879fde547b6ddc1"
CANDIDATE_SHA = "560f42b8a97ec005fc178166229cc6e6e18e9daa03153b99b53e5112e229591d"
COUNTER_BATCHES = 256


def supported(logits, torch):
    return (logits.ndim == 2 and logits.is_cuda and logits.dtype == torch.float32
            and logits.shape[1] == 130560 and 1 <= logits.shape[0] < 8
            and logits.stride(1) == 1 and logits.is_contiguous())


def transform(source):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "apply_top_k_top_p_pytorch")
    replaced = []

    class ReplaceSort(ast.NodeTransformer):
        def visit_Call(self, call):
            self.generic_visit(call)
            if (isinstance(call.func, ast.Attribute) and call.func.attr == "sort"
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "logits"):
                assert not call.args
                assert {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords} == dict(dim=-1, descending=False)
                call.func = ast.Name(id="_tianshu_sampler_sort", ctx=ast.Load())
                call.args = [ast.Name(id="logits", ctx=ast.Load())]
                replaced.append(call)
            return call

    ReplaceSort().visit(node)
    assert len(replaced) == 1
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


def install(module, mode, output, candidate_path):
    assert mode in ("control", "candidate")
    original_source = Path(module.__file__).read_bytes()
    assert hashlib.sha256(original_source).hexdigest() == SAMPLER_SHA
    candidate_source = candidate_path.read_bytes()
    assert hashlib.sha256(candidate_source).hexdigest() == CANDIDATE_SHA
    output.mkdir(parents=True, exist_ok=True)
    path = output / ("counts_" + str(os.getpid()) + ".bin")
    with path.open("x+b") as descriptor:
        descriptor.truncate(3 * COUNTER_BATCHES * 8)
        counts = mmap.mmap(descriptor.fileno(), 0)

    def increment(group, batch):
        index = (group * COUNTER_BATCHES + min(batch, COUNTER_BATCHES - 1)) * 8
        struct.pack_into("<Q", counts, index, struct.unpack_from("<Q", counts, index)[0] + 1)

    candidate_module = None
    original_pytorch = module.apply_top_k_top_p_pytorch
    patched_pytorch = None
    transformed = transform(original_source.decode())
    if mode == "candidate":
        name = "_tianshu_sampler_sweep_repeat_" + str(os.getpid())
        spec = importlib.util.spec_from_file_location(name, candidate_path)
        candidate_module = importlib.util.module_from_spec(spec)
        sys.modules[name] = candidate_module
        spec.loader.exec_module(candidate_module)

        def sort(logits, dim, descending):
            assert dim == -1 and descending is False
            return candidate_module.sort_stable(logits, stable=False, dim=dim, descending=descending)

        namespace = dict(module.__dict__, _tianshu_sampler_sort=sort)
        exec(compile(transformed, str(Path(__file__).resolve()) + ":sampler_sort", "exec"), namespace)
        patched_pytorch = namespace["apply_top_k_top_p_pytorch"]

    def pytorch(logits, k, p, allow_cpu_sync=False):
        increment(1, logits.shape[0])
        if mode == "candidate" and supported(logits, module.torch):
            increment(2, logits.shape[0])
            return patched_pytorch(logits, k, p, allow_cpu_sync=allow_cpu_sync)
        return original_pytorch(logits, k, p, allow_cpu_sync=allow_cpu_sync)

    original_apply = module.apply_top_k_top_p

    def apply(logits, k, p):
        increment(0, logits.shape[0])
        return original_apply(logits, k, p)

    module.apply_top_k_top_p_pytorch = pytorch
    module.apply_top_k_top_p = apply
    provenance = dict(mode=mode, pid=os.getpid(), sampler_sha256=SAMPLER_SHA,
        candidate_sha256=CANDIDATE_SHA, helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        changed_sort_calls=1, unchanged_function_ast_except_sort_call=True,
        transformed_function_ast_sha256=hashlib.sha256(ast.dump(transformed).encode()).hexdigest(),
        supported=dict(batch_min=1, batch_max=7, vocab=130560, dtype="torch.float32", contiguous=True),
        count_groups=["apply_top_k_top_p", "pytorch_fallback", "candidate_sort"],
        count_batch_slots=COUNTER_BATCHES, overflow_batch_slot=COUNTER_BATCHES-1,
        count_scope="entire worker lifecycle including startup, canaries and all rounds",
        original_native_topp_function_unchanged=True, sampling_parameters_and_RNG_unchanged=True,
        installed_sources_modified=False)
    (output / ("route_" + str(os.getpid()) + ".json")).write_text(json.dumps(provenance, indent=2) + "\n")
    # Closures keep the mmap alive until the worker exits; no per-call filesystem I/O.
    return provenance
