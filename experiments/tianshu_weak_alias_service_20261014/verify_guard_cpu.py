# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual native eligibility guard, all negative guards and ceiling-map boundaries."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    base = Path(__file__).resolve().parent
    native = base.parent.parent.parent / "service-results/20261014_tianshu/topp_precision_maxseq64_service/20261014_topp_precision_maxseq64_service_tianshu/control/source_snapshot/iluvatar_prefill_buckets.py"
    module = load("_native_guard", native)
    assert hashlib.sha256(native.read_bytes()).hexdigest() == "966aad18aad80ef488928cafc8672cbfa2d547a832f9b151346f719ea3abc6db"
    load("_extend83", base.parent / "tianshu_budget8192_buckets_20261014/extend_buckets.py").extend(module)
    original = module.enabled
    probe = load("_piecewise_guard", base / "allow_piecewise.py")
    proof = probe.extend(module)
    config = NS(model_config=NS(hf_text_config=NS(model_type="llama", hidden_size=2048,
        intermediate_size=6144, num_hidden_layers=42, num_attention_heads=16, num_key_value_heads=2),
        quantization=None, is_multimodal_model=False, is_encoder_decoder=False),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, use_ubatching=False),
        scheduler_config=NS(max_num_batched_tokens=4096),
        compilation_config=NS(cudagraph_mode=NS(name="FULL_DECODE_ONLY")), lora_config=None, speculative_config=None)
    before_env = dict(os.environ)
    os.environ["VLLM_FL_ILUVATAR_PREFILL_BUCKETS"] = "1"
    try:
        checks = 0
        variants = [(None, None)]
        variants += [("model_config.hf_text_config." + key, value) for key, value in
                     [("model_type", "other"), ("hidden_size", 2049), ("intermediate_size", 6000),
                      ("num_hidden_layers", 41), ("num_attention_heads", 8), ("num_key_value_heads", 1)]]
        variants += [(key, value) for key, value in
                     [("model_config.quantization", "fp8"), ("model_config.is_multimodal_model", True),
                      ("model_config.is_encoder_decoder", True), ("parallel_config.tensor_parallel_size", 2),
                      ("parallel_config.pipeline_parallel_size", 2), ("parallel_config.data_parallel_size", 2),
                      ("parallel_config.use_ubatching", True), ("lora_config", object()),
                      ("speculative_config", object()), ("scheduler_config.max_num_batched_tokens", 8193),
                      ("scheduler_config.max_num_batched_tokens", 0)]]
        for mode in ("FULL_DECODE_ONLY", "FULL_AND_PIECEWISE", "FULL", "PIECEWISE", "NONE"):
            for path, value in variants:
                for vendor in ("iluvatar", "other"):
                    current = copy.deepcopy(config)
                    current.compilation_config.cudagraph_mode.name = mode
                    if path:
                        parent = current
                        parts = path.split(".")
                        for part in parts[:-1]: parent = getattr(parent, part)
                        setattr(parent, parts[-1], value)
                    actual = module.enabled(current, NS(vendor_name=vendor))
                    assert actual == (mode in ("FULL_DECODE_ONLY", "FULL_AND_PIECEWISE") and path is None and vendor == "iluvatar")
                    assert current.compilation_config.cudagraph_mode.name == mode
                    if mode != "FULL_AND_PIECEWISE": assert actual == original(current, NS(vendor_name=vendor))
                    checks += 1
        sizes = module.buckets_for_budget(4096)
        assert len(sizes) == 83 and sizes[-1] == 4096
        for value in range(1, 4097):
            assert module.padded_tokens(value, 4096) == min(x for x in sizes if x >= value)
        result = dict(complete=True, cpu_only=True, eligibility_cases=checks, ceiling_cases=4096,
                      real_native_and_extended_guard_executed=True, no_config_mutation=True,
                      proof=proof, sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path(__file__), base / "allow_piecewise.py", native)})
        out = base / "results/20261014/service_prepared_32g"
        out.mkdir(parents=True, exist_ok=True)
        (out / "CPU_GUARD_AUDIT_32g.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        os.environ.clear()
        os.environ.update(before_env)


if __name__ == "__main__": main()
