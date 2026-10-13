# SPDX-License-Identifier: Apache-2.0
"""Skip full-vocabulary FP32 materialization only for unrestricted greedy batches."""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / "tianshu_decode_low_warps_20261014/hooks/sitecustomize.py"))
SOURCE_SHA = "1a2f18ddc91c89f3744ab5a93624be056bf20b7a8b2e6e13d0236182718a2253"


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == SOURCE_SHA
        implementation = BASE / "tianshu_greedy_bf16_20261014/implementation.py"
        spec = importlib.util.spec_from_file_location("_tianshu_greedy_bf16_impl", implementation)
        code = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = code
        spec.loader.exec_module(code)
        original = module.Sampler.forward
        mode = os.environ["TIANSHU_GREEDY_ARGMAX_MODE"]
        assert mode in ("control", "candidate")
        marker = Path(os.environ["TIANSHU_GREEDY_REQUEST_MARKER"])
        root = Path(os.environ["TIANSHU_GREEDY_ARGMAX_RECORD_DIR"])
        seen = set()

        def forward(self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None):
            supported = code.supported(logits, sampling_metadata)
            selected = supported and mode == "candidate"
            if selected:
                output = module.SamplerOutput(sampled_token_ids=code.sample_direct(logits), logprobs_tensors=None)
            else:
                output = original(self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override)
            phase = "request" if marker.exists() else "warmup"
            key = phase, tuple(logits.shape), tuple(logits.stride()), supported, selected
            if key not in seen:
                seen.add(key)
                root.mkdir(parents=True, exist_ok=True)
                record = dict(phase=phase, batch=logits.shape[0], vocab=logits.shape[-1],
                              dtype=str(logits.dtype), stride=list(logits.stride()), supported=supported,
                              selected=selected, mode=mode,
                              configuration=dict(tile=1024 if logits.shape[0] == 1 else 2048, warps=4) if selected else None,
                              output_dtype=str(output.sampled_token_ids.dtype), output_shape=list(output.sampled_token_ids.shape),
                              all_greedy=sampling_metadata.all_greedy, no_penalties=sampling_metadata.no_penalties,
                              max_num_logprobs=sampling_metadata.max_num_logprobs,
                              actual_sampler_sha256=SOURCE_SHA,
                              implementation_sha256=hashlib.sha256(implementation.read_bytes()).hexdigest())
                (root / f"route_{os.getpid()}_{len(seen)}.json").write_text(json.dumps(record, indent=2) + "\n")
            return output

        module.Sampler.forward = forward


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm.v1.sample.sampler":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


if os.environ.get("TIANSHU_GREEDY_ARGMAX_PROBE") == "1":
    sys.meta_path.insert(0, Finder())
