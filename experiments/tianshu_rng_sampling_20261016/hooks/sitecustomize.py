# SPDX-License-Identifier: Apache-2.0
"""Compose original-command CV/KV16/S8 with the independently probed RNG kernel."""
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
runpy.run_path(str(BASE / 'tianshu_max_seqs_20261014/hooks/sitecustomize.py'))
SAMPLER_SHA = '35b9a1ffeb1e50b73b0ddd1ffc0f968910ee21cf9cc5455dd879fde547b6ddc1'
KERNEL_SHA = 'a60c31b5aa9b2df5a8f7858dda6e17ae4a8c4e71ee48470298f842774c721e60'


class SamplerLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == SAMPLER_SHA
        path = BASE / 'tianshu_rng_sampling_20261016/experimental_rng_sampler_v2.py'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == KERNEL_SHA
        spec = importlib.util.spec_from_file_location('tianshu_rng_sampling_kernel', path)
        kernel = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = kernel
        spec.loader.exec_module(kernel)
        import torch
        original = module.TopKTopPSampler.forward_native
        recorded = set()

        def record(batch, route, generators):
            key = (batch, route)
            directory = os.environ['TIANSHU_RNG_SAMPLER_RECORD_DIR']
            if key in recorded:
                return
            recorded.add(key)
            out = Path(directory)
            out.mkdir(parents=True, exist_ok=True)
            (out / f'sampler_{os.getpid()}_{batch}_{route}.json').write_text(json.dumps(dict(
                pid=os.getpid(), batch=batch, route=route, seeded_rows=len(generators),
                sampler_source_sha256=SAMPLER_SHA, kernel_sha256=KERNEL_SHA,
                top_k_top_p_unchanged=True, global_random_sequence_changed=True,
                request_generators_use_original=True, capture_uses_original=True), indent=2) + '\n')

        def forward(self, logits, generators, k, p):
            from vllm.platforms import current_platform
            if getattr(current_platform, 'vendor_name', '') != 'iluvatar':
                return original(self, logits, generators, k, p)
            if not kernel.eligible(logits, generators, self.use_fp64_gumbel):
                if generators:
                    record(logits.shape[0], 'seeded_fallback', generators)
                return original(self, logits, generators, k, p)
            logits = module.apply_top_k_top_p(logits, k, p)
            returned = None
            if self.logprobs_mode == 'processed_logits':
                returned = logits
            elif self.logprobs_mode == 'processed_logprobs':
                returned = logits.log_softmax(dim=-1, dtype=torch.float32)
            result = kernel.sample_global(logits)
            record(logits.shape[0], 'fused', generators)
            return result, returned

        module.TopKTopPSampler.forward_native = forward


class SamplerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm.v1.sample.ops.topk_topp_sampler':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise RuntimeError('vLLM sampling module unavailable')
        spec.loader = SamplerLoader(spec.loader)
        return spec


if os.environ.get('TIANSHU_RNG_SAMPLER_PROBE') == '1':
    sys.meta_path.insert(0, SamplerFinder())
