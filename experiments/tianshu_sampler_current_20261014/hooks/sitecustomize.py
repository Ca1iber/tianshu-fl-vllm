# SPDX-License-Identifier: Apache-2.0
"""Compose the current CV/KV16/S8 profile with the previously checked sampler fusion."""
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
KERNEL_SHA = '83739846d48d6e12710d9cdbb0da61debb733241e8aa45e1ea1317bdd2950e93'


class SamplerLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SAMPLER_SHA:
            raise RuntimeError('Sampling experiment requires the inspected vLLM source')
        source = BASE / 'tianshu_sampler_fusion_20261012/experimental_sampler.py'
        if hashlib.sha256(source.read_bytes()).hexdigest() != KERNEL_SHA:
            raise RuntimeError('Sampling kernel differs from the 105 checked cases')
        spec = importlib.util.spec_from_file_location('tianshu_sampling_kernel', source)
        kernel = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = kernel
        spec.loader.exec_module(kernel)
        import torch
        original = module.TopKTopPSampler.forward_native
        recorded = set()

        def forward(self, logits, generators, k, p):
            from vllm.platforms import current_platform
            if (getattr(current_platform, 'vendor_name', '') != 'iluvatar'
                    or logits.ndim != 2 or logits.shape[1] != 130560
                    or logits.dtype != torch.float32 or logits.stride(1) != 1
                    or logits.device.type != 'cuda' or self.use_fp64_gumbel
                    or logits.shape[0] == 0):
                return original(self, logits, generators, k, p)
            # Temperature scaling happens upstream. Preserve filtering, optional
            # logprobs, and the original global/per-request exponential draws.
            logits = module.apply_top_k_top_p(logits, k, p)
            to_return = None
            if self.logprobs_mode == 'processed_logits':
                to_return = logits
            elif self.logprobs_mode == 'processed_logprobs':
                to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
            q = module.empty_exponential_noise_like(logits, False)
            if len(generators) != logits.shape[0]:
                q.exponential_()
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
            result = kernel.sample_with_noise(logits, q, kernel.prepare(logits), 2048)
            directory = os.environ.get('TIANSHU_SAMPLER_RECORD_DIR')
            batch = logits.shape[0]
            if directory and batch not in recorded:
                recorded.add(batch)
                out = Path(directory)
                out.mkdir(parents=True, exist_ok=True)
                (out / f'sampler_{os.getpid()}_{batch}.json').write_text(json.dumps({
                    'pid': os.getpid(), 'batch': batch, 'vocab': logits.shape[1],
                    'sampler_source_sha256': SAMPLER_SHA,
                    'kernel_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'top_k_top_p_unchanged': True, 'rng_generation_unchanged': True,
                    'seeded_rows': len(generators),
                    'bitwise_for_every_float_input_claimed': False}, indent=2) + '\n')
            return result, to_return

        module.TopKTopPSampler.forward_native = forward


class SamplerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm.v1.sample.ops.topk_topp_sampler':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise RuntimeError('vLLM sampler module unavailable')
        spec.loader = SamplerLoader(spec.loader)
        return spec


if os.environ.get('TIANSHU_SAMPLER_FUSION_PROBE') == '1':
    sys.meta_path.insert(0, SamplerFinder())
