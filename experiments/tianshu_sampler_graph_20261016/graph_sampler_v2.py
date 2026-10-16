# SPDX-License-Identifier: Apache-2.0
"""Capture deterministic Sampler work; draw unchanged exponential RNG eagerly."""
from dataclasses import replace

import torch

from graph_sampler import eligible, neutral_processors


class SamplerGraph:
    """C64 original algorithm; native random draw stays outside CUDA Graph.

    FlagGems constructs Philox seed/offset in Python. Capturing that launch
    does not establish identical replay RNG semantics on this backend.
    A fresh original exponential_ draw before each replay avoids frozen RNG
    arguments while preserving the exact eager draw shape, dtype and offset.
    """

    def __init__(self, sampler, original_forward, logits, metadata):
        assert eligible(sampler, logits, metadata)
        self.sampler = sampler
        self.device = logits.device
        self.logits = torch.empty_like(logits)
        self.temperature = torch.empty_like(metadata.temperature)
        self.top_p = torch.empty_like(metadata.top_p)
        self.noise = torch.empty(logits.shape, dtype=torch.float32, device=self.device)
        self.metadata = replace(metadata, temperature=self.temperature, top_p=self.top_p)
        self.copy_inputs(logits, metadata)
        saved_rng = torch.cuda.get_rng_state(self.device).clone()
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        import vllm.v1.sample.ops.topk_topp_sampler as top_module
        original_random_sample = top_module.random_sample

        def fixed_noise_sample(probs, generators, use_fp64_gumbel=False):
            assert not generators and not use_fp64_gumbel
            assert probs.shape == self.noise.shape and probs.dtype == self.noise.dtype
            return top_module.sample_with_exponential_noise(probs, self.noise)

        try:
            top_module.random_sample = fixed_noise_sample
            with torch.cuda.stream(stream):
                # Compile the original random launch outside graph, then keep
                # capture deterministic. Restore this warmup draw in finally.
                self.noise.exponential_()
                for _ in range(3):
                    original_forward(sampler, self.logits, self.metadata)
            stream.synchronize()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = original_forward(sampler, self.logits, self.metadata)
            stream.synchronize()
            assert self.output.logprobs_tensors is None
            assert self.output.sampled_token_ids.dtype == torch.int32
            assert tuple(self.output.sampled_token_ids.shape) == (64, 1)
        finally:
            top_module.random_sample = original_random_sample
            torch.cuda.set_rng_state(saved_rng, self.device)
        self.replays = 0

    def copy_inputs(self, logits, metadata):
        self.logits.copy_(logits)
        self.temperature.copy_(metadata.temperature)
        self.top_p.copy_(metadata.top_p)

    def __call__(self, logits, metadata):
        assert eligible(self.sampler, logits, metadata)
        self.copy_inputs(logits, metadata)
        # Identical original RNG, executed once per real sample. No sampled
        # token or random draw is cached or reused across graph replays.
        self.noise.exponential_()
        self.graph.replay()
        self.replays += 1
        from vllm.v1.outputs import SamplerOutput
        return SamplerOutput(sampled_token_ids=self.output.sampled_token_ids.clone(), logprobs_tensors=None)
