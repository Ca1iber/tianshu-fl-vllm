# SPDX-License-Identifier: Apache-2.0
"""Guarded full-Sampler CUDA Graph experiment; original algorithm and RNG."""
from dataclasses import replace

import torch


def neutral_processors(metadata):
    expected = {
        'argmax_invariant': {'MinPLogitsProcessor': ('min_p_count', 0)},
        'non_argmax_invariant': {
            'MinTokensLogitsProcessor': ('min_toks', {}),
            'LogitBiasLogitsProcessor': ('biases', {}),
        },
    }
    for group, known in expected.items():
        for processor in getattr(metadata.logitsprocs, group):
            cls = type(processor)
            if cls.__module__ != 'vllm.v1.sample.logits_processor.builtin' or cls.__name__ not in known:
                return False
            attribute, value = known[cls.__name__]
            if getattr(processor, attribute, None) != value:
                return False
    return True


def eligible(sampler, logits, metadata, predict_bonus_token=False, logprobs_mode_override=None):
    if predict_bonus_token or logprobs_mode_override is not None:
        return False
    if sampler.logprobs_mode != 'raw_logprobs' or sampler.use_fp64_gumbel:
        return False
    if getattr(sampler.topk_topp_sampler.forward, '__name__', None) != 'forward_native':
        return False
    if (logits.device.type != 'cuda' or logits.dtype != torch.bfloat16
            or tuple(logits.shape) != (64, 130560) or not logits.is_contiguous()):
        return False
    if (not metadata.all_random or metadata.all_greedy or not metadata.no_penalties
            or metadata.generators or metadata.max_num_logprobs is not None
            or metadata.logprob_token_ids or metadata.bad_words_token_ids
            or metadata.allowed_token_ids_mask is not None
            or metadata.thinking_budget_state_holder is not None
            or metadata.top_k is not None):
        return False
    for tensor in (metadata.temperature, metadata.top_p):
        if (tensor is None or tensor.device != logits.device or tensor.dtype != torch.float32
                or tuple(tensor.shape) != (64,) or not tensor.is_contiguous()):
            return False
    return neutral_processors(metadata) and not torch.cuda.is_current_stream_capturing()


class SamplerGraph:
    """One C64 graph; copy each dynamic input, return owned async-safe output.

    Capture and warmup preserve the default generator state. The capability
    probe must establish state/sequence equivalence on the actual backend.
    BF16 input is required because the original forward mutates its FP32 copy.
    """

    def __init__(self, sampler, original_forward, logits, metadata):
        assert eligible(sampler, logits, metadata)
        self.sampler = sampler
        self.original_forward = original_forward
        self.device = logits.device
        self.logits = torch.empty_like(logits)
        self.temperature = torch.empty_like(metadata.temperature)
        self.top_p = torch.empty_like(metadata.top_p)
        self.metadata = replace(metadata, temperature=self.temperature, top_p=self.top_p)
        self.copy_inputs(logits, metadata)
        saved_rng = torch.cuda.get_rng_state(self.device).clone()
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            with torch.cuda.stream(stream):
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
            torch.cuda.set_rng_state(saved_rng, self.device)
        self.replays = 0

    def copy_inputs(self, logits, metadata):
        self.logits.copy_(logits)
        self.temperature.copy_(metadata.temperature)
        self.top_p.copy_(metadata.top_p)

    def __call__(self, logits, metadata):
        assert eligible(self.sampler, logits, metadata)
        self.copy_inputs(logits, metadata)
        self.graph.replay()
        self.replays += 1
        # Graph storage is overwritten on the next replay. Keep earlier async
        # sampler outputs valid by returning a separate device allocation.
        from vllm.v1.outputs import SamplerOutput
        return SamplerOutput(sampled_token_ids=self.output.sampled_token_ids.clone(), logprobs_tensors=None)
