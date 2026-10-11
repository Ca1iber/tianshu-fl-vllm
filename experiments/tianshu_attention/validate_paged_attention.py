#!/usr/bin/env python3
"""BI-V150 BF16/GQA attention checks and short paired device timings.

Uses the unchanged FlagGems v5.3.5 paged-attention tolerance (atol=.02,
rtol=.01). This is an operator check, not a MATH-500 or scored benchmark.
Run on an otherwise idle GPU before starting a model service.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import vllm_fl  # platform compatibility initialization before importing vLLM
from flag_gems import flash_attn_varlen_func
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm_fl.dispatch.backends.flaggems.impl.iluvatar_mixed_attention import flash_attn_split_mixed
from vllm_fl.dispatch.backends.flaggems.impl.attention import AttentionFLMetadataBuilder, AttentionFLImpl
from vllm.v1.attention.backend import AttentionType


def reference(q, k, v, table, q_lens, kv_lens):
    outputs = []
    start = 0
    for row, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        pages = table[row, :(kl + 15) // 16].long()
        keys = k[pages].reshape(-1, 2, 128)[:kl].float().repeat_interleave(8, dim=1)
        values = v[pages].reshape(-1, 2, 128)[:kl].float().repeat_interleave(8, dim=1)
        queries = q[start:start + ql].float().transpose(0, 1)
        # Explicit bottom-right causal mask; q may be a chunk after a KV prefix.
        pieces = []
        for offset in range(0, ql, 64):
            chunk = queries[:, offset:offset + 64]
            scores = torch.matmul(chunk, keys.permute(1, 2, 0)) * (128 ** -0.5)
            positions = torch.arange(offset, offset + chunk.shape[1], device=q.device)
            allowed = torch.arange(kl, device=q.device)[None, :] <= (kl - ql + positions[:, None])
            scores.masked_fill_(~allowed[None], float('-inf'))
            pieces.append(torch.matmul(scores.softmax(-1), values.transpose(0, 1)))
        outputs.append(torch.cat(pieces, dim=1).transpose(0, 1))
        start += ql
    return torch.cat(outputs).to(q.dtype)


def case(name, q_lens, kv_lens, strided, seed):
    torch.manual_seed(seed)
    heads, kv_heads, dim, block = 16, 2, 128, 16
    width = max((n + block - 1) // block for n in kv_lens)
    pages = len(q_lens) * width
    table = torch.randperm(pages, device='cuda', dtype=torch.int64).to(torch.int32).view(len(q_lens), width)
    # Simulate Q sliced from a combined QKV projection and noncontiguous pages.
    q = torch.randn(sum(q_lens), heads + (4 if strided else 0), dim, device='cuda', dtype=torch.bfloat16)[:, :heads]
    k = torch.randn(pages * (2 if strided else 1), block, kv_heads, dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    if strided:
        k, v = k[::2], v[::2]
    cu = torch.tensor([0] + list(__import__('itertools').accumulate(q_lens)), device='cuda', dtype=torch.int32)
    used = torch.tensor(kv_lens, device='cuda', dtype=torch.int32)
    original, direct, candidate, split = [torch.empty_like(q) for _ in range(4)]
    args = dict(q=q, k=k, v=v, cu_seqlens_q=cu, max_seqlen_q=max(q_lens),
                seqused_k=used, max_seqlen_k=max(kv_lens), softmax_scale=dim ** -0.5,
                causal=True, window_size=(-1, -1), block_table=table, softcap=0.0)

    def old():
        unified_attention(**args, out=original, q_descale=None, k_descale=None, v_descale=None)

    def new():
        os.environ['FLAGGEMS_ILUVATAR_COMPACT_VARLEN'] = '1'
        flash_attn_varlen_func(**args, out=candidate, num_splits=0, fa_version=2)

    def unmodified_flash():
        os.environ['FLAGGEMS_ILUVATAR_COMPACT_VARLEN'] = '0'
        flash_attn_varlen_func(**args, out=direct, num_splits=0, fa_version=2)

    num_decodes = 0
    for length in q_lens:
        if length != 1:
            break
        num_decodes += 1
    can_split = 0 < num_decodes < len(q_lens) and all(n > 1 for n in q_lens[num_decodes:])
    prefill_starts = cu[num_decodes:] - num_decodes if can_split else None

    def split_mixed():
        os.environ['FLAGGEMS_ILUVATAR_COMPACT_VARLEN'] = '1'
        if can_split:
            flash_attn_split_mixed(**args, out=split, num_decodes=num_decodes,
                                  prefill_cu_seqlens_q=prefill_starts)
        else:
            flash_attn_varlen_func(**args, out=split, num_splits=0, fa_version=2)

    old()
    unmodified_flash()
    new()
    split_mixed()
    expected = reference(q, k, v, table, q_lens, kv_lens)
    torch.cuda.synchronize()
    errors = {}
    for label, value in [('original', original), ('direct_flaggems', direct), ('candidate', candidate), ('split_mixed', split)]:
        diff = (value.float() - expected.float()).abs()
        errors[label] = dict(max_abs=diff.max().item(), mean_abs=diff.mean().item())
        print(json.dumps({'case': name, 'backend': label, 'errors': errors[label],
                          'first_output': value[0, 0, :8].tolist(),
                          'first_expected': expected[0, 0, :8].tolist(),
                          'first_value': v[table[0, 0].long(), 0, 0, :8].tolist()}), flush=True)
    for label, value in [('original', original), ('direct_flaggems', direct), ('candidate', candidate), ('split_mixed', split)]:
        if not torch.isfinite(value).all():
            raise AssertionError(f'{name}: {label} nonfinite output')
        try:
            torch.testing.assert_close(value, expected, atol=2e-2, rtol=1e-2,
                                       msg=lambda msg: f'{name}: {label}: {msg}')
            errors[label]['reference_check_passed'] = True
        except AssertionError as error:
            errors[label]['reference_check_passed'] = False
            errors[label]['failure'] = str(error)
            # Preserve any baseline discrepancy, but gate the candidate on the
            # independent FP32 reference rather than a possibly incorrect baseline.
            if label != 'original':
                raise
    torch.testing.assert_close(candidate, direct, atol=0, rtol=0)
    # Exercise the production metadata builder and forward adapter, including
    # fallback for unreordered input and preservation of the padded output.
    builder = object.__new__(AttentionFLMetadataBuilder)
    builder.aot_schedule = builder.use_full_cuda_graph = False
    builder.aot_sliding_window = (-1, -1)
    builder.dcp_world_size = 1
    builder.max_num_splits = 0
    builder.max_cudagraph_size = 512
    builder.split_mixed_enabled = True
    builder._split_logged = False
    common = SimpleNamespace(num_reqs=len(q_lens), num_actual_tokens=sum(q_lens),
                             max_query_len=max(q_lens), max_seq_len=max(kv_lens),
                             query_start_loc=cu, seq_lens=used, block_table_tensor=table,
                             slot_mapping=None, causal=True,
                             query_start_loc_cpu=torch.tensor([0] + list(__import__('itertools').accumulate(q_lens)), dtype=torch.int32))
    metadata = builder.build(0, common)
    assert metadata.split_num_decodes == (num_decodes if can_split else 0)
    impl = object.__new__(AttentionFLImpl)
    impl.attn_type = AttentionType.DECODER
    impl.dcp_world_size = 1
    impl.num_kv_heads = 2
    impl.alibi_slopes = None
    impl.sliding_window = (-1, -1)
    impl.scale = 128 ** -0.5
    impl.logits_soft_cap = 0.0
    impl.vllm_flash_attn_version = 2
    padded_output = torch.full((sum(q_lens) + 13, 16, 128), 17.0, device='cuda', dtype=q.dtype)
    scales = torch.ones(1, device='cuda')
    layer = SimpleNamespace(_q_scale=scales, _k_scale=scales, _v_scale=scales)
    returned = impl.forward(layer, q, None, None, torch.stack((k, v)), metadata, output=padded_output)
    assert returned.shape == padded_output.shape and returned.data_ptr() == padded_output.data_ptr()
    torch.testing.assert_close(padded_output[:sum(q_lens)], expected, atol=2e-2, rtol=1e-2)
    assert torch.all(padded_output[sum(q_lens):] == 17.0)
    for _ in range(3):
        old()
        unmodified_flash()
        new()
        split_mixed()
    torch.cuda.synchronize()
    samples = {'original': [], 'direct_flaggems': [], 'candidate': [], 'split_mixed': []}
    for repeat in range(9):
        order = [('original', old), ('direct_flaggems', unmodified_flash), ('candidate', new), ('split_mixed', split_mixed)]
        if repeat % 2:
            order.reverse()
        for label, fn in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples[label].append(start.elapsed_time(end))
    medians = {label: statistics.median(values) for label, values in samples.items()}
    result = dict(name=name, query_lengths=q_lens, kv_lengths=kv_lens, strided=strided,
                  q_stride=list(q.stride()), kv_stride=list(k.stride()), errors=errors,
                  milliseconds=samples, median_ms=medians,
                  speedup=medians['original'] / medians['candidate'],
                  compact_speedup_over_direct=medians['direct_flaggems'] / medians['candidate'],
                  split_speedup_over_original=medians['original'] / medians['split_mixed'],
                  split_active=can_split,
                  metadata_and_forward_adapter_passed=True,
                  compact_bitwise_equal_to_direct=True, passed=True)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.time()
    specs = [('prefill_2048', [2048], [2048], False),
             ('chunk_2048_prefix2048', [2048], [4096], True),
             ('decode_b1_4k', [1], [4096], False),
             ('decode_b64_4k', [1] * 64, [4096] * 64, False),
             ('mixed_ragged', [16, 1, 257], [1023, 4097, 513], True),
             ('mixed_4k_b64', [1985] + [1] * 63, [4033] + [4096] * 63, True),
             ('mixed_4k_b64_prefill_last', [1] * 63 + [1985], [4096] * 63 + [4033], True)]
    report = dict(purpose='Unscored operator correctness and paired timing',
                  tolerance={'atol': .02, 'rtol': .01, 'source': 'FlagGems v5.3.5 tests/test_flash_attn_varlen_func.py'},
                  cases=[], passed=False)
    try:
        for index, spec in enumerate(specs):
            report['cases'].append(case(*spec, seed=20261011 + index))
        report['passed'] = True
    finally:
        report['elapsed_seconds'] = time.time() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
