# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Split a reordered mixed batch to preserve the FA2 single-query GQA path."""
from flag_gems import flash_attn_varlen_func


def flash_attn_split_mixed(
    *, q, k, v, out, cu_seqlens_q, prefill_cu_seqlens_q, seqused_k,
    block_table, num_decodes: int, max_seqlen_q: int, max_seqlen_k: int,
    softmax_scale: float, causal: bool = True, window_size=(-1, -1),
    softcap: float = 0.0, alibi_slopes=None,
):
    """Decode requests occupy the first num_decodes query tokens/sequences.

    The metadata builder validates this CPU-side and rebases prefill starts
    once per engine step. Both calls read the same paged KV cache and write
    disjoint slices of the caller's output. No GPU-to-CPU metadata copy.
    """
    common = dict(k=k, v=v, max_seqlen_k=max_seqlen_k,
                  softmax_scale=softmax_scale, causal=causal,
                  window_size=window_size, softcap=softcap,
                  alibi_slopes=alibi_slopes, num_splits=0, fa_version=2)
    flash_attn_varlen_func(
        **common, q=q[:num_decodes], out=out[:num_decodes],
        cu_seqlens_q=cu_seqlens_q[:num_decodes + 1], max_seqlen_q=1,
        seqused_k=seqused_k[:num_decodes], block_table=block_table[:num_decodes],
    )
    flash_attn_varlen_func(
        **common, q=q[num_decodes:], out=out[num_decodes:],
        cu_seqlens_q=prefill_cu_seqlens_q, max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k[num_decodes:], block_table=block_table[num_decodes:],
    )
    return out
