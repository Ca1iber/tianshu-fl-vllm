# SPDX-License-Identifier: Apache-2.0
"""Combine two GQA KV heads in one BF16 matrix tile, masking cross-head scores."""
import hashlib
import importlib.util
from pathlib import Path
import sys

CV_SHA='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'
SIGNATURE='    KV_CACHE_MODIFIER: tl.constexpr = "",\n):'
NEW_SIGNATURE='    KV_CACHE_MODIFIER: tl.constexpr = "",\n    PACK_KV_HEADS: tl.constexpr = False,\n):'
ACTIVE='''    PACK_ACTIVE: tl.constexpr = (
        PACK_KV_HEADS and IS_3D and BLOCK_M == 16 and BLOCK_Q == 2
        and num_query_heads == 16 and num_queries_per_kv == 8
        and HEAD_SIZE == 128 and TILE_SIZE == 16 and KV_QUANT_MODE == 0
        and not USE_TD and not USE_TD_QO and not USE_SINKS and not Q_IS_FP8
        and not USE_FP8 and not USE_QQ_BIAS and not USE_ALIBI_SLOPES
        and not USE_SOFTCAP and not SLIDING_WINDOW and not USE_MM_PREFIX
        and not USE_PER_SEQ_CAUSAL and CHUNK_LOOKBACK < 0
    )
    PACKED_TILE_SIZE: tl.constexpr = 2 * TILE_SIZE if PACK_ACTIVE else TILE_SIZE
'''


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest()==CV_SHA
    text=raw.decode();edits=[]
    def edit(before,after):
        nonlocal text
        assert text.count(before)==1,before
        text=text.replace(before,after);edits.append((before,after))
    edit(SIGNATURE,NEW_SIGNATURE)
    anchor='    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2\n'
    edit(anchor,ACTIVE+anchor)
    edit('    offs_t = tl.arange(0, TILE_SIZE)\n','    offs_t = tl.arange(0, PACKED_TILE_SIZE)\n')
    edit('    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv\n', '''    if PACK_ACTIVE:
        query_pos = q_block_local_idx * BLOCK_Q + tl.zeros((BLOCK_M,), tl.int32)
    else:
        query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
''')
    edit('    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv\n', '''    if PACK_ACTIVE:
        query_offset_1 = offs_m
    else:
        query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
''')
    edit('        seq_offset = j * TILE_SIZE + offs_t\n', '''        if PACK_ACTIVE:
            seq_offset = j * TILE_SIZE + offs_t % TILE_SIZE
            column_kv_head = offs_t // TILE_SIZE
        else:
            seq_offset = j * TILE_SIZE + offs_t
            column_kv_head = kv_head_idx + tl.zeros((TILE_SIZE,), tl.int32)
''')
    edit('                + kv_head_idx * stride_v_cache_2\n','                + column_kv_head[:, None] * stride_v_cache_2\n')
    edit('                + kv_head_idx * stride_k_cache_2\n','                + column_kv_head[None, :] * stride_k_cache_2\n')
    edit('        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)\n','        S = tl.zeros(shape=(BLOCK_M, PACKED_TILE_SIZE), dtype=tl.float32)\n')
    anchor='        M, L, P, alpha = softmax_step(S, M, L)\n'
    edit(anchor,'''        if PACK_ACTIVE:
            same_kv_head = (offs_m[:, None] // num_queries_per_kv) == column_kv_head[None, :]
            S = tl.where(same_kv_head, S, -float("inf"))

'''+anchor)
    assert text[text.index('    # ---- Epilogue '):]==raw.decode()[raw.decode().index('    # ---- Epilogue '):]
    original=text
    for before,after in reversed(edits):
        assert original.count(after)==1
        original=original.replace(after,before)
    assert original.encode()==raw
    return text


def load_kernel(cv_source,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    source=directory/'generated_pack_kv_heads.py';source.write_text(render_source(Path(cv_source).read_bytes()))
    name='_tianshu_pack_two_kv_heads'
    spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module.kernel_unified_attention,dict(base_cv_sha256=CV_SHA,generated_source=str(source),generated_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        two_KV_heads_in_one_matrix_tile=True,cross_head_scores_masked_before_softmax=True,
        same_logical_tokens_per_iteration=True,BF16_dot_and_FP32_accumulators_preserved=True,
        integer_widths_unchanged=True,epilogue_and_reducer_source_unchanged=True,
        native_sequence_grid_resolution_preserved=True,installed_sources_modified=False)
