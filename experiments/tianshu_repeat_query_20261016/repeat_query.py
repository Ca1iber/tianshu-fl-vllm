# SPDX-License-Identifier: Apache-2.0
"""Repeat the one decode query into unused matrix rows; keep all store masks."""
import hashlib
import importlib.util
from pathlib import Path
import sys

CV_SHA='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'
SIGNATURE='    KV_CACHE_MODIFIER: tl.constexpr = "",\n):'
NEW_SIGNATURE='    KV_CACHE_MODIFIER: tl.constexpr = "",\n    REPEAT_DECODE_QUERY: tl.constexpr = False,\n):'
ANCHOR='    # Q : (BLOCK_M, HEAD_SIZE_PADDED)'
PREPARE='''    # IS_3D is selected by the native wrapper only for max_seqlen_q <= 1.
    # Empty queries and extra query blocks returned above. Keep original
    # query_offset_0/query_mask_0 for the epilogue: only eight rows may store.
    if (REPEAT_DECODE_QUERY and IS_3D and BLOCK_M == 16 and BLOCK_Q == 2
        and num_query_heads == 16 and num_queries_per_kv == 8
        and not USE_TD_QO and not USE_SINKS and not USE_QQ_BIAS
        and not USE_ALIBI_SLOPES and KV_QUANT_MODE == 0):
        compute_query_pos = q_block_local_idx * BLOCK_Q + tl.zeros((BLOCK_M,), tl.int32)
        compute_query_mask_0 = tl.full((BLOCK_M,), True, tl.int1)
        query_offset = (
            (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * query_stride_0
            + query_offset_1[:, None] * query_stride_1 + offs_d[None, :]
        )
    else:
        compute_query_pos = query_pos
        compute_query_mask_0 = query_mask_0

''' + ANCHOR
LOAD='mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],\n            other=0.0,'
NEW_LOAD='mask=dim_mask[None, :] & compute_query_mask_0[:, None] & query_mask_1[:, None],\n            other=0.0,'
POSITION='        query_abs_pos = context_len + query_pos[:, None]'
NEW_POSITION='        query_abs_pos = context_len + compute_query_pos[:, None]'
MASK='            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")'
NEW_MASK='            query_mask_1[:, None] & compute_query_mask_0[:, None] & seq_mask, S, float("-inf")'
REPLACEMENTS=((SIGNATURE,NEW_SIGNATURE),(ANCHOR,PREPARE),(LOAD,NEW_LOAD),(POSITION,NEW_POSITION),(MASK,NEW_MASK))


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest()==CV_SHA
    result=raw.decode()
    for before,after in REPLACEMENTS:
        assert result.count(before)==1,before
        result=result.replace(before,after)
    original=result
    for before,after in reversed(REPLACEMENTS):original=original.replace(after,before)
    assert original.encode()==raw
    # No epilogue code, segment contract or reducer code may change.
    marker='    # ---- Epilogue '
    assert result[result.index(marker):]==raw.decode()[raw.decode().index(marker):]
    return result


def load_kernel(cv_source,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    source=directory/'generated_repeat_query.py';source.write_text(render_source(Path(cv_source).read_bytes()))
    name='_tianshu_repeat_single_decode_query'
    spec=importlib.util.spec_from_file_location(name,source)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module.kernel_unified_attention,dict(base_cv_sha256=CV_SHA,generated_source=str(source),
        generated_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        valid_rows_query_and_attention_arithmetic_preserved=True,inactive_compute_rows_repeated=True,
        integer_widths_unchanged=True,epilogue_masks_offsets_and_reducer_source_unchanged=True,
        requires_native_wrapper_3D_single_query_contract=True,installed_sources_modified=False)
