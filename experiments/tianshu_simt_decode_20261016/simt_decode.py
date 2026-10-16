# SPDX-License-Identifier: Apache-2.0
"""Explicit FP32 reduction of BF16 products for one-query GQA decode."""
import hashlib
import importlib.util
from pathlib import Path
import sys
CV_SHA='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest()==CV_SHA
    text=raw.decode();edits=[]
    def edit(a,b):
        nonlocal text
        assert text.count(a)==1,a
        text=text.replace(a,b);edits.append((a,b))
    signature='    KV_CACHE_MODIFIER: tl.constexpr = "",\n):'
    edit(signature,'    KV_CACHE_MODIFIER: tl.constexpr = "",\n    SIMT_DECODE: tl.constexpr = False,\n):')
    before='    USE_FP8_Q_DESCALE: tl.constexpr = KV_QUANT_MODE == 1 and Q_IS_FP8\n'
    edit(before,before+'''    USE_SIMT: tl.constexpr = (SIMT_DECODE and IS_3D and BLOCK_M == 8 and BLOCK_Q == 1
        and num_query_heads == 16 and num_queries_per_kv == 8 and HEAD_SIZE == 128
        and KV_QUANT_MODE == 0 and not USE_SINKS and not USE_FP8
        and not USE_TD and not USE_TD_QO)
''')
    resolve='''    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )'''
    direct='''    if USE_SIMT:
        seq_idx = q_block_global_idx
        q_block_local_idx = 0
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_query_len = tl.load(query_start_len_ptr + seq_idx + 1) - cur_batch_in_all_start_index
        seq_len = tl.load(seq_lens_ptr + seq_idx)
    else:
'''+ '\n'.join('    '+line for line in resolve.splitlines())
    edit(resolve,direct)
    edit('            S += score_scale * tl.dot(Q, K)', '''            if USE_SIMT:
                S += score_scale * tl.sum(Q.to(tl.float32)[:, :, None] * K.to(tl.float32)[None, :, :], axis=1)
            else:
                S += score_scale * tl.dot(Q, K)''')
    edit('            acc += tl.dot(P.to(V.dtype), V)', '''            if USE_SIMT:
                acc += tl.sum(P.to(V.dtype).to(tl.float32)[:, :, None] * V.to(tl.float32)[None, :, :], axis=1)
            else:
                acc += tl.dot(P.to(V.dtype), V)''')
    restored=text
    for a,b in reversed(edits):
        assert restored.count(b)==1
        restored=restored.replace(b,a)
    assert restored.encode()==raw
    assert text.count('tl.int64')==raw.decode().count('tl.int64')
    return text


def load_kernel(cv_source,directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    source=root/'generated_simt_decode.py';source.write_text(render_source(Path(cv_source).read_bytes()))
    name='_tianshu_simt_decode';spec=importlib.util.spec_from_file_location(name,source)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module.kernel_unified_attention,dict(base_cv_sha256=CV_SHA,generated_source=str(source),generated_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        explicit_FP32_sums_of_BF16_products=True,probabilities_cast_to_original_BF16_before_PV=True,
        query_rows=8,direct_single_query_grid=True,integer_widths_unchanged=True,
        reducer_source_unchanged=True,floating_reduction_order_changed=True,installed_sources_modified=False)
