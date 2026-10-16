# SPDX-License-Identifier: Apache-2.0
"""Keep decode online softmax in log2 units; restore natural-log segment maxima."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

CV_SHA='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'
HELPER_SHA='59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1'
HELPERS=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_attention_helpers.py')


def render_source(raw,helper_raw):
    assert hashlib.sha256(raw).hexdigest()==CV_SHA
    assert hashlib.sha256(helper_raw).hexdigest()==HELPER_SHA
    helper_text=helper_raw.decode();tree=ast.parse(helper_text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='softmax_step')
    function='\n'.join(helper_text.splitlines()[node.lineno-2:node.end_lineno])+'\n\n'
    assert function.count('tl.exp(')==2
    helper=function.replace('def softmax_step(', 'def _tianshu_softmax_step_log2(').replace('tl.exp(', 'tl.exp2(')
    text=raw.decode();edits=[]
    def edit(before,after):
        nonlocal text
        assert text.count(before)==1,before
        text=text.replace(before,after);edits.append((before,after))
    marker='@triton.jit\ndef kernel_unified_attention('
    edit(marker,helper+marker)
    signature='    KV_CACHE_MODIFIER: tl.constexpr = "",\n):'
    edit(signature,'    KV_CACHE_MODIFIER: tl.constexpr = "",\n    LOG2_DECODE: tl.constexpr = False,\n):')
    before='    USE_FP8_Q_DESCALE: tl.constexpr = KV_QUANT_MODE == 1 and Q_IS_FP8\n'
    edit(before,before+'''    USE_LOG2_DECODE: tl.constexpr = (LOG2_DECODE and IS_3D and not USE_SINKS
        and not USE_SOFTCAP and not USE_ALIBI_SLOPES and not USE_QQ_BIAS
        and not USE_MM_PREFIX and not SLIDING_WINDOW and not USE_FP8
        and KV_QUANT_MODE == 0 and not USE_TD and not USE_TD_QO)
''')
    edit('    score_scale = scale\n','    score_scale = scale * 1.4426950408889634 if USE_LOG2_DECODE else scale\n')
    edit('        M, L, P, alpha = softmax_step(S, M, L)', '''        if USE_LOG2_DECODE:
            M, L, P, alpha = _tianshu_softmax_step_log2(S, M, L)
        else:
            M, L, P, alpha = softmax_step(S, M, L)''')
    edit('        store_segm_reduce_scalars(\n','        stored_M = M * 0.6931471805599453 if USE_LOG2_DECODE else M\n        store_segm_reduce_scalars(\n')
    edit('            M,\n            L,\n            query_mask_0,','            stored_M,\n            L,\n            query_mask_0,')
    restored=text
    for before,after in reversed(edits):
        assert restored.count(after)==1
        restored=restored.replace(after,before)
    assert restored.encode()==raw
    assert text.count('tl.int64')==raw.decode().count('tl.int64')
    return text


def load_kernel(cv_source,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    source=directory/'generated_log2_decode.py';source.write_text(render_source(Path(cv_source).read_bytes(),HELPERS.read_bytes()))
    module_name='_tianshu_log2_decode'
    spec=importlib.util.spec_from_file_location(module_name,source)
    module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module;spec.loader.exec_module(module)
    return module.kernel_unified_attention,dict(base_cv_sha256=CV_SHA,helpers_source=str(HELPERS),helpers_sha256=HELPER_SHA,
        generated_source=str(source),generated_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        score_scale_log2_folded=True,softmax_exp2_domain=True,segment_maximum_restored_to_natural_log=True,
        reducer_source_unchanged=True,integer_widths_and_offsets_unchanged=True,
        floating_rounding_may_change=True,installed_sources_modified=False)
