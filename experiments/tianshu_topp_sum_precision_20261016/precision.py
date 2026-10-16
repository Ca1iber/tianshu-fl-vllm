# SPDX-License-Identifier: Apache-2.0
"""FP32 compensated sums for standalone TopP; native installed files untouched."""
import hashlib
import importlib.util
from pathlib import Path
import sys

SOURCE_SHA = '64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
BASE_SHA = '029d805b1a19be91f74afa03afb25c834409c8f08b3d30f7eabd1771c4aefedf'
BASE_PATH = Path(__file__).resolve().parent.parent/'tianshu_topp_precision_20261014/precision.py'
spec = importlib.util.spec_from_file_location('_tianshu_base_precision_generator', BASE_PATH)
base = importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
topk_ast = base.topk_ast

HELPERS = '''@triton.jit
def _sum_pair(ha, la, hb, lb):
    value = ha + hb
    error = tl.where(tl.abs(ha) >= tl.abs(hb), (ha - value) + hb, (hb - value) + ha)
    return value, la + lb + error


@triton.jit
def _sum_compensated(values):
    high, low = tl.reduce((values, tl.full(values.shape, 0.0, tl.float32)), 0, _sum_pair)
    return high + low


'''


def render_source(raw, mode):
    assert mode in ('base','comp','maxcomp')
    text = base.render_source(raw)
    assert hashlib.sha256(text.encode()).hexdigest() == BASE_SHA
    if mode == 'base': return text
    start = text.index('        if TOPP_ENABLED and final_pivot == -float("inf"):')
    end = text.index('        # Sixth pass:', start)
    original = text[start:end]; part = original
    if mode == 'maxcomp':
        # Obtain the actual maximum, then evaluate exp relative to that maximum.
        # Keeps p and the independent strict CPU tolerance unchanged.
        part = part.replace('                # If no finite logits exist (all -inf), clamp min to', '''                max_sample = max_logit
                sum_exp_logits = 0.0
                sum_exp_correction = 0.0
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))
                    probs_blk = tl.where(mask_n, tl.exp(logits_blk - max_sample), 0.0)
                    partial = _sum_compensated(probs_blk)
                    adjusted = partial - sum_exp_correction
                    updated = sum_exp_logits + adjusted
                    sum_exp_correction = (updated - sum_exp_logits) - adjusted
                    sum_exp_logits = updated

                # If no finite logits exist (all -inf), clamp min to''')
        assert part != original
    else:
        part = part.replace('                sum_exp_logits = 0.0\n', '                sum_exp_logits = 0.0\n                sum_exp_correction = 0.0\n',1)
        before='                    sum_exp_logits += tl.sum(probs_blk)\n'
        assert part.count(before)==1
        part=part.replace(before,'''                    partial = _sum_compensated(probs_blk)
                    adjusted = partial - sum_exp_correction
                    updated = sum_exp_logits + adjusted
                    sum_exp_correction = (updated - sum_exp_logits) - adjusted
                    sum_exp_logits = updated
''')
    before='                    sum_outlier_probs += tl.sum(outlier_mask * probs_blk)'
    assert part.count(before)==1
    part=part.replace('                sum_outlier_probs = 0.0\n','                sum_outlier_probs = 0.0\n                outlier_correction = 0.0\n')
    part=part.replace(before,'''                    partial = _sum_compensated(outlier_mask * probs_blk)
                    adjusted = partial - outlier_correction
                    updated = sum_outlier_probs + adjusted
                    outlier_correction = (updated - sum_outlier_probs) - adjusted
                    sum_outlier_probs = updated''')
    init='                        p_pivots_sum_0 = 0.0\n'
    assert part.count(init)==2
    part=part.replace(init,init+'                        pivot_correction = 0.0\n')
    before='                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)'
    assert part.count(before)==2
    part=part.replace(before,'''                            partial = _sum_compensated(probs_blk * above_0)
                            adjusted = partial - pivot_correction
                            updated = p_pivots_sum_0 + adjusted
                            pivot_correction = (updated - p_pivots_sum_0) - adjusted
                            p_pivots_sum_0 = updated''')
    text=text[:start]+part+text[end:]
    anchor='@triton.jit\ndef _topk_topp_kernel('
    assert text.count(anchor)==1
    text=text.replace(anchor, HELPERS+anchor)
    assert topk_ast(text)==topk_ast(raw.decode())
    assert '1e-6' not in part and 'tl.load(P + row_id)' in part
    return text


def load_kernel(native_source, directory):
    raw=Path(native_source).read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    kernels={};records={}
    for mode in ('base','comp','maxcomp'):
        source=directory/f'generated_{mode}.py';source.write_text(render_source(raw,mode))
        name=f'_tianshu_topp_sum_{mode}'
        spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module);kernels[mode]=module._topk_topp_kernel
        records[mode]=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),native_TopK_branch_AST_unchanged=True)
    return kernels,dict(native_source_sha256=SOURCE_SHA,base_sha256=BASE_SHA,modes=records,
        strict_CPU_tolerance_unchanged=1e-6,p_unchanged=True,RNG_unchanged=True,
        all_GPU_arithmetic_FP32=True,installed_source_modified=False)
