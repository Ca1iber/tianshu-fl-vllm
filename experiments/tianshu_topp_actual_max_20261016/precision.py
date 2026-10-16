# SPDX-License-Identifier: Apache-2.0
"""Actual global maximum normalization; retain ordinary FP32 tree sums."""
import hashlib
import importlib.util
from pathlib import Path
import sys

SOURCE_SHA='64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
PREVIOUS_PATH=Path(__file__).resolve().parent.parent/'tianshu_topp_sum_precision_20261016/precision.py'
spec=importlib.util.spec_from_file_location('_tianshu_sum_generator',PREVIOUS_PATH)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
topk_ast=previous.topk_ast


def render_source(raw, mode):
    assert mode in ('base','maxcomp','max')
    if mode!='max':return previous.render_source(raw,mode)
    text=previous.render_source(raw,'base')
    start=text.index('        if TOPP_ENABLED and final_pivot == -float("inf"):')
    end=text.index('        # Sixth pass:',start);part=text[start:end]
    anchor='                # If no finite logits exist (all -inf), clamp min to'
    assert part.count(anchor)==1
    part=part.replace(anchor,'''                max_sample = max_logit
                sum_exp_logits = 0.0
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))
                    probs_blk = tl.where(mask_n, tl.exp(logits_blk - max_sample), 0.0)
                    sum_exp_logits += tl.sum(probs_blk)

'''+anchor)
    text=text[:start]+part+text[end:]
    assert topk_ast(text)==topk_ast(raw.decode())
    return text


def load_kernel(native_source,directory):
    raw=Path(native_source).read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    kernels={};records={}
    for mode in ('base','maxcomp','max'):
        source=directory/f'generated_{mode}.py';source.write_text(render_source(raw,mode))
        name=f'_tianshu_topp_actual_{mode}'
        spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module);kernels[mode]=module._topk_topp_kernel
        records[mode]=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),native_TopK_branch_AST_unchanged=True)
    return kernels,dict(native_source_sha256=SOURCE_SHA,base_sha256=previous.BASE_SHA,modes=records,
        strict_CPU_tolerance_unchanged=1e-6,p_unchanged=True,RNG_unchanged=True,
        all_GPU_arithmetic_FP32=True,installed_source_modified=False)
