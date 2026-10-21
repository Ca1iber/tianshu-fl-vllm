# SPDX-License-Identifier: Apache-2.0
"""Fuse the actual qualified TopP maximum scan and stable exponential sum."""
import hashlib, importlib.util, sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_topp_geometric_20261016/precision.py'
spec=importlib.util.spec_from_file_location('_topp_online_actual_geo_parent',PARENT)
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
NATIVE_SHA='64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
QUALIFIED_SHA='09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
MODES=('geo','online')
MAX_BEFORE='                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))\n'
MAX_AFTER='''                    previous_max_logit = max_logit
                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
'''
SUM_BEFORE='''                    probs_blk = tl.exp(logits_blk - max_sample)
                    probs_blk = tl.where(mask_n, probs_blk, 0.0)
                    sum_exp_logits += tl.sum(probs_blk)
'''
SUM_AFTER='''                    rescale = tl.where(previous_max_logit > -float("inf"),
                                       tl.exp(previous_max_logit - max_logit), 0.0)
                    finite_probs = mask_n & (logits_blk > -float("inf"))
                    probs_blk = tl.where(finite_probs, tl.exp(logits_blk - max_logit), 0.0)
                    sum_exp_logits = sum_exp_logits * rescale + tl.sum(probs_blk)
'''
RECOMPUTE='''                max_sample = max_logit
                sum_exp_logits = 0.0
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))
                    probs_blk = tl.where(mask_n, tl.exp(logits_blk - max_sample), 0.0)
                    sum_exp_logits += tl.sum(probs_blk)

'''

def generate(raw,mode):
    assert hashlib.sha256(raw).hexdigest()==NATIVE_SHA and mode in MODES
    baseline=parent.render_source(raw,'geo')
    assert hashlib.sha256(baseline.encode()).hexdigest()==QUALIFIED_SHA
    start=baseline.index('        if TOPP_ENABLED and final_pivot == -float("inf"):')
    end=baseline.index('        # Sixth pass:',start)
    original=baseline[start:end];part=original
    if mode=='online':
        for old in (MAX_BEFORE,SUM_BEFORE,RECOMPUTE):assert part.count(old)==1
        part=part.replace(MAX_BEFORE,MAX_AFTER).replace(SUM_BEFORE,SUM_AFTER).replace(RECOMPUTE,'                max_sample = max_logit\n\n')
        restored=part.replace(MAX_AFTER,MAX_BEFORE).replace(SUM_AFTER,SUM_BEFORE).replace('                max_sample = max_logit\n\n',RECOMPUTE)
        assert restored==original
    text=baseline[:start]+part+baseline[end:]
    assert parent.topk_ast(text)==parent.topk_ast(baseline)
    return text,dict(mode=mode,native_sha256=NATIVE_SHA,qualified_parent_sha256=QUALIFIED_SHA,
        generated_sha256=hashlib.sha256(text.encode()).hexdigest(),TopK_AST_unchanged=True,
        only_standalone_max_and_normalization_scan_changed=True,
        actual_global_max_normalization_retained=True,geometric_search_and_final_mask_unchanged=True,
        outlier_threshold_and_mass_fallback_unchanged=True,search_limit_unchanged=32,
        interval_tolerance_unchanged=1e-12,CPU_semantic_tolerance_unchanged=1e-6,
        p_and_RNG_unchanged=True,FP32_reduction_order_may_differ=mode=='online')

def load_kernels(native,directory):
    raw=Path(native).read_bytes();directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    kernels={};proofs={}
    for mode in MODES:
        text,proof=generate(raw,mode);path=directory/f'generated_{mode}.py';path.write_text(text)
        name='_topp_online_'+mode+'_'+proof['generated_sha256'][:12]
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
        kernels[mode]=module._topk_topp_kernel;proofs[mode]=dict(proof,generated_source=str(path))
    return kernels,proofs

def dependency_sources():
    return [Path(__file__),PARENT,parent.PARENT,parent.previous.PREVIOUS_PATH,parent.previous.previous.BASE_PATH,
            Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_triton.py')]
