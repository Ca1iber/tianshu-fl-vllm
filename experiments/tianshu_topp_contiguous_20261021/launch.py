# SPDX-License-Identifier: Apache-2.0
"""Original top-p JIT, fixed original tiles, only explicit warp-count variants."""
LABELS = ('native', 'warp2', 'warp4', 'warp8', 'warp16')
WARPS = dict(warp2=2, warp4=4, warp8=8, warp16=16)
CASES = [(b, kind) for b in (8,64) for kind in ('normal1','normal4','peaked','uniform','masked','nonfinite')]

def direct(module, logits, p, buffer, tables, programs, warps):
    normal, percentile = tables
    return module._topk_topp_kernel[(programs,)](
        logits, logits.stride(0), buffer, percentile, normal, logits, p,
        BATCH_SIZE=logits.shape[0], VOCAB_SIZE=logits.shape[1],
        MASK_VALUE=-float('inf'), BLOCK_SIZE=8192, BLOCK_SIZE_TRUNC=4096,
        TOPK_ENABLED=False, TOPP_ENABLED=True, num_warps=warps)
