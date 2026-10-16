# SPDX-License-Identifier: Apache-2.0
"""Gather request positions without shape-specific generic index autotuning."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['n'])
def _positions(computed, requests, query, output, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    req = tl.load(requests + i, valid, other=0)
    base = tl.load(computed + req, valid, other=0).to(tl.int64)
    offset = tl.load(query + i, valid, other=0)
    tl.store(output + i, base + offset, valid)


def launch(owner, n):
    _positions[(triton.cdiv(n, 256),)](
        owner.num_computed_tokens, owner.req_indices.gpu, owner.query_pos.gpu,
        owner.positions, n, BLOCK=256, num_warps=4, num_stages=1)
