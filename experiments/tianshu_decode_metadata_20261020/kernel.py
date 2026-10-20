# SPDX-License-Identifier: Apache-2.0
"""Fuse only integer metadata production for one scheduled token per request."""
import triton
import triton.language as tl


@triton.jit
def prepare_decode_metadata(COMPUTED, REQ_INDEX, QUERY_POS, NUM_SCHEDULED,
                            POSITION, SEQ_LEN, N: tl.constexpr,
                            CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < N
    computed = tl.load(COMPUTED + index, active, 0)
    tl.store(REQ_INDEX + index, index.to(tl.int64), active)
    tl.store(QUERY_POS + index, 0, active)
    tl.store(NUM_SCHEDULED + index, 1, active)
    tl.store(POSITION + index, computed.to(tl.int64), active)
    tl.store(SEQ_LEN + index, tl.where(active, computed + 1, 0), index < CAPACITY)


def launch(owner, n):
    assert 1 <= n <= 64
    capacity = owner.seq_lens.numel()
    return prepare_decode_metadata[(triton.cdiv(capacity, 256),)](
        owner.num_computed_tokens, owner.req_indices.gpu, owner.query_pos.gpu,
        owner.num_scheduled_tokens.gpu, owner.positions, owner.seq_lens,
        N=n, CAPACITY=capacity, BLOCK=256, num_warps=4, num_stages=1)
