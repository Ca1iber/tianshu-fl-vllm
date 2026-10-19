# SPDX-License-Identifier: Apache-2.0
"""Owned K-only physical layout sources; V and attention arithmetic stay unchanged."""
import ast
import hashlib

ATTENTION_SHA='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'
WRITER_SHA='f18224e51162e37c14a2e2b3045bca55baf0b207b1d203ad6ee7d8ed417139eb'
CHUNKS=(2,4,8,16)
K_ADDRESS='''            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )'''
OLD_STORE='''        tgt_idx_k = (
            block_idx * block_stride
            + block_offset * page_stride
            + cur_head * head_stride
            + cur_dim
        )
        tgt_idx_v = tgt_idx_k'''


def k_local(token,head,dim,x):
    assert x in CHUNKS and 0<=token<16 and 0<=head<2 and 0<=dim<128
    return head*16*128+(dim//x)*16*x+token*x+dim%x


def render_attention(raw,x):
    assert hashlib.sha256(raw).hexdigest()==ATTENTION_SHA and x in CHUNKS
    text=raw.decode();assert text.count(K_ADDRESS)==1
    replacement=f'''            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * BLOCK_SIZE * HEAD_SIZE
                + (offs_d[:, None] // {x}) * BLOCK_SIZE * {x}
                + (seq_offset % BLOCK_SIZE)[None, :] * {x}
                + offs_d[:, None] % {x}
            )'''
    changed=text.replace(K_ADDRESS,replacement)
    assert changed.replace(replacement,K_ADDRESS)==text
    ast.parse(changed)
    return changed


def render_writer(raw,x):
    assert hashlib.sha256(raw).hexdigest()==WRITER_SHA and x in CHUNKS
    text=raw.decode();assert text.count(OLD_STORE)==1
    replacement=f'''        tgt_idx_v = (
            block_idx * block_stride
            + block_offset * page_stride
            + cur_head * head_stride
            + cur_dim
        )
        tgt_idx_k = (
            block_idx * block_stride
            + cur_head * block_size * head_size
            + (cur_dim // {x}) * block_size * {x}
            + block_offset * {x}
            + cur_dim % {x}
        )'''
    changed=text.replace(OLD_STORE,replacement)
    assert changed.replace(replacement,OLD_STORE)==text
    ast.parse(changed)
    return changed
