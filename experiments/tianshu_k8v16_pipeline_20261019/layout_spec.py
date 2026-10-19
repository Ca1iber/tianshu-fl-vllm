# SPDX-License-Identifier: Apache-2.0
"""A guarded immutable dtype transition for real FullAttentionSpec mappings."""
from dataclasses import replace

def convert_specs(specs,full_attention_type,bf16_dtype,uint8_dtype):
    assert len(specs)==42
    for name,s in specs.items():
        assert isinstance(s,full_attention_type)
        assert s.dtype==bf16_dtype and s.block_size==16 and s.num_kv_heads==2
        assert s.head_size==s.head_size_v==128 and s.sliding_window is None
        assert not s.non_causal and s.attention_chunk_size is None and s.page_size_padded is None
    result={name:replace(s,dtype=uint8_dtype,page_size_padded=12288) for name,s in specs.items()}
    for name,s in result.items():
        assert s.page_size_bytes==12288 and specs[name].page_size_bytes==16384
        assert specs[name].dtype==bf16_dtype
    return result
