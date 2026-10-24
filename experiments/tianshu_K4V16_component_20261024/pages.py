# SPDX-License-Identifier: Apache-2.0
"""Typed views of one actual 10496-byte raw page; no detached scale allocation."""
import torch
PAGE_BYTES=10496
SCALE_OFFSET=2048
V_OFFSET=2304

def views(raw,physical_pages,strided=False):
    assert raw.dtype==torch.uint8 and raw.is_contiguous() and raw.numel()==physical_pages*PAGE_BYTES
    assert raw.storage_offset()==0 and not torch.cuda.is_current_stream_capturing()
    k=torch.as_strided(raw,(physical_pages,16,2,64),(PAGE_BYTES,128,64,1))
    scales=torch.empty(0,device=raw.device,dtype=torch.float32).set_(raw.untyped_storage(),SCALE_OFFSET//4,
        (physical_pages,16,2),(PAGE_BYTES//4,2,1))
    v=torch.empty(0,device=raw.device,dtype=torch.bfloat16).set_(raw.untyped_storage(),V_OFFSET//2,
        (physical_pages,16,2,128),(PAGE_BYTES//2,256,128,1))
    assert k.data_ptr()==raw.data_ptr() and scales.data_ptr()-raw.data_ptr()==SCALE_OFFSET
    assert v.data_ptr()-raw.data_ptr()==V_OFFSET
    assert all(t.untyped_storage().data_ptr()==raw.untyped_storage().data_ptr() for t in (k,v,scales))
    return (k[::2],v[::2],scales[::2]) if strided else (k,v,scales)

def native_spec():
    from vllm.v1.kv_cache_interface import FullAttentionSpec,KVQuantMode
    spec=FullAttentionSpec(block_size=16,num_kv_heads=2,head_size=128,dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NONE,page_size_padded=PAGE_BYTES,indexes_kv_by_block_stride=True)
    assert spec.real_page_size_bytes==8192 and spec.page_size_bytes==PAGE_BYTES
    return dict(actual_FullAttentionSpec_type=type(spec).__name__,real_page_size_bytes=spec.real_page_size_bytes,
        page_size_bytes=spec.page_size_bytes,indexes_kv_by_block_stride=spec.indexes_kv_by_block_stride,
        raw_KV_mode='NONE: custom K4 reader, not native INT8 cache format',
        actual_reader_mode='INT8_PER_TOKEN_HEAD ABI reused only in standalone custom reader',
        engine_allocator_connected=False)
