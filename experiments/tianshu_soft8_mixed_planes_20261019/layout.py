# SPDX-License-Identifier: Apache-2.0
"""Real page-padded raw allocation with nonoverlapping, independently typed planes."""
MODES = ('BF16', 'Soft8', 'Soft8OwnWriter', 'K16V8', 'K8V16')


def geometry(mode):
    assert mode in MODES
    key_bytes = 2 if mode in ('BF16', 'K16V8') else 1
    value_bytes = 2 if mode in ('BF16', 'K8V16') else 1
    return dict(key_bytes=key_bytes, value_bytes=value_bytes,
        page_bytes=4096 * (key_bytes + value_bytes), value_offset=4096 * key_bytes)


def allocate(torch, FullAttentionSpec, reshape, mode, physical_pages, strided):
    g = geometry(mode)
    dtype = torch.bfloat16 if mode == 'BF16' else torch.uint8
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=128, dtype=dtype,
        page_size_padded=g['page_bytes'] if mode in ('K16V8', 'K8V16') else None)
    assert spec.page_size_bytes == g['page_bytes']
    raw = torch.full((physical_pages * g['page_bytes'],), 17, device='cuda', dtype=torch.uint8)
    tensor = reshape(raw, spec, (physical_pages, 2, 16, 2, 128), tuple(range(5)), physical_pages, None)
    if strided:
        tensor = tensor[::2]
    step = tensor.stride(0) * tensor.element_size()
    count = tensor.shape[0]
    planes = []
    for size, offset in ((g['key_bytes'], 0), (g['value_bytes'], g['value_offset'])):
        plane_dtype = torch.bfloat16 if size == 2 else torch.uint8
        plane = torch.empty(0, device='cuda', dtype=plane_dtype).set_(
            raw.untyped_storage(), offset // size, (count, 16, 2, 128),
            (step // size, 256, 128, 1))
        assert plane.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
        assert plane.data_ptr() == raw.data_ptr() + offset
        planes.append(plane)
    k, v = planes
    proof = dict(mode=mode, **g, physical_pages=physical_pages, num_blocks=count,
        actual_spec_dtype=str(spec.dtype), actual_spec_page_bytes=spec.page_size_bytes,
        actual_raw_bytes=raw.numel(), actual_native_view_shape=list(tensor.shape),
        actual_native_view_stride_bytes=step, key_dtype=str(k.dtype), value_dtype=str(v.dtype),
        key_strides=list(k.stride()), value_strides=list(v.stride()),
        plane_aliases_actual_raw=True, no_hidden_BF16_cache=True,
        full_engine_allocator_not_installed=True)
    return raw, tensor, k, v, proof
