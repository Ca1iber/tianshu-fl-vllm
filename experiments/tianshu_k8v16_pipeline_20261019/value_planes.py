# SPDX-License-Identifier: Apache-2.0
"""Interpret the V region of a real 12KiB page; retain metadata views, never copy KV."""
import torch

_views = {}
records = {}


def value_plane(raw_value):
    if raw_value.dtype == torch.bfloat16:
        return raw_value
    assert raw_value.dtype == torch.uint8 and raw_value.ndim == 4
    assert tuple(raw_value.shape[1:]) == (16, 2, 128)
    assert tuple(raw_value.stride()[1:]) == (256, 128, 1)
    assert raw_value.stride(0) >= 12288 and raw_value.stride(0) % 2 == 0
    assert raw_value.storage_offset() % 2 == 0
    key = (raw_value.data_ptr(), tuple(raw_value.shape), tuple(raw_value.stride()))
    if key in _views:
        return _views[key]
    if raw_value.device.type == 'cuda':
        assert not torch.cuda.is_current_stream_capturing(), 'prepare typed V views before Graph capture'
    storage = raw_value.untyped_storage()
    step = raw_value.stride(0)
    offset = raw_value.storage_offset()
    end = offset + (raw_value.shape[0] - 1) * step + 8192
    assert end <= storage.nbytes(), (end, storage.nbytes())
    view = torch.empty(0, device=raw_value.device, dtype=torch.bfloat16).set_(
        storage, offset // 2, tuple(raw_value.shape), (step // 2, 256, 128, 1))
    assert view.data_ptr() == raw_value.data_ptr()
    assert view.untyped_storage().data_ptr() == storage.data_ptr()
    _views[key] = view
    records[str(key)] = dict(shape=list(view.shape), raw_dtype='torch.uint8',
        actual_dtype=str(view.dtype), raw_storage_offset=offset,
        actual_stride=list(view.stride()), page_stride_bytes=step,
        actual_storage_end_bytes=end, storage_bytes=storage.nbytes(),
        same_data_pointer=True, same_untyped_storage=True, no_data_copy=True)
    return view


def clear_cache():
    """Release per-fixture views; the real engine retains one view per layer."""
    _views.clear()
