# SPDX-License-Identifier: Apache-2.0
"""Adapt the native raw V argument to its physical BF16 view, then write both planes."""
import importlib.util
from pathlib import Path
import torch
from value_planes import value_plane

WRITER = Path(__file__).resolve().parents[1] / 'tianshu_soft8_mixed_planes_20261019/writer.py'
spec = importlib.util.spec_from_file_location('_qualified_mixed_KV_writer', WRITER)
owned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owned)
_pipeline_encode_all = owned.encode_all


def triton_reshape_and_cache_flash(key, value, k, v, slots, dtype, k_scale, v_scale):
    assert dtype == 'auto' and k.dtype == v.dtype == torch.uint8
    return owned.write(key, value, k, value_plane(v), slots)
