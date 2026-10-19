# SPDX-License-Identifier: Apache-2.0
"""Every sweep arm uses the same measured physical K8V16 allocation."""
import importlib.util
from pathlib import Path
from plans import MODES

PARENT = Path(__file__).resolve().parents[1] / 'tianshu_soft8_mixed_planes_20261019/layout.py'
spec = importlib.util.spec_from_file_location('_k8v16_sweep_real_layout', PARENT)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)


def geometry(mode):
    assert mode in MODES
    return parent.geometry('K8V16')


def allocate(torch, FullAttentionSpec, reshape, mode, physical_pages, strided):
    assert mode in MODES
    raw, view, k, v, proof = parent.allocate(torch, FullAttentionSpec, reshape,
        'K8V16', physical_pages, strided)
    return raw, view, k, v, dict(proof, mode=mode, storage_mode='K8V16')
