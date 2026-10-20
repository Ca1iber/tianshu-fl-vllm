# SPDX-License-Identifier: Apache-2.0
"""Preserve real 8448-byte K8VI8 page layout for both integer PV tiles."""
import importlib.util
from pathlib import Path
PARENT=Path(__file__).resolve().parent.parent/'tianshu_k8vi8_heads_v2_20261021'
spec=importlib.util.spec_from_file_location('_PV_attention_native_layout',PARENT/'layout.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
MODES=('K8V16','K8VI8Heads','PV32')
def allocate(torch,spec,reshape,mode,pages,strided):
    assert mode in MODES
    underlying='K8VI8Heads' if mode.startswith('PV') else mode
    raw,native,k,v,ks,vs,proof=parent.allocate(torch,spec,reshape,underlying,pages,strided)
    return raw,native,k,v,ks,vs,dict(proof,mode=mode,underlying_layout=underlying)
