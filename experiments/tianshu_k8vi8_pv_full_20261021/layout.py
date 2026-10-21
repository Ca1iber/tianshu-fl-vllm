# SPDX-License-Identifier: Apache-2.0
"""Label-to-format adapter; all real native page/scale accounting is unchanged."""
import importlib.util
from pathlib import Path
from plans import MODES,PLANS
PARENT=Path(__file__).resolve().parent.parent/'tianshu_k8vi8_heads_v2_20261021/layout.py'
spec=importlib.util.spec_from_file_location('_full_PV_original_native_layout',PARENT)
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def allocate(torch,FullAttentionSpec,reshape,mode,physical_pages,strided,device='cuda'):
    storage_mode=PLANS[mode]['storage']
    result=parent.allocate(torch,FullAttentionSpec,reshape,storage_mode,physical_pages,strided,device)
    result[-1].update(mode=mode,storage_mode=storage_mode)
    return result
