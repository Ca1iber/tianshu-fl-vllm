# SPDX-License-Identifier: Apache-2.0
"""Only offset arithmetic changes; both arms use the measured K8V16 policy."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parents[1]/'tianshu_k8v16_pipeline_20261019/dispatch_policy.py'
spec=importlib.util.spec_from_file_location('_k8v16_offset_original_policy',path)
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
PLANS={'current':None,'address32':None}
MODES=tuple(PLANS)
def configuration(name,batch,context):
    assert name in PLANS and 1<=batch<=64 and context>0
    return dict(policy.choose_config(batch,context))
