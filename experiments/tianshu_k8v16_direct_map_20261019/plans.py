# SPDX-License-Identifier: Apache-2.0
"""Same K8V16 launch policy; candidate changes only sequence mapping/grid."""
import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tianshu_k8v16_pipeline_20261019/dispatch_policy.py'
s=importlib.util.spec_from_file_location('_k8v16_direct_policy',p)
policy=importlib.util.module_from_spec(s);s.loader.exec_module(policy)
PLANS={'current':False,'direct':True}
MODES=tuple(PLANS)
def configuration(name,batch,context):
    assert name in PLANS and 1<=batch<=64 and context>0
    return dict(policy.choose_config(batch,context))
