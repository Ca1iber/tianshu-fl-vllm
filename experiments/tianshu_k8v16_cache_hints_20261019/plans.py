# SPDX-License-Identifier: Apache-2.0
"""Same qualified launch policy in all arms; only native K/V load hints vary."""
import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tianshu_k8v16_pipeline_20261019/dispatch_policy.py'
s=importlib.util.spec_from_file_location('_k8v16_cache_hint_policy',p)
policy=importlib.util.module_from_spec(s);s.loader.exec_module(policy)
PLANS={'current':'.cv','cg':'.cg','ca':'.ca','default':''}
MODES=tuple(PLANS)
def configuration(name,batch,context):
    assert name in PLANS and 1<=batch<=64 and context>0
    return dict(policy.choose_config(batch,context))
