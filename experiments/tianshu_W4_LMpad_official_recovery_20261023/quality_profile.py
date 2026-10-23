# SPDX-License-Identifier: Apache-2.0
import importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_formal_CPU_recovery_original',BASE.parent/'tianshu_W4_LMpad_official4k_v3_20261023/quality_profile.py');original=importlib.util.module_from_spec(s);sys.modules[s.name]=original;s.loader.exec_module(original)

def explicit_official_C64(module):
 old=module.audit_runtime
 def audit_runtime(root,state,minimum_calls=0,batch=None):
  if batch is None and state.get('stage')=='official':
   assert state['workload']['concurrency']==64 and state['workload']['requests_per_round']==256
   batch=64
  return old(root,state,minimum_calls=minimum_calls,batch=batch)
 module.audit_runtime=audit_runtime
_seen=set()
for m in (original.full.original.own,original.full.original.dense.own):
 if id(m) not in _seen:explicit_official_C64(m);_seen.add(id(m))

def __getattr__(name):return getattr(original,name)
