# SPDX-License-Identifier: Apache-2.0
"""New QKV64 source graph with the entire qualified471 map unchanged."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;PARENT=BASE.parent/'tianshu_m_range_prefill_service_v2_20261021'
s=importlib.util.spec_from_file_location('_qkv_NN_base471_profile',PARENT/'candidate_profile.py');parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
 assert mode=='candidate'
 command,env,profile,sources=parent.prepare(root,'candidate');assert len(sources)==471 and len(profile)==62 and len(command)==13
 original=dict(sources);old_env=dict(profile);selection=read(BASE/'selected_plan.json');gate=read(selection['component_gate']);assert gate['complete'] and gate['service_candidate_worth_testing'] and digest(selection['component_gate'])==selection['component_gate_sha256']
 sources.update(gate['frozen_sources'])
 files=[BASE/'policy.py',BASE/'candidate_profile.py',BASE/'audit_profile.py',BASE/'run_service_trial.py',BASE/'check_cpu.py',BASE/'hooks_candidate/sitecustomize.py',BASE/'PLAN_32g.md',BASE/'selected_plan.json',Path(selection['component_gate']),BASE.parent/'tianshu_qkv_prefill_int8_service_v2_20261021/qkv_prefill_weight_pack.py',BASE.parent/'tianshu_qkv_prefill_int8_service_v2_20261021/CHECKPOINT_SIGNATURES_32g.json']
 sources.update({str(p):digest(p) for p in files});assert all(sources[p]==h for p,h in original.items());frozen_check(sources)
 profile['PYTHONPATH']=str(BASE/'hooks_candidate');env.update(profile);assert all(profile[k]==v for k,v in old_env.items() if k!='PYTHONPATH')
 for p,h in sources.items():
  target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,env,profile,sources
