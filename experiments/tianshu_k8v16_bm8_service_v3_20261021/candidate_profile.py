# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_BM8_base471_profile',BASE.parent/'tianshu_m_range_prefill_service_v2_20261021/candidate_profile.py');parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
 assert mode=='candidate'
 command,env,profile,sources=parent.prepare(root,'candidate');assert len(sources)==471 and len(profile)==62 and len(command)==13;original=dict(sources)
 selection=read(BASE/'selected_plan.json');gate=read(selection['component_gate']);assert gate['complete'] and digest(selection['component_gate'])==selection['component_gate_sha256']
 sources.update(gate['frozen_sources'])
 files=[BASE/n for n in ('policy.py','candidate_profile.py','audit_profile.py','run_service_trial.py','check_cpu.py','hooks_candidate/sitecustomize.py','PLAN_32g.md','selected_plan.json','generated_sources.json')]+list((BASE/'archived_generated').glob('*.py'))+[Path(selection['component_gate']),Path(selection['recovery_verifier_source']),Path(selection['binary_signature_source']),Path(selection['binary_code_gate'])]
 sources.update({str(p):digest(p) for p in files});assert all(sources[p]==h for p,h in original.items());frozen_check(sources)
 profile['PYTHONPATH']=str(BASE/'hooks_candidate');env.update(profile)
 for p,h in sources.items():
  target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,env,profile,sources
