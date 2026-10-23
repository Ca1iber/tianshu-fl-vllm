# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
spec=importlib.util.spec_from_file_location('_LMpad_Down64_parent',EXP/'tianshu_dotsaddz_Down64_service_v2_20261023/candidate_profile.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent);digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode='candidate',check_gate=True):
 assert mode=='candidate';command,env,profile,sources=parent.prepare(root,'candidate');assert len(sources)==916;inherited=dict(sources);old_profile=dict(profile)
 component=EXP/'tianshu_native_W4_lm64_20261023';pad=EXP/'tianshu_W4_LM_pad32_64_component_v2_20261023';small=EXP/'tianshu_native_W4_lm_small_v2_20261023'
 for folder in (component,pad,small):
  proof=read(folder/'INDEPENDENT_CPU_GATE_32g.json');assert proof['complete'] and proof['result_sha256']==digest(Path(proof['raw_directory'])/'result.json')
 assert read(pad/'INDEPENDENT_CPU_GATE_32g.json')['all_64_M_3phase_full_outputs_pass'] and all(r['candidate_numeric_and_determinism_pass'] for r in read(small/'INDEPENDENT_CPU_GATE_32g.json')['timings'])
 paths=set(BASE.rglob('*.py'))|{BASE/'PLAN_32g.md',BASE/'SELECTED_LM_32g.json'}|set(component.glob('*.py'))|{component/name for name in ('PLAN_32g.md','CPU_GATE_32g.json','INDEPENDENT_CPU_GATE_32g.json','EXECUTION.sha256')}|{pad/name for name in ('kernel.py','PLAN_32g.md','CPU_GATE_32g.json','INDEPENDENT_CPU_GATE_32g.json','EXECUTION.sha256')}|{small/name for name in ('CPU_GATE_32g.json','INDEPENDENT_CPU_GATE_32g.json','EXECUTION.sha256')}
 sources.update({str(p):digest(p) for p in paths});frozen_check(sources);assert all(sources[p]==h for p,h in inherited.items())
 if check_gate:
  gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete'] and gate['sources']==sources;sources[str(BASE/'CPU_GATE_32g.json')]=digest(BASE/'CPU_GATE_32g.json')
 profile['PYTHONPATH']=str(BASE/'hooks_candidate');env.update(profile);assert all(profile[k]==v for k,v in old_profile.items() if k!='PYTHONPATH')
 for p,h in sources.items():
  target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,env,profile,sources
