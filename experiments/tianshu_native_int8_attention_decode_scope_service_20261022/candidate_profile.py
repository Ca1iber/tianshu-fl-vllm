# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
spec=importlib.util.spec_from_file_location('_decode_scope_parent915',EXP/'tianshu_native_int8_attention_decode64_service_20261022/candidate_profile.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode='candidate',check_gate=True):
 assert mode in ('control','candidate');command,environment,profile,sources=parent.prepare(root,'candidate');assert len(command)==13 and len(profile)==62 and len(sources)==915;inherited=dict(sources);old_env=dict(profile);selection=read(BASE/'selected_plan.json');assert selection['complete'] and selection['quality_not_yet_established'] and selection['QKV64_scope']=='FULL_uniform64_decode' and selection['QKV128_BF16'] and selection['rows_by_role']==dict(QKV=[64,512,4096],O=[512,4096]);paths=[*BASE.rglob('*.py'),BASE/'PLAN_32g.md',BASE/'selected_plan.json',Path('/usr/local/lib/python3.12/site-packages/vllm/forward_context.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/v1/cudagraph_dispatcher.py')];sources.update({str(p):digest(p) for p in paths});frozen_check(sources);assert all(sources[p]==h for p,h in inherited.items())
 if check_gate:own=read(BASE/'CPU_GATE_32g.json');assert own['complete'] and own['sources']==sources;sources[str(BASE/'CPU_GATE_32g.json')]=digest(BASE/'CPU_GATE_32g.json')
 profile['PYTHONPATH']=str(BASE/f'hooks_{mode}');environment.update(profile);assert all(profile[k]==v for k,v in old_env.items() if k!='PYTHONPATH')
 for p,h in sources.items():
  target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,environment,profile,sources
