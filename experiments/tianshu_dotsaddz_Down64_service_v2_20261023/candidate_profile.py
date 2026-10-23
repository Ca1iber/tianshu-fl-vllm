# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
spec=importlib.util.spec_from_file_location('_dots_Down64_original_r3',EXP/'tianshu_native_int8_attention_service_20261022/candidate_profile.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent);digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode='candidate',check_gate=True):
 assert mode=='candidate';command,environment,profile,sources=parent.prepare(root,'candidate');release=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');assert sources==release['frozen_sources'] and len(sources)==903 and command==release['server_command'];inherited=dict(sources);old_profile=dict(profile);component=EXP/'tianshu_native_dotsaddz_down42_20261023';g=read(component/'INDEPENDENT_CPU_GATE_32g.json');assert g['complete'] and g['all_numeric_passed'] and g['result_sha256']==digest(Path(g['raw_directory'])/'result.json')
 paths=set(BASE.rglob('*.py'))|{BASE/'PLAN_32g.md',BASE/'SELECTED_BANKS_32g.json',component/'CPU_GATE_32g.json',component/'INDEPENDENT_CPU_GATE_32g.json',EXP/'tianshu_native_dotsaddz_formats_20261023/FUNCTIONAL_GATE_32g.json',Path('/usr/local/lib/python3.12/site-packages/torch/_library/custom_ops.py')};sources.update({str(p):digest(p) for p in paths});frozen_check(sources);assert all(sources[p]==h for p,h in inherited.items())
 if check_gate:own=read(BASE/'CPU_GATE_32g.json');assert own['complete'] and own['sources']==sources;sources[str(BASE/'CPU_GATE_32g.json')]=digest(BASE/'CPU_GATE_32g.json')
 profile['PYTHONPATH']=str(BASE/'hooks_candidate');environment.update(profile);assert all(profile[k]==v for k,v in old_profile.items() if k!='PYTHONPATH')
 for p,h in sources.items():
  target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,environment,profile,sources
