# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;COMP=EXP/'tianshu_LM_pad64_20261022'
spec=importlib.util.spec_from_file_location('_lm_pad_parent_r3',EXP/'tianshu_native_int8_attention_service_20261022/candidate_profile.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode='candidate',check_gate=True):
 assert mode in ('control','candidate');command,environment,profile,sources=parent.prepare(root,'candidate');assert len(sources)==903 and len(command)==13 and len(profile)==62;original=dict(sources);selection=read(BASE/'selected_plan.json');gate=read(COMP/'INDEPENDENT_CPU_GATE_32g.json');assert selection['complete'] and selection['minimum_M'] in (32,48) and selection['maximum_M']==64 and gate['complete'] and gate['dynamic_range_lead'] and digest(COMP/'INDEPENDENT_CPU_GATE_32g.json')==selection['component_gate_sha256'];sources.update(read(COMP/'CPU_PREPARE_GATE_32g.json')['sources']);paths=[*BASE.rglob('*.py'),BASE/'PLAN_32g.md',BASE/'selected_plan.json',COMP/'INDEPENDENT_CPU_GATE_32g.json',COMP/'CPU_PREPARE_GATE_32g.json',COMP/'EXECUTION.sha256',Path('/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/sampler.py')];sources.update({str(p):digest(p) for p in paths});frozen_check(sources);assert all(sources[p]==h for p,h in original.items())
 if check_gate:own=read(BASE/'CPU_GATE_32g.json');assert own['complete'] and own['sources']==sources;sources[str(BASE/'CPU_GATE_32g.json')]=digest(BASE/'CPU_GATE_32g.json')
 profile['PYTHONPATH']=str(BASE/f'hooks_{mode}');environment.update(profile)
 for p,h in sources.items():
  target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,environment,profile,sources
