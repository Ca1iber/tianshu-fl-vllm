# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;UNIT=EXP/'tianshu_unit_temperature_20261023'
s=importlib.util.spec_from_file_location('_unit_screen_exact_r4',EXP/'tianshu_W4_LMpad_service_v2_20261023/candidate_profile.py');model=importlib.util.module_from_spec(s);s.loader.exec_module(model);digest,read,frozen_check=model.digest,model.read,model.frozen_check;CACHE_GATE=model.parent.parent.parent.parent.CACHE_GATE if False else None
# Existing cache gate is explicitly exported by the qualified short controller.
s=importlib.util.spec_from_file_location('_unit_screen_cache_gate',EXP/'tianshu_W4_LMpad_short4k_v3_20261023/candidate_profile.py');old=importlib.util.module_from_spec(s);s.loader.exec_module(old);CACHE_GATE=old.CACHE_GATE

def prepare(root,mode,scope='prefill'):
 assert mode in ('control','candidate') and scope=='prefill';gate=read(UNIT/'COMPONENT_CPU_GATE_32g.json');assert gate['complete'] and gate['component_GPU_checks_passed'];assert digest(Path(gate['raw_directory'])/'result.json')==gate['result_sha256']
 command,env,profile,sources=model.prepare(root,'candidate');manifest={}
 for line in (EXP/'tianshu_W4_Down64_LM1to64_release_20261023/MODEL_SOURCES.sha256').read_text().splitlines():h,p=line.split('  ',1);manifest[p]=h
 assert sources==manifest and len(manifest)==944
 profile.update(PYTHONPATH=str(UNIT/'hooks'),TIANSHU_UNIT_TEMP_MODE=mode,TIANSHU_UNIT_TEMP_RECORD_DIR=str(root/'unit_records'));assert len(profile)==64;env.update(profile)
 paths=set(BASE.glob('*.py'))|set(UNIT.rglob('*.py'))|{UNIT/'COMPONENT_CPU_GATE_32g.json',UNIT/'PLAN_32g.md',UNIT/'PLAN_V2_32g.md',UNIT/'EXECUTION.sha256',UNIT/'EXECUTION_V2.sha256',BASE/'PLAN_32g.md',BASE/'EXECUTION.sha256'};controllers={str(p.resolve()):digest(p) for p in paths};sources.update(controllers);frozen_check(sources)
 (root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in sources.items():
  dest=root/'source_snapshot'/Path(p).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes());assert digest(dest)==h
 return command,env,profile,sources
