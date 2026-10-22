# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
r1=load('_down_short_exact_r1_profile',EXP/'tianshu_native_int8_prefill_service_v2_20261022/candidate_profile.py');down=load('_down_short_candidate_model',EXP/'tianshu_native_int8_down_service_20261022/candidate_profile.py')
digest,read,frozen_check=r1.digest,r1.read,r1.frozen_check
CACHE_GATE=r1.CACHE_GATE
normalizer=load("_down_short_original_normalizer",EXP/"tianshu_original_command_20261016/validation_profile.py");normalized_environment=normalizer.normalized_environment

def quality_gate(scope,sources):
 for batch in (8,64):
  root=Path(f'/workspace/benchmark-results/20261022_native_int8_down_{scope}_math{batch}_tianshu');state=read(root/'result.json');quality=read(root/'accuracy_audit.json');assert state['phase']=='complete' and state['accuracy_passed'] and state['quality_batch']==batch and state['down_scope']==scope and quality['complete'] and quality['total']==105 and quality['accuracy']>.95 and quality['down_scope']==scope and state['frozen_sources']==quality['frozen_sources']==sources;frozen_check(state['frozen_sources']);frozen_check(state['controller_sources'])
  if batch==64:assert quality['actual_native_Down_runtime_banks']==42 and 4096 in quality['actual_native_Down_runtime_M_values'] and quality['actual_Down_M64_capture_banks']==(42 if scope=='all' else 0)

def prepare(root,mode,scope='all',check_quality=True):
 assert mode in ('control','candidate') and scope in ('all','prefill')
 command,environment,profile,sources=(r1.prepare(root,'candidate') if mode=='control' else down.prepare(root,'candidate',scope))
 if check_quality:
  if mode=='candidate':quality_gate(scope,sources)
  else:
   expected=read(EXP/'tianshu_native_int8_down_service_20261022/CPU_GATE_32g.json')['sources'];expected=dict(expected);expected[str(EXP/'tianshu_native_int8_down_service_20261022/CPU_GATE_32g.json')]=digest(EXP/'tianshu_native_int8_down_service_20261022/CPU_GATE_32g.json');quality_gate(scope,expected)
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_pair.sh'};paths.update(Path(p) for p in read(EXP/'tianshu_native_int8_prefill_release_v2_20261022/serve_profile_template.json')['controller_sources']);paths.update((EXP/'tianshu_native_int8_down_quality_20261022').glob('*.py'))
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,environment,profile,sources
