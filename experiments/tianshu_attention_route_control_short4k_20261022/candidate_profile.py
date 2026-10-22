# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
attention=load('_route_control_model',EXP/'tianshu_native_int8_attention_service_20261022/candidate_profile.py');pair=load('_route_control_fresh_candidate_quality',EXP/'tianshu_native_int8_attention_short4k_20261022/candidate_profile.py');digest,read,frozen_check=attention.digest,attention.read,attention.frozen_check;CACHE_GATE=pair.CACHE_GATE

def prepare(root,mode,scope='prefill'):
 assert mode=='control' and scope=='prefill';pair.quality_gate(scope);command,environment,profile,sources=attention.prepare(root,'control');gate=read(EXP/'tianshu_native_int8_attention_quality_results_20261022/QUALITY_GATE_32g.json');assert sources==gate['model_frozen_sources'] and len(sources)==903
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_control.sh',EXP/'tianshu_native_int8_attention_quality_results_20261022/QUALITY_GATE_32g.json'};paths.update(Path(p) for p in read(EXP/'tianshu_native_int8_down_prefill_release_v2_20261022/serve_profile_template.json')['controller_sources'])
 for module in tuple(sys.modules.values()):
  path=getattr(module,'__file__',None)
  if path and path.endswith('.py') and Path(path).is_relative_to(EXP):paths.add(Path(path).resolve())
 controllers={str(p):digest(p) for p in paths};(root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,environment,profile,sources
