# SPDX-License-Identifier: Apache-2.0
import ast,json,sys
from pathlib import Path
import candidate_profile as p
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261023_QKV64_keep128_service_CPU_tianshu')
assert not ROOT.exists() and not (BASE/'CPU_GATE_32g.json').exists();ROOT.mkdir()
original=p.read(BASE.parent/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');sources0=None;checks=[]
from decode_scope import eligible_m64
scope_cases=0;accepted=0
for mode in ('NONE','PIECEWISE','FULL'):
 for tokens in (0,1,8,63,64,65,128):
  for reqs in (None,0,1,8,63,64,65,128):
   for uniform in (False,True):
    for lora in (False,True):
     got=eligible_m64(mode,tokens,reqs,uniform,lora);scope_cases+=1;accepted+=int(got)
     if got:assert (mode,tokens,reqs,uniform,lora)==('FULL',64,64,True,False)
assert scope_cases==672 and accepted==1
for path in BASE.rglob('*.py'):ast.parse(path.read_text())
for mode in ('control','candidate'):
 root=ROOT/mode;root.mkdir();command,environment,profile,sources=p.prepare(root,mode,check_gate=False)
 assert all(sources[k]==v for k,v in original['frozen_sources'].items()) and command==original['server_command']
 full={**profile,'TRITON_CACHE_DIR':environment['TRITON_CACHE_DIR']};old=original['environment'];assert len(full)==len(old)==63
 actual={k:v.replace(str(root),'<OUTPUT>') for k,v in full.items() if k!='PYTHONPATH'}
 target={k:v.replace(original['old_output_root'],'<OUTPUT>') for k,v in old.items() if k!='PYTHONPATH'}
 assert actual==target,{k:[actual.get(k),target.get(k)] for k in actual.keys()|target.keys() if actual.get(k)!=target.get(k)}
 previous_path=BASE.parent/'tianshu_native_int8_attention_decode64_service_20261022/CPU_GATE_32g.json';previous=dict(p.read(previous_path)['sources']);previous[str(previous_path)]=p.digest(previous_path);assert len(previous)==915 and all(sources[k]==v for k,v in previous.items())
 assert sources0 is None or sources==sources0;sources0=sources;checks.append(dict(mode=mode,source_count=len(sources),environment_count=len(full),server_command=command,PYTHONPATH=profile['PYTHONPATH'],all_other62_keys_exact_after_output_root_normalization=True))
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,scope_metadata_cases=672,scope_accepted_cases=1,original915_sources_exact_subset=True,sources=sources0,checks=checks);(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
