# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,tempfile
from pathlib import Path
BASE=Path(__file__).resolve().parent
from scope import eligible
proof=json.loads((BASE.parent/'tianshu_W4_attention64_20261024/CPU_GATE_32g.json').read_text())
banks=[]
for b in proof['banks']:
    if b['role']=='O':
        selected=b['variants']['affine']
        banks.append(dict(prefix=f"model.layers.{b['layer']}.self_attn.o_proj",layer=b['layer'],
            original_sha256=b['original_sha256'],**{k:selected[k] for k in ('packed_sha256','scale_sha256','zero_sha256','payload_bytes')}))
assert len(banks)==42 and sum(b['payload_bytes'] for b in banks)==93585408
(BASE/'SELECTED_O_32g.json').write_text(json.dumps(dict(complete=True,group_size=128,format='affine',banks=banks,
    component_CPU_gate_sha256=__import__('hashlib').sha256((BASE.parent/'tianshu_W4_attention64_20261024/CPU_GATE_32g.json').read_bytes()).hexdigest()),indent=2)+'\n')
checks=0
for role in ('O','QKV','Down'):
 for m in (1,8,32,63,64,65,128,4096):
  for mode in ('FULL','PIECEWISE','NONE','MISSING'):
   for uniform in (False,True):
    for lora in (False,True):
     result=eligible(role,(m,2048),True,True,mode,m,m,uniform,lora)
     assert result==(role=='O' and m==64 and mode=='FULL' and uniform and not lora)
     checks+=1
for change in ((True,False,64,64),(False,True,64,64),(True,True,63,64),(True,True,64,63)):
 assert not eligible('O',(64,2048),change[0],change[1],'FULL',change[2],change[3],True,False)
 checks+=1
spec=importlib.util.spec_from_file_location('_O64_CPU_profile',BASE/'candidate_profile.py')
profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
root=Path(tempfile.mkdtemp(prefix='O64_CPU_',dir='/dev/shm'))
command,env,values,sources=profile.prepare(root,'candidate',check_gate=False)
assert len(command)==13 and len(values)==64 and len({**values,'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']})==65
assert 'torch' not in __import__('sys').modules
result=dict(complete=True,CPU_only=True,first_model_GPU_not_run=True,sources=sources,
    scope_cases_passed=checks,original_r4_sources=944,source_count=len(sources),environment_count=65,
    additional_payload_bytes=93585408,banks=42,command=command)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
