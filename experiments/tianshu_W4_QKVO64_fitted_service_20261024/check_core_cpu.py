# SPDX-License-Identifier: Apache-2.0
import hashlib,json,sys
from pathlib import Path
from common import reader,quant,ahash,unpack
from scope import eligible
BASE=Path(__file__).resolve().parent
checks=0
for role in ('QKV','O','Down'):
 for m in (1,8,32,63,64,65,128,4096):
  for mode in ('FULL','PIECEWISE','NONE','MISSING'):
   for uniform in (False,True):
    for lora in (False,True):
     got=eligible(role,(m,2048),True,True,mode,m,m,uniform,lora)
     assert got==(role in ('QKV','O') and m==64 and mode=='FULL' and uniform and not lora)
     checks+=1
for dtype,contiguous,tokens,reqs in ((True,False,64,64),(False,True,64,64),(True,True,63,64),(True,True,64,63)):
 for role in ('QKV','O'):
  assert not eligible(role,(64,2048),dtype,contiguous,'FULL',tokens,reqs,True,False);checks+=1
banks=[]
for role,folder in [('O','tianshu_W4_affine_fit_CPU_20261024'),('QKV','tianshu_W4_QKV_affine_fit_CPU_20261024')]:
 root=Path('/dev/shm')/folder
 proof=json.loads((BASE.parent/folder/'INDEPENDENT_CPU_GATE_32g.json').read_text())
 assert proof['complete'] and reader.sha(root/'result.json')==proof['raw_result_sha256']
 expected={b['layer']:b for b in json.loads((root/'result.json').read_text())['banks']}
 for layer in (0,20,41):
  words=reader.original(layer,role);q,s,z,packed=quant(words,'affine')
  assert ahash(words)==expected[layer]['original_sha256']
  assert [ahash(a) for a in (packed,s,z)]==[expected[layer][key] for key in ('packed_sha256','scale_sha256','zero_sha256')]
  assert (unpack(packed)==q).all()
  banks.append(dict(role=role,layer=layer,original_and_derived_all_bytes_exact=True,packed_shape=list(packed.shape)))
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
files=[*BASE.rglob('*.py'),BASE/'PLAN_CORE_32g.md',BASE.parent/'tianshu_W4_attention64_20261024/common.py',
 BASE.parent/'tianshu_W4_affine_fit_CPU_20261024/study_cpu.py',BASE.parent/'tianshu_W4_QKV_affine_fit_CPU_20261024/study_cpu.py']
result=dict(complete=True,CPU_only=True,model_ready=False,QKV_GPU_qualification_still_required=True,
 scope_checks=checks,representative_banks=banks,all84_model_buffers_not_yet_tested=True,
 sources={str(p.resolve()):reader.sha(p) for p in files},additional_model_payload_bytes=210567168,
 model_quality_qualified=False,adopted=False)
(BASE/'CORE_CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
