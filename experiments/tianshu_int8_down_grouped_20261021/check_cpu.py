# SPDX-License-Identifier: Apache-2.0
"""CPU geometry and checkpoint-signature guards for Down INT8; no GPU numerics."""
import ast, hashlib, json
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_int8_gateup_prefill_grouped_20261021'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (BASE/'kernel.py').read_bytes()==(PARENT/'kernel.py').read_bytes()
assert (BASE/'tiles.json').read_bytes()==(PARENT/'tiles.json').read_bytes()
for p in BASE.glob('*.py'):ast.parse(p.read_text())
plans=json.loads((BASE/'tiles.json').read_text());assert len(plans)==12
sig=json.loads((BASE/'weight_signature.json').read_text());assert sig['shape']==[2048,6144]
assert sig['keys']==['model.layers.0.mlp.down_proj.weight']
assert sig['BF16_bits_sha256']=='1a6e6d547d5f0ffd08fd020690ccf75323ba4dd54181617e406a84f644cee501'
assert 6144*127*127==99096576<2**31
checks=0
for c in plans:
  for m in (1,3,8,17,63,64,65,127,128,129,255,256,511,512,1024,2048,4095,4096):
    for n in (130,2048):
      rows=(m+c['bm']-1)//c['bm'];cols=(n+c['bn']-1)//c['bn'];g=c['group_m'];seen=set()
      for pid in range(rows*cols):
        first=pid//(g*cols)*g;size=min(rows-first,g);local=pid%(g*cols);r=first+local%size;column=local//size
        assert 0<=r<rows and 0<=column<cols and (r,column) not in seen;seen.add((r,column))
      assert len(seen)==rows*cols;checks+=1
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'tiles.json',BASE/'weight_signature.json',PARENT/'kernel.py',PARENT/'tiles.json']
result=dict(complete=True,CPU_only=True,no_GPU_started=True,INT32_no_overflow=True,worst_absolute_integer_sum=99096576,
 kernel_bytes_identical_to_completed_grouped_GateUp=True,all_tile_geometry_permutations_complete_unique_in_bounds=True,geometry_cases=checks,
 model_accuracy_and_GPU_numerics_unproven=True,source_checkpoint_never_modified=True,sources={str(p.resolve()):sha(p) for p in paths})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
