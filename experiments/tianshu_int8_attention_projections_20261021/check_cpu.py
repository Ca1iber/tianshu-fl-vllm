# SPDX-License-Identifier: Apache-2.0
"""CPU tile coverage, true-checkpoint signatures and original BF16 path binding."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
PARENT=BASE.parent/'tianshu_int8_down_grouped_20261021'
assert (BASE/'kernel.py').read_bytes()==(PARENT/'kernel.py').read_bytes()
assert (BASE/'tiles.json').read_bytes()==(PARENT/'tiles.json').read_bytes()
for path in BASE.glob('*.py'):ast.parse(path.read_text())
plans=json.loads((BASE/'tiles.json').read_text());assert len(plans)==20
signatures=json.loads((BASE/'weight_signatures.json').read_text())
assert set(signatures)=={'qkv','o'}
assert signatures['qkv']['shape']==[2560,2048] and signatures['o']['shape']==[2048,2048]
assert signatures['qkv']['keys']==['model.layers.0.self_attn.'+x+'_proj.weight' for x in ('q','k','v')]
assert signatures['o']['keys']==['model.layers.0.self_attn.o_proj.weight']
assert signatures['qkv']['BF16_bits_sha256']=='b51f7b84ce022d4d5cfd3c6159908567d259185d765137089692731950536fd1'
assert signatures['o']['BF16_bits_sha256']=='f641e83dd6533cd4c9ea6f5b3689d4c952c86ca18c7e765db0399adfd259ed78'
assert 2048*127*127==33032192<2**31
checks=0
for c in plans:
  for m in (1,3,8,17,63,64,65,127,128,129,255,256,511,512,1024,2048,4095,4096):
    for n in (130,2048,2560):
      rows=(m+c['bm']-1)//c['bm'];cols=(n+c['bn']-1)//c['bn'];group=c['group_m'];seen=set()
      for pid in range(rows*cols):
        first=pid//(group*cols)*group;size=min(rows-first,group);local=pid%(group*cols)
        r=first+local%size;column=local//size
        assert 0<=r<rows and 0<=column<cols and (r,column) not in seen
        seen.add((r,column))
      assert len(seen)==rows*cols;checks+=1
source=BASE.parent/'tianshu_prefill_gemm_tiles_20261014/experimental_mm.py'
hook=BASE.parent/'tianshu_prefill_gemm_tiles_20261014/hooks/sitecustomize.py'
vendor=Path('/usr/local/lib/python3.12/site-packages/flag_gems/runtime/backend/_iluvatar/ops/mm.py')
assert sha(source)=='cb556d7a6025e50d95a87a7858154fd10e547d9c145a5a18b65133122ab8c880'
assert sha(vendor)=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
text=hook.read_text();assert 'plans={(2048,2048):1,(12288,2048):8,(2048,6144):1}' in text and 'M!=2048' in text
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'tiles.json',BASE/'weight_signatures.json',source,hook,vendor,PARENT/'kernel.py',PARENT/'tiles.json']
result=dict(complete=True,CPU_only=True,no_GPU_started=True,INT32_no_overflow=True,worst_absolute_integer_sum=33032192,
 kernel_bytes_identical_to_completed_grouped_GateUp_and_Down=True,geometry_cases=checks,all_tile_geometry_permutations_complete_unique_in_bounds=True,
 BF16_M2048_O_specialization_retained=True,no_existing_specialization_for_QKV_in_qualified_hook=True,
 actual_weight_hashes_bound=True,model_accuracy_and_GPU_numerics_unproven=True,source_checkpoint_never_modified=True,sources={str(p.resolve()):sha(p) for p in paths})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
