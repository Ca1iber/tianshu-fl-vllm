# SPDX-License-Identifier: Apache-2.0
"""CPU proof: same qualified kernel, explicit large M, no GPU imports."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_int8_gateup_prefill_grouped_20261021'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (BASE/'kernel.py').read_bytes()==(PARENT/'kernel.py').read_bytes()
plans=json.loads((BASE/'tiles.json').read_text());original=json.loads((PARENT/'tiles.json').read_text())
assert len(plans)==2 and all(p in original and p['group_m']==8 for p in plans)
assert 2048*127*127<2**31 and 16384*12288<2**31
for p in BASE.glob('*.py'):ast.parse(p.read_text())
checks=0
for c in plans:
  for m in (4096,8192,16384):
    nr=(m+c['bm']-1)//c['bm'];nc=(12288+c['bn']-1)//c['bn'];g=c['group_m'];seen=set()
    for pid in range(nr*nc):
      first=pid//(g*nc)*g;size=min(nr-first,g);local=pid%(g*nc);r=first+local%size;n=local//size
      assert 0<=r<nr and 0<=n<nc and (r,n) not in seen;seen.add((r,n))
    assert len(seen)==nr*nc;checks+=1
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'tiles.json',PARENT/'kernel.py',PARENT/'tiles.json',BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json']
gate=dict(complete=True,CPU_only=True,no_GPU_started=True,INT32_no_overflow=True,actual_original_kernel_bytes_identical=True,geometry_cases=checks,
 M=[4096,8192,16384],request_context_still_4096_not_full_16K_benchmark=True,GPU_numerics_and_scheduling_gain_unproven=True,
 sources={str(p.resolve()):sha(p) for p in paths})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
