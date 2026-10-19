# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json
from pathlib import Path
from plans import SHAPES,configurations
base=Path(__file__).resolve().parent;cases=0
for m in (64,2048):
 for name,n,k in SHAPES:
  for c in configurations(m,n):
   assert k%(c['bk']*c['splits'])==0
   gm=(m+c['bm']-1)//c['bm'];gn=(n+c['bn']-1)//c['bn'];out=set()
   for pid in range(gm*gn):
    if c['group_m']==1:i,j=divmod(pid,gn)
    else:
     width=c['group_m']*gn;first=(pid//width)*c['group_m'];size=min(gm-first,c['group_m'])
     i=first+(pid%width)%size;j=(pid%width)//size
    assert (i,j) not in out;out.add((i,j))
   assert out=={(i,j) for i in range(gm) for j in range(gn)}
   covered=[segment*c['bk']+step*c['bk']*c['splits']+lane for segment in range(c['splits']) for step in range(k//(c['bk']*c['splits'])) for lane in range(c['bk'])]
   assert sorted(covered)==list(range(k));cases+=1
assert cases==16
for p in base.glob('*.py'):ast.parse(p.read_text(),str(p))
r=dict(complete=True,CPU_only=True,tile_and_splitK_coverage_cases=cases,GPU_measured=False,
 sources={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in ('kernel.py','plans.py','check_cpu.py','probe.py','run_probe.py')})
(base/'CPU_GATE_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
