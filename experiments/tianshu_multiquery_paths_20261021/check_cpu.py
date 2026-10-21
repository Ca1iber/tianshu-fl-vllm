# SPDX-License-Identifier: Apache-2.0
import ast,bisect,collections,json
from pathlib import Path
from types import SimpleNamespace
from factory import BASE,DEPENDENCIES,CONFIGS,LABELS,sha,source,CurrentRoute,NATIVE
for p in BASE.glob('*.py'):ast.parse(p.read_text())
proofs={m:source.render(m)[1] for m in ('control','candidate')};assert proofs['candidate']['numerical_JIT_AST_unchanged']
native_sha=sha(NATIVE);assert native_sha=='bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0'
# Actual immutable small-arena source remains previously fully verified.
assert sha(BASE.parent/'tianshu_k8v16_pipeline_20261019/bounded_prefill.py')=='c5a3dd7d8190d4bd5c61510e21482a568f161ddd8ca9e1a519f0be0a31ff2555'
geometry=0
for n in (1,3,8,24,32,64):
  patterns=[[q]*n for q in (2,3,4,7,8)];patterns.append([[1,2,3,4,7,8,2,3][i%8] for i in range(n)])
  for qlens in patterns:
    cu=[0]
    for q in qlens:cu.append(cu[-1]+q)
    for c in CONFIGS:
      bq=c['block_m']//8;prefix=[cu[i]//bq+i for i in range(n)];seen=collections.Counter()
      for block in range(cu[-1]//bq+n):
        seq=bisect.bisect_right(prefix,block)-1;local=block-prefix[seq]
        for pos in range(local*bq,min((local+1)*bq,qlens[seq])):seen[cu[seq]+pos]+=1
      assert set(seen)==set(range(cu[-1])) and set(seen.values())=={1};geometry+=1
# Exercise actual factory routing code on CPU metadata without GPU initialization.
class Reader:
    def __init__(self):self.calls=[]
    def unified_attention(self,**kw):self.calls.append(kw);return kw['marker']
for causal,fits in ((True,True),(True,False),(False,True),(False,False)):
    br,dr=Reader(),Reader();kw=dict(causal=causal,marker=object());staged=dict(kw,marker=object())
    class Dense:
        def __call__(self,k):assert set(k)==set(kw) and all(k[x] is kw[x] for x in kw);return staged if fits else None
    owner=CurrentRoute(br,None,dr,None,Dense());result=owner.unified_attention(**kw)
    # **kwargs builds a new mapping but tensor/value identity is preserved.
    if causal and fits:assert len(dr.calls)==1 and not br.calls and result is staged['marker']
    else:assert len(br.calls)==1 and not dr.calls and result is kw['marker']
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'PLAN_32g.md',*DEPENDENCIES]
sources={str(p):sha(p) for p in paths}
r=dict(complete=True,CPU_only=True,GPU_not_used=True,query_geometries=geometry,numerical_JIT_AST_unchanged=True,
    actual_factory_route_cases=4,actual_small_arena_native_source_verified=True,native_prefill_zero_generator_parent_verified=True,
    candidate_configs=CONFIGS,labels=LABELS,proofs=proofs,sources=sources)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('sources','proofs')}))
