# SPDX-License-Identifier: Apache-2.0
"""Source reversibility and permutation-of-tiles proof, no device numerics."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_int8_gateup_prefill_20261021'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
gate=json.loads((PARENT/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['INT32_no_overflow']
assert all(sha(p)==v for p,v in gate['sources'].items())
old=(PARENT/'kernel.py').read_text();new=(BASE/'kernel.py').read_text()
before='''    tiles_n=tl.cdiv(N,BN)
    rows=(pid//tiles_n)*BM+tl.arange(0,BM)
    columns=(pid%tiles_n)*BN+tl.arange(0,BN)'''
after='''    tiles_n=tl.cdiv(N,BN)
    tiles_m=tl.cdiv(M,BM)
    per_group=GROUP_M*tiles_n
    group=pid//per_group
    first_m=group*GROUP_M
    size_m=tl.minimum(tiles_m-first_m,GROUP_M)
    local=pid%per_group
    tile_m=first_m+(local%size_m)
    tile_n=local//size_m
    rows=tile_m*BM+tl.arange(0,BM)
    columns=tile_n*BN+tl.arange(0,BN)'''
expected=old
for a,b in [('RETURN_INT:tl.constexpr):','RETURN_INT:tl.constexpr,GROUP_M:tl.constexpr):'),("return_int,num_warps=","return_int,config.get('group_m',1),num_warps="),(before,after)]:
    assert expected.count(a)==1;expected=expected.replace(a,b)
assert expected==new
functions=lambda text:{n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
a,b=functions(old),functions(new)
assert all(ast.dump(a[k])==ast.dump(b[k]) for k in ('_quantize','quantize'))
for p in BASE.glob('*.py'):ast.parse(p.read_text())
tiles=json.loads((BASE/'tiles.json').read_text());assert len(tiles)==12
checks=0
for c in tiles:
    for m in (1,3,8,17,63,64,65,127,128,129,255,256,511,512,1024,1152,1536,2048,4095,4096):
      for n in (130,12288):
        rows=(m+c['bm']-1)//c['bm'];cols=(n+c['bn']-1)//c['bn'];g=c['group_m'];seen=set()
        for pid in range(rows*cols):
            first=(pid//(g*cols))*g;size=min(rows-first,g);local=pid%(g*cols);r=first+local%size;col=local//size
            assert 0<=r<rows and 0<=col<cols and (r,col) not in seen;seen.add((r,col))
        assert seen=={(r,col) for r in range(rows) for col in range(cols)};checks+=1
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'tiles.json',PARENT/'CPU_GATE_32g.json',*[Path(p) for p in gate['sources']]]
result=dict(complete=True,CPU_only=True,no_GPU_started=True,INT32_no_overflow=True,exactly_three_reversible_source_changes=True,quantizer_AST_unchanged=True,
    all_tile_geometry_permutations_complete_unique_in_bounds=True,geometry_cases=checks,group_sizes=[1,8],M=[128,129,256,512,1024,1152,2048,4096],
    GPU_numeric_timing_and_model_unproven=True,sources={str(p.resolve()):sha(p) for p in paths})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
