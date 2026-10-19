"""Static preflight: preserve the original affine algorithm and cover grid tails."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
original=root.parent/'tianshu_gateup_silu_20261012/experimental_fusion.py'
candidate=root/'experimental_fusion.py'
def affine(path):
    module=ast.parse(path.read_text())
    return next(node for node in module.body if isinstance(node,ast.FunctionDef) and node.name=='_affine_gateup')
assert ast.dump(affine(original),include_attributes=False)==ast.dump(affine(candidate),include_attributes=False)
node=next(node for node in ast.walk(affine(candidate)) if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=='half')
assert ast.unparse(node.annotation)=='tl.constexpr'
for path in root.glob('*.py'):ast.parse(path.read_text())
count=0
for m,inter in ((64,6144),(17,6144),(17,130)):
    for bm,bn in ((64,128),(64,256),(32,128),(32,256)):
        half=bn//2;gn=(inter+half-1)//half;coverage=set()
        for pid in range(((m+bm-1)//bm)*gn):
            for row in range((pid//gn)*bm,min(m,(pid//gn+1)*bm)):
                for col in range((pid%gn)*half,min(inter,(pid%gn+1)*half)):
                    assert (row,col) not in coverage
                    coverage.add((row,col))
        assert len(coverage)==m*inter
        count+=1
result=dict(passed=True,static_only=True,actual_gpu_compilation_not_claimed=True,
    affine_ast_identical_to_existing_wip=True,constexpr_half=True,grid_tail_cases=count,
    original_wip_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
    candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
(root/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
