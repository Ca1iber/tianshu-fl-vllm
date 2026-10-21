# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
class Original(ast.NodeTransformer):
    def visit_If(self,node):
        if isinstance(node.test,ast.Name) and node.test.id=='CAP_CONTIGUITY':return []
        if isinstance(node.test,ast.Name) and node.test.id=='SPLIT_LOAD':return [self.visit(n) for n in node.orelse]
        return self.generic_visit(node)
    def visit_FunctionDef(self,node):
        node=self.generic_visit(node);node.args.args=[a for a in node.args.args if a.arg not in ('CAP_CONTIGUITY','SPLIT_LOAD')];return node
assert not torch.cuda.is_initialized()
parent=BASE.parent/'tianshu_int8_nn_gateup_silu_20261021'
old=next(n for n in ast.parse((parent/'kernel.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_fused')
new=next(n for n in ast.parse((BASE/'kernel.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_fused')
assert ast.dump(old,include_attributes=False)==ast.dump(Original().visit(new),include_attributes=False)
for p in BASE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
for bn in (64,128):
    for block in range(6144//bn):
        pair=block*bn+torch.arange(2*bn)%bn+(torch.arange(2*bn)//bn)*6144
        explicit=torch.cat((block*bn+torch.arange(bn),block*bn+torch.arange(bn)+6144))
        assert torch.equal(pair,explicit)
sources=dict(json.loads((parent/'CPU_GATE_32g.json').read_text())['sources'])
for p in [*BASE.glob('*.py'),BASE/'run_debug.sh',BASE/'PLAN_32g.md',parent/'CPU_GATE_32g.json']:sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items()) and not torch.cuda.is_initialized()
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(dict(complete=True,CPU_only=True,CUDA_not_initialized=True,original_default_flags_AST_identical=True,split_load_column_sets_exact=True,sources=sources),indent=2)+'\n')
print(json.dumps(dict(complete=True,CPU_only=True,original_default_flags_AST_identical=True,source_count=len(sources))))
