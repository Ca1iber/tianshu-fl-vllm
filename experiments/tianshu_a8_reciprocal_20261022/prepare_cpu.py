import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;nt=ROOT/'tianshu_dense_w8a8_20261020/kernel.py';nn=ROOT/'tianshu_int8_weight_nn_20261021/kernel_nn.py'
def function(p,name):return next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
for name in ('_quantize','quantize'):assert ast.dump(function(nt,name))==ast.dump(function(nn,name))
files=[BASE/f for f in ['PLAN_CPU_32g.md','PLAN_GPU_32g.md','domain_cpu.py','analyze_cpu.py','CPU_FORMULA_GATE_32g.json','kernel.py','probe_gpu.py','verify_cpu.py','compile_cpu.py','OFFLINE_COMPILE_GATE_32g.json','prepare_cpu.py']]+[nt,nn]
for p in files:
 if p.suffix=='.py':ast.parse(p.read_text(),str(p))
assert json.loads((BASE/'OFFLINE_COMPILE_GATE_32g.json').read_text())['complete']
(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(dict(complete=True,CPU_only=True,production_NT_NN_quantizers_AST_equal=True,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n');print('Frozen reciprocal GPU/independent CPU validation')
