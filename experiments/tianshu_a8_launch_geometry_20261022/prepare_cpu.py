# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
nt=ROOT/'tianshu_dense_w8a8_20261020/kernel.py';nn=ROOT/'tianshu_int8_weight_nn_20261021/kernel_nn.py'
def function(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
for name in ('_quantize','quantize'):assert ast.dump(function(nt,name))==ast.dump(function(nn,name))
compile_gate=json.loads((BASE/'OFFLINE_COMPILE_GATE_32g.json').read_text());assert compile_gate['complete'] and len(compile_gate['cases'])==42
files=[BASE/name for name in ['PLAN_32g.md','configs.json','kernel.py','probe_gpu.py','verify_cpu.py','compile_cpu.py','OFFLINE_COMPILE_GATE_32g.json','prepare_cpu.py','run.sh']]+[nt,nn]
for path in files:
 if path.suffix=='.py':ast.parse(path.read_text(),str(path))
(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(dict(complete=True,CPU_only=True,production_NT_NN_quantizers_AST_equal=True,sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}),indent=2)+'\n')
print('Frozen original-A8 geometry experiment')
