# SPDX-License-Identifier: Apache-2.0
"""Prove that only the unused quantize M and GEMM M specialization changed."""
import ast,copy,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_dense_w8a8_20261020'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    old=ast.parse((PARENT/'kernel.py').read_text());new=ast.parse((BASE/'kernel_runtime_m.py').read_text())
    old_fn={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
    new_fn={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
    assert not any(isinstance(n,ast.Name) and n.id=='M' for stmt in old_fn['_quantize'].body for n in ast.walk(stmt))
    q=new_fn['_quantize'];assert 'M' not in [a.arg for a in q.args.args]
    q.args.args.insert(3,copy.deepcopy(old_fn['_quantize'].args.args[3]))
    gemm=new_fn['_gemm'];m=next(a for a in gemm.args.args if a.arg=='M');assert m.annotation is None
    m.annotation=copy.deepcopy(next(a.annotation for a in old_fn['_gemm'].args.args if a.arg=='M'))
    assert ast.dump(gemm.decorator_list[0])==ast.dump(ast.parse('@triton.jit(do_not_specialize=["M"])\ndef f():pass').body[0].decorator_list[0])
    gemm.decorator_list=copy.deepcopy(old_fn['_gemm'].decorator_list)
    calls=[n for n in ast.walk(new_fn['quantize']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript) and isinstance(n.func.value,ast.Name) and n.func.value.id=='_quantize']
    assert len(calls)==1;calls[0].args.insert(3,ast.Name(id='m',ctx=ast.Load()))
    assert ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False)
    generic=json.loads((PARENT/'CPU_GATE_32g.json').read_text());assert generic['complete'] and generic['INT32_no_overflow']
    assert all(sha(p)==digest for p,digest in generic['sources'].items())
    assert sha(PARENT/'kernel.py')=='b9b692cf561a35ef9c38d05d1edf7aa804436728923efa4800f914ff6b656152'
    for path in BASE.glob('*.py'):ast.parse(path.read_text())
    paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),PARENT/'kernel.py',PARENT/'CPU_GATE_32g.json',Path('/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py'),Path('/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py')]
    result=dict(complete=True,no_GPU_started=True,quantize_M_unused_in_actual_body=True,only_four_AST_changes=True,
        quantize_removes_M_argument_and_call=True,GEMM_runtime_M_with_do_not_specialize=True,all_computation_bodies_otherwise_identical=True,
        generic_INT32_bound_rechecked=True,actual_GPU_binary_reuse_not_yet_proven=True,sources={str(p):sha(p) for p in paths})
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
if __name__=='__main__':main()
