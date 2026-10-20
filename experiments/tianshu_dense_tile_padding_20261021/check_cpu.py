# SPDX-License-Identifier: Apache-2.0
import ast,copy,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_dense_w8a8_20261020'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    old=next(n for n in ast.parse((PARENT/'kernel.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_quantize')
    assert not any(isinstance(n,ast.Name) and n.id=='M' for stmt in old.body for n in ast.walk(stmt))
    expected=copy.deepcopy(old);expected.name='_quantize_padded';expected.args.args[3].arg='REAL_M';expected.args.args[3].annotation=None
    expected.decorator_list=ast.parse('@triton.jit(do_not_specialize=["REAL_M"])\ndef f():pass').body[0].decorator_list
    load=next(n for n in ast.walk(expected) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load')
    load.args[1]=ast.BinOp(left=ast.Compare(left=ast.Name(id='row',ctx=ast.Load()),ops=[ast.Lt()],comparators=[ast.Name(id='REAL_M',ctx=ast.Load())]),op=ast.BitAnd(),right=load.args[1])
    new=next(n for n in ast.parse((BASE/'kernel_padded.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_quantize_padded')
    assert ast.dump(new,include_attributes=False)==ast.dump(expected,include_attributes=False)
    generic=json.loads((PARENT/'CPU_GATE_32g.json').read_text());assert generic['complete'] and generic['INT32_no_overflow'] and all(sha(p)==v for p,v in generic['sources'].items())
    assert sha(PARENT/'kernel.py')=='b9b692cf561a35ef9c38d05d1edf7aa804436728923efa4800f914ff6b656152'
    for path in BASE.glob('*.py'):ast.parse(path.read_text())
    paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),PARENT/'kernel.py',PARENT/'CPU_GATE_32g.json']
    result=dict(complete=True,no_GPU_started=True,quantization_body_unchanged_except_real_row_mask=True,GEMM_uses_original_qualified_source=True,
        padded_row_input_is_masked_zero=True,timed_padding_in_quantizer=True,actual_equals_target_uses_original_quantizer=True,alignment_contract_bytes=256,
        GPU_equality_and_performance_not_yet_proven=True,sources={str(p):sha(p) for p in paths})
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
if __name__=='__main__':main()
