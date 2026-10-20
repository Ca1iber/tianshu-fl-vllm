# SPDX-License-Identifier: Apache-2.0
"""Source/ABI proof only; device numerical correctness and timing remain unproven."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_dense_w8a8_20261020'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    old=(PARENT/'kernel.py').read_text();new=(BASE/'kernel_nn.py').read_text()
    assert sha(PARENT/'kernel.py')=='b9b692cf561a35ef9c38d05d1edf7aa804436728923efa4800f914ff6b656152'
    replacements=[('W+columns[None,:]*K+absolute[:,None]','W+absolute[:,None]*N+columns[None,:]'),
        ('m,k=q.shape;n=w.shape[0]','m,k=q.shape;n=w.shape[1]'),('w.shape[1]==k','w.shape[0]==k')]
    expected=old
    for before,after in replacements:assert expected.count(before)==1;expected=expected.replace(before,after)
    assert new==expected
    functions=lambda text:{n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(old),functions(new);assert a.keys()==b.keys()
    assert ast.dump(a['_quantize'])==ast.dump(b['_quantize']) and ast.dump(a['quantize'])==ast.dump(b['quantize'])
    assert all(ast.dump(a[name].args)==ast.dump(b[name].args) for name in a)
    generic=json.loads((PARENT/'CPU_GATE_32g.json').read_text());assert generic['complete'] and generic['INT32_no_overflow']
    for p,digest in generic['sources'].items():assert sha(p)==digest
    for p in BASE.glob('*.py'):ast.parse(p.read_text())
    paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),PARENT/'kernel.py',PARENT/'CPU_GATE_32g.json',
        PARENT/'MODEL_INTEGRATION_PLAN_32g.json',BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json',*[Path(p) for p in generic['sources']]]
    result=dict(complete=True,CPU_only=True,no_GPU_started=True,exactly_three_reversible_source_edits=True,
        A_quantization_full_AST_identical=True,all_function_ABI_identical=True,integer_dot_scaling_mask_and_output_body_unchanged=True,
        parent_INT32_no_overflow_proof=True,alignment_contract_bytes=256,W_static_transpose_excluded_from_timing=True,
        GPU_numerical_and_performance_not_yet_proven=True,sources={str(p.resolve()):sha(p) for p in paths})
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
if __name__=='__main__':main()
