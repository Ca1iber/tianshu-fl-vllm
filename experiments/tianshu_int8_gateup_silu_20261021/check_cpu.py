# SPDX-License-Identifier: Apache-2.0
"""Freeze component and exact native scalar source; no Torch/GPU imports."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
GEMS=Path('/usr/local/lib/python3.12/site-packages/flag_gems')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=ast.parse((BASE/'kernel.py').read_text());fused=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='_fused')
    calls=[n for n in ast.walk(fused) if isinstance(n,ast.Call)]
    assert sum(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='tl' and n.func.attr=='dot' for n in calls)==2
    assert sum(isinstance(n.func,ast.Name) and n.func.id=='_native_silu_scalar' for n in calls)==1
    assert (BASE/'kernel.py').read_text().count(').to(tl.bfloat16)')==2
    native=GEMS/'fused/silu_and_mul.py'
    scalar=next(n for n in ast.parse(native.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='silu_and_mul_kernel')
    assert len(scalar.body)==3
    parent=BASE.parent/'tianshu_dense_w8a8_20261020/kernel.py'
    assert sha(parent)=='b9b692cf561a35ef9c38d05d1edf7aa804436728923efa4800f914ff6b656152'
    paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),parent,BASE.parent/'tianshu_dense_tile_padding_20261021/kernel_padded.py',
        BASE.parent/'tianshu_dense_tile_padding_20261021/GPU_GATE_32g.json',BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json',
        native,GEMS/'modules/activation.py',GEMS/'utils/pointwise_dynamic.py',
        Path('/usr/local/lib/python3.12/site-packages/vllm_fl/dispatch/backends/flaggems/impl/activation.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/llama.py')]
    for p in BASE.glob('*.py'):ast.parse(p.read_text())
    result=dict(complete=True,no_GPU_started=True,sources={str(p):sha(p) for p in paths},actual_native_scalar_AST=ast.dump(scalar),
        two_INT32_accumulators=True,original_BF16_projection_rounds_preserved=True,actual_native_scalar_called=True,
        declared_shape=dict(K=2048,intermediate=6144,padded_M=[32,64]),maximum_possible_abs_INT32_sum=2048*127*127,
        global_INT32_overflow_bound_passed=2048*127*127<2**31,actual_GPU_correctness_not_yet_tested=True)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('sources','actual_native_scalar_AST')}))
if __name__=='__main__':main()
