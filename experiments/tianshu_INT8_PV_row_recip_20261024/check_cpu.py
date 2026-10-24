# SPDX-License-Identifier: Apache-2.0
"""Prove exact helper edit scope, scalar RNE boundaries and freeze all CPU sources."""
import ast,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import codec,domain_cpu,launch_sweep
base=Path(__file__).resolve().parent
assert not (base/'CPU_GATE_32g.json').exists()
raw=codec.HELPER.read_text();fast=codec.FAST_HELPER.read_text()
before='    quantized=libdevice.nearbyint(tl.div_rn(weighted,scale[:,None]))'
after='    reciprocal=tl.div_rn(1.,scale)\n    quantized=libdevice.nearbyint(weighted*reciprocal[:,None])'
assert raw.count(before)==fast.count(after)==1
restored=fast.replace(after,before).replace(
    'Per-row exact reciprocal followed by FP32 multiply; explicit changed rounding',
    'Per-query/tile P quantization after V-scale folding')
assert restored==raw
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
old=json.loads((codec.PARENT/'CPU_GATE_32g.json').read_text())
for path in (codec.READER,codec.WRITER,codec.LAYOUT,codec.REFERENCE,codec.HELPER):
    assert old['sources'][str(path)]==sha(path)
primitive=json.loads(codec.PRIMITIVE_GATE.read_text())
assert primitive['complete'] and primitive['frozen_sources'][str(codec.HELPER)]==sha(codec.HELPER)
writer=json.loads(codec.WRITER_GATE.read_text())
assert writer['complete'] and writer['source_snapshots'][str(codec.WRITER)]==sha(codec.WRITER)
root=Path('/dev/shm/tianshu_INT8_PV_row_recip_CPU_20261024');root.mkdir(exist_ok=False)
proofs={}
for variant in ('strict','row_recip'):
    text,proofs[variant]=codec.render_reader(codec.READER.read_bytes(),variant)
    compile(text,str(root/f'reader_{variant}.py'),'exec')
    (root/f'reader_{variant}.py').write_text(text)
checked=changed=0
for exponent in (-80,-40,-10,0,10,40,80):
    peak=np.float32(2.**exponent);scale=np.float32(peak/np.float32(127.))
    row=[float(peak),0.]
    for k in range(127):
        value=np.float32(np.float32(k+.5)*scale)
        row += [float(np.nextafter(value,np.float32(-np.inf))),float(value),float(np.nextafter(value,np.float32(np.inf)))]
    strict,ss=domain_cpu.probability_quant(row,'strict')
    recip,rs=domain_cpu.probability_quant(row,'row_recip')
    assert ss==rs and all(0<=q<=127 for q in strict+recip)
    for quantized in (strict,recip):
        assert all(abs(value-q*ss)<=.5001*ss for value,q in zip(row,quantized))
    checked+=len(row);changed+=sum(a!=b for a,b in zip(strict,recip))
for variant in ('strict','row_recip'):
    q,scale=domain_cpu.probability_quant([0.]*32,variant);assert q==[0]*32 and scale>0
paths=set(base.glob('*.py'))|{base/'PLAN_32g.md',codec.PARENT/'codec.py',codec.PARENT/'CPU_GATE_32g.json',
    codec.READER,codec.WRITER,codec.LAYOUT,codec.REFERENCE,codec.HELPER,codec.PRIMITIVE_GATE,codec.WRITER_GATE,
    base.parent/'tianshu_soft8_mixed_planes_20261019/writer.py',
    base.parent/'tianshu_dense_w8a8_20261020/kernel.py',
    base.parent/'tianshu_V_e5m2_launch_sweep_20261024/launch_sweep.py'}
for path in paths:
    if path.suffix=='.py':ast.parse(path.read_text())
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
gate=dict(complete=True,CPU_only=True,created_utc=datetime.now(timezone.utc).isoformat(),
    exact_inverse_helper_edit_proof=True,source_scope_proofs=proofs,
    scales_preserved_exact_formula=True,rounding_contract_explicitly_changed=True,
    CPU_boundary_neighbors_checked=checked,CPU_codes_differing_from_strict=changed,
    quantization_error_bound='.5001scale; independent real residual',
    declared_grid=launch_sweep.CONFIGS,sources={str(path):sha(path) for path in paths},
    GPU_component_not_run=True,model_or_service_qualified=False,adopted=False,raw_root=str(root))
(base/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
(root/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','source_scope_proofs')},indent=2))
