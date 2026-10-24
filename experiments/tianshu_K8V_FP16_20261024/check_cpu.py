# SPDX-License-Identifier: Apache-2.0
"""Independent full BF16/FP16 domain and exact source-scope checks; no GPU imports."""
from datetime import datetime, timezone
import hashlib,json,math,struct,sys
from pathlib import Path
import codec
base=Path(__file__).resolve().parent
out=Path('/dev/shm/tianshu_K8V_FP16_CPU_20261024')
out.mkdir(exist_ok=False)
assert 'torch' not in sys.modules and 'triton' not in sys.modules
counts=dict(BF16_finite=0,BF16_nonfinite=0,finite_input_overflow=0,finite_nonzero_underflow=0,FP16_finite=0,FP16_nonfinite=0)
for bits in range(65536):
    value=codec.bf16_scalar(bits)
    got=codec.encode_bf16(bits); reference=codec.reference_encode(value)
    if math.isnan(value):
        assert got&0x7c00==0x7c00 and got&1023 and reference&1023
    else: assert got==reference,(bits,got,reference)
    counts['BF16_finite' if math.isfinite(value) else 'BF16_nonfinite']+=1
    if math.isfinite(value):
        counts['finite_input_overflow']+=int(got&0x7fff==0x7c00)
        counts['finite_nonzero_underflow']+=int(value!=0 and got&0x7fff==0)
    value=struct.unpack('<e',struct.pack('<H',bits))[0]
    reference=codec.reference_decode(bits)
    if math.isnan(value): assert math.isnan(reference)
    else: assert value==reference and math.copysign(1.,value)==math.copysign(1.,reference)
    counts['FP16_finite' if math.isfinite(value) else 'FP16_nonfinite']+=1
proofs={}
for kind,parent,render in [('reader',codec.READER,codec.render_reader),('writer',codec.WRITER,codec.render_writer)]:
    text,proofs[kind]=render(parent.read_bytes())
    compile(text,str(out/(kind+'.py')),'exec')
    (out/(kind+'.py')).write_text(text)
paths=[base/p for p in ('codec.py','check_cpu.py','probe_gpu.py','launch_sweep.py','verify_gpu_cpu.py','component_launch.py','value_planes.py','PLAN_32g.md','domain_gpu.py')]
paths += [codec.READER,codec.WRITER]
gate=dict(complete=True,CPU_only=True,created_utc=datetime.now(timezone.utc).isoformat(),
    all65536_BF16_encodings_independent_rounding_passed=True,all65536_FP16_decodes_independent_real_reference_passed=True,
    counts=counts,source_scope_proof=proofs,no_Torch_Triton_import=True,physical_page_bytes=12288,
    memory_capacity_gain_claimed=False,model_or_service_qualified=False,adopted=False,
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},raw_root=str(out))
(base/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
(out/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k!='sources'},indent=2))
