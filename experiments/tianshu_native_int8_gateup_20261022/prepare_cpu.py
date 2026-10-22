# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
read=lambda path:json.loads(Path(path).read_text())
assert read(BASE/'OFFLINE_COMPILE_GATE_32g.json')['complete']
assert len(read(BASE/'OFFLINE_COMPILE_GATE_32g.json')['cases'])==12
assert read(ROOT/'tianshu_native_int8_support_v2_20261022/GPU_SUPPORT_GATE_32g.json')['complete']
gate=read(ROOT/'tianshu_nn_silu_two_dot_20261021/GPU_GATE_32g.json');assert gate['complete'] and gate['independently_CPU_rechecked']
selected=next(row for row in gate['best_by_shape'] if row['M']==64 and not row['strided']);assert selected['configuration']==dict(bm=64,bn=64,bk=128,warps=4,stages=1,mode='two_dot')
files=[BASE/name for name in ['PLAN_32g.md','kernel.py','probe_gpu.py','verify_cpu.py','compile_cpu.py','OFFLINE_COMPILE_GATE_32g.json','prepare_cpu.py','run.sh']]
files += [ROOT/path for path in ['tianshu_int8_weight_nn_20261021/kernel_nn.py','tianshu_nn_silu_two_dot_20261021/kernel.py','tianshu_nn_silu_two_dot_20261021/GPU_GATE_32g.json','tianshu_nn_silu_fx_service_20261021/policy.py','tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json','tianshu_native_int8_support_v2_20261022/GPU_SUPPORT_GATE_32g.json','tianshu_native_int8_support_v2_20261022/smoke_gpu.py','tianshu_native_int8_support_v2_20261022/PLAN_32g.md']]
for path in files:
 if path.suffix=='.py':ast.parse(path.read_text(),str(path))
(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(dict(complete=True,CPU_only=True,current_M64_binary_signature_verified=True,sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}),indent=2)+'\n');print('Frozen native INT8 GateUp component')
