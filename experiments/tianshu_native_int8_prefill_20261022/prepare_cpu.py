# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
read=lambda path:json.loads(Path(path).read_text())
assert read(BASE/'OFFLINE_COMPILE_GATE_32g.json')['complete'] and len(read(BASE/'OFFLINE_COMPILE_GATE_32g.json')['cases'])==12
assert read(ROOT/'tianshu_native_int8_gateup_20261022/INDEPENDENT_CPU_GATE_32g.json')['complete']
selected=ROOT/'tianshu_m_range_prefill_service_v2_20261021/selected_plan.json';selection=read(selected);component=Path(selection['component_gate']);gate=read(component)
assert gate['complete'] and gate['performance_valid'] and hashlib.sha256(component.read_bytes()).hexdigest()==selection['component_gate_sha256']
assert all(str(m) in selection['configuration_by_M'] for m in (128,512,4096))
files=[BASE/name for name in ['PLAN_32g.md','kernel.py','probe_gpu.py','verify_cpu.py','compile_cpu.py','OFFLINE_COMPILE_GATE_32g.json','prepare_cpu.py','run.sh']]+[selected,component]
files += [ROOT/path for path in ['tianshu_int8_gateup_prefill_grouped_20261021/kernel.py','tianshu_nn_silu_large_tiles_v2_20261021/kernel.py','tianshu_m_range_prefill_service_v2_20261021/range_policy.py','tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json','tianshu_native_int8_gateup_20261022/INDEPENDENT_CPU_GATE_32g.json']]
for path in files:
 if path.suffix=='.py':ast.parse(path.read_text(),str(path))
sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
for path,digest in gate['source_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest;sources[path]=digest
(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(dict(complete=True,CPU_only=True,actual_current_M128_M512_M4096_controls_verified=True,sources=sources),indent=2)+'\n');print('Frozen native INT8 prefill component')
