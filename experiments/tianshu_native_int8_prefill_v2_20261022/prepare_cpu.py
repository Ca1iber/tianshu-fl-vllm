# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(BASE/'OFFLINE_COMPILE_GATE_32g.json')['complete'] and len(read(BASE/'OFFLINE_COMPILE_GATE_32g.json')['cases'])==12
assert read(ROOT/'tianshu_native_int8_gateup_20261022/INDEPENDENT_CPU_GATE_32g.json')['complete']
cache=read(BASE/'QUALIFIED_CACHE_GATE_32g.json');assert cache['complete'];selected=ROOT/'tianshu_m_range_prefill_service_v2_20261021/selected_plan.json';selection=read(selected);component=Path(selection['component_gate']);gate=read(component);assert gate['complete'] and gate['performance_valid'] and sha(component)==selection['component_gate_sha256']
files=[BASE/name for name in ['PLAN_32g.md','kernel.py','probe_gpu.py','verify_cpu.py','compile_cpu.py','OFFLINE_COMPILE_GATE_32g.json','prepare_cpu.py','run.sh','audit_cache_cpu_v2.py','QUALIFIED_CACHE_GATE_32g.json']]+[selected,component]
files += [ROOT/path for path in ['tianshu_int8_gateup_prefill_grouped_20261021/kernel.py','tianshu_nn_silu_large_tiles_v2_20261021/kernel.py','tianshu_m_range_prefill_service_v2_20261021/range_policy.py','tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json','tianshu_native_int8_gateup_20261022/INDEPENDENT_CPU_GATE_32g.json','tianshu_prefill_control_binary_20261022/GPU_BINARY_GATE_32g.json']]
for path in files:
 if path.suffix=='.py':ast.parse(path.read_text(),str(path))
sources={str(path):sha(path) for path in files}
for group in (gate['source_hashes'],cache['qualified_cache_files'],cache['binary_origin_sources']):
 for path,digest in group.items():assert sha(path)==digest;sources[path]=digest
(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(dict(complete=True,CPU_only=True,actual_current_M128_M512_M4096_controls_verified=True,full_binary_gate_preserved=True,sources=sources),indent=2)+'\n');print('Frozen original qualified cubins and native prefill candidate')
