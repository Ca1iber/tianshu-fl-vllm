import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
files=[BASE/'PLAN_32g.md',BASE/'tiles.json',BASE/'probe_gpu.py',BASE/'verify_cpu.py',BASE/'kernel_w4a8.py',BASE/'compile_cpu.py',BASE/'OFFLINE_COMPILE_GATE_32g.json',Path(__file__),ROOT/'tianshu_quantization_survey_20261022/formats_cpu.py',ROOT/'tianshu_dense_w8a8_20261020/kernel.py',ROOT/'tianshu_dense_w8a8_20261020/GPU_GATE_32g.json']
for p in files:
 if p.suffix=='.py':ast.parse(p.read_text(),str(p))
assert json.loads((BASE/'OFFLINE_COMPILE_GATE_32g.json').read_text())['complete'];tiles=json.loads((BASE/'tiles.json').read_text());assert len(tiles)==12 and all(t['bk']<=128 and 128%t['bk']==0 for t in tiles)
(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(dict(complete=True,CPU_only=True,formats=2,tiles=12,expected_GPU_cases=48,requires_exact_INT32_partial_proof=True,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n');print('Frozen W4A8 48-case experiment')
