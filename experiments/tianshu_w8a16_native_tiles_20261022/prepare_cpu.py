import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
files=[BASE/'PLAN_32g.md',BASE/'tiles.json',BASE/'probe_gpu.py',BASE/'verify_cpu.py',BASE/'kernel_nt_postscale.py',Path(__file__),ROOT/'tianshu_quantization_survey_20261022/formats_cpu.py',ROOT/'tianshu_dense_w8a8_20261020/kernel.py',ROOT/'tianshu_dense_w8a8_20261020/GPU_GATE_32g.json',ROOT/'tianshu_quant_ixkn_v2_20261022/PROFILE_GATE_32g.json',ROOT/'tianshu_quant_ixkn_scoped_v3_20261022/PROFILE_GATE_32g.json',ROOT/'tianshu_quant_ixkn_scoped_v3_20261022/NATIVE_CONFIG_32g.json',Path('/usr/local/lib/python3.12/site-packages/flag_gems/runtime/backend/_iluvatar/ops/mm.py')]
for p in files:
 if p.suffix=='.py':ast.parse(p.read_text(),str(p))
tiles=json.loads((BASE/'tiles.json').read_text());assert len(tiles)==12 and len({tuple(t.values()) for t in tiles})==12 and any(t==dict(bm=64,bn=64,bk=128,warps=16,stages=2) for t in tiles)
(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(dict(complete=True,CPU_only=True,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},native_reference_verified=True,GPU_performance_not_yet_measured=True),indent=2)+'\n');print('Frozen',len(files),'sources and12 tiles')
