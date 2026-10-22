import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
files=[BASE/'PLAN_32g.md',BASE/'profile_one.py',Path(__file__),ROOT/'tianshu_quantization_survey_20261022/formats_cpu.py',ROOT/'tianshu_dense_w8a8_20261020/kernel.py',ROOT/'tianshu_w8a16_nt_postscale_20261022/kernel_nt_postscale.py',ROOT/'tianshu_w4a16_nt_group_postscale_20261022/kernel_nt_postscale.py']
for p in files:
 if p.suffix=='.py':ast.parse(p.read_text(),str(p))
(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(dict(complete=True,CPU_only=True,GPU_profiled=False,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},requires_future_independent_GPU_gates=True),indent=2)+'\n')
print('CPU-only profiler preparation complete')
