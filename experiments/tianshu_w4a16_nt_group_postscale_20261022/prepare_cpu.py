# SPDX-License-Identifier: Apache-2.0
"""Freeze files and validate syntax without importing a GPU framework."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
files=[BASE/'PLAN_32g.md',BASE/'probe_gpu.py',BASE/'verify_cpu.py',Path(__file__),ROOT/'tianshu_quantization_survey_20261022/formats_cpu.py',ROOT/'tianshu_quantization_survey_20261022/kernel_weight_only.py',ROOT/'tianshu_quantization_survey_20261022/CPU_SURVEY_GATE_32g.json',ROOT/'tianshu_quantization_survey_20261022/RESERVATION_20261022.md',ROOT/'tianshu_dense_w8a8_20261020/kernel.py',ROOT/'tianshu_dense_w8a8_20261020/GPU_GATE_32g.json']
for p in files:
 if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
files.extend([BASE/'kernel_nt_postscale.py',BASE/'compile_cpu.py',BASE/'OFFLINE_COMPILE_GATE_32g.json'])
value=dict(complete=True,CPU_only=True,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},stop_at_utc='2026-10-22T15:58:00+00:00',single_run_max_seconds=1800)
(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(value,indent=2)+'\n');print('Frozen',len(files),'files')
