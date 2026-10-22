# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
read=lambda path:json.loads(Path(path).read_text());sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest();previous=ROOT/'tianshu_native_int8_prefill_v2_20261022';cpu=read(previous/'INDEPENDENT_CPU_GATE_32g.json');gpu=read(previous/'GPU_GATE_32g.json');assert cpu['complete'] and gpu['complete'] and cpu['raw_result_sha256']==sha(previous/'GPU_GATE_32g.json');assert cpu['key_timings_sha256']==sha(previous/'KEY_TIMINGS_32g.csv')
files=[BASE/name for name in ['PLAN_32g.md','probe_gpu.py','verify_cpu.py','prepare_cpu.py','run.sh']]+[previous/name for name in ['GPU_GATE_32g.json','INDEPENDENT_CPU_GATE_32g.json','KEY_TIMINGS_32g.csv']]
for path in files:
 if path.suffix=='.py':ast.parse(path.read_text(),str(path))
sources={str(path):sha(path) for path in files}
for path,digest in gpu['frozen_sources'].items():assert sha(path)==digest;sources[path]=digest
(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(dict(complete=True,CPU_only=True,fixed_M=[512,4096],fixed_layout='NN',fixed_epilogue_rows=1,current_and_candidate_full_binary_hash_required=True,sources=sources),indent=2)+'\n');print('Frozen all42 actual-bank MLP holdout')
