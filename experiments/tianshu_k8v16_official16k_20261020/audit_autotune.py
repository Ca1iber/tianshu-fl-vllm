# SPDX-License-Identifier: Apache-2.0
"""Summarize logged post-ready tuning; absence of logs is not a hardware trace."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261020_k8v16_official16k_tianshu')
assert (ROOT/'benchmark_ended.txt').exists()
path=ROOT/'server.log'
lines=path.read_text().splitlines()
ready=[index for index,line in enumerate(lines) if 'Application startup complete' in line]
assert len(ready)==1
candidates=[]
completed=[]
http_headers=0
for index,line in enumerate(lines[ready[0]+1:],ready[0]+1):
    if 'POST /v1/completions' in line and '200 OK' in line:
        http_headers+=1
    match=re.search(r'Autotuning kernel (\S+) with config (.*)',line)
    if match:
        candidates.append(dict(line=index+1,function=match[1],configuration=match[2]))
    match=re.search(r'Triton autotuning for function (\S+) finished after ([0-9.]+)s; (.*)',line)
    if match:
        completed.append(dict(line=index+1,function=match[1],duration_seconds=float(match[2]),
                              prior_HTTP200_headers=http_headers,detail=match[3]))
counts=dict(Counter(row['function'] for row in completed))
result=dict(complete=True,post_ready_candidate_log_count=len(candidates),post_ready_completed_tuning_count=len(completed),
            completed_functions=counts,logged_completed_tuning_seconds=sum(row['duration_seconds'] for row in completed),
            observed_completed_functions_all_index=(bool(completed) and set(counts)=={'_index_linearized_jit_function'}),
            no_GEMM_autotune_record_observed=bool(completed) and set(row['function'] for row in completed+candidates)=={'_index_linearized_jit_function'},
            zero_runtime_autotune=not candidates and not completed,
            HTTP200_headers_are_stream_start_not_completion_or_reliable_round_boundary=True,
            no_claim_of_absolute_absence_of_unlogged_tuning=True,
            completed=completed,candidates=candidates,server_log_path=str(path),
            server_log_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            analyzer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
text=json.dumps(result,indent=2)+'\n'
(ROOT/'AUTOTUNE_AUDIT_32g.json').write_text(text)
(BASE/'AUTOTUNE_AUDIT_32g.json').write_text(text)
print(json.dumps({key:result[key] for key in ('completed_functions','logged_completed_tuning_seconds','zero_runtime_autotune','no_GEMM_autotune_record_observed')}))
