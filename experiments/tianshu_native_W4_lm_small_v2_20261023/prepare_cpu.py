# SPDX-License-Identifier: Apache-2.0
import ast,sys
from common import *
previous=EXP/'tianshu_native_W4_lm_small_20261023';diagnostic=EXP/'tianshu_LM_M4_BF16_graph_diagnostic_20261023'
assert not CPU.exists() and not (BASE/'CPU_GATE_32g.json').exists();CPU.mkdir();g=json.loads((previous/'CPU_GATE_32g.json').read_text());d=json.loads((diagnostic/'DIAGNOSTIC_GATE_32g.json').read_text())
assert d['complete'] and d['all_finite'] and not d['all_FP64_pass'] and not d['all_graph_eager_word_exact']
assert all(sha(p)==h for p,h in g['sources'].items()) and all(sha(p)==h for p,h in g['files'].items()) and sha(MODEL)==g['checkpoint_sha256']
for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run.sh',previous/'CPU_GATE_32g.json',previous/'FAILURE_GATE_32g.json',diagnostic/'DIAGNOSTIC_GATE_32g.json',diagnostic/'probe_gpu.py',diagnostic/'verify_cpu.py']:
    if p.suffix=='.py':ast.parse(p.read_text())
    g['sources'][str(p)]=sha(p)
g.update(first_version_failure_preserved=True,baseline_M4_diagnostic_failure_preserved=True,no_reference_threshold_relaxed=True,first_GPU_not_run=True,new_large_arrays_created=False)
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps(dict(complete=True,CPU_only=True,MS=MS,sources=len(g['sources']),no_reference_threshold_relaxed=True)))
