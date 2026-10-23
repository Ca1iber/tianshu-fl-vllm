# SPDX-License-Identifier: Apache-2.0
import ast,sys
from common import *
previous=EXP/'tianshu_native_dots_groups_Down64_20261023'
assert not CPU.exists() and not (BASE/'CPU_GATE_32g.json').exists();CPU.mkdir()
g=json.loads((previous/'CPU_GATE_32g.json').read_text());failure=json.loads((previous/'FAILURE_GATE_32g.json').read_text())
assert g['complete'] and failure['first_GPU_failed_before_numeric_or_timing'] and failure['rounds']==0 and failure['numeric_records']==0
assert all(sha(p)==h for p,h in g['sources'].items()) and all(sha(p)==h for p,h in g['files'].items()) and sha(MODEL)==g['checkpoint_sha256']
g['sources'].update({str(p):sha(p) for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run.sh',previous/'CPU_GATE_32g.json',previous/'FAILURE_GATE_32g.json']})
for p in BASE.glob('*.py'):ast.parse(p.read_text())
g.update(groups=GROUPS,first_version_failure_explicitly_preserved=True,unsupported_G256_excluded_from_operations=True,new_arrays_created=False)
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps(dict(complete=True,CPU_only=True,groups=GROUPS,new_arrays_created=False,sources=len(g['sources']))))
