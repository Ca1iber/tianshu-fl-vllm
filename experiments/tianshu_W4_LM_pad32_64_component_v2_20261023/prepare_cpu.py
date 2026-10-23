# SPDX-License-Identifier: Apache-2.0
import ast,sys
from common import *
previous=EXP/'tianshu_W4_LM_pad32_64_component_20261023'
assert not CPU.exists() and not (BASE/'CPU_GATE_32g.json').exists();CPU.mkdir();g=json.loads((previous/'CPU_GATE_32g.json').read_text());f=json.loads((previous/'FAILURE_GATE_32g.json').read_text())
assert 'OutOfMemoryError' in f['error'] and f['numeric_records']==f['rounds']==0 and sha(BASE/'kernel.py')==sha(previous/'kernel.py')
assert all(sha(p)==h for p,h in g['sources'].items()) and all(sha(p)==h for p,h in g['files'].items()) and sha(MODEL)==g['checkpoint_sha256']
for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run.sh',previous/'CPU_GATE_32g.json',previous/'FAILURE_GATE_32g.json']:
    if p.suffix=='.py':ast.parse(p.read_text())
    g['sources'][str(p)]=sha(p)
g.update(old_verifier_OOM_preserved=True,kernel_SHA_identical_to_first_version=True,verifier_shared_graph_pool=True,first_GPU_not_run=True)
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps(dict(complete=True,CPU_only=True,sources=len(g['sources']),kernel_unchanged=True,verifier_shared_pool=True)))
