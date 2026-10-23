# SPDX-License-Identifier: Apache-2.0
import ast,sys
from common import *
assert not CPU.exists() and not (BASE/'CPU_GATE_32g.json').exists();CPU.mkdir()
g=json.loads((PARENT/'CPU_GATE_32g.json').read_text());proof=json.loads((PARENT/'INDEPENDENT_CPU_GATE_32g.json').read_text())
assert g['complete'] and proof['complete'] and proof['all_numeric_passed'] and sha(Path(proof['raw_directory'])/'result.json')==proof['result_sha256']
assert all(sha(p)==h for p,h in g['sources'].items()) and all(sha(p)==h for p,h in g['files'].items()) and sha(MODEL)==g['checkpoint_sha256']
extra=[*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run.sh',PARENT/'CPU_GATE_32g.json',PARENT/'INDEPENDENT_CPU_GATE_32g.json',Path('/usr/local/lib/python3.12/site-packages/flag_gems/runtime/backend/_iluvatar/ops/linear.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/vocab_parallel_embedding.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/utils.py')]
for p in extra:
    if p.suffix=='.py':ast.parse(p.read_text())
    g['sources'][str(p)]=sha(p)
for phase in range(3):
    x=rows(INPUTS/f'X{phase}.npy',63);assert x.shape==(63,K) and np.isfinite(f32(x)).all() and (x[0]!=0).any() and (x[1]==0).all()
g.update(no_GPU_imports=True,all_new_sources_frozen=True,MS=MS,actual_BF16_control='UnquantizedEmbeddingMethod.apply under FlagGems enable-all',row_roll=1,new_large_arrays_created=False,first_GPU_not_run=True,adopted=False)
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps(dict(complete=True,CPU_only=True,MS=MS,sources=len(g['sources']),new_large_arrays_created=False)))
