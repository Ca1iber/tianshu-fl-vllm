# SPDX-License-Identifier: Apache-2.0
import ast,sys
from common import *
assert not CPU.exists() and not (BASE/'CPU_GATE_32g.json').exists();CPU.mkdir();parent=json.loads((PARENT/'CPU_GATE_32g.json').read_text());proof=json.loads((PARENT/'INDEPENDENT_CPU_GATE_32g.json').read_text());assert parent['complete'] and proof['complete'] and proof['all_numeric_passed'] and sha(Path(proof['raw_directory'])/'result.json')==proof['result_sha256']
sources=dict(parent['sources']);sources.update({str(p):sha(p) for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run.sh',PARENT/'CPU_GATE_32g.json',PARENT/'INDEPENDENT_CPU_GATE_32g.json']})
for p in BASE.glob('*.py'):ast.parse(p.read_text())
assert all(sha(p)==h for p,h in sources.items()) and all(sha(p)==h for p,h in parent['files'].items())
cells=0
for bank in parent['banks']:
 with np.load(bank['file'],allow_pickle=False) as a:
  s=f32(a['S']);z=f32(a['Z']);assert np.isfinite(s).all() and (s>0).all() and np.array_equal(z,-8*s)
  for value in range(16):assert np.array_equal(bf16(np.float32(value)*s+z),bf16(np.float32(value-8)*s));cells+=s.size
assert cells==66060288 and len(CONFIGS)==32 and sha(MODEL)==parent['checkpoint_sha256']
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
g=dict(complete=True,CPU_only=True,no_GPU_imports=True,first_GPU_not_run=True,sources=sources,files=parent['files'],banks=parent['banks'],references=parent['references'],checkpoint_sha256=parent['checkpoint_sha256'],configs=CONFIGS,all42_existing_banks_and_FP64_references_SHA_verified=True,symmetric_zero_BF16_equivalence_cells=cells,new_large_arrays_created=False,model_or_original_source_changed=False,numeric_predicate_fixed=dict(relative_L2_max=.006,per_cell_relative=.012,per_cell_RMS=.006));(BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps({k:v for k,v in g.items() if k not in ('sources','files','banks','references','configs')}))
