# SPDX-License-Identifier: Apache-2.0
import ast,sys
from common import *
assert not CPU.exists() and not (BASE/'CPU_GATE_32g.json').exists();CPU.mkdir()
g=json.loads((PARENT/'CPU_GATE_32g.json').read_text());proof=json.loads((PARENT/'INDEPENDENT_CPU_GATE_32g.json').read_text());full=json.loads((FULL/'INDEPENDENT_CPU_GATE_32g.json').read_text());raw=json.loads((Path(proof['raw_directory'])/'result.json').read_text())
assert proof['complete'] and full['complete'] and full['all_numeric_passed'] and all(r['candidate_numeric_and_determinism_pass'] for r in proof['timings']) and not proof['all_numeric_passed']
assert sha(Path(proof['raw_directory'])/'result.json')==proof['result_sha256'] and sha(Path(full['raw_directory'])/'result.json')==full['result_sha256']
assert all(sha(p)==h for p,h in g['sources'].items()) and all(sha(p)==h for p,h in g['files'].items()) and sha(MODEL)==g['checkpoint_sha256']
for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run.sh',PARENT/'CPU_GATE_32g.json',PARENT/'INDEPENDENT_CPU_GATE_32g.json',FULL/'INDEPENDENT_CPU_GATE_32g.json']:
    if p.suffix=='.py':ast.parse(p.read_text())
    g['sources'][str(p)]=sha(p)
anchors=[]
for phase in range(3):
    r=next(r for r in raw['numeric'] if r['M']==32 and r['phase']==phase and r['variant']=='W4_DOTS_RTN');assert r['full_graph_poison_eager_exact'] and sha(r['file'])==r['sha256']
    for b,p,roll in ((32,Path(r['file']),0),(64,Path(full['raw_directory'])/f'W4_DOTS_RTN_phase{phase}.npy',1)):
        a=np.load(p,mmap_mode='r');assert a.shape==(b,N) and a.dtype==np.uint16 and np.isfinite(f32(a)).all();g['files'][str(p)]=sha(p);anchors.append(dict(bucket=b,phase=phase,file=str(p),sha256=sha(p),roll=roll))
g.update(anchors=anchors,all_M=list(range(1,65)),timing_M=TIMING_MS,new_large_arrays_created=False,first_GPU_not_run=True,all_candidate_W4_old_points_pass=True,old_BF16_failures_not_erased=True)
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps(dict(complete=True,CPU_only=True,sources=len(g['sources']),anchors=len(anchors),new_large_arrays_created=False)))
