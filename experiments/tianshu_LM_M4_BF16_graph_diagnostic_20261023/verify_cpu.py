# SPDX-License-Identifier: Apache-2.0
import sys,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;sys.path.insert(0,str(EXP/'tianshu_native_W4_lm_small_20261023'))
from common import sha,f32,rows,INPUTS,MODEL,compare,np
ROOT=Path('/workspace/benchmark-results/20261023_LM_M4_BF16_graph_diagnostic_GPU_tianshu');g=json.loads((BASE/'CPU_GATE_32g.json').read_text());s=json.loads((ROOT/'result.json').read_text())
assert s['complete'] and s['checkpoint_sources_full_inputs_guards_versions_RNG_unchanged'] and len(s['samples'])==27
assert sha(MODEL)==g['checkpoint_sha256'] and all(sha(p)==h for p,h in g['sources'].items()) and all(sha(p)==h for p,h in g['files'].items());checks=[]
for r in s['samples']:
    assert sha(r['file'])==r['sha256'];a=np.load(r['file']);assert a.dtype==np.uint16 and a.shape==(4,130560)
    control=next(v for v in s['samples'] if v['phase']==r['phase'] and v['mode']=='eager' and v['iteration']==0);original=np.load(control['file']);af=f32(a);of=f32(original);finite=np.isfinite(af);diff=np.abs(af.astype(np.float64)-of.astype(np.float64));ref=rows(INPUTS/f'original_FP64_{r["phase"]}.npy',4)
    checks.append(dict(phase=r['phase'],mode=r['mode'],iteration=r['iteration'],nan_cells=int(np.isnan(af).sum()),inf_cells=int(np.isinf(af).sum()),word_differences_vs_eager0=int((a!=original).sum()),numeric_differences_vs_eager0=int((af!=of).sum()),signed_zero_only_differences=int(((a!=original)&(af==of)).sum()),maximum_abs_difference_vs_eager0=float(diff.max()) if finite.all() else None,finite=bool(finite.all()),rows=[dict(row=i,nan_cells=int(np.isnan(af[i]).sum()),max_abs=float(abs(af[i]).max()) if np.isfinite(af[i]).all() else None) for i in range(4)],FP64=compare(a,ref)))
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
r=dict(complete=True,CPU_only=True,result_sha256=sha(ROOT/'result.json'),checks=checks,all_finite=all(c['finite'] for c in checks),all_FP64_pass=all(c['FP64']['passed'] for c in checks),all_graph_eager_word_exact=all(c['word_differences_vs_eager0']==0 for c in checks),adopted=False,official_benchmark=False,diagnostic_only=True)
(BASE/'DIAGNOSTIC_GATE_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='checks'}))
