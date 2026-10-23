# SPDX-License-Identifier: Apache-2.0
import csv,sys
from common import *
g=json.loads((BASE/'CPU_GATE_32g.json').read_text());s=json.loads((GPU/'result.json').read_text())
assert s['complete'] and s['checkpoint_sources_and_inputs_unchanged'] and s['all_guards_versions_full_bytes_preserved'] and s['RNG_unchanged']
assert sha(MODEL)==g['checkpoint_sha256']==s['checkpoint_sha256_after'] and all(sha(p)==h for p,h in g['sources'].items()) and all(sha(p)==h for p,h in g['files'].items())
assert len(s['numeric'])==len(MS)*6 and len(s['rounds'])==len(MS)*14 and [r['M'] for r in s['routes']]==MS
for route in s['routes']:
    assert route['actual_embedding_apply'] and route['vendor_linear_entered']
    for frame in route['frames']:assert sha(frame['file'])==frame['sha256']
checks=[];seen=set()
for r in s['numeric']:
    m=r['M'];phase=r['phase'];name=r['variant'];key=(m,phase,name);assert key not in seen and m in MS and phase in range(3) and name in VARIANTS;seen.add(key)
    assert r['full_graph_poison_eager_exact'] and r['all_output_finite'] and sha(r['file'])==r['sha256'];a=np.load(r['file'],allow_pickle=False);assert a.shape==(m,N) and a.dtype==np.uint16
    reference=rows(INPUTS/f'{"original" if name==VARIANTS[0] else "quantized"}_FP64_{phase}.npy',m);check=compare(a,reference)
    original=rows(INPUTS/f'original_FP64_{phase}.npy',m);error=float(np.linalg.norm(f32(a).astype(np.float64)-original)/max(np.linalg.norm(original),1e-30))
    checks.append(dict(M=m,phase=phase,variant=name,quantization_relative_L2_vs_original=error,**check))
rows_out=[]
for m in MS:
    for mode in ('eager','graph'):
        rr=[r for r in s['rounds'] if r['M']==m and r['mode']==mode];assert len(rr)==7 and {r['round'] for r in rr}==set(range(7))
        for r in rr:
            assert r['replays']==20 and r['excluded']==(r['round']==0) and r['order']==(VARIANTS[::-1] if r['round']%2 else VARIANTS) and set(r['milliseconds'])==set(VARIANTS)
        med={v:float(np.median([r['milliseconds'][v] for r in rr if r['round']>0])) for v in VARIANTS};assert all(np.isfinite(v) and v>0 for v in med.values())
        sample=[r['milliseconds'][VARIANTS[1]] for r in rr if r['round']>0]
        rows_out.append(dict(M=m,mode=mode,control_ms=med[VARIANTS[0]],candidate_ms=med[VARIANTS[1]],time_saved_ms=med[VARIANTS[0]]-med[VARIANTS[1]],speedup_pct=100*(med[VARIANTS[0]]/med[VARIANTS[1]]-1),candidate_hot_spread_pct=100*(max(sample)-min(sample))/np.mean(sample),numeric_pass=all(c['passed'] for c in checks if c['M']==m)))
with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows_out[0]),lineterminator='\n');w.writeheader();w.writerows(rows_out)
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
r=dict(complete=True,CPU_only=True,all_numeric_passed=all(c['passed'] for c in checks),full_FP64_checked_cells=sum(c['cells'] for c in checks),checks=checks,timings=rows_out,raw_directory=str(GPU),result_sha256=sha(GPU/'result.json'),adopted=False,model_quality_not_tested=True,official_benchmark=False)
(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='checks'}))
