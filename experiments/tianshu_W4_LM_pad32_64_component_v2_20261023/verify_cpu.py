# SPDX-License-Identifier: Apache-2.0
import csv,sys
from common import *
g=json.loads((BASE/'CPU_GATE_32g.json').read_text());s=json.loads((GPU/'result.json').read_text());assert s['complete'] and s['checkpoint_sources_full_inputs_versions_guards_RNG_unchanged'] and len(s['numeric'])==192 and len(s['rounds'])==112
assert sha(MODEL)==g['checkpoint_sha256'] and all(sha(p)==h for p,h in g['sources'].items()) and all(sha(p)==h for p,h in g['files'].items());checks=[];seen=set()
for phase in range(3):
    reference=rows(INPUTS/f'quantized_FP64_{phase}.npy',64);anchors={};stats={}
    for b in (32,64):
        a=next(a for a in g['anchors'] if a['bucket']==b and a['phase']==phase);assert sha(a['file'])==a['sha256'];out=np.roll(np.load(a['file']),a['roll'],axis=0);anchors[b]=out;af=f32(out).astype(np.float64);ref=reference[:b];error=af-ref
        assert np.isfinite(af).all() and np.isfinite(ref).all()
        stats[b]=(np.cumsum(np.sum(error*error,axis=1)),np.cumsum(np.sum(ref*ref,axis=1)),np.maximum.accumulate(np.max(abs(error)-.012*abs(ref),axis=1)))
    for m in range(1,65):
        r=next(r for r in s['numeric'] if r['M']==m and r['phase']==phase);key=(m,phase);assert key not in seen;seen.add(key);b=bucket(m);assert r['bucket']==b and r['padded_X_full_prefix_tail_guards_exact'] and r['eager_all_output_finite'] and r['graph_all_output_finite']
        expected=ahash(anchors[b][:m]);assert r['expected_words_sha256']==r['eager_words_sha256']==r['graph_words_sha256']==expected
        err2,ref2,margin=stats[b];rms=float(np.sqrt(ref2[m-1]/(m*N)));rl=float(np.sqrt(err2[m-1]/max(ref2[m-1],1e-60)));passed=bool(rl<=.006 and margin[m-1]<=.006*rms)
        checks.append(dict(M=m,phase=phase,bucket=b,full_prefix_hash_verified=True,relative_L2=rl,per_cell_max_margin=float(margin[m-1]),RMS=rms,cells=m*N,passed=passed))
timings=[]
for m in TIMING_MS:
    for mode in ('eager','graph'):
        rr=[r for r in s['rounds'] if r['M']==m and r['mode']==mode];assert len(rr)==7 and {r['round'] for r in rr}==set(range(7))
        for r in rr:assert r['replays']==20 and r['excluded']==(r['round']==0) and r['order']==(['padded','direct'] if r['round']%2 else ['direct','padded']) and set(r['milliseconds'])=={'direct','padded'}
        med={v:float(np.median([r['milliseconds'][v] for r in rr if r['round']>0])) for v in ('direct','padded')};assert all(np.isfinite(v) and v>0 for v in med.values());timings.append(dict(M=m,bucket=bucket(m),mode=mode,direct_W4_ms=med['direct'],padded_W4_ms=med['padded'],overhead_ms=med['padded']-med['direct'],overhead_pct=100*(med['padded']/med['direct']-1)))
with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(timings[0]),lineterminator='\n');w.writeheader();w.writerows(timings)
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
r=dict(complete=True,CPU_only=True,all_64_M_3phase_full_outputs_pass=all(c['passed'] for c in checks),full_FP64_prefix_logical_cells=sum(c['cells'] for c in checks),prefixes_share_reference_cells_not_unique=True,checks=checks,timings=timings,raw_directory=str(GPU),result_sha256=sha(GPU/'result.json'),adopted=False,model_quality_not_tested=True,official_benchmark=False)
(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='checks'}))
