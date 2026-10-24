# SPDX-License-Identifier: Apache-2.0
import csv,statistics
from common import *
gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());s=json.loads((GPU/'result.json').read_text());assert s['complete'] and s['phase']=='complete' and s['CPU_gate_sha256']==sha(BASE/'CPU_GATE_32g.json') and s['RNG_unchanged'] and s['full_input_and_weight_bytes_versions_guards_unchanged'] and s['checkpoint_unchanged'];assert len(s['numeric'])==len(MS)*len(PHASES)*len(VARIANTS)
assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items())
checks=[];ok={m:{v:True for v in VARIANTS} for m in MS}
for record in s['numeric']:
 assert sha(record['file'])==record['sha256'] and sha(record['eager_file'])==record['eager_sha256'];actual=np.load(record['file'],allow_pickle=False);eager=np.load(record['eager_file'],allow_pickle=False);m=record['M'];name=record['variant'];stable=record['all_42_poison_graph_eager_exact'] and record['all_42_finite'] and all(r['all_42_word_exact'] for r in record['repeats']);assert all(np.array_equal(actual[f'L{l}'],eager[f'L{l}']) for l in range(42))==record['all_42_poison_graph_eager_exact']
 for layer in LAYERS:
  ref=next(r for r in gate['references'] if r['layer']==layer and r['phase']==record['phase']);data=np.load(ref['file'],allow_pickle=False);target=data['original' if name==VARIANTS[0] else 'quantized'][:m];a=actual[f'L{layer}'];assert a.shape==(m,2048);r=compare(a,target);r.update(M=m,phase=record['phase'],layer=layer,variant=name,all_42_stable=bool(stable));checks.append(r);ok[m][name]&=r['pass_numeric'] and stable
rows=[]
for m in MS:
 for mode in ('graph','eager'):
  rounds=[r for r in s['rounds'] if r['M']==m and r['mode']==mode];assert len(rounds)==7 and all(r['excluded']==(r['round']==0) and r['actual_distinct_banks']==42 for r in rounds);values={v:[r['milliseconds_42banks'][v] for r in rounds[1:]] for v in VARIANTS};b,c=[statistics.median(values[v]) for v in VARIANTS];both=all(ok[m].values());rows.append(dict(M=m,mode=mode,BF16_ms=b,W4_ms=c,candidate_latency_change_percent=100*(c/b-1),baseline_numeric_qualified=ok[m][VARIANTS[0]],candidate_numeric_qualified=ok[m][VARIANTS[1]],performance_comparison_qualified=both,candidate_faster_qualified=both and c<b,baseline_spread_percent=100*(max(values[VARIANTS[0]])-min(values[VARIANTS[0]]))/b,candidate_spread_percent=100*(max(values[VARIANTS[1]])-min(values[VARIANTS[1]]))/c))
proof=dict(complete=True,scope='8 coarse M points, all42 banks determinism/timing, only3layers full FP64',all_full_FP64_checks_passed=all(r['pass_numeric'] for r in checks),independent_FP64_cells=sum(r['cells'] for r in checks),qualification_by_M=ok,checks=checks,timings=rows,all_sources_and_files_SHA_verified=True,raw_result_sha256=sha(GPU/'result.json'),raw_directory=str(GPU),verifier_source_sha256=sha(Path(__file__)),adopted=False,model_quality_qualified=False,official_benchmark=False)
(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(proof,indent=2)+'\n')
with (BASE/'KEY_TIMINGS_32g.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(dict(complete=True,all_numeric=proof['all_full_FP64_checks_passed'],FP64_cells=proof['independent_FP64_cells'],timings=rows)),flush=True)
