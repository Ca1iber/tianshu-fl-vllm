# SPDX-License-Identifier: Apache-2.0
import csv,statistics
from common import *
gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());s=json.loads((GPU/'result.json').read_text());assert s['complete'] and s['phase']=='complete' and s['CPU_gate_sha256']==sha(BASE/'CPU_GATE_32g.json') and s['RNG_unchanged'] and s['full_inputs_weights_versions_guards_unchanged'];assert len(s['numeric'])==len(ROLES)*len(PHASES)*len(VARIANTS);assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items())
checks=[];projections=[];ok={role:{v:True for v in VARIANTS} for role in ROLES}
for row in s['numeric']:
 assert sha(row['file'])==row['sha256'] and sha(row['eager_file'])==row['eager_sha256'];out=np.load(row['file'],allow_pickle=False);eager=np.load(row['eager_file'],allow_pickle=False);stable=bool(row['all_42_poison_graph_eager_exact'] and row['all_42_finite'] and all(r['all_42_word_exact'] for r in row['repeats']));assert all(np.array_equal(out[f'L{l}'],eager[f'L{l}']) for l in range(42))==row['all_42_poison_graph_eager_exact']
 for layer in LAYERS:
  ref=next(r for r in gate['references'] if r['role']==row['role'] and r['layer']==layer and r['phase']==row['phase']);r=np.load(ref['file'],allow_pickle=False);target='original' if row['variant']==VARIANTS[0] else 'sym' if row['variant']==VARIANTS[1] else 'affine';words=out[f'L{layer}'];assert words.shape==r[target].shape;entry=compare(words,r[target]);entry.update(role=row['role'],variant=row['variant'],layer=layer,phase=row['phase'],all42_stable=stable);checks.append(entry);ok[row['role']][row['variant']]&=entry['pass_numeric'] and stable
  if target!='original':a=f32(words).astype(np.float64);projections.append(dict(role=row['role'],variant=row['variant'],layer=layer,phase=row['phase'],relative_L2_vs_original_FP64=float(np.linalg.norm(a-r['original'])/max(np.linalg.norm(r['original']),1e-30))))
rows=[]
for role in ROLES:
 for mode in ('graph','eager'):
  rr=[r for r in s['rounds'] if r['role']==role and r['mode']==mode];assert len(rr)==7 and all(r['excluded']==(r['round']==0) and r['actual_distinct_banks']==42 for r in rr);values={v:[r['milliseconds_42banks'][v] for r in rr[1:]] for v in VARIANTS};base=statistics.median(values[VARIANTS[0]])
  for v in VARIANTS:
   med=statistics.median(values[v]);spread=100*(max(values[v])-min(values[v]))/med;qualified=bool(ok[role][VARIANTS[0]] and ok[role][v]);gain=100*(1-med/base);rows.append(dict(role=role,mode=mode,variant=v,median_42banks_ms=med,latency_change_percent=100*(med/base-1),spread_percent=spread,baseline_numeric_qualified=ok[role][VARIANTS[0]],variant_numeric_qualified=ok[role][v],performance_qualified=qualified,component_advancement_gate=bool(v!=VARIANTS[0] and mode=='graph' and qualified and gain>=5 and spread<1)))
proof=dict(complete=True,all_sources_files_SHA_verified=True,all_full_FP64_numeric_passed=all(r['pass_numeric'] for r in checks),independent_FP64_cells=sum(r['cells'] for r in checks),only3_layers_independent_FP64=True,all84_determinism_guard_and_byte_scope=True,qualification=ok,checks=checks,projection_errors=projections,timings=rows,raw_directory=str(GPU),result_sha256=sha(GPU/'result.json'),verifier_source_sha256=sha(Path(__file__)),adopted=False,model_quality_qualified=False,official_benchmark=False)
(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(proof,indent=2)+'\n')
for name,entries in [('KEY_TIMINGS_32g.csv',rows),('KEY_ERRORS_32g.csv',projections)]:
 with (BASE/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
print(json.dumps(dict(complete=True,all_numeric=proof['all_full_FP64_numeric_passed'],FP64cells=proof['independent_FP64_cells'],timings=rows)),flush=True)
