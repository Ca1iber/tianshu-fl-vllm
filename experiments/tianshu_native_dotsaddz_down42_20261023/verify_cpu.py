# SPDX-License-Identifier: Apache-2.0
import csv,sys
from common import *
gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());state=json.loads((GPU/'result.json').read_text());assert state['complete'] and state['checkpoint_unchanged'] and state['all_input_weight_scale_zero_full_bytes_versions_guards_preserved'] and state['RNG_unchanged'];assert sha(MODEL)==gate['checkpoint_sha256']==state['checkpoint_sha256_after'];assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items());assert len(state['numeric'])==6 and len(state['quantization_errors'])==126
records=[]
for record in state['numeric']:
 assert sha(record['file'])==record['sha256'] and record['all_42_graph_poison_eager_exact'] and record['all_outputs_finite']
 with np.load(record['file'],allow_pickle=False) as out:
  assert set(out.files)=={f'L{x}' for x in range(42)}
  for layer in range(42):assert out[f'L{layer}'].shape==(64,2048) and np.isfinite(f32(out[f'L{layer}'])).all()
  for layer in LAYERS:
   reference=next(v for v in gate['references'] if v['layer']==layer and v['phase']==record['phase']);assert sha(reference['file'])==reference['sha256']
   with np.load(reference['file'],allow_pickle=False) as ref:r=compare(out[f'L{layer}'],ref['original' if record['variant']=='BF16_qualified' else 'quantized'])
   records.append(dict(variant=record['variant'],layer=layer,phase=record['phase'],**r))
rows=[]
for mode in ('graph','eager'):
 rounds=[r for r in state['rounds'] if r['mode']==mode];assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7))
 for r in rounds:assert r['replays']==10 and r['banks_per_replay']==42 and set(r['milliseconds_42banks'])=={'BF16_qualified','W4_DOTS_RTN'} and r['excluded']==(r['round']==0)
 med={name:float(np.median([r['milliseconds_42banks'][name] for r in rounds if r['round']>0])) for name in ('BF16_qualified','W4_DOTS_RTN')};assert all(np.isfinite(v) and v>0 for v in med.values());rows.append(dict(mode=mode,control_ms_42banks=med['BF16_qualified'],candidate_ms_42banks=med['W4_DOTS_RTN'],time_saved_ms=med['BF16_qualified']-med['W4_DOTS_RTN'],speedup_pct=100*(med['BF16_qualified']/med['W4_DOTS_RTN']-1)))
with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
assert not any(k.split('.')[0] in ('torch','triton','vllm','flag_gems') for k in sys.modules)
g=dict(complete=True,CPU_only=True,all_numeric_passed=all(r['pass_numeric'] for r in records),full_FP64_checked_cells=sum(r['cells'] for r in records),checks=records,timings=rows,maximum_quantization_relative_L2_vs_BF16=max(r['relative_L2'] for r in state['quantization_errors']),all_42_full_weights_and_3phase_outputs_checked=True,checkpoint_and_sources_unchanged=True,raw_directory=str(GPU),result_sha256=sha(GPU/'result.json'),new_precision=True,model_quality_not_tested=True,official_benchmark=False,adopted=False)
(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps({k:v for k,v in g.items() if k!='checks'}))
