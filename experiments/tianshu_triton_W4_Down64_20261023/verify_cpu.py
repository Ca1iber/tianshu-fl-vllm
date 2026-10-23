# SPDX-License-Identifier: Apache-2.0
import csv,sys
from common import *
gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());state=json.loads((GPU/'result.json').read_text())
assert state['complete'] and state['checkpoint_unchanged'] and state['all_input_weight_scale_zero_full_bytes_versions_guards_preserved'] and state['RNG_unchanged']
assert sha(MODEL)==gate['checkpoint_sha256']==state['checkpoint_sha256_after']
assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items())
assert len(state['compilation'])==32 and [v['config'] for v in state['compilation']]==CONFIGS
compiled={v['config']['name'] for v in state['compilation'] if v['status']=='compiled'}
assert all(v['status'] in ('compiled','compile_or_resource_failed') for v in state['compilation'])
variants=['BF16_qualified','W4_DOTS_RTN']+[c['name'] for c in CONFIGS if c['name'] in compiled]
assert state['active_variants']==variants and len(state['numeric'])==3*len(variants) and len(state['quantization_errors'])==126
assert set(state['resources'])==compiled
for resource in state['resources'].values():
    assert resource['artifacts']
    for a in resource['artifacts'].values():assert sha(a['file'])==a['sha256'] and Path(a['file']).stat().st_size==a['bytes']
records=[];seen=set()
for record in state['numeric']:
    key=(record['variant'],record['phase']);assert key not in seen and key[0] in variants and key[1] in PHASES;seen.add(key)
    assert sha(record['file'])==record['sha256'] and record['all_42_graph_poison_eager_exact'] and record['all_outputs_finite']
    with np.load(record['file'],allow_pickle=False) as out:
        assert set(out.files)=={f'L{x}' for x in range(42)}
        for layer in range(42):assert out[f'L{layer}'].dtype==np.uint16 and out[f'L{layer}'].shape==(64,2048) and np.isfinite(f32(out[f'L{layer}'])).all()
        for layer in LAYERS:
            reference=next(v for v in gate['references'] if v['layer']==layer and v['phase']==record['phase']);assert sha(reference['file'])==reference['sha256']
            with np.load(reference['file'],allow_pickle=False) as ref:r=compare(out[f'L{layer}'],ref['original' if record['variant']=='BF16_qualified' else 'quantized'])
            records.append(dict(variant=record['variant'],layer=layer,phase=record['phase'],**r))
valid={name:all(r['pass_numeric'] for r in records if r['variant']==name) for name in variants}
assert valid['BF16_qualified'] and valid['W4_DOTS_RTN']
rows=[]
for mode in ('graph','eager'):
    rounds=[r for r in state['rounds'] if r['mode']==mode];assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7))
    for r in rounds:
        rid=r['round'];order=variants[rid%len(variants):]+variants[:rid%len(variants)]
        if rid%2:order.reverse()
        assert r['order']==order and r['replays']==10 and r['banks_per_replay']==42 and set(r['milliseconds_42banks'])==set(variants) and r['excluded']==(rid==0)
    med={name:float(np.median([r['milliseconds_42banks'][name] for r in rounds if r['round']>0])) for name in variants}
    assert all(np.isfinite(v) and v>0 for v in med.values())
    for name in variants:
        samples=[r['milliseconds_42banks'][name] for r in rounds if r['round']>0]
        rows.append(dict(mode=mode,variant=name,numeric_pass=valid[name],median_ms_42banks=med[name],speedup_vs_BF16_pct=100*(med['BF16_qualified']/med[name]-1),speedup_vs_native_W4_pct=100*(med['W4_DOTS_RTN']/med[name]-1),hot_spread_pct=100*(max(samples)-min(samples))/np.mean(samples),n_regs=state['resources'].get(name,{}).get('n_regs'),n_spills=state['resources'].get(name,{}).get('n_spills'),shared=state['resources'].get(name,{}).get('shared')))
with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
custom_rows=[r for r in rows if r['mode']=='graph' and r['variant'] in compiled and r['numeric_pass']]
best=min(custom_rows,key=lambda r:r['median_ms_42banks']) if custom_rows else None
assert not any(k.split('.')[0] in ('torch','triton','vllm','flag_gems') for k in sys.modules)
g=dict(complete=True,CPU_only=True,all_compiled_candidates_numeric_passed=all(valid.values()),candidate_numeric_status=valid,full_FP64_checked_cells=sum(r['cells'] for r in records),checks=records,timings=rows,best_graph_candidate_best_of_32=best,requested_configs=32,compiled_configs=len(compiled),compile_failures=[r for r in state['compilation'] if r['status']!='compiled'],all_42_full_weights_and_3phase_outputs_shape_finite_graph_eager_checked=True,full_FP64_reference_layers=LAYERS,checkpoint_and_sources_unchanged=True,raw_directory=str(GPU),result_sha256=sha(GPU/'result.json'),model_quality_not_tested=True,official_benchmark=False,adopted=False)
(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n')
print(json.dumps({k:v for k,v in g.items() if k not in ('checks','timings','candidate_numeric_status','compile_failures')}))
