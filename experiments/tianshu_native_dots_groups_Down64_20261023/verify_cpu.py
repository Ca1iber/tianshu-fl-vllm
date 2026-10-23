# SPDX-License-Identifier: Apache-2.0
import csv,sys
from common import *
gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());state=json.loads((GPU/'result.json').read_text());variants=['BF16_qualified']+[f'W4_G{g}' for g in GROUPS]
assert state['complete'] and state['checkpoint_unchanged'] and state['all_input_weight_scale_zero_full_bytes_versions_guards_preserved'] and state['RNG_unchanged']
assert sha(MODEL)==gate['checkpoint_sha256']==state['checkpoint_sha256_after'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items())
assert len(state['numeric'])==12 and len(state['quantization_errors'])==378
checks=[];seen=set()
for record in state['numeric']:
    key=(record['variant'],record['phase']);assert key not in seen and key[0] in variants and key[1] in PHASES;seen.add(key)
    assert sha(record['file'])==record['sha256'] and record['all_42_graph_poison_eager_exact'] and record['all_outputs_finite']
    with np.load(record['file'],allow_pickle=False) as out:
        assert set(out.files)=={f'L{x}' for x in range(42)}
        for layer in range(42):assert out[f'L{layer}'].dtype==np.uint16 and out[f'L{layer}'].shape==(64,2048) and np.isfinite(f32(out[f'L{layer}'])).all()
        for layer in LAYERS:
            group=128 if record['variant']=='BF16_qualified' else int(record['variant'].split('G')[1]);reference=next(r for r in gate['references'] if r['group']==group and r['layer']==layer and r['phase']==record['phase']);assert sha(reference['file'])==reference['sha256']
            with np.load(reference['file'],allow_pickle=False) as ref:r=compare(out[f'L{layer}'],ref['original' if record['variant']=='BF16_qualified' else 'quantized'])
            checks.append(dict(variant=record['variant'],layer=layer,phase=record['phase'],**r))
valid={v:all(r['pass_numeric'] for r in checks if r['variant']==v) for v in variants};rows=[]
for mode in ('graph','eager'):
    rounds=[r for r in state['rounds'] if r['mode']==mode];assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7))
    for r in rounds:
        rid=r['round'];order=variants[rid%4:]+variants[:rid%4]
        if rid%2:order.reverse()
        assert r['order']==order and r['replays']==10 and r['banks_per_replay']==42 and set(r['milliseconds_42banks'])==set(variants) and r['excluded']==(rid==0)
    med={v:float(np.median([r['milliseconds_42banks'][v] for r in rounds if r['round']>0])) for v in variants};assert all(np.isfinite(v) and v>0 for v in med.values())
    for v in variants:
        samples=[r['milliseconds_42banks'][v] for r in rounds if r['round']>0];errors=[r['relative_L2'] for r in state['quantization_errors'] if f"W4_G{r['group']}"==v]
        rows.append(dict(mode=mode,variant=v,numeric_pass=valid[v],median_ms_42banks=med[v],speedup_vs_BF16_pct=100*(med['BF16_qualified']/med[v]-1),speedup_vs_G128_pct=100*(med['W4_G128']/med[v]-1),hot_spread_pct=100*(max(samples)-min(samples))/np.mean(samples),maximum_projection_relative_L2_vs_BF16=max(errors) if errors else 0))
with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
assert not any(k.split('.')[0] in ('torch','triton','vllm','flag_gems') for k in sys.modules)
g=dict(complete=True,CPU_only=True,all_numeric_passed=all(valid.values()),numeric_status=valid,full_FP64_checked_cells=sum(r['cells'] for r in checks),checks=checks,timings=rows,checkpoint_and_sources_unchanged=True,raw_directory=str(GPU),result_sha256=sha(GPU/'result.json'),model_quality_not_tested=True,official_benchmark=False,adopted=False)
(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps({k:v for k,v in g.items() if k!='checks'}))
