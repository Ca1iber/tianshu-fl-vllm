# SPDX-License-Identifier: Apache-2.0
"""Full saved outputs for selected banks against independent references, all timings."""
import csv,statistics,subprocess
from common import *
assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());data=json.loads((GPU/'result.json').read_text())
assert data['complete'] and data['phase']=='complete' and data['CPU_gate_sha256']==sha(BASE/'CPU_GATE_32g.json')
assert data['full_inputs_versions_guards_RNG_unchanged'] and not data['model_executed'] and not data['official_benchmark'] and not data['adopted']
assert gate['current_r5_Down_all42_payloads_exact'] and len(gate['banks'])==42
assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['libraries'].items())
assert all(sha(p)==h for p,h in gate['files'].items()) and sha(parent.MODEL)==gate['checkpoint_sha256']
for path,h in gate['sources'].items():assert sha(CPU/'source_snapshot'/Path(path).relative_to('/'))==h
assert {(row['phase'],row['variant']) for row in data['numeric']}=={(p,v) for p in PHASES for v in VARIANTS}
refs={(r['layer'],r['phase']):r for r in gate['references']}
cells=0;quality=[];rows=[]
for row in data['numeric']:
    assert row['all42_repeat_eager_exact'] and row['all42_poison_Graph_eager_exact'] and row['all42_finite']
    path=Path(row['file']);assert path.is_relative_to(GPU) and sha(path)==row['sha256']
    arrays=np.load(path,allow_pickle=False);assert set(arrays.files)=={f'L{layer}' for layer in range(42)}
    kind='FP16' if row['variant']==VARIANTS[1] else 'BF16'
    assert row['actual_dtype']==('torch.float16' if kind=='FP16' else 'torch.bfloat16')
    for layer in range(42):
        raw=arrays[f'L{layer}'];assert raw.dtype==np.uint16 and raw.shape==(64,2048)
        values=raw.view(np.float16).astype(np.float32) if kind=='FP16' else f32(raw)
        assert np.isfinite(values).all()
        if layer not in LAYERS:continue
        reference=np.load(refs[(layer,row['phase'])]['file'],allow_pickle=False)
        target=reference['BF16' if row['variant']==VARIANTS[0] else 'FP16']
        actual=metrics(values,target,kind);expected=next(c for c in row['selected_FP64'] if c['layer']==layer)
        assert {'layer':layer,**actual}==expected and actual['passed'];cells+=actual['cells']
        orig=reference['original'];delta=values.astype(np.float64)-orig
        quality.append(dict(layer=layer,phase=row['phase'],variant=row['variant'],
            relative_L2_to_original=float(np.linalg.norm(delta)/max(np.linalg.norm(orig),1.e-30)),
            max_abs_to_original=float(abs(delta).max())))
assert len(data['rounds'])==9 and [r['round'] for r in data['rounds']]==list(range(9))
for r in data['rounds']:
    assert r['replays']==20 and r['order']==(VARIANTS if r['round']%2==0 else list(reversed(VARIANTS)))
medians={}
for variant in VARIANTS:
    raw=[r['milliseconds_42banks'][variant] for r in data['rounds']]
    assert all(np.isfinite(t) and t>0 for t in raw)
    medians[variant]=statistics.median(raw);assert medians[variant]==data['medians_ms'][variant]
    rows.append(dict(variant=variant,median_ms_42banks=medians[variant],
        spread_percent=100*(max(raw)-min(raw))/medians[variant],
        native_scope='diagnostic only; excludes casting' if variant==VARIANTS[1] else 'actual input/output dtype contract'))
baseline=medians[VARIANTS[0]];full=medians[VARIANTS[2]]
result=dict(complete=True,CPU_only=True,all42_runtime_repeats_Graph_finite_guards_verified=True,
    independent_FP64_cells=cells,numeric_layers=LAYERS,numeric_phases=PHASES,
    all_sources_CPU_snapshots_and_SDK_hashes_verified=True,current_r5_all42_W4_payloads_exact=True,
    medians_ms=medians,raw_FP16_time_reduction_percent=100*(1-medians[VARIANTS[1]]/baseline),
    full_cast_pipeline_time_reduction_percent=100*(1-full/baseline),
    full_cast_pipeline_qualified_5percent=full<=.95*baseline,
    additional_FP16_scale_zero_bytes=data['actual_added_FP16_scale_zero_bytes'],
    projection_errors_to_original=quality,result_sha256=sha(GPU/'result.json'),
    performance_scope='42realDown banks only; selected3banks fullFP64; no model/service speed qualification',
    model_or_service_qualified=False,adopted=False)
with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(BASE/'GPU_CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='projection_errors_to_original'},indent=2))
