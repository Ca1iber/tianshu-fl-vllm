# SPDX-License-Identifier: Apache-2.0
"""Audit current payload/format, all saved numeric types and every timing round."""
import csv,statistics,subprocess
from common import *
assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());data=json.loads((GPU/'result.json').read_text())
assert data['complete'] and data['phase']=='complete' and data['CPU_gate_sha256']==sha(BASE/'CPU_GATE_32g.json')
assert gate['current_r5_all42_payloads_exact'] and gate['all42_same_full_dequantized_BF16_words']
assert data['full_inputs_native_pack_versions_guards_RNG_unchanged'] and not data['prepacking_inside_timing']
assert not data['model_executed'] and not data['official_benchmark'] and not data['adopted']
assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['libraries'].items())
assert all(sha(p)==h for p,h in gate['files'].items()) and sha(parent.MODEL)==gate['checkpoint_sha256']
for path,h in gate['sources'].items():assert sha(CPU/'source_snapshot'/Path(path).relative_to('/'))==h
assert len(data['native_packs'])==42
payload_bytes=0
for bank,pack in zip(gate['banks'],data['native_packs']):
    assert bank['layer']==pack['layer'] and pack['shape']==[256,48,32,4] and pack['dtype']=='torch.int32'
    assert pack['bytes']==6291456 and pack['params_bytes']==393216
    assert pack['raw_input_sha256']==bank['generic']['raw_high_even_sha256'] and pack['params_sha256']==bank['generic']['params_sha256']
    assert pack['converter_executed_before_timing']
    payload_bytes+=pack['bytes']+pack['params_bytes']
assert payload_bytes==data['additional_native_pack_params_bytes']
assert {(r['phase'],r['variant']) for r in data['numeric']}=={(p,v) for p in PHASES for v in VARIANTS}
refs={(r['layer'],r['phase']):r for r in gate['references']}
cells=0;all_exact=True;words_total=0
for row in data['numeric']:
    assert row['all42_finite'] and row['all42_repeat_numeric_passed'] and row['all42_Graph_numeric_passed']
    assert sha(row['file'])==row['sha256'] and sha(row['extra_file'])==row['extra_sha256']
    outputs=np.load(row['file'],allow_pickle=False);extra=np.load(row['extra_file'],allow_pickle=False)
    assert set(outputs.files)=={f'L{layer}' for layer in range(42)} and len(row['relations'])==42
    for layer in range(42):
        g=outputs[f'L{layer}'];assert g.dtype==np.uint16 and g.shape==(64,2048) and np.isfinite(f32(g)).all();words_total+=g.size
        relation=next(r for r in row['relations'] if r['layer']==layer)
        if f'E{layer}' in extra:
            e,r=extra[f'E{layer}'],extra[f'R{layer}']
            assert int(np.count_nonzero(e!=r))==relation['repeat_word_differences']
            assert int(np.count_nonzero(e!=g))==relation['Graph_word_differences']
            assert metrics(f32(r),f32(e),'BF16')==relation['repeat_vs_eager']
            assert metrics(f32(g),f32(e),'BF16')==relation['Graph_vs_eager']
        else:
            assert relation['repeat_word_differences']==relation['Graph_word_differences']==0
        assert relation['repeat_vs_eager']['passed'] and relation['Graph_vs_eager']['passed']
        all_exact=all_exact and relation['repeat_word_differences']==relation['Graph_word_differences']==0
        if layer not in LAYERS:continue
        reference=np.load(refs[(layer,row['phase'])]['file'],allow_pickle=False)['quantized']
        for execution,raw in [('eager',extra[f'E{layer}']),('repeat',extra[f'R{layer}']),('Graph',g)]:
            check=metrics(f32(raw),reference,'BF16')
            expected=next(c for c in row['selected_FP64'] if c['layer']==layer and c['execution']==execution)
            assert {'layer':layer,'execution':execution,**check}==expected and check['passed'];cells+=check['cells']
assert len(data['rounds'])==9 and [r['round'] for r in data['rounds']]==list(range(9))
for r in data['rounds']:
    assert r['replays']==20 and r['order']==(VARIANTS if r['round']%2==0 else list(reversed(VARIANTS)))
rows=[];medians={}
for variant in VARIANTS:
    raw=[r['milliseconds_42banks'][variant] for r in data['rounds']]
    assert all(np.isfinite(t) and t>0 for t in raw)
    medians[variant]=statistics.median(raw);assert medians[variant]==data['medians_ms'][variant]
    rows.append(dict(variant=variant,median_ms_42banks=medians[variant],spread_percent=100*(max(raw)-min(raw))/medians[variant]))
reduction=100*(1-medians[VARIANTS[1]]/medians[VARIANTS[0]])
result=dict(complete=True,CPU_only=True,all42_current_payload_format_and_runtime_guards_verified=True,
    all_saved_Graph_output_words=words_total,independent_selected3_FP64_cells=cells,
    all42_repeats_and_Graph_word_exact_observed=bool(all_exact),numeric_layers=LAYERS,numeric_phases=PHASES,
    all_sources_CPU_snapshots_SDK_and_checkpoint_SHA_verified=True,native_pack_metadata_count=42,
    medians_ms=medians,time_reduction_percent=reduction,component_qualified_5percent=reduction>=5,
    additional_native_pack_params_bytes=payload_bytes,result_sha256=sha(GPU/'result.json'),
    performance_scope='42realDown banks only, prepackingstartup; no model/service qualification',
    model_or_service_qualified=False,adopted=False)
with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(BASE/'GPU_CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
