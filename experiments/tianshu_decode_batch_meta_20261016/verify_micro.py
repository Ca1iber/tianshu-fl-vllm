# SPDX-License-Identifier: Apache-2.0
"""Audit the downloaded controlled-buffer probe; never infer model speed."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve()
    state=json.loads((root/'result.json').read_text())
    assert state['phase']=='complete' and state['complete'] and not state['official_benchmark'] and not state['adopted']
    assert len(state['source_snapshots'])==5
    repo=Path(__file__).resolve().parents[2]
    for source,sha in state['source_snapshots'].items():
        frozen=root/'source_snapshot'/Path(source).relative_to('/')
        assert hashlib.sha256(frozen.read_bytes()).hexdigest()==sha
        if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
            assert hashlib.sha256((repo/Path(source).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha
    spec=importlib.util.spec_from_file_location('_audit_batch_render',Path(__file__).with_name('render.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=root/'source_snapshot/usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py'
    generated,derivation=module.render_source(original.read_bytes())
    assert generated==(root/'generated_prepare.py').read_text() and derivation==state['derivation']
    assert derivation['replacements']==10 and derivation['reversible_to_original_method']
    assert len(state['cases'])==20
    assert {(c['batch'],c['seed']) for c in state['cases']}=={(n,s) for n in (1,7,8,16,24,32,40,48,56,64) for s in (20261016,20261017)}
    keys={'positions','seq_lens','num_computed_tokens','optimistic_cpu','logits_indices','no_spec','slot_mapping_arguments'}
    for name in ('input_ids','query_start_loc','req_indices','query_pos','num_scheduled_tokens','discard_request_mask','num_accepted_tokens','prev_positions'):
        keys.update((name+'_gpu',name+'_cpu'))
    for case in state['cases']:
        for field in ('all_buffers_and_padding_equal','changed_inputs_equal'):
            assert set(case[field])==keys and all(v is True for v in case[field].values())
        assert case['positions_dtype']==case['req_indices_dtype']=='torch.int64'
    assert {g['name'] for g in state['guards']}=={'mixed_prefill','no_previous','reordered','async_off','speculation','Mrope'}
    assert all(g['rejected'] is True for g in state['guards'])
    assert len(state['graph_checks'])==2
    for g in state['graph_checks']:
        assert g['cleared_outputs'] and set(g['all_checks_equal'])==keys and all(v is True for v in g['all_checks_equal'].values())
    timings=state['timings'];assert len(timings)==14
    assert {(t['mode'],t['repeat']) for t in timings}=={(m,i) for m in ('original','fused') for i in range(7)}
    assert all(t['iterations']==20 and t['host_wall_ms']>0 and t['gpu_stream_ms']>0 for t in timings)
    medians={m:{k:statistics.median(t[k] for t in timings if t['mode']==m) for k in ('host_wall_ms','gpu_stream_ms')} for m in ('original','fused')}
    assert medians==state['timing_medians']
    report=dict(complete=True,all_five_source_snapshots_checked=True,generated_method_recreated_and_reversible=True,
        twenty_case_record_maps_checked=True,six_guard_checks=True,two_changed_input_graph_checks=True,
        full_buffer_values_available_for_independent_recompute=False,slot_mapping_observer_stub=True,
        timing_medians=medians,official_benchmark=False,model_speedup_not_measured=True)
    (root/'local_download_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
