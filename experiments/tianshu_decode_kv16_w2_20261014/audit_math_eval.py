# SPDX-License-Identifier: Apache-2.0
"""Audit independent EvalScope completion; --ignore-errors is not sufficient."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    state = json.loads((root / 'result.json').read_text())
    assert state['phase'] == 'evaluation_finished_pending_report_audit', state['phase']
    assert state['evalscope_exit_code'] == 0
    reports = list(root.glob('evalscope/*/reports/minicpm/math_500.json'))
    assert len(reports) == 1, reports
    report_path = reports[0]
    report = json.loads(report_path.read_text())
    execution = report['execution_summary']
    assert execution['requested'] == execution['succeeded'] == report['num'] == 105
    assert execution['errored'] == 0 and execution['incomplete'] is False
    metric = next(m for m in report['metrics'] if m['identity']['name'] == 'accuracy')
    assert metric['num'] == 105
    run = report_path.parents[2]
    review_path = run / 'reviews/minicpm/math_500_Level 3.jsonl'
    prediction_path = run / 'predictions/minicpm/math_500_Level 3.jsonl'
    reviews = [json.loads(line) for line in review_path.read_text().splitlines()]
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    assert len(reviews) == len(predictions) == 105
    assert len({r['index'] for r in reviews}) == 105
    assert len({p['index'] for p in predictions}) == 105
    assert {r['index'] for r in reviews} == {p['index'] for p in predictions}
    assert all(p['model_output']['error'] is None for p in predictions)
    stop_reasons = Counter(p['model_output']['choices'][0]['stop_reason'] for p in predictions)
    scores = [r['sample_score']['score'] for r in reviews]
    assert all(s['status'] == 'success' for s in scores)
    values = [s['value']['accuracy'] for s in scores]
    assert all(v in (0, 1) for v in values)
    correct = int(sum(values))
    assert abs(metric['score'] - correct / 105) < 0.000051
    hardware = (root / 'hardware_after.txt').read_text()
    assert 'No running processes found' in hardware, 'GPU cleanup requires investigation'
    routes=state['candidate_routing_records']
    assert routes and state['experimental_candidate']=='decode-coherent'
    assert all(x['zero_initialization']['original_source_sha256']==state['experimental_kernel']['sha256'] for x in routes)
    assert state['fresh_math_for_coherent_decode']
    assert all(x['generated_source_sha256']=='4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e' for x in routes)
    reducers=state['decode_coherent_reducer_records']
    assert reducers and all(x['actual_tile']==(16 if x['batch'] in (8,16,32,40,48,56,64) else 32) for x in reducers)
    assert {(x['batch'],x['segments']) for x in routes} == {(x['batch'],x['segments']) for x in reducers}
    assert hashlib.sha256((root/'source_snapshot/candidate_sitecustomize.py').read_bytes()).hexdigest()==state['candidate_hook']['sha256']
    if state['experimental_candidate']=='decode-coherent':
        for x in routes:
            tile = 16 if x["batch"] in (8,16,32,40,48,56,64) else 32
            warps = 1 if tile == 16 else 2 if 8 <= x["batch"] <= 64 else 4
            assert x["configuration"] == dict(TILE_SIZE=tile, num_warps=warps, num_stages=1)
            assert x["compiled_metadata"]["num_warps"] == warps
        assert state["fresh_math_for_decode_kv16_w1"]
        assert any(x["batch"] == 8 and x["configuration"] == dict(TILE_SIZE=16,num_warps=1,num_stages=1) for x in routes)
        assert state["topp_launch_records"] and all(x["configuration"] == dict(num_warps=16, num_stages=1) for x in state["topp_launch_records"])
        for path, sha in state["kv16_sources"].items():
            assert hashlib.sha256((root / "source_snapshot" / ("kv16_" + Path(path).name)).read_bytes()).hexdigest() == sha
        assert state['native_prefill_routing_records'] and all(x['kernel_sha256']==state['zero_generator']['sha256']
            for x in state['native_prefill_routing_records'])
    if state.get('buckets','0')=='1':
        assert state['environment_flags']['VLLM_FL_ILUVATAR_PREFILL_BUCKETS']=='1'
        for name,meta in state['bucket_sources'].items():
            assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()==meta['sha256']
        traces=root/'bucket_trace_records'
        configured=[json.loads(p.read_text()) for p in traces.glob('configured_*.json')]
        warmed=[json.loads(p.read_text()) for p in traces.glob('warmup_completed_*.json')]
        assert any(x['enabled'] and len(x['points'])==75 for x in configured)
        assert any(x['complete'] and len(x['calls'])==75 for x in warmed)
        if state.get('gather')=='1':
            gather=[json.loads(p.read_text()) for p in traces.glob('gather_*.json')]
            assert gather and any(x['index_dtype']=='torch.int32' for x in gather)
    audit = {
        'matched_service_root':state['matched_service_root'],
        'installed_sources':state['installed_sources'],
        'fresh_math_for_coherent_decode':True,
        'fresh_math_for_decode_kv16_w1':True,
        'kv16_sources':state['kv16_sources'],
        'topp_launch_records':state['topp_launch_records'],
        'decode_coherent_reducer_records':reducers,
        'experimental_candidate':state['experimental_candidate'],
        'buckets':state.get('buckets','0'),'gather':state.get('gather','0'),
        'bucket_sources':state.get('bucket_sources'),'bucket_trace_hook':state.get('bucket_trace_hook'),
        'candidate_hook':state['candidate_hook'],
        'experimental_kernel':state['experimental_kernel'],
        'candidate_routing_records':routes,
        'native_prefill_hook':state.get('native_prefill_hook'),
        'native_prefill_routing_records':state.get('native_prefill_routing_records'),
        'zero_generator':state.get('zero_generator'),
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'candidate': state['candidate'], 'complete': True,
        'adaptive_flag': state['adaptive_flag'], 'adaptive_sha256': state['adaptive_sha256'],
        'adaptive_source': state['adaptive_source'],
        'reused_original_accuracy': False, 'correct': correct, 'total': 105,
        'accuracy': correct / 105, 'report_accuracy': metric['score'],
        'official_requirement': 'accuracy > 0.95',
        'official_requirement_passed': correct / 105 > 0.95,
        'execution_summary': execution, 'review_statuses': {'success': 105},
        'incorrect_indices': [r['index'] for r, v in zip(reviews, values) if not v],
        'generation_stop_reasons': dict(stop_reasons),
        'max_tokens_indices': [p['index'] for p in predictions
                               if p['model_output']['choices'][0]['stop_reason'] == 'max_tokens'],
        'hybrid_source': state['hybrid_source'], 'hybrid_sha256': state['hybrid_sha256'],
        'evaluation_command': state['evaluation_command'],
        'server_command': state['server_command'],
        'max_batched_tokens': state.get('max_batched_tokens', 2048),
        'worker_sha256': state.get('worker_sha256'),
        'tile_hook': state.get('tile_hook'),
        'tile_configuration': state.get('tile_configuration'),
        'tile_routing_records': state.get('tile_routing_records'),
        'gpu_idle_after': True,
        'raw_files': {str(p.relative_to(root)): {
            'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()
        } for p in (report_path, review_path, prediction_path)},
    }
    (root / 'accuracy_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
