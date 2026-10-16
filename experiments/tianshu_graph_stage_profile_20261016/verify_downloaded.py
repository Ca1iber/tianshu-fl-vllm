# SPDX-License-Identifier: Apache-2.0
"""Recompute capability failures and optional unprofiled model stages offline."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

from audit_profile import audit_runtime


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capability(root, version):
    state = read(root / 'result.json')
    assert state['phase'] == 'failed' and not state['complete'] and not state['coverage_verified']
    repo = Path(__file__).resolve().parents[2]
    for source, digest in state['sources'].items():
        assert sha(root / Path(source).name) == digest
        assert sha(repo / Path(source).relative_to('/workspace/vllm-plugin-FL')) == digest
    db = sqlite3.connect('file:' + str(root / 'trace') + '?mode=ro', uri=True)
    graphs = db.execute('select processId,ts,dur from cuda_graph_data order by ts').fetchall()
    kernels = db.execute('select k.processId,f.name,k.dur from cuda_kernel_data k '
                         'join function_name_map f on f.id=k.name_id').fetchall()
    assert len(graphs) == (20 if version == 1 else 25)
    assert len(kernels) == 20 and all(name == 'tianshu_stage_outside' and dur > 0 for _, name, dur in kernels)
    assert all(dur > 0 for _, _, dur in graphs)
    assert 'No running processes found' in (root / 'hardware_after.txt').read_text()
    result = dict(original_phase='failed', full_capability_passed=False,
                  graph_events=len(graphs), outside_kernel_events=len(kernels),
                  internal_graph_kernel_events=0, trace_sha256=sha(root / 'trace'))
    if version == 1:
        assert not (root / 'workload/result.json').exists()
        assert 'AssertionError' in (root / 'workload/child.log').read_text()
    else:
        child = read(root / 'workload/child/result.json')
        assert read(root / 'workload/result.json')['complete']
        assert child['complete'] and child['graph_output_exact']
        pids = {child['child_pid'], *child['namespace_pids']}
        assert {pid for pid, _, _ in graphs} <= pids and {pid for pid, _, _ in kernels} <= pids
        before = statistics.median(child['unprofiled_single_before_ms'])
        after = statistics.median(child['unprofiled_single_after_ms'])
        for name, ratio in (('before', child['unprofiled_bulk10_before_ms'] / before),
                            ('after', child['unprofiled_bulk10_after_ms'] / after)):
            assert ratio == child['unprofiled_bulk_to_single_' + name + '_ratio']
            assert 5 < ratio < 20
        event = statistics.median(child['unprofiled_single_before_ms'] + child['unprofiled_single_after_ms'])
        trace = statistics.median(dur / 1e6 for _, _, dur in graphs[5:])
        assert not .5 < event / trace < 1.5
        result.update(process_tree_counts_verified=True, output_exact=True,
                      unprofiled_10_to_1_before=child['unprofiled_bulk_to_single_before_ratio'],
                      unprofiled_10_to_1_after=child['unprofiled_bulk_to_single_after_ratio'],
                      unprofiled_event_ms=event, profiled_postwarm_trace_ms=trace,
                      timing_calibration_passed=False, profiled_to_unprofiled_ratio=trace / event)
    (root / 'local_download_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def model(root, cap_root):
    state = read(root / 'result.json')
    assert state['complete'] and state['phase'] == 'complete'
    assert not state['official_benchmark'] and not state['adopted'] and not state['ixsys_used']
    assert state['unprofiled_event_scaling_precondition_verified'] and not state['ixsys_timing_calibration_passed']
    assert state['precondition_sha256'] == sha(cap_root / 'workload/child/result.json')
    runtime = audit_runtime(root, state, official=True, online=False)
    repo = Path(__file__).resolve().parents[2]
    assert len(state['frozen_sources']) >= 49
    for source, digest in state['frozen_sources'].items():
        if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
            assert sha(repo / Path(source).relative_to('/workspace/vllm-plugin-FL')) == digest
    template = read(repo / 'experiments/tianshu_original_command_20261016/profile_environment_template.json')
    original = {k: v.replace(template['old_output_root'] + '/', state['output_root'] + '/')
                for k, v in template['environment'].items()}
    environment = state['profile_environment']
    for key, value in original.items():
        if key != 'PYTHONPATH':
            assert environment[key] == value
    assert set(environment) - set(original) == {
        'TIANSHU_STAGE_PROFILE_PROBE', 'TIANSHU_STAGE_PROFILE_MODE',
        'TIANSHU_STAGE_PROFILE_RECORD_DIR', 'TIANSHU_STAGE_PROFILE_ARM_FILE'}
    assert environment['TIANSHU_STAGE_PROFILE_PROBE'] == '1' and environment['TIANSHU_STAGE_PROFILE_MODE'] == 'events'
    assert len(state['workloads']) == 2
    for index, load in enumerate(state['workloads']):
        assert load['label'] == ('warmup', 'measured')[index]
        assert len(load['responses']) == load['concurrency'] == 64
        assert [r['input_seed'] for r in load['responses']] == [20261016000 + index * 1000 + i for i in range(64)]
        for response in load['responses']:
            assert response['api_seed'] is None
            assert (response['input_tokens'], response['output_tokens']) == (4096, 512)
            assert response['usage']['prompt_tokens'] == 4096 and response['usage']['completion_tokens'] == 512
            assert response['usage']['total_tokens'] == 4608 and response['text']
            assert 0 < response['ttft_seconds'] <= response['wall_seconds'] <= load['duration_seconds'] + .1
    for label in ('before', 'after'):
        c = state['canary_' + label]
        assert c['api_seed'] == 777 and c['input_seed'] == 2026101688
        assert c['usage']['prompt_tokens'] == 128 and c['usage']['completion_tokens'] == 16 and c['usage']['total_tokens'] == 144
    assert state['canary_before']['text'] == state['canary_after']['text']
    files = list((root / 'stage_records').glob('stages_*.json'))
    assert len(files) == 1
    record = read(files[0]); frames = record['frames']
    assert record['complete'] and record['ixsys_not_used'] and not record['events_inside_graph']
    assert record['boundary_synchronizations'] == 2 and len(frames) == record['frame_count'] == 40
    hook = '/workspace/vllm-plugin-FL/experiments/tianshu_graph_stage_profile_20261016/hooks/sitecustomize.py'
    runner = '/usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py'
    assert record['hook_source_sha256'] == state['frozen_sources'][hook]
    assert record['runner_source_sha256'] == state['frozen_sources'][runner]
    for source, metadata in record['extra_sources'].items():
        assert sha(root / 'stage_records' / metadata['snapshot']) == state['frozen_sources'][source] == metadata['sha256']
    for index, frame in enumerate(frames):
        assert frame['index'] == index and frame['dispatch'] == dict(mode='FULL', padded_tokens=64)
        c = frame['context']
        assert c['requests'] == 64 and c['min_prompt'] == c['max_prompt'] == 4096
        assert c['max_computed'] >= c['min_computed'] >= 4096
        if index:
            for key in ('min_computed', 'max_computed'):
                assert c[key] == frames[index-1]['context'][key] + 1
        g = frame['gpu_stream_intervals_ms']
        assert set(g) == {'total', 'forward', 'logits', 'sample'} and all(v > 0 for v in g.values())
        assert g['total'] + .1 >= g['forward'] + g['logits'] + g['sample']
        host = frame['cpu_call_spans']
        assert set(host) == {'execute', 'sample_tokens', 'forward', 'logits', 'sample'}
        assert len({v['thread_id'] for v in host.values()}) == 1
        for span in host.values():
            assert 0 <= span['thread_cpu_ms'] <= span['wall_ms'] + .1
    summary = read(root / 'summary.json'); assert summary == state['summary']
    sums = {key: sum(f['gpu_stream_intervals_ms'][key] for f in frames) for key in ('total', 'forward', 'logits', 'sample')}
    for key, value in sums.items():
        assert value == summary['gpu_stream_interval_sum_ms'][key]
        assert statistics.median(f['gpu_stream_intervals_ms'][key] for f in frames) == summary['gpu_stream_interval_median_ms'][key]
        if key != 'total':
            assert value / sums['total'] == summary['fractions_of_total_stream_interval'][key]
    assert sums['total'] - sums['forward'] - sums['logits'] - sums['sample'] == summary['remaining_stream_interval_ms']
    for key, values in summary['host_call_median'].items():
        for name, value in values.items():
            assert statistics.median(f['cpu_call_spans'][key][name] for f in frames) == value
    result = dict(complete=True, all128_load_responses_and_two_seeded_canaries_verified=True,
                  frozen_sources=len(state['frozen_sources']), runtime=runtime,
                  all40_FULL_C64_context_and_stage_frames_verified=True,
                  current_repo_sources_match=True, ixsys_used=False, formal_score=False,
                  summary=summary)
    (root / 'local_download_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cap-v1', type=Path, required=True)
    parser.add_argument('--cap-v2', type=Path, required=True)
    parser.add_argument('--model-root', type=Path)
    args = parser.parse_args()
    results = dict(v1=capability(args.cap_v1.resolve(), 1), v2=capability(args.cap_v2.resolve(), 2))
    if args.model_root:
        results['model'] = model(args.model_root.resolve(), args.cap_v2.resolve())
    print(json.dumps(results))


if __name__ == '__main__':
    main()
