# SPDX-License-Identifier: Apache-2.0
"""Recompute scalar-page micro results, generated source and actual artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from scalar_page import render_source


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve(); state = read(root/'result.json')
    assert state['complete'] and state['phase'] == 'complete' and state['child_exit_code'] == 0
    assert not state['official_benchmark'] and not state['adopted'] and len(state['frozen_sources']) == 10
    assert 'No running processes found' in (root/'hardware_after.txt').read_text()
    repo = Path(__file__).resolve().parents[2]
    for path, digest in state['frozen_sources'].items():
        assert sha(root/'source_snapshot'/Path(path).relative_to('/')) == digest
        if Path(path).is_relative_to('/workspace/vllm-plugin-FL'):
            assert sha(repo/Path(path).relative_to('/workspace/vllm-plugin-FL')) == digest
    micro = root/'micro'; data = read(micro/'result.json'); snap = micro/'source_snapshot'
    assert data['complete'] and data['numerical_checks_passed'] and not data['official_benchmark']
    assert not data.get('compile_errors') and not data.get('changed_input_failures')
    assert data['valid_scalar_page_modes'] == [False, True] and len(data['cases']) == 17
    assert sha(snap/'probe_page.py') == data['probe_sha256'] == sha(Path(__file__).with_name('probe_page.py'))
    for source in ('scalar_page.py','launch_router.py','zero_initialization.py','experimental_cache.py','triton_unified_attention.py'):
        candidates = [path for path in state['frozen_sources'] if Path(path).name == source]
        assert len(candidates) == 1
        assert sha(snap/source) == state['frozen_sources'][candidates[0]]
    for name, digest in data['backend_source_sha256'].items(): assert sha(snap/name) == digest
    cv = micro/'cache_policy_source/generated_cache_policy.py'
    assert sha(cv) == '8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'
    generated = micro/'scalar_source/generated_scalar_page.py'
    assert generated.read_text() == render_source(cv.read_bytes())
    gen = data['provenance']['scalar_page_generation']
    assert sha(generated) == gen['generated_sha256']
    assert gen['integer_widths_unchanged'] and gen['floating_arithmetic_and_reducer_source_unchanged']
    assert gen['guarded_by_IS_3D_and_equal_page_tile'] and not gen['installed_sources_modified']
    expected = [(b,4096) for b in (8,16,24,32,64)] + [(64,5120),(64,65)]
    expected += [(8,n) for n in (1,15,16,17,31,32,33,63,64,65)]
    rows = []; all_bitwise = True
    for case, (batch, length) in zip(data['cases'], expected):
        assert case['batch'] == batch and max(case['kv_lengths']) == length
        assert case['logical_KV_identical'] and case['query_row_tiles'] == [16]*3
        assert len(case['variants']) == 2 and len(case['changed_graph_bitwise_equal']) == 3
        all_bitwise &= all(case['changed_graph_bitwise_equal'])
        for name in ('cpu_reference_checks','changed_graph_cpu_checks','changed_graph_fp32_checks'):
            assert case[name] and all(c['passed'] and c['failed_elements'] == 0 and c['atol'] == .02 and c['rtol'] == .01 for c in case[name])
        assert case['original_vs_fp32']['passed'] and case['baseline_tail_preserved']
        tile = 32 if batch == 24 else 16; warps = 2 if batch == 24 else 1
        segments = 8 if batch == 64 else 4 if batch >= 16 else 16
        configuration = dict(TILE_SIZE=tile,num_warps=warps,num_stages=1,KV_CACHE_MODIFIER='.cv')
        assert case['baseline_configuration'] == configuration and case['segments'] == segments
        controls = [(None,case['baseline_launch_metadata'],case['baseline_raw_ms'],case['baseline_ms'])]
        for requested, variant in zip((False,True), case['variants']):
            assert variant['configuration'] == dict(configuration,SCALAR_KV_PAGE=requested)
            assert variant['tails_preserved']
            for name in ('candidate_vs_fp32','candidate_vs_native'):
                assert variant[name]['passed'] and variant[name]['failed_elements'] == 0
            all_bitwise &= variant['candidate_bitwise_equal']
            change = 100*(variant['candidate_ms']/case['baseline_ms']-1)
            assert abs(change-variant['time_change_percent']) < 1e-10
            controls.append((requested,variant['launch_metadata'],variant['raw_ms'],variant['candidate_ms']))
            rows.append(dict(batch=batch,kv_max=length,scalar_page=requested,baseline_ms=case['baseline_ms'],
                             candidate_ms=variant['candidate_ms'],time_change_percent=change,
                             bitwise_equal=variant['candidate_bitwise_equal']))
        for requested, metadata, raw, median in controls:
            assert metadata['num_warps'] == warps and metadata['num_stages'] == 1
            assert metadata['actual_BLOCK_M'] == 16 and metadata['actual_BLOCK_Q'] == 2 and metadata['actual_BLOCK_SIZE'] == 16
            assert metadata['producer_and_reducer_tile'] == tile and metadata['requested_cache_modifier'] == '.cv'
            assert metadata['requested_scalar_page'] is requested
            assert metadata['scalar_page_effective'] == bool(requested and tile == 16)
            assert metadata['actual_grid'] == [sum(case['query_lengths'])//2+batch,2,segments]
            assert len(raw) == 9 and all(t > 0 for t in raw) and statistics.median(raw) == median
    artifacts = list((micro/'kernel_artifacts').glob('*/manifest.json'))
    assert {p.parent.name for p in artifacts} == {'baseline_cv','False','True'}
    for p in artifacts:
        record = read(p); assert record['files'] and record['requested_cache_modifier'] == '.cv'
        expected_flag = None if p.parent.name == 'baseline_cv' else p.parent.name == 'True'
        assert record['register_info']['requested_scalar_page'] is expected_flag
        for name, digest in record['files'].items(): assert sha(p.parent/name) == digest
    result = dict(complete=True,official_benchmark=False,adopted=False,scenarios=17,candidate_cases=34,
                  recomputed_events=459,all_original_and_changed_KV_bitwise_equal=all_bitwise,
                  all10_source_snapshots_and_current_repo_files_verified=True,
                  generated_source_exact_and_guarded_page_load_verified=True,
                  actual_launch_parameters_and_artifacts_verified=True,rows=rows)
    (root/'local_download_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
