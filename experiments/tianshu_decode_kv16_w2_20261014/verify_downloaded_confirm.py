# SPDX-License-Identifier: Apache-2.0
"""Independently audit downloaded KV16/w1 five-round request and source evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[2]
    state = json.loads((root/'result.json').read_text())
    proof = json.loads((root/'comparison.json').read_text())
    assert state['phase'] == 'complete' and proof['complete']
    assert proof['excluded_rounds'] == [1,2] and not proof['official_benchmark']
    assert not proof['adopted'] and not proof['control_reused']
    assert state['frozen_sources'] == proof['frozen_sources']
    assert len(proof['frozen_sources']) == 7
    for source, digest in proof['frozen_sources'].items():
        rel = Path(source).relative_to('/workspace/vllm-plugin-FL')
        assert sha(repo/rel) == digest
        assert sha(root/'source_snapshot'/rel.relative_to('experiments')) == digest
    trials = []
    rows = []
    for mode in ('control','candidate'):
        trial = json.loads((root/mode/'result.json').read_text())
        trials.append(trial)
        assert trial['phase'] == 'complete' and trial['lowwarps_mode'] == mode
        assert trial['actual_budget'] == 2048 and trial['actual_threshold'] == 0
        assert trial['actual_bucket_points'] == 75
        assert trial['request_sampling_config'] == dict(temperature=0.,top_p=1.,repetition_penalty=1.,logprobs=None,ignore_eos=True,stream=True)
        for name, meta in trial['installed_sources'].items():
            assert sha(root/mode/'source_snapshot'/name) == meta['sha256']
        rounds = trial['short_4k_rounds']
        assert len(rounds) == 5
        for i,row in enumerate(rounds,1):
            assert row['round'] == i and row['successful_requests'] == 64
            responses = row['responses']
            assert len(responses) == 64 and len({x['seed'] for x in responses}) == 64
            for response in responses:
                assert response['usage']['prompt_tokens'] == response['input_tokens'] == 4096
                assert response['usage']['completion_tokens'] == response['output_tokens'] == 1024
                assert response['usage']['total_tokens'] == 5120
                assert isinstance(response['text'],str) and response['text']
                assert 0 < response['ttft_seconds'] <= response['wall_seconds'] <= row['duration_seconds']
            assert abs(row['total_tokens_per_second'] - 64*5120/row['duration_seconds']) < 1e-8
            assert abs(row['output_tokens_per_second'] - 64*1024/row['duration_seconds']) < 1e-8
            assert abs(row['mean_ttft_seconds'] - statistics.mean(x['ttft_seconds'] for x in responses)) < 1e-9
            rows.append(dict(mode=mode,round=i,total_tok_s=row['total_tokens_per_second'],output_tok_s=row['output_tokens_per_second'],ttft_ms=row['mean_ttft_seconds']*1000,duration_s=row['duration_seconds']))
        values = [x['total_tokens_per_second'] for x in rounds]
        assert values == proof[mode+'_rounds']
        assert trial['confirm_scoring']['excluded_rounds'] == [1,2]
        assert abs(trial['confirm_scoring']['total_tok_s'] - statistics.mean(values[2:])) < 1e-9
        assert abs(proof[mode+'_total_tok_s'] - statistics.mean(values[2:])) < 1e-9
        records = root/mode/'decode_coherent_records'
        producers = [json.loads(x.read_text()) for x in records.glob('producer_*.json')]
        reducers = [json.loads(x.read_text()) for x in records.glob('reducer_*.json')]
        assert producers == trial['decode_coherent_records'] or sorted(producers,key=lambda x:(x['batch'],x['segments'])) == sorted(trial['decode_coherent_records'],key=lambda x:(x['batch'],x['segments']))
        assert producers and reducers
        tiles = {}
        for record in producers:
            selected = mode == 'candidate' and record['batch'] in (8,16,32,40,48,56,64)
            tile = 16 if selected else 32
            warps = 1 if selected else 2 if 8 <= record['batch'] <= 64 else 4
            assert record['configuration'] == dict(TILE_SIZE=tile,num_warps=warps,num_stages=1)
            assert record['compiled_metadata']['num_warps'] == warps
            assert record['generated_source_sha256'] == '4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e'
            assert record['grid_unchanged'] and record['integer_widths_and_pointer_types_unchanged']
            tiles[record['batch'],record['segments']] = tile
            generated = Path(record['zero_initialization']['generated_source'])
            relative = generated.relative_to(Path('/workspace/benchmark-results')/root.name)
            assert sha(root/relative) == record['generated_source_sha256']
        assert {(x['batch'],x['segments']) for x in reducers} == set(tiles)
        assert all(x['actual_tile'] == tiles[x['batch'],x['segments']] for x in reducers)
        assert any(x['batch'] == 64 and x['configuration']['TILE_SIZE'] == (16 if mode == 'candidate' else 32) for x in producers)
        assert any(x['batch'] == 24 and x['configuration'] == dict(TILE_SIZE=32,num_warps=2,num_stages=1) for x in producers)
        bucket = root/mode/'bucket_trace_records'
        configured = [json.loads(x.read_text()) for x in bucket.glob('configured_*.json')]
        warmed = [json.loads(x.read_text()) for x in bucket.glob('warmup_completed_*.json')]
        assert any(x['enabled'] and len(x['points']) == 75 and max(x['points']) == 2048 for x in configured)
        assert any(x['complete'] and len(x['calls']) == 75 for x in warmed)
        assert 'No running processes found' in (root/mode/'hardware_after.txt').read_text()
    control,candidate = trials
    for key in ('command','installed_sources','workload','request_sampling_config'):
        assert control[key] == candidate[key],key
    equal = 0
    for aa,bb in zip(control['short_4k_rounds'],candidate['short_4k_rounds']):
        for a,b in zip(aa['responses'],bb['responses']):
            assert a['seed'] == b['seed']
            equal += a['text'] == b['text']
    canaries = sum(a['text'] == b['text'] for a,b in zip(control['rare_cases'],candidate['rare_cases']))
    assert canaries == proof['greedy_cases_equal']
    gain = 100*(proof['candidate_total_tok_s']/proof['control_total_tok_s']-1)
    assert abs(gain - proof['gain_percent']) < 1e-9
    audit = dict(complete=True,official_benchmark=False,adopted=False,successful_requests=640,source7_verified=True,installed_snapshot_count=12,
        all_usage_seed_and_text_checked=True,paired_texts_equal=equal,paired_total=320,greedy_canaries_equal=canaries,
        actual_model_compiled_B64_KV16_w1_and_B24_fallback_verified=True,matching_reducer_tile_verified=True,warm_points=[75,75],
        excluded_rounds=[1,2],control_total_tok_s=proof['control_total_tok_s'],candidate_total_tok_s=proof['candidate_total_tok_s'],gain_percent=gain,
        candidate_all_scored_rounds_faster_than_control=min(proof['candidate_rounds'][2:])>max(proof['control_rounds'][2:]),rows=rows,new_math_run=False)
    (root/'local_download_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
