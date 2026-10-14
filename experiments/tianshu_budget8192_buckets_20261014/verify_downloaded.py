# SPDX-License-Identifier: Apache-2.0
"""Audit budget4096/8192 with retained75 and extended99 bucket points."""
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
    assert proof['excluded_rounds'] == [1] and not proof['official_benchmark']
    assert not proof['adopted'] and not proof['control_reused']
    assert state['frozen_sources'] == proof['frozen_sources']
    assert len(proof['frozen_sources']) == 11
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
        assert trial['active_hook']['sha256'] == proof['frozen_sources'][trial['active_hook']['path']]
        budget=4096 if mode=='control' else 8192
        points=83 if mode=='control' else 99
        assert trial['actual_budget']==budget and trial['actual_threshold']==0
        assert trial['actual_bucket_points']==points
        assert trial['request_sampling_config'] == dict(temperature=0.,top_p=1.,repetition_penalty=1.,logprobs=None,ignore_eos=True,stream=True)
        for name, meta in trial['installed_sources'].items():
            assert sha(root/mode/'source_snapshot'/name) == meta['sha256']
        rounds = trial['short_4k_rounds']
        assert len(rounds) == 3
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
        assert abs(proof[mode+'_total_tok_s'] - statistics.mean(values[1:])) < 1e-9
        records = root/mode/'decode_coherent_records'
        producers = [json.loads(x.read_text()) for x in records.glob('producer_*.json')]
        reducers = [json.loads(x.read_text()) for x in records.glob('reducer_*.json')]
        assert producers == trial['decode_coherent_records'] or sorted(producers,key=lambda x:(x['batch'],x['segments'])) == sorted(trial['decode_coherent_records'],key=lambda x:(x['batch'],x['segments']))
        assert producers and reducers
        tiles = {}
        for record in producers:
            selected = record['batch'] in (8,16,32,40,48,56,64)
            tile = 16 if selected else 32
            warps = 1 if selected else 2 if 8 <= record['batch'] <= 64 else 4
            assert record['configuration'] == dict(TILE_SIZE=tile,num_warps=warps,num_stages=1)
            assert record['compiled_metadata']['num_warps'] == warps
            assert record['generated_source_sha256'] == '4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e'
            assert record['grid_unchanged'] and record['integer_widths_and_pointer_types_unchanged']
            expected_segments = 8 if record['batch'] == 64 else 64 if record['batch'] == 1 else 4 if record['batch'] >= 16 else 16
            assert record['segments'] == expected_segments
            tiles[record['batch'],record['segments']] = tile
            generated = Path(record['zero_initialization']['generated_source'])
            remote_trial=Path(trial['actual_composed_environment']['TIANSHU_DECODE_COHERENT_RECORD_DIR']).parent
            relative = generated.relative_to(remote_trial)
            assert sha(root/mode/relative) == record['generated_source_sha256']
        assert {(x['batch'],x['segments']) for x in reducers} == set(tiles)
        assert all(x['actual_tile'] == tiles[x['batch'],x['segments']] for x in reducers)
        assert any(x['batch'] == 64 and x['configuration']['TILE_SIZE'] == 16 for x in producers)
        assert any(x['batch'] == 24 and x['configuration'] == dict(TILE_SIZE=32,num_warps=2,num_stages=1) for x in producers)
        bucket = root/mode/'bucket_trace_records'
        configured = [json.loads(x.read_text()) for x in bucket.glob('configured_*.json')]
        warmed = [json.loads(x.read_text()) for x in bucket.glob('warmup_completed_*.json')]
        assert any(x['enabled'] and len(x['points'])==points and max(x['points'])==budget for x in configured)
        assert any(x['complete'] and len(x['calls'])==points and max(c['size'] for c in x['calls'])==budget for x in warmed)
        policy = trial['segment_policy_records']
        policy_sha=proof['frozen_sources']['/workspace/vllm-plugin-FL/experiments/tianshu_kv16_segments_20261014/hooks/sitecustomize.py']
        assert policy and all(x['policy_hook_sha256']==policy_sha for x in policy)
        assert any(x['batch']==64 and x['baseline_segments']==4 and x['selected_segments']==8 for x in policy)
        assert all(x['selected_segments']==(8 if x['batch']==64 else x['baseline_segments']) for x in policy)
        enabled=mode=='candidate'
        assert trial['sql_reflection_enabled'] and trial['actual_sql_mode_verified']
        assert trial['actual_extended_policy_verified'] and trial['shared_SQL_workflow']
        assert trial['model_GEMM_kernel_unchanged']
        sql=[json.loads(x.read_text()) for x in (root/mode/'sql_reflection_records').glob('route_*.json')]
        if True:
            assert sql and all(x['reflection_scope']=='one_requested_table' and not x['configuration_values_modified'] for x in sql)
            assert all(x['helper_sha256']=='4cfc6a34e29e0cc220b83d79898c1da4d986e35d4eb0796c37947b1631064c02' and x['original_sql_sha256']=='6dc19cb581d70154f14b95ed9112d682c6b5919b6cf3f8dea01a78a75ec0efd7' for x in sql)
        extensions=[json.loads(x.read_text()) for x in (root/mode/'extended_bucket_records').glob('extended_*.json')]
        assert extensions and trial['actual_extended_bucket_records']
        helper_sha=proof['frozen_sources']['/workspace/vllm-plugin-FL/experiments/tianshu_budget8192_buckets_20261014/extend_buckets.py']
        for record in extensions:
            assert record['helper_sha256']==helper_sha and record['original_bucket_source_sha256']=='966aad18aad80ef488928cafc8672cbfa2d547a832f9b151346f719ea3abc6db'
            assert len(record['original_points'])==75 and len(record['extended_points'])==99
            assert record['extended_points'][:75]==record['original_points']
            assert record['extended_points'][75:]==list(range(2304,8193,256))
            assert record['changed_constants']==[[2048,8192]] and record['hidden_size_guard_preserved']
            assert record['padded_tokens_function_unmodified'] and record['configure_and_warmup_functions_unmodified']
            assert not record['installed_sources_modified']
        assert any(x['points']==[n for n in extensions[0]['extended_points'] if n<=budget] for x in configured)
        assert any([c['size'] for c in x['calls']]==list(reversed([n for n in extensions[0]['extended_points'] if n<=budget])) for x in warmed)
        gather = [json.loads(x.read_text()) for x in bucket.glob('gather_*.json')]
        assert any(x['index_dtype'] == 'torch.int32' for x in gather)
        assert 'No running processes found' in (root/mode/'hardware_after.txt').read_text()
    control,candidate = trials
    for key in ('installed_sources','workload','request_sampling_config'):
        assert control[key] == candidate[key],key
    aa,bb=map(list,(trials[0]['command'],trials[1]['command']))
    for command in (aa,bb):command[command.index('--max-num-batched-tokens')+1]='BUDGET'
    assert aa==bb
    cold=[t['short_4k_rounds'][0]['duration_seconds'] for t in trials]
    assert cold==proof['first_round_seconds']
    assert abs(100*(cold[1]/cold[0]-1)-proof['cold_first_round_change_percent'])<1e-9
    assert [t['startup_seconds'] for t in trials]==proof['startup_seconds']
    equal = 0
    for aa,bb in zip(control['short_4k_rounds'],candidate['short_4k_rounds']):
        for a,b in zip(aa['responses'],bb['responses']):
            assert a['seed'] == b['seed']
            equal += a['text'] == b['text']
    canaries = sum(a['text'] == b['text'] for a,b in zip(control['rare_cases'],candidate['rare_cases']))
    assert canaries == proof['greedy_cases_equal']
    gain = 100*(proof['candidate_total_tok_s']/proof['control_total_tok_s']-1)
    assert abs(gain - proof['gain_percent']) < 1e-9
    audit = dict(complete=True,official_benchmark=False,adopted=False,successful_requests=384,source11_verified=True,installed_snapshot_count=12,
        all_usage_seed_and_text_checked=True,paired_texts_equal=equal,paired_total=192,greedy_canaries_equal=canaries,
        actual_model_B64S8_and_83_99_buckets_and_SQL_verified=True,matching_reducer_tile_verified=True,warm_points=[83,99],
        excluded_rounds=[1],control_total_tok_s=proof['control_total_tok_s'],candidate_total_tok_s=proof['candidate_total_tok_s'],gain_percent=gain,
        cold_first_round_seconds=cold,cold_first_round_change_percent=proof['cold_first_round_change_percent'],candidate_all_scored_rounds_faster_than_control=min(proof['candidate_rounds'][1:])>max(proof['control_rounds'][1:]),rows=rows,new_math_run=False)
    (root/'local_download_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
