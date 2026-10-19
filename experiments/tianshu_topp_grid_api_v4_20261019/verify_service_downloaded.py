# SPDX-License-Identifier: Apache-2.0
"""Recompute the TopP grid screen and all qualified runtime evidence."""
import argparse
import json
from pathlib import Path
import re
import statistics
from audit_profile import audit_runtime


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    comparison=read(root/'comparison.json');assert comparison['complete']
    repo=Path(__file__).resolve().parents[2]
    import hashlib
    modes={}
    for mode in ('control','candidate'):
        run=root/mode;state=read(run/'result.json');assert state['phase']=='complete' and state['mode']==mode
        gate=read(repo/'experiments/tianshu_topp_grid_api_v4_20261019/GPU_GATE_32g.json')
        expected=dict(gate['profile_frozen_sources'])
        gatepath='/workspace/vllm-plugin-FL/experiments/tianshu_topp_grid_api_v4_20261019/GPU_GATE_32g.json'
        expected[gatepath]=hashlib.sha256((repo/Path(gatepath).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()
        assert state['frozen_sources']==expected
        profile=audit_runtime(run,state,official=True,online=False)
        assert profile['original_sampler_and_RNG']
        assert profile['strict_GEO_arithmetic'] and profile['M2048_three_projection_tiles']
        assert profile['TopP_16K_grid2_API']==(mode=='candidate')
        route_files=list((run/'grid_topp_records').glob('actual_*.json'))
        routes=[r for p in route_files for r in read(p)['routes'].values()]
        b64=[r for r in routes if r['batch']==64 and r['topp'] and not r['topk']]
        assert b64
        if mode=='candidate':
            assert all(r['active'] and r['actual_candidate']['grid']==[32] and r['actual_candidate']['actual_buffer_rows']==64 for r in b64)
        else:
            base_routes=[read(p) for p in (run/'topp_launch_records').glob('route_*.json')]
            assert any(r['batch']==64 and r['topp'] and not r['topk'] and r['actual_BLOCK_SIZE']==32768 for r in base_routes)

        for source,sha in state['frozen_sources'].items():
            if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
                assert hashlib.sha256((repo/Path(source).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha
        rows=state['short_4k_rounds'];assert len(rows)==3
        all_seeds=[]
        for i,row in enumerate(rows):
            responses=row['responses'];assert row['round']==i+1 and row['successful_requests']==len(responses)==64
            expected=[202610193000+100*i+j for j in range(64)]
            assert [r['input_seed'] for r in responses]==expected
            all_seeds.extend(expected)
            for r in responses:
                assert r['api_seed'] is None
                assert (r['input_tokens'],r['output_tokens'])==(4096,1024)
                assert r['usage']['prompt_tokens']==4096 and r['usage']['completion_tokens']==1024 and r['usage']['total_tokens']==5120
                assert isinstance(r['text'],str) and r['text']
                assert 0<r['ttft_seconds']<=r['wall_seconds']<=row['duration_seconds']+.1
            assert abs(row['total_tokens_per_second']-64*5120/row['duration_seconds'])<1e-9
            assert abs(row['output_tokens_per_second']-64*1024/row['duration_seconds'])<1e-9
            assert abs(row['mean_ttft_seconds']-statistics.mean(r['ttft_seconds'] for r in responses))<1e-12
        assert len(set(all_seeds))==192
        assert abs(state['mean_total_tok_s']-statistics.mean(r['total_tokens_per_second'] for r in rows[1:]))<1e-9
        canaries=state['seeded_canaries'];assert len(canaries)==2
        assert canaries[0]['text']==canaries[1]['text']
        for c in canaries:assert c['api_seed']==777 and c['input_seed']==2026101699 and c['usage']['prompt_tokens']==128 and c['usage']['completion_tokens']==16
        metrics=list(run.glob('*metrics.txt'));assert len(metrics)==8
        for p in metrics:
            vals=re.findall(r'^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$',p.read_text(),re.M)
            assert vals and all(float(v)==0 for v in vals)
        modes[mode]=state
    control,candidate=modes['control'],modes['candidate']
    assert control['frozen_sources']==candidate['frozen_sources']==comparison['frozen_sources']
    assert control['server_command']==candidate['server_command']
    for key,value in control['profile_environment'].items():
        other=candidate['profile_environment'][key]
        if key=='TIANSHU_TOPP_GRID_CANDIDATE':assert (value,other)==('0','1')
        else:assert value.replace(control['output_root'],'<RUN>')==other.replace(candidate['output_root'],'<RUN>'),key
    assert control['seeded_canaries'][0]['text']==candidate['seeded_canaries'][0]['text']
    before,after=control['mean_total_tok_s'],candidate['mean_total_tok_s']
    assert comparison['control_total_tok_s']==before and comparison['candidate_total_tok_s']==after
    assert comparison['gain_percent']==100*(after/before-1)
    report=dict(complete=True,all384_generation_texts_usage_seeds_and_timing_checked=True,
        all_source_snapshots_and_current_repo_sources_checked=True,actual_original2048_256seq75warm_CV_KV16_S8_TopP_modes_verified=True,
        actual40_environment_one_grid_switch_only_verified=True,actual_candidate_B64_strictTopP16K_grid2_and_three_M2048_configs_verified=True,
        all4_seeded_canaries_equal=True,all16_metrics_zero_preemptions=True,
        control_total_tok_s=before,candidate_total_tok_s=after,gain_percent=comparison['gain_percent'],
        official_benchmark=False,fresh_Math_not_run=True,RNG_unchanged=True,grid_API_and_actual_compiled_route_verified=True)
    (root/'local_download_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
