# SPDX-License-Identifier: Apache-2.0
"""Recompute all RNG sampler screen responses and runtime/source evidence offline."""
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
    parser.add_argument('--micro-root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();micro=read(args.micro_root/'result.json')
    assert micro['complete'] and len(micro['cases'])==105 and len(micro['rng_words'])==12
    assert all(c['cpu_reference_equal'] and c['production_same_noise_equal'] and c['materialized_matches_fused'] for c in micro['cases'])
    comparison=read(root/'comparison.json');assert comparison['complete']
    repo=Path(__file__).resolve().parents[2]
    import hashlib
    modes={}
    for mode in ('control','candidate'):
        run=root/mode;state=read(run/'result.json');assert state['phase']=='complete' and state['mode']==mode
        assert len(state['frozen_sources'])==52
        profile=audit_runtime(run,state,official=True,online=False)
        assert profile['original_sampler']==(mode=='control')
        for source,sha in state['frozen_sources'].items():
            if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
                assert hashlib.sha256((repo/Path(source).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha
        rows=state['short_4k_rounds'];assert len(rows)==3
        all_seeds=[]
        for i,row in enumerate(rows):
            responses=row['responses'];assert row['round']==i+1 and row['successful_requests']==len(responses)==64
            expected=[2026101600+100*i+j for j in range(64)]
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
        records=[read(p) for p in (run/'rng_sampler_records').glob('sampler_*.json')]
        if mode=='control':assert not records
        else:
            assert any(r['batch']==64 and r['route']=='fused' and r['seeded_rows']==0 for r in records)
            assert any(r['route']=='seeded_fallback' and r['seeded_rows']==1 for r in records)
            for r in records:
                assert r['top_k_top_p_unchanged'] and r['global_random_sequence_changed']
                assert r['kernel_sha256']==micro['sources']['/workspace/vllm-plugin-FL/experiments/tianshu_rng_sampling_20261016/experimental_rng_sampler_v2.py']
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
        if key=='TIANSHU_RNG_SAMPLER_PROBE':assert (value,other)==('0','1')
        else:assert value.replace(control['output_root'],'<RUN>')==other.replace(candidate['output_root'],'<RUN>'),key
    assert control['seeded_canaries'][0]['text']==candidate['seeded_canaries'][0]['text']
    before,after=control['mean_total_tok_s'],candidate['mean_total_tok_s']
    assert comparison['control_total_tok_s']==before and comparison['candidate_total_tok_s']==after
    assert comparison['gain_percent']==100*(after/before-1)
    report=dict(complete=True,all384_generation_texts_usage_seeds_and_timing_checked=True,
        all52_source_snapshots_and_current_repo_sources_checked=True,actual_original2048_256seq75warm_CV_KV16_S8_nativeTopP_verified=True,
        actual37_environment_RNG_only_difference_verified=True,actual_candidate_B64_and_seeded_fallback_verified=True,
        all4_seeded_canaries_equal=True,all16_metrics_zero_preemptions=True,
        control_total_tok_s=before,candidate_total_tok_s=after,gain_percent=comparison['gain_percent'],
        official_benchmark=False,fresh_Math_not_run=True,new_global_random_sequence=True)
    (root/'local_download_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
