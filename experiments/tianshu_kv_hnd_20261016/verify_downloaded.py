# SPDX-License-Identifier: Apache-2.0
"""Audit controlled-layout GPU records, event medians and frozen sources."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tianshu_original_command_20261016'))
from validation_profile import CV_SHA


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve();state=read(root/'result.json')
    assert state['phase']=='complete' and state['complete'] and not state['official_benchmark'] and not state['adoption_allowed']
    assert state['actual_builder_and_KV_writer_checked'] and state['changed_Graph_replays_verified'] and state['numerical_checks_passed']
    repo=Path(__file__).resolve().parents[2]
    for source,h in state['frozen_sources'].items():
        assert sha(root/'source_snapshot'/Path(source).relative_to('/'))==h
        if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
            assert sha(repo/Path(source).relative_to('/workspace/vllm-plugin-FL'))==h
    best=read(repo.parent/'service-results/20261016_tianshu/gemm_topp_geo_official/20261016_gemm_topp_geo_official_tianshu/result.json')
    assert state['server_command']==best['server_command']
    profile=state['profile_environment'];assert len(profile)==38
    remote_root=Path(profile['TIANSHU_NATIVE_PREFILL_RECORD_DIR']).parent
    assert remote_root.name==root.name
    normalize=lambda p,r:{k:v.replace(r+'/', '<RUN>/') for k,v in p.items()}
    assert normalize(profile,str(remote_root))==normalize(best['profile_environment'],best['output_root'])
    names={'prefill32','prefill33','prefill2048','chunk2048','ragged','mixed64','mixed_ragged','decode64','decode64_5k','decode64_65','decode1_33','decode8','decode16','decode24','decode32','unreordered'}
    cases=state['cases'];assert len(cases)==16 and {c['name'] for c in cases}==names
    measurements=0;timings={}
    for c in cases:
        assert c['query_and_KV_and_slots_and_lengths_changed_in_place']
        assert len(c['phases'])==2
        for p in c['phases']:
            assert p['KV_writer_all_storage_bitwise_exact'] and p['output_padding_preserved'] and p['segment_scratch_tail_preserved']
            assert p['layouts_bitwise_equal']
            for check in [*p['reference_checks'],*p['cpu_checks'],p['NHD_vs_HND']]:
                assert check['passed'] and check['failed_elements']==0 and check['atol']==.02 and check['rtol']==.01
        layouts=c['cache_layouts'];assert [v['layout'] for v in layouts]==['NHD','HND']
        block=16384 if c['strided'] else 8192
        assert layouts[0]['strides']==[block,4096,256,128,1]
        assert layouts[1]['strides']==[block,4096,128,2048,1]
        raw=c['raw_ms'];assert len(raw)==2 and all(len(v)==7 for v in raw)
        assert all(v>0 for values in raw for v in values)
        medians=list(map(statistics.median,raw));measurements+=14
        assert medians==c['median_ms']
        delta=100*(medians[1]/medians[0]-1);assert abs(delta-c['time_change_percent'])<1e-12
        timings[c['name']]=dict(NHD_ms=medians[0],HND_ms=medians[1],time_change_percent=delta)
        calls=c['actual_native_calls'];assert calls and all(v['page_size']==16 for v in calls)
        if not c['name'].startswith('decode') and c['name']!='unreordered':
            assert any(v['max_query']>1 and v['kernel_router_source'].endswith('tianshu_native_prefill_20261012/hooks/sitecustomize.py') for v in calls)
    def artifact(path):return root/Path(path).relative_to(remote_root)
    directory=artifact(profile['TIANSHU_DECODE_COHERENT_RECORD_DIR'])
    producers=[read(p) for p in directory.glob('producer_*.json')]
    assert any(p['batch']==64 and p['segments']==8 for p in producers)
    for p in producers:
        assert p['cache_service_mode']=='candidate' and p['actual_cache_modifier']=='.cv'
        assert p['generated_source_sha256']==CV_SHA
        assert sha(artifact(p['zero_initialization']['generated_source']))==CV_SHA
        if p['batch']==64:
            assert p['configuration']==dict(TILE_SIZE=16,num_warps=1,num_stages=1)
            assert p['compiled_metadata']['num_warps']==1
    report=dict(complete=True,all_frozen_source_snapshots_and_repo_sources_verified=len(state['frozen_sources']),
        all16_case_check_records_and_strides_verified=True,all32_changed_Graph_phase_records_verified=True,
        full_GPU_tensor_values_not_serialized=True,original_writer_and_attention_used=True,
        actual_CV_KV16_w1_S8_and_native_prefill_verified=True,current_best38_environment_replayed=True,
        raw_event_measurements_recomputed=measurements,timings=timings,
        official_benchmark=False,model_throughput_and_Math_not_measured=True,adopted=False)
    (root/'local_download_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
