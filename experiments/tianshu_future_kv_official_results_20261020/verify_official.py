# SPDX-License-Identifier: Apache-2.0
"""Independently verify both official workloads and the same fresh Math profile."""
import hashlib,json,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_future_kv_validation_20261020'))
import framework_profile as profile

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    root=Path(sys.argv[1]);math=Path(sys.argv[2]);state=read(root/'result.json');ms=read(math/'result.json')
    assert state['phase']=='complete' and ms['phase']=='complete'
    assert (root.parent/'20261020_future_kv_validation_pipeline_tianshu/complete.txt').is_file()
    quality=read(math/'accuracy_audit.json')
    mruntime=profile.audit_runtime(math,ms,official=False)
    assert profile.audit_math(math,ms,mruntime,write=False)==quality
    assert quality['correct']==103 and quality['total']==105 and quality['accuracy']>.95
    profile.frozen_check(state['frozen_sources']);profile.frozen_check(state['controller_sources'])
    assert state['frozen_sources']==ms['frozen_sources'] and state['controller_sources']==ms['controller_sources']
    assert len(state['frozen_sources'])==261 and len(state['controller_sources'])==7
    runtime=profile.audit_runtime(root,state,official=True)
    current=profile.audit_benchmark(root,state,runtime,quality,write=False)
    assert current==read(root/'comparison.json')
    assert current['complete'] and current['successful_requests']==1536 and current['failed_requests']==0
    assert runtime['zero_preemptions'] and runtime['future_reservation']['tracked_requests']==0
    assert runtime['future_reservation']['outstanding_blocks']==0
    prior_path=root.parent/'20261020_long_segments_official4k16k_tianshu/comparison.json'
    prior=read(prior_path);assert prior['complete'] and prior['successful_requests']==1536
    old={r['input_tokens']:r for r in prior['workloads']}
    assert old[4096]['total_tok_s']==5099.65 and old[16384]['total_tok_s']==3723.12
    rows=[]
    for item in current['workloads']:
        length=item['input_tokens'];baseline=old[length]
        assert item['passes_public_performance_gates']
        assert item['steady_spread_percent']<1
        rows.append(dict(input_tokens=length,total_tok_s=item['total_tok_s'],output_tok_s=item['output_tok_s'],
            mean_TTFT_ms=item['mean_TTFT_ms'],steady_spread_percent=item['steady_spread_percent'],
            previous_total_tok_s=baseline['total_tok_s'],vs_previous_percent=100*(item['total_tok_s']/baseline['total_tok_s']-1),
            public_baseline_total_tok_s=item['public_baseline_tok_s'],vs_public_percent=item['improvement_vs_public_percent'],
            successful_requests=item['successful_requests'],excluded_rounds=item['excluded_rounds']))
    values={r['input_tokens']:r for r in rows};assert set(values)=={4096,16384}
    adopted=values[4096]['vs_previous_percent']>=-1 and values[16384]['vs_previous_percent']>=2
    assert adopted,'Candidate did not meet the predeclared short-context non-regression and long-context improvement gate'
    trace=read(next((root/'graph_segments_records').glob('trace_*.json')))
    long_frames=[dict(batch=int(n),**frame) for graph in trace['graphs'].values() for n,frame in graph['frames'].items()
        if 33<=int(n)<=56 and frame['max_computed']>=16384]
    assert long_frames,'No actual long-context selected-batch Graph replay observed'
    launch=[v for v in trace['launches'].values() if v['actually_inside_capture'] and v['configuration']['IS_3D']
        and 33<=v['configuration']['sequences']<=56 and v['configuration']['segments']==16]
    assert launch
    files=[root/'comparison.json',root/'result.json',root/'before_metrics.txt',root/'after_metrics.txt',
        root/'hardware_after.txt',math/'accuracy_audit.json',*sorted((root/'benchmark_results').glob('*.csv'))]
    gate=dict(complete=True,independently_rechecked=True,adopted=True,original_comparison_not_overwritten=True,
        one_profile_for_both_workloads=True,math_correct=103,math_total=105,math_accuracy=103/105,
        successful_requests=1536,failed_requests=0,runtime_source_count=261,environment_count=52,separate_controller_count=7,
        both_original_command_and_actual_process_environment_verified=True,official_script_sha256=profile.BENCH_SHA,
        preemptions=0,all_virtual_reservations_released=True,metadata_fusion_not_included=True,
        actual_long_context_selected_batch_Graph_frames=long_frames,actual_S16_captures=sum(v['calls'] for v in launch),
        workloads=rows,raw_root=str(root),math_root=str(math),prior_qualified_raw_root=str(prior_path.parent),
        raw_files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},
        original_response_body_scope=current['raw_response_scope'],verifier_source_sha256=sha(__file__))
    (BASE/'OFFICIAL_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    (BASE/'RUNTIME_BINDING_GATE_32g.json').write_text(json.dumps(runtime,indent=2)+'\n')
    (root/'independent_official_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k not in ('raw_files','actual_long_context_selected_batch_Graph_frames')},ensure_ascii=False))
if __name__=='__main__':main()
