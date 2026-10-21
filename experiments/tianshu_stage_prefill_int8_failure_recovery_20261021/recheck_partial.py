# SPDX-License-Identifier: Apache-2.0
"""CPU-only salvage of timing evidence; retain failed canary and absent audit."""
import hashlib,json,math,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261021_stage_prefill_int8_tianshu')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    state=read(ROOT/'result.json');assert state==read(ROOT/'state.json')
    assert state['phase']=='failed' and state['error']=='AssertionError()' and 'runtime_audit' not in state
    assert not state['official_benchmark'] and not state['adopted']
    assert not (ROOT/'after_metrics.txt').exists() and not (ROOT/'summary.json').exists()
    assert len(state['workloads'])==2
    for index,work in enumerate(state['workloads']):
        assert work['label']==('warmup','measured')[index] and len(work['responses'])==64
        for r in work['responses']:
            assert r['usage']==dict(prompt_tokens=4096,completion_tokens=512,total_tokens=4608)
            assert r['text'] and r['wall_seconds']>=r['ttft_seconds']>0
    first,last=state['canary_before'],state['canary_after']
    assert first['text']!=last['text'] and first['api_seed']==last['api_seed']==777
    assert first['input_seed']==last['input_seed']==20261021088
    for r in (first,last):assert r['usage']==dict(prompt_tokens=128,completion_tokens=16,total_tokens=144)
    binding=read(BASE.parent/'tianshu_prefill_int8_official_results_20261021/RUNTIME_BINDING_GATE_32g.json')
    assert len(binding['frozen_sources'])==407
    assert all(state['frozen_sources'][p]==h for p,h in binding['frozen_sources'].items())
    for p,h in state['frozen_sources'].items():assert sha(p)==h==sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))
    observed=read(ROOT/'observed_server_environment.json')
    assert observed['environment']==state['profile_environment'] and len(observed['environment'])==62 and observed['command']==state['server_command']
    files=list((ROOT/'stage_records').glob('stages_*.json'));assert len(files)==1
    stage=read(files[0]);assert stage['complete'] and stage['frame_count']==40 and len(stage['frames'])==40
    assert stage['model_type']=='vllm_fl.compilation.graph.GraphWrapper'
    assert not stage['events_inside_graph'] and stage['ixsys_not_used'] and stage['boundary_synchronizations']==2
    hook=str(BASE.parent/'tianshu_stage_prefill_int8_20261021/hooks/sitecustomize.py')
    assert stage['hook_source_sha256']==state['frozen_sources'][hook]
    for i,r in enumerate(stage['frames']):
        assert r['index']==i and r['dispatch']==dict(mode='FULL',padded_tokens=64)
        assert r['context']['requests']==64 and r['context']['min_computed']>=4096
        assert r['context']['min_prompt']==r['context']['max_prompt']==4096
        v=r['gpu_stream_intervals_ms'];assert set(v)=={'total','forward','logits','sample'}
        assert all(math.isfinite(x) and x>0 for x in v.values())
        assert v['total']+.1>=v['forward']+v['logits']+v['sample']
    for p,m in stage['extra_sources'].items():assert sha(p)==m['sha256']==sha(ROOT/'stage_records'/m['snapshot'])
    assert 'No running processes found' in (ROOT/'hardware_before.txt').read_text() and 'No running processes found' in (ROOT/'hardware_after.txt').read_text()
    keys=('total','forward','logits','sample')
    sums={k:sum(r['gpu_stream_intervals_ms'][k] for r in stage['frames']) for k in keys}
    medians={k:statistics.median(r['gpu_stream_intervals_ms'][k] for r in stage['frames']) for k in keys}
    fractions={k:sums[k]/sums['total'] for k in keys[1:]}
    host={k:{kind:statistics.median(r['cpu_call_spans'][k][kind] for r in stage['frames']) for kind in ('wall_ms','thread_cpu_ms')} for k in ('execute','sample_tokens','forward','logits','sample')}
    summary=dict(frame_count=40,gpu_stream_interval_sum_ms=sums,gpu_stream_interval_median_ms=medians,fractions_of_total_stream_interval=fractions,host_call_median=host,
                 interpretation='Partial diagnostic only: canary failed, full runtime audit absent; stream intervals include submission gaps, not kernel active time.')
    log=(ROOT/'server.log').read_text(errors='replace');assert log.count('Application startup complete.')==1
    newtunes=[r for r in log.split('Application startup complete.',1)[1].splitlines() if 'Triton autotuning for function' in r]
    gate=dict(complete=True,recovery_scope='CPU-only validation of saved evidence, not conversion of failure to pass',original_experiment_passed=False,original_phase='failed',canary_equal=False,
              full_runtime_audit_completed=False,after_metrics_missing=True,not_a_quality_or_performance_acceptance=True,GPU_rerun=False,official_benchmark=False,adopted=False,
              original_407_sources_byte_identical=True,actual_environment_count=62,observed_GraphWrapper_not_Llama_class_expected_by_unexecuted_verifier=True,
              successful_load_requests=128,frame_count=40,summary=summary,post_ready_autotuning_records=len(newtunes),
              raw_root=str(ROOT),raw_result_sha256=sha(ROOT/'result.json'),raw_stage_sha256=sha(files[0]),verifier_sha256=sha(__file__))
    (BASE/'PARTIAL_DIAGNOSTIC_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    (BASE/'STAGE_FRAMES_32g.json').write_text(json.dumps(stage,indent=2)+'\n')
    print(json.dumps(gate))
if __name__=='__main__':main()
