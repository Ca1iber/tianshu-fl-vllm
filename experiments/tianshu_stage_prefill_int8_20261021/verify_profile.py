# SPDX-License-Identifier: Apache-2.0
"""Offline recheck all stage frames, responses and qualified runtime evidence."""
import json,statistics,sys
from pathlib import Path
import torch
import stage_measurement_profile as profile
BASE=Path(__file__).resolve().parent

def main():
    assert not torch.cuda.is_initialized()
    root=Path(sys.argv[1]);state=profile.read(root/'result.json')
    assert state['phase']=='complete' and state['complete'] and state['actual_budget']==4096
    assert not state['official_benchmark'] and not state['adopted']
    profile.frozen_check(state['frozen_sources'])
    binding=profile.read(BASE.parent/'tianshu_prefill_int8_official_results_20261021/RUNTIME_BINDING_GATE_32g.json')
    assert len(binding['frozen_sources'])==407
    assert all(state['frozen_sources'][p]==sha for p,sha in binding['frozen_sources'].items())
    for p,sha in state['frozen_sources'].items():assert profile.digest(root/'source_snapshot'/Path(p).relative_to('/'))==sha
    runtime=profile.audit_runtime(root,state,official=True)
    assert state['runtime_audit']==runtime and runtime['zero_preemptions']
    assert runtime['post_ready_autotuning_records']==0
    assert len(state['workloads'])==2
    for index,work in enumerate(state['workloads']):
        assert work['label']==('warmup','measured')[index]
        assert work['input_tokens']==4096 and work['output_tokens']==512 and work['concurrency']==64
        assert len(work['responses'])==64 and work['duration_seconds']>0
        assert len({r['input_seed'] for r in work['responses']})==64
        for r in work['responses']:
            assert r['usage']==dict(prompt_tokens=4096,completion_tokens=512,total_tokens=4608)
            assert r['text'] and r['wall_seconds']>=r['ttft_seconds']>0 and r['api_seed'] is None
    before,after=state['canary_before'],state['canary_after']
    assert before['text']==after['text'] and before['api_seed']==after['api_seed']==777
    assert before['input_seed']==after['input_seed']==20261021088
    for r in (before,after):assert r['usage']==dict(prompt_tokens=128,completion_tokens=16,total_tokens=144)
    files=list((root/'stage_records').glob('stages_*.json'));assert len(files)==1
    stages=profile.read(files[0]);assert stages['complete'] and stages['frame_count']==40 and len(stages['frames'])==40
    assert stages['events_inside_graph'] is False and stages['ixsys_not_used'] and stages['boundary_synchronizations']==2
    assert stages['runner_source_sha256']=='b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'
    assert stages['hook_source_sha256']==profile.digest(BASE/'hooks/sitecustomize.py')
    assert stages['model_type'].endswith('LlamaForCausalLM')
    frames=stages['frames'];keys=('total','forward','logits','sample')
    for index,row in enumerate(frames):
        assert row['index']==index and row['dispatch']==dict(mode='FULL',padded_tokens=64)
        context=row['context'];assert context['requests']==64 and context['min_computed']>=4096
        assert context['min_prompt']==context['max_prompt']==4096
        gpu=row['gpu_stream_intervals_ms'];assert set(gpu)==set(keys) and all(v>0 for v in gpu.values())
        assert gpu['total']+.1>=sum(gpu[k] for k in keys[1:])
        assert set(row['cpu_call_spans'])=={'forward','logits','sample','execute','sample_tokens'}
        assert len({v['thread_id'] for v in row['cpu_call_spans'].values()})==1
        assert all(v['wall_ms']>=0 and v['thread_cpu_ms']>=0 for v in row['cpu_call_spans'].values())
    for p,meta in stages['extra_sources'].items():assert profile.digest(p)==meta['sha256']==profile.digest(root/'stage_records'/meta['snapshot'])
    sums={k:sum(row['gpu_stream_intervals_ms'][k] for row in frames) for k in keys}
    medians={k:statistics.median(row['gpu_stream_intervals_ms'][k] for row in frames) for k in keys}
    fractions={k:sums[k]/sums['total'] for k in keys[1:]}
    summary=profile.read(root/'summary.json')
    assert summary==state['summary'] and summary['complete'] and summary['frame_count']==40
    assert summary['gpu_stream_interval_sum_ms']==sums and summary['gpu_stream_interval_median_ms']==medians
    assert summary['fractions_of_total_stream_interval']==fractions
    expected_host={k:{kind:statistics.median(r['cpu_call_spans'][k][kind] for r in frames) for kind in ('wall_ms','thread_cpu_ms')} for k in ('execute','sample_tokens','forward','logits','sample')}
    assert summary['host_call_median']==expected_host
    assert 'No running processes found' in (root/'hardware_before.txt').read_text() and 'No running processes found' in (root/'hardware_after.txt').read_text()
    assert not torch.cuda.is_initialized()
    gate=dict(complete=True,independently_CPU_rechecked=True,CUDA_not_initialized=True,
              official_benchmark=False,adopted=False,diagnostic_only=True,qualified_407_source_subset_byte_identical=True,
              original_13_argv_and_62_environment_keys_verified=True,source_count=len(state['frozen_sources']),
              actual_42_GateUp_prefill_NN_decode_and_LM_routes_verified=True,seeded_canary_before_after_equal=True,
              successful_load_requests=128,canary_requests=2,frame_count=40,events_inside_graph=False,
              boundary_synchronizations=2,zero_preemptions=True,post_ready_autotuning_records=0,
              summary=summary,raw_root=str(root),raw_result_sha256=profile.digest(root/'result.json'),
              frozen_sources=state['frozen_sources'],verifier_sha256=profile.digest(__file__))
    (BASE/'STAGE_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    (BASE/'STAGE_SUMMARY_32g.json').write_text(json.dumps(summary,indent=2)+'\n')
    (BASE/'STAGE_FRAMES_32g.json').write_text(json.dumps(stages,indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k!='frozen_sources'}))
if __name__=='__main__':main()
