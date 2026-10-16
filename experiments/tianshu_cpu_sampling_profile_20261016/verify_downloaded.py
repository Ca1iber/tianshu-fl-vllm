# SPDX-License-Identifier: Apache-2.0
"""Verify raw CPU call diagnostic, all responses, source snapshots and timing ranks."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from audit_profile import audit_runtime


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve();s=read(root/'result.json')
    assert s['phase']=='complete' and s['complete'] and not s['official_benchmark'] and not s['adopted']
    assert not s['ixsys_used'] and len(s['frozen_sources'])>=47
    repo=Path(__file__).resolve().parents[2]
    for source,digest in s['frozen_sources'].items():
        assert sha(root/'source_snapshot'/Path(source).relative_to('/'))==digest
        if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
            assert sha(repo/Path(source).relative_to('/workspace/vllm-plugin-FL'))==digest
    audit_runtime(root,s,official=True,online=False)
    files=list((root/'cpu_records').glob('cpu_*.json'));assert len(files)==1
    record=read(files[0]);assert record['complete'] and record['frame_count']==len(record['frames'])==16
    assert not record['GPU_events_used'] and record['pid']>0
    hook='/workspace/vllm-plugin-FL/experiments/tianshu_cpu_sampling_profile_20261016/hooks/sitecustomize.py'
    assert record['hook_source_sha256']==s['frozen_sources'][hook]
    assert record['runner_source_sha256']=='b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'
    for source,metadata in record['extra_sources'].items():
        assert sha(root/'cpu_records'/metadata['snapshot'])==s['frozen_sources'][source]==metadata['sha256']
    phases=('execute_model','sample_tokens');hot={p:defaultdict(lambda:[0.,0.,0]) for p in phases};count=0
    for index,frame in enumerate(record['frames']):
        assert frame['index']==index and frame['dispatch']==dict(mode='FULL',padded_tokens=64)
        context=frame['context'];assert context['requests']==64 and context['min_computed']>=4096 and context['min_prompt']==context['max_prompt']==4096
        buffers=frame['buffers'];assert set(buffers)=={'logits','hidden_states','sample_hidden_states','temperature','top_p','top_k'}
        assert buffers['logits']['shape']==[64,130560] and buffers['sample_hidden_states']['shape']==[64,2048]
        for name,description in buffers.items():
            if description is None:continue
            assert description['data_ptr']>0 and description['storage_ptr']>0 and description['storage_offset']>=0
            assert len(description['shape'])==len(description['stride']) and all(n>=0 for n in description['shape'])
            assert description['device'].startswith('cuda')
        flags=frame['sampling_flags']
        assert flags['all_random'] and not flags['all_greedy'] and flags['no_penalties']
        assert flags['max_num_logprobs'] is None and flags['generators']==flags['bad_words_requests']==0
        assert flags['allowed_token_ids_mask'] is None
        for phase in phases:
            call=frame[phase];assert 0<=call['thread_cpu_ms']<=call['wall_ms']+.1 and call['wall_ms']>0 and call['thread_id']>0
            rows=call['rows'];assert len(rows)>10 and any(r['name']==phase for r in rows)
            for row in rows:
                assert row['primitive_calls']<=row['total_calls'] and row['total_calls']>0
                assert math.isfinite(row['local_wall_seconds']) and math.isfinite(row['cumulative_wall_seconds'])
                assert 0<=row['local_wall_seconds']<=row['cumulative_wall_seconds']+1e-9
                key=(row['file'],row['line'],row['name']);v=hot[phase][key]
                v[0]+=row['local_wall_seconds']*1000;v[1]+=row['cumulative_wall_seconds']*1000;v[2]+=row['total_calls'];count+=1
                assert all(isinstance(x['file'],str) and isinstance(x['name'],str) for x in row['callers'])
    for index,workload in enumerate(s['workloads']):
        assert workload['label']==('warmup' if index==0 else 'measured') and workload['input_tokens']==4096 and workload['output_tokens']==256
        responses=workload['responses'];assert len(responses)==64
        assert [v['input_seed'] for v in responses]==[20261016000+1000*index+i for i in range(64)]
        for response in responses:
            assert response['usage']['prompt_tokens']==4096 and response['usage']['completion_tokens']==256 and response['usage']['total_tokens']==4352
            assert response['api_seed'] is None and response['text'] and 0<response['ttft_seconds']<=response['wall_seconds']<=workload['duration_seconds']+.1
    assert len(s['workloads'])==2
    for key in ('canary_before','canary_after'):
        c=s[key];assert c['api_seed']==777 and c['input_seed']==2026101688 and c['usage']['prompt_tokens']==128 and c['usage']['completion_tokens']==16
    assert s['canary_before']['text']==s['canary_after']['text']
    medians={p:{k:statistics.median(f[p][k] for f in record['frames']) for k in ('wall_ms','thread_cpu_ms')} for p in phases}
    assert medians==read(root/'summary.json')['host_profiled_call_median']
    tops={p:[dict(file=k[0],line=k[1],name=k[2],total_self_ms=v[0],total_cumulative_ms=v[1],total_calls=v[2]) for k,v in sorted(hot[p].items(),key=lambda item:-item[1][0])[:30]] for p in phases}
    pointers={name:dict(unique_data_ptrs=sorted({f['buffers'][name]['data_ptr'] for f in record['frames'] if f['buffers'][name] is not None}),
        all_frames_present=all(f['buffers'][name] is not None for f in record['frames'])) for name in record['frames'][0]['buffers']}
    out=dict(complete=True,official_benchmark=False,adopted=False,all128_responses_and_two_canaries_verified=True,
        all16_FULL_C64_frames_and32_function_profiles_verified=True,all_source_snapshots_verified=True,
        raw_function_records_recomputed=count,profiled_host_medians=medians,top_self_time_functions=tops,
        recorded_tensor_pointer_variation=pointers,all16_buffer_descriptions_and_sampling_flags_verified=True,
        interpretation='Intrusive CPU host-wall profiler, cumulative call times overlap. Not unprofiled throughput or GPU kernel-active fractions.')
    (root/'local_download_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='top_self_time_functions'}))


if __name__=='__main__':main()
