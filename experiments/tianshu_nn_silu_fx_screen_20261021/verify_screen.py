# SPDX-License-Identifier: Apache-2.0
"""Recompute every recorded round and verify original407 versus static/fused arms."""
import json,math,os,statistics
os.environ['VLLM_PLUGINS']=''
from pathlib import Path
import torch
from candidate_profile import BASE,read,digest,frozen_check
assert not torch.cuda.is_initialized()
original=read(BASE.parent/'tianshu_prefill_int8_official_results_20261021/RUNTIME_BINDING_GATE_32g.json')['frozen_sources'];assert len(original)==407
rows=[];arms={};evidence={};shared_seeds=None
for arm in ('control','candidate','reference'):
    root=Path('/workspace/benchmark-results')/('20261021_nn_silu_fx_screen_'+arm+'_tianshu');state=read(root/'result.json');arms[arm]=state
    assert state['phase']=='complete' and len(state['screen_rounds'])==3
    assert state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1])
    assert len(state['profile_environment'])==62 and len(state['server_command'])==13
    for p,h in state['frozen_sources'].items():assert digest(p)==h==digest(root/'source_snapshot'/Path(p).relative_to('/'))
    assert all(state['frozen_sources'][p]==h for p,h in original.items())
    if arm=='reference':assert state['frozen_sources']==original
    else:
        audit=state['profile_audit'];assert audit['mode']==arm and audit['fusion_mode']==arm and audit['prefill_mode']=='candidate'
        assert audit['full_inherited_attention_KV_scheduler_weights_LM_prefill_checks_retained'] and audit['native_dynamic_graphs_preserved'] and audit['original_407_source_map_exact_subset']
    seeds=[]
    for index,r in enumerate(state['screen_rounds']):
        assert r['round']==index+1 and r['successful_requests']==64 and len(r['responses'])==64 and not r['tuning_records']
        duration=r['duration_seconds'];assert math.isfinite(duration) and duration>0
        assert math.isclose(r['total_tokens_per_second'],64*5120/duration,rel_tol=1e-10)
        assert math.isclose(r['output_tokens_per_second'],64*1024/duration,rel_tol=1e-10)
        assert math.isclose(r['mean_ttft_seconds'],statistics.mean(a['ttft_seconds'] for a in r['responses']),rel_tol=1e-10)
        for a in r['responses']:
            assert a['usage']['prompt_tokens']==4096 and a['usage']['completion_tokens']==1024 and a['usage']['total_tokens']==5120
            assert a['input_tokens']==4096 and a['output_tokens']==1024 and a['text'] and 0<a['ttft_seconds']<=a['wall_seconds']
        seeds.append([a['input_seed'] for a in r['responses']])
        rows.append(dict(arm=arm,round=index+1,excluded=index==0,duration_seconds=duration,total_tokens_per_second=r['total_tokens_per_second'],output_tokens_per_second=r['output_tokens_per_second'],mean_TTFT_ms=1000*r['mean_ttft_seconds'],successful_requests=64,tuning_records=0))
    if shared_seeds is None:shared_seeds=seeds
    else:assert seeds==shared_seeds
    selected=[r for r in rows if r['arm']==arm and not r['excluded']]
    total=statistics.mean(r['total_tokens_per_second'] for r in selected);output=statistics.mean(r['output_tokens_per_second'] for r in selected);ttft=statistics.mean(r['mean_TTFT_ms'] for r in selected)
    spread=100*(max(r['total_tokens_per_second'] for r in selected)-min(r['total_tokens_per_second'] for r in selected))/total
    assert math.isclose(state['mean_total_tok_s'],total,rel_tol=1e-10) and math.isclose(state['spread_percent'],spread,abs_tol=1e-10)
    evidence[arm]=dict(raw_root=str(root),raw_result_sha256=digest(root/'result.json'),source_count=len(state['frozen_sources']),total_tok_s=total,output_tok_s=output,mean_TTFT_ms=ttft,spread_percent=spread,complete_requests=192,after_first_requests=128,all_rounds_successful=True,post_ready_autotune=0,full_runtime_audit_passed=True)
assert arms['control']['frozen_sources']==arms['candidate']['frozen_sources']
ratio=100*(evidence['candidate']['total_tok_s']/evidence['reference']['total_tok_s']-1)
static_ratio=100*(evidence['control']['total_tok_s']/evidence['reference']['total_tok_s']-1)
fusion_ratio=100*(evidence['candidate']['total_tok_s']/evidence['control']['total_tok_s']-1)
stable=all(row['spread_percent']<1 for row in evidence.values())
assert not torch.cuda.is_initialized()
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,adopted=False,official_benchmark=False,model_quality_not_yet_evaluated=True,
          real_original407_reference_verified=True,static_and_fused_share_same_sources=True,three_arms_same_request_seed_workloads=True,all_576_requests_successful=True,
          all_qualified_runtime_checks_passed=True,all_source_snapshots_rechecked=True,excluded_first_round_each_arm=True,
          candidate_vs_original407_percent=ratio,static64_vs_original407_percent=static_ratio,fusion_vs_static64_percent=fusion_ratio,all_spreads_below1_percent=stable,
          candidate_passes_perf_screen=(ratio>.5 and stable),static64_passes_perf_screen=(static_ratio>.5 and stable),new_official_and_C64_covered_quality_required=True,
          arms=evidence,rounds=rows,controller_verifier_sha256=digest(__file__),frozen_sources=arms['candidate']['frozen_sources'])
(BASE/'SCREEN_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'ROUNDS_32g.json').write_text(json.dumps(rows,indent=2)+'\n')
with (BASE/'KEY_ROUNDS_32g.csv').open('w') as f:
    import csv
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in gate.items() if k not in ('frozen_sources','rounds')}))
