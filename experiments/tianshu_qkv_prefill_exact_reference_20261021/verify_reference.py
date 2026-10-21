# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,importlib.util,json,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent
SERVICE=BASE.parent/'tianshu_qkv_prefill_int8_service_v2_20261021'
ORIGINAL=BASE.parent/'tianshu_prefill_int8_service_20261021'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
old=module('_exact407_current_audit',ORIGINAL/'audit_profile.py')
new=module('_exact407_QKV_candidate_audit',SERVICE/'audit_profile.py')
prior=read(SERVICE/'SCREEN_GATE_32g.json');assert prior['complete'] and prior['predeclared_service_gate_passed']
roots=dict(reference=Path('/workspace/benchmark-results/20261021_qkv_prefill_exact407_reference_tianshu'),candidate=Path(prior['raw_roots']['candidate']))
states={};summary={};rows=[]
for label,root in roots.items():
    state=read(root/'result.json');states[label]=state;assert state['phase']=='complete' and state['mode']=='candidate' and state['actual_budget']==4096
    audit=(old if label=='reference' else new).audit_runtime(root,state,official=True)
    assert audit==state['profile_audit'] and audit['zero_preemptions']
    assert state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1])
    assert len(state['screen_rounds'])==3
    for p,h in state['frozen_sources'].items():assert sha(p)==h==sha(root/'source_snapshot'/Path(p).relative_to('/'))
    canary=state['seeded_canaries'];assert len(canary)==2 and canary[0]['text']==canary[1]['text'] and canary[0]['cache_salt']!=canary[1]['cache_salt']
    for run in state['screen_rounds']:
        assert run['successful_requests']==len(run['responses'])==64
        assert abs(run['total_tokens_per_second']-64*5120/run['duration_seconds'])<1.e-8
        assert all(r['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120) and r['api_seed'] is None and r['cache_salt'] is None for r in run['responses'])
        rows.append(dict(arm=label,round=run['round'],total_tok_s=run['total_tokens_per_second'],output_tok_s=run['output_tokens_per_second'],TTFT_ms=run['mean_ttft_seconds']*1000,tuning_records=len(run['tuning_records'])))
    measured=state['screen_rounds'][1:];values=[r['total_tokens_per_second'] for r in measured];mean=statistics.mean(values)
    summary[label]=dict(total_tok_s=mean,mean_TTFT_ms=statistics.mean(r['mean_ttft_seconds'] for r in measured)*1000,spread_percent=100*(max(values)-min(values))/mean,
        postfirst_autotuning_records=sum(len(r['tuning_records']) for r in measured),actual_KV_blocks=audit['actual_KV_blocks'])
reference,candidate=states['reference'],states['candidate']
assert len(reference['frozen_sources'])==407 and len(candidate['frozen_sources'])==432
assert all(candidate['frozen_sources'].get(p)==h for p,h in reference['frozen_sources'].items())
assert reference['server_command']==candidate['server_command']
normalized={k:v.replace(str(roots['candidate'])+'/',str(roots['reference'])+'/').replace(str(SERVICE/'hooks_candidate'),str(ORIGINAL/'hooks_candidate')) for k,v in candidate['profile_environment'].items()}
assert len(normalized)==62 and normalized==reference['profile_environment']
gain=100*(summary['candidate']['total_tok_s']/summary['reference']['total_tok_s']-1)
ttft=100*(summary['candidate']['mean_TTFT_ms']/summary['reference']['mean_TTFT_ms']-1)
passed=bool(gain>max(.5,*[s['spread_percent'] for s in summary.values()]) and max(s['spread_percent'] for s in summary.values())<.5 and ttft<=1 and sum(s['postfirst_autotuning_records'] for s in summary.values())==0)
gate=dict(complete=True,official_benchmark=False,adopted=False,independently_rechecked=True,reference_is_unmodified_current_407=True,all_407_runtime_sources_exact_subset_of_candidate=True,
    V2_wrapper_control_alone_not_used_for_adoption=True,predeclared_exact_reference_gate_passed=passed,gain_percent=gain,TTFT_change_percent=ttft,summaries=summary,
    raw_roots={k:str(v) for k,v in roots.items()},raw_result_sha256={k:sha(v/'result.json') for k,v in roots.items()},new_Math_required=passed,model_quality_not_evaluated=True,verifier_sha256=sha(__file__))
(BASE/'REFERENCE_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'REFERENCE_ROUNDS_32g.json').write_text(json.dumps(rows,indent=2)+'\n')
with (BASE/'KEY_ROUNDS_32g.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(json.dumps(gate))
