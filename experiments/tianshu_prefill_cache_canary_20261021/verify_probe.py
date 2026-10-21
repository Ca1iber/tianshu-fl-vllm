# SPDX-License-Identifier: Apache-2.0
import json,sys
from pathlib import Path
from cache_probe_profile import BASE,CASES,payload,read,digest,frozen_check

def main():
    root=Path(sys.argv[1]);s=read(root/'result.json');assert s['phase']=='finished' and s['complete'] and s==read(root/'state.json')
    assert s['diagnostic_only'] and not s['official_benchmark'] and not s['adopted']
    frozen_check(s['frozen_sources']);frozen_check(s['controller_sources'])
    binding=read(BASE.parent/'tianshu_prefill_int8_official_results_20261021/RUNTIME_BINDING_GATE_32g.json')
    assert s['frozen_sources']==binding['frozen_sources'] and len(s['frozen_sources'])==407 and len(s['profile_environment'])==62
    for p,h in s['frozen_sources'].items():assert digest(root/'source_snapshot'/Path(p).relative_to('/'))==h
    for p,h in s['controller_sources'].items():assert digest(root/'controller_source_snapshot'/Path(p).relative_to('/'))==h
    observed=read(root/'observed_server_environment.json');assert observed['environment']==s['profile_environment'] and observed['command']==s['server_command']
    assert len(s['cases'])==8;ids={};rows=[]
    for i,(r,(label,salt)) in enumerate(zip(s['cases'],CASES)):
        assert r['label']==label and r['cache_salt']==salt and r['payload']==payload(salt)
        assert r['response']['usage']==dict(prompt_tokens=128,completion_tokens=16,total_tokens=144)
        choice=r['response']['choices'][0];assert len(choice['token_ids'])==16 and choice['text']
        ids[label]=choice['token_ids'];b,a=r['before_metrics'],r['after_metrics']
        assert b['num_preemptions_total']==a['num_preemptions_total']==0
        rows.append(dict(label=label,cache_salt=salt,token_ids=choice['token_ids'],prefix_queries_delta=a['prefix_cache_queries_total']-b['prefix_cache_queries_total'],prefix_hits_delta=a['prefix_cache_hits_total']-b['prefix_cache_hits_total']))
    cold=[ids[x+'_cold'] for x in 'ABC'];cached=[ids[x+'_cached'] for x in 'ABC']
    comparisons=dict(cold_cold_token_equal=all(x==cold[0] for x in cold),cached_cached_token_equal=all(x==cached[0] for x in cached),
                     all_cold_cached_token_equal=all(ids[x+'_cold']==ids[x+'_cached'] for x in 'ABC'),unsalted_first_repeat_equal=ids['unsalted_first']==ids['unsalted_repeat'])
    for name in ('hardware_before.txt','hardware_after.txt'):assert 'No running processes found' in (root/name).read_text()
    gate=dict(complete=True,independently_CPU_rechecked=True,diagnostic_only=True,model_quality_not_evaluated=True,official_benchmark=False,adopted=False,
              exact_407_sources_and_62_environment_retained=True,no_stage_observer_hook=True,cases=rows,comparisons=comparisons,raw_root=str(root),raw_result_sha256=digest(root/'result.json'),verifier_sha256=digest(__file__))
    (BASE/'CACHE_CANARY_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
