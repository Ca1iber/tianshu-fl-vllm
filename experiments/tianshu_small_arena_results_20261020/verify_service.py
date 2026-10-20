# SPDX-License-Identifier: Apache-2.0
"""Offline source, graph, exact request/token and paired measurement verification."""
import argparse,csv,hashlib,json,math,re,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_small_arena_service_20261020'))
from audit_profile import audit_runtime

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def success(path):
    values={}
    for line in path.read_text().splitlines():
        if line.startswith('vllm:request_success_total'):
            reason=re.search(r'finished_reason="([^"]+)"',line)[1]
            values[reason]=values.get(reason,0)+float(line.rsplit(' ',1)[1])
    return values
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--label',choices=('4K','16K'),required=True);args=p.parse_args()
root=args.root.resolve();assert root.is_relative_to('/workspace/benchmark-results')
comparison=read(root/'comparison.json');assert comparison['complete']
assert comparison['input_tokens']==(16384 if args.label=='16K' else 4096)
result=dict(comparison);result['modes']={}
for mode in ('control','candidate'):
    path=root/mode/'result.json';state=read(path)
    assert state['phase']=='complete' and len(state['screen_rounds'])==3
    assert state['frozen_sources']==comparison['frozen_sources']
    audit=audit_runtime(root/mode,state,official=(args.label=='4K'))
    assert audit['actual_FL_Graph_capture_and_replay_verified']
    assert audit['actual_arena_reserved_inside_memory_profiling_verified'] and audit['arena_capacity_pages']==(8192 if mode=='candidate' else 32768)
    assert audit['active_kernel_sha256']=='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'
    rounds=[];round_seeds=[]
    for row in state['screen_rounds']:
        assert len(row['responses'])==row['successful_requests']==64
        assert math.isfinite(row['duration_seconds']) and row['duration_seconds']>0
        for response in row['responses']:
            assert response['usage']==dict(prompt_tokens=comparison['input_tokens'],completion_tokens=1024,total_tokens=comparison['input_tokens']+1024)
            assert response['text'] and response['ttft_seconds']>0 and math.isfinite(response['ttft_seconds'])
        seeds=[response['input_seed'] for response in row['responses']]
        assert len(set(seeds))==64 and all(response['api_seed'] is None for response in row['responses'])
        round_seeds.append(seeds)
        total=64*(comparison['input_tokens']+1024)/row['duration_seconds']
        assert abs(total-row['total_tokens_per_second'])<1e-10
        rounds.append({key:value for key,value in row.items() if key!='responses'})
    before,after=success(root/mode/'before_metrics.txt'),success(root/mode/'after_metrics.txt')
    assert after.get('error',0)==after.get('abort',0)==0
    assert sum(after.values())-sum(before.values())==192
    canary=[c['text'] for c in state['seeded_canaries']];assert len(canary)==2 and canary[0]==canary[1]
    steady=[r['total_tokens_per_second'] for r in rounds[1:]]
    assert statistics.mean(steady)==state['mean_total_tok_s']
    result['modes'][mode]=dict(rounds=rounds,round_input_seeds=round_seeds,mean_total_tok_s=state['mean_total_tok_s'],spread_percent=state['spread_percent'],
                              mean_TTFT_seconds=statistics.mean(r['mean_ttft_seconds'] for r in rounds[1:]),
                              profile_audit=audit,profile_environment=state['profile_environment'],
                              seeded_canary_sha256=hashlib.sha256(canary[0].encode()).hexdigest(),
                              metrics_success_before=before,metrics_success_after=after,
                              original_result_path=str(path),original_result_sha256=sha(path))
a,b=(result['modes'][x] for x in ('control','candidate'))
assert a['round_input_seeds']==b['round_input_seeds']
assert len(set(seed for row in a['round_input_seeds'] for seed in row))==192
assert a['seeded_canary_sha256']==b['seeded_canary_sha256']
assert abs(100*(b['mean_total_tok_s']/a['mean_total_tok_s']-1)-comparison['gain_percent'])<1e-10
result.update(complete=True,independently_rechecked=True,exact_screen_requests_verified=384,
              original_comparison_sha256=sha(root/'comparison.json'),analyzer_source_sha256=sha(__file__),
              no_full_responses_or_tensor_data_in_key_result=True,official_benchmark=False,adopted=False)
(BASE/f'SERVICE_{args.label}_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
with (BASE/f'SERVICE_{args.label}_ROUNDS_32g.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['mode','round','total_tok_s','output_tok_s','TTFT_ms','duration_s','success','excluded'])
    for mode,part in result['modes'].items():
        for row in part['rounds']:
            writer.writerow([mode,row['round'],row['total_tokens_per_second'],row['output_tokens_per_second'],row['mean_ttft_seconds']*1000,row['duration_seconds'],64,row['round']==1])
print(json.dumps(dict(complete=True,label=args.label,control_total_tok_s=a['mean_total_tok_s'],candidate_total_tok_s=b['mean_total_tok_s'],gain_percent=comparison['gain_percent'],frozen_source_count=len(result['frozen_sources']))))
