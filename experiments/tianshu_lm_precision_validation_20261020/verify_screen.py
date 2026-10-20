# SPDX-License-Identifier: Apache-2.0
"""Independently verify the complete ready-path service pair, never a formal score."""
import argparse,hashlib,json,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalized(env,root):return {k:v.replace(str(root)+'/', '<OUTPUT>/') for k,v in env.items() if k!='TIANSHU_LM_PRECISION_MODE'}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runner',type=Path,required=True);args=parser.parse_args()
    assert (args.runner/'phase.txt').read_text().strip()=='complete'
    paths={mode:Path(f'/workspace/benchmark-results/20261020_lm_precision_service_{mode}_tianshu') for mode in ('control','candidate')}
    states={mode:read(root/'result.json') for mode,root in paths.items()}
    assert states['control']['frozen_sources']==states['candidate']['frozen_sources']
    sources=states['candidate']['frozen_sources']
    for path,digest in sources.items():assert sha(path)==digest
    assert len(sources)==359
    rows=[]
    for mode,s in states.items():
        root=paths[mode]
        assert s['phase']=='complete' and s['mode']==mode and len(s['profile_environment'])==62
        assert s['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1])
        assert len(s['screen_rounds'])==3 and all(r['successful_requests']==64 for r in s['screen_rounds'])
        for path,digest in sources.items():assert sha(root/'source_snapshot'/Path(path).relative_to('/'))==digest
        observed=read(root/'observed_server_environment.json');assert observed['environment']==s['profile_environment'] and observed['command']==s['server_command']
        assert s['profile_audit']['actual_ready_request_path_all_42_GateUp_verified']
        assert s['profile_audit']['all_43_banks_same_as_qualified_Math']
        assert not any(r['tuning_records'] for r in s['screen_rounds'][1:])
        values=[r['total_tokens_per_second'] for r in s['screen_rounds'][1:]]
        spread=100*(max(values)-min(values))/statistics.mean(values)
        assert abs(spread-s['spread_percent'])<1.e-9 and abs(statistics.mean(values)-s['mean_total_tok_s'])<1.e-9
        for r in s['screen_rounds']:
            assert len(r['responses'])==64
            assert all(x['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120,prompt_tokens_details=None) or (x['usage']['prompt_tokens']==4096 and x['usage']['completion_tokens']==1024 and x['usage']['total_tokens']==5120) for x in r['responses'])
            rows.append(dict(mode=mode,round=r['round'],total_tok_s=r['total_tokens_per_second'],output_tok_s=r['output_tokens_per_second'],ttft_ms=r['mean_ttft_seconds']*1000,successful_requests=64,excluded=r['round']==1))
    control,candidate=states['control'],states['candidate']
    assert control['server_command']==candidate['server_command']
    assert normalized(control['profile_environment'],paths['control'])==normalized(candidate['profile_environment'],paths['candidate'])
    gain=100*(candidate['mean_total_tok_s']/control['mean_total_tok_s']-1)
    ttft={m:statistics.mean(r['mean_ttft_seconds'] for r in s['screen_rounds'][1:])*1000 for m,s in states.items()}
    stable=all(s['spread_percent']<1 for s in states.values())
    previous=read(BASE.parent/"tianshu_prefill_linear_validation_20261020/SCREEN_GATE_32g.json")
    assert previous["complete"] and previous["gain_percent"]>1
    vs_qualified_mm=100*(candidate["mean_total_tok_s"]/previous["control_total_tok_s"]-1)
    result=dict(complete=True,independently_rechecked=True,official_benchmark=False,adopted=False,input_tokens=4096,output_tokens=1024,concurrency=64,
        requests_per_round=64,rounds_per_mode=3,excluded_rounds=[1],gain_percent=gain,control_total_tok_s=control['mean_total_tok_s'],candidate_total_tok_s=candidate['mean_total_tok_s'],
        control_ttft_ms=ttft['control'],candidate_ttft_ms=ttft['candidate'],control_spread_percent=control['spread_percent'],candidate_spread_percent=candidate['spread_percent'],
        frozen_sources=sources,runtime_source_count=len(sources),environment_count=62,raw_roots={k:str(v) for k,v in paths.items()},
        all_42_GateUp_actual_ready_calls_verified=True,all_43_banks_same_as_previous_qualified_Math=True,
        new_Math_not_run=True,small_LM_precision_gain_percent=gain,vs_previous_qualified_mm_screen_percent=vs_qualified_mm,prior_prefill_gain_percent=previous["gain_percent"],quality_worth_running=gain>=-.5 and vs_qualified_mm>=1 and stable and ttft['candidate']<=ttft['control']*1.01,
        postfirst_no_runtime_autotuning=True,same_cold_GPU_packing_and_SQL=True)
    for p in (args.runner/'SCREEN_GATE_32g.json',BASE/'SCREEN_GATE_32g.json'):p.write_text(json.dumps(result,indent=2)+'\n')
    (BASE/'SCREEN_ROUNDS_32g.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('frozen_sources','raw_roots')}))
if __name__=='__main__':main()
