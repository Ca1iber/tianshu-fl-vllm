# SPDX-License-Identifier: Apache-2.0
"""Run only after the complete quality-gated control/candidate N128 screens."""
import csv,hashlib,importlib.util,json,math,re,statistics,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
EXP=BASE.parent
PAIR=Path('/dev/shm/tianshu_r5_BM8_runs_20261024/fresh128_pair')
SERVICE=EXP/'tianshu_r5_BM8_KV_20261024'
sys.path.insert(0,str(SERVICE))
from audit import audit_model,audit_math
from candidate_profile import digest,read,frozen_check

def csv_rows(root,pattern):
    files=list(root.glob('benchmark_results/'+pattern));assert len(files)==1,(root,pattern,files)
    with files[0].open(newline='') as f:return list(csv.DictReader(f)),files[0]
def counter(path,name):
    rows=[float(line.rsplit(' ',1)[1]) for line in path.read_text().splitlines()
        if line.startswith('vllm:'+name+'{')]
    assert len(rows)==1,(path,name,len(rows));return rows[0]
def normalized(env,root):return {k:v.replace(str(root),'<ROOT>') for k,v in env.items()}
def main():
    assert (PAIR/'phase.txt').read_text().strip()=='both_screens_complete_pending_comparison'
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    assert not (BASE/'SCREEN_GATE_32g.json').exists()
    quality={};states={};raw_hashes={};scores=[];arms={}
    for batch in (8,64):
        root=Path(f'/dev/shm/tianshu_r5_BM8_runs_20261024/math{batch}')
        state=read(root/'result.json');assert state['phase']=='complete' and state['evaluation_exit_code']==0
        assert state['batch']==batch and state['accuracy_passed'] and state['mode']=='candidate' and state['bm8_mode']=='candidate'
        q=audit_math(root,state);assert q==read(root/'accuracy_audit.json') and q['above95'] and q['total']==105
        profile=audit_model(root,state,runtime=True)
        assert json.loads(json.dumps(profile))==read(root/'profile_audit.json')
        quality[str(batch)]=dict(correct=q['correct'],total=105,accuracy=q['accuracy'],all105_counted=True,
            stop_reasons=q['stop_reasons'],actual_FULL64_replays=profile['actual_FULL64_replays'],raw_root=str(root))
        states['math'+str(batch)]=state
        for name in ('result.json','accuracy_audit.json','profile_audit.json'):raw_hashes[str(root/name)]=digest(root/name)
    for arm in ('control','candidate'):
        root=PAIR/arm;state=read(root/'result.json');states[arm]=state
        assert state['phase']=='complete' and state['evaluation_exit_code']==0
        assert state['mode']=='candidate' and state['bm8_mode']==arm and state['stage']=='screen' and state['batch']==64
        assert state['performance_preexcluded_rounds']==[1]
        assert len(state['frozen_sources'])==1023 and len(state['full_environment'])==67
        assert state['frozen_sources']==states['math8']['frozen_sources']==states['math64']['frozen_sources']
        assert state['server_command']==states['math8']['server_command']==states['math64']['server_command']
        assert read(root/'observed_environment.json')==state['full_environment']
        assert state['full_environment']['TIANSHU_FUTURE_KV_MODE']=='candidate'
        for p,h in state['frozen_sources'].items():
            assert digest(Path(p))==h==digest(root/'source_snapshot'/Path(p).relative_to('/'))
        profile=audit_model(root,state,runtime=True)
        assert json.loads(json.dumps(profile))==read(root/'profile_audit.json')
        assert profile['physical_KV_blocks']*16>=64*(4096+1024)
        assert profile['additional_payload_bytes']==210567168
        assert profile['QKVO64_capture_banks']==84
        assert profile['QKV64_BF16_capture_banks']==0 and profile['actual_FULL64_replays']>=10
        assert profile['O64_BF16_capture_banks']==0
        rows,rp=csv_rows(root,'raw_runs_*.csv');summary,sp=csv_rows(root,'summary_*.csv')
        assert len(rows)==4 and len(summary)==1
        for i,row in enumerate(rows,1):
            config=read(root/'client_configuration.json')
            assert config==dict(case=[4096,1024,64,128],seeds=[2026102401,2026102402,2026102403,2026102404],excluded_rounds=[1],max_prefix_hit_fraction=.01,max_preemptions=0,save_full_input_bodies=False)
            fresh=read(root/f'client_rounds/round_{i}.json')
            assert fresh['round']==i and fresh['seed']==config['seeds'][i-1] and fresh['excluded']==(i==1)
            assert fresh['metrics']['successful_requests']==128
            before,after=root/f'client_rounds/before_{i}.txt',root/f'client_rounds/after_{i}.txt'
            deltas={n:counter(after,n)-counter(before,n) for n in ('prefix_cache_queries_total','prefix_cache_hits_total','num_preemptions_total')}
            assert deltas==fresh['counters'] and deltas['num_preemptions_total']==0 and deltas['prefix_cache_queries_total']>=524288
            fraction=deltas['prefix_cache_hits_total']/deltas['prefix_cache_queries_total']
            assert 0<=fraction<=.01 and fraction==fresh['prefix_hit_fraction']
            actual_seeds=re.findall(r'(?<![A-Za-z0-9_])seed=([0-9]+)',(root/f'client_rounds/round_{i}.log').read_text())
            assert actual_seeds==[str(config['seeds'][i-1])]
            assert read((PAIR/('candidate' if arm=='control' else 'control'))/'client_configuration.json')==config
            for p,h in fresh['raw_sha256'].items():assert digest(Path(p))==h
            for p in (before,after,root/f'client_rounds/round_{i}.json',root/f'client_rounds/round_{i}.log',root/'client_configuration.json'):raw_hashes[str(p)]=digest(p)
            assert [int(row[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,128]
            assert row['Run Status']=='SUCCESS' and float(row['Successful Requests'])==128
            assert float(row['Total Input Tokens'])==524288 and float(row['Total Output Tokens'])==131072
            assert all(math.isfinite(float(row[k])) and float(row[k])>0 for k in ('Benchmark Duration (s)','Total tok/s','Output tok/s','Mean TTFT (ms)'))
            assert math.isclose(float(row['Total tok/s']),655360/float(row['Benchmark Duration (s)']),rel_tol=.0002)
            scores.append(dict(arm=arm,round=i,excluded=i==1,total_tok_s=float(row['Total tok/s']),
                output_tok_s=float(row['Output tok/s']),ttft_ms=float(row['Mean TTFT (ms)']),successful=128))
        hot=rows[1:];means={k:statistics.mean(float(r[k]) for r in hot) for k in ('Total tok/s','Output tok/s','Mean TTFT (ms)')}
        for k,mean in means.items():assert abs(float(summary[0][k])-mean)<=.011
        spread=100*(max(float(r['Total tok/s']) for r in hot)-min(float(r['Total tok/s']) for r in hot))/means['Total tok/s']
        preempt=counter(root/'after_metrics.txt','num_preemptions_total')-counter(root/'before_metrics.txt','num_preemptions_total')
        assert preempt==0
        arms[arm]=dict(total_tok_s=means['Total tok/s'],output_tok_s=means['Output tok/s'],ttft_ms=means['Mean TTFT (ms)'],
            spread_percent=spread,physical_KV_blocks=profile['physical_KV_blocks'],actual_FULL64_replays=profile['actual_FULL64_replays'],preemptions=preempt,
            source_count=1023,environment_count=67,additional_payload_bytes=210567168,all512_requests_succeeded=True)
        for p in [rp,sp,root/'result.json',root/'evaluation.log',root/'profile_audit.json',root/'observed_environment.json',root/'before_metrics.txt',root/'after_metrics.txt']:
            raw_hashes[str(p)]=digest(p)
    assert states["control"]["controller_sources"]==states["candidate"]["controller_sources"]
    for arm in ("control","candidate"):
        frozen_check(states[arm]["controller_sources"])
        assert states[arm]["evaluation_command"]==["bash",str(BASE/"run_fresh_clients.sh"),str(PAIR/arm)]
    envs={arm:normalized(states[arm]['full_environment'],PAIR/arm) for arm in arms}
    assert set(envs['control'])==set(envs['candidate'])
    assert {k for k in envs['control'] if envs['control'][k]!=envs['candidate'][k]}=={'TIANSHU_R5_BM8_MODE'}
    change=100*(arms['candidate']['total_tok_s']/arms['control']['total_tok_s']-1)
    ttft=100*(arms['candidate']['ttft_ms']/arms['control']['ttft_ms']-1)
    formal_gate=change>=1 and ttft<=2 and all(a['spread_percent']<1 for a in arms.values())
    gate=dict(complete=True,CPU_only=True,quality=quality,model_quality_qualified=True,arms=arms,
        paired_total_throughput_change_percent=change,paired_TTFT_change_percent=ttft,
        meets_throughput_spread_and_TTFT_gate=formal_gate,
        exact_model_sources_and_original_commands_verified=True,only_normalized_env_difference='TIANSHU_R5_BM8_MODE',
        both_arms_equal_additional_weight_payload=True,all1024_performance_requests_succeeded=True,
        source_sha256=digest(Path(__file__)),raw_files=raw_hashes,screen_N128=True,official_benchmark=False,adopted=False,
        published_r5_6123_31_not_replaced_by_N128=True)
    with (BASE/'SCREEN_KEY_RUNS_32g.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(scores[0]),lineterminator='\n');writer.writeheader();writer.writerows(scores)
    (BASE/'SCREEN_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k!='raw_files'},indent=2))
if __name__=='__main__':main()
