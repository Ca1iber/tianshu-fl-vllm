# SPDX-License-Identifier: Apache-2.0
"""Independent full-workload gate; previous failed experiments remain immutable."""
import csv,json,statistics,subprocess
from pathlib import Path
from candidate_profile import BASE,read,digest,frozen_check,released_sources
from audit import audit_model
from verify_statistics import verify
ROOT=Path('/dev/shm/tianshu_r5_BM8_capacity_runs_20261024')

def main():
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    assert not (BASE/'OFFICIAL_PAIR_GATE_32g.json').exists()
    qualities={}
    for batch in (8,64):
        root=ROOT/f'math{batch}';s=read(root/'result.json');q=read(root/'accuracy_audit.json')
        assert s['phase']=='complete' and s['accuracy_passed'] and s['total']==105 and s['batch']==batch
        assert q['above95'] and q['fresh_entire_105'] and q['all_caps_in_denominator']
        qualities[str(batch)]=dict(correct=q['correct'],total=105,accuracy=q['accuracy'],root=str(root),result_sha256=digest(root/'result.json'))
    arms={};states={};keyrows=[];files={}
    paths={arm:ROOT/f'official_{arm}' for arm in ('control','candidate')}
    for arm,root in paths.items():
        s=read(root/'result.json');states[arm]=s
        assert s['phase']=='complete' and s['evaluation_exit_code']==0 and s['official']
        assert s['bm8_mode']==arm and s['mode']=='candidate' and s['benchmark_case']==[4096,1024,64,256]
        profile=audit_model(root,s,runtime=True)
        assert json.loads(json.dumps(profile))==read(root/'profile_audit.json')
        assert profile['physical_KV_blocks']==36096 and profile['additional_payload_bytes']==210567168
        stats=verify(root,s,read,digest);assert stats==read(root/'statistics_audit.json')==s['statistics']
        arms[arm]=dict(stats,physical_KV_blocks=profile['physical_KV_blocks'],actual_FULL64_replays=profile['actual_FULL64_replays'])
        for batch in (8,64):
            quality=read(ROOT/f'math{batch}'/'result.json')
            assert quality['frozen_sources']==s['frozen_sources'] and quality['server_command']==s['server_command']
        assert all(s['frozen_sources'][p]==h for p,h in released_sources().items())
        frozen_check(s['controller_sources'])
        with Path(stats['raw_csv']).open(newline='') as f:
            for i,row in enumerate(csv.DictReader(f),1):
                keyrows.append(dict(arm=arm,round=i,excluded=i==1,total_tok_s=float(row['Total tok/s']),output_tok_s=float(row['Output tok/s']),ttft_ms=float(row['Mean TTFT (ms)']),successful=int(float(row['Successful Requests']))))
        for p in (root/'result.json',root/'statistics_audit.json',root/'profile_audit.json',root/'before_metrics.txt',root/'after_metrics.txt',Path(stats['raw_csv']),Path(stats['summary_csv'])):files[str(p)]=digest(p)
    assert states['control']['frozen_sources']==states['candidate']['frozen_sources']
    assert states['control']['controller_sources']==states['candidate']['controller_sources']
    env={arm:{k:v.replace(str(paths[arm]),'<ROOT>') for k,v in states[arm]['full_environment'].items()} for arm in paths}
    assert set(env['control'])==set(env['candidate']) and {k for k in env['control'] if env['control'][k]!=env['candidate'][k]}=={'TIANSHU_R5_BM8_MODE'}
    gain=100*(arms['candidate']['total_tok_s']/arms['control']['total_tok_s']-1)
    latency=100*(arms['candidate']['ttft_ms']/arms['control']['ttft_ms']-1)
    passed=gain>=1 and latency<=2 and all(a['hot_spread_below1_percent'] and a['prefix_below1_percent'] and a['zero_preemptions'] for a in arms.values())
    result=dict(complete=True,all2048_requests_succeeded=True,case=[4096,1024,64,256],preexcluded_rounds=[1],
        arms=arms,qualities=qualities,equal_actual_physical_KV_blocks=36096,
        paired_throughput_change_percent=gain,paired_TTFT_change_percent=latency,
        unchanged_full_workload_gate_passed=passed,original997_exact_subset=True,
        only_normalized_environment_difference='TIANSHU_R5_BM8_MODE',
        previous_failed_gates_not_modified=True,raw_files=files,source_sha256=digest(__file__),adopted=False)
    (BASE/'OFFICIAL_PAIR_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    with (BASE/'OFFICIAL_PAIR_KEY_RUNS_32g.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(keyrows[0]),lineterminator='\n');w.writeheader();w.writerows(keyrows)
    print(json.dumps({k:v for k,v in result.items() if k!='raw_files'},indent=2))
if __name__=='__main__':main()
