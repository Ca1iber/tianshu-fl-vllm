# SPDX-License-Identifier: Apache-2.0
"""One complete official paired workload; preserve the earlier failed short gate."""
import csv,json,statistics,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
MODEL=BASE.parent/'tianshu_r5_BM8_KV_20261024'
sys.path.insert(0,str(MODEL))
from candidate_profile import read,digest,frozen_check
from audit import audit_model
from verify_statistics import verify
from screen_prerequisite import check
META=Path('/dev/shm/tianshu_BM8_official4k_pair_20261024')
ROOT=Path('/dev/shm/tianshu_official_followup_20261024')
PATHS=dict(control=ROOT/'4k_r5_control_bm8_inactive',candidate=ROOT/'4k_bm8')

def main():
    assert (META/'phase.txt').read_text().strip()=='both_official4K_complete_pending_comparison'
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    original=check();assert not (BASE/'OFFICIAL_PAIR_GATE_32g.json').exists()
    states={};arms={};raw={};keyrows=[]
    for arm,root in PATHS.items():
        s=read(root/'result.json');states[arm]=s
        assert s['phase']=='complete' and s['evaluation_exit_code']==0 and s['official'] and s['stage']=='official'
        assert s['mode']=='candidate' and s['bm8_mode']==arm and s['benchmark_case']==[4096,1024,64,256]
        assert s['frozen_sources']==original['model_sources'] and s['server_command']==original['server_command']
        assert len(s['frozen_sources'])==1023 and len(s['full_environment'])==67
        profile=audit_model(root,s,runtime=True)
        assert json.loads(json.dumps(profile))==read(root/'profile_audit.json')
        assert profile['additional_payload_bytes']==210567168 and profile['QKVO64_capture_banks']==84
        stats=verify(root,s,read,digest);assert stats==read(root/'statistics_audit.json')==s['statistics']
        arms[arm]=dict(stats,physical_KV_blocks=profile['physical_KV_blocks'],actual_FULL64_replays=profile['actual_FULL64_replays'])
        with Path(stats['raw_csv']).open(newline='') as f:
            for i,row in enumerate(csv.DictReader(f),1):keyrows.append(dict(arm=arm,round=i,excluded=i==1,total_tok_s=float(row['Total tok/s']),output_tok_s=float(row['Output tok/s']),ttft_ms=float(row['Mean TTFT (ms)']),successful=int(float(row['Successful Requests']))))
        for p in (root/'result.json',root/'statistics_audit.json',root/'profile_audit.json',root/'before_metrics.txt',root/'after_metrics.txt',Path(stats['raw_csv']),Path(stats['summary_csv'])):raw[str(p)]=digest(p)
        frozen_check(s['controller_sources'])
    assert states['control']['controller_sources']==states['candidate']['controller_sources']
    env={arm:{k:v.replace(str(PATHS[arm]),'<ROOT>') for k,v in states[arm]['full_environment'].items()} for arm in PATHS}
    assert set(env['control'])==set(env['candidate']) and {k for k in env['control'] if env['control'][k]!=env['candidate'][k]}=={'TIANSHU_R5_BM8_MODE'}
    assert arms['control']['physical_KV_blocks']==arms['candidate']['physical_KV_blocks']
    gain=100*(arms['candidate']['total_tok_s']/arms['control']['total_tok_s']-1)
    latency=100*(arms['candidate']['ttft_ms']/arms['control']['ttft_ms']-1)
    passed=gain>=1 and latency<=2 and all(a['hot_spread_below1_percent'] and a['prefix_below1_percent'] and a['zero_preemptions'] for a in arms.values())
    result=dict(complete=True,official_original_case=[4096,1024,64,256],all2048_requests_succeeded=True,
        original_N128_gate_remains_false=True,original_gate_sha256=original['original_screen_gate_sha256'],
        arms=arms,paired_throughput_change_percent=gain,paired_TTFT_change_percent=latency,
        unchanged_full_workload_gate_passed=passed,selected_profile_for_next16K='bm8' if passed else 'r5',
        same_full_sources_payload_original_argv=True,only_normalized_environment_difference='TIANSHU_R5_BM8_MODE',
        raw_files=raw,source_sha256=digest(Path(__file__)),adopted=False)
    with (BASE/'OFFICIAL_PAIR_KEY_RUNS_32g.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(keyrows[0]),lineterminator='\n');w.writeheader();w.writerows(keyrows)
    (BASE/'OFFICIAL_PAIR_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='raw_files'},indent=2))
if __name__=='__main__':main()
