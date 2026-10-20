# SPDX-License-Identifier: Apache-2.0
"""Check real-service startup, full bank identity and same C64 N64 workload."""
import csv,hashlib,json,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(e,root):return {k:v.replace(str(root)+'/', '<OUTPUT>/') for k,v in e.items() if k!='TIANSHU_COLD_START_MODE'}
def main():
    queue=Path('/workspace/benchmark-results/20261020_cold_start_v2_comparison_runner_tianshu')
    assert (queue/'phase.txt').read_text().strip()=='complete'
    signature={r['prefix']:r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')}
    sources=None;environment=None;command=None;summary=[];rounds=[];raw={}
    for mode in ('native','cpu4','gpu4'):
        root=Path(f'/workspace/benchmark-results/20261020_cold_start_v2_service_{mode}_tianshu');state=read(root/'result.json');raw[mode]=str(root)
        assert state['phase']=='complete' and state['mode']==mode and len(state['profile_environment'])==59
        current=state['frozen_sources'];assert len(current)==329
        if sources is None:sources=current;environment=norm(state['profile_environment'],root);command=state['server_command']
        assert sources==current and norm(state['profile_environment'],root)==environment and state['server_command']==command
        assert read(root/'observed_server_environment.json')['environment']==state['profile_environment']
        for path,digest in current.items():assert sha(path)==sha(root/'source_snapshot'/Path(path).relative_to('/'))==digest
        assert state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1])
        bindings=list((root/'dense_int8_records').glob('bindings_*.json'));assert len(bindings)==1
        dense=read(bindings[0]);assert len(dense['banks'])==43 and dense['load_complete'] and dense['packed_payload_bytes']==1326938112
        assert dense['cpu_threads_before_load']==dense['cpu_threads_after_load']
        actual_threads=dense['cpu_threads_during_load'];assert actual_threads==(dense['cpu_threads_before_load'] if mode=='native' else 4)
        for bank in dense['banks']:
            expected=signature[bank['prefix']]
            assert bank['packing_CPU_threads']==actual_threads and bank['packing_method']==('GPU' if mode=='gpu4' else 'CPU')
            assert all(bank[key]==expected[key] for key in ('shape','original_BF16_sha256','packed_sha256','scales_sha256','packed_bytes'))
        audit=state['profile_audit'];assert audit['all_43_banks_same_as_qualified_Math'] and audit['CPU_threads_restored'] and audit['already_adopted_reflect_one_actually_executed']
        assert audit['zero_preemptions'] and audit['future_reservation']['outstanding_blocks']==0
        assert len(state['screen_rounds'])==3
        for r in state['screen_rounds']:
            assert r['successful_requests']==len(r['responses'])==64
            assert abs(r['total_tokens_per_second']-64*5120/r['duration_seconds'])<1.e-8
            assert all(v['usage']['prompt_tokens']==4096 and v['usage']['completion_tokens']==1024 and v['usage']['total_tokens']==5120 for v in r['responses'])
            rounds.append(dict(mode=mode,round=r['round'],total_tok_s=r['total_tokens_per_second'],output_tok_s=r['output_tokens_per_second'],ttft_ms=r['mean_ttft_seconds']*1000,successful_requests=64,excluded=r['round']==1))
        post=state['screen_rounds'][1:];values=[r['total_tokens_per_second'] for r in post]
        assert not any(r['tuning_records'] for r in post)
        mean=statistics.mean(values);spread=100*(max(values)-min(values))/mean
        assert abs(mean-state['mean_total_tok_s'])<1.e-8 and abs(spread-state['spread_percent'])<1.e-8
        summary.append(dict(mode=mode,CPU_threads=actual_threads,server_ready_seconds=state['server_ready_seconds'],loader_seconds=dense['load_model_seconds'],
            packing_seconds=sum(r['packing_seconds'] for r in dense['banks']),mean_total_tok_s=mean,mean_ttft_ms=statistics.mean(r['mean_ttft_seconds'] for r in post)*1000,
            postfirst_spread_percent=spread,actual_KV_blocks=audit['actual_KV_blocks'],requests_successful=192,
            result_json_sha256=sha(root/'result.json'),binding_sha256=sha(bindings[0])))
    result=dict(complete=True,independently_rechecked=True,official_benchmark=False,adopted_as_new_formal_score=False,new_Math_not_run=True,
        all_129_full_bank_signatures_match_previously_qualified_Math=True,original_BF16_GPU_weights_retained=True,all_CPU_thread_counts_restored=True,
        identical_runtime_sources_across_all_three=True,runtime_source_count=329,environment_count=59,original_serve_command=command,
        same_already_adopted_target_table_SQL_in_all_modes=True,zero_preemptions=True,all_future_reservations_released=True,
        all_576_requests_successful=True,excluded_rounds=[1],postfirst_no_runtime_autotuning=True,
        summary=summary,raw_roots=raw,frozen_sources=sources,
        cpu4_loader_reduction_percent=100*(1-summary[1]['loader_seconds']/summary[0]['loader_seconds']),
        gpu4_loader_reduction_percent=100*(1-summary[2]['loader_seconds']/summary[0]['loader_seconds']),
        cpu4_ready_reduction_percent=100*(1-summary[1]['server_ready_seconds']/summary[0]['server_ready_seconds']),
        gpu4_ready_reduction_percent=100*(1-summary[2]['server_ready_seconds']/summary[0]['server_ready_seconds']),
        scope='Three single cold starts plus same small service screen; not a repeated cold-start statistical distribution or new formal/model quality result.')
    (BASE/'COLD_START_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('LOADERS_32g.csv',summary),('ROUNDS_32g.csv',rounds)]:
        with (BASE/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ('frozen_sources','raw_roots')}))
if __name__=='__main__':main()
