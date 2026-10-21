# SPDX-License-Identifier: Apache-2.0
"""Independent CPU structural/source/route validation of real fusion integration."""
import argparse,json,os
os.environ['VLLM_PLUGINS']=''
from pathlib import Path
import torch
import sys
RECOVERY=Path(__file__).resolve().parent
sys.path.insert(0,str(RECOVERY.parent/'tianshu_nn_silu_fx_service_20261021'))
from candidate_profile import BASE,read,digest,frozen_check
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
assert not torch.cuda.is_initialized();state=read(root/'result.json');assert state['phase']=='failed' and state['error']=='AssertionError()' and state['mode']=='candidate'
assert "line 60, in main" in state['traceback'] and "log.split('Application startup complete.'" in state['traceback']
assert len(state['profile_environment'])==62 and len(state['server_command'])==13
for path,h in state['frozen_sources'].items():assert digest(path)==h==digest(root/'source_snapshot'/Path(path).relative_to('/'))
frozen_check(state['controller_sources'])
original=read(BASE.parent/'tianshu_prefill_int8_official_results_20261021/RUNTIME_BINDING_GATE_32g.json')['frozen_sources'];assert len(original)==407 and all(state['frozen_sources'][p]==h for p,h in original.items())
observed=read(root/'observed_server_environment.json');assert observed['command']==state['server_command'] and observed['environment']==state['profile_environment']
responses=state['requests'];assert len(responses)==3 and len(state['wave'])==64
assert all(r['response']['choices'][0]['token_ids']==responses[0]['response']['choices'][0]['token_ids'] for r in responses)
assert len({r['payload']['cache_salt'] for r in [*responses,*state['wave']]})==67
assert len({tuple(r['payload']['prompt']) for r in state['wave']})==64
fusion=read(next((root/'nn_silu_fx_records').glob('route_*.json')));assert fusion['completed'] and fusion['all_42_M64_real_capture_routes_verified'] and fusion['warm_before_capture_RNG_unchanged']
assert fusion['no_additional_packed_weight_payload'] and fusion['M8_decode_unchanged'] and fusion['dynamic_prefill_graph_unchanged']
for name in ('policy','kernel','quantizer'):assert fusion[name+'_sha256']==state['frozen_sources'][fusion[name+'_source']]
pass_rows=[read(p) for p in (root/'nn_silu_fx_records/fx_pass').glob('pass_*.json')]
assert len(pass_rows)==86
static=[r for r in pass_rows if r['compile_range']==[64,64]];dynamic=[r for r in pass_rows if r['compile_range']==[1,4096]]
assert len(static)==len(dynamic)==43 and sorted(i for r in static for i in r['bank_ids'])==list(range(42))
assert sum(r['replaced'] for r in static)==42 and all(r['replaced']==0 and r['before_sha256']==r['after_sha256'] and r['no_other_ranges_changed'] for r in dynamic)
nn=read(next((root/'int8_nn_gateup_records').glob('route_*.json')));assert nn['completed'] and len(nn['banks'])==42
assert len({r['prefix'] for r in nn['routes'].values() if r['M']==8 and r['actually_inside_capture']})==42
assert nn['M64_capture_validator_sha256']==state['frozen_sources'][nn['M64_capture_validator_source']]
assert not [r for r in nn['routes'].values() if r['M']==64 and r['actually_inside_capture']]
dense=read(next((root/'dense_int8_records').glob('bindings_*.json')));assert dense['load_complete'] and dense['actual_packed_banks']==43 and dense['packed_payload_bytes']==1326938112
reference={r['prefix']:r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')}
for bank in dense['banks']:
    expected=reference[bank['prefix']]
    assert all(bank[k]==expected[k] for k in ('shape','original_BF16_sha256','packed_sha256','scales_sha256','packed_bytes'))
trace=read(next((root/'graph_segments_records').glob('trace_*.json')))
routes=[r for r in fusion['routes'].values() if r['M']==64 and r['actually_inside_capture']];assert len({r['prefix'] for r in routes})==42
qualified=set();used_graphs={}
for r in routes:
    assert r['implementation']=='NN_two_dot_fused_SiLU' and r['configuration']==fusion['configuration'] and r['calls']>0
    assert r['kernel_sha256']==state['frozen_sources'][r['kernel_source']] and r['graph_key'] is not None
    graph=trace['graphs'][r['graph_key']]
    if graph['capture_calls']>=1 and graph['replay_calls']>=32 and any(int(n)==64 and f['max_computed']>=128 for n,f in graph['frames'].items()):
        qualified.add(r['prefix']);used_graphs[r['graph_key']]=dict(replays=graph['replay_calls'],capture_calls=graph['capture_calls'],frames=graph['frames'])
assert len(qualified)==42,(len(qualified),used_graphs)
prefill=read(next((root/'prefill_int8_records').glob('route_*.json')));assert prefill['completed'] and prefill['stage']=='active' and prefill['warm_CPU_GPU_RNG_unchanged']
assert len(prefill['banks'])==42 and len(prefill['warmups'])==24 and prefill['kernel_sha256']==state['frozen_sources'][prefill['kernel_source']]
assert len({r['prefix'] for r in prefill['routes'].values() if r['stage']=='active' and not r['actually_inside_capture'] and r['implementation']=='prefill_A8W8'})==42
log=(root/'server.log').read_text(errors='replace');assert log.count('Application startup complete.')==1
late=[line for line in log.split('Application startup complete.',1)[1].splitlines() if 'Triton autotuning for function' in line]
assert len(late)==2 and all('_index_linearized_jit_function' in line for line in late)
assert {line.split('config key: ',1)[1].split(', benchmark key:',1)[0] for line in late}=={"(32, 'torch.int32', 'torch.int64', 'torch.int32')","(4032, 'torch.int32', 'torch.int64', 'torch.int32')"}
assert 'No running processes found' in (root.parent/'20261021_nn_silu_fx_service_candidate_runner_tianshu/hardware_after.txt').read_text()
assert not torch.cuda.is_initialized()
gate=dict(complete=True,diagnostic_integration_only=True,adopted=False,benchmark_not_run=True,model_quality_not_evaluated=True,original407_sources_exact_subset=True,environment62_and_argv13_observed=True,all_43_checkpoint_and_packed_signatures_preserved=True,all_42_real_M64_fused_captures_and_replays_verified=True,all_M8_original_NN_captures_retained=True,all_43_dynamic_graphs_unmodified=True,replaced_M64_GateUp_nodes=42,prefill24_warm_shapes_and_all42_actual_request_routes_retained=True,cold_C1_tokens_identical_before_after_C64=True,original_experiment_passed=False,original_failure_retained=True,post_ready_autotune=len(late),late_autotune_records=late,fix_required_before_full_screen='Startup prewarm exact index32 in a new experiment',all_capture_binaries_equal_component=all(r['component_binary_exact_match'] for r in routes),actual_compiled_metadata=[{k:r[k] for k in ('prefix','configuration','binary_sha256','n_regs','n_spills','shared')} for r in routes],actual_graphs=used_graphs,raw_root=str(root),raw_result_sha256=digest(root/'result.json'),frozen_sources=state['frozen_sources'],verifier_sha256=digest(__file__))
(RECOVERY/'INTEGRATION_RECOVERY_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('frozen_sources','actual_compiled_metadata')}))
