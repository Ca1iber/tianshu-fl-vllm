# SPDX-License-Identifier: Apache-2.0
"""Independently audit direct471 reference and declared QKV source extension."""
import csv,hashlib,importlib.util,json,math,os,statistics,subprocess
from pathlib import Path
os.environ['VLLM_PLUGINS']=''
import torch
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;SERVICE=EXP/'tianshu_k8v16_bm8_service_v3_20261021'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
assert not torch.cuda.is_initialized()
runner=Path('/workspace/benchmark-results/20261021_BM8_candidate_v3_runner_tianshu')
assert (runner/'phase.txt').read_text().strip()=='GPU_complete_pending_CPU_comparison'
subprocess.run(['sha256sum','--check','--quiet',str(runner/'frozen_entrypoints.sha256')],check=True,cwd='/workspace/vllm-plugin-FL')
original=read(EXP/'tianshu_m_range_prefill_official_results_20261021/OFFICIAL_GATE_32g.json')['model_frozen_sources'];assert len(original)==471
old_audit=load('_QKV_direct_reference_reaudit',EXP/'tianshu_m_range_prefill_service_v2_20261021/audit_profile.py')
new_audit=load('_QKV_declared_extension_reaudit',SERVICE/'audit_profile.py')
roots={arm:Path('/workspace/benchmark-results')/name for arm,name in [('reference','20261021_BM8_direct_reference_tianshu'),('candidate','20261021_BM8_candidate_v3_tianshu')]}
states={};evidence={};rows=[];envs={};shared_seeds=None
for arm,root in roots.items():
 state=read(root/'result.json');states[arm]=state
 assert state['phase']=='complete' and len(state['screen_rounds'])==3 and state['mode']=='candidate'
 assert state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1])
 assert len(state['profile_environment'])==62 and len(state['server_command'])==13
 for p,h in state['frozen_sources'].items():assert sha(p)==h==sha(root/'source_snapshot'/Path(p).relative_to('/'))
 assert all(state['frozen_sources'][p]==h for p,h in original.items())
 if arm=='reference':assert state['frozen_sources']==original
 else:assert state['frozen_sources']==read(SERVICE/'CPU_GATE_32g.json')['sources']
 envs[arm]={k:v.replace(str(root),'__ARM_OUTPUT_ROOT__') for k,v in state['profile_environment'].items()}
 live=(old_audit if arm=='reference' else new_audit).audit_runtime(root,state,official=True)
 assert live['full_inherited_attention_KV_scheduler_weights_LM_checks_retained'] and live['zero_preemptions']
 if arm=='candidate':
  assert live['BM8_actual_C64_request_graph_replay_verified'] and live['original471_exact_subset']
  assert live['BM8_record']==state['profile_audit']['BM8_record']
 assert len(state['seeded_canaries'])==2 and state['seeded_canaries'][0]['text']==state['seeded_canaries'][1]['text']
 assert len(state['int8_smoke_wave'])==8
 seeds=[]
 for index,r in enumerate(state['screen_rounds']):
  assert r['round']==index+1 and r['successful_requests']==64 and len(r['responses'])==64 and not r['tuning_records']
  d=r['duration_seconds'];assert math.isfinite(d) and d>0
  assert math.isclose(r['total_tokens_per_second'],64*5120/d,rel_tol=1e-10)
  assert math.isclose(r['output_tokens_per_second'],64*1024/d,rel_tol=1e-10)
  assert math.isclose(r['mean_ttft_seconds'],statistics.mean(a['ttft_seconds'] for a in r['responses']),rel_tol=1e-10)
  for a in r['responses']:
   assert a['usage']==dict(prompt_tokens=4096,completion_tokens=1024,total_tokens=5120)
   assert a['input_tokens']==4096 and a['output_tokens']==1024 and a['text'] and 0<a['ttft_seconds']<=a['wall_seconds']
  seeds.append([a['input_seed'] for a in r['responses']])
  rows.append(dict(arm=arm,round=index+1,excluded=index==0,duration_seconds=d,total_tokens_per_second=r['total_tokens_per_second'],output_tokens_per_second=r['output_tokens_per_second'],mean_TTFT_ms=1000*r['mean_ttft_seconds'],successful_requests=64,tuning_records=0))
 if shared_seeds is None:shared_seeds=seeds
 else:assert seeds==shared_seeds
 hot=[r for r in rows if r['arm']==arm and not r['excluded']];total=statistics.mean(r['total_tokens_per_second'] for r in hot);output=statistics.mean(r['output_tokens_per_second'] for r in hot);ttft=statistics.mean(r['mean_TTFT_ms'] for r in hot)
 spread=100*(max(r['total_tokens_per_second'] for r in hot)-min(r['total_tokens_per_second'] for r in hot))/total
 assert math.isclose(state['mean_total_tok_s'],total,rel_tol=1e-10) and math.isclose(state['spread_percent'],spread,abs_tol=1e-10)
 evidence[arm]=dict(raw_root=str(root),raw_result_sha256=sha(root/'result.json'),source_count=len(state['frozen_sources']),total_tok_s=total,output_tok_s=output,mean_TTFT_ms=ttft,spread_percent=spread,successful_requests=192,post_ready_autotune=0,model_memory_usage_bytes=live['dense_int8_record']['model_memory_usage_bytes'],actual_KV_blocks=live['actual_KV_blocks'],full_runtime_audit_rechecked=True)
assert states['reference']['frozen_sources']!=states['candidate']['frozen_sources']
assert states['reference']['server_command']==states['candidate']['server_command']
assert sorted(k for k in envs['reference'] if envs['reference'][k]!=envs['candidate'][k])==['PYTHONPATH']
assert sha(SERVICE/'run_service_trial.py')==sha(EXP/'tianshu_m_range_prefill_service_v2_20261021/run_service_trial.py')
ratio=100*(evidence['candidate']['total_tok_s']/evidence['reference']['total_tok_s']-1);ttft_ratio=100*(evidence['candidate']['mean_TTFT_ms']/evidence['reference']['mean_TTFT_ms']-1);stable=all(a['spread_percent']<1 for a in evidence.values())
assert not torch.cuda.is_initialized()
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,adopted=False,official_benchmark=False,new_official_Math_and_actual_C64_required=True,all_384_requests_successful=True,original471_exact_subset_both_arms=True,different_source_graphs_explicitly_declared=True,client_source_byte_identical=True,same_request_seed_workloads=True,only_normalized_environment_difference='PYTHONPATH',post_ready_autotune=0,all_runtime_audits_rechecked=True,BM8_capture_and_actual_request_graph_replay_verified=True,excluded_first_round_each_arm=True,throughput_improvement_percent=ratio,TTFT_change_percent=ttft_ratio,all_spreads_below1_percent=stable,candidate_passes_perf_screen=(ratio>.5 and stable),arms=evidence,rounds=rows,verifier_source_sha256=sha(__file__),candidate_frozen_sources=states['candidate']['frozen_sources'])
assert not (BASE/'SCREEN_GATE_32g.json').exists();(BASE/'SCREEN_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
with (BASE/'KEY_ROUNDS_32g.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# BM8+直接映射M64：现行版本直接对照','', '4K/1K、C64/N64三轮排首轮；直接reference不加载新QKV hook。candidate声明新增attention源，无新增packed权重，原471源逐字节保留。客户端源码、输入种子、13服务参数相同，62配置归一化后仅PYTHONPATH不同。候选仅3D/M64改BM8+直接序列映射，组件112输出全逐位相同，所有42层实捕获并绑定实际C64请求Graph replay；其他批次及2D原JIT完整委托。旧pair记录为子选择器前的源/参数，最终源/grid/二进制由新BM8记录审核。','', '|模式|总tok/s|输出tok/s|TTFT ms|后两轮差幅|KV blocks|','|---|---:|---:|---:|---:|---:|']
for arm,e in evidence.items():lines.append(f"|{arm}|{e['total_tok_s']:.2f}|{e['output_tok_s']:.2f}|{e['mean_TTFT_ms']:.2f}|{e['spread_percent']:.4f}%|{e['actual_KV_blocks']}|")
lines+=['',f'整模型吞吐变化 {ratio:+.4f}%；TTFT变化 {ttft_ratio:+.4f}%；预设>.5%且差幅<1%性能门槛通过：{gate["candidate_passes_perf_screen"]}。', '', '本短测不能替代官方N256成绩。正式推荐版仍为5695.95 total tok/s；新候选需要重新通过官方C8 Math及实际C64新路径Math，然后正式4K。全384响应由CPU复算tokens、吞吐、TTFT和输入种子，0preemption/0就绪后autotune。完整响应/日志/张量只在容器，GitHub保留代码、报告和关键证据。']
(BASE/'REPORT_32g.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('rounds','candidate_frozen_sources')},indent=2))
