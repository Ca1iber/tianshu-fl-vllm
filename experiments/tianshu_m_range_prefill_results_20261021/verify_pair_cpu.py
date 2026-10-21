# SPDX-License-Identifier: Apache-2.0
"""Independently recheck same-source 4K service trials; never invokes GPU."""
import csv,hashlib,importlib.util,json,math,os,statistics,subprocess
from pathlib import Path
os.environ['VLLM_PLUGINS']=''
import torch
BASE=Path(__file__).resolve().parent
SERVICE=BASE.parent/'tianshu_m_range_prefill_service_20261021'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not torch.cuda.is_initialized()
audit=load('_range_pair_independent_audit',SERVICE/'audit_profile.py')
runner=Path('/workspace/benchmark-results/20261021_m_range_prefill_pair_runner_tianshu')
assert (runner/'phase.txt').read_text().strip()=='GPU_pair_complete_pending_CPU_comparison'
subprocess.run(['sha256sum','--check','--quiet',str(runner/'frozen_entrypoints.sha256')],check=True,cwd='/workspace/vllm-plugin-FL')
original=read(BASE.parent/'tianshu_nn_silu_fx_c64_quality_20261021/C64_QUALITY_GATE_32g.json')['model_frozen_sources'];assert len(original)==450
states={};evidence={};rows=[];envs={};shared_seeds=None
for arm in ('control','candidate'):
 root=Path('/workspace/benchmark-results')/f'20261021_m_range_prefill_pair_{arm}_tianshu'
 state=read(root/'result.json');states[arm]=state
 assert state['phase']=='complete' and len(state['screen_rounds'])==3
 assert state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=64,rounds=3,excluded_rounds=[1])
 assert len(state['profile_environment'])==62 and len(state['server_command'])==13
 for p,h in state['frozen_sources'].items():assert sha(p)==h==sha(root/'source_snapshot'/Path(p).relative_to('/'))
 assert all(state['frozen_sources'][p]==h for p,h in original.items())
 normalized={k:v.replace(str(root),'__ARM_OUTPUT_ROOT__') for k,v in state['profile_environment'].items()};envs[arm]=normalized
 live=audit.audit_runtime(root,state,official=True)
 assert live['mode']==arm and live['qualified450_same_in_both_arms'] and live['original450_source_map_exact_subset'] and live['full_inherited_attention_KV_scheduler_weights_LM_checks_retained']
 assert live['native_dynamic_graphs_preserved']==(arm=='control')
 assert live['range_prefill_record']==state['profile_audit']['range_prefill_record']
 seeds=[]
 for index,r in enumerate(state['screen_rounds']):
  assert r['round']==index+1 and r['successful_requests']==64 and len(r['responses'])==64 and not r['tuning_records']
  duration=r['duration_seconds'];assert math.isfinite(duration) and duration>0
  assert math.isclose(r['total_tokens_per_second'],64*5120/duration,rel_tol=1e-10)
  assert math.isclose(r['output_tokens_per_second'],64*1024/duration,rel_tol=1e-10)
  assert math.isclose(r['mean_ttft_seconds'],statistics.mean(a['ttft_seconds'] for a in r['responses']),rel_tol=1e-10)
  for a in r['responses']:
   assert a['usage']['prompt_tokens']==4096 and a['usage']['completion_tokens']==1024 and a['usage']['total_tokens']==5120
   assert a['input_tokens']==4096 and a['output_tokens']==1024 and a['text'] and 0<a['ttft_seconds']<=a['wall_seconds']
  seeds.append([a['input_seed'] for a in r['responses']])
  rows.append(dict(arm=arm,round=index+1,excluded=index==0,duration_seconds=duration,total_tokens_per_second=r['total_tokens_per_second'],output_tokens_per_second=r['output_tokens_per_second'],mean_TTFT_ms=1000*r['mean_ttft_seconds'],successful_requests=64,tuning_records=0))
 if shared_seeds is None:shared_seeds=seeds
 else:assert seeds==shared_seeds
 hot=[r for r in rows if r['arm']==arm and not r['excluded']]
 total=statistics.mean(r['total_tokens_per_second'] for r in hot);output=statistics.mean(r['output_tokens_per_second'] for r in hot);ttft=statistics.mean(r['mean_TTFT_ms'] for r in hot)
 spread=100*(max(r['total_tokens_per_second'] for r in hot)-min(r['total_tokens_per_second'] for r in hot))/total
 assert math.isclose(state['mean_total_tok_s'],total,rel_tol=1e-10) and math.isclose(state['spread_percent'],spread,abs_tol=1e-10)
 evidence[arm]=dict(raw_root=str(root),raw_result_sha256=sha(root/'result.json'),source_count=len(state['frozen_sources']),total_tok_s=total,output_tok_s=output,mean_TTFT_ms=ttft,spread_percent=spread,successful_requests=192,post_ready_autotune=0,full_runtime_audit_rechecked=True)
assert states['control']['frozen_sources']==states['candidate']['frozen_sources']
assert states['control']['server_command']==states['candidate']['server_command']
assert [k for k in envs['control'] if envs['control'][k]!=envs['candidate'][k]]==['PYTHONPATH']
ratio=100*(evidence['candidate']['total_tok_s']/evidence['control']['total_tok_s']-1)
ttft_ratio=100*(evidence['candidate']['mean_TTFT_ms']/evidence['control']['mean_TTFT_ms']-1)
stable=all(a['spread_percent']<1 for a in evidence.values())
assert not torch.cuda.is_initialized()
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,adopted=False,official_benchmark=False,new_official_and_C64_quality_required=True,all_384_requests_successful=True,qualified450_exact_subset_both_arms=True,same_frozen_sources=True,same_request_seed_workloads=True,only_normalized_environment_difference='PYTHONPATH',post_ready_autotune=0,all_runtime_audits_rechecked=True,excluded_first_round_each_arm=True,throughput_improvement_percent=ratio,TTFT_change_percent=ttft_ratio,all_spreads_below1_percent=stable,candidate_passes_perf_screen=(ratio>.5 and stable),arms=evidence,rounds=rows,verifier_source_sha256=sha(__file__),frozen_sources=states['candidate']['frozen_sources'])
assert not (BASE/'SCREEN_GATE_32g.json').exists()
(BASE/'SCREEN_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
with (BASE/'KEY_ROUNDS_32g.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# 逐M GateUp融合：同源完整服务4K短测','', '两臂保留正式5618.01版本的原450源码；仅分段GateUp融合不同。4K/1K，C64/N64，各三轮排首轮；不能作为正式N256成绩。完整继承运行时审核并独立重算384个响应的tokens、TTFT、吞吐与输入种子，实际kernel binary与完整42bank组件严格绑定。','', '|模式|总tok/s|输出tok/s|TTFT ms|后两轮差幅|','|---|---:|---:|---:|---:|']
for arm,e in evidence.items():lines.append(f"|{arm}|{e['total_tok_s']:.2f}|{e['output_tok_s']:.2f}|{e['mean_TTFT_ms']:.2f}|{e['spread_percent']:.4f}%|")
lines+=['',f'候选吞吐变化 {ratio:+.4f}%；TTFT变化 {ttft_ratio:+.4f}%。性能筛选通过：{gate["candidate_passes_perf_screen"]}。正式最好成绩继续引用已通过官方Math及C64质量的5618.01，候选必须独立通过新质量评测及正式4K。','', '所有请求成功，就绪后无autotune；完整旧KV/attention/scheduler/权重/LM检查保留，原M64已采用融合两臂相同。动态[1,4096]图在候选显式改写；不能声称所有动态图不变。源文件与原始结果SHA记录在SCREEN_GATE；大体积响应和原始日志只留容器。']
(BASE/'REPORT_32g.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k not in ('rounds','frozen_sources')}))
