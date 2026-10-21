# SPDX-License-Identifier: Apache-2.0
"""Recompute the40 recorded event windows; no GPU or score promotion."""
import hashlib,importlib.util,json,math,os,statistics,sys
from pathlib import Path
os.environ['VLLM_PLUGINS']=''
import torch
BASE=Path(__file__).resolve().parent;SERVICE=BASE.parent/'tianshu_current_stage_events_20261021';ROOT=Path('/workspace/benchmark-results/20261021_current_stage_events_tianshu')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not torch.cuda.is_initialized()
sys.path.insert(0,str(SERVICE));spec=importlib.util.spec_from_file_location('_independent_stage_audit',SERVICE/'audit_profile.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
state=read(ROOT/'result.json');assert state['phase']=='complete' and state['complete'] and not state['official_benchmark'] and not state['adopted']
rechecked=audit.audit_runtime(ROOT,state,official=True);assert rechecked==state['runtime_audit']
stages=read(state['stage_record']);frames=stages['frames'];summary=read(ROOT/'summary.json');assert summary==state['summary'] and len(frames)==40 and stages['frame_count']==40
assert not stages['events_inside_graph'] and stages['boundary_synchronizations']==2 and stages['ixsys_not_used']
series={k:[r['gpu_stream_intervals_ms'][k] for r in frames] for k in ('total','forward','logits','sample')}
for i,r in enumerate(frames):
 assert r['index']==i and r['dispatch']==dict(mode='FULL',padded_tokens=64) and r['context']['requests']==64 and r['context']['min_computed']>=4096
 assert set(r['cpu_call_spans'])=={'execute','sample_tokens','forward','logits','sample'}
 assert all(math.isfinite(v) and v>0 for v in r['gpu_stream_intervals_ms'].values())
 assert r['gpu_stream_intervals_ms']['total']+.1>=sum(r['gpu_stream_intervals_ms'][k] for k in ('forward','logits','sample'))
sums={k:sum(v) for k,v in series.items()};medians={k:statistics.median(v) for k,v in series.items()};fractions={k:sums[k]/sums['total'] for k in ('forward','logits','sample')}
assert sums==summary['gpu_stream_interval_sum_ms'] and medians==summary['gpu_stream_interval_median_ms'] and fractions==summary['fractions_of_total_stream_interval']
host={k:{kind:statistics.median(r['cpu_call_spans'][k][kind] for r in frames) for kind in ('wall_ms','thread_cpu_ms')} for k in ('execute','sample_tokens','forward','logits','sample')};assert host==summary['host_call_median']
assert not torch.cuda.is_initialized()
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,independent_runtime_audit_reexecuted=True,all40_event_windows_recomputed=True,original471_exact_subset=True,actual_environment_count=66,model_source_count=len(state['frozen_sources']),source_map=state['frozen_sources'],seed_canary_preserved=True,all128_timed_load_responses_and_C8_smoke_verified=True,full_qualification_not_claimed=True,no_formal_throughput_or_kernel_active_time_claim=True,GPU_interval_medians_ms=medians,GPU_interval_fractions=fractions,host_call_medians=host,raw_root=str(ROOT),raw_result_sha256=sha(ROOT/'result.json'),raw_stage_record_sha256=sha(state['stage_record']),raw_summary_sha256=sha(ROOT/'summary.json'),verifier_sha256=sha(__file__),adopted=False)
assert not (BASE/'DIAGNOSTIC_GATE_32g.json').exists();(BASE/'DIAGNOSTIC_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
lines=['# 现行5695.95推荐版的阶段事件诊断','', '直接继承原471数值源，13服务参数不变，增加观察源码和4项诊断配置(真实66项环境)。40步实际C64/FULL Graph，输入上下文>=4096；Graph外计GPU事件，窗口边界各同步一次，不逐步同步。128个4K/512响应、8个C8 smoke和前后seed777文本一致，43bank签名、M64MLP与LM路径、M4096全42层融合及0preemption审核重跑通过。','', '|阶段|流区间中位 ms|流区间占比|','|---|---:|---:|']
for k in ('forward','logits','sample'):lines.append(f"|{k}|{medians[k]:.6f}|{100*fractions[k]:.3f}%|")
lines+=['',f"total流区间中位 {medians['total']:.6f}ms。前向仍占约88.48%，所以优先关注attention/MLP；单独词表cast即使变快也不能将组件节省直接外推模型。",'', '这些区间可包含host提交间隙；host wall可含等待，thread CPU不含阻塞。父子区间不能相加当纯GPU活动，不是单kernel计时，也不是正式吞吐。诊断审核不替代新模型质量资格审核。完整40帧、响应与日志只在容器；本报告和关键门槛在GitHub。']
(BASE/'REPORT_32g.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='source_map'},indent=2))
