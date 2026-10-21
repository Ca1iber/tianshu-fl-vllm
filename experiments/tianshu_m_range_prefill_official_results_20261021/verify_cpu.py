# SPDX-License-Identifier: Apache-2.0
"""Re-execute retained quality/runtime/formal validators on original records."""
import hashlib,json,os,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',VLLM_PLUGINS='',PYTHONPATH='')
BASE=Path(__file__).resolve().parent;VALIDATION=BASE.parent/'tianshu_m_range_prefill_resume_20261021';ROOT=Path('/workspace/benchmark-results/20261021_m_range_prefill_resume_official4k_tianshu')
sys.path.insert(0,str(VALIDATION))
import framework_profile as f
import torch
assert not torch.cuda.is_initialized()
state=f.read(ROOT/'result.json');assert state['phase']=='complete' and state['completed_runs']==4
original=f.digest(ROOT/'result.json');comparison_hash=f.digest(ROOT/'comparison.json')
f.frozen_check(state['frozen_sources']);f.frozen_check(state['controller_sources'])
for key,folder in [('frozen_sources','source_snapshot'),('controller_sources','controller_source_snapshot')]:
 for p,h in state[key].items():assert f.digest(ROOT/folder/Path(p).relative_to('/'))==h
quality=f.read(f.RECOVERY_RAW/'accuracy_audit.json');assert quality['correct']==103 and quality['total']==105
c64=f.read(VALIDATION/'C64_QUALITY_GATE_32g.json');assert c64['complete'] and c64['correct']==101 and c64['actual_M4096_range_fused_all42_verified']
profile=f.audit_runtime(ROOT,state,official=True)
result=f.audit_benchmark(ROOT,state,profile,quality,write=False)
stored=f.read(ROOT/'comparison.json');assert result==stored
assert result['requests_successful']==1024 and result['requests_per_run']==256 and result['excluded_runs']==[1]
assert state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=256,rounds=4,excluded_rounds=[1])
assert len(state['frozen_sources'])==471 and len(state['profile_environment'])==62 and len(state['server_command'])==13
assert f.digest(ROOT/'result.json')==original and f.digest(ROOT/'comparison.json')==comparison_hash and not torch.cuda.is_initialized()
keys=['total_tok_s','output_tok_s','ttft_ms','raw_runs','postfirst_spread_percent','requests_successful','requests_per_run','excluded_runs','passes_public_4k_performance_gates','vs_public_throughput_percent','vs_previous_formal_percent']
gate={k:result[k] for k in keys};gate.update(complete=True,CPU_only=True,CUDA_not_initialized=True,original_experiment_passed=True,all_original_runtime_quality_and_benchmark_validators_executed=True,official_benchmark=True,old_Math8_original_experiment_passed=False,Math8_CPU_reconciliation=True,fresh_official_Math8_correct=103,fresh_C64_Math_correct=101,actual_C64_M4096_new_fused_all42_verified=True,runtime_source_count=471,environment_count=62,original_command_arguments=13,model_frozen_sources=state['frozen_sources'],server_command=state['server_command'],controller_sources=state['controller_sources'],raw_root=str(ROOT),original_result_sha256=original,comparison_sha256=comparison_hash,verifier_source=str(Path(__file__).resolve()),verifier_sha256=f.digest(__file__),prior_formal_total_tok_s=5618.01,prior_formal_ttft_ms=3835.46,vs_previous_formal_ttft_percent=100*(result['ttft_ms']/3835.46-1),promotion_passed=result['passes_public_4k_performance_gates'] and result['postfirst_spread_percent']<1 and result['vs_previous_formal_percent']>=.5,adopted=False,full16K_not_run=True)
(BASE/'OFFICIAL_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
for pattern,name in [('raw_runs_*.csv','RAW_ROUNDS_32g.csv'),('summary_*.csv','SUMMARY_32g.csv')]:
 srcs=list((ROOT/'benchmark_results').glob(pattern));assert len(srcs)==1;(BASE/name).write_text(srcs[0].read_text())
lines=['# M范围融合：正式4K完整结果','','原N256四轮、排第一轮；独立CPU重跑原runtime/quality/benchmark核验，1024请求成功。','', '|轮次|Total tok/s|Output tok/s|TTFT ms|','|---:|---:|---:|---:|']
for i,r in enumerate(result['raw_runs'],1):lines.append(f"|{i}{'（排除）' if i==1 else ''}|{float(r['Total tok/s']):.2f}|{float(r['Output tok/s']):.2f}|{float(r['Mean TTFT (ms)']):.2f}|")
lines += ['',f"热轮均值 **{result['total_tok_s']:.2f} total / {result['output_tok_s']:.2f} output tok/s，TTFT {result['ttft_ms']:.2f}ms**。",f"相较旧正式5618.01，总吞吐{result['vs_previous_formal_percent']:+.4f}%；TTFT相较3835.46ms变化{gate['vs_previous_formal_ttft_percent']:+.4f}%；热轮差幅{result['postfirst_spread_percent']:.5f}%。",f"公开baseline2028.01 tok/s，吞吐{result['vs_public_throughput_percent']:+.3f}%。晋升条件是否满足：{gate['promotion_passed']}。",'','官方C8新题103/105的原GPU实验因新融合0层覆盖而audit_failed，保持原状；独立CPU复核保留准确率与coverage gap。','新C64补充Math101/105=96.1905%，真实M128/768/4096覆盖42层新融合、M64旧融合图实际重放和LM64 INT8全部通过。','471模型源、62配置、13官方serve参数与小测/正确率绑定，控制器分别冻结；旧450源逐字节保持。','23个正收益M使用预选GateUp融合，M256与未知M保持原路径，不新增packed权重。','没有例行跑16K；代码、报告和关键结果上GitHub，完整raw/tensors/回答仅容器。']
(BASE/'REPORT_32g.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k not in ('model_frozen_sources','controller_sources','raw_runs')}))
