# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;SHORT=EXP/'tianshu_LM_pad64_short4k_20261023';HOLDOUT=EXP/'tianshu_native_int8_attention64_20261022'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
short=load('_LM_pad64_official_short',SHORT/'candidate_profile.py');full=load('_LM_pad64_official_all_runtime',SHORT/'audit_profile.py');quality=load('_LM_pad64_official_fresh_quality',EXP/'tianshu_LM_pad64_quality_20261022/quality_profile.py');digest,read,frozen_check=short.digest,short.read,short.frozen_check;BENCH_SHA,csv_rows,normalized_environment=quality.BENCH_SHA,quality.csv_rows,quality.normalized_environment;audit_startup=quality.audit_startup;original=short.normalizer

def audit_math(*args,**kwargs):raise AssertionError('QKV64 official-only controller')
def screen_gate():
 gate=read(SHORT/'SCREEN_GATE_prefill_32g.json');assert gate['complete'] and gate['candidate_passes_perf_screen'] and gate['vs_exact_r3_short_gain_percent']>.5 and gate['arms']['candidate']['hot_spread_pct']<1
 for arm in ('control','candidate'):
  root=Path(gate['arms'][arm]['raw_root']);assert digest(root/'result.json')==gate['arms'][arm]['raw_result_sha256'];s=read(root/'result.json');assert s['phase']=='complete';frozen_check(s['frozen_sources']);frozen_check(s['controller_sources'])
 return gate

def prepare(root,screen_root,batch=8,scope='prefill',check_screen=True,check_quality=True):
 assert batch==8 and scope=='prefill' and screen_root.resolve()==HOLDOUT.resolve()
 if check_screen:screen_gate()
 command,environment,profile,sources=short.prepare(root,'candidate',scope,check_quality=check_quality);assert len(sources)==925
 if check_quality:short.quality_gate(scope,sources)
 if check_screen:assert sources==read(Path(screen_gate()['arms']['candidate']['raw_root'])/'result.json')['frozen_sources']
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run.sh'}
 if check_screen:paths.add(BASE/'EXECUTION.sha256')
 if check_screen:paths.update((SHORT/'SCREEN_GATE_prefill_32g.json',SHORT/'KEY_RUNS_prefill_32g.csv'))
 for batch_id in (8,64):
  raw=(Path('/workspace/benchmark-results/20261022_LM_pad64_math64_CPU_recovery_tianshu') if batch_id==64 else Path('/workspace/benchmark-results/20261022_LM_pad64_math8_tianshu'));items=[raw/'result.json',raw/'accuracy_audit.json',raw/'controller_sources.json']
  if check_quality or all(p.exists() for p in items):paths.update(items)
 controllers=read(root/'controller_sources.json');controllers.update({str(p):digest(p) for p in paths})
 for m in tuple(sys.modules.values()):
  path=getattr(m,'__file__',None)
  if path and path.endswith('.py') and Path(path).is_relative_to(EXP):controllers[str(Path(path).resolve())]=digest(path)
 (root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,environment,profile,sources

def audit_runtime(root,state,official=False):
 assert official and state['stage']=='official' and state['attention_mode']=='candidate' and state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=256,rounds=4,excluded_rounds=[1]);return full.audit_runtime(root,state,official=True)

def audit_benchmark(root,state,profile,c8,write=True):
 gate=short.quality_gate('prefill',state['frozen_sources']);screen=screen_gate();r64=Path(state['math64_root']);q64=read(r64/'accuracy_audit.json');assert c8['correct']==gate['C8_correct'] and q64['correct']==gate['C64_correct'];assert all(v['banks']==v['M4096_banks']==42 for v in q64['actual_attention_runtime_native_by_role'].values()) and q64['actual_attention_M64_capture_native_banks']==0
 assert c8['frozen_sources']==q64['frozen_sources']==state['frozen_sources'] and c8['attention_mode']==q64['attention_mode']=='candidate' and q64['server_command']==state['server_command'] and c8['additional_environment']==q64['additional_environment']==state['additional_environment'];a=normalized_environment(q64['profile_environment'],Path(gate['arms']['64']['raw_root']));b=normalized_environment(state['profile_environment'],root);assert {k for k in a if a[k]!=b[k]}=={'TIANSHU_FUTURE_KV_MODE'} and a['TIANSHU_FUTURE_KV_MODE']=='control' and b['TIANSHU_FUTURE_KV_MODE']=='candidate';frozen_check(c8['controller_sources']);frozen_check(q64['controller_sources']);frozen_check(state['controller_sources'])
 result=original.audit_benchmark(root,state,profile,c8,write=False);result.update(runtime_source_count=925,environment_count=63,fresh_C8_correct=c8['correct'],fresh_C64_correct=q64['correct'],actual_C64_native_QKV_O_Down_all42_verified=True,M64_QKV_O_BF16_all84_retained=True,Down_M64_BF16_retained=True,original_three_BF16_GEMM_warm_paths_verified=True,previous_formal_total_tok_s=5950.75,vs_previous_formal_percent=100*(result['total_tok_s']/5950.75-1),historical_comparison_is_not_same_run_AB=True,short_same_run_screen_sha256=digest(SHORT/'SCREEN_GATE_prefill_32g.json'),short_same_run_gain_percent=screen['vs_exact_r3_short_gain_percent'],quality_gate_sha256=digest(EXP/'tianshu_LM_pad64_quality_results_v2_20261022/QUALITY_GATE_32g.json'));result['candidate_passes_promotion_performance_gates']=result['vs_previous_formal_percent']>.5 and result['postfirst_spread_percent']<1 and result['ttft_ms']<=3174.58*1.01
 if write:(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
