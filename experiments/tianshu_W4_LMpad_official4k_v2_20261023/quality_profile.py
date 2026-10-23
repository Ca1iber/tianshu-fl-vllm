# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;SHORT=EXP/'tianshu_W4_LMpad_short4k_v2_20261023';HOLDOUT=EXP/'tianshu_native_int8_attention42_20261022';RESULTS=EXP/'tianshu_W4_LMpad_quality_results_v2_20261023'
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
short=load('_LMpad_official_short_profile',SHORT/'candidate_profile.py');full=load('_LMpad_official_all_runtime',SHORT/'audit_profile.py');old=load('_LMpad_official_original_helpers',EXP/'tianshu_W4_LMpad_quality_v2_20261023/quality_profile.py');digest,read,frozen_check=short.digest,short.read,short.frozen_check;BENCH_SHA,csv_rows,normalized_environment=old.BENCH_SHA,old.csv_rows,old.normalized_environment;original=short.normalizer;audit_startup=full.audit_startup

def audit_math(*args,**kwargs):raise AssertionError('Official-only after complete fresh model quality')

def screen_gate():
 g=read(SHORT/'SCREEN_GATE_prefill_32g.json');assert g['complete'] and g['candidate_passes_perf_screen'] and g['vs_exact_r3_short_gain_percent']>.5 and g['arms']['candidate']['hot_spread_pct']<1
 for name in ('control','candidate'):
  root=Path(g['arms'][name]['raw_root']);assert digest(root/'result.json')==g['arms'][name]['raw_result_sha256'];s=read(root/'result.json');assert s['phase']=='complete';frozen_check(s['frozen_sources']);frozen_check(s['controller_sources'])
 return g

def prepare(root,screen_root,batch=8,scope='prefill',check_screen=True,check_quality=True):
 assert batch==8 and scope=='prefill' and screen_root.resolve()==HOLDOUT.resolve()
 if check_screen:screen_gate()
 command,env,profile,sources=short.prepare(root,'candidate',scope,check_quality=check_quality);assert len(sources)==944
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run.sh'}
 if check_screen:assert read(BASE/'CPU_GATE_32g.json')['complete'];paths.update((BASE/'CPU_GATE_32g.json',BASE/'EXECUTION.sha256',SHORT/'SCREEN_GATE_prefill_32g.json',SHORT/'KEY_RUNS_prefill_32g.csv'))
 if check_quality:
  for raw in ('/workspace/benchmark-results/20261023_W4_LMpad_v2_math8_tianshu','/workspace/benchmark-results/20261023_W4_LMpad_v2_math64_tianshu'):
   paths.update({Path(raw)/'result.json',Path(raw)/'accuracy_audit.json'})
 controllers=read(root/'controller_sources.json');controllers.update({str(p):digest(p) for p in paths})
 for module in tuple(sys.modules.values()):
  p=getattr(module,'__file__',None)
  if p and p.endswith('.py') and Path(p).is_relative_to(EXP):controllers[str(Path(p).resolve())]=digest(p)
 (root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,env,profile,sources

def audit_runtime(root,state,official=False):
 assert official and state['stage']=='official' and state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=256,rounds=4,excluded_rounds=[1]);return full.audit_runtime(root,state,official=True)

def audit_benchmark(root,state,profile,c8,write=True):
 quality=short.quality_gate('prefill',state['frozen_sources']);screen=screen_gate();c64root=Path(state['math64_root']);c64=read(c64root/'accuracy_audit.json');s64=read(c64root/'result.json');assert s64['phase']=='complete' and s64['accuracy_passed'] and not c64.get('CPU_coverage_reconciliation',False)
 assert c8['correct']==quality['C8_correct'] and c64['correct']==quality['C64_correct'] and c8['accuracy']>.95 and c64['accuracy']>.95 and c8['frozen_sources']==c64['frozen_sources']==state['frozen_sources'] and c64['server_command']==state['server_command'] and c8['additional_environment']==c64['additional_environment']==state['additional_environment']
 a=normalized_environment(c64['profile_environment'],c64root);b=normalized_environment(state['profile_environment'],root);assert {k for k in a if a[k]!=b[k]}=={'TIANSHU_FUTURE_KV_MODE'} and a['TIANSHU_FUTURE_KV_MODE']=='control' and b['TIANSHU_FUTURE_KV_MODE']=='candidate';frozen_check(c64['controller_sources']);frozen_check(c8['controller_sources']);frozen_check(state['controller_sources'])
 r=original.audit_benchmark(root,state,profile,c8,write=False);r.update(runtime_source_count=944,environment_count=63,fresh_C8_correct=c8['correct'],fresh_C64_correct=c64['correct'],C8_and_C64_original_normal_runs_complete=True,new_Down64_W4_all42_actual_capture_verified=True,new_LM64_actual_native_W4_runtime_calls=profile['new_LM64_W4_actual_runtime_calls'],new_LM8_actual_native_W4_runtime_calls=profile['new_LM8_W4_actual_runtime_calls'],LM1_64_old_BF16_or_W8_guards_explicitly_replaced=True,QKV64_O64_BF16_retained=True,previous_formal_total_tok_s=5950.75,vs_previous_formal_percent=100*(r['total_tok_s']/5950.75-1),historical_comparison_is_not_same_run_AB=True,short_same_run_gain_percent=screen['vs_exact_r3_short_gain_percent'],short_same_run_screen_sha256=digest(SHORT/'SCREEN_GATE_prefill_32g.json'),quality_gate_sha256=digest(RESULTS/'QUALITY_GATE_32g.json'));r['candidate_passes_promotion_performance_gates']=r['vs_previous_formal_percent']>.5 and r['postfirst_spread_percent']<1 and r['ttft_ms']<=3174.58*1.01
 if write:(root/'comparison.json').write_text(json.dumps(r,indent=2)+'\n')
 return r
