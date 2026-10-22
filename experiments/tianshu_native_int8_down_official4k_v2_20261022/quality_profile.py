# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
SHORT=EXP/'tianshu_native_int8_down_short4k_v2_20261022';HOLDOUT=EXP/'tianshu_native_int8_down42_20261022'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
quality=load('_down_official_original_fresh_quality',EXP/'tianshu_native_int8_down_quality_v2_20261022/quality_profile.py')
short=load('_down_official_short_profile',SHORT/'candidate_profile.py')
full=load('_down_official_all_runtime_audits',SHORT/'audit_profile.py')
digest,read,frozen_check=quality.digest,quality.read,quality.frozen_check
BENCH_SHA,csv_rows,normalized_environment=quality.BENCH_SHA,quality.csv_rows,quality.normalized_environment
audit_startup=quality.audit_startup
def audit_math(*args,**kwargs):raise AssertionError('Official-only controller')

def prepare(root,screen_root,batch=8,scope='prefill'):
 assert batch==8 and scope=='prefill' and screen_root.resolve()==HOLDOUT.resolve()
 screen=read(SHORT/'SCREEN_GATE_prefill_32g.json');assert screen['complete'] and screen['candidate_passes_perf_screen'] and screen['vs_exact_r1_short_gain_percent']>.5 and screen['arms']['candidate']['hot_spread_pct']<1
 for arm in ('control','candidate'):
  raw=Path(screen['arms'][arm]['raw_root']);assert digest(raw/'result.json')==screen['arms'][arm]['raw_result_sha256'];prior=read(raw/'result.json');assert prior['phase']=='complete';frozen_check(prior['frozen_sources']);frozen_check(prior['controller_sources'])
 command,environment,profile,sources=quality.prepare(root,screen_root,batch=batch,scope=scope)
 assert len(sources)==890 and sources==read(Path(screen['arms']['candidate']['raw_root'])/'result.json')['frozen_sources'];short.quality_gate(scope,sources)
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run.sh',SHORT/'SCREEN_GATE_prefill_32g.json',SHORT/'KEY_RUNS_prefill_32g.csv'}|set(SHORT.glob('*.py'))
 for batch_id in (8,64):
  raw=Path(f'/workspace/benchmark-results/20261022_native_int8_down_prefill_math{batch_id}_v2_retry1_tianshu');paths.update((raw/'result.json',raw/'accuracy_audit.json',raw/'controller_sources.json'))
 controllers=read(root/'controller_sources.json');controllers.update({str(p):digest(p) for p in paths})
 (root/'controller_sources.json').write_text(json.dumps(controllers,indent=2)+'\n')
 for p,h in controllers.items():
  target=root/'controller_source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert digest(target)==h
 return command,environment,profile,sources

def audit_runtime(root,state,official=False):
 assert official and state['stage']=='official' and state['workload']==dict(input_tokens=4096,output_tokens=1024,concurrency=64,requests_per_round=256,rounds=4,excluded_rounds=[1])
 return full.audit_runtime(root,state,official=True)

def audit_benchmark(root,state,profile,c8,write=True):
 short.quality_gate('prefill',state['frozen_sources'])
 c64root=Path(state['math64_root']);c64=read(c64root/'accuracy_audit.json')
 assert c64['actual_native_Down_runtime_banks']==42 and 4096 in c64['actual_native_Down_runtime_M_values'] and c64['actual_Down_M64_capture_banks']==0
 assert c8['down_scope']==c64['down_scope']=='prefill' and c8['frozen_sources']==c64['frozen_sources']==state['frozen_sources']
 assert c64['server_command']==state['server_command'] and c8['additional_environment']==c64['additional_environment']==state['additional_environment']
 a=normalized_environment(c64['profile_environment'],c64root);b=normalized_environment(state['profile_environment'],root)
 assert {key for key in a if a[key]!=b[key]}=={'TIANSHU_FUTURE_KV_MODE'} and a['TIANSHU_FUTURE_KV_MODE']=='control' and b['TIANSHU_FUTURE_KV_MODE']=='candidate'
 frozen_check(c8['controller_sources']);frozen_check(c64['controller_sources']);frozen_check(state['controller_sources'])
 result=quality.legacy.original_benchmark(root,state,profile,c8,write=False)
 result.update(runtime_source_count=890,environment_count=63,fresh_C8_correct=c8['correct'],fresh_C64_correct=c64['correct'],actual_C64_native_Down_all42_verified=True,Down_M64_BF16_retained=True,original_three_BF16_GEMM_warm_paths_verified=True,previous_formal_total_tok_s=5777.22,vs_previous_formal_percent=100*(result['total_tok_s']/5777.22-1),historical_comparison_is_not_same_run_AB=True,short_same_run_screen_sha256=digest(SHORT/'SCREEN_GATE_prefill_32g.json'),short_same_run_gain_percent=read(SHORT/'SCREEN_GATE_prefill_32g.json')['vs_exact_r1_short_gain_percent'])
 if write:(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
