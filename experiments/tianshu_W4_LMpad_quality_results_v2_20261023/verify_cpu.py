# SPDX-License-Identifier: Apache-2.0
import argparse,collections,csv,hashlib,importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--batches',nargs='+',type=int,choices=(8,64),default=[8,64]);args=parser.parse_args();assert len(set(args.batches))==len(args.batches)
 prefix=f'C{args.batches[0]}_' if len(args.batches)==1 else '';out=BASE/f'{prefix}QUALITY_GATE_32g.json';csvfile=BASE/f'{prefix}KEY_SCORES_32g.csv';assert not out.exists() and not csvfile.exists()
 spec=importlib.util.spec_from_file_location('_LMpad_v2_independent_quality',EXP/'tianshu_W4_LMpad_quality_v2_20261023/quality_profile.py');q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
 model=read(q.SERVICE/'CPU_GATE_32g.json');cpu=read(q.BASE/'CPU_GATE_32g.json');expected={**model['sources'],str(q.SERVICE/'CPU_GATE_32g.json'):sha(q.SERVICE/'CPU_GATE_32g.json')};published=read(EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json');raw={};arms={};rows=[];shared=controllers=None
 for batch in args.batches:
  root=Path(f'/workspace/benchmark-results/20261023_W4_LMpad_v2_math{batch}_tianshu');s=read(root/'result.json');a=read(root/'accuracy_audit.json')
  assert s['phase']=='complete' and s['evalscope_exit_code']==0 and s['quality_batch']==batch and a['complete'] and a['quality_batch']==batch and not s['reused_original_accuracy']
  assert s['frozen_sources']==a['frozen_sources']==expected and len(expected)==model['expected_model_source_count']==944
  assert all(expected[p]==h for p,h in published['frozen_sources'].items()) and len(published['frozen_sources'])==903
  before=cpu['controllers_before_CPU_gate_and_EXECUTION'];assert all(s['controller_sources'][p]==h for p,h in before.items()) and set(s['controller_sources'])-set(before)=={str(q.BASE/'CPU_GATE_32g.json'),str(q.BASE/'EXECUTION.sha256')}
  assert shared is None or shared==s['frozen_sources'];assert controllers is None or controllers==s['controller_sources'];shared=s['frozen_sources'];controllers=s['controller_sources']
  for field,folder in [('frozen_sources','source_snapshot'),('controller_sources','controller_source_snapshot')]:
   for p,h in s[field].items():assert sha(p)==h and sha(root/folder/Path(p).relative_to('/'))==h
  reports=list(root.glob('evalscope/*/reports/minicpm/math_500.json'));assert len(reports)==1;rp=reports[0];report=read(rp);run=rp.parents[2];vp=run/'reviews/minicpm/math_500_Level 3.jsonl';pp=run/'predictions/minicpm/math_500_Level 3.jsonl';reviews=[json.loads(x) for x in vp.read_text().splitlines()];preds=[json.loads(x) for x in pp.read_text().splitlines()]
  assert len(reviews)==len(preds)==report['num']==105 and len({x['index'] for x in reviews})==len({x['index'] for x in preds})==105 and {x['index'] for x in reviews}=={x['index'] for x in preds}
  execution=report['execution_summary'];assert execution['requested']==execution['succeeded']==105 and execution['errored']==0 and execution['incomplete'] is False
  scores=[x['sample_score']['score'] for x in reviews];assert all(x['status']=='success' and x['value']['accuracy'] in (0,1) for x in scores) and all(p['model_output']['error'] is None for p in preds)
  correct=int(sum(x['value']['accuracy'] for x in scores));metric=next(x for x in report['metrics'] if x['identity']['name']=='accuracy');assert metric['num']==105 and abs(metric['score']-correct/105)<.000051 and correct==s['correct']==a['correct'] and s['total']==a['total']==105 and s['accuracy_passed']==a['official_requirement_passed']==(correct/105>.95)
  for p in (rp,vp,pp):
   v=a['raw_files'][str(p.relative_to(root))];assert sha(p)==v['sha256'] and p.stat().st_size==v['bytes'];raw[str(p)]=sha(p)
  profile=q.audit_runtime(root,s,official=False);calculated=q.audit_math(root,s,profile,write=False);assert profile==a['profile'] and calculated==a
  assert profile['actual_Down64_W4_capture_banks']==42 and profile['old_Down64_BF16_capture_banks']==0 and profile['dots_Down64_load_and_native_memory_scratch_all42_accounted'] and profile['actual_attention_M64_capture_BF16_banks']==84 and profile['actual_attention_M64_capture_native_banks']==0
  assert profile['new_LM64_actual_W4_bank_verified'] and profile['new_LM64_load_and_native_peak_accounting_verified'] and profile['new_LM1_64_range_and_padding_profile_verified'] and profile['old_LM1_64_BF16_W8_calls_absent'] and profile['LM_head_precision_in_verified_batch']=='W4A16' and profile['new_LM_verified_batch_W4_runtime_calls']>=10
  assert profile['all_original_GU_KV_attention_scheduler_sampling_checks_retained'] and profile['only_original_LM1_64_precision_requirements_explicitly_replaced']
  if batch==64:assert profile['dots_Down64_captured_replay_linked_to_inherited_model_M64_execution'] and profile['new_LM64_W4_actual_runtime_calls']>=10
  else:assert profile['new_LM8_W4_actual_runtime_calls']>=10
  observed=read(root/'observed_full_server_environment.json');assert observed['environment']==s['profile_environment']|s['additional_environment'] and len(observed['environment'])==63 and observed['command']==s['server_command']==published['server_command'] and len(s['server_command'])==13 and observed['environment']['TIANSHU_FUTURE_KV_MODE']==('control' if batch==64 else 'candidate')
  normalized=q.normalized_environment(s['profile_environment'],root);baseline=q.normalized_environment(published['environment'],Path(published['old_output_root']));assert {k for k in normalized if normalized[k]!=baseline[k]}==({'PYTHONPATH','TIANSHU_FUTURE_KV_MODE'} if batch==64 else {'PYTHONPATH'})
  assert 'No running processes found' in (root/'hardware_after.txt').read_text()
  for p in (root/'result.json',root/'accuracy_audit.json',next((root/'dots_LM64_records').glob('route_*.json')),next((root/'dots_Down64_records').glob('route_*.json')),next((root/'native_attention_records').glob('route_*.json')),next((root/'native_down_records').glob('route_*.json'))):raw[str(p)]=sha(p)
  byindex={p['index']:p for p in preds}
  for r in reviews:rows.append(dict(batch=batch,index=r['index'],accuracy=r['sample_score']['score']['value']['accuracy'],stop_reason=byindex[r['index']]['model_output']['choices'][0]['stop_reason']))
  arms[str(batch)]=dict(root=str(root),correct=correct,total=105,accuracy=correct/105,above95=correct/105>.95,incorrect_indices=[x['index'] for x in reviews if not x['sample_score']['score']['value']['accuracy']],stop_reasons=dict(collections.Counter(p['model_output']['choices'][0]['stop_reason'] for p in preds)),all105_counted_including_caps=True,actual_Down64_W4_capture_banks=42,actual_LM8_W4_runtime_calls=profile['new_LM8_W4_actual_runtime_calls'],actual_LM64_W4_runtime_calls=profile['new_LM64_W4_actual_runtime_calls'],actual_LM_W4_calls_by_M=profile['new_LM_W4_calls_by_M'],full_runtime_audits_replayed_exact=True)
 for item in read(q.SERVICE/'SELECTED_LM_32g.json')['buffers'].values():assert sha(item['file'])==item['file_sha256'];raw[item['file']]=item['file_sha256']
 for item in read(EXP/'tianshu_dotsaddz_Down64_service_v2_20261023/SELECTED_BANKS_32g.json')['banks']:assert sha(item['file'])==item['file_sha256'];raw[item['file']]=item['file_sha256']
 checkpoint=sha('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');assert checkpoint=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
 with csvfile.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,model_quality_qualified=set(arms)=={'8','64'} and all(v['above95'] for v in arms.values()),C8_correct=arms.get('8',{}).get('correct'),C64_correct=arms.get('64',{}).get('correct'),total_per_batch=105,evaluated_batches=args.batches,model_source_count=len(expected),controller_count=len(controllers),model_frozen_sources=shared,controller_sources=controllers,original903_r3_sources_exact_subset=True,new_precision_vs_r3='M64_Down_and_LM_M1_to64_W4',fresh_entire_batch_8_LM_precision_changed=True,old_LM8_BF16_and_LM64_W8_guards_explicitly_replaced=True,new_LM1_64_load_and_native_peak_accounting_verified=True,all_105_reviews_predictions_report_and_raw_SHA_verified=True,all_caps_in_denominator=True,checkpoint_sha256=checkpoint,raw_files=raw,arms=arms,verifier_source_sha256=sha(__file__),key_scores_sha256=sha(csvfile),no_original_raw_modified=True,official_benchmark=False,adopted=False)
 assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);out.write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('model_frozen_sources','controller_sources','raw_files','arms')}));print(json.dumps(arms))
if __name__=='__main__':main()
