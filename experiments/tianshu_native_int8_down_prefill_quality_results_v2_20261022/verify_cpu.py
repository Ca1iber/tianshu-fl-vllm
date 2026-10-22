# SPDX-License-Identifier: Apache-2.0
"""Independently count every raw review, verify immutable snapshots and actual Down routes."""
import collections,hashlib,importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (BASE/'QUALITY_GATE_32g.json').exists()
 spec=importlib.util.spec_from_file_location('_down_quality_CPU_own_RO_audit',EXP/'tianshu_native_int8_down_service_v2_20261022/audit_profile.py');own=importlib.util.module_from_spec(spec);spec.loader.exec_module(own)
 r1=read(EXP/'tianshu_native_int8_prefill_official_results_v2_20261022/OFFICIAL_GATE_32g.json');arms={};raw_hashes={};shared=None
 for batch in (8,64):
  root=Path(f'/workspace/benchmark-results/20261022_native_int8_down_prefill_math{batch}_v2_retry1_tianshu');state=read(root/'result.json');quality=read(root/'accuracy_audit.json')
  assert state['phase']=='complete' and state['evalscope_exit_code']==0 and state['accuracy_passed'] and state['quality_batch']==batch and state['down_scope']=='prefill' and quality['complete'] and quality['quality_batch']==batch and quality['down_scope']=='prefill'
  assert state['frozen_sources']==quality['frozen_sources'] and len(state['frozen_sources'])==890 and all(state['frozen_sources'][p]==h for p,h in r1['model_frozen_sources'].items())
  if shared is None:shared=state['frozen_sources']
  else:assert shared==state['frozen_sources']
  for field,folder in [('frozen_sources','source_snapshot'),('controller_sources','controller_source_snapshot')]:
   for p,h in state[field].items():assert sha(p)==h and sha(root/folder/Path(p).relative_to('/'))==h
  reports=list(root.glob('evalscope/*/reports/minicpm/math_500.json'));assert len(reports)==1;report_path=reports[0];report=read(report_path);run=report_path.parents[2];review_path=run/'reviews/minicpm/math_500_Level 3.jsonl';prediction_path=run/'predictions/minicpm/math_500_Level 3.jsonl'
  reviews=[json.loads(line) for line in review_path.read_text().splitlines()];predictions=[json.loads(line) for line in prediction_path.read_text().splitlines()];assert len(reviews)==len(predictions)==report['num']==105 and len({r['index'] for r in reviews})==len({p['index'] for p in predictions})==105 and {r['index'] for r in reviews}=={p['index'] for p in predictions}
  execution=report['execution_summary'];assert execution['requested']==execution['succeeded']==105 and execution['errored']==0 and execution['incomplete'] is False
  scores=[r['sample_score']['score'] for r in reviews];assert all(s['status']=='success' and s['value']['accuracy'] in (0,1) for s in scores) and all(p['model_output']['error'] is None for p in predictions);correct=int(sum(s['value']['accuracy'] for s in scores));metric=next(m for m in report['metrics'] if m['identity']['name']=='accuracy');assert metric['num']==105 and abs(metric['score']-correct/105)<.000051 and correct==state['correct']==quality['correct'] and correct/105>.95
  for p in (report_path,review_path,prediction_path):
   recorded=quality['raw_files'][str(p.relative_to(root))];assert sha(p)==recorded['sha256'] and p.stat().st_size==recorded['bytes'];raw_hashes[str(p)]=sha(p)
  actual=own.audit(root,state);actual.update(own.audit_startup(root,state));assert actual['actual_Down_M64_capture_banks']==0 and actual['new_Down_memory_profile_all42_accounted']
  for key in ('actual_native_Down_runtime_banks','actual_native_Down_runtime_M_values','actual_Down_M64_capture_banks'):assert actual[key]==quality[key]
  assert actual['native_down_record']==quality['profile']['native_down_record']
  if batch==64:assert actual['actual_native_Down_runtime_banks']==42 and 4096 in actual['actual_native_Down_runtime_M_values']
  observed=read(root/'observed_full_server_environment.json');assert len(observed['environment'])==63 and observed['environment']==state['profile_environment']|state['additional_environment'] and observed['command']==state['server_command'] and len(observed['command'])==13
  assert 'No running processes found' in (root/'hardware_after.txt').read_text()
  raw_hashes[str(root/'result.json')]=sha(root/'result.json');raw_hashes[str(root/'accuracy_audit.json')]=sha(root/'accuracy_audit.json');raw_hashes[str(next((root/'native_down_records').glob('route_*.json')))]=sha(next((root/'native_down_records').glob('route_*.json')))
  arms[str(batch)]=dict(root=str(root),correct=correct,total=105,accuracy=correct/105,incorrect_indices=[r['index'] for r in reviews if not r['sample_score']['score']['value']['accuracy']],stop_reasons=dict(collections.Counter(p['model_output']['choices'][0]['stop_reason'] for p in predictions)),max_tokens_indices=[p['index'] for p in predictions if p['model_output']['choices'][0]['stop_reason']=='max_tokens'],actual_native_Down_runtime_banks=actual['actual_native_Down_runtime_banks'],actual_native_Down_runtime_M_values=actual['actual_native_Down_runtime_M_values'],actual_Down_M64_capture_banks=0,startup_three_vendor_mm_verified=True,all105_counted_including_caps=True,source_snapshots_all_checked=True,environment_count=63,original13_serve_argv=True)
 checkpoint_hash=sha(MODEL);assert checkpoint_hash=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
 result=dict(complete=True,CPU_only=True,no_GPU_imports=True,scope='prefill',new_model_accuracies_independently_recomputed=True,C8_correct=arms['8']['correct'],C64_correct=arms['64']['correct'],total_per_batch=105,official_C8_and_actual_C64_both_above95=True,original877_sources_exact_subset=True,model_source_count=890,model_frozen_sources=shared,actual_Down_M64_INT8_capture_absent=True,C64_actual_Down_M4096_all42=True,own_actual_runtime_and_startup_audits_independently_recomputed=True,inherited_runtime_audits_passed_in_original_validator=True,inherited_audits_not_separately_rerun_by_this_script=True,source_snapshot_hashes_rechecked=True,checkpoint_sha256=checkpoint_hash,raw_files=raw_hashes,arms=arms,original_disk_failure_retained=True,no_original_raw_modified=True,official_benchmark=False,adopted=False,verifier_source_sha256=sha(__file__))
 (BASE/'QUALITY_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in result.items() if k not in ('model_frozen_sources','raw_files','arms')}));print(json.dumps(arms))
if __name__=='__main__':main()
