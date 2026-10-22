# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json,sys,traceback
from pathlib import Path
import patch_runtime as patch
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/workspace/benchmark-results/20261022_LM_pad64_math64_tianshu');OUT=Path('/workspace/benchmark-results/20261022_LM_pad64_math64_CPU_recovery_tianshu')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not OUT.exists();OUT.mkdir();s=read(ROOT/'result.json');assert s['phase']=='audit_failed' and s['evalscope_exit_code']==0 and s['quality_batch']==64 and len(s['frozen_sources'])==925 and len(s['controller_sources'])==82;assert 'No running processes found' in (ROOT/'hardware_after.txt').read_text()
 for key,folder in (('frozen_sources','source_snapshot'),('controller_sources','controller_source_snapshot')):
  for p,h in s[key].items():assert sha(p)==h and sha(ROOT/folder/Path(p).relative_to('/'))==h
 own={str(p):sha(p) for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md']}
 for p,h in own.items():d=OUT/'recovery_controller_source_snapshot'/Path(p).relative_to('/');d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(Path(p).read_bytes());assert sha(d)==h
 spec=importlib.util.spec_from_file_location('_LM_recovery_original_frozen_quality',EXP/'tianshu_LM_pad64_quality_20261022/quality_profile.py');q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
 try:q.audit_runtime(ROOT,s,official=False)
 except AssertionError as error:
  tb=traceback.extract_tb(error.__traceback__);failure=tb[-1];assert Path(failure.filename).resolve()==patch.TARGETS['math'][0].resolve() and 'sum(x["calls"] for x in lm)' in (failure.line or '');original_failure=dict(file=failure.filename,line=failure.lineno,statement=failure.line,error=repr(error))
 else:raise AssertionError('Expected original actual Math LM64 failure was not reproduced')
 installed=patch.install(q,'math');calls=[]
 def trace(frame,event,arg):
  if event=='call' and frame.f_code.co_name=='audit_runtime':calls.append(str(Path(frame.f_code.co_filename).resolve()))
 sys.setprofile(trace)
 try:profile=q.audit_runtime(ROOT,s,official=False)
 finally:sys.setprofile(None)
 assert str(patch.TARGETS['math'][1].resolve()) in calls and str(patch.TARGETS['math'][0].resolve()) not in calls;quality=q.audit_math(ROOT,s,profile,write=False);assert quality['total']==105 and quality['complete'] and quality['quality_batch']==64;quality.update(CPU_coverage_reconciliation=True,actual_math_raw_root=str(ROOT),original_result_sha256=sha(ROOT/'result.json'),original_experiment_passed=False,original_failure=original_failure,actual_LM64_audit_replacement_targets=installed,recovery_controller_sources=own,original82_controller_sources_preserved=True,all_other_runtime_checks_recomputed=True)
 derived=dict(s,phase='CPU_recovery_complete',original_phase=s['phase'],original_error=s.get('error'),original_result_sha256=sha(ROOT/'result.json'),original_experiment_passed=False,CPU_runtime_audit_recovered=True,correct=quality['correct'],total=105,accuracy=quality['accuracy'],accuracy_passed=quality['official_requirement_passed'],profile_audit=profile,CPU_recovery_controller_sources=own,actual_math_raw_root=str(ROOT));(OUT/'accuracy_audit.json').write_text(json.dumps(quality,indent=2)+'\n');(OUT/'result.json').write_text(json.dumps(derived,indent=2)+'\n');(OUT/'AUDIT_CALL_TRACE_32g.json').write_text(json.dumps(calls,indent=2)+'\n');gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,original_experiment_passed=False,original_phase=s['phase'],original_failure_reproduced=original_failure,original_result_sha256=sha(ROOT/'result.json'),model_source_count=925,original_controller_count=82,recovery_controller_sources=own,recovery_snapshot_hashes_verified=True,new_actual_Math_guard_execution_traced=True,all_other_runtime_checks_recomputed=True,correct=quality['correct'],total=105,accuracy=quality['accuracy'],accuracy_passed=quality['official_requirement_passed'],raw_root=str(ROOT),recovery_root=str(OUT),recovery_result_sha256=sha(OUT/'result.json'),recovery_accuracy_sha256=sha(OUT/'accuracy_audit.json'),raw_untouched=True,adopted=False);(BASE/'RECOVERY_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in gate.items() if k!='recovery_controller_sources'}))
if __name__=='__main__':main()
