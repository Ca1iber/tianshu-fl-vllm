# SPDX-License-Identifier: Apache-2.0
"""Recompute fresh Math or official4K audit from downloaded original artifacts."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_soft8_fresh_validation_audit',BASE/'audit_profile.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
from validation_profile import audit_math,audit_benchmark
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--math-root',type=Path);a=p.parse_args();root=a.root.resolve()
def read(p):return json.loads(p.read_text())
s=read(root/'result.json');assert s['phase']=='complete' and not s['reused_original_accuracy']
repo=BASE.parents[1]
for path,h in s['frozen_sources'].items():
 if Path(path).is_relative_to('/workspace/vllm-plugin-FL'):assert hashlib.sha256((repo/Path(path).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==h
profile=audit.audit_runtime(root,s,official=s['stage']=='official',online=False)
assert profile['software8_KV'] and profile['K8V16'] and profile['bounded_arena_accounted']
if s['stage']=='math':
 quality=audit_math(root,s,profile,write=False);assert quality==read(root/'accuracy_audit.json')
 assert s['correct']==quality['correct'] and s['total']==105 and s['accuracy']==quality['accuracy'] and s['accuracy_passed']==quality['official_requirement_passed']
 r=dict(complete=True,stage='math',correct=quality['correct'],total=105,accuracy=quality['accuracy'],accuracy_passed=quality['official_requirement_passed'],all105_prediction_and_review_records_checked=True,source_snapshots=len(s['frozen_sources']),fresh_original_command_profile=True,software8_KV=True,official_benchmark=False,adopted=False)
else:
 assert a.math_root is not None
 mr=a.math_root.resolve();ms=read(mr/'result.json');assert ms['phase']=='complete' and ms['stage']=='math'
 mp=audit.audit_runtime(mr,ms,official=False,online=False)
 quality=audit_math(mr,ms,mp,write=False);assert quality==read(mr/'accuracy_audit.json') and quality['official_requirement_passed']
 comparison=audit_benchmark(root,s,profile,quality,write=False);assert comparison==read(root/'comparison.json')
 r=dict(complete=True,stage='official',total_tok_s=comparison['total_tok_s'],output_tok_s=comparison['output_tok_s'],ttft_ms=comparison['ttft_ms'],official_requests_checked=comparison['requests_successful'],fresh_Math_correct=quality['correct'],fresh_Math_total=105,source_snapshots=len(s['frozen_sources']),raw_CSVs_recomputed=True,full16K_not_run=True,adopted=False)
(root/'local_validation_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
