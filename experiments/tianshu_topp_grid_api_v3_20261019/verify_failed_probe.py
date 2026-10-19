"""Audit V3 batch-level gate failure without claiming request-level outcomes."""
import argparse,ast,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
s=json.loads((root/'result.json').read_text());d=json.loads((root/'micro/result.json').read_text())
assert s['phase']==d['phase']=='failed' and not d['complete'];f=ast.literal_eval(d['error'][len('AssertionError('):-1]);assert f[:3]==(3,.95,1)
repo=Path(__file__).resolve().parents[2]
for name,sha in d['source_snapshots'].items():
 assert hashlib.sha256((root/'micro/source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==sha
 if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
  assert hashlib.sha256((repo/Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha
columns=('kept_mass','smallest_kept_probability','CPU_p','minimum_kept_logit','maximum_removed_logit','nonempty');rows=strict_rows=0
for e in d['masks']:
 c=e['checks']['candidate'];assert c['relative_gate'] and c['retained_values_unchanged'] and (c['exact_adopted_baseline'] or c['CPU_strict_semantics'])
 strict=[]
 for mass,smallest,p,lo,hi,nonempty in zip(*(c[k] for k in columns)):
  strict.append(bool(nonempty and mass>=p-1e-6 and mass-smallest<=p+1e-6 and hi<=lo));rows+=1
 assert all(strict)==c['CPU_strict_semantics'];strict_rows+=sum(strict)
c=f[3]['candidate'];b=f[3]['original'];assert b['exact_adopted_baseline'] and not c['exact_adopted_baseline']
assert c['adopted_mask_difference']==1 and c['retained_values_unchanged'] and not c['relative_gate']
assert not b['CPU_strict_semantics'] and not c['CPU_strict_semantics']
failed=[]
for i,(mass,smallest,p,lo,hi,nonempty) in enumerate(zip(*(c[k] for k in columns))):
 if not (nonempty and mass>=p-1e-6 and mass-smallest<=p+1e-6 and hi<=lo):
  assert all(c[k][i]==b[k][i] for k in columns)
  failed.append(dict(row=i,inversion=hi-lo,scalar_diagnostics_identical_to_baseline=True))
assert len(failed)==1 and failed[0]['row']==21
assert len(d['masks'])==43 and len(d['events'])==70 and not d['RNG_cases'] and not d['fallbacks'] and not d.get('lifecycle_cases')
assert 'No running processes found' in (root/'hardware_after_process_exit.txt').read_text()
r=dict(complete_scoped_audit=True,original_failure_preserved=True,completed_relative_masks=43,completed_rows=rows,strict_rows=strict_rows,event_records=70,source_snapshots=len(d['source_snapshots']),failed_case=dict(case=3,batch=48,p=.95,phase=1,candidate_mask_difference=1,non_strict_rows=failed),failed_row_bitwise_equality_not_serialized=True,full_API_gate_passed=False,RNG_not_run=True,lifecycle_not_run=True,service_not_run=True,adopted=False)
(root/'local_failed_download_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
