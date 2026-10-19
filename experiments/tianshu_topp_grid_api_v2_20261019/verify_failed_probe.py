"""Audit the preserved failed strict gate; never manufacture an API pass."""
import argparse,ast,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
s=json.loads((root/'result.json').read_text());d=json.loads((root/'micro/result.json').read_text())
assert s['phase']==d['phase']=='failed' and not d['complete']
assert d['error'].startswith('AssertionError(')
failed=ast.literal_eval(d['error'][len('AssertionError('):-1]);assert failed[:3]==(3,.5,0)
repo=Path(__file__).resolve().parents[2]
for name,sha in d['source_snapshots'].items():
 snapshot=root/'micro/source_snapshot'/Path(name).relative_to('/')
 assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==sha
 if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
  assert hashlib.sha256((repo/Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha
assert len(d['masks'])==40 and not d['RNG_cases'] and not d['fallbacks'] and not d.get('lifecycle_cases')
columns=('kept_mass','smallest_kept_probability','CPU_p','minimum_kept_logit','maximum_removed_logit','nonempty')
rows=0
for entry in d['masks']:
 c=entry['checks']['candidate'];assert c['relative_gate'] and c['CPU_strict_semantics'] and c['retained_values_unchanged']
 assert all(len(c[k])==entry['batch'] for k in columns)
 for mass,smallest,p,lo,hi,nonempty in zip(*(c[k] for k in columns)):
  assert nonempty and mass>=p-1e-6 and mass-smallest<=p+1e-6 and hi<=lo;rows+=1
c=failed[3]['candidate'];b=failed[3]['original']
assert c==b and c['exact_adopted_baseline'] and not c['CPU_strict_semantics'] and not c['relative_gate']
violations=[]
for i,(mass,smallest,p,lo,hi,nonempty) in enumerate(zip(*(c[k] for k in columns))):
 if not (nonempty and mass>=p-1e-6 and mass-smallest<=p+1e-6 and hi<=lo):
  violations.append(dict(row=i,kept_mass=mass,mass_minus_smallest=mass-smallest,probability=p,minimum_kept_logit=lo,maximum_removed_logit=hi,inversion=hi-lo))
assert len(violations)==1 and violations[0]['row']==6 and violations[0]['inversion']==2**-23
assert len(d['events'])==60
assert 'No running processes found' in (root/'hardware_after_process_exit.txt').read_text()
report=dict(complete_scoped_audit=True,original_failure_preserved=True,completed_masks=40,
 completed_candidate_semantic_rows=rows,event_records=60,sources=len(d['source_snapshots']),
 failed_case=dict(case=3,batch=48,probability=.5,phase=0,exact_current_baseline=True,strict_CPU=False,violations=violations),
 full_API_gate_passed=False,RNG_not_run=True,lifecycle_not_run=True,service_not_run=True,adopted=False)
(root/'local_failed_download_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
