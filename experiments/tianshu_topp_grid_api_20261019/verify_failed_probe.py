"""Scope-limited audit of the unmodified failed API checker, never a service gate."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
state=json.loads((root/'result.json').read_text());d=json.loads((root/'micro/result.json').read_text())
assert state['phase']==d['phase']=='failed' and not d['complete']
assert d['error']=="RuntimeError('Triton Error [CUDA]: invalid argument')"
assert 'assess(working[name], expected, values, reference' in d['traceback']
assert 'flag_gems/ops/index.py' in d['traceback']
repo=Path(__file__).resolve().parents[2]
for name,sha in d['source_snapshots'].items():
    snapshot=root/'micro/source_snapshot'/Path(name).relative_to('/')
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==sha
    if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
        assert hashlib.sha256((repo/Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha
assert len(d['masks'])==20 and not d['RNG_cases'] and not d['fallbacks']
assert not d.get('lifecycle_cases')
rows=0
for entry in d['masks']:
    c=entry['checks']['candidate'];assert c['relative_gate'] and c['CPU_strict_semantics'] and c['retained_values_unchanged']
    keys=('kept_mass','smallest_kept_probability','CPU_p','minimum_kept_logit','maximum_removed_logit','nonempty')
    assert all(len(c[k])==entry['batch'] for k in keys)
    for mass,smallest,p,lo,hi,nonempty in zip(*(c[k] for k in keys)):
        assert nonempty and mass>=p-1e-6 and mass-smallest<=p+1e-6 and hi<=lo;rows+=1
    if entry['probability']==.95:
        for mode in ('original','candidate'):
            events=[e for e in d['events'] if (e['case'],e['phase'],e['mode'])==(entry['case'],entry['phase'],mode)]
            assert len(events)==5 and all(e['ms']>0 for e in events)
            assert statistics.median(e['ms'] for e in events)==entry['medians_ms'][mode]
assert 'No running processes found' in (root/'hardware_after_process_exit.txt').read_text()
report=dict(complete_scoped_audit=True,original_failure_preserved=True,completed_masks=20,
    completed_candidate_semantic_rows=rows,event_records=len(d['events']),sources=len(d['source_snapshots']),
    full_API_gate_passed=False,RNG_not_run=True,lifecycle_not_run=True,service_not_run=True,adopted=False)
(root/'local_failed_download_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
