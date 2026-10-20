# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_prefill_chunk_parent_audit',BASE.parent/'tianshu_sampler_exp_service_20261021/audit_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit_runtime(root,state,official=False):
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    threshold=0 if state['mode']=='control' else 2048
    rows=[r for path in (root/'prefill_chunk_records').glob('default_*.json') for r in read(path)];assert rows and any(r['default_selected'] for r in rows)
    for row in rows:
        assert row['mode']==state['mode'] and row['original_called_once'] and row['original_arguments_and_return_forwarded'] and row['nonzero_explicit_threshold_preserved']
        if row['default_selected']:assert row['before_threshold']==0 and row['after_threshold']==threshold and row['actual_budget']==4096 and row['actual_max_seqs']==64
        for field in ('native','policy'):
            path=row[field+'_source'];assert sha(path)==state['frozen_sources'][path]==row[field+'_source_sha256' if field=='native' else field+'_sha256']
    files=list((root/'prefill_chunk_records').glob('schedule_*.json'));assert len(files)==1
    scheduler=read(files[0]);assert scheduler['mode']==state['mode'] and scheduler['actual_threshold']==threshold and scheduler['actual_budget']==4096 and scheduler['actual_max_seqs']==64
    assert scheduler['steps']>0 and scheduler['observed_prefill_events']>0 and scheduler['max_observed_prefill_work']==(2048 if threshold else 4096)
    assert all(0<int(key)<=(threshold or 4096) for key in scheduler['prefill_histogram'])
    for field in ('scheduler_module','actual_schedule','policy'):
        path=scheduler[field+'_source'];assert sha(path)==state['frozen_sources'][path]==scheduler[field+'_sha256']
    result.update(mode=state['mode'],qualified_sampler_warmup_mode='candidate',actual_prefill_chunk_threshold=threshold,prefill_default_records=rows,
        actual_scheduler_chunk_observation=scheduler,actual_prefill_cap_enforced_by_native_schedule=True,new_Math_required=True)
    return result
