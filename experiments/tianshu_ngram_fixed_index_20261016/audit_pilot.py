# SPDX-License-Identifier: Apache-2.0
"""Original ngram stack checks plus real fixed-index route evidence."""
import importlib.util
import json
from pathlib import Path

path=Path(__file__).resolve().parents[1]/'tianshu_ngram_framework_20261016/audit_pilot.py'
spec=importlib.util.spec_from_file_location('_ngram_base_pilot_audit',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def audit_runtime(root,state,official=False,online=True):
    result=base.audit_runtime(root,state,official=official,online=online)
    records=[json.loads(p.read_text()) for p in (root/'ngram_records/fixed_index').glob('route_*.json')]
    assert records and all(r['installed_before_profile_and_capture'] and r['python_launch_calls']>0 for r in records)
    assert all(r['source_sha256']=='f48351f7ad1251acb680927d742123f92ea4f3e23505d83dfde6a3b7c713cc06' for r in records)
    widths={v['width'] for r in records for v in r['routes'].values()}
    assert {2048,130560}<=widths
    result.update(fixed_gather_actual_hidden_and_logits_routes_verified=True,Python_launch_observer_not_GPU_replay_counter=True)
    return result
