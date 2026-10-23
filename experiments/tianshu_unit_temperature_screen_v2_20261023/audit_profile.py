# SPDX-License-Identifier: Apache-2.0
import hashlib,json
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def audit_startup(root,state):
 files=list((root/'unit_records').glob('route_*.json'));assert len(files)==1;s=read(files[0]);assert s['mode']==state['arm'] and s['CPU_metadata_verified'];assert len(state['server_command'])==13
 return dict(unit_hook_installed=True,original_r4_model_sources_preserved=True,performance_screen_only=True,full_inherited_formal_audit_not_claimed=True)

def audit_runtime(root,state,official=False):
 s=read(next((root/'unit_records').glob('route_*.json')));assert s['mode']==state['arm'] and s['metadata_refreshes']>0 and s['unit_metadata']>0
 key='skipped' if state['arm']=='candidate' else 'unit_control';assert s['by_batch']['64'][key]>=1000
 if state['arm']=='control':assert s['skipped']==0
 else:assert s['unit_control']==0
 assert all(r['successful_requests']==64 for r in state['screen_rounds']) and all(not r['tuning_records'] for r in state['screen_rounds'][1:])
 for p,h in state['frozen_sources'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 return dict(unit_routing=s,performance_screen_only=True,full_inherited_formal_audit_not_claimed=True,not_model_quality_or_adoption_gate=True)
