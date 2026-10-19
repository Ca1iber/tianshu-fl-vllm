# SPDX-License-Identifier: Apache-2.0
"""Audit only observed graph coverage; preserve the original stricter audit failure."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();s=json.loads((r/'result.json').read_text());assert s['phase']=='audit_failed' and s['error']=='AssertionError()'
assert not s['official_benchmark'] and not s['adopted']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in s['frozen_sources'].items():assert sha(r/'source_snapshot'/Path(n).relative_to('/'))==h,n
spec=importlib.util.spec_from_file_location('_graph_scope_layout_audit',Path(__file__).with_name('layout_audit.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
assert mod.audit_runtime(r,s,official=True,online=False)==s['profile_audit']
ref=s['canary_reference'];rp=r/'source_snapshot'/Path(ref['path']).relative_to('/');assert sha(rp)==ref['sha256'];gold=json.loads(rp.read_text())['seeded_canaries'][0]['text']
assert len(s['seeded_canaries'])==3
for c in s['seeded_canaries']:assert c['text']==gold and c['api_seed']==777 and c['usage']==dict(prompt_tokens=128,completion_tokens=16,total_tokens=144)
assert len(s['diagnostic_workloads'])==6
for i,(e,n) in enumerate(zip(s['diagnostic_workloads'],[1,8,16,24,32,64])):
 assert e['concurrency']==e['successful_requests']==len(e['responses'])==n
 for j,c in enumerate(e['responses']):
  assert c['input_seed']==202610193000+100*i+j and c['usage']==dict(prompt_tokens=4096,completion_tokens=128,total_tokens=4224)
  assert c['wall_seconds']>=c['ttft_seconds']>0
workers=[json.loads(p.read_text()) for p in (r/'graph_segments_records').glob('trace_*.json')]
workers=[d for d in workers if set(d['bound'])=={'graph','runner','routing'}];assert len(workers)==1;t=workers[0]
assert t['sources']['/usr/local/lib/python3.12/site-packages/vllm_fl/compilation/graph.py']=='ba78ed0a736bb21a4f2c307fc9775f15e3742b94533eff333b0c364eb8213fc2'
for n,h in t['sources'].items():assert s['frozen_sources'][n]==h
assert t['diagnostic_only'] and t['RNG_not_read_or_modified'] and t['attention_arguments_and_return_values_forwarded']
assert len(t['graphs'])==35
ids=[];links=[]
for k,g in t['graphs'].items():
 assert g['runtime_mode']=='FULL' and g['capture_calls']==1 and g['graph_object_id']>0
 ids.append(g['graph_object_id']);cap=[x for x in t['launches'].values() if x['graph_key']==k and x['actually_inside_capture']]
 assert cap and sum(x['calls'] for x in cap)==42
 assert len({json.dumps(x['configuration'],sort_keys=True) for x in cap})==1
 assert all(x['min_max_seqlen_k']==x['max_max_seqlen_k']==131072 for x in cap)
 config=cap[0]['configuration']
 if config['IS_3D']:
  n=config['sequences'];expected=64 if n==1 else 32 if n<=3 else 16 if n<=8 else 8 if n<=16 else 16 if n<=24 else 4
  assert (config['segments'],config['tile'],config['warps'],config['stages'])==(expected,16,1,1)
 for n,f in g['frames'].items():
  assert g['replay_calls']>=f['count']>0
  if f['max_computed']<4096:continue
  assert config['sequences']>=int(n)>0
  links.append(dict(actual_requests=int(n),descriptor=g['descriptor'],graph_object_id=g['graph_object_id'],configuration=config,frame=f,capture_max_seqlen_k=131072))
assert len(set(ids))==35
actual={x['actual_requests'] for x in links};assert actual==set(range(1,64))
assert any(x['configuration']['sequences']==64 and x['actual_requests']==63 for x in links)
assert 'No running processes found' in (r/'hardware_after.txt').read_text()
out=dict(complete_for_observed_scope=True,original_phase_preserved='audit_failed',original_failure_reason='No observed long-context frame with exactly64 live requests',all145_load_requests_and3_canaries_checked=True,sources=len(s['frozen_sources']),actual_single_FL_graph_objects=35,actual_attention_capture_calls=35*42,all_captured_max_seqlen_k=131072,actual_live_request_counts=sorted(actual),exact64_live_requests_not_observed=True,descriptor64_GPU_replay_observed_at63_live_requests=True,long_context_capture_policy_correct=True,actual_links=links,adopted=False,performance_and_new_quality_not_claimed=True)
(r/'local_observed_scope_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='actual_links'}))
