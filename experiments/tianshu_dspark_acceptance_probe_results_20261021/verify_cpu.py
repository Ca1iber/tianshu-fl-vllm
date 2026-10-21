# SPDX-License-Identifier: Apache-2.0
"""Independently parse raw Prometheus/text responses and recompute acceptance."""
import argparse,csv,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
BASE=Path(__file__).resolve().parent.parent/'tianshu_dspark_acceptance_probe_20261021';gate=json.loads((BASE/'CPU_GATE.json').read_text());r=json.loads((a.raw/'client_result.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert r['complete'] and r['phase']=='complete' and len(r['groups'])==9 and not r['official_benchmark'] and not r['accuracy_test']
terminal=json.loads((a.raw/'run_terminal.json').read_text());assert terminal['complete'] and terminal['exit_code']==0 and terminal['cleanup_attempted']
for path,h in gate['sources'].items():assert sha(path)==h and sha(a.raw/'source_snapshot'/Path(path).relative_to('/'))==h
assert sha(BASE/'CPU_GATE.json')==sha(a.raw/'source_snapshot'/BASE.relative_to('/')/'CPU_GATE.json')
assert sha(a.raw/'inputs.json')==gate['inputs_sha256']==sha(gate['inputs_path'])
inputs=json.loads((a.raw/'inputs.json').read_text());assert [x['input_tokens'] for x in inputs['natural_math']]==[x['input_tokens'] for x in inputs['random_matched']]
for name,items in inputs.items():
 assert len(items)==8
 for item in items:
  assert len(item['prompt'])==item['input_tokens'] and all(isinstance(x,int) and 0<=x<130560 for x in item['prompt'])
  if name=='random_4k':assert item['input_tokens']==4096
  if name!='natural_math':assert all(1<=x<120000 for x in item['prompt'])
def raw_counters(path):
 found={}
 for line in path.read_text().splitlines():
  if not line.startswith('vllm:spec_decode_num_'):continue
  metric,_,tail=line.partition('{');labels,_,number=tail.partition('}')
  if not metric.endswith('_total'):continue
  suffix=''
  for label in labels.split(','):
   key,_,value=label.partition('=')
   if key=='position':suffix='_position_'+value.strip('"')
  assert metric+suffix not in found
  value=float(number.strip());assert value>=0 and value.is_integer();found[metric+suffix]=int(value)
 return found
rows=[];responses=0;aggregates={}
expected_order=[(0,n,True) for n in inputs]+[(i,n,False) for i,order in enumerate(gate['round_orders'],1) for n in order]
assert [(g['round'],g['name'],g['warmup']) for g in r['groups']]==expected_order
for g in r['groups']:
 name=g['name'];tag=('warmup' if g['warmup'] else 'round'+str(g['round']))+'_'+name
 before=raw_counters(a.raw/(tag+'_metrics_before.txt'));after=raw_counters(a.raw/(tag+'_metrics_after.txt'))
 delta={k:after[k]-before.get(k,0) for k in after};assert delta==g['counter_deltas'] and all(x>=0 for x in delta.values())
 n=delta['vllm:spec_decode_num_drafts_total'];d=delta['vllm:spec_decode_num_draft_tokens_total'];accepted=delta['vllm:spec_decode_num_accepted_tokens_total'];p0=delta['vllm:spec_decode_num_accepted_tokens_per_pos_total_position_0'];p1=delta['vllm:spec_decode_num_accepted_tokens_per_pos_total_position_1']
 assert n>0 and d==2*n and accepted==p0+p1 and 0<=p1<=p0<=n
 assert abs(accepted/n-g['mean_accepted_draft_tokens_per_verification'])<1e-14
 data=json.loads((a.raw/(tag+'_responses.json')).read_text());assert len(data)==8
 for index,(saved,item) in enumerate(zip(data,inputs[name],strict=True)):
  assert saved['seed']==202610211000+100*g['round']+index and saved['source_id']==item['source_id']
  assert saved['prompt_sha256']==hashlib.sha256(json.dumps(item['prompt'],separators=(',',':')).encode()).hexdigest()
  assert saved['response']['usage']['prompt_tokens']==item['input_tokens'] and saved['response']['usage']['completion_tokens']==(16 if g['warmup'] else 128)
  assert saved['response']['choices'] and isinstance(saved['response']['choices'][0]['text'],str);responses+=1
 rows.append(dict(round=g['round'],group=name,warmup=g['warmup'],requests=8,verifications=n,proposed=d,accepted=accepted,mean_accepted_draft_tokens_per_verification=accepted/n,position0_unconditional=p0/n,position1_unconditional=p1/n,position1_conditional=p1/p0 if p0 else None))
 if not g['warmup']:
  agg=aggregates.setdefault(name,dict(verifications=0,proposed=0,accepted=0,position0=0,position1=0))
  for key,value in (('verifications',n),('proposed',d),('accepted',accepted),('position0',p0),('position1',p1)):agg[key]+=value
for item in aggregates.values():
 item['mean_accepted_draft_tokens_per_verification']=item['accepted']/item['verifications'];item['position0_unconditional']=item['position0']/item['verifications'];item['position1_unconditional']=item['position1']/item['verifications']
live=[json.loads(p.read_text()) for p in (a.raw/'adapter_records/seven_slot_input_fit').glob('*_request.json')];assert live
for x in live:assert x['physical_query_width']==7 and x['target_proposals']==2 and x['draft_sample_method']=='probabilistic' and x['rejection_sample_method']=='standard' and x['target_layers']==42 and x['draft_layers']==5 and x['auxiliary_layers']==[2,11,21,31,40] and x['lm_head_shared'] and x['target_head_dtype']==x['draft_head_dtype']=='torch.bfloat16'
threshold=[json.loads(p.read_text()) for p in (a.raw/'adapter_records/rejection_threshold').glob('*.json')];assert threshold
for x in threshold:assert x['dtype']=='torch.float32' and 0<x['minimum']<=x['maximum']<=1 and not x['rejection_kernel_modified'] and not x['synthetic_acceptance_enabled'] and not x['probability_ratio_changed']
samplers=[json.loads(p.read_text()) for p in (a.raw/'adapter_records').glob('sequential_sampler_*.json')];assert samplers
for x in samplers:assert x['backbone_block_size']==7 and x['sampled_prefix_positions']==2 and x['probabilistic'] and x['target_sampling_is_stochastic'] and x['draft_probability_representation']=='dense'
assert 'No running processes found' in (a.raw/'hardware_after.txt').read_text()
proof=dict(complete=True,CPU_reverified=True,raw_root=str(a.raw),primary_source_count=len(gate['sources']),source_inventory_scope=gate['source_inventory_scope'],CPU_gate_sha256=sha(BASE/'CPU_GATE.json'),raw_result_sha256=sha(a.raw/'client_result.json'),verified_responses=responses,official_benchmark=False,accuracy_test=False,performance_claim=False,natural_and_random_input_lengths_matched=True,natural_input_lengths=gate['input_lengths'],aggregates=aggregates,groups=rows,actual_model_runner_records=live,actual_threshold_records=threshold,standard_rejection_and_dense_proposals_verified=True,server_cleanup_and_GPU_idle_verified=True,verifier_sha256=sha(__file__))
(a.output/'ACCEPTANCE_GATE.json').write_text(json.dumps(proof,indent=2)+'\n')
with (a.output/'KEY_ACCEPTANCE.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in proof.items() if k not in ('groups','actual_model_runner_records','actual_threshold_records')}))
