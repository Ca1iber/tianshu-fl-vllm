# SPDX-License-Identifier: Apache-2.0
"""Short natural/random acceptance diagnostics; no throughput or accuracy claim."""
import argparse,concurrent.futures,hashlib,json,re,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
BASE=Path(__file__).resolve().parent;gate=json.loads((BASE/'CPU_GATE.json').read_text());inputs=Path(gate['inputs_path']);assert hashlib.sha256(inputs.read_bytes()).hexdigest()==gate['inputs_sha256'];groups=json.loads(inputs.read_text())
state=dict(complete=False,phase='requesting',official_benchmark=False,accuracy_test=False,new_speculative_performance_score=False,concurrency=8,output_tokens=128,temperature=1.0,top_p=.95,ignore_eos=True,groups=[])
def save():
 tmp=root/'client_result.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'client_result.json')
def metrics():
 with urllib.request.urlopen('http://127.0.0.1:9031/metrics',timeout=10) as response:text=response.read().decode()
 counters={}
 for line in text.splitlines():
  if not line.startswith('vllm:spec_decode_num_'):continue
  match=re.match(r'([^\{]+)\{([^}]+)\}\s+(\S+)',line)
  if not match or not match[1].endswith('_total'):continue
  position=re.search(r'position="(\d+)"',match[2]);key=match[1]+('_position_'+position[1] if position else '')
  counters[key]=float(match[3])
 return text,counters
def settled():
 for _ in range(100):
  text,counters=metrics();busy=[]
  for line in text.splitlines():
   if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')):busy.append(float(line.rsplit(' ',1)[1]))
  if busy and all(x==0 for x in busy):return text,counters
  time.sleep(.1)
 raise RuntimeError('Engine did not become idle after completed requests')
def request(item,seed,output):
 body=dict(model='minicpm',prompt=item['prompt'],max_tokens=output,temperature=1.0,top_p=.95,ignore_eos=True,seed=seed,stream=False)
 req=urllib.request.Request('http://127.0.0.1:9031/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 start=time.monotonic()
 with urllib.request.urlopen(req,timeout=300) as response:data=json.load(response)
 assert data['usage']['prompt_tokens']==item['input_tokens'] and data['usage']['completion_tokens']==output,data
 assert data['choices'] and isinstance(data['choices'][0]['text'],str)
 return dict(source_id=item['source_id'],seed=seed,prompt_tokens=item['input_tokens'],prompt_sha256=hashlib.sha256(json.dumps(item['prompt'],separators=(',',':')).encode()).hexdigest(),response=data,seconds=time.monotonic()-start)
def group(name,round_number,output,warmup=False):
 before_text,before=settled();log_start=(root/'server.log').stat().st_size
 start=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  rows=list(pool.map(lambda pair:request(pair[1],202610211000+100*round_number+pair[0],output),enumerate(groups[name])))
 duration=time.monotonic()-start;after_text,after=settled()
 tag=f"{'warmup' if warmup else 'round'+str(round_number)}_{name}"
 (root/(tag+'_metrics_before.txt')).write_text(before_text);(root/(tag+'_metrics_after.txt')).write_text(after_text)
 (root/(tag+'_responses.json')).write_text(json.dumps(rows,ensure_ascii=False)+'\n')
 delta={k:after[k]-before.get(k,0) for k in after};n=delta['vllm:spec_decode_num_drafts_total'];accepted=delta['vllm:spec_decode_num_accepted_tokens_total'];proposed=delta['vllm:spec_decode_num_draft_tokens_total'];pos0=delta['vllm:spec_decode_num_accepted_tokens_per_pos_total_position_0'];pos1=delta['vllm:spec_decode_num_accepted_tokens_per_pos_total_position_1']
 assert n>0 and proposed==2*n and accepted==pos0+pos1 and 0<=pos1<=pos0<=n
 with (root/'server.log').open('rb') as f:f.seek(log_start);segment=f.read().decode(errors='replace')
 tuning=[x for x in segment.splitlines() if 'Tuned shape' in x or 'Best Config' in x or 'tuning' in x.lower()]
 row=dict(name=name,round=round_number,warmup=warmup,requests=8,input_tokens=[x['input_tokens'] for x in groups[name]],output_tokens_per_request=output,seconds=duration,counter_deltas=delta,verifications=n,proposed_tokens=proposed,accepted_tokens=accepted,mean_accepted_draft_tokens_per_verification=accepted/n,position0_unconditional_acceptance=pos0/n,position1_unconditional_acceptance=pos1/n,position1_conditional_acceptance=pos1/pos0 if pos0 else None,autotune_log_lines=tuning,formal_throughput_comparison=False)
 state['groups'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k not in ('counter_deltas','autotune_log_lines','input_tokens')}),flush=True)
save();(root/'READY_FOR_REQUESTS').touch()
try:
 for name in groups:group(name,0,16,True)
 for number,order in enumerate(gate['round_orders'],1):
  for name in order:group(name,number,128)
 hits=list((root/'adapter_records/seven_slot_input_fit').glob('*_request.json'));assert hits,'No actual seven-slot model-runner call recorded'
 live=[json.loads(p.read_text()) for p in hits]
 assert all(x['physical_query_width']==7 and x['target_proposals']==2 and x['draft_sample_method']=='probabilistic' and x['rejection_sample_method']=='standard' and x['target_layers']==42 and x['draft_layers']==5 and x['auxiliary_layers']==[2,11,21,31,40] and x['lm_head_shared'] for x in live),live
 state.update(complete=True,phase='complete',actual_model_runner_records=live)
except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
save()
