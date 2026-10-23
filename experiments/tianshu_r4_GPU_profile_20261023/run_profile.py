# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,random,signal,socket,subprocess,time,traceback,urllib.request
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
s=importlib.util.spec_from_file_location('_GPU_profile_exact_r4',EXP/'tianshu_W4_LMpad_service_v2_20261023/candidate_profile.py');model=importlib.util.module_from_spec(s);s.loader.exec_module(model)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args();root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>600
 state=dict(phase='preparing',runner_pid=os.getpid(),official_benchmark=False,adopted=False,model_quality_tested=False,workloads=[]);server=None
 def save():(root/'result.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps({k:state.get(k) for k in ('phase','error')}),flush=True)
 def stop(sig,frame):raise TimeoutError(f'stop_signal={sig}')
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(sig,stop)
 signal.alarm(min(900,remaining))
 def metrics(name):
  with urllib.request.urlopen('http://127.0.0.1:9031/metrics',timeout=10) as response:(root/name).write_bytes(response.read())
 def request(length,output,input_seed,api_seed=None,cache_salt=None):
  rng=random.Random(input_seed);payload=dict(model='minicpm',prompt=[rng.randrange(512,120000) for _ in range(length)],max_tokens=output,temperature=1.,top_p=.95,ignore_eos=True,stream=True,stream_options=dict(include_usage=True))
  if api_seed is not None:payload['seed']=api_seed
  if cache_salt is not None:payload['cache_salt']=cache_salt
  req=urllib.request.Request('http://127.0.0.1:9031/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST');text='';usage=None
  with urllib.request.urlopen(req,timeout=600) as response:
   for raw in response:
    line=raw.decode().strip()
    if not line.startswith('data: ') or line=='data: [DONE]':continue
    event=json.loads(line[6:]);choices=event.get('choices',[])
    if choices:text+=choices[0].get('text','')
    if event.get('usage'):usage=event['usage']
  assert usage==dict(prompt_tokens=length,completion_tokens=output,total_tokens=length+output) and text
  return dict(usage=usage,text=text,input_seed=input_seed,api_seed=api_seed,cache_salt=cache_salt)
 try:
  command,env,profile,sources=model.prepare(root,'candidate');manifest={}
  for line in (EXP/'tianshu_W4_Down64_LM1to64_release_20261023/MODEL_SOURCES.sha256').read_text().splitlines():h,p=line.split('  ',1);manifest[p]=h
  assert sources==manifest and len(sources)==944
  profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_GPU_PROFILE_PROBE='1',TIANSHU_GPU_PROFILE_MODE='torchCUDA',TIANSHU_GPU_PROFILE_RECORD_DIR=str(root/'gpu_records'),TIANSHU_GPU_PROFILE_ARM_FILE=str(root/'profile_armed'));assert len(profile)==66;env.update(profile)
  for p in [*BASE.rglob('*.py'),BASE/'PLAN_32g.md',BASE/'EXECUTION.sha256']:sources[str(p)]=sha(p)
  model.frozen_check(sources)
  for p,h in sources.items():dest=root/'source_snapshot'/Path(p).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes());assert sha(dest)==h
  state.update(server_command=command,profile_environment=profile,additional_environment={'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']},frozen_sources=sources,original_model_sources=manifest)
  assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
  with (root/'server.log').open('w') as log:server=subprocess.Popen(command,cwd='/workspace',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state.update(phase='starting_server',server_pid=server.pid);save();start=time.monotonic()
  while True:
   assert server.poll() is None and time.monotonic()-start<1000
   try:
    with urllib.request.urlopen('http://127.0.0.1:9031/v1/models',timeout=3) as response:ready=any(m['id']=='minicpm' for m in json.load(response)['data'])
    if ready:break
   except Exception:pass
   time.sleep(3)
  observed=dict(v.decode().split('=',1) for v in Path(f'/proc/{server.pid}/environ').read_bytes().split(b'\0') if b'=' in v);full=profile|state['additional_environment'];assert len(full)==67 and {k:observed.get(k) for k in full}==full and len(command)==13;(root/'observed_full_server_environment.json').write_text(json.dumps(dict(environment=full,command=command),indent=2)+'\n')
  salt='r4-gpu-canary-first';state['canary_cold_before']=request(128,16,2026101688,777,salt)
  with ThreadPoolExecutor(max_workers=8) as pool:state['smoke']=list(pool.map(lambda i:request(128,16,202610236000+i,777),range(8)))
  for index,label in enumerate(('warmup','measured')):
   if label=='measured':(root/'profile_armed').touch()
   state['phase']=label;save()
   with ThreadPoolExecutor(max_workers=64) as pool:responses=list(pool.map(lambda i:request(4096,256,202610230000+1000*index+i),range(64)))
   state['workloads'].append(dict(label=label,responses=responses));save()
  records=list((root/'gpu_records').glob('gpu_*.json'));assert len(records)==1;record=json.loads(records[0].read_text());assert record['complete'] and record['frame_count']==len(record['frames'])==16 and record['GPU_events_used'] and record['GPU_kernel_events']>0 and record['hook_source_sha256']==sha(BASE/'hooks/sitecustomize.py');assert sha(record['trace_file'])==record['trace_sha256'];state['GPU_profile_record']=str(records[0]);state['GPU_trace_file']=record['trace_file']
  metrics('before_reused_salt_metrics.txt');state['canary_reused_salt_after']=request(128,16,2026101688,777,salt);metrics('after_reused_salt_metrics.txt');state['canary_cold_after']=request(128,16,2026101688,777,'r4-gpu-canary-fresh-second');metrics('after_fresh_salt_metrics.txt')
  state['same_seed_cold_cache_text_equal']=state['canary_cold_before']['text']==state['canary_cold_after']['text'];state['same_seed_reused_salt_text_equal']=state['canary_cold_before']['text']==state['canary_reused_salt_after']['text'];model.frozen_check(sources);state.update(phase='complete',complete=True,GPU_frames=16,GPU_kernel_events=record['GPU_kernel_events'],diagnostic_only=True,no_benchmark_or_model_qualification=True);save()
 except BaseException as e:state.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
 finally:
  signal.alarm(0)
  if server is not None and server.poll() is None:
   os.killpg(server.pid,signal.SIGTERM)
   try:server.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=5)
  for _ in range(30):
   hw=subprocess.check_output(['ixsmi'],text=True);(root/'hardware_after.txt').write_text(hw)
   if 'No running processes found' in hw:break
   time.sleep(1)
  save()
if __name__=='__main__':main()
