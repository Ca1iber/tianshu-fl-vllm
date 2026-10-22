# SPDX-License-Identifier: Apache-2.0
"""Profile a proven exact fixture; CUDA profiler boundaries exclude preparation."""
import argparse,csv,hashlib,importlib.util,inspect,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
EXPERIMENTS={'W8_channel':('tianshu_w8a16_nt_postscale_20261022','kernel_nt_postscale.py'),'W4_aff128':('tianshu_w4a16_nt_group_postscale_20261022','kernel_nt_postscale.py')}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--format',choices=EXPERIMENTS,required=True);p.add_argument('--role',choices=('down','lm_head'),required=True);p.add_argument('--m',type=int,choices=(8,64),required=True);p.add_argument('--path',choices=('candidate','BF16','W8A8'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args()
 remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>900;signal.alarm(min(900,remain))
 folder,kfile=EXPERIMENTS[args.format];experiment=ROOT/folder;gate=json.loads((experiment/'INDEPENDENT_CPU_GATE_32g.json').read_text());assert gate['complete'];raw=Path(gate['raw_directory']);assert sha(raw/'result.json')==gate['raw_result_sha256'];state=json.loads((raw/'result.json').read_text());assert state['complete']
 own=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text());assert own['complete'];sources={**state['frozen_sources'],**own['sources'],str(experiment/'INDEPENDENT_CPU_GATE_32g.json'):sha(experiment/'INDEPENDENT_CPU_GATE_32g.json')};assert all(sha(f)==h for f,h in sources.items())
 timings=list(csv.DictReader((experiment/'KEY_TIMINGS_32g.csv').open()));assert sha(experiment/'KEY_TIMINGS_32g.csv')==gate['key_timings_sha256'];best=min([r for r in timings if r['role']==args.role and r['format']==args.format and int(r['M'])==args.m],key=lambda r:float(r['candidate_ms']));case=next(c for c in state['cases'] if (c['role'],c['format'],c['M'],c['tile_id'])==(args.role,args.format,args.m,int(best['tile_id'])));record=case['phases'][2];assert sha(raw/record['artifact'])==record['sha256']
 root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
 for f,h in sources.items():
  target=root/'source_snapshot'/Path(f).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(f).read_bytes());assert sha(target)==h
 result=dict(complete=False,phase='prepare',runner_pid=os.getpid(),format=args.format,role=args.role,M=args.m,path=args.path,sources=sources,raw_reference=str(raw),reference_artifact=record['artifact'],reference_sha256=record['sha256'],configuration=case['configuration'],component_graph_median_ms=best,official_benchmark=False,adopted=False,model_executed=False)
 def save():
  p=root/'result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(p)
 save()
 try:
  import torch,vllm_fl,flag_gems
  from safetensors import safe_open
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;assert getattr(flag_gems,'current_work_registrar',None) is None
  result["actual_BF16_callable"]={"module":flag_gems.mm_out.__module__,"qualname":flag_gems.mm_out.__qualname__,"source":inspect.getsourcefile(flag_gems.mm_out)};save()
  fmt=load('_ixkn_formats',ROOT/'tianshu_quantization_survey_20261022/formats_cpu.py');kernel=load('_ixkn_candidate',experiment/kfile);nt=load('_ixkn_nt',ROOT/'tianshu_dense_w8a8_20261020/kernel.py')
  with np.load(raw/record['artifact'],allow_pickle=False) as src:fixture={key:src[key].copy() for key in src.files}
  a=torch.from_numpy(fixture['A_words']).view(torch.bfloat16).cuda();m,k=a.shape;n=case['N'];out=torch.empty((m,n),dtype=torch.bfloat16,device='cuda');flo=torch.empty((m,n),dtype=torch.float32,device='cuda')
  name='lm_head.weight' if args.role=='lm_head' else 'model.layers.0.mlp.down_proj.weight'
  with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as checkpoint:wc=checkpoint.get_tensor(name).contiguous()
  assert hashlib.sha256(wc.view(torch.uint16).numpy().tobytes()).hexdigest()==state['weights'][args.role]['BF16_sha256']
  if args.path=='BF16':
   weight=(wc*(-.5)).cuda()
   def operation():return flag_gems.mm_out(a,weight.T,out=out)
  else:
   bits,group,aff=(4,128,True) if args.path=='candidate' and args.format=='W4_aff128' else (8,0,False);ng=(k+(group or k)-1)//(group or k);q=np.empty((n,k),np.int8);s=np.empty((n,ng),np.float32);z=np.empty((n,ng),np.uint8)
   for start in range(0,n,128):
    block=fmt.bf16_from_words(wc[start:start+128].view(torch.uint16).numpy());qq,ss,zz,_,_=fmt.quantize(block,bits,group,aff);q[start:start+128]=qq;s[start:start+128]=ss;z[start:start+128]=zz
   q=(15-q) if aff else -q;z=(15-z) if aff else z;s=s*.5
   if args.path=='candidate':
    packed=fmt.pack_nibbles_nn(q.T,aff) if bits==4 else q.view(np.uint8);weight=torch.from_numpy(np.ascontiguousarray(packed)).cuda();scales=torch.from_numpy(s.T.copy()).cuda();zero=torch.from_numpy(z.T.copy()).cuda()
    def operation():return kernel.matmul(a,weight,scales,zero,out,flo,case['configuration'],bits,group,aff)
   else:
    weight=torch.from_numpy(q).cuda();scales=torch.from_numpy(s[:,0].copy()).cuda();aq=torch.empty((m,k),dtype=torch.int8,device='cuda');asc=torch.empty((m,),dtype=torch.float32,device='cuda');integer=torch.empty((m,n),dtype=torch.int32,device='cuda');plans=json.loads((ROOT/'tianshu_dense_w8a8_20261020/GPU_GATE_32g.json').read_text())['best_by_case'];config=next(r['configuration'] for r in plans if (r['case'],r['M'])==('lm_head' if args.role=='lm_head' else 'down',m))
    def operation():nt.quantize(a,aq,asc);return nt.mm(aq,weight,asc,scales,out,integer,config)
  for _ in range(5):compiled=operation()
  torch.cuda.synchronize();rows=torch.from_numpy(fixture['rows']).cuda();columns=torch.from_numpy(fixture['columns']).cuda()
  def verify():
   if args.path=='candidate':assert np.array_equal(out.view(torch.uint16).cpu().numpy(),fixture['full_BF16_words'])
   elif args.path=='BF16':
    actual=out[rows[:,None],columns[None,:]].float().cpu().numpy();reference=fixture['native_selected'];assert np.all(abs(actual-reference)<=.02+.01*abs(reference));result['native_selected_max_difference']=float(abs(actual-reference).max())
   else:assert np.array_equal(out[rows[:,None],columns[None,:]].float().cpu().numpy(),fixture['W8A8_output_selected'])
  verify()
  if args.path=='W8A8':
   nt.mm(aq,weight,asc,scales,out,integer,config,True);torch.cuda.synchronize();assert np.array_equal(integer[rows[:,None],columns[None,:]].cpu().numpy(),fixture['W8A8_integer_selected']);assert np.array_equal(aq.cpu().numpy(),fixture['Aq']) and np.array_equal(asc.cpu().numpy(),fixture['As'])
  if args.path!='BF16':result['compiled']=dict(hash=compiled.hash,binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared=compiled.metadata.shared)
  for _ in range(5):operation()
  torch.cuda.synchronize();result.update(phase='profiling',pre_profile_output_verified=True);save()
  torch.cuda.profiler.start();operation();torch.cuda.synchronize();torch.cuda.profiler.stop()
  verify();assert all(sha(f)==h for f,h in sources.items());result.update(complete=True,phase='complete',post_profile_output_verified=True,actual_kernel_analysis_requires_external_ixkn_report=True);save();print(json.dumps(result),flush=True)
 except BaseException as error:result.update(phase='failed',error=repr(error));save();raise
if __name__=='__main__':main()
