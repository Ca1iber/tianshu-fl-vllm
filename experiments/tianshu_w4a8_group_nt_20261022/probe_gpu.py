# SPDX-License-Identifier: Apache-2.0
"""Frozen packed weight-only component experiment; model checkpoint stays read-only."""
import argparse,gc,hashlib,importlib.util,json,os,signal,statistics,struct,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
SURVEY=ROOT/'tianshu_quantization_survey_20261022';NT=ROOT/'tianshu_dense_w8a8_20261020'
MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def save(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args()
 remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1800;signal.alarm(min(1800,remaining))
 root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
 hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
 manifest=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text());assert manifest['complete'];frozen=manifest['sources'];assert all(sha(p)==v for p,v in frozen.items())
 for p,v in frozen.items():
  target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==v
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),started_at_utc=datetime.now(timezone.utc).isoformat(),frozen_sources=frozen,checkpoint_sha256_before=sha(MODEL),model_executed=False,official_benchmark=False,adopted=False,cases=[],decodes=[],timings=[])
 def checkpoint():save(root/'result.json',state)
 checkpoint()
 try:
  import torch,vllm_fl,flag_gems
  from safetensors import safe_open
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;assert getattr(flag_gems,'current_work_registrar',None) is None
  fmt=load('_quant_formats',SURVEY/'formats_cpu.py');kernel=load('_quant_weight',BASE/'kernel_w4a8.py');nt=load('_quant_nt',NT/'kernel.py')
  state['versions']=dict(torch=torch.__version__,triton=__import__('triton').__version__,target=str(__import__('triton').runtime.driver.active.get_current_target()));checkpoint()
  def tensor(a):return torch.from_numpy(np.ascontiguousarray(a)).cuda()
  def bf_cpu(a):return torch.from_numpy(np.ascontiguousarray((a.view(np.uint32)>>16).astype(np.uint16))).view(torch.bfloat16)
  def cpu_words(a):return a.detach().contiguous().view(torch.uint16).cpu().numpy().copy()
  def guarded(shape,dtype,value):
   count=int(np.prod(shape));store=torch.full((count+512,),value,dtype=dtype,device='cuda');view=store[256:256+count].view(shape);assert view.data_ptr()%256==0;return [store,view,value]
  def guards(buffers):
   for store,view,v in buffers:assert bool((store[:256]==v).all() and (store[-256:]==v).all())
  def meta(c):return dict(hash=c.hash,binary_sha256=hashlib.sha256(c.kernel).hexdigest(),n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared)
  def timing(graph):
   start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
   for _ in range(20):graph.replay()
   end.record();end.synchronize();return start.elapsed_time(end)/20
  def prepare(w,bits,group,aff):
   n,k=w.shape;g=group or k;ng=(k+g-1)//g;q=np.empty((n,k),np.int8);s=np.empty((n,ng),np.float32);z=np.empty((n,ng),np.uint8);d=np.empty((n,k),np.float32)
   for start in range(0,n,128):
    a,b,c,e,_=fmt.quantize(w[start:start+128],bits,group,aff);q[start:start+128]=a;s[start:start+128]=b;z[start:start+128]=c;d[start:start+128]=e
   return q,s,z,d
  def candidate_mm(a,w,scales,zero,out,flo,config,bits,group,aff,return_float=False):
   aq,asc=q_buffers[tuple(a.shape)];nt.quantize(a,aq,asc)
   return kernel.mm(aq,w,scales,zero,asc,out,flo,integer_buffers[out.data_ptr()],config,group,aff,return_float)
  with safe_open(str(MODEL),framework='pt',device='cpu') as checkpoint_model:
   weights={role:checkpoint_model.get_tensor(name).contiguous() for role,name in [('down','model.layers.0.mlp.down_proj.weight')]}
  tail=fmt.bf16_round(np.sin(np.arange(131*259,dtype=np.float32)*.19).reshape(131,259)*.1);tail[0]=0
  weights['tail']=bf_cpu(tail)
  plans={(r['case'],r['M']):r['configuration'] for r in json.loads((NT/'GPU_GATE_32g.json').read_text())['best_by_case']}
  for role in ('tail','down'):
   wc=weights[role];w=fmt.bf16_from_words(wc.view(torch.uint16).numpy());n,k=w.shape;state.update(phase=f'{role}_prepare');checkpoint()
   baseq,bases,_,_=prepare(w,8,0,False);baseq_gpu=guarded((n,k),torch.int8,17);bases_gpu=guarded((n,),torch.float32,17.);bfweight=guarded((n,k),torch.bfloat16,17.);basebuffers=[baseq_gpu,bases_gpu,bfweight]
   state.setdefault('weights',{})[role]=dict(N=n,K=k,BF16_sha256=digest(wc.view(torch.uint16).numpy()),original_bytes=int(n*k*2));checkpoint()
   for label,bits,group,aff in [('W4_sym128',4,128,False),('W4_aff128',4,128,True)]:
    q_buffers={};integer_buffers={}
    q,s,z,decoded=prepare(w,bits,group,aff);ng=s.shape[1];pw=guarded(((n,(k+1)//2) if bits==4 else (k,n)),torch.uint8,17);ps=guarded((ng,n),torch.float32,17.);pz=guarded((ng,n),torch.uint8,17);decoded_gpu=guarded((n,k),torch.bfloat16,17.);buffers=basebuffers+[pw,ps,pz,decoded_gpu]
    unrounded=np.empty((n,k),np.float32)
    for start in range(0,n,128):unrounded[start:start+128]=(q[start:start+128].astype(np.float32)-z[start:start+128,np.arange(k)//group].astype(np.float32))*s[start:start+128,np.arange(k)//group]
    unrounded_gpu=tensor(unrounded)
    prepared=[]
    for m in ((17,) if role=='tail' else (64,)):
     rng=np.random.default_rng(202610220000+m+k);ac=fmt.bf16_round(rng.standard_normal((m,k),dtype=np.float32));ac[0]=0;ac[1]=0;ac[1,:9]=[0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127]
     abuf=guarded((m,k+16 if role=='tail' else k),torch.bfloat16,17.);a=abuf[1][:,:k];buffers.append(abuf)
     bout=guarded((m,n),torch.bfloat16,17.);aq=guarded((m,k),torch.int8,17);asc=guarded((m,),torch.float32,17.);iout=guarded((m,n),torch.int32,-777);qout=guarded((m,n),torch.bfloat16,17.);buffers.extend([bout,aq,asc,iout,qout])
     cfgnt=plans[('tail',17)] if role=='tail' else plans[('lm_head' if role=='lm_head' else 'down',m)]
     q_buffers[(m,k)]=(aq[1],asc[1]);a.copy_(bf_cpu(ac));bfweight[1].copy_(wc);baseq_gpu[1].copy_(torch.from_numpy(baseq));bases_gpu[1].copy_(torch.from_numpy(bases[:,0]));ps[1].copy_(torch.from_numpy(s.T.copy()));pz[1].copy_(torch.from_numpy(z.T.copy()));pw[1].copy_(torch.from_numpy(fmt.pack_nibbles_nn(q.T,aff) if bits==4 else q.T.copy().view(np.uint8)))
     for _ in range(3):flag_gems.mm_out(a,bfweight[1].T,out=bout[1]);nt.quantize(a,aq[1],asc[1]);nt.mm(aq[1],baseq_gpu[1],asc[1],bases_gpu[1],qout[1],iout[1],cfgnt)
     bg=torch.cuda.CUDAGraph();ig=torch.cuda.CUDAGraph()
     with torch.cuda.graph(bg):flag_gems.mm_out(a,bfweight[1].T,out=bout[1])
     with torch.cuda.graph(ig):nt.quantize(a,aq[1],asc[1]);nt.mm(aq[1],baseq_gpu[1],asc[1],bases_gpu[1],qout[1],iout[1],cfgnt)
     tiles=json.loads((BASE/'tiles.json').read_text())
     variants=[]
     for tile_id,config in enumerate(tiles):
      out=guarded((m,n),torch.bfloat16,17.);flo=guarded((m,n),torch.float32,17.);ibuf=guarded(((k+config['bk']-1)//config['bk'],m,n),torch.int32,-777);buffers.extend([out,flo,ibuf]);integer_buffers[out[1].data_ptr()]=ibuf[1]
      row=dict(role=role,format=label,M=m,N=n,K=k,tile_id=tile_id,configuration=config,phases=[],complete=False);state['cases'].append(row);checkpoint()
      try:
       for _ in range(3):compiled=candidate_mm(a,pw[1],ps[1],pz[1],out[1],flo[1],config,bits,group,aff)
       diagnostic=candidate_mm(a,pw[1],ps[1],pz[1],out[1],flo[1],config,bits,group,aff,True);graph=torch.cuda.CUDAGraph()
       with torch.cuda.graph(graph):candidate_mm(a,pw[1],ps[1],pz[1],out[1],flo[1],config,bits,group,aff)
       row.update(compiled=meta(compiled),diagnostic_compiled=meta(diagnostic));variants.append(dict(out=out[1],flo=flo[1],integer=ibuf[1],graph=graph,row=row,config=config))
      except Exception as error:
       if type(error).__name__ not in ('OutOfResources','CompilationError'):raise
       row.update(compile_failed=True,error=repr(error));checkpoint()
     assert variants
     prepared.append(dict(m=m,ac=ac,a=a,abuf=abuf,bg=bg,ig=ig,bout=bout[1],aq=aq[1],asc=asc[1],qout=qout[1],iout=iout[1],cfgnt=cfgnt,variants=variants))
    for phase in range(3):
     sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.;qphase=q if phase==0 else ((15-q) if aff else -q);zphase=z if phase==0 or not aff else 15-z
     packed=fmt.pack_nibbles_nn(qphase.T,aff) if bits==4 else np.ascontiguousarray(qphase.T).view(np.uint8);scales=np.ascontiguousarray((s*factor).T);zeros=np.ascontiguousarray(zphase.T)
     pw[1].copy_(torch.from_numpy(packed));ps[1].copy_(torch.from_numpy(scales));pz[1].copy_(torch.from_numpy(zeros));baseq_gpu[1].copy_(torch.from_numpy(baseq*sign));bases_gpu[1].copy_(torch.from_numpy(bases[:,0]*factor));bfweight[1].copy_(wc*sign*factor)
     unrounded_gpu.copy_(torch.from_numpy(unrounded*sign*factor));kernel.decode(pw[1],ps[1],pz[1],decoded_gpu[1],bits,group,aff);torch.cuda.synchronize();actual_decoded=cpu_words(decoded_gpu[1]);expected_words=(decoded.view(np.uint32)>>16).astype(np.uint16);expected_words=((fmt.bf16_round(decoded*sign*factor).view(np.uint32)>>16).astype(np.uint16))
     expected_words[(expected_words&0x7fff)==0]=0
     assert np.array_equal(actual_decoded,expected_words)
     state['decodes'].append(dict(role=role,format=label,phase=phase,elements=n*k,actual_BF16_sha256=digest(actual_decoded),expected_BF16_sha256=digest(expected_words),bitwise_equal=True));del actual_decoded,expected_words
     for data in prepared:
      m=data['m'];a=data['a'];actual_a=data['ac']*(1. if phase==0 else .5 if phase==1 else -.5);a.copy_(bf_cpu(actual_a));rngcpu=torch.get_rng_state().clone();rnggpu=torch.cuda.get_rng_state().clone();data['bg'].replay();data['ig'].replay();nt.mm(data['aq'],baseq_gpu[1],data['asc'],bases_gpu[1],data['qout'],data['iout'],data['cfgnt'],True)
      reference=(data['aq'].float()@unrounded_gpu.T)*data['asc'][:,None];native_reference=(a.float()@bfweight[1].float().T).to(torch.bfloat16);native_error=(data['bout'].float()-native_reference.float()).abs();assert bool(torch.isfinite(data['bout']).all() and not(native_error>(.02+.01*native_reference.float().abs())).any())
      rows=np.arange(m) if role=='tail' else np.unique(np.linspace(0,m-1,min(m,8)).astype(np.int64));cols=np.arange(n) if role=='tail' else np.unique(np.linspace(0,n-1,32).astype(np.int64));rd=tensor(rows);cd=tensor(cols)
      for variant in data['variants']:
       variant['graph'].replay();replayed=variant['out'].clone();candidate_mm(a,pw[1],ps[1],pz[1],variant['out'],variant['flo'],variant['config'],bits,group,aff,True);torch.cuda.synchronize()
       assert torch.equal(replayed.view(torch.int16),variant['out'].view(torch.int16)) and torch.equal(variant['out'].view(torch.int16),variant['flo'].to(torch.bfloat16).view(torch.int16));error=(variant['flo']-reference).abs();assert bool(torch.isfinite(variant['flo']).all() and not(error>(.002+.002*reference.abs())).any())
       words=cpu_words(variant['out']);flo=variant['flo'].cpu().numpy();filename=f'{role}_{label}_M{m}_tile{variant["row"]["tile_id"]}_phase{phase}.npz'
       np.savez(root/filename,A_words=cpu_words(a),full_BF16_words=words,full_FP32=flo,W4_partial_INT32_selected=variant['integer'][:,rd[:,None],cd[None,:]].cpu().numpy(),rows=rows,columns=cols,native_selected=data['bout'][rd[:,None],cd[None,:]].float().cpu().numpy(),W8A8_integer_selected=data['iout'][rd[:,None],cd[None,:]].cpu().numpy(),W8A8_output_selected=data['qout'][rd[:,None],cd[None,:]].float().cpu().numpy(),Aq=data['aq'].cpu().numpy(),As=data['asc'].cpu().numpy())
       variant['row']['phases'].append(dict(phase=phase,artifact=filename,sha256=sha(root/filename),full_GPU_FP32_max_error=float(error.max()),full_BF16_FP32_cast_equal=True,captured_equals_diagnostic=True,reference_full_passed=True,native_full_passed=True));del replayed,words,flo,error
      assert np.array_equal(cpu_words(a),(actual_a.view(np.uint32)>>16).astype(np.uint16));assert torch.equal(torch.get_rng_state(),rngcpu) and torch.equal(torch.cuda.get_rng_state(),rnggpu)
      if role=='tail':assert bool((data['abuf'][1][:,k:]==17).all())
      del reference,native_reference,native_error
     assert np.array_equal(pw[1].cpu().numpy(),packed) and np.array_equal(ps[1].cpu().numpy(),scales) and np.array_equal(pz[1].cpu().numpy(),zeros);guards(buffers);checkpoint()
    for data in prepared:
     graphs={'BF16':data['bg'],'W8A8':data['ig'],**{f'tile{v["row"]["tile_id"]}':v['graph'] for v in data['variants']}};names=list(graphs)
     for roundid in range(7):
      order=names[roundid%len(names):]+names[:roundid%len(names)];order=order[::-1] if roundid%2 else order
      times={name:timing(graphs[name]) for name in order};state['timings'].append(dict(role=role,format=label,M=data['m'],round=roundid,order=order,replays=20,milliseconds=times))
     for v in data['variants']:v['row'].update(complete=True,guards=True,RNG_unchanged=True,packed_inputs_immutable=True)
    guards(buffers);checkpoint();print(json.dumps(dict(role=role,format=label,phase='complete',cases=len(state['cases']))),flush=True)
    del prepared,data,graphs,variants,variant,v,buffers,pw,ps,pz,decoded_gpu,q,s,z,decoded,packed,scales,zeros,unrounded,unrounded_gpu
    gc.collect();torch.cuda.empty_cache()
   del w,wc,baseq,bases,baseq_gpu,bases_gpu,bfweight,basebuffers;gc.collect();torch.cuda.empty_cache()
  assert all(sha(p)==v for p,v in frozen.items());state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_after']==state['checkpoint_sha256_before'];assert all(c['complete'] or c.get('compile_failed') for c in state['cases'])
  state.update(complete=True,phase='complete',checkpoint_unchanged=True,ended_at_utc=datetime.now(timezone.utc).isoformat());checkpoint()
 except BaseException as error:state.update(phase='failed',error=repr(error));checkpoint();raise
 finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
