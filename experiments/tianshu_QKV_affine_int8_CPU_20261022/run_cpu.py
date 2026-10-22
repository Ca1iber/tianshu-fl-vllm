# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/workspace/benchmark-results/20261022_QKV_affine_int8_CPU_tianshu');MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');K=2048

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def from_words(w):return (np.asarray(w,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def to_words(x):
 b=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32);return ((b+32767+((b>>16)&1))>>16).astype(np.uint16)
def quant(x,affine):
 if affine:
  lo=x.min(-1);scale=np.maximum(x.max(-1)-lo,np.float32(1.e-8))/np.float32(255.);zero=np.clip(np.rint(np.float32(-128.)-lo/scale),-128,127).astype(np.int32);code=np.clip(np.rint(x/scale[...,None])+zero[...,None],-128,127).astype(np.int8)
 else:
  scale=np.maximum(np.abs(x).max(-1),np.float32(1.e-8))/np.float32(127.);zero=np.zeros(scale.shape,np.int32);code=np.clip(np.rint(x/scale[...,None]),-127,127).astype(np.int8)
 return code,scale,zero

def main():
 import torch
 from safetensors import safe_open
 torch.set_num_threads(2);assert not torch.cuda.is_initialized() and not ROOT.exists();ROOT.mkdir();release=json.loads((EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json').read_text());sources=dict(release['frozen_sources']);assert len(sources)==903
 plan=EXP/'tianshu_native_int8_attention_service_20261022/selected_plan.json'
 for p in (BASE/'run_cpu.py',BASE/'PLAN_32g.md',plan):sources[str(p)]=sha(p)
 assert all(sha(p)==h for p,h in sources.items());checkpoint=sha(MODEL);assert checkpoint=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
 for p,h in sources.items():
  dest=ROOT/'source_snapshot'/Path(p).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes());assert sha(dest)==h
 expected={b['layer']:b for b in json.loads(plan.read_text())['banks'] if b['role']=='QKV'};assert set(expected)==set(range(42));rows=np.unique(np.linspace(0,63,8).astype(np.int64));cols=np.concatenate([np.linspace(0,2047,64).astype(np.int64),2048+np.linspace(0,255,16).astype(np.int64),2304+np.linspace(0,255,16).astype(np.int64)]);assert len(cols)==len(set(cols))==96
 fixtures=[]
 for phase in range(2):
  sample=np.random.default_rng(202610241000+phase).standard_normal((64,K),dtype=np.float32);words=to_words(sample);words[0]=0;x=from_words(words);fixtures.append(dict(words=words,x=x,symmetric=quant(x,False),affine=quant(x,True)))
 np.savez(ROOT/'fixtures.npz',rows=rows,columns=cols,A0_words=fixtures[0]['words'],A1_words=fixtures[1]['words']);modes={'symmetric_A_symmetric_W':(False,False),'symmetric_A_affine_W':(False,True),'affine_A_symmetric_W':(True,False),'affine_A_affine_W':(True,True)};totals={m:dict(error2=0.,reference2=0.,maxabs=0.) for m in modes};results=[];banks=[];cells=0
 with safe_open(str(MODEL),framework='pt',device='cpu') as cp:
  for layer in range(42):
   weight=torch.cat([cp.get_tensor(f'model.layers.{layer}.self_attn.{part}_proj.weight') for part in ('q','k','v')],0).contiguous();words=weight.view(torch.uint16).numpy().copy();w=from_words(words);sq,ss,sz=quant(w,False);aq,asc,az=quant(w,True);proof=expected[layer];assert w.shape==(2560,K) and ah(words)==proof['original_BF16_sha256'] and ah(sq)==proof['packed_sha256'] and ah(ss)==proof['scales_sha256'];weight_error={}
   for name,q,sc,z in [('symmetric',sq,ss,sz),('affine',aq,asc,az)]:
    diff=(q.astype(np.float32)-z[:,None])*sc[:,None]-w;weight_error[name]=float(np.linalg.norm(diff.astype(np.float64))/np.linalg.norm(w.astype(np.float64)))
   arrays=dict(columns=cols,selected_weight_words=words[cols],symmetric_Q=sq[cols],symmetric_S=ss[cols],affine_Q=aq[cols],affine_S=asc[cols],affine_Z=az[cols])
   for phase,fixture in enumerate(fixtures):
    reference=fixture['x'][rows].astype(np.float64)@w[cols].astype(np.float64).T
    for mode,(a_aff,w_aff) in modes.items():
     qa,sa,za=fixture['affine' if a_aff else 'symmetric'];qw,sw,zw=(aq,asc,az) if w_aff else (sq,ss,sz);qa=qa[rows].astype(np.int64);qw=qw[cols].astype(np.int64);za=za[rows].astype(np.int64);zw=zw[cols].astype(np.int64);raw=qa@qw.T;term_a=za[:,None]*qw.sum(1)[None,:];term_w=qa.sum(1)[:,None]*zw[None,:];term_cross=K*za[:,None]*zw[None,:];corrected=raw-term_a-term_w+term_cross;direct=(qa-za[:,None])@(qw-zw[:,None]).T;assert np.array_equal(corrected,direct)
     for v in (raw,term_a,term_w,term_cross,raw-term_a,raw-term_a-term_w,corrected):assert np.abs(v).max()<2**31
     projected=(corrected.astype(np.float32)*sa[rows,None])*sw[cols][None,:];out_words=to_words(projected);output=from_words(out_words).astype(np.float64);assert not corrected[0].any() and not output[0].any();delta=output-reference;e2=float(np.square(delta).sum());r2=float(np.square(reference).sum());maxabs=float(np.abs(delta).max());totals[mode]['error2']+=e2;totals[mode]['reference2']+=r2;totals[mode]['maxabs']=max(totals[mode]['maxabs'],maxabs);cells+=corrected.size;results.append(dict(layer=layer,phase=phase,mode=mode,relative_L2=(e2/r2)**.5,max_abs=maxabs,INT64_direct_centered_exact=True,INT32_all_intermediates_safe=True,zero_row_exact=True));arrays[f'{phase}_{mode}_INT32']=corrected.astype(np.int32);arrays[f'{phase}_{mode}_out_words']=out_words
   artifact=ROOT/f'layer{layer}.npz';np.savez(artifact,**arrays);banks.append(dict(layer=layer,original_BF16_sha256=ah(words),original_symmetric_pack_sha256=ah(sq),affine_pack_sha256=ah(aq),affine_scale_sha256=ah(asc),affine_zero_sha256=ah(az),weight_relative_L2=weight_error,artifact=artifact.name,artifact_sha256=sha(artifact)));print(json.dumps(dict(layer=layer,weight_relative_L2=weight_error)),flush=True)
 for v in totals.values():v['relative_L2']=(v['error2']/v['reference2'])**.5
 baseline=totals['symmetric_A_symmetric_W']['relative_L2'];improvement={m:100*(1-v['relative_L2']/baseline) for m,v in totals.items() if m!='symmetric_A_symmetric_W'};best=min(totals,key=lambda m:totals[m]['relative_L2']);assert sha(MODEL)==checkpoint and all(sha(p)==h for p,h in sources.items()) and not torch.cuda.is_initialized()
 with (BASE/'KEY_ERRORS_32g.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
 gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,original42_QKV_BF16_and_symmetric_pack_full_hash_exact=True,source_count=len(sources),frozen_sources=sources,banks=banks,phases=2,seeds=[202610241000,202610241001],rows_per_phase=len(rows),columns_per_bank=len(cols),INT64_direct_centered_checked_cells=cells,INT32_centered_absolute_bound=K*255**2,summary=totals,relative_error_reduction_percent=improvement,best_CPU_scheme=best,CPU_precision_lead_gt5pct=best!='symmetric_A_symmetric_W' and improvement[best]>5,actual_model_activations=False,calibration_used=False,GPU_not_run=True,performance_unknown=True,model_quality_unknown=True,adopted=False,checkpoint_sha256=checkpoint,key_errors_sha256=sha(BASE/'KEY_ERRORS_32g.csv'),fixtures_sha256=sha(ROOT/'fixtures.npz'),no_current_service_modified=True);(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('frozen_sources','banks')}))
if __name__=='__main__':main()
