# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
HOLDOUT=EXP/'tianshu_native_int8_down42_20261022'
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def f32(words):return (np.asarray(words,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16(values):
 u=np.asarray(values,np.float32).view(np.uint32);return ((u+32767+((u>>16)&1))>>16).astype(np.uint16)
def ahash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def quantize(values):
 scale=np.asarray(np.maximum(abs(values).max(1),np.float32(1e-8)).astype(np.float64)/127.,np.float32);codes=np.clip(np.rint(np.asarray(values.astype(np.float64)/scale[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8);return codes,scale

def main():
 root=Path('/workspace/benchmark-results/20261022_down_channel_balance_CPU_tianshu');assert not root.exists();root.mkdir();gate=read(HOLDOUT/'INDEPENDENT_CPU_GATE_32g.json');rawroot=Path(gate['raw_directory']);state=read(rawroot/'result.json');assert gate['complete'] and state['complete'] and gate['raw_result_sha256']==sha(rawroot/'result.json');model_hash=sha(MODEL);assert model_hash==state['checkpoint_sha256_before']==state['checkpoint_sha256_after'];sources={str(p):sha(p) for p in (BASE/'analyze_cpu.py',BASE/'PLAN_32g.md',HOLDOUT/'INDEPENDENT_CPU_GATE_32g.json')};(root/'CPU_FREEZE.json').write_text(json.dumps(dict(sources=sources,checkpoint_sha256=model_hash,holdout_result_sha256=sha(rawroot/'result.json')),indent=2)+'\n')
 with MODEL.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
 rows=[];scale_vectors=[];groups={name:{'calibration_error':0.,'calibration_ref':0.,'validation_error':0.,'validation_ref':0.} for name in ('baseline','alpha0','alpha0.25','alpha0.5','alpha0.75','alpha1')};artifact_count=0
 for layer,bank in enumerate(state['banks']):
  info=header[f'model.layers.{layer}.mlp.down_proj.weight'];assert info['shape']==[2048,6144] and info['dtype']=='BF16';ww=np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=(2048,6144));wf=f32(ww);assert ahash(ww)==bank['Down_original_BF16_sha256'];fixtures=[];calibration=[]
  for case in state['cases']:
   for phase in case['phases']:
    record=phase['banks'][layer];path=rawroot/record['artifact'];assert sha(path)==record['sha256']
    with np.load(path,allow_pickle=False) as data:fixture=dict(M=case['M'],phase=phase['phase'],H=f32(data['H_selected_rows_words']),columns=data['columns'].copy(),reference=f32(data['control_selected_words']).astype(np.float64))
    fixtures.append(fixture);artifact_count+=1
    if phase['phase']==0:calibration.append(fixture['H'])
  acmax=np.maximum(abs(np.concatenate(calibration)).max(0),np.float32(1e-8));wcmax=np.maximum(abs(wf).max(0),np.float32(1e-8));configs=[('baseline',None)]+[(f'alpha{alpha:g}',alpha) for alpha in (0.,.25,.5,.75,1.)];layer_vectors=[]
  for name,alpha in configs:
   if alpha is None:s=np.ones(6144,np.float32)
   else:log_s=alpha*np.log(acmax.astype(np.float64))-(1-alpha)*np.log(wcmax.astype(np.float64));log_s-=log_s.mean();s=np.clip(np.exp(log_s),.25,4).astype(np.float32)
   inverse=np.asarray(1./s.astype(np.float64),np.float32);packed,ws=quantize(np.asarray(wf*s[None,:],np.float32));layer_vectors.append(s);acc={'calibration_error':0.,'calibration_ref':0.,'validation_error':0.,'validation_ref':0.}
   if name=='baseline':assert ahash(packed)==bank['Down_packed_sha256'] and ahash(ws)==bank['Down_scales_sha256']
   for fixture in fixtures:
    activation=np.asarray(fixture['H']*inverse[None,:],np.float32);aq,asc=quantize(activation);integer=aq.astype(np.float64)@packed[fixture['columns']].astype(np.float64).T;assert np.max(abs(integer))<2**31 and np.array_equal(integer,np.rint(integer));projected=f32(bf16((integer.astype(np.float32)*asc[:,None])*ws[None,fixture['columns']])).astype(np.float64);error=projected-fixture['reference'];label='calibration' if fixture['phase']==0 else 'validation';acc[label+'_error']+=float(np.square(error).sum());acc[label+'_ref']+=float(np.square(fixture['reference']).sum())
   for key,value in acc.items():groups[name][key]+=value
   rows.append(dict(layer=layer,variant=name,alpha=alpha,calibration_relative_L2=(acc['calibration_error']/max(acc['calibration_ref'],1e-30))**.5,related_validation_relative_L2=(acc['validation_error']/max(acc['validation_ref'],1e-30))**.5,minimum_S=float(s.min()),maximum_S=float(s.max())))
  scale_vectors.append(np.stack(layer_vectors));print(json.dumps(dict(layer=layer,complete=True)),flush=True)
 assert artifact_count==504 and len(rows)==252 and all(sha(p)==h for p,h in sources.items()) and sha(MODEL)==model_hash
 summaries={name:dict(calibration_relative_L2=(g['calibration_error']/g['calibration_ref'])**.5,related_validation_relative_L2=(g['validation_error']/g['validation_ref'])**.5) for name,g in groups.items()};best=min((name for name in summaries if name!='baseline'),key=lambda name:summaries[name]['related_validation_relative_L2']);ratio=summaries[best]['related_validation_relative_L2']/summaries['baseline']['related_validation_relative_L2'];np.savez(root/'channel_scales.npz',S=np.stack(scale_vectors),variants=np.array(list(groups)))
 with (BASE/'KEY_ERRORS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 result=dict(complete=True,CPU_only=True,no_GPU_imports=True,checkpoint_unchanged=True,artifacts_checked=artifact_count,real_weight_banks=42,alpha_variants=5,baseline_separately_checked=True,aggregate_errors=summaries,best_global_alpha_variant=best,relative_validation_error_ratio_vs_baseline=ratio,independent_random_holdout_not_yet_run=True,related_validation_not_independent=True,model_executed=False,model_quality_not_tested=True,official_benchmark=False,adopted=False,sources=sources,raw_directory=str(root),scale_vectors_sha256=sha(root/'channel_scales.npz'),key_errors_sha256=sha(BASE/'KEY_ERRORS_32g.csv'));(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
if __name__=='__main__':main()
