# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');HOLDOUT=EXP/'tianshu_native_int8_down42_20261022'
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def ahash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def f32(words):return (np.asarray(words,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16(values):
 u=np.asarray(values,np.float32).view(np.uint32);return ((u+32767+((u>>16)&1))>>16).astype(np.uint16)
def quantize(values):
 scale=np.asarray(np.maximum(abs(values).max(1),np.float32(1e-8)).astype(np.float64)/127.,np.float32);codes=np.clip(np.rint(np.asarray(values.astype(np.float64)/scale[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8);return codes,scale

def main():
 root=Path('/workspace/benchmark-results/20261022_down_outlier_partition_CPU_tianshu');assert not root.exists();root.mkdir();gate=read(HOLDOUT/'INDEPENDENT_CPU_GATE_32g.json');raw=Path(gate['raw_directory']);state=read(raw/'result.json');assert gate['complete'] and state['complete'] and gate['raw_result_sha256']==sha(raw/'result.json');model_hash=sha(MODEL);assert model_hash==state['checkpoint_sha256_before']==state['checkpoint_sha256_after'];sources={str(p):sha(p) for p in (BASE/'analyze_cpu.py',BASE/'PLAN_32g.md',HOLDOUT/'INDEPENDENT_CPU_GATE_32g.json')};(root/'CPU_FREEZE.json').write_text(json.dumps(dict(sources=sources,checkpoint_sha256=model_hash,holdout_result_sha256=sha(raw/'result.json')),indent=2)+'\n')
 with MODEL.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
 rows=[];ranks=[];groups={k:dict(calibration_error=0.,calibration_ref=0.,validation_error=0.,validation_ref=0.) for k in (0,32,64,128)};artifact_count=0
 for layer,bank in enumerate(state['banks']):
  info=header[f'model.layers.{layer}.mlp.down_proj.weight'];assert info['shape']==[2048,6144] and info['dtype']=='BF16';ww=np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=(2048,6144));wf=f32(ww);assert ahash(ww)==bank['Down_original_BF16_sha256'];fixtures=[];calibration=[]
  for case in state['cases']:
   for phase in case['phases']:
    record=phase['banks'][layer];path=raw/record['artifact'];assert sha(path)==record['sha256']
    with np.load(path,allow_pickle=False) as data:fixture=dict(M=case['M'],phase=phase['phase'],H=f32(data['H_selected_rows_words']),columns=data['columns'].copy(),reference=f32(data['control_selected_words']).astype(np.float64))
    fixtures.append(fixture);artifact_count+=1
    if phase['phase']==0:calibration.append(fixture['H'])
  channel_max=abs(np.concatenate(calibration)).max(0);rank=np.argsort(-channel_max,kind='stable').astype(np.int32);ranks.append(rank[:128])
  for side in (0,32,64,128):
   selected=np.sort(rank[:side]);bulk_weight=wf.copy();bulk_weight[:,selected]=0.;packed,ws=quantize(bulk_weight);acc=dict(calibration_error=0.,calibration_ref=0.,validation_error=0.,validation_ref=0.)
   if side==0:assert ahash(packed)==bank['Down_packed_sha256'] and ahash(ws)==bank['Down_scales_sha256']
   for fixture in fixtures:
    h=fixture['H'];columns=fixture['columns'];bulk=h.copy();bulk[:,selected]=0.;aq,asc=quantize(bulk);integer=aq.astype(np.float64)@packed[columns].astype(np.float64).T;assert np.max(abs(integer))<2**31 and np.array_equal(integer,np.rint(integer));bulk_out=(integer.astype(np.float32)*asc[:,None])*ws[None,columns]
    correction=np.asarray(h[:,selected].astype(np.float64)@wf[columns][:,selected].astype(np.float64).T,np.float32);projected=f32(bf16(np.asarray(bulk_out+correction,np.float32))).astype(np.float64);error=projected-fixture['reference'];label='calibration' if fixture['phase']==0 else 'validation';acc[label+'_error']+=float(np.square(error).sum());acc[label+'_ref']+=float(np.square(fixture['reference']).sum())
   for key,value in acc.items():groups[side][key]+=value
   rows.append(dict(layer=layer,BF16_channels=side,calibration_relative_L2=(acc['calibration_error']/max(acc['calibration_ref'],1e-30))**.5,related_validation_relative_L2=(acc['validation_error']/max(acc['validation_ref'],1e-30))**.5,extra_all42_side_weight_bytes=42*side*2048*2))
  print(json.dumps(dict(layer=layer,complete=True)),flush=True)
 assert artifact_count==504 and len(rows)==168 and all(sha(p)==h for p,h in sources.items()) and sha(MODEL)==model_hash
 summary={str(k):dict(calibration_relative_L2=(g['calibration_error']/g['calibration_ref'])**.5,related_validation_relative_L2=(g['validation_error']/g['validation_ref'])**.5) for k,g in groups.items()};baseline=summary['0']['related_validation_relative_L2'];eligible=[k for k in (32,64,128) if summary[str(k)]['related_validation_relative_L2']/baseline<=.75];selected=min(eligible) if eligible else None;np.savez(root/'channel_ranks.npz',top128_indices=np.stack(ranks))
 with (BASE/'KEY_ERRORS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 result=dict(complete=True,CPU_only=True,no_GPU_imports=True,checkpoint_unchanged=True,artifacts_checked=504,real_weight_banks=42,aggregate_errors=summary,selected_global_BF16_channels=selected,independent_random_GPU_holdout_required=True,related_validation_not_independent=True,model_executed=False,model_quality_not_tested=True,official_benchmark=False,adopted=False,sources=sources,raw_directory=str(root),channel_ranks_sha256=sha(root/'channel_ranks.npz'),key_errors_sha256=sha(BASE/'KEY_ERRORS_32g.csv'));(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
if __name__=='__main__':main()
