# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');HOLDOUT=EXP/'tianshu_native_int8_down42_20261022'
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def f32(a):return (np.asarray(a,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16(a):
 u=np.asarray(a,np.float32).view(np.uint32);return ((u+32767+((u>>16)&1))>>16).astype(np.uint16)
def quantize(a):
 s=np.asarray(np.maximum(abs(a).max(1),np.float32(1.e-8)).astype(np.float64)/127.,np.float32);return np.clip(np.rint(np.asarray(a.astype(np.float64)/s[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8),s

def main():
 root=Path('/workspace/benchmark-results/20261022_down_group_activation_CPU_tianshu');assert not root.exists();root.mkdir();gate=read(HOLDOUT/'INDEPENDENT_CPU_GATE_32g.json');raw=Path(gate['raw_directory']);state=read(raw/'result.json');assert gate['complete'] and state['complete'] and sha(raw/'result.json')==gate['raw_result_sha256'];checkpoint_hash=sha(MODEL);assert checkpoint_hash==state['checkpoint_sha256_before']==state['checkpoint_sha256_after'];sources={str(p):sha(p) for p in (BASE/'PLAN_32g.md',BASE/'analyze_cpu.py',HOLDOUT/'INDEPENDENT_CPU_GATE_32g.json')};(root/'CPU_FREEZE.json').write_text(json.dumps(dict(sources=sources,checkpoint_sha256=checkpoint_hash,holdout_result_sha256=sha(raw/'result.json')),indent=2)+'\n')
 with MODEL.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
 groups={k:dict(calibration_error=0.,calibration_ref=0.,validation_error=0.,validation_ref=0.) for k in (128,512,1024,6144)};rows=[];artifacts=0
 for layer,bank in enumerate(state['banks']):
  info=header[f'model.layers.{layer}.mlp.down_proj.weight'];assert info['shape']==[2048,6144] and info['dtype']=='BF16';ww=np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=(2048,6144));weight=f32(ww);assert ahash(ww)==bank['Down_original_BF16_sha256'];packed,ws=quantize(weight);assert ahash(packed)==bank['Down_packed_sha256'] and ahash(ws)==bank['Down_scales_sha256'];local={k:dict(calibration_error=0.,calibration_ref=0.,validation_error=0.,validation_ref=0.) for k in groups}
  for case in state['cases']:
   for phase in case['phases']:
    record=phase['banks'][layer];path=raw/record['artifact'];assert sha(path)==record['sha256']
    with np.load(path,allow_pickle=False) as data:
     h=f32(data['H_selected_rows_words']);cols=data['columns'];reference=f32(data['control_selected_words']).astype(np.float64)
     for k in groups:
      acc=np.zeros((8,32),np.float32)
      for start in range(0,6144,k):
       aq,asc=quantize(h[:,start:start+k]);integer=aq.astype(np.float64)@packed[cols,start:start+k].astype(np.float64).T;assert np.max(abs(integer))<2**31 and np.array_equal(integer,np.rint(integer));acc=np.asarray(acc+integer.astype(np.float32)*asc[:,None],np.float32)
       if k==6144:assert np.array_equal(aq,data['Aq_selected_rows']) and np.array_equal(asc,data['As_selected_rows']) and np.array_equal(integer,data['integer_selected'])
      result=bf16(np.asarray(acc*ws[None,cols],np.float32))
      if k==6144:assert np.array_equal(result,data['candidate_selected_words'])
      error=f32(result).astype(np.float64)-reference;label='calibration' if phase['phase']==0 else 'validation';local[k][label+'_error']+=float(np.square(error).sum());local[k][label+'_ref']+=float(np.square(reference).sum())
    artifacts+=1
  for k,values in local.items():
   for key,value in values.items():groups[k][key]+=value
   rows.append(dict(layer=layer,activation_K_group=k,groups_per_row=6144//k,calibration_relative_L2=(values['calibration_error']/values['calibration_ref'])**.5,related_validation_relative_L2=(values['validation_error']/values['validation_ref'])**.5))
  print(json.dumps(dict(layer=layer,complete=True)),flush=True)
 assert artifacts==504 and sha(MODEL)==checkpoint_hash and all(sha(p)==h for p,h in sources.items());summary={str(k):dict(calibration_relative_L2=(v['calibration_error']/v['calibration_ref'])**.5,related_validation_relative_L2=(v['validation_error']/v['validation_ref'])**.5) for k,v in groups.items()}
 with (BASE/'KEY_ERRORS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 result=dict(complete=True,CPU_only=True,no_GPU_imports=True,checkpoint_unchanged=True,all42_weight_banks=True,artifacts_checked=504,baseline_codes_scales_INT32_BF16_exact=True,aggregate_errors=summary,related_validation_not_independent=True,no_calibration_or_evaluation_prompts_used=True,GPU_implementation_not_yet_written=True,monolithic_native_int_mm_cannot_apply_K_group_scales_directly=True,model_quality_not_tested=True,official_benchmark=False,adopted=False,sources=sources,raw_directory=str(root),key_errors_sha256=sha(BASE/'KEY_ERRORS_32g.csv'));(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
if __name__=='__main__':main()
