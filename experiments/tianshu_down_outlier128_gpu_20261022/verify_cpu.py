# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,math,statistics,struct,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');RAW=Path('/workspace/benchmark-results/20261022_down_outlier128_GPU_tianshu')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def f32(a):return (np.asarray(a,np.uint16).astype(np.uint32)<<16).view(np.float32)
def quantize(a):
 scale=np.asarray(np.maximum(abs(a).max(1),np.float32(1.e-8)).astype(np.float64)/127.,np.float32);ratio=np.asarray(a.astype(np.float64)/scale[:,None].astype(np.float64),np.float32);return np.clip(np.rint(ratio),-127,127).astype(np.int8),scale

def main():
 assert not (BASE/'INDEPENDENT_CPU_GATE_32g.json').exists();state=read(RAW/'result.json');assert state['complete'] and state['phase']=='complete' and state['checkpoint_unchanged'] and state['all42_bank_hashes_before_after'] and len(state['banks'])==42 and [c['M'] for c in state['cases']]==[64,512,4096]
 for p,h in state['frozen_sources'].items():assert sha(p)==h and sha(RAW/'source_snapshot'/Path(p).relative_to('/'))==h
 assert sha(MODEL)==state['checkpoint_sha256_before']==state['checkpoint_sha256_after']=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
 with MODEL.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
 aggregate={name:dict(error=0.,reference=0.) for name in ('ordinary_INT8','side128')};artifacts=0;cells=0;gamma128=128*np.finfo(np.float32).eps/(1-128*np.finfo(np.float32).eps);gamma6144=6144*np.finfo(np.float32).eps/(1-6144*np.finfo(np.float32).eps)
 for layer,bank in enumerate(state['banks']):
  info=header[f'model.layers.{layer}.mlp.down_proj.weight'];assert info['shape']==[2048,6144] and info['dtype']=='BF16';weight_words=np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=(2048,6144));weight=f32(weight_words);assert ahash(weight_words)==bank['Down_original_BF16_sha256'];selected=np.asarray(bank['sorted_indices'],np.int32);assert selected.shape==(128,) and np.array_equal(selected,np.unique(selected));mask=np.zeros(6144,np.uint8);mask[selected]=1;bulk=weight.copy();bulk[:,selected]=0.;packed,scale=quantize(bulk);side_weight=weight[:,selected].T.copy()
  for array,key in ((packed,'side_bulk_weight_sha256'),(scale,'side_bulk_scales_sha256'),(side_weight.view(np.uint32).astype(np.uint32)>>16,'side_weight_sha256'),(selected,'side_indices_sha256'),(mask,'side_mask_sha256')):
   if key=='side_weight_sha256':array=array.astype(np.uint16)
   assert ahash(array)==bank[key]
  for case in state['cases']:
   assert case['complete'] and len(case['phases'])==2
   for phase in case['phases']:
    assert phase['seed']==202610224700+case['M']*100+phase['phase'] and phase['whole42_graph_poison_eager_verified'] and phase['guards_inputs_RNG'];record=phase['banks'][layer];assert record['layer']==layer and record['full_native_INT32_equals_independent_NT'] and record['full_corrected_output_inside_predeclared_error_bound'];path=RAW/record['artifact'];assert sha(path)==record['sha256']
    with np.load(path,allow_pickle=False) as data:
     h=f32(data['H_selected_rows_words']);cols=data['columns'];assert h.shape==(8,6144) and cols.shape==(32,) and not h[0].any();masked=h.copy();masked[:,selected]=0.;aq,asc=quantize(masked);assert np.array_equal(aq,data['bulk_Aq_selected_rows']) and np.array_equal(asc,data['bulk_As_selected_rows'])
     integer=aq.astype(np.int64)@packed[cols].astype(np.int64).T;assert np.array_equal(integer,data['bulk_integer_selected']);bulk_out=(integer.astype(np.float32)*asc[:,None])*scale[None,cols]
     side=h[:,selected].astype(np.float64)@weight[cols][:,selected].astype(np.float64).T;absolute=abs(h[:,selected].astype(np.float64))@abs(weight[cols][:,selected].astype(np.float64)).T;ideal=bulk_out.astype(np.float64)+side;limit=.00390625*abs(ideal)+2*gamma128*absolute+5.e-7*(abs(bulk_out.astype(np.float64))+abs(side))+1.e-7
     observed=f32(data['candidate_selected_words']).astype(np.float64);assert np.all(abs(observed-ideal)<=limit)
     assert np.all(abs(data['reference_selected'].astype(np.float64)-ideal)<=gamma128*absolute+5.e-7*(abs(bulk_out.astype(np.float64))+abs(side))+1.e-7)
     original=h.astype(np.float64)@weight[cols].astype(np.float64).T;original_abs=abs(h.astype(np.float64))@abs(weight[cols].astype(np.float64)).T;assert np.all(abs(f32(data['control_selected_words']).astype(np.float64)-original)<=.00390625*abs(original)+gamma6144*original_abs+1.e-7)
     cells+=integer.size
    for name in aggregate:
     errors=record['errors_vs_original_BF16'][name];aggregate[name]['error']+=errors['squared_error'];aggregate[name]['reference']+=errors['squared_reference']
    artifacts+=1
  print(json.dumps(dict(layer=layer,independent_CPU=True)),flush=True)
 assert artifacts==252 and cells==64512;timings=[]
 for m in (64,512,4096):
  for scope in ('whole_MLP','Down_only_shared_last_H'):
   rows=[r for r in state['rounds'] if r['M']==m and r['scope']==scope];assert len(rows)==7 and sorted(r['round'] for r in rows)==list(range(7))
   medians={name:statistics.median(r['milliseconds_per_bank'][name] for r in rows) for name in aggregate|{'control':None}}
   assert all(math.isfinite(v) and v>0 for v in medians.values());timings.append(dict(M=m,scope=scope,control_ms=medians['control'],ordinary_INT8_ms=medians['ordinary_INT8'],side128_ms=medians['side128'],side128_speedup_vs_BF16_pct=100*(medians['control']/medians['side128']-1),side128_speedup_vs_ordinary_INT8_pct=100*(medians['ordinary_INT8']/medians['side128']-1)))
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(timings[0]),lineterminator='\n');writer.writeheader();writer.writerows(timings)
 errors={name:(v['error']/v['reference'])**.5 for name,v in aggregate.items()};result=dict(complete=True,CPU_only=True,no_GPU_imports=True,independent_of_calibration_seed=True,all42_weight_banks=True,artifacts_checked=artifacts,independent_INT64_full_K_and_FP64_selected_cells=cells,full_GPU_INT32_equals_independent_NT=True,full_GPU_side_reference_bound_passed=True,bit_exact_against_original_BF16_not_claimed=True,whole42_poison_graph_eager_verified=True,all_source_snapshots_and_checkpoint_verified=True,aggregate_relative_L2_vs_original_BF16=errors,relative_error_reduction_pct=100*(1-errors['side128']/errors['ordinary_INT8']),timings=timings,model_executed=False,new_full_C8_C64_Math_required=True,official_benchmark=False,adopted=False,raw_directory=str(RAW),raw_result_sha256=sha(RAW/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'))
 (BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps(result))
if __name__=='__main__':main()
