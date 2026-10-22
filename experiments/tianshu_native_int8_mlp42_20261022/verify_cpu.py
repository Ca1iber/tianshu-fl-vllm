# SPDX-License-Identifier: Apache-2.0
"""Independent all-bank signatures, full A8, selected INT64/activation/Down checks."""
import argparse,csv,hashlib,json,math,struct,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def f32(words):return (np.asarray(words,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16(values):
 u=np.asarray(values,np.float32).view(np.uint32);return ((u+32767+((u>>16)&1))>>16).astype(np.uint16)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);args=p.parse_args();root=args.input.resolve();assert root.is_relative_to('/workspace/benchmark-results');state=json.loads((root/'result.json').read_text());assert state['complete'] and state['all42_bank_hashes_before_after'] and state['checkpoint_unchanged'] and sha(MODEL)==state['checkpoint_sha256_before']==state['checkpoint_sha256_after'];assert all(sha(path)==digest for path,digest in state['frozen_sources'].items())
 with MODEL.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
 def weight(name,shape):
  info=header[name];assert info['shape']==list(shape) and info['dtype']=='BF16';return np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=shape)
 assert len(state['banks'])==42 and [r['layer'] for r in state['banks']]==list(range(42));cols=np.unique(np.linspace(0,6143,32).astype(np.int64));both=np.concatenate([cols,cols+6144]);dcols=np.unique(np.linspace(0,2047,16).astype(np.int64));selected={}
 for index,proof in enumerate(state['banks']):
  prefix=f'model.layers.{index}.mlp.';w=np.concatenate([np.asarray(weight(prefix+role+'_proj.weight',(6144,2048))) for role in ('gate','up')]);v=f32(w);scale=np.maximum(abs(v).max(1),np.float32(1e-8))/np.float32(127.);codes=np.clip(np.rint(v/scale[:,None]),-127,127).astype(np.int8);down=weight(prefix+'down_proj.weight',(2048,6144))
  assert ahash(w)==proof['original_BF16_sha256'] and ahash(codes)==proof['packed_sha256'] and ahash(scale)==proof['scales_sha256'] and ahash(down)==proof['Down_BF16_sha256']
  if index in (0,7,20,41):selected[index]=dict(code=codes[both].copy(),scale=scale[both].copy(),down=f32(np.asarray(down[dcols])).copy())
 assert len(state['cases'])==2 and {c['M'] for c in state['cases']}=={512,4096};timings=[];quant_elements=0;integer_outputs=0;down_outputs=0;artifact_count=0
 for case in state['cases']:
  m=case['M'];assert case['complete'] and case['weight_banks_per_graph']==42 and len(case['phases'])==3;sample=np.random.default_rng(202610221300+m).standard_normal((m,2048),dtype=np.float32);base=bf16(sample);base[0]=0;rows=np.unique(np.linspace(0,m-1,4).astype(np.int64))
  for phase,entry in enumerate(case['phases']):
   assert entry['phase']==phase and entry['all42_full_GateUp_INT32_projection_Down_exact'] and entry['guards_input_RNG'] and len(entry['all42_whole_graph_poison_eager_hashes'])==42 and all(len(h)==64 for h in entry['all42_whole_graph_poison_eager_hashes'])
   fixture=root/entry['fixture'];assert sha(fixture)==entry['fixture_sha256'];a=f32(base)*np.float32(1 if phase==0 else .5 if phase==1 else -.5);asc=np.asarray(np.maximum(abs(a).max(1),np.float32(1e-8)).astype(np.float64)/127.,np.float32);aq=np.clip(np.rint(np.asarray(a.astype(np.float64)/asc[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8)
   with np.load(fixture,allow_pickle=False) as data:assert np.array_equal(data['A_words'],bf16(a)) and np.array_equal(data['Aq'],aq) and np.array_equal(data['As'].view(np.uint32),asc.view(np.uint32))
   quant_elements+=aq.size;assert len(entry['selected_bank_artifacts'])==4 and {r['layer'] for r in entry['selected_bank_artifacts']}=={0,7,20,41}
   for record in entry['selected_bank_artifacts']:
    path=root/record['artifact'];assert sha(path)==record['sha256'];bank=selected[record['layer']]
    with np.load(path,allow_pickle=False) as data:
     assert np.array_equal(data['rows'],rows) and np.array_equal(data['columns'],cols) and np.array_equal(data['down_columns'],dcols);dot=aq[rows].astype(np.int64)@bank['code'].astype(np.int64).T;assert np.array_equal(dot,data['integer'].astype(np.int64));integer_outputs+=dot.size;projected=bf16((dot.astype(np.float32)*asc[rows,None])*bank['scale'][None,:]);assert np.array_equal(projected,data['projected_words']);g,u=np.split(f32(projected).astype(np.float64),2,axis=1);expected=g/(1+np.exp(-g))*u;activation=f32(data['activation_selected_words']).astype(np.float64);assert np.isfinite(activation).all() and np.all(abs(activation-expected)<=.001+.02*abs(expected));hidden=f32(data['Down_input_rows_words']).astype(np.float64);assert hidden.shape==(len(rows),6144) and np.array_equal(data['Down_input_rows_words'][:,cols],data['activation_selected_words']);down_ref=hidden@bank['down'].astype(np.float64).T;down_actual=f32(data['Down_output_selected_words']).astype(np.float64);assert np.isfinite(down_actual).all() and np.all(abs(down_actual-down_ref)<=.02+.01*abs(down_ref));down_outputs+=down_ref.size
    artifact_count+=1
  rounds=[r for r in state['rounds'] if r['M']==m];assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7)) and all(r['replays']==3 and r['banks_per_graph']==42 and set(r['order'])=={'current','native'} for r in rounds);med={name:float(np.median([r['milliseconds_per_MLP_layer'][name] for r in rounds])) for name in ('current','native')};assert all(math.isfinite(v) and v>0 for v in med.values());timings.append(dict(M=m,current_MLP_ms=med['current'],native_MLP_ms=med['native'],speedup_pct=(med['current']/med['native']-1)*100,absolute_ms_saved=med['current']-med['native'],weight_banks=42,layout='NN',epilogue_rows=1))
 assert quant_elements==28311552 and artifact_count==24 and all(sha(path)==digest for path,digest in state['frozen_sources'].items());assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(timings[0]),lineterminator='\n');writer.writeheader();writer.writerows(timings)
 gate=dict(complete=True,CPU_only=True,sources_unchanged=True,checkpoint_unchanged=True,all42_weight_signatures_independently_recomputed=True,all42_GPU_dynamic_phase_checks=252,whole_graph_poison_eager_all42_verified=True,full_A8_elements_exact=quant_elements,selected_INT64_outputs_exact=integer_outputs,selected_FP64_Down_outputs=down_outputs,selected_bank_artifacts_verified=artifact_count,model_executed=False,official_benchmark=False,adopted=False,raw_directory=str(root),raw_result_sha256=sha(root/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'));(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
