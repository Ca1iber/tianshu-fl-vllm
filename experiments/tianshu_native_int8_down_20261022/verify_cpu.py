# SPDX-License-Identifier: Apache-2.0
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
 parser=argparse.ArgumentParser();parser.add_argument('--input',required=True,type=Path);args=parser.parse_args();root=args.input.resolve();assert root.is_relative_to('/workspace/benchmark-results');state=json.loads((root/'result.json').read_text());assert state['complete'] and state['checkpoint_unchanged'] and sha(MODEL)==state['checkpoint_sha256_before']==state['checkpoint_sha256_after'];assert all(sha(p)==h for p,h in state['frozen_sources'].items())
 with MODEL.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
 info=header['model.layers.0.mlp.down_proj.weight'];assert info['shape']==[2048,6144] and info['dtype']=='BF16';weight=np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=(2048,6144));wf=f32(weight);scale=np.maximum(abs(wf).max(1),np.float32(1e-8))/np.float32(127.);code=np.clip(np.rint(wf/scale[:,None]),-127,127).astype(np.int8);assert all(ahash(array)==state['Down_signature'][key] for key,array in [('original_BF16_sha256',weight),('packed_sha256',code),('scales_sha256',scale)])
 assert [c['M'] for c in state['cases']]==[64,128,512,4096];variants={f'{layout}_B{block}' for layout in ('NT','NN') for block in (512,1024)};timings=[];count=0
 for case in state['cases']:
  m=case['M'];assert case['complete'] and len(case['phases'])==3 and set(case['variants'])==variants
  for phase,entry in enumerate(case['phases']):
   sign=1 if phase==0 else -1;factor=np.float32(.5 if phase==2 else 1.);assert entry['phase']==phase and entry['weight_sign']==sign and entry['weight_scale_factor']==factor and sha(root/entry['artifact'])==entry['sha256'] and entry['guards_inputs_weights_RNG'];assert math.isfinite(entry['relative_L2_error_vs_original_BF16']) and math.isfinite(entry['maximum_absolute_error_vs_original_BF16'])
   with np.load(root/entry['artifact'],allow_pickle=False) as data:
    h=f32(data['H_words']);assert h.shape==(m,6144) and ahash(data['H_words'])==entry['H_words_sha256'] and np.isfinite(h).all() and np.count_nonzero(h[0])==0;asc=np.asarray(np.maximum(abs(h).max(1),np.float32(1e-8)).astype(np.float64)/127.,np.float32);aq=np.clip(np.rint(np.asarray(h.astype(np.float64)/asc[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8);assert ahash(aq)==entry['Aq_sha256'] and ahash(asc)==entry['As_sha256'] and ahash(scale*factor)==entry['WS_sha256'];r,c=data['rows'],data['columns'];assert np.array_equal(r,np.unique(np.linspace(0,m-1,8).astype(np.int64))) and np.array_equal(c,np.unique(np.concatenate([np.linspace(0,2047,32).astype(np.int64),np.arange(2040,2048)])));integer=aq[r].astype(np.int64)@(code[c].astype(np.int64)*sign).T;assert np.array_equal(integer,data['integer_selected']);projected=bf16((integer.astype(np.float32)*asc[r,None])*(scale*factor)[None,c]);assert np.array_equal(projected,data['candidate_selected_words']);count+=integer.size
    native_ref=h[r].astype(np.float64)@((wf[c]*sign*factor).astype(np.float64)).T;actual=f32(data['control_selected_words']).astype(np.float64);assert np.isfinite(actual).all() and np.all(abs(actual-native_ref)<=.02+.01*abs(native_ref))
   assert len(entry['entries'])==5 and {v['name'] for v in entry['entries']}==variants|{'control'}
   for record in entry['entries']:
    assert record['graph_poison_eager_equal'] and record['full_INT32_exact']==(record['name']!='control') and record['full_quantized_BF16_exact']==(record['name']!='control')
    if record['name']!='control':assert record['output_sha256']==entry['quantized_output_sha256']
  for mode in ('graph','eager'):
   rounds=[r for r in state['rounds'] if r['M']==m and r['mode']==mode];assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7)) and all(r['replays']==10 and set(r['order'])==variants|{'control'} and set(r['milliseconds'])==variants|{'control'} for r in rounds);med={name:float(np.median([r['milliseconds'][name] for r in rounds])) for name in variants|{'control'}};assert all(math.isfinite(v) and v>0 for v in med.values())
   for name in sorted(variants):timings.append(dict(M=m,mode=mode,variant=name,current_BF16_ms=med['control'],candidate_ms=med[name],speedup_pct=(med['control']/med[name]-1)*100,maximum_phase_relative_L2_error=max(p['relative_L2_error_vs_original_BF16'] for p in case['phases']),INT32_workspace_bytes=case['native_workspace_bytes']))
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(timings[0]),lineterminator='\n');writer.writeheader();writer.writerows(timings)
 assert all(sha(p)==h for p,h in state['frozen_sources'].items()) and not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules);gate=dict(complete=True,CPU_only=True,sources_unchanged=True,checkpoint_unchanged=True,cases=4,phases=12,variants=4,all_hidden_A8_and_scales_independently_recomputed=True,selected_INT64_quantized_BF16_and_original_FP64_elements=count,full_GPU_quantized_INT32_BF16_exact=True,all_graph_poison_eager_checks=True,new_precision_approximation=True,model_quality_not_tested=True,raw_directory=str(root),raw_result_sha256=sha(root/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'),model_executed=False,official_benchmark=False,adopted=False);(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
