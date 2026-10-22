# SPDX-License-Identifier: Apache-2.0
import argparse,csv,hashlib,json,math,struct,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def f32(words):return (np.asarray(words,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16(values):
 u=np.asarray(values,np.float32).view(np.uint32);return ((u+32767+((u>>16)&1))>>16).astype(np.uint16)
def ahash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);args=p.parse_args();root=args.input.resolve();assert root.is_relative_to('/workspace/benchmark-results');state=json.loads((root/'result.json').read_text());assert state['complete'] and state['checkpoint_unchanged'] and sha(MODEL)==state['checkpoint_sha256_before']==state['checkpoint_sha256_after'];assert all(sha(path)==digest for path,digest in state['frozen_sources'].items())
 with MODEL.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
 pieces=[]
 for role in ('gate','up'):
  info=header[f'model.layers.0.mlp.{role}_proj.weight'];assert info['shape']==[6144,2048] and info['dtype']=='BF16';pieces.append(np.asarray(np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=(6144,2048))))
 weight_words=np.concatenate(pieces);wf=f32(weight_words);ws=np.maximum(abs(wf).max(1),np.float32(1e-8))/np.float32(127.);codes=np.clip(np.rint(wf/ws[:,None]),-127,127).astype(np.int8)
 assert all(ahash(array)==state['weight_signature'][key] for key,array in [('original_BF16_sha256',weight_words),('packed_sha256',codes),('scales_sha256',ws)])
 assert len(state['cases'])==3 and {c['M'] for c in state['cases']}=={128,512,4096};timings=[];selected_count=0;projection_count=0;phase_count=0;variants={'NN_R1','NN_R4','NT_R1','NT_R4'}
 for case in state['cases']:
  m,n,k=case['M'],case['N'],case['K'];assert n==12288 and k==2048 and case['complete'] and set(case['variants'])==variants and len(case['phases'])==3
  sample=np.random.default_rng(202610221200+m).standard_normal((m,k),dtype=np.float32);base=bf16(sample);base[0]=0
  for phase,entry in enumerate(case['phases']):
   sign=1 if phase==0 else -1;factor=np.float32(.5 if phase==2 else 1.);a=f32(base)*np.float32(1 if phase==0 else .5 if phase==1 else -.5);asc=np.asarray(np.maximum(abs(a).max(1),np.float32(1e-8)).astype(np.float64)/127.,np.float32);aq=np.clip(np.rint(np.asarray(a.astype(np.float64)/asc[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8)
   assert entry['phase']==phase and entry['weight_sign']==sign and entry['weight_scale_factor']==factor and entry['guards_inputs_weights_RNG'] and len(entry['entries'])==4 and {v['variant'] for v in entry['entries']}==variants and sha(root/entry['artifact'])==entry['sha256']
   with np.load(root/entry['artifact'],allow_pickle=False) as data:
    assert np.array_equal(data['A_words'],bf16(a)) and np.array_equal(data['Aq'],aq) and np.array_equal(data['As'].view(np.uint32),asc.view(np.uint32)) and np.array_equal(data['WS'].view(np.uint32),(ws*factor).view(np.uint32))
    r,c=data['rows'],data['columns'];assert np.array_equal(r,np.unique(np.linspace(0,m-1,8).astype(np.int64))) and np.array_equal(c,np.unique(np.linspace(0,6143,32).astype(np.int64)));both=np.concatenate([c,c+6144]);integer=data['integer'];assert integer.dtype==np.int32 and integer.shape==(m,n);reference=aq[r].astype(np.int64)@(codes[both].astype(np.int64)*sign).T;assert np.array_equal(reference,integer[np.ix_(r,both)].astype(np.int64));selected_count+=reference.size
    actual_hash=hashlib.sha256();selected_projection=np.empty((len(r),len(both)),np.uint16)
    for start in range(0,m,64):
     end=min(m,start+64);proj=bf16((integer[start:end].astype(np.float32)*asc[start:end,None])*(ws*factor)[None,:]);actual_hash.update(proj.tobytes());ids=np.flatnonzero((r>=start)&(r<end));selected_projection[ids]=proj[np.ix_(r[ids]-start,both)]
    assert actual_hash.hexdigest()==entry['current_projection_BF16_sha256'] and np.array_equal(selected_projection,data['projected_selected_words']);projection_count+=integer.size
    selected=f32(selected_projection).astype(np.float64);g,u=selected[:,:len(c)],selected[:,len(c):];expected=g/(1+np.exp(-g))*u;output=f32(data['current_output_selected_words']).astype(np.float64);assert np.isfinite(output).all() and np.all(abs(output-expected)<=.001+.02*abs(expected))
   assert all(v['full_INT32_exact'] and v['full_projected_BF16_sha256']==entry['current_projection_BF16_sha256'] and v['full_output_BF16_sha256']==entry['current_output_BF16_sha256'] for v in entry['entries']);phase_count+=1
  rounds=[r for r in state['rounds'] if r['M']==m];assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7)) and all(r['replays']==20 and set(r['order'])==variants|{'current'} and set(r['milliseconds'])==variants|{'current'} for r in rounds)
  med={v:float(np.median([r['milliseconds'][v] for r in rounds])) for v in variants|{'current'}};assert all(math.isfinite(v) and v>0 for v in med.values())
  for name in sorted(variants):timings.append(dict(M=m,variant=name,layout=case['variants'][name]['layout'],epilogue_rows=case['variants'][name]['rows'],current_ms=med['current'],candidate_ms=med[name],speedup_pct=(med['current']/med[name]-1)*100,INT32_workspace_bytes=case['native_workspace_bytes']))
 assert len(timings)==12 and phase_count==9 and all(sha(path)==digest for path,digest in state['frozen_sources'].items());assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(timings[0]),lineterminator='\n');writer.writeheader();writer.writerows(timings)
 gate=dict(complete=True,CPU_only=True,sources_unchanged=True,checkpoint_unchanged=True,phase_artifacts_verified=phase_count,candidate_phase_comparisons=36,selected_INT64_outputs_exact=selected_count,full_projection_BF16_elements_verified=projection_count,all_candidate_projection_and_output_hashes_current=True,model_executed=False,official_benchmark=False,adopted=False,raw_directory=str(root),raw_result_sha256=sha(root/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'));(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
