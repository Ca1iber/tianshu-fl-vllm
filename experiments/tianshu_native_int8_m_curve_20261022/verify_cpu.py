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
 parser=argparse.ArgumentParser();parser.add_argument('--input',required=True,type=Path);args=parser.parse_args();root=args.input.resolve();assert root.is_relative_to('/workspace/benchmark-results');state=json.loads((root/'result.json').read_text());assert state['complete'] and state['checkpoint_unchanged'] and sha(MODEL)==state['checkpoint_sha256_before']==state['checkpoint_sha256_after'];assert all(sha(p)==h for p,h in state['frozen_sources'].items())
 with MODEL.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
 pieces=[]
 for role in ('gate','up'):
  info=header[f'model.layers.0.mlp.{role}_proj.weight'];assert info['shape']==[6144,2048] and info['dtype']=='BF16';pieces.append(np.asarray(np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=(6144,2048))))
 words=np.concatenate(pieces);wf=f32(words);ws=np.maximum(abs(wf).max(1),np.float32(1e-8))/np.float32(127.);codes=np.clip(np.rint(wf/ws[:,None]),-127,127).astype(np.int8)
 assert all(ahash(array)==state['weight_signature'][key] for key,array in [('original_BF16_sha256',words),('packed_sha256',codes),('scales_sha256',ws)])
 assert len(state['cases'])==32 and [c['M'] for c in state['cases']]==list(range(128,4097,128));timings=[];phases=0;selected_count=0;routes={}
 for case in state['cases']:
  m=case['M'];assert case['complete'] and len(case['phases'])==3 and case['K']==2048 and case['N']==12288;routes[case['current_route']]=routes.get(case['current_route'],0)+1
  sample=np.random.default_rng(202610221200+m).standard_normal((m,2048),dtype=np.float32);base=bf16(sample);base[0]=0
  for phase,entry in enumerate(case['phases']):
   sign=1 if phase==0 else -1;factor=np.float32(.5 if phase==2 else 1.);a=f32(base)*np.float32(1 if phase==0 else .5 if phase==1 else -.5);asc=np.asarray(np.maximum(abs(a).max(1),np.float32(1e-8)).astype(np.float64)/127.,np.float32);aq=np.clip(np.rint(np.asarray(a.astype(np.float64)/asc[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8)
   assert entry['phase']==phase and entry['weight_sign']==sign and entry['weight_scale_factor']==factor and sha(root/entry['artifact'])==entry['sha256'];assert all(entry[name] for name in ('full_integer_exact','full_projected_BF16_exact','full_output_tolerance_passed','graph_poison_and_eager_equality','guards_inputs_weights_RNG'))
   assert ahash(bf16(a))==entry['A_words_sha256'] and ahash(aq)==entry['Aq_sha256'] and ahash(asc)==entry['As_sha256'] and ahash(ws*factor)==entry['WS_sha256']
   with np.load(root/entry['artifact'],allow_pickle=False) as data:
    r,c=data['rows'],data['columns'];assert np.array_equal(r,np.unique(np.linspace(0,m-1,8).astype(np.int64))) and np.array_equal(c,np.unique(np.linspace(0,6143,32).astype(np.int64)));both=np.concatenate([c,c+6144]);integer=aq[r].astype(np.int64)@(codes[both].astype(np.int64)*sign).T;assert np.array_equal(integer,data['integer_selected']);projected=bf16((integer.astype(np.float32)*asc[r,None])*(ws*factor)[None,both]);assert np.array_equal(projected,data['projected_selected_words']);selected_count+=integer.size
    proj=f32(projected).astype(np.float64);g,u=proj[:,:len(c)],proj[:,len(c):];expected=g/(1+np.exp(-g))*u
    for key in ('control_selected_words','candidate_selected_words'):
     out=f32(data[key]).astype(np.float64);assert np.isfinite(out).all() and np.all(abs(out-expected)<=.001+.02*abs(expected))
   if case['current_route']=='native_r1':assert entry['full_output_BF16_bit_mismatches']==0 and entry['control_output_sha256']==entry['candidate_output_sha256']
   phases+=1
  for mode in ('graph','eager'):
   rounds=[r for r in state['rounds'] if r['M']==m and r['mode']==mode];assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7)) and all(r['replays']==10 and set(r['order'])=={'control','candidate'} and set(r['milliseconds'])=={'control','candidate'} for r in rounds);med={name:float(np.median([r['milliseconds'][name] for r in rounds])) for name in ('control','candidate')};assert all(math.isfinite(v) and v>0 for v in med.values());timings.append(dict(M=m,mode=mode,current_route=case['current_route'],current_ms=med['control'],candidate_ms=med['candidate'],speedup_pct=(med['control']/med['candidate']-1)*100,INT32_workspace_bytes=case['native_workspace_bytes'],maximum_output_bit_mismatches=max(p['full_output_BF16_bit_mismatches'] for p in case['phases'])))
 assert routes=={'fused':21,'unfused':9,'native_r1':2} and phases==96 and len(timings)==64 and all(sha(p)==h for p,h in state['frozen_sources'].items());assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(timings[0]),lineterminator='\n');writer.writeheader();writer.writerows(timings)
 gate=dict(complete=True,CPU_only=True,sources_unchanged=True,checkpoint_unchanged=True,case_count=32,phase_artifacts_verified=phases,current_route_counts=routes,independent_selected_INT64_outputs_exact=selected_count,all_A8_and_scale_hashes_recomputed=True,selected_FP64_SiLU_both_arms_passed=True,full_GPU_integer_and_projection_equality=True,full_graph_poison_eager_checks=True,numerical_scope='CPU full A8/scales and selected INT64/projection/SiLU; GPU full integer/projection/output-tolerance comparisons. Not full CPU GEMM recomputation.',model_executed=False,official_benchmark=False,adopted=False,raw_directory=str(root),raw_result_sha256=sha(root/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'));(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
