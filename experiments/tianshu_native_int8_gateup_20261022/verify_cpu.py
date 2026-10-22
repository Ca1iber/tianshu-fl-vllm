# SPDX-License-Identifier: Apache-2.0
"""Rebuild original checkpoint quantization and validate saved native INT32/projections."""
import argparse,csv,hashlib,json,math,struct,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def f32(words):return (np.asarray(words,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16_words(values):
 u=np.asarray(values,np.float32).view(np.uint32);return ((u+32767+((u>>16)&1))>>16).astype(np.uint16)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args();root=args.input.resolve();assert root.is_relative_to('/workspace/benchmark-results')
 state=json.loads((root/'result.json').read_text());assert state['complete'] and state['checkpoint_unchanged'];assert sha(MODEL)==state['checkpoint_sha256_before']==state['checkpoint_sha256_after'];assert all(sha(path)==digest for path,digest in state['frozen_sources'].items())
 with MODEL.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
 parts=[]
 for role in ('gate','up'):
  info=header[f'model.layers.0.mlp.{role}_proj.weight'];assert info['shape']==[6144,2048] and info['dtype']=='BF16';parts.append(np.asarray(np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=(6144,2048))))
 weight_words=np.concatenate(parts);values=f32(weight_words);weight_scale=np.maximum(np.max(np.abs(values),axis=1),np.float32(1.e-8))/np.float32(127.)
 codes=np.clip(np.rint(values/weight_scale[:,None]),-127,127).astype(np.int8)
 for key,array in [('original_BF16_sha256',weight_words),('packed_sha256',codes),('scales_sha256',weight_scale)]:assert hashlib.sha256(array.tobytes()).hexdigest()==state['weight_signature'][key]
 expected_names={f'{layout}_B{block}W{warps}' for layout in ('NN','NT') for block in (256,512,1024) for warps in (4,8)}
 assert len(state['cases'])==2 and {c['strided'] for c in state['cases']}=={False,True};artifacts=0;int64_outputs=0;full_projection_elements=0;timings=[]
 for case in state['cases']:
  m,n,k=case['M'],case['N'],case['K'];assert (m,n,k)==(64,12288,2048) and case['complete'] and set(case['variants'])==expected_names and len(case['phases'])==3
  sample=np.random.default_rng(202610221100+int(case['strided'])).standard_normal((m,k),dtype=np.float32);base=bf16_words(sample);base[0]=0
  for phase,entry in enumerate(case['phases']):
   sign=1 if phase==0 else -1;factor=np.float32(.5 if phase==2 else 1.);activation=f32(base)*np.float32(1 if phase==0 else .5 if phase==1 else -.5);wanted_a=bf16_words(activation)
   asc=np.asarray(np.maximum(abs(activation).max(1),np.float32(1.e-8)).astype(np.float64)/127.,np.float32);aq=np.clip(np.rint(np.asarray(activation.astype(np.float64)/asc[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8)
   assert entry['phase']==phase and entry['weight_sign']==sign and entry['weight_scale_factor']==factor and entry['guards_inputs_weights_RNG'] and len(entry['entries'])==12 and {e['variant'] for e in entry['entries']}==expected_names
   for record in entry['entries']:
    path=root/record['artifact'];assert sha(path)==record['sha256'] and record['full_INT32_BF16_projection_output_exact']
    with np.load(path,allow_pickle=False) as data:
     assert np.array_equal(data['A_words'],wanted_a) and np.array_equal(data['Aq'],aq) and np.array_equal(data['As'].view(np.uint32),asc.view(np.uint32)) and np.array_equal(data['WS'].view(np.uint32),(weight_scale*factor).view(np.uint32))
     rows,columns=data['rows'],data['columns'];assert np.array_equal(rows,np.unique(np.linspace(0,63,8).astype(np.int64))) and np.array_equal(columns,np.unique(np.linspace(0,6143,32).astype(np.int64)));both=np.concatenate([columns,columns+6144]);reference=aq[rows].astype(np.int64)@(codes[both].astype(np.int64)*sign).T;assert np.array_equal(reference,data['integer'][np.ix_(rows,both)].astype(np.int64));int64_outputs+=reference.size
     integer=data['integer'];assert integer.dtype==np.int32 and integer.shape==(m,n);projected=(integer.astype(np.float32)*asc[:,None])*(weight_scale*factor)[None,:];assert np.array_equal(bf16_words(projected),data['projected_words']);full_projection_elements+=integer.size
     output=data['output_words'];assert output.dtype==np.uint16 and output.shape==(m,n//2) and np.array_equal(output,data['current_output_words']) and np.isfinite(f32(output)).all()
     chosen=f32(data['projected_words'][np.ix_(rows,both)]).astype(np.float64);gate,up=chosen[:,:len(columns)],chosen[:,len(columns):];expected=gate/(1+np.exp(-gate))*up;actual=f32(output[np.ix_(rows,columns)]).astype(np.float64);assert np.all(abs(actual-expected)<=.001+.02*abs(expected))
    artifacts+=1
  rounds=[r for r in state['rounds'] if r['strided']==case['strided']];assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7)) and all(r['replays']==20 and set(r['order'])==expected_names|{'current'} and set(r['milliseconds'])==expected_names|{'current'} for r in rounds)
  medians={name:float(np.median([r['milliseconds'][name] for r in rounds])) for name in expected_names|{'current'}};assert all(math.isfinite(value) and value>0 for value in medians.values())
  for name in sorted(expected_names):
   variant=case['variants'][name];timings.append(dict(M=m,strided=case['strided'],variant=name,layout=variant['layout'],block=variant['block'],warps=variant['warps'],current_ms=medians['current'],candidate_ms=medians[name],speedup_pct=(medians['current']/medians[name]-1)*100,n_regs=variant['compiled']['n_regs'],n_spills=variant['compiled']['n_spills']))
 assert artifacts==72 and len(timings)==24 and all(sha(path)==digest for path,digest in state['frozen_sources'].items())
 assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(timings[0]),lineterminator='\n');writer.writeheader();writer.writerows(timings)
 gate=dict(complete=True,CPU_only=True,checkpoint_unchanged=True,sources_unchanged=True,full_original_weight_format_signature=True,phase_artifacts_verified=artifacts,selected_INT64_outputs_exact=int64_outputs,full_projected_BF16_elements_exact=full_projection_elements,full_output_bitwise_current=True,model_executed=False,official_benchmark=False,adopted=False,raw_directory=str(root),raw_result_sha256=sha(root/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'))
 (BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
