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
 info=header['lm_head.weight'];assert info['shape']==[130560,2048] and info['dtype']=='BF16';words=np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=(130560,2048));signature=state['weight_signature'];assert ahash(words)==signature['original_BF16_sha256'];scales=np.empty(130560,np.float32);codes=np.empty((130560,2048),np.int8)
 for start in range(0,130560,8192):
  end=min(130560,start+8192);wf=f32(words[start:end]);scales[start:end]=np.maximum(abs(wf).max(1),np.float32(1e-8))/np.float32(127.);codes[start:end]=np.clip(np.rint(wf/scales[start:end,None]),-127,127).astype(np.int8)
 assert ahash(scales)==signature['scales_sha256'] and ahash(codes)==signature['packed_sha256'];assert (state['M'],state['N'],state['K'])==(64,130560,2048) and len(state['phases'])==3 and len(state['variants'])==8
 sample=np.random.default_rng(202610221410).standard_normal((64,2048),dtype=np.float32);base=bf16(sample);base[0]=0;count=0
 expected_variants={f'B{b}_W{w}' for b in (256,512,1024,2048) for w in (4,8)};assert set(state['variants'])==expected_variants
 for phase,entry in enumerate(state['phases']):
  sign=1 if phase==0 else -1;factor=np.float32(.5 if phase==2 else 1.);a=f32(base)*np.float32(1 if phase==0 else .5 if phase==1 else -.5);asc=np.asarray(np.maximum(abs(a).max(1),np.float32(1e-8)).astype(np.float64)/127.,np.float32);aq=np.clip(np.rint(np.asarray(a.astype(np.float64)/asc[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8)
  assert entry['phase']==phase and entry['weight_sign']==sign and entry['weight_scale_factor']==factor and entry['guards_inputs_weights_RNG'] and sha(root/entry['artifact'])==entry['sha256'];assert ahash(bf16(a))==entry['A_words_sha256'] and ahash(aq)==entry['Aq_sha256'] and ahash(asc)==entry['As_sha256'] and ahash(scales*factor)==entry['WS_sha256'];assert len(entry['entries'])==9 and {v['name'] for v in entry['entries']}==expected_variants|{'control'}
  for row in entry['entries']:assert row['full_BF16_output_exact'] and row['graph_poison_matches_eager'] and row['output_sha256']==entry['output_sha256'] and row['full_INT32_exact']==(row['name']!='control')
  with np.load(root/entry['artifact'],allow_pickle=False) as data:
   r,c=data['rows'],data['columns'];assert np.array_equal(r,np.unique(np.linspace(0,63,8).astype(np.int64))) and np.array_equal(c,np.unique(np.concatenate([np.linspace(0,130559,64).astype(np.int64),np.arange(130552,130560),np.array([0,1,1023,1024,65535,65536])]).astype(np.int64)));integer=aq[r].astype(np.int64)@(codes[c].astype(np.int64)*sign).T;assert np.array_equal(integer,data['integer_selected']);projected=bf16((integer.astype(np.float32)*asc[r,None])*(scales*factor)[None,c]);assert np.array_equal(projected,data['output_selected_words']);count+=integer.size
 rows=[]
 for mode in ('graph','eager'):
  rounds=[r for r in state['rounds'] if r['mode']==mode];assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7)) and all(r['replays']==20 and set(r['milliseconds'])==expected_variants|{'control'} and set(r['order'])==expected_variants|{'control'} for r in rounds);med={name:float(np.median([r['milliseconds'][name] for r in rounds])) for name in expected_variants|{'control'}};assert all(math.isfinite(v) and v>0 for v in med.values())
  for name in sorted(expected_variants):rows.append(dict(mode=mode,variant=name,block=state['variants'][name]['block'],warps=state['variants'][name]['warps'],current_ms=med['control'],candidate_ms=med[name],speedup_pct=(med['control']/med[name]-1)*100))
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 breakdown=state['breakdowns'];assert len(breakdown)==7 and {r['round'] for r in breakdown}==set(range(7));summary={name:float(np.median([r['milliseconds'][name] for r in breakdown])) for name in breakdown[0]['milliseconds']};assert all(math.isfinite(v) and v>0 for v in summary.values());(BASE/'BREAKDOWN_32g.json').write_text(json.dumps(summary,indent=2)+'\n');assert all(sha(p)==h for p,h in state['frozen_sources'].items()) and not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
 gate=dict(complete=True,CPU_only=True,sources_unchanged=True,checkpoint_unchanged=True,phases=3,variants=8,full_GPU_INT32_BF16_exact=True,full_A8_and_scales_independently_recomputed=True,selected_CPU_INT64_and_BF16_elements=count,all_graph_poison_matches_eager=True,raw_directory=str(root),raw_result_sha256=sha(root/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'),model_executed=False,official_benchmark=False,adopted=False);(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
