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
 p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);args=p.parse_args();root=args.input.resolve();assert root.is_relative_to('/workspace/benchmark-results');state=json.loads((root/'result.json').read_text());assert state['complete'] and state['checkpoint_unchanged'] and state['all42_bank_hashes_before_after'] and sha(MODEL)==state['checkpoint_sha256_before']==state['checkpoint_sha256_after'];assert all(sha(p)==h for p,h in state['frozen_sources'].items()) and len(state['banks'])==42 and [c['M'] for c in state['cases']]==[64,128,512,4096]
 with MODEL.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
 artifacts=0;count=0
 for layer,proof in enumerate(state['banks']):
  assert proof['layer']==layer;info=header[f'model.layers.{layer}.mlp.down_proj.weight'];assert info['shape']==[2048,6144] and info['dtype']=='BF16';words=np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=(2048,6144));wf=f32(words);scale=np.maximum(abs(wf).max(1),np.float32(1e-8))/np.float32(127.);code=np.clip(np.rint(wf/scale[:,None]),-127,127).astype(np.int8);assert ahash(words)==proof['Down_original_BF16_sha256'] and ahash(code)==proof['Down_packed_sha256'] and ahash(scale)==proof['Down_scales_sha256']
  for case in state['cases']:
   m=case['M'];assert case['complete'] and len(case['phases'])==3
   for phase,entry in enumerate(case['phases']):
    assert entry['phase']==phase and entry['whole42_graph_poison_eager_verified'] and entry['guards_input_RNG'] and len(entry['banks'])==42 and len(entry['eager_output_hashes']['control'])==len(entry['eager_output_hashes']['candidate'])==42;row=entry['banks'][layer];assert row['layer']==layer and row['full_quantized_INT32_BF16_exact'] and sha(root/row['artifact'])==row['sha256'] and math.isfinite(row['relative_L2_error_vs_BF16'])
    with np.load(root/row['artifact'],allow_pickle=False) as data:
     r,c=data['rows'],data['columns'];assert np.array_equal(r,np.unique(np.linspace(0,m-1,8).astype(np.int64))) and np.array_equal(c,np.unique(np.linspace(0,2047,32).astype(np.int64)));h=f32(data['H_selected_rows_words']);assert h.shape==(8,6144) and np.isfinite(h).all() and np.count_nonzero(h[0])==0;asc=np.asarray(np.maximum(abs(h).max(1),np.float32(1e-8)).astype(np.float64)/127.,np.float32);aq=np.clip(np.rint(np.asarray(h.astype(np.float64)/asc[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8);assert np.array_equal(aq,data['Aq_selected_rows']) and np.array_equal(asc.view(np.uint32),data['As_selected_rows'].view(np.uint32));integer=aq.astype(np.int64)@code[c].astype(np.int64).T;assert np.array_equal(integer,data['integer_selected']);projected=bf16((integer.astype(np.float32)*asc[:,None])*scale[None,c]);assert np.array_equal(projected,data['candidate_selected_words']);native_ref=h.astype(np.float64)@wf[c].astype(np.float64).T;actual=f32(data['control_selected_words']).astype(np.float64);assert np.isfinite(actual).all() and np.all(abs(actual-native_ref)<=.02+.01*abs(native_ref));count+=integer.size
    artifacts+=1
 rows=[]
 for case in state['cases']:
  for scope in ('whole_MLP','Down_only'):
   rounds=[r for r in state['rounds'] if r['M']==case['M'] and r['scope']==scope];assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7)) and all(r['replays']==3 and r['banks_per_graph']==42 and set(r['order'])==set(r['milliseconds_per_bank'])=={'control','candidate'} for r in rounds);med={name:float(np.median([r['milliseconds_per_bank'][name] for r in rounds])) for name in ('control','candidate')};assert all(math.isfinite(v) and v>0 for v in med.values());rows.append(dict(M=case['M'],scope=scope,current_ms_per_bank=med['control'],candidate_ms_per_bank=med['candidate'],speedup_pct=(med['control']/med['candidate']-1)*100,minimum_relative_L2_error=min(b['relative_L2_error_vs_BF16'] for p in case['phases'] for b in p['banks']),maximum_relative_L2_error=max(b['relative_L2_error_vs_BF16'] for p in case['phases'] for b in p['banks'])))
 assert artifacts==504 and len(rows)==8 and all(sha(p)==h for p,h in state['frozen_sources'].items()) and not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 gate=dict(complete=True,CPU_only=True,all42_original_Down_W8_bank_hashes_recomputed=True,sources_unchanged=True,checkpoint_unchanged=True,phase_artifacts_verified=artifacts,selected_INT64_quantized_BF16_original_FP64_outputs=count,all42_full_GPU_quantized_INT32_BF16_exact=True,whole42_graph_poison_eager_hash_checks=True,numerical_scope='CPU all42 weight quantization, selected8 activation rows and32 outputs eachphase; GPU full integer/quantized output equality. Not full Transformer or full CPU GEMM.',model_executed=False,model_quality_not_tested=True,official_benchmark=False,adopted=False,raw_directory=str(root),raw_result_sha256=sha(root/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'));(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
