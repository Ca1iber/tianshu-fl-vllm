# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,statistics,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');ROOT=Path('/workspace/benchmark-results/20261022_native_int8_attention42_GPU_tianshu')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def f32(words):return (np.asarray(words,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16(v):
 bits=np.asarray(v,np.float32).view(np.uint32);return ((bits+32767+((bits>>16)&1))>>16).astype(np.uint16)
def pack(v):
 s=np.asarray(np.maximum(abs(v).max(1),np.float32(1e-8)).astype(np.float64)/127.,np.float32);return np.clip(np.rint(np.asarray(v.astype(np.float64)/s[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8),s
def main():
 assert not (BASE/'INDEPENDENT_CPU_GATE_32g.json').exists();r=json.loads((ROOT/'result.json').read_text());assert r['complete'] and r['all84_bank_hashes_before_after'] and r['checkpoint_before']==r['checkpoint_after']==sha(MODEL) and len(r['banks'])==84 and [c['M'] for c in r['cases']]==[128,512,4096]
 for p,h in r['frozen_sources'].items():assert sha(p)==h and sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))==h
 with MODEL.open('rb') as stream:header_size=int.from_bytes(stream.read(8),'little');header=json.loads(stream.read(header_size))
 def get(layer,role):
  info=header[f'model.layers.{layer}.self_attn.{role}_proj.weight'];assert info['dtype']=='BF16';return np.array(np.memmap(MODEL,mode='r',dtype='<u2',offset=8+header_size+info['data_offsets'][0],shape=tuple(info['shape'])))
 count=artifacts=0
 for index,proof in enumerate(r['banks']):
  role=proof['role'];layer=proof['layer'];assert (layer,role)==(index//2,'QKV' if index%2==0 else 'O');words=np.concatenate([get(layer,k) for k in ('q','k','v')],0) if role=='QKV' else get(layer,'o');weight=f32(words);q,ws=pack(weight);assert ahash(words)==proof['original_BF16_sha256'] and ahash(q)==proof['packed_sha256'] and ahash(ws)==proof['scales_sha256'];n,k=weight.shape
  for case in r['cases']:
   assert case['complete'] and len(case['phases'])==2
   for p in case['phases']:
    assert p['whole84_graph_poison_eager_exact'] and p['inputs_RNG_canaries_unchanged'] and len(p['banks'])==84 and p['seed']==202610233000+case['M']+p['phase'];record=p['banks'][index];assert record['index']==index and record['full_INT32_BF16_exact'] and sha(ROOT/record['artifact'])==record['sha256']
    with np.load(ROOT/record['artifact']) as a:
     x=f32(a['X_selected_words']);assert x.shape==(8,k) and not x[0].any();aq,asc=pack(x);assert np.array_equal(aq,a['Aq_selected']) and np.array_equal(asc,a['As_selected']);cols=a['columns'];integer=aq.astype(np.int64)@q[cols].astype(np.int64).T;assert np.array_equal(integer,a['integer_selected']) and np.array_equal(bf16((integer.astype(np.float32)*asc[:,None])*ws[None,cols]),a['candidate_selected_words']);ideal=x.astype(np.float64)@weight[cols].astype(np.float64).T;absolute=abs(x.astype(np.float64))@abs(weight[cols].astype(np.float64)).T;bits=a['control_selected_words'];actual=f32(bits).astype(np.float64);exponent=((bits&0x7f80)>>7).astype(np.int32);ulp=np.exp2(np.where(exponent==0,-133,exponent-134).astype(np.float64));eps=2.**-24;gamma=(k+1)*eps/(1-(k+1)*eps);assert np.all(abs(actual-ideal)<=gamma*absolute+.5*ulp+1e-10);count+=integer.size
    artifacts+=1
 timings=[]
 for m in (128,512,4096):
  for scope in ('QKV','O','combined_pair'):
   rounds=[v for v in r['rounds'] if v['M']==m and v['scope']==scope];assert len(rounds)==7 and {v['round'] for v in rounds}==set(range(7));values={mode:statistics.median(v['milliseconds_per_layer'][mode] for v in rounds) for mode in ('control','candidate')};timings.append(dict(M=m,scope=scope,BF16_ms_per_layer=values['control'],INT8_ms_per_layer=values['candidate'],speedup_pct=100*(values['control']/values['candidate']-1)))
 assert artifacts==504 and not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
 with (BASE/'KEY_TIMINGS_32g.csv').open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(timings[0]));writer.writeheader();writer.writerows(timings)
 gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,all84_original_and_W8_scale_hashes_recomputed=True,independent_INT64_exact_BF16_and_FP64_control_cells=count,artifacts_checked=artifacts,full_GPU_native_and_Triton_INT32_BF16_exact=True,full84_graph_poison_eager_exact=True,sources_snapshots_checkpoint_inputs_weights_RNG_canaries_verified=True,actual_service_route_not_yet_verified=True,model_executed=False,official_benchmark=False,adopted=False,new_full_C8_C64_Math_required=True,timings=timings,raw_directory=str(ROOT),raw_result_sha256=sha(ROOT/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'));(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
