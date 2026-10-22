# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,statistics,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');ROOT=Path('/workspace/benchmark-results/20261022_native_int8_attention_projections_GPU_tianshu')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def f32(words):return (np.asarray(words,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16(v):
 bits=np.asarray(v,np.float32).view(np.uint32);return ((bits+32767+((bits>>16)&1))>>16).astype(np.uint16)
def pack(v):
 s=np.asarray(np.maximum(abs(v).max(1),np.float32(1e-8)).astype(np.float64)/127.,np.float32);return np.clip(np.rint(np.asarray(v.astype(np.float64)/s[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8),s
def main():
 assert not (BASE/'INDEPENDENT_CPU_GATE_32g.json').exists();r=read(ROOT/'result.json');assert r['complete'] and r['phase']=='complete' and r['checkpoint_before']==r['checkpoint_after']==sha(MODEL)
 for p,h in r['frozen_sources'].items():assert sha(p)==h and sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))==h
 with MODEL.open('rb') as stream:header_size=int.from_bytes(stream.read(8),'little');header=json.loads(stream.read(header_size))
 def get(role):
  info=header[f'model.layers.0.self_attn.{role}_proj.weight'];assert info['dtype']=='BF16';return np.array(np.memmap(MODEL,mode='r',dtype='<u2',offset=8+header_size+info['data_offsets'][0],shape=tuple(info['shape'])))
 weights=dict(QKV=np.concatenate([get(k) for k in ('q','k','v')],0),O=get('o'));count=0;phases=0;timings=[];errors=[];assert [(c['role'],c['M']) for c in r['cases']]==[(role,m) for role in ('QKV','O') for m in (64,128,512,4096)]
 for case in r['cases']:
  assert case['complete'];role=case['role'];m,n,k=(case[key] for key in ('M','N','K'));weight_words=weights[role];weight=f32(weight_words);q,ws=pack(weight);signature=case['weight_signature'];assert ahash(weight_words)==signature['original_BF16_sha256'] and ahash(q)==signature['packed_sha256'] and ahash(ws)==signature['scales_sha256'];assert len(case['phases'])==3
  for phase,p in enumerate(case['phases']):
   assert p['phase']==phase and p['guards_inputs_weights_RNG'] and sha(ROOT/p['artifact'])==p['sha256'];sign=1 if phase==0 else -1;factor=np.float32(.5 if phase==2 else 1.)
   with np.load(ROOT/p['artifact']) as a:
    x=f32(a['X_words']);assert x.shape==(m,k) and not x[0].any() and ahash(a['X_words'])==p['X_words_sha256'];aq,asc=pack(x);assert ahash(aq)==p['Aq_sha256'] and ahash(asc)==p['As_sha256'] and ahash(ws*factor)==p['WS_sha256'];rows,cols=a['rows'],a['columns'];integer=aq[rows].astype(np.int64)@(q[cols].astype(np.int64)*sign).T;assert np.array_equal(integer,a['integer_selected']);expected=bf16((integer.astype(np.float32)*asc[rows,None])*(ws*factor)[None,cols]);assert np.array_equal(expected,a['candidate_selected_words']);count+=integer.size
    ideal=x[rows].astype(np.float64)@(weight[cols]*sign*factor).astype(np.float64).T;absolute=abs(x[rows].astype(np.float64))@abs((weight[cols]*sign*factor).astype(np.float64)).T;actual=f32(a['control_selected_words']).astype(np.float64);eps=2.**-24;gamma=(k+1)*eps/(1-(k+1)*eps);bits=a['control_selected_words'];exponent=((bits&0x7f80)>>7).astype(np.int32);ulp=np.exp2(np.where(exponent==0,-133,exponent-134).astype(np.float64));assert np.all(abs(actual-ideal)<=gamma*absolute+.5*ulp+1.e-10)
   assert {v['name'] for v in p['entries']}=={'control','NN_B512','NN_B1024','NT_B512','NT_B1024'}
   for record in p['entries']:
    assert record['graph_poison_eager_exact']
    if record['name']!='control':assert record['full_INT32_exact'] and record['full_quantized_BF16_exact'] and record['output_sha256']==p['reference_output_sha256']
   phases+=1
  errors.append(dict(role=role,M=m,relative_L2_vs_BF16=[p['relative_L2_vs_BF16'] for p in case['phases']]))
  for mode in ('graph','eager'):
   rounds=[v for v in r['rounds'] if v['role']==role and v['M']==m and v['mode']==mode];assert len(rounds)==7 and {v['round'] for v in rounds}==set(range(7));values={name:statistics.median(v['milliseconds'][name] for v in rounds) for name in rounds[0]['milliseconds']}
   for name,ms in values.items():timings.append(dict(role=role,M=m,mode=mode,variant=name,median_ms=ms,speedup_vs_explicit_vendor_BF16_pct=100*(values['control']/ms-1)))
 with (BASE/'KEY_TIMINGS_32g.csv').open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(timings[0]));writer.writeheader();writer.writerows(timings)
 assert phases==24 and not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules);gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,cases=8,phases=phases,independent_INT64_exact_BF16_and_FP64_control_cells=count,full_GPU_INT32_and_BF16_exact=True,full_quantizer_codes_scales_and_static_weight_packing_exact=True,checkpoint_and_source_snapshots_verified=True,graph_poison_eager_canaries_inputs_weights_RNG_verified=True,timings=timings,numerical=errors,actual_service_route_not_yet_verified=True,new42bank_holdout_and_C8_C64_Math_required=True,model_executed=False,official_benchmark=False,adopted=False,raw_directory=str(ROOT),raw_result_sha256=sha(ROOT/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'));(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('timings','numerical')}))
if __name__=='__main__':main()
