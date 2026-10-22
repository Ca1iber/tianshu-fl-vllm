# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,statistics,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/workspace/benchmark-results/20261022_down_W8A16_v2_GPU_tianshu');MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def values(words):return (np.asarray(words,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16bits(x):
 bits=np.asarray(x,np.float32).view(np.uint32);return ((bits+32767+((bits>>16)&1))>>16).astype(np.uint16)
def main():
 assert not (BASE/'INDEPENDENT_CPU_GATE_32g.json').exists();r=read(ROOT/'result.json');assert r['complete'] and r['phase']=='complete' and r['checkpoint_before']==r['checkpoint_after']==sha(MODEL)
 for path,digest in r['frozen_sources'].items():assert sha(path)==digest and sha(ROOT/'source_snapshot'/Path(path).relative_to('/'))==digest
 with MODEL.open('rb') as stream:size=int.from_bytes(stream.read(8),'little');header=json.loads(stream.read(size))
 entry=header['model.layers.0.mlp.down_proj.weight'];assert entry['dtype']=='BF16' and entry['shape']==[2048,6144];w=values(np.memmap(MODEL,mode='r',dtype='<u2',offset=8+size+entry['data_offsets'][0],shape=(2048,6144)));scale=np.maximum(np.max(np.abs(w),axis=1),np.float32(1e-8))/np.float32(127.);q=np.clip(np.rint(w/scale[:,None]),-127,127).astype(np.int8);wd=w.astype(np.float64);qd=q.astype(np.float64);tiny=np.finfo(np.float64).tiny;eps32=2.**-24;gamma32=6145*eps32/(1-6145*eps32);eps64=2.**-53;gamma64=6145*eps64/(1-6145*eps64);checked=0;results=[]
 for phase in r['phases']:
  assert phase['graph_poison_eager_exact'] and phase['guards_inputs_weights_RNG'];path=ROOT/phase['artifact'];assert sha(path)==phase['sha256']
  with np.load(path) as a:
   assert np.array_equal(q,a['Down_packed']) and np.array_equal(scale,a['Down_scales']);h=values(a['H_words']).astype(np.float64);ideal=(h@qd.T)*scale.astype(np.float64)[None,:];absdot=(np.abs(h)@np.abs(qd).T)*scale.astype(np.float64)[None,:];original=h@wd.T;record=dict(phase=phase['phase'],seed=phase['seed'],variants={})
   for configuration in r['configurations']:
    name=configuration['name'];bits=a[name+'_words'];out=values(bits).astype(np.float64);exponent=((bits&0x7f80)>>7).astype(np.int32);ulp=np.exp2(np.where(exponent==0,-133,exponent-134).astype(np.float64));bound=(gamma32+gamma64)*absdot+.5*ulp;difference=np.abs(out-ideal);assert np.isfinite(out).all() and np.all(difference<=bound) and np.all(out[0]==0)
    digest=hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest();assert digest==phase['output_hashes'][name];checked+=out.size;record['variants'][name]=dict(full_output_cells=out.size,maximum_bound_utilization=float(np.max(difference/np.maximum(bound,tiny))),relative_L2_vs_ideal_W8A16=float(np.linalg.norm(out-ideal)/np.linalg.norm(ideal)),different_BF16_cells_vs_ideal=int(np.count_nonzero(bits!=bf16bits(ideal))),relative_L2_vs_original_BF16_ideal=float(np.linalg.norm(out-original)/np.linalg.norm(original)))
  results.append(record)
 rows=[]
 for scope in ('whole_MLP','Down_only'):
  for mode in ('graph','eager'):
   rounds=[v for v in r['rounds'] if v['scope']==scope and v['mode']==mode];assert len(rounds)==7
   medians={name:statistics.median(v['milliseconds'][name] for v in rounds) for name in rounds[0]['milliseconds']}
   for name,ms in medians.items():rows.append(dict(M=64,scope=scope,mode=mode,variant=name,median_ms=ms,speedup_vs_service_BF16_pct=(medians['service_BF16']/ms-1)*100,speedup_vs_torch_BF16_pct=(medians['torch_BF16']/ms-1)*100))
 with (BASE/'KEY_TIMINGS_32g.csv').open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
 gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,independent_full_FP64_BF16_times_INT8_dot_cells=checked,predeclared_forward_error_bound=dict(gamma32=gamma32,gamma64=gamma64,rounding='one half BF16 output ULP'),all_full_outputs_within_bound=True,original_checkpoint_and_source_snapshots_exact=True,weight_codes_scales_exact_CPU=True,independent_fixtures=2,single_bank_cache_screen_only=True,full42_bank_holdout_still_required=True,new_full_C8_C64_Math_required=True,model_executed=False,official_benchmark=False,adopted=False,numerical=results,timings=rows,raw_directory=str(ROOT),raw_result_sha256=sha(ROOT/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'));(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('numerical','timings')}));print(json.dumps(sorted((v for v in rows if v['scope']=='whole_MLP' and v['mode']=='graph'),key=lambda v:v['median_ms'])[:5]))
if __name__=='__main__':main()
