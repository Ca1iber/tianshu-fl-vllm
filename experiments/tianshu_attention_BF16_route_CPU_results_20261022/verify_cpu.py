# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,statistics,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');ROOT=Path('/workspace/benchmark-results/20261022_attention_BF16_route_GPU_tianshu')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def f32(words):return (np.asarray(words,np.uint16).astype(np.uint32)<<16).view(np.float32)
def main():
 assert not (BASE/'CPU_GATE_32g.json').exists();r=json.loads((ROOT/'result.json').read_text());assert r['complete'] and r['checkpoint_before']==r['checkpoint_after']==sha(MODEL)
 for p,h in r['sources'].items():assert sha(p)==h and sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))==h
 with MODEL.open('rb') as stream:size=int.from_bytes(stream.read(8),'little');header=json.loads(stream.read(size))
 def get(name):
  e=header[f'model.layers.0.self_attn.{name}_proj.weight'];assert e['dtype']=='BF16';return f32(np.array(np.memmap(MODEL,dtype='<u2',mode='r',offset=8+size+e['data_offsets'][0],shape=tuple(e['shape']))))
 weights=dict(QKV=np.concatenate([get(k) for k in ('q','k','v')],0),O=get('o'));count=0
 for case in r['cases']:
  m,n,k=(case[v] for v in ('M','N','K'));assert case['python_warm_call_trace']==[dict(function='linear',source=r['flag_gems_linear_source'],line=117)]
  for phase in case['phases']:
   assert phase['graph_eager_poison_exact'] and phase['input_weight_RNG_unchanged'] and phase['different_full_BF16_cells']==0;path=ROOT/phase['artifact'];assert sha(path)==phase['sha256']
   with np.load(path) as a:
    x=f32(a['X_words']).astype(np.float64);w=weights[case['role']][a['columns']].astype(np.float64);ideal=x@w.T;absolute=abs(x)@abs(w).T;eps=2.**-24;gamma=(k+1)*eps/(1-(k+1)*eps)
    for name in ('actual_unquantized_apply','explicit_vendor_mm'):
     bits=a[name+'_words'];actual=f32(bits).astype(np.float64);exponent=((bits&0x7f80)>>7).astype(np.int32);ulp=np.exp2(np.where(exponent==0,-133,exponent-134).astype(np.float64));assert np.isfinite(actual).all() and np.all(abs(actual-ideal)<=gamma*absolute+.5*ulp+1e-10);count+=actual.size
 rows=[]
 for role in ('QKV','O'):
  for m in (64,128,512,4096):
   for mode in ('graph','eager'):
    rounds=[v for v in r['rounds'] if v['role']==role and v['M']==m and v['mode']==mode];assert len(rounds)==7 and {v['round'] for v in rounds}==set(range(7));ms={name:statistics.median(v['milliseconds'][name] for v in rounds) for name in rounds[0]['milliseconds']};rows.append(dict(role=role,M=m,mode=mode,**ms))
 with (BASE/'KEY_TIMINGS_32g.csv').open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 assert count==8192 and not any(v.split('.')[0] in ('torch','triton','vllm','flag_gems') for v in sys.modules);gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,FP64_control_cells=count,sources_snapshots_checkpoint_verified=True,full_outputs_actual_and_explicit_mm_BF16_equal=True,eager_actual_unquantized_apply_calls_vendor_linear=True,full_compiled_model_route_not_yet_verified=True,baseline_diagnostic_only=True,model_executed=False,official_benchmark=False,adopted=False,timings=rows,raw_directory=str(ROOT),raw_result_sha256=sha(ROOT/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'));(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='timings'}))
if __name__=='__main__':main()
