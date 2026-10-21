# SPDX-License-Identifier: Apache-2.0
"""Independent original-W/A quantization and every saved NN INT32 output."""
import hashlib,json,os,statistics,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',VLLM_PLUGINS='',PYTHONPATH='')
import numpy as np
import torch
from safetensors import safe_open
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261021_QKV_NN_decode_GPU_tianshu')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
state=read(ROOT/'result.json');assert state['complete'] and state['phase']=='complete' and not torch.cuda.is_initialized()
for p,h in state['frozen_sources'].items():assert sha(p)==h==sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))
assert len(state['cases'])==34 and len(state['rounds'])==34*7
with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as ck:
 wc=torch.cat([ck.get_tensor('model.layers.0.self_attn.'+k+'_proj.weight') for k in ('q','k','v')]).contiguous()
f=wc.float().numpy();ws=np.maximum(np.max(np.abs(f),axis=1),np.float32(1.e-8))/np.float32(127);wq=np.clip(np.rint(f/ws[:,None]),-127,127).astype(np.int8)
sig=read(BASE/'CPU_GATE_32g.json')['weight_signature'];assert thash(wc)==sig['original_BF16_sha256'] and hashlib.sha256(wq.tobytes()).hexdigest()==sig['packed_sha256'] and hashlib.sha256(ws.tobytes()).hexdigest()==sig['scales_sha256']
refs={};samples=0;rows=[];seen=set();torch.set_num_threads(8)
generator=torch.Generator().manual_seed(202610212100);fixtures={}
for m in (8,64):
 a=torch.randn((m,2048),generator=generator,dtype=torch.bfloat16);a[0].zero_();a[1].zero_();a[1,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],dtype=torch.bfloat16);fixtures[m]=a
for row in state['cases']:
 m,ti=row['M'],row['tile_id'];assert (m,ti) not in seen;seen.add((m,ti));assert row['complete'] and not row.get('compile_failed') and len(row['phases'])==3
 native_selected_all=True
 for phase in row['phases']:
  p=phase['phase'];data=torch.load(ROOT/phase['artifact'],map_location='cpu',weights_only=False);a=data['a'];assert torch.equal(a,fixtures[m]*(1. if p==0 else .5 if p==1 else -.5));v=a.float().numpy();asc=np.maximum(np.max(np.abs(v),axis=1),np.float32(1.e-8))/np.float32(127);aq=np.clip(np.rint(v/asc[:,None]),-127,127).astype(np.int8)
  assert np.array_equal(aq,data['q'].numpy()) and np.array_equal(asc,data['ascale'].numpy()) and torch.equal(data['original_W_rows'],wc[data['columns']])
  key=(m,p);sign=1 if p==0 else -1;factor=.5 if p==2 else 1.
  if key not in refs:refs[key]=(torch.from_numpy(aq.astype(np.int64))@torch.from_numpy((wq*sign).astype(np.int64)).T,aq.copy())
  integer,expected_a=refs[key];assert np.array_equal(expected_a,aq)
  assert torch.equal(data['full_GPU_integer'].to(torch.int64),integer)
  expected=(integer.float()*torch.from_numpy(asc)[:,None]*torch.from_numpy(ws*factor)[None,:]).to(torch.bfloat16);actual=data['full_GPU_BF16'];assert torch.isfinite(actual).all() and ((actual.float()-expected.float()).abs()<=.02+.01*expected.float().abs()).all()
  assert phase['full_INT32_bitwise'] and phase['full_BF16_bitwise'] and phase['guards'] and phase['RNG_unchanged'] and phase['full_weights_immutable']
  original=(a[data['rows']].double()@(wc[data['columns']]*sign*factor).double().T).to(torch.bfloat16);native=data['native_BF16_selected'];native_selected=bool(torch.isfinite(native).all() and ((native.float()-original.float()).abs()<=.02+.01*original.float().abs()).all())
  native_selected_all = native_selected_all and native_selected
  samples+=integer.numel()
 ts=[r for r in state['rounds'] if r['M']==m and r['tile_id']==ti];assert len(ts)==7 and all(r['replays']==40 for r in ts)
 times={n:statistics.median(r[n] for r in ts) for n in ('NT','NN','BF16')};eligible=native_selected_all and all(p['native_BF16_full_valid'] for p in row['phases'])
 rows.append(dict(M=m,tile_id=ti,NN_configuration=row['NN_configuration'],NT_configuration=row['NT_configuration'],native_BF16_eligible=eligible,medians_ms=times,NN_vs_BF16_percent=100*(times['BF16']/times['NN']-1),NN_vs_NT_percent=100*(times['NT']/times['NN']-1)))
assert seen=={(m,t) for m in (8,64) for t in range(17)} and len(refs)==6 and not torch.cuda.is_initialized()
best=[]
for m in (8,64):
 eligible=[r for r in rows if r['M']==m and r['native_BF16_eligible']]
 if eligible:best.append(max(eligible,key=lambda r:r['NN_vs_BF16_percent']))
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,component_only=True,model_executed=False,adopted=False,all_34_cases_102_dynamic_phases_verified=True,all_saved_INT32_elements_independently_CPU_recomputed=samples,full_original_checkpoint_quantization_verified=True,raw_root=str(ROOT),raw_result_sha256=sha(ROOT/'result.json'),frozen_sources=state['frozen_sources'],best_by_M=best,invalid_BF16_reference_cases=[dict(M=r['M'],tile_id=r['tile_id']) for r in rows if not r['native_BF16_eligible']],service_candidate_worth_testing=any(r['M']==64 and r['NN_vs_BF16_percent']>10 for r in best),verifier_sha256=sha(__file__))
(BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'KEY_TIMINGS_32g.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# QKV decode NN：独立组件结果','','完整候选INT32由原始checkpoint独立CPU重算；BF16误差未当成等价整数参考。三臂时间包含动态A量化。不是服务或正式成绩。','', '|M|BF16 ms|NT ms|NN ms|NN vs BF16|NN vs NT|','|---:|---:|---:|---:|---:|---:|']
for r in best:
 t=r['medians_ms'];lines.append(f"|{r['M']}|{t['BF16']:.6f}|{t['NT']:.6f}|{t['NN']:.6f}|{r['NN_vs_BF16_percent']:+.3f}%|{r['NN_vs_NT_percent']:+.3f}%|")
lines+=['','接入模型仍需实际42层权重/profiler计量、相同配置小4K、真实C64新Math105>95%及正式4K。'];(BASE/'REPORT_32g.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k!='frozen_sources'}))
