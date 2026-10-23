# SPDX-License-Identifier: Apache-2.0
import ast,sys
from common import *
assert not CPU.exists();CPU.mkdir();profile=json.loads(PROFILE.read_text());assert len(profile['frozen_sources'])==903;sources=dict(profile['frozen_sources']);sources.update(profile['controller_sources']);sources[str(PROFILE)]=sha(PROFILE)
for p in BASE.iterdir():
 if p.is_file():
  if p.suffix=='.py':ast.parse(p.read_text())
  sources[str(p)]=sha(p)
for p in [EXP/'tianshu_native_dotsaddz_formats_20261023/FUNCTIONAL_GATE_32g.json',EXP/'tianshu_native_W4_formats_20261023/FUNCTIONAL_GATE_32g.json',Path('/usr/local/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so'),Path('/usr/local/corex-4.5.0/lib64/libcuinfer.so.7'),Path('/usr/local/corex-4.5.0/lib64/libcuinfer_blas.so')]:sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items());checkpoint=sha(MODEL);assert checkpoint=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c';w=weight();wf=f32(w);s8=np.maximum(abs(wf).max(1),np.float32(1e-8))/np.float32(127);q8=np.clip(np.rint(wf/s8[:,None]),-127,127).astype(np.int8);expected=next(r for r in json.loads((EXP/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text()) if r['prefix']=='lm_head');assert all(ahash(x)==expected[k] for k,x in [('original_BF16_sha256',w),('packed_sha256',q8),('scales_sha256',s8)])
grouped=wf.reshape(N,16,128);s4=bf16(np.maximum(abs(grouped).max(2),np.float32(1e-8))/np.float32(7));q4=(np.clip(np.rint(grouped/f32(s4)[:,:,None]),-8,7).astype(np.int8)+8).astype(np.uint8).reshape(N,K).T.copy();s4=s4.T.copy();awq=pack(q4,[0,2,4,6,1,3,5,7]);dots=pack(q4,list(range(8)));assert np.array_equal(unpack(awq,[0,2,4,6,1,3,5,7]),q4) and np.array_equal(unpack(dots,list(range(8))),q4);za=pack(np.full((16,N),8,np.uint8),[0,2,4,6,1,3,5,7]);zd=bf16(-8*f32(s4));files={}
def save(name,a):
 p=CPU/f'{name}.npy';np.save(p,a,allow_pickle=False);files[str(p)]=sha(p)
for name,a in [('W8',q8),('S8',s8),('WA',awq),('WD',dots),('S4',s4),('ZA',za),('ZD',zd)]:save(name,a)
print(json.dumps(dict(packing_complete=True,W8_signature_exact=True,all_267386880_W4_values_roundtrip_exact=True)),flush=True)
deq=f32(bf16((q4.astype(np.int16)-8).astype(np.float32)*np.repeat(f32(s4),128,axis=0))).astype(np.float64);original=wf.T.astype(np.float64);del wf,grouped,awq,dots,q4
for phase in range(3):
 x=bf16(np.random.default_rng(202610231900+phase).standard_normal((M,K),dtype=np.float32));x[0]=0;save(f'X{phase}',x);aq,asc=quant_a(x);integer=aq.astype(np.int64)@q8[COLS].astype(np.int64).T;exact=bf16((integer.astype(np.float32)*asc[:,None])*s8[COLS][None,:]);save(f'INT64_selected{phase}',integer);save(f'W8_exact_selected{phase}',exact);save(f'Aq{phase}',aq);save(f'As{phase}',asc);save(f'original_FP64_{phase}',f32(x).astype(np.float64)@original);save(f'quantized_FP64_{phase}',f32(x).astype(np.float64)@deq);print(json.dumps(dict(phase=phase,all_full_FP64_complete=True)),flush=True)
assert all(sha(p)==h for p,h in sources.items()) and sha(MODEL)==checkpoint and not any(k.split('.')[0] in ('torch','triton','vllm','flag_gems') for k in sys.modules)
g=dict(complete=True,CPU_only=True,CUDA_initialized=False,checkpoint_sha256=checkpoint,sources=sources,files=files,W8_signature=expected,all_W4_full_roundtrips_exact=True,full_FP64_cells=3*M*N*2,INT64_selected_cells=3*M*len(COLS),adopted=False,GPU_not_run=True);(BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps({k:v for k,v in g.items() if k not in ('sources','files','W8_signature')}))
