# SPDX-License-Identifier: Apache-2.0
import ast,os,sys
from common import *
assert not CPU.exists();CPU.mkdir();profile=json.loads(PROFILE.read_text());assert len(profile['frozen_sources'])==903
sources=dict(profile['frozen_sources']);sources.update(profile['controller_sources']);sources[str(PROFILE)]=sha(PROFILE)
for p in BASE.iterdir():
 if p.is_file():
  if p.suffix=='.py':ast.parse(p.read_text())
  sources[str(p)]=sha(p)
for p in (BASE.parent/'tianshu_native_dotsaddz_formats_20261023/FUNCTIONAL_GATE_32g.json',Path('/usr/local/corex-4.5.0/lib64/libcuinfer.so.7'),Path('/usr/local/corex-4.5.0/lib64/libcuinfer_blas.so'),Path('/usr/local/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so')):sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items());proof=json.loads((BASE.parent/'tianshu_native_dotsaddz_formats_20261023/FUNCTIONAL_GATE_32g.json').read_text());assert 'n_natural__unsigned_add' in proof['passed_variants']
checkpoint=sha(MODEL);assert checkpoint=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
files={};banks=[];refs=[]
for phase in PHASES:
 p=FIXTURE/f'M64_phase{phase}.npz';files[str(p)]=sha(p)
old=json.loads((FIXTURE/'result.json').read_text());assert old['complete']
for phase in PHASES:assert next(v for c in old['cases'] if c['M']==64 for v in c['phases'] if v['phase']==phase)['sha256']==files[str(FIXTURE/f'M64_phase{phase}.npz')]
for layer in range(42):
 w=weight(layer);q,s,p,z=quant(w);assert np.array_equal(unpack(p),q) and np.array_equal(f32(z),-8*f32(s));path=CPU/f'layer{layer:02}.npz';np.savez(path,W=p,S=s,Z=z);files[str(path)]=sha(path);banks.append(dict(layer=layer,original_sha256=ahash(w),pack_sha256=ahash(p),scale_sha256=ahash(s),zero_sha256=ahash(z),file=str(path),file_sha256=files[str(path)]))
 if layer in LAYERS:
  deq=bf16((q.astype(np.int16)-8).astype(np.float32)*np.repeat(f32(s),128,axis=0));dq=f32(deq).astype(np.float64);original=f32(w).T.astype(np.float64)
  for phase in PHASES:
   h=hwords(phase,layer);hf=f32(h).astype(np.float64);path=CPU/f'reference_L{layer}_phase{phase}.npz';np.savez(path,H=h,original=hf@original,quantized=hf@dq);files[str(path)]=sha(path);refs.append(dict(layer=layer,phase=phase,file=str(path),sha256=files[str(path)]))
 print(json.dumps(dict(layer=layer,CPU_pack_roundtrip_exact=True,FP64_references=layer in LAYERS)),flush=True)
assert all(sha(p)==h for p,h in sources.items()) and sha(MODEL)==checkpoint
assert not any(k.split('.')[0] in ('torch','triton','vllm','flag_gems') for k in sys.modules)
g=dict(complete=True,CPU_only=True,CUDA_initialized=False,checkpoint_sha256=checkpoint,sources=sources,files=files,banks=banks,references=refs,all_42_full_pack_roundtrip_exact=True,FP64_full_cells=9*64*2048*2,reference_quantized_weights_BF16_rounded=True,first_GPU_not_run=True,adopted=False)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps({k:v for k,v in g.items() if k not in ('sources','files','banks','references')}))
