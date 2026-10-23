# SPDX-License-Identifier: Apache-2.0
import ast,json
from inputs import *
import torch
assert not ROOT.exists() and not (BASE/'CPU_GATE_32g.json').exists() and not torch.cuda.is_initialized();ROOT.mkdir()
for p in BASE.glob('*.py'):ast.parse(p.read_text())
m,k,n=64,256,256;xi=((3*np.arange(m)[:,None]+5*np.arange(k)[None,:])%17-8).astype(np.int16);q=((3*np.arange(k)[:,None]+7*np.arange(n)[None,:]+5*(np.arange(k)[:,None]//128))%16).astype(np.uint8);zero=(4+(np.arange(2)[:,None]+np.arange(n)[None,:])%8).astype(np.int16);scale_num=((np.arange(2)[:,None]+1)*(1+np.arange(n)[None,:]%4)).astype(np.int16);deq=(q.astype(np.int16)-np.repeat(zero,128,axis=0))*np.repeat(scale_num,128,axis=0);integer=xi.astype(np.int64)@deq.astype(np.int64);reference=bf16(integer.astype(np.float32)/128);x=bf16(xi.astype(np.float32)/16);scales=bf16(scale_num.astype(np.float32)/8);offsets=bf16((8-zero)*scale_num/8);params=np.stack([scales,offsets],axis=-1);np.savez(ROOT/'logical.npz',xi=xi,q=q,zero=zero,scale_num=scale_num,integer_reference=integer,reference=reference)
# Independent native CPU operator uses its own converter, not either GPU byte format.
cpu_q=torch.from_numpy(q.T.copy().astype(np.int32));cpu_pack=torch.ops.aten._convert_weight_to_int4pack_for_cpu.default(cpu_q,8);cpu_x=torch.from_numpy(x.copy()).view(torch.bfloat16);cpu_params=torch.from_numpy(params.copy()).view(torch.bfloat16);out=torch.ops.aten._weight_int4pack_mm_for_cpu.default(cpu_x,cpu_pack,128,cpu_params);assert np.array_equal(out.contiguous().view(torch.uint16).numpy(),reference)
rows=[]
for name in VARIANTS:
 if name.startswith('generic_'):
  a=q.T[:,::2];b=q.T[:,1::2];w=(a|(b<<4)) if name=='generic_low_even' else ((a<<4)|b);decoded=np.empty((n,k),dtype=np.uint8)
  if name=='generic_low_even':decoded[:,::2]=w&15;decoded[:,1::2]=w>>4
  else:decoded[:,::2]=w>>4;decoded[:,1::2]=w&15
  assert np.array_equal(decoded,q.T);z=np.empty((0,),dtype=np.uint16)
 else:
  order=permutation(name);w=pack32(q,order);assert np.array_equal(unpack32(w,order),q)
  if name.endswith('zeros_packed'):z=pack32(zero.astype(np.uint8),order);assert np.array_equal(unpack32(z,order),zero)
  elif name.endswith('zeros_bf16'):z=bf16(zero);assert np.array_equal(f32(z),zero)
  else:z=bf16(-zero*scale_num/8);assert np.array_equal(f32(z),-zero*scale_num/8)
 path=ROOT/f'{name}.npz';np.savez(path,X=x,W=w.copy(),S=scales,Z=z,P=params,REF=reference);tx=torch.empty((m,k),device='meta',dtype=torch.bfloat16);tw=torch.empty(w.shape,device='meta',dtype=torch.uint8 if w.dtype==np.uint8 else torch.int32)
 if name.startswith('generic_'):converted=torch.ops.aten._convert_weight_to_int4pack.default(tw,8);meta=torch.ops.aten._weight_int4pack_mm.default(tx,converted,128,torch.empty((2,n,2),device='meta',dtype=torch.bfloat16))
 else:meta=torch.ops.aten._weight_int4pack_mm_awq.default(tx,tw,128,torch.empty((2,n),device='meta',dtype=torch.bfloat16),torch.empty(z.shape,device='meta',dtype=torch.int32 if z.dtype==np.int32 else torch.bfloat16))
 assert list(meta.shape)==[m,n];rows.append(dict(name=name,file=str(path),sha256=sha(path),weight_shape=list(w.shape),weight_dtype=str(w.dtype),zero_shape=list(z.shape),zero_dtype=str(z.dtype),packing_hypothesis=True,all_CPU_logical_physical_roundtrips_exact=True,actual_Meta_shape_passed=True))
header=Path('/usr/local/corex-4.5.0/include/ixinfer.h');contract=Path('/workspace/benchmark-results/20261023_native_AWQ_binary_CPU_tianshu/CONTRACT_STRINGS.txt');paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run.sh',BASE/'run_after_smoke.sh',header,contract,Path('/usr/local/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so'),Path('/usr/local/corex-4.5.0/lib64/libcuinfer.so.7').resolve(),Path('/usr/local/corex-4.5.0/lib64/libcuinfer_blas.so').resolve()};sources={str(p):sha(p) for p in paths};assert not torch.cuda.is_initialized();result=dict(complete=True,CPU_only=True,CUDA_initialized=False,variants=rows,shape=[m,k,n],groups=2,group_size=128,all_65536_logical_weight_values_generated_deterministically=True,all_16384_full_K_INT64_projection_references=True,native_CPU_generic_complete_BF16_reference_exact=True,nonconstant_rows_columns_groups_and_zeros=True,Meta_is_not_GPU_support_proof=True,AWQ_formats_are_predeclared_hypotheses_not_validated_contract=True,torch_version=torch.__version__,logical_input=str(ROOT/'logical.npz'),logical_input_sha256=sha(ROOT/'logical.npz'),sources=sources,GPU_not_run=True,performance_not_tested=True,real_model_weights_not_tested=True);(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('sources','variants')}))
