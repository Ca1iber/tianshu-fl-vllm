# SPDX-License-Identifier: Apache-2.0
import ast,sys
from common import *
assert not CPU.exists();CPU.mkdir();old=json.loads((OLD/'CPU_GATE_32g.json').read_text());assert sha(LOGICAL)==old['logical_input_sha256'];a=np.load(LOGICAL);q,zero,sn=a['q'],a['zero'],a['scale_num'];integer=a['xi'].astype(np.int64)@((q.astype(np.int64)-np.repeat(zero.astype(np.int64),128,axis=0))*np.repeat(sn.astype(np.int64),128,axis=0));expected=bf16(integer.astype(np.float32)/128);assert np.array_equal(expected,a['reference']) and np.array_equal(integer,a['integer_reference']);rows=[]
for name in VARIANTS:
 layout,sem=name.split('__');order=[0,2,4,6,1,3,5,7] if layout.endswith('permuted') else list(range(8));codes=q^np.uint8(8) if sem.startswith('signed') else q
 if layout.startswith('n_'):w=pack(codes,order);decoded=unpack(w,order)
 elif layout.startswith('kn_'):w=pack(codes.T,order).T.copy();decoded=unpack(w.T.copy(),order).T
 else:w=pack(codes.T,order);decoded=unpack(w,order).T
 assert np.array_equal(decoded,codes)
 if sem=='unsigned_add':z=-zero*sn/8
 elif sem in ('center_add','signed_add'):z=(8-zero)*sn/8
 elif sem=='unsigned_subtract':z=zero
 else:z=zero-8
 path=CPU/f'{name}.npz';np.savez(path,X=bf16(a['xi'].astype(np.float32)/16),W=w.copy(),S=bf16(sn/8),Z=bf16(z));rows.append(dict(name=name,file=str(path),sha256=sha(path),shape=list(w.shape),complete_roundtrip_exact=True))
paths=list(BASE.glob('*.py'))+[BASE/'PLAN_32g.md',BASE/'run.sh',LOGICAL,OLD/'FUNCTIONAL_GATE_32g.json',Path('/usr/local/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so'),Path('/usr/local/corex-4.5.0/lib64/libcuinfer.so.7'),Path('/usr/local/corex-4.5.0/lib64/libcuinfer_blas.so')];sources={str(p):sha(p) for p in paths}
for p in BASE.glob('*.py'):ast.parse(p.read_text())
assert not any(k.split('.')[0] in ('torch','triton','vllm','flag_gems') for k in sys.modules)
g=dict(complete=True,CPU_only=True,CUDA_initialized=False,variants=rows,sources=sources,all_30_CPU_roundtrips_exact=True,all_16384_full_K_INT64_reference_exact=True,model_executed=False,GPU_not_run=True,performance_not_tested=True,adopted=False);(BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps({k:v for k,v in g.items() if k not in ('sources','variants')}))
