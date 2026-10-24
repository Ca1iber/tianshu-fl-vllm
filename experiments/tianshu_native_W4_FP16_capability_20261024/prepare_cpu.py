# SPDX-License-Identifier: Apache-2.0
"""Real layer0/M64 native dtype capability, full independent FP64 references."""
import ast,sys
from common import *
assert not ROOT.exists();CPU.mkdir(parents=True)
gate=json.loads((PARENT/'CPU_GATE_32g.json').read_text())
proof=json.loads((PARENT/'INDEPENDENT_CPU_GATE_32g.json').read_text())
assert gate['complete'] and proof['complete'] and proof['all_numeric_passed']
bank=next(b for b in gate['banks'] if b['layer']==0)
assert sha(bank['file'])==bank['file_sha256']
a=np.load(bank['file'],allow_pickle=False)
w,s,z=a['W'],a['S'],a['Z'];q=parent.unpack(w)
assert ahash(w)==bank['pack_sha256'] and ahash(s)==bank['scale_sha256'] and ahash(z)==bank['zero_sha256']
sf,zf=f32(s),f32(z);sh,zh=sf.astype(np.float16),zf.astype(np.float16)
assert np.isfinite(sh).all() and np.isfinite(zh).all()
scale_exact=bool(np.array_equal(sh.astype(np.float32),sf));zero_exact=bool(np.array_equal(zh.astype(np.float32),zf))
deq=q.astype(np.float32)*np.repeat(sf,128,axis=0)+np.repeat(zf,128,axis=0)
bf=f32(parent.bf16(deq)).astype(np.float64)
half=(q.astype(np.float32)*np.repeat(sh.astype(np.float32),128,axis=0)+np.repeat(zh.astype(np.float32),128,axis=0)).astype(np.float16).astype(np.float64)
files={bank['file']:bank['file_sha256']};refs=[]
for phase in (0,1,2):
    h=parent.hwords(phase,0);x=f32(h);xh=x.astype(np.float16)
    assert np.isfinite(xh).all()
    path=CPU/f'reference_phase{phase}.npz'
    np.savez(path,H=h,H_half_words=xh.view(np.uint16),S_half_words=sh.view(np.uint16),Z_half_words=zh.view(np.uint16),
        reference_BF16=x.astype(np.float64)@bf,reference_FP16=xh.astype(np.float64)@half)
    files[str(path)]=sha(path)
    refs.append(dict(phase=phase,file=str(path),sha256=files[str(path)],
        activation_changed_cells=int(np.count_nonzero(xh.astype(np.float32)!=x)),
        activation_max_conversion_abs=float(abs(xh.astype(np.float32)-x).max())))
    print(json.dumps(refs[-1]),flush=True)
sources={str(p):sha(p) for p in set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',PARENT/'common.py',PARENT/'CPU_GATE_32g.json',PARENT/'INDEPENDENT_CPU_GATE_32g.json'}}
libraries={p:h for p,h in gate['sources'].items() if Path(p).name in ('libtorch_cuda.so','libcuinfer.so.7','libcuinfer_blas.so')}
assert len(libraries)==3 and all(sha(p)==h for p,h in libraries.items())
assert sha(parent.MODEL)==gate['checkpoint_sha256']=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
for p,h in sources.items():
    if Path(p).suffix=='.py':ast.parse(Path(p).read_text())
    target=CPU/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(Path(p).read_bytes());assert sha(target)==h
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
result=dict(complete=True,CPU_only=True,sources=sources,libraries=libraries,files=files,bank=bank,references=refs,
    shape=dict(M=64,K=6144,N=2048,group=128),scale_conversion_exact_on_actual_bank=scale_exact,
    zero_conversion_exact_on_actual_bank=zero_exact,checkpoint_sha256=gate['checkpoint_sha256'],
    scope='One real layer0 dtype acceptance and full FP64 reference; not performance or model qualification',
    model_or_service_qualified=False,adopted=False)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('sources','libraries','files','bank','references')},indent=2))
