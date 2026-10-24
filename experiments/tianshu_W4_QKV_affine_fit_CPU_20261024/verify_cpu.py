# SPDX-License-Identifier: Apache-2.0
"""Recompute every fitted group's actual BF16 reconstruction error from payload."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
ROOT=Path('/dev/shm/tianshu_W4_QKV_affine_fit_CPU_20261024')
spec=importlib.util.spec_from_file_location('_fit_verify_original_reader',BASE.parent/'tianshu_W4_attention64_20261024/common.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
r=json.loads((ROOT/'result.json').read_text());assert r['complete'] and r['CPU_only']
for p,h in {**r['sources'],**r['files']}.items(): assert c.sha(p)==h

def words_to_float(a): return (np.asarray(a,np.uint16).astype(np.uint32)<<16).view(np.float32)
def round_words(a):
    v=np.asarray(a,np.float32).view(np.uint32)
    return ((v+np.uint32(0x7fff)+((v>>16)&1))>>16).astype(np.uint16)
def error(w,q,s,z):
    dequant=words_to_float(round_words(q.astype(np.float32)*np.repeat(words_to_float(s),128,axis=0)
        +np.repeat(words_to_float(z),128,axis=0))).T.reshape(2560,16,128)
    return np.sum((dequant.astype(np.float64)-words_to_float(w).reshape(2560,16,128))**2,axis=2)
checks=[]
for b in r['banks']:
    layer=b['layer'];w=c.original(layer,'QKV');assert c.ahash(w)==b['original_sha256']
    data=np.load(ROOT/f'QKV_L{layer}_fitted.npz',allow_pickle=False)
    packed=data['w'];s=data['s'];z=data['z']
    assert packed.shape==(2048,320) and s.shape==z.shape==(16,2560)
    assert c.ahash(packed)==b['packed_sha256'] and c.ahash(s)==b['scale_sha256'] and c.ahash(z)==b['zero_sha256']
    q=(((packed.view(np.uint32)[...,None]>>(4*np.arange(8,dtype=np.uint32)))&15).reshape(2048,2560)).astype(np.uint8)
    iq,isc,iz,_=c.quant(w,'affine')
    actual=error(w,q,s,z);initial=error(w,iq,isc,iz)
    assert np.all(actual<=initial) and np.isfinite(actual).all()
    assert abs(actual.sum()-b['fitted_MSE'])<1e-10 and abs(initial.sum()-b['initial_MSE'])<1e-10
    checks.append(dict(layer=layer,groups=int(actual.size),all_groups_not_worse=True,
        initial_MSE=float(initial.sum()),fitted_MSE=float(actual.sum())))
assert c.sha(c.MODEL)==r['checkpoint_sha256']
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
result=dict(complete=True,CPU_only=True,full_weight_elements_verified=42*2560*2048,
    full_groups_verified=sum(b['groups'] for b in checks),all42_byte_scope_verified=True,
    all_groups_actual_BF16_MSE_nonincreasing=True,checks=checks,
    raw_result_sha256=c.sha(ROOT/'result.json'),verifier_source_sha256=c.sha(__file__),
    GPU_qualified=False,model_quality_qualified=False,adopted=False)
(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
