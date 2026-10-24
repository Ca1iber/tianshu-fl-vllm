# SPDX-License-Identifier: Apache-2.0
"""Numerical and layout qualification only; no GPU/device imports or model claims."""
import hashlib,json,math,struct
from pathlib import Path
import numpy as np
from codec import MIN_AMPLITUDE,PAGE_BYTES,V_OFFSET,SCALE_OFFSET,encode,pack,unpack,reconstruct
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(x):return struct.unpack('<f',struct.pack('<f',float(x)))[0]
def reference(row):
    maximum=max(abs(float(v)) for v in row)
    scale=1.0 if maximum<float(MIN_AMPLITUDE) else f32(maximum/7.0)
    inverse=f32(1.0/scale)
    q=[max(-7,min(7,int(round(f32(float(v)*inverse))))) for v in row]
    result=bytes((q[d]+8)|((q[d+1]+8)<<4) for d in range(0,128,2))
    return result,struct.pack('<f',scale)

def main():
    root=Path('/dev/shm/tianshu_K4V16_contract_CPU_20261024')
    assert not root.exists();root.mkdir()
    alphabet=np.arange(65536,dtype=np.uint32)
    finite=alphabet[((alphabet>>7)&255)!=255]
    assert len(finite)==65280 and math.gcd(4051,len(finite))==1
    orders=[finite,finite[(np.arange(len(finite))*4051+17)%len(finite)]]
    checks=0;stats=[];arrays={}
    for index,bits in enumerate(orders):
        x=(bits.astype(np.uint32)<<16).view(np.float32).reshape(-1,128)
        p,s=encode(x);restored=reconstruct(p,s)
        assert p.shape==(510,64) and s.shape==(510,) and np.isfinite(restored).all()
        for i,row in enumerate(x):
            expected,scale=reference(row)
            assert p[i].tobytes()==expected and s[i].tobytes()==scale,(index,i)
            checks+=1
        assert np.array_equal(pack(unpack(p)),p)
        x64=x.astype(np.float64)
        norms=np.sum(x64*x64,axis=1)
        eligible=(np.max(np.abs(x),axis=1)>=MIN_AMPLITUDE)&(norms>0)
        rel=np.sqrt(np.sum((restored.astype(np.float64)-x64)**2,axis=1)[eligible]/norms[eligible])
        stats.append(dict(order=index,eligible_rows=int(eligible.sum()),mean_relative_L2=float(rel.mean()),max_relative_L2=float(rel.max())))
        arrays[f'input_bits_{index}']=bits.astype(np.uint16);arrays[f'packed_{index}']=p;arrays[f'scale_{index}']=s
    values=np.arange(256,dtype=np.uint8)
    full=np.repeat(values[:,None],64,axis=1)
    assert np.array_equal(pack(unpack(full)),full)
    ties=np.zeros((1,128),dtype=np.float32)
    cases=np.array([-7,-6.5,-5.5,-4.5,-3.5,-2.5,-1.5,-.5,-0.0,0.0,.5,1.5,2.5,3.5,4.5,5.5,6.5,7],dtype=np.float32)
    ties[0,:len(cases)]=cases
    p,s=encode(ties);expected,scale=reference(ties[0])
    assert p[0].tobytes()==expected and s[0].tobytes()==scale and s[0]==1
    assert np.array_equal(unpack(p)[0,:len(cases)],np.rint(cases).astype(np.int8))
    zeros=np.zeros((3,128),dtype=np.float32);zeros[1]=-0.0;zeros[2]=np.float32(2.0**-110)
    p,s=encode(zeros);assert np.all(p==136) and np.all(s==1) and np.all(reconstruct(p,s)==0)
    k=set();sc=set();v=set()
    for slot in range(16):
        for head in range(2):
            row=slot*2+head
            k.update(range(row*64,(row+1)*64))
            sc.update(range(SCALE_OFFSET+row*4,SCALE_OFFSET+(row+1)*4))
            v.update(range(V_OFFSET+row*256,V_OFFSET+(row+1)*256))
    assert len(k)==2048 and len(sc)==128 and len(v)==8192
    assert not(k&sc or k&v or sc&v) and max(v)+1==PAGE_BYTES
    assert len(set(range(PAGE_BYTES))-(k|sc|v))==128 and PAGE_BYTES%256==0
    np.savez_compressed(root/'finite_fixture.npz',**arrays)
    source_paths=[BASE/'PLAN_32g.md',BASE/'codec.py',BASE/'check_cpu.py']
    result=dict(complete=True,CPU_only=True,finite_BF16_patterns_per_order=65280,orders=2,independent_scalar_rows=checks,
        all_scalar_FP32_rounding_and_packed_bits_match=True,all256_packed_bytes_roundtrip=True,
        all_half_integer_RNE_boundaries_passed=True,zero_sign_and_tiny_contract_passed=True,
        proposed_layout_regions_nonoverlapping=True,proposed_page_bytes=PAGE_BYTES,
        theoretical_page_reduction_percent=100*(1-PAGE_BYTES/12288),key_quantization_error_statistics=stats,
        sources={str(p):sha(p) for p in source_paths},raw_fixture=str(root/'finite_fixture.npz'),raw_fixture_sha256=sha(root/'finite_fixture.npz'),
        native_Spec_allocator_not_verified=True,GPU_not_run=True,model_not_ready=True,Math_not_run=True,performance_not_measured=True,adopted=False)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','raw_fixture_sha256')},indent=2))
if __name__=='__main__':main()
