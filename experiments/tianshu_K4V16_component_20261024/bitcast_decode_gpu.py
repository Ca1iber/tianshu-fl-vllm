# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual generated helper for every packed byte, also reserved code0."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl
import fixtures_cpu as fx

@triton.jit
def _decode(P,O,N:tl.constexpr,STRIDE:tl.constexpr,Decoder:tl.constexpr):
    i=tl.program_id(0)*256+tl.arange(0,256)
    p=tl.load(P+i*STRIDE,i<N,other=136)
    low=Decoder((p&15).to(tl.int16)-8)
    high=Decoder((p>>4).to(tl.int16)-8)
    tl.store(O+2*i,low,i<N);tl.store(O+2*i+1,high,i<N)

def verify(reader,root):
    root=Path(root);root.mkdir(exist_ok=False);cases=[]
    base=np.tile(np.arange(256,dtype=np.uint8),17);n=base.size
    for stride in (1,2):
        storage=torch.full((n*stride+32,),17,dtype=torch.uint8,device='cuda');p=storage[16:16+n*stride:stride]
        out_storage=torch.full((n*2+64,),17.,dtype=torch.bfloat16,device='cuda');out=out_storage[:n*2]
        graph=None
        for phase in (0,1):
            packed=base.copy() if not phase else base[::-1].copy()^np.uint8(165)
            p.copy_(torch.from_numpy(packed).to('cuda'));out_storage.fill_(17.)
            before=storage.clone()
            q=np.empty((n,2),np.int16);q[:,0]=(packed&15).astype(np.int16)-8;q[:,1]=(packed>>4).astype(np.int16)-8
            expected=fx.bits(q.astype(np.float32)).reshape(-1)
            if graph is None:
                compiled=_decode[(triton.cdiv(n,256),)](p,out,n,stride,reader._k4_int_to_bf16,num_warps=1,num_stages=1)
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):_decode[(triton.cdiv(n,256),)](p,out,n,stride,reader._k4_int_to_bf16,num_warps=1,num_stages=1)
            rng=torch.cuda.get_rng_state().clone();graph.replay();torch.cuda.synchronize()
            observed=out.cpu().view(torch.int16).numpy().view(np.uint16)
            assert np.array_equal(observed,expected),(stride,phase,np.flatnonzero(observed!=expected)[:8].tolist())
            assert torch.equal(storage,before) and bool((out_storage[n*2:]==17).all())
            assert torch.equal(torch.cuda.get_rng_state(),rng)
            binary=root/f'{stride}_{hashlib.sha256(compiled.kernel).hexdigest()}.bin';binary.write_bytes(compiled.kernel)
            path=root/f'decode_stride{stride}_phase{phase}.npz'
            np.savez_compressed(path,packed=packed,observed_BF16_bits=observed,expected_BF16_bits=expected)
            cases.append(dict(stride=stride,phase=phase,all256_packed_bytes=True,reserved_code0_q_minus8=True,
                output_elements=n*2,bitwise_equal=True,input_and_output_guards=True,RNG_unchanged=True,
                same_Graph_changed_input=bool(phase),n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),
                binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),artifact=str(path),artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result=dict(complete=True,cases=cases,total_output_elements=sum(x['output_elements'] for x in cases),
        generated_source_sha256=hashlib.sha256(Path(reader.__file__).read_bytes()).hexdigest())
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
