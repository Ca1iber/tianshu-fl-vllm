# SPDX-License-Identifier: Apache-2.0
"""Actual generated uint16 -> four BF16 values helper, all 65536 input words."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl
import fixtures_cpu as fx

@triton.jit
def _decode_words(P,O,N:tl.constexpr,STRIDE:tl.constexpr,Decoder:tl.constexpr):
    i=tl.program_id(0)*256+tl.arange(0,256)
    p=tl.load(P+i*STRIDE,i<N,other=-30584).to(tl.uint16)
    values=Decoder(p)
    output=4*i[:,None,None]+tl.arange(0,4).reshape(1,2,2)
    tl.store(O+output,values,(i<N)[:,None,None])

def verify(reader,root):
    root=Path(root);root.mkdir(exist_ok=False);cases=[]
    base=np.arange(65536,dtype=np.uint16);n=base.size
    for stride in (1,2):
        storage=torch.full((n*stride+32,),17,dtype=torch.int16,device='cuda');p=storage[16:16+n*stride:stride]
        out_storage=torch.full((n*4+64,),17.,dtype=torch.bfloat16,device='cuda');out=out_storage[:n*4]
        graph=None
        for phase in (0,1):
            words=base.copy() if not phase else base[::-1].copy()^np.uint16(42405)
            p.copy_(torch.from_numpy(words.view(np.int16)).to('cuda'));out_storage.fill_(17.)
            before=storage.clone()
            q=np.stack(tuple(((words>>(4*d))&15).astype(np.int16)-8 for d in range(4)),axis=-1)
            expected=fx.bits(q.astype(np.float32)).reshape(-1)
            if graph is None:
                compiled=_decode_words[(triton.cdiv(n,256),)](p,out,n,stride,reader._k4_decode_word,num_warps=1,num_stages=1)
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):_decode_words[(triton.cdiv(n,256),)](p,out,n,stride,reader._k4_decode_word,num_warps=1,num_stages=1)
            rng=torch.cuda.get_rng_state().clone();graph.replay();torch.cuda.synchronize()
            observed=out.cpu().view(torch.int16).numpy().view(np.uint16)
            assert np.array_equal(observed,expected),(stride,phase,np.flatnonzero(observed!=expected)[:8].tolist())
            assert torch.equal(storage,before) and bool((out_storage[n*4:]==17).all()) and torch.equal(torch.cuda.get_rng_state(),rng)
            binary=root/f'{stride}_{hashlib.sha256(compiled.kernel).hexdigest()}.bin';binary.write_bytes(compiled.kernel)
            path=root/f'decode_word_stride{stride}_phase{phase}.npz'
            np.savez_compressed(path,packed_words=words,observed_BF16_bits=observed,expected_BF16_bits=expected)
            cases.append(dict(stride=stride,phase=phase,all65536_packed_words=True,reserved_code0_q_minus8=True,
                output_elements=n*4,bitwise_equal=True,input_and_output_guards=True,RNG_unchanged=True,
                same_Graph_changed_input=bool(phase),n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),
                binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),artifact=str(path),artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result=dict(complete=True,cases=cases,total_output_elements=sum(x['output_elements'] for x in cases),
        generated_source_sha256=hashlib.sha256(Path(reader.__file__).read_bytes()).hexdigest())
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
