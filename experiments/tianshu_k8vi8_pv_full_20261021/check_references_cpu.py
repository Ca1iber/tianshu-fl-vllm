# SPDX-License-Identifier: Apache-2.0
"""Pure CPU subprocess; no vLLM/platform import or device initialization."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from plans import MODES,CASES
from reference_numpy import Oracle
import reference_torch
from verification import compare
BASE=Path(__file__).resolve().parent
torch.set_num_threads(4);assert not torch.cuda.is_initialized()
rng=torch.get_rng_state().clone();tests=0;writer_values=0
for index,length in enumerate((1,17,33,129)):
    generator=torch.Generator().manual_seed(202610211250+index)
    q=torch.randn(2,16,128,generator=generator,dtype=torch.bfloat16)
    k=torch.randn(2,length,2,128,generator=generator,dtype=torch.bfloat16)
    v=torch.randn(2,length,2,128,generator=generator,dtype=torch.bfloat16);v[0,:,0].zero_()
    data=dict(batch=2,q=q,k=k,v=v,lengths=[length,max(1,length-3)])
    oracle=Oracle(data);keys=torch.from_numpy(oracle.k)
    vq=torch.from_numpy(oracle.vq);vs=torch.from_numpy(oracle.vs)
    score=reference_torch.scores(q,keys,data['lengths'])
    assert bool((vq[0,:,0]==0).all());writer_values+=vq.numel()
    for tile in (32,64):
      for segments in (4,16,64):
        actual=reference_torch.integer_pv(score,vq,vs,data['lengths'],tile,segments)
        for sequence in range(2):
          for head in (0,7,8,15):
            reference,bound=oracle.integer_pv(sequence,head,tile,segments)
            check=compare(actual[sequence,head],torch.from_numpy(reference),True);assert check['passed'],(length,tile,segments,sequence,head,check)
            ideal=oracle.ideal(sequence,head)
            assert bool(np.all(np.abs(actual[sequence,head].numpy()-ideal)<=bound+.004*np.abs(reference)+5.e-4))
            tests+=1
assert torch.equal(rng,torch.get_rng_state()) and len(MODES)==7 and len(CASES)==10
assert not torch.cuda.is_initialized()
result=dict(complete=True,pure_CPU=True,reference_vectors=tests,writer_fixture_elements=writer_values,default_RNG_unchanged=True,CUDA_not_initialized_before_or_after=True,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(BASE/'CPU_REFERENCE_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
