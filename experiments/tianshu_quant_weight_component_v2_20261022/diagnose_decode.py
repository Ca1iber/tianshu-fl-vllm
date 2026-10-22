import hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;SURVEY=BASE.parent/'tianshu_quantization_survey_20261022'
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
import torch,vllm_fl
fmt=load('_diag_fmt',SURVEY/'formats_cpu.py');kernel=load('_diag_kernel',SURVEY/'kernel_weight_only.py');w=fmt.bf16_round(np.sin(np.arange(131*259,dtype=np.float32)*.19).reshape(131,259)*.1);w[0]=0;records=[]
for bits,group,aff in [(4,128,False),(4,128,True),(8,0,False)]:
 q,s,z,d,_=fmt.quantize(w,bits,group,aff)
 for phase in range(3):
  sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.;qp=q if phase==0 else (15-q if aff else -q);zp=z if phase==0 or not aff else 15-z
  packed=fmt.pack_nibbles_nn(qp,aff) if bits==4 else qp.T.copy().view(np.uint8)
  tensors=[torch.from_numpy(np.ascontiguousarray(a)).cuda() for a in (packed,(s*factor).T,zp.astype(np.uint8).T)];out=torch.empty(w.shape,dtype=torch.bfloat16,device='cuda');kernel.decode(*tensors,out,bits,group,aff);actual=out.view(torch.uint16).cpu().numpy();expected=(fmt.bf16_round(d*sign*factor).view(np.uint32)>>16).astype(np.uint16);mismatch=actual!=expected
  count=int(mismatch.sum());zero_only=bool(np.all((actual[mismatch]&0x7fff)==0) and np.all((expected[mismatch]&0x7fff)==0));assert zero_only
  canonical=expected.copy();canonical[(canonical&0x7fff)==0]=0;assert np.array_equal(actual,canonical)
  records.append(dict(bits=bits,affine=aff,phase=phase,bit_mismatches=count,all_mismatches_signed_zero=zero_only,canonical_zero_bitwise_pass=True))
result=dict(complete=True,records=records,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),SURVEY/'formats_cpu.py',SURVEY/'kernel_weight_only.py']},reason='Negating decoded CPU float zero produces -0; negating integer zero then decoding has +0. Expected reference must reconstruct integer format, not negate floating signed zeros.')
(BASE/'SIGNED_ZERO_DIAGNOSIS_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
