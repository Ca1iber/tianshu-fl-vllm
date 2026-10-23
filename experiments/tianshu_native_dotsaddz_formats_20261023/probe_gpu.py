# SPDX-License-Identifier: Apache-2.0
import argparse,traceback
from common import *
p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,required=True);args=p.parse_args();gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());v=next(v for v in gate['variants'] if v['name']==args.variant);assert sha(v['file'])==v['sha256'];root=GPU/args.variant;root.mkdir(parents=True,exist_ok=False);state=dict(complete=False,phase='starting',variant=args.variant,CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),input_sha256=v['sha256'],source_sha256=sha(__file__),performance_not_tested=True,real_model_weights_not_tested=True)
def save():(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 import torch
 data=np.load(v['file']);guards=[]
 def tensor(key):
  arr=data[key].copy();t=torch.from_numpy(arr);t=t.view(torch.bfloat16) if arr.dtype==np.uint16 else t;margin=262144;raw=torch.zeros(arr.nbytes+2*margin,device='cuda',dtype=torch.uint8);view=raw[margin:margin+arr.nbytes].view(t.dtype).view(t.shape);view.copy_(t);guards.append((raw,raw.cpu().numpy().tobytes(),raw._version));return view
 x,w,s,z=[tensor(k) for k in ('X','W','S','Z')];state['phase']='executing';save();out=torch.ops.aten._weight_int4pack_mm_dotsaddz.default(x,w,128,s,z,size_n=256);torch.cuda.synchronize();assert out.shape==(64,256) and out.dtype==torch.bfloat16;assert all(raw._version==version and raw.cpu().numpy().tobytes()==before for raw,before,version in guards);path=root/'output.bf16';path.write_bytes(out.contiguous().view(torch.uint16).cpu().numpy().tobytes());state.update(complete=True,phase='completed_numeric_pending',output=str(path),output_sha256=sha(path),all_input_guards_versions_and_full_bytes_preserved=True)
except Exception as error:state.update(complete=True,phase='operation_failed',error=repr(error),traceback=traceback.format_exc())
save();print(json.dumps(state))
