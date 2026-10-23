# SPDX-License-Identifier: Apache-2.0
import argparse,json,traceback
from inputs import *

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=VARIANTS,required=True);a=parser.parse_args();cpu=json.loads((BASE/'CPU_GATE_32g.json').read_text());proof=next(v for v in cpu['variants'] if v['name']==a.variant);assert sha(proof['file'])==proof['sha256'];root=GPU/a.variant;root.mkdir(parents=True,exist_ok=False);r=dict(complete=False,phase='starting',variant=a.variant,CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),input_sha256=sha(proof['file']),source_sha256=sha(__file__),functional_test_only=True,performance_not_tested=True,real_model_weights_not_tested=True)
 def save():(root/'result.json').write_text(json.dumps(r,indent=2)+'\n')
 save()
 try:
  import torch
  data=np.load(proof['file']);guards=[]
  def tensor(key):
   arr=data[key].copy();t=torch.from_numpy(arr)
   if arr.dtype==np.uint16:t=t.view(torch.bfloat16)
   # Extra initialized storage keeps alternate zero/scale format reads within allocated memory.
   margin=262144;raw=torch.zeros(arr.nbytes+2*margin,device='cuda',dtype=torch.uint8);view=raw[margin:margin+arr.nbytes].view(t.dtype).view(t.shape);view.copy_(t);before=raw.cpu().numpy().tobytes();guards.append((raw,before,int(raw._version)));return view
  x=tensor('X');w=tensor('W');r['phase']='executing';save()
  if a.variant.startswith('generic_'):
   p=tensor('P');packed=torch.ops.aten._convert_weight_to_int4pack.default(w,8);out=torch.ops.aten._weight_int4pack_mm.default(x,packed,128,p)
  else:s=tensor('S');z=tensor('Z');out=torch.ops.aten._weight_int4pack_mm_awq.default(x,w,128,s,z)
  torch.cuda.synchronize();assert tuple(out.shape)==(64,256) and out.dtype==torch.bfloat16;assert all(int(raw._version)==version and raw.cpu().numpy().tobytes()==before for raw,before,version in guards);path=root/'output.bf16';path.write_bytes(out.contiguous().view(torch.uint16).cpu().numpy().tobytes());r.update(complete=True,phase='operations_completed_numeric_pending',output=str(path),output_sha256=sha(path),all_input_storage_versions_bytes_and_guards_preserved=True)
 except Exception as error:r.update(complete=True,phase='operation_failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
 save();print(json.dumps(r))
if __name__=='__main__':main()
