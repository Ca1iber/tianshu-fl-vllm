# SPDX-License-Identifier: Apache-2.0
import hashlib,json,subprocess,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261023_native_INT4_smoke_GPU_tianshu')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not ROOT.exists();ROOT.mkdir();cpu=json.loads((BASE/'CPU_PREPARE_32g.json').read_text());assert cpu['complete']
 for p,h in cpu['sources'].items():assert sha(p)==h
 hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(ROOT/'hardware_before.txt').write_text(hardware)
 result=dict(complete=False,phase='starting',rows=[],CPU_prepare_sha256=sha(BASE/'CPU_PREPARE_32g.json'),source_sha256=sha(__file__),functional_smoke_only=True,performance_not_tested=True,real_model_weights_not_tested=True,checkpoint_not_opened=True)
 def save():(ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 save()
 try:
  import torch
  assert torch.__version__==cpu['torch_version'];x=torch.full((64,256),.125,dtype=torch.bfloat16,device='cuda');params=torch.zeros((2,256,2),device='cuda',dtype=torch.bfloat16);params[:,:,0]=.125
  for nibble in (7,8,9,10):
   result['phase']=f'pack_q{nibble}';save();w=torch.full((256,128),nibble*17,device='cuda',dtype=torch.uint8);packed=torch.ops.aten._convert_weight_to_int4pack.default(w,8);torch.cuda.synchronize();assert list(packed.shape)==[32,2,32,4] and packed.dtype==torch.int32
   result['phase']=f'mm_q{nibble}';save();out=torch.ops.aten._weight_int4pack_mm.default(x,packed,128,params);torch.cuda.synchronize();assert list(out.shape)==[64,256] and out.dtype==torch.bfloat16;raw=out.contiguous().view(torch.uint16).cpu().numpy().tobytes();path=ROOT/f'q{nibble}.bf16';path.write_bytes(raw);result['rows'].append(dict(nibble=nibble,packed_shape=list(packed.shape),output_shape=list(out.shape),raw=str(path),raw_sha256=sha(path),raw_bytes=len(raw),expected_scalar=4*(nibble-8)));save()
  result.update(complete=True,phase='operations_completed_numeric_pending',CUDA_operations_completed=True)
 except Exception as error:result.update(complete=True,phase='operation_failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),CUDA_operations_completed=False)
 save();print(json.dumps(result))
if __name__=='__main__':main()
