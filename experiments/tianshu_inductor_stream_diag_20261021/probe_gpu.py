# SPDX-License-Identifier: Apache-2.0
"""Isolate cached-stream versus capture-current-stream with poisoned outputs."""
import hashlib,importlib.util,json,signal,subprocess,sys,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261021_inductor_stream_diag_GPU_tianshu')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 signal.alarm(180);hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
 gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and all(sha(f)==d for f,d in gate['sources'].items())
 ROOT.mkdir(exist_ok=False);(ROOT/'hardware_before.txt').write_text(hardware);r=dict(complete=False,cases=[],launches=[],sources=gate['sources'])
 def save():(ROOT/'result.json').write_text(json.dumps(r,indent=2)+'\n')
 save()
 try:
  import torch
  torch.set_num_threads(4);path=BASE.parent/'tianshu_m_range_mlp42_20261021/inductor_activation.py';sp=importlib.util.spec_from_file_location('_stream_diag_act',path);module=importlib.util.module_from_spec(sp);sys.modules[sp.name]=module;sp.loader.exec_module(module)
  x_cpu=torch.randn(8,12288,dtype=torch.bfloat16,generator=torch.Generator().manual_seed(202610215501));x=x_cpu.cuda();fixed_out=torch.empty(8,6144,device='cuda',dtype=torch.bfloat16);dynamic_out=torch.empty_like(fixed_out);eager_out=torch.empty_like(fixed_out);fixed=int(torch.cuda.current_stream().cuda_stream)
  def launch(out,handle,kind):
   r['launches'].append(dict(kind=kind,explicit_handle=int(handle),current_raw_handle=int(torch._C._cuda_getCurrentRawStream(0)),capturing=torch.cuda.is_current_stream_capturing()))
   module.triton_poi_fused_mul_silu_slice_1.run(x,out,8*6144,stream=handle)
  def bad():launch(fixed_out,fixed,'fixed')
  def good():launch(dynamic_out,torch._C._cuda_getCurrentRawStream(0),'dynamic')
  for _ in range(2):bad();good()
  torch.cuda.synchronize();gb,gg=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
  with torch.cuda.graph(gb):bad()
  with torch.cuda.graph(gg):good()
  for phase,factor in enumerate((1.,-.5,.25)):
   actual=x_cpu*factor;x.copy_(actual);fixed_out.fill_(17.);dynamic_out.fill_(17.)
   gb.replay();gg.replay();torch.cuda.synchronize();fb,dg=fixed_out.cpu().clone(),dynamic_out.cpu().clone()
   launch(eager_out,torch._C._cuda_getCurrentRawStream(0),'eager');torch.cuda.synchronize();eg=eager_out.cpu()
   double=actual.double();g,u=double[:,:6144],double[:,6144:];expected=(g/(1+torch.exp(-g))*u).to(torch.bfloat16)
   dynamic_bitwise=bool(torch.equal(dg.view(torch.int16),eg.view(torch.int16)));fixed_bitwise=bool(torch.equal(fb.view(torch.int16),eg.view(torch.int16)))
   cpu=bool(torch.isfinite(dg).all() and ((dg.float()-expected.float()).abs()<=.001+.02*expected.float().abs()).all())
   assert dynamic_bitwise and cpu
   torch.save(dict(input=actual,fixed_graph_output=fb,dynamic_graph_output=dg,eager_output=eg,CPU_reference=expected),ROOT/f'phase{phase}.pt')
   r['cases'].append(dict(phase=phase,factor=factor,poison=17.,dynamic_graph_eager_full_BF16_bitwise=dynamic_bitwise,fixed_graph_eager_full_BF16_bitwise=fixed_bitwise,dynamic_CPU_full_FP64_tolerance=cpu,fixed_unchanged_poison=bool((fb==17).all())))
  assert all(sha(f)==d for f,d in gate['sources'].items());r.update(complete=True,phase='complete',cached_stream_captured_different_from_actual=any(z['kind']=='fixed' and z['capturing'] and z['explicit_handle']!=z['current_raw_handle'] for z in r['launches']),fixed_failed_cases=sum(not c['fixed_graph_eager_full_BF16_bitwise'] for c in r['cases']));save();print(json.dumps(r,indent=2))
 except BaseException as e:r.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
