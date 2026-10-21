# SPDX-License-Identifier: Apache-2.0
"""Read actual cached kernels and evaluate real SME argument-shape classifier on meta tensors."""
import ast,base64,hashlib,json
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
helper=Path('/usr/local/lib/python3.12/site-packages/triton/spec/iluvatar/triton/runtime/jit.py')
tree=ast.parse(helper.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_corex_sme');namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(helper),'exec'),namespace)
component=BASE.parent/'tianshu_m_range_mlp42_tiles_v2_results_20261021/GPU_GATE_32g.json';gate=json.loads(component.read_text());cache=Path('/tmp/torchinductor_root/triton/0')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def folder(h):return cache/base64.b32encode(bytes.fromhex(h)).decode().rstrip('=')
old=getattr(torch,'corex',None);setattr(torch,'corex',True)
try:
 rows=[]
 for m,item in gate['actual_timed_candidate_kernels'].items():
  m=int(m);metadata=folder(item['cache_hash'])/'_fused.json';record=json.loads(metadata.read_text());assert record['use_sme']==83 and digest(metadata.with_suffix('.cubin'))==item['binary_sha256']
  q=torch.empty((m,2048),dtype=torch.int8,device='meta');w=torch.empty((2048,12288),dtype=torch.int8,device='meta');s=torch.empty(m,dtype=torch.float32,device='meta');ws=torch.empty(12288,dtype=torch.float32,device='meta');out=torch.empty((m,6144),dtype=torch.bfloat16,device='meta');dummy=torch.empty(1,dtype=torch.int32,device='meta')
  baseline=[q,w,s,ws,out,torch.empty((m,12288),dtype=torch.int32,device='meta'),torch.empty((m,12288),dtype=torch.bfloat16,device='meta')]
  previous=[q,w,s,ws,out,dummy,dummy.view(torch.bfloat16)];fixed=[q,w,s,ws,out,dummy,out]
  classify=lambda args:namespace['get_corex_sme'](args,[('tensor',None)]*len(args))
  assert classify(baseline)==classify(fixed)==83 and classify(previous)==19
  rows.append(dict(M=m,component_mask=83,old_tiny_placeholder_mask=19,fixed_existing_output_alias_mask=83,component_binary_sha256=item['binary_sha256']))
finally:
 if old is None:delattr(torch,'corex')
 else:setattr(torch,'corex',old)
actual=cache/'JFJ34KBSUKP5NWSC53BIRGTYN5EOIPXGDJBWMAWPSJ5FDIY3TF5A/_fused.json';actual_meta=json.loads(actual.read_text());assert actual_meta['use_sme']==19
expected=folder(gate['actual_timed_candidate_kernels']['128']['cache_hash'])/'_fused.json';expected_meta=json.loads(expected.read_text())
option_diff={k:[expected_meta.get(k),actual_meta.get(k)] for k in set(expected_meta)|set(actual_meta) if expected_meta.get(k)!=actual_meta.get(k)};assert set(option_diff)=={'hash','use_sme'}
assert not torch.cuda.is_initialized()
result=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,actual_backend_classifier_source=str(helper),actual_backend_classifier_sha256=digest(helper),classifier_backend_identity_assumed_True_consistent_with_real_GPU_metadata=True,component_gate_sha256=digest(component),actual_failed_service_metadata_sha256=digest(actual),actual_failed_service_binary_sha256=digest(actual.with_suffix('.cubin')),only_compiler_option_difference='use_sme:83→19',full_metadata_difference=option_diff,rows=rows,source_sha256=digest(__file__),fix='For diagnostics=False pass already allocated 2D BF16 output as unused Projected operand; Integer remains tiny int32 dummy. No new storage and no global compiler switches.',model_performance_not_measured=True)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('rows','full_metadata_difference')}))
