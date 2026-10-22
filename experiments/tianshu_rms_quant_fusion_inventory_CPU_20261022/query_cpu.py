# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
 import torch,vllm._custom_ops,vllm.ir.ops
 assert not torch.cuda.is_initialized();root=Path(vllm.__file__).resolve().parent;paths=[BASE/'query_cpu.py',root/'_custom_ops.py',root/'model_executor/layers/layernorm.py',root/'model_executor/models/llama.py',root/'compilation/passes/fusion/rms_quant_fusion.py',root/'compilation/passes/pass_manager.py'];sources={str(p):sha(p) for p in paths};ops={}
 for name in ('_C::rms_norm','_C::fused_add_rms_norm','_C::dynamic_scaled_int8_quant','_C::rms_norm_dynamic_per_token_quant','vllm_ir::rms_norm','vllm_ir::fused_add_rms_norm'):
  try:
   handle=torch._C._dispatch_find_schema_or_throw(name,'');ops[name]=dict(schema=str(handle.schema()),CUDA_registered=torch._C._dispatch_has_kernel_for_dispatch_key(name,'CUDA'),Meta_registered=torch._C._dispatch_has_kernel_for_dispatch_key(name,'Meta'))
  except RuntimeError as e:ops[name]=dict(query_error=str(e))
 fusion=(root/'compilation/passes/fusion/rms_quant_fusion.py').read_text();tree=ast.parse(fusion);key_names=sorted({n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id.startswith(('kFp8','kNvfp4','kInt8','kI8'))});providers={}
 for p in root.glob('_C*.so'):providers[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p))
 assert not torch.cuda.is_initialized();model=json.loads(Path('/workspace/MiniCPM5-2B/config.json').read_text());out=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,actual_architectures=model['architectures'],model_type=model['model_type'],source_hashes=sources,native_providers=providers,operators=ops,standard_fusion_key_names=key_names,standard_fusion_int8_key_found=any('int8' in n.lower() or n.startswith('kI8') for n in key_names),native_runtime_or_INT8_output_dtype_NOT_tested=True,framework_or_model_NOT_modified=True,model_executed=False,performance_measured=False,adopted=False);(BASE/'CPU_INVENTORY_32g.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('source_hashes','native_providers')}))
if __name__=='__main__':main()
