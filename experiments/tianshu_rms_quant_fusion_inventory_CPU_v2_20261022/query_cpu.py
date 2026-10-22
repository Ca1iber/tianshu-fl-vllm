# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=Path('/usr/local/lib/python3.12/site-packages/vllm')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
 assert not (BASE/'CPU_INVENTORY_32g.json').exists();paths=[BASE/'query_cpu.py',ROOT/'_custom_ops.py',ROOT/'model_executor/layers/layernorm.py',ROOT/'model_executor/models/llama.py',ROOT/'compilation/passes/fusion/rms_quant_fusion.py',ROOT/'compilation/passes/pass_manager.py'];sources={str(p):sha(p) for p in paths};tree=ast.parse((ROOT/'compilation/passes/fusion/rms_quant_fusion.py').read_text());keys=sorted({n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id.startswith(('kFp8','kNvfp4','kInt8','kI8'))});wrappers={}
 for n in ast.parse((ROOT/'_custom_ops.py').read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ('scaled_int8_quant','rms_norm_dynamic_per_token_quant'):
   calls=[]
   for c in ast.walk(n):
    if isinstance(c,ast.Call):
     text=ast.unparse(c.func)
     if text.startswith('torch.ops.'):calls.append(text)
   wrappers[n.name]=dict(line=n.lineno,torch_op_calls=calls)
 conf=Path('/workspace/MiniCPM5-2B/config.json');model=json.loads(conf.read_text());providers=[str(p) for p in ROOT.glob('_C*.so')];assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
 result=dict(complete=True,CPU_only=True,stdlib_only=True,no_GPU_libraries_imported=True,CUDA_and_schema_queries_NOT_executed=True,actual_architectures=model['architectures'],model_type=model['model_type'],model_config_sha256=sha(conf),source_hashes=sources,standard_fusion_key_names=keys,standard_fusion_INT8_key_found=any('int8' in k.lower() or k.startswith('kI8') for k in keys),standard_C_shared_objects_found=providers,missing_standard_C_extension_does_not_prove_no_plugin_provider=True,wrappers=wrappers,previous_import_attempt_rejected_by_CPU_only_guard=True,native_runtime_and_INT8_dtype_NOT_tested=True,model_or_framework_NOT_modified=True,performance_measured=False,adopted=False);(BASE/'CPU_INVENTORY_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))
if __name__=='__main__':main()
