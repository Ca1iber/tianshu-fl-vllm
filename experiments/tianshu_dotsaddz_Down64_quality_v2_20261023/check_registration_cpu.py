# SPDX-License-Identifier: Apache-2.0
"""Public kernel replacement in an existing CPU FX graph, no CUDA/Dynamo proof."""
from pathlib import Path
import hashlib,inspect,json,os
import torch
from torch._library.custom_ops import OPDEFS
BASE=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
banks={0:'CPU-only closure sentinel'}
def install():
 actual_banks=banks
 @torch.library.custom_op('tianshu_dots_Down64_CPU_API::linear',mutates_args=(),device_types='cpu')
 def linear(x:torch.Tensor)->torch.Tensor:
  assert actual_banks[0]=='CPU-only closure sentinel';return x*2
 @linear.register_fake
 def fake(x):return torch.empty_like(x)
 return linear
linear=install();definition=OPDEFS['tianshu_dots_Down64_CPU_API::linear'];original_fn=definition._init_fn;original_backend=definition._backend_fns['cpu'];schema=str(definition._opoverload._schema);cells=dict(zip(original_fn.__code__.co_freevars,original_fn.__closure__));assert cells['actual_banks'].cell_contents is banks
def caller(x):return torch.ops.tianshu_dots_Down64_CPU_API.linear.default(x)
compiled=torch.fx.symbolic_trace(caller);graph_before=str(compiled.graph);x=torch.full((2,3),2.);assert torch.equal(compiled(x),torch.full_like(x,4.))
def routed(x):return x*3 if x.shape[0]==64 else original_backend(x)
definition.register_kernel('cpu',routed)
for m in (2,64,5,64,2):
 x=torch.full((m,3),2.);assert torch.equal(compiled(x),torch.full_like(x,6. if m==64 else 4.))
assert definition._init_fn is original_fn and str(definition._opoverload._schema)==schema and str(compiled.graph)==graph_before and 'tianshu_dots_Down64_CPU_API.linear' in graph_before and not torch.cuda.is_initialized()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
registry=Path(inspect.getsourcefile(type(definition)));g=dict(complete=True,CPU_only=True,CUDA_initialized=False,public_register_kernel_cpu_dynamic_opaque_op_test_passed=True,CPU_FX_graph_only_not_TorchDynamo_or_GPU_compile=True,original_schema_and_fake_preserved=True,genuine_original_init_and_bank_closure_preserved=True,other_M_delegates_actual_retained_backend=True,already_constructed_FX_graph_calls_same_original_opaque_op_after_replacement=True,test_shapes=[2,64,5,64,2],graph=graph_before,torch_version=torch.__version__,registry_source=str(registry),registry_sha256=sha(registry),source_sha256=sha(__file__),not_a_GPU_support_or_model_accuracy_proof=True);(BASE/'REGISTRATION_CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps({k:v for k,v in g.items() if k!='graph'}))
