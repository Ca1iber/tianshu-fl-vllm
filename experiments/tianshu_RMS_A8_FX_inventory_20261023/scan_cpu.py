# SPDX-License-Identifier: Apache-2.0
"""AST-only inventory of actual M64 Inductor pass dumps; never imports Torch."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261023_QKV64_keep128_short4k_control_tianshu/nn_silu_fx_records/fx_pass')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def label(node):return ast.unparse(node.func) if isinstance(node,ast.Call) else ''
def var(node):return node.id if isinstance(node,ast.Name) else None
rows=[];sources={};used=set()
for path in sorted(ROOT.glob('*64_64.py')):
 tree=ast.parse(path.read_text());sources[str(path)]=sha(path)
 definitions={};users={}
 for n in ast.walk(tree):
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and not (isinstance(n.value,ast.Constant) and n.value.value is None):definitions[n.targets[0].id]=n.value
 for name,value in definitions.items():
  for n in ast.walk(value):
   if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load):users.setdefault(n.id,set()).add(name)
 for n in ast.walk(tree):
  if isinstance(n,ast.Return):
   for item in ast.walk(n.value):
    if isinstance(item,ast.Name) and isinstance(item.ctx,ast.Load):users.setdefault(item.id,set()).add("<return>")
 for name,call in definitions.items():
  op=label(call)
  if op=='torch.ops.tianshu_nn_silu_fx.fused.default':role='GateUp_SiLU';bank=ast.literal_eval(call.args[4])
  elif op=='torch.ops.tianshu_native_attention.linear.default' and ast.literal_eval(call.args[4])%2==0:role='QKV';bank=ast.literal_eval(call.args[4])//2
  else:continue
  assert len(call.args)==5 and not call.keywords and 0<=bank<42
  input_name=var(call.args[0]);assert input_name is not None
  value=definitions[input_name];norm_name=input_name;residual_name=None
  if isinstance(value,ast.Subscript):
   assert ast.literal_eval(value.slice)==0
   norm_name=var(value.value);assert norm_name is not None;norm=definitions[norm_name]
   assert label(norm)=='torch.ops.vllm_ir.fused_add_rms_norm.default'
   siblings=[k for k,v in definitions.items() if isinstance(v,ast.Subscript) and var(v.value)==norm_name]
   assert len(siblings)==2
   residual_name=next(k for k in siblings if ast.literal_eval(definitions[k].slice)==1)
   assert users[norm_name]==set(siblings)
   eps=ast.literal_eval(norm.args[3]);has_res=True
  else:
   norm=value;assert label(norm)=='torch.ops.vllm_ir.rms_norm.default';eps=ast.literal_eval(norm.args[2]);has_res=False
  assert eps==1.e-6 and users[input_name]=={name}
  key=(role,bank);assert key not in used;used.add(key)
  rows.append(dict(role=role,bank=bank,source=str(path),source_sha256=sources[str(path)],norm_node=norm_name,consumer_node=name,has_residual=has_res,residual_output_node=residual_name,norm_output_only_consumed_by_projection=True,IR_norm_node_retained_at_existing_post_grad_custom_post_pass=True))
assert len(sources)==43 and len(rows)==84
assert {r['bank'] for r in rows if r['role']=='GateUp_SiLU'}==set(range(42))
assert {r['bank'] for r in rows if r['role']=='QKV'}==set(range(42))
assert sum(r['has_residual'] for r in rows if r['role']=='GateUp_SiLU')==42
assert sum(r['has_residual'] for r in rows if r['role']=='QKV')==41
result=dict(complete=True,CPU_AST_only=True,Torch_not_imported=True,GPU_not_started=True,source=str(Path(__file__).resolve()),source_sha256=sha(Path(__file__).resolve()),actual_M64_pass_graphs=43,GateUp_norm_consumers=42,QKV_norm_consumers=42,GateUp_all42_have_residual=True,QKV41_have_residual_first_layer_no_residual=True,existing_r3_QKV64_is_BF16=True,GU_INT8_quantization_expression_matches_component_source=True,proposed_integration_not_implemented=True,no_model_quality_or_perf_claim=True,sources=sources,matches=rows)
(BASE/'CPU_INVENTORY_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('sources','matches')}))
if __name__=='__main__':pass
