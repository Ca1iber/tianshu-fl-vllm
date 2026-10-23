# SPDX-License-Identifier: Apache-2.0
"""AST-only inspection of actual general-range model graphs, no GPU imports."""
import ast,collections,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
RAW=Path('/workspace/benchmark-results/20261023_RMS_LM_math64_tianshu')
ROOT=RAW/'RMS_A8_GU_records/fx_pass'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def label(node):return ast.unparse(node.func) if isinstance(node,ast.Call) else ''
def var(node):return node.id if isinstance(node,ast.Name) else None
state=json.loads((RAW/'result.json').read_text());assert state['phase']=='complete' and state['quality_batch']==64 and state['correct']==101 and len(state['frozen_sources'])==944
sources={};rows=[];used=set();ranges=collections.Counter()
for path in ROOT.glob('pass_*.json'):
 meta=json.loads(path.read_text());ranges[tuple(meta['compile_range'])]+=1
for path in sorted(ROOT.glob('*_1_4096.py')):
 meta_path=path.with_suffix('.json');meta=json.loads(meta_path.read_text());assert meta['compile_range']==[1,4096] and meta['replaced']==0 and meta['before_sha256']==meta['after_sha256']==sha(path)
 sources[str(path)]=sha(path);sources[str(meta_path)]=sha(meta_path);tree=ast.parse(path.read_text());definitions={};users={}
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and not (isinstance(node.value,ast.Constant) and node.value.value is None):definitions[node.targets[0].id]=node.value
 for name,value in definitions.items():
  for node in ast.walk(value):
   if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load):users.setdefault(node.id,set()).add(name)
 for node in ast.walk(tree):
  if isinstance(node,ast.Return):
   for item in ast.walk(node.value):
    if isinstance(item,ast.Name) and isinstance(item.ctx,ast.Load):users.setdefault(item.id,set()).add('<return>')
 for name,call in definitions.items():
  op=label(call)
  if op=='torch.ops.tianshu_native_prefill.fused_or_original.default':role='GateUp_SiLU';bank=ast.literal_eval(call.args[4])
  elif op=='torch.ops.tianshu_native_attention.linear.default' and ast.literal_eval(call.args[4])%2==0:role='QKV';bank=ast.literal_eval(call.args[4])//2
  else:continue
  assert len(call.args)==5 and not call.keywords and 0<=bank<42
  input_name=var(call.args[0]);assert input_name is not None;value=definitions[input_name];norm_name=input_name;residual_name=None
  if isinstance(value,ast.Subscript):
   assert ast.literal_eval(value.slice)==0;norm_name=var(value.value);norm=definitions[norm_name];assert label(norm)=='torch.ops.vllm_ir.fused_add_rms_norm.default';siblings=[k for k,v in definitions.items() if isinstance(v,ast.Subscript) and var(v.value)==norm_name];assert len(siblings)==2 and users[norm_name]==set(siblings);residual_name=next(k for k in siblings if ast.literal_eval(definitions[k].slice)==1);assert ast.literal_eval(norm.args[3])==1e-6;has_residual=True
  else:
   norm=value;assert label(norm)=='torch.ops.vllm_ir.rms_norm.default' and ast.literal_eval(norm.args[2])==1e-6;has_residual=False
  assert users[input_name]=={name};key=(role,bank);assert key not in used;used.add(key);rows.append(dict(role=role,bank=bank,source=str(path),source_sha256=sources[str(path)],norm_node=norm_name,projection_node=name,projection_op=op,has_residual=has_residual,residual_output_node=residual_name,norm_output_only_consumed_by_projection=True,compile_range=[1,4096]))
assert len(sources)==86 and len(rows)==84 and ranges=={(1,4096):43,(64,64):43}
assert all({r['bank'] for r in rows if r['role']==role}==set(range(42)) for role in ('QKV','GateUp_SiLU'))
assert sum(r['has_residual'] for r in rows if r['role']=='QKV')==41 and all(r['has_residual'] for r in rows if r['role']=='GateUp_SiLU')
config=Path('/usr/local/lib/python3.12/site-packages/vllm/config/compilation.py');text=config.read_text();assert 'compile_sizes: list[int | str] | None = None' in text and 'graph for compile size 4 will be compiled and used instead' in text
result=dict(complete=True,CPU_AST_only=True,GPU_not_started=True,Torch_not_imported=True,raw_C64_result_sha256=sha(RAW/'result.json'),model_source_count=944,actual_general_graphs=43,actual_static64_graphs=43,current_no_static4096_graph=True,current_general_range_all_unchanged=True,matched_QKV_norms=42,matched_GateUp_norms=42,QKV41_residual_first_layer_no_residual=True,GateUp42_residual=True,current_prefill_4096_uses_native_INT8_dispatch=True,general_graph_cannot_be_wholly_quantized_without_changing_BF16_fallback=True,possible_separate_static4096_compile_size_requires_new_component_and_quality=True,no_integration_implemented=True,no_quality_or_perf_gain_claim=True,configuration_source=str(config),configuration_source_sha256=sha(config),source_sha256=sha(__file__),sources=sources,matches=rows)
(BASE/'CPU_INVENTORY_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in result.items() if k not in ('sources','matches')}))
