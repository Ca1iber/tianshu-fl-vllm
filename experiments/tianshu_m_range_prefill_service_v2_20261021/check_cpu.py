# SPDX-License-Identifier: Apache-2.0
"""Actually exercise matcher/FX surgery and both source/argv prepares without CUDA."""
import hashlib,importlib.util,json,sys,tempfile,types
from pathlib import Path
BASE=Path(__file__).resolve().parent
import torch
assert not torch.cuda.is_initialized()
sys.path.insert(0,str(BASE))
from range_match import match
lib=torch.library.Library('tianshu_prefill_int8','DEF');lib.define('linear(Tensor x, Tensor weight, Tensor packed, Tensor scales, int bank_id) -> Tensor')
newlib=torch.library.Library('tianshu_m_range_prefill','DEF');newlib.define('fused_or_original(Tensor x, Tensor weight, Tensor packed, Tensor scales, int bank_id) -> Tensor')

def make(m,n=12288,bank=7):
 graph=torch.fx.Graph();x,w,p,s=[graph.placeholder(name) for name in ('x','weight','packed','scales')]
 linear=graph.call_function(torch.ops.tianshu_prefill_int8.linear.default,(x,w,p,s,bank));linear.meta['val']=torch.empty((m,n),device='meta',dtype=torch.bfloat16)
 gate=graph.call_function(torch.ops.aten.slice.Tensor,(linear,1,0,6144));up=graph.call_function(torch.ops.aten.slice.Tensor,(linear,1,6144,9223372036854775807))
 cast=graph.call_function(torch.ops.prims.convert_element_type.default,(gate,torch.float32));sig=graph.call_function(torch.ops.aten.sigmoid.default,(cast,));mul=graph.call_function(torch.ops.aten.mul.Tensor,(cast,sig));rounded=graph.call_function(torch.ops.prims.convert_element_type.default,(mul,torch.bfloat16));out=graph.call_function(torch.ops.aten.mul.Tensor,(rounded,up));out.meta['val']=torch.empty((m,6144),device='meta',dtype=torch.bfloat16);graph.output(out)
 return graph,linear
class StubPass:
 def hash_dict(self,value):return str(value)
class Parent:
 def __call__(self,graph):pass
 def uuid(self):return 'actual-parent-pass-runtime-is-separately-audited'
context=types.SimpleNamespace(compile_range=types.SimpleNamespace(start=1,end=4096))
text=(BASE/'range_pass.py').read_text();needle='from vllm.compilation.passes.inductor_pass import InductorPass,get_pass_context\n';assert text.count(needle)==1
namespace=dict(__file__=str(BASE/'range_pass.py'),InductorPass=StubPass,get_pass_context=lambda:context);exec(compile(text.replace(needle,''),str(BASE/'range_pass.py'),'exec'),namespace);RangePass=namespace['RangePass']
root=Path('/workspace/benchmark-results/20261021_m_range_prefill_v2_CPU_prepare_tianshu');root.mkdir(exist_ok=False);checks=0
for m in (1,8,63,64,128,256,384,4096):
 for mode in ('control','candidate'):
  for bounds in ((1,4096),(64,64)):
   context.compile_range.start,context.compile_range.end=bounds;graph,node=make(m);assert match(node)['bank_id']==7
   before=graph.python_code('self').src;RangePass(Parent(),mode,root/f'fx_{m}_{mode}_{bounds[0]}_{bounds[1]}',BASE/'selected_plan.json')(graph);graph.lint();after=graph.python_code('self').src
   new_calls=[n for n in graph.nodes if n.op=='call_function' and str(n.target)=='tianshu_m_range_prefill.fused_or_original.default']
   assert len(new_calls)==(1 if mode=='candidate' and bounds==(1,4096) else 0)
   if not new_calls:assert before==after
   checks+=1
_,bad=make(128,n=1024);assert match(bad) is None
try:
 _,bad=make(128,bank=42);match(bad)
except AssertionError:pass
else:raise AssertionError('invalid bank accepted')
from torch.fx.experimental.symbolic_shapes import ShapeEnv
symbol=ShapeEnv().create_unbacked_symint();graph,node=make(symbol);assert match(node)['bank_id']==7
context.compile_range.start,context.compile_range.end=1,4096;RangePass(Parent(),'candidate',root/'fx_symbolic',BASE/'selected_plan.json')(graph);graph.lint();checks+=1
spec=importlib.util.spec_from_file_location('_range_CPU_profile',BASE/'candidate_profile.py');profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
arms={}
for mode in ('control','candidate'):
 armroot=root/mode;armroot.mkdir();command,env,selected,sources=profile.prepare(armroot,mode);assert len(command)==13 and len(selected)==62
 arms[mode]=dict(command=command,environment=selected,sources=sources)
assert arms['control']['command']==arms['candidate']['command'] and arms['control']['sources']==arms['candidate']['sources']
raw_differences={k for k in arms['control']['environment'] if arms['control']['environment'][k]!=arms['candidate']['environment'][k]}
normalized={mode:{k:v.replace(str(root/mode),'__ARM_OUTPUT_ROOT__') for k,v in arm['environment'].items()} for mode,arm in arms.items()}
normalized_differences={k for k in normalized['control'] if normalized['control'][k]!=normalized['candidate'][k]}
assert normalized_differences=={'PYTHONPATH'},sorted(normalized_differences)
(root/'prepared_arm_environments.json').write_text(json.dumps({mode:arm['environment'] for mode,arm in arms.items()},indent=2)+'\n')
assert not torch.cuda.is_initialized() and 'flag_gems' not in sys.modules
report=dict(complete=True,CPU_only=True,actual_entry_prepare_passed=True,CUDA_not_initialized=True,actual_meta_and_symbolic_FX_cases=checks,context_stubbed_only_for_CPU_pass_logic=True,real_runtime_context_still_requires_GPU_audit=True,both_modes_prepared=True,original_argv_length=13,environment_keys=62,identical_numerical_source_maps=True,source_count=len(arms['candidate']['sources']),only_normalized_environment_difference='PYTHONPATH',output_roots_canonicalized_for_comparison=True,raw_environment_difference_keys=sorted(raw_differences),raw_root=str(root),sources=arms['candidate']['sources'])
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))
