# SPDX-License-Identifier: Apache-2.0
"""Exercise actual outer Router around a parent with no .fn, plus current profile binding."""
import ast,hashlib,importlib.util,inspect,json
from pathlib import Path
from types import SimpleNamespace as NS
from candidate_profile import BASE,prepare,digest
import grid_route
from semantic_gate import check_cpu as semantic_check
for path in BASE.rglob('*.py'):ast.parse(path.read_text())
root=Path('/workspace/benchmark-results/20261021_topp_grid_current_service_CPU_tianshu');root.mkdir(parents=True,exist_ok=False)
generator=BASE.parent/'tianshu_topp_geometric_20261016/precision.py'
spec=importlib.util.spec_from_file_location('_grid_CPU_geo_source',generator)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
native_path=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_triton.py')
geo=root/'generated_geo.py';geo.write_text(helper.render_source(native_path.read_bytes(),'geo'))
assert digest(geo)==grid_route.GEO_SHA
class Device:
 def __init__(self,name):self.type=name
cuda,cpu=Device('cuda'),Device('cpu')
class Tensor:
 def __init__(self,b=64,v=130560,dtype='torch.float32',device=cuda,col=1,row=None,ndim=2):self.shape=(b,v);self.dtype=dtype;self.device=device;self.col=col;self.row=v if row is None else row;self.ndim=ndim
 def stride(self,index):return self.row if index==0 else self.col
 def new_empty(self,shape):return Tensor(*shape,dtype=self.dtype,device=self.device)
compiled=lambda w=16:NS(metadata=NS(num_warps=w,num_stages=1,shared=0),n_regs=1,n_spills=0,kernel=b'CPU_fixture_only',hash='CPU_fixture_only')
original_calls=[];candidate_calls=[];observed=[]
class ParentRouter:
 def __getitem__(self,grid):
  def launch(*args,**kw):original_calls.append((grid,args,kw));return compiled(4)
  return launch
class GeoJIT:
 fn=NS(__module__='CPU_fake_geo',__code__=NS(co_filename=str(geo)))
 def __getitem__(self,grid):
  def launch(*args,**kw):candidate_calls.append((grid,args,kw));return compiled()
  return launch
original=ParentRouter();assert not hasattr(original,'fn')
native=NS(__file__=str(native_path),_topk_topp_kernel=original)
route=grid_route.install(native,NS(__file__=str(geo),_topk_topp_kernel=GeoJIT()),True,observed.append)
base=dict(TOPK_ENABLED=False,TOPP_ENABLED=True,BLOCK_SIZE=8192,BLOCK_SIZE_TRUNC=4096,VOCAB_SIZE=130560,BATCH_SIZE=64)
cases=[]
for b in (1,7,8,16,17,31,32,33,63,64,65):cases.append((Tensor(b=b),dict(base,BATCH_SIZE=b),8<=b<=64))
for change,want in ((dict(dtype='torch.bfloat16'),False),(dict(device=cpu),False),(dict(col=2),False),(dict(row=130567),True),(dict(v=130559),False),(dict(ndim=3),False)):
 cases.append((Tensor(**change),dict(base),want))
for change in (dict(TOPK_ENABLED=True),dict(TOPP_ENABLED=False),dict(BLOCK_SIZE=16384),dict(BLOCK_SIZE_TRUNC=8192),dict(VOCAB_SIZE=130559),dict(BATCH_SIZE=8)):
 cases.append((Tensor(),dict(base,**change),False))
count=0
for tensor,kwargs,want in cases:
 grid=(min(kwargs['BATCH_SIZE'],16),);args=(tensor,tensor.stride(0),Tensor(b=grid[0]))
 assert grid_route.supported(args,kwargs)==want
 for enabled in (False,True):
  route.enabled=enabled;before=dict(kwargs);oc,cc=len(original_calls),len(candidate_calls)
  route[grid](*args,**kwargs);assert kwargs==before
  active=want and enabled
  assert len(candidate_calls)==cc+int(active) and len(original_calls)==oc+int(not active)
  if active:
   actual_grid,actual_args,actual_kw=candidate_calls[-1]
   assert actual_grid==(min(tensor.shape[0],grid[0]*2),)
   assert actual_kw==dict(kwargs,BLOCK_SIZE=16384,BLOCK_SIZE_TRUNC=8192,num_warps=16,num_stages=1)
   assert actual_args[:2]==args[:2] and actual_args[2].shape[0]>=actual_grid[0]
   assert route.last_metadata['actual_candidate']['source_sha256']==grid_route.GEO_SHA
  else:assert original_calls[-1]==(grid,args,kwargs)
  assert route.last_metadata['active']==active;count+=1
inner=inspect.getclosurevars(route.__getitem__).nonlocals['inner']
assert len(inner.__class__.__getitem__.__globals__['SCRATCH'])==1
assert semantic_check()==10
from contextlib import contextmanager
from profile_context import with_warmup
events=[];token=object()
@contextmanager
def native_context(*args,**kwargs):
 assert args==(11,) and kwargs==dict(weights_memory=22);events.append("native_enter");yield token;events.append("native_exit")
def warm(result):assert result is token;events.append("warm")
def finished(result):assert result is token;events.append("finished")
with with_warmup(native_context,warm,finished)(11,weights_memory=22) as result:
 assert result is token;events.append("body")
assert events==["native_enter","warm","body","native_exit","finished"]
def launch_prefix(path):
 tree=ast.parse(path.read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="launch")
 end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="compiled" for t in n.targets))
 return ast.dump(ast.Module(body=fn.body[:end+1],type_ignores=[]),include_attributes=False)
assert launch_prefix(BASE/"kernel_route.py")==launch_prefix(BASE.parent/"tianshu_topp_grid_current_api_20261021/kernel_route.py")
assert (BASE/"grid_route.py").read_bytes()==(BASE.parent/"tianshu_topp_grid_current_api_20261021/grid_route.py").read_bytes()
worker=Path("/usr/local/lib/python3.12/site-packages/vllm_fl/worker/worker.py").read_text()
assert worker.index("profile_torch_peak =",worker.index("with memory_profiling_fl("))<worker.index("profile_result.torch_peak_increase =",worker.index("with memory_profiling_fl("))
runner_tree=ast.parse(Path("/usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py").read_text())
fn=next(n for n in ast.walk(runner_tree) if isinstance(n,ast.FunctionDef) and n.name=="profile_cudagraph_memory")
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="capture_model" for n in ast.walk(fn))
profiles=[]
for mode in ('control','candidate'):
 path=root/mode;path.mkdir();command,environment,profile,sources=prepare(path,mode)
 assert len(command)==13 and len(profile)==62 and profile['PYTHONPATH']==str(BASE/('hooks_'+mode))
 assert all(profile[k]=='candidate' for k in ('TIANSHU_DENSE_INT8_MODE','TIANSHU_LM_PRECISION_MODE','TIANSHU_PREFILL_LINEAR_MODE','TIANSHU_SEQ_LIMIT_MODE'))
 profiles.append(dict(mode=mode,root=str(path),command=command,profile=profile,sources=sources))
a,b=profiles;assert a['sources']==b['sources'] and a['command']==b['command']
assert {k:v.replace(a['root']+'/',b['root']+'/').replace('/hooks_control','/hooks_candidate') for k,v in a['profile'].items()}==b['profile']
gate=dict(complete=True,CPU_only=True,no_GPU_started=True,parent_runtime_sources=407,runtime_source_count=len(a['sources']),environment_count=62,
 actual_outer_router_tests=count,parent_Router_without_fn_supported=True,single_shared_scratch_buffer=True,semantic_fixture_checks=10,
 native_profile_enter_warm_body_exit_order_executed=True,GPU_launch_prefix_AST_identical_to_verified_API=True,only_post_launch_metadata_hash_caching_added=True,
 exact_current_GEO_source_sha256=digest(geo),generated_CPU_source=str(geo),profiles=profiles)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k!='profiles'}))
