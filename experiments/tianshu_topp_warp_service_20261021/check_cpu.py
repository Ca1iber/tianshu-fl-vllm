# SPDX-License-Identifier: Apache-2.0
import ast,json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from route import KernelRouter,eligible,profile_with_warmup
from candidate_profile import BASE,prepare,read,digest
for p in BASE.rglob('*.py'):ast.parse(p.read_text())
# Execute the actual context wrapper and check enter/warm/body/exit identity/order.
events=[];token=object()
@contextmanager
def native(*args,**kw):
 assert args==(11,) and kw==dict(weights_memory=22);events.append('native_enter');yield token;events.append('native_exit')
def warm(result):assert result is token;events.append('warm')
def finished(result):assert result is token;events.append('finished')
with profile_with_warmup(native,warm,finished)(11,weights_memory=22) as result:
 assert result is token;events.append('body')
assert events==['native_enter','warm','body','native_exit','finished']
class Tensor:
 def __init__(self,b=64,v=130560,dtype='torch.float32',device='cuda',row=130560,col=1):self.shape=(b,v);self.dtype=dtype;self.device=NS(type=device);self.strides=(row,col)
 def stride(self,i):return self.strides[i]
base=dict(TOPK_ENABLED=False,TOPP_ENABLED=True,BLOCK_SIZE=8192,BLOCK_SIZE_TRUNC=4096,VOCAB_SIZE=130560,BATCH_SIZE=64)
seen=[];observed=[]
class Kernel:
 marker=object()
 def __getitem__(self,grid):
  def call(*args,**kw):seen.append((grid,args,kw));return self.marker
  return call
original=Kernel();selection=dict(value=None)
router=KernelRouter(original,lambda:selection['value'],lambda *x:observed.append(x))
assert router.marker is original.marker
cases=[(Tensor(),dict(base),True)]
for b in (1,7,8,63,64,65):cases.append((Tensor(b=b),dict(base,BATCH_SIZE=b),8<=b<=64))
for change in (dict(dtype='torch.bfloat16'),dict(device='cpu'),dict(v=130559),dict(row=130567),dict(col=2)):cases.append((Tensor(**change),dict(base),False))
for change in (dict(TOPK_ENABLED=True),dict(TOPP_ENABLED=False),dict(BLOCK_SIZE=4096),dict(BLOCK_SIZE_TRUNC=2048),dict(VOCAB_SIZE=130559),dict(BATCH_SIZE=8)):cases.append((Tensor(),dict(base,**change),False))
count=0
for tensor,kw,expected in cases:
 assert eligible(tensor,kw)==expected
 for wanted in (None,4,16):
  selection['value']=wanted;before=dict(kw);n=len(observed);answer=router[(64,)](tensor,130560,object(),**kw)
  assert answer is original.marker and kw==before and seen[-1][0]==(64,) and seen[-1][1][0] is tensor
  assert seen[-1][2]==(dict(kw,num_warps=wanted) if expected and wanted is not None else kw)
  assert len(observed)==n+int(expected);count+=1
selection=read(BASE/'selected_plan.json');gate=read(selection['component_gate'])
assert gate['complete'] and gate['CPU_reverified'] and gate['full_FP32_outputs_reconstructed_from_masks_and_original_values']==180
assert not gate['bitwise_native_all_cases_by_config']['warp16'] and selection['exploratory_non_bitwise_candidate']
assert read(selection['layout_equivalence'])['all_full_output_SHA_and_masks_identical']
# Native worker freezes profile_torch_peak inside the with block: pre-yield warm is mandatory.
worker=Path('/usr/local/lib/python3.12/site-packages/vllm_fl/worker/worker.py').read_text()
assert worker.index('profile_torch_peak =',worker.index('with memory_profiling_fl('))<worker.index('profile_result.torch_peak_increase =',worker.index('with memory_profiling_fl('))
profiles=[]
for mode in ('control','candidate'):
 initial=Path(f'/workspace/benchmark-results/20261021_topp_warp_service_CPU_{mode}_tianshu');root=initial;revision=0
 while root.exists():revision+=1;root=initial.with_name(initial.name+f'_v{revision}')
 root.mkdir();command,env,profile,sources=prepare(root,mode)
 assert len(profile)==62 and len(command)==13 and profile['PYTHONPATH']==str(BASE/('hooks_'+mode))
 assert all(profile[k]=='candidate' for k in ('TIANSHU_DENSE_INT8_MODE','TIANSHU_LM_PRECISION_MODE','TIANSHU_PREFILL_LINEAR_MODE','TIANSHU_SEQ_LIMIT_MODE'))
 profiles.append(dict(mode=mode,root=str(root),command=command,profile=profile,sources=sources))
a,b=profiles;assert a['sources']==b['sources'] and a['command']==b['command']
assert {k:v.replace(a['root']+'/',b['root']+'/').replace('/hooks_control','/hooks_candidate') for k,v in a['profile'].items()}==b['profile']
r=dict(complete=True,CPU_only=True,no_GPU_started=True,parent_runtime_sources=407,runtime_source_count=len(a['sources']),environment_count=62,original13_serve_arguments_preserved=True,both_arms_current_prefill_INT8=True,actual_router_forwarding_cases=count,native_context_pre_yield_warm_order_executed=True,old_component_bitwise_failure_preserved=True,new_model_Math_required=True,profiles=profiles)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='profiles'}))
