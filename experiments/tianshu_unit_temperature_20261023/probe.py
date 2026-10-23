# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json,os,runpy,signal,subprocess,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
s=importlib.util.spec_from_file_location('_unit_exact_r4',EXP/'tianshu_W4_LMpad_service_v2_20261023/candidate_profile.py');model=importlib.util.module_from_spec(s);s.loader.exec_module(model)
root=Path('/workspace/benchmark-results/20261023_unit_temperature_component_tianshu');assert not root.exists();root.mkdir();assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True);signal.alarm(300)
result=dict(phase='preparing',cases=[],official_benchmark=False,adopted=False)
def save():(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
try:
 command,env,profile,sources=model.prepare(root,'candidate');os.environ.update(profile);runpy.run_path(str(Path(profile['PYTHONPATH'])/'sitecustomize.py'))
 import numpy as np
 import torch,flag_gems,vllm_fl
 from types import SimpleNamespace
 from vllm.v1.sample.sampler import Sampler
 from vllm.v1.sample.metadata import SamplingMetadata
 import policy
 torch.set_num_threads(8);installed=policy.install('candidate',root/'unit_records');sampler=Sampler();assert not sampler.use_fp64_gumbel;blank=torch.empty(0,device='cuda')
 result['sources']=sources|{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*BASE.rglob('*.py'),BASE/'PLAN_32g.md',BASE/'EXECUTION.sha256']};save()
 with torch.inference_mode(),flag_gems.use_gems():
  for index,(b,t,gens) in enumerate([(64,1.,'none'),(64,.7,'none'),(64,1.4,'none'),(64,1.,'some'),(8,1.,'all'),(1,1.,'all'),(8,'mixed','some'),(8,1.,'unknown')]):
   torch.manual_seed(2026102370+index);x=torch.randn((b,130560),device='cuda').bfloat16();before=x.view(torch.int16).clone();temps=np.full(b,1. if isinstance(t,str) else t,dtype=np.float32)
   if t=='mixed':temps[0]=.7
   generators={i:torch.Generator(device='cuda').manual_seed(700+i) for i in (range(b) if gens=='all' else ([0] if gens=='some' else []))}
   meta=SamplingMetadata(temperature=torch.tensor(temps,device='cuda'),all_greedy=False,all_random=True,top_p=torch.full((b,),.95,device='cuda'),top_k=None,generators=generators,max_num_logprobs=None,no_penalties=True,prompt_token_ids=None,frequency_penalties=blank,presence_penalties=blank,repetition_penalties=blank,output_token_ids=[[] for _ in range(b)],allowed_token_ids_mask=None,bad_words_token_ids={},logitsprocs=SimpleNamespace(argmax_invariant=[],non_argmax_invariant=[]))
   if gens!='unknown':policy.attach(meta,temps,b,True)
   global0=torch.cuda.get_rng_state().clone();states={i:g.get_state().clone() for i,g in generators.items()};policy.STATE['mode']='control';ref=sampler(x,meta);global1=torch.cuda.get_rng_state().clone();states1={i:g.get_state().clone() for i,g in generators.items()}
   torch.cuda.set_rng_state(global0)
   for i,g in generators.items():g.set_state(states[i])
   policy.STATE['mode']='candidate';out=sampler(x,meta);torch.cuda.synchronize();assert torch.equal(ref.sampled_token_ids,out.sampled_token_ids) and torch.equal(global1,torch.cuda.get_rng_state());assert all(torch.equal(g.get_state(),states1[i]) for i,g in generators.items());assert torch.equal(before,x.view(torch.int16))
   # Complete operator-level equivalence, not only sampled IDs.
   original=x.float();candidate=original.clone();expected=installed['original_temperature'](original,meta.temperature,True);token=policy.CTX.set(meta)
   try:actual=Sampler.apply_temperature(candidate,meta.temperature,True)
   finally:policy.CTX.reset(token)
   assert torch.equal(expected.view(torch.int32),actual.view(torch.int32));result['cases'].append(dict(batch=b,temperature=t,generators=gens,full_temperature_output_bits_equal=True,tokens_equal=True,global_and_per_request_RNG_equal=True,input_unchanged=True));save()
 assert policy.STATE['skipped']>0 and policy.STATE['fallback']>0;installed['save']();model.frozen_check(result['sources']);result.update(phase='complete',complete=True,GPU_checks_passed=True)
except BaseException as e:result.update(phase='failed',error=repr(e),traceback=traceback.format_exc());raise
finally:save()
