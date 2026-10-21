# SPDX-License-Identifier: Apache-2.0
import ast, json
from types import SimpleNamespace as NS
from common import BASE, SITE, CASES, actual_method, manifest, oracle
from policy import seven_slot_input_fit
for path in BASE.glob('*.py'): ast.parse(path.read_text())
actual = actual_method(SITE/'vllm_fl/worker/model_runner.py','ModelRunnerFL','_input_fits_in_drafter')
wrapped = seven_slot_input_fit(actual); checks=[]; differences=[]
for arch in ('Qwen3DSparkModel','DFlashQwen3DSparkModel','DFlashQwen3ForCausalLM'):
 for count in (1,2,3,7):
  for limit in (128,131072):
   for length in range(limit-10,limit+3):
    spec=NS(use_dflash=lambda:True,draft_model_config=NS(hf_config=NS(architectures=[arch])))
    runner=NS(speculative_config=spec,num_spec_tokens=count,effective_drafter_max_model_len=limit)
    cad=NS(max_seq_len=length)
    old=actual(runner,cad); new=wrapped(runner,cad)
    expected=length+(7 if arch!='DFlashQwen3ForCausalLM' else count+1)<=limit
    assert new==expected and actual(runner,None) is False and wrapped(runner,None) is False
    checks.append(new)
    if old!=new: differences.append(dict(arch=arch,count=count,limit=limit,seq_len=length,native=old,seven_slots=new))
aux=actual_method(SITE/'vllm_fl/worker/model_runner.py','ModelRunnerFL','_get_eagle3_aux_layers_from_config')
cfg=json.loads((__import__('pathlib').Path('/workspace/MiniCPM5-2B-DSpark/config.json')).read_text())
hf=NS(dflash_config=dict(target_layer_ids=cfg['target_layer_ids']))
assert aux(NS(speculative_config=NS(draft_model_config=NS(hf_config=hf))))==(2,11,21,31,40)
fixtures=0
for case in CASES:
 for phase in (0,1):
  past=[x+phase for x in case['past']]; longest=max(a+b for a,b in zip(case['context'],past))
  cols=(longest+7+15)//16+3; pages=[[101+i*cols+j for j in range(cols)] for i in range(len(past))]
  r=oracle(case['context'],past,case['rejected'],[20261021+i+phase for i in range(len(past))],pages)
  assert len(r['query_positions'])==7*len(past) and max(r['query_positions'])<131072
  assert len(r['context_positions'])==sum(case['context']) and len(set(r['query_slots']))==7*len(past)
  fixtures+=1
r=dict(complete=True,CPU_only=True,GPU_not_used=True,actual_installed_input_fit_executed=True,input_limit_cases=len(checks),native_rule_differences=differences,actual_installed_aux_layer_method_checked=True,aux_layers=[2,11,21,31,40],input_fixtures=fixtures,model_executed=False,sources=manifest())
(BASE/'CPU_GATE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('sources','native_rule_differences')}))
