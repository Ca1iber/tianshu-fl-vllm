# SPDX-License-Identifier: Apache-2.0
import ast,json
import os
os.environ['VLLM_PLUGINS']=''
from pathlib import Path
import torch
from cache_probe_profile import BASE,CASES,payload,prepare,digest

def main():
    assert not torch.cuda.is_initialized()
    for p in BASE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest,UsageInfo
    usage=UsageInfo(prompt_tokens=128,completion_tokens=16,total_tokens=144).model_dump()
    assert {k:usage.get(k) for k in ('prompt_tokens','completion_tokens','total_tokens')}==dict(prompt_tokens=128,completion_tokens=16,total_tokens=144)
    normalized=[]
    for label,salt in CASES:
        body=payload(salt);request=CompletionRequest.model_validate(body)
        assert request.cache_salt==salt and request.return_token_ids and request.seed==777
        assert request.prompt==body['prompt'] and len(request.prompt)==128 and request.max_tokens==16
        row=dict(body);row.pop('cache_salt',None);normalized.append(row)
    assert all(row==normalized[0] for row in normalized) and len(CASES)==8
    assert len({salt for _,salt in CASES if salt is not None})==3
    root=Path('/workspace/benchmark-results/20261021_prefill_cache_canary_v2_CPU_prepare_tianshu');assert not root.exists();root.mkdir()
    command,environment,profile,sources=prepare(root)
    for p,h in sources.items():assert digest(p)==h==digest(root/'source_snapshot'/Path(p).relative_to('/'))
    controllers={str(p):digest(p) for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run_probe.sh']}
    assert not torch.cuda.is_initialized()
    gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,GPU_not_executed=True,
              original_407_source_map_exactly_equal=True,original_13_argv_and_62_environment_keys=True,
              actual_CompletionRequest_salt_and_token_ids_validated=True,eight_payloads_only_differ_by_cache_salt=True,
              no_stage_observer_hook=True,actual_UsageInfo_serialized_keys=sorted(usage),sources=sources,controllers=controllers)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','controllers')}))
if __name__=='__main__':main()
