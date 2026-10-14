# SPDX-License-Identifier: Apache-2.0
"""Execute installed benchmark payload construction without network/GPU I/O."""
import ast
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace as NS


def main():
    base=Path(__file__).resolve().parent;source=base/'reference_sources/endpoint_request_func.py'
    tree=ast.parse(source.read_text())
    helper=deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_update_payload_common'))
    function=deepcopy(next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='async_request_openai_completions'))
    end=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='headers' for t in n.targets))
    function.body=function.body[:end]+[ast.Return(value=ast.Name(id='payload',ctx=ast.Load()))]
    for fn in (helper,function):
        fn.returns=None
        for arg in [*fn.args.args,*fn.args.kwonlyargs]:arg.annotation=None
    module=ast.Module(body=[helper,function],type_ignores=[]);ast.fix_missing_locations(module)
    scope={'_validate_api_url':lambda *a:None}
    exec(compile(module,str(source),'exec'),scope)
    request=NS(api_url='http://127.0.0.1:9031/v1/completions',model_name='minicpm',model='minicpm',
               prompt=[1,2,3],output_len=1024,logprobs=None,ignore_eos=True,extra_body={})
    payload=asyncio.run(scope['async_request_openai_completions'](request,None))
    assert 'temperature' not in payload and 'top_p' not in payload
    model=json.loads((base/'reference_sources/generation_config.json').read_text())
    assert model['temperature']==1.0 and model['top_p']==.95
    cli=(base/'reference_sources/serve.py').read_text()
    assert 'from vllm.benchmarks.lib.endpoint_request_func import' in cli
    assert 'vllm bench serve no longer sets temperature==0' in cli
    # Exercise the actual future8192 request function and throughput submit args.
    foundation=base.parent/'tianshu_budget8192_buckets_20261014/foundation_service_trial.py'
    tree=ast.parse(foundation.read_text())
    fn=deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='request'))
    end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='req' for t in n.targets))
    fn.body=fn.body[:end]+[ast.Return(value=ast.Name(id='payload',ctx=ast.Load()))]
    ast.fix_missing_locations(fn);scope={'random':random}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(foundation),'exec'),scope)
    submit=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='submit')
    args=NS(input_tokens=4096,output_tokens=1024,requests=64)
    values=[eval(compile(ast.Expression(n),str(foundation),'eval'),dict(args=args,round_index=0,i=0,max=max,request=scope['request'])) for n in submit.args]
    future_payload=values[0](*values[1:])
    assert future_payload['temperature']==1.0 and future_payload['top_p']==.95
    canary=scope['request'](565,16,2026101120,False,0)
    assert canary['temperature']==0 and canary['top_p']==1
    result=dict(complete=True,cpu_only=True,network_or_GPU_not_used=True,
                actual_installed_CLI_import_path_verified=True,
                actual_official_payload_omits_temperature_and_top_p=True,
                model_default_temperature=1.0,model_default_top_p=.95,
                future8192_actual_submit_and_payload_match_model_defaults=True,
                greedy_canary_preserved=True,old_greedy_routing_claim_invalid=True,
                request_payload={k:v for k,v in payload.items() if k!='prompt'},
                future8192_payload={k:v for k,v in future_payload.items() if k!='prompt'},
                source_sha256={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (base/'reference_sources').iterdir()})
    result['future8192_foundation_sha256']=hashlib.sha256(foundation.read_bytes()).hexdigest()
    (base/'CPU_ROUTING_AUDIT_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if 'payload' not in k and 'source' not in k}))


if __name__=='__main__':main()
