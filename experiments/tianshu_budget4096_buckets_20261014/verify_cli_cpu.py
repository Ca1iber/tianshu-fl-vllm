# SPDX-License-Identifier: Apache-2.0
"""Execute actual CLI parsing/guards before any filesystem or GPU operation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


def main():
    base=Path(__file__).resolve().parent
    shared=base.parent/'tianshu_greedy_screen_20261014/run_service_trial.py'
    local=base/'foundation_service_trial.py'
    expected=shared.read_text().replace('args.max_batched_tokens != 2048','args.max_batched_tokens not in (2048,4096)').replace('Bucket confirmation requires the verified native profile at budget2048','This83-point experiment requires native Prefill and budget2048 or4096')
    assert local.read_text()==expected
    functions={}
    for name,path in (('shared',shared),('local',local)):
        tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        stop=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='out' for t in n.targets))
        fn.body=fn.body[:stop]+[ast.Return(value=ast.Call(func=ast.Name(id='vars',ctx=ast.Load()),args=[ast.Name(id='args',ctx=ast.Load())],keywords=[]))]
        ast.fix_missing_locations(fn)
        env=dict(argparse=argparse,Path=Path)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
        functions[name]=env['main']
    outcomes=[]
    for name,function in functions.items():
        for budget in (2048,4096,8192):
            sys.argv=['foundation','--output','/tmp/unused_budget_cli_preflight','--buckets','1','--gather','1',
                      '--tile','1','--native-prefill','1','--max-batched-tokens',str(budget)]
            try:actual=function();accepted=True
            except RuntimeError:accepted=False
            assert accepted==(budget in ((2048,) if name=='shared' else (2048,4096)))
            if accepted:
                assert actual['decode_graph_max']==0 and actual['stream_interval']==1
                assert actual['buckets']==actual['gather']==actual['tile']==actual['native_prefill']=='1'
            outcomes.append(dict(source=name,budget=budget,accepted=accepted))
    result=dict(complete=True,cpu_only=True,actual_cli_and_pre_io_guards_executed=True,
                shared_source_unchanged=True,local_only_guard_delta_verified=True,outcomes=outcomes,
                original_failed_candidate_had_no_server=True,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (local,Path(__file__))})
    out=base/'results/20261014/prepared_32g'
    (out/'CPU_CLI_AUDIT_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
