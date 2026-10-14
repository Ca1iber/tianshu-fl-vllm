# SPDX-License-Identifier: Apache-2.0
"""Run the actual confirm argv through the actual foundation's pre-I/O guards."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS


def main():
    base=Path(__file__).resolve().parent
    trial=base/'run_confirm_trial.py';foundation=base/'foundation_service_trial.py'
    tree=ast.parse(trial.read_text())
    argv=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='sys' and t.attr=='argv' for t in n.targets))
    environment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='actual_env' for t in n.targets))
    fn=next(n for n in ast.parse(foundation.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    stop=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='out' for t in n.targets))
    fn.body=fn.body[:stop]+[ast.Return(value=ast.Call(func=ast.Name(id='vars',ctx=ast.Load()),args=[ast.Name(id='args',ctx=ast.Load())],keywords=[]))]
    ast.fix_missing_locations(fn);env=dict(argparse=argparse,Path=Path)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(foundation),'exec'),env)
    results=[]
    for mode,budget in (('control',2048),('candidate',4096)):
        scope=dict(trial=foundation,root=Path('/tmp/unused_confirm_'+mode),budget=budget,hook=base/'hooks/sitecustomize.py',
                   args=NS(mode=mode),sql_enabled=True,sys=NS(argv=[]))
        for assignment in (argv,environment):exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(trial),'exec'),scope)
        sys.argv=scope['sys'].argv
        args=env['main']()
        assert args['rounds']==5 and args['requests']==args['concurrency']==64
        assert args['input_tokens']==4096 and args['output_tokens']==1024
        assert args['max_batched_tokens']==budget and args['buckets']==args['gather']==args['native_prefill']=='1'
        assert scope['actual_env']['TIANSHU_SQL_REFLECTION_PROBE']==scope['actual_env']['TIANSHU_EXTENDED_BUCKET_PROBE']=='1'
        results.append(dict(mode=mode,budget=budget,rounds=5,requests_per_round=64,excluded_rounds=[1,2],actual_argv=sys.argv))
    out=base/'results/20261014/confirm_prepared_32g';out.mkdir(parents=True,exist_ok=True)
    data=dict(complete=True,cpu_only=True,actual_confirm_argv_and_foundation_guards_exercised=True,
              actual_shared_SQL_and_extended_bucket_environment_exercised=True,runs=results,
              source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (trial,foundation,base/'run_confirm_pair.py',Path(__file__))},
              GPU_confirmation_not_run=True)
    (out/'CPU_CLI_AUDIT_32g.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k!='runs'}))


if __name__=='__main__':main()
