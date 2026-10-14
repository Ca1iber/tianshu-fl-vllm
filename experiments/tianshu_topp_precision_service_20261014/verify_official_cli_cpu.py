# SPDX-License-Identifier: Apache-2.0
"""Real official-entrypoint argv/guard preflight with explicit synthetic inputs.

Intercepts the first Popen: no GPU or serving process, no performance/accuracy
measurement. The synthetic quality proof never reaches a GPU continuation.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser()
    for key in ('screen-root','reference-math-root','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    base=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('_budget_official_cpu',base/'run_official_4k.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    captured={}

    class CPUStop(BaseException):
        pass

    def popen(command, **kwargs):
        assert command[:2]==['vllm','serve']
        assert command[command.index('--max-num-seqs')+1] == '64'
        env=kwargs['env']
        assert command[command.index('--max-num-batched-tokens')+1]=='4096'
        assert command[command.index('--long-prefill-token-threshold')+1]=='0'
        for key in ('TIANSHU_EXTENDED_BUCKET_PROBE','TIANSHU_SQL_REFLECTION_PROBE',
                    'TIANSHU_SEGMENT_POLICY_PROBE','TIANSHU_DECODE_COHERENT_PROBE'):
            assert env[key]=='1',key
        assert env['PYTHONPATH']==str(base/'hooks')
        assert env['TIANSHU_CACHE_SERVICE_MODE']=='candidate'
        assert env['TIANSHU_TOPP_PRECISION_SERVICE_MODE']=='candidate' and env['TIANSHU_MAX_SEQS_MODE']=='candidate'
        captured.update(command=command,environment={k:v for k,v in env.items()
            if k.startswith(('TIANSHU_','VLLM_FL_','PYTHONPATH'))})
        raise CPUStop('CPU-only interception before any formal server process')

    def check_output(command, **kwargs):
        assert command==['ixsmi'],command
        return 'No running processes found (CPU fixture; no hardware query)'

    with tempfile.TemporaryDirectory(prefix='budget_official_cpu_') as directory:
        root=Path(directory);confirm=root/'SYNTHETIC_CONFIRM';quality=root/'SYNTHETIC_QUALITY'
        trials=[]
        for mode in ('control','candidate'):
            trial=json.loads((args.screen_root/mode/'result.json').read_text())
            trial['short_4k_rounds']+=copy.deepcopy(trial['short_4k_rounds'][-2:])
            for r in trial['short_4k_rounds']:r['total_tokens_per_second']=100. if mode=='control' else 103.
            d=confirm/mode;d.mkdir(parents=True);(d/'result.json').write_text(json.dumps(trial));trials.append(trial)
        shutil.copytree(args.reference_math_root/'source_snapshot',quality/'source_snapshot')
        audit=json.loads((args.reference_math_root/'accuracy_audit.json').read_text())
        audit.update(max_batched_tokens=4096,server_command=trials[1]['command'],
            candidate_hook=trials[1]['active_hook'],matched_service_root=str(confirm),
            budget4096_profile=dict(complete=True,bucket_points=83),fresh_math_for_decode_cache_cv=True,fresh_math_for_TopP_precision_maxseq64=True)
        tree=ast.parse((base/'run_math_eval.py').read_text())
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id=='own_sources' for t in n.targets))
        kv16=base.parent/'tianshu_decode_kv16_w2_20261014/hooks/sitecustomize.py'
        scope=dict(Path=Path,__file__=str(base/'run_math_eval.py'),hook=base/'hooks/sitecustomize.py',
                   kv16_hook=kv16,decode_hook=kv16.with_name('decode_sitecustomize.py'),base=base.parent)
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<actual_math_sources>','exec'),scope)
        augmentation=next(n for n in ast.walk(tree) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='own_sources')
        exec(compile(ast.Module(body=[augmentation],type_ignores=[]),'<actual_math_additional_sources>','exec'),scope)
        audit['segment_sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in scope['own_sources']}
        assert len(audit['segment_sources'])==18
        (quality/'source_snapshot/candidate_sitecustomize.py').write_bytes((base/'hooks/sitecustomize.py').read_bytes())
        (quality/'accuracy_audit.json').write_text(json.dumps(audit))
        output=root/'output'
        argv=[str(base/'run_official_4k.py'),'--output',str(output),'--control-trial',str(confirm/'control'),
              '--candidate-trial',str(confirm/'candidate'),'--math-results',str(quality),
              '--max-batched-tokens','4096','--buckets','1','--gather','1','--bucket-trace','2']
        with patch.object(sys,'argv',argv),patch.object(module.subprocess,'Popen',popen), \
             patch.object(module.subprocess,'check_output',check_output), \
             patch.object(module.socket,'create_connection',side_effect=OSError('CPU mock')), \
             patch.object(module.threading.Thread,'start',return_value=None):
            assert module.main()==1
        state=json.loads((output/'result.json').read_text())
        assert state['phase']=='failed' and state['error'].startswith('CPU-only interception')
        assert captured['command']==trials[1]['command']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    data=dict(complete=True,cpu_only=True,synthetic_fixture_not_performance_or_accuracy=True,
              real_official_entrypoint_guards_exercised=True,intercepted_before_server_process=True,
              actual_command_matches4096_service=True,GPU_benchmark_not_run=True,**captured,
              source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (base/'run_official_4k.py',base/'audit_official.py',base/'audit_profile.py',Path(__file__))})
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k not in ('command','environment')}))


if __name__=='__main__':main()
