# SPDX-License-Identifier: Apache-2.0
"""Check executable launch argv against actual completed Math and official4K."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--launcher',type=Path,required=True)
    p.add_argument('--math-root',type=Path,required=True);p.add_argument('--formal-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    launcher=args.launcher.resolve()
    qa=json.loads((args.math_root/'result.json').read_text());formal=json.loads((args.formal_root/'result.json').read_text())
    quality=json.loads((args.math_root/'accuracy_audit.json').read_text());score=json.loads((args.formal_root/'comparison.json').read_text())
    assert formal['phase']=='complete' and score['complete'] and score['official_benchmark']
    assert quality['complete'] and quality['correct']/105>.95 and quality['fresh_math_for_decode_kv16_w1']
    assert quality['fresh_math_for_B64S8_segments'] and score['fresh_math_for_B64S8_segments']
    assert quality['fresh_math_for_decode_cache_cv'] and score['fresh_math_for_decode_cache_cv']
    assert score['math_correct']==quality['correct'] and score['math_total']==105
    subprocess.run(['bash','-n',str(launcher)],check=True)
    with tempfile.TemporaryDirectory(prefix='tianshu-final-launcher-',dir='/tmp') as tmp:
        d=Path(tmp);fake=d/'vllm';out=d/'argv.json'
        fake.write_text('#!/usr/bin/python3 -I\nimport json,os,sys,pathlib\npathlib.Path(os.environ["TIANSHU_FAKE_ARGV_OUT"]).write_text(json.dumps({"argv":["vllm"]+sys.argv[1:],"env":{k:v for k,v in os.environ.items() if k.startswith(("VLLM_PLUGINS","VLLM_FL_","FLAGGEMS_ILUVATAR_","TIANSHU_","PYTHONPATH"))}}))\n');fake.chmod(0o755)
        subprocess.run(['bash',str(launcher)],check=True,env=dict(os.environ,PATH=str(d)+':'+os.environ['PATH'],TIANSHU_FAKE_ARGV_OUT=str(out)))
        actual=json.loads(out.read_text());assert actual['argv']==formal['server_command']==qa['server_command']
        expected={'VLLM_PLUGINS':'fl','VLLM_FL_ILUVATAR_HYBRID_ATTENTION':'1','FLAGGEMS_ILUVATAR_COMPACT_VARLEN':'1','VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS':'1','VLLM_FL_USE_FLAGGEMS_ATTN':'0','VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN':'0','VLLM_FL_ILUVATAR_PREFILL_BUCKETS':'1','VLLM_FL_ILUVATAR_LOGITS_GATHER':'1','TIANSHU_DECODE_LAUNCH_PROBE':'1','TIANSHU_DECODE_COHERENT_PROBE':'1','TIANSHU_KV16_SERVICE_MODE':'candidate','TIANSHU_SEGMENT_POLICY_PROBE':'1','TIANSHU_CACHE_SERVICE_MODE':'candidate','TIANSHU_SQL_REFLECTION_PROBE':'1','TIANSHU_EXTENDED_BUCKET_PROBE':'1'}
        recorded_formal = dict(formal['flags'], **formal['environment_flags'])
        for k,v in expected.items():assert actual['env'][k]==recorded_formal[k]==v,k
        recorded_keys=[k for k in expected if k.startswith(('VLLM_FL_','TIANSHU_'))]
        assert all(qa['environment_flags'][k]==expected[k] for k in recorded_keys)
        tree=ast.parse((args.math_root/'source_snapshot/run_math_eval.py').read_text())
        envdicts=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict' and any(k.arg=='VLLM_PLUGINS' for k in n.keywords)]
        assert len(envdicts)==1
        literals={k.arg:k.value.value for k in envdicts[0].keywords if isinstance(k.value,ast.Constant)}
        assert literals['VLLM_PLUGINS']=='fl' and literals['FLAGGEMS_ILUVATAR_COMPACT_VARLEN']=='1'
        assert Path(actual['env']['PYTHONPATH'])==launcher.parent/'hooks'
        base=launcher.parent.parent
        assert len(quality['segment_sources']) == 13
        for remote,digest in quality['segment_sources'].items():
            source = base / Path(remote).relative_to('/workspace/vllm-plugin-FL/experiments')
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        hook = launcher.parent/'hooks/sitecustomize.py'
        assert hashlib.sha256(hook.read_bytes()).hexdigest() == quality['candidate_hook']['sha256']
        assert hook == base / Path(quality['candidate_hook']['path']).relative_to('/workspace/vllm-plugin-FL/experiments')
    audit=dict(complete=True,launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),syntax_passed=True,actual_completed_math_and_formal_argv_equal=True,
               critical_formal_flags_checked=len(expected),actual_math_recorded_flags_verified=len(recorded_keys),math_plugin_and_compact_flags_verified_from_frozen_runner_AST=True,
               active_hook_path_and_sources13_verified=True,stub_interpreter_isolated_no_plugin_import=True,gpu_job_started=False,
               math_correct=quality['correct'],math_total=105,formal_total_tok_s=score['total_tok_s'],argv=actual['argv'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
