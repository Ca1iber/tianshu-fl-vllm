# SPDX-License-Identifier: Apache-2.0
"""Check copied request contract and unchanged parser on historical105 only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import textwrap
from validation_profile_cadence import audit_math


def main():
    p=argparse.ArgumentParser();p.add_argument('--historical-math-root',type=Path,required=True);args=p.parse_args()
    base=Path(__file__).resolve().parent
    source=(base.parent/'tianshu_prefill_cadence16_20261016/run_service_trial.py').read_text();new=(base/'canary.py').read_text()
    main_node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    old_request=next(n for n in main_node.body if isinstance(n,ast.FunctionDef) and n.name=='request')
    new_request=next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name=='request')
    assert ast.dump(old_request)==ast.dump(new_request)
    a=source.index("        state['phase']='seeded_canaries';save()");b=source.index("        metrics('before_metrics.txt')",a)
    body=textwrap.dedent(source[a:b]).replace('args.mode',"state['mode']")
    expected="def canary(root,state,save):\n    state['seeded_canaries']=[]\n"+textwrap.indent(body,'    ')
    actual=next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name=='canary')
    assert ast.dump(ast.parse(expected).body[0])==ast.dump(actual)
    state=json.loads((args.historical_math_root/'result.json').read_text());assert state['phase']=='complete'
    parsed=audit_math(args.historical_math_root,state,{},write=False)
    assert parsed['total']==105 and parsed['correct']==103 and parsed['accuracy']==103/105
    dig=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    record=dict(complete=True,request_AST_identical_to_real_screen=True,
        canary_body_identical_except_mode_source_and_list_init=True,historical_parser_replay_only=True,
        historical_requested105_succeeded105_zero_errors=True,historical_correct=103,
        new_profile_GPU_Math_and_formal_not_run=True,new_accuracy_not_claimed=True,
        owned_source_SHA256={name:dig(base/name) for name in ('run_validation.py','validation_profile_cadence.py','canary.py','check_cpu.py')},
        historical_raw_SHA256=parsed['raw_files'])
    print(json.dumps(record))


if __name__=='__main__':main()
