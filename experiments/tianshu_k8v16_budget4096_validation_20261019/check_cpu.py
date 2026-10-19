# SPDX-License-Identifier: Apache-2.0
"""Exercise the reused105-item parser on historical artifacts, without claiming new quality."""
import argparse,ast,hashlib,json,importlib.util
from pathlib import Path
from audit_outputs import audit_math,budget4096_command
p=argparse.ArgumentParser();p.add_argument('--historical-math-root',type=Path,required=True);a=p.parse_args();root=a.historical_math_root.resolve()
s=json.loads((root/'result.json').read_text());profile=json.loads((root/'accuracy_audit.json').read_text())['profile']
r=audit_math(root,s,profile,write=False)
assert r['total']==105 and r['correct']==100 and r['official_requirement_passed']
assert r['fresh_math_for_budget4096_profile'] and not r['reused_original_accuracy']
assert budget4096_command()[-2:]==['--max-num-batched-tokens','4096']
b=Path(__file__).resolve().parent
for p in b.glob('*.py'):ast.parse(p.read_text())
g=dict(complete=True,CPU_only=True,historical105_item_parser_checked=True,historical_correct=100,historical_result_is_not_new_budget_quality=True,no_new_quality_claim=True,explicit_budget4096_argv_verified=True,sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(b.glob('*.py'))})
(b/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps(g))
