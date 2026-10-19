# SPDX-License-Identifier: Apache-2.0
"""Exercise actual argv composition and bind pre-run sources without a GPU."""
import ast,hashlib,json,importlib.util
from pathlib import Path
b=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_budget_CPU_profile',b/'candidate_profile.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
c=['python3','-m','vllm.entrypoints.cli.main','serve','/workspace/MiniCPM5-2B','--port','9031','--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}','--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072'];original=list(c)
assert m.budget_command(c,'control')==c and m.budget_command(c,'control') is not c
assert m.budget_command(c,'candidate')==c+['--max-num-batched-tokens','4096'] and c==original
for bad in ('unknown',None):
 try:m.budget_command(c,bad)
 except AssertionError:pass
 else:raise AssertionError('bad mode accepted')
try:m.budget_command(c+['--max-num-batched-tokens','2048'],'candidate')
except AssertionError:pass
else:raise AssertionError('duplicate flag accepted')
for p in b.glob('*.py'):ast.parse(p.read_text())
r=dict(complete=True,CPU_only=True,actual_argv_composition_checked=True,unchanged_input_argv_and_duplicate_flag_guards=True,both_arms_force_qualified_K8V16=True,candidate_explicit_budget_cli_only=True,sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(b.glob('*.py'))},GPU_not_run=True,performance_and_Math_not_measured=True,adopted=False)
(b/'CPU_GATE_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
