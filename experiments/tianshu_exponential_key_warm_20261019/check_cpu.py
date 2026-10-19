# SPDX-License-Identifier: Apache-2.0
"""Execute the installed LibEntry key method, independently of Torch or CUDA."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from warm import KEYS

p = argparse.ArgumentParser()
p.add_argument('--libentry-source', type=Path, required=True)
p.add_argument('--exponential-source', type=Path, required=True)
a = p.parse_args()
tree = ast.parse(a.libentry_source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LibEntry')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'key')
ns = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(a.libentry_source), 'exec'), ns)
key = lambda n, seed, offset: ns['key'](SimpleNamespace(divisibility=16), [], [n, seed, offset], [])
assert len(KEYS) == len({key(8355840, seed, offset) for seed, offset in KEYS}) == 4
for seed, offset in KEYS:
    assert key(130560, seed, offset) == key(8355840, seed, offset)
assert key(130560, 7, 2**31-4) == ('i32', 'i32', 'i32')
assert key(130560, 7, 2**31) == ('i32', 'i32', 'i64')
exp = a.exponential_source.read_text()
assert '@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])' in exp
base = Path(__file__).resolve().parent
for path in base.rglob('*.py'):
    ast.parse(path.read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
out = dict(complete=True, CPU_only=True, actual_LibEntry_key_AST_executed=True,
    four_scalar_type_entries_proved=True, N_within_i32_entry_alias_proved=True,
    offset_i32_i64_boundary_proved=True, GPU_not_run=True, adopted=False,
    native_sources_sha256={'libentry.py': digest(a.libentry_source), 'exponential_.py': digest(a.exponential_source)},
    sources={str(path.relative_to(base)): digest(path) for path in sorted(base.rglob('*.py'))})
(base/'CPU_GATE_32g.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out))
