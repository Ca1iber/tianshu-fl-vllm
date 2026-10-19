# SPDX-License-Identifier: Apache-2.0
"""Prove encoder identity and disjoint real byte addresses for both mixed formats."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from layout import MODES, geometry

base = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_mixed_planes_CPU_parent', base.parent / 'tianshu_soft8_engine_v2_20261019/source.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
expected = ast.parse(parent.HELPER).body[0]
actual = next(n for n in ast.parse((base / 'writer.py').read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == '_soft8_encode_bf16')
assert ast.dump(actual) == ast.dump(expected)
for f in base.glob('*.py'):
    ast.parse(f.read_text())
checked = 0
for mode in MODES:
    g = geometry(mode)
    for factor in (1, 2):
        visited = set()
        for block in range(3):
            for size, offset in ((g['key_bytes'], 0), (g['value_bytes'], g['value_offset'])):
                assert offset % size == 0
                for token in range(16):
                    for head in range(2):
                        for dim in range(128):
                            begin = block * factor * g['page_bytes'] + offset + ((token * 2 + head) * 128 + dim) * size
                            for byte in range(begin, begin + size):
                                assert byte not in visited and block * factor * g['page_bytes'] <= byte < (block * factor + 1) * g['page_bytes']
                                visited.add(byte)
                                checked += 1
        assert len(visited) == 3 * g['page_bytes']
sources = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(base.glob('*.py'))}
r = dict(complete=True, CPU_only=True, formats=list(MODES), padded_page_bytes=[geometry(m)['page_bytes'] for m in MODES],
    byte_addresses_checked=checked, no_overlapping_planes_or_pages=True,
    contiguous_and_strided_page_geometry=True, encoder_AST_identical_to_qualified_software8=True,
    sources=sources, GPU_not_run=True, adopted=False)
(base / 'CPU_GATE_32g.json').write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps(r))
