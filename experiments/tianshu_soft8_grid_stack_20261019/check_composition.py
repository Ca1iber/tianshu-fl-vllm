# SPDX-License-Identifier: Apache-2.0
"""Check single-base hook composition and bind immutable pre-existing evidence."""
import ast
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent
repo = base.parents[1]
grid = base.parent / 'tianshu_topp_grid_api_v4_20261019'
expected = (grid / 'hooks/sitecustomize.py').read_text().replace(
    'FOLDER = Path(__file__).resolve().parents[1]',
    "FOLDER = Path(__file__).resolve().parents[2] / 'tianshu_topp_grid_api_v4_20261019'").replace(
    'tianshu_gemm_topp_geo_20261016/hooks/sitecustomize.py',
    'tianshu_soft8_engine_v2_20261019/hooks/sitecustomize.py')
assert (base / 'hooks/sitecustomize.py').read_text() == expected
for path in base.rglob('*.py'):
    ast.parse(path.read_text())
gate = json.loads((base / 'GPU_GATE_32g.json').read_text())
for source, sha in gate['frozen_sources'].items():
    if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
        local = repo / Path(source).relative_to('/workspace/vllm-plugin-FL')
        assert hashlib.sha256(local.read_bytes()).hexdigest() == sha
actual = json.loads((base / 'PREREQUISITE_32g.json').read_text())
assert actual['canonical_profile']['TIANSHU_SOFT8_SERVICE_MODE'] == 'candidate'
assert len(actual['canonical_profile']) == 40
assert gate['TopP_row_contract'] == 'EVERY_ROW: retained_unchanged AND (exact_current_baseline OR strict_CPU_1e-6)'
sources = {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(base.rglob('*.py'))}
record = dict(complete=True, CPU_only=True, single_qualified_base_hook=True,
    grid_loader_only_two_path_composition_edits=True,
    canonical_software8_profile_keys=40, combined_environment_keys=42,
    prior_frozen_sources=len(gate['frozen_sources']), sources=sources,
    GPU_combination_not_run=True, adopted=False)
(base / 'COMPOSITION_CPU_GATE_32g.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
