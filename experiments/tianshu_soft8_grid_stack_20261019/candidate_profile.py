# SPDX-License-Identifier: Apache-2.0
"""Hold the qualified byte KV profile fixed; vary only the measured Top-P route."""
import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOFT = BASE.parent / 'tianshu_soft8_engine_v2_20261019'
GRID = BASE.parent / 'tianshu_topp_grid_api_v4_20261019'
spec = importlib.util.spec_from_file_location('_qualified_soft8_stack_parent', SOFT / 'candidate_profile.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
digest, read, frozen_check = parent.digest, parent.read, parent.frozen_check


def prepare(root, mode):
    assert mode in ('control', 'candidate')
    prerequisite = read(BASE / 'PREREQUISITE_32g.json')
    for name in ('official', 'math'):
        path = Path(prerequisite[name + '_remote'])
        assert digest(path) == prerequisite[name + '_result_sha256']
        assert read(path)['phase'] == 'complete'
    quality = read(Path(prerequisite['math_remote']))
    assert quality['total'] == 105 and quality['accuracy'] > .95
    gate = read(BASE / 'GPU_GATE_32g.json')
    assert gate['complete'] and digest(Path(gate['TopP_API_gate_remote'])) == gate['TopP_API_gate_sha256']
    assert read(Path(gate['TopP_API_gate_remote']))['complete']
    frozen_check(gate['frozen_sources'])
    command, env, profile, sources = parent.prepare(root, 'candidate')
    expected = {k: v.replace(prerequisite['canonical_output_root'] + '/', str(root) + '/')
                for k, v in prerequisite['canonical_profile'].items()}
    assert command == prerequisite['canonical_serve_argv'] and profile == expected
    assert profile['TIANSHU_SOFT8_SERVICE_MODE'] == 'candidate' and len(profile) == 40
    profile.update(PYTHONPATH=str(BASE / 'hooks'),
        TIANSHU_TOPP_GRID_CANDIDATE='1' if mode == 'candidate' else '0',
        TIANSHU_TOPP_GRID_RECORD_DIR=str(root / 'grid_topp_records'))
    assert len(profile) == 42
    env.update(profile)
    for path, sha in gate['frozen_sources'].items():
        assert path not in sources or sources[path] == sha
        sources[path] = sha
    for name in ('candidate_profile.py', 'audit_profile.py', 'run_service_trial.py',
                 'run_service_pair.py', 'hooks/sitecustomize.py', 'PREREQUISITE_32g.json',
                 'GPU_GATE_32g.json', 'COMPOSITION_CPU_GATE_32g.json', 'check_composition.py'):
        sources[str(BASE / name)] = digest(BASE / name)
    frozen_check(sources)
    for name, sha in sources.items():
        target = root / 'source_snapshot' / Path(name).relative_to('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
        assert digest(target) == sha
    return command, env, profile, sources
