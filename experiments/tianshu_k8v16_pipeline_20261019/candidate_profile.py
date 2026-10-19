# SPDX-License-Identifier: Apache-2.0
"""Bind a new real-engine test to the complete K8V16 attention component."""
import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parent
PARENT = BASE.parent / 'tianshu_soft8_engine_v2_20261019'
spec = importlib.util.spec_from_file_location('_k8v16_canonical_byte_profile', PARENT / 'candidate_profile.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
digest, read, frozen_check = parent.digest, parent.read, parent.frozen_check


def prepare(root, mode):
    assert mode in ('control', 'candidate')
    cpu = read(BASE / 'ENGINE_CPU_GATE_32g.json')
    assert cpu['complete'] and cpu['actual_Worker_loader_modes'] == 2
    for name, sha in cpu['sources'].items():
        assert digest(BASE / name) == sha, name
    gate = read(BASE / 'GPU_GATE_32g.json')
    assert gate['complete'] and gate['actual_builder_writer_forward_cases'] == 21
    result = Path(gate['remote_result'])
    assert digest(result) == gate['remote_result_sha256'] and read(result)['complete']
    frozen_check(gate['frozen_sources'])
    command, env, profile, sources = parent.prepare(root, 'candidate')
    assert len(profile) == 40 and profile['TIANSHU_SOFT8_SERVICE_MODE'] == 'candidate'
    profile.update(PYTHONPATH=str(BASE / 'hooks'), TIANSHU_K8V16_SERVICE_MODE=mode)
    assert len(profile) == 41
    env.update(profile)
    for path, sha in gate['frozen_sources'].items():
        assert path not in sources or sources[path] == sha, path
        sources[path] = sha
    for name in ('candidate_profile.py', 'engine_hooks.py', 'ENGINE_BINDINGS_32g.json',
                 'hooks/sitecustomize.py', 'run_service_trial.py', 'run_service_pair.py',
                 'audit_profile.py', 'GPU_GATE_32g.json', 'ENGINE_CPU_GATE_32g.json',
                 'check_engine_selection_cpu.py'):
        sources[str(BASE / name)] = digest(BASE / name)
    sources.update(read(BASE / 'ENGINE_BINDINGS_32g.json'))
    frozen_check(sources)
    for path, sha in sources.items():
        dest = root / 'source_snapshot' / Path(path).relative_to('/')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(path).read_bytes())
        assert digest(dest) == sha
    return command, env, profile, sources
