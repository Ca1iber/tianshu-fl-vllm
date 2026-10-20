# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_future_combo_profile', BASE.parent / 'tianshu_memory_combo_service_20261020/candidate_profile.py')
parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
digest, read, frozen_check = parent.digest, parent.read, parent.frozen_check


def prepare(root, mode):
    assert mode in ('control', 'candidate')
    policy = BASE.parent / 'tianshu_future_kv_reservation_20261020'
    gate = read(policy / 'CPU_GATE_32g.json')
    assert gate['complete'] and gate['CPU_only']
    frozen_check(gate['sources'])
    command, environment, profile, sources = parent.prepare(root, 'candidate')
    profile.update(PYTHONPATH=str(BASE / 'hooks'), TIANSHU_FUTURE_KV_MODE=mode,
                   TIANSHU_FUTURE_KV_RECORD_DIR=str(root / 'future_kv_records'))
    assert len(profile) == 52
    assert profile['TIANSHU_SMALL_ARENA_MODE'] == profile['TIANSHU_SEQ_LIMIT_MODE'] == 'candidate'
    environment.update(profile)
    paths = [*BASE.rglob('*.py'), policy / 'policy.py', policy / 'check_cpu.py', policy / 'CPU_GATE_32g.json',
             *[Path(p) for p in gate['sources']]]
    for path in paths:
        sources[str(path)] = digest(path)
    frozen_check(sources)
    for name, sha in sources.items():
        target = root / 'source_snapshot' / Path(name).relative_to('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
        assert digest(target) == sha
    return command, environment, profile, sources
