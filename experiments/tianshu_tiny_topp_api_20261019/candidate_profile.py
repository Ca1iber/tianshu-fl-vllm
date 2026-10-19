# SPDX-License-Identifier: Apache-2.0
"""Compose the qualified serve profile and the small-batch API hook."""
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'tianshu_gemm_topp_geo_20261016'))
from validation_profile_new import prepare as prepare_base, digest, frozen_check, read


def prepare(root, mode):
    assert mode in ('control', 'candidate')
    gate = read(BASE / 'CPU_GATE_32g.json')
    assert gate['complete'] and gate['routing_boundaries'] == 23
    assert gate['actual_forward_native_contract_cases'] == 30
    assert gate['import_delegation_and_restoration_cases'] == 3
    assert digest(BASE / 'api_route.py') == gate['api_route_sha256']
    assert digest(BASE / 'check_cpu.py') == gate['source_sha256']
    assert digest(BASE / 'hooks/sitecustomize.py') == gate['hook_sha256']
    command, env, profile, sources = prepare_base(
        root, Path('/workspace/benchmark-results/20261016_gemm_topp_geo_service_tianshu'))
    assert len(profile) == 38
    profile.update(PYTHONPATH=str(BASE / 'hooks'),
                   TIANSHU_TINY_TOPP_CANDIDATE='1' if mode == 'candidate' else '0',
                   TIANSHU_TINY_TOPP_RECORD_DIR=str(root / 'tiny_topp_records'))
    assert len(profile) == 40
    env.update(profile)
    for path in (*BASE.glob('*.py'), BASE / 'hooks/sitecustomize.py', BASE / 'CPU_GATE_32g.json'):
        sources[str(path)] = digest(path)
    frozen_check(sources)
    for path, expected in sources.items():
        destination = root / 'source_snapshot' / Path(path).relative_to('/')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(path).read_bytes())
        assert digest(destination) == expected
    return command, env, profile, sources
