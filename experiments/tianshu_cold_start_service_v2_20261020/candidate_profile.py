# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_cold_existing_W8_profile',BASE.parent/'tianshu_dense_w8a8_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('native','cpu4','gpu4')
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(profile)==57 and len(sources)==311
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_COLD_START_MODE=mode,TIANSHU_COLD_START_RECORD_DIR=str(root/'cold_start_records'))
    assert len(profile)==59;environment.update(profile)
    gate=read(BASE.parent/'tianshu_w8_gpu_packing_20261020/GPU_GATE_32g.json');assert gate['complete'] and gate['all_43_payloads_match_fresh_Math_signature']
    sql=read(BASE.parent/'tianshu_sql_target_reflection_20261020/CPU_GATE_32g.json');assert sql['complete'] and sql['all_config_values_schema_and_primary_keys_identical']
    paths=[*BASE.rglob('*.py'),*[Path(p) for p in gate['frozen_sources']],BASE.parent/'tianshu_w8_gpu_packing_20261020/GPU_GATE_32g.json',
        BASE.parent/'tianshu_sql_target_reflection_20261020/candidate_sql.py',BASE.parent/'tianshu_sql_target_reflection_20261020/CPU_GATE_32g.json',
        Path('/usr/local/lib/python3.12/site-packages/flag_gems/utils/models/sql.py')]
    for path in paths:sources[str(path)]=digest(path)
    frozen_check(sources)
    for name,sha in sources.items():
        target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
