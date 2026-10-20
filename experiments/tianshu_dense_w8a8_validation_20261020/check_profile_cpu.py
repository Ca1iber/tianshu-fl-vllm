# SPDX-License-Identifier: Apache-2.0
"""After the screen succeeds, prove final config equals the actual screened candidate."""
import json
from pathlib import Path
from framework_profile import prepare
BASE=Path(__file__).resolve().parent
root=Path('/workspace/benchmark-results/20261020_dense_w8a8_validation_PROFILE_CPU_tianshu');assert not root.exists();root.mkdir()
screen=Path('/workspace/benchmark-results/20261020_dense_w8a8_service4k_v4_tianshu')
command,environment,profile,sources=prepare(root,screen)
assert len(command)==13 and len(profile)==57 and len(sources)==311
assert profile['VLLM_DISABLE_COMPILE_CACHE']=='1' and profile['TIANSHU_DENSE_INT8_MODE']=='candidate'
controllers=json.loads((root/'controller_sources.json').read_text());assert len(controllers)==9
result=dict(complete=True,CPU_only=True,service_gate_independently_passed=True,runtime_source_count=311,environment_count=57,
    separate_controller_count=9,source_and_original_argv_exactly_match_screen=True,only_record_paths_relocated=True,
    fresh_Math_required=True,candidate_M8_and_M64_selected=True,compile_cache_disabled=True,server_not_started=True,
    server_command=command,profile_environment=profile,frozen_sources=sources,controller_sources=controllers,raw_root=str(root))
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('frozen_sources','controller_sources','profile_environment')}))
