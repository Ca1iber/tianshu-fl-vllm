# SPDX-License-Identifier: Apache-2.0
"""Run the exact qualified 450-source profile with the original serve argv."""
import argparse,importlib.util,json,os
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
gate=json.loads((BASE.parent/'tianshu_nn_silu_fx_official_recovery_20261021/OFFICIAL_RECOVERY_GATE_32g.json').read_text())
assert gate['complete'] and gate['passes_public_4k_performance_gates'] and gate['all_original_runtime_Math_benchmark_and_C64_validators_executed_unchanged']
assert gate['fresh_official_Math8_correct']==101 and gate['fresh_C64_Math_correct']==103
spec=importlib.util.spec_from_file_location('_qualified_NN_SiLU_release',BASE.parent/'tianshu_nn_silu_fx_screen_20261021/candidate_profile.py')
profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results');root.mkdir(exist_ok=False)
command,environment,selection,sources=profile.prepare(root,'candidate')
quality=json.loads((BASE.parent/'tianshu_nn_silu_fx_c64_quality_20261021/C64_QUALITY_GATE_32g.json').read_text())
assert len(sources)==450 and sources==quality['model_frozen_sources']
assert len(selection)==62 and selection['TIANSHU_FUTURE_KV_MODE']=='candidate'
assert command==quality['server_command']
(root/'qualified_launch.json').write_text(json.dumps(dict(command=command,profile_environment=selection,frozen_sources=sources,
    prepared_utc=datetime.now(timezone.utc).isoformat(),prepare_only=args.prepare_only,qualification_gate=str(BASE.parent/'tianshu_nn_silu_fx_official_recovery_20261021/OFFICIAL_RECOVERY_GATE_32g.json')),indent=2)+'\n')
print(json.dumps(dict(command=command,runtime_sources=len(sources),profile_environment_keys=len(selection),prepare_only=args.prepare_only)),flush=True)
if not args.prepare_only:os.execvpe(command[0],command,environment)
