# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,importlib.util,json,tempfile
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/'tianshu_prefill_int8_service_20261021'
SERVICE=BASE.parent/'tianshu_qkv_prefill_int8_service_v2_20261021'
spec=importlib.util.spec_from_file_location('_qkv_exact_407_profile',ROOT/'candidate_profile.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
for p in BASE.glob('*.py'):ast.parse(p.read_text())
root=Path(tempfile.mkdtemp(prefix='20261021_qkv_exact407_CPU_',dir='/workspace/benchmark-results'))
command,environment,profile,sources=parent.prepare(root,'candidate')
assert len(sources)==407 and len(profile)==62 and len(command)==13
assert profile['PYTHONPATH']==str(ROOT/'hooks_candidate')
current=parent.read(SERVICE/'CPU_GATE_32g.json');assert current['complete'] and current['runtime_source_count']==432
assert all(current['profiles'][1]['sources'].get(p)==h for p,h in sources.items())
gate=dict(complete=True,CPU_only=True,no_GPU_started=True,actual_reference_runtime_sources=sources,runtime_source_count=407,candidate_runtime_sources=432,
    all_407_sources_exact_subset_of_candidate=True,original_unmodified_controller=str(ROOT/'run_service_trial.py'),controller_sha256=parent.digest(ROOT/'run_service_trial.py'),
    original_13_command=command,profile=profile,CPU_profile_root=str(root))
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('actual_reference_runtime_sources','profile')}))
