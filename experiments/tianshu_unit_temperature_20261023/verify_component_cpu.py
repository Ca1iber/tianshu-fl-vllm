# SPDX-License-Identifier: Apache-2.0
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;root=Path('/workspace/benchmark-results/20261023_unit_temperature_component_v2_tianshu');p=root/'result.json';s=json.loads(p.read_text());assert s['phase']=='complete' and s['GPU_checks_passed'] and len(s['cases'])==8
for row in s['cases']:assert all(row[k] for k in ('full_temperature_output_bits_equal','tokens_equal','global_and_per_request_RNG_equal','input_unchanged'))
for name,h in s['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
record=json.loads(next((root/'unit_records').glob('route_*.json')).read_text());assert record['skipped']>0 and record['unit_control']>0 and record['fallback']>0;assert record['policy_SHA256']==s['sources'][str(BASE/'policy.py')]
gate=dict(complete=True,component_GPU_checks_passed=True,all_recorded_sources_current_SHA_verified=True,source_count=len(s['sources']),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),raw_directory=str(root),cases=s['cases'],routing=record,original_failed_probe_retained=True,adopted=False,model_quality_qualified=False,official_benchmark=False)
(BASE/'COMPONENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print('component CPU audit passed')
