# SPDX-License-Identifier: Apache-2.0
"""Audit actual shape/config records, cold/warm logs and original sampler sequences."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();d=json.loads((r/'result.json').read_text())
assert d['phase']=='complete' and d['complete'] and not d['adopted']
for n,h in d['source_snapshots'].items():assert hashlib.sha256((r/'source_snapshot'/Path(n).relative_to('/')).read_bytes()).hexdigest()==h,n
assert d['shared_FlagGems_tuning_DB_not_used'] and d['isolated_FlagGems_config_and_code_cache'] and d['existing_compiler_cache_not_cleared']
assert d['observed_libcache_db_url']=='sqlite:///'+d['isolated_cold_tuning_database']
assert d['observed_FlagGems_cache_dir']==d['isolated_cache']
assert d['cold_warm_rng_before']==d['cold_warm_rng_after']
assert d['real_failure_rng']['before']==d['real_failure_rng']['after'] and d['real_failure_rng']['original_exception_identity_preserved']
assert d['postwarm_batch_order']==list(range(1,65,2))+list(range(64,0,-2))
records=d['producer_records'];cold=[x for x in records if x['phase']=='cold_warm'];post=[x for x in records if x['phase']=='postwarm']
assert len(cold)==len(post)==64 and {x['batch'] for x in cold}==set(range(1,65))
assert [x['batch'] for x in post]==d['postwarm_batch_order']
for x in records:
    assert x['N']==x['batch']*130560 and 1<=x['batch']<=64
    assert x['BLOCK'] in (64,128,256,512) and x['warps'] in (4,8) and x['stages'] in (2,3)
    assert isinstance(x['kernel_hash'],str) and len(x['kernel_hash'])>=32
assert len(d['warm_calls'])==34
for w in d['warm_calls']:
    assert w['complete'] and w['maximum_noise_bytes']==33423360
    assert [x['batch'] for x in w['shapes']]==list(range(64,0,-1))
assert len(d['sampling_cases'])==33
assert {(x['batch'],x['generator_mode']) for x in d['sampling_cases']}=={(n,m) for n in (1,2,3,7,8,16,24,32,62,63,64) for m in ('global','mixed','all_seeded')}
tokens=0
for x in d['sampling_cases']:
    assert len(x['actual_tokens'])==x['batch'] and x['expected_tokens']==x['actual_tokens']
    assert x['after_warm_default']==x['initial_default'] and x['after_warm_per_request']==x['initial_per_request']
    assert x['actual_final_default']==x['expected_final_default']
    assert x['actual_final_per_request']==x['expected_final_per_request']
    assert x['warm_preserved_default_and_request_generators']
    assert set(x['actual_final_per_request'])=={str(i) for i in x['seeded_rows']}
    tokens+=x['batch']
log=(r.parent/'probe.log').read_text(errors='replace')
assert log.count('COLD_WARM_BEGIN')==log.count('COLD_WARM_END')==log.count('POSTWARM_BEGIN')==log.count('POSTWARM_END')==1
coldlog=log.split('COLD_WARM_BEGIN',1)[1].split('COLD_WARM_END',1)[0]
postlog=log.split('POSTWARM_BEGIN',1)[1].split('POSTWARM_END',1)[0]
summary=[line for line in coldlog.splitlines() if 'Triton autotuning for function fused_exponential_kernel_f32_unroll8 finished' in line]
keys=[int(re.search(r'config key: \((\d+),',line).group(1)) for line in summary]
assert len(keys)==64 and set(keys)=={n*130560 for n in range(1,65)}
assert not any('fused_exponential_kernel_f32_unroll8' in line and ('Autotuning kernel' in line or 'Triton autotuning for function' in line) for line in postlog.splitlines())
out=dict(complete=True,component_only=True,all64_cold_config_keys_tuned_during_warm=True,all64_postwarm_shapes_zero_new_exponential_tuning=True,actual_compiled_configurations_checked=True,sampling_cases=33,exact_sample_tokens=tokens,default_CPU_CUDA_and_each_request_rng_states_checked=True,sources=len(d['source_snapshots']),cold_warm_seconds=d['warm_calls'][0]['elapsed_seconds'],shared_FlagGems_tuning_DB_not_used=True,new_service_and_Math_not_run=True,adopted=False)
(r/'local_download_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
