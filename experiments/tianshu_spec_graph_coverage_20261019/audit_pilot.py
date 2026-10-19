# SPDX-License-Identifier: Apache-2.0
"""Audit the diagnostic ngram stack without borrowing baseline quality scores."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def audit_runtime(root,state,official=False,online=True):
    def read(p):return json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    base=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('_ngram_original_profile',base/'tianshu_original_command_20261016/validation_profile.py')
    profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
    command=state['server_command'];assert command[:-2]==profile.official_command()
    assert command[-2]=='--speculative-config'
    expected=dict(method='ngram',num_speculative_tokens=1,prompt_lookup_min=3,prompt_lookup_max=5)
    assert json.loads(command[-1])==expected==state['speculative_config']
    for name,h in state['frozen_sources'].items():
        assert sha(root/'source_snapshot'/Path(name).relative_to('/'))==h
        if online:assert sha(Path(name))==h
    observed=read(root/'observed_server_environment.json')
    assert observed['environment']==state['profile_environment'] and len(observed['environment'])==42
    assert observed['command']==command
    records=root/'ngram_records'
    models=[read(p) for p in records.glob('model_*.json')];assert models
    for model in models:
        assert all(model[k]==v for k,v in expected.items())
        assert model['prefill75_enabled'] is False and model['hybrid_installed'] is False
        assert model['small_attention_installed_before_profile_and_capture']
        assert model['native_CPU_proposer_source_sha256']=='5051e1bca645fe4befec0a6340f0dfc169fbcbe070ac1da508b72f8ee9c91736'
    rejected=[read(p) for p in records.glob('rejection_*.json')];assert rejected
    assert all(not r['use_fp64_gumbel'] and not r['synthetic_mode'] and r['standard_acceptance_ratio_unchanged'] for r in rejected)
    threshold=[read(p) for p in (records/'rejection_threshold').glob('route_*.json')];assert threshold
    assert all(r['dtype']=='torch.float32' and 0<r['minimum']<=r['maximum']<=1 and not r['synthetic_acceptance_enabled'] for r in threshold)
    routefiles=[read(p) for p in records.glob('verification_routes_*.json')];assert routefiles
    routes=[r for f in routefiles for r in f['routes']]
    assert any(r['route']=='L0_2D' for r in routes)
    assert any(r['route']=='Q2_split' for r in routes),'No actual split verification route; do not infer exercised optimization'
    render=importlib.util.spec_from_file_location('_audit_ngram_verify',base/'tianshu_ngram_framework_20261016/native_verify.py')
    verifier=importlib.util.module_from_spec(render);render.loader.exec_module(verifier)
    native=root/'source_snapshot/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py'
    text,derivation=verifier.render(native.read_bytes())
    assert (records/'split_generated/generated_split_unified.py').read_text()==text
    assert all(r['derivation']==derivation and r['scratch_capacity_tokens']==128 for r in routes if r['route']=='Q2_split')
    config=[read(p) for p in (root/'bucket_trace_records').glob('configured_*.json')]
    assert config and all(r['enabled'] is False for r in config)
    metrics=list(root.glob('*metrics.txt'));assert len(metrics)==6
    for p in metrics:
        values=re.findall(r'^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$',p.read_text(),re.M)
        assert values and all(float(v)==0 for v in values)
    assert 'No running processes found' in (root/'hardware_after.txt').read_text()
    fixed=[read(p) for p in (records/'fixed_index').glob('route_*.json')]
    assert fixed and all(r['installed_before_profile_and_capture'] and r['python_launch_calls']>0 for r in fixed)
    assert all(r['source_sha256']=='f48351f7ad1251acb680927d742123f92ea4f3e23505d83dfde6a3b7c713cc06' for r in fixed)
    assert {2048,130560}<={v['width'] for r in fixed for v in r['routes'].values()}
    from coverage_audit import audit_coverage
    coverage=audit_coverage(root,state)
    return dict(complete=True,coverage=coverage,native_ngram=True,positive_FP32_rejection=True,synthetic_acceptance=False,
        explicit_extra_speculative_CLI_verified=True,original_main_serve_arguments_preserved=True,
        actual_Q2_split_and_L0_prefill_routes_verified=True,all_frozen_sources_and42_environment_verified=True,
        prefill75_and_hybrid_explicitly_disabled_by_spec_guard=True,zero_preemptions=True,
        fresh_Math_not_run=True,official_benchmark=False,adopted=False)
