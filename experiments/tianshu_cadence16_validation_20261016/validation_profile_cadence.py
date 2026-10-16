# SPDX-License-Identifier: Apache-2.0
"""Replay a qualifying cadence16 screen; preserve the proven GEO operator base."""
import importlib.util
from pathlib import Path
import statistics
import sys
EVALSCOPE_SOURCES={
    'arguments.py':'daedad68adce7e2d047d0ca720ec1424dd8a5e3687e2e3ad5925a9fd4ecfbfa9',
    'config.py':'ecd0ed8be4485213ce67eaf0a6047d72b4a669d17b51014cc32eef425b1640e9',
    'evaluator/evaluator.py':'a7c1934ea24effe91ef1622fdf4df823055b76f5dcd246356cebc4a29236081a',
    'api/evaluator/cache.py':'f0d127bc00898aba93efe404f1574bcc32ab04757824984425d24eb9346184d3',
}
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_gemm_topp_geo_20261016'))
from validation_profile_new import BENCH_SHA,audit_math,audit_benchmark,csv_rows,digest,frozen_check,normalized_environment,prepare as geo_prepare,read
spec=importlib.util.spec_from_file_location('_cadence16_validation_audit',BASE.parent/'tianshu_prefill_cadence16_20261016/audit_profile.py')
_audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(_audit)
audit_runtime=_audit.audit_runtime


def check_screen(confirm,control,candidate):
    assert confirm['complete'] and not confirm['official_benchmark']
    assert confirm['requests_per_round']==128 and confirm['rounds_per_mode']==3 and confirm['excluded_rounds']==[1]
    assert control['phase']==candidate['phase']=='complete'
    assert control['frozen_sources']==candidate['frozen_sources']==confirm['frozen_sources']
    values={}
    for mode,state in (('control',control),('candidate',candidate)):
        rows=state['short_4k_rounds'];assert len(rows)==3
        for i,row in enumerate(rows):
            assert row['round']==i+1 and row['actual_queue_replenishment_verified']
            assert row['successful_requests']==len(row['responses'])==128
            assert abs(row['total_tokens_per_second']-128*5120/row['duration_seconds'])<1e-9
            assert [r['input_seed'] for r in row['responses']]==[2026102700+1000*i+j for j in range(128)]
            assert all(r['usage']['prompt_tokens']==4096 and r['usage']['completion_tokens']==1024 and r['usage']['total_tokens']==5120 for r in row['responses'])
        values[mode]=[r['total_tokens_per_second'] for r in rows[1:]]
        mean=statistics.mean(values[mode]);assert abs(state['mean_total_tok_s']-mean)<1e-9
        assert 100*(max(values[mode])-min(values[mode]))/mean<.5
    gain=100*(statistics.mean(values['candidate'])/statistics.mean(values['control'])-1)
    assert abs(gain-confirm['gain_percent'])<1e-12
    assert gain>=1., 'Need a clear >=1% N128 improvement before expensive new quality/formal validation'
    assert min(values['candidate'])>max(values['control'])
    assert candidate['profile_environment']['TIANSHU_PREFILL_CADENCE_MODE']=='candidate'
    assert candidate['frozen_sources']['/workspace/vllm-plugin-FL/experiments/tianshu_prefill_cadence16_20261016/policy.py']=='392802e60c222f9fe166f4c8755cd39b783a98fc8075ecd9ae8b8f6513c1e6fc'
    return dict(complete=True,N128_min_gain_percent=1.,gain_percent=gain)


def prepare(root,screen_root):
    confirm=read(screen_root/'comparison.json');control=read(screen_root/'control/result.json');candidate=read(screen_root/'candidate/result.json')
    gate=check_screen(confirm,control,candidate);frozen_check(confirm['frozen_sources'])
    cpu=read(BASE/'CPU_GATE_32g.json');assert cpu['complete'] and cpu['historical_parser_replay_only']
    for name,sha in cpu['owned_source_SHA256'].items():assert digest(BASE/name)==sha,name
    audit_runtime(screen_root/'candidate',candidate,official=True)
    command,environment,base_profile,base_sources=geo_prepare(root,Path('/workspace/benchmark-results/20261016_gemm_topp_geo_service_tianshu'))
    assert command==candidate['server_command']==control['server_command']
    for path,sha in base_sources.items():assert confirm['frozen_sources'][path]==sha,path
    profile={k:v.replace(candidate['output_root']+'/',str(root)+'/') for k,v in candidate['profile_environment'].items()}
    assert len(profile)==40 and profile['TIANSHU_PREFILL_CADENCE_MODE']=='candidate'
    assert profile['PYTHONPATH']==str(BASE.parent/'tianshu_prefill_cadence16_20261016/hooks')
    environment.update(profile);sources=dict(confirm['frozen_sources'])
    for path in (Path(__file__),BASE/'run_validation.py',BASE/'canary.py',BASE/'check_cpu.py',BASE/'CPU_GATE_32g.json'):sources[str(path)]=digest(path)
    package=Path('/usr/local/lib/python3.12/site-packages/evalscope')
    for name,sha in EVALSCOPE_SOURCES.items():
        path=package/name;assert digest(path)==sha;sources[str(path)]=sha
    sources[str(package/'run.py')]=digest(package/'run.py')
    assert len(sources)==80
    frozen_check(sources)
    for path,sha in sources.items():
        target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert digest(target)==sha
    (root/'screen_qualification.json').write_text(__import__('json').dumps(gate,indent=2)+'\n')
    return command,environment,profile,sources
