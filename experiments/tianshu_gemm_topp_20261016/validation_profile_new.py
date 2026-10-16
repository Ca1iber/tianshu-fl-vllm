# SPDX-License-Identifier: Apache-2.0
"""Replay the confirmed candidate for fresh Math and official4K; no scheduler edits."""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tianshu_original_command_20261016'))
from validation_profile import BENCH_SHA,audit_math,audit_benchmark,csv_rows,digest,frozen_check,normalized_environment,prepare as prepare_original,read
from audit_profile import audit_runtime


def prepare(root,confirm_root):
    confirm=read(confirm_root/'comparison.json')
    control=read(confirm_root/'control/result.json');candidate=read(confirm_root/'candidate/result.json')
    assert confirm['complete'] and confirm['gain_percent']>=2
    assert control['phase']==candidate['phase']=='complete'
    assert confirm['rounds_per_mode']==5 and confirm['excluded_rounds']==[1,2]
    assert min(r['total_tokens_per_second'] for r in candidate['short_4k_rounds'][2:])>max(r['total_tokens_per_second'] for r in control['short_4k_rounds'][2:])
    assert candidate['frozen_sources']==control['frozen_sources']==confirm['frozen_sources']
    frozen_check(confirm['frozen_sources'])
    command,environment,base_profile,base_sources=prepare_original(root,Path('/workspace/benchmark-results/20261016_combined_screen_service_tianshu'))
    assert command==candidate['server_command']==control['server_command']
    for path,sha in base_sources.items():assert confirm['frozen_sources'][path]==sha,path
    profile={k:v.replace(candidate['output_root']+'/',str(root)+'/') for k,v in candidate['profile_environment'].items()}
    assert len(profile)==38 and profile['TIANSHU_TOPP_PRECISION_SERVICE_MODE']=='candidate'
    assert profile['TIANSHU_PREFILL_GEMM_TILES_PROBE']=='1' and profile['TIANSHU_MAX_SEQS_MODE']=='control'
    environment.update(profile);sources=dict(confirm['frozen_sources'])
    for path in (Path(__file__),Path(__file__).with_name('run_validation.py')):sources[str(path)]=digest(path)
    frozen_check(sources)
    for path,sha in sources.items():
        target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
