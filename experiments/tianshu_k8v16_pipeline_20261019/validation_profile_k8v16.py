# SPDX-License-Identifier: Apache-2.0
"""Fresh Math/formal validation bound to the measured byte-KV service profile."""
import importlib.util
from pathlib import Path
from candidate_profile import prepare as prepare_screen,digest,frozen_check,read
from validation_profile import BENCH_SHA,audit_math,audit_benchmark,csv_rows,normalized_environment
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_soft8_validation_audit',BASE/'audit_profile.py')
_audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(_audit)
audit_runtime=_audit.audit_runtime

def prepare(root,screen_root):
    confirm=read(screen_root/'comparison.json')
    control=read(screen_root/'control/result.json');candidate=read(screen_root/'candidate/result.json')
    assert confirm['complete'] and confirm['gain_percent']>=2
    assert control['phase']==candidate['phase']=='complete' and confirm['rounds_per_mode']==3 and confirm['excluded_rounds']==[1]
    assert min(r['total_tokens_per_second'] for r in candidate['short_4k_rounds'][1:])>max(r['total_tokens_per_second'] for r in control['short_4k_rounds'][1:])
    assert control['spread_percent']<1 and candidate['spread_percent']<1
    assert candidate['frozen_sources']==control['frozen_sources']==confirm['frozen_sources']
    frozen_check(confirm['frozen_sources'])
    command,environment,profile,sources=prepare_screen(root,'candidate')
    assert sources==confirm['frozen_sources'] and command==candidate['server_command']==control['server_command']
    expected={k:v.replace(candidate['output_root']+'/',str(root)+'/') for k,v in candidate['profile_environment'].items()}
    assert profile==expected and len(profile)==41 and profile['TIANSHU_SOFT8_SERVICE_MODE']=='candidate' and profile['TIANSHU_K8V16_SERVICE_MODE']=='candidate'
    for path in (Path(__file__),BASE/'run_validation.py'):sources[str(path)]=digest(path)
    frozen_check(sources)
    for path,h in sources.items():
        dest=root/'source_snapshot'/Path(path).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(Path(path).read_bytes());assert digest(dest)==h
    return command,environment,profile,sources
