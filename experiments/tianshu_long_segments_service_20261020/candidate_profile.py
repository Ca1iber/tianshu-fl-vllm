# SPDX-License-Identifier: Apache-2.0
import hashlib,json,os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_original_command_20261016'))
from validation_profile import digest,read,frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete']
    for path,expected in gate['sources'].items():assert digest(path)==expected,path
    template=read(BASE.parent/'tianshu_k8v16_budget4096_release_20261019/serve_profile_template.json')
    old_root,old_repo=template['old_output_root'],template['old_repository_root']
    repo=str(BASE.parents[1])
    profile={key:value.replace(old_root+'/',str(root)+'/').replace(old_repo+'/',repo+'/') for key,value in template['environment'].items()}
    assert len(profile)==43 and profile['TIANSHU_BUDGET_DEFAULT_MODE']=='candidate'
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_LONG_SEGMENT_MODE=mode,
                   TIANSHU_LONG_SEGMENT_RECORD_DIR=str(root/'long_segments_records'),
                   TIANSHU_GRAPH_SEGMENTS_RECORD_DIR=str(root/'graph_segments_records'))
    assert len(profile)==46
    sources={name.replace(old_repo+'/',repo+'/'):sha for name,sha in template['frozen_sources'].items()}
    extra=[*BASE.rglob('*.py'),BASE/'batch_policy.json',BASE/'CPU_GATE_32g.json',
           BASE.parent/'tianshu_k8v16_budget4096_release_20261019/serve_profile_template.json',
           BASE.parent/'tianshu_long_segments_20261020/source.py',
           BASE.parent/'tianshu_long_segments_20261020/DECODE_GATE_32g.json',
           BASE.parent/'tianshu_k8v16_graph_segments_v3_20261019/trace.py',
           Path('/usr/local/lib/python3.12/site-packages/vllm_fl/compilation/graph.py')]
    for path in extra:sources[str(path)]=digest(path)
    frozen_check(sources)
    for name,sha in sources.items():
        dest=root/'source_snapshot'/Path(name).relative_to('/')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
        assert digest(dest)==sha
    environment=dict(os.environ)
    for key in list(environment):
        if key.startswith(('TIANSHU_','VLLM_FL_','FLAGGEMS_ILUVATAR_')) or key in ('VLLM_PLUGINS','PYTHONPATH','PYTHONUNBUFFERED','TRITON_PRINT_AUTOTUNING','CUDA_VISIBLE_DEVICES'):
            del environment[key]
    environment.update(profile)
    return template['server_command'],environment,profile,sources
