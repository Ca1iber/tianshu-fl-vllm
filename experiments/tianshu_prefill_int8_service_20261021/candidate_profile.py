# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_prefill_same_NN_decode_profile',BASE.parent/'tianshu_int8_nn_gateup_service_20261021/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate');assert len(profile)==62 and len(sources)==381
    profile['PYTHONPATH']=str(BASE/('hooks_'+mode));environment.update(profile)
    selected=read(BASE/'selected_plan.json');gate_path=Path(selected['component_gate']);gate=read(gate_path)
    assert gate['complete'] and gate['independently_CPU_rechecked'] and selected['component_gate_sha256']==digest(gate_path)
    paths=[*BASE.rglob('*.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/entrypoints/openai/completion/protocol.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/entrypoints/serve/render/serving.py'),BASE/'selected_plan.json',gate_path,gate_path.parent/'verify_gpu.py',*[Path(p) for p in gate['frozen_sources']]]
    sources.update({str(p):digest(p) for p in paths});frozen_check(sources)
    for path,sha in sources.items():
        target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
