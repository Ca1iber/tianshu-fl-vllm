# SPDX-License-Identifier: Apache-2.0
import importlib.util,random
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_NN_silu_observer_parent',BASE.parent/'tianshu_prefill_int8_service_20261021/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
CASES=[('cold_A','NN-observer-cold-A-20261021'),('cold_B','NN-observer-cold-B-20261021')]
def payload(salt):
    rng=random.Random(20261021088)
    return dict(model='minicpm',prompt=[rng.randrange(512,120000) for _ in range(128)],max_tokens=16,temperature=1.,top_p=.95,ignore_eos=True,stream=False,seed=777,return_token_ids=True,cache_salt=salt)
def prepare(root):
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(sources)==407 and len(profile)==62 and len(command)==13
    binding=read(BASE.parent/'tianshu_prefill_int8_official_results_20261021/RUNTIME_BINDING_GATE_32g.json')
    assert sources==binding['frozen_sources'];original=dict(sources)
    profile['PYTHONPATH']=str(BASE/'hooks');environment.update(profile)
    paths=[*BASE.rglob('*.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/compilation/backends.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/compilation/piecewise_backend.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/compilation/passes/inductor_pass.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/config/compilation.py')]
    sources.update({str(p):digest(p) for p in paths});frozen_check(sources)
    assert all(sources[p]==h for p,h in original.items())
    for p,h in sources.items():
        dest=root/'source_snapshot'/Path(p).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes());assert digest(dest)==h
    return command,environment,profile,sources
