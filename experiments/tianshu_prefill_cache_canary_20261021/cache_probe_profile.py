# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,random
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_cache_canary_exact_prefill407',BASE.parent/'tianshu_prefill_int8_service_20261021/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
CASES=[('unsalted_first',None),('unsalted_repeat',None),('A_cold','canary-20261021-A'),('A_cached','canary-20261021-A'),('B_cold','canary-20261021-B'),('B_cached','canary-20261021-B'),('C_cold','canary-20261021-C'),('C_cached','canary-20261021-C')]
def payload(salt):
    rng=random.Random(20261021088)
    data=dict(model='minicpm',prompt=[rng.randrange(512,120000) for _ in range(128)],max_tokens=16,temperature=1.,top_p=.95,ignore_eos=True,stream=False,seed=777,return_token_ids=True)
    if salt is not None:data['cache_salt']=salt
    return data

def prepare(root):
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(sources)==407 and len(profile)==62 and len(command)==13
    binding=read(BASE.parent/'tianshu_prefill_int8_official_results_20261021/RUNTIME_BINDING_GATE_32g.json')
    assert sources==binding['frozen_sources']
    return command,environment,profile,sources
