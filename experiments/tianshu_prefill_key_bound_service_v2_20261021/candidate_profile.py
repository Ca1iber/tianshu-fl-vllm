# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_prefill_key_bound_qualified407',BASE.parent/'tianshu_prefill_int8_service_20261021/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
POLICY=BASE.parent/'tianshu_prefill_key_bound_policy_20261021'
HELPER=BASE.parent/'tianshu_prefill_key_bound_20261021'
COMPONENT=BASE.parent/'tianshu_prefill_key_bound_results_20261021/COMPONENT_GATE_32g.json'
NATIVE=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/backend.py')
def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate');assert len(profile)==62 and len(sources)==407
    gate=read(COMPONENT);assert gate['complete'] and gate['all24cases_every_active_bit_and_inactive_guard_passed'] and len(gate['cases'])==24
    frozen_check(gate['source_manifest'])
    cpu=read(POLICY/'CPU_GATE_32g.json');assert cpu['complete'] and cpu['actual_native_route_AST_executed'];frozen_check(cpu['sources'])
    profile['PYTHONPATH']=str(BASE/('hooks_'+mode));environment.update(profile)
    paths=[*BASE.rglob('*.py'),*BASE.glob('*.sh'),*POLICY.glob('*.py'),POLICY/'CPU_GATE_32g.json',*HELPER.glob('*.py'),HELPER/'CPU_GATE_32g.json',COMPONENT,NATIVE,Path('/usr/local/lib/python3.12/site-packages/vllm/envs.py')]
    sources.update(gate['source_manifest']);sources.update({str(p):digest(p) for p in paths});frozen_check(sources)
    for name,sha in sources.items():
        target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
