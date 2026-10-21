# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_static_compile_qualified407',BASE.parent/'tianshu_prefill_int8_service_20261021/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
NATIVE=[Path('/usr/local/lib/python3.12/site-packages/vllm')/p for p in ('engine/arg_utils.py','config/compilation.py','config/utils.py','compilation/piecewise_backend.py','compilation/backends.py','compilation/compiler_interface.py')]
def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate');assert len(profile)==62 and len(sources)==407
    profile['PYTHONPATH']=str(BASE/('hooks_'+mode));environment.update(profile)
    paths=[*BASE.rglob('*.py'),*BASE.glob('*.sh'),*NATIVE]
    sources.update({str(p):digest(p) for p in paths});frozen_check(sources)
    for name,sha in sources.items():
        target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
