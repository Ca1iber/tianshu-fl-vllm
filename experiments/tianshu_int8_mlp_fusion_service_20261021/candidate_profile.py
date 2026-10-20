# SPDX-License-Identifier: Apache-2.0
"""Original qualified profile with one independent MLP fusion flag."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_int8_MLP_qualified_parent',BASE.parent/'tianshu_lm_precision_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(profile)==62 and len(sources)==359
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_MLP_FUSION_MODE=mode);assert len(profile)==63;environment.update(profile)
    component=BASE.parent/'tianshu_int8_gateup_silu_20261021'
    installed=Path('/usr/local/lib/python3.12/site-packages')
    paths=[*BASE.rglob('*.py'),component/'kernel.py',component/'GPU_GATE_32g.json',component/'SELECTED_CONFIGURATIONS_32g.json',
        BASE.parent/'tianshu_dense_tile_padding_20261021/kernel_padded.py',
        installed/'vllm/model_executor/models/llama.py',installed/'vllm/model_executor/custom_op.py',installed/'vllm/model_executor/layers/activation.py',
        installed/'flag_gems/fused/silu_and_mul.py',installed/'flag_gems/modules/activation.py',installed/'flag_gems/utils/pointwise_dynamic.py',
        *(installed/'vllm_fl').rglob('*.py')]
    for path in paths:sources[str(path)]=digest(path)
    frozen_check(sources)
    for name,sha in sources.items():
        dst=root/'source_snapshot'/Path(name).relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(name).read_bytes());assert digest(dst)==sha
    return command,environment,profile,sources
