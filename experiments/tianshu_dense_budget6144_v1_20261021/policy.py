# SPDX-License-Identifier: Apache-2.0
"""Change only an eligible default after the qualified budget and seq wrappers."""
import functools,hashlib,importlib.util,inspect,json,os
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_latest_budget_native_guard',BASE.parent/'tianshu_k8v16_budget_default_20261019/default_policy.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)

def install(module,mode,directory):
    assert mode in ('control','candidate')
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==parent.ARG_UTILS_SHA
    cls=module.EngineArgs;assert not getattr(cls,'_tianshu_dense_budget_installed',False)
    original=cls._set_default_max_num_seqs_and_batched_tokens_args;signature=inspect.signature(original);rows=[]
    @functools.wraps(original)
    def select(owner,*args,**kwargs):
        bound=signature.bind(owner,*args,**kwargs).arguments
        usage,model,parallel=(bound[n] for n in ('usage_context','model_config','parallel_config'))
        before=owner.max_num_batched_tokens;supported=parent.eligible(module,owner,usage,model,parallel)
        applied=supported and before is None
        answer=original(owner,*args,**kwargs)
        inherited=owner.max_num_batched_tokens;seqs=owner.max_num_seqs
        if applied:
            assert inherited==4096 and seqs==64
            owner.max_num_batched_tokens=4096 if mode=='control' else 6144
        rows.append(dict(mode=mode,eligible=supported,default_applied=applied,before_budget=before,
            inherited_budget=inherited,after_budget=owner.max_num_batched_tokens,max_num_seqs=seqs,
            inherited_wrappers_called_once=True,original_arguments_and_return_value_forwarded=True,
            explicit_budget_preserved=before is None or owner.max_num_batched_tokens==before,
            native_source=str(Path(module.__file__).resolve()),native_source_sha256=parent.ARG_UTILS_SHA,
            policy_source=str(Path(__file__).resolve()),policy_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            RNG_and_request_data_not_accessed=True,applied_after_inner_budget_and_seq_records=True))
        directory.mkdir(parents=True,exist_ok=True);(directory/f'budget_{os.getpid()}.json').write_text(json.dumps(rows,indent=2)+'\n')
        return answer
    cls._set_default_max_num_seqs_and_batched_tokens_args=select;cls._tianshu_dense_budget_installed=True
