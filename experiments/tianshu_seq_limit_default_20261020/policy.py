# SPDX-License-Identifier: Apache-2.0
"""Guard a lower native sequence default after the qualified budget policy runs."""
import functools,hashlib,importlib.util,inspect,json,os
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_seq_limit_budget_guard',BASE.parent/'tianshu_k8v16_budget_default_20261019/default_policy.py')
budget=importlib.util.module_from_spec(s);s.loader.exec_module(budget)
LIMIT=64

def install(module,mode,directory):
    assert mode in ('control','candidate')
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==budget.ARG_UTILS_SHA
    cls=module.EngineArgs;assert not getattr(cls,'_tianshu_seq_default_installed',False)
    original=cls._set_default_max_num_seqs_and_batched_tokens_args;signature=inspect.signature(original);rows=[]
    @functools.wraps(original)
    def select(owner,*args,**kwargs):
        bound=signature.bind(owner,*args,**kwargs).arguments
        usage,model,parallel=(bound[name] for name in ('usage_context','model_config','parallel_config'))
        before=owner.max_num_seqs
        eligible=budget.eligible(module,owner,usage,model,parallel)
        applied=mode=='candidate' and eligible and before is None
        answer=original(owner,*args,**kwargs)
        native_value=owner.max_num_seqs
        if applied:
            assert native_value==256
            owner.max_num_seqs=LIMIT
        rows.append(dict(mode=mode,eligible=eligible,default_applied=applied,before_max_num_seqs=before,
                         after_native_max_num_seqs=native_value,after_max_num_seqs=owner.max_num_seqs,
                         actual_budget=owner.max_num_batched_tokens,explicit_user_value_preserved=before is not None,
                         native_budget_policy_called_once=True,original_arguments_and_return_value_forwarded=True,RNG_not_read_or_modified=True,
                         native_source_sha256=budget.ARG_UTILS_SHA))
        directory.mkdir(parents=True,exist_ok=True)
        (directory/f'seq_default_{os.getpid()}.json').write_text(json.dumps(rows,indent=2)+'\n')
        return answer
    cls._set_default_max_num_seqs_and_batched_tokens_args=select
    cls._tianshu_seq_default_installed=True
