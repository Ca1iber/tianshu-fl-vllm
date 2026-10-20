# SPDX-License-Identifier: Apache-2.0
"""Select native KV watermark at the existing guarded configuration boundary."""
import functools
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
BASE=Path(__file__).resolve().parent
p=BASE.parent/'tianshu_k8v16_budget_default_20261019/default_policy.py'
s=importlib.util.spec_from_file_location('_tianshu_headroom_budget_guard',p)
budget=importlib.util.module_from_spec(s)
s.loader.exec_module(budget)
RATIO=1/16

def install(module,mode,directory):
    assert mode in ('control','candidate')
    path=Path(module.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==budget.ARG_UTILS_SHA
    cls=module.EngineArgs
    assert not getattr(cls,'_tianshu_kv_headroom_installed',False)
    original=cls._set_default_max_num_seqs_and_batched_tokens_args
    signature=inspect.signature(original)
    records=[]
    @functools.wraps(original)
    def select(owner,*args,**kwargs):
        bound=signature.bind(owner,*args,**kwargs).arguments
        usage,model,parallel=(bound[name] for name in ('usage_context','model_config','parallel_config'))
        eligible=budget.eligible(module,owner,usage,model,parallel)
        before=owner.watermark
        applied=mode=='candidate' and eligible and before==0.0
        if applied:owner.watermark=RATIO
        try:
            answer=original(owner,*args,**kwargs)
        except BaseException:
            if applied:owner.watermark=before
            raise
        if applied:assert owner.watermark==RATIO
        records.append(dict(mode=mode,eligible=eligible,applied=applied,before_watermark=before,
                            after_watermark=owner.watermark,budget=owner.max_num_batched_tokens,
                            max_num_seqs=owner.max_num_seqs,original_arguments_forwarded=True,
                            native_KV_admission_implementation_unchanged=True,RNG_not_read_or_modified=True))
        directory.mkdir(parents=True,exist_ok=True)
        (directory/f'watermark_{os.getpid()}.json').write_text(json.dumps(records,indent=2)+'\n')
        return answer
    cls._set_default_max_num_seqs_and_batched_tokens_args=select
    cls._tianshu_kv_headroom_installed=True
