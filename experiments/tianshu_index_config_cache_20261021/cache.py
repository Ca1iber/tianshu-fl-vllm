# SPDX-License-Identifier: Apache-2.0
"""Prime one missing fallback entry in FlagGems' existing Config cache."""
OP='index_adjacent'
def signature(configs):
    return [dict(kwargs=dict(c.kwargs),num_warps=c.num_warps,num_stages=c.num_stages,
        num_ctas=c.num_ctas,maxnreg=c.maxnreg,pre_hook=repr(c.pre_hook)) for c in configs]
def prime(loader):
    cache=loader.loaded_triton_config
    existed=OP in cache
    configs=loader.get_tuned_config(OP)
    assert configs and all(c.pre_hook is None for c in configs)
    expected=signature(configs)
    if not existed:cache[OP]=configs
    actual=loader.get_tuned_config(OP)
    assert actual is cache[OP] and signature(actual)==expected
    return dict(complete=True,op=OP,entry_existed=existed,configuration_count=len(actual),signature=expected,
        existing_native_cache_used=True,native_method_unchanged=True,other_entries_unchanged=True)
