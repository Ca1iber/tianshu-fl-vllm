# SPDX-License-Identifier: Apache-2.0
"""Stage-only maxK tightening. Native route and decode kwargs remain owned by FL."""
import functools
import importlib.util
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_prefill_key_bound_CPU_metadata',BASE.parent/'tianshu_prefill_key_bound_20261021/bound.py')
helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)

def narrow(route,kw):
    bound=getattr(route,'_tianshu_prefill_key_bound',None)
    if not isinstance(bound,helper.PrefillBound):return kw
    if (getattr(route,'num_decodes',None)!=bound.num_decodes
        or kw.get('max_seqlen_k')!=bound.original_max_key_length
        or kw.get('max_seqlen_q')!=getattr(route,'max_prefill_query',None)
        or kw.get('seqused_k') is None
        or kw['seqused_k'].numel()!=bound.num_prefills
        or not 0<bound.max_key_length<bound.original_max_key_length):return kw
    return dict(kw,max_seqlen_k=bound.max_key_length)

def install(hybrid,dense,mode):
    assert mode in ('control','candidate') and not getattr(hybrid,'_tianshu_prefill_bound_installed',False)
    original_route=hybrid.route_from_common;cls=type(dense);original_stage=cls.__call__
    state=dict(mode=mode,route_calls=0,metadata_bounds=0,metadata_unavailable=0,
        stage_calls=0,tightened_calls=0,unchanged_calls=0,
        original_route_identity_preserved=True,original_route_called_once=True,
        decode_dispatch_and_original_kwargs_unchanged=True,
        no_GPU_CPU_transfers_added=True,no_files_written_in_hooks=True,
        Python_calls_are_not_Graph_replays=True)
    @functools.wraps(original_route)
    def route(common):
        result=original_route(common);state['route_calls']+=1
        if mode=='candidate' and result is not None:
            bound=helper.derive(common,result)
            if bound is None:state['metadata_unavailable']+=1
            else:
                assert not hasattr(result,'_tianshu_prefill_key_bound')
                result._tianshu_prefill_key_bound=bound;state['metadata_bounds']+=1
        return result
    @functools.wraps(original_stage)
    def stage(owner,kw):
        if owner is not dense:return original_stage(owner,kw)
        selected=narrow(hybrid._ROUTE.get(),kw) if mode=='candidate' else kw
        state['stage_calls']+=1
        state['tightened_calls' if selected is not kw else 'unchanged_calls']+=1
        answer=original_stage(owner,selected)
        # Extend the existing shape record. Its existing observer persists it at
        # shape changes/4096dispatches; this hook does not add any file writes.
        if answer is not None:
            n=selected['seqused_k'].numel();blocks=(selected['max_seqlen_k']+15)//16
            row=owner.records.get(str((n,blocks)))
            if row is not None:
                row.update(prefill_bound_mode=mode,stage_bound=selected['max_seqlen_k'],
                    original_global_maxK=kw['max_seqlen_k'],original_global_pages=n*((kw['max_seqlen_k']+15)//16),
                    actual_stage_pages=n*blocks,only_prefill_stage_kwargs_copied=(selected is not kw),
                    helper_source=str(BASE.parent/'tianshu_prefill_key_bound_20261021/bound.py'),
                    stage_policy_source=str(BASE/'policy.py'))
        return answer
    hybrid.route_from_common=route;cls.__call__=stage;hybrid._tianshu_prefill_bound_installed=True
    return state
