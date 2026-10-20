# SPDX-License-Identifier: Apache-2.0
"""Retain the FutureKV admission wrapper, replace only its native no-work delegate."""
import functools,hashlib,importlib.util,inspect,json,os
from pathlib import Path
BASE=Path(__file__).resolve().parent
COMPONENT=BASE.parent/'tianshu_kv_empty_allocate_20261021'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def install(scheduler,mode,directory):
    assert mode in ('control','candidate') and not getattr(scheduler,'_tianshu_kv_empty_installed',False)
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request,RequestStatus
    assert scheduler.max_model_len==131072 and scheduler.max_num_running_reqs==64 and scheduler.max_num_scheduled_tokens==4096
    assert scheduler.connector is None and scheduler.ec_connector is None and not scheduler.is_encoder_decoder and not scheduler.use_eagle
    assert scheduler.num_lookahead_tokens==0 and scheduler.dcp_world_size==scheduler.pcp_world_size==1
    controller=scheduler._tianshu_future_kv_reservation;manager=scheduler.kv_cache_manager
    assert controller.mode=='candidate' and controller.manager is manager and not controller.plans
    original=controller.original;assert original.__self__ is manager and original.__func__ is KVCacheManager.allocate_slots
    for name in ('CPU_GATE_32g.json','CPU_GUARD_GATE_32g.json'):
        gate=json.loads((COMPONENT/name).read_text());assert gate['complete']
        for path,digest in gate['sources'].items():assert sha(path)==digest
    spec=importlib.util.spec_from_file_location('_kv_empty_actual_verified_shortcut',COMPONENT/'shortcut.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    selected,stats=helper.wrap(manager,original,Request,RequestStatus.RUNNING,mode)
    assert inspect.signature(selected)==inspect.signature(original)
    controller.original=selected
    original_emit=controller.emit;directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    allocator=Path(inspect.getsourcefile(original)).resolve();reservation=Path(inspect.getsourcefile(type(controller))).resolve()
    record=dict(mode=mode,installed=True,after_original_Scheduler_init=True,FutureKV_admission_wrapper_retained=True,
        only_inner_no_work_allocator_delegate_wrapped=True,original_native_allocate_source=str(allocator),original_native_allocate_sha256=sha(allocator),
        policy_source=str(Path(__file__).resolve()),policy_sha256=sha(__file__),helper_source=str(COMPONENT/'shortcut.py'),helper_sha256=sha(COMPONENT/'shortcut.py'),
        future_policy_source=str(reservation),future_policy_sha256=sha(reservation),actual_budget=4096,actual_max_seqs=64,model_precision_mode='candidate',
        actual_KV_blocks=manager.block_pool.num_gpu_blocks,metrics_collector_is_None=manager.metrics_collector is None,
        no_KV_tensor_model_weight_RNG_or_request_data_modified=True)
    def save():
        row=dict(record,stats=dict(stats),future_native_delegate_calls=controller.stats['native_calls'],future_gate_rejections=controller.stats['gate_rejections'],
            tracked_future_requests=len(controller.plans),original_allocate_signature=str(inspect.signature(original)),
            selected_allocate_signature=str(inspect.signature(selected)))
        path=directory/f'route_{os.getpid()}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(path)
    @functools.wraps(original_emit)
    def emit(*args,**kwargs):
        before=controller.last_write;answer=original_emit(*args,**kwargs)
        if controller.last_write!=before:save()
        return answer
    controller.emit=emit;scheduler._tianshu_kv_empty_installed=True;save()
