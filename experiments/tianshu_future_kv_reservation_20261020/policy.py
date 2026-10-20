# SPDX-License-Identifier: Apache-2.0
"""Admission-only future-output reservation for a guarded full-attention pool.

No KV data, tokens, RNG, native allocation size or running-request allocation is
changed. Claims are virtual and disappear on normal free or deferred-free pop.
"""
import atexit
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

BASE = Path(__file__).resolve().parent


def supported(scheduler):
    manager = scheduler.kv_cache_manager
    groups = manager.coordinator.single_type_managers
    return (len(groups) == 1 and type(groups[0]).__name__ == 'FullAttentionManager'
        and type(groups[0].kv_cache_spec).__name__ == 'FullAttentionSpec'
        and groups[0].block_size == 16 and not groups[0].use_eagle
        and scheduler.connector is None and scheduler.ec_connector is None
        and not scheduler.is_encoder_decoder and not scheduler.use_eagle
        and scheduler.num_lookahead_tokens == 0 and scheduler.scheduler_reserve_full_isl
        and scheduler.max_model_len == 131072 and manager.watermark_blocks == 0
        and scheduler.vllm_config.model_config.model == '/workspace/MiniCPM5-2B'
        and scheduler.parallel_config.tensor_parallel_size == 1
        and scheduler.parallel_config.pipeline_parallel_size == 1
        and scheduler.dcp_world_size == scheduler.pcp_world_size == 1)


class Reservation:
    def __init__(self, manager, mode, directory):
        assert mode in ('control', 'candidate')
        self.manager, self.mode, self.directory = manager, mode, Path(directory)
        self.unit = manager.coordinator.single_type_managers[0]
        self.plans = {}
        self.last_write = 0.
        self.stats = dict(mode=mode, installed=True, admission_checks=0, gate_rejections=0,
            diagnostic_would_reject=0, native_calls=0, successful_admissions=0,
            free_calls=0, pop_for_free_calls=0, max_tracked_requests=0,
            max_outstanding_blocks=0, rejection_samples=[],
            no_KV_data_tokens_or_RNG_access=True, running_allocations_forwarded=True,
            native_allocation_sizes_and_return_values_preserved=True)
        self.original = manager.allocate_slots
        self.signature = inspect.signature(self.original)
        self.original_free = manager.free
        self.original_pop = manager.pop_blocks_for_free
        @functools.wraps(self.original)
        def allocate(*args, **kwargs):
            return self.allocate(*args, **kwargs)
        @functools.wraps(self.original_free)
        def free(request):
            return self.free(request)
        @functools.wraps(self.original_pop)
        def pop(request):
            return self.pop(request)
        manager.allocate_slots, manager.free, manager.pop_blocks_for_free = allocate, free, pop

    def emit(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_write < 5:
            return
        self.last_write = now
        self.directory.mkdir(parents=True, exist_ok=True)
        row = dict(self.stats, tracked_requests=len(self.plans), outstanding_blocks=self.outstanding())
        target = self.directory / f'future_reservations_{os.getpid()}.json'
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
        tmp.replace(target)

    def outstanding(self, exclude=None):
        return sum(max(planned - len(self.unit.req_to_blocks.get(key, ())), 0)
                   for key, planned in self.plans.items() if key != exclude)

    def proposal(self, args, kwargs):
        # Native running allocations omit this flag. Keep that hot path direct.
        if not kwargs.get('full_sequence_must_fit', False) and len(args) < 9:
            return None
        try:
            bound = self.signature.bind(*args, **kwargs)
        except TypeError:
            return None  # The original call produces its own unchanged error.
        bound.apply_defaults()
        b = bound.arguments
        request = b['request']
        status = getattr(getattr(request, 'status', None), 'name', None)
        if (not b['full_sequence_must_fit'] or status not in ('WAITING', 'PREEMPTED')
            or b['num_new_tokens'] <= 0
            or b['num_lookahead_tokens'] or b['num_external_computed_tokens']
            or b['delay_cache_blocks'] or b['num_encoder_tokens'] or b['reserved_blocks'] < 0
            or request.sampling_params is None
            or type(request.max_tokens) is not int or request.max_tokens <= 0
            or type(request.num_prompt_tokens) is not int or request.num_prompt_tokens <= 0):
            return None
        planned_tokens = min(self.manager.max_model_len,
                             request.num_prompt_tokens + request.max_tokens)
        if planned_tokens < request.num_tokens:
            return None
        planned_blocks = (planned_tokens + self.unit.block_size - 1) // self.unit.block_size
        if planned_blocks >= self.manager.block_pool.num_gpu_blocks:
            return None  # Preserve progress if even one forecast cannot fit.
        prefix = b['new_computed_blocks']
        blocks = prefix.blocks if prefix is not None else self.manager.empty_kv_cache_blocks.blocks
        total_computed = min(request.num_computed_tokens + b['num_new_computed_tokens'], self.manager.max_model_len)
        needed = self.manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id, num_tokens=planned_tokens,
            new_computed_blocks=blocks, num_encoder_tokens=0,
            total_computed_tokens=total_computed,
            num_tokens_main_model=planned_tokens, apply_admission_cap=True)
        outstanding = self.outstanding(exclude=request.request_id)
        free = self.manager.block_pool.get_num_free_blocks()
        required = needed + outstanding + b['reserved_blocks']
        return request, planned_blocks, needed, outstanding, required, free

    def allocate(self, *args, **kwargs):
        proposal = self.proposal(args, kwargs)
        if proposal is not None:
            request, planned, needed, outstanding, required, free = proposal
            self.stats['admission_checks'] += 1
            self.stats['max_outstanding_blocks'] = max(self.stats['max_outstanding_blocks'], outstanding)
            if required > free:
                self.stats['diagnostic_would_reject'] += 1
                if self.mode == 'candidate':
                    self.stats['gate_rejections'] += 1
                    if len(self.stats['rejection_samples']) < 8:
                        self.stats['rejection_samples'].append(dict(
                            needed_future_blocks=needed, other_future_blocks=outstanding,
                            required_blocks=required, free_blocks=free, tracked_requests=len(self.plans)))
                    self.emit()
                    return None
        self.stats['native_calls'] += 1
        answer = self.original(*args, **kwargs)
        if proposal is not None and answer is not None:
            self.plans[request.request_id] = planned
            self.stats['successful_admissions'] += 1
            self.stats['max_tracked_requests'] = max(self.stats['max_tracked_requests'], len(self.plans))
            self.emit()
        return answer

    def free(self, request):
        answer = self.original_free(request)
        self.plans.pop(request.request_id, None)
        self.stats['free_calls'] += 1
        self.emit(force=not self.plans)
        return answer

    def pop(self, request):
        answer = self.original_pop(request)
        self.plans.pop(request.request_id, None)
        self.stats['pop_for_free_calls'] += 1
        self.emit(force=not self.plans)
        return answer


def install(scheduler, mode, directory):
    gate = json.loads((BASE / 'CPU_GATE_32g.json').read_text())
    assert gate['complete'] and gate['CPU_only']
    for name, digest in gate['sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    assert supported(scheduler), 'Future reservation is restricted to the verified full-attention setup'
    manager = scheduler.kv_cache_manager
    native = Path(inspect.getsourcefile(manager.allocate_slots))
    assert str(native) in gate['sources']
    controller = Reservation(manager, mode, directory)
    controller.stats.update(native_allocator_source=str(native), native_allocator_sha256=gate['sources'][str(native)],
        block_size=controller.unit.block_size, KV_num_blocks=manager.block_pool.num_gpu_blocks,
        max_model_len=manager.max_model_len, only_full_attention_single_pool=True)
    controller.emit(force=True)
    atexit.register(controller.emit, True)
    return controller
