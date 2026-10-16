# SPDX-License-Identifier: Apache-2.0
"""Bounded prefill cadence via the scheduler's existing eligibility check."""
import functools


def supported(owner):
    config = owner.vllm_config
    model, parallel = config.model_config, owner.parallel_config
    hf = model.hf_config
    return (not owner.has_mamba_layers
            and not owner.is_encoder_decoder and not model.is_multimodal_model
            and (hf.hidden_size, hf.num_hidden_layers, hf.num_attention_heads, hf.num_key_value_heads)
            == (2048, 42, 16, 2)
            and config.speculative_config is None and owner.lora_config is None
            and owner.connector is None and owner.ec_connector is None
            and parallel.tensor_parallel_size == parallel.pipeline_parallel_size
            == parallel.data_parallel_size == parallel.decode_context_parallel_size
            == parallel.prefill_context_parallel_size == 1
            and parallel.worker_cls == 'vllm_fl.worker.worker.WorkerFL'
            and owner.scheduler_config.enable_chunked_prefill
            and owner.scheduler_config.async_scheduling
            and owner.policy.name == 'FCFS'
            and owner.max_model_len == 131072 and owner.block_size == 16)


def pending(request):
    return request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens


def pressure(owner):
    if any(r.num_computed_tokens < r.num_prompt_tokens and pending(r) > 1 for r in owner.running):
        return True
    return (len(owner.running) < owner.max_num_running_reqs
            and any(r.status.name == 'WAITING' and r.num_tokens - r.num_computed_tokens > 1
                    for r in owner.waiting))


def wrap(original, *, enabled, observe):
    @functools.wraps(original)
    def schedule(owner, throttle_prefills=False):
        next_step = owner.current_step + 1
        active = supported(owner)
        has_pressure = active and pressure(owner)
        enough_blocks = active and owner.kv_cache_manager.block_pool.get_num_free_blocks() >= (
            2 * ((owner.max_num_scheduled_tokens + owner.block_size - 1) // owner.block_size)
            + owner.max_num_running_reqs)
        # Every seventeenth step follows the original policy. With no prefill pressure,
        # low free KV, or a caller prefill throttle, all steps follow it.
        defer = (enabled and has_pressure and enough_blocks and not throttle_prefills
                 and (next_step - 1) % 17 < 16)
        held = []
        configured_v2 = owner.use_v2_model_runner
        # WorkerFL explicitly constructs the MRV1-style ModelRunnerFL, while this
        # installed VllmConfig defaults its flag to V2 for Llama. MRV1 needs the
        # original scheduler's all_token_ids recovery for temporarily omitted
        # requests. Select that existing metadata path for the guarded candidate.
        use_mrv1_metadata = enabled and active
        if use_mrv1_metadata:
            owner.use_v2_model_runner = False
        if defer:
            for request in owner.running:
                if request.num_computed_tokens >= request.num_prompt_tokens and pending(request) == 1:
                    old = request.next_decode_eligible_step
                    temporary = max(old, next_step + 1)
                    held.append((request, old, temporary))
                    request.next_decode_eligible_step = temporary
        try:
            result = original(owner, throttle_prefills=throttle_prefills)
        finally:
            owner.use_v2_model_runner = configured_v2
            # MRV1 does not change this field in _update_after_schedule. Preserve
            # any unexpected legitimate update instead of overwriting it.
            for request, old, temporary in held:
                if request.next_decode_eligible_step == temporary:
                    request.next_decode_eligible_step = old
        violations = sum(r.request_id in result.num_scheduled_tokens for r, _, _ in held)
        assert violations == 0, 'Original scheduler scheduled a temporarily ineligible request'
        observe(owner, result, dict(supported=active, pressure=has_pressure, enough_blocks=enough_blocks,
                                   defer=defer, held=len(held), step=next_step,
                                   scheduled_held_requests=violations,
                                   configured_use_v2=configured_v2,
                                   used_mrv1_metadata=use_mrv1_metadata,
                                   restored_use_v2=owner.use_v2_model_runner))
        return result
    return schedule
