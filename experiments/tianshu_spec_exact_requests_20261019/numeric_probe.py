# SPDX-License-Identifier: Apache-2.0
"""Real model Graph vs padded/unpadded eager checks with candidate KV restored."""
from dataclasses import replace
from functools import wraps
import hashlib
import json
import os
from pathlib import Path


def install(module, directory):
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import (BatchDescriptor, get_forward_context,
                                     override_forward_context)
    directory = Path(directory)
    original = module.ModelRunnerFL._model_forward
    quotas = {'peak64': 3, 'tail_nonbucket': 3, 'tail_small': 2}
    state = dict(complete=False, cases=[], errors=[], quotas=quotas,
                 scope='Full GPU reductions; full hidden/logits not serialized; diagnostic only',
                 observer_has_synchronization_and_extra_forward=True, performance_score=False)

    def save():
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f'numeric_{os.getpid()}.json'
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(target)

    def tensor_hash(tensor):
        return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

    def check(a, b):
        assert a.shape == b.shape
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        if not finite:
            return dict(passed=False, finite=False)
        af, bf = a.float(), b.float()
        diff = (af-bf).abs()
        limit = .02 + .01*bf.abs()
        bad = int((diff > limit).sum().item())
        return dict(passed=bad == 0, finite=True, bad_elements=bad,
                    max_abs=float(diff.max().item()),
                    max_tolerance_ratio=float((diff/limit).max().item()),
                    atol=.02, rtol=.01,
                    bitwise=bool(torch.equal(a.contiguous().view(torch.uint8),
                                              b.contiguous().view(torch.uint8))))

    def flatten(value):
        if isinstance(value, torch.Tensor):
            return [value]
        assert isinstance(value, (list, tuple)), type(value)
        return [t for item in value for t in flatten(item)]

    @wraps(original)
    def forward(owner, *args, **kwargs):
        if state['complete']:
            return original(owner, *args, **kwargs)
        context = get_forward_context()
        desc = context.batch_descriptor
        n = owner.input_batch.num_reqs
        if (context.cudagraph_runtime_mode != CUDAGraphMode.FULL or desc is None or desc.uniform
                or desc not in getattr(owner.cudagraph_dispatcher, '_tianshu_mixed_graph_keys', set())
                or not 16 <= n <= 64):
            return original(owner, *args, **kwargs)
        N, T = desc.num_reqs, desc.num_tokens
        assert N == n, 'This candidate never pads the request count'
        group = 'peak64' if n == 64 else 'tail_nonbucket' if n > 56 else 'tail_small'
        if group is None or sum(c['group'] == group for c in state['cases']) >= quotas[group]:
            return original(owner, *args, **kwargs)
        row = dict(group=group, actual_requests=int(n), request_capacity=N, query_capacity=T)
        try:
            assert not args and kwargs['inputs_embeds'] is None and kwargs['intermediate_tensors'] is None
            assert owner.model.concrete_graph_entries[desc].graph is not None
            assert isinstance(context.attn_metadata, dict) and context.attn_metadata
            assert context.dp_metadata is None and context.ubatch_slices is None
            assert not context.additional_kwargs
            assert len(owner.kv_cache_config.kv_cache_groups) == 1
            meta = next(iter(context.attn_metadata.values()))
            assert meta.max_query_len == 2 and meta.num_actual_tokens == T
            cu = meta.query_start_loc.cpu().tolist()
            seq = meta.seq_lens.cpu().tolist()
            q = int(cu[n])
            assert len(cu) == N+1 and len(seq) == N and cu[0] == 0
            qlens = [cu[i+1]-cu[i] for i in range(n)]
            assert set(qlens) == {1, 2} and all(v == q for v in cu[n:])
            assert all(v == 0 for v in seq[n:]) and q <= T and T*8 <= q*9
            slots = meta.slot_mapping.cpu().tolist()
            assert len(slots) == T and all(v >= 0 for v in slots[:q])
            assert len(set(slots[:q])) == q and all(v == -1 for v in slots[q:])
            assert kwargs['input_ids'].shape == kwargs['positions'].shape == (T,)
            assert all(torch.equal(m.slot_mapping, meta.slot_mapping) for m in context.attn_metadata.values())
            row.update(actual_queries=q, query_lengths=qlens, cu=cu, sequence_lengths=seq,
                       slots=slots, input_ids=kwargs['input_ids'][:q].cpu().tolist(),
                       positions=kwargs['positions'][:q].cpu().tolist(),
                       GPU_block_table_sha256=tensor_hash(meta.block_table),
                       padding_cu_seq_slots_verified=True)
            table_cpu = meta.block_table.cpu().numpy()
            mapped_blocks = []
            for index, length in enumerate(qlens):
                positions = row['positions'][cu[index]:cu[index+1]]
                assert positions == list(range(seq[index]-length, seq[index]))
                mapped_blocks.extend(int(table_cpu[index, position//16]) for position in positions)
            assert [block*16 + position % 16 for block, position in zip(mapped_blocks, row['positions'])] == slots[:q]
            row['mapped_KV_blocks'] = mapped_blocks
            row['positions_and_block_table_slots_verified'] = True
            row['input_sha256'] = hashlib.sha256(json.dumps(
                {k: row[k] for k in ('cu', 'sequence_lengths', 'input_ids', 'positions')},
                sort_keys=True).encode()).hexdigest()
            assert all(c['input_sha256'] != row['input_sha256'] for c in state['cases'])
            caches = flatten(owner.kv_caches)
            assert len(caches) == len(context.attn_metadata) == 42
            assert all(t.dtype == torch.bfloat16 and t.ndim == 5
                       and tuple(t.shape[1:]) == (2, 16, 2, 128) for t in caches)
            assert len({t.data_ptr() for t in caches}) == 42
            row['KV_cache_shapes'] = [list(t.shape) for t in caches]
            row['KV_cache_strides'] = [list(t.stride()) for t in caches]
            blocks = sorted({s//16 for s in slots[:q]})
            block_ids = torch.tensor(blocks, device='cuda', dtype=torch.int64)
            guards = []
            before_guards = []
            for cache in caches:
                assert blocks[-1] < cache.shape[0]
                ids = sorted({0, cache.shape[0]-1, cache.shape[0]//2, cache.shape[0]//3} - set(blocks))
                assert ids
                index = torch.tensor(ids, device='cuda', dtype=torch.int64)
                guards.append(index)
                before_guards.append(cache.index_select(0, index))

            def guard_check():
                return all(torch.equal(cache.index_select(0, index).view(torch.int16), saved.view(torch.int16))
                           for cache, index, saved in zip(caches, guards, before_guards))

            rng_before = torch.cuda.get_rng_state().clone()
            candidate = original(owner, *args, **kwargs)
            assert isinstance(candidate, torch.Tensor) and candidate.shape == (T, 2048)
            candidate = candidate.clone()
            assert guard_check(), 'Candidate modified an untargeted KV page'
            candidate_pages = [t.index_select(0, block_ids) for t in caches]

            def restore():
                for cache, saved in zip(caches, candidate_pages):
                    cache.index_copy_(0, block_ids, saved)
                assert all(torch.equal(cache.index_select(0, block_ids).view(torch.int16), saved.view(torch.int16))
                           for cache, saved in zip(caches, candidate_pages)), 'Candidate KV restore failed'
                assert guard_check(), 'Reference modified an untargeted KV page'

            # The pinned FL wrapper's NONE branch calls this exact runnable.
            # Calling it directly keeps reference passes out of replay counters.
            padded_context = replace(context, cudagraph_runtime_mode=CUDAGraphMode.NONE)
            with override_forward_context(padded_context):
                padded = owner.model.runnable(*args, **kwargs)
            restore()
            unpadded_metadata = {}
            copies = {}
            for name, value in context.attn_metadata.items():
                if id(value) not in copies:
                    copies[id(value)] = replace(value, num_actual_tokens=q, slot_mapping=value.slot_mapping[:q],
                        query_start_loc=value.query_start_loc[:n+1], seq_lens=value.seq_lens[:n],
                        block_table=value.block_table[:n])
                unpadded_metadata[name] = copies[id(value)]
            unpadded_slots = {k: v[:q] for k, v in context.slot_mapping.items()}
            unpadded_context = replace(context, attn_metadata=unpadded_metadata, slot_mapping=unpadded_slots,
                cudagraph_runtime_mode=CUDAGraphMode.NONE, batch_descriptor=BatchDescriptor(q, n, False))
            unpadded_kwargs = dict(kwargs, input_ids=kwargs['input_ids'][:q], positions=kwargs['positions'][:q])
            with override_forward_context(unpadded_context):
                unpadded = owner.model.runnable(*args, **unpadded_kwargs)
            restore()
            assert unpadded.shape == (q, 2048) and padded.shape == (T, 2048)
            comparisons = dict(hidden_padded=check(candidate[:q], padded[:q]),
                               hidden_unpadded=check(candidate[:q], unpadded))
            logits = owner.model.compute_logits(candidate[:q])
            comparisons['logits_padded'] = check(logits, owner.model.compute_logits(padded[:q]))
            comparisons['logits_unpadded'] = check(logits, owner.model.compute_logits(unpadded))
            rng_after = torch.cuda.get_rng_state()
            row.update(comparisons=comparisons, all42_candidate_KV_pages_restored=True,
                       all42_untargeted_guard_pages_unchanged=True, touched_KV_blocks=len(blocks),
                       RNG_before_sha256=tensor_hash(rng_before), RNG_after_sha256=tensor_hash(rng_after),
                       RNG_unchanged=bool(torch.equal(rng_before, rng_after)),
                       full_hidden_and_logits_serialized=False)
            state['pending_case'] = row
            save()
            assert row['RNG_unchanged'] and all(c['passed'] for c in comparisons.values())
            state['cases'].append(row)
            state.pop('pending_case', None)
            state['complete'] = all(sum(c['group'] == k for c in state['cases']) == v for k, v in quotas.items())
            save()
            return candidate
        except BaseException as error:
            state['pending_case'] = row
            state['errors'].append(repr(error))
            save()
            raise

    module.ModelRunnerFL._model_forward = forward
    save()
    return state
