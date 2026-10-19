# SPDX-License-Identifier: Apache-2.0
"""Keep measured software8 decode; stage prefill through the original BF16 route."""
from types import SimpleNamespace
import torch
import legacy_routing
from dense_prefill import DensePrefill


def install(backend, hybrid, adaptive, reader, writer):
    bf16_dispatch = hybrid.dispatch
    state = legacy_routing.install(backend, hybrid, adaptive, reader, writer)
    byte_dispatch = hybrid.dispatch
    dense = DensePrefill()
    counts = dict(prefill=0, mixed=0, native_BF16=0, direct_byte_decode=0)

    def dispatch(original, route, kw):
        if kw['k'].dtype != torch.uint8:
            counts['native_BF16'] += 1
            return byte_dispatch(original, route, kw)
        if kw['max_seqlen_q'] <= 1:
            counts['direct_byte_decode'] += 1
            return byte_dispatch(original, route, kw)
        nd = route.num_decodes if route is not None else 0
        prefill = dict(kw)
        prefill_route = None
        if route is not None:
            if nd:
                decode = dict(kw)
                decode.update(q=kw['q'][:nd], out=kw['out'][:nd],
                    cu_seqlens_q=kw['cu_seqlens_q'][:nd + 1], max_seqlen_q=1,
                    seqused_k=kw['seqused_k'][:nd], block_table=kw['block_table'][:nd])
                byte_dispatch(original, None, decode)
                counts['mixed'] += 1
            prefill.update(q=kw['q'][nd:], out=kw['out'][nd:], cu_seqlens_q=route.prefill_cu,
                max_seqlen_q=route.max_prefill_query, seqused_k=kw['seqused_k'][nd:],
                block_table=kw['block_table'][nd:])
            prefill_route = SimpleNamespace(num_decodes=0, prefill_cu=route.prefill_cu,
                                            max_prefill_query=route.max_prefill_query)
        counts['prefill'] += 1
        return bf16_dispatch(original, prefill_route, dense(prefill))

    hybrid.dispatch = dispatch
    state.update(dense=dense, dense_counts=counts)
    return state
