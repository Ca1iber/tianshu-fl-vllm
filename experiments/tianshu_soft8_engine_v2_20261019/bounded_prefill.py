# SPDX-License-Identifier: Apache-2.0
"""One compute-stream scratch arena, reserved inside memory profiling."""
import torch
from dense_prefill import DensePrefill, gather_decode, shape_for

CAPACITY_PAGES = 32768


def fits(n, length, capacity=CAPACITY_PAGES):
    blocks, pages = shape_for(n, length)
    return pages <= capacity


class BoundedDensePrefill(DensePrefill):
    def __init__(self):
        super().__init__()
        self.limit_pages = CAPACITY_PAGES
        self.storage = None
        self.dense_ids = None
        self.reservation = None

    def reserve(self, device='cuda'):
        if self.storage is not None:
            return
        assert not torch.cuda.is_current_stream_capturing()
        before = torch.cuda.memory_allocated()
        self.storage = torch.full((CAPACITY_PAGES + 1, 2, 16, 2, 128), 17.,
                                  device=device, dtype=torch.bfloat16)
        self.dense_ids = torch.arange(CAPACITY_PAGES, device=device, dtype=torch.int32)
        self.reservation = dict(capacity_pages=CAPACITY_PAGES,
            scratch_bytes=self.storage.numel() * self.storage.element_size(),
            table_bytes=self.dense_ids.numel() * self.dense_ids.element_size(),
            allocated_before=before, allocated_after=torch.cuda.memory_allocated(),
            single_reusable_arena=True)

    def __call__(self, kw):
        k, v, table, used = (kw[n] for n in ('k', 'v', 'block_table', 'seqused_k'))
        n = used.numel()
        blocks, pages = shape_for(n, kw['max_seqlen_k'])
        if not fits(n, kw['max_seqlen_k'], self.limit_pages):
            return None
        assert k.dtype == v.dtype == torch.uint8 and k.shape[1:] == v.shape[1:] == (16, 2, 128)
        assert table.dtype == torch.int32 and table.shape[0] == n and table.shape[1] >= blocks
        assert k.stride(-1) == v.stride(-1) == 1
        self.reserve(k.device)
        storage = self.storage
        dense_table = self.dense_ids[:pages].reshape(n, blocks)
        dk, dv = storage[:pages].unbind(1)
        result = gather_decode[(pages, 2)](k, v, table, used, dk, dv,
            *k.stride()[:3], *v.stride()[:3], table.stride(0), dk.stride(0),
            blocks, 4096, num_warps=4, num_stages=1)
        self.last = dict(storage=storage, k=k, v=v, table=table, used=used,
            dense_k=dk, dense_v=dv, blocks=blocks, dense_table=dense_table)
        self.records[str((n, blocks))] = dict(sequences=n, blocks_per_sequence=blocks,
            scratch_bytes=storage.numel() * storage.element_size(),
            key_dtype=str(k.dtype), scratch_dtype=str(storage.dtype),
            n_regs=getattr(result, 'n_regs', None), n_spills=getattr(result, 'n_spills', None),
            grid=[pages, 2], warps=result.metadata.num_warps,
            arena_ptr=storage.data_ptr(), table_ptr=self.dense_ids.data_ptr())
        updated = dict(kw)
        updated.update(k=dk, v=dv, block_table=dense_table)
        return updated
