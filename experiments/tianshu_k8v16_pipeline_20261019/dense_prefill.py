# SPDX-License-Identifier: Apache-2.0
"""Decode only referenced pages to reusable dense BF16 prefill storage."""
import torch
import triton
import triton.language as tl


def shape_for(n, max_length):
    assert n > 0 and max_length > 0
    blocks = (max_length + 15) // 16
    return blocks, n * blocks


@triton.jit
def gather_decode(K, V, Table, Lengths, OutK, OutV,
                  KB: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr,
                  VB: tl.constexpr, VT: tl.constexpr, VH: tl.constexpr,
                  TS: tl.constexpr, OB: tl.constexpr,
                  BLOCKS: tl.constexpr, TILE: tl.constexpr):
    page = tl.program_id(0)
    plane = tl.program_id(1)
    seq, logical_page = page // BLOCKS, page % BLOCKS
    length = tl.load(Lengths + seq)
    active = logical_page * 16 < length
    physical_page = tl.load(Table + seq * TS + logical_page, active, other=0)
    offsets = tl.arange(0, TILE)
    token = offsets // 256
    head = offsets // 128 % 2
    dim = offsets % 128
    valid = active & (logical_page * 16 + token < length) & (offsets < 4096)
    if plane == 0:
        packed = tl.load(K + physical_page * KB + token * KT + head * KH + dim,
                         valid, other=0).to(tl.uint32)
        magnitude = packed & 127
        bits = tl.where(magnitude >= 8, (magnitude << 4) + 14848, 0)
        bits = bits | ((packed & 128) << 8)
        decoded_k = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        tl.store(OutK + page * OB + offsets, decoded_k, valid)
    else:
        value = tl.load(V + physical_page * VB + token * VT + head * VH + dim,
                        valid, other=0)
        tl.store(OutV + page * OB + offsets, value, valid)


class DensePrefill:
    def __init__(self):
        self.buffers = {}
        self.last = None
        self.records = {}

    def __call__(self, kw):
        k, v, table, used = (kw[n] for n in ('k', 'v', 'block_table', 'seqused_k'))
        n = used.numel()
        blocks, pages = shape_for(n, kw['max_seqlen_k'])
        assert k.dtype == torch.uint8 and v.dtype == torch.bfloat16 and k.shape[1:] == v.shape[1:] == (16, 2, 128)
        assert k.stride(-1) == v.stride(-1) == 1 and table.shape[0] == n and table.shape[1] >= blocks
        key = (k.device, n, blocks)
        if key not in self.buffers:
            assert not torch.cuda.is_current_stream_capturing(), 'warm scratch before graph capture'
            storage = torch.full((pages + 1, 2, 16, 2, 128), 17., device=k.device, dtype=torch.bfloat16)
            dense_table = torch.arange(pages, device=k.device, dtype=table.dtype).reshape(n, blocks)
            self.buffers[key] = storage, dense_table
        storage, dense_table = self.buffers[key]
        dk, dv = storage[:pages].unbind(1)
        result = gather_decode[(pages, 2)](
            k, v, table, used, dk, dv,
            *k.stride()[:3], *v.stride()[:3], table.stride(0), dk.stride(0),
            blocks, 4096, num_warps=4, num_stages=1)
        self.last = dict(storage=storage, k=k, v=v, table=table, used=used, dense_k=dk,
                         dense_v=dv, blocks=blocks, dense_table=dense_table)
        self.records[str((n, blocks))] = dict(sequences=n, blocks_per_sequence=blocks,
            scratch_bytes=storage.numel() * storage.element_size(), key_dtype=str(k.dtype),
            scratch_dtype=str(storage.dtype), n_regs=getattr(result, 'n_regs', None),
            n_spills=getattr(result, 'n_spills', None), grid=[pages, 2], warps=result.metadata.num_warps)
        updated = dict(kw)
        updated.update(k=dk, v=dv, block_table=dense_table)
        return updated

    def verify_last(self, dequantize):
        """Outside timing: compare every active gathered cell and guard page."""
        f = self.last
        assert f is not None and bool((f['storage'][-1] == 17.).all())
        checked = 0
        for seq, length in enumerate(f['used'].cpu().tolist()):
            pages = (length + 15) // 16
            ids = f['table'][seq, :pages].long()
            begin = seq * f['blocks']
            for src, dst in ((f['k'], f['dense_k']), (f['v'], f['dense_v'])):
                expected = (src[ids] if src.dtype == torch.bfloat16 else dequantize(src[ids])).reshape(-1, 2, 128)[:length]
                actual = dst[begin:begin + pages].reshape(-1, 2, 128)[:length]
                assert torch.equal(actual, expected)
                checked += actual.numel()
        return dict(active_cells_bitwise=True, checked_elements=checked,
                    scratch_guard_preserved=True, distinct_dense_page_ids=True)
