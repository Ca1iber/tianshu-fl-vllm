# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.topk import _get_finfo_val, _get_iinfo_val, argsort
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

DETECTOR_WORDS = 2

def unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def get_int_t(num_bits: tl.constexpr, signed: tl.constexpr) -> tl.dtype:
    num_bits = unwrap_if_constexpr(num_bits)
    signed = unwrap_if_constexpr(signed)
    return tl.core.get_int_dtype(num_bits, signed)


@tl.constexpr
def one_zeros(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)


@tl.constexpr
def zero_ones(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return (1 << (num_bits - 1)) - 1


@triton.jit
def uint_to_uint(x, descending: tl.constexpr = False):
    out = ~x if descending else x
    return out


@triton.jit
def int_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    udtype = get_int_t(num_bits, False)
    ux = tl.cast(x, udtype, bitcast=True)
    if descending:
        # 0111111....1
        bit_mask: tl.constexpr = zero_ones(num_bits)
        bit_mask_tensor = tl.full((), value=bit_mask, dtype=udtype)
        out = ux ^ bit_mask_tensor
    else:
        # 1000000...0
        sign_bit_mask: tl.constexpr = one_zeros(num_bits)
        sign_bit_mask_tensor = tl.full((), value=sign_bit_mask, dtype=udtype)
        out = ux ^ sign_bit_mask_tensor
    return out


@triton.jit
def floating_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    sdtype = get_int_t(num_bits, True)
    udtype = get_int_t(num_bits, False)
    sx = x.to(sdtype, bitcast=True)
    ux = x.to(udtype, bitcast=True)

    sign_bit_mask_v: tl.constexpr = one_zeros(num_bits)
    sign_bit_mask = tl.full((), value=sign_bit_mask_v, dtype=udtype)
    # mind the dtype, right_shift for signed is arithmetic right shift
    # Fix for triton 3.1 or else `sx >> rshift_bits` is promoted to int32
    rshift_bits = tl.full((), value=num_bits - 1, dtype=sdtype)
    mask = sign_bit_mask | (sx >> rshift_bits).to(udtype, bitcast=True)
    tl.static_assert(mask.dtype == udtype, "type mismatch")
    # 1000000000...0 for positive
    # 1111111111...1 for negative
    if descending:
        out = ux ^ (~mask)
    else:
        out = ux ^ mask
    return out.to(udtype, bitcast=True)


@triton.jit
def convert_to_uint_preverse_order(x: tl.tensor, descending: tl.constexpr = False):
    if x.dtype.is_floating():
        out = floating_to_uint(x, descending)
    elif x.dtype.is_int_signed():
        out = int_to_uint(x, descending)
    elif x.dtype.is_int_unsigned():
        out = uint_to_uint(x, descending)
    return out

@triton.jit
def _combine_or(a, b):
    return a | b



@triton.jit
def compute_global_hist_kernel(
    arr_ptr,
    out_ptr,
    num_passes,
    m,
    n,
    tiles_n_per_cta,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    num_bits_per_pass: tl.constexpr,
    descending: tl.constexpr,
    identity_mismatch_ptr=None, KEY_BITS: tl.constexpr=32,
):
    # arr_ptr: (m, n)
    # out_ptr: (m, n_passes, r), where r = 2 ** k_bits is the number of bins
    pid = tl.program_id(0)
    pid_n = pid // m
    pid_m = pid % m

    r: tl.constexpr = 2**num_bits_per_pass
    bfe_mask: tl.constexpr = (1 << num_bits_per_pass) - 1  # a.k.a. 2 ** k_bits - 1
    CTA_TILE_N: tl.constexpr = TILE_N * tiles_n_per_cta
    cta_n_start = CTA_TILE_N * pid_n
    cta_n_end = tl.minimum(cta_n_start + CTA_TILE_N, n)

    for p in range(0, num_passes):  # parallel
        bit_offset = p * num_bits_per_pass
        if KEY_BITS > 32:
            mismatch_found = tl.zeros((), dtype=tl.uint64)
        else:
            mismatch_found = tl.zeros((), dtype=tl.uint32)
        for r_start in range(0, r, TILE_R):  # parallel
            bin_indices = r_start + tl.arange(0, TILE_R)
            # radix_sort bounds n < 2**30, so all bin counts fit in int32.
            acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int32)
            for n_start in range(cta_n_start, cta_n_end, TILE_N):  # sequantial
                n_offsets = n_start + tl.arange(0, TILE_N)  # (TILE_N, )
                mask = n_offsets < cta_n_end
                arr = tl.load(arr_ptr + pid_m * n + n_offsets, mask=mask)
                arr = convert_to_uint_preverse_order(arr, descending)
                key = (arr >> bit_offset) & bfe_mask  # (TILE_N, )
                if identity_mismatch_ptr is not None:
                    if (p == 0) & (r_start == 0):
                        differences = tl.where(mask, arr ^ (arr >> num_bits_per_pass), 0)
                        differences = differences.to(mismatch_found.dtype)
                        mismatch_found |= tl.reduce(differences, 0, _combine_or)
                matches = tl.where(
                    mask, (bin_indices[:, None] == key), False
                )  # (TILE_R, TILE_N)
                acc += matches
            local_sum = tl.sum(acc, axis=1)
            tl.atomic_add(
                out_ptr + pid_m * num_passes * r + p * r + bin_indices,
                local_sum,
                sem="relaxed",
            )
        if identity_mismatch_ptr is not None:
            if p == 0:
                tl.atomic_or(identity_mismatch_ptr + pid_m * 2,
                             mismatch_found.to(tl.uint32), sem="relaxed")
                if KEY_BITS > 32:
                    tl.atomic_or(identity_mismatch_ptr + pid_m * 2 + 1,
                                 (mismatch_found >> 32).to(tl.uint32), sem="relaxed")


@triton.jit
def sweep(
    arr_ptr,
    associate_arr_ptr,  # inputs: (key & value)
    out_ptr,
    associate_out_ptr,  # outputs: (key & value)
    excumsum_bins_ptr,
    status_ptr,  # aux input and status
    n_passes,
    pass_id,
    bit_offset,
    m,
    N,
    OUT_N,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    k_bits: tl.constexpr,
    descending: tl.constexpr,
):
    # r: num_bins = 2 ** k_bits
    # OUT_N: grid_n = cdiv(N, )

    # arr_ptr: (m, N)
    # out_ptr: (m, N)
    # excumsum_bins_ptr: (m, n_passes, r)
    # flag_ptr: (m, r, OUT_N)

    # grid: (m, grid_r, grid_n)

    # load data
    pid = tl.program_id(0)
    pid_m = pid % m
    pid_n = pid // m
    pid_r = tl.program_id(1)

    # bit masks
    aggregate_mask: tl.constexpr = 1 << 30
    inclusive_prefix_mask: tl.constexpr = 1 << 31
    v_mask: tl.constexpr = (1 << 30) - 1
    bfe_mask: tl.constexpr = (1 << k_bits) - 1  # a.k.a. 2 ** k_bits - 1

    # initialize flag to zero-local sum is not ready
    r: tl.constexpr = 2**k_bits
    cta_r_start = pid_r * TILE_R
    cta_r_end = tl.minimum(cta_r_start + TILE_R, r)

    # cumsum for a bin_index
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)  # (TILE_N, )
    mask = n_offsets < N
    arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask)
    arr_u = convert_to_uint_preverse_order(arr, descending)
    key = (arr_u >> bit_offset) & bfe_mask  # (TILE_N, )
    if associate_arr_ptr is not None:
        associate_arr = tl.load(associate_arr_ptr + pid_m * N + n_offsets, mask=mask)
    # since triton can only use scalar as condition, loop by bin_index
    # status must be pre zero-initialized, or else we have to initialize it
    for bin_index in range(cta_r_start, cta_r_end):
        matches = tl.where(mask, key == bin_index, False)  # (TILE_N, ) bool
        # cta level cumsum per bin
        # CAUTION: tl.sum in triton 3.2 does not promote type
        local_sum = tl.sum(matches.to(tl.uint32), axis=0)
        pack0 = aggregate_mask | local_sum
        status_offset = pid_m * (r * OUT_N) + bin_index * OUT_N + pid_n
        tl.store(status_ptr + status_offset, pack0, cache_modifier=".cg")

        # decoupled lookback
        # 还没查自己前面的桶 bin_index 有多少个元素，所以计数为 0 
        exclusive_prefix = tl.zeros((), dtype=tl.uint32)
        # 插前面的那块
        i_lookback = pid_n - 1
        # 进入循环，读取第二块的记录
        while i_lookback >= 0:
            flag_offset_i = pid_m * (r * OUT_N) + bin_index * OUT_N + i_lookback
            # pack1 就是读回来的记录，有可能是“我自己有多少个：01”，也可能是“截止我之前总共有多少个：10”
            pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)  # uin32
            # 没读到就一直读
            while pack1 == 0:
                pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)
            # 把督导的数量加进计数器
            exclusive_prefix += pack1 & v_mask
            # 判断读到的是“我有”还是“截止我共有”
            if (pack1 & aggregate_mask) == aggregate_mask:
                i_lookback -= 1
            else:
                i_lookback = -1
        # 退出了刚才的 while 循环，意味着当前 CTA 已经拿到了当前桶在所有前面的数据块中的完整数量
        # 算出“截止我共有”的数量并给这个数量贴上“截止我共有”的标签
        pack2 = inclusive_prefix_mask | (exclusive_prefix + local_sum)
        # 覆盖之前的标签
        tl.store(status_ptr + status_offset, pack2, cache_modifier=".cg")
        # Empty bins must still publish both status states above.
        if local_sum != 0:
            # 算出他们在本块内的先后编号，l_e_c 表示在本数据块内，自己前面有几个同桶元素
            local_ex_cumsum = (
                tl.cumsum(matches.to(tl.uint32), axis=0) - matches
            )  # (TILE_N, )
            # 加上前面数据块的同桶数量，得到这一数据块里这个桶在整个对应编号桶内部的编号
            ex_cumsum_in_bin = (
                exclusive_prefix + local_ex_cumsum
            )  # global ex_cumsum_in_bin (TILE_N, )

            # ex_cumsum_bins (m, n_passes, r)
            # 加上这个桶在输出中的起点
            ex_cumsum_bins = tl.load(
                excumsum_bins_ptr + pid_m * (n_passes * r) + pass_id * r + bin_index
            )  # scalar
            # 得到最终输出的下标
            pos = ex_cumsum_bins + ex_cumsum_in_bin  # (TILE_N, )

            # scatter
            # 只写出属于当前桶的元素
            tl.store(out_ptr + pid_m * N + pos, arr, mask=matches)
            if associate_arr_ptr is not None:
                tl.store(associate_out_ptr + pid_m * N + pos, associate_arr, mask=matches)


@triton.jit
def _sweep_pair_lookback(S, row, tile, bin_id, NT, R: tl.constexpr, count):
    aggregate_mask: tl.constexpr = 1 << 30
    inclusive_mask: tl.constexpr = 1 << 31
    value_mask: tl.constexpr = (1 << 30) - 1
    offset = row * (R * NT) + bin_id * NT + tile
    tl.store(S + offset, aggregate_mask | count, cache_modifier=".cg")
    prefix = tl.zeros((), tl.uint32)
    prev = tile - 1
    while prev >= 0:
        location = row * (R * NT) + bin_id * NT + prev
        record = tl.load(S + location, volatile=True)
        while record == 0:
            record = tl.load(S + location, volatile=True)
        prefix += record & value_mask
        if (record & aggregate_mask) == aggregate_mask:
            prev -= 1
        else:
            prev = -1
    tl.store(S + offset, inclusive_mask | (prefix + count), cache_modifier=".cg")
    return prefix


@triton.jit
def _sweep_two_bin_rank(
    X, I, Y, J, GLOBAL_PREFIX, STATUS, PASSES, PASS, SHIFT,
    M, N, NT,
    TILE_N: tl.constexpr, TILE_R: tl.constexpr, k_bits: tl.constexpr,
    descending: tl.constexpr,
    identity_flags_ptr=None,
):
    tl.static_assert(TILE_N < 65536, "Each packed counter must fit in 16 bits")
    tl.static_assert(TILE_R % 2 == 0)
    R: tl.constexpr = 2 ** k_bits
    tl.static_assert(R % 2 == 0)
    pid = tl.program_id(0)
    row = pid % M
    tile = pid // M
    group = tl.program_id(1)
    cols = tile * TILE_N + tl.arange(0, TILE_N)
    valid = cols < N
    identity = tl.full((), False, tl.int1)
    if identity_flags_ptr is not None:
        identity = tl.load(identity_flags_ptr + row * PASSES + PASS) != 0
    if identity:
        if group == 0:
            values = tl.load(X + row * N + cols, valid)
            tl.store(Y + row * N + cols, values, valid)
            if I is not None:
                indices = tl.load(I + row * N + cols, valid)
                tl.store(J + row * N + cols, indices, valid)
    else:
        x = tl.load(X + row * N + cols, valid)
        key = (convert_to_uint_preverse_order(x, descending) >> SHIFT) & (R - 1)
        if I is not None:
            idx = tl.load(I + row * N + cols, valid)
        begin = group * TILE_R
        end = tl.minimum(begin + TILE_R, R)
        for b in range(begin, end, 2):
            match0 = valid & (key == b)
            match1 = valid & (key == b + 1)
            weights = match0.to(tl.uint32) | (match1.to(tl.uint32) << 16)
            aggregate = tl.sum(weights, 0)
            count0 = aggregate & 65535
            count1 = aggregate >> 16
            # Both counters and all inter-CTA states remain separate.
            before0 = _sweep_pair_lookback(STATUS, row, tile, b, NT, R, count0)
            before1 = _sweep_pair_lookback(STATUS, row, tile, b + 1, NT, R, count1)
            if aggregate != 0:
                packed_rank = tl.cumsum(weights, 0) - weights
                if count0 != 0:
                    base0 = tl.load(GLOBAL_PREFIX + row * (PASSES * R) + PASS * R + b)
                    pos0 = base0 + before0 + (packed_rank & 65535)
                    tl.store(Y + row * N + pos0, x, match0)
                    if I is not None:
                        tl.store(J + row * N + pos0, idx, match0)
                if count1 != 0:
                    base1 = tl.load(GLOBAL_PREFIX + row * (PASSES * R) + PASS * R + b + 1)
                    pos1 = base1 + before1 + (packed_rank >> 16)
                    tl.store(Y + row * N + pos1, x, match1)
                    if I is not None:
                        tl.store(J + row * N + pos1, idx, match1)


@triton.jit
def _histogram_exclusive_scan_kernel(
    hist_ptr, prefix_ptr, NUM_BINS: tl.constexpr,
    mismatch_ptr=None, identity_ptr=None, N_PASSES: tl.constexpr=1, BITS_PER_PASS: tl.constexpr=4,
):
    # radix_sort bounds each nonnegative row sum by n < 2**30.
    # Keep address arithmetic wide; only bounded counts/prefixes use INT32.
    row = tl.program_id(0).to(tl.int64)
    bins = tl.arange(0, NUM_BINS)
    counts = tl.load(hist_ptr + row * NUM_BINS + bins).to(tl.int32)
    prefix = tl.cumsum(counts, axis=0) - counts
    tl.store(prefix_ptr + row * NUM_BINS + bins, prefix.to(tl.uint32))
    if identity_ptr is not None:
        uniform = tl.sum((counts != 0).to(tl.int32), 0) == 1
        repeated = tl.full((), False, tl.int1)
        if row % N_PASSES != 0:
            bit_offset = ((row % N_PASSES) - 1) * BITS_PER_PASS
            word = tl.load(mismatch_ptr + (row // N_PASSES) * 2 + bit_offset // 32)
            repeated = (
                ((word >> (bit_offset % 32)) & (NUM_BINS - 1)) == 0
            )
        tl.store(identity_ptr + row, (uniform | repeated).to(tl.uint32))


def radix_sort(arr, k_bits=8, descending=False):
    n = arr.shape[-1]
    m = arr.numel() // n
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else (arr.itemsize * 8)

    TILE_N = 1024
    tiles_n_per_cta = 8
    CTA_TILE_N = tiles_n_per_cta * TILE_N

    num_bins = 2**k_bits
    n_passes = triton.cdiv(num_bits, k_bits)
    TILE_R = 16

    grid_n = triton.cdiv(n, CTA_TILE_N)
    grid_for_global_hist = (m * grid_n, 1, 1)

    with torch_device_fn.device(arr.device):
        global_hist = torch.zeros(
            (m, n_passes, num_bins), device=arr.device, dtype=torch.int32
        )
        # Packed mismatch words require digit fields not to straddle a word.
        if 32 % k_bits == 0:
            identity_mismatch = torch.zeros(
                (m, 2), device=arr.device, dtype=torch.uint32
            )
            identity_flags = torch.empty((m,n_passes),device=arr.device,dtype=torch.uint32)
        else:
            identity_mismatch = None
            identity_flags = None
        compute_global_hist_kernel[grid_for_global_hist](
            arr,
            global_hist,
            n_passes,
            m,
            n,
            tiles_n_per_cta,
            TILE_N,
            TILE_R,
            k_bits,
            descending,
            identity_mismatch,
            KEY_BITS=num_bits,
        )
        ex_cumsum_bins = torch.empty_like(global_hist, dtype=torch.uint32)
        _histogram_exclusive_scan_kernel[(m * n_passes,)](
            global_hist, ex_cumsum_bins, NUM_BINS=num_bins, num_warps=4,
            mismatch_ptr=identity_mismatch, identity_ptr=identity_flags,
            N_PASSES=n_passes, BITS_PER_PASS=k_bits,
        )

        # sort
        arr_in = torch.clone(arr)
        indices_in = (
            torch.arange(0, n, dtype=torch.int32, device=arr_in.device)
            .broadcast_to(arr.shape)
            .contiguous()
        )
        arr_out = torch.empty_like(arr)
        indices_out = torch.empty_like(indices_in)

        TILE_R = 8
        grid_r = triton.cdiv(num_bins, TILE_R)
        TILE_N = 2048
        grid_n = triton.cdiv(n, TILE_N)
        grid_for_sweep = (m * grid_n, grid_r)
        # Packed fields are exact only within these structural bounds.
        digit_sweep = (
            _sweep_two_bin_rank
            if TILE_N < 65536 and TILE_R % 2 == 0 and num_bins % 2 == 0
            else sweep
        )

        status = torch.empty(
            (m, num_bins, grid_n), device=arr.device, dtype=torch.uint32
        )

        sweep_kwargs = (
            {"identity_flags_ptr": identity_flags}
            if digit_sweep is _sweep_two_bin_rank else {}
        )
        for i in range(0, n_passes):
            if i == n_passes - 1:
                # Intermediate indices are bounded by n-1 < 2**30.
                # Widen on the last sweep store to preserve torch.sort's API.
                indices_out = torch.empty_like(indices_in, dtype=torch.int64)
            bit_offset = i * k_bits
            status.zero_()
            digit_sweep[grid_for_sweep](
                arr_in,
                indices_in,
                arr_out,
                indices_out,
                ex_cumsum_bins,
                status,
                n_passes,
                i,
                bit_offset,
                m,
                n,
                grid_n,
                TILE_N,
                TILE_R,
                k_bits,
                descending,
                **sweep_kwargs,
            )
            # print(f"< sorted last {bit_offset + k_bits:>2d} bits: {arr_out}")
            arr_in, arr_out = arr_out, arr_in
            indices_in, indices_out = indices_out, indices_in

    return arr_in, indices_in


@libentry()
@triton.jit()
def sort_kernel(
    in_ptr,
    out_ptr,
    out_index_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offset = tl.program_id(0) * N + cols
    in_ptr += offset
    out_ptr += offset
    out_index_ptr += offset

    if IS_FLOAT:
        mask_val = _get_finfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)
    else:
        mask_val = _get_iinfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)

    index_val = tl.arange(0, BLOCK_SIZE)

    sorted_in_val, sorted_index_val = argsort(
        in_val, index_val, 0, descending=DESCENDING
    )
    tl.store(out_ptr, sorted_in_val, mask=mask)
    tl.store(out_index_ptr, sorted_index_val, mask=mask)


def sort(inp, dim=-1, descending=False):
    # We only implement stable radix sort here
    logger.debug("GEMS SORT")
    return sort_stable(inp, stable=False, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    logger.debug("GEMS SORT.STABLE")
    # We only implement stable radix sort here
    _ = stable
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 1:
        return inp, torch.zeros_like(inp, dtype=torch.int64)

    if dim < 0:
        dim = dim + inp.ndim
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()
    else:
        inp = inp.contiguous()

    dtype = inp.dtype
    num_bits_per_pass = 1 if dtype == torch.bool else 4
    out, out_index = radix_sort(inp, num_bits_per_pass, descending)

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index
