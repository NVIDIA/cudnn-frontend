# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory compaction for Relax's four-token index groups."""

import hashlib
from pathlib import Path
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cuda.bindings import driver as cuda
from cudnn._cutlass_compat import SmemAllocator
from cudnn.frost.compiled_cache import compile_cached, template_key
from cudnn.QSA._contract import BLK_BITS

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@cute.kernel
def frost_qsa_indexed_metadata(indices, ids, counts, lo, hi, first, last, NQ: cutlass.Constexpr[int], TABLE: cutlass.Constexpr[int]):
    tid = Int32(cute.arch.thread_idx()[0])
    pair = Int32(cute.arch.block_idx()[0])
    total = Int32(indices.shape[0])
    capacity = Int32(indices.shape[1] // 4)
    smem = SmemAllocator()
    keys = smem.allocate_tensor(Int32, cute.make_layout(TABLE), byte_alignment=128)
    flags = smem.allocate_tensor(Int32, cute.make_layout(TABLE), byte_alignment=128)
    prefix = smem.allocate_tensor(Int32, cute.make_layout(9), byte_alignment=16)
    for slot in cutlass.range(tid, TABLE, 256):
        keys[slot], flags[slot] = Int32(-1), Int32(0)
    if tid < 9:
        prefix[tid] = Int32(0)
    for member in cutlass.range_constexpr(NQ):
        row = pair * NQ + member
        if tid == 0 and row < total:
            lo[row], hi[row], first[row] = Int32(0), row, row
            last[row] = cute.math.min(row + 3, total - 1)
    cute.arch.sync_threads()
    for member in cutlass.range_constexpr(NQ):
        row = pair * NQ + member
        if row < total:
            for block in cutlass.range(tid, capacity, 256):
                start = indices[Int64(row), Int64(block) * 4]
                if start >= 0 and start <= row:
                    # Mix higher bits too: starts in one packed sequence share their low two bits.
                    hashed = Uint32(start) * Uint32(2654435761)
                    slot = Int32((hashed ^ (hashed >> 16)) & Uint32(TABLE - 1))
                    inserted = Int32(0)
                    probe = Int32(0)
                    while inserted == 0 and probe < TABLE:
                        old = cute.arch.atomic_cas((keys.iterator + slot).llvm_ptr, cmp=Int32(-1), val=start, sem="relaxed", scope="cta")
                        if old == -1 or old == start:
                            cute.arch.atomic_or((flags.iterator + slot).llvm_ptr, Int32(1 << member), sem="relaxed", scope="cta")
                            inserted = Int32(1)
                        else:
                            slot = (slot + 1) & Int32(TABLE - 1)
                            probe += 1
    cute.arch.sync_threads()
    lane, warp = tid % 32, tid // 32
    lower = (Uint32(1) << Uint32(lane)) - Uint32(1)
    for base in cutlass.range(0, TABLE, 256, unroll=1):
        slot = base + tid
        key = keys[slot]
        ballot = Uint32(cute.arch.vote_ballot_sync(key >= 0))
        if lane == 0:
            prefix[warp + 1] = Int32(cute.arch.popc(ballot))
        cute.arch.sync_threads()
        offset = prefix[0]
        for other in cutlass.range(warp, unroll=1):
            offset += prefix[other + 1]
        if key >= 0:
            code = key
            if cutlass.const_expr(NQ > 1):
                code = key | (flags[slot] << Int32(BLK_BITS))  # one owner bit per batched query above the start
            ids[Int64(pair), Int64(offset + Int32(cute.arch.popc(ballot & lower)))] = code
        cute.arch.sync_threads()
        if tid == 0:
            for other in cutlass.range_constexpr(8):
                prefix[0] += prefix[other + 1]
        cute.arch.sync_threads()
    if tid == 0:
        counts[pair] = prefix[0]


frost_qsa_indexed_metadata.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(indices, ids, counts, lo, hi, first, last, stream: cuda.CUstream, NQ: cutlass.Constexpr[int], TABLE: cutlass.Constexpr[int]):
    frost_qsa_indexed_metadata(indices, ids, counts, lo, hi, first, last, NQ, TABLE).launch(grid=(ids.shape[0], 1, 1), block=(256, 1, 1), stream=stream)


def compile_metadata(arch, nq, capacity):
    total, width, groups, columns = [cute.sym_int64() for _ in range(4)]

    def fake(shape):
        return make_fake_compact_tensor(Int32, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=4)

    table = max(256, 1 << (2 * nq * capacity - 1).bit_length())
    return compile_cached(
        _host,
        fake((total, width)),
        fake((groups, columns)),
        fake((groups,)),
        fake((total,)),
        fake((total,)),
        fake((total,)),
        fake((total,)),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        nq,
        table,
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a",
        cache_key=template_key(globals(), {"arch": arch, "nq": nq, "capacity": capacity}, "compile_metadata"),
        symbol="qsa_indexed_metadata",
    )
