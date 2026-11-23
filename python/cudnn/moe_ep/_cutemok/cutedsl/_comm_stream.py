# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""The communication CTAs' task stream loop, shared by the megakernels' forward and backward.

Every communication warp works through one linear stream of blocks (rows x columns of one macrobatch, the caller's
``block_shape``) in dependency order: it takes its first block statically and every later one by an atomic add on a
global head, waits until the block's producer is done, loads the block into one of its own buffers in the arena
(MXFP8: two of 16 KiB, the next block's load in flight while the current one is processed; BF16: one of 32 KiB),
and processes it.  No buffer is shared between warps and no warp has a fixed role: what a block is, what it waits
for, how it is loaded, what is done with it once it landed and which counter it arrives on come from the caller's
hooks (``run_stream``).

Private device module of the megakernels.  Every size, count and index is a runtime value.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from ._bwd_device_helpers import _nanosleep_compat
from ._host_common import TAIL_WIDTH
from ._mxfp8_quant import _store_shared_v4_words
from ._tma_1d import tma_load_1d_raw, tma_store_1d_raw, zero_shared_bulk

# The task stream (``run_stream``).  The comm CTA's warps 0 .. STREAM_WARPS - 1 each own STREAM_WARP_BYTES of the arena
# and one mbarrier per buffer (one arrival per lane and phase) in the storage struct; a claim takes STREAM_GROUP_BLOCKS
# consecutive column blocks of one tile row (one atomic per group, the group's readiness read once).  The constants
# below are the MXFP8 geometry -- STREAM_BUFFERS buffers of STREAM_ROWS x STREAM_COLUMNS BF16 (16 KiB each), whose
# quantization writes a pulled block's normal and transposed e4m3 and both scale layouts straight to global memory
# (``_stream_quantize``: STREAM_ROW_TILES_PER_SCALE_TILE 32-row tiles share one 128-row scale tile); the BF16 hooks
# take one 32 KiB buffer holding a comm_tile_shape tile (``block_shape``).
STREAM_ROWS = 32
STREAM_COLUMNS = 256  # a hidden that is a multiple of 256 fills every block; a 128-wide H tail (the has_tail
#                       variant) leaves the last column block half full (_stream_row_bytes)
STREAM_BUFFERS = 2
STREAM_GROUP_BLOCKS = 4
STREAM_WARPS = 7
# 1 KiB: a warp's STREAM_ROWS rows of 1 KiB hold its two 16 KiB buffers
STREAM_ROW_BYTES = STREAM_BUFFERS * STREAM_COLUMNS * 2
STREAM_WARP_BYTES = STREAM_ROWS * STREAM_ROW_BYTES
STREAM_MBARRIERS = STREAM_WARPS * STREAM_BUFFERS  # the most a CTA needs (two buffers per warp)
STREAM_PULL, STREAM_PUSH = 0, 1  # a block's queue: a dispatch pull or a combine push
WARP_LANES = 32
POLL_SLEEP_NS = 64  # between two readiness polls of a block whose producer has not finished (nanosleep)
# The has_tail kernels' tail code is written for a tail of exactly TAIL_WIDTH: half a stream column block (the host
# admits only H that are whole TAIL_WIDTH halves, _host_common.has_tail).
assert TAIL_WIDTH == 128 and STREAM_COLUMNS == 2 * TAIL_WIDTH


@dsl_user_op
def _fence_proxy_async_global(*, loc=None, ip=None) -> None:
    """``fence.proxy.async.global``: order this thread's generic-proxy global stores before async-proxy
    reads of the same bytes.  The quantizing threads store e4m3 rows and scale bytes to global memory by plain
    stores and the GEMMs load them by TMA (tensor loads, async proxy): each thread fences, the warp (the
    backward's pipeline: the block) syncs, then one thread releases the counter."""
    llvm.inline_asm(None, [], "fence.proxy.async.global;", "~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def _red_release_add_gpu(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mul_hi_u32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """``mul.hi.u32``."""
    return Int32(
        llvm.inline_asm(
            cutlass.cutlass_dsl.T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "mul.hi.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _reciprocal_u32(divisor: Int32) -> Int32:
    """floor((2^32 - 1) / divisor) as a u32 bit pattern: one division, computed once per warp."""
    return Int32(Int64(0xFFFFFFFF) // Int64(divisor))


@cute.jit
def _divide_u32(n: Int32, divisor: Int32, reciprocal: Int32) -> Int32:
    """n / divisor for 0 <= n < 2^31 without a division: ``mul.hi`` by the reciprocal is the quotient or one
    less (the stream decodes its block numbers with these instead of a runtime division per block)."""
    q = _mul_hi_u32(n, reciprocal)
    if n - q * divisor >= divisor:
        q = q + Int32(1)
    return q


@cute.jit
def _stream_row_bytes(hidden_size: Int32, column_tile: Int32, has_tail: cutlass.Constexpr = False, columns: cutlass.Constexpr = STREAM_COLUMNS) -> Int32:
    """Bytes of one row of column block ``column_tile`` of ``columns`` BF16 columns (compile time).  STREAM_COLUMNS:
    the full block, or with a 128-wide H tail (``has_tail``, compile time) TAIL_WIDTH columns in the last block -- a
    bulk copy of the whole block would run into the next row; a choice between the two constants, made once per row
    copy.  Wider blocks (the BF16 tile shapes): the columns left in the row, at most ``columns``."""
    if cutlass.const_expr(columns != STREAM_COLUMNS):
        return cutlass.min(Int32(columns), hidden_size - column_tile * Int32(columns)) * Int32(2)
    if cutlass.const_expr(has_tail):
        row_bytes = Int32(STREAM_COLUMNS * 2)
        if (column_tile + Int32(1)) * Int32(STREAM_COLUMNS) > hidden_size:
            row_bytes = Int32(TAIL_WIDTH * 2)
        return row_bytes
    return Int32(STREAM_COLUMNS * 2)


@cute.jit
def pull_row(
    buffer: cute.Pointer,
    mbarrier: cute.Pointer,
    lane: Int32,
    column_tile: Int32,
    peer_rank: Int32,
    source_row: Int32,
    hidden_size: Int32,
    peer_buffer: Int64,
    has_tail: cutlass.Constexpr = False,
    columns: cutlass.Constexpr = STREAM_COLUMNS,
) -> None:
    """Lane ``lane`` pulls row ``lane`` of a block of ``columns`` columns into ``buffer`` (rows ``columns`` x 2 B
    apart) on ``mbarrier`` (one arrival per lane and phase): one bulk copy of row ``source_row`` (the schedule's
    route index divided by topk, by the caller) of the peer's buffer at ``peer_buffer`` (the caller's peer-table
    read; unused for a padding row), or for a padding row zeros fenced against the async proxy that later reads or
    writes the row (BF16: the block's bulk copy out; MXFP8: the next TMA load into the buffer) --
    16 B stores starting at chunk ``lane`` (the lanes' stores spread over the row's bank groups) for STREAM_COLUMNS,
    one ``st.bulk`` for a wider row.  Every pull of the streams (dispatch x, the backward's dY and replayed x)."""
    chunk_bytes = _stream_row_bytes(hidden_size, column_tile, has_tail, columns)
    row_address = buffer + lane * Int32(columns * 2)
    if peer_rank >= Int32(0):
        source = peer_buffer + Int64(source_row) * Int64(hidden_size) * Int64(2) + Int64(column_tile) * Int64(columns * 2)
        cute.arch.mbarrier_arrive_and_expect_tx(mbarrier, chunk_bytes)
        tma_load_1d_raw(row_address, source, mbarrier, chunk_bytes)
    else:
        if cutlass.const_expr(columns == STREAM_COLUMNS):
            for step in cutlass.range_constexpr(STREAM_COLUMNS * 2 // 16):
                chunk = (Int32(step) + lane) & Int32(STREAM_COLUMNS * 2 // 16 - 1)
                _store_shared_v4_words(row_address.toint() + chunk * Int32(16), [Uint32(0)] * 4)
            cute.arch.fence_view_async_shared()
        else:
            zero_shared_bulk(row_address, columns * 2)
            cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.mbarrier_arrive(mbarrier)


@cute.jit
def load_local_row(
    buffer: cute.Pointer,
    mbarrier: cute.Pointer,
    lane: Int32,
    local_base: Int64,
    row: Int32,
    column_tile: Int32,
    peer: Int32,
    hidden_size: Int32,
    has_tail: cutlass.Constexpr = False,
    columns: cutlass.Constexpr = STREAM_COLUMNS,
) -> None:
    """A push block's read: lane ``lane`` loads row ``row + lane`` of the local row-major BF16 buffer at
    ``local_base`` (``hidden_size`` columns) into its row of ``buffer`` on ``mbarrier``, one bulk copy; a padding
    lane (``peer`` -1, nobody receives the row) arrives with no bytes."""
    chunk_bytes = _stream_row_bytes(hidden_size, column_tile, has_tail, columns)
    if peer >= Int32(0):
        source = local_base + Int64(row + lane) * Int64(hidden_size) * Int64(2) + Int64(column_tile) * Int64(columns * 2)
        cute.arch.mbarrier_arrive_and_expect_tx(mbarrier, chunk_bytes)
        tma_load_1d_raw(buffer + lane * Int32(columns * 2), source, mbarrier, chunk_bytes)
    else:
        cute.arch.mbarrier_arrive(mbarrier)


@cute.jit
def store_row(
    buffer: cute.Pointer,
    lane: Int32,
    destination_row: Int64,
    column_tile: Int32,
    hidden_size: Int32,
    has_tail: cutlass.Constexpr = False,
    columns: cutlass.Constexpr = STREAM_COLUMNS,
) -> None:
    """Lane ``lane``'s row of ``buffer`` to column block ``column_tile`` of the row-major BF16 row at
    ``destination_row`` (``hidden_size`` columns), one bulk store; the caller commits and waits."""
    chunk_bytes = _stream_row_bytes(hidden_size, column_tile, has_tail, columns)
    tma_store_1d_raw(destination_row + Int64(column_tile) * Int64(columns * 2), buffer + lane * Int32(columns * 2), chunk_bytes)


@cute.jit
def _group_block(group: Int32, groups: Int32, reciprocal_groups: Int32) -> tuple[Int32, Int32]:
    """(stream task, first column block) of group ``group``: task ``group / groups``, column block
    STREAM_GROUP_BLOCKS x (group % groups); ``reciprocal_groups`` is ``_reciprocal_u32(groups)``."""
    task = _divide_u32(group, groups, reciprocal_groups)
    return task, (group - task * groups) * Int32(STREAM_GROUP_BLOCKS)


@cute.jit
def _group_end(column: Int32, column_blocks: Int32, reciprocal_group: Int32) -> Int32:
    """One past the last column block of the group of column block ``column``: the next multiple of
    STREAM_GROUP_BLOCKS, at most ``column_blocks``; ``reciprocal_group`` is ``_reciprocal_u32(STREAM_GROUP_BLOCKS)``."""
    group_blocks = Int32(STREAM_GROUP_BLOCKS)
    return cutlass.min((_divide_u32(column, group_blocks, reciprocal_group) + Int32(1)) * group_blocks, column_blocks)


def block_shape(rows: int, columns: int, buffers: int) -> tuple:
    """A hooks class's ``_stream_block``: (rows, columns, buffers) -- a warp's STREAM_WARP_BYTES as ``buffers`` (1 or 2)
    buffers, each holding one block of ``rows`` x ``columns`` BF16, a lane per row."""
    assert buffers in (1, 2) and rows * columns * 2 * buffers == STREAM_WARP_BYTES
    assert rows <= WARP_LANES and columns % STREAM_COLUMNS == 0
    return (rows, columns, buffers)


@cute.jit
def _lane_entry(table: cute.Tensor, index: Int32, lane: Int32, block_rows: cutlass.Constexpr) -> Int32:
    """``table[index]``, the schedule entry of this lane's row; -1 (a padding row: nothing moved) for a lane past the
    block's ``block_rows`` rows (compile time), which reads nothing."""
    if cutlass.const_expr(block_rows == WARP_LANES):
        entry = table[index]
    else:
        entry = Int32(-1)
        if lane < Int32(block_rows):
            entry = Int32(table[index])
    return entry


@cute.jit
def run_stream(
    self,
    operands: list,
    schedule_peer_rank: cute.Tensor,
    schedule_peer_token_idx: cute.Tensor,
    head: cute.Pointer,
    arena: cute.Pointer,
    stream_mbarriers: cute.Pointer,
    blocks: Int32,
    groups: Int32,
    reciprocal_groups: Int32,
    reciprocal_group: Int32,
    column_blocks: Int32,
    comm_cta: Int32,
    num_comm_sms: Int32,
    phase_bits: Int32,
) -> Int32:
    """The communication task stream's loop, shared by the forward (``_forward_stream``) and the backward
    (``_backward_stream``): warps 0 .. STREAM_WARPS - 1 of every comm CTA (warp 7 idles) work through the
    stream's blocks, numbered per group of STREAM_GROUP_BLOCKS consecutive column blocks of one stream task:
    ``blocks`` is the number of groups (tasks x ``groups``), group ``g`` is column blocks ``STREAM_GROUP_BLOCKS x
    (g % groups)`` .. of task ``g / groups`` (``_group_block`` / ``_group_end``).  Warp ``w`` takes group ``w x
    num_comm_sms + comm_cta`` first, then its group's next column block (same rows, already ready), else
    ``atomic_add(head) + warps_total`` (``head`` zero at launch).  With two buffers the next block is issued into
    the other buffer before the current one is processed when its producer is done; otherwise (and always with
    one buffer) at the top of the next iteration, after the pending arrivals were released when the producer is
    not done (a waiting warp never holds back anything).  One issue site per case and one decode: the first
    block is taken by an iteration with no current block.  Arrivals are batched:
    the warp keeps one pending counter address and releases the count when the next block's address differs
    (``red.release.gpu``).  ``phase_bits`` carries the warp's buffer-barrier phases in and out (the backward
    calls this once, its one stream covering every macrobatch).

    What a block is comes from the caller's hooks, which read their operands from ``operands`` (a list whose
    layout the caller owns): ``_stream_decode`` (task -> queue, macrobatch, tile of ``block_rows`` rows), ``_stream_ready_lane0``
    (lane 0's answer whether the block's producer is done), ``_stream_issue`` (the block into a buffer with
    the lane's schedule entries read here), ``_stream_arrival`` (the counter word the processed block arrives
    on, 0 for none), ``_stream_process`` (the block's bytes, landed), ``_stream_drain`` (every lane, before the
    pending arrivals of a queue are released: complete what the counted blocks wrote, when ``_stream_process`` left
    that open) and ``_stream_schedule_row`` (the schedule index of the block's first row; the lane's entries are those
    of that row plus ``lane``).  The
    backward's hooks turn ``macro`` and ``row`` into the schedule row (``_stream_schedule_row``).
    ``self._stream_block`` (compile time, ``block_shape``): (rows, columns, buffers) -- MXFP8 two 16 KiB buffers of
    STREAM_ROWS x STREAM_COLUMNS (the next block loads while the current one is quantized), BF16 one 32 KiB buffer
    holding a whole communication tile (the next tile's decode, readiness and schedule reads overlap the current
    one's load and stores); lanes past the rows read no schedule entry.
    ``reciprocal_groups`` /
    ``reciprocal_group`` are the caller's ``_reciprocal_u32`` of ``groups`` / STREAM_GROUP_BLOCKS (the per-block
    divisors; no division per block)."""
    thread = cute.arch.thread_idx()[0]
    warp = thread >> Int32(5)
    lane = thread & Int32(31)
    warps_total = Int32(STREAM_WARPS) * num_comm_sms
    # Compile time (``block_shape``): the block's rows and the warp's buffer count; its columns are the hooks' business.
    block_rows = self._stream_block[0]
    buffers = self._stream_block[2]
    buffer_bytes = Int32(STREAM_WARP_BYTES // buffers)
    new_phase_bits = phase_bits
    if warp < Int32(STREAM_WARPS):
        warp_base = arena + warp * Int32(STREAM_WARP_BYTES)
        mbar_base = stream_mbarriers + warp * Int32(buffers)  # one transaction barrier per buffer
        pending = Int32(0)
        pending_address = Int64(0)
        pending_queue = Int32(STREAM_PULL)  # the queue of the pending arrivals' blocks
        # The current block, in buffer ``current`` (issued unless ``issued`` is 0); -1 before the first one.
        block = Int32(-1)
        current = Int32(buffers - 1)  # two buffers: the first block goes to buffer 0, the "other" one
        issued = Int32(1)
        known_ready = Int32(0)  # lane 0's readiness answer for the current block when it was taken
        queue = Int32(STREAM_PULL)
        macro = Int32(0)
        row = Int32(0)
        column = Int32(0)
        while block < blocks:
            other = Int32(0)  # one buffer: every block goes through it
            if cutlass.const_expr(buffers == 2):
                other = Int32(1) - current
            # A current block not issued yet -- one buffer: every block, two: one whose producer was not done when it
            # was taken -- waits for its producer (holding no pending arrival, released below), then is issued.
            if issued == Int32(0):
                ready = known_ready  # the answer taken with the block (one buffer: usually 1, no second poll)
                while ready == Int32(0):
                    ready = cute.arch.shuffle_sync(self._stream_ready_lane0(operands, queue, macro, row), 0)
                    if ready == Int32(0):
                        _nanosleep_compat(POLL_SLEEP_NS)
                cute.arch.sync_warp()
                schedule_row = self._stream_schedule_row(operands, macro, row) + lane
                self._stream_issue(
                    operands,
                    queue,
                    row,
                    column,
                    warp_base + current * buffer_bytes,
                    mbar_base + current,
                    lane,
                    _lane_entry(schedule_peer_rank, schedule_row, lane, block_rows),
                    _lane_entry(schedule_peer_token_idx, schedule_row, lane, block_rows),
                )
            # The next block: the current group's next column block (same rows, already ready), else a new group --
            # the warp's first one dealt statically, every later one by the atomic.
            n_block = block
            n_column = column + Int32(1)
            n_queue = queue
            n_macro = macro
            n_row = row
            n_ready = Int32(1)
            group_end = _group_end(column, column_blocks, reciprocal_group)
            new_group = Int32(0)
            if n_column >= group_end:
                new_group = Int32(1)
            if block < Int32(0):
                new_group = Int32(1)
            if new_group == Int32(1):
                n_block = warp * num_comm_sms + comm_cta
                if block >= Int32(0):
                    taken = Int32(0)
                    if lane == Int32(0):
                        taken = cute.arch.atomic_add(head, Int32(1), sem="relaxed", scope="gpu")
                    n_block = cute.arch.shuffle_sync(taken, 0) + warps_total
                if n_block < blocks:
                    n_task, n_column = _group_block(n_block, groups, reciprocal_groups)
                    n_queue, n_macro, n_tile = self._stream_decode(operands, n_task)
                    n_row = n_tile * Int32(block_rows)
                    n_ready = cute.arch.shuffle_sync(self._stream_ready_lane0(operands, n_queue, n_macro, n_row), 0)
            # Issue it into the other buffer now when its producer is done, else after the current one (always after
            # it with one buffer: the next iteration's top waits for the current block's stores to read the buffer).
            n_issued = Int32(0)
            if cutlass.const_expr(buffers == 2):
                if n_block < blocks:
                    if n_ready == Int32(1):
                        cute.arch.sync_warp()
                        n_schedule_row = self._stream_schedule_row(operands, n_macro, n_row) + lane
                        self._stream_issue(
                            operands,
                            n_queue,
                            n_row,
                            n_column,
                            warp_base + other * buffer_bytes,
                            mbar_base + other,
                            lane,
                            _lane_entry(schedule_peer_rank, n_schedule_row, lane, block_rows),
                            _lane_entry(schedule_peer_token_idx, n_schedule_row, lane, block_rows),
                        )
                        n_issued = Int32(1)
            if block >= Int32(0):
                # Process the current block; a change of arrival word releases the pending arrivals first.
                address = self._stream_arrival(operands, queue, macro, row)
                if pending > Int32(0):
                    if address != pending_address:
                        self._stream_drain(operands, pending_queue)
                        _fence_proxy_async_global()
                        cute.arch.sync_warp()
                        if lane == Int32(0):
                            _red_release_add_gpu(pending_address, pending)
                        pending = Int32(0)
                if address != Int64(0):
                    pending_address = address
                    pending_queue = queue
                    pending = pending + Int32(1)
                cute.arch.mbarrier_wait(mbar_base + current, (new_phase_bits >> current) & Int32(1))
                new_phase_bits = new_phase_bits ^ (Int32(1) << current)
                self._stream_process(operands, queue, macro, row, column, warp_base + current * buffer_bytes, lane, schedule_peer_rank, schedule_peer_token_idx)
            # A next block that waits for its producer: release what this warp counted before it waits.
            waits = Int32(0)  # the next block's producer is not done: this warp waits at the next iteration's top
            if n_block < blocks:
                if cutlass.const_expr(buffers == 2):
                    waits = Int32(1) - n_issued
                else:
                    waits = Int32(1) - n_ready
            if waits == Int32(1):
                if pending > Int32(0):
                    self._stream_drain(operands, pending_queue)
                    _fence_proxy_async_global()
                    cute.arch.sync_warp()
                    if lane == Int32(0):
                        _red_release_add_gpu(pending_address, pending)
                    pending = Int32(0)
            current = other
            issued = n_issued
            known_ready = n_ready
            block = n_block
            queue = n_queue
            macro = n_macro
            row = n_row
            column = n_column
        if pending > Int32(0):
            self._stream_drain(operands, pending_queue)
            _fence_proxy_async_global()
            cute.arch.sync_warp()
            if lane == Int32(0):
                _red_release_add_gpu(pending_address, pending)
    # Every warp has released its pending arrivals (above) before the CTA goes on (the backward's thread 0 waits
    # for arrivals this CTA's own pushes count towards).  The buffers' reuse needs no CTA barrier: each warp's
    # buffers are its own, and the hooks wait for the bulk stores to have read a buffer before it is loaded again
    # (MXFP8 in _stream_process, BF16 at the next _stream_issue); the full wait completes the last stores (the
    # pushes' peer writes) before the kernel ends.
    cute.arch.cp_async_bulk_wait_group(0)
    cute.arch.sync_threads()
    return new_phase_bits
