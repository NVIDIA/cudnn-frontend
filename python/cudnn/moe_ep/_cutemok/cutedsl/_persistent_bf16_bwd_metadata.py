# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Decoder metadata builder of the ordinary BF16 backward megakernel.

One CTA of 32 threads (one warp) builds four ABI tables (two are aliases when capacity <= macrobatch) before the
megakernel consumes them.  Geometry is runtime metadata / scalars, never a code key.
Wide (Int64) guards run once per table / minibatch; the per-chunk expert walk and the owner bucket search then
run in Int32, on values the guards have already bounded below 2**31.
The builder consumes counts, not route payload values: the schedule producer / caller must provide valid routes
for the consuming megakernel.  The peer / token and local_tokens / topk arguments remain for binding
compatibility; only the peer tensor's runtime capacity is used here, never peer / token payloads.
Device-value errors trap asynchronously; they are not synchronous Python ValueErrors.
"""

import threading
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import from_dlpack
from ._host_common import has_tail as _has_tail, persistent_executor
from ._arch import COMPILE_OPTIONS


@dsl_user_op
def _trap(*, loc=None, ip=None):
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="trap;",
        constraints="~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _warp_inclusive_sum(value: Int32) -> Int32:
    """Inclusive prefix sum over the 32 lanes (the scheduler's shuffle-up scan)."""
    lane = cute.arch.lane_idx()
    for shift in cutlass.range_constexpr(5):
        offset = Int32(1 << shift)
        previous = cute.arch.shuffle_sync_up(value, offset, mask=0xFFFFFFFF, mask_and_clamp=0)
        if lane >= offset:
            value = value + previous
    return value


@cute.jit
def _warp_sum(value: Int32) -> Int32:
    return cute.arch.shuffle_sync(_warp_inclusive_sum(value), 31)


@cute.jit
def _task_prefix(owner: Int64, saved: Int64, down: Int64, replay: Int64, wgrad: Int64, completed: cute.Tensor):
    return owner * down + cutlass.max(Int64(0), owner - saved) * replay + Int64(completed[owner]) * wgrad


@cute.jit
def _task_prefix_i32(owner: Int32, saved: Int32, down: Int32, replay: Int32, wgrad: Int32, completed: cute.Tensor):
    # Valid only for owner <= n after the wide total passed the int32 check:
    # every term is nonnegative and monotone in owner, so each fits int32.
    return owner * down + cutlass.max(Int32(0), owner - saved) * replay + completed[owner] * wgrad


class _BwdMetadata:
    def __init__(self, has_tail: bool = False, recompute_forward_context: bool = False) -> None:
        # Compile time, the megakernel's flag: H or I has a 128-wide tail, counted as one more tile.  It only
        # selects the tile-count arithmetic (two executors, one for every aligned shape and one for every shape
        # with a tail, as one more entry of the executor key); H and I stay runtime arguments.
        self.has_tail = bool(has_tail)
        # Compile time, the megakernel's flag: every minibatch has replay tasks (none is saved) and the
        # megakernel reads the owner buckets and segment experts at any capacity.
        self.recompute_forward_context = bool(recompute_forward_context)

    @cute.jit
    def __call__(
        self,
        peer: cute.Tensor,
        token: cute.Tensor,
        num_tokens: cute.Tensor,
        counts: cute.Tensor,
        pairs: cute.Tensor,
        completed: cute.Tensor,
        experts: cute.Tensor,
        buckets: cute.Tensor,
        macro: Int32,
        mini: Int32,
        local_tokens: Int32,
        topk: Int32,
        hidden: Int32,
        intermediate: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(peer, token, num_tokens, counts, pairs, completed, experts, buckets, macro, mini, local_tokens, topk, hidden, intermediate).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        peer: cute.Tensor,
        token: cute.Tensor,
        num_tokens: cute.Tensor,
        counts: cute.Tensor,
        pairs: cute.Tensor,
        completed: cute.Tensor,
        experts: cute.Tensor,
        buckets: cute.Tensor,
        macro: Int32,
        mini: Int32,
        local_tokens: Int32,
        topk: Int32,
        hidden: Int32,
        intermediate: Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        # Widen BEFORE additions/products, including ceil and task prefixes.  Every lane
        # evaluates the same guards on the same values (warp-uniform control flow).
        nrows = Int64(num_tokens[0])
        capacity = Int64(peer.shape[0])
        b = Int64(mini)
        big = Int64(macro)
        if nrows < 0 or nrows > capacity or b <= 0 or big <= 0:
            _trap()
        if big % b != 0:
            _trap()
        n = (nrows + b - 1) // b
        k = Int64(pairs.shape[0]) - 1
        if n > k:
            _trap()
        o1 = capacity > big
        if cutlass.const_expr(self.recompute_forward_context):
            o1 = Boolean(True)
        num_experts = Int32(counts.shape[0])

        # Counts: lane-strided validation and an exact total.  A lane partial above 2**31 - 1
        # means the total is too (all counts are >= 0), so the full scan's running-total trap
        # is kept; below it each partial is split in 16-bit halves whose warp sums fit Int32.
        partial = Int64(0)
        e = lane
        while e < num_experts:
            rows = counts[e]
            if rows < 0 or rows % 256 != 0:
                _trap()
            partial = partial + Int64(rows)
            e = e + Int32(32)
        if partial > Int64(0x7FFFFFFF):
            _trap()
        high = _warp_sum(Int32(partial) >> Int32(16))
        low = _warp_sum(Int32(partial) & Int32(0xFFFF))
        total = Int64(high) * Int64(65536) + Int64(low)
        if total > Int64(0x7FFFFFFF):
            _trap()
        if total != nrows:
            _trap()

        # Per minibatch, the experts it overlaps are a contiguous run starting at the first
        # expert whose rows end after the minibatch's begin: the warp walks that run 32 experts
        # at a time (row prefix by one warp scan per chunk) instead of scanning every expert per
        # minibatch, and the same chunk tells where the next minibatch's run starts.  Minibatch 0
        # walks every expert, because it registers the experts without rows.  Same tables, same
        # expert order (slots by the register ballot below each lane).
        pair_count = Int64(0)
        segment_count = Int64(0)
        if lane == Int32(0):
            pairs[0] = Int32(0)
            completed[0] = Int32(0)
        below = Int32(cute.arch.lanemask_lt())
        first = Int32(0)  # first expert whose rows end after this minibatch's begin
        first_begin = Int32(0)  # its first row
        # The macrobatch origin advances every big // b minibatches (big % b == 0 above): kept as
        # a running value instead of a 64-bit division per minibatch.
        minis_per_macro = big // b
        macro_begin = Int64(0)
        mini_in_macro = Int64(0)
        m = Int64(0)
        while m < n:
            if mini_in_macro == minis_per_macro:
                macro_begin = macro_begin + big
                mini_in_macro = Int64(0)
            mini_in_macro = mini_in_macro + Int64(1)
            mini_begin = Int64(m) * b
            mini_end = cutlass.min(mini_begin + b, nrows)
            macro_end = cutlass.min(macro_begin + big, nrows)
            # total == nrows <= capacity < 2**31 (Int32 shape), so every row
            # offset below fits Int32; only the per-mini counts stay wide.
            begin = Int32(mini_begin)
            end = Int32(mini_end)
            segment_floor = Int32(macro_begin)
            segment_ceiling = Int32(macro_end)
            chunk = first
            expert_base = first_begin
            pairs_here = Int32(0)
            segments_here = Int32(0)
            # The next minibatch begins at this one's end: the experts ending at or before it
            # (a prefix of the run, rows are in expert order) are skipped next time.
            counting = Boolean(True)
            walking = Boolean(chunk < num_experts)
            while walking:
                e = chunk + lane
                valid = e < num_experts
                rows = Int32(0)
                if valid:
                    rows = counts[e]
                ends = _warp_inclusive_sum(rows)
                expert_end = expert_base + ends
                expert_begin = expert_end - rows
                pair = valid and cutlass.max(expert_begin, begin) < cutlass.min(expert_end, end)
                segment_begin = cutlass.max(expert_begin, segment_floor)
                segment_end = cutlass.min(expert_end, segment_ceiling)
                # A segment is registered in the minibatch holding its last row.  An
                # expert without rows is registered once, in minibatch 0, so its Wgrad
                # tasks write the zero dW tiles (no external zero kernel).
                register = Boolean(False)
                if valid:
                    if segment_begin < segment_end:
                        if segment_end > begin and segment_end <= end:
                            register = Boolean(True)
                    elif rows == Int32(0) and m == Int64(0):
                        register = Boolean(True)
                registered = cute.arch.vote_ballot_sync(register)
                if o1 and register:
                    slot = segment_count + Int64(segments_here + Int32(cute.arch.popc(registered & below)))
                    if slot >= Int64(experts.shape[0]):
                        _trap()
                    experts[slot] = e
                pairs_here = pairs_here + Int32(cute.arch.popc(cute.arch.vote_ballot_sync(pair)))
                segments_here = segments_here + Int32(cute.arch.popc(registered))
                if counting:
                    done = Int32(cute.arch.popc(cute.arch.vote_ballot_sync(valid and expert_end <= end)))
                    if done > Int32(0):
                        first = chunk + done
                        first_begin = expert_base + cute.arch.shuffle_sync(ends, done - Int32(1))
                    counting = Boolean(done == Int32(32))
                expert_base = expert_base + cute.arch.shuffle_sync(ends, 31)
                chunk = chunk + Int32(32)
                # Past this minibatch's rows nothing more overlaps (minibatch 0 keeps going
                # for the experts without rows).
                walking = Boolean(chunk < num_experts and (m == Int64(0) or expert_base < end))
            pair_count = pair_count + Int64(pairs_here)
            segment_count = segment_count + Int64(segments_here)
            if pair_count > Int64(0x7FFFFFFF) or segment_count > Int64(0x7FFFFFFF):
                _trap()
            if lane == Int32(0):
                pairs[m + 1] = Int32(pair_count)
                completed[m + 1] = Int32(segment_count)
            m = m + Int64(1)
        m = n + Int64(1) + Int64(lane)
        while m < k + Int64(1):
            pairs[m] = Int32(pair_count)
            completed[m] = Int32(segment_count)
            m = m + Int64(32)

        if o1:
            q = b // 256
            i = Int64(intermediate) // 256
            h = Int64(hidden) // 256
            if cutlass.const_expr(self.has_tail):
                # Ceil, as the kernel's i_tiles and h_tiles: a 128-wide tail is a whole tile.  A floor
                # here undercounts every minibatch's tasks and the owner buckets then send tasks to the
                # wrong minibatch (only in the multi-macrobatch path, which reads the buckets).
                i = (Int64(intermediate) + 255) // 256
                h = (Int64(hidden) + 255) // 256
            # Routed tasks per minibatch: Down dgrad (with the fused SwiGLU-backward) and
            # Gate/Up dgrad; must equal the kernel's routed_tasks and the contract's
            # TaskCounts.bwd_per_minibatch.
            down = q * i + q * h
            replay = q * i  # Replay Gate+Up only (no replay SwiGLU)
            wgrad = Int64(3) * i * h
            width = Int64(1)
            while width * 2 <= down:
                width = width * 2
            bucket_capacity = Int64(buckets.shape[0])
            if down <= 0 or replay < 0 or wgrad <= 0 or bucket_capacity <= 0:
                _trap()
            saved = cutlass.min(n, big // b)
            if cutlass.const_expr(self.recompute_forward_context):
                saved = Int64(0)
            # completed[] was written by lane 0 above; the barrier makes it visible to the warp.
            cute.arch.sync_threads()
            tasks = _task_prefix(n, saved, down, replay, wgrad, completed)
            if tasks > Int64(0x7FFFFFFF):
                _trap()
            # Each of the three must fit Int32 as well.
            if down > Int64(0x7FFFFFFF) or replay > Int64(0x7FFFFFFF) or wgrad > Int64(0x7FFFFFFF):
                _trap()
            count = (tasks + width - 1) // width
            if count > bucket_capacity:
                _trap()
            # Every searched prefix and bucket start is below tasks < 2**31,
            # and count <= bucket_capacity is an Int32 shape.
            minibatches = Int32(n)
            saved_i32 = Int32(saved)
            down_i32 = Int32(down)
            replay_i32 = Int32(replay)
            wgrad_i32 = Int32(wgrad)
            width_i32 = Int32(width)
            bucket_count = Int32(count)
            # Bucket b's owner is the last minibatch o < n with prefix(o) <= b * width (prefix is
            # nondecreasing): each lane takes a contiguous run of buckets, finds the owner of its
            # first bucket by bisection and then moves forward like the sequential search.
            per_lane = (bucket_count + Int32(31)) // Int32(32)
            bucket = lane * per_lane
            bucket_end = cutlass.min(bucket + per_lane, bucket_count)
            owner = Int32(0)
            if bucket < bucket_end:
                high_owner = cutlass.max(Int32(0), minibatches - Int32(1))
                start = bucket * width_i32
                while owner < high_owner:
                    middle = (owner + high_owner + Int32(1)) >> Int32(1)
                    if _task_prefix_i32(middle, saved_i32, down_i32, replay_i32, wgrad_i32, completed) <= start:
                        owner = middle
                    else:
                        high_owner = middle - Int32(1)
            while bucket < bucket_end:
                start = bucket * width_i32
                searching = Boolean(owner + Int32(1) < minibatches)
                while searching:
                    next_start = _task_prefix_i32(owner + Int32(1), saved_i32, down_i32, replay_i32, wgrad_i32, completed)
                    if next_start <= start:
                        owner = owner + Int32(1)
                        searching = Boolean(owner + Int32(1) < minibatches)
                    else:
                        searching = Boolean(False)
                buckets[bucket] = owner
                bucket = bucket + Int32(1)
            bucket = bucket_count + lane
            while bucket < buckets.shape[0]:
                buckets[bucket] = cutlass.max(Int32(0), minibatches - Int32(1))
                bucket = bucket + Int32(32)
            segment = segment_count + Int64(lane)
            while segment < Int64(experts.shape[0]):
                experts[segment] = Int32(-1)
                segment = segment + Int64(32)


_CACHE = {}
_LOCK = threading.Lock()


def _dynamic(tensor):
    # Contiguous singleton tensors may carry arbitrary strides; normalize only
    # the view. No source bytes, offsets or pointer identities are changed.
    return from_dlpack(tensor.as_strided(tensor.shape, (1,)), assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=0)


def make_bwd_metadata_args(inputs, state, *, config, stream, launch: bool = False):
    """CuTe tensors and DSL scalars for ``cute.compile`` (``launch=False``), or the torch views
    and plain scalars of a call to the compiled TVM FFI function (``launch=True``), as the
    FWD / BWD mega hosts do."""

    tensors = (inputs.schedule_peer_rank, inputs.schedule_peer_token_idx, inputs.num_tokens, inputs.tokens_per_expert, *state.metadata)
    if launch:
        scalar = int
        tensors = tuple(t if t.stride() == (1,) else t.as_strided(t.shape, (1,)) for t in tensors)
    else:
        scalar = Int32
        tensors = tuple(_dynamic(t) for t in tensors)
    return (
        *tensors,
        scalar(config.macrobatch_size),
        scalar(config.minibatch_size),
        scalar(config.num_local_tokens),
        scalar(config.topk),
        scalar(config.routed_hidden_size),
        scalar(config.intermediate_size),  # routed per-minibatch / Wgrad tasks
        stream if isinstance(stream, cuda.CUstream) else cuda.CUstream(stream.cuda_stream),
    )


def prepare_bwd_metadata_launch(inputs, state, *, config, stream):
    """Executor and launch arguments without the trailing stream; the builder still runs on every
    call (schedule values are never cached)."""
    from .persistent_bf16_bwd import _current_cuda_context_key

    # The BF16 ABI, CTA32 and the resolved target (sm_100a / sm_103a) are the only static modes; geometry is runtime.
    # The kernel counts the routed tasks only: the routed hidden and the intermediate size are its extents.
    has_tail = _has_tail(config.routed_hidden_size, config.intermediate_size)
    recompute_forward_context = bool(getattr(config, "recompute_forward_context", False))
    key = (inputs.num_tokens.device.index, _current_cuda_context_key(), has_tail, recompute_forward_context)
    with _LOCK:
        executor = _CACHE.get(key)
        if executor is None:
            executor = persistent_executor(
                ("bwd_metadata", COMPILE_OPTIONS, ("has_tail", has_tail), ("recompute_forward_context", recompute_forward_context)),
                lambda: cute.compile(
                    _BwdMetadata(has_tail, recompute_forward_context),
                    *make_bwd_metadata_args(inputs, state, config=config, stream=stream),
                    options=COMPILE_OPTIONS,
                ),
            )
            _CACHE[key] = executor
    return executor, make_bwd_metadata_args(inputs, state, config=config, stream=stream, launch=True)[:-1]


def fill_bwd_metadata_slots(args, schedule, alias_slots):
    """Fill the caller-owned slots of the arguments from prepare_bwd_metadata_launch: the four
    schedule tensors (peer_rank, peer_token_idx, num_tokens, tokens_per_expert) and the num_tokens
    aliases in the decoder metadata.  Positions follow make_bwd_metadata_args."""
    for slot, tensor in enumerate(schedule):
        args[slot] = tensor if tensor.stride() == (1,) else tensor.as_strided(tensor.shape, (1,))
    for slot in alias_slots:
        args[slot] = args[2]
    return args


def clear_bwd_metadata_cache():
    # Caller has established launch quiescence, same as the mega cache.
    with _LOCK:
        _CACHE.clear()
