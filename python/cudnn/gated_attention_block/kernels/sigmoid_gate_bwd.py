# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""B3 -- the sigmoid-gate BACKWARD of the gated attention block, one pass.

The forward's stage (5) is ``O_gated = O * sigmoid(GATE)`` (``elementwise.py``).
Its backward, given ``dO_gated`` from the out-projection's dgrad (B2)::

    t  = tanh(g / 2)                      one MUFU per element
    s  = 0.5 * t + 0.5                    == sigmoid(g)
    dO = dO_gated * s                     compact [T, H, D]; MAY alias dO_gated
    dG = (dO_gated * O) * 0.25 * (1 - t) * (1 + t)   == dO_gated * O * s * (1 - s)
    Og = O * s                            OPTIONAL third output (dW_o's operand)

``dG`` is written straight into the GATE band of the ``[T, N]`` ``dqkvg``
scratch (token stride ``N``, head stride ``D``) and never touched again: the
norm backward walks only the Q / K / V bands. ``GATE`` is read from the
``proj_slab`` GATE band or from a compact saved copy -- every one of the six
operands carries its OWN symbolic token stride, so no repack exists anywhere.

**Why ``0.25 * (1 - t) * (1 + t)`` and not ``s * (1 - s)``.** They are equal
algebraically, but ``s * (1 - s)`` underflows in fp32 at ``|g| = 17.0`` while the
tanh form holds to ``|g| = 18.5``; the gradient it drops beyond that is
``<= 4e-8 * |dO_gated * O|``, below any bf16 tolerance (computed in fp64 on the host). And
at ``|g| >= 30`` ``tanh(15)`` is exactly ``1.0f``, so ``dG`` is EXACTLY zero and
``dO`` is exactly ``dO_gated`` / ``0`` -- pinned by
``test_gate_bwd_saturation_is_exact``.

**Dead rows.** With ``seq_lens`` bound, ``dead = (token % s) >= seq_lens[token // s]``
and ``dO`` / ``dG`` / ``Og`` are SELECTED to exact ``0`` and STORED: never
``* 0`` (the ``O`` residue of a dead row may be NaN, and ``NaN * 0`` is NaN) and
never skipped (the buffer would keep stale rows). With ``seq_lens`` None the
select and both integer divides are folded OUT at trace time
(``test_gate_bwd_dense_artifact_matches_the_seq_lens_one_on_live_rows``).

**Shape.** Bandwidth-bound and shaped like its forward twin: a lane owns 16 B
of a ``[D]`` row (``LANES = D // 8``, one ``ld.global.v4`` per operand per
chunk), 128 threads, ``rows_per_group`` rows per lane group, two passes (every
load first, then math + stores). Tail rows clamp their loads to the last valid
row and skip every store, so a ragged last CTA costs a redundant read and never
a wild write. Live fp32 in pass 1 is 24 per row against the forward's 16, so
the default is ``rows_per_group = 1``; ``2`` is an A/B knob (the norm kernel is
register-bound and 1 beat 2 there with more live state, ``qk_norm_rope.py``).

Scalar sigmoid (the tanh identity), not the packed ``fmul2`` / ``ffma2``
helpers: those need sm_100+ and this kernel runs, and is unit-tested, on an
A100; it has ~57x MUFU headroom at Rubin's HBM rate.

**Optional ``delta`` output -- the SDPA backward's ``dot_do_o`` pre-pass, fused.**
With ``has_delta`` the kernel also writes ``delta[b, h, q] = sum_d O * dO`` as fp32
into a ``[B, H, S_pad]`` buffer -- the RAW row dot the Rubin d=256 backward chain
computes in its own first launch (``sdpa/bwd/kernels/bprop_chain_common.py::dot_do_o_kernel``;
``attn_scale`` is applied in the main kernel) -- so the chain can skip that launch
and its second read of ``O`` and ``dO`` (``SdpaBwdDslSm107(external_delta=True)``).
Two facts make the result BITWISE the chain's own, which is what lets a fused block
pin its gradients ``torch.equal`` against the unfused one:

* **the operand is the STORED ``dO``**: ``dot_do_o`` reads the bf16 / fp16 ``dO`` this
  kernel wrote, so the product uses ``dO`` rounded to the io dtype and widened back
  (the packed words of the store, unpacked), never the fp32 value;
* **the reduction ORDER is ``dot_do_o``'s.** That kernel gives a row to 8 threads;
  thread ``j`` owns columns ``8j + 64c + kk`` (``c`` over the ``D / 64`` chunks,
  ``kk < 8``) and chains ``acc = acc + O * dO`` sequentially from ``0.0`` over them, then
  the 8 partials are summed by a butterfly (xor 4, 2, 1).  Here a lane owns 8
  consecutive columns, so thread ``j``'s chain runs through lanes ``j, j+8, j+16,
  j+24`` (per 16-B chunk) in that order: the chain is handed lane to lane by
  ``shfl.idx`` (``D / 64`` rounds of 8 fma each, every lane computing, the owning
  group keeping), then the same butterfly over lanes ``24..31`` and ONE lane stores.
  Same fp32 operations in the same order = the same bits (pinned on every CUDA
  device by ``test_gate_bwd_delta_is_bitwise_the_chains_dot_do_o``).

Rows past the real length of a sequence, ``q in [S, S_pad)``, are written as exact
zeros by the row at the sequence's last token (what ``dot_do_o``'s unconditional
tile store leaves there); a dead row (``seq_lens``) gets a SELECTED zero.  Requires
``d % 64 == 0`` (``dot_do_o``'s chunk) and ``s`` at launch (``T == B * s``; the
position decides the delta column).

SMEM buffer table: NONE (no SMEM). Barrier table: NONE (no mbarrier, no named
barrier, no TMA). Nothing here can hang.

**Bytes.** ``moved_bytes`` counts 3 reads + 2 writes (+ 1 write with ``Og``) of
``T * H * D`` elements: 80 KiB / 96 KiB per token at H=32, D=256 (bf16) --
S=32K 2.68 / 3.22 GB -> 209 / 251 us at the 12846 GB/s Rubin pin (+ 4 B per
row with ``delta``: 128 B per token at H=32, 0.1 %).

**Footguns inherited from the forward kernels.** (1) Never ``.view()`` a
column slice of the ``[T, N]`` slab on the kernel path -- the caller hands
``[T, H, D]`` views of the bands (``slab[:, lo:hi].view(t, h, d)`` is fine on
a ``[T, W]`` slice; merging T with the columns is not). (2) Every operand is
traced with ``assumed_align=16``: a band base ``qkvg_offsets[i] * 2 B`` must
be 16-B aligned -- guaranteed by ``QKVG_TILE_ALIGN = 64`` elements (128 B).
``run_sigmoid_gate_bwd`` refuses, on the host, any ``[T, H, D]`` operand off
that 16-B grid (``_check_row_layout``: token stride a multiple of 8 elements,
head stride D, unit element stride, 16-B base) and any operand off the launch
device (``_check_one_cuda_device``); the token stride is SYMBOLIC in the
artifact, so nothing past the host would.
(3) Strides are ``Int32`` at the tvm-ffi boundary: a tensor whose token stride
times T exceeds 2^31 elements is the DSL's limit, not this kernel's.
"""

from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.experimental import primitives as nvvm

from cudnn.frost.device import current_device
from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16, opaque_f32_zero
from cudnn.frost.tile_dsl.tma import ld_global, ld_global_v4, st_global, st_global_v4

from .elementwise import validate_shape
from .qk_norm_rope import ACCESS_BYTES, ELEMS_PER_ACCESS, fake_rowmajor_dynamic_token_stride, lanes_per_row, vec_chunks

DEFAULT_THREADS_PER_CTA = 128
# 24 live fp32 per row in pass 1 (three operands) against the forward's 16:
# start at one row per lane group and A/B 2 (see the module docstring).
DEFAULT_ROWS_PER_GROUP = 1
# Same knob, same reason as `elementwise.DEFAULT_CONST_HEAD_COUNT`: `row // h`
# strength-reduces to a shift/multiply instead of a software integer divide.
DEFAULT_CONST_HEAD_COUNT = True
# The chain's ``dot_do_o`` row geometry the optional ``delta`` output reproduces (module docstring): a 64-element chunk per
# 8 threads of 8 elements.  Pinned equal to ``bprop_chain_common.DOT_CHUNK_ELEMS`` / ``sm120._common._COPY_ELEMS`` by
# ``test_gate_bwd_delta_geometry_is_the_chains``; a change there is a change of the reduction order here.
DOT_CHUNK_ELEMS = 64
DOT_THREADS_PER_ROW = 8

_FAKE_STREAM = None

__all__ = [
    "DEFAULT_CONST_HEAD_COUNT",
    "DEFAULT_ROWS_PER_GROUP",
    "DEFAULT_THREADS_PER_CTA",
    "DOT_CHUNK_ELEMS",
    "DOT_THREADS_PER_ROW",
    "SigmoidGateBwdRecipe",
    "compile_sigmoid_gate_bwd",
    "moved_bytes",
    "run_sigmoid_gate_bwd",
    "validate_shape",
]


@cute.kernel
def frost_sigmoid_gate_bwd(
    mDOg: cute.Tensor,  # [T, H, D]  dO_gated (B2's output, compact)
    mO: cute.Tensor,  # [T, H, D]  PRE-gate O (saved.o, compact)
    mG: cute.Tensor,  # [T, H, D]  gate: a proj_slab column band (token stride N) or compact saved.gate
    mDO: cute.Tensor,  # [T, H, D]  OUT dO = dOg * s          (compact; MAY alias mDOg)
    mDG: cute.Tensor,  # [T, H, D]  OUT dG = dOg * O * s(1-s) (the dqkvg GATE band, token stride N)
    mOg: Optional[cute.Tensor],  # [T, H, D]  OUT O_gated = O * s   (compact; None when not wanted)
    mSeqLens: Optional[cute.Tensor],  # [B] int32 per-batch valid length, or None (dense: the select is folded out)
    mDelta: Optional[cute.Tensor],  # [B * H * S_pad] fp32 OUT delta = rowsum(O * dO_stored), dot_do_o's order; None = not wanted
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    s: cutlass.Int32,
    s_pad: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """One ``[D]`` row per lane group per ``rows_per_group``; see the module docstring.

    ``mDO`` may alias ``mDOg``: every lane reads its own 16 B of every operand
    (pass 1) before any lane stores (pass 2), and no lane touches another's.
    """
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(vec_chunks(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    has_og = cutlass.const_expr(mOg is not None)
    has_seq_lens = cutlass.const_expr(mSeqLens is not None)
    has_delta = cutlass.const_expr(mDelta is not None)
    needs_pos = cutlass.const_expr(has_seq_lens or has_delta)
    io_dtype = mDOg.element_type

    _h = cutlass.Int32(h_ct) if cutlass.const_expr(const_head_count) else h

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    warp_lane = tidx % cutlass.Int32(32)
    row0 = (cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(groups_per_cta) + grp) * cutlass.Int32(rows_per_group)
    lane_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
    bpe = cutlass.Int64(2)

    # --- PASS 1: every load first -------------------------------------------
    rows = []
    deads = []
    heads = []
    bidxs = []
    poss = []
    do_addrs = []
    dg_addrs = []
    og_addrs = []
    dogs = []
    os_ = []
    gs = []
    for r in cutlass.range_constexpr(rows_per_group):
        row = row0 + cutlass.Int32(r)
        row_r = row if row < n_rows else n_rows - cutlass.Int32(1)
        token = row_r // _h
        head = row_r % _h
        tok64 = token.to(cutlass.Int64)
        head64 = head.to(cutlass.Int64)
        # (batch, position) of the token: only traced with seq_lens bound (the dead-row predicate) or with the delta
        # output (its column); the divide does not exist in the plain dense artifact.
        bidx = cutlass.Int32(0)
        pos = cutlass.Int32(0)
        if cutlass.const_expr(needs_pos):
            bidx = token // s
            pos = token - bidx * s
        dead = cutlass.Boolean(False)
        if cutlass.const_expr(has_seq_lens):
            slen = ld_global(mSeqLens.iterator.toint() + bidx.to(cutlass.Int64) * cutlass.Int64(4), cutlass.Int32)
            dead = pos >= slen
        dog_addr = mDOg.iterator.toint() + (tok64 * cutlass.Int64(mDOg.stride[0]) + head64 * cutlass.Int64(mDOg.stride[1])) * bpe
        o_addr = mO.iterator.toint() + (tok64 * cutlass.Int64(mO.stride[0]) + head64 * cutlass.Int64(mO.stride[1])) * bpe
        g_addr = mG.iterator.toint() + (tok64 * cutlass.Int64(mG.stride[0]) + head64 * cutlass.Int64(mG.stride[1])) * bpe
        do_addr = mDO.iterator.toint() + (tok64 * cutlass.Int64(mDO.stride[0]) + head64 * cutlass.Int64(mDO.stride[1])) * bpe
        dg_addr = mDG.iterator.toint() + (tok64 * cutlass.Int64(mDG.stride[0]) + head64 * cutlass.Int64(mDG.stride[1])) * bpe
        og_addr = cutlass.Int64(0)
        if cutlass.const_expr(has_og):
            og_addr = mOg.iterator.toint() + (tok64 * cutlass.Int64(mOg.stride[0]) + head64 * cutlass.Int64(mOg.stride[1])) * bpe
        row_dog = []
        row_o = []
        row_g = []
        for c in cutlass.range_constexpr(chunks):
            off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
            row_dog.append([v for w in ld_global_v4(dog_addr + off, cutlass.Int32) for v in f16x2_to_f32(w, dtype=io_dtype)])
            row_o.append([v for w in ld_global_v4(o_addr + off, cutlass.Int32) for v in f16x2_to_f32(w, dtype=io_dtype)])
            row_g.append([v for w in ld_global_v4(g_addr + off, cutlass.Int32) for v in f16x2_to_f32(w, dtype=io_dtype)])
        rows.append(row)
        deads.append(dead)
        heads.append(head)
        bidxs.append(bidx)
        poss.append(pos)
        do_addrs.append(do_addr)
        dg_addrs.append(dg_addr)
        og_addrs.append(og_addr)
        dogs.append(row_dog)
        os_.append(row_o)
        gs.append(row_g)

    # --- PASS 2: sigmoid via tanh in fp32, the three products, pack, store ---
    half = cutlass.Float32(0.5)
    one = cutlass.Float32(1.0)
    quarter = cutlass.Float32(0.25)
    # The pad-tail store hands this to inline PTX: an OPAQUE zero, never a constant (a folded float constant takes the
    # immediate 'n' constraint and fails NVVM on CuTe DSL 4.7.1; the 4.8.0 toolchain happens to accept it).
    zero = opaque_f32_zero()
    for r in cutlass.range_constexpr(rows_per_group):
        live = rows[r] < n_rows
        # The row's math is pure register work, hoisted OUT of the live branch: the delta's shuffles below must be
        # reached by every lane of the warp (a tail row group computes on its clamped loads and stores nothing) -- the
        # forward's RoPE shuffle sits outside the same branch for the same reason.
        packed_do = []
        packed_dg = []
        packed_og = []
        for c in cutlass.range_constexpr(chunks):
            do_v = []
            dg_v = []
            og_v = []
            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                g = gs[r][c][i]
                t = cute.math.tanh(g * half, approx=True)
                sg = t * half + half
                ds = ((one - t) * (one + t)) * quarter  # == s * (1 - s), holds to |g| = 18.5, EXACT 0 at saturation
                do = dogs[r][c][i] * sg
                # dOg (NOT the gated dO) times O: with the gated dO the product would carry an extra s.
                dg = (dogs[r][c][i] * os_[r][c][i]) * ds
                og = os_[r][c][i] * sg
                if cutlass.const_expr(has_seq_lens):
                    # SELECT, never `* 0`: a dead row's O residue may be NaN.
                    do = zero if deads[r] else do
                    dg = zero if deads[r] else dg
                    og = zero if deads[r] else og
                do_v.append(do)
                dg_v.append(dg)
                og_v.append(og)
            packed_do.append([fp32_to_fp16(do_v[2 * i], do_v[2 * i + 1], dtype=io_dtype) for i in range(ELEMS_PER_ACCESS // 2)])
            packed_dg.append([fp32_to_fp16(dg_v[2 * i], dg_v[2 * i + 1], dtype=io_dtype) for i in range(ELEMS_PER_ACCESS // 2)])
            if cutlass.const_expr(has_og):
                packed_og.append([fp32_to_fp16(og_v[2 * i], og_v[2 * i + 1], dtype=io_dtype) for i in range(ELEMS_PER_ACCESS // 2)])
        if live:
            for c in cutlass.range_constexpr(chunks):
                off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                st_global_v4(do_addrs[r] + off, packed_do[c], cutlass.Int32)
                st_global_v4(dg_addrs[r] + off, packed_dg[c], cutlass.Int32)
                if cutlass.const_expr(has_og):
                    st_global_v4(og_addrs[r] + off, packed_og[c], cutlass.Int32)
        if cutlass.const_expr(has_delta):
            # delta = rowsum(O * dO) over the STORED dO (the packed words, unpacked: exactly what dot_do_o reads back),
            # in dot_do_o's order (module docstring): dot_do_o thread j owns this kernel's lanes {j, j+8, ..} per 16-B
            # chunk, chaining `acc = acc + O * dO` from 0.0 through them in that order; the chain is handed lane to lane
            # by shfl.idx (from 8 lanes below, rotating within the row's lane segment so the last group feeds the first
            # at a chunk boundary), every lane computing, the owning group keeping; then dot_do_o's butterfly.
            dot_groups = cutlass.const_expr(lanes // DOT_THREADS_PER_ROW)
            dot_grp = lane // cutlass.Int32(DOT_THREADS_PER_ROW)
            src_lane = (warp_lane - lane) + ((lane - cutlass.Int32(DOT_THREADS_PER_ROW)) & cutlass.Int32(lanes - 1))
            acc = zero
            for step in cutlass.range_constexpr(chunks * dot_groups):
                c = step // dot_groups
                start = acc
                if cutlass.const_expr(step > 0):
                    start = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, acc, src_lane, 0x1F, kind=nvvm.Shfl.IDX))
                chain = start
                for w in cutlass.range_constexpr(ELEMS_PER_ACCESS // 2):
                    do_lo, do_hi = f16x2_to_f32(packed_do[c][w], dtype=io_dtype)
                    chain = chain + os_[r][c][2 * w] * do_lo
                    chain = chain + os_[r][c][2 * w + 1] * do_hi
                acc = chain if dot_grp == cutlass.Int32(step % dot_groups) else acc
            for sh in cutlass.range_constexpr(3):
                acc = acc + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, acc, 1 << (2 - sh), 0x1F, kind=nvvm.Shfl.BFLY))
            if cutlass.const_expr(has_seq_lens):
                acc = zero if deads[r] else acc  # SELECT (a dead row's O residue may be NaN)
            # delta[b, h, pos]: ONE lane of the final chain group stores; the row at the sequence's last token also
            # zeroes the pad tail [s, s_pad) of its (b, h) -- what dot_do_o's unconditional tile store leaves there.
            row_base = mDelta.iterator.toint() + ((bidxs[r] * _h + heads[r]).to(cutlass.Int64) * s_pad.to(cutlass.Int64)) * cutlass.Int64(4)
            if live & (lane == cutlass.Int32(lanes - DOT_THREADS_PER_ROW)):
                st_global(row_base + poss[r].to(cutlass.Int64) * cutlass.Int64(4), acc, cutlass.Float32)
            pad_count = (s_pad - s) if (live & (poss[r] == s - cutlass.Int32(1))) else cutlass.Int32(0)
            k = lane
            while k < pad_count:
                st_global(row_base + (s + k).to(cutlass.Int64) * cutlass.Int64(4), zero, cutlass.Float32)
                k = k + cutlass.Int32(lanes)

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()


@cute.jit
def sigmoid_gate_bwd_launch(
    dog: cute.Tensor,
    o: cute.Tensor,
    gate: cute.Tensor,
    do: cute.Tensor,
    dg: cute.Tensor,
    og: Optional[cute.Tensor],
    seq_lens: Optional[cute.Tensor],
    delta: Optional[cute.Tensor],
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    s: cutlass.Int32,
    s_pad: cutlass.Int32,
    n_blocks: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_sigmoid_gate_bwd(
        dog, o, gate, do, dg, og, seq_lens, delta, n_rows, h, s, s_pad, h_ct, const_head_count, d, threads_per_cta, rows_per_group, use_pdl
    ).launch(grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl)


compiled_cache = {}


class SigmoidGateBwdRecipe(NamedTuple):
    """Build-time facts of one gate-backward launch (every field plan-time
    derivable; the token count rides in as a runtime ``Int32`` -- Rule 4).

    ``has_og`` / ``has_seq_lens`` / ``has_delta`` record which optional operands the
    artifact TRACED: an artifact compiled with one bound to ``None`` dereferences a
    null pointer, one compiled without would silently ignore it, so
    :func:`run_sigmoid_gate_bwd` checks both directions (Rule 1)."""

    compiled: object
    h: int
    d: int
    rows_per_cta: int
    has_og: bool
    has_seq_lens: bool
    dtype: object
    has_delta: bool = False  # APPENDED: the fused dot_do_o output (module docstring)


def compile_sigmoid_gate_bwd(
    *,
    dtype,
    h: int,
    d: int,
    has_og: bool,
    has_seq_lens: bool,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
    const_head_count: bool = DEFAULT_CONST_HEAD_COUNT,
    use_pdl: bool = False,
    has_delta: bool = False,
) -> SigmoidGateBwdRecipe:
    """Build from SHAPES ALONE -- no allocation, no launch. Every knob is in the cache key.

    ``has_delta`` adds the fp32 ``[B, H, S_pad]`` ``delta = rowsum(O * dO)`` output in the
    SDPA backward chain's own reduction order (module docstring); it needs ``d % 64 == 0``."""
    global _FAKE_STREAM
    validate_shape(d, threads_per_cta)
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"sigmoid_gate_bwd serves bf16/f16 only, got {dtype}")
    if has_delta and (d % DOT_CHUNK_ELEMS != 0 or lanes_per_row(d) < DOT_THREADS_PER_ROW):
        raise ValueError(
            f"the delta output reproduces the SDPA backward's dot_do_o reduction, whose row is {DOT_THREADS_PER_ROW} threads x {DOT_CHUNK_ELEMS}-element "
            f"chunks: it needs d_head % {DOT_CHUNK_ELEMS} == 0, got d_head={d}"
        )
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    key = (
        str(dtype),
        int(h),
        int(d),
        bool(has_og),
        bool(has_seq_lens),
        int(threads_per_cta),
        int(rows_per_group),
        bool(const_head_count),
        bool(use_pdl),
        current_device(),
        bool(has_delta),
    )
    if key not in compiled_cache:
        tok = cute.sym_int()
        # Six independent symbolic token strides: dOg / O / dO / Og are compact
        # in the block, gate and dG are column bands of [T, N] slabs.
        dense = [fake_rowmajor_dynamic_token_stride(dtype, tok, h, d) for _ in range(5)]
        og = fake_rowmajor_dynamic_token_stride(dtype, tok, h, d) if has_og else None
        seq_lens = (
            cute.runtime.make_fake_compact_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride_order=(0,), assumed_align=4) if has_seq_lens else None
        )
        # delta: the [B, H, S_pad] fp32 buffer flattened (contiguous), indexed by the kernel's own (b, h, pos) arithmetic
        delta = (
            cute.runtime.make_fake_compact_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(),), stride_order=(0,), assumed_align=ACCESS_BYTES)
            if has_delta
            else None
        )
        compiled_cache[key] = cute.compile(
            sigmoid_gate_bwd_launch,
            *dense,
            og,
            seq_lens,
            delta,
            cutlass.Int32(0),  # n_rows   ) runtime; the zeros pin the TYPE only
            cutlass.Int32(h),  # h        )
            cutlass.Int32(0),  # s        )
            cutlass.Int32(0),  # s_pad    )
            cutlass.Int32(0),  # n_blocks )
            int(h),
            bool(const_head_count),
            int(d),
            int(threads_per_cta),
            int(rows_per_group),
            bool(use_pdl),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return SigmoidGateBwdRecipe(
        compiled=compiled_cache[key],
        h=int(h),
        d=int(d),
        rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group,
        has_og=bool(has_og),
        has_seq_lens=bool(has_seq_lens),
        dtype=dtype,
        has_delta=bool(has_delta),
    )


def _check_row_layout(name: str, ten, d: int) -> None:
    """The ``[T, H, D]`` layout contract the artifact cannot check for itself.

    Every such operand is traced with a SYMBOLIC token stride
    (``fake_rowmajor_dynamic_token_stride``: strides ``(N, D, 1)``) so one artifact
    serves the compact tensors and the slab bands alike -- and nothing past this
    point constrains ``N``. A lane moves ``ACCESS_BYTES`` per access, so the token
    stride must be a whole number of accesses (``N % ELEMS_PER_ACCESS == 0``; the
    head stride ``D`` already is, ``validate_shape``) on an ``ACCESS_BYTES``-aligned
    base: a band of a slab whose width is not a multiple of ``ELEMS_PER_ACCESS``,
    or a view at an odd storage offset, would reach the kernel and fault on odd
    tokens with a misaligned ``ld/st.global.v4`` -- a sticky CUDA error, not a
    wrong number. The block's bands are safe by construction (``n_qkvg`` and every
    ``qkvg_offsets`` entry are multiples of ``QKVG_TILE_ALIGN`` = 64 elements);
    this names anything else. A size-1 token or head dim enters no address
    arithmetic, so torch's normalised stride there is admitted."""
    t, h = int(ten.shape[0]), int(ten.shape[1])
    s_t, s_h, s_e = (int(x) for x in ten.stride())
    base_off = ten.data_ptr() % ACCESS_BYTES
    ok = s_e == 1 and (h == 1 or s_h == d) and (t == 1 or s_t % ELEMS_PER_ACCESS == 0) and base_off == 0
    if not ok:
        raise ValueError(
            f"{name} must be a [T, H, D] view with heads contiguous within a token -- strides (N, {d}, 1) with the token stride N a multiple of "
            f"{ELEMS_PER_ACCESS} elements -- on a {ACCESS_BYTES}-B-aligned base (a lane moves {ACCESS_BYTES} B per access; anything else is a "
            f"misaligned access on odd tokens); got strides {(s_t, s_h, s_e)}, base {base_off} B past a {ACCESS_BYTES}-B boundary"
        )


def _check_one_cuda_device(anchor_name: str, anchor, operands) -> None:
    """Every bound operand on ONE CUDA device, the anchor's.

    The kernel reads each operand through a device pointer. A CPU ``seq_lens``
    (``torch.tensor(lens, dtype=torch.int32)`` with no ``device=``) passes the
    dtype / rank / length checks and would be dereferenced as a HOST address --
    an illegal-address fault at the next synchronize, sticky for the process --
    and the same holds for any other operand left on the host or on another
    device. Named here, before the launch."""
    if not anchor.is_cuda:
        raise ValueError(f"{anchor_name} must be a CUDA tensor (the kernel reads every operand through a device pointer), got device {anchor.device}")
    for name, ten in operands:
        if ten is not None and ten.device != anchor.device:
            raise ValueError(f"{name} must be on {anchor.device} with {anchor_name}, got {ten.device}")


def _check_operand(r: SigmoidGateBwdRecipe, name: str, ten, t: int) -> None:
    if ten.dtype != r.dtype:
        # The tvm-ffi boundary also rejects this on cutlass-dsl >= 4.8, indexed by
        # argument position; name the operand here (the `run_elementwise_gate` guard).
        raise ValueError(f"{name} is {ten.dtype} but this artifact was compiled for {r.dtype}; dtype is fixed per artifact")
    if ten.dim() != 3 or int(ten.shape[1]) != r.h:
        raise ValueError(f"{name} has H={tuple(ten.shape)} but this artifact was compiled for H={r.h}; H is fixed per artifact (operands are [T, H, D])")
    if int(ten.shape[2]) != r.d:
        raise ValueError(f"{name} has D={int(ten.shape[2])} but this artifact was compiled for D={r.d}; D is fixed per artifact")
    if int(ten.shape[0]) != t:
        raise ValueError(f"{name} has T={int(ten.shape[0])} but dO_gated has T={t}; every operand covers the same tokens")
    _check_row_layout(name, ten, r.d)


def run_sigmoid_gate_bwd(
    r: SigmoidGateBwdRecipe, dog, o, gate, do, dg, og=None, seq_lens=None, *, s: Optional[int] = None, stream, delta: Optional[torch.Tensor] = None
) -> None:
    """Launch. ``dog / o / gate / do / dg (/ og)`` are ``[T, H, D]`` (``T = B*S``);
    ``do`` may alias ``dog``. ``seq_lens`` (``[B]`` int32 on the device) needs
    ``s``, the per-batch sequence length (``T == B * s``); so does ``delta``, the
    fp32 contiguous ``[B, H, S_pad]`` (``S_pad >= s``) output of a ``has_delta``
    artifact (module docstring). Host-only checks, no allocation."""
    if r.has_delta and delta is None:
        raise ValueError("this artifact was compiled WITH a delta output (has_delta=True); it must be bound at execute (Rule 1: no silent fallback)")
    if not r.has_delta and delta is not None:
        raise ValueError("this artifact was compiled WITHOUT a delta output (has_delta=False); passing one would silently ignore it (Rule 1)")
    if r.has_og and og is None:
        raise ValueError("this artifact was compiled WITH an O_gated output (has_og=True); it must be bound at execute (Rule 1: no silent fallback)")
    if not r.has_og and og is not None:
        raise ValueError("this artifact was compiled WITHOUT an O_gated output (has_og=False); passing one would silently ignore it (Rule 1)")
    if r.has_seq_lens and seq_lens is None:
        raise ValueError("this artifact was compiled WITH seq_lens (the dead-row select); it must be bound at execute (Rule 1: no silent fallback)")
    if not r.has_seq_lens and seq_lens is not None:
        raise ValueError("this artifact was compiled WITHOUT seq_lens (dense, the select folded out); passing one would silently ignore it (Rule 1)")
    t = int(dog.shape[0])
    for name, ten in (("dog", dog), ("o", o), ("gate", gate), ("do", do), ("dg", dg)):
        _check_operand(r, name, ten, t)
    if og is not None:
        _check_operand(r, "og", og, t)
    if seq_lens is not None:
        if s is None:
            raise ValueError("seq_lens needs s (the per-batch sequence length, T == B * s) to map a token to its batch entry")
        if seq_lens.dtype != torch.int32 or seq_lens.dim() != 1:
            raise ValueError(f"seq_lens must be a 1-D int32 tensor, got {seq_lens.dtype} of shape {tuple(seq_lens.shape)}")
        if int(s) <= 0 or t % int(s) != 0:
            raise ValueError(f"s={s} must divide T={t} (T == B * s)")
        if int(seq_lens.numel()) != t // int(s):
            raise ValueError(f"seq_lens has {int(seq_lens.numel())} entries but T // s = {t // int(s)} batches")
    s_pad = 0
    if delta is not None:
        if s is None:
            raise ValueError("delta needs s (the per-batch sequence length, T == B * s): delta is [B, H, S_pad], indexed by (batch, head, position)")
        if int(s) <= 0 or t % int(s) != 0:
            raise ValueError(f"s={s} must divide T={t} (T == B * s)")
        shape = tuple(int(x) for x in delta.shape)
        if delta.dtype != torch.float32 or delta.dim() != 3 or not delta.is_contiguous():
            raise ValueError(
                f"delta must be a CONTIGUOUS fp32 [B, H, S_pad] tensor (the SDPA backward's dot_do_o layout), got {delta.dtype} of shape {shape} "
                f"with strides {tuple(delta.stride())}"
            )
        if shape[:2] != (t // int(s), r.h) or shape[2] < int(s):
            raise ValueError(f"delta must be [B={t // int(s)}, H={r.h}, S_pad >= {int(s)}] (T == B * s, S_pad the chain's padded extent), got {shape}")
        if delta.data_ptr() % ACCESS_BYTES:
            raise ValueError(
                f"delta base must be {ACCESS_BYTES}-B aligned (the chain reads it through a {ACCESS_BYTES}-B-aligned view), got {delta.data_ptr():#x}"
            )
        s_pad = shape[2]
    _check_one_cuda_device("dog", dog, (("o", o), ("gate", gate), ("do", do), ("dg", dg), ("og", og), ("seq_lens", seq_lens), ("delta", delta)))
    n_rows = t * r.h
    n_blocks = (n_rows + r.rows_per_cta - 1) // r.rows_per_cta
    # The optional slots stay in the ABI even when they traced to None (the
    # artifact folded out the STORES / the select, not the parameters).
    r.compiled(
        dog,
        o,
        gate,
        do,
        dg,
        og,
        seq_lens,
        delta.view(-1) if delta is not None else None,
        cutlass.Int32(n_rows),
        cutlass.Int32(r.h),
        cutlass.Int32(int(s) if s is not None else 0),
        cutlass.Int32(s_pad),
        cutlass.Int32(n_blocks),
        cuda.CUstream(int(stream)),
    )


def moved_bytes(t: int, h: int, d: int, *, elem_bytes: int = 2, has_og: bool, has_delta: bool = False) -> int:
    """HBM traffic of one launch -- the denominator for an SOL number.

    Reads dO_gated, O and gate; writes dO and dG (+ O_gated; + the 4-B ``delta``
    per row). In-place dO still moves all of it: a write is a write whether or
    not it lands on the read's address. ``seq_lens`` (4 B per batch) and the
    delta's pad tail are not counted.
    """
    return (5 + (1 if has_og else 0)) * t * h * d * elem_bytes + (4 * t * h if has_delta else 0)


frost_sigmoid_gate_bwd.set_name_prefix("cudnn", remove_cutlass_symbol=True)
