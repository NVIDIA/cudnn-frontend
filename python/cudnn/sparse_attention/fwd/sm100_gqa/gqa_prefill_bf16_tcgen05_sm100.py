# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-5 PR4 MSA-cell tcgen05 tensor-core mainloop.

Narrow envelope this module serves -- the exact "MSA cell" scope for this
round, nothing wider:

* ``D_k == D_v == 128``, ``index_granularity == 128`` (one gathered KV block
  == one full MMA N-tile, ``TILE_N``), ``G == H_kv``.
* ``uniform_within_tile=True`` **required** -- same caller contract as
  ``gqa_prefill_bf16_tile_sm100.py`` (every row/head in a Q tile shares one
  ``topk_idxs`` row, read once from the tile's first row): a tcgen05 MMA
  issues ONE QK^T / P@V pair per CTA per selected KV block, batched over the
  whole M tile, so per-row-varying selection is out of scope by construction
  (use the scalar kernel for that).
* BF16 only, BSHD only (THD is an explicitly out-of-scope follow-up --
  raises ``NotImplementedError`` naming the gap, not a silent dense-batch=1
  degrade).

What "real tcgen05 MMA" means here, concretely (grep-verifiable):
``cudnn.frost.tile_dsl.mma.mma_ss``/``mma_ts`` (``nvvm.tcgen05_mma`` under the
hood, the SAME primitives ``prefill_d128_f16_sm100.py`` uses in production)
issue the QK^T and P@V contractions; ``Q``/``K``/``V`` land in TMEM-MMA-
ready swizzled SMEM via genuine hardware TMA (``cudnn.frost.tile_dsl.tma
.tma_load_tile`` against a ``cutlass.experimental.cuda.tensor_map`` box built
with ``create_tensor_map_tiled_from_view``) with a *runtime* row coordinate
per gathered block -- not ``cp.async``, not a compile-time-constant address.
No scalar FFMA path exists in this file at all.

Why this shape is tractable where round-3/4's more general tcgen05 attempts
were not (see this package's other modules' docstrings for the fuller
history): ``index_granularity == 128 == TILE_N`` means one ``topk_idxs``
entry maps onto exactly one MMA-N-sized, TMA-box-aligned KV block -- no
gather-then-mask-over-a-union machinery, no swizzle math to hand-derive (the
box lands pre-swizzled in HW, exactly like every other TMA-fed SM100 SDPA
kernel in this repo), and the row/head packing this envelope needs
(``TOKENS_PER_TILE = 128 // heads_per_kv`` Q rows x ``heads_per_kv`` Q heads
sharing one KV-head's selection, giving the full ``M=128`` tile) is the
*same* GQA head-packing box shape (``(1, TOKENS_PER_TILE, HEADS_PER_TILE,
D)``) ``prefill_d128_f16_sm100.py`` already uses for its own dense GQA
packing -- not new swizzle/TMA-descriptor design, just a new (sparse,
runtime-indexed) row coordinate into the same box shape.

Design: single warpgroup (128 threads = 4 warps, ``block=(128,1,1)``,
``cta_group=1``, no CGA cluster), fully blocking/sequential per selected KV
block -- no software pipelining across ``topk_idxs`` entries or Q/K/V
double-buffering (a single-stage K/V SMEM buffer, re-armed each entry). This
gives up the multi-role warp-specialized overlap
``prefill_d128_f16_sm100.py`` uses for its dense sweep, in exchange for a
mainloop simple enough to land and validate against the oracle in this
round's budget; the async TMA loads and tcgen05 MMAs are still genuinely
async hardware ops (mbarrier-gated), just not overlapped against each other
across iterations. A follow-up round double-buffering K/V across entries
(the same ``STAGES_KV=2`` idea ``prefill_d128`` uses) is the natural next
perf lever once this lands correctly -- tracked, not attempted here.

TMEM layout (256 of the 512-column Blackwell cap; single sub-tile, no
dual-softmax-warpgroup aliasing since there is only one):

    S  (QK^T, f32)      : cols   0..127
    P  (softmax out,    : cols  64..127  (bf16-packed BMM2 A-operand;
        bf16, aliases         aliases S's tail exactly like
        S's tail)              ``prefill_d128_f16_sm100.LAYOUT.P0_OFF``)
    O  (P@V accum, f32) : cols 128..255

Online-softmax rescale of the running O accumulator (TMEM-resident, so a
fresh MMA can't simply "start from" a rescaled value the way an RMEM
accumulator would) follows ``prefill_d128_f16_sm100``'s own correction-warp
recipe: read O back from TMEM, multiply by ``alpha``, write it back, THEN
issue the P@V MMA with ``accumulate=True`` -- skipped only for the very
first valid entry of a (Q-tile, KV-head), which instead writes O directly
(``accumulate=False``) since TMEM starts uninitialized (``alpha`` would be
exactly ``0.0`` there per the online-softmax identity, but ``0.0 *
uninitialized-NaN-bit-pattern`` is still NaN -- the explicit skip, not a
reliance on that identity, is what keeps this correct).

**Round-6 status -- compile-hang root-caused and fixed; correctness not
independently confirmed by every session this round, read before assuming
this kernel is production-ready.** Round-5 bisected the >10-minute
``cute.compile()`` hang to a single ``mma_ss(...)`` call issued from
inside a *dynamic* ``cutlass.range(0, topk_max, 1, unroll=1)`` loop over
``topk_idxs`` entries. Round-6's primary hypothesis -- that ``topk_max``
(the static last-dim extent of ``topk_idxs``, already threaded through
``_compile``'s ``@lru_cache`` key and ``fake_idx``'s shape, i.e. known at
DSL trace time, NOT a per-call runtime scalar the way ``n_entries``/
``topk_length`` genuinely are) should drive a plain Python-level ``for j
in range(topk_max):`` (fully unrolled at trace time) instead of a
device-side dynamic loop -- is confirmed correct by two independent
measurements this round:

* An isolated minimal repro (TMA-only Q/K load + a single ``mma_ss`` per
  iteration, everything else in this file's mainloop stripped away)
  compiled in well under a second for BOTH a dynamic-``cutlass.range``
  and a static-Python-``range`` variant with the same trip count -- i.e.
  "``mma_ss`` inside *any* dynamic loop is what hangs" does NOT reproduce
  in isolation; whatever triggers the pathological compile time is an
  interaction between the dynamic loop and this file's full mainloop
  complexity (5 mbarriers, phase tracking, nested ``if j < n_entries: if
  is_valid:`` branches, tcgen05_ld/st + softmax between the two MMAs), not
  a blanket "dynamic-loop-with-MMA" compiler defect.
* Recompiling this file's actual, full kernel (``_compile(h_q=64, h_kv=4,
  has_topk_length=False, has_attn_sink=False, arch="sm_100a",
  topk_max=16)``) with the loop below now a static Python unroll completed
  in well under a second -- a >1000x improvement over the >10-minute
  round-5 hang on the same shape family.

The mainloop below now uses that static unroll (see the comment on
``kernel_fn``'s ``topk_max`` parameter). This resolves the compile-time
problem; it does **not** by itself establish correctness -- and a later
round-6 pass (this session) found a NEW, real bug it uncovered: **the
compiled kernel deadlocks at launch time.** Re-ran
``verify_round5_repro.py`` (full ``sparse_attention_forward_wrapper``:
compile + launch + ``torch.cuda.synchronize()``) on this box confirmed
*quiescent* immediately beforehand (``nvidia-smi`` showed 0 MiB used / no
other compute processes right before launching) -- so, unlike the prior
ambiguous 200s timeout blamed on box contention, this run rules contention
out. The process never printed past "starting compile+run..." and
``nvidia-smi --query-gpu=utilization.gpu`` read a sustained **100%** for
the entire duration observed (30s+ of direct polling, and the prior,
contention-confounded attempt ran a full 200s the same way) -- i.e. the
GPU is not idle/blocked-on-host, some kernel is genuinely resident and
spinning, but the host-side ``torch.cuda.synchronize()`` never returns.
That combination (100% SM occupancy, zero forward progress) is the
signature of an on-device spin-wait deadlock -- most likely an mbarrier
arrive/wait pairing that a subset of the single warpgroup's warps never
satisfies (see the module docstring's "single-stage K/V re-arm" design
note above) -- not a compiler or contention artifact. This is a
**confirmed, real runtime bug**, independent of and in addition to the
now-fixed compile-time hang; it was killed with ``kill -9`` rather than
left to complete (it will not complete on its own). A future round needs
to bisect this the same way round 5 bisected the compile hang: strip the
mainloop down (single topk entry, no softmax rescale, etc.) until the
launch returns, to isolate which mbarrier phase/warp-role pairing is
unsatisfied. Given this, the round's hard gate's correctness leg is not
just "unconfirmed" but actively contradicted -- do not attempt to verify
correctness against the oracle until this deadlock is fixed, since the
kernel cannot currently produce output to compare. Also still unconfirmed
(unreachable while the deadlock stands): the determinism repeat-run check
and the per-D-column-chunk
audit of the V-operand ``mma_ts`` smem-descriptor stride math (this
file's ``_V_PC_COLS``/``LEADING_BYTE_OFFSET_PV``/``STRIDE_BYTE_OFFSET_PV``
were inspected against ``prefill_d128_f16_sm100.py``'s cga1-equivalent
formula -- ``_V_PC_COLS = TILE_O // CTA_MMA`` there vs ``TILE_N`` here,
which are numerically identical only because this cell's fixed geometry
has ``D == TILE_N == 128`` -- and appear structurally consistent, but
"appears consistent by inspection" is not the same as "passes the oracle
per-D-column-chunk", which is what actually caught the analogous bug in
round 5's separate, unpersisted draft). Per the round's hard gate (all of:
fast compile, correctness, AND a real compile-timeout catch in
``dispatch.py``'s fallback must be independently confirmed before
defaulting real callers to this path), ``dispatch.py`` keeps
``_DEFAULT_TRY_TCGEN05_FAST_PATH = False`` this round -- the compile-speed
leg of the gate is now met, but the other two are not, so the default
stays reverted to the scalar kernel, full stop. A future round must FIRST
fix the launch-time deadlock confirmed above (the kernel cannot be
correctness-tested or benchmarked until it returns from
``torch.cuda.synchronize()`` at all); only then run
``test_sparse_attention_fwd_tile_sm100.py``'s ``tcgen05``-prefixed suite
and this round's ``bench_tcgen05_minimax.py`` on a quiescent (single-
tenant) SM100 box, then add a real timeout-catching fallback in
``dispatch.py`` before flipping the default.

**Round-7 status -- primary hypothesis tested and falsified; deadlock is
NOT gated by the per-slot validity guard, keep looking elsewhere.** This
round's working hypothesis was that the mainloop's mbarrier/pipeline
primitives (``mb_k``/``mb_v``/``mb_mma1``/``mb_mma2`` arrive+wait, the
CTA-wide ``nvvm.barrier_cta_sync()``) were nested inside the ``j <
n_entries: is_valid:`` guard alongside the MMA issue, and that some
lanes/warps could diverge on whether to reach them. The mainloop below
was restructured so those barrier/pipeline calls run unconditionally
every unrolled ``j`` iteration; only the MMA issue (``mma_ss``/
``mma_ts``), the TMEM softmax/accumulation math, and the ``first_done``
flip stay predicated on ``do_work`` (see the loop body for the exact
split). **This did not fix the deadlock, and -- more importantly -- a
new isolation experiment this round shows the hypothesis itself doesn't
hold**: a from-scratch minimal repro (``h_kv=2, h_q=2, heads_per_kv=1,
topk_max=1, exactly one KV block, one Q tile -- so ``n_entries=1``,
``is_valid`` true for the kernel's only iteration, i.e. the validity
guard is never exercised at all, nothing is ever masked out) deadlocks
identically to the round-6 "realistic MSA-cell" repro, on a GPU confirmed
quiescent (``nvidia-smi`` 0% util / 0 MiB before launch) both immediately
before and via live polling during the hang (sustained 100% GPU
utilization, host-side ``torch.cuda.synchronize()`` never returns, killed
by an external wall-clock timeout after 200s+ rather than completing).
Since this minimal case never takes the "masked" branch of the guard at
all, hoisting barrier participation out of that guard (this round's fix,
still landed below as a defensively-correct change worth keeping) cannot
be what's needed -- the real bug lives somewhere in the mainloop's
*unconditionally executed* path: candidates not yet individually isolated
this round include the TMA tensor-map / box setup (``tma_load_tile``
against ``create_tensor_map_tiled_from_view``-built descriptors),
``tmem_alloc``/``tmem_dealloc`` under ``CTA_1`` on this specific
grid/block launch shape (``block=[TILE_M, 1, 1]``), or a basic
init/arrive/wait phase mismatch in ``MBarrier`` itself independent of any
conditional gating. A future round needs to bisect the *always-taken*
path the same way round 5 bisected the compile hang -- e.g. strip the
kernel down to Q-load-only (no K/V TMA, no MMA at all) and add back one
primitive at a time (K-TMA, then V-TMA, then ``mma_ss``, then the
softmax TMEM round-trip, then ``mma_ts``) until the launch stops
returning, rather than continuing to vary ``topk_idxs`` validity, which
this round's minimal repro shows is not the lever. ``dispatch.py`` stays
exactly as round 6 left it (``_DEFAULT_TRY_TCGEN05_FAST_PATH = False``,
re-verified unchanged this round) -- none of the round's hard-gate legs
(fast compile under real conditions, launch completing at all,
oracle correctness, a real timeout-catching fallback) are met.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap
import torch

from cudnn.frost.tile_dsl.barrier import MBarrier, Producer
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts
from cudnn.frost.tile_dsl.pointwise import row_max_reduction, row_reduction_pair
from cudnn.frost.tile_dsl.tma import tma_load_tile
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.sdpa.fwd.kernels._common_sm100 import row_max_for_exp2

from ._common_sm100 import resolve_entry_window

NEG_INF = float("-inf")

# Fixed MSA-cell geometry -- see module docstring.
TILE_M = 128  # MMA M dim: TOKENS_PER_TILE Q rows x HEADS_PER_TILE Q heads
TILE_N = 128  # KV tokens per gathered block == index_granularity (fixed)
D = 128  # d_k == d_v (fixed)

STORAGE_DTYPE = cutlass.BFloat16
BPE = 2
MMA_KIND = nvvm.Tcgen05MMAKind.F16  # covers both f16/bf16 A/B (dtype lives in idesc), matches prefill_d128
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_1
CTA_MMA = 1  # this MSA cell is cga1-only (no collective MMA) -- see module docstring; every V-operand
# smem-descriptor constant below is derived FROM this, not borrowed from prefill_d128's cga2 constants.
TILE_K_HW = 16  # f16/bf16 1-chunk-per-subtile QMMA K-step, matches CfgD128.TILE_K_HW_BMM1/2


def _swz_from_inner(inner_bytes: int) -> int:
    """Largest swizzle atom (128/64/32B) the innermost smem dimension supports.

    Mirrors ``MmaDesc._swz_from_inner`` / ``config_sm100.v_swz_bytes`` exactly --
    same formula, kept local here so this module's constants are self-derived
    instead of importing a cga2-shaped config object. See the round-6 audit
    note below for why this matters.
    """
    if inner_bytes % 128 == 0:
        return 128
    if inner_bytes % 64 == 0:
        return 64
    if inner_bytes % 32 == 0:
        return 32
    raise ValueError(f"inner bytes {inner_bytes} not a multiple of 32/64/128")


# Round-6 audit (V-operand smem-descriptor stride math, cga1 vs prefill_d128's
# cga2 default -- see module docstring's PR4 subtask list): Q/K (BMM1, both
# non-transposed operands) swizzle off their K-dim inner extent, which is D
# for both; V (BMM2's B-transposed operand) swizzles off its PER-CTA N extent,
# which is D // CTA_MMA (``MmaDesc.n_per_cta`` when ``btranspose``), matching
# ``config_sm100.v_swz_bytes(tile_o, cta_mma, bpe)``. At CTA_MMA=1 (this
# kernel, always) those two derivations happen to land on the same 128B atom
# for D=128/BPE=2 (256 % 128 == 0 either way) -- confirmed by deriving each
# independently below and asserting they agree, rather than assuming one
# shared literal covers both, which is what actually needs re-deriving (not
# ``k_subtile``, see the comment at ``bmm2_desc`` below) if this envelope
# ever widens to D != TILE_N or CTA_MMA != 1.
SWZ_BYTES_QK = _swz_from_inner(D * BPE)  # Q/K inner = K-dim = D (non-transposed)
SWZ_BYTES_V = _swz_from_inner((D // CTA_MMA) * BPE)  # V inner = n_per_cta = D // CTA_MMA (btranspose)
assert SWZ_BYTES_QK == SWZ_BYTES_V, (
    f"gqa_prefill_bf16_tcgen05_sm100: Q/K swizzle ({SWZ_BYTES_QK}B) and V swizzle ({SWZ_BYTES_V}B) "
    "diverged -- this module shares one SMEM_LAYOUT_QKV constant across Q/K/V on the assumption they "
    "match at this envelope's fixed D=TILE_N=128/CTA_MMA=1 geometry; if a future round changes D, "
    "TILE_N, or CTA_MMA, split this back into independent Q/K vs V swizzle constants (as "
    "prefill_d128_f16_sm100.py does) instead of forcing this assert to pass."
)
SWZ_BYTES = SWZ_BYTES_QK  # == SWZ_BYTES_V, asserted above
GRANU_ELEMS = SWZ_BYTES // BPE  # 64
TMA_ITERS = (D * BPE) // SWZ_BYTES  # 2

_SWZ_ENUM = {128: 2, 64: 4, 32: 6}  # matches prefill_d128_f16_sm100._SWZ_ENUM
SMEM_LAYOUT_QKV = _SWZ_ENUM[SWZ_BYTES]

LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * SWZ_BYTES
_CORE_MATRIX_ROWS = 8
# V's per-CTA N-column count (mirrors prefill_d128_f16_sm100's own
# ``_V_PC_COLS = CFG.TILE_O // CFG.CTA_MMA``, i.e. MmaDesc.n_per_cta for the
# btranspose operand) -- derived from D/CTA_MMA, NOT reused from TILE_N. They
# are numerically equal here only because this envelope fixes D == TILE_N ==
# 128; deriving from D keeps this formula correct if that coincidence ever
# breaks (e.g. a future non-square D/index_granularity cell).
_V_PC_COLS = D // CTA_MMA
assert _V_PC_COLS == TILE_N, "gqa_prefill_bf16_tcgen05_sm100: D // CTA_MMA != TILE_N -- LEADING/STRIDE_BYTE_OFFSET_PV below assume they match at this envelope's fixed geometry"
LEADING_BYTE_OFFSET_PV = 0 if (_V_PC_COLS // _CORE_MATRIX_ROWS) <= 8 else TILE_N * SWZ_BYTES_V
STRIDE_BYTE_OFFSET_PV = 8 * SWZ_BYTES_V

# TMEM column layout (see module docstring).
S_OFF = 0
P_OFF = 64  # bf16-packed alias of S's tail (2 probs / f32 cell)
O_OFF = 128
TOTAL_TMEM_COLS = 256

qBufferElems = TILE_M * D
kBufferElems = TILE_N * D
vBufferElems = TILE_N * D


def _make_kernel(heads_per_kv: int, has_topk_length: bool, has_attn_sink: bool, topk_max: int):
    if TILE_M % heads_per_kv != 0:
        raise NotImplementedError(
            f"gqa_prefill_bf16_tcgen05_sm100: heads_per_kv={heads_per_kv} must divide TILE_M={TILE_M} "
            "(row/head packing to the MMA M dimension) -- outside this round's MSA-cell scope"
        )
    HEADS_PER_TILE = heads_per_kv
    TOKENS_PER_TILE = TILE_M // HEADS_PER_TILE

    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=TILE_N,
        m_dim=TILE_M,
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=D,
        m_dim=TILE_M,
        b_major=1,
    )
    bmm1_desc = MmaDesc(
        M=TILE_M,
        N=TILE_N,
        K=D,
        bpe_a=BPE,
        bpe_b=BPE,
        tile_k_hw=TILE_K_HW,
        btranspose=False,
        cta_group=1,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    # NOTE (round-6 V-operand audit): ``k_subtile`` below is passed for
    # consistency with prefill_d128_f16_sm100.py's own ``bmm2_desc`` (and to
    # skip MmaDesc.__post_init__'s default, which for btranspose=True would
    # set it to K=TILE_N, i.e. one giant subtile), but it is NOT what governs
    # the actual per-k-step smem address math mma_ts() issues: mma_ts/mma_ss
    # only read ``desc.smem_advance_B_intra`` / ``desc.smem_subtile_B`` /
    # ``desc.sps_B``, all of which derive from ``swz_b_bytes`` (itself from
    # ``n_per_cta * bpe_b``, i.e. D/CTA_MMA and BPE here) -- NOT from
    # ``k_subtile`` (``steps_per_subtile``/``num_subtiles``, the only
    # properties that read ``k_subtile``, have no callers anywhere in
    # ``frost.tile_dsl``; grep confirms). A wrong ``k_subtile`` value here
    # cannot by itself cause the round-5 D-column-chunk P@V error -- the real
    # levers are ``SWZ_BYTES_V``/``LEADING_BYTE_OFFSET_PV``/
    # ``STRIDE_BYTE_OFFSET_PV`` above, which this round re-derived from this
    # kernel's own cga1 V geometry (not copied from prefill_d128's cga2/fp8
    # ``V_SWZ_BYTES=64`` constant) and asserted self-consistent. Do not spend
    # a future round re-chasing ``k_subtile`` as the fix lever without first
    # re-confirming this dead-code finding still holds.
    bmm2_desc = MmaDesc(
        M=TILE_M,
        N=D,
        K=TILE_N,
        bpe_a=BPE,
        bpe_b=BPE,
        tile_k_hw=TILE_K_HW,
        btranspose=True,
        k_subtile=SWZ_BYTES_V // BPE,
        cta_group=1,
        idesc=idesc_pv,
        kind=MMA_KIND,
    )

    @cute.kernel
    def kernel_fn(
        tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
        tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
        tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
        topk_idxs: cute.Tensor,
        topk_length: Optional[cute.Tensor],
        attn_sink: Optional[cute.Tensor],
        out: cute.Tensor,
        lse: cute.Tensor,
        kv_bound: cutlass.Int32,
        s_q: cutlass.Int32,
        scale_log2: cutlass.Float32,
        rows_total: cutlass.Int32,
    ) -> None:
        # ``topk_max`` (the last dim of ``topk_idxs``) is a Python-level
        # compile-time int here (closed over from ``_make_kernel``'s
        # argument, itself threaded from ``_compile``'s ``@lru_cache``
        # key) -- NOT a device-side ``cutlass.Int32`` kernel argument. It is
        # a tensor SHAPE dimension (known at DSL trace time), unlike
        # ``n_entries``/``topk_length`` below, which stay genuine per-call
        # runtime values. This lets the per-topk-entry loop below be a
        # plain Python ``for`` (fully unrolled at trace time, in ascending
        # slot order -- satisfying the frozen contract's determinism
        # requirement for free) instead of a dynamic ``cutlass.range``.
        tidx, _, _ = cute.arch.thread_idx()
        warp = tidx // cutlass.Int32(32)
        row_in_tile = tidx // cutlass.Int32(HEADS_PER_TILE)
        head_in_group = tidx % cutlass.Int32(HEADS_PER_TILE)

        tile_idx = cute.arch.block_idx()[0]
        kv_head = cute.arch.block_idx()[1]
        batch = cute.arch.block_idx()[2]

        row_base = tile_idx * cutlass.Int32(TOKENS_PER_TILE)
        t_row = row_base + row_in_tile
        t_q = cute.math.min(t_row, rows_total - cutlass.Int32(1)) + batch * s_q
        q_head = kv_head * cutlass.Int32(HEADS_PER_TILE) + head_in_group

        kv_base = batch * kv_bound  # BSHD only this round (see module docstring)

        idx_v = cutlass.make_array_view(topk_idxs)
        out_v = cutlass.make_array_view(out)
        lse_v = cutlass.make_array_view(lse)
        len_v = cutlass.make_array_view(topk_length) if cutlass.const_expr(topk_length is not None) else None
        sink_v = cutlass.make_array_view(attn_sink) if cutlass.const_expr(attn_sink is not None) else None

        t_q_rep = row_base + batch * s_q  # tile's row 0 -- uniform_within_tile contract
        n_entries = cutlass.Int32(topk_max)
        if cutlass.const_expr(len_v is not None):
            n_entries = cutlass.Int32(len_v[t_q_rep, kv_head])

        # === SMEM: single-stage Q/K/V (no cross-entry double buffering this round) ===
        tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)
        sQ_raw = cutlass.Array(STORAGE_DTYPE, qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
        sK_raw = cutlass.Array(STORAGE_DTYPE, kBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
        sV_raw = cutlass.Array(STORAGE_DTYPE, vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)

        sQ = SmemTile(
            base=sQ_raw,
            elems_per_stage=qBufferElems,
            leading_byte_offset=LEADING_BYTE_OFFSET_QK,
            stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
            layout=SMEM_LAYOUT_QKV,
            tma_loads_per_tile=TMA_ITERS,
            tma_granu_elems=GRANU_ELEMS,
            tma_subtile_stride_elems=TILE_M * GRANU_ELEMS,
        )
        sK = SmemTile(
            base=sK_raw,
            elems_per_stage=kBufferElems,
            leading_byte_offset=LEADING_BYTE_OFFSET_QK,
            stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
            layout=SMEM_LAYOUT_QKV,
            tma_loads_per_tile=TMA_ITERS,
            tma_granu_elems=GRANU_ELEMS,
            tma_subtile_stride_elems=TILE_N * GRANU_ELEMS,
        )
        sV = SmemTile(
            base=sV_raw,
            elems_per_stage=vBufferElems,
            leading_byte_offset=LEADING_BYTE_OFFSET_PV,
            stride_byte_offset=STRIDE_BYTE_OFFSET_PV,
            layout=SMEM_LAYOUT_QKV,
            tma_loads_per_tile=TMA_ITERS,
            tma_granu_elems=GRANU_ELEMS,
            tma_subtile_stride_elems=TILE_N * GRANU_ELEMS,
        )

        mb_q = MBarrier(base_ptr=cutlass.Array(cutlass.Int64, 1, alignment=8, space=cutlass.AddressSpace.smem), stages=1, init_count=1, producer=int(Producer.TMA_LOAD))
        mb_k = MBarrier(base_ptr=cutlass.Array(cutlass.Int64, 1, alignment=8, space=cutlass.AddressSpace.smem), stages=1, init_count=1, producer=int(Producer.TMA_LOAD))
        mb_v = MBarrier(base_ptr=cutlass.Array(cutlass.Int64, 1, alignment=8, space=cutlass.AddressSpace.smem), stages=1, init_count=1, producer=int(Producer.TMA_LOAD))
        mb_mma1 = MBarrier(base_ptr=cutlass.Array(cutlass.Int64, 1, alignment=8, space=cutlass.AddressSpace.smem), stages=1, init_count=1, producer=int(Producer.MMA_COMMIT))
        mb_mma2 = MBarrier(base_ptr=cutlass.Array(cutlass.Int64, 1, alignment=8, space=cutlass.AddressSpace.smem), stages=1, init_count=1, producer=int(Producer.MMA_COMMIT))

        if tidx == cutlass.Int32(0):
            mb_q.init()
            mb_k.init()
            mb_v.init()
            mb_mma1.init()
            mb_mma2.init()
        nvvm.fence_mbarrier_init()
        nvvm.barrier_cta_sync()

        if warp == cutlass.Int32(0):
            tmem_alloc(tmem_ptr_i32, TOTAL_TMEM_COLS, CTA_GROUP_KIND)
        nvvm.barrier_cta_sync()
        tmem_base = tmem_ptr_i32.load()
        # MMA C/A-operand args (mma_ss/mma_ts) need a subview()-capable TMEM
        # pointer, not raw Int32 address arithmetic -- matches
        # prefill_d128_f16_sm100's tmem_raw / LAYOUT.subview() usage. The
        # plain tcgen05_ld/st calls below address TMEM by raw Int32 addition
        # instead (also matching prefill's s_addr_base/p_addr/o_addr pattern).
        tmem_raw = nvvm.make_tmem_ptr(tmem_base, cutlass.Int8)

        tma_q = GmemTileTma(tma_q_desc)
        tma_k = GmemTileTma(tma_k_desc)
        tma_v = GmemTileTma(tma_v_desc)

        q_head_base = kv_head * cutlass.Int32(HEADS_PER_TILE)
        qTmaBytes = qBufferElems * BPE
        kTmaBytes = kBufferElems * BPE
        vTmaBytes = vBufferElems * BPE

        # --- Q load: once per CTA (shared by every gathered KV block below) ---
        q_row_abs = row_base + batch * s_q  # flat row into the pre-flattened (total_q, H, D) Q tensor
        if warp == cutlass.Int32(0):
            mb_q.arrive(n_bytes=qTmaBytes, pred=nvvm.elect_sync())
            tma_load_tile(sQ, tma_q(cutlass.Int32(0), q_head_base, q_row_abs), mb_q.smem_ptr, cta_group=1)
        mb_q.wait(cutlass.Int32(0))

        desc_Q = sQ.desc()

        total_max = cutlass.Float32(NEG_INF)
        total_max_safe = cutlass.Float32(NEG_INF)
        total_sum = cutlass.Float32(0.0)
        first_done = cutlass.Int32(0)
        k_phase = cutlass.Int32(0)
        v_phase = cutlass.Int32(0)
        m1_phase = cutlass.Int32(0)
        m2_phase = cutlass.Int32(0)

        # Static Python-level unroll (round-6 fix): ``topk_max`` is a
        # compile-time-known tensor shape dim (see closure-var comment on
        # ``kernel_fn`` above), not a dynamic bound -- a bare ``for j in
        # range(topk_max):`` fully unrolls the mma_ss/mma_ts call sites at
        # DSL trace time, matching how this codebase's other frost tile_dsl
        # kernels avoid MMA calls inside a true device-side dynamic loop.
        # ``n_entries``/``is_valid`` masking below stays a genuine per-
        # iteration runtime branch -- only the loop *bound* is now static.
        # Round-7 restructure (see module docstring's round-7 note): every
        # mbarrier/pipeline primitive that ALL warps of the CTA must reach
        # (mb_k/mb_v arrive+wait, mb_mma1 arrive+wait, the CTA-wide
        # ``barrier_cta_sync``, mb_mma2 arrive+wait) now runs
        # UNCONDITIONALLY on every unrolled ``j`` iteration -- it no longer
        # sits inside the ``j < n_entries`` / ``is_valid`` guard. Only the
        # actual MMA issue (``mma_ss``/``mma_ts``), the TMEM softmax/
        # accumulation math, and the ``first_done`` flip stay predicated on
        # ``do_work``. ``resolve_entry_window`` already returns a safe,
        # in-bounds ``tile_start=0`` for an invalid/absent entry (see its
        # docstring), so an inactive slot's TMA loads block 0 of this
        # CTA's KV head -- wasted bandwidth, never touched by masked-out
        # compute, but always present so the mbarrier phase counters the
        # consumer (all 4 warps) waits on stay in lock-step with the
        # producer (warp 0) regardless of this slot's validity.
        for j in range(topk_max):
            j_i32 = cutlass.Int32(j)
            active = j_i32 < n_entries
            raw_entry = cutlass.Int32(idx_v[t_q_rep, kv_head, j_i32])
            entry = raw_entry if active else cutlass.Int32(-1)
            tile_start, num_valid, is_valid = resolve_entry_window(entry, TILE_N, kv_bound)
            do_work = is_valid
            kv_row0 = kv_base + tile_start  # always in-bounds (0 for an inactive/invalid slot)

            if warp == cutlass.Int32(0):
                mb_k.arrive(n_bytes=kTmaBytes, pred=nvvm.elect_sync())
                tma_load_tile(sK, tma_k(cutlass.Int32(0), kv_head, kv_row0), mb_k.smem_ptr, cta_group=1)
                mb_v.arrive(n_bytes=vTmaBytes, pred=nvvm.elect_sync())
                tma_load_tile(sV, tma_v(cutlass.Int32(0), kv_head, kv_row0), mb_v.smem_ptr, cta_group=1)
            mb_k.wait(k_phase)
            mb_v.wait(v_phase)
            k_phase = k_phase ^ cutlass.Int32(1)
            v_phase = v_phase ^ cutlass.Int32(1)

            desc_K = sK.desc()
            desc_V = sV.desc()

            if warp == cutlass.Int32(0):
                if do_work:
                    mma_ss(bmm1_desc, desc_Q, desc_K, tmem_raw.subview(S_OFF))
                mb_mma1.arrive(cta_group=1)
            mb_mma1.wait(m1_phase)
            m1_phase = m1_phase ^ cutlass.Int32(1)

            if do_work:
                s_addr = tmem_base + cutlass.Int32(S_OFF)
                p_addr = tmem_base + cutlass.Int32(P_OFF)
                o_addr = tmem_base + cutlass.Int32(O_OFF)

                raw_S = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr, cutlass.Float32), num=TILE_N)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                row_max_raw = row_max_reduction(raw_S)
                cur_max = row_max_raw * scale_log2

                new_max = cute.math.max(total_max, cur_max, ftz=True)
                new_max_safe = row_max_for_exp2(new_max)
                alpha = cute.math.exp2(cute.math.min(total_max_safe - new_max_safe, cutlass.Float32(0.0)), fastmath=True)
                total_max = new_max
                total_max_safe = new_max_safe

                reg_S = raw_S * scale_log2 - total_max_safe
                P = cute.math.exp2(reg_S, fastmath=True)
                sum_pair = row_reduction_pair(P)
                total_sum = total_sum * alpha + sum_pair[0] + sum_pair[1]

                P_bf16 = P.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr, cutlass.Float32), P_bf16)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

                if first_done != cutlass.Int32(0):
                    o_row = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), num=D)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    o_scaled = o_row * alpha
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), o_scaled)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

            accum_b2 = first_done != cutlass.Int32(0)
            # The P store and O rescale just above are each collective
            # across all 4 warps but only self-synchronizing per warp
            # (tcgen05_wait(STORE) only confirms THIS thread's own op) --
            # warp 0 can otherwise race ahead into mma_ts before warps
            # 1-3 have issued (let alone completed) their P/O TMEM
            # writes, since warps are independently scheduled. A real
            # CTA-wide barrier is required before the accumulate MMA
            # can safely read every warp's contribution. Unconditional
            # (round-7): every warp must reach this every iteration, same
            # reasoning as the mbarrier arrive/wait calls above.
            nvvm.barrier_cta_sync()
            if warp == cutlass.Int32(0):
                if do_work:
                    mma_ts(bmm2_desc, tmem_raw.subview(P_OFF), desc_V, tmem_raw.subview(O_OFF), accumulate=accum_b2)
                mb_mma2.arrive(cta_group=1)
            mb_mma2.wait(m2_phase)
            m2_phase = m2_phase ^ cutlass.Int32(1)

            if do_work:
                first_done = cutlass.Int32(1)

        # === Epilogue: identical LSE/dead-row/attn_sink convention as the
        # scalar and cp.async-gather sibling kernels (see module docstrings). ===
        if total_max == cutlass.Float32(NEG_INF):
            if t_row < rows_total:
                lse_v[t_q, q_head] = cutlass.Float32(NEG_INF)
                for d in cutlass.range_constexpr(D):
                    out_v[t_q, q_head, d] = cutlass.Float32(0.0).to(out.element_type)
        else:
            sink_term = cutlass.Float32(0.0)
            if cutlass.const_expr(sink_v is not None):
                sink_term = cute.math.exp(cutlass.Float32(sink_v[q_head]) - total_max, fastmath=True)
            denom = total_sum + sink_term
            inv_denom = cutlass.Float32(1.0) / denom
            o_final = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(O_OFF), cutlass.Float32), num=D)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            if t_row < rows_total:
                lse_v[t_q, q_head] = total_max + cute.math.log(total_sum, fastmath=True)
                for d in cutlass.range_constexpr(D):
                    out_v[t_q, q_head, d] = (o_final[d] * inv_denom).to(out.element_type)

        nvvm.barrier_cta_sync()
        if warp == cutlass.Int32(0):
            tmem_dealloc(tmem_ptr_i32, TOTAL_TMEM_COLS, CTA_GROUP_KIND)

    return kernel_fn, HEADS_PER_TILE, TOKENS_PER_TILE


def _make_host(heads_per_kv: int, has_topk_length: bool, has_attn_sink: bool, topk_max: int):
    kernel_fn, HEADS_PER_TILE, TOKENS_PER_TILE = _make_kernel(heads_per_kv, has_topk_length, has_attn_sink, topk_max)

    @cute.jit
    def host_fn(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        topk_idxs: cute.Tensor,
        topk_length: Optional[cute.Tensor],
        attn_sink: Optional[cute.Tensor],
        out: cute.Tensor,
        lse: cute.Tensor,
        kv_bound: cutlass.Int32,
        s_q: cutlass.Int32,
        scale: cutlass.Float32,
        rows_per_batch: cutlass.Int32,
        n_batch: cutlass.Int32,
        stream: _cuda_driver.CUstream = None,
    ) -> None:
        def _tma_swz():
            return tmap.TensorMapSwizzle.s128b

        # q/k/v are pre-flattened (batch, seq) -> total_rows by the wrapper
        # (matching gqa_prefill_bf16_tile_sm100's convention), so these are
        # rank-3 (T, H, D) tensors and the TMA box is rank-3 too -- no
        # separate batch coordinate; row coordinates below already fold the
        # batch offset in (t_q_rep / kv_row0).
        q_box = (TOKENS_PER_TILE, HEADS_PER_TILE, GRANU_ELEMS)
        kv_box = (TILE_N, 1, GRANU_ELEMS)

        tma_q_desc = tmap.create_tensor_map_tiled_from_view(
            q, box_dims=q_box, stride_order=(2, 1, 0), swizzle=_tma_swz(), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
        )
        tma_k_desc = tmap.create_tensor_map_tiled_from_view(
            k, box_dims=kv_box, stride_order=(2, 1, 0), swizzle=_tma_swz(), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
        )
        tma_v_desc = tmap.create_tensor_map_tiled_from_view(
            v, box_dims=kv_box, stride_order=(2, 1, 0), swizzle=_tma_swz(), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
        )

        n_tiles = (rows_per_batch + cutlass.Int32(TOKENS_PER_TILE) - cutlass.Int32(1)) // cutlass.Int32(TOKENS_PER_TILE)
        h_kv = q.shape[1] // heads_per_kv
        kernel_fn(
            tma_q_desc,
            tma_k_desc,
            tma_v_desc,
            topk_idxs,
            topk_length,
            attn_sink,
            out,
            lse,
            kv_bound,
            s_q,
            scale,
            rows_per_batch,
        ).launch(
            grid=(n_tiles, cutlass.Int32(h_kv), n_batch),
            block=[TILE_M, 1, 1],
            stream=stream,
        )

    return host_fn


def _gpu_arch_flag(device: torch.device) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("gqa_prefill_bf16_tcgen05_sm100 compilation requires CUDA")
    major, minor = torch.cuda.get_device_capability(device)
    if major != 10:
        raise RuntimeError(f"gqa_prefill_bf16_tcgen05_sm100 requires an SM100-family GPU, found SM{major}{minor}")
    return {0: "sm_100a", 3: "sm_103a", 7: "sm_100f"}.get(minor, "sm_100a")


@lru_cache(maxsize=None)
def _compile(h_q: int, h_kv: int, has_topk_length: bool, has_attn_sink: bool, arch: str, topk_max: int):
    # ``topk_max`` is part of the cache key (and the trace-time argument
    # list below) precisely because it is now a compile-time constant --
    # see the round-6 ``kernel_fn`` closure-var comment for why: it is the
    # static shape of ``topk_idxs``'s last dim, not a runtime value, so a
    # distinct ``topk_max`` genuinely needs its own compiled kernel (a
    # different Python-unrolled loop trip count), unlike ``h_q``/``h_kv``/
    # ``has_topk_length``/``has_attn_sink``/``arch`` which already varied
    # the cache key for the same reason (each picks a structurally
    # different compiled kernel).
    heads_per_kv = h_q // h_kv
    bf16 = cutlass.BFloat16
    t_q_sym = cute.sym_int(divisibility=1)
    t_kv_sym = cute.sym_int(divisibility=1)

    fake_q = cute.runtime.make_fake_compact_tensor(bf16, (t_q_sym, h_q, D), stride_order=(2, 1, 0), assumed_align=16)
    fake_k = cute.runtime.make_fake_compact_tensor(bf16, (t_kv_sym, h_kv, D), stride_order=(2, 1, 0), assumed_align=16)
    fake_v = cute.runtime.make_fake_compact_tensor(bf16, (t_kv_sym, h_kv, D), stride_order=(2, 1, 0), assumed_align=16)
    fake_idx = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (t_q_sym, h_kv, topk_max), stride_order=(2, 1, 0), assumed_align=4)
    fake_len = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (t_q_sym, h_kv), stride_order=(1, 0), assumed_align=4) if has_topk_length else None
    fake_sink = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (h_q,), stride_order=(0,), assumed_align=4) if has_attn_sink else None
    fake_out = cute.runtime.make_fake_compact_tensor(bf16, (t_q_sym, h_q, D), stride_order=(2, 1, 0), assumed_align=16)
    fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (t_q_sym, h_q), stride_order=(1, 0), assumed_align=4)

    host_fn = _make_host(heads_per_kv, has_topk_length, has_attn_sink, topk_max)
    return cute.compile(
        host_fn,
        fake_q,
        fake_k,
        fake_v,
        fake_idx,
        fake_len,
        fake_sink,
        fake_out,
        fake_lse,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch {arch} --opt-level 2",
    )


def _flatten_leading(t: Optional[torch.Tensor], keep_trailing: int) -> Optional[torch.Tensor]:
    if t is None:
        return None
    lead = t.shape[: t.ndim - keep_trailing]
    trail = t.shape[t.ndim - keep_trailing :]
    return t.reshape((math.prod(lead),) + trail) if len(lead) > 1 else t


def fast_path_eligible(*, d_k: int, d_v: int, h_q: int, h_kv: int, index_granularity: int) -> bool:
    """Cheap, side-effect-free eligibility probe for ``dispatch.py``.

    MSA-cell envelope only: D_k == D_v == 128, granularity == 128, G == H_kv
    (H_kv > 1), and heads_per_kv must divide the MMA M tile (128).
    """
    if d_k != D or d_v != D or index_granularity != TILE_N or h_kv <= 1 or h_q % h_kv != 0:
        return False
    heads_per_kv = h_q // h_kv
    return TILE_M % heads_per_kv == 0


def sparse_attention_forward_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    index_granularity: int = 128,
    softmax_scale: Optional[float] = None,
    uniform_within_tile: bool = False,
    validate_uniform: bool = False,
    stream=None,
) -> dict:
    """Real tcgen05 tensor-core mainloop for the PR4 MSA cell -- see module
    docstring for the exact envelope. ``uniform_within_tile=True`` is a
    **required, caller-verified precondition** (same contract as
    ``gqa_prefill_bf16_tile_sm100``'s fast path): every row/head in a Q tile
    must share one ``topk_idxs``/``topk_length`` row. ``validate_uniform=True``
    adds one explicit, opt-in D2H-synchronizing host-side check (same recipe
    as ``gqa_prefill_bf16_tile_sm100._check_uniform_within_tile``) and raises
    ``ValueError`` on violation -- never the default hot path, but required
    on ``dispatch.py``'s probe (which decides FOR the caller, not on the
    caller's own verified assertion).
    """
    if not uniform_within_tile:
        raise ValueError(
            "gqa_prefill_bf16_tcgen05_sm100 only serves uniform_within_tile=True "
            "(per-Q-tile row-uniform selection, e.g. MSA-style block attention); "
            "for the general per-row-varying case use gqa_prefill_bf16_sm100's scalar kernel"
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError(f"gqa_prefill_bf16_tcgen05_sm100 is BF16-only, got Q/K/V dtypes {q.dtype}/{k.dtype}/{v.dtype}")
    if q.ndim != 4:
        raise NotImplementedError("gqa_prefill_bf16_tcgen05_sm100 is BSHD-only this round (THD is a scoped follow-up); use the scalar or cp.async-gather kernel for THD")
    if index_granularity != TILE_N:
        raise ValueError(f"gqa_prefill_bf16_tcgen05_sm100 only serves index_granularity == {TILE_N}, got {index_granularity}")

    device = q.device
    if device.type != "cuda":
        raise ValueError(f"Q must live on CUDA, got {device}")

    with torch.cuda.device(device):
        arch = _gpu_arch_flag(device)

        b, s_q_, h_q, d_k = q.shape
        _, s_kv, h_kv, d_k_kv = k.shape
        _, _, _, d_v = v.shape
        rows_per_batch, n_batch = s_q_, b
        kv_bound = s_kv
        s_q = s_q_

        if d_k_kv != d_k:
            raise ValueError(f"K head dim ({d_k_kv}) must match Q ({d_k})")
        if not fast_path_eligible(d_k=d_k, d_v=d_v, h_q=h_q, h_kv=h_kv, index_granularity=index_granularity):
            raise NotImplementedError(f"gqa_prefill_bf16_tcgen05_sm100 envelope rejects D_k={d_k} D_v={d_v} H_q={h_q} H_kv={h_kv}")
        if topk_idxs.shape[-2] != h_kv:
            raise ValueError(f"topk_idxs group dim must be H_kv ({h_kv}) for this kernel's envelope, got {topk_idxs.shape}")

        q_flat = _flatten_leading(q, 2)
        k_flat = _flatten_leading(k, 2)
        v_flat = _flatten_leading(v, 2)
        idx_flat = _flatten_leading(topk_idxs, 2)
        len_flat = _flatten_leading(topk_length, 1)

        if validate_uniform:
            tokens_per_tile = TILE_M // (h_q // h_kv)
            _check_uniform_within_tile(idx_flat, len_flat, rows_per_batch, n_batch, tokens_per_tile)

        q_flat = q_flat.contiguous()
        k_flat = k_flat.contiguous()
        v_flat = v_flat.contiguous()
        idx_flat = idx_flat.contiguous()
        if len_flat is not None:
            len_flat = len_flat.contiguous()
        if attn_sink is not None:
            attn_sink = attn_sink.contiguous()

        total_q = rows_per_batch * n_batch
        out = torch.empty((total_q, h_q, d_v), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, h_q), dtype=torch.float32, device=device)

        scale = 1.0 / math.sqrt(d_k) if softmax_scale is None else float(softmax_scale)
        log2_e = math.log2(math.e)
        scale_log2 = scale * log2_e
        topk_max = idx_flat.shape[-1]

        compiled = _compile(int(h_q), int(h_kv), len_flat is not None, attn_sink is not None, arch, int(topk_max))

        cu_stream = stream if stream is not None else _cuda_current_stream(device)
        compiled(
            q_flat,
            k_flat,
            v_flat,
            idx_flat,
            len_flat,
            attn_sink,
            out,
            lse,
            cutlass.Int32(int(kv_bound)),
            cutlass.Int32(int(s_q)),
            cutlass.Float32(scale_log2),
            cutlass.Int32(int(rows_per_batch)),
            cutlass.Int32(int(n_batch)),
            cu_stream,
        )

    return {"out": out.reshape(b, s_q_, h_q, d_v), "lse": lse.reshape(b, s_q_, h_q)}


def _check_uniform_within_tile(
    idx_flat: torch.Tensor,
    len_flat: Optional[torch.Tensor],
    rows_per_batch: int,
    n_batch: int,
    tokens_per_tile: int,
) -> None:
    """Opt-in, D2H-synchronizing precondition check for ``validate_uniform=True``
    -- same recipe as ``gqa_prefill_bf16_tile_sm100._check_uniform_within_tile``,
    just against this module's (generally narrower) ``tokens_per_tile``."""
    n_tiles = (rows_per_batch + tokens_per_tile - 1) // tokens_per_tile
    for b in range(n_batch):
        base = b * rows_per_batch
        for t in range(n_tiles):
            r0 = base + t * tokens_per_tile
            r1 = min(base + rows_per_batch, r0 + tokens_per_tile)
            if r1 <= r0:
                continue
            tile_idx = idx_flat[r0:r1]
            if not torch.equal(tile_idx, tile_idx[0:1].expand_as(tile_idx)):
                raise ValueError(f"validate_uniform: topk_idxs not uniform within Q tile rows [{r0}, {r1}) -- uniform_within_tile=True contract violated")
            if len_flat is not None:
                tile_len = len_flat[r0:r1]
                if not torch.equal(tile_len, tile_len[0:1].expand_as(tile_len)):
                    raise ValueError(f"validate_uniform: topk_length not uniform within Q tile rows [{r0}, {r1}) -- uniform_within_tile=True contract violated")


def _cuda_current_stream(device: torch.device):
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
