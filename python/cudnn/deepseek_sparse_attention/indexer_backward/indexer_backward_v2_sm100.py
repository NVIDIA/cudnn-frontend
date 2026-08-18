# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indexer backward v2 — SM100 CuTe-DSL, persistent dynamic-ticket kernel.

Opt-in alternative backend for ``IndexerBackward`` (selected via
``backend="sm100_v2"`` in ``api.py``). It keeps the exact 3-stage wrapper contract of
``indexer_backward_sm100`` — kernel 1 (score-grad precompute) is literally
shared with the default backend — and replaces the GEMM stage (kernel 2) with
a **two-term bf16-expansion** formulation:

* ``weights`` are upcast to fp32 in-register (exact) and the per-slot fp32
  gradient matrix ``A = grad_signal * weights`` is split into a two-term
  bf16 expansion ``hi + lo`` (``hi = bf16(A)``, ``lo = bf16(A - hi)``)
  before the MMA. Each individual bf16 x bf16 product in the dQ / dK
  contractions is then exact in the fp32 accumulator when it is finite
  there (<= 16 significand bits); the expansion itself is NOT exact — ``hi + lo`` carries ~16 of
  ``A``'s 24 significand bits, so the summed result is not the correctly
  rounded ``A @ K``. Measured on 1e6 random fp32/bf16 pairs, the two-term
  expansion reduces the representation error of ``A`` (and the resulting
  per-product error) from 1.66e-3 to 2.46e-6 rms relative (~675x) over the
  default backend's single-bf16 rounding of ``A``, which dominates its fp32
  dK error and adds to its bf16-stored dQ error. That ratio is an aggregate
  over that random population, not a
  per-element guarantee: individual values with a small ``lo`` term gain
  less (e.g. A = 1.7880821228027344 gains ~257x). The gain is bounded by
  bf16's exponent range, not just its mantissa: ``lo`` is ~2^-9 of ``hi``,
  so it underflows for |A| below ~1e-35 (measured 518x at 1e-35, 63x at
  1e-36, 1x at 1e-38), and the fp32 -> bf16 conversion is non-saturating, so
  |A| at or above bf16's round-to-nearest overflow threshold
  (2^128 - 2^119 ~= 3.3962e38, itself above bf16's 3.3895e38 maximum) becomes
  an infinity in ``hi`` and the opposite infinity in ``lo`` (NaN in ``lo`` if
  ``A`` was already infinite), which the two MMA terms then sum to NaN
  instead of clamping.
* ``d_weights`` is accumulated in fp32 and reduced **deterministically**
  in-CTA (per-warp butterfly, smem partial exchange, fixed-order combine,
  plain store) — bitwise run-to-run stable. The final store rounds to the
  caller's ``d_weights`` dtype (fp32 buffers receive the accumulator
  verbatim; bf16 buffers receive its bf16 rounding — supply an fp32 buffer
  to keep the full accuracy gain).
* ``d_index_k`` is accumulated with vectorized fp32 atomics (four-element
  ``cute.arch.atomic_add``), the same numerics class as the default backend,
  into the caller's fp32 buffer (zeroed here) or a per-call fp32 scratch
  that is cast to a bf16 output buffer.

Semantics (H = indexer heads, D = head dim; per query row ``s`` and top-k
slot ``t`` with id ``n = idx[s, t]``; invalid slots contribute nothing —
``n`` is invalid iff ``n < 0`` or ``n >= S_k`` where the bound is the
per-batch ``S_k`` for local ids (``topk_indices_global=False``, masked
in-kernel BEFORE the batch offset is applied, so a positive out-of-range
local id can never alias a neighbouring batch) or the flattened
``B * S_k`` for global ids):

    S[t,h]    = sum_d K[n,d] * Q[s,h,d]             (fp32 acc)
    g'[s,t]   = sm_scale * g[s,t]                   (runtime fold, kernel 2)
    dW[s,h]   = sum_t g'[s,t] * max(S[t,h], 0)      (fp32, deterministic)
    A[t,h]    = (S[t,h] > 0) ? g'[s,t] * w[s,h] : 0 (fp32, split hi+lo bf16)
    dQ[s,h,d] = sum_t A[t,h] * K[n,d]               (bf16 out)
    dK[n,d]  += sum_h A[t,h] * Q[s,h,d]             (fp32 atomic)

``sm_scale`` is a runtime kernel-2 argument folded multiplicatively into the
grad-signal read (no recompilation across values, and the ``attn_score``
scratch is left holding exactly kernel 1's ``grad_signal``, identical to the
default backend). The relu gate reads the unscaled ``S``, which matches the
default backend's gate on ``sm_scale * S`` for positive scales, up to the
underflow corner documented at the kernel-2 gate itself — ``check_support``
therefore requires ``sm_scale > 0`` for this backend.

Kernel design — persistent grid (one CTA per SM), 512 threads / 16 warps
per CTA, dynamic row tickets:

    warp  0     : TMA Q row tile (64 x 128 bf16), double-buffered across rows
    warp  1     : MMA — software-pipelined issue of the three tcgen05
                  contractions per 128-slot tile (the hi and lo terms of
                  MMA2 / MMA3 are separate accumulating issues); S runs up
                  to 2 tiles ahead of its consumer (one MMA warp feeding
                  multiple pipelines, latency hidden by buffer depth):
                    MMA1  S(128x64)    = Ksel(128x128) @ Q^T      (K,K)
                    MMA2  dQT(128x64) += Ksel^T @ A_hi, then A_lo (MN,MN) acc
                    MMA3  dKp(128x128) = A_hi @ Q^T-view, += A_lo (K,MN)
                  Ksel^T / Q^T-view / A^T are MN-major descriptor views of
                  the same smem (``recast_ptr`` idiom).
    warp  2     : dynamic ticket writer — lane 0 draws global row tickets
                  (a scalar global atomic fetch-add) into a smem ring
                  consumed by all warpgroups
    warps 4-7   : S-epilogue — TMEM S -> RF (thread == slot row), relu, fp32
                  dW accumulation, A split hi+lo -> smem; then the
                  deterministic dW combine and the dQT TMEM -> bf16 dQ store
    warps 8-11  : metadata restage (idx/grad, parity double-buffered) +
                  sparse K gather (cp.async 16B, invalid rows zero-filled so
                  garbage never reaches S/dW/dQ)
    warps 12-15 : dK reduce — TMEM dKp -> RF -> a four-element fp32 atomic
                  add per valid row, two independent select/shuffle/reduce
                  chains per column block to expose ILP to the scheduler

The ticket ring makes the row <-> CTA assignment dynamic (work stealing),
which removes the end-of-kernel tail imbalance of a static row stripe. All
warpgroups consume the identical ticket sequence, so the cross-row skew
pipeline (warps 0-11 run row i+1's front while warps 12-15 drain row i's dK
tail) is preserved; cross-row hazards ride the existing mbarrier pipelines
whose phases roll across rows without reset. Ring arrivals are warp-elected
(14 consumer warps arrive via lane 0 each) and the ring depth bounds the
writer's pre-commit lead over the TMA front.

Requires SM100, H == 64, D == 128, and topk % 128 == 0 with
128 <= topk <= 2048:

* ``topk % 128 == 0`` — 128-slot tiles; the metadata restage loop moves 512
  elements/step with a predicated tail covering any 128-multiple;
* ``topk >= 128`` (>= 1 tile/row) — for T >= 3 the S-epilogue warpgroup's sG
  reads ride the implicit A->K pipe chain, which covers the previous row's
  last tile only from 3 tiles/row upward, and the pipe depths keep their full
  values. For T < 3 (topk 128/256) the K/S pipes and ``mma_lookahead`` are
  clamped to the tile count (see ``__init__``) so a whole row's tile sequence
  is resident at once and the S chain does not lap within the row. The dK
  warpgroup's sIdx reads are protected by a parity-slot WAR barrier that is
  EXPLICIT where it is load-bearing (``pipe_M``, compiled in for <= 7
  tiles/row: the gather warpgroup waits on a metadata slot before restaging
  it; the dK warpgroup releases the slot only after its last sIdx read of the
  row) and left to the K->S->A->DK acquire-chain arithmetic for >= 8
  tiles/row, where that chain is long enough by the tile-count arithmetic
  in ``__init__`` (a conservative bound, not a measured boundary).
  The explicit barrier is load-bearing in the low-tile regime: forcing it off
  corrupts ``d_index_k`` only (dQ and dW keep the same error to four digits at
  the same seed), by 8.1e-3 .. 5.9e-1 rms relative against an fp64 recompute
  where it is otherwise 2.4e-6, because the restage of a later metadata slot
  laps the dK drain of an earlier one and the dK scatter reads overwritten
  sIdx. Measured at 2 and 3 tiles/row (topk 256/384 at S_q = 512, S_k = 4096;
  5 of 5 runs each); at 1, 4 and 5 tiles/row (topk 128/512/640, same shape)
  forcing it off reproduced the barriered result exactly in 3 of 3 runs, so
  ``<= 7`` is a conservative bound, not a measured boundary. Being a race, the
  rate depends strongly on the shape: at the S_q = 128 / S_k = 512 shape the
  other v2 tests use it appeared in only 1 of 5 seeds, against 5 of 5 at
  S_q = 512 / S_k = 4096 — which is the shape
  ``test_DSA_indexer_backward_wrapper_v2_low_tile_metadata_war`` forces. The
  low-tile regime (topk 128/256/384) is additionally checked against an fp64
  oracle by ``test_DSA_indexer_backward_wrapper_v2``, at its own smaller shape;
* ``topk <= 2048`` (16 tiles/row) — the SM100 dynamic-smem cap: 2048 fills
  the 232448 B CTA budget exactly (the ticket ring lives in the alignment
  padding between the dW partials and the 1024-aligned sQ buffer, so it
  costs no extra bytes); 2176 would need 234496 B.

``api.py`` raises before compile otherwise.

Workspace ownership / concurrency contract: the dynamic-ticket counter is
**per-plan workspace**, allocated once when the plan first executes and reused
by every subsequent execute of that plan. (The fp32 dK accumulator a bf16
``d_index_k`` needs is not per-plan: it comes from the caching allocator on
every call.) The kernel self-resets the counter (the CTA
that draws the final raw ticket stores 0), so back-to-back launches need no
host reset — but that also means executions of the SAME plan must be
serialized: they may target any stream, provided the launches do not overlap
on the device (in-flight overlap of two launches sharing one counter would
interleave ticket draws/reset). The workspace is device-resident: the plan
binds the device of its first execute and rejects any indexer tensor from
another device (one plan serves one device). ``indexer_backward_wrapper``
enforces
both by keying its plan cache on the CUDA device and on the *resolved*
stream (``stream`` if given, else ``torch.cuda.current_stream()`` at call
time — the same resolution this module uses to pick the launch stream, plus
the calling thread's id for ``cudaStreamPerThread``, the one handle CUDA does
not make unique across host threads), so each device/stream owns a private
plan/counter and same-stream executes are stream-ordered. Users driving ``IndexerBackward``
objects directly across streams (or replaying captured graphs concurrently)
must use one object per device, and one per stream wherever those streams'
executions can overlap (a single object may target any stream as long as its
executions are serialized). A launch killed before its final ticket draw
resets the counter can leave it non-zero; re-creating the plan (or the
process) clears it.
"""

from __future__ import annotations

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import BFloat16, Float32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import (
    resolve_stream as _resolve_stream,
    torch_stream_context as _torch_stream_context,
)
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

from .indexer_backward_sm100 import _score_grad_inplace


@dsl_user_op
def _cvt_f32x2_bf16x2_rn_pure(
    src0: Float32,
    src1: Float32,
    *,
    loc=None,
    ip=None,
) -> ir.Value:
    """Pack two scalar round-to-nearest FP32->BF16 conversions into one
    ``cvt.rn.bf16x2.f32`` (raw PTX, NON-satfinite -> bit-identical to the
    scalar ``BFloat16(x)`` cvt.rn.bf16.f32 element-wise; the satfinite
    bf16x2 variant flushes NaN payloads and saturates infinities, which
    would silently change the hi/lo split). PURE (no side-effect fence):
    the conversion carries no ordering constraint, and an ordered inline-asm
    variant forces ptxas to serialize the register-heavy S-epilogue.
    Returned vector[0] = BFloat16(src0), [1] = BFloat16(src1)."""
    packed = llvm.inline_asm(
        Int32.mlir_type,
        [
            Float32(src1).ir_value(loc=loc, ip=ip),
            Float32(src0).ir_value(loc=loc, ip=ip),
        ],
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    vec_type = ir.VectorType.get([2], BFloat16.mlir_type, loc=loc)
    return llvm.bitcast(vec_type, packed, loc=loc, ip=ip)


class IndexerBackwardV2Sm100:
    """Persistent gather-GEMM indexer backward with dynamic row tickets."""

    arch = 100

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        topk: int,
        num_ctas: int,
        ticket_slots: int,
        ring_depth: int = 4,
        dw_out_bf16: bool = False,
        idx_local: bool = False,
    ):
        assert num_heads == 64, "specialized for H=64"
        assert head_dim == 128, "specialized for D=128"
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.topk = topk
        self.n_block = 128
        # dW output dtype variant: fp32 buffers receive the deterministic
        # accumulator verbatim; bf16 buffers receive its bf16 rounding at
        # the store, which is the same value a separate fp32 scratch buffer
        # plus a cast kernel would produce.
        self.dw_out_bf16 = bool(dw_out_bf16)
        # idx_local: sIdx holds per-batch local ids. They are masked against
        # the per-batch S_k BEFORE the batch offset is applied (in-kernel),
        # so positive out-of-range local ids contribute nothing instead of
        # aliasing the next batch. False: sIdx holds global flat ids checked
        # against B * S_k. (B == 1 local ids compile the global variant —
        # offset 0 and identical bound.)
        self.idx_local = bool(idx_local)
        # 128-slot tiles; the restage loop moves 512 elements/step with a
        # predicated constexpr tail covering any 128-multiple.
        assert topk > 0 and topk % 128 == 0, topk
        self.num_tiles = topk // self.n_block
        # >= 1 tile/row. For T >= 3 the S-epilogue's sG reads ride the implicit
        # A->K pipe chain (which covers the previous row's last tile only from
        # T >= 3 upward) and the pipe depths keep their full production values.
        # For T < 3 (topk 128/256) the S/K pipes and mma_lookahead are clamped
        # to the tile count below, so a whole row's tile sequence is resident
        # at once and there is no intra-row lap of the S chain; the dK-side
        # metadata WAR is then covered by the explicit pipe_M barrier (compiled
        # in for T <= 7 regardless). The low-tile regime is checked against a
        # strict fp64 recompute by the topk 128/256/384 cases of
        # test_DSA_indexer_backward_wrapper_v2 in
        # test/python/fe_api/dsa/test_DSA_indexer_backward.py.
        assert self.num_tiles >= 1, self.num_tiles
        # metadata parity slots (sIdx/sG double buffer + pipe_M stages)
        self.meta_stage = 2

        # Persistent schedule parameters. ``num_ctas`` is the grid size (one
        # CTA per SM); every CTA runs exactly ``ticket_slots`` ring slots, so
        # ``ticket_slots * num_ctas >= rows`` must hold — the surplus slots
        # come back as -1 tickets: no row work, but each still costs the
        # writer's atomic draw plus one ring publish and 14 consumer releases.
        # ``ring_depth`` bounds how far the ticket writer can pre-commit
        # ahead of the slowest consumer warp.
        assert num_ctas >= 1, num_ctas
        assert ticket_slots >= 1, ticket_slots
        assert 2 <= ring_depth <= ticket_slots, (ring_depth, ticket_slots)
        self.num_ctas = num_ctas
        self.ticket_slots = ticket_slots
        self.ring_depth = ring_depth
        # Idle warp 2 of warpgroup 0 is the ticket writer; the ring is
        # consumed by 14 warps (TMA + MMA + 4 each for warpgroups 1/2/3),
        # each arriving via its elected lane 0 only. The producer <=
        # min(all consumers) + ring invariant is unchanged by the elected
        # arrival, and the ring read is warp-uniform converged code, so
        # lanes 1..31 have always read the slot by the time lane 0's
        # arrive can enable its reuse.
        self.ticket_warp_id = 2
        self.ring_consumer_arrivals = 14

        # (M, N, K) tilers for the three contractions
        self.tiler_S = (self.n_block, num_heads, head_dim)  # S    = Ksel @ Q^T
        self.tiler_dQ = (head_dim, num_heads, self.n_block)  # dQT += Ksel^T @ A
        self.tiler_dK = (self.n_block, head_dim, num_heads)  # dKp  = A @ Q^T-view

        # Pipeline depths. The single MMA warp issues the S contraction as
        # an async producer running ``mma_lookahead`` (= 2) tiles ahead of
        # the S-epilogue consumer, so the S TMEM accumulators and the K
        # gather pipe are 3 deep; the A and dK pipes stay 2 deep (TMEM is
        # exactly full at 512 columns). sQ is double-buffered so the TMA
        # fetch of row i+1 only waits on the MMA-warp release from row i-1,
        # keeping the Q fetch off the row-boundary critical chain.
        # For T >= 3 (topk >= 384) these min-clamps are all no-ops, so the
        # schedule is the unclamped one; for
        # T < 3 (topk 128/256) the S/K pipes and the lookahead shrink to the
        # tile count so a whole row is resident at once. a_stage/dk_stage stay
        # 2 unconditionally: the odd-tile dK drain references tAccdK[1] and the
        # T==2 paired drain rolls by dk_stage==2, so both need two dK
        # accumulators even at T in {1, 2}.
        self.q_stage = 2
        self.kv_stage = min(3, self.num_tiles)
        self.a_stage = 2
        self.s_stage = min(3, self.num_tiles)
        self.dk_stage = 2
        self.mma_lookahead = min(2, self.num_tiles)
        # Explicit metadata WAR barrier gate. The implicit K->S->A->DK chain
        # orders the gather warpgroup's restage of valid row i+2 behind the
        # dK warpgroup's completed sIdx reads of tile (i+2)*T - kv_stage -
        # dk_stage - 3; that covers row i's last tile (i*T + T - 1) iff
        # T >= kv_stage + dk_stage + 2 (== 7 with these stages). pipe_M is
        # compiled in below that bound PLUS one tile of margin (explicit for
        # T <= 7), so the regime that relies on the implicit chain is strictly
        # inside the arithmetic proof. Forcing pipe_M off at T == 2 breaks
        # d_index_k (8.1e-3 .. 5.2e-2 vs 2.4e-6 rms relative against an fp64
        # recompute, 5 of 5 runs at S_q = 512 / S_k = 4096; see the module
        # docstring), which is what makes the explicit barrier load-bearing
        # here.
        self.meta_war_explicit = self.num_tiles < self.kv_stage + self.dk_stage + 3

        self.load_warp_id = 0
        self.mma_warp_id = 1
        self.epi_warp_ids = (4, 5, 6, 7)
        self.gather_warp_ids = (8, 9, 10, 11)
        self.dk_warp_ids = (12, 13, 14, 15)
        self.num_warps = 16
        self.threads_per_cta = 32 * self.num_warps

        # TMEM column map (fp32 cols): S slots, dQT, dK slots. 512 = full.
        self.tmem_S_off = (0, 64, 448)
        self.tmem_dQ_off = 128
        self.tmem_dK_off = (192, 320)
        self.tmem_alloc_cols = 512

        # Warp-specialized register budgets. ptxas honors setmaxnreg only when
        # it can derive the entry register count from a thread-count bound; the
        # DSL emits that as ``.reqntid`` from ``block=`` at the launch below.
        # The epilogue warpgroup carries the S-epilogue, the hi/lo conversions
        # and the dQ drain, hence the asymmetric split.
        self.num_regs_wg0 = 32
        self.num_regs_epi = 256
        self.num_regs_gather = 64
        self.num_regs_dk = 160

        # participants: MMA warp + S-epi warps + dK warps (TMEM users)
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * (1 + len(self.epi_warp_ids) + len(self.dk_warp_ids)),
        )
        # TMEM dealloc gate: S-epi + dK warps
        self.tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * (len(self.epi_warp_ids) + len(self.dk_warp_ids)),
        )
        # dW partial exchange between the 4 S-epi warps. Split-phase: each
        # warp arrive()s right after staging its smem partial
        # (non-blocking), the combine arrive_and_wait()s after the dQ
        # store -> count = 2 x 128.
        self.epi_dw_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=2 * 32 * len(self.epi_warp_ids),
        )
        # Metadata restage barrier, internal to the gather warpgroup (128
        # threads). The gather warpgroup alone restages sIdx/sG (vectorized
        # 16B); the S-epilogue warpgroup reads them through the K->S pipe
        # acquire chain, the dK warpgroup through the A->DK chain, so
        # neither joins this barrier and it never stalls them directly
        # (those pipe chains, not this barrier, are what order the next row's
        # front behind the restage). The S-epilogue warpgroup loads its
        # weights row
        # straight from gmem (mW is read-only), so weights are never staged
        # in smem.
        self.stage_barrier = pipeline.NamedBarrier(barrier_id=5, num_threads=128)

    # -----------------------------------------------------------------
    # host side
    # -----------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (S, H, D) bf16
        mK: cute.Tensor,  # (SK, D) bf16
        mW: cute.Tensor,  # (S, H) bf16 (upcast to fp32 in-register, exact)
        mIdx: cute.Tensor,  # (S, topk) int32
        mG: cute.Tensor,  # (S, topk) fp32
        mdQ: cute.Tensor,  # (S, H, D) bf16 out
        mdW: cute.Tensor,  # (S, H) out — fp32 or bf16 per dw_out_bf16
        mdK: cute.Tensor,  # (SK, D) fp32 out (zero-initialized)
        sm_scale: Float32,  # runtime fold into the grad signal (positive)
        s_q_per_batch: Int32,  # rows per batch (idx_local row -> batch map)
        s_k_local: Int32,  # per-batch KV extent (idx_local mask bound)
        stream: cuda.CUstream,
        mCnt: cute.Tensor,  # (2,) i32 persistent ticket counter
    ):
        # (S, H, D) -> (H, D, S): B-operand view (N=heads, K=dim, rest=row)
        mQ_v = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 2, 0]))

        cta_group = tcgen05.CtaGroup.ONE
        mma_S = sm100_utils.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            tcgen05.OperandMajorMode.K,  # A = gathered K rows (slot, dim)
            tcgen05.OperandMajorMode.K,  # B = Q rows (head, dim)
            Float32,
            cta_group,
            self.tiler_S[:2],
        )
        mma_dQ = sm100_utils.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            tcgen05.OperandMajorMode.MN,  # A = Ksel^T view (dim, slot)
            tcgen05.OperandMajorMode.MN,  # B = A_mat^T view (head, slot)
            Float32,
            cta_group,
            self.tiler_dQ[:2],
        )
        mma_dK = sm100_utils.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            tcgen05.OperandMajorMode.K,  # A = A_mat (slot, head)
            tcgen05.OperandMajorMode.MN,  # B = Q^T view (dim, head)
            Float32,
            cta_group,
            self.tiler_dK[:2],
        )

        sK_layout = sm100_utils.make_smem_layout_a(mma_S, self.tiler_S, BFloat16, self.kv_stage)
        sKT_layout = sm100_utils.make_smem_layout_a(mma_dQ, self.tiler_dQ, BFloat16, self.kv_stage)
        sQ_layout = sm100_utils.make_smem_layout_b(mma_S, self.tiler_S, BFloat16, self.q_stage)
        sQT_layout = sm100_utils.make_smem_layout_b(mma_dK, self.tiler_dK, BFloat16, self.q_stage)
        sA_layout = sm100_utils.make_smem_layout_a(mma_dK, self.tiler_dK, BFloat16, self.a_stage)
        sAT_layout = sm100_utils.make_smem_layout_b(mma_dQ, self.tiler_dQ, BFloat16, self.a_stage)

        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma_S.thr_id.shape,))
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ_v,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.tiler_S,
            mma_S,
            cluster_layout_vmnk.shape,
        )
        self.tma_copy_q_bytes = self.num_heads * self.head_dim * (BFloat16.width // 8)

        self.kernel(
            mma_S,
            mma_dQ,
            mma_dK,
            tma_atom_Q,
            tma_tensor_Q,
            mK,
            mW,
            mIdx,
            mG,
            mdQ,
            mdW,
            mdK,
            sQ_layout,
            sK_layout,
            sKT_layout,
            sQT_layout,
            sA_layout,
            sAT_layout,
            mCnt,
            sm_scale,
            s_q_per_batch,
            s_k_local,
        ).launch(
            grid=(self.num_ctas, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    # -----------------------------------------------------------------
    # device side — persistent dynamic-ticket schedule
    # -----------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        mma_S: cute.TiledMma,
        mma_dQ: cute.TiledMma,
        mma_dK: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        tma_tensor_Q: cute.Tensor,
        mK: cute.Tensor,  # (SK, D) bf16
        mW: cute.Tensor,  # (S, H) bf16
        mIdx: cute.Tensor,  # (S, topk) int32
        mG: cute.Tensor,  # (S, topk) fp32
        mdQ: cute.Tensor,  # (S, H, D) bf16
        mdW: cute.Tensor,  # (S, H) fp32 or bf16 (dw_out_bf16)
        mdK: cute.Tensor,  # (SK, D) fp32
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKT_layout: cute.ComposedLayout,
        sQT_layout: cute.ComposedLayout,
        sA_layout: cute.ComposedLayout,
        sAT_layout: cute.ComposedLayout,
        mCnt: cute.Tensor,  # (2,) i32 ticket counter
        sm_scale: Float32,
        s_q_per_batch: Int32,
        s_k_local: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        seq_k = cute.size(mdK.shape[0])
        seq_len = cute.size(mIdx.shape[0])

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_Q)

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.q_stage]
            K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.kv_stage]
            A_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.a_stage]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.s_stage]
            DK_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dk_stage]
            DQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            # sIdx/sG parity-slot WAR pipe (gather WG producer / dK WG
            # consumer); inert 32B unless meta_war_explicit
            M_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.meta_stage]
            tmem_holding_buf: cutlass.Int32
            # sIdx/sG parity double-buffered: the dK warpgroup (and the
            # S-epilogue's late sG reads) still consume row i while the
            # gather warpgroup restages row i+1.
            # No sW: the S-epilogue loads its weights row straight from gmem.
            sIdx: cute.struct.Align[cute.struct.MemRange[Int32, 2 * self.topk], 16]
            sG: cute.struct.Align[cute.struct.MemRange[Float32, 2 * self.topk], 16]
            # per-epi-warp dW partials, parity double-buffered (row i's
            # fixed-order combine reads vs row i+1's restage)
            sdWp: cute.struct.Align[
                cute.struct.MemRange[Float32, 2 * self.num_heads * len(self.epi_warp_ids)],
                16,
            ]
            # Ticket ring (mbar pairs + row ids). Placed HERE — between
            # sdWp and the 1024-aligned sQ — so it sits inside the
            # pre-existing alignment-padding hole (768B at this config;
            # the ring fits for depth <= 38): ZERO smem growth and
            # identical sQ/sK/sAhi/sAlo offsets. As a separate trailing
            # allocation it added 80B past the sAlo end, which pushed
            # topk == 2048 to 232528B — 80B over the SM100 232448B
            # dynamic-smem cap.
            T_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.ring_depth]
            sRowQ: cute.struct.Align[cute.struct.MemRange[Int32, self.ring_depth], 16]
            sQ: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sK_layout)], 1024]
            sAhi: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sA_layout)], 1024]
            sAlo: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sA_layout)], 1024]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # ---- pipelines (phases roll across rows without reset) ----
        pipe_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_copy_q_bytes,
            defer_sync=True,
        )
        pipe_K = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.K_mbar_ptr.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            defer_sync=True,
        )
        pipe_A = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.A_mbar_ptr.data_ptr(),
            num_stages=self.a_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            defer_sync=True,
        )
        pipe_S = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            num_stages=self.s_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            defer_sync=True,
        )
        pipe_DK = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.DK_mbar_ptr.data_ptr(),
            num_stages=self.dk_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            defer_sync=True,
        )
        pipe_DQ = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.DQ_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            defer_sync=True,
        )
        # pipe_M: EXPLICIT sIdx/sG parity-slot WAR protection, compiled in
        # iff meta_war_explicit (num_tiles <= 7; see __init__ for the bound
        # derivation). The implicit K->S->A->DK acquire chain is too short
        # there to delay the gather warpgroup's restage of row i+2 past the
        # dK warpgroup's last sIdx read of row i (measured at topk=256 -> 2
        # tiles: forcing this barrier off makes the restage lap the drain, the
        # dK scatter reads overwritten sIdx, and dk rms_rel goes 2.4e-6 ->
        # 8.1e-3 .. 5.2e-2). Consumer = the dK warpgroup: after the
        # last sIdx read of a row, each warp syncs and its lane 0 arrives
        # on the slot's empty barrier (4 arrivals). Producer = the gather
        # warpgroup: before restaging a parity slot, each warp's lane 0
        # waits that barrier (sync_warp holds the warp's stores behind it).
        # States advance once per VALID row, so pipeline index == parity
        # buffer and phases roll across rows exactly like every other pipe
        # here. The full direction is unused (the dK warpgroup's visibility
        # of the fresh sIdx rides the K->S->A->DK chain as before).
        if const_expr(self.meta_war_explicit):
            pipe_M = pipeline.PipelineAsync.create(
                barrier_storage=storage.M_mbar_ptr.data_ptr(),
                num_stages=self.meta_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.dk_warp_ids)),
                defer_sync=True,
            )
        # Ticket ring: producer = idle warp 2 lane 0, consumers = the 14
        # worker warps (TMA + MMA + 4 each for warpgroups 1/2/3), each
        # arriving via its elected lane 0 once per slot. The ring depth is
        # the writer's pre-commit lead over the slowest consumer's loop top
        # (the writer leads the front by <= q_stage + 1 rows; the dK
        # warpgroup's loop top lags < 2 rows).
        pipe_T = pipeline.PipelineAsync.create(
            barrier_storage=storage.T_mbar_ptr.data_ptr(),
            num_stages=self.ring_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.ring_consumer_arrivals),
            defer_sync=True,
        )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epi_warp_ids[0],
        )

        pipeline.pipeline_init_arrive(is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sAhi = storage.sAhi.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sAlo = storage.sAlo.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        # MN-major descriptor views over the same smem
        sKT = cute.make_tensor(cute.recast_ptr(sK.iterator, sKT_layout.inner), sKT_layout.outer)
        sQT = cute.make_tensor(cute.recast_ptr(sQ.iterator, sQT_layout.inner), sQT_layout.outer)
        sAThi = cute.make_tensor(cute.recast_ptr(sAhi.iterator, sAT_layout.inner), sAT_layout.outer)
        sATlo = cute.make_tensor(cute.recast_ptr(sAlo.iterator, sAT_layout.inner), sAT_layout.outer)
        sIdx = storage.sIdx.get_tensor(cute.make_layout((self.topk, 2)))
        sG = storage.sG.get_tensor(cute.make_layout((self.topk, 2)))
        sdWp = storage.sdWp.get_tensor(cute.make_layout((self.num_heads, len(self.epi_warp_ids), 2)))
        sRowQ = storage.sRowQ.get_tensor(cute.make_layout((self.ring_depth,)))

        pipeline.pipeline_init_wait()

        # accumulator reference layouts
        acc_S_ref = mma_S.make_fragment_C(mma_S.partition_shape_C(cute.select(self.tiler_S, mode=[0, 1])))
        acc_dQ_ref = mma_dQ.make_fragment_C(mma_dQ.partition_shape_C(cute.select(self.tiler_dQ, mode=[0, 1])))
        acc_dK_ref = mma_dK.make_fragment_C(mma_dK.partition_shape_C(cute.select(self.tiler_dK, mode=[0, 1])))

        wg_idx = tidx // 128
        # Every CTA and every warpgroup runs exactly ticket_slots ring slots
        # (compile-time trip count); overdrawn slots publish row = -1.
        n_slots = self.ticket_slots

        # =============================================================
        # Warpgroup 0: TMA warp + MMA warp + ticket-writer warp (+1 idle)
        # =============================================================
        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(self.num_regs_wg0)

            if warp_idx == self.load_warp_id:
                thr_mma = mma_S.get_slice(0)
                # partition ONCE against the full row axis; per row only the
                # coordinate is sliced (at the 32-reg budget a per-row
                # tma_partition recompute spills)
                pt_gQ = cute.local_tile(
                    tma_tensor_Q,
                    cute.select(self.tiler_S, mode=[1, 2]),
                    (None, None, None),
                )
                pt_tSgQ = thr_mma.partition_B(pt_gQ)
                pt_tQsQ, pt_tQgQ = cpasync.tma_partition(
                    tma_atom_Q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sQ, 0, 3),
                    cute.group_modes(pt_tSgQ, 0, 3),
                )
                q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.q_stage)
                pt_tst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ring_depth)
                for pt_k in cutlass.range(n_slots, unroll=1):
                    # ticket ring row (row < 0 -> skip slot); warp-elected
                    # arrive (lane 0)
                    pipe_T.consumer_wait(pt_tst)
                    pt_row = Int32(sRowQ[pt_tst.index])
                    if tidx % 32 == 0:
                        pipe_T.consumer_release(pt_tst)
                    pt_tst.advance()
                    if pt_row >= 0:
                        # 2-stage sQ: waits the MMA warp's UMMA-commit
                        # release from row i-2, i.e. the Q fetch runs a full
                        # row ahead of its consumer
                        pipe_Q.producer_acquire(q_state)
                        pt_q_bar = pipe_Q.producer_get_barrier(q_state)
                        cute.copy(
                            tma_atom_Q,
                            pt_tQgQ[None, 0, 0, pt_row],
                            pt_tQsQ[None, q_state.index],
                            tma_bar_ptr=pt_q_bar,
                        )
                        q_state.advance()

            elif warp_idx == self.mma_warp_id:
                tmem.wait_for_alloc()
                tmem_base = tmem.retrieve_ptr(Float32)
                tAccdQ = cute.make_tensor(tmem_base + self.tmem_dQ_off, acc_dQ_ref.layout)

                tSrK = mma_S.make_fragment_A(sK)
                tSrQ = mma_S.make_fragment_B(sQ)
                tdQrKT = mma_dQ.make_fragment_A(sKT)
                tdQrAThi = mma_dQ.make_fragment_B(sAThi)
                tdQrATlo = mma_dQ.make_fragment_B(sATlo)
                tdKrAhi = mma_dK.make_fragment_A(sAhi)
                tdKrAlo = mma_dK.make_fragment_A(sAlo)
                tdKrQT = mma_dK.make_fragment_B(sQT)

                q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.q_stage)
                # dual K states: lead feeds the hoisted MMA1(i+lookahead),
                # trail feeds MMA2(i) and is released afterwards
                k_lead = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
                k_trail = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
                a_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.a_stage)
                s_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.s_stage)
                dk_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.dk_stage)
                dq_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

                pm_tst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ring_depth)
                for pm_k in cutlass.range(n_slots, unroll=1):
                    # ring read for the shared skip/trip bookkeeping (the row
                    # value itself is unused by the MMAs); warp-elected arrive
                    pipe_T.consumer_wait(pm_tst)
                    pm_rowt = Int32(sRowQ[pm_tst.index])
                    if tidx % 32 == 0:
                        pipe_T.consumer_release(pm_tst)
                    pm_tst.advance()
                    if pm_rowt >= 0:
                        pipe_Q.consumer_wait(q_cons)

                        # ---- prologue: fill the S pipe mma_lookahead deep.
                        # The S TMEM slot comes from the pipeline state (S
                        # slots misalign across rows, e.g. 8 tiles over 3
                        # slots at topk 1024).
                        for p in cutlass.range_constexpr(self.mma_lookahead):
                            if const_expr(p < self.num_tiles):
                                pipe_S.producer_acquire(s_prod)
                                pipe_K.consumer_wait(k_lead)
                                pm_psel1 = s_prod.index == 1
                                pm_psel2 = s_prod.index == 2
                                pm_po01 = Int32(self.tmem_S_off[1]) if pm_psel1 else Int32(self.tmem_S_off[0])
                                pm_poff = Int32(self.tmem_S_off[2]) if pm_psel2 else pm_po01
                                pm_ptAccS = cute.make_tensor(tmem_base + pm_poff, acc_S_ref.layout)
                                mma_S.set(tcgen05.Field.ACCUMULATE, False)
                                for kb in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll_full=True):
                                    cute.gemm(
                                        mma_S,
                                        pm_ptAccS,
                                        tSrK[None, None, kb, k_lead.index],
                                        tSrQ[None, None, kb, q_cons.index],
                                        pm_ptAccS,
                                    )
                                    mma_S.set(tcgen05.Field.ACCUMULATE, True)
                                pipe_S.producer_commit(s_prod)
                                s_prod.advance()
                                k_lead.advance()

                        # 1-slot dQ TMEM: wait for the S-epilogue's post-drain
                        # release of the previous row before MMA2 overwrites it
                        pipe_DQ.producer_acquire(dq_prod)

                        # ---- rolled tile loop (runtime i). A fully unrolled
                        # body at the 32-reg budget spills the pipeline states
                        # + MMA descriptor bases to local memory; rolling
                        # shrinks the live working set. dk slot / S slot both
                        # come from the pipeline-state index at runtime.
                        for pm_i in cutlass.range(self.num_tiles, unroll=1):
                            # ---- hoisted MMA1(i+lookahead), predicated ----
                            if pm_i + self.mma_lookahead < self.num_tiles:
                                pipe_S.producer_acquire(s_prod)
                                pipe_K.consumer_wait(k_lead)
                                pm_hsel1 = s_prod.index == 1
                                pm_hsel2 = s_prod.index == 2
                                pm_ho01 = Int32(self.tmem_S_off[1]) if pm_hsel1 else Int32(self.tmem_S_off[0])
                                pm_hoff = Int32(self.tmem_S_off[2]) if pm_hsel2 else pm_ho01
                                pm_htAccS = cute.make_tensor(tmem_base + pm_hoff, acc_S_ref.layout)
                                mma_S.set(tcgen05.Field.ACCUMULATE, False)
                                for kb in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll_full=True):
                                    cute.gemm(
                                        mma_S,
                                        pm_htAccS,
                                        tSrK[None, None, kb, k_lead.index],
                                        tSrQ[None, None, kb, q_cons.index],
                                        pm_htAccS,
                                    )
                                    mma_S.set(tcgen05.Field.ACCUMULATE, True)
                                pipe_S.producer_commit(s_prod)
                                s_prod.advance()
                                k_lead.advance()

                            # ---- MMA2: dQT += Ksel^T @ (A_hi + A_lo) ----
                            pipe_A.consumer_wait(a_cons)
                            mma_dQ.set(tcgen05.Field.ACCUMULATE, pm_i != 0)
                            for kb in cutlass.range(0, cute.size(tdQrAThi, mode=[2]), unroll_full=True):
                                cute.gemm(
                                    mma_dQ,
                                    tAccdQ,
                                    tdQrKT[None, None, kb, k_trail.index],
                                    tdQrAThi[None, None, kb, a_cons.index],
                                    tAccdQ,
                                )
                                mma_dQ.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(
                                    mma_dQ,
                                    tAccdQ,
                                    tdQrKT[None, None, kb, k_trail.index],
                                    tdQrATlo[None, None, kb, a_cons.index],
                                    tAccdQ,
                                )
                            pipe_K.consumer_release(k_trail)
                            k_trail.advance()

                            # ---- MMA3: dKp = (A_hi + A_lo) @ Q^T-view ----
                            if pm_i == self.num_tiles - 1:
                                # dQT is complete after MMA2(last tile):
                                # commit it BEFORE MMA3's DK-slot acquire so
                                # the S-epilogue's dQ drain is not gated on
                                # the dK warpgroup's drain tail (the acquire
                                # waits on the dK release two tiles back).
                                pipe_DQ.producer_commit(dq_prod)
                                dq_prod.advance()
                            pipe_DK.producer_acquire(dk_prod)
                            pm_dksel = dk_prod.index == 1
                            pm_dkoff = Int32(self.tmem_dK_off[1]) if pm_dksel else Int32(self.tmem_dK_off[0])
                            pm_tAccdK = cute.make_tensor(tmem_base + pm_dkoff, acc_dK_ref.layout)
                            mma_dK.set(tcgen05.Field.ACCUMULATE, False)
                            for kb in cutlass.range(0, cute.size(tdKrQT, mode=[2]), unroll_full=True):
                                cute.gemm(
                                    mma_dK,
                                    pm_tAccdK,
                                    tdKrAhi[None, None, kb, a_cons.index],
                                    tdKrQT[None, None, kb, q_cons.index],
                                    pm_tAccdK,
                                )
                                mma_dK.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(
                                    mma_dK,
                                    pm_tAccdK,
                                    tdKrAlo[None, None, kb, a_cons.index],
                                    tdKrQT[None, None, kb, q_cons.index],
                                    pm_tAccdK,
                                )
                            pipe_DK.producer_commit(dk_prod)
                            dk_prod.advance()
                            pipe_A.consumer_release(a_cons)
                            a_cons.advance()

                        # row epilogue: release this row's sQ slot for the
                        # TMA warp (UMMA-commit arrive -> fires once all of
                        # this row's MMAs have actually completed).
                        pipe_Q.consumer_release(q_cons)
                        q_cons.advance()
            else:
                # Ticket writer: idle warp 2, lane 0. Draws global row
                # tickets (a scalar global atomic fetch-add) and publishes
                # them into the sRowQ ring; every CTA runs exactly
                # ticket_slots slots,
                # overdrawn slots publish -1. The drawer of the LAST raw ticket
                # (ticket_slots * num_ctas - 1, unique by atomic total
                # order — all other CTAs have already drawn by then) resets
                # the counter for the next launch. Warp 3 stays idle.
                if warp_idx == self.ticket_warp_id:
                    if tidx % 32 == 0:
                        tw_tst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ring_depth)
                        for tw_k in cutlass.range(self.ticket_slots, unroll=1):
                            pipe_T.producer_acquire(tw_tst)
                            tw_raw = Int32(cute.arch.atomic_add(mCnt.iterator.llvm_ptr, Int32(1)))
                            tw_row = tw_raw if tw_raw < seq_len else Int32(-1)
                            sRowQ[tw_tst.index] = tw_row
                            pipe_T.producer_commit(tw_tst)
                            tw_tst.advance()
                            if tw_raw == Int32(self.ticket_slots * self.num_ctas - 1):
                                mCnt[0] = Int32(0)
        # =============================================================
        # Warpgroup 1: S-epilogue -> A(hi/lo) + dW ; then dQ store
        # =============================================================
        elif wg_idx == 1:
            cute.arch.setmaxregister_increase(self.num_regs_epi)
            if warp_idx == self.epi_warp_ids[0]:
                tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_base = tmem.retrieve_ptr(Float32)
            tAccS0 = cute.make_tensor(tmem_base + self.tmem_S_off[0], acc_S_ref.layout)
            tAccdQ = cute.make_tensor(tmem_base + self.tmem_dQ_off, acc_dQ_ref.layout)

            tidx_wg = tidx % 128
            lane = tidx % 32
            # wide S-epilogue TMEM load: Ld32x32b.x32 keeps the thread ==
            # slot-row mapping of the narrow x8 atom with 4x fewer
            # tcgen05.ld
            tmem_load_atom_S = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                Float32,
            )
            tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom_S, tAccS0)
            thr_t2r = tiled_t2r.get_slice(tidx_wg)

            thr_mma_epi = mma_S.get_slice(tidx_wg)
            cS = cute.make_identity_tensor(cute.select(self.tiler_S, mode=[0, 1]))
            tScS = thr_t2r.partition_D(thr_mma_epi.partition_C(cS))
            # this thread's topk slot within the tile (== TMEM lane)
            pe_row = cute.get(tScS[0], mode=[0])

            tSrS_shape = thr_t2r.partition_D(cute.make_identity_tensor(tAccS0.shape)).shape
            tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

            s_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.s_stage)
            a_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.a_stage)
            dq_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            rW = cute.make_rmem_tensor((self.num_heads,), Float32)
            rWb = cute.make_rmem_tensor((self.num_heads,), BFloat16)
            rdW = cute.make_rmem_tensor((self.num_heads,), Float32)
            # A row buffers, chunked (8, 8) for 16B vector STS
            rAhi = cute.make_rmem_tensor((8, 8), BFloat16)
            rAlo = cute.make_rmem_tensor((8, 8), BFloat16)

            epi_wid = warp_idx - self.epi_warp_ids[0]

            pe_tst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ring_depth)
            pe_vcnt = Int32(0)
            for pe_k in cutlass.range(n_slots, unroll=1):
                # ticket ring row (row < 0 -> skip slot); the sG/sdWp parity
                # buffer alternates per VALID row and must match the gather /
                # dK warpgroups' valid-count parity exactly; warp-elected
                # arrive
                pipe_T.consumer_wait(pe_tst)
                pe_rowi = Int32(sRowQ[pe_tst.index])
                if lane == 0:
                    pipe_T.consumer_release(pe_tst)
                pe_tst.advance()
                pe_ok = pe_rowi >= 0
                pe_buf = pe_vcnt % 2
                pe_vcnt = pe_vcnt + (Int32(1) if pe_ok else Int32(0))
                if pe_ok:
                    # rW straight from gmem (mW is a read-only input; a smem
                    # stage + barrier would put this warpgroup on the restage
                    # critical path). The row is loaded bf16 (16B vectors)
                    # and upcast to fp32 in-register — bf16 -> fp32 is exact,
                    # so this is bit-for-bit the fp32 view of mW without
                    # needing a per-execute upcast kernel and buffer.
                    pe_wptr = cute.make_ptr(
                        BFloat16,
                        mW[pe_rowi, None].iterator.llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    pe_wrow = cute.make_tensor(pe_wptr, cute.make_layout((self.num_heads,)))
                    cute.autovec_copy(pe_wrow, rWb)
                    for h in cutlass.range_constexpr(self.num_heads):
                        rW[h] = Float32(rWb[h])
                        rdW[h] = Float32(0.0)

                    # rolled tile loop: the inner per-head-pair loop stays
                    # static (register fragments cannot be dynamically
                    # indexed); the S TMEM slot offsets are non-uniform so
                    # the accumulator tensor is selected at runtime
                    for pe_i in cutlass.range(self.num_tiles, unroll=1):
                        # S TMEM slot from the pipeline state (rolls across
                        # rows; the local tile number is not slot-aligned)
                        pe_sel1 = s_cons.index == 1
                        pe_sel2 = s_cons.index == 2
                        pe_o01 = Int32(self.tmem_S_off[1]) if pe_sel1 else Int32(self.tmem_S_off[0])
                        pe_off = Int32(self.tmem_S_off[2]) if pe_sel2 else pe_o01
                        pe_tAccS = cute.make_tensor(tmem_base + pe_off, acc_S_ref.layout)
                        pipe_S.consumer_wait(s_cons)
                        pe_tTR = thr_t2r.partition_S(pe_tAccS)
                        cute.copy(tiled_t2r, pe_tTR, tSrS)
                        cute.arch.fence_view_async_tmem_load()
                        pipe_S.consumer_release(s_cons)
                        s_cons.advance()

                        pe_pos = Int32(pe_i * self.n_block) + pe_row
                        # runtime sm_scale fold (g' = sm_scale * g): one FMUL
                        # per (thread, tile); exact identity at the
                        # production sm_scale == 1.0. The relu gate below
                        # reads the unscaled S — equivalent to the default
                        # backend's gate on sm_scale * S for the positive
                        # scales this backend accepts (check_support). The
                        # one gap is underflow to zero: if sm_scale * S
                        # rounds to +0 for a tiny positive S, the default
                        # backend's gate on the scaled score sees 0 and drops
                        # the slot while this gate keeps it. The gate is a
                        # step function, so on such a slot the two backends
                        # can differ by that slot's whole contribution: dW's
                        # term is g' * relu(S) and stays negligible, but the
                        # dQ/dK term is g' * w (times K / Q) and is not
                        # bounded by the score magnitude at all. The default
                        # backend's multiply preserves subnormals (its
                        # packed mul sets no .ftz), so reaching it needs the
                        # exact sm_scale * S below 2^-150 (~7.0e-46).
                        pe_gv = Float32(sG[pe_pos, pe_buf]) * sm_scale

                        pipe_A.producer_acquire(a_prod)
                        # invalid rows were zero-filled by the gather ->
                        # S == 0 -> relu = 0, A = 0: no masking needed here.
                        # Packed hi/lo cvt: 2 heads per cvt.rn.bf16x2.f32.
                        # The relu mask is applied to the fp32 input (a := 0
                        # for gated heads) so bf16(0) = 0 reproduces the
                        # scalar else-branch exactly; hi is computed before
                        # lo and lo uses fp32(hi), keeping the hi/lo
                        # accumulate order of the MMA stage exact.
                        for pe_hp in cutlass.range_constexpr(self.num_heads // 2):
                            pe_h0 = 2 * pe_hp
                            pe_h1 = pe_h0 + 1
                            pe_s0 = Float32(tSrS[pe_h0])
                            pe_s1 = Float32(tSrS[pe_h1])
                            pe_k0 = pe_s0 > Float32(0.0)
                            pe_k1 = pe_s1 > Float32(0.0)
                            pe_a0 = pe_gv * rW[pe_h0] if pe_k0 else Float32(0.0)
                            pe_a1 = pe_gv * rW[pe_h1] if pe_k1 else Float32(0.0)
                            pe_hi = cute.TensorSSA(
                                _cvt_f32x2_bf16x2_rn_pure(pe_a0, pe_a1),
                                (2,),
                                BFloat16,
                            )
                            pe_lo = cute.TensorSSA(
                                _cvt_f32x2_bf16x2_rn_pure(
                                    pe_a0 - Float32(pe_hi[0]),
                                    pe_a1 - Float32(pe_hi[1]),
                                ),
                                (2,),
                                BFloat16,
                            )
                            rdW[pe_h0] = rdW[pe_h0] + (pe_gv * pe_s0 if pe_k0 else Float32(0.0))
                            rdW[pe_h1] = rdW[pe_h1] + (pe_gv * pe_s1 if pe_k1 else Float32(0.0))
                            rAhi[pe_h0 % 8, pe_h0 // 8] = pe_hi[0]
                            rAlo[pe_h0 % 8, pe_h0 // 8] = pe_lo[0]
                            rAhi[pe_h1 % 8, pe_h1 // 8] = pe_hi[1]
                            rAlo[pe_h1 % 8, pe_h1 // 8] = pe_lo[1]

                        pe_sAhi = cute.composition(
                            sAhi[None, None, None, a_prod.index],
                            cute.make_layout((self.n_block, self.num_heads)),
                        )
                        pe_sAlo = cute.composition(
                            sAlo[None, None, None, a_prod.index],
                            cute.make_layout((self.n_block, self.num_heads)),
                        )
                        pe_hic = cute.flat_divide(pe_sAhi[pe_row, None], (8,))
                        pe_loc = cute.flat_divide(pe_sAlo[pe_row, None], (8,))
                        for pe_cc in cutlass.range_constexpr(8):
                            cute.autovec_copy(rAhi[None, pe_cc], pe_hic[None, pe_cc])
                            cute.autovec_copy(rAlo[None, pe_cc], pe_loc[None, pe_cc])
                        cute.arch.fence_view_async_shared()
                        pipe_A.producer_commit(a_prod)
                        a_prod.advance()

                    # ---- dW butterfly + stage partial (parity smem buffer)
                    for h in cutlass.range_constexpr(self.num_heads):
                        pe_v = rdW[h]
                        for off in cutlass.range_constexpr(5):
                            pe_v += cute.arch.shuffle_sync_bfly(pe_v, 1 << off)
                        rdW[h] = pe_v
                    if lane == 0:
                        cute.autovec_copy(rdW, sdWp[None, epi_wid, pe_buf])
                    self.epi_dw_barrier.arrive()

                    # ---- dQ drain; the partition setup is rebuilt per row
                    # (pure address math) so nothing dQ-related stays live
                    # across the whole row loop (this warpgroup sits at the
                    # 256-reg ceiling; hoisting these spills rW/rdW)
                    tiled_t2r_dq = tcgen05.make_tmem_copy(tmem_load_atom_S, tAccdQ)
                    thr_t2r_dq = tiled_t2r_dq.get_slice(tidx_wg)
                    cdq = cute.make_identity_tensor(cute.select(self.tiler_dQ, mode=[0, 1]))
                    tdqcdq = thr_t2r_dq.partition_D(thr_mma_epi.partition_C(cdq))
                    pe_drow = cute.get(tdqcdq[0], mode=[0])
                    tSrdQ_shape = thr_t2r_dq.partition_D(cute.make_identity_tensor(tAccdQ.shape)).shape
                    tSrdQ = cute.make_rmem_tensor(tSrdQ_shape, Float32)
                    tTR_tdQ = thr_t2r_dq.partition_S(tAccdQ)
                    # release the TMEM slot right after the fence so the next
                    # row's MMA2 can start while this warp is still storing
                    # registers to gmem
                    pipe_DQ.consumer_wait(dq_cons)
                    cute.copy(tiled_t2r_dq, tTR_tdQ, tSrdQ)
                    cute.arch.fence_view_async_tmem_load()
                    pipe_DQ.consumer_release(dq_cons)
                    dq_cons.advance()
                    for h in cutlass.range_constexpr(self.num_heads):
                        mdQ[pe_rowi, h, pe_drow] = BFloat16(Float32(tSrdQ[h]))

                    # ---- deterministic dW combine (fixed order, plain
                    # store; this CTA exclusively owns mdW[pe_rowi]). The
                    # store rounds to the output dtype: fp32 buffers get the
                    # accumulator verbatim, bf16 buffers its bf16 rounding
                    # (the same value an fp32 scratch plus a cast would give).
                    self.epi_dw_barrier.arrive_and_wait()
                    if tidx_wg < self.num_heads:
                        pe_dw = Float32(sdWp[tidx_wg, 0, pe_buf])
                        pe_dw += Float32(sdWp[tidx_wg, 1, pe_buf])
                        pe_dw += Float32(sdWp[tidx_wg, 2, pe_buf])
                        pe_dw += Float32(sdWp[tidx_wg, 3, pe_buf])
                        if const_expr(self.dw_out_bf16):
                            mdW[pe_rowi, tidx_wg] = BFloat16(pe_dw)
                        else:
                            mdW[pe_rowi, tidx_wg] = pe_dw

            self.tmem_dealloc_barrier.arrive_and_wait()
            if warp_idx == self.epi_warp_ids[0]:
                cute.arch.dealloc_tmem(tmem_base, self.tmem_alloc_cols)

        # =============================================================
        # Warpgroup 2: metadata restage + sparse K gather (cp.async)
        # =============================================================
        elif wg_idx == 2:
            cute.arch.setmaxregister_decrease(self.num_regs_gather)
            wg_tidx = tidx % 128
            idx_in_group = wg_tidx % 8
            group_idx = wg_tidx // 8  # 16 groups x 8 threads
            NUM_GROUPS = 16
            ROWS_PER_GROUP = self.n_block // NUM_GROUPS

            gather_atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                BFloat16,
                num_bits_per_copy=128,
            )
            gather_tv = cute.make_tiled_copy_tv(gather_atom, cute.make_layout((1,)), cute.make_layout((8,)))
            thr_gather = gather_tv.get_slice(0)

            rZero = cute.make_rmem_tensor((8,), BFloat16)
            for z in cutlass.range_constexpr(8):
                rZero[z] = BFloat16(0.0)

            k_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
            if const_expr(self.meta_war_explicit):
                # sIdx/sG slot WAR pipe (advances once per VALID row)
                pg_mst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.meta_stage)

            pg_tst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ring_depth)
            pg_vcnt = Int32(0)
            for pg_k in cutlass.range(n_slots, unroll=1):
                # ticket ring row (row < 0 -> skip slot; the sIdx/sG parity
                # buffer alternates per VALID row so the restage-vs-dK-drain
                # protection chain keeps its 2-valid-row distance);
                # warp-elected arrive
                pipe_T.consumer_wait(pg_tst)
                pg_rowi = Int32(sRowQ[pg_tst.index])
                if wg_tidx % 32 == 0:
                    pipe_T.consumer_release(pg_tst)
                pg_tst.advance()
                pg_ok = pg_rowi >= 0
                pg_buf = pg_vcnt % 2
                pg_vcnt = pg_vcnt + (Int32(1) if pg_ok else Int32(0))
                if pg_ok:
                    # local-id mode: per-row batch offset, applied AFTER the
                    # validity mask against the per-batch bound (below), so
                    # positive out-of-range local ids contribute nothing
                    # instead of aliasing the next batch's K rows.
                    pg_off = Int32(0)
                    if const_expr(self.idx_local):
                        pg_off = (pg_rowi // s_q_per_batch) * s_k_local
                    # ---- explicit slot WAR acquire (pipe_M): wait until the
                    # dK warpgroup has finished the LAST sIdx read of the row
                    # that was staged in this parity slot 2 valid rows ago.
                    # Warp-elected: lane 0 waits, sync_warp holds the warp's
                    # restage stores behind it. This is the fix that stops
                    # the restage from lapping the dK drain at <= 7
                    # tiles/row.
                    if const_expr(self.meta_war_explicit):
                        if wg_tidx % 32 == 0:
                            pipe_M.producer_acquire(pg_mst)
                        cute.arch.sync_warp()
                        pg_mst.advance()
                    # ---- metadata restage (this warpgroup only, 16B
                    # vectorized). The S-epilogue warpgroup sees the writes
                    # through the K->S pipe acquire chain, the dK warpgroup
                    # through the A->DK chain; neither joins this barrier, so
                    # it never stalls them directly (those pipe chains are what
                    # order the next row's front behind the restage).
                    pg_gIptr = cute.make_ptr(
                        Int32,
                        mIdx[pg_rowi, None].iterator.llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    pg_gIrow = cute.make_tensor(pg_gIptr, cute.make_layout((self.topk,)))
                    pg_gIch = cute.flat_divide(pg_gIrow, (4,))
                    pg_gGptr = cute.make_ptr(
                        Float32,
                        mG[pg_rowi, None].iterator.llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    pg_gGrow = cute.make_tensor(pg_gGptr, cute.make_layout((self.topk,)))
                    pg_gGch = cute.flat_divide(pg_gGrow, (4,))
                    pg_sIch = cute.flat_divide(sIdx[None, pg_buf], (4,))
                    pg_sGch = cute.flat_divide(sG[None, pg_buf], (4,))
                    for pg_c in cutlass.range_constexpr(self.topk // (4 * 128)):
                        pg_e4 = wg_tidx + pg_c * 128
                        cute.autovec_copy(pg_gIch[None, pg_e4], pg_sIch[None, pg_e4])
                        cute.autovec_copy(pg_gGch[None, pg_e4], pg_sGch[None, pg_e4])
                    # predicated tail: covers topk % 512 != 0 (e.g. 640 /
                    # 896); constexpr-eliminated whenever topk % 512 == 0
                    # (512 / 1024 / 1536 / 2048)
                    if const_expr((self.topk // 4) % 128 != 0):
                        pg_e4t = wg_tidx + (self.topk // (4 * 128)) * 128
                        if pg_e4t < self.topk // 4:
                            cute.autovec_copy(pg_gIch[None, pg_e4t], pg_sIch[None, pg_e4t])
                            cute.autovec_copy(pg_gGch[None, pg_e4t], pg_sGch[None, pg_e4t])
                    self.stage_barrier.arrive_and_wait()

                    for i in cutlass.range_constexpr(self.num_tiles):
                        pipe_K.producer_acquire(k_prod)
                        pg_sK = cute.composition(
                            sK[None, None, None, k_prod.index],
                            cute.make_layout((self.n_block, self.head_dim)),
                        )
                        for r in cutlass.range_constexpr(ROWS_PER_GROUP):
                            pg_rrow = r * NUM_GROUPS + group_idx
                            pg_pos = Int32(i * self.n_block) + pg_rrow
                            pg_raw = Int32(sIdx[pg_pos, pg_buf])
                            if const_expr(self.idx_local):
                                # mask against the per-batch bound BEFORE the
                                # batch offset (invalid ids never alias)
                                pg_valid = pg_raw >= 0 and pg_raw < s_k_local
                                pg_kv = pg_raw + pg_off
                            else:
                                pg_valid = pg_raw >= 0 and pg_raw < seq_k
                                pg_kv = pg_raw
                            pg_sKrow = pg_sK[pg_rrow, None]
                            pg_sKch = cute.flat_divide(pg_sKrow, (8,))
                            if pg_valid:
                                pg_gKptr = cute.make_ptr(
                                    BFloat16,
                                    mK[pg_kv, None].iterator.llvm_ptr,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                )
                                pg_gKrow = cute.make_tensor(pg_gKptr, cute.make_layout((self.head_dim,)))
                                pg_gKch = cute.flat_divide(pg_gKrow, (8,))
                                for t in cutlass.range_constexpr(self.head_dim // 64):
                                    pg_ch = t * 8 + idx_in_group
                                    pg_tSg = thr_gather.partition_S(pg_gKch[None, pg_ch])
                                    pg_tSs = thr_gather.partition_D(pg_sKch[None, pg_ch])
                                    cute.copy(gather_atom, pg_tSg, pg_tSs)
                            else:
                                # zero-fill: garbage must not reach S/dW/dQT
                                for t in cutlass.range_constexpr(self.head_dim // 64):
                                    pg_ch = t * 8 + idx_in_group
                                    cute.autovec_copy(rZero, pg_sKch[None, pg_ch])
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                        pipe_K.producer_commit(k_prod)
                        k_prod.advance()

        # =============================================================
        # Warpgroup 3: dK reduce — TMEM -> RF -> atom.global.add.v4.f32
        # =============================================================
        else:
            cute.arch.setmaxregister_increase(self.num_regs_dk)
            tmem.wait_for_alloc()
            tmem_base = tmem.retrieve_ptr(Float32)
            tAccdK = [cute.make_tensor(tmem_base + self.tmem_dK_off[s], acc_dK_ref.layout) for s in range(self.dk_stage)]

            tidx_wg = tidx % 128
            # wide dK TMEM drain: Ld16x256b.x16 — 16 256-bit-datapath loads
            # replace 128 narrow loads per 8 tiles of a row (a whole row at
            # topk 1024; both counts scale with the tile count). The wide
            # atom repartitions four logical rows across adjacent threads,
            # so the drain below uses xor-1 register exchanges to
            # reconstruct the exact contiguous 4-element atomic chunks.
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(16)),
                Float32,
            )
            tiled_t2r_dk = tcgen05.make_tmem_copy(tmem_load_atom, tAccdK[0])
            thr_t2r_dk = tiled_t2r_dk.get_slice(tidx_wg)

            thr_mma_dk = mma_dK.get_slice(tidx_wg)
            cdk = cute.make_identity_tensor(cute.select(self.tiler_dK, mode=[0, 1]))
            tdkcdk = thr_t2r_dk.partition_D(thr_mma_dk.partition_C(cdk))

            tSrdK_shape = thr_t2r_dk.partition_D(cute.make_identity_tensor(tAccdK[0].shape)).shape
            tSrdK = cute.make_rmem_tensor(tSrdK_shape, Float32)
            frag4 = cute.make_rmem_tensor((4,), Float32)

            dk_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.dk_stage)
            pk_lane_odd = tidx_wg % 2 != 0
            if const_expr(self.meta_war_explicit):
                # sIdx/sG slot WAR pipe (advances once per VALID row)
                pk_mst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.meta_stage)

            pk_tst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ring_depth)
            pk_vcnt = Int32(0)
            for pk_k in cutlass.range(n_slots, unroll=1):
                # identical ticket sequence in every warpgroup; the row value
                # feeds only the local-id batch offset (sIdx carries the
                # scatter targets) plus skip/parity bookkeeping; warp-elected
                # arrive
                pipe_T.consumer_wait(pk_tst)
                pk_ticket_row = Int32(sRowQ[pk_tst.index])
                if tidx_wg % 32 == 0:
                    pipe_T.consumer_release(pk_tst)
                pk_tst.advance()
                pk_ok = pk_ticket_row >= 0
                pk_buf = pk_vcnt % 2
                pk_vcnt = pk_vcnt + (Int32(1) if pk_ok else Int32(0))
                if pk_ok:
                    # local-id mode: this row's batch offset (the ticket row
                    # value is load-bearing here), applied after the validity
                    # mask against the per-batch bound at each sIdx read.
                    pk_off = Int32(0)
                    if const_expr(self.idx_local):
                        pk_off = (pk_ticket_row // s_q_per_batch) * s_k_local
                    # Two-chain software-pipelined drain: prepare the
                    # select+shuffle results of the second column block
                    # before issuing the first block's reductions, exposing
                    # two independent chains to the scheduler while keeping
                    # the reduction order and grouping exact. The outer tile
                    # loop is rolled by dk_stage (the wide TMEM atom needs a
                    # statically aligned column offset, so the inner slot
                    # loop stays constexpr over the 2 aligned accumulators);
                    # dk slots realign every row when the tile count is
                    # even (e.g. 8 tiles over 2 slots at topk 1024).
                    # (pipe_M's full direction is unused: this warpgroup's
                    # VISIBILITY of the freshly staged sIdx rides the
                    # K->S->A->DK acquire chain as before; only the WAR
                    # arrive below is load-bearing.)
                    if const_expr(self.num_tiles % self.dk_stage == 0):
                        for pk_i2 in cutlass.range(self.num_tiles // self.dk_stage, unroll=1):
                            for pk_slot in cutlass.range_constexpr(self.dk_stage):
                                pk_i = pk_i2 * self.dk_stage + pk_slot
                                pipe_DK.consumer_wait(dk_cons)
                                pk_tTR = thr_t2r_dk.partition_S(tAccdK[pk_slot])
                                cute.copy(tiled_t2r_dk, pk_tTR, tSrdK)
                                cute.arch.fence_view_async_tmem_load()
                                pipe_DK.consumer_release(dk_cons)
                                dk_cons.advance()

                                for pk_rp in cutlass.range_constexpr(2):
                                    pk_re = cute.get(tdkcdk[pk_rp * 64], mode=[0])
                                    pk_ro = cute.get(tdkcdk[pk_rp * 64 + 2], mode=[0])
                                    pk_row = pk_ro if pk_lane_odd else pk_re
                                    pk_pos = Int32(pk_i * self.n_block) + pk_row
                                    pk_raw = Int32(sIdx[pk_pos, pk_buf])
                                    if const_expr(self.idx_local):
                                        pk_valid = pk_raw >= 0 and pk_raw < s_k_local
                                        pk_glob = pk_raw + pk_off
                                    else:
                                        pk_valid = pk_raw >= 0 and pk_raw < seq_k
                                        pk_glob = pk_raw
                                    pk_safe = pk_glob if pk_valid else Int32(0)
                                    pk_ptr = cute.make_ptr(
                                        Float32,
                                        mdK[pk_safe, None].iterator.llvm_ptr,
                                        cute.AddressSpace.gmem,
                                        assumed_align=16,
                                    )
                                    pk_rowt = cute.make_tensor(pk_ptr, cute.make_layout((self.head_dim,)))
                                    pk_ch = cute.flat_divide(pk_rowt, (4,))

                                    for pk_cb2 in cutlass.range_constexpr(8):
                                        pk_e0 = pk_rp * 64 + pk_cb2 * 8
                                        pk_e1 = pk_e0 + 4

                                        # chain A: pre-select the OPPOSITE value
                                        # to send (a single xor-1 pair delivers
                                        # exactly the two partner values each
                                        # lane needs -> 2 SHFL per chunk)
                                        pk_aev0 = Float32(tSrdK[pk_e0])
                                        pk_aev1 = Float32(tSrdK[pk_e0 + 1])
                                        pk_aov0 = Float32(tSrdK[pk_e0 + 2])
                                        pk_aov1 = Float32(tSrdK[pk_e0 + 3])
                                        pk_av0 = pk_aov0 if pk_lane_odd else pk_aev0
                                        pk_av1 = pk_aov1 if pk_lane_odd else pk_aev1
                                        pk_as0 = pk_aev0 if pk_lane_odd else pk_aov0
                                        pk_as1 = pk_aev1 if pk_lane_odd else pk_aov1
                                        pk_ap0 = cute.arch.shuffle_sync_bfly(pk_as0, 1)
                                        pk_ap1 = cute.arch.shuffle_sync_bfly(pk_as1, 1)

                                        # chain B: independent select+shuffle
                                        pk_bev0 = Float32(tSrdK[pk_e1])
                                        pk_bev1 = Float32(tSrdK[pk_e1 + 1])
                                        pk_bov0 = Float32(tSrdK[pk_e1 + 2])
                                        pk_bov1 = Float32(tSrdK[pk_e1 + 3])
                                        pk_bv0 = pk_bov0 if pk_lane_odd else pk_bev0
                                        pk_bv1 = pk_bov1 if pk_lane_odd else pk_bev1
                                        pk_bs0 = pk_bev0 if pk_lane_odd else pk_bov0
                                        pk_bs1 = pk_bev1 if pk_lane_odd else pk_bov1
                                        pk_bp0 = cute.arch.shuffle_sync_bfly(pk_bs0, 1)
                                        pk_bp1 = cute.arch.shuffle_sync_bfly(pk_bs1, 1)

                                        pk_aec = cute.get(tdkcdk[pk_e0], mode=[1])
                                        pk_aoc = cute.get(tdkcdk[pk_e0 + 2], mode=[1])
                                        pk_aown = pk_aoc if pk_lane_odd else pk_aec
                                        pk_acol = pk_aown - (Int32(2) if pk_lane_odd else Int32(0))
                                        pk_bec = cute.get(tdkcdk[pk_e1], mode=[1])
                                        pk_boc = cute.get(tdkcdk[pk_e1 + 2], mode=[1])
                                        pk_bown = pk_boc if pk_lane_odd else pk_bec
                                        pk_bcol = pk_bown - (Int32(2) if pk_lane_odd else Int32(0))

                                        if pk_valid:
                                            frag4[0] = pk_ap0 if pk_lane_odd else pk_av0
                                            frag4[1] = pk_ap1 if pk_lane_odd else pk_av1
                                            frag4[2] = pk_av0 if pk_lane_odd else pk_ap0
                                            frag4[3] = pk_av1 if pk_lane_odd else pk_ap1
                                            cute.arch.atomic_add(
                                                pk_ch[None, pk_acol // 4].iterator.llvm_ptr,
                                                frag4.load(),
                                            )
                                            frag4[0] = pk_bp0 if pk_lane_odd else pk_bv0
                                            frag4[1] = pk_bp1 if pk_lane_odd else pk_bv1
                                            frag4[2] = pk_bv0 if pk_lane_odd else pk_bp0
                                            frag4[3] = pk_bv1 if pk_lane_odd else pk_bp1
                                            cute.arch.atomic_add(
                                                pk_ch[None, pk_bcol // 4].iterator.llvm_ptr,
                                                frag4.load(),
                                            )
                    else:
                        # odd tile count (topk in {384, 640, 896, ...}): the dk TMEM
                        # slot no longer realigns at row start (dk_prod / dk_cons
                        # roll continuously across rows), so the slot is picked at
                        # runtime. A runtime column offset cannot prove the 2-col
                        # TMEM alignment the wide Ld16x256b atom requires, so this
                        # BRANCHES between the two statically-aligned accumulators;
                        # dk_cons.index is warp-uniform, so the branch is converged.
                        for pk_i in cutlass.range(self.num_tiles, unroll=1):
                            pipe_DK.consumer_wait(dk_cons)
                            if dk_cons.index == 1:
                                pk_tTR1 = thr_t2r_dk.partition_S(tAccdK[1])
                                cute.copy(tiled_t2r_dk, pk_tTR1, tSrdK)
                            else:
                                pk_tTR0 = thr_t2r_dk.partition_S(tAccdK[0])
                                cute.copy(tiled_t2r_dk, pk_tTR0, tSrdK)
                            cute.arch.fence_view_async_tmem_load()
                            pipe_DK.consumer_release(dk_cons)
                            dk_cons.advance()

                            for pk_rp in cutlass.range_constexpr(2):
                                pk_re = cute.get(tdkcdk[pk_rp * 64], mode=[0])
                                pk_ro = cute.get(tdkcdk[pk_rp * 64 + 2], mode=[0])
                                pk_row = pk_ro if pk_lane_odd else pk_re
                                pk_pos = Int32(pk_i * self.n_block) + pk_row
                                pk_raw = Int32(sIdx[pk_pos, pk_buf])
                                if const_expr(self.idx_local):
                                    pk_valid = pk_raw >= 0 and pk_raw < s_k_local
                                    pk_glob = pk_raw + pk_off
                                else:
                                    pk_valid = pk_raw >= 0 and pk_raw < seq_k
                                    pk_glob = pk_raw
                                pk_safe = pk_glob if pk_valid else Int32(0)
                                pk_ptr = cute.make_ptr(
                                    Float32,
                                    mdK[pk_safe, None].iterator.llvm_ptr,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                )
                                pk_rowt = cute.make_tensor(pk_ptr, cute.make_layout((self.head_dim,)))
                                pk_ch = cute.flat_divide(pk_rowt, (4,))

                                for pk_cb2 in cutlass.range_constexpr(8):
                                    pk_e0 = pk_rp * 64 + pk_cb2 * 8
                                    pk_e1 = pk_e0 + 4

                                    # chain A: pre-select the OPPOSITE value
                                    # to send (a single xor-1 pair delivers
                                    # exactly the two partner values each
                                    # lane needs -> 2 SHFL per chunk)
                                    pk_aev0 = Float32(tSrdK[pk_e0])
                                    pk_aev1 = Float32(tSrdK[pk_e0 + 1])
                                    pk_aov0 = Float32(tSrdK[pk_e0 + 2])
                                    pk_aov1 = Float32(tSrdK[pk_e0 + 3])
                                    pk_av0 = pk_aov0 if pk_lane_odd else pk_aev0
                                    pk_av1 = pk_aov1 if pk_lane_odd else pk_aev1
                                    pk_as0 = pk_aev0 if pk_lane_odd else pk_aov0
                                    pk_as1 = pk_aev1 if pk_lane_odd else pk_aov1
                                    pk_ap0 = cute.arch.shuffle_sync_bfly(pk_as0, 1)
                                    pk_ap1 = cute.arch.shuffle_sync_bfly(pk_as1, 1)

                                    # chain B: independent select+shuffle
                                    pk_bev0 = Float32(tSrdK[pk_e1])
                                    pk_bev1 = Float32(tSrdK[pk_e1 + 1])
                                    pk_bov0 = Float32(tSrdK[pk_e1 + 2])
                                    pk_bov1 = Float32(tSrdK[pk_e1 + 3])
                                    pk_bv0 = pk_bov0 if pk_lane_odd else pk_bev0
                                    pk_bv1 = pk_bov1 if pk_lane_odd else pk_bev1
                                    pk_bs0 = pk_bev0 if pk_lane_odd else pk_bov0
                                    pk_bs1 = pk_bev1 if pk_lane_odd else pk_bov1
                                    pk_bp0 = cute.arch.shuffle_sync_bfly(pk_bs0, 1)
                                    pk_bp1 = cute.arch.shuffle_sync_bfly(pk_bs1, 1)

                                    pk_aec = cute.get(tdkcdk[pk_e0], mode=[1])
                                    pk_aoc = cute.get(tdkcdk[pk_e0 + 2], mode=[1])
                                    pk_aown = pk_aoc if pk_lane_odd else pk_aec
                                    pk_acol = pk_aown - (Int32(2) if pk_lane_odd else Int32(0))
                                    pk_bec = cute.get(tdkcdk[pk_e1], mode=[1])
                                    pk_boc = cute.get(tdkcdk[pk_e1 + 2], mode=[1])
                                    pk_bown = pk_boc if pk_lane_odd else pk_bec
                                    pk_bcol = pk_bown - (Int32(2) if pk_lane_odd else Int32(0))

                                    if pk_valid:
                                        frag4[0] = pk_ap0 if pk_lane_odd else pk_av0
                                        frag4[1] = pk_ap1 if pk_lane_odd else pk_av1
                                        frag4[2] = pk_av0 if pk_lane_odd else pk_ap0
                                        frag4[3] = pk_av1 if pk_lane_odd else pk_ap1
                                        cute.arch.atomic_add(
                                            pk_ch[None, pk_acol // 4].iterator.llvm_ptr,
                                            frag4.load(),
                                        )
                                        frag4[0] = pk_bp0 if pk_lane_odd else pk_bv0
                                        frag4[1] = pk_bp1 if pk_lane_odd else pk_bv1
                                        frag4[2] = pk_bv0 if pk_lane_odd else pk_bp0
                                        frag4[3] = pk_bv1 if pk_lane_odd else pk_bp1
                                        cute.arch.atomic_add(
                                            pk_ch[None, pk_bcol // 4].iterator.llvm_ptr,
                                            frag4.load(),
                                        )

                    # arrive AFTER the LAST sIdx read of this row -> completes
                    # the empty phase that the gather warpgroup's restage of
                    # this parity slot (2 valid rows later) waits on. THE
                    # explicit timing invariant. Warp-elected: sync_warp
                    # orders all 32 lanes' reads before lane 0's arrive
                    # (4 arrivals/slot, see pipe_M create).
                    if const_expr(self.meta_war_explicit):
                        cute.arch.sync_warp()
                        if tidx_wg % 32 == 0:
                            pipe_M.consumer_release(pk_mst)
                        pk_mst.advance()
            self.tmem_dealloc_barrier.arrive_and_wait()


# =============================================================================
# Factory
# =============================================================================
# compile_key -> compiled kernel. The key holds only params that change the
# generated code: topk (tile trip counts), the persistent grid size, the
# ticket-slot trip count, the ring depth, the dW output dtype, and the
# local-vs-global id variant. Row count / seq_k / sm_scale are dynamic
# runtime arguments.
_compile_cache: dict = {}


def indexer_backward_v2_sm100(
    batch,
    seqlen,
    seqlen_k,
    heads,
    dim,
    topk,
    sm_scale=1.0,
    block_I=128,
    topk_indices_global: bool = True,
    dw_out_dtype: torch.dtype = torch.float32,
):
    """SM100 indexer backward v2 factory.

    Mirrors the ``indexer_backward_sm100`` factory API and the returned
    callable's signature so ``api.py`` can dispatch on ``backend="sm100_v2"``
    without changing the execute path:

    * Kernel 1 (score-grad precompute) is shared with the default backend —
      both paths consume bit-identical grad signals.
    * Kernel 2 is the persistent v2 GEMM kernel above. It
      consumes a (rows, topk) flattened view of the inputs; weights are
      upcast bf16 -> fp32 in-register (exact), ``sm_scale`` is folded into
      the grad-signal read as a runtime argument (``attn_score`` is left
      holding exactly kernel 1's ``grad_signal``, like the default
      backend), local per-batch top-k ids are masked against the per-batch
      ``S_k`` and offset in-kernel when ``topk_indices_global`` is False,
      and ``d_weights`` is stored directly in the output dtype.
    * A trailing fp32 -> bf16 cast runs only for bf16 ``d_index_k`` buffers
      (the atomic accumulator is fp32); fp32 ``d_index_k`` buffers are
      written directly (zeroed here, no caller pre-zero needed).

    The returned callable allocates its ticket counter on the first execute
    and reuses it on every later one; a bf16 ``d_index_k`` additionally takes
    a ``B * S_k * D`` fp32 accumulator from the caching allocator per call (it
    has to be re-zeroed each time regardless). The per-plan ticket-counter
    workspace is device-resident: the plan binds the device of its first
    execute and raises on any indexer tensor from another device
    (``grad_loss`` is validated by ``api.py``'s wrapper). Executions
    of one plan must not overlap on the device (see the module docstring's
    concurrency contract); ``api.py``'s wrapper keys its plan cache on the
    CUDA device and on the resolved stream to keep the wrapper contract
    device- and stream-safe.

    ``sm_scale`` must be positive (the in-kernel relu gate reads the
    unscaled scores; positive scales keep that equivalent to the default
    backend's gate on the scaled scores, except in the underflow corner noted
    at the gate itself). ``batch``/``seqlen``/``seqlen_k``
    fix the plan's shape; execute re-validates the real tensors against it.
    """
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("indexer_backward_v2_sm100 requires SM100; use backend='default' elsewhere")
    if heads != 64 or dim != 128:
        raise ValueError(f"indexer backward v2 is specialized for H=64, D=128 (got H={heads}, D={dim})")
    if block_I != 128:
        raise ValueError(f"indexer backward v2 requires block_I=128 (got block_I={block_I})")
    if topk % 128 != 0 or not (128 <= topk <= 2048):
        raise ValueError(
            f"indexer backward v2 supports topk % 128 == 0 with 128 <= topk <= 2048 — 128-slot tiles, >=1 tile/row (the S/K pipes clamp to the tile count below 3 tiles/row and the dK-side metadata hazard is explicitly barriered), <=2048 smem cap (got topk={topk})"
        )
    if not float(sm_scale) > 0.0:
        raise ValueError(f"indexer backward v2 requires sm_scale > 0 (got sm_scale={sm_scale}); the relu gate reads unscaled scores")
    if dw_out_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"indexer backward v2 supports d_weights dtype float32 or bfloat16 (got {dw_out_dtype})")

    rows = batch * seqlen
    seq_k_total = batch * seqlen_k
    idx_local = (not topk_indices_global) and batch > 1
    # Per-plan workspace (module docstring: concurrency contract): the
    # dynamic-ticket counter, created once on the plan's first execute. The
    # fp32 dK accumulator a bf16 ``d_index_k`` needs is deliberately not in
    # here; it comes from the caching allocator on every call, the same way
    # the sm90 / sm100 backends take theirs.
    plan_ws: dict = {}

    def _check(cond, msg):
        if not cond:
            raise ValueError(f"indexer_backward_v2_sm100 execute: {msg}")

    def _run(
        IndexQ,
        Weights,
        IndexK,
        dIndexQ,
        dWeights,
        dIndexK,
        AttnScore,
        IndexScore,
        TopkIndices,
        GradLoss,
        grad_scale,
        current_stream=None,
    ):
        # ---- execute-time contract validation: views only, no implicit
        # conversion, no allocation (python/cudnn/AGENTS.md Rule 1).
        # api.check_support() already validated the descriptor metadata;
        # this re-checks the real tensors against the compiled plan.
        b, s_q, h, d = IndexQ.shape
        s_k = IndexK.shape[1]
        _check(
            (b, s_q, s_k, h, d) == (batch, seqlen, seqlen_k, heads, dim),
            f"tensor shape {(b, s_q, s_k, h, d)} does not match the compiled plan {(batch, seqlen, seqlen_k, heads, dim)}",
        )
        # index_k is consumed as a dense flat (b * s_k, d) view, so its dim 0
        # and dim 2 must be exactly index_q's b and d. Only dim 1 reaches the
        # wrapper's plan-cache key, so a cache hit can deliver an index_k whose
        # dim 0 / dim 2 disagree without api.check_support() re-running.
        _check(
            tuple(IndexK.shape) == (b, s_k, d),
            f"index_k shape {tuple(IndexK.shape)} is not the dense {(b, s_k, d)} layout implied by index_q (index_k is read as a flat (b * s_k, d) view)",
        )
        _check(TopkIndices.shape == (b, s_q, topk), f"topk_indices shape {tuple(TopkIndices.shape)} != {(b, s_q, topk)}")
        _check(Weights.shape == (b, s_q, h), f"weights shape {tuple(Weights.shape)} != {(b, s_q, h)}")
        _check(AttnScore.shape == (b, s_q, topk) and IndexScore.shape == (b, s_q, topk), "attn_score/index_score shape mismatch")
        _check(dIndexQ.shape == IndexQ.shape and dWeights.shape == Weights.shape and dIndexK.shape == IndexK.shape, "gradient output shape mismatch")
        _check(dIndexQ.dtype == torch.bfloat16, f"d_index_q must be bfloat16 (got {dIndexQ.dtype})")
        _check(dWeights.dtype == dw_out_dtype, f"d_weights dtype {dWeights.dtype} does not match the compiled plan ({dw_out_dtype})")
        _check(dIndexK.dtype in (torch.float32, torch.bfloat16), f"d_index_k must be float32 or bfloat16 (got {dIndexK.dtype})")
        _check(
            IndexQ.dtype == torch.bfloat16 and Weights.dtype == torch.bfloat16 and IndexK.dtype == torch.bfloat16,
            f"index_q/weights/index_k must be bfloat16 (got {IndexQ.dtype}/{Weights.dtype}/{IndexK.dtype})",
        )
        _check(TopkIndices.dtype == torch.int32, f"topk_indices must be int32 (got {TopkIndices.dtype})")
        for name, t in (
            ("index_q", IndexQ),
            ("weights", Weights),
            ("index_k", IndexK),
            ("d_index_q", dIndexQ),
            ("d_weights", dWeights),
            ("d_index_k", dIndexK),
            ("attn_score", AttnScore),
            ("index_score", IndexScore),
            ("topk_indices", TopkIndices),
        ):
            _check(t.is_cuda and t.device == IndexQ.device, f"{name} must be on {IndexQ.device}")
            _check(t.is_contiguous(), f"{name} must be contiguous")

        # One plan serves one device: the per-plan ticket counter is
        # device-resident, so bind the plan to the device of its first
        # execute and reject any other device BEFORE kernel 1 mutates the
        # score buffers. api.py keys its plan cache on the device; this
        # check keeps direct users of the factory safe independently of
        # that cache.
        plan_device = plan_ws.get("device")
        if plan_device is None:
            plan_ws["device"] = plan_device = IndexQ.device
        _check(
            IndexQ.device == plan_device,
            f"tensors are on {IndexQ.device} but this plan's workspace is bound to {plan_device}; one plan serves one device — build one plan per device",
        )

        # Kernel 1: shared in-place score-grad precompute.
        #   AttnScore  <- grad_signal, IndexScore <- sum_grad
        _score_grad_inplace(AttnScore, IndexScore, GradLoss, grad_scale, block_I=block_I, current_stream=current_stream)

        # true views (validated contiguous above — cannot copy)
        q_flat = IndexQ.view(rows, h, d)
        k_flat = IndexK.view(seq_k_total, d)
        w_flat = Weights.view(rows, h)
        g_flat = AttnScore.view(rows, topk)
        idx_flat = TopkIndices.view(rows, topk)
        dq_flat = dIndexQ.view(rows, h, d)
        dw_flat = dWeights.view(rows, h)

        with _torch_stream_context(current_stream):
            if dIndexK.dtype == torch.float32:
                # write the caller's f32 buffer directly (atomic target;
                # zeroed here — no caller pre-zero contract)
                dk_f32_flat = dIndexK.view(seq_k_total, d)
            else:
                # fp32 accumulator for a bf16 ``d_index_k``, taken from the
                # caching allocator per call instead of being pinned to the
                # plan: it is B * S_k * D * 4 B and has to be re-zeroed on
                # every call either way, so caching it inside the plan would
                # save no work while keeping it alive (and unreclaimable by
                # ``empty_cache()``) for as long as the plan is cached. The
                # allocator hands the same block back for the same size on the
                # same stream, so steady state is a pool hit, not a cudaMalloc.
                dk_f32_flat = torch.empty((seq_k_total, d), dtype=torch.float32, device=IndexK.device)
            dk_f32_flat.zero_()
            cnt = plan_ws.get("counter")
            if cnt is None:
                # one-time per-plan dynamic-ticket counter (self-resetting;
                # see the module docstring's concurrency contract)
                cnt = torch.zeros(2, dtype=torch.int32, device=IndexQ.device)
                plan_ws["counter"] = cnt

        num_ctas = torch.cuda.get_device_properties(IndexQ.device).multi_processor_count
        # per-CTA ring-slot trip count; the +8 surplus slots are the
        # work-stealing headroom (they come back as -1 tickets)
        ticket_slots = (rows + num_ctas - 1) // num_ctas + 8
        ring_depth = 4

        s = _resolve_stream(current_stream)
        compile_key = (heads, dim, topk, num_ctas, ticket_slots, ring_depth, dw_out_dtype, idx_local)
        fn = _compile_cache.get(compile_key)
        if fn is None:
            kernel_obj = IndexerBackwardV2Sm100(
                num_heads=heads,
                head_dim=dim,
                topk=topk,
                num_ctas=num_ctas,
                ticket_slots=ticket_slots,
                ring_depth=ring_depth,
                dw_out_bf16=dw_out_dtype == torch.bfloat16,
                idx_local=idx_local,
            )
            fn = cute.compile(
                kernel_obj,
                to_cute_tensor(q_flat, divisibility=dim),
                to_cute_tensor(k_flat, divisibility=dim),
                to_cute_tensor(w_flat),
                to_cute_tensor(idx_flat),
                to_cute_tensor(g_flat),
                to_cute_tensor(dq_flat, divisibility=dim),
                to_cute_tensor(dw_flat),
                to_cute_tensor(dk_f32_flat, divisibility=dim),
                cutlass.Float32(float(sm_scale)),
                cutlass.Int32(s_q),
                cutlass.Int32(s_k),
                s,
                to_cute_tensor(cnt),
                options=compile_options(),
            )
            _compile_cache[compile_key] = fn

        # Kernel 2: persistent v2 gather-GEMM backward.
        with torch.cuda.nvtx.range("indexer_backward_v2_gemm"):
            fn(
                q_flat,
                k_flat,
                w_flat,
                idx_flat,
                g_flat,
                dq_flat,
                dw_flat,
                dk_f32_flat,
                cutlass.Float32(float(sm_scale)),
                cutlass.Int32(s_q),
                cutlass.Int32(s_k),
                s,
                cnt,
            )

        # Trailing cast only for bf16 d_index_k buffers (fp32 atomic
        # accumulator -> output dtype); fp32 buffers were written directly.
        if dIndexK.dtype != torch.float32:
            with _torch_stream_context(current_stream):
                dIndexK.copy_(dk_f32_flat.view(dIndexK.shape))

    return _run
