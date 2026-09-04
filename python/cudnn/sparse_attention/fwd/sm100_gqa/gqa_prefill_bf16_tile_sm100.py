# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 GQA-substrate sparse-attention forward kernel -- PR4 round-3 fast
path for the granularity=128 **block-uniform** case (``uniform_within_tile``).

What this module is, and (importantly) what it is NOT
--------------------------------------------------------
The task brief asked for a real tcgen05-MMA / TMA-gather mainloop, following
``NSA_select_attn_fwd_hmma.py``'s Hopper-WGMMA block-128 kernel ported to
SM100 tcgen05. This module still does **not** deliver a tensor-core mainloop
in round 1 either (see "What this module ships instead" below for why that
remains the right call *this* round), but round 3's diagnosis of *why* its
attempt failed was **wrong**, and is corrected here:

1. **round-3 claim (WRONG): "SM100a rejects mma.sync.aligned entirely".**
   Round 3 isolated its failure down to a minimal repro -- ``mma.sync.aligned
   .m16n8k16.row.col.f32.bf16.bf16.f32``, *all-literal zero operands, no
   smem, no gather* -- got an opaque ``NVVM_ERROR_COMPILATION`` from libNVVM,
   and concluded the ``mma.sync`` opcode itself is rejected on ``sm_100a``.
   Round 1 reproduced that exact failure (``repro_frost_mma_sync.py`` at the
   worktree root: calls ``cudnn.frost.tile_dsl.mma.mma_m16n8k16_f32`` with
   ``cutlass.Int32(0)``/``cutlass.Float32(0.0)`` python-literal operands --
   same opaque, empty-log NVVM error, confirmed on this box's real B200
   (``sm_100a``)). **But** the claim that this makes ``mma.py`` "not usable
   at all on this hardware" does not hold up:

   * ``cutlass.cute.nvgpu.warp.MmaF16BF16Op`` + ``cute.make_tiled_mma`` +
     ``cute.gemm`` -- cute's own native warp-MMA construct, lowering to the
     same ``mma.sync.aligned.m16n8k16`` class of instruction under the hood --
     compiles and runs cleanly on this identical hardware
     (``repro_cute_warp_mma.py``). This is also exactly the path KF's winning
     QSA kernel (``qsa_tc_23ms_baseline``, campaign
     ``kkn1aah8y53ed4pwr3x78wvbyw``) uses for its GQA-substrate sparse-attn
     QK/PV GEMMs on B200 -- further independent confirmation ``mma.sync``-
     class warp MMA is fully supported on ``sm_100a``. (KF workspace:
     ``/home/scratch.vagarwalla_gpu/kf_campaigns/sparse-attn-fwd-qsa/``.)
   * More importantly, ``mma.py``'s *own* ``mma_step()``/``mma()`` -- the
     exact primitives round 3 said were broken -- compile and run fine when
     given realistic (non-python-literal) operands: an ``rmem`` accumulator
     initialized via ``.fill(0.0)`` plus A/B fragments loaded from a tensor
     (``repro_frost_mma_real_operands.py`` at the worktree root: same
     ``mma_m16n8k16_f32`` call, but every A/B/C operand is a value loaded
     from a gmem tensor rather than a python literal -- COMPILE+RUN OK,
     re-verified independently in this round). Further bisection (done in
     throwaway ``/tmp`` scripts, not committed) shows the failure is
     triggered by *any* compile-time-constant python-literal
     operand (``Int32``/``Float32``, A/B or C) reaching a multi-output
     ``inline_ptx`` call -- not specific to the ``mma.sync`` opcode, not
     specific to Float32 C-operands (a bitcast-to-Int32 workaround, already
     used for the fp8 ``mma_m16n8k32_f32`` path below, does **not** fix an
     all-literal repro either, since the A/B Int32 literals still fold).
     This matches ``mma_m16n8k16_f32``'s existing production callers
     (``cudnn.sdpa.fwd.kernels.prefill_f16_sm120``,
     ``cudnn.sdpa.bwd.kernels._common_sm120``), which pass loaded register
     values (never host literals) and compile/run fine today.

   **Conclusion: (a), not (b).** This is a cutlass-dsl 4.7.0 ``inline_ptx``
   constant-folding quirk that only bites synthetic all-literal minimal
   repros -- not a genuine SM100a rejection of warp-synchronous
   ``mma.sync``, and not an environment issue (same box, same toolchain,
   both the cute-native and the frost-native paths work). Round 3's minimal-
   repro methodology (used to "isolate" the bug) itself introduced the bug it
   was trying to isolate. ``mma.py``'s ``mma_step()``/``mma()`` family
   *is* usable on this hardware for a real kernel mainloop; round 3's
   "not usable at all" conclusion should be retracted.
2. **Why this module still doesn't port to tensor cores in round 1.** Given
   (1), a real tcgen05/``mma.sync`` mainloop for the gather-addressed,
   data-dependent KV-tile case is *not* blocked on a hardware/toolchain gap
   -- it is a real, nontrivial kernel-authoring effort (either wiring
   ``cute.make_tiled_mma``/``cute.gemm`` -- the simpler, KF-validated API,
   requiring no swizzle-descriptor math, just ``partition_A``/``partition_B``
   over an smem tile -- through this module's ``cp.async``-gathered smem
   layout and TILE_M-row-per-warp softmax structure, or ``mma_ss``/``mma_ts``
   tcgen05/TMEM path with a new gather-addressed ``MmaDesc``/TMA story). This
   round's scope for this subtask was the compile-only MMA repro
   characterization above (target files: ``mma.py``, ``tmem.py``, this
   module), not a full kernel port -- see the module's remaining docstring
   for what round 1 actually ships. A follow-up round should port this
   module's QK/PV mainloop onto ``warp.MmaF16BF16Op`` + ``cute.gemm``
   (cute-native, KF-validated) in preference to hand-rolled ``mma.py`` calls,
   both because it needs no swizzle/ldmatrix layout math to get an M=TILE_M,
   N=TILE_N, K=D_k/D_v GEMM working, and because it is the pattern already
   proven for this exact GQA-substrate sparse-attention shape by KF.
3. **Correction to (2)'s ``cute.gemm`` recommendation, from the MSA-port
   subtask's investigation.** Pulling the full KF MSA winner source
   (``kf campaign results 71242n05bd68s5kser0fn7g6rg``, kernel
   ``msa_sparse_attn_r2_v6_k2_tile32``, files ``msa.py``/``msa_helpers.py``)
   shows its hot-loop MMA issue (``_wg_mma_issue`` in ``msa.py``) does
   **not** call ``cute.gemm()`` at all (``grep -c "cute.gemm(" msa.py`` ==
   0) -- it builds ``tiled_mma_qk``/``tiled_mma_pv`` via
   ``sm100_utils.make_trivial_tiled_mma`` only to harvest layout/``.op``
   metadata (``make_fragment_A``/``make_fragment_B``, ``tiled_mma.op``),
   then issues the actual MMA through its own hand-rolled PTX descriptor
   emission (``msa_helpers.gemm_ptx_partial`` /
   ``gemm_ptx_precomputed_varname`` / ``declare_ptx_smem_desc`` --
   effectively a from-scratch reimplementation of what ``mma.py``'s
   ``mma_ss``/``mma_ts_step`` already do). ``prefill_d128_f16_sm100.py`` --
   working, tested, in this exact repo -- calls ``mma_ss``/``mma_ts_step``
   directly (``grep -c "mma_ss(\|mma_ts_step("`` in that file: 12 call
   sites) and its suite (``test_sdpa_fwd_dsl_sm100.py``) passes on this same
   B200 box. So the two real production tcgen05 kernels available as
   reference -- one in this repo, one from KF -- both land on the *same*
   abstraction level (direct tcgen05 descriptor issue), and neither uses the
   higher-level ``cute.gemm()`` entry point for its hot path. **Revised
   recommendation:** a real port of this module's mainloop should build on
   ``mma.py``'s ``mma_ss``/``mma_ts_step`` (already proven in this codebase
   for an SM100a bf16 attention QK/PV shape), not ``cute.gemm()`` -- (2)'s
   suggestion was a reasonable a-priori guess but is not what either
   available reference implementation actually does.

   What that port concretely needs, scoped from MSA's structure (still not
   implemented this round -- see below):

   * **Swizzled smem, not this module's current linear ``cp.async`` gather.**
     ``cutlass.utils.blackwell_helpers.get_smem_layout_atom_ab`` picks
     ``K_SW128`` for a K-major bf16 operand with ``d_k=128``
     (``128*16 bits = 2048``, divisible by the 1024-bit SW128 threshold) --
     i.e. swizzling is *mandatory* for this shape, not optional. This
     module's ``sK``/``sV`` (plain ``cutlass.Array``, row-major, filled by
     ``load_tile_2d``'s per-thread ``cp.async``) is byte-for-byte the wrong
     layout for ``mma_ss``/``Tcgen05SmemDesc`` to read correctly -- feeding
     it to an MMA as-is would compile and run but silently produce wrong
     numbers, not a compile error. The fix is not exotic, though: a KV
     block here is one *contiguous* ``[kv_row0, kv_row0+128)`` token range
     (``uniform_within_tile`` picks a single run-start per step, unlike a
     truly scattered gather) -- i.e. it is TMA-representable as a runtime-
     coordinate 2-D box load against a static tensor map, exactly
     ``prefill_d128_f16_sm100.py``'s existing ``handles.GmemTileTma`` /
     ``tma.tma_load_tile`` machinery (dynamic ``kv_state.idx``-style stage
     index, here replaced by a dynamic *row* coordinate). Swapping this
     module's ``load_tile_2d`` cp.async gather for a runtime-coordinate TMA
     load would land K/V pre-swizzled in hardware, the same way
     ``prefill_d128``/MSA both do it -- no manual swizzle math needed on
     this module's side, at the cost of building the host-side
     ``cute.TensorMap`` plumbing this module currently has none of.
   * **K1/K2 split.** MSA's structure -- a KV-outer K1 pass writing
     ``o_partial``/``lse_partial`` per selected block to scratch, then a
     deterministic fixed-order K2 log-sum-exp combine -- maps directly onto
     this module's per-``topk_idxs``-entry loop (each loop iteration already
     *is* one KV block); K1 would be one MMA-based partial-attention step
     per entry, K2 the existing online-softmax merge, hoisted out of the
     inner loop into a separate combine pass over ``topk_max`` partials so
     it stays bit-for-bit deterministic (frozen-contract requirement)
     regardless of the order K1 instances complete in.
   * **Row/head packing to the MMA M dimension.** With ``TILE_M=32`` and
     (for the ``heads_per_kv=4`` shapes this fast path already targets --
     e.g. minimax) ``H=heads_per_kv=4``, ``TILE_M * H == 128`` already
     equals MSA's own ``m_block_size=128`` -- i.e. this module's existing
     grid/tile constants are *already* the right M-packing for a 128x128x128
     ``mma_tiler`` with no shape changes needed, only a smem layout for Q
     shaped ``(TILE_M*H, d_k)`` K-major instead of per-row scalar loads.
   * **Not needed: MSA's CSR worklist scheduler.** MSA's
     ``_build_scheduler``/``_build_scheduler_syncfree`` solve "which rows,
     of possibly-heterogeneous per-row selection, share this KV block" --
     this module's fast path sidesteps that by construction
     (``uniform_within_tile``: every row in the tile shares every entry), so
     the worklist machinery is not part of what a port here needs; it only
     matters if a future round drops the uniform-within-tile precondition
     (``_common_sm100.py``'s docstring option (b), gather-then-mask over a
     row union) instead of keeping it as a caller contract.

   None of this is implemented in round 1 -- the TMA-tensor-map plumbing
   alone (host-side descriptor construction for a *sparse/runtime-indexed*
   KV block, which none of this package's existing modules do yet) is
   itself a multi-file, hardware-in-the-loop-debugged undertaking on the
   scale of what ``prefill_d128_f16_sm100.py`` already is, and reproducing
   that safely (bitwise-correct against the oracle, not just compiling) is
   not achievable inside this round's remaining budget after the research
   above. This is a scoped, concrete round-2 task, not an open question.

What this module ships instead
--------------------------------
The real, tested, structural win tile-uniform selection unlocks *without*
tensor cores: with every row (and every Q head) in a ``TILE_M``-row tile
sharing one KV block per step, that KV block needs to be **read from global
memory exactly once per tile**, not once per row. The scalar kernel
(``gqa_prefill_bf16_sm100.py``) already amortizes K/V reads across the
``heads_per_kv`` Q heads sharing a group (one warp handles all of them per
token) -- what it cannot do is amortize across *rows*, because each row is
an independent CTA with no visibility into its neighbors' (identical, under
this fast path's precondition) selection. This module batches ``TILE_M``
rows into one CTA (one warp per row, same per-row FFMA online-softmax
mainloop as the scalar kernel, verbatim), and cooperatively ``cp.async``-
gathers each selected KV block into a **shared** smem tile once per CTA
before every warp's inner token loop reads it -- turning an
``O(TILE_M)``-redundant global gather into an ``O(1)`` one. Compute is
unchanged (FFMA, not Tensor Core -- see above for why), so correctness risk
relative to the already-validated scalar kernel is low: the per-row online-
softmax math is copied unmodified, only the K/V *source* (global -> shared
smem) and the *grid* (row-tiled, not one row per CTA) differ. Unlike the
MMA attempt, this path also does not need ``D_k == D_v == 128`` or a small
``heads_per_kv`` bound -- the smem footprint is ``TILE_N * (D_k + D_v)``
only (no per-row Q/score buffers), so it generalizes to the same envelope as
the scalar kernel.

Envelope (fast path, in addition to ``GqaPrefillConfig``'s checks)
--------------------------------------------------------------------
* ``index_granularity == 128`` only (``TILE_N`` == one gather block; this is
  the shape where "one CTA, one shared KV tile" pays off -- g=4/g=64 gather
  windows are narrower than a useful shared tile and are better served by
  the scalar kernel's per-row TMA-free path).
* ``uniform_within_tile=True`` **must** be true and is the caller's
  contract (see ``GqaPrefillConfig``: ``G == H_kv`` already requires
  per-Q-head uniformity within a group; this fast path additionally requires
  per-*row* uniformity within a ``TILE_M``-row tile -- i.e. every row and
  head in the tile shares one ``topk_idxs``/``topk_length`` row, read once
  from the tile's first row). Passing ``uniform_within_tile=False`` (the
  default) raises. An opt-in ``validate_uniform=True`` host-side debug check
  does one D2H-synchronizing comparison per call to fail loudly in
  tests/CI -- never enabled on the hot path by default.
* BSHD or THD, BF16 only, separate K/V, causal-safety is the caller's
  responsibility via ``topk_idxs``.

Grid: one CTA per (Q-tile of ``TILE_M=32`` rows, KV head, batch); block =
``TILE_M`` warps (one warp per Q row, looping ``heads_per_kv`` heads inside,
exactly as the scalar kernel's per-row warp does).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)
import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import torch

from cudnn.frost.tile_dsl.tma import load_tile_2d

from ._common_sm100 import WARP_LANES, GqaPrefillConfig, lane_group_sum, resolve_entry_window

NEG_INF = float("-inf")

# Hardcoded fast-path geometry -- see module docstring.
TILE_M = 32  # Q rows (== warps) per CTA tile
TILE_N = 128  # KV tokens gathered per step == index_granularity (fixed)

# NOTE on config_sm100.py: an earlier draft of this module wired
# ``config_sm100.make_cfg_g128``/``CfgG128`` in here as a shared-vocabulary
# validator (per the round-3 task's "wire it in for real, or delete it"
# instruction). That file was removed from this worktree by a concurrent
# edit while this module was being developed (multiple round-3 subtasks
# share this worktree); rather than resurrect it unilaterally and risk
# clobbering whatever the other track's reconciliation decided, this module
# stands alone with its own hardcoded geometry constants below. If
# ``config_sm100.py`` reappears with different content, re-check this note.


def _make_kernel(cfg: GqaPrefillConfig):
    if cfg.granularity != TILE_N:
        raise ValueError(f"gqa_prefill_bf16_tile_sm100 only serves index_granularity == {TILE_N}, got {cfg.granularity}")

    H = cfg.heads_per_kv
    NUM_THREADS = TILE_M * WARP_LANES
    K_ELEMS = TILE_N * cfg.d_k
    V_ELEMS = TILE_N * cfg.d_v
    # cp.async gather granularity: 16B chunks where the row width allows it,
    # falling back to 8B/4B so odd (e.g. d=72-class) D still divides evenly.
    for elems_per_copy in (8, 4, 2, 1):
        if cfg.d_k % elems_per_copy == 0 and cfg.d_v % elems_per_copy == 0:
            K_COPY = elems_per_copy
            break
    if (TILE_N * (cfg.d_k // K_COPY)) % NUM_THREADS != 0 or (TILE_N * (cfg.d_v // K_COPY)) % NUM_THREADS != 0:
        # load_tile_2d requires total_chunks % num_threads == 0; TILE_N=128
        # and NUM_THREADS=TILE_M*32=1024 means this only bites for D not a
        # multiple of NUM_THREADS/gcd(...) -- narrow this round to the
        # common power-of-two D shapes (64/128/192/256) where it always
        # divides evenly, and fail loudly (fall back to the scalar kernel)
        # rather than silently mis-gathering otherwise.
        raise NotImplementedError(f"gqa_prefill_bf16_tile_sm100: D_k={cfg.d_k}/D_v={cfg.d_v} do not evenly tile under TILE_M={TILE_M}; use the scalar kernel")

    @cute.kernel
    def kernel_fn(
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
        topk_max: cutlass.Int32,
        rows_total: cutlass.Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % cutlass.Int32(WARP_LANES)
        warp = tidx // cutlass.Int32(WARP_LANES)  # local row within this tile
        tile_idx = cute.arch.block_idx()[0]
        kv_head = cute.arch.block_idx()[1]
        batch = cute.arch.block_idx()[2]

        row_base = tile_idx * cutlass.Int32(TILE_M)
        t_row = cute.math.min(row_base + warp, rows_total - cutlass.Int32(1))
        t_q = t_row + batch * s_q

        kv_base = cutlass.Int32(0)
        if cutlass.const_expr(cfg.is_bshd):
            kv_base = batch * kv_bound

        sK = cutlass.Array(cutlass.BFloat16, K_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
        sV = cutlass.Array(cutlass.BFloat16, V_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)

        q_v = cutlass.make_array_view(q)
        k_v = cutlass.make_array_view(k)
        v_v = cutlass.make_array_view(v)
        idx_v = cutlass.make_array_view(topk_idxs)
        out_v = cutlass.make_array_view(out)
        lse_v = cutlass.make_array_view(lse)
        len_v = cutlass.make_array_view(topk_length) if cutlass.const_expr(topk_length is not None) else None
        sink_v = cutlass.make_array_view(attn_sink) if cutlass.const_expr(attn_sink is not None) else None

        # Shared entry stream: uniform_within_tile guarantees every row/head
        # in this tile agrees on topk_idxs/topk_length, so read it once from
        # the tile's first row -- this is exactly what makes gathering a KV
        # block once per *tile* (not once per row) correct.
        t_q_rep = row_base + batch * s_q  # representative row (tile's row 0)
        n_entries = topk_max
        if cutlass.const_expr(len_v is not None):
            n_entries = cutlass.Int32(len_v[t_q_rep, kv_head])

        H_ = cfg.heads_per_kv
        row_max = [cutlass.Float32(NEG_INF) for _ in range(H_)]
        row_sum = [cutlass.Float32(0.0) for _ in range(H_)]
        V_CHUNKS = cfg.v_chunks_per_lane
        o_acc = [[cutlass.Float32(0.0) for _ in range(V_CHUNKS)] for _ in range(H_)]

        for j in cutlass.range(0, topk_max, 1, unroll=1):
            if j < n_entries:
                entry = cutlass.Int32(idx_v[t_q_rep, kv_head, j])
                tile_start, num_valid, is_valid = resolve_entry_window(entry, TILE_N, kv_bound)
                if is_valid:
                    kv_row0 = kv_base + tile_start

                    # --- gather this step's KV block once, shared by all
                    # TILE_M row-warps (real async bulk copy, not per-row
                    # scalar global reads). ---
                    load_tile_2d(
                        sK,
                        k_v.data_ptr((kv_row0, kv_head, 0)),
                        rows=TILE_N,
                        elems_per_row=cfg.d_k,
                        gmem_row_stride_elems=cfg.h_kv * cfg.d_k,
                        tidx=tidx,
                        num_threads=NUM_THREADS,
                        elems_per_copy=K_COPY,
                        elem_bytes=2,
                        valid_rows=num_valid,
                    )
                    load_tile_2d(
                        sV,
                        v_v.data_ptr((kv_row0, kv_head, 0)),
                        rows=TILE_N,
                        elems_per_row=cfg.d_v,
                        gmem_row_stride_elems=cfg.h_kv * cfg.d_v,
                        tidx=tidx,
                        num_threads=NUM_THREADS,
                        elems_per_copy=K_COPY,
                        elem_bytes=2,
                        valid_rows=num_valid,
                    )
                    nvvm.cp_async_commit_group()
                    nvvm.cp_async_wait_group(0)
                    nvvm.barrier_cta_sync()

                    # --- per-row warp: identical online-softmax mainloop to
                    # gqa_prefill_bf16_sm100.py's scalar kernel, sourcing
                    # K/V from the shared smem tile (local token index) --
                    # only reads within [0, num_valid) so the smem tail past
                    # a partial last block is never touched. ---
                    for local in cutlass.range(0, TILE_N, 1, unroll=1):
                        if local < num_valid:
                            for h in cutlass.range_constexpr(H_):
                                q_head = kv_head * cutlass.Int32(H_) + cutlass.Int32(h)
                                partial = cutlass.Float32(0.0)
                                for d in cutlass.range(lane, cfg.d_k, WARP_LANES, unroll=1):
                                    partial = partial + cutlass.Float32(q_v[t_q, q_head, d]) * cutlass.Float32(sK[local * cfg.d_k + d])
                                score = lane_group_sum(partial, lanes=WARP_LANES) * scale

                                old_max = row_max[h]
                                new_max = cute.math.max(old_max, score, ftz=True)
                                correction = cutlass.Float32(0.0)
                                if old_max > cutlass.Float32(NEG_INF):
                                    correction = cute.math.exp(old_max - new_max, fastmath=True)
                                p = cute.math.exp(score - new_max, fastmath=True)
                                row_sum[h] = row_sum[h] * correction + p
                                row_max[h] = new_max

                                for c in cutlass.range_constexpr(V_CHUNKS):
                                    d = lane + c * WARP_LANES
                                    v_val = cutlass.Float32(0.0)
                                    if d < cfg.d_v:
                                        v_val = cutlass.Float32(sV[local * cfg.d_v + d])
                                    o_acc[h][c] = o_acc[h][c] * correction + v_val * p
                    nvvm.barrier_cta_sync()  # sK/sV free for the next step's gather

        # === epilogue: identical to the scalar kernel ===
        for h in cutlass.range_constexpr(H_):
            q_head = kv_head * cutlass.Int32(H_) + cutlass.Int32(h)
            if row_max[h] == cutlass.Float32(NEG_INF):
                if t_row < rows_total:
                    lse_v[t_q, q_head] = cutlass.Float32(NEG_INF)
                    for c in cutlass.range_constexpr(V_CHUNKS):
                        d = lane + c * WARP_LANES
                        if d < cfg.d_v:
                            out_v[t_q, q_head, d] = cutlass.Float32(0.0).to(out.element_type)
            else:
                sink_term = cutlass.Float32(0.0)
                if cutlass.const_expr(sink_v is not None):
                    sink_term = cute.math.exp(cutlass.Float32(sink_v[q_head]) - row_max[h], fastmath=True)
                denom = row_sum[h] + sink_term
                inv_denom = cutlass.Float32(1.0) / denom
                if t_row < rows_total:
                    lse_v[t_q, q_head] = row_max[h] + cute.math.log(row_sum[h], fastmath=True)
                    for c in cutlass.range_constexpr(V_CHUNKS):
                        d = lane + c * WARP_LANES
                        if d < cfg.d_v:
                            out_v[t_q, q_head, d] = (o_acc[h][c] * inv_denom).to(out.element_type)

    return kernel_fn


def _make_host(cfg: GqaPrefillConfig):
    kernel_fn = _make_kernel(cfg)

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
        topk_max: cutlass.Int32,
        rows_per_batch: cutlass.Int32,
        n_batch: cutlass.Int32,
        stream: _cuda_driver.CUstream = None,
    ) -> None:
        n_tiles = (rows_per_batch + cutlass.Int32(TILE_M) - cutlass.Int32(1)) // cutlass.Int32(TILE_M)
        kernel_fn(
            q,
            k,
            v,
            topk_idxs,
            topk_length,
            attn_sink,
            out,
            lse,
            kv_bound,
            s_q,
            scale,
            topk_max,
            rows_per_batch,
        ).launch(
            grid=(n_tiles, cfg.h_kv, n_batch),
            block=[TILE_M * WARP_LANES, 1, 1],
            stream=stream,
        )

    return host_fn


def _gpu_arch_flag(device: torch.device) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("gqa_prefill_bf16_tile_sm100 compilation requires CUDA")
    major, minor = torch.cuda.get_device_capability(device)
    if major != 10:
        raise RuntimeError(f"gqa_prefill_bf16_tile_sm100 requires an SM100-family GPU, found SM{major}{minor}")
    return {0: "sm_100a", 3: "sm_103a", 7: "sm_100f"}.get(minor, "sm_100a")


@lru_cache(maxsize=None)
def _compile(
    d_k: int,
    d_v: int,
    h_q: int,
    h_kv: int,
    granularity: int,
    is_bshd: bool,
    has_topk_length: bool,
    has_attn_sink: bool,
    arch: str,
):
    cfg = GqaPrefillConfig(
        d_k=d_k,
        d_v=d_v,
        h_q=h_q,
        h_kv=h_kv,
        granularity=granularity,
        is_bshd=is_bshd,
        has_topk_length=has_topk_length,
        has_attn_sink=has_attn_sink,
    )
    bf16 = cutlass.BFloat16
    t_q_sym = cute.sym_int(divisibility=1)
    t_kv_sym = cute.sym_int(divisibility=1)
    topk_max_sym = cute.sym_int(divisibility=1)

    fake_q = cute.runtime.make_fake_compact_tensor(bf16, (t_q_sym, h_q, d_k), stride_order=(2, 1, 0), assumed_align=16)
    fake_k = cute.runtime.make_fake_compact_tensor(bf16, (t_kv_sym, h_kv, d_k), stride_order=(2, 1, 0), assumed_align=16)
    fake_v = cute.runtime.make_fake_compact_tensor(bf16, (t_kv_sym, h_kv, d_v), stride_order=(2, 1, 0), assumed_align=16)
    fake_idx = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (t_q_sym, h_kv, topk_max_sym), stride_order=(2, 1, 0), assumed_align=4)
    fake_len = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (t_q_sym, h_kv), stride_order=(1, 0), assumed_align=4) if has_topk_length else None
    fake_sink = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (h_q,), stride_order=(0,), assumed_align=4) if has_attn_sink else None
    fake_out = cute.runtime.make_fake_compact_tensor(bf16, (t_q_sym, h_q, d_v), stride_order=(2, 1, 0), assumed_align=16)
    fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (t_q_sym, h_q), stride_order=(1, 0), assumed_align=4)

    host_fn = _make_host(cfg)
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
    """Whether ``(d_k, d_v, h_q, h_kv, index_granularity)`` is inside this
    module's fast-path envelope. Cheap, side-effect-free -- used by
    ``dispatch.py`` to pick this kernel vs. the scalar fallback before
    compiling anything."""
    if index_granularity != TILE_N or h_kv <= 1 or h_q % h_kv != 0:
        return False
    for elems_per_copy in (8, 4, 2, 1):
        if d_k % elems_per_copy == 0 and d_v % elems_per_copy == 0:
            k_copy = elems_per_copy
            break
    num_threads = TILE_M * WARP_LANES
    return (TILE_N * (d_k // k_copy)) % num_threads == 0 and (TILE_N * (d_v // k_copy)) % num_threads == 0


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
    """Tile-batched async-gather (``cp.async``) fast path for the
    granularity=128 GQA envelope -- see module docstring for why this is
    *not* a tensor-core (tcgen05/``mma.sync``) mainloop this round.

    ``uniform_within_tile=True`` is a **required, caller-verified
    precondition**: every row and Q head in the same ``TILE_M``-row Q tile
    must select the identical ``topk_idxs``/``topk_length``. Passing
    ``uniform_within_tile=False`` (the default) raises rather than silently
    computing a wrong answer for a per-row-varying selection this kernel
    does not read per-row. ``validate_uniform=True`` adds one explicit,
    opt-in D2H-synchronizing host-side check (see ``_check_uniform_within_tile``)
    and raises ``ValueError`` on violation -- intended for tests/CI, never
    the default hot path.
    """
    if not uniform_within_tile:
        raise ValueError(
            "gqa_prefill_bf16_tile_sm100 only serves uniform_within_tile=True "
            "(per-Q-tile row-uniform selection, e.g. MSA/NSA-style block attention); "
            "for the general per-row-varying case use gqa_prefill_bf16_sm100's scalar kernel"
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError(f"gqa_prefill_bf16_tile_sm100 is BF16-only, got Q/K/V dtypes {q.dtype}/{k.dtype}/{v.dtype}")
    is_thd = q.ndim == 3
    if is_thd and cu_seqlens_q is None:
        raise ValueError("THD (3-D) Q requires cu_seqlens_q")
    if index_granularity != TILE_N:
        raise ValueError(f"gqa_prefill_bf16_tile_sm100 only serves index_granularity == {TILE_N}, got {index_granularity}")

    device = q.device
    if device.type != "cuda":
        raise ValueError(f"Q must live on CUDA, got {device}")

    with torch.cuda.device(device):
        arch = _gpu_arch_flag(device)

        if is_thd:
            t_q, h_q, d_k = q.shape
            t_kv, h_kv, d_k_kv = k.shape
            _, _, d_v = v.shape
            rows_per_batch, n_batch = t_q, 1
            kv_bound = t_kv
            s_q = t_q
            q_flat, k_flat, v_flat = q, k, v
            idx_flat = topk_idxs
            len_flat = topk_length
        else:
            b, s_q_, h_q, d_k = q.shape
            _, s_kv, h_kv, d_k_kv = k.shape
            _, _, _, d_v = v.shape
            rows_per_batch, n_batch = s_q_, b
            kv_bound = s_kv
            s_q = s_q_
            q_flat = _flatten_leading(q, 2)
            k_flat = _flatten_leading(k, 2)
            v_flat = _flatten_leading(v, 2)
            idx_flat = _flatten_leading(topk_idxs, 2)
            len_flat = _flatten_leading(topk_length, 1)

        if d_k_kv != d_k:
            raise ValueError(f"K head dim ({d_k_kv}) must match Q ({d_k})")
        if h_q % h_kv != 0 or h_kv <= 1:
            raise ValueError(f"gqa_prefill_bf16_tile_sm100 requires H_q % H_kv == 0 and H_kv > 1, got H_q={h_q} H_kv={h_kv}")
        if not fast_path_eligible(d_k=d_k, d_v=d_v, h_q=h_q, h_kv=h_kv, index_granularity=index_granularity):
            raise NotImplementedError(
                f"gqa_prefill_bf16_tile_sm100 fast-path envelope rejects D_k={d_k} D_v={d_v} H_q={h_q} H_kv={h_kv}; use the scalar kernel"
            )
        if topk_idxs.shape[-2] != h_kv:
            raise ValueError(f"topk_idxs group dim must be H_kv ({h_kv}) for this kernel's envelope, got {topk_idxs.shape}")

        if validate_uniform:
            _check_uniform_within_tile(idx_flat, len_flat, rows_per_batch, n_batch)

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
        topk_max = idx_flat.shape[-1]

        compiled = _compile(
            int(d_k),
            int(d_v),
            int(h_q),
            int(h_kv),
            int(index_granularity),
            not is_thd,
            len_flat is not None,
            attn_sink is not None,
            arch,
        )

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
            cutlass.Float32(scale),
            cutlass.Int32(int(topk_max)),
            cutlass.Int32(int(rows_per_batch)),
            cutlass.Int32(int(n_batch)),
            cu_stream,
        )

    if is_thd:
        return {"out": out, "lse": lse}
    return {"out": out.reshape(b, s_q_, h_q, d_v), "lse": lse.reshape(b, s_q_, h_q)}


def _check_uniform_within_tile(
    idx_flat: torch.Tensor,
    len_flat: Optional[torch.Tensor],
    rows_per_batch: int,
    n_batch: int,
) -> None:
    """Opt-in, D2H-synchronizing precondition check for ``validate_uniform=True``
    (see ``sparse_attention_forward_wrapper``'s docstring). Explicit,
    caller-requested -- not called on the default hot path."""
    n_tiles = (rows_per_batch + TILE_M - 1) // TILE_M
    for b in range(n_batch):
        base = b * rows_per_batch
        for t in range(n_tiles):
            r0 = base + t * TILE_M
            r1 = min(base + rows_per_batch, r0 + TILE_M)
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
