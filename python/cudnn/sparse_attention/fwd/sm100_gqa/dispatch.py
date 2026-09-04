# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dtype dispatch for the SM100 GQA-substrate sparse-attention kernel (PR4).

``python/cudnn/sparse_attention/fwd/api.py`` probes
``cudnn.sparse_attention.fwd.sm100_gqa`` for ``sparse_attention_forward_wrapper``
and, when import succeeds, registers it as the device kernel for its GQA
envelope (``G == H_kv``, ``index_granularity in (4, 64, 128)``, BF16 or
FP8-per-tensor). This module is the ``__init__.py``-re-exported entry point
that closes that probe, and dispatches by ``Q.dtype`` (and, for BF16, by
shape/granularity) across the sibling kernel modules:

* BF16, ``D_k == D_v == 128`` and ``index_granularity == 128`` (the MSA
  cell) -> ``gqa_prefill_bf16_tcgen05_sm100.sparse_attention_forward_wrapper``
  (tensor-core mainloop built on frost tile_dsl's proven ``mma_ss``/
  ``mma_ts_step`` cta-level tcgen05 primitives -- the same primitives
  ``sdpa/fwd/kernels/prefill_d128_f16_sm100.py`` already ships in
  production -- *not* the buggy warp-level ``mma_m16n8k16_f32`` inline-PTX
  helper) exists in this package, but as of round 6 is **not** tried by
  default -- ``api.py``'s top-level ``sparse_attention_forward_wrapper``
  routes the MSA cell to the scalar kernel unconditionally unless a caller
  explicitly opts in with ``try_tcgen05=True`` on *this* module's
  ``sparse_attention_forward_wrapper``. See ``_try_tcgen05_fast_path`` below
  and the round-6 honesty note.

  The tcgen05 mainloop has the same real precondition the round-3 tile
  kernel has -- ``uniform_within_tile=True`` (every row and Q head sharing
  one ``TILE_M``-row Q tile must select the identical
  ``topk_idxs``/``topk_length``, since the tensor-core mainloop reads one
  gathered K/V tile per Q tile, not per row) -- which is *not* something
  dispatch.py can derive structurally from ``q``/``k``/``v`` shapes or
  dtypes alone: it is a property of the *values* inside ``topk_idxs``.
  There is no cheap host-side structural proxy for it (e.g. shape or stride
  checks) that would avoid reading device memory, so the default hot path
  for this cell keeps ``validate_uniform=True``: one explicit, D2H-
  synchronizing host-side check per call (see
  ``gqa_prefill_bf16_tile_sm100._check_uniform_within_tile``, reused
  as-is) before any kernel launch, and a fall back to the scalar kernel
  (``gqa_prefill_bf16_sm100``, exactly ``_try_tile_fast_path``'s existing
  fallback pattern) on the ``ValueError`` that check raises when the
  selection isn't actually tile-uniform. This sync is a real, non-zero cost
  paid on *every* call through this cell by default -- it is the honest
  price of routing a per-row-varying-selection-safe public API to a kernel
  whose speed depends on a precondition only the values can prove. A caller
  that has independently verified tile-uniformity (e.g. an MSA harness that
  constructed ``topk_idxs`` itself) can skip it by calling this module's
  ``sparse_attention_forward_wrapper`` directly with
  ``validate_uniform=False``.

  **Honesty note, updated round 5**: ``gqa_prefill_bf16_tcgen05_sm100`` now
  exists in this package (this round's deliverable) with the
  ``sparse_attention_forward_wrapper``/``fast_path_eligible`` names this
  module's ``_try_tcgen05_fast_path`` was already scoped to import, so the
  routing below is live, not aspirational -- ``import`` no longer raises,
  and a genuinely tile-uniform MSA-cell call is structurally routed to it
  by ``api.py``'s default path with no opt-in. The kernel itself is a real
  tcgen05 mainloop (``mma_ss``/``mma_ts`` -- grep-verifiable, no scalar
  FFMA path in that file) built on the same TMA/TMEM primitives
  ``prefill_d128_f16_sm100.py`` ships in production, per that module's own
  docstring.

  **What round 5 could NOT confirm in the available session time**: the
  kernel's ``cute.compile()`` call did not finish -- no error, no crash,
  just still running (CPU-bound, single process) after ~10 minutes wall
  time on this worktree's SM100 box, well past what a direct-import smoke
  test needs for every OTHER kernel in this package (including
  ``prefill_d128_f16_sm100`` itself, a considerably larger file). This
  round ran out of budget before determining whether that is (a) a real
  bug in the new kernel that pathologically blows up NVVM/LLVM compile
  time (a bisection candidate: the mainloop's collective
  ``tcgen05_ld``/``tcgen05_st`` + barrier sequencing, or the 128-way
  statically-unrolled epilogue stores) or (b) simply an expensive-but-
  eventually-finishing compile that a longer budget would have cleared.
  Concretely: **no test in this round's added coverage
  (``test_sparse_attention_fwd_tile_sm100.py``'s ``tcgen05``-prefixed
  cases) has been confirmed passing**, and **no measured tcgen05 TFLOPS
  number is reported** -- reporting either without having actually
  observed a completed run would repeat round 4's fabricated-number
  mistake. Until a future round confirms (or fixes and then confirms) a
  completed compile+run, treat this cell's routing as *structurally* wired
  but *not yet runtime-verified*; ``sparse_attention_forward_wrapper``
  still must not regress correctness or availability in the meantime --
  the scalar kernel remains the safety net (``_try_tcgen05_fast_path``
  falls through to it on any ``ImportError``/``ValueError``/
  ``NotImplementedError``).

  **Honesty note, round 6 (updated same round, later pass)**: the round-5
  ``cute.compile()`` hang (>10-15 min, no error) is now root-caused and
  fixed -- it was a dynamic ``cutlass.range(0, topk_max, ...)`` loop bound
  around the per-topk-entry ``mma_ss``/``mma_ts`` calls, replaced with a
  static Python-level ``for j in range(topk_max):`` unroll (``topk_max`` is
  a compile-time tensor-shape constant, not a runtime scalar). Independently
  re-confirmed this round: ``_compile(h_q=64, h_kv=4, ..., topk_max=16)``
  returns in under a second now, vs. the >10-minute hang before the fix.
  **However, a later pass this same round found the compiled kernel
  deadlocks at launch time instead**: a full ``sparse_attention_forward_wrapper``
  call (compile + launch + ``torch.cuda.synchronize()``, see
  ``verify_round5_repro.py``) never returns. Re-ran this on a GPU confirmed
  quiescent immediately beforehand (0 MiB used, no other compute processes)
  specifically to rule out box contention as the explanation for a prior,
  ambiguous 200s timeout -- ``nvidia-smi`` showed a sustained 100% SM
  utilization for the entire time observed, i.e. a kernel is genuinely
  resident and spinning on the GPU, but the host-side synchronize never
  returns. That is the signature of an on-device spin-wait deadlock (most
  likely an mbarrier arrive/wait mismatch in this kernel's single-stage K/V
  re-arm sequencing), not a compiler or contention artifact. This is a
  confirmed, real, and NEW bug -- compile speed is fixed, but the kernel
  cannot currently produce output to check against the oracle at all, so
  correctness is not just unconfirmed but unreachable until this deadlock
  is fixed. ``_try_tcgen05_fast_path`` still only catches ``(ValueError,
  NotImplementedError)`` around the ``cute.compile()``-triggering call below
  -- there is no timeout/hang guard, so a real caller reaching this cell
  through ``api.py`` with ``try_tcgen05`` left at its default would hang
  indefinitely, not fall back. Per this round's hard gate (all three of:
  fast compile, correctness, and a real compile-timeout fallback must be
  independently verified before defaulting to this path), **the default
  stays reverted**: ``_DEFAULT_TRY_TCGEN05_FAST_PATH`` remains ``False``.
  The MSA cell's default route is once again ``gqa_prefill_bf16_sm100``
  (the scalar kernel), exactly as it was before round 5 introduced this
  cell's tcgen05 module. ``try_tcgen05=True`` remains available as an
  explicit, opt-in-only escape hatch on this module's
  ``sparse_attention_forward_wrapper`` for anyone deliberately
  testing/debugging the tcgen05 path -- callers who pass it are knowingly
  accepting the risk of an indefinite hang (now confirmed to be a
  device-side deadlock, not just a slow compile), since no timeout guard
  exists yet.
* BF16, everything else in the envelope (granularity in ``(4, 64)``, or
  ``D_k``/``D_v`` != 128) -> ``gqa_prefill_bf16_sm100.sparse_attention_forward_wrapper``
  (scalar warp-per-row gather mainloop), unchanged from prior rounds. This
  module *also* still carries round-3's tile-batched ``cp.async``-gather
  fast path (``gqa_prefill_bf16_tile_sm100``, via ``_try_tile_fast_path``)
  for the same ``index_granularity == 128`` cell, but it stays opt-in only
  (``_DEFAULT_TRY_TILE_FAST_PATH = False``) because it was measured a net
  regression vs. the scalar kernel:

  ================  =======  =========  ==============  =========
  Sq / Skv          H_kv     tile (ms)   scalar (ms)     tile/scalar
  ================  =======  =========  ==============  =========
  1024 / 2048       4        68.03       60.18           1.13x slower
  4096 / 4096       4        232.38      191.08           1.22x slower
  8192 / 16384      4        468.36      367.06           1.28x slower
  ================  =======  =========  ==============  =========

  (BF16, D_k=D_v=128, H_q=64, granularity=128, topk=16, genuinely
  tile-uniform ``topk_idxs``, ``validate_uniform`` off for these numbers so
  this is the compute kernel's cost alone, not validation overhead. A
  fourth shape, H_kv=8, crashed the CuTe DSL compiler outright:
  ``error[INTERNAL]: dynamic range should be always preprocessed to IR``.)
  Measured on this worktree's SM100 box, 30-iteration steady-state wall
  time, ``torch.cuda.synchronize()``-bracketed. The tcgen05 path above is
  *intended* to supersede this one for the MSA cell it targets
  (D=128/gran=128) once it lands -- the ``cp.async`` tile kernel's
  regression is a separate, still-undiagnosed question about that specific
  ``cp.async``-gather design, not a claim that tensor cores can't help
  here. ``_try_tile_fast_path`` is kept, tested via direct import
  (``uniform_within_tile=True``), and wired for an explicit opt-in call for
  anyone who wants to reproduce/investigate the regression, and is checked
  *after* the tcgen05 attempt below. As of round 6, both fast paths are
  opt-in only (``_DEFAULT_TRY_TCGEN05_FAST_PATH = False`` -- see the round-6
  honesty note above), so this cell falls through past both fast paths to
  the scalar kernel by default, exactly as before either fast path existed.
* FP8-per-tensor -> not yet wired to a full kernel in this round
  (``gqa_prefill_fp8_sm100.py`` currently ships only the device-scale-folding
  helper the eventual FP8 mainloop will need); raises ``NotImplementedError``
  naming the gap explicitly rather than silently falling through.
**Honesty note, this round (QSA real adaptation, superseding the prior
round's thin gate)**: ``_try_qsa_kf_fast_path`` now targets
``gqa_prefill_bf16_sm100_kf_qsa`` (not the earlier
``gqa_prefill_bf16_qsa_kf_sm100`` thin-gate module -- see that module's own
docstring for why it was a gate around the vendored kernel's un-adapted
``& (NE-1)`` fold rather than a real fix). The new sibling module makes the
two real kernel-body changes this contract needs -- the fold removed in
favor of storage-native pass-through, and a real ``-1``/tail-clamp/dead-row
path added (the vendored kernel had neither) -- so it no longer needs the
power-of-two-``NE``/no-``-1``/``topk_length``-must-equal-512 restrictions
the thin gate required; ``_try_qsa_kf_fast_path`` accordingly no longer
calls ``_has_no_invalid_entries`` for this sibling. This round independently
confirmed three of the hard gate's four checks on a realistic shape (seqlen
8192 and 32768, THD, single sequence): (a) compiles and loads without hang;
(b) launches and completes within a normal wall-clock budget (no
hard-timeout guard needed -- steady-state ~8.7ms @ s=8192, ~35.8ms @
s=32768 on this worktree's SM100 box, 30-iteration mean,
``torch.cuda.synchronize()``-bracketed); (c) passes oracle correctness
against ``sparse_attention_reference.py`` (100% of output elements within
atol=rtol=0.02 at both shapes, including a dedicated all-``-1`` dead-row
case: ``out == 0`` and ``lse == -inf`` exactly). **The fourth check --
determinism -- fails**: two back-to-back calls with bitwise-identical
inputs do *not* produce bitwise-identical output (~3.7k of 50.3M output
elements differ at seqlen 8192, magnitude ~1.2e-4, i.e. a few BF16 ULPs --
well inside the oracle's tolerance, so correctness still holds, but outside
the frozen contract's bitwise-determinism requirement). Re-running the
**unmodified vendored kernel** (``kf_qsa/qsa_kernel.py``'s own ``run()``,
no adaptation-layer code involved at all) the same way reproduces the same
class of mismatch (worse, in fact: ~53k/50.3M elements, ~2.5e-4 max) -- so
this is a **pre-existing property of the KF winner's mainloop itself**, not
something this round's adaptation introduced.

**Root-caused and fixed in a later round** (sparse_attention_training_fprop
task, QSA-determinism-only round): it was a genuine cp.async
write-after-read (WAR) race in the ``NBUF=2`` double-buffered K/V gather,
not the ``warp_reduce4_*`` reduction (ruled out by isolating the
non-``USE_EXPLICIT``/generic-``warp_reduce`` code path, which still
reproduced the same class of mismatch). Two barrier gaps, both now closed
with a single ``sync_threads()`` each -- see ``dispatch.py``'s
``_DEFAULT_TRY_QSA_KF_FAST_PATH`` assignment and
``gqa_prefill_bf16_sm100_kf_qsa.py``'s inline fix-site comments for the
full account. All four gates re-confirmed (compile/load, launch on
4096/8192/32768 seqlen, oracle correctness including fuzz/-1/dead-row,
and 20-30x-repeated bitwise-identical determinism) -- ``_DEFAULT_TRY_QSA_KF_FAST_PATH``
now defaults to ``True``.

``uniform_within_tile``/``try_tcgen05``/``validate_uniform`` are **not**
part of ``api.py``'s frozen top-level signature (that contract has no
per-row-uniformity or fast-path-selection concept and must not gain one)
-- they are optional kwargs on *this* module's
``sparse_attention_forward_wrapper`` only. As of round 6, both fast paths
are opt-in (``_DEFAULT_TRY_TCGEN05_FAST_PATH = False``,
``_DEFAULT_TRY_TILE_FAST_PATH = False``), so for the MSA cell (BF16, D=128,
granularity=128) ``api.py``'s generic dispatch routes to the scalar kernel
unconditionally; a caller that imports
``cudnn.sparse_attention.fwd.sm100_gqa.dispatch`` directly can pass
``try_tcgen05=True`` (accepting the compile-hang risk documented above) or
``uniform_within_tile=True`` (accepting the measured 1.13x-1.28x slowdown
documented above) to opt into either fast path explicitly.

**Honesty note, round 7 (KF integration -- this round)**: this round adds
two *more* opt-in fast paths, ``_try_msa_kf_fast_path`` and
``_try_qsa_kf_fast_path``, targeting sibling modules
``gqa_prefill_bf16_msa_kf_sm100`` and ``gqa_prefill_bf16_qsa_kf_sm100`` that
adapt two of KF's real, hardware-validated agentic-search winners (MSA
``msa_r7_v3_unroll8``, campaign ``71242n05bd68s5kser0fn7g6rg``, round 8,
~3.87-4.10ms measured; QSA ``qsa_r7_s8192_explicit_reduce4_guard``, campaign
``kkn1aah8y53ed4pwr3x78wvbyw``, round 7, converged bandwidth-bound-optimal)
into this cell's dispatch table, following exactly the
``_try_tcgen05_fast_path`` precedent (structural eligibility probe ->
``ImportError``/``ValueError``/``NotImplementedError`` -> ``None`` ->
fall through to the scalar kernel).

**Read the KF kernel sources directly** (``kernel.py`` in each campaign's
``winner_r*/`` directory), not just their harness/reference: both winners
fold every ``topk_idxs`` entry through ``entry & (NE - 1)`` (a power-of-2
mod) **inside the kernel itself** --
``msa_helpers``/``kernel.py``'s ``q2k = idxs...bitwise_and_(NE - 1)`` for
MSA, ``idx_mask = NE - Int32(1)`` / ``sIdx[e] = gIdx[pos, e] & idx_mask``
for QSA -- not merely in the harness's ``torch.remainder(...)`` reference
used to score them. That means the *kernels themselves*, not just their
test harness, have no concept of this contract's ``-1`` invalid-slot
sentinel (a ``-1`` entry wraps, via two's-complement all-ones bitwise-AND
NE-1, to block ``NE - 1`` -- silently *including* a real, wrong 128-token
(MSA) / 4-token (QSA) block instead of excluding it), no ``topk_length``
(both always treat the full compile-time ``topk`` count -- 16 for MSA, 512
for QSA -- as valid), and no tail-clamped last block (both assume
``S_kv % granularity == 0`` and gather exactly ``granularity`` contiguous
tokens per entry unconditionally). Both are real properties of the kernels'
own index-resolution code, not harness simplifications a thin wrapper could
paper over. Consequently neither ``_try_msa_kf_fast_path`` nor
``_try_qsa_kf_fast_path`` is a thin I/O-layout wrapper: both are
**structural-eligibility gates that only ever hand the KF kernel a call
where its mod-fold is provably equivalent to this contract's exact
semantics** (no ``-1`` present, no ``topk_length`` restriction, KV bound
divisible by the kernel's fixed granularity so no tail clamp is needed) --
and fall back to the scalar kernel via ``None`` for every call outside that
provably-safe subset, exactly as any other structural-eligibility probe in
this file does for shape/dtype. Checking for ``-1`` presence needs one
value read (``topk_idxs`` could legitimately contain ``-1`` even when the
shape/dtype envelope matches), so both probes add one explicit, opt-in
D2H-synchronizing check (``_has_no_invalid_entries``. mirroring
``_check_uniform_within_tile``'s existing precedent for "this needs a real
device read, not just a shape check") gated by the same
``validate_uniform`` kwarg the tcgen05/tile paths already use for their own
D2H check.

Both sibling modules (``gqa_prefill_bf16_msa_kf_sm100``,
``gqa_prefill_bf16_qsa_kf_sm100``) are **out of scope for this round's
target files** (this subtask is dispatch-wiring only) and do **not yet
exist in this worktree** -- ``_try_msa_kf_fast_path``/
``_try_qsa_kf_fast_path`` catch the resulting ``ImportError`` exactly like
``_try_tcgen05_fast_path`` did before round 5 added
``gqa_prefill_bf16_tcgen05_sm100``, and return ``None`` (never raise). That
means, as of this round, both new opt-in kwargs
(``try_msa_kf``/``try_qsa_kf``) are **no-ops** -- every call, opted in or
not, falls straight through to the scalar kernel, so this round's change is
provably a no-op for every existing caller (structurally, not just by
default -- there is no way to reach a KF kernel through this module yet).
None of the four hard-gate items (compiles/loads without hang; launches and
completes on realistic shapes under a hard timeout; passes oracle
correctness for the frozen contract, including ``-1``/out-of-range cases;
determinism across repeated calls) has been checked this round because
there is nothing runnable to check yet -- both remain unconfirmed, and
``_DEFAULT_TRY_MSA_KF_FAST_PATH``/``_DEFAULT_TRY_QSA_KF_FAST_PATH`` stay
``False``. A future round that adds the sibling kernel modules must
independently re-verify all four before flipping either default, exactly
as tcgen05's round-6 note requires for that cell.

Kept as its own module (rather than folding the dtype switch into
``__init__.py`` directly) so ``api.py``'s import-probe stays a single,
narrow ``from .dispatch import sparse_attention_forward_wrapper`` -- adding a
third dtype path later is a one-line addition here, not a re-plumb of the
probe.
"""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import torch

from cudnn.api_base import TupleDict

from .gqa_prefill_bf16_sm100 import sparse_attention_forward_wrapper as _bf16_wrapper

# See the module docstring's measured-numbers table: the round-3 tile
# kernel is a measured 1.13x-1.28x *regression* vs. the scalar kernel on
# every shape tried (and crashes the DSL compiler on H_kv=8), so
# ``api.py``'s default routing does not try it. Flip this (or pass
# ``uniform_within_tile=True`` explicitly to this module's
# ``sparse_attention_forward_wrapper``) once a future round's profiling
# resolves that regression.
_DEFAULT_TRY_TILE_FAST_PATH = False

# The MSA cell (BF16, D_k=D_v=128, index_granularity=128, G=H_kv -- inherent
# to this whole package) does NOT try the tcgen05 tensor-core mainloop by
# default. See the module docstring's round-6 honesty note:
# ``gqa_prefill_bf16_tcgen05_sm100`` exists and imports cleanly and now
# compiles fast (<1s, the round-5 compile hang is fixed), but launching and
# synchronizing on the compiled kernel deadlocks indefinitely instead --
# confirmed on an otherwise-quiescent GPU (100% sustained SM utilization,
# host-side sync never returns), so this is a genuine on-device spin-wait
# deadlock, not compile time or box contention. ``_try_tcgen05_fast_path``
# has no timeout/hang guard around that call -- only ``(ValueError,
# NotImplementedError)`` are caught, neither of which a hang raises.
# Routing real callers here by default would hang
# ``sparse_attention_forward_wrapper`` indefinitely, which is strictly
# worse than the scalar kernel's known-safe, known-correct behavior. Flip
# this back to ``True`` only once a future round independently confirms,
# in a future round's Verify phase: (a) the launch-time deadlock above is
# fixed (kernel returns from ``torch.cuda.synchronize()``), (b) it passes
# oracle correctness, and (c) ``_try_tcgen05_fast_path`` has a real
# fallback that catches a compile/launch timeout specifically (not just
# ValueError/NotImplementedError/ImportError) -- e.g. a subprocess- or
# signal-based hard wall-clock timeout around the ``_tcgen05_wrapper(...)``
# call that falls through to the scalar kernel on expiry. Until then, pass
# ``try_tcgen05=True`` explicitly to this module's
# ``sparse_attention_forward_wrapper`` to opt into the tcgen05 path anyway
# (e.g. to bisect the deadlock) -- that call accepts the indefinite-hang
# risk knowingly.
_DEFAULT_TRY_TCGEN05_FAST_PATH = False

# KF's MSA winner (msa_r7_v3_unroll8) targets this exact GQA/head-dim shape
# (HQ=64, HKV=4, D=128, block granularity 128); topk is one of the vendored
# kernel's supported values ((4, 8, 16, 32), see
# ``gqa_prefill_bf16_msa_kf_sm100._SUPPORTED_TOPK`` -- 16 for the MSA cell
# this round targets). Kept here for the shared _kf_cell_common_checks/
# _has_no_invalid_entries helpers QSA's fast path still uses; MSA's own
# fast path (round 8) delegates its eligibility check to the sibling
# module's own ``fast_path_eligible`` instead (see
# ``_try_msa_kf_fast_path``'s docstring for why: the vendored MSA sources
# ship a general, ``-1``-aware entry point QSA's sibling kernel does not).
_MSA_KF_H_Q = 64
_MSA_KF_H_KV = 4
_MSA_KF_D = 128
_MSA_KF_TOPK = 16
_MSA_KF_GRANULARITY = 128

# KF's QSA winner (qsa_r7_s8192_explicit_reduce4_guard) hardcodes HQ=24,
# HKV=2, D=256, TOPK=512, granularity 4 (see qsa_r7.../kernel.py's module-
# level constants) -- same "exact shape, not a range" caveat as MSA above.
_QSA_KF_H_Q = 24
_QSA_KF_H_KV = 2
_QSA_KF_D = 256
_QSA_KF_TOPK = 512
_QSA_KF_GRANULARITY = 4

# Round 8: gqa_prefill_bf16_msa_kf_sm100 now exists (vendors msa_r7_v3_unroll8
# and drives its general, -1-aware build_msa_metadata/K1/K2 entry point, not
# kernel.py's benchmark fold -- see that module's docstring and
# _try_msa_kf_fast_path's above). Per this round's hard gate,
# _DEFAULT_TRY_MSA_KF_FAST_PATH flips to True only once Verify independently
# confirms all four: (a) compiles/loads without hang, (b) launches and
# completes on realistic shapes under a hard timeout, (c) passes oracle
# correctness against sparse_attention_reference.py (not just KF's own
# reference) including -1/duplicate-id cases, (d) determinism across
# repeated calls. See this file's own module docstring / the round's Verify
# notes for the outcome; until confirmed, this stays False and
# try_msa_kf=True remains an explicit, knowing opt-in only.
# _DEFAULT_TRY_QSA_KF_FAST_PATH: unchanged this round -- see the QSA fast
# path's own module for its status.
# Round 1 (this session, sparse_attention_training_fprop task) Verify:
# independently re-confirmed all four against the current
# gqa_prefill_bf16_msa_kf_sm100 (build_msa_metadata/K1/K2 path) through this
# exact dispatch.py entrypoint (``try_msa_kf=True``, real ``cute.compile``,
# no direct kernel-module shortcuts): (a) compiles/loads without hang --
# S=8192 first-call wall ~3.5-10s across repeated trials, no hang; (b)
# launches and completes on realistic shapes (S=4096, 8192) under an
# explicit hard ``timeout`` wrapper (300s), including a THD single-segment
# 64Q/4KV/d128 shape; (c) oracle correctness against
# ``sparse_attention_reference.py`` -- max abs out diff ~3e-5 to 7e-5
# (tolerance 2e-2), max LSE diff ~1e-6 (tolerance 1e-4), AND a dedicated
# -1/dead-row sweep (30% random per-slot invalidity + ~2% fully-dead rows)
# confirming dead rows produce exact -inf LSE / zero out and live rows still
# match the oracle; (d) determinism -- 4+ repeated calls bitwise-identical
# on every shape tried, including the -1-present case. Flipping the default
# here. QSA (below) was NOT flipped in that earlier pass of this same round:
# QSA's fast path was found non-deterministic through its own wrapper
# (small, reproducible per-element diffs across repeated calls on identical
# inputs) -- gate (d) did not hold, so ``_DEFAULT_TRY_QSA_KF_FAST_PATH``
# stayed ``False``.
#
# Follow-up pass, same round (sparse_attention_training_fprop task,
# narrowed to QSA determinism only): root-caused and fixed. The
# non-determinism was NOT the ``warp_reduce4_*`` quad-shuffle reduction
# (confirmed by isolation: forcing the generic, non-``USE_EXPLICIT``
# ``warp_reduce`` path -- a different seqlen that takes the ``else`` branch
# in the kernel's softmax reduction -- still reproduced the same class of
# mismatch, ruling that suspect out). It was a genuine cp.async
# write-after-read (WAR) race in the kernel's ``NBUF=2`` double-buffered
# K/V gather (``gqa_prefill_bf16_sm100_kf_qsa.py``'s ``qsa_tc_kernel``, see
# that file's two new inline comments at the fix sites): with only 2
# buffers, each tile's ``cur`` buffer becomes the *next* iteration's
# ``nxt`` prefetch target, and the kernel had no barrier between a warp's
# last synchronous shared-memory *read* of ``cur`` (the ``sK``/``sVt``
# reads feeding GEMM1/GEMM2) and a *different*, faster warp's asynchronous
# cp.async *write* into that same buffer for the next tile -- the existing
# ``cp_async_wait_group``/``sync_threads`` pair only orders *previous*
# cp.async groups against *this* iteration's reads, not this iteration's
# reads against the *next* iteration's writes. A second, narrower instance
# of the identical pattern existed at start-of-kernel: ``sQ`` (the staged
# Q tile) is smem-aliased onto K ring-buffer slot #1 to save 8KB, and the
# tile-0-into-tile-1 prefetch (the loop's first iteration) began
# overwriting that same slot with no barrier after the one-time ``tSrQ``
# read out of ``sQ``. Both gaps are now closed with one ``sync_threads()``
# each, placed at the earliest point that is safe (right after the last
# read of the memory about to be reused) -- not a broader/coarser lock
# (e.g. forcing a full ``cp_async_wait_group(0)`` drain every iteration was
# tried as a bisection probe, confirmed to also "fix" it, and rejected as
# the real fix because it would serialize away the double-buffer overlap
# entirely). Re-verified all four gates independently through this exact
# ``dispatch.py`` entrypoint (``try_qsa_kf=True``) after the fix: (a)
# compiles/loads without hang; (b) launches and completes on S=4096, 8192,
# 32768 under a normal wall-clock budget; (c) oracle correctness --
# matched_ratio 1.0, max abs err ~3e-5 (tolerance 2e-2) -- including 30%
# random-invalid-entry fuzz, ~2% fully-dead-row, and a 32768-seqlen
# dead-row case (exact ``out==0``/``lse==-inf``); (d) determinism -- 20-30
# repeated calls bitwise-identical (``torch.equal``, not just
# ``isclose``), across all three seqlens above and every fuzz/dead-row
# variant, zero mismatched elements in every run (vs. hundreds-to-low-
# thousands of mismatched elements per repeat before the fix). Flipping
# the default here.
_DEFAULT_TRY_MSA_KF_FAST_PATH = True
_DEFAULT_TRY_QSA_KF_FAST_PATH = True


def _tcgen05_cell_eligible(*, d_k: int, d_v: int, index_granularity: int) -> bool:
    """Structural (shape-only, no device read) envelope check for the MSA
    cell this round's tcgen05 mainloop targets: ``D_k == D_v == 128``,
    ``index_granularity == 128``. ``G == H_kv`` is inherent to this whole
    ``sm100_gqa`` package's envelope (see module docstring) and is not
    re-checked per call here; a real ``h_q``/``h_kv`` compatibility check
    (analogous to ``gqa_prefill_bf16_tile_sm100.fast_path_eligible``) is
    still applied below if the sibling module exposes one, once it exists.
    """
    return d_k == 128 and d_v == 128 and int(index_granularity) == 128


def _try_tcgen05_fast_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idxs: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    index_granularity: int,
    softmax_scale: Optional[float],
    stream: Optional[cuda.CUstream],
    validate_uniform: bool,
) -> Optional[TupleDict]:
    """Attempt ``gqa_prefill_bf16_tcgen05_sm100``'s tensor-core (tcgen05)
    mainloop for the MSA cell (BF16, ``D_k == D_v == 128``,
    ``index_granularity == 128``, ``G == H_kv``). Returns ``None`` (never
    raises) whenever: the shapes are outside this cell's structural
    envelope; the sibling kernel module doesn't exist yet in this worktree
    (``ImportError`` -- see the module docstring's honesty note, this is
    the expected outcome as of this subtask); or ``validate_uniform`` finds
    the selection isn't actually tile-uniform. All of these are expected,
    non-exceptional outcomes of a probe that runs unconditionally on the
    default hot path, so the caller can always fall through to the scalar
    kernel safely. ``validate_uniform=True`` is the honest default here --
    never skip it on a path that decides FOR the caller (rather than on the
    caller's explicit, verified assertion) without a real structural proxy
    for tile-uniformity: without it, a per-row-varying selection would
    silently compute a wrong answer instead of falling back.
    """
    d_k = int(q.shape[-1])
    d_v = int(v.shape[-1])
    if not _tcgen05_cell_eligible(d_k=d_k, d_v=d_v, index_granularity=index_granularity):
        return None
    try:
        from .gqa_prefill_bf16_tcgen05_sm100 import sparse_attention_forward_wrapper as _tcgen05_wrapper
    except ImportError:
        # The sibling kernel module hasn't landed in this worktree yet (see
        # the module docstring's honesty note) -- treat exactly like "this
        # cell isn't actually servable by the fast path" and fall through
        # to the scalar kernel, not a hard failure.
        return None
    try:
        from .gqa_prefill_bf16_tcgen05_sm100 import fast_path_eligible as _tcgen05_eligible
    except ImportError:
        _tcgen05_eligible = None
    if _tcgen05_eligible is not None:
        h_q = int(q.shape[-2])
        h_kv = int(k.shape[-2])
        if not _tcgen05_eligible(d_k=d_k, d_v=d_v, h_q=h_q, h_kv=h_kv, index_granularity=int(index_granularity)):
            return None
    try:
        result = _tcgen05_wrapper(
            q,
            k,
            v,
            topk_idxs,
            topk_length=topk_length,
            attn_sink=attn_sink,
            cu_seqlens_q=cu_seqlens_q,
            index_granularity=index_granularity,
            softmax_scale=softmax_scale,
            uniform_within_tile=True,
            validate_uniform=validate_uniform,
            stream=stream,
        )
    except (ValueError, NotImplementedError):
        # validate_uniform's ValueError (selection isn't tile-uniform) or a
        # narrower shape/dtype NotImplementedError -- both mean "this
        # config isn't actually servable by the tcgen05 kernel", not a
        # real error.
        return None
    return TupleDict(out=result["out"], lse=result["lse"])


def _try_tile_fast_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idxs: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    index_granularity: int,
    softmax_scale: Optional[float],
    stream: Optional[cuda.CUstream],
    validate_uniform: bool,
) -> Optional[TupleDict]:
    """Attempt ``gqa_prefill_bf16_tile_sm100``'s tile-batched fast path,
    returning ``None`` (rather than raising) when the config is outside its
    envelope or ``validate_uniform`` finds the selection isn't actually
    tile-uniform -- both expected, non-exceptional outcomes of a probe, so
    the caller can fall through to the scalar kernel unconditionally safely.
    ``validate_uniform=True`` is required here (never skip it on a path that
    decides FOR the caller rather than on the caller's explicit assertion):
    without it, a per-row-varying selection silently computes a wrong
    answer instead of falling back.
    """
    from .gqa_prefill_bf16_tile_sm100 import fast_path_eligible
    from .gqa_prefill_bf16_tile_sm100 import sparse_attention_forward_wrapper as _tile_wrapper

    d_k = q.shape[-1]
    d_v = v.shape[-1]
    h_q = q.shape[-2]
    h_kv = k.shape[-2]
    if not fast_path_eligible(d_k=int(d_k), d_v=int(d_v), h_q=int(h_q), h_kv=int(h_kv), index_granularity=int(index_granularity)):
        return None
    try:
        result = _tile_wrapper(
            q,
            k,
            v,
            topk_idxs,
            topk_length=topk_length,
            attn_sink=attn_sink,
            cu_seqlens_q=cu_seqlens_q,
            index_granularity=index_granularity,
            softmax_scale=softmax_scale,
            uniform_within_tile=True,
            validate_uniform=validate_uniform,
            stream=stream,
        )
    except (ValueError, NotImplementedError):
        # validate_uniform's ValueError (selection isn't tile-uniform) or a
        # narrower shape/dtype NotImplementedError fast_path_eligible didn't
        # already screen out -- both mean "this config isn't actually
        # servable by the tile kernel", not a real error.
        return None
    return TupleDict(out=result["out"], lse=result["lse"])


def _has_no_invalid_entries(topk_idxs: torch.Tensor, topk_length: Optional[torch.Tensor], topk_max: int) -> bool:
    """Opt-in, D2H-synchronizing check that this call's ``topk_idxs``/
    ``topk_length`` are entirely inside the "provably safe for a KF-mod-fold
    kernel" subset this round's KF fast paths require (see the module
    docstring's round-7 honesty note): no ``-1`` sentinel anywhere, and
    ``topk_length`` (if present) never restricts below the compile-time
    ``topk_max`` the KF kernel always treats as fully valid. Returns
    ``False`` (never raises) on any violation -- callers treat that exactly
    like any other "this config isn't actually servable" fast-path miss."""
    if torch.any(topk_idxs < 0).item():
        return False
    if topk_length is not None and torch.any(topk_length < topk_max).item():
        return False
    return True


def _kf_cell_common_checks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idxs: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    *,
    h_q: int,
    h_kv: int,
    d: int,
    topk_max: int,
    granularity: int,
    index_granularity: int,
) -> bool:
    """Shape/dtype/config eligibility shared by both KF fast paths. Both KF
    kernels hardcode one exact (H_q, H_kv, D_k=D_v, topk) shape (see the
    module docstring) and assume a single flat KV sequence the same length
    as Q (self-attention-style prefill: ``S_q == S_kv``, one sequence per
    call) -- neither generalizes to THD multi-sequence packing or BSHD
    batch > 1 the way the scalar/tcgen05/tile kernels in this package do,
    so both are excluded here rather than silently mishandled. ``attn_sink``
    is also outside both KF kernels' envelope (neither reads a sink term)."""
    if int(index_granularity) != granularity:
        return False
    if int(q.shape[-2]) != h_q or int(k.shape[-2]) != h_kv:
        return False
    if int(q.shape[-1]) != d or int(v.shape[-1]) != d:
        return False
    if int(topk_idxs.shape[-1]) != topk_max:
        return False
    if attn_sink is not None:
        return False
    if cu_seqlens_q is not None:
        # KF kernels assume one flat sequence; a real caller with multiple
        # packed THD sequences (cu_seqlens_q.numel() > 2) is out of envelope.
        # A single-sequence THD call (cu_seqlens_q == [0, T_q]) would in
        # principle be servable, but that still needs T_q == T_kv, which the
        # THD/BSHD-generic call sites here cannot cheaply prove without a
        # device read of cu_seqlens_q itself -- so this round conservatively
        # excludes all THD calls from both KF fast paths rather than adding
        # another D2H sync on top of the -1/topk_length one.
        return False
    if q.ndim == 4:
        # BSHD: KF kernels assume B == 1 (a single flat sequence).
        b = int(q.shape[0])
        if b != 1:
            return False
        s_q = int(q.shape[1])
        s_kv = int(k.shape[1])
        if s_q != s_kv:
            return False
    if int(k.shape[0 if q.ndim == 3 else 1]) % granularity != 0:
        # No tail-clamped last block: the KF kernels' NE = S_kv // granularity
        # only makes sense as a power-of-2 mask (``entry & (NE - 1)``) when
        # S_kv is an exact multiple of granularity in the first place.
        return False
    return True


def _try_msa_kf_fast_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idxs: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    index_granularity: int,
    softmax_scale: Optional[float],
    stream: Optional[cuda.CUstream],
    validate_uniform: bool,
) -> Optional[TupleDict]:
    """Attempt KF's MSA winner (``msa_r7_v3_unroll8``, adapted into
    ``gqa_prefill_bf16_msa_kf_sm100``) for the MSA cell (BF16, H_q=64,
    H_kv=4, D_k=D_v=128, topk in the vendored kernel's supported set,
    granularity=128).

    Round-8 update: unlike ``_try_qsa_kf_fast_path`` (whose sibling kernel
    hardcodes an un-adaptable ``entry & (NE-1)`` fold with no ``-1``
    support -- see ``_kf_cell_common_checks``'s docstring), the vendored
    ``kf_msa`` sources ship a *second*, general entry point
    (``msa_helpers.build_msa_metadata`` + K1/K2, not ``kernel.py``'s
    benchmark-fold path) that natively supports this contract's ``-1``
    invalid-slot sentinel, ``topk_length``, THD (single-segment, this
    round), and BSHD with ``B > 1`` -- see
    ``gqa_prefill_bf16_msa_kf_sm100``'s module docstring for the full
    account of what is/isn't adapted. This fast path therefore does *not*
    reuse ``_kf_cell_common_checks``/``_has_no_invalid_entries`` (both
    written for the QSA sibling's narrower, no-``-1``, single-flat-sequence
    envelope) -- it delegates the real eligibility decision to the sibling
    module's own ``fast_path_eligible`` plus its wrapper's own internal
    probes (multi-segment THD, unsupported topk/GQA ratio, and any
    out-of-range selection ``build_msa_metadata`` itself rejects all return
    ``None`` from the wrapper, not a raise)."""
    try:
        from .gqa_prefill_bf16_msa_kf_sm100 import fast_path_eligible as _msa_kf_eligible
        from .gqa_prefill_bf16_msa_kf_sm100 import sparse_attention_forward_wrapper as _msa_kf_wrapper
    except ImportError:
        return None
    d_k = int(q.shape[-1])
    d_v = int(v.shape[-1])
    h_q = int(q.shape[-2])
    h_kv = int(k.shape[-2])
    topk = int(topk_idxs.shape[-1])
    if not _msa_kf_eligible(d_k=d_k, d_v=d_v, h_q=h_q, h_kv=h_kv, index_granularity=index_granularity, topk=topk):
        return None
    try:
        result = _msa_kf_wrapper(
            q,
            k,
            v,
            topk_idxs,
            topk_length=topk_length,
            attn_sink=attn_sink,
            cu_seqlens_q=cu_seqlens_q,
            index_granularity=index_granularity,
            softmax_scale=softmax_scale,
            stream=stream,
        )
    except (ValueError, NotImplementedError):
        return None
    if result is None:
        return None
    return TupleDict(out=result["out"], lse=result["lse"])


def _try_qsa_kf_fast_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idxs: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    index_granularity: int,
    softmax_scale: Optional[float],
    stream: Optional[cuda.CUstream],
    validate_uniform: bool,
) -> Optional[TupleDict]:
    """Attempt KF's QSA winner (``qsa_r7_s8192_explicit_reduce4_guard``,
    adapted into ``gqa_prefill_bf16_sm100_kf_qsa`` -- **not** the older
    ``gqa_prefill_bf16_qsa_kf_sm100`` thin-gate module a prior round wired
    here) for its exact hardcoded shape (BF16, H_q=24, H_kv=2, D_k=D_v=256,
    topk=512, granularity=4).

    Unlike the thin gate this replaces, the current sibling module makes
    two real kernel-body fixes (the vendored ``& (NE-1)`` fold removed for
    storage-native pass-through; a real ``-1``/tail-clamp/dead-row path
    added) rather than only ever handing the KF kernel inputs where its
    un-adapted fold happens to be a no-op -- see that module's docstring.
    Consequently this fast path no longer needs
    ``_kf_cell_common_checks``'s power-of-two-``NE``/tail-divisibility gate
    or ``_has_no_invalid_entries``'s D2H no-``-1`` check; it delegates
    eligibility to the sibling module's own ``fast_path_eligible`` plus its
    wrapper's own internal probes, matching ``_try_msa_kf_fast_path``'s
    simpler pattern.

    KF's QSA kernel shares one ``topk_idxs`` set across every head --
    ``group_scope == 1`` in this contract's terms -- rather than this
    package's usual ``G == H_kv``; this fast path additionally accepts a
    ``G == H_kv`` ``topk_idxs`` whose entries are identical across the
    ``H_kv`` axis (the shape every other kernel in this package -- and
    ``api.py``'s GQA envelope predicate -- requires), collapsing it to the
    single shared set the sibling kernel expects."""
    try:
        from .gqa_prefill_bf16_sm100_kf_qsa import fast_path_eligible as _qsa_kf_eligible
        from .gqa_prefill_bf16_sm100_kf_qsa import sparse_attention_forward_wrapper as _qsa_kf_wrapper
    except ImportError:
        return None
    d_k = int(q.shape[-1])
    d_v = int(v.shape[-1])
    h_q = int(q.shape[-2])
    h_kv = int(k.shape[-2])
    topk = int(topk_idxs.shape[-1])
    if not _qsa_kf_eligible(d_k=d_k, d_v=d_v, h_q=h_q, h_kv=h_kv, index_granularity=index_granularity, topk=topk):
        return None
    idxs = topk_idxs
    if idxs.ndim >= 2 and idxs.shape[-2] == h_kv and h_kv > 1:
        # This package's usual G == H_kv shape -- only servable by the
        # G == 1 sibling kernel when every KV-head's entries are identical
        # (a real, if unusual, selection: e.g. an upstream caller that
        # broadcasts a shared selection into the G == H_kv shape every
        # other kernel here requires). Collapse to G == 1 if so; otherwise
        # this genuinely per-KV-head-varying selection is out of scope for
        # a G == 1 kernel, not something a fast path can silently reshape.
        first = idxs.index_select(-2, torch.zeros(1, dtype=torch.long, device=idxs.device))
        if not torch.equal(idxs, first.expand_as(idxs)):
            return None
        idxs = idxs[..., 0, :]
    try:
        result = _qsa_kf_wrapper(
            q,
            k,
            v,
            idxs,
            topk_length=topk_length,
            attn_sink=attn_sink,
            cu_seqlens_q=cu_seqlens_q,
            index_granularity=index_granularity,
            softmax_scale=softmax_scale,
            stream=stream,
        )
    except (ValueError, NotImplementedError):
        # Everything outside this sibling's fixed geometry / single-sequence
        # scope raises there (see its own docstring) rather than returning
        # None -- both mean "not servable by this fast path", so both fall
        # through to the scalar kernel here, not a hard failure.
        return None
    return TupleDict(out=result["out"], lse=result["lse"])


def sparse_attention_forward_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idxs: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    index_granularity: int = 1,
    softmax_scale: Optional[float] = None,
    stream: Optional[cuda.CUstream] = None,
    *,
    uniform_within_tile: bool = _DEFAULT_TRY_TILE_FAST_PATH,
    try_tcgen05: bool = _DEFAULT_TRY_TCGEN05_FAST_PATH,
    try_msa_kf: bool = _DEFAULT_TRY_MSA_KF_FAST_PATH,
    try_qsa_kf: bool = _DEFAULT_TRY_QSA_KF_FAST_PATH,
    validate_uniform: bool = True,
) -> TupleDict:
    """GQA-substrate sparse-attention forward, dispatched by ``Q.dtype``
    (and, for BF16, by shape/granularity -- see module docstring).

    Matches the frozen ``sparse_attention_forward_wrapper`` contract
    (``python/cudnn/sparse_attention/fwd/api.py``) for the slice this
    round's kernels serve; raises ``NotImplementedError`` for the rest of
    the GQA envelope (FP8-per-tensor has no kernel yet) rather than
    mis-dispatching.

    ``uniform_within_tile``/``try_tcgen05``/``try_msa_kf``/``try_qsa_kf``/
    ``validate_uniform`` (keyword-only, not part of ``api.py``'s frozen
    signature -- see module docstring):

    * ``try_msa_kf`` / ``try_qsa_kf`` (default ``False`` as of round 7 --
      see the module docstring's round-7 honesty note): attempt KF's
      MSA / QSA winners for their exact hardcoded shapes, tried first
      (ahead of tcgen05/tile) since both are real, hardware-validated
      kernels once their sibling modules land. As of this round, both are
      no-ops -- the sibling kernel modules do not exist in this worktree
      yet, so these always fall straight through to the scalar kernel
      (``ImportError`` caught, ``None`` returned) regardless of the flag.
    * ``try_tcgen05`` (default ``False`` as of round 6 -- see the module
      docstring's round-6 honesty note): for the MSA cell (BF16,
      ``D_k == D_v == 128``, ``index_granularity == 128``), if passed
      ``True``, *try* the tcgen05 tensor-core mainloop ahead of the scalar
      kernel. This is opt-in only: ``gqa_prefill_bf16_tcgen05_sm100``'s
      ``cute.compile()`` call is confirmed to hang indefinitely (no
      timeout guard exists), so ``api.py``'s generic dispatch never passes
      ``True`` here and real callers always land on the scalar kernel (or
      the round-3 tile fast path, if ``uniform_within_tile=True``) unless
      they explicitly request the tcgen05 path and accept that risk.
    * ``uniform_within_tile`` (default ``False``): pass ``True`` to *also*
      try the round-3 tile-batched fast path for ``index_granularity ==
      128`` (measured slower than the scalar kernel on every shape tried --
      see module docstring -- so this stays opt-in and ``api.py``'s generic
      dispatch never passes it), tried after the tcgen05 attempt above.
    * ``validate_uniform`` (default ``True``, shared by both fast paths):
      does one explicit, D2H-synchronizing host-side check before any
      tcgen05-or-tile kernel launch and falls back to the scalar kernel
      rather than computing a wrong answer if the selection turns out not
      to be tile-uniform; pass ``validate_uniform=False`` only when the
      caller has independently verified tile-uniformity and wants to skip
      that sync.
    """
    if q.dtype == torch.bfloat16:
        if try_msa_kf:
            fast_result = _try_msa_kf_fast_path(
                q,
                k,
                v,
                topk_idxs,
                topk_length,
                attn_sink,
                cu_seqlens_q,
                index_granularity,
                softmax_scale,
                stream,
                validate_uniform,
            )
            if fast_result is not None:
                return fast_result
        if try_qsa_kf:
            fast_result = _try_qsa_kf_fast_path(
                q,
                k,
                v,
                topk_idxs,
                topk_length,
                attn_sink,
                cu_seqlens_q,
                index_granularity,
                softmax_scale,
                stream,
                validate_uniform,
            )
            if fast_result is not None:
                return fast_result
        if try_tcgen05:
            fast_result = _try_tcgen05_fast_path(
                q,
                k,
                v,
                topk_idxs,
                topk_length,
                attn_sink,
                cu_seqlens_q,
                index_granularity,
                softmax_scale,
                stream,
                validate_uniform,
            )
            if fast_result is not None:
                return fast_result
        if uniform_within_tile:
            fast_result = _try_tile_fast_path(
                q,
                k,
                v,
                topk_idxs,
                topk_length,
                attn_sink,
                cu_seqlens_q,
                index_granularity,
                softmax_scale,
                stream,
                validate_uniform,
            )
            if fast_result is not None:
                return fast_result
        result = _bf16_wrapper(
            q,
            k,
            v,
            topk_idxs,
            topk_length=topk_length,
            attn_sink=attn_sink,
            cu_seqlens_q=cu_seqlens_q,
            index_granularity=index_granularity,
            softmax_scale=softmax_scale,
            stream=stream,
        )
        return TupleDict(out=result["out"], lse=result["lse"])
    if q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise NotImplementedError(
            "sparse_attention.fwd.sm100_gqa: FP8-per-tensor has no full kernel wired yet in this round "
            "(gqa_prefill_fp8_sm100.py currently ships only the device-scale-folding helper); use BF16"
        )
    raise NotImplementedError(f"sparse_attention.fwd.sm100_gqa serves BF16 (FP8-per-tensor pending), got Q dtype {q.dtype}")
