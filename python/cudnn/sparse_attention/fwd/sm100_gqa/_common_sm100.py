# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared config + device helpers for the SM100 GQA-substrate sparse-attention
forward kernel (``gqa_prefill_bf16_sm100.py``).

PR4 envelope: ``G = H_kv`` (one ``topk_idxs`` row per KV-head group, shared by
every Q head in that group), ``index_granularity in (4, 64, 128)`` (QSA / MSA
block shapes), separate (non-aliased) K/V, BF16, SM100/GB300/GR100-class
Blackwell only.

Style note (round-1 scope, read before extending): this module intentionally
does **not** build on ``cudnn.frost.tile_dsl.{mma,tma,tmem,handles}`` (the
tcgen05-tile-MMA / TMA-tensor-map / TMEM machinery ``prefill_d128_f16_sm100``
uses) the way the task's reference style points at. That machinery batches
the M (query-row) dimension of a `tcgen05.mma` call over ONE SHARED K/V smem
tile per step -- exactly what the block-sparse (mask-driven) BSA kernel
(``block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_sm100.py``) and the
existing pipeline-style DSA kernels do, because in a *mask*-driven or
*block-uniform* scheme every M-row shares the same KV-block set for a whole
Q tile. The frozen contract here is explicitly **per-query-token**
(``topk_idxs`` is ``(T_q, G, topk)`` -- row ``t``'s selection is independent
of row ``t+1``'s), which breaks that batching assumption in general: two
rows in the same would-be M-tile can select entirely different KV blocks, so
there is no single K/V smem tile a tcgen05 MMA step could share across them
without either (a) a per-Q-tile-uniform-selection precondition the generic
wrapper does not (and per the frozen contract cannot) guarantee, or (b) a
gather-then-dense-recompute over the *union* of a tile's rows' selections,
masking each row's non-selected union members -- both real designs, neither
implementable correctly inside this round's budget without hardware-in-the-
loop descriptor debugging.

Round-1 lands the mainloop at **warp granularity instead: one warp per
(query row, KV-head group, batch)**, using ``cute.Tensor`` /
``cutlass.make_array_view`` global-memory reads (the same primitive
``split_combine_sm100.py`` -- also part of the SM100 DSL prefill family --
uses for its reduction pass) plus ``cudnn.frost.tile_dsl.pointwise`` for the
lane-parallel dot-product reduction and ``cudnn.frost.tile_dsl.regtile`` for
the per-lane O accumulator. This is real, house-style, per-token-general and
numerically exact; the tradeoff is no TMA/tensor-core throughput. Migrating
to a tcgen05-tile-MMA mainloop (option (b) above, restricted to callers that
can promise per-Q-tile-uniform selection, e.g. NSA/MoBA-style block
attention) is the natural round-2 follow-up and is exactly the discrepancy
the task asked to surface rather than silently resolve.

Round-1 MMA-port research note: ``gqa_prefill_bf16_tile_sm100.py``'s module
docstring (item 3) has the concrete finding for *why* a tcgen05-MMA mainloop
still isn't landed here -- ``mma_ss``/``mma_ts_step`` (``cudnn.frost.tile_dsl
.mma``) are the right, already-proven-in-repo primitive (``prefill_d128_f16
_sm100.py`` uses them and passes its suite on this same SM100 box), not
``cute.gemm()`` (neither that kernel nor KF's MSA winner call it); the
remaining blocker is swizzled-smem plumbing (K-major bf16 d_k=128 forces
``K_SW128``, which this package's ``cp.async``/``load_tile_2d`` gather does
not produce) needing a runtime-coordinate TMA load, scoped but not
implemented this round.

Config note (round-3): an earlier ``config_sm100.py`` sibling sketched an
aspirational ``TemplateParams``/``Cfg``/``MAKE_CFG`` config module (mirroring
``cudnn.sdpa.fwd.config_sm100``) for a *future* tcgen05-tile-MMA mainloop
kernel file layout (``kernels/gqa_g4_*.py`` / ``gqa_g64_*.py`` /
``gqa_g128_*.py``). That module was never imported by ``dispatch.py``,
``gqa_prefill_bf16_sm100.py``/``gqa_prefill_fp8_sm100.py``, or this module --
dead code. Since this round's mainloop is still the warp-per-row
scalar-gather kernel described above (the tcgen05/TMA migration remains an
un-landed follow-up, not something this round's dispatch table routes to),
there is no compile-time-config-per-geometry consumer for it yet, so it was
deleted rather than wired in. ``GqaPrefillConfig`` below (a plain frozen
dataclass of runtime-orthogonal shape/dtype constants captured by kernel
closures) remains the single config definition for this package. If/when a
real tcgen05-tile-MMA mainloop lands for this envelope, resurrect the
``TemplateParams``/``Cfg`` pattern at that point (from version control
history) rather than reintroducing it speculatively ahead of a consumer.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.experimental import primitives as nvvm

from cudnn.frost.tile_dsl.pointwise import lane_group_sum  # noqa: F401  (re-exported for kernel callers)

WARP_LANES = 32


@dataclass(frozen=True)
class GqaPrefillConfig:
    """Compile-time shape/dtype constants for one traced kernel instance.

    ``d_k``/``d_v`` and the head counts are baked in (they gate register
    counts and unroll factors); everything row/entry-count-shaped
    (``T_q``, ``topk_max``, ``S_kv``/``T_kv``) stays a runtime CuTe tensor
    extent so one compiled kernel serves every problem size at a given
    (d_k, d_v, h_q, h_kv, granularity) point.
    """

    d_k: int
    d_v: int
    h_q: int
    h_kv: int
    granularity: int
    is_bshd: bool
    has_topk_length: bool
    has_attn_sink: bool

    def __post_init__(self):
        assert self.granularity in (4, 64, 128), f"GqaPrefillConfig targets PR4 block granularities, got {self.granularity}"
        assert self.h_kv > 1, f"GqaPrefillConfig (PR4, G=H_kv) requires H_kv > 1, got {self.h_kv}"
        assert self.h_q % self.h_kv == 0, f"H_q ({self.h_q}) must be a multiple of H_kv ({self.h_kv})"

    @property
    def heads_per_kv(self) -> int:
        return self.h_q // self.h_kv

    @property
    def v_chunks_per_lane(self) -> int:
        """``ceil(d_v / WARP_LANES)`` -- how many strided elements each lane
        of the O accumulator owns."""
        return (self.d_v + WARP_LANES - 1) // WARP_LANES


@cute.jit
def lane_group_max(value: cutlass.Float32, lanes: cutlass.Constexpr[int] = WARP_LANES) -> cutlass.Float32:
    """Max of ``value`` across a power-of-two group of consecutive lanes via
    butterfly shuffles (every lane ends up holding the group max).

    Sibling of :func:`cudnn.frost.tile_dsl.pointwise.lane_group_sum` (that
    module only ships the sum variant); kept here rather than upstreamed
    into ``pointwise.py`` for this round since it is only needed by this
    kernel family so far.
    """
    offset = lanes // 2
    while offset >= 1:
        other = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, value, offset, 31, kind=nvvm.Shfl.BFLY))
        value = cute.math.max(value, other, ftz=True)
        offset = offset // 2
    return value


@cute.jit
def resolve_entry_window(entry_idx: cutlass.Int32, granularity: cutlass.Constexpr[int], kv_bound: cutlass.Int32):
    """Expand one ``topk_idxs`` entry into its ``[start, start + width)``
    token window inside ``[0, kv_bound)``.

    ``entry_idx`` is the raw storage-native entry value (may be ``-1``).
    ``kv_bound`` is the row's true KV length (``T_kv`` for THD -- shared by
    every row since this contract has no ragged KV split for THD -- or
    ``S_kv`` for BSHD). Mirrors the frozen contract's tail-clamp exactly:
    entry ``i``'s window is ``[i*g, i*g+g)`` intersected with
    ``[0, kv_bound)``.

    Returns ``(tile_start, num_valid, is_valid)``:

    * ``tile_start`` -- offset in ``[0, kv_bound)`` (``0``, a safe in-bounds
      dummy, for an invalid entry -- so downstream address math never
      underflows even though the caller must still gate on ``is_valid``).
    * ``num_valid`` -- tokens actually covered after the tail clamp, in
      ``[0, g]``; ``0`` for an invalid entry or one whose window starts at
      or past ``kv_bound``.
    * ``is_valid`` -- whether this entry contributes at all.
    """
    g = cutlass.Int32(granularity)
    clamped_idx = cute.math.max(entry_idx, cutlass.Int32(0))
    tile_start = clamped_idx * g
    tile_end = tile_start + g
    clamped_end = cute.math.min(tile_end, kv_bound)
    num_valid = cute.math.max(clamped_end - tile_start, cutlass.Int32(0))

    entry_present = entry_idx >= cutlass.Int32(0)
    starts_in_bound = tile_start < kv_bound
    is_valid = entry_present & starts_in_bound

    # entry_present alone (not is_valid) gates num_valid here: a negative
    # entry_idx was already routed to the safe dummy window at tile_start=0,
    # whose num_valid_raw the tail clamp above may compute as nonzero (it
    # only sees kv_bound, not entry_idx) -- so a negative entry must still be
    # forced to zero explicitly.
    num_valid = cutlass.Int32(arith.select(entry_present.ir_value(), num_valid.ir_value(), cutlass.Int32(0).ir_value()))
    return tile_start, num_valid, is_valid
