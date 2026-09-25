# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stages (2)+(3) of the gated attention block, TMA-staged.

An A/B against ``qk_norm_rope.py``, not a replacement. Same math, same
bit-identical result; a different pipeline. Keep both until one wins on the
perf node, then delete the loser.

**Why this exists, from the measurement rather than from taste.** The LDG
kernel was register-bound, and ``defer_secondary_loads`` fixed that: ncu says
registers/thread 95 -> 64, occupancy limited by registers 5 -> 8 blocks against
a warp limit of 8, achieved occupancy 55% -> 87%. Registers stopped being the
binding constraint, and the WARP limit took over: ~28 warps/SM x 2 rows x 512 B
= ~28 KiB in flight per SM, against the ~35 KiB Little's law wants to saturate
10.8 TB/s at ~700 ns. It cannot buy more, because more rows per thread costs
the registers back and more warps per SM are not available.

**A staged SMEM ring is the one lever that adds in-flight bytes without either.**
``STAGES x TILE_ROWS x 512 B`` per CTA lives in SMEM, not registers, so the
occupancy stays where the defer fix put it.

Three structural wins fall out, and they are the reason to prefer this shape
even at parity:

* **``is_q`` becomes ONE warp-uniform decision per TILE.** A TMA tile is either
  all-Q or all-K by construction, so the per-row branch that picked between two
  base pointers and two strides is gone from the inner loop entirely.
* **The token stride lives in the DESCRIPTOR, not in kernel arithmetic.** That
  is what made the symbolic-stride artifact cost 25% against the compact one:
  the address math, not the traffic. A descriptor is built once on the host and
  costs the kernel nothing, so the strided source is free here.
* **The stores are TILE stores.** The old epilogue's ``rstd`` write was one
  lane emitting 4 scattered bytes per row, which is a 32-byte sector per row;
  here the tile's ``TILE_ROWS`` values are gathered in SMEM and leave as one
  coalesced store, and Q/K leave through TMA.

**Not swizzled, deliberately, and the arithmetic is in the SMEM buffer table
below.** Warp-per-row means 16 lanes cover a 256-byte contiguous half-row, so
the access already spreads all 32 banks: 4 wavefronts per instruction, which is
the ideal for 32 lanes x 16 B. TMA's 128 B swizzle would not improve it, and
the thread-per-row alternative it is designed against would be 4x worse (16
wavefronts) because a 512-byte row stride is a multiple of the 128-byte bank
cycle.
"""

from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.frost.device import current_device, multiprocessor_count
from cudnn.frost.tile_dsl.barrier import PipelineState, advance, arrive_expect_tx, wait
from cudnn.frost.tile_dsl.handles import GmemTileTma, SmemTile
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16, lane_group_sum
from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global, tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait

from .qk_norm_rope import check_norm_weights_match_recipe

ELEMS_PER_ACCESS = 8  # bf16 elements in one 16-byte access
WORDS_PER_ACCESS = 4  # 32-bit words in one 16-byte access
BPE = 2
TMA_GRANU_ELEMS = 128  # 256 B per row: the UNSWIZZLED TMA inner-box cap
WARP = 32

DEFAULT_TILE_ROWS = 16
DEFAULT_STAGES = 2
DEFAULT_STAGES_O = 1
"""Output ring depth, INDEPENDENT of the input ring.

They are different resources -- input depth buys load-ahead, output depth buys
store slack -- and SMEM spent on one is SMEM not spent on the other. Measured
cold on the Rubin dev node, S=32768, 16-row tiles, L2-flushed against 4453 GB/s;
each buffer is 8 KiB, so SMEM per CTA is ``(stages + stages_o) * 8 KiB``:

    stages  stages_o  buffers  KiB/CTA    GB/s   % ceiling
      2         1        3        24      4437      99.6    <- default
      2         2        4        32      4439      99.7
      3         1        4        32      4416      99.2
      3         2        5        40      3593      80.7
      4         1        5        40      3533      79.3
      4         2        6        48      3520      79.1

**What decides it is the TOTAL buffer count, not which ring owns them.** Equal
footprints tie to within 0.5%; on the dev node the cliff sits between 4 and 5
buffers, where the resident CTA count steps down. So there is nothing to gain by
deepening EITHER ring, and the whole ``stages_o`` sweep at fixed ``stages`` is
flat: 1, 2, 3 and 4 all land within 0.4% at S=8192 and at S=32768 alike.

**The CLIFF is node-dependent; the flatness is not.** Re-run on the perf node at
S=32768, all four of (stages, stages_o) in {2,3} x {1,2} land within 0.5%
(8310 / 8302 / 8291 / 8273 GB/s) -- no cliff at 5 buffers there at all. So do not
port the cliff position between parts. What survives both is that deepening buys
nothing, which is why the default is the SMALLEST configuration: never worse,
and 25% less SMEM than the coupled one.

The useful consequence runs the other way. ``stages_o=1`` is exactly as fast as
4 and uses **25% less SMEM**, so it is the default: same throughput, 8 KiB per
CTA back. It costs a full ``tma_store_wait(0)`` drain each tile, which measures
free here because the TMA store is never the bottleneck -- if a part ever shows
one, ``stages_o=2`` is the flip and it is equally fast."""
DEFAULT_THREADS = 128
REFILL_LATE = 0
REFILL_AFTER_COMPUTE = 1
REFILL_TOP = 2
DEFAULT_REFILL_POS = REFILL_AFTER_COMPUTE
"""Where the NEXT tile's TMA load is issued.

* ``REFILL_LATE`` -- after the store issue and the rstd write.
* ``REFILL_AFTER_COMPUTE`` -- immediately after the post-compute barrier, so the
  load overlaps this tile's epilogue as well.
* ``REFILL_TOP`` -- at the top of the loop, refilling the stage the PREVIOUS
  iteration finished with.

**``REFILL_AFTER_COMPUTE`` is the earliest CORRECT point.** A stage's input
buffer is provably free only once every warp has finished reading it, which is
exactly what the post-compute barrier establishes. Issuing at the top of the loop
for the CURRENT stage would overwrite the data this iteration is about to read,
which is why ``REFILL_TOP`` refills one stage behind.

**And it does not matter.** All three tie, measured cold on the Rubin dev node,
16-row tiles, ``stages_o=1``, L2-flushed against 4453 GB/s:

    S       refill=0   refill=1   refill=2
    4096      3162       3155       3162
    8192      3822       3830       3839
    32768     4511       4515       4528

That is a 0.6% spread at S=32768 where the kernel is already at 101% of a
measured device-to-device copy, and 0.2% at S=4096 where it is at 71%. Load
placement is simply not what this pipeline is short of. The default stays at the
earliest correct point because it is the least surprising, not because it won.

Getting the load out *before the math* -- which is the thing worth wanting --
needs the row loop split into a read phase and a math phase with a barrier
between, so the refill can go out while the math runs. That holds every row's
operands in registers across the barrier (4 rows x 8 fp32 at the default tile),
i.e. it spends the exact resource the LDG kernel had to stop spending. Untested;
worth trying only on a part where this kernel is not already at the ceiling."""
DEFAULT_FUSED_STORE_WAIT = False
"""Two loop-ordering choices, both measured; the FUSED drain is legal only at ``stages_o >= 2``.

``fused_store_wait`` moves the store drain onto the barrier the epilogue already
runs, deleting a whole CTA barrier from the per-tile critical path.  It shipped
as the default at ``stages_o=1`` and that combination is a WRITE-AFTER-READ RACE
on ``sOut`` (found 2026-09-15): the fused ``tma_store_wait(stages_o - 1)`` runs
AFTER tile i's lanes have written ``sOut[s]``, so with one output stage the lanes
of tile i+1 overwrite ``sOut[0]`` while tile i's bulk store may still be reading
it -- tile i's GMEM rows then carry tile i+1's normed rows.  Reachable only when a
CTA processes >= 2 tiles (> SMs x 8 tiles per launch: h_q=32 at S >= 1024 on
Rubin; the unit test's 640-tile shape never reached it), and it shows up as
non-deterministic Q rows equal to the reference row of token ``tok + n_ctas /
q_tiles_per_token`` -- ~0.1 % of rows at S=16K, which moved a downstream causal
attention's cosine vs a second run to 0.998 and read as a "long-S SDPA bug".
The fused form needs ``tma_store_wait(stages_o - 2)`` (tile i+1's writer must
find tile i+1-stages_o's store retired) and therefore ``stages_o >= 2``;
``compile_qk_norm_rope_tma`` refuses the racy pair.  Its measured gain over the
unfused drain was +0.5 %, inside the control noise, so the default is the
unfused drain.
``refill_pos=REFILL_AFTER_COMPUTE`` issues the next tile's TMA before the store
issue and the rstd write rather than after them, so the load overlaps this
tile's epilogue too.

Measured cold on the Rubin dev node, S=32768, fused layout, L2-flushed against a
4453 GB/s ceiling flushed the same way:

    refill fused    GB/s   % ceiling   vs LDG
      0      0      4367       98.1     +41.8
      1      0      4375       98.2     +42.0
      0      1      4465      100.3     +45.0
      1      1      4487      100.8     +45.7      <- default

**Read that as a floor, not as the size of the win.** At 98-100% of a measured
device-to-device copy the kernel is saturated on this part, so the two orderings
have almost nothing left to buy and the 2.7% spread is compressed against the
ceiling. Both are free and both are structurally right, so both are on; re-measure
on a part with headroom before quoting either number as the effect size.

Final A/B at these defaults, LDG baseline PINNED with ``impl="ldg"``, cold,
fused layout, aarch64 Rubin PERF node (212 SMs, 2376 MHz):

    S        LDG    TMA    gain
    4096    4561   6046   +32.6%
    8192    5013   7130   +42.2%
    32768   5439   8308   +52.8%

**Achieved GB/s only -- do NOT quote a fraction on that node.** Its torch-`copy_`
ceiling probe reads 6791 GB/s while this kernel does 8308, i.e. 122%, which is
impossible; the same probe read 10859 GB/s on a different Rubin perf node at
identical clocks. The kernel number is exact and cold; the probe is what is
wrong there, and `probe_norm_rope_tma.py` now prints ``>CEIL!`` rather than a
plausible percentage.

Pinning matters: `_QkNormRope`'s default `impl="auto"` resolves to THIS kernel on
sm_90+, so an unpinned "baseline" benchmarks TMA against TMA and reports the
whole win as -2.6%."""
"""Ring geometry, set from measurement rather than from the SMEM budget.

Measured cold on the aarch64 Rubin perf node, B=1 bf16 S=32768, the block's real
fused layout, every point L2-flushed against a ceiling flushed the same way
(10859 GB/s). The LDG kernel here is the CURRENT one, i.e. it already carries
the +33% deferred-load fix:

    kernel                       GB/s   % ceiling   vs LDG
    LDG (shipped)                5564       51.2       --
    TMA rows=16 stages=2         8664       79.8    +55.7      <- default
    TMA rows=32 stages=1         7901       72.8    +42.0
    TMA rows=8  stages=2         7839       72.2    +40.3
    TMA rows=32 stages=2         7131       65.7    +28.2
    TMA rows=16 stages=1         6924       63.8    +24.5
    TMA rows=16 stages=3         6408       59.0    +15.2
    TMA rows=32 stages=3         5867       54.0     +5.5

Re-measured on the perf node with the shipped ``stages_o=1``, ``tile_rows`` is
confirmed: 8 -> 8201, **16 -> 8314**, 32 -> 6780 GB/s. 32 rows is an 18% loss,
so the peak is real and it is not an artifact of the dev node.

**Both axes are a genuine peak, and both peaks are the same trade.** ``stages=1``
loses 24 points because nothing overlaps; ``stages=3`` loses 21 because SMEM per
CTA goes 32 -> 48 KiB and the resident CTA count falls. ``rows=32`` doubles the
tile and pays the same way. The default sits at **32 KiB per CTA**
(``stages * tile_rows * 512 B * 2``, separate in and out buffers), which keeps 7
CTAs resident against the 227 KiB opt-in cap -- deliberately UNDER the 327 KiB
Rubin carveout, because reaching it costs L1 down to 8 kB and this kernel still
reads cos/sin and the norm weights through L1.

Re-measure per arch and per node: the two Rubin boxes here differ by 2.4x on the
copy ceiling alone. Reproduce with ``frost_dev/probe_norm_rope_tma.py``."""

_FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)


# ---------------------------------------------------------------------------
# BARRIER TABLE (cga1, no clusters: every arrive is LOCAL, so P6/P15 drains do
# not apply -- there is no cross-CTA arrive that could outlive a peer).
#
#   mbar          stages  producer          issuing lanes            consumers      init  phase
#   mb_full[s]    STAGES  TMA_LOAD          1 (warp_id==0 AND elect)  all threads   1     start(0)
#                         `elect_sync()` alone is one lane PER WARP = 4 arrives in
#                         a 4-warp CTA. The warp gate is load-bearing, and it must
#                         sit OUTSIDE tma_load_tile, which elects internally.
#                         arrive_expect_tx = TILE_ROWS*512 B; the two subtile
#                         TMA calls both complete into the SAME mbar, so the
#                         byte count is the whole tile, once.
#   There is NO _empty ring, deliberately. In this pipeline the producer and
#   the consumers are the SAME 128 threads, and the compute phase already ends
#   in a nvvm.barrier_cta_sync() -- which proves every thread finished reading
#   sIn[s] strictly before the refill is issued. An mbarrier would re-prove
#   that at the cost of 128 arrives per tile. (It was in the first draft, and
#   it was also MIS-PHASED: arriving and then waiting on the same barrier in
#   one iteration needs phase 0, not the pre-armed 1.)
#
#   SUM(issuing lanes) == init count:  mb_full 1 == 1.
#
# SMEM BUFFER TABLE
#
#   name     dtype  elems                    writer          reader            per-lane stride  swizzle
#   sIn      Int32  STAGES*TILE_ROWS*128     TMA             lanes, 16 B each  16 B (contig)    NONE: 16 lanes
#                                                                                               span 256 contiguous
#                                                                                               bytes -> all 32 banks,
#                                                                                               4 wavefronts = ideal
#   sOut     Int32  STAGES*TILE_ROWS*128     lanes, 16 B     TMA               16 B (contig)    NONE, same argument
#   sRstd    Fp32   STAGES*TILE_ROWS         1 lane per row  lanes 0..R-1      4 B              n/a (R*4 <= 128 B)
#            (only when want_rstd traces True; absent from the RoPE-only / no-rstd kernel)
#   mb_*     Int64  STAGES each              --              --                --               n/a
# ---------------------------------------------------------------------------


def validate_shape(d: int, rope_dim: int, h_q: int, h_kv: int, tile_rows: int, threads: int) -> None:
    """Raise on any geometry this pipeline cannot tile. Never an ``assert``:
    these come from user-facing geometry and must survive ``python -O``."""
    if d % TMA_GRANU_ELEMS != 0:
        raise ValueError(f"d_head must be a multiple of {TMA_GRANU_ELEMS} (the unswizzled TMA inner-box cap), got {d}")
    if d // ELEMS_PER_ACCESS != WARP:
        raise ValueError(
            f"this pipeline is warp-per-row: d_head must be {WARP * ELEMS_PER_ACCESS} so one warp covers a row in one 16-byte access each, got {d}"
        )
    if threads % WARP:
        raise ValueError(f"threads_per_cta must be a whole number of warps, got {threads}")
    warps = threads // WARP
    if tile_rows % warps:
        raise ValueError(f"tile_rows={tile_rows} must divide evenly across the {warps} warps of the CTA")
    if h_q % tile_rows:
        raise ValueError(f"tile_rows={tile_rows} must divide h_q={h_q}: a Q tile is tile_rows consecutive heads of ONE token")
    if tile_rows % h_kv:
        raise ValueError(f"tile_rows={tile_rows} must be a multiple of h_kv={h_kv}: a K tile is whole tokens of {h_kv} heads")
    if rope_dim:
        if rope_dim % (2 * ELEMS_PER_ACCESS):
            raise ValueError(f"rope_dim must be a multiple of {2 * ELEMS_PER_ACCESS} so the rotate_half partner is a whole-lane shuffle, got {rope_dim}")
        half_lanes = rope_dim // (2 * ELEMS_PER_ACCESS)
        if half_lanes & (half_lanes - 1):
            raise ValueError(f"rope_dim/{2 * ELEMS_PER_ACCESS} must be a power of two for the butterfly shuffle, got {rope_dim}")
        if rope_dim > TMA_GRANU_ELEMS:
            raise ValueError(f"rope_dim={rope_dim} must fit the first TMA subtile ({TMA_GRANU_ELEMS} elements) so the shuffle needs no cross-subtile exchange")


def _tile_handle(raw, off_elems, tile_elems: int, subtiles: int, subtile_elems: int):
    """A ``SmemTile`` view of one ring stage.

    Module-level and fully explicit on purpose: the DSL rejects a nested
    function that CAPTURES a variable when it is called from inside staged
    control flow (``SCOPE_CLOSURE_CAPTURE``), and every call here sits inside an
    ``if`` or a ``while``.
    """
    return SmemTile(
        base=raw.subview(off_elems),
        elems_per_stage=tile_elems,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=subtiles,
        tma_granu_elems=TMA_GRANU_ELEMS,
        tma_subtile_stride_elems=subtile_elems,
    )


@cute.jit
def _issue_tile_load(
    sIn_raw,
    s_off,
    mb_ptr,
    tile_idx,
    n_q_tiles,
    tma_q,
    tma_k,
    tile_elems: cutlass.Constexpr[int],
    subtiles: cutlass.Constexpr[int],
    subtile_elems: cutlass.Constexpr[int],
    tile_bytes: cutlass.Constexpr[int],
    tile_rows: cutlass.Constexpr[int],
    q_tiles_per_token: cutlass.Constexpr[int],
    toks_per_ktile: cutlass.Constexpr[int],
):
    """Arm the tile's mbar and fire both subtile TMAs into it.

    **The caller must ALREADY have selected one warp.** `nvvm.elect_sync()`
    elects a lane PER WARP, so in a 4-warp CTA a bare `if elect_sync():` arms
    `expect_tx` four times against `init(1)` and issues the TMA four times.
    `tma_load_tile` elects internally, which is exactly why the warp gate cannot
    live in here -- it has to be outside. Symptom of getting it wrong: a bare
    `unspecified launch failure` that survives every swizzle and box you sweep,
    while a TMA *store* on the same descriptors is fine.

    Both subtile TMAs complete into the SAME mbar, so `expect_tx` is the whole
    tile ONCE, not once per subtile.
    """
    if nvvm.elect_sync():
        arrive_expect_tx(mb_ptr, tile_bytes)
    handle = _tile_handle(sIn_raw, s_off, tile_elems, subtiles, subtile_elems)
    if tile_idx < n_q_tiles:
        tma_load_tile(
            handle,
            tma_q(
                cutlass.Int32(0),
                (tile_idx % cutlass.Int32(q_tiles_per_token)) * cutlass.Int32(tile_rows),
                tile_idx // cutlass.Int32(q_tiles_per_token),
            ),
            mb_ptr,
        )
    else:
        tma_load_tile(handle, tma_k(cutlass.Int32(0), cutlass.Int32(0), (tile_idx - n_q_tiles) * cutlass.Int32(toks_per_ktile)), mb_ptr)


@cute.kernel
def frost_qk_norm_rope_tma(
    mQ: cute.Tensor,  # [T, H_q, D] -- bound for its element_type only; Q/K data
    mWq: Optional[cute.Tensor],  # [D]  moves entirely through the descriptors
    mWk: Optional[cute.Tensor],  # [D]  None (both): RoPE-only -- no RMSNorm, no weight loads, no rstd
    mCos: cute.Tensor,  # [T, ROPE_DIM]
    mSin: cute.Tensor,  # [T, ROPE_DIM]
    mRstdQ: Optional[cute.Tensor],  # [T, H_q]  fp32, or None
    mRstdK: Optional[cute.Tensor],  # [T, H_kv] fp32, or None
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_qo_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_ko_desc: cutlass.GridConstant[tmap.TensorMap],
    n_tokens: cutlass.Int32,
    n_q_tiles: cutlass.Int32,
    n_tiles: cutlass.Int32,
    n_ctas: cutlass.Int32,
    eps: cutlass.Float32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    tile_rows: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    stages_o: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    refill_pos: cutlass.Constexpr[int],
    fused_store_wait: cutlass.Constexpr[bool],
) -> None:
    """Grid-stride over tiles, ``stages``-deep TMA ring, warp-per-row compute.

    Q tiles come first in the flat tile space, then K tiles, so ``is_q`` is ONE
    comparison per TILE and warp-uniform. The LDG kernel's per-row branch, which
    selected between two base pointers and two token strides inside the hot
    loop, does not exist here.
    """
    warps = cutlass.const_expr(threads_per_cta // WARP)
    rows_per_warp = cutlass.const_expr(tile_rows // warps)
    tile_elems = cutlass.const_expr(tile_rows * d)
    subtile_elems = cutlass.const_expr(tile_rows * TMA_GRANU_ELEMS)
    subtiles = cutlass.const_expr(d // TMA_GRANU_ELEMS)
    q_tiles_per_token = cutlass.const_expr(h_q_ct // tile_rows)
    toks_per_ktile = cutlass.const_expr(tile_rows // h_kv_ct)
    rope_lanes = cutlass.const_expr(rope_dim // ELEMS_PER_ACCESS)
    half_lanes = cutlass.const_expr(rope_lanes // 2)
    tile_bytes = cutlass.const_expr(tile_rows * d * BPE)
    lanes_per_subtile = cutlass.const_expr(TMA_GRANU_ELEMS // ELEMS_PER_ACCESS)  # 16
    # Presence switches, decided at trace time (same idiom for both): None
    # weights trace the RoPE-only kernel (no sum-of-squares, no rsqrt, no
    # weight loads), None rstd traces the store-less one. The host refuses
    # want_rstd without apply_norm, so the conjunction below never hides a
    # requested store -- it only keeps the kernel self-consistent.
    apply_norm = cutlass.const_expr(mWq is not None)
    want_rstd = cutlass.const_expr(apply_norm and mRstdQ is not None)

    io_dtype = mQ.element_type  # a trace-time type object, NOT a const_expr candidate
    sIn_raw = cutlass.Array(io_dtype, stages * tile_elems, alignment=128, space=cutlass.AddressSpace.smem)
    sOut_raw = cutlass.Array(io_dtype, stages_o * tile_elems, alignment=128, space=cutlass.AddressSpace.smem)
    # Allocated only when the rstd store exists: both uses sit under
    # const_expr(want_rstd), so the RoPE-only / no-rstd trace keeps the SMEM too.
    sRstd = cutlass.Array(cutlass.Float32, stages * tile_rows, alignment=16, space=cutlass.AddressSpace.smem) if cutlass.const_expr(want_rstd) else None
    mb_full = cutlass.Array(cutlass.Int64, stages, alignment=16, space=cutlass.AddressSpace.smem)

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(WARP)
    warp_id = tidx // cutlass.Int32(WARP)
    sub = lane // cutlass.Int32(lanes_per_subtile)  # which TMA subtile this lane reads
    col = lane % cutlass.Int32(lanes_per_subtile)  # 16-byte slot within it

    # --- P4: ONE warp, ONE lane inits EVERY stage of every ring -------------
    if warp_id == 0:
        if nvvm.elect_sync():
            for s in cutlass.range_constexpr(stages):
                nvvm.mbarrier_init(mb_full.subview(s), 1)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    tma_q = GmemTileTma(tma_q_desc)
    tma_k = GmemTileTma(tma_k_desc)
    tma_qo = GmemTileTma(tma_qo_desc)
    tma_ko = GmemTileTma(tma_ko_desc)

    my0 = cutlass.Int32(cute.arch.block_idx()[0])

    # --- prologue: fill the ring -------------------------------------------
    for s in cutlass.range_constexpr(stages):
        t0 = my0 + cutlass.Int32(s) * n_ctas
        if t0 < n_tiles and warp_id == 0:  # ONE WARP; _issue_tile_load elects the lane
            _issue_tile_load(
                sIn_raw,
                s * tile_elems,
                mb_full.subview(s),
                t0,
                n_q_tiles,
                tma_q,
                tma_k,
                tile_elems,
                subtiles,
                subtile_elems,
                tile_bytes,
                tile_rows,
                q_tiles_per_token,
                toks_per_ktile,
            )

    full_state = PipelineState.start(phase=0)
    # The OUTPUT ring is independent of the input ring. They are two different
    # resources: the input depth buys load-ahead, the output depth buys store
    # slack, and SMEM spent on one is SMEM not spent on the other. Only .idx is
    # used -- there is no mbarrier on the output side, the drain is the
    # `tma_store_wait` group count.
    out_state = PipelineState.start(phase=0)
    # Loop-carried, for refill_pos == REFILL_TOP only: the stage the PREVIOUS
    # iteration finished with, and the tile it should be refilled with. -1 means
    # "nothing pending". This exists because the top of the loop cannot refill
    # its OWN stage -- that stage still holds the data this iteration is about
    # to read -- so the top placement necessarily refills one stage behind.
    prev_s = cutlass.Int32(0)
    prev_nxt = cutlass.Int32(-1)

    tile = my0
    while tile < n_tiles:
        s = full_state.idx
        so = out_state.idx
        wait(mb_full.subview(s), full_state.phase)
        full_state = advance(full_state, stages)
        out_state = advance(out_state, stages_o)

        if cutlass.const_expr(refill_pos == REFILL_TOP):
            if prev_nxt >= cutlass.Int32(0) and warp_id == 0:
                _issue_tile_load(
                    sIn_raw,
                    prev_s * cutlass.Int32(tile_elems),
                    mb_full.subview(prev_s),
                    prev_nxt,
                    n_q_tiles,
                    tma_q,
                    tma_k,
                    tile_elems,
                    subtiles,
                    subtile_elems,
                    tile_bytes,
                    tile_rows,
                    q_tiles_per_token,
                    toks_per_ktile,
                )

        # sOut[s]'s PREVIOUS store must have drained before any lane rewrites
        # it. Only the elected lane owns a bulk store group, so it waits and a
        # CTA barrier publishes that to every writer.
        #
        # `fused_store_wait` moves that pair onto the barrier the epilogue
        # ALREADY runs, which deletes a whole CTA barrier from the per-tile
        # critical path. Its drain runs one iteration ahead: the wait sits AFTER
        # tile i's lanes wrote sOut[s], so the buffer it must protect is the one
        # tile i+1 writes next, last read by the store of tile i+1-stages_o.
        # At that point the outstanding groups are tiles <= i-1 (tile i's store
        # is issued after the barrier), so `tma_store_wait(stages_o - 2)` is the
        # bound -- which needs stages_o >= 2.  `stages_o - 1` at stages_o=1 was
        # the WAR race described at DEFAULT_FUSED_STORE_WAIT.
        if cutlass.const_expr(not fused_store_wait):
            if warp_id == 0:
                if nvvm.elect_sync():
                    tma_store_wait(stages_o - 1)
            nvvm.barrier_cta_sync()

        is_q = tile < n_q_tiles  # ONE warp-uniform decision per TILE
        tok_q = tile // cutlass.Int32(q_tiles_per_token)
        head0_q = (tile % cutlass.Int32(q_tiles_per_token)) * cutlass.Int32(tile_rows)
        tok0_k = (tile - n_q_tiles) * cutlass.Int32(toks_per_ktile)

        for u in cutlass.range_constexpr(rows_per_warp):
            r = warp_id + cutlass.Int32(u * warps)
            # K's LAST tile may overshoot T (a K tile is `toks_per_ktile` whole
            # tokens; Q never does -- tile_rows divides h_q). TMA clips the
            # overshooting rows on both the load (zero-fill) and the store, and
            # the rstd write below is predicated on `tok < n_tokens` -- but the
            # cos/sin TABLE loads are ordinary `ld.global`, so an unclamped
            # token here reads 16 B per rope lane past the end of a `[T, ROPE]`
            # table (T=3, h_kv=2, tile_rows=8: rows 6..7 are token 3; memcheck
            # on PR #1102). Clamp the padded rows to the last real token: the
            # table row is valid, the result is discarded by the TMA clip, and
            # the warp collectives (lane_group_sum, the RoPE shfl.bfly) stay
            # unconditional -- a `select`, not a branch.
            tok_k = tok0_k + r // cutlass.Int32(h_kv_ct)
            tok_k = tok_k if tok_k < n_tokens else n_tokens - cutlass.Int32(1)
            tok = tok_q if is_q else tok_k
            # subtile-major: TMA laid the tile down as [subtile][row][GRANU].
            # The lane offset is shared, but the STAGE is not -- the input ring
            # and the output ring have independent depths, so a single `off`
            # would write this tile into whichever output buffer the INPUT ring
            # happened to be using. Silent, and wrong only once the two depths
            # differ.
            lane_off = sub * cutlass.Int32(subtile_elems) + r * cutlass.Int32(TMA_GRANU_ELEMS) + col * cutlass.Int32(ELEMS_PER_ACCESS)
            off_in = s * cutlass.Int32(tile_elems) + lane_off
            off_out = so * cutlass.Int32(tile_elems) + lane_off

            xs = sIn_raw.load(off_in, vector_size=ELEMS_PER_ACCESS, alignment=16).to(cutlass.Float32).to_elements()
            # Pre-bound so the names exist on both trace paths; under
            # apply_norm they are replaced by the real rsqrt / normed row, and
            # under RoPE-only `ys IS xs`: the passthrough dims [rope_dim, D)
            # are widened and re-narrowed with no fp32 op in between -> a
            # bit-exact copy (asserted by the tests).
            rstd = cutlass.Float32(1.0)
            ys = list(xs)
            if cutlass.const_expr(apply_norm):
                acc = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    acc = acc + xs[i] * xs[i]
                rstd = cute.math.rsqrt(lane_group_sum(acc, WARP) * cutlass.Float32(1.0 / d) + eps, fastmath=True)

                # Secondary loads at their POINT OF USE, for the same reason as the
                # LDG kernel: both are cache hits by construction, and hoisting them
                # holds 24 live fp32 per row across the reduction.
                w_addr = (mWq.iterator.toint() if is_q else mWk.iterator.toint()) + (
                    sub.to(cutlass.Int64) * cutlass.Int64(TMA_GRANU_ELEMS) + col.to(cutlass.Int64) * cutlass.Int64(ELEMS_PER_ACCESS)
                ) * cutlass.Int64(BPE)
                ws = [v for pair in [f16x2_to_f32(w, dtype=mWq.element_type) for w in ld_global_v4(w_addr, cutlass.Int32)] for v in pair]
                ys = [xs[i] * rstd * ws[i] for i in range(ELEMS_PER_ACCESS)]

            if cutlass.const_expr(rope_dim > 0):
                in_rope = lane < cutlass.Int32(rope_lanes)
                cs = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
                sn = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
                if in_rope:
                    rope_off = lane.to(cutlass.Int64) * cutlass.Int64(ELEMS_PER_ACCESS * BPE)
                    cb = mCos.iterator.toint() + tok.to(cutlass.Int64) * cutlass.Int64(mCos.stride[0]) * cutlass.Int64(BPE) + rope_off
                    sb = mSin.iterator.toint() + tok.to(cutlass.Int64) * cutlass.Int64(mSin.stride[0]) * cutlass.Int64(BPE) + rope_off
                    cs = [v for pair in [f16x2_to_f32(w, dtype=mCos.element_type) for w in ld_global_v4(cb, cutlass.Int32)] for v in pair]
                    sn = [v for pair in [f16x2_to_f32(w, dtype=mSin.element_type) for w in ld_global_v4(sb, cutlass.Int32)] for v in pair]
                # Unconditional: shfl.sync must be reached by every lane.
                rot = []
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    partner = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, ys[i], cutlass.Int32(half_lanes), 31, kind=nvvm.Shfl.BFLY))
                    signed = -partner if lane < cutlass.Int32(half_lanes) else partner
                    rot.append(ys[i] * cs[i] + signed * sn[i])
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    ys[i] = rot[i] if in_rope else ys[i]

            sOut_raw.store(cutlass.Vector.from_elements(tuple(ys), cutlass.Float32).to(io_dtype), off_out, vector_size=ELEMS_PER_ACCESS, alignment=16)
            if cutlass.const_expr(want_rstd):
                # gathered here, emitted below as ONE coalesced store for the
                # whole tile -- not one 4-byte scattered store per row
                if lane == cutlass.Int32(0):
                    sRstd.subview(s * cutlass.Int32(tile_rows) + r).store(rstd)

        nvvm.fence_proxy("async.shared", space="cta")  # lane writes -> TMA (async proxy)
        if cutlass.const_expr(fused_store_wait):
            # stages_o >= 2 here (compile_qk_norm_rope_tma refuses the pair otherwise):
            # tile i+1 rewrites sOut[(i+1) % stages_o], last read by tile i+1-stages_o's
            # store; with tiles <= i-1 outstanding that leaves stages_o - 2 groups in flight.
            if warp_id == 0:
                if nvvm.elect_sync():
                    tma_store_wait(stages_o - 2)
        nvvm.barrier_cta_sync()  # publishes BOTH the lane writes and the drain

        # This stage's sIn is provably free the moment that barrier passes, so
        # the refill can go out BEFORE the store issue and the rstd write rather
        # than after them -- the load then overlaps this tile's epilogue too.
        nxt = tile + n_ctas * cutlass.Int32(stages)
        if cutlass.const_expr(refill_pos == REFILL_AFTER_COMPUTE):
            if nxt < n_tiles and warp_id == 0:
                _issue_tile_load(
                    sIn_raw,
                    s * cutlass.Int32(tile_elems),
                    mb_full.subview(s),
                    nxt,
                    n_q_tiles,
                    tma_q,
                    tma_k,
                    tile_elems,
                    subtiles,
                    subtile_elems,
                    tile_bytes,
                    tile_rows,
                    q_tiles_per_token,
                    toks_per_ktile,
                )

        if warp_id == 0:
            if nvvm.elect_sync():
                if is_q:
                    tma_store_tile(
                        _tile_handle(sOut_raw, so * cutlass.Int32(tile_elems), tile_elems, subtiles, subtile_elems), tma_qo(cutlass.Int32(0), head0_q, tok_q)
                    )
                else:
                    tma_store_tile(
                        _tile_handle(sOut_raw, so * cutlass.Int32(tile_elems), tile_elems, subtiles, subtile_elems),
                        tma_ko(cutlass.Int32(0), cutlass.Int32(0), tok0_k),
                    )
                tma_store_commit()

        if cutlass.const_expr(want_rstd):
            if warp_id == 0:
                if lane < cutlass.Int32(tile_rows):
                    tok_l = tok_q if is_q else (tok0_k + lane // cutlass.Int32(h_kv_ct))
                    idx = (tok_q * cutlass.Int32(h_q_ct) + head0_q + lane) if is_q else (tok0_k * cutlass.Int32(h_kv_ct) + lane)
                    rbase = mRstdQ.iterator.toint() if is_q else mRstdK.iterator.toint()
                    val = sRstd.subview(s * cutlass.Int32(tile_rows) + lane).load()
                    if tok_l < n_tokens:  # K's last tile may overshoot; Q never does
                        st_global(rbase + idx.to(cutlass.Int64) * cutlass.Int64(4), val, cutlass.Float32)

        if cutlass.const_expr(refill_pos == REFILL_LATE):
            if nxt < n_tiles and warp_id == 0:
                _issue_tile_load(
                    sIn_raw,
                    s * cutlass.Int32(tile_elems),
                    mb_full.subview(s),
                    nxt,
                    n_q_tiles,
                    tma_q,
                    tma_k,
                    tile_elems,
                    subtiles,
                    subtile_elems,
                    tile_bytes,
                    tile_rows,
                    q_tiles_per_token,
                    toks_per_ktile,
                )
        if cutlass.const_expr(refill_pos == REFILL_TOP):
            prev_s = s
            prev_nxt = nxt if nxt < n_tiles else cutlass.Int32(-1)
        tile = tile + n_ctas

    # Every arrive above is LOCAL, so there is no cross-CTA drain to do (P15
    # does not apply at cga1). The only thing that must not outlive the CTA is
    # the last bulk store.
    tma_store_wait(0)


@cute.jit
def qk_norm_rope_tma_launch(
    q: cute.Tensor,
    k: cute.Tensor,
    q_out: cute.Tensor,
    k_out: cute.Tensor,
    w_q: Optional[cute.Tensor],
    w_k: Optional[cute.Tensor],
    cos: cute.Tensor,
    sin: cute.Tensor,
    rstd_q: Optional[cute.Tensor],
    rstd_k: Optional[cute.Tensor],
    n_tokens: cutlass.Int32,
    n_q_tiles: cutlass.Int32,
    n_tiles: cutlass.Int32,
    n_ctas: cutlass.Int32,
    eps: cutlass.Float32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    tile_rows: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    stages_o: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    refill_pos: cutlass.Constexpr[int],
    fused_store_wait: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Build the four descriptors, then launch.

    Box shapes, innermost-LAST to match the tensor's ``[T, H, D]`` order:

    * **Q**: ``(1, tile_rows, GRANU)`` -- one token, ``tile_rows`` consecutive
      heads. Legal because ``tile_rows`` divides ``h_q``.
    * **K**: ``(tile_rows // h_kv, h_kv, GRANU)`` -- whole tokens of ``h_kv``
      heads, because ``h_kv`` is small (2 at the 397B geometry) and a tile of
      ``tile_rows`` therefore spans several tokens.

    ``swizzle=none`` caps the innermost box at 256 B, which is why ``d=256``
    bf16 needs ``d // GRANU`` TMA calls per tile. Unswizzled is not a
    compromise here: see the SMEM buffer table at the top -- warp-per-row is
    already conflict-free, and TMA's 128 B swizzle is built for the opposite
    access pattern.

    The per-batch token stride rides in the descriptor, so a strided source
    (Q and K as column slices of the fused projection) costs the KERNEL nothing.
    That is the address-math tax the LDG kernel pays in registers.
    """
    box_q = (1, tile_rows, TMA_GRANU_ELEMS)
    box_k = (tile_rows // h_kv_ct, h_kv_ct, TMA_GRANU_ELEMS)
    order = (2, 1, 0)
    mk = lambda t, box: tmap.create_tensor_map_tiled_from_view(
        t,
        box_dims=box,
        stride_order=order,
        swizzle=tmap.TensorMapSwizzle.none,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    frost_qk_norm_rope_tma(
        q,
        w_q,
        w_k,
        cos,
        sin,
        rstd_q,
        rstd_k,
        mk(q, box_q),
        mk(k, box_k),
        mk(q_out, box_q),
        mk(k_out, box_k),
        n_tokens,
        n_q_tiles,
        n_tiles,
        n_ctas,
        eps,
        d,
        rope_dim,
        tile_rows,
        stages,
        stages_o,
        threads_per_cta,
        h_q_ct,
        h_kv_ct,
        refill_pos,
        fused_store_wait,
    ).launch(grid=(n_ctas, 1, 1), block=(threads_per_cta, 1, 1), stream=stream)


compiled_cache = {}


class QkNormRopeTmaRecipe(NamedTuple):
    """Build-time facts of one TMA norm+RoPE launch (a legal compile key, like
    ``QkNormRopeRecipe``). ``apply_norm`` is appended with a default so every
    existing recipe reads the same; ``run_qk_norm_rope_tma`` checks it against
    the bound weights in both directions."""

    compiled: object
    h_q: int
    h_kv: int
    d: int
    eps: float
    tile_rows: int
    stages: int
    stages_o: int
    threads: int
    want_rstd: bool
    ctas_per_sm: int
    apply_norm: bool = True


def _fake(dtype, shape, stride_order):
    return cute.runtime.make_fake_compact_tensor(
        dtype=_convert_to_cutlass_data_type(dtype),
        shape=shape,
        stride_order=stride_order,
        assumed_align=16,
    )


def _fake_thd(dtype, tok, h: int, d: int):
    """``[T, H, D]`` with a SYMBOLIC token stride, so one artifact serves both
    the fused-projection slice (stride ``N``) and a compact buffer (stride
    ``h*d``). Unlike the LDG kernel this costs the kernel nothing: the stride
    reaches the hardware through the descriptor, not through address math."""
    return cute.runtime.make_fake_tensor(
        dtype=_convert_to_cutlass_data_type(dtype),
        shape=(tok, h, d),
        stride=(cute.sym_int(), d, 1),
        assumed_align=16,
    )


def tile_counts(t: int, h_q: int, h_kv: int, tile_rows: int) -> tuple[int, int]:
    """(n_q_tiles, n_tiles). A Q tile is ``tile_rows`` heads of one token; a K
    tile is ``tile_rows // h_kv`` whole tokens, so its LAST tile may overshoot
    ``T`` -- TMA clips the store against the descriptor extent and the rstd
    write is predicated."""
    n_q = t * (h_q // tile_rows)
    toks_per_ktile = tile_rows // h_kv
    n_k = (t + toks_per_ktile - 1) // toks_per_ktile
    return n_q, n_q + n_k


def compile_qk_norm_rope_tma(
    *,
    dtype,
    h_q: int,
    h_kv: int,
    d: int,
    rope_dim: int,
    eps: float,
    want_rstd: bool,
    tile_rows: int = DEFAULT_TILE_ROWS,
    stages: int = DEFAULT_STAGES,
    stages_o: Optional[int] = None,  # None = DEFAULT_STAGES_O
    threads_per_cta: int = DEFAULT_THREADS,
    refill_pos: int = DEFAULT_REFILL_POS,
    fused_store_wait: bool = DEFAULT_FUSED_STORE_WAIT,
    ctas_per_sm: int = 8,
    apply_norm: bool = True,
) -> QkNormRopeTmaRecipe:
    """Build the TMA artifact from shapes alone. ``apply_norm=False`` traces the
    RoPE-only kernel (weights traced as ``None``, no rstd allowed) and is part
    of the cache key so a norm-on artifact is never reused with ``None`` weights."""
    validate_shape(d, rope_dim, h_q, h_kv, tile_rows, threads_per_cta)
    # None means the DEFAULT, not "match stages" -- otherwise changing
    # DEFAULT_STAGES_O silently does nothing for every caller that omits it,
    # which is exactly what happened when it moved 2 -> 1.
    stages_o = int(DEFAULT_STAGES_O if stages_o is None else stages_o)
    if stages_o < 1:
        raise ValueError(f"stages_o must be >= 1, got {stages_o}")
    if fused_store_wait and stages_o < 2:
        # The fused drain waits AFTER this tile's lanes wrote sOut, so it can only
        # protect the NEXT tile's buffer when there is a second one: at stages_o=1
        # the lanes of tile i+1 overwrite sOut[0] under tile i's in-flight bulk
        # store (see DEFAULT_FUSED_STORE_WAIT) -- silent wrong Q/K rows, no error.
        raise ValueError(f"fused_store_wait=True needs stages_o >= 2 (the drain protects the next tile's output stage), got stages_o={stages_o}")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"qk_norm_rope_tma serves bf16/f16 only, got {dtype}")
    if not apply_norm and rope_dim == 0:
        raise ValueError("apply_norm=False with rope_dim=0 is an identity copy of Q/K; drop the stage instead of launching it")
    if want_rstd and not apply_norm:
        raise ValueError("apply_norm=False (RoPE-only Q/K) computes no RMSNorm and therefore emits no rstd; want_rstd must be False")
    # Every Constexpr handed to cute.compile below is in the key. `stages_o`
    # was missing (PR #1102 review): it sizes sOut_raw and bounds the store
    # drain, so the second depth requested in a process was served the FIRST
    # one's artifact under a recipe that reported the second.
    key = (
        str(dtype),
        h_q,
        h_kv,
        d,
        int(rope_dim),
        int(tile_rows),
        int(stages),
        int(stages_o),
        int(threads_per_cta),
        bool(want_rstd),
        int(refill_pos),
        bool(fused_store_wait),
        current_device(),
        bool(apply_norm),
    )
    if key not in compiled_cache:
        tok = cute.sym_int()
        dense = [_fake_thd(dtype, tok, h, d) for h in (h_q, h_kv, h_q, h_kv)]
        weights = [_fake(dtype, (d,), (0,)) for _ in range(2)] if apply_norm else [None, None]
        tables = [_fake(dtype, (tok, rope_dim if rope_dim else 1), (1, 0)) for _ in range(2)]
        rstd = [_fake(torch.float32, (tok, h), (1, 0)) for h in (h_q, h_kv)] if want_rstd else [None, None]
        compiled_cache[key] = cute.compile(
            qk_norm_rope_tma_launch,
            *dense,
            *weights,
            *tables,
            *rstd,
            cutlass.Int32(0),  # n_tokens  ) runtime; the zeros pin only the TYPE
            cutlass.Int32(0),  # n_q_tiles )
            cutlass.Int32(0),  # n_tiles   )
            cutlass.Int32(1),  # n_ctas    )
            cutlass.Float32(eps),
            d,
            int(rope_dim),
            int(tile_rows),
            int(stages),
            int(stages_o),
            int(threads_per_cta),
            int(h_q),
            int(h_kv),
            int(refill_pos),
            bool(fused_store_wait),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return QkNormRopeTmaRecipe(
        compiled=compiled_cache[key],
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        eps=float(eps),
        tile_rows=int(tile_rows),
        stages=int(stages),
        stages_o=int(stages_o),
        threads=int(threads_per_cta),
        want_rstd=bool(want_rstd),
        ctas_per_sm=int(ctas_per_sm),
        apply_norm=bool(apply_norm),
    )


def run_qk_norm_rope_tma(r, q, k, q_out, k_out, w_q, w_k, cos, sin, rstd_q=None, rstd_k=None, *, stream) -> None:
    """The lowered launch. ``w_q``/``w_k`` are both ``None`` for a RoPE-only
    recipe (``r.apply_norm`` False) and both tensors otherwise -- checked, both
    directions, exactly as the LDG runner does."""
    t = int(q.shape[0])
    n_q_tiles, n_tiles = tile_counts(t, r.h_q, r.h_kv, r.tile_rows)
    check_norm_weights_match_recipe(r.apply_norm, w_q, w_k)
    if r.want_rstd and (rstd_q is None or rstd_k is None):
        raise ValueError("this artifact was compiled with rstd outputs; both must be bound at execute (Rule 1: no silent fallback)")
    if not r.want_rstd and (rstd_q is not None or rstd_k is not None):
        raise ValueError("this artifact was compiled WITHOUT rstd outputs (want_rstd=False); rstd_q / rstd_k would be silently ignored -- pass None")
    n_ctas = min(n_tiles, multiprocessor_count(current_device()) * r.ctas_per_sm)
    r.compiled(
        q,
        k,
        q_out,
        k_out,
        w_q,
        w_k,
        cos,
        sin,
        rstd_q,
        rstd_k,
        cutlass.Int32(t),
        cutlass.Int32(n_q_tiles),
        cutlass.Int32(n_tiles),
        cutlass.Int32(n_ctas),
        cutlass.Float32(r.eps),
        cuda.CUstream(int(stream)),
    )


frost_qk_norm_rope_tma.set_name_prefix("cudnn", remove_cutlass_symbol=True)
