# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-aware joint geometry, split-K and orientation selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import NamedTuple

from .arch_family import family_for_arch
from .dtypes import DTYPE_BITS
from .fusion_ir import FusionChain, swap_ab, wide_quant
from .kernel_registry import GENERAL_BLOCK_SCALE_GRAPH_TYPES, classify_graph_type, general_glu_pair
from .tile_config import PLANNER_K_BOUNDS, PLANNER_MAX_BATCH, PLANNER_MN_BOUNDS, TileConfig, as_mma_tile_k, as_pipeline, by_name, select_config

MAX_PLAN_CONFIGS = 8


@dataclass(frozen=True)
class DeviceProperties:
    arch: int
    name: str
    sm_count: int
    l2_bytes: int
    # Entry i: the clusters of i + 1 CTAs the part runs at once (`cudnn.frost.occupancy.occupancy_map`). None: the current device's
    # table when this describes it (`_clusters_cover_sms`).
    occupancy: tuple[int, ...] | None = None


def current_device_properties() -> DeviceProperties | None:
    from cudnn.frost import device

    if not device.is_available():
        return None
    ordinal = device.resolve_device(None)
    major, minor = device.compute_capability(ordinal)
    return DeviceProperties(major * 10 + minor, device.device_name(ordinal), device.multiprocessor_count(ordinal), device.l2_cache_bytes(ordinal))


class CostFeatures(NamedTuple):
    launch: float
    waves: float
    k_steps: float
    mma_work: float
    input_mb: float
    partial_cache_mb: float
    partial_spill_mb: float
    reduce_launch: float
    reduce_steps: float
    m_store: float
    m_reduce: float
    scale_input_mb: float = 0
    scale_stage_kb: float = 0


@dataclass(frozen=True)
class CostProfile:
    name: str
    device_name: str
    coefficients: CostFeatures
    extra_geometries: tuple[str, ...]
    switch_margin: float = 0.05
    tie_margin: float = 0.01
    diversity_margin: float = 0.5
    # the shape envelope this profile claims; the defaults are the engine-wide planner envelope `tile_config` publishes (a host-side
    # resolver reads it there without importing this module), and every profile stays inside it (`test_tile_select_analytic.py` pins that)
    mn_bounds: tuple[int, int] = PLANNER_MN_BOUNDS
    k_bounds: tuple[int, int] = PLANNER_K_BOUNDS
    max_batch: int = PLANNER_MAX_BATCH
    b_n_major_ok: bool = False
    scalar_scale_epilogue_ok: bool = False


_SM100_EXTRA_GEOMETRIES = (
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
)

_PROFILES = {
    (100, "sm100", "fp32", 16, 16, 16, False): CostProfile(
        name="b200_f16",
        device_name="NVIDIA B200",
        coefficients=CostFeatures(3.815061, 0.597739, 0.108050, 0.158420, 0.065296, 0.130176, 0.171868, 3.377794, 0.007157, 0.243471, 5.152299),
        extra_geometries=_SM100_EXTRA_GEOMETRIES,
    ),
    (100, "sm100", "fp32", 8, 8, 16, False): CostProfile(
        name="b200_f8",
        device_name="NVIDIA B200",
        coefficients=CostFeatures(4.306918, 0.697884, 0.082157, 0.210438, 0.060373, 0.155048, 0.176854, 3.691604, 0.0, 0.0, 5.601445),
        extra_geometries=_SM100_EXTRA_GEOMETRIES,
    ),
    (100, "sm100", "fp32", 4, 4, 16, True): CostProfile(
        name="b200_block_f4",
        device_name="NVIDIA B200",
        coefficients=CostFeatures(
            4.154048, 1.255628, 0.025315, 0.220332, 0.027872, 0.105108, 0.134126, 3.717077, 0.052194, 0.000000, 5.997925, 0.000000, 0.012659
        ),
        extra_geometries=_SM100_EXTRA_GEOMETRIES,
    ),
    (100, "sm100", "fp32", 8, 8, 16, True): CostProfile(
        name="b200_block_f8",
        device_name="NVIDIA B200",
        coefficients=CostFeatures(
            3.964425, 1.290942, 0.012677, 0.241671, 0.029366, 0.131104, 0.159307, 3.929639, 0.000603, 0.061931, 5.948227, 0.000000, 0.000124
        ),
        extra_geometries=_SM100_EXTRA_GEOMETRIES,
    ),
    (107, "sm100", "fp32", 16, 16, 16, False): CostProfile(
        name="rubin_f16",
        device_name="NVIDIA Graphics Device",
        coefficients=CostFeatures(3.471536, 0.765374, 0.079354, 0.212594, 0.087731, 0.066042, 0.085379, 4.795416, 0.211385, 0.0, 6.228140),
        extra_geometries=_SM100_EXTRA_GEOMETRIES,
    ),
    (107, "sm100", "fp32", 8, 8, 16, False): CostProfile(
        name="rubin_f8",
        device_name="NVIDIA Graphics Device",
        coefficients=CostFeatures(3.650657, 0.704300, 0.061594, 0.241686, 0.128494, 0.113544, 0.120002, 4.853120, 0.109531, 0.0, 6.428416),
        extra_geometries=_SM100_EXTRA_GEOMETRIES,
        switch_margin=0.35,
    ),
    (107, "sm100", "fp32", 4, 4, 16, True): CostProfile(
        name="rubin_block_f4",
        device_name="NVIDIA Graphics Device",
        coefficients=CostFeatures(3.416040, 1.612528, 0.120783, 0.131915, 0.111067, 0.080589, 0.074068, 6.561233, 0.197336, 0.582108, 6.548785, 0.0, 0.005212),
        extra_geometries=_SM100_EXTRA_GEOMETRIES,
    ),
    (107, "sm100", "fp32", 8, 8, 16, True): CostProfile(
        name="rubin_block_f8",
        device_name="NVIDIA Graphics Device",
        coefficients=CostFeatures(3.716719, 1.445248, 0.127218, 0.092622, 0.080023, 0.145638, 0.148422, 4.498155, 0.0, 0.568267, 6.559288),
        extra_geometries=_SM100_EXTRA_GEOMETRIES,
    ),
    (120, "sm120", "fp32", 16, 16, 16, False): CostProfile(
        name="rtx_pro_6000_f16",
        b_n_major_ok=True,
        scalar_scale_epilogue_ok=True,
        device_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        coefficients=CostFeatures(
            10.742818, 0.000000, 0.290090, 0.268702, 0.773501, 0.361878, 0.000000, 0.000000, 0.000000, 6.027293, 4.269089, 0.000000, 0.000000
        ),
        extra_geometries=(
            "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",
            "CONFIG_sm120_128x64x128_16x16x32_cluster1x1_warps4x2",
        ),
        switch_margin=0.15,
    ),
    (120, "sm120", "fp32", 8, 8, 16, False): CostProfile(
        name="rtx_pro_6000_f8",
        b_n_major_ok=True,
        scalar_scale_epilogue_ok=True,
        device_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        coefficients=CostFeatures(
            11.486993, 1.290724, 0.218370, 0.286799, 0.716421, 0.232828, 0.000000, 0.000000, 0.000000, 2.733481, 6.148131, 0.000000, 0.000000
        ),
        extra_geometries=(
            "CONFIG_sm120_64x128x128_16x16x32_cluster1x1_warps4x2",
            "CONFIG_sm120_64x128x128_16x16x32_cluster1x1_warps2x4",
        ),
    ),
    (120, "sm120", "fp32", 4, 4, 16, True): CostProfile(
        name="rtx_pro_6000_block_f4",
        b_n_major_ok=True,
        scalar_scale_epilogue_ok=True,
        device_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        coefficients=CostFeatures(
            8.148357, 0.451229, 0.272887, 0.946870, 1.139299, 0.224587, 0.000000, 0.636982, 0.000000, 2.322341, 6.716882, 0.028324, 0.000000
        ),
        extra_geometries=(
            "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",
            "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4",
        ),
        switch_margin=0.15,
    ),
    (120, "sm120", "fp32", 8, 8, 16, True): CostProfile(
        name="rtx_pro_6000_block_f8",
        b_n_major_ok=True,
        scalar_scale_epilogue_ok=True,
        device_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        coefficients=CostFeatures(
            0.000000, 0.035038, 0.137006, 0.817486, 0.000000, 0.277948, 0.694882, 2.379944, 0.000000, 2.737760, 4.149503, 6.818484, 0.000000
        ),
        extra_geometries=(
            "CONFIG_sm120_64x64x128_16x16x32_cluster1x1_warps4x2",
            "CONFIG_sm120_64x128x128_16x16x32_cluster1x1_warps2x4",
        ),
        switch_margin=0.12,
    ),
}


def _block_scale_supported(chain: FusionChain) -> bool:
    bs = chain.block_scale
    if bs is None:
        return True
    return not (
        bs.fake_dequant_a
        or bs.fake_dequant_b
        or bs.block_size_a not in ((1, 16), (1, 32))
        or bs.block_size_b not in ((16, 1), (32, 1))
        or bs.sfa_reorder != "F8_128x4"
        or bs.sfb_reorder != "F8_128x4"
        or DTYPE_BITS.get(bs.sf_dtype_a) != 8
        or DTYPE_BITS.get(bs.sf_dtype_b) != 8
    )


def _simple_graph(chain: FusionChain) -> bool:
    mm = chain.matmul
    if chain.has_moe or chain.is_multi_gemm or chain.has_mainloop_fusion:
        return False
    if chain.ops or chain.aux_tensors or chain.reductions or chain.quants or len(chain.output_specs) != 1:
        return False
    if not _block_scale_supported(chain):
        return False
    out = chain.output_specs[0]
    return (
        mm.a_major == mm.b_major == "k"
        and mm.out_dtype == out.dtype
        and mm.a_batch == mm.b_batch == mm.batch
        and out.source_ref == -1
        and out.major == "n"
        and out.stride in (None, (mm.M * mm.N, mm.N, 1))
    )


def _eligible_graph(chain: FusionChain, profile: "CostProfile") -> bool:
    """The strict gate, plus the relaxations this profile opted into."""
    if _simple_graph(chain):
        return True
    mm = chain.matmul
    if chain.has_moe or chain.is_multi_gemm or chain.has_mainloop_fusion:
        return False
    if chain.reductions or chain.quants or len(chain.output_specs) != 1:
        return False
    if not _block_scale_supported(chain):
        return False
    out = chain.output_specs[0]
    if not (mm.a_major == "k" and mm.a_batch == mm.b_batch == mm.batch and out.major == "n" and out.stride in (None, (mm.M * mm.N, mm.N, 1))):
        return False
    if mm.b_major != "k" and not (profile.b_n_major_ok and mm.b_major == "n"):
        return False
    if chain.ops or chain.aux_tensors:
        if not profile.scalar_scale_epilogue_ok:
            return False
        if not all(op.op == "mul" and op.aux is not None and op.parent_idx_b is None and op.parent_idx_c is None for op in chain.ops):
            return False
        if not all(getattr(t, "bcast_mode", None) == "scalar" for t in chain.aux_tensors):
            return False
        if out.source_ref < 0:
            return False
    elif not (mm.out_dtype == out.dtype and out.source_ref == -1):
        return False
    return True


def profile_for(chain: FusionChain, baseline: TileConfig, device: DeviceProperties) -> CostProfile | None:
    mm = chain.matmul
    key = (
        device.arch,
        baseline.pipeline,
        mm.accum_dtype,
        DTYPE_BITS[mm.a_dtype],
        DTYPE_BITS[mm.b_dtype],
        DTYPE_BITS[chain.output_dtype],
        chain.has_block_scale,
    )
    profile = _PROFILES.get(key)
    if profile is None or profile.device_name != device.name or not _eligible_graph(chain, profile):
        return None
    if not (profile.mn_bounds[0] <= mm.M <= profile.mn_bounds[1] and profile.mn_bounds[0] <= mm.N <= profile.mn_bounds[1]):
        return None
    if not (profile.k_bounds[0] <= mm.K <= profile.k_bounds[1] and mm.batch <= profile.max_batch):
        return None
    return profile


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def split_candidates(chain: FusionChain, config: TileConfig) -> range:
    if chain.matmul.accum_dtype != "fp32":
        return range(1, 2)
    k_tile = config.cta_tile_k_bytes * 8 // max(DTYPE_BITS[chain.matmul.a_dtype], DTYPE_BITS[chain.matmul.b_dtype])
    maximum = min(32, _ceil_div(chain.matmul.K, k_tile), 65535 // chain.matmul.batch)
    return range(1, maximum + 1)


_SMALL_M_BLOCK_SCALE_MAX_TOKENS = 32
_SMALL_M_WIDE_TILE_MAX_WEIGHT_TILES = 48

# The narrow-token geometry of the sm120 (CC 12.x, warp-MMA) block-scale pipeline on a part no cost profile above names (`_calibrated`).
# The sm120 catalog's smallest M tile is 32 rows, so one CTA row covers every decode batch of up to 32 tokens with the tokens left on M
# (no swap); the 2x4 warp grid is the measured form.  Measured on an RTX 5090 (compute capability 12.0, 170 SMs, 96 MiB L2) at boost
# clocks with no clock lock, cold-L2 CUDA-graph timing (3 rounds x 100 replays, A/B/A order; in-process round spread <= 0.8 % on every
# cited cell), M in {1, 6, 24}, against the sm120 auto pick, which `select_config` makes 128x128x128 for every block-scaled GEMM.  The
# percentages are of a 1,575 GB/s pin measured on the board with a 1 GiB device-memory add stream (read + write bytes counted; the GDDR7
# peak is 1,792 GB/s), so a weight stream can exceed 100 % of it.  NVFP4 (fp4_e2m1 x fp4_e2m1, e4m3 block-16 scales): N=34816 K=5120
# M=1 93.5 -> 62.1 us (68 -> 103 %), N=5120 K=17408 M=1 35.7 (split-K 4) -> 32.7 us (89 -> 98 %, split-K 4), N=248320 K=5120 M=1
# 639 -> 426 us (71 -> 107 %) -- the 128-row times from one sweep invocation, the 32-row times from a later one the same day (the
# third-party reference arms repeat within 0.6 % across the two).  At M=6 and M=24 the 32-row tile stays within 3.5 % of the best
# FlashInfer kernel on every cell (-3.0..+3.5 %), while that kernel is +2.4..+51 % faster than the 128-row auto pick (time ratio minus
# one; the +2.4 % is the K=17408 M=24 cell, which the auto split-K had already cut to 4 slices).  The split-K count follows the sm100
# rule (one wave of CTAs over the weight tiles, at most 16 slices): the measured N=5120 / K=17408 cell wants 4 slices, the two wide-N
# cells none.  Admitted but NOT measured: M other than 1, 6 and 24 (25-32 lie past the measured range), other (N, K), override-shape
# graphs (up to the 48-weight-tile hedge), and MXFP4 (e8m0 block-32 scales: the same 4-bit weight stream, admitted by width).  MXFP8
# (8-bit data) and the calibrated parts keep their picks.
_SM120_SMALL_M_BLOCK_SCALE_GEOMETRY = "CONFIG_sm120_32x128x128_16x16x32_cluster1x1_warps2x4"

# The narrow-token geometry of the sm120 DENSE fp8 pipeline on a part no cost profile above names, for the GEMMs whose weight tiles alone
# leave more than half the SMs idle (the shape needs split-K to fill the machine).  Not a catalog row -- the catalog's smallest sm120 M tile
# is 32 rows, so `by_name` synthesizes the 16-row tile from its canonical name: its 1x8 warp grid lays the eight compute warps along N, and
# one CTA row covers every decode batch of up to 16 tokens with the tokens left on M.  Measured on an RTX 5090 (compute capability 12.0,
# 170 SMs, 96 MiB L2) at boost clocks with no clock lock, cold-L2 CUDA-graph timing (3 rounds x 100 replays, A/B/A order; in-process round
# spread <= 1.2 % on every cited cell) in an explicit-tile search at M in {1, 2, 4, 6, 8} on the fp8_e4m3 x fp8_e4m3 -> bf16 GEMM N=5120
# K=6144 (40 weight tiles), percentages of the pin named above: this tile with 4 slices (160 CTAs, one 94 %-full wave) ran 21.33-21.59 us
# = 93-94 % at every M, +1.9..+2.8 % over the best 32-row tile of the search (32x128x128 with 4 slices, in its 2x4 or 1x8 warp grid) and
# +2.4..+3.2 % over the auto pick (32x64x128 warps2x4 with its own 2-slice split, 90-91 %); the same tile with a 128-byte K tile was the
# runner-up (92-93 %), and the planner path reproduced the pick in the decode-GEMM sweep (21.30 / 21.54 us at M = 1 / 6).  On the wide-N
# fp8 GEMMs of that sweep (N=16384 and N=14336: 128 and 112 weight tiles) the auto pick already streams at 97-99 % unsplit, so the rule
# declines every shape whose weight tiles alone fill more than half the SMs (`_SM120_SMALL_M_FP8_MIN_SLICES`), and -- like the family's own
# auto split-K -- every shallow-K GEMM.  Admitted but NOT measured: fp8_e5m2 and mixed e4m3 x e5m2 pairs (the same bytes per element,
# admitted by width), M 9-16, K other than 6144 (down to the 2048-element floor), N other than 5120 (every N up to 128 x (SMs // 2)
# columns is admitted), and override-shape graphs (up to the 48-weight-tile hedge).  Block-scaled data has its own rule above; 16-bit
# data and the calibrated parts keep their picks.
_SM120_SMALL_M_FP8_GEOMETRY = "CONFIG_sm120_16x128x64_16x16x32_cluster1x1_warps1x8"
_SM120_SMALL_M_FP8_MAX_TOKENS = 16  # the tile's rows: one CTA row covers the batch
_SM120_SMALL_M_FP8_MIN_SLICES = 2  # the rule applies where at least two slices fit in one wave beside the weight tiles
_SM120_SMALL_M_FP8_MIN_K_BYTES = 2048  # per row; the sm120 auto split-K's own depth floor (`sm120.compiler._auto_split_k`)
_FP8_DTYPES = frozenset({"fp8_e4m3", "fp8_e5m2"})


def _calibrated(device: DeviceProperties) -> bool:
    """Whether a cost profile above names this part (by its SM version and reported name).  The two sm120 small-M rules below apply only
    where this is False: today the only CC 12.x name in ``_PROFILES`` is the RTX PRO 6000 Blackwell Server Edition, so every other CC 12.x
    part -- the measured GeForce RTX 5090, the Workstation / Max-Q editions, the 84-SM parts, CC 12.1 -- takes them, measured or not.  A
    future profile for such a name retires both rules for that part on every dtype (intended: a fitted cost model outranks a rule)."""
    return any(key[0] == device.arch and profile.device_name == device.name for key, profile in _PROFILES.items())


def _one_wave_split_k(chain: FusionChain, config: TileConfig, device: DeviceProperties, weight_tiles: int) -> int:
    """The power-of-two split-K count that fills one wave of CTAs over ``weight_tiles`` weight tiles: at most 16 slices, never more than the
    K tiles (``split_candidates``), never past the SM count."""
    max_slices = min(16, max(split_candidates(chain, config)), device.sm_count // weight_tiles)
    return 1 << (max(1, max_slices).bit_length() - 1)


def _small_m_block_scale_config(chain: FusionChain, baseline: TileConfig, device: DeviceProperties, shapes_fixed: bool) -> TileConfig | None:
    """Swap decode tokens onto kernel N and split K within one CTA wave (sm100); on an uncalibrated CC 12.x part keep the tokens on M in
    the family's 32-row tile (``_SM120_SMALL_M_BLOCK_SCALE_GEOMETRY``) with the same one-wave split-K rule."""
    mm = chain.matmul
    sm120 = baseline.pipeline == "sm120" and family_for_arch(device.arch) == "sm120" and not _calibrated(device)
    if not (baseline.pipeline == "sm100" or sm120) or not chain.has_block_scale or not _block_scale_supported(chain):
        return None
    if sm120 and not (DTYPE_BITS[mm.a_dtype] == DTYPE_BITS[mm.b_dtype] == 4):
        return None  # only the fp4 x fp4 weight stream was measured on the sm120 family
    if chain.has_moe or chain.is_multi_gemm or chain.has_mainloop_fusion or chain.quants or chain.reductions or mm.batch != 1:
        return None
    if not 1 <= mm.M <= _SMALL_M_BLOCK_SCALE_MAX_TOKENS or mm.a_major != "k" or mm.b_major != "k" or mm.accum_dtype != "fp32":
        return None
    weight_tiles = _ceil_div(mm.N, 128)
    if shapes_fixed:
        width = next(w for w in (8, 16, 32) if w >= mm.M)
    elif weight_tiles <= _SMALL_M_WIDE_TILE_MAX_WEIGHT_TILES:
        width = 64
    else:
        return None
    if sm120:
        config = by_name(_SM120_SMALL_M_BLOCK_SCALE_GEOMETRY)
    else:
        config = by_name(f"CONFIG_sm100_128x{width}x{baseline.cta_tile_k_bytes}_128x{width}x{baseline.mma_tile_k_bytes}_cluster1x1_1ctamma")
    return replace(config, split_k_slices=_one_wave_split_k(chain, config, device, weight_tiles), swap_ab=not sm120)


def _sm120_small_m_fp8_config(chain: FusionChain, baseline: TileConfig, device: DeviceProperties, shapes_fixed: bool) -> TileConfig | None:
    """On an uncalibrated CC 12.x part, plan a dense fp8 GEMM of up to 16 tokens on the family's 16-row tile
    (``_SM120_SMALL_M_FP8_GEOMETRY``) with the one-wave split-K -- only where the weight tiles alone leave room for at least two slices."""
    mm = chain.matmul
    if baseline.pipeline != "sm120" or family_for_arch(device.arch) != "sm120" or _calibrated(device):
        return None
    if chain.has_block_scale or mm.a_dtype not in _FP8_DTYPES or mm.b_dtype not in _FP8_DTYPES:
        return None  # the dense fp8 x fp8 weight stream is what was measured; block-scaled data has its own rule above
    if chain.has_moe or chain.is_multi_gemm or chain.has_mainloop_fusion or chain.quants or chain.reductions or mm.batch != 1:
        return None
    if not 1 <= mm.M <= _SM120_SMALL_M_FP8_MAX_TOKENS or mm.a_major != "k" or mm.b_major != "k" or mm.accum_dtype != "fp32":
        return None
    if mm.K * DTYPE_BITS[mm.a_dtype] < _SM120_SMALL_M_FP8_MIN_K_BYTES * 8:
        return None  # a shallow K stays unsplit on this family (the auto pick does not split it either)
    config = by_name(_SM120_SMALL_M_FP8_GEOMETRY)
    weight_tiles = _ceil_div(mm.N, config.cta_tile_n)
    if not shapes_fixed and weight_tiles > _SMALL_M_WIDE_TILE_MAX_WEIGHT_TILES:
        return None  # an override-shape graph: the block-scale rule's weight-grid hedge
    slices = _one_wave_split_k(chain, config, device, weight_tiles)
    if slices < _SM120_SMALL_M_FP8_MIN_SLICES:
        return None  # the weight tiles alone fill more than half the SMs: the auto pick was measured at 97-99 % of the pin there
    return replace(config, split_k_slices=slices)


# The general block-scale templates (`kernel_registry.GENERAL_BLOCK_SCALE_GRAPH_TYPES`: linear scales, e.g. FP8 block-128 with fp32
# scales, applied to the accumulator by promotion warps) are planned by the rules below.  No cost profile claims these chains
# (`_block_scale_supported` admits only the block-scaled MMA kinds), and `select_config`'s geometry is wrong for them in three ways:
#   * a 256-column accumulator leaves one partial TMEM stage and moves the promotion into TMEM, so MMA, promotion and drain run in
#     series: 1.8-2.9x the time of a 128-column tile on every measured shape (dense and grouped);
#   * a CTA pair pads every expert of a grouped template to 256 rows (2-CTA / 1-CTA 1.02-1.5x; it never won a measured grouped
#     cell).  Re-measured once the promotion warps' per-window scale addressing was strength-reduced (an Int32 K block, one scale
#     line per tile): the 2x1 pair took 1.10-1.22x the time of the default plan of one NONE GEMM on either layer shape, and
#     1.03-1.09x the layer (that NONE FC1 then the default COMBINE, or the default gated FC1 then a NONE FC2), at 256 and 512
#     rows per expert (qwen3_30b and qwen3_235b at 4096 and 8192 tokens); one GATHER GEMM 1.64-1.77x.  A gated pair and a wide
#     quantize run on 1-CTA tiles only.  On the dense template the pair lost 1.2-1.5x at up to 64 rows, and now wins where its
#     rows fill whole pairs and its tiles more than one wave (the dense rules below);
#   * the grouped templates launch max_active_clusters(size) clusters with no fallback cluster, so a cluster size whose clusters do
#     not cover the part (4 and 8 CTAs on a 68-SM part: 64 SMs) idles the rest for the whole launch.
# So the planner's default non-swapped grouped plans are 1-CTA (the swapped COMBINE tile below pairs CTAs along the weight rows),
# its dense plans 1-CTA but the CTA pairs below, and every general plan has at most 128 accumulator columns (knobs still reach
# the rest).  Measured on a 68-SM SM100 part at the expert shapes of 14 MoE models (FP8 e4m3 data, 1x128 token and 128x128 weight
# scales, 1-8192 tokens), same-session A/B against the previous picks; the thresholds are unverified on other SM100 parts.


def _general_config(tile_m: int, tile_n: int, cluster_m: int = 1, cluster_n: int = 1, cta_group: int = 1, *, slices: int = 1, swap: bool = False) -> TileConfig:
    """A geometry of the general templates: 128-byte K tiles of 32-byte MMA instructions, 128-row MMA blocks."""
    config = by_name(f"CONFIG_sm100_{tile_m}x{tile_n}x128_{min(tile_m, 128)}x{tile_n}x32_cluster{cluster_m}x{cluster_n}_{cta_group}ctamma")
    return replace(config, split_k_slices=slices, swap_ab=swap)


_GENERAL_TILE_N = 128
# Grouped templates with fixed extents: the rows per expert (routed rows over groups) up to which the plan swaps the operands, so
# the weight rows fill the 128-row MMA and the expert's tokens a narrow N tile.  A swapped tile streams an expert's weights once
# per token tile its rows need -- the rows scatter around the mean, so at a mean equal to the tile width about half the experts
# need two -- and saves the MMA and promotion work the 128-row tile spends on padding rows.
#   * A gated pair (two GEMMs per tile, kernel_registry.general_glu_pair) saves twice that work: the 16-wide token tile up to 16
#     rows (up to 1 row at a K of _GENERAL_MOE_GLU_LONG_K_BYTES or more), the 32-wide one up to 32, the 64-wide one up to
#     _GENERAL_MOE_GLU_WIDE_ROWS.  The layer (GATHER + SwiGLU + FP8 out, then COMBINE) of the 14 models with the 16- / 32-wide
#     picks against the 128x128 1x4 tile it had: 1.085 / 1.077 / 1.068 / 1.042x faster at 1 / 8 / 64 / 512 tokens (up to 1.20x;
#     the fused FC1 up to 1.39x).  Re-measured on the current templates (FC1 in its layer, COMBINE at its pick): at 2-16 rows the
#     32-wide tile took 0.988-1.001x the layer of the 16-wide one at a K of 4096-8192 bytes (15 cells) and 1.007-1.012x at
#     2048-3072 (9 cells); at 1 row 1.003-1.146x at 1 token and 0.996-1.022x at 8.  At 40-57 rows the non-swapped 1x1 tile took
#     1.009-1.023x the layer of the 64-wide one (6 models), at 64 rows the 64-wide one 1.043-1.093x the 1x1 tile (4 models): the
#     rows past the tile width send part of the experts through a second weight stream.
#   * One GEMM (NONE, GATHER) takes a token tile of at least twice the mean rows: 16-wide up to 8 (up to 1 at a K of
#     _GENERAL_MOE_LONG_K_BYTES or more), 32-wide up to 16.  At a mean of 16 rows the 16-wide tile took 0.99-1.22x the time of the
#     non-swapped one and the 32-wide one 0.96-1.01x; at 32 rows the 32-wide tile took 1.02-1.10x.  Re-measured on the current
#     templates (NONE on both layer shapes, GATHER): at 2-8 rows the 32-wide tile took 0.990-1.000x the time of the 16-wide one
#     at a K of 2560-8192 bytes (24 cells) and 0.998-1.005x at 512-2048 (18 cells).
#   * COMBINE takes the 16-wide tile up to 1 row per expert, and with a K of at most _GENERAL_MOE_COMBINE_SHORT_K_BYTES up to 4;
#     past that, while rows x K bytes stay within _GENERAL_MOE_COMBINE_PAIR_ROW_K_BYTES, the 256x16 CTA pair (2x1, the 16-wide
#     token tile split between the pair).  The layer (GATHER + SwiGLU + FP8 out, then COMBINE, fp32 out) of 11 models at 8-512
#     tokens, against the non-swapped 128x128 1x1 tile: 16-wide at 1 row 0.86-0.93x (N <= 4096; 1.003-1.005x at N = 6144-7168,
#     K >= 2048), at 2-4 rows 0.980-0.995x for K <= 768 and 0.999-1.003x for K >= 1024; the CTA pair at 5-16 rows and K <= 768
#     0.963-0.994x (0.96-1.00x the 16-wide tile), and 1.023x at 16 rows and K = 768, which the rows x K bound leaves 1x1.  Every
#     pick was within 0.5 % of the fastest of 128x128, 128x16, 128x32 and the 256x16 / 256x32 pairs.  Re-measured on the current
#     templates: the 256x32 pair was the fastest COMBINE in no cell of the 14 models at 1-512 tokens; at 1 row and a K of 2048
#     bytes or more the non-swapped 1x2 tile took 0.980-1.005x the layer of the 16-wide one at 1-8 tokens (7 models) but 1.010x
#     at 64 tokens (dsv4_pro), and the 16-wide tile stays.
# At 1 token and an N of 4096-7168, a swapped single GEMM (NONE, COMBINE) took up to 1.044x the time of the 1x4 tile it had (most
# cells 1.01-1.03x), which the FC1 of the same layer more than repays.  (COMBINE with evict_first weight loads, N of 4096-7168:
# 1.011-1.014x the layer of the 1x4 tile.)  Re-measured for COMBINE at 1 row per expert on the 9 models with N >= 4096 (the layer
# at 1 / 2 / 4 / 8 tokens, the non-swapped 128x128 1x4 tile against the 16-wide one): 0.982-0.991x at 1 token and 0.988-1.000x at
# 2-8 on 7 models, but 1.003-1.013x on dsv4_pro (K = 3072 bytes, 384 experts) and 1.014x / 1.003x at 1 / 2 tokens on qwen35_397b
# (K = 1024, 512 experts); the 1x2 tile 0.995-1.006x and 1.015x / 1.005x there.  The 16-wide tile stays.
_GENERAL_MOE_SWAP_TILES = ((8, 16), (16, 32))  # (max rows per expert, token tile)
_GENERAL_MOE_LONG_K_BYTES = 2560
_GENERAL_MOE_LONG_K_SWAP_TILES = ((1, 16), (16, 32))
_GENERAL_MOE_GLU_WIDE_ROWS = 56
_GENERAL_MOE_GLU_SWAP_TILES = ((16, 16), (32, 32), (_GENERAL_MOE_GLU_WIDE_ROWS, 64))
_GENERAL_MOE_GLU_LONG_K_BYTES = 4096
_GENERAL_MOE_GLU_LONG_K_SWAP_TILES = ((1, 16), (32, 32), (_GENERAL_MOE_GLU_WIDE_ROWS, 64))
_GENERAL_MOE_COMBINE_SWAP_ROWS = 1
_GENERAL_MOE_COMBINE_SHORT_K_BYTES = 768
_GENERAL_MOE_COMBINE_SHORT_K_SWAP_ROWS = 4
_GENERAL_MOE_COMBINE_PAIR_ROW_K_BYTES = 8192
_GENERAL_MOE_SWAP_MODES = frozenset({"none", "gather"})
# Grouped templates, non-swapped: a plan pairs CTAs along N (1x2: one token tile multicast to two weight tiles) where the N tiles
# pair up, two-CTA clusters cover the part, and the mode pays for it.  Same-session sweeps of the 14 models on the current
# templates, 1x2 / 1x1:
#   * GATHER never pairs: 0.998-1.009 for one GEMM at 512-4096 tokens (34 cells), 0.998-1.010 for the gated FC1 in its layer at
#     512-4096 tokens (46 cells), 0.993-1.071 at 1-8 tokens (earlier template revisions measured 0.984 at 4096 tokens).
#   * NONE pairs from _GENERAL_MOE_PAIR_MIN_ROWS rows per expert -- whole 128-row token tiles -- at a K of at least
#     _GENERAL_MOE_PAIR_MIN_K_BYTES: there 0.971-1.002 (14 cells at 2048-4096 tokens); below those rows 0.999-1.006 (40 cells at
#     512-4096 tokens), and at a K of at most 1024 bytes 0.993-1.202 (14 cells; 1.020-1.202 at K = 512-768).
#   * SCATTER pairs (not re-measured).
#   * A COMBINE of fixed extents stays 1x1: 1x2 / 1x1 = 0.975-1.020 from 5 to 256 rows per expert over 10 models (1x2 ahead by
#     up to 2.5 % at 5-16 rows and an N of 3072 or more, behind by up to 2 % at N = 2048 and by 0.5-1.8 % on six models at 4096
#     tokens), 1x4 / 1x1 = 1.024-1.047; re-measured in the layer 0.992-1.062 at 512-4096 tokens (the highest at K = 512-640).
# A 1x1 token box of the sorted templates (NONE, COMBINE) loads at that speed through the host's whole-tensor token map, whose
# size-class bit the encoder sets (a per-group rebased map with that bit cleared took 1.2-1.3x the time).  Under override or
# dynamic shapes every mode but GATHER pairs, since the graph may run at decode sizes, where small token tensors load slowly in a
# 1x1 box: NONE 1x2 / 1x1 = 0.72-1.02 at 1-8 tokens (K of 768-8192 bytes), COMBINE 0.85-1.08.  The 4096-token layer: 1.076x
# faster than the previous 2-CTA / 256-column / 1x4 picks (up to 1.35x).
_GENERAL_MOE_PAIR_MIN_ROWS = 128
_GENERAL_MOE_PAIR_MIN_K_BYTES = 1536
# Dense template.  Up to _GENERAL_DENSE_SMALL_M rows, while the tiles (the M tiles of every batch times the N tiles) leave room for
# a second CTA per SM -- with two M tiles or a batch past that, 64-column tiles took 1.07-1.43x the time of 128-column ones -- and
# only at the extents the graph declares:
#   * a K of at most _GENERAL_DENSE_NARROW_K_BYTES per row takes 64-column tiles (twice the CTAs; at 1536 bytes they were 14-21 %
#     slower), except under a wide quantize (one 128-column block per tile) or an N-major weight (128-column swizzle groups);
#   * a deeper K, from _GENERAL_DENSE_SPLIT_MIN_K_BYTES, splits into the largest power of two of at most _GENERAL_DENSE_MAX_SLICES
#     slices that keeps the CTAs within one wave and each slice at two or more K tiles.  Two slices pay from
#     _GENERAL_DENSE_TWO_SLICES_MIN_K_BYTES while the unsplit grid fills at most 2/5 of the SMs (24 tiles at K = 4096: 0.86-0.91x
#     the time of one slice at 1-64 rows) and from _GENERAL_DENSE_TWO_SLICES_FILLED_MIN_K_BYTES above that (32 tiles: 1.01-1.10x
#     at K = 4096, 0.92-0.97x at K = 7168-8192).  Where the split stops at two K tiles per slice, 64-column tiles at half the
#     slices run as many CTAs with twice the K each (N = 1024, K = 2048: 4-9.5 % faster at 1-128 rows).
# An override- or dynamic-shape graph runs other M than it declares: it takes the plan of more rows, which does not depend on M
# and needs no workspace (split K needs one that grows with M; declared at M = 1, the split and 64-column picks took 1.25-1.44x
# the time of that plan at M = 512-4096).
# A 1-CTA plan's CTAs pair along N in the widest of 1x4 / 1x2 clusters the N tiles fill: the token tile is multicast (at 1-8 rows
# the 128-row token box otherwise streams mostly out-of-bounds zero rows: 1x1 took 1.10-1.11x the time of the 1x4 pick).  The
# dense template launches its leftover SMs as fallback clusters, so the cluster size idles none.  The 140 cells of the 14 models
# (M = 1 / 8 / 64 / 512 / 4096): 1.35 / 1.34 / 1.31 / 1.70 / 1.74x faster than the previous picks (up to 2.8x), 0.98x the time of
# the best config of a 1x1 / 2x1 sweep, and 1.006x the best of every config timed in the same session (at most 1.08x).
# Past _GENERAL_DENSE_SMALL_M rows at the extents the graph declares, the chains the following was measured on -- one GEMM of
# K-major operands, batch 1, no output quantize -- take, since the promotion warps' scale addressing was strength-reduced:
#   * the 2x1 CTA pair (2-CTA MMA, 128x128 per CTA: a 256x128 tile whose weight tile each CTA loads half of) from
#     _GENERAL_DENSE_PAIR_MIN_M rows at a K of at least _GENERAL_DENSE_PAIR_MIN_K_BYTES, with at least
#     _GENERAL_DENSE_PAIR_MIN_N_TILES N tiles, an even count of 128-row tiles (the rows fill whole pairs) and more tiles than SMs;
#   * elsewhere, at a K of at least _GENERAL_DENSE_TWO_CLUSTER_MIN_K_BYTES, the 1x2 cluster.
# Swept against the 1x4 tile they had (258 cells: the 24 dense shapes of the 14 models at M = 256-8192, rows that leave a pair half
# empty at M = 384-4224, K of 1024-2048 bytes at N = 2048-7168): the pair 0.883-1.002x the time (geomean 0.944, 85 cells), the 1x2
# cluster 0.931-1.010x (geomean 0.972, 105 cells).  Outside its bounds the pair lost: a half-empty pair that adds a wave of pairs
# up to 1.34x (M = 640 / 1152 / 2176), one wave of tiles 1.12x (M = 512, N = 2048), a K of 1024-1792 bytes at 16 N tiles up to
# 1.07x, 8-12 N tiles up to 1.06x; the 2x2 pair won more at 2048-4096 rows but took up to 1.05x at 8192, the 2x4 pair up to 1.5x.
# A K of 512-1280 bytes keeps 1x4 (1x2 up to 1.05x at 2048 rows), and so does an FP8 block-128 output (1x2 up to 1.08x).  Through
# the planner, old and new rule in one session on the 24 shapes at M = 256-8192 (and 640, 2176): 0.954x the time where the plan
# changed (126 cells, 0.883-1.006x; per model 0.93-1.00x at 512-8192 rows), with the bytes of the old plans.
_GENERAL_DENSE_SMALL_M = 256
_GENERAL_DENSE_NARROW_K_BYTES = 1024
_GENERAL_DENSE_SPLIT_MIN_K_BYTES = 2048
_GENERAL_DENSE_TWO_SLICES_MIN_K_BYTES = 3072
_GENERAL_DENSE_TWO_SLICES_FILLED_MIN_K_BYTES = 6144
_GENERAL_DENSE_MAX_SLICES = 8
_GENERAL_DENSE_MIN_K_TILES_PER_SLICE = 2
_GENERAL_DENSE_PAIR_MIN_M = 512
_GENERAL_DENSE_PAIR_MIN_K_BYTES = 2048
_GENERAL_DENSE_PAIR_MIN_N_TILES = 16
_GENERAL_DENSE_TWO_CLUSTER_MIN_K_BYTES = 1536


def is_general_block_scale(chain: FusionChain) -> bool:
    """Whether ``chain`` runs a general block-scale template (dense or grouped)."""
    return chain.has_block_scale and classify_graph_type(chain) in GENERAL_BLOCK_SCALE_GRAPH_TYPES


def _current_occupancy(device: DeviceProperties) -> tuple[int, ...] | None:
    """The current device's cluster occupancy table when ``device`` describes it, else None."""
    from cudnn.frost import device as frost_device

    try:
        if not frost_device.is_available():
            return None
        ordinal = frost_device.resolve_device(None)
        if (frost_device.device_name(ordinal), frost_device.multiprocessor_count(ordinal)) != (device.name, device.sm_count):
            return None
        from cudnn.frost.occupancy import occupancy_map

        return tuple(occupancy_map(ordinal))
    except (RuntimeError, NotImplementedError, ValueError):
        return None


def _clusters_cover_sms(device: DeviceProperties | None, cluster_size: int) -> bool:
    """Whether the clusters of ``cluster_size`` CTAs the device runs at once keep every SM busy.  Without an occupancy table (no
    device, or a described part other than the current one), clusters of up to two CTAs are taken to; larger ones are not."""
    table = None if device is None else device.occupancy if device.occupancy is not None else _current_occupancy(device)
    if table is None or len(table) < cluster_size:
        return cluster_size <= 2
    return table[cluster_size - 1] * cluster_size >= device.sm_count


def _general_moe_pairs(chain: FusionChain, shapes_fixed: bool) -> bool:
    """Whether the non-swapped plan of a grouped general chain pairs its CTAs along N (see the rules above)."""
    mode = chain.moe.mode
    if mode == "gather":
        return False
    if not shapes_fixed:
        return True  # it may run at decode sizes, where a 1x1 box of a small token tensor loads slowly
    if mode == "none":
        rows = _ceil_div(chain.matmul.M, chain.moe.num_groups)  # routed rows per group (expert)
        k_bytes = chain.matmul.K * DTYPE_BITS[chain.matmul.a_dtype] // 8
        return rows >= _GENERAL_MOE_PAIR_MIN_ROWS and k_bytes >= _GENERAL_MOE_PAIR_MIN_K_BYTES
    return mode == "scatter"


def _general_moe_baseline(chain: FusionChain, device: DeviceProperties | None, shapes_fixed: bool) -> TileConfig:
    """The non-swapped plan of a grouped general chain: 128x128, 1-CTA, a 1x2 cluster where it pays and covers the part."""
    n_tiles = _ceil_div(chain.matmul.N, _GENERAL_TILE_N)
    pair = _general_moe_pairs(chain, shapes_fixed)
    cluster_n = 2 if pair and n_tiles % 2 == 0 and _clusters_cover_sms(device, 2) else 1
    return _general_config(128, _GENERAL_TILE_N, 1, cluster_n)


def _general_dense_cluster_n(n_tiles: int) -> int:
    return 4 if n_tiles % 4 == 0 else 2 if n_tiles % 2 == 0 else 1


def _general_dense_baseline(chain: FusionChain, sm_count: int, *, split_k: bool, shapes_fixed: bool) -> TileConfig:
    """The plan of a dense general chain (see the rules above): past the small-M rows, the CTA pair or the 1x2 cluster where they
    were measured to pay, else the 1-CTA plan.  ``split_k`` is whether the chain may split K at all, ``shapes_fixed`` whether it
    runs the M it declares."""
    mm = chain.matmul
    if shapes_fixed and mm.M > _GENERAL_DENSE_SMALL_M and _general_dense_measured(chain):
        k_bytes = mm.K * DTYPE_BITS[mm.a_dtype] // 8
        m_tiles, n_tiles = _ceil_div(mm.M, 128), _ceil_div(mm.N, _GENERAL_TILE_N)
        if (
            mm.M >= _GENERAL_DENSE_PAIR_MIN_M
            and m_tiles % 2 == 0
            and k_bytes >= _GENERAL_DENSE_PAIR_MIN_K_BYTES
            and n_tiles >= _GENERAL_DENSE_PAIR_MIN_N_TILES
            and m_tiles * n_tiles > sm_count
        ):
            return _general_config(128, _GENERAL_TILE_N, 2, 1, 2)
        if k_bytes >= _GENERAL_DENSE_TWO_CLUSTER_MIN_K_BYTES and n_tiles % 2 == 0:
            return _general_config(128, _GENERAL_TILE_N, 1, 2)
    return _general_dense_one_cta(chain, sm_count, split_k=split_k, shapes_fixed=shapes_fixed)


def _general_dense_measured(chain: FusionChain) -> bool:
    """Whether a dense general chain is of the class the CTA-pair and 1x2 rules were measured on: one GEMM of K-major operands,
    batch 1, no output quantize (a wide quantize and a gated pair also run on 1-CTA tiles only, and an 8-bit N-major weight
    fills half a swizzle group per CTA of a pair)."""
    mm = chain.matmul
    return not chain.is_multi_gemm and not chain.quants and mm.batch == 1 and mm.a_major == "k" and mm.b_major == "k"


def _general_dense_one_cta(chain: FusionChain, sm_count: int, *, split_k: bool, shapes_fixed: bool) -> TileConfig:
    """The 1-CTA plan of a dense general chain: the small-M narrow-tile and split-K rules, the widest 1x4 / 1x2 cluster."""
    mm = chain.matmul
    k_bytes = mm.K * DTYPE_BITS[mm.a_dtype] // 8
    m_tiles = mm.batch * _ceil_div(mm.M, 128)
    n_tiles = _ceil_div(mm.N, _GENERAL_TILE_N)
    tile_n, slices = _GENERAL_TILE_N, 1
    if shapes_fixed and mm.M <= _GENERAL_DENSE_SMALL_M and 2 * m_tiles * n_tiles <= sm_count:
        # A wide quantize needs one 128-column block per tile; an N-major weight a whole 128-column swizzle group.
        narrow_ok = wide_quant(chain) is None and mm.b_major == "k"
        if k_bytes <= _GENERAL_DENSE_NARROW_K_BYTES and narrow_ok:
            tile_n = 64
        elif split_k and k_bytes >= _GENERAL_DENSE_SPLIT_MIN_K_BYTES:
            k_tiles = _ceil_div(k_bytes, 128)
            while (
                slices < _GENERAL_DENSE_MAX_SLICES
                and 2 * slices * m_tiles * n_tiles <= sm_count
                and k_tiles // (2 * slices) >= _GENERAL_DENSE_MIN_K_TILES_PER_SLICE
            ):
                slices *= 2
            if slices == 2:
                # Two slices pay from a shorter K when the unsplit grid leaves most SMs idle.
                filled = 5 * m_tiles * n_tiles > 2 * sm_count
                if k_bytes < (_GENERAL_DENSE_TWO_SLICES_FILLED_MIN_K_BYTES if filled else _GENERAL_DENSE_TWO_SLICES_MIN_K_BYTES):
                    slices = 1
            elif slices > 2 and k_tiles // slices == _GENERAL_DENSE_MIN_K_TILES_PER_SLICE and narrow_ok:
                # The slices stopped at the per-slice K floor: as many CTAs as 64-column tiles, with twice the K each.
                tile_n, slices = 64, slices // 2
    return _general_config(128, tile_n, 1, _general_dense_cluster_n(_ceil_div(mm.N, tile_n)), slices=slices)


def general_block_scale_baseline(chain: FusionChain, device: DeviceProperties | None, *, split_k: bool, shapes_fixed: bool) -> TileConfig:
    """The non-swapped plan of a general block-scale chain: the engine's pick wherever the planner does not lead with a swapped
    grouped plan (dynamic shapes, no device, override shapes, many rows per expert).  ``split_k``: the dense chain may split K;
    ``shapes_fixed``: the graph runs the token count (dense: the M) it declares."""
    if chain.has_moe:
        return _general_moe_baseline(chain, device, shapes_fixed)
    from .tile_config import _sm_count

    return _general_dense_baseline(chain, device.sm_count if device is not None else _sm_count(), split_k=split_k, shapes_fixed=shapes_fixed)


def _general_moe_swap_pick(chain: FusionChain, shapes_fixed: bool) -> TileConfig | None:
    """The swapped plan a grouped general chain with few rows per expert leads with; None where the non-swapped plan leads."""
    if not shapes_fixed or not chain.has_moe:
        return None  # an override-shape graph runs other token counts than it declares: the non-swapped plan serves them all
    rows = _ceil_div(chain.matmul.M, chain.moe.num_groups)  # routed rows per group (expert)
    if chain.moe.mode == "combine":
        k_bytes = chain.matmul.K * DTYPE_BITS[chain.matmul.a_dtype] // 8
        short_k = k_bytes <= _GENERAL_MOE_COMBINE_SHORT_K_BYTES
        if rows <= _GENERAL_MOE_COMBINE_SWAP_ROWS or (short_k and rows <= _GENERAL_MOE_COMBINE_SHORT_K_SWAP_ROWS):
            return _general_config(128, 16, swap=True)
        if short_k and rows * k_bytes <= _GENERAL_MOE_COMBINE_PAIR_ROW_K_BYTES:
            return _general_config(256, 16, 2, 1, 2, swap=True)
        return None
    if chain.moe.mode in _GENERAL_MOE_SWAP_MODES:
        k_bytes = chain.matmul.K * DTYPE_BITS[chain.matmul.a_dtype] // 8
        if general_glu_pair(chain):
            tiles = _GENERAL_MOE_GLU_LONG_K_SWAP_TILES if k_bytes >= _GENERAL_MOE_GLU_LONG_K_BYTES else _GENERAL_MOE_GLU_SWAP_TILES
        else:
            tiles = _GENERAL_MOE_LONG_K_SWAP_TILES if k_bytes >= _GENERAL_MOE_LONG_K_BYTES else _GENERAL_MOE_SWAP_TILES
        width = next((tile for max_rows, tile in tiles if rows <= max_rows), None)
    else:
        width = None
    return None if width is None else _general_config(128, width, swap=True)


def _general_block_scale_candidates(
    chain: FusionChain, baseline: TileConfig, device: DeviceProperties, *, shapes_fixed: bool, shortlist: bool
) -> list[TileConfig]:
    """The ordered candidates of a general chain: the swapped pick, the baseline, the 1-CTA plan a dense chain takes where the
    probe declines its CTA pair, the 128x128 1x1 plan every general chain the templates serve runs (a wide quantize included),
    then -- for a shortlist -- the geometries an autotuner should time."""
    pick = _general_moe_swap_pick(chain, shapes_fixed)
    one_cta = baseline
    if not chain.has_moe and baseline.is_cta_pair_mma:
        one_cta = _general_dense_one_cta(chain, device.sm_count, split_k=False, shapes_fixed=shapes_fixed)
    fallback = _general_config(128, _GENERAL_TILE_N)
    out = [config for config in (pick, baseline, one_cta, fallback) if config is not None]
    if not shortlist:
        return out
    mm = chain.matmul
    if chain.has_moe:
        if shapes_fixed:
            out += [_general_config(128, width, swap=True) for width in (16, 32, 64)]
            # CTA pairs along the weight rows; a wide quantize declines them
            out += [_general_config(256, width, 2, 1, 2, swap=True) for width in (16, 32)]
        out += [_general_config(128, _GENERAL_TILE_N, 1, cluster_n) for cluster_n in (1, 2, 4)]
        return out
    tile_n, cluster_n, slices = one_cta.cta_tile_n, one_cta.cga_size_n, one_cta.split_k_slices
    if mm.M <= _GENERAL_DENSE_SMALL_M:
        # The other slice counts within two waves (the probe drops them where the chain cannot split K).
        tiles = mm.batch * _ceil_div(mm.M, 128) * _ceil_div(mm.N, tile_n)
        k_tiles = _ceil_div(mm.K * DTYPE_BITS[mm.a_dtype] // 8, 128)
        for other in (1, 2, 4, 8):
            if other != slices and other * tiles <= 2 * device.sm_count and k_tiles // other >= _GENERAL_DENSE_MIN_K_TILES_PER_SLICE:
                out.append(_general_config(128, tile_n, 1, cluster_n, slices=other))
    out += [_general_config(128, tile_n, 1, other, slices=slices) for other in (1, 2, 4) if other != cluster_n]
    out.append(_general_config(128, 64 if tile_n == _GENERAL_TILE_N else _GENERAL_TILE_N))
    if mm.M > _GENERAL_DENSE_SMALL_M:
        # CTA pairs, 2x1 and 2x2 (a wide quantize, a gated pair and an 8-bit N-major weight decline them)
        out += [_general_config(128, _GENERAL_TILE_N, 2, cluster, 2) for cluster in (1, 2)]
    return out


def _select_general_block_scale(
    chain: FusionChain, baseline: TileConfig, *, probe: Callable[[FusionChain, TileConfig], None], device: DeviceProperties, limit: int, shapes_fixed: bool
) -> list[TileConfig]:
    """The probed, de-duplicated candidates of a general chain, first choice first.  When none runs, the baseline's decline."""
    selected: list[TileConfig] = []
    declines: dict[TileConfig, Exception] = {}
    for config in _general_block_scale_candidates(chain, baseline, device, shapes_fixed=shapes_fixed, shortlist=limit > 1):
        if config in selected or config in declines:
            continue
        try:
            probe(chain, config)
        except (NotImplementedError, ValueError) as exc:
            declines[config] = exc
            continue
        selected.append(config)
        if len(selected) == limit:
            break
    if not selected:
        raise declines.get(baseline) or next(iter(declines.values()))
    return selected


def candidate_configs(chain: FusionChain, baseline: TileConfig, device: DeviceProperties, profile: CostProfile) -> list[TileConfig]:
    candidates = {baseline: None}
    for swapped, view in ((False, chain), (True, swap_ab(chain))):
        mm = view.matmul
        if swapped and mm.a_major != "k":
            continue
        family = {"_pipeline": "sm120", "_batch": mm.batch} if baseline.pipeline == "sm120" else {}
        auto = (
            select_config(
                mm.M,
                mm.N,
                1,
                K=mm.K,
                block_scale=view.has_block_scale,
                b_n_major=mm.b_major == "n",
                b_elem_bytes=max(1, DTYPE_BITS[mm.b_dtype] // 8),
                sm_count=device.sm_count,
                **family,
            )
            if swapped
            else replace(baseline, split_k_slices=1)
        )
        for geometry in (auto, *(by_name(name) for name in profile.extra_geometries)):
            geometry = as_mma_tile_k(as_pipeline(geometry, baseline.pipeline), baseline.mma_tile_k_bytes)
            for slices in split_candidates(view, geometry):
                candidates[replace(geometry, split_k_slices=slices, swap_ab=swapped)] = None
    return list(candidates)


def cost_features(view: FusionChain, config: TileConfig, device: DeviceProperties) -> CostFeatures:
    """Features in kernel coordinates; apply swap_ab before calling."""
    mm = view.matmul
    slices = config.split_k_slices
    cluster_m, cluster_n = config.cga_tile_mn
    clusters = mm.batch * _ceil_div(mm.M, cluster_m) * _ceil_div(mm.N, cluster_n) * slices
    resident_clusters = max(1, device.sm_count // (config.cga_size_m * config.cga_size_n))
    waves = _ceil_div(clusters, resident_clusters)
    k_tile = config.cta_tile_k_bytes * 8 // max(DTYPE_BITS[mm.a_dtype], DTYPE_BITS[mm.b_dtype])
    k_steps = waves * _ceil_div(_ceil_div(mm.K, k_tile), slices)
    area = config.cta_tile_m * config.cta_tile_n / (128 * 128)
    input_mb = mm.K * (mm.a_batch * mm.M * DTYPE_BITS[mm.a_dtype] + mm.b_batch * mm.N * DTYPE_BITS[mm.b_dtype]) / 8e6
    scale_input_mb = scale_stage_kb = 0
    bs = view.block_scale
    if bs is not None:
        for batch, rows, tile_rows, block, dtype, fake in (
            (mm.a_batch, mm.M, config.cta_tile_m, bs.block_size_a[-1], bs.sf_dtype_a, bs.fake_dequant_a),
            (mm.b_batch, mm.N, config.cta_tile_n, bs.block_size_b[0], bs.sf_dtype_b, bs.fake_dequant_b),
        ):
            if not fake:
                scale_bytes = DTYPE_BITS[dtype] / 8
                scale_input_mb += batch * _ceil_div(rows, 128) * 128 * _ceil_div(_ceil_div(mm.K, block), 4) * 4 * scale_bytes / 1e6
                scale_stage_kb += k_steps * tile_rows * (k_tile // block) * scale_bytes / 1024
    partial_bytes = 4 * slices * mm.batch * mm.M * mm.N if slices > 1 else 0
    groups = 1 if slices <= 4 else 2 if slices <= 8 else 4
    reduce_elems = 4
    while mm.N % reduce_elems:
        reduce_elems //= 2
    reduce_n = 32 // groups * reduce_elems
    reduce_waves = _ceil_div(mm.batch * _ceil_div(mm.M, 4) * _ceil_div(mm.N, reduce_n), device.sm_count * 4) if slices > 1 else 0
    return CostFeatures(
        launch=1,
        waves=waves,
        k_steps=k_steps,
        mma_work=k_steps * area,
        input_mb=input_mb,
        partial_cache_mb=2 * min(partial_bytes, device.l2_bytes) / 1e6,
        partial_spill_mb=2 * max(0, partial_bytes - device.l2_bytes) / 1e6,
        reduce_launch=int(slices > 1),
        reduce_steps=reduce_waves * _ceil_div(slices, groups),
        m_store=waves * area if view.out_major == "m" and slices == 1 else 0,
        m_reduce=2 * mm.batch * mm.M * mm.N / 1e6 if view.out_major == "m" and slices > 1 else 0,
        scale_input_mb=scale_input_mb,
        scale_stage_kb=scale_stage_kb,
    )


def estimate_us(view: FusionChain, config: TileConfig, device: DeviceProperties, profile: CostProfile) -> float:
    return sum(x * w for x, w in zip(cost_features(view, config, device), profile.coefficients))


def rank_candidates(chain: FusionChain, configs: list[TileConfig], device: DeviceProperties, profile: CostProfile) -> list[tuple[float, TileConfig]]:
    views = {False: chain, True: swap_ab(chain)}
    return sorted(((estimate_us(views[config.swap_ab], config, device, profile), config) for config in configs), key=lambda pair: pair[0])


def select_strategies(
    chain: FusionChain,
    baseline: TileConfig,
    *,
    probe: Callable[[FusionChain, TileConfig], None],
    device: DeviceProperties | None = None,
    limit: int = MAX_PLAN_CONFIGS,
    shapes_fixed: bool = False,
) -> list[TileConfig]:
    """Rank supported plans with a stable first choice and a bounded shortlist."""
    if not 1 <= limit <= MAX_PLAN_CONFIGS:
        raise ValueError(f"strategy limit must be between 1 and {MAX_PLAN_CONFIGS}")
    device = current_device_properties() if device is None else device
    if device is None:
        return [baseline]
    if baseline.pipeline == "sm100" and is_general_block_scale(chain):
        return _select_general_block_scale(chain, baseline, probe=probe, device=device, limit=limit, shapes_fixed=shapes_fixed)
    small_m = _small_m_block_scale_config(chain, baseline, device, shapes_fixed)
    if small_m is None:
        small_m = _sm120_small_m_fp8_config(chain, baseline, device, shapes_fixed)
    if small_m is not None:
        try:
            probe(chain, small_m)
        except (NotImplementedError, ValueError):
            small_m = None
        else:
            # The parts where the narrow-token plan is the measured first choice: B200 (the swap-AB form) and an uncalibrated CC 12.x
            # part (the 32-row block-scale form, the 16-row fp8 form); Rubin keeps its incumbent and is offered the candidate second
            # (``include_small_m``).
            if device.arch == 100 or small_m.pipeline == "sm120":
                selected = [small_m]
                if limit > 1 and baseline != small_m:
                    try:
                        probe(chain, baseline)
                    except (NotImplementedError, ValueError):
                        pass
                    else:
                        selected.append(baseline)
                return selected

    def include_small_m(configs: list[TileConfig]) -> list[TileConfig]:
        """Include the probed narrow-token candidate without replacing the incumbent."""
        if small_m is None or small_m in configs or limit == 1:
            return configs
        return [configs[0], small_m, *configs[1:]][:limit]

    profile = profile_for(chain, baseline, device)
    if profile is None:
        return include_small_m([baseline])
    ranked = rank_candidates(chain, candidate_configs(chain, baseline, device, profile), device, profile)
    checked: dict[TileConfig, bool] = {}

    def accepted(config: TileConfig) -> bool:
        if config not in checked:
            try:
                probe(chain, config)
            except (NotImplementedError, ValueError):
                checked[config] = False
            else:
                checked[config] = True
        return checked[config]

    best = next(((cost, config) for cost, config in ranked if accepted(config)), None)
    if best is None:
        if baseline.pipeline == "sm120":
            probe(chain, baseline)
        raise NotImplementedError("frost_gemm: no supported joint strategy candidate")
    best_cost, _ = best
    baseline_cost = next(cost for cost, config in ranked if config == baseline)
    if baseline_cost <= best_cost * (1 + profile.switch_margin) and accepted(baseline):
        first = baseline
    else:
        tied = [(cost, config) for cost, config in ranked if cost <= best_cost * (1 + profile.tie_margin) and accepted(config)]
        first = min(tied, key=lambda pair: (pair[1].swap_ab, pair[1].split_k_slices, pair[0]))[1]
    selected = [first]
    if limit == 1:
        return selected
    for _, config in ranked:
        if len(selected) >= max(1, limit - 2):
            break
        if config not in selected and accepted(config):
            selected.append(config)
    represented = {(replace(config, split_k_slices=1), config.split_k_slices > 1) for config in selected}
    for cost, config in ranked:
        if len(selected) >= limit or cost > best_cost * (1 + profile.diversity_margin):
            break
        strategy = (replace(config, split_k_slices=1), config.split_k_slices > 1)
        if strategy not in represented and accepted(config):
            selected.append(config)
            represented.add(strategy)
    for _, config in ranked:
        if len(selected) >= limit:
            break
        if config not in selected and accepted(config):
            selected.append(config)
    return include_small_m([first, *(config for _, config in ranked if config != first and config in selected)])


def select_strategy(
    chain: FusionChain,
    baseline: TileConfig,
    *,
    probe: Callable[[FusionChain, TileConfig], None],
    device: DeviceProperties | None = None,
    shapes_fixed: bool = False,
) -> TileConfig:
    """Return the first supported strategy using the same policy as the shortlist."""
    return select_strategies(chain, baseline, probe=probe, device=device, limit=1, shapes_fixed=shapes_fixed)[0]
