# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-aware joint geometry, split-K and orientation selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import NamedTuple

from .arch_family import family_for_arch
from .dtypes import DTYPE_BITS
from .fusion_ir import FusionChain, swap_ab
from .tile_config import PLANNER_K_BOUNDS, PLANNER_MAX_BATCH, PLANNER_MN_BOUNDS, TileConfig, as_mma_tile_k, as_pipeline, by_name, select_config

MAX_PLAN_CONFIGS = 8


@dataclass(frozen=True)
class DeviceProperties:
    arch: int
    name: str
    sm_count: int
    l2_bytes: int


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
