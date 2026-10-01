# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-aware joint geometry, split-K and orientation selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import NamedTuple

from .dtypes import DTYPE_BITS
from .fusion_ir import FusionChain, swap_ab
from .tile_config import TileConfig, as_mma_tile_k, as_pipeline, by_name, select_config

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
    mn_bounds: tuple[int, int] = (32, 4096)
    k_bounds: tuple[int, int] = (1024, 16384)
    max_batch: int = 8


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
}


def _simple_graph(chain: FusionChain) -> bool:
    mm = chain.matmul
    if chain.has_moe or chain.is_multi_gemm or chain.has_mainloop_fusion:
        return False
    if chain.ops or chain.aux_tensors or chain.reductions or chain.quants or len(chain.output_specs) != 1:
        return False
    bs = chain.block_scale
    if bs is not None and (
        bs.fake_dequant_a
        or bs.fake_dequant_b
        or bs.block_size_a not in ((1, 16), (1, 32))
        or bs.block_size_b not in ((16, 1), (32, 1))
        or bs.sfa_reorder != "F8_128x4"
        or bs.sfb_reorder != "F8_128x4"
        or DTYPE_BITS.get(bs.sf_dtype_a) != 8
        or DTYPE_BITS.get(bs.sf_dtype_b) != 8
    ):
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
    if profile is None or profile.device_name != device.name or not _simple_graph(chain):
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


def candidate_configs(chain: FusionChain, baseline: TileConfig, device: DeviceProperties, profile: CostProfile) -> list[TileConfig]:
    candidates = {baseline: None}
    for swapped, view in ((False, chain), (True, swap_ab(chain))):
        mm = view.matmul
        auto = (
            select_config(mm.M, mm.N, 1, K=mm.K, block_scale=view.has_block_scale, sm_count=device.sm_count) if swapped else replace(baseline, split_k_slices=1)
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
) -> list[TileConfig]:
    if not 1 <= limit <= MAX_PLAN_CONFIGS:
        raise ValueError(f"strategy limit must be between 1 and {MAX_PLAN_CONFIGS}")
    if not _simple_graph(chain):
        return [baseline]
    device = current_device_properties() if device is None else device
    if device is None:
        return [baseline]
    profile = profile_for(chain, baseline, device)
    if profile is None:
        return [baseline]
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
    return [first, *(config for _, config in ranked if config != first and config in selected)]


def select_strategy(
    chain: FusionChain,
    baseline: TileConfig,
    *,
    probe: Callable[[FusionChain, TileConfig], None],
    device: DeviceProperties | None = None,
) -> TileConfig:
    return select_strategies(chain, baseline, probe=probe, device=device, limit=1)[0]
