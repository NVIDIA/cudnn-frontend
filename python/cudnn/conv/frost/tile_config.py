# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-geometry tile configurations for the SM100 Frost convolution.

The convolution kernels share tcgen05 geometry with Frost GEMM, but not its
warp layout, multicast paths, split-K support, or scheduler policy. The catalog
enumerates every geometry representable by :class:`ConvTileConfig`; each
template supplies a predicate that filters this universe to configurations it
can implement. Values derived from the kernel structure (MMA shape, minimal
cluster, pipeline depth) are properties rather than additional tuning axes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from cudnn.frost.device import is_available, multiprocessor_count, resolve_device

__all__ = ["CATALOG", "DEFAULT_CONFIG", "ConvTileConfig", "by_name"]


_CONFIG_NAME_RE = re.compile(
    r"^CONFIG_sm100_"
    r"(?P<cta_m>\d+)x(?P<cta_n>\d+)x(?P<cta_k_bytes>\d+)_"
    r"(?P<mma_m>\d+)x(?P<mma_n>\d+)x32_"
    r"cluster(?P<cluster_m>\d+)x1_(?P<cta_group>[12])ctamma$"
)


@dataclass(frozen=True)
class ConvTileConfig:
    """One supported convolution output/K tile.

    ``cta_tile_k_bytes`` is dtype-independent.  The tcgen05 instruction always
    consumes 32 bytes of GEMM-K per issue; a CTA K tile therefore contains one
    to four MMA issues.  A 2-CTA instruction spans two adjacent M CTAs, so its
    minimal legal cluster is derived as ``(2, 1)``.
    """

    cta_tile_m: int
    cta_tile_n: int
    cta_tile_k_bytes: int
    cta_group: int

    def __post_init__(self) -> None:
        if self.cta_tile_m not in (64, 128):
            raise ValueError(f"cta_tile_m must be 64 or 128, got {self.cta_tile_m}")
        if self.cta_tile_n not in range(32, 257, 32):
            raise ValueError(f"cta_tile_n must be in [32, 256] with step 32, got {self.cta_tile_n}")
        if self.cta_tile_k_bytes not in (32, 64, 96, 128):
            raise ValueError(f"cta_tile_k_bytes must be 32, 64, 96, or 128, got {self.cta_tile_k_bytes}")
        if self.cta_group not in (1, 2):
            raise ValueError(f"cta_group must be 1 or 2, got {self.cta_group}")

    @property
    def use_2cta_instrs(self) -> bool:
        return self.cta_group == 2

    @property
    def cluster_shape_mn(self) -> tuple[int, int]:
        return (self.cta_group, 1)

    @property
    def mma_tiler_mn(self) -> tuple[int, int]:
        """Logical shape covered by one (possibly paired-CTA) MMA tile."""
        return (self.cta_tile_m * self.cta_group, self.cta_tile_n)

    @property
    def mma_tile_k_bytes(self) -> int:
        return 32

    @property
    def mma_inst_tile_k(self) -> int:
        return self.cta_tile_k_bytes // self.mma_tile_k_bytes

    def cta_tile_k(self, element_width_bits: int) -> int:
        bits = self.cta_tile_k_bytes * 8
        if element_width_bits <= 0 or bits % element_width_bits:
            raise ValueError(f"cta_tile_k_bytes={self.cta_tile_k_bytes} cannot be expressed in elements " f"with width={element_width_bits} bits")
        return bits // element_width_bits

    @property
    def name(self) -> str:
        cluster_m, cluster_n = self.cluster_shape_mn
        return (
            f"CONFIG_sm100_{self.cta_tile_m}x{self.cta_tile_n}x{self.cta_tile_k_bytes}_"
            f"{self.cta_tile_m}x{self.cta_tile_n}x{self.mma_tile_k_bytes}_"
            f"cluster{cluster_m}x{cluster_n}_{self.cta_group}ctamma"
        )


CATALOG: tuple[ConvTileConfig, ...] = tuple(
    ConvTileConfig(cta_m, cta_n, cta_k_bytes, cta_group)
    for cta_m in (128, 64)
    for cta_n in range(256, 31, -32)
    for cta_k_bytes in (128, 96, 64, 32)
    for cta_group in (2, 1)
)

_CATALOG_BY_NAME = {config.name: config for config in CATALOG}

# Preserve GEMM's convenient ``from tile_config import CONFIG_sm100_...`` form.
for _config in CATALOG:
    globals()[_config.name] = _config
del _config


def by_name(name: str) -> ConvTileConfig:
    """Return a catalog entry by its canonical GEMM-style name."""
    config = _CATALOG_BY_NAME.get(name)
    if config is not None:
        return config

    match = _CONFIG_NAME_RE.match(name)
    if match is None:
        raise KeyError(f"unknown convolution tile config {name!r}")

    values = {key: int(value) for key, value in match.groupdict().items()}
    candidate = ConvTileConfig(
        values["cta_m"],
        values["cta_n"],
        values["cta_k_bytes"],
        values["cta_group"],
    )
    if (
        values["mma_m"] != candidate.cta_tile_m
        or values["mma_n"] != candidate.cta_tile_n
        or values["cluster_m"] != candidate.cluster_shape_mn[0]
        or candidate.name != name
    ):
        raise KeyError(f"convolution tile config {name!r} is not canonical")

    return _CATALOG_BY_NAME[candidate.name]


DEFAULT_CONFIG = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma")

_DEFAULT_SM_COUNT = 148


def _sm_count() -> int:
    """Return the active device's SM count, with a B200-shaped fallback."""
    try:
        if is_available():
            return multiprocessor_count(resolve_device(None))
    except Exception:
        pass
    return _DEFAULT_SM_COUNT


def _floor_power_of_two(value: int) -> int:
    return 1 << (value.bit_length() - 1)


def _tile_score(M: int, N: int, K: int, cta_m: int, cta_n: int, sm_count: int) -> float:
    """Score N tiles by tail efficiency, tile area, and wave quantization."""
    representative_m = min(_floor_power_of_two(max(1, M)), 4096)
    small_k = 0.50 if K <= 1024 else 0.80 if K <= 2048 else 1.0
    n_tiles = -(-N // cta_n)
    n_efficiency = N / (n_tiles * cta_n)
    score_n = n_efficiency * ((cta_m * cta_n) / (256 * 256)) ** 0.5
    if cta_m * cta_n > 128 * 128:
        score_n *= small_k
    m_tiles = -(-representative_m // cta_m)
    total_ctas = m_tiles * n_tiles
    waves = -(-total_ctas // sm_count)
    return representative_m * total_ctas * score_n / (cta_m * waves * sm_count)


def _select_best_config(
    M: int,
    N: int,
    K: int,
    candidates: Sequence[ConvTileConfig],
    *,
    sm_count: int | None = None,
) -> ConvTileConfig:
    """Rank an already-filtered sequence of convolution tile candidates."""
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"M, N, and K must be positive, got {(M, N, K)}")
    if not candidates:
        raise ValueError(f"no compatible convolution tile configurations for M={M}, N={N}, K={K}")

    sm = _sm_count() if sm_count is None else sm_count
    if sm <= 0:
        raise ValueError(f"sm_count must be positive, got {sm}")
    return max(candidates, key=lambda config: _tile_score(M, N, K, config.cta_tile_m, config.cta_tile_n, sm))


def _get_config(
    M: int,
    N: int,
    K: int,
    *,
    predicate: Callable[[ConvTileConfig], bool],
    sm_count: int | None = None,
) -> ConvTileConfig:
    """Select a predicate-compatible tile from implicit-GEMM geometry.

    ``M`` is ``batch * output_depth * output_height * output_width``, ``N`` is
    the output-channel count, and ``K`` is ``filter_t * filter_r * filter_s *
    input_channels``. The caller-owned predicate defines every variant-specific
    constraint; this helper only filters the complete geometry catalog and
    applies the shared analytic ranking policy.
    """
    candidates = tuple(config for config in CATALOG if predicate(config))
    return _select_best_config(M, N, K, candidates, sm_count=sm_count)
