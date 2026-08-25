# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tile config catalog — PURE GEOMETRY for the fused GEMM kernels.

A ``TileConfig`` describes ONLY dtype-independent tile geometry (cta tile,
MMA-inst tile, cluster shape, pipeline). It does NOT carry ``cta_group`` /
``ab_stages`` — those are execution strategy chosen by the
kernel template. K is stored in *bytes*, so one config covers every dtype.
Name: ``CONFIG_<pipeline>_<CTA_M>x<CTA_N>x<K_BYTES>_<MMA_M>x<MMA_N>x<MMA_K_BYTES>_cluster<cgrp_m>x<cgrp_n>``.
See ``kernel_registry`` for the template registry and the support funnel.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from cudnn.frost.occupancy import MAX_CLUSTER_SIZE as _FROST_MAX_CLUSTER_SIZE


@functools.lru_cache(maxsize=None)
def _sm_smem_budget_bytes_of(device: int) -> int:
    """Largest per-CTA SMEM the device gives a CTA — the oversized carveout where the
    part has one, else the opt-in limit."""
    from cudnn.frost.device import device_name, is_available, oversized_shared_memory_per_block, shared_memory_per_block_optin

    if not is_available():
        raise RuntimeError("cannot size the SMEM pipeline: no CUDA device is visible to query MaxSharedMemoryPerBlockOptin")
    optin = shared_memory_per_block_optin(device)
    if not optin:
        raise RuntimeError(f"the driver did not report MaxSharedMemoryPerBlockOptin for device {device_name(device)!r}; cannot size the SMEM pipeline")
    return max(int(optin), oversized_shared_memory_per_block(device))


def _sm_smem_budget_bytes(device=None) -> int:
    from cudnn.frost.device import resolve_device

    return _sm_smem_budget_bytes_of(resolve_device(device))


# Per-CTA SMEM held back off the top when sizing the ab pipeline: the scheduler ring,
# every smem barrier, the TMEM base address and — on the MoE templates — the per-CTA TMA
# tensormap scratch. The ab pipeline itself only counts the operand tensors, so these
# fragments are budgeted once here instead of being modelled stage by stage.
_SMEM_FIXED_RESERVE_BY_PIPELINE = {"sm100": 2048, "sm103": 2048, "sm107": 2048}
_SMEM_FIXED_RESERVE_MOE_BY_PIPELINE = {"sm100": 4096, "sm103": 4096, "sm107": 4096}


def _sm_smem_ab_budget_bytes(pipeline: str, device=None, *, moe: bool = False) -> int:
    table = _SMEM_FIXED_RESERVE_MOE_BY_PIPELINE if moe else _SMEM_FIXED_RESERVE_BY_PIPELINE
    return _sm_smem_budget_bytes(device) - _pipeline_fact(table, pipeline, "SMEM fixed reserve")


_L2_RETENTION_DIVISOR = 3


@functools.lru_cache(maxsize=None)
def _l2_swizzle_budget_bytes_of(device: int) -> int:
    from cudnn.frost.device import device_name, is_available, l2_cache_bytes

    if not is_available():
        raise RuntimeError("cannot size the L2 tile-rasterization budget: no CUDA device is visible to query L2CacheSize")
    l2 = l2_cache_bytes(device)
    if not l2:
        raise RuntimeError(f"the driver did not report L2CacheSize for device {device_name(device)!r}; cannot size the L2 tile-rasterization budget")
    return int(l2) // _L2_RETENTION_DIVISOR


def l2_swizzle_budget_bytes(device=None) -> int:
    """Operand bytes the N-super-block rasterization may assume stay resident in L2."""
    from cudnn.frost.device import resolve_device

    return _l2_swizzle_budget_bytes_of(resolve_device(device))


_CTA_TILE_M_MAX = 128
_CTA_TILE_N_MAX = 256
_MAX_CLUSTER_SIZE = _FROST_MAX_CLUSTER_SIZE
_CTA_TILE_K_BYTES_MAX_BY_PIPELINE = {"sm100": 128, "sm103": 384, "sm107": 128}

# MMA-inst K in bytes
_MMA_INST_K_BYTES = 32
_MMA_INST_K_BYTES_BY_PIPELINE = {"sm100": 32, "sm103": 48, "sm107": 64}


def _pipeline_fact(table: dict, pipeline: str, what: str):
    """A per-pipeline hardware fact, by EXPLICIT membership. Every table keyed by
    pipeline goes through here so a family added without an entry raises instead
    of silently inheriting another family's value."""
    if pipeline not in table:
        raise NotImplementedError(f"{what} not known for pipeline {pipeline!r}; known: {sorted(table)}")
    return table[pipeline]


# MMA instructions the CTA tile spans along M. N is deliberately NOT an axis:
# it measured within noise of a single instruction (0.96x vs 0.97x cuBLAS at
# 4096^3), cuBLAS does not split N either, and neither the 2-CTA nor the
# block-scale pipeline can express it (the pair already splits B's N; the SF
# words are indexed per 128-column block).
_NUM_MMA_MAX = 2

_AB_STAGES_CAP = 16  # cap even if SMEM permits more


def smem_max_ab_stages(
    cta_tile_m: int,
    cta_tile_n: int,
    cta_tile_k_bytes: int,
    *,
    cta_group: int = 1,
    extra_smem_bytes: int = 0,
    extra_per_stage_bytes: int = 0,
    pipeline: str,
    moe: bool = False,
    device=None,
) -> int:
    smem_b_n = cta_tile_n // cta_group
    per_stage = (cta_tile_m + smem_b_n) * cta_tile_k_bytes + extra_per_stage_bytes
    avail = _sm_smem_ab_budget_bytes(pipeline, device, moe=moe) - extra_smem_bytes
    if avail < per_stage:
        raise ValueError(
            f"tile ({cta_tile_m},{cta_tile_n},K={cta_tile_k_bytes}B) "
            f"cta_group={cta_group} per-stage SMEM {per_stage} bytes exceeds "
            f"the budget (extra_smem_bytes={extra_smem_bytes}) — can't fit "
            f"even 1 stage"
        )
    return min(avail // per_stage, _AB_STAGES_CAP)


@dataclass(frozen=True)
class TileConfig:
    """One pure-geometry tile config. Dtype- AND execution-independent.

    NOT here (template strategy, not geometry): cta_group,
    ab_stages — those, plus K-in-elements, the SMEM tile, and the hardware MMA
    shape, are derived per (dtype, cta_group) at render time.

    The CTA tile may span SEVERAL MMA instructions along M:
    ``num_mma_m = cta_tile_m / mma_inst_m``. The whole CTA tile is still
    TMA-loaded in one go; the MMA warp issues one instruction per M sub-block
    into its own TMEM accumulator region. N is never split
    (``mma_inst_n == cta_tile_n``).
    """

    cta_tile_m: int
    cta_tile_n: int
    cta_tile_k_bytes: int  # K in BYTES (dtype-independent)
    cgrp_size_m: int
    cgrp_size_n: int
    epi_tile_mn: tuple[int, int]  # epilogue subtile (M, 32)
    threads_per_cta: int  # block size (256 = 8-warp warp-spec)
    pipeline: str  # pipeline family (sm100 / sm103); matched against the template's pipeline
    acc_stages: int = 2  # TMEM accumulator stages (double-buffer)
    # N-direction super-block width (L2 reuse). 0 = adaptive: the kernel picks it per
    # launch from the runtime M/N/K, since the best width flips with the tile-grid
    # aspect ratio and one compiled kernel serves many shapes. >0 pins the width.
    tile_swizzle_n: int = 0
    # Per-CTA MMA-inst tile (None → CTA tile M/N + _MMA_INST_K_BYTES K, filled in
    # __post_init__). Forward-looking for MMA-tile-smaller-than-CTA-tile configs.
    mma_inst_m: int | None = None
    mma_inst_n: int | None = None
    mma_inst_k_bytes: int = _MMA_INST_K_BYTES

    def __post_init__(self) -> None:
        m, n, kb = self.cta_tile_m, self.cta_tile_n, self.cta_tile_k_bytes
        cm, cn = self.cgrp_size_m, self.cgrp_size_n

        kb_max = _pipeline_fact(_CTA_TILE_K_BYTES_MAX_BY_PIPELINE, self.pipeline, "max cta_tile_k_bytes")
        if kb <= 0 or kb > kb_max:
            raise NotImplementedError(f"TileConfig {self.name!r}: cta_tile_k_bytes={kb} — must be " f"positive, ≤ {kb_max} for pipeline {self.pipeline}")
        # sm103's K-tile is not free geometry either (K-tile = lcm(128, 48)).
        if self.pipeline == "sm103" and kb != 384:
            raise NotImplementedError(f"TileConfig {self.name!r}: sm103 fixes cta_tile_k_bytes=384 " f"(K-tile = lcm(128, 48)); got {kb}")
        # A pipeline whose MMA instruction fixes its K width owns that axis — it
        # is not free geometry (sm103 K=48B UTCOMMA, sm107 K=64B), and the K-tile
        # walks that instruction, so it is a multiple of the SAME width.
        mkb_want = _pipeline_fact(_MMA_INST_K_BYTES_BY_PIPELINE, self.pipeline, "MMA-inst K width")
        if self.mma_inst_k_bytes != mkb_want:
            raise NotImplementedError(f"TileConfig {self.name!r}: {self.pipeline} fixes " f"mma_inst_k_bytes={mkb_want}; got {self.mma_inst_k_bytes}")
        if kb % mkb_want != 0:
            raise NotImplementedError(
                f"TileConfig {self.name!r}: cta_tile_k_bytes={kb} — must be " f"a multiple of {self.pipeline}'s mma_inst_k_bytes={mkb_want}"
            )

        # CGRP size sanity. (cta_group-specific constraints — e.g. cgrp_size_m
        # even for 2-CTA MMA — live on the 2ctamma template in the registry.)
        if cm <= 0 or cn <= 0:
            raise NotImplementedError(f"TileConfig {self.name!r}: cgrp_size_mn=({cm},{cn})")
        if cm * cn > _MAX_CLUSTER_SIZE:
            raise NotImplementedError(f"TileConfig {self.name!r}: cluster {cm}x{cn} is {cm * cn} CTAs — " f"the architecture CGA limit is {_MAX_CLUSTER_SIZE}")

        # MMA-inst tile defaults to the CTA tile M/N + s128b K. Frozen → setattr.
        if self.mma_inst_m is None:
            object.__setattr__(self, "mma_inst_m", m)
        if self.mma_inst_n is None:
            object.__setattr__(self, "mma_inst_n", n)
        mm, mn, mkb = self.mma_inst_m, self.mma_inst_n, self.mma_inst_k_bytes
        if mm <= 0 or m % mm != 0:
            raise NotImplementedError(f"TileConfig {self.name!r}: mma_inst_m={mm} must be positive " f"and divide cta_tile_m={m}")
        if mn <= 0 or n % mn != 0:
            raise NotImplementedError(f"TileConfig {self.name!r}: mma_inst_n={mn} must be positive " f"and divide cta_tile_n={n}")
        if mkb <= 0 or kb % mkb != 0:
            raise NotImplementedError(f"TileConfig {self.name!r}: mma_inst_k_bytes={mkb} must be " f"positive and divide cta_tile_k_bytes={kb}")
        # The M/N hardware bounds. These have always been the MMA INSTRUCTION's;
        # they read as the CTA tile's only while one instruction covered it.
        # M = the two TMEM row layouts (128 full lanes / 64 packed half-lanes);
        # N = the idesc n_dim, encoded with its 3 LSBs dropped.
        if mm not in (64, _CTA_TILE_M_MAX):
            raise NotImplementedError(
                f"TileConfig {self.name!r}: mma_inst_m={mm} (= cta_tile_m {m} / " f"num_mma_m {self.num_mma_m}) — supported values are 64 and {_CTA_TILE_M_MAX}"
            )
        if mn < 8 or mn > _CTA_TILE_N_MAX or mn % 8 != 0:
            raise NotImplementedError(f"TileConfig {self.name!r}: mma_inst_n={mn} — supported range is " f"8 ≤ N ≤ {_CTA_TILE_N_MAX}, multiple of 8")
        # The one NEW axis: how many instructions the CTA tile spans along M. K is
        # not an instruction-count axis (the mainloop already walks
        # cta_tile_k / mma_inst_k), and N is not one at all.
        if self.num_mma_m > _NUM_MMA_MAX:
            raise NotImplementedError(
                f"TileConfig {self.name!r}: num_mma_m={self.num_mma_m} — at most " f"{_NUM_MMA_MAX} MMA instructions per CTA tile along M"
            )
        if mn != n:
            raise NotImplementedError(f"TileConfig {self.name!r}: N is not split across MMA instructions — " f"mma_inst_n must equal cta_tile_n={n}; got {mn}")
        # The epilogue drains one MMA-M block at a time, so its subtile height
        # is the instruction's M, not the CTA tile's.
        if self.epi_tile_mn[0] != mm:
            raise NotImplementedError(f"TileConfig {self.name!r}: epi_tile_mn[0]={self.epi_tile_mn[0]} " f"must equal mma_inst_m={mm}")

    @property
    def geometry_name(self) -> str:
        """Geometry token (no ``CONFIG_``/pipeline prefix) used in the kernel symbol."""
        return (
            f"{self.cta_tile_m}x{self.cta_tile_n}x{self.cta_tile_k_bytes}"
            f"_{self.mma_inst_m}x{self.mma_inst_n}x{self.mma_inst_k_bytes}"
            f"_cluster{self.cgrp_size_m}x{self.cgrp_size_n}"
        )

    @property
    def name(self) -> str:
        """Canonical identifier ``CONFIG_<pipeline>_<geometry_name>`` — pure geometry
        (cta_group/ab_stages are the template's, not in the name)."""
        return f"CONFIG_{self.pipeline}_{self.geometry_name}"

    @property
    def cta_tile_mn(self) -> tuple[int, int]:
        """Per-CTA logical output tile (M, N) in elements."""
        return (self.cta_tile_m, self.cta_tile_n)

    @property
    def num_mma_m(self) -> int:
        """MMA instructions the CTA tile spans along M."""
        return self.cta_tile_m // self.mma_inst_m

    @property
    def cgrp_size_mn(self) -> tuple[int, int]:
        """CTAs per cluster along (M, N). K is never split (cgrp_size_k == 1)."""
        return (self.cgrp_size_m, self.cgrp_size_n)

    @property
    def cgrp_size_mnk(self) -> tuple[int, int, int]:
        """cgrp_size as a cluster_shape-style triple (cgrp_size_k always 1)."""
        return (self.cgrp_size_m, self.cgrp_size_n, 1)

    @property
    def cgrp_tile_mn(self) -> tuple[int, int]:
        """Cluster aggregate output tile (M, N) in elements."""
        return (self.cta_tile_m * self.cgrp_size_m, self.cta_tile_n * self.cgrp_size_n)

    @property
    def cluster_shape(self) -> tuple[int, int, int]:
        """Alias for cgrp_size_mnk (cluster-launch terminology)."""
        return self.cgrp_size_mnk

    # -- dtype-dependent shape views (require elem_bytes) -------------------

    def cta_tile_k(self, elem_bytes: int) -> int:
        """K tile in *elements*, given the input dtype's byte width."""
        if self.cta_tile_k_bytes % elem_bytes != 0:
            raise ValueError(f"TileConfig {self.name!r}: cta_tile_k_bytes " f"({self.cta_tile_k_bytes}) is not divisible by " f"elem_bytes={elem_bytes}")
        return self.cta_tile_k_bytes // elem_bytes

    def cta_tile_mnk(self, elem_bytes: int) -> tuple[int, int, int]:
        """Per-CTA logical tile in elements (M, N, K)."""
        return (self.cta_tile_m, self.cta_tile_n, self.cta_tile_k(elem_bytes))

    def cgrp_tile_mnk(self, elem_bytes: int) -> tuple[int, int, int]:
        """Cluster aggregate tile in elements (M, N, K)."""
        m, n = self.cgrp_tile_mn
        return (m, n, self.cta_tile_k(elem_bytes))

    def cta_smem_tile_mnk(self, elem_bytes: int, cta_group: int) -> tuple[int, int, int]:
        """Per-CTA SMEM tile in elements. B's N is halved under 2-CTA MMA (needs
        the template's ``cta_group``)."""
        return (
            self.cta_tile_m,
            self.cta_tile_n // cta_group,
            self.cta_tile_k(elem_bytes),
        )

    def mma_inst_mnk(self, elem_bytes: int, cta_group: int) -> tuple[int, int, int]:
        """Hardware MMA-inst shape in elements. M spans the CTA pair
        (``mma_inst_m × cta_group``); K is ``mma_inst_k_bytes`` in elements."""
        k_inst = self.mma_inst_k_bytes // elem_bytes
        return (self.mma_inst_m * cta_group, self.mma_inst_n, k_inst)

    def max_ab_stages(
        self,
        cta_group: int,
        *,
        extra_smem_bytes: int = 0,
        extra_per_stage_bytes: int = 0,
        moe: bool = False,
    ) -> int:
        """Largest SMEM pipeline depth under ``cta_group`` (2-CTA MMA halves B's
        SMEM N, so it fits more stages)."""
        return smem_max_ab_stages(
            self.cta_tile_m,
            self.cta_tile_n,
            self.cta_tile_k_bytes,
            cta_group=cta_group,
            extra_smem_bytes=extra_smem_bytes,
            extra_per_stage_bytes=extra_per_stage_bytes,
            pipeline=self.pipeline,
            moe=moe,
        )

    # -- multicast model -----------------------------------------------------

    @property
    def multicast_a_factor(self) -> int:
        """# CTAs sharing the same M slice of A. Independent of cta_group."""
        return self.cgrp_size_n

    def multicast_b_factor(self, cta_group: int) -> int:
        """# CTAs sharing the same N slice of B. Under 2-CTA MMA the leading
        factor of 2 in cgrp_size_m is consumed by the MMA pair, so B-multicast
        only kicks in when cgrp_size_m ≥ 4."""
        return self.cgrp_size_m // cta_group

    @property
    def multicast_a(self) -> bool:
        return self.multicast_a_factor > 1

    def multicast_b(self, cta_group: int) -> bool:
        return self.multicast_b_factor(cta_group) > 1


# ---------------------------------------------------------------------------
# Per-pipeline config families. A kernel template pairs with the config class
# matching the ``sm<NNN>`` token in its filename (see kernel_registry). Each
# family may fix axes that its pipeline's MMA instruction dictates.
# ---------------------------------------------------------------------------


class ConfigSm100(TileConfig):
    """sm100 geometry — every axis is free (the original pure-geometry config).
    Type marker for template pairing; callers pass ``pipeline="sm100"``."""


class ConfigSm103(TileConfig):
    """sm103 geometry — ``cta_tile_k_bytes`` (384) and ``mma_inst_k_bytes``
    (48) are fixed by the fp4 K=48B UTCOMMA (K-tile = lcm(128, 48)); the other
    axes match :class:`ConfigSm100`. Type marker; callers pass ``pipeline="sm103"``."""


class ConfigSm107(TileConfig):
    """sm107 geometry — identical to :class:`ConfigSm100` except
    ``mma_inst_k_bytes`` is 64 (the SM 10.7 block-scale MMA reads a 64-byte K
    per instruction, twice sm100's 32). Type marker; callers pass
    ``pipeline="sm107"``."""


_CONFIG_CLASS_BY_PIPELINE: dict[str, type[TileConfig]] = {
    "sm100": ConfigSm100,
    "sm103": ConfigSm103,
    "sm107": ConfigSm107,
}


def config_class_for_pipeline(pipeline: str) -> type[TileConfig]:
    """The config family a template of pipeline-family ``pipeline`` pairs with."""
    try:
        return _CONFIG_CLASS_BY_PIPELINE[pipeline]
    except KeyError:
        raise KeyError(f"no config family for pipeline {pipeline!r}; known: " f"{sorted(_CONFIG_CLASS_BY_PIPELINE)}") from None


# ---------------------------------------------------------------------------
# Catalog — pure-geometry enumeration. cta_group / mainloop are NOT
# enumerated here; the registry expands each geometry across accepting templates.
# Axes: M ∈ _M_AXES (the UTCMMA instruction M and how many of them the CTA tile
# spans — cta_tile_m is the PRODUCT, not an axis), cta_n ∈ {8..256 step 8},
# K_bytes ∈ {128,64}, cluster ∈ _CLUSTERS. N < 8 / N % 8 rejected by
# __post_init__ (tcgen05 idesc n_dim is a multiple of 8). 2-CTA templates accept
# only cgrp_size_m % 2 == 0 (registry predicate).
# ---------------------------------------------------------------------------

# (mma_inst_m, num_mma_m) — the UTCMMA instruction M and how many of them one CTA
# tile spans; cta_tile_m is their product. num_mma_m == 1 first, so a lookup like
# `next(c for c in CATALOG if c.cta_tile_m == 128)` still lands on the unsplit tile.
_M_AXES: tuple[tuple[int, int], ...] = ((128, 1), (64, 1), (128, 2), (64, 2))

_CLUSTERS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 16),
    (2, 1),
    (2, 2),
    (2, 4),
    (2, 8),
    (4, 1),
    (4, 2),
    (4, 4),
    (8, 1),
    (8, 2),
    (16, 1),
)


def _geom_sm100(mma_m: int, num_mma_m: int, cta_n: int, k_bytes: int, cgrp_m: int, cgrp_n: int) -> ConfigSm100:
    """Build one sm100 pure-geometry config from the MMA-instruction M and the
    number of them the CTA tile spans."""
    return ConfigSm100(
        cta_tile_m=mma_m * num_mma_m,
        cta_tile_n=cta_n,
        cta_tile_k_bytes=k_bytes,
        cgrp_size_m=cgrp_m,
        cgrp_size_n=cgrp_n,
        mma_inst_m=mma_m,
        epi_tile_mn=(mma_m, 32),
        threads_per_cta=256,
        pipeline="sm100",
        acc_stages=2,
    )


def _geom_sm103(cta_m: int, cta_n: int, cgrp_m: int, cgrp_n: int) -> ConfigSm103:
    """Build one sm103 config (the fixed K axes are the family's, not ours)."""
    return ConfigSm103(
        cta_tile_m=cta_m,
        cta_tile_n=cta_n,
        cta_tile_k_bytes=384,
        cgrp_size_m=cgrp_m,
        cgrp_size_n=cgrp_n,
        epi_tile_mn=(cta_m, 32),
        threads_per_cta=256,
        pipeline="sm103",
        acc_stages=2,
        mma_inst_k_bytes=48,
    )


def _geom_sm107(num_mma_m: int, cta_n: int, cgrp_m: int, cgrp_n: int) -> ConfigSm107:
    """Build one sm107 config (the 64-byte MMA-inst K is the family's, not ours).
    mma_inst_m is pinned to 128 — the block-scale F8_128x4 SF swizzle needs
    mma_inst_m % 128 == 0, so 64 is not an axis here."""
    return ConfigSm107(
        cta_tile_m=_CTA_TILE_M_MAX * num_mma_m,
        cta_tile_n=cta_n,
        cta_tile_k_bytes=128,
        cgrp_size_m=cgrp_m,
        cgrp_size_n=cgrp_n,
        mma_inst_m=_CTA_TILE_M_MAX,
        epi_tile_mn=(_CTA_TILE_M_MAX, 32),
        threads_per_cta=256,
        pipeline="sm107",
        acc_stages=2,
        mma_inst_k_bytes=64,
    )


def _build_catalog() -> tuple[TileConfig, ...]:
    cfgs: list[TileConfig] = []
    for mma_m, num_mma_m in _M_AXES:
        for cta_n in range(256, 0, -8):
            for k_bytes in (128, 64):
                for cgrp_m, cgrp_n in _CLUSTERS:
                    cfgs.append(_geom_sm100(mma_m, num_mma_m, cta_n, k_bytes, cgrp_m, cgrp_n))
    # sm103 block-scale geometries: M pinned to 128 by the 1-CTA MMA atom;
    # clusters share the sm100 enumeration (the templates use the same generic
    # rank-decomposition multicast masks for data + SF).
    for cta_n in (256, 128):
        for cgrp_m, cgrp_n in _CLUSTERS:
            cfgs.append(_geom_sm103(128, cta_n, cgrp_m, cgrp_n))
    # sm107 block-scale geometries: the sm100 axes narrowed to what the F8_128x4
    # SF swizzle admits (M/N multiples of 128, K-tile 128 B) — the rest of the
    # sm100 enumeration would only be rejected by validate_block_scale_config.
    for num_mma_m in (1, 2):
        for cta_n in (256, 128):
            for cgrp_m, cgrp_n in _CLUSTERS:
                cfgs.append(_geom_sm107(num_mma_m, cta_n, cgrp_m, cgrp_n))
    return tuple(cfgs)


CATALOG: tuple[TileConfig, ...] = _build_catalog()


# Expose each catalog entry as a module-level variable matching its canonical
# name — preserves `from .tile_config import CONFIG_sm100_...`.
for _cfg in CATALOG:
    globals()[_cfg.name] = _cfg
del _cfg


_CATALOG_BY_NAME: dict[str, TileConfig] = {c.name: c for c in CATALOG}

_CONFIG_NAME_RE = re.compile(
    r"^CONFIG_(?P<pipeline>sm\d+)_"
    r"(?P<cta_m>\d+)x(?P<cta_n>\d+)x(?P<k_bytes>\d+)_"
    r"(?P<mma_m>\d+)x(?P<mma_n>\d+)x(?P<mma_k_bytes>\d+)_"
    r"cluster(?P<cgrp_m>\d+)x(?P<cgrp_n>\d+)$"
)


def _synthesize_config(name: str) -> TileConfig:
    """Build a TileConfig from a well-formed canonical name that isn't in the
    catalog (e.g. an MMA-inst tile smaller than the CTA tile). Geometry bounds
    are still enforced by ``TileConfig.__post_init__``."""
    m = _CONFIG_NAME_RE.match(name)
    if m is None:
        raise KeyError(f"unknown tile config {name!r} (not in the catalog and not a " f"canonical CONFIG_<pipeline>_MxNxKB_MxNxKB_clusterMxN name)")
    pipeline = m.group("pipeline")
    cls = config_class_for_pipeline(pipeline)  # KeyError for unknown pipelines
    cta_m = int(m.group("cta_m"))
    mma_m = int(m.group("mma_m"))
    cfg = cls(
        cta_tile_m=cta_m,
        cta_tile_n=int(m.group("cta_n")),
        cta_tile_k_bytes=int(m.group("k_bytes")),
        cgrp_size_m=int(m.group("cgrp_m")),
        cgrp_size_n=int(m.group("cgrp_n")),
        # The epilogue drains one MMA-M block per pass.
        epi_tile_mn=(mma_m, 32),
        threads_per_cta=256,
        pipeline=pipeline,
        acc_stages=2,
        mma_inst_m=mma_m,
        mma_inst_n=int(m.group("mma_n")),
        mma_inst_k_bytes=int(m.group("mma_k_bytes")),
    )
    if cfg.name != name:  # e.g. leading zeros; keep the canonical spelling unique
        raise KeyError(f"tile config name {name!r} is not canonical (round-trips to {cfg.name!r})")
    return cfg


def by_name(name: str) -> TileConfig:
    cfg = _CATALOG_BY_NAME.get(name)
    if cfg is not None:
        return cfg
    return _synthesize_config(name)


DEFAULT_CONFIG: TileConfig = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1")


def _floor_pow2(v: int) -> int:
    """Largest power of two <= v (v >= 1)."""
    p = 1
    while p * 2 <= v:
        p *= 2
    return p


_DEFAULT_SM_COUNT = 148


def _sm_count() -> int:
    """SM count of the active device, with a B200-shaped fallback for CPU-only use.

    Queried through frost's device layer (not torch) so it honours a
    ``build_device()`` scope — i.e. follows the handle's GPU during a build, like
    every other device-derived constant."""
    from cudnn.frost.device import is_available, multiprocessor_count, resolve_device

    try:
        if is_available():
            return multiprocessor_count(resolve_device(None))
    except Exception:
        pass
    return _DEFAULT_SM_COUNT


# Cluster shapes worth considering. 2-D shapes are included but, with the operand-reuse
# term switched off (see _cluster_score), the scorer has so far never selected one.
_CLUSTERS_1D = ((1, 1), (1, 2), (1, 4), (2, 1), (4, 1), (8, 1))
_CLUSTERS_2D = ((2, 2), (2, 4), (2, 8), (4, 2), (4, 4), (8, 2))


def _hang_prone(cta_m: int, cta_n: int, cgrp_m: int, cgrp_n: int) -> bool:
    """64x256 with both cluster dims >= 2 can deadlock; never emit it."""
    return (cta_m, cta_n) == (64, 256) and cgrp_m >= 2 and cgrp_n >= 2


def _tile_score(rep_m: int, N: int, K: int, cta_m: int, cta_n: int, sm_count: int) -> float:
    """Wave-quantisation tile score (higher is better).

    Geometry only: ceil-divisions plus one sqrt over (M, N, K, tile, SM count). No
    measured timings and no fitted coefficients. Inspired by the analytic tile scorer
    in flashinfer PR #2940, with its aspect-ratio bias removed -- that term penalised
    exactly the wide 128x256 tiles this workload wants.
    """
    small_k = 1.0 if K is None else (0.50 if K <= 1024 else 0.80 if K <= 2048 else 1.0)
    max_tile_area = 256 * 256
    n_tiles = -(-N // cta_n)
    n_eff = N / (n_tiles * cta_n)
    score_n = n_eff * ((cta_m * cta_n) / max_tile_area) ** 0.5
    if cta_m * cta_n > 128 * 128:
        score_n *= small_k
    m_tiles = -(-rep_m // cta_m)
    total_ctas = m_tiles * n_tiles
    waves = -(-total_ctas // sm_count)
    return rep_m * total_ctas * score_n / (cta_m * waves * sm_count)


def _cluster_score(M: int, N: int, cta_m: int, cta_n: int, cta_group: int, cgrp_m: int, cgrp_n: int, sm_count: int) -> float:
    """Grid quantisation x last-wave occupancy (higher is better)."""
    eff_m = cta_m * cta_group
    m_tiles = -(-M // eff_m)
    n_tiles = -(-N // cta_n)
    launched = (-(-m_tiles // cgrp_m)) * cgrp_m * (-(-n_tiles // cgrp_n)) * cgrp_n
    if launched <= 0:
        return -1.0
    quant = (m_tiles * n_tiles) / launched
    waves = -(-launched // sm_count)
    return quant * (launched / (waves * sm_count))


def select_config(
    M: int,
    N: int,
    num_gemms: int,
    *,
    K: int | None = None,
    block_scale: bool = False,
    b_n_major: bool = False,
    b_elem_bytes: int = 2,
    sm_count: int | None = None,
) -> tuple[TileConfig, int]:
    """Pick (TileConfig, cta_group) from problem geometry.

    One selection path for every graph type frost supports. The tile and the cluster
    are each chosen by an analytic score over (M, N, K, tile geometry,
    SM count) -- integer ceil-divisions plus one sqrt, no measured timings and no fitted
    coefficients. The support constraints that used to sit around the old N-bucket rule
    (multi-GEMM N-tile budget, block-scale 128-multiples, N-major swizzle groups) are
    applied to the scored result, so they hold exactly as before.

    ``K`` is optional only for callers that do not have it to hand; without it the
    small-K bias is neutral and everything else is unchanged. The pick is an sm100
    geometry; a caller building for another template family passes it through
    :func:`as_pipeline`.
    """
    sm = sm_count if sm_count is not None else _sm_count()
    x = max(1, num_gemms)
    # Multi-GEMM shares the 256-wide N budget across the parallel GEMMs.
    cta_n_max = max(32, min(256, _floor_pow2(256 // x)))

    cta_m = 128
    # 2-CTA needs a second M-tile to be worth it. Multi-GEMM is only implemented by the
    # 1ctamma template (see compiler._check_multi_gemm), so it stays at 1.
    cta_group = 1 if x > 1 else (2 if M > cta_m else 1)

    # --- tile ---------------------------------------------------------------
    rep_m = min(_floor_pow2(max(1, M)), 4096)
    choices = [c for c in (32, 64, 128, 256) if c <= cta_n_max]
    if block_scale or N >= 512:
        # Widening beat every narrower tile family across the measured range. Applied
        # only for N >= 512, the range that was characterised; narrower problems keep
        # the full tile list rather than extrapolating.
        wide = [c for c in choices if c >= 128]
        if wide:
            choices = wide
    cta_n = max(choices, key=lambda c: _tile_score(rep_m, N, K, cta_m, c, sm))

    if block_scale:
        cta_m = max(cta_m, 128)
        cta_n = max(cta_n, 128)
        if cta_n > cta_n_max:
            raise NotImplementedError(
                f"block-scaled matmul needs cta_tile_n % 128 == 0, but {x} parallel "
                f"GEMM(s) cap the N tile at {cta_n_max}; pick a geometry explicitly "
                "via jit_from_cudnn_graph"
            )

    if b_n_major:
        # N-major B is TMA-loaded one swizzle group of columns at a time, so the
        # PER-CTA N extent must be a whole number of groups (K_BYTES=128 here).
        group_elems = 128 // b_elem_bytes
        group = group_elems * cta_group
        cta_n = -(-max(cta_n, group) // group) * group
        if cta_n > cta_n_max:
            raise NotImplementedError(
                f"N-major B needs cta_tile_n % {group} == 0 (a {group_elems}-element "
                f"swizzle group per CTA under cta_group={cta_group}), but {x} parallel "
                f"GEMM(s) cap the N tile at {cta_n_max}; pick a geometry explicitly "
                "via jit_from_cudnn_graph"
            )

    # --- cluster ------------------------------------------------------------
    pool = [g for g in _CLUSTERS_1D + _CLUSTERS_2D if not _hang_prone(cta_m, cta_n, g[0], g[1])]
    if cta_group == 2:
        pool = [g for g in pool if g[0] % 2 == 0]  # 2-CTA needs cgrp_size_m % 2 == 0
    cgrp_m, cgrp_n = max(pool, key=lambda g: _cluster_score(M, N, cta_m, cta_n, cta_group, g[0], g[1], sm))

    name = f"CONFIG_sm100_{cta_m}x{cta_n}x128_{cta_m}x{cta_n}x{_MMA_INST_K_BYTES}_cluster{cgrp_m}x{cgrp_n}"
    return by_name(name), cta_group


def as_pipeline(cfg: TileConfig, pipeline: str) -> TileConfig:
    """The same geometry as a ``pipeline``-family config — only the family-fixed
    MMA-inst K width moves. A family whose K axes this geometry cannot satisfy
    (sm103 fixes a 384-byte K-tile) raises from the config's ``__post_init__``,
    so the invariant stays in one place."""
    if cfg.pipeline == pipeline:
        return cfg
    cls = config_class_for_pipeline(pipeline)
    return cls(
        cta_tile_m=cfg.cta_tile_m,
        cta_tile_n=cfg.cta_tile_n,
        cta_tile_k_bytes=cfg.cta_tile_k_bytes,
        cgrp_size_m=cfg.cgrp_size_m,
        cgrp_size_n=cfg.cgrp_size_n,
        epi_tile_mn=cfg.epi_tile_mn,
        threads_per_cta=cfg.threads_per_cta,
        pipeline=pipeline,
        acc_stages=cfg.acc_stages,
        tile_swizzle_n=cfg.tile_swizzle_n,
        mma_inst_m=cfg.mma_inst_m,
        mma_inst_n=cfg.mma_inst_n,
        mma_inst_k_bytes=_pipeline_fact(_MMA_INST_K_BYTES_BY_PIPELINE, pipeline, "MMA-inst K width"),
    )


# ---------------------------------------------------------------------------
# Block-scaled matmul config validation (geometry-only; cta_group lives on the
# template). The F8_128x4 SF swizzle + 32x128b.warpx4 utccp atom impose:
# cta_tile_m/n % 128 == 0 and cta_tile_k (elements) % (4*block_size) == 0.
# ---------------------------------------------------------------------------


def validate_block_scale_config(cfg: TileConfig, block_size: int, cta_tile_k_elems: int) -> None:
    """Raise if ``cfg``'s GEOMETRY cannot run a block-scaled matmul.
    ``cta_tile_k_elems`` is the K-tile in *elements* (FP4: 256 on sm100 / 768
    on sm103; FP8: 128)."""
    # The SF 128x4 swizzle gives one scale-factor word per 128-row / 128-column
    # block, and the rule is that EACH MMA INSTRUCTION must land on whole blocks —
    # so it applies to the instruction tile, not the CTA tile. They coincide at
    # num_mma == 1, which is why this used to read cta_tile_m/n.
    if cfg.mma_inst_m % 128 != 0:
        raise NotImplementedError(
            f"block-scaled matmul requires mma_inst_m % 128 == 0 (SF 128x4 swizzle: "
            f"each MMA instruction must cover whole SF blocks); config {cfg.name!r} "
            f"has cta_tile_m={cfg.cta_tile_m} / num_mma_m={cfg.num_mma_m} = {cfg.mma_inst_m}"
        )
    if cfg.mma_inst_n % 128 != 0:
        raise NotImplementedError(
            f"block-scaled matmul requires mma_inst_n % 128 == 0 (SF 128x4 swizzle: "
            f"each MMA instruction must cover whole SF blocks); config {cfg.name!r} "
            f"has mma_inst_n={cfg.mma_inst_n}"
        )
    # K-tile bytes: sm100 = one 128-B swizzled SMEM row; sm103 = 384 B
    kb_want = 384 if cfg.pipeline == "sm103" else 128
    if cfg.cta_tile_k_bytes != kb_want:
        raise NotImplementedError(
            f"block-scaled matmul on {cfg.pipeline} requires cta_tile_k_bytes == " f"{kb_want}; config {cfg.name!r} has {cfg.cta_tile_k_bytes}"
        )
    if cta_tile_k_elems % (4 * block_size) != 0:
        raise NotImplementedError(
            f"block-scaled matmul requires cta_tile_k (elements) % (4*block_size) " f"== 0; got cta_tile_k={cta_tile_k_elems}, block_size={block_size}"
        )
