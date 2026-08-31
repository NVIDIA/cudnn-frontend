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
from dataclasses import dataclass, replace
from typing import ClassVar

from cudnn.frost.occupancy import MAX_CLUSTER_SIZE


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


def smem_ab_budget_bytes(smem_fixed_reserve: int, device=None) -> int:
    """SMEM left for the ab pipeline once the template's fixed reserve is off
    the top. That reserve -- the scheduler ring, every smem barrier, the TMEM
    base address and, on MoE, the per-CTA TMA tensormap scratch -- is the
    TEMPLATE's, so it is passed in rather than looked up from the geometry."""
    return _sm_smem_budget_bytes(device) - smem_fixed_reserve


@functools.lru_cache(maxsize=None)
def _l2_swizzle_budget_bytes_of(device: int) -> int:
    from cudnn.frost.device import device_name, is_available, l2_cache_bytes

    # A third of L2: the streaming operand and the C output take the rest.
    retention_divisor = 3
    if not is_available():
        raise RuntimeError("cannot size the L2 tile-rasterization budget: no CUDA device is visible to query L2CacheSize")
    l2 = l2_cache_bytes(device)
    if not l2:
        raise RuntimeError(f"the driver did not report L2CacheSize for device {device_name(device)!r}; cannot size the L2 tile-rasterization budget")
    return int(l2) // retention_divisor


def l2_swizzle_budget_bytes(device=None) -> int:
    """Operand bytes the N-super-block rasterization may assume stay resident in L2."""
    from cudnn.frost.device import resolve_device

    return _l2_swizzle_budget_bytes_of(resolve_device(device))


def _issuable_mma_tile_k_bytes(cls: "type[TileConfig]") -> tuple[int, ...]:
    """The K widths ``cls``'s MMA can issue. A hardware fact, so each family
    DECLARES it -- the base has no such attribute, and a family that forgot is
    an error rather than a silent inheritance of somebody else's width."""
    ok = getattr(cls, "MMA_TILE_K_BYTES", None)
    if not ok:
        raise NotImplementedError(f"MMA-inst K width not known for pipeline {cls.__name__} -- declare MMA_TILE_K_BYTES on its config class")
    return ok


def smem_ab_stages(per_stage_bytes: int, *, smem_fixed_reserve: int, extra_smem_bytes: int = 0, device=None) -> int:
    """How many ab-pipeline stages of ``per_stage_bytes`` fit, capped.

    The caller owns the stage LAYOUT -- a plain K-tile, a packed tile plus its
    scale factors, or one 128-byte K chunk -- so it passes the size in; this
    owns the budget and the ceiling. 0 when not even one stage fits, so each
    caller can complain in its own terms.
    """
    cap = 16  # nothing in the templates assumes it: mbar arrays, ab_iter % ab_stages and the tail drain are all parameterized
    avail = smem_ab_budget_bytes(smem_fixed_reserve, device) - extra_smem_bytes
    return min(avail // per_stage_bytes, cap) if avail >= per_stage_bytes else 0


@dataclass(frozen=True)
class TileConfig:
    """One pure-geometry tile config. Dtype- AND execution-independent.

    Four nested tiles, outermost first -- CGA (a cluster of CTAs), CTA, warp,
    MMA instruction -- plus the K-split axis. Each level must divide the one
    above it. K is stored in BYTES throughout, so one config serves every dtype.

    NOT here, because they are not choices: `ab_stages` (the device SMEM budget
    decides), the epilogue subtile N (the output dtype and drain width decide),
    the acc-stage count (the TMEM budget decides) and the L2 rasterization width
    (the kernel picks it per launch from the runtime shape). Anything the
    compiler can derive is derived at render time, not stored here.

    Family facts live on the family CLASS, not in a table keyed by pipeline
    name. ``cta_group`` is the clearest case: it does NOT exist here -- the
    families whose MMA spans a CTA pair declare it, and a family without the
    pair has no such attribute at all, so reading it is an AttributeError rather
    than a silent 1.
    """

    pipeline: str  # family; pairs with the template whose filename leads with it

    cta_tile_m: int
    cta_tile_n: int
    cta_tile_k_bytes: int

    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k_bytes: int

    mma_tile_m: int
    mma_tile_n: int
    mma_tile_k_bytes: int

    mma_size_m: int
    mma_size_n: int
    mma_size_k: int

    cga_size_m: int
    cga_size_n: int
    cga_size_k: int

    warps_per_cta: int

    split_k_slices: int

    def __post_init__(self) -> None:
        name = self.name

        for label, v in (
            ("cta_tile_m", self.cta_tile_m),
            ("cta_tile_n", self.cta_tile_n),
            ("cta_tile_k_bytes", self.cta_tile_k_bytes),
            ("warp_tile_m", self.warp_tile_m),
            ("warp_tile_n", self.warp_tile_n),
            ("warp_tile_k_bytes", self.warp_tile_k_bytes),
            ("mma_tile_m", self.mma_tile_m),
            ("mma_tile_n", self.mma_tile_n),
            ("mma_tile_k_bytes", self.mma_tile_k_bytes),
            ("cga_size_m", self.cga_size_m),
            ("cga_size_n", self.cga_size_n),
            ("cga_size_k", self.cga_size_k),
            ("warps_per_cta", self.warps_per_cta),
            ("split_k_slices", self.split_k_slices),
        ):
            if v <= 0:
                raise NotImplementedError(f"TileConfig {name!r}: {label}={v} must be positive")

        # Each tile level divides the one above it.
        for label, outer, inner in (
            ("m", self.cta_tile_m, self.warp_tile_m),
            ("n", self.cta_tile_n, self.warp_tile_n),
            ("k_bytes", self.cta_tile_k_bytes, self.warp_tile_k_bytes),
        ):
            if outer % inner:
                raise NotImplementedError(f"TileConfig {name!r}: warp_tile_{label}={inner} must divide cta_tile_{label}={outer}")
        for label, outer, inner in (
            ("m", self.warp_tile_m, self.mma_tile_m),
            ("n", self.warp_tile_n, self.mma_tile_n),
            ("k_bytes", self.warp_tile_k_bytes, self.mma_tile_k_bytes),
        ):
            if outer % inner:
                raise NotImplementedError(f"TileConfig {name!r}: mma_tile_{label}={inner} must divide warp_tile_{label}={outer}")

        for axis, warp, mma in (
            ("m", self.warp_tile_m, self.mma_tile_m),
            ("n", self.warp_tile_n, self.mma_tile_n),
            ("k", self.warp_tile_k_bytes, self.mma_tile_k_bytes),
        ):
            want = warp // mma
            got = getattr(self, f"mma_size_{axis}")
            if got != want:
                raise NotImplementedError(f"TileConfig {name!r}: mma_size_{axis}={got} but the tiles say {want}")

        # K is never split across CTAs of a cluster, and the workspace-reduced
        # split is not implemented -- both are axes reserved for later.
        if self.cga_size_k != 1:
            raise NotImplementedError(f"TileConfig {name!r}: cga_size_k={self.cga_size_k} — cross-CTA K reduction is not implemented")
        if self.split_k_slices != 1:
            raise NotImplementedError(f"TileConfig {name!r}: split_k_slices={self.split_k_slices} — split-K is not implemented")

        cga = self.cga_size_m * self.cga_size_n * self.cga_size_k
        if cga > MAX_CLUSTER_SIZE:
            raise NotImplementedError(
                f"TileConfig {name!r}: cluster {self.cga_size_m}x{self.cga_size_n} is {cga} CTAs — the architecture CGA limit is {MAX_CLUSTER_SIZE}"
            )

    # -- derived -------------------------------------------------------------

    @property
    def threads_per_cta(self) -> int:
        # Imported here, not at module scope: the whole gemm.frost package is
        # importable without the [cutedsl] extra, and this is the only thing in
        # the geometry that would have pulled cutlass in.
        from cutlass.cute.arch import WARP_SIZE

        return self.warps_per_cta * WARP_SIZE

    @property
    def ctas_per_mma(self) -> int:
        """CTAs ONE MMA instruction spans. Everything a multi-CTA MMA changes
        follows from this number, so a family without one answers 1 truthfully
        rather than being asked for an axis it does not have."""
        return 1

    @property
    def is_cta_pair_mma(self) -> bool:
        return self.ctas_per_mma == 2

    @property
    def cta_smem_tile_n(self) -> int:
        """B's SMEM N per CTA -- an MMA spanning several CTAs splits it."""
        return self.cta_tile_n // self.ctas_per_mma

    @property
    def mma_tile_m_hw(self) -> int:
        """M the MMA INSTRUCTION covers, across however many CTAs it spans."""
        return self.mma_tile_m * self.ctas_per_mma

    @property
    def epi_tile_m(self) -> int:
        """Rows the epilogue drains per pass -- one MMA-M block."""
        return self.mma_tile_m

    @property
    def fallback_cga_size_mnk(self) -> tuple[int, int, int]:
        """The cluster a mixed-CGA launch falls back to on the SMs a wider one
        cannot fill: the smallest this geometry is legal in. An MMA that spans
        several CTAs must keep them inside one cluster, so it can never fall
        back below its own span."""
        return (self.ctas_per_mma, 1, 1)

    @property
    def geometry_name(self) -> str:
        """Geometry token (no ``CONFIG_``/pipeline prefix) used in the kernel symbol."""
        return (
            f"{self.cta_tile_m}x{self.cta_tile_n}x{self.cta_tile_k_bytes}"
            f"_{self.mma_tile_m}x{self.mma_tile_n}x{self.mma_tile_k_bytes}"
            f"_cluster{self.cga_size_m}x{self.cga_size_n}"
            # Named only where it is an AXIS -- a pipeline without the CTA pair
            # declares no such field, so there is nothing to spell.
            + (f"_{self.cta_group}ctamma" if isinstance(self, CtaPairTileConfig) else "")
        )

    @property
    def name(self) -> str:
        """Canonical identifier ``CONFIG_<pipeline>_<geometry_name>`` — pure
        geometry (``ab_stages`` is derived from the device budget, not named)."""
        return f"CONFIG_{self.pipeline}_{self.geometry_name}"

    @property
    def cta_tile_mn(self) -> tuple[int, int]:
        """Per-CTA logical output tile (M, N) in elements."""
        return (self.cta_tile_m, self.cta_tile_n)

    @property
    def cga_size_mn(self) -> tuple[int, int]:
        """CTAs per cluster along (M, N)."""
        return (self.cga_size_m, self.cga_size_n)

    @property
    def cga_size_mnk(self) -> tuple[int, int, int]:
        """cga_size as a cluster_shape-style triple."""
        return (self.cga_size_m, self.cga_size_n, 1)

    @property
    def cga_tile_mn(self) -> tuple[int, int]:
        """Cluster aggregate output tile (M, N) in elements."""
        return (self.cta_tile_m * self.cga_size_m, self.cta_tile_n * self.cga_size_n)

    @property
    def cluster_shape(self) -> tuple[int, int, int]:
        """Alias for cga_size_mnk (cluster-launch terminology)."""
        return self.cga_size_mnk

    # -- dtype-dependent shape views (require elem_bytes) -------------------

    def cta_tile_k(self, elem_bytes: int) -> int:
        """K tile in *elements*, given the input dtype's byte width."""
        if self.cta_tile_k_bytes % elem_bytes != 0:
            raise ValueError(f"TileConfig {self.name!r}: cta_tile_k_bytes " f"({self.cta_tile_k_bytes}) is not divisible by " f"elem_bytes={elem_bytes}")
        return self.cta_tile_k_bytes // elem_bytes

    def cta_tile_mnk(self, elem_bytes: int) -> tuple[int, int, int]:
        """Per-CTA logical tile in elements (M, N, K)."""
        return (self.cta_tile_m, self.cta_tile_n, self.cta_tile_k(elem_bytes))

    def cga_tile_mnk(self, elem_bytes: int) -> tuple[int, int, int]:
        """Cluster aggregate tile in elements (M, N, K)."""
        m, n = self.cga_tile_mn
        return (m, n, self.cta_tile_k(elem_bytes))

    def cta_smem_tile_mnk(self, elem_bytes: int) -> tuple[int, int, int]:
        """Per-CTA SMEM tile in elements."""
        return (self.cta_tile_m, self.cta_smem_tile_n, self.cta_tile_k(elem_bytes))

    def mma_tile_mnk(self, elem_bytes: int) -> tuple[int, int, int]:
        """Hardware MMA-instruction shape in elements; K is ``mma_tile_k_bytes``."""
        return (self.mma_tile_m_hw, self.mma_tile_n, self.mma_tile_k_bytes // elem_bytes)

    def max_ab_stages(self, *, smem_fixed_reserve: int, extra_smem_bytes: int = 0, extra_per_stage_bytes: int = 0) -> int:
        """Largest ab-pipeline depth this geometry fits, once the TEMPLATE's
        fixed reserve is off the top (which is why the caller supplies it)."""
        per_stage = (self.cta_tile_m + self.cta_smem_tile_n) * self.cta_tile_k_bytes + extra_per_stage_bytes
        stages = smem_ab_stages(per_stage, smem_fixed_reserve=smem_fixed_reserve, extra_smem_bytes=extra_smem_bytes)
        if stages == 0:
            raise ValueError(
                f"TileConfig {self.name!r}: per-stage SMEM {per_stage} bytes exceeds the budget "
                f"(reserve={smem_fixed_reserve}, extra_smem_bytes={extra_smem_bytes}) — cannot fit even 1 stage"
            )
        return stages

    # -- multicast model -----------------------------------------------------

    @property
    def multicast_a_factor(self) -> int:
        """# CTAs sharing the same M slice of A. Independent of cta_group."""
        return self.cga_size_n

    @property
    def multicast_b_factor(self) -> int:
        """# CTAs sharing the same N slice of B. The CTAs an MMA spans are
        consumed by the instruction, not by B-multicast."""
        return self.cga_size_m // self.ctas_per_mma

    @property
    def multicast_a(self) -> bool:
        return self.multicast_a_factor > 1

    @property
    def multicast_b(self) -> bool:
        return self.multicast_b_factor > 1


# ---------------------------------------------------------------------------
# Per-pipeline config families. A kernel template pairs with the config class
# matching the ``sm<NNN>`` token in its filename (see kernel_registry). Each
# family may fix axes that its pipeline's MMA instruction dictates.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CtaPairTileConfig(TileConfig):
    """Base for the tcgen05 families, whose MMA may span a PAIR of CTAs.

    Their tensor core is CTA-scoped, so a warp tile IS the CTA tile -- the axis
    exists in the base for families that really have one (sm120), and here it is
    pinned rather than validated. ``cta_group`` is a real geometry axis, and
    everything the pair changes is an override so the base never needs it: the
    pair splits B's N across the two CTAs, one instruction covers twice the M,
    the leading factor of 2 in ``cga_size_m`` is consumed by the pair rather
    than by B-multicast, and the pair must stay inside one cluster."""

    # The tcgen05 instruction's own M/N bounds: M is the two TMEM row layouts
    # (128 full lanes / 64 packed half-lanes), N is the idesc n_dim encoded with
    # its 3 LSBs dropped. These were never the CTA tile's -- they read that way
    # only while one instruction covered the whole tile.
    _MMA_TILE_M_VALUES: ClassVar[tuple[int, ...]] = (64, 128)
    _MMA_TILE_N_MAX: ClassVar[int] = 256
    # MMA instructions one warp tile may span along M. K is not an axis (the
    # mainloop already walks cta_tile_k / mma_tile_k) and N is deliberately not
    # one at all: splitting it measured within noise of a single instruction,
    # cuBLAS does not do it either, and neither the CTA pair nor the block-scale
    # pipeline can express it.
    MMA_SIZE_M_MAX: ClassVar[int] = 2

    cta_group: int

    def __post_init__(self) -> None:
        # Their tensor core is CTA-scoped, so a warp tile IS the CTA tile: pin
        # it rather than ask the caller to spell an axis they do not have.
        object.__setattr__(self, "warp_tile_m", self.cta_tile_m)
        object.__setattr__(self, "warp_tile_n", self.cta_tile_n)
        object.__setattr__(self, "warp_tile_k_bytes", self.cta_tile_k_bytes)
        if self.cta_group not in (1, 2):
            raise NotImplementedError(f"TileConfig {self.name!r}: cta_group must be 1 or 2, got {self.cta_group}")
        # The M/N hardware bounds of the tcgen05 instruction: M is the two TMEM
        # row layouts (128 full lanes / 64 packed half-lanes); N is the idesc
        # n_dim, encoded with its 3 LSBs dropped. Checked BEFORE the generic
        # instruction-count rule so an illegal M is named as such, not as a count.
        if self.mma_tile_m not in self._MMA_TILE_M_VALUES:
            raise NotImplementedError(f"TileConfig {self.name!r}: mma_tile_m={self.mma_tile_m} — supported values are {list(self._MMA_TILE_M_VALUES)}")
        if self.mma_tile_n < 8 or self.mma_tile_n > self._MMA_TILE_N_MAX or self.mma_tile_n % 8:
            raise NotImplementedError(
                f"TileConfig {self.name!r}: mma_tile_n={self.mma_tile_n} — supported range is 8 ≤ N ≤ {self._MMA_TILE_N_MAX}, multiple of 8"
            )
        # One tcgen05 instruction covers the tile's whole N; only M is an
        # instruction-count axis here.
        if self.mma_tile_n != self.warp_tile_n:
            raise NotImplementedError(
                f"TileConfig {self.name!r}: N is not split across MMA instructions — mma_tile_n must equal warp_tile_n={self.warp_tile_n}; got {self.mma_tile_n}"
            )
        issuable = _issuable_mma_tile_k_bytes(type(self))
        if self.mma_tile_k_bytes not in issuable:
            raise NotImplementedError(f"TileConfig {self.name!r}: {self.pipeline} issues mma_tile_k_bytes {sorted(issuable)}; got {self.mma_tile_k_bytes}")
        if self.mma_size_m > self.MMA_SIZE_M_MAX:
            raise NotImplementedError(
                f"TileConfig {self.name!r}: {self.pipeline} spans at most {self.MMA_SIZE_M_MAX} MMA instruction(s) along M per warp tile; got mma_size_m={self.mma_size_m}"
            )
        super().__post_init__()

    @property
    def ctas_per_mma(self) -> int:
        return self.cta_group


@dataclass(frozen=True)
class ConfigSm100(CtaPairTileConfig):
    """sm100 geometry — every axis is free, including ``mma_tile_k_bytes``
    ∈ {32, 64} and the 2-CTA MMA pair. The 64-byte block-scale MMA is SM 10.7+
    SILICON, so which of the two a given GPU may issue is decided by
    :func:`validate_block_scale_config`, not by the config family."""

    MMA_TILE_K_BYTES: ClassVar[tuple[int, ...]] = (32, 64)
    # The widest SMEM row swizzle.
    CTA_TILE_K_BYTES_MAX: ClassVar[int] = 128

    def __post_init__(self) -> None:
        if self.cta_tile_k_bytes > self.CTA_TILE_K_BYTES_MAX:
            raise NotImplementedError(
                f"TileConfig {self.name!r}: cta_tile_k_bytes={self.cta_tile_k_bytes} — at most {self.CTA_TILE_K_BYTES_MAX} for pipeline {self.pipeline}"
            )
        super().__post_init__()


# 4 epilogue + MMA + TMA + CLC scheduler + a donor. The mainloop variant of the
# same geometry needs 4 more; that is the TEMPLATE's override, not a config.
_TCGEN05_WARPS_PER_CTA = 8

# The fp4 K=48B UTCOMMA fixes both: the K-tile is lcm(128, 48).
_SM103_CTA_TILE_K_BYTES = 384
_SM103_MMA_TILE_K_BYTES = 48


@dataclass(frozen=True)
class ConfigSm103(CtaPairTileConfig):
    """sm103 geometry — ``cta_tile_k_bytes`` (384) and ``mma_tile_k_bytes``
    (48) are fixed by the fp4 K=48B UTCOMMA (K-tile = lcm(128, 48)); the other
    axes match :class:`ConfigSm100`."""

    MMA_TILE_K_BYTES: ClassVar[tuple[int, ...]] = (_SM103_MMA_TILE_K_BYTES,)
    # The chunk pipeline miscomputes at a split M tile (A reads unwritten SMEM
    # in K) and its ab_stages budget under-counts; both are silent-wrong.
    MMA_SIZE_M_MAX: ClassVar[int] = 1
    CTA_TILE_K_BYTES_FIXED: ClassVar[int] = _SM103_CTA_TILE_K_BYTES

    def __post_init__(self) -> None:
        if self.cta_tile_k_bytes != self.CTA_TILE_K_BYTES_FIXED:
            raise NotImplementedError(
                f"TileConfig {self.name!r}: {self.pipeline} fixes cta_tile_k_bytes={self.CTA_TILE_K_BYTES_FIXED}; got {self.cta_tile_k_bytes}"
            )
        super().__post_init__()


# sm120's tensor core is WARP-scoped, so the warp tile is a real level: the CTA
# tile is split over a fixed grid of compute warps. Both are pinned to the one
# geometry the template is validated at; widening them is the next step.
_SM120_WARP_GRID_MN = (4, 2)
_SM120_WARPS_PER_CTA = 12  # 8 compute + TMA + CLC scheduler + 2 donors
_SM120_CTA_TILE_MNK_BYTES = (128, 128, 128)
_SM120_MMA_TILE_MNK_BYTES = (16, 16, 32)


@dataclass(frozen=True)
class ConfigSm120(TileConfig):
    """sm120 geometry — warp-scoped MMA on consumer Blackwell (CC 12.x).

    ``mma.sync`` is m16n8; the template issues two of them back to back along N
    and treats the pair as ONE 16x16 instruction, which is what ``mma_tile_n``
    counts. There is no CTA pair and no cluster (CC 12.x has no CGA), and the
    12-warp block is 8 compute + TMA + CLC scheduler + 2 donors."""

    MMA_TILE_K_BYTES: ClassVar[tuple[int, ...]] = (32,)

    def __post_init__(self) -> None:
        # PINNED, not validated: every axis below is fixed for now, so a config
        # crossing in from another family (`as_pipeline`) needs no sm120
        # knowledge -- it is rewritten to the one geometry the template is
        # validated at. Widening them is the next step.
        cta_m, cta_n, cta_kb = _SM120_CTA_TILE_MNK_BYTES
        warps_m, warps_n = _SM120_WARP_GRID_MN
        mma_m, mma_n, mma_kb = _SM120_MMA_TILE_MNK_BYTES
        for field, value in (
            ("cta_tile_m", cta_m),
            ("cta_tile_n", cta_n),
            ("cta_tile_k_bytes", cta_kb),
            ("warp_tile_m", cta_m // warps_m),
            ("warp_tile_n", cta_n // warps_n),
            ("warp_tile_k_bytes", cta_kb),
            ("mma_tile_m", mma_m),
            ("mma_tile_n", mma_n),
            ("mma_tile_k_bytes", mma_kb),
            ("mma_size_m", cta_m // warps_m // mma_m),
            ("mma_size_n", cta_n // warps_n // mma_n),
            ("mma_size_k", cta_kb // mma_kb),
            # CC 12.x has no CGA.
            ("cga_size_m", 1),
            ("cga_size_n", 1),
            ("warps_per_cta", _SM120_WARPS_PER_CTA),
        ):
            object.__setattr__(self, field, value)
        super().__post_init__()

    @property
    def epi_tile_m(self) -> int:
        """The accumulators are already in registers -- the epilogue drains the
        whole CTA tile in one pass, not an MMA-M block at a time."""
        return self.cta_tile_m


_CONFIG_CLASS_BY_PIPELINE: dict[str, type[TileConfig]] = {
    "sm100": ConfigSm100,
    "sm103": ConfigSm103,
    "sm120": ConfigSm120,
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
# only cga_size_m % 2 == 0 (registry predicate).
# ---------------------------------------------------------------------------

# (mma_tile_m, mma_size_m) — the UTCMMA instruction M and how many of them one CTA
# tile spans; cta_tile_m is their product. mma_size_m == 1 first, so a lookup like
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


def _geom_sm100(
    cta_tile_m: int,
    cta_tile_n: int,
    cta_tile_k_bytes: int,
    mma_tile_m: int,
    mma_tile_n: int,
    mma_tile_k_bytes: int,
    cga_size_m: int,
    cga_size_n: int,
    cta_group: int,
) -> ConfigSm100:
    """One sm100 geometry. The warp tile is the CTA tile (a CTA-scoped MMA), so
    the family pins it and it is not spelled here."""
    return ConfigSm100(
        pipeline="sm100",
        cta_tile_m=cta_tile_m,
        cta_tile_n=cta_tile_n,
        cta_tile_k_bytes=cta_tile_k_bytes,
        warp_tile_m=cta_tile_m,
        warp_tile_n=cta_tile_n,
        warp_tile_k_bytes=cta_tile_k_bytes,
        mma_tile_m=mma_tile_m,
        mma_tile_n=mma_tile_n,
        mma_tile_k_bytes=mma_tile_k_bytes,
        mma_size_m=cta_tile_m // mma_tile_m,
        mma_size_n=cta_tile_n // mma_tile_n,
        mma_size_k=cta_tile_k_bytes // mma_tile_k_bytes,
        cga_size_m=cga_size_m,
        cga_size_n=cga_size_n,
        cga_size_k=1,
        warps_per_cta=_TCGEN05_WARPS_PER_CTA,
        split_k_slices=1,
        cta_group=cta_group,
    )


def _geom_sm103(cta_tile_m: int, cta_tile_n: int, cga_size_m: int, cga_size_n: int, cta_group: int) -> ConfigSm103:
    """One sm103 geometry. The K axes are the FAMILY's (the fp4 K=48B UTCOMMA
    fixes both), and one instruction covers the whole tile."""
    return ConfigSm103(
        pipeline="sm103",
        cta_tile_m=cta_tile_m,
        cta_tile_n=cta_tile_n,
        cta_tile_k_bytes=_SM103_CTA_TILE_K_BYTES,
        warp_tile_m=cta_tile_m,
        warp_tile_n=cta_tile_n,
        warp_tile_k_bytes=_SM103_CTA_TILE_K_BYTES,
        mma_tile_m=cta_tile_m,
        mma_tile_n=cta_tile_n,
        mma_tile_k_bytes=_SM103_MMA_TILE_K_BYTES,
        mma_size_m=1,
        mma_size_n=1,
        mma_size_k=_SM103_CTA_TILE_K_BYTES // _SM103_MMA_TILE_K_BYTES,
        cga_size_m=cga_size_m,
        cga_size_n=cga_size_n,
        cga_size_k=1,
        warps_per_cta=_TCGEN05_WARPS_PER_CTA,
        split_k_slices=1,
        cta_group=cta_group,
    )


def _geom_sm120(
    cta_tile_m: int,
    cta_tile_n: int,
    cta_tile_k_bytes: int,
    warps_size_m: int,
    warps_size_n: int,
    cga_size_m: int,
    cga_size_n: int,
) -> ConfigSm120:
    """One sm120 geometry. The CTA tile is split over a ``warps_size_m`` x
    ``warps_size_n`` grid of compute warps; the MMA tile is the family's."""
    mma_tile_m, mma_tile_n, mma_tile_k_bytes = _SM120_MMA_TILE_MNK_BYTES
    return ConfigSm120(
        pipeline="sm120",
        cta_tile_m=cta_tile_m,
        cta_tile_n=cta_tile_n,
        cta_tile_k_bytes=cta_tile_k_bytes,
        warp_tile_m=cta_tile_m // warps_size_m,
        warp_tile_n=cta_tile_n // warps_size_n,
        warp_tile_k_bytes=cta_tile_k_bytes,
        mma_tile_m=mma_tile_m,
        mma_tile_n=mma_tile_n,
        mma_tile_k_bytes=mma_tile_k_bytes,
        mma_size_m=cta_tile_m // warps_size_m // mma_tile_m,
        mma_size_n=cta_tile_n // warps_size_n // mma_tile_n,
        mma_size_k=cta_tile_k_bytes // mma_tile_k_bytes,
        cga_size_m=cga_size_m,
        cga_size_n=cga_size_n,
        cga_size_k=1,
        warps_per_cta=_SM120_WARPS_PER_CTA,
        split_k_slices=1,
    )


def _build_catalog() -> tuple[TileConfig, ...]:
    cfgs: list[TileConfig] = []
    for mma_m, mma_size_m in _M_AXES:
        for cta_n in range(256, 0, -8):
            for k_bytes in (128, 64):
                for mma_k_bytes in (32, 64):
                    if k_bytes % mma_k_bytes:
                        continue
                    for cga_m, cga_n in _CLUSTERS:
                        for cta_group in (1, 2):
                            cfgs.append(_geom_sm100(mma_m * mma_size_m, cta_n, k_bytes, mma_m, cta_n, mma_k_bytes, cga_m, cga_n, cta_group))
    # sm103 block-scale geometries: M pinned to 128 by the 1-CTA MMA atom;
    # clusters share the sm100 enumeration (the templates use the same generic
    # rank-decomposition multicast masks for data + SF).
    for cta_n in (256, 128):
        for cga_m, cga_n in _CLUSTERS:
            for cta_group in (1, 2):
                cfgs.append(_geom_sm103(128, cta_n, cga_m, cga_n, cta_group))
    # sm120 is pinned to the one geometry its template is validated at.
    cfgs.append(_geom_sm120(*_SM120_CTA_TILE_MNK_BYTES, *_SM120_WARP_GRID_MN, 1, 1))
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
    r"cluster(?P<cga_m>\d+)x(?P<cga_n>\d+)(?:_(?P<cta_group>\d+)ctamma)?$"
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
        pipeline=pipeline,
        cta_tile_m=cta_m,
        cta_tile_n=int(m.group("cta_n")),
        cta_tile_k_bytes=int(m.group("k_bytes")),
        # The warp tile is family-derived, not named: a CTA-scoped MMA pins it
        # to the CTA tile, a warp-scoped one to its own warp grid. Passing the
        # CTA tile lets each family's __post_init__ overwrite or reject it.
        warp_tile_m=cta_m,
        warp_tile_n=int(m.group("cta_n")),
        warp_tile_k_bytes=int(m.group("k_bytes")),
        mma_tile_m=mma_m,
        mma_tile_n=int(m.group("mma_n")),
        mma_tile_k_bytes=int(m.group("mma_k_bytes")),
        mma_size_m=cta_m // mma_m,
        mma_size_n=int(m.group("cta_n")) // int(m.group("mma_n")),
        mma_size_k=int(m.group("k_bytes")) // int(m.group("mma_k_bytes")),
        cga_size_m=int(m.group("cga_m")),
        cga_size_n=int(m.group("cga_n")),
        cga_size_k=1,
        # Not spelled by the name; a family that fixes its own block pins it.
        warps_per_cta=_TCGEN05_WARPS_PER_CTA,
        split_k_slices=1,
        # Only where the family HAS the axis; a name for one that does not
        # carries no such token either.
        **({"cta_group": int(m.group("cta_group") or 1)} if "cta_group" in cls.__dataclass_fields__ else {}),
    )
    # Leading zeros etc. must still round-trip; a name that merely omitted the
    # cta_group suffix is accepted as the 1-CTA spelling.
    if cfg.name != name and cfg.name != f"{name}_{getattr(cfg, 'cta_group', 1)}ctamma":
        raise KeyError(f"tile config name {name!r} is not canonical (round-trips to {cfg.name!r})")
    return cfg


def by_name(name: str) -> TileConfig:
    cfg = _CATALOG_BY_NAME.get(name) or _CATALOG_BY_NAME.get(f"{name}_1ctamma")
    if cfg is not None:
        return cfg
    return _synthesize_config(name)


DEFAULT_CONFIG: TileConfig = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma")


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
) -> TileConfig:
    """Pick a TileConfig from problem geometry.

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

    # --- tile ---------------------------------------------------------------
    rep_m = min(_floor_pow2(max(1, M)), 4096)
    # M-starved problems (M <= 128: at most one 128-tall tile row) are occupancy-
    # bound, and the sweep shows 64-tall / narrow tiles winning them outright, so
    # the 64-tall family joins the scan and the wide-N restriction steps aside
    # there; taller problems keep the single 128-tall wide family. Restricted to
    # the single-GEMM row-major-B path the sweep characterised.
    cta_m_choices = (64, 128) if (M <= 128 and x == 1 and not block_scale and not b_n_major) else (128,)
    tiles = []
    for tm in cta_m_choices:
        choices = [c for c in (32, 64, 128, 256) if c <= cta_n_max]
        if block_scale or (N >= 512 and M > 128):
            # Widening beat every narrower tile family across the measured range —
            # where M supplies more than one 128-tall tile row. M-starved problems
            # keep the full tile list rather than extrapolating.
            wide = [c for c in choices if c >= 128]
            if wide:
                choices = wide
        tiles += [(tm, c) for c in choices]
    cta_m, cta_n = max(tiles, key=lambda t: _tile_score(rep_m, N, K, t[0], t[1], sm))

    # 2-CTA needs a second M-tile to be worth it. Multi-GEMM is only implemented by the
    # 1ctamma template (see compiler._check_multi_gemm), so it stays at 1.
    cta_group = 1 if x > 1 else (2 if M > cta_m else 1)

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
        pool = [g for g in pool if g[0] % 2 == 0]  # 2-CTA needs cga_size_m % 2 == 0
    # _cluster_score depends only on the wave count, so shapes tie in whole groups
    # and the winner used to be whichever came first in the pool. Break those ties
    # with the sweep's measured preference instead: A-multicast x4 for 1-CTA
    # non-block-scale picks, and 2x4 / 2x2 for block-scale 2-CTA picks once the
    # grid runs deep (> 4 waves); everything else keeps the old order.
    ctas = -(-M // (cta_m * cta_group)) * cta_group * -(-N // cta_n)
    if cta_group == 1 and not block_scale and x == 1:
        prefer = ((1, 4),)
    elif cta_group == 2 and block_scale and ctas > 4 * sm:
        prefer = ((2, 4), (2, 2))
    else:
        prefer = ()
    rank = {g: len(prefer) - i for i, g in enumerate(prefer)}
    cgrp_m, cgrp_n = max(pool, key=lambda g: (_cluster_score(M, N, cta_m, cta_n, cta_group, g[0], g[1], sm), rank.get(g, 0)))

    name = f"CONFIG_sm100_{cta_m}x{cta_n}x128_{cta_m}x{cta_n}x32_cluster{cgrp_m}x{cgrp_n}_{cta_group}ctamma"
    return by_name(name)


def as_mma_tile_k_bytes(k_bytes: int, pipeline: str) -> int:
    """The MMA-inst K width ``pipeline`` issues closest to ``k_bytes``: itself
    when the family can issue it, else the family's narrowest. A family that
    fixes the axis (sm103's 48-byte UTCOMMA) therefore always wins."""
    ok = _issuable_mma_tile_k_bytes(config_class_for_pipeline(pipeline))
    return k_bytes if k_bytes in ok else min(ok)


def as_mma_tile_k(cfg: TileConfig, k_bytes: int) -> TileConfig:
    """The same geometry at a different MMA-inst K width. A no-op when the width
    already matches or the config's family cannot issue the requested one, so
    callers need no arch or family knowledge."""
    ok = _issuable_mma_tile_k_bytes(type(cfg))
    if cfg.mma_tile_k_bytes == k_bytes or k_bytes not in ok or cfg.cta_tile_k_bytes % k_bytes:
        return cfg
    # mma_size_k counts instructions per warp tile, so it moves with the width.
    return replace(cfg, mma_tile_k_bytes=k_bytes, mma_size_k=cfg.warp_tile_k_bytes // k_bytes)


def as_pipeline(cfg: TileConfig, pipeline: str) -> TileConfig:
    """The same geometry as a ``pipeline``-family config — only the family-fixed
    MMA-inst K width moves (a family that fixes MORE axes pins them in its own
    ``__post_init__``, e.g. ConfigSm120's cluster and block size). A family
    whose K axes this geometry cannot satisfy (sm103 fixes a 384-byte K-tile)
    raises from the config's ``__post_init__``, so the invariant stays in one
    place."""
    if cfg.pipeline == pipeline:
        return cfg
    cls = config_class_for_pipeline(pipeline)
    return cls(
        pipeline=pipeline,
        cta_tile_m=cfg.cta_tile_m,
        cta_tile_n=cfg.cta_tile_n,
        cta_tile_k_bytes=cfg.cta_tile_k_bytes,
        warp_tile_m=cfg.cta_tile_m,
        warp_tile_n=cfg.cta_tile_n,
        warp_tile_k_bytes=cfg.cta_tile_k_bytes,
        mma_tile_m=cfg.mma_tile_m,
        mma_tile_n=cfg.mma_tile_n,
        mma_tile_k_bytes=as_mma_tile_k_bytes(cfg.mma_tile_k_bytes, pipeline),
        mma_size_m=cfg.cta_tile_m // cfg.mma_tile_m,
        mma_size_n=cfg.cta_tile_n // cfg.mma_tile_n,
        mma_size_k=cfg.cta_tile_k_bytes // as_mma_tile_k_bytes(cfg.mma_tile_k_bytes, pipeline),
        cga_size_m=cfg.cga_size_m,
        cga_size_n=cfg.cga_size_n,
        cga_size_k=cfg.cga_size_k,
        warps_per_cta=cfg.warps_per_cta,
        split_k_slices=cfg.split_k_slices,
        **({"cta_group": getattr(cfg, "cta_group", 1)} if "cta_group" in cls.__dataclass_fields__ else {}),
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
    if cfg.mma_tile_m % 128 != 0:
        raise NotImplementedError(
            f"block-scaled matmul requires mma_tile_m % 128 == 0 (SF 128x4 swizzle: "
            f"each MMA instruction must cover whole SF blocks); config {cfg.name!r} "
            f"has cta_tile_m={cfg.cta_tile_m} / mma_size_m={cfg.mma_size_m} = {cfg.mma_tile_m}"
        )
    if cfg.mma_tile_n % 128 != 0:
        raise NotImplementedError(
            f"block-scaled matmul requires mma_tile_n % 128 == 0 (SF 128x4 swizzle: "
            f"each MMA instruction must cover whole SF blocks); config {cfg.name!r} "
            f"has mma_tile_n={cfg.mma_tile_n}"
        )
    # The 64-byte MMA-inst K is SM 10.7+ SILICON, not a pipeline property, so it
    # is gated on the ACTIVE GPU here rather than by a config family. No GPU
    # visible (render-only / CI) leaves it open, like every other arch gate.
    if cfg.mma_tile_k_bytes == 64:
        from .kernel_registry import MMA_INST_K64_ARCH_RANGES
        from . import compiler as _C

        arch = _C._current_arch()
        if arch is not None and not any(lo <= arch < hi for lo, hi in MMA_INST_K64_ARCH_RANGES):
            spans = " or ".join(f"{lo} <= SM < {hi}" for lo, hi in MMA_INST_K64_ARCH_RANGES)
            raise NotImplementedError(f"block-scaled matmul at mma_tile_k_bytes=64 needs {spans}, but the " f"active GPU is sm_{arch}; config {cfg.name!r}")
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
