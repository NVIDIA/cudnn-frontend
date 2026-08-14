"""Norm variant definitions, layout derivation, and sm_100 kernel config.

This mirrors ``cudnn.sdpa.fwd.config_sm100`` for the norm op family. Three layers:

1. :class:`NormVariant` — the five norm flavors, and the shape-derivation
   helpers (:func:`rowwise_spec`, :func:`batchnorm_spec`) that turn a torch
   input shape + norm arguments into a canonical reduction layout.
2. :class:`TemplateParams` — the *per-graph compile-time* knob record. Frozen
   and hashable so it doubles as the kernel-module cache key; it holds only
   things that change *traced code* (dtype, variant, centering, affine),
   never shapes (those flow through the per-shape ``compile`` cache as dynamic
   kernel args).
3. :class:`Cfg` + :func:`make_cfg` — the resolved sm_100 launch geometry
   (block threads, 128-bit vector width, cp.async smem-staging decision) for a
   given ``TemplateParams`` and reduction length ``M``.

The four *per-sample* norms (LayerNorm, RMSNorm, InstanceNorm, GroupNorm) reduce
over an inner axis; BatchNorm reduces across the batch per channel. Each flavor
has its own sm_100 kernel (distinct reduction pattern), but they share these
descriptors and the :mod:`cudnn.norm.fprop.kernels._common_sm100` building
blocks.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Sequence


class NormVariant(str, enum.Enum):
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    INSTANCE_NORM = "instance_norm"
    GROUP_NORM = "group_norm"
    BATCH_NORM = "batch_norm"


# The per-sample (inner-axis) family; each still gets its own kernel file, but
# they share the RowwiseSpec layout descriptor. BatchNorm is separate.
ROWWISE_VARIANTS = (
    NormVariant.LAYER_NORM,
    NormVariant.RMS_NORM,
    NormVariant.INSTANCE_NORM,
    NormVariant.GROUP_NORM,
)

# Centering: RMSNorm normalizes by RMS only (no mean subtraction).
HAS_MEAN = {
    NormVariant.LAYER_NORM: True,
    NormVariant.RMS_NORM: False,
    NormVariant.INSTANCE_NORM: True,
    NormVariant.GROUP_NORM: True,
    NormVariant.BATCH_NORM: True,
}


# ---------------------------------------------------------------------------
# Layout descriptors + shape derivation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowwiseSpec:
    """Layout for a per-sample norm reduced over the inner ``M`` axis.

    The contiguous input is viewed as ``[R, M]`` (R = number of normalization
    groups = CTAs, M = reduction length). The affine channel index for element
    ``(r, j)`` is ``(r % groups_per_sample) * channels_per_group + j // gamma_inner_span``.
    """

    variant: NormVariant
    R: int
    M: int
    gamma_len: int
    groups_per_sample: int
    channels_per_group: int
    gamma_inner_span: int
    has_mean: bool

    @property
    def stats_len(self) -> int:
        return self.R


@dataclass(frozen=True)
class BatchNormSpec:
    """Layout for BatchNorm: reduce over ``(N, spatial)`` per channel (one CTA/channel)."""

    N: int
    C: int
    S: int  # product of spatial dims (1 for 2-D [N, C])

    @property
    def count(self) -> int:
        return self.N * self.S


def _prod(xs: Sequence[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def rowwise_spec(
    variant: NormVariant,
    shape: Sequence[int],
    *,
    normalized_shape: Optional[Sequence[int]] = None,
    num_groups: Optional[int] = None,
) -> RowwiseSpec:
    """Derive a :class:`RowwiseSpec` for one of the per-sample norms."""
    shape = [int(s) for s in shape]

    if variant in (NormVariant.LAYER_NORM, NormVariant.RMS_NORM):
        if normalized_shape is None:
            normalized_shape = shape[-1:]
        D = _prod(normalized_shape)
        total = _prod(shape)
        if total % D != 0:
            raise ValueError(f"numel {total} not divisible by normalized size {D}")
        N = total // D
        return RowwiseSpec(
            variant=variant, R=N, M=D, gamma_len=D,
            groups_per_sample=1, channels_per_group=0, gamma_inner_span=1,
            has_mean=(variant == NormVariant.LAYER_NORM),
        )

    if variant == NormVariant.INSTANCE_NORM:
        if len(shape) < 2:
            raise ValueError("InstanceNorm expects [N, C, *spatial]")
        N, C = shape[0], shape[1]
        HW = _prod(shape[2:]) if len(shape) > 2 else 1
        return RowwiseSpec(
            variant=variant, R=N * C, M=HW, gamma_len=C,
            groups_per_sample=C, channels_per_group=1, gamma_inner_span=HW,
            has_mean=True,
        )

    if variant == NormVariant.GROUP_NORM:
        if len(shape) < 2:
            raise ValueError("GroupNorm expects [N, C, *spatial]")
        if num_groups is None:
            raise ValueError("GroupNorm requires num_groups")
        N, C = shape[0], shape[1]
        G = int(num_groups)
        if C % G != 0:
            raise ValueError(f"channels {C} not divisible by num_groups {G}")
        Cg = C // G
        HW = _prod(shape[2:]) if len(shape) > 2 else 1
        return RowwiseSpec(
            variant=variant, R=N * G, M=Cg * HW, gamma_len=C,
            groups_per_sample=G, channels_per_group=Cg, gamma_inner_span=HW,
            has_mean=True,
        )

    raise ValueError(f"{variant} is not a per-sample (row-wise) norm")


def batchnorm_spec(shape: Sequence[int]) -> BatchNormSpec:
    shape = [int(s) for s in shape]
    if len(shape) < 2:
        raise ValueError("BatchNorm expects [N, C, *spatial]")
    N, C = shape[0], shape[1]
    S = _prod(shape[2:]) if len(shape) > 2 else 1
    return BatchNormSpec(N=N, C=C, S=S)


# ---------------------------------------------------------------------------
# TemplateParams — trace-changing compile-time knobs (kernel-module cache key)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateParams:
    """Compile-time norm kernel parameters. Frozen + hashable so it doubles as
    the compiled-kernel cache key. Holds ONLY things that change traced code;
    shapes (R, M, group sizes, eps, momentum) are dynamic runtime args.
    """

    variant: NormVariant
    io_dtype: str  # 'bf16' | 'fp16' | 'fp32'
    has_beta: bool = True
    # BatchNorm-only trace switches (ignored for row-wise variants).
    training: bool = True
    update_running: bool = False

    @property
    def has_mean(self) -> bool:
        return HAS_MEAN[self.variant]


def _validate_params(p: TemplateParams) -> None:
    from .dtypes import SUPPORTED_IO_DTYPES

    if p.io_dtype not in SUPPORTED_IO_DTYPES:
        raise ValueError(f"io_dtype {p.io_dtype!r} not in {SUPPORTED_IO_DTYPES}")
    if not isinstance(p.variant, NormVariant):
        raise ValueError(f"variant must be a NormVariant, got {p.variant!r}")


# ---------------------------------------------------------------------------
# Cfg — resolved sm_100 launch geometry
# ---------------------------------------------------------------------------

# Memory-bound norm kernels favour many resident CTAs over huge blocks.
_MAX_BLOCK_THREADS = 256
# cp.async row staging needs the whole row in static smem; cap at 48 KB.
_STAGE_SMEM_CAP = 48 * 1024


# Staging modes (kept in sync with cudnn.norm._common_sm100).
STAGE_NONE = 0
STAGE_CPASYNC = 1
STAGE_BULK = 2


@dataclass(frozen=True)
class Cfg:
    """Resolved launch config for one (TemplateParams, M) pair."""

    block_threads: int
    V: int  # 128-bit vector width in elements
    stage_mode: int  # STAGE_NONE / STAGE_CPASYNC / STAGE_BULK
    vec: bool  # vectorized 128-bit load/store in the compute loops (M % V == 0)
    elem_bytes: int

    @property
    def stage_smem(self) -> bool:
        return self.stage_mode != STAGE_NONE


def vector_width(elem_bytes: int) -> int:
    """Elements per 128-bit vector (fp32->4, fp16/bf16->8)."""
    return max(1, 16 // int(elem_bytes))


def choose_block_threads(M: int) -> int:
    """CTA thread count (multiple of 32, <= 256) for reduction length ``M``."""
    if M >= _MAX_BLOCK_THREADS:
        return _MAX_BLOCK_THREADS
    if M <= 32:
        return 32
    t = 32
    while t < M and t < _MAX_BLOCK_THREADS:
        t *= 2
    return min(t, _MAX_BLOCK_THREADS)


def _stage_block_threads(nv: int) -> int:
    """Largest power-of-two thread count (<=256) dividing ``nv`` (=M//V)."""
    bt = 32
    while bt * 2 <= min(_MAX_BLOCK_THREADS, nv) and nv % (bt * 2) == 0:
        bt *= 2
    return bt


def make_cfg(params: TemplateParams, M: int, *, staged_rows: int = 1) -> Cfg:
    """Resolve the sm_100 launch geometry for a reduction of length ``M``.

    Picks a 128-bit vector width and decides whether the row(s) can be cp.async
    smem-staged (``staged_rows`` whole rows in <=48 KB smem, ``M % V == 0``,
    ``(M//V) % 32 == 0``). ``staged_rows`` is 1 for forward (stage X) and 2 for
    backward (stage X and DY). When staging is off the kernel reads from global.
    """
    _validate_params(params)
    from .dtypes import DTYPE_BYTES

    elem_bytes = DTYPE_BYTES[params.io_dtype]
    V = vector_width(elem_bytes)
    aligned = M % V == 0  # rows are 128-bit aligned -> bulk-copyable
    fits = staged_rows * M * elem_bytes <= _STAGE_SMEM_CAP
    # Prefer TMA bulk-async staging when the row is aligned and fits smem: one
    # instruction stages the whole row (no M % (bt*V) constraint like cp.async).
    stage_mode = STAGE_BULK if (aligned and fits) else STAGE_NONE
    # vec (128-bit register-fragment load/store) is OFF by default: it is correct
    # and available (STAGE_* x vec paths are all tested), but measured SLOWER than
    # scalar stores for this one-CTA-per-row pattern (the fragment round-trip via
    # smem adds bank conflicts). Scalar global stores are already coalesced.
    vec = False
    bt = choose_block_threads(M)
    return Cfg(block_threads=bt, V=V, stage_mode=stage_mode, vec=vec, elem_bytes=elem_bytes)


# Target per-thread X-register footprint (LDGS): keeping ldgs ~4 (measured sweet
# spot) avoids the register-spill cliff. For large C we widen the row to WARPS_N
# warps (THREADS_PER_ROW = WARPS_N*32, cross-warp reduce via smem) rather than
# growing ldgs — this is cuDNN's ln_fwd approach and the key to large-C bandwidth.
_WARP_TARGET_LDGS = 4


def make_warp_cfg(params: TemplateParams, C: int):
    """Launch geometry for the warp-per-row LN/RMS forward, or None if unsuitable.

    Returns ``(tpr, wn, intra, ldgs, rpc, block_threads, V)``:
    ``tpr`` threads reduce one row (``wn`` = tpr//32 warps cooperate for C large
    enough; ``intra`` = shfl group = min(tpr,32)), a CTA packs ``rpc`` rows, each
    thread caches ``ldgs*V`` elements. None when C isn't 128-bit aligned or has no
    suitable divisor (-> staged fallback).
    """
    from .dtypes import DTYPE_BYTES

    eb = DTYPE_BYTES[params.io_dtype]
    V = vector_width(eb)
    if C % V != 0:
        return None
    vec_cols = C // V

    if vec_cols < 32:  # tiny C: sub-warp, one warp packs several rows
        tpr = 32
        while vec_cols % tpr != 0 or tpr > vec_cols:
            tpr //= 2
        wn, intra, ldgs = 1, tpr, vec_cols // tpr
    else:  # widen to WARPS_N so ldgs stays ~ target
        best = None
        for wn in (1, 2, 4, 8):
            tpr = wn * 32
            if tpr > 256 or vec_cols % tpr != 0:
                continue
            ldgs = vec_cols // tpr
            if ldgs < 1:
                continue
            if best is None or abs(ldgs - _WARP_TARGET_LDGS) < abs(best[2] - _WARP_TARGET_LDGS):
                best = (wn * 32, wn, ldgs)
        if best is None:
            return None
        tpr, wn, ldgs = best
        intra = 32

    block_threads = max(tpr, (256 // tpr) * tpr)
    rpc = block_threads // tpr
    return (tpr, wn, intra, ldgs, rpc, block_threads, V)


def warp_cfg_candidates(params: TemplateParams, C: int):
    """All feasible warp wcfgs (one per valid ``wn``) for autotuning. Same tuple
    shape as :func:`make_warp_cfg`; empty if C is not warp-eligible."""
    from .dtypes import DTYPE_BYTES

    eb = DTYPE_BYTES[params.io_dtype]
    V = vector_width(eb)
    if C % V != 0:
        return []
    vec_cols = C // V
    if vec_cols < 32:
        tpr = 32
        while vec_cols % tpr != 0 or tpr > vec_cols:
            tpr //= 2
        bt = max(tpr, (256 // tpr) * tpr)
        return [(tpr, 1, tpr, vec_cols // tpr, bt // tpr, bt, V)]
    out = []
    for wn in (1, 2, 4, 8):
        tpr = wn * 32
        if tpr > 256 or vec_cols % tpr != 0:
            continue
        ldgs = vec_cols // tpr
        if ldgs < 1 or ldgs > 32:  # register-footprint cap
            continue
        bt = max(tpr, (256 // tpr) * tpr)
        out.append((tpr, wn, 32, ldgs, bt // tpr, bt, V))
    return out


__all__ = [
    "NormVariant",
    "make_warp_cfg",
    "warp_cfg_candidates",
    "ROWWISE_VARIANTS",
    "HAS_MEAN",
    "RowwiseSpec",
    "BatchNormSpec",
    "rowwise_spec",
    "batchnorm_spec",
    "TemplateParams",
    "Cfg",
    "make_cfg",
    "vector_width",
    "choose_block_threads",
]
