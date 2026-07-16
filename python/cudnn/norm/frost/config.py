"""Norm variant definitions and layout derivation.

The four *per-sample* norms (LayerNorm, RMSNorm, InstanceNorm, GroupNorm) all
share the exact same forward/backward math: for each normalization group they
compute ``mean``/``rstd`` over the group and apply ``y = gamma * xhat + beta``.
They differ *only* in (a) how the input is partitioned into groups and (b) how
the affine parameters ``gamma``/``beta`` are indexed. We capture that with a
single :class:`RowwiseSpec` so one kernel implements all four.

Given a contiguous input viewed as a 2-D ``[R, M]`` tensor (R = number of
groups, M = reduction length), the affine parameter index for element
``(r, j)`` is::

    channel(r, j) = (r % groups_per_sample) * channels_per_group
                    + (j // gamma_inner_span)

which specializes to each variant:

    LayerNorm  : R=N, M=D,            gps=1, cpg=0,   span=1    -> channel = j
    RMSNorm    : R=N, M=D,            gps=1, cpg=0,   span=1    -> channel = j   (has_mean=False)
    InstanceNorm: R=N*C, M=HW,        gps=C, cpg=1,   span=M    -> channel = r % C
    GroupNorm  : R=N*G, M=(C/G)*HW,   gps=G, cpg=C/G, span=HW   -> channel = g*Cg + j//HW

BatchNorm is structurally different (it reduces across the batch per channel)
and is handled by its own :class:`BatchNormSpec` and kernels.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional, Sequence


class NormVariant(str, enum.Enum):
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    INSTANCE_NORM = "instance_norm"
    GROUP_NORM = "group_norm"
    BATCH_NORM = "batch_norm"


# The per-sample (row-wise) family that shares one kernel implementation.
ROWWISE_VARIANTS = (
    NormVariant.LAYER_NORM,
    NormVariant.RMS_NORM,
    NormVariant.INSTANCE_NORM,
    NormVariant.GROUP_NORM,
)


@dataclass(frozen=True)
class RowwiseSpec:
    """Layout descriptor for a per-sample norm reduced over the inner ``M`` axis."""

    variant: NormVariant
    R: int  # number of normalization groups (CTAs)
    M: int  # reduction length per group
    gamma_len: int  # number of distinct affine channels
    groups_per_sample: int
    channels_per_group: int
    gamma_inner_span: int
    has_mean: bool  # centering (False for RMSNorm)

    @property
    def stats_len(self) -> int:
        return self.R


@dataclass(frozen=True)
class BatchNormSpec:
    """Layout descriptor for BatchNorm: reduce over (N, spatial) per channel."""

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
    """Derive a :class:`RowwiseSpec` for one of the per-sample norms.

    - LayerNorm / RMSNorm: ``normalized_shape`` gives the trailing dims that are
      normalized together (like ``torch.nn.LayerNorm``). ``D = prod(normalized_shape)``,
      ``N = numel / D``.
    - InstanceNorm: ``shape = [N, C, *spatial]``; each (n, c) is a group.
    - GroupNorm: ``shape = [N, C, *spatial]`` with ``num_groups=G``.
    """
    shape = [int(s) for s in shape]

    if variant in (NormVariant.LAYER_NORM, NormVariant.RMS_NORM):
        if normalized_shape is None:
            normalized_shape = shape[-1:]
        D = _prod(normalized_shape)
        total = _prod(shape)
        assert total % D == 0, f"numel {total} not divisible by normalized size {D}"
        N = total // D
        return RowwiseSpec(
            variant=variant,
            R=N,
            M=D,
            gamma_len=D,
            groups_per_sample=1,
            channels_per_group=0,
            gamma_inner_span=1,
            has_mean=(variant == NormVariant.LAYER_NORM),
        )

    if variant == NormVariant.INSTANCE_NORM:
        assert len(shape) >= 2, "InstanceNorm expects [N, C, *spatial]"
        N, C = shape[0], shape[1]
        HW = _prod(shape[2:]) if len(shape) > 2 else 1
        return RowwiseSpec(
            variant=variant,
            R=N * C,
            M=HW,
            gamma_len=C,
            groups_per_sample=C,
            channels_per_group=1,
            gamma_inner_span=HW,
            has_mean=True,
        )

    if variant == NormVariant.GROUP_NORM:
        assert len(shape) >= 2, "GroupNorm expects [N, C, *spatial]"
        assert num_groups is not None, "GroupNorm requires num_groups"
        N, C = shape[0], shape[1]
        G = int(num_groups)
        assert C % G == 0, f"channels {C} not divisible by num_groups {G}"
        Cg = C // G
        HW = _prod(shape[2:]) if len(shape) > 2 else 1
        return RowwiseSpec(
            variant=variant,
            R=N * G,
            M=Cg * HW,
            gamma_len=C,
            groups_per_sample=G,
            channels_per_group=Cg,
            gamma_inner_span=HW,
            has_mean=True,
        )

    raise ValueError(f"{variant} is not a per-sample (row-wise) norm")


def batchnorm_spec(shape: Sequence[int]) -> BatchNormSpec:
    shape = [int(s) for s in shape]
    assert len(shape) >= 2, "BatchNorm expects [N, C, *spatial]"
    N, C = shape[0], shape[1]
    S = _prod(shape[2:]) if len(shape) > 2 else 1
    return BatchNormSpec(N=N, C=C, S=S)


def vector_width(elem_bytes: int) -> int:
    """Elements per 128-bit vector for a given element size (fp32->4, fp16/bf16->8)."""
    return max(1, 16 // int(elem_bytes))


def choose_cpasync_config(M: int, V: int, elem_bytes: int, smem_cap: int = 48 * 1024):
    """Return (ok, block_threads) for the smem-staged cp.async path.

    The whole row (M * elem_bytes) is staged into static shared memory, so it
    must fit under ``smem_cap`` (48 KB static default on sm_80). Requires
    ``M % V == 0`` and ``(M // V) % 32 == 0``; picks the largest power-of-two
    thread count (<=1024) that divides ``M // V``.
    """
    if M % V != 0:
        return False, 0
    nv = M // V
    if nv % 32 != 0:
        return False, 0
    if M * elem_bytes > smem_cap:
        return False, 0
    bt = 32
    while bt * 2 <= min(_MAX_BLOCK_THREADS, nv) and nv % (bt * 2) == 0:
        bt *= 2
    return True, bt


# Memory-bound norm kernels favour many resident CTAs over huge blocks; 256
# threads/CTA keeps occupancy high (a 1024-thread block leaves few CTAs per SM).
_MAX_BLOCK_THREADS = 256


def choose_block_threads(M: int) -> int:
    """Pick a CTA thread count (multiple of 32, <= 256) for reduction length M."""
    if M >= _MAX_BLOCK_THREADS:
        return _MAX_BLOCK_THREADS
    if M <= 32:
        return 32
    # round up to next power-of-two multiple of 32
    t = 32
    while t < M and t < _MAX_BLOCK_THREADS:
        t *= 2
    return min(t, _MAX_BLOCK_THREADS)
