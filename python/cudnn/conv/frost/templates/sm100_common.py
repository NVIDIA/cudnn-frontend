# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SM100 convolution MMA, swizzle, and epilogue datapath helpers."""

from dataclasses import dataclass
from typing import Literal, Type

import cutlass
import cutlass.experimental.cuda as cuda
from cutlass import Numeric
from cutlass.experimental import primitives as prims


def _mma_kind_for_dtype(ab_dtype: Type[Numeric]) -> "prims.Tcgen05MMAKind":
    """Select the tcgen05 MMA kind from the A/B operand dtype.

    BF16 shares the ``f16`` kind (the instruction descriptor's a_dtype/b_dtype
    disambiguates it); the per-instruction K-shape is derived from the dtype
    width upstream, so only the MMA kind is chosen here. Accepted dtypes match
    check_supported_dtypes; any other dtype raises.
    """
    if ab_dtype in {cutlass.Float16, cutlass.BFloat16}:
        return prims.Tcgen05MMAKind.F16
    if ab_dtype is cutlass.Float32:
        return prims.Tcgen05MMAKind.TF32
    if ab_dtype in {cutlass.Float8E4M3FN, cutlass.Float8E5M2}:
        return prims.Tcgen05MMAKind.F8F6F4
    raise TypeError(f"Unsupported tcgen05 MMA dtype: {ab_dtype!r}")


def _c_smem_swizzle(c_dtype: Type[Numeric], subtile_n: int) -> "cutlass.Swizzle":
    """Select the epilogue C-SMEM swizzle from the output dtype and subtile width.

    The store packs ``subtile_n`` output channels per row, so the swizzled
    region spans ``subtile_n * c_dtype.width // 8`` bytes. The tcgen05 SMEM
    swizzle family is named by that byte width: 32B/64B/128B. Keying off bytes
    (not a fixed 2-byte assumption) keeps fp32 (4B) and fp8 (1B) outputs
    correct, not just fp16/bf16. The host C TMA descriptor and the device
    ``store_swizzled`` both derive from this one value, so they stay in
    lockstep (a mismatch silently transposes the store).
    """
    row_bytes = subtile_n * c_dtype.width // 8
    if row_bytes == 32:
        return cutlass.Swizzle(1, 4, 3)  # s32b
    if row_bytes == 64:
        return cutlass.Swizzle(2, 4, 3)  # s64b
    if row_bytes == 128:
        return cutlass.Swizzle(3, 4, 3)  # s128b
    raise TypeError(f"No SMEM swizzle for {row_bytes}B C row " f"(subtile_n={subtile_n}, dtype={c_dtype!r}); " "expected 32/64/128B.")


def _to_tensor_map_swizzle(swizzle: "cutlass.Swizzle") -> "cuda.TensorMapSwizzle":
    """Convert the semantic SMEM swizzle to the local tensor-map enum.

    ``cutlass.Swizzle.to`` only recognizes the enum class exported by the
    installed ``cutlass.experimental.cuda`` package.  The convolution uses the
    pasted tensor-map module as a unit, so convert explicitly instead of mixing
    enum classes from the two implementations.
    """
    mapping = {
        cutlass.Swizzle(0, 0, 0): cuda.TensorMapSwizzle.none,
        cutlass.Swizzle(1, 4, 3): cuda.TensorMapSwizzle.s32b,
        cutlass.Swizzle(2, 4, 3): cuda.TensorMapSwizzle.s64b,
        cutlass.Swizzle(3, 4, 3): cuda.TensorMapSwizzle.s128b,
    }
    try:
        return mapping[swizzle]
    except KeyError as exc:
        raise ValueError(f"No local tensor-map swizzle preset for {swizzle!r}.") from exc


@dataclass(frozen=True)
class _EpiDatapath:
    """Single source of truth for the epilogue (TMEM->RMEM->SMEM) datapath.

    One place resolves the ``(m_per_cta, use_2cta, c_dtype.width)`` config into
    the concrete LDTM shape, store kind, subtile width, and epilogue-warp split.
    The four decision sites (device SMEM sizing, device epilogue, host attribute
    setup, host C-descriptor) all read this descriptor instead of recomputing the
    same predicate chain, so a stale copy can no longer silently transpose or
    drop the C store.

    store_kind selects the SMEM store path:
      - "stmatrix"  : M==64, 1cta, width>=16  -> stmatrix (16x256b/16x128b LDTM)
      - "fp8_simt"  : M==64, 1cta, width==8   -> SIMT store_swizzled (16x32bx2)
      - "swizzled"  : everything else (cfgA M==128, cfgC M==64+2cta) -> 32x32b
    """

    store_kind: Literal["stmatrix", "fp8_simt", "swizzled"]
    subtile_n: int
    c_tile_rows: int
    ldtm_shape: str
    ldtm_repx: int
    ldtm_offset: int
    warp_m_count: int
    warp_n_count: int

    @property
    def use_stmatrix(self) -> bool:
        return self.store_kind == "stmatrix"

    @property
    def use_fp8_simt(self) -> bool:
        return self.store_kind == "fp8_simt"


def _select_epi_datapath(
    m_per_cta: int,
    use_2cta_instrs: bool,
    c_dtype: Type[Numeric],
    *,
    num_epi_warps: int = 4,
) -> _EpiDatapath:
    """Resolve the epilogue datapath descriptor from the tile/dtype config.

    Every field is a Python constant resolved here, so each caller just reads the
    field it needs. ``num_epi_warps`` is the structural epilogue-warp count (the
    kernel uses warps 0-3).
    """
    is_cfgB = m_per_cta == 64 and not use_2cta_instrs
    if is_cfgB and c_dtype.width >= 16:
        return _EpiDatapath(
            store_kind="stmatrix",
            subtile_n=(128 // c_dtype.width) * 2,
            c_tile_rows=m_per_cta,
            ldtm_shape="16x256b" if c_dtype.width == 16 else "16x128b",
            ldtm_repx=2,
            ldtm_offset=0,
            warp_m_count=num_epi_warps,
            warp_n_count=1,
        )
    if is_cfgB and c_dtype.width == 8:
        return _EpiDatapath(
            store_kind="fp8_simt",
            subtile_n=64,
            c_tile_rows=m_per_cta,
            ldtm_shape="16x32bx2",
            ldtm_repx=32,
            ldtm_offset=32,
            warp_m_count=num_epi_warps,
            warp_n_count=1,
        )
    is_cfgC = m_per_cta == 64 and use_2cta_instrs
    return _EpiDatapath(
        store_kind="swizzled",
        subtile_n=32,
        c_tile_rows=32 * num_epi_warps,
        ldtm_shape="32x32b",
        ldtm_repx=32,
        ldtm_offset=0,
        warp_m_count=2 if is_cfgC else num_epi_warps,
        warp_n_count=2 if is_cfgC else 1,
    )
