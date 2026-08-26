# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matmul correctness regression gate over (config × dtype-pair × shape).

Each case builds a frontend graph, JITs through the GEMM hook, and asserts
bit-tight equality vs torch-fp32 (small-integer inputs keep the reduction exact).
Parametrize order is (config, dtype, shape) so each (config, dtype) block reuses
one compiled kernel. CUDNN_GEMM_TEST_FULL=1 expands the config axis to the whole
CATALOG. Also runnable as a script (forwards argv to pytest).
"""

from __future__ import annotations

import os
import pathlib
import sys
import textwrap

import pytest
import torch

from gemm_test_utils import (
    requires_int8_mma,
    requires_sm100,
    Plan as _plan,
    vp as _vp,
    resolve as _resolve,
    e5m3_quant_ref as _e5m3_quant_ref,
    e5m3_to_float as _e5m3_to_float,
    requires_sm107,
)

# Module-wide GPU gate — every test here is end-to-end and needs a B200.
pytestmark = [pytest.mark.L0, requires_sm100]


import cudnn
import cudnn.gemm.frost  # noqa: F401  — installs the cudnn.pygraph recorder hook
from cudnn.gemm.frost.compiler import _current_arch, _epi_chunk_elems, _epi_vec_bytes
from cudnn.gemm.frost.tile_config import CATALOG, by_name

_TORCH_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}
_CUDNN_DTYPE = {
    "bf16": cudnn.data_type.BFLOAT16,
    "fp16": cudnn.data_type.HALF,
    "fp8_e4m3": cudnn.data_type.FP8_E4M3,
    "fp8_e5m2": cudnn.data_type.FP8_E5M2,
}
_ELEM_BYTES = {"bf16": 2, "fp16": 2, "fp8_e4m3": 1, "fp8_e5m2": 1}


# Shape menu: tile-aligned baseline + M-OOB + K-OOB + combined M+K-OOB. N stays
# aligned across the whole menu — see _compatible() for why.
_WEIRD_SHAPES: tuple[tuple[int, int, int], ...] = (
    # Tile-aligned baseline.
    (384, 768, 384),
    (640, 384, 512),
    (256, 1280, 256),
    (512, 1024, 640),  # K = 5×128
    # M-OOB (N aligned, K aligned).
    (255, 256, 256),  # one row short of a tile
    (200, 256, 256),  # deep inside a partial tile
    # K-OOB (M aligned, N aligned).
    (
        256,
        256,
        200,
    ),  # partial K-tile (valid for BF16/FP16, SKIP for FP8: 16B TMA stride)
    (256, 256, 96),  # smaller than one K_BYTES=128 BF16 tile
    # M + K OOB.
    (255, 256, 240),
)

# (input_dtype, output_dtype) pairs: same-dtype BF16/FP16, FP8 E4M3/E5M2 → FP16,
# and one mixed FP8 → BF16.
_CORE_DTYPE_PAIRS: tuple[tuple[str, str], ...] = (
    ("bf16", "bf16"),
    ("fp16", "fp16"),
    ("fp8_e4m3", "fp16"),
    ("fp8_e5m2", "fp16"),
    ("fp8_e4m3", "bf16"),
)

# Curated config subset — each entry covers a distinct template-architectural
# corner. Full CATALOG sweep is opt-in via CUDNN_GEMM_TEST_FULL=1.
_BF16 = cudnn.data_type.BFLOAT16
_QUICK_CONFIGS: tuple[str, ...] = (
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",  # baseline cta1 single-CTA
    "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma",  # large N
    "CONFIG_sm100_128x128x64_128x128x32_cluster1x1_1ctamma",  # K_BYTES=64
    "CONFIG_sm100_128x32x128_128x32x32_cluster1x1_1ctamma",  # smallest N=32
    "CONFIG_sm100_64x128x128_64x128x32_cluster2x1_1ctamma",  # cta1 + B-multicast
    "CONFIG_sm100_128x64x128_128x64x32_cluster1x2_1ctamma",  # cta1 + A-multicast
    "CONFIG_sm100_64x64x128_64x64x32_cluster2x2_1ctamma",  # cta1 both multicasts
    "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",  # cta2 baseline
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",  # cta2 large N
    "CONFIG_sm100_128x256x64_128x256x32_cluster2x1_2ctamma",  # cta2 K_BYTES=64
    "CONFIG_sm100_128x128x128_128x128x32_cluster4x2_2ctamma",  # cta2 big cluster
    "CONFIG_sm100_64x64x128_64x64x32_cluster2x4_2ctamma",  # cta2 cluster-m=128 (cta_tile_m=64)
    # K_BYTES=64 large-cluster coverage — mirrors the K_BYTES=128 entries above.
    "CONFIG_sm100_64x64x64_64x64x32_cluster2x2_1ctamma",  # cta1 both multicasts, K_BYTES=64
    "CONFIG_sm100_128x128x64_128x128x32_cluster4x2_2ctamma",  # cta2 big cluster, K_BYTES=64
    "CONFIG_sm100_64x64x64_64x64x32_cluster2x4_2ctamma",  # cta2 cluster-m=128, K_BYTES=64
    # N not a multiple of 32 (pow2 epilogue subtile spans + tile-clamped vsize).
    "CONFIG_sm100_128x8x128_128x8x32_cluster1x1_1ctamma",  # minimum N
    "CONFIG_sm100_128x40x128_128x40x32_cluster1x1_1ctamma",  # 32+8 tail span
    "CONFIG_sm100_64x24x128_64x24x32_cluster1x1_1ctamma",  # cta_m=64, 16+8 spans
    "CONFIG_sm100_128x144x128_128x144x32_cluster2x1_2ctamma",  # cta2 (N%16), 16-col tail
    # CTA tiles split across num_mma_m MMA instructions along M. Split is just
    # another geometry axis, so it rides the same dtype x layout x shape matrix;
    # the invariants it has to hold are pinned by the unit tests at the end.
    "CONFIG_sm100_256x128x128_128x128x32_cluster1x1_1ctamma",  # num_mma_m=2
    "CONFIG_sm100_256x256x128_128x256x32_cluster2x1_1ctamma",  # num_mma_m=2, acc_stages drops to 1
    "CONFIG_sm100_128x128x128_64x128x32_cluster1x1_1ctamma",  # num_mma_m=2 at mma_inst_m=64 (packed drain)
    "CONFIG_sm100_256x256x128_128x256x32_cluster2x1_2ctamma",  # on the pair (cuBLAS geometry)
    "CONFIG_sm100_256x128x128_128x128x32_cluster2x2_1ctamma",  # split + both multicasts
    "CONFIG_sm100_256x256x64_128x256x32_cluster2x1_1ctamma",  # split + K_BYTES=64
    # Split at mma_inst_m=64 on the pair drains N/2 per M block, so an N whose
    # half is not a power of two exercises the tile-clamped epilogue chunk.
    "CONFIG_sm100_128x48x128_64x48x32_cluster2x1_2ctamma",  # 2x2 DP, 24-col per-M-block drain
    "CONFIG_sm100_128x16x128_64x16x32_cluster2x1_2ctamma",  # 2x2 DP, 8-col per-M-block drain
)

_BATCHED_CONFIGS: tuple[str, ...] = (
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",  # rank-3 baseline
    "CONFIG_sm100_64x64x128_64x64x32_cluster2x2_1ctamma",  # rank-3 + TMA multicast
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",  # rank-3 + CTA_2 MMA
)

_BATCHED_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    # _WEIRD_SHAPES classes prefixed with batch. Batches in {1,2,3}; the runtime
    # binds to the graph batch, so each distinct batch needs its own compiled anchor.
    (1, 384, 768, 384),
    (2, 640, 384, 512),
    (3, 256, 1280, 256),
    (2, 512, 1024, 640),
    (3, 255, 256, 256),  # M-OOB
    (1, 200, 256, 256),  # M-OOB
    (2, 256, 256, 200),  # K-OOB: SKIP for FP8
    (3, 256, 256, 96),  # K-OOB
    (2, 255, 256, 240),  # M + K OOB
)

_BATCH_BROADCAST_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    # _WEIRD_SHAPES M/N/K classes, output batch > 1 so one input is a real broadcast.
    (2, 384, 768, 384),
    (3, 640, 384, 512),
    (2, 256, 1280, 256),
    (3, 512, 1024, 640),
    (2, 255, 256, 256),  # M-OOB
    (3, 200, 256, 256),  # M-OOB
    (2, 256, 256, 200),  # K-OOB: SKIP for FP8
    (3, 256, 256, 96),  # K-OOB
    (2, 255, 256, 240),  # M + K OOB
)

_BATCH_BROADCAST_CASES = tuple((side, shape) for side in ("A", "B") for shape in _BATCH_BROADCAST_SHAPES)

_INPUT_LAYOUTS: tuple[tuple[str, str], ...] = (
    ("k", "k"),
    ("m", "k"),
    ("k", "n"),
    ("m", "n"),
)
# ("k", "k") is exactly test_matmul's matrix — sweep only the non-canonical combos.
_NONCANONICAL_LAYOUTS: tuple[tuple[str, str], ...] = tuple(p for p in _INPUT_LAYOUTS if p != ("k", "k"))
_NONPACKED_CONFIGS: tuple[str, ...] = (
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
)
_NONPACKED_LAYOUTS: tuple[tuple[str, str], ...] = (
    ("k", "k"),
    ("m", "n"),
)
_NONPACKED_DTYPE_PAIRS: tuple[tuple[str, str], ...] = (
    ("bf16", "bf16"),
    ("fp8_e4m3", "fp16"),
)


def _sweep_config_names() -> list[str]:
    """Quick subset by default; full catalog under CUDNN_GEMM_TEST_FULL=1.

    The full sweep expands every catalog geometry (all arches) across both
    cta_groups; combos the engine rejects — sm103 configs (block-scale-only
    family), odd-cga_m 2ctamma — SKIP via the engine's own clean rejection."""
    if os.environ.get("CUDNN_GEMM_TEST_FULL", "0") == "1":
        return [f"{c.name}_{g}ctamma" for c in CATALOG for g in (1, 2)]
    return list(_QUICK_CONFIGS)


# Pretty IDs (drive the `-k` filter and the failure report line).


def _shape_id(s: tuple[int, int, int]) -> str:
    return f"{s[0]}x{s[1]}x{s[2]}"


def _batched_shape_id(s: tuple[int, int, int, int]) -> str:
    return f"B{s[0]}_{s[1]}x{s[2]}x{s[3]}"


def _batch_broadcast_id(p: tuple[str, tuple[int, int, int, int]]) -> str:
    side, s = p
    return f"broadcast{side}_B{s[0]}_{s[1]}x{s[2]}x{s[3]}"


def _dtype_id(p: tuple[str, str]) -> str:
    return f"{p[0]}->{p[1]}"


def _config_id(name: str) -> str:
    """Strip the redundant CONFIG_ prefix and _sm100 suffix from pytest IDs."""
    out = name
    if out.startswith("CONFIG_"):
        out = out[len("CONFIG_") :]
    if out.endswith("_sm100"):
        out = out[: -len("_sm100")]
    return out


def _layout_id(p: tuple[str, str]) -> str:
    return f"A{p[0]}_B{p[1]}"


# Compatibility gate.


def _compatible(
    cfg,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
    out_major: str = "n",
) -> tuple[bool, str]:
    """Reject only shapes the kernel can't service. Returns (ok, reason)."""
    in_eb = _ELEM_BYTES[in_dtype]
    out_eb = _ELEM_BYTES[out_dtype]
    if cfg.cta_tile_k_bytes % in_eb != 0:
        return False, (f"K_BYTES={cfg.cta_tile_k_bytes} not divisible by in_elem_bytes={in_eb} " f"(catalog × dtype mismatch)")
    a_contig_extent = K if a_major == "k" else M
    b_contig_extent = K if b_major == "k" else N
    if (a_contig_extent * in_eb) % 16 != 0:
        return False, (
            f"A {a_major}-major contiguous extent * in_eb="
            f"{a_contig_extent * in_eb} not 16B-aligned. "
            f"{in_dtype!r} needs that extent % {16 // in_eb} == 0."
        )
    if (b_contig_extent * in_eb) % 16 != 0:
        return False, (
            f"B {b_major}-major contiguous extent * in_eb="
            f"{b_contig_extent * in_eb} not 16B-aligned. "
            f"{in_dtype!r} needs that extent % {16 // in_eb} == 0."
        )
    cta_smem_m, cta_smem_n, _ = cfg.cta_smem_tile_mnk(in_eb, cta_group)
    mn_group_elems = cfg.cta_tile_k_bytes // in_eb
    # Each MMA instruction reads its own M sub-block of the SMEM tile, so the
    # swizzle-group rule applies per MMA (== the whole extent at num_mma_m == 1).
    mma_smem_m = cta_smem_m // cfg.num_mma_m
    mma_smem_n = cta_smem_n
    if a_major == "m" and (mma_smem_m < mn_group_elems or mma_smem_m % mn_group_elems != 0):
        return False, (f"A M-major per-MMA SMEM M={mma_smem_m} is not compatible with " f"the {mn_group_elems}-element swizzle group")
    if b_major == "n" and (mma_smem_n < mn_group_elems or mma_smem_n % mn_group_elems != 0):
        return False, (f"B N-major per-MMA SMEM N={mma_smem_n} is not compatible with " f"the {mn_group_elems}-element swizzle group")
    out_contig_name, out_contig_extent = ("N", N) if out_major == "n" else ("M", M)
    if (out_contig_extent * out_eb) % 32 != 0:
        return False, (
            f"{out_contig_name}*out_eb={out_contig_extent * out_eb} not 32B-aligned — "
            f"STG full-vec store bakes alignment=VEC_BYTES=32 at JIT. "
            f"{out_dtype!r} needs {out_contig_name} % {32 // out_eb} == 0."
        )
    return True, ""


# Graph + data + reference.


def _a_stride_batched(M: int, K: int, a_major: str) -> list[int]:
    return [M * K, K, 1] if a_major == "k" else [M * K, 1, M]


def _b_stride_batched(N: int, K: int, b_major: str) -> list[int]:
    return [N * K, 1, K] if b_major == "k" else [N * K, N, 1]


def _build_graph(
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dtype],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, a_major))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, b_major))
    C = g.matmul(A=A, B=B, name="mm")
    if out_major == "m":
        C.set_stride([M * N, 1, M])
    C.set_output(True)
    if out_dtype != in_dtype:
        C.set_data_type(_CUDNN_DTYPE[out_dtype])
    return g


def _build_block_quant_graph(
    M: int,
    N: int,
    K: int,
    block_size: int = 32,
) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    Q, QS = g.block_scale_quantize(input=C, block_size=block_size, name="q")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QS.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    return g


def _build_batched_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dtype],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=_a_stride_batched(M, K, a_major))
    B = g.tensor(name="B", dim=[batch, K, N], stride=_b_stride_batched(N, K, b_major))
    C = g.matmul(A=A, B=B, name="mm")
    if out_major == "m":
        C.set_stride([M * N, 1, M])
    C.set_output(True)
    if out_dtype != in_dtype:
        C.set_data_type(_CUDNN_DTYPE[out_dtype])
    return g


def _build_batch_broadcast_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    broadcast_side: str,
    a_major: str = "k",
    b_major: str = "k",
) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dtype],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    a_batch = 1 if broadcast_side == "A" else batch
    b_batch = 1 if broadcast_side == "B" else batch
    A = g.tensor(name="A", dim=[a_batch, M, K], stride=_a_stride_batched(M, K, a_major))
    B = g.tensor(name="B", dim=[b_batch, K, N], stride=_b_stride_batched(N, K, b_major))
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    if out_dtype != in_dtype:
        C.set_data_type(_CUDNN_DTYPE[out_dtype])
    return g


def _mkdata(
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    seed: int = 0,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
):
    """Small-integer inputs ⇒ exact FP32 reduction ⇒ kernel and reference differ
    only by the final deterministic downcast. All shapes rank-3, batch=1."""
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dtype.startswith("fp8") else (-2, 2)
    a_shape = (1, M, K) if a_major == "k" else (1, K, M)
    b_shape = (1, N, K) if b_major == "k" else (1, K, N)
    a = torch.empty(*a_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    b = torch.empty(*b_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    if a_major == "m":
        a = a.transpose(1, 2)
    if b_major == "n":
        b = b.transpose(1, 2)
    if out_major == "m":
        c = torch.empty(1, N, M, dtype=_TORCH_DTYPE[out_dtype], device="cuda").transpose(1, 2)
    else:
        c = torch.empty(1, M, N, dtype=_TORCH_DTYPE[out_dtype], device="cuda")
    return a, b, c


def _mkbatched_data(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    seed: int = 0,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
):
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dtype.startswith("fp8") else (-2, 2)
    a_shape = (batch, M, K) if a_major == "k" else (batch, K, M)
    b_shape = (batch, N, K) if b_major == "k" else (batch, K, N)
    a = torch.empty(*a_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    b = torch.empty(*b_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    if a_major == "m":
        a = a.transpose(1, 2)
    if b_major == "n":
        b = b.transpose(1, 2)
    if out_major == "m":
        c = torch.empty(batch, N, M, dtype=_TORCH_DTYPE[out_dtype], device="cuda").transpose(1, 2)
    else:
        c = torch.empty(batch, M, N, dtype=_TORCH_DTYPE[out_dtype], device="cuda")
    return a, b, c


def _mkbatch_broadcast_data(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    broadcast_side: str,
    seed: int = 0,
    a_major: str = "k",
    b_major: str = "k",
):
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dtype.startswith("fp8") else (-2, 2)
    a_batch = 1 if broadcast_side == "A" else batch
    b_batch = 1 if broadcast_side == "B" else batch
    a_shape = (a_batch, M, K) if a_major == "k" else (a_batch, K, M)
    b_shape = (b_batch, N, K) if b_major == "k" else (b_batch, K, N)
    a = torch.empty(*a_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    b = torch.empty(*b_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    if a_major == "m":
        a = a.transpose(1, 2)
    if b_major == "n":
        b = b.transpose(1, 2)
    c = torch.empty(batch, M, N, dtype=_TORCH_DTYPE[out_dtype], device="cuda")
    return a, b, c


def _mkbatched_nonpacked_data(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    seed: int = 0,
    a_major: str = "k",
    b_major: str = "k",
):
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dtype.startswith("fp8") else (-2, 2)
    pad = 16
    device = "cuda"
    if a_major == "k":
        a_storage = torch.empty(batch, M, K + pad, dtype=torch.int32, device=device).random_(*rng)
        a_storage = a_storage.to(dtype=_TORCH_DTYPE[in_dtype])
        a = a_storage[:, :, :K]
    else:
        a_storage = torch.empty(batch, K, M + pad, dtype=torch.int32, device=device).random_(*rng)
        a_storage = a_storage.to(dtype=_TORCH_DTYPE[in_dtype])
        a = a_storage[:, :, :M].transpose(1, 2)
    if b_major == "k":
        b_storage = torch.empty(batch, N, K + pad, dtype=torch.int32, device=device).random_(*rng)
        b_storage = b_storage.to(dtype=_TORCH_DTYPE[in_dtype])
        b = b_storage[:, :, :K]
    else:
        b_storage = torch.empty(batch, K, N + pad, dtype=torch.int32, device=device).random_(*rng)
        b_storage = b_storage.to(dtype=_TORCH_DTYPE[in_dtype])
        b = b_storage[:, :, :N].transpose(1, 2)
    c_storage = torch.empty(batch, M, N + pad, dtype=_TORCH_DTYPE[out_dtype], device=device)
    c = c_storage[:, :, :N]
    return a, b, c


def _mkbatched_zero_stride_input_data(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    seed: int = 0,
):
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dtype.startswith("fp8") else (-2, 2)
    device = "cuda"
    a_base = torch.empty(K, dtype=torch.int32, device=device).random_(*rng)
    b_base = torch.empty(K, dtype=torch.int32, device=device).random_(*rng)
    a_base = a_base.to(dtype=_TORCH_DTYPE[in_dtype])
    b_base = b_base.to(dtype=_TORCH_DTYPE[in_dtype])
    a = torch.as_strided(a_base, (batch, M, K), (0, 0, 1))
    b = torch.as_strided(b_base, (batch, N, K), (0, 0, 1))
    c_storage = torch.empty(batch, M, N + 16, dtype=_TORCH_DTYPE[out_dtype], device=device)
    c = c_storage[:, :, :N]
    return a, b, c


def _reference(a: torch.Tensor, b: torch.Tensor, out_dtype: str) -> torch.Tensor:
    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    return ref.to(_TORCH_DTYPE[out_dtype])


def _block_quant_reference(
    x: torch.Tensor,
    block_size: int,
    out_dtype: torch.dtype,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, M, N = x.shape
    blocks = x.view(B, M, N // block_size, block_size)
    output_max = 448.0 if out_dtype is torch.float8_e4m3fn else 57344.0
    scale_f = blocks.abs().amax(dim=-1) / output_max
    if scale_dtype is torch.float8_e8m0fnu:
        safe = torch.where(scale_f > 0, scale_f, 1.0)
        scale_f = torch.where(
            scale_f > 0,
            torch.pow(2.0, torch.ceil(torch.log2(safe))),
            0.0,
        )
    scale = scale_f.to(scale_dtype)
    inv = torch.where(scale.float() > 0, scale.float().reciprocal(), 0.0)
    q = (blocks * inv.unsqueeze(-1)).clamp(-output_max, output_max)
    return q.to(out_dtype).view(B, M, N), scale


def _batch_broadcast_reference(a: torch.Tensor, b: torch.Tensor, batch: int, out_dtype: str) -> torch.Tensor:
    a_full = a.to(torch.float32).expand(batch, -1, -1)
    b_full = b.to(torch.float32).expand(batch, -1, -1)
    ref = torch.einsum("bmk,bnk->bmn", a_full, b_full)
    return ref.to(_TORCH_DTYPE[out_dtype])


def _tolerance(in_dtype: str, out_dtype: str) -> float:
    """One out-dtype ULP: both accumulator and reference are exact FP32, so the
    only rounding is the deterministic downcast."""
    return 1.0 if in_dtype.startswith("fp8") else 0.5


# Compile cache (session-scoped) — one compile per (config, in_dt, out_dt).


@pytest.fixture(scope="session")
def _compile_cache() -> dict:
    """Maps (config_name, in_dt, out_dt) → CompiledFusedGemm | ("skip"|"fail", msg).
    Cases visit in (config, dtype, shape) order, so each (config, dtype) block
    of 9 shapes shares one compile."""
    return {}


def _cached_outcome(entry):
    """Replay a cached skip/fail; return a cached compiled kernel."""
    if isinstance(entry, tuple) and entry[0] in ("skip", "fail"):
        kind, msg = entry
        if kind == "skip":
            pytest.skip(msg)
        pytest.fail(msg, pytrace=False)
    return entry


def _plan_or_skip(cache, key, build_graph, cfg, cta_group):
    """JIT the anchor graph; the engine's clean "unsupported" rejections —
    NotImplementedError from the compiler gates, or the registry's "no kernel
    template" — SKIP, any other compile error FAILS."""
    try:
        compiled = _plan(build_graph(), config=cfg, cta_group=cta_group)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        if isinstance(e, NotImplementedError) or (isinstance(e, ValueError) and "no kernel template" in str(e)):
            msg = first[:300]
            cache[key] = ("skip", msg)
            pytest.skip(msg)
        msg = f"JIT compile failed: {type(e).__name__}: {first[:200]}"
        cache[key] = ("fail", msg)
        pytest.fail(msg, pytrace=False)
    cache[key] = compiled
    return compiled


def _pick_anchor(
    cfg,
    in_dt: str,
    out_dt: str,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
    out_major: str = "n",
) -> tuple[int, int, int] | None:
    """First menu shape compatible with (cfg, in_dt, out_dt), as the JIT anchor.
    The kernel is M/N/K-symbolic so any compatible shape works.

    Foot-gun: the C row-stride alignment is baked at JIT from the anchor's
    contiguous output dim (N, or M for M-major output), so the menu must be
    uniform in that dim's alignment class (enforced by `_compatible`) for
    runtime shapes to be drop-in.
    """
    for shape in _WEIRD_SHAPES:
        ok, _ = _compatible(cfg, *shape, in_dt, out_dt, a_major, b_major, cta_group, out_major)
        if ok:
            return shape
    return None


def _get_compiled(
    cache: dict,
    cfg,
    in_dt: str,
    out_dt: str,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
    out_major: str = "n",
):
    """Return the cached compiled kernel, building it on first miss."""
    key = (cfg.name, in_dt, out_dt, a_major, b_major, cta_group, out_major)
    if key in cache:
        return _cached_outcome(cache[key])

    anchor = _pick_anchor(cfg, in_dt, out_dt, a_major, b_major, cta_group, out_major)
    if anchor is None:
        # No compatible anchor to build against (rare; e.g. K_BYTES vs elem_bytes mismatch).
        msg = f"no menu shape is compatible with ({cfg.name}, {in_dt}->{out_dt})"
        cache[key] = ("skip", msg)
        pytest.skip(msg)

    return _plan_or_skip(
        cache,
        key,
        lambda: _build_graph(*anchor, in_dt, out_dt, a_major, b_major, out_major),
        cfg,
        cta_group,
    )


def _pick_batched_anchor(
    cfg,
    in_dt: str,
    out_dt: str,
    batch: int,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
    out_major: str = "n",
) -> tuple[int, int, int] | None:
    """Pick a compatible M/N/K anchor for a fixed batch size."""
    for b, M, N, K in _BATCHED_SHAPES:
        if b != batch:
            continue
        ok, _ = _compatible(cfg, M, N, K, in_dt, out_dt, a_major, b_major, cta_group, out_major)
        if ok:
            return M, N, K
    return None


def _get_batched_compiled(
    cache: dict,
    cfg,
    in_dt: str,
    out_dt: str,
    batch: int,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
    out_major: str = "n",
):
    """Return the cached rank-3 compiled kernel for this graph batch."""
    key = (
        "batched",
        cfg.name,
        in_dt,
        out_dt,
        batch,
        a_major,
        b_major,
        cta_group,
        out_major,
    )
    if key in cache:
        return _cached_outcome(cache[key])

    anchor = _pick_batched_anchor(cfg, in_dt, out_dt, batch, a_major, b_major, cta_group, out_major)
    if anchor is None:
        msg = f"no batched menu shape is compatible with " f"({cfg.name}, {in_dt}->{out_dt}, batch={batch})"
        cache[key] = ("skip", msg)
        pytest.skip(msg)

    return _plan_or_skip(
        cache,
        key,
        lambda: _build_batched_graph(batch, *anchor, in_dt, out_dt, a_major, b_major, out_major),
        cfg,
        cta_group,
    )


def _pick_batch_broadcast_anchor(
    cfg,
    in_dt: str,
    out_dt: str,
    batch: int,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
) -> tuple[int, int, int] | None:
    """Pick a compatible M/N/K anchor for a fixed broadcast output batch."""
    for b, M, N, K in _BATCH_BROADCAST_SHAPES:
        if b != batch:
            continue
        ok, _ = _compatible(cfg, M, N, K, in_dt, out_dt, a_major, b_major, cta_group)
        if ok:
            return M, N, K
    return None


def _get_batch_broadcast_compiled(
    cache: dict,
    cfg,
    in_dt: str,
    out_dt: str,
    batch: int,
    broadcast_side: str,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
):
    """Return the cached rank-3 compiled kernel for a batch-broadcast graph."""
    key = (
        "batch_broadcast",
        broadcast_side,
        cfg.name,
        in_dt,
        out_dt,
        batch,
        a_major,
        b_major,
        cta_group,
    )
    if key in cache:
        return _cached_outcome(cache[key])

    anchor = _pick_batch_broadcast_anchor(cfg, in_dt, out_dt, batch, a_major, b_major, cta_group)
    if anchor is None:
        msg = f"no batch-broadcast menu shape is compatible with " f"({cfg.name}, {in_dt}->{out_dt}, batch={batch})"
        cache[key] = ("skip", msg)
        pytest.skip(msg)

    return _plan_or_skip(
        cache,
        key,
        lambda: _build_batch_broadcast_graph(batch, *anchor, in_dt, out_dt, broadcast_side, a_major, b_major),
        cfg,
        cta_group,
    )


@pytest.mark.parametrize("shape", _WEIRD_SHAPES, ids=[_shape_id(s) for s in _WEIRD_SHAPES])
@pytest.mark.parametrize(
    "in_dt,out_dt",
    _CORE_DTYPE_PAIRS,
    ids=[_dtype_id(p) for p in _CORE_DTYPE_PAIRS],
)
@pytest.mark.parametrize(
    "config_name",
    _sweep_config_names(),
    ids=[_config_id(n) for n in _sweep_config_names()],
)
def test_matmul(
    _compile_cache,
    config_name: str,
    in_dt: str,
    out_dt: str,
    shape: tuple[int, int, int],
) -> None:
    """One (config, dtype-pair, shape); incompatible combos SKIP, else bit-tight."""
    cfg, cta_group = _resolve(config_name)
    ok, reason = _compatible(cfg, *shape, in_dt, out_dt, cta_group=cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_compiled(_compile_cache, cfg, in_dt, out_dt, cta_group=cta_group)

    M, N, K = shape
    a, b, c = _mkdata(M, N, K, in_dt, out_dt)
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _reference(a, b, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    tol = _tolerance(in_dt, out_dt)
    bad = int((diff > tol).sum().item())
    max_diff = float(diff.max().item())
    max_ref = float(ref.abs().max().item())

    # Rich diagnostic so a CI failure is self-explanatory without re-running.
    assert bad == 0, (
        f"\n  config:    {config_name}"
        f"\n  dtype:     {in_dt} -> {out_dt}"
        f"\n  shape:     {M}x{N}x{K}"
        f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
        f"\n  max|diff|: {max_diff:.4g}  (tol={tol})"
        f"\n  max|ref|:  {max_ref:.4g}"
        f"\n  hint:      sample c[0,0,:8]   = {c[0, 0, :8].to(torch.float32).tolist()}"
        f"\n             sample ref[0,0,:8] = {ref[0, 0, :8].to(torch.float32).tolist()}"
    )


# Every N in _WEIRD_SHAPES is a multiple of 256, so the epilogue's last chunk of
# a tile is always whole and the partial-chunk store predicate is never exercised.
# These two straddle it — the valid column count inside the final N tile is not a
# multiple of the chunk — while M keeps a tail. N still clears the row-stride rule
# (_compatible: N * out_eb % 32 == 0), so this is chunk overhang, not N-OOB.
_N_CHUNK_STRADDLE_SHAPES: tuple[tuple[int, int, int], ...] = (
    (200, 144, 256),
    (255, 208, 240),
)

_STRADDLE_CONFIGS: tuple[str, ...] = (
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
    "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma",
    # cta_group=2 with a per-CTA n of 16: the only TMA-store coverage of the
    # narrow-N pair, which no _QUICK_CONFIGS entry reaches.
    "CONFIG_sm100_128x32x128_128x32x32_cluster2x1_2ctamma",
)


def _straddle_graph(kind: str, M: int, N: int, K: int) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=_BF16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    if kind == "plain":
        Y = C
    elif kind == "relu":
        Y = g.relu(input=C, name="r")
    elif kind == "scalar_aux":
        s = g.tensor(name="s", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
        Y = g.mul(a=C, b=s, name="sc")
    elif kind == "row_aux":
        r = g.tensor(name="r", dim=[1, M, 1], stride=[M, 1, 1], data_type=cudnn.data_type.FLOAT)
        Y = g.mul(a=C, b=r, name="rw")
    elif kind == "gen_index_axis1":
        Y = g.add(a=C, b=g.gen_index(input=C, axis=1, name="gi"), name="ad")
    elif kind == "per_col_aux":
        xc = g.tensor(name="xc", dim=[1, 1, N], stride=[N, N, 1], data_type=cudnn.data_type.FLOAT)
        Y = g.mul(a=C, b=xc, name="cw")
    elif kind == "per_elem_aux":
        xe = g.tensor(name="xe", dim=[1, M, N], stride=[M * N, N, 1], data_type=cudnn.data_type.FLOAT)
        Y = g.mul(a=C, b=xe, name="ew")
    elif kind == "matmul_tap":
        C.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        Y = g.relu(input=C, name="r")
    elif kind == "two_taps":
        C.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        R = g.relu(input=C, name="r")
        R.set_output(True).set_data_type(cudnn.data_type.HALF)
        Y = g.gelu_approx_tanh(input=R, name="ge")
    else:
        raise AssertionError(kind)
    Y.set_output(True)
    return g


# _TORCH_DTYPE covers the INPUT dtypes; a tap may also be fp32 / int32.
_TAP_TORCH_DTYPE = {**_TORCH_DTYPE, "fp32": torch.float32, "int32": torch.int32, "fp8_e8m0": torch.float8_e8m0fnu}


def _straddle_outs(plan, M: int, N: int) -> list[torch.Tensor]:
    return [torch.full((1, M, N), float("nan"), dtype=_TAP_TORCH_DTYPE[o.dtype], device="cuda") for o in plan.chain.outputs]


def _straddle_aux(plan, M: int, N: int) -> list[torch.Tensor]:
    out = []
    for t in plan.chain.aux_tensors:
        if t.name == "s":
            out.append(torch.full((1, 1, 1), 2.0, device="cuda"))
        elif t.name == "r":
            out.append(torch.arange(M, dtype=torch.float32, device="cuda").reshape(1, M, 1) % 3 + 1)
        elif t.name == "xc":
            out.append(torch.arange(N, dtype=torch.float32, device="cuda").reshape(1, 1, N) % 5 + 1)
        elif t.name == "xe":
            out.append(torch.arange(M * N, dtype=torch.float32, device="cuda").reshape(1, M, N) % 7 + 1)
        else:
            raise AssertionError(t.name)
    return out


@pytest.mark.parametrize(
    "kind",
    ("plain", "relu", "scalar_aux", "row_aux", "gen_index_axis1", "per_col_aux", "per_elem_aux", "matmul_tap", "two_taps"),
)
@pytest.mark.parametrize(
    "shape",
    _N_CHUNK_STRADDLE_SHAPES,
    ids=[_shape_id(s) for s in _N_CHUNK_STRADDLE_SHAPES],
)
@pytest.mark.parametrize("config_name", _STRADDLE_CONFIGS, ids=[_config_id(n) for n in _STRADDLE_CONFIGS])
def test_partial_n_chunk_store_matches_stg(config_name: str, shape: tuple[int, int, int], kind: str) -> None:
    """The two store paths are bit-identical when the last chunk of the N tile is
    partial. Guards the gate-relaxation work: a store rule that only holds for a
    whole chunk passes every other shape in the suite."""
    cfg, cta_group = _resolve(config_name)
    M, N, K = shape
    ok, reason = _compatible(cfg, M, N, K, "bf16", "bf16", cta_group=cta_group)
    if not ok:
        pytest.skip(reason)

    tma = _plan(_straddle_graph(kind, M, N, K), config=cfg, cta_group=cta_group)
    if not tma._compiled.use_tma_store:
        pytest.skip(f"{kind} does not take the TMA-store path on {config_name}")

    chunk = _epi_chunk_elems(tma.chain, cfg, cta_group, True)
    assert N % chunk != 0, f"shape {shape} no longer straddles the {chunk}-element chunk — the test is vacuous"

    stg = _plan(_straddle_graph(kind, M, N, K), config=cfg, cta_group=cta_group, force_stg_epi=True)
    assert not stg._compiled.use_tma_store

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    outs_tma = _straddle_outs(tma, M, N)
    outs_stg = _straddle_outs(stg, M, N)
    tma(_vp(tma, a, b, outs_tma, *_straddle_aux(tma, M, N)))
    stg(_vp(stg, a, b, outs_stg, *_straddle_aux(stg, M, N)))
    torch.cuda.synchronize()

    assert len(outs_tma) == len(tma.chain.outputs)
    unwritten = sum(int(torch.isnan(o.float()).sum()) for o in outs_tma)
    assert unwritten == 0, f"TMA store left {unwritten} cells unwritten across {len(outs_tma)} slots"
    bad = sum(int((x.float() != y.float()).sum()) for x, y in zip(outs_tma, outs_stg))
    assert bad == 0, (
        f"\n  config: {config_name}"
        f"\n  shape:  {M}x{N}x{K}  (chunk={chunk}, tail={N % chunk})"
        f"\n  kind:   {kind}"
        f"\n  slots:  {[o.dtype for o in tma.chain.outputs]}"
        f"\n  TMA != STG at {bad} cells"
    )


def test_dense_block_scale_quant_epilogue() -> None:
    """Plain dense GEMM can use terminal block_scale_quantize epilogue."""
    config_name = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"
    cfg, cta_group = _resolve(config_name)
    M = N = K = 128
    block_size = 32
    g = _build_block_quant_graph(M, N, K, block_size)
    compiled = _plan(
        g,
        config=cfg,
        cta_group=cta_group,
    )
    assert not compiled.block_scale
    assert compiled.chain.quants

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    q = torch.empty(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    q_scale = torch.empty(1, M, N // block_size, dtype=torch.float8_e8m0fnu, device="cuda")
    compiled(_vp(compiled, a, b, [q, q_scale]))
    torch.cuda.synchronize()

    ref_mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    q_ref, scale_ref = _block_quant_reference(
        ref_mm,
        block_size,
        torch.float8_e4m3fn,
        torch.float8_e8m0fnu,
    )
    torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=0, rtol=0)


def _col_quant_reference(
    x: torch.Tensor,
    block_size: int,
    out_dtype: torch.dtype,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, M, N = x.shape
    blocks = x.view(B, M // block_size, block_size, N)
    output_max = 448.0 if out_dtype is torch.float8_e4m3fn else 57344.0
    scale_f = blocks.abs().amax(dim=2) / output_max
    if scale_dtype == "e5m3":
        scale = _e5m3_quant_ref(scale_f)
        back = _e5m3_to_float(scale.to(torch.int32))
        inv = torch.where(back > 0, back.reciprocal(), 0.0)
        q = (blocks * inv.unsqueeze(2)).clamp(-output_max, output_max)
        return q.to(out_dtype).view(B, M, N), scale
    if scale_dtype is torch.float8_e8m0fnu:
        safe = torch.where(scale_f > 0, scale_f, 1.0)
        scale_f = torch.where(
            scale_f > 0,
            torch.pow(2.0, torch.ceil(torch.log2(safe))),
            0.0,
        )
    scale = scale_f.to(scale_dtype)
    inv = torch.where(scale.float() > 0, scale.float().reciprocal(), 0.0)
    q = (blocks * inv.unsqueeze(2)).clamp(-output_max, output_max)
    return q.to(out_dtype).view(B, M, N), scale


def _f8_col_scale_addr(M: int, N: int, block_size: int) -> torch.Tensor:
    """(N, M//bs) element offsets into the transposed F8_128x4 col-scale buffer."""
    mb_cnt = M // block_size
    mcb = (mb_cnt + 3) // 4
    n = torch.arange(N, device="cuda").unsqueeze(1)
    mb = torch.arange(mb_cnt, device="cuda").unsqueeze(0)
    return ((n // 128) * mcb + (mb // 4)) * 512 + (n % 32) * 16 + ((n % 128) // 32) * 4 + (mb % 4)


def _f8_row_scale_addr(M: int, N: int, block_size: int) -> torch.Tensor:
    """(M, N//bs) element offsets into the F8_128x4 row-scale buffer."""
    nb_cnt = N // block_size
    ncb = (nb_cnt + 3) // 4
    m = torch.arange(M, device="cuda").unsqueeze(1)
    nb = torch.arange(nb_cnt, device="cuda").unsqueeze(0)
    return ((m // 128) * ncb + (nb // 4)) * 512 + (m % 32) * 16 + ((m % 128) // 32) * 4 + (nb % 4)


@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",
    ],
    ids=["1ctamma", "2ctamma"],
)
def test_dense_col_block_scale_quant(config_name) -> None:
    """M-axis (col) quant: warp-redux amax, compact (B, M/32, N) scale."""
    cfg, cta_group = _resolve(config_name)
    M, N, K = 256, 128, 128
    block_size = 32
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    S = g.swish(input=C, name="sw")
    Q, QS = g.block_scale_quantize(input=S, block_size=block_size, axis=1, name="q")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QS.set_dim([1, M // block_size, N]).set_stride([M // block_size * N, N, 1])
    QS.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    assert compiled.chain.quants[0].axis == 1

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    q = torch.empty(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    q_scale = torch.empty(1, M // block_size, N, dtype=torch.float8_e8m0fnu, device="cuda")
    compiled(_vp(compiled, a, b, [q, q_scale]))
    torch.cuda.synchronize()

    ref_mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref_sw = ref_mm * torch.sigmoid(ref_mm)
    q_ref, scale_ref = _col_quant_reference(ref_sw, block_size, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=0, rtol=0)


@requires_sm107
@pytest.mark.parametrize("block_size", [16, 32])
def test_dense_col_quant_e5m3_scale(block_size) -> None:
    """COL block quantize with an E5M3 scale. This path keeps the scale in a
    per-lane register (`_scale_mine_*`) before storing, so it exercises the
    zero-init + assignment + store of the byte carrier that row quant does not."""
    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    M, N, K = 256, 128, 128
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    S = g.swish(input=C, name="sw")
    Q, QS = g.block_scale_quantize(input=S, block_size=block_size, axis=1, name="q")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QS.set_dim([1, M // block_size, N]).set_stride([M // block_size * N, N, 1])
    QS.set_output(True).set_data_type(cudnn.data_type.FP8_E5M3)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    assert compiled.chain.quants[0].axis == 1 and compiled.chain.quants[0].scale_dtype == "fp8_e5m3"

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    q = torch.empty(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    q_scale = torch.empty(1, M // block_size, N, dtype=torch.int8, device="cuda")
    compiled(_vp(compiled, a, b, [q, q_scale]))
    torch.cuda.synchronize()

    ref_mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref_sw = ref_mm * torch.sigmoid(ref_mm)
    q_ref, scale_ref = _col_quant_reference(ref_sw, block_size, torch.float8_e4m3fn, "e5m3")
    torch.testing.assert_close(q_scale.view(torch.uint8).float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=0, rtol=0)


def test_dense_row_col_dual_quant_f8_reorder() -> None:
    """The cutedsl dual-output pattern: one producer -> row quant + col quant,
    both with F8_128x4 scale reordering."""
    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    M, N, K = 256, 128, 128
    bs = 32
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    S = g.swish(input=C, name="sw")
    Qr, QSr = g.block_scale_quantize(input=S, block_size=bs, axis=-1, name="qrow")
    Qr.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QSr.set_dim([1, M, ((N // bs) + 3) // 4 * 4]).set_stride([M * (((N // bs) + 3) // 4 * 4), ((N // bs) + 3) // 4 * 4, 1])
    QSr.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    QSr.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
    Qc, QSc = g.block_scale_quantize(input=S, block_size=bs, axis=1, name="qcol")
    Qc.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QSc.set_dim([1, ((N + 127) // 128) * 128, ((M // bs) + 3) // 4 * 4]).set_stride(
        [((N + 127) // 128) * 128 * (((M // bs) + 3) // 4 * 4), ((M // bs) + 3) // 4 * 4, 1]
    )
    QSc.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    QSc.set_reordering_type(cudnn.tensor_reordering.F8_128x4)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    chain = compiled.chain
    assert len(chain.quants) == 2 and chain.quants[0].axis in (-1, 2) and chain.quants[1].axis == 1

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    q_row = torch.empty(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    qs_row = torch.zeros(1, M, ((N // bs) + 3) // 4 * 4, dtype=torch.float8_e8m0fnu, device="cuda")
    q_col = torch.empty(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    qs_col = torch.zeros(1, ((N + 127) // 128) * 128, ((M // bs) + 3) // 4 * 4, dtype=torch.float8_e8m0fnu, device="cuda")
    # recorder slot order: row-quant data, col-quant data, then scales in
    # quants order (row first).
    compiled(_vp(compiled, a, b, [q_row, q_col, qs_row, qs_col]))
    torch.cuda.synchronize()

    ref_mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref_sw = ref_mm * torch.sigmoid(ref_mm)
    qr_ref, sr_ref = _block_quant_reference(ref_sw, bs, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    qc_ref, sc_ref = _col_quant_reference(ref_sw, bs, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    torch.testing.assert_close(q_row.float(), qr_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q_col.float(), qc_ref.float(), atol=0, rtol=0)
    got_sr = qs_row.view(1, -1)[:, _f8_row_scale_addr(M, N, bs)]
    torch.testing.assert_close(got_sr.float(), sr_ref.float(), atol=0, rtol=0)
    got_sc = qs_col.view(1, -1)[:, _f8_col_scale_addr(M, N, bs)]
    torch.testing.assert_close(got_sc.float(), sc_ref.permute(0, 2, 1).float(), atol=0, rtol=0)


_E2M1_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _unpack_e2m1(u8: torch.Tensor, N: int) -> torch.Tensor:
    """(B, M, N/2) packed uint8 -> (B, M, N) float values, low nibble first."""
    vals = torch.tensor(_E2M1_VALS, device=u8.device)
    lo = vals[(u8 & 0xF).long()]
    hi = vals[(u8 >> 4).long()]
    return torch.stack([lo, hi], dim=-1).view(u8.shape[0], u8.shape[1], N)


def _to_e2m1_rn(x: torch.Tensor) -> torch.Tensor:
    cands = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    ax = x.abs().clamp(max=6.0)
    idx = torch.bucketize(ax, bounds, right=True)
    tie = (idx > 0) & (ax == bounds[(idx - 1).clamp(min=0)])
    idx = torch.where(tie & (idx % 2 == 1), idx - 1, idx)
    v = cands[idx]
    return torch.where(x < 0, -v, v)


def _fp4_quant_ref(x, bs, scale_dtype, axis):
    B, M, N = x.shape
    if axis == 1:
        blocks = x.view(B, M // bs, bs, N)
        s = blocks.abs().amax(dim=2) / 6.0
    else:
        blocks = x.view(B, M, N // bs, bs)
        s = blocks.abs().amax(dim=-1) / 6.0
    if scale_dtype is torch.float8_e8m0fnu:
        safe = torch.where(s > 0, s, 1.0)
        s = torch.where(s > 0, torch.pow(2.0, torch.ceil(torch.log2(safe))), 0.0)
    scale = s.to(scale_dtype)
    inv = torch.where(scale.float() > 0, scale.float().reciprocal(), 0.0)
    inv = inv.unsqueeze(2) if axis == 1 else inv.unsqueeze(-1)
    q = _to_e2m1_rn((blocks * inv).clamp(-6.0, 6.0))
    return q.view(B, M, N), scale


def _fp4_dual_graph(M, N, K, fp4_axis, fp4_bs, fp4_scale_dt):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    S = g.swish(input=C, name="sw")
    Q4, QS4 = g.block_scale_quantize(input=S, block_size=fp4_bs, axis=fp4_axis, name="q4")
    Q4.set_output(True).set_data_type(cudnn.data_type.FP4_E2M1)
    if fp4_axis == 1:
        QS4.set_dim([1, M // fp4_bs, N]).set_stride([M // fp4_bs * N, N, 1])
    else:
        QS4.set_dim([1, M, N // fp4_bs]).set_stride([M * (N // fp4_bs), N // fp4_bs, 1])
    QS4.set_output(True).set_data_type(fp4_scale_dt)
    Q8, QS8 = g.block_scale_quantize(input=S, block_size=32, axis=-1, name="q8")
    Q8.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QS8.set_dim([1, M, N // 32]).set_stride([M * (N // 32), N // 32, 1])
    QS8.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    return g


@pytest.mark.parametrize("fp4_axis", [-1, 1], ids=["row", "col"])
def test_dense_mxfp4_quant_output(fp4_axis) -> None:
    """mxfp4 (fp4 data + e8m0 SF, block32) quant tap beside an fp8 row quant."""
    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    M, N, K = 256, 128, 128
    g = _fp4_dual_graph(M, N, K, fp4_axis, 32, cudnn.data_type.FP8_E8M0)
    compiled = _plan(g, config=cfg, cta_group=cta_group)
    chain = compiled.chain
    assert chain.output_specs[0].dtype == "fp4_e2m1" and chain.output_specs[1].dtype == "fp8_e4m3"

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    q8 = torch.empty(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    qs8 = torch.empty(1, M, N // 32, dtype=torch.float8_e8m0fnu, device="cuda")
    q4 = torch.zeros(1, M, N // 2, dtype=torch.uint8, device="cuda")
    qs4_shape = (1, M // 32, N) if fp4_axis == 1 else (1, M, N // 32)
    qs4 = torch.empty(*qs4_shape, dtype=torch.float8_e8m0fnu, device="cuda")
    compiled(_vp(compiled, a, b, [q4, q8, qs4, qs8]))
    torch.cuda.synchronize()

    ref_mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref_sw = ref_mm * torch.sigmoid(ref_mm)
    q4_ref, qs4_ref = _fp4_quant_ref(ref_sw, 32, torch.float8_e8m0fnu, fp4_axis)
    torch.testing.assert_close(qs4.float(), qs4_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(_unpack_e2m1(q4, N), q4_ref, atol=0, rtol=0)
    q8_ref, qs8_ref = _block_quant_reference(ref_sw, 32, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    torch.testing.assert_close(qs8.float(), qs8_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q8.float(), q8_ref.float(), atol=0, rtol=0)


def test_dense_nvfp4_quant_output() -> None:
    """nvfp4 (fp4 data + e4m3 SF, block16) quant tap; slot0 = bf16 pre-quant tap
    (vsize 16). Dequantized-value tolerance (rcp_approx of a non-pow2 e4m3
    scale is inexact vs the torch reference)."""
    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    M, N, K = 256, 128, 128
    bs = 16
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    S = g.swish(input=C, name="sw")
    Q4, QS4 = g.block_scale_quantize(input=S, block_size=bs, axis=-1, name="q4")
    Q4.set_output(True).set_data_type(cudnn.data_type.FP4_E2M1)
    QS4.set_dim([1, M, N // bs]).set_stride([M * (N // bs), N // bs, 1])
    QS4.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    S.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    compiled = _plan(g, config=cfg, cta_group=cta_group)

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    tap = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    q4 = torch.zeros(1, M, N // 2, dtype=torch.uint8, device="cuda")
    qs4 = torch.empty(1, M, N // bs, dtype=torch.float8_e4m3fn, device="cuda")
    compiled(_vp(compiled, a, b, [tap, q4, qs4]))
    torch.cuda.synchronize()

    ref_mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref_sw = (ref_mm * torch.sigmoid(ref_mm)).to(torch.bfloat16).float()
    _, qs4_ref = _fp4_quant_ref(ref_sw, bs, torch.float8_e4m3fn, -1)
    torch.testing.assert_close(qs4.float(), qs4_ref.float(), atol=0, rtol=0.07)
    deq = _unpack_e2m1(q4, N) * qs4.float().repeat_interleave(bs, dim=2)
    err = (deq - ref_sw).abs()
    tol = 0.34 * ref_sw.abs() + 0.05 * ref_sw.abs().max()
    assert (err <= tol).float().mean().item() > 0.999


@pytest.mark.parametrize("fp4_axis,bs,scale_dt", [(-1, 16, "e4m3"), (-1, 32, "e8m0"), (1, 32, "e8m0")], ids=["nvfp4-row", "mxfp4-row", "mxfp4-col"])
def test_dense_sole_fp4_quant_output(fp4_axis, bs, scale_dt) -> None:
    """fp4 quant data as the ONLY output (slot 0): Phase C codegen-emitted store."""
    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    M, N, K = 256, 128, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    S = g.swish(input=C, name="sw")
    Q4, QS4 = g.block_scale_quantize(input=S, block_size=bs, axis=fp4_axis, name="q4")
    Q4.set_output(True).set_data_type(cudnn.data_type.FP4_E2M1)
    scale_cudnn = cudnn.data_type.FP8_E4M3 if scale_dt == "e4m3" else cudnn.data_type.FP8_E8M0
    scale_torch = torch.float8_e4m3fn if scale_dt == "e4m3" else torch.float8_e8m0fnu
    if fp4_axis == 1:
        QS4.set_dim([1, M // bs, N]).set_stride([M // bs * N, N, 1])
    else:
        QS4.set_dim([1, M, N // bs]).set_stride([M * (N // bs), N // bs, 1])
    QS4.set_output(True).set_data_type(scale_cudnn)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    assert compiled.chain.output_specs[0].dtype == "fp4_e2m1"

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    q4 = torch.zeros(1, M, N // 2, dtype=torch.uint8, device="cuda")
    qs_shape = (1, M // bs, N) if fp4_axis == 1 else (1, M, N // bs)
    qs4 = torch.empty(*qs_shape, dtype=scale_torch, device="cuda")
    compiled(_vp(compiled, a, b, [q4, qs4]))
    torch.cuda.synchronize()

    ref_mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref_sw = ref_mm * torch.sigmoid(ref_mm)
    q4_ref, qs4_ref = _fp4_quant_ref(ref_sw, bs, scale_torch, fp4_axis)
    if scale_dt == "e8m0":
        torch.testing.assert_close(qs4.float(), qs4_ref.float(), atol=0, rtol=0)
        torch.testing.assert_close(_unpack_e2m1(q4, N), q4_ref, atol=0, rtol=0)
    else:
        torch.testing.assert_close(qs4.float(), qs4_ref.float(), atol=0, rtol=0.07)
        deq = _unpack_e2m1(q4, N) * (qs4.float().repeat_interleave(bs, dim=1) if fp4_axis == 1 else qs4.float().repeat_interleave(bs, dim=2))
        err = (deq - ref_sw).abs()
        tol = 0.34 * ref_sw.abs() + 0.05 * ref_sw.abs().max()
        assert (err <= tol).float().mean().item() > 0.999


@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_64x128x128_64x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",
    ],
    ids=["m128-1cta", "m64-1cta", "m128-2cta"],
)
def test_dense_col_block16_quant(config_name) -> None:
    """block16 col quant: half-warp redux; the cta_tile_m=64 1-CTA packed
    layout (16 active rows per warp = exactly one block) is supported."""
    cfg, cta_group = _resolve(config_name)
    M, N, K, bs = 256, 128, 128, 16
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    S = g.swish(input=C, name="sw")
    Q, QS = g.block_scale_quantize(input=S, block_size=bs, axis=1, name="q")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QS.set_dim([1, M // bs, N]).set_stride([M // bs * N, N, 1])
    QS.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    q = torch.zeros(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    qs = torch.zeros(1, M // bs, N, dtype=torch.float8_e8m0fnu, device="cuda")
    compiled(_vp(compiled, a, b, [q, qs]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    sw = mm * torch.sigmoid(mm)
    q_ref, s_ref = _col_quant_reference(sw, bs, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    torch.testing.assert_close(qs.float(), s_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=0, rtol=0)


def test_dense_block16_fp8_quant() -> None:
    """block_size=16 row quant with fp8 data (vsize pinned to the block)."""
    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    M, N, K, bs = 256, 128, 128, 16
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    S = g.swish(input=C, name="sw")
    Q, QS = g.block_scale_quantize(input=S, block_size=bs, axis=-1, name="q")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QS.set_dim([1, M, N // bs]).set_stride([M * (N // bs), N // bs, 1])
    QS.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    q = torch.empty(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    qs = torch.empty(1, M, N // bs, dtype=torch.float8_e8m0fnu, device="cuda")
    compiled(_vp(compiled, a, b, [q, qs]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    sw = mm * torch.sigmoid(mm)
    q_ref, s_ref = _block_quant_reference(sw, bs, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    torch.testing.assert_close(qs.float(), s_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=0, rtol=0)


def test_dense_mixed_block_quants() -> None:
    """Two quants with DIFFERENT block sizes in one graph: nvfp4 row block16 +
    fp8 row block32 (vsize = max block; the small block quantizes per
    sub-chunk), plus a col16 alongside a row32 variant."""
    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    M, N, K = 256, 128, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    S = g.swish(input=C, name="sw")
    Q4, QS4 = g.block_scale_quantize(input=S, block_size=16, axis=-1, name="q4")
    Q4.set_output(True).set_data_type(cudnn.data_type.FP4_E2M1)
    QS4.set_dim([1, M, N // 16]).set_stride([M * (N // 16), N // 16, 1])
    QS4.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    Q8, QS8 = g.block_scale_quantize(input=S, block_size=32, axis=-1, name="q8")
    Q8.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QS8.set_dim([1, M, N // 32]).set_stride([M * (N // 32), N // 32, 1])
    QS8.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    Qc, QSc = g.block_scale_quantize(input=S, block_size=16, axis=1, name="qc")
    Qc.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QSc.set_dim([1, M // 16, N]).set_stride([M // 16 * N, N, 1])
    QSc.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    assert sorted(q.block_size for q in compiled.chain.quants) == [16, 16, 32]

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    q4 = torch.zeros(1, M, N // 2, dtype=torch.uint8, device="cuda")
    qs4 = torch.zeros(1, M, N // 16, dtype=torch.float8_e4m3fn, device="cuda")
    q8 = torch.zeros(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    qs8 = torch.zeros(1, M, N // 32, dtype=torch.float8_e8m0fnu, device="cuda")
    qc = torch.zeros(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    qsc = torch.zeros(1, M // 16, N, dtype=torch.float8_e8m0fnu, device="cuda")
    compiled(_vp(compiled, a, b, [q4, q8, qc, qs4, qs8, qsc]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    sw = mm * torch.sigmoid(mm)
    # fp8 row block32: bit-exact (e8m0 scale)
    q8_ref, s8_ref = _block_quant_reference(sw, 32, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    torch.testing.assert_close(qs8.float(), s8_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q8.float(), q8_ref.float(), atol=0, rtol=0)
    # col16: bit-exact (e8m0 scale)
    qc_ref, sc_ref = _col_quant_reference(sw, 16, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    torch.testing.assert_close(qsc.float(), sc_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(qc.float(), qc_ref.float(), atol=0, rtol=0)
    # nvfp4 row block16: e4m3 scale 1-ulp + dequant tolerance
    _, s4_ref = _fp4_quant_ref(sw, 16, torch.float8_e4m3fn, -1)
    torch.testing.assert_close(qs4.float(), s4_ref.float(), atol=0, rtol=0.07)
    deq = _unpack_e2m1(q4, N) * qs4.float().repeat_interleave(16, dim=2)
    err = (deq - sw).abs()
    tol = 0.34 * sw.abs() + 0.05 * sw.abs().max()
    assert (err <= tol).float().mean().item() > 0.999


def test_dense_col_quant_rejections() -> None:
    def _col_graph(M, N, K, block_size):
        g = cudnn.pygraph(
            io_data_type=cudnn.data_type.BFLOAT16,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
        B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
        C = g.matmul(A=A, B=B, name="mm")
        S = g.swish(input=C, name="sw")
        Q, QS = g.block_scale_quantize(input=S, block_size=block_size, axis=1, name="q")
        Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
        QS.set_dim([1, M // block_size, N]).set_stride([M // block_size * N, N, 1])
        QS.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
        return g

    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    with pytest.raises(ValueError, match="divisible by block_size"):
        _plan(_col_graph(160 + 8, 128, 128, 32), config=cfg, cta_group=cta_group)
    with pytest.raises(NotImplementedError, match="block_size 32"):
        _plan(_col_graph(256, 128, 128, 8), config=cfg, cta_group=cta_group)


def test_dense_block_scale_quant_with_dense_tap() -> None:
    """Quant data rides slot 0 while the pre-quant producer is also tapped."""
    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    M = N = K = 128
    block_size = 32
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    S = g.swish(input=C, name="sw")
    S.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    Q, QS = g.block_scale_quantize(input=S, block_size=block_size, name="q")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QS.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    specs = compiled.chain.output_specs
    assert len(specs) == 2 and specs[0].quant_idx is None and specs[1].quant_idx == 0
    assert [o.source for o in compiled.chain.outputs] == ["op_0", "quant_0", "quant_scale_0"]

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    q = torch.empty(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    tap = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    q_scale = torch.empty(1, M, N // block_size, dtype=torch.float8_e8m0fnu, device="cuda")
    compiled(_vp(compiled, a, b, [tap, q, q_scale]))
    torch.cuda.synchronize()

    ref_mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref_sw = ref_mm * torch.sigmoid(ref_mm)
    # S's declared bf16 dtype rounds the swish output before every consumer —
    # both the tap store and the quant read the rounded value.
    ref_sw = ref_sw.to(torch.bfloat16).float()
    q_ref, scale_ref = _block_quant_reference(ref_sw, block_size, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    torch.testing.assert_close(tap.float(), ref_sw, atol=0, rtol=0)
    torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=0, rtol=0)


def test_dense_block_scale_quant_with_fp32_dense_tap() -> None:
    """The chunk stays pinned to the quant block even when the widest dense
    output is 4 bytes: 32 elements x 4 B = 128 B, split into four 32 B stores."""
    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    M = N = K = 128
    block_size = 32
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    Q, QS = g.block_scale_quantize(input=C, block_size=block_size, name="q")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QS.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    assert _epi_vec_bytes(compiled.chain, cfg, cta_group) == block_size * 4

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    tap = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    q = torch.empty(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    q_scale = torch.empty(1, M, N // block_size, dtype=torch.float8_e8m0fnu, device="cuda")
    compiled(_vp(compiled, a, b, [tap, q, q_scale]))
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    q_ref, scale_ref = _block_quant_reference(ref, block_size, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    torch.testing.assert_close(tap, ref, atol=0, rtol=0)
    torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=0, rtol=0)


def test_dense_col_block_scale_quant_with_dense_tap() -> None:
    """Col quant behind a 2-byte dense tap. The chunk must stay as wide as the
    32-lane block group: a narrower one has the upper lanes store a scale they
    never computed, to columns past the chunk (and past the scale tensor on the
    last chunk of a row block)."""
    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    M, N, K, bs = 256, 128, 128, 32
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    S = g.swish(input=C, name="sw")
    S.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    Q, QS = g.block_scale_quantize(input=S, block_size=bs, axis=1, name="q")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QS.set_dim([1, M // bs, N]).set_stride([M // bs * N, N, 1])
    QS.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    assert _epi_vec_bytes(compiled.chain, cfg, cta_group) == bs * 2

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    tap = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    q = torch.empty(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    # One guard row past the scale tensor catches an over-wide scale store.
    n_scale = (M // bs) * N
    scale_buf = torch.full((n_scale + N,), 0x7F, dtype=torch.uint8, device="cuda")
    q_scale = scale_buf.view(torch.float8_e8m0fnu)[:n_scale].view(1, M // bs, N)
    compiled(_vp(compiled, a, b, [tap, q, q_scale]))
    torch.cuda.synchronize()

    ref_mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref_sw = (ref_mm * torch.sigmoid(ref_mm)).to(torch.bfloat16).float()
    q_ref, scale_ref = _col_quant_reference(ref_sw, bs, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    torch.testing.assert_close(tap.float(), ref_sw, atol=0, rtol=0)
    torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=0, rtol=0)
    assert (scale_buf[n_scale:] == 0x7F).all()


def test_dense_dual_block_scale_quant() -> None:
    """Two quant nodes fan out from one producer (e4m3 + e5m2 data)."""
    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    M = N = K = 128
    block_size = 32
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    S = g.swish(input=C, name="sw")
    Q1, QS1 = g.block_scale_quantize(input=S, block_size=block_size, name="q1")
    Q1.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QS1.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    Q2, QS2 = g.block_scale_quantize(input=S, block_size=block_size, name="q2")
    Q2.set_output(True).set_data_type(cudnn.data_type.FP8_E5M2)
    QS2.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    chain = compiled.chain
    assert len(chain.quants) == 2
    assert [o.source for o in chain.outputs] == ["quant_0", "quant_1", "quant_scale_0", "quant_scale_1"]

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    q1 = torch.empty(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    qs1 = torch.empty(1, M, N // block_size, dtype=torch.float8_e8m0fnu, device="cuda")
    q2 = torch.empty(1, M, N, dtype=torch.float8_e5m2, device="cuda")
    qs2 = torch.empty(1, M, N // block_size, dtype=torch.float8_e8m0fnu, device="cuda")
    compiled(_vp(compiled, a, b, [q1, q2, qs1, qs2]))
    torch.cuda.synchronize()

    ref_mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref_sw = ref_mm * torch.sigmoid(ref_mm)
    q1_ref, qs1_ref = _block_quant_reference(ref_sw, block_size, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    q2_ref, qs2_ref = _block_quant_reference(ref_sw, block_size, torch.float8_e5m2, torch.float8_e8m0fnu)
    torch.testing.assert_close(qs1.float(), qs1_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q1.float(), q1_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(qs2.float(), qs2_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q2.float(), q2_ref.float(), atol=0, rtol=0)


def test_dense_amax_only_no_dense_output() -> None:
    """Reduction-only graph: pointwise chain + AMAX, no dense output at all."""
    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    M = N = K = 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    R = g.relu(input=C, name="r")
    amax = g.reduction(input=R, mode=cudnn.reduction_mode.AMAX, name="amax")
    amax.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    amax.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    assert not compiled.chain.output_specs
    assert [o.source for o in compiled.chain.outputs] == ["reduction_0"]

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    out_am = torch.zeros(1, 1, 1, dtype=torch.float32, device="cuda")
    compiled(_vp(compiled, a, b, [out_am]))
    torch.cuda.synchronize()

    ref = torch.relu(torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)))
    torch.testing.assert_close(out_am.flatten()[0], ref.abs().amax(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("shape", _WEIRD_SHAPES, ids=[_shape_id(s) for s in _WEIRD_SHAPES])
@pytest.mark.parametrize(
    "a_major,b_major",
    _NONCANONICAL_LAYOUTS,
    ids=[_layout_id(p) for p in _NONCANONICAL_LAYOUTS],
)
@pytest.mark.parametrize(
    "in_dt,out_dt",
    _CORE_DTYPE_PAIRS,
    ids=[_dtype_id(p) for p in _CORE_DTYPE_PAIRS],
)
@pytest.mark.parametrize(
    "config_name",
    _sweep_config_names(),
    ids=[_config_id(n) for n in _sweep_config_names()],
)
def test_input_layout_matmul(
    _compile_cache,
    config_name: str,
    in_dt: str,
    out_dt: str,
    a_major: str,
    b_major: str,
    shape: tuple[int, int, int],
) -> None:
    """A M-major / B N-major inputs over the full (config, dtype, shape) matrix."""
    M, N, K = shape
    cfg, cta_group = _resolve(config_name)
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, a_major, b_major, cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        a_major,
        b_major,
        cta_group=cta_group,
    )
    a, b, c = _mkdata(M, N, K, in_dt, out_dt, a_major=a_major, b_major=b_major)
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _reference(a, b, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    tol = _tolerance(in_dt, out_dt)
    bad = int((diff > tol).sum().item())
    assert bad == 0, (
        f"\n  config:    {config_name}"
        f"\n  dtype:     {in_dt} -> {out_dt}"
        f"\n  layout:    A{a_major}/B{b_major}"
        f"\n  shape:     {M}x{N}x{K}"
        f"\n  bad:       {bad}/{diff.numel()}"
        f"\n  max|diff|: {float(diff.max().item()):.4g}  (tol={tol})"
    )


@pytest.mark.parametrize(
    "a_major,b_major",
    _NONPACKED_LAYOUTS,
    ids=[_layout_id(p) for p in _NONPACKED_LAYOUTS],
)
@pytest.mark.parametrize(
    "in_dt,out_dt",
    _NONPACKED_DTYPE_PAIRS,
    ids=[_dtype_id(p) for p in _NONPACKED_DTYPE_PAIRS],
)
@pytest.mark.parametrize(
    "config_name",
    _NONPACKED_CONFIGS,
    ids=[_config_id(n) for n in _NONPACKED_CONFIGS],
)
def test_nonpacked_batched_matmul(
    _compile_cache,
    config_name: str,
    in_dt: str,
    out_dt: str,
    a_major: str,
    b_major: str,
) -> None:
    """Padded A/B/C views exercise dynamic strides in TMA descriptors and stores."""
    batch, M, N, K = 2, 256, 256, 256
    cfg, cta_group = _resolve(config_name)
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, a_major, b_major, cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_batched_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        a_major,
        b_major,
        cta_group=cta_group,
    )
    a, b, c = _mkbatched_nonpacked_data(batch, M, N, K, in_dt, out_dt, a_major=a_major, b_major=b_major)
    assert not a.is_contiguous() and not b.is_contiguous() and not c.is_contiguous()

    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _reference(a, b, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    tol = _tolerance(in_dt, out_dt)
    bad = int((diff > tol).sum().item())
    assert bad == 0, (
        f"\n  config:    {config_name}"
        f"\n  dtype:     {in_dt} -> {out_dt}"
        f"\n  layout:    A{a_major}/B{b_major}"
        f"\n  strides:   A{tuple(a.stride())} B{tuple(b.stride())} C{tuple(c.stride())}"
        f"\n  max|diff|: {float(diff.max().item()):.4g}  (tol={tol})"
    )


@pytest.mark.parametrize(
    "config_name",
    _NONPACKED_CONFIGS[:2],
    ids=[_config_id(n) for n in _NONPACKED_CONFIGS[:2]],
)
def test_zero_stride_broadcast_input_matmul(
    _compile_cache,
    config_name: str,
) -> None:
    batch, M, N, K = 2, 256, 256, 256
    in_dt = out_dt = "bf16"
    cfg, cta_group = _resolve(config_name)
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, "k", "k", cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_batched_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        "k",
        "k",
        cta_group=cta_group,
    )
    a, b, c = _mkbatched_zero_stride_input_data(batch, M, N, K, in_dt, out_dt)
    assert a.stride() == (0, 0, 1)
    assert b.stride() == (0, 0, 1)
    assert not c.is_contiguous()

    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _reference(a, b, out_dt)
    torch.testing.assert_close(c, ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    "shape",
    _BATCHED_SHAPES,
    ids=[_batched_shape_id(s) for s in _BATCHED_SHAPES],
)
@pytest.mark.parametrize(
    "in_dt,out_dt",
    _CORE_DTYPE_PAIRS,
    ids=[_dtype_id(p) for p in _CORE_DTYPE_PAIRS],
)
@pytest.mark.parametrize(
    "config_name",
    _BATCHED_CONFIGS,
    ids=[_config_id(n) for n in _BATCHED_CONFIGS],
)
def test_batched_matmul(
    _compile_cache,
    config_name: str,
    in_dt: str,
    out_dt: str,
    shape: tuple[int, int, int, int],
) -> None:
    """Rank-3 matmul keeps batch as the native L mode and maps it to grid.z."""
    batch, M, N, K = shape
    cfg, cta_group = _resolve(config_name)
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, cta_group=cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_batched_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        cta_group=cta_group,
    )

    a, b, c = _mkbatched_data(batch, M, N, K, in_dt, out_dt)
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _reference(a, b, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    tol = _tolerance(in_dt, out_dt)
    bad = int((diff > tol).sum().item())

    assert bad == 0, (
        f"\n  config:    {config_name}"
        f"\n  dtype:     {in_dt} -> {out_dt}"
        f"\n  shape:     B{batch} {M}x{N}x{K}"
        f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
        f"\n  max|diff|: {float(diff.max().item()):.4g}  (tol={tol})"
        f"\n  max|ref|:  {float(ref.abs().max().item()):.4g}"
        f"\n  hint:      sample c[0,0,:8]   = {c[0, 0, :8].to(torch.float32).tolist()}"
        f"\n             sample ref[0,0,:8] = {ref[0, 0, :8].to(torch.float32).tolist()}"
    )


@pytest.mark.parametrize(
    "a_major,b_major",
    _INPUT_LAYOUTS,
    ids=[_layout_id(p) for p in _INPUT_LAYOUTS],
)
def test_input_layout_batched_matmul(
    _compile_cache,
    a_major: str,
    b_major: str,
) -> None:
    """Rank-3 layout coverage keeps batch native while varying A/B major mode."""
    config_name = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"
    cfg, cta_group = _resolve(config_name)
    batch, M, N, K = 2, 256, 256, 256
    in_dt = out_dt = "bf16"
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, a_major, b_major, cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_batched_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        a_major,
        b_major,
        cta_group=cta_group,
    )
    a, b, c = _mkbatched_data(batch, M, N, K, in_dt, out_dt, a_major=a_major, b_major=b_major)
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _reference(a, b, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    assert int((diff > _tolerance(in_dt, out_dt)).sum().item()) == 0


@pytest.mark.parametrize(
    "broadcast_side,shape",
    _BATCH_BROADCAST_CASES,
    ids=[_batch_broadcast_id(p) for p in _BATCH_BROADCAST_CASES],
)
@pytest.mark.parametrize(
    "in_dt,out_dt",
    _CORE_DTYPE_PAIRS,
    ids=[_dtype_id(p) for p in _CORE_DTYPE_PAIRS],
)
@pytest.mark.parametrize(
    "config_name",
    _BATCHED_CONFIGS,
    ids=[_config_id(n) for n in _BATCHED_CONFIGS],
)
def test_batch_broadcast_matmul(
    _compile_cache,
    config_name: str,
    in_dt: str,
    out_dt: str,
    broadcast_side: str,
    shape: tuple[int, int, int, int],
) -> None:
    """Rank-3 matmul with one input broadcast across the output batch."""
    batch, M, N, K = shape
    cfg, cta_group = _resolve(config_name)
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, cta_group=cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_batch_broadcast_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        broadcast_side,
        cta_group=cta_group,
    )

    a, b, c = _mkbatch_broadcast_data(batch, M, N, K, in_dt, out_dt, broadcast_side)
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _batch_broadcast_reference(a, b, batch, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    tol = _tolerance(in_dt, out_dt)
    bad = int((diff > tol).sum().item())

    assert bad == 0, (
        f"\n  config:    {config_name}"
        f"\n  dtype:     {in_dt} -> {out_dt}"
        f"\n  broadcast: {broadcast_side}"
        f"\n  shape:     B{batch} {M}x{N}x{K}"
        f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
        f"\n  max|diff|: {float(diff.max().item()):.4g}  (tol={tol})"
        f"\n  max|ref|:  {float(ref.abs().max().item()):.4g}"
        f"\n  hint:      sample c[0,0,:8]   = {c[0, 0, :8].to(torch.float32).tolist()}"
        f"\n             sample ref[0,0,:8] = {ref[0, 0, :8].to(torch.float32).tolist()}"
    )


@pytest.mark.parametrize(
    "a_major,b_major",
    _INPUT_LAYOUTS,
    ids=[_layout_id(p) for p in _INPUT_LAYOUTS],
)
@pytest.mark.parametrize("broadcast_side", ("A", "B"), ids=("bcastA", "bcastB"))
def test_input_layout_batch_broadcast_matmul(
    _compile_cache,
    broadcast_side: str,
    a_major: str,
    b_major: str,
) -> None:
    """Input-layout coverage when one operand is broadcast across batch."""
    config_name = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"
    cfg, cta_group = _resolve(config_name)
    batch, M, N, K = 3, 256, 256, 256
    in_dt = out_dt = "bf16"
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, a_major, b_major, cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_batch_broadcast_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        broadcast_side,
        a_major,
        b_major,
        cta_group=cta_group,
    )
    a, b, c = _mkbatch_broadcast_data(
        batch,
        M,
        N,
        K,
        in_dt,
        out_dt,
        broadcast_side,
        a_major=a_major,
        b_major=b_major,
    )
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _batch_broadcast_reference(a, b, batch, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    assert int((diff > _tolerance(in_dt, out_dt)).sum().item()) == 0


# Mixed FP8: tcgen05 F8F6F4 takes A/B FP8 variants independently, so every
# {E4M3,E5M2}² pair is valid. The main sweep only drives A==B; this covers mixed.

_FP8_AB_PAIRS = [
    ("fp8_e4m3", "fp8_e4m3"),
    ("fp8_e5m2", "fp8_e5m2"),
    ("fp8_e4m3", "fp8_e5m2"),  # mixed
    ("fp8_e5m2", "fp8_e4m3"),  # mixed
]
_FP8_MIXED_CONFIGS = [
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
]


@pytest.mark.parametrize("a_dt,b_dt", _FP8_AB_PAIRS, ids=[f"{a}_x_{b}" for a, b in _FP8_AB_PAIRS])
@pytest.mark.parametrize("config_name", _FP8_MIXED_CONFIGS, ids=[_config_id(n) for n in _FP8_MIXED_CONFIGS])
def test_mixed_fp8_matmul(config_name: str, a_dt: str, b_dt: str) -> None:
    """A and B each an arbitrary FP8 variant -> FP16 out, bit-exact vs fp32."""
    cfg, cta_group = _resolve(config_name)
    M = N = K = 256

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(
        name="A",
        dim=[1, M, K],
        stride=_a_stride_batched(M, K, "k"),
        data_type=_CUDNN_DTYPE[a_dt],
    )
    B = g.tensor(
        name="B",
        dim=[1, K, N],
        stride=_b_stride_batched(N, K, "k"),
        data_type=_CUDNN_DTYPE[b_dt],
    )
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    C.set_data_type(cudnn.data_type.HALF)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    assert compiled.chain.matmul.a_dtype == a_dt
    assert compiled.chain.matmul.b_dtype == b_dt

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=_TORCH_DTYPE[a_dt], device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=_TORCH_DTYPE[b_dt], device="cuda")
    c = torch.empty(1, M, N, dtype=torch.float16, device="cuda")

    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)).to(torch.float16)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    assert int((diff > 1e-1).sum().item()) == 0, f"{config_name} {a_dt} x {b_dt}: max|diff|={diff.max().item():.4g}"


# INT8 × INT8 → INT32 (integer tensor-core MMA). Epilogue widens int32 → fp32;
# bit-exact vs an fp32 reference (small-magnitude products are exact).

_INT8_CONFIGS = [
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
]


# Output dtype → (cudnn enum, torch dtype, input value range). fp8 needs tiny
# inputs so the int32 accumulator stays in fp8's range; others use a wider range.
_INT8_OUT_DTYPES = {
    "fp32": (cudnn.data_type.FLOAT, torch.float32, 8),
    "bf16": (cudnn.data_type.BFLOAT16, torch.bfloat16, 8),
    "fp16": (cudnn.data_type.HALF, torch.float16, 8),
    "int32": (cudnn.data_type.INT32, torch.int32, 8),
    "fp8_e4m3": (cudnn.data_type.FP8_E4M3, torch.float8_e4m3fn, 1),
}


@requires_int8_mma
@pytest.mark.parametrize("config_name", _INT8_CONFIGS, ids=[_config_id(n) for n in _INT8_CONFIGS])
@pytest.mark.parametrize("out_dt", list(_INT8_OUT_DTYPES))
def test_int8_matmul(config_name: str, out_dt: str) -> None:
    """INT8×INT8→INT32, output ∈ {fp32,bf16,fp16,int32,fp8}; bit-exact vs a
    rounded integer reference (values small enough that the rounding is exact)."""
    cfg, cta_group = _resolve(config_name)
    M = N = K = 256
    cudnn_dt, torch_dt, vmax = _INT8_OUT_DTYPES[out_dt]

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.INT8,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.INT32,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    C.set_data_type(cudnn_dt)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    assert compiled.chain.matmul.accum_dtype == "int32"
    assert compiled.chain.output_dtype == out_dt

    torch.manual_seed(0)
    a = torch.randint(-vmax, vmax, (1, M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-vmax, vmax, (1, N, K), dtype=torch.int8, device="cuda")
    c = torch.empty(1, M, N, dtype=torch_dt, device="cuda")

    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.cpu().to(torch.int64), b.cpu().to(torch.int64))
    if out_dt == "int32":
        diff = (c.cpu().to(torch.int64) - ref).abs().max().item()
    else:
        diff = (c.float().cpu() - ref.to(torch_dt).float()).abs().max().item()
    assert diff == 0.0, f"{config_name} -> {out_dt}: max|diff|={diff} (expected bit-exact)"


# M-major batched output. The 32B store-alignment rule moves from N to M
# (out_major="m"), so M-OOB menu shapes SKIP here.


@pytest.mark.parametrize("shape", _WEIRD_SHAPES, ids=[_shape_id(s) for s in _WEIRD_SHAPES])
@pytest.mark.parametrize(
    "in_dt,out_dt",
    _CORE_DTYPE_PAIRS,
    ids=[_dtype_id(p) for p in _CORE_DTYPE_PAIRS],
)
@pytest.mark.parametrize(
    "config_name",
    _sweep_config_names(),
    ids=[_config_id(n) for n in _sweep_config_names()],
)
def test_m_major_output_batched(
    _compile_cache,
    config_name: str,
    in_dt: str,
    out_dt: str,
    shape: tuple[int, int, int],
) -> None:
    """M-major + batch>1 over the full (config, dtype, shape) matrix: covers
    the TMA-store and STG m-major store paths."""
    batch = 3
    M, N, K = shape
    cfg, cta_group = _resolve(config_name)
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, cta_group=cta_group, out_major="m")
    if not ok:
        pytest.skip(reason)
    compiled = _get_batched_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        cta_group=cta_group,
        out_major="m",
    )
    a, b, c = _mkbatched_data(batch, M, N, K, in_dt, out_dt, out_major="m")
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()
    ref = _reference(a, b, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    tol = _tolerance(in_dt, out_dt)
    bad = int((diff > tol).sum().item())
    assert bad == 0, (
        f"\n  config:    {config_name}"
        f"\n  dtype:     {in_dt} -> {out_dt}"
        f"\n  shape:     B{batch}_{M}x{N}x{K} (M-major out)"
        f"\n  bad:       {bad}/{diff.numel()}"
        f"\n  max|diff|: {float(diff.max().item()):.4g}  (tol={tol})"
    )


def _mm_plan(b_major: str):
    """A plain bf16 matmul plan (M=N=256, K=128) with B declared ``b_major``."""
    M = N = 256
    K = 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, b_major))
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    cfg, cta_group = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    return _plan(g, config=cfg, cta_group=cta_group), M, N, K


def test_operand_k_mismatch_rejected():
    """M/N/K are inferred from the FIRST A and B buffer, so a B whose K disagrees
    would be read past its end. Reject instead of running off the buffer."""
    compiled, M, N, K = _mm_plan("k")
    a = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(1, M, N, device="cuda", dtype=torch.bfloat16)
    good = torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16)
    compiled(_vp(compiled, a, good, c))  # baseline: the matching operand runs
    short = torch.randn(1, N, K // 2, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="operand shapes disagree"):
        compiled(_vp(compiled, a, short, c))


@pytest.mark.parametrize("b_major", ["k", "n"])
def test_operand_major_mismatch_rejected(b_major: str):
    """The B major is baked into the TMA/MMA descriptors at JIT time while the
    launch reads the runtime strides — a mismatched buffer computes silently
    wrong numbers, so it must be rejected."""
    compiled, M, N, K = _mm_plan(b_major)
    a = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(1, M, N, device="cuda", dtype=torch.bfloat16)
    # (1, N, K) contiguous is K-major; its transpose-view is N-major.
    b_k = torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16)
    b_n = torch.randn(1, K, N, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    matching, mismatched = (b_k, b_n) if b_major == "k" else (b_n, b_k)

    compiled(_vp(compiled, a, matching, c))
    torch.cuda.synchronize()
    ref = torch.einsum("bmk,bnk->bmn", a.float(), matching.float())
    torch.testing.assert_close(c.float(), ref, atol=2e-1, rtol=2e-2)

    with pytest.raises(ValueError, match="operand layout does not match"):
        compiled(_vp(compiled, a, mismatched, c))


# Standalone CLI shim — forwards remaining argv to pytest on this file.


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))


def test_cache_dir_falls_back_when_unwritable(tmp_path, monkeypatch, caplog):
    """An unwritable cache location must degrade to a temp dir with a warning,
    not abort the compile with a raw PermissionError."""
    import logging

    from cudnn.gemm.frost.compiler import _cache_dir, _fallback_cache_dir, _usable_cache_dir

    # A regular file as a path component makes mkdir raise even for root
    # (CI runs as root, where a chmod-0o500 directory is still writable).
    readonly = tmp_path / "readonly"
    readonly.touch()
    try:
        _usable_cache_dir.cache_clear()
        monkeypatch.setenv("CUDNN_FRONTEND_GEMM_KERNEL_CACHE", str(readonly / "nested"))
        with caplog.at_level(logging.WARNING, logger="cudnn.gemm.frost.compiler"):
            got = _cache_dir()
        assert got == _fallback_cache_dir()
        assert os.access(got, os.W_OK)
        assert any("falling back" in r.getMessage() for r in caplog.records)
        assert _cache_dir() == got  # cached: diagnosed once, not per compile
    finally:
        _usable_cache_dir.cache_clear()

    _usable_cache_dir.cache_clear()
    monkeypatch.setenv("CUDNN_FRONTEND_GEMM_KERNEL_CACHE", str(tmp_path / "writable"))
    assert _cache_dir() == tmp_path / "writable"
    _usable_cache_dir.cache_clear()


def test_generated_kernel_write_is_atomic(tmp_path, monkeypatch):
    """Concurrent JITs of one kernel share a content-addressed path, so the write
    must publish by rename: the target never holds partial source and a failed
    write leaves neither a truncated target nor a stray temp file."""
    from cudnn.gemm.frost.compiler import _write_atomic

    target = tmp_path / "generated_kernel.py"
    src = "X = [\n" + "".join(f"    {i},\n" for i in range(5000)) + "]\n"

    _write_atomic(target, src)
    assert target.read_text() == src

    target.write_text("OLD")
    boom = OSError("rename failed")

    def _raise(*_a, **_k):
        raise boom

    monkeypatch.setattr(os, "replace", _raise)
    with pytest.raises(OSError):
        _write_atomic(target, src)
    assert target.read_text() == "OLD"  # never truncated in place
    monkeypatch.undo()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["generated_kernel.py"]


def test_import_kernel_publishes_by_rename(tmp_path, monkeypatch):
    """The atomic write must be wired into _import_kernel, not merely available:
    the generated module has to reach its cache path through os.replace."""
    from cudnn.gemm.frost.compiler import _import_kernel, _usable_cache_dir

    _usable_cache_dir.cache_clear()
    monkeypatch.setenv("CUDNN_FRONTEND_GEMM_KERNEL_CACHE", str(tmp_path))
    real_replace = os.replace
    renamed = []

    def _spy(src, dst, *a, **k):
        renamed.append(str(dst))
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", _spy)
    try:
        mod = _import_kernel("MARKER = 4321\n")
    finally:
        _usable_cache_dir.cache_clear()
    assert mod.MARKER == 4321
    assert any(name.endswith("generated_kernel.py") for name in renamed)


# --- num_mma_m geometry invariants ------------------------------------------
#
# The CTA tile may span several `tcgen05.mma` along M (`num_mma_m = cta_tile_m /
# mma_inst_m`). The e2e behaviour rides `_QUICK_CONFIGS` above; what needs its own
# assertion is the geometry contract those configs rely on.


@pytest.mark.parametrize(
    "name,num_mma_m",
    [
        ("CONFIG_sm100_128x128x128_128x128x32_cluster1x1", 1),
        ("CONFIG_sm100_256x128x128_128x128x32_cluster1x1", 2),
        ("CONFIG_sm100_256x256x128_128x256x32_cluster1x1", 2),
        ("CONFIG_sm100_128x128x128_64x128x32_cluster1x1", 2),
    ],
)
def test_num_mma_m_is_derived_from_the_two_tiles(name: str, num_mma_m: int) -> None:
    cfg = by_name(name)
    assert cfg.num_mma_m == num_mma_m
    assert cfg.cta_tile_m == cfg.num_mma_m * cfg.mma_inst_m
    assert cfg.mma_inst_n == cfg.cta_tile_n  # N is never split
    # The epilogue drains one MMA-M block per pass, so its subtile height is the
    # instruction's M, not the CTA tile's.
    assert cfg.epi_tile_mn[0] == cfg.mma_inst_m


@pytest.mark.parametrize(
    "name,reason",
    [
        # The pre-existing M/N bounds read off the MMA INSTRUCTION tile now that
        # the CTA tile can be several instructions wide.
        ("CONFIG_sm100_128x128x128_32x128x32_cluster1x1", "mma_inst_m=32"),
        ("CONFIG_sm100_128x512x128_128x512x32_cluster1x1", "mma_inst_n=512"),
        ("CONFIG_sm100_128x24x128_128x12x32_cluster1x1", "mma_inst_n=12"),
        # At most 2 instructions along M this pass...
        ("CONFIG_sm100_256x128x128_64x128x32_cluster1x1", "num_mma_m=4"),
        ("CONFIG_sm100_512x128x128_128x128x32_cluster1x1", "num_mma_m=4"),
        # ... and N is not an instruction-count axis at all.
        ("CONFIG_sm100_128x256x128_128x128x32_cluster1x1", "N is not split"),
    ],
)
def test_illegal_mma_decomposition_rejected(name: str, reason: str) -> None:
    with pytest.raises(NotImplementedError) as e:
        by_name(name)
    assert reason in str(e.value)


def test_catalog_enumerates_the_mma_m_axis() -> None:
    """M is enumerated as (mma_inst_m, num_mma_m); `cta_tile_m` is their PRODUCT,
    not an axis. So a split tile is an ordinary catalog geometry and reaches the
    funnel, the CUDNN_GEMM_TEST_FULL sweep and the benchmarks' default config set.
    `num_mma_m == 1` is enumerated first, so an unqualified `cta_tile_m` lookup
    still lands on the unsplit tile — several suites rely on that."""
    from cudnn.gemm.frost.graph_analyzer import analyze
    from cudnn.gemm.frost.kernel_registry import candidates
    from cudnn.gemm.frost.tile_config import _M_AXES

    sm100 = [c for c in CATALOG if c.pipeline == "sm100"]
    assert {(c.mma_inst_m, c.num_mma_m) for c in sm100} == set(_M_AXES)
    assert all(c.cta_tile_m == c.mma_inst_m * c.num_mma_m for c in CATALOG)
    # sm103 / sm107 pin the INSTRUCTION M (the block-scale SF 128x4 swizzle needs
    # mma_inst_m % 128 == 0); sm107 still splits, sm103 does not (supports_multi_mma_m=False).
    assert {c.mma_inst_m for c in CATALOG if c.pipeline != "sm100"} == {128}
    assert {c.num_mma_m for c in CATALOG if c.pipeline == "sm107"} == {1, 2}
    assert {c.num_mma_m for c in CATALOG if c.pipeline == "sm103"} == {1}
    # cta_tile_m=128 is the one value two axes produce (128x1 and 64x2).
    assert next(c for c in sm100 if c.cta_tile_m == 128).num_mma_m == 1

    g = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, 256, 128], stride=[256 * 128, 128, 1])
    B = g.tensor(name="B", dim=[1, 128, 256], stride=[128 * 256, 1, 128])
    g.matmul(A=A, B=B, name="mm").set_output(True)
    assert {cfg.num_mma_m for _t, cfg in candidates(analyze(g))} == {1, 2}


@pytest.mark.parametrize(
    "name,cta_group,want",
    [
        ("CONFIG_sm100_128x128x128_128x128x32_cluster2x1", 2, 128),  # mma_inst_m=128 -> full N
        ("CONFIG_sm100_64x128x128_64x128x32_cluster2x1", 2, 64),  # mma_inst_m=64, num_mma_m=1
        ("CONFIG_sm100_128x128x128_64x128x32_cluster2x1", 2, 64),  # mma_inst_m=64, num_mma_m=2
        ("CONFIG_sm100_128x256x128_64x256x32_cluster2x1", 2, 128),
        ("CONFIG_sm100_128x128x128_64x128x32_cluster1x1", 1, 128),  # 1-CTA never halves
    ],
)
def test_drain_width_keys_on_the_mma_block_height(name: str, cta_group: int, want: int) -> None:
    """The 2-CTA 2x2-DP drain splits N across the two 64-lane halves. That is a
    property of the per-CTA MMA block height (cluster-MMA m=128), NOT of the CTA
    tile — they agree only at num_mma_m == 1. Keying it on cta_tile_m made the
    compiler over-report the drain width, which both over-reserved TMEM (losing an
    acc stage) and let the baked STG chunk exceed the real subtile: the 2-CTA
    cta_tile_n=48 split entry of _QUICK_CONFIGS faulted with `misaligned address`."""
    from cudnn.gemm.frost.compiler import _epi_tile_cols

    assert _epi_tile_cols(by_name(name), cta_group) == want


_MMAJOR_COORD_KINDS = ("gen_index_axis1", "gen_index_axis2", "per_row", "per_col", "per_elem")


def _mmajor_coord_graph(kind: str, M: int, N: int, K: int) -> tuple[cudnn.pygraph, str | None]:
    g = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    aux = None
    if kind.startswith("gen_index"):
        axis = int(kind[-1])
        Y = g.add(a=C, b=g.gen_index(input=C, axis=axis, name="gi"), name="ad")
    else:
        dim, stride = {
            "per_row": ([1, M, 1], [M, 1, 1]),
            "per_col": ([1, 1, N], [N, N, 1]),
            "per_elem": ([1, M, N], [M * N, N, 1]),
        }[kind]
        x = g.tensor(name="x", dim=dim, stride=stride, data_type=cudnn.data_type.FLOAT)
        Y = g.mul(a=C, b=x, name="w")
        aux = "x"
    Y.set_stride([M * N, 1, M])
    Y.set_output(True)
    return g, aux


@pytest.mark.parametrize("kind", _MMAJOR_COORD_KINDS)
@pytest.mark.parametrize("shape", ((256, 256, 256), (512, 128, 256)), ids=("256x256x256", "512x128x256"))
@pytest.mark.parametrize("config_name", _STRADDLE_CONFIGS[:2], ids=[_config_id(n) for n in _STRADDLE_CONFIGS[:2]])
def test_m_major_tma_store_serves_coordinate_reading_pointwise(config_name: str, shape: tuple[int, int, int], kind: str) -> None:
    """`row` / `col_j` on the M-major TMA arm are the subtile base, not the
    coordinates of the fragment a lane holds -- that arm loads 16x256b and each
    lane owns a TRANSPOSED patch. Anything asking "which row / column am I"
    (gen_index, per_row / per_col / per_elem aux) goes through
    `_mmajor_elem_coord`; before it did not, and gen_index measured wrong on
    BOTH axes here while being exact through STG."""
    cfg, cta_group = _resolve(config_name)
    M, N, K = shape
    ok, reason = _compatible(cfg, M, N, K, "bf16", "bf16", cta_group=cta_group, out_major="m")
    if not ok:
        pytest.skip(reason)

    g, aux_name = _mmajor_coord_graph(kind, M, N, K)
    plan = _plan(g, config=cfg, cta_group=cta_group)
    if not plan._compiled.use_tma_store:
        pytest.skip(f"{kind} does not take the M-major TMA-store path on {config_name}")

    a, b, c = _mkdata(M, N, K, "bf16", "bf16", out_major="m")
    args = []
    if aux_name is not None:
        n = {"per_row": M, "per_col": N, "per_elem": M * N}[kind]
        shape3 = {"per_row": (1, M, 1), "per_col": (1, 1, N), "per_elem": (1, M, N)}[kind]
        args.append((torch.arange(n, dtype=torch.float32, device="cuda") % 5 + 1).reshape(shape3))
    plan(_vp(plan, a, b, [c], *args))
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    if kind == "gen_index_axis1":
        ref = ref + torch.arange(M, device="cuda", dtype=torch.float32).reshape(1, M, 1)
    elif kind == "gen_index_axis2":
        ref = ref + torch.arange(N, device="cuda", dtype=torch.float32).reshape(1, 1, N)
    else:
        ref = ref * args[0]
    ref = ref.to(torch.bfloat16)

    bad = int((c.to(torch.float32) != ref.to(torch.float32)).sum())
    assert bad == 0, f"\n  {config_name}\n  m-major {M}x{N}x{K}  {kind}\n  wrong at {bad}/{c.numel()}"


def test_tma_arm_never_writes_past_the_output_rows() -> None:
    """The TMA arm runs the epilogue for the whole tile and clips only its OWN
    store; every other store must re-apply the row bound. A missed one is
    INVISIBLE in the output tensor -- it writes past the last row, into whatever
    the allocator put next -- so this test gives each output slack and poisons it.

    The shape matters: M must not be a multiple of the CLUSTER tile height, or
    there are no overhang rows to write. 384 with cluster2x1 x cta_tile_m=128
    (cgrp_tile_m=256) leaves 128."""
    M, N, K = 384, 256, 256
    cfg, cta_group = _resolve("CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma")
    assert M % (cfg.cta_tile_m * cfg.cgrp_size_m) != 0

    g = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    R = g.relu(input=g.matmul(A=A, B=B, name="mm"), name="r")
    R.set_output(True)
    q, sc = g.block_scale_quantize(input=R, block_size=32, axis=-1, name="qr")
    q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    sc.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)

    plan = _plan(g, config=cfg, cta_group=cta_group)
    if not plan._compiled.use_tma_store:
        pytest.skip("this graph does not take the TMA-store arm")

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    slack = 256
    backing, views = [], []
    for o in plan.chain.outputs:
        shape = o.dim if o.dim is not None else (1, M, N)
        buf = torch.empty(shape[0], shape[1] + slack, shape[2], dtype=_TAP_TORCH_DTYPE[o.dtype], device="cuda")
        buf.view(torch.uint8).fill_(0xAB)
        backing.append(buf)
        views.append(buf[:, : shape[1]])
    plan(_vp(plan, a, b, views))
    torch.cuda.synchronize()

    for o, buf in zip(plan.chain.outputs, backing):
        rows = (o.dim if o.dim is not None else (1, M, N))[1]
        tail = buf[:, rows:].view(torch.uint8)
        touched = int((tail != 0xAB).sum())
        assert touched == 0, f"{o.source}: the epilogue wrote {touched} bytes past its {rows} rows"


@pytest.mark.parametrize("name", ["CONFIG_sm100_128x16x128_128x16x32_cluster2x1", "CONFIG_sm100_128x32x128_128x32x32_cluster2x1"])
def test_narrow_n_under_two_cta_mma_still_takes_the_tma_store(name: str) -> None:
    """`cta_group == 2 and cta_tile_n < 64` used to fall back to STG on the
    theory that a subtile would split across the MMA pair. It does not: each
    CTA drains its own half and the subtile span walk already halves to fit.
    Measured bit-identical to STG on 16 (config, cta_group) pairs."""
    from cudnn.gemm.frost.compiler import _use_tma_store_epi
    from cudnn.gemm.frost.graph_analyzer import analyze

    g = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, 256, 128], stride=[256 * 128, 128, 1])
    B = g.tensor(name="B", dim=[1, 128, 256], stride=[128 * 256, 1, 128])
    g.matmul(A=A, B=B, name="mm").set_output(True)
    cfg = by_name(name)
    assert cfg.cta_tile_n < 64
    assert _use_tma_store_epi(analyze(g), cfg, 2) is True


@pytest.mark.parametrize(
    "name,cta_group",
    [
        ("CONFIG_sm100_256x128x128_128x128x32_cluster1x1", 1),
        ("CONFIG_sm100_128x128x128_64x128x32_cluster1x1", 1),
        ("CONFIG_sm100_64x128x128_64x128x32_cluster1x1", 1),
        ("CONFIG_sm100_64x128x128_64x128x32_cluster2x1", 2),
    ],
)
@pytest.mark.parametrize("shape", [(4096, 4096, 512), (255, 256, 240)])
def test_tma_store_serves_every_drain_height(name: str, cta_group: int, shape: tuple) -> None:
    """`mma_inst_m` is 64 or 128 by construction, and the TMA-store epilogue
    stages all three drain layouts: the full thread->row one at 128, the 1-CTA
    packed `lane < 16` one at 64 (half the lanes carry nothing, so the staging
    indexes by `row` and the side effects carry `row_active`), and the 2-CTA
    2x2-DP one at 64 (two column halves per stage, one TMA box each)."""
    from cudnn.gemm.frost.compiler import _epi_vec_bytes, _store_modes, jit_from_cudnn_graph
    from cudnn.gemm.frost.graph_analyzer import analyze

    M, N, K = shape
    cfg = by_name(name)

    def build():
        g = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
        B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
        C = g.matmul(A=A, B=B, name="mm")
        C.set_output(True).set_data_type(_BF16)
        return g, A, B, C

    g, _, _, _ = build()
    assert _store_modes(analyze(g), cfg, cta_group) == ("tma",)

    torch.manual_seed(0)
    a = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16)

    def run(force_stg: bool):
        slack = 4096
        raw = torch.full((2 * M * N + slack,), 0xAB, device="cuda", dtype=torch.uint8)
        c = raw.view(torch.bfloat16)[: M * N].view(1, M, N)
        c.zero_()
        tail = raw[2 * M * N :].clone()
        gg, aa, bb, cc = build()
        jit_from_cudnn_graph(gg, config=cfg, cta_group=cta_group, force_stg_epi=force_stg)({aa: a, bb: b, cc: c})
        torch.cuda.synchronize()
        assert torch.equal(raw[2 * M * N :], tail), "the store ran past the output"
        return c.clone()

    assert torch.equal(run(False).view(torch.uint8), run(True).view(torch.uint8))


@pytest.mark.parametrize(
    "name,cta_group,out_dt,want",
    [
        ("CONFIG_sm100_128x16x128_128x16x32_cluster1x1", 1, "bf16", 32),
        ("CONFIG_sm100_128x128x128_128x128x32_cluster1x1", 1, "bf16", 64),
        ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1", 2, "bf16", 64),
        ("CONFIG_sm100_256x256x128_128x256x32_cluster2x1", 1, "bf16", 128),
        ("CONFIG_sm100_128x128x128_128x128x32_cluster1x1", 1, "fp32", 128),
    ],
)
def test_epi_smem_row_bytes_is_the_real_staging_alignment(name: str, cta_group: int, out_dt: str, want: int) -> None:
    """The TMA-store SMEM stage is written at `smem_subtile_ptr + tidx * epi_n`,
    so the tightest true alignment is one subtile row -- `epi_n * elem_bytes`,
    attained at odd tidx. It was a hardcoded 64, which over-claims at
    cta_tile_n=16 (a 32-byte row) and under-claims at epi_n=64 / 4-byte output.
    `store_swizzled` forwards it as `assumed_align`, so an over-claim is a false
    promise the backend may exploit. It is per OUTPUT, not shared: `epi_n` is a
    column count and only the element width differs."""
    from cudnn.gemm.frost.compiler import _epi_n, _tma_store_sequence
    from cudnn.gemm.frost.dtypes import DTYPE_BYTES
    from cudnn.gemm.frost.graph_analyzer import analyze

    cfg = by_name(name)
    epi_n = _epi_n(cfg, cta_group, out_dt)
    assert epi_n * DTYPE_BYTES[out_dt] == want

    # The width is no longer a shared constant -- it is per output, so it is a
    # literal in the store sequence the renderer unrolls.
    M, N, K = 256, cfg.cta_tile_n, 256
    g = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    g.matmul(A=A, B=B, name="mm").set_output(True).set_data_type(_CUDNN_DTYPE.get(out_dt, cudnn.data_type.FLOAT))
    seq = _tma_store_sequence(analyze(g), cfg, cta_group, frozenset({0}), epi_n)
    assert f"alignment={want}" in seq, seq


def test_no_template_hardcodes_the_staging_alignment() -> None:
    """The staging alignment is the SMEM row width, which is per output
    (`epi_n * its own element width`), so no template carries it: every one hands
    its store sequence to `@@INJECT_TMA_STORE_SEQUENCE@@` and the renderer
    unrolls it per output with that output's own width. A LITERAL in a template
    would go stale the next time `_epi_n` moves."""
    import pathlib

    from cudnn.gemm.frost.compiler import _epi_n, _tma_store_sequence
    from cudnn.gemm.frost.graph_analyzer import analyze
    from cudnn.gemm.frost.tile_config import by_name

    tmpl_dir = pathlib.Path(cudnn.__file__).parent / "gemm" / "frost" / "kernel_templates"
    files = sorted(p for p in tmpl_dir.glob("sm*.py"))
    assert len(files) == 16, [p.name for p in files]
    for path in files:
        src = path.read_text()
        assert "alignment=64" not in src, path.name
        assert "@@INJECT_TMA_STORE_SEQUENCE@@" in src, path.name
        assert "alignment=epi_smem_row_bytes" not in src, path.name

    # The renderer emits the row width the output's OWN dtype implies.
    M = N = K = 256
    g = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    g.matmul(A=A, B=B, name="mm").set_output(True)
    chain = analyze(g)
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1")
    epi_n = _epi_n(cfg, 1, chain.output_dtype)
    seq = _tma_store_sequence(chain, cfg, 1, frozenset({0}), epi_n)
    assert f"alignment={epi_n * 2}" in seq, seq  # bf16


@pytest.mark.parametrize(
    "name,cta_group",
    [
        ("CONFIG_sm100_128x16x128_128x16x32_cluster1x1", 1),
        ("CONFIG_sm100_128x128x128_128x128x32_cluster1x1", 1),
        ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1", 2),
        ("CONFIG_sm100_256x256x128_128x256x32_cluster2x1", 1),
    ],
)
def test_epilogue_chunk_is_the_width_the_arm_hands_the_snippet(name: str, cta_group: int) -> None:
    """The injected snippet is handed the whole staged subtile on the TMA arm
    and one store-vector slice on the STG arm, and the codegen must be told
    which. Getting it wrong is silent: every `vsize`-keyed emitter (aux loads,
    gen_index, the reduction unrolls, the quant block) then walks the wrong
    number of elements."""
    from cudnn.gemm.frost.compiler import _epi_chunk_elems, _epi_n, _epi_vec_bytes
    from cudnn.gemm.frost.dtypes import DTYPE_BYTES
    from cudnn.gemm.frost.graph_analyzer import analyze

    g = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, 256, 128], stride=[256 * 128, 128, 1])
    B = g.tensor(name="B", dim=[1, 128, 256], stride=[128 * 256, 1, 128])
    g.matmul(A=A, B=B, name="mm").set_output(True)
    chain = analyze(g)
    cfg = by_name(name)
    assert _epi_chunk_elems(chain, cfg, cta_group, True) == _epi_n(cfg, cta_group, chain.output_dtype)
    assert _epi_chunk_elems(chain, cfg, cta_group, False) == _epi_vec_bytes(chain, cfg, cta_group) // DTYPE_BYTES[chain.output_dtype]


@pytest.mark.parametrize(
    "name,cta_group",
    [
        ("CONFIG_sm100_128x8x128_128x8x32_cluster1x1", 1),
        ("CONFIG_sm100_128x16x128_128x16x32_cluster1x1", 1),
        ("CONFIG_sm100_128x128x128_128x128x32_cluster1x1", 1),
        ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1", 2),
        # the two `mma_inst_m == 64` drains: 1-CTA packed and 2-CTA 2x2-DP
        ("CONFIG_sm100_64x128x128_64x128x32_cluster1x1", 1),
        ("CONFIG_sm100_64x128x128_64x128x32_cluster2x1", 2),
    ],
)
def test_m_major_tma_store_serves_a_narrow_drain(name: str, cta_group: int) -> None:
    """The M-major store stages an M-contiguous column with one scalar store per
    drain column, so a narrow `epi_n` is just fewer stores. The arm it replaced
    walked `range(epi_n // 16)` stmatrix blocks and emitted ZERO of them below
    16 -- silent, not a fault -- which is what this geometry used to be rejected
    for."""
    from cudnn.gemm.frost.compiler import _store_modes, jit_from_cudnn_graph
    from cudnn.gemm.frost.graph_analyzer import analyze

    m, n, k = 256, 256, 256
    cfg = by_name(name)

    def build():
        g = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        A = g.tensor(name="A", dim=[1, m, k], stride=[m * k, k, 1])
        B = g.tensor(name="B", dim=[1, k, n], stride=[k * n, 1, k])
        C = g.matmul(A=A, B=B, name="mm")
        C.set_output(True).set_data_type(_BF16)
        C.set_stride([m * n, 1, m])
        return g, A, B, C

    g, _, _, _ = build()
    chain = analyze(g)
    assert chain.out_major == "m"
    assert _store_modes(chain, cfg, cta_group) == ("tma",)

    torch.manual_seed(0)
    a = torch.randn(1, m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, n, k, device="cuda", dtype=torch.bfloat16)

    def run(force_stg: bool):
        slack = 4096
        raw = torch.full((2 * n * m + slack,), 0xAB, device="cuda", dtype=torch.uint8)
        c = raw.view(torch.bfloat16)[: n * m].view(1, n, m).transpose(1, 2)
        tail = raw[2 * n * m :].clone()
        gg, aa, bb, cc = build()
        jit_from_cudnn_graph(gg, config=cfg, cta_group=cta_group, force_stg_epi=force_stg)({aa: a, bb: b, cc: c})
        torch.cuda.synchronize()
        assert torch.equal(raw[2 * n * m :], tail), "store ran past the output"
        return c.contiguous()

    assert torch.equal(run(False).view(torch.uint8), run(True).view(torch.uint8))


@pytest.mark.parametrize("N,want_chunk", [(4096, 32), (384, 32), (40, 8), (24, 8), (8, 8)])
def test_m_major_chunk_follows_n(N: int, want_chunk: int) -> None:
    """An M-major output stores one element per chunk column through its own
    strides, so its allowed store width does not bound the chunk -- N does. The
    chunk walks N and the baked `sym_n` divisibility IS the chunk, so it must be
    a power of two dividing N. An M-major output's own alignment measures the M
    extent and says nothing about N, which is why it cannot supply this.

    It used to be pinned at one element, which is always legal but costs ~1.5x on
    the STG arm (4096^3 bf16 `64x128` 1ctamma: 160.5 -> 105.7 us)."""
    from cudnn.gemm.frost.compiler import _epi_chunk_elems
    from cudnn.gemm.frost.graph_analyzer import analyze

    M, K = 256, 256
    cfg = by_name("CONFIG_sm100_64x128x128_64x128x32_cluster1x1")
    g = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(_BF16)
    C.set_stride([M * N, 1, M])
    chain = analyze(g)
    assert chain.out_major == "m"
    # the STG chunk specifically -- `_epi_chunk_elems(..., use_tma=False)` -- which
    # is what an M-major scatter walks, whichever arm this config ends up on.
    assert _epi_chunk_elems(chain, cfg, 1, False) == want_chunk


def test_epi_n_divides_the_drain_width() -> None:
    """`epi_n` is the TMA arm's subtile width, and an N-tile that is not a whole
    number of subtiles would store a box past the tile edge into its neighbour
    (TMA clamps only at the GLOBAL extent). So it must DIVIDE the drain width --
    which is why it is a power of two dividing `cols`, not the power-of-two floor
    OF `cols`. Flooring by value rejected every 8-multiple that is not a
    32-multiple (48 -> 32, and 48 % 32 != 0), i.e. 65 % of the catalog."""
    from cudnn.gemm.frost.compiler import _epi_n, _epi_tile_cols
    from cudnn.gemm.frost.tile_config import CATALOG

    seen_non_pow2_tile = False
    for cfg in CATALOG:
        for cta_group in (1, 2):
            if cta_group == 2 and (cfg.cgrp_size_m % 2 or cfg.cta_tile_n % 16 or cfg.mma_inst_n % 16):
                continue
            cols = _epi_tile_cols(cfg, cta_group)
            seen_non_pow2_tile |= cols & (cols - 1) != 0
            for dt in ("bf16", "fp32", "fp8_e4m3", "fp4_e2m1"):
                n = _epi_n(cfg, cta_group, dt)
                assert n > 0 and n & (n - 1) == 0, (cfg.name, cta_group, dt, n)
                assert cols % n == 0, f"{cfg.name} cta_group={cta_group} {dt}: epi_n {n} does not divide {cols}"
    assert seen_non_pow2_tile, "the catalog no longer carries a non-power-of-2 drain width — the test is vacuous"


def test_the_auto_path_never_picks_a_non_power_of_two_tile() -> None:
    """`select_config` scores N only over {32,64,128,256}, so the widths the
    divisor rule newly admits are reachable through a forced config, not through
    the engine's own choice."""
    from cudnn.gemm.frost.tile_config import select_config

    widths = set()
    for M in (64, 128, 512, 4096, 16384):
        for N in (64, 128, 512, 4096, 11008):
            for num_gemms in (1, 2, 3):
                for block_scale in (False, True):
                    try:
                        widths.add(select_config(M, N, num_gemms=num_gemms, block_scale=block_scale)[0].cta_tile_n)
                    except NotImplementedError:
                        pass
    assert widths, "select_config produced nothing — the sweep is vacuous"
    assert all(w & (w - 1) == 0 for w in widths), sorted(widths)


def test_templates_take_the_chunk_from_the_rendered_constant() -> None:
    """`vsize` is the fusion chunk, not a memory-access width -- it must come
    from `epi_chunk_elems`, which differs per store arm."""
    import pathlib

    tmpl_dir = pathlib.Path(cudnn.__file__).parent / "gemm" / "frost" / "kernel_templates"
    for path in sorted(tmpl_dir.glob("sm*.py")):
        src = path.read_text()
        assert "vsize = epi_chunk_elems" in src, path.name
        assert "vsize = (VEC_BYTES" not in src, path.name


# --- cutlass-dsl version-gated kwargs ----------------------------------------

# `is_exclusive` (the >512-column TMEM grant) and `b_collector_op` (B-operand
# collector reuse) only reached the cutlass-dsl `nvvm.*` wrappers in 4.8. Those
# wrappers take no **kwargs, so NAMING one on an older DSL is a TypeError at JIT
# regardless of the value -- `is_exclusive=False` is just as fatal as True.

_VERSION_GATED_KWARGS = {
    "tcgen05_alloc": "is_exclusive",
    "tcgen05_dealloc": "is_exclusive",
    "tcgen05_mma_block_scale": "b_collector_op",
}


def _template_dir():
    # kernel_templates has no __init__.py (it is exec'd per render), so go
    # through the package that does.
    return pathlib.Path(cudnn.gemm.frost.__file__).parent / "kernel_templates"


def test_templates_route_version_gated_kwargs_through_the_guarded_wrappers():
    """Every template must reach these ops through `_tile_helpers`, which emits
    the kwarg only on the branch that wants it. Calling `nvvm.<op>` directly and
    passing the inert False/None compiles on an internal wheel and fails every
    single JIT on the public one -- which is how the whole gemm suite went red."""
    import ast

    offenders = []
    for path in sorted(_template_dir().glob("sm*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "nvvm"
                and node.func.attr in _VERSION_GATED_KWARGS
            ):
                offenders.append(f"{path.name}:{node.lineno} nvvm.{node.func.attr}(...)")
    assert not offenders, "call the _tile_helpers wrapper instead of nvvm directly:\n  " + "\n  ".join(offenders)


def test_the_guarded_wrappers_keep_the_kwarg_off_the_default_branch():
    """...and the wrappers themselves only name it under the flag. Pinned as
    source structure because the failure mode is a TypeError at trace time on a
    DSL we cannot install here, so no runtime assertion can see it."""
    import ast
    import inspect

    import cudnn.gemm.frost.kernel_templates._tile_helpers as helpers

    for fn_name, kwarg in _VERSION_GATED_KWARGS.items():
        fn = getattr(helpers, fn_name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "nvvm"
        ]
        assert len(calls) == 2, f"{fn_name} should have exactly one guarded and one plain nvvm call, got {len(calls)}"
        named = [c for c in calls if any(k.arg == kwarg for k in c.keywords)]
        assert len(named) == 1, f"{fn_name}: exactly one branch may name {kwarg!r}, got {len(named)}"
        # ...and the other branch must be reachable without the newer DSL.
        assert len(calls) - len(named) == 1, f"{fn_name}: no branch left that omits {kwarg!r}"
