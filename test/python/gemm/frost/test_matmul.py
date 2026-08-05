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
import sys

import pytest
import torch

from gemm_test_utils import (
    requires_sm100,
    Plan as _plan,
    vp as _vp,
    resolve as _resolve,
)

# Module-wide GPU gate — every test here is end-to-end and needs a B200.
pytestmark = [pytest.mark.L0, requires_sm100]


import cudnn
import cudnn.gemm.frost  # noqa: F401  — installs the cudnn.pygraph recorder hook
from cudnn.gemm.frost.compiler import _current_arch
from cudnn.gemm.frost.tile_config import CATALOG

# INT8 matmul runs only on SM 100 or SM 110 (disjoint range).
_INT8_SM_RANGES = ((100, 101), (110, 111))


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
    "CONFIG_sm100_128x64x128_128x64x32_cluster1x4_1ctamma_static",  # static scheduler (cta1)
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma_static",  # static scheduler (cta2)
    # N not a multiple of 32 (pow2 epilogue subtile spans + tile-clamped vsize).
    "CONFIG_sm100_128x8x128_128x8x32_cluster1x1_1ctamma",  # minimum N
    "CONFIG_sm100_128x40x128_128x40x32_cluster1x1_1ctamma",  # 32+8 tail span
    "CONFIG_sm100_64x24x128_64x24x32_cluster1x1_1ctamma",  # cta_m=64, 16+8 spans
    "CONFIG_sm100_128x144x128_128x144x32_cluster2x1_2ctamma",  # cta2 (N%16), 16-col tail
    "CONFIG_sm100_128x48x128_128x48x32_cluster2x1_2ctamma_static",  # static cta2, 16-col tail
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
    "CONFIG_sm100_128x64x128_128x64x32_cluster1x4_1ctamma_static",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma_static",
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
    if a_major == "m" and (cta_smem_m < mn_group_elems or cta_smem_m % mn_group_elems != 0):
        return False, (f"A M-major per-CTA SMEM M={cta_smem_m} is not compatible with " f"the {mn_group_elems}-element swizzle group")
    if b_major == "n" and (cta_smem_n < mn_group_elems or cta_smem_n % mn_group_elems != 0):
        return False, (f"B N-major per-CTA SMEM N={cta_smem_n} is not compatible with " f"the {mn_group_elems}-element swizzle group")
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


def _plan_or_skip(cache, key, build_graph, cfg, cta_group, scheduler):
    """JIT the anchor graph; the engine's clean "unsupported" rejections —
    NotImplementedError from the compiler gates, or the registry's "no kernel
    template" — SKIP, any other compile error FAILS."""
    try:
        compiled = _plan(build_graph(), config=cfg, cta_group=cta_group, scheduler=scheduler)
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
    scheduler: str = "clc",
    out_major: str = "n",
):
    """Return the cached compiled kernel, building it on first miss."""
    key = (cfg.name, in_dt, out_dt, a_major, b_major, cta_group, scheduler, out_major)
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
        scheduler,
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
    scheduler: str = "clc",
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
        scheduler,
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
        scheduler,
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
    scheduler: str = "clc",
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
        scheduler,
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
        scheduler,
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
    cfg, cta_group, scheduler = _resolve(config_name)
    ok, reason = _compatible(cfg, *shape, in_dt, out_dt, cta_group=cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_compiled(_compile_cache, cfg, in_dt, out_dt, cta_group=cta_group, scheduler=scheduler)

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


def test_dense_block_scale_quant_epilogue() -> None:
    """Plain dense GEMM can use terminal block_scale_quantize epilogue."""
    config_name = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"
    cfg, cta_group, scheduler = _resolve(config_name)
    M = N = K = 128
    block_size = 32
    g = _build_block_quant_graph(M, N, K, block_size)
    compiled = _plan(
        g,
        config=cfg,
        cta_group=cta_group,
        scheduler=scheduler,
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
    cfg, cta_group, scheduler = _resolve(config_name)
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

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
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


def test_dense_row_col_dual_quant_f8_reorder() -> None:
    """The cutedsl dual-output pattern: one producer -> row quant + col quant,
    both with F8_128x4 scale reordering."""
    cfg, cta_group, scheduler = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
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

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
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
    cfg, cta_group, scheduler = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    M, N, K = 256, 128, 128
    g = _fp4_dual_graph(M, N, K, fp4_axis, 32, cudnn.data_type.FP8_E8M0)
    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
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
    cfg, cta_group, scheduler = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
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

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)

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
    cfg, cta_group, scheduler = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
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

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
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
    cfg, cta_group, scheduler = _resolve(config_name)
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

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
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
    cfg, cta_group, scheduler = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
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

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
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
    cfg, cta_group, scheduler = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
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

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
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

    cfg, cta_group, scheduler = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    with pytest.raises(ValueError, match="divisible by block_size"):
        _plan(_col_graph(160 + 8, 128, 128, 32), config=cfg, cta_group=cta_group, scheduler=scheduler)
    with pytest.raises(NotImplementedError, match="block_size 32"):
        _plan(_col_graph(256, 128, 128, 8), config=cfg, cta_group=cta_group, scheduler=scheduler)


def test_dense_block_scale_quant_with_dense_tap() -> None:
    """Quant data rides slot 0 while the pre-quant producer is also tapped."""
    cfg, cta_group, scheduler = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
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

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
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


def test_dense_dual_block_scale_quant() -> None:
    """Two quant nodes fan out from one producer (e4m3 + e5m2 data)."""
    cfg, cta_group, scheduler = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
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

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
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
    cfg, cta_group, scheduler = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
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

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
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
    cfg, cta_group, scheduler = _resolve(config_name)
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
        scheduler=scheduler,
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
    cfg, cta_group, scheduler = _resolve(config_name)
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
        scheduler=scheduler,
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
    cfg, cta_group, scheduler = _resolve(config_name)
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
        scheduler=scheduler,
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
    cfg, cta_group, scheduler = _resolve(config_name)
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
        scheduler=scheduler,
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
    cfg, cta_group, scheduler = _resolve(config_name)
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
        scheduler=scheduler,
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
    cfg, cta_group, scheduler = _resolve(config_name)
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
        scheduler=scheduler,
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
    cfg, cta_group, scheduler = _resolve(config_name)
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
        scheduler=scheduler,
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
    cfg, cta_group, scheduler = _resolve(config_name)
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

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
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
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma_static",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma_static",
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


@pytest.mark.parametrize("config_name", _INT8_CONFIGS, ids=[_config_id(n) for n in _INT8_CONFIGS])
@pytest.mark.parametrize("out_dt", list(_INT8_OUT_DTYPES))
def test_int8_matmul(config_name: str, out_dt: str) -> None:
    """INT8×INT8→INT32, output ∈ {fp32,bf16,fp16,int32,fp8}; bit-exact vs a
    rounded integer reference (values small enough that the rounding is exact)."""
    sm = _current_arch()
    if sm is not None and not any(lo <= sm < hi for lo, hi in _INT8_SM_RANGES):
        pytest.skip(f"int8 matmul unsupported on sm_{sm} (SM 100/110 only)")
    cfg, cta_group, scheduler = _resolve(config_name)
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

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
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
    cfg, cta_group, scheduler = _resolve(config_name)
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
        scheduler=scheduler,
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
    cfg, cta_group, scheduler = _resolve("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    return _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler), M, N, K


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
