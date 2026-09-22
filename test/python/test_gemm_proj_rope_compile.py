# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile projection kernels with metadata-only arguments; no GPU launch."""

import importlib.util
from pathlib import Path
import sys

import pytest

cutlass = pytest.importorskip("cutlass", minversion="4.6.2")
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

pytestmark = pytest.mark.L0

_KERNEL_DIR = Path(__file__).resolve().parents[2] / "python/cudnn/gemm/cutedsl/dense/proj_rope_mxfp8"


@pytest.mark.parametrize(
    "variant,weight_order",
    [("", (0, 1)), ("", (1, 0)), ("_bf16in", (0, 1)), ("_bf16in", (1, 0)), ("_mxfp8in", (1, 0))],
)
def test_projection_rope_kernel_compiles(variant, weight_order, monkeypatch):
    # Importing a decorated kernel never evaluates its device body. Compile the
    # real host entry to cover allocator lookups and scheduler serialization.
    name = f"gemm_proj_rope_mxfp8{variant}"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", _KERNEL_DIR / f"{name}.py")
    kernel = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, kernel)
    spec.loader.exec_module(kernel)

    tokens, heads, k_dim = 2048, 2, 1536
    n_dim = heads * kernel.HEAD_DIM

    def tensor(dtype, shape, stride_order=None):
        if stride_order is None:
            stride_order = tuple(reversed(range(len(shape))))
        return make_fake_compact_tensor(dtype, shape, stride_order=stride_order, assumed_align=16)

    a = tensor(kernel.io_dtype, (tokens, k_dim))
    b = tensor(kernel.io_dtype, (n_dim, k_dim), weight_order)
    if variant == "_mxfp8in":
        operands = [
            a,
            tensor(cutlass.Float8E8M0FNU, (tokens, k_dim // kernel.BLOCK)),
            b,
            tensor(cutlass.Float8E8M0FNU, (n_dim, k_dim // kernel.BLOCK)),
        ]
    else:
        operands = [a, b]
    operands.extend(
        [
            tensor(cutlass.BFloat16, (tokens, kernel.QK_ROPE)),
            tensor(cutlass.BFloat16, (tokens, kernel.QK_ROPE)),
            tensor(cutlass.Float8E4M3FN, (tokens, heads, kernel.HEAD_DIM)),
            tensor(cutlass.Uint8, (tokens, heads, kernel.HEAD_DIM // kernel.BLOCK)),
            tensor(cutlass.Float8E4M3FN, (tokens, heads, kernel.HEAD_DIM)),
            tensor(cutlass.Uint8, (tokens // kernel.BLOCK, heads, kernel.HEAD_DIM)),
        ]
    )
    config = [tokens // kernel.TILE_M, heads, 16, 4]
    if variant == "_mxfp8in":
        config.extend([True, k_dim // 128])
    compiled = cute.compile(
        kernel.gemm_proj_rope_mxfp8_host,
        *operands,
        *config,
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--gpu-arch=sm_100a",
    )
    assert compiled is not None
