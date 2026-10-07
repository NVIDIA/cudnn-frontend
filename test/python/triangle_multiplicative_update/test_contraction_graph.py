# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The K2 contraction contract, GPU-free: the batched-matmul graph of each direction analyzes to the FROST GEMM engine's geometry with
the operand majors the direction needs, batch = C, M = N = K = N_res; the reference views compute the contraction; the model counts."""

import pytest
import torch

import cudnn
import cudnn.gemm.frost  # noqa: F401  -- installs the pygraph recorder hook the analyzer reads
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.triangle_multiplicative_update.kernels import contraction as k2

pytestmark = [pytest.mark.L0]


@pytest.mark.parametrize("direction,majors", [("outgoing", ("k", "k")), ("incoming", ("m", "n"))])
def test_contraction_graph_analyzes_to_the_direction_majors(direction, majors):
    g = k2.contraction_graph(cudnn, 8, 64, direction)
    chain = analyze(g)
    mm = chain.matmul
    assert (mm.a_major, mm.b_major) == majors
    assert (mm.batch, mm.M, mm.N, mm.K) == (8, 64, 64, 64)
    assert (mm.a_batch, mm.b_batch) == (8, 8)
    assert mm.a_dtype == "bf16" and mm.b_dtype == "bf16"


def test_reference_views_compute_the_contraction():
    torch.manual_seed(0)
    a = torch.randn(3, 16, 16, dtype=torch.float64)
    b = torch.randn(3, 16, 16, dtype=torch.float64)
    out = torch.bmm(*k2.reference_views(a, b, "outgoing"))
    inc = torch.bmm(*k2.reference_views(a, b, "incoming"))
    assert torch.allclose(out, torch.einsum("cik,cjk->cij", a, b))
    assert torch.allclose(inc, torch.einsum("cki,ckj->cij", a, b))


def test_kernel_operand_views_carry_the_declared_contiguity():
    a = torch.empty(2, 32, 32, dtype=torch.bfloat16)
    b = torch.empty(2, 32, 32, dtype=torch.bfloat16)
    for direction in k2.DIRECTIONS:
        ka, kb = k2.kernel_operand_views(a, b, direction)
        a_stride, b_stride = k2.operand_strides(32, direction)
        a_major, b_major = k2.operand_majors(direction)
        # A view [C, M, K]: K-major -> stride[-1] == 1, M-major -> stride[-2] == 1 (the analyzer's rule); the graph's A stride is the view's
        assert tuple(ka.stride()) == a_stride
        assert (ka.stride(-1) == 1) == (a_major == "k")
        # B view [C, N, K] in kernel order: K-major -> stride[-1] == 1; N-major -> stride[-2] == 1 == the graph's B [C, K, N] stride[-1]
        assert (kb.stride(-1) == 1) == (b_major == "k")
        assert (kb.stride(-2) == 1) == (b_major == "n")
        assert b_stride[-2] == 1 if b_major == "k" else b_stride[-1] == 1


def test_direction_is_validated():
    with pytest.raises(ValueError):
        k2.operand_majors("sideways")


def test_model_counts():
    assert k2.contraction_bytes(128, 512) == 6 * 512 * 512 * 128
    assert k2.contraction_flop(128, 512) == 2 * 128 * 512**3
