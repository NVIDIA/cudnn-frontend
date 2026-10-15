# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The K2 contraction contract, GPU-free: the batched-matmul graph of each direction analyzes to the FROST GEMM engine's geometry with
the operand majors the direction needs, batch = C, M = N = K = N_res -- the dense prototype and the API's padded / folded-batch graph; the
kernel operand views carry the declared contiguity; the direction and the padded incoming tail are validated.  (The reference views and the
byte / FLOP model are consumed by the API's oracle and ``test_workspace_and_traffic_models``.)"""

import pytest
import torch

import cudnn
import cudnn.gemm.frost  # noqa: F401  -- installs the pygraph recorder hook the analyzer reads
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.triangle_multiplicative_update.kernels import contraction as k2

pytestmark = [pytest.mark.L0]


# (direction, majors, C, N, batch, padded, (batch, M, N, K), (a_batch, b_batch) or None): the dense prototype's graph per direction, and the
# API's padded N = 217 / folded-batch graph (K padded to 224, the batch folded into the channel axis: 128 x 2 = 256 planes) in both directions
_GRAPHS = {
    "outgoing": [("outgoing", ("k", "k"), 8, 64, 1, False, (8, 64, 64, 64), (8, 8))],
    "incoming": [("incoming", ("m", "n"), 8, 64, 1, False, (8, 64, 64, 64), (8, 8))],
    "padded_folded_batch": [
        ("outgoing", ("k", "k"), 128, 217, 2, True, (256, 217, 217, 224), None),
        ("incoming", ("m", "n"), 128, 256, 2, True, (256, 256, 256, 256), None),
    ],
}


@pytest.mark.parametrize("graphs", list(_GRAPHS.values()), ids=list(_GRAPHS))
def test_contraction_graph_analyzes_to_the_direction_majors(graphs):
    for direction, majors, c, n, batch, padded, dims, ab_batch in graphs:
        g = k2.contraction_graph(cudnn, c, n, direction, batch=batch, padded=True) if padded else k2.contraction_graph(cudnn, c, n, direction)
        chain = analyze(g)
        mm = chain.matmul
        assert (mm.a_major, mm.b_major) == majors
        assert (mm.batch, mm.M, mm.N, mm.K) == dims
        if ab_batch is not None:
            assert (mm.a_batch, mm.b_batch) == ab_batch
        assert mm.a_dtype == "bf16" and mm.b_dtype == "bf16"


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


def test_direction_and_the_padded_incoming_tail_are_validated():
    """A direction outside the vocabulary is a ValueError; so is the padded INCOMING form at an N that is not a multiple of 8 (its contiguous
    extent); the padded outgoing strides carry the K pad, the dense prototype's form is unchanged."""
    with pytest.raises(ValueError):
        k2.operand_majors("sideways")
    with pytest.raises(ValueError, match="multiple of 8"):
        k2.contraction_graph(cudnn, 128, 217, "incoming", batch=1, padded=True)
    assert k2.operand_strides(217, "outgoing", padded=True) == ((217 * 224, 224, 1), (217 * 224, 1, 224))
    assert k2.operand_strides(64, "outgoing") == ((64 * 64, 64, 1), (64 * 64, 1, 64))  # the dense prototype's form is unchanged
