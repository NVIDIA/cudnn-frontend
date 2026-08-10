# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fluent ``cudnn.Graph`` wrapper (python/cudnn/wrapper.py)."""

import pytest
import torch

import cudnn


def _matmul_graph(**kwargs):
    """A 64x64 half matmul through the fluent wrapper."""
    with cudnn.Graph(
        handle="auto",
        io_data_type=cudnn.data_type.HALF,
        compute_data_type=cudnn.data_type.FLOAT,
        inputs=["X", "W"],
        outputs=["Y"],
        **kwargs,
    ) as graph:
        X = graph.tensor(name="X", dim=[1, 64, 64], stride=[64 * 64, 64, 1])
        W = graph.tensor(name="W", dim=[1, 64, 64], stride=[64 * 64, 64, 1])
        Y = graph.matmul(name="mm", A=X, B=W)
        Y.set_output(True).set_name("Y")
    return graph


@pytest.mark.L0
def test_workspace_alloc_default_allocates():
    """The default path allocates a workspace the caller never has to think about."""
    graph = _matmul_graph()
    assert torch.is_tensor(graph._Graph__workspace)


@pytest.mark.L0
def test_workspace_alloc_false_is_honored():
    """``workspace_alloc=False`` means the CALLER owns the workspace.

    Regression: the sentinel is written as ``self.__workspace`` (mangled to
    ``_Graph__workspace``) but was read back with ``hasattr(self, "__workspace")``
    — a plain string, which is NOT name-mangled. That probe was therefore always
    False, the sentinel was overwritten with a fresh allocation on every
    ``__exit__``, and the "Need to specify workspace" guard below was unreachable.
    """
    graph = _matmul_graph(workspace_alloc=False)
    assert graph._Graph__workspace is False

    x = torch.randn(1, 64, 64, dtype=torch.half, device="cuda")
    w = torch.randn(1, 64, 64, dtype=torch.half, device="cuda")

    with pytest.raises(RuntimeError, match="Need to specify workspace"):
        graph(x, w)

    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    out = graph(x, w, workspace=workspace)
    torch.testing.assert_close(out.float(), (x @ w).float(), atol=1e-2, rtol=1e-2)
