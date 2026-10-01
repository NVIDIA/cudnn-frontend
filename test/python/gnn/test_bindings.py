# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

import cudnn


@pytest.mark.L0
@pytest.mark.skipif(sys.platform.startswith("win"), reason="cuDNN GNN APIs are not supported on Windows")
def test_gnn_mha_bindings_do_not_depend_on_build_headers():
    assert hasattr(cudnn, "gnn_activation_op")
    assert hasattr(cudnn, "gnn_mha_gat_forward")
    assert hasattr(cudnn, "gnn_mha_gat_backward")
    assert hasattr(cudnn, "gnn_mha_gat_v2_forward")
    assert hasattr(cudnn, "gnn_mha_gat_v2_backward")
