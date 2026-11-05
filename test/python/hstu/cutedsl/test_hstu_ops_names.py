# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cudnn.ops names of the PyTorch HSTU ops are the existing top-level functions."""

import pytest

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("name", ["hstu_attention_forward", "hstu_attention_backward", "hstu_lmsd_forward", "hstu_lmsd_backward"])
def test_cudnn_ops_name_is_the_existing_function(name):
    import cudnn
    import cudnn.ops

    try:
        existing = getattr(cudnn, name)
    except ImportError as error:
        pytest.skip(f"cudnn optional dependencies not installed: {error}")
    assert getattr(cudnn.ops, name) is existing
