# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cudnn.ops.nsa names of the PyTorch NSA ops are the existing NSA wrappers."""

import pytest

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "name,wrapper",
    [
        ("selection_attention_forward", "selection_attention_wrapper"),
        ("compression_attention_forward", "compression_attention_wrapper"),
        ("sliding_window_attention_forward", "sliding_window_attention_wrapper"),
        ("top_k_reduction_forward", "topk_reduction_wrapper"),
    ],
)
def test_cudnn_ops_nsa_name_is_the_existing_wrapper(name, wrapper):
    try:
        from cudnn import NSA
    except ImportError as error:
        pytest.skip(f"cudnn optional dependencies not installed: {error}")
    else:
        from cudnn.ops import nsa

        assert getattr(nsa, name) is getattr(NSA, wrapper)
