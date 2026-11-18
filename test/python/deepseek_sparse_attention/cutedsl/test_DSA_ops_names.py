# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cudnn.ops.dsa names of the PyTorch DSA ops are the existing wrappers."""

import importlib

import pytest

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "name,module,wrapper",
    [
        ("compressor_forward", "cudnn.csa", "csa_compressor_forward_wrapper"),
        ("compressor_backward", "cudnn.csa", "csa_compressor_backward_wrapper"),
        ("indexer_forward", "cudnn.deepseek_sparse_attention", "indexer_forward_wrapper"),
        ("indexer_backward", "cudnn.deepseek_sparse_attention", "indexer_backward_wrapper"),
        ("dense_indexer_backward", "cudnn.deepseek_sparse_attention", "dense_indexer_backward_wrapper"),
        ("top_k_forward", "cudnn.deepseek_sparse_attention", "indexer_top_k_wrapper"),
        ("sparse_attention_forward", "cudnn.deepseek_sparse_attention", "sparse_attention_forward_wrapper"),
        ("sparse_attention_backward", "cudnn.deepseek_sparse_attention", "sparse_attention_backward_wrapper"),
    ],
)
def test_cudnn_ops_dsa_name_is_the_existing_wrapper(name, module, wrapper):
    try:
        existing = getattr(importlib.import_module(module), wrapper)
    except ImportError as error:
        pytest.skip(f"cudnn optional dependencies not installed: {error}")
    else:
        from cudnn.ops import dsa

        assert getattr(dsa, name) is existing
