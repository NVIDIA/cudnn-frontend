# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.mark.L0
def test_torch_stream_context_preserves_legacy_default_stream():
    try:
        from cuda.bindings import driver as cuda
        from cudnn.deepseek_sparse_attention.utils.runtime import torch_stream_context
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    outer_stream = torch.cuda.Stream()
    with torch.cuda.stream(outer_stream):
        with torch_stream_context(cuda.CUstream(0)):
            assert torch.cuda.current_stream().cuda_stream == 0
        assert torch.cuda.current_stream().cuda_stream == outer_stream.cuda_stream
