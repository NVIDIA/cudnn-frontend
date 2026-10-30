# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the C96 gate with the installed DSL, including public cu12 wheels."""

import builtins

import cudnn
import pytest
import torch

from cudnn.conv.frost._cutedsl import DENSE_CUTEDSL_MIN_VERSION, requirement_error
from cudnn.conv.frost.engine import FrostConvEngine

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("k", (96, 192))
def test_frost_c96_real_dependency_gate_before_template_import(monkeypatch, k):
    error = requirement_error("frost_conv dense convolution", DENSE_CUTEDSL_MIN_VERSION)
    if error is None:
        pytest.skip("installed DSL satisfies the dense version gate; covered by C96 execution tests")
    image = torch.empty((1, 96, 1, 7, 9), dtype=torch.bfloat16, device="cuda", memory_format=torch.channels_last_3d)
    weight = torch.empty((k, 96, 3, 3, 3), dtype=image.dtype, device=image.device, memory_format=torch.channels_last_3d)
    output = torch.empty((1, k, 1, 7, 9), dtype=image.dtype, device=image.device, memory_format=torch.channels_last_3d)
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x, w = graph.tensor_like(image), graph.tensor_like(weight)
    y = graph.conv_fprop(x, w, pre_padding=(2, 1, 1), post_padding=(0, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1))
    y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim(output.shape).set_stride(output.stride())
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.endswith("templates.sm100_conv"), "the version gate must precede dense template import"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    engine = FrostConvEngine()
    for call in (lambda: engine.check_support(graph), lambda: engine.build_plan(graph, None)):
        with pytest.raises(NotImplementedError) as caught:
            call()
        assert str(caught.value) == error
