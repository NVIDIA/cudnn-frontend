# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""C96 architecture admission and device-local compilation, without kernel launches."""

from contextlib import contextmanager

import cudnn
import pytest

from cudnn.conv.frost.engine import FrostConvEngine
from cudnn.frost import buffers
from utils import requires_dense_cutedsl

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("sm", (100, 103))
@pytest.mark.parametrize(
    "channels,k,dtype",
    [(96, 96, cudnn.data_type.BFLOAT16), (96, 192, cudnn.data_type.BFLOAT16), (128, 96, cudnn.data_type.BFLOAT16), (96, 96, cudnn.data_type.FLOAT)],
)
def test_frost_c96_architecture_admission(monkeypatch, sm, channels, k, dtype):
    # Pin the version so the architecture contract is tested with older wheels too.
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.9.0")))
    monkeypatch.setattr(buffers, "current_sm", lambda: sm)
    graph = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x = graph.tensor(dim=(1, channels, 1, 7, 9), stride=(channels * 63, 1, channels * 63, channels * 9, channels), data_type=dtype)
    w = graph.tensor(dim=(k, channels, 3, 3, 3), stride=(channels * 27, 1, channels * 9, channels * 3, channels), data_type=dtype)
    y = graph.conv_fprop(x, w, pre_padding=(2, 1, 1), post_padding=(0, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1))
    y.set_output(True).set_data_type(dtype).set_dim((1, k, 1, 7, 9)).set_stride((k * 63, 1, k * 63, k * 9, k))
    if sm == 103 and channels == 96 and dtype == cudnn.data_type.BFLOAT16:
        with pytest.raises(NotImplementedError, match="BF16 C96 causal convolution is qualified only on SM100"):
            FrostConvEngine().check_support(graph)
    else:
        FrostConvEngine().check_support(graph)


@requires_dense_cutedsl
@pytest.mark.parametrize("depth,k", [(1, 96), (5, 96), (1, 192), (5, 192)])
def test_frost_c96_compiler_uses_build_device_and_target(monkeypatch, depth, k):
    from cudnn.conv.frost.templates import sm100_conv

    target, active = [0, "sm_100a"], [99]
    compiled, queried = [], []
    monkeypatch.setattr(sm100_conv, "build_target", lambda: tuple(target))

    @contextmanager
    def device_context(device):
        previous = active[0]
        active[0] = device
        try:
            yield
        finally:
            active[0] = previous

    class HardwareInfo:
        def __init__(self):
            assert active[0] == target[0]

        def get_max_active_clusters(self, cluster_size):
            queried.append(active[0])
            return 10 + active[0]

    def compile_kernel(*args, options):
        assert active[0] == target[0]
        assert args[4] == 10 + active[0]
        assert options == f"--enable-tvm-ffi --gpu-arch {target[1]}"
        assert len(args) == (8 if depth == 1 and k == 96 else 7)
        result = object()
        compiled.append(result)
        return result

    monkeypatch.setattr(sm100_conv, "device_context", device_context)
    monkeypatch.setattr(sm100_conv.utils, "HardwareInfo", HardwareInfo)
    monkeypatch.setattr(sm100_conv.cute, "compile", compile_kernel)
    sm100_conv._compile_cached.cache_clear()
    try:
        results = []
        for device in (0, 1, 0, 1):
            target[0] = device
            result = sm100_conv.compile(ncdhw=(1, 96, depth, 7, 9), ktrs=(k, 3, 3, 3), lower_padding_dhw=(2, 1, 1), upper_padding_dhw=(0, 1, 1))
            if depth == 1 and k == 96:
                assert isinstance(result, sm100_conv._PaddedFrameConv)
                assert result.workspace_bytes == 384
                result = result.launch
            results.append(result)
            assert active[0] == 99
        assert results[0] is results[2] is compiled[0]
        assert results[1] is results[3] is compiled[1]
        assert results[0] is not results[1]
        assert queried == [0, 1]
    finally:
        sm100_conv._compile_cached.cache_clear()
