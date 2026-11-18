# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic artifact reuse, specialization boundaries and shape-specific replay."""

from contextlib import nullcontext
from dataclasses import replace

import pytest

from utils import make_input, requires_dense_cutedsl, requires_sm100, select_and_build_frost_plan
from test_conv_shapes import ConvCase, _SHAPE_CASES, test_frost_conv_supported_shapes_and_geometry as run_shape

pytestmark = [requires_dense_cutedsl]


@pytest.fixture
def dynamic():
    from cudnn.conv.frost.templates import sm100_conv as module

    module._compile_cached.cache_clear()
    yield module
    module._compile_cached.cache_clear()


@pytest.mark.L0
def test_geometry_cache_boundaries(monkeypatch, dynamic):
    import cutlass
    from cudnn.conv.frost.tile_config import ConvTileConfig

    calls = []
    monkeypatch.setattr(dynamic, "build_target", lambda: (0, "sm_100a"))
    monkeypatch.setattr(dynamic, "device_context", lambda device: nullcontext())
    monkeypatch.setattr(dynamic.utils.HardwareInfo, "__init__", lambda self, device_id=0: None)
    monkeypatch.setattr(dynamic.utils.HardwareInfo, "get_max_active_clusters", lambda self, size: 64)
    monkeypatch.setattr(cutlass.memory, "get_smem_capacity_in_bytes", lambda: 232448)

    def compile_spy(*args, **kwargs):
        calls.append(args)
        return object()

    monkeypatch.setattr(dynamic.cute, "compile", compile_spy)
    tile = ConvTileConfig(128, 128, 128, 1)
    a = dict(ncdhw=(1, 64, 3, 5, 7), ktrs=(40, 1, 2, 3), tile_config=tile)
    b = dict(
        ncdhw=(2, 512, 5, 7, 9),
        ktrs=(128, 2, 3, 1),
        tile_config=tile,
        upper_padding_dhw=(1, 0, 2),
        lower_padding_dhw=(0, 1, 0),
        stride_dhw=(2, 1, 2),
        dilation_dhw=(1, 2, 1),
    )
    first = dynamic.compile(**a)
    assert dynamic.compile(**b) is first
    assert dynamic.compile(**a) is first
    assert len(calls) == 1
    _, fa, fb, fc, *_ = calls[0]
    assert fa.shape[0] is fc.shape[0]
    assert fa.shape[4] is fb.shape[4]
    assert fb.shape[0] is fc.shape[4]
    assert all(not isinstance(extent, int) and extent.divisibility == 1 for extent in fb.shape[1:4])
    assert dynamic.compile(**(a | dict(ktrs=(40, 1, 1, 1)))) is first
    assert fa.shape[4].divisibility == 64
    assert fc.shape[4].divisibility == 8
    assert fa.stride[0].width == fb.stride[0].width == fc.stride[0].width == 64
    for change in (
        dict(tile_config=replace(tile, cta_tile_n=64)),
        dict(ab_dtype=cutlass.Float16),
        dict(c_dtype=cutlass.Float32),
        dict(epilogue="relu"),
    ):
        assert dynamic.compile(**(a | change)) is not first
    assert len(calls) == 5
    assert dynamic.compile(**a) is first
    monkeypatch.setattr(dynamic, "build_target", lambda: (1, "sm_100a"))
    assert dynamic.compile(**a) is not first
    monkeypatch.setattr(dynamic, "build_target", lambda: (0, "sm_103a"))
    assert dynamic.compile(**a) is not first
    monkeypatch.setattr(dynamic, "build_target", lambda: (0, "sm_100a"))
    assert dynamic.compile(**a) is first


@pytest.mark.L1
@requires_sm100
@pytest.mark.parametrize("case", _SHAPE_CASES, ids=lambda case: case.name)
def test_dynamic_shapes(monkeypatch, case):
    run_shape(monkeypatch, case)


@pytest.mark.L1
@requires_sm100
@pytest.mark.parametrize("cta_group", [1, 2])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_separate_graphs_reuse_and_replay(monkeypatch, dynamic, cta_group, dtype_name):
    import cudnn
    import torch
    from cudnn.conv.frost.tile_config import ConvTileConfig

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    monkeypatch.setattr(dynamic, "_get_config", lambda *args, **kwargs: ConvTileConfig(128, 128, 128, cta_group))
    dtype = getattr(torch, dtype_name)
    fe_dtype = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    cases = [
        ConvCase("a", (1, 64, 3, 5, 7), (40, 64, 1, 2, 3)),
        ConvCase("b", (2, 512, 5, 7, 9), (128, 512, 2, 3, 1), pre_padding=(0, 1, 0), post_padding=(1, 0, 2), stride=(2, 1, 2), dilation=(1, 2, 1)),
        ConvCase("unit", (1, 128, 3, 4, 5), (64, 128, 1, 1, 1)),
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)
    with torch.cuda.stream(stream):
        for index, case in enumerate((cases[0], cases[1], cases[2], cases[0])):
            x = make_input(case.image_shape, dtype, 11 + index)
            w = make_input(case.weight_shape, dtype, 31 + index)
            pad = tuple(v for pair in zip(reversed(case.pre_padding), reversed(case.post_padding)) for v in pair)
            ref = torch.nn.functional.conv3d(torch.nn.functional.pad(x.float(), pad), w.float(), stride=case.stride, dilation=case.dilation).to(dtype)
            y = torch.empty(ref.shape, dtype=dtype, device="cuda", memory_format=torch.channels_last_3d)
            graph = cudnn.pygraph(handle=handle, io_data_type=fe_dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
            gx, gw = graph.tensor_like(x), graph.tensor_like(w)
            gy = graph.conv_fprop(gx, gw, pre_padding=case.pre_padding, post_padding=case.post_padding, stride=case.stride, dilation=case.dilation)
            gy.set_output(True).set_data_type(fe_dtype).set_dim(y.shape).set_stride(y.stride())
            select_and_build_frost_plan(graph, "frost_conv")
            if index == 0:
                monkeypatch.setattr(dynamic.cute, "compile", lambda *a, **k: pytest.fail("runtime geometry recompiled"))
            workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
            pack = {gx: x, gw: w, gy: y}
            graph.execute(pack, workspace)
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture, stream=stream):
                graph.execute(pack, workspace)
            x.mul_(0.5)
            capture.replay()
            stream.synchronize()
            eps = torch.finfo(dtype).eps
            torch.testing.assert_close(y.float(), (ref * 0.5).float(), atol=2 * eps, rtol=2 * eps)
    torch.cuda.current_stream().wait_stream(stream)

    cudnn.destroy_handle(handle)


@pytest.mark.L0
@pytest.mark.parametrize("bad_role", [0, 1, 2])
@pytest.mark.parametrize("bad_field", ["shape", "stride", "dtype", "alignment"])
def test_dynamic_plan_rejects_changed_binding_before_launch(monkeypatch, bad_role, bad_field):
    from types import SimpleNamespace
    import cudnn
    from cudnn.conv.frost import engine
    from cudnn.engines.base import ExecutionContext

    shapes = [(1, 64, 3, 5, 7), (40, 64, 1, 2, 3), (1, 40, 3, 4, 5)]
    declarations = [SimpleNamespace(dim=s, data_type=cudnn.data_type.BFLOAT16, uid=i) for i, s in enumerate(shapes)]
    calls = []
    monkeypatch.setattr(engine, "current_device", lambda: 0)
    plan = engine._Sm100FrostConvPlan(lambda *a: calls.append(a), declarations, 0, ((0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 1, 1)))
    operands = []
    for index, shape in enumerate(shapes):
        stride = engine._channels_last_3d_stride(shape)
        operand = SimpleNamespace(shape=shape, dtype="bfloat16", stride=lambda stride=stride: stride, data_ptr=lambda: 16, permute=lambda *a: object())
        if index == bad_role:
            if bad_field == "shape":
                operand.shape = (shape[0], shape[1] + 8, *shape[2:])
            elif bad_field == "stride":
                operand.stride = lambda: (1, 1, 1, 1, 1)
            elif bad_field == "dtype":
                operand.dtype = "float16"
            else:
                operand.data_ptr = lambda: 17
        operands.append(operand)
    pack = SimpleNamespace(index_of=lambda uid: uid, operands=lambda indices: operands)
    with pytest.raises(ValueError):
        plan.execute(None, pack, ExecutionContext(stream=0))
    assert not calls
