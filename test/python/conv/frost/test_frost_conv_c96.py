# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""C96 causal Conv3D numerics, channel remainder, and history boundaries."""

import cudnn
import pytest
import torch

from utils import make_input, requires_dense_cutedsl, requires_sm100, select_and_build_frost_plan

pytestmark = [pytest.mark.L1, requires_sm100, requires_dense_cutedsl]


@pytest.fixture(autouse=True)
def _ieee_reference(monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    with torch.backends.cudnn.flags(allow_tf32=False):
        yield


def _reference(image, weight, history_frames=0):
    padded = torch.nn.functional.pad(image.float(), (1, 1, 1, 1, 2 - history_frames, 0))
    return torch.nn.functional.conv3d(padded, weight.float()).to(torch.bfloat16)


def _plan(image, weight, history_frames=0):
    shape = (image.shape[0], weight.shape[0], image.shape[2] - history_frames, *image.shape[3:])
    output = torch.empty(shape, dtype=torch.bfloat16, device=image.device, memory_format=torch.channels_last_3d)
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x = graph.tensor_like(image)
    w = graph.tensor_like(weight)
    y = graph.conv_fprop(x, w, pre_padding=(2 - history_frames, 1, 1), post_padding=(0, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1))
    y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim(output.shape).set_stride(output.stride())
    select_and_build_frost_plan(graph, "frost_conv")
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device=image.device)
    return graph, (x, w, y), output, workspace


def _conv(image, weight, history_frames=0):
    graph, (x, w, y), output, workspace = _plan(image, weight, history_frames)
    graph.execute({x: image, w: weight, y: output}, workspace)
    torch.cuda.synchronize()
    return output


def _close(actual, expected):
    # The existing BF16 shape suite's tolerance, fixed before changing the kernel.
    epsilon = float(torch.finfo(torch.bfloat16).eps)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2 * epsilon, rtol=2 * epsilon)


def _compact(tensor):
    result = torch.empty(tensor.shape, device=tensor.device, dtype=tensor.dtype, memory_format=torch.channels_last_3d)
    return result.copy_(tensor)


@pytest.mark.parametrize("shape", [(1, 96, 1, 5, 7), (1, 96, 5, 9, 11), (2, 96, 3, 3, 5)])
@pytest.mark.parametrize("k", [96, 192])
@pytest.mark.parametrize("pattern", ["random", "tail-only", "cancellation", "zero"])
def test_frost_conv_c96_channel_remainder(shape, k, pattern):
    image = make_input(shape, torch.bfloat16, seed=1234)
    weight = make_input((k, 96, 3, 3, 3), torch.bfloat16, seed=5678)
    if pattern == "tail-only":
        image[:, :64].zero_()
    elif pattern == "cancellation":
        image.fill_(1)
        image[:, 1::2].fill_(-1)
        weight.fill_(0.25)
        # A final-channel contribution survives the cancelling channel pairs.
        weight[:, 95].fill_(0.5)
    elif pattern == "zero":
        image.zero_()
    _close(_conv(image, weight), _reference(image, weight))


@pytest.mark.parametrize("k", [96, 192])
@pytest.mark.parametrize("history_frames", [0, 1, 2])
def test_frost_conv_c96_nonintegral_values_and_bias(k, history_frames):
    torch.manual_seed(712)
    image = _compact(torch.randn((1, 96, 3 + history_frames, 7, 9), device="cuda", dtype=torch.bfloat16))
    weight = _compact(torch.randn((k, 96, 3, 3, 3), device="cuda", dtype=torch.bfloat16) * 0.1)
    bias = torch.randn((1, k, 1, 1, 1), device="cuda", dtype=torch.bfloat16)
    # Preserve the caller's BF16 convolution rounding before its separate bias.
    result = _conv(image, weight, history_frames).add_(bias)
    expected = _reference(image, weight, history_frames).add_(bias)
    _close(result, expected)


def test_frost_conv_c96_causal_impulse_and_future_perturbation():
    image = make_input((1, 96, 5, 5, 7), torch.bfloat16, seed=112)
    weight = make_input((96, 96, 3, 3, 3), torch.bfloat16, seed=113)
    before = _conv(image, weight)
    changed = _compact(image)
    changed[:, :, 3:] += 1
    after = _conv(changed, weight)
    torch.testing.assert_close(before[:, :, :3], after[:, :, :3], atol=0, rtol=0)
    image.zero_()
    image[0, 95, 2, 0, 0] = 1
    weight.zero_()
    weight[0, 95] = 1
    impulse = _conv(image, weight)
    torch.testing.assert_close(impulse, _reference(image, weight), atol=0, rtol=0)
    assert torch.count_nonzero(impulse[:, :, :2]) == 0
    assert torch.count_nonzero(impulse[:, :, 2:]) > 0


@pytest.mark.parametrize("k", [96, 192])
def test_frost_conv_c96_history_transitions_and_reset(k):
    image = make_input((1, 96, 9, 5, 7), torch.bfloat16, seed=210)
    weight = make_input((k, 96, 3, 3, 3), torch.bfloat16, seed=211)
    whole = _conv(image, weight)
    cache = None
    outputs = []
    start = 0
    for length in (1, 1, 4, 3):
        current = _compact(image[:, :, start : start + length])
        h = 0 if cache is None else cache.shape[2]
        assembled = current if cache is None else _compact(torch.cat((cache, current), dim=2))
        result = _conv(assembled, weight, h)
        _close(result, _reference(assembled, weight, h))
        outputs.append(result)
        cache = _compact(assembled[:, :, -2:])
        start += length
        torch.testing.assert_close(cache, image[:, :, max(0, start - 2) : start], atol=0, rtol=0)
        # The saved history must survive caller-buffer reuse.
        assembled.zero_()
        current.zero_()
        torch.testing.assert_close(cache, image[:, :, max(0, start - 2) : start], atol=0, rtol=0)
    _close(torch.cat(outputs, dim=2), whole)
    # Reset between unrelated sequences: no stale history may enter this call.
    fresh = make_input((1, 96, 1, 5, 7), torch.bfloat16, seed=212)
    _close(_conv(fresh, weight), _reference(fresh, weight))


def test_frost_conv_c96_oracle_detects_wrong_padding_and_missing_tail():
    image = make_input((1, 96, 5, 5, 7), torch.bfloat16, seed=310)
    weight = make_input((96, 96, 3, 3, 3), torch.bfloat16, seed=311)
    causal = _reference(image, weight)
    symmetric = torch.nn.functional.conv3d(torch.nn.functional.pad(image.float(), (1, 1, 1, 1, 1, 1)), weight.float()).to(torch.bfloat16)
    missing_tail = _reference(image[:, :64], weight[:, :64])
    for wrong in (symmetric, missing_tail):
        with pytest.raises(AssertionError):
            _close(wrong, causal)


@pytest.mark.parametrize("unsupported", ["c32", "c80", "c95", "c97", "c112", "c160", "k128", "half", "stride2", "filter1", "symmetric", "ncdhw", "relu"])
def test_frost_conv_c96_declines_outside_qualified_envelope(unsupported):
    from cudnn.conv.frost.engine import FrostConvEngine

    c = int(unsupported[1:]) if unsupported.startswith("c") else 96
    k = 128 if unsupported == "k128" else 96
    dtype = torch.float16 if unsupported == "half" else torch.bfloat16
    image = make_input((1, c, 5, 5, 7), dtype, seed=410)
    filt = (1, 1, 1) if unsupported == "filter1" else (3, 3, 3)
    weight = make_input((k, c, *filt), dtype, seed=411)
    if unsupported == "ncdhw":
        image = image.contiguous()
    stride = (2, 1, 1) if unsupported == "stride2" else (1, 1, 1)
    pre = (1, 1, 1) if unsupported == "symmetric" else (2, 1, 1)
    post = (1, 1, 1) if unsupported == "symmetric" else (0, 1, 1)
    spatial = tuple((size + lo + hi - extent) // step + 1 for size, lo, hi, extent, step in zip(image.shape[2:], pre, post, filt, stride))
    output = torch.empty((1, k, *spatial), device="cuda", dtype=dtype, memory_format=torch.channels_last_3d)
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x, w = graph.tensor_like(image), graph.tensor_like(weight)
    y = graph.conv_fprop(x, w, pre_padding=pre, post_padding=post, stride=stride, dilation=(1, 1, 1))
    if unsupported == "relu":
        y = graph.relu(y)
    y.set_output(True).set_dim(output.shape).set_stride(output.stride()).set_data_type(
        cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    )
    graph.validate()
    with pytest.raises(NotImplementedError, match="frost_conv"):
        FrostConvEngine().check_support(graph)


def test_frost_conv_c96_stream_rebinding_and_capture(cudnn_handle):
    image = make_input((1, 96, 3, 5, 7), torch.bfloat16, seed=510)
    weight = make_input((192, 96, 3, 3, 3), torch.bfloat16, seed=511)
    graph, (x, w, y), output, workspace = _plan(image, weight)
    stream = torch.cuda.Stream()
    original_stream = cudnn.get_stream(cudnn_handle)
    captured = torch.cuda.CUDAGraph()
    try:
        for seed in (512, 513):
            rebound = make_input(tuple(image.shape), torch.bfloat16, seed=seed)
            output.fill_(float("nan"))
            stream.wait_stream(torch.cuda.current_stream())
            cudnn.set_stream(cudnn_handle, stream.cuda_stream)
            with torch.cuda.stream(stream):
                graph.execute({x: rebound, w: weight, y: output}, workspace, handle=cudnn_handle)
            stream.synchronize()
            _close(output, _reference(rebound, weight))
        # Capture with explicit handle/stream binding, then change contents at
        # the captured addresses so replay cannot pass using stale outputs.
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(captured, stream=stream):
            previous_mode = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                graph.execute({x: image, w: weight, y: output}, workspace, handle=cudnn_handle)
            finally:
                torch.cuda.set_sync_debug_mode(previous_mode)
        for value in (0.25, -0.5):
            image.fill_(value)
            output.fill_(float("nan"))
            captured.replay()
            torch.cuda.synchronize()
            _close(output, _reference(image, weight))
    finally:
        captured.reset()
        cudnn.set_stream(cudnn_handle, original_stream)


@pytest.mark.parametrize("c", [64, 96])
@pytest.mark.parametrize("mode", [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK, cudnn.heur_mode.OPENSOURCE])
def test_frost_conv_c96_default_placement_and_explicit_selection(c, mode):
    image = make_input((1, c, 1, 5, 7), torch.bfloat16, seed=610)
    weight = make_input((96, c, 3, 3, 3), torch.bfloat16, seed=611)
    output = torch.empty((1, 96, 1, 5, 7), dtype=torch.bfloat16, device="cuda", memory_format=torch.channels_last_3d)
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x, w = graph.tensor_like(image), graph.tensor_like(weight)
    y = graph.conv_fprop(x, w, pre_padding=(2, 1, 1), post_padding=(0, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1))
    y.set_output(True).set_dim(output.shape).set_stride(output.stride()).set_data_type(cudnn.data_type.BFLOAT16)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([mode])
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    index = names.index("frost_conv")
    if c == 64 or mode == cudnn.heur_mode.OPENSOURCE:
        assert index == 0
    else:
        assert index > 0, names
    graph.select_plan(index)
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(max(1, graph.get_workspace_size()), dtype=torch.uint8, device="cuda")
    graph.execute({x: image, w: weight, y: output}, workspace)
    _close(output, _reference(image, weight))
