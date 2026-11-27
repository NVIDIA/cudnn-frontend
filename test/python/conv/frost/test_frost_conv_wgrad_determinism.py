# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wgrad replay coverage; dgrad/higher derivatives are not provided by this engine.

The batch axis is reduced into shared dW. Reordering or splitting that reduction
is a distinct numerical contract and is characterized separately below.
"""

import pytest
import torch
import cudnn

from determinism_utils import assert_bitwise, assert_replays, snapshot
from test_frost_conv_wgrad import DTYPES, prepare, reference, requires_dense_cutedsl, requires_wgrad_gpu

pytestmark = [pytest.mark.L0, requires_dense_cutedsl, requires_wgrad_gpu]


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: "in-" + str(dt).removeprefix("torch."))
@pytest.mark.parametrize("output_dtype", DTYPES, ids=lambda dt: "out-" + str(dt).removeprefix("torch."))
@pytest.mark.parametrize("group", (1, 2))
def test_frost_wgrad_replay(monkeypatch, cudnn_handle, dtype, output_dtype, group):
    # Multiple reduction-ring wraps, partial K/C tiles, and multiple persistent
    # work items. Preserve quantized input payloads across fresh allocations.
    knobs = {cudnn.knob_type.CTA_GROUP: group}
    g, pack, ws, (x, dy, dw) = prepare(
        monkeypatch, (2, 80, 5, 7, 19), (144, 80, 2, 2, 2), dtype=dtype, output_dtype=output_dtype, knobs=knobs, handle=cudnn_handle
    )
    want = reference(x, dy, dw.shape, output_dtype=output_dtype)
    keys = tuple(pack)
    original_inputs = snapshot({"x": x, "dy": dy})

    def run():
        current = {keys[0]: x.clone(), keys[1]: dy.clone(), keys[2]: torch.empty_like(dw)}
        g.execute(current, ws)
        torch.testing.assert_close(current[keys[2]].float(), want.float(), atol=0, rtol=0)
        return {"dw": current[keys[2]]}

    assert_replays(run)
    assert_bitwise(snapshot({"x": x, "dy": dy}), original_inputs)
    g.execute(pack, ws)
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    old_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        for _ in range(3):
            g.execute(pack, ws)
    finally:
        torch.cuda.set_sync_debug_mode(old_mode)
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    capture = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cudnn.set_stream(cudnn_handle, stream.cuda_stream)
    with torch.cuda.graph(capture, stream=stream):
        g.execute(pack, ws)
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    x.copy_(-x.float())
    capture.replay()
    torch.testing.assert_close(dw.float(), reference(x, dy, dw.shape, output_dtype=output_dtype).float(), atol=0, rtol=0)


def test_frost_wgrad_batch_reduction_characterization(monkeypatch):
    # FP32 output makes changes in FP32 addition order visible. Input values are
    # built once; every graph gets those same values and upstream gradients.
    g, pack, ws, (x, dy, dw) = prepare(monkeypatch, (3, 16, 1, 1, 257), (16, 16, 1, 1, 1), output_dtype=torch.float32)
    x.normal_()
    dy.normal_()
    g.execute(pack, ws)
    baseline = snapshot({"dw": dw})
    torch.testing.assert_close(dw, reference(x, dy, dw.shape, output_dtype=torch.float32), atol=1e-4, rtol=1e-4)
    saved_x, saved_dy = x.clone(), dy.clone()
    x.copy_(saved_x.flip(0))
    dy.copy_(saved_dy.flip(0))
    g.execute(pack, ws)
    torch.testing.assert_close(dw, reference(x, dy, dw.shape, output_dtype=torch.float32), atol=1e-4, rtol=1e-4)
    # Characterization records a known non-guarantee, rather than disguising
    # batch-reduction differences with a tolerant determinism assertion.
    different = not torch.equal(dw.reshape(-1).contiguous().view(torch.uint8), baseline["dw"].reshape(-1).contiguous().view(torch.uint8))
    print(f"batch reordering changed raw dW bytes: {different}")
    partials = []
    for i in range(3):
        sg, sp, sw, (sx, sy, sdw) = prepare(monkeypatch, (1, 16, 1, 1, 257), (16, 16, 1, 1, 1), output_dtype=torch.float32)
        sx.copy_(saved_x[i : i + 1])
        sy.copy_(saved_dy[i : i + 1])
        sg.execute(sp, sw)
        partials.append(sdw.clone())
    combined = sum(partials)
    torch.testing.assert_close(combined, baseline["dw"], atol=1e-4, rtol=1e-4)
    print(
        f"batch splitting changed raw dW bytes: {not torch.equal(combined.reshape(-1).contiguous().view(torch.uint8), baseline['dw'].reshape(-1).contiguous().view(torch.uint8))}"
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: "in-" + str(dt).removeprefix("torch."))
def test_frost_wgrad_golden(monkeypatch, dtype):
    from determinism_utils import assert_golden

    g, pack, ws, (x, dy, dw) = prepare(monkeypatch, (1, 16, 1, 3, 7), (16, 16, 1, 1, 1), dtype=dtype, output_dtype=torch.float32)
    g.execute(pack, ws)
    torch.testing.assert_close(dw, reference(x, dy, dw.shape, output_dtype=torch.float32), atol=0, rtol=0)
    saved = snapshot({"dw": dw})
    g.execute(pack, ws)
    assert_bitwise(snapshot({"dw": dw}), saved)
    assert_golden(f"frost_conv_wgrad_{str(dtype).split('.')[-1]}", {"x": x, "dy": dy}, saved)
