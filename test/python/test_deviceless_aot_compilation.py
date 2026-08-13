# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json as _json

import cudnn
import pytest
import torch
from looseversion import LooseVersion

"""
Test suite for DeviceProperties functionality in cuDNN Frontend.
Tests deviceless AoT compilation.
"""


@pytest.mark.L0
def test_deserialize_enforce_precompiled_rejects_json_graph():
    with pytest.raises(Exception, match="enforce_precompiled requires plan serialization"):
        cudnn.pygraph().deserialize("{}", enforce_precompiled=True)


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "9.11",
    reason="requires cudnn 9.11 or higher",
)
@pytest.mark.L0
def test_device_properties():
    # Step 1
    # Create original device properties and initialize it with device 0
    device_props_original = cudnn.create_device_properties(0)

    # Serialize
    json_str_original = device_props_original.serialize()

    # Deserialize a new object
    device_props_deserialized = cudnn.create_device_properties(json_str_original)

    # Serialize the deserialized object
    json_str_deserialized = device_props_deserialized.serialize()
    # print(f"Device properties: {json_str_deserialized}")

    # Verify the serialized JSON strings are identical
    assert json_str_original == json_str_deserialized

    # Step 2
    # Create a conv graph with the deserialized device properties
    N, K, C, H, W, R, S = 16, 256, 128, 64, 64, 3, 3
    padding = stride = dilation = [1, 1]

    # NHWC layout
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        device_property=device_props_deserialized,
    )
    X_tensor = graph.tensor(
        name="X",
        dim=[N, C, H, W],
        stride=[C * H * W, 1, C * W, C],
    )
    W_tensor = graph.tensor(
        name="W",
        dim=[K, C, R, S],
        stride=[C * R * S, 1, C * S, C],
    )
    Y_tensor = graph.conv_fprop(X_tensor, W_tensor, padding=padding, stride=stride, dilation=dilation)
    Y_tensor.set_output(True)

    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    json_str = graph.serialize()

    # Step 3
    # Compute reference
    X_gpu = torch.randn(N, C, H, W, dtype=torch.float32, device="cuda").to(memory_format=torch.channels_last)
    W_gpu = torch.randn(K, C, R, S, dtype=torch.float32, device="cuda").to(memory_format=torch.channels_last)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
        Y_ref = torch.nn.functional.conv2d(X_gpu, W_gpu, padding=padding, stride=stride, dilation=dilation)

    # Create handle only when needed (for graph execution)
    cudnn_handle = cudnn.create_handle()
    try:
        stream = torch.cuda.current_stream().cuda_stream
        cudnn.set_stream(handle=cudnn_handle, stream=stream)

        graph_deserialized = cudnn.pygraph()
        graph_deserialized.deserialize(cudnn_handle, json_str, enforce_precompiled=True)

        Y_actual = torch.zeros_like(Y_ref)

        workspace = torch.empty(graph_deserialized.get_workspace_size(), device="cuda", dtype=torch.uint8)

        graph_deserialized.execute(
            {X_tensor: X_gpu, W_tensor: W_gpu, Y_tensor: Y_actual},
            workspace,
            handle=cudnn_handle,
        )

        torch.cuda.synchronize()

        # Compare
        torch.testing.assert_close(Y_ref, Y_actual, atol=1e-3, rtol=1e-3)
    finally:
        cudnn.destroy_handle(cudnn_handle)


def _make_wrong_arch_devprop():
    """Return a DeviceProperties whose deviceVer has been changed to a different SM."""
    dp = cudnn.create_device_properties(0)
    raw = dp.serialize()  # returns a JSON string
    data = _json.loads(raw)
    real_ver = data["deviceVer"]
    # Pick a different SM (one major step up, or down if already high)
    wrong_ver = (real_ver + 100) if real_ver < 900 else (real_ver - 100)
    data["deviceVer"] = wrong_ver
    modified = _json.dumps(data, separators=(",", ":"))
    return cudnn.create_device_properties(modified)


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "9.11",
    reason="requires cudnn 9.11 or higher",
)
@pytest.mark.L0
def test_deviceless_deserialize():
    """Handle-less deserialize: cudnn.pygraph(device_property=dp).deserialize(blob).

    The wrong-arch sub-test is the discriminator: if the device property were
    silently ignored (as it would be without this feature), the wrong-arch devprop
    would succeed and the test would pass vacuously. It must raise to prove the
    devprop is load-bearing.
    """
    assert cudnn.backend_version() >= 90800, f"expected cuDNN >= 9.8, got {cudnn.backend_version_string()}"

    N, K, C, H, W, R, S = 1, 4, 4, 8, 8, 1, 1
    padding = [0, 0]
    stride = [1, 1]
    dilation = [1, 1]

    dp = cudnn.create_device_properties(0)

    # Build with devprop (handle-less AoT path).
    graph_build = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        device_property=dp,
    )
    X_t = graph_build.tensor(name="X", dim=[N, C, H, W], stride=[C * H * W, 1, C * W, C])
    W_t = graph_build.tensor(name="W", dim=[K, C, R, S], stride=[C * R * S, 1, C * S, C])
    Y_t = graph_build.conv_fprop(X_t, W_t, padding=padding, stride=stride, dilation=dilation)
    Y_t.set_output(True)
    graph_build.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    blob = graph_build.serialize()

    # ── sub-test 1: wrong-arch devprop must raise ────────────────────────────
    dp_wrong = _make_wrong_arch_devprop()
    graph_wrong = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        device_property=dp_wrong,
    )
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="NOT_SUPPORTED"):
        graph_wrong.deserialize(blob)

    # ── sub-test 2: correct devprop → deserialize + execute ─────────────────
    graph_deser = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        device_property=dp,
    )
    # Must NOT create a cuDNN handle internally; verify by confirming no handle
    # attribute is set (the C++ PyGraph stores None / nullptr for handle_).
    graph_deser.deserialize(blob)

    # Reference output via torch
    X_gpu = torch.ones(N, C, H, W, dtype=torch.float32, device="cuda").to(memory_format=torch.channels_last)
    W_gpu = torch.ones(K, C, R, S, dtype=torch.float32, device="cuda").to(memory_format=torch.channels_last)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
        Y_ref = torch.nn.functional.conv2d(X_gpu, W_gpu, padding=[0, 0], stride=[1, 1], dilation=[1, 1])

    cudnn_handle = cudnn.create_handle()
    try:
        stream = torch.cuda.current_stream().cuda_stream
        cudnn.set_stream(handle=cudnn_handle, stream=stream)

        Y_actual = torch.zeros_like(Y_ref)
        workspace = torch.empty(graph_deser.get_workspace_size(), device="cuda", dtype=torch.uint8)
        graph_deser.execute(
            {X_t: X_gpu, W_t: W_gpu, Y_t: Y_actual},
            workspace,
            handle=cudnn_handle,
        )
        torch.cuda.synchronize()

        # With X=1 and W=1 and a 1x1 conv with no padding, Y = C = 4.
        torch.testing.assert_close(Y_actual, Y_ref, atol=1e-4, rtol=1e-4)
    finally:
        cudnn.destroy_handle(cudnn_handle)
