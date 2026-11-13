# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary serialization preserves the legacy format and executable plan."""

import cudnn
import pytest
import torch

pytestmark = pytest.mark.L0


@pytest.fixture
def serialized_conv(cudnn_handle):
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    x = graph.tensor(dim=[2, 8, 16, 16], stride=[2048, 1, 128, 8], data_type=cudnn.data_type.HALF)
    w = graph.tensor(dim=[16, 8, 3, 3], stride=[72, 1, 24, 8], data_type=cudnn.data_type.HALF)
    y = graph.conv_fprop(image=x, weight=w, padding=[1, 1], stride=[1, 1], dilation=[1, 1])
    y.set_output(True)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    return graph, x, w, y


def test_serialize_bytes_preserves_default(serialized_conv):
    graph, *_ = serialized_conv
    legacy = graph.serialize()
    binary = graph.serialize(as_bytes=True)
    assert type(legacy) is list
    assert type(binary) is bytes
    assert graph.serialize(as_bytes=False) == legacy
    assert binary == bytes(legacy)
    assert b"\0" in binary  # Embedded NULs must not truncate the blob.
    native = graph._lowered_graph
    assert type(native.serialize()) is list
    assert native.serialize(as_bytes=True) == binary
    for target in (graph, native):
        with pytest.raises(TypeError):
            target.serialize(True)  # The new option is keyword-only.


class BytesSubclass(bytes):
    pass


@pytest.mark.parametrize("payload_type", [list, tuple, bytes, BytesSubclass])
@pytest.mark.parametrize("handle_form", ["implicit", "positional", "keyword"])
def test_deserialize_bytes_executes(serialized_conv, cudnn_handle, payload_type, handle_form):
    graph, x, w, y = serialized_conv
    payload = payload_type(graph.serialize())
    restored = cudnn.pygraph(handle=cudnn_handle)
    if handle_form == "positional":
        restored.deserialize(cudnn_handle, payload, enforce_precompiled=True)
    elif handle_form == "keyword":
        restored.deserialize(handle_=cudnn_handle, pyobj=payload, enforce_precompiled=True)
    else:
        restored.deserialize(payload, enforce_precompiled=True)
    stream = torch.cuda.ExternalStream(cudnn.get_stream(cudnn_handle))
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        weight = torch.ones([16, 8, 3, 3], device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
        output = torch.empty([2, 16, 16, 16], device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
        workspace = torch.empty(restored.get_workspace_size(), device="cuda", dtype=torch.uint8)
        for value in (1, 2):
            image = torch.full([2, 8, 16, 16], value, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
            output.fill_(float("nan"))
            restored.execute({x: image, w: weight, y: output}, workspace, handle=cudnn_handle)
            expected = torch.nn.functional.conv2d(image, weight, padding=1)
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
    stream.synchronize()


@pytest.mark.parametrize("payload", [b"", b"invalid serialized graph"])
def test_deserialize_malformed_bytes_matches_list(cudnn_handle, payload):
    errors = []
    for data in (list(payload), payload):
        graph = cudnn.pygraph(handle=cudnn_handle)
        with pytest.raises(Exception) as exc:
            graph.deserialize(data)
        errors.append((type(exc.value), str(exc.value)))
    assert errors[0] == errors[1]
