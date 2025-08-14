import cudnn
import pytest
import torch
from looseversion import LooseVersion

"""
Test suite for DeviceProperties functionality in cuDNN Frontend.
Tests deviceless AoT compilation.
"""


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
    Y_tensor = graph.conv_fprop(
        X_tensor, W_tensor, padding=padding, stride=stride, dilation=dilation
    )
    Y_tensor.set_output(True)

    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    json_str = graph.serialize()

    # Step 3
    # Compute reference
    X_gpu = torch.randn(N, C, H, W, dtype=torch.float32, device="cuda").to(
        memory_format=torch.channels_last
    )
    W_gpu = torch.randn(K, C, R, S, dtype=torch.float32, device="cuda").to(
        memory_format=torch.channels_last
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
        Y_ref = torch.nn.functional.conv2d(
            X_gpu, W_gpu, padding=padding, stride=stride, dilation=dilation
        )

    # Create handle only when needed (for graph execution)
    cudnn_handle = cudnn.create_handle()
    try:
        stream = torch.cuda.current_stream().cuda_stream
        cudnn.set_stream(handle=cudnn_handle, stream=stream)

        graph_deserialized = cudnn.pygraph()
        graph_deserialized.deserialize(cudnn_handle, json_str)

        Y_actual = torch.zeros_like(Y_ref)

        workspace = torch.empty(
            graph_deserialized.get_workspace_size(), device="cuda", dtype=torch.uint8
        )

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
