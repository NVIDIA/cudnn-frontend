import cudnn
import pytest
import torch
from looseversion import LooseVersion

from test_utils import torch_fork_set_rng


class Conv_Genstats(torch.nn.Module):
    def forward(
        self, scale, bias, x, w, padding=[1, 1], stride=[1, 1], dilation=[1, 1]
    ):
        x_conv = torch.relu(x * scale + bias)
        conv_output = torch.nn.functional.conv2d(
            x_conv, w, padding=padding, stride=stride, dilation=dilation
        )
        sum = torch.sum(conv_output, dim=(0, 2, 3), dtype=torch.float32)
        sq_sum = torch.sum(
            torch.square(conv_output), dim=(0, 2, 3), dtype=torch.float32
        )
        return conv_output, sum, sq_sum


model = Conv_Genstats().eval().to("cuda")

n = 4
c = 32
k = 64
padding = [1, 1]
stride = [1, 1]
dilation = [1, 1]


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "8.8",
    reason="requires cudnn 8.8 or higher",
)
@torch_fork_set_rng(seed=0)
def test_conv_genstats():

    # Reference
    X_gpu = torch.randn(
        n, c, 32, 32, requires_grad=False, device="cuda", dtype=torch.float16
    ).to(memory_format=torch.channels_last)
    W_gpu = torch.randn(
        k, c, 3, 3, requires_grad=False, device="cuda", dtype=torch.float16
    ).to(memory_format=torch.channels_last)
    scale = (
        torch.randn(1, c, 1, 1, device="cuda", dtype=torch.float16).to(
            memory_format=torch.channels_last
        )
        * 0.01
    )
    bias = (
        torch.randn(1, c, 1, 1, device="cuda", dtype=torch.float16).to(
            memory_format=torch.channels_last
        )
        * 0.01
    )
    Y_expected, sum_expected, sq_sum_expected = model(
        scale, bias, X_gpu, W_gpu, padding=padding, stride=stride, dilation=dilation
    )

    handle = cudnn.create_handle()
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=handle, stream=stream)

    # Cudnn code
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.HALF,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )

    X = graph.tensor(
        name="X", dim=X_gpu.size(), stride=X_gpu.stride(), data_type=X_gpu.dtype
    )
    W = graph.tensor(
        name="W", dim=W_gpu.size(), stride=W_gpu.stride(), data_type=W_gpu.dtype
    )

    S = graph.tensor(
        name="S", dim=scale.size(), stride=scale.stride(), data_type=scale.dtype
    )
    B = graph.tensor(
        name="B", dim=bias.size(), stride=bias.stride(), data_type=bias.dtype
    )

    S_OUT = graph.scale(name="scale", input=X, scale=S)
    B_OUT = graph.bias(name="bias", input=S_OUT, bias=B)
    CONV_IN = graph.relu(name="relu", input=B_OUT)
    Y = graph.conv_fprop(
        image=CONV_IN, weight=W, padding=padding, stride=stride, dilation=dilation
    )
    Y.set_output(True)

    SUM, SQ_SUM = graph.genstats(name="genstats", input=Y)
    SUM.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    SQ_SUM.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    sum_dev = torch.zeros_like(sum_expected)
    sq_sum_dev = torch.zeros_like(sq_sum_expected)
    Y_actual = torch.zeros_like(Y_expected)

    # Below tests capability to run with just device pointers
    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    graph.execute(
        {
            X: X_gpu.data_ptr(),
            W: W_gpu,
            Y: Y_actual.data_ptr(),
            SUM: sum_dev,
            SQ_SUM: sq_sum_dev,
            S: scale,
            B: bias,
        },
        workspace.data_ptr(),
        handle=handle,
    )

    # Compare
    torch.cuda.synchronize()
    torch.testing.assert_close(sum_expected, sum_dev, atol=0.5, rtol=1e-2)
    torch.testing.assert_close(sq_sum_expected, sq_sum_dev, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-3, rtol=1e-3)
    cudnn.destroy_handle(handle)


if __name__ == "__main__":
    test_conv_genstats()
