import cudnn
import pytest
import torch
from looseversion import LooseVersion

from test_utils import torch_fork_set_rng


class CSBR(torch.nn.Module):
    def forward(
        self,
        x,
        w,
        b=None,
        padding=[1, 1],
        stride=[1, 1],
        dilation=[1, 1],
        lower_clip=0.0,
        upper_clip=128,
    ):
        if b is not None:
            b = b.reshape(-1)  # Conv2d needs a 1D tensor
        conv_output = torch.nn.functional.conv2d(
            x, w, bias=b, padding=padding, stride=stride, dilation=dilation
        )
        return torch.clamp(conv_output, min=lower_clip, max=upper_clip)


@cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
def create_conv_bias_relu_graph(
    handle,
    X_gpu,
    W_gpu,
    B_gpu,
    padding,
    stride,
    dilation,
    lower_clip=0.5,
    upper_clip=0.55,
):
    with cudnn.graph(
        handle,
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    ) as (g, _):
        X = g.tensor_like(X_gpu)
        W = g.tensor_like(W_gpu)
        B = g.tensor_like(B_gpu)

        conv_output = g.conv_fprop(
            image=X,
            weight=W,
            pre_padding=padding,
            post_padding=padding,
            stride=stride,
            dilation=dilation,
        )

        bias_output = g.bias(name="bias", input=conv_output, bias=B)
        Y = g.relu(
            name="relu", input=bias_output, lower_clip=lower_clip, upper_clip=upper_clip
        )
        Y.set_output(True)

        return g, [X, W, B, Y]


@cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
def create_conv_relu_graph(
    handle, X_gpu, W_gpu, padding, stride, dilation, lower_clip=0.5, upper_clip=0.55
):
    with cudnn.graph(
        handle,
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    ) as (g, _):
        X = g.tensor_like(X_gpu)
        W = g.tensor_like(W_gpu)

        conv_output = g.conv_fprop(
            image=X, weight=W, padding=padding, stride=stride, dilation=dilation
        )

        Y = g.relu(
            name="relu", input=conv_output, lower_clip=lower_clip, upper_clip=upper_clip
        )
        Y.set_output(True)

        return g, [X, W, Y]


@pytest.mark.L0
@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "9.5.0",
    reason="requires cudnn 9.5.0 or higher",
)
@torch_fork_set_rng(seed=0)
def test_conv_bias_relu(cudnn_handle):
    # Reference code
    X_gpu = torch.randn(4, 16, 56, 56, device="cuda", dtype=torch.float16).to(
        memory_format=torch.channels_last
    )
    W_gpu = torch.randn(16, 16, 3, 3, device="cuda", dtype=torch.float16).to(
        memory_format=torch.channels_last
    )
    B_gpu = torch.randn(1, 16, 1, 1, device="cuda", dtype=torch.float16).to(
        memory_format=torch.channels_last
    )
    padding = [1, 1]
    stride = [3, 3]
    dilation = [1, 1]

    # Get reference result
    model = CSBR().eval().to("cuda").to(torch.float16)
    Y_expected = model(
        X_gpu,
        W_gpu,
        b=B_gpu,
        padding=padding,
        stride=stride,
        dilation=dilation,
        lower_clip=0.5,
        upper_clip=0.55,
    )

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    single_mode_graph = cudnn.jit(heur_modes=cudnn.heur_mode.A)(
        create_conv_bias_relu_graph.__wrapped__
    )
    g, uids = single_mode_graph(
        cudnn_handle, X_gpu, W_gpu, B_gpu, padding, stride, dilation
    )

    X_uid, W_uid, B_uid, Y_uid = uids

    Y_actual = torch.zeros_like(Y_expected)
    workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)

    g.execute(
        {X_uid: X_gpu, W_uid: W_gpu, B_uid: B_gpu, Y_uid: Y_actual},
        workspace,
        handle=cudnn_handle,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(Y_expected, Y_actual, atol=0.05, rtol=1e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_conv_relu(cudnn_handle):
    # Reference code
    X_gpu = torch.randn(20, 40, 30, 40, device="cuda", dtype=torch.float16).to(
        memory_format=torch.channels_last
    )
    W_gpu = torch.randn(54, 40, 3, 4, device="cuda", dtype=torch.float16).to(
        memory_format=torch.channels_last
    )
    padding = [0, 1]
    stride = [2, 3]
    dilation = [1, 1]

    # Get reference result
    model = CSBR().eval().to("cuda").to(torch.float16)
    Y_expected = model(
        X_gpu,
        W_gpu,
        padding=padding,
        stride=stride,
        dilation=dilation,
        lower_clip=0.5,
        upper_clip=0.55,
    )

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    g, uids = create_conv_relu_graph(
        cudnn_handle, X_gpu, W_gpu, padding, stride, dilation
    )
    X_uid, W_uid, Y_uid = uids

    Y_actual = torch.zeros_like(Y_expected)
    workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)

    g.execute(
        {X_uid: X_gpu, W_uid: W_gpu, Y_uid: Y_actual}, workspace, handle=cudnn_handle
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-3, rtol=1e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_conv_relu_execution_plan_creation(cudnn_handle):
    # Reference code
    X_gpu = torch.randn(
        20, 40, 30, 40, requires_grad=False, device="cuda", dtype=torch.float16
    ).to(memory_format=torch.channels_last)
    W_gpu = torch.randn(
        54, 40, 3, 4, requires_grad=False, device="cuda", dtype=torch.float16
    ).to(memory_format=torch.channels_last)
    padding = [0, 1]
    stride = [2, 3]
    dilation = [1, 1]
    model = CSBR().eval().to("cuda").to(torch.float16)
    Y_expected = model(
        X_gpu,
        W_gpu,
        padding=padding,
        stride=stride,
        dilation=dilation,
        lower_clip=0.5,
        upper_clip=0.55,
    )

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    # Cudnn code
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )

    X = graph.tensor(
        name="X", dim=X_gpu.size(), stride=X_gpu.stride(), data_type=X_gpu.dtype
    )
    W = graph.tensor(
        name="W", dim=W_gpu.size(), stride=W_gpu.stride(), data_type=W_gpu.dtype
    )

    conv_output = graph.conv_fprop(
        image=X, weight=W, padding=padding, stride=stride, dilation=dilation
    )

    Y = graph.relu(name="relu", input=conv_output, lower_clip=0.5, upper_clip=0.55)
    Y.set_output(True)

    graph.validate()
    graph.build_operation_graph()

    # Build all unique kernel cfg plans
    for engine in range(graph.get_engine_count()):
        try:
            knobs = graph.get_knobs_for_engine(engine)
        except RuntimeError:
            continue

        for knob in knobs:
            if knob.type == cudnn.knob_type.KERNEL_CFG:
                for kernel_cfg in range(
                    knob.min_value, knob.max_value + 1, knob.stride
                ):
                    try:
                        graph.create_execution_plan(
                            engine, {cudnn.knob_type.KERNEL_CFG: kernel_cfg}
                        )
                    except RuntimeError:
                        continue

    graph.check_support()
    graph.build_plans()

    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )

    Y_actual = torch.zeros_like(Y_expected)
    graph.execute({X: X_gpu, W: W_gpu, Y: Y_actual}, workspace, handle=cudnn_handle)
    # Compare
    torch.cuda.synchronize()
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-3, rtol=1e-3)


@cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
def create_conv3d_bias_leaky_relu_graph(
    handle, X_gpu, W_gpu, B_gpu, padding, stride, dilation, negative_slope
):
    with cudnn.graph(
        handle,
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    ) as (g, _):
        X = g.tensor_like(X_gpu)
        W = g.tensor_like(W_gpu)
        B = g.tensor_like(B_gpu)

        conv_output = g.conv_fprop(
            image=X, weight=W, padding=padding, stride=stride, dilation=dilation
        )

        bias_output = g.bias(name="bias", input=conv_output, bias=B)
        Y = g.leaky_relu(name="relu", input=bias_output, negative_slope=negative_slope)
        Y.set_output(True)

        return g, [X, W, B, Y]


@pytest.mark.L0
@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "9.5.0",
    reason="requires cudnn 9.5.0 or higher",
)
@torch_fork_set_rng(seed=0)
def test_conv3d_bias_leaky_relu(cudnn_handle):
    N, C, D, H, W = 4, 16, 52, 54, 56
    K, R, S, T = 32, 3, 3, 3
    padding = [0, 1, 2]
    stride = [2, 3, 4]
    dilation = [1, 1, 1]
    negative_slope = 0.01

    # Reference code
    X_gpu = torch.randn(N, D, H, W, C, device="cuda", dtype=torch.float16).permute(
        0, 4, 1, 2, 3
    )
    W_gpu = torch.randn(K, R, S, T, C, device="cuda", dtype=torch.float16).permute(
        0, 4, 1, 2, 3
    )
    B_gpu = torch.randn(1, 1, 1, 1, K, device="cuda", dtype=torch.float16).permute(
        0, 4, 1, 2, 3
    )

    # Get reference result
    conv_out_expected = (
        torch.nn.functional.conv3d(
            X_gpu,
            W_gpu,
            bias=B_gpu.reshape(-1),
            padding=padding,
            stride=stride,
            dilation=dilation,
        )
        .to("cuda")
        .to(torch.float16)
    )
    Y_expected = torch.nn.functional.leaky_relu(
        conv_out_expected, negative_slope=negative_slope
    )

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    g, uids = create_conv3d_bias_leaky_relu_graph(
        cudnn_handle, X_gpu, W_gpu, B_gpu, padding, stride, dilation, negative_slope
    )
    X_uid, W_uid, B_uid, Y_uid = uids

    Y_actual = torch.zeros_like(Y_expected)
    workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)

    g.execute(
        {X_uid: X_gpu, W_uid: W_gpu, B_uid: B_gpu, Y_uid: Y_actual},
        workspace,
        handle=cudnn_handle,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-2, rtol=1e-2)


@cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
def create_leaky_relu_backward_graph(handle, loss_gpu, input_gpu, negative_slope):
    with cudnn.graph(
        handle,
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    ) as (g, _):
        loss = g.tensor_like(loss_gpu)
        input = g.tensor_like(input_gpu)

        Y = g.leaky_relu_backward(loss=loss, input=input, negative_slope=negative_slope)
        Y.set_output(True)

        return g, [loss, input, Y]


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_leaky_relu_backward(cudnn_handle):
    N, C, H, W = 4, 16, 56, 56
    negative_slope = 0.01

    # Reference code
    loss_gpu = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).to(
        memory_format=torch.channels_last
    )
    input_gpu = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).to(
        memory_format=torch.channels_last
    )

    def dleaky_relu(grad: torch.Tensor, mask: torch.Tensor, negative_slope: float):
        return torch.ones_like(grad).masked_fill_(mask <= 0.0, negative_slope) * grad

    Y_expected = dleaky_relu(loss_gpu, input_gpu, negative_slope)

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    g, uids = create_leaky_relu_backward_graph(
        cudnn_handle, loss_gpu, input_gpu, negative_slope
    )
    loss_uid, input_uid, Y_uid = uids

    Y_actual = torch.zeros_like(Y_expected)
    workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)

    g.execute(
        {loss_uid: loss_gpu, input_uid: input_gpu, Y_uid: Y_actual},
        workspace,
        handle=cudnn_handle,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-4, rtol=1e-4)


@cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
def create_conv_int8_graph(handle, X_gpu, W_gpu, padding, stride, dilation):
    with cudnn.graph(
        handle,
        io_data_type=cudnn.data_type.INT8,
        intermediate_data_type=cudnn.data_type.INT32,
        compute_data_type=cudnn.data_type.INT32,
    ) as (g, _):
        X = g.tensor_like(X_gpu)
        W = g.tensor_like(W_gpu)

        conv_output = g.conv_fprop(
            image=X, weight=W, padding=padding, stride=stride, dilation=dilation
        )
        Y = g.identity(name="identity", input=conv_output)
        Y.set_output(True).set_data_type(cudnn.data_type.INT32)

        return g, [X, W, Y]


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "8.6",
    reason="requires cudnn 8.6.0 or higher",
)
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_conv_int8(cudnn_handle):
    N, C, H, W = 2, 64, 32, 32
    K, R, S = 4, 3, 3
    padding = [1, 1]
    stride = [1, 1]
    dilation = [1, 1]

    compare_output = True

    # Reference code
    X_gpu = torch.randint(-127, 128, (N, C, H, W), device="cuda", dtype=torch.int8).to(
        memory_format=torch.channels_last
    )
    W_gpu = torch.randint(-127, 128, (K, C, R, S), device="cuda", dtype=torch.int8).to(
        memory_format=torch.channels_last
    )

    try:
        Y_expected = (
            torch.nn.functional.conv2d(
                X_gpu, W_gpu, padding=padding, stride=stride, dilation=dilation
            )
            .to("cuda")
            .to(torch.int32)
        )
    except:
        print(
            "Torch does not support int8 convolution. Disabling comparison of output tensor"
        )
        compare_output = False

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    g, uids = create_conv_int8_graph(
        cudnn_handle, X_gpu, W_gpu, padding, stride, dilation
    )
    X_uid, W_uid, Y_uid = uids

    Y_actual = torch.randint(0, 127, X_gpu.size(), device="cuda", dtype=torch.int32).to(
        memory_format=torch.channels_last
    )
    workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)

    g.execute(
        {X_uid: X_gpu, W_uid: W_gpu, Y_uid: Y_actual}, workspace, handle=cudnn_handle
    )

    torch.cuda.synchronize()

    if compare_output:
        torch.testing.assert_close(Y_expected, Y_actual, atol=1e-2, rtol=1e-2)
