"""Slice a tensor and compare with PyTorch."""

import cudnn
import torch


def sample():
    x = torch.randn((2, 3, 4), device="cuda", dtype=torch.float32)
    y = torch.empty((2, 2, 4), device="cuda", dtype=torch.float32)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x_desc = graph.tensor_like(x, name="X")
    y_desc = graph.slice(
        input=x_desc,
        slices=[slice(0, 2), slice(0, 2), slice(0, 4)],
        name="slice",
    )
    y_desc.set_output(True).set_name("Y").set_dim(y.shape).set_stride(y.stride())

    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute({x_desc: x, y_desc: y}, workspace)
    torch.cuda.synchronize()

    torch.testing.assert_close(y, x[:, :2, :])
    print(f"slice result matches reference: shape={tuple(y.shape)}")


if __name__ == "__main__":
    sample()
