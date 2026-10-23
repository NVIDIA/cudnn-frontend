"""Concatenate tensors and compare with PyTorch."""

import cudnn
import torch


def sample():
    x = torch.randn((2, 3, 4), device="cuda", dtype=torch.float32)
    z = torch.randn_like(x)
    y = torch.empty((2, 6, 4), device="cuda", dtype=torch.float32)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x_desc = graph.tensor_like(x, name="X")
    z_desc = graph.tensor_like(z, name="Z")
    y_desc = graph.concatenate(inputs=[x_desc, z_desc], axis=1, name="concatenate")
    y_desc.set_output(True).set_name("Y").set_dim(y.shape).set_stride(y.stride())

    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute({x_desc: x, z_desc: z, y_desc: y}, workspace)
    torch.cuda.synchronize()

    torch.testing.assert_close(y, torch.cat((x, z), dim=1))
    print(f"concatenate result matches reference: shape={tuple(y.shape)}")


if __name__ == "__main__":
    sample()
