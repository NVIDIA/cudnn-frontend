"""Build and execute a BF16 matmul with the high-level cudnn.Graph wrapper."""

import cudnn
import torch


def main() -> None:
    # 1. Allocate the runtime tensors that define the problem and hold its data.
    torch.manual_seed(0)
    a_data = torch.randn(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    b_data = torch.randn(1, 8, 6, device="cuda", dtype=torch.bfloat16)

    # 2. Describe the graph. Exiting the block validates it, selects an
    #    execution plan, builds that plan, and allocates its workspace.
    with cudnn.Graph(
        io_data_type=torch.bfloat16,
        compute_data_type=torch.float32,
        inputs=["matmul::A", "matmul::B"],
        outputs=["C"],
        handle="auto",
    ) as graph:
        result_desc = graph.matmul(name="matmul", A=a_data, B=b_data)
        result_desc.set_name("C").set_output(True).set_data_type(torch.bfloat16)

    # 3. Execute with the actual input buffers. The wrapper allocates and
    #    returns the output tensor in the order declared by outputs=[...].
    result = graph(a_data, b_data)

    # 4. Check the cuDNN result against the same operation in PyTorch.
    expected = torch.matmul(a_data.float(), b_data.float()).to(torch.bfloat16)
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
    print(f"matmul ok: {tuple(result.shape)}, workspace={graph.get_workspace_size()} bytes")


if __name__ == "__main__":
    main()
