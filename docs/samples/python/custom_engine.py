"""Inspect automatic Python-engine and backend plan selection."""

import cudnn
import torch


def main() -> None:
    a = torch.randn((1, 16, 32), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((1, 32, 8), device="cuda", dtype=torch.bfloat16)
    result = torch.empty((1, 16, 8), device="cuda", dtype=torch.bfloat16)

    graph = cudnn.pygraph(
        io_data_type=torch.bfloat16,
        intermediate_data_type=torch.float32,
        compute_data_type=torch.float32,
    )
    A = graph.tensor_like(a, name="A")
    B = graph.tensor_like(b, name="B")
    C = graph.matmul(A=A, B=B, compute_data_type=torch.float32, name="matmul")
    C.set_output(True).set_name("C").set_data_type(torch.bfloat16)

    # --8<-- [start:inspection]
    assert graph.find_tensor("A") is A
    assert graph.get_node("matmul") is graph.nodes[0]
    assert graph.get_inputs() == [A, B]
    assert graph.get_outputs() == [C]
    assert graph.inspect()["nodes"][0]["name"] == "matmul"
    # --8<-- [end:inspection]

    # --8<-- [start:routing]
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B])
    assert graph.plans
    graph.select_plan(0)
    graph.check_support()
    graph.build_plans()
    owner = graph.selected_engine.name if graph.selected_engine else "cuDNN backend"
    # --8<-- [end:routing]

    # --8<-- [start:execution]
    workspace = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)
    graph.execute({A: a, B: b, C: result}, workspace)
    torch.testing.assert_close(result, torch.matmul(a.float(), b.float()).to(torch.bfloat16), rtol=1e-2, atol=1e-2)
    # --8<-- [end:execution]

    print(f"selected plan owner: {owner}")


if __name__ == "__main__":
    main()
