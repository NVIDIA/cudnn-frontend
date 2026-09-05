"""Functional coverage for the binary modulo pointwise ops.

Each op is lowered into a single-node cuDNN graph that is built and executed on
the GPU, then compared against a torch reference:

  * mod        -> truncated remainder (C fmod/%; the result follows the sign of
                  the dividend), reference torch.fmod
  * floor_mod  -> floored remainder (the result follows the sign of the divisor),
                  reference torch.remainder

floor_mod maps to the backend CUDNN_POINTWISE_FLOOR_MOD enum, which was added in
cuDNN 9.26.0. On older backends the frontend resolves it to
CUDNN_STATUS_NOT_SUPPORTED, so the graph build declines. The floor_mod case is
therefore version-gated and additionally wrapped in the standard
cudnnGraphNotSupportedError skip, so it is waived (never failed) on pre-9.26
cuDNN or on hardware/configs that cannot serve it.
"""

import cudnn
import pytest
import torch

# First cuDNN backend version exposing CUDNN_POINTWISE_FLOOR_MOD.
FLOOR_MOD_MIN_BACKEND_VERSION = 92600


@pytest.mark.L0
@pytest.mark.parametrize(
    "op_name, torch_reference",
    [
        ("mod", torch.fmod),  # truncated: result follows the sign of the dividend
        ("floor_mod", torch.remainder),  # floored: result follows the sign of the divisor
    ],
)
def test_pointwise_modulo(op_name, torch_reference, cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend not available")

    if op_name == "floor_mod" and cudnn.backend_version() < FLOOR_MOD_MIN_BACKEND_VERSION:
        pytest.skip("floor_mod (CUDNN_POINTWISE_FLOOR_MOD) requires cuDNN 9.26.0 or newer")

    # A mixed-sign dividend with a positive divisor is exactly where truncated (mod)
    # and floored (floor_mod) remainder disagree, so this genuinely exercises the
    # floored semantics rather than a case both modes share. Integer-valued fp32
    # keeps a - trunc/floor(a / b) * b exact, so no tolerance slack is needed.
    a_gpu = torch.tensor([-7.0, -3.0, -1.0, 2.0, 5.0, 8.0, -6.0, 9.0], device="cuda", dtype=torch.float32).reshape(2, 4)
    b_gpu = torch.full_like(a_gpu, 4.0)

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )

    a = graph.tensor_like(a_gpu)
    b = graph.tensor_like(b_gpu)

    c = getattr(graph, op_name)(a, b)
    c.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()

    try:
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"TEST WAIVED: {op_name} not supported on this backend/config. {e}")

    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

    c_expected = torch_reference(a_gpu, b_gpu)
    c_actual = torch.zeros_like(c_expected)

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute({a: a_gpu, b: b_gpu, c: c_actual}, workspace, handle=cudnn_handle)

    torch.cuda.synchronize()
    torch.testing.assert_close(c_expected, c_actual)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
