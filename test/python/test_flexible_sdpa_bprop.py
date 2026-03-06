import cudnn
import torch
import pytest
from functools import partial
import math

from test_utils import torch_fork_set_rng
from looseversion import LooseVersion


def causal_mask(sdpa_graph, q_kt_tensor, neg_inf):
    """Forward causal mask: mask out future positions with -inf."""
    row_index = sdpa_graph.gen_index(input=q_kt_tensor, axis=2)
    row_index.set_data_type(cudnn.data_type.INT32)

    col_index = sdpa_graph.gen_index(input=q_kt_tensor, axis=3)
    col_index.set_data_type(cudnn.data_type.INT32)

    mask = sdpa_graph.cmp_ge(input=row_index, comparison=col_index, compute_data_type=cudnn.data_type.BOOLEAN)
    mask.set_data_type(cudnn.data_type.BOOLEAN)

    out = sdpa_graph.binary_select(input0=q_kt_tensor, input1=neg_inf, mask=mask)

    return out


def causal_mask_bprop(sdpa_graph, dP_tensor, zero_tensor):
    """Backward causal mask: zero out gradients at masked positions."""
    row_index = sdpa_graph.gen_index(input=dP_tensor, axis=2)
    row_index.set_data_type(cudnn.data_type.INT32)

    col_index = sdpa_graph.gen_index(input=dP_tensor, axis=3)
    col_index.set_data_type(cudnn.data_type.INT32)

    mask = sdpa_graph.cmp_ge(input=row_index, comparison=col_index, compute_data_type=cudnn.data_type.BOOLEAN)
    mask.set_data_type(cudnn.data_type.BOOLEAN)

    out = sdpa_graph.binary_select(input0=dP_tensor, input1=zero_tensor, mask=mask)

    return out


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_bprop_with_causal_mask(cudnn_handle):
    """Test SDPA backward pass with causal mask score modifier."""

    b = 3  # batch size
    h_q = 4  # query number of heads
    h_k = 4  # key number of heads
    h_v = 4  # value number of heads
    s_q = 256  # maximum sequence length
    s_kv = 256  # maximum sequence length
    d = 128  # embedding dimension per head

    attn_scale = 1.0 / math.sqrt(d)

    neg_inf_scalar_value = -1e9
    zero_scalar_value = 0.0

    q_dims = (b, h_q, s_q, d)
    q_strides = (s_q * h_q * d, d, h_q * d, 1)
    k_dims = (b, h_k, s_kv, d)
    k_strides = (s_kv * h_k * d, d, h_k * d, 1)
    v_dims = (b, h_v, s_kv, d)
    v_strides = (s_kv * h_v * d, d, h_v * d, 1)

    q_gpu = torch.randn(b * s_q * h_q * d).half().cuda().as_strided(q_dims, q_strides)
    k_gpu = torch.randn(b * s_kv * h_k * d).half().cuda().as_strided(k_dims, k_strides)
    v_gpu = torch.randn(b * s_kv * h_v * d).half().cuda().as_strided(v_dims, v_strides)
    o_gpu = torch.empty(b * s_q * h_q * d).half().cuda().as_strided(q_dims, q_strides)

    cudnn_version = LooseVersion(cudnn.backend_version_string())

    if cudnn_version < "9.6.0":
        pytest.skip("SDPA with flexible graphs requires cudnn 9.6.0 or higher")

    # ========================
    # Forward pass
    # ========================
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu)
    v = graph.tensor_like(v_gpu)

    neg_inf_tensor_cpu = torch.full((1, 1, 1, 1), neg_inf_scalar_value)
    neg_inf_tensor = graph.tensor(
        name="neg_inf_scalar",
        dim=neg_inf_tensor_cpu.size(),
        stride=neg_inf_tensor_cpu.stride(),
        is_pass_by_value=True,
        data_type=neg_inf_tensor_cpu.dtype,
    )

    o, stats = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=True,
        attn_scale=attn_scale,
        use_causal_mask=False,
        score_mod=partial(causal_mask, neg_inf=neg_inf_tensor),
    )

    o.set_output(True).set_dim(q_dims).set_stride(q_strides)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()

    try:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"TEST WAIVED: unsupported graph. {e}")
        pytest.skip("TEST WAIVED: unsupported forward graph.")

    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

    stats_gpu = torch.empty(b, h_q, s_q, 1, dtype=torch.float32, device="cuda")

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        stats: stats_gpu,
        neg_inf_tensor: neg_inf_tensor_cpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()

    # ========================
    # Backward pass
    # ========================
    dO_gpu = torch.randn_like(o_gpu)
    dQ_gpu = torch.empty_like(q_gpu)
    dK_gpu = torch.empty_like(k_gpu)
    dV_gpu = torch.empty_like(v_gpu)

    bwd_graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q_bwd = bwd_graph.tensor_like(q_gpu)
    k_bwd = bwd_graph.tensor_like(k_gpu)
    v_bwd = bwd_graph.tensor_like(v_gpu)

    o_bwd = bwd_graph.tensor_like(o_gpu)
    dO_bwd = bwd_graph.tensor_like(dO_gpu)
    stats_bwd = bwd_graph.tensor_like(stats_gpu)

    neg_inf_tensor_bwd_cpu = torch.full((1, 1, 1, 1), neg_inf_scalar_value)
    neg_inf_tensor_bwd = bwd_graph.tensor(
        name="neg_inf_scalar",
        dim=neg_inf_tensor_bwd_cpu.size(),
        stride=neg_inf_tensor_bwd_cpu.stride(),
        is_pass_by_value=True,
        data_type=neg_inf_tensor_bwd_cpu.dtype,
    )

    zero_tensor_cpu = torch.full((1, 1, 1, 1), zero_scalar_value)
    zero_tensor = bwd_graph.tensor(
        name="zero_scalar",
        dim=zero_tensor_cpu.size(),
        stride=zero_tensor_cpu.stride(),
        is_pass_by_value=True,
        data_type=zero_tensor_cpu.dtype,
    )

    dQ, dK, dV = bwd_graph.sdpa_backward(
        name="sdpa_backward",
        q=q_bwd,
        k=k_bwd,
        v=v_bwd,
        o=o_bwd,
        dO=dO_bwd,
        stats=stats_bwd,
        attn_scale=attn_scale,
        use_causal_mask=False,
        score_mod=partial(causal_mask, neg_inf=neg_inf_tensor_bwd),
        score_mod_bprop=partial(causal_mask_bprop, zero_tensor=zero_tensor),
    )

    dQ.set_output(True).set_dim(q_dims).set_stride(q_strides)
    dK.set_output(True).set_dim(k_dims).set_stride(k_strides)
    dV.set_output(True).set_dim(v_dims).set_stride(v_strides)

    bwd_graph.validate()
    bwd_graph.build_operation_graph()

    try:
        bwd_graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        bwd_graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"TEST WAIVED: unsupported graph. {e}")
        pytest.skip("TEST WAIVED: unsupported backward graph.")

    bwd_graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

    bwd_variant_pack = {
        q_bwd: q_gpu,
        k_bwd: k_gpu,
        v_bwd: v_gpu,
        o_bwd: o_gpu,
        dO_bwd: dO_gpu,
        stats_bwd: stats_gpu,
        dQ: dQ_gpu,
        dK: dK_gpu,
        dV: dV_gpu,
        neg_inf_tensor_bwd: neg_inf_tensor_bwd_cpu,
        zero_tensor: zero_tensor_cpu,
    }

    bwd_workspace = torch.empty(bwd_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    bwd_graph.execute(bwd_variant_pack, bwd_workspace)
    torch.cuda.synchronize()
