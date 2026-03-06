"""
Test SM90 Prefill SDPA: cuDNN Graph API (HeurMode A) vs OSS Engine (HeurMode OPENSOURCE).

Both paths use the same graph API — only the HeurMode_t differs.
Tensors use BSHD physical layout. Output is poisoned before the OSS path
to verify the kernel overwrites all elements.
"""

import math
import pytest
import torch

import cudnn


def _is_hopper():
    if not torch.cuda.is_available():
        return False
    prop = torch.cuda.get_device_properties(0)
    return prop.major == 9


def _poison_tensor(t: torch.Tensor, nan_frac=0.05, inf_frac=0.05, zero_frac=0.05):
    """Sprinkle NaN, Inf, and zeros into a tensor (in-place)."""
    flat = t.view(-1)
    n = flat.numel()
    perm = torch.randperm(n, device=t.device)
    n_nan = int(n * nan_frac)
    n_inf = int(n * inf_frac)
    n_zero = int(n * zero_frac)
    flat[perm[:n_nan]] = float("nan")
    flat[perm[n_nan : n_nan + n_inf]] = float("inf")
    flat[perm[n_nan + n_inf : n_nan + n_inf + n_zero]] = 0.0


def _bshd_strides(b, s, h, d):
    """Return (stride_b, stride_h, stride_s, stride_d) for BSHD physical layout
    expressed in the graph API's (B,H,S,D) dimension order."""
    return [h * s * d, d, h * d, 1]


def _build_sdpa_graph(b, h_q, h_kv, s_q, s_kv, d, attn_scale):
    """Build an SDPA forward graph with max and sum_exp outputs (BSHD layout)."""
    q_bshd = _bshd_strides(b, s_q, h_q, d)
    k_bshd = _bshd_strides(b, s_kv, h_kv, d)
    o_bshd = _bshd_strides(b, s_q, h_q, d)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q_t = graph.tensor(name="Q", dim=[b, h_q, s_q, d], stride=q_bshd, data_type=cudnn.data_type.BFLOAT16)
    k_t = graph.tensor(name="K", dim=[b, h_kv, s_kv, d], stride=k_bshd, data_type=cudnn.data_type.BFLOAT16)
    v_t = graph.tensor(name="V", dim=[b, h_kv, s_kv, d], stride=k_bshd, data_type=cudnn.data_type.BFLOAT16)

    max_t = graph.tensor(name="max", dim=[b, h_q, s_q, 1], stride=[h_q * s_q, s_q, 1, 1], data_type=cudnn.data_type.FLOAT)
    se_t = graph.tensor(name="sum_exp", dim=[b, h_q, s_q, 1], stride=[h_q * s_q, s_q, 1, 1], data_type=cudnn.data_type.FLOAT)

    o_t, _ = graph.sdpa(
        q=q_t,
        k=k_t,
        v=v_t,
        attn_scale=attn_scale,
        use_causal_mask=True,
        generate_stats=False,
        score_max=max_t,
        score_sum_exp=se_t,
    )
    o_t.set_output(True).set_dim([b, h_q, s_q, d]).set_stride(o_bshd)

    return graph, q_t, k_t, v_t, o_t, max_t, se_t


@pytest.mark.skipif(not _is_hopper(), reason="Requires SM90 (Hopper)")
def test_sm90_prefill_oss_engine_vs_cudnn_graph():
    # ---- Config (matches C++ sample) ----
    b, h_q, h_kv, s_q, s_kv, d = 2, 4, 2, 1024, 2048, 128
    attn_scale = 1.0 / math.sqrt(d)

    # ---- Allocate BSHD-contiguous tensors ----
    q_gpu = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device="cuda")
    k_gpu = torch.randn(b, s_kv, h_kv, d, dtype=torch.bfloat16, device="cuda")
    v_gpu = torch.randn(b, s_kv, h_kv, d, dtype=torch.bfloat16, device="cuda")

    # ================================================================
    # Path 1: cuDNN Graph API (reference, HeurMode A)
    # ================================================================
    o_cudnn = torch.empty(b, s_q, h_q, d, dtype=torch.bfloat16, device="cuda")
    max_cudnn = torch.empty(b, h_q, s_q, dtype=torch.float32, device="cuda")
    se_cudnn = torch.empty(b, h_q, s_q, dtype=torch.float32, device="cuda")

    graph, q_t, k_t, v_t, o_t, max_t, se_t = _build_sdpa_graph(b, h_q, h_kv, s_q, s_kv, d, attn_scale)
    graph.validate()
    graph.build_operation_graph()

    try:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"cuDNN graph not supported: {e}")

    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

    variant_pack = {q_t: q_gpu, k_t: k_gpu, v_t: v_gpu, o_t: o_cudnn, max_t: max_cudnn, se_t: se_cudnn}
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()

    # ================================================================
    # Path 2: OSS Engine via Graph API (HeurMode OPENSOURCE)
    # ================================================================
    o_oss = torch.empty(b, s_q, h_q, d, dtype=torch.bfloat16, device="cuda")
    max_oss = torch.empty(b, h_q, s_q, dtype=torch.float32, device="cuda")
    se_oss = torch.empty(b, h_q, s_q, dtype=torch.float32, device="cuda")

    # Poison output tensors
    _poison_tensor(o_oss)
    _poison_tensor(max_oss)
    _poison_tensor(se_oss)

    graph2, q_t2, k_t2, v_t2, o_t2, max_t2, se_t2 = _build_sdpa_graph(b, h_q, h_kv, s_q, s_kv, d, attn_scale)
    graph2.validate()
    graph2.build_operation_graph()
    graph2.create_execution_plans([cudnn.heur_mode.OPENSOURCE])
    graph2.check_support()
    graph2.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

    variant_pack2 = {q_t2: q_gpu, k_t2: k_gpu, v_t2: v_gpu, o_t2: o_oss, max_t2: max_oss, se_t2: se_oss}
    workspace2 = torch.empty(graph2.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph2.execute(variant_pack2, workspace2)
    torch.cuda.synchronize()

    # ================================================================
    # Verification
    # ================================================================
    # 1. No NaN/Inf in output (proves kernel overwrote poisoned elements)
    assert not torch.isnan(o_oss).any(), f"Output O has {torch.isnan(o_oss).sum().item()} NaN values"
    assert not torch.isnan(max_oss).any(), "max output has NaN"
    assert not torch.isnan(se_oss).any(), "sum_exp output has NaN"
    assert not torch.isinf(o_oss).any(), "Output O has Inf values"

    # 2. Compare against cuDNN reference
    atol, rtol = 5e-2, 5e-2
    mismatches = ~torch.isclose(o_oss.float(), o_cudnn.float(), atol=atol, rtol=rtol)
    num_mismatches = mismatches.sum().item()
    total = o_oss.numel()
    max_diff = (o_oss.float() - o_cudnn.float()).abs().max().item()
    mean_diff = (o_oss.float() - o_cudnn.float()).abs().mean().item()

    print(f"\n===== SM90 Prefill Engine vs cuDNN Graph (Python) =====")
    print(f"  Total elements:  {total}")
    print(f"  Max abs diff:    {max_diff:.6e}")
    print(f"  Mean abs diff:   {mean_diff:.6e}")
    print(f"  Mismatches:      {num_mismatches} / {total} (atol={atol}, rtol={rtol})")
    print(f"=======================================================\n")

    assert num_mismatches == 0, f"{num_mismatches}/{total} mismatches (max_diff={max_diff:.6e})"


if __name__ == "__main__":
    test_sm90_prefill_oss_engine_vs_cudnn_graph()
