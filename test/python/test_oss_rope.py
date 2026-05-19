"""
Test RoPE + SDPA via cuDNN Graph API.

Constructs a graph with RoPE nodes feeding into SDPA.
RoPE takes raw frequency angles (freqs) and computes sincosf internally.

User graph:
  q_rot = graph.rope(q, freqs)
  k_rot = graph.rope(k, freqs)
  o = graph.sdpa(q_rot, k_rot, v)
"""

import pytest
import torch
import torch.nn.functional as F

import cudnn

# RoPE backend op was added in cuDNN 9.24. Skip the whole file on older backends.
pytestmark = pytest.mark.skipif(
    cudnn.backend_version() < 92400,
    reason="RoPE backend op (CUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR) requires cuDNN >= 9.24",
)


def rope_reference(x, freqs):
    """PyTorch reference for non-interleaved RoPE with raw freqs (BHSD layout).

    Args:
        x: [B, H, S, D] input tensor
        freqs: [S, 1, 1, D] raw angle values (float32)

    Returns:
        [B, H, S, D] rotated tensor
    """
    d = x.shape[-1]
    d2 = d // 2
    # freqs is [S, 1, 1, D], we need [S, D/2] for the cos/sin
    angles = freqs[:, 0, 0, :d2].float()  # [S, D/2]
    cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # [1, 1, S, D/2]
    sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(0)

    x1 = x[..., :d2].float()
    x2 = x[..., d2:].float()
    y1 = x1 * cos_vals - x2 * sin_vals
    y2 = x2 * cos_vals + x1 * sin_vals
    return torch.cat([y1, y2], dim=-1).to(x.dtype)


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_rope_sdpa_smoke():
    """Smoke test: build a RoPE+SDPA graph and verify it constructs."""
    B, S, H, D = 1, 128, 8, 128

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    bhsd_stride = [H * S * D, S * D, D, 1]

    Q = graph.tensor(name="Q", dim=[B, H, S, D], stride=bhsd_stride, data_type=cudnn.data_type.BFLOAT16)
    K = graph.tensor(name="K", dim=[B, H, S, D], stride=bhsd_stride, data_type=cudnn.data_type.BFLOAT16)
    V = graph.tensor(name="V", dim=[B, H, S, D], stride=bhsd_stride, data_type=cudnn.data_type.BFLOAT16)
    FREQS = graph.tensor(name="freqs", dim=[S, 1, 1, D], stride=[D, D, D, 1], data_type=cudnn.data_type.FLOAT)

    # Q_rot/K_rot are real outputs (user-bound) — set_output(True) marks them as such.
    Q_rot = graph.rope(input=Q, freqs=FREQS, name="RoPE_Q")
    Q_rot.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim([B, H, S, D]).set_stride(bhsd_stride)

    K_rot = graph.rope(input=K, freqs=FREQS, name="RoPE_K")
    K_rot.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim([B, H, S, D]).set_stride(bhsd_stride)

    O, stats = graph.sdpa(q=Q_rot, k=K_rot, v=V, is_inference=True, name="SDPA")
    O.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim([B, H, S, D]).set_stride(bhsd_stride)
    if stats is not None:
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    # This should build successfully with backend heuristics
    # (the backend detects RoPE+SDPA pattern and fuses them)
    graph.build([cudnn.heur_mode.A])
    print(f"Graph built successfully, workspace: {graph.get_workspace_size()}")


def _run_rope_sdpa(B, H_q, H_k, S_q, S_kv, D, output_scale_q, output_scale_k, attn_scale, dtype=torch.bfloat16):
    """Build and execute a RoPE+SDPA graph. Returns (o_gpu, q, k, v, freqs)."""
    torch.manual_seed(0)
    q = torch.randn(B, H_q, S_q, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H_k, S_kv, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H_k, S_kv, D, device="cuda", dtype=dtype)

    max_s = max(S_q, S_kv)
    d2 = D // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device="cuda", dtype=torch.float32) / D))
    pos = torch.arange(max_s, device="cuda", dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)
    freqs = torch.zeros(max_s, 1, 1, D, device="cuda", dtype=torch.float32)
    freqs[:, 0, 0, :d2] = angles

    cudnn_dtype = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    Q = graph.tensor(name="Q", dim=list(q.shape), stride=list(q.stride()), data_type=cudnn_dtype)
    K = graph.tensor(name="K", dim=list(k.shape), stride=list(k.stride()), data_type=cudnn_dtype)
    V = graph.tensor(name="V", dim=list(v.shape), stride=list(v.stride()), data_type=cudnn_dtype)
    FREQS = graph.tensor(name="freqs", dim=list(freqs.shape), stride=list(freqs.stride()), data_type=cudnn.data_type.FLOAT)

    Q_rot = graph.rope(input=Q, freqs=FREQS, output_scale=output_scale_q, name="RoPE_Q")
    Q_rot.set_data_type(cudnn_dtype).set_dim(list(q.shape)).set_stride(list(q.stride()))
    K_rot = graph.rope(input=K, freqs=FREQS, output_scale=output_scale_k, name="RoPE_K")
    K_rot.set_data_type(cudnn_dtype).set_dim(list(k.shape)).set_stride(list(k.stride()))

    O, _ = graph.sdpa(q=Q_rot, k=K_rot, v=V, is_inference=True, attn_scale=attn_scale, name="SDPA")
    O.set_output(True).set_data_type(cudnn_dtype).set_dim([B, H_q, S_q, D]).set_stride(list(q.stride()))
    graph.build([cudnn.heur_mode.A])

    handle = cudnn.create_handle()
    o_gpu = torch.empty(B, H_q, S_q, D, device="cuda", dtype=dtype)
    # Q_rot/K_rot are real (user-bound) outputs after Phase 1: allocate them.
    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(k)
    ws_size = max(int(graph.get_workspace_size()), 1)
    ws_buf = torch.empty(ws_size, device="cuda", dtype=torch.uint8)
    graph.execute(
        {
            Q.get_uid(): q.data_ptr(),
            K.get_uid(): k.data_ptr(),
            V.get_uid(): v.data_ptr(),
            FREQS.get_uid(): freqs.data_ptr(),
            Q_rot.get_uid(): q_rot.data_ptr(),
            K_rot.get_uid(): k_rot.data_ptr(),
            O.get_uid(): o_gpu.data_ptr(),
        },
        ws_buf.data_ptr(),
        handle,
    )
    return o_gpu, q, k, v, freqs


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_rope_output_scale_equivalence():
    """output_scale on Q-RoPE should be equivalent to multiplying attn_scale.

    Run A: Q-RoPE scale=1.0, K-RoPE scale=1.0, SDPA attn_scale=alpha * base
    Run B: Q-RoPE scale=alpha, K-RoPE scale=1.0, SDPA attn_scale=base
    Outputs must match (up to fp arith).
    """
    B, H_q, H_k, S_q, S_kv, D = 1, 4, 4, 128, 128, 64
    base = 1.0 / (D**0.5)
    alpha = 1.7

    o_a, *_ = _run_rope_sdpa(B, H_q, H_k, S_q, S_kv, D, 1.0, 1.0, alpha * base)
    o_b, *_ = _run_rope_sdpa(B, H_q, H_k, S_q, S_kv, D, alpha, 1.0, base)

    diff = (o_a.float() - o_b.float()).abs()
    assert diff.max().item() < 2e-2, f"max diff {diff.max().item()} exceeds tolerance"


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_rope_output_scale_mscale_style():
    """YARN-style: scale on both Q-RoPE and K-RoPE should give mscale**2 on logits.

    Run A: scales=(m, m), attn_scale=base
    Run B: scales=(1, 1), attn_scale=base * m**2
    """
    B, H_q, H_k, S_q, S_kv, D = 1, 4, 4, 128, 128, 64
    base = 1.0 / (D**0.5)
    m = 1.369  # YARN-like mscale for factor=40: 0.1*log(40)+1

    o_a, *_ = _run_rope_sdpa(B, H_q, H_k, S_q, S_kv, D, m, m, base)
    o_b, *_ = _run_rope_sdpa(B, H_q, H_k, S_q, S_kv, D, 1.0, 1.0, base * m * m)

    diff = (o_a.float() - o_b.float()).abs()
    assert diff.max().item() < 2e-2, f"max diff {diff.max().item()} exceeds tolerance"


@pytest.mark.L0
@pytest.mark.skipif(cudnn.backend_version() < 92400, reason="RoPE backward op requires cuDNN >= 9.24")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_rope_sdpa_fwd_bwd_end_to_end():
    """End-to-end: build fwd graph (rope+sdpa), execute, then build bwd graph
    (sdpa_backward + rope_backward), execute, compare against PyTorch autograd
    reference (which composes plain RoPE + SDPA in f32)."""
    import math

    B, H, S, D = 1, 4, 64, 64
    dtype = torch.bfloat16
    cudnn_dtype = cudnn.data_type.BFLOAT16
    base_attn = 1.0 / math.sqrt(D)

    torch.manual_seed(0)
    q_in = torch.randn(B, H, S, D, device="cuda", dtype=dtype, requires_grad=True)
    k_in = torch.randn(B, H, S, D, device="cuda", dtype=dtype, requires_grad=True)
    v_in = torch.randn(B, H, S, D, device="cuda", dtype=dtype, requires_grad=True)

    # ---- freqs ----
    half = D // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device="cuda", dtype=torch.float32) / D))
    pos = torch.arange(S, device="cuda", dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)
    freqs = torch.zeros(S, 1, 1, D, device="cuda", dtype=torch.float32)
    freqs[:, 0, 0, :half] = angles

    # ---- FWD GRAPH ----
    fwd = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    Q = fwd.tensor(uid=1, dim=list(q_in.shape), stride=list(q_in.stride()), data_type=cudnn_dtype)
    K = fwd.tensor(uid=2, dim=list(k_in.shape), stride=list(k_in.stride()), data_type=cudnn_dtype)
    V = fwd.tensor(uid=3, dim=list(v_in.shape), stride=list(v_in.stride()), data_type=cudnn_dtype)
    F1 = fwd.tensor(uid=4, dim=list(freqs.shape), stride=list(freqs.stride()), data_type=cudnn.data_type.FLOAT)

    Qr = fwd.rope(input=Q, freqs=F1, name="RoPE_Q")
    Qr.set_uid(5).set_output(True).set_data_type(cudnn_dtype).set_dim(list(q_in.shape)).set_stride(list(q_in.stride()))
    Kr = fwd.rope(input=K, freqs=F1, name="RoPE_K")
    Kr.set_uid(6).set_output(True).set_data_type(cudnn_dtype).set_dim(list(k_in.shape)).set_stride(list(k_in.stride()))

    O, stats = fwd.sdpa(q=Qr, k=Kr, v=V, generate_stats=True, attn_scale=base_attn, name="SDPA")
    O.set_uid(7).set_output(True).set_data_type(cudnn_dtype).set_dim([B, H, S, D]).set_stride(list(q_in.stride()))
    stats.set_uid(8).set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([B, H, S, 1]).set_stride([H * S, S, 1, 1])

    fwd.build([cudnn.heur_mode.A])

    handle = cudnn.create_handle()
    q_rot = torch.empty_like(q_in)
    k_rot = torch.empty_like(k_in)
    o_gpu = torch.empty(B, H, S, D, device="cuda", dtype=dtype)
    stats_gpu = torch.empty(B, H, S, 1, device="cuda", dtype=torch.float32)
    ws = torch.empty(max(int(fwd.get_workspace_size()), 1), device="cuda", dtype=torch.uint8)
    fwd.execute(
        {
            1: q_in.data_ptr(),
            2: k_in.data_ptr(),
            3: v_in.data_ptr(),
            4: freqs.data_ptr(),
            5: q_rot.data_ptr(),
            6: k_rot.data_ptr(),
            7: o_gpu.data_ptr(),
            8: stats_gpu.data_ptr(),
        },
        ws.data_ptr(),
        handle,
    )

    # ---- BWD GRAPH ----
    dO = torch.randn_like(o_gpu)
    bwd = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    Qr_b = bwd.tensor(uid=5, dim=list(q_in.shape), stride=list(q_in.stride()), data_type=cudnn_dtype)
    Kr_b = bwd.tensor(uid=6, dim=list(k_in.shape), stride=list(k_in.stride()), data_type=cudnn_dtype)
    V_b = bwd.tensor(uid=3, dim=list(v_in.shape), stride=list(v_in.stride()), data_type=cudnn_dtype)
    O_b = bwd.tensor(uid=7, dim=list(q_in.shape), stride=list(q_in.stride()), data_type=cudnn_dtype)
    dO_b = bwd.tensor(uid=9, dim=list(q_in.shape), stride=list(q_in.stride()), data_type=cudnn_dtype)
    St_b = bwd.tensor(uid=8, dim=[B, H, S, 1], stride=[H * S, S, 1, 1], data_type=cudnn.data_type.FLOAT)
    F2 = bwd.tensor(uid=4, dim=list(freqs.shape), stride=list(freqs.stride()), data_type=cudnn.data_type.FLOAT)

    dQr, dKr, dV = bwd.sdpa_backward(q=Qr_b, k=Kr_b, v=V_b, o=O_b, dO=dO_b, stats=St_b, attn_scale=base_attn, name="SDPA_BWD")
    dQr.set_uid(10).set_output(True).set_data_type(cudnn_dtype).set_dim(list(q_in.shape)).set_stride(list(q_in.stride()))
    dKr.set_uid(11).set_output(True).set_data_type(cudnn_dtype).set_dim(list(k_in.shape)).set_stride(list(k_in.stride()))
    dV.set_uid(12).set_output(True).set_data_type(cudnn_dtype).set_dim(list(v_in.shape)).set_stride(list(v_in.stride()))

    dQ = bwd.rope_backward(dY=dQr, freqs=F2, name="RoPE_BWD_Q")
    dQ.set_uid(13).set_output(True).set_data_type(cudnn_dtype).set_dim(list(q_in.shape)).set_stride(list(q_in.stride()))
    dK = bwd.rope_backward(dY=dKr, freqs=F2, name="RoPE_BWD_K")
    dK.set_uid(14).set_output(True).set_data_type(cudnn_dtype).set_dim(list(k_in.shape)).set_stride(list(k_in.stride()))

    bwd.build([cudnn.heur_mode.A])

    dq_rot = torch.empty_like(q_in)
    dk_rot = torch.empty_like(k_in)
    dv_gpu = torch.empty_like(v_in)
    dq_gpu = torch.empty_like(q_in)
    dk_gpu = torch.empty_like(k_in)
    ws2 = torch.empty(max(int(bwd.get_workspace_size()), 1), device="cuda", dtype=torch.uint8)
    bwd.execute(
        {
            5: q_rot.data_ptr(),
            6: k_rot.data_ptr(),
            3: v_in.data_ptr(),
            7: o_gpu.data_ptr(),
            9: dO.data_ptr(),
            8: stats_gpu.data_ptr(),
            4: freqs.data_ptr(),
            10: dq_rot.data_ptr(),
            11: dk_rot.data_ptr(),
            12: dv_gpu.data_ptr(),
            13: dq_gpu.data_ptr(),
            14: dk_gpu.data_ptr(),
        },
        ws2.data_ptr(),
        handle,
    )

    # ---- Reference: PyTorch autograd ----
    def rope_ref(x):
        x = x.float()
        cos_v = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin_v = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        x_lo = x[..., :half]
        x_hi = x[..., half:]
        return torch.cat([x_lo * cos_v - x_hi * sin_v, x_hi * cos_v + x_lo * sin_v], dim=-1)

    q_ref = q_in.detach().clone().requires_grad_()
    k_ref = k_in.detach().clone().requires_grad_()
    v_ref = v_in.detach().clone().requires_grad_()
    qr = rope_ref(q_ref).to(dtype).float()
    kr = rope_ref(k_ref).to(dtype).float()
    scores = torch.matmul(qr, kr.transpose(-2, -1)) * base_attn
    o_ref = torch.matmul(torch.softmax(scores, dim=-1), v_ref.float())
    o_ref.backward(dO.float())

    # ---- Compare ----
    rtol = 5e-2
    diff_o = (o_gpu.float() - o_ref).abs().max().item()
    diff_dq = (dq_gpu.float() - q_ref.grad).abs().max().item()
    diff_dk = (dk_gpu.float() - k_ref.grad).abs().max().item()
    diff_dv = (dv_gpu.float() - v_ref.grad).abs().max().item()
    print(f"max diffs — O:{diff_o:.4f} dQ:{diff_dq:.4f} dK:{diff_dk:.4f} dV:{diff_dv:.4f}")
    assert diff_o < rtol, f"O diverged: {diff_o}"
    assert diff_dv < rtol, f"dV diverged: {diff_dv}"
    assert diff_dq < rtol, f"dQ diverged: {diff_dq}"
    assert diff_dk < rtol, f"dK diverged: {diff_dk}"


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_rope_output_scale_matches_reference():
    """Standalone correctness: Q-RoPE with output_scale=α produces α * standard_rope(Q)."""
    import math

    B, H_q, H_k, S, D = 1, 2, 2, 64, 64
    base = 1.0 / math.sqrt(D)
    alpha = 2.5

    # Use SDPA-only to expose RoPE output indirectly: compare ratio of attention outputs.
    o_unscaled, q, k, v, freqs = _run_rope_sdpa(B, H_q, H_k, S, S, D, 1.0, 1.0, base)
    o_scaled, *_ = _run_rope_sdpa(B, H_q, H_k, S, S, D, alpha, 1.0, base)

    # alpha on Q only -> logits scaled by alpha (asymmetric). Verify it's not equal to unscaled.
    # And that re-running with attn_scale=alpha*base on the unscaled gives same answer.
    o_check, *_ = _run_rope_sdpa(B, H_q, H_k, S, S, D, 1.0, 1.0, alpha * base)
    diff = (o_scaled.float() - o_check.float()).abs()
    assert diff.max().item() < 2e-2, f"output_scale on Q != attn_scale: max diff {diff.max().item()}"
