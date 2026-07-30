# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU parity: pygraph builds natively, lowers to cuDNN, executes correctly.

Covers the native -> _lower_to_cpp -> cuDNN execute path (uid propagation, handle
threading, pointwise dispatch). Skipped without a GPU / cuDNN.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

import cudnn
from cudnn._pygraph import pygraph

pytestmark = pytest.mark.L0

M, K, N = 64, 32, 48


def _handle():
    return cudnn.create_handle()


def _assert_ran_on_backend(g):
    """Dispatch-level proof the execution took the cuDNN backend plan path:
    the selected routed plan is the backend entry (no python engine), the graph
    was really lowered, and backend plans were created and built. Kernel
    identity below the backend API is deliberately not asserted (kernel names
    are backend-internal and version-dependent)."""
    assert g.selected_engine is None
    assert g._lowered_graph is not None
    assert g._cpp_plans_created and g._is_built


def test_native_matmul_lowers_to_backend():
    h = _handle()
    a = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(1, K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(1, M, N, device="cuda", dtype=torch.float16)

    g = pygraph(handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.HALF)
    B = g.tensor(dim=[1, K, N], stride=[K * N, N, 1], data_type=cudnn.data_type.HALF)
    C = g.matmul(A, B)
    C.set_output(True).set_data_type(cudnn.data_type.HALF)

    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, C: c}, ws, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g)

    torch.testing.assert_close(c.float(), a.float() @ b.float(), atol=2e-2, rtol=2e-2)


def test_native_matmul_bias_relu_lowers_to_backend():
    h = _handle()
    a = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(1, K, N, device="cuda", dtype=torch.float16)
    bias = torch.randn(1, M, N, device="cuda", dtype=torch.float16)
    c = torch.empty(1, M, N, device="cuda", dtype=torch.float16)

    g = pygraph(handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.HALF)
    B = g.tensor(dim=[1, K, N], stride=[K * N, N, 1], data_type=cudnn.data_type.HALF)
    Bi = g.tensor(dim=[1, M, N], stride=[M * N, N, 1], data_type=cudnn.data_type.HALF)
    Y = g.relu(g.bias(g.matmul(A, B), Bi))
    Y.set_output(True).set_data_type(cudnn.data_type.HALF)

    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, Bi: bias, Y: c}, ws, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g)

    torch.testing.assert_close(c.float(), torch.relu(a.float() @ b.float() + bias.float()), atol=2e-2, rtol=2e-2)


def test_native_matmul_reduction_lowers_to_backend():
    """matmul -> reduction(ADD) over N; cuDNN needs explicit reduced output dims."""
    h = _handle()
    a = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(1, K, N, device="cuda", dtype=torch.float16)
    r = torch.empty(1, M, 1, device="cuda", dtype=torch.float32)

    g = pygraph(handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.HALF)
    B = g.tensor(dim=[1, K, N], stride=[K * N, N, 1], data_type=cudnn.data_type.HALF)
    R = g.reduction(g.matmul(A, B), mode=cudnn.reduction_mode.ADD, out_dims=[1, M, 1])
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, R: r}, ws, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g)

    torch.testing.assert_close(r, (a.float() @ b.float()).sum(dim=2, keepdim=True), atol=5e-2, rtol=5e-2)


def test_native_block_scale_nvfp4_lowers_to_backend():
    """block_scale_dequantize(A)@block_scale_dequantize(B), nvfp4 -> cuDNN (SM100)."""
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("torch lacks float4_e2m1fn_x2")
    if torch.cuda.get_device_properties(0).major < 10:
        pytest.skip("block-scale MMA needs SM100+")

    h = _handle()
    b, Mb, Nb, Kb, BS = 1, 128, 128, 64, 16
    A = torch.randint(0, 256, (b, Mb, Kb // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    B = torch.randint(0, 256, (b, Kb, Nb // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    k_scale = ((Kb + BS - 1) // BS + 3) // 4 * 4
    A_ds = torch.full((b, 128, k_scale), 1.0, dtype=torch.float8_e4m3fn, device="cuda")
    B_ds = torch.full((b, k_scale, 128), 1.0, dtype=torch.float8_e4m3fn, device="cuda")
    C = torch.empty((b, Mb, Nb), dtype=torch.bfloat16, device="cuda")

    g = pygraph(handle=h, compute_data_type=cudnn.data_type.FLOAT)
    At = g.tensor(dim=[b, Mb, Kb], stride=[Mb * Kb, Kb, 1], data_type=cudnn.data_type.FP4_E2M1)
    Bt = g.tensor(dim=[b, Kb, Nb], stride=[Nb * Kb, 1, Kb], data_type=cudnn.data_type.FP4_E2M1)
    Ad = g.tensor(
        dim=[b, 128, k_scale], stride=[128 * k_scale, k_scale, 1], data_type=cudnn.data_type.FP8_E4M3, reordering_type=cudnn.tensor_reordering.F8_128x4
    )
    Bd = g.tensor(
        dim=[b, k_scale, 128], stride=[k_scale * 128, 1, k_scale], data_type=cudnn.data_type.FP8_E4M3, reordering_type=cudnn.tensor_reordering.F8_128x4
    )
    Cc = g.matmul(
        g.block_scale_dequantize(At, Ad, block_size=[1, BS]), g.block_scale_dequantize(Bt, Bd, block_size=[BS, 1]), compute_data_type=cudnn.data_type.FLOAT
    )
    Cc.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    g.build([cudnn.heur_mode.A, cudnn.heur_mode.B])
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({At: A, Bt: B, Ad: A_ds, Bd: B_ds, Cc: C}, ws, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g)  # builds + executes without error (parity harness = repo's fp4 test)


def test_native_moe_grouped_matmul_lowers_to_backend():
    """moe_grouped_matmul (mode=NONE) built natively -> cuDNN, parity vs a
    self-contained per-expert reference."""
    if cudnn.backend_version() < 91500:
        pytest.skip("moe_grouped_matmul requires cuDNN 9.15+")
    h = _handle()
    E, T, Wt, Hd = 8, 256, 64, 128
    fto = [i * (T // E) for i in range(E)]  # one contiguous token chunk per expert

    g = pygraph(handle=h, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tok = g.tensor(dim=[1, T, Hd], stride=[T * Hd, Hd, 1], data_type=cudnn.data_type.BFLOAT16)
    wt = g.tensor(dim=[E, Hd, Wt], stride=[Hd * Wt, 1, Hd], data_type=cudnn.data_type.BFLOAT16)
    off = g.tensor(dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    out = g.moe_grouped_matmul(tok, wt, off, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT)
    out.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    g.build([cudnn.heur_mode.A])
    tok_d = torch.randn(T * Hd, dtype=torch.bfloat16, device="cuda")
    wt_d = torch.randn(E * Hd * Wt, dtype=torch.bfloat16, device="cuda")
    off_d = torch.tensor(fto, dtype=torch.int32, device="cuda")
    out_d = torch.empty(T * Wt, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    g.execute({tok: tok_d, wt: wt_d, off: off_d, out: out_d}, ws, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g)

    # reference: per-expert token-chunk @ weight[e] (weights stored H-contiguous)
    token = tok_d.view(T, Hd).float()
    weight = torch.as_strided(wt_d.float(), (E, Hd, Wt), (Hd * Wt, 1, Hd))
    ref = torch.empty(T, Wt)
    bounds = fto + [T]
    for e in range(E):
        s, en = bounds[e], bounds[e + 1]
        if en > s:
            ref[s:en] = token[s:en] @ weight[e]
    torch.testing.assert_close(out_d.view(T, Wt).float(), ref.cuda(), rtol=5e-2, atol=5e-2)


def test_native_sdpa_fwd_lowers_to_backend():
    """sdpa (captured-op family) -> cuDNN execution parity vs torch SDPA."""
    h = _handle()
    B, Hh, S, D = 2, 4, 128, 64
    q = torch.randn(B, Hh, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, Hh, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, Hh, S, D, device="cuda", dtype=torch.float16)
    o = torch.empty(B, Hh, S, D, device="cuda", dtype=torch.float16)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    g = pygraph(handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    Q = g.tensor(dim=[B, Hh, S, D], stride=list(q.stride()), data_type=cudnn.data_type.HALF)
    K = g.tensor(dim=[B, Hh, S, D], stride=list(k.stride()), data_type=cudnn.data_type.HALF)
    V = g.tensor(dim=[B, Hh, S, D], stride=list(v.stride()), data_type=cudnn.data_type.HALF)
    O, stats = g.sdpa(Q, K, V, is_inference=True, use_causal_mask=True, attn_scale=1.0 / (D**0.5))
    assert stats is None and O.dim == [B, Hh, S, D]
    O.set_output(True).set_data_type(cudnn.data_type.HALF)

    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({Q: q, K: k, V: v, O: o}, ws, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g)

    torch.testing.assert_close(o, ref, atol=5e-2, rtol=5e-2)


def test_native_conv_fprop_lowers_to_backend():
    """conv_fprop (structured-table op) -> cuDNN parity vs torch conv2d (NHWC)."""
    h = _handle()
    x = torch.randn(4, 16, 32, 32, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    w = torch.randn(32, 16, 3, 3, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    ref = torch.nn.functional.conv2d(x, w, padding=[1, 1], stride=[1, 1], dilation=[1, 1])
    y = torch.empty_like(ref).to(memory_format=torch.channels_last)

    g = pygraph(handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    X = g.tensor(dim=list(x.shape), stride=list(x.stride()), data_type=cudnn.data_type.HALF)
    W = g.tensor(dim=list(w.shape), stride=list(w.stride()), data_type=cudnn.data_type.HALF)
    Y = g.conv_fprop(image=X, weight=W, padding=[1, 1], stride=[1, 1], dilation=[1, 1])
    assert Y.dim == list(ref.shape)  # table shape inference
    Y.set_output(True).set_data_type(cudnn.data_type.HALF)

    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({X: x, W: w, Y: y}, ws, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g)

    torch.testing.assert_close(y, ref, atol=5e-2, rtol=5e-2)


def test_native_layernorm_fwd_bwd_lowers_to_backend():
    """layernorm fwd (3 outputs) + layernorm_backward (3 outputs) through the
    generic structured-op lowering, parity vs torch autograd.

    Uses the cuDNN-supported LN config ([N, C, 1, 1] channels_last, i.e. LN over
    the embedding dim) — same as the classic test_layernorm.py."""
    h = _handle()
    Nb, C = 64, 128
    eps = 1e-3

    def cl(t):
        return t.to(memory_format=torch.channels_last)

    x = cl(torch.randn(Nb, C, 1, 1, device="cuda", dtype=torch.float16)).requires_grad_()
    scale = cl(torch.randn(1, C, 1, 1, device="cuda", dtype=torch.float16)).requires_grad_()
    bias = cl(torch.randn(1, C, 1, 1, device="cuda", dtype=torch.float16)).requires_grad_()
    eps_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32)

    # torch reference (normalize over all non-batch dims)
    xf = x.float()
    mean_ref = xf.mean(dim=(1, 2, 3), keepdim=True)
    inv_ref = torch.rsqrt(xf.var(dim=(1, 2, 3), keepdim=True, unbiased=False) + eps)
    Y_ref = (xf - mean_ref) * inv_ref * scale.float() + bias.float()
    grad = torch.randn_like(Y_ref)
    Y_ref.backward(grad)

    cl_stride = [C, 1, C, C]  # channels_last for [*, C, 1, 1]

    # ---- forward ----
    g = pygraph(handle=h, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    X = g.tensor(dim=[Nb, C, 1, 1], stride=cl_stride, data_type=cudnn.data_type.HALF)
    S = g.tensor(dim=[1, C, 1, 1], stride=cl_stride, data_type=cudnn.data_type.HALF)
    Bi = g.tensor(dim=[1, C, 1, 1], stride=cl_stride, data_type=cudnn.data_type.HALF)
    E = g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT, is_pass_by_value=True)
    Y, mean, iv = g.layernorm(norm_forward_phase=cudnn.norm_forward_phase.TRAINING, input=X, scale=S, bias=Bi, epsilon=E)
    Y.set_output(True).set_data_type(cudnn.data_type.HALF)
    mean.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    iv.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    Yb = cl(torch.empty(Nb, C, 1, 1, device="cuda", dtype=torch.float16))
    mb = torch.empty(Nb, 1, 1, 1, device="cuda", dtype=torch.float32)
    ivb = torch.empty(Nb, 1, 1, 1, device="cuda", dtype=torch.float32)
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({X: x.detach(), S: scale.detach(), Bi: bias.detach(), E: eps_cpu, Y: Yb, mean: mb, iv: ivb}, ws, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g)
    torch.testing.assert_close(Yb.float(), Y_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(mb, mean_ref, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(ivb, inv_ref, atol=5e-3, rtol=5e-3)

    # ---- backward ----
    g2 = pygraph(handle=h, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    DY = g2.tensor(dim=[Nb, C, 1, 1], stride=cl_stride, data_type=cudnn.data_type.HALF)
    X2 = g2.tensor(dim=[Nb, C, 1, 1], stride=cl_stride, data_type=cudnn.data_type.HALF)
    S2 = g2.tensor(dim=[1, C, 1, 1], stride=cl_stride, data_type=cudnn.data_type.HALF)
    M2 = g2.tensor(dim=[Nb, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
    IV2 = g2.tensor(dim=[Nb, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
    DX, DS, DB = g2.layernorm_backward(grad=DY, input=X2, scale=S2, mean=M2, inv_variance=IV2)
    for t in (DX, DS, DB):
        t.set_output(True).set_data_type(cudnn.data_type.HALF)
    g2.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    dxb = cl(torch.empty(Nb, C, 1, 1, device="cuda", dtype=torch.float16))
    dsb = cl(torch.empty(1, C, 1, 1, device="cuda", dtype=torch.float16))
    dbb = cl(torch.empty(1, C, 1, 1, device="cuda", dtype=torch.float16))
    ws2 = torch.empty(max(g2.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g2.execute({DY: cl(grad.half()), X2: x.detach(), S2: scale.detach(), M2: mb, IV2: ivb, DX: dxb, DS: dsb, DB: dbb}, ws2, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g2)
    torch.testing.assert_close(dxb.float(), x.grad.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dsb.float(), scale.grad.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dbb.float(), bias.grad.float(), atol=5e-2, rtol=5e-2)


def test_native_pointwise_batch_lowers_to_backend():
    """Generated pointwise builders through real cuDNN: sqrt(abs(A@B)) clamped
    via binary max/min (keyword call style, input0/input1)."""
    h = _handle()
    a = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(1, K, N, device="cuda", dtype=torch.float16)
    lo = torch.full((1, 1, 1), 0.5, device="cuda", dtype=torch.float32)
    hi = torch.full((1, 1, 1), 2.0, device="cuda", dtype=torch.float32)
    c = torch.empty(1, M, N, device="cuda", dtype=torch.float16)

    g = pygraph(handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.HALF)
    B = g.tensor(dim=[1, K, N], stride=[K * N, N, 1], data_type=cudnn.data_type.HALF)
    Lo = g.tensor(dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    Hi = g.tensor(dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    Y = g.min(input0=g.max(input0=g.sqrt(g.abs(g.matmul(A, B))), input1=Lo), input1=Hi)
    Y.set_output(True).set_data_type(cudnn.data_type.HALF)

    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, Lo: lo, Hi: hi, Y: c}, ws, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g)

    ref = (a.float() @ b.float()).abs().sqrt().clamp(0.5, 2.0)
    torch.testing.assert_close(c.float(), ref, atol=2e-2, rtol=2e-2)


def test_native_rmsnorm_lowers_to_backend():
    """rmsnorm (multi-output: Y + inv_var, pass-by-value epsilon) -> cuDNN parity.

    Regression cover for uid ownership: the Python IR assigns every uid eagerly
    and lowering pushes them all explicitly (set_uid on op outputs), so the C++
    FE's build-time auto-assignment never runs. Without this, multi-output ops
    get C++ uids in FE enumeration order (inv_var before Y here) != IR order —
    keying the variant pack by IR uids then bound Y's buffer to inv_var
    (heap corruption / NaN)."""
    h = _handle()
    Nb, C, Hh, W = 4, 8, 4, 4
    eps = 1e-3
    x = torch.randn(Nb, C, Hh, W, device="cuda", dtype=torch.float16)
    scale = torch.randn(1, C, Hh, W, device="cuda", dtype=torch.float16)
    bias = torch.randn(1, C, Hh, W, device="cuda", dtype=torch.float16)
    eps_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32)
    Yb = torch.empty_like(x)
    ivb = torch.empty(Nb, 1, 1, 1, device="cuda", dtype=torch.float32)

    g = pygraph(handle=h, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    X = g.tensor(dim=[Nb, C, Hh, W], stride=[C * Hh * W, Hh * W, W, 1], data_type=cudnn.data_type.HALF)
    S = g.tensor(dim=[1, C, Hh, W], stride=[C * Hh * W, Hh * W, W, 1], data_type=cudnn.data_type.HALF)
    Bi = g.tensor(dim=[1, C, Hh, W], stride=[C * Hh * W, Hh * W, W, 1], data_type=cudnn.data_type.HALF)
    E = g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT, is_pass_by_value=True)
    Y, iv = g.rmsnorm(input=X, scale=S, epsilon=E, bias=Bi, norm_forward_phase=cudnn.norm_forward_phase.TRAINING)
    Y.set_output(True).set_data_type(cudnn.data_type.HALF)
    iv.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    # first-class introspection: named ports + params
    node = g.nodes[0]
    assert node.node_type.name == "RMSNORM"
    assert set(node.inputs) == {"input", "scale", "epsilon", "bias"}
    assert set(node.outputs) == {"Y", "inv_var"}

    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({X: x, S: scale, Bi: bias, E: eps_cpu, Y: Yb, iv: ivb}, ws, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g)

    xf = x.float()
    ivref = torch.rsqrt(xf.pow(2).mean(dim=(1, 2, 3), keepdim=True) + eps)
    Yref = scale.float() * (xf * ivref) + bias.float()
    torch.testing.assert_close(Yb.float(), Yref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(ivb, ivref, atol=5e-3, rtol=5e-3)


def test_mixed_router_backend_slot_executes():
    """Review round 4: the backend entry of a MIXED router is selectable and
    actually executes through the backend (lowering triggered), with routed
    indices stable across that lowering; the pinned python plan still runs
    afterwards with its own knobs."""
    from cudnn.engines import BaseEngine, PlanConfig, Router
    from cudnn.engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID, PYTHON_ENGINE_ID_BASE

    ran = []

    class PyMatmul(BaseEngine):
        name = "py_matmul"
        engine_id = PYTHON_ENGINE_ID_BASE + 90

        def execute(self, graph, tensor_data, ctx=None):
            node = graph.nodes[0]
            a = tensor_data[node.inputs["A"].uid]
            b = tensor_data[node.inputs["B"].uid]
            c = tensor_data[node.outputs["C"].uid]
            c.copy_((a.float() @ b.float()).to(c.dtype))
            ran.append("python")

    class CudnnFirst(Router):
        def plan(self, graph, backends):
            return [PlanConfig(BACKEND_HEURISTIC_ENGINE_ID), PlanConfig(backends[0].engine_id)]

    h = _handle()
    a = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(1, K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(1, M, N, device="cuda", dtype=torch.float16)
    ref = (a.float() @ b.float()).half()

    g = pygraph(
        handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, router=CudnnFirst()
    )
    g.register_backend(PyMatmul())
    A = g.tensor(dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(dim=[1, K, N], stride=[K * N, N, 1])
    C = g.matmul(A, B)
    C.set_output(True).set_data_type(cudnn.data_type.HALF)

    g.create_execution_plans()
    assert [p.engine_id for p in g.plans] == [BACKEND_HEURISTIC_ENGINE_ID, PYTHON_ENGINE_ID_BASE + 90]

    # slot 0 = cuDNN: this build/execute lowers and runs the real backend
    assert g.selected_engine is None
    g.build()
    assert g._lowered_graph is not None  # lowering really happened
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, C: c}, ws, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g)
    torch.testing.assert_close(c.float(), ref.float(), atol=2e-2, rtol=2e-2)
    assert ran == []  # the python engine did NOT run

    # backend count is the classic passthrough space; routed indices unmoved
    assert g.get_execution_plan_count() >= 1
    assert [p.engine_id for p in g.plans] == [BACKEND_HEURISTIC_ENGINE_ID, PYTHON_ENGINE_ID_BASE + 90]

    # slot 1 = the python plan, still selectable AFTER backend lowering
    c.zero_()
    g.select_plan(1)
    assert g.selected_engine.name == "py_matmul"
    g.execute({A: a, B: b, C: c}, ws, handle=h)
    torch.cuda.synchronize()
    assert g.selected_engine is not None  # this slot is the python engine
    assert ran == ["python"]
    torch.testing.assert_close(c.float(), ref.float(), atol=2e-2, rtol=2e-2)


def test_planning_one_shot_backend_only():
    """Review round 4: one-shot planning also covers the pure-cuDNN graph (no
    python engines registered) — a second create_execution_plans() raises."""
    fn = pygraph.create_execution_plans
    if "pygraph" not in getattr(fn, "__qualname__", ""):
        pytest.skip(f"cudnn.pygraph.create_execution_plans is monkey-patched ({getattr(fn, '__qualname__', '?')}); the wrapper swallows the one-shot error")
    h = _handle()
    g = pygraph(handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(dim=[1, K, N], stride=[K * N, N, 1])
    C = g.matmul(A, B)
    C.set_output(True).set_data_type(cudnn.data_type.HALF)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    with pytest.raises(RuntimeError, match="one-shot"):
        g.create_execution_plans([cudnn.heur_mode.A])
    # the first plan set is intact and usable
    g.check_support()
    g.build_plans()
    a = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(1, K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(1, M, N, device="cuda", dtype=torch.float16)
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, C: c}, ws, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g)
    torch.testing.assert_close(c.float(), (a.float() @ b.float()), atol=2e-2, rtol=2e-2)


def test_output_layout_contract():
    """Review round 5: USER-assigned output dim/stride must reach the lowered
    C++ tensor; IR-INFERRED strides must NOT be pushed — the backend keeps its
    classic per-op layout inference (channels-last conv) when the user did not
    pin one. Checked on the lowered graph JSON and by execution."""
    import json

    # (a) explicit matmul output stride is honored end to end
    h = _handle()
    g = pygraph(handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(dim=[1, K, N], stride=[K * N, N, 1])
    C = g.matmul(A, B)
    C.set_output(True).set_data_type(cudnn.data_type.HALF).set_dim([1, M, N]).set_stride([M * N, 1, M])  # column-major
    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])

    lowered = json.loads(str(g._lowered_graph))
    (c_entry,) = [t for t in lowered["tensors"] if t["uid"] == C.uid]
    assert c_entry["stride"] == [M * N, 1, M]  # user layout pushed verbatim

    a = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(1, K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(1, N, M, device="cuda", dtype=torch.float16).permute(0, 2, 1)  # column-major buffer
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, C: c}, ws, handle=h)
    torch.cuda.synchronize()
    _assert_ran_on_backend(g)
    torch.testing.assert_close(c.float(), (a.float() @ b.float()), atol=2e-2, rtol=2e-2)

    # (b) inferred conv output keeps the backend's channels-last inference
    # (the IR's provisional row-major stride must NOT leak into C++)
    h2 = _handle()
    Nn, Cc, Hh, Ww, Kk = 4, 32, 16, 16, 16
    g2 = pygraph(handle=h2, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    X = g2.tensor(dim=[Nn, Cc, Hh, Ww], stride=[Cc * Hh * Ww, 1, Cc * Ww, Cc])  # NHWC
    W = g2.tensor(dim=[Kk, Cc, 3, 3], stride=[Cc * 9, 1, Cc * 3, Cc])
    Y = g2.conv_fprop(X, W, padding=[1, 1], stride=[1, 1], dilation=[1, 1])
    Y.set_output(True).set_data_type(cudnn.data_type.HALF)
    g2.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])

    lowered2 = json.loads(str(g2._lowered_graph))
    (y_entry,) = [t for t in lowered2["tensors"] if t["uid"] == Y.uid]
    assert y_entry["stride"][1] == 1, y_entry["stride"]  # channels-last kept, not row-major
