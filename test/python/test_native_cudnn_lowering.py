"""GPU parity: NativeGraph builds natively, lowers to cuDNN, executes correctly.

Covers the native -> _lower_to_cpp -> cuDNN execute path (uid propagation, handle
threading, pointwise dispatch). Skipped without a GPU / cuDNN.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

import cudnn
from cudnn.graph_native import NativeGraph

pytestmark = pytest.mark.L0

M, K, N = 64, 32, 48


def _handle():
    return cudnn.create_handle()


def test_native_matmul_lowers_to_cudnn():
    h = _handle()
    a = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(1, K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(1, M, N, device="cuda", dtype=torch.float16)

    g = NativeGraph(handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.HALF)
    B = g.tensor(dim=[1, K, N], stride=[K * N, N, 1], data_type=cudnn.data_type.HALF)
    C = g.matmul(A, B)
    C.set_output(True).set_data_type(cudnn.data_type.HALF)

    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, C: c}, ws, handle=h)
    torch.cuda.synchronize()

    torch.testing.assert_close(c.float(), a.float() @ b.float(), atol=2e-2, rtol=2e-2)


def test_native_matmul_bias_relu_lowers_to_cudnn():
    h = _handle()
    a = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(1, K, N, device="cuda", dtype=torch.float16)
    bias = torch.randn(1, M, N, device="cuda", dtype=torch.float16)
    c = torch.empty(1, M, N, device="cuda", dtype=torch.float16)

    g = NativeGraph(handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.HALF)
    B = g.tensor(dim=[1, K, N], stride=[K * N, N, 1], data_type=cudnn.data_type.HALF)
    Bi = g.tensor(dim=[1, M, N], stride=[M * N, N, 1], data_type=cudnn.data_type.HALF)
    Y = g.relu(g.bias(g.matmul(A, B), Bi))
    Y.set_output(True).set_data_type(cudnn.data_type.HALF)

    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, Bi: bias, Y: c}, ws, handle=h)
    torch.cuda.synchronize()

    torch.testing.assert_close(c.float(), torch.relu(a.float() @ b.float() + bias.float()), atol=2e-2, rtol=2e-2)


def test_native_matmul_reduction_lowers_to_cudnn():
    """matmul -> reduction(ADD) over N; cuDNN needs explicit reduced output dims."""
    h = _handle()
    a = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(1, K, N, device="cuda", dtype=torch.float16)
    r = torch.empty(1, M, 1, device="cuda", dtype=torch.float32)

    g = NativeGraph(handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.HALF)
    B = g.tensor(dim=[1, K, N], stride=[K * N, N, 1], data_type=cudnn.data_type.HALF)
    R = g.reduction(g.matmul(A, B), cudnn.reduction_mode.ADD, dim=[1, M, 1])
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, R: r}, ws, handle=h)
    torch.cuda.synchronize()

    torch.testing.assert_close(r, (a.float() @ b.float()).sum(dim=2, keepdim=True), atol=5e-2, rtol=5e-2)


def test_native_block_scale_nvfp4_lowers_to_cudnn():
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

    g = NativeGraph(handle=h, compute_data_type=cudnn.data_type.FLOAT)
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
    torch.cuda.synchronize()  # builds + executes without error (parity harness = repo's fp4 test)


def test_native_moe_grouped_matmul_lowers_to_cudnn():
    """moe_grouped_matmul (mode=NONE) built natively -> cuDNN, parity vs a
    self-contained per-expert reference."""
    h = _handle()
    E, T, Wt, Hd = 8, 256, 64, 128
    fto = [i * (T // E) for i in range(E)]  # one contiguous token chunk per expert

    g = NativeGraph(handle=h, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
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


def test_native_rmsnorm_lowers_to_cudnn():
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

    g = NativeGraph(handle=h, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
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

    xf = x.float()
    ivref = torch.rsqrt(xf.pow(2).mean(dim=(1, 2, 3), keepdim=True) + eps)
    Yref = scale.float() * (xf * ivref) + bias.float()
    torch.testing.assert_close(Yb.float(), Yref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(ivb, ivref, atol=5e-3, rtol=5e-3)
