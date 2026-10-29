# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real cuDNN-graph round trip through the frost norm engines.

``pygraph`` -> node -> ``create_execution_plans`` -> select OUR engine ->
``check_support`` -> ``build_plans`` -> ``execute`` -> compare against PyTorch.

This is the one thing ``test_frost_norm_engine_e2e`` cannot do: that file drives
the lowering with synthetic node objects, so it exercises buffer resolution and
the kernel launch but never the C++ graph lifecycle, and it never asks whether an
output the graph declared actually came back written.

That gap hid a real defect. The analyzer derived ``has_beta`` from the bias
INPUT, which exists only on a forward graph; every backward graph therefore
reported ``has_beta=False``, the lowering passed that to ``norm_bprop``, which
returned ``dbeta=None``, and the engine skipped the DBias copy. The gradient was
never written and nothing complained -- the buffer simply kept whatever the
caller put there.

Hence the shape of the assertions here: every output buffer is filled with NaN
before ``execute`` and must come back entirely finite. A missing write is then a
failure rather than a silent pass, whatever the cause, and the check generalises
to any output a future node gains.
"""

import pytest
import torch

import cudnn

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")

_HEUR = [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
_PY_ENGINE_FLOOR = 20_000  # python engine ids start here; the backend's are small
_OK = True


def _expect(cond, msg):
    global _OK
    _OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    return bool(cond)


def _nan(shape, dtype):
    return torch.full(shape, float("nan"), device="cuda", dtype=dtype)


def _mark(t, shape, stride, dtype):
    t.set_output(True).set_data_type(dtype)
    t.set_dim(shape)
    t.set_stride(stride)
    return t


def _select_our_engine(graph, label):
    """Rank, then pick the frost engine. Returns its plan index, or None."""
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans(_HEUR)
    idx = next((i for i, p in enumerate(graph.plans) if p.engine_id >= _PY_ENGINE_FLOOR), None)
    if not _expect(idx is not None, f"{label}: frost engine appears in the {len(graph.plans)} ranked plans"):
        return None
    graph.select_plan(idx)
    graph.check_support()
    graph.build_plans()
    name = getattr(graph.selected_engine, "name", graph.selected_engine)
    _expect("norm" in str(name), f"{label}: selected engine is {name} (id {graph.plans[idx].engine_id})")
    return idx


def _check_written(label, **buffers):
    """Every declared output must come back fully finite -- see module docstring."""
    for name, buf in buffers.items():
        finite = int(torch.isfinite(buf).sum())
        _expect(finite == buf.numel(), f"{label}: {name} fully written ({finite}/{buf.numel()} finite)")


def _close(label, name, got, ref):
    rel = (got.float() - ref.float()).abs().max().item() / max(1.0, ref.float().abs().max().item())
    try:
        torch.testing.assert_close(got, ref, check_dtype=False)
        ok = True
    except AssertionError as exc:
        print(f"{label}: {name}: {exc}")
        ok = False
    _expect(ok, f"{label}: {name} matches PyTorch (rel {rel:.2e})")


def _forward(handle, B, S, D, dtype):
    label = f"LN fwd {B}x{S}x{D} {str(dtype).replace('torch.', '')}"
    print(f"\n=== {label} ===")
    torch.manual_seed(0)
    x = torch.randn(B, S, D, device="cuda", dtype=dtype)
    sc = torch.randn(1, 1, D, device="cuda", dtype=dtype)
    bi = torch.randn(1, 1, D, device="cuda", dtype=dtype)
    eps_cpu = torch.full((1, 1, 1), 1e-5, device="cpu", dtype=torch.float32)

    g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    X = g.tensor(name="X", dim=x.size(), stride=x.stride(), data_type=x.dtype)
    SC = g.tensor(name="scale", dim=sc.size(), stride=sc.stride(), data_type=sc.dtype)
    BI = g.tensor(name="bias", dim=bi.size(), stride=bi.stride(), data_type=bi.dtype)
    EP = g.tensor_like(eps_cpu, name="epsilon")
    Y, MEAN, IVAR = g.layernorm(name="LayerNorm", norm_forward_phase=cudnn.norm_forward_phase.TRAINING, input=X, scale=SC, bias=BI, epsilon=EP)
    stats, sstride = (B, S, 1), (S, 1, 1)
    _mark(Y, (B, S, D), x.stride(), dtype)
    _mark(MEAN, stats, sstride, torch.float32)
    _mark(IVAR, stats, sstride, torch.float32)

    if _select_our_engine(g, label) is None:
        return
    y, mean, ivar = _nan((B, S, D), dtype), _nan(stats, torch.float32), _nan(stats, torch.float32)
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    g.execute({X: x, SC: sc, BI: bi, EP: eps_cpu, Y: y, MEAN: mean, IVAR: ivar}, ws, handle=handle)
    torch.cuda.synchronize()

    _check_written(label, y=y, mean=mean, inv_variance=ivar)
    ref = torch.nn.functional.layer_norm(x, [D], weight=sc.reshape(-1), bias=bi.reshape(-1), eps=1e-5)
    _close(label, "y", y, ref)
    _close(label, "mean", mean, x.float().mean(-1, keepdim=True))
    _close(label, "inv_variance", ivar, torch.rsqrt(x.float().var(-1, keepdim=True, correction=0) + 1e-5))


def _backward(handle, B, S, D, dtype):
    label = f"LN bwd {B}x{S}x{D} {str(dtype).replace('torch.', '')}"
    print(f"\n=== {label} ===")
    torch.manual_seed(0)
    x = torch.randn(B, S, D, device="cuda", dtype=dtype)
    sc = torch.randn(1, 1, D, device="cuda", dtype=dtype)
    bi = torch.randn(1, 1, D, device="cuda", dtype=dtype)
    xf = x.float()
    mean = xf.mean(-1, keepdim=True).contiguous()
    ivar = torch.rsqrt(xf.var(-1, keepdim=True, correction=0) + 1e-5).contiguous()

    xr = xf.detach().requires_grad_(True)
    gr = sc.float().reshape(-1).detach().requires_grad_(True)
    br = bi.float().reshape(-1).detach().requires_grad_(True)
    yv = torch.nn.functional.layer_norm(xr, [D], weight=gr, bias=br, eps=1e-5)
    # Match the norm notebooks' mean-MSE backward construction.
    loss = torch.nn.functional.mse_loss(yv, torch.randn_like(x).float(), reduction="mean")
    dy = torch.autograd.grad(loss, yv, retain_graph=True)[0].to(dtype)

    g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    X = g.tensor(name="X", dim=x.size(), stride=x.stride(), data_type=x.dtype)
    DY = g.tensor(name="grad", dim=dy.size(), stride=dy.stride(), data_type=dy.dtype)
    SC = g.tensor(name="scale", dim=sc.size(), stride=sc.stride(), data_type=sc.dtype)
    MN = g.tensor(name="mean", dim=mean.size(), stride=mean.stride(), data_type=mean.dtype)
    IV = g.tensor(name="ivar", dim=ivar.size(), stride=ivar.stride(), data_type=ivar.dtype)
    DX, DS, DB = g.layernorm_backward(name="LayerNormBackward", grad=DY, input=X, scale=SC, mean=MN, inv_variance=IV)
    _mark(DX, (B, S, D), x.stride(), dtype)
    _mark(DS, (1, 1, D), sc.stride(), torch.float32)
    _mark(DB, (1, 1, D), sc.stride(), torch.float32)

    if _select_our_engine(g, label) is None:
        return
    dx, ds, db = _nan((B, S, D), dtype), _nan((1, 1, D), torch.float32), _nan((1, 1, D), torch.float32)
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    g.execute({X: x, DY: dy, SC: sc, MN: mean, IV: ivar, DX: dx, DS: ds, DB: db}, ws, handle=handle)
    torch.cuda.synchronize()

    # DBias is the one that regressed: has_beta was read off the bias INPUT, which
    # a backward graph does not have, so this buffer was left untouched.
    _check_written(label, dx=dx, dscale=ds, dbias=db)

    rdx, rds, rdb = torch.autograd.grad(yv, [xr, gr, br], grad_outputs=dy.float())
    _close(label, "dx", dx, rdx)
    _close(label, "dscale", ds.reshape(-1), rds)
    _close(label, "dbias", db.reshape(-1), rdb)


def main():
    assert torch.cuda.is_available(), "CUDA required"
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
    for dtype in (torch.bfloat16, torch.float32):
        _forward(handle, 8, 128, 1024, dtype)
        _backward(handle, 8, 128, 1024, dtype)
    print("\n" + ("ALL PASS" if _OK else "SOME FAILED"))
    raise SystemExit(0 if _OK else 1)


if __name__ == "__main__":
    main()


@pytest.mark.L0
def test_frost_norm_graph_roundtrip():
    """pytest entry point; this module also runs standalone via ``__main__``."""
    from norm_test_utils import frost_norm_rows_offered_here

    if not torch.cuda.is_available() or not frost_norm_rows_offered_here():
        pytest.skip("no norm engine row on offer serves this device")
    try:
        main()
    except SystemExit as exc:
        assert not exc.code, "checks failed -- see captured stdout"
