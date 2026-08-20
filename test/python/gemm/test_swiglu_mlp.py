# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for cudnn.gemm.ops.swiglu_mlp (dense bf16 SwiGLU-MLP).

The fused SwiGLU forward runs on cuDNN's runtime-fusion engine, which needs an
SM100 (Blackwell) device; the op must match torch to bf16 noise on the output and
all four gradients.
"""

import pytest
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from cudnn.gemm.ops import swiglu_mlp


def _cc():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _rel_l2(a, b):
    return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-9)


def _ref(x, Wg, Wu, Wd):
    return (F.silu(x @ Wg.t()) * (x @ Wu.t())) @ Wd.t()


# bf16 noise across three chained GEMMs + the SwiGLU; cuDNN vs torch differ only by
# accumulation order, so a relative-L2 at this level is the meaningful bar.
_TOL = 2e-2


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() >= 100),
    reason="cuDNN SwiGLU-MLP fusion requires SM100 (Blackwell)",
)
@pytest.mark.parametrize(
    "M,H,inter",
    [(512, 512, 1024), (2048, 1024, 2048)],
    ids=["small", "mlp"],
)
def test_swiglu_mlp_parity(M, H, inter):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
    Wg = (torch.randn(inter, H, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    Wu = (torch.randn(inter, H, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    Wd = (torch.randn(H, inter, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    do = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16)

    xr, Wgr, Wur, Wdr = (t.detach().clone().requires_grad_(True) for t in (x, Wg, Wu, Wd))

    out = swiglu_mlp(x, Wg, Wu, Wd)
    out.backward(do)
    ref = _ref(xr, Wgr, Wur, Wdr)
    ref.backward(do)

    assert _rel_l2(out, ref) < _TOL, f"fwd rel={_rel_l2(out, ref):.2e}"
    for name, a, b in (("dx", x.grad, xr.grad), ("dWg", Wg.grad, Wgr.grad), ("dWu", Wu.grad, Wur.grad), ("dWd", Wd.grad, Wdr.grad)):
        assert _rel_l2(a, b) < _TOL, f"{name} rel={_rel_l2(a, b):.2e}"


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() >= 100),
    reason="cuDNN SwiGLU-MLP fusion requires SM100 (Blackwell)",
)
@pytest.mark.parametrize(
    "required,expected_saved",
    [
        ((True, False, False, False), ("Wg", "Wu", "Wd", "gate", "up")),
        ((False, True, False, False), ("x2", "Wd", "gate", "up")),
        ((False, False, True, False), ("x2", "Wd", "gate", "up")),
        ((False, False, False, True), ("h",)),
        ((False, True, True, True), ("x2", "Wd", "gate", "up", "h")),
        ((True, True, True, True), ("x2", "Wg", "Wu", "Wd", "gate", "up", "h")),
    ],
    ids=["x-only", "Wg-only", "Wu-only", "Wd-only", "weights-only", "all"],
)
def test_swiglu_mlp_partial_grad(required, expected_saved):
    """Only requested input gradients are retained and computed."""
    torch.manual_seed(1)
    M, H, inter = 512, 512, 1024
    base = (
        torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16),
        torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(H, inter, device="cuda", dtype=torch.bfloat16) * 0.02,
    )
    args = tuple(t.detach().clone().requires_grad_(need) for t, need in zip(base, required))
    refs = tuple(t.detach().clone().requires_grad_(need) for t, need in zip(base, required))
    dout = torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16)

    packed = []

    def pack_hook(tensor):
        packed.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        out = swiglu_mlp(*args)
    ref = _ref(*refs)
    assert tuple(out.grad_fn.saved_names) == expected_saved
    assert len(packed) == len(expected_saved)
    out.backward(dout)
    ref.backward(dout)

    assert _rel_l2(out, ref) < _TOL, f"fwd rel={_rel_l2(out, ref):.2e}"
    for name, got, expected, need in zip(("dx", "dWg", "dWu", "dWd"), args, refs, required):
        if need:
            assert got.grad is not None
            assert expected.grad is not None
            assert _rel_l2(got.grad, expected.grad) < _TOL, f"{name} rel={_rel_l2(got.grad, expected.grad):.2e}"
        else:
            assert got.grad is None, f"{name} should not have been materialized"


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() >= 100),
    reason="cuDNN SwiGLU-MLP fusion requires SM100 (Blackwell)",
)
@pytest.mark.parametrize("grad_mode", [torch.no_grad, torch.inference_mode], ids=["no-grad", "inference-mode"])
def test_swiglu_mlp_inference_uses_h_only(monkeypatch, grad_mode):
    """Outer inference state must override trainable parameter flags."""
    import importlib

    mod = importlib.import_module("cudnn.gemm.ops.swiglu_mlp")
    original = mod._swiglu_act
    save_preacts_seen = []

    def record_swiglu_act(*args, **kwargs):
        save_preacts_seen.append(kwargs["save_preacts"])
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, "_swiglu_act", record_swiglu_act)
    torch.manual_seed(2)
    M, H, inter = 512, 512, 1024
    x = torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    Wg = (torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    Wu = (torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    Wd = (torch.randn(H, inter, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(True)

    with grad_mode():
        out = mod.swiglu_mlp(x, Wg, Wu, Wd)
        ref = _ref(x, Wg, Wu, Wd)

    assert not out.requires_grad
    assert save_preacts_seen == [False]
    assert _rel_l2(out, ref) < _TOL, f"fwd rel={_rel_l2(out, ref):.2e}"


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() >= 100),
    reason="cuDNN SwiGLU-MLP fusion requires SM100 (Blackwell)",
)
def test_swiglu_mlp_all_frozen_uses_h_only(monkeypatch):
    """GradMode can be enabled while no input participates in autograd."""
    import importlib

    mod = importlib.import_module("cudnn.gemm.ops.swiglu_mlp")
    original = mod._swiglu_act
    save_preacts_seen = []

    def record_swiglu_act(*args, **kwargs):
        save_preacts_seen.append(kwargs["save_preacts"])
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, "_swiglu_act", record_swiglu_act)
    torch.manual_seed(3)
    M, H, inter = 512, 512, 1024
    x = torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16)
    Wg = torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02
    Wu = torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02
    Wd = torch.randn(H, inter, device="cuda", dtype=torch.bfloat16) * 0.02

    assert torch.is_grad_enabled()
    out = mod.swiglu_mlp(x, Wg, Wu, Wd)
    ref = _ref(x, Wg, Wu, Wd)

    assert not out.requires_grad
    assert save_preacts_seen == [False]
    assert _rel_l2(out, ref) < _TOL, f"fwd rel={_rel_l2(out, ref):.2e}"


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() >= 100),
    reason="cuDNN SwiGLU-MLP fusion requires SM100 (Blackwell)",
)
def test_swiglu_mlp_h_only_matches_full_and_cache_switches():
    """The output mask is part of the plan cache and must not change h rounding."""
    import importlib

    mod = importlib.import_module("cudnn.gemm.ops.swiglu_mlp")
    torch.manual_seed(4)
    M, H, inter = 512, 512, 1024
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    Wg = torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02
    Wu = torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02

    h_only_1, gate_1, up_1 = mod._swiglu_act(x, Wg, Wu, save_preacts=False)
    h_full, gate_full, up_full = mod._swiglu_act(x, Wg, Wu, save_preacts=True)
    h_only_2, gate_2, up_2 = mod._swiglu_act(x, Wg, Wu, save_preacts=False)

    assert gate_1 is up_1 is gate_2 is up_2 is None
    assert gate_full is not None and up_full is not None
    assert torch.equal(h_only_1, h_full)
    assert torch.equal(h_only_1, h_only_2)
    matching = [key for key in mod._SWIGLU_CACHE if key[:3] == (M, H, inter)]
    assert {key[7] for key in matching} == {False, True}


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() >= 100),
    reason="cuDNN SwiGLU-MLP fusion requires SM100 (Blackwell)",
)
@pytest.mark.parametrize("use_reentrant", [True, False], ids=["reentrant", "non-reentrant"])
def test_swiglu_mlp_checkpoint_parity(use_reentrant):
    """Checkpoint's no-grad first pass must not suppress the recomputed backward state."""
    torch.manual_seed(5)
    M, H, inter = 512, 512, 1024
    base = (
        torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16),
        torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(H, inter, device="cuda", dtype=torch.bfloat16) * 0.02,
    )
    args = tuple(t.detach().clone().requires_grad_(True) for t in base)
    refs = tuple(t.detach().clone().requires_grad_(True) for t in base)
    dout = torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16)

    out = torch.utils.checkpoint.checkpoint(swiglu_mlp, *args, use_reentrant=use_reentrant)
    ref = _ref(*refs)
    out.backward(dout)
    ref.backward(dout)

    assert _rel_l2(out, ref) < _TOL, f"fwd rel={_rel_l2(out, ref):.2e}"
    for name, got, expected in zip(("dx", "dWg", "dWu", "dWd"), args, refs):
        assert _rel_l2(got.grad, expected.grad) < _TOL, f"{name} rel={_rel_l2(got.grad, expected.grad):.2e}"


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() >= 100),
    reason="cuDNN SwiGLU-MLP fusion requires SM100 (Blackwell)",
)
@pytest.mark.parametrize("M,H,inter", [(512, 512, 1024), (2048, 1024, 2048)], ids=["small", "mlp"])
def test_frost_dswiglu_matches_pointwise(M, H, inter):
    """The fused FROST backward (dh GEMM + dSwiGLU epilogue) must match the
    separate dh GEMM + two pointwise kernels it replaces, to bf16 noise."""
    from cudnn.gemm.ops.swiglu_mlp import _frost_dswiglu, _dswiglu

    torch.manual_seed(0)
    dev = "cuda"
    dout = torch.randn(M, H, device=dev, dtype=torch.bfloat16) * 0.1
    Wd = torch.randn(H, inter, device=dev, dtype=torch.bfloat16) * 0.02
    gate = torch.randn(M, inter, device=dev, dtype=torch.bfloat16) * 0.8
    up = torch.randn(M, inter, device=dev, dtype=torch.bfloat16) * 0.8

    dgate_f, dup_f = _frost_dswiglu(dout, Wd, gate, up)  # dh = dout@Wd fused with dSwiGLU
    dgate_p, dup_p = _dswiglu(torch.mm(dout, Wd), gate, up)  # separate dh GEMM + two pointwise kernels
    assert _rel_l2(dgate_f, dgate_p) < _TOL, f"dgate rel={_rel_l2(dgate_f, dgate_p):.2e}"
    assert _rel_l2(dup_f, dup_p) < _TOL, f"dup rel={_rel_l2(dup_f, dup_p):.2e}"


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() >= 100),
    reason="cuDNN SwiGLU-MLP fusion requires SM100 (Blackwell)",
)
@pytest.mark.parametrize("save_preacts", [False, True], ids=["h-only", "with-preacts"])
def test_swiglu_mlp_forward_is_single_kernel(save_preacts):
    """The gate GEMM + up GEMM + SiLU + mul must fuse into ONE cuDNN launch."""
    from cudnn.gemm.ops.swiglu_mlp import _swiglu_act

    torch.manual_seed(0)
    M, H, inter = 512, 512, 1024
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    Wg = torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02
    Wu = torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02
    h, gate, up = _swiglu_act(x, Wg, Wu, save_preacts=save_preacts)  # warm/cache/autotune
    assert h.shape == (M, inter)
    assert (gate is not None, up is not None) == (save_preacts, save_preacts)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        _swiglu_act(x, Wg, Wu, save_preacts=save_preacts)
        torch.cuda.synchronize()
    launches = sum(ev.count for ev in prof.key_averages() if ev.self_device_time_total > 0)
    assert launches == 1, f"expected 1 fused kernel, saw {launches}"
