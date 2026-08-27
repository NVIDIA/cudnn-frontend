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


def _dswiglu_ref(dh, gate, up):
    sigmoid = torch.sigmoid(gate)
    silu = gate * sigmoid
    return dh * up * (sigmoid + silu * (1 - sigmoid)), dh * silu


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
def test_swiglu_mlp_sum_loss_parity():
    """Expanded zero-stride upstream gradients are normalized before GEMMs."""
    torch.manual_seed(8)
    M, H, inter = 128, 256, 256
    base = (
        torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16),
        torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(H, inter, device="cuda", dtype=torch.bfloat16) * 0.02,
    )
    args = tuple(t.detach().clone().requires_grad_(True) for t in base)
    refs = tuple(t.detach().clone().requires_grad_(True) for t in base)

    out = swiglu_mlp(*args)
    ref = _ref(*refs)
    out.sum().backward()
    ref.sum().backward()

    assert _rel_l2(out, ref) < _TOL, f"fwd rel={_rel_l2(out, ref):.2e}"
    for name, got, expected in zip(("dx", "dWg", "dWu", "dWd"), args, refs):
        assert _rel_l2(got.grad, expected.grad) < _TOL, f"{name} rel={_rel_l2(got.grad, expected.grad):.2e}"


@pytest.mark.L0
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="mixed-device validation needs two CUDA devices")
def test_swiglu_mlp_rejects_mixed_devices():
    """Reject cross-device pointers before building or launching any graph."""
    H, inter = 16, 32
    current_index = torch.cuda.current_device()
    other_index = next(index for index in range(torch.cuda.device_count()) if index != current_index)
    current = torch.device("cuda", current_index)
    other = torch.device("cuda", other_index)
    x = torch.empty(1, 1, H, device=current, dtype=torch.bfloat16)
    Wg = torch.empty(inter, H, device=current, dtype=torch.bfloat16)
    Wu = torch.empty(inter, H, device=current, dtype=torch.bfloat16)
    Wd = torch.empty(H, inter, device=other, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="must be on the same CUDA device"):
        swiglu_mlp(x, Wg, Wu, Wd)


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
def test_swiglu_mlp_saved_preacts_restore_noncontiguous(monkeypatch):
    """Saved-tensor hooks may restore gate/up with valid non-dense strides;
    backward must normalize them before either dense dSwiGLU implementation."""
    import importlib

    mod = importlib.import_module("cudnn.gemm.ops.swiglu_mlp")
    original_dswiglu = mod._dswiglu
    dswiglu_strides = []

    def record_dswiglu(dh, gate, up):
        dswiglu_strides.append((gate.stride(), up.stride()))
        return original_dswiglu(dh, gate, up)

    monkeypatch.setattr(mod, "_dswiglu", record_dswiglu)
    monkeypatch.setattr(mod, "_FROST_BWD", False)
    torch.manual_seed(9)
    M, H, inter = 128, 256, 256
    base = (
        torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16),
        torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(H, inter, device="cuda", dtype=torch.bfloat16) * 0.02,
    )
    required = (True, True, True, False)
    args = tuple(t.detach().clone().requires_grad_(need) for t, need in zip(base, required))
    refs = tuple(t.detach().clone().requires_grad_(need) for t, need in zip(base, required))
    dout = torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16)

    packed_count = 0

    def pack_hook(tensor):
        nonlocal packed_count
        index = packed_count
        packed_count += 1
        # For this grad mask the custom Function saves
        # (x2, Wg, Wu, Wd, gate, up), in that order. Store the two
        # preactivations transposed-contiguous, then restore a logically equal
        # tensor with column-major strides in unpack_hook.
        if index in (4, 5):
            return True, tensor.t().contiguous()
        return False, tensor

    def unpack_hook(packed):
        transposed, tensor = packed
        return tensor.t() if transposed else tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        out = mod.swiglu_mlp(*args)
    assert tuple(out.grad_fn.saved_names) == ("x2", "Wg", "Wu", "Wd", "gate", "up")
    assert packed_count == 6
    ref = _ref(*refs)
    out.backward(dout)
    ref.backward(dout)

    assert dswiglu_strides == [((inter, 1), (inter, 1))]
    assert _rel_l2(out, ref) < _TOL, f"fwd rel={_rel_l2(out, ref):.2e}"
    for name, got, expected, need in zip(("dx", "dWg", "dWu", "dWd"), args, refs, required):
        if need:
            assert _rel_l2(got.grad, expected.grad) < _TOL, f"{name} rel={_rel_l2(got.grad, expected.grad):.2e}"
        else:
            assert got.grad is None


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
@pytest.mark.parametrize(
    "M,H,inter",
    [(128, 512, 1024), (512, 512, 1024), (2048, 1024, 2048)],
    ids=["1cta", "small", "mlp"],
)
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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FROST layout preflight requires a CUDA tensor")
def test_frost_dswiglu_declines_misaligned_contiguous_view():
    """A row-major offset view still needs the direct kernel's pointer alignment."""
    from cudnn.gemm.ops.swiglu_mlp import _frost_dswiglu

    M = H = inter = 8
    # Use visible ordinal 0 so the independent heterogeneous-architecture gate
    # cannot preempt the alignment diagnostic this test targets.
    device = "cuda:0"
    storage = torch.empty(M * H + 1, device=device, dtype=torch.bfloat16)
    dout = storage[1:].view(M, H)
    assert dout.stride() == (H, 1) and dout.data_ptr() % 32
    Wd = torch.empty(H, inter, device=device, dtype=torch.bfloat16)
    gate = torch.empty(M, inter, device=device, dtype=torch.bfloat16)
    up = torch.empty(M, inter, device=device, dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="32-byte-aligned"):
        _frost_dswiglu(dout, Wd, gate, up)


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() >= 100),
    reason="cuDNN SwiGLU-MLP fusion requires SM100 (Blackwell)",
)
def test_noncontiguous_square_wd_falls_back_and_matches(monkeypatch):
    """A square transpose keeps Wd's shape but must not be bound to FROST's
    hard-coded contiguous descriptor; the typed decline takes the nvjet path."""
    import importlib

    mod = importlib.import_module("cudnn.gemm.ops.swiglu_mlp")
    original_dswiglu = mod._dswiglu
    fallback_calls = []

    def record_dswiglu(*args, **kwargs):
        fallback_calls.append(True)
        return original_dswiglu(*args, **kwargs)

    monkeypatch.setattr(mod, "_dswiglu", record_dswiglu)
    monkeypatch.setattr(mod, "_FROST_BWD", True)
    torch.manual_seed(6)
    M = H = inter = 512
    x = torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    Wg = (torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    Wu = (torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    Wd = (torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02).t().requires_grad_(True)
    assert Wd.shape == (H, inter) and not Wd.is_contiguous()
    dout = torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16)

    xr, Wgr, Wur = (t.detach().clone().requires_grad_(True) for t in (x, Wg, Wu))
    Wdr = Wd.detach().contiguous().requires_grad_(True)
    out = mod.swiglu_mlp(x, Wg, Wu, Wd)
    out.backward(dout)
    ref = _ref(xr, Wgr, Wur, Wdr)
    ref.backward(dout)

    assert fallback_calls == [True]
    assert _rel_l2(out, ref) < _TOL, f"fwd rel={_rel_l2(out, ref):.2e}"
    for name, got, expected in (("dx", x.grad, xr.grad), ("dWg", Wg.grad, Wgr.grad), ("dWu", Wu.grad, Wur.grad), ("dWd", Wd.grad, Wdr.grad)):
        assert _rel_l2(got, expected) < _TOL, f"{name} rel={_rel_l2(got, expected):.2e}"


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() >= 100),
    reason="cuDNN SwiGLU-MLP fusion requires SM100 (Blackwell)",
)
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError], ids=["runtime", "value"])
def test_frost_runtime_failure_is_not_hidden(monkeypatch, error_type):
    """Only a typed unsupported-case decline may select the pointwise fallback."""
    import importlib

    mod = importlib.import_module("cudnn.gemm.ops.swiglu_mlp")

    def fail_launch(*args, **kwargs):
        raise error_type("synthetic FROST launch failure")

    monkeypatch.setattr(mod, "_frost_dswiglu", fail_launch)
    monkeypatch.setattr(mod, "_FROST_BWD", True)
    M, H, inter = 512, 512, 1024
    args = (
        torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(inter, H, device="cuda", dtype=torch.bfloat16),
        torch.randn(inter, H, device="cuda", dtype=torch.bfloat16),
        torch.randn(H, inter, device="cuda", dtype=torch.bfloat16),
    )
    out = mod.swiglu_mlp(*args)
    with pytest.raises(error_type, match="synthetic FROST launch failure"):
        out.sum().backward()


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() >= 100),
    reason="cuDNN SwiGLU-MLP fusion requires SM100 (Blackwell)",
)
def test_frost_dswiglu_cuda_graph_capture():
    """A warmed direct FROST call must launch on CUDA graph's capture stream;
    launching on stream 0 would invalidate capture deterministically."""
    from cudnn.gemm.ops.swiglu_mlp import _frost_dswiglu

    torch.manual_seed(7)
    M, H, inter = 128, 256, 256
    static_inputs = (
        torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.5,
        torch.randn(H, inter, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(M, inter, device="cuda", dtype=torch.bfloat16) * 0.8,
        torch.randn(M, inter, device="cuda", dtype=torch.bfloat16) * 0.8,
    )

    # Warm JIT/import/cache and allocator state on a side stream. The capture
    # below must contain only the already-compiled direct launch.
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        warm_outputs = _frost_dswiglu(*static_inputs)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    del warm_outputs

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        dgate, dup = _frost_dswiglu(*static_inputs)
    capture_stream.synchronize()

    # Capture itself executes the body once. Clear those results so only a
    # correctly captured/replayed launch can restore them; a stream-0 escape
    # would otherwise leave plausible outputs behind and mask an empty graph.
    dgate.zero_()
    dup.zero_()
    graph.replay()
    torch.cuda.synchronize()

    dgate_ref, dup_ref = _dswiglu_ref(torch.mm(static_inputs[0], static_inputs[1]), static_inputs[2], static_inputs[3])
    assert _rel_l2(dgate, dgate_ref) < _TOL, f"dgate rel={_rel_l2(dgate, dgate_ref):.2e}"
    assert _rel_l2(dup, dup_ref) < _TOL, f"dup rel={_rel_l2(dup, dup_ref):.2e}"


@pytest.mark.L0
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="target-device context test needs two CUDA devices")
def test_frost_dswiglu_uses_tensor_device_not_current_device():
    """The direct JIT/cache/launch path must use the operands' CUDA context."""
    from cudnn.gemm.ops.swiglu_mlp import _frost_dswiglu

    current_index = torch.cuda.current_device()
    jit_capability = torch.cuda.get_device_capability(0)
    target_index = next(
        (
            index
            for index in range(torch.cuda.device_count())
            if index != current_index and torch.cuda.get_device_capability(index) == jit_capability and jit_capability[0] >= 10
        ),
        None,
    )
    if target_index is None:
        pytest.skip("no non-current SM100 device matching visible device 0's CuTeDSL JIT architecture is visible")
    target = torch.device("cuda", target_index)
    M, H, inter = 128, 256, 256
    with torch.cuda.device(current_index):
        dout = torch.randn(M, H, device=target, dtype=torch.bfloat16) * 0.1
        Wd = torch.randn(H, inter, device=target, dtype=torch.bfloat16) * 0.02
        gate = torch.randn(M, inter, device=target, dtype=torch.bfloat16) * 0.8
        up = torch.randn(M, inter, device=target, dtype=torch.bfloat16) * 0.8
        assert torch.cuda.current_device() == current_index
        dgate, dup = _frost_dswiglu(dout, Wd, gate, up)
        assert torch.cuda.current_device() == current_index

    with torch.cuda.device(target):
        dgate_ref, dup_ref = _dswiglu_ref(torch.mm(dout, Wd), gate, up)
        torch.cuda.synchronize(target)
    assert _rel_l2(dgate, dgate_ref) < _TOL, f"dgate rel={_rel_l2(dgate, dgate_ref):.2e}"
    assert _rel_l2(dup, dup_ref) < _TOL, f"dup rel={_rel_l2(dup, dup_ref):.2e}"


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
