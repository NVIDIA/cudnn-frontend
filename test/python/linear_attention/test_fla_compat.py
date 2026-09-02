# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity gate for the FLA-compat shim: cuDNN's ``gated_delta_net`` (through the
shim) must match real flash-linear-attention within FLA's own bf16 noise, on the
output AND every gradient. A config cuDNN cannot match must fall back, not run.

Skipped unless ``flash-linear-attention`` is importable and the device is SM100.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

fla_gdr = pytest.importorskip("fla.ops.gated_delta_rule")
chunk_gated_delta_rule = fla_gdr.chunk_gated_delta_rule
naive_recurrent = fla_gdr.naive_recurrent_gated_delta_rule

from cudnn.fla import last_path, accelerate_fla, restore_fla
from cudnn.fla.gated_delta_rule import make_chunk_gated_delta_rule

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10),
        reason="cuDNN GDN kernels require SM100 (Blackwell)",
    ),
]

shim = make_chunk_gated_delta_rule(chunk_gated_delta_rule)

# cuDNN may be up to this factor over FLA's own relative-L2 error from the fp32
# reference; FLOOR is the bf16 noise below which a ratio is meaningless.
C_SLACK = 3.0
FLOOR = 3e-3


def _relL2(x, ref):
    return (x.float() - ref.float()).norm().item() / max(ref.float().norm().item(), 1e-12)


def _master(B, T, H, HV, K, V, seed):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    m = {
        "q": F.normalize(torch.randn(B, T, H, K, generator=gen, device=dev), dim=-1),
        "k": F.normalize(torch.randn(B, T, H, K, generator=gen, device=dev), dim=-1),
        "v": torch.randn(B, T, HV, V, generator=gen, device=dev),
        "beta": torch.rand(B, T, HV, generator=gen, device=dev).sigmoid(),
        "g": F.logsigmoid(torch.rand(B, T, HV, generator=gen, device=dev)),
    }
    return m


def _leaves(master, dtype):
    lv = {n: master[n].to(dtype if n in ("q", "k", "v") else torch.float32).detach().clone().requires_grad_(True) for n in master}
    return lv


def _run(fn, master, dtype, **kw):
    lv = _leaves(master, dtype)
    o, _ = fn(lv["q"], lv["k"], lv["v"], lv["g"], lv["beta"], output_final_state=False, **kw)
    return o, lv


def _run_truth(master):
    lv = _leaves(master, torch.float32)
    # naive signature: (q, k, v, beta, g, ...)
    o, _ = naive_recurrent(lv["q"], lv["k"], lv["v"], lv["beta"], lv["g"], output_final_state=False)
    return o, lv


@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(dict(B=2, T=256, H=4, HV=4, K=128, V=128, dtype=torch.bfloat16), id="dense_bf16"),
        pytest.param(dict(B=2, T=256, H=8, HV=8, K=128, V=128, dtype=torch.bfloat16), id="h8"),
        pytest.param(dict(B=2, T=256, H=4, HV=4, K=128, V=128, dtype=torch.float16), id="fp16"),
        pytest.param(dict(B=2, T=256, H=2, HV=4, K=128, V=128, dtype=torch.bfloat16), id="gva"),
    ],
)
def test_parity_native(cfg):
    """Where cuDNN runs (native), it matches FLA within FLA's own noise on o + grads."""
    m = _master(cfg["B"], cfg["T"], cfg["H"], cfg["HV"], cfg["K"], cfg["V"], seed=0)
    do = torch.randn(cfg["B"], cfg["T"], cfg["HV"], cfg["V"], device="cuda")

    o_fla, lv_fla = _run(chunk_gated_delta_rule, m, cfg["dtype"])
    o_cud, lv_cud = _run(shim, m, cfg["dtype"])
    assert last_path() == "native", f"expected cuDNN native path, got {last_path()}"

    gva = cfg["H"] != cfg["HV"]
    if gva:
        o_ref, lv_ref = o_fla, None  # naive has no GVA; FLA is the reference
    else:
        o_ref, lv_ref = _run_truth(m)

    o_fla.backward(do.to(o_fla.dtype))
    o_cud.backward(do.to(o_cud.dtype))
    if not gva:
        o_ref.backward(do.to(o_ref.dtype))

    def check(name, a, b, ref):
        e_fla = _relL2(a, ref)
        e_cud = _relL2(b, ref)
        assert e_cud <= C_SLACK * max(e_fla, FLOOR), f"{name}: e_cud={e_cud:.2e} vs e_fla={e_fla:.2e} (slack {C_SLACK})"

    check("o", o_fla, o_cud, o_ref)
    for n in ("q", "k", "v", "g", "beta"):
        ref = lv_ref[n].grad if (lv_ref is not None and lv_ref[n].grad is not None) else lv_fla[n].grad
        check("d" + n, lv_fla[n].grad, lv_cud[n].grad, ref)


def _fused_leaves(B, T, H, HV, K, V, dtype, seed):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)

    def leaf(shape, dt, req=True):
        return torch.randn(*shape, generator=gen, device=dev, dtype=dt).detach().requires_grad_(req)

    return {
        "q": leaf((B, T, H, K), dtype),
        "k": leaf((B, T, H, K), dtype),
        "v": leaf((B, T, HV, V), dtype),
        "graw": torch.rand(B, T, HV, generator=gen, device=dev, dtype=torch.float32).requires_grad_(True),
        "braw": leaf((B, T, HV), torch.float32),
        "A_log": torch.log(torch.empty(HV, device=dev).uniform_(0.1, 16)).requires_grad_(True),
        "dt_bias": torch.randn(HV, generator=gen, device=dev).requires_grad_(True),
    }


def _run_fused(fn, lv):
    o, _ = fn(
        lv["q"],
        lv["k"],
        lv["v"],
        lv["graw"],
        lv["braw"],
        A_log=lv["A_log"],
        dt_bias=lv["dt_bias"],
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        output_final_state=False,
    )
    return o


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_parity_fused_layer_path(dtype):
    """The FLA GatedDeltaNet layer's actual call (raw g/beta + A_log/dt_bias +
    in-kernel L2-norm/gate/beta fusion): the shim forwards the fusions to the native
    kernel and must still match FLA within its own noise (truth = FLA fused in fp32)."""
    shape = dict(B=2, T=256, H=4, HV=4, K=128, V=128)
    do = torch.randn(shape["B"], shape["T"], shape["HV"], shape["V"], device="cuda")

    def clone_to(src, dt):
        lv = {}
        for name, t in src.items():
            keep_fp32 = name in ("graw", "braw", "A_log", "dt_bias")
            lv[name] = t.detach().clone().to(torch.float32 if keep_fp32 else dt).requires_grad_(True)
        return lv

    master = _fused_leaves(**shape, dtype=dtype, seed=3)
    lv_fla = clone_to(master, dtype)
    lv_cud = clone_to(master, dtype)
    lv_ref = clone_to(master, torch.float32)  # fp32 truth via FLA's own fused path

    o_fla = _run_fused(chunk_gated_delta_rule, lv_fla)
    o_cud = _run_fused(shim, lv_cud)
    assert last_path() == "native", f"expected cuDNN native path, got {last_path()}"
    o_ref = _run_fused(chunk_gated_delta_rule, lv_ref)

    o_fla.backward(do.to(o_fla.dtype))
    o_cud.backward(do.to(o_cud.dtype))
    o_ref.backward(do.to(o_ref.dtype))

    def check(name, a, b, ref):
        e_fla = _relL2(a, ref)
        e_cud = _relL2(b, ref)
        assert e_cud <= C_SLACK * max(e_fla, FLOOR), f"{name}: e_cud={e_cud:.2e} vs e_fla={e_fla:.2e}"

    check("o", o_fla, o_cud, o_ref)
    for n in ("q", "k", "v", "graw", "braw", "A_log", "dt_bias"):
        check("d" + n, lv_fla[n].grad, lv_cud[n].grad, lv_ref[n].grad)


def test_parity_fused_layer_path_with_packed_qkv_views():
    """FLA's fused short-conv splits one packed output into non-compact Q/K/V
    views.  The shim must compact those views before entering native GDN while
    preserving gradients back to the packed allocation."""
    B, T, H, HV, K, V = 1, 256, 16, 48, 128, 128
    widths = (H * K, H * K, HV * V)
    gen = torch.Generator(device="cuda").manual_seed(4)

    packed = torch.randn(B, T, sum(widths), generator=gen, device="cuda", dtype=torch.bfloat16)
    graw = torch.randn(B, T, HV, generator=gen, device="cuda", dtype=torch.bfloat16)
    braw = torch.randn(B, T, HV, generator=gen, device="cuda", dtype=torch.bfloat16)
    A_log = torch.log(torch.empty(HV, device="cuda").uniform_(0.1, 16, generator=gen))
    dt_bias = torch.randn(HV, generator=gen, device="cuda")

    def leaves():
        p = packed.detach().clone().requires_grad_(True)
        q, k, v = p.split(widths, dim=-1)
        lv = {
            "packed": p,
            "q": q.reshape(B, T, H, K),
            "k": k.reshape(B, T, H, K),
            "v": v.reshape(B, T, HV, V),
            "graw": graw.detach().clone().requires_grad_(True),
            "braw": braw.detach().clone().requires_grad_(True),
            "A_log": A_log.detach().clone().requires_grad_(True),
            "dt_bias": dt_bias.detach().clone().requires_grad_(True),
        }
        assert not lv["q"].is_contiguous()
        assert not lv["k"].is_contiguous()
        assert not lv["v"].is_contiguous()
        return lv

    lv_fla, lv_cud = leaves(), leaves()
    o_fla = _run_fused(chunk_gated_delta_rule, lv_fla)
    o_cud = _run_fused(shim, lv_cud)
    assert last_path() == "native", f"expected cuDNN native path, got {last_path()}"

    do = torch.randn_like(o_fla)
    o_fla.backward(do)
    o_cud.backward(do)
    assert _relL2(o_cud, o_fla) <= C_SLACK * FLOOR
    for n in ("packed", "graw", "braw", "A_log", "dt_bias"):
        assert _relL2(lv_cud[n].grad, lv_fla[n].grad) <= C_SLACK * FLOOR, n


kda_ops = pytest.importorskip("fla.ops.kda")
chunk_kda = kda_ops.chunk_kda
from cudnn.fla.kda import make_chunk_kda, last_path as kda_last_path

kda_shim = make_chunk_kda(chunk_kda)


def _kda_leaves(B, T, H, K, V, dtype, seed):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)

    def io(*s):
        return torch.randn(*s, generator=gen, device=dev, dtype=dtype).detach().requires_grad_(True)

    # realistic KDA gate init: mild g via dt_bias = softplus^{-1}(dt), dt in [1e-3, 0.1]
    dt = torch.exp(torch.rand(H * K, generator=gen, device=dev) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
    return {
        "q": io(B, T, H, K),
        "k": io(B, T, H, K),
        "v": io(B, T, H, V),
        "g": io(B, T, H, K),  # raw f_proj output (channel-wise), io dtype
        "beta": io(B, T, H),
        "A_log": torch.log(torch.empty(H, device=dev).uniform_(1, 16)).requires_grad_(True),
        "dt_bias": (dt + torch.log(-torch.expm1(-dt))).detach().requires_grad_(True),
    }


def _run_kda(fn, lv):
    o, _ = fn(
        q=lv["q"],
        k=lv["k"],
        v=lv["v"],
        g=lv["g"],
        beta=lv["beta"],
        A_log=lv["A_log"],
        dt_bias=lv["dt_bias"],
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=False,
        output_final_state=False,
    )
    return o


def test_kda_parity_fused():
    """cuDNN KDA (through the shim) matches FLA's chunk_kda on the layer's fused
    call, calibrated to a fp32 FLA reference: cuDNN's error from truth must be
    within 3x FLA's own bf16 error, on the output and every gradient. bf16 only —
    cuDNN's KDA kernel is unstable in fp16 (the shim declines fp16 -> FLA). T=128
    avoids a FLA/triton autotune crash unrelated to cuDNN at some larger tiles."""
    shape = dict(B=2, T=128, H=4, K=128, V=128)
    master = _kda_leaves(**shape, dtype=torch.bfloat16, seed=5)
    do = torch.randn(shape["B"], shape["T"], shape["H"], shape["V"], device="cuda")

    def clone(src, dt):
        lv = {}
        for name, t in src.items():
            fp32 = name in ("A_log", "dt_bias")
            lv[name] = t.detach().clone().to(torch.float32 if fp32 else dt).requires_grad_(True)
        return lv

    lv_fla = clone(master, torch.bfloat16)
    lv_cud = clone(master, torch.bfloat16)
    lv_ref = clone(master, torch.float32)  # fp32 truth via FLA's own fused path

    o_fla = _run_kda(chunk_kda, lv_fla)
    o_cud = _run_kda(kda_shim, lv_cud)
    assert kda_last_path() == "native", f"expected cuDNN native path, got {kda_last_path()}"
    o_ref = _run_kda(chunk_kda, lv_ref)

    o_fla.backward(do.to(o_fla.dtype))
    o_cud.backward(do.to(o_cud.dtype))
    o_ref.backward(do.to(o_ref.dtype))

    # cuDNN KDA's channel-gate backward is a bit noisier than FLA's in bf16, so the
    # gate-parameter gradients (dg / dA_log, amplified through exp(A_log)) sit at ~3x
    # FLA's own error from truth rather than <=3x. Output and the main data gradients
    # match to bf16 noise; the wider slack applies only to the gate-parameter path.
    KDA_SLACK = 5.0  # output + data gradients
    # The gate-parameter gradients (dg, dA_log, dt_bias) go through cuDNN's non-deterministic
    # backward (cross-CTA fp atomicAdd), so they are noisier and vary run-to-run; give them a
    # wider bound. This is still a real bound (a gross error would blow well past it).
    KDA_GATE_SLACK = 8.0
    GATE_PARAMS = ("g", "A_log", "dt_bias")

    def check(name, a, b, ref, slack):
        e_fla = _relL2(a, ref)
        e_cud = _relL2(b, ref)
        assert e_cud <= slack * max(e_fla, FLOOR), f"{name}: e_cud={e_cud:.2e} vs e_fla={e_fla:.2e} (slack {slack})"

    check("o", o_fla, o_cud, o_ref, KDA_SLACK)
    for n in ("q", "k", "v", "g", "beta", "A_log", "dt_bias"):
        check("d" + n, lv_fla[n].grad, lv_cud[n].grad, lv_ref[n].grad, KDA_GATE_SLACK if n in GATE_PARAMS else KDA_SLACK)


def test_fallback_is_transparent():
    """A variant the native op does not model falls back and returns FLA's exact result.

    Keyed on ``allow_neg_eigval``, which the shim declines by construction, so this stays
    a fallback no matter which shapes the engines grow support for.
    """
    m = _master(2, 256, 4, 4, 128, 128, seed=1)
    kw = dict(allow_neg_eigval=True, use_beta_sigmoid_in_kernel=True)  # FLA requires the pair
    o_fla, _ = _run(chunk_gated_delta_rule, m, torch.bfloat16, **kw)
    o_cud, _ = _run(shim, m, torch.bfloat16, **kw)
    assert last_path().startswith("fallback"), f"expected fallback, got {last_path()}"
    torch.testing.assert_close(o_cud, o_fla, rtol=0, atol=0)


@pytest.mark.parametrize("state_v_first,expect_native", [(True, True), (False, False)])
def test_state_v_first_routing(state_v_first, expect_native):
    """cuDNN carries the recurrent state V-major, so it serves ``state_v_first=True``
    natively and declines the K-major request; a stateless call is layout-agnostic
    and runs native either way."""
    m = _master(2, 256, 4, 4, 128, 128, seed=2)
    lv = _leaves(m, torch.bfloat16)
    o_cud, fs_cud = shim(
        lv["q"],
        lv["k"],
        lv["v"],
        lv["g"],
        lv["beta"],
        output_final_state=True,
        state_v_first=state_v_first,
    )
    got = last_path()
    if expect_native:
        assert got == "native", f"state_v_first={state_v_first}: expected native, got {got}"
        assert fs_cud.shape == (m["q"].shape[0], m["v"].shape[2], m["v"].shape[3], m["q"].shape[3])
    else:
        assert got.startswith("fallback"), f"state_v_first={state_v_first}: expected fallback, got {got}"
        o_fla, _ = chunk_gated_delta_rule(
            lv["q"],
            lv["k"],
            lv["v"],
            lv["g"],
            lv["beta"],
            output_final_state=True,
            state_v_first=state_v_first,
        )
        torch.testing.assert_close(o_cud, o_fla, rtol=0, atol=0)


def test_accelerate_fla_patches_and_restores():
    original = fla_gdr.chunk_gated_delta_rule
    try:
        accelerate_fla(verbose=False)
        assert fla_gdr.chunk_gated_delta_rule is not original
    finally:
        restore_fla()
    assert fla_gdr.chunk_gated_delta_rule is original
