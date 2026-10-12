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


def _fused_leaves(B, T, H, HV, K, V, dtype, seed, with_dt_bias=True):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)

    def leaf(shape, dt, req=True):
        return torch.randn(*shape, generator=gen, device=dev, dtype=dt).detach().requires_grad_(req)

    lv = {
        "q": leaf((B, T, H, K), dtype),
        "k": leaf((B, T, H, K), dtype),
        "v": leaf((B, T, HV, V), dtype),
        "graw": torch.rand(B, T, HV, generator=gen, device=dev, dtype=torch.float32).requires_grad_(True),
        "braw": leaf((B, T, HV), torch.float32),
        "A_log": torch.log(torch.empty(HV, device=dev).uniform_(0.1, 16)).requires_grad_(True),
    }
    if with_dt_bias:
        lv["dt_bias"] = torch.randn(HV, generator=gen, device=dev).requires_grad_(True)
    return lv


def _run_fused(fn, lv, allow_neg_eigval=False):
    o, _ = fn(
        lv["q"],
        lv["k"],
        lv["v"],
        lv["graw"],
        lv["braw"],
        A_log=lv["A_log"],
        dt_bias=lv.get("dt_bias"),
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=allow_neg_eigval,
        use_qk_l2norm_in_kernel=True,
        output_final_state=False,
    )
    return o


@pytest.mark.parametrize("with_dt_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("allow_neg_eigval", [False, True], ids=["sigmoid", "neg_eigval"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_parity_fused_layer_path(dtype, allow_neg_eigval, with_dt_bias):
    """The FLA GatedDeltaNet layer's actual call (raw g/beta + A_log/dt_bias +
    in-kernel L2-norm/gate/beta fusion, with and without the ``2 * sigmoid`` beta,
    with and without ``dt_bias``): the shim forwards the fusions to the native
    kernel and must still match FLA within its own noise (truth = FLA fused in fp32)."""
    shape = dict(B=2, T=256, H=4, HV=4, K=128, V=128)
    do = torch.randn(shape["B"], shape["T"], shape["HV"], shape["V"], device="cuda")

    def clone_to(src, dt):
        lv = {}
        for name, t in src.items():
            keep_fp32 = name in ("graw", "braw", "A_log", "dt_bias")
            lv[name] = t.detach().clone().to(torch.float32 if keep_fp32 else dt).requires_grad_(True)
        return lv

    master = _fused_leaves(**shape, dtype=dtype, seed=3, with_dt_bias=with_dt_bias)
    lv_fla = clone_to(master, dtype)
    lv_cud = clone_to(master, dtype)
    lv_ref = clone_to(master, torch.float32)

    o_fla = _run_fused(chunk_gated_delta_rule, lv_fla, allow_neg_eigval)
    o_cud = _run_fused(shim, lv_cud, allow_neg_eigval)
    assert last_path() == "native", f"expected cuDNN native path, got {last_path()}"
    o_ref = _run_fused(chunk_gated_delta_rule, lv_ref, allow_neg_eigval)

    o_fla.backward(do.to(o_fla.dtype))
    o_cud.backward(do.to(o_cud.dtype))
    o_ref.backward(do.to(o_ref.dtype))

    def check(name, a, b, ref):
        e_fla = _relL2(a, ref)
        e_cud = _relL2(b, ref)
        assert e_cud <= C_SLACK * max(e_fla, FLOOR), f"{name}: e_cud={e_cud:.2e} vs e_fla={e_fla:.2e}"

    check("o", o_fla, o_cud, o_ref)
    for n in master:
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


def _kda_leaves(B, T, H, HV, K, V, dtype, seed, with_a_log=True, with_dt_bias=True, mild_decay=False):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)

    def io(*s, scale=1.0):
        return (torch.randn(*s, generator=gen, device=dev, dtype=dtype) * scale).detach().requires_grad_(True)

    dt = torch.exp(torch.rand(HV * K, generator=gen, device=dev) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
    amplitude = (0.1, 0.3) if mild_decay else (1, 16)
    lv = {
        "q": io(B, T, H, K),
        "k": io(B, T, H, K),
        "v": io(B, T, HV, V),
        "g": io(B, T, HV, K, scale=0.1 if mild_decay else 1.0),
        "beta": torch.randn(B, T, HV, generator=gen, device=dev).requires_grad_(True),
    }
    if with_dt_bias:
        lv["dt_bias"] = (dt + torch.log(-torch.expm1(-dt))).detach().requires_grad_(True)
    if with_a_log:
        lv["A_log"] = torch.log(torch.empty(HV, device=dev).uniform_(*amplitude)).requires_grad_(True)
    return lv


KDA_GATES = {
    "softplus": dict(safe_gate=False, lower_bound=None),
    "lower_bound": dict(safe_gate=True, lower_bound=-5.0),
    "lower_bound_no_a_log": dict(safe_gate=False, lower_bound=-5.0),
    "lower_bound_no_dt_bias": dict(safe_gate=True, lower_bound=-5.0),
    "lower_bound_no_params": dict(safe_gate=False, lower_bound=-5.0),
}


def _run_kda(fn, lv, gate, allow_neg_eigval=False):
    o, _ = fn(
        q=lv["q"],
        k=lv["k"],
        v=lv["v"],
        g=lv["g"],
        beta=lv["beta"],
        A_log=lv.get("A_log"),
        dt_bias=lv.get("dt_bias"),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=allow_neg_eigval,
        output_final_state=False,
        **KDA_GATES[gate],
    )
    return o


@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(dict(gate="softplus", allow_neg_eigval=False, H=4, HV=4), id="softplus"),
        pytest.param(dict(gate="softplus", allow_neg_eigval=True, H=4, HV=4), id="softplus-neg_eigval"),
        pytest.param(dict(gate="lower_bound", allow_neg_eigval=False, H=4, HV=4), id="lower_bound"),
        pytest.param(dict(gate="lower_bound", allow_neg_eigval=True, H=4, HV=4), id="lower_bound-neg_eigval"),
        pytest.param(dict(gate="lower_bound_no_a_log", allow_neg_eigval=False, H=4, HV=4), id="lower_bound_no_a_log"),
        pytest.param(dict(gate="lower_bound_no_dt_bias", allow_neg_eigval=False, H=4, HV=4), id="lower_bound_no_dt_bias"),
        pytest.param(dict(gate="lower_bound_no_params", allow_neg_eigval=False, H=4, HV=4), id="lower_bound_no_params"),
        pytest.param(dict(gate="lower_bound", allow_neg_eigval=False, H=2, HV=4), id="lower_bound-gva"),
        pytest.param(dict(gate="softplus", allow_neg_eigval=False, H=4, HV=4, dtype=torch.float16, mild_decay=True), id="softplus-fp16"),
    ],
)
def test_kda_parity_fused(cfg):
    """cuDNN KDA (through the shim) matches FLA's chunk_kda on the layer's fused
    call, calibrated to a fp32 FLA reference: cuDNN's error from truth must be
    within a fixed factor of FLA's own 16-bit error, on the output and every
    gradient. Covers both FLA gate transforms (softplus and the lower-bounded
    sigmoid, with and without ``A_log`` / ``dt_bias``), the ``2 * sigmoid`` beta, grouped
    value heads, and fp16 io at a per-token decay the fp16 range represents over
    a chunk. T=128 avoids a FLA/triton autotune crash unrelated to cuDNN at some
    larger tiles."""
    shape = dict(B=2, T=128, H=cfg["H"], HV=cfg["HV"], K=128, V=128)
    dtype = cfg.get("dtype", torch.bfloat16)
    master = _kda_leaves(
        **shape,
        dtype=dtype,
        seed=5,
        with_a_log=cfg["gate"] not in ("lower_bound_no_a_log", "lower_bound_no_params"),
        with_dt_bias=cfg["gate"] not in ("lower_bound_no_dt_bias", "lower_bound_no_params"),
        mild_decay=cfg.get("mild_decay", False),
    )
    do = torch.randn(shape["B"], shape["T"], shape["HV"], shape["V"], device="cuda")

    def clone(src, dt):
        lv = {}
        for name, t in src.items():
            fp32 = name in ("beta", "A_log", "dt_bias")
            lv[name] = t.detach().clone().to(torch.float32 if fp32 else dt).requires_grad_(True)
        return lv

    lv_fla = clone(master, dtype)
    lv_cud = clone(master, dtype)
    lv_ref = clone(master, torch.float32)

    o_fla = _run_kda(chunk_kda, lv_fla, cfg["gate"], cfg["allow_neg_eigval"])
    o_cud = _run_kda(kda_shim, lv_cud, cfg["gate"], cfg["allow_neg_eigval"])
    assert kda_last_path() == "native", f"expected cuDNN native path, got {kda_last_path()}"
    o_ref = _run_kda(chunk_kda, lv_ref, cfg["gate"], cfg["allow_neg_eigval"])

    o_fla.backward(do.to(o_fla.dtype))
    o_cud.backward(do.to(o_cud.dtype))
    o_ref.backward(do.to(o_ref.dtype))

    KDA_SLACK = 5.0
    KDA_GATE_SLACK = 8.0
    GATE_PARAMS = ("g", "A_log", "dt_bias")

    def check(name, a, b, ref, slack):
        e_fla = _relL2(a, ref)
        e_cud = _relL2(b, ref)
        assert e_cud <= slack * max(e_fla, FLOOR), f"{name}: e_cud={e_cud:.2e} vs e_fla={e_fla:.2e} (slack {slack})"

    check("o", o_fla, o_cud, o_ref, KDA_SLACK)
    for n in master:
        check("d" + n, lv_fla[n].grad, lv_cud[n].grad, lv_ref[n].grad, KDA_GATE_SLACK if n in GATE_PARAMS else KDA_SLACK)
    assert lv_cud["beta"].grad.dtype == torch.float32


def test_fallback_is_transparent():
    """A config the native engine does not serve falls back and returns FLA's exact result.

    Keyed on K = 256: the FROST GDN engine takes head dims 64 and 128 only and the
    shim pins its plans to FROST, so this stays a fallback whatever the other engines
    grow support for.
    """
    m = _master(2, 256, 4, 4, 256, 128, seed=1)
    o_fla, _ = _run(chunk_gated_delta_rule, m, torch.bfloat16)
    o_cud, _ = _run(shim, m, torch.bfloat16)
    assert last_path().startswith("fallback"), f"expected fallback, got {last_path()}"
    torch.testing.assert_close(o_cud, o_fla, rtol=0, atol=0)


@pytest.mark.parametrize("state_v_first", [True, False], ids=["v_major", "k_major"])
@pytest.mark.parametrize("with_initial_state", [False, True], ids=["zero_state", "initial_state"])
def test_state_layouts(state_v_first, with_initial_state):
    """The recurrent state is exchanged in FLA's layout for both ``state_v_first``
    settings, with and without an incoming state, on the native path; the output,
    the final state and the state gradient match FLA within its own noise."""
    B, T, H, HV, K, V = 2, 256, 4, 4, 128, 128
    m = _master(B, T, H, HV, K, V, seed=2)
    state_shape = (B, HV, V, K) if state_v_first else (B, HV, K, V)
    h0_master = torch.randn(*state_shape, device="cuda") if with_initial_state else None
    do = torch.randn(B, T, HV, V, device="cuda")
    dfs = torch.randn(*state_shape, device="cuda")

    def run(fn):
        lv = _leaves(m, torch.bfloat16)
        h0 = None if h0_master is None else h0_master.detach().clone().requires_grad_(True)
        o, fs = fn(lv["q"], lv["k"], lv["v"], lv["g"], lv["beta"], initial_state=h0, output_final_state=True, state_v_first=state_v_first)
        (o.float() * do.to(o.dtype)).sum().add((fs * dfs).sum()).backward()
        return o, fs, lv, h0

    o_fla, fs_fla, lv_fla, h0_fla = run(chunk_gated_delta_rule)
    o_cud, fs_cud, lv_cud, h0_cud = run(shim)
    assert last_path() == "native", f"expected native, got {last_path()}"
    assert fs_cud.shape == fs_fla.shape == state_shape
    assert fs_cud.dtype == fs_fla.dtype == torch.float32
    assert _relL2(o_cud, o_fla) <= C_SLACK * FLOOR
    assert _relL2(fs_cud, fs_fla) <= C_SLACK * FLOOR
    for n in ("q", "k", "v", "g", "beta"):
        assert _relL2(lv_cud[n].grad, lv_fla[n].grad) <= C_SLACK * FLOOR, n
    if with_initial_state:
        assert _relL2(h0_cud.grad, h0_fla.grad) <= C_SLACK * FLOOR


@pytest.mark.parametrize("state_v_first", [True, False], ids=["v_major", "k_major"])
@pytest.mark.parametrize("layout", ["dense", "varlen"])
def test_kda_intermediate_states(layout, state_v_first):
    """``return_intermediate_states`` (FLA's inference path) is served from cuDNN's
    per-chunk state series: same chunking (64 tokens, state before each chunk,
    ``ceil(len / 64)`` entries per sequence packed in order), same layout as
    FLA's ``h`` for both state layouts, same dtype; the output and the final
    state match FLA within its own noise."""
    H, K, V = 4, 128, 128
    if layout == "dense":
        B, T, cu, n_chunks = 2, 200, None, 4
    else:
        seq_lens = [96, 32, 160]
        B, T, cu, n_chunks = 1, sum(seq_lens), torch.tensor([0, 96, 128, 288], dtype=torch.long, device="cuda"), 2 + 1 + 3
    gen = torch.Generator(device="cuda").manual_seed(7)
    io = lambda *shape: torch.randn(*shape, generator=gen, device="cuda", dtype=torch.bfloat16)
    q, k, v = io(B, T, H, K), io(B, T, H, K), io(B, T, H, V)
    g = -torch.rand(B, T, H, K, generator=gen, device="cuda").mul(0.2)
    beta = torch.rand(B, T, H, generator=gen, device="cuda")
    n_seq = B if cu is None else len(cu) - 1
    h0 = torch.randn(n_seq, H, *((V, K) if state_v_first else (K, V)), generator=gen, device="cuda")

    def run(fn):
        with torch.inference_mode():
            return fn(
                q,
                k,
                v,
                g,
                beta,
                initial_state=h0,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                return_intermediate_states=True,
                state_v_first=state_v_first,
                cu_seqlens=cu,
            )

    o_fla, fs_fla, h_fla = run(chunk_kda)
    o_cud, fs_cud, h_cud = run(kda_shim)
    assert kda_last_path() == "native", f"expected native, got {kda_last_path()}"
    assert h_fla.shape == (B, n_chunks, H) + ((V, K) if state_v_first else (K, V))
    assert h_cud.shape == h_fla.shape and h_cud.dtype == h_fla.dtype
    assert fs_cud.shape == fs_fla.shape
    assert _relL2(o_cud, o_fla) <= C_SLACK * FLOOR
    assert _relL2(fs_cud, fs_fla) <= C_SLACK * FLOOR
    assert _relL2(h_cud, h_fla) <= C_SLACK * FLOOR
    expect0 = (h0 if cu is None else h0[:1]).to(h_cud.dtype)
    torch.testing.assert_close(h_cud[:, 0], expect0, rtol=0, atol=0)
    torch.testing.assert_close(h_fla[:, 0], expect0, rtol=0, atol=0)


def test_accelerate_fla_patches_and_restores():
    original = fla_gdr.chunk_gated_delta_rule
    try:
        accelerate_fla(verbose=False)
        assert fla_gdr.chunk_gated_delta_rule is not original
    finally:
        restore_fla()
    assert fla_gdr.chunk_gated_delta_rule is original
