# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm100``: the SM100 large-head-dim backward chain.

Every capability the row claims gets an ACCEPT test that runs the kernel against
a torch reference, and every capability it declines gets a REJECT test that
asserts the decline -- per the engine contract, an unsupported feature is
asserted, never skipped, so a row that quietly grows a capability fails here.

The engine is a three-stage chain (do_dot -> S/dS workspace -> three GEMMs), so
these are end-to-end graph-API tests: a unit test of one stage would not catch
the seams, which is where every bug in this kernel has actually lived.
"""

from __future__ import annotations

import math

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

import cudnn

pytestmark = [pytest.mark.L0, requires_pre_rubin_blackwell, requires_dsl]

_ENGINE = "sdpa_bwd_sm100"
_D = 512
_TOL_COS = 0.9999
_TOL_REL = 2e-2

# The row claims HALF and BFLOAT16. Both get run: the two differ only by one
# ternary in the adapter (`dtype_code`), which is exactly the kind of line that
# is easy to leave pointing at the wrong template.
_DTYPES = (torch.bfloat16, torch.float16)
_DTYPE_IDS = ("bf16", "fp16")


def test_prepared_chain_codegen_targets(monkeypatch):
    import cutlass
    from cudnn.sdpa.bwd.kernels.sm100 import prepared_host

    calls = []

    def fake_compile(*args, **kwargs):
        calls.append(kwargs["options"])
        return object()

    monkeypatch.setattr(prepared_host, "compile_cached", fake_compile)
    params = prepared_host.Params(1, 2, 2, 512, 128, 128, 256, 128, 2, False, False, 0, 256)
    for sm in (100, 103, 107, 110):
        prepared_host.compile_host(None, None, None, params, (), (), cutlass.BFloat16, sm, "target-probe")
    with pytest.raises(ValueError, match="got SM101"):
        prepared_host.compile_host(None, None, None, params, (), (), cutlass.BFloat16, 101, "target-probe")
    assert calls == [f"--enable-tvm-ffi --gpu-arch sm_{sm}a" for sm in (100, 103, 107, 110)]


def _io_dtype(dt):
    return cudnn.data_type.HALF if dt == torch.float16 else cudnn.data_type.BFLOAT16


def _bshd_stride(shape):
    """cuDNN declares logical BHSD; the engine needs BSHD-physical storage."""
    b, h, s, d = shape
    return [s * h * d, d, h * d, 1]


def _bshd(b, s, h, d, dev="cuda", dt=torch.bfloat16, fill=True):
    """A [B, H, S, D] view over BSHD memory -- what the engine expects."""
    t = torch.randn(b, s, h, d, device=dev, dtype=dt) if fill else torch.zeros(b, s, h, d, device=dev, dtype=dt)
    return t.mul_(0.1).permute(0, 2, 1, 3) if fill else t.permute(0, 2, 1, 3)


def _reference(q, k, v, do, keep=None, group=1):
    """fp32 attention backward. ``keep`` is a [S_q, S_kv] bool mask."""
    kx = k.repeat_interleave(group, dim=1) if group > 1 else k
    vx = v.repeat_interleave(group, dim=1) if group > 1 else v
    scale = 1.0 / math.sqrt(q.shape[3])
    sa = (q.float() @ kx.float().transpose(-1, -2)) * scale
    if keep is not None:
        sa = sa.masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(sa, dim=-1)
    S = torch.exp(sa - lse.unsqueeze(-1)).nan_to_num_(0.0)
    o = S @ vx.float()
    dd = (o * do.float()).sum(-1)
    dS = scale * (do.float() @ vx.float().transpose(-1, -2) - dd.unsqueeze(-1)) * S
    dq = dS @ kx.float()
    dk_q = dS.transpose(-1, -2) @ q.float()
    dv_q = S.transpose(-1, -2) @ do.float()
    if group > 1:
        hkv = k.shape[1]
        dk_q = dk_q.view(dk_q.shape[0], hkv, group, dk_q.shape[2], dk_q.shape[3]).sum(2)
        dv_q = dv_q.view(dv_q.shape[0], hkv, group, dv_q.shape[2], dv_q.shape[3]).sum(2)
    all_masked = (~keep.any(-1)) if keep is not None else None
    return o, lse, all_masked, dq, dk_q, dv_q


def _build_graph(b, hq, hkv, sq, skv, d, scale, dt=torch.bfloat16, **sdpa_kwargs):
    io = _io_dtype(dt)
    g = cudnn.pygraph(
        io_data_type=io,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    shq, shk = [b, hq, sq, d], [b, hkv, skv, d]
    t = {n: g.tensor(name=n, dim=sh, stride=_bshd_stride(sh)) for n, sh in (("q", shq), ("k", shk), ("v", shk), ("o", shq), ("do", shq))}
    t["stats"] = g.tensor(name="stats", dim=[b, hq, sq, 1], stride=[hq * sq, sq, 1, 1], data_type=cudnn.data_type.FLOAT)
    # scale=None omits attn_scale entirely -- it is optional on the graph.
    if scale is not None:
        sdpa_kwargs["attn_scale"] = scale
    dq, dk, dv = g.sdpa_backward(name="bwd", q=t["q"], k=t["k"], v=t["v"], o=t["o"], dO=t["do"], stats=t["stats"], **sdpa_kwargs)
    for out, sh in ((dq, shq), (dk, shk), (dv, shk)):
        out.set_output(True).set_data_type(io).set_stride(_bshd_stride(sh))
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    return g, t, (dq, dk, dv)


def _plan_index(g, name=_ENGINE):
    for i in range(g.get_execution_plan_count()):
        if name in g.get_plan_name_at_index(i):
            return i
    return None


def _run(b=2, hq=2, hkv=None, sq=512, skv=512, d=_D, keep=None, dt=torch.bfloat16, omit_scale=False, **sdpa_kwargs):
    """Build, pin the engine, execute, and compare against fp32 torch."""
    hkv = hq if hkv is None else hkv
    group = hq // hkv
    # None => omit attn_scale on the graph; the engine must then default it to
    # 1/sqrt(d), which is what _reference assumes either way.
    scale = None if omit_scale else 1.0 / math.sqrt(d)
    q, do = _bshd(b, sq, hq, d, dt=dt), _bshd(b, sq, hq, d, dt=dt)
    k, v = _bshd(b, skv, hkv, d, dt=dt), _bshd(b, skv, hkv, d, dt=dt)
    o_ref, lse, all_masked, dq_r, dk_r, dv_r = _reference(q, k, v, do, keep, group)
    o = _bshd(b, sq, hq, d, dt=dt, fill=False)
    o.copy_(o_ref.to(dt))

    g, t, (dq_t, dk_t, dv_t) = _build_graph(b, hq, hkv, sq, skv, d, scale, dt=dt, **sdpa_kwargs)
    idx = _plan_index(g)
    assert idx is not None, f"{_ENGINE} not offered; plans = {[g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]}"
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    dq, dk, dv = _bshd(b, sq, hq, d, dt=dt, fill=False), _bshd(b, skv, hkv, d, dt=dt, fill=False), _bshd(b, skv, hkv, d, dt=dt, fill=False)
    stats = (lse if all_masked is None else lse.masked_fill(all_masked, 0.0)).unsqueeze(-1).contiguous()
    g.execute(
        {t["q"]: q, t["k"]: k, t["v"]: v, t["o"]: o, t["do"]: do, t["stats"]: stats, dq_t: dq, dk_t: dk, dv_t: dv},
        ws,
    )
    torch.cuda.synchronize()
    for name, got, ref in (("dQ", dq, dq_r), ("dK", dk, dk_r), ("dV", dv, dv_r)):
        cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.flatten(), dim=0).item()
        rel = ((got.float() - ref).abs().max() / max(ref.abs().max().item(), 1e-30)).item()
        assert cos > _TOL_COS and rel < _TOL_REL, f"{name}: cos={cos:.6f} max_rel_err={rel:.2e}"


def _causal_keep(sq, skv, dev="cuda", bottom_right=False, left=None, right=0):
    qi = torch.arange(sq, device=dev).view(-1, 1)
    ki = torch.arange(skv, device=dev).view(1, -1)
    diag = (skv - sq) if bottom_right else 0
    keep = ki <= qi + diag + right
    if left is not None:
        keep &= ki >= qi + diag - (left - 1)
    return keep


# --------------------------------------------------------------------------- #
# ACCEPT — every capability the row claims                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_dense(dt):
    _run(dt=dt)


@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_causal_dtypes(dt):
    """Both dtypes through the masked path too: the causal chain reads the
    workspace back in the io dtype, so a dtype mix-up shows up here and not in
    the dense case."""
    _run(dt=dt, keep=_causal_keep(512, 512), use_causal_mask=True)


@pytest.mark.parametrize("d", [264, 320, 384, 511 - 7, _D])
def test_head_dim_band(d):
    """d in (256, 512], any multiple of 8. 264 and 504 are not multiples of 16,
    which narrows the stage-3 epilogue store vector from 32 B to 16 B."""
    _run(sq=256, skv=256, d=d)


@pytest.mark.parametrize("hq,hkv", [(8, 8), (8, 4), (8, 2), (8, 1), (6, 3)])
def test_gqa_mqa(hq, hkv):
    _run(hq=hq, hkv=hkv, sq=256, skv=256)


def test_causal_top_left():
    _run(keep=_causal_keep(512, 512), use_causal_mask=True)


def test_causal_bottom_right():
    _run(keep=_causal_keep(512, 512, bottom_right=True), use_causal_mask_bottom_right=True)


def test_causal_bottom_right_rectangular():
    """S_kv > S_q shifts the diagonal, which the stage-3 K-trim has to follow."""
    _run(sq=512, skv=1024, keep=_causal_keep(512, 1024, bottom_right=True), use_causal_mask_bottom_right=True)


def test_sliding_window():
    _run(keep=_causal_keep(512, 512, left=256), use_causal_mask=True, diagonal_band_left_bound=256)


def test_right_band_widening():
    """`diagonal_band_right_bound` alone -- passing use_causal_mask with it
    forces the bound back to 0 and the widening is silently dropped."""
    _run(keep=_causal_keep(512, 512, right=64), diagonal_band_right_bound=64)


@pytest.mark.parametrize("sq,skv", [(500, 500), (300, 200), (257, 129), (384, 640)])
def test_non_tile_multiple_seqlens(sq, skv):
    """Neither S_q nor S_kv has to be a tile multiple: the compile shape rounds
    up and the tail is masked."""
    _run(sq=sq, skv=skv)


@pytest.mark.parametrize("sq,skv", [(500, 500), (1000, 1000)])
def test_non_tile_multiple_causal(sq, skv):
    _run(sq=sq, skv=skv, keep=_causal_keep(sq, skv), use_causal_mask=True)


def test_default_attn_scale():
    """attn_scale is OPTIONAL on the graph; omitting it must mean 1/sqrt(d).

    The adapter used to leave `scale_softmax` at None and die in execute with
    `TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'`,
    after the row had already admitted the graph and check_support had passed.
    _reference always uses 1/sqrt(d), so a wrong default fails the comparison
    rather than merely not raising.
    """
    _run(omit_scale=True)


def test_reject_rectangular_head_dims():
    """d_qk != d_v is declined: the kernel asserts d_qk == d_v.

    The C++ node validation's d512 exception is deliberately permissive here
    (it admits any pair in the band), so `mismatch()` is the only thing keeping
    a rectangular graph away from an adapter that would raise on it.
    """
    assert _decline_reason(d=_D) is None, "sanity: the square case must be served"
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    b, hq, sq, skv = 2, 2, 256, 256
    shq, shk = [b, hq, sq, 384], [b, hq, skv, 384]
    shv = [b, hq, skv, 320]
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    t = {
        n: g.tensor(name=n, dim=sh, stride=_bshd_stride(sh))
        for n, sh in (("q", shq), ("k", shk), ("v", shv), ("o", [b, hq, sq, 320]), ("do", [b, hq, sq, 320]))
    }
    st = g.tensor(name="stats", dim=[b, hq, sq, 1], stride=[hq * sq, sq, 1, 1], data_type=cudnn.data_type.FLOAT)
    try:
        dq, dk, dv = g.sdpa_backward(name="bwd", q=t["q"], k=t["k"], v=t["v"], o=t["o"], dO=t["do"], stats=st, attn_scale=1.0 / math.sqrt(384))
        for out, sh in ((dq, shq), (dk, shk), (dv, shv)):
            out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_stride(_bshd_stride(sh))
        g.validate()
        g.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError:
        return  # refused before engine selection is also a decline
    facts = ga.analyze(g)
    spec = next(s_ for s_ in ENGINE_SPECS if s_.name == _ENGINE)
    assert facts is None or mismatch(spec.capabilities, facts) is not None


def test_non_bshd_do_is_native(monkeypatch):
    """A BHSD-contiguous dO is addressed natively without workspace copies.

    This is the exact shape benchmark_single_sdpa produces: building dO with
    torch.randn(o.shape) instead of torch.empty_like(o) loses o's memory format.
    The stride is DECLARED as BHSD here -- declaring BSHD over a BHSD buffer is
    simply a lie to the engine, and it would then (correctly) read the wrong
    elements.
    """
    b, hq, sq, skv, d = 2, 2, 256, 256, _D
    scale = 1.0 / math.sqrt(d)
    q, k, v = _bshd(b, sq, hq, d), _bshd(b, skv, hq, d), _bshd(b, skv, hq, d)
    do = torch.randn(b, hq, sq, d, device="cuda", dtype=torch.bfloat16).mul_(0.1)  # BHSD-contiguous
    bhsd_stride = [hq * sq * d, sq * d, d, 1]
    assert tuple(do.stride()) == tuple(bhsd_stride)

    o_ref, lse, _, dq_r, _, _ = _reference(q, k, v, do)
    o = _bshd(b, sq, hq, d, fill=False)
    o.copy_(o_ref.to(torch.bfloat16))

    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    shq, shk = [b, hq, sq, d], [b, hq, skv, d]
    t = {n: g.tensor(name=n, dim=sh, stride=_bshd_stride(sh)) for n, sh in (("q", shq), ("k", shk), ("v", shk), ("o", shq))}
    t["do"] = g.tensor(name="do", dim=shq, stride=bhsd_stride)  # the odd one out
    t["stats"] = g.tensor(name="stats", dim=[b, hq, sq, 1], stride=[hq * sq, sq, 1, 1], data_type=cudnn.data_type.FLOAT)
    dq_t, dk_t, dv_t = g.sdpa_backward(name="bwd", q=t["q"], k=t["k"], v=t["v"], o=t["o"], dO=t["do"], stats=t["stats"], attn_scale=scale)
    for out, sh in ((dq_t, shq), (dk_t, shk), (dv_t, shk)):
        out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_stride(_bshd_stride(sh))
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    idx = _plan_index(g)
    assert idx is not None, "a BHSD dO must be served natively"
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    dq, dk, dv = (_bshd(b, s_, hq, d, fill=False) for s_ in (sq, skv, skv))
    assert g._compiled_plans[g._plan_index]._prepared is not None
    monkeypatch.setattr(torch.Tensor, "copy_", lambda *a, **k: pytest.fail("prepared backward must address declared dO strides without copying"))
    g.execute(
        {t["q"]: q, t["k"]: k, t["v"]: v, t["o"]: o, t["do"]: do, t["stats"]: lse.unsqueeze(-1).contiguous(), dq_t: dq, dk_t: dk, dv_t: dv},
        ws,
    )
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(dq.float().flatten(), dq_r.flatten(), dim=0).item()
    assert cos > _TOL_COS, f"dQ cos={cos:.6f}"


def test_workspace_is_build_time_honest():
    """get_workspace_size() must be a pure function of the shape, not grow at
    execute -- that is what makes the plan CUDA-graph friendly."""
    b, hq, sq, skv = 2, 2, 256, 256
    g, _, _ = _build_graph(b, hq, hq, sq, skv, _D, 1.0 / math.sqrt(_D))
    idx = _plan_index(g)
    assert idx is not None
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    assert g.get_workspace_size() == g.get_workspace_size()
    assert g.get_workspace_size() > 0


# --------------------------------------------------------------------------- #
# REJECT — asserted, never skipped                                             #
# --------------------------------------------------------------------------- #


def _decline_reason(**kw):
    """Why this engine declines the graph, or None if it would serve it.

    Asks the row's own ``mismatch()`` rather than walking the ranked plan list:
    the plan list also contains BACKEND plans, so "my engine is absent" there is
    confounded by whatever the backend does (it serves d=128/256 backward, and
    it raises outright on some graphs). This tests exactly the contract -- the
    Capabilities row rejecting the facts -- and nothing else.
    """
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    b, hq, hkv = kw.pop("b", 2), kw.pop("hq", 2), kw.pop("hkv", 2)
    sq, skv, d = kw.pop("sq", 256), kw.pop("skv", 256), kw.pop("d", _D)
    spec = next(s for s in ENGINE_SPECS if s.name == _ENGINE)
    try:
        g = _build_graph_only(b, hq, hkv, sq, skv, d, 1.0 / math.sqrt(d), **kw)
    except cudnn.cudnnGraphNotSupportedError as e:
        return f"frontend refused the graph: {e}"
    facts = ga.analyze(g)
    if facts is None:
        return "analyzer did not recognise the graph"
    return mismatch(spec.capabilities, facts)


def _build_graph_only(b, hq, hkv, sq, skv, d, scale, stats_batch_stride=None, **sdpa_kwargs):
    """_build_graph without create_execution_plans (which would involve the backend)."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    shq, shk = [b, hq, sq, d], [b, hkv, skv, d]
    t = {n: g.tensor(name=n, dim=sh, stride=_bshd_stride(sh)) for n, sh in (("q", shq), ("k", shk), ("v", shk), ("o", shq), ("do", shq))}
    t["stats"] = g.tensor(
        name="stats", dim=[b, hq, sq, 1], stride=[hq * sq if stats_batch_stride is None else stats_batch_stride, sq, 1, 1], data_type=cudnn.data_type.FLOAT
    )
    dq, dk, dv = g.sdpa_backward(name="bwd", q=t["q"], k=t["k"], v=t["v"], o=t["o"], dO=t["do"], stats=t["stats"], attn_scale=scale, **sdpa_kwargs)
    for out, sh in ((dq, shq), (dk, shk), (dv, shk)):
        out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_stride(_bshd_stride(sh))
    g.validate()
    g.build_operation_graph()
    return g


@pytest.mark.parametrize("d", [128, 256])
def test_reject_head_dim_at_or_below_256(d):
    """The d256 flavors own these; routing them here would pad by >2x."""
    assert _decline_reason(d=d) is not None


def test_reject_head_dim_not_multiple_of_8():
    """TMA's innermost extent must be 16-byte aligned; at 2 B/elem that is d%8."""
    assert _decline_reason(d=260) is not None


def test_reject_head_dim_above_512():
    assert _decline_reason(d=576) is not None


def test_reject_gqa_ratio_not_integer():
    assert _decline_reason(hq=6, hkv=4) is not None


def test_reject_bias():
    """A bias input is not implemented; the row must not claim it."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    b, hq, sq, skv = 2, 2, 256, 256
    shq = [b, hq, sq, _D]
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    t = {n: g.tensor(name=n, dim=shq, stride=_bshd_stride(shq)) for n in ("q", "k", "v", "o", "do")}
    st = g.tensor(name="stats", dim=[b, hq, sq, 1], stride=[hq * sq, sq, 1, 1], data_type=cudnn.data_type.FLOAT)
    bias = g.tensor(name="bias", dim=[1, 1, sq, skv], stride=[sq * skv, sq * skv, skv, 1])
    try:
        dq, dk, dv = g.sdpa_backward(name="bwd", q=t["q"], k=t["k"], v=t["v"], o=t["o"], dO=t["do"], stats=st, bias=bias, attn_scale=1.0 / math.sqrt(_D))
        for out in (dq, dk, dv):
            out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_stride(_bshd_stride(shq))
        g.validate()
        g.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError:
        return  # refused before engine selection is also a decline
    facts = ga.analyze(g)
    spec = next(s_ for s_ in ENGINE_SPECS if s_.name == _ENGINE)
    assert facts is None or mismatch(spec.capabilities, facts) is not None


def test_reject_deterministic():
    assert _decline_reason(use_deterministic_algorithm=True) is not None


def test_reject_padding_mask():
    """Per-batch seq_len is declined: the kernel threads a scalar length.

    Asserted on a REAL graph, not just on the Capabilities field, because the
    decline has to survive the analyzer too. When per-batch lengths land, this
    test inverts (assert the decline is None) rather than being deleted.
    """
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    b, hq, sq, skv = 2, 2, 256, 256
    shq = [b, hq, sq, _D]
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    t = {n: g.tensor(name=n, dim=shq, stride=_bshd_stride(shq)) for n in ("q", "k", "v", "o", "do")}
    st = g.tensor(name="stats", dim=[b, hq, sq, 1], stride=[hq * sq, sq, 1, 1], data_type=cudnn.data_type.FLOAT)
    slq = g.tensor(name="seq_len_q", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    slk = g.tensor(name="seq_len_kv", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    try:
        dq, dk, dv = g.sdpa_backward(
            name="bwd",
            q=t["q"],
            k=t["k"],
            v=t["v"],
            o=t["o"],
            dO=t["do"],
            stats=st,
            attn_scale=1.0 / math.sqrt(_D),
            use_padding_mask=True,
            seq_len_q=slq,
            seq_len_kv=slk,
        )
        for out in (dq, dk, dv):
            out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_stride(_bshd_stride(shq))
        g.validate()
        g.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError:
        return  # refused before engine selection is also a decline
    facts = ga.analyze(g)
    spec = next(s_ for s_ in ENGINE_SPECS if s_.name == _ENGINE)
    assert facts is None or mismatch(spec.capabilities, facts) is not None


def test_accept_thd():
    """THD/ragged is SERVED: the packed path with a row-blocked S/dS workspace.

    Same shape as the forward's THD test -- packed data behind dense-sized
    descriptors plus a per-operand ragged_offset. This was ``test_reject_thd``
    until the packed lowering landed; it is inverted rather than deleted so the
    claim keeps a test on this side of the row too (test/AGENTS.md). The
    numerics and every THD conjunction live in ``test_sdpa_bwd_thd_sm100.py``.

    Note the graph declares ``max_total_seq_len_q/kv``: the row REQUIRES them
    (the blocked workspace is sized at build time), and a graph without them is
    declined -- asserted by ``test_reject_thd_without_declared_totals`` there.
    """
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    b, hq, s_max = 2, 2, 256
    shq = [b, hq, s_max, _D]
    stride = [s_max * hq * _D, _D, hq * _D, 1]
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    t = {n: g.tensor(name=n, dim=shq, stride=stride) for n in ("q", "k", "v", "o", "do")}
    # Stats is RAGGED too, token-major (stride_h == 1, stride_s == H_q): the
    # packed path declines a dense per-batch Stats, whose stride would read as
    # head-major over per-batch rectangles (test_reject_thd_dense_stats).
    st = g.tensor(name="stats", dim=[b, hq, s_max, 1], stride=[s_max * hq, 1, hq, 1], data_type=cudnn.data_type.FLOAT)
    st.set_ragged_offset(g.tensor(name="stats_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64))
    slq = g.tensor(name="seq_len_q", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    slk = g.tensor(name="seq_len_kv", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    for n in ("q", "k", "v", "o", "do"):
        t[n].set_ragged_offset(g.tensor(name=f"{n}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64))
    try:
        dq, dk, dv = g.sdpa_backward(
            name="bwd",
            q=t["q"],
            k=t["k"],
            v=t["v"],
            o=t["o"],
            dO=t["do"],
            stats=st,
            attn_scale=1.0 / math.sqrt(_D),
            use_padding_mask=True,
            seq_len_q=slq,
            seq_len_kv=slk,
            max_total_seq_len_q=b * s_max,
            max_total_seq_len_kv=b * s_max,
        )
        for out in (dq, dk, dv):
            out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_stride(stride)
            out.set_ragged_offset(g.tensor(name=f"{out.get_name()}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64))
        g.validate()
        g.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError as exc:
        pytest.fail(f"the row claims THD, but the node refused the graph: {exc}")
    facts = ga.analyze(g)
    spec = next(s_ for s_ in ENGINE_SPECS if s_.name == _ENGINE)
    assert facts is not None
    assert mismatch(spec.capabilities, facts) is None


def test_unserved_band_graph_declines_as_not_supported():
    """An unserved d512 backward must raise cudnnGraphNotSupportedError.

    Regression test for the error TYPE, not the message. The C++ node admits
    d in (256, 512] so the FROST engine can claim it, but a graph in the band
    that NO engine serves (here: deterministic, which this row declines and the
    backend has no plan for) used to reach plan build via
    `override_heuristics_query()`, which pins a backend engine id and bypasses
    the heuristics query entirely. That pinned config then failed to finalize
    with CUDNN_STATUS_NOT_SUPPORTED, which `_CUDNN_CHECK_CUDNN_ERROR` folded
    into CUDNN_BACKEND_API_FAILED -- reaching Python as a bare RuntimeError.

    That distinction is load-bearing: every SDPA harness in this repo skips on
    cudnnGraphNotSupportedError and FAILS on anything else, so the wrong type
    turned "nobody serves this" into 91 red cases in
    test_mhas_v2.py::test_sdpa_random_bwd_L0 as soon as its head-dim sweep was
    widened to 512.
    """
    b, hq, sq, skv = 2, 2, 256, 256
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        g = _build_graph_only(b, hq, hq, sq, skv, _D, 1.0 / math.sqrt(_D), use_deterministic_algorithm=True)
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
        g.build_plans()


def test_capabilities_match_what_is_implemented():
    """The row must not claim anything the adapter would refuse at build. Guards
    against a Capabilities field being flipped on without the lowering."""
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == _ENGINE)
    c = spec.capabilities
    assert c.causal and c.bottom_right and c.swa and c.right_band_widening
    assert c.gqa
    assert c.d_envelope and c.d_pad_multiple == 8 and max(c.d) == 512
    assert c.d_envelope_floor == 256, "an envelope with no floor silently claims every small head dim"
    assert c.thd, "the packed path is served (blocked S/dS workspace + per-sequence descriptors)"
    assert c.thd_declared_totals, "the blocked workspace is sized from the declared totals at BUILD time"
    for gone in ("thd_causal", "thd_gqa"):
        assert not hasattr(c, gone), (
            f"{gone} must stay DELETED, not re-added: every row that serves THD now serves that "
            "feature too, so the flag can never fire. THD conjunction flags are transitional by "
            "design -- they exist only while some row genuinely cannot serve the pair."
        )
    assert not c.cu_seq_len, "sdpa_backward has no cu_seq_len_* port; no row can claim or test it"
    assert not c.padded, "DENSE padding masks: the kernel compiles a scalar length; THD carries its own"
    assert not c.bias and not c.dbias and not c.sink and not c.dsink
    assert not c.deterministic
    assert c.sm_lo == 100 and c.sm_hi == 103


def test_engine_is_registered_and_opt_in():
    from cudnn.engines.manifest import MANIFEST

    fam = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    assert _ENGINE in fam.slots
    assert fam.slots[_ENGINE].opt_in, "new engines stay opt-in until they earn arch coverage + benchmarks"


def _prepared_case(*, dtype=torch.bfloat16, causal=True, hkv=2, chunks=False, wide=None, wide_product=False):
    from types import SimpleNamespace
    from unittest.mock import patch
    from contextlib import nullcontext
    from cudnn.sdpa.bwd import api_dsl

    torch.manual_seed(2907)
    b, h, sq, skv, d = (5 if wide_product else 2), 4, 128, 128, 512
    tensors = {name: _bshd(b, length, heads, d, dt=dtype) for name, length, heads in (("q", sq, h), ("k", skv, hkv), ("v", skv, hkv), ("do", sq, h))}
    keep = _causal_keep(sq, skv) if causal else None
    o, stats, _, dq, dk, dv = _reference(*(tensors[name] for name in ("q", "k", "v", "do")), keep, h // hkv)
    tensors["o"] = _bshd(b, sq, h, d, dt=dtype, fill=False).copy_(o)
    tensors["stats"] = stats.unsqueeze(-1).contiguous()
    for dst, src in (("dq", "q"), ("dk", "k"), ("dv", "v")):
        tensors[dst] = torch.empty_like(tensors[src]).fill_(float("nan"))
    if wide is not None:
        import ctypes

        source = tensors[wide]
        strides = ((2**30 if wide_product else 2**32) + source.stride(0), *source.stride()[1:])
        elements = 1 + sum((n - 1) * st for n, st in zip(source.shape, strides))
        origin = 2**31 if wide_product else 0
        free, _ = torch.cuda.mem_get_info()
        if (elements + origin) * source.element_size() + 512 * 2**20 > free:
            pytest.skip("physical Int64 probe needs room for one wide buffer")
        try:
            backing = torch.empty(elements + origin, device="cuda", dtype=source.dtype)
        except torch.OutOfMemoryError:
            pytest.skip("physical Int64 probe could not allocate its guarded storage")
        if wide_product:
            for batch in range(b):
                decoy = origin + ctypes.c_int32(batch * strides[0]).value
                backing.as_strided((1, *source.shape[1:]), source.stride(), decoy).fill_(float("nan"))
        else:
            backing.as_strided(source.shape, source.stride()).fill_(float("nan"))
        tensors[wide] = backing.as_strided(source.shape, strides, origin).copy_(source)
    guard = patch.object(api_dsl, "_sm100_head_chunk", side_effect=lambda *a, group=1, **kw: group) if chunks else nullcontext()
    with guard:
        if wide is None:
            graph, refs, outputs = _build_graph(b, h, hkv, sq, skv, d, d**-0.5, dt=dtype, use_causal_mask=causal)
        else:
            graph = cudnn.pygraph(io_data_type=_io_dtype(dtype), intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
            refs = {
                name: graph.tensor(
                    name=name,
                    dim=list(tensors[name].shape),
                    stride=list(tensors[name].stride()),
                    data_type=cudnn.data_type.FLOAT if name == "stats" else _io_dtype(dtype),
                )
                for name in ("q", "k", "v", "o", "do", "stats")
            }
            outputs = graph.sdpa_backward(
                q=refs["q"], k=refs["k"], v=refs["v"], o=refs["o"], dO=refs["do"], stats=refs["stats"], attn_scale=d**-0.5, use_causal_mask=causal
            )
            for out, name in zip(outputs, ("dq", "dk", "dv")):
                out.set_output(True).set_data_type(_io_dtype(dtype)).set_stride(list(tensors[name].stride()))
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plans([cudnn.heur_mode.A])
        refs.update(zip(("dq", "dk", "dv"), outputs))
        index = _plan_index(graph)
        assert index is not None
        graph.select_plan(index)
        graph.check_support()
        graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8).fill_(0xBD)
    pack = {refs[name]: value for name, value in tensors.items()}
    graph.execute(pack, workspace)
    case = SimpleNamespace(graph=graph, refs=refs, tensors=tensors, pack=pack, workspace=workspace, keep=keep, group=h // hkv, expected=(dq, dk, dv))
    _check_prepared(case)
    return case


def _check_prepared(case, tensors=None, expected=None):
    tensors = case.tensors if tensors is None else tensors
    expected = case.expected if expected is None else expected
    for name, want in zip(("dq", "dk", "dv"), expected):
        actual = tensors[name].float()
        cos = torch.nn.functional.cosine_similarity(actual.flatten(), want.flatten(), dim=0).item()
        rel = ((actual - want).abs().max() / max(want.abs().max().item(), 1e-30)).item()
        assert cos > _TOL_COS and rel < _TOL_REL, f"{name}: cos={cos:.6f} max_rel_err={rel:.2e}"


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("hkv", [4, 2])
def test_prepared_sm100_rebind_stream_and_replay(dtype, causal, hkv):
    case = _prepared_case(dtype=dtype, causal=causal, hkv=hkv, chunks=True)
    tensors = {name: value.clone() for name, value in case.tensors.items()}
    pack = {case.refs[name]: value for name, value in tensors.items()}
    workspace = torch.empty_like(case.workspace).fill_(0xBD)
    stream, other = torch.cuda.Stream(), torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)
    capture = torch.cuda.CUDAGraph()

    def refresh():
        tensors["q"].mul_(0.75)
        tensors["do"].mul_(1.25)
        o, stats, _, dq, dk, dv = _reference(*(tensors[name] for name in ("q", "k", "v", "do")), case.keep, case.group)
        tensors["o"].copy_(o)
        tensors["stats"].copy_(stats.unsqueeze(-1))
        for name in ("dq", "dk", "dv"):
            tensors[name].fill_(float("nan"))
        workspace.fill_(0xBD)
        return dq, dk, dv

    try:
        expected = refresh()
        stream.wait_stream(torch.cuda.current_stream())
        other.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            case.graph.execute(pack, workspace, handle=handle)
        torch.cuda.current_stream().wait_stream(stream)
        _check_prepared(case, tensors, expected)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            with torch.cuda.stream(other):
                case.graph.execute(pack, workspace, handle=handle)
        expected = refresh()
        capture.replay()
        _check_prepared(case, tensors, expected)
    finally:
        capture.reset()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("hkv", [4, 2])
def test_prepared_sm100_execute_has_no_tensor_wrapping(causal, hkv, monkeypatch):
    import cutlass.cute as cute
    from cudnn.sdpa.bwd.api_dsl import WorkspaceCarver

    case = _prepared_case(causal=causal, hkv=hkv, chunks=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared backward rebuilt tensor operands, allocated, synchronized or compiled")

    with monkeypatch.context() as patcher:
        for name in ("view", "reshape", "as_strided", "permute", "transpose", "copy_", "zero_"):
            patcher.setattr(torch.Tensor, name, forbidden)
        for name in ("empty", "empty_like", "zeros", "zeros_like"):
            patcher.setattr(torch, name, forbidden)
        patcher.setattr(WorkspaceCarver, "__init__", forbidden)
        patcher.setattr(cute, "compile", forbidden)
        torch.cuda.set_sync_debug_mode("error")
        try:
            case.graph.execute(case.pack, case.workspace)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    _check_prepared(case)


@pytest.mark.parametrize("stride", [2**32 + 512, 2**30 + 512])
@pytest.mark.parametrize("hkv", [4, 2])
def test_prepared_sm100_retains_strided_stats_decline(stride, hkv):
    # This family only advertises contiguous dense Stats. Large-stride Stats
    # are declined before compilation; the eight native I/O ports run the
    # physical addressing probes in the separate L1 module.
    reason = _decline_reason(b=5, hq=4, hkv=hkv, sq=128, skv=128, stats_batch_stride=stride)
    assert reason is not None and "stats must be contiguous" in reason


@pytest.mark.parametrize("role", ["q", "k", "v", "o", "do", "dq", "dk", "dv"])
def test_prepared_sm100_standalone_rejects_changed_layout(role):
    from dataclasses import replace
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100

    case = _prepared_case()
    api = SdpaBwdDslSm100(**{"sample_" + name: value for name, value in case.tensors.items()}, is_causal=True, scale_softmax=512**-0.5)
    api.check_support()
    api.compile()
    launches = []
    api._prepared = replace(api._prepared, fn=lambda *args: launches.append(args))
    args = {name + "_tensor": value for name, value in case.tensors.items()}
    args[role + "_tensor"] = case.tensors[role].contiguous()
    assert args[role + "_tensor"].stride() != case.tensors[role].stride()
    with pytest.raises(ValueError, match="runtime geometry"):
        api.execute(**args, workspace=case.workspace)
    assert not launches


@pytest.mark.parametrize("role", ["q", "stats", "dq"])
@pytest.mark.parametrize("ordered", [False, True])
def test_prepared_sm100_raw_storage_and_explicit_overrides(role, ordered, monkeypatch):
    from dataclasses import replace

    case = _prepared_case()
    tensor = case.tensors[role]
    backing = torch.empty(tensor.numel() + 2, dtype=tensor.dtype, device="cuda")
    declared = backing.as_strided(tensor.shape, tensor.stride()).copy_(tensor)
    case.pack[case.refs[role]] = backing[::2]
    case.tensors[role] = declared
    ref = case.refs[role]
    kwargs = {}
    pack = case.pack
    if ordered:
        items = list(reversed(list(pack.items())))
        kwargs["tensor_uids"] = [t.get_uid() for t, _ in items]
        pack = [buffer for _, buffer in items]
    for name in ("dq", "dk", "dv"):
        case.tensors[name].fill_(float("nan"))
    case.graph.execute(pack, case.workspace, **kwargs)
    _check_prepared(case)
    kwargs.update(override_uids=[ref.get_uid()], override_shapes=[list(ref.get_dim())], override_strides=[list(ref.get_stride())])
    case.graph.execute(pack, case.workspace, **kwargs)
    _check_prepared(case)
    plan = case.graph._compiled_plans[case.graph._plan_index]
    launches = []
    monkeypatch.setattr(plan._prepared, "spec", replace(plan._prepared.spec, fn=lambda *args: launches.append(args)))
    kwargs["override_shapes"][0][2] //= 2
    with pytest.raises(ValueError, match="runtime geometry"):
        case.graph.execute(pack, case.workspace, **kwargs)
    assert not launches


@pytest.mark.parametrize("route", ["dense", "thd"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_prepared_backward_artifact_reloads_in_fresh_process(route, dtype, tmp_path):
    from prepared_bwd_cache_utils import check_backward_artifact_reload

    check_backward_artifact_reload("sm100", route, dtype, tmp_path)
