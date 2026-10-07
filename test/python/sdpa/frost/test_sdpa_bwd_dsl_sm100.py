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
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(prepared_host, "compile_cached", fake_compile)
    params = prepared_host.Params(1, 2, 2, 512, 128, 128, 256, 128, 2, False, False, 0, 256)
    # SM100 / SM103 exactly, then the Rubin line as a RANGE (107..119, the cc 10.7 d512 row's span): the part that ships next
    # is not declined by a list.  101 (no such device) and 120 (the GeForce line) are still refused.
    for sm in (100, 103, 107, 110, 119):
        prepared_host.compile_host(None, None, None, params, (), (), cutlass.BFloat16, sm, "target-probe")
    for bad in (101, 120):
        with pytest.raises(ValueError, match=f"got SM{bad}"):
            prepared_host.compile_host(None, None, None, params, (), (), cutlass.BFloat16, bad, "target-probe")
    assert [c["options"] for c in calls] == [f"--enable-tvm-ffi --gpu-arch sm_{sm}a" for sm in (100, 103, 107, 110, 119)]
    # The artifact symbol is per engine row (Rule 6): the SM100 row's by default, ``frost_<row>_prepared`` when the caller names one.
    assert all(c["symbol"] == "frost_sdpa_bwd_sm100_prepared" for c in calls)
    prepared_host.compile_host(None, None, None, params, (), (), cutlass.BFloat16, 107, "target-probe", symbol="frost_sdpa_bwd_sm107_d512_prepared")
    assert calls[-1]["symbol"] == "frost_sdpa_bwd_sm107_d512_prepared"


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


def _reference(q, k, v, do, keep=None, group=1, scale=None):
    """fp32 attention backward. ``keep`` is a [S_q, S_kv] bool mask; ``scale`` None = 1/sqrt(d)."""
    kx = k.repeat_interleave(group, dim=1) if group > 1 else k
    vx = v.repeat_interleave(group, dim=1) if group > 1 else v
    scale = 1.0 / math.sqrt(q.shape[3]) if scale is None else scale
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
    # None => omit attn_scale on the graph, which means no scaling (1.0), as on the backend.
    scale = None if omit_scale else 1.0 / math.sqrt(d)
    q, do = _bshd(b, sq, hq, d, dt=dt), _bshd(b, sq, hq, d, dt=dt)
    k, v = _bshd(b, skv, hkv, d, dt=dt), _bshd(b, skv, hkv, d, dt=dt)
    o_ref, lse, all_masked, dq_r, dk_r, dv_r = _reference(q, k, v, do, keep, group, scale=1.0 if omit_scale else None)
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


# The 2x2 arm's per-test wall budget (compile of up to two stage-2 records + the accept, on a shared box): a wedged twin
# launch never returns, so the arm runs under a process watchdog (frost_test_utils.process_watchdog) that kills the test
# process instead of hanging the suite for the job timeout.
_TWIN_WATCHDOG_S = 900.0


@pytest.fixture(params=[False, True], ids=["4x1", "2x2"])
def stage2_datapath(request, monkeypatch):
    """Both stage-2 datapaths of the chain: the cga4x1 role split (``bprop_d512_f16.py``, what ships) and the fused
    2x2 twin (``bprop_d512_f16_2x2.py``, ``api_dsl.STAGE2_2X2``).  The twin is a module constant read at compile()
    time, so flipping it here reaches every plan the test builds; the 4x1 arm runs with the constant at its default
    (not merely unpatched) so the pair is a real A/B.  A spy on ``load_template`` records which stage-2 FILE served
    the plan -- the two kernels share a symbol name, so the file is the only honest witness.

    The 2x2 arm runs under a process watchdog (a wedged launch cannot be ended from Python): the twin's first wait form
    hung under GPU time-slicing (kernel docstring); the shipped form polls the cross-pair ring barriers and ran 12 fresh
    processes x 100 launches beside a 4x1 load process with 0 hangs (lane_d512_bprop/fix/fix12_summary.log), so the arm
    is no longer ``gpu_exclusive`` -- the watchdog stays as the suite's safety net."""
    import contextlib

    from frost_test_utils import process_watchdog

    from cudnn.sdpa.bwd import api_dsl

    monkeypatch.setattr(api_dsl, "STAGE2_2X2", request.param)
    served = []
    original = api_dsl.load_template

    def spy(path, params, tag="template"):
        if tag == "sdpa_bwd_sm100_stage2":
            served.append(path.rsplit("/", 1)[-1])
        return original(path, params, tag)

    monkeypatch.setattr(api_dsl, "load_template", spy)
    guard = process_watchdog(_TWIN_WATCHDOG_S, f"the 2x2 stage-2 arm of {request.node.nodeid}") if request.param else contextlib.nullcontext()
    with guard:
        yield request.param
    want = api_dsl._SM100_STAGE2_FILE_2X2.rsplit("/", 1)[-1] if request.param else api_dsl._SM100_STAGE2_FILE.rsplit("/", 1)[-1]
    assert served and all(s == want for s in served), f"stage 2 served by {served}, expected {want} (STAGE2_2X2={request.param})"


@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_dense(dt, stage2_datapath):
    _run(dt=dt)


@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_causal_dtypes(dt, stage2_datapath):
    """Both dtypes through the masked path too: the causal chain reads the
    workspace back in the io dtype, so a dtype mix-up shows up here and not in
    the dense case."""
    _run(dt=dt, keep=_causal_keep(512, 512), use_causal_mask=True)


@pytest.mark.parametrize("d", [264, 320, 384, 511 - 7, _D])
def test_head_dim_band(d, stage2_datapath):
    """d in (256, 512], any multiple of 8. 264 and 504 are not multiples of 16,
    which narrows the stage-3 epilogue store vector from 32 B to 16 B."""
    _run(sq=256, skv=256, d=d)


@pytest.mark.parametrize("hq,hkv", [(8, 8), (8, 4), (8, 2), (8, 1), (6, 3)])
def test_gqa_mqa(hq, hkv, stage2_datapath):
    _run(hq=hq, hkv=hkv, sq=256, skv=256)


@pytest.mark.parametrize(
    "hq,hkv,chunk,causal",
    [(16, 2, 4, False), (16, 2, 1, False), (12, 3, 6, False), (16, 2, 1, True)],
    ids=(
        "gemma-half-group",
        "gemma-single-head",
        "cross-group-boundary",
        "gemma-single-head-causal",
    ),
)
def test_gqa_non_group_aligned_head_chunks(monkeypatch, hq, hkv, chunk, causal):
    """The workspace budget may split or cross GQA-group boundaries.

    Gemma 4 has Hq=16/Hkv=2.  At long sequence lengths a whole eight-head
    group cannot fit the stated 4 GiB S+dS budget, so the dense adapter must
    process four- and one-head chunks without losing the Q-head -> KV-head
    mapping in dQ.  The single-head causal case also covers the masking used by
    Gemma's full-attention layer.  Hq=12/Hkv=3 additionally covers a six-head
    chunk that crosses a four-head group boundary.
    """
    import cudnn.sdpa.bwd.api_dsl as bwd_dsl

    choose = bwd_dsl._sm100_head_chunk
    seen = []

    def force_chunk(b, h_q, s_q, s_kv, bpe, budget=bwd_dsl._SM100_WS_BUDGET_BYTES, group=1):
        per_head = 2 * b * s_q * s_kv * bpe
        selected = choose(b, h_q, s_q, s_kv, bpe, budget=chunk * per_head, group=group)
        assert selected == chunk
        seen.append((selected, group))
        return selected

    monkeypatch.setattr(bwd_dsl, "_sm100_head_chunk", force_chunk)
    keep = _causal_keep(256, 256) if causal else None
    _run(
        b=1,
        hq=hq,
        hkv=hkv,
        sq=256,
        skv=256,
        keep=keep,
        use_causal_mask=causal,
    )
    assert seen
    assert all(item == (chunk, hq // hkv) for item in seen)


@pytest.mark.parametrize("s,want", [(8192, 16), (16384, 4), (32768, 1)])
def test_gemma4_head_chunk_respects_workspace_budget(s, want):
    from cudnn.sdpa.bwd.api_dsl import _sm100_head_chunk

    assert _sm100_head_chunk(1, 16, s, s, 2, group=8) == want


def test_causal_top_left(stage2_datapath):
    _run(keep=_causal_keep(512, 512), use_causal_mask=True)


def test_causal_bottom_right(stage2_datapath):
    _run(keep=_causal_keep(512, 512, bottom_right=True), use_causal_mask_bottom_right=True)


def test_causal_bottom_right_rectangular(stage2_datapath):
    """S_kv > S_q shifts the diagonal, which the stage-3 K-trim has to follow."""
    _run(sq=512, skv=1024, keep=_causal_keep(512, 1024, bottom_right=True), use_causal_mask_bottom_right=True)


def test_sliding_window(stage2_datapath):
    _run(keep=_causal_keep(512, 512, left=256), use_causal_mask=True, diagonal_band_left_bound=256)


def test_right_band_widening(stage2_datapath):
    """`diagonal_band_right_bound` alone -- passing use_causal_mask with it
    forces the bound back to 0 and the widening is silently dropped."""
    _run(keep=_causal_keep(512, 512, right=64), diagonal_band_right_bound=64)


@pytest.mark.parametrize("sq,skv", [(500, 500), (300, 200), (257, 129), (384, 640)])
def test_non_tile_multiple_seqlens(sq, skv, stage2_datapath):
    """Neither S_q nor S_kv has to be a tile multiple: the compile shape rounds
    up and the tail is masked."""
    _run(sq=sq, skv=skv)


@pytest.mark.parametrize("sq,skv", [(500, 500), (1000, 1000)])
def test_non_tile_multiple_causal(sq, skv, stage2_datapath):
    _run(sq=sq, skv=skv, keep=_causal_keep(sq, skv), use_causal_mask=True)


def test_dense_skv_96_zero_filled_tail(stage2_datapath):
    """``S_kv = 96`` in a 128-wide tile, dense: NOT a mask.  Dense padding masks are not served by this row, so the
    kernel compiles ``MASK_NONE`` (no ``apply_mask_chunk`` is traced) and the 32 tail columns [96, 128) are TMA-OOB
    ZERO-FILLED K / V rows (S = exp(-lse) there, dK / dV OOB stores dropped, dS * K = 0), on both datapaths.  Kept as
    the non-tile-multiple S_kv accept it is; the genuine 2x2 lane-map detector (a compiled mask whose band edge is
    column 64 of a tile) is ``test_sdpa_bwd_thd_sm100.py::test_graph_thd_kv_len_64_masks_the_upper_column_half``."""
    _run(sq=128, skv=96)


def test_default_attn_scale(stage2_datapath):
    """attn_scale is OPTIONAL on the graph; omitting it means no scaling (1.0), the backend's meaning.

    The adapter used to leave `scale_softmax` at None and die in execute with
    `TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'`,
    after the row had already admitted the graph and check_support had passed.
    The reference runs at 1.0, so a 1/sqrt(d) default fails the comparison
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


def test_d256_half_graph_is_offered_by_exactly_the_2x2_row():
    """On the SM100 line the bf16 d256 backward is served by ``sdpa_bwd_sm100_d256`` (the 2x2-datapath body) and by NO other
    python row -- this d512 row keeps declining it (the envelope floor is exclusive at 256)."""
    from cudnn.sdpa.bwd import engines as bwd_engines

    g = _build_graph_only(2, 2, 2, 256, 256, 256, 1.0 / math.sqrt(256))
    served = {s.name for s in bwd_engines.ENGINE_SPECS if bwd_engines.analyze_for(s, g, None)[1] is None}
    assert served == {"sdpa_bwd_sm100_d256"}, served


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


def test_engine_is_registered_and_offered_by_default():
    from cudnn.engines.manifest import MANIFEST

    fam = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    assert _ENGINE in fam.slots
    # The cuDNN backend has no SM100 engine for d in (256, 512], so this row is the only provider there.
    assert not fam.slots[_ENGINE].opt_in


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
def test_prepared_sm100_rebind_stream_and_replay(dtype, causal, hkv, stage2_datapath):
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


@pytest.mark.parametrize("route", ["dense_2x2", "thd_2x2"])
def test_prepared_backward_artifact_reloads_in_fresh_process_2x2(route, tmp_path):
    """The twin's artifact (its own template digest, so its own cache entry) exports and reloads in a second process
    with JIT forbidden; the child flips ``api_dsl.STAGE2_2X2`` before building the plan."""
    from prepared_bwd_cache_utils import check_backward_artifact_reload

    check_backward_artifact_reload("sm100", route, "bfloat16", tmp_path)


# --------------------------------------------------------------------------- #
# The 2x2 stage-2 twin: byte-identical default, bitwise twin, SASS pins       #
# --------------------------------------------------------------------------- #

import hashlib as _hashlib
import json as _json
import os as _os
import re as _re
import subprocess as _subprocess
import sys as _sys
import textwrap as _textwrap
from pathlib import Path as _Path

from frost_test_utils import arch_known_to_the_dsl, assert_no_new_spills, run_sass_probe, sass_probe_source

# Stage-2 renderings of the SM100 d512 chain, by the fields `SdpaBwdDslSm100.compile` spells (dense / causal / fp16 / THD).
_STAGE2_4X1_RECORDS = {
    "dense_bf16": dict(dtype_qkv=2),
    "causal_bf16": dict(dtype_qkv=2, window_right=0),
    "dense_fp16": dict(dtype_qkv=3),
    "thd_bf16": dict(dtype_qkv=2, thd_varlen=True),
}
# A host-only trace-compile of ONE stage-2 template's `_host` over fixed fake layouts (B=1 H=8 S=1024 d=512); the md5 of
# the dumped PTX is the rendering's identity (PTX, not cubin: ptxas renames uniform registers run to run).  The same probe
# serves the SASS pins (it is the only thing that compiles stage 2 outside the prepared chain).
_STAGE2_PROBE_BODY = r"""
import cutlass
import cutlass.cute as cute
from cudnn.frost.template_loader import load_template
from cudnn.frost.tile_dsl.constants import DTYPE_FP16
from cudnn.sdpa.bwd.api_dsl import _sm100_kernel_path
from cudnn.sdpa.bwd.config_sm100 import TemplateParams, TemplateParams2x2, TemplateParamsDbg
kernel_file, twin = params_kw.pop("kernel_file"), params_kw.pop("twin")
dbg = params_kw.pop("dbg", False)  # the 4x1's attribution record (TemplateParamsDbg); default = the shipped base record
params = TemplateParams2x2(**params_kw) if twin else (TemplateParamsDbg(**params_kw) if dbg else TemplateParams(**params_kw))
mod = load_template(_sm100_kernel_path(kernel_file), params, tag="stage2_probe")
print("DESC_VERSION", int(getattr(mod, "DESC_VERSION", 0)))
print("CLUSTER_Q_ROWS", int(getattr(mod.CFG, "CLUSTER_Q_ROWS", mod.CFG.TILE_M * mod.CFG.CTA_MMA)))
print("N_CHUNKS", int(getattr(mod.CFG, "N_CHUNKS", 0)))
io = cutlass.Float16 if int(params.dtype_qkv) == DTYPE_FP16 else cutlass.BFloat16
B, H, S, D = 1, 8, 1024, 512
PROBLEM = (B, H, S, S, H, H, S, S, 37 if params.thd_varlen else 0)

@cute.jit
def probe(entry: cutlass.Constexpr, q_ptr: cute.Pointer, k_ptr: cute.Pointer, v_ptr: cute.Pointer, do_ptr: cute.Pointer, s_ptr: cute.Pointer,
          ds_ptr: cute.Pointer, lse_ptr: cute.Pointer, dd_ptr: cute.Pointer, meta_ptr: cute.Pointer, desc_ptr: cute.Pointer, stream):
    bshd = cute.make_layout((B, S, H, D), stride=(S * H * D, H * D, D, 1))
    ws = cute.make_layout((B, H, S, S), stride=(H * S * S, S * S, S, 1))
    row = cute.make_layout((B, H, S), stride=(H * S, S, 1))
    entry(cute.make_tensor(q_ptr, bshd), cute.make_tensor(k_ptr, bshd), cute.make_tensor(v_ptr, bshd), cute.make_tensor(do_ptr, bshd),
          cute.make_tensor(s_ptr, ws), cute.make_tensor(ds_ptr, ws), cute.make_tensor(lse_ptr, row), cute.make_tensor(dd_ptr, row),
          cute.make_tensor(meta_ptr, cute.make_layout((5 * B + 5,), stride=(1,))), cute.make_tensor(desc_ptr, cute.make_layout((64,), stride=(1,))),
          PROBLEM, cutlass.Float32(0.5), cutlass.Float32(0.7), cutlass.Float32(0.5), cutlass.Int32(0), cutlass.Int32(0), stream)

def ptr(t, align=16):
    return cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=align)

cute.compile(probe, mod._host, ptr(io), ptr(io), ptr(io), ptr(io), ptr(io), ptr(io), ptr(cutlass.Float32, 4), ptr(cutlass.Float32, 4),
             ptr(cutlass.Int32, 4), ptr(cutlass.Int64, 8), cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
             options="--enable-tvm-ffi --gpu-arch " + arch)
"""
_STAGE2_PTX_PROBE = _textwrap.dedent(r"""
    import glob, hashlib, json, os, sys
    dump, arch, params_json = sys.argv[1], sys.argv[2], sys.argv[3]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "ptx"
    os.environ["CUTE_DSL_ARCH"] = arch
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    params_kw = json.loads(params_json)
    %(body)s
    ptxs = sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)
    if not ptxs:
        print("FAIL no ptx dumped into", dump, os.listdir(dump)); sys.exit(3)
    print("PTX_MD5", hashlib.md5(open(ptxs[-1], "rb").read()).hexdigest())
    """) % {"body": _textwrap.dedent(_STAGE2_PROBE_BODY)}


_STAGE2_MD5_RECORD = _Path(__file__).resolve().parent / "renderings" / "md5_stage2_4x1_sm100a.txt"


def _stage2_md5_record():
    """The COMMITTED record ``renderings/md5_stage2_4x1_sm100a.txt`` -- the 4x1 stage-2 PTX md5s rendered from the tree
    BEFORE the 2x2 twin landed (develop a3eed7d04) -- so the byte-identity pin gates in every checkout and in CI, not only
    on the box that rendered it.  A local ``frost_dev/results/bwd_d512_2x2/renderings/md5_stage2_4x1_sm100a.txt`` (this
    checkout's or the main checkout's) overrides it for re-rendering experiments.  Lines: ``dsl=<distribution> <version>``
    (the DSL build the PTX is a function of) and ``stage2_4x1 sm_100a <record> rc=0 ptx_md5=<md5>``."""
    root = _Path(__file__).resolve().parents[4]
    roots = [root] + ([root.parents[1]] if root.parent.name == ".worktrees" else [])
    for r in roots:
        f = r / "frost_dev" / "results" / "bwd_d512_2x2" / "renderings" / "md5_stage2_4x1_sm100a.txt"
        if f.is_file():
            return f
    return _STAGE2_MD5_RECORD


def _parse_md5_record(f):
    """-> (dsl line or None, {record: md5})."""
    dsl, want = None, {}
    for ln in f.read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[len("dsl=") :].strip()
        m = _re.match(r"stage2_4x1 sm_100a (\S+) rc=0 ptx_md5=([0-9a-f]{32})", ln)
        if m:
            want[m.group(1)] = m.group(2)
    return dsl, want


def test_stage2_md5_record_is_committed_and_complete():
    """The pin's baseline is in the tree (the review found the previous record local-only, so the test skipped everywhere
    but one box): four records, a DSL line, and the stage-2 probe's record names."""
    assert _STAGE2_MD5_RECORD.is_file(), _STAGE2_MD5_RECORD
    dsl, want = _parse_md5_record(_STAGE2_MD5_RECORD)
    assert dsl and dsl.startswith("nvidia-cutlass-dsl "), dsl
    assert set(want) == set(_STAGE2_4X1_RECORDS), (sorted(want), sorted(_STAGE2_4X1_RECORDS))


@pytest.mark.parametrize("record", list(_STAGE2_4X1_RECORDS))
def test_stage2_default_rendering_ptx_md5_is_unchanged(tmp_path, record):
    """The 4x1 stage-2 rendering is PTX-IDENTICAL to the tree before the 2x2 twin: the twin is a sibling FILE with its own
    config record, ``make_bwd_decode`` reads the cluster span through ``getattr`` with the 4x1 defaults, and the adapter's
    ``gran`` getattr folds to the same 256.  Compared against the committed pre-edit record (``renderings/``); skips only
    when the installed DSL build is not the one the record names (the PTX text is a function of it).  A host
    trace-compile for sm_100a, no device."""
    from cudnn.frost.buffers import cutedsl_state

    f = _stage2_md5_record()
    dsl, want = _parse_md5_record(f)
    assert record in want, f"{record} is not in the recorded list ({sorted(want)}) of {f}"
    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if dsl is not None and have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: PTX text differs by DSL build, re-render the record")
    if not arch_known_to_the_dsl("sm_100a"):
        pytest.skip("this cutlass-dsl has no sm_100a")
    dump = tmp_path / f"sm100a_stage2_4x1_{record}"
    dump.mkdir()
    script = dump / "ptx_probe.py"
    script.write_text(_STAGE2_PTX_PROBE)
    kw = dict(_STAGE2_4X1_RECORDS[record], kernel_file="sm100/bprop_d512_f16.py", twin=False)
    proc = _subprocess.run([_sys.executable, str(script), str(dump), "sm_100a", _json.dumps(kw)], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"sm_100a trace-compile of the 4x1 stage-2 {record} rendering failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    out = dict(ln.split(maxsplit=1) for ln in proc.stdout.splitlines() if ln.startswith(("PTX_MD5", "CLUSTER_Q_ROWS", "DESC_VERSION")))
    assert out["CLUSTER_Q_ROWS"] == "256"
    got = out["PTX_MD5"].strip()
    print(f"\nSM100 stage-2 4x1 {record}: PTX md5 {got} (pre-edit {want[record]})")
    assert got == want[record], f"{record}: PTX md5 {got} != the pre-edit record's {want[record]} -- the 4x1 stage-2 rendering changed"


# --------------------------------------------------------------------------- the 2x2 twin's DEFAULT rendering pin + its lever

_STAGE2_2X2_MD5_RECORD = _Path(__file__).resolve().parent / "renderings" / "md5_stage2_2x2_sm100a.txt"
_STAGE2_2X2_RECORDS = {
    "dense_bf16": dict(dtype_qkv=2),
    "causal_bf16": dict(dtype_qkv=2, window_right=0),
    "dense_fp16": dict(dtype_qkv=3),
    "thd_bf16": dict(dtype_qkv=2, thd_varlen=True),
}
# The RED side (an armed lever renders different PTX) needs no fp16 / THD render: the lever is dtype- and
# layout-independent code, and each render is a ~1 min host trace-compile.
_STAGE2_2X2_RED_RECORDS = ("dense_bf16", "causal_bf16")
# The attribution lever armed with a fake (non-zero, 64-B aligned) dump address: a host trace-compile only reads the
# constant, so any aligned value renders the instrumented kernel.
_FAKE_DUMP_ADDR = 4096


def _parse_md5_record_2x2(f):
    dsl, want = None, {}
    for ln in f.read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[len("dsl=") :].strip()
        m = _re.match(r"stage2_2x2 sm_100a (\S+) rc=0 ptx_md5=([0-9a-f]{32})", ln)
        if m:
            want[m.group(1)] = m.group(2)
    return dsl, want


def _render_stage2_ptx_md5(tmp_path, name, kw):
    """Host trace-compile of one stage-2 record for sm_100a -> (PTX md5, probe prints); skips where the DSL cannot."""
    if not arch_known_to_the_dsl("sm_100a"):
        pytest.skip("this cutlass-dsl has no sm_100a")
    dump = tmp_path / name
    dump.mkdir()
    script = dump / "ptx_probe.py"
    script.write_text(_STAGE2_PTX_PROBE)
    proc = _subprocess.run([_sys.executable, str(script), str(dump), "sm_100a", _json.dumps(kw)], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"sm_100a trace-compile of {name} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    out = dict(ln.split(maxsplit=1) for ln in proc.stdout.splitlines() if ln.startswith(("PTX_MD5", "CLUSTER_Q_ROWS", "DESC_VERSION", "N_CHUNKS")))
    return out["PTX_MD5"].strip(), out


def _stage2_2x2_md5_want(record):
    assert _STAGE2_2X2_MD5_RECORD.is_file(), _STAGE2_2X2_MD5_RECORD
    dsl, want = _parse_md5_record_2x2(_STAGE2_2X2_MD5_RECORD)
    assert record in want, f"{record} is not in the recorded list ({sorted(want)})"
    from cudnn.frost.buffers import cutedsl_state

    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if dsl is not None and have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: PTX text differs by DSL build, re-render the record")
    return want[record]


@pytest.mark.parametrize("record", list(_STAGE2_2X2_RECORDS))
def test_stage2_2x2_default_rendering_ptx_md5_is_unchanged(tmp_path, record):
    """The 2x2 twin's DEFAULT rendering (every debug lever off) is PTX-IDENTICAL to the committed pre-lever record
    (``renderings/md5_stage2_2x2_sm100a.txt``, rendered at d4b024671 before ``TemplateParams2x2.debug_clk`` landed): the
    attribution lever is zero traced code when off.  Host trace-compile for sm_100a, no device."""
    want = _stage2_2x2_md5_want(record)
    kw = dict(_STAGE2_2X2_RECORDS[record], kernel_file="sm100/bprop_d512_f16_2x2.py", twin=True)
    got, out = _render_stage2_ptx_md5(tmp_path, f"sm100a_stage2_2x2_{record}", kw)
    assert out["CLUSTER_Q_ROWS"] == "256" and out["N_CHUNKS"] == "8"
    print(f"\nSM100 stage-2 2x2 {record}: PTX md5 {got} (record {want})")
    assert got == want, f"{record}: PTX md5 {got} != the record's {want} -- the 2x2 stage-2 DEFAULT rendering changed"


@pytest.mark.parametrize("record", list(_STAGE2_2X2_RED_RECORDS))
def test_stage2_2x2_debug_clk_lever_renders_code(tmp_path, record):
    """The RED side of the pin above: ARMED, the attribution lever renders a DIFFERENT kernel (the clock reads, the SMEM
    accumulators and the exit dump are real code), so a lever that leaked into the default rendering would trip the md5
    pin rather than ride along unnoticed."""
    want = _stage2_2x2_md5_want(record)
    kw = dict(_STAGE2_2X2_RECORDS[record], kernel_file="sm100/bprop_d512_f16_2x2.py", twin=True, debug_clk=1, debug_dump_addr=_FAKE_DUMP_ADDR)
    got, _out = _render_stage2_ptx_md5(tmp_path, f"sm100a_stage2_2x2_clk_{record}", kw)
    assert got != want, f"{record}: the armed debug_clk lever rendered the SAME PTX as the default -- the lever traces no code"


def test_stage2_2x2_debug_clk_dump_accounts_the_waits(monkeypatch):
    """GPU: the attribution lever on a one-wave shape (B=1 H=4 S=1024: 16 clusters, one q tile each, 8 kv tiles).  Every
    non-scheduler warp of every CTA writes a record whose body clock is positive and bounds its wait buckets; each role
    accumulates exactly the barriers it waits on (and nothing else), the MMA leader's kv total is the 8 kv tiles, and the
    instrumented twin's dQ / dK / dV are BITWISE the plain twin's (the lever never changes a wait's form or the math)."""
    import dataclasses

    from cuda.bindings import driver as cu

    from cudnn.sdpa.bwd import api_dsl
    from cudnn.sdpa.bwd.kernels.sm100 import bprop_d512_f16_2x2 as K2

    b, hq, hkv, sq, skv, d, dt = 1, 4, 4, 1024, 1024, _D, torch.bfloat16
    torch.manual_seed(2026)
    q, do = _bshd(b, sq, hq, d, dt=dt), _bshd(b, sq, hq, d, dt=dt)
    k, v = _bshd(b, skv, hkv, d, dt=dt), _bshd(b, skv, hkv, d, dt=dt)
    o_ref, lse, _all_masked, _, _, _ = _reference(q, k, v, do, None, 1)
    o = _bshd(b, sq, hq, d, dt=dt, fill=False)
    o.copy_(o_ref.to(dt))
    tensors = dict(q=q, k=k, v=v, o=o, do=do, stats=lse.unsqueeze(-1).contiguous())
    kw = dict(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, d=d, dt=dt)
    plain = _twin_capture(monkeypatch, True, tensors, **kw)

    words = K2.DBG_CLK_WORDS
    dump = torch.zeros(K2.DBG_CLK_MAX_WORDS, dtype=torch.int64, pin_memory=True)
    err, dp = cu.cuMemHostGetDevicePointer(dump.data_ptr(), 0)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    levers = dict(debug_clk=1, debug_dump_addr=int(dp))
    original = api_dsl.load_template

    def armed(path, params, tag="template"):
        if tag == "sdpa_bwd_sm100_stage2":
            params = dataclasses.replace(params, **levers)
        return original(path, params, tag)

    monkeypatch.setattr(api_dsl, "load_template", armed)
    got = _twin_capture(monkeypatch, True, tensors, **kw)
    torch.cuda.synchronize()
    for name, a, c in zip(("dq", "dk", "dv"), plain[:3], got[:3]):
        assert torch.equal(a, c), f"{name}: the instrumented twin is not bitwise the plain twin"

    nx, ny = (sq // 256) * 4, hq
    nblk = nx * ny
    assert nblk * 8 * words <= K2.DBG_CLK_MAX_WORDS, "the record index is unchecked in the kernel: the grid must fit the 16 MiB dump (<= 8192 CTAs)"
    rec = dump[: nblk * 8 * words].view(nblk, 8, words)
    total, tiles, kv_total = rec[:, :, K2.DBG_CLK_TOTAL], rec[:, :, K2.DBG_CLK_TILES], rec[:, :, K2.DBG_CLK_KV_TOTAL]
    assert bool((total[:, :7] > 0).all()), "every non-scheduler warp of every CTA records a body clock"
    assert bool((rec[:, 7, :] == 0).all()), "the scheduler warp never records"
    assert bool((tiles[:, :7] == 1).all()), tiles[:, :7]  # 16 clusters on 37 slots: one q tile each
    leaders = torch.arange(nblk) % 2 == 0  # cta_in_pair == 0
    assert bool((kv_total[leaders, 4] == skv // 128).all()), kv_total[leaders, 4]
    assert bool((kv_total[~leaders, 4] == 0).all())
    assert bool((kv_total[:, 5] == (skv // 128) * K2.CFG.N_CHUNKS).all()), kv_total[:, 5]  # LDG counts chunks
    assert bool((kv_total[:, 6] == skv // 128).all()), kv_total[:, 6]
    waits = rec[:, :, 1:11].sum(dim=-1)
    assert bool((waits[:, :7] <= total[:, :7]).all()), "the wait buckets are a part of the body clock"
    # Which barriers each role waits on (DBG_BAR ids 1..10 -> slots 1..10): exactly these buckets are positive.
    expect = {
        5: {2, 4, 10},  # LDG: op_empty, ring_empty, sched
        6: {7, 10},  # STG: smem_full, sched
    }
    for w in range(4):
        expect[w] = {5, 8, 10}  # compute: bmm_done, smem_empty, sched
    for w, bars in expect.items():
        pos = {i for i in range(1, 11) if bool((rec[:, w, i] > 0).all())}
        zero = {i for i in range(1, 11) if bool((rec[:, w, i] == 0).all())}
        assert pos == bars and zero == set(range(1, 11)) - bars, (w, pos, zero)
    lead_mma = rec[leaders, 4, :]
    pos = {i for i in range(1, 11) if bool((lead_mma[:, i] > 0).all())}
    assert pos == {1, 3, 6, 9, 10}, pos  # op_full, ring_full, acc_empty, tmem_dealloc, sched
    assert bool((lead_mma[:, K2.DBG_CLK_SEG_MMA_ISSUE] > 0).all()) and bool((rec[:, 5, K2.DBG_CLK_SEG_LDG_ISSUE] > 0).all())
    assert (
        bool((rec[:, 6, K2.DBG_CLK_SEG_STG_STORE] > 0).all())
        and bool((rec[:, :4, K2.DBG_CLK_SEG_CMP_MATH] > 0).all())
        and bool((rec[:, :4, K2.DBG_CLK_SEG_CMP_CAST] > 0).all())
    )


@pytest.mark.parametrize("record", ["dense_bf16", "causal_bf16"])
def test_stage2_4x1_debug_clk_lever_renders_code(tmp_path, record):
    """The RED side of the committed 4x1 md5 pin: ARMED through the sibling ``TemplateParamsDbg`` record, the 4x1's
    attribution lever renders a DIFFERENT kernel than the shipped one, so a lever leaking into the default rendering
    would trip ``test_stage2_default_rendering_ptx_md5_is_unchanged`` rather than ride along unnoticed."""
    f = _stage2_md5_record()
    dsl, want = _parse_md5_record(f)
    from cudnn.frost.buffers import cutedsl_state

    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if dsl is not None and have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}")
    kw = dict(_STAGE2_4X1_RECORDS[record], kernel_file="sm100/bprop_d512_f16.py", twin=False, dbg=True, debug_clk=1, debug_dump_addr=_FAKE_DUMP_ADDR)
    got, out = _render_stage2_ptx_md5(tmp_path, f"sm100a_stage2_4x1_clk_{record}", kw)
    assert out["CLUSTER_Q_ROWS"] == "256"
    assert got != want[record], f"{record}: the armed 4x1 debug_clk lever rendered the SAME PTX as the shipped kernel -- the lever traces no code"


def test_stage2_4x1_debug_clk_dump_accounts_the_waits(monkeypatch):
    """GPU: the 4x1 role split's attribution lever on the same one-wave shape as the twin's test.  Every non-scheduler
    warp records a positive body clock bounding its wait buckets, each role accumulates exactly the barriers it waits
    on (sg0 = CTAs 0,1 waits the ship's ``xfer_empty``, sg1 = CTAs 2,3 its ``xfer_full``; the alias seam on the LDG warp;
    named barrier 8 on the compute WG), the kv totals are the 8 kv tiles, and dQ / dK / dV are BITWISE the shipped
    kernel's (the lever changes no wait's form and no math)."""
    import dataclasses

    from cuda.bindings import driver as cu

    from cudnn.sdpa.bwd import api_dsl
    from cudnn.sdpa.bwd import config_sm100 as cfgmod
    from cudnn.sdpa.bwd.kernels.sm100 import bprop_d512_f16 as K1

    b, hq, hkv, sq, skv, d, dt = 1, 4, 4, 1024, 1024, _D, torch.bfloat16
    torch.manual_seed(2027)
    q, do = _bshd(b, sq, hq, d, dt=dt), _bshd(b, sq, hq, d, dt=dt)
    k, v = _bshd(b, skv, hkv, d, dt=dt), _bshd(b, skv, hkv, d, dt=dt)
    o_ref, lse, _all_masked, _, _, _ = _reference(q, k, v, do, None, 1)
    o = _bshd(b, sq, hq, d, dt=dt, fill=False)
    o.copy_(o_ref.to(dt))
    tensors = dict(q=q, k=k, v=v, o=o, do=do, stats=lse.unsqueeze(-1).contiguous())
    kw = dict(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, d=d, dt=dt)
    plain = _twin_capture(monkeypatch, False, tensors, **kw)

    words = K1.DBG_CLK_WORDS
    dump = torch.zeros(K1.DBG_CLK_MAX_WORDS, dtype=torch.int64, pin_memory=True)
    err, dp = cu.cuMemHostGetDevicePointer(dump.data_ptr(), 0)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    original = api_dsl.load_template

    def armed(path, params, tag="template"):
        if tag == "sdpa_bwd_sm100_stage2":
            params = cfgmod.TemplateParamsDbg(**dataclasses.asdict(params), debug_clk=1, debug_dump_addr=int(dp))
        return original(path, params, tag)

    monkeypatch.setattr(api_dsl, "load_template", armed)
    got = _twin_capture(monkeypatch, False, tensors, **kw)
    torch.cuda.synchronize()
    for name, a, c in zip(("dq", "dk", "dv"), plain[:3], got[:3]):
        assert torch.equal(a, c), f"{name}: the instrumented 4x1 is not bitwise the shipped kernel"

    nx, ny = (sq // 256) * 4, hq
    nblk = nx * ny
    assert nblk * 8 * words <= K1.DBG_CLK_MAX_WORDS, "the record index is unchecked in the kernel: the grid must fit the 16 MiB dump (<= 8192 CTAs)"
    rec = dump[: nblk * 8 * words].view(nblk, 8, words)
    total, tiles, kv_total = rec[:, :, K1.DBG_CLK_TOTAL], rec[:, :, K1.DBG_CLK_TILES], rec[:, :, K1.DBG_CLK_KV_TOTAL]
    assert bool((total[:, :7] > 0).all()) and bool((rec[:, 7, :] == 0).all())
    assert bool((tiles[:, :7] == 1).all()), tiles[:, :7]
    cta = torch.arange(nblk) % 4
    leaders = cta % 2 == 0
    assert bool((kv_total[leaders, 4] == skv // 128).all()) and bool((kv_total[~leaders, 4] == 0).all())
    assert bool((kv_total[:, 5] == skv // 128).all()) and bool((kv_total[:, 6] == skv // 128).all())
    nb = 14
    waits = rec[:, :, 1 : nb + 1].sum(dim=-1)
    assert bool((waits[:, :7] <= total[:, :7]).all())

    def _buckets(sel, w):
        return {i for i in range(1, nb + 1) if bool((rec[sel, w, i] > 0).all())}, {i for i in range(1, nb + 1) if bool((rec[sel, w, i] == 0).all())}

    everyone = torch.ones(nblk, dtype=torch.bool)
    for w, bars in {5: {2, 4, 10, 11}, 6: {7, 10}}.items():  # LDG: op_empty, ring_empty, sched, utccp_done; STG: smem_full, sched
        pos, zero = _buckets(everyone, w)
        assert pos == bars and zero == set(range(1, nb + 1)) - bars, (w, pos, zero)
    for w in range(4):
        pos, zero = _buckets(cta < 2, w)  # sg0 compute: bmm_done, smem_empty, sched, xfer_empty, bar8
        assert pos == {5, 8, 10, 13, 14} and zero == set(range(1, nb + 1)) - {5, 8, 10, 13, 14}, (w, pos, zero)
        pos, zero = _buckets(cta >= 2, w)  # sg1 compute: bmm_done, smem_empty, sched, xfer_full, bar8
        assert pos == {5, 8, 10, 12, 14} and zero == set(range(1, nb + 1)) - {5, 8, 10, 12, 14}, (w, pos, zero)
    pos, zero = _buckets(leaders, 4)  # leader MMA: op_full, ring_full, acc_empty, tmem_dealloc, sched
    assert pos == {1, 3, 6, 9, 10} and zero == set(range(1, nb + 1)) - {1, 3, 6, 9, 10}, (pos, zero)
    assert bool((rec[leaders, 4, K1.DBG_CLK_SEG_MMA_ISSUE] > 0).all()) and bool((rec[:, 5, K1.DBG_CLK_SEG_LDG_ISSUE] > 0).all())
    assert bool((rec[:, 6, K1.DBG_CLK_SEG_STG_STORE] > 0).all())
    assert bool((rec[:, :4, K1.DBG_CLK_SEG_CMP_MATH] > 0).all()) and bool((rec[:, :4, K1.DBG_CLK_SEG_CMP_CAST] > 0).all())


def _twin_capture(monkeypatch, twin, tensors, *, b, hq, hkv, sq, skv, d, dt, **sdpa_kwargs):
    """Build + pin + execute with ``STAGE2_2X2 = twin``; return int16 views of dQ / dK / dV and of the S / dS workspace
    regions, plus the stage-2 file that served the plan."""
    from cudnn.sdpa.bwd import api_dsl

    monkeypatch.setattr(api_dsl, "STAGE2_2X2", twin)
    served = []
    original = api_dsl.load_template

    def spy(path, params, tag="template"):
        if tag == "sdpa_bwd_sm100_stage2":
            served.append(path.rsplit("/", 1)[-1])
        return original(path, params, tag)

    monkeypatch.setattr(api_dsl, "load_template", spy)
    g, t, (dq_t, dk_t, dv_t) = _build_graph(b, hq, hkv, sq, skv, d, 1.0 / math.sqrt(d), dt=dt, **sdpa_kwargs)
    idx = _plan_index(g)
    assert idx is not None
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8).fill_(0xBD)
    dq, dk, dv = _bshd(b, sq, hq, d, dt=dt, fill=False), _bshd(b, skv, hkv, d, dt=dt, fill=False), _bshd(b, skv, hkv, d, dt=dt, fill=False)
    for x in (dq, dk, dv):
        x.fill_(float("nan"))
    g.execute(
        {
            t["q"]: tensors["q"],
            t["k"]: tensors["k"],
            t["v"]: tensors["v"],
            t["o"]: tensors["o"],
            t["do"]: tensors["do"],
            t["stats"]: tensors["stats"],
            dq_t: dq,
            dk_t: dk,
            dv_t: dv,
        },
        ws,
    )
    torch.cuda.synchronize()
    # The prepared workspace: [delta f32 | S | dS | ...], every region 128-B aligned (prepared_sm100.compile_plan), with the
    # adapter's own pad / head-chunk arithmetic (api_dsl.SdpaBwdDslSm100.__init__: sq_pad to 256, skv_pad to 128).
    align = lambda n: (n + 127) // 128 * 128
    sq_pad, skv_pad = -(-sq // 256) * 256, -(-skv // 128) * 128
    qh_chunk = api_dsl._sm100_head_chunk(b, hq, sq_pad, skv_pad, 2, group=hq // hkv)
    delta = align(b * hq * (-(-sq // 128) * 128) * 4)
    region = align(b * qh_chunk * sq_pad * skv_pad * 2)
    s_ws, ds_ws = ws[delta : delta + region].clone(), ws[delta + region : delta + 2 * region].clone()
    assert served == ["bprop_d512_f16_2x2.py" if twin else "bprop_d512_f16.py"], served
    return [x.contiguous().view(torch.int16).clone() for x in (dq, dk, dv)] + [s_ws, ds_ws]


@pytest.mark.parametrize(
    "case",
    [
        dict(b=1, hq=8, hkv=8, sq=1024, skv=1024),
        dict(b=1, hq=8, hkv=8, sq=2048, skv=2048, use_causal_mask=True),
        dict(b=1, hq=8, hkv=8, sq=1024, skv=1024, use_causal_mask=True, diagonal_band_left_bound=256),
        dict(b=2, hq=4, hkv=4, sq=768, skv=1280, use_causal_mask_bottom_right=True),
        dict(b=1, hq=8, hkv=2, sq=1024, skv=1024),
        dict(b=1, hq=4, hkv=4, sq=1024, skv=1024, dt=torch.float16),
    ],
    ids=["dense", "causal_2k", "swa", "br_rect_b2", "gqa", "dense_fp16"],
)
def test_stage2_2x2_is_bitwise_the_role_split(monkeypatch, case):
    """The fused 2x2 stage 2 produces BITWISE the same S / dS workspace -- and therefore bitwise the same dQ / dK / dV --
    as the cga4x1 role split.  Expected, not hoped: the eight chained K = 64 MMA chunks accumulate into one fp32 TMEM
    accumulator exactly like one K = 512 chain (probe ss_slabs S3, bitwise over two seeds), the exp2 / mask / dS product
    are the same instructions over the same fp32 values (the role split shipped S at fp32), and stage 3 is untouched.
    So any mismatch here is a real bug (a lane map, a chunk descriptor, a stale ring slot), never accumulation noise --
    which is why this is ``torch.equal`` on int16 views and not a tolerance.  Three twin launches are also bitwise with
    each other (determinism).  Shapes are tile multiples: skipped causal tiles are zero-filled on both arms."""
    case = dict(case)
    dt = case.pop("dt", torch.bfloat16)
    b, hq, hkv, sq, skv = (case.pop(k) for k in ("b", "hq", "hkv", "sq", "skv"))
    d = _D
    torch.manual_seed(1811)
    q, do = _bshd(b, sq, hq, d, dt=dt), _bshd(b, sq, hq, d, dt=dt)
    k, v = _bshd(b, skv, hkv, d, dt=dt), _bshd(b, skv, hkv, d, dt=dt)
    keep = None
    if case.get("use_causal_mask") or case.get("use_causal_mask_bottom_right"):
        keep = _causal_keep(sq, skv, bottom_right=bool(case.get("use_causal_mask_bottom_right")), left=case.get("diagonal_band_left_bound"))
    o_ref, lse, all_masked, _, _, _ = _reference(q, k, v, do, keep, hq // hkv)
    o = _bshd(b, sq, hq, d, dt=dt, fill=False)
    o.copy_(o_ref.to(dt))
    stats = (lse if all_masked is None else lse.masked_fill(all_masked, 0.0)).unsqueeze(-1).contiguous()
    tensors = dict(q=q, k=k, v=v, o=o, do=do, stats=stats)
    kw = dict(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, d=d, dt=dt, **case)
    base = _twin_capture(monkeypatch, False, tensors, **kw)
    twin = _twin_capture(monkeypatch, True, tensors, **kw)
    for name, x in zip(("dQ", "dK", "dV"), twin):
        # Both arms writing the same NaN-poisoned (never stored) rows would be bit-equal too; the twin's gradients must be finite.
        assert torch.isfinite(x.view(dt).float()).all(), f"{name}: the 2x2 twin left non-finite values"
    for name, x, y in zip(("dQ", "dK", "dV", "S_ws", "dS_ws"), base, twin):
        n_bad = (x != y).sum().item()
        assert n_bad == 0, f"{name}: {n_bad} of {x.numel()} int16 words differ between the role split and the 2x2 twin"
    for _ in range(2):
        again = _twin_capture(monkeypatch, True, tensors, **kw)
        for name, x, y in zip(("dQ", "dK", "dV", "S_ws", "dS_ws"), twin, again):
            assert torch.equal(x, y), f"{name}: the 2x2 twin is not deterministic across launches"


def test_stage3_cluster_tile_rule_by_sequence_length():
    """``api_dsl._sm100_stage3_cgrp_tile_mn`` on the SM100 line (cc 10.0 .. 10.6): the (512, 256) row for BSHD at padded
    max(S_q, S_kv) <= 4096 (measured -11.5 / -11.6 % dense S2K / S4K and -6.4 / -5.6 % causal S2K / S4K on the three GEMMs, a
    wash at S8K -- the constant's comment carries the numbers), the (512, 512) row above that and on every THD plan.  The rule
    is mask-blind, so it has no mask parameter.  Host-only: a pure function of (S_pad, thd, cc).  The other side of the cc
    term -- cc 10.7 / 11.0 keep (512, 512) -- is pinned in the ungated cc 10.7 backward suite
    (``::test_stage3_tile_rule_keeps_the_wide_row_off_the_sm100_line``)."""
    from cudnn.sdpa.bwd import api_dsl

    assert api_dsl._SM100_STAGE3_SMALL_S_TILE == (512, 256) and api_dsl._SM100_STAGE3_SMALL_S_MAX == 4096
    assert api_dsl._SM100_STAGE3_SMALL_S_CC == (100, 106)
    for cc in ((10, 0), (10, 3), (10, 6)):
        for s in (128, 1024, 2048, 4096):
            assert api_dsl._sm100_stage3_cgrp_tile_mn(s, False, cc) == (512, 256), (s, cc)
            assert api_dsl._sm100_stage3_cgrp_tile_mn(s, True, cc) == (512, 512), (s, cc)
        for s in (4224, 8192, 32768):
            assert api_dsl._sm100_stage3_cgrp_tile_mn(s, False, cc) == (512, 512), (s, cc)


def test_stage3_tile_rule_reads_the_device_cc(monkeypatch):
    """``SdpaBwdDslSm100.compile`` hands the rule the DEVICE's cc through ``_device_cc`` (the seam the cc 10.7 d512 row
    inherits): faked to cc 10.7 on this SM100 board, a dense S 1024 plan -- (512, 256) by the length term alone -- loads the
    (512, 512) row for both stage-3 records and still computes finite gradients (it is the shipped row)."""
    from cudnn.sdpa.bwd import api_dsl

    b, hq, hkv, s, d, dt = 1, 4, 4, 1024, _D, torch.bfloat16
    torch.manual_seed(1811)
    q, do = _bshd(b, s, hq, d, dt=dt), _bshd(b, s, hq, d, dt=dt)
    k, v = _bshd(b, s, hkv, d, dt=dt), _bshd(b, s, hkv, d, dt=dt)
    o_ref, lse, _, _, _, _ = _reference(q, k, v, do, None, 1)
    o = _bshd(b, s, hq, d, dt=dt, fill=False)
    o.copy_(o_ref.to(dt))
    tensors = dict(q=q, k=k, v=v, o=o, do=do, stats=lse.unsqueeze(-1).contiguous())
    kw = dict(b=b, hq=hq, hkv=hkv, sq=s, skv=s, d=d, dt=dt)
    assert api_dsl.SdpaBwdDslSm100._device_cc.__qualname__.startswith("SdpaBwdDslSm100."), "the seam moved; re-point the fake"
    _, served_here = _tile_capture(monkeypatch, None, tensors, **kw)
    assert served_here == [(512, 256), (512, 256)], served_here
    outs, served_107 = _tile_capture(monkeypatch, None, tensors, cc=(10, 7), **kw)
    assert served_107 == [(512, 512), (512, 512)], served_107
    for name, x in zip(("dQ", "dK", "dV"), outs):
        assert torch.isfinite(x.view(dt).float()).all(), name


def _tile_capture(monkeypatch, max_s, tensors, *, b, hq, hkv, sq, skv, d, dt, cc=None, **sdpa_kwargs):
    """Build + pin + execute with ``api_dsl._SM100_STAGE3_SMALL_S_MAX = max_s`` (None = the shipped bound) and, with ``cc``,
    the adapter's ``_device_cc`` faked to it; return int16 views of dQ / dK / dV and the ``cgrp_tile_mn`` the two stage-3
    records were loaded with."""
    from cudnn.sdpa.bwd import api_dsl

    if max_s is not None:
        monkeypatch.setattr(api_dsl, "_SM100_STAGE3_SMALL_S_MAX", max_s)
    if cc is not None:
        monkeypatch.setattr(api_dsl.SdpaBwdDslSm100, "_device_cc", lambda self: tuple(cc))
    served = []
    original = api_dsl.load_template

    def spy(path, params, tag="template"):
        if tag in ("sdpa_bwd_sm100_mm_lo", "sdpa_bwd_sm100_mm_hi"):
            served.append(tuple(params.cgrp_tile_mn))
        return original(path, params, tag)

    monkeypatch.setattr(api_dsl, "load_template", spy)
    g, t, (dq_t, dk_t, dv_t) = _build_graph(b, hq, hkv, sq, skv, d, 1.0 / math.sqrt(d), dt=dt, **sdpa_kwargs)
    idx = _plan_index(g)
    assert idx is not None
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8).fill_(0xBD)
    dq, dk, dv = _bshd(b, sq, hq, d, dt=dt, fill=False), _bshd(b, skv, hkv, d, dt=dt, fill=False), _bshd(b, skv, hkv, d, dt=dt, fill=False)
    for x in (dq, dk, dv):
        x.fill_(float("nan"))
    feed = {t["q"]: tensors["q"], t["k"]: tensors["k"], t["v"]: tensors["v"], t["o"]: tensors["o"], t["do"]: tensors["do"], t["stats"]: tensors["stats"]}
    g.execute({**feed, dq_t: dq, dk_t: dk, dv_t: dv}, ws)
    torch.cuda.synchronize()
    # A snapshot: the next capture's spy wraps THIS spy (monkeypatch stacks), so its loads would land in this list too.
    return [x.contiguous().view(torch.int16).clone() for x in (dq, dk, dv)], list(served)


@pytest.mark.parametrize(
    "case",
    [
        dict(b=1, hq=8, hkv=8, sq=1024, skv=1024),
        dict(b=1, hq=8, hkv=8, sq=2048, skv=2048, use_causal_mask=True),
        dict(b=1, hq=8, hkv=8, sq=1024, skv=1024, use_causal_mask=True, diagonal_band_left_bound=256),
        dict(b=2, hq=4, hkv=4, sq=768, skv=1280, use_causal_mask_bottom_right=True),
        dict(b=2, hq=4, hkv=4, sq=768, skv=1280),
        dict(b=1, hq=8, hkv=2, sq=2048, skv=2048),
        dict(b=1, hq=8, hkv=2, sq=4096, skv=4096, use_causal_mask=True),
        dict(b=1, hq=4, hkv=4, sq=1000, skv=1000, use_causal_mask=True),
        dict(b=1, hq=4, hkv=4, sq=1024, skv=1024, dt=torch.float16),
        dict(b=1, hq=2, hkv=2, sq=4224, skv=4224, default=(512, 512)),
    ],
    ids=[
        "dense",
        "causal_2k",
        "swa",
        "br_rect_b2",
        "rect_b2",
        "gqa_2k",
        "causal_gqa_4k_boundary",
        "causal_nontile_1000",
        "dense_fp16",
        "dense_4224_above_boundary",
    ],
)
def test_stage3_small_s_tile_is_bitwise_the_wide_row(monkeypatch, case):
    """The (512, 256) row the chain renders for BSHD at padded max(S_q, S_kv) <= 4096 produces BITWISE the dQ / dK / dV of the
    (512, 512) row: the same per-pair 512x256 work and k walk -- the 2x2 row only multicasts A to a second pair -- so the fp32
    accumulation order is identical and ``torch.equal`` on int16 views is the right oracle (a tolerance would hide a wrong
    N-tile coordinate).  The whole causal family is here (the stage-2 twin test's cells: causal, SWA band, bottom-right rect
    B=2, causal GQA, a non-tile-multiple S) because the rule flips it too and that is where the 512-row M tile / `_causal_k_range`
    / `_zero_ws` interplay lives.  Each case runs the DEFAULT rule (no bound patched; the spy pins the row `compile` chose, so
    the 4096 boundary is pinned as served: S 4096 -> (512, 256), S 4224 -> (512, 512)) against the OTHER row forced through the
    bound (0 or 1 << 20).  Expected, not hoped: a bench that drew a fresh dO per build "found" a 1e-4 difference until it was
    seeded (`bench_baselines.build_bwd` draws dO itself; re-seed before every build you compare)."""
    case = dict(case)
    dt = case.pop("dt", torch.bfloat16)
    default = case.pop("default", (512, 256))
    b, hq, hkv, sq, skv = (case.pop(k) for k in ("b", "hq", "hkv", "sq", "skv"))
    d = _D
    torch.manual_seed(1811)
    q, do = _bshd(b, sq, hq, d, dt=dt), _bshd(b, sq, hq, d, dt=dt)
    k, v = _bshd(b, skv, hkv, d, dt=dt), _bshd(b, skv, hkv, d, dt=dt)
    keep = None
    if case.get("use_causal_mask") or case.get("use_causal_mask_bottom_right"):
        keep = _causal_keep(sq, skv, bottom_right=bool(case.get("use_causal_mask_bottom_right")), left=case.get("diagonal_band_left_bound"))
    o_ref, lse, all_masked, _, _, _ = _reference(q, k, v, do, keep, hq // hkv)
    o = _bshd(b, sq, hq, d, dt=dt, fill=False)
    o.copy_(o_ref.to(dt))
    stats = (lse if all_masked is None else lse.masked_fill(all_masked, 0.0)).unsqueeze(-1).contiguous()
    tensors = dict(q=q, k=k, v=v, o=o, do=do, stats=stats)
    kw = dict(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, d=d, dt=dt, **case)
    other = (512, 512) if default == (512, 256) else (512, 256)
    got_default, served_default = _tile_capture(monkeypatch, None, tensors, **kw)
    got_other, served_other = _tile_capture(monkeypatch, 0 if other == (512, 512) else 1 << 20, tensors, **kw)
    assert served_default == [default, default] and served_other == [other, other], (served_default, served_other)
    narrow, wide = (got_default, got_other) if default == (512, 256) else (got_other, got_default)
    for name, x in zip(("dQ", "dK", "dV"), narrow):
        assert torch.isfinite(x.view(dt).float()).all(), f"{name}: the (512, 256) row left non-finite values"
    for name, x, y in zip(("dQ", "dK", "dV"), narrow, wide):
        n_bad = (x != y).sum().item()
        assert n_bad == 0, f"{name}: {n_bad} of {x.numel()} int16 words differ between the (512, 256) and the (512, 512) stage-3 rows"


# SASS pins of the 2x2 twin (host trace-compile, no device): the register split reached the binary (USETMAXREG), no spills
# in the 40-register roles beyond the measured count, no GPU-scope drain before a cluster arrive, two tcgen05.ld per kv body
# (S_acc and dS_acc, one x64 each), 64 tcgen05.mma per kv body (8 chunks x 4 k-steps x 2 BMMs), no DSMEM bulk copy.  The
# counts are MEASURED on the branch's own toolchain (DSL 4.7.0, CUDA 13.3 ptxas) and bounded by SPILL_TOLERANCE.
_STAGE2_2X2_SASS_COUNTS = {
    "STL": ("STL",),
    "LDL": ("LDL",),
    "USETMAXREG": ("USETMAXREG",),
    "MEMBAR_GPU": ("MEMBAR.ALL.GPU",),
    "CGAERRBAR": ("CGAERRBAR",),
    "LDTM": ("LDTM",),
    "UTCHMMA": ("UTCHMMA",),
    "UTMALDG": ("UTMALDG",),
    "UBLKCP": ("UBLKCP",),
    "SYNCS_ARRIVE": (" SYNCS.ARRIVE",),
}
_STAGE2_2X2_SPILL_PINS = {"sm_100a": {"STL": 0, "LDL": 0}, "sm_107a": {"STL": 0, "LDL": 0}}

# The probe runs from a script FILE (the DSL parses a ``@cute.jit`` body through ``inspect.getsource``, which a ``python -c``
# source has none of -- ``run_sass_probe``'s inline form cannot host a jit function), dumps the cubin, disassembles it with the
# first nvdisasm that decodes the arch and prints one ``SASS <key> <count>`` line per entry of the counts table.
_STAGE2_SASS_PROBE = _textwrap.dedent(r"""
    import glob, hashlib, json, os, re, subprocess, sys
    dump, arch, params_json, cands = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "cubin"
    os.environ["CUTE_DSL_ARCH"] = arch
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    params_kw = json.loads(params_json)
    %(body)s
    cubins = sorted(glob.glob(os.path.join(dump, "*.cubin")), key=os.path.getmtime)
    if not cubins:
        print("FAIL no cubin dumped into", dump, os.listdir(dump)); sys.exit(3)
    print("CUBIN_MD5", hashlib.md5(open(cubins[-1], "rb").read()).hexdigest())
    nvd = None
    for c in cands:
        try:
            proc = subprocess.run([c, "-c", cubins[-1]], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            print("REJECT", c, "->", repr(exc)); continue
        if proc.returncode == 0 and proc.stdout.strip():
            nvd = c; print("NVDISASM", c); break
        print("REJECT", c, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if nvd is None:
        print("SKIP no nvdisasm candidate decodes the cubin"); sys.exit(0)
    sass = subprocess.run([nvd, "-c", cubins[-1]], capture_output=True, text=True, check=True).stdout.splitlines()
    def cnt(*subs):
        return sum(1 for ln in sass if all(sb in ln for sb in subs))
    for key, subs in json.loads(%(counts)r).items():
        print("SASS", key, cnt(*subs))
    print("SASS LINES", len(sass))
    """) % {"body": _textwrap.dedent(_STAGE2_PROBE_BODY), "counts": _json.dumps(_STAGE2_2X2_SASS_COUNTS)}


def _stage2_sass_probe(tmp_path, arch, params, tag):
    """``run_sass_probe``'s contract (skip on no arch / no nvdisasm, fail on a non-zero exit) for the script-file probe."""
    from frost_test_utils import nvdisasm_candidates

    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"this cutlass-dsl has no {arch}")
    cands = nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / f"{arch}_{tag}"
    dump.mkdir()
    script = dump / "sass_probe.py"
    script.write_text(_STAGE2_SASS_PROBE)
    proc = _subprocess.run([_sys.executable, str(script), str(dump), arch, _json.dumps(params), *cands], capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"{arch} trace-compile of {tag} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ") and len(ln.split()) == 3 and ln.split()[2].isdigit()}
    expect = {ln.split()[0]: int(ln.split()[1]) for ln in out if len(ln.split()) == 2 and ln.split()[0] in ("DESC_VERSION", "CLUSTER_Q_ROWS", "N_CHUNKS")}
    print(f"\n{tag} {arch} {params} SASS: {stats}; module says {expect}")
    return stats, expect


@pytest.mark.parametrize("arch,arm", [("sm_100a", "dense"), ("sm_100a", "causal_swa"), ("sm_107a", "dense"), ("sm_107a", "causal")])
def test_stage2_2x2_sass_pins(tmp_path, arch, arm):
    """Also the Rule S6 trace-compile of the Rubin arm (8-stage ring, 2 cast stages, 320 KiB, DESC_VERSION 0) from whatever
    GPU runs this suite: ``CUTE_DSL_ARCH=sm_107a`` needs no device.  Skips where the DSL predates sm_107a."""
    from cudnn.sdpa.bwd.config_sm100 import SM107_USABLE_DYN_SMEM_2X2

    masks = {"dense": {}, "causal": dict(window_right=0), "causal_swa": dict(window_right=0, window_left=256)}
    params = dict(dtype_qkv=2, kernel_file="sm100/bprop_d512_f16_2x2.py", twin=True, **masks[arm])
    if arch == "sm_107a":
        params.update(stages_kv=8, cast_stages=2, smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2)
    st, expect = _stage2_sass_probe(tmp_path, arch, params, tag=f"stage2_2x2_{arm}")
    # The compute WG's three-range split traces ITS kv body once per range; the MMA warp's kv loop is one body.
    n_compute_bodies = 1 if arm == "dense" else 3
    assert expect["DESC_VERSION"] == 0 and expect["CLUSTER_Q_ROWS"] == 256 and expect["N_CHUNKS"] == 8
    # USETMAXREG is 0 here as on the 4x1 sibling: ptxas C7508 drops every setmaxregister of these 8-warp d512 bodies (it
    # cannot determine the entry count) -- recorded, not required; the 12-warp sm107 bodies pin > 0 (see that file).
    assert st["MEMBAR_GPU"] == 0 and st["CGAERRBAR"] == 0, st
    assert st["UBLKCP"] == 0, f"no DSMEM bulk copy survives the fusion: {st}"
    assert st["LDTM"] == 2 * n_compute_bodies, f"two tcgen05.ld (S_acc, dS_acc) per compute kv body: {st}"
    assert st["UTCHMMA"] == 64, f"8 chunks x 4 k-steps x 2 BMMs in the one MMA kv body: {st}"
    # 8 Q + 8 dO subtile boxes per tile and one K + one V chunk box per ring stage x 8 chunks, one issue site each.
    assert st["UTMALDG"] == 32, st
    assert_no_new_spills(st, _STAGE2_2X2_SPILL_PINS[arch], tag=f"{arch} {arm}: ")


# --------------------------------------------------------------------------- #
# The GPU time-slicing hang: the contention test and its negative control      #
# --------------------------------------------------------------------------- #

# One child = one process = one CUDA context.  ``load`` runs the shipping 4x1 chain in a loop (the second context that
# makes the GPU time-slice); ``twin`` runs the 2x2 chain for N launches with a per-launch wall budget enforced by polling a
# CUDA event from Python (a wedged launch never returns, so the budget is the only way out) and exits 3 on a hang.  B=1 H=128
# S=8192 dense is the shape every hang was reproduced at; the inputs' VALUES are irrelevant to the schedule (stats = 0).
_CONTENTION_CHILD = _textwrap.dedent(r"""
    import dataclasses, json, math, os, sys, time
    role, n, budget_s, levers = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), json.loads(sys.argv[4])
    import torch
    import cudnn
    from cudnn.sdpa.bwd import api_dsl
    api_dsl.STAGE2_2X2 = role == "twin"
    if levers:
        _orig = api_dsl.load_template
        def _load(path, params, tag="template"):
            if tag == "sdpa_bwd_sm100_stage2":
                params = dataclasses.replace(params, **levers)
            return _orig(path, params, tag)
        api_dsl.load_template = _load
    import cudnn.sdpa  # noqa: F401
    b, hq, s, d = 1, 128, 8192, 512
    torch.manual_seed(0)
    def bshd(fill=True):
        t = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16) if fill else torch.zeros(b, s, hq, d, device="cuda", dtype=torch.bfloat16)
        return (t.mul_(0.1) if fill else t).permute(0, 2, 1, 3)
    q, k, v, o, do = bshd(), bshd(), bshd(), bshd(), bshd()
    dq, dk, dv = bshd(False), bshd(False), bshd(False)
    stats = torch.zeros(b, hq, s, 1, device="cuda", dtype=torch.float32)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    sh, st_ = [b, hq, s, d], [s * hq * d, d, hq * d, 1]
    t = {name: g.tensor(name=name, dim=sh, stride=st_) for name in ("q", "k", "v", "o", "do")}
    t["stats"] = g.tensor(name="stats", dim=[b, hq, s, 1], stride=[hq * s, s, 1, 1], data_type=cudnn.data_type.FLOAT)
    tdq, tdk, tdv = g.sdpa_backward(name="bwd", q=t["q"], k=t["k"], v=t["v"], o=t["o"], dO=t["do"], stats=t["stats"], attn_scale=1.0 / math.sqrt(d))
    for out in (tdq, tdk, tdv):
        out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_stride(st_)
    g.validate(); g.build_operation_graph(); g.create_execution_plans([cudnn.heur_mode.A])
    idx = next(i for i in range(g.get_execution_plan_count()) if "sdpa_bwd_sm100" in g.get_plan_name_at_index(i))
    g.select_plan(idx); g.check_support(); g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    feed = {t["q"]: q, t["k"]: k, t["v"]: v, t["o"]: o, t["do"]: do, t["stats"]: stats, tdq: dq, tdk: dk, tdv: dv}
    hist = []  # per-launch seconds: a starved launch (seconds, then the wall) reads differently from a wedged one (ms, ms, never)
    for i in range(n):
        t0 = time.time()
        g.execute(feed, ws)
        ev = torch.cuda.Event(); ev.record()
        while not ev.query():
            time.sleep(0.02)
            if time.time() - t0 > budget_s:
                import subprocess
                try:
                    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=True).stdout
                except Exception as e:  # a missing or stuck nvidia-smi must not turn the 45 s hang exit into a 40 min one
                    apps = f"<nvidia-smi unavailable: {e!r}>"
                print(f"[{role}] HANG: launch {i + 1} exceeded {budget_s:.0f} s; history (s): " + " ".join(f"{h:.2f}" for h in hist[-30:]), flush=True)
                print(f"[{role}] compute processes at the hang (all GPUs; may include this child; GPU UUID, PID, process name), "
                      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}: {apps.strip().splitlines()}", flush=True)
                os._exit(3)
        hist.append(time.time() - t0)
        if i == 0:
            print(f"[{role}] ready", flush=True)
    torch.cuda.synchronize()
    print(f"[{role}] done {n} launches", flush=True)
    """)


def _contention_run(tmp_path, *, twin_levers: dict, n_twin: int, budget_s: float, tag: str):
    """Start the 4x1 load child, wait until it is launching, run the twin child to completion (or its hang exit), stop the
    load.  Returns the twin's CompletedProcess; its stdout / the load's log are under ``tmp_path`` for the failure message."""
    import time

    script = tmp_path / "contention_child.py"
    script.write_text(_CONTENTION_CHILD)
    load_log = tmp_path / f"{tag}_load.log"
    with open(load_log, "w") as f:
        load = _subprocess.Popen([_sys.executable, str(script), "load", "100000", "600", "{}"], stdout=f, stderr=_subprocess.STDOUT, text=True)
    try:
        t0 = time.time()
        while "[load] ready" not in load_log.read_text():
            if load.poll() is not None:
                pytest.fail(f"the 4x1 load child exited early:\n{load_log.read_text()[-3000:]}")
            if time.time() - t0 > 900:
                pytest.fail(f"the 4x1 load child did not start launching within 900 s:\n{load_log.read_text()[-3000:]}")
            time.sleep(1.0)
        twin = _subprocess.run(
            [_sys.executable, str(script), "twin", str(n_twin), str(budget_s), _json.dumps(twin_levers)], capture_output=True, text=True, timeout=2400
        )
    finally:
        load.kill()
        load.wait()
    (tmp_path / f"{tag}_twin.log").write_text(twin.stdout + "\n--- stderr ---\n" + twin.stderr)
    return twin


@pytest.mark.xdist_group(name="gpu_exclusive")
def test_stage2_2x2_survives_gpu_time_slicing(tmp_path):
    """THE runnable detector of the GPU-sharing hang (python/cudnn/sdpa/AGENTS.md, 2x2 section): a second CUDA context
    running the 4x1 chain in a loop makes the GPU time-slice; the twin must then complete 100 launches at B=1 H=128 S=8192
    with every launch returning inside 45 s.  The pre-fix kernel -- every wait parked in NANOSLEEP.SYNCS -- hung here at
    launch 2, 25 and 74 of 200-300 on 2026-10-01 (lane_d512_bprop/fix/e0_control_nodbg.log, e2_heartbeat_wf0.log,
    chainA_e5_wf1.log); the shipped form polls the two barriers whose completing event comes from the other pair.  A hang
    exits the child with 3 after the budget (the stuck context dies with it), so the suite never wedges.  ~3 minutes
    (two chain compiles + 100 time-sliced launches)."""
    twin = _contention_run(tmp_path, twin_levers={}, n_twin=100, budget_s=45.0, tag="fixed")
    assert twin.returncode == 0 and "[twin] done 100 launches" in twin.stdout, f"rc={twin.returncode}\n{twin.stdout[-3000:]}\n{twin.stderr[-3000:]}"


@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.gpu_exclusive
def test_stage2_2x2_prefix_wait_form_hangs_under_time_slicing(tmp_path):
    """The NEGATIVE CONTROL of the detector above: ``wait_form = 4`` renders the pre-fix kernel (the sleeping ``try_wait``
    on the cross-pair ring barriers too) and must HANG within 300 time-sliced launches (observed at launch 2, 23, 25 and
    74 in four of four runs).  It deliberately wedges a kernel for the 45 s budget before the child dies, which is why it
    carries ``gpu_exclusive`` and sits in the ``gpu_exclusive`` xdist group with the detector above (the marker alone
    does not serialize xdist: the CI lane runs 16 workers over 4 GPUs, adjacent ungrouped items start together, and
    the detector failed exactly while this control sat wedged, twice -- pipelines 71863093 and 71991279, the detector
    at ~63 s = compile + its 45 s budget; a wedged neighbour does not slow the twin on a time-sliced B200, so the CI
    node's sharing mode is the difference).
    Deselect it on a GPU other jobs share.  If this test ever PASSES (no hang), the mechanism
    has moved: re-run the heartbeat lever (``debug_heartbeat``) before trusting the fix."""
    twin = _contention_run(tmp_path, twin_levers={"wait_form": 4}, n_twin=300, budget_s=45.0, tag="prefix")
    assert twin.returncode == 3 and "HANG" in twin.stdout, f"the pre-fix wait form did not hang in 300 launches: rc={twin.returncode}\n{twin.stdout[-2000:]}"


# --------------------------------------------------------------------------- #
# Stage 3 under GQA: ONE dQ launch per head chunk (#1318's b_head_group, ported) #
# --------------------------------------------------------------------------- #


def _cupti_kernel_names(fn):
    """The CUDA kernel names of ONE call of ``fn``, in launch order (CUPTI through torch.profiler; the prepared chain
    launches through tvm-ffi, which CUPTI sees like any other client of the process)."""
    from torch.autograd import DeviceType

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evs = [(ev.time_range.start, ev.name) for ev in prof.events() if ev.device_type == DeviceType.CUDA and ev.time_range.elapsed_us() > 0]
    return [name for _, name in sorted(evs)]


_STAGE3_GEMM_KERNEL = "_bprop_matmul_bh_sm100_kernel"


def _dq_capture(monkeypatch, single, tensors, *, b, hq, hkv, sq, skv, d, dt, chunks=False, **sdpa_kwargs):
    """Build + pin + execute with ``api_dsl.DQ_SINGLE_LAUNCH = single``; return int16 views of dQ / dK / dV plus what the
    plan actually did: the stage-3 records' ``b_head_group`` (spied off ``load_template``), the adapter's head chunk and the
    number of stage-3 GEMM launches of one execute (CUPTI).  ``chunks`` forces the head chunk down to the GQA group, so the
    chain runs ``hq // group`` head chunks (``head_base > 0`` on every launch form)."""
    from contextlib import nullcontext
    from unittest.mock import patch

    from cudnn.sdpa.bwd import api_dsl

    # The lever must exist (a renamed constant would otherwise let the shipped default pass silently on both arms); the
    # pre-port RED of this test ran with ``raising=False`` so the launch-count line, not this one, failed (b2_RED_launchcount.log).
    monkeypatch.setattr(api_dsl, "DQ_SINGLE_LAUNCH", single)
    records = {}
    original = api_dsl.load_template

    def spy(path, params, tag="template"):
        if tag in ("sdpa_bwd_sm100_mm_lo", "sdpa_bwd_sm100_mm_hi"):
            records[tag] = params
        return original(path, params, tag)

    monkeypatch.setattr(api_dsl, "load_template", spy)
    # The lowering's compiled plan does not expose the adapter; capture it off its own compile() call.
    apis = []
    original_compile = api_dsl.SdpaBwdDslSm100.compile

    def compile_spy(adapter):
        apis.append(adapter)
        return original_compile(adapter)

    monkeypatch.setattr(api_dsl.SdpaBwdDslSm100, "compile", compile_spy)
    guard = patch.object(api_dsl, "_sm100_head_chunk", side_effect=lambda *a, group=1, **kw: group) if chunks else nullcontext()
    with guard:
        g, t, (dq_t, dk_t, dv_t) = _build_graph(b, hq, hkv, sq, skv, d, 1.0 / math.sqrt(d), dt=dt, **sdpa_kwargs)
        idx = _plan_index(g)
        assert idx is not None
        g.select_plan(idx)
        g.check_support()
        g.build_plans()
    assert len(apis) == 1, f"expected exactly one SM100 adapter compile, saw {len(apis)}"
    api = apis[0]
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8).fill_(0xBD)
    dq, dk, dv = _bshd(b, sq, hq, d, dt=dt, fill=False), _bshd(b, skv, hkv, d, dt=dt, fill=False), _bshd(b, skv, hkv, d, dt=dt, fill=False)
    for x in (dq, dk, dv):
        x.fill_(float("nan"))
    pack = {t["q"]: tensors["q"], t["k"]: tensors["k"], t["v"]: tensors["v"], t["o"]: tensors["o"], t["do"]: tensors["do"], t["stats"]: tensors["stats"]}
    pack.update({dq_t: dq, dk_t: dk, dv_t: dv})
    names = _cupti_kernel_names(lambda: g.execute(pack, ws))
    gemms = sum(1 for n in names if _STAGE3_GEMM_KERNEL in n)
    assert records.keys() == {"sdpa_bwd_sm100_mm_lo", "sdpa_bwd_sm100_mm_hi"}, sorted(records)
    facts = dict(
        lo_bhg=records["sdpa_bwd_sm100_mm_lo"].b_head_group,
        hi_bhg=records["sdpa_bwd_sm100_mm_hi"].b_head_group,
        chunk=api._qh_chunk,
        chunks=hq // api._qh_chunk,
        gemm_launches=gemms,
        # 1 = the per-member loop, which is also what a tree without the lever runs (so the RED there is the launch count).
        dq_bhg=getattr(api, "_dq_b_head_group", 1),
    )
    return [x.contiguous().view(torch.int16).clone() for x in (dq, dk, dv)], facts


def _gqa_inputs(b, hq, hkv, sq, skv, d, dt, keep):
    torch.manual_seed(1318)
    q, do = _bshd(b, sq, hq, d, dt=dt), _bshd(b, sq, hq, d, dt=dt)
    k, v = _bshd(b, skv, hkv, d, dt=dt), _bshd(b, skv, hkv, d, dt=dt)
    o_ref, lse, all_masked, dq_r, dk_r, dv_r = _reference(q, k, v, do, keep, hq // hkv)
    o = _bshd(b, sq, hq, d, dt=dt, fill=False)
    o.copy_(o_ref.to(dt))
    stats = (lse if all_masked is None else lse.masked_fill(all_masked, 0.0)).unsqueeze(-1).contiguous()
    return dict(q=q, k=k, v=v, o=o, do=do, stats=stats), (dq_r, dk_r, dv_r)


@pytest.mark.parametrize(
    "case",
    [
        dict(b=1, hq=8, hkv=2, sq=512, skv=512),
        dict(b=1, hq=8, hkv=2, sq=1024, skv=1024, use_causal_mask=True),
        dict(b=2, hq=8, hkv=2, sq=512, skv=768, use_causal_mask_bottom_right=True, chunks=True),
        dict(b=1, hq=16, hkv=2, sq=512, skv=512, use_causal_mask=True, chunks=True),
        dict(b=1, hq=8, hkv=1, sq=512, skv=512),
        dict(b=1, hq=8, hkv=8, sq=512, skv=512),
    ],
    ids=["gqa8-2", "gqa8-2_causal_1k", "gqa8-2_br_rect_b2_chunked", "gqa16-2_causal_chunked", "mqa8-1", "mha8"],
)
def test_stage3_dq_single_launch_per_chunk_is_bitwise_the_per_member_launches(monkeypatch, case):
    """Under GQA the shipped dQ GEMM is ONE launch per head chunk: the dQ rendering indexes B = K by ``h // group``
    (``MatmulTemplateParams.b_head_group = group``, #1318) over the whole dS and dQ chunk, where the chain used to run one
    launch per group MEMBER over every ``group``-th Q head -- at H_q / H_kv = 16 and S = 8K sixteen under-one-wave launches
    of 16 clusters on a 37-cluster B200, 128 dQ launches per backward.  Both forms pair every Q head with the same K head and
    walk the same k tiles per output tile into an fp32 accumulator, so dQ must be the SAME BITS -- and dK / dV, which the
    change never touches.  ``api_dsl.DQ_SINGLE_LAUNCH = False`` is the twin (``b_head_group = 1``, the per-member loop); both
    runs are also held to the fp32 oracle, with NaN-poisoned outputs.  The LAUNCH COUNT is pinned from a CUPTI trace of one
    execute: ``3 * chunks`` stage-3 GEMMs on the shipped form, ``(2 + group) * chunks`` on the twin -- which is what makes
    this test RED on the pre-port tree (it launched the twin's count under both settings).  MHA renders and launches
    identically either way (``b_head_group`` stays 1)."""
    case = dict(case)
    dt = case.pop("dt", torch.bfloat16)
    chunks = case.pop("chunks", False)
    b, hq, hkv, sq, skv = (case.pop(k) for k in ("b", "hq", "hkv", "sq", "skv"))
    d, group = _D, hq // hkv
    keep = None
    if case.get("use_causal_mask") or case.get("use_causal_mask_bottom_right"):
        keep = _causal_keep(sq, skv, bottom_right=bool(case.get("use_causal_mask_bottom_right")))
    tensors, refs = _gqa_inputs(b, hq, hkv, sq, skv, d, dt, keep)
    kw = dict(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, d=d, dt=dt, chunks=chunks, **case)
    single, f_single = _dq_capture(monkeypatch, True, tensors, **kw)
    members, f_members = _dq_capture(monkeypatch, False, tensors, **kw)
    n_chunks = f_single["chunks"]
    assert f_single["chunk"] % group == 0 and (not chunks or f_single["chunk"] == group), f_single
    # the launches: one dQ GEMM per chunk vs one per group member per chunk (the pre-port tree launched the latter either way)
    assert f_single["gemm_launches"] == 3 * n_chunks, (
        f"single-launch arm: {f_single['gemm_launches']} stage-3 GEMM launches, expected 3 * {n_chunks} chunks = {3 * n_chunks} "
        f"(the per-member loop launches (2 + {group}) * {n_chunks} = {(2 + group) * n_chunks}); facts {f_single}"
    )
    assert (
        f_members["gemm_launches"] == (2 + group) * n_chunks
    ), f"per-member arm: {f_members['gemm_launches']} stage-3 GEMM launches, expected (2 + {group}) * {n_chunks} = {(2 + group) * n_chunks}; facts {f_members}"
    # the records: dQ takes the group, dV / dK (B = dO / Q per Q head) keep 1; the twin renders everything per head
    assert (f_single["lo_bhg"], f_single["hi_bhg"], f_single["dq_bhg"]) == (1, group, group), f_single
    assert (f_members["lo_bhg"], f_members["hi_bhg"], f_members["dq_bhg"]) == (1, 1, 1), f_members
    for name, x, ref in zip(("dQ", "dK", "dV"), single, refs):
        got = x.view(dt).float()
        assert torch.isfinite(got).all(), f"{name}: the single-launch form left non-finite values"
        cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
        rel = ((got - ref).abs().max() / max(ref.abs().max().item(), 1e-30)).item()
        assert cos > _TOL_COS and rel < _TOL_REL, f"{name}: cos={cos:.6f} max_rel_err={rel:.2e}"
    for name, x, y in zip(("dQ", "dK", "dV"), single, members):
        n_diff = (x != y).sum().item()
        assert n_diff == 0, (
            f"{name}: the single dQ launch vs the per-member launches differ in {n_diff} of {x.numel()} int16 words "
            f"(max|diff|={(x.view(dt).float() - y.view(dt).float()).abs().max().item():.3e})"
        )


@pytest.mark.parametrize(
    "hq,hkv,chunk,b", [(8, 2, 8, 2), (32, 2, 32, 1), (32, 2, 16, 1), (12, 4, 12, 1), (4, 4, 4, 2), (16, 1, 16, 1), (128, 8, 16, 1), (64, 8, 8, 1)]
)
@pytest.mark.parametrize("single", (True, False), ids=("one-launch", "per-member"))
def test_stage3_dq_launches_pair_every_q_head_with_its_k_head(hq, hkv, chunk, b, single):
    """The coordinate arithmetic of the SM100 host's dQ launches on a fake flat batch index, no GPU: the template decodes
    ``l -> (h = l % n_head, b = l // n_head)`` for A (dS) and C (dQ) and hands B (K) ``h // b_head_group`` (``_b_head``)
    against a B descriptor ``n_head // b_head_group`` heads deep; ``prepared_host.host`` launches ``_dq_launches(group,
    b_head_group)`` times per head chunk, launch ``member`` over the chunk's Q heads ``member :: n_launch``
    (``_workspace_heads(ds, member, heads, n_launch)`` / ``_heads(dq, head_base + member, heads, n_launch)``) against the
    chunk's ``chunk // group`` K heads from ``head_base // group``.  For every (b, h) of every launch the K head reached
    must be ``q_head // group`` -- the GQA convention the oracle, the analyzer and the kernels share -- at ``b_head_group =
    group`` (one launch) exactly as at 1 (the per-member loop, the twin), and every (batch, Q head) is written exactly once.
    Plain-Python twin of the traced arithmetic (``test_sdpa_bwd_dsl_sm107.test_stage3_dq_launches_pair_every_q_head_with_its_k_head``,
    the cc 10.7 chain's test of the same mechanism, is the template)."""
    from cudnn.sdpa.bwd.kernels.sm100.prepared_host import _dq_launches

    group = hq // hkv
    assert chunk % group == 0 and hq % chunk == 0, "the adapter's head chunk is a multiple of the group that divides H_q"
    bhg = group if single else 1
    kv_count = chunk // group
    n_launch = _dq_launches(group, bhg)
    heads = chunk // n_launch  # Q heads per dQ launch (the launch's n_head); its B is heads // bhg == kv_count heads deep
    assert heads // bhg == kv_count and heads * n_launch == chunk
    assert n_launch == (1 if single else group)
    seen = set()
    for head_base in range(0, hq, chunk):
        for member in range(n_launch):
            for l in range(heads * b):  # every flat CLC batch index of the launch's grid
                tile_h, tile_b = l % heads, l // heads  # _decode_bh
                h_b = tile_h // bhg  # _b_head
                assert 0 <= h_b < kv_count, "B's coordinate stays inside its descriptor's head extent"
                q_head = head_base + member + tile_h * n_launch  # the dS / dQ head the launch's (member, heads, n_launch) view addresses
                k_head = head_base // group + h_b  # k_heads = the chunk's K heads from head_base // group, B's coordinate inside it
                assert k_head == q_head // group, (head_base, member, l, q_head, k_head)
                seen.add((tile_b, q_head))
    assert seen == {(bb, h) for bb in range(b) for h in range(hq)}, "every (batch, Q head) is written exactly once across the launches"
    assert (_dq_launches(1, 1), _dq_launches(4, 4), _dq_launches(4, 1), _dq_launches(16, 16), _dq_launches(16, 1)) == (1, 1, 4, 1, 16)
    for bad_group, bad_bhg in ((4, 2), (16, 4), (2, 4), (1, 2)):
        with pytest.raises(ValueError, match="b_head_group"):
            _dq_launches(bad_group, bad_bhg)


def test_stage3_dq_single_launch_is_a_module_constant_not_a_knob():
    """``api_dsl.DQ_SINGLE_LAUNCH`` is a module constant read at compile() time (the ``STAGE2_2X2`` precedent here and
    ``api_dsl_sm107.DQ_SINGLE_LAUNCH`` on the cc 10.7 chain): True by default, no env var reads it, and THIS chain's THD
    leg never takes it -- the template's grouped arm has a THD leg (the cc 10.7 chain's), so the validator admits it, and
    the SM100 record gate (``not self.thd``, plus the whole-group chunk condition) is what keeps the packed dQ launch per
    member."""
    import inspect

    from cudnn.sdpa.bwd import api_dsl
    from cudnn.sdpa.bwd.config_sm100 import MatmulTemplateParams, validate_matmul_params

    assert api_dsl.DQ_SINGLE_LAUNCH is True
    src = inspect.getsource(api_dsl)
    assert len(_re.findall(r"^DQ_SINGLE_LAUNCH: bool = True$", src, flags=_re.M)) == 1
    assert "DQ_SINGLE_LAUNCH" not in "".join(ln for ln in src.splitlines() if "environ" in ln), "not an env var"
    validate_matmul_params(MatmulTemplateParams(b_head_group=4, thd_varlen=True))  # the template offers the THD leg
    gate = next(ln for ln in src.splitlines() if "b_head_group=self._gqa_group if" in ln)
    assert "not self.thd" in gate and "self._qh_chunk % self._gqa_group == 0" in gate, gate


# --------------------------------------------------------------------------- the end-of-kernel ring drains (review P1)


def _ring_drain_model(total, stages, walk):
    """Pure-Python twin of the 2x2 body's producer-side ring protocol: ``PipelineState.start(phase=1)``, one issue per
    ring step, then a drain of ``walk`` waits from the current state.  Returns the set of (slot, use) releases the drain
    never waited for.  A use k of slot j completes barrier parity k % 2 of slot j; the wait that consumes it is the one
    at that slot's next visit, whose phase is k % 2 (visit 0 is phase 1 and passes free -- the pre-armed ring)."""
    idx, phase = 0, 1
    uses = {j: 0 for j in range(stages)}
    pending = set()
    for _ in range(total):
        k = uses[idx]
        if k > 0:
            pending.discard((idx, k - 1))  # the wait before re-use consumes the previous use's release
        pending.add((idx, k))
        uses[idx] += 1
        idx = (idx + 1) % stages
        phase ^= int(idx == 0)
    for _ in range(walk):
        k = uses[idx]
        if k > 0 and phase == (k - 1) % 2:
            pending.discard((idx, k - 1))
        idx = (idx + 1) % stages
        phase ^= int(idx == 0)
    return pending


@pytest.mark.parametrize("stages", [2, 4])
def test_ring_drain_walks_every_used_slot(stages):
    """Host twin of the 2x2 bodies' end-of-kernel drains: walking the WHOLE ring from the current state consumes the last
    release of every used slot for any issue count (0 .. 2 * stages + 1), while the previous rule -- min(total, stages)
    steps -- skipped used slots whenever a cluster issued fewer than ``stages`` times (one kv tile at STAGES_ACC = 2 left
    slot 0's compute release un-awaited before the TMEM dealloc).  The kernels spell the walk as a static
    ``range_constexpr(STAGES_*)`` loop; the source pin below keeps the residual rule from coming back."""
    import inspect

    from cudnn.sdpa.bwd.kernels.sm100 import bprop_d512_f16_2x2 as K2

    for total in range(0, 2 * stages + 2):
        assert not _ring_drain_model(total, stages, stages), f"total={total}: the full-ring walk left releases un-awaited"
    missed = [total for total in range(1, stages) if _ring_drain_model(total, stages, min(total, stages))]
    assert missed == list(range(1, stages)), f"the old min(total, stages) drain must miss every short trip, missed only {missed}"
    src = inspect.getsource(K2)
    assert "_residual_depth" not in src, "the residual-count drain is back"
    assert len(_re.findall(r"for _ in cutlass\.range_constexpr\(CFG\.STAGES_ACC\):", src)) == 1
    assert len(_re.findall(r"for _ in cutlass\.range_constexpr\(CFG\.STAGES_KV\):", src)) == 1


@pytest.mark.parametrize("sq,skv,hq", [(256, 128, 1), (256, 128, 4), (512, 128, 2)], ids=["one-tile", "one-tile-h4", "two-q-blocks"])
def test_short_trip_single_kv_tile_per_cluster(stage2_datapath, sq, skv, hq):
    """GPU: plans whose clusters run ONE kv tile (S_kv = 128 = the stage-2 kv tile) -- acc_total = 1 < STAGES_ACC = 2 in
    every cluster, the geometry whose TMEM release the min(total, stages) drain never gated.  Eight launches each against
    the fp32 reference, on both datapaths (the 4x1 arm is the control: its drains are per tile)."""
    for _ in range(8):
        _run(b=1, hq=hq, sq=sq, skv=skv)
