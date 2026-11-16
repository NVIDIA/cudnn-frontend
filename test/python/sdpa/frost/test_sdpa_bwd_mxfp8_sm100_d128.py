# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm100_d128_mxfp8``: the SM100 / SM103 D=128 block-scale MXFP8 backward (a CuTeDSL chain).

ACCEPT tests run the row end to end through the graph API (``sdpa_mxfp8_backward`` over BHSD-compact tensors, the
row pinned by plan name) against the repo's MXFP8 backward reference (``sdpa.mxfp8_ref``), and compare its error
against the FP32 reference with the cuDNN backend's on the same graph where the backend serves it.  Determinism
tests use raw-byte equality: repeats into poisoned outputs (eager and CUDA-graph replay) and batch invariance (the
B = 2 problem against two B = 1 problems on the same data).  REJECT tests assert the row's declines.

Opt-in row: the module enables ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES`` for its tests.
"""

from __future__ import annotations

import math

import pytest
import torch

import cudnn

_ENGINE = "sdpa_bwd_sm100_d128_mxfp8"
_D = 128
_BLOCK = 32
_TOL_COS = 0.9995

requires_sm100_family = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason=f"{_ENGINE} targets compute capability 10.0 / 10.3",
)


def _dsl_below_floor() -> bool:
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
    from cudnn.sdpa.bwd.kernels.sm100._geometry_d128_mxfp8 import CUTEDSL_FLOOR

    installed, version = cutedsl_state()
    return not installed or cutedsl_too_old(version, floor=CUTEDSL_FLOOR)


requires_dsl = pytest.mark.skipif(_dsl_below_floor(), reason=f"{_ENGINE} needs nvidia-cutlass-dsl at the row's floor (Rule 7)")
pytestmark = [requires_sm100_family, requires_dsl]


@pytest.fixture(autouse=True)
def _opt_in(monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


def _cdiv(a, b):
    return -(-a // b)


def _bhsd_stride(shape):
    b, h, s, d = shape
    return [h * s * d, s * d, d, 1]


def _bshd_stride(shape):
    b, h, s, d = shape
    return [s * h * d, d, h * d, 1]


def _quantize(t, b, h, s, d):
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    d8, sf_d_ref, sf_d_swz, s8, sf_s_ref, sf_s_swz = quantize_to_mxfp8(t, b, h, s, d, _BLOCK, torch.float8_e4m3fn, with_ref=True)
    return dict(row=d8.contiguous(), col=s8.contiguous(), sf_row_ref=sf_d_ref, sf_col_ref=sf_s_ref, sf_row=sf_d_swz.contiguous(), sf_col=sf_s_swz.contiguous())


def _sf_dims(b, h, s, d):
    row = (b, h, _cdiv(s, 128) * 128, _cdiv(_cdiv(d, _BLOCK), 4) * 4)
    col = (b, h, _cdiv(_cdiv(s, _BLOCK), 4) * 4, _cdiv(d, 128) * 128)
    return row, col


def _build_graph(b, hq, hkv, s, scale=None, stride_fn=_bhsd_stride, declare_amax=False, sq=None, stats_stride=None, **sdpa_kwargs):
    sq = s if sq is None else sq
    fp8 = cudnn.data_type.FP8_E4M3
    g = cudnn.pygraph(io_data_type=fp8, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    shq, shk = (b, hq, sq, _D), (b, hkv, s, _D)
    t = {}
    for name, sh in (("q", shq), ("q_T", shq), ("k", shk), ("k_T", shk), ("v", shk), ("dO", shq), ("dO_T", shq)):
        t[name] = g.tensor(name=name, dim=list(sh), stride=stride_fn(sh), data_type=fp8)
    for name in ("o_f16", "dO_f16"):
        t[name] = g.tensor(name=name, dim=list(shq), stride=stride_fn(shq), data_type=cudnn.data_type.BFLOAT16)
    stats_stride = [hq * sq, sq, 1, 1] if stats_stride is None else list(stats_stride)
    t["stats"] = g.tensor(name="stats", dim=[b, hq, sq, 1], stride=stats_stride, data_type=cudnn.data_type.FLOAT)
    q_row, q_col = _sf_dims(b, hq, sq, _D)
    k_row, k_col = _sf_dims(b, hkv, s, _D)
    for name, dims in (("sf_q", q_row), ("sf_q_T", q_col), ("sf_k", k_row), ("sf_k_T", k_col), ("sf_v", k_row), ("sf_dO", q_row), ("sf_dO_T", q_col)):
        t[name] = g.tensor(
            name=name,
            dim=list(dims),
            stride=[dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1],
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
    if scale is not None:
        sdpa_kwargs["attn_scale"] = scale
    outs = g.sdpa_mxfp8_backward(
        name="bwd",
        q=t["q"],
        q_T=t["q_T"],
        k=t["k"],
        k_T=t["k_T"],
        v=t["v"],
        o_f16=t["o_f16"],
        dO_f16=t["dO_f16"],
        dO=t["dO"],
        dO_T=t["dO_T"],
        stats=t["stats"],
        descale_q=t["sf_q"],
        descale_q_T=t["sf_q_T"],
        descale_k=t["sf_k"],
        descale_k_T=t["sf_k_T"],
        descale_v=t["sf_v"],
        descale_dO=t["sf_dO"],
        descale_dO_T=t["sf_dO_T"],
        **sdpa_kwargs,
    )
    dq, dk, dv, amax_dq, amax_dk, amax_dv = outs
    for out, sh in ((dq, shq), (dk, shk), (dv, shk)):
        out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim(list(sh)).set_stride(stride_fn(sh))
    wanted = ("dQ", "dK", "dV") if declare_amax is True else tuple(declare_amax or ())
    for name, a in zip(("dQ", "dK", "dV"), (amax_dq, amax_dk, amax_dv)):
        a.set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
        if name in wanted:
            a.set_output(True)
    g.validate()
    g.build_operation_graph()
    return g, t, outs


def _plan_names(g):
    return [g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]


def _plan_index(g, name=_ENGINE, backend=False):
    for i, pn in enumerate(_plan_names(g)):
        if backend and not pn.startswith("sdpa_"):
            return i
        if not backend and (pn == name or pn.startswith(name + "[")):
            return i
    return None


def _inputs(b, hq, hkv, s, seed=0, scale=None, causal=False):
    from sdpa.mxfp8_ref import compute_ref

    torch.manual_seed(seed)
    dev = "cuda"
    scale = 1.0 / math.sqrt(_D) if scale is None else scale
    q, dO = torch.randn(b, hq, s, _D, device=dev), torch.randn(b, hq, s, _D, device=dev)
    k, v = torch.randn(b, hkv, s, _D, device=dev), torch.randn(b, hkv, s, _D, device=dev)
    Q, K, V, DO = (_quantize(x, b, h, s, _D) for x, h in ((q, hq), (k, hkv), (v, hkv), (dO, hq)))
    band = dict(right_bound=0, diag_align=cudnn.diagonal_alignment.TOP_LEFT) if causal else {}
    o_ref, stats = compute_ref(Q["row"], K["row"], V["col"], Q["sf_row_ref"], K["sf_row_ref"], V["sf_col_ref"], scale, output_type=torch.bfloat16, **band)
    return dict(
        Q=Q, K=K, V=V, DO=DO, o=o_ref.to(torch.bfloat16).contiguous(), dO=dO.to(torch.bfloat16).contiguous(), stats=stats.contiguous(), scale=scale,
        raw=(q, k, v, dO), causal=causal,
    )  # fmt: skip


def _execute(g, t, outs, inp, b, hq, hkv, s, amax=False):
    dev = "cuda"
    dq_t, dk_t, dv_t, *amax_t = outs
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    dq = torch.full((b, hq, s, _D), float("nan"), device=dev, dtype=torch.bfloat16)
    dk = torch.full((b, hkv, s, _D), float("nan"), device=dev, dtype=torch.bfloat16)
    dv = torch.full((b, hkv, s, _D), float("nan"), device=dev, dtype=torch.bfloat16)
    Q, K, V, DO = inp["Q"], inp["K"], inp["V"], inp["DO"]
    pack = {
        t["q"]: Q["row"],
        t["q_T"]: Q["col"],
        t["k"]: K["row"],
        t["k_T"]: K["col"],
        t["v"]: V["row"],
        t["o_f16"]: inp["o"],
        t["dO_f16"]: inp["dO"],
        t["dO"]: DO["row"],
        t["dO_T"]: DO["col"],
        t["stats"]: inp["stats"],
        t["sf_q"]: Q["sf_row"],
        t["sf_q_T"]: Q["sf_col"],
        t["sf_k"]: K["sf_row"],
        t["sf_k_T"]: K["sf_col"],
        t["sf_v"]: V["sf_row"],
        t["sf_dO"]: DO["sf_row"],
        t["sf_dO_T"]: DO["sf_col"],
        dq_t: dq,
        dk_t: dk,
        dv_t: dv,
    }
    wanted = ("dQ", "dK", "dV") if amax is True else tuple(amax or ())
    amaxes = {}
    for name, a_t in zip(("dQ", "dK", "dV"), amax_t):
        if name in wanted:
            amaxes[name] = torch.full((1, 1, 1, 1), -1.0, device=dev)
            pack[a_t] = amaxes[name]
    g.execute(pack, ws)
    torch.cuda.synchronize()
    return (dq, dk, dv), amaxes, pack, ws


def _reference(inp, b, hq, hkv, s):
    from sdpa.mxfp8_ref import compute_ref_backward

    Q, K, V, DO = inp["Q"], inp["K"], inp["V"], inp["DO"]
    return compute_ref_backward(
        Q["row"], Q["col"], K["row"], K["col"], V["row"], inp["o"], inp["dO"], DO["row"], DO["col"], inp["scale"],
        Q["sf_row_ref"], Q["sf_col_ref"], K["sf_row_ref"], K["sf_col_ref"], V["sf_row_ref"], DO["sf_row_ref"], DO["sf_col_ref"],
        torch_otype=torch.bfloat16, stats=inp["stats"],
        **(dict(right_bound=0, diag_align=cudnn.diagonal_alignment.TOP_LEFT) if inp.get("causal") else {}),
    )[:3]  # fmt: skip


def _run(b=1, hq=8, hkv=2, s=256, scale=None, amax=False, seed=0, **sdpa_kwargs):
    # S_q == S_kv: a bottom-right causal graph has the top-left mask, so the same reference serves both anchors
    causal = bool(sdpa_kwargs.get("use_causal_mask") or sdpa_kwargs.get("use_causal_mask_bottom_right"))
    inp = _inputs(b, hq, hkv, s, seed=seed, scale=scale, causal=causal)
    g, t, outs = _build_graph(b, hq, hkv, s, scale=inp["scale"], declare_amax=amax, **sdpa_kwargs)
    g.create_execution_plans([cudnn.heur_mode.A])
    idx = _plan_index(g)
    assert idx is not None, f"{_ENGINE} not offered; plans = {_plan_names(g)}"
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    got, amaxes, pack, ws = _execute(g, t, outs, inp, b, hq, hkv, s, amax=amax)
    ref = _reference(inp, b, hq, hkv, s)
    for name, x, r in zip(("dQ", "dK", "dV"), got, ref):
        assert torch.isfinite(x).all(), f"{name} has non-finite values"
        cos = torch.nn.functional.cosine_similarity(x.float().flatten(), r.float().flatten(), dim=0).item()
        assert cos > _TOL_COS, f"{name}: cos={cos:.6f}"
    for name, x in zip(("dQ", "dK", "dV"), got):
        if name in amaxes:
            a, want = amaxes[name].item(), x.float().abs().max().item()  # amax is of the fp32 values before the bf16 cast
            assert abs(a - want) <= want * 2.0**-8, f"amax_{name}: {a} vs max |{name}| {want}"
    return SimpleRun(g, t, outs, inp, got, pack, ws, amaxes)


class SimpleRun:
    def __init__(self, g, t, outs, inp, got, pack, ws, amaxes=None):
        self.g, self.t, self.outs, self.inp, self.got, self.pack, self.ws = g, t, outs, inp, got, pack, ws
        self.amaxes = amaxes or {}

    @property
    def outputs(self):
        """Every output the graph wrote: dQ, dK, dV and the requested amax values."""
        return tuple(self.got) + tuple(self.amaxes[k] for k in sorted(self.amaxes))


def _bits(x):
    """Raw bytes as integers (bf16 gradients, fp32 amax), for raw-byte equality."""
    return x.view(torch.int16 if x.element_size() == 2 else torch.int32)


# --------------------------------------------------------------------------- #
# ACCEPT                                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.L0
@pytest.mark.parametrize("hq,hkv", [(8, 2), (4, 4), (8, 1)], ids=["gqa", "mha", "mqa"])
def test_dense(hq, hkv):
    _run(hq=hq, hkv=hkv, s=512)


@pytest.mark.L0
@pytest.mark.parametrize("s", [100, 384, 1000, 1152], ids=lambda s: f"s{s}")
def test_sequence_tails(s):
    """S % 256 != 0: 3D maps clipped at S, -inf LSE rows, dQ's last-tile key mask; 384 / 1152 have an odd block count."""
    _run(hq=8, hkv=2, s=s)


@pytest.mark.L0
def test_batch():
    _run(b=2, hq=4, hkv=2, s=640)


@pytest.mark.L0
def test_attn_scale():
    _run(hq=4, hkv=2, s=256, scale=0.5)


@pytest.mark.L0
@pytest.mark.parametrize("hq,hkv,s", [(8, 2, 512), (4, 4, 1024), (8, 1, 768), (8, 2, 1000), (8, 2, 1152), (4, 4, 384)],
                         ids=["gqa-512", "mha-1024", "mqa-768", "gqa-1000", "gqa-1152", "mha-384"])  # fmt: skip
def test_causal(hq, hkv, s):
    """Top-left causal: dK/dV walks from the diagonal (lockstep walk groups mask their later blocks' upper tiles), dQ walks
    stop at it; tails and odd block counts included (mha-384: walk split off for causal MHA with an odd block count)."""
    _run(hq=hq, hkv=hkv, s=s, use_causal_mask=True)


@pytest.mark.L0
@pytest.mark.parametrize("hq,hkv,s", [(8, 2, 512), (8, 2, 1000), (8, 1, 768)], ids=["gqa-512", "gqa-1000-tail", "mqa-768"])
def test_bottom_right_causal_is_top_left(hq, hkv, s):
    """S_q == S_kv puts the bottom-right diagonal (offset S_kv - S_q) on the top-left one: a bottom-right causal graph is
    served by the same kernels, passes the reference and writes the top-left graph's bytes."""
    bottom_right = _run(hq=hq, hkv=hkv, s=s, seed=11, use_causal_mask_bottom_right=True)
    top_left = _run(hq=hq, hkv=hkv, s=s, seed=11, use_causal_mask=True)
    for name, a, b1 in zip(("dQ", "dK", "dV"), bottom_right.outputs, top_left.outputs):
        assert torch.equal(_bits(a), _bits(b1)), name


@pytest.mark.L0
def test_causal_batch_amax():
    _run(b=2, hq=8, hkv=2, s=640, amax=True, use_causal_mask=True)


@pytest.mark.L1
def test_causal_issue_heads():
    for s in (1024, 2048):
        _run(hq=88, hkv=8, s=s, use_causal_mask=True)


@pytest.mark.L0
def test_amax():
    _run(hq=8, hkv=2, s=1000, amax=True)


@pytest.mark.L0
def test_amax_partial():
    """Only amax_dQ real: the virtual two are written to workspace scratch, never to a missing tensor."""
    _run(hq=8, hkv=2, s=512, amax=("dQ",))


@pytest.mark.L0
def test_deterministic_request():
    """use_deterministic_algorithm=True: the row serves it (every plan is bitwise repeatable; this one is also batch
    invariant)."""
    _run(hq=8, hkv=2, s=1024, use_deterministic_algorithm=True)


@pytest.mark.L0
def test_large_batch_plans():
    """B = 8: the throughput plan (no deterministic request) keeps the dK/dV walk unsplit, the batch-invariant plan
    splits it per batch element; both are correct."""
    from cudnn.frost.occupancy import max_active_clusters
    from cudnn.sdpa.bwd.kernels.sm100 import _geometry_d128_mxfp8 as G

    for det in (False, True):
        _run(b=8, hq=4, hkv=2, s=1024, use_deterministic_algorithm=det)
    device = torch.cuda.current_device()

    def caps(n):
        return max_active_clusters(n, device)

    g8 = G.geometry(1024, 8, 4, 2, 1 / math.sqrt(_D), False, False)
    g1 = G.geometry(1024, 1, 4, 2, 1 / math.sqrt(_D), False, False)
    # the deterministic plan decides the walk split from one batch element (no split count is assumed: it follows the
    # device's cluster capacity)
    assert G.plan(g8, caps, batch_invariant=True).split == G.plan(g1, caps, batch_invariant=True).split
    assert G.plan(g8, caps, batch_invariant=False).split <= G.plan(g8, caps, batch_invariant=True).split


@pytest.mark.L1
def test_issue_heads():
    """The issue-659 heads (88 / 8) at the walk-split and the plain dK/dV plans."""
    for s in (1024, 2048):
        _run(hq=88, hkv=8, s=s)


def _fp32_reference(raw, scale, hq, hkv, causal=False):
    """dQ / dK / dV in fp32 from the UNQUANTIZED operands (TF32 off): the yardstick both kernels' errors are measured
    against, so input quantization is in the error and an engine is judged by matching the backend's error."""
    q, k, v, do = (x.float() for x in raw)
    grp = hq // hkv
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        dq, dk, dv = torch.empty_like(q), torch.zeros_like(k), torch.zeros_like(v)
        for b in range(q.shape[0]):
            for h in range(hq):
                kv = h // grp
                sc = (q[b, h] @ k[b, kv].T) * scale
                if causal:
                    sc = sc.masked_fill(torch.ones_like(sc, dtype=torch.bool).triu(1), float("-inf"))
                p = torch.softmax(sc, dim=-1)
                o = p @ v[b, kv]
                ds = p * ((do[b, h] @ v[b, kv].T) - (do[b, h] * o).sum(-1, keepdim=True))
                dv[b, kv] += p.T @ do[b, h]
                dq[b, h] = (ds @ k[b, kv]) * scale
                dk[b, kv] += (ds.T @ q[b, h]) * scale
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return dq, dk, dv


@pytest.mark.L1
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("s", [1024, 1000], ids=lambda s: f"s{s}")
def test_error_matches_backend(s, causal):
    """Relative L2 error vs the fp32 reference of the unquantized operands within 1 % of the cuDNN backend's on the same
    graph (input quantization dominates both; this is the project's accuracy criterion for low-precision kernels)."""
    b, hq, hkv = 1, 16, 2
    inp = _inputs(b, hq, hkv, s, causal=causal)
    ref = _fp32_reference(inp["raw"], inp["scale"], hq, hkv, causal=causal)
    errs = {}
    for backend in (False, True):
        g, t, outs = _build_graph(b, hq, hkv, s, scale=inp["scale"], **({"use_causal_mask": True} if causal else {}))
        g.create_execution_plans([cudnn.heur_mode.A])
        idx = _plan_index(g, backend=backend)
        if idx is None:
            pytest.skip("the cuDNN backend offers no plan for this graph") if backend else pytest.fail(f"{_ENGINE} not offered")
        g.select_plan(idx)
        g.check_support()
        g.build_plans()
        got = _execute(g, t, outs, inp, b, hq, hkv, s)[0]
        errs[backend] = [((x.float() - r).norm() / r.norm()).item() for x, r in zip(got, ref)]
    for name, mine, theirs in zip(("dQ", "dK", "dV"), errs[False], errs[True]):
        assert mine <= theirs * 1.01, f"{name}: rel-L2 vs fp32 {mine:.5f}; backend {theirs:.5f}"


# --------------------------------------------------------------------------- #
# DETERMINISM (raw bytes)                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.L0
@pytest.mark.parametrize(
    "s,causal,amax",
    [(1024, False, False), (1152, False, False), (1024, True, False), (1000, True, True)],
    ids=["s1024", "s1152", "s1024-causal", "s1000-causal-amax"],
)
def test_repeat_and_graph_replay(s, causal, amax):
    """dQ / dK / dV (and the amax outputs, when requested) raw-byte equal across eager calls and CUDA-graph replays into
    outputs poisoned between runs."""
    run = _run(hq=8, hkv=2, s=s, amax=amax, **({"use_causal_mask": True} if causal else {}))
    expected = tuple(x.clone() for x in run.outputs)
    replay = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()  # graphs cannot be captured on the default stream
    side.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.graph(replay, stream=side):
            run.g.execute(run.pack, run.ws)
        for i in range(6):
            for x in run.outputs:
                x.fill_(float("nan") if i % 4 < 2 else 123.0)
            if i % 2:
                replay.replay()
            else:
                run.g.execute(run.pack, run.ws)
            torch.cuda.synchronize()
            for got, exp in zip(run.outputs, expected):
                assert torch.equal(_bits(got), _bits(exp))
    finally:
        replay.reset()


# One golden per distinct numerical path of the chain (skills/determinism_tests): (B, H_q, H_kv, S, causal, amax,
# use_deterministic_algorithm). On B200 and B300 the first plans the walk-split dK/dV (two CTAs per key block, fp32 partials
# added in a fixed order) and the second the unsplit walk -- the plan's only rounding-relevant choice.
_GOLDEN_CELLS = {
    "dense-s1024-split": (1, 8, 2, 1024, False, False, True),
    "dense-b8-unsplit": (8, 4, 2, 1024, False, False, False),
    "causal-s1152": (1, 8, 2, 1152, True, False, True),
    "tail-amax-s1000": (1, 8, 2, 1000, False, True, True),
}


@pytest.mark.L0
@pytest.mark.parametrize("cell", sorted(_GOLDEN_CELLS))
def test_golden(cell):
    """Determinism golden: the reference check (in ``_run``) and repeated replays pass, then the input and output bytes
    are compared with -- or, under ``CUDNN_DETERMINISM_RECORD``, recorded into -- ``determinism_goldens/``."""
    from determinism_utils import assert_golden, assert_replays

    b, hq, hkv, s, causal, amax, deterministic = _GOLDEN_CELLS[cell]
    kwargs = {}
    if causal:
        kwargs["use_causal_mask"] = True
    if deterministic:
        kwargs["use_deterministic_algorithm"] = True
    run = _run(b=b, hq=hq, hkv=hkv, s=s, amax=amax, seed=29, **kwargs)
    outputs = {"dQ": run.got[0], "dK": run.got[1], "dV": run.got[2], **{f"amax_{k}": v for k, v in run.amaxes.items()}}

    def execute():
        for x in outputs.values():
            x.fill_(float("nan"))
        run.g.execute(run.pack, run.ws)
        return outputs

    got = assert_replays(execute, repeats=3)
    Q, K, V, DO = run.inp["Q"], run.inp["K"], run.inp["V"], run.inp["DO"]
    inputs = {
        "q": Q["row"], "q_T": Q["col"], "k": K["row"], "k_T": K["col"], "v": V["row"], "dO": DO["row"], "dO_T": DO["col"],
        "o": run.inp["o"], "dO_f16": run.inp["dO"], "stats": run.inp["stats"], "sf_q": Q["sf_row"], "sf_q_T": Q["sf_col"],
        "sf_k": K["sf_row"], "sf_k_T": K["sf_col"], "sf_v": V["sf_row"], "sf_dO": DO["sf_row"], "sf_dO_T": DO["sf_col"],
    }  # fmt: skip
    assert_golden(f"sdpa-bwd-mxfp8-sm100-d128-{cell}", inputs, got)


@pytest.mark.L0
def test_batch_invariance():
    """Under use_deterministic_algorithm each batch element of a B = 2 problem is bitwise the B = 1 problem on its data (the
    walk split is then planned per batch element, so the two plans compute the same function)."""
    hq, hkv, s = 8, 2, 512
    inp2 = _inputs(2, hq, hkv, s, seed=3)
    got2 = _deterministic_run(inp2, 2, hq, hkv, s)
    for i in range(2):
        got1 = _deterministic_run(_batch_element(inp2, i, 2), 1, hq, hkv, s)
        for a, b1 in zip(got2, got1):
            assert torch.equal(a[i : i + 1].view(torch.int16), b1.view(torch.int16))


@pytest.mark.L0
@pytest.mark.parametrize("s,causal", [(1024, False), (1000, False), (1024, True)], ids=["s1024", "s1000-tail", "s1024-causal"])
def test_batch_invariance_where_the_default_plan_differs(s, causal):
    """B = 8, 4/2 heads: without the deterministic request this batch keeps the dK/dV walk unsplit (the split decided
    over the whole batch), while B = 1 splits it. Under use_deterministic_algorithm the split is decided per batch
    element, so every element of the B = 8 problem is bitwise the B = 1 problem on its data (aligned, tail, causal)."""
    from cudnn.frost.occupancy import max_active_clusters
    from cudnn.sdpa.bwd.kernels.sm100 import _geometry_d128_mxfp8 as G

    b, hq, hkv = 8, 4, 2
    g = G.geometry(s, b, hq, hkv, 1 / math.sqrt(_D), False, causal)
    caps = lambda n: max_active_clusters(n, torch.cuda.current_device())  # noqa: E731
    if G.plan(g, caps, batch_invariant=False).split == G.plan(g, caps, batch_invariant=True).split:
        pytest.skip("on this device the default and deterministic plans of this shape coincide")
    mask = {"use_causal_mask": True} if causal else {}
    inp = _inputs(b, hq, hkv, s, seed=5, causal=causal)
    got = _deterministic_run(inp, b, hq, hkv, s, **mask)
    for i in range(b):
        got1 = _deterministic_run(_batch_element(inp, i, b), 1, hq, hkv, s, **mask)
        for name, a, b1 in zip(("dQ", "dK", "dV"), got, got1):
            assert torch.equal(a[i : i + 1].view(torch.int16), b1.view(torch.int16)), f"{name} of batch element {i}"


@pytest.mark.L0
def test_concurrent_load_and_fresh_allocations():
    """Raw-byte equal while another stream keeps the GPU busy, and into freshly allocated (poisoned) outputs and
    workspace -- a scheduling- or allocation-dependent result would differ here."""
    run = _run(hq=8, hkv=2, s=1152)
    expected = tuple(x.clone() for x in run.got)
    a = torch.randn(8192, 8192, device="cuda")
    side = torch.cuda.Stream()
    for i in range(3):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(4):
                a = a @ a.T * 1e-4  # concurrent load
        fresh = tuple(torch.full_like(x, float("nan")) for x in run.got)
        ws = torch.full_like(run.ws, 0xA5)
        pack = dict(run.pack)
        for t_out, buf in zip(run.outs[:3], fresh):
            pack[t_out] = buf
        run.g.execute(pack, ws)
        torch.cuda.synchronize()
        for got, exp in zip(fresh, expected):
            assert torch.equal(got.view(torch.int16), exp.view(torch.int16)), f"round {i}"


def _sf_slice(sf, i, b=2):
    """Batch element i of a B-element F8_128x4 scale tensor: (B, H) planes are contiguous, so a batch slice is a byte slice."""
    flat = sf.reshape(-1)
    n = flat.numel() // b
    return flat[i * n : (i + 1) * n].clone()


def _batch_element(inp, i, b):
    """The B = 1 inputs of batch element i (payloads, scales, o / dO / stats; the same values, not re-quantized)."""
    one = {}
    for key in ("Q", "K", "V", "DO"):
        one[key] = {n: (x[i : i + 1].contiguous() if not n.startswith("sf_") or n.endswith("_ref") else _sf_slice(x, i, b)) for n, x in inp[key].items()}
    for key in ("o", "dO", "stats"):
        one[key] = inp[key][i : i + 1].contiguous()
    one["scale"] = inp["scale"]
    return one


def _deterministic_run(inp, b, hq, hkv, s, **sdpa_kwargs):
    g, t, outs = _build_graph(b, hq, hkv, s, scale=inp["scale"], use_deterministic_algorithm=True, **sdpa_kwargs)
    g.create_execution_plans([cudnn.heur_mode.A])
    g.select_plan(_plan_index(g))
    g.check_support()
    g.build_plans()
    return _execute(g, t, outs, inp, b, hq, hkv, s)[0]


# --------------------------------------------------------------------------- #
# Rules 1 / 4 / 8: plan build and execute                                     #
# --------------------------------------------------------------------------- #


def _assert_matches_reference(got, inp, b, hq, hkv, s):
    for name, x, r in zip(("dQ", "dK", "dV"), got, _reference(inp, b, hq, hkv, s)):
        assert torch.isfinite(x).all(), f"{name} has non-finite values"
        cos = torch.nn.functional.cosine_similarity(x.float().flatten(), r.float().flatten(), dim=0).item()
        assert cos > _TOL_COS, f"{name}: cos={cos:.6f}"


def _fresh_build_state(monkeypatch, tmp_path):
    """A fresh compiled-object cache (no in-process memo) and occupancy table: the next build does all of its work."""
    from cudnn.frost import occupancy

    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")
    monkeypatch.setattr(occupancy, "_OCCUPANCY_MAPS", {})


def _planned_graph(b, hq, hkv, s, inp, **sdpa_kwargs):
    g, t, outs = _build_graph(b, hq, hkv, s, scale=inp["scale"], **sdpa_kwargs)
    g.create_execution_plans([cudnn.heur_mode.A])
    g.select_plan(_plan_index(g))
    g.check_support()
    return g, t, outs


@pytest.mark.L0
def test_warm_execute_compiles_allocates_and_syncs_nothing(monkeypatch):
    """Rules 1, 4 and 8 on execute: a warm execute launches the compiled chain and nothing else -- no compile, no tensor
    plumbing, no allocation, no synchronization -- and writes the same bytes."""
    import cutlass.cute as cute
    import cutlass.cute.runtime as runtime

    run = _run(hq=8, hkv=2, s=1000, amax=True, use_causal_mask=True)
    expected = tuple(x.clone() for x in run.outputs)
    torch.cuda.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]

    def forbidden(*args, **kwargs):
        raise AssertionError("execute compiled a kernel or rebuilt tensor arguments")

    with monkeypatch.context() as patch:
        patch.setattr(cute, "compile", forbidden)
        patch.setattr(runtime, "from_dlpack", forbidden)
        for name in ("view", "reshape", "permute", "contiguous", "as_strided"):
            patch.setattr(torch.Tensor, name, forbidden)
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(3):
                run.g.execute(run.pack, run.ws)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
    for got, exp in zip(run.outputs, expected):
        assert torch.equal(_bits(got), _bits(exp))


@pytest.mark.L0
@pytest.mark.parametrize("s,causal,amax", [(1024, False, False), (1000, True, True)], ids=["s1024", "s1000-causal-amax"])
def test_execute_launches_exactly_the_chain(s, causal, amax):
    """Rule 2: one execute launches the three-kernel chain (prep, dK/dV, dQ) and nothing else -- no copy, memset or
    layout kernel, including for the amax outputs -- and each kernel carries the cuDNN name prefix (Rule 6)."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    run = _run(hq=8, hkv=2, s=s, amax=amax, **({"use_causal_mask": True} if causal else {}))
    torch.cuda.synchronize()
    calls = 3
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            run.g.execute(run.pack, run.ws)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]
    assert len(names) == 3 * calls, names
    assert len(set(names)) == 3 and all(n.startswith("cudnn") for n in names), sorted(set(names))


@pytest.mark.L0
@pytest.mark.parametrize("side_stream", [False, True], ids=["default-stream", "side-stream"])
def test_execute_from_a_worker_thread(side_stream):
    """Rule 5: autograd runs backward on its own worker threads, which start with their own default stream and carry
    nothing over from the thread that built the plan. Executed on a fresh thread, behind a sleep and a poison fill on
    that thread's current stream: a launch on any other stream would run before the fill and be overwritten. Same
    bytes as on the building thread, and the worker's current device and stream are unchanged."""
    import threading

    run = _run(hq=8, hkv=2, s=1152, amax=True)
    expected = tuple(x.clone() for x in run.outputs)
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    state = {}

    def worker():
        try:
            torch.cuda.set_device(device)
            stream = torch.cuda.Stream() if side_stream else None
            with torch.cuda.stream(stream):
                before = (torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream)
                torch.cuda._sleep(50_000_000)  # tens of ms of GPU time ahead of the poison fill
                for x in run.outputs:
                    x.fill_(float("nan"))
                run.g.execute(run.pack, run.ws)
                state["after"] = (torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream)
                state["before"] = before
            torch.cuda.synchronize()
        except BaseException as e:  # surfaced on the test thread
            state["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=120)
    assert not t.is_alive(), "execute on a worker thread did not finish in 120 s"
    if "error" in state:
        raise state["error"]
    assert state["after"] == state["before"], "execute changed the worker thread's current device or stream"
    for got, exp in zip(run.outputs, expected):
        assert torch.equal(_bits(got), _bits(exp))


@pytest.mark.L0
def test_build_allocates_and_syncs_nothing(monkeypatch, tmp_path):
    """Rule 8 on plan build: the occupancy query and the chain compile (fresh cache and occupancy table) allocate no
    device memory and synchronize nothing."""
    _fresh_build_state(monkeypatch, tmp_path)
    b, hq, hkv, s = 1, 8, 2, 896
    inp = _inputs(b, hq, hkv, s, seed=13)
    g, t, outs = _planned_graph(b, hq, hkv, s, inp)
    torch.cuda.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.build_plans()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
    _assert_matches_reference(_execute(g, t, outs, inp, b, hq, hkv, s)[0], inp, b, hq, hkv, s)


@pytest.mark.L0
def test_first_build_inside_graph_capture(monkeypatch, tmp_path):
    """Rule 8: a plan's FIRST build (fresh compiled-object cache: trace, compile, export and load of all three kernels)
    may run inside torch.cuda.graph capture together with its execute; the replay matches the reference.

    The device's occupancy table is built before the capture. ``cudnn.frost.occupancy`` builds it once per process from
    a probe module that it then unloads, and ``cuModuleUnload`` invalidates a global-mode capture -- a library-wide
    limit of the shared helper, not of this row."""
    import cutlass.cute as cute
    from cudnn.frost import occupancy

    _fresh_build_state(monkeypatch, tmp_path)
    occupancy.occupancy_map()
    compiles, real_compile = [], cute.compile
    monkeypatch.setattr(cute, "compile", lambda *a, **k: compiles.append(a[0]) or real_compile(*a, **k))
    b, hq, hkv, s = 1, 8, 2, 768
    inp = _inputs(b, hq, hkv, s, seed=17, causal=True)
    g, t, outs = _planned_graph(b, hq, hkv, s, inp, use_causal_mask=True)
    from cudnn.sdpa.bwd.kernels.sm100 import _geometry_d128_mxfp8 as G

    ws = torch.empty(G.workspace_bytes(G.geometry(s, b, hq, hkv, inp["scale"], False, True)) + 16, device="cuda", dtype=torch.uint8)
    dq = torch.full((b, hq, s, _D), float("nan"), device="cuda", dtype=torch.bfloat16)
    dk = torch.full((b, hkv, s, _D), float("nan"), device="cuda", dtype=torch.bfloat16)
    dv = torch.full_like(dk, float("nan"))
    Q, K, V, DO = inp["Q"], inp["K"], inp["V"], inp["DO"]
    pack = {
        t["q"]: Q["row"], t["q_T"]: Q["col"], t["k"]: K["row"], t["k_T"]: K["col"], t["v"]: V["row"], t["o_f16"]: inp["o"],
        t["dO_f16"]: inp["dO"], t["dO"]: DO["row"], t["dO_T"]: DO["col"], t["stats"]: inp["stats"], t["sf_q"]: Q["sf_row"],
        t["sf_q_T"]: Q["sf_col"], t["sf_k"]: K["sf_row"], t["sf_k_T"]: K["sf_col"], t["sf_v"]: V["sf_row"], t["sf_dO"]: DO["sf_row"],
        t["sf_dO_T"]: DO["sf_col"], outs[0]: dq, outs[1]: dk, outs[2]: dv,
    }  # fmt: skip
    capture = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.graph(capture, stream=side):
            g.build_plans()
            assert g.get_workspace_size() <= ws.numel()
            g.execute(pack, ws)
        assert compiles, "the build inside the capture compiled nothing: the test did not exercise a first build"
        capture.replay()
        torch.cuda.synchronize()
        _assert_matches_reference((dq, dk, dv), inp, b, hq, hkv, s)
    finally:
        capture.reset()


@pytest.mark.L0
def test_replan_reloads_the_compiled_chain(monkeypatch, tmp_path):
    """Rule 4: a second plan of the same problem reloads the chain exported by the first from the compiled-object cache
    (no trace, no compile) and writes the same bytes."""
    import cutlass.cute as cute
    from cudnn.frost import compiled_cache

    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")
    first = _run(hq=8, hkv=2, s=640)
    compiled_cache.reset_stats()

    def forbidden(*args, **kwargs):
        raise AssertionError("the second plan re-traced the chain instead of reloading it")

    monkeypatch.setattr(cute, "compile", forbidden)
    second = _run(hq=8, hkv=2, s=640)
    stats = compiled_cache.stats()
    assert stats.get("hits", 0) >= 1 and stats.get("misses", 0) == 0, stats
    for a, b1 in zip(first.outputs, second.outputs):
        assert torch.equal(_bits(a), _bits(b1))


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group("gpu_exclusive")
def test_dq_shared_memory_releases_warm_stress():
    """Rule 11: the persistent dQ kernel's ld.shared-read stages -- the scheduler slots (``sched``, released on rank 0
    through ``consume_then_arrive_on_peer``) and the Q / dO / stats / delta staging (``stq`` / ``strow``, released
    through ``consume_then_arrive``) -- under warm and shared-memory-poisoned back-to-back launches of the chain.
    B = 1, 88 / 8 heads, S = 2048: 704 unit pairs on at most 74 two-CTA clusters, so every cluster refills both stages
    about nine times per launch."""
    from cudnn.frost.occupancy import max_active_clusters
    from cudnn.sdpa.bwd.kernels.sm100 import _geometry_d128_mxfp8 as G
    from warm_stress import StressCase, assert_warm_stress, stress_validation

    b, hq, hkv, s = 1, 88, 8, 2048
    run = _run(b=b, hq=hq, hkv=hkv, s=s)
    device = torch.cuda.current_device()
    geo = G.geometry(s, b, hq, hkv, 1 / math.sqrt(_D), False, False)
    p = G.plan(geo, lambda n: max_active_clusters(n, device), batch_invariant=False)
    dq, dk, dv = run.outputs
    assert_warm_stress(
        StressCase(
            "SM100 / SM103 D128 MXFP8 backward: persistent dQ scheduler slots and Q / dO staging",
            lambda: run.g.execute(run.pack, run.ws),
            {"dQ": dq, "dK": dk, "dV": dv},
            thread_blocks=2 * p.dq_clusters,
            refills_per_launch=geo.p_nunits // p.dq_clusters,
        ),
        validation=stress_validation(dq.device, cluster=2),
    )


def _bulk_inputs(b, hq, hkv, s, seed):
    """Quantized random operands with a random O and a constant LSE of log(S) + 0.5 (the log-sum-exp of unit-variance
    logits): no forward pass, finite P, enough for a bitwise B-versus-1 comparison at sizes a reference cannot reach."""
    torch.manual_seed(seed)
    q, dO = torch.randn(b, hq, s, _D, device="cuda"), torch.randn(b, hq, s, _D, device="cuda")
    k, v = torch.randn(b, hkv, s, _D, device="cuda"), torch.randn(b, hkv, s, _D, device="cuda")
    Q, K, V, DO = (_quantize(x, b, h, s, _D) for x, h in ((q, hq), (k, hkv), (v, hkv), (dO, hq)))
    o = torch.randn(b, hq, s, _D, device="cuda").to(torch.bfloat16)
    stats = torch.full((b, hq, s, 1), math.log(s) + 0.5, device="cuda")
    return dict(Q=Q, K=K, V=V, DO=DO, o=o, dO=dO.to(torch.bfloat16), stats=stats, scale=1.0 / math.sqrt(_D))


@pytest.mark.L1
def test_offsets_past_4_gib():
    """B = 34, 64/8 heads, S = 8192 puts the last batch element of every BF16 tensor (o, dO_f16, dQ: 4.6 GB each)
    past 2^32 bytes. Under use_deterministic_algorithm the last element is bitwise the B = 1 problem on its data."""
    b, hq, hkv, s = 34, 64, 8, 8192
    assert (b - 1) * hq * s * _D * 2 > 2**32  # the last element's first byte of o / dO_f16 / dQ
    need = 48 * 2**30
    free, _total = torch.cuda.mem_get_info()
    if free < need:
        pytest.skip(f"needs about {need >> 30} GiB of free device memory, {free >> 30} GiB free")
    inp = _bulk_inputs(b, hq, hkv, s, seed=23)
    got = _deterministic_run(inp, b, hq, hkv, s)
    got1 = _deterministic_run(_batch_element(inp, b - 1, b), 1, hq, hkv, s)
    for name, x, x1 in zip(("dQ", "dK", "dV"), got, got1):
        assert torch.isfinite(x1).all(), f"{name}: non-finite values"
        assert torch.equal(x[b - 1 : b].view(torch.int16), x1.view(torch.int16)), f"{name} of the last batch element"


# --------------------------------------------------------------------------- #
# REJECT                                                                      #
# --------------------------------------------------------------------------- #


def _offered(**kwargs):
    g, _t, _outs = _build_graph(**kwargs)
    g.create_execution_plans([cudnn.heur_mode.A])
    return _plan_index(g) is not None


@pytest.mark.L0
def test_declines_bshd():
    assert not _offered(b=1, hq=4, hkv=2, s=256, scale=0.088, stride_fn=_bshd_stride)


@pytest.mark.L0
def test_declines_sliding_window():
    assert not _offered(b=1, hq=4, hkv=2, s=256, scale=0.088, use_causal_mask=True, left_bound=64)


@pytest.mark.L0
def test_declines_unequal_lengths():
    assert not _offered(b=1, hq=4, hkv=2, s=256, sq=128, scale=0.088)
