# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The Rubin (SM107) d512 f16/bf16 forward on the 2x2 DATAPATH (sm107/prefill_d512_f16_2x2.py, TemplateParams.mma_2x2).

GPU half of the sm107 2x2 coverage (the host-only structural pins -- Cfg / ledger / DESC_VERSION / SPIN_RING_WAITS /
levers / arrive sites / sm_107a SASS -- live in test_sdpa_fwd_dsl_sm107.py next to the role-split rows).  requires_rubin:
the graph API under the ``two_by_two`` fixture (api_dsl.D512_2X2 flipped for the test) on the d512 cases, the directed
cells the 2x2 atom needs (column-half-skewed S = the row-max exchange; a rescale storm; non-tile-multiple seqlens; causal
S_q = S_kv = 512 = the cluster-union bounds; SWA with empty tiles + q-trim; sink + stats; THD), the CGA_M=2 vs CGA_M=4
bitwise twin through the direct template ABI, and the multi-tile PERSISTENT detector for the pair-wide O u V alias gate
(several tiles per CTA so every tile boundary runs the twin-multicast V(t+1) against the previous O(t) store; two
back-to-back launches must be bitwise equal).  The shared cells and helpers come from the SM100 2x2 test module, which
this file imports the way the per-arch dsl suites import test_sdpa_fwd_dsl_sm100.

The pre-folded softmax scale (TemplateParams.softmax_scale_prefolded / graph.sdpa ``attn_scale_prefolded``, the kernel's
SCALE_PREFOLDED arm) runs as a second arm of the graph-API, SWA + q-trim, sink + Stats, THD, directed-numerics and
CGA twin cells ("fold" ids), plus the exact-elision oracle against the scaled chain on the direct template ABI."""

import importlib.util
import math
import os

import pytest
import torch

from test_utils import torch_fork_set_rng

from frost_test_utils import launch_f16, requires_dsl, requires_rubin

import test_sdpa_fwd_dsl_sm100 as _dsl
import test_sdpa_fwd_d512_2x2_sm100 as _t2x2

pytestmark = requires_dsl

_KERNEL_FILE = "sm107/prefill_d512_f16_2x2.py"
_TEMPLATE = _t2x2._TEMPLATE  # the template stem is the same on both arch lines ("prefill_d512_f16_2x2")
_ROLE_SPLIT_TEMPLATE = _t2x2._ROLE_SPLIT_TEMPLATE
_D = 512
_TOL = _t2x2._TOL
_run_graph = _t2x2._run_graph
_DTYPES, _DTYPE_IDS = _t2x2._DTYPES, _t2x2._DTYPE_IDS

# ------------------------------------------------------------------------------ the pre-folded softmax scale (fold arm)
# TemplateParams.softmax_scale_prefolded / graph.sdpa attn_scale_prefolded on the 2x2 kernel (SCALE_PREFOLDED: the raw row
# max and the FADD2 shift, no per-score scale).  The HOST multiplies Q by attn_scale * log2(e) (in f32, then the input
# dtype), the graph leaves attn_scale unset, and the reference runs on the Q the kernel saw with ln 2 as its logit scale.
# Every lever-bearing cell below runs both arms: "float" (the scaled chain) and "fold".
_FOLD = [False, True]
_FOLD_IDS = ["float", "fold"]
_LOG2E = math.log2(math.e)
_LN2 = math.log(2.0)  # _LN2 * _LOG2E == 1.0 exactly in double: the fold's direct-ABI launches hand the kernel scale_log2 = 1.0
_LSE_TOL = dict(atol=5e-4, rtol=0.0)  # natural-log LSE against the float64 oracle, both arms (the fold keeps the Stats domain)
_GARBAGE_SCALE = -5.0  # a runtime scale no live chain survives: a negative max flips the shift, the P tail overflows


def _fold_q(q, scale):
    """The caller's side of the pre-folded contract: Q * attn_scale * log2(e), rounded once to the input dtype."""
    return (q.float() * (scale * _LOG2E)).to(q.dtype)


def _arm(q, scale, fold, sdpa_kwargs=None):
    """(q_in, graph attn_scale, sdpa kwargs, reference scale) for one arm.  The fold pre-multiplies Q on the host, leaves
    attn_scale unset (None is dropped by the graph builder) and sets the op attribute; its reference uses ln 2 on the
    folded Q -- exactly the logits the kernel saw."""
    kw = dict(sdpa_kwargs or {})
    if not fold:
        return q, scale, kw, scale
    kw["attn_scale_prefolded"] = True
    return _fold_q(q, scale), None, kw, _LN2


def _lse64(q, k, scale, *, causal=False, swa_window=None, seq_q_lens=None, seq_kv_lens=None, sink=None):
    """float64 natural-log LSE oracle on BHSD inputs with _dsl._ref_sdpa_full's top-left mask semantics: GQA by expansion,
    causal, sliding window, per-batch q / kv lengths (rows at or past seq_len_q are -inf), the sink as one extra column."""
    b, hq, sq, _ = q.shape
    skv = k.shape[2]
    dev = q.device
    s = (q.double() @ k.double().repeat_interleave(hq // k.shape[1], 1).transpose(-1, -2)) * scale
    i = torch.arange(sq, device=dev).view(1, 1, sq, 1)
    j = torch.arange(skv, device=dev).view(1, 1, 1, skv)
    q_lens = seq_q_lens.view(b, 1, 1, 1).long() if seq_q_lens is not None else torch.full((b, 1, 1, 1), sq, device=dev)
    kv_lens = seq_kv_lens.view(b, 1, 1, 1).long() if seq_kv_lens is not None else torch.full((b, 1, 1, 1), skv, device=dev)
    dead_q = i >= q_lens
    masked = dead_q | (j >= kv_lens)
    if causal:
        masked = masked | (j > i)
    if swa_window is not None:
        masked = masked | (j < i - swa_window)
    s = s.masked_fill(masked, float("-inf"))
    if sink is not None:
        s = torch.cat([s, sink.double().view(1, hq, 1, 1).expand(b, hq, sq, 1)], dim=-1)
    return torch.logsumexp(s, dim=-1).masked_fill(dead_q.squeeze(-1), float("-inf"))


@pytest.fixture
def two_by_two(monkeypatch):
    """Flip the call-time twin so every d512 half plan built in the test lowers onto the sm107 2x2 kernel."""
    from cudnn.sdpa.fwd import api_dsl

    monkeypatch.setattr(api_dsl, "D512_2X2", True)
    yield


def _load_2x2(params, cga_m, tag):
    """Exec the sm107 template the way the loader does, with the CGA_M arm injected next to the params."""
    path = os.path.join(_t2x2._kernels_dir(), _KERNEL_FILE)
    spec = importlib.util.spec_from_file_location(f"cudnn.frost._templates.test2x2sm107_{tag}_{cga_m}", path)
    mod = importlib.util.module_from_spec(spec)
    setattr(mod, "FROST_TEMPLATE_PARAMS", params)
    setattr(mod, "FROST_D512_2X2_CGA_M", cga_m)
    # The record is part of the digest (``_t2x2._params_digest``): the compiled-plan cache keys on digest + compile args only.
    setattr(mod, "FROST_SOURCE_DIGEST", f"test2x2sm107_{tag}_{cga_m}_{_t2x2._params_digest(params)}")
    spec.loader.exec_module(mod)
    assert mod.__file__.endswith(_KERNEL_FILE) and mod.DESC_VERSION == 1 and mod.CFG.STAGES_K_SUB == 3
    return mod


# ------------------------------------------------------------------------------------------------------ GPU: graph API


@requires_rubin
@pytest.mark.L0
@pytest.mark.parametrize("fold", _FOLD, ids=_FOLD_IDS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("is_causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=0)
def test_two_by_two_graph_api(two_by_two, dtype, is_causal, fold):
    """The existing dsv4_d512 graph-API case served by the sm107 2x2 kernel (b=2 h=8 s=256 -> one 4-CTA cluster per head);
    the fold arm carries attn_scale_prefolded=True on a host-folded Q.  O within the family bound, the LSE within 5e-4 of the
    float64 oracle on both arms."""
    _dsl._require_dsl()
    b, h, s = 2, 8, 256
    scale = 1.0 / math.sqrt(_D)
    q, k, v = (_dsl._bhsd(b, h, s, _D, dtype) for _ in range(3))
    q_in, g_scale, kw, ref_scale = _arm(q, scale, fold, dict(use_causal_mask=is_causal))
    o, stats = _run_graph(q_in, k, v, scale=g_scale, dtype=dtype, sdpa_kwargs=kw, return_stats=True)
    torch.testing.assert_close(o, _dsl._ref_sdpa_full(q_in, k, v, scale=ref_scale, is_causal=is_causal), **_TOL)
    torch.testing.assert_close(stats.squeeze(-1).double(), _lse64(q_in, k, ref_scale, causal=is_causal), **_LSE_TOL)


@requires_rubin
@pytest.mark.L0
@torch_fork_set_rng(seed=1)
def test_two_by_two_role_split_untouched_when_twin_off(monkeypatch):
    """With the twin switched OFF (`api_dsl.D512_2X2 = False`; True is the default since 2026-10-06) the same graph keeps
    the Rubin role-split kernel -- the A/B control."""
    from cudnn.sdpa.fwd import api_dsl

    monkeypatch.setattr(api_dsl, "D512_2X2", False)
    _dsl._require_dsl()
    b, h, s = 1, 4, 256
    scale = 1.0 / math.sqrt(_D)
    q, k, v = (_dsl._bhsd(b, h, s, _D, torch.bfloat16) for _ in range(3))
    o = _run_graph(q, k, v, scale=scale, dtype=torch.bfloat16, sdpa_kwargs=dict(use_causal_mask=True), expect_template=_ROLE_SPLIT_TEMPLATE)
    torch.testing.assert_close(o, _dsl._ref_sdpa_full(q, k, v, scale=scale, is_causal=True), **_TOL)


@requires_rubin
@pytest.mark.L0
@pytest.mark.parametrize("mask", ["causal_br", "swa", "band", "band_br", "swa_br", "band_swa", "padded"])
@torch_fork_set_rng(seed=2)
def test_two_by_two_mask_family(two_by_two, mask):
    """The causal-family masks + KV padding on the sm107 2x2 kernel (bottom-right, sliding window, right band)."""
    _dsl._require_dsl()
    dtype = torch.bfloat16
    b, h = 2, 4
    s_q, s_kv = (128, 256) if mask in ("causal_br", "band_br", "swa_br") else (256, 256)
    scale = 1.0 / math.sqrt(_D)
    q = _dsl._bhsd(b, h, s_q, _D, dtype)
    k = _dsl._bhsd(b, h, s_kv, _D, dtype)
    v = _dsl._bhsd(b, h, s_kv, _D, dtype)
    seq_len_kv = None
    if mask == "padded":
        seq_len_kv = torch.tensor([s_kv - 76, s_kv - 16], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
        graph_kw, ref_kw = {}, dict(seq_kv_lens=seq_len_kv.flatten())
    else:
        graph_kw, ref_kw = _dsl._mask_graph_kwargs(mask), _dsl._mask_ref_kwargs(mask)
    o = _run_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=graph_kw, seq_len_kv=seq_len_kv)
    torch.testing.assert_close(o, _dsl._ref_sdpa_full(q, k, v, scale=scale, **ref_kw), **_TOL)


@requires_rubin
@pytest.mark.L0
@torch_fork_set_rng(seed=3)
def test_two_by_two_causal_512_cluster_union_bounds(two_by_two):
    """Causal S_q = S_kv = 512: one 4-CTA cluster covers the whole triangle, so the two pairs' natural causal ranges
    differ (rows 0..127 vs 128..255) -- the cluster-UNION bounds keep them on the identical KV range (anything else
    deadlocks the shared k/v_empty ring) and the per-cell mask does the trimming."""
    _dsl._require_dsl()
    dtype = torch.float16
    b, h, s = 1, 2, 512
    scale = 1.0 / math.sqrt(_D)
    q, k, v = (_dsl._bhsd(b, h, s, _D, dtype) for _ in range(3))
    o, stats = _run_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True), return_stats=True)
    o_ref, lse_ref = _dsl._ref_sdpa_full(q, k, v, scale=scale, is_causal=True, return_stats=True)
    torch.testing.assert_close(o, o_ref, **_TOL)
    torch.testing.assert_close(stats.squeeze(-1), lse_ref, **_TOL)


@requires_rubin
@pytest.mark.L0
@pytest.mark.parametrize("s_q,s_kv", [(200, 333), (65, 129), (4096 + 64, 4096 + 64)], ids=["200x333", "65x129", "4160x4160"])
@torch_fork_set_rng(seed=4)
def test_two_by_two_non_tile_multiple_seqlens(two_by_two, s_q, s_kv):
    """Seqlens that are not multiples of 64 / 128: the 64-row Q boxes zero-fill, the KV tail is padded-masked."""
    _dsl._require_dsl()
    dtype = torch.bfloat16
    b, h = 1, 2
    scale = 1.0 / math.sqrt(_D)
    q = _dsl._bhsd(b, h, s_q, _D, dtype)
    k = _dsl._bhsd(b, h, s_kv, _D, dtype)
    v = _dsl._bhsd(b, h, s_kv, _D, dtype)
    seq_len_kv = torch.full((b, 1, 1, 1), s_kv, dtype=torch.int32, device="cuda")
    o = _run_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True), seq_len_kv=seq_len_kv)
    torch.testing.assert_close(o, _dsl._ref_sdpa_full(q, k, v, scale=scale, is_causal=True, seq_kv_lens=seq_len_kv.flatten()), **_TOL)


@requires_rubin
@pytest.mark.L0
@pytest.mark.parametrize("fold", _FOLD, ids=_FOLD_IDS)
@torch_fork_set_rng(seed=5)
def test_two_by_two_swa_empty_tiles_and_q_trim(two_by_two, fold):
    """Sliding window past the padded KV tail (empty KV loops: the empty-mainloop protocol and the O u V alias phase
    bookkeeping, now pair-wide) plus the dense padded-Q trim (rows >= seq_len_q[b] -> O = 0, LSE = -inf).  Both arms: the
    keyless rows (windowed out, trimmed) must come back as the empty-row select under the fold too."""
    _dsl._require_dsl()
    dtype = torch.float16
    b, h, s_q, s_kv, W = 2, 2, 512, 512, 100
    scale = 1.0 / math.sqrt(_D)
    q, k, v = (_dsl._bhsd(b, h, s_q, _D, dtype) for _ in range(3))
    seq_len_kv = torch.tensor([160, 512], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    seq_len_q = torch.tensor([300, 450], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    q_in, g_scale, kw, ref_scale = _arm(q, scale, fold, dict(use_causal_mask=True, sliding_window_length=W + 1))
    o, stats = _run_graph(q_in, k, v, scale=g_scale, dtype=dtype, sdpa_kwargs=kw, seq_len_kv=seq_len_kv, seq_len_q=seq_len_q, return_stats=True)
    lens = dict(seq_q_lens=seq_len_q.flatten(), seq_kv_lens=seq_len_kv.flatten())
    o_ref = _dsl._ref_sdpa_full(q_in, k, v, scale=ref_scale, is_causal=True, swa_window=W, **lens)
    lse_ref = _lse64(q_in, k, ref_scale, causal=True, swa_window=W, **lens)
    torch.testing.assert_close(o, o_ref.nan_to_num(0.0), **_TOL)
    finite = torch.isfinite(lse_ref)
    torch.testing.assert_close(stats.squeeze(-1).double()[finite], lse_ref[finite], **_LSE_TOL)
    assert torch.isneginf(stats.squeeze(-1)[~finite]).all(), "trimmed / windowed-out rows must carry LSE = -inf"


@requires_rubin
@pytest.mark.L0
@pytest.mark.parametrize("fold", _FOLD, ids=_FOLD_IDS)
@pytest.mark.parametrize("stats_use_log2", [False, True], ids=["ln", "log2"])
@torch_fork_set_rng(seed=6)
def test_two_by_two_sink_and_stats(two_by_two, stats_use_log2, fold):
    """Sink + Stats (ln / log2 units) on both arms: the sink logit stays a NATURAL-domain scalar under the fold (the kernel
    folds it against final_max * ln 2, which is scale-free), and the published LSE keeps its domain."""
    _dsl._require_dsl()
    dtype = torch.bfloat16
    b, h, s = 1, 4, 384
    scale = 1.0 / math.sqrt(_D)
    q, k, v = (_dsl._bhsd(b, h, s, _D, dtype) for _ in range(3))
    sink = torch.randn(1, h, 1, 1, device="cuda", dtype=torch.float32)
    q_in, g_scale, kw, ref_scale = _arm(q, scale, fold, dict(use_causal_mask=True, stats_use_log2=stats_use_log2))
    o, stats = _run_graph(q_in, k, v, scale=g_scale, dtype=dtype, sdpa_kwargs=kw, sink=sink, return_stats=True)
    torch.testing.assert_close(o, _dsl._ref_sdpa_full(q_in, k, v, scale=ref_scale, is_causal=True, sinks=sink.flatten()), **_TOL)
    lse_nat = stats.squeeze(-1).double() / (_LOG2E if stats_use_log2 else 1.0)
    torch.testing.assert_close(lse_nat, _lse64(q_in, k, ref_scale, causal=True, sink=sink.flatten()), **_LSE_TOL)


@requires_rubin
@pytest.mark.L0
@pytest.mark.parametrize("fold", _FOLD, ids=_FOLD_IDS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@torch_fork_set_rng(seed=8)
def test_two_by_two_thd(two_by_two, dtype, fold):
    """Packed THD (two sequences of unequal length, per-sequence causal) through the persistent scheduler with 256-row
    units on the sm107 2x2 kernel, with the token-major Stats; the sentinel outside the packed region must come back
    untouched.  Both arms: the fold's packed Q is host-folded as a whole, the per-sequence references use ln 2."""
    from cudnn.sdpa.fwd.config_sm107 import SM107_F16_THD_SHAPES

    _dsl._require_dsl()
    assert (_D, _D) in SM107_F16_THD_SHAPES
    H = 4
    seq_lens = [333, 150]
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    T = cu[-1]
    scale = 1.0 / math.sqrt(_D)
    q_pk, k_pk, v_pk = (torch.randn(T, H, _D, device="cuda", dtype=dtype) for _ in range(3))
    q_in, g_scale, kw, ref_scale = _arm(q_pk, scale, fold)
    o_stor, lse_stor, _ = _dsl._run_dsl_thd_graph(
        q_in, k_pk, v_pk, cu, cu, seq_lens, seq_lens, scale=g_scale, dtype=dtype, H_q=H, H_kv=H, d=_D, mask="causal", check_stats=True, sdpa_kwargs=kw
    )
    o_pk = o_stor[: T * H * _D].view(T, H, _D)
    lse_pk = lse_stor[: T * H].view(T, H)  # token-major [t, h]
    for bi, s in enumerate(seq_lens):
        qs, ks, vs = (t[cu[bi] : cu[bi + 1]].permute(1, 0, 2).unsqueeze(0) for t in (q_in, k_pk, v_pk))
        o_ref = _dsl._ref_sdpa_full(qs, ks, vs, scale=ref_scale, is_causal=True)[0].permute(1, 0, 2)
        torch.testing.assert_close(o_pk[cu[bi] : cu[bi + 1]], o_ref, **_TOL)
        torch.testing.assert_close(lse_pk[cu[bi] : cu[bi + 1]].double(), _lse64(qs, ks, ref_scale, causal=True)[0].t(), **_LSE_TOL)
    assert (o_stor[T * H * _D :] == _dsl._THD_SENTINEL).all()


# ------------------------------------------------------------------------------------------- GPU: direct template cells


@requires_rubin
@pytest.mark.L0
@pytest.mark.parametrize("fold", _FOLD, ids=_FOLD_IDS)
@torch_fork_set_rng(seed=9)
def test_two_by_two_cga2_vs_cga4_bitwise(fold):
    """The CGA_M=2 bring-up arm (one pair, own-bit loads, own-warp O-empty gate) and the CGA_M=4 twin-multicast arm
    (pair-wide gate) run the same per-pair arithmetic: O and LSE must be BITWISE identical -- on the scaled chain and on
    the fold (the direct ABI then takes scale_log2 = 1.0 on a host-folded Q)."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    dtype = torch.float16
    B, H, KH, SQ, SKV = 1, 4, 4, 512, 1024
    scale = 1.0 / math.sqrt(_D)
    q = torch.randn(B, SQ, H, _D, device="cuda", dtype=dtype)
    k = torch.randn(B, SKV, KH, _D, device="cuda", dtype=dtype)
    v = torch.randn(B, SKV, KH, _D, device="cuda", dtype=dtype)
    params = TemplateParams(mma_2x2=True, dtype_qkv=3, dtype_o=3, window_right=0, softmax_scale_prefolded=fold)
    q_in, _, _, ref_scale = _arm(q, scale, fold)
    outs = {}
    for cga_m in (4, 2):
        mod = _load_2x2(params, cga_m, "twin_fold" if fold else "twin")
        assert mod.CFG.CGA_M == cga_m and mod.KV_SHARE == cga_m // 2 and mod.CFG.O_EMPTY_ARRIVERS == 32 * (cga_m // 2)
        assert mod.SCALE_PREFOLDED == int(fold)
        outs[cga_m] = _t2x2._direct_launch(mod, q_in, k, v, ref_scale, causal=True)
    assert torch.equal(outs[4][0], outs[2][0]) and torch.equal(outs[4][1], outs[2][1])
    o_ref, lse_ref = _t2x2._ref_bshd(q_in, k, v, ref_scale, True)
    torch.testing.assert_close(outs[4][0].float(), o_ref, **_TOL)
    torch.testing.assert_close(outs[4][1], lse_ref, **_TOL)


@requires_rubin
@pytest.mark.L0
@torch_fork_set_rng(seed=24)
def test_two_by_two_prefolded_scale_is_an_exact_elision_of_the_scaled_chain():
    """The fold arm against the scaled chain on the SAME host-folded Q, causal S_q = S_kv = 512 (the cluster-union bounds
    make CTAs 0 / 1 run a FULLY-MASKED iteration, where the raw max must be -inf exactly as the scaled chain's).  (1) The
    fold module launched with a GARBAGE scale_softmax_log2 matches the reference: both lever sites are dead under
    SCALE_PREFOLDED (a surviving multiply at the max site flips the shift, one at the shift site scales every logit --
    either blows O / LSE).  (2) Its O and LSE are BITWISE the scaled chain's at scale_log2 = 1.0 (x * 1.0 and an FFMA2 at a
    unit scale round exactly like the FADD2): the kernel-level statement of 'numerically neutral'."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    dtype = torch.float16
    B, H, S = 1, 2, 512
    scale = 1.0 / math.sqrt(_D)
    q = torch.randn(B, S, H, _D, device="cuda", dtype=dtype)
    k = torch.randn(B, S, H, _D, device="cuda", dtype=dtype)
    v = torch.randn(B, S, H, _D, device="cuda", dtype=dtype)
    q_in = _fold_q(q, scale)
    base = dict(mma_2x2=True, dtype_qkv=3, dtype_o=3, window_right=0)
    fold_mod = _load_2x2(TemplateParams(**base, softmax_scale_prefolded=True), 4, "twin_fold")
    scaled_mod = _load_2x2(TemplateParams(**base), 4, "twin")
    assert fold_mod.SCALE_PREFOLDED == 1 and scaled_mod.SCALE_PREFOLDED == 0
    o_f, lse_f = _t2x2._direct_launch(fold_mod, q_in, k, v, _GARBAGE_SCALE, causal=True)  # scale_log2 = -5 log2 e: dead under the fold
    o_s, lse_s = _t2x2._direct_launch(scaled_mod, q_in, k, v, _LN2, causal=True)  # scale_log2 = 1.0 exactly
    assert torch.isfinite(o_f).all() and torch.isfinite(lse_f).all(), "the fold read the runtime scale"
    assert torch.equal(o_f, o_s) and torch.equal(lse_f, lse_s), "the fold must be bit-for-bit the scaled chain at a unit scale"
    o_ref, lse_ref = _t2x2._ref_bshd(q_in, k, v, _LN2, True)
    torch.testing.assert_close(o_f.float(), o_ref, **_TOL)
    bhsd = lambda t: t.permute(0, 2, 1, 3)  # noqa: E731
    torch.testing.assert_close(lse_f.double(), _lse64(bhsd(q_in), bhsd(k), _LN2, causal=True), **_LSE_TOL)


@requires_rubin
@pytest.mark.L0
@torch_fork_set_rng(seed=21)
def test_two_by_two_trimmed_rows_with_nan_inputs_store_zero():
    """The SM100 twin's detector on the cc 10.7 fork (review P2: the fork had kept `o * beta`): a q-trimmed row whose Q memory
    holds NaN must come back exactly 0 with LSE = -inf (a SELECT, never NaN * 0), the live rows exact vs the reference."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    dtype = torch.bfloat16
    B, H, SQ, SKV = 2, 2, 256, 512
    scale = 1.0 / math.sqrt(_D)
    q = torch.randn(B, SQ, H, _D, device="cuda", dtype=dtype)
    k = torch.randn(B, SKV, H, _D, device="cuda", dtype=dtype)
    v = torch.randn(B, SKV, H, _D, device="cuda", dtype=dtype)
    q_lens = torch.tensor([200, 70], dtype=torch.int32, device="cuda")
    for bi in range(B):
        q[bi, int(q_lens[bi]) :] = float("nan")  # the trimmed rows' memory is poisoned
    seq_kv = torch.full((B,), SKV, dtype=torch.int32, device="cuda")
    mod = _load_2x2(TemplateParams(mma_2x2=True, dtype_qkv=2, dtype_o=2, seq_kv_lens_present=True, seq_q_lens_present=True), 4, "nantrim")
    o, lse = _t2x2._direct_launch_trim(mod, q, k, v, scale, seq_kv=seq_kv, q_lens=q_lens)
    q_ref = q.clone()
    for bi in range(B):
        q_ref[bi, int(q_lens[bi]) :] = 0.0
    o_ref, lse_ref = _t2x2._ref_trim(q_ref, k, v, scale, q_lens)
    for bi in range(B):
        ql = int(q_lens[bi])
        assert (o[bi, ql:] == 0).all(), f"batch {bi}: trimmed rows carry non-zero / NaN output (NaN count {int(torch.isnan(o[bi, ql:]).sum())})"
        assert torch.isneginf(lse[bi, :, ql:]).all()
    _t2x2._assert_rows_close(o, lse, o_ref, lse_ref, "nan-trim")


@requires_rubin
@pytest.mark.L0
@pytest.mark.parametrize("fold", _FOLD, ids=_FOLD_IDS)
@pytest.mark.parametrize("cell", ["skewed_halves", "rescale_storm"])
@torch_fork_set_rng(seed=10)
def test_two_by_two_directed_numerics(cell, fold):
    """skewed_halves: every odd 64-key half of each 128-key tile is scaled by 2^10, so lanes r and r + 64 see row maxima
    ~1000x apart -- wrong without the row-max exchange (on Rubin the dense arm's ld.red.max returns exactly that HALF-row
    max).  rescale_storm: K tile t scaled by 2^t drives alpha != 1 on every iteration (the slow correction arm and the
    per-N-block credits on every step).  Both arms (the fold exchanges and thresholds the RAW max); the logits reach ~60
    here, where the f32 reference's own accumulation noise exceeds the 5e-4 LSE bound, so these cells keep the family
    tolerance on the LSE."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    dtype = torch.bfloat16
    B, H, SQ, SKV = 1, 2, 256, 1024
    scale = 1.0 / math.sqrt(_D)
    q = torch.randn(B, SQ, H, _D, device="cuda", dtype=dtype)
    k = torch.randn(B, SKV, H, _D, device="cuda", dtype=dtype)
    v = torch.randn(B, SKV, H, _D, device="cuda", dtype=dtype)
    kf = k.float()
    if cell == "skewed_halves":
        kf = kf.view(B, SKV // 64, 64, H, _D)
        kf[:, 1::2] *= 2.0**10
        k = (kf.view(B, SKV, H, _D) / 2.0**5).to(dtype)
    else:
        kf = kf.view(B, SKV // 128, 128, H, _D)
        for t in range(SKV // 128):
            kf[:, t] *= 2.0 ** min(t, 12)
        k = (kf.view(B, SKV, H, _D) / 2.0**6).to(dtype)
    mod = _load_2x2(TemplateParams(mma_2x2=True, dtype_qkv=2, dtype_o=2, softmax_scale_prefolded=fold), 4, "directed_fold" if fold else "directed")
    q_in, _, _, ref_scale = _arm(q, scale, fold)
    o, lse = _t2x2._direct_launch(mod, q_in, k, v, ref_scale, causal=False)  # the fold hands the ABI scale_log2 = ln 2 * log2 e = 1.0
    o_ref, lse_ref = _t2x2._ref_bshd(q_in, k, v, ref_scale, False)
    assert not torch.isnan(o).any()
    torch.testing.assert_close(o.float(), o_ref, **_TOL)
    torch.testing.assert_close(lse, lse_ref, **_TOL)


# ------------------------------------------------------------------------------------- GPU: pair-wide O u V gate detector

# Several tiles per CTA (B*H*q_clusters cluster-tiles over the 53 four-CTA clusters a 212-SM Rubin co-resides), so every
# CTA runs tile boundaries where the twin's multicast V(t+1) share lands in ITS sVO while its own O(t) staging may still
# be read by the TMA store -- the race the pair-wide mb_o_empty gate (init ONE_WARP x KV_SHARE, arrive + arrive_on_peer)
# closes.  Single-tile cells cannot see it.  Causal makes consecutive tiles unequal in length (pair skew at the boundary).
_DETECTOR_SHAPES = [
    pytest.param(2, 16, 2048, False, id="dense-b2h16s2048"),
    pytest.param(2, 16, 2048, True, id="causal-b2h16s2048"),
    pytest.param(1, 8, 4096, True, id="causal-b1h8s4096"),
]


@requires_rubin
@pytest.mark.L0
@pytest.mark.parametrize("b,h,s,causal", _DETECTOR_SHAPES)
@torch_fork_set_rng(seed=11)
def test_two_by_two_multi_tile_pair_wide_o_empty_gate(two_by_two, b, h, s, causal):
    """Multi-tile persistent detector for the pair-wide O u V alias gate (fix-lane FATAL-1): every output row within the
    suite tolerance of the reference, no non-finite cell, and two back-to-back launches on identical inputs bitwise
    equal (a race shows as a launch-to-launch delta)."""
    _dsl._require_dsl()
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(_D)
    q, k, v = (_dsl._bhsd(b, h, s, _D, dtype) for _ in range(3))
    o1, stats1 = _run_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=causal), return_stats=True)
    o2, stats2 = _run_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=causal), return_stats=True)
    assert torch.isfinite(o1).all() and torch.isfinite(stats1).all()
    assert torch.equal(o1, o2) and torch.equal(stats1, stats2), "two launches on identical inputs must be bitwise equal (a tile-boundary race otherwise)"
    o_ref, lse_ref = _dsl._ref_sdpa_full(q, k, v, scale=scale, is_causal=causal, return_stats=True)
    torch.testing.assert_close(o1, o_ref, **_TOL)
    torch.testing.assert_close(stats1.squeeze(-1), lse_ref, **_TOL)
