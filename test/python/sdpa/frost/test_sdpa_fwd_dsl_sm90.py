# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the FROST SM90 DSL SDPA-forward engine against a torch reference.

SM90 serves one flavor -- (512, 512) f16/bf16 prefill -- so this file is the arch line's whole
forward suite. The end-to-end half runs through test_sdpa_fwd_dsl_sm100.py: its runners, fp32
references and tolerances are arch-neutral apart from a module-level ``_ARCH`` fed to
``engine_name(arch=...)``, so the ``sm100`` fixture repins that one string and SM90 runs the same
graphs and assertions at its own head dim (test/AGENTS.md). The rest is what only SM90 has -- the
row's envelope floor, the options its adapter declines, and its own template, THD arm and wrapper.
"""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

import cudnn
from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
from frost_test_utils import _SM, _is_plan_for, requires_dsl

import test_sdpa_fwd_dsl_sm100 as _sm100  # module scope: the mask table below is read at collection
from test_sdpa_fwd_dsl_sm100 import _bhsd
from test_sdpa_fwd_dsl_sm107 import _caps, _f16_facts, _gate_facts
from test_sdpa_fwd_heuristics import _facts as _heuristic_facts

# Own device gate: test_sdpa_fwd_dsl_sm100.py's ``requires_blackwell`` is a pytestmark, so it
# does not reach its runners called as functions, and widening it would put three other arch
# lines at risk. Same shape as test_sdpa_fwd_dsl_sm120.py's.
pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(_SM != 90, reason="needs an SM90 Hopper GPU, have " + ("none" if _SM is None else f"sm_{_SM}")),
    requires_dsl,
]

_D = 512
_SCALE = 1.0 / math.sqrt(_D)
_ROW = "sdpa_fwd_prefill_sm90"
_SHAPE = (1, 2, 64, _D)


@pytest.fixture
def sm100(monkeypatch):
    """The SM100 suite's runners, pinned to the SM90 row. The pin is strict: no other engine
    can satisfy these tests (delete the row and every ``select_engine`` here fails)."""
    monkeypatch.setattr(_sm100, "_ARCH", "sm90")
    return _sm100


def _api(s_kv=_SHAPE[2], dtype=torch.float16, **kw):
    """A dense SM90 adapter; the decline tests below stop at check_support."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    b, h, s_q, d = _SHAPE
    q, k, v, o = (_bhsd(b, h, s, d, dtype) for s in (s_q, s_kv, s_kv, s_q))
    return SdpaFwdDslSm90(q, k, v, o, **kw)


def _offered_id():
    """The row's public engine id, as the manifest assigns it (family base + slot)."""
    from cudnn.engines import manifest

    return next(f for f in manifest.MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()[_ROW]


def _facts(gated=False, **kw):
    """(512, 512) f16 facts for a Hopper graph, with or without the gate tail."""
    make = _gate_facts if gated else _f16_facts
    return make(**{"d_qk": _D, "d_v": _D, "device_cc": (9, 0), **kw})


# --- Graph API, mask families and the schedules -------


@torch_fork_set_rng(seed=0)
def test_dsl_sm90_graph_api(sm100):
    """The canonical smoke case, repinned: a graph with no length tensor (both seq-lens slots
    bind the cached dummy) and no Stats, so the LSE store compiles out and the workspace is 0."""
    sm100.test_sdpa_fwd_dsl_sm100_graph_api(torch.float16, True, _D)


# One case per mask family. dtype / GQA / sink rotate with the index, so each value runs under
# several masks without the 128-case cartesian (a one-off probe; its result is in the tracker).
_MASK_CASES = [pytest.param(layout, mask, i, id=f"{layout}-{mask}") for layout in ("dense", "thd") for i, mask in enumerate(_sm100._COMBO_MASKS[layout])]


@pytest.mark.parametrize("layout,mask,i", _MASK_CASES)
@torch_fork_set_rng(seed=0)
def test_dsl_sm90_mask_families(sm100, layout, mask, i):
    """Each family is its own ``config_sm90.TemplateParams`` specialization."""
    dtype = (torch.float16, torch.bfloat16)[i % 2]
    h_q, h_kv = ((8, 8), (8, 2))[i // 2 % 2]
    sink = torch.randn(1, h_q, 1, 1, dtype=torch.float32, device="cuda") if i // 4 % 2 else None
    (sm100._combo_dense if layout == "dense" else sm100._combo_thd)(_D, dtype, h_q, h_kv, _SCALE, sink, mask)


@torch_fork_set_rng(seed=0)
def test_dsl_sm90_dense_unmasked_partial_kv_tile(sm100):
    """No mask and no device lengths: the last KV tile's padding mask folds at compile time, and
    the fold's operand must stay a runtime value -- a folded float literal cannot be encoded into
    the inline PTX and raises a constraint 'n' ICE. The mask families all land on a whole tile."""
    b, h_q, h_kv, s_q, s_kv, dtype = 2, 4, 2, 70, 100, torch.float16
    q, k, v = (sm100._bhsd(b, h, s, _D, dtype) for h, s in ((h_q, s_q), (h_kv, s_kv), (h_kv, s_kv)))
    o = sm100._run_dsl_graph(q, k, v, scale=_SCALE, dtype=dtype, sdpa_kwargs={})
    torch.testing.assert_close(o, sm100._ref_sdpa_full(q, k, v, scale=_SCALE), atol=5e-2, rtol=3e-2)


@pytest.mark.parametrize("scale", [-0.25, 0.0], ids=["negative", "zero"])
@torch_fork_set_rng(seed=0)
def test_dsl_sm90_scale_sign(sm100, scale):
    """The scale's sign is compiled in: negative flips the row anchor to a minimum, zero drops the scores."""
    q, k, v = (sm100._bhsd(2, 4, 100, _D, torch.float16) for _ in range(3))
    o, lse = sm100._run_dsl_graph(q, k, v, scale=scale, dtype=torch.float16, sdpa_kwargs=dict(use_causal_mask=True), return_stats=True)
    o_ref, lse_ref = sm100._ref_sdpa_full(q, k, v, scale=scale, is_causal=True, return_stats=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(lse.squeeze(-1), lse_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("case", "args"),
    [
        # A short seq_len_q trims declared rows to O = 0 / LSE = -inf, sink or not; a keyless row with no sink is -inf too.
        pytest.param("test_dsl_sm100_bottom_right_keyless_rows", (_D, _D, False), id="q-trim-keyless-nosink"),
        pytest.param("test_dsl_sm100_bottom_right_keyless_rows", (_D, _D, True), id="q-trim-keyless-sink"),
        # Head-major (plain contiguous BHSD) storage: the kernel sorts each port's strides into its TMA order.
        pytest.param("test_dsl_sm100_singleton_seq_bhsd_storage", ("kv", _D), id="bhsd-contiguous-storage"),
        # The setup launch clamps the K/V tensor maps to the packed total; a zero-filled capacity tail cannot show it.
        pytest.param("test_dsl_sm100_thd_nan_capacity_tail", (torch.bfloat16, _D, _D), id="thd-nan-capacity-tail"),
        # Dense multi-row Stats in permuted, gapped storage: the store reads all three declared strides.
        pytest.param("_check_dsl_sm100_strided_stats", (_D, _D), id="strided-stats"),
    ],
)
@torch_fork_set_rng(seed=0)
def test_dsl_sm90_runs_the_sm100_case(sm100, case, args):
    """Whole SM100 cases whose own tables carry a D512 row, each reaching SM90 code no other case does."""
    getattr(sm100, case)(*args)


@torch_fork_set_rng(seed=0)
def test_dsl_sm90_plain_lpt_equals_the_default_schedule(sm100, monkeypatch):
    """A small causal graph always leads with LPT_L2 (plain LPT takes over above 50 MiB of K+V per
    KV head), so nothing else compiles the plain-LPT kernel. Schedulers are performance-only."""

    def select_lpt(graph, name, **_):
        index = next(i for i, p in enumerate(graph.plans) if _is_plan_for(graph.get_plan_name_at_index(i), name) and p.knobs.sched_policy == SCHED_LPT)
        graph.select_plan(index)
        return graph.plans[index]

    # S_q != S_kv keeps the diagonal top-left, the one dense mask arm the families above spell bottom-right.
    q, k, v = (sm100._bhsd(2, 4, s, _D, torch.float16) for s in (200, 300, 300))
    run = lambda: sm100._run_dsl_graph(q, k, v, scale=_SCALE, dtype=torch.float16, sdpa_kwargs=dict(use_causal_mask=True))  # noqa: E731
    o_default = run()
    torch.testing.assert_close(o_default, sm100._ref_sdpa_full(q, k, v, scale=_SCALE, is_causal=True), atol=5e-2, rtol=3e-2)
    monkeypatch.setattr(sm100, "_select_engine", select_lpt)
    assert torch.equal(run(), o_default)


# --- Attention sink at S_q == 1 (decode) -------
#
# #1095 lifted the python validator's ``sink_token`` x ``s_q == 1`` rule (pinned by
# test_sdpa_python_validate.py) and proved the SM100 row with the accept tests reused below.


@pytest.mark.parametrize("pack_gqa", [False, True], ids=["unpacked", "packed"])
@torch_fork_set_rng(seed=0)
def test_dsl_sm90_decode_sink(sm100, pack_gqa):
    """Dense, bottom-right causal, padded KV with a zero-length batch: a keyless row is O = 0 / LSE = sink."""
    sm100.test_dsl_sm100_decode_sink(_D, 1, pack_gqa)


@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
@torch_fork_set_rng(seed=0)
def test_dsl_sm90_decode_sink_thd(sm100, stats_layout):
    """A THD envelope of one: a sequence with no query, one with no keys, in either packed Stats layout (Rule S1)."""
    sm100._run_thd_stats_case(
        seq_lens_q=[1, 0, 1, 1], seq_lens_kv=[63, 64, 0, 129], d=_D, H_q=4, H_kv=1, mask="causal_br", with_sink=True, stats_layout=stats_layout
    )


@torch_fork_set_rng(seed=0)
def test_dsl_sm90_decode_sink_stats_log2(sm100):
    """Base-2 Stats convert at the one store site, so the keyless row's LSE is the sink logit in base 2 as well."""
    b, h_q, h_kv, s_kv, dtype = 3, 8, 2, 129, torch.bfloat16
    q, k, v = sm100._bhsd(b, h_q, 1, _D, dtype), sm100._bhsd(b, h_kv, s_kv, _D, dtype), sm100._bhsd(b, h_kv, s_kv, _D, dtype)
    seq_kv_lens = torch.tensor([s_kv, 0, 65], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    sink = torch.randn(1, h_q, 1, 1, dtype=torch.float32, device="cuda")
    sdpa_kwargs = dict(use_causal_mask_bottom_right=True, stats_use_log2=True)
    o, lse = sm100._run_dsl_graph(q, k, v, scale=_SCALE, dtype=dtype, sdpa_kwargs=sdpa_kwargs, seq_len_kv=seq_kv_lens, sink=sink, return_stats=True)
    o_ref, lse_ref = sm100._ref_sdpa_full(
        q, k, v, scale=_SCALE, is_causal=True, bottom_right=True, seq_kv_lens=seq_kv_lens, sinks=sink.flatten(), return_stats=True
    )
    log2_e = math.log2(math.e)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(lse.squeeze(-1), lse_ref * log2_e, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse.squeeze(-1)[1], sink.view(h_q, 1) * log2_e, atol=1e-4, rtol=0)


# --- Gemma4 full attention: GQA 32/4 at head_dim 512 -------


@pytest.mark.parametrize("scale", [_SCALE, 1.0], ids=["rsqrt_d", "unit_scale"])
@pytest.mark.parametrize("pack_gqa", [False, True], ids=["unpacked", "packed"])
@torch_fork_set_rng(seed=0)
def test_dsl_sm90_gemma4_full_attention(sm100, pack_gqa, scale):
    """Gemma4's full-attention layers are the D512 GQA workload #991 names next to DeepSeek-V4's MQA.
    Causal over a ragged last tile, O and Stats checked, V a separate tensor from K
    (``attention_k_eq_v`` shares the projection, not the storage). ``unit_scale`` is Gemma4's own:
    Q and K arrive RMS-normed, which puts the logits near +-sqrt(D) -- the sharp-softmax regime the
    exact row anchor exists for."""
    h_q, h_kv, s, dtype = 32, 4, 200, torch.bfloat16
    q, k, v = (sm100._bhsd(1, h, s, _D, dtype) for h in (h_q, h_kv, h_kv))
    assert k.data_ptr() != v.data_ptr()
    o, lse = sm100._run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True), pack_gqa=pack_gqa, return_stats=True)
    o_ref, lse_ref = sm100._ref_sdpa_full(q, k, v, scale=scale, is_causal=True, return_stats=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(lse.squeeze(-1), lse_ref, atol=2e-2, rtol=2e-2)


# --- THD launch ABI and the carved workspace -------
#
# SM90 launches on the SM120 ABI: both seq-lens slots always bind, and THD binds its metadata and
# tensor maps as ``seq_kv_lens``, with the S_q envelope and the length form as launch arguments.


@torch_fork_set_rng(seed=0)
def test_dsl_sm90_thd_cu_ragged(sm100):
    """Multi-tile ragged prefix sums; the decode cases above run the one-tile length form."""
    sm100._run_thd_stats_case(seq_lens_q=[130, 0, 64, 7], seq_lens_kv=[130, 5, 200, 7], d=_D, H_q=4, H_kv=2, mask="causal_br", cu_lens=True)


def test_sm90_thd_compile_key_is_plan_time_only():
    """Rule 4: neither the declared envelope nor the length form keys the THD artifact."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    def compiled(s, **kw):
        q, k, v, o = (_bhsd(2, 4, s, _D, torch.float16) for _ in range(4))
        api = SdpaFwdDslSm90(q, k, v, o, thd=True, **kw)
        assert api.check_support()
        api.compile()
        return api._compiled_kernel

    assert compiled(64) is compiled(200, cu_seq_q_lens=True, cu_seq_kv_lens=True)


def test_sm90_thd_workspace_is_carved_without_per_execute_allocation(monkeypatch):
    """Python Rule 1 on SM90's own arm (it never takes SM100's prepared launch): the frontend suite's
    THD case, repinned -- a real workspace size, loud undersized / absent refusals, zero warm allocations."""
    import test_sdpa_frontend_integration as frontend

    monkeypatch.setattr(frontend, "_FROST", _ROW)
    frontend.test_workspace_carve_no_per_execute_allocs_and_guards()


# --- The execute contract, on SM90's own launch arm -------


def test_sm90_execute_rejects_operands_that_contradict_the_compiled_plan():
    """SM90 never takes SM100's prepared launch, so its own ``execute`` owns the presence contract,
    and every mismatch here is otherwise silent: a substituted zero sink still contributes exp(0)
    mass to the denominator, an unbound Stats output is simply never written, and the scale's SIGN
    is compiled in (the row anchor is a maximum or a minimum), so a flipped one is a wrong answer."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    b, h, s, d = _SHAPE
    q, k, v, o = (_bhsd(b, h, s, d, torch.float16) for _ in range(4))
    lse = torch.empty(b, h, s, dtype=torch.float32, device="cuda")
    sink = torch.randn(1, h, 1, 1, dtype=torch.float32, device="cuda")
    run = lambda api, **kw: api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, **kw)  # noqa: E731

    full = SdpaFwdDslSm90(q, k, v, o, sample_lse=lse, has_sink=True)
    assert full.check_support()
    with pytest.raises(RuntimeError, match="requires a compiled plan"):
        run(full, lse_tensor=lse, sinks=sink)
    full.compile()
    for kw, why in ((dict(lse_tensor=lse), "sinks is required"), (dict(sinks=sink), "lse_tensor is required")):
        with pytest.raises(ValueError, match=why):
            run(full, **kw)
    with pytest.raises(ValueError, match="does not match its compile-time sign mode"):
        run(full, lse_tensor=lse, sinks=sink, scale_softmax=-1.0)

    plain = SdpaFwdDslSm90(q, k, v, o)
    assert plain.check_support()
    plain.compile()
    lens = torch.full((b,), s, dtype=torch.int32, device="cuda")
    for kw, why in (
        (dict(sinks=sink), "without sink support"),
        (dict(lse_tensor=lse), "without an LSE output"),
        (dict(seq_kv_lens=lens), "without per-batch KV lengths"),
    ):
        with pytest.raises(ValueError, match=why):
            run(plain, **kw)


def test_sm90_thd_execute_does_not_sync():
    """Rule 3 on the arm that carries device lengths: after the warm run, execute reads nothing back
    to the host -- the lengths become the THD metadata on device, and the grid and the S_q envelope
    are plan-time. The ``.item()`` probe first proves the mode is armed, so this is RED against a
    deliberate read."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    b, h, s, d = _SHAPE
    q, k, v, o = (_bhsd(b, h, s, d, torch.float16) for _ in range(4))
    api = SdpaFwdDslSm90(q, k, v, o, thd=True)
    assert api.check_support()
    api.compile()
    lens = torch.full((b,), s, dtype=torch.int32, device="cuda")
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    run = lambda: api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=lens, seq_kv_lens=lens, workspace=workspace)  # noqa: E731
    run()  # warm: the first launch may legitimately sync
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            torch.zeros(1, device="cuda").item()
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


# --- Routing and the public wrapper -------


@torch_fork_set_rng(seed=0)
def test_sm90_graph_loads_its_own_template(sm100, monkeypatch):
    """The suite's one routing proof: SM120 ships a prefill_d512_f16.py too, so the directory is the evidence."""
    from cudnn.sdpa.fwd import api_dsl

    loaded, load = [], api_dsl._load_kernel_template
    monkeypatch.setattr(api_dsl, "_load_kernel_template", lambda filename, *a, **kw: loaded.append(filename) or load(filename, *a, **kw))
    q, k, v = (sm100._bhsd(1, 2, 96, _D, torch.float16) for _ in range(3))
    sm100._run_dsl_graph(q, k, v, scale=_SCALE, dtype=torch.float16, sdpa_kwargs={})
    assert loaded == ["sm90/prefill_d512_f16.py"], loaded


@torch_fork_set_rng(seed=0)
def test_dsl_sm90_wrapper_matches_the_reference(sm100):
    """``sdpa_fwd_wrapper_dsl_sm90`` is public (api_index) and has no other caller under test/."""
    from cudnn.sdpa.fwd import sdpa_fwd_wrapper_dsl_sm90

    q, k, v = (sm100._bhsd(2, h, 100, _D, torch.float16) for h in (4, 2, 2))
    sinks = torch.randn(1, 4, 1, 1, dtype=torch.float32, device="cuda")
    o, lse = sdpa_fwd_wrapper_dsl_sm90(q, k, v, is_causal=True, sinks=sinks)
    o_ref, lse_ref = sm100._ref_sdpa_full(q, k, v, scale=_SCALE, is_causal=True, sinks=sinks.flatten(), return_stats=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(lse.squeeze(-1), lse_ref, atol=2e-2, rtol=2e-2)


# --- The row's head-dim envelope, floored above the template's -------


@pytest.mark.parametrize(
    ("d_qk", "d_v", "served"),
    [(512, 512, True), (264, 504, True), (512, 264, True), (256, 512, False), (192, 128, False), (8, 8, False)],
    ids=["native", "both-inside", "v-inside", "at-floor", "d192x128", "tiny"],
)
def test_sm90_row_floors_the_d_envelope_at_256(d_qk, d_v, served):
    """Two levels, deliberately different reach. The ROW serves (256, 512] on both dims
    (``d_envelope_floors``), so a small graph is declined rather than run at up to 64x
    zero-padding; the TEMPLATE still serves every multiple of 8 in (0, 512], so a direct
    adapter caller keeps the full envelope. The declines are the floor's ONLY detector:
    without them, reverting ``d_envelope_floors`` to () is silent."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.config_sm90 import head_dims_mismatch

    why = engines.mismatch(_caps(_ROW), _facts(d_qk=d_qk, d_v=d_v))
    assert (why is None) is served, why
    assert head_dims_mismatch(d_qk, d_v) is None, "the template serves every one of these"


@pytest.mark.parametrize("bad", [0, 511, 520, True, 504.0], ids=["non-positive", "off-align", "over-tile", "bool", "float"])
def test_sm90_template_d_envelope_rejects(bad):
    """``config_sm90.head_dims_mismatch``, one case per guard clause, on either dim."""
    from cudnn.sdpa.fwd.config_sm90 import head_dims_mismatch

    for pair in ((bad, _D), (_D, bad)):
        assert "multiples of 8" in (head_dims_mismatch(*pair) or ""), pair


@pytest.mark.parametrize(
    ("over", "why"),
    [
        pytest.param(dict(dtype_qkv=0), "uniform FP16 or BF16", id="dtype"),
        pytest.param(dict(pack_gqa=True, qh_per_kh=3), "ratio dividing tile_m=64", id="pack-ratio"),
        pytest.param(dict(sched_policy=SCHED_LPT_L2 + 1), "NATURAL/LPT/LPT_L2", id="sched"),
        pytest.param(dict(thd_varlen=True, seq_kv_lens_present=True, sched_policy=SCHED_LPT), "natural single-tile work decoder", id="thd-sched"),
        pytest.param(dict(scale_mode=2), "scale mode must be positive, zero, or negative", id="scale-mode"),
        pytest.param(dict(thd_varlen=True), "requires seq_kv_lens_present", id="thd-meta"),
        pytest.param(dict(thd_varlen=True, seq_kv_lens_present=True, seq_q_lens_present=True), "seq_q_lens_present is dense-only", id="thd-seq-q"),
        pytest.param(dict(causal=True, window_right=4), "window_right holds only a widened right bound", id="causal-right"),
        pytest.param(dict(window_left=-1), r"window_left must be an integer in \[0, 2\*\*30\)", id="band-range"),
        pytest.param(dict(window_right=2.0), r"window_right must be an integer in \[1, 2\*\*30\)", id="band-type"),
        pytest.param(dict(bottom_right=True), "diagonal alignment needs a band bound", id="align-bare"),
        pytest.param(dict(causal=True, bottom_right=True), "repeats the alignment causal implies", id="align-repeat"),
    ],
)
def test_sm90_template_params_are_validated(over, why):
    """``validate_params`` runs once, at template import, on whatever the adapter built, so each
    branch fires only on a ROUTING bug -- and each rejects a specialization that would otherwise
    trace: a THD kernel with no metadata operand, a band bound the Int32 token coordinates cannot
    hold, an alignment spelled two ways. They are the record of what the kernel may be shown."""
    import dataclasses

    from cudnn.sdpa.fwd import config_sm90

    with pytest.raises(ValueError, match=why):
        config_sm90.validate_params(dataclasses.replace(config_sm90.TemplateParams(), **over))


@pytest.mark.parametrize(
    "declared",
    [
        pytest.param(dict(device_cc=(8, 9)), id="sm89"),
        pytest.param(dict(device_cc=(10, 0)), id="sm100"),
        pytest.param(dict(has_paged_kv=True, page_size=128, padded=True), id="paged"),
    ],
)
def test_sm90_row_declines_what_it_never_claimed(declared):
    """sm_lo == sm_hi == 90, and paged KV is declined by the ROW as well as by the adapter: a row
    that admitted what its adapter refuses would rank a plan that cannot build, and no graph can
    reach these through the end-to-end cases. The served (9, 0) control is the gate test's."""
    from cudnn.sdpa.fwd import engines

    assert engines.mismatch(_caps(_ROW), _facts(**declared)) is not None


# --- Options the base adapter accepts and SM90 declines -------
#
# ``SdpaFwdDsl`` accepts every optional operand, so an adapter that cannot lower one declines it
# from ``check_support``. SM90 is half-only and ungated.


def test_sm90_declines_the_epilogue_gate(monkeypatch):
    """Row and adapter declare the gate the same way, and the adapter says so before any JIT."""
    from cudnn.sdpa.fwd import api_dsl, engines

    caps = _caps(_ROW)
    assert caps.epilogue_gate is False and caps.epilogue_gate_d_shapes is None
    assert engines.mismatch(caps, _facts()) is None, "the ungated control must be served"
    why = engines.mismatch(caps, _facts(gated=True))
    assert why is not None and "sigmoid(G)" in why, why

    monkeypatch.setattr(api_dsl, "_load_kernel_template", lambda *a, **k: pytest.fail("a gated plan reached the JIT"))
    with pytest.raises(NotImplementedError, match="epilogue gate"):
        _api(sample_gate=_bhsd(*_SHAPE, torch.float16)).check_support()


def test_sm90_half_plans_are_unchanged_by_has_amax_o():
    """The flag is quantized-path only: a half plan neither refuses it nor specializes on it."""
    default, off = _api(), _api(has_amax_o=False)
    assert default.check_support() and off.check_support()
    assert default.has_amax_o is True and off.has_amax_o is False
    assert default.params == off.params


def test_sm90_declines_an_amax_o_output():
    """The declared output is refused naming ONLY itself, not the FP8 inputs it used to be lumped with."""
    with pytest.raises(NotImplementedError, match=r"does not support an Amax_O output$"):
        _api(sample_amax_o=torch.empty(1, device="cuda", dtype=torch.float32)).check_support()


def test_sm90_declines_pv_bf16():
    """Hybrid PV BF16 is the pre-Rubin SM100 implementation's alone. SM90 refuses it natively:
    the shared table in test_sdpa_fwd_api_contract.py cannot cover SM90, whose uniform-dtype
    contract rejects that table's FP8 Q/K at construction, before ``check_support``."""
    with pytest.raises(NotImplementedError, match="PV BF16"):
        _api(pv_bf16=True).check_support()


@pytest.mark.parametrize(
    ("declared", "why"),
    [
        pytest.param(dict(paged_page_size=128, paged_max_seq_len_kv=256), "does not support paged KV", id="paged"),
        pytest.param(dict(pertensor_fp8=True), "does not support FP8 or PV BF16 inputs", id="fp8"),
        pytest.param(dict(dtype=torch.float32), "Q/K/V/O other than FP16/BF16", id="qkv-dtype"),
        pytest.param(dict(dtype_o=torch.bfloat16), "an output dtype different from Q", id="o-dtype"),
        pytest.param(dict(softmax_precision=cudnn.data_type.BFLOAT16), "non-FP32 softmax precision", id="softmax-precision"),
        pytest.param(dict(split_kv=2), "does not support split-KV", id="split-kv"),
        pytest.param(dict(tile_m=128), "tiles other than 64/64", id="tile"),
        pytest.param(dict(cga=2), "CGA other than 1", id="cga"),
        pytest.param(dict(sched_policy=SCHED_LPT_L2 + 1), "only NATURAL/LPT/LPT_L2", id="sched"),
        pytest.param(dict(thd=True, pack_gqa=True), "THD PackGQA is not yet supported on SM90", id="thd-pack-gqa"),
        pytest.param(dict(thd=True, thd_stats_padded=True), "THD Stats must stay packed", id="thd-padded-stats"),
        pytest.param(dict(cu_seq_q_lens=True), "cumulative lengths are THD-only", id="dense-cu-lens"),
    ],
)
def test_sm90_declines_what_it_cannot_lower(declared, why):
    """The base accepts all of these, so an unserved one must be DECLINED, never quietly dropped:
    the knob half (tiles / CGA / split / scheduler) is the engine contract -- a knob is honored or
    the engine is ineligible -- and the rest would be a wrong answer. The THD three are the
    subtlest: packed Stats into a padded buffer (Rule S1), prefix sums read as lengths, a THD grid
    packed by a ratio its work decoder does not carry."""
    with pytest.raises(NotImplementedError, match=why):
        _api(**declared).check_support()


def test_sm90_binds_only_the_layouts_it_can_address():
    """Rule 2: the declared strides ARE the kernel's tensor maps -- no BSHD gather stands between
    them -- so a layout the maps cannot express is refused, never silently copied or mis-addressed.
    The Stats store carries the same rule: packed THD Stats must be one of the two layouts Rule S1
    names, and a dense declaration must be dense_flex."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    b, h, s, d = 2, _SHAPE[1], _SHAPE[2], _D
    q, k, v, o = (_bhsd(b, h, s, d, torch.float16) for _ in range(4))
    stats = lambda stride: torch.empty(b * h * s * 2, dtype=torch.float32, device="cuda").as_strided((b, h, s), stride)  # noqa: E731
    # A padded last dim leaves the S stride off the 8-element (16-byte) TMA quantum.
    padded = torch.randn(b, h, s, d + 4, dtype=torch.float16, device="cuda")[..., :d]
    with pytest.raises(NotImplementedError, match="TMA strides must be 16-byte multiples"):
        SdpaFwdDslSm90(padded, k, v, o).check_support()
    # A broadcast batch: one storage row read by every batch, which dense_flex forbids outright.
    with pytest.raises(NotImplementedError, match="outside dense_flex"):
        SdpaFwdDslSm90(_bhsd(1, h, s, d, torch.float16).expand(b, h, s, d), k, v, o).check_support()
    for declared, stride, why in (
        (dict(thd=True), (h * s * 2, s * 2, 2), "token-major or head-major"),  # a padded sequence stride is neither
        ({}, (h * s, s, 0), "outside dense_flex"),
    ):
        with pytest.raises(NotImplementedError, match=why):
            SdpaFwdDslSm90(q, k, v, o, sample_lse=stats(stride), **declared).check_support()
    with pytest.raises(NotImplementedError, match="Stats other than FP32"):
        _api(sample_lse=torch.empty(*_SHAPE[:3], dtype=torch.float16, device="cuda")).check_support()


def test_sm90_declines_a_too_old_dsl(monkeypatch):
    """Rule 7: SM90's is the only forward adapter that raises ImportError here, naming the installed version."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
    with pytest.raises(ImportError, match=r"found 4\.6\.2"):
        _api().check_support()


@pytest.mark.parametrize("declared", [dict(causal_bottom_right=True), dict(window_size_left=16)], ids=["bottom-right", "sliding-window"])
def test_sm90_pending_s_q_above_s_kv_is_ranked_by_the_row_and_declined_at_build(declared):
    """A known row/adapter disagreement, pinned on BOTH sides: ``mismatch`` has no unpadded
    S_q > S_kv rule, so the row ranks the graph and the adapter declines it at build. This goes
    RED when the row learns the rule or the feature lands -- convert it to a one-sided assertion
    (or a positive case) then."""
    from cudnn.sdpa.fwd import engines

    facts = _facts(s_q=_SHAPE[2], s_kv=32, causal=True, bottom_right="causal_bottom_right" in declared, window_left=declared.get("window_size_left"))
    assert engines.mismatch(_caps(_ROW), facts) is None
    with pytest.raises(NotImplementedError, match="is not yet supported on SM90"):
        _api(s_kv=32, is_causal=True, **declared).check_support()


# --- Registration and the plans offered for the row -------


def test_sm90_is_an_opt_in_row_at_its_shipped_id(monkeypatch):
    """A shipped slot is fixed forever: an autotune record is ``(engine_id, knobs)`` and downstream
    caches persist the integer, so the id is pinned here rather than spelled at each use. FROST rows
    are opt-in, so the row is WITHHELD until the env flag is set -- sdpa/frost/conftest.py sets it
    for this directory, which is why every case above is offered one at all."""
    from cudnn.engines import manifest

    family = next(f for f in manifest.MANIFEST if f.name == "frost_sdpa_fwd")
    slot = family.slots[_ROW]
    assert slot.opt_in and family.engine_id + slot.slot == 20517
    assert _offered_id() == 20517 and manifest.engine_for_id(20517).name == _ROW
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    assert _ROW not in family.offered_ids(), "an opt-in row must be withheld without the flag"


def test_sm90_every_candidate_is_valid():
    """Every knob set the heuristics offer for the SM90 row is admissible on that row, deduplicated,
    bounded, and carries its fixed geometry. THD narrows to NATURAL / unpacked: the adapter declines
    THD under LPT or PackGQA, so offering either would rank a plan that cannot build."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.heuristics import _MAX_SETS_PER_ENGINE, recommend

    shape = dict(h_kv=2, s_q=1024, s_kv=8192, device_sm_count=132)
    dense, thd = _facts(causal=True, **shape), _facts(thd=True, padded=True, **shape)
    for facts, policies, packs in ((dense, {SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}, {False, True}), (thd, {SCHED_NATURAL}, {False})):
        plans = recommend("A", facts, {_ROW: _offered_id()})
        assert plans and len(plans) <= _MAX_SETS_PER_ENGINE and len({p.knobs for p in plans}) == len(plans), plans
        for knobs in (p.knobs for p in plans):
            assert engines.mismatch(_caps(_ROW), facts, knobs) is None, knobs
            assert (knobs.tile_m, knobs.tile_n, knobs.cga, knobs.split_kv) == (64, 64, 1, 1), knobs
            assert knobs.sched_policy in policies and knobs.pack_gqa in packs, knobs


def test_sm90_pack_gqa_uses_its_64_row_cta():
    """An SM90 grid tile is one 64-row CTA whatever the head dims. Through the SM100 cluster chain
    the row answered 128 -- 256 at d192x128 -- so S_q = 96 led packed. Order only: both legs stay
    listed either way."""
    from cudnn.sdpa.fwd.heuristics import _pack_gqa_tile_q, recommend

    caps = _caps(_ROW)

    def sm90(**over):
        return _heuristic_facts(**{**dict(h_q=8, h_kv=1, s_q=96, d_qk=_D, d_v=_D, device_cc=(9, 0), device_sm_count=132), **over})

    for d_qk, d_v in ((512, 512), (264, 504), (192, 128)):
        assert _pack_gqa_tile_q(caps, sm90(d_qk=d_qk, d_v=d_v), 64, 1) == 64, (d_qk, d_v)

    def leads_packed(s_q):
        plans = recommend("A", sm90(s_q=s_q), {_ROW: _offered_id()})
        assert {p.knobs.pack_gqa for p in plans} == {True, False}, plans
        return plans[0].knobs.pack_gqa

    assert [leads_packed(s) for s in (32, 64, 96, 128)] == [True, False, True, False]
