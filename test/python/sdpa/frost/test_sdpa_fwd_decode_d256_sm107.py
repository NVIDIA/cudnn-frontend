# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The d256 DECODE tile of the FROST Rubin (SM107) f16/bf16 engine (sm107/decode_d256_f16.py).

The Rubin sibling of the SM100 swap-AB tile: the same body (KV tokens on the MMA M
axis, the packed Q rows on N, one cta_group::1 CTA per unit) under
``config_sm107.make_cfg_d256_decode``, routed on cc 10.7 by the same record field
(``TemplateParams.decode_q_tile``) for f16/bf16 d256 graphs whose S_q x packed heads
fit 16 rows (``config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS``).  Same graph contract,
same engine row (``sdpa_fwd_prefill_sm107``): every graph test pins the engine with
``select_engine`` and asserts WHICH template served the plan through the executor's
``kernel_template`` (a decline or a fallback to the prefill tile fails instead of
passing on the wrong kernel).  The reference is an fp32 torch softmax over the dense
K/V the pools were built from (shared with the SM100 suite).

Coverage, the SM100 case list re-run on the Rubin tile: paged (page 16/32/64/128,
NHD/HND) and dense padded caches, mixed lengths incl. 0 and 1, the head groups 8:1 /
16:1 / 32:2 / 16:4 / MHA and the 24/2 geometry (G = 12), MTP bottom-right causal at S_q
2 (8:1) and 4 (4:1) with per-batch Q lengths, sliding window, right band, sink (dense
and paged, keyless rows), Stats natural and base-2, fp16 and bf16, CUDA-graph replay
under ``set_sync_debug_mode("error")`` (Rule 3), the 32-column tile (compiled and driven
at the template level, NOT routed), and the routing boundary (larger S_q x G, THD
queries, d128).

The row claims the SM100 tile's PackGQA and split-KV on this tile (``pack_gqa_d_shapes``
carries (256, 256) for exactly this route; the SM107 D256 split rule exempts it), so the
SM100 suite's policy assertions apply here verbatim: whole-group packing (24/2: 12 live
rows + 4 zero tail rows per (batch, KV head) unit), the decode split model
(``choose_decode_tile_split_kv``: the eager-safe lead, the captured runner-up, swapped under
``is_cuda_graph_replay_expected``), plus the three equalities the claim rests on -- paged ==
dense BITWISE on the same tokens, packed == unpacked BITWISE, and split == unsplit within
the combine's fp32 reassociation (one output ulp; a derived budget, not a tuned one).  What
does NOT invert: the 32-row shapes (32/2 at S_q = 2) ride the tile UNPACKED -- the Rubin
d256 prefill kernel wires no PackGQA, so a packed route past the tile does not exist and a
packed request there is a typed decline (the REJECT cases) -- and the tile's packing / split
partials are still driven at the TEMPLATE level too (the adapter's module loader +
``launch_f16``), the kernel-level evidence under the graph-level claim.

Structural pins (host-runnable, no Rubin GPU): the module is the SM100 body modulo the
declared Rubin deltas (the twin-diff snapshot), ``DESC_VERSION`` derived from the layout
and wired into every ``SmemTile``, ``SPIN_RING_WAITS`` on exactly the per-KV-iteration
ring waits, the config twin's backstops, and the CuTe DSL version gate (Rule 7).
"""

import math
import re
from pathlib import Path

import pytest

import torch

from cudnn.frost.compiled_cache import positional_entry
from cudnn.sdpa.fwd.heuristics import choose_decode_tile_split_kv

from frost_test_utils import _is_plan_for, launch_f16, offers_engine, requires_dsl, requires_rubin, select_engine
from test_sdpa_fwd_decode_d256_sm100 import _pools, _ref, _served_by

pytestmark = [pytest.mark.L0, requires_dsl]

D = 256
DECODE = "decode_d256_f16"
PREFILL = "prefill_d256_f16"
ARCH = "sm107"


def _dsl_floor_reason():
    """Rule 7: the Rubin tile needs a CuTe DSL with the sm_107a target (the public 4.8.0
    wheel); below that floor every test here SKIPS (never fails) with the typed message."""
    try:
        from cudnn.frost.buffers import cutedsl_arch_requirement_error

        return cutedsl_arch_requirement_error((10, 7))
    except Exception as e:  # pragma: no cover - a DSL that cannot be probed reads as below the floor
        return f"cannot probe the CuTe DSL for the sm_107a target: {e}"


_DSL_FLOOR_WHY = _dsl_floor_reason()
requires_sm107_dsl = pytest.mark.skipif(_DSL_FLOOR_WHY is not None, reason=_DSL_FLOOR_WHY or "")


def _gpu(f):
    """A test that runs the tile: a Rubin-line GPU and a DSL with its target."""
    return requires_sm107_dsl(requires_rubin(f))


def _engine():
    from cudnn.sdpa.fwd.engines import engine_name

    return engine_name(arch=ARCH)


def _load(rubin=True, **params):
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    return _load_sm100_kernel_module((D, D), TemplateParams(**params), rubin=rubin)


def _code_lines(src):
    """Source with whole-line comments dropped -- the module's prose quotes the
    spellings these pins count."""
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def _wait_sites(code):
    """Every mbarrier wait call site of the kernel source: (target, balanced argument text)."""
    out = []
    for m in re.finditer(r"^\s*(?:bars\.)?(mb_\w+)(?:\[[^\]]*\])*\.wait\(", code, re.M):
        target = m.group(1)
        i, depth = m.end(), 1
        while depth:
            depth += (code[i] == "(") - (code[i] == ")")
            i += 1
        out.append((target, code[m.end() : i - 1]))
    return out


# --- structural pins: the port is the SM100 body plus the declared Rubin deltas -------


def test_sm107_decode_routes_to_the_rubin_sibling():
    """The decode record routes to sm107/decode_d256_f16.py on cc 10.7 (ahead of the shared
    paged-prefill arm: the paged record too) and to the SM100 file otherwise."""
    for extra in ({}, dict(paged_kv=True, page_size=16)):
        mod = _load(rubin=True, decode_q_tile=16, seq_kv_lens_present=True, **extra)
        assert mod.__file__.endswith("/sm107/decode_d256_f16.py") and "sm107_decode" in mod.__name__, mod.__name__
    sm100 = _load(rubin=False, decode_q_tile=16, seq_kv_lens_present=True)
    assert sm100.__file__.endswith("/sm100/decode_d256_f16.py") and "sm100_decode" in sm100.__name__, sm100.__name__


def test_sm107_decode_body_matches_the_sm100_body():
    """The SNAPSHOT pin: the Rubin module is the SM100 body line for line, modulo the
    declared deltas -- the config import, the DSL gate + the two module constants, the
    ``desc_version=DESC_VERSION`` keyword on every SmemTile, the ``spin=SPIN_RING_WAITS``
    keyword on the ring waits and the header prose.  A fix that lands in one file and
    not the other fails here; so does a body change smuggled into the port."""
    from cudnn.sdpa.fwd import api_dsl

    kdir = Path(api_dsl.__file__).parent / "kernels"
    sm100 = (kdir / "sm100" / "decode_d256_f16.py").read_text(encoding="utf-8")
    sm107 = (kdir / "sm107" / "decode_d256_f16.py").read_text(encoding="utf-8")

    def _body(src):
        # everything after the module docstring, comments dropped
        end = src.index('"""', src.index('"""') + 3) + 3
        return [ln for ln in _code_lines(src[end:]).splitlines() if ln.strip()]

    a, b = _body(sm100), _body(sm107)
    b = [ln for ln in b if ln.strip() not in ("desc_version=DESC_VERSION,",)]
    b = [ln.replace(", spin=SPIN_RING_WAITS)", ")") for ln in b]
    b = [
        ln
        for ln in b
        if not ln.startswith(
            (
                "DESC_VERSION: int = ",
                "SPIN_RING_WAITS: bool = ",
                "_DSL_ARCH_ERROR = ",
                "if _DSL_ARCH_ERROR is not None:",
                '    raise NotImplementedError(f"decode_d256_f16 (sm107)',
            )
        )
    ]
    b = [ln for ln in b if ln != "from cudnn.frost.buffers import cutedsl_arch_requirement_error"]
    b = [
        ln.replace(
            "from cudnn.sdpa.fwd.config_sm107 import TemplateParams, d256_decode_desc_version, make_cfg_d256_decode",
            "from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d256_decode",
        )
        for ln in b
    ]
    assert a == b, "the Rubin decode module diverged from the SM100 body beyond the declared deltas:\n" + "\n".join(
        f"{'-' if x else '+'} {x or y}" for x, y in zip(a, b) if x != y
    )


@pytest.mark.parametrize("n_q", [16, 32])
def test_sm107_decode_descriptor_version_matches_the_smem_budget(n_q):
    """DESC_VERSION is derived from the layout (config_sm107.d256_decode_desc_version) and
    is 0 at both tile widths: the highest MMA-operand slab (P^T) starts at 200 / 208 KiB,
    under the 256 KiB version-0 window, and the whole tile (209 / 226 KiB) fits the
    STANDARD 227 KiB carveout (no oversized-SMEM launch, no L1 shrink).  A record that
    moved an operand past the line must flip the module constant, never a tile literal."""
    from cudnn.sdpa.fwd.config_sm107 import d256_decode_desc_version, d256_decode_smem_layout

    mod = _load(decode_q_tile=n_q, seq_kv_lens_present=True)
    lay = d256_decode_smem_layout(mod.CFG)
    assert mod.DESC_VERSION == d256_decode_desc_version(mod.CFG) == 0, (mod.DESC_VERSION, lay)
    assert lay["p_start"] == (200 if n_q == 16 else 208) * 1024 and lay["p_start"] < 256 * 1024, lay
    assert lay["fits_standard_carveout"] and lay["total"] <= 227 * 1024, lay
    assert (mod.CFG.N_Q, mod.CFG.SOFTMAX_WARPS, mod.CFG.TOTAL_WARPS, mod.CFG.STAGES_KV, mod.CFG.TILE_K_HW) == (n_q, n_q // 4, n_q // 4 + 2, 3, 16)


def test_sm107_decode_every_smem_tile_takes_the_module_desc_version():
    """Every SmemTile of the module is wired to DESC_VERSION (4 tiles: Q^T, the K/V ring as
    K and as V^T, P^T); a re-literalled desc_version bypasses the constant."""
    mod = _load(decode_q_tile=16, seq_kv_lens_present=True)
    code = _code_lines(Path(mod.__file__).read_text(encoding="utf-8"))
    n_tiles = len(re.findall(r"\bSmemTile\(", code))
    assert n_tiles == 4 == code.count("desc_version=DESC_VERSION"), (n_tiles, code.count("desc_version=DESC_VERSION"))
    assert not re.search(r"desc_version=[01]\b", code), "a re-literalled desc_version bypasses DESC_VERSION"


def test_sm107_decode_ring_waits_take_the_module_spin_constant():
    """SPIN_RING_WAITS holds the measured per-kernel value (False: the sleeping form the SM100
    body runs, until this tile's own A/B), is defined exactly once, and every per-KV-iteration
    RING wait -- the TMA warp's three kv_empty waits, the MMA warp's kv_full x3 / s_empty x2 /
    p_full, the softmax warps' s_full and in-loop bmm2_done: 11 sites -- passes it, while the
    three one-shot waits (q_full, the epilogue's last bmm2_done, tmem_dealloc) keep the default."""
    mod = _load(decode_q_tile=16, seq_kv_lens_present=True)
    assert mod.SPIN_RING_WAITS is False
    code = _code_lines(Path(mod.__file__).read_text(encoding="utf-8"))
    assert len(re.findall(r"^SPIN_RING_WAITS: bool = (?:True|False)$", code, re.M)) == 1
    assert not re.search(r"spin=(?:True|False)\b", code), "a spin= literal at a call site bypasses SPIN_RING_WAITS"
    sites = _wait_sites(code)
    spun = sorted(t for t, args in sites if "spin=SPIN_RING_WAITS" in args)
    idle = sorted(t for t, args in sites if "spin=SPIN_RING_WAITS" not in args)
    assert spun == sorted(["mb_kv_empty"] * 3 + ["mb_kv_full"] * 3 + ["mb_s_empty"] * 2 + ["mb_p_full", "mb_s_full", "mb_bmm2_done"]), spun
    assert idle == ["mb_bmm2_done", "mb_q_full", "mb_tmem_dealloc"], idle
    assert code.count("spin=") == 11, "a spin= outside a .wait( call"


def test_sm107_decode_config_twin_backstops():
    """config_sm107.make_cfg_d256_decode: the SM100 record's predicates plus the Rubin ones --
    the gate / pre-folded scale / other-template records are refused by name, the Rubin prefill
    factory refuses a decode record (the loader routes on it first), and both tile widths build."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.config_sm107 import d256_decode_desc_version, make_cfg_d256, make_cfg_d256_decode

    base = dict(decode_q_tile=16, seq_kv_lens_present=True)
    for bad, pattern in (
        (dict(epilogue_gate=True), "epilogue gate is not wired on the decode tile"),
        (dict(softmax_scale_prefolded=True), "softmax_scale_prefolded is not wired on the decode tile"),
        (dict(mma_2x2=True), "select other templates"),
        (dict(decode_tile=True), "select other templates"),
        (dict(thd_varlen=True), "THD"),
        (dict(pack_gqa=True, qh_per_kh=32), "fit the Q tile"),
        (dict(paged_kv=True, page_size=48), "page_size must be a positive multiple of 8"),
    ):
        with pytest.raises(ValueError, match=pattern):
            make_cfg_d256_decode(TemplateParams(**{**base, **bad}))
    with pytest.raises(ValueError, match="decode_q_tile"):
        make_cfg_d256(TemplateParams(**base))
    with pytest.raises(ValueError, match="decode_q_tile"):
        make_cfg_d256_decode(TemplateParams())
    cfg, _ = make_cfg_d256_decode(TemplateParams(**base))
    assert (cfg.N_Q, cfg.TILE_N, cfg.STAGES_KV, cfg.TILE_K_HW, d256_decode_desc_version(cfg)) == (16, 128, 3, 16, 0)
    wide, _ = make_cfg_d256_decode(TemplateParams(decode_q_tile=32, seq_kv_lens_present=True, dtype_qkv=2))
    assert (wide.N_Q, wide.SOFTMAX_WARPS, wide.TOTAL_WARPS, wide.P_SWZ_BYTES, d256_decode_desc_version(wide)) == (32, 8, 10, 64, 0)


def test_sm107_decode_module_declines_below_the_dsl_floor(monkeypatch):
    """Rule 7 probe: a DSL without the sm_107a target makes the module raise the TYPED decline
    (NotImplementedError naming the installed version) before any kernel body is defined.  A
    params record no other test loads, so the loader cannot serve a cached module."""
    from cudnn.frost import buffers
    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    want = buffers.cutedsl_arch_requirement_error((10, 7))
    assert want is not None and "sm_107a" in want
    path = str(Path(api_dsl.__file__).parent / "kernels" / "sm107" / "decode_d256_f16.py")
    with pytest.raises(NotImplementedError, match=re.escape(want)):
        load_template(path, TemplateParams(decode_q_tile=16, seq_kv_lens_present=True, stats_log2=True, window_left=77, window_right=5), tag="dsl_floor_probe")


def test_decode_q_tile_rule():
    """The routing rule shared with SM100: 16 rows ride the decode tile; the 32-column tile is a
    valid config (D256_DECODE_MAX_Q_ROWS) the adapter does not select."""
    from cudnn.sdpa.fwd.config_sm100 import D256_DECODE_MAX_Q_ROWS, D256_DECODE_ROUTED_MAX_Q_ROWS, decode_d256_q_tile

    assert (D256_DECODE_ROUTED_MAX_Q_ROWS, D256_DECODE_MAX_Q_ROWS) == (16, 32)
    assert decode_d256_q_tile(1, 12) == 16  # the 24/2 geometry: a group that does not divide the tile still fits it
    assert decode_d256_q_tile(1, 16) == decode_d256_q_tile(2, 8) == decode_d256_q_tile(16, 1) == 16
    for s_q, g in ((2, 16), (1, 32), (17, 1), (32, 1), (3, 16), (33, 1), (1, 64), (0, 16)):
        assert decode_d256_q_tile(s_q, g) == 0, (s_q, g)


def test_sm107_row_and_predicate_agree_on_the_decode_tile():
    """The facts-level twin (engines.d256_decode_tile_selected) of the adapter's _decode_q_tile on the
    Rubin row: a decode-shaped half d256 graph is selected (paged or dense, sink or not); a gated,
    pre-folded, THD, quantized or wider graph is not -- and the row's paged rule admits exactly the
    selected form while the paged PREFILL pipeline keeps its THD-only scope."""
    import cudnn
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    caps = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == _engine())
    assert caps.decode is True
    base = dict(
        b=4, h_q=24, h_kv=2, s_q=1, s_kv=4096, d_qk=D, d_v=D, dtype=cudnn.data_type.BFLOAT16, padded=True, has_paged_kv=True, page_size=16, device_cc=(10, 7)
    )
    f = SdpaGraphFacts(**base)
    assert engines.d256_decode_tile_selected(caps, f, 1) and engines.d256_decode_tile_selected(caps, f, 12)
    assert engines.mismatch(caps, f, None) is None
    assert engines.mismatch(caps, SdpaGraphFacts(**{**base, "has_sink": True}), None) is None
    for change in (dict(s_q=17), dict(thd=True), dict(has_epilogue_gate=True), dict(attn_scale_prefolded=True), dict(d_qk=128, d_v=128)):
        g = SdpaGraphFacts(**{**base, **change})
        assert not engines.d256_decode_tile_selected(caps, g, 1), change
    for change in (dict(s_q=64), dict(d_qk=128, d_v=128)):
        g = SdpaGraphFacts(**{**base, **change})
        reason = engines.mismatch(caps, g, None)
        assert reason and "Rubin paged KV requires THD" in reason, (change, reason)
    assert not engines.d256_decode_tile_selected(caps, SdpaGraphFacts(**{**base, "s_q": 2}), 16)  # 32 packed rows: the unrouted tile


# --- the graph path on a Rubin GPU -------------------------------------------------------------


def _run_graph(
    *,
    B,
    H,
    KH,
    s_q,
    lens,
    d=D,
    page=16,
    hnd=True,
    dtype=torch.float16,
    stats=True,
    stats_log2=False,
    causal_br=False,
    window_left=None,
    window_right=None,
    sink=False,
    q_lens=None,
    expect=DECODE,
    seed=0,
    graph_kwargs=None,
    split_kv=None,
    pack_gqa=None,
    padded=True,
    capture=None,
):
    """Build cuDNN's paged (``page`` > 0) or dense SDPA graph -- padded (per-batch
    ``seq_len_kv`` / ``seq_len_q``, the default) or, with ``padded=False`` on a dense cache
    of uniform tile-aligned ``lens``, UNPADDED (no padding mask at all: the one dense form
    the shared split rules admit) -- pin the Rubin FROST engine (its first entry: the
    heuristics' own choice, or with ``split_kv`` / ``pack_gqa`` its ranked entry carrying
    those knobs, a runner-up when the heuristics lead elsewhere), assert the serving
    template, execute under the D2H detector and compare O / Stats against the SM100
    suite's fp32 reference.  Returns the pinned plan (its knobs are what the row-contract
    tests read); ``capture`` (a dict) additionally receives the fp32 ``o`` ([B, S_q, H, d]),
    the ``lse`` ([B, H, S_q], or None) and the kernel ``module`` the executor loaded, for
    the bitwise comparisons."""
    import cudnn
    import cudnn.sdpa  # noqa: F401

    torch.manual_seed(seed)
    dev = "cuda"
    S = max(max(lens), 1)
    assert padded or (not page and set(lens) == {S} and S % 128 == 0 and q_lens is None), "the unpadded form: a dense cache of uniform tile-aligned lengths"
    scale = 1.0 / math.sqrt(d)
    q_gpu = torch.randn(B, s_q, H, d, device=dev, dtype=dtype).transpose(1, 2)  # BHSD strides over a BSHD buffer
    k_dense = torch.randn(B, S, KH, d, device=dev, dtype=dtype)
    v_dense = torch.randn(B, S, KH, d, device=dev, dtype=dtype)
    o_gpu = torch.empty(B, s_q, H, d, device=dev, dtype=dtype).transpose(1, 2)
    seq_kv = torch.tensor(lens, dtype=torch.int32, device=dev)
    seq_q = torch.tensor(q_lens if q_lens is not None else [s_q] * B, dtype=torch.int32, device=dev)
    # ``sink``: False = no sink, True = random logits, a number = that logit on every head.
    sinks = None if sink is False else (torch.randn(H, device=dev) * 2.0 if sink is True else torch.full((H,), float(sink), device=dev))

    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, **(graph_kwargs or {}))
    q = g.tensor_like(q_gpu)
    kw = dict(name="sdpa", generate_stats=stats, attn_scale=scale, use_padding_mask=padded, stats_use_log2=stats_log2)
    if page:
        k_c, v_c, bt = _pools(k_dense, v_dense, page, hnd, seed)
        k, v = g.tensor_like(k_c), g.tensor_like(v_c)
        tk, tv = g.tensor_like(bt), g.tensor_like(bt)
        kw.update(paged_attention_k_table=tk, paged_attention_v_table=tv, paged_attention_max_seq_len_kv=S)
    else:
        k_c, v_c = k_dense.transpose(1, 2), v_dense.transpose(1, 2)  # BHSD view of the BSHD buffers
        k, v = g.tensor_like(k_c), g.tensor_like(v_c)
    slq, slk = seq_q.view(B, 1, 1, 1), seq_kv.view(B, 1, 1, 1)
    kw.update(q=q, k=k, v=v)
    if padded:
        sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
        kw.update(seq_len_q=sq_t, seq_len_kv=sk_t)
    if causal_br:
        kw["use_causal_mask_bottom_right"] = True
    elif window_right is not None:
        kw["use_causal_mask"] = window_right == 0
        if window_right > 0:
            kw["diagonal_band_right_bound"] = window_right
    if window_left is not None:
        kw["sliding_window_length"] = window_left + 1
    sink_t = None
    if sinks is not None:
        sink_gpu = sinks.view(1, H, 1, 1).contiguous()
        sink_t = g.tensor_like(sink_gpu)
        kw["sink_token"] = sink_t
    o, st = g.sdpa(**kw)
    o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())
    stats_gpu = None
    if stats:
        stats_gpu = torch.empty(B, H, s_q, 1, device=dev, dtype=torch.float32)
        st.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    plan = select_engine(g, _engine(), pack_gqa=pack_gqa, split_kv=split_kv)
    idx = g._plan_index
    g.check_support()
    g.build_plans()
    assert _served_by(g, idx) == expect, f"plan {g.get_plan_name_at_index(idx)} served by {_served_by(g, idx)}, expected {expect}"
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    vp = {q: q_gpu, k: k_c, v: v_c, o: o_gpu}
    if padded:
        vp.update({sq_t: slq, sk_t: slk})
    if page:
        vp.update({tk: bt, tv: bt})
    if stats:
        vp[st] = stats_gpu
    if sinks is not None:
        vp[sink_t] = sink_gpu
    # Rule 3: the execute path reads the per-batch lengths and page tables on
    # device; any blocking D2H here is a bug, not a slow path.
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute(vp, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()

    ref_o, ref_lse = _ref(
        q_gpu.transpose(1, 2), k_dense, v_dense, lens, q_lens, scale, causal_br=causal_br, window_left=window_left, window_right=window_right, sinks=sinks
    )
    out = o_gpu.transpose(1, 2).float()
    assert not torch.isnan(out).any(), "NaN in O"
    torch.testing.assert_close(out, ref_o, atol=2e-2 if dtype == torch.float16 else 5e-2, rtol=0)
    dead = torch.isinf(ref_lse)
    if dead.any():
        assert out.transpose(1, 2)[dead].abs().max().item() == 0.0, "dead rows must write O := 0"
    if stats:
        got_lse = stats_gpu.view(B, H, s_q)
        exp_lse = ref_lse * (1.4426950408889634 if stats_log2 else 1.0)
        torch.testing.assert_close(got_lse[~dead], exp_lse[~dead], atol=5e-3, rtol=0)
        if dead.any():
            assert torch.isinf(got_lse[dead]).all() and (got_lse[dead] < 0).all(), "dead rows must write LSE := -inf"
    if capture is not None:
        capture.update(o=out.clone(), lse=stats_gpu.view(B, H, s_q).clone() if stats else None, module=g._compiled_plans[idx]._compiled.kernel_module)
    return plan


def _sm_count():
    return torch.cuda.get_device_properties(0).multi_processor_count


def _assert_decode_tile_plan(plan, *, G, units=None, kv_tiles=None, replay=False):
    """The SM100 suite's policy on the Rubin row, now that it claims PackGQA + split-KV on the tile
    (the inversion of the former unpacked / unsplit pin): the plan is PACKED exactly when there is a
    group to pack (``G > 1``; MHA stays the bit-exact unpacked fold) and, for a shape the decode split
    model governs (``units`` x ``kv_tiles`` given: an unmasked, sink-free paged or unpadded cache), its
    split is that model's eager-safe choice -- the captured optimum under ``replay`` -- at THIS part's
    SM count.  Shapes the shared no-split rules bind (a padded dense cache, a sink, per-batch Q lengths)
    pass no model inputs and are pinned unsplit by their caller."""
    assert (plan.knobs.pack_gqa is True) == (G > 1), (G, plan.knobs)
    if units is not None:
        want = choose_decode_tile_split_kv(units=units, kv_tiles=kv_tiles, sm_count=_sm_count(), **({"launch_cost": 0.0} if replay else {}))
        assert (plan.knobs.split_kv or 1) == want, (plan.knobs, want)


def _assert_within_one_output_ulp(a, b, dtype, what):
    """``|a - b| <= one ulp of dtype`` at the larger magnitude, per element (fp64 on the CPU): the
    budget of two fp32 reductions that differ only in their association -- the split path
    renormalises each partial by ``exp(lse_s - lse)`` in the combine where the unsplit path rescales
    the running accumulator per tile, so the two fp32 sums round to the output dtype identically or
    one ulp apart.  A derived budget, not a tuned one; the measured magnitude is printed so the log
    carries it."""
    mbits = {torch.bfloat16: 7, torch.float16: 10}[dtype]
    a64, b64 = a.double().cpu(), b.double().cpu()
    mag = torch.maximum(a64.abs(), b64.abs())
    _, exp = torch.frexp(mag)  # mag = m * 2**exp, m in [0.5, 1): the dtype's ulp in that binade is 2**(exp - 1 - mbits)
    ulp = torch.ldexp(torch.ones_like(mag), exp - 1 - mbits)
    diff = (a64 - b64).abs()
    worst = (diff / ulp).max().item()
    flipped = (diff > 0).double().mean().item()
    print(f"{what}: max |diff| = {diff.max().item():.3e} = {worst:.2f} output ulp; {flipped:.4%} of the elements differ")
    assert worst <= 1.0, (what, worst, flipped)


# --- accept: the decode tile serves the contract on Rubin --------------------------------------


@_gpu
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize("page", [16, 32, 64, 128])
def test_decode_graph_paged_page_sizes(page, hnd):
    """32/2-head decode (PackGQA 16:1 on the tile) over pages of every admitted size: mixed lengths
    incl. 0 and 1, a tile-unaligned tail, a length ending on a page/tile boundary; Stats out; the
    split is the decode model's at 10 units x 8 tiles."""
    _assert_decode_tile_plan(_run_graph(B=5, H=32, KH=2, s_q=1, lens=[300, 77, 0, 1, 1024], page=page, hnd=hnd), G=16, units=5 * 2, kv_tiles=8)


@_gpu
@pytest.mark.parametrize(("H", "KH"), [(8, 1), (16, 1), (32, 2), (16, 4), (4, 4), (24, 2)], ids=["8to1", "16to1", "32to2", "16to4", "mha", "24to2"])
def test_decode_graph_head_groups(H, KH):
    """PackGQA 8:1 and 16:1 (one and two tokens per 16-row tile at S_q = 1), 32:2, a 4-wide group,
    MHA (one row per unit, unpacked) and the 24/2 geometry (G = 12: 12 live rows + 4 zero tail rows
    per unit -- the whole-group packing no prefill tile has) -- every one decode-shaped on the Rubin
    row, packed exactly when there is a group, split by the decode model (3 x KH units x 8 tiles)."""
    G = H // KH
    _assert_decode_tile_plan(_run_graph(B=3, H=H, KH=KH, s_q=1, lens=[1000, 129, 640], page=32, dtype=torch.bfloat16), G=G, units=3 * (H // G), kv_tiles=8)


@_gpu
@pytest.mark.parametrize(("s_q", "H"), [(2, 16), (4, 8)], ids=["sq2_8to1", "sq4_4to1"])
def test_decode_graph_mtp_bottom_right(s_q, H):
    """MTP: S_q in {2, 4} bottom-right causal over a paged cache, packed 8:1 and 4:1 (16 rows either
    way: two and four tokens per 16-row tile), with per-batch Q lengths below S_q (dense padded-Q
    trim: O := 0 / LSE := -inf past them, the diagonal anchored at seq_len_kv[b] - seq_len_q[b]);
    the per-batch Q lengths bind the shared no-split rule."""
    plan = _run_graph(B=3, H=H, KH=2, s_q=s_q, lens=[700, 130, 5], q_lens=[s_q, max(1, s_q - 1), 0], page=16, causal_br=True)
    _assert_decode_tile_plan(plan, G=H // 2)
    assert (plan.knobs.split_kv or 1) == 1, plan.knobs


@_gpu
def test_decode_graph_sliding_window_bottom_right():
    """Sliding window (left bound) on the bottom-right diagonal, S_q = 1, packed 16:1."""
    _assert_decode_tile_plan(_run_graph(B=2, H=32, KH=2, s_q=1, lens=[700, 130], page=16, causal_br=True, window_left=200), G=16)


@_gpu
def test_decode_graph_right_band_dense():
    """Top-left causal widened by a right band on a dense padded cache, S_q = 2, packed 4:1 (8 rows);
    the padded dense cache binds the shared no-split rule."""
    plan = _run_graph(B=2, H=8, KH=2, s_q=2, lens=[300, 129], page=0, window_right=100)
    _assert_decode_tile_plan(plan, G=4)
    assert (plan.knobs.split_kv or 1) == 1, plan.knobs


@_gpu
def test_decode_graph_sink_dense():
    """Attention sink folded once per Q row over a dense padded cache (S_q = 2, packed 8:1); a
    keyless batch keeps the sink's finite LSE and O := 0; a sink never splits."""
    plan = _run_graph(B=3, H=16, KH=2, s_q=2, lens=[700, 0, 300], page=0, sink=True)
    _assert_decode_tile_plan(plan, G=8)
    assert (plan.knobs.split_kv or 1) == 1, plan.knobs


@_gpu
@pytest.mark.parametrize("s_q", [1, 2])
def test_decode_graph_sink_paged(s_q):
    """The sink fold over a paged cache (page 16) at S_q = 1 and 2 -- the one paged + sink form
    the Rubin row serves (the paged PREFILL pipeline there is THD-only without a sink; the
    decode tile walks the block table itself).  A keyless batch keeps the sink's finite LSE
    and O := 0; packed 8:1, and a sink never splits."""
    plan = _run_graph(B=3, H=16, KH=2, s_q=s_q, lens=[700, 0, 300], page=16, sink=True)
    _assert_decode_tile_plan(plan, G=8)
    assert (plan.knobs.split_kv or 1) == 1, plan.knobs


@_gpu
@pytest.mark.parametrize("sink_logit", [-100.0, -1000.0, -2.0], ids=["sink_-100", "sink_-1000", "sink_-2_control"])
def test_decode_graph_sink_keyless_rows_keep_the_sink_logit(sink_logit):
    """A row with no live key and a sink has exactly one softmax column, so its LSE is the
    sink logit itself and O := 0 -- for the empty batch (KV length 0) and for the
    bottom-right row a one-key batch masks entirely alike; finite sinks far below zero."""
    _assert_decode_tile_plan(_run_graph(B=3, H=16, KH=2, s_q=2, lens=[0, 1, 128], page=0, sink=sink_logit, causal_br=True), G=8)


@_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
def test_decode_graph_stats_log2(dtype):
    """Base-2 Stats (stats_use_log2) on the decode tile, both half dtypes, packed 16:1."""
    _assert_decode_tile_plan(_run_graph(B=2, H=32, KH=2, s_q=1, lens=[700, 130], page=16, dtype=dtype, stats_log2=True), G=16, units=4, kv_tiles=6)


@_gpu
def test_decode_graph_dense_padded_no_stats():
    plan = _run_graph(B=2, H=32, KH=2, s_q=1, lens=[1000, 77], page=0, stats=False, dtype=torch.bfloat16)
    _assert_decode_tile_plan(plan, G=16)
    assert (plan.knobs.split_kv or 1) == 1, plan.knobs  # a padded dense cache: the shared no-split rule


@_gpu
def test_decode_graph_serving_shape_leads_unsplit_with_the_split_as_runner_up():
    """b=32 x 2 KV heads over 4096 keys (the serving shape) at S_q=1 on the Rubin row: the LEADING
    plan is packed 16:1 and does not split -- split 2 saves GPU time but costs an eager caller a
    second host launch per execute -- and the captured caller's split-2 plan is the runner-up,
    reachable by select_plan.  Both run and match the reference.  The literal splits are what the
    decode model answers at 148 / 204 / 212 SMs alike (test_split_kv_heuristic pins the policy); on
    another part the plans must still agree with the model.  The engine's ranked entries carry
    exactly that: a packed unsplit lead and a packed split-2 entry, every one admissible."""
    import cudnn

    sm = _sm_count()
    shape = dict(B=32, H=32, KH=2, s_q=1, lens=[4096] * 32, page=16, stats=False, dtype=torch.bfloat16)
    lead = _run_graph(**shape)
    _assert_decode_tile_plan(lead, G=16, units=64, kv_tiles=32)
    captured = choose_decode_tile_split_kv(units=64, kv_tiles=32, sm_count=sm, launch_cost=0.0)
    runner = _run_graph(**shape, split_kv=captured)
    assert runner.knobs.split_kv == captured and runner.knobs.pack_gqa is True, runner.knobs
    if sm in (148, 204, 212):
        assert (lead.knobs.split_kv, runner.knobs.split_kv) == (1, 2), (lead.knobs, runner.knobs)
    # The engine's ranked entries for the serving shape: the packed unsplit lead, a packed split entry.
    torch.manual_seed(0)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor(dim=[32, 32, 1, D], stride=[32 * D, D, 32 * D, 1], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_dense = torch.randn(32, 4096, 2, D, device="cuda", dtype=torch.bfloat16)
    k_c, v_c, bt = _pools(k_dense, k_dense, 16, True, 0)
    k, v = g.tensor_like(k_c), g.tensor_like(v_c)
    tk, tv = g.tensor_like(bt), g.tensor_like(bt)
    sq_t = g.tensor(dim=[32, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="sq")
    sk_t = g.tensor(dim=[32, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="sk")
    o, _ = g.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=False,
        attn_scale=1.0 / math.sqrt(D),
        use_padding_mask=True,
        seq_len_q=sq_t,
        seq_len_kv=sk_t,
        paged_attention_k_table=tk,
        paged_attention_v_table=tv,
        paged_attention_max_seq_len_kv=4096,
    )
    o.set_output(True).set_dim([32, 32, 1, D]).set_stride([32 * D, D, 32 * D, 1])
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    ours = [g.plans[i].knobs for i, n in enumerate(names) if _is_plan_for(n, _engine())]
    assert ours and ours[0].pack_gqa is True and (ours[0].split_kv or 1) == 1, ours
    assert any(kn.pack_gqa is True and kn.split_kv == captured for kn in ours), ours
    # The same shape one token wider (32/2 at S_q=2 bottom-right: 32 packed rows) is the 32-column
    # tile's, which is NOT routed -- and on the Rubin row no OTHER kernel packs it either (the d256
    # prefill kernel wires no PackGQA, unlike SM100's partial-PackGQA prefill tile that takes this
    # shape there), so it rides the decode tile UNPACKED: two live rows per (batch, Q head) unit,
    # unsplit (1024 units already saturate the machine).  This pin does NOT invert with the row's
    # PackGQA claim; it inverts only when a Rubin kernel packs 32 rows (the routed 32-column tile).
    mtp = _run_graph(B=32, H=32, KH=2, s_q=2, lens=[4096] * 32, page=16, stats=False, dtype=torch.bfloat16, causal_br=True)
    assert mtp.knobs.pack_gqa is not True and (mtp.knobs.split_kv or 1) == 1, mtp.knobs
    # Its 16-row sibling (16/2 at S_q=2: 8:1 packing) is decode-shaped PACKED and follows the serving
    # shape's policy: unsplit lead, split-2 runner-up.
    mtp16 = _run_graph(B=32, H=16, KH=2, s_q=2, lens=[4096] * 32, page=16, stats=False, dtype=torch.bfloat16, causal_br=True)
    _assert_decode_tile_plan(mtp16, G=8, units=64, kv_tiles=32)


@_gpu
def test_decode_graph_serving_shape_leads_with_the_split_when_replay_is_expected():
    """The serving shape on a graph created with is_cuda_graph_replay_expected=True: the captured
    caller's optimum (split 2) LEADS and the eager-safe unsplit plan follows as a runner-up,
    reachable by select_plan.  The hint changes no numerics: both plans run and match the reference."""
    sm = _sm_count()
    shape = dict(B=32, H=32, KH=2, s_q=1, lens=[4096] * 32, page=16, stats=False, dtype=torch.bfloat16, graph_kwargs=dict(is_cuda_graph_replay_expected=True))
    lead = _run_graph(**shape)
    _assert_decode_tile_plan(lead, G=16, units=64, kv_tiles=32, replay=True)
    eager = choose_decode_tile_split_kv(units=64, kv_tiles=32, sm_count=sm)
    runner = _run_graph(**shape, split_kv=eager)
    assert runner.knobs.split_kv == eager and runner.knobs.pack_gqa is True, runner.knobs
    if sm in (148, 204, 212):
        assert (lead.knobs.split_kv, runner.knobs.split_kv) == (2, 1), (lead.knobs, runner.knobs)


@_gpu
def test_decode_graph_small_batch_splits_and_recombines():
    """b=8 x 2 KV heads is 16 units: unsplit they would stream 32 tiles each on 16 of the SMs, so
    the decode model splits (8 ways: 128 CTAs, and the GPU saving covers the second launch) and the
    recombined O / LSE over mixed lengths (incl. 1 and 77) match the reference."""
    plan = _run_graph(B=8, H=32, KH=2, s_q=1, lens=[4096, 4000, 129, 1, 2048, 4096, 300, 77], page=16, dtype=torch.bfloat16)
    _assert_decode_tile_plan(plan, G=16, units=16, kv_tiles=32)
    if _sm_count() in (148, 204, 212):
        assert plan.knobs.split_kv == 8, plan.knobs


@_gpu
def test_decode_graph_deep_split_empty_ranges():
    """Six units over a 4096-key table: the decode rule splits 16 ways (96 CTAs), so the 1- and
    129-key batches leave most split ranges empty next to live ones; those must yield -inf / 0
    partials the combine ignores."""
    plan = _run_graph(B=3, H=32, KH=2, s_q=1, lens=[4096, 1, 129], page=16)
    _assert_decode_tile_plan(plan, G=16, units=6, kv_tiles=32)
    assert plan.knobs.split_kv >= (8 if _sm_count() >= 96 else 2), plan.knobs


# --- the three equalities the row's claim rests on ------------------------------------------------


@_gpu
def test_decode_graph_qwen_24_2_packs_twelve_rows_into_the_tile():
    """The acceptance rule at the Qwen 24/2 geometry, graph level: the plan is PACKED 12:1 and the
    module the executor loaded packs the whole group into the 16-row tile (HEADS_PER_TILE = 12,
    Q_BOX_TOKENS = 1, 12 live rows + 4 zero tail rows), one CTA per (batch, KV head) unit instead of
    twelve; the output matches the reference and the split is the decode model's (8 units x 8 tiles)."""
    cap = {}
    plan = _run_graph(B=4, H=24, KH=2, s_q=1, lens=[700, 130, 5, 1024], page=16, dtype=torch.bfloat16, capture=cap)
    _assert_decode_tile_plan(plan, G=12, units=4 * 2, kv_tiles=8)
    mod = cap["module"]
    assert "sm107" in mod.__name__ and mod.__file__.endswith("/sm107/decode_d256_f16.py"), mod.__name__
    assert (mod.HEADS_PER_TILE, mod.Q_BOX_TOKENS, mod.Q_BOX_ROWS, mod.N_Q - mod.Q_BOX_ROWS, mod.N_Q, mod.CFG.SPLIT_KV) == (
        12,
        1,
        12,
        4,
        16,
        plan.knobs.split_kv or 1,
    )


@_gpu
@pytest.mark.parametrize(("page", "hnd"), [(16, True), (64, False)], ids=["page16_HND", "page64_NHD"])
def test_decode_graph_paged_equals_dense_bitwise(page, hnd):
    """The same tokens through a dense padded cache and through page pools (24/2, packed 12:1, both
    unsplit -- the padded dense graph cannot split, the paged one is pinned to its unsplit entry):
    the tile's PAGED_KV specialization only redirects the K/V tile loads through the block table
    (same tiles, same accumulation order, the padding mask zeroing the same columns), so O and LSE
    are BITWISE equal."""
    kw = dict(B=4, H=24, KH=2, s_q=1, lens=[700, 130, 5, 1024], dtype=torch.bfloat16, seed=3)
    dense, paged = {}, {}
    pd = _run_graph(**kw, page=0, capture=dense)
    pp = _run_graph(**kw, page=page, hnd=hnd, split_kv=1, capture=paged)
    _assert_decode_tile_plan(pd, G=12)
    _assert_decode_tile_plan(pp, G=12)
    assert (pd.knobs.split_kv or 1) == 1 == (pp.knobs.split_kv or 1), (pd.knobs, pp.knobs)
    assert torch.equal(dense["o"], paged["o"]), f"paged != dense: max |diff| {(dense['o'] - paged['o']).abs().max().item():.3e}"
    assert torch.equal(dense["lse"], paged["lse"]), f"paged != dense LSE: max |diff| {(dense['lse'] - paged['lse']).abs().max().item():.3e}"


@_gpu
def test_decode_graph_packed_equals_unpacked_bitwise():
    """24/2 over pages: the packed plan (12 live rows in one 16-row unit per (batch, KV head)) and the
    unpacked plan (one live row per (batch, Q head) unit, the former row contract) compute each Q
    row's softmax over the same KV tiles in the same order -- a column of S^T does not depend on
    which N column it occupies -- so O and LSE are BITWISE equal.  Both pinned unsplit."""
    kw = dict(B=3, H=24, KH=2, s_q=1, lens=[1000, 129, 640], page=32, dtype=torch.bfloat16, seed=5)
    packed, unpacked = {}, {}
    pp = _run_graph(**kw, pack_gqa=True, split_kv=1, capture=packed)
    pu = _run_graph(**kw, pack_gqa=False, split_kv=1, capture=unpacked)
    assert pp.knobs.pack_gqa is True and pu.knobs.pack_gqa is not True, (pp.knobs, pu.knobs)
    assert (packed["module"].HEADS_PER_TILE, unpacked["module"].HEADS_PER_TILE) == (12, 1)
    assert torch.equal(packed["o"], unpacked["o"]), f"packed != unpacked: max |diff| {(packed['o'] - unpacked['o']).abs().max().item():.3e}"
    assert torch.equal(packed["lse"], unpacked["lse"]), f"packed != unpacked LSE: max |diff| {(packed['lse'] - unpacked['lse']).abs().max().item():.3e}"


@_gpu
@pytest.mark.parametrize(
    ("form", "dtype"),
    [("paged", torch.bfloat16), ("paged", torch.float16), ("dense_unpadded", torch.bfloat16)],
    ids=["paged_bf16", "paged_f16", "dense_unpadded_bf16"],
)
def test_decode_graph_split_equals_unsplit_within_the_combine_rounding(form, dtype):
    """The decode model's split (the leading plan: 8 ways at b=8 x 2 KV heads over pages, 16 ways at
    b=3 x 2 KV heads over an UNPADDED dense cache -- the dense split the row claims) against the
    pinned unsplit plan, same graph: NOT bitwise, and bounded.  The split path renormalises each
    fp32 partial by exp(lse_s - lse) in the combine where the unsplit path rescales its running
    accumulator per KV tile, so the two fp32 sums are the same value associated differently: after
    the single cast to the output dtype they agree or differ by ONE output ulp (asserted per element
    at the larger magnitude; the measured magnitude is printed), and the fp32 LSEs agree to fp32
    rounding (2e-5 on values of order 10).  Both plans match the fp32 reference on their own."""
    if form == "paged":
        kw = dict(B=8, H=32, KH=2, s_q=1, lens=[4096, 4000, 129, 1, 2048, 4096, 300, 77], page=16, dtype=dtype, seed=11)
        G, units, kv_tiles = 16, 16, 32
    else:
        kw = dict(B=3, H=24, KH=2, s_q=1, lens=[4096] * 3, page=0, padded=False, dtype=dtype, seed=13)
        G, units, kv_tiles = 12, 6, 32
    split, unsplit = {}, {}
    ps = _run_graph(**kw, capture=split)
    _assert_decode_tile_plan(ps, G=G, units=units, kv_tiles=kv_tiles)
    assert ps.knobs.split_kv > 1, ps.knobs
    pu = _run_graph(**kw, split_kv=1, capture=unsplit)
    assert pu.knobs.pack_gqa is True and (pu.knobs.split_kv or 1) == 1, pu.knobs
    _assert_within_one_output_ulp(split["o"], unsplit["o"], dtype, f"split {ps.knobs.split_kv} vs unsplit O ({form}, {dtype})")
    lse_diff = (split["lse"] - unsplit["lse"]).abs().max().item()
    print(f"split {ps.knobs.split_kv} vs unsplit LSE ({form}): max |diff| = {lse_diff:.3e}")
    assert lse_diff <= 2e-5, lse_diff


# --- the template level: the tile's packing and split partials, under the row's claims --------------


def _launch_template(mod, *, q, k_view, v_view, lens, scale, splits, dtype, bt=None, page=0, seq_q_lens_addr=0):
    """Drive the Rubin decode module directly through launch_f16 (dense or paged, split or not)
    and recombine the fp32 partials with the shared combine; returns (O, LSE)."""
    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb
    from test_sdpa_fwd_split_kv_sm100 import _partial_kwargs, _partial_o_dtype, _partial_tag

    B, s_q, H, _ = q.shape
    KH = k_view.shape[2]
    dev = q.device
    fn = mod.compile(d_qk=D, d_v=D, has_lse=True, paged_hnd=(bt is not None))
    o_p = torch.zeros(splits * B, s_q, H, D, device=dev, dtype=_partial_o_dtype(splits, dtype))
    lse_p = torch.zeros(splits * B, H, s_q, device=dev, dtype=torch.float32)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    launch_f16(
        fn,
        q,
        k_view,
        v_view,
        o_p,
        lse_p,
        torch.zeros(H, dtype=torch.float32, device=dev),
        torch.tensor(lens, dtype=torch.int32, device=dev),
        torch.zeros(1, dtype=torch.int64, device=dev),
        (B, H, KH, s_q, 0 if bt is not None else k_view.shape[1], 0),
        cutlass.Float32(scale * math.log2(math.e)),
        cutlass.Int32(0),
        seq_q_lens_addr,
        **_partial_kwargs(splits, o_p),
        block_table_tensor=bt,
        block_table_v_tensor=bt,
        page_size=page,
        stream=stream,
        host=mod._host,
    )
    if splits == 1:
        torch.cuda.synchronize()
        return o_p, lse_p
    o_out = torch.zeros(B, s_q, H, D, device=dev, dtype=dtype)
    lse_out = torch.zeros(B, H, s_q, device=dev, dtype=torch.float32)
    cfn = positional_entry(comb.compile_ptr(dtype_o="f16" if dtype == torch.float16 else "bf16", has_lse=True, dtype_partial=_partial_tag(splits, dtype)))
    cfn(o_p.data_ptr(), lse_p.data_ptr(), o_out.data_ptr(), lse_out.data_ptr(), (B, H, s_q, D), splits, o_out.stride(), lse_out.stride(), int(stream))
    torch.cuda.synchronize()
    return o_out, lse_out


@_gpu
@pytest.mark.parametrize("splits", [1, 2], ids=["unsplit", "split2"])
def test_decode_kernel_qwen_24_2_packs_twelve_rows(splits):
    """The 24/2 geometry on the tile's WHOLE-GROUP packing: HEADS_PER_TILE = 12, one token per
    16-row Q box, 12 live rows and 4 zero-filled tail rows per (batch, KV head) unit -- driven
    through the module loader (pack_gqa=True, qh_per_kh=12) over a paged cache, unsplit and
    split 2 with the fp32 partials recombined, against the fp32 reference -- the kernel-level
    evidence under the row's PackGQA claim (test_decode_graph_qwen_24_2_packs_twelve_rows_into_the_tile
    is its graph-level twin)."""
    B, H, KH, P, s_q = 3, 24, 2, 16, 1
    lens = [700, 130, 5]
    dtype = torch.bfloat16
    torch.manual_seed(7)
    S = max(lens)
    q = torch.randn(B, s_q, H, D, device="cuda", dtype=dtype)
    k_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    k_c, v_c, bt4 = _pools(k_dense, v_dense, P, True, seed=7)
    mod = _load(dtype_qkv=2, seq_kv_lens_present=True, paged_kv=True, page_size=P, split_kv=splits, pack_gqa=True, qh_per_kh=H // KH, decode_q_tile=16)
    assert (mod.HEADS_PER_TILE, mod.Q_BOX_TOKENS, mod.Q_BOX_ROWS, mod.N_Q - mod.Q_BOX_ROWS, mod.CFG.SPLIT_KV) == (12, 1, 12, 4, splits)
    o, lse = _launch_template(
        mod,
        q=q,
        k_view=k_c.permute(0, 2, 1, 3),
        v_view=v_c.permute(0, 2, 1, 3),
        lens=lens,
        scale=1.0 / math.sqrt(D),
        splits=splits,
        dtype=dtype,
        bt=bt4.view(B, -1),
        page=P,
    )
    ref_o, ref_lse = _ref(q, k_dense, v_dense, lens, None, 1.0 / math.sqrt(D))
    torch.testing.assert_close(o.float(), ref_o, atol=5e-2, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=5e-3, rtol=0)


@_gpu
@pytest.mark.parametrize("splits", [1, 2], ids=["unsplit", "split2"])
def test_decode_kernel_two_column_groups(splits):
    """The 32-column tile (N_Q = 32: two softmax column groups, 8 softmax warps over the same 128
    TMEM lanes) compiles and computes correctly on Rubin but is not routed (the graph tests pin
    those shapes on the prefill tile) -- S_q = 2 x 16:1 packing = 32 rows over a paged cache,
    unsplit and split 2 with the partials recombined."""
    B, H, KH, P, s_q = 3, 32, 2, 16, 2
    lens = [700, 130, 5]
    dtype = torch.float16
    torch.manual_seed(5)
    S = max(lens)
    q = torch.randn(B, s_q, H, D, device="cuda", dtype=dtype)
    k_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    k_c, v_c, bt4 = _pools(k_dense, v_dense, P, True, seed=5)
    mod = _load(dtype_qkv=3, seq_kv_lens_present=True, paged_kv=True, page_size=P, split_kv=splits, pack_gqa=True, qh_per_kh=H // KH, decode_q_tile=32)
    assert (mod.CFG.N_Q, mod.CFG.SOFTMAX_WARPS, mod.COL_GROUPS, mod.DESC_VERSION) == (32, 8, 2, 0)
    o, lse = _launch_template(
        mod,
        q=q,
        k_view=k_c.permute(0, 2, 1, 3),
        v_view=v_c.permute(0, 2, 1, 3),
        lens=lens,
        scale=1.0 / math.sqrt(D),
        splits=splits,
        dtype=dtype,
        bt=bt4.view(B, -1),
        page=P,
    )
    ref_o, ref_lse = _ref(q, k_dense, v_dense, lens, None, 1.0 / math.sqrt(D))
    torch.testing.assert_close(o.float(), ref_o, atol=2e-2, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=5e-3, rtol=0)


@_gpu
def test_decode_kernel_dense_split_partials_recombine():
    """Dense padded K/V, split 4 at the routed width (N_Q = 16, unpacked 16:1 at S_q = 1): the
    fp32 partials over mixed lengths (an empty range next to live ones) recombine to the
    reference -- the kernel-level evidence under the row's dense split claim
    (test_decode_graph_split_equals_unsplit_within_the_combine_rounding is its graph-level twin)."""
    B, H, KH, s_q = 3, 8, 2, 1
    lens = [4096, 1, 129]
    dtype = torch.bfloat16
    torch.manual_seed(11)
    S = max(lens)
    q = torch.randn(B, s_q, H, D, device="cuda", dtype=dtype)
    k_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    mod = _load(dtype_qkv=2, seq_kv_lens_present=True, split_kv=4, pack_gqa=True, qh_per_kh=H // KH, decode_q_tile=16)
    assert (mod.HEADS_PER_TILE, mod.CFG.SPLIT_KV, mod.PAGED_KV) == (4, 4, False)
    o, lse = _launch_template(mod, q=q, k_view=k_dense, v_view=v_dense, lens=lens, scale=1.0 / math.sqrt(D), splits=4, dtype=dtype)
    ref_o, ref_lse = _ref(q, k_dense, v_dense, lens, None, 1.0 / math.sqrt(D))
    torch.testing.assert_close(o.float(), ref_o, atol=5e-2, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=5e-3, rtol=0)


# --- routing boundary and the adapter ---------------------------------------------------------------


@_gpu
def test_decode_adapter_routes_decode_shaped_graphs_only():
    """The adapter's routing rule on cc 10.7 (the acceptance rule): 16 for S_q x pack_g <= 16 rows
    and 0 above.  Positives through check_support on paged pools (the form the Rubin adapter admits
    PackGQA on): the 24/2 geometry packed (12 rows), 16:1 packed, the 16-row MTP step, 16 MHA rows
    and the unpacked 32/2 step; the rule's zeros through the same adapter's shape-explicit twin
    (_decode_q_tile_for, what check_support and _decode_q_tile both read), because a paged packed
    graph PAST the tile is not a graph the Rubin row serves at all (its paged prefill pipeline is
    THD-only) -- asserted as the typed decline."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    def api(*, s_q, H, KH, pack):
        B, P, S = 2, 16, 256
        dtype = torch.bfloat16
        k_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
        k_c, v_c, _ = _pools(k_dense, k_dense, P, False, seed=1)
        q_gpu = torch.randn(B, s_q, H, D, device="cuda", dtype=dtype).transpose(1, 2)
        o_gpu = torch.empty_like(q_gpu)
        lse = torch.empty(B, H, s_q, device="cuda", dtype=torch.float32)
        a = SdpaFwdDslSm100(
            sample_q=q_gpu,
            sample_k=k_c,
            sample_v=v_c,
            sample_o=o_gpu,
            sample_lse=lse,
            seq_kv_lens_present=True,
            paged_page_size=P,
            paged_max_seq_len_kv=S,
            pack_gqa=pack,
        )
        a.check_support()
        return a

    packed = api(s_q=1, H=24, KH=2, pack=True)
    assert packed._decode_q_tile() == 16  # 12 packed rows: the 24/2 geometry
    assert api(s_q=1, H=32, KH=2, pack=True)._decode_q_tile() == 16  # 16 packed rows
    assert api(s_q=2, H=16, KH=2, pack=True)._decode_q_tile() == 16  # the 16-row MTP step
    assert api(s_q=16, H=4, KH=4, pack=False)._decode_q_tile() == 16  # 16 MHA rows
    assert api(s_q=1, H=32, KH=2, pack=False)._decode_q_tile() == 16  # unpacked: one row per head
    # The rule's zeros, read through the packed adapter's shape-explicit twin:
    assert packed._decode_q_tile_for(2, 32, 2) == 0  # 32 packed rows: the unrouted wide tile
    assert packed._decode_q_tile_for(1, 32, 1) == 0  # 32 packed rows at S_q = 1
    assert packed._decode_q_tile_for(3, 16, 2) == 0  # 24 packed rows
    assert packed._decode_q_tile_for(1, 24, 2) == 16 and packed._decode_q_tile_for(16, 1, 1) == 16
    unpacked = api(s_q=1, H=32, KH=2, pack=False)
    assert unpacked._decode_q_tile_for(17, 4, 4) == 0 and unpacked._decode_q_tile_for(16, 4, 4) == 16  # 17 vs 16 MHA rows
    # ... and a paged packed graph past the tile is the row's typed decline, never the prefill tile.
    with pytest.raises(NotImplementedError, match="decode-shaped half D256 graph"):
        api(s_q=2, H=32, KH=2, pack=True)


@_gpu
def test_decode_adapter_dense_pack_and_split_follow_the_tile():
    """The standalone adapter on cc 10.7, DENSE cache (the form its former 'Rubin half PackGQA
    requires paged KV' gate declined): ACCEPT -- a decode-shaped graph packs the whole group (24/2
    -> 12 rows) and splits (the fp32 partials + the shared combine), the 16-row MTP step packs 8:1;
    REJECT -- a packed graph past the tile (32/2 at S_q = 2: 32 rows) and a split past the tile
    (17 MHA rows) are typed declines naming the decode tile, because the d256 prefill kernel wires
    neither (never a silent unpacked / unsplit run)."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    def api(*, s_q, H, KH, pack, split=1, seq_lens=True):
        B, S = 2, 256
        dtype = torch.bfloat16
        q_gpu = torch.randn(B, s_q, H, D, device="cuda", dtype=dtype).transpose(1, 2)
        k_c = torch.randn(B, S, KH, D, device="cuda", dtype=dtype).transpose(1, 2)
        v_c = torch.randn(B, S, KH, D, device="cuda", dtype=dtype).transpose(1, 2)
        a = SdpaFwdDslSm100(
            sample_q=q_gpu,
            sample_k=k_c,
            sample_v=v_c,
            sample_o=torch.empty_like(q_gpu),
            sample_lse=torch.empty(B, H, s_q, device="cuda", dtype=torch.float32),
            seq_kv_lens_present=seq_lens,
            pack_gqa=pack,
            split_kv=split,
        )
        a.check_support()
        return a

    assert api(s_q=1, H=24, KH=2, pack=True)._decode_q_tile() == 16  # 12 packed rows over a dense cache
    assert api(s_q=2, H=16, KH=2, pack=True)._decode_q_tile() == 16  # the 16-row MTP step
    a = api(s_q=1, H=24, KH=2, pack=True, split=2, seq_lens=False)  # the dense (unpadded) split, packed
    assert a._decode_q_tile() == 16 and a.split_kv == 2
    a = api(s_q=16, H=4, KH=4, pack=False, split=4, seq_lens=False)  # 16 MHA rows, unpacked split
    assert a._decode_q_tile() == 16 and a.split_kv == 4
    with pytest.raises(NotImplementedError, match="d256 decode tile"):
        api(s_q=2, H=32, KH=2, pack=True)  # 32 packed rows: no Rubin d256 kernel packs them
    with pytest.raises(NotImplementedError, match="d256 decode tile"):
        api(s_q=17, H=4, KH=4, pack=False, split=2, seq_lens=False)  # 17 MHA rows: the prefill kernel, no dense split
    with pytest.raises(ValueError, match="unpadded dense graphs only"):
        api(s_q=1, H=24, KH=2, pack=True, split=2, seq_lens=True)  # a PADDED dense split: the shared structural rule


def _paged_graph_offers_engine(*, B, H, KH, s_q, d, lens, page=16, causal_br=False):
    """Build a paged dense-Q graph and report whether the Rubin FROST engine lists ANY plan for it
    (a typed absence at plan creation, never a bare error)."""
    import cudnn

    torch.manual_seed(0)
    S = max(lens)
    k_dense = torch.randn(B, S, KH, d, device="cuda", dtype=torch.float16)
    k_c, v_c, bt = _pools(k_dense, k_dense, page, True, 0)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor(dim=[B, H, s_q, d], stride=[s_q * H * d, d, H * d, 1], data_type=cudnn.data_type.HALF, name="q")
    k, v = g.tensor_like(k_c), g.tensor_like(v_c)
    tk, tv = g.tensor_like(bt), g.tensor_like(bt)
    sq_t = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="sq")
    sk_t = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="sk")
    kw = dict(use_causal_mask_bottom_right=True) if causal_br else {}
    o, _ = g.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=False,
        attn_scale=1.0 / math.sqrt(d),
        use_padding_mask=True,
        seq_len_q=sq_t,
        seq_len_kv=sk_t,
        paged_attention_k_table=tk,
        paged_attention_v_table=tv,
        paged_attention_max_seq_len_kv=S,
        **kw,
    )
    o.set_output(True).set_dim([B, H, s_q, d]).set_stride([s_q * H * d, d, H * d, 1])
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    return offers_engine(g, _engine())


@_gpu
def test_decode_routing_boundary():
    """S_q x G past the routed tile stays on the prefill d256 tile over a DENSE cache (the Rubin
    row's prefill tile serves dense Q; counted at its unpacked geometry: S_q rows per head), while
    over a PAGED cache a dense-Q graph that is not decode-shaped is not served on the Rubin row at
    all -- its paged prefill pipeline is THD-only -- and a decode-shaped paged d128 graph has no
    Rubin decode tile either: both are typed absences at plan creation.  INVERTS when the Rubin
    row serves dense-Q paged prefill or gains a d128 decode tile."""
    _run_graph(B=2, H=8, KH=2, s_q=24, lens=[700, 130], page=0, causal_br=True, expect=PREFILL)  # dense, 24 rows per head
    _run_graph(B=2, H=4, KH=4, s_q=17, lens=[700, 130], page=0, expect=PREFILL)  # dense, 17 MHA rows
    _run_graph(B=2, H=32, KH=2, s_q=32, lens=[700, 130], page=0, causal_br=True, expect=PREFILL)  # dense, 32 rows per head
    assert _paged_graph_offers_engine(B=2, H=8, KH=2, s_q=1, d=D, lens=[700, 130]), "the decode-shaped paged d256 graph IS served (the decode tile)"
    assert not _paged_graph_offers_engine(
        B=2, H=4, KH=4, s_q=17, d=D, lens=[700, 130]
    ), "17 MHA rows over a paged cache: not decode-shaped, THD-only paged prefill"
    assert not _paged_graph_offers_engine(B=2, H=32, KH=2, s_q=32, d=D, lens=[700, 130], causal_br=True), "32 rows per head over a paged cache"
    assert not _paged_graph_offers_engine(B=2, H=8, KH=2, s_q=1, d=128, lens=[700, 130]), "paged d128 decode graph: no Rubin d128 decode tile"


@_gpu
def test_decode_routing_boundary_thd_queries():
    """Ragged (THD) queries -- packed [T, H, d] Q/O storage with ragged offsets, one token per
    sequence -- over a paged cache stay on the prefill d256 tile (the Rubin row's paged THD
    pipeline; the decode tile has no THD scheduler); the graph runs under the D2H detector and
    matches the reference."""
    import cudnn
    import cudnn.sdpa  # noqa: F401

    dev, dtype = "cuda", torch.float16
    B, H, KH, P = 3, 32, 2, 16
    lens = [700, 130, 5]
    S = max(lens)
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(3)
    k_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    k_c, v_c, bt = _pools(k_dense, v_dense, P, True, seed=3)
    q_pk = torch.randn(B, H, D, device=dev, dtype=dtype)  # T = B tokens, one per sequence
    stride = (H * D, D, H * D, 1)  # (B, H, S_max=1, d) over the packed [T, H, d] storage
    q_gpu = q_pk.view(-1).as_strided((B, H, 1, D), stride)
    o_stor = torch.zeros(B * H * D, device=dev, dtype=dtype)
    o_gpu = o_stor.as_strided((B, H, 1, D), stride)
    slq = torch.ones(B, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    ro = (torch.arange(B + 1, dtype=torch.int64, device=dev) * H * D).view(B + 1, 1, 1, 1)

    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H, 1, D], stride=list(stride), data_type=cudnn.data_type.HALF, name="q")
    k, v = g.tensor_like(k_c), g.tensor_like(v_c)
    tk, tv = g.tensor_like(bt), g.tensor_like(bt)
    sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
    qro, oro = g.tensor_like(ro), g.tensor_like(ro)
    tq.set_ragged_offset(qro)
    o, _ = g.sdpa(
        name="sdpa",
        q=tq,
        k=k,
        v=v,
        generate_stats=False,
        attn_scale=scale,
        use_padding_mask=True,
        seq_len_q=sq_t,
        seq_len_kv=sk_t,
        paged_attention_k_table=tk,
        paged_attention_v_table=tv,
        paged_attention_max_seq_len_kv=S,
    )
    o.set_output(True).set_dim([B, H, 1, D]).set_stride(list(stride))
    o.set_ragged_offset(oro)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, _engine())
    idx = g._plan_index
    g.check_support()
    g.build_plans()
    assert _served_by(g, idx) == PREFILL, f"plan {g.get_plan_name_at_index(idx)} served by {_served_by(g, idx)}, expected {PREFILL}"
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute({tq: q_gpu, k: k_c, v: v_c, tk: bt, tv: bt, sq_t: slq, sk_t: slk, qro: ro, oro: ro, o: o_gpu}, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    ref_o, _ = _ref(q_pk.view(B, 1, H, D), k_dense, v_dense, lens, None, scale)
    torch.testing.assert_close(o_stor.view(B, 1, H, D).float(), ref_o, atol=2e-2, rtol=0)


@_gpu
def test_decode_adapter_cuda_graph_replay_no_host_sync():
    """The adapter's execute path captured once at fixed B; seq_lens CONTENT changes between
    replays (Rule 3, the SM100 suite's protocol), in the SM100 suite's packed split_kv=4 form:
    the two launches (the tile's partials, the combine) replay without a host sync."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    B, H, KH, P, S = 8, 32, 2, 16, 1024
    dev, dtype = "cuda", torch.float16
    torch.manual_seed(2)
    k_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    k_c, v_c, bt4 = _pools(k_dense, v_dense, P, False, seed=2)
    bt = bt4.view(B, -1)
    q_gpu = torch.randn(B, 1, H, D, device=dev, dtype=dtype).transpose(1, 2)
    o_gpu = torch.empty(B, 1, H, D, device=dev, dtype=dtype).transpose(1, 2)
    lse = torch.empty(B, H, 1, device=dev, dtype=torch.float32)
    seq_lens = torch.full((B,), 1000, dtype=torch.int32, device=dev)
    seq_q = torch.ones(B, dtype=torch.int32, device=dev)
    api = SdpaFwdDslSm100(
        sample_q=q_gpu,
        sample_k=k_c,
        sample_v=v_c,
        sample_o=o_gpu,
        sample_lse=lse,
        seq_kv_lens_present=True,
        seq_q_lens_present=True,
        paged_page_size=P,
        paged_max_seq_len_kv=S,
        split_kv=4,
        pack_gqa=True,
    )
    api.check_support()
    api.compile()
    assert api.kernel_template == DECODE and "sm107" in type(api._k_mod).__module__ + api._k_mod.__name__
    assert api.split_kv == 4 and api._k_mod.CFG.SPLIT_KV == 4 and api._k_mod.HEADS_PER_TILE == 16
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        api.execute(q_gpu, k_c, v_c, o_gpu, lse_tensor=lse, seq_kv_lens=seq_lens, seq_q_lens=seq_q, block_table=bt, workspace=ws)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        prev = torch.cuda.get_sync_debug_mode()
        with torch.cuda.graph(g, stream=s):
            torch.cuda.set_sync_debug_mode("error")
            try:
                api.execute(q_gpu, k_c, v_c, o_gpu, lse_tensor=lse, seq_kv_lens=seq_lens, seq_q_lens=seq_q, block_table=bt, workspace=ws)
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        scale = 1.0 / math.sqrt(D)
        for new_lens in ([5, 1024, 77, 128, 129, 1, 512, 1000], [1024] * B, [0, 1, 2, 3, 4, 5, 6, 7]):
            seq_lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
            g.replay()
            torch.cuda.synchronize()
            ref_o, ref_lse = _ref(q_gpu.transpose(1, 2), k_dense, v_dense, new_lens, None, scale)
            live = ~torch.isinf(ref_lse)
            torch.testing.assert_close(o_gpu.transpose(1, 2).float(), ref_o, atol=2e-2, rtol=0)
            torch.testing.assert_close(lse.view(B, H, 1)[live], ref_lse[live], atol=5e-3, rtol=0)
    finally:
        g.reset()
