# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A graph with a score modifier is served by the cuDNN backend, never by a FROST row.

No FROST SDPA row implements ``score_mod`` (none sets ``Capabilities.score_mod``), so every row must
DECLINE such a graph, and the row that would serve the same graph WITHOUT the modifier must decline it
by name. The analyzer used to read the callback from the record key ``fn`` -- the kwarg name of
``sdpa_fp8()`` only -- while ``sdpa()`` / ``sdpa_backward()`` spell it ``score_mod`` (and
``score_mod_bprop``). The fact was therefore False for every f16 graph, the SM100 prefill row served
TransformerEngine's flex-attention graphs (b2 h4 s16 d64, sbhd / bshd strides) and the kernel ran plain
attention: the user's modifier was silently dropped (38 % of O outside a 5e-2 tolerance on the
post-scale-bias variant, 33 / 8192 elements on softcap). Host-only: facts and row admission are
evaluated without building a plan; the by-name check runs wherever a row admits the control graph.
"""

import pytest

import cudnn

pytestmark = pytest.mark.L0

_B, _H, _S, _D = 2, 4, 16, 64


def _graph(io=cudnn.data_type.BFLOAT16):
    return cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)


def _bhsd(g, name, layout="bshd", d=_D, dtype=cudnn.data_type.BFLOAT16):
    # TE declares BHSD dims with the strides of its physical sbhd / bshd buffers.
    if layout == "sbhd":
        stride = (_H * d, d, _B * _H * d, 1)
    else:
        stride = (_S * _H * d, d, _H * d, 1)
    return g.tensor(dim=(_B, _H, _S, d), stride=stride, data_type=dtype, name=name)


def _identity_mod(graph, score):
    return score


def _post_scale_bias(graph, score):
    """TE's post_scale_bias score_mod: score + (row - col)."""
    row = graph.gen_index(input=score, axis=2)
    row.set_data_type(cudnn.data_type.INT32)
    col = graph.gen_index(input=score, axis=3)
    col.set_data_type(cudnn.data_type.INT32)
    bias = graph.sub(a=row, b=col, compute_data_type=cudnn.data_type.FLOAT)
    bias.set_data_type(cudnn.data_type.FLOAT)
    return graph.add(a=score, b=bias, compute_data_type=cudnn.data_type.FLOAT)


def _fwd_graph(layout, score_mod):
    g = _graph()
    q, k, v = (_bhsd(g, n, layout) for n in "qkv")
    kwargs = {"score_mod": score_mod} if score_mod is not None else {}
    o, stats = g.sdpa(q=q, k=k, v=v, attn_scale=0.125, generate_stats=True, use_causal_mask=False, **kwargs)
    o.set_output(True).set_dim((_B, _H, _S, _D)).set_stride(q.stride).set_data_type(cudnn.data_type.BFLOAT16)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return g


def _bwd_graph(layout, score_mod, score_mod_bprop):
    g = _graph()
    q, k, v, o, do = (_bhsd(g, n, layout) for n in ("q", "k", "v", "o", "dO"))
    stats = g.tensor(dim=(_B, _H, _S, 1), stride=(_H * _S, _S, 1, 1), data_type=cudnn.data_type.FLOAT, name="stats")
    kwargs = {}
    if score_mod is not None:
        kwargs["score_mod"] = score_mod
    if score_mod_bprop is not None:
        kwargs["score_mod_bprop"] = score_mod_bprop
    dq, dk, dv = g.sdpa_backward(q=q, k=k, v=v, o=o, dO=do, stats=stats, attn_scale=0.125, use_causal_mask=False, **kwargs)
    for t in (dq, dk, dv):
        t.set_output(True).set_dim((_B, _H, _S, _D)).set_stride(q.stride).set_data_type(cudnn.data_type.BFLOAT16)
    return g


def _fp8_graph(fn):
    g = _graph(cudnn.data_type.FP8_E4M3)
    q, k, v = (_bhsd(g, n, "bshd", d=128, dtype=cudnn.data_type.FP8_E4M3) for n in "qkv")
    scalars = {
        n: g.tensor(dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name=n)
        for n in ("descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o")
    }
    kwargs = {"fn": fn} if fn is not None else {}
    o, stats, amax_s, amax_o = g.sdpa_fp8(q=q, k=k, v=v, generate_stats=True, attn_scale=0.125, **scalars, **kwargs)
    o.set_output(True).set_dim((_B, _H, _S, 128)).set_stride(q.stride).set_data_type(cudnn.data_type.FP8_E4M3)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    amax_s.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return g


def _facts(g):
    from cudnn.sdpa import graph_analyzer as ga

    facts = ga.analyze(g)
    assert facts is not None and facts.invalid is None, getattr(facts, "invalid", facts)
    return facts


def _row_serves_this_device(spec, facts):
    """The row's own device gate, as ``engines.mismatch`` applies it: ``sm_lo..sm_hi`` covers the device and the
    installed CuTe DSL can run on the part (version floor + arch target, ``cudnn.frost.cutedsl_runtime_available``).
    Elsewhere the row is declined for the device, not for the modifier, and nothing about the TE shape is asserted."""
    import cudnn.frost

    cc = facts.device_cc
    if cc is None:
        return False
    sm = cc[0] * 10 + cc[1]
    return spec.capabilities.sm_lo <= sm <= spec.capabilities.sm_hi and cudnn.frost.cutedsl_runtime_available(device_cc=cc)


def _assert_every_row_declines_by_name(engines, control_facts, mod_facts):
    """Every row declines the score_mod graph; the rows that ADMIT the control graph (the same
    graph without the modifier -- on this device, with these dtypes) name score_mod as the reason."""
    assert not control_facts.has_score_mod and mod_facts.has_score_mod
    admitting = [spec for spec in engines.ENGINE_SPECS if engines.mismatch(spec.capabilities, control_facts) is None]
    for spec in engines.ENGINE_SPECS:
        why = engines.mismatch(spec.capabilities, mod_facts)
        assert why is not None, (spec.name, "a FROST row admitted a score_mod graph")
        if spec in admitting:
            assert "score_mod" in why, (spec.name, why)
    return admitting


@pytest.mark.parametrize("layout", ["sbhd", "bshd"])
@pytest.mark.parametrize("score_mod", [_identity_mod, _post_scale_bias], ids=["identity", "post_scale_bias"])
def test_forward_score_mod_is_a_fact_every_row_declines(layout, score_mod):
    from cudnn.sdpa.fwd import engines

    control = _facts(_fwd_graph(layout, None))
    facts = _facts(_fwd_graph(layout, score_mod))
    assert facts.has_score_mod, "sdpa(score_mod=...) must be recorded as a score_mod graph"
    admitting = _assert_every_row_declines_by_name(engines, control, facts)
    sm100_row = next(spec for spec in engines.ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm100")
    if _row_serves_this_device(sm100_row, facts):
        # The TE shape (d64, S_kv 16) is served by the SM100 prefill row since 1.31 -- the row that dropped the
        # modifier -- so where that row can run, the by-name check above must not have been vacuous.
        assert sm100_row in admitting, [spec.name for spec in admitting]


@pytest.mark.parametrize(
    "score_mod, score_mod_bprop",
    [(_post_scale_bias, _identity_mod), (_post_scale_bias, None), (None, _identity_mod)],
    ids=["both", "score_mod_only", "bprop_only"],
)
def test_backward_score_mod_is_a_fact_every_row_declines(score_mod, score_mod_bprop):
    from cudnn.sdpa.bwd import engines

    control = _facts(_bwd_graph("sbhd", None, None))
    facts = _facts(_bwd_graph("sbhd", score_mod, score_mod_bprop))
    assert facts.has_score_mod, "sdpa_backward(score_mod= / score_mod_bprop=...) must be recorded as a score_mod graph"
    _assert_every_row_declines_by_name(engines, control, facts)


def test_fp8_forward_fn_is_the_same_fact():
    from cudnn.sdpa.fwd import engines

    control = _facts(_fp8_graph(None))
    facts = _facts(_fp8_graph(_identity_mod))
    assert facts.has_score_mod, "sdpa_fp8(fn=...) spells the score modifier `fn`"
    _assert_every_row_declines_by_name(engines, control, facts)
