# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent dense recipes for FP8 backward's P and dS rounding formats."""

import ast
import inspect

import pytest
import torch

from sdpa import fp8
from sdpa.fp8_ref import compute_ref_backward

pytestmark = pytest.mark.L0
FP8_TYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _problem(qkv_type, do_type, h_q, causal):
    rng = torch.Generator(device="cpu").manual_seed(955)
    b, h_kv, sq, skv, d = 1, 2, 8, 16, 16
    q = (2 * torch.randn(b, sq, h_q, d, generator=rng)).to(qkv_type)
    k = (2 * torch.randn(b, skv, h_kv, d, generator=rng)).to(qkv_type)
    v = torch.randn(b, skv, h_kv, d, generator=rng).to(qkv_type)
    do = (3 * torch.randn(b, sq, h_q, d, generator=rng)).to(do_type)
    qh = q.float().transpose(1, 2)
    kh = k.float().transpose(1, 2).repeat_interleave(h_q // h_kv, dim=1)
    vh = v.float().transpose(1, 2).repeat_interleave(h_q // h_kv, dim=1)
    scores = (qh @ kh.transpose(-1, -2)) * 0.125
    if causal:
        scores.masked_fill_(torch.arange(skv)[None, :] > torch.arange(sq)[:, None], -torch.inf)
    stats = torch.logsumexp(scores, dim=-1, keepdim=True)
    p = torch.exp(scores - stats)
    o = (p @ vh).transpose(1, 2).to(torch.float16)
    kwargs = dict(
        attn_scale=0.125,
        q_descale=1.0,
        k_descale=1.0,
        v_descale=1.0,
        s_scale=8.0,
        s_descale=0.125,
        torch_itype=qkv_type,
        o_descale=1.0,
        dO_descale=1.0,
        torch_otype=torch.float16,
        stats=stats,
        right_bound=0 if causal else None,
        dP_scale=1.0,
        quantize_grads=False,
    )

    def oracle(**overrides):
        return compute_ref_backward(q, k, v, o, do, **{**kwargs, **overrides})[:3]

    def explicit(p_type, ds_type):
        # Deliberately use full per-query-head matrices instead of the
        # oracle's blocked GEMM helpers, then reduce each KV head's GQA group.
        doh = do.float().transpose(1, 2)
        delta = (o.float().transpose(1, 2) * doh).sum(dim=-1, keepdim=True)
        ds = p * ((doh @ vh.transpose(-1, -2)) - delta) * 0.125
        ds = ds if ds_type is None else ds.to(ds_type).float()
        pq = (p * 8.0).to(p_type).float() * 0.125
        dq = ds @ kh
        dk = (ds.transpose(-1, -2) @ qh).reshape(b, h_kv, h_q // h_kv, skv, d).sum(dim=2)
        dv = (pq.transpose(-1, -2) @ doh).reshape(b, h_kv, h_q // h_kv, skv, d).sum(dim=2)
        return tuple(t.transpose(1, 2) for t in (dq, dk, dv))

    return oracle, explicit


@pytest.mark.parametrize("qkv_type", FP8_TYPES)
@pytest.mark.parametrize("do_type", FP8_TYPES)
@pytest.mark.parametrize("h_q,causal", ((2, False), (4, True)))
def test_p_and_ds_use_independent_formats(qkv_type, do_type, h_q, causal):
    oracle, explicit = _problem(qkv_type, do_type, h_q, causal)
    actual = oracle(torch_ds_type=do_type)
    expected = explicit(qkv_type, do_type)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)
    if do_type != qkv_type:
        legacy = oracle()
        assert not torch.equal(actual[0], legacy[0])
        assert not torch.equal(actual[1], legacy[1])
        torch.testing.assert_close(actual[2], legacy[2], atol=0, rtol=0)
        # Changing P to dO's format too is a different recipe and must be detected.
        assert not torch.equal(actual[2], explicit(do_type, do_type)[2])


@pytest.mark.parametrize("qkv_type", FP8_TYPES)
def test_omitted_ds_format_preserves_the_old_recipe(qkv_type):
    oracle, _ = _problem(qkv_type, qkv_type, 4, False)
    for omitted, explicit in zip(oracle(), oracle(torch_ds_type=qkv_type)):
        torch.testing.assert_close(omitted, explicit, atol=0, rtol=0)


@pytest.mark.parametrize("ds_type", FP8_TYPES)
def test_unquantized_ds_ignores_the_cast_format(ds_type):
    oracle, explicit = _problem(torch.float8_e4m3fn, torch.float8_e5m2, 4, False)
    for got, want in zip(oracle(torch_ds_type=ds_type, quantize_ds=False), explicit(torch.float8_e4m3fn, None)):
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


def test_harness_uses_the_ds_format_for_dq_dk_rounding_attribution():
    tree = ast.parse(inspect.getsource(fp8.exec_sdpa_fp8))
    checks = {}
    reference_formats = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        keywords = {kw.arg: kw.value for kw in call.keywords}
        if call.func.id == "compute_ref_backward":
            reference_formats.append(ast.unparse(keywords["torch_ds_type"]))
        if call.func.id != "assert_close_fp8_grad":
            continue
        tag = keywords.get("tag")
        if isinstance(tag, ast.Constant) and tag.value in ("dQ", "dK", "dV"):
            checks[tag.value] = ast.unparse(keywords["fp8_dtype"])
    assert checks == {"dQ": "torch_dO_type", "dK": "torch_dO_type", "dV": "torch_itype"}
    assert reference_formats == ["torch_dO_type"]
