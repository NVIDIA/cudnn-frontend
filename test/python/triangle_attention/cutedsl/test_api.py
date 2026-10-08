# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``cudnn.triangle_attention`` -- the API layer over the FROST SDPA engine row behind cuEquivariance's signature -- against the fp64
reference on the acceptance grid (S x D x B x the three key-mask forms), the layout contract (zero-copy on the token-major form, COUNTED
repacks elsewhere, the output following q), the masked-row contract (o = 0, lse = -inf), CUDA-graph capture, the execute-path contract
(no allocation, no synchronisation), autograd (forward served, backward a typed decline) and the typed declines quoting the engine; plus
the GPU-free contract tests (the lazy export in every import order, the signature pin, mask_to_kv_lengths, the reference, the declines
a declaration raises before any device work).

Tolerances are the SM107 f16 row's own (test_sdpa_bias_sm107.py), not widened for this layer: O against the fp64 oracle at
atol 5e-2 / rtol 3e-2 (an fp32 accumulation, one 16-bit rounding at the store), the LSE at 1e-4 / 1e-4 (the bias enters the fp32 score
by one FFMA of bias * (1 / scale), then the scale multiply).  The reference reads the bf16 bias the kernel reads.

The kernels target Rubin (cc 10.7); the GPU cells skip elsewhere with the reason stated.  Every launch runs under the harness's ``timeout``
(exit 124 = hang).
"""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Triangle attention tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.api_base import TensorDesc  # noqa: E402
from cudnn.triangle_attention import api as tapi  # noqa: E402
from cudnn.triangle_attention import reference as tref  # noqa: E402

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"triangle_attention is served on SM107 (Rubin) only; found {_cc()}")

CUEQ_PARAMS = ["q", "k", "v", "bias", "mask", "scale", "return_aux", "kv_lengths"]

# The SM107 f16 row's tolerances (test_sdpa_bias_sm107.py), reused -- never widened for the API layer.
_TOL_O = dict(atol=5e-2, rtol=3e-2)
_TOL_LSE = dict(atol=1e-4, rtol=1e-4)


@pytest.fixture(autouse=True)
def _frost_opt_in(monkeypatch):
    """The engine row is opt-in; per test rather than at import so the flag cannot leak into the rest of a session."""
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


# ---------------------------------------------------------------------------
# GPU-free: the export, the signature, mask_to_kv_lengths, the reference, the declines a declaration raises before any device work
# ---------------------------------------------------------------------------
def _probe(code: str) -> str:
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-1500:]
    return run.stdout


@pytest.mark.parametrize("order", ["symbol_first", "package_first", "reference_first"])
def test_lazy_export_is_callable_in_every_import_order(order):
    """The function shares its package's name: after `import cudnn.triangle_attention...` the attribute is the PACKAGE (Python binds a
    submodule on its parent, bypassing the lazy table) -- the package module is callable, so every spelling runs the function."""
    code = f"""
import cudnn
order = {order!r}
if order == "symbol_first":
    f = cudnn.triangle_attention
elif order == "package_first":
    import cudnn.triangle_attention
    f = cudnn.triangle_attention
else:
    import cudnn.triangle_attention.reference
    f = cudnn.triangle_attention
assert callable(f), type(f)
from cudnn.triangle_attention import triangle_attention as fn, mask_to_kv_lengths, TriangleAttentionFwd
assert callable(fn) and callable(mask_to_kv_lengths)
assert cudnn.mask_to_kv_lengths is mask_to_kv_lengths and cudnn.TriangleAttentionFwd is TriangleAttentionFwd
print(type(f).__name__)
"""
    out = _probe(code)
    assert out.strip() in ("function", "_CallablePackage"), out


def test_package_import_pulls_no_torch():
    out = _probe("import sys, cudnn\nimport cudnn.triangle_attention\nprint('torch' in sys.modules)")
    assert out.strip() == "False", out


def test_gated_attention_block_is_a_lazy_export_too():
    """The gated attention block's package had no entry in the lazy table; it is one now (the same table edit)."""
    out = _probe("import cudnn\nprint(callable(cudnn.gated_attention_block_forward), cudnn.GatedAttentionBlockFwd.__name__)")
    assert out.strip() == "True GatedAttentionBlockFwd", out


def test_signature_mirrors_cueq_0_12_0():
    """cuEquivariance's parameter names, order and defaults, verbatim: in 0.12.0 ``kv_lengths`` is the eighth POSITIONAL-OR-KEYWORD
    parameter (no ``*`` before it -- a caller passing it positionally runs there and must run here); the two additions are keyword-only
    and last (append-only API)."""
    sig = inspect.signature(tapi.triangle_attention)
    names = list(sig.parameters)
    assert names[: len(CUEQ_PARAMS)] == CUEQ_PARAMS
    assert names[len(CUEQ_PARAMS) :] == ["out", "stream"]
    p = sig.parameters
    assert p["mask"].default is None and p["scale"].default is None and p["return_aux"].default is False and p["kv_lengths"].default is None
    assert p["kv_lengths"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert p["out"].kind is inspect.Parameter.KEYWORD_ONLY and p["stream"].kind is inspect.Parameter.KEYWORD_ONLY
    assert tapi.CUEQ_SIGNATURE_VERSION == "0.12.0"


def test_mask_to_kv_lengths_detects_prefix_masks():
    b, n, s = 2, 3, 10
    lengths = torch.tensor([[7, 0, 10], [1, 5, 3]], dtype=torch.int32).view(b, n, 1, 1, 1)
    prefix = torch.arange(s).view(1, 1, 1, 1, s) < lengths
    got = tapi.mask_to_kv_lengths(prefix)
    assert got is not None and got.dtype == torch.int32 and tuple(got.shape) == (b, n, 1, 1, 1) and torch.equal(got, lengths)
    holed = prefix.clone()
    holed[0, 0, 0, 0, 2] = False  # a hole inside the live prefix: not a prefix mask -> the dense path
    assert tapi.mask_to_kv_lengths(holed) is None
    with pytest.raises(ValueError, match="boolean"):
        tapi.mask_to_kv_lengths(prefix.to(torch.int32))
    with pytest.raises(ValueError, match=r"\(B, N, 1, 1, S_kv\)"):
        tapi.mask_to_kv_lengths(prefix.view(b, n, s))


def test_reference_is_the_formula_and_keeps_the_empty_row_contract():
    """The fp64 reference equals a hand-written softmax on live rows; a row without a live key reads o = 0, lse = max = -inf on both mask
    forms; the GQA fold repeats the KV heads."""
    torch.manual_seed(0)
    b, n, h, s, d = 1, 2, 2, 6, 4
    q, k, v = (torch.randn(b, n, h, s, d, dtype=torch.float64) for _ in range(3))
    bias = torch.randn(1, 1, h, s, s, dtype=torch.float64)
    lengths = torch.tensor([[4, 0]], dtype=torch.int32).view(b, n, 1, 1, 1)
    o, lse, mx = tref.triangle_attention_reference(q, k, v, bias, kv_lengths=lengths, scale=0.5)
    scores = torch.einsum("bnhqd,bnhkd->bnhqk", q, k) * 0.5 + bias
    live = scores[0, 0, :, :, :4]
    assert torch.allclose(o[0, 0], torch.softmax(live, -1) @ v[0, 0, :, :4], atol=1e-12)
    assert torch.allclose(lse[0, 0], torch.logsumexp(live, -1), atol=1e-12) and torch.allclose(mx[0, 0], live.amax(-1), atol=1e-12)
    assert (o[0, 1] == 0).all() and torch.isneginf(lse[0, 1]).all() and torch.isneginf(mx[0, 1]).all()
    mask = torch.arange(s).view(1, 1, 1, 1, s) < lengths
    o2, lse2, mx2 = tref.triangle_attention_reference(q, k, v, bias, mask=mask, scale=0.5)
    assert torch.equal(o, o2) and torch.equal(lse, lse2) and torch.equal(mx, mx2)
    k1, v1 = k[:, :, :1], v[:, :, :1]
    o3, _, _ = tref.triangle_attention_reference(q, k1, v1, bias, scale=0.5)
    o3_manual, _, _ = tref.triangle_attention_reference(q, k1.expand(-1, -1, h, -1, -1), v1.expand(-1, -1, h, -1, -1), bias, scale=0.5)
    assert torch.equal(o3, o3_manual)
    with pytest.raises(ValueError, match="not both"):
        tref.triangle_attention_reference(q, k, v, bias, mask=mask, kv_lengths=lengths)


def _desc(shape, stride, dtype=torch.bfloat16, device="cuda"):
    return TensorDesc(dtype=dtype, shape=shape, stride=stride, stride_order=tuple(sorted(range(len(shape)), key=lambda i: -stride[i])), device=device)


def _left_desc(b, n, h, s, d, dtype=torch.bfloat16, device="cuda"):
    """The zero-copy activation declaration: (B, N, S, H, D) storage presented as (B, N, H, S, D)."""
    return _desc((b, n, h, s, d), (n * s * h * d, s * h * d, d, h * d, 1), dtype, device)


def _head_major_desc(b, n, h, s, d, dtype=torch.bfloat16, device="cuda"):
    """cuEquivariance's contiguous (B, N, H, S, D): head-major -> not bindable without a copy."""
    return _desc((b, n, h, s, d), (n * h * s * d, h * s * d, s * d, d, 1), dtype, device)


def _bias_desc(bb, h, s_q, s_kv, dtype=torch.bfloat16, device="cuda", row=None):
    row = row or s_kv
    return _desc((bb, 1, h, s_q, s_kv), (h * s_q * row, h * s_q * row, s_q * row, row, 1), dtype, device)


@pytest.mark.parametrize(
    "kw,exc,match",
    [
        (dict(kv_lengths=True, mask=True), ValueError, "not both"),
        (dict(sample_q=_desc((4, 256, 32), (256 * 32, 32, 1))), ValueError, "rank 5"),
        (
            dict(sample_bias=_desc((1, 1, 4, 256, 128), (4 * 256 * 128, 4 * 256 * 128, 256 * 128, 128, 1))),
            ValueError,
            r"bias must be \(B\|1, 1, H\|1, S_q, S_kv\)",
        ),
        (dict(sample_k=_left_desc(1, 256, 4, 256, 64)), ValueError, "head dim"),
        (dict(sample_k=_left_desc(1, 256, 3, 256, 32), sample_v=_left_desc(1, 256, 3, 256, 32)), ValueError, "divide"),
        (dict(scale=0.0), ValueError, "positive"),
    ],
)
def test_declaration_declines_malformed_inputs_typed(kw, exc, match):
    args = dict(
        sample_q=_left_desc(1, 256, 4, 256, 32),
        sample_k=_left_desc(1, 256, 4, 256, 32),
        sample_v=_left_desc(1, 256, 4, 256, 32),
        sample_bias=_bias_desc(1, 4, 256, 256),
    )
    args.update(kw)
    with pytest.raises(exc, match=match):
        tapi.TriangleAttentionFwd(**args)


@pytest.mark.parametrize(
    "kw,exc,match",
    [
        (dict(dtype=torch.float8_e4m3fn), NotImplementedError, "bf16 and fp16"),
        (dict(bias_dtype=torch.float32), NotImplementedError, "the activations' dtype"),
        (dict(q=_head_major_desc(1, 256, 4, 256, 32)), NotImplementedError, "does not bind"),
        (dict(bias=_bias_desc(1, 4, 217, 217), s=217), NotImplementedError, "16-byte"),
        (dict(bias=_bias_desc(1, 4, 256, 256, row=257)), NotImplementedError, "16-byte"),
    ],
)
def test_check_support_declines_unserved_declarations_before_any_device_work(kw, exc, match):
    """The envelope's typed declines fire on the declaration alone: the dtype, the fp32 bias (the function casts it, the class API does
    not), a head-major activation layout, a compact bias at S_kv % 8 != 0 and an odd bias row stride (the TMA 16-byte rule)."""
    s = kw.pop("s", 256)
    dtype = kw.pop("dtype", torch.bfloat16)
    bias_dtype = kw.pop("bias_dtype", dtype)
    q = kw.get("q", _left_desc(1, 256, 4, s, 32, dtype))
    plan = tapi.TriangleAttentionFwd(
        q, _left_desc(1, 256, 4, s, 32, dtype), _left_desc(1, 256, 4, s, 32, dtype), kw.get("bias", _bias_desc(1, 4, s, s, bias_dtype))
    )
    with pytest.raises(exc, match=match):
        plan.check_support()


@requires_cuda
def test_check_support_declines_a_foreign_arch_typed():
    if _cc() == _SM107:
        pytest.skip("this device IS the served arch")
    plan = tapi.TriangleAttentionFwd(_left_desc(1, 64, 4, 64, 32), _left_desc(1, 64, 4, 64, 32), _left_desc(1, 64, 4, 64, 32), _bias_desc(1, 4, 64, 64))
    with pytest.raises(NotImplementedError, match="compute capability 10.7"):
        plan.check_support()


@requires_cuda
def test_execute_declines_tensors_on_another_device_than_the_plan():
    """A plan declared on cuda:1 handed cuda:0 tensors: every shape agrees, but the tensors are not on the PLAN's device -- its kernel is
    loaded on cuda:1.  A ValueError before any launch, on ONE GPU (the declaration never touches cuda:1 until compile)."""
    b, n, h, s, d = 1, 64, 4, 64, 32
    plan = tapi.TriangleAttentionFwd(
        _left_desc(b, n, h, s, d, device="cuda:1"),
        _left_desc(b, n, h, s, d, device="cuda:1"),
        _left_desc(b, n, h, s, d, device="cuda:1"),
        _bias_desc(1, h, s, s, device="cuda:1"),
    )
    assert plan.device_index == 1
    q = torch.empty(b, n, s, h, d, device="cuda:0", dtype=torch.bfloat16).permute(0, 1, 3, 2, 4)
    bias = torch.empty(1, 1, h, s, s, device="cuda:0", dtype=torch.bfloat16)
    o = torch.empty_like(q)
    with pytest.raises(ValueError, match="declared on cuda:1"):
        plan._check_execute_args(q, q, q, bias, None, o, None, None, None, None)
    plan0 = tapi.TriangleAttentionFwd(
        _left_desc(b, n, h, s, d, device="cuda:0"),
        _left_desc(b, n, h, s, d, device="cuda:0"),
        _left_desc(b, n, h, s, d, device="cuda:0"),
        _bias_desc(1, h, s, s, device="cuda:0"),
    )
    with pytest.raises(ValueError, match="must not alias"):
        plan0._check_execute_args(q, q, q, bias, None, q, None, None, None, None)  # the same tensors on the plan's device pass every layout check
    plan0._check_execute_args(q, q, q, bias, None, o, None, None, None, None)


# ---------------------------------------------------------------------------
# Rubin: the acceptance grid, the layouts, the masked rows, CUDA graphs, autograd, the live declines
# ---------------------------------------------------------------------------
def _act(b, n, h, s, d, *, dtype, gen, layout="left"):
    if layout == "left":  # (B, N, S, H, D) storage presented as (B, N, H, S, D): the zero-copy form
        return torch.randn(b, n, s, h, d, generator=gen, device="cuda", dtype=dtype).permute(0, 1, 3, 2, 4)
    return torch.randn(b, n, h, s, d, generator=gen, device="cuda", dtype=dtype)  # cuEquivariance-contiguous (head-major)


def _inputs(s, d, *, b=1, n=None, h=4, h_kv=None, dtype=torch.bfloat16, seed=0, layout="left", bias_form="h", bias_dtype=None, s_q=None):
    n = n or s
    s_q = s_q or s
    h_kv = h_kv or h
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = _act(b, n, h, s_q, d, dtype=dtype, gen=gen, layout=layout)
    k = _act(b, n, h_kv, s, d, dtype=dtype, gen=gen, layout=layout)
    v = _act(b, n, h_kv, s, d, dtype=dtype, gen=gen, layout=layout)
    bb = b if bias_form.startswith("b") else 1
    hh = h if "h" in bias_form else 1
    bias = torch.randn(bb, 1, hh, s_q, s, generator=gen, device="cuda", dtype=torch.float32) * 2.0
    bias = bias.to(bias_dtype or dtype)
    return q, k, v, bias, gen


def _lengths(b, n, s, gen, *, empty_rows=()):
    lengths = torch.randint(1, s + 1, (b, n, 1, 1, 1), generator=gen, device="cuda", dtype=torch.int32)
    for bi, ni in empty_rows:
        lengths[bi, ni] = 0
    return lengths


def _dense_mask(b, n, s, gen, *, empty_rows=()):
    mask = torch.rand(b, n, 1, 1, s, generator=gen, device="cuda") < 0.7
    mask[..., 0] = True  # at least one live key per row
    for bi, ni in empty_rows:
        mask[bi, ni] = False
    return mask


def _reference(q, k, v, bias, *, mask=None, kv_lengths=None, scale=None):
    """Per outer entry to bound the fp64 scores' memory."""
    outs = []
    for bi in range(q.shape[0]):
        bb = bias[bi : bi + 1] if bias.shape[0] > 1 else bias
        outs.append(
            tref.triangle_attention_reference(
                q[bi : bi + 1],
                k[bi : bi + 1],
                v[bi : bi + 1],
                bb,
                mask=None if mask is None else mask[bi : bi + 1],
                scale=scale,
                kv_lengths=None if kv_lengths is None else kv_lengths[bi : bi + 1],
            )
        )
    return tuple(torch.cat([o[i] for o in outs], dim=0) for i in range(3))


def _check(o, ref, lse=None):
    """Keyless rows EXACTLY 0 / -inf (never NaN, never a residue); live rows against the fp64 oracle at the row's tolerances."""
    o_ref, lse_ref, _ = ref
    empty = torch.isneginf(lse_ref)
    assert torch.isfinite(o).all(), "o has a non-finite value"
    assert (o[empty] == 0).all(), "a keyless row's o must be exactly 0"
    torch.testing.assert_close(o[~empty].double(), o_ref[~empty], **_TOL_O)
    if lse is not None:
        assert torch.isneginf(lse[empty]).all(), "a keyless row's lse must be exactly -inf"
        torch.testing.assert_close(lse[~empty].double(), lse_ref[~empty], **_TOL_LSE)


def _no_copies():
    c = tapi.copy_counters()
    return c.get("fallback_copies", 0) == 0, c


@requires_rubin
@pytest.mark.parametrize("form", ["none", "kv_lengths", "mask"])
@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("s", [64, 128, 217, 256, 384, 512])
def test_function_matches_the_fp64_reference(s, d, b, form):
    """The drop-in on the zero-copy layout over the acceptance grid: S (one tile, the budget edges, a tail at 217 whose bias is padded,
    the streamed ring above the resident budget), D 32 / 64, B 1 / 2 (a per-entry bias at B = 2: one launch per entry), the three key-mask
    forms with empty rows on both mask forms.  No layout copy except the documented bias pad at S % 8 != 0."""
    if b == 2 and s == 512:
        pytest.skip("B = 2 at S = 512 is the same cells with twice the fp64 oracle memory; B = 2 is covered to S = 384")
    q, k, v, bias, gen = _inputs(s, d, b=b, bias_form="bh" if b > 1 else "h", seed=s * 7 + d)
    kw = {}
    if form == "kv_lengths":
        kw["kv_lengths"] = _lengths(b, s, s, gen, empty_rows=((0, 1),))
    elif form == "mask":
        kw["mask"] = _dense_mask(b, s, s, gen, empty_rows=((0, 3),))
    tapi.reset_copy_counters()
    o = tapi.triangle_attention(q, k, v, bias, **kw)
    torch.cuda.synchronize()
    assert tuple(o.shape) == tuple(q.shape) and o.dtype == q.dtype and tuple(o.stride()) == tuple(q.stride()), (o.stride(), q.stride())
    clean, counts = _no_copies()
    if s % 8:
        assert counts == {"bias_repack": 1, "fallback_copies": 1}, counts  # the documented pad of a compact bias at S_kv % 8 != 0
    else:
        assert clean, counts
    _check(o, _reference(q, k, v, bias, **kw))
    o2 = tapi.triangle_attention(q, k, v, bias, **kw)
    torch.cuda.synchronize()
    assert torch.equal(o, o2), "two launches differ"


@requires_rubin
@pytest.mark.parametrize("form", ["none", "kv_lengths", "mask"])
@pytest.mark.parametrize("d", [24, 48])
def test_head_dims_below_the_tile_are_served_by_zero_fill(d, form):
    s = 256
    q, k, v, bias, gen = _inputs(s, d, seed=d)
    kw = {"kv_lengths": _lengths(1, s, s, gen)} if form == "kv_lengths" else ({"mask": _dense_mask(1, s, s, gen)} if form == "mask" else {})
    o = tapi.triangle_attention(q, k, v, bias, **kw)
    _check(o, _reference(q, k, v, bias, **kw))


@requires_rubin
def test_fp16_and_gqa_and_a_head_broadcast_bias_and_a_scale():
    s = 256
    q, k, v, bias, gen = _inputs(s, 32, dtype=torch.float16, seed=5)
    o = tapi.triangle_attention(q, k, v, bias, scale=0.3)
    _check(o, _reference(q, k, v, bias, scale=0.3))
    q, k, v, bias, gen = _inputs(s, 32, h=4, h_kv=2, bias_form="1", seed=6)
    mask = _dense_mask(1, s, s, gen)
    o = tapi.triangle_attention(q, k, v, bias, mask=mask)
    _check(o, _reference(q, k, v, bias, mask=mask))


@requires_rubin
def test_s_q_and_s_kv_may_differ():
    q, k, v, bias, gen = _inputs(384, 32, s_q=256, seed=8)  # N = S_kv = 384 rows, each attending 256 queries over 384 keys
    lengths = _lengths(1, 384, 384, gen)
    o = tapi.triangle_attention(q, k, v, bias, kv_lengths=lengths)
    _check(o, _reference(q, k, v, bias, kv_lengths=lengths))


@requires_rubin
def test_kv_lengths_and_the_prefix_mask_agree_bitwise_including_empty_rows():
    """mask_to_kv_lengths on a prefix mask hands the padding path the same rows; both forms yield the same o bitwise and the same
    empty-row contract (o = 0, lse = -inf) -- on this provider the two conventions agree."""
    s = 256
    q, k, v, bias, gen = _inputs(s, 32, seed=9)
    lengths = _lengths(1, s, s, gen, empty_rows=((0, 0), (0, 17)))
    mask = torch.arange(s, device="cuda").view(1, 1, 1, 1, s) < lengths
    got = tapi.mask_to_kv_lengths(mask)
    assert torch.equal(got, lengths)
    o_len = tapi.triangle_attention(q, k, v, bias, kv_lengths=got)
    o_mask = tapi.triangle_attention(q, k, v, bias, mask=mask)
    torch.cuda.synchronize()
    assert torch.equal(o_len, o_mask)
    assert (o_len[0, 0] == 0).all() and (o_len[0, 17] == 0).all()
    _check(o_len, _reference(q, k, v, bias, kv_lengths=lengths))


@requires_rubin
def test_class_api_lse_and_the_fully_masked_row_contract():
    """The prepared API with lse=True: lse against the oracle; a row with no live key (a zero length, an all-False mask row) reads o = 0
    and lse = -inf exactly -- the documented masked-row convention of this provider."""
    s = 128
    q, k, v, bias, gen = _inputs(s, 32, b=2, bias_form="bh", seed=10)
    lengths = _lengths(2, s, s, gen, empty_rows=((1, 5),))
    plan = tapi.TriangleAttentionFwd(q, k, v, bias, kv_lengths=True, lse=True)
    assert plan.check_support() and plan.launches_per_call == 2
    plan.compile()
    assert plan.plan_name.startswith(tapi.ENGINE_ROW), plan.plan_name
    o = tapi._bshd_empty(*q.shape, q.dtype, q.device)
    lse = torch.full((2, s, 4, s), float("nan"), device="cuda", dtype=torch.float32)
    o.fill_(float("nan"))
    ws = torch.empty(max(plan.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    plan.execute(q, k, v, bias, lengths, o, lse, None, ws if plan.get_workspace_size() else None)
    torch.cuda.synchronize()
    ref = _reference(q, k, v, bias, kv_lengths=lengths)
    _check(o, ref, lse)
    assert (o[1, 5] == 0).all() and torch.isneginf(lse[1, 5]).all()
    mask = _dense_mask(2, s, s, gen, empty_rows=((0, 2),))
    plan_m = tapi.TriangleAttentionFwd(q, k, v, bias, mask=True, lse=True)
    plan_m.check_support()
    plan_m.compile()
    o.fill_(float("nan"))
    lse.fill_(float("nan"))
    plan_m.execute(q, k, v, bias, None, o, lse, None, None, mask=mask)
    torch.cuda.synchronize()
    _check(o, _reference(q, k, v, bias, mask=mask), lse)
    assert (o[0, 2] == 0).all() and torch.isneginf(lse[0, 2]).all()


@requires_rubin
def test_per_entry_bias_at_b2_equals_the_two_single_entry_calls_bitwise():
    s = 128
    q, k, v, bias, gen = _inputs(s, 32, b=2, bias_form="bh", seed=12)
    mask = _dense_mask(2, s, s, gen)
    o = tapi.triangle_attention(q, k, v, bias, mask=mask)
    o0 = tapi.triangle_attention(q[:1], k[:1], v[:1], bias[:1], mask=mask[:1])
    o1 = tapi.triangle_attention(q[1:], k[1:], v[1:], bias[1:], mask=mask[1:])
    torch.cuda.synchronize()
    assert torch.equal(o[:1], o0) and torch.equal(o[1:], o1)
    # a bias shared by both entries (B_bias = 1) is the same math as the per-entry bias repeated
    shared = tapi.triangle_attention(q, k, v, bias[:1], mask=mask)
    rep = tapi.triangle_attention(q, k, v, bias[:1].expand(2, -1, -1, -1, -1).contiguous(), mask=mask)
    torch.cuda.synchronize()
    assert torch.equal(shared, rep)


@requires_rubin
def test_single_entry_slices_of_a_b2_batch_at_a_tail_s_bind_on_both_surfaces():
    """B = 2 at S = N = 217, H = 2: the per-entry slices ``mask[1:]`` (base 1 byte past a 16-byte boundary), ``kv_lengths[1:]`` (4 past)
    and ``lse[1:]`` (8 past) are valid operands -- the kernel reads the mask one byte per lane and the lengths / lse as words -- so both
    surfaces bind them as they are (no decline, no counted copy) and the slice call equals the batched call bitwise; ``kv_lengths`` is
    also passed POSITIONALLY (the eighth parameter), as cuEquivariance 0.12.0 admits."""
    s = 217
    q, k, v, bias, gen = _inputs(s, 32, b=2, h=2, bias_form="bh", seed=25)
    bias_p = torch.empty(2, 1, 2, s, 224, device="cuda", dtype=bias.dtype)[..., :s].copy_(bias)  # the caller's own padded row: zero-copy
    lengths = _lengths(2, s, s, gen, empty_rows=((1, 3),))
    mask = _dense_mask(2, s, s, gen, empty_rows=((1, 9),))
    assert mask[1:].data_ptr() % 16 == 1 and lengths[1:].data_ptr() % 16 == 4, (mask[1:].data_ptr() % 16, lengths[1:].data_ptr() % 16)
    tapi.reset_copy_counters()
    o_len = tapi.triangle_attention(q, k, v, bias_p, None, None, False, lengths)  # kv_lengths positional, as in 0.12.0
    o_len1 = tapi.triangle_attention(q[1:], k[1:], v[1:], bias_p[1:], kv_lengths=lengths[1:])
    o_msk = tapi.triangle_attention(q, k, v, bias_p, mask=mask)
    o_msk1 = tapi.triangle_attention(q[1:], k[1:], v[1:], bias_p[1:], mask[1:])
    torch.cuda.synchronize()
    assert tapi.copy_counters() == {}, tapi.copy_counters()
    assert torch.equal(o_len[1:], o_len1) and torch.equal(o_msk[1:], o_msk1)
    assert (o_len1[0, 3] == 0).all() and (o_msk1[0, 9] == 0).all()
    _check(o_len1, _reference(q[1:], k[1:], v[1:], bias_p[1:], kv_lengths=lengths[1:]))
    # the prepared API on the same slices, with an lse slice (B, N, H, S_q)[1:] whose base is 8 bytes past a 16-byte boundary
    lse = torch.full((2, s, 2, s), float("nan"), device="cuda", dtype=torch.float32)
    assert lse[1:].data_ptr() % 16 == 8
    o_c = tapi._bshd_empty(1, s, 2, s, 32, q.dtype, q.device)
    plan = tapi.TriangleAttentionFwd(q[1:], k[1:], v[1:], bias_p[1:], kv_lengths=True, lse=True)
    plan.check_support()
    plan.compile()
    ws = torch.empty(max(plan.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    plan.execute(q[1:], k[1:], v[1:], bias_p[1:], lengths[1:], o_c, lse[1:], None, ws if plan.get_workspace_size() else None)
    torch.cuda.synchronize()
    assert torch.equal(o_c, o_len1) and torch.isneginf(lse[1, 3]).all() and torch.isnan(lse[0]).all()
    _check(o_c, _reference(q[1:], k[1:], v[1:], bias_p[1:], kv_lengths=lengths[1:]), lse[1:])
    plan_m = tapi.TriangleAttentionFwd(q[1:], k[1:], v[1:], bias_p[1:], mask=True)
    plan_m.check_support()
    plan_m.compile()
    o_c.fill_(float("nan"))
    ws = torch.empty(max(plan_m.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    plan_m.execute(q[1:], k[1:], v[1:], bias_p[1:], None, o_c, None, None, ws if plan_m.get_workspace_size() else None, mask=mask[1:])
    torch.cuda.synchronize()
    assert torch.equal(o_c, o_msk1)


@requires_cuda
def test_execute_checks_apply_the_binders_alignment_rule_per_operand():
    """The argument checks hold each operand to the base alignment ITS reader needs, never a blanket 16 bytes: a per-entry slice of the
    byte mask (any base), of the int32 lengths and of the fp32 lse (4 bytes) passes; a TMA-read activation 8 bytes past a 16-byte
    boundary is declined with its own reason.  Checks only -- no plan is built, so this runs on any CUDA device."""
    b, n, h, s, d = 2, 217, 2, 217, 32
    q, k, v, o = (tapi._bshd_empty(b, n, h, s, d, torch.bfloat16, "cuda") for _ in range(4))
    bias = torch.empty(b, 1, h, s, 224, device="cuda", dtype=torch.bfloat16)[..., :s]
    lengths = torch.full((b, n, 1, 1, 1), s, device="cuda", dtype=torch.int32)
    mask = torch.ones(b, n, 1, 1, s, device="cuda", dtype=torch.bool)
    lse = torch.empty(b, n, h, s, device="cuda", dtype=torch.float32)
    assert mask[1:].data_ptr() % 16 == 1 and lengths[1:].data_ptr() % 16 == 4 and lse[1:].data_ptr() % 16 == 8
    plan_l = tapi.TriangleAttentionFwd(q[1:], k[1:], v[1:], bias[1:], kv_lengths=True, lse=True)
    plan_l._check_execute_args(q[1:], k[1:], v[1:], bias[1:], lengths[1:], o[1:], lse[1:], None, None, None)
    plan_m = tapi.TriangleAttentionFwd(q[1:], k[1:], v[1:], bias[1:], mask=True)
    plan_m._check_execute_args(q[1:], k[1:], v[1:], bias[1:], None, o[1:], None, None, None, mask[1:])
    raw = torch.empty(n * s * h * d + 4, device="cuda", dtype=torch.bfloat16)
    q_off = raw[4:].view(1, n, s, h, d).permute(0, 1, 3, 2, 4)  # 8 bytes past a 16-byte boundary: the TMA rule applies to q
    assert q_off.data_ptr() % 16 == 8
    with pytest.raises(ValueError, match="q must be 16-byte aligned"):
        plan_m._check_execute_args(q_off, k[1:], v[1:], bias[1:], None, o[1:], None, None, None, mask[1:])


@requires_rubin
def test_head_major_inputs_are_repacked_with_counted_copies_and_the_output_follows_q():
    """cuEquivariance's contiguous (B, N, H, S, D): q, k, v are repacked (3 counted copies, one warning per layout), the output comes
    back in q's own contiguous layout (a 4th counted copy) and equals the zero-copy call bitwise."""
    s = 256
    q, k, v, bias, gen = _inputs(s, 32, layout="head_major", seed=13)
    assert q.is_contiguous()
    tapi.reset_copy_counters()
    tapi._WARNED.clear()
    with pytest.warns(RuntimeWarning, match="not in the \\(B, N, S, H, D\\) memory form") as rec:
        o = tapi.triangle_attention(q, k, v, bias)
        o_again = tapi.triangle_attention(q, k, v, bias)
    torch.cuda.synchronize()
    assert o.is_contiguous() and tuple(o.shape) == tuple(q.shape)
    counts = tapi.copy_counters()
    assert counts == {"q_repack": 2, "k_repack": 2, "v_repack": 2, "o_repack": 2, "fallback_copies": 8}, counts
    assert len([w for w in rec if "repack" in str(w.message)]) == 4, [str(w.message) for w in rec]  # one warning per (kind, layout), not per call
    q_l = torch.empty(*q.shape[:2], s, 4, 32, device="cuda", dtype=q.dtype).permute(0, 1, 3, 2, 4).copy_(q)
    k_l = torch.empty(*q.shape[:2], s, 4, 32, device="cuda", dtype=q.dtype).permute(0, 1, 3, 2, 4).copy_(k)
    v_l = torch.empty(*q.shape[:2], s, 4, 32, device="cuda", dtype=q.dtype).permute(0, 1, 3, 2, 4).copy_(v)
    tapi.reset_copy_counters()
    o_l = tapi.triangle_attention(q_l, k_l, v_l, bias)
    torch.cuda.synchronize()
    assert tapi.copy_counters() == {}
    assert torch.equal(o, o_l) and torch.equal(o_again, o_l)
    _check(o, _reference(q, k, v, bias))


@requires_rubin
def test_bias_layouts_are_staged_with_counted_copies():
    """A bias projected as (B, S_q, S_kv, H) and permuted (key dim not contiguous), an fp32 bias (cast once per call) and the compact
    bias at S_kv % 8 != 0 (padded to a multiple of 8): each ONE counted copy per call, bitwise the head-major bf16 bias where the values
    agree, within the oracle tolerance everywhere."""
    s = 256
    q, k, v, bias, gen = _inputs(s, 32, seed=14)
    base = tapi.triangle_attention(q, k, v, bias)
    permuted = bias.permute(0, 1, 3, 4, 2).contiguous().permute(0, 1, 4, 2, 3)  # (B, 1, S_q, S_kv, H) storage presented as (B, 1, H, S_q, S_kv)
    assert torch.equal(permuted, bias) and permuted.stride(-1) != 1
    tapi.reset_copy_counters()
    o = tapi.triangle_attention(q, k, v, permuted)
    torch.cuda.synchronize()
    assert tapi.copy_counters() == {"bias_repack": 1, "fallback_copies": 1}
    assert torch.equal(o, base)
    bias32 = bias.float()
    tapi.reset_copy_counters()
    o32 = tapi.triangle_attention(q, k, v, bias32)
    torch.cuda.synchronize()
    assert tapi.copy_counters() == {"bias_cast": 1, "fallback_copies": 1}
    assert torch.equal(o32, base)  # the cast lands on the same bf16 values
    q7, k7, v7, bias7, gen = _inputs(217, 32, seed=15)
    tapi.reset_copy_counters()
    o7 = tapi.triangle_attention(q7, k7, v7, bias7)
    torch.cuda.synchronize()
    assert tapi.copy_counters() == {"bias_repack": 1, "fallback_copies": 1}
    _check(o7, _reference(q7, k7, v7, bias7))
    padded = torch.empty(1, 1, 4, 217, 224, device="cuda", dtype=bias7.dtype)[..., :217].copy_(bias7)  # the caller's own padded row: zero-copy
    tapi.reset_copy_counters()
    o7p = tapi.triangle_attention(q7, k7, v7, padded)
    torch.cuda.synchronize()
    assert tapi.copy_counters() == {} and torch.equal(o7, o7p)


@requires_rubin
def test_out_and_stream_arguments():
    s = 128
    q, k, v, bias, gen = _inputs(s, 32, seed=16)
    base = tapi.triangle_attention(q, k, v, bias)
    out = tapi._bshd_empty(*q.shape, q.dtype, q.device)
    tapi.reset_copy_counters()
    got = tapi.triangle_attention(q, k, v, bias, out=out)
    torch.cuda.synchronize()
    assert got.data_ptr() == out.data_ptr() and torch.equal(out, base) and tapi.copy_counters() == {}
    out_c = torch.empty_like(q, memory_format=torch.contiguous_format)
    got = tapi.triangle_attention(q, k, v, bias, out=out_c)
    torch.cuda.synchronize()
    assert got.data_ptr() == out_c.data_ptr() and torch.equal(out_c, base) and tapi.copy_counters() == {"out_copy": 1, "fallback_copies": 1}
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        got = tapi.triangle_attention(q, k, v, bias, stream=side.cuda_stream)
    side.synchronize()
    assert torch.equal(got, base)


def test_workspace_is_sized_and_carved_by_the_sdpa_adapters_recipe():
    """GPU-free: the kv_lengths form's Q-side lengths are the second chunk of the SDPA adapters' shared carve (`ws_align` 128-byte chunks,
    `WorkspaceCarver`), so they start at `ws_align(plan bytes)`; the other forms need exactly the plan's bytes; a missing or short buffer
    raises the shared carver's own error (never a local variant)."""
    from cudnn.sdpa.fwd.api_dsl import ws_align

    wb = tapi.workspace_bytes
    assert wb(0, 0) == 0 and wb(1024, 0) == 1024
    assert wb(0, 217) == ws_align(4 * 217) == 896 and wb(100, 217) == 128 + 896 and wb(256, 5) == 256 + 128 and wb(257, 1) == 384 + 128
    buf = torch.empty(wb(100, 217), dtype=torch.uint8)  # a CPU buffer: the carve is a pure view arithmetic
    plan_ws, q_lens = tapi.carve_workspace(buf, 100, 217, "TriangleAttentionFwd")
    assert plan_ws.dtype == torch.uint8 and plan_ws.numel() == 100 and plan_ws.data_ptr() == buf.data_ptr()
    assert q_lens.dtype == torch.int32 and tuple(q_lens.shape) == (217,) and q_lens.data_ptr() - buf.data_ptr() == 128
    assert tapi.carve_workspace(buf[:896], 0, 217, "TriangleAttentionFwd")[1].data_ptr() == buf.data_ptr()  # no plan bytes: the lengths lead
    with pytest.raises(ValueError, match=r"TriangleAttentionFwd requires a 1024-byte workspace but execute\(\) received none"):
        tapi.carve_workspace(None, 100, 217, "TriangleAttentionFwd")
    with pytest.raises(ValueError, match="requires a 1024-byte workspace; the provided buffer has only 1000 bytes"):
        tapi.carve_workspace(buf[:1000], 100, 217, "TriangleAttentionFwd")


@requires_rubin
@pytest.mark.parametrize("form", ["mask", "kv_lengths"])
def test_execute_allocates_nothing_and_never_synchronizes(form):
    """Both key-mask forms: a warm execute allocates no device memory and issues no host synchronisation -- on the kv_lengths form the
    Q-side lengths the graph's padding contract requires are written into the CALLER's workspace (an async memset), never allocated."""
    s = 128
    q, k, v, bias, gen = _inputs(s, 32, seed=17)
    mask = _dense_mask(1, s, s, gen) if form == "mask" else None
    lengths = _lengths(1, s, s, gen, empty_rows=((0, 4),)) if form == "kv_lengths" else None
    plan = tapi.TriangleAttentionFwd(q, k, v, bias, mask=(form == "mask"), kv_lengths=(form == "kv_lengths"))
    plan.check_support()
    plan.compile()
    if form == "kv_lengths":
        assert plan.get_workspace_size() == tapi.workspace_bytes(plan._plan_workspace_bytes, s) >= 4 * s
    o = tapi._bshd_empty(*q.shape, q.dtype, q.device)
    ws = torch.empty(max(plan.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    plan.execute(q, k, v, bias, lengths, o, None, None, ws if plan.get_workspace_size() else None, mask=mask)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.set_sync_debug_mode("error")
    try:
        plan.execute(q, k, v, bias, lengths, o, None, None, ws if plan.get_workspace_size() else None, mask=mask)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.cuda.memory_allocated() == before
    torch.cuda.synchronize()
    _check(o, _reference(q, k, v, bias, mask=mask, kv_lengths=lengths))
    if form == "kv_lengths":
        _plan_ws, q_lens = tapi.carve_workspace(ws, plan._plan_workspace_bytes, s, "TriangleAttentionFwd")  # the same carve execute ran
        assert (q_lens == s).all(), "the Q-side lengths chunk must hold S_q on every entry"
        assert (o[0, 4] == 0).all()


@requires_rubin
def test_cuda_graph_capture_replays_bitwise_and_follows_new_data():
    s = 256
    q, k, v, bias, gen = _inputs(s, 32, seed=18)
    mask = _dense_mask(1, s, s, gen)
    eager = tapi.triangle_attention(q, k, v, bias, mask=mask)  # warm: the plan is cached
    torch.cuda.synchronize()
    out = tapi._bshd_empty(*q.shape, q.dtype, q.device)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        tapi.triangle_attention(q, k, v, bias, mask=mask, out=out)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        tapi.triangle_attention(q, k, v, bias, mask=mask, out=out)
    out.fill_(float("nan"))
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    q_new, _, _, _, _ = _inputs(s, 32, seed=19)
    q.copy_(q_new)
    eager_new = tapi.triangle_attention(q, k, v, bias, mask=mask)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager_new)
    plan = tapi.TriangleAttentionFwd(q, k, v, bias, mask=True)
    plan.check_support()
    plan.compile()
    out2 = tapi._bshd_empty(*q.shape, q.dtype, q.device)
    ws = torch.empty(max(plan.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    g2 = plan.make_cuda_graph(q, k, v, bias, None, out2, None, ws if plan.get_workspace_size() else None, mask=mask)
    out2.fill_(float("nan"))
    g2.replay()
    torch.cuda.synchronize()
    assert torch.equal(out2, eager_new)
    # the kv_lengths form: the workspace fill (the Q-side lengths) is captured with the launches and the replay follows new lengths
    lengths = _lengths(1, s, s, gen, empty_rows=((0, 3),))
    eager_len = tapi.triangle_attention(q, k, v, bias, kv_lengths=lengths)
    plan_l = tapi.TriangleAttentionFwd(q, k, v, bias, kv_lengths=True)
    plan_l.check_support()
    plan_l.compile()
    out3 = tapi._bshd_empty(*q.shape, q.dtype, q.device)
    ws_l = torch.empty(plan_l.get_workspace_size(), dtype=torch.uint8, device="cuda")
    g3 = plan_l.make_cuda_graph(q, k, v, bias, lengths, out3, None, ws_l)
    out3.fill_(float("nan"))
    ws_l.fill_(0)  # a replay must re-write the Q-side lengths, not rely on a previous call's fill
    g3.replay()
    torch.cuda.synchronize()
    assert torch.equal(out3, eager_len) and (out3[0, 3] == 0).all()
    lengths.copy_(_lengths(1, s, s, gen, empty_rows=((0, 7),)))
    eager_len2 = tapi.triangle_attention(q, k, v, bias, kv_lengths=lengths)
    g3.replay()
    torch.cuda.synchronize()
    assert torch.equal(out3, eager_len2) and (out3[0, 7] == 0).all()


@requires_rubin
def test_return_aux_is_a_typed_decline_quoting_the_engine():
    """The row max is a side output the engine row's bias arm does not store: return_aux=True is declined with the ENGINE's reason."""
    q, k, v, bias, gen = _inputs(128, 32, seed=20)
    with pytest.raises(NotImplementedError, match="score side outputs") as info:
        tapi.triangle_attention(q, k, v, bias, return_aux=True)
    assert tapi.ENGINE_ROW in str(info.value)


@requires_rubin
def test_autograd_forward_is_served_and_the_backward_is_a_typed_decline():
    s = 128
    q, k, v, bias, gen = _inputs(s, 32, seed=21)
    q.requires_grad_(True)
    bias.requires_grad_(True)
    o = tapi.triangle_attention(q, k, v, bias)
    assert o.grad_fn is not None and o.requires_grad
    _check(o.detach(), _reference(q.detach(), k, v, bias.detach()))
    with pytest.raises(NotImplementedError, match="no backward registered yet"):
        o.float().sum().backward()
    with torch.no_grad():
        o2 = tapi.triangle_attention(q, k, v, bias)
    assert o2.grad_fn is None and torch.equal(o2, o.detach())


@requires_rubin
def test_rejects_on_a_live_device_are_typed():
    s = 128
    q, k, v, bias, gen = _inputs(s, 32, seed=22)
    lengths = _lengths(1, s, s, gen)
    mask = torch.arange(s, device="cuda").view(1, 1, 1, 1, s) < lengths
    with pytest.raises(ValueError, match="not both"):
        tapi.triangle_attention(q, k, v, bias, mask=mask, kv_lengths=lengths)
    with pytest.raises(ValueError, match="rank 5"):
        tapi.triangle_attention(q[0], k[0], v[0], bias[0])
    with pytest.raises(ValueError, match="boolean"):
        tapi.triangle_attention(q, k, v, bias, mask=mask.to(torch.float32))
    with pytest.raises(ValueError, match="int32"):
        tapi.triangle_attention(q, k, v, bias, kv_lengths=lengths.to(torch.int64))
    with pytest.raises(NotImplementedError, match="bf16 and fp16"):
        tapi.triangle_attention(q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn), bias)
    q20, k20, v20, bias20, _ = _inputs(s, 24, seed=23)
    with pytest.raises(ValueError, match="multiple of 8"):  # the FE graph contract itself refuses D % 8 != 0 (a malformed sdpa node)
        tapi.triangle_attention(q20[..., :20], k20[..., :20], v20[..., :20], bias20)
    q128, k128, v128, bias128, _ = _inputs(s, 128, seed=24)
    with pytest.raises(NotImplementedError, match="flavors") as info:
        tapi.triangle_attention(q128, k128, v128, bias128)
    assert tapi.ENGINE_ROW in str(info.value)
    plan = tapi.TriangleAttentionFwd(q, k, v, bias, kv_lengths=True)
    plan.check_support()
    plan.compile()
    o = tapi._bshd_empty(*q.shape, q.dtype, q.device)
    with pytest.raises(ValueError, match="kv_lengths=True"):
        plan.execute(q, k, v, bias, None, o)
    with pytest.raises(ValueError, match="never allocates"):
        plan.execute(q, k, v, bias, lengths, None)
    with pytest.raises(ValueError, match=r"requires a \d+-byte workspace but execute\(\) received none"):  # the shared carver's error
        plan.execute(q, k, v, bias, lengths, o)
    with pytest.raises(ValueError, match="16-byte aligned"):  # the shared carver's base rule (the lengths' reader needs 4; a torch allocation is 512)
        plan.execute(q, k, v, bias, lengths, o, None, None, torch.empty(plan.get_workspace_size() + 8, dtype=torch.uint8, device="cuda")[8:])
    with pytest.raises(ValueError, match="contiguous"):  # the one check the carver leaves to the API: a strided buffer would be carved on a copy
        plan.execute(q, k, v, bias, lengths, o, None, None, torch.empty(2 * plan.get_workspace_size(), dtype=torch.uint8, device="cuda")[::2])
    with pytest.raises(ValueError, match="strides"):
        plan.execute(q.contiguous(), k, v, bias, lengths, o)
    with pytest.raises(ValueError, match="lse=False"):
        plan.execute(q, k, v, bias, lengths, o, torch.empty(1, s, 4, s, device="cuda"))
    with pytest.raises(ValueError, match="score_max"):
        plan.execute(q, k, v, bias, lengths, o, None, torch.empty(1, s, 4, s, device="cuda"))
    with pytest.raises(NotImplementedError, match="score side outputs"):
        tapi.TriangleAttentionFwd(q, k, v, bias, score_max=True).check_support()
