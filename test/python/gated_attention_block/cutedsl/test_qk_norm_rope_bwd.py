# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""B5 + B6 fused -- the norm + RoPE BACKWARD kernel of the gated attention block.

Per Q / K row: the exact RoPE adjoint ``dy*cos - rotate_half(dy*sin)`` on the
leading ``rope_dim`` columns, then the RMSNorm backward
``dx = rstd * (g*w - x_hat * mean(g*w*x_hat))`` in place into the Q / K bands
of ``dqkvg``; ``dW_norm`` as per-CTA fp32 partials plus a fixed-order reduce
launch; the V band a bit-exact copy. Plain LDG/STG + one butterfly shuffle +
a 4 KiB SMEM combine at CTA end: no tcgen05, no TMA, so like the forward twin
(``test_qk_norm_rope.py``) it runs, and is tested, anywhere CuTe DSL does.

Oracles: fp64 autograd through ``qk_norm_rope_reference(..., acc_dtype=float64)``
on the STORAGE-rounded inputs (``test/AGENTS.md``: build the reference in fp64,
never a TF32 fp32 one). Also here: the oracle's new window / bottom-right mask
arm, pinned against ``F.scaled_dot_product_attention`` with the SAME explicit
mask (a graph flag with no reference-side branch is a test defect, not evidence).
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.gated_attention_block.kernels.qk_norm_rope_bwd import (  # noqa: E402
    CTAS_PER_SM,
    DEFAULT_ROWS_PER_GROUP,
    DEFAULT_ROWS_PER_GROUP_ROPE_ONLY,
    REDUCE_LANES,
    QkNormRopeBwdRecipe,
    compile_qk_norm_rope_bwd,
    moved_bytes,
    n_ctas_for,
    run_dw_norm_reduce,
    run_qk_norm_rope_bwd,
    validate_shape,
)
from cudnn.gated_attention_block.kernels.qk_norm_rope import ACCESS_BYTES, ELEMS_PER_ACCESS  # noqa: E402
from cudnn.gated_attention_block import QKVG_TILE_ALIGN, GatedAttentionBlockGeometry  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import (  # noqa: E402
    RefGeometry,
    _key_padding_and_causal_mask,
    gated_attention_block_reference,
    make_inputs,
    qk_norm_rope_reference,
    split_qkvg,
)

pytestmark = pytest.mark.L0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# Defined PER MODULE on purpose (hoisting it into the conftest is a separate change).
_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"the 397B-geometry confirmation runs on SM107 only; found {_cc()}")

_EPS = 1e-6
_DTYPES = pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
_QK_NORM = pytest.mark.parametrize("qk_norm", [True, False], ids=["norm", "rope_only"])


@pytest.fixture(autouse=True)
def _no_tf32():
    """An fp32 torch reference is a TF32 reference on Blackwell+ unless pinned (``test/AGENTS.md``)."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _make(t, h_q, h_kv, d, rope_dim, dtype, seed=0, random_tables=False):
    """dq / dk / dv (B4's outputs), x_q / x_k (pre-norm), the two [D] weights,
    cos / sin ``[T, rope_dim]`` -- duplicated halves like ``build_rope_tables``
    unless ``random_tables`` (the adjoint pin)."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32).to(dtype)

    dq, dk, dv = rnd(t, h_q, d), rnd(t, h_kv, d), rnd(t, h_kv, d)
    xq, xk = rnd(t, h_q, d), rnd(t, h_kv, d)
    w_q, w_k = rnd(d), rnd(d)
    rd = max(rope_dim, 1)
    if random_tables:
        cos, sin = rnd(t, rd), rnd(t, rd)
    else:
        ang = torch.randn(t, rd // 2 if rope_dim else 1, generator=g, device="cuda", dtype=torch.float32)
        emb = torch.cat((ang, ang), dim=-1) if rope_dim else ang
        cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
    return dict(dq=dq, dk=dk, dv=dv, xq=xq, xk=xk, w_q=w_q, w_k=w_k, cos=cos, sin=sin)


def _rstd(x):
    """The forward's saved rstd: fp32 ``rsqrt(mean(x^2) + eps)`` per row, ``[T, H]``."""
    return torch.rsqrt(x.float().pow(2).mean(-1) + _EPS).contiguous()


def _ref(dy, x, w, cos, sin, rope_dim, qk_norm):
    """fp64 autograd of the fused forward oracle: ``(dx, dW or None)`` in fp64, ``[T, H, D]`` / ``[D]``."""
    x64 = x.double().requires_grad_(True)
    w64 = w.double().requires_grad_(True) if qk_norm else None
    y, _ = qk_norm_rope_reference(x64[None], w64, cos[None].double(), sin[None].double(), rope_dim, _EPS, qk_norm=qk_norm, acc_dtype=torch.float64)
    assert y.dtype is torch.float64, "the acc_dtype knob must keep the fp64 chain unrounded"
    grads = torch.autograd.grad(y, (x64, w64) if qk_norm else (x64,), dy[None].double())
    return grads[0], (grads[1] if qk_norm else None)


def _compile(inp, *, rope_dim, qk_norm, want_dw, has_seq_lens=False, rows_per_group=None, n_ctas_policy="sm_fill", const_head_counts=True):
    """``rows_per_group=None`` = the SHIPPED per-arm default (1 with the norm, 2 RoPE-only): every
    numerics test that does not name a value traces exactly the artifact the block backward gets."""
    return compile_qk_norm_rope_bwd(
        dtype=inp["dq"].dtype,
        h_q=int(inp["dq"].shape[1]),
        h_kv=int(inp["dk"].shape[1]),
        d=int(inp["dq"].shape[2]),
        rope_dim=rope_dim,
        eps=_EPS,
        apply_norm=qk_norm,
        want_dw=want_dw,
        has_seq_lens=has_seq_lens,
        rows_per_group=rows_per_group,
        n_ctas_policy=n_ctas_policy,
        const_head_counts=const_head_counts,
    )


def _launch(inp, r, out_q, out_k, out_v, *, qk_norm, want_dw, seq_lens=None, s=None, poison_plane=True):
    """One kernel launch (+ the reduce when ``want_dw``). Returns ``(partials, dw)``."""
    t, d = int(inp["dq"].shape[0]), int(inp["dq"].shape[2])
    nq, nk, _ = n_ctas_for(r, t)
    if want_dw:
        fill = float("nan") if poison_plane else 0.0
        pq = torch.full((nq, d), fill, device="cuda", dtype=torch.float32)
        pk = torch.full((nk, d), fill, device="cuda", dtype=torch.float32)
    else:
        pq = pk = None
    xq, xk = (inp["xq"], inp["xk"]) if qk_norm else (None, None)
    rq, rk = (_rstd(inp["xq"]), _rstd(inp["xk"])) if qk_norm else (None, None)
    wq, wk = (inp["w_q"], inp["w_k"]) if qk_norm else (None, None)
    run_qk_norm_rope_bwd(
        r, inp["dq"], inp["dk"], inp["dv"], xq, xk, rq, rk, wq, wk, inp["cos"], inp["sin"], out_q, out_k, out_v, pq, pk, seq_lens, s=s, stream=_stream()
    )
    dw = None
    if want_dw:
        dwq = torch.full((d,), float("nan"), device="cuda", dtype=torch.float32)
        dwk = torch.full((d,), float("nan"), device="cuda", dtype=torch.float32)
        run_dw_norm_reduce(r, pq, pk, dwq, dwk, stream=_stream(), t=t)
        dw = (dwq, dwk)
    torch.cuda.synchronize()
    return (pq, pk), dw


def _check_dx(got, want64, dtype):
    """The forward norm bound (test_qk_norm_rope.py:85-89), ``rtol=0, atol=8e-3`` bf16 /
    ``1e-3`` f16 against the fp64 reference ROUNDED to the io dtype -- kept as stated,
    NOT widened.

    One thing that bound cannot express: the reference is fp64 and the kernel is fp32,
    so on ~1e-4 of the elements the two land on OPPOSITE sides of an io-dtype rounding
    midpoint and the rounded values differ by exactly one ulp (0.015625 at |v| in [2, 4)
    for bf16 -- measured 1 element in 526336 at t=257, h_q=8). That is not a kernel
    error, and it is PROVED per element rather than tolerated: an element over ``atol``
    is accepted only if the kernel's value sits within half an ulp (+ fp32 slack) of the
    UNROUNDED fp64 value -- i.e. it is the other nearest representable -- and such
    elements are capped at ``max(1, 0.1 %)`` of the tensor (measured rate ~2e-6 on bf16;
    a wrong rounding mode, or a chain biased by more than ~64 fp32 ulps, flips >= 0.1 %,
    which a per-element-proven straddle rate cannot). Stated plainly: the acceptance SET
    is "the stated atol, plus PROVEN one-ulp straddles <= 0.1 %" -- the constant is
    unchanged, the criterion on the extra elements is STRICTER than the bound, and nothing
    else is admitted. Every other element must meet the bound. Precedent:
    ``assert_close_fp8_grad`` proves a flip from the reference's intermediates
    (test/AGENTS.md)."""
    atol = 8e-3 if dtype is torch.bfloat16 else 1e-3
    got64 = got.double()
    rounded = want64.to(dtype).double()
    over = (got64 - rounded).abs() > atol
    if over.any():
        tiny = torch.finfo(dtype).tiny
        ulp = torch.finfo(dtype).eps * torch.exp2(torch.floor(torch.log2(want64.abs().clamp_min(tiny))))
        straddle = (got64 - want64).abs() <= 0.5 * ulp * (1.0 + 2.0**-6)
        unproven = over & ~straddle
        if unproven.any():
            idx = unproven.nonzero()[0].tolist()
            raise AssertionError(
                f"{int(unproven.sum())} element(s) exceed atol={atol} and are NOT rounding straddles; first at {idx}: "
                f"kernel={got64[tuple(idx)].item():.6g} exact={want64[tuple(idx)].item():.6g} rounded={rounded[tuple(idx)].item():.6g}; "
                f"max |kernel - rounded| = {(got64 - rounded).abs().max().item():.6g}"
            )
        n_straddle = int(over.sum())
        cap = max(1, int(1e-3 * got.numel()))
        assert n_straddle <= cap, f"{n_straddle} one-ulp straddles in {got.numel()} elements (cap {cap}): a biased chain, not rounding noise"
    torch.testing.assert_close(torch.where(over, rounded, got64).float(), rounded.float(), rtol=0, atol=atol)


def _check_dw(got, want64):
    # fp32 accumulation of bf16-rounded products vs the fp64 sum of the same
    # products: 1e-3 relative on the vector's own scale.
    torch.testing.assert_close(got, want64.float(), rtol=1e-3, atol=1e-3 * want64.abs().max().item())


# ---------------------------------------------------------------------------
# Host-only
# ---------------------------------------------------------------------------


def test_moved_bytes_matches_the_bandwidth_budget():
    """397B: rows/token = h_q + 2*h_kv = 36; Q/K three passes + 4 B rstd, V two passes."""
    t, h_q, h_kv, d = 1, 32, 2, 256
    qk_rows = h_q + h_kv
    assert moved_bytes(t, h_q, h_kv, d, apply_norm=True) == 3 * qk_rows * d * 2 + qk_rows * 4 + 2 * h_kv * d * 2  # 54408 B/token
    assert moved_bytes(t, h_q, h_kv, d, apply_norm=False) == 2 * qk_rows * d * 2 + 2 * h_kv * d * 2  # 36 KiB/token
    assert moved_bytes(t, h_q, h_kv, d, apply_norm=False) == 36 * 1024


@pytest.mark.parametrize(
    "d, rope_dim, threads, match",
    [(250, 64, 128, "multiple of 8"), (256, 24, 128, "multiple of 16"), (256, 96, 128, "power of two"), (256, 64, 100, "multiple of the 32 lanes")],
)
def test_validate_shape_rejects(d, rope_dim, threads, match):
    with pytest.raises(ValueError, match=match):
        validate_shape(d, rope_dim, threads)


def test_compile_time_refusals():
    base = dict(dtype=torch.bfloat16, h_q=8, h_kv=2, d=256, rope_dim=64, eps=_EPS, has_seq_lens=False)
    with pytest.raises(ValueError, match="no RMSNorm.*dW"):
        compile_qk_norm_rope_bwd(**base, apply_norm=False, want_dw=True)
    with pytest.raises(ValueError, match="bf16/f16 only"):
        compile_qk_norm_rope_bwd(**{**base, "dtype": torch.float32}, apply_norm=True, want_dw=True)
    with pytest.raises(ValueError, match="n_ctas_policy"):
        compile_qk_norm_rope_bwd(**base, apply_norm=True, want_dw=True, n_ctas_policy="nonsense")
    with pytest.raises(ValueError, match="n_ctas_policy"):
        compile_qk_norm_rope_bwd(**base, apply_norm=True, want_dw=True, n_ctas_policy="fixed:0")


def test_compile_time_refuses_rows_per_group_below_one(monkeypatch):
    """``rows_per_group`` is a row count per lane group. 0 would trace lane groups that own no
    rows and record ``rows_per_cta == 0`` (``n_ctas_for`` then divides by it), a negative value
    a wrong CTA count, a non-int a float ``rows_per_cta``: each is a typed ValueError before the
    device is queried or the artifact traced. CPU-only: ``cute.compile`` is stubbed to FAIL, so
    reaching it is the failure."""
    from cudnn.gated_attention_block.kernels import qk_norm_rope_bwd as kern

    def never(*a, **k):
        raise AssertionError("cute.compile was reached for a refused rows_per_group")

    monkeypatch.setattr(kern, "compiled_cache", {})
    monkeypatch.setattr(kern, "reduce_cache", {})
    monkeypatch.setattr(kern, "current_device", lambda: 0)
    monkeypatch.setattr(kern, "multiprocessor_count", lambda dev: 100)
    monkeypatch.setattr(kern.cute, "compile", never)
    kw = dict(dtype=torch.bfloat16, h_q=8, h_kv=2, d=256, rope_dim=64, eps=_EPS, has_seq_lens=False, apply_norm=True, want_dw=True)
    for bad in (0, -1, 2.0, "2"):
        with pytest.raises(ValueError, match="rows_per_group must be an int >= 1"):
            kern.compile_qk_norm_rope_bwd(**kw, rows_per_group=bad)


def test_rows_per_group_default_resolves_per_arm(monkeypatch):
    """``rows_per_group=None`` -> 1 with the norm, 2 for the RoPE-only adjoint (each arm's
    measured optimum); an explicit value is honoured. CPU-only: ``cute.compile`` stubbed."""
    from cudnn.gated_attention_block.kernels import qk_norm_rope_bwd as kern

    monkeypatch.setattr(kern, "compiled_cache", {})
    monkeypatch.setattr(kern, "reduce_cache", {})
    monkeypatch.setattr(kern, "current_device", lambda: 0)
    monkeypatch.setattr(kern, "multiprocessor_count", lambda dev: 100)
    monkeypatch.setattr(kern.cute, "compile", lambda fn, *a, **k: object())
    kw = dict(dtype=torch.bfloat16, h_q=8, h_kv=2, d=256, rope_dim=64, eps=_EPS, has_seq_lens=False)
    norm = kern.compile_qk_norm_rope_bwd(**kw, apply_norm=True, want_dw=True)
    rope = kern.compile_qk_norm_rope_bwd(**kw, apply_norm=False, want_dw=False)
    assert (norm.rows_per_group, rope.rows_per_group) == (DEFAULT_ROWS_PER_GROUP, DEFAULT_ROWS_PER_GROUP_ROPE_ONLY) == (1, 2)
    assert (norm.rows_per_cta, rope.rows_per_cta) == (4, 8)
    explicit = kern.compile_qk_norm_rope_bwd(**kw, apply_norm=False, want_dw=False, rows_per_group=1)
    assert explicit.rows_per_group == 1 and explicit.compiled is not rope.compiled
    assert norm.n_ctas_cap == 100 * CTAS_PER_SM


def test_n_ctas_is_per_class_sm_fill_or_fixed():
    """``min(row_groups_x, cap)`` PER CLASS; the cap is SMs x CTAS_PER_SM under
    ``sm_fill`` and N under ``fixed:N``. Hand-built recipes: no compile."""
    kw = dict(
        compiled=None,
        reduce_compiled=None,
        h_q=32,
        h_kv=2,
        d=256,
        rope_dim=64,
        eps=_EPS,
        rows_per_cta=4,
        apply_norm=True,
        want_dw=True,
        has_seq_lens=False,
        dtype=torch.bfloat16,
    )
    r = QkNormRopeBwdRecipe(n_ctas_cap=1696, **kw)
    # 397B S=32K: Q 32*32768/4 = 262144 groups -> capped; K/V 2*32768/4 = 16384 -> capped
    assert n_ctas_for(r, 32768) == (1696, 1696, 1696)
    # tiny T: every class below the cap, at least one CTA each
    assert n_ctas_for(r, 1) == (8, 1, 1)
    assert n_ctas_for(r, 3) == (24, 2, 2)
    r7 = QkNormRopeBwdRecipe(n_ctas_cap=7, **kw)
    assert n_ctas_for(r7, 3) == (7, 2, 2)
    assert CTAS_PER_SM == 8


def test_execute_refusals_are_typed_and_pre_compile():
    """Presence in BOTH directions for x / rstd / weights / dW partials / seq_lens,
    dtype and H per operand, the partial plane's exact geometry -- all on the
    host, before the artifact is touched (Rule 1)."""
    kw = dict(
        compiled=None, reduce_compiled=None, h_q=8, h_kv=2, d=256, rope_dim=64, eps=_EPS, rows_per_cta=4, has_seq_lens=False, n_ctas_cap=4, dtype=torch.bfloat16
    )
    normed = QkNormRopeBwdRecipe(apply_norm=True, want_dw=True, **kw)
    normed_nodw = QkNormRopeBwdRecipe(apply_norm=True, want_dw=False, **kw)
    rope_only = QkNormRopeBwdRecipe(apply_norm=False, want_dw=False, **kw)
    t = 4
    q = torch.empty(t, 8, 256, dtype=torch.bfloat16)
    k = torch.empty(t, 2, 256, dtype=torch.bfloat16)
    tab = torch.empty(t, 64, dtype=torch.bfloat16)
    w = torch.ones(256, dtype=torch.bfloat16)
    rq, rk = torch.empty(t, 8), torch.empty(t, 2)
    pq, pk = torch.empty(4, 256), torch.empty(2, 256)
    full = dict(xq=q, xk=k, rstd_q=rq, rstd_k=rk, w_q=w, w_k=w, dw_partials_q=pq, dw_partials_k=pk)

    def go(r, **over):
        a = {**full, **over}
        run_qk_norm_rope_bwd(
            r,
            q,
            k,
            k,
            a["xq"],
            a["xk"],
            a["rstd_q"],
            a["rstd_k"],
            a["w_q"],
            a["w_k"],
            tab,
            tab,
            q,
            k,
            k,
            a["dw_partials_q"],
            a["dw_partials_k"],
            a.get("seq_lens"),
            s=a.get("s"),
            stream=0,
        )

    with pytest.raises(ValueError, match="WITH the RMSNorm backward"):
        go(normed, w_q=None, w_k=None)
    with pytest.raises(ValueError, match="WITH the RMSNorm backward"):
        go(normed, xq=None, xk=None)
    with pytest.raises(ValueError, match="WITH the RMSNorm backward"):
        go(normed, rstd_q=None, rstd_k=None)
    with pytest.raises(ValueError, match="together"):
        go(normed, w_k=None)
    with pytest.raises(ValueError, match="WITHOUT the RMSNorm"):
        go(rope_only, dw_partials_q=None, dw_partials_k=None)
    with pytest.raises(ValueError, match="WITHOUT dW partials"):
        go(rope_only, xq=None, xk=None, rstd_q=None, rstd_k=None, w_q=None, w_k=None)
    with pytest.raises(ValueError, match="WITHOUT dW partials"):
        go(normed_nodw)
    with pytest.raises(ValueError, match="WITH dW partials"):
        go(normed, dw_partials_q=None, dw_partials_k=None)
    with pytest.raises(ValueError, match="WITHOUT seq_lens"):
        go(normed, seq_lens=torch.tensor([4], dtype=torch.int32), s=4)
    with pytest.raises(ValueError, match=r"dw_partials_q must be a contiguous fp32 \[4, 256\]"):
        go(normed, dw_partials_q=torch.empty(5, 256))
    with pytest.raises(ValueError, match=r"dw_partials_k must be a contiguous fp32 \[2, 256\]"):
        go(normed, dw_partials_k=torch.empty(2, 256, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match=r"rstd_q must be a contiguous fp32 \[4, 8\]"):
        go(normed, rstd_q=torch.empty(t, 8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="dq is torch.float16 but this artifact was compiled for torch.bfloat16"):
        run_qk_norm_rope_bwd(normed, q.to(torch.float16), k, k, q, k, rq, rk, w, w, tab, tab, q, k, k, pq, pk, stream=0)
    with pytest.raises(ValueError, match="out_v is torch.float16"):
        run_qk_norm_rope_bwd(normed, q, k, k, q, k, rq, rk, w, w, tab, tab, q, k, k.to(torch.float16), pq, pk, stream=0)
    with pytest.raises(ValueError, match="H is fixed per artifact"):
        run_qk_norm_rope_bwd(normed, q, k, k, q, k, rq, rk, w, w, tab, tab, q, torch.empty(t, 3, 256, dtype=torch.bfloat16), k, pq, pk, stream=0)
    with pytest.raises(ValueError, match="out_v has D=128 but this artifact was compiled for D=256; D is fixed per artifact"):
        run_qk_norm_rope_bwd(normed, q, k, k, q, k, rq, rk, w, w, tab, tab, q, k, torch.empty(t, 2, 128, dtype=torch.bfloat16), pq, pk, stream=0)
    with pytest.raises(ValueError, match="x_q has T=2 but dq has T=4; every operand covers the same tokens"):
        run_qk_norm_rope_bwd(normed, q, k, k, q[:2], k, rq, rk, w, w, tab, tab, q, k, k, pq, pk, stream=0)
    with pytest.raises(ValueError, match="cos"):
        run_qk_norm_rope_bwd(normed, q, k, k, q, k, rq, rk, w, w, torch.empty(t, 32, dtype=torch.bfloat16), tab, q, k, k, pq, pk, stream=0)
    # the reduce refuses a plane / output that does not match the recipe
    with pytest.raises(ValueError, match="WITHOUT dW partials"):
        run_dw_norm_reduce(rope_only, pq, pk, torch.empty(256), torch.empty(256), stream=0)
    with pytest.raises(ValueError, match=r"dw_q_norm must be a contiguous fp32 \[256\]"):
        run_dw_norm_reduce(normed, pq, pk, torch.empty(255), torch.empty(256), stream=0)
    with pytest.raises(ValueError, match=r"dw_partials_k must be a contiguous fp32 \[\*, 256\]"):
        run_dw_norm_reduce(normed, pq, torch.empty(2, 128), torch.empty(256), torch.empty(256), stream=0)
    # told the launch's t, the reduce cross-checks the planes against n_ctas_for(r, t): the
    # full 4-row workspace plane handed in for a t=1 launch (2 Q partials written) is refused,
    # not summed over 2 unwritten rows
    assert n_ctas_for(normed, 1) == (2, 1, 1)
    with pytest.raises(ValueError, match=r"dw_partials_q must be a contiguous fp32 \[2, 256\]"):
        run_dw_norm_reduce(normed, pq, pk, torch.empty(256), torch.empty(256), stream=0, t=1)
    with pytest.raises(ValueError, match=r"dw_partials_k must be a contiguous fp32 \[1, 256\]"):
        run_dw_norm_reduce(normed, pq[:2], pk, torch.empty(256), torch.empty(256), stream=0, t=1)


# A hand-built recipe (no compile) for the host-only launch checks: norm arm, no dW partials.
_HOST_RECIPE = dict(
    reduce_compiled=None, h_q=8, h_kv=2, d=256, rope_dim=64, eps=_EPS, rows_per_cta=4, apply_norm=True, want_dw=False, n_ctas_cap=4, dtype=torch.bfloat16
)


def _launch_operands(t, h_q, h_kv, d, rope_dim, dtype, device):
    """A complete, layout-clean operand set for ``_HOST_RECIPE`` (compact ``[T, H, D]`` everywhere)."""
    q = torch.empty(t, h_q, d, dtype=dtype, device=device)
    k = torch.empty(t, h_kv, d, dtype=dtype, device=device)
    w = torch.ones(d, dtype=dtype, device=device)
    tab = torch.empty(t, rope_dim, dtype=dtype, device=device)
    rq, rk = torch.empty(t, h_q, device=device), torch.empty(t, h_kv, device=device)
    return dict(dq=q, dk=k, dv=k, xq=q, xk=k, rstd_q=rq, rstd_k=rk, w_q=w, w_k=w, cos=tab, sin=tab, out_q=q, out_k=k, out_v=k)


def _host_launch(r, ops, **over):
    a = {**ops, **over}
    run_qk_norm_rope_bwd(
        r,
        a["dq"],
        a["dk"],
        a["dv"],
        a["xq"],
        a["xk"],
        a["rstd_q"],
        a["rstd_k"],
        a["w_q"],
        a["w_k"],
        a["cos"],
        a["sin"],
        a["out_q"],
        a["out_k"],
        a["out_v"],
        a.get("dw_partials_q"),
        a.get("dw_partials_k"),
        a.get("seq_lens"),
        s=a.get("s"),
        stream=0,
    )


def test_execute_refuses_a_dense_operand_off_the_16_byte_access_grid():
    """Every ``[T, H, D]`` operand is traced with a SYMBOLIC token stride, so only the host can
    check that the stride is a whole number of 16-B accesses: a band of a slab whose width is
    not a multiple of ``ELEMS_PER_ACCESS`` (N = 8k + 4), a view at storage offset 1, or a
    permuted layout would reach the kernel and fault on odd tokens (a sticky misaligned-address
    CUDA error). Each is a typed ValueError naming the operand and its strides, before the
    artifact is touched."""
    r = QkNormRopeBwdRecipe(compiled=None, has_seq_lens=False, **_HOST_RECIPE)
    t, h_q, h_kv, d = 4, 8, 2, 256
    ops = _launch_operands(t, h_q, h_kv, d, 64, torch.bfloat16, "cpu")
    odd = torch.empty(t, h_q * d + 4, dtype=torch.bfloat16)[:, : h_q * d].view(t, h_q, d)
    assert odd.stride() == (h_q * d + 4, d, 1) and odd.stride(0) % ELEMS_PER_ACCESS == 4
    with pytest.raises(ValueError, match=r"x_q .*strides \(2052, 256, 1\)"):
        _host_launch(r, ops, xq=odd)
    shifted = torch.empty(t * h_kv * d + ELEMS_PER_ACCESS, dtype=torch.bfloat16)[1 : 1 + t * h_kv * d].view(t, h_kv, d)
    assert shifted.data_ptr() % ACCESS_BYTES == 2, "one bf16 element past a 16-B boundary"
    with pytest.raises(ValueError, match=r"out_k .*base 2 B past"):
        _host_launch(r, ops, out_k=shifted)
    permuted = torch.empty(t, d, h_q, dtype=torch.bfloat16).transpose(1, 2)  # strides (d*h_q, 1, h_q)
    with pytest.raises(ValueError, match=r"dq .*strides \(2048, 1, 8\)"):
        _host_launch(r, ops, dq=permuted)


@requires_cuda
def test_execute_admits_the_blocks_compact_and_band_views():
    """The block's real operands pass the layout check: compact ``[T, H, D]`` tensors and the
    proj_slab / dqkvg bands (token stride N = n_qkvg, base at a ``qkvg_offsets`` column), both
    multiples of ``QKVG_TILE_ALIGN`` = 64 elements (128 B), so the 16-B access grid holds for
    every geometry the block admits; a size-1 token dim (torch normalises its stride) too. A
    recording stub stands in for the artifact: the launch is reached once per layout."""
    calls = []
    r = QkNormRopeBwdRecipe(compiled=lambda *a: calls.append(a), has_seq_lens=False, **_HOST_RECIPE)
    h_q, h_kv, d, rope_dim = 8, 2, 256, 64
    geom = GatedAttentionBlockGeometry(d_model=h_q * d, h_q=h_q, h_kv=h_kv, d_head=d, rope_dim=rope_dim)
    o_q, o_g, o_k, o_v = geom.qkvg_offsets
    n = geom.n_qkvg
    assert QKVG_TILE_ALIGN % ELEMS_PER_ACCESS == 0 and n % ELEMS_PER_ACCESS == 0 and all(o % ELEMS_PER_ACCESS == 0 for o in geom.qkvg_offsets)
    for t in (3, 1):
        ops = _launch_operands(t, h_q, h_kv, d, rope_dim, torch.bfloat16, "cuda")
        _host_launch(r, ops)  # compact
        slab = torch.empty(t, n, dtype=torch.bfloat16, device="cuda")
        dqkvg = torch.empty(t, n, dtype=torch.bfloat16, device="cuda")
        bands = dict(
            xq=slab[:, o_q:o_g].view(t, h_q, d),
            xk=slab[:, o_k:o_v].view(t, h_kv, d),
            out_q=dqkvg[:, o_q:o_g].view(t, h_q, d),
            out_k=dqkvg[:, o_k:o_v].view(t, h_kv, d),
            out_v=dqkvg[:, o_v:].view(t, h_kv, d),
        )
        if t > 1:
            assert bands["xq"].stride() == (n, d, 1)
        _host_launch(r, ops, **bands)  # the strided bands
    assert len(calls) == 4


@requires_cuda
def test_execute_refuses_operands_off_the_launch_device():
    """The kernel dereferences every operand as a device pointer. A CPU ``seq_lens``
    (``torch.tensor(lens, dtype=torch.int32)`` without ``device=``) passes the dtype / rank /
    length checks and would be read as a HOST address -- an illegal-address fault at the next
    synchronize, sticky for the process. Refused by name before the artifact is touched, as is
    any other operand off the launch device; the reduce launch checks its planes the same way."""
    t = 4
    ops = _launch_operands(t, 8, 2, 256, 64, torch.bfloat16, "cuda")
    dev = ops["dq"].device
    with_sl = QkNormRopeBwdRecipe(compiled=None, has_seq_lens=True, **_HOST_RECIPE)
    with pytest.raises(ValueError, match=rf"seq_lens must be on {dev} .*got cpu"):
        _host_launch(with_sl, ops, seq_lens=torch.tensor([2, 2], dtype=torch.int32), s=2)
    with pytest.raises(ValueError, match=rf"x_k must be on {dev} .*got cpu"):
        _host_launch(with_sl, ops, xk=ops["xk"].cpu(), seq_lens=torch.tensor([2, 2], dtype=torch.int32, device=dev), s=2)
    dense = QkNormRopeBwdRecipe(compiled=None, has_seq_lens=False, **_HOST_RECIPE)
    with pytest.raises(ValueError, match="dq must be a CUDA tensor"):
        _host_launch(dense, _launch_operands(t, 8, 2, 256, 64, torch.bfloat16, "cpu"))
    reduce = QkNormRopeBwdRecipe(compiled=None, has_seq_lens=False, **{**_HOST_RECIPE, "want_dw": True})
    pq, pk = torch.empty(4, 256, device=dev), torch.empty(2, 256, device=dev)
    with pytest.raises(ValueError, match=rf"dw_k_norm must be on {dev} .*got cpu"):
        run_dw_norm_reduce(reduce, pq, pk, torch.empty(256, device=dev), torch.empty(256), stream=0)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@requires_cuda
@_QK_NORM
@_DTYPES
@pytest.mark.parametrize("t, h_q, h_kv, d, rope_dim", [(64, 8, 2, 256, 64), (64, 4, 1, 128, 32), (64, 2, 2, 64, 16)])
def test_norm_rope_bwd_matches_fp64_autograd(dtype, t, h_q, h_kv, d, rope_dim, qk_norm):
    inp = _make(t, h_q, h_kv, d, rope_dim, dtype)
    r = _compile(inp, rope_dim=rope_dim, qk_norm=qk_norm, want_dw=qk_norm)
    assert r.apply_norm is qk_norm and r.want_dw is qk_norm and r.dtype is dtype
    # the SHIPPED per-arm default, on a REAL compile: 1 with the norm, 2 RoPE-only
    assert r.rows_per_group == (DEFAULT_ROWS_PER_GROUP if qk_norm else DEFAULT_ROWS_PER_GROUP_ROPE_ONLY)
    out_q, out_k, out_v = torch.empty_like(inp["dq"]), torch.empty_like(inp["dk"]), torch.full_like(inp["dv"], 1.5e3)
    _, dw = _launch(inp, r, out_q, out_k, out_v, qk_norm=qk_norm, want_dw=qk_norm)
    want_dq, want_dwq = _ref(inp["dq"], inp["xq"], inp["w_q"], inp["cos"], inp["sin"], rope_dim, qk_norm)
    want_dk, want_dwk = _ref(inp["dk"], inp["xk"], inp["w_k"], inp["cos"], inp["sin"], rope_dim, qk_norm)
    _check_dx(out_q, want_dq, dtype)
    _check_dx(out_k, want_dk, dtype)
    assert torch.equal(out_v, inp["dv"]), "the V band is a copy and must be bit-exact"
    if qk_norm:
        _check_dw(dw[0], want_dwq)
        _check_dw(dw[1], want_dwk)
    else:
        assert dw is None and want_dwq is None


@requires_cuda
@_QK_NORM
@pytest.mark.parametrize("rows_per_group", [1, 2])
def test_norm_rope_bwd_every_rows_per_group_arm_matches_fp64_autograd(qk_norm, rows_per_group):
    """BOTH ``rows_per_group`` traces of BOTH arms against the oracle (dx, dW, the V copy),
    at row counts that leave a ragged last group under either value (H=3 / 1). The shipped
    default of one arm is the non-default of the other, and a defect confined to the SECOND
    row of a two-row trace (``tokens[1]`` / ``out_addrs[1]`` / ``deads[1]``) passes every test
    that compiles the other value (a dense PASS proves nothing about a
    const_expr-folded arm)."""
    t, h_q, h_kv, d, rope_dim, dtype = 257, 3, 1, 256, 64, torch.bfloat16
    inp = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=21)
    r = _compile(inp, rope_dim=rope_dim, qk_norm=qk_norm, want_dw=qk_norm, rows_per_group=rows_per_group)
    assert r.rows_per_group == rows_per_group and r.rows_per_cta == 4 * rows_per_group
    assert (t * h_q) % r.rows_per_cta != 0 and (t * h_kv) % r.rows_per_cta != 0, "the shape must leave a ragged last group"
    out_q, out_k, out_v = (torch.full_like(x, 1.5e3) for x in (inp["dq"], inp["dk"], inp["dv"]))
    _, dw = _launch(inp, r, out_q, out_k, out_v, qk_norm=qk_norm, want_dw=qk_norm)
    for ten in (out_q, out_k, out_v):
        assert not (ten == 1.5e3).any(), "a row kept its poison: not every row was written"
    want_dq, want_dwq = _ref(inp["dq"], inp["xq"], inp["w_q"], inp["cos"], inp["sin"], rope_dim, qk_norm)
    want_dk, want_dwk = _ref(inp["dk"], inp["xk"], inp["w_k"], inp["cos"], inp["sin"], rope_dim, qk_norm)
    _check_dx(out_q, want_dq, dtype)
    _check_dx(out_k, want_dk, dtype)
    assert torch.equal(out_v, inp["dv"])
    if qk_norm:
        _check_dw(dw[0], want_dwq)
        _check_dw(dw[1], want_dwk)


@requires_cuda
def test_norm_rope_bwd_exact_adjoint_on_random_tables():
    """``cos, sin = randn`` (NOT duplicated halves): the exact adjoint
    ``dy*cos - rotate_half(dy*sin)`` still matches autograd; the naive
    ``dy*cos - rotate_half(dy)*sin`` is off by O(1) here."""
    t, h_q, h_kv, d, rope_dim, dtype = 64, 8, 2, 256, 64, torch.bfloat16
    inp = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=11, random_tables=True)
    for qk_norm in (True, False):
        r = _compile(inp, rope_dim=rope_dim, qk_norm=qk_norm, want_dw=False)
        out_q, out_k, out_v = torch.empty_like(inp["dq"]), torch.empty_like(inp["dk"]), torch.empty_like(inp["dv"])
        _launch(inp, r, out_q, out_k, out_v, qk_norm=qk_norm, want_dw=False)
        want_dq, _ = _ref(inp["dq"], inp["xq"], inp["w_q"], inp["cos"], inp["sin"], rope_dim, qk_norm)
        want_dk, _ = _ref(inp["dk"], inp["xk"], inp["w_k"], inp["cos"], inp["sin"], rope_dim, qk_norm)
        _check_dx(out_q, want_dq, dtype)
        _check_dx(out_k, want_dk, dtype)
    # and the naive form really is wrong on these tables (so the pin has teeth)
    dy = inp["dq"][..., :rope_dim].double()
    c, s = inp["cos"][:, None, :].double(), inp["sin"][:, None, :].double()
    half = rope_dim // 2
    rot = lambda x: torch.cat((-x[..., half:], x[..., :half]), dim=-1)  # noqa: E731
    exact = dy * c - rot(dy * s)
    naive = dy * c - rot(dy) * s
    assert (exact - naive).abs().max().item() > 0.1


@requires_cuda
@pytest.mark.parametrize("t", [64, 257])
def test_norm_rope_bwd_passthrough_is_bit_exact(t):
    """RoPE-only: no fp32 op touches ``[rope_dim, D)`` (widen, narrow, same value),
    the V band is a copy, and a second launch is bit-identical."""
    h_q, h_kv, d, rope_dim, dtype = 8, 2, 256, 64, torch.bfloat16
    inp = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=12)
    r = _compile(inp, rope_dim=rope_dim, qk_norm=False, want_dw=False)
    outs = []
    for _ in range(2):
        out_q, out_k, out_v = (torch.full_like(x, 1.5e3) for x in (inp["dq"], inp["dk"], inp["dv"]))
        _launch(inp, r, out_q, out_k, out_v, qk_norm=False, want_dw=False)
        outs.append((out_q, out_k, out_v))
    out_q, out_k, out_v = outs[0]
    assert torch.equal(out_q[..., rope_dim:], inp["dq"][..., rope_dim:]), "RoPE-only Q passthrough dims are not a bit-exact copy"
    assert torch.equal(out_k[..., rope_dim:], inp["dk"][..., rope_dim:]), "RoPE-only K passthrough dims are not a bit-exact copy"
    assert torch.equal(out_v, inp["dv"]), "the V band is not a bit-exact copy"
    assert not torch.allclose(out_q[..., :rope_dim].float(), inp["dq"][..., :rope_dim].float(), atol=1e-2), "the rope band did not rotate"
    for a, b in zip(outs[0], outs[1]):
        assert torch.equal(a, b)


@requires_cuda
@pytest.mark.parametrize("rows_per_group", [1, 2])
def test_norm_rope_bwd_dw_is_bitwise_across_runs_and_partial_plane_is_written(rows_per_group):
    """Determinism PER KNOB: the partials plane is poison-NaN before the launch
    and no NaN survives (every partial row written), two runs are ``torch.equal``
    on dx AND dW. Equality ACROSS ``rows_per_group`` values is NOT asserted -- the
    row-to-CTA assignment, hence the fp32 summation order, changes with the knob."""
    t, h_q, h_kv, d, rope_dim, dtype = 1000, 8, 2, 256, 64, torch.bfloat16  # ~2000 Q groups: real persistence per CTA
    inp = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=13)
    r = _compile(inp, rope_dim=rope_dim, qk_norm=True, want_dw=True, rows_per_group=rows_per_group)
    assert r.rows_per_cta == (128 // 32) * rows_per_group
    runs = []
    for _ in range(2):
        out_q, out_k, out_v = torch.empty_like(inp["dq"]), torch.empty_like(inp["dk"]), torch.empty_like(inp["dv"])
        (pq, pk), (dwq, dwk) = _launch(inp, r, out_q, out_k, out_v, qk_norm=True, want_dw=True)
        assert not torch.isnan(pq).any() and not torch.isnan(pk).any(), "a partial row kept its NaN poison"
        assert not torch.isnan(dwq).any() and not torch.isnan(dwk).any()
        runs.append((out_q, out_k, out_v, pq, pk, dwq, dwk))
    for name, a, b in zip(("out_q", "out_k", "out_v", "pq", "pk", "dwq", "dwk"), runs[0], runs[1]):
        assert torch.equal(a, b), f"{name}: two runs differ"
    _, want_dwq = _ref(inp["dq"], inp["xq"], inp["w_q"], inp["cos"], inp["sin"], rope_dim, True)
    _, want_dwk = _ref(inp["dk"], inp["xk"], inp["w_k"], inp["cos"], inp["sin"], rope_dim, True)
    _check_dw(runs[0][5], want_dwq)
    _check_dw(runs[0][6], want_dwk)


@requires_cuda
def test_norm_rope_bwd_fixed_n_ctas_policy_reduces_the_same_sum():
    """``fixed:N`` trades occupancy for a cross-device stable summation order; its
    plane has N rows per class (or fewer when the class is smaller), the reduce
    consumes exactly that many, and the total matches the oracle."""
    t, h_q, h_kv, d, rope_dim, dtype = 300, 8, 2, 256, 64, torch.bfloat16
    inp = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=14)
    r = _compile(inp, rope_dim=rope_dim, qk_norm=True, want_dw=True, n_ctas_policy="fixed:37")
    assert r.n_ctas_cap == 37 and n_ctas_for(r, t) == (37, 37, 37)
    out_q, out_k, out_v = torch.empty_like(inp["dq"]), torch.empty_like(inp["dk"]), torch.empty_like(inp["dv"])
    (pq, pk), (dwq, dwk) = _launch(inp, r, out_q, out_k, out_v, qk_norm=True, want_dw=True)
    assert pq.shape == (37, d) and pk.shape == (37, d)
    _, want_dwq = _ref(inp["dq"], inp["xq"], inp["w_q"], inp["cos"], inp["sin"], rope_dim, True)
    _, want_dwk = _ref(inp["dk"], inp["xk"], inp["w_k"], inp["cos"], inp["sin"], rope_dim, True)
    _check_dw(dwq, want_dwq)
    _check_dw(dwk, want_dwk)
    # the reduce is a FIXED-order fp32 sum of the plane rows -- pin it against a host
    # mirror of exactly that order (the accumulation order IS the contract): REDUCE_LANES
    # residue classes of rows, each summed ascending from 0, then the classes ascending
    for plane, dw in ((pq, dwq), (pk, dwk)):
        host = torch.zeros(d, device="cuda", dtype=torch.float32)
        for lane in range(REDUCE_LANES):
            part = torch.zeros(d, device="cuda", dtype=torch.float32)
            for c in range(lane, plane.shape[0], REDUCE_LANES):
                part = part + plane[c]
            host = host + part
        assert torch.equal(host, dw), "the reduce is not the documented fixed-order fp32 sum of the partials"


@requires_cuda
@pytest.mark.parametrize("rows_per_group", [1, 2])
@pytest.mark.parametrize("lens", [(8, 0), (8, 3), (0, 8)], ids=["batch1-empty", "batch1-partial", "batch0-empty"])
def test_norm_rope_bwd_dead_rows(lens, rows_per_group):
    """Dead rows (``pos >= seq_lens[b]``) SELECT ``dx := 0`` (stored, all three
    bands) and contribute 0 to dW -- the select sits BEFORE the FMA, so the NaN
    planted in the dead rows' dq / dk / dv must not poison dW. dW equals, bit for
    bit, the same run with finite dead rows, and matches the oracle over the
    live rows. Both ``rows_per_group`` traces: ``deads[1]`` is a second-row fact."""
    b, s, h_q, h_kv, d, rope_dim, dtype = 2, 8, 4, 2, 256, 64, torch.bfloat16
    t = b * s
    inp = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=15)
    seq_lens = torch.tensor(lens, device="cuda", dtype=torch.int32)
    tok = torch.arange(t, device="cuda")
    dead = (tok % s) >= seq_lens[tok // s]
    clean = {k: v.clone() for k, v in inp.items()}
    for name in ("dq", "dk", "dv"):
        inp[name][dead] = float("nan")
    r = _compile(inp, rope_dim=rope_dim, qk_norm=True, want_dw=True, has_seq_lens=True, rows_per_group=rows_per_group)
    assert r.has_seq_lens is True and r.rows_per_group == rows_per_group
    outs = {}
    for tag, data in (("nan", inp), ("clean", clean)):
        out_q, out_k, out_v = (torch.full_like(x, 1.5e3) for x in (data["dq"], data["dk"], data["dv"]))
        _, (dwq, dwk) = _launch(data, r, out_q, out_k, out_v, qk_norm=True, want_dw=True, seq_lens=seq_lens, s=s)
        outs[tag] = (out_q, out_k, out_v, dwq, dwk)
    out_q, out_k, out_v, dwq, dwk = outs["nan"]
    for name, ten in (("out_q", out_q), ("out_k", out_k), ("out_v", out_v), ("dW_q", dwq), ("dW_k", dwk)):
        assert torch.isfinite(ten).all(), f"{name} carries a non-finite value"
    for ten in (out_q, out_k, out_v):
        assert torch.equal(ten[dead], torch.zeros_like(ten[dead])), "dead rows are not exactly 0"
    assert torch.equal(dwq, outs["clean"][3]) and torch.equal(dwk, outs["clean"][4]), "a NaN dead row leaked into dW"
    live = ~dead
    if live.any():
        want_dq, want_dwq = _ref(clean["dq"][live], clean["xq"][live], clean["w_q"], clean["cos"][live], clean["sin"][live], rope_dim, True)
        want_dk, want_dwk = _ref(clean["dk"][live], clean["xk"][live], clean["w_k"], clean["cos"][live], clean["sin"][live], rope_dim, True)
        _check_dx(out_q[live], want_dq, dtype)
        _check_dx(out_k[live], want_dk, dtype)
        assert torch.equal(out_v[live], clean["dv"][live])
        _check_dw(dwq, want_dwq)
        _check_dw(dwk, want_dwk)
    else:
        assert torch.equal(dwq, torch.zeros_like(dwq)) and torch.equal(dwk, torch.zeros_like(dwk))


@requires_cuda
@pytest.mark.parametrize("h_q", [8, 32])
@pytest.mark.parametrize("t", [1, 3, 13, 257])
def test_norm_rope_bwd_bands_and_shapes(t, h_q):
    """The block's real shape: dq / dk / dv compact (B4's outputs), x_q / x_k read
    from proj_slab bands, dx written into the Q / K / V bands of a poison-filled
    ``dqkvg`` (token stride N). GQA ``h_kv=2``; ragged tails."""
    h_kv, d, rope_dim, dtype = 2, 256, 64, torch.bfloat16
    # the band offsets are the API's layout contract (qkvg_offsets), not a hand-rolled copy of it
    geom = GatedAttentionBlockGeometry(d_model=h_q * d, h_q=h_q, h_kv=h_kv, d_head=d, rope_dim=rope_dim)
    o_q, o_g, o_k, o_v = geom.qkvg_offsets
    n = geom.n_qkvg
    inp = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=16)
    slab = torch.randn(t, n, device="cuda", dtype=dtype)
    slab[:, o_q:o_g] = inp["xq"].reshape(t, h_q * d)
    slab[:, o_k:o_v] = inp["xk"].reshape(t, h_kv * d)
    banded = {**inp, "xq": slab[:, o_q:o_g].view(t, h_q, d), "xk": slab[:, o_k:o_v].view(t, h_kv, d)}
    if t > 1:  # torch normalises the stride of a size-1 leading dim
        assert banded["xq"].stride() == (n, d, 1)
    dqkvg = torch.full((t, n), 1.5e3, device="cuda", dtype=dtype)
    out_q, out_k, out_v = dqkvg[:, o_q:o_g].view(t, h_q, d), dqkvg[:, o_k:o_v].view(t, h_kv, d), dqkvg[:, o_v:].view(t, h_kv, d)
    r = _compile(inp, rope_dim=rope_dim, qk_norm=True, want_dw=True)
    _, (dwq, dwk) = _launch(banded, r, out_q, out_k, out_v, qk_norm=True, want_dw=True)
    assert (dqkvg[:, o_g:o_k] == 1.5e3).all(), "the GATE band was written by the norm backward"
    for ten in (out_q, out_k, out_v):
        assert not (ten == 1.5e3).any(), "a row kept its poison: not every row was written"
    want_dq, want_dwq = _ref(inp["dq"], inp["xq"], inp["w_q"], inp["cos"], inp["sin"], rope_dim, True)
    want_dk, want_dwk = _ref(inp["dk"], inp["xk"], inp["w_k"], inp["cos"], inp["sin"], rope_dim, True)
    _check_dx(out_q, want_dq, dtype)
    _check_dx(out_k, want_dk, dtype)
    assert torch.equal(out_v, inp["dv"])
    _check_dw(dwq, want_dwq)
    _check_dw(dwk, want_dwk)
    # the compact run agrees bit for bit with the banded one (same rows, same CTAs)
    cq, ck, cv = torch.empty_like(inp["dq"]), torch.empty_like(inp["dk"]), torch.empty_like(inp["dv"])
    _, (cdwq, cdwk) = _launch(inp, r, cq, ck, cv, qk_norm=True, want_dw=True)
    assert torch.equal(cq, out_q) and torch.equal(ck, out_k) and torch.equal(cv, out_v) and torch.equal(cdwq, dwq) and torch.equal(cdwk, dwk)


@requires_cuda
def test_norm_rope_bwd_compile_cache_keys_on_every_knob():
    """Every knob that changes the traced body is a key; ``eps`` is a recorded fact of the
    forward (the kernel consumes the SAVED rstd and never recomputes it) and deliberately
    is NOT -- two callers with different eps share one artifact, correctly."""
    kw = dict(dtype=torch.bfloat16, h_q=8, h_kv=2, d=256, rope_dim=64, eps=_EPS)
    a = compile_qk_norm_rope_bwd(apply_norm=True, want_dw=True, has_seq_lens=False, **kw)
    assert compile_qk_norm_rope_bwd(apply_norm=True, want_dw=True, has_seq_lens=False, **kw).compiled is a.compiled, "a repeat request is a cache hit"
    b = compile_qk_norm_rope_bwd(apply_norm=True, want_dw=True, has_seq_lens=False, **{**kw, "eps": 1e-5})
    assert b.compiled is a.compiled and b.eps == 1e-5, "eps is a recorded fact, not a compile key"
    for other in (
        compile_qk_norm_rope_bwd(apply_norm=True, want_dw=False, has_seq_lens=False, **kw),
        compile_qk_norm_rope_bwd(apply_norm=False, want_dw=False, has_seq_lens=False, **kw),
        compile_qk_norm_rope_bwd(apply_norm=True, want_dw=True, has_seq_lens=True, **kw),
        compile_qk_norm_rope_bwd(apply_norm=True, want_dw=True, has_seq_lens=False, rows_per_group=2, **kw),
        compile_qk_norm_rope_bwd(apply_norm=True, want_dw=True, has_seq_lens=False, const_head_counts=False, **kw),
        compile_qk_norm_rope_bwd(apply_norm=True, want_dw=True, has_seq_lens=False, **{**kw, "dtype": torch.float16}),
    ):
        assert other.compiled is not a.compiled, "two different knob sets share an artifact"


@requires_cuda
def test_norm_rope_bwd_const_head_counts_is_bit_identical_to_the_runtime_arm():
    t, h_q, h_kv, d, rope_dim, dtype = 37, 3, 1, 256, 64, torch.bfloat16  # H=3: non-power-of-two divide
    inp = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=17)
    outs = []
    for const in (True, False):
        r = _compile(inp, rope_dim=rope_dim, qk_norm=True, want_dw=True, const_head_counts=const)
        out_q, out_k, out_v = torch.empty_like(inp["dq"]), torch.empty_like(inp["dk"]), torch.empty_like(inp["dv"])
        _, (dwq, dwk) = _launch(inp, r, out_q, out_k, out_v, qk_norm=True, want_dw=True)
        outs.append((out_q, out_k, out_v, dwq, dwk))
    for a, b in zip(outs[0], outs[1]):
        assert torch.equal(a, b), "const_head_counts changed a bit"


@requires_cuda
def test_norm_rope_bwd_in_place_over_dq_matches_out_of_place():
    """``out_q is dq`` (the compact in-place form): every lane reads its whole row
    before any lane stores, the shuffle stays inside the row."""
    t, h_q, h_kv, d, rope_dim, dtype = 96, 8, 2, 256, 64, torch.bfloat16
    inp = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=18)
    r = _compile(inp, rope_dim=rope_dim, qk_norm=True, want_dw=False)
    out_q, out_k, out_v = torch.empty_like(inp["dq"]), torch.empty_like(inp["dk"]), torch.empty_like(inp["dv"])
    _launch(inp, r, out_q, out_k, out_v, qk_norm=True, want_dw=False)
    ip = {**inp, "dq": inp["dq"].clone(), "dk": inp["dk"].clone()}
    _launch(ip, r, ip["dq"], ip["dk"], torch.empty_like(inp["dv"]), qk_norm=True, want_dw=False)
    assert torch.equal(ip["dq"], out_q) and torch.equal(ip["dk"], out_k)


@requires_rubin
@_QK_NORM
def test_norm_rope_bwd_397b_geometry_on_rubin(qk_norm):
    """The block's layer geometry (H_q=32, H_kv=2, D=256, rope 64) at a few thousand
    tokens with the persistent grid capped at SMs x 8: dx and dW vs fp64 autograd."""
    t, h_q, h_kv, d, rope_dim, dtype = 4096, 32, 2, 256, 64, torch.bfloat16
    inp = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=19)
    r = _compile(inp, rope_dim=rope_dim, qk_norm=qk_norm, want_dw=qk_norm)
    out_q, out_k, out_v = torch.empty_like(inp["dq"]), torch.empty_like(inp["dk"]), torch.empty_like(inp["dv"])
    _, dw = _launch(inp, r, out_q, out_k, out_v, qk_norm=qk_norm, want_dw=qk_norm)
    want_dq, want_dwq = _ref(inp["dq"], inp["xq"], inp["w_q"], inp["cos"], inp["sin"], rope_dim, qk_norm)
    want_dk, want_dwk = _ref(inp["dk"], inp["xk"], inp["w_k"], inp["cos"], inp["sin"], rope_dim, qk_norm)
    _check_dx(out_q, want_dq, dtype)
    _check_dx(out_k, want_dk, dtype)
    assert torch.equal(out_v, inp["dv"])
    if qk_norm:
        _check_dw(dw[0], want_dwq)
        _check_dw(dw[1], want_dwk)


# ---------------------------------------------------------------------------
# The oracle's window / bottom-right mask arm (any device, fp64)
# ---------------------------------------------------------------------------


def _mask_pins(device):
    """The band the block's SDPA lowers, spelled as index arithmetic: ``window_left = W``
    keeps ``k >= q + diag - W``; causal keeps ``k <= q + diag + window_right``;
    ``diag = S_kv - S_q`` under bottom-right, else 0."""
    s = 8
    q = torch.arange(s, device=device)[:, None]
    k = torch.arange(s, device=device)[None, :]
    kw = dict(seq_lens=None, batch_index=0, q_lo=0, device=device)
    m = _key_padding_and_causal_mask(s, s, is_causal=True, window_left=2, **kw)
    assert torch.equal(m, (k <= q) & (k >= q - 2))
    m = _key_padding_and_causal_mask(s, s, is_causal=False, window_left=2, **kw)
    assert torch.equal(m, k >= q - 2)
    m = _key_padding_and_causal_mask(s, s, is_causal=True, window_right=1, **kw)
    assert torch.equal(m, k <= q + 1)
    m = _key_padding_and_causal_mask(s, s, is_causal=True, causal_bottom_right=True, **kw)
    assert torch.equal(m, k <= q), "bottom-right with S_q == S_kv is plain causal"
    m = _key_padding_and_causal_mask(3, s, is_causal=True, causal_bottom_right=True, s_q_total=3, **kw)
    assert torch.equal(m, k <= torch.arange(3, device=device)[:, None] + (s - 3))
    # chunking must compose: rows [q_lo, q_lo + s_q) of the full mask
    full = _key_padding_and_causal_mask(s, s, is_causal=True, window_left=3, **kw)
    chunk = _key_padding_and_causal_mask(3, s, is_causal=True, window_left=3, **{**kw, "q_lo": 4})
    assert torch.equal(chunk, full[4:7])
    # the pre-existing arms are untouched: default kwargs == the old behaviour
    m = _key_padding_and_causal_mask(s, s, is_causal=True, **kw)
    assert torch.equal(m, k <= q)
    # the two combinations the block API refuses (api.py: GatedAttentionBlockGeometry) are
    # refused here too -- a flag the reference silently ignored would agree with no lowered mask
    with pytest.raises(ValueError, match="window_right requires is_causal=True"):
        _key_padding_and_causal_mask(s, s, is_causal=False, window_right=1, **kw)
    with pytest.raises(ValueError, match="causal_bottom_right=True requires is_causal=True"):
        _key_padding_and_causal_mask(s, s, is_causal=False, causal_bottom_right=True, **kw)


def test_reference_mask_arm_pins_on_the_host():
    """The index-arithmetic pins and the two refusals need no GPU."""
    _mask_pins(torch.device("cpu"))


@requires_cuda
@pytest.mark.parametrize("case", ["swa640", "swa640_causal", "bottom_right"])
def test_reference_window_arm_matches_torch_sdpa(case):
    """The oracle's masked attention vs ``F.scaled_dot_product_attention`` with
    the SAME explicit bool mask, all fp64, ``atol=1e-12`` -- the block backward's ``[swa640]``
    e2e case is only as good as this branch."""
    dev = torch.device("cuda")
    _mask_pins(dev)
    geom_kw = dict(d_model=256, h_q=4, h_kv=2, d_head=64, rope_dim=16)
    knobs = {
        "swa640": dict(is_causal=False, window_left=640),
        "swa640_causal": dict(is_causal=True, window_left=640),
        "bottom_right": dict(is_causal=True, causal_bottom_right=True),
    }[case]
    geom = RefGeometry(**geom_kw, **knobs)
    plain = RefGeometry(**geom_kw, is_causal=knobs["is_causal"])
    s = 1024
    inp = make_inputs(geom, batch=1, seq_len=s, dtype=torch.float64, device=dev)
    ref = gated_attention_block_reference(**inp, geom=geom, acc_dtype=torch.float64)
    assert ref.o.dtype is torch.float64 and ref.lse.dtype is torch.float64

    # torch side: the same fp64 projection / norm / RoPE, then F.sdpa under the explicit mask
    proj = inp["h"] @ inp["w_qkvg"].t()
    q_pre, _, k_pre, v = split_qkvg(proj, geom)
    q, _ = qk_norm_rope_reference(q_pre, inp["w_q_norm"], inp["cos"], inp["sin"], geom.rope_dim, geom.qk_norm_eps, acc_dtype=torch.float64)
    k, _ = qk_norm_rope_reference(k_pre, inp["w_k_norm"], inp["cos"], inp["sin"], geom.rope_dim, geom.qk_norm_eps, acc_dtype=torch.float64)
    allowed = _key_padding_and_causal_mask(
        s,
        s,
        is_causal=geom.is_causal,
        seq_lens=None,
        batch_index=0,
        q_lo=0,
        device=dev,
        window_left=geom.window_left,
        window_right=geom.window_right,
        causal_bottom_right=geom.causal_bottom_right,
    )
    assert allowed.any(dim=-1).all(), "this case must have no dead row (F.sdpa would return NaN there)"
    rep = geom.h_q // geom.h_kv
    o = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.repeat_interleave(rep, dim=2).transpose(1, 2), v.repeat_interleave(rep, dim=2).transpose(1, 2), attn_mask=allowed, scale=geom.scale
    ).transpose(1, 2)
    torch.testing.assert_close(ref.o, o, rtol=0, atol=1e-12)
    # the arm bites: the windowed oracle differs from the un-windowed one (bottom-right == causal at S_q == S_kv)
    ref_plain = gated_attention_block_reference(**inp, geom=plain, acc_dtype=torch.float64)
    if case == "bottom_right":
        assert torch.equal(ref.o, ref_plain.o)
    else:
        assert (ref.o - ref_plain.o).abs().max().item() > 1e-3
    # LSE contract under the window: finite everywhere (no dead row), and exact vs a direct fp64 log-sum-exp
    scores = torch.einsum("bshd,bthd->bhst", q, k.repeat_interleave(rep, dim=2)) * geom.scale
    lse = scores.masked_fill(~allowed[None, None], float("-inf")).logsumexp(-1)
    torch.testing.assert_close(ref.lse, lse, rtol=0, atol=1e-10)
