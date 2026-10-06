# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""B3 -- the sigmoid-gate BACKWARD kernel of the gated attention block.

``dO = dO_gated * s``, ``dG = dO_gated * O * s * (1 - s)``, optional
``O_gated = O * s`` with ``s = sigmoid(GATE)``. Plain vectorized LDG/STG, no
shuffle, no TMA, no arch-specific path -- so like its forward twin
(``test_elementwise.py``) it runs, and is tested, anywhere CuTe DSL does.

Oracles are fp64 autograd of ``o * sigmoid(g)`` on the STORAGE-rounded inputs
(``test/AGENTS.md``: build the reference in fp64; TF32 pinned off for the
duration of every test in this module).
"""

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.gated_attention_block.kernels.sigmoid_gate_bwd import (  # noqa: E402
    DEFAULT_ROWS_PER_GROUP,
    DOT_CHUNK_ELEMS,
    DOT_THREADS_PER_ROW,
    SigmoidGateBwdRecipe,
    compile_sigmoid_gate_bwd,
    moved_bytes,
    run_sigmoid_gate_bwd,
    validate_shape,
)
from cudnn.gated_attention_block.kernels.qk_norm_rope import ACCESS_BYTES, ELEMS_PER_ACCESS  # noqa: E402
from cudnn.gated_attention_block import GatedAttentionBlockGeometry  # noqa: E402

pytestmark = pytest.mark.L0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# Defined PER MODULE on purpose (hoisting it into the conftest is a separate change;
# this module must not depend on it). Used only for the 397B-geometry confirmation.
_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"the 397B-geometry confirmation runs on SM107 only; found {_cc()}")

_DTYPES = pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
_OG = pytest.mark.parametrize("want_og", [True, False], ids=["og", "no_og"])

# The gate kernel's own bound (test_elementwise.py:54-70): kernel and oracle
# round ONCE from fp32-class math, so the budget is ~1 ulp of the OUTPUT --
# relative, plus a small atol for values near zero.
_TOL = {torch.bfloat16: dict(rtol=2.0**-7, atol=1e-3), torch.float16: dict(rtol=2.0**-10, atol=1e-4)}


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


def _make(t, h, d, dtype, seed=0, gate_std=3.0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape, std=1.0):
        return (torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32) * std).to(dtype)

    return rnd(t, h, d), rnd(t, h, d), rnd(t, h, d, std=gate_std)  # dOg, O (pre-gate), gate (pre-sigmoid)


def _ref(dog, o, gate):
    """fp64 autograd of ``og = o * sigmoid(g)``: ``(dO, dG, O_gated)`` in fp64."""
    o64 = o.double().requires_grad_(True)
    g64 = gate.double().requires_grad_(True)
    og = o64 * torch.sigmoid(g64)
    do, dg = torch.autograd.grad(og, (o64, g64), dog.double())
    return do, dg, og.detach()


def _run(dog, o, gate, do, dg, og=None, seq_lens=None, *, h, d, s=None, rows_per_group=DEFAULT_ROWS_PER_GROUP, const_head_count=True, delta=None):
    r = compile_sigmoid_gate_bwd(
        dtype=dog.dtype,
        h=h,
        d=d,
        has_og=og is not None,
        has_seq_lens=seq_lens is not None,
        rows_per_group=rows_per_group,
        const_head_count=const_head_count,
        has_delta=delta is not None,
    )
    run_sigmoid_gate_bwd(r, dog, o, gate, do, dg, og, seq_lens, s=s, stream=_stream(), delta=delta)
    torch.cuda.synchronize()
    return r


def _chain_dot_do_o(o_bshd, do_bshd):
    """The SDPA backward chain's OWN pre-pass (``bprop_chain_common.dot_do_o_host``, the launch the sm107 pointer host issues)
    over compact ``[B, S, H, D]`` tensors -> the ``[B, H, ceil128(S)]`` fp32 delta it carves, NaN-poisoned first."""
    import cuda.bindings.driver as cuda_drv
    from cutlass.cute.runtime import from_dlpack

    from cudnn.sdpa.bwd.kernels.bprop_chain_common import DOT_CHUNK_ELEMS as CHAIN_CHUNK, DOT_Q_TILE, dot_do_o_host

    b, s, h, d = (int(x) for x in o_bshd.shape)
    s_pad = -(-s // DOT_Q_TILE) * DOT_Q_TILE
    delta = torch.full((b, h, s_pad), float("nan"), device="cuda", dtype=torch.float32)
    args = [from_dlpack(x, assumed_align=16) for x in (o_bshd, do_bshd, delta)]
    dot_do_o_host(*args, None, None, DOT_Q_TILE, d, d, CHAIN_CHUNK, False, False, cuda_drv.CUstream(_stream()))
    torch.cuda.synchronize()
    return delta


def _check(got, want64, dtype):
    torch.testing.assert_close(got.float(), want64.to(dtype).float(), **_TOL[dtype])


# ---------------------------------------------------------------------------
# Host-only
# ---------------------------------------------------------------------------


def test_moved_bytes_counts_three_reads_two_writes_and_the_optional_og():
    t, h, d = 8192, 32, 256
    assert moved_bytes(t, h, d, has_og=False) == 5 * t * h * d * 2
    assert moved_bytes(t, h, d, has_og=True) == 6 * t * h * d * 2
    # the 397B-geometry budget: 80 KiB / 96 KiB per token at H=32, D=256
    assert moved_bytes(1, 32, 256, has_og=False) == 80 * 1024
    assert moved_bytes(1, 32, 256, has_og=True) == 96 * 1024


def test_validate_shape_rejects():
    with pytest.raises(ValueError, match="multiple of 8"):
        validate_shape(250, 128)
    with pytest.raises(ValueError, match="multiple of the 32 lanes"):
        validate_shape(256, 100)


def test_recipe_refusals_are_typed_and_pre_compile():
    """Presence (both ways), dtype and H are checked on the HOST before the
    artifact is touched (Rule 1: no silent fallback, no silent ignore)."""
    base = dict(compiled=None, h=8, d=256, rows_per_cta=4, dtype=torch.bfloat16)
    with_og = SigmoidGateBwdRecipe(has_og=True, has_seq_lens=False, **base)
    without_og = SigmoidGateBwdRecipe(has_og=False, has_seq_lens=False, **base)
    with_sl = SigmoidGateBwdRecipe(has_og=False, has_seq_lens=True, **base)
    x = torch.empty(4, 8, 256, dtype=torch.bfloat16)
    sl = torch.tensor([2, 2], dtype=torch.int32)
    with pytest.raises(ValueError, match="WITH an O_gated output"):
        run_sigmoid_gate_bwd(with_og, x, x, x, x, x, None, None, stream=0)
    with pytest.raises(ValueError, match="WITHOUT an O_gated output"):
        run_sigmoid_gate_bwd(without_og, x, x, x, x, x, x, None, stream=0)
    with pytest.raises(ValueError, match="WITH seq_lens"):
        run_sigmoid_gate_bwd(with_sl, x, x, x, x, x, None, None, s=2, stream=0)
    with pytest.raises(ValueError, match="WITHOUT seq_lens"):
        run_sigmoid_gate_bwd(without_og, x, x, x, x, x, None, sl, s=2, stream=0)
    with pytest.raises(ValueError, match="seq_lens needs s"):
        run_sigmoid_gate_bwd(with_sl, x, x, x, x, x, None, sl, stream=0)
    with pytest.raises(ValueError, match="int32"):
        run_sigmoid_gate_bwd(with_sl, x, x, x, x, x, None, sl.to(torch.int64), s=2, stream=0)
    with pytest.raises(ValueError, match=r"T // s"):
        run_sigmoid_gate_bwd(with_sl, x, x, x, x, x, None, torch.tensor([2], dtype=torch.int32), s=2, stream=0)
    with pytest.raises(ValueError, match="divide"):
        run_sigmoid_gate_bwd(with_sl, x, x, x, x, x, None, sl, s=3, stream=0)
    f16 = x.to(torch.float16)
    for name, args in (("dog", (f16, x, x, x, x)), ("o", (x, f16, x, x, x)), ("gate", (x, x, f16, x, x)), ("do", (x, x, x, f16, x)), ("dg", (x, x, x, x, f16))):
        with pytest.raises(ValueError, match=f"{name} is torch.float16 but this artifact was compiled for torch.bfloat16"):
            run_sigmoid_gate_bwd(without_og, *args, None, None, stream=0)
    with pytest.raises(ValueError, match="og is torch.float16"):
        run_sigmoid_gate_bwd(with_og, x, x, x, x, x, f16, None, stream=0)
    h4 = torch.empty(4, 4, 256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="H is fixed per artifact"):
        run_sigmoid_gate_bwd(without_og, x, h4, x, x, x, None, None, stream=0)
    with pytest.raises(ValueError, match="dg has D=128 but this artifact was compiled for D=256; D is fixed per artifact"):
        run_sigmoid_gate_bwd(without_og, x, x, x, x, torch.empty(4, 8, 128, dtype=torch.bfloat16), None, None, stream=0)
    with pytest.raises(ValueError, match="o has T=2 but dO_gated has T=4; every operand covers the same tokens"):
        run_sigmoid_gate_bwd(without_og, x, x[:2], x, x, x, None, None, stream=0)
    # compile-time: dtype
    with pytest.raises(ValueError, match="bf16/f16 only"):
        compile_sigmoid_gate_bwd(dtype=torch.float32, h=8, d=256, has_og=False, has_seq_lens=False)


def test_launch_refuses_an_operand_off_the_16_byte_access_grid():
    """Every ``[T, H, D]`` operand is traced with a SYMBOLIC token stride, so only the host can
    check that the stride is a whole number of 16-B accesses: a band of a slab whose width is
    not a multiple of ``ELEMS_PER_ACCESS`` (N = 8k + 4), a view at storage offset 1, or a
    permuted layout would reach the kernel and fault on odd tokens (a sticky misaligned-address
    CUDA error). Each is a typed ValueError naming the operand and its strides, before the
    artifact is touched. (``test_gate_bwd_writes_the_gate_band_only`` launches through the same
    check with the block's real GATE band views.)"""
    r = SigmoidGateBwdRecipe(compiled=None, h=8, d=256, rows_per_cta=4, dtype=torch.bfloat16, has_og=False, has_seq_lens=False)
    t, h, d = 4, 8, 256
    x = torch.empty(t, h, d, dtype=torch.bfloat16)
    odd = torch.empty(t, h * d + 4, dtype=torch.bfloat16)[:, : h * d].view(t, h, d)
    assert odd.stride() == (h * d + 4, d, 1) and odd.stride(0) % ELEMS_PER_ACCESS == 4
    with pytest.raises(ValueError, match=r"gate .*strides \(2052, 256, 1\)"):
        run_sigmoid_gate_bwd(r, x, x, odd, x, x, None, None, stream=0)
    shifted = torch.empty(t * h * d + ELEMS_PER_ACCESS, dtype=torch.bfloat16)[1 : 1 + t * h * d].view(t, h, d)
    assert shifted.data_ptr() % ACCESS_BYTES == 2, "one bf16 element past a 16-B boundary"
    with pytest.raises(ValueError, match=r"dg .*base 2 B past"):
        run_sigmoid_gate_bwd(r, x, x, x, x, shifted, None, None, stream=0)
    permuted = torch.empty(t, d, h, dtype=torch.bfloat16).transpose(1, 2)  # strides (d*h, 1, h)
    with pytest.raises(ValueError, match=r"do .*strides \(2048, 1, 8\)"):
        run_sigmoid_gate_bwd(r, x, x, x, permuted, x, None, None, stream=0)


@requires_cuda
def test_launch_refuses_operands_off_the_launch_device():
    """The kernel dereferences every operand as a device pointer. A CPU ``seq_lens``
    (``torch.tensor(lens, dtype=torch.int32)`` without ``device=``) passes the dtype / rank /
    length checks and would be read as a HOST address -- an illegal-address fault at the next
    synchronize, sticky for the process. Refused by name before the artifact is touched, as is
    any other operand off the launch device."""
    base = dict(compiled=None, h=8, d=256, rows_per_cta=4, dtype=torch.bfloat16)
    with_sl = SigmoidGateBwdRecipe(has_og=False, has_seq_lens=True, **base)
    with_og = SigmoidGateBwdRecipe(has_og=True, has_seq_lens=False, **base)
    x = torch.empty(4, 8, 256, dtype=torch.bfloat16, device="cuda")
    dev = x.device
    with pytest.raises(ValueError, match=rf"seq_lens must be on {dev} .*got cpu"):
        run_sigmoid_gate_bwd(with_sl, x, x, x, x, x, None, torch.tensor([2, 2], dtype=torch.int32), s=2, stream=0)
    with pytest.raises(ValueError, match=rf"og must be on {dev} .*got cpu"):
        run_sigmoid_gate_bwd(with_og, x, x, x, x, x, x.cpu(), None, stream=0)
    c = x.cpu()
    with pytest.raises(ValueError, match="dog must be a CUDA tensor"):
        run_sigmoid_gate_bwd(with_og, c, c, c, c, c, c, None, stream=0)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@requires_cuda
@_DTYPES
@_OG
@pytest.mark.parametrize("t, h, d", [(64, 8, 256), (37, 4, 128), (1, 2, 64)])
def test_gate_bwd_matches_fp64_autograd(dtype, want_og, t, h, d):
    dog, o, gate = _make(t, h, d, dtype)
    do, dg = torch.empty_like(dog), torch.empty_like(dog)
    og = torch.empty_like(dog) if want_og else None
    r = _run(dog, o, gate, do, dg, og, h=h, d=d)
    assert r.has_og is want_og and r.has_seq_lens is False and r.dtype is dtype
    want_do, want_dg, want_og_ = _ref(dog, o, gate)
    _check(do, want_do, dtype)
    _check(dg, want_dg, dtype)
    if want_og:
        _check(og, want_og_, dtype)


@requires_cuda
@pytest.mark.parametrize("rows_per_group", [1, 2])
def test_gate_bwd_rows_per_group_knob_is_bit_identical(rows_per_group):
    """The occupancy knob changes scheduling, never arithmetic: the two artifacts
    must agree bit for bit (and both with the runtime-H arm)."""
    t, h, d = 96, 8, 256
    dog, o, gate = _make(t, h, d, torch.bfloat16, seed=4)
    outs = []
    for const in (True, False):
        do, dg, og = (torch.zeros_like(dog) for _ in range(3))
        _run(dog, o, gate, do, dg, og, h=h, d=d, rows_per_group=rows_per_group, const_head_count=const)
        outs.append((do, dg, og))
    ref = _run(dog, o, gate, torch.zeros_like(dog), torch.zeros_like(dog), torch.zeros_like(dog), h=h, d=d)
    do1, dg1, og1 = torch.zeros_like(dog), torch.zeros_like(dog), torch.zeros_like(dog)
    _run(dog, o, gate, do1, dg1, og1, h=h, d=d)
    for a, b in zip(outs[0], outs[1]):
        assert torch.equal(a, b), "const_head_count changed a bit"
    for a, b in zip(outs[0], (do1, dg1, og1)):
        assert torch.equal(a, b), "rows_per_group changed a bit"
    assert ref.rows_per_cta == (128 // 32) * DEFAULT_ROWS_PER_GROUP


@requires_cuda
def test_gate_bwd_saturation_is_exact():
    """``s(1-s)`` in the ``0.25*(1-t)*(1+t)`` tanh form: at ``g = +-30``
    ``tanh(15)`` is exactly 1.0f, so ``dG == 0`` EXACTLY, ``dO == dOg`` at +30
    and ``dO == 0`` at -30 -- a saturation that is finite-but-wrong would pass a
    tolerance and silently leak gradient into a gate that should be frozen."""
    t, h, d = 8, 2, 256
    dog, o, _ = _make(t, h, d, torch.bfloat16, seed=1)
    for sign in (+1.0, -1.0):
        gate = torch.full((t, h, d), sign * 30.0, device="cuda", dtype=torch.bfloat16)
        do, dg, og = (torch.full_like(dog, 1.5e3) for _ in range(3))
        _run(dog, o, gate, do, dg, og, h=h, d=d)
        assert torch.equal(dg, torch.zeros_like(dg)), f"dG must be exactly 0 at g={sign * 30}"
        if sign > 0:
            assert torch.equal(do, dog), "dO must equal dOg exactly at g=+30 (s == 1)"
            assert torch.equal(og, o), "O_gated must equal O exactly at g=+30"
        else:
            assert torch.equal(do, torch.zeros_like(do)), "dO must be exactly 0 at g=-30 (s == 0)"
            assert torch.equal(og, torch.zeros_like(og)), "O_gated must be exactly 0 at g=-30"


@requires_cuda
def test_gate_bwd_in_place_dO_equals_out_of_place():
    """``do is dog`` is the block's default (dO overwrites dO_gated): every lane
    reads its 16 B before it writes them and no lane touches another's."""
    t, h, d = 96, 8, 256
    dog, o, gate = _make(t, h, d, torch.bfloat16, seed=2)
    do, dg, og = (torch.empty_like(dog) for _ in range(3))
    _run(dog, o, gate, do, dg, og, h=h, d=d)
    ip = dog.clone()
    dg2, og2 = torch.empty_like(dog), torch.empty_like(dog)
    _run(ip, o, gate, ip, dg2, og2, h=h, d=d)
    assert torch.equal(ip, do) and torch.equal(dg2, dg) and torch.equal(og2, og)


@requires_cuda
def test_gate_bwd_writes_the_gate_band_only():
    """The block's real shape: dG lands in the GATE band of the ``[T, N]`` dqkvg
    scratch (token stride N) and the gate is read from the GATE band of a
    proj_slab; O / dOg / dO are compact. Poison the slab: only the GATE band may
    change, and it must equal the compact run bit for bit."""
    t, h_q, h_kv, d = 64, 8, 2, 256
    # the band offsets are the API's layout contract (qkvg_offsets), not a hand-rolled copy of it
    geom = GatedAttentionBlockGeometry(d_model=h_q * d, h_q=h_q, h_kv=h_kv, d_head=d, rope_dim=64)
    o_q, o_g, o_k, o_v = geom.qkvg_offsets
    n = geom.n_qkvg
    dog, o, gate_c = _make(t, h_q, d, torch.bfloat16, seed=3)
    proj = torch.randn(t, n, device="cuda", dtype=torch.bfloat16)
    proj[:, o_g:o_k] = gate_c.reshape(t, h_q * d)
    gate_band = proj[:, o_g:o_k].view(t, h_q, d)
    assert gate_band.stride() == (n, d, 1)
    dqkvg = torch.full((t, n), 1.5e3, device="cuda", dtype=torch.bfloat16)
    dg_band = dqkvg[:, o_g:o_k].view(t, h_q, d)
    do, og = torch.empty_like(dog), torch.empty_like(dog)
    _run(dog, o, gate_band, do, dg_band, og, h=h_q, d=d)
    do_c, dg_c, og_c = (torch.empty_like(dog) for _ in range(3))
    _run(dog, o, gate_c, do_c, dg_c, og_c, h=h_q, d=d)
    assert torch.equal(dg_band, dg_c) and torch.equal(do, do_c) and torch.equal(og, og_c)
    for lo, hi in ((o_q, o_g), (o_k, o_v), (o_v, n)):
        assert (dqkvg[:, lo:hi] == 1.5e3).all(), f"columns [{lo}, {hi}) outside the GATE band were written"


@requires_cuda
@pytest.mark.parametrize("lens", [(8, 0), (8, 3), (0, 8)], ids=["batch1-empty", "batch1-partial", "batch0-empty"])
def test_gate_bwd_dead_rows_are_exactly_zero(lens):
    """Rows at or past ``seq_lens[b]`` are SELECTED to exact 0 and STORED --
    never ``* 0`` (the O residue may be NaN) and never skipped (the buffer would
    keep stale rows). The dead rows' dOg / O are NaN here by construction."""
    b, s, h, d = 2, 8, 4, 256
    t = b * s
    dog, o, gate = _make(t, h, d, torch.bfloat16, seed=5)
    seq_lens = torch.tensor(lens, device="cuda", dtype=torch.int32)
    tok = torch.arange(t, device="cuda")
    dead = (tok % s) >= seq_lens[tok // s]
    dog[dead] = float("nan")
    o[dead] = float("nan")
    do, dg, og = (torch.full_like(dog, 1.5e3) for _ in range(3))
    r = _run(dog, o, gate, do, dg, og, seq_lens, h=h, d=d, s=s)
    assert r.has_seq_lens is True
    for name, ten in (("dO", do), ("dG", dg), ("O_gated", og)):
        assert torch.isfinite(ten).all(), f"{name} carries a non-finite value"
        assert torch.equal(ten[dead], torch.zeros_like(ten[dead])), f"{name} dead rows are not exactly 0"
    live = ~dead
    if live.any():
        want_do, want_dg, want_og = _ref(dog[live], o[live], gate[live])
        _check(do[live], want_do, torch.bfloat16)
        _check(dg[live], want_dg, torch.bfloat16)
        _check(og[live], want_og, torch.bfloat16)


@requires_cuda
def test_gate_bwd_dense_artifact_matches_the_seq_lens_one_on_live_rows():
    """The select is folded OUT when ``seq_lens`` is None (no divide traced); with
    every entry at full length the two artifacts must agree bit for bit."""
    b, s, h, d = 3, 16, 4, 256
    t = b * s
    dog, o, gate = _make(t, h, d, torch.bfloat16, seed=6)
    dense = [torch.empty_like(dog) for _ in range(3)]
    padded = [torch.empty_like(dog) for _ in range(3)]
    _run(dog, o, gate, *dense, h=h, d=d)
    _run(dog, o, gate, *padded, torch.full((b,), s, device="cuda", dtype=torch.int32), h=h, d=d, s=s)
    for a, c in zip(dense, padded):
        assert torch.equal(a, c)


@requires_cuda
@pytest.mark.parametrize("h", [1, 2, 3, 8, 32])
@pytest.mark.parametrize("t", [1, 3, 13, 257])
def test_gate_bwd_ragged_tails(t, h):
    """Row counts that leave a partial last CTA: every row is written (poison
    gone), no wild write (a compute-sanitizer memcheck run of this case is
    recorded in the PR), and the values are right. H=3 exercises the
    non-power-of-two divide in the constant-head arm."""
    d = 256
    dog, o, gate = _make(t, h, d, torch.bfloat16, seed=7)
    do, dg, og = (torch.full_like(dog, 1.5e3) for _ in range(3))
    _run(dog, o, gate, do, dg, og, h=h, d=d)
    for ten in (do, dg, og):
        assert not (ten == 1.5e3).any(), "a row kept its poison: not every row was written"
    want_do, want_dg, want_og = _ref(dog, o, gate)
    _check(do, want_do, torch.bfloat16)
    _check(dg, want_dg, torch.bfloat16)
    _check(og, want_og, torch.bfloat16)


@requires_cuda
def test_gate_bwd_two_runs_bitwise():
    t, h, d = 128, 8, 256
    dog, o, gate = _make(t, h, d, torch.bfloat16, seed=8)
    runs = []
    for _ in range(2):
        do, dg, og = (torch.empty_like(dog) for _ in range(3))
        _run(dog, o, gate, do, dg, og, h=h, d=d)
        runs.append((do, dg, og))
    for a, c in zip(*runs):
        assert torch.equal(a, c)


@requires_cuda
def test_gate_bwd_compile_cache_keys_on_every_knob():
    kw = dict(dtype=torch.bfloat16, h=8, d=256)
    a = compile_sigmoid_gate_bwd(has_og=True, has_seq_lens=False, **kw)
    assert compile_sigmoid_gate_bwd(has_og=True, has_seq_lens=False, **kw).compiled is a.compiled, "a repeat request is a cache hit"
    for other in (
        compile_sigmoid_gate_bwd(has_og=False, has_seq_lens=False, **kw),
        compile_sigmoid_gate_bwd(has_og=True, has_seq_lens=True, **kw),
        compile_sigmoid_gate_bwd(has_og=True, has_seq_lens=False, rows_per_group=2, **kw),
        compile_sigmoid_gate_bwd(has_og=True, has_seq_lens=False, const_head_count=False, **kw),
        compile_sigmoid_gate_bwd(has_og=True, has_seq_lens=False, **{**kw, "dtype": torch.float16}),
    ):
        assert other.compiled is not a.compiled, "two different knob sets share an artifact"


@requires_rubin
@_OG
def test_gate_bwd_397b_geometry_on_rubin(want_og):
    """The block's layer geometry (H=32, D=256) at a few thousand tokens, GATE
    read from a slab band and dG written into a slab band, vs fp64 autograd."""
    t, h_q, h_kv, d = 2048, 32, 2, 256
    geom = GatedAttentionBlockGeometry(d_model=4096, h_q=h_q, h_kv=h_kv, d_head=d, rope_dim=64)  # the 397B layer
    _, o_g, o_k, _ = geom.qkvg_offsets
    n = geom.n_qkvg
    dog, o, gate_c = _make(t, h_q, d, torch.bfloat16, seed=9)
    proj = torch.zeros(t, n, device="cuda", dtype=torch.bfloat16)
    proj[:, o_g:o_k] = gate_c.reshape(t, h_q * d)
    dqkvg = torch.full((t, n), 1.5e3, device="cuda", dtype=torch.bfloat16)
    do = torch.empty_like(dog)
    og = torch.empty_like(dog) if want_og else None
    _run(dog, o, proj[:, o_g:o_k].view(t, h_q, d), do, dqkvg[:, o_g:o_k].view(t, h_q, d), og, h=h_q, d=d)
    want_do, want_dg, want_og_ = _ref(dog, o, gate_c)
    _check(do, want_do, torch.bfloat16)
    _check(dqkvg[:, o_g:o_k].view(t, h_q, d), want_dg, torch.bfloat16)
    if want_og:
        _check(og, want_og_, torch.bfloat16)
    assert (dqkvg[:, :o_g] == 1.5e3).all() and (dqkvg[:, o_k:] == 1.5e3).all()


# ---------------------------------------------------------------------------
# The optional delta output: the SDPA backward's dot_do_o pre-pass, bitwise
# ---------------------------------------------------------------------------


def test_gate_bwd_delta_geometry_is_the_chains():
    """The reduction order the delta reproduces is defined by the chain's row geometry (8 threads x 64-element chunks); the
    kernel keeps its own copy of the two numbers so this module does not import the chain -- pinned equal here, so a
    change on either side is a visible break, not a silent last-bit drift."""
    from cudnn.sdpa.bwd.kernels.bprop_chain_common import DOT_CHUNK_ELEMS as chain_chunk
    from cudnn.sdpa.bwd.kernels.sm120._common import _COPY_ELEMS as chain_copy

    assert DOT_CHUNK_ELEMS == chain_chunk == 64
    assert DOT_THREADS_PER_ROW == chain_chunk // chain_copy == 8
    assert moved_bytes(10, 4, 256, has_og=True, has_delta=True) == moved_bytes(10, 4, 256, has_og=True) + 4 * 10 * 4


@requires_cuda
@_DTYPES
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("b, s", [(1, 128), (3, 100), (2, 257)], ids=["b1-aligned", "b3-ragged", "b2-two-tiles-ragged"])
def test_gate_bwd_delta_is_bitwise_the_chains_dot_do_o(dtype, d, b, s):
    """``delta`` (fp32 ``[B, H, S_pad]``) equals the chain's own ``dot_do_o`` over the dO this kernel STORED -- bit for bit,
    pad tail included (zeros past ``S``; both buffers NaN-poisoned first), on every CUDA device: the same fp32 products
    summed in the same order (module docstring).  ``d`` walks 1, 2 and 4 chain hand-off rounds; a ragged ``S`` exercises
    the pad tail and a two-tile ``S`` the chain's second q tile."""
    h = 4
    t = b * s
    s_pad = -(-s // 128) * 128
    dog, o, gate = _make(t, h, d, dtype, seed=11)
    do, dg, og = (torch.empty_like(dog) for _ in range(3))
    delta = torch.full((b, h, s_pad), float("nan"), device="cuda", dtype=torch.float32)
    r = _run(dog, o, gate, do, dg, og, h=h, d=d, s=s, delta=delta)
    assert r.has_delta is True
    want = _chain_dot_do_o(o.view(b, s, h, d), do.view(b, s, h, d))
    assert torch.isfinite(delta).all(), "an unwritten (NaN-poisoned) delta cell"
    assert torch.equal(delta, want), f"delta differs from dot_do_o: max|diff|={(delta - want).abs().max().item():.3e}"
    assert torch.equal(delta[:, :, s:], torch.zeros_like(delta[:, :, s:]))
    # the other outputs are untouched by the extra output: bitwise the no-delta artifact's
    do2, dg2, og2 = (torch.empty_like(dog) for _ in range(3))
    _run(dog, o, gate, do2, dg2, og2, h=h, d=d)
    assert torch.equal(do, do2) and torch.equal(dg, dg2) and torch.equal(og, og2)


@requires_cuda
def test_gate_bwd_delta_dead_rows_are_exactly_zero_and_live_rows_the_chains():
    """With ``seq_lens`` a dead row's delta is a SELECTED exact 0 (its O / dOg are NaN here); the live rows are the
    chain's ``dot_do_o`` over the stored (zeroed on dead rows) dO."""
    b, s, h, d = 2, 8, 4, 256
    t = b * s
    dog, o, gate = _make(t, h, d, torch.bfloat16, seed=12)
    seq_lens = torch.tensor((8, 3), device="cuda", dtype=torch.int32)
    tok = torch.arange(t, device="cuda")
    dead = (tok % s) >= seq_lens[tok // s]
    dog[dead] = float("nan")
    o[dead] = float("nan")
    do, dg, og = (torch.full_like(dog, 1.5e3) for _ in range(3))
    delta = torch.full((b, h, 128), float("nan"), device="cuda", dtype=torch.float32)
    _run(dog, o, gate, do, dg, og, seq_lens, h=h, d=d, s=s, delta=delta)
    assert torch.isfinite(delta).all()
    dead_bhs = dead.view(b, s)[:, None, :].expand(b, h, s)
    assert torch.equal(delta[:, :, :s][dead_bhs], torch.zeros_like(delta[:, :, :s][dead_bhs]))
    o_live, do_live = o.clone(), do.clone()
    o_live[dead] = 0  # the chain over the stored dO would multiply the NaN residue by the zero dO; the kernel selects instead
    want = _chain_dot_do_o(o_live.view(b, s, h, d), do_live.view(b, s, h, d))
    assert torch.equal(delta, want)


@requires_cuda
def test_gate_bwd_delta_contract_is_typed():
    """Both directions of ``has_delta`` (Rule 1), ``s`` required, the fp32 contiguous ``[B, H, S_pad >= s]`` layout, the
    16-B base, the launch device -- every one a ValueError naming the operand, before any launch; and a ``d_head`` the
    chain's 64-element chunk cannot tile is refused at compile."""
    b, s, h, d = 2, 16, 4, 256
    t = b * s
    dog, o, gate = _make(t, h, d, torch.bfloat16, seed=13)
    do, dg = torch.empty_like(dog), torch.empty_like(dog)
    delta = torch.zeros(b, h, 128, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="d_head % 64"):
        compile_sigmoid_gate_bwd(dtype=torch.bfloat16, h=h, d=32, has_og=False, has_seq_lens=False, has_delta=True)
    with_delta = compile_sigmoid_gate_bwd(dtype=torch.bfloat16, h=h, d=d, has_og=False, has_seq_lens=False, has_delta=True)
    without = compile_sigmoid_gate_bwd(dtype=torch.bfloat16, h=h, d=d, has_og=False, has_seq_lens=False)
    assert with_delta.compiled is not without.compiled and without.has_delta is False
    st = _stream()
    with pytest.raises(ValueError, match="WITH a delta"):
        run_sigmoid_gate_bwd(with_delta, dog, o, gate, do, dg, s=s, stream=st)
    with pytest.raises(ValueError, match="WITHOUT a delta"):
        run_sigmoid_gate_bwd(without, dog, o, gate, do, dg, s=s, stream=st, delta=delta)
    with pytest.raises(ValueError, match="delta needs s"):
        run_sigmoid_gate_bwd(with_delta, dog, o, gate, do, dg, stream=st, delta=delta)
    with pytest.raises(ValueError, match="must divide"):
        run_sigmoid_gate_bwd(with_delta, dog, o, gate, do, dg, s=7, stream=st, delta=delta)
    with pytest.raises(ValueError, match="CONTIGUOUS fp32"):
        run_sigmoid_gate_bwd(with_delta, dog, o, gate, do, dg, s=s, stream=st, delta=delta.to(torch.bfloat16))
    with pytest.raises(ValueError, match="CONTIGUOUS fp32"):
        run_sigmoid_gate_bwd(with_delta, dog, o, gate, do, dg, s=s, stream=st, delta=torch.zeros(b, h, 256, device="cuda")[:, :, ::2])
    with pytest.raises(ValueError, match=r"delta must be \[B=2, H=4, S_pad >= 16\]"):
        run_sigmoid_gate_bwd(with_delta, dog, o, gate, do, dg, s=s, stream=st, delta=torch.zeros(b, h + 1, 128, device="cuda"))
    with pytest.raises(ValueError, match=r"S_pad >= 16"):
        run_sigmoid_gate_bwd(with_delta, dog, o, gate, do, dg, s=s, stream=st, delta=torch.zeros(b, h, 8, device="cuda"))
    with pytest.raises(ValueError, match="16-B aligned"):
        run_sigmoid_gate_bwd(with_delta, dog, o, gate, do, dg, s=s, stream=st, delta=torch.zeros(b * h * 128 + 1, device="cuda")[1:].view(b, h, 128))
    with pytest.raises(ValueError, match="delta must be on"):
        run_sigmoid_gate_bwd(with_delta, dog, o, gate, do, dg, s=s, stream=st, delta=torch.zeros(b, h, 128))
    assert torch.isfinite(dog).all()  # no launch happened: nothing wrote the outputs (they are torch.empty, so only the inputs are checked)
