# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fused CSA/HCA Compressor gated-pooling kernels (``cudnn.csa``).

Ported with the kernels from Megatron-LM (https://github.com/NVIDIA/Megatron-LM/pull/5984,
measurements and numerics in https://github.com/NVIDIA/Megatron-LM/issues/5968). Covers:

  - numerics of the fused region vs an fp32-intermediate eager reference, per ratio
    family: at ``ratio == 4`` ``dKV``/``dScore`` are bit-identical and the forward is
    within one bf16 rounding step on a tiny fraction of elements; at ``ratio == 128``
    the contract is faithfulness to the fp32 eager reference: all three match it
    within tolerance thresholds (differing elements <= max(1, 0.1%), max_abs <=
    1.6e-2) calibrated on this suite's input distribution — bf16's grid is relative,
    so absolute deviations scale with input magnitude — and carry fp64-oracle parity
    on inputs whose fp32 intermediates stay finite (inputs that overflow the
    reference's fp32 intermediates reproduce its NaN pattern instead: the committed
    gate's overflow-intermediate case; see docs/fe-oss-apis/csa.md and the gate's
    scaled-input case). Both families are also compared against the verbatim
    upstream eager numerics (tolerance), over ragged THD packs including segments
    shorter than ``ratio``;
  - static-capacity padding rows (``total_comp > cu_seqlens_comp[-1]``);
  - kernel-side zero-writes to never-consumed ``dKV``/``dScore`` slots: NaN-canary
    (uninitialized) gradient buffers stay bitwise-equal to zero-initialized runs,
    every never-consumed slot class is asserted EXACTLY zero, the consumed slots
    match the eager reference per the ratio's contract, and the ``total_comp == 0``
    host fallback still hands back exact zeros;
  - run-to-run determinism of forward / ``dKV`` / ``dScore`` (``dAPE`` uses fp32 atomics
    and is exempt by design; the backward refuses to run under
    ``torch.use_deterministic_algorithms(True)``, and only accumulates into
    ``grad_ape`` — re-zeroing is the caller's job on the class API);
  - the ratio=128 dispatch envelope: schedule selection at every nb_total bucket
    boundary, and (L1) one execution of every shipped (config, schedule) kernel —
    fast-exp, two-phase and vec=1 buckets included — against the contract;
  - CUDA graph capture: warmup -> capture fwd+bwd -> replay (including replay with new
    data and a smaller device-side true row count, checked on all four outputs and
    bitwise against a direct call), and the loud error when the first call for a
    configuration would JIT under capture;
  - ``check_support`` boundaries (validated envelope: CC 10.0; ratio 4 with coff
    {1, 2}, ratio 128 with coff {1, 2} x head_dim {128, 512}; BF16 kv/score, FP32 ape,
    int32 cu_seqlens and int32 flat-offset bounds).

The eager reference below mirrors the exact region of Megatron-LM
``Compressor._forward_thd`` (non-pre-grouped THD path) that the fused kernels replace:
gather-index build -> gather -> ``+ APE`` -> overlap-window transform (``coff == 2``
only, ``Compressor._overlap_transform_thd``; ``coff == 1`` keeps the block's own
``ratio``-token window) -> fp32 softmax -> gated weighted sum -> bf16 cast. ``mode``
selects the numerics: "upstream" reproduces the eager code exactly
(softmax weights rounded to bf16, bf16 multiply); "fp32" keeps all intermediates fp32
with a single final bf16 rounding (the fused kernels' numerics); "fp64" is an oracle.
"""

import pytest
import torch


def _import_compressor():
    """Import ``cudnn.csa.compressor``, skipping only on a missing cutedsl stack."""
    # Skip only when the optional cutedsl dependency stack is missing; a broken
    # cudnn.csa package itself must fail the tests, not skip them.
    pytest.importorskip("cutlass", reason="Environment not supported: cudnn[cutedsl] not installed")
    pytest.importorskip("cuda.bindings", reason="Environment not supported: cuda-python not installed")
    from cudnn.csa import compressor

    return compressor


def _require_sm100():
    """Skip the test unless a CC 10.0 (Blackwell) CUDA GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("compute capability 10.0 GPU required")


# ---------------------------------------------------------------------------
# Eager reference (self-contained mirror of the Megatron-LM eager region)
# ---------------------------------------------------------------------------


def _batch_of_row(cu_seqlens, total):
    """Segment index owning each packed row (mirror of Megatron-LM ``batch_of_row``)."""
    n_seg = cu_seqlens.shape[0] - 1
    row_idx = torch.arange(total, device=cu_seqlens.device, dtype=torch.int64)
    return torch.bucketize(row_idx, cu_seqlens[1:], right=True).clamp(max=max(n_seg - 1, 0))


def _overlap_transform_thd(tensor, is_first_in_seg, head_dim, fill_value=0):
    """Mirror of Megatron-LM ``Compressor._overlap_transform_thd``.

    Input shape:  [total_comp, ratio, b, coff * head_dim]
    Output shape: [total_comp, 2 * ratio, b, head_dim]
    """
    n, ratio, b_dim, _ = tensor.size()
    d = head_dim
    new_tensor = tensor.new_full((n, 2 * ratio, b_dim, d), fill_value)
    new_tensor[:, ratio:] = tensor[:, :, :, d:]
    # Previous group's first-half data -- shift by 1 along dim-0.
    prev_data = torch.roll(tensor[:, :, :, :d], shifts=1, dims=0)
    # Zero-fill (or fill_value-fill) segment boundaries.
    prev_data[is_first_in_seg] = fill_value
    new_tensor[:, :ratio] = prev_data
    return new_tensor


def _eager_pool(kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, d, coff, mode):
    """Eager pooling region (see module docstring); ``coff == 1`` skips the overlap transform."""
    device = kv.device
    row_idx = torch.arange(total_comp, device=device, dtype=cu_seqlens_comp.dtype)
    batch_ids = _batch_of_row(cu_seqlens_comp, total_comp)
    valid_comp = row_idx < cu_seqlens_comp[-1]
    local_pos = row_idx - cu_seqlens_comp[batch_ids]
    local_pos = torch.where(valid_comp, local_pos, torch.zeros_like(local_pos))
    base = cu_seqlens[batch_ids].unsqueeze(1) + local_pos.unsqueeze(1) * ratio
    base = torch.where(valid_comp.unsqueeze(1), base, torch.zeros_like(base))
    offsets = torch.arange(ratio, device=device, dtype=base.dtype).unsqueeze(0)
    gather_idx = base + offsets  # (total_comp, ratio)

    if mode == "fp32":
        kv = kv.float()
        score = score.float()
    elif mode == "fp64":
        kv = kv.double()
        score = score.double()
        ape = ape.double()

    kv_grouped = kv[gather_idx]  # (total_comp, ratio, 1, coff * d)
    score_grouped = score[gather_idx]
    score_grouped = score_grouped + ape.view(1, ratio, 1, -1)

    if coff == 2:
        is_first = local_pos == 0
        kv_grouped = _overlap_transform_thd(kv_grouped, is_first, d, fill_value=0)
        score_grouped = _overlap_transform_thd(score_grouped, is_first, d, fill_value=float("-inf"))

    if mode == "upstream":
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float32).to(kv_grouped.dtype)
        out = (kv_grouped * weights).sum(dim=1)
    elif mode == "fp32":
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float32)
        out = (kv_grouped * weights).sum(dim=1).to(torch.bfloat16)
    else:  # fp64 oracle
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float64)
        out = (kv_grouped * weights).sum(dim=1)
    return out  # (total_comp, 1, d)


# ---------------------------------------------------------------------------
# Input construction and runners
# ---------------------------------------------------------------------------


def _make_inputs(lens, d, ratio, coff, seed=1234, device="cuda"):
    """Build a seeded random THD pack (kv, score, ape, cu, cuc, total_comp, grad_out) for ``lens``."""
    total = sum(lens)
    w = coff * d
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kv = torch.randn(total, 1, w, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    score = (torch.randn(total, 1, w, generator=gen, dtype=torch.float32).mul_(1.5)).to(torch.bfloat16)
    ape = torch.randn(ratio, w, generator=gen, dtype=torch.float32).mul_(0.25)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=device)
    seg_comp = torch.tensor([seg_len // ratio for seg_len in lens])
    cuc = torch.tensor([0] + list(seg_comp.cumsum(0)), dtype=torch.int32, device=device)
    total_comp = int(cuc[-1].item())
    go = torch.randn(total_comp, 1, d, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    return kv.to(device), score.to(device), ape.to(device), cu, cuc, total_comp, go.to(device)


def _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode):
    """Forward + backward through the eager reference; returns (out, dKV, dScore, dAPE)."""
    dtype = torch.float64 if mode == "fp64" else None
    kv_l = (kv.to(dtype) if dtype else kv.clone()).requires_grad_(True)
    score_l = (score.to(dtype) if dtype else score.clone()).requires_grad_(True)
    ape_l = (ape.to(dtype) if dtype else ape.clone()).requires_grad_(True)
    out = _eager_pool(kv_l, score_l, ape_l, cu, cuc, total_comp, ratio, d, coff, mode)
    out.backward(go.to(out.dtype))
    torch.cuda.synchronize()
    return out.detach(), kv_l.grad.detach(), score_l.grad.detach(), ape_l.grad.detach()


def _run_fused(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go):
    """Forward + backward through the fused wrappers; returns (out, dKV, dScore, dAPE)."""
    compressor = _import_compressor()
    total = kv.shape[0]
    out = compressor.csa_compressor_forward_wrapper(
        kv.view(total, -1),
        score.view(total, -1),
        ape,
        cu,
        cuc,
        ratio=ratio,
        head_dim=d,
        coff=coff,
        total_comp=total_comp,
    )["out"]
    grads = compressor.csa_compressor_backward_wrapper(
        kv.view(total, -1),
        score.view(total, -1),
        ape,
        cu,
        cuc,
        go.view(total_comp, d),
        ratio=ratio,
        head_dim=d,
        coff=coff,
    )
    torch.cuda.synchronize()
    return (
        out.view(total_comp, 1, d),
        grads["grad_kv"].view_as(kv),
        grads["grad_score"].view_as(score),
        grads["grad_ape"],
    )


def _assert_grads_vs_fp32(gkv, gs, ref_kv, ref_s, ratio):
    """dKV/dScore vs the fp32-intermediate eager reference, per the ratio's contract.

    ratio=4: bit-identical (the production bitwise contract, unchanged).
    ratio=128: deterministic tolerance contract — the fused backward reorders the
    den/S reductions (fixed chunk merge) and hoists 1/den, and some forward buckets
    use the ex2.approx fast exp, so dKV/dScore match eager within the forward-style
    tolerances instead of bitwise (thresholds calibrated on this suite's input
    distribution; they stay bitwise run-to-run, and the fp64-oracle parity assertion
    below keeps the accuracy honest on the tested finite-intermediate inputs).
    """
    if ratio == 4:
        assert torch.equal(gkv, ref_kv), "dKV must be bit-identical to the fp32 reference at ratio=4"
        assert torch.equal(gs, ref_s), "dScore must be bit-identical to the fp32 reference at ratio=4"
        return
    for name, fused_t, ref_t in (("dKV", gkv, ref_kv), ("dScore", gs, ref_s)):
        diff = (fused_t.float() - ref_t.float()).abs()
        n_diff = (fused_t != ref_t).sum().item()
        assert n_diff <= max(1, int(0.001 * fused_t.numel())), (name, n_diff)
        assert diff.max().item() <= 1.6e-2, (name, diff.max().item())


_SHAPES = [
    # (lens, head_dim, ratio, coff)
    pytest.param([2048], 128, 4, 2, id="b1-d128-r4"),
    pytest.param([1023, 2048, 509], 128, 4, 2, id="ragged3-d128-r4"),
    pytest.param([2048], 512, 4, 2, id="b1-d512-r4"),
    pytest.param([3, 515, 1024, 129], 128, 4, 2, id="short-seg-d128-r4"),
    # odd head_dim exercises the scalar (vec == 1) forward layout; even head_dims all
    # take the vectorized (vec == 2) one.
    pytest.param([260], 65, 4, 2, id="b1-d65-odd-r4"),
    # a zero-length segment inside the pack (degenerate cu_seqlens entry).
    pytest.param([64, 0, 253, 3], 128, 4, 2, id="empty-seg-d128-r4"),
    # coff == 1: the non-overlapping window form (win = ratio, own-block tokens only).
    pytest.param([2048], 128, 4, 1, id="b1-d128-r4-coff1"),
    pytest.param([1023, 2048, 509], 128, 4, 1, id="ragged3-d128-r4-coff1"),
    pytest.param([2048], 512, 4, 1, id="b1-d512-r4-coff1"),
    pytest.param([3, 515, 1024, 129], 128, 4, 1, id="short-seg-d128-r4-coff1"),
    pytest.param([260], 65, 4, 1, id="b1-d65-odd-r4-coff1"),
    pytest.param([64, 0, 253, 3], 128, 4, 1, id="empty-seg-d128-r4-coff1"),
    # ratio=128 (dedicated r128 kernels; coff {1, 2} x head_dim {128, 512}). The edge
    # pack covers zero-block segments (127, 3), a literal empty segment, an
    # exactly-one-block segment (128), a 1-token tail (129) and other tails.
    pytest.param([8192], 128, 128, 1, id="b1x8192-d128-r128c1"),
    pytest.param([1023, 2048, 509], 128, 128, 1, id="ragged3-d128-r128c1"),
    pytest.param([127, 8192, 0, 129, 128, 3, 515, 1024], 128, 128, 1, id="edgepack-d128-r128c1"),
    pytest.param([8192], 128, 128, 2, id="b1x8192-d128-r128c2"),
    pytest.param([127, 8192, 0, 129, 128, 3, 515, 1024], 128, 128, 2, id="edgepack-d128-r128c2"),
    pytest.param([2048, 509], 512, 128, 1, id="ragged2-d512-r128c1"),
    pytest.param([8192], 512, 128, 2, id="b1x8192-d512-r128c2"),
]


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("lens,d,ratio,coff", _SHAPES)
def test_numerics_vs_references(lens, d, ratio, coff):
    """Fused fwd+bwd vs fp32-eager (bitwise dKV/dScore at ratio=4, tolerance at
    ratio=128), upstream eager, and fp64 oracle."""
    _require_sm100()
    kv, score, ape, cu, cuc, total_comp, go = _make_inputs(lens, d, ratio, coff)

    r_fused = _run_fused(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go)
    r_fp32 = _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode="fp32")
    r_up = _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode="upstream")
    r_fp64 = _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode="fp64")

    # vs fp32-intermediate eager reference (the fused kernels' numerics contract):
    # dKV / dScore bit-identical at ratio=4 / within the forward-style tolerances at
    # ratio=128 (see _assert_grads_vs_fp32); forward within one bf16 rounding step on
    # a tiny fraction of elements; dAPE within fp32 atomics reorder noise.
    _assert_grads_vs_fp32(r_fused[1], r_fused[2], r_fp32[1], r_fp32[2], ratio)
    fwd_diff = (r_fused[0].float() - r_fp32[0].float()).abs()
    n_diff = (r_fused[0] != r_fp32[0]).sum().item()
    assert n_diff <= max(1, int(0.001 * r_fused[0].numel())), n_diff
    assert fwd_diff.max().item() <= 1.6e-2
    assert (r_fused[3] - r_fp32[3]).abs().max().item() <= 1e-3

    # vs the verbatim upstream eager numerics: not bit-identical (the eager path rounds
    # softmax weights to bf16 and multiplies in bf16), but close.
    for fused_t, up_t in zip(r_fused, r_up):
        assert torch.allclose(fused_t.float(), up_t.float(), rtol=0, atol=0.1)

    # vs the fp64 oracle. ratio=128: the contract's fp64-parity gate — per tensor the
    # fused output must be at least as close to the oracle as the FP32-INTERMEDIATE
    # eager reference, the comparator the contract names (comparing against the
    # bf16-weight upstream path instead would be materially looser). ratio=4 keeps its
    # historical check against the upstream numerics it replaced (its contract pins
    # dKV/dScore bitwise-to-fp32-eager above and has no fp64-parity clause).
    eager_ref = r_fp32 if ratio == 128 else r_up
    for i in range(4):
        err_fused = (r_fused[i].double() - r_fp64[i].double()).abs().max().item()
        err_eager = (eager_ref[i].double() - r_fp64[i].double()).abs().max().item()
        assert err_fused <= err_eager * (1 + 1e-6) + 1e-4, (i, err_fused, err_eager)


@pytest.mark.L0
@pytest.mark.parametrize("coff", [1, 2])
def test_replay_determinism(coff):
    """Forward, dKV and dScore replay bitwise identically run to run (dAPE is exempt)."""
    _require_sm100()
    kv, score, ape, cu, cuc, total_comp, go = _make_inputs([1023, 2048, 509], 128, 4, coff)
    runs = [_run_fused(kv, score, ape, cu, cuc, total_comp, 4, 128, coff, go) for _ in range(3)]
    for other in runs[1:]:
        assert torch.equal(runs[0][0], other[0])
        assert torch.equal(runs[0][1], other[1])
        assert torch.equal(runs[0][2], other[2])
        # dAPE is accumulated with fp32 atomics; equality is not guaranteed, closeness is.
        assert torch.allclose(runs[0][3], other[3], rtol=0, atol=1e-3)


@pytest.mark.L0
def test_replay_determinism_r128():
    """ratio=128 mirror of test_replay_determinism (fixed chunk boundaries, no dKV/dScore atomics)."""
    _require_sm100()
    kv, score, ape, cu, cuc, total_comp, go = _make_inputs([1023, 2048, 509], 128, 128, 1)
    runs = [_run_fused(kv, score, ape, cu, cuc, total_comp, 128, 128, 1, go) for _ in range(3)]
    for other in runs[1:]:
        assert torch.equal(runs[0][0], other[0])
        assert torch.equal(runs[0][1], other[1])
        assert torch.equal(runs[0][2], other[2])
        assert torch.allclose(runs[0][3], other[3], rtol=0, atol=1e-3)


# ---------------------------------------------------------------------------
# ratio=128 dispatch envelope: every shipped schedule bucket
# ---------------------------------------------------------------------------


def _r128_module():
    _import_compressor()
    from cudnn.csa.compressor import compressor_sm100_r128

    return compressor_sm100_r128


# Expected shipped schedules per (coff, d, nb_total) bucket — hardcoded on purpose: an
# edit that silently changes any shipped launch geometry or bucket boundary must fail
# here. Forward tuples are (vec, tchunks, threads_x, twophase, fastexp); backward
# tuples are (vec, tchunks, threads_x, fastexp).
_FWD_BUCKETS = [
    (1, 128, 64, (2, 8, 32, False, False)),  # small
    (1, 128, 256, (2, 4, 32, False, False)),  # default
    (1, 128, 1024, (4, 4, 32, True, True)),  # large: two-phase + fast exp
    (2, 128, 64, (2, 16, 32, False, True)),  # small (fast exp)
    (2, 128, 256, (2, 8, 32, False, True)),  # default (fast exp; no large entry)
    (1, 512, 256, (2, 4, 32, False, False)),  # default (no small entry)
    (1, 512, 1024, (4, 4, 32, False, True)),  # large
    (2, 512, 64, (2, 8, 32, False, True)),  # small
    (2, 512, 256, (4, 4, 32, False, False)),  # default
    (2, 512, 1024, (4, 4, 32, False, True)),  # large
]
_BWD_BUCKETS = [
    (1, 128, 64, (1, 8, 32, False)),  # small pack: vec=1, exact exp
    (1, 128, 256, (2, 8, 32, True)),  # default (fast exp)
    (2, 128, 64, (1, 8, 32, False)),  # small
    (2, 128, 256, (2, 8, 32, True)),  # default
    (1, 512, 256, (2, 4, 32, True)),  # default (tchunks=4 at coff=1, d>=512)
    (2, 512, 64, (2, 8, 32, True)),  # no bwd small entry at d=512 -> default
    (2, 512, 256, (2, 8, 32, True)),  # default
]


@pytest.mark.L0
def test_r128_dispatch_boundaries():
    """Schedule selection flips exactly at the documented nb_total bucket boundaries."""
    M = _r128_module()
    for coff, d in ((1, 128), (2, 128), (1, 512), (2, 512)):
        small = M._fwd_schedule_r128(128, d, coff, M._SMALL_NB_MAX)
        first_default = M._fwd_schedule_r128(128, d, coff, M._SMALL_NB_MAX + 1)
        last_default = M._fwd_schedule_r128(128, d, coff, M._LARGE_NB_MIN - 1)
        large = M._fwd_schedule_r128(128, d, coff, M._LARGE_NB_MIN)
        assert first_default == last_default, (coff, d)  # one default bucket in between
        if (coff, d) in M._SMALL_SCHEDULES:
            assert small != first_default, (coff, d)
        else:
            assert small == first_default, (coff, d)
        if (coff, d) in M._LARGE_SCHEDULES:
            assert large != last_default, (coff, d)
        else:
            assert large == last_default, (coff, d)
        bwd_small = M._bwd_schedule_r128(128, d, coff, M._BWD_SMALL_NB_MAX)
        bwd_default = M._bwd_schedule_r128(128, d, coff, M._BWD_SMALL_NB_MAX + 1)
        if (coff, d) in M._BWD_SMALL_SCHEDULES:
            assert bwd_small != bwd_default, (coff, d)
        else:
            assert bwd_small == bwd_default, (coff, d)


@pytest.mark.L0
def test_r128_dispatch_expected_schedules():
    """Every shipped (config, nb_total bucket) selects exactly the audited schedule."""
    M = _r128_module()
    for coff, d, nb, expected in _FWD_BUCKETS:
        assert M._fwd_schedule_r128(128, d, coff, nb) == expected, ("fwd", coff, d, nb)
    for coff, d, nb, expected in _BWD_BUCKETS:
        assert M._bwd_schedule_r128(128, d, coff, nb) == expected, ("bwd", coff, d, nb)


_ENVELOPE_CASES = [
    # (coff, d, nb_rows): one case per unique shipped (config, schedule) kernel, fwd
    # and bwd together — the 10 forward + 6 backward shipped kernels across 10 cases
    # (single-segment packs of nb_rows * ratio tokens land in the intended buckets).
    pytest.param(1, 128, 64, id="c1d128-small"),
    pytest.param(1, 128, 256, id="c1d128-default"),
    pytest.param(1, 128, 1024, id="c1d128-large"),
    pytest.param(2, 128, 64, id="c2d128-small"),
    pytest.param(2, 128, 256, id="c2d128-default"),
    pytest.param(1, 512, 256, id="c1d512-default"),
    pytest.param(1, 512, 1024, id="c1d512-large"),
    pytest.param(2, 512, 64, id="c2d512-small"),
    pytest.param(2, 512, 256, id="c2d512-default"),
    pytest.param(2, 512, 1024, id="c2d512-large"),
]


@pytest.mark.L1
@pytest.mark.parametrize("coff,d,nb_rows", _ENVELOPE_CASES)
def test_r128_envelope_execution(coff, d, nb_rows):
    """Execute every shipped schedule bucket once against the full contract: tolerance
    vs the fp32 eager reference, fp64-oracle parity, and 2-run bitwise determinism —
    covering the fast-exp, two-phase and vec=1 variants the (smaller) L0 shapes never
    reach."""
    _require_sm100()
    M = _r128_module()
    ratio = 128
    # The shape must land in the intended bucket for BOTH directions (the wrappers
    # select schedules through these exact functions).
    exp_fwd = {(c, dd, n): e for c, dd, n, e in _FWD_BUCKETS}[(coff, d, nb_rows)]
    assert M._fwd_schedule_r128(ratio, d, coff, nb_rows) == exp_fwd
    exp_bwd = {(c, dd, n): e for c, dd, n, e in _BWD_BUCKETS}.get((coff, d, nb_rows))
    if exp_bwd is not None:
        assert M._bwd_schedule_r128(ratio, d, coff, nb_rows) == exp_bwd

    kv, score, ape, cu, cuc, total_comp, go = _make_inputs([nb_rows * ratio], d, ratio, coff)
    assert total_comp == nb_rows
    r1 = _run_fused(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go)
    r2 = _run_fused(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go)
    # Bitwise run-to-run determinism holds in EVERY bucket (incl. fast exp/two-phase).
    for i in range(3):
        assert torch.equal(r1[i], r2[i]), i
    assert torch.allclose(r1[3], r2[3], rtol=0, atol=1e-3)

    r_fp32 = _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode="fp32")
    _assert_grads_vs_fp32(r1[1], r1[2], r_fp32[1], r_fp32[2], ratio)
    assert (r1[0] != r_fp32[0]).sum().item() <= max(1, int(0.001 * r1[0].numel()))
    assert (r1[0].float() - r_fp32[0].float()).abs().max().item() <= 1.6e-2
    assert (r1[3] - r_fp32[3]).abs().max().item() <= 1e-3

    # fp64-oracle parity in EVERY shipped bucket (the contract's accuracy clause on
    # finite-intermediate inputs), so the large-bucket schedules are held to the full
    # contract here too.
    r_fp64 = _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode="fp64")
    for i in range(4):
        err_fused = (r1[i].double() - r_fp64[i].double()).abs().max().item()
        err_eager = (r_fp32[i].double() - r_fp64[i].double()).abs().max().item()
        assert err_fused <= err_eager * (1 + 1e-6) + 1e-4, (i, err_fused, err_eager)


_PADDING_SHAPES = [
    # (lens, head_dim, ratio, coff, pad); the first case has a LEADING segment shorter
    # than ratio (0 compressed blocks), so padding rows gather tokens [0, ratio) that
    # span a segment boundary -- exactly like the eager gather.
    ([3, 515, 1024, 129], 128, 4, 2, 8),
    ([1023, 2048, 509], 128, 4, 2, 8),
    ([3, 515, 1024, 129], 128, 4, 1, 8),
    ([1023, 2048, 509], 128, 4, 1, 8),
]


@pytest.mark.L0
@pytest.mark.parametrize("lens,d,ratio,coff,pad", _PADDING_SHAPES)
def test_static_capacity_padding(lens, d, ratio, coff, pad):
    """Static-capacity padding rows: eager-matching forward, ignored padding gradients."""
    _require_sm100()
    kv, score, ape, cu, cuc, total_true, _ = _make_inputs(lens, d, ratio, coff)
    capacity = total_true + pad
    gen = torch.Generator(device="cpu").manual_seed(7)
    go = torch.randn(capacity, 1, d, generator=gen, dtype=torch.float32)
    go = go.to(torch.bfloat16).cuda()
    go_zero_pad = go.clone()
    go_zero_pad[total_true:] = 0

    r_fused = _run_fused(kv, score, ape, cu, cuc, capacity, ratio, d, coff, go)
    r_fp32 = _run_eager(kv, score, ape, cu, cuc, capacity, ratio, d, coff, go_zero_pad, mode="fp32")

    # Forward: padding rows replicate row 0's window exactly like the eager code, so the
    # full padded output (valid + padding rows) obeys the same criteria as the unpadded
    # comparison.
    assert (r_fused[0] != r_fp32[0]).sum().item() <= max(1, int(0.001 * r_fused[0].numel()))
    assert (r_fused[0].float() - r_fp32[0].float()).abs().max().item() <= 1.6e-2

    # Backward: incoming gradients on padding rows are ignored by design -- the fused
    # gradients (computed with NONZERO padding-row grads) match the eager reference run
    # with zeroed padding-row grads bit-for-bit on dKV/dScore.
    assert torch.equal(r_fused[1], r_fp32[1])
    assert torch.equal(r_fused[2], r_fp32[2])
    assert (r_fused[3] - r_fp32[3]).abs().max().item() <= 1e-3

    # And explicitly: nonzero vs zero padding-row grads produce identical fused grads.
    r_fused_zero = _run_fused(kv, score, ape, cu, cuc, capacity, ratio, d, coff, go_zero_pad)
    assert torch.equal(r_fused[1], r_fused_zero[1])
    assert torch.equal(r_fused[2], r_fused_zero[2])


@pytest.mark.L0
@pytest.mark.parametrize("coff", [1, 2])
def test_empty_output(coff):
    """total_comp == 0 launches nothing and returns well-formed empty/zero tensors."""
    _require_sm100()
    compressor = _import_compressor()
    d, ratio = 128, 4
    w = coff * d
    kv = torch.randn(2, w, device="cuda").to(torch.bfloat16)
    score = torch.randn(2, w, device="cuda").to(torch.bfloat16)
    ape = torch.randn(ratio, w, device="cuda")
    cu = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    cuc = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    out = compressor.csa_compressor_forward_wrapper(kv, score, ape, cu, cuc, ratio=ratio, head_dim=d, coff=coff, total_comp=0)["out"]
    assert out.shape == (0, d) and out.dtype == torch.bfloat16
    grads = compressor.csa_compressor_backward_wrapper(kv, score, ape, cu, cuc, out, ratio=ratio, head_dim=d, coff=coff)
    assert grads["grad_kv"].abs().sum().item() == 0
    assert grads["grad_score"].abs().sum().item() == 0
    assert grads["grad_ape"].abs().sum().item() == 0


@pytest.mark.L0
@pytest.mark.parametrize("coff", [1, 2])
def test_backward_wrapper_zeros_when_no_blocks(coff):
    """Multi-segment packs where NO segment reaches ratio tokens: total_comp == 0, so
    the kernel cannot launch and the wrapper must hand back exact-zero grads (the
    host-side zeros fallback behind the uninitialized-buffer optimization)."""
    _require_sm100()
    compressor = _import_compressor()
    d, ratio = 128, 4
    w = coff * d
    lens = [3, 2, 1]
    total = sum(lens)
    kv = torch.randn(total, w, device="cuda").to(torch.bfloat16)
    score = torch.randn(total, w, device="cuda").to(torch.bfloat16)
    ape = torch.randn(ratio, w, device="cuda")
    cu = torch.tensor([0, 3, 5, 6], dtype=torch.int32, device="cuda")
    cuc = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device="cuda")
    go = torch.empty(0, d, dtype=torch.bfloat16, device="cuda")
    grads = compressor.csa_compressor_backward_wrapper(kv, score, ape, cu, cuc, go, ratio=ratio, head_dim=d, coff=coff)
    assert grads["grad_kv"].shape == kv.shape and grads["grad_kv"].abs().sum().item() == 0
    assert grads["grad_score"].shape == score.shape and grads["grad_score"].abs().sum().item() == 0
    assert grads["grad_ape"].abs().sum().item() == 0


def _never_consumed_mask(lens, total_tokens, d, ratio, coff):
    """Boolean ``(total_tokens, coff * d)`` mask of dKV/dScore slots no output row reads.

    Mirrors the kernel's zero-write ownership classes exactly: per-segment tail tokens
    (``seqlen % ratio``, all columns), whole segments shorter than ``ratio``, for
    ``coff == 2`` the first-half columns of each segment's LAST block's own tokens (no
    next in-segment block consumes them), and token-capacity padding beyond
    ``cu_seqlens[-1]``.
    """
    w = coff * d
    mask = torch.zeros(total_tokens, w, dtype=torch.bool)
    pos = 0
    for seg_len in lens:
        nb = seg_len // ratio
        if nb == 0:
            mask[pos : pos + seg_len] = True
        else:
            mask[pos + nb * ratio : pos + seg_len] = True
            if coff == 2:
                mask[pos + (nb - 1) * ratio : pos + nb * ratio, :d] = True
        pos += seg_len
    mask[pos:] = True
    return mask


_CANARY_SHAPES = [
    # (lens, head_dim, ratio, coff, pad, tok_pad) — every never-consumed dKV/dScore
    # slot class must be hit: segment tails (seqlen % ratio), the last block's
    # first-half columns (coff == 2 only), whole segments shorter than ratio (zero
    # blocks), static-capacity padding rows (pad > 0 extra grad_out rows), and static
    # token-capacity padding of the gradient buffers themselves (tok_pad > 0 tokens
    # beyond cu_seqlens[-1]).
    pytest.param([2048], 128, 4, 2, 0, 0, id="b1-d128"),
    pytest.param([1023, 2048, 509], 128, 4, 2, 0, 0, id="ragged3-d128"),
    pytest.param([3, 515, 1024, 129], 128, 4, 2, 0, 0, id="short-seg-d128"),
    pytest.param([5, 6, 7], 128, 4, 2, 0, 0, id="all-tiny-d128"),
    pytest.param([1023, 2048, 509], 512, 4, 2, 0, 0, id="ragged3-d512"),
    pytest.param([3, 515, 1024, 129], 128, 4, 2, 8, 0, id="short-seg-d128-padded"),
    pytest.param([1023, 2048, 509], 128, 4, 2, 8, 0, id="ragged3-d128-padded"),
    pytest.param([1023, 2048, 509], 128, 4, 2, 0, 37, id="ragged3-d128-tokpad"),
    pytest.param([3, 515, 1024, 129], 128, 4, 2, 8, 21, id="short-seg-d128-padded-tokpad"),
    # ratio=4, coff == 1 (own-block window): same shapes as the coff=2 rows above; the
    # zero classes drop the first-half-column class (no overlap halves at coff=1).
    pytest.param([2048], 128, 4, 1, 0, 0, id="b1-d128-coff1"),
    pytest.param([1023, 2048, 509], 128, 4, 1, 0, 0, id="ragged3-d128-coff1"),
    pytest.param([3, 515, 1024, 129], 128, 4, 1, 0, 0, id="short-seg-d128-coff1"),
    pytest.param([5, 6, 7], 128, 4, 1, 0, 0, id="all-tiny-d128-coff1"),
    pytest.param([1023, 2048, 509], 512, 4, 1, 0, 0, id="ragged3-d512-coff1"),
    pytest.param([3, 515, 1024, 129], 128, 4, 1, 8, 0, id="short-seg-d128-padded-coff1"),
    pytest.param([1023, 2048, 509], 128, 4, 1, 8, 0, id="ragged3-d128-padded-coff1"),
    pytest.param([1023, 2048, 509], 128, 4, 1, 0, 37, id="ragged3-d128-tokpad-coff1"),
    pytest.param([3, 515, 1024, 129], 128, 4, 1, 8, 21, id="short-seg-d128-padded-tokpad-coff1"),
    # ratio=128: the zero classes are up to 127 tokens each (tails, zero-block
    # segments) plus the coff=2 last-block first-half (128 rows).
    pytest.param([1023, 2048, 509], 128, 128, 1, 0, 0, id="ragged3-d128-r128c1"),
    pytest.param([127, 8192, 0, 129, 128, 3, 515, 1024], 128, 128, 1, 8, 21, id="edgepack-d128-r128c1-padded-tokpad"),
    pytest.param([127, 8192, 0, 129, 128, 3, 515, 1024], 128, 128, 2, 8, 21, id="edgepack-d128-r128c2-padded-tokpad"),
    pytest.param([2048, 509], 512, 128, 2, 8, 21, id="ragged2-d512-r128c2-padded-tokpad"),
]


@pytest.mark.L0
@pytest.mark.parametrize("lens,d,ratio,coff,pad,tok_pad", _CANARY_SHAPES)
def test_backward_fills_uninitialized_buffers(lens, d, ratio, coff, pad, tok_pad):
    """NaN-canary: the backward kernel fully overwrites garbage dKV/dScore buffers.

    The kernel writes exact zeros to every never-consumed slot itself (there are no
    separate zero-fill kernels anymore), so running it into NaN-poisoned buffers must
    produce bitwise the same dKV/dScore as running it into zero-initialized buffers.
    """
    _require_sm100()
    compressor = _import_compressor()
    kv, score, ape, cu, cuc, total_true, _ = _make_inputs(lens, d, ratio, coff)
    total_comp = total_true + pad
    gen = torch.Generator(device="cpu").manual_seed(11)
    go = torch.randn(total_comp, d, generator=gen, dtype=torch.float32).to(torch.bfloat16).cuda()
    total = kv.shape[0] + tok_pad
    kv2 = torch.cat([kv.view(kv.shape[0], -1), torch.randn(tok_pad, coff * d, generator=gen, dtype=torch.float32).to(torch.bfloat16).cuda()])
    score2 = torch.cat([score.view(score.shape[0], -1), torch.randn(tok_pad, coff * d, generator=gen, dtype=torch.float32).to(torch.bfloat16).cuda()])

    bwd = compressor.CSACompressorBackward(
        sample_kv=kv2,
        sample_score=score2,
        sample_ape=ape,
        sample_cu_seqlens=cu,
        sample_cu_seqlens_comp=cuc,
        sample_out=torch.empty(total_comp, d, dtype=torch.bfloat16, device="meta"),
        ratio=ratio,
        coff=coff,
    )
    assert bwd.check_support()
    bwd.compile()

    def run(poison):
        """One backward into poisoned (NaN) or zeroed grad buffers; returns the grads."""
        grad_kv = torch.empty_like(kv2)
        grad_score = torch.empty_like(score2)
        if poison:
            grad_kv.fill_(float("nan"))
            grad_score.fill_(float("nan"))
        else:
            grad_kv.zero_()
            grad_score.zero_()
        grad_ape = torch.zeros_like(ape)
        bwd.execute(kv2, score2, ape, cu, cuc, go, grad_kv, grad_score, grad_ape)
        torch.cuda.synchronize()
        return grad_kv, grad_score, grad_ape

    gkv_ref, gs_ref, gape_ref = run(poison=False)
    gkv_nan, gs_nan, gape_nan = run(poison=True)
    assert not torch.isnan(gkv_nan).any(), "unwritten dKV slots survived (NaN canary)"
    assert not torch.isnan(gs_nan).any(), "unwritten dScore slots survived (NaN canary)"
    assert torch.equal(gkv_nan, gkv_ref)
    assert torch.equal(gs_nan, gs_ref)
    assert torch.allclose(gape_nan, gape_ref, rtol=0, atol=1e-3)

    # And the zero-slot pattern matches autograd: never-consumed slots are exact zeros,
    # exactly as the fp32 eager reference computes them (bitwise at ratio=4, tolerance
    # at ratio=128 — the zero slots themselves are exact in both). (The fused backward
    # ignores incoming gradients on static-capacity padding rows by design, so the
    # eager reference runs with those rows zeroed.)
    go_ref = go.clone()
    go_ref[total_true:] = 0
    r_fp32 = _run_eager(
        kv2.view(total, 1, -1),
        score2.view(total, 1, -1),
        ape,
        cu,
        cuc,
        total_comp,
        ratio,
        d,
        coff,
        go_ref.view(total_comp, 1, d),
        mode="fp32",
    )
    _assert_grads_vs_fp32(gkv_nan.view_as(r_fp32[1]), gs_nan.view_as(r_fp32[2]), r_fp32[1], r_fp32[2], ratio)

    # The zero-owned classes themselves must be EXACT zeros, not merely within the
    # ratio=128 tolerance (a deterministic small nonzero written into a never-consumed
    # slot would otherwise pass). The mask is sanity-checked against the eager
    # reference first: autograd produces exact zeros on precisely these slots.
    mask = _never_consumed_mask(lens, total, d, ratio, coff).cuda()
    assert (r_fp32[1].view(total, -1)[mask] == 0).all(), "mask does not match the eager zero slots (dKV)"
    assert (r_fp32[2].view(total, -1)[mask] == 0).all(), "mask does not match the eager zero slots (dScore)"
    assert (gkv_nan.view(total, -1)[mask] == 0).all(), "never-consumed dKV slots must be exact zeros"
    assert (gs_nan.view(total, -1)[mask] == 0).all(), "never-consumed dScore slots must be exact zeros"


# ---------------------------------------------------------------------------
# Deterministic mode
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("coff", [1, 2])
def test_backward_rejects_deterministic_mode(coff):
    """The backward raises under torch.use_deterministic_algorithms (dAPE fp32 atomics)."""
    _require_sm100()
    compressor = _import_compressor()
    kv, score, ape, cu, cuc, total_comp, go = _make_inputs([512, 256], 128, 4, coff)
    total = kv.shape[0]
    # Forward is deterministic and keeps working.
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        out = compressor.csa_compressor_forward_wrapper(
            kv.view(total, -1), score.view(total, -1), ape, cu, cuc, ratio=4, head_dim=128, coff=coff, total_comp=total_comp
        )["out"]
        assert out.shape == (total_comp, 128)
        with pytest.raises(RuntimeError, match="not deterministic"):
            compressor.csa_compressor_backward_wrapper(
                kv.view(total, -1), score.view(total, -1), ape, cu, cuc, go.view(total_comp, 128), ratio=4, head_dim=128, coff=coff
            )
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)


@pytest.mark.L0
@pytest.mark.parametrize("coff", [1, 2])
def test_backward_warns_in_warn_only_deterministic_mode(coff):
    """warn_only deterministic mode warns (torch semantics) and still runs the backward."""
    _require_sm100()
    compressor = _import_compressor()
    kv, score, ape, cu, cuc, total_comp, go = _make_inputs([512, 256], 128, 4, coff)
    r_ref = _run_fused(kv, score, ape, cu, cuc, total_comp, 4, 128, coff, go)
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with pytest.warns(RuntimeWarning, match="not deterministic") as record:
            grads = compressor.csa_compressor_backward_wrapper(
                kv.view(kv.shape[0], -1), score.view(score.shape[0], -1), ape, cu, cuc, go.view(total_comp, 128), ratio=4, head_dim=128, coff=coff
            )
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)
    # Exactly ONE warning on the wrapper path: execute() is the single check point
    # (the wrapper does not duplicate it).
    det_warnings = [w for w in record if issubclass(w.category, RuntimeWarning) and "not deterministic" in str(w.message)]
    assert len(det_warnings) == 1, f"expected exactly one deterministic-mode warning, got {len(det_warnings)}"
    assert torch.equal(grads["grad_kv"].view_as(r_ref[1]), r_ref[1])
    assert torch.equal(grads["grad_score"].view_as(r_ref[2]), r_ref[2])


# ---------------------------------------------------------------------------
# CUDA graph capture
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("coff", [1, 2])
def test_cuda_graph_capture(coff):
    """Warmup -> capture fwd+bwd -> replay; JIT under capture raises a clear error."""
    _require_sm100()
    compressor = _import_compressor()
    ratio, d = 4, 128
    lens = [512, 256]
    kv, score, ape, cu, cuc, total_true, _ = _make_inputs(lens, d, ratio, coff)
    capacity = total_true + 8  # static capacity, as with CUDA-graph static shapes
    total = kv.shape[0]

    kv_s = kv.view(total, -1).clone()
    score_s = score.view(total, -1).clone()
    ape_s = ape.clone()
    go_s = torch.zeros(capacity, d, device="cuda", dtype=torch.bfloat16)
    go_s[:total_true] = torch.randn(total_true, d, device="cuda").to(torch.bfloat16)

    def _fused_fwd_bwd():
        """One fused forward + backward over the static-capacity buffers."""
        out = compressor.csa_compressor_forward_wrapper(kv_s, score_s, ape_s, cu, cuc, ratio=ratio, head_dim=d, coff=coff, total_comp=capacity)["out"]
        grads = compressor.csa_compressor_backward_wrapper(kv_s, score_s, ape_s, cu, cuc, go_s, ratio=ratio, head_dim=d, coff=coff)
        return out, grads

    # One warmup per configuration on a side stream (JIT-compiles both kernels).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _fused_fwd_bwd()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_c, grads_c = _fused_fwd_bwd()
    gkv_c, gscore_c, gape_c = grads_c["grad_kv"], grads_c["grad_score"], grads_c["grad_ape"]

    # Replay on the same data must reproduce the direct (non-captured) fused results
    # bitwise on forward/dKV/dScore.
    graph.replay()
    torch.cuda.synchronize()
    ref_out, ref_grads = _fused_fwd_bwd()
    torch.cuda.synchronize()
    assert torch.equal(out_c, ref_out)
    assert torch.equal(gkv_c, ref_grads["grad_kv"])
    assert torch.equal(gscore_c, ref_grads["grad_score"])
    assert torch.allclose(gape_c, ref_grads["grad_ape"], rtol=0, atol=1e-3)

    # Replay with new data and a SMALLER device-side true row count (the fixed capacity
    # stays static, cu/cuc contents change) -- the graph-replayed gradients must match
    # the fp32 eager reference bitwise on dKV/dScore.
    lens2 = [384, 128]
    kv2, score2, ape2, cu2, cuc2, total2, _ = _make_inputs(lens2, d, ratio, coff, seed=99)
    n2 = sum(lens2)
    kv_s.zero_()
    score_s.zero_()
    kv_s[:n2] = kv2.view(n2, -1)
    score_s[:n2] = score2.view(n2, -1)
    ape_s.copy_(ape2)
    cu.copy_(cu2)
    cuc.copy_(cuc2)
    go_s.zero_()
    go_s[:total2] = torch.randn(total2, d, device="cuda").to(torch.bfloat16)
    graph.replay()
    torch.cuda.synchronize()
    r_fp32 = _run_eager(
        kv_s.view(-1, 1, coff * d),
        score_s.view(-1, 1, coff * d),
        ape_s,
        cu,
        cuc,
        capacity,
        ratio,
        d,
        coff,
        go_s.view(capacity, 1, d),
        mode="fp32",
    )
    assert torch.equal(gkv_c.view_as(r_fp32[1]), r_fp32[1])
    assert torch.equal(gscore_c.view_as(r_fp32[2]), r_fp32[2])
    # The narrowed replay is checked on ALL FOUR outputs, not just dKV/dScore:
    # forward vs eager (contract tolerance), dAPE vs eager (atomics tolerance), and
    # the whole replay vs a direct (non-captured) fused call on the same mutated
    # inputs — bitwise on forward/dKV/dScore.
    assert (out_c.view_as(r_fp32[0]) != r_fp32[0]).sum().item() <= max(1, int(0.001 * r_fp32[0].numel()))
    assert (out_c.view_as(r_fp32[0]).float() - r_fp32[0].float()).abs().max().item() <= 1.6e-2
    assert (gape_c - r_fp32[3]).abs().max().item() <= 1e-3
    direct_out, direct_grads = _fused_fwd_bwd()
    torch.cuda.synchronize()
    assert torch.equal(out_c, direct_out)
    assert torch.equal(gkv_c, direct_grads["grad_kv"])
    assert torch.equal(gscore_c, direct_grads["grad_score"])

    # A first call for a NEW configuration under capture must raise loudly instead of
    # JIT-compiling (which is not capture-safe). head_dim 192 is used by no other test
    # in this module (and kernels are compiled per (ratio, head_dim, coff)), so this
    # configuration is guaranteed to be uncompiled regardless of test execution order.
    d_new = 192
    kv3 = torch.randn(256, coff * d_new, device="cuda").to(torch.bfloat16)
    score3 = torch.randn(256, coff * d_new, device="cuda").to(torch.bfloat16)
    ape3 = torch.randn(ratio, coff * d_new, device="cuda")
    cu3 = torch.tensor([0, 256], dtype=torch.int32, device="cuda")
    cuc3 = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
    graph2 = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        with torch.cuda.graph(graph2):
            compressor.csa_compressor_forward_wrapper(kv3, score3, ape3, cu3, cuc3, ratio=ratio, head_dim=d_new, coff=coff, total_comp=64)
    # The CUDA context must remain usable after the aborted capture.
    torch.cuda.synchronize()
    probe = torch.ones(8, device="cuda")
    assert probe.sum().item() == 8


@pytest.mark.L0
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cuda_graph_capture_r128():
    """ratio=128 mirror of test_cuda_graph_capture: warmup -> capture fwd+bwd -> replay.

    The ratio=128 backward's rows_per_cta launch parameter is derived from the static
    row capacity on the host, so capture/replay must reproduce the direct call bitwise
    on forward/dKV/dScore exactly like the ratio=4 path.
    """
    _require_sm100()
    compressor = _import_compressor()
    ratio, d, coff = 128, 128, 1
    lens = [640, 259]  # 5 + 2 blocks, 3-token tail
    kv, score, ape, cu, cuc, total_true, _ = _make_inputs(lens, d, ratio, coff)
    capacity = total_true + 8
    total = kv.shape[0]

    kv_s = kv.view(total, -1).clone()
    score_s = score.view(total, -1).clone()
    ape_s = ape.clone()
    go_s = torch.zeros(capacity, d, device="cuda", dtype=torch.bfloat16)
    go_s[:total_true] = torch.randn(total_true, d, device="cuda").to(torch.bfloat16)

    def _fused_fwd_bwd():
        out = compressor.csa_compressor_forward_wrapper(kv_s, score_s, ape_s, cu, cuc, ratio=ratio, head_dim=d, coff=coff, total_comp=capacity)["out"]
        grads = compressor.csa_compressor_backward_wrapper(kv_s, score_s, ape_s, cu, cuc, go_s, ratio=ratio, head_dim=d, coff=coff)
        return out, grads

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _fused_fwd_bwd()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_c, grads_c = _fused_fwd_bwd()
    gkv_c, gscore_c, gape_c = grads_c["grad_kv"], grads_c["grad_score"], grads_c["grad_ape"]

    graph.replay()
    torch.cuda.synchronize()
    ref_out, ref_grads = _fused_fwd_bwd()
    torch.cuda.synchronize()
    assert torch.equal(out_c, ref_out)
    assert torch.equal(gkv_c, ref_grads["grad_kv"])
    assert torch.equal(gscore_c, ref_grads["grad_score"])
    assert torch.allclose(gape_c, ref_grads["grad_ape"], rtol=0, atol=1e-3)

    # Replay with new data and a SMALLER device-side true row count (the fixed capacity
    # stays static, cu/cuc contents change) -- the graph-replayed gradients must match
    # the fp32 eager reference within the ratio=128 tolerance contract (capture/replay
    # itself stays bitwise vs the direct call, asserted above).
    lens2 = [384, 128]
    kv2, score2, ape2, cu2, cuc2, total2, _ = _make_inputs(lens2, d, ratio, coff, seed=99)
    n2 = sum(lens2)
    kv_s.zero_()
    score_s.zero_()
    kv_s[:n2] = kv2.view(n2, -1)
    score_s[:n2] = score2.view(n2, -1)
    ape_s.copy_(ape2)
    cu.copy_(cu2)
    cuc.copy_(cuc2)
    go_s.zero_()
    go_s[:total2] = torch.randn(total2, d, device="cuda").to(torch.bfloat16)
    graph.replay()
    torch.cuda.synchronize()
    r_fp32 = _run_eager(
        kv_s.view(-1, 1, coff * d),
        score_s.view(-1, 1, coff * d),
        ape_s,
        cu,
        cuc,
        capacity,
        ratio,
        d,
        coff,
        go_s.view(capacity, 1, d),
        mode="fp32",
    )
    _assert_grads_vs_fp32(gkv_c.view_as(r_fp32[1]), gscore_c.view_as(r_fp32[2]), r_fp32[1], r_fp32[2], ratio)
    # The narrowed replay is checked on ALL FOUR outputs, not just dKV/dScore:
    # forward vs eager (contract tolerance), dAPE vs eager (atomics tolerance), and
    # the whole replay vs a direct (non-captured) fused call on the same mutated
    # inputs — bitwise on forward/dKV/dScore.
    assert (out_c.view_as(r_fp32[0]) != r_fp32[0]).sum().item() <= max(1, int(0.001 * r_fp32[0].numel()))
    assert (out_c.view_as(r_fp32[0]).float() - r_fp32[0].float()).abs().max().item() <= 1.6e-2
    assert (gape_c - r_fp32[3]).abs().max().item() <= 1e-3
    direct_out, direct_grads = _fused_fwd_bwd()
    torch.cuda.synchronize()
    assert torch.equal(out_c, direct_out)
    assert torch.equal(gkv_c, direct_grads["grad_kv"])
    assert torch.equal(gscore_c, direct_grads["grad_score"])


# ---------------------------------------------------------------------------
# check_support boundaries
# ---------------------------------------------------------------------------


def _meta(shape, dtype, stride=None):
    """Metadata-only sample tensor (meta device) for check_support tests."""
    if stride is None:
        stride = []
        acc = 1
        for s in reversed(shape):
            stride.append(acc)
            acc *= s
        stride = tuple(reversed(stride))
    return torch.empty_strided(shape, stride, dtype=dtype, device="meta")


def _meta_samples(
    total=512,
    d=128,
    ratio=4,
    coff=2,
    n_seg=2,
    total_comp=None,
    kv_dtype=torch.bfloat16,
    ape_dtype=torch.float32,
    cu_dtype=torch.int32,
    out_dtype=torch.bfloat16,
    score_shape=None,
    cuc_len=None,
    kv_stride=None,
):
    """Consistent meta-device sample-tensor kwargs, with per-field overrides for negatives."""
    w = coff * d
    if total_comp is None:
        total_comp = total // ratio
    kv = _meta((total, w), kv_dtype, stride=kv_stride)
    score = _meta(score_shape or (total, w), kv_dtype)
    ape = _meta((ratio, w), ape_dtype)
    cu = _meta((n_seg + 1,), cu_dtype)
    cuc = _meta((cuc_len or (n_seg + 1),), cu_dtype)
    out = _meta((total_comp, d), out_dtype)
    return dict(
        sample_kv=kv,
        sample_score=score,
        sample_ape=ape,
        sample_cu_seqlens=cu,
        sample_cu_seqlens_comp=cuc,
        sample_out=out,
    )


@pytest.mark.L0
@pytest.mark.parametrize("coff", [1, 2])
def test_check_support_accepts_envelope(coff):
    """Metadata-only samples inside the validated envelope pass check_support."""
    _require_sm100()
    compressor = _import_compressor()
    for cls in (compressor.CSACompressorForward, compressor.CSACompressorBackward):
        api = cls(**_meta_samples(coff=coff), ratio=4, coff=coff)
        assert api.check_support() is True
        assert api.head_dim == 128 and api.total_tokens == 512 and api.total_comp == 128


@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 512])
@pytest.mark.parametrize("coff", [1, 2])
def test_check_support_accepts_envelope_r128(d, coff):
    """ratio=128 envelope: coff {1, 2} x head_dim {128, 512} pass check_support."""
    _require_sm100()
    compressor = _import_compressor()
    for cls in (compressor.CSACompressorForward, compressor.CSACompressorBackward):
        api = cls(**_meta_samples(total=1024, d=d, ratio=128, coff=coff), ratio=128, coff=coff)
        assert api.check_support() is True
        assert api.head_dim == d and api.total_tokens == 1024 and api.total_comp == 8


@pytest.mark.L0
@pytest.mark.parametrize(
    "kwargs,ctor,match",
    [
        (dict(), dict(ratio=128, coff=3), "ratio=128 supports coff"),
        (dict(), dict(ratio=8, coff=2), "ratio in \\{4, 128\\}"),
        (dict(coff=3), dict(ratio=4, coff=3), "coff in"),
        (dict(), dict(ratio=4, coff=0), "coff in"),
        (dict(kv_dtype=torch.float16), dict(), "kv"),
        (dict(ape_dtype=torch.bfloat16), dict(), "ape"),
        (dict(cu_dtype=torch.int64), dict(), "cu_seqlens"),
        (dict(out_dtype=torch.float32), dict(), "out"),
        (dict(score_shape=(512, 128)), dict(), "score shape"),
        (dict(cuc_len=4), dict(), "B \\+ 1"),
        (dict(total=2**25, d=128), dict(), "int32 flat offsets"),
        (dict(total=4, d=8388482, total_comp=1), dict(), "head_dim"),
        (dict(total=2, total_comp=1), dict(), "requires at least ratio"),
        (dict(kv_stride=(512, 2)), dict(), "contiguous"),
        (dict(total=1024, d=96, ratio=128, coff=1), dict(ratio=128, coff=1), "ratio=128 is validated for head_dim"),
    ],
)
def test_check_support_rejects(kwargs, ctor, match):
    """check_support raises ValueError for configurations outside the envelope."""
    _require_sm100()
    compressor = _import_compressor()
    samples = _meta_samples(**kwargs)
    api = compressor.CSACompressorForward(**samples, ratio=ctor.get("ratio", 4), coff=ctor.get("coff", 2))
    with pytest.raises(ValueError, match=match):
        api.check_support()


@pytest.mark.L0
def test_check_support_rejects_cpu_tensors():
    """CPU sample tensors are rejected with a clear error."""
    _require_sm100()
    compressor = _import_compressor()
    samples = _meta_samples()
    samples["sample_kv"] = torch.empty(512, 256, dtype=torch.bfloat16, device="cpu")
    api = compressor.CSACompressorForward(**samples, ratio=4, coff=2)
    with pytest.raises(ValueError, match="CUDA"):
        api.check_support()


# ---------------------------------------------------------------------------
# Class API vs wrapper equivalence
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("coff", [1, 2])
def test_class_api_matches_wrapper(coff):
    """The explicit class API produces bitwise-identical results to the wrappers."""
    _require_sm100()
    compressor = _import_compressor()
    ratio, d = 4, 128
    kv, score, ape, cu, cuc, total_comp, go = _make_inputs([1023, 509], d, ratio, coff)
    total = kv.shape[0]
    kv2, score2, go2 = kv.view(total, -1), score.view(total, -1), go.view(total_comp, d)

    r_wrapped = _run_fused(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go)

    fwd = compressor.CSACompressorForward(
        sample_kv=kv2,
        sample_score=score2,
        sample_ape=ape,
        sample_cu_seqlens=cu,
        sample_cu_seqlens_comp=cuc,
        sample_out=torch.empty(total_comp, d, dtype=torch.bfloat16, device="meta"),
        ratio=ratio,
        coff=coff,
    )
    assert fwd.check_support()
    fwd.compile()
    out = torch.empty(total_comp, d, dtype=torch.bfloat16, device="cuda")
    fwd.execute(kv2, score2, ape, cu, cuc, out)

    bwd = compressor.CSACompressorBackward(
        sample_kv=kv2,
        sample_score=score2,
        sample_ape=ape,
        sample_cu_seqlens=cu,
        sample_cu_seqlens_comp=cuc,
        sample_out=torch.empty(total_comp, d, dtype=torch.bfloat16, device="meta"),
        ratio=ratio,
        coff=coff,
    )
    assert bwd.check_support()
    bwd.compile()
    grad_kv = torch.zeros_like(kv2)
    grad_score = torch.zeros_like(score2)
    grad_ape = torch.zeros_like(ape)
    bwd.execute(kv2, score2, ape, cu, cuc, go2, grad_kv, grad_score, grad_ape)
    torch.cuda.synchronize()

    assert torch.equal(out.view_as(r_wrapped[0]), r_wrapped[0])
    assert torch.equal(grad_kv.view_as(r_wrapped[1]), r_wrapped[1])
    assert torch.equal(grad_score.view_as(r_wrapped[2]), r_wrapped[2])
    assert torch.allclose(grad_ape, r_wrapped[3], rtol=0, atol=1e-3)


@pytest.mark.L0
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_backward_grad_ape_zeroing_contract():
    """grad_ape ownership: the kernel only ACCUMULATES, so the class-API caller must
    re-zero grad_ape before every execute and before every CUDA-graph replay that
    reuses the buffer. The high-level wrapper allocates a fresh zeroed buffer per call
    (and its zero-fill is captured with the kernel, so wrapper replays re-zero)."""
    _require_sm100()
    compressor = _import_compressor()
    ratio, d, coff = 128, 128, 1
    kv, score, ape, cu, cuc, total_comp, go = _make_inputs([640, 259], d, ratio, coff)
    total = kv.shape[0]
    kv2, score2, go2 = kv.view(total, -1), score.view(total, -1), go.view(total_comp, d)

    bwd = compressor.CSACompressorBackward(
        sample_kv=kv2,
        sample_score=score2,
        sample_ape=ape,
        sample_cu_seqlens=cu,
        sample_cu_seqlens_comp=cuc,
        sample_out=torch.empty(total_comp, d, dtype=torch.bfloat16, device="meta"),
        ratio=ratio,
        coff=coff,
    )
    assert bwd.check_support()
    bwd.compile()
    grad_kv = torch.empty_like(kv2)
    grad_score = torch.empty_like(score2)

    # Single-run reference (also warms the launch path up for the capture below).
    ref = torch.zeros_like(ape)
    bwd.execute(kv2, score2, ape, cu, cuc, go2, grad_kv, grad_score, ref)
    torch.cuda.synchronize()

    # Two executes WITHOUT re-zeroing accumulate (the documented sharp edge).
    acc = torch.zeros_like(ape)
    bwd.execute(kv2, score2, ape, cu, cuc, go2, grad_kv, grad_score, acc)
    bwd.execute(kv2, score2, ape, cu, cuc, go2, grad_kv, grad_score, acc)
    torch.cuda.synchronize()
    assert torch.allclose(acc, 2 * ref, rtol=0, atol=2e-3)

    # Graph replays of a captured class-API execute accumulate the same way: the
    # zero-fill happened BEFORE capture, so it is not part of the graph.
    gape_graph = torch.zeros_like(ape)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bwd.execute(kv2, score2, ape, cu, cuc, go2, grad_kv, grad_score, gape_graph)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.allclose(gape_graph, 2 * ref, rtol=0, atol=2e-3)

    # The wrapper path does NOT accumulate: fresh zeroed buffer per call.
    g1 = compressor.csa_compressor_backward_wrapper(kv2, score2, ape, cu, cuc, go2, ratio=ratio, head_dim=d, coff=coff)
    g2 = compressor.csa_compressor_backward_wrapper(kv2, score2, ape, cu, cuc, go2, ratio=ratio, head_dim=d, coff=coff)
    torch.cuda.synchronize()
    assert torch.allclose(g1["grad_ape"], ref, rtol=0, atol=1e-3)
    assert torch.allclose(g2["grad_ape"], ref, rtol=0, atol=1e-3)


# ---------------------------------------------------------------------------
# Runtime hazards: alignment, explicit streams, multi-device
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_misaligned_base_pointer_rejected():
    """Contiguous storage-offset views with unaligned base pointers raise ValueError."""
    _require_sm100()
    compressor = _import_compressor()
    d, ratio, coff = 128, 4, 2
    w = coff * d
    total = 512
    buf = torch.randn(total * w + 8, device="cuda").to(torch.bfloat16)
    kv = buf[2 : 2 + total * w].view(total, w)  # contiguous, 4-byte-aligned base
    assert kv.is_contiguous() and kv.data_ptr() % 16 != 0
    score = torch.randn(total, w, device="cuda").to(torch.bfloat16)
    ape = torch.randn(ratio, w, device="cuda")
    cu = torch.tensor([0, total], dtype=torch.int32, device="cuda")
    cuc = torch.tensor([0, total // ratio], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="aligned"):
        compressor.csa_compressor_forward_wrapper(kv, score, ape, cu, cuc, ratio=ratio, head_dim=d, coff=coff, total_comp=total // ratio)


@pytest.mark.L0
def test_explicit_stream_input_lifetime():
    """Inputs released right after an explicit-stream call are not recycled early.

    The launch path takes raw pointers, so the wrapper must ``record_stream`` its
    operands on the external stream; otherwise the caching allocator can hand the freed
    input storage to allocations on another stream while the kernel is still pending.
    """
    _require_sm100()
    compressor = _import_compressor()
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep not available")
    import cuda.bindings.driver as cuda_driver

    d, ratio, coff = 128, 4, 2
    w = coff * d
    lens = [4096]
    kv, score, ape, cu, cuc, total_comp, _ = _make_inputs(lens, d, ratio, coff)
    total = kv.shape[0]
    kv2, score2 = kv.view(total, w).contiguous(), score.view(total, w).contiguous()

    # Ground truth on the default stream (from private clones).
    expected = compressor.csa_compressor_forward_wrapper(
        kv2.clone(), score2.clone(), ape.clone(), cu.clone(), cuc.clone(), ratio=ratio, head_dim=d, coff=coff, total_comp=total_comp
    )["out"]
    torch.cuda.synchronize()

    side = torch.cuda.Stream()
    ext = cuda_driver.CUstream(side.cuda_stream)
    with torch.cuda.stream(side):
        torch.cuda._sleep(int(5e8))  # block the side stream so the kernel stays pending
    out = compressor.csa_compressor_forward_wrapper(kv2, score2, ape, cu, cuc, ratio=ratio, head_dim=d, coff=coff, total_comp=total_comp, stream=ext)["out"]
    # Drop every caller reference while the kernel is still queued behind the sleep,
    # then try hard to get the freed storages reallocated and scribbled on the default
    # (idle) stream.
    del kv, score, kv2, score2, ape, cu, cuc
    junk = [torch.full((total, w), 7.0, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    junk.append(torch.full((ratio, w), 7.0, device="cuda", dtype=torch.float32))
    junk.append(torch.full((64,), 7, device="cuda", dtype=torch.int32))
    torch.cuda.synchronize()
    assert torch.equal(out, expected), "explicit-stream inputs were recycled before the kernel consumed them"


@pytest.mark.L0
def test_multi_device_launch():
    """Tensors on a non-current device produce correct results (device anchoring)."""
    _require_sm100()
    compressor = _import_compressor()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >= 2 visible GPUs")
    if torch.cuda.get_device_capability(1) != (10, 0):
        pytest.skip("second GPU is not CC 10.0")
    d, ratio, coff = 128, 4, 2
    kv, score, ape, cu, cuc, total_comp, go = _make_inputs([1024], d, ratio, coff, device="cuda:1")
    total = kv.shape[0]
    kv2, score2, go2 = kv.view(total, -1), score.view(total, -1), go.view(total_comp, d)

    assert torch.cuda.current_device() == 0  # launch with a FOREIGN current device
    out_foreign = compressor.csa_compressor_forward_wrapper(kv2, score2, ape, cu, cuc, ratio=ratio, head_dim=d, coff=coff, total_comp=total_comp)["out"]
    grads_foreign = compressor.csa_compressor_backward_wrapper(kv2, score2, ape, cu, cuc, go2, ratio=ratio, head_dim=d, coff=coff)
    torch.cuda.synchronize(torch.device("cuda", 1))

    with torch.cuda.device(1):
        out_native = compressor.csa_compressor_forward_wrapper(kv2, score2, ape, cu, cuc, ratio=ratio, head_dim=d, coff=coff, total_comp=total_comp)["out"]
        grads_native = compressor.csa_compressor_backward_wrapper(kv2, score2, ape, cu, cuc, go2, ratio=ratio, head_dim=d, coff=coff)
        torch.cuda.synchronize()

    assert out_foreign.device == torch.device("cuda", 1)
    assert out_foreign.abs().sum().item() > 0  # the historic failure mode was all-zeros
    assert torch.equal(out_foreign, out_native)
    assert torch.equal(grads_foreign["grad_kv"], grads_native["grad_kv"])
    assert torch.equal(grads_foreign["grad_score"], grads_native["grad_score"])
