"""Tests for the fused CSA/HCA Compressor gated-pooling kernels (``cudnn.csa``).

Ported with the kernels from Megatron-LM (https://github.com/NVIDIA/Megatron-LM/pull/5984,
measurements and numerics in https://github.com/NVIDIA/Megatron-LM/issues/5968). Covers:

  - numerics of the fused region vs an fp32-intermediate eager reference
    (``dKV``/``dScore`` bit-identical, forward within one bf16 rounding step), vs the
    upstream eager numerics (tolerance), and vs an fp64 oracle (fused error <= eager
    error), over ragged THD packs including segments shorter than ``ratio``;
  - static-capacity padding rows (``total_comp > cu_seqlens_comp[-1]``);
  - kernel-side zero-writes to never-consumed ``dKV``/``dScore`` slots: NaN-canary
    (uninitialized) gradient buffers stay bitwise-equal to zero-initialized runs and to
    the eager reference, and the ``total_comp == 0`` host fallback still hands back
    exact zeros;
  - run-to-run determinism of forward / ``dKV`` / ``dScore`` (``dAPE`` uses fp32 atomics
    and is exempt by design; the backward refuses to run under
    ``torch.use_deterministic_algorithms(True)``);
  - CUDA graph capture: warmup -> capture fwd+bwd -> replay (including replay with new
    data and a smaller device-side true row count), and the loud error when the first
    call for a configuration would JIT under capture;
  - ``check_support`` boundaries (validated envelope: CC 10.0, ratio 4, coff in {1, 2},
    BF16 kv/score, FP32 ape, int32 cu_seqlens and int32 flat-offset bounds).

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
]


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("lens,d,ratio,coff", _SHAPES)
def test_numerics_vs_references(lens, d, ratio, coff):
    """Fused fwd+bwd vs fp32-eager (bitwise dKV/dScore), upstream eager, and fp64 oracle."""
    _require_sm100()
    kv, score, ape, cu, cuc, total_comp, go = _make_inputs(lens, d, ratio, coff)

    r_fused = _run_fused(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go)
    r_fp32 = _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode="fp32")
    r_up = _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode="upstream")
    r_fp64 = _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode="fp64")

    # vs fp32-intermediate eager reference (the fused kernels' numerics contract):
    # dKV / dScore bit-identical; forward within one bf16 rounding step on a tiny
    # fraction of elements; dAPE within fp32 atomics reorder noise.
    assert torch.equal(r_fused[1], r_fp32[1]), "dKV must be bit-identical to the fp32 reference"
    assert torch.equal(r_fused[2], r_fp32[2]), "dScore must be bit-identical to the fp32 reference"
    fwd_diff = (r_fused[0].float() - r_fp32[0].float()).abs()
    n_diff = (r_fused[0] != r_fp32[0]).sum().item()
    assert n_diff <= max(1, int(0.001 * r_fused[0].numel())), n_diff
    assert fwd_diff.max().item() <= 1.6e-2
    assert (r_fused[3] - r_fp32[3]).abs().max().item() <= 1e-3

    # vs the verbatim upstream eager numerics: not bit-identical (the eager path rounds
    # softmax weights to bf16 and multiplies in bf16), but close.
    for fused_t, up_t in zip(r_fused, r_up):
        assert torch.allclose(fused_t.float(), up_t.float(), rtol=0, atol=0.1)

    # vs the fp64 oracle: the fused kernel is at least as accurate as the eager code on
    # every output.
    for i in range(4):
        err_fused = (r_fused[i].double() - r_fp64[i].double()).abs().max().item()
        err_up = (r_up[i].double() - r_fp64[i].double()).abs().max().item()
        assert err_fused <= err_up * (1 + 1e-6) + 1e-4, (i, err_fused, err_up)


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


_CANARY_SHAPES = [
    # (lens, head_dim, pad, tok_pad) — every never-consumed dKV/dScore slot class must
    # be hit: segment tails (seqlen % ratio), the last block's first-half columns
    # (coff == 2 only), whole segments shorter than ratio (zero blocks),
    # static-capacity padding rows (pad > 0 extra grad_out rows), and static
    # token-capacity padding of the gradient buffers themselves (tok_pad > 0 tokens
    # beyond cu_seqlens[-1]).
    pytest.param([2048], 128, 0, 0, id="b1-d128"),
    pytest.param([1023, 2048, 509], 128, 0, 0, id="ragged3-d128"),
    pytest.param([3, 515, 1024, 129], 128, 0, 0, id="short-seg-d128"),
    pytest.param([5, 6, 7], 128, 0, 0, id="all-tiny-d128"),
    pytest.param([1023, 2048, 509], 512, 0, 0, id="ragged3-d512"),
    pytest.param([3, 515, 1024, 129], 128, 8, 0, id="short-seg-d128-padded"),
    pytest.param([1023, 2048, 509], 128, 8, 0, id="ragged3-d128-padded"),
    pytest.param([1023, 2048, 509], 128, 0, 37, id="ragged3-d128-tokpad"),
    pytest.param([3, 515, 1024, 129], 128, 8, 21, id="short-seg-d128-padded-tokpad"),
]


@pytest.mark.L0
@pytest.mark.parametrize("coff", [1, 2])
@pytest.mark.parametrize("lens,d,pad,tok_pad", _CANARY_SHAPES)
def test_backward_fills_uninitialized_buffers(lens, d, pad, tok_pad, coff):
    """NaN-canary: the backward kernel fully overwrites garbage dKV/dScore buffers.

    The kernel writes exact zeros to every never-consumed slot itself (there are no
    separate zero-fill kernels anymore), so running it into NaN-poisoned buffers must
    produce bitwise the same dKV/dScore as running it into zero-initialized buffers.
    """
    _require_sm100()
    compressor = _import_compressor()
    ratio = 4
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
    # bitwise as the fp32 eager reference computes them. (The fused backward ignores
    # incoming gradients on static-capacity padding rows by design, so the eager
    # reference runs with those rows zeroed.)
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
    assert torch.equal(gkv_nan.view_as(r_fp32[1]), r_fp32[1])
    assert torch.equal(gs_nan.view_as(r_fp32[2]), r_fp32[2])


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
        with pytest.warns(RuntimeWarning, match="not deterministic"):
            grads = compressor.csa_compressor_backward_wrapper(
                kv.view(kv.shape[0], -1), score.view(score.shape[0], -1), ape, cu, cuc, go.view(total_comp, 128), ratio=4, head_dim=128, coff=coff
            )
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)
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
@pytest.mark.parametrize(
    "kwargs,ctor,match",
    [
        (dict(coff=1), dict(ratio=128, coff=1), "validated for ratio=4"),
        (dict(), dict(ratio=8, coff=2), "validated for ratio=4"),
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
