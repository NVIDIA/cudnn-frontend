# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import inspect
import threading

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import dsa_init, with_dsa_indexer_backward_params, _require_sm100
from fe_api.dsa.dsa_reference import (
    _indexer_predict_distribution,
    check_ref_indexer_backward,
)


def _require_exact_sm100():
    """Skip unless the current device is exactly SM100 (capability ``(10, 0)``).

    ``dsa_init(min_compute_capability=100)`` means "SM100 **or newer**" — it
    compares ``major * 10 + minor`` — so an SM103/SM107/SM120 runner falls
    straight through it. ``backend="sm100_v2"`` is exact-``(10, 0)`` by
    construction (``check_support`` raises otherwise), so the v2 tests need
    the exact gate on top of ``dsa_init``'s floor.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("backend='sm100_v2' requires an SM100 (10, 0) GPU")


def _allocate(cfg, sm_scale: float):
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    h = cfg["qhead_per_kv_head"]
    topk = cfg["topk"]
    device = "cuda"

    index_q = torch.randn(b, s_q, h, d, dtype=torch.bfloat16, device=device)
    weights = torch.randn(b, s_q, h, dtype=torch.bfloat16, device=device)
    index_k = torch.randn(b, s_k, d, dtype=torch.bfloat16, device=device)

    topk_k = min(topk, s_k)
    topk_indices = torch.stack([torch.stack([torch.randperm(s_k, device=device)[:topk_k] for _ in range(s_q)]) for _ in range(b)]).to(torch.int32)
    if topk_k < topk:
        pad = torch.full((b, s_q, topk - topk_k), -1, dtype=torch.int32, device=device)
        topk_indices = torch.cat([topk_indices, pad], dim=-1)

    # ``index_score`` is the predict distribution the kernel consumes. It
    # must be consistent with ``(index_q, weights, index_k)`` under the
    # forward scoring math; otherwise the kernel's grad_signal will be
    # computed for a predict that doesn't match the reference's
    # autograd-through-the-forward computation.
    with torch.no_grad():
        index_score = _indexer_predict_distribution(
            index_q.float(),
            index_k.float(),
            weights.float(),
            topk_indices,
            sm_scale,
        ).contiguous()

    # Target distribution — keep random so the KL grad is non-trivial.
    attn_score = (
        torch.softmax(
            torch.randn(b, s_q, topk, device=device),
            dim=-1,
        )
        .float()
        .contiguous()
    )

    return index_q, weights, index_k, attn_score, index_score, topk_indices


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    request,
):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        block_I=block_I,
        min_compute_capability=90,
        s_q_default=128,
        s_kv_default=512,
    )
    sm_scale = 1.0
    # Configure loss_coeff and grad_loss so the kernel's internal
    # grad_scale = (loss_coeff / (B * S_q)) * grad_loss equals 1.0 — then
    # the reference (which uses a unit grad_scale) and the kernel agree.
    b_cfg = cfg["b"]
    s_q_cfg = cfg["s_q"]
    loss_coeff = float(b_cfg * s_q_cfg)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    grad_scale_expected = loss_coeff / (b_cfg * s_q_cfg)  # = 1.0

    (
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
    ) = _allocate(cfg, sm_scale=sm_scale)
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    # The kernel mutates attn_score + index_score in-place during its
    # score-grad stage. Keep pre-call copies so the reference can consume the
    # same inputs the kernel was given.
    attn_score_ref = attn_score.clone()
    index_score_ref = index_score.clone()
    torch_stream.wait_stream(torch.cuda.current_stream())
    try:
        result = DSA.indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            block_I=block_I,
            stream=stream,
        )
    except (ValueError, NotImplementedError, RuntimeError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch_stream.synchronize()

    d_index_q = result["d_index_q"]
    d_weights = result["d_weights"]
    d_index_k = result["d_index_k"]

    assert d_index_q.shape == index_q.shape
    assert d_weights.shape == weights.shape
    assert d_index_k.shape == index_k.shape
    assert torch.isfinite(d_index_q.float()).all()
    assert torch.isfinite(d_weights.float()).all()
    assert torch.isfinite(d_index_k.float()).all()

    if not cfg["skip_ref"]:
        check_ref_indexer_backward(
            index_q,
            weights,
            index_k,
            attn_score_ref,
            index_score_ref,
            topk_indices,
            d_index_q,
            d_weights,
            d_index_k,
            sm_scale=sm_scale,
            grad_scale=grad_scale_expected,
        )


# ===========================================================================
# Regression coverage for the output/plan-signature validation on the default
# indexer backward (SM100/SM90):
#   * illegal output dtypes raise BEFORE kernel 1 mutates the score buffers;
#   * wrong-rank / wrong-shape / non-contiguous plans are rejected up front;
#   * a direct plan or a cached wrapper plan reused with a mismatched
#     shape/stride signature raises (no fail-dirty);
#   * an fp32 d_index_k buffer is zeroed internally (result independent of the
#     buffer's incoming contents).
# The validation lives in the arch-independent plan/wrapper layer; the runs
# are pinned to SM100, the arch this suite's default kernels are built on.
# ===========================================================================

_IDXBWD_CFG = {
    "b": 1,
    "s_q": 128,
    "s_kv": 512,
    "head_dim": 128,
    "qhead_per_kv_head": 64,
    "topk": 512,
}
_IDXBWD_SM_SCALE = 1.0


def _import_dsa():
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    return DSA, cuda


def _idxbwd_inputs():
    """Deterministic inputs + a grad_loss/loss_coeff pair giving grad_scale=1."""
    (
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
    ) = _allocate(_IDXBWD_CFG, sm_scale=_IDXBWD_SM_SCALE)
    loss_coeff = float(_IDXBWD_CFG["b"] * _IDXBWD_CFG["s_q"])  # grad_scale = 1.0
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    return index_q, weights, index_k, attn_score, index_score, topk_indices, loss_coeff, grad_loss


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_backward_illegal_output_dtype_raises_before_mutation():
    """Each illegal output dtype raises ValueError, and the raise happens
    before kernel 1 mutates attn_score / index_score in place (no fail-dirty)."""
    _require_sm100()
    DSA, _ = _import_dsa()

    for label, kw in (
        ("fp32 d_index_q", {"d_index_q_dtype": torch.float32}),
        # fp32 d_weights is rejected: the kernel's dW store rounds the fp32
        # accumulator to bf16, so an fp32 buffer cannot be produced faithfully
        # and would silently carry bf16 precision.
        ("fp32 d_weights", {"d_weights_dtype": torch.float32}),
        ("fp16 d_weights", {"d_weights_dtype": torch.float16}),
        ("fp16 d_index_k", {"d_index_k_dtype": torch.float16}),
    ):
        iq, w, ik, attn, idx, tk, loss_coeff, grad_loss = _idxbwd_inputs()
        aq = attn.clone()
        isc = idx.clone()
        aq_pre = aq.clone()
        isc_pre = isc.clone()
        outputs = {}
        if "d_index_q_dtype" in kw:
            outputs["d_index_q"] = torch.empty_like(iq, dtype=kw["d_index_q_dtype"])
        if "d_weights_dtype" in kw:
            outputs["d_weights"] = torch.empty_like(w, dtype=kw["d_weights_dtype"])
        if "d_index_k_dtype" in kw:
            outputs["d_index_k"] = torch.empty_like(ik, dtype=kw["d_index_k_dtype"])

        with pytest.raises(ValueError):
            DSA.indexer_backward_wrapper(iq, w, ik, aq, isc, tk, sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128, **outputs)
        torch.cuda.synchronize()
        assert torch.equal(aq, aq_pre), f"{label}: attn_score mutated before the dtype check (fail-dirty)"
        assert torch.equal(isc, isc_pre), f"{label}: index_score mutated before the dtype check (fail-dirty)"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_backward_fp32_dindexk_zeroed_internally():
    """An fp32 d_index_k buffer is zeroed inside execute (the dK epilogue
    atomic-adds), so the result is independent of the buffer's incoming
    contents — a nonzero buffer does NOT return old + gradient."""
    _require_sm100()
    DSA, _ = _import_dsa()

    iq, w, ik, attn, idx, tk, loss_coeff, grad_loss = _idxbwd_inputs()

    dk_zero = torch.zeros_like(ik, dtype=torch.float32)
    res_zero = DSA.indexer_backward_wrapper(
        iq, w, ik, attn.clone(), idx.clone(), tk, sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128, d_index_k=dk_zero
    )
    torch.cuda.synchronize()
    dk_from_zero = res_zero["d_index_k"].clone()

    dk_nonzero = torch.full_like(ik, 999.0, dtype=torch.float32)
    res_nonzero = DSA.indexer_backward_wrapper(
        iq, w, ik, attn.clone(), idx.clone(), tk, sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128, d_index_k=dk_nonzero
    )
    torch.cuda.synchronize()
    dk_from_nonzero = res_nonzero["d_index_k"]

    assert torch.isfinite(dk_from_nonzero).all()
    max_diff = (dk_from_nonzero.double() - dk_from_zero.double()).abs().max().item()
    # If the buffer were not zeroed, the 999.0 offset would survive; the fp32
    # atomic-add jitter between two runs is ~1e-3, so a <1.0 bound cleanly
    # proves the internal zero-init.
    assert max_diff < 1.0, f"fp32 d_index_k not zeroed internally: max|nonzero-init - zero-init| = {max_diff:.3e}"


def _noncontiguous_like(sample):
    """A tensor with ``sample``'s shape/dtype/device but a strided (non-dense)
    layout, so its stride differs from a contiguous descriptor's."""
    padded = torch.empty(*sample.shape, 2, dtype=sample.dtype, device=sample.device)
    view = padded[..., 0]
    assert view.shape == sample.shape and not view.is_contiguous()
    view.copy_(sample)
    return view


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_backward_direct_plan_shape_stride_mismatch_raises():
    """A directly-built plan rejects a runtime tensor whose shape or
    stride/layout differs from the descriptor it was compiled for, raising a
    clean ValueError BEFORE kernel 1 mutates the score buffers (no fail-dirty)."""
    _require_sm100()
    DSA, _ = _import_dsa()

    iq, w, ik, attn, idx, tk, loss_coeff, grad_loss = _idxbwd_inputs()

    plan = DSA.IndexerBackward(
        sample_index_q=iq,
        sample_weights=w,
        sample_index_k=ik,
        sample_d_index_q=torch.empty_like(iq),
        sample_d_weights=torch.empty_like(w),
        sample_d_index_k=torch.empty_like(ik),
        sample_attn_score=attn,
        sample_index_score=idx,
        sample_topk_indices=tk,
        sm_scale=_IDXBWD_SM_SCALE,
        block_I=128,
    )
    assert plan.check_support()
    plan.compile()

    b, s_q, _, _ = iq.shape
    topk = tk.shape[-1]

    # Case 1: wrong-shaped attn_score (extra top-k column). Fill it with
    # meaningful (nonzero) data and snapshot both score buffers so we can assert
    # neither is mutated before the shape check raises.
    aq_bad = torch.softmax(torch.randn(b, s_q, topk + 1, device=iq.device), dim=-1).float()
    isc = idx.clone()
    aq_bad_pre = aq_bad.clone()
    isc_pre = isc.clone()
    with pytest.raises(ValueError):
        plan.execute(iq, w, ik, torch.empty_like(iq), torch.empty_like(w), torch.empty_like(ik), aq_bad, isc, tk, grad_loss, loss_coeff=loss_coeff)
    torch.cuda.synchronize()
    assert torch.equal(aq_bad, aq_bad_pre), "attn_score mutated before the shape check (fail-dirty)"
    assert torch.equal(isc, isc_pre), "index_score mutated before the shape check (fail-dirty)"

    # Case 2: non-contiguous attn_score (right shape, strided storage).
    aq_nc = _noncontiguous_like(attn)
    isc2 = idx.clone()
    aq_nc_pre = aq_nc.clone()
    isc2_pre = isc2.clone()
    with pytest.raises(ValueError):
        plan.execute(iq, w, ik, torch.empty_like(iq), torch.empty_like(w), torch.empty_like(ik), aq_nc, isc2, tk, grad_loss, loss_coeff=loss_coeff)
    torch.cuda.synchronize()
    assert torch.equal(aq_nc, aq_nc_pre), "attn_score mutated before the stride check (fail-dirty)"
    assert torch.equal(isc2, isc2_pre), "index_score mutated before the stride check (fail-dirty)"

    # Case 3: non-contiguous OUTPUT (d_weights) is rejected before mutation too.
    aq3 = attn.clone()
    isc3 = idx.clone()
    aq3_pre = aq3.clone()
    isc3_pre = isc3.clone()
    dw_nc = _noncontiguous_like(torch.empty_like(w))
    with pytest.raises(ValueError):
        plan.execute(iq, w, ik, torch.empty_like(iq), dw_nc, torch.empty_like(ik), aq3, isc3, tk, grad_loss, loss_coeff=loss_coeff)
    torch.cuda.synchronize()
    assert torch.equal(aq3, aq3_pre), "attn_score mutated before the output-stride check (fail-dirty)"
    assert torch.equal(isc3, isc3_pre), "index_score mutated before the output-stride check (fail-dirty)"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_backward_wrapper_cache_hit_stride_mismatch_raises():
    """A wrapper cache hit (identical shape/dtype/device key) still rejects a
    non-contiguous score buffer via the cached plan's execute-time signature
    check, before kernel 1 mutates either score buffer (no fail-dirty)."""
    _require_sm100()
    DSA, _ = _import_dsa()

    iq, w, ik, attn, idx, tk, loss_coeff, grad_loss = _idxbwd_inputs()

    # First call with contiguous tensors populates the wrapper cache (its
    # descriptor is contiguous).
    DSA.indexer_backward_wrapper(iq, w, ik, attn.clone(), idx.clone(), tk, sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128)
    torch.cuda.synchronize()

    # Second call: identical shape/dtype/device (cache hit) but a
    # non-contiguous attn_score. The cached plan must raise before mutation.
    aq_nc = _noncontiguous_like(attn)
    isc = idx.clone()
    aq_nc_pre = aq_nc.clone()
    isc_pre = isc.clone()
    with pytest.raises(ValueError):
        DSA.indexer_backward_wrapper(iq, w, ik, aq_nc, isc, tk, sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128)
    torch.cuda.synchronize()
    assert torch.equal(aq_nc, aq_nc_pre), "attn_score mutated before the cache-hit stride check (fail-dirty)"
    assert torch.equal(isc, isc_pre), "index_score mutated before the cache-hit stride check (fail-dirty)"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_backward_wrong_shape_plan_rejected_before_kernel1():
    """A first-call / directly-built plan whose output/score shape is
    inconsistent with ``index_q`` is rejected by the semantic shape validation
    in ``check_support`` BEFORE kernel 1 mutates the score buffers — not
    silently run on a mismatched signature that only faults in the GEMM. Covers
    both the wrapper (cache miss) and a directly-built plan. Fail-hard."""
    _require_sm100()
    DSA, _ = _import_dsa()

    iq, w, ik, attn, idx, tk, loss_coeff, grad_loss = _idxbwd_inputs()
    b, s_q, h, _ = iq.shape
    topk = tk.shape[-1]

    # (a) Wrapper first call (cache miss) with a wrong-shaped d_weights output
    #     buffer. The plan is built from the mismatched sample, and check_support
    #     must raise before any kernel launch; the score buffers stay pristine.
    aq = attn.clone()
    isc = idx.clone()
    aq_pre = aq.clone()
    isc_pre = isc.clone()
    dw_bad = torch.empty(b, s_q, h + 1, dtype=torch.bfloat16, device=iq.device)
    with pytest.raises(ValueError):
        DSA.indexer_backward_wrapper(
            iq, w, ik, aq, isc, tk, sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128, d_weights=dw_bad
        )
    torch.cuda.synchronize()
    assert torch.equal(aq, aq_pre), "attn_score mutated before the wrong-shape check (fail-dirty)"
    assert torch.equal(isc, isc_pre), "index_score mutated before the wrong-shape check (fail-dirty)"

    # (b) Directly-built plan with a wrong-shaped attn_score sample: check_support
    #     rejects the inconsistent signature (before compile / any kernel launch).
    plan = DSA.IndexerBackward(
        sample_index_q=iq,
        sample_weights=w,
        sample_index_k=ik,
        sample_d_index_q=torch.empty_like(iq),
        sample_d_weights=torch.empty_like(w),
        sample_d_index_k=torch.empty_like(ik),
        sample_attn_score=torch.empty(b, s_q, topk + 1, dtype=torch.float32, device=iq.device),
        sample_index_score=idx,
        sample_topk_indices=tk,
        sm_scale=_IDXBWD_SM_SCALE,
        block_I=128,
    )
    with pytest.raises(ValueError):
        plan.check_support()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_backward_noncontiguous_index_k_plan_rejected():
    """A plan built from a consistently non-contiguous ``index_k`` is rejected
    by ``check_support``: the SM100 kernel addresses K/dK with a hard-coded
    compact (D, 1) stride and the backend caches do not key the layout, so a
    non-compact plan would be silently mis-addressed. Covers a directly-built
    plan and a wrapper first call (rejected before kernel 1). Fail-hard."""
    _require_sm100()
    DSA, _ = _import_dsa()

    iq, w, ik, attn, idx, tk, loss_coeff, grad_loss = _idxbwd_inputs()

    ik_nc = _noncontiguous_like(ik)
    assert not ik_nc.is_contiguous()

    # (a) Directly-built plan with a non-contiguous index_k sample.
    plan = DSA.IndexerBackward(
        sample_index_q=iq,
        sample_weights=w,
        sample_index_k=ik_nc,
        sample_d_index_q=torch.empty_like(iq),
        sample_d_weights=torch.empty_like(w),
        sample_d_index_k=torch.empty_like(ik),
        sample_attn_score=attn,
        sample_index_score=idx,
        sample_topk_indices=tk,
        sm_scale=_IDXBWD_SM_SCALE,
        block_I=128,
    )
    with pytest.raises(ValueError):
        plan.check_support()

    # (b) Wrapper first call (cache miss) with a non-contiguous index_k is
    #     rejected before kernel 1 mutates the score buffers.
    aq = attn.clone()
    isc = idx.clone()
    aq_pre = aq.clone()
    isc_pre = isc.clone()
    with pytest.raises(ValueError):
        DSA.indexer_backward_wrapper(iq, w, ik_nc, aq, isc, tk, sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128)
    torch.cuda.synchronize()
    assert torch.equal(aq, aq_pre), "attn_score mutated before the non-contiguous check (fail-dirty)"
    assert torch.equal(isc, isc_pre), "index_score mutated before the non-contiguous check (fail-dirty)"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_backward_wrong_ndim_rejected():
    """A wrong-rank ``index_q`` / ``index_k`` / ``topk_indices`` is rejected
    with a clean ValueError (not a cryptic tuple-unpack error) by both the
    wrapper (before any allocation) and a directly-built plan."""
    _require_sm100()
    DSA, _ = _import_dsa()

    iq, w, ik, attn, idx, tk, loss_coeff, grad_loss = _idxbwd_inputs()
    common = dict(sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128)

    with pytest.raises(ValueError, match="index_q must be 4D"):
        DSA.indexer_backward_wrapper(iq[:, :, 0], w, ik, attn.clone(), idx.clone(), tk, **common)
    with pytest.raises(ValueError, match="index_k must be 3D"):
        DSA.indexer_backward_wrapper(iq, w, ik[0], attn.clone(), idx.clone(), tk, **common)
    with pytest.raises(ValueError, match="topk_indices must be 3D"):
        DSA.indexer_backward_wrapper(iq, w, ik, attn.clone(), idx.clone(), tk[0], **common)

    with pytest.raises(ValueError, match="sample_index_q must be 4D"):
        DSA.IndexerBackward(
            sample_index_q=iq[:, :, 0],
            sample_weights=w,
            sample_index_k=ik,
            sample_d_index_q=torch.empty_like(iq),
            sample_d_weights=torch.empty_like(w),
            sample_d_index_k=torch.empty_like(ik),
            sample_attn_score=attn,
            sample_index_score=idx,
            sample_topk_indices=tk,
            sm_scale=_IDXBWD_SM_SCALE,
            block_I=128,
        )


def _v2_call(
    index_q,
    weights,
    index_k,
    attn_score,
    index_score,
    topk_indices,
    loss_coeff,
    grad_loss,
    block_I,
    sm_scale=1.0,
    topk_indices_global=False,
    d_weights=None,
    d_index_k=None,
    stream=None,
):
    """Run the wrapper with backend="sm100_v2" on CLONED score buffers.

    Returns (result, grad_signal, predict) -- the two in-place score buffers
    after the call: grad_signal is the attn_score scratch (kernel 1's output)
    and predict is the consumed index_score buffer. Both are shared
    bit-for-bit with the default backend.
    """
    from cudnn import DSA
    from cuda.bindings import driver as cuda

    attn = attn_score.clone()
    index = index_score.clone()
    torch_stream = torch.cuda.Stream()
    cu_stream = stream if stream is not None else cuda.CUstream(torch_stream.cuda_stream)
    if stream is None:
        torch_stream.wait_stream(torch.cuda.current_stream())
    result = DSA.indexer_backward_wrapper(
        index_q,
        weights,
        index_k,
        attn,
        index,
        topk_indices,
        sm_scale=sm_scale,
        loss_coeff=loss_coeff,
        grad_loss=grad_loss,
        block_I=block_I,
        topk_indices_global=topk_indices_global,
        backend="sm100_v2",
        d_weights=d_weights,
        d_index_k=d_index_k,
        stream=cu_stream,
    )
    if stream is None:
        torch_stream.synchronize()
    return result, attn, index


def _fp64_oracle(index_q, weights, index_k, grad_signal, topk_indices):
    """Strict fp64 recompute of kernel 2's math on the captured grad signal
    (B=1, local == global ids). Pass ``grad_signal`` already multiplied by
    ``sm_scale`` when the kernel ran with a non-unit scale: kernel 2 folds the
    scale into the dQ/dW/dK products but gates on the *unscaled* score, which
    is what ``S > 0`` reproduces here."""
    b, s_q, h, d = index_q.shape
    s_k = index_k.shape[1]
    topk = topk_indices.shape[-1]
    assert b == 1
    qd = index_q.view(s_q, h, d).double()
    kd = index_k.view(s_k, d).double()
    wd = weights.view(s_q, h).double()
    ix = topk_indices.view(s_q, topk).long()
    gg = grad_signal.view(s_q, topk).double()
    valid = (ix >= 0) & (ix < s_k)
    k_g = kd[ix.clamp(0, s_k - 1)]
    S = torch.einsum("rtd,rhd->rth", k_g, qd) * valid.unsqueeze(-1)
    dw = torch.einsum("rt,rth->rh", gg, S.clamp(min=0))
    A = torch.where(
        S > 0,
        gg.unsqueeze(-1) * wd.unsqueeze(1),
        torch.zeros((), dtype=torch.float64, device=index_q.device),
    ) * valid.unsqueeze(-1)
    dq = torch.einsum("rth,rtd->rhd", A, k_g)
    dk = torch.zeros(s_k, d, dtype=torch.float64, device=index_q.device)
    dk.index_add_(0, ix.clamp(0, s_k - 1)[valid], torch.einsum("rth,rhd->rtd", A, qd)[valid])
    return dq, dw, dk


def _rms_rel(actual, oracle):
    err = actual.double().view(oracle.shape) - oracle
    return (err.pow(2).mean().sqrt() / oracle.pow(2).mean().sqrt().clamp(min=1e-30)).item()


# Band for a bf16 *output buffer* compared against an fp64 oracle: the store
# rounds to bf16, whose maximum relative roundoff is 2**-8 = 3.9e-3, so on
# inputs like these (randn, no catastrophic cancellation in the reference) the
# achievable
# rms_rel floor is a fraction of that however exact the kernel math is.
# Measured floor on B200 at s_q=128: 1.607e-3 .. 1.679e-3 over topk
# 128/256/384/512 and all three outputs -- flat in topk, as expected of a
# per-element quantisation floor -- so the band below keeps ~1.8x headroom.
# The v2 compute path itself is far tighter (the fp32-output test asserts
# 1e-5 / 1e-3 bands on the same oracle), so this band only certifies "no
# structurally wrong math", which is what the low-tile drain/clamp shapes and
# the sm_scale fold need from it: a dropped or doubled sm_scale fold measures
# 1.0 / 0.5 rms_rel here, i.e. 333x / 167x outside the band. What it covers only
# unreliably is the metadata-WAR hazard: at this shape a barrier-off kernel
# corrupted d_index_k in 1 of 5 seeds (4.0e-2 / 5.1e-2), against 5 of 5 at the
# larger shape -- hence the separate low-tile test.
_BF16_OUT_BAND = 3e-3

# Band for comparing two runs of the SAME shape against each other when
# ``d_index_k`` is a bf16 buffer. ``d_index_k`` is the one output reduced with
# fp32 atomics, so its summation order varies run to run *even serially*, and
# the bf16 store turns a sub-ulp fp32 difference into a whole-LSB flip on some
# elements. Measured serial run-to-run rms_rel on B200: 1.1e-6 at S_q=128 and
# 3.9e-5 at S_q=512 (both topk=512), i.e. it grows with the number of rows
# accumulating into a key, so the band is set well above the largest of those
# while still catching the class of defect these tests look for -- a scatter
# that reads stale metadata measures 8.1e-3 .. 5.9e-1 (see
# ``test_DSA_indexer_backward_wrapper_v2_low_tile_metadata_war``). A row whose
# contribution is near zero would of course do proportionally less damage.
_DK_ATOMIC_BAND = 1e-3


# Each topk compiles its own SM100 kernel variant (20-60 s apiece), so the
# default L0 smoke run keeps exactly one numeric point: topk=512, which
# exercises the multi-I-block path (4 blocks of 128). The rest of the sweep --
# including the topk=2048 smem-cap boundary -- runs at L1, per test/AGENTS.md
# ("L0 must stay fast (default CI smoke); big parameter sweeps go to higher
# levels").
_V2_TOPK_PARAMS = [
    pytest.param(128, marks=pytest.mark.L1),
    pytest.param(256, marks=pytest.mark.L1),
    pytest.param(384, marks=pytest.mark.L1),
    pytest.param(512, marks=pytest.mark.L0),
    pytest.param(640, marks=pytest.mark.L1),
    pytest.param(1024, marks=pytest.mark.L1),
    pytest.param(2048, marks=pytest.mark.L1),
]


@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("topk", _V2_TOPK_PARAMS)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper_v2(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    topk,
    request,
):
    """``backend="sm100_v2"`` keeps the wrapper contract with two-term
    bf16-expansion GEMMs and a deterministic d_weights reduction. Exactly
    SM100 (``_require_exact_sm100``); every parametrized config here is inside
    the declared support envelope, so any error past the gate is a real failure
    and MUST fail -- no skip conversion. topk coverage: 128 = the 1-tile floor
    (odd-tile dK drain, K/S pipes clamped to 1); 256 = 2 tiles (paired dK
    drain, pipes clamped to 2); 384 = 3 tiles, where the min-clamps become
    no-ops; 512 = 4 tiles, the smallest topk whose metadata restage needs no
    predicated tail (topk % 512 == 0), with the explicit metadata-WAR barrier
    still compiled in; 640 = odd tile count + predicated
    restage tail (topk % 512 != 0); 1024 = the compiled-out implicit regime;
    2048 = fills the SM100 dynamic-smem budget exactly (232448 B) and guards
    the ticket-ring placement."""
    try:
        from cudnn import DSA  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_exact_sm100()

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=topk,
        block_I=block_I,
        min_compute_capability=100,
        s_q_default=128,
        s_kv_default=512,
    )
    sm_scale = 1.0
    b_cfg = cfg["b"]
    s_q_cfg = cfg["s_q"]
    loss_coeff = float(b_cfg * s_q_cfg)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    grad_scale_expected = loss_coeff / (b_cfg * s_q_cfg)  # = 1.0

    (
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
    ) = _allocate(cfg, sm_scale=sm_scale)

    result, grad_signal, _ = _v2_call(
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
        loss_coeff,
        grad_loss,
        block_I,
        sm_scale=sm_scale,
    )

    d_index_q = result["d_index_q"]
    d_weights = result["d_weights"]
    d_index_k = result["d_index_k"]

    assert d_index_q.shape == index_q.shape
    assert d_weights.shape == weights.shape
    assert d_index_k.shape == index_k.shape
    assert d_index_q.dtype == index_q.dtype
    assert d_weights.dtype == weights.dtype
    assert d_index_k.dtype == index_k.dtype
    assert torch.isfinite(d_index_q.float()).all()
    assert torch.isfinite(d_weights.float()).all()
    assert torch.isfinite(d_index_k.float()).all()

    if not cfg["skip_ref"]:
        check_ref_indexer_backward(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            d_index_q,
            d_weights,
            d_index_k,
            sm_scale=sm_scale,
            grad_scale=grad_scale_expected,
        )

    # Low-tile regime (the clamped-pipe shapes) additionally against the strict
    # fp64 recompute of kernel 2 on the captured grad signal, which pins the
    # odd-/paired-dK-drain and pipe-clamp paths to exact math instead of only to
    # the autograd reference's own bf16-rounded recompute. Restricted to the
    # small-topk cases to keep the fp64 einsums cheap; ``_fp64_oracle`` models
    # B == 1 with local ids, which is this test's default configuration. This
    # is not a reliable detector for the metadata-WAR barrier: at the default
    # S_q = 128 a barrier-off kernel corrupted d_index_k in only 1 of 5 seeds
    # (5 of 5 at S_q = 512) -- see
    # ``test_DSA_indexer_backward_wrapper_v2_low_tile_metadata_war``.
    if topk <= 384 and cfg["b"] == 1:
        dq_o, dw_o, dk_o = _fp64_oracle(index_q, weights, index_k, grad_signal, topk_indices)
        dq_rr = _rms_rel(d_index_q, dq_o)
        dw_rr = _rms_rel(d_weights, dw_o)
        dk_rr = _rms_rel(d_index_k, dk_o)
        assert dq_rr < _BF16_OUT_BAND, f"topk={topk}: d_index_q rms_rel {dq_rr:.3e} vs fp64 oracle above the bf16-output band"
        assert dw_rr < _BF16_OUT_BAND, f"topk={topk}: d_weights rms_rel {dw_rr:.3e} vs fp64 oracle above the bf16-output band"
        assert dk_rr < _BF16_OUT_BAND, f"topk={topk}: d_index_k rms_rel {dk_rr:.3e} vs fp64 oracle above the bf16-output band"


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("topk", (256, 384))
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper_v2_low_tile_metadata_war(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    topk,
    request,
):
    """Regression cover for the low-tile metadata-WAR barrier (``pipe_M``),
    which stops the gather warpgroup from restaging a metadata slot the dK
    warpgroup has not finished reading.

    The hazard is a race whose rate depends strongly on the shape: the more
    rows a CTA processes (the grid is one CTA per SM, 148 on B200) and the
    larger the dK atomic scatter, the further its drain lags the gather front.
    So this test forces S_q >= 512 and S_k >= 4096 and takes fp32 outputs,
    where the kernel's own d_index_k error is ~2.5e-6 and a corrupted scatter
    cannot hide under the bf16 store floor. Measured on B200 with the barrier
    compiled out: d_index_k rms_rel 8.1e-3 .. 5.2e-2 at topk 256 and
    4.6e-1 .. 5.9e-1 at topk 384, in 5 of 5 runs each (2.4e-6 with the barrier
    in), while d_index_q / d_weights keep the same error to four digits. At the
    S_q = 128 / S_k = 512 shape the other v2 tests use, the same barrier-off
    kernel tripped in only 1 of 5 seeds, which is why this test exists; a
    correct kernel never trips the bound."""
    try:
        from cudnn import DSA  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_exact_sm100()

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=topk,
        block_I=block_I,
        min_compute_capability=100,
        s_q_default=512,
        s_kv_default=4096,
    )
    if cfg["b"] != 1:
        pytest.skip("B == 1 required for this test: _fp64_oracle models a single batch")
    # ``dsa_init`` honours --dsa-s_q / --dsa-s_kv; raise them back up if the
    # run requested something smaller, otherwise the hazard window closes and
    # the test silently stops covering anything.
    cfg = dict(cfg)
    cfg["s_q"] = max(cfg["s_q"], 512)
    cfg["s_kv"] = max(cfg["s_kv"], 4096)

    sm_scale = 1.0
    b, s_q, s_k, d = cfg["b"], cfg["s_q"], cfg["s_kv"], cfg["head_dim"]
    loss_coeff = float(b * s_q)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")

    index_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=sm_scale)
    dw32 = torch.empty(b, s_q, cfg["qhead_per_kv_head"], dtype=torch.float32, device="cuda")
    dk32 = torch.empty(b, s_k, d, dtype=torch.float32, device="cuda")

    result, grad_signal, _ = _v2_call(
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
        loss_coeff,
        grad_loss,
        block_I,
        sm_scale=sm_scale,
        d_weights=dw32,
        d_index_k=dk32,
    )

    dq_o, dw_o, dk_o = _fp64_oracle(index_q, weights, index_k, grad_signal, topk_indices)
    dk_rr = _rms_rel(result["d_index_k"], dk_o)
    dw_rr = _rms_rel(result["d_weights"], dw_o)
    dq_rr = _rms_rel(result["d_index_q"], dq_o)
    assert dk_rr < 1e-3, f"topk={topk}: fp32 d_index_k rms_rel {dk_rr:.3e} vs fp64 oracle above the v2 band (metadata-WAR corruption?)"
    assert dw_rr < 1e-5, f"topk={topk}: fp32 d_weights rms_rel {dw_rr:.3e} above the deterministic-accumulator band"
    assert dq_rr < _BF16_OUT_BAND, f"topk={topk}: d_index_q rms_rel {dq_rr:.3e} above the bf16-output band"


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper_v2_full_valid_topk2048(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    request,
):
    """topk=2048 with S_k = 2048: every slot holds a valid id (no -1
    padding), so the maximum supported topk is exercised at its full
    top-k workload, not mostly padding. Checked against the strict fp64
    oracle on the captured grad signal (the autograd reference's recomputed
    predict distribution loses fidelity at full-valid topk=2048 — the
    default backend deviates from it by the same amount v2 does, measured
    0.604 rms relative for both on the same inputs, so the deviation is a
    reference artifact, not a v2 one)."""
    try:
        from cudnn import DSA  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_exact_sm100()

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=2048,
        block_I=block_I,
        min_compute_capability=100,
        s_q_default=128,
        s_kv_default=2048,
    )
    # ``dsa_init`` honours --dsa-s_kv; topk=2048 needs at least that many keys
    # per batch or the all-slots-valid premise below cannot hold, so raise it
    # back up rather than failing the run.
    cfg = dict(cfg)
    cfg["s_kv"] = max(cfg["s_kv"], 2048)
    sm_scale = 1.0
    b, s_q = cfg["b"], cfg["s_q"]
    loss_coeff = float(b * s_q)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")

    index_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=sm_scale)
    assert int((topk_indices >= 0).sum()) == topk_indices.numel(), "test setup: all slots must be valid"

    dw32 = torch.empty(b, s_q, cfg["qhead_per_kv_head"], dtype=torch.float32, device="cuda")
    dk32 = torch.empty(b, cfg["s_kv"], cfg["head_dim"], dtype=torch.float32, device="cuda")
    result, g, _ = _v2_call(index_q, weights, index_k, attn_score, index_score, topk_indices, loss_coeff, grad_loss, block_I, d_weights=dw32, d_index_k=dk32)

    for t in (result["d_index_q"], result["d_weights"], result["d_index_k"]):
        assert torch.isfinite(t.float()).all()
    # ``_fp64_oracle`` recomputes at B == 1 only (same guard as
    # ``test_DSA_indexer_backward_wrapper_v2``): a ``--dsa-b`` override keeps
    # everything above and drops just the oracle bands.
    if cfg["b"] == 1:
        dq_o, dw_o, dk_o = _fp64_oracle(index_q, weights, index_k, g, topk_indices)
        dw_rr = _rms_rel(result["d_weights"], dw_o)
        dk_rr = _rms_rel(result["d_index_k"], dk_o)
        dq_rr = _rms_rel(result["d_index_q"], dq_o)
        assert dw_rr < 1e-5, f"fp32 d_weights rms_rel {dw_rr:.3e} above the deterministic-accumulator band"
        assert dk_rr < 1e-3, f"fp32 d_index_k rms_rel {dk_rr:.3e} above the v2 band"
        assert dq_rr < 3e-3, f"d_index_q rms_rel {dq_rr:.3e} above the bf16-output floor band"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "bad",
    (
        "topk_2176",
        "topk_not_mult_128",
        "sm_scale_zero",
        "sm_scale_negative",
        "d_index_q_fp32",
        "d_weights_fp16",
        "topk_indices_noncontig",
        "index_k_dims",
    ),
)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper_v2_envelope_rejection(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    bad,
    request,
):
    """check_support must reject out-of-envelope requests with ValueError
    BEFORE kernel 1 mutates the score buffers: topk bounds (2176 above the
    smem cap, non-multiple-of-128), non-positive sm_scale, unsupported output
    dtypes, non-contiguous metadata, and an index_k whose dim 0 / dim 2
    disagree with index_q. (topk 128/256 are now inside the envelope and are
    exercised by the acceptance test above.)"""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_exact_sm100()

    topk = {"topk_2176": 2176, "topk_not_mult_128": 1000}.get(bad, 512)
    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=topk,
        block_I=block_I,
        min_compute_capability=100,
        s_q_default=128,
        s_kv_default=512,
    )
    sm_scale = {"sm_scale_zero": 0.0, "sm_scale_negative": -1.0}.get(bad, 1.0)
    loss_coeff = float(cfg["b"] * cfg["s_q"])
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")

    index_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=1.0)
    d_index_q = None
    d_weights = None
    if bad == "d_index_q_fp32":
        d_index_q = torch.empty_like(index_q, dtype=torch.float32)
    if bad == "d_weights_fp16":
        d_weights = torch.empty_like(weights, dtype=torch.float16)
    if bad == "topk_indices_noncontig":
        wide = torch.full((cfg["b"], cfg["s_q"], 2 * topk), -1, dtype=torch.int32, device="cuda")
        wide[..., ::2] = topk_indices
        topk_indices = wide[..., ::2]
        assert not topk_indices.is_contiguous()
    d_index_k = None
    if bad == "index_k_dims":
        # Only index_k dim 1 reaches the wrapper plan-cache key, so warm the
        # cache with a valid call first: the mis-shaped index_k below then hits
        # that entry without re-running check_support, and the backend's
        # execute-time contract check is the only thing left to reject it. The
        # element count is unchanged, so the flat (b * s_k, d) view alone cannot
        # catch it.
        DSA.indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score.clone(),
            index_score.clone(),
            topk_indices,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            block_I=block_I,
            backend="sm100_v2",
        )
        torch.cuda.synchronize()
        index_k = index_k.reshape(2 * cfg["b"], cfg["s_kv"], head_dim // 2).contiguous()
        d_index_k = torch.empty_like(index_k)

    attn_before = attn_score.clone()
    with pytest.raises(ValueError):
        DSA.indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            block_I=block_I,
            backend="sm100_v2",
            d_index_q=d_index_q,
            d_weights=d_weights,
            d_index_k=d_index_k,
        )
    torch.cuda.synchronize()
    assert torch.equal(attn_score, attn_before), "rejected request must not mutate attn_score"


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper_v2_batch_local_global_oob(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    request,
):
    """B > 1 id handling: (a) local per-batch ids match the equivalent
    global flat ids bit-for-bit on dq/dw; (b) a positive out-of-range local
    id (== S_k) contributes nothing — bitwise-identical dq/dw to the same
    slot holding -1 — instead of aliasing batch 1's first KV row; (c) a
    negative id likewise; (d) the clean local run matches the autograd
    reference."""
    try:
        from cudnn import DSA  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_exact_sm100()

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=512,
        block_I=block_I,
        min_compute_capability=100,
        b_default=2,
        s_q_default=128,
        s_kv_default=512,
    )
    if cfg["b"] < 2:
        pytest.skip("B > 1 required for this test")
    sm_scale = 1.0
    b, s_k = cfg["b"], cfg["s_kv"]
    loss_coeff = float(b * cfg["s_q"])
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")

    index_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=sm_scale)

    def run(idx, global_ids):
        dk32 = torch.empty(b, s_k, cfg["head_dim"], dtype=torch.float32, device="cuda")
        return _v2_call(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            idx,
            loss_coeff,
            grad_loss,
            block_I,
            topk_indices_global=global_ids,
            d_index_k=dk32,
        )[0]

    # (a) local ids vs equivalent global flat ids
    r_local = run(topk_indices, False)
    offs = (torch.arange(b, device="cuda", dtype=torch.int32) * s_k).view(b, 1, 1)
    idx_global = torch.where(topk_indices >= 0, topk_indices + offs, topk_indices)
    r_global = run(idx_global, True)
    assert torch.equal(r_local["d_index_q"], r_global["d_index_q"])
    assert torch.equal(r_local["d_weights"], r_global["d_weights"])
    assert _rms_rel(r_global["d_index_k"], r_local["d_index_k"].double()) < 1e-5  # fp32-atomic order class

    # (b)/(c) positive-OOB and negative ids behave as "contributes nothing"
    idx_inv = topk_indices.clone()
    idx_inv[0, 0, 0] = -1
    r_ref = run(idx_inv, False)
    for oob in (s_k, -7):
        idx_oob = topk_indices.clone()
        idx_oob[0, 0, 0] = oob
        r_oob = run(idx_oob, False)
        assert torch.equal(r_oob["d_index_q"], r_ref["d_index_q"]), f"id={oob} leaked into d_index_q"
        assert torch.equal(r_oob["d_weights"], r_ref["d_weights"]), f"id={oob} leaked into d_weights"
        assert _rms_rel(r_oob["d_index_k"], r_ref["d_index_k"].double()) < 1e-5, f"id={oob} leaked into d_index_k"

    # (d) clean local run vs autograd reference
    if not cfg["skip_ref"]:
        check_ref_indexer_backward(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            r_local["d_index_q"],
            r_local["d_weights"],
            r_local["d_index_k"],
            sm_scale=sm_scale,
            grad_scale=1.0,
        )


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper_v2_sm_scale(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    request,
):
    """Non-unit sm_scale: (a) the gradients match a strict fp64 recompute that
    folds the scale into the grad signal -- an autograd-reference check
    alone historically could not catch a mis-applied scale here (its noise
    floor has since dropped to rms_rel <= 0.005, but the fp64 recompute stays
    the authoritative scale check); (b) both in-place score
    buffers are left bitwise identical to the default backend's, i.e. the
    scratch holds exactly kernel 1's grad_signal and ``index_score`` is
    consumed the same way (the scale folds inside kernel 2, no host-side
    buffer mutation)."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_exact_sm100()

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=512,
        block_I=block_I,
        min_compute_capability=100,
        s_q_default=128,
        s_kv_default=512,
    )
    sm_scale = 0.5
    loss_coeff = float(cfg["b"] * cfg["s_q"])
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")

    index_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=sm_scale)

    result, g_v2, predict_v2 = _v2_call(index_q, weights, index_k, attn_score, index_score, topk_indices, loss_coeff, grad_loss, block_I, sm_scale=sm_scale)

    # default backend on the identical inputs: scratch must agree bitwise
    attn_def = attn_score.clone()
    index_def = index_score.clone()
    torch_stream = torch.cuda.Stream()
    torch_stream.wait_stream(torch.cuda.current_stream())
    DSA.indexer_backward_wrapper(
        index_q,
        weights,
        index_k,
        attn_def,
        index_def,
        topk_indices,
        sm_scale=sm_scale,
        loss_coeff=loss_coeff,
        grad_loss=grad_loss,
        block_I=block_I,
        stream=cuda.CUstream(torch_stream.cuda_stream),
    )
    torch_stream.synchronize()
    assert torch.equal(g_v2, attn_def), "v2 must leave attn_score holding exactly kernel 1's grad_signal"
    assert torch.equal(predict_v2, index_def), "v2 must consume index_score exactly like the default backend"

    # The scale is the whole point of this test, so check it against an oracle
    # that can actually see it: folding sm_scale into the captured grad signal
    # reproduces kernel 2's math, while dropping the fold lands at rms_rel 1.0
    # and doubling it at 0.5 -- 333x / 167x outside the 3e-3 band below. The
    # autograd reference further down cannot do this job: its own bf16 noise
    # floor is rms_rel ~0.29 / cos 0.9935 at this shape and seed, so a 1.2x
    # scale error still reads 0.53 (inside the default 0.55) and cosine is
    # scale-blind.
    if cfg["b"] == 1:
        dq_o, dw_o, dk_o = _fp64_oracle(index_q, weights, index_k, g_v2 * sm_scale, topk_indices)
        dq_rr = _rms_rel(result["d_index_q"], dq_o)
        dw_rr = _rms_rel(result["d_weights"], dw_o)
        dk_rr = _rms_rel(result["d_index_k"], dk_o)
        assert dq_rr < _BF16_OUT_BAND, f"d_index_q rms_rel {dq_rr:.3e} vs the scale-folded fp64 oracle above the bf16-output band"
        assert dw_rr < _BF16_OUT_BAND, f"d_weights rms_rel {dw_rr:.3e} vs the scale-folded fp64 oracle above the bf16-output band"
        assert dk_rr < _BF16_OUT_BAND, f"d_index_k rms_rel {dk_rr:.3e} vs the scale-folded fp64 oracle above the bf16-output band"

    if not cfg["skip_ref"]:
        check_ref_indexer_backward(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            result["d_index_q"],
            result["d_weights"],
            result["d_index_k"],
            sm_scale=sm_scale,
            grad_scale=1.0,
        )


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper_v2_fp32_outputs(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    request,
):
    """fp32 d_weights / d_index_k output contract: caller-supplied fp32
    buffers receive the fp32 accumulators directly. Checks the advertised
    precision property against a strict fp64 oracle consuming the captured
    grad signal (asserted below at rms_rel < 1e-5 for d_weights and < 1e-3
    for d_index_k — both bands well under the ~1.7e-3 bf16 output floor the
    same outputs hit when they are bf16), plus bitwise run-to-run determinism
    of d_weights/d_index_q."""
    try:
        from cudnn import DSA  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_exact_sm100()

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=512,
        block_I=block_I,
        min_compute_capability=100,
        s_q_default=128,
        s_kv_default=512,
    )
    sm_scale = 1.0
    b, s_q, s_k, d = cfg["b"], cfg["s_q"], cfg["s_kv"], cfg["head_dim"]
    loss_coeff = float(b * s_q)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")

    index_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=sm_scale)

    def run_fp32():
        dw32 = torch.empty(b, s_q, cfg["qhead_per_kv_head"], dtype=torch.float32, device="cuda")
        dk32 = torch.empty(b, s_k, d, dtype=torch.float32, device="cuda")
        return _v2_call(index_q, weights, index_k, attn_score, index_score, topk_indices, loss_coeff, grad_loss, block_I, d_weights=dw32, d_index_k=dk32)

    r1, g1, _ = run_fp32()
    r2, g2, _ = run_fp32()
    assert torch.equal(g1, g2)
    assert r1["d_weights"].dtype == torch.float32 and r1["d_index_k"].dtype == torch.float32

    # determinism: dw and dq bitwise run-to-run (dk is fp32-atomic class)
    assert torch.equal(r1["d_weights"], r2["d_weights"]), "d_weights must be bitwise deterministic"
    assert torch.equal(r1["d_index_q"], r2["d_index_q"]), "d_index_q must be bitwise deterministic"

    # advertised precision property vs strict fp64 oracle on the captured g
    # ``_fp64_oracle`` recomputes at B == 1 only (same guard as
    # ``test_DSA_indexer_backward_wrapper_v2``): a ``--dsa-b`` override keeps
    # the determinism checks above and drops just the oracle bands.
    if cfg["b"] == 1:
        dq_o, dw_o, dk_o = _fp64_oracle(index_q, weights, index_k, g1, topk_indices)
        dw_rr = _rms_rel(r1["d_weights"], dw_o)
        dk_rr = _rms_rel(r1["d_index_k"], dk_o)
        dq_rr = _rms_rel(r1["d_index_q"], dq_o)
        assert dw_rr < 1e-5, f"fp32 d_weights rms_rel {dw_rr:.3e} above the deterministic-accumulator band"
        assert dk_rr < 1e-3, f"fp32 d_index_k rms_rel {dk_rr:.3e} above the v2 band"
        assert dq_rr < 3e-3, f"d_index_q rms_rel {dq_rr:.3e} above the bf16-output floor band"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper_v2_multi_stream(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    request,
):
    """Ticket-counter ownership under concurrency: the wrapper keys its plan
    cache on the stream, so two explicit streams get private plans/counters.
    Interleave executes on two streams without host synchronization between
    launches and check every result against its serial reference (dq/dw
    bitwise, dk in the fp32-atomic class)."""
    try:
        from cudnn import DSA  # noqa: F401
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_exact_sm100()

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=512,
        block_I=block_I,
        min_compute_capability=100,
        s_q_default=128,
        s_kv_default=512,
    )
    sm_scale = 1.0
    loss_coeff = float(cfg["b"] * cfg["s_q"])
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    n_iters = 6

    inputs = [_allocate(cfg, sm_scale=sm_scale) for _ in range(2)]

    # serial references (one call at a time)
    refs = []
    for index_q, weights, index_k, attn_score, index_score, topk_indices in inputs:
        r, _, _ = _v2_call(index_q, weights, index_k, attn_score, index_score, topk_indices, loss_coeff, grad_loss, block_I)
        refs.append(r)

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    cu_streams = [cuda.CUstream(s.cuda_stream) for s in streams]
    # pre-clone the in-place score buffers for every iteration up front so
    # the interleaved phase enqueues only wrapper work
    clones = [[(inp[3].clone(), inp[4].clone()) for _ in range(n_iters)] for inp in inputs]
    results = [[None] * n_iters for _ in range(2)]
    torch.cuda.synchronize()

    for it in range(n_iters):
        for lane in range(2):
            index_q, weights, index_k, _, _, topk_indices = inputs[lane]
            attn, index = clones[lane][it]
            streams[lane].wait_stream(torch.cuda.current_stream())
            results[lane][it] = DSA.indexer_backward_wrapper(
                index_q,
                weights,
                index_k,
                attn,
                index,
                topk_indices,
                sm_scale=sm_scale,
                loss_coeff=loss_coeff,
                grad_loss=grad_loss,
                block_I=block_I,
                backend="sm100_v2",
                stream=cu_streams[lane],
            )
        # no host sync inside the loop: the two lanes are enqueued back to back,
        # so the device is free to overlap them (whether it does depends on the
        # shape -- at the default S_q = 128 each call is short enough that the
        # host stays ahead)
    torch.cuda.synchronize()

    for lane in range(2):
        for it in range(n_iters):
            r = results[lane][it]
            assert torch.equal(r["d_index_q"], refs[lane]["d_index_q"]), f"stream {lane} iter {it}: d_index_q diverged"
            assert torch.equal(r["d_weights"], refs[lane]["d_weights"]), f"stream {lane} iter {it}: d_weights diverged"
            assert _rms_rel(r["d_index_k"], refs[lane]["d_index_k"].double()) < _DK_ATOMIC_BAND, f"stream {lane} iter {it}: d_index_k diverged"


def _v2_plan_cache():
    """The wrapper's plan cache. Reached into on purpose: how many plans a call
    pattern creates is exactly what the cache key controls, and it is the only
    *deterministic* observable of two streams sharing one plan."""
    from cudnn.deepseek_sparse_attention.indexer_backward import api as _api

    return _api._cache_of_IndexerBackwardObjects


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper_v2_stream_none_ambient_streams(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    request,
):
    """``stream=None`` under two different ambient ``torch.cuda.stream``
    contexts must not share one plan.

    ``stream=None`` does not mean "the default stream": the backend resolves it
    with ``torch.cuda.current_stream()``, i.e. to the *ambient* stream of the
    calling context. Keying the plan cache on the ``stream`` argument alone
    therefore mapped two genuinely different execution streams onto one cached
    plan, and so onto one per-plan workspace -- the self-resetting ticket
    counter -- which the backend documents as single-execution-at-a-time
    state. The committed multi-stream test only
    passes ``stream=`` explicitly, which is the gap this covers.

    What this test asserts is the *invariant* (one plan per resolved stream),
    not gradient corruption: the two checks are (a) the two ambient streams get
    separate plan-cache entries, which is deterministic and is what a
    regression in the cache key breaks (keying the raw ``stream`` argument
    yields one entry instead of two), and (b) the interleaved executions still
    reproduce their serial results, which is a correctness guard on the keyed
    path. (b) is not a second detector for the same bug -- a deliberately
    shared plan does not reliably corrupt on a dedicated B200, because the v2
    kernel is a persistent grid of one CTA per SM that still requests 207872 of
    the 232448 B per-CTA shared-memory budget at this shape (the layout's only
    topk-dependent part is 2 KB per 128 slots), so a second launch mostly cannot
    get co-resident CTAs and the damage window stays closed. That is a property of this device being
    fully available to one launch, not a guarantee the backend offers.

    ``s_q``/``S_k``/topk together are deliberately different from every other
    v2 test's, so this test's plan-cache keys cannot collide with theirs
    regardless of test order."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_exact_sm100()

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=512,
        block_I=block_I,
        min_compute_capability=100,
        s_q_default=512,
        s_kv_default=512,
    )
    sm_scale = 1.0
    loss_coeff = float(cfg["b"] * cfg["s_q"])
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    n_iters = 8

    inputs = [_allocate(cfg, sm_scale=sm_scale) for _ in range(2)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    default_stream = torch.cuda.current_stream()

    def call_ambient(lane, attn, index):
        """Run the v2 wrapper with NO ``stream`` argument inside lane's
        ambient stream context."""
        index_q, weights, index_k, _, _, topk_indices = inputs[lane]
        streams[lane].wait_stream(default_stream)
        with torch.cuda.stream(streams[lane]):
            return DSA.indexer_backward_wrapper(
                index_q,
                weights,
                index_k,
                attn,
                index,
                topk_indices,
                sm_scale=sm_scale,
                loss_coeff=loss_coeff,
                grad_loss=grad_loss,
                block_I=block_I,
                backend="sm100_v2",
            )

    # (a) serial warm-up: one lane at a time, fully synchronized -- each
    # lane's result is the serial baseline that the interleaved run below
    # must reproduce (a self-generated baseline, not an oracle; the fp64
    # oracle checks live in the accuracy tests).
    cache = _v2_plan_cache()
    keys_before = set(cache.keys())
    refs = []
    for lane in range(2):
        attn, index = inputs[lane][3].clone(), inputs[lane][4].clone()
        refs.append(call_ambient(lane, attn, index))
        streams[lane].synchronize()
    added = set(cache.keys()) - keys_before
    assert len(added) == 2, (
        "stream=None under two different ambient streams must key two separate plans "
        f"(the per-plan ticket counter cannot be shared across concurrent streams); "
        f"got {len(added)} new plan-cache entries"
    )

    # (b) interleaved: enqueue both lanes without host synchronization so the
    # launches are free to overlap on the device.
    clones = [[(inp[3].clone(), inp[4].clone()) for _ in range(n_iters)] for inp in inputs]
    results = [[None] * n_iters for _ in range(2)]
    torch.cuda.synchronize()
    for it in range(n_iters):
        for lane in range(2):
            attn, index = clones[lane][it]
            results[lane][it] = call_ambient(lane, attn, index)
    torch.cuda.synchronize()

    for lane in range(2):
        for it in range(n_iters):
            r = results[lane][it]
            assert torch.equal(r["d_index_q"], refs[lane]["d_index_q"]), f"ambient stream {lane} iter {it}: d_index_q diverged"
            assert torch.equal(r["d_weights"], refs[lane]["d_weights"]), f"ambient stream {lane} iter {it}: d_weights diverged"
            assert _rms_rel(r["d_index_k"], refs[lane]["d_index_k"].double()) < _DK_ATOMIC_BAND, f"ambient stream {lane} iter {it}: d_index_k diverged"


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper_v2_stream_per_thread_two_threads(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    request,
):
    """Two host threads passing ``cudaStreamPerThread`` explicitly must not
    share one plan.

    ``cudaStreamPerThread`` is the integer 2 in every host thread, each time
    denoting that thread's own stream, so the handle alone cannot tell the two
    streams apart -- but the caller can: that handle means "the calling
    thread's stream" by definition, so the wrapper appends the calling thread's
    id to the key for that one value. Without it both threads land on one plan
    and so on one per-plan workspace -- the self-resetting ticket counter --
    which the backend documents as single-execution-at-a-time state.

    This is the explicit-handle twin of the ``stream=None`` ambient-stream test
    above and asserts the same invariant: one plan per resolved stream. Two
    implementation details of the test matter. The threads are held alive across
    the whole measured window by a barrier, because ``threading.get_ident()`` is
    only unique among *live* threads and Python does hand a dead thread's id to
    a later one. Their wrapper calls are serialized with a lock, so what is
    asserted is the cache key and not the thread-safety of first-call
    compilation.

    ``s_q``/``S_k``/topk together are deliberately different from every other
    v2 test's, so this test's plan-cache keys cannot collide with theirs
    regardless of test order."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_exact_sm100()

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=512,
        block_I=block_I,
        min_compute_capability=100,
        s_q_default=256,
        s_kv_default=512,
    )
    sm_scale = 1.0
    loss_coeff = float(cfg["b"] * cfg["s_q"])
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    index_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=sm_scale)
    per_thread_stream = cuda.CUstream(2)

    def call(stream):
        return DSA.indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score.clone(),
            index_score.clone(),
            topk_indices,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            block_I=block_I,
            backend="sm100_v2",
            stream=stream,
        )

    # Main-thread reference on the ambient stream; also warms the compile cache
    # so the threaded phase below cannot be a first-call compile race.
    ref = call(None)
    torch.cuda.synchronize()

    cache = _v2_plan_cache()
    keys_before = set(cache.keys())
    results, errors = {}, {}
    barrier = threading.Barrier(2)
    serialize = threading.Lock()

    def lane(lane_id):
        try:
            barrier.wait()  # both threads alive before any id is observed
            with serialize:
                results[lane_id] = (threading.get_ident(), call(per_thread_stream))
                torch.cuda.synchronize()
            barrier.wait()  # still alive, so neither id can have been recycled
        except BaseException as exc:  # surfaced in the main thread below
            errors[lane_id] = exc
            barrier.abort()

    threads = [threading.Thread(target=lane, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"worker thread(s) raised: {errors}"
    assert len({ident for ident, _ in results.values()}) == 2, "test setup: the two worker threads must have distinct live ids"

    added = set(cache.keys()) - keys_before
    assert len(added) == 2, (
        "two host threads passing cudaStreamPerThread must key two separate plans "
        "(the handle is the integer 2 in both threads but denotes a different stream in each, "
        "and the per-plan ticket counter cannot be shared across concurrent streams); "
        f"got {len(added)} new plan-cache entries"
    )

    # Correctness guard on the keyed path: d_index_q / d_weights are
    # deterministic, so each thread must reproduce the main-thread reference
    # bitwise; d_index_k is fp32-atomic and only reproducible within its band.
    for lane_id, (_, result) in sorted(results.items()):
        assert torch.equal(result["d_index_q"], ref["d_index_q"]), f"per-thread stream lane {lane_id}: d_index_q diverged"
        assert torch.equal(result["d_weights"], ref["d_weights"]), f"per-thread stream lane {lane_id}: d_weights diverged"
        assert _rms_rel(result["d_index_k"], ref["d_index_k"].double()) < _DK_ATOMIC_BAND, f"per-thread stream lane {lane_id}: d_index_k diverged"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper_v2_multi_device(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    request,
):
    """Per-device plan/workspace ownership: the wrapper keys its plan cache
    on the CUDA device, so same-shape default-stream calls on two devices get
    private plans (the ticket-counter workspace is device-resident, so sharing
    one cached plan across devices would hand device 1 the device-0 counter).
    Run the identical input bits on each device, then interleave the devices,
    checking dq/dw bitwise against each device's serial result and dk in the
    fp32-atomic class; finally, executing a plan with tensors on the wrong
    device must raise ValueError before kernel 1 mutates the score buffers."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_exact_sm100()

    if torch.cuda.device_count() < 2:
        pytest.skip("2+ CUDA devices required")
    if torch.cuda.get_device_capability(1) != (10, 0):
        pytest.skip("SM100 required on the second device")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=512,
        block_I=block_I,
        min_compute_capability=100,
        s_q_default=128,
        s_kv_default=512,
    )
    sm_scale = 1.0
    loss_coeff = float(cfg["b"] * cfg["s_q"])
    n_iters = 3

    with torch.cuda.device(0):
        base = _allocate(cfg, sm_scale=sm_scale)
    # identical input bits on both devices
    inputs = {0: base, 1: tuple(t.to("cuda:1") for t in base)}
    grad_loss = {dev: torch.ones((), dtype=torch.float32, device=f"cuda:{dev}") for dev in (0, 1)}

    def call(dev):
        index_q, weights, index_k, attn_score, index_score, topk_indices = inputs[dev]
        with torch.cuda.device(dev):
            # stream=None on purpose: the default-stream path shares one
            # stream_key, so only the device key separates the two plans
            r = DSA.indexer_backward_wrapper(
                index_q,
                weights,
                index_k,
                attn_score.clone(),
                index_score.clone(),
                topk_indices,
                sm_scale=sm_scale,
                loss_coeff=loss_coeff,
                grad_loss=grad_loss[dev],
                block_I=block_I,
                backend="sm100_v2",
            )
            torch.cuda.synchronize()
        return r

    # each device serially: without the device in the cache key, device 1 would
    # reuse device 0's plan and with it device 0's ticket counter
    refs = {dev: call(dev) for dev in (0, 1)}

    # identical bits + deterministic dq/dw => cross-device bitwise equality,
    # asserted only when both devices report the same model name (on mixed
    # models this test checks d_index_k alone)
    if torch.cuda.get_device_properties(0).name == torch.cuda.get_device_properties(1).name:
        assert torch.equal(refs[1]["d_index_q"].cpu(), refs[0]["d_index_q"].cpu()), "cross-device d_index_q diverged"
        assert torch.equal(refs[1]["d_weights"].cpu(), refs[0]["d_weights"].cpu()), "cross-device d_weights diverged"
    assert _rms_rel(refs[1]["d_index_k"].cpu(), refs[0]["d_index_k"].cpu().double()) < _DK_ATOMIC_BAND, "cross-device d_index_k diverged"

    # interleave the devices: every call must keep hitting its own plan
    for it in range(n_iters):
        for dev in (0, 1):
            r = call(dev)
            assert torch.equal(r["d_index_q"], refs[dev]["d_index_q"]), f"device {dev} iter {it}: d_index_q diverged"
            assert torch.equal(r["d_weights"], refs[dev]["d_weights"]), f"device {dev} iter {it}: d_weights diverged"
            assert _rms_rel(r["d_index_k"], refs[dev]["d_index_k"].double()) < _DK_ATOMIC_BAND, f"device {dev} iter {it}: d_index_k diverged"

    # direct-object contract: a plan built on device 0 must reject device-1
    # tensors BEFORE kernel 1 mutates the score buffers
    iq0, w0, ik0, attn0, index0, tki0 = inputs[0]
    with torch.cuda.device(0):
        plan = DSA.IndexerBackward(
            sample_index_q=iq0,
            sample_weights=w0,
            sample_index_k=ik0,
            sample_d_index_q=torch.empty_like(iq0),
            sample_d_weights=torch.empty_like(w0),
            sample_d_index_k=torch.empty_like(ik0),
            sample_attn_score=attn0,
            sample_index_score=index0,
            sample_topk_indices=tki0,
            sm_scale=sm_scale,
            block_I=block_I,
            backend="sm100_v2",
        )
        assert plan.check_support()
        plan.compile()
    iq1, w1, ik1, attn1, index1, tki1 = inputs[1]
    attn1 = attn1.clone()
    index1 = index1.clone()
    attn1_before = attn1.clone()
    index1_before = index1.clone()
    with torch.cuda.device(1):
        with pytest.raises(ValueError, match="device"):
            plan.execute(
                iq1,
                w1,
                ik1,
                torch.empty_like(iq1),
                torch.empty_like(w1),
                torch.empty_like(ik1),
                attn1,
                index1,
                tki1,
                grad_loss=grad_loss[1],
                loss_coeff=loss_coeff,
            )
        torch.cuda.synchronize()
    assert torch.equal(attn1, attn1_before), "cross-device rejection must not mutate attn_score"
    assert torch.equal(index1, index1_before), "cross-device rejection must not mutate index_score"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_v2_plan_device_capability(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    request,
    monkeypatch,
):
    """Architecture gating for ``backend="sm100_v2"`` must query the plan's
    (sample-tensor) device, not whatever CUDA device is current.

    The plan's kernel and its device-resident workspace belong to the sample
    tensors' device, so on a heterogeneous machine a valid
    ``backend="sm100_v2"`` plan whose tensors live on an SM100 device must not
    be rejected because an unrelated pre-SM90 device happens to be current. The
    failure this pins is ``IndexerBackward.check_support`` reading a *param-less*
    ``torch.cuda.get_device_capability()`` in its generic SM90/SM100+ gate,
    which runs BEFORE the device-bound SM100 gate and would raise "requires
    SM90 or SM100+, found SM8" for a perfectly valid plan.

    A real SM100 device cannot be made to report a pre-SM90 capability, so the
    test patches only the *current-device* (no-argument) capability query to
    report SM80 while every *device-bound* query keeps returning the real
    capability. That is what separates current-device capability from
    sample-device capability. (The default backend deliberately keeps the
    param-less query: its factories check the current device themselves.)
    """
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_exact_sm100()

    # Build a real, valid backend="sm100_v2" plan on the true (SM100) device
    # BEFORE patching, so dsa_init's own capability gate and the tensor
    # allocation observe the real hardware.
    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=512,
        block_I=block_I,
        min_compute_capability=100,
        s_q_default=128,
        s_kv_default=512,
    )
    index_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=1.0)
    plan = DSA.IndexerBackward(
        sample_index_q=index_q,
        sample_weights=weights,
        sample_index_k=index_k,
        sample_d_index_q=torch.empty_like(index_q),
        sample_d_weights=torch.empty_like(weights),
        sample_d_index_k=torch.empty_like(index_k),
        sample_attn_score=attn_score,
        sample_index_score=index_score,
        sample_topk_indices=topk_indices,
        sm_scale=1.0,
        block_I=block_I,
        backend="sm100_v2",
    )
    sample_dev = index_q.device
    real_get_cap = torch.cuda.get_device_capability
    assert real_get_cap(sample_dev)[0] >= 9, "test presumes an SM90+ sample device"

    # Model a heterogeneous machine: no-arg (current-device) query -> SM80;
    # any device-bound query -> the real capability.
    pre_sm90 = (8, 0)

    def fake_get_device_capability(device=None):
        if device is None:
            return pre_sm90
        return real_get_cap(device)

    monkeypatch.setattr(torch.cuda, "get_device_capability", fake_get_device_capability)

    # The patch genuinely separates the two devices: a param-less gate WOULD
    # reject and a device-bound gate WOULD accept. Guards against a no-op patch
    # silently passing the test.
    assert torch.cuda.get_device_capability() == pre_sm90
    assert torch.cuda.get_device_capability(sample_dev)[0] >= 9

    # Both of ``check_support``'s architecture gates read the sample device for
    # this backend -> accepted. A param-less generic gate would read SM8 and
    # raise "requires SM90 or SM100+, found SM8".
    assert plan.check_support() is True


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper_invalid_backend(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    request,
):
    """An unknown ``backend`` value is rejected with a ValueError that lists the
    legal set, before any score buffer is touched. Covers both the public
    wrapper and the directly-constructed IndexerBackward object. Nothing here is
    SM100-specific (no v2 plan is ever built), so it gates on SM90 and runs on
    every supported architecture."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=512,
        block_I=block_I,
        min_compute_capability=90,
        s_q_default=128,
        s_kv_default=512,
    )
    loss_coeff = float(cfg["b"] * cfg["s_q"])
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    index_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=1.0)

    attn_before = attn_score.clone()
    with pytest.raises(ValueError, match="backend must be one of"):
        DSA.indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            sm_scale=1.0,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            block_I=block_I,
            backend="not_a_backend",  # not a legal enum value
        )
    torch.cuda.synchronize()
    assert torch.equal(attn_score, attn_before), "invalid-backend rejection must not mutate attn_score"

    # A directly-constructed object rejects the value in __init__ too.
    with pytest.raises(ValueError, match="backend must be one of"):
        DSA.IndexerBackward(
            sample_index_q=index_q,
            sample_weights=weights,
            sample_index_k=index_k,
            sample_d_index_q=torch.empty_like(index_q),
            sample_d_weights=torch.empty_like(weights),
            sample_d_index_k=torch.empty_like(index_k),
            sample_attn_score=attn_score,
            sample_index_score=index_score,
            sample_topk_indices=topk_indices,
            sm_scale=1.0,
            block_I=block_I,
            backend="nope",
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_backend_sm100_v2_requires_sm100(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    request,
    monkeypatch,
):
    """``backend="sm100_v2"`` is capability-gated (request-or-fail, no silent
    fallback): on a non-SM100 device check_support raises. The capability query
    is patched to report SM90 (9, 0) for the plan's device — the generic
    SM90/SM100+ gate still passes at that capability, so the raise is
    attributable to the sm100_v2-specific SM100 gate — and check_support must
    raise naming the backend. The plan and its tensors are built on the real
    hardware BEFORE the patch, and no v2 kernel is ever compiled or launched, so
    this gates on SM90 and runs on every supported architecture (including a
    real SM90 device, where the patch is a no-op)."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=512,
        block_I=block_I,
        min_compute_capability=90,
        s_q_default=128,
        s_kv_default=512,
    )
    index_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=1.0)
    plan = DSA.IndexerBackward(
        sample_index_q=index_q,
        sample_weights=weights,
        sample_index_k=index_k,
        sample_d_index_q=torch.empty_like(index_q),
        sample_d_weights=torch.empty_like(weights),
        sample_d_index_k=torch.empty_like(index_k),
        sample_attn_score=attn_score,
        sample_index_score=index_score,
        sample_topk_indices=topk_indices,
        sm_scale=1.0,
        block_I=block_I,
        backend="sm100_v2",
    )

    # Simulate a non-SM100 (but still generically supported SM90) plan device:
    # the generic gate passes, so only the sm100_v2 SM100 gate can fire.
    sm90 = (9, 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: sm90)
    assert torch.cuda.get_device_capability(index_q.device) == sm90

    attn_before = attn_score.clone()
    with pytest.raises(RuntimeError, match="sm100_v2"):
        plan.check_support()
    assert torch.equal(attn_score, attn_before), "capability rejection must not mutate attn_score"


# The frozen ``develop`` positional-or-keyword parameter order of
# ``indexer_backward_wrapper`` (the public signature before ``backend`` was
# added). Legacy positional callers depend on this exact order; ``backend``
# must be appended keyword-only so it never perturbs a positional binding.
_DEVELOP_POSITIONAL_PARAMS = [
    "index_q",
    "weights",
    "index_k",
    "attn_score",
    "index_score",
    "topk_indices",
    "grad_loss",
    "sm_scale",
    "loss_coeff",
    "block_I",
    "topk_indices_global",
    "d_index_q",
    "d_weights",
    "d_index_k",
    "stream",
]


@pytest.mark.L0
def test_DSA_indexer_backward_wrapper_backend_keyword_only_signature():
    """``backend`` is appended keyword-only after the develop parameter list, so
    a develop-era positional call keeps binding
    ``d_index_q``/``d_weights``/``d_index_k``/``stream`` to the right
    parameters. Inserting ``backend`` ahead of those optionals would bind a
    caller's ``d_index_q`` tensor to ``backend`` instead."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    sig = inspect.signature(DSA.indexer_backward_wrapper)
    params = sig.parameters

    positional = [
        name
        for name, p in params.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        )
    ]
    assert positional == _DEVELOP_POSITIONAL_PARAMS, (
        "positional parameter order must match develop exactly and backend " f"must not appear among positional params; got {positional}"
    )

    assert "backend" in params
    assert params["backend"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["backend"].default == "default"

    # A legacy positional call in develop order binds the trailing tensors to
    # d_index_q/d_weights/d_index_k/stream (not to backend) and leaves backend
    # at its default.
    sentinels = [object() for _ in _DEVELOP_POSITIONAL_PARAMS]
    bound = sig.bind(*sentinels)
    for name, value in zip(_DEVELOP_POSITIONAL_PARAMS, sentinels):
        assert bound.arguments[name] is value
    assert "backend" not in bound.arguments  # untouched -> default "default"

    # backend is reachable only by keyword: a 16th positional arg is a
    # TypeError.
    with pytest.raises(TypeError):
        sig.bind(*(sentinels + [object()]))
    bound_kw = sig.bind(*sentinels, backend="sm100_v2")
    assert bound_kw.arguments["backend"] == "sm100_v2"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper_legacy_positional_call(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    request,
):
    """A develop-era positional call that supplies
    ``d_index_q``/``d_weights``/``d_index_k``/``stream`` positionally (no
    ``backend`` kwarg) still binds them correctly and runs the default
    backend. Were ``backend`` not keyword-only-and-last, position 12 would
    bind the ``d_index_q`` tensor to ``backend`` and raise
    ``ValueError('backend must be one of ...')``."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        block_I=block_I,
        min_compute_capability=90,
        s_q_default=128,
        s_kv_default=512,
    )
    sm_scale = 1.0
    b_cfg = cfg["b"]
    s_q_cfg = cfg["s_q"]
    loss_coeff = float(b_cfg * s_q_cfg)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")

    (
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
    ) = _allocate(cfg, sm_scale=sm_scale)

    # Caller-supplied output buffers, passed POSITIONALLY in develop order.
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights)
    d_index_k = torch.empty_like(index_k)

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    torch_stream.wait_stream(torch.cuda.current_stream())
    # Run the develop-era positional call directly -- NO failure-to-skip here.
    # The only legitimate skips are environmental and already happened above:
    # the cudnn[cutedsl] ImportError guard and the dsa_init SM90+ capability
    # gate. Every parametrized shape (head_dim=128 / qhead=64 / block_I=128,
    # s_q=128 / s_kv=512) is inside the default backend's envelope, so any
    # error raised by the wrapper past this point is a real regression and
    # MUST fail the suite -- it must not be swallowed into a skip. In
    # particular, this is the guard for the positional-API contract: if
    # `backend` ever stops being keyword-only-and-last, the trailing develop
    # positional args rebind onto `backend`, the invalid value raises
    # ``ValueError("backend must be one of ...")``, and this call fails instead
    # of silently skipping.
    result = DSA.indexer_backward_wrapper(
        index_q,  # index_q
        weights,  # weights
        index_k,  # index_k
        attn_score,  # attn_score
        index_score,  # index_score
        topk_indices,  # topk_indices
        grad_loss,  # grad_loss
        sm_scale,  # sm_scale
        loss_coeff,  # loss_coeff
        block_I,  # block_I
        False,  # topk_indices_global
        d_index_q,  # d_index_q (would bind to backend if backend moved)
        d_weights,  # d_weights
        d_index_k,  # d_index_k
        stream,  # stream
    )
    torch_stream.synchronize()

    # The positional buffers must be exactly the ones the wrapper filled and
    # returned -- i.e. the develop-order positional args bound to their
    # intended parameters, not onto the keyword-only ``backend``.
    assert result["d_index_q"] is d_index_q
    assert result["d_weights"] is d_weights
    assert result["d_index_k"] is d_index_k
    assert torch.isfinite(d_index_q.float()).all()
    assert torch.isfinite(d_weights.float()).all()
    assert torch.isfinite(d_index_k.float()).all()
