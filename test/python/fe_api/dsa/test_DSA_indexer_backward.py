# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import dsa_init, with_dsa_indexer_backward_params, _require_sm100
from fe_api.dsa.dsa_reference import (
    _indexer_predict_distribution,
    check_ref_indexer_backward,
)


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
