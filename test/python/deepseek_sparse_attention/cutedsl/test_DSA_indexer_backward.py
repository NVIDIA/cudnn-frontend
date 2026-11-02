# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import math

import pytest
import torch
import torch.nn.functional as F

import inspect

from test_utils import assert_bitwise_runs, torch_fork_set_rng

from cudnn.api_base import ws_align
from deepseek_sparse_attention.cutedsl.dsa_utils import dsa_init, with_dsa_indexer_backward_params, _require_sm100_family, _require_sm100
from deepseek_sparse_attention.cutedsl.dsa_reference import (
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


def _require_indexer_backward_gpu():
    """Skip unless the current GPU runs the sm90 or sm100 kernels."""
    from cudnn.deepseek_sparse_attention.indexer_backward.deterministic_dk_sm100 import INDEXER_BACKWARD_SM100_CAPABILITIES

    if not torch.cuda.is_available():
        pytest.skip("IndexerBackward requires a CUDA GPU")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 9 and capability not in INDEXER_BACKWARD_SM100_CAPABILITIES:
        pytest.skip(f"IndexerBackward requires SM90 or capability in {INDEXER_BACKWARD_SM100_CAPABILITIES}, found {capability}")


# A non-unit softmax scale exercises the kernels' sm_scale fold (SM100 folds it
# into dS; sm_scale == 1 is bit-identical either way) and keeps it runtime-only:
# both values share one compiled kernel per shape.
_SM_SCALE_PARAMS = pytest.mark.parametrize("sm_scale", [1.0, 128**-0.5], ids=["sm_scale_1", "sm_scale_rsqrt_d"])

# The analytic grad-signal reference leaves a noise floor well below these
# bounds, while a mis-applied sm_scale (or a dropped/doubled fold) measures far
# outside them; the helper's loose defaults would not catch a 1.2x scale error.
_REF_TOL = dict(cos_min=0.99, rms_rel_max=0.02)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
@_SM_SCALE_PARAMS
def test_DSA_indexer_backward_wrapper(
    sm_scale,
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
    _require_indexer_backward_gpu()
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

    # The score-grad stage overwrites attn_score but treats index_score as
    # read-only. Keep pre-call copies for both reference input and the
    # preservation check.
    attn_score_ref = attn_score.clone()
    index_score_ref = index_score.clone()
    torch_stream.wait_stream(torch.cuda.current_stream())
    # Every parametrized shape is inside the supported envelope on the gated
    # architectures, so a raise here is a regression, not a skip.
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
    torch_stream.synchronize()

    assert torch.equal(index_score, index_score_ref)

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
            **_REF_TOL,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@_SM_SCALE_PARAMS
def test_DSA_indexer_backward_wrapper_compact_h32(sm_scale):
    """SM100 keeps the public tensors at H32 while using an internal M64 tile."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100()

    cfg = {
        "b": 1,
        "s_q": 128,
        "s_kv": 512,
        "head_dim": 128,
        "qhead_per_kv_head": 32,
        "topk": 512,
    }
    index_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=sm_scale)
    attn_score_ref = attn_score.clone()
    index_score_ref = index_score.clone()

    # Back the compact outputs with canary tails. The internal M64 dQ tile
    # must be clipped by the H32 tensor map, and dW must predicate heads 32:64.
    guard_elems = 256
    guard_value = 23.0
    dq_storage = torch.full((index_q.numel() + guard_elems,), guard_value, dtype=index_q.dtype, device=index_q.device)
    dw_storage = torch.full((weights.numel() + guard_elems,), guard_value, dtype=weights.dtype, device=weights.device)
    d_index_q = dq_storage[: index_q.numel()].view_as(index_q)
    d_weights = dw_storage[: weights.numel()].view_as(weights)
    d_index_k = torch.empty_like(index_k, dtype=torch.float32)

    result = DSA.indexer_backward_wrapper(
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
        sm_scale=sm_scale,
        loss_coeff=float(cfg["b"] * cfg["s_q"]),
        grad_loss=torch.ones((), dtype=torch.float32, device=index_q.device),
        block_I=128,
        d_index_q=d_index_q,
        d_weights=d_weights,
        d_index_k=d_index_k,
    )

    assert result["d_index_q"] is d_index_q
    assert result["d_weights"] is d_weights
    assert result["d_index_k"] is d_index_k
    torch.testing.assert_close(dq_storage[index_q.numel() :], torch.full_like(dq_storage[index_q.numel() :], guard_value), rtol=0.0, atol=0.0)
    torch.testing.assert_close(dw_storage[weights.numel() :], torch.full_like(dw_storage[weights.numel() :], guard_value), rtol=0.0, atol=0.0)
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
        grad_scale=1.0,
        **_REF_TOL,
    )


@pytest.mark.L0
@pytest.mark.parametrize(
    "h,d,block_I,topk,sm_scale,match",
    [
        pytest.param(48, 128, 128, 128, 1.0, "requires H=32 or H=64", id="h48"),
        pytest.param(128, 128, 128, 128, 1.0, "requires H=32 or H=64", id="h128"),
        pytest.param(64, 64, 128, 128, 1.0, "requires D=128", id="d64"),
        pytest.param(64, 128, 64, 128, 1.0, "requires block_I=128", id="block_i64"),
        pytest.param(64, 128, 128, 192, 1.0, "multiple of block_I", id="topk192"),
        pytest.param(64, 128, 128, 128, 0.0, "finite sm_scale > 0", id="sm_scale_zero"),
        pytest.param(64, 128, 128, 128, -0.5, "finite sm_scale > 0", id="sm_scale_negative"),
        pytest.param(64, 128, 128, 128, float("nan"), "finite sm_scale > 0", id="sm_scale_nan"),
    ],
)
def test_DSA_indexer_backward_sm100_rejects_unsupported_plans(h, d, block_I, topk, sm_scale, match):
    """check_support admits only what the SM100 kernel serves natively: H32/H64,
    D=128, block_I=128, whole top-k blocks, and a finite positive sm_scale (the
    relu gate reads the scaled score). Nothing reaches a kernel assert."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100()

    b, s_q, s_k = 1, 1, 256
    plan = DSA.IndexerBackward(
        sample_index_q=torch.empty((b, s_q, h, d), dtype=torch.bfloat16, device="cuda"),
        sample_weights=torch.empty((b, s_q, h), dtype=torch.bfloat16, device="cuda"),
        sample_index_k=torch.empty((b, s_k, d), dtype=torch.bfloat16, device="cuda"),
        sample_d_index_q=torch.empty((b, s_q, h, d), dtype=torch.bfloat16, device="cuda"),
        sample_d_weights=torch.empty((b, s_q, h), dtype=torch.bfloat16, device="cuda"),
        sample_d_index_k=torch.empty((b, s_k, d), dtype=torch.bfloat16, device="cuda"),
        sample_attn_score=torch.empty((b, s_q, topk), dtype=torch.float32, device="cuda"),
        sample_index_score=torch.empty((b, s_q, topk), dtype=torch.float32, device="cuda"),
        sample_topk_indices=torch.empty((b, s_q, topk), dtype=torch.int32, device="cuda"),
        sm_scale=sm_scale,
        block_I=block_I,
    )
    with pytest.raises(ValueError, match=match):
        plan.check_support()


@pytest.mark.L0
def test_DSA_indexer_backward_packed_shapes_reuse_compilation(monkeypatch):
    """GitHub #1188: packed training changes (B, Sq, Sk) every step; once a TopK bucket's
    kernels are compiled, new shapes must run without compiling again."""
    _require_sm100()
    try:
        import cutlass.cute as cute
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("indexer backward recompiled for a new packed shape")

    for i, (s_q, s_kv) in enumerate([(64, 1024), (160, 3072), (96, 2048)]):
        cfg = dict(b=1, s_q=s_q, s_kv=s_kv, head_dim=128, qhead_per_kv_head=64, topk=512)
        index_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=1.0)
        attn_score_ref, index_score_ref = attn_score.clone(), index_score.clone()
        result = DSA.indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            sm_scale=1.0,
            loss_coeff=float(s_q),
            grad_loss=torch.ones((), dtype=torch.float32, device="cuda"),
            block_I=128,
        )
        torch.cuda.synchronize()
        check_ref_indexer_backward(
            index_q,
            weights,
            index_k,
            attn_score_ref,
            index_score_ref,
            topk_indices,
            result["d_index_q"],
            result["d_weights"],
            result["d_index_k"],
            sm_scale=1.0,
            grad_scale=1.0,
        )
        if i == 0:
            monkeypatch.setattr(cute, "compile", forbidden_compile)


# ===========================================================================
# Regression coverage for the output/plan-signature validation on the default
# indexer backward (SM100/SM90):
#   * illegal output dtypes raise BEFORE kernel 1 overwrites attn_score;
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
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    return DSA


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


def _assert_rejected_before_mutation(check, call, attn_score, index_score):
    """``call`` raises ValueError and leaves both score buffers untouched.

    Kernel 1 overwrites ``attn_score`` in place, so a check that runs after
    it is fail-dirty even when it raises; ``index_score`` is read-only.
    """
    snapshots = (attn_score.clone(), index_score.clone())
    with pytest.raises(ValueError):
        call()
    torch.cuda.synchronize()
    for name, buffer, before in zip(("attn_score", "index_score"), (attn_score, index_score), snapshots):
        assert torch.equal(buffer, before), f"{name} mutated before the {check} (fail-dirty)"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_backward_illegal_output_dtype_raises_before_mutation():
    """Each illegal output dtype raises ValueError, and the raise happens
    before kernel 1 overwrites attn_score (index_score is always read-only)."""
    _require_sm100()
    DSA = _import_dsa()

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
        outputs = {}
        if "d_index_q_dtype" in kw:
            outputs["d_index_q"] = torch.empty_like(iq, dtype=kw["d_index_q_dtype"])
        if "d_weights_dtype" in kw:
            outputs["d_weights"] = torch.empty_like(w, dtype=kw["d_weights_dtype"])
        if "d_index_k_dtype" in kw:
            outputs["d_index_k"] = torch.empty_like(ik, dtype=kw["d_index_k_dtype"])

        _assert_rejected_before_mutation(
            f"{label} dtype check",
            lambda: DSA.indexer_backward_wrapper(
                iq, w, ik, attn, idx, tk, sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128, **outputs
            ),
            attn,
            idx,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_backward_fp32_dindexk_zeroed_internally():
    """An fp32 d_index_k buffer is zeroed inside execute (the dK epilogue
    atomic-adds), so the result is independent of the buffer's incoming
    contents — a nonzero buffer does NOT return old + gradient."""
    _require_sm100()
    DSA = _import_dsa()

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


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_backward_bf16_dindexk_matches_fp32_accumulator():
    """A bf16 d_index_k accumulates in private column-permuted scratch and is
    un-permuted by the trailing cast; an fp32 d_index_k is accumulated in the
    natural layout. Both must hold the same gradient, column for column."""
    _require_sm100()
    DSA = _import_dsa()

    iq, w, ik, attn, idx, tk, loss_coeff, grad_loss = _idxbwd_inputs()
    common = dict(sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128)

    def run(dk_dtype):
        d_index_k = torch.empty_like(ik, dtype=dk_dtype)
        return DSA.indexer_backward_wrapper(iq, w, ik, attn.clone(), idx.clone(), tk, d_index_k=d_index_k, **common)["d_index_k"]

    dk_fp32 = run(torch.float32)
    dk_bf16 = run(torch.bfloat16)
    torch.cuda.synchronize()

    assert dk_bf16.dtype == torch.bfloat16
    assert dk_fp32.abs().amax() > 0
    # Only the FP32 atomic order differs between the two runs; a wrong column
    # order would move whole values, far outside one bf16 rounding step.
    torch.testing.assert_close(dk_bf16.float(), dk_fp32.to(torch.bfloat16).float(), rtol=2**-7, atol=1e-3 * dk_fp32.abs().amax().item())


def _workspace_for(plan, device="cuda"):
    """The caller-owned scratch a plan carves at execute (R2): ``None`` when it needs none."""
    nbytes = plan.scratch_workspace_bytes()
    return torch.empty(nbytes, dtype=torch.uint8, device=device) if nbytes else None


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
    clean ValueError BEFORE kernel 1 overwrites attn_score (no fail-dirty)."""
    _require_sm100()
    DSA = _import_dsa()

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
    ws = _workspace_for(plan)

    b, s_q, _, _ = iq.shape
    topk = tk.shape[-1]

    def run(attn_score, d_weights=None):
        d_weights = torch.empty_like(w) if d_weights is None else d_weights
        idx_copy = idx.clone()
        return (
            lambda: plan.execute(
                iq, w, ik, torch.empty_like(iq), d_weights, torch.empty_like(ik), attn_score, idx_copy, tk, grad_loss, loss_coeff=loss_coeff, workspace=ws
            ),
            idx_copy,
        )

    # Case 1: wrong-shaped attn_score (extra top-k column) holding meaningful data.
    aq_bad = torch.softmax(torch.randn(b, s_q, topk + 1, device=iq.device), dim=-1).float()
    call, isc = run(aq_bad)
    _assert_rejected_before_mutation("shape check", call, aq_bad, isc)
    # Case 2: non-contiguous attn_score (right shape, strided storage).
    aq_nc = _noncontiguous_like(attn)
    call, isc = run(aq_nc)
    _assert_rejected_before_mutation("stride check", call, aq_nc, isc)
    # Case 3: a non-contiguous OUTPUT (d_weights) is rejected before mutation too.
    aq = attn.clone()
    call, isc = run(aq, d_weights=_noncontiguous_like(torch.empty_like(w)))
    _assert_rejected_before_mutation("output-stride check", call, aq, isc)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_backward_wrapper_cache_hit_stride_mismatch_raises():
    """A wrapper cache hit (identical shape/dtype/device key) still rejects a
    non-contiguous score buffer via the cached plan's execute-time signature
    check, before kernel 1 mutates either score buffer (no fail-dirty)."""
    _require_sm100()
    DSA = _import_dsa()

    iq, w, ik, attn, idx, tk, loss_coeff, grad_loss = _idxbwd_inputs()

    # First call with contiguous tensors populates the wrapper cache (its
    # descriptor is contiguous).
    DSA.indexer_backward_wrapper(iq, w, ik, attn.clone(), idx.clone(), tk, sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128)
    torch.cuda.synchronize()

    # Second call: identical shape/dtype/device (cache hit) but a
    # non-contiguous attn_score. The cached plan must raise before mutation.
    aq_nc = _noncontiguous_like(attn)
    isc = idx.clone()
    _assert_rejected_before_mutation(
        "cache-hit stride check",
        lambda: DSA.indexer_backward_wrapper(iq, w, ik, aq_nc, isc, tk, sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128),
        aq_nc,
        isc,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_backward_wrong_shape_plan_rejected_before_kernel1():
    """A first-call / directly-built plan whose output/score shape is
    inconsistent with ``index_q`` is rejected by the semantic shape validation
    in ``check_support`` BEFORE kernel 1 overwrites attn_score — not
    silently run on a mismatched signature that only faults in the GEMM. Covers
    both the wrapper (cache miss) and a directly-built plan. Fail-hard."""
    _require_sm100()
    DSA = _import_dsa()

    iq, w, ik, attn, idx, tk, loss_coeff, grad_loss = _idxbwd_inputs()
    b, s_q, h, _ = iq.shape
    topk = tk.shape[-1]

    # (a) Wrapper first call (cache miss) with a wrong-shaped d_weights output
    #     buffer. The plan is built from the mismatched sample, and check_support
    #     must raise before any kernel launch; the score buffers stay pristine.
    aq = attn.clone()
    isc = idx.clone()
    dw_bad = torch.empty(b, s_q, h + 1, dtype=torch.bfloat16, device=iq.device)
    _assert_rejected_before_mutation(
        "wrong-shape check",
        lambda: DSA.indexer_backward_wrapper(
            iq, w, ik, aq, isc, tk, sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128, d_weights=dw_bad
        ),
        aq,
        isc,
    )

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
    DSA = _import_dsa()

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
    #     rejected before kernel 1 overwrites attn_score.
    aq = attn.clone()
    isc = idx.clone()
    _assert_rejected_before_mutation(
        "non-contiguous check",
        lambda: DSA.indexer_backward_wrapper(iq, w, ik_nc, aq, isc, tk, sm_scale=_IDXBWD_SM_SCALE, loss_coeff=loss_coeff, grad_loss=grad_loss, block_I=128),
        aq,
        isc,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_backward_wrong_ndim_rejected():
    """A wrong-rank ``index_q`` / ``index_k`` / ``topk_indices`` is rejected
    with a clean ValueError (not a cryptic tuple-unpack error) by both the
    wrapper (before any allocation) and a directly-built plan."""
    _require_sm100()
    DSA = _import_dsa()

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


# ===========================================================================
# Rule 8 detectors (python/cudnn/AGENTS.md R2 / R9): the plan owns no device
# memory. The fp32 dK accumulator behind a bf16 d_index_k is carved from the
# caller's workspace; execute() allocates nothing and never blocks the host.
# ===========================================================================


def _build_plan(DSA, dk_dtype, inputs):
    """A checked, compiled IndexerBackward plus its output buffers (bf16 dW: the
    kernel stores dW in bf16 only)."""
    iq, w, ik, attn, idx, tk, _, _ = inputs
    outputs = (torch.empty_like(iq), torch.empty_like(w), torch.empty_like(ik, dtype=dk_dtype))
    plan = DSA.IndexerBackward(
        sample_index_q=iq,
        sample_weights=w,
        sample_index_k=ik,
        sample_d_index_q=outputs[0],
        sample_d_weights=outputs[1],
        sample_d_index_k=outputs[2],
        sample_attn_score=attn,
        sample_index_score=idx,
        sample_topk_indices=tk,
        sm_scale=_IDXBWD_SM_SCALE,
        block_I=128,
    )
    assert plan.check_support()
    return plan, outputs


_DK_DTYPE_PARAMS = pytest.mark.parametrize("dk_dtype", [torch.bfloat16, torch.float32], ids=["bf16_d_index_k", "fp32_d_index_k"])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@_DK_DTYPE_PARAMS
def test_execute_allocates_nothing_and_never_synchronizes(dk_dtype, compile_allocates_nothing):
    """R9: with the caller's workspace, three warm executes leave the torch
    caching allocator's allocation count unchanged; the rule8_detector conftest hook arms
    ``torch.cuda.set_sync_debug_mode("error")`` around every execute, so a
    host sync fails the test too. ``compile()`` allocates nothing (R11)."""
    _require_indexer_backward_gpu()
    DSA = _import_dsa()
    inputs = _idxbwd_inputs()
    iq, w, ik, attn, idx, tk, loss_coeff, grad_loss = inputs
    plan, (dq, dw, dk) = _build_plan(DSA, dk_dtype, inputs)
    compile_allocates_nothing(plan)
    ws = _workspace_for(plan)
    if dk_dtype == torch.bfloat16:
        assert ws is not None, "this configuration carves scratch and must declare it"

    # kernel 1 overwrites attn_score: one pristine copy per execute, all made
    # before the allocation accounting starts
    attns = [attn.clone() for _ in range(4)]

    def run(aq):
        plan.execute(iq, w, ik, dq, dw, dk, aq, idx, tk, grad_loss, loss_coeff=loss_coeff, workspace=ws)

    # warm: the factories cute.compile their kernels lazily on the first call
    run(attns[0])
    torch.cuda.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    for aq in attns[1:]:
        run(aq)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations, "IndexerBackward.execute allocated device memory (Rule 8, R2)"
    for name, t in (("d_index_q", dq), ("d_weights", dw), ("d_index_k", dk)):
        assert torch.isfinite(t.float()).all(), f"{name} contains NaN/Inf"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_bf16_d_index_k_requires_workspace():
    """A bf16 d_index_k needs the ``B * S_k * D`` fp32 accumulator:
    ``scratch_workspace_bytes()`` says so,
    ``execute(workspace=None)`` raises the WorkspaceCarver error before kernel 1
    overwrites attn_score, and an undersized buffer is rejected the same way.
    An fp32 d_index_k needs no scratch at all."""
    _require_indexer_backward_gpu()
    DSA = _import_dsa()
    inputs = _idxbwd_inputs()
    iq, w, ik, attn, idx, tk, loss_coeff, grad_loss = inputs
    b, s_k, d = ik.shape

    plan, (dq, dw, dk) = _build_plan(DSA, torch.bfloat16, inputs)
    plan.compile()
    expected = ws_align(b * s_k * d * 4)
    assert plan.scratch_workspace_bytes() == expected

    aq = attn.clone()
    aq_pre = aq.clone()
    with pytest.raises(ValueError, match=r"requires a \d+-byte workspace"):
        plan.execute(iq, w, ik, dq, dw, dk, aq, idx, tk, grad_loss, loss_coeff=loss_coeff, workspace=None)
    short = torch.empty(expected - 128, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match=r"requires a \d+-byte workspace"):
        plan.execute(iq, w, ik, dq, dw, dk, aq, idx, tk, grad_loss, loss_coeff=loss_coeff, workspace=short)
    torch.cuda.synchronize()
    assert torch.equal(aq, aq_pre), "missing workspace must be rejected before kernel 1 mutates attn_score (fail-dirty)"

    plan32, (dq32, dw32, dk32) = _build_plan(DSA, torch.float32, inputs)
    plan32.compile()
    assert plan32.scratch_workspace_bytes() == 0
    plan32.execute(iq, w, ik, dq32, dw32, dk32, attn.clone(), idx, tk, grad_loss, loss_coeff=loss_coeff)
    torch.cuda.synchronize()
    assert torch.isfinite(dk32).all()


@pytest.mark.parametrize(
    "dk_dtype,repeats,topk_indices_global",
    [
        pytest.param(torch.float32, 3, False, marks=pytest.mark.L0, id="fp32-dk-local-ids"),
        pytest.param(torch.float32, 3, True, marks=pytest.mark.L0, id="fp32-dk-global-ids"),
        pytest.param(torch.bfloat16, 3, False, marks=pytest.mark.L1, id="bf16-dk-local-ids"),
        pytest.param(torch.float32, 1000, False, marks=pytest.mark.L2, id="fp32-dk-repeat1000"),
    ],
)
@torch_fork_set_rng(seed=1701)
def test_DSA_indexer_backward_deterministic_contract(dk_dtype, repeats, topk_indices_global):
    """The opt-in path owns/reset its exact scratch and reproduces every
    output bitwise across the fixed producer-grid boundary."""
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_backward.deterministic_dk_sm100 import (
            NUM_DK_PRODUCER_CTAS,
            NUM_DK_PLANES,
            deterministic_dk_workspace_bytes,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100_family()

    cfg = dict(b=2, s_q=NUM_DK_PRODUCER_CTAS // 2 + 1, s_kv=257, head_dim=128, qhead_per_kv_head=64, topk=512)
    index_q, weights, index_k, attn_source, index_score, topk_indices = _allocate(cfg, sm_scale=1.0)
    reference_topk_indices = topk_indices
    if topk_indices_global:
        batch_offsets = torch.arange(cfg["b"], dtype=torch.int32, device="cuda").view(cfg["b"], 1, 1) * cfg["s_kv"]
        topk_indices = torch.where(topk_indices >= 0, topk_indices + batch_offsets, topk_indices)
    attn = attn_source.clone()
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights, dtype=torch.bfloat16)
    d_index_k = torch.empty_like(index_k, dtype=dk_dtype)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    plan = DSA.IndexerBackward(
        sample_index_q=index_q,
        sample_weights=weights,
        sample_index_k=index_k,
        sample_d_index_q=d_index_q,
        sample_d_weights=d_weights,
        sample_d_index_k=d_index_k,
        sample_attn_score=attn,
        sample_index_score=index_score,
        sample_topk_indices=topk_indices,
        sm_scale=1.0,
        block_I=128,
        topk_indices_global=topk_indices_global,
        deterministic=True,
    )
    assert plan.check_support()
    plan.compile()
    expected = deterministic_dk_workspace_bytes(cfg["b"], cfg["s_kv"], cfg["head_dim"])
    assert expected == ws_align(NUM_DK_PLANES * cfg["b"] * cfg["s_kv"] * cfg["head_dim"] * 4)
    assert plan.scratch_workspace_bytes() == expected
    workspace = torch.empty(expected, dtype=torch.uint8, device="cuda")

    # Workspace validation is fail-clean: score scratch has not been touched.
    attn.copy_(attn_source)
    with pytest.raises(ValueError, match="workspace"):
        plan.execute(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k,
            attn,
            index_score,
            topk_indices,
            grad_loss,
            loss_coeff=float(cfg["b"] * cfg["s_q"]),
            workspace=workspace[:-1],
        )
    torch.testing.assert_close(attn, attn_source, rtol=0.0, atol=0.0)

    def run():
        attn.copy_(attn_source)
        workspace.fill_(0xA5)
        d_index_q.fill_(float("nan"))
        d_weights.fill_(float("nan"))
        d_index_k.fill_(float("nan"))
        plan.execute(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k,
            attn,
            index_score,
            topk_indices,
            grad_loss,
            loss_coeff=float(cfg["b"] * cfg["s_q"]),
            workspace=workspace,
        )
        torch.cuda.synchronize()
        return tuple(t.clone() for t in (d_index_q, d_weights, d_index_k, attn))

    reference = run()
    for repeat in range(repeats):
        actual = run()
        for name, actual_tensor, reference_tensor in zip(("d_index_q", "d_weights", "d_index_k", "grad_signal"), actual, reference):
            assert torch.equal(actual_tensor, reference_tensor), f"{name} changed at repeat {repeat}"

    check_ref_indexer_backward(
        index_q,
        weights,
        index_k,
        attn_source,
        index_score,
        reference_topk_indices,
        reference[0],
        reference[1],
        reference[2],
        sm_scale=1.0,
        grad_scale=1.0,
    )


@pytest.mark.L1
@pytest.mark.parametrize(
    "dk_dtype,topk_indices_global,topk,seqlen_k",
    [
        pytest.param(torch.float32, False, 512, 512, id="topk512-fp32-dk-local-ids"),
        pytest.param(torch.bfloat16, True, 512, 512, id="topk512-bf16-dk-global-ids"),
        pytest.param(torch.float32, True, 2048, 2048, id="topk2048-fp32-dk-global-ids"),
    ],
)
@torch_fork_set_rng(seed=1711)
def test_DSA_indexer_backward_deterministic_compact_h32(dk_dtype, topk_indices_global, topk, seqlen_k):
    """Compact H32 preserves its public extent across fixed producer handoffs."""
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_backward.deterministic_dk_sm100 import deterministic_dk_workspace_bytes
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100_family()

    # With 96 planes, 194 rows makes producer zero revisit its dK plane twice.
    cfg = dict(b=2, s_q=97, s_kv=seqlen_k, head_dim=128, qhead_per_kv_head=32, topk=topk)
    index_q, weights, index_k, attn_source, index_score, topk_indices = _allocate(cfg, sm_scale=1.0)
    reference_topk_indices = topk_indices
    if topk_indices_global:
        batch_offsets = torch.arange(cfg["b"], dtype=torch.int32, device="cuda").view(cfg["b"], 1, 1) * cfg["s_kv"]
        topk_indices = torch.where(topk_indices >= 0, topk_indices + batch_offsets, topk_indices)
    attn = attn_source.clone()
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    # Match the unit grad_scale used by the numerical reference:
    # loss_coeff / (B * S_q) == 1.0.
    loss_coeff = float(cfg["b"] * cfg["s_q"])

    # Canary tails ensure the internal M64 tile cannot escape the compact H32
    # dQ/dW stores while exercising the deterministic persistent-row schedule.
    guard_elems = 256
    guard_value = 23.0
    dq_storage = torch.full((index_q.numel() + guard_elems,), guard_value, dtype=index_q.dtype, device=index_q.device)
    dw_storage = torch.full((weights.numel() + guard_elems,), guard_value, dtype=weights.dtype, device=weights.device)
    d_index_q = dq_storage[: index_q.numel()].view_as(index_q)
    d_weights = dw_storage[: weights.numel()].view_as(weights)
    d_index_k = torch.empty_like(index_k, dtype=dk_dtype)
    workspace = torch.empty(deterministic_dk_workspace_bytes(cfg["b"], cfg["s_kv"], cfg["head_dim"]), dtype=torch.uint8, device="cuda")

    def run():
        attn.copy_(attn_source)
        workspace.fill_(0xA5)
        for output in (d_index_q, d_weights, d_index_k):
            output.fill_(float("nan"))
        DSA.indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn,
            index_score,
            topk_indices,
            grad_loss=grad_loss,
            loss_coeff=loss_coeff,
            d_index_q=d_index_q,
            d_weights=d_weights,
            d_index_k=d_index_k,
            workspace=workspace,
            deterministic=True,
            topk_indices_global=topk_indices_global,
        )
        return tuple(t.clone() for t in (d_index_q, d_weights, d_index_k, attn))

    assert_bitwise_runs(run, label=f"default H32 {dk_dtype=} {topk_indices_global=}")
    torch.testing.assert_close(dq_storage[index_q.numel() :], torch.full_like(dq_storage[index_q.numel() :], guard_value), rtol=0.0, atol=0.0)
    torch.testing.assert_close(dw_storage[weights.numel() :], torch.full_like(dw_storage[weights.numel() :], guard_value), rtol=0.0, atol=0.0)
    check_ref_indexer_backward(
        index_q,
        weights,
        index_k,
        attn_source,
        index_score,
        reference_topk_indices,
        d_index_q,
        d_weights,
        d_index_k,
        sm_scale=1.0,
        grad_scale=1.0,
        cos_min=0.99,
        rms_rel_max=0.02,
    )


@pytest.mark.L1
@pytest.mark.parametrize(
    "dk_planes,s_q,dk_dtype",
    [
        pytest.param(128, 128, torch.float32, id="planes128-fp32-single-row-stripe"),
        pytest.param(148, 148, torch.float32, id="planes148-fp32-single-row-stripe"),
        pytest.param(148, 149, torch.bfloat16, id="planes148-bf16-reused-plane"),
    ],
)
@torch_fork_set_rng(seed=1702)
def test_DSA_indexer_backward_deterministic_plane_specializations(dk_planes, s_q, dk_dtype):
    """Larger compile-time grids own exact scratch and remain deterministic.

    The BF16 case has one more query row than planes, so CTA/plane zero is
    reused before the fixed-order reducer converts its accumulator.
    """
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_backward.deterministic_dk_sm100 import deterministic_dk_workspace_bytes
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100_family()

    cfg = dict(b=1, s_q=s_q, s_kv=128, head_dim=128, qhead_per_kv_head=64, topk=128)
    index_q, weights, index_k, attn_source, index_score, topk_indices = _allocate(cfg, sm_scale=1.0)
    attn = attn_source.clone()
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights, dtype=torch.bfloat16)
    d_index_k = torch.empty_like(index_k, dtype=dk_dtype)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    plan = DSA.IndexerBackward(
        sample_index_q=index_q,
        sample_weights=weights,
        sample_index_k=index_k,
        sample_d_index_q=d_index_q,
        sample_d_weights=d_weights,
        sample_d_index_k=d_index_k,
        sample_attn_score=attn,
        sample_index_score=index_score,
        sample_topk_indices=topk_indices,
        deterministic=True,
        deterministic_dk_planes=dk_planes,
    )
    assert plan.check_support()
    plan.compile()
    expected_bytes = deterministic_dk_workspace_bytes(cfg["b"], cfg["s_kv"], cfg["head_dim"], dk_planes)
    assert expected_bytes == ws_align(dk_planes * cfg["b"] * cfg["s_kv"] * cfg["head_dim"] * 4)
    assert plan.scratch_workspace_bytes() == expected_bytes
    workspace = torch.empty(expected_bytes, dtype=torch.uint8, device="cuda")

    def run():
        attn.copy_(attn_source)
        plan.execute(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k,
            attn,
            index_score,
            topk_indices,
            grad_loss,
            loss_coeff=float(cfg["s_q"]),
            workspace=workspace,
        )
        torch.cuda.synchronize()
        return tuple(t.clone() for t in (d_index_q, d_weights, d_index_k, attn))

    reference = run()
    actual = run()
    for name, actual_tensor, reference_tensor in zip(("d_index_q", "d_weights", "d_index_k", "grad_signal"), actual, reference):
        assert torch.equal(actual_tensor, reference_tensor), f"planes={dk_planes} changed {name}"

    check_ref_indexer_backward(
        index_q,
        weights,
        index_k,
        attn_source,
        index_score,
        topk_indices,
        reference[0],
        reference[1],
        reference[2],
        sm_scale=1.0,
        grad_scale=1.0,
    )


@pytest.mark.L0
@pytest.mark.parametrize("aliased_name", ["index_q", "d_index_k"])
@torch_fork_set_rng(seed=1703)
def test_DSA_indexer_backward_deterministic_workspace_alias_rejected_before_launch(aliased_name):
    """The live plane scratch cannot overlap a read or write operand."""
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_backward.deterministic_dk_sm100 import deterministic_dk_workspace_bytes
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100_family()

    cfg = dict(b=1, s_q=96, s_kv=128, head_dim=128, qhead_per_kv_head=64, topk=128)
    index_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=1.0)
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights)
    d_index_k = torch.empty_like(index_k)
    required = deterministic_dk_workspace_bytes(cfg["b"], cfg["s_kv"], cfg["head_dim"])
    workspace = torch.empty(required, dtype=torch.uint8, device="cuda")

    if aliased_name == "index_q":
        aliased = workspace[: index_q.numel() * index_q.element_size()].view(index_q.dtype).view_as(index_q)
        aliased.copy_(index_q)
        index_q = aliased
    else:
        aliased = workspace[: d_index_k.numel() * d_index_k.element_size()].view(d_index_k.dtype).view_as(d_index_k)
        aliased.zero_()
        d_index_k = aliased

    plan = DSA.IndexerBackward(
        index_q,
        weights,
        index_k,
        d_index_q,
        d_weights,
        d_index_k,
        attn_score,
        index_score,
        topk_indices,
        deterministic=True,
    )

    def launch_tripwire(*args, **kwargs):
        raise AssertionError("overlapping workspace reached the compiled kernel")

    plan._compiled_kernel = launch_tripwire
    attn_before = attn_score.clone()
    aliased_before = aliased.clone()
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match=rf"workspace must not overlap {aliased_name}"):
        plan.execute(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k,
            attn_score,
            index_score,
            topk_indices,
            grad_loss,
            loss_coeff=float(cfg["s_q"]),
            workspace=workspace,
        )
    assert torch.equal(attn_score, attn_before)
    assert torch.equal(aliased, aliased_before)


@pytest.mark.L0
@torch_fork_set_rng(seed=1704)
def test_DSA_indexer_backward_deterministic_workspace_allows_disjoint_view_of_same_allocation():
    """Only the live scratch prefix aliases; a disjoint shared-storage tail is valid."""
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_backward.deterministic_dk_sm100 import deterministic_dk_workspace_bytes
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100_family()

    cfg = dict(b=1, s_q=96, s_kv=128, head_dim=128, qhead_per_kv_head=64, topk=128)
    source_q, weights, index_k, attn_score, index_score, topk_indices = _allocate(cfg, sm_scale=1.0)
    required = deterministic_dk_workspace_bytes(cfg["b"], cfg["s_kv"], cfg["head_dim"])
    q_bytes = source_q.numel() * source_q.element_size()
    backing = torch.empty(required + q_bytes, dtype=torch.uint8, device="cuda")
    workspace = backing
    index_q = backing[required:].view(source_q.dtype).view_as(source_q)
    index_q.copy_(source_q)
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights)
    d_index_k = torch.empty_like(index_k)
    plan = DSA.IndexerBackward(
        index_q,
        weights,
        index_k,
        d_index_q,
        d_weights,
        d_index_k,
        attn_score,
        index_score,
        topk_indices,
        deterministic=True,
    )

    class LaunchIntercepted(Exception):
        pass

    def launch_tripwire(*args, **kwargs):
        raise LaunchIntercepted

    plan._compiled_kernel = launch_tripwire
    with pytest.raises(LaunchIntercepted):
        plan.execute(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k,
            attn_score,
            index_score,
            topk_indices,
            torch.ones((), dtype=torch.float32, device="cuda"),
            loss_coeff=float(cfg["s_q"]),
            workspace=workspace,
        )


@pytest.mark.L0
def test_DSA_indexer_backward_deterministic_plane_policy_rejects_invalid_values_before_tensors():
    """Validate plane selection and the nonempty fixed-owner grid before tensors."""
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_backward.deterministic_dk_sm100 import deterministic_dk_producer_ctas
        from cudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_sm100 import indexer_backward_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    args = (None,) * 7
    for value in (0, 95, 97, 149, 96.0, True, None):
        with pytest.raises(ValueError, match="deterministic_dk_planes must be one of"):
            DSA.indexer_backward_wrapper(*args, deterministic=True, deterministic_dk_planes=value)
    with pytest.raises(ValueError, match="only configurable when deterministic=True"):
        DSA.indexer_backward_wrapper(*args, deterministic=False, deterministic_dk_planes=128)
    for total_rows in (0, -1):
        with pytest.raises(ValueError, match="at least one query row"):
            deterministic_dk_producer_ctas(total_rows)
    with pytest.raises(ValueError, match="only configurable when deterministic=True"):
        indexer_backward_sm100(1, 1, 128, 64, 128, 128, deterministic=False, deterministic_dk_planes=128)
    with pytest.raises(ValueError, match="at least one query row"):
        indexer_backward_sm100(1, 0, 128, 64, 128, 128, deterministic=True)


@pytest.mark.L0
def test_DSA_indexer_backward_deterministic_empty_query_rejected_before_allocation(monkeypatch):
    """Reject an empty deterministic grid before output allocation or kernel dispatch."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to build IndexerBackward tensor descriptors")

    b, s_q, s_k, h, d, topk = 1, 0, 128, 64, 128, 128
    index_q = torch.empty((b, s_q, h, d), dtype=torch.bfloat16, device="cuda")
    weights = torch.empty((b, s_q, h), dtype=torch.bfloat16, device="cuda")
    index_k = torch.empty((b, s_k, d), dtype=torch.bfloat16, device="cuda")
    attn_score = torch.empty((b, s_q, topk), dtype=torch.float32, device="cuda")
    index_score = torch.empty_like(attn_score)
    topk_indices = torch.empty((b, s_q, topk), dtype=torch.int32, device="cuda")
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights)
    d_index_k = torch.empty_like(index_k)

    plan = DSA.IndexerBackward(
        index_q,
        weights,
        index_k,
        d_index_q,
        d_weights,
        d_index_k,
        attn_score,
        index_score,
        topk_indices,
        deterministic=True,
    )
    with pytest.raises(ValueError, match="at least one query row"):
        plan.check_support()

    monkeypatch.setattr(
        torch,
        "empty_like",
        lambda *args, **kwargs: pytest.fail("output allocation happened before deterministic row validation"),
    )
    with pytest.raises(ValueError, match="at least one query row"):
        DSA.indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            grad_loss,
            deterministic=True,
        )


@pytest.mark.L1
@pytest.mark.parametrize("s_q", [7, 17], ids=["short-grid", "uneven-grid"])
@torch_fork_set_rng(seed=1703)
def test_DSA_indexer_backward_deterministic_cuda_graph(s_q):
    """Short and uneven warmed plans with fixed scratch are replay-safe."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100_family()

    cfg = dict(b=1, s_q=s_q, s_kv=128, head_dim=128, qhead_per_kv_head=64, topk=128)
    index_q, weights, index_k, attn_source, index_score, topk_indices = _allocate(cfg, sm_scale=1.0)
    attn = attn_source.clone()
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights, dtype=torch.bfloat16)
    d_index_k = torch.empty_like(index_k, dtype=torch.float32)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    plan = DSA.IndexerBackward(
        sample_index_q=index_q,
        sample_weights=weights,
        sample_index_k=index_k,
        sample_d_index_q=d_index_q,
        sample_d_weights=d_weights,
        sample_d_index_k=d_index_k,
        sample_attn_score=attn,
        sample_index_score=index_score,
        sample_topk_indices=topk_indices,
        deterministic=True,
    )
    assert plan.check_support()
    plan.compile()
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")

    def launch():
        plan.execute(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k,
            attn,
            index_score,
            topk_indices,
            grad_loss,
            loss_coeff=float(cfg["s_q"]),
            workspace=workspace,
        )

    # Compile every lazy kernel before capture.
    attn.copy_(attn_source)
    launch()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            attn.copy_(attn_source)
            launch()
        torch.cuda.synchronize()
        expected = tuple(t.clone() for t in (d_index_q, d_weights, d_index_k, attn))

        for repeat in range(3):
            workspace.fill_(0xA5)
            d_index_q.fill_(float("nan"))
            d_weights.fill_(float("nan"))
            d_index_k.fill_(float("nan"))
            attn.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for name, actual, reference in zip(("d_index_q", "d_weights", "d_index_k", "grad_signal"), (d_index_q, d_weights, d_index_k, attn), expected):
                assert torch.equal(actual, reference), f"graph replay changed {name} at repeat {repeat}"
    finally:
        graph.reset()


@pytest.mark.L1
@pytest.mark.parametrize("s_q,topk", [(2, 2048), (193, 896), (193, 2048)], ids=["short-k2048", "persistent-k896", "persistent-k2048"])
@torch_fork_set_rng(seed=1705)
def test_DSA_indexer_backward_deterministic_topk2048_boundary(s_q, topk):
    """Large TopK and repeated producer-row handoffs preserve every output bit."""
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_backward.deterministic_dk_sm100 import deterministic_dk_workspace_bytes
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100_family()

    # With 96 planes, 193 rows makes a producer consume three different rows.
    cfg = dict(b=1, s_q=s_q, s_kv=2048, head_dim=128, qhead_per_kv_head=64, topk=topk)
    index_q, weights, index_k, attn_source, index_score, topk_indices = _allocate(cfg, sm_scale=1.0)
    attn = attn_source.clone()
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights, dtype=torch.bfloat16)
    d_index_k = torch.empty_like(index_k, dtype=torch.float32)
    workspace = torch.empty(deterministic_dk_workspace_bytes(cfg["b"], cfg["s_kv"], cfg["head_dim"]), dtype=torch.uint8, device="cuda")

    def run():
        attn.copy_(attn_source)
        workspace.fill_(0xA5)
        for output in (d_index_q, d_weights, d_index_k):
            output.fill_(float("nan"))
        DSA.indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn,
            index_score,
            topk_indices,
            grad_loss=grad_loss,
            loss_coeff=float(cfg["s_q"]),
            d_index_q=d_index_q,
            d_weights=d_weights,
            d_index_k=d_index_k,
            workspace=workspace,
            deterministic=True,
        )
        return tuple(t.clone() for t in (d_index_q, d_weights, d_index_k, attn))

    assert_bitwise_runs(run, label=f"s_q={s_q} topk={topk}")
    check_ref_indexer_backward(
        index_q,
        weights,
        index_k,
        attn_source,
        index_score,
        topk_indices,
        d_index_q,
        d_weights,
        d_index_k,
        sm_scale=1.0,
        grad_scale=1.0,
    )


_DETERMINISTIC_CAPABILITY_ERROR = (ValueError, "deterministic IndexerBackward requires capability")


@pytest.mark.L0
@pytest.mark.parametrize(
    "capability,error",
    [
        pytest.param((9, 0), _DETERMINISTIC_CAPABILITY_ERROR, id="sm90"),
        pytest.param((10, 3), None, id="sm103"),
        pytest.param((10, 7), _DETERMINISTIC_CAPABILITY_ERROR, id="sm107"),
    ],
)
@torch_fork_set_rng(seed=1707)
def test_DSA_indexer_backward_deterministic_capability_gate(monkeypatch, capability, error):
    """Deterministic execution accepts SM100/SM103 and fails closed elsewhere."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dict(b=1, s_q=2, s_kv=128, head_dim=128, qhead_per_kv_head=64, topk=128)
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
        deterministic=True,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)

    attn_before = attn_score.clone()
    if error is None:
        assert plan.check_support()
    else:
        with pytest.raises(error[0], match=error[1]):
            plan.check_support()
    assert torch.equal(attn_score, attn_before)


_ARCH_ERROR = (RuntimeError, "IndexerBackward requires SM90 or capability in")


@pytest.mark.L0
@pytest.mark.parametrize(
    "capability,heads,error",
    [
        pytest.param((9, 0), 64, None, id="sm90-h64"),
        pytest.param((9, 0), 32, (ValueError, "SM90 IndexerBackward requires H=64"), id="sm90-h32"),
        pytest.param((9, 0), 128, (ValueError, "SM90 IndexerBackward requires H=64"), id="sm90-h128"),
        pytest.param((10, 0), 32, None, id="sm100-h32"),
        pytest.param((10, 3), 64, None, id="sm103-h64"),
        pytest.param((10, 7), 64, _ARCH_ERROR, id="sm107"),
        pytest.param((12, 0), 64, _ARCH_ERROR, id="sm120"),
        pytest.param((8, 0), 64, _ARCH_ERROR, id="sm80"),
    ],
)
@torch_fork_set_rng(seed=1708)
def test_DSA_indexer_backward_architecture_gate(monkeypatch, capability, heads, error):
    """Ordinary execution admits SM90 and the validated SM100-family devices only
    (the SM100 kernel is a TCGEN05/TMEM design), and applies each arch's head
    contract, all before ``attn_score`` is touched."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to build IndexerBackward tensor descriptors")

    cfg = dict(b=1, s_q=2, s_kv=128, head_dim=128, qhead_per_kv_head=heads, topk=128)
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
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)

    attn_before = attn_score.clone()
    if error is None:
        assert plan.check_support()
    else:
        with pytest.raises(error[0], match=error[1]):
            plan.check_support()
    assert torch.equal(attn_score, attn_before)


# The frozen ``develop`` positional-or-keyword parameter order of
# ``indexer_backward_wrapper``. Legacy positional callers depend on this exact
# order; newer options are appended keyword-only so they never perturb a
# positional binding.
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
def test_DSA_indexer_backward_wrapper_keyword_only_signature():
    """The newer options are appended keyword-only after the develop parameter
    list, so a develop-era positional call keeps binding
    ``d_index_q``/``d_weights``/``d_index_k``/``stream`` to the right
    parameters."""
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
    assert positional == _DEVELOP_POSITIONAL_PARAMS, f"positional parameter order must match develop exactly; got {positional}"

    assert params["workspace"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["deterministic"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["deterministic"].default is False
    assert params["deterministic_dk_planes"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["deterministic_dk_planes"].default == 96
    keyword_only = [name for name, parameter in params.items() if parameter.kind is inspect.Parameter.KEYWORD_ONLY]
    assert keyword_only == ["workspace", "deterministic", "deterministic_dk_planes"]

    # A legacy positional call in develop order binds the trailing tensors to
    # d_index_q/d_weights/d_index_k/stream.
    sentinels = [object() for _ in _DEVELOP_POSITIONAL_PARAMS]
    bound = sig.bind(*sentinels)
    for name, value in zip(_DEVELOP_POSITIONAL_PARAMS, sentinels):
        assert bound.arguments[name] is value

    # The keyword-only options are reachable only by keyword: a 16th
    # positional arg is a TypeError.
    with pytest.raises(TypeError):
        sig.bind(*(sentinels + [object()]))
    bound_deterministic = sig.bind(*sentinels, deterministic=True)
    assert bound_deterministic.arguments["deterministic"] is True
    bound_planes = sig.bind(*sentinels, deterministic=True, deterministic_dk_planes=148)
    assert bound_planes.arguments["deterministic_dk_planes"] == 148


@pytest.mark.L0
def test_DSA_indexer_backward_plan_keyword_only_signature():
    """``IndexerBackward``'s execution-policy options are keyword-only after
    ``topk_indices_global``. That slot used to hold the removed ``backend``
    selector, so a stale positional caller must raise ``TypeError`` instead of
    binding its backend string to ``deterministic``."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    sig = inspect.signature(DSA.IndexerBackward.__init__)
    params = sig.parameters
    positional = [name for name, p in params.items() if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional == [
        "self",
        "sample_index_q",
        "sample_weights",
        "sample_index_k",
        "sample_d_index_q",
        "sample_d_weights",
        "sample_d_index_k",
        "sample_attn_score",
        "sample_index_score",
        "sample_topk_indices",
        "sm_scale",
        "block_I",
        "topk_indices_global",
    ]
    keyword_only = [name for name, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY]
    assert keyword_only == ["deterministic", "deterministic_dk_planes"]

    sentinels = [object() for _ in positional]
    with pytest.raises(TypeError):
        sig.bind(*sentinels, "default")


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
    ``d_index_q``/``d_weights``/``d_index_k``/``stream`` positionally still
    binds them correctly and runs."""
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
    # s_q=128 / s_kv=512) is inside the kernel's envelope, so any error raised
    # by the wrapper past this point is a real regression and MUST fail the
    # suite -- it must not be swallowed into a skip.
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
        d_index_q,  # d_index_q
        d_weights,  # d_weights
        d_index_k,  # d_index_k
        stream,  # stream
    )
    torch_stream.synchronize()

    # The positional buffers must be exactly the ones the wrapper filled and
    # returned -- i.e. the develop-order positional args bound to their
    # intended parameters.
    assert result["d_index_q"] is d_index_q
    assert result["d_weights"] is d_weights
    assert result["d_index_k"] is d_index_k
    assert torch.isfinite(d_index_q.float()).all()
    assert torch.isfinite(d_weights.float()).all()
    assert torch.isfinite(d_index_k.float()).all()


@pytest.mark.L1
def test_DSA_indexer_backward_sm100_persistent_short_topk_dispatch_threshold():
    """Use persistence only once a CTA can amortize setup across rows."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100+ required")

    try:
        from cudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_sm100 import IndexerBackwardSm100, _HAS_TMA_GATHER4
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    for topk in (128, 256, 384):
        short = IndexerBackwardSm100(
            128,
            heads=64,
            block_I=128,
            topk=topk,
            total_rows=sm_count,
            persistent_grid_size=sm_count,
            topk_indices_global=True,
        )
        long = IndexerBackwardSm100(
            128,
            heads=64,
            block_I=128,
            topk=topk,
            total_rows=sm_count + 1,
            persistent_grid_size=sm_count,
            topk_indices_global=True,
        )
        assert not short.use_persistent
        assert long.use_persistent
        assert short.use_tma_gather is _HAS_TMA_GATHER4
        assert long.use_tma_gather is _HAS_TMA_GATHER4
        assert long.use_cross_row_persistent is _HAS_TMA_GATHER4


@pytest.mark.L1
@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_DSA_indexer_backward_sm100_local_ids_use_tma_gather(topk):
    """Normalize local ids in-kernel without giving up the Gather4 path."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100+ required")

    try:
        from cudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_sm100 import IndexerBackwardSm100, _HAS_TMA_GATHER4
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    kernel = IndexerBackwardSm100(
        128,
        heads=64,
        block_I=128,
        topk=topk,
        total_rows=sm_count + 1,
        persistent_grid_size=sm_count,
        topk_indices_global=False,
    )
    assert kernel.use_tma_gather is _HAS_TMA_GATHER4
    assert kernel.use_cross_row_persistent is (_HAS_TMA_GATHER4 and topk == 512)


@pytest.mark.L1
@pytest.mark.parametrize("topk", [128, 256, 384, 512])
def test_DSA_indexer_backward_sm100_score_grad_packed_rows_and_fused_dk_zero(topk):
    """Packed tail rows preserve score math and clear the FP32 dK scratch."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100+ required")

    try:
        from cudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_sm100 import indexer_backward_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260826 + topk)
    batch, seqlen_q, seqlen_k, heads, head_dim = 2, 151, max(512, topk), 64, 128
    target = torch.softmax(
        torch.randn((batch, seqlen_q, topk), device=device, dtype=torch.float32, generator=generator),
        dim=-1,
    )
    predict = torch.softmax(
        torch.randn((batch, seqlen_q, topk), device=device, dtype=torch.float32, generator=generator),
        dim=-1,
    )
    target[:, ::23, ::31] = 0.0
    predict[:, ::29, ::37] = 0.0
    grad_loss = torch.tensor([0.7], device=device, dtype=torch.float32)
    grad_scale = 0.13
    actual = target.clone()
    d_index_k_f32 = torch.full(
        (batch, seqlen_k, head_dim),
        7.0,
        device=device,
        dtype=torch.float32,
    )

    kernel = indexer_backward_sm100(
        batch,
        seqlen_q,
        seqlen_k,
        heads,
        head_dim,
        topk,
        sm_scale=head_dim**-0.5,
        block_I=128,
        topk_indices_global=True,
    )
    kernel.score_grad(
        actual,
        predict,
        grad_loss,
        grad_scale,
        dIndexK_f32=d_index_k_f32,
    )
    torch.cuda.synchronize()

    clip_probability_min = math.exp(-100.0)
    grad = -target.clamp_min(clip_probability_min) * (predict >= clip_probability_min) * (grad_scale * grad_loss.item())
    expected = grad - predict * grad.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(d_index_k_f32, torch.zeros_like(d_index_k_f32), rtol=0.0, atol=0.0)


@pytest.mark.L1
@pytest.mark.parametrize("topk", [128, 256, 384])
def test_DSA_indexer_backward_sm100_score_grad_pdl_full_pipeline_matches_serial(topk):
    """PDL prologue overlap remains ordered across back-to-back calls."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100+ required")

    try:
        from cudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_sm100 import indexer_backward_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260827 + topk)
    batch, seqlen_q, seqlen_k, heads, head_dim = 1, 149, 512, 64, 128
    index_q = torch.randn((batch, seqlen_q, heads, head_dim), device=device, dtype=torch.bfloat16, generator=generator)
    index_k = torch.randn((batch, seqlen_k, head_dim), device=device, dtype=torch.bfloat16, generator=generator)
    weights = torch.randn((batch, seqlen_q, heads), device=device, dtype=torch.bfloat16, generator=generator).abs().mul_(0.1)
    topk_indices = torch.randint(0, seqlen_k, (batch, seqlen_q, topk), device=device, dtype=torch.int32, generator=generator)
    target = torch.softmax(torch.randn((batch, seqlen_q, topk), device=device, dtype=torch.float32, generator=generator), dim=-1)
    predict = torch.softmax(torch.randn((batch, seqlen_q, topk), device=device, dtype=torch.float32, generator=generator), dim=-1)
    grad_loss = torch.tensor([0.7], device=device, dtype=torch.float32)
    grad_scale = 0.13
    clip_probability_min = math.exp(-100.0)
    grad = -target.clamp_min(clip_probability_min) * (predict >= clip_probability_min) * (grad_scale * grad_loss.item())
    grad_signal = (grad - predict * grad.sum(dim=-1, keepdim=True)).contiguous()

    kernel = indexer_backward_sm100(
        batch,
        seqlen_q,
        seqlen_k,
        heads,
        head_dim,
        topk,
        sm_scale=head_dim**-0.5,
        block_I=128,
        topk_indices_global=True,
    )
    d_index_q_ref = torch.empty_like(index_q)
    d_weights_ref = torch.empty_like(weights)
    d_index_k_ref_f32 = torch.zeros_like(index_k, dtype=torch.float32)
    kernel.gemm_only(
        index_q,
        weights,
        index_k,
        d_index_q_ref,
        d_weights_ref,
        d_index_k_ref_f32,
        grad_signal,
        topk_indices,
    )
    d_index_k_ref = d_index_k_ref_f32.to(torch.bfloat16)

    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights)
    d_index_k = torch.empty_like(index_k)
    # bf16 d_index_k: the factory carves its fp32 accumulator from the caller's workspace (R2)
    workspace = torch.empty(ws_align(d_index_k.numel() * 4), dtype=torch.uint8, device=device)
    for _ in range(2):
        kernel(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k,
            target.clone(),
            predict,
            topk_indices,
            grad_loss,
            grad_scale,
            workspace=workspace,
        )
    torch.cuda.synchronize()

    for name, actual, expected in (
        ("dQ", d_index_q, d_index_q_ref),
        ("dW", d_weights, d_weights_ref),
        ("dK", d_index_k, d_index_k_ref),
    ):
        assert torch.isfinite(actual.float()).all(), f"{name} contains NaN/Inf"
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2, msg=lambda msg, name=name: f"{name}: {msg}")


@pytest.mark.L1
@pytest.mark.parametrize(
    "topk,topk_indices_global,force_manual_k_load",
    [
        pytest.param(topk, topk_indices_global, False, id=f"{'global' if topk_indices_global else 'local'}-{topk}")
        for topk_indices_global in (True, False)
        for topk in (128, 256, 384, 1024)
    ]
    + [
        pytest.param(512, False, False, id="local-512"),
        pytest.param(512, True, True, id="global-512-manual-k-load"),
    ],
)
@pytest.mark.parametrize("seqlen_q", [17, 151])
@pytest.mark.parametrize("dk_column_permuted", [False, True], ids=["dk-natural", "dk-permuted"])
def test_DSA_indexer_backward_sm100_gemm_matches_reference(
    dk_column_permuted,
    seqlen_q,
    topk,
    topk_indices_global,
    force_manual_k_load,
    monkeypatch,
    request,
):
    """Check persistent and one-row-per-CTA grids, the manual K-load fallback,
    invalid and duplicate IDs, and both dK accumulation layouts against the reference."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100+ required")

    try:
        indexer_backward_sm100_module = importlib.import_module("cudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_sm100")
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if force_manual_k_load:
        monkeypatch.setattr(indexer_backward_sm100_module, "_HAS_TMA_GATHER4", False)
        indexer_backward_sm100_module._compile_cache.clear()
        request.addfinalizer(indexer_backward_sm100_module._compile_cache.clear)
    indexer_backward_sm100 = indexer_backward_sm100_module.indexer_backward_sm100

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260825 + topk + int(topk_indices_global))
    batch, seqlen_k, heads, head_dim = 2, 512, 64, 128
    sm_scale = head_dim**-0.5
    index_q = torch.randn((batch, seqlen_q, heads, head_dim), device=device, dtype=torch.bfloat16, generator=generator)
    index_k = torch.randn((batch, seqlen_k, head_dim), device=device, dtype=torch.bfloat16, generator=generator)
    weights = torch.randn((batch, seqlen_q, heads), device=device, dtype=torch.bfloat16, generator=generator).abs().mul_(0.1)
    local_ids = torch.randint(0, seqlen_k, (batch, seqlen_q, topk), device=device, dtype=torch.int32, generator=generator)
    local_ids[:, ::17, ::19] = -1
    valid = local_ids >= 0
    batch_offsets = torch.arange(batch, device=device, dtype=torch.int32)[:, None, None] * seqlen_k
    global_ids = torch.where(valid, local_ids + batch_offsets, local_ids)
    if topk_indices_global:
        kernel_ids = global_ids.clone()
        kernel_ids[0, 1, 3] = batch * seqlen_k
    else:
        kernel_ids = local_ids.clone()
        kernel_ids[0, 1, 3] = seqlen_k
    local_ids[0, 1, 3] = -1
    global_ids[0, 1, 3] = -1
    valid[0, 1, 3] = False
    grad_signal = torch.randn((batch, seqlen_q, topk), device=device, dtype=torch.float32, generator=generator)

    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights)
    d_index_k = torch.zeros_like(index_k, dtype=torch.float32)
    kernel = indexer_backward_sm100(
        batch,
        seqlen_q,
        seqlen_k,
        heads,
        head_dim,
        topk,
        sm_scale=sm_scale,
        block_I=128,
        topk_indices_global=topk_indices_global,
    )
    kernel.gemm_only(index_q, weights, index_k, d_index_q, d_weights, d_index_k, grad_signal, kernel_ids, dk_column_permuted=dk_column_permuted)
    torch.cuda.synchronize()
    if dk_column_permuted:
        d_index_k_natural = torch.empty_like(d_index_k)
        indexer_backward_sm100_module.dk_column_permuted_view(d_index_k_natural).copy_(d_index_k.view(*d_index_k.shape[:-1], 8, 4, 2, 2))
        d_index_k = d_index_k_natural

    safe_ids = local_ids.clamp(min=0).long()
    selected_k = torch.gather(
        index_k[:, None].expand(-1, seqlen_q, -1, -1),
        2,
        safe_ids[..., None].expand(-1, -1, -1, head_dim),
    ).float()
    scores = torch.einsum("bqhd,bqtd->bqht", index_q.float(), selected_k) * sm_scale
    d_scores = grad_signal[:, :, None, :] * weights.float()[:, :, :, None] * ((scores > 0) & valid[:, :, None, :])
    d_index_q_ref = torch.einsum("bqht,bqtd->bqhd", d_scores, selected_k) * sm_scale
    d_weights_ref = (grad_signal[:, :, None, :] * torch.relu(scores) * valid[:, :, None, :]).sum(dim=-1)
    d_index_k_ref = torch.zeros_like(index_k, dtype=torch.float32)
    d_index_k_contrib = torch.einsum("bqht,bqhd->bqtd", d_scores, index_q.float()) * sm_scale
    d_index_k_ref.view(-1, head_dim).index_add_(
        0,
        global_ids.clamp(min=0).reshape(-1).long(),
        d_index_k_contrib.reshape(-1, head_dim),
    )

    for name, actual, expected in (
        ("dQ", d_index_q.float(), d_index_q_ref),
        ("dW", d_weights.float(), d_weights_ref),
        ("dK", d_index_k, d_index_k_ref),
    ):
        actual_flat = actual.flatten()
        expected_flat = expected.flatten()
        cosine = F.cosine_similarity(actual_flat, expected_flat, dim=0).item()
        rms_relative = ((actual_flat - expected_flat).square().mean().sqrt() / expected_flat.square().mean().sqrt().clamp_min(1e-12)).item()
        assert torch.isfinite(actual_flat).all(), f"{name} contains NaN/Inf"
        assert cosine >= 0.99, f"{name} cosine={cosine:.6f}"
        assert rms_relative <= 0.02, f"{name} RMS-relative error={rms_relative:.6f}"
