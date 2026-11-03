# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cudnn.deepseek_sparse_attention.indexer_backward.dense_indexer_backward_sm100 import (
    dense_indexer_backward_sm100,
)
from test_utils import torch_fork_set_rng

from deepseek_sparse_attention.cutedsl.dsa_utils import dsa_init, with_dsa_dense_indexer_backward_params
from deepseek_sparse_attention.cutedsl.dsa_reference import (
    _batched_ratio_causal_mask,
    check_ref_dense_indexer_backward,
    ref_dense_indexer_score_recompute,
)


def _allocate(cfg, sm_scale: float, ratio: int, q_causal_offsets: torch.Tensor):
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    h = cfg["qhead_per_kv_head"]
    device = "cuda"

    index_q = torch.randn(b, s_q, h, d, dtype=torch.bfloat16, device=device)
    weights = torch.randn(b, s_q, h, dtype=torch.bfloat16, device=device)
    index_k = torch.randn(b, s_k, d, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        index_score, index_lse = ref_dense_indexer_score_recompute(
            index_q,
            index_k.unsqueeze(2),
            weights,
            ratio=ratio,
            q_causal_offsets=q_causal_offsets,
        )
        valid = _batched_ratio_causal_mask(s_q, s_k, ratio, device, b, q_causal_offsets)
        if sm_scale != 1.0:
            index_score = index_score * sm_scale
            index_lse = torch.logsumexp(
                index_score.masked_fill(~valid, float("-inf")),
                dim=-1,
            )

        attn_score = torch.rand(b, s_q, s_k, dtype=torch.float32, device=device)
        attn_score = attn_score.masked_fill(~valid, 0.0).contiguous()
        attn_l1norm = attn_score.sum(dim=-1).contiguous()

    return (
        index_q,
        weights,
        index_k,
        attn_score.contiguous(),
        attn_l1norm,
        index_score.contiguous(),
        index_lse.contiguous(),
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_dense_indexer_backward_params
def test_DSA_dense_indexer_backward_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    ratio,
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
        ratio=ratio,
        min_compute_capability=90,
        s_q_default=128,
        s_kv_default=512,
    )
    sm_scale = 1.0
    b_cfg = cfg["b"]
    s_q_cfg = cfg["s_q"]
    q_causal_offsets = torch.full((b_cfg,), 8, dtype=torch.int32, device="cuda")
    loss_coeff = float(b_cfg * s_q_cfg)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    grad_scale_expected = loss_coeff / (b_cfg * s_q_cfg)

    (
        index_q,
        weights,
        index_k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
    ) = _allocate(cfg, sm_scale=sm_scale, ratio=ratio, q_causal_offsets=q_causal_offsets)
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    attn_score_ref = attn_score.clone()
    attn_l1norm_ref = attn_l1norm.clone()
    torch_stream.wait_stream(torch.cuda.current_stream())
    try:
        result = DSA.dense_indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            block_I=block_I,
            ratio=ratio,
            q_causal_offsets=q_causal_offsets,
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
        check_ref_dense_indexer_backward(
            index_q,
            weights,
            index_k,
            attn_score_ref,
            attn_l1norm_ref,
            d_index_q,
            d_weights,
            d_index_k,
            sm_scale=sm_scale,
            ratio=ratio,
            grad_scale=grad_scale_expected,
            q_causal_offsets=q_causal_offsets,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_dense_indexer_backward_cuda_graph():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Dense indexer backward requires SM90+")

    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = {
        "b": 1,
        "s_q": 128,
        "s_kv": 128,
        "head_dim": 128,
        "qhead_per_kv_head": 64,
    }
    (
        index_q,
        weights,
        index_k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
    ) = _allocate(cfg, sm_scale=1.0, ratio=1, q_causal_offsets=None)
    attn_score_source = attn_score.clone()
    index_score_source = index_score.clone()
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights)
    d_index_k = torch.empty_like(index_k, dtype=torch.float32)
    grad_loss = torch.ones(1, dtype=torch.float32, device="cuda")

    def run():
        attn_score.copy_(attn_score_source)
        index_score.copy_(index_score_source)
        DSA.dense_indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            loss_coeff=float(cfg["b"] * cfg["s_q"]),
            grad_loss=grad_loss,
            block_I=128,
            ratio=1,
            d_index_q=d_index_q,
            d_weights=d_weights,
            d_index_k=d_index_k,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        run()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for scale in (0.5, 1.5):
        grad_loss.fill_(scale)
        graph.replay()
        check_ref_dense_indexer_backward(
            index_q,
            weights,
            index_k,
            attn_score_source,
            attn_l1norm,
            d_index_q,
            d_weights,
            d_index_k,
            grad_scale=scale,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("dk_dtype", [torch.bfloat16, torch.float32], ids=["bf16_d_index_k", "fp32_d_index_k"])
def test_execute_allocates_nothing_and_never_synchronizes(dk_dtype, compile_allocates_nothing):
    """Rule 8 (R2 / R9): the fp32 dK accumulator behind a bf16 d_index_k is
    carved from the caller's workspace (``scratch_workspace_bytes()``), so
    three warm executes leave the torch caching allocator's allocation count
    unchanged; the rule8_detector conftest hook arms ``set_sync_debug_mode("error")`` around
    every execute. An fp32 d_index_k declares zero scratch. Without a workspace
    the bf16 plan raises before either score buffer is mutated."""
    major = torch.cuda.get_device_capability()[0]
    if major != 9 and major < 10:
        pytest.skip("Dense indexer backward requires SM90 or SM100+")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = {"b": 1, "s_q": 128, "s_kv": 128, "head_dim": 128, "qhead_per_kv_head": 64}
    index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse = _allocate(cfg, sm_scale=1.0, ratio=1, q_causal_offsets=None)
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights)
    d_index_k = torch.empty_like(index_k, dtype=dk_dtype)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    loss_coeff = float(cfg["b"] * cfg["s_q"])

    plan = DSA.DenseIndexerBackward(
        sample_index_q=index_q,
        sample_weights=weights,
        sample_index_k=index_k,
        sample_d_index_q=d_index_q,
        sample_d_weights=d_weights,
        sample_d_index_k=d_index_k,
        sample_attn_score=attn_score,
        sample_attn_l1norm=attn_l1norm,
        sample_index_score=index_score,
        sample_index_lse=index_lse,
        sm_scale=1.0,
        block_I=128,
        ratio=1,
    )
    assert plan.check_support()
    compile_allocates_nothing(plan)
    nbytes = plan.scratch_workspace_bytes()
    assert nbytes == (0 if dk_dtype == torch.float32 else index_k.numel() * 4)
    workspace = torch.empty(nbytes, dtype=torch.uint8, device="cuda") if nbytes else None

    # kernel 1 consumes both score buffers in place: pristine copies per
    # execute, all made before the allocation accounting starts
    scores = [(attn_score.clone(), index_score.clone()) for _ in range(4)]

    def run(attn, index, ws):
        plan.execute(
            index_q, weights, index_k, d_index_q, d_weights, d_index_k, attn, attn_l1norm, index, index_lse, grad_loss, loss_coeff=loss_coeff, workspace=ws
        )

    if workspace is not None:
        attn, index = attn_score.clone(), index_score.clone()
        with pytest.raises(ValueError, match=r"requires a \d+-byte workspace"):
            run(attn, index, None)
        torch.cuda.synchronize()
        assert torch.equal(attn, attn_score) and torch.equal(index, index_score), "missing workspace must be rejected before kernel 1 (fail-dirty)"

    run(*scores[0], workspace)  # warm: kernels cute.compile lazily on the first call
    torch.cuda.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    for attn, index in scores[1:]:
        run(attn, index, workspace)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations, "DenseIndexerBackward.execute allocated device memory (Rule 8, R2)"
    for name, t in (("d_index_q", d_index_q), ("d_weights", d_weights), ("d_index_k", d_index_k)):
        assert torch.isfinite(t.float()).all(), f"{name} contains NaN/Inf"
    check_ref_dense_indexer_backward(index_q, weights, index_k, attn_score, attn_l1norm, d_index_q, d_weights, d_index_k, grad_scale=1.0)


_DENSE_ARCH_ERROR = (RuntimeError, "DenseIndexerBackward requires SM90 or capability in")
_DENSE_HEADS_ERROR = (ValueError, "requires heads == 64")
_DENSE_SM90_TILE_ERROR = (ValueError, "SM90 DenseIndexerBackward requires D=128 and block_I=128")


@pytest.mark.L0
@pytest.mark.parametrize(
    "capability,heads,head_dim,block_I,error",
    [
        pytest.param((11, 0), 64, 128, 128, _DENSE_ARCH_ERROR, id="sm110"),
        pytest.param((12, 0), 64, 128, 128, _DENSE_ARCH_ERROR, id="sm120"),
        pytest.param((10, 7), 64, 128, 128, (RuntimeError, "SM107 requires a CuTe DSL build supporting sm_107a"), id="sm107-no-sm107a-dsl"),
        pytest.param((10, 0), 32, 128, 128, _DENSE_HEADS_ERROR, id="sm100-h32"),
        pytest.param((10, 0), 128, 128, 128, _DENSE_HEADS_ERROR, id="sm100-h128"),
        pytest.param((10, 0), 64, 96, 128, (ValueError, r"SM100 DenseIndexerBackward requires D in \(64, 128\)"), id="sm100-d96"),
        pytest.param((10, 0), 64, 128, 256, (ValueError, r"SM100 DenseIndexerBackward requires block_I in \(64, 128\)"), id="sm100-block_i256"),
        pytest.param((9, 0), 128, 128, 128, _DENSE_HEADS_ERROR, id="sm90-h128"),
        pytest.param((9, 0), 64, 64, 128, _DENSE_SM90_TILE_ERROR, id="sm90-d64"),
        pytest.param((9, 0), 64, 128, 64, _DENSE_SM90_TILE_ERROR, id="sm90-block_i64"),
    ],
)
@torch_fork_set_rng(seed=0)
def test_DSA_dense_indexer_backward_rejects_unsupported_plans(monkeypatch, capability, heads, head_dim, block_I, error):
    """check_support admits SM90 and the validated SM100-family devices only (SM107
    with a DSL that has the sm_107a target), and
    only the tiles each dense kernel serves: H64 on both
    arches (H96 is no legal TCGEN05 M and H128 exceeds the SM100 launch
    resources; SM90 WGMMA M is 64), D and block_I in {64, 128} on SM100 (TCGEN05
    M), and the validated D128 x I128 tile on SM90. Declines happen before any
    score buffer moves."""
    try:
        from cudnn import DSA
        import cudnn.frost.buffers as buffers
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to build DenseIndexerBackward tensor descriptors")

    cfg = dict(b=1, s_q=8, s_kv=128, head_dim=head_dim, qhead_per_kv_head=heads)
    index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse = _allocate(cfg, sm_scale=1.0, ratio=1, q_causal_offsets=None)
    attn_before, index_before = attn_score.clone(), index_score.clone()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    # SM107 admission additionally needs a DSL with the sm_107a target (Rule 7);
    # the SM107 case here checks the decline when that target is missing.
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    with pytest.raises(error[0], match=error[1]):
        DSA.dense_indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            grad_loss=torch.ones((), dtype=torch.float32, device="cuda"),
            block_I=block_I,
        )
    assert torch.equal(attn_score, attn_before)
    assert torch.equal(index_score, index_before)


@pytest.mark.L0
def test_dense_plan_rejects_cross_tensor_shape_mismatch():
    """A directly built plan validates its descriptors against each other in
    ``check_support`` (the wrapper's ``_dense_shapes`` contract): the kernel sizes
    its grid and the dK store from index_q / index_k, so a gradient, score or
    denominator buffer of another extent is rejected before compile, on both layouts."""
    major = torch.cuda.get_device_capability()[0]
    if major != 9 and major < 10:
        pytest.skip("Dense indexer backward requires SM90 or SM100+")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    b, s_q, s_k, h, d = 1, 128, 128, 64, 128
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    bshd = dict(
        sample_index_q=torch.empty(b, s_q, h, d, **bf16),
        sample_weights=torch.empty(b, s_q, h, **bf16),
        sample_index_k=torch.empty(b, s_k, d, **bf16),
        sample_d_index_q=torch.empty(b, s_q, h, d, **bf16),
        sample_d_weights=torch.empty(b, s_q, h, **bf16),
        sample_d_index_k=torch.empty(b, s_k, d, **bf16),
        sample_attn_score=torch.empty(b, s_q, s_k, **f32),
        sample_attn_l1norm=torch.empty(b, s_q, **f32),
        sample_index_score=torch.empty(b, s_q, s_k, **f32),
        sample_index_lse=torch.empty(b, s_q, **f32),
    )
    assert DSA.DenseIndexerBackward(**bshd).check_support()
    for name, bad in (
        ("d_index_k", torch.empty(b, s_k // 2, d, **bf16)),
        ("attn_score", torch.empty(b, s_q, 2 * s_k, **f32)),
        ("d_weights", torch.empty(b, s_q // 2, h, **bf16)),
        ("index_lse", torch.empty(b, 2 * s_q, **f32)),
    ):
        with pytest.raises(ValueError, match=name):
            DSA.DenseIndexerBackward(**{**bshd, f"sample_{name}": bad}).check_support()

    t_q, t_k = s_q, s_k
    thd = dict(
        sample_index_q=torch.empty(t_q, h, d, **bf16),
        sample_weights=torch.empty(t_q, h, **bf16),
        sample_index_k=torch.empty(t_k, d, **bf16),
        sample_d_index_q=torch.empty(t_q, h, d, **bf16),
        sample_d_weights=torch.empty(t_q, h, **bf16),
        sample_d_index_k=torch.empty(t_k, d, **bf16),
        sample_attn_score=torch.empty(t_q, s_k, **f32),
        sample_attn_l1norm=torch.empty(t_q, **f32),
        sample_index_score=torch.empty(t_q, s_k, **f32),
        sample_index_lse=torch.empty(t_q, **f32),
        is_thd=True,
        batch=1,
        max_seqlen_q=s_q,
        max_seqlen_k=s_k,
    )
    assert DSA.DenseIndexerBackward(**thd).check_support()
    for name, bad in (
        ("d_index_k", torch.empty(t_k // 2, d, **bf16)),
        ("index_score", torch.empty(t_q, 2 * s_k, **f32)),
        ("attn_l1norm", torch.empty(2 * t_q, **f32)),
    ):
        with pytest.raises(ValueError, match=name):
            DSA.DenseIndexerBackward(**{**thd, f"sample_{name}": bad}).check_support()


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(("queries", "keys"), [(1, 1), (17, 31), (32, 128), (33, 129), (32, 513)])
def test_DSA_dense_indexer_backward_staging_sync(queries: int, keys: int, head_dim: int) -> None:
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Dense indexer SM100 staging test requires Blackwell+")

    query = torch.ones(queries, 64, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.ones(keys, head_dim, device="cuda", dtype=query.dtype)
    weights = torch.full(query.shape[:2], 0.01, device="cuda", dtype=query.dtype)
    grad_signal = torch.full((queries, keys), 0.0001, device="cuda", dtype=torch.float32)
    grad_signal.masked_fill_(
        torch.arange(keys, device="cuda")[None, :] > torch.arange(queries, device="cuda")[:, None] + keys - queries,
        0,
    )
    cu_query = torch.tensor([0, queries], device="cuda", dtype=torch.int32)
    cu_key = torch.tensor([0, keys], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([keys - queries], device="cuda", dtype=torch.int32)
    query_grad = torch.empty_like(query)
    weights_grad = torch.empty_like(weights)
    key_grad = torch.zeros_like(key, dtype=torch.float32)
    kernel = dense_indexer_backward_sm100(
        1,
        queries,
        keys,
        64,
        head_dim,
        sm_scale=1.0,
        ratio=1,
        is_varlen=True,
        has_q_causal_offsets=True,
    )
    for sign in (1.0, -1.0, 1.0, -1.0):
        query.fill_(sign)
        key_grad.zero_()
        kernel.gemm_only(
            query,
            weights,
            key,
            query_grad,
            weights_grad,
            key_grad,
            grad_signal,
            cu_query,
            cu_key,
            offsets,
        )
        for name, gradient in (("dQ", query_grad), ("dK", key_grad), ("dW", weights_grad)):
            assert torch.isfinite(gradient).all(), f"Nonfinite {name}: {queries=}, {keys=}, {head_dim=}, {sign=}"
            if sign < 0:
                assert torch.count_nonzero(gradient).item() == 0, f"Negative QK requires zero {name}: {queries=}, {keys=}, {head_dim=}"
