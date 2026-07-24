"""Tests for SparseAttentionBackward.

The forward pass is a PyTorch reference (see dsa_reference.ref_sparse_attention_forward);
the production forward is FlashMLA (C++, out of scope). Gradients are generated
via autograd on the reference forward.
"""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import (
    _require_sm90,
    dsa_init,
    with_dsa_sparse_attention_backward_params,
)
from fe_api.dsa.dsa_reference import (
    ref_sparse_attention_forward,
    check_ref_dsa_sparse_attention_backward,
)


def _allocate(cfg, has_topk_length: bool):
    total_s_q = cfg["s_q"]
    total_s_kv = cfg["s_kv"]
    h = cfg["h_q"] if cfg.get("h_q") else 64
    d = cfg["head_dim"]
    topk = cfg["topk"]
    device = "cuda"

    q = torch.randn(total_s_q, h, d, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total_s_kv, d, dtype=torch.bfloat16, device=device)
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)

    topk_k = min(topk, total_s_kv)
    topk_idxs = torch.stack([torch.randperm(total_s_kv, device=device)[:topk_k] for _ in range(total_s_q)]).to(torch.int32)
    if topk_k < topk:
        pad = torch.full((total_s_q, topk - topk_k), -1, dtype=torch.int32, device=device)
        topk_idxs = torch.cat([topk_idxs, pad], dim=-1)

    topk_length = None
    if has_topk_length:
        topk_length = torch.randint(1, topk_k + 1, (total_s_q,), dtype=torch.int32, device=device)

    return q, kv, attn_sink, topk_idxs, topk_length


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_sparse_attention_backward_params
def test_DSA_sparse_attention_backward_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    head_dim_v,
    num_heads,
    topk,
    has_topk_length,
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
        head_dim_v=head_dim_v,
        num_heads=num_heads,
        topk=topk,
        has_topk_length=has_topk_length,
        min_compute_capability=90,
        s_q_default=1024,
        s_kv_default=4096,
    )
    cfg["h_q"] = num_heads

    q, kv, attn_sink, topk_idxs, topk_length = _allocate(cfg, has_topk_length)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Run reference forward to get out + FlashMLA-style KV-only lse for backward.
    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)

    try:
        result = DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            stream=stream,
        )
    except (ValueError, NotImplementedError, RuntimeError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]

    if not cfg["skip_ref"]:
        # The DKV accumulation is known to have precision limitations
        # vs. the FP32 autograd reference; use generous tolerances.
        check_ref_dsa_sparse_attention_backward(
            q,
            kv,
            attn_sink,
            topk_idxs,
            out,
            dout,
            lse,
            dq,
            dkv,
            d_sink,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=385)
def test_DSA_sparse_attention_backward_qh32_uses_per_query_topk_without_padding(monkeypatch):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm90
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 2, 128, 32
    head_dim, head_dim_v, topk = 576, 512, 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    # The old packed-M mapping used row 0 for both queries in one CTA. Make
    # the two rows disjoint so sharing a top-k row is unambiguously incorrect.
    topk_idxs = torch.stack(
        (
            torch.arange(0, topk, dtype=torch.int32, device=device),
            torch.arange(topk, 2 * topk, dtype=torch.int32, device=device),
        )
    )

    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
    )
    # Match the head-major backing used by the original #385 reproduction.
    out = out.transpose(0, 1).contiguous().transpose(0, 1)
    dout = torch.randn(num_heads, s_q, head_dim_v, dtype=torch.bfloat16, device=device).transpose(0, 1)
    assert out.stride() == dout.stride() == (head_dim_v, s_q * head_dim_v, 1)

    def fail_on_pad(*args, **kwargs):
        pytest.fail("qh32 must be handled by the SM90 main kernel without torch padding")

    monkeypatch.setattr(torch.nn.functional, "pad", fail_on_pad)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        stream=stream,
    )
    torch.cuda.synchronize()

    # qhpkv controls the query-head tiling and must remain part of the main
    # compile key, independently of the runtime tensor shape.
    assert any(key[1:4] == (head_dim, head_dim_v, num_heads) for key in _interface_sm90.flash_attn_bwd_sm90.compile_cache)

    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
    assert torch.isfinite(dq).all()
    assert torch.isfinite(dkv).all()
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        dq,
        dkv,
        d_sink,
        softmax_scale=softmax_scale,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_sparse_attention_backward_staged_store():
    """Regression test for the staged P/dS store paths of the SM100 kernel.

    The compute->MMA smem handoff for P and dS must stay correct when
    compute_mma_P_stage / compute_mma_dS_stage are raised above 1: the
    store-side layouts and the per-iteration store slot have to follow the
    producer pipeline state. This test deepens both pipelines to 2 and
    requires dq to match the stage-1 baseline bitwise (dq is written by TMA
    with a single writer per element, hence deterministic; dkv/d_sink are
    fp32 atomic reductions, so they are only gated on NaN and relative
    parity).
    """
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import (
            api as _dsa_bwd_api,
            _interface_sm100 as _dsa_bwd_iface,
            dsa_bwd_sm100 as _dsa_bwd_kmod,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("staged-store regression test targets the SM100 kernel")

    s_q = s_kv = 512
    h, d, topk = 64, 512, 256  # topk/64 = 4 tiles -> the pipelines really cycle
    device = "cuda"
    q = torch.randn(s_q, h, d, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, d, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)
    softmax_scale = 1.0 / math.sqrt(d)

    out, lse = ref_sparse_attention_forward(q, kv, attn_sink, topk_idxs, topk_length=topk_length, softmax_scale=softmax_scale)
    dout = torch.randn_like(out)

    orig_setup = _dsa_bwd_kmod.FlashAttentionDSABackwardSm100._setup_attributes

    def run(stage_overrides):
        def patched(self):
            orig_setup(self)
            for name, value in stage_overrides.items():
                setattr(self, name, value)

        _dsa_bwd_kmod.FlashAttentionDSABackwardSm100._setup_attributes = patched
        _dsa_bwd_iface.flash_attn_bwd_sm100.compile_cache.clear()
        _dsa_bwd_api._cache_of_SparseAttentionBackwardObjects.clear()
        try:
            result = DSA.sparse_attention_backward_wrapper(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                softmax_scale=softmax_scale,
                topk_length=topk_length,
            )
            torch.cuda.synchronize()
            return result["dq"], result["dkv"], result["d_sink"]
        finally:
            _dsa_bwd_kmod.FlashAttentionDSABackwardSm100._setup_attributes = orig_setup
            _dsa_bwd_iface.flash_attn_bwd_sm100.compile_cache.clear()
            _dsa_bwd_api._cache_of_SparseAttentionBackwardObjects.clear()

    dq_ref, dkv_ref, d_sink_ref = run({})
    assert not torch.isnan(dq_ref).any(), "stage-1 baseline produced NaN"

    dq, dkv, d_sink = run({"compute_mma_P_stage": 2, "compute_mma_dS_stage": 2})

    assert not torch.isnan(dq).any() and not torch.isnan(dkv).any() and not torch.isnan(d_sink).any(), "staged store paths produced NaN gradients"
    assert torch.equal(dq, dq_ref), "dq must be bitwise-identical between stage 1 and stage 2"

    def rel_l2(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()

    assert rel_l2(dkv, dkv_ref) < 1e-4, "dkv parity vs stage-1 baseline"
    assert rel_l2(d_sink, d_sink_ref) < 1e-4, "d_sink parity vs stage-1 baseline"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_sparse_attention_backward_topk_length_zero_raises():
    """Rows with topk_length == 0 must be rejected loudly on SM100.

    A zero entry gives the kernel zero KV tiles for that token
    (tile_count == 0 in dsa_bwd_sm100), which has no defined behavior:
    measured at s_q=128, s_kv=4096, head_dim=512, topk=512 on B200, a call
    whose topk_length contains zeros hangs until killed. The interface
    validates the precondition (outside CUDA graph capture) and raises
    before launch; this test exercises that guard path.

    L0 is intentional: the guard raises before the interface consults its
    compile cache, and ``SparseAttentionBackward.compile()`` defers kernel
    compilation to ``execute()`` (see api.py), so this test pays no CuTe
    compilation cost even on a cold cache.
    """
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("the topk_length >= 1 guard is implemented on the SM100 interface")

    s_q, s_kv, h, d, d_v, topk = 8, 256, 64, 576, 512, 64
    device = "cuda"
    q = torch.randn(s_q, h, d, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, d, dtype=torch.bfloat16, device=device)
    out = torch.randn(s_q, h, d_v, dtype=torch.bfloat16, device=device)
    dout = torch.randn_like(out)
    lse = torch.randn(s_q, h, dtype=torch.float32, device=device)
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)

    for bad_value in (0, -3):
        topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)
        topk_length[3] = bad_value
        with pytest.raises(ValueError, match="topk_length"):
            DSA.sparse_attention_backward_wrapper(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                softmax_scale=1.0 / math.sqrt(d),
                topk_length=topk_length,
            )


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_DSA_sparse_attention_backward_topk_length_guard_explicit_stream():
    """The topk_length guard must bind to the resolved launch stream.

    ``sparse_attention_backward_wrapper(..., stream=...)`` forwards an
    explicit launch stream to the SM100 interface; the guard's capture
    detection and its device-to-host sync must run on that stream, not on
    whatever torch stream happens to be current at call time:

    - outside capture, a zero row is rejected even when the launch targets
      an explicit, non-current stream (the sync is ordered on that stream);
    - while the explicit stream IS capturing and the torch-current stream
      is a different, non-capturing stream, the guard must detect the
      capture on the resolved stream and skip validation -- the documented
      capture semantics (the caller upholds the precondition) -- instead of
      syncing mid-capture. The captured graph is never replayed (a replay
      would launch the kernel with a zero-length row).
    """
    try:
        import cuda.bindings.driver as cuda_driver
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("the topk_length >= 1 guard is implemented on the SM100 interface")

    s_q, s_kv, h, d, d_v, topk = 8, 256, 64, 576, 512, 64
    device = "cuda"
    q = torch.randn(s_q, h, d, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, d, dtype=torch.bfloat16, device=device)
    out = torch.randn(s_q, h, d_v, dtype=torch.bfloat16, device=device)
    dout = torch.randn_like(out)
    lse = torch.randn(s_q, h, dtype=torch.float32, device=device)
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)

    side_stream = torch.cuda.Stream()

    def call(topk_length):
        return DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=1.0 / math.sqrt(d),
            topk_length=topk_length,
            stream=cuda_driver.CUstream(side_stream.cuda_stream),
        )

    bad_topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)
    bad_topk_length[3] = 0

    # Outside capture, an explicit non-current launch stream still rejects
    # zero rows (raises before the interface's compile cache is consulted).
    with pytest.raises(ValueError, match="topk_length"):
        call(bad_topk_length)

    # Warm the compile cache off the capture path: capture must not compile.
    good_topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)
    call(good_topk_length)
    torch.cuda.synchronize()

    # Capture on the explicit stream while the torch-current stream is a
    # different, non-capturing stream. The guard must see the capture on the
    # resolved stream and skip its sync; the pre-fix guard consulted the
    # torch-current stream instead, ran the validation mid-capture, and
    # raised here. Relaxed capture mode keeps the interface's unrelated
    # eager work (output/workspace allocation) legal during capture.
    other_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side_stream, capture_error_mode="relaxed"):
        with torch.cuda.stream(other_stream):
            assert not torch.cuda.is_current_stream_capturing(), "the guard must not rely on the torch-current stream"
            call(bad_topk_length)
    del graph
