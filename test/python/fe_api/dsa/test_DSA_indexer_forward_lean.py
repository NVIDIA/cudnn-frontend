import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_reference import check_ref_indexer_forward, ref_indexer_forward

INT32_MAX = 2**31 - 1


def _import_lean():
    try:
        from cudnn.deepseek_sparse_attention.indexer_forward import api_lean

        return api_lean
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


def _require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 GPU required")


def _lean_min_s_q(api_lean) -> int:
    """Smallest S_q that saturates the lean static persistent grid."""
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return api_lean.LEAN_MIN_WAVES * sm_count * api_lean.LEAN_TILE_TOKENS


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
    api_lean, b=1, s_q=None, s_k=None, h_q=64, h_kv=1, d=128, q_dtype=torch.bfloat16, w_dtype=torch.bfloat16, o_dtype=torch.float32, s_k_out=None
):
    if s_q is None:
        s_q = _lean_min_s_q(api_lean)
    if s_k is None:
        s_k = max(s_q // 2, 1)
    if s_k_out is None:
        s_k_out = s_k
    q = _meta((b, s_q, h_q, d), q_dtype)
    k = _meta((b, s_k, h_kv, d), q_dtype)
    w = _meta((b, s_q, h_q), w_dtype)
    o = _meta((b, s_q, s_k_out), o_dtype)
    return q, k, w, o


def _alloc_inputs(b, s_q, s_k, h_q=64, h_kv=1, d=128, w_dtype=torch.bfloat16):
    q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, s_k, h_kv, d, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device="cuda").to(w_dtype)
    return q, k, w


def _spy_lean_execute(monkeypatch, api_lean):
    """Count IndexerForwardLean.execute invocations (per-batch launches count once)."""
    calls = []
    original = api_lean.IndexerForwardLean.execute

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(api_lean.IndexerForwardLean, "execute", spy)
    return calls


# ---------------------------------------------------------------------------
# check_support: pass / soft-fail / hard-fail
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_DSA_indexer_forward_lean_check_support_pass():
    api_lean = _import_lean()
    _require_sm100()
    for kwargs in (
        {},
        {"b": 3},
        {"w_dtype": torch.float32},
        {"s_k": 999},  # S_k need not be a multiple of the 128-column KV tile
    ):
        q, k, w, o = _meta_samples(api_lean, **kwargs)
        api = api_lean.IndexerForwardLean(q, k, w, o, ratio=2)
        assert api.check_support() is True, f"expected supported for {kwargs}"


@pytest.mark.L0
@pytest.mark.parametrize(
    "case",
    [
        "h32",  # qhead_per_kv_head == 32 stays on the legacy kernel
        "hkv2",  # h_kv != 1
        "d64",  # head_dim != 128
        "q_fp16",
        "w_fp64",
        "o_bf16",
        "sq_nonmult4",  # S_q not a multiple of the 4-token tile
        "grid_small",  # grid too small for the static single-wave schedule
        "skout_mismatch",  # Out column dim must equal S_k
    ],
)
def test_DSA_indexer_forward_lean_check_support_false(case):
    api_lean = _import_lean()
    _require_sm100()
    min_s_q = _lean_min_s_q(api_lean)
    kwargs = {
        "h32": {"h_q": 32},
        "hkv2": {"h_q": 128, "h_kv": 2},
        "d64": {"d": 64},
        "q_fp16": {"q_dtype": torch.float16},
        "w_fp64": {"w_dtype": torch.float64},
        "o_bf16": {"o_dtype": torch.bfloat16},
        # derived from the device's SM count (not hardcoded): a saturating
        # tile count made non-multiple-of-4
        "sq_nonmult4": {"s_q": min_s_q * 8 + 2},
        "grid_small": {"s_q": 8},
        "skout_mismatch": {"s_k": 128, "s_k_out": 132},
    }[case]
    q, k, w, o = _meta_samples(api_lean, **kwargs)
    api = api_lean.IndexerForwardLean(q, k, w, o, ratio=2)
    assert api.check_support() is False


@pytest.mark.L0
def test_DSA_indexer_forward_lean_check_support_noncontiguous_false():
    api_lean = _import_lean()
    _require_sm100()
    q, k, w, o = _meta_samples(api_lean)
    s = q.shape
    q_pad = _meta(s, torch.bfloat16, stride=(s[1] * s[2] * (s[3] + 8), s[2] * (s[3] + 8), s[3] + 8, 1))
    api = api_lean.IndexerForwardLean(q_pad, k, w, o, ratio=2)
    assert api.check_support() is False


@pytest.mark.L0
def test_DSA_indexer_forward_lean_check_support_raises():
    api_lean = _import_lean()
    _require_sm100()
    q, k, w, o = _meta_samples(api_lean)
    with pytest.raises(ValueError):
        api_lean.IndexerForwardLean(q[0], k, w, o, ratio=2).check_support()  # Q not 4-D
    k2 = _meta((2, k.shape[1], 1, 128), torch.bfloat16)
    with pytest.raises(ValueError):
        api_lean.IndexerForwardLean(q, k2, w, o, ratio=2).check_support()  # batch mismatch
    w2 = _meta((1, q.shape[1], 32), torch.bfloat16)
    with pytest.raises(ValueError):
        api_lean.IndexerForwardLean(q, k, w2, o, ratio=2).check_support()  # W shape mismatch


# ---------------------------------------------------------------------------
# numerics: lean wrapper vs the pure-torch reference
# ---------------------------------------------------------------------------


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("batch,with_offsets", [(1, False), (2, True)])
def test_DSA_indexer_forward_lean_wrapper_numerics(batch, with_offsets):
    api_lean = _import_lean()
    _require_sm100()
    ratio = 4
    s_q = _lean_min_s_q(api_lean)
    s_k = s_q // 2
    q, k, w = _alloc_inputs(batch, s_q, s_k)
    q_causal_offsets = None
    if with_offsets:
        q_causal_offsets = torch.arange(3, 3 + 253 * batch, 253, dtype=torch.int32, device="cuda")

    result = api_lean.indexer_forward_lean_wrapper(q, k, w, ratio=ratio, q_causal_offsets=q_causal_offsets)
    scores = result["scores"]
    assert scores.shape == (batch, s_q, s_k)
    assert scores.is_contiguous()
    check_ref_indexer_forward(q, k, w, scores, ratio, q_causal_offsets=q_causal_offsets)


@pytest.mark.L0
@torch_fork_set_rng(seed=1)
def test_DSA_indexer_forward_lean_wrapper_sm_scale_and_fp32_w():
    api_lean = _import_lean()
    _require_sm100()
    ratio, sm_scale = 4, 0.125
    s_q = _lean_min_s_q(api_lean)
    s_k = s_q // 2
    q, k, w = _alloc_inputs(1, s_q, s_k, w_dtype=torch.float32)

    scores = api_lean.indexer_forward_lean_wrapper(q, k, w, ratio=ratio, sm_scale=sm_scale)["scores"]
    ref = ref_indexer_forward(q, k, w, ratio)
    finite = torch.isfinite(ref)
    assert torch.equal(torch.isneginf(scores), torch.isneginf(ref))
    torch.testing.assert_close(scores[finite], ref[finite] * sm_scale, atol=1e-4, rtol=1e-4)


@pytest.mark.L0
@torch_fork_set_rng(seed=2)
@pytest.mark.parametrize(
    "s_k,ratio,offset_mode",
    [
        (1, 1, None),  # single-column K: one 1-row partial KV tile
        (127, 1, None),  # partial tile just below the 128-row boundary
        (129, 1, None),  # one full tile + a 1-row partial tail
        (999, 1, None),  # heavy clamping + partial tail, visibility sweeps it
        (999, 4, "full"),  # offsets push every row's window to the full S_k
    ],
)
def test_DSA_indexer_forward_lean_wrapper_ragged_s_k_tail(s_k, ratio, offset_mode):
    """Ragged S_k where the visibility window actually REACHES the physical
    tail columns of the trailing partial KV tile (ratio=1 rows sweep ke
    across the 128-column tile boundary and up to S_k; the offsets variant
    drives every row's window to S_k), exercising the TMA zero-fill and the
    bounds-guarded fp32 tail stores."""
    api_lean = _import_lean()
    _require_sm100()
    min_s_q = _lean_min_s_q(api_lean)
    # ratio=1 windows reach column S_k-1 once i+1 >= S_k, so make S_q >= S_k
    s_q = max(min_s_q, (s_k + 4) // 4 * 4)
    q, k, w = _alloc_inputs(1, s_q, s_k)
    q_causal_offsets = None
    if offset_mode == "full":
        q_causal_offsets = torch.full((1,), ratio * s_k, dtype=torch.int32, device="cuda")

    scores = api_lean.indexer_forward_lean_wrapper(q, k, w, ratio=ratio, q_causal_offsets=q_causal_offsets)["scores"]
    assert scores.shape == (1, s_q, s_k)
    # prove the tail was reached: the last row's window covers column S_k-1
    assert torch.isfinite(scores[0, -1, s_k - 1]), "tail column never became visible — test would be vacuous"
    if offset_mode == "full":
        assert torch.isfinite(scores).all(), "full-visibility offsets must produce finite scores everywhere"
    check_ref_indexer_forward(q, k, w, scores, ratio, q_causal_offsets=q_causal_offsets)


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
def test_DSA_indexer_forward_lean_wrapper_all_masked_windows():
    """Offsets so negative that every per-row window is empty: the kernel
    sweeps nothing and the -inf pre-fill must survive untouched."""
    api_lean = _import_lean()
    _require_sm100()
    ratio = 4
    s_q = _lean_min_s_q(api_lean)
    s_k = s_q // 2
    q, k, w = _alloc_inputs(1, s_q, s_k)
    q_causal_offsets = torch.full((1,), -(s_q + ratio), dtype=torch.int32, device="cuda")

    scores = api_lean.indexer_forward_lean_wrapper(q, k, w, ratio=ratio, q_causal_offsets=q_causal_offsets)["scores"]
    assert torch.isneginf(scores).all()
    ref = ref_indexer_forward(q, k, w, ratio, q_causal_offsets=q_causal_offsets)
    assert torch.equal(torch.isneginf(scores), torch.isneginf(ref))


@pytest.mark.L0
@torch_fork_set_rng(seed=8)
def test_DSA_indexer_forward_lean_wrapper_w_dtype_bitwise_and_determinism():
    """BF16 W is up-converted in-kernel exactly, so BF16-W and FP32-W runs
    must be bitwise identical; the fixed-order epilogue reduction must be
    deterministic run-to-run."""
    api_lean = _import_lean()
    _require_sm100()
    ratio = 2
    s_q = _lean_min_s_q(api_lean)
    s_k = s_q // 2
    q, k, w = _alloc_inputs(1, s_q, s_k)

    scores_bf16w = api_lean.indexer_forward_lean_wrapper(q, k, w, ratio=ratio)["scores"]
    scores_fp32w = api_lean.indexer_forward_lean_wrapper(q, k, w.float().contiguous(), ratio=ratio)["scores"]
    assert torch.equal(scores_bf16w, scores_fp32w), "BF16-W ingest must be bitwise identical to FP32-W"

    scores_again = api_lean.indexer_forward_lean_wrapper(q, k, w, ratio=ratio)["scores"]
    assert torch.equal(scores_bf16w, scores_again), "lean kernel must be deterministic run-to-run"


# ---------------------------------------------------------------------------
# transparent dispatch through indexer_forward_wrapper
# ---------------------------------------------------------------------------


@pytest.mark.L0
@torch_fork_set_rng(seed=3)
def test_DSA_indexer_forward_dispatch_routes_lean(monkeypatch):
    api_lean = _import_lean()
    from cudnn import DSA

    _require_sm100()
    monkeypatch.delenv("CUDNNFE_DSA_INDEXER_FWD_DISABLE_LEAN", raising=False)
    ratio, sm_scale = 4, 0.25
    s_q = _lean_min_s_q(api_lean)
    s_k = s_q // 2
    q, k, w = _alloc_inputs(1, s_q, s_k)
    q_causal_offsets = torch.tensor([7], dtype=torch.int32, device="cuda")

    calls = _spy_lean_execute(monkeypatch, api_lean)
    dispatched = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, sm_scale=sm_scale, q_causal_offsets=q_causal_offsets)["scores"]
    assert len(calls) == 1, "lean fast path was not dispatched"

    # the family wrapper must return exactly what the lean wrapper returns
    # (ratio / sm_scale / q_causal_offsets plumbed through unchanged)
    direct = api_lean.indexer_forward_lean_wrapper(q, k, w, ratio=ratio, sm_scale=sm_scale, q_causal_offsets=q_causal_offsets)["scores"]
    assert torch.equal(dispatched, direct)

    ref = ref_indexer_forward(q, k, w, ratio, q_causal_offsets=q_causal_offsets)
    finite = torch.isfinite(ref)
    assert torch.equal(torch.isneginf(dispatched), torch.isneginf(ref))
    torch.testing.assert_close(dispatched[finite], ref[finite] * sm_scale, atol=1e-4, rtol=1e-4)


@pytest.mark.L0
@torch_fork_set_rng(seed=9)
def test_DSA_indexer_forward_dispatch_shares_compile_across_batch(monkeypatch):
    """The lean kernel runs on flattened per-batch views, so one compiled
    instance must serve every B of the same (S_q, S_k): the B=3 call after a
    B=1 call must trigger zero additional cute.compile invocations AND be
    numerically correct (empirical batch-independence of the codegen key)."""
    api_lean = _import_lean()
    from cudnn import DSA

    _require_sm100()
    monkeypatch.delenv("CUDNNFE_DSA_INDEXER_FWD_DISABLE_LEAN", raising=False)
    ratio = 4
    s_q = _lean_min_s_q(api_lean)
    s_k = s_q // 2 + 16  # unique S_k so this test owns its compile-cache entry

    compile_calls = []
    original_compile = api_lean.cute.compile

    def compile_spy(*args, **kwargs):
        compile_calls.append(1)
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(api_lean.cute, "compile", compile_spy)
    calls = _spy_lean_execute(monkeypatch, api_lean)

    q1, k1, w1 = _alloc_inputs(1, s_q, s_k)
    scores1 = DSA.indexer_forward_wrapper(q1, k1, w1, ratio=ratio)["scores"]
    assert len(calls) == 1, "lean fast path was not dispatched for B=1"
    compiles_after_b1 = len(compile_calls)
    assert compiles_after_b1 >= 1, "expected the fresh (S_q, S_k) to JIT once"
    check_ref_indexer_forward(q1, k1, w1, scores1, ratio)

    q3, k3, w3 = _alloc_inputs(3, s_q, s_k)
    scores3 = DSA.indexer_forward_wrapper(q3, k3, w3, ratio=ratio)["scores"]
    assert len(calls) == 2, "lean fast path was not dispatched for B=3"
    assert len(compile_calls) == compiles_after_b1, "B must not appear in the codegen key — no recompile for a new batch size"
    check_ref_indexer_forward(q3, k3, w3, scores3, ratio)


@pytest.mark.L0
@torch_fork_set_rng(seed=10)
def test_DSA_indexer_forward_dispatch_int32_offset_bound(monkeypatch):
    """Legacy evaluates (offset + i + 1) in int32 on device; the dispatcher
    must keep offsets that would overflow that arithmetic on the legacy path
    (offset + S_q + 1 > INT32_MAX) and may dispatch anything below the bound."""
    api_lean = _import_lean()
    from cudnn import DSA

    _require_sm100()
    monkeypatch.delenv("CUDNNFE_DSA_INDEXER_FWD_DISABLE_LEAN", raising=False)
    ratio = 4
    s_q = _lean_min_s_q(api_lean)
    s_k = s_q // 2
    q, k, w = _alloc_inputs(1, s_q, s_k)

    calls = _spy_lean_execute(monkeypatch, api_lean)

    # largest accepted offset: offset + S_q + 1 == INT32_MAX -> lean
    off_ok = torch.full((1,), INT32_MAX - s_q - 1, dtype=torch.int32, device="cuda")
    scores_ok = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, q_causal_offsets=off_ok)["scores"]
    assert len(calls) == 1, "boundary-accepted offsets must still dispatch lean"
    check_ref_indexer_forward(q, k, w, scores_ok, ratio, q_causal_offsets=off_ok)

    # one past the bound: legacy int32 window math could overflow -> legacy
    # (legacy itself still stays exactly at INT32_MAX here, so its output
    # remains reference-correct; the conservative bound is on the dispatch)
    off_over = torch.full((1,), INT32_MAX - s_q, dtype=torch.int32, device="cuda")
    scores_over = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, q_causal_offsets=off_over)["scores"]
    assert len(calls) == 1, "offsets past the int32 bound must stay on the legacy path"
    check_ref_indexer_forward(q, k, w, scores_over, ratio, q_causal_offsets=off_over)


@pytest.mark.L0
@torch_fork_set_rng(seed=4)
def test_DSA_indexer_forward_dispatch_env_disable(monkeypatch):
    api_lean = _import_lean()
    from cudnn import DSA

    _require_sm100()
    ratio = 4
    s_q = _lean_min_s_q(api_lean)
    s_k = s_q // 2
    q, k, w = _alloc_inputs(1, s_q, s_k)
    q_causal_offsets = torch.tensor([4], dtype=torch.int32, device="cuda")

    monkeypatch.setenv("CUDNNFE_DSA_INDEXER_FWD_DISABLE_LEAN", "1")
    calls = _spy_lean_execute(monkeypatch, api_lean)
    scores = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, q_causal_offsets=q_causal_offsets)["scores"]
    assert len(calls) == 0, "lean fast path must stay off when disabled by env"
    check_ref_indexer_forward(q, k, w, scores, ratio, q_causal_offsets=q_causal_offsets)


@pytest.mark.L0
@torch_fork_set_rng(seed=5)
def test_DSA_indexer_forward_dispatch_keeps_legacy_for_h32(monkeypatch):
    api_lean = _import_lean()
    from cudnn import DSA

    _require_sm100()
    ratio = 4
    # S_q saturates the lean grid, so qhead_per_kv_head=32 is the ONLY
    # reason this config is rejected (isolates the H gate)
    b, s_q = 1, _lean_min_s_q(api_lean)
    s_k = s_q // 2
    q, k, w = _alloc_inputs(b, s_q, s_k, h_q=32)
    q_causal_offsets = torch.full((b,), 4, dtype=torch.int32, device="cuda")

    calls = _spy_lean_execute(monkeypatch, api_lean)
    scores = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, qhead_per_kv_head=32, q_causal_offsets=q_causal_offsets)["scores"]
    assert len(calls) == 0, "qhead_per_kv_head=32 must keep the legacy path"
    check_ref_indexer_forward(q, k, w, scores, ratio, q_causal_offsets=q_causal_offsets)


@pytest.mark.L0
@torch_fork_set_rng(seed=11)
def test_DSA_indexer_forward_dispatch_keeps_legacy_for_non_default_knobs(monkeypatch):
    api_lean = _import_lean()
    from cudnn import DSA

    _require_sm100()
    ratio = 4
    # lean-eligible shape, but a non-default tuning knob must force legacy
    s_q = _lean_min_s_q(api_lean)
    s_k = s_q // 2
    q, k, w = _alloc_inputs(1, s_q, s_k)

    calls = _spy_lean_execute(monkeypatch, api_lean)
    scores = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, kv_stage=2)["scores"]
    assert len(calls) == 0, "non-default tuning knobs must keep the legacy path"
    check_ref_indexer_forward(q, k, w, scores, ratio)


@pytest.mark.L0
@torch_fork_set_rng(seed=6)
def test_DSA_indexer_forward_dispatch_keeps_legacy_for_thd(monkeypatch):
    api_lean = _import_lean()
    from cudnn import DSA

    _require_sm100()
    device = torch.device("cuda")
    shapes = [(8, 64), (12, 72)]
    ratio, h_q, h_kv, d = 4, 64, 1, 128
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    cu_seqlens_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total_q, total_k = int(cu_seqlens_q[-1]), int(cu_seqlens_k[-1])
    q = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device)

    calls = _spy_lean_execute(monkeypatch, api_lean)
    scores = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=h_q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
    )["scores"]
    torch.cuda.synchronize()
    assert len(calls) == 0, "THD/varlen must keep the legacy path"

    cu_q_host = cu_seqlens_q.tolist()
    cu_k_host = cu_seqlens_k.tolist()
    for batch, (_, s_k) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        check_ref_indexer_forward(
            q[q0:q1].unsqueeze(0),
            k[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            scores[q0:q1, :s_k].unsqueeze(0),
            ratio,
        )


# ---------------------------------------------------------------------------
# THD / varlen (ragged packed) lean fast path — GLOBAL compressed-KV columns
# ---------------------------------------------------------------------------


def _cu(seg_lens, device="cuda"):
    """(len+1,) int32 cu_seqlens from per-segment lengths."""
    cs = torch.tensor(seg_lens, dtype=torch.int64).cumsum(0).tolist()
    return torch.tensor([0, *cs], dtype=torch.int32, device=device)


def _alloc_thd(seg_q, seg_k, h_q=64, h_kv=1, d=128, w_dtype=torch.bfloat16):
    """Packed THD inputs for unequal segments plus their cu_seqlens."""
    t_q = int(sum(seg_q))
    m_total = int(sum(seg_k))
    q = torch.randn(t_q, h_q, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(m_total, h_kv, d, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(t_q, h_q, dtype=torch.bfloat16, device="cuda").to(w_dtype)
    return q, k, w, _cu(seg_q), _cu(seg_k)


def _meta_samples_thd(seg_q, seg_k, h_q=64, h_kv=1, d=128, q_dtype=torch.bfloat16, w_dtype=torch.bfloat16, o_dtype=torch.float32):
    t_q = int(sum(seg_q))
    m_total = int(sum(seg_k))
    q = _meta((t_q, h_q, d), q_dtype)
    k = _meta((m_total, h_kv, d), q_dtype)
    w = _meta((t_q, h_q), w_dtype)
    o = _meta((t_q, m_total), o_dtype)
    return q, k, w, o, _cu(seg_q), _cu(seg_k)


def _thd_segments_saturating(api_lean, n_seg=3):
    """Unequal q-segments whose T_q saturates the lean grid (each %4)."""
    base = _lean_min_s_q(api_lean)  # already a multiple of LEAN_TILE_TOKENS
    seg_q = [base + 4 * i for i in range(n_seg)]  # unequal, each %4, sums > base
    seg_k = [max(s // 2, 4) for s in seg_q]  # ratio-2-like geometry, unequal
    return seg_q, seg_k


@pytest.mark.L0
def test_DSA_indexer_forward_lean_thd_check_support_pass():
    api_lean = _import_lean()
    _require_sm100()
    seg_q, seg_k = _thd_segments_saturating(api_lean)
    for w_dtype in (torch.bfloat16, torch.float32):
        q, k, w, o, cu_q, cu_k = _meta_samples_thd(seg_q, seg_k, w_dtype=w_dtype)
        api = api_lean.IndexerForwardLean(q, k, w, o, ratio=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k)
        assert api.check_support() is True, f"expected THD supported for w_dtype={w_dtype}"
        assert api._thd is True and api.s_q == int(sum(seg_q)) and api.s_k == int(sum(seg_k))


@pytest.mark.L0
@pytest.mark.parametrize("case", ["h32", "d64", "q_fp16", "grid_small", "tq_nonmult4"])
def test_DSA_indexer_forward_lean_thd_check_support_false(case):
    api_lean = _import_lean()
    _require_sm100()
    seg_q, seg_k = _thd_segments_saturating(api_lean)
    kwargs = {
        "h32": dict(h_q=32),
        "d64": dict(d=64),
        "q_fp16": dict(q_dtype=torch.float16),
        "grid_small": None,  # handled below
        "tq_nonmult4": None,
    }[case]
    if case == "grid_small":
        seg_q, seg_k = [8], [4]
        kwargs = {}
    elif case == "tq_nonmult4":
        seg_q = [_lean_min_s_q(api_lean) + 2]  # T_q not a multiple of 4
        seg_k = [64]
        kwargs = {}
    q, k, w, o, cu_q, cu_k = _meta_samples_thd(seg_q, seg_k, **kwargs)
    api = api_lean.IndexerForwardLean(q, k, w, o, ratio=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k)
    assert api.check_support() is False


@pytest.mark.L0
def test_DSA_indexer_forward_lean_thd_check_support_raises():
    api_lean = _import_lean()
    _require_sm100()
    seg_q, seg_k = _thd_segments_saturating(api_lean, n_seg=2)
    t_q, m_total = int(sum(seg_q)), int(sum(seg_k))

    # missing cu_seqlens_k
    q, k, w, o, cu_q, cu_k = _meta_samples_thd(seg_q, seg_k)
    with pytest.raises(ValueError):
        api_lean.IndexerForwardLean(q, k, w, o, ratio=2, cu_seqlens_q=cu_q).check_support()

    # cu_seqlens_k[-1] disagrees with packed K rows (m_total)
    q, k, w, o, cu_q, _ = _meta_samples_thd(seg_q, seg_k)
    bad_k = torch.tensor([0, seg_k[0], m_total + 8], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):
        api_lean.IndexerForwardLean(q, k, w, o, ratio=2, cu_seqlens_q=cu_q, cu_seqlens_k=bad_k).check_support()

    # non-monotonic / non-zero-start cu_seqlens_q
    q, k, w, o, _, cu_k = _meta_samples_thd(seg_q, seg_k)
    nonmono = torch.tensor([seg_q[0], 0, t_q], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):
        api_lean.IndexerForwardLean(q, k, w, o, ratio=2, cu_seqlens_q=nonmono, cu_seqlens_k=cu_k).check_support()


def _check_thd_scores(api_lean, q, k, w, cu_q, cu_k, scores, ratio, sm_scale=1.0):
    """Per-segment: global-column block matches the BSHD oracle; every
    column outside a row's own segment block is -inf (segment isolation)."""
    t_q, m_total = int(cu_q[-1]), int(cu_k[-1])
    assert tuple(scores.shape) == (t_q, m_total)
    assert scores.is_contiguous()
    cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()
    n_seg = len(cu_q_host) - 1
    for b in range(n_seg):
        q0, q1 = cu_q_host[b], cu_q_host[b + 1]
        k0, k1 = cu_k_host[b], cu_k_host[b + 1]
        if q1 <= q0:
            continue
        block = scores[q0:q1, k0:k1].unsqueeze(0)
        ref = ref_indexer_forward(q[q0:q1].unsqueeze(0), k[k0:k1].unsqueeze(0), w[q0:q1].unsqueeze(0), ratio)
        finite = torch.isfinite(ref)
        assert torch.equal(torch.isneginf(block), torch.isneginf(ref)), f"seg {b} mask mismatch"
        torch.testing.assert_close(block[finite], ref[finite] * sm_scale, atol=1e-4, rtol=1e-4)
        # segment isolation: everything OUTSIDE this segment's KV block, for
        # these rows, must be -inf (a query in segment b sees only seg b's KV)
        outside = scores[q0:q1].clone()
        outside[:, k0:k1] = float("-inf")
        assert torch.isneginf(outside).all(), f"seg {b} leaked finite scores into another segment's columns"


@pytest.mark.L0
@torch_fork_set_rng(seed=20)
@pytest.mark.parametrize("ratio", [2, 4])
def test_DSA_indexer_forward_lean_thd_numerics_ragged(ratio):
    """Ragged multi-segment THD vs the per-segment fp32 oracle + strict
    segment isolation, at the same 1e-4 tolerance the BSHD/legacy suite uses."""
    api_lean = _import_lean()
    _require_sm100()
    seg_q, seg_k = _thd_segments_saturating(api_lean, n_seg=3)
    q, k, w, cu_q, cu_k = _alloc_thd(seg_q, seg_k)
    scores = api_lean.indexer_forward_lean_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(seg_q),
        max_seqlen_k=max(seg_k),
    )["scores"]
    _check_thd_scores(api_lean, q, k, w, cu_q, cu_k, scores, ratio)


@pytest.mark.L0
@torch_fork_set_rng(seed=21)
def test_DSA_indexer_forward_lean_thd_sm_scale_and_fp32_w():
    api_lean = _import_lean()
    _require_sm100()
    ratio, sm_scale = 4, 0.125
    seg_q, seg_k = _thd_segments_saturating(api_lean, n_seg=2)
    q, k, w, cu_q, cu_k = _alloc_thd(seg_q, seg_k, w_dtype=torch.float32)
    scores = api_lean.indexer_forward_lean_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(seg_q),
        max_seqlen_k=max(seg_k),
    )["scores"]
    _check_thd_scores(api_lean, q, k, w, cu_q, cu_k, scores, ratio, sm_scale=sm_scale)


@pytest.mark.L0
@torch_fork_set_rng(seed=22)
def test_DSA_indexer_forward_lean_thd_single_segment_equals_bshd():
    """A single-segment THD problem (cu_seqlens = [0, T]) has ks == 0 windows
    identical to the B=1 BSHD path, so the shared kernel must produce a
    BITWISE-identical score matrix — proving THD reuses the exact schedule."""
    api_lean = _import_lean()
    _require_sm100()
    ratio = 4
    s_q = _lean_min_s_q(api_lean)
    s_k = s_q // 2
    q, k, w = _alloc_inputs(1, s_q, s_k)  # BSHD (1, s_q, H, D)

    bshd = api_lean.indexer_forward_lean_wrapper(q, k, w, ratio=ratio)["scores"][0]

    q_thd = q.view(s_q, 64, 128).contiguous()
    k_thd = k.view(s_k, 1, 128).contiguous()
    w_thd = w.view(s_q, 64).contiguous()
    cu_q = torch.tensor([0, s_q], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, s_k], dtype=torch.int32, device="cuda")
    thd = api_lean.indexer_forward_lean_wrapper(
        q_thd,
        k_thd,
        w_thd,
        ratio=ratio,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=s_q,
        max_seqlen_k=s_k,
    )["scores"]
    assert thd.shape == (s_q, s_k)
    assert torch.equal(thd, bshd), "single-segment THD must be bitwise identical to B=1 BSHD"


@pytest.mark.L0
@torch_fork_set_rng(seed=23)
def test_DSA_indexer_forward_lean_thd_q_causal_offsets_rejected():
    """q_causal_offsets is intentionally unsupported on the THD lean path."""
    api_lean = _import_lean()
    _require_sm100()
    seg_q, seg_k = _thd_segments_saturating(api_lean, n_seg=2)
    q, k, w, cu_q, cu_k = _alloc_thd(seg_q, seg_k)
    off = torch.zeros(2, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):
        api_lean.indexer_forward_lean_wrapper(
            q,
            k,
            w,
            ratio=4,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(seg_q),
            max_seqlen_k=max(seg_k),
            q_causal_offsets=off,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=24)
def test_DSA_indexer_forward_lean_thd_empty_and_tiny_segments():
    """Empty query segment + a single-token KV segment: windows collapse to
    empty ([-inf] rows) or 1-column, and the ragged K tail is exercised."""
    api_lean = _import_lean()
    _require_sm100()
    ratio = 2
    base = _lean_min_s_q(api_lean)
    seg_q = [base, 0, 8]  # middle segment has zero queries
    seg_k = [base // 2, 16, 1]
    q, k, w, cu_q, cu_k = _alloc_thd(seg_q, seg_k)
    scores = api_lean.indexer_forward_lean_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(seg_q),
        max_seqlen_k=max(seg_k),
    )["scores"]
    _check_thd_scores(api_lean, q, k, w, cu_q, cu_k, scores, ratio)
