# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the FROST SM100 DSL SDPA-forward engines against a torch reference."""

import math
import os

import pytest
import torch

from test_utils import torch_fork_set_rng

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import make_dense_stats, requires_pre_rubin_blackwell, requires_dsl, _dsl_installed


from frost_test_utils import select_engine as _select_engine  # noqa: F401

pytestmark = requires_pre_rubin_blackwell


def _ref_sdpa(q, k, v, *, is_causal, scale):
    """Reference SDPA in fp32 (dense, top-left causal or none)."""
    q_ref, k_ref, v_ref = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * scale
    if is_causal:
        s_q, s_kv = q.shape[2], k.shape[2]
        i = torch.arange(s_q, device=q.device).view(s_q, 1)
        j = torch.arange(s_kv, device=q.device).view(1, s_kv)
        keep = j <= (i + (s_kv - s_q))
        scores = scores.masked_fill(~keep, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_ref).to(q.dtype)


@pytest.mark.L0
@pytest.mark.parametrize("d", [512, 256, 128], ids=["dsv4_d512", "qwen_d256", "llama_d128"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=0)
def test_sdpa_fwd_dsl_sm100_graph_api(dtype, is_causal, d):
    try:
        import cudnn
        import cudnn.sdpa  # noqa: F401 — registers the FROST DSL engines
    except ImportError as e:
        pytest.skip(f"SM100 DSL engine not available: {e}")
    if not _dsl_installed():
        pytest.skip("cutlass/dsl not installed")

    b, h, s = 2, 8, 256
    device = "cuda"
    scale = 1.0 / math.sqrt(d)

    # BSHD-physical / BHSD-logical (transpose gives the strides the kernel expects).
    q_gpu = torch.randn(b, s, h, d, device=device, dtype=dtype).transpose(1, 2)
    k_gpu = torch.randn(b, s, h, d, device=device, dtype=dtype).transpose(1, 2)
    v_gpu = torch.randn(b, s, h, d, device=device, dtype=dtype).transpose(1, 2)
    o_gpu = torch.empty(b, s, h, d, device=device, dtype=dtype).transpose(1, 2)

    io_dtype = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu)
    v = graph.tensor_like(v_gpu)
    o, _ = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=False,
        attn_scale=scale,
        use_causal_mask=is_causal,
    )
    o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(graph, engine_name())
    graph.check_support()
    graph.build_plans()
    # Honest workspace: no Stats output, so the kernel compiles the LSE store
    # out (has_lse=False) — no dummy buffer exists at any level.
    assert graph.get_workspace_size() == 0

    workspace = torch.empty(max(graph.get_workspace_size(), 1), device=device, dtype=torch.uint8)
    graph.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu}, workspace)
    torch.cuda.synchronize()

    o_ref = _ref_sdpa(q_gpu, k_gpu, v_gpu, is_causal=is_causal, scale=scale)
    torch.testing.assert_close(o_gpu, o_ref, atol=5e-2, rtol=3e-2)


# Feature coverage — mask / sink / GQA. _ref_sdpa_full below encodes the kernel's
# exact mask + sink semantics (masks OR-ed; sink = one extra softmax column, V=0).
_FLAVORS = [512, 256, 128]
_FLAVOR_IDS = ["dsv4_d512", "qwen_d256", "llama_d128"]
_DTYPES = [torch.float16, torch.bfloat16]
_DTYPE_IDS = ["fp16", "bf16"]
# Exact in fp16/bf16/fp32: pre-fills O/Stats storages in the THD harness so
# no-op paths (t_q == 0) can assert the buffers came back untouched.
_THD_SENTINEL = 2048.0


def _ref_sdpa_full(
    q, k, v, *, scale, is_causal=False, bottom_right=False, band_right=0, swa_window=None, seq_q_lens=None, seq_kv_lens=None, sinks=None, return_stats=False
):
    """fp32 reference matching the SM100 DSL kernel's mask + sink semantics.
    q/k/v are BHSD; GQA (h_q > h_kv) is handled by expanding K/V. Bottom-right
    anchors the diagonal at the per-batch (seq_len_q[b], seq_len_kv[b]) corner
    — the global (S_q, S_kv) when no lengths are given. With ``return_stats``
    also returns the (B, H_q, S_q) LSE — logsumexp over the masked scores
    (the sink joins as one extra column; fully-masked rows are -inf without
    one; rows at/past seq_len_q[b] are -inf even with one)."""
    b, h_q, s_q, _ = q.shape
    _, h_kv, s_kv, _ = v.shape
    dev = q.device
    g = h_q // h_kv
    k_ref = k.repeat_interleave(g, dim=1).float()
    v_ref = v.repeat_interleave(g, dim=1).float()
    scores = torch.matmul(q.float(), k_ref.transpose(-1, -2)) * scale

    q_lens = seq_q_lens.flatten().to(device=dev, dtype=torch.int64) if seq_q_lens is not None else torch.full((b,), s_q, dtype=torch.int64, device=dev)
    kv_lens = seq_kv_lens.flatten().to(device=dev, dtype=torch.int64) if seq_kv_lens is not None else torch.full((b,), s_kv, dtype=torch.int64, device=dev)
    i = torch.arange(s_q, device=dev).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=dev).view(1, 1, 1, s_kv)
    diag = i + (kv_lens - q_lens).view(b, 1, 1, 1) if bottom_right else i
    dead_q = i >= q_lens.view(b, 1, 1, 1)
    masked = dead_q | (j >= kv_lens.view(b, 1, 1, 1))
    if is_causal:
        masked = masked | (j > diag + band_right)
    if swa_window is not None:
        masked = masked | (j < diag - swa_window)
    scores = scores.masked_fill(masked, float("-inf"))

    if sinks is not None:
        sink_col = sinks.view(1, h_q, 1, 1).float().expand(b, h_q, s_q, 1).to(dev)
        full_scores = torch.cat([scores, sink_col], dim=-1)
        probs = torch.softmax(full_scores, dim=-1)
        o = torch.matmul(probs[..., :s_kv], v_ref)
    else:
        full_scores = scores
        o = torch.matmul(torch.softmax(scores, dim=-1).nan_to_num(0.0), v_ref)
    if not return_stats:
        return o.to(q.dtype)
    lse = torch.logsumexp(full_scores, dim=-1)  # fully-masked rows -> -inf (sink-less)
    # Rows at/past seq_len_q[b] trim to -inf even with a sink.
    lse = lse.masked_fill(dead_q.squeeze(-1), float("-inf"))
    return o.to(q.dtype), lse


def _require_dsl():
    try:
        import cudnn  # noqa: F401
        import cudnn.sdpa  # noqa: F401 — registers the FROST DSL engines
    except ImportError as e:
        pytest.skip(f"SM100 DSL engine not available: {e}")
    if not _dsl_installed():
        pytest.skip("cutlass/dsl not installed")


def _bhsd(b, h, s, d, dtype, device="cuda"):
    """Random BHSD tensor with BSHD-physical strides (notebook convention)."""
    return torch.randn(b, s, h, d, device=device, dtype=dtype).transpose(1, 2)


def _run_dsl_graph(
    q_gpu,
    k_gpu,
    v_gpu,
    *,
    scale,
    dtype,
    sdpa_kwargs,
    seq_len_kv=None,
    seq_len_q=None,
    sink=None,
    pack_gqa=None,
    return_stats=False,
    stats_layout="contiguous",
):
    """Build the graph, opt into the matching FROST DSL engine, execute, return O (BHSD)."""
    import cudnn

    b, h_q, s_q, _ = q_gpu.shape
    d_v = v_gpu.shape[-1]
    o_gpu = torch.empty(b, s_q, h_q, d_v, device="cuda", dtype=dtype).transpose(1, 2)
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor_like(q_gpu)
    k = g.tensor_like(k_gpu)
    v = g.tensor_like(v_gpu)
    kw = dict(name="sdpa", q=q, k=k, v=v, generate_stats=return_stats, attn_scale=scale)
    vp = {q: q_gpu, k: k_gpu, v: v_gpu}
    if seq_len_kv is not None:
        slk = g.tensor_like(seq_len_kv)
        kw["seq_len_kv"] = slk
        kw["use_padding_mask"] = True
        vp[slk] = seq_len_kv
        # padding_mask requires a seq_len_q companion; without a caller-provided
        # seq_len_q the kernel trims only KV (Q full).
        slq_t = seq_len_q if seq_len_q is not None else torch.full((b, 1, 1, 1), s_q, dtype=torch.int32, device="cuda")
        slq = g.tensor_like(slq_t)
        kw["seq_len_q"] = slq
        vp[slq] = slq_t
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    kw.update(sdpa_kwargs)
    o, stats = g.sdpa(**kw)
    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats_gpu = None
    if return_stats:
        assert stats is not None
        stats_gpu = make_dense_stats(b, h_q, s_q, stats_layout)
        stats.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(), pack_gqa=pack_gqa)
    g.check_support()
    g.build_plans()
    vp[o] = o_gpu
    if stats_gpu is not None:
        vp[stats] = stats_gpu
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8))
    torch.cuda.synchronize()
    if return_stats:
        return o_gpu, stats_gpu
    return o_gpu


def _check_dsl_sm100_strided_stats(d_qk, d_v):
    _require_dsl()
    if torch.cuda.get_device_capability() == (10, 7):
        pytest.skip("SM107 serves only the per-tensor FP8 d128 forward path")
    b, h, s = 2, 4, 128
    dtype = torch.float16
    scale = 1.0 / math.sqrt(d_qk)
    q = _bhsd(b, h, s, d_qk, dtype)
    k = _bhsd(b, h, s, d_qk, dtype)
    v = _bhsd(b, h, s, d_v, dtype)

    _, contiguous_stats = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True), return_stats=True)
    o, strided_stats = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True), return_stats=True, stats_layout="strided")
    o_ref, stats_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True, return_stats=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(strided_stats, contiguous_stats, atol=0, rtol=0)
    torch.testing.assert_close(strided_stats.squeeze(-1), stats_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=59)
def test_dsl_sm100_strided_stats():
    """The SM100 half L0 flavor writes permuted, gapped LSE directly."""
    _check_dsl_sm100_strided_stats(128, 128)


@pytest.mark.L1
@pytest.mark.parametrize(("d_qk", "d_v"), [(192, 128), (256, 256), (512, 512)], ids=["d192_d128", "d256", "d512"])
@torch_fork_set_rng(seed=59)
def test_dsl_sm100_strided_stats_other_flavors(d_qk, d_v):
    """The remaining SM100 half flavors preserve dense Stats strides."""
    _check_dsl_sm100_strided_stats(d_qk, d_v)


@pytest.mark.L0
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("is_causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_d192_d128(dtype, is_causal):
    """Native DSv3 MLA shape: Q/K use d_qk=192 while V/O use d_v=128."""
    _require_dsl()
    b, h, s = 2, 8, 256
    d_qk, d_v = 192, 128
    scale = 1.0 / math.sqrt(d_qk)
    q = _bhsd(b, h, s, d_qk, dtype)
    k = _bhsd(b, h, s, d_qk, dtype)
    v = _bhsd(b, h, s, d_v, dtype)

    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=is_causal))
    o_ref = _ref_sdpa(q, k, v, is_causal=is_causal, scale=scale)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("d", _FLAVORS, ids=_FLAVOR_IDS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_causal_swa(dtype, d):
    """Causal + sliding-window (the standard SWA band): keep q-W <= kv <= q."""
    _require_dsl()
    b, h, s, W = 2, 8, 256, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v = (_bhsd(b, h, s, d, dtype) for _ in range(3))
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True, sliding_window_length=W + 1))
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True, swa_window=W)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("d", _FLAVORS, ids=_FLAVOR_IDS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_causal_bottom_right(dtype, d):
    """Bottom-right causal with SQ != SKV: keep kv <= q + (SKV - SQ)."""
    _require_dsl()
    b, h, s_q, s_kv = 2, 8, 128, 256
    scale = 1.0 / math.sqrt(d)
    q = _bhsd(b, h, s_q, d, dtype)
    k = _bhsd(b, h, s_kv, d, dtype)
    v = _bhsd(b, h, s_kv, d, dtype)
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask_bottom_right=True))
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True, bottom_right=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("d,d_v", [(128, 128), (192, 128), (256, 256), (512, 512)], ids=["llama", "dsv3", "qwen", "dsv4"])
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_band_right_partial_kv_tile(d, d_v):
    """Right-widened band over a PARTIAL final KV tile, tail covered by the band.

    S_kv % 128 != 0 with no padding mask is only admitted when the widened band
    provably masks the garbage tail columns (s_q + R <= s_kv — see
    engines._band_covers_kv_tail); this pins that the fast causal mask paths
    (which never consult eff_seqlen_kv) really do keep the tail masked."""
    _require_dsl()
    dtype = torch.bfloat16
    b, h, s_q, s_kv, R = 2, 4, 192, 328, 40  # s_q + R = 232 <= 328; 328 % 128 = 72
    scale = 1.0 / math.sqrt(d)
    q = _bhsd(b, h, s_q, d, dtype)
    k = _bhsd(b, h, s_kv, d, dtype)
    v = _bhsd(b, h, s_kv, d_v, dtype)
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(diagonal_band_right_bound=R))
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True, band_right=R)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("d", _FLAVORS, ids=_FLAVOR_IDS)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_band_right_multi_cluster(d):
    """Right-widened band across MULTIPLE Q-tile clusters: the widened columns
    of a cluster's bottom rows spill past the cluster's plain-causal KV-tile
    bound, so this fails if the widening reaches the per-element mask but not
    compute_kv_loop_bounds (the two must widen together — regression for the
    CFG.WINDOW_RIGHT bounds feed)."""
    _require_dsl()
    dtype = torch.bfloat16
    b, h, s, R = 1, 2, 1280, 40
    scale = 1.0 / math.sqrt(d)
    q, k, v = (_bhsd(b, h, s, d, dtype) for _ in range(3))
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(diagonal_band_right_bound=R))
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True, band_right=R)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("d", _FLAVORS, ids=_FLAVOR_IDS)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_ragged_skv_tail_via_padding(d):
    """Ragged S_kv (not a multiple of 128) with NO mask covering the tail:
    served through the kernel's padded path with synthesized full-length
    per-batch KV lengths (Capabilities.skv_tail_via_padding on the f16 rows —
    mathematically identical, tail masked by the synthesized lengths)."""
    _require_dsl()
    dtype = torch.bfloat16
    b, h, s_q, s_kv = 2, 4, 128, 200  # 200 % 128 != 0, no mask at all
    scale = 1.0 / math.sqrt(d)
    q = _bhsd(b, h, s_q, d, dtype)
    k = _bhsd(b, h, s_kv, d, dtype)
    v = _bhsd(b, h, s_kv, d, dtype)
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict())
    o_ref = _ref_sdpa_full(q, k, v, scale=scale)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_band_right_uncovered_tail_via_padding():
    """A widened band whose last unmasked column reaches past S_kv
    (s_q + R > s_kv) cannot rely on the band to mask the ragged tail; the f16
    rows serve it through the synthesized-padding path instead (the FP8 row's
    long-standing mechanism), so the garbage tail columns stay masked."""
    _require_dsl()
    dtype = torch.bfloat16
    b, h, s_q, s_kv, d, R = 2, 4, 192, 200, 128, 40  # s_q + R = 232 > 200; 200 % 128 != 0
    scale = 1.0 / math.sqrt(d)
    q = _bhsd(b, h, s_q, d, dtype)
    k = _bhsd(b, h, s_kv, d, dtype)
    v = _bhsd(b, h, s_kv, d, dtype)
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(diagonal_band_right_bound=R))
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True, band_right=R)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("d", _FLAVORS, ids=_FLAVOR_IDS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_padded(dtype, d):
    """Per-batch KV lengths (padded mask): keep kv < seq_kv_len[b]."""
    _require_dsl()
    b, h, s = 2, 8, 256
    scale = 1.0 / math.sqrt(d)
    q, k, v = (_bhsd(b, h, s, d, dtype) for _ in range(3))
    seq_len_kv = torch.tensor([180, 240], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(), seq_len_kv=seq_len_kv)
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, seq_kv_lens=seq_len_kv.flatten())
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_graph_api_padded_bottom_right_gqa():
    """Dense padded graph carrying SHORT per-batch seq_len_q + bottom-right
    causal + SWA + GQA, served end to end through the graph probe (the
    conjunction the retired bottom_right_padded_seq_q capability used to
    decline while the kernels anchored the BR diagonal at the global S_q)."""
    _require_dsl()
    d = 128
    b, h_q, h_kv, s_q, s_kv, W = 2, 8, 2, 128, 256, 47
    scale = 1.0 / math.sqrt(d)
    dtype = torch.float16
    q = _bhsd(b, h_q, s_q, d, dtype)
    k = _bhsd(b, h_kv, s_kv, d, dtype)
    v = _bhsd(b, h_kv, s_kv, d, dtype)
    seq_len_q = torch.tensor([93, 51], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    seq_len_kv = torch.tensor([137, 79], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    o = _run_dsl_graph(
        q,
        k,
        v,
        scale=scale,
        dtype=dtype,
        sdpa_kwargs=dict(use_causal_mask_bottom_right=True, sliding_window_length=W + 1),
        seq_len_kv=seq_len_kv,
        seq_len_q=seq_len_q,
    )
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True, bottom_right=True, swa_window=W, seq_q_lens=seq_len_q, seq_kv_lens=seq_len_kv)
    rows = torch.arange(s_q, device="cuda").view(1, 1, s_q, 1)
    dead = rows >= seq_len_q.view(b, 1, 1, 1)
    assert o[dead.expand_as(o)].abs().max().item() == 0.0, "trimmed Q rows are not zero"
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("d_qk,d_v", [(128, 128), (192, 128), (256, 256), (512, 512)], ids=["llama_d128", "mla_d192_d128", "qwen_d256", "dsv4_d512"])
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_bottom_right_padded_seq_q(d_qk, d_v):
    """Dense bottom-right causal with per-batch seq_len_q: the diagonal (and
    the SWA lower bound riding it) must anchor at the per-batch
    (seq_len_q[b], seq_len_kv[b]) corner, not the global S_q. GQA + LSE;
    batches one full-length / one trimming mid-tile / one zero-length, S_q
    spanning several CGA tiles so both the dead-tile collapse and a mid-tile
    diagonal are hit. Keep seq_len_kv[b] >= seq_len_q[b] — the suite-wide BR
    convention (no sequence starts with rows that see no keys)."""
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    dtype = torch.float16
    b, h_q, h_kv, W = 3, 8, 4, 200
    s_q = s_kv = 1024
    scale = 1.0 / math.sqrt(d_qk)
    q = _bhsd(b, h_q, s_q, d_qk, dtype)
    k = _bhsd(b, h_kv, s_kv, d_qk, dtype)
    v = _bhsd(b, h_kv, s_kv, d_v, dtype)
    o = torch.full((b, s_q, h_q, d_v), _THD_SENTINEL, device="cuda", dtype=dtype).transpose(1, 2)
    lse = torch.full((b, h_q, s_q), _THD_SENTINEL, dtype=torch.float32, device="cuda")
    seq_q_lens = torch.tensor([1024, 517, 0], dtype=torch.int32, device="cuda")
    seq_kv_lens = torch.tensor([1024, 800, 640], dtype=torch.int32, device="cuda")

    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_lse=lse,
        is_causal=True,
        causal_bottom_right=True,
        window_size_left=W,
        scale_softmax=scale,
        seq_kv_lens_present=True,
        seq_q_lens_present=True,
    )
    assert api.check_support()
    api.compile()
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse, seq_q_lens=seq_q_lens, seq_kv_lens=seq_kv_lens)
    torch.cuda.synchronize()

    o_ref, lse_ref = _ref_sdpa_full(
        q, k, v, scale=scale, is_causal=True, bottom_right=True, swa_window=W, seq_q_lens=seq_q_lens, seq_kv_lens=seq_kv_lens, return_stats=True
    )
    rows = torch.arange(s_q, device="cuda").view(1, 1, s_q, 1)
    dead = (rows >= seq_q_lens.view(b, 1, 1, 1)).expand_as(o)
    assert o[dead].abs().max().item() == 0.0, "trimmed Q rows are not zero"
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(lse, lse_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.L0
@pytest.mark.parametrize("d", _FLAVORS, ids=_FLAVOR_IDS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_sink(dtype, d):
    """Causal + attention sink: per-Q-head logit in the softmax denominator."""
    _require_dsl()
    b, h, s = 2, 8, 256
    scale = 1.0 / math.sqrt(d)
    q, k, v = (_bhsd(b, h, s, d, dtype) for _ in range(3))
    sink = torch.randn(1, h, 1, 1, dtype=torch.float32, device="cuda")
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True), sink=sink)
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True, sinks=sink.flatten())
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_execute_sink_lse_contract():
    """execute() rejects sinks inconsistent with the compiled specialization.

    has_sink is a compile-time specialization: substituting a zeros dummy for
    missing sinks would silently change the softmax denominator (a zero sink
    logit still contributes exp(0) mass), and sinks passed to a sink-less
    kernel would be silently dropped. Same for the LSE: has_lse is keyed on
    sample_lse (no Stats output -> the store is compiled out), so a requested
    LSE must be bound and an unrequested one is rejected, both directions.
    """
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, s, d = 1, 4, 256, 128
    q, k, v = (_bhsd(b, h, s, d, torch.float16) for _ in range(3))
    o = torch.empty_like(q)
    lse = torch.empty(b, h, s, dtype=torch.float32, device="cuda")
    sink = torch.randn(1, h, 1, 1, dtype=torch.float32, device="cuda")
    scale = 1.0 / math.sqrt(d)

    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse, is_causal=True, has_sink=True)
    assert api.check_support()
    api.compile()
    with pytest.raises(ValueError, match="sinks is required"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)
    with pytest.raises(ValueError, match="lse_tensor is required"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, sinks=sink)
    # Sinks are consumed as fp32 directly — no implicit cast (which would
    # allocate and launch a kernel on the execute hot path).
    with pytest.raises(ValueError, match="sinks must be float32"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse, sinks=sink.to(torch.bfloat16))

    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, is_causal=True)
    assert api.check_support()
    api.compile()
    with pytest.raises(ValueError, match="without sink support"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, sinks=sink)
    # Same contract for per-batch lengths: a specialization compiled without
    # them must not silently ignore a provided tensor (nor, the other way,
    # substitute a zeros dummy that would mask every row).
    seq_kv = torch.full((b,), s, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="without per-batch KV lengths"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_kv_lens=seq_kv)
    with pytest.raises(ValueError, match="without per-batch Q lengths"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=seq_kv)
    # No sample_lse: the kernel compiles with has_lse=False (LSE store folded
    # out, no dummy buffer anywhere) and the output is still correct — while
    # an unrequested lse_tensor is rejected (there is no LSE slot to bind).
    with pytest.raises(ValueError, match="without an LSE output"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o)
    torch.cuda.synchronize()
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)

    # THD LSE must be declared packed: token-major [t, h] or head-major
    # [h, t]. A dense-contiguous declaration (stride (H*S, S, 1)) is valid
    # head-major (head_stride S); a padded sequence stride matches NEITHER
    # layout and is rejected up front instead of being silently mis-addressed.
    lse_padded = torch.empty(h * s * 2, dtype=torch.float32, device="cuda").as_strided((b, h, s), (h * s * 2, s * 2, 2))
    with pytest.raises(ValueError, match="token-major"):
        SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse_padded, thd=True).check_support()
    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse, thd=True)
    assert api.check_support() and api.thd_stats_head_major and api.thd_stats_head_stride == s
    lse_tm = torch.empty(s * h, dtype=torch.float32, device="cuda").as_strided((b, h, s), (s * h, 1, h))
    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse_tm, thd=True)
    assert api.check_support() and not api.thd_stats_head_major

    # THD execute keeps the same strict presence contract in both directions:
    # the raises fire before any packing or launch.
    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse_tm, thd=True)
    assert api.check_support()
    api.compile()
    with pytest.raises(ValueError, match="lse_tensor is required"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=seq_kv, seq_kv_lens=seq_kv)
    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, thd=True)
    assert api.check_support()
    api.compile()
    with pytest.raises(ValueError, match="without an LSE output"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=seq_kv, seq_kv_lens=seq_kv, lse_tensor=lse_tm)


@pytest.mark.L0
@pytest.mark.parametrize("d", _FLAVORS, ids=_FLAVOR_IDS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_gqa(dtype, d):
    """GQA: H_q=8, H_kv=2 (K/V shared across groups), causal."""
    _require_dsl()
    b, h_q, h_kv, s = 2, 8, 2, 256
    scale = 1.0 / math.sqrt(d)
    q = _bhsd(b, h_q, s, d, dtype)
    k = _bhsd(b, h_kv, s, d, dtype)
    v = _bhsd(b, h_kv, s, d, dtype)
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True))
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


# --- PackGQA: TILE_M/G tokens x G query heads per tile -------
def _pack_gqa_case(d, h_q, h_kv, s_q, s_kv, dtype, *, sdpa_kwargs):
    q = _bhsd(2, h_q, s_q, d, dtype)
    k = _bhsd(2, h_kv, s_kv, d, dtype)
    v = _bhsd(2, h_kv, s_kv, d, dtype)
    scale = 1.0 / math.sqrt(d)
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=sdpa_kwargs, pack_gqa=True)
    return q, k, v, scale, o


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_pack_gqa_false_pinned():
    """pack_gqa=False pinned on a GQA graph runs the classic unpacked kernel
    and matches the reference — the knob's False side verified numerically,
    independent of what the auto rule would rank first on this device."""
    _require_dsl()
    d, h_q, h_kv = 128, 8, 2
    q = _bhsd(2, h_q, 40, d, torch.float16)
    k = _bhsd(2, h_kv, 256, d, torch.float16)
    v = _bhsd(2, h_kv, 256, d, torch.float16)
    scale = 1.0 / math.sqrt(d)
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=torch.float16, sdpa_kwargs=dict(use_causal_mask=True), pack_gqa=False)
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize(
    "h_q,h_kv",
    [(8, 4), (8, 2), (8, 1), (16, 1)],
    ids=["g2", "g4", "g8_mqa", "g16_mqa"],
)
@pytest.mark.parametrize("d", [128, 256], ids=["d128", "d256"])
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_pack_gqa_ratios(d, h_q, h_kv):
    """Packed plans across GQA ratios (incl. MQA), causal, tile-unaligned s_q."""
    _require_dsl()
    q, k, v, scale, o = _pack_gqa_case(d, h_q, h_kv, 40, 256, torch.float16, sdpa_kwargs=dict(use_causal_mask=True))
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L1
@pytest.mark.parametrize("h_q,h_kv", [(8, 2), (64, 1), (128, 1)], ids=["g4", "g64_mqa", "g128_mqa"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_pack_gqa_d512(dtype, h_q, h_kv):
    """Packed d512 flavor (SPLIT_PIPELINE / cga4; epilogue in the softmax WG).

    G=64 packs 2 tokens per tile and G=128 one token per tile — the whole
    tile is a single token's head family (MQA decode geometry)."""
    _require_dsl()
    q, k, v, scale, o = _pack_gqa_case(512, h_q, h_kv, 40, 256, dtype, sdpa_kwargs=dict(use_causal_mask=True))
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("h_q,s_q", [(64, 1), (64, 3), (128, 4)], ids=["g64_q1", "g64_q3_mtp", "g128_q4_mtp"])
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_pack_gqa_d512_mqa_decode(h_q, s_q):
    """MQA decode/MTP geometry on d512 (single shared KV head, long cache):
    q_tokens 1-4 against s_kv=2048 with a bottom-right band — the packed
    tile is dominated by one token's head family, and the whole q span fits
    inside one packed tile span."""
    _require_dsl()
    dtype = torch.bfloat16
    q = _bhsd(2, h_q, s_q, 512, dtype)
    k = _bhsd(2, 1, 2048, 512, dtype)
    v = _bhsd(2, 1, 2048, 512, dtype)
    scale = 1.0 / math.sqrt(512)
    kw = dict(use_causal_mask_bottom_right=True) if s_q > 1 else dict()
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=kw, pack_gqa=True)
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=s_q > 1, bottom_right=s_q > 1)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L1
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_pack_gqa_d192_d128(dtype):
    """Packed native DSv3 shape (d_qk=192 / d_v=128, QO_ALIAS, LPT-only)."""
    _require_dsl()
    b, h_q, h_kv, s_q, s_kv = 2, 8, 2, 40, 256
    scale = 1.0 / math.sqrt(192)
    q = _bhsd(b, h_q, s_q, 192, dtype)
    k = _bhsd(b, h_kv, s_kv, 192, dtype)
    v = _bhsd(b, h_kv, s_kv, 128, dtype)
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True), pack_gqa=True)
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize(
    "d,h_q,h_kv,s_q",
    [
        # d128: CGA token span 512/G.  G=8: sub-span (8), exact span (64), tail (100).
        (128, 64, 8, 8),
        (128, 64, 8, 64),
        (128, 64, 8, 100),
        # d256: CGA token span 256/G.  G=16: sub-span (4), exact span (16), tail (25).
        (256, 32, 2, 4),
        (256, 32, 2, 16),
        (256, 32, 2, 25),
    ],
    ids=["d128_subspan", "d128_exact", "d128_tail", "d256_subspan", "d256_exact", "d256_tail"],
)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_pack_gqa_tiles(d, h_q, h_kv, s_q):
    """Packed tile geometry edges: s_q*G below / at / past the CGA tile span."""
    _require_dsl()
    q, k, v, scale, o = _pack_gqa_case(d, h_q, h_kv, s_q, 256, torch.float16, sdpa_kwargs=dict(use_causal_mask=True))
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L1
@pytest.mark.parametrize("s_q", [1, 129, 1023, 2048])
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_pack_gqa_tiles_deep(s_q):
    """Packed multi-wave / odd-length shapes (persistent scheduler wraps)."""
    _require_dsl()
    q, k, v, scale, o = _pack_gqa_case(128, 64, 8, s_q, 2048, torch.float16, sdpa_kwargs=dict(use_causal_mask=True))
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize(
    "mask",
    ["none", "causal", "causal_br", "swa", "band_right", "padded", "sink_swa"],
)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_pack_gqa_features(mask):
    """Packed plans x the dense mask/sink/stats envelope, with LSE checked —
    the packed epilogue scatters LSE across G head rows and reads per-row
    sinks, so stats are the sharpest probe of the row-identity math."""
    _require_dsl()
    d, b, h_q, h_kv, s_q, s_kv = 128, 2, 8, 2, 40, 256
    dtype = torch.float16
    scale = 1.0 / math.sqrt(d)
    q = _bhsd(b, h_q, s_q, d, dtype)
    k = _bhsd(b, h_kv, s_kv, d, dtype)
    v = _bhsd(b, h_kv, s_kv, d, dtype)
    kw = dict()
    ref = dict()
    seq_len_kv = None
    sink = None
    if mask == "causal":
        kw, ref = dict(use_causal_mask=True), dict(is_causal=True)
    elif mask == "causal_br":
        kw, ref = dict(use_causal_mask_bottom_right=True), dict(is_causal=True, bottom_right=True)
    elif mask == "swa":
        kw, ref = dict(use_causal_mask=True, sliding_window_length=17), dict(is_causal=True, swa_window=16)
    elif mask == "band_right":
        # diagonal_band_right_bound implies the causal diagonal (the graph API
        # rejects combining it with use_causal_mask).
        kw, ref = dict(diagonal_band_right_bound=8), dict(is_causal=True, band_right=8)
    elif mask == "padded":
        seq_len_kv = torch.tensor([180, 240], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
        ref = dict(seq_kv_lens=seq_len_kv.flatten())
    elif mask == "sink_swa":
        sink = torch.randn(1, h_q, 1, 1, dtype=torch.float32, device="cuda")
        kw, ref = dict(use_causal_mask=True, sliding_window_length=17), dict(is_causal=True, swa_window=16, sinks=sink.flatten())
    o, lse = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=kw, seq_len_kv=seq_len_kv, sink=sink, pack_gqa=True, return_stats=True)
    o_ref, lse_ref = _ref_sdpa_full(q, k, v, scale=scale, return_stats=True, **ref)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(lse, lse_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_pack_gqa_qtrim():
    """Packed dense padded-Q trim: rows past seq_len_q[b] write O := 0 and
    LSE := -inf — under packing the trim predicate is per-TOKEN, exercised
    here with per-batch lengths that are not multiples of the token span."""
    _require_dsl()
    d, b, h_q, h_kv, s_q, s_kv = 128, 2, 8, 2, 128, 256
    dtype = torch.float16
    scale = 1.0 / math.sqrt(d)
    q = _bhsd(b, h_q, s_q, d, dtype)
    k = _bhsd(b, h_kv, s_kv, d, dtype)
    v = _bhsd(b, h_kv, s_kv, d, dtype)
    q_lens = [37, 90]
    kv_lens = [180, 240]
    seq_len_q = torch.tensor(q_lens, dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    seq_len_kv = torch.tensor(kv_lens, dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    o, lse = _run_dsl_graph(
        q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True), seq_len_kv=seq_len_kv, seq_len_q=seq_len_q, pack_gqa=True, return_stats=True
    )
    o_ref, lse_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True, seq_kv_lens=seq_len_kv.flatten(), return_stats=True)
    for i, ql in enumerate(q_lens):
        o_ref[i, :, ql:, :] = 0.0
        lse_ref[i, :, ql:] = float("-inf")
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(lse, lse_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_pack_gqa_knob_contract():
    """pack_gqa is honored-or-ineligible: no packed plan exists for MHA or a
    ratio that does not divide tile_m, and an unpacked plan always does."""
    _require_dsl()
    import cudnn

    def _plans_for(h_q, h_kv, s, b=2):
        d, dtype = 128, torch.float16
        g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        q = g.tensor_like(_bhsd(b, h_q, s, d, dtype))
        k = g.tensor_like(_bhsd(b, h_kv, s, d, dtype))
        v = g.tensor_like(_bhsd(b, h_kv, s, d, dtype))
        o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, generate_stats=False, attn_scale=1.0 / math.sqrt(d), use_causal_mask=True)
        o.set_output(True).set_dim((b, h_q, s, d)).set_stride((s * h_q * d, d, h_q * d, 1))
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        name = engine_name()
        out = []
        for i in range(len(g.plans)):
            pn = g.get_plan_name_at_index(i)
            if pn == name or pn.startswith(name + "["):
                out.append(getattr(g.plans[i].knobs, "pack_gqa", None))
        return out

    # MHA: only unpacked plans.
    pg = _plans_for(8, 8, 64)
    assert pg and True not in pg, f"MHA graph must not rank a packed plan; got {pg}"
    # ratios that do not divide tile_m (full-ratio contract): only unpacked plans.
    pg = _plans_for(6, 2, 64)
    assert pg and True not in pg, f"ratio-3 graph must not rank a packed plan; got {pg}"
    pg = _plans_for(12, 2, 64)
    assert pg and True not in pg, f"ratio-6 graph must not rank a packed plan; got {pg}"
    # GQA small s_q (s_q < the CGA tile): both variants ranked, packed
    # first — the rule is shape-only, so no batch/SM-count staging needed.
    pg = _plans_for(64, 8, 64)
    assert pg[0] is True and False in pg, f"small-s_q GQA should rank packed first with unpacked runner-up; got {pg}"
    # GQA full prefill: both variants ranked, unpacked first.
    pg = _plans_for(64, 8, 8192)
    assert pg[0] is False and True in pg, f"full-prefill GQA should rank unpacked first with packed runner-up; got {pg}"


# THD/varlen: packed [T,H,D] + per-operand ragged_offset (exclusive-prefix-sum of
# seq_len) + seq_len_q/kv + use_padding_mask. Each sequence attends only within
# itself (no cross-sequence attention).
@pytest.mark.L0
@pytest.mark.parametrize("d", _FLAVORS, ids=_FLAVOR_IDS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_thd(dtype, d):
    """THD/varlen self-attention, per-sequence causal; two packed sequences of unequal length."""
    _require_dsl()
    import cudnn

    dev = "cuda"
    H = 8
    seq_lens = [200, 150]
    B = len(seq_lens)
    S_max = max(seq_lens)
    T = sum(seq_lens)
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    scale = 1.0 / math.sqrt(d)

    q_pk = torch.randn(T, H, d, device=dev, dtype=dtype)
    k_pk = torch.randn(T, H, d, device=dev, dtype=dtype)
    v_pk = torch.randn(T, H, d, device=dev, dtype=dtype)

    # Dense-sized storage [B*S_max*H*D] with the packed data in the first T*H*D
    # elements, viewed as dense [B,H,S_max,D]; ragged_offset maps each sequence to
    # its packed slice.
    stride = (S_max * H * d, d, H * d, 1)

    def _dense_buf(packed):
        stor = torch.zeros(B * S_max * H * d, device=dev, dtype=dtype)
        stor[: T * H * d] = packed.reshape(-1)
        return stor, stor.as_strided((B, H, S_max, d), stride)

    q_stor, q_gpu = _dense_buf(q_pk)
    k_stor, k_gpu = _dense_buf(k_pk)
    v_stor, v_gpu = _dense_buf(v_pk)
    o_stor = torch.zeros(B * S_max * H * d, device=dev, dtype=dtype)
    o_gpu = o_stor.as_strided((B, H, S_max, d), stride)

    slq = torch.tensor(seq_lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = slq.clone()
    cu_t = torch.tensor(cu, dtype=torch.int64, device=dev)
    ro = (cu_t * H * d).view(B + 1, 1, 1, 1)

    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H, S_max, d], stride=list(stride), data_type=io, name="q")
    tk = g.tensor(dim=[B, H, S_max, d], stride=list(stride), data_type=io, name="k")
    tv = g.tensor(dim=[B, H, S_max, d], stride=list(stride), data_type=io, name="v")
    sq = g.tensor_like(slq)
    skv = g.tensor_like(slk)
    qro = g.tensor_like(ro)
    kro = g.tensor_like(ro)
    vro = g.tensor_like(ro)
    oro = g.tensor_like(ro)
    tq.set_ragged_offset(qro)
    tk.set_ragged_offset(kro)
    tv.set_ragged_offset(vro)
    o, _ = g.sdpa(
        name="sdpa", q=tq, k=tk, v=tv, generate_stats=False, attn_scale=scale, use_causal_mask=True, use_padding_mask=True, seq_len_q=sq, seq_len_kv=skv
    )
    o.set_output(True).set_dim([B, H, S_max, d]).set_stride(list(stride))
    o.set_ragged_offset(oro)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name())
    g.check_support()
    g.build_plans()
    vp = {tq: q_gpu, tk: k_gpu, tv: v_gpu, o: o_gpu, sq: slq, skv: slk, qro: ro, kro: ro, vro: ro, oro: ro}
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()

    o_ref = torch.zeros(T, H, d, device=dev, dtype=dtype)
    for b in range(B):
        lo, hi = cu[b], cu[b + 1]
        qb = q_pk[lo:hi].permute(1, 0, 2).unsqueeze(0)
        kb = k_pk[lo:hi].permute(1, 0, 2).unsqueeze(0)
        vb = v_pk[lo:hi].permute(1, 0, 2).unsqueeze(0)
        ob = _ref_sdpa_full(qb, kb, vb, scale=scale, is_causal=True)
        o_ref[lo:hi] = ob.squeeze(0).permute(1, 0, 2)

    o_out = o_stor[: T * H * d].reshape(T, H, d)
    torch.testing.assert_close(o_out, o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("d", _FLAVORS, ids=_FLAVOR_IDS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_thd_cross(dtype, d):
    """THD/varlen cross-attention: unequal per-sequence Q and K/V lengths, with unequal packed totals."""
    _require_dsl()
    import cudnn

    dev = "cuda"
    H = 8
    seq_lens_q = [200, 150]
    seq_lens_kv = [180, 120]
    B = len(seq_lens_q)
    scale = 1.0 / math.sqrt(d)

    def _cu(seq_lens):
        cu = [0]
        for s in seq_lens:
            cu.append(cu[-1] + s)
        return cu

    cu_q, cu_k = _cu(seq_lens_q), _cu(seq_lens_kv)
    S_max_q, S_max_kv = max(seq_lens_q), max(seq_lens_kv)
    T_q, T_kv = cu_q[-1], cu_k[-1]

    q_pk = torch.randn(T_q, H, d, device=dev, dtype=dtype)
    k_pk = torch.randn(T_kv, H, d, device=dev, dtype=dtype)
    v_pk = torch.randn(T_kv, H, d, device=dev, dtype=dtype)

    def _dense_buf(packed, s_max, t):
        stride = (s_max * H * d, d, H * d, 1)
        stor = torch.zeros(B * s_max * H * d, device=dev, dtype=dtype)
        stor[: t * H * d] = packed.reshape(-1)
        return stor, stor.as_strided((B, H, s_max, d), stride), stride

    q_stor, q_gpu, stride_q = _dense_buf(q_pk, S_max_q, T_q)
    k_stor, k_gpu, stride_kv = _dense_buf(k_pk, S_max_kv, T_kv)
    v_stor, v_gpu, _ = _dense_buf(v_pk, S_max_kv, T_kv)
    o_stor = torch.zeros(B * S_max_q * H * d, device=dev, dtype=dtype)
    o_gpu = o_stor.as_strided((B, H, S_max_q, d), stride_q)

    slq = torch.tensor(seq_lens_q, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(seq_lens_kv, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    ro_q = (torch.tensor(cu_q, dtype=torch.int64, device=dev) * H * d).view(B + 1, 1, 1, 1)
    ro_k = (torch.tensor(cu_k, dtype=torch.int64, device=dev) * H * d).view(B + 1, 1, 1, 1)

    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H, S_max_q, d], stride=list(stride_q), data_type=io, name="q")
    tk = g.tensor(dim=[B, H, S_max_kv, d], stride=list(stride_kv), data_type=io, name="k")
    tv = g.tensor(dim=[B, H, S_max_kv, d], stride=list(stride_kv), data_type=io, name="v")
    sq = g.tensor_like(slq)
    skv = g.tensor_like(slk)
    qro = g.tensor_like(ro_q)
    kro = g.tensor_like(ro_k)
    vro = g.tensor_like(ro_k)
    oro = g.tensor_like(ro_q)
    tq.set_ragged_offset(qro)
    tk.set_ragged_offset(kro)
    tv.set_ragged_offset(vro)
    o, _ = g.sdpa(
        name="sdpa", q=tq, k=tk, v=tv, generate_stats=False, attn_scale=scale, use_causal_mask=True, use_padding_mask=True, seq_len_q=sq, seq_len_kv=skv
    )
    o.set_output(True).set_dim([B, H, S_max_q, d]).set_stride(list(stride_q))
    o.set_ragged_offset(oro)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name())
    g.check_support()
    g.build_plans()
    vp = {tq: q_gpu, tk: k_gpu, tv: v_gpu, o: o_gpu, sq: slq, skv: slk, qro: ro_q, kro: ro_k, vro: ro_k, oro: ro_q}
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()

    o_ref = torch.zeros(T_q, H, d, device=dev, dtype=dtype)
    for b in range(B):
        qb = q_pk[cu_q[b] : cu_q[b + 1]].permute(1, 0, 2).unsqueeze(0)
        kb = k_pk[cu_k[b] : cu_k[b + 1]].permute(1, 0, 2).unsqueeze(0)
        vb = v_pk[cu_k[b] : cu_k[b + 1]].permute(1, 0, 2).unsqueeze(0)
        ob = _ref_sdpa_full(qb, kb, vb, scale=scale, is_causal=True)
        o_ref[cu_q[b] : cu_q[b + 1]] = ob.squeeze(0).permute(1, 0, 2)

    o_out = o_stor[: T_q * H * d].reshape(T_q, H, d)
    torch.testing.assert_close(o_out, o_ref, atol=5e-2, rtol=3e-2)


# Full-cartesian combo sweep: d x mask x sink x gqa/mha x layout x dtype. THD
# forces padding internally, so its mask axis is on top of that (no standalone
# "padded" entry); dense carries the explicit "padded" case.
_COMBO_SWA_W = 64
_COMBO_BAND_R = 40  # right-band widening bound (diagonal_band_right_bound)


def _mask_graph_kwargs(mask):
    """graph.sdpa kwargs for the causal-family masks (padded handled separately)."""
    import cudnn

    return {
        "none": {},
        "causal": dict(use_causal_mask=True),
        "causal_br": dict(use_causal_mask_bottom_right=True),
        "swa": dict(use_causal_mask=True, sliding_window_length=_COMBO_SWA_W + 1),
        "band": dict(diagonal_band_right_bound=_COMBO_BAND_R),
        "band_br": dict(diagonal_band_right_bound=_COMBO_BAND_R, diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT),
        "band_swa": dict(diagonal_band_right_bound=_COMBO_BAND_R, diagonal_band_left_bound=_COMBO_SWA_W + 1),
        "swa_br": dict(use_causal_mask_bottom_right=True, sliding_window_length=_COMBO_SWA_W + 1),
    }[mask]


def _mask_ref_kwargs(mask):
    """_ref_sdpa_full kwargs matching _mask_graph_kwargs."""
    return {
        "none": {},
        "causal": dict(is_causal=True),
        "causal_br": dict(is_causal=True, bottom_right=True),
        "swa": dict(is_causal=True, swa_window=_COMBO_SWA_W),
        "band": dict(is_causal=True, band_right=_COMBO_BAND_R),
        "band_br": dict(is_causal=True, bottom_right=True, band_right=_COMBO_BAND_R),
        "band_swa": dict(is_causal=True, band_right=_COMBO_BAND_R, swa_window=_COMBO_SWA_W),
        "swa_br": dict(is_causal=True, bottom_right=True, swa_window=_COMBO_SWA_W),
    }[mask]


def _run_dsl_thd_graph(
    q_pk,
    k_pk,
    v_pk,
    cu_q,
    cu_k,
    seq_lens_q,
    seq_lens_kv,
    *,
    scale,
    dtype,
    H_q,
    H_kv,
    d,
    sink=None,
    mask="causal",
    check_stats=False,
    stats_layout="token_major",
    cu_lens=False,
):
    """Build + execute a packed THD/varlen graph; returns the flat packed O
    storage buffer — plus, with ``check_stats``, the flat Stats storage and
    the padded token capacity of its head-major head stride.

    ``stats_layout`` selects the ragged Stats declaration: ``token_major``
    (``[t, h]``, sequence stride ``h_q``) or ``head_major`` (``[h, t]``,
    sequence stride 1 with a padded token-capacity head stride —
    FlashAttention's ``softmax_lse`` layout)."""
    import cudnn

    dev = "cuda"
    B = len(seq_lens_q)
    T_q, T_kv = cu_q[-1], cu_k[-1]
    # Clamp the declared extents: an all-zero seq_len batch still needs a
    # rank-legal (>0) graph dim; the padding mask carries the real lengths.
    S_max_q, S_max_kv = max(max(seq_lens_q), 1), max(max(seq_lens_kv), 1)

    def _dense_buf(packed, s_max, t, H):
        stride = (s_max * H * d, d, H * d, 1)
        stor = torch.zeros(B * s_max * H * d, device=dev, dtype=dtype)
        stor[: t * H * d] = packed.reshape(-1)
        return stor, stor.as_strided((B, H, s_max, d), stride), stride

    _, q_gpu, stride_q = _dense_buf(q_pk, S_max_q, T_q, H_q)
    _, k_gpu, stride_kv = _dense_buf(k_pk, S_max_kv, T_kv, H_kv)
    _, v_gpu, _ = _dense_buf(v_pk, S_max_kv, T_kv, H_kv)
    # Output/Stats storages carry a SENTINEL: the valid packed region is fully
    # written by the kernel (compared against the reference), while everything
    # else — the whole buffer when t_q == 0 — must come back untouched.
    o_stor = torch.full((B * S_max_q * H_q * d,), _THD_SENTINEL, device=dev, dtype=dtype)
    o_gpu = o_stor.as_strided((B, H_q, S_max_q, d), stride_q)

    slq = torch.tensor(seq_lens_q, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(seq_lens_kv, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    # cu_lens: bind the (B+1,) prefix-sum form (cu_seq_len_q/kv, cuDNN 9.24+)
    # instead of per-batch lengths.
    cuq_t = torch.tensor(cu_q, dtype=torch.int32, device=dev).view(B + 1, 1, 1, 1)
    cuk_t = torch.tensor(cu_k, dtype=torch.int32, device=dev).view(B + 1, 1, 1, 1)
    ro_q = (torch.tensor(cu_q, dtype=torch.int64, device=dev) * H_q * d).view(B + 1, 1, 1, 1)
    ro_k = (torch.tensor(cu_k, dtype=torch.int64, device=dev) * H_kv * d).view(B + 1, 1, 1, 1)

    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H_q, S_max_q, d], stride=list(stride_q), data_type=io, name="q")
    tk = g.tensor(dim=[B, H_kv, S_max_kv, d], stride=list(stride_kv), data_type=io, name="k")
    tv = g.tensor(dim=[B, H_kv, S_max_kv, d], stride=list(stride_kv), data_type=io, name="v")
    sq = g.tensor_like(cuq_t if cu_lens else slq)
    skv = g.tensor_like(cuk_t if cu_lens else slk)
    qro = g.tensor_like(ro_q)
    kro = g.tensor_like(ro_k)
    vro = g.tensor_like(ro_k)
    oro = g.tensor_like(ro_q)
    tq.set_ragged_offset(qro)
    tk.set_ragged_offset(kro)
    tv.set_ragged_offset(vro)
    kw = dict(name="sdpa", q=tq, k=tk, v=tv, generate_stats=check_stats, attn_scale=scale, use_padding_mask=True)
    if cu_lens:
        kw.update(cu_seq_len_q=sq, cu_seq_len_kv=skv)
    else:
        kw.update(seq_len_q=sq, seq_len_kv=skv)
    kw.update(_mask_graph_kwargs(mask))
    vp = {tq: q_gpu, tk: k_gpu, tv: v_gpu, sq: (cuq_t if cu_lens else slq), skv: (cuk_t if cu_lens else slk), qro: ro_q, kro: ro_k, vro: ro_k, oro: ro_q}
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    o, stats = g.sdpa(**kw)
    o.set_output(True).set_dim([B, H_q, S_max_q, d]).set_stride(list(stride_q))
    o.set_ragged_offset(oro)
    stats_stor = None
    t_cap = max(64, -(-T_q // 64) * 64)
    if check_stats:
        assert stats is not None
        stats.set_output(True)
        stats.set_data_type(cudnn.data_type.FLOAT)
        if stats_layout == "head_major":
            # [h, t]: tokens contiguous within a head, heads strided by the
            # padded token capacity; offsets = cu_q * stride_s = cu_q.
            stats_stor = torch.full((H_q * t_cap,), _THD_SENTINEL, dtype=torch.float32, device=dev)
            stats.set_dim((B, H_q, S_max_q, 1)).set_stride((H_q * t_cap, t_cap, 1, 1))
            stats_ro_t = (ro_q.flatten() // (H_q * d)).view(B + 1, 1, 1, 1).contiguous()
        else:
            # [t, h]: heads contiguous within a token; offsets = cu_q * h_q.
            stats_stor = torch.full((B * S_max_q * H_q,), _THD_SENTINEL, dtype=torch.float32, device=dev)
            stats.set_dim((B, H_q, S_max_q, 1)).set_stride((S_max_q * H_q, 1, H_q, 1))
            stats_ro_t = (ro_q.flatten() // d).view(B + 1, 1, 1, 1).contiguous()
        stats_ro = g.tensor_like(stats_ro_t, name="stats_ro")
        stats.set_ragged_offset(stats_ro)
        vp[stats_ro] = stats_ro_t
        vp[stats] = stats_stor

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name())
    g.check_support()
    g.build_plans()
    vp[o] = o_gpu
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()
    return (o_stor, stats_stor, t_cap) if check_stats else o_stor


def _combo_dense(d, dtype, H_q, H_kv, scale, sink_t, mask):
    b = 2
    s_q, s_kv = (128, 256) if mask in ("causal_br", "band_br", "swa_br") else (256, 256)
    q = _bhsd(b, H_q, s_q, d, dtype)
    k = _bhsd(b, H_kv, s_kv, d, dtype)
    v = _bhsd(b, H_kv, s_kv, d, dtype)
    seq_len_kv = None
    ref_kw = {}
    graph_kw = {}
    if mask == "padded":
        kv0, kv1 = max(1, s_kv - 76), s_kv - 16
        seq_len_kv = torch.tensor([kv0, kv1], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
        ref_kw = dict(seq_kv_lens=seq_len_kv.flatten())
    else:
        graph_kw = _mask_graph_kwargs(mask)
        ref_kw = _mask_ref_kwargs(mask)
    o = _run_dsl_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=graph_kw, seq_len_kv=seq_len_kv, sink=sink_t)
    o_ref = _ref_sdpa_full(q, k, v, scale=scale, sinks=(sink_t.flatten() if sink_t is not None else None), **ref_kw)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)


def _combo_thd(d, dtype, H_q, H_kv, scale, sink_t, mask):
    dev = "cuda"
    seq_lens_q = [200, 150]
    seq_lens_kv = [180, 120]
    if mask in ("causal_br", "band_br", "swa_br"):
        # Bottom-right masks: keep seq_len_kv[b] >= seq_len_q[b] so no sequence
        # has fully-masked rows (the torch softmax reference NaNs on those).
        seq_lens_q = [150, 90]
        seq_lens_kv = [180, 120]
    B = len(seq_lens_q)

    def _cu(sl):
        c = [0]
        for s in sl:
            c.append(c[-1] + s)
        return c

    cu_q, cu_k = _cu(seq_lens_q), _cu(seq_lens_kv)
    T_q, T_kv = cu_q[-1], cu_k[-1]
    q_pk = torch.randn(T_q, H_q, d, device=dev, dtype=dtype)
    k_pk = torch.randn(T_kv, H_kv, d, device=dev, dtype=dtype)
    v_pk = torch.randn(T_kv, H_kv, d, device=dev, dtype=dtype)

    o_stor = _run_dsl_thd_graph(
        q_pk, k_pk, v_pk, cu_q, cu_k, seq_lens_q, seq_lens_kv, scale=scale, dtype=dtype, H_q=H_q, H_kv=H_kv, d=d, sink=sink_t, mask=mask
    )

    ref_kw = _mask_ref_kwargs(mask)
    sinks = sink_t.flatten() if sink_t is not None else None
    o_ref = torch.zeros(T_q, H_q, d, device=dev, dtype=dtype)
    for b in range(B):
        qb = q_pk[cu_q[b] : cu_q[b + 1]].permute(1, 0, 2).unsqueeze(0)
        kb = k_pk[cu_k[b] : cu_k[b + 1]].permute(1, 0, 2).unsqueeze(0)
        vb = v_pk[cu_k[b] : cu_k[b + 1]].permute(1, 0, 2).unsqueeze(0)
        ob = _ref_sdpa_full(qb, kb, vb, scale=scale, sinks=sinks, **ref_kw)
        o_ref[cu_q[b] : cu_q[b + 1]] = ob.squeeze(0).permute(1, 0, 2)

    o_out = o_stor[: T_q * H_q * d].reshape(T_q, H_q, d)
    torch.testing.assert_close(o_out, o_ref, atol=5e-2, rtol=3e-2)


def _run_thd_stats_case(
    *, seq_lens_q, seq_lens_kv, d=128, dtype=torch.float16, H_q=8, H_kv=8, mask="causal", with_sink=False, stats_layout="token_major", cu_lens=False
):
    """Run a THD (ragged) graph with generate_stats and check O and the ragged
    Stats against per-sequence references, in the declared Stats layout."""
    _require_dsl()

    dev = "cuda"
    scale = 1.0 / math.sqrt(d)
    B = len(seq_lens_q)

    def _cu(sl):
        c = [0]
        for s in sl:
            c.append(c[-1] + s)
        return c

    cu_q, cu_k = _cu(seq_lens_q), _cu(seq_lens_kv)
    T_q, T_kv = cu_q[-1], cu_k[-1]
    q_pk = torch.randn(T_q, H_q, d, device=dev, dtype=dtype)
    k_pk = torch.randn(T_kv, H_kv, d, device=dev, dtype=dtype)
    v_pk = torch.randn(T_kv, H_kv, d, device=dev, dtype=dtype)
    sink_t = torch.randn(1, H_q, 1, 1, dtype=torch.float32, device=dev) if with_sink else None

    o_stor, stats_stor, t_cap = _run_dsl_thd_graph(
        q_pk,
        k_pk,
        v_pk,
        cu_q,
        cu_k,
        seq_lens_q,
        seq_lens_kv,
        scale=scale,
        dtype=dtype,
        H_q=H_q,
        H_kv=H_kv,
        d=d,
        sink=sink_t,
        mask=mask,
        check_stats=True,
        stats_layout=stats_layout,
        cu_lens=cu_lens,
    )

    if T_q == 0:
        # No query token exists anywhere: execute must be a complete no-op —
        # the sentinel-filled O and ragged Stats storages come back untouched.
        assert (o_stor == _THD_SENTINEL).all(), "t_q == 0 wrote to O"
        assert (stats_stor == _THD_SENTINEL).all(), "t_q == 0 wrote to the ragged Stats"
        return

    if stats_layout == "head_major":
        packed_stats = stats_stor.view(H_q, t_cap)  # (H, head_stride); tokens at [:, cu[i]:cu[i+1]]
    else:
        packed_stats = stats_stor[: max(T_q, 1) * H_q].view(max(T_q, 1), H_q)  # (T, H)
    ref_kw = _mask_ref_kwargs(mask)
    sinks = sink_t.flatten() if sink_t is not None else None
    packed_o = o_stor[: max(T_q, 1) * H_q * d].view(max(T_q, 1), H_q, d)
    for i, (nq, _nkv) in enumerate(zip(seq_lens_q, seq_lens_kv)):
        if nq == 0:
            continue
        qb = q_pk[cu_q[i] : cu_q[i + 1]].permute(1, 0, 2).unsqueeze(0)
        kb = k_pk[cu_k[i] : cu_k[i + 1]].permute(1, 0, 2).unsqueeze(0)
        vb = v_pk[cu_k[i] : cu_k[i + 1]].permute(1, 0, 2).unsqueeze(0)
        expected, expected_lse = _ref_sdpa_full(qb, kb, vb, scale=scale, sinks=sinks, return_stats=True, **ref_kw)
        got_o = packed_o[cu_q[i] : cu_q[i + 1]].permute(1, 0, 2).unsqueeze(0)
        torch.testing.assert_close(got_o, expected, atol=5e-2, rtol=3e-2)
        if stats_layout == "head_major":
            got_lse = packed_stats[:, cu_q[i] : cu_q[i + 1]].unsqueeze(0)  # (H, T_i) -> (1, H, T_i)
        else:
            got_lse = packed_stats[cu_q[i] : cu_q[i + 1]].t().unsqueeze(0)  # (T_i, H) -> (1, H, T_i)
        torch.testing.assert_close(got_lse, expected_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.L0
@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
@pytest.mark.parametrize("d", _FLAVORS, ids=_FLAVOR_IDS)
@torch_fork_set_rng(seed=30)
def test_dsl_sm100_thd_stats(d, stats_layout):
    """THD + generate_stats: the ragged Stats output is written in the
    caller's declared layout — token-major [t, h] or head-major [h, t] —
    across every f16 flavor."""

    _run_thd_stats_case(seq_lens_q=[200, 150], seq_lens_kv=[200, 150], d=d, mask="causal", stats_layout=stats_layout)


@pytest.mark.L1
@torch_fork_set_rng(seed=32)
def test_dsl_sm100_thd_swa_stats():
    """THD + causal left sliding window + ragged Stats: the window trims the
    per-sequence LSE denominator."""

    _run_thd_stats_case(seq_lens_q=[150, 90], seq_lens_kv=[150, 90], mask="swa", stats_layout="token_major")


@pytest.mark.L1
@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
@torch_fork_set_rng(seed=25)
def test_dsl_sm100_thd_gqa_sink_stats(stats_layout):
    """THD + GQA + attention sink, with the sink entering the ragged Stats
    (both declared layouts)."""

    _run_thd_stats_case(seq_lens_q=[130, 70], seq_lens_kv=[130, 70], H_q=8, H_kv=2, mask="causal", with_sink=True, stats_layout=stats_layout)


@pytest.mark.L1
@torch_fork_set_rng(seed=26)
def test_dsl_sm100_thd_zero_length_sequence_stats():
    """A zero-length sequence contributes no tokens and must not perturb its
    packed neighbors (O and ragged Stats). The last sequence has Q tokens but
    ZERO keys inside a live launch: its rows must come back O := 0 with
    LSE := -inf through the kernel's row_dead guard, not stale memory."""

    _run_thd_stats_case(seq_lens_q=[128, 0, 64], seq_lens_kv=[100, 0, 0], mask="causal", stats_layout="token_major")


@pytest.mark.L1
@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
@pytest.mark.parametrize("with_sink", [False, True], ids=["no_sink", "sink"])
@torch_fork_set_rng(seed=31)
def test_dsl_sm100_thd_all_kv_zero_stats(with_sink, stats_layout):
    """Every KV length zero: the launch goes through the KERNEL's dead-row
    path (O := 0, LSE := -inf, or the sink value alone — the sink column
    keeps the softmax denominator alive) with the packed KV extent clamped
    to one never-dereferenced token (a zero-token K/V view cannot back a
    CuTe layout) — no adapter-side fills, in either declared layout."""

    _run_thd_stats_case(seq_lens_q=[64, 32], seq_lens_kv=[0, 0], mask="none", with_sink=with_sink, stats_layout=stats_layout)


@pytest.mark.L1
@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
@torch_fork_set_rng(seed=34)
def test_dsl_sm100_thd_all_q_zero_stats(stats_layout):
    """Every Q length zero (t_q == 0): no query token exists anywhere, so the
    packed O/Stats have zero rows and execute must be a complete NO-OP — the
    sentinel-filled buffers come back untouched, with live KV and with the
    fully-degenerate all-zero KV as well."""

    _run_thd_stats_case(seq_lens_q=[0, 0], seq_lens_kv=[50, 30], mask="none", stats_layout=stats_layout)
    _run_thd_stats_case(seq_lens_q=[0, 0], seq_lens_kv=[0, 0], mask="none", stats_layout=stats_layout)


@pytest.mark.L0
@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
@torch_fork_set_rng(seed=35)
def test_dsl_sm100_thd_cu_seq_len_stats(stats_layout):
    """THD with the cu_seq_len_q/kv length form ((B+1,) prefix sums, cuDNN
    9.24+ — the form TE/PyT/vLLM natively hold): the lowering derives the
    per-batch lengths host-side from the same inherent tolist round-trip, so
    results are identical to the seq_len form, ragged Stats included."""

    _run_thd_stats_case(seq_lens_q=[200, 150], seq_lens_kv=[200, 150], mask="causal", stats_layout=stats_layout, cu_lens=True)


@pytest.mark.L1
@torch_fork_set_rng(seed=36)
def test_dsl_sm100_thd_cu_seq_len_zero_lens():
    """cu_seq_len form with degenerate lengths: a zero-length sequence
    (repeated prefix value) and an all-zero KV side keep the same dead-row /
    no-op semantics as the seq_len form."""

    _run_thd_stats_case(seq_lens_q=[128, 0, 64], seq_lens_kv=[100, 0, 0], mask="causal", cu_lens=True)
    _run_thd_stats_case(seq_lens_q=[64, 32], seq_lens_kv=[0, 0], mask="none", cu_lens=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=37)
def test_dsl_sm100_thd_compile_key_plan_time_only():
    """Issue #552: the THD compile key carries NO packed totals.

    ``compile()`` builds the one artifact at plan time (the token extents
    compile dynamic), and executes with DIFFERENT packed totals re-bind it —
    zero ``cute.compile`` calls on the execute path. Keying the compile on
    the totals degenerated into a fresh multi-second compile per step under
    continuous batching (the totals change every step), and correctness is
    checked per total to prove one artifact serves them all.
    """
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, s, d = 2, 4, 256, 128
    dtype = torch.float16
    scale = 1.0 / math.sqrt(d)
    q, k, v = (_bhsd(b, h, s, d, dtype) for _ in range(3))
    o = torch.zeros_like(q)
    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, thd=True)
    assert api.check_support()
    api.compile()
    # Plan-time compile: no deferred sentinel, the artifact already exists.
    assert api._compiled_kernel != "thd-deferred"
    info_plan = api._k_mod.compile.cache_info()

    def _run_and_check(seq_lens):
        lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=lens, seq_kv_lens=lens)
        torch.cuda.synchronize()
        base_q = q.transpose(1, 2).reshape(b * s, h, d)
        base_k = k.transpose(1, 2).reshape(b * s, h, d)
        base_v = v.transpose(1, 2).reshape(b * s, h, d)
        base_o = o.transpose(1, 2).reshape(b * s, h, d)
        off = 0
        for length in seq_lens:
            qs = base_q[off : off + length].float()
            ks = base_k[off : off + length].float()
            vs = base_v[off : off + length].float()
            scores = torch.einsum("lhd,mhd->hlm", qs, ks) * scale
            ref = torch.einsum("hlm,mhd->lhd", torch.softmax(scores, dim=-1), vs)
            torch.testing.assert_close(base_o[off : off + length].float(), ref, atol=5e-2, rtol=3e-2)
            off += length

    _run_and_check([200, 150])
    _run_and_check([64, 33])
    info_exec = api._k_mod.compile.cache_info()
    assert info_exec.misses == info_plan.misses, "a THD execute minted a new kernel compile (runtime data leaked into the compile key)"
    assert info_exec.hits >= info_plan.hits + 2


@pytest.mark.L0
@torch_fork_set_rng(seed=38)
def test_dsl_sm100_thd_over_launched_units_are_dead(monkeypatch):
    """Issue #552 (envelope-grid enabler): grid units past the live total are
    DEAD by kernel contract — the decode maps them to the batch == n_batch
    sentinel, every role takes the empty-KV path (eff_seqlen_kv reads
    cu_q[0] == 0), and neither O nor the ragged Stats is written (LSE
    predicate goes negative; the O-store role skips the TMA store). Padding
    the launch grid must therefore change nothing. Covers a zero-length
    LEADING sequence (pre-fix, dead units decoded to batch 0 and ran full KV
    loops there), a zero-length middle sequence, all-KV-zero (dead units on
    top of the packed-KV clamp), and both Stats layouts."""
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    envelope = SdpaFwdDslSm100._thd_unit_envelope
    monkeypatch.setattr(SdpaFwdDslSm100, "_thd_unit_envelope", lambda self: envelope(self) + 7)
    _run_thd_stats_case(seq_lens_q=[128, 0, 64], seq_lens_kv=[100, 0, 30], mask="causal", stats_layout="token_major")
    _run_thd_stats_case(seq_lens_q=[0, 64, 32], seq_lens_kv=[50, 40, 0], mask="none", stats_layout="head_major")
    _run_thd_stats_case(seq_lens_q=[64, 32], seq_lens_kv=[0, 0], mask="none", stats_layout="token_major")


@pytest.mark.L0
@torch_fork_set_rng(seed=39)
def test_dsl_sm100_thd_lens_never_reach_host():
    """Issue #552 (D2H removal): the length tensors are consumed ONLY on
    device — the setup kernel builds the metadata, the views bind buffer
    capacities, and the grid is the plan-time envelope. The old host
    round-trip helper (_thd_host_lens) is GONE from the adapter entirely —
    a host read of the lengths is structurally impossible — while full
    numerics run in both length forms."""
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDsl, SdpaFwdDslSm100

    assert not hasattr(SdpaFwdDsl, "_thd_host_lens") and not hasattr(SdpaFwdDslSm100, "_thd_host_lens")
    _run_thd_stats_case(seq_lens_q=[200, 150], seq_lens_kv=[180, 120], mask="causal", stats_layout="token_major")
    _run_thd_stats_case(seq_lens_q=[200, 150], seq_lens_kv=[180, 120], mask="causal", stats_layout="head_major", cu_lens=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=41)
def test_dsl_sm100_thd_execute_never_syncs():
    """Issue #552 endgame (d128): the THD execute performs NO synchronizing
    CUDA call — no length D2H, no pageable H2D, no device/stream sync —
    pinned by torch's sync debug mode ("error"), which raises on any. The
    grid is the plan-time envelope, the ragged views bind buffer
    capacities, and the metadata is built device-side; results are bitwise
    identical to an unguarded execute."""
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, s, d = 2, 4, 256, 128
    dtype = torch.float16
    q, k, v = (_bhsd(b, h, s, d, dtype) for _ in range(3))
    o = torch.zeros_like(q)
    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, thd=True)
    assert api.check_support()
    api.compile()
    lens = torch.tensor([200, 150], dtype=torch.int32, device="cuda")

    # Warm-up outside the guarded region: allocator pools and lazy launcher
    # state populate here, so the guarded execute reuses cached blocks.
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=lens, seq_kv_lens=lens)
    torch.cuda.synchronize()
    o_ref = o.clone()
    o.zero_()
    prev_sync_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(2)
    try:
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=lens, seq_kv_lens=lens)
    finally:
        torch.cuda.set_sync_debug_mode(prev_sync_mode)
    torch.cuda.synchronize()
    assert torch.equal(o, o_ref)


@pytest.mark.L1
@torch_fork_set_rng(seed=44)
def test_dsl_sm100_thd_cu_nonzero_base_normalized():
    """The device-side metadata build NORMALIZES cu prefix sums (subtracts
    element 0): the packed buffers are addressed from token 0, so a cu
    tensor sliced from a larger prefix (cu[0] != 0) means the same lengths
    and must produce bitwise-identical results. The old host path errored
    on cu[0] != 0; a device-side build cannot raise (Rule 3), it
    normalizes."""
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, s, d = 2, 4, 256, 128
    dtype = torch.float16
    q, k, v = (_bhsd(b, h, s, d, dtype) for _ in range(3))
    o = torch.zeros_like(q)
    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, thd=True, cu_seq_q_lens=True, cu_seq_kv_lens=True)
    assert api.check_support()
    api.compile()

    def _run(base_q, base_kv):
        # Distinct Q/KV prefix tensors with DIFFERENT lengths and bases: a
        # normalization that subtracts one side's base from the other (or
        # shares one tensor for both) cannot pass this by accident.
        cu_q = torch.tensor([base_q, base_q + 200, base_q + 350], dtype=torch.int32, device="cuda")
        cu_kv = torch.tensor([base_kv, base_kv + 180, base_kv + 310], dtype=torch.int32, device="cuda")
        o.zero_()
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=cu_q, seq_kv_lens=cu_kv)
        torch.cuda.synchronize()
        return o.clone()

    assert torch.equal(_run(0, 0), _run(1000, 7000))


@pytest.mark.L1
@torch_fork_set_rng(seed=43)
def test_dsl_sm100_thd_d192_d128_device_meta():
    """d192/d128 (native MLA head dims) THD through the device-meta +
    envelope path: per-sequence numerics via the direct API — the graph THD
    harness assumes d_qk == d_v, so this flavor's THD leg is pinned here."""
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, s = 2, 4, 256
    d_qk, d_v = 192, 128
    dtype = torch.float16
    scale = 1.0 / math.sqrt(d_qk)
    q, k = (_bhsd(b, h, s, d_qk, dtype) for _ in range(2))
    v = _bhsd(b, h, s, d_v, dtype)
    o = torch.zeros_like(v)
    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, thd=True)
    assert api.check_support()
    api.compile()
    seq_lens = [200, 150]
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=lens, seq_kv_lens=lens)
    torch.cuda.synchronize()
    base_q = q.transpose(1, 2).reshape(b * s, h, d_qk)
    base_k = k.transpose(1, 2).reshape(b * s, h, d_qk)
    base_v = v.transpose(1, 2).reshape(b * s, h, d_v)
    base_o = o.transpose(1, 2).reshape(b * s, h, d_v)
    off = 0
    for length in seq_lens:
        qs = base_q[off : off + length].float()
        ks = base_k[off : off + length].float()
        vs = base_v[off : off + length].float()
        scores = torch.einsum("lhd,mhd->hlm", qs, ks) * scale
        ref = torch.einsum("hlm,mhd->lhd", torch.softmax(scores, dim=-1), vs)
        torch.testing.assert_close(base_o[off : off + length].float(), ref, atol=5e-2, rtol=3e-2)
        off += length


@pytest.mark.L0
@torch_fork_set_rng(seed=42)
def test_dsl_sm100_thd_execute_cuda_graph_capture():
    """Issue #552 endgame (d128): THD execute is CUDA-GRAPH CAPTURABLE — no
    D2H, no pageable H2D, plan-time envelope grid. Capture once, then
    replay with DIFFERENT lengths written into the same device tensors: the
    replay must honor them (per-sequence lengths are read on device by the
    setup and main kernels), proving no host value was baked into the
    graph."""
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, s, d = 2, 4, 256, 128
    dtype = torch.float16
    scale = 1.0 / math.sqrt(d)
    q, k, v = (_bhsd(b, h, s, d, dtype) for _ in range(3))
    o = torch.zeros_like(q)
    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, thd=True)
    assert api.check_support()
    api.compile()
    lens = torch.tensor([200, 150], dtype=torch.int32, device="cuda")

    def _check(seq_lens):
        base_q = q.transpose(1, 2).reshape(b * s, h, d)
        base_k = k.transpose(1, 2).reshape(b * s, h, d)
        base_v = v.transpose(1, 2).reshape(b * s, h, d)
        base_o = o.transpose(1, 2).reshape(b * s, h, d)
        off = 0
        for length in seq_lens:
            qs = base_q[off : off + length].float()
            ks = base_k[off : off + length].float()
            vs = base_v[off : off + length].float()
            scores = torch.einsum("lhd,mhd->hlm", qs, ks) * scale
            ref = torch.einsum("hlm,mhd->lhd", torch.softmax(scores, dim=-1), vs)
            torch.testing.assert_close(base_o[off : off + length].float(), ref, atol=5e-2, rtol=3e-2)
            off += length

    # Warm-up (compile artifacts, allocator, launcher state), then capture.
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=lens, seq_kv_lens=lens)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=lens, seq_kv_lens=lens)
    # Clobber O before each replay: the warm-up (and nothing else) has already
    # produced the [200, 150] answer, so without this the first assertion
    # would be satisfied by stale warm-up output even if replay did nothing.
    o.zero_()
    graph.replay()
    torch.cuda.synchronize()
    _check([200, 150])
    # New lengths into the SAME device tensor — replay must honor them.
    lens.copy_(torch.tensor([64, 33], dtype=torch.int32, device="cuda"))
    o.zero_()
    graph.replay()
    torch.cuda.synchronize()
    _check([64, 33])


@pytest.mark.L1
@torch_fork_set_rng(seed=40)
def test_dsl_sm100_thd_kv_zero_capacity_clamp():
    """All-zero KV lengths, both storage shapes (issue #552): with LIVE K/V
    storage the launch binds the buffers' capacity and the kernel's
    per-sequence dead-row path zeroes O; with ZERO-numel K/V buffers the
    packed-KV clamp binds the one never-dereferenced dummy token instead (a
    zero-token view cannot back a TMA descriptor). Both must produce O == 0
    on every live row."""
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, s, d = 2, 4, 128, 128
    dtype = torch.float16
    q = _bhsd(b, h, s, d, dtype)
    k, v = _bhsd(b, h, s, d, dtype), _bhsd(b, h, s, d, dtype)
    o = torch.full_like(q, 7.0)
    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, thd=True)
    assert api.check_support()
    api.compile()
    lens_q = torch.tensor([100, 80], dtype=torch.int32, device="cuda")
    lens_kv = torch.zeros(2, dtype=torch.int32, device="cuda")

    def _run(k_buf, v_buf):
        o.fill_(7.0)
        api.execute(q_tensor=q, k_tensor=k_buf, v_tensor=v_buf, o_tensor=o, seq_q_lens=lens_q, seq_kv_lens=lens_kv)
        torch.cuda.synchronize()
        return o.transpose(1, 2).reshape(b * s, h, d)[: 100 + 80].clone()

    live_capacity = _run(k, v)
    zero_capacity = _run(torch.empty(0, dtype=dtype, device="cuda"), torch.empty(0, dtype=dtype, device="cuda"))
    assert (live_capacity == 0).all(), "dead rows must be zeroed by the kernel's dead-row path"
    assert torch.equal(live_capacity, zero_capacity)


_COMBO_MASKS = {
    "dense": ["none", "causal", "causal_br", "swa", "padded", "band", "band_br", "band_swa", "swa_br"],
    # THD forces padding internally, so its mask axis rides on top of that.
    # causal_br: the kernels anchor the BR diagonal at each sequence's own
    # (seq_len_q[b], seq_len_kv[b]) from the cu_seqlen metadata.
    # band/band_br: diagonal-band right-bound widening (BAND_RIGHT).
    "thd": ["none", "causal", "swa", "causal_br", "band", "band_br", "swa_br"],
}


def _combo_cases():
    import itertools

    cases, ids = [], []
    for flavor, dtype, heads, sink, layout in itertools.product(_FLAVORS, _DTYPES, ["mha", "gqa"], [False, True], ["dense", "thd"]):
        for mask in _COMBO_MASKS[layout]:
            cases.append((flavor, dtype, heads, sink, layout, mask))
            ids.append(
                "-".join(
                    [
                        {512: "dsv4", 256: "qwen", 128: "llama"}[flavor],
                        "fp16" if dtype == torch.float16 else "bf16",
                        heads,
                        "sink" if sink else "nosink",
                        layout,
                        mask,
                    ]
                )
            )
    return cases, ids


_COMBO_CASES, _COMBO_IDS = _combo_cases()


@pytest.mark.L0
@pytest.mark.parametrize("flavor,dtype,heads,sink,layout,mask", _COMBO_CASES, ids=_COMBO_IDS)
@torch_fork_set_rng(seed=0)
def test_dsl_sm100_combo(flavor, dtype, heads, sink, layout, mask):
    """Full-cartesian feature-interaction sweep."""
    _require_dsl()
    d = flavor
    scale = 1.0 / math.sqrt(d)
    H_q, H_kv = (8, 8) if heads == "mha" else (8, 2)
    sink_t = torch.randn(1, H_q, 1, 1, dtype=torch.float32, device="cuda") if sink else None
    if layout == "dense":
        _combo_dense(d, dtype, H_q, H_kv, scale, sink_t, mask)
    else:
        _combo_thd(d, dtype, H_q, H_kv, scale, sink_t, mask)
