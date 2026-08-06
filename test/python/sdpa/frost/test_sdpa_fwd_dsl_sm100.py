# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the FROST SM100 DSL SDPA-forward engines against a torch reference."""

import math
import os

import pytest
import torch

from test_utils import torch_fork_set_rng

from cudnn.sdpa.fwd.engines import engine_name


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return (major, minor) == (10, 0)


def _dsl_deps_available() -> bool:
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


def _select_engine(graph, name):
    """Pin the ranked entry named ``name`` (graph.plans holds the backend's
    plans and the python engines' in one list). A pin is strict: check_support /
    build_plans raise if that engine declines, so an ineligible config cannot
    silently fall back to native cuDNN."""
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    assert name in names, f"engine {name!r} did not claim this graph; plans={names}"
    graph.select_plan(names.index(name))
    return graph


pytestmark = pytest.mark.skipif(
    not _is_sm100(),
    reason="SM100 DSL SDPA engine requires an SM100 (Blackwell) device.",
)


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
    if not _dsl_deps_available():
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
    _select_engine(graph, engine_name(d))
    graph.check_support()
    graph.build_plans()
    # Honest workspace: no Stats output, so the engine carves a dummy LSE
    # (b*h*s fp32) from the caller's buffer.
    assert graph.get_workspace_size() == b * h * s * 4

    workspace = torch.empty(graph.get_workspace_size(), device=device, dtype=torch.uint8)
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


def _ref_sdpa_full(q, k, v, *, scale, is_causal=False, bottom_right=False, swa_window=None, seq_kv_lens=None, sinks=None):
    """fp32 reference matching the SM100 DSL kernel's mask + sink semantics.
    q/k/v are BHSD; GQA (h_q > h_kv) is handled by expanding K/V."""
    b, h_q, s_q, _ = q.shape
    _, h_kv, s_kv, _ = v.shape
    dev = q.device
    g = h_q // h_kv
    k_ref = k.repeat_interleave(g, dim=1).float()
    v_ref = v.repeat_interleave(g, dim=1).float()
    scores = torch.matmul(q.float(), k_ref.transpose(-1, -2)) * scale

    i = torch.arange(s_q, device=dev).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=dev).view(1, 1, 1, s_kv)
    masked = torch.zeros(b, 1, s_q, s_kv, dtype=torch.bool, device=dev)
    if is_causal:
        lim = i + (s_kv - s_q) if bottom_right else i
        masked = masked | (j > lim)
    if swa_window is not None:
        masked = masked | (j < i - swa_window)
    if seq_kv_lens is not None:
        lens = seq_kv_lens.view(b, 1, 1, 1).to(dev)
        masked = masked | (j >= lens)
    scores = scores.masked_fill(masked, float("-inf"))

    if sinks is not None:
        sink_col = sinks.view(1, h_q, 1, 1).float().expand(b, h_q, s_q, 1).to(dev)
        probs = torch.softmax(torch.cat([scores, sink_col], dim=-1), dim=-1)
        o = torch.matmul(probs[..., :s_kv], v_ref)
    else:
        o = torch.matmul(torch.softmax(scores, dim=-1), v_ref)
    return o.to(q.dtype)


def _require_dsl():
    try:
        import cudnn  # noqa: F401
        import cudnn.sdpa  # noqa: F401 — registers the FROST DSL engines
    except ImportError as e:
        pytest.skip(f"SM100 DSL engine not available: {e}")
    if not _dsl_deps_available():
        pytest.skip("cutlass/dsl not installed")


def _bhsd(b, h, s, d, dtype, device="cuda"):
    """Random BHSD tensor with BSHD-physical strides (notebook convention)."""
    return torch.randn(b, s, h, d, device=device, dtype=dtype).transpose(1, 2)


def _run_dsl_graph(q_gpu, k_gpu, v_gpu, *, scale, dtype, sdpa_kwargs, seq_len_kv=None, sink=None):
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
    kw = dict(name="sdpa", q=q, k=k, v=v, generate_stats=False, attn_scale=scale)
    vp = {q: q_gpu, k: k_gpu, v: v_gpu}
    if seq_len_kv is not None:
        slk = g.tensor_like(seq_len_kv)
        kw["seq_len_kv"] = slk
        kw["use_padding_mask"] = True
        vp[slk] = seq_len_kv
        # padding_mask requires a seq_len_q companion; the kernel trims only KV, so
        # Q is full and seq_len_q is accepted but unused.
        seq_len_q = torch.full((b, 1, 1, 1), s_q, dtype=torch.int32, device="cuda")
        slq = g.tensor_like(seq_len_q)
        kw["seq_len_q"] = slq
        vp[slq] = seq_len_q
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    kw.update(sdpa_kwargs)
    o, _ = g.sdpa(**kw)
    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(q_gpu.shape[-1], d_v=d_v))
    g.check_support()
    g.build_plans()
    vp[o] = o_gpu
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8))
    torch.cuda.synchronize()
    return o_gpu


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
    _select_engine(g, engine_name(d))
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
    _select_engine(g, engine_name(d))
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


def _mask_graph_kwargs(mask):
    """graph.sdpa kwargs for the causal-family masks (padded handled separately)."""
    return {
        "none": {},
        "causal": dict(use_causal_mask=True),
        "causal_br": dict(use_causal_mask_bottom_right=True),
        "swa": dict(use_causal_mask=True, sliding_window_length=_COMBO_SWA_W + 1),
    }[mask]


def _mask_ref_kwargs(mask):
    """_ref_sdpa_full kwargs matching _mask_graph_kwargs."""
    return {
        "none": {},
        "causal": dict(is_causal=True),
        "causal_br": dict(is_causal=True, bottom_right=True),
        "swa": dict(is_causal=True, swa_window=_COMBO_SWA_W),
    }[mask]


def _run_dsl_thd_graph(q_pk, k_pk, v_pk, cu_q, cu_k, seq_lens_q, seq_lens_kv, *, scale, dtype, H_q, H_kv, d, sink=None, mask="causal"):
    """Build + execute a packed THD/varlen graph; returns the flat packed O storage buffer."""
    import cudnn

    dev = "cuda"
    B = len(seq_lens_q)
    T_q, T_kv = cu_q[-1], cu_k[-1]
    S_max_q, S_max_kv = max(seq_lens_q), max(seq_lens_kv)

    def _dense_buf(packed, s_max, t, H):
        stride = (s_max * H * d, d, H * d, 1)
        stor = torch.zeros(B * s_max * H * d, device=dev, dtype=dtype)
        stor[: t * H * d] = packed.reshape(-1)
        return stor, stor.as_strided((B, H, s_max, d), stride), stride

    _, q_gpu, stride_q = _dense_buf(q_pk, S_max_q, T_q, H_q)
    _, k_gpu, stride_kv = _dense_buf(k_pk, S_max_kv, T_kv, H_kv)
    _, v_gpu, _ = _dense_buf(v_pk, S_max_kv, T_kv, H_kv)
    o_stor = torch.zeros(B * S_max_q * H_q * d, device=dev, dtype=dtype)
    o_gpu = o_stor.as_strided((B, H_q, S_max_q, d), stride_q)

    slq = torch.tensor(seq_lens_q, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(seq_lens_kv, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    ro_q = (torch.tensor(cu_q, dtype=torch.int64, device=dev) * H_q * d).view(B + 1, 1, 1, 1)
    ro_k = (torch.tensor(cu_k, dtype=torch.int64, device=dev) * H_kv * d).view(B + 1, 1, 1, 1)

    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H_q, S_max_q, d], stride=list(stride_q), data_type=io, name="q")
    tk = g.tensor(dim=[B, H_kv, S_max_kv, d], stride=list(stride_kv), data_type=io, name="k")
    tv = g.tensor(dim=[B, H_kv, S_max_kv, d], stride=list(stride_kv), data_type=io, name="v")
    sq = g.tensor_like(slq)
    skv = g.tensor_like(slk)
    qro = g.tensor_like(ro_q)
    kro = g.tensor_like(ro_k)
    vro = g.tensor_like(ro_k)
    oro = g.tensor_like(ro_q)
    tq.set_ragged_offset(qro)
    tk.set_ragged_offset(kro)
    tv.set_ragged_offset(vro)
    kw = dict(name="sdpa", q=tq, k=tk, v=tv, generate_stats=False, attn_scale=scale, use_padding_mask=True, seq_len_q=sq, seq_len_kv=skv)
    kw.update(_mask_graph_kwargs(mask))
    vp = {tq: q_gpu, tk: k_gpu, tv: v_gpu, sq: slq, skv: slk, qro: ro_q, kro: ro_k, vro: ro_k, oro: ro_q}
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    o, _ = g.sdpa(**kw)
    o.set_output(True).set_dim([B, H_q, S_max_q, d]).set_stride(list(stride_q))
    o.set_ragged_offset(oro)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(d))
    g.check_support()
    g.build_plans()
    vp[o] = o_gpu
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()
    return o_stor


def _combo_dense(d, dtype, H_q, H_kv, scale, sink_t, mask):
    b = 2
    s_q, s_kv = (128, 256) if mask == "causal_br" else (256, 256)
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


_COMBO_MASKS = {
    "dense": ["none", "causal", "causal_br", "swa", "padded"],
    # THD forces padding internally; bottom-right causal is a kernel gap (BR
    # diagonal needs global, not per-sequence, Q length), so it is excluded.
    "thd": ["none", "causal", "swa"],
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
