# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single attention inference benchmark.

Runs one (phase, backend, shape) case and prints a parseable result line.
Called as a subprocess by runner.py so each case gets a clean CUDA context and
failures stay independent.

Phases:
  context     s_q == s_kv prefill, contiguous Q/K/V, causal by default.
  generation  q_tokens new tokens attend to a kv_len cache (aligned at the
              end, i.e. bottom-right causal when q_tokens > 1).

Backends:
  cudnn              cuDNN frontend graph API (contiguous KV).
  cudnn_oss          Same graph, planned with heur_mode.OPENSOURCE so only the
                     frontend's open-source engines (frost python engines +
                     the backend's OSS candidates) may serve it.
  flashinfer         FlashInfer (paged KV for generation, ragged for context).
  b12x               b12x CuTe DSL kernels (SM12x only).
  flash_mla          FlashMLA (mla_absorbed generation only, SM90/SM100).
  flash_attention_4  FlashAttention-4 CuTe DSL (dense, with auto split-KV).

Timing uses CUDA events around the steady-state loop (median of iterations).
All setup (plans, page tables, workspace) happens outside the timed region —
this measures per-step kernel cost, not host planning.
"""

import argparse
import math
import os
import sys
from typing import Optional, Tuple

import torch

# GPU-name substring -> peak DRAM bandwidth in GB/s, for % SOL reporting.
# Override with CUDNN_BENCH_PEAK_BW_GBPS.
PEAK_BW_GBPS = {
    "GB300": 8000.0,
    "B300": 8000.0,
    "GB200": 8000.0,
    "B200": 8000.0,
    "GB100": 8000.0,
    "RTX PRO 6000": 1792.0,
    "RTX 5090": 1792.0,
    "H200": 4800.0,
    "H100 NVL": 3350.0,
    "H100": 3350.0,
    "A100": 1555.0,
    "L40S": 864.0,
}


def get_peak_bw_gbps() -> Optional[float]:
    env = os.environ.get("CUDNN_BENCH_PEAK_BW_GBPS")
    if env:
        return float(env)
    name = torch.cuda.get_device_name()
    for key, bw in PEAK_BW_GBPS.items():
        if key in name:
            return bw
    return None


def attention_flops(batch, q_tokens, kv_len, num_q_heads, d_qk, d_vo, causal_prefill) -> float:
    """2*B*H*Sq*Skv*(d_qk+d_vo), halved for square causal prefill."""
    f = 2.0 * batch * num_q_heads * q_tokens * kv_len * (d_qk + d_vo)
    if causal_prefill and q_tokens == kv_len:
        f *= 0.5
    return f


def attention_bytes(batch, q_tokens, kv_len, num_q_heads, num_kv_heads, d_qk, d_vo, kv_shared, sliding_window, kv_elt_size=2, qo_elt_size=2) -> float:
    """Algorithmic-minimum bytes: full KV read + Q read + O write."""
    kv_tokens_read = kv_len
    if sliding_window is not None:
        kv_tokens_read = min(kv_len, sliding_window + q_tokens)
    kv_width = max(d_qk, d_vo) if kv_shared else (d_qk + d_vo)
    kv_bytes = batch * kv_tokens_read * num_kv_heads * kv_width * kv_elt_size
    qo_bytes = batch * q_tokens * num_q_heads * (d_qk + d_vo) * qo_elt_size
    return kv_bytes + qo_bytes


def time_fn(fn, num_warmup, num_iters) -> float:
    """Median wall time of fn() in ms via CUDA events."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(num_iters):
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Backend setup functions. Each returns (fn, backend_detail) where fn() runs
# one attention step end to end with everything pre-planned.
# ---------------------------------------------------------------------------


def setup_cudnn(args, dtype, oss=False):
    import cudnn

    b, hq, hkv = args.batch_size, args.num_q_heads, args.num_kv_heads
    dqk, dvo = args.head_dim_qk, args.head_dim_vo
    sq, skv = args.q_tokens, args.kv_len

    if args.kv_cache_dtype == "fp8_e4m3":
        return setup_cudnn_fp8(args, cudnn, oss=oss)

    q_gpu = torch.randn(b, hq, sq, dqk, device="cuda", dtype=dtype) * 0.1
    if args.kind == "mla_absorbed":
        # Single shared record: K = full record, V = leading d_vo of the same
        # storage. hkv is expected to be 1 here.
        rec = torch.randn(b, hkv, skv, dqk, device="cuda", dtype=dtype) * 0.1
        k_gpu = rec
        v_gpu = rec[..., :dvo]
    else:
        k_gpu = torch.randn(b, hkv, skv, dqk, device="cuda", dtype=dtype) * 0.1
        v_gpu = torch.randn(b, hkv, skv, dvo, device="cuda", dtype=dtype) * 0.1
    o_gpu = torch.empty(b, hq, sq, dvo, device="cuda", dtype=dtype)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu)
    v = graph.tensor_like(v_gpu)

    sdpa_kwargs = dict(name="sdpa", q=q, k=k, v=v, generate_stats=False, attn_scale=args.sm_scale)
    sink_gpu = None
    if args.has_sink:
        sink_gpu = torch.randn(1, hq, 1, 1, device="cuda", dtype=torch.float32)
        sdpa_kwargs["sink_token"] = graph.tensor_like(sink_gpu)
    if args.phase == "context":
        if args.causal:
            sdpa_kwargs["diagonal_band_right_bound"] = 0
            # Chunked prefill (s_q < s_kv): the new chunk sits at the end of
            # the sequence, so the causal diagonal anchors bottom-right.
            sdpa_kwargs["diagonal_alignment"] = cudnn.diagonal_alignment.TOP_LEFT if sq == skv else cudnn.diagonal_alignment.BOTTOM_RIGHT
        if args.sliding_window_size:
            sdpa_kwargs["diagonal_band_left_bound"] = args.sliding_window_size
    else:
        # generation: q rows sit at the end of the kv sequence
        if sq > 1:
            sdpa_kwargs["diagonal_band_right_bound"] = 0
            sdpa_kwargs["diagonal_alignment"] = cudnn.diagonal_alignment.BOTTOM_RIGHT
        if args.sliding_window_size:
            sdpa_kwargs["diagonal_band_left_bound"] = args.sliding_window_size
            if sq == 1:
                sdpa_kwargs["diagonal_band_right_bound"] = 0
                sdpa_kwargs["diagonal_alignment"] = cudnn.diagonal_alignment.BOTTOM_RIGHT

    o, _ = graph.sdpa(**sdpa_kwargs)
    o.set_output(True).set_dim(o_gpu.size()).set_stride(o_gpu.stride())

    graph.validate()
    graph.build_operation_graph()
    modes = [cudnn.heur_mode.OPENSOURCE] if oss else [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
    graph.create_execution_plans(modes)
    graph.check_support()
    graph.build_plans()

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    variant_pack = {q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu}
    if sink_gpu is not None:
        variant_pack[sdpa_kwargs["sink_token"]] = sink_gpu
    handle = cudnn.create_handle()

    def fn():
        graph.execute(variant_pack, workspace, handle=handle)

    if oss:
        return fn, f"cudnn_oss plan={_require_python_oss_plan(graph)}"
    return fn, f"cudnn {cudnn.backend_version_string()}"


def _built_plan_name(graph) -> str:
    """Name of the plan build_plans() settled on ('backend_heuristics' = the
    backend's own OSS candidate space)."""
    try:
        return graph.get_plan_name_at_index(graph._plan_index)
    except Exception:
        return "unknown"


def _require_python_oss_plan(graph) -> str:
    """The built plan's name, rejecting the backend's delegating entry: the
    C++ OSS candidate engine only serves graphs carrying max/sum_exp stats
    outputs, which these inference graphs (generate_stats=False) don't have,
    so 'backend_heuristics' would die at execute with a confusing error."""
    name = _built_plan_name(graph)
    if name == "backend_heuristics":
        raise NotImplementedError("cudnn_oss: no python OSS engine serves this case")
    return name


def setup_cudnn_oss(args, dtype):
    """cuDNN frontend open-source engines only: frost python engines are
    opted in and the graph is planned with heur_mode.OPENSOURCE, so a native
    backend kernel can never answer for this backend."""
    os.environ["CUDNN_FRONTEND_ENABLE_FROST_ENGINES"] = "1"
    return setup_cudnn(args, dtype, oss=True)


def setup_cudnn_fp8(args, cudnn, oss=False):
    """FP8 attention graph (q/k/v e4m3 with unit descales) for the fp8-KV axis."""
    if args.has_sink:
        raise NotImplementedError("sink tokens are not wired into the fp8 attention graph")
    b, hq, hkv = args.batch_size, args.num_q_heads, args.num_kv_heads
    dqk, dvo = args.head_dim_qk, args.head_dim_vo
    sq, skv = args.q_tokens, args.kv_len
    f8 = torch.float8_e4m3fn

    q_gpu = (torch.randn(b, hq, sq, dqk, device="cuda") * 0.1).to(f8)
    if args.kind == "mla_absorbed":
        rec = (torch.randn(b, hkv, skv, dqk, device="cuda") * 0.1).to(f8)
        k_gpu, v_gpu = rec, rec[..., :dvo]
    else:
        k_gpu = (torch.randn(b, hkv, skv, dqk, device="cuda") * 0.1).to(f8)
        v_gpu = (torch.randn(b, hkv, skv, dvo, device="cuda") * 0.1).to(f8)
    o_gpu = torch.empty(b, hq, sq, dvo, device="cuda", dtype=f8)
    amax_s_gpu = torch.empty(1, 1, 1, 1, device="cuda", dtype=torch.float32)
    amax_o_gpu = torch.empty(1, 1, 1, 1, device="cuda", dtype=torch.float32)
    ones = {n: torch.ones(1, 1, 1, 1, device="cuda", dtype=torch.float32) for n in ("dq", "dk", "dv", "ds", "ss", "so")}

    graph = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v = graph.tensor_like(q_gpu), graph.tensor_like(k_gpu), graph.tensor_like(v_gpu)
    scalars = {n: graph.tensor_like(t) for n, t in ones.items()}
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        descale_q=scalars["dq"],
        descale_k=scalars["dk"],
        descale_v=scalars["dv"],
        descale_s=scalars["ds"],
        scale_s=scalars["ss"],
        scale_o=scalars["so"],
        is_inference=True,
        attn_scale=args.sm_scale,
        name="sdpa_fp8",
    )
    if args.phase == "generation" and sq > 1:
        kwargs["use_causal_mask_bottom_right"] = True
    elif args.phase == "context" and args.causal:
        if sq == skv:
            kwargs["use_causal_mask"] = True
        else:
            kwargs["use_causal_mask_bottom_right"] = True
    if args.sliding_window_size:
        kwargs["sliding_window"] = args.sliding_window_size
        if args.phase == "generation" and sq == 1:
            # anchor the window to the END of the cache, mirroring the bf16
            # path — top-left anchoring would window the wrong tokens
            kwargs["use_causal_mask_bottom_right"] = True
    ret = graph.sdpa_fp8(**kwargs)
    o, amax_s, amax_o = ret[0], ret[2], ret[3]
    o.set_output(True).set_dim(o_gpu.size()).set_stride(o_gpu.stride())
    amax_s.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    modes = [cudnn.heur_mode.OPENSOURCE] if oss else [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
    graph.create_execution_plans(modes)
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    variant_pack = {q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu, amax_s: amax_s_gpu, amax_o: amax_o_gpu}
    for n in ones:
        variant_pack[scalars[n]] = ones[n]
    handle = cudnn.create_handle()

    def fn():
        graph.execute(variant_pack, workspace, handle=handle)

    if oss:
        return fn, f"cudnn_oss fp8 plan={_require_python_oss_plan(graph)}"
    return fn, f"cudnn fp8 {cudnn.backend_version_string()}"


def _paged_kv_layout(batch, kv_len, page_size, device="cuda"):
    """Uniform full pages per sequence: returns (num_pages_total, indptr, indices, last_page_len)."""
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total = pages_per_seq * batch
    indptr = torch.arange(0, batch + 1, device=device, dtype=torch.int32) * pages_per_seq
    indices = torch.arange(0, total, device=device, dtype=torch.int32)
    last_len = kv_len - (pages_per_seq - 1) * page_size
    last_page_len = torch.full((batch,), last_len, device=device, dtype=torch.int32)
    return total, indptr, indices, last_page_len


def setup_flashinfer(args, dtype):
    import flashinfer

    b, hq, hkv = args.batch_size, args.num_q_heads, args.num_kv_heads
    dqk, dvo = args.head_dim_qk, args.head_dim_vo
    sq, skv = args.q_tokens, args.kv_len
    fi_backend = os.environ.get("FLASHINFER_BENCH_BACKEND", "auto")
    # Large batch x long-KV split-KV plans need a big float workspace.
    workspace = torch.empty(1024 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    if args.phase == "context":
        # Ragged (contiguous) batch prefill.
        q_gpu = torch.randn(b * sq, hq, dqk, device="cuda", dtype=dtype) * 0.1
        k_gpu = torch.randn(b * skv, hkv, dqk, device="cuda", dtype=dtype) * 0.1
        v_gpu = torch.randn(b * skv, hkv, dvo, device="cuda", dtype=dtype) * 0.1
        qo_indptr = torch.arange(0, b + 1, device="cuda", dtype=torch.int32) * sq
        kv_indptr = torch.arange(0, b + 1, device="cuda", dtype=torch.int32) * skv
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD", backend=fi_backend)
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            hq,
            hkv,
            dqk,
            head_dim_vo=dvo,
            causal=args.causal,
            window_left=args.sliding_window_size if args.sliding_window_size else -1,
            sm_scale=args.sm_scale,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        def fn():
            wrapper.run(q_gpu, k_gpu, v_gpu)

        return fn, f"flashinfer prefill-ragged backend={getattr(wrapper, '_backend', fi_backend)}"

    if args.kind == "mla_absorbed":
        # MLA paged decode: latent record split as ckv (d_vo) + rope/pe tail.
        d_ckv, d_kpe = dvo, dqk - dvo
        total, kv_indptr, kv_indices, _ = _paged_kv_layout(b, skv, args.page_size)
        q_nope = torch.randn(b * sq, hq, d_ckv, device="cuda", dtype=dtype) * 0.1
        q_pe = torch.randn(b * sq, hq, d_kpe, device="cuda", dtype=dtype) * 0.1
        kv_dtype = torch.float8_e4m3fn if args.kv_cache_dtype == "fp8_e4m3" else dtype
        ckv_cache = (torch.randn(total, args.page_size, d_ckv, device="cuda") * 0.1).to(kv_dtype)
        kpe_cache = (torch.randn(total, args.page_size, d_kpe, device="cuda") * 0.1).to(kv_dtype)
        qo_indptr = torch.arange(0, b + 1, device="cuda", dtype=torch.int32) * sq
        kv_len_arr = torch.full((b,), skv, device="cuda", dtype=torch.int32)
        mla_backend = os.environ.get("FLASHINFER_BENCH_MLA_BACKEND", fi_backend)
        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend=mla_backend)
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            hq,
            d_ckv,
            d_kpe,
            args.page_size,
            sq > 1,  # causal within the draft window
            args.sm_scale,
            dtype,
            kv_dtype,
        )

        def fn():
            wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache)

        return fn, f"flashinfer mla backend={getattr(wrapper, '_backend', mla_backend)}"

    kv_dtype = torch.float8_e4m3fn if args.kv_cache_dtype == "fp8_e4m3" else dtype
    total, kv_indptr, kv_indices, last_page_len = _paged_kv_layout(b, skv, args.page_size)
    kv_cache = (torch.randn(total, 2, args.page_size, hkv, dqk, device="cuda") * 0.1).to(kv_dtype)

    if args.has_sink:
        # Per-head sinks go through the trtllm decode entry point. NB: unlike
        # the wrappers it does NOT default the softmax scale — bmm1_scale
        # must be passed explicitly or results are silently wrong.
        q_gpu = torch.randn(b * sq, hq, dqk, device="cuda", dtype=dtype) * 0.1
        sinks = torch.randn(hq, device="cuda", dtype=torch.float32)
        # trtllm layout: (pages, 2, num_kv_heads, page_size, head_dim)
        kv_trt = (torch.randn(total, 2, hkv, args.page_size, dqk, device="cuda") * 0.1).to(kv_dtype)
        pages_per_seq = (skv + args.page_size - 1) // args.page_size
        block_tables = torch.arange(total, device="cuda", dtype=torch.int32).view(b, pages_per_seq)
        seq_lens = torch.full((b,), skv, device="cuda", dtype=torch.int32)
        wl = args.sliding_window_size - 1 if args.sliding_window_size else -1

        def fn():
            flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                q_gpu,
                kv_trt,
                workspace,
                block_tables,
                seq_lens,
                skv,
                bmm1_scale=args.sm_scale,
                sinks=sinks,
                window_left=wl,
                q_len_per_req=sq,
            )

        return fn, "flashinfer trtllm-decode sinks"

    if sq == 1:
        q_gpu = torch.randn(b, hq, dqk, device="cuda", dtype=dtype) * 0.1
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=True, backend=fi_backend)
        wrapper.plan(
            kv_indptr,
            kv_indices,
            last_page_len,
            hq,
            hkv,
            dqk,
            args.page_size,
            window_left=args.sliding_window_size if args.sliding_window_size else -1,
            sm_scale=args.sm_scale,
            q_data_type=dtype,
            kv_data_type=kv_dtype,
        )

        def fn():
            wrapper.run(q_gpu, kv_cache)

        return fn, f"flashinfer decode-paged backend={getattr(wrapper, '_backend', fi_backend)}"

    # q_tokens > 1 (MTP / chunked): paged prefill with bottom-right causal.
    q_gpu = torch.randn(b * sq, hq, dqk, device="cuda", dtype=dtype) * 0.1
    qo_indptr = torch.arange(0, b + 1, device="cuda", dtype=torch.int32) * sq
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend=fi_backend)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        hq,
        hkv,
        dqk,
        args.page_size,
        causal=True,
        window_left=args.sliding_window_size if args.sliding_window_size else -1,
        sm_scale=args.sm_scale,
        q_data_type=dtype,
        kv_data_type=kv_dtype,
    )

    def fn():
        wrapper.run(q_gpu, kv_cache)

    return fn, f"flashinfer prefill-paged backend={getattr(wrapper, '_backend', fi_backend)}"


def _b12x_work_items(batch, num_q_heads):
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    v = max(1024, 4 * sms, batch * num_q_heads)
    v = min(v, 16384)
    return 1 << (v - 1).bit_length()


def _b12x_paging(batch, kv_len, page_size):
    import torch

    pages_per_seq = (kv_len + page_size - 1) // page_size
    total = pages_per_seq * batch
    page_table = torch.arange(total, device="cuda", dtype=torch.int32).view(batch, pages_per_seq)
    cache_seqlens = torch.full((batch,), kv_len, device="cuda", dtype=torch.int32)
    return total, page_table, cache_seqlens


def setup_b12x(args, dtype):
    b, hq, hkv = args.batch_size, args.num_q_heads, args.num_kv_heads
    dqk, dvo = args.head_dim_qk, args.head_dim_vo
    sq, skv = args.q_tokens, args.kv_len

    if args.phase == "context" or (args.kind == "gqa" and args.sliding_window_size):
        # Contiguous batched attention: context phase, and the SWA-generation
        # fallback (the paged module's window knob is not wired here yet).
        from b12x.attention import varlen

        q_gpu = torch.randn(b, sq, hq, dqk, device="cuda", dtype=dtype) * 0.1
        k_gpu = torch.randn(b, skv, hkv, dqk, device="cuda", dtype=dtype) * 0.1
        v_gpu = torch.randn(b, skv, hkv, dvo, device="cuda", dtype=dtype) * 0.1
        causal = args.causal if args.phase == "context" else sq > 1
        window = (args.sliding_window_size, 0) if args.sliding_window_size else None
        sink = torch.randn(hq, device="cuda", dtype=torch.float32) if args.has_sink else None
        kplan = varlen.create_plan_batched(q_gpu, k_gpu, v_gpu, causal=causal, window_size=window, attention_sink_bias=sink)
        splan = varlen.plan_batched(kplan)
        spec = splan.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = splan.bind(scratch=scratch, q=q_gpu, k=k_gpu, v=v_gpu, softmax_scale=args.sm_scale, attention_sink_bias=sink)

        def fn():
            varlen.run_batched(binding=binding)

        return fn, "b12x varlen-batched"

    total_q = b * sq
    cu_seqlens_q = torch.arange(0, b + 1, device="cuda", dtype=torch.int32) * sq
    mode = "decode" if sq == 1 else "extend"

    if args.kind == "mla_absorbed" and args.head_dim_qk == 576:
        # Kimi K3 absorbed record (the module is K3-specific: head_dim=576).
        from b12x.attention import dense_mla

        if args.kv_cache_dtype != "bfloat16":
            raise NotImplementedError("fp8 record not wired for b12x dense_mla in this harness")

        pages, page_table, cache_seqlens = _b12x_paging(b, skv, args.page_size)
        q_gpu = torch.randn(total_q, hq, dqk, device="cuda", dtype=dtype) * 0.1
        cache = torch.randn(pages, args.page_size, dqk, device="cuda", dtype=dtype) * 0.1
        out = torch.empty(total_q, hq, dvo, device="cuda", dtype=dtype)
        plan = dense_mla.plan(
            dense_mla.Caps(
                device="cuda",
                mode=mode,
                dtype=dtype,
                kv_dtype=dtype,
                num_q_heads=hq,
                head_dim=dqk,
                v_head_dim=dvo,
                page_size=args.page_size,
                max_total_q=total_q,
                max_batch=b,
                max_cache_tokens=skv,
                max_page_table_width=page_table.shape[1],
                num_cache_pages=pages,
            )
        )
        spec = plan.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = dense_mla.bind(
            plan,
            scratch=scratch,
            q=q_gpu,
            kv_cache=cache,
            output=out,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        )

        def fn():
            dense_mla.run(binding=binding)

        return fn, f"b12x dense-mla ({mode})"

    # GQA generation: paged decode/extend (the serving path). Shared-K=V MQA
    # (DeepSeek-V4) also lands here, with the V cache aliased to the K cache.
    from b12x.attention import paged

    shared_kv = args.kind == "mla_absorbed"
    kv_dtype = torch.float8_e4m3fn if args.kv_cache_dtype == "fp8_e4m3" else dtype
    # b12x paged kernels require page_size >= 64; the page size is an internal
    # layout knob (KV bytes are unchanged), so floor it and record.
    b12x_page = max(args.page_size, 64)
    pages, page_table, cache_seqlens = _b12x_paging(b, skv, b12x_page)
    q_gpu = torch.randn(total_q, hq, dqk, device="cuda", dtype=dtype) * 0.1
    k_cache = (torch.randn(pages, b12x_page, hkv, dqk, device="cuda") * 0.1).to(kv_dtype)
    v_cache = k_cache[..., :dvo] if shared_kv else (torch.randn(pages, b12x_page, hkv, dvo, device="cuda") * 0.1).to(kv_dtype)
    out = torch.empty(total_q, hq, dvo, device="cuda", dtype=dtype)
    plan = paged.plan(
        paged.Caps(
            device="cuda",
            mode=mode,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=hq,
            num_kv_heads=hkv,
            head_dim_qk=dqk,
            head_dim_vo=dvo,
            page_size=b12x_page,
            max_total_q=total_q,
            max_batch=b,
            max_page_table_width=page_table.shape[1],
            # The scratch arena is sized from these caps and oversizing costs
            # per-run time; b12x production uses a tuned device-LUT policy —
            # this is a simple shape-derived stand-in (see README).
            max_work_items=_b12x_work_items(b, hq),
            max_partial_rows=_b12x_work_items(b, hq),
            num_cache_pages=pages,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    bind_kwargs = dict(
        scratch=scratch,
        q=q_gpu,
        k_cache=k_cache,
        v_cache=v_cache,
        output=out,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    if args.has_sink:
        bind_kwargs["attention_sink_bias"] = torch.randn(hq, device="cuda", dtype=torch.float32)
    if kv_dtype != dtype:
        bind_kwargs["k_descale"] = torch.ones(1, device="cuda", dtype=torch.float32)
        bind_kwargs["v_descale"] = torch.ones(1, device="cuda", dtype=torch.float32)
    binding = paged.bind(plan, **bind_kwargs)

    def fn():
        paged.run(binding=binding)

    return fn, f"b12x paged ({mode}{', shared-kv' if shared_kv else ''}, page={b12x_page})"


def setup_flash_mla(args, dtype):
    import flash_mla

    if args.kv_cache_dtype != "bfloat16":
        raise NotImplementedError("fp8 KV not wired for flash_mla in this harness")

    b, hq = args.batch_size, args.num_q_heads
    dqk, dvo = args.head_dim_qk, args.head_dim_vo
    sq, skv = args.q_tokens, args.kv_len
    page = 64  # FlashMLA block size

    pages_per_seq = (skv + page - 1) // page
    q_gpu = torch.randn(b, sq, hq, dqk, device="cuda", dtype=dtype) * 0.1
    kv_cache = torch.randn(b * pages_per_seq, page, 1, dqk, device="cuda", dtype=dtype) * 0.1
    block_table = torch.arange(b * pages_per_seq, device="cuda", dtype=torch.int32).view(b, pages_per_seq)
    cache_seqlens = torch.full((b,), skv, device="cuda", dtype=torch.int32)

    tile_md, num_splits = flash_mla.get_mla_metadata(cache_seqlens, sq * hq // 1, 1)

    def fn():
        flash_mla.flash_mla_with_kvcache(
            q_gpu,
            kv_cache,
            block_table,
            cache_seqlens,
            dvo,
            tile_md,
            num_splits,
            softmax_scale=args.sm_scale,
            causal=sq > 1,
        )

    return fn, "flash_mla"


def setup_flash_attention_4(args, dtype):
    import flash_attn.cute.interface as fa4

    if args.kv_cache_dtype != "bfloat16" or args.has_sink:
        raise NotImplementedError("fp8 KV / sinks not wired for fa4 in this harness")

    b, hq, hkv = args.batch_size, args.num_q_heads, args.num_kv_heads
    dqk, dvo = args.head_dim_qk, args.head_dim_vo
    sq, skv = args.q_tokens, args.kv_len

    q_gpu = torch.randn(b, sq, hq, dqk, device="cuda", dtype=dtype) * 0.1
    k_gpu = torch.randn(b, skv, hkv, dqk, device="cuda", dtype=dtype) * 0.1
    v_gpu = torch.randn(b, skv, hkv, dvo, device="cuda", dtype=dtype) * 0.1
    causal = args.causal if args.phase == "context" else sq > 1
    window = (args.sliding_window_size, 0) if args.sliding_window_size else (-1, -1)

    detail = "fa4"
    kwargs = dict(causal=causal, window_size=window, softmax_scale=args.sm_scale)
    if args.phase == "generation":
        if args.fa4_num_splits == "auto":
            # 0 = let FA4's split-KV heuristic decide.
            try:
                fa4.flash_attn_func(q_gpu, k_gpu, v_gpu, num_splits=0, **kwargs)
                kwargs["num_splits"] = 0
                detail = "fa4 num_splits=auto"
            except Exception:
                # Fall back to a one-time sweep, keep the best.
                best = (None, float("inf"))
                for ns in (1, 2, 4, 8, 16, 32):
                    try:
                        t = time_fn(lambda: fa4.flash_attn_func(q_gpu, k_gpu, v_gpu, num_splits=ns, **kwargs), 2, 5)
                        if t < best[1]:
                            best = (ns, t)
                    except Exception:
                        continue
                kwargs["num_splits"] = best[0] if best[0] else 1
                detail = f"fa4 num_splits={kwargs['num_splits']} (swept)"
        elif args.fa4_num_splits is not None:
            kwargs["num_splits"] = int(args.fa4_num_splits)
            detail = f"fa4 num_splits={kwargs['num_splits']}"

    def fn():
        fa4.flash_attn_func(q_gpu, k_gpu, v_gpu, **kwargs)

    return fn, detail


SETUPS = {
    "cudnn": setup_cudnn,
    "cudnn_oss": setup_cudnn_oss,
    "flashinfer": setup_flashinfer,
    "b12x": setup_b12x,
    "flash_mla": setup_flash_mla,
    "flash_attention_4": setup_flash_attention_4,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=["context", "generation"])
    p.add_argument("--backend", required=True, choices=sorted(SETUPS))
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--q_tokens", type=int, required=True)
    p.add_argument("--kv_len", type=int, required=True)
    p.add_argument("--num_q_heads", type=int, required=True)
    p.add_argument("--num_kv_heads", type=int, required=True)
    p.add_argument("--head_dim_qk", type=int, required=True)
    p.add_argument("--head_dim_vo", type=int, required=True)
    p.add_argument("--kind", choices=["gqa", "mla_absorbed"], default="gqa")
    p.add_argument("--causal", action="store_true", help="causal mask for the context phase")
    p.add_argument("--sliding_window_size", type=int, default=None)
    p.add_argument("--sm_scale", type=float, default=None)
    p.add_argument("--page_size", type=int, default=64)
    p.add_argument("--data_type", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--kv_cache_dtype", choices=["bfloat16", "fp8_e4m3"], default="bfloat16")
    p.add_argument("--has_sink", action="store_true", help="per-head attention sinks")
    p.add_argument("--num_iterations", type=int, default=20)
    p.add_argument("--num_warmup_iterations", type=int, default=5)
    p.add_argument("--fa4_num_splits", default="auto", help="'auto', or an integer for a fixed split count")
    args = p.parse_args()

    if args.sm_scale is None:
        args.sm_scale = 1.0 / math.sqrt(args.head_dim_qk)
    dtype = torch.bfloat16 if args.data_type == "bfloat16" else torch.float16

    fn, detail = SETUPS[args.backend](args, dtype)
    ms = time_fn(fn, args.num_warmup_iterations, args.num_iterations)

    flops = attention_flops(
        args.batch_size,
        args.q_tokens,
        args.kv_len,
        args.num_q_heads,
        args.head_dim_qk,
        args.head_dim_vo,
        causal_prefill=args.causal and args.phase == "context",
    )
    byts = attention_bytes(
        args.batch_size,
        args.q_tokens,
        args.kv_len,
        args.num_q_heads,
        args.num_kv_heads,
        args.head_dim_qk,
        args.head_dim_vo,
        kv_shared=args.kind == "mla_absorbed",
        sliding_window=args.sliding_window_size,
        kv_elt_size=1 if args.kv_cache_dtype == "fp8_e4m3" else 2,
        # the cudnn fp8 graph holds Q and O in e4m3 as well
        qo_elt_size=1 if args.kv_cache_dtype == "fp8_e4m3" and args.backend in ("cudnn", "cudnn_oss") else 2,
    )
    tflops = flops / (ms * 1e-3) / 1e12
    gbps = byts / (ms * 1e-3) / 1e9
    peak = get_peak_bw_gbps()
    sol = f"{gbps / peak * 100.0:.4f}" if peak else ""

    print(f"RESULT,{ms:.6f},{tflops:.3f},{gbps:.3f},{sol}," f"{torch.cuda.get_device_name().replace(',', ';')},{detail.replace(',', ';')}")


if __name__ == "__main__":
    main()
