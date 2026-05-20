# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026, Jerry Chen
import math
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute

from cudnn.deepseek_sparse_attention.utils.runtime import (
    device_major as _get_device_capability,
    maybe_contiguous,
    resolve_stream,
)
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

from .dsa_bwd_sm90 import (
    FlashAttentionDSABackwardSm90,
    _FlashAttentionDSABackwardPostprocessSm90,
    _FlashAttentionDSABackwardPreprocessSm90,
)

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def flash_attn_bwd_sm90(
    q: torch.Tensor,
    kv: torch.Tensor,  # K=V unified tensor
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dkv: Optional[torch.Tensor] = None,
    d_sink: Optional[torch.Tensor] = None,
    dkv_accum: Optional[torch.Tensor] = None,
    topk_idxs: torch.Tensor = None,
    topk_length: torch.Tensor = None,
    need_d_sink: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """FlashAttention (DSA) Backward Pass for Hopper (SM90), with K=V.

    Accepts flat (unbatched) tensors with global topk indices.
    Internally wraps as batch=1 for the CuTe DSL kernel.

    Args:
        q: (total_S_q, nheads, headdim) bfloat16
        kv: (total_S_kv, headdim) bfloat16  (K=V, MQA h_kv=1)
        out: (total_S_q, nheads, headdim_v) bfloat16
        dout: (total_S_q, nheads, headdim_v) bfloat16
        lse: (total_S_q, nheads) float32
        attn_sink: (nheads,) float32, optional. When provided, the backward
            probabilities use sink-aware LSE.
        softmax_scale: float (default: 1/sqrt(headdim))
        dq: pre-allocated (total_S_q, nheads, headdim), optional
        dkv: pre-allocated (total_S_kv, headdim), optional
        d_sink: pre-allocated (nheads,), optional
        topk_idxs: (total_S_q, topk_max) int32, global indices
        topk_length: (total_S_q,) int32, per-query valid count, optional
        need_d_sink: return and compute d_sink when True

    Returns:
        (dq, dkv) or (dq, dkv, d_sink) — flat layout gradients
    """
    compute_capability = _get_device_capability()
    assert compute_capability == 9, f"Only SM90, got SM{compute_capability}0"

    total_S_q, num_head, head_dim = q.shape
    total_S_kv = kv.shape[0]
    head_dim_v = 512 if head_dim == 576 else head_dim
    num_head_kv = 1

    # --- wrap flat tensors as batch=1 4D for the CuTe DSL kernel ---
    q4 = q.unsqueeze(0)  # (1, total_S_q, H, D)
    kv4 = kv.unsqueeze(0).unsqueeze(2)  # (1, total_S_kv, 1, D)
    out4 = out.unsqueeze(0)  # (1, total_S_q, H, D_v)
    dout4 = dout.unsqueeze(0)  # (1, total_S_q, H, D_v)
    lse4 = lse.unsqueeze(0)  # (1, total_S_q, H)
    topk4 = topk_idxs.unsqueeze(0) if topk_idxs is not None else None  # (1, total_S_q, TopK)
    tlen4 = topk_length.unsqueeze(0) if topk_length is not None else None  # (1, total_S_q)

    m_block_size = 64
    n_block_size = 64
    KV_stage = 1
    PdS_stage = 1
    SdP_swapAB = False
    dKV_swapAB = False
    dQ_swapAB = False

    q4, kv4, out4, dout4, lse4 = [maybe_contiguous(t) for t in (q4, kv4, out4, dout4, lse4)]

    batch_size = 1
    seqlen_q = total_S_q
    seqlen_k = total_S_kv

    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size

    assert q4.dtype in [torch.float16, torch.bfloat16]
    assert q4.dtype == kv4.dtype == out4.dtype == dout4.dtype
    assert lse4.dtype == torch.float32
    assert all(t.is_cuda for t in (q4, kv4, out4, dout4, lse4))
    if attn_sink is not None:
        assert attn_sink.dtype == torch.float32
        assert attn_sink.shape == (num_head,)
        assert attn_sink.is_cuda
    assert num_head > num_head_kv, "MLA/MQA requires num_head > num_head_kv"
    assert head_dim in [512, 576]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv

    device = q.device

    # Allocate flat output buffers, then view as 4D for the kernel
    if dq is None:
        dq = torch.empty_like(q)  # (total_S_q, H, D)
    if dkv is None:
        dkv = torch.empty(total_S_kv, head_dim, dtype=kv.dtype, device=device)

    dq4 = dq.unsqueeze(0)  # (1, total_S_q, H, D)
    dkv4 = dkv.unsqueeze(0).unsqueeze(2)  # (1, total_S_kv, 1, D)

    sink_enabled = attn_sink is not None
    return_d_sink = need_d_sink or d_sink is not None
    if return_d_sink:
        assert attn_sink is not None, "attn_sink is required when requesting d_sink"
        if d_sink is None:
            d_sink = torch.zeros_like(attn_sink)
        else:
            d_sink.fill_(0)
    write_d_sink = sink_enabled and return_d_sink

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    dpsum = torch.empty(
        batch_size,
        seqlen_q_rounded,
        num_head,
        dtype=torch.float32,
        device=device,
    )
    lse_log2 = torch.empty(
        batch_size,
        seqlen_q_rounded,
        num_head,
        dtype=torch.float32,
        device=device,
    )

    if dkv_accum is None:
        dkv_accum = torch.zeros(
            batch_size,
            num_head_kv,
            seqlen_k_rounded * head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
    else:
        dkv_accum.fill_(0)

    dtype = torch2cute_dtype_map[q4.dtype]
    current_stream = resolve_stream()
    arch = 90
    num_threads = 256

    # --- preprocess ---
    compile_key_pre = (
        dtype,
        head_dim_v,
        m_block_size,
        num_threads,
        sink_enabled,
        write_d_sink,
    )
    if compile_key_pre not in flash_attn_bwd_sm90.compile_cache_pre:
        o_tensor, do_tensor = [to_cute_tensor(t) for t in (out4, dout4)]
        dpsum_tensor, lse_log2_tensor = [to_cute_tensor(t) for t in (dpsum, lse_log2)]
        lse_tensor = to_cute_tensor(lse4, assumed_align=4)
        attn_sink_tensor = to_cute_tensor(attn_sink) if sink_enabled else None
        d_sink_tensor = to_cute_tensor(d_sink) if write_d_sink else None
        fa_bwd_pre = _FlashAttentionDSABackwardPreprocessSm90(
            dtype,
            head_dim_v,
            arch,
            m_block_size,
            num_threads=num_threads,
        )
        flash_attn_bwd_sm90.compile_cache_pre[compile_key_pre] = cute.compile(
            fa_bwd_pre,
            o_tensor,
            do_tensor,
            dpsum_tensor,
            lse_tensor,
            lse_log2_tensor,
            attn_sink_tensor,
            d_sink_tensor,
            None,
            None,
            None,
            current_stream,
            options="--enable-tvm-ffi",
        )
    flash_attn_bwd_sm90.compile_cache_pre[compile_key_pre](
        out4,
        dout4,
        dpsum,
        lse4,
        lse_log2,
        attn_sink if sink_enabled else None,
        d_sink if write_d_sink else None,
        None,
        None,
        None,
        current_stream,
    )

    # --- main kernel ---
    assert topk4 is not None
    assert topk4.dtype == torch.int32
    have_topk_length = tlen4 is not None
    if have_topk_length:
        assert tlen4.dtype == torch.int32
    else:
        tlen4 = torch.empty(1, dtype=torch.int32, device=device)
    max_topk = topk4.shape[-1]

    num_threads = 256
    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        m_block_size,
        n_block_size,
        num_threads,
        KV_stage,
        PdS_stage,
        SdP_swapAB,
        dKV_swapAB,
        dQ_swapAB,
        have_topk_length,
        max_topk,
    )
    if compile_key not in flash_attn_bwd_sm90.compile_cache:
        q_tensor = to_cute_tensor(q4)
        kv_tensor = to_cute_tensor(kv4)
        do_tensor = to_cute_tensor(dout4)
        dpsum_tensor, lse_log2_tensor = [to_cute_tensor(t) for t in (dpsum, lse_log2)]
        dq_tensor = to_cute_tensor(dq4)
        dkv_accum_tensor = to_cute_tensor(dkv_accum)
        topk_idxs_tensor = to_cute_tensor(topk4)
        topk_length_tensor = to_cute_tensor(tlen4)

        fa_bwd_obj = FlashAttentionDSABackwardSm90(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            tile_m=m_block_size,
            tile_n=n_block_size,
            KV_stage=KV_stage,
            PdS_stage=PdS_stage,
            SdP_swapAB=SdP_swapAB,
            dKV_swapAB=dKV_swapAB,
            dQ_swapAB=dQ_swapAB,
            num_threads=num_threads,
            have_topk_length=have_topk_length,
            max_topk=max_topk,
        )
        flash_attn_bwd_sm90.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            kv_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_tensor,
            dkv_accum_tensor,
            topk_idxs_tensor,
            topk_length_tensor,
            softmax_scale,
            current_stream,
            options="--enable-tvm-ffi",
        )
    with torch.cuda.nvtx.range("flash_attn_bwd_sm90_kernel"):
        flash_attn_bwd_sm90.compile_cache[compile_key](
            q4,
            kv4,
            dout4,
            lse_log2,
            dpsum,
            dq4,
            dkv_accum,
            topk4,
            tlen4,
            softmax_scale,
            current_stream,
        )

    # --- postprocess: fake-col f32 dKVAccum -> real-col bf16 dKV ---
    hdim_chunk = 64 if head_dim == 576 else min(128, head_dim)
    N_hdim_chunks = head_dim // hdim_chunk
    num_threads_post = hdim_chunk

    compile_key_post = (
        dtype,
        hdim_chunk,
        n_block_size,
        head_dim,
        num_threads_post,
        N_hdim_chunks,
    )
    if compile_key_post not in flash_attn_bwd_sm90.compile_cache_post:
        dkv_accum_tensor = to_cute_tensor(dkv_accum)
        dkv_tensor = to_cute_tensor(dkv4)

        postprocess = _FlashAttentionDSABackwardPostprocessSm90(
            dtype,
            hdim_chunk=hdim_chunk,
            tile_n=n_block_size,
            head_dim=head_dim,
            num_threads=num_threads_post,
            N_hdim_chunks=N_hdim_chunks,
        )
        flash_attn_bwd_sm90.compile_cache_post[compile_key_post] = cute.compile(
            postprocess,
            dkv_accum_tensor,
            dkv_tensor,
            seqlen_k,
            current_stream,
            options="--enable-tvm-ffi",
        )
    flash_attn_bwd_sm90.compile_cache_post[compile_key_post](
        dkv_accum,
        dkv4,
        seqlen_k,
        current_stream,
    )

    # dq / dkv are already the flat tensors (unsqueeze was a view)
    if return_d_sink:
        return dq, dkv, d_sink
    return dq, dkv


flash_attn_bwd_sm90.compile_cache_pre = {}
flash_attn_bwd_sm90.compile_cache = {}
flash_attn_bwd_sm90.compile_cache_post = {}
