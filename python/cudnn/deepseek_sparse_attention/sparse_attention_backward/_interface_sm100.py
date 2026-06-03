# Copyright (c) 2026, Jerry Chen
import math
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute

from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor
from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def flash_attn_bwd_sm100(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    dq: Optional[torch.Tensor] = None,
    dkv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention (DSA) Backward Pass for Blackwell (SM100), with K=V.

    Accepts flat (unbatched) tensors with global topk indices.
    Internally wraps as batch=1 for the CuTe DSL kernel.

    Args:
        q: (total_S_q, nheads, headdim) bfloat16
        kv: (total_S_kv, headdim) bfloat16  (K=V, MQA h_kv=1)
        out: (total_S_q, nheads, headdim_v) bfloat16
        dout: (total_S_q, nheads, headdim_v) bfloat16
        lse: (total_S_q, nheads) float32, FlashMLA KV-only LSE excluding sink
        attn_sink: (nheads,) float32
        topk_idxs: (total_S_q, topk_max) int32, global indices
        softmax_scale: float (default: 1/sqrt(headdim))
        topk_length: (total_S_q,) int32, per-query valid count, optional
        dq: pre-allocated (total_S_q, nheads, headdim), optional
        dkv: pre-allocated (total_S_kv, headdim), optional

    Returns:
        (dq, dkv, d_sink) -- flat layout gradients
    """
    total_S_q, num_head, head_dim = q.shape
    total_S_kv = kv.shape[0]
    head_dim_v = 512 if head_dim == 576 else head_dim
    device = q.device

    assert q.dtype in [torch.float16, torch.bfloat16]
    assert q.dtype == kv.dtype == out.dtype == dout.dtype
    assert lse.dtype == torch.float32
    assert attn_sink.dtype == torch.float32
    assert topk_idxs.dtype == torch.int32
    tensors_to_check = [q, kv, out, dout, lse, attn_sink, topk_idxs]
    if topk_length is not None:
        tensors_to_check.append(topk_length)
    assert all(t.is_cuda for t in tensors_to_check)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    block_tile = 64
    num_head_blocks = (num_head + block_tile - 1) // block_tile
    batch_size = 1

    # Ensure contiguous
    q, kv, out, dout = [t.contiguous() for t in (q, kv, out, dout)]
    lse = lse.contiguous()

    # Allocate output tensors
    if dq is None:
        dq = torch.empty_like(q)
    else:
        assert dq.shape == q.shape, f"dq shape mismatch: expected {q.shape}, got {dq.shape}"
        assert dq.dtype == q.dtype, f"dq dtype mismatch: expected {q.dtype}, got {dq.dtype}"
        assert dq.device == device, f"dq device mismatch: expected {device}, got {dq.device}"
    if dkv is None:
        dkv = torch.zeros(total_S_kv, head_dim, dtype=kv.dtype, device=device)
    else:
        expected_dkv_shape = (total_S_kv, head_dim)
        assert dkv.shape == expected_dkv_shape, f"dkv shape mismatch: expected {expected_dkv_shape}, got {dkv.shape}"
        assert dkv.dtype == kv.dtype, f"dkv dtype mismatch: expected {kv.dtype}, got {dkv.dtype}"
        assert dkv.device == device, f"dkv device mismatch: expected {device}, got {dkv.device}"
        dkv.fill_(0)
    d_sink = torch.zeros_like(attn_sink)

    # Allocate workspace tensors
    acc_dtype = cutlass.Float32
    ws_lse_odo_shape = FlashAttentionDSABackwardSm100._get_workspace_size_LSE_OdO(
        total_S_q,
        head_dim,
        num_head,
        batch_size,
        acc_dtype,
    )
    workspace_LSE_OdO = torch.zeros(
        *ws_lse_odo_shape,
        dtype=torch.uint8,
        device=device,
    )

    ws_dkv_shape = FlashAttentionDSABackwardSm100._get_workspace_size_dKV(
        total_S_kv,
        head_dim,
        batch_size,
        acc_dtype,
    )
    workspace_dKV = torch.zeros(
        *ws_dkv_shape,
        dtype=torch.uint8,
        device=device,
    )

    problem_shape = (total_S_q, total_S_kv, head_dim, (num_head, batch_size))

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = resolve_stream()

    has_topk_length = topk_length is not None
    compile_key = (dtype, head_dim, head_dim_v, block_tile, has_topk_length, num_head)

    if compile_key not in flash_attn_bwd_sm100.compile_cache:
        q_tensor = to_cute_tensor(q, divisibility=head_dim)
        kv_tensor = to_cute_tensor(kv, divisibility=head_dim)
        out_tensor = to_cute_tensor(out, divisibility=head_dim_v)
        dout_tensor = to_cute_tensor(dout, divisibility=head_dim_v)
        lse_tensor = to_cute_tensor(lse, assumed_align=4)
        attn_sink_tensor = to_cute_tensor(attn_sink)
        topk_idxs_tensor = to_cute_tensor(topk_idxs)
        topk_length_tensor = to_cute_tensor(topk_length) if has_topk_length else None
        dq_tensor = to_cute_tensor(dq, divisibility=head_dim)
        dkv_tensor = to_cute_tensor(dkv, divisibility=head_dim)
        d_sink_tensor = to_cute_tensor(d_sink)
        workspace_LSE_OdO_tensor = to_cute_tensor(workspace_LSE_OdO)
        workspace_dKV_tensor = to_cute_tensor(workspace_dKV)

        kernel_obj = FlashAttentionDSABackwardSm100(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            block_tile=block_tile,
        )

        with torch.cuda.nvtx.range("flash_attn_bwd_sm100_compile"):
            flash_attn_bwd_sm100.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                problem_shape,
                q_tensor,
                kv_tensor,
                out_tensor,
                dout_tensor,
                lse_tensor,
                attn_sink_tensor,
                topk_idxs_tensor,
                topk_length_tensor,
                dq_tensor,
                dkv_tensor,
                d_sink_tensor,
                workspace_LSE_OdO_tensor,
                workspace_dKV_tensor,
                softmax_scale,
                current_stream,
                options="--enable-tvm-ffi",
            )

    with torch.cuda.nvtx.range("flash_attn_bwd_sm100_kernel"):
        flash_attn_bwd_sm100.compile_cache[compile_key](
            problem_shape,
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            topk_length,
            dq,
            dkv,
            d_sink,
            workspace_LSE_OdO,
            workspace_dKV,
            softmax_scale,
            current_stream,
        )

    return dq, dkv, d_sink


flash_attn_bwd_sm100.compile_cache = {}
