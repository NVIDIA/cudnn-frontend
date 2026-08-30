# Copyright (c) 2026, Jerry Chen
# SPDX-License-Identifier: MIT
import math
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream, torch_stream_context
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor
from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _select_sm100_backend(
    num_heads: int,
    head_dim: int,
    *,
    head_dim_v: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    max_topk: int = 0,
    device_capability: Optional[Tuple[int, int]] = None,
) -> Tuple[str, int]:
    """Return the tuned SM100 kernel variant and its sparse-row tile size.

    The two-CTA specialization is deliberately fail-closed.  It is selected
    only for the exact B200/BF16/H128/D512 envelope validated by its kernel;
    every other supported configuration retains the established backend.
    """
    if (
        device_capability == (10, 0)
        and dtype == torch.bfloat16
        and num_heads == 128
        and head_dim == 512
        and head_dim_v == 512
        and max_topk in (128, 512, 1024, 2048)
    ):
        return "h128_2cta_m64", 64
    if num_heads == 16 and head_dim == 576:
        return "h16_m128", 128
    return "generic_m64", 64


def _get_sm100_kernel_class(backend: str):
    """Load the selected implementation without perturbing other variants."""
    if backend == "h128_2cta_m64":
        from .dsa_bwd_sm100_h128_2cta import FlashAttentionDSABackwardSm100H128TwoCTA

        return FlashAttentionDSABackwardSm100H128TwoCTA
    if backend == "h16_m128":
        from .dsa_bwd_sm100_h16 import FlashAttentionDSABackwardSm100H16

        return FlashAttentionDSABackwardSm100H16
    return FlashAttentionDSABackwardSm100


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
    current_stream=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention (DSA) Backward Pass for Blackwell (SM100), with K=V.

    Accepts flat (unbatched) tensors with global topk indices.
    Internally wraps as batch=1 for the CuTe DSL kernel.

    Args:
        q: (total_S_q, nheads, headdim) float16 or bfloat16
        kv: (total_S_kv, headdim) float16 or bfloat16  (K=V, MQA h_kv=1)
        out: (total_S_q, nheads, headdim_v) float16 or bfloat16
        dout: (total_S_q, nheads, headdim_v) float16 or bfloat16
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
    # Mirror the check_support gate: the SM100 kernel is tiled only for
    # head_dim in {512, 576}; any other value indexes shared memory out of
    # bounds and crashes inside the kernel.
    assert head_dim in (512, 576), f"head_dim must be 512 or 576, got {head_dim}"
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
    assert all(t.is_cuda and t.device == device for t in tensors_to_check), f"all inputs must be CUDA tensors on {device}"

    # Cross-tensor shape validation: every tensor below is indexed with
    # coordinates derived from q, so a mismatched shape silently reads or
    # writes out of place instead of failing.
    assert kv.ndim == 2 and kv.shape[1] == head_dim, f"kv shape mismatch: expected (total_S_kv, {head_dim}), got {tuple(kv.shape)}"
    expected_o_shape = (total_S_q, num_head, head_dim_v)
    assert out.shape == expected_o_shape, f"out shape mismatch: expected {expected_o_shape}, got {tuple(out.shape)}"
    assert dout.shape == expected_o_shape, f"dout shape mismatch: expected {expected_o_shape}, got {tuple(dout.shape)}"
    assert lse.shape == (total_S_q, num_head), f"lse shape mismatch: expected {(total_S_q, num_head)}, got {tuple(lse.shape)}"
    assert attn_sink.shape == (num_head,), f"attn_sink shape mismatch: expected {(num_head,)}, got {tuple(attn_sink.shape)}"
    assert topk_idxs.ndim == 2 and topk_idxs.shape[0] == total_S_q, f"topk_idxs shape mismatch: expected ({total_S_q}, topk_max), got {tuple(topk_idxs.shape)}"
    if topk_length is not None:
        assert topk_length.dtype == torch.int32, f"topk_length dtype mismatch: expected torch.int32, got {topk_length.dtype}"
        assert topk_length.shape == (total_S_q,), f"topk_length shape mismatch: expected {(total_S_q,)}, got {tuple(topk_length.shape)}"

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    max_topk = topk_idxs.shape[1]
    device_capability = torch.cuda.get_device_capability(q.device)
    backend, block_tile = _select_sm100_backend(
        num_head,
        head_dim,
        head_dim_v=head_dim_v,
        dtype=q.dtype,
        max_topk=max_topk,
        device_capability=device_capability,
    )
    kernel_cls = _get_sm100_kernel_class(backend)
    uses_h128_two_cta = backend == "h128_2cta_m64"
    batch_size = 1

    current_stream = resolve_stream(current_stream)

    # Normalize inputs and allocate outputs/workspaces on the execution stream:
    # the kernel below launches on `current_stream`, so the semantically
    # required zero-initialization of dkv/d_sink and both workspaces (and any
    # contiguity copies) must be stream-ordered with it, not with the ambient
    # torch stream the caller happens to be on.
    with torch_stream_context(current_stream):
        # Ensure contiguous
        q, kv, out, dout = [t.contiguous() for t in (q, kv, out, dout)]
        lse = lse.contiguous()
        attn_sink = attn_sink.contiguous()
        topk_idxs = topk_idxs.contiguous()
        if topk_length is not None:
            topk_length = topk_length.contiguous()

        # Allocate output tensors
        if dq is None:
            dq = torch.empty_like(q)
        else:
            assert dq.shape == q.shape, f"dq shape mismatch: expected {q.shape}, got {dq.shape}"
            assert dq.dtype == q.dtype, f"dq dtype mismatch: expected {q.dtype}, got {dq.dtype}"
            assert dq.device == device, f"dq device mismatch: expected {device}, got {dq.device}"
            # The compile cache is keyed without output strides, so a caller
            # provided output must match the contiguous layout the kernel was
            # compiled for (it is not copied: that would break out-parameter
            # identity).
            assert dq.is_contiguous(), "dq must be contiguous"
        if dkv is None:
            dkv = (
                torch.empty(total_S_kv, head_dim, dtype=kv.dtype, device=device)
                if uses_h128_two_cta
                else torch.zeros(total_S_kv, head_dim, dtype=kv.dtype, device=device)
            )
        else:
            expected_dkv_shape = (total_S_kv, head_dim)
            assert dkv.shape == expected_dkv_shape, f"dkv shape mismatch: expected {expected_dkv_shape}, got {dkv.shape}"
            assert dkv.dtype == kv.dtype, f"dkv dtype mismatch: expected {kv.dtype}, got {dkv.dtype}"
            assert dkv.device == device, f"dkv device mismatch: expected {device}, got {dkv.device}"
            assert dkv.is_contiguous(), "dkv must be contiguous"
            # The H128 two-CTA kernel accumulates into a separate zeroed FP32
            # workspace and its conversion kernel overwrites every public dKV
            # element. Keep caller poison intact to enforce that contract.
            if not uses_h128_two_cta:
                dkv.fill_(0)
        d_sink = torch.empty_like(attn_sink) if uses_h128_two_cta else torch.zeros_like(attn_sink)

        # Allocate workspace tensors
        acc_dtype = cutlass.Float32
        ws_lse_odo_shape = kernel_cls._get_workspace_size_LSE_OdO(
            total_S_q,
            head_dim,
            num_head,
            batch_size,
            acc_dtype,
        )
        if uses_h128_two_cta:
            # The main kernel uniquely publishes both FP32 statistics planes;
            # the dSink helper never reads padded query rows.
            workspace_LSE_OdO = torch.empty(*ws_lse_odo_shape, dtype=torch.uint8, device=device)
        else:
            workspace_LSE_OdO = torch.zeros(
                *ws_lse_odo_shape,
                dtype=torch.uint8,
                device=device,
            )

        ws_dkv_shape = kernel_cls._get_workspace_size_dKV(
            total_S_kv,
            head_dim,
            batch_size,
            acc_dtype,
        )
        if uses_h128_two_cta:
            # The specialization establishes its own call-local FP32 zero
            # state in a stream-ordered device kernel.
            workspace_dKV = torch.empty(*ws_dkv_shape, dtype=torch.uint8, device=device)
        else:
            workspace_dKV = torch.zeros(
                *ws_dkv_shape,
                dtype=torch.uint8,
                device=device,
            )

    problem_shape = (total_S_q, total_S_kv, head_dim, (num_head, batch_size))

    dtype = torch2cute_dtype_map[q.dtype]

    has_topk_length = topk_length is not None
    compile_key = (backend, dtype, head_dim, head_dim_v, num_head, block_tile, max_topk, has_topk_length)

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

        kernel_obj = kernel_cls(
            element_dtype=dtype,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            block_tile=block_tile,
            max_topk=max_topk,
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
                options=compile_options(),
            )

    with torch.cuda.nvtx.range(f"flash_attn_bwd_sm100_kernel[{backend}]"):
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
