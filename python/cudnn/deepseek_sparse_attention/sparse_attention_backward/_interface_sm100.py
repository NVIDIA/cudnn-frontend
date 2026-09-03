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
from .dsa_bwd_sm100_deterministic import FlashAttentionDSABackwardSm100Deterministic

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

_WORKSPACE_ALIGNMENT = 128
_DETERMINISTIC_HEAD_COUNTS = (16, 32, 64, 96, 128)


def _align_workspace_bytes(num_bytes: int) -> int:
    """Round a workspace segment size up to the shared alignment boundary."""
    return -(-int(num_bytes) // _WORKSPACE_ALIGNMENT) * _WORKSPACE_ALIGNMENT


def _workspace_shapes_sm100(
    total_s_q: int,
    total_s_kv: int,
    head_dim: int,
    num_heads: int,
    deterministic: bool,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return byte-shaped LSE/OdO and dKV scratch layouts for one launch."""
    acc_dtype = cutlass.Float32
    workspace_lse_odo_shape = FlashAttentionDSABackwardSm100._get_workspace_size_LSE_OdO(
        total_s_q,
        head_dim,
        num_heads,
        1,
        acc_dtype,
    )
    dkv_kernel_cls = FlashAttentionDSABackwardSm100Deterministic if deterministic else FlashAttentionDSABackwardSm100
    workspace_dkv_shape = dkv_kernel_cls._get_workspace_size_dKV(
        total_s_kv,
        head_dim,
        1,
        acc_dtype,
    )
    return workspace_lse_odo_shape, workspace_dkv_shape


def flash_attn_bwd_sm100_workspace_size(
    total_s_q: int,
    total_s_kv: int,
    head_dim: int,
    num_heads: int,
    deterministic: bool = False,
) -> int:
    """Return the reusable caller scratch required by the SM100 launch."""
    workspace_lse_odo_shape, workspace_dkv_shape = _workspace_shapes_sm100(
        total_s_q,
        total_s_kv,
        head_dim,
        num_heads,
        deterministic,
    )
    return _align_workspace_bytes(math.prod(workspace_lse_odo_shape)) + _align_workspace_bytes(math.prod(workspace_dkv_shape))


def _carve_workspace_sm100(
    workspace: Optional[torch.Tensor],
    device: torch.device,
    workspace_lse_odo_shape: Tuple[int, ...],
    workspace_dkv_shape: Tuple[int, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate and partition caller scratch into aligned kernel workspaces."""
    required = _align_workspace_bytes(math.prod(workspace_lse_odo_shape)) + _align_workspace_bytes(math.prod(workspace_dkv_shape))
    if workspace is None:
        raise ValueError(f"SM100 DSA backward requires a {required}-byte caller workspace; pass a reusable uint8 CUDA tensor")
    if not isinstance(workspace, torch.Tensor):
        raise TypeError(f"workspace must be a torch.Tensor, got {type(workspace).__name__}")
    if workspace.dtype != torch.uint8:
        raise ValueError(f"workspace must have dtype torch.uint8, got {workspace.dtype}")
    if workspace.device != device:
        raise ValueError(f"workspace must be on {device}, got {workspace.device}")
    if not workspace.is_contiguous():
        raise ValueError("workspace must be contiguous")
    flat = workspace.view(-1)
    if flat.numel() < required:
        raise ValueError(f"SM100 DSA backward requires a {required}-byte workspace; the provided buffer has {flat.numel()} bytes")
    if flat.data_ptr() % 16 != 0:
        raise ValueError(f"workspace must be at least 16-byte aligned; got data_ptr=0x{flat.data_ptr():x}")

    lse_odo_bytes = math.prod(workspace_lse_odo_shape)
    lse_odo_end = lse_odo_bytes
    dkv_begin = _align_workspace_bytes(lse_odo_bytes)
    dkv_end = dkv_begin + math.prod(workspace_dkv_shape)
    return flat[:lse_odo_end].view(workspace_lse_odo_shape), flat[dkv_begin:dkv_end].view(workspace_dkv_shape)


def _select_sm100_backend(num_heads: int, head_dim: int) -> Tuple[str, int]:
    """Return the tuned SM100 kernel variant and its sparse-row tile size."""
    if num_heads == 16 and head_dim == 576:
        return "h16_m128", 128
    if num_heads == 32 and head_dim == 576:
        return "h32_m64", 64
    return "generic_m64", 64


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
    deterministic: bool = False,
    current_stream=None,
    workspace: Optional[torch.Tensor] = None,
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
        deterministic: use the bounded-wave deterministic M64 kernel
        workspace: reusable uint8 scratch sized by
            ``flash_attn_bwd_sm100_workspace_size``

    Returns:
        (dq, dkv, d_sink) -- flat layout gradients
    """
    total_S_q, num_head, head_dim = q.shape
    total_S_kv = kv.shape[0]
    # Mirror the check_support gate: the SM100 kernel is tiled only for
    # head_dim in {512, 576}; any other value indexes shared memory out of
    # bounds and crashes inside the kernel.
    assert head_dim in (512, 576), f"head_dim must be 512 or 576, got {head_dim}"
    assert (
        not deterministic or num_head in _DETERMINISTIC_HEAD_COUNTS
    ), f"deterministic SM100 DSA backward supports heads in {_DETERMINISTIC_HEAD_COUNTS}, got H{num_head}"
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

    # H16 KV-major specialization can use the full M128 UMMA tile.  This
    # halves the top-k loop count while keeping one CTA per query token.
    # Deterministic execution always uses the generic M64 kernel. Head counts
    # below 64 use its masked tail, while counts above 64 serialize their M64
    # head blocks so each dKV shard retains a single writer.
    backend, block_tile = ("generic_m64", 64) if deterministic else _select_sm100_backend(num_head, head_dim)
    num_head_blocks = (num_head + block_tile - 1) // block_tile
    batch_size = 1

    current_stream = resolve_stream(current_stream)

    workspace_lse_odo_shape, workspace_dkv_shape = _workspace_shapes_sm100(
        total_S_q,
        total_S_kv,
        head_dim,
        num_head,
        deterministic,
    )
    workspace_LSE_OdO, workspace_dKV = _carve_workspace_sm100(
        workspace,
        device,
        workspace_lse_odo_shape,
        workspace_dkv_shape,
    )

    # Normalize inputs and allocate outputs on the execution stream:
    # the kernel below launches on `current_stream`, so the semantically
    # required output initialization and any contiguity copies must be
    # stream-ordered with it, not with the ambient torch stream the caller
    # happens to be on. Caller scratch is initialized inside the compiled
    # kernel sequence.
    with torch_stream_context(current_stream):
        # Ensure contiguous
        q, kv, out, dout = [t.contiguous() for t in (q, kv, out, dout)]
        lse = lse.contiguous()
        if lse.data_ptr() % 8 != 0:
            raise ValueError(f"lse must be 8-byte aligned for the SM100 FP32 pair-copy path; got data_ptr=0x{lse.data_ptr():x}")
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
            dkv = torch.zeros(total_S_kv, head_dim, dtype=kv.dtype, device=device)
        else:
            expected_dkv_shape = (total_S_kv, head_dim)
            assert dkv.shape == expected_dkv_shape, f"dkv shape mismatch: expected {expected_dkv_shape}, got {dkv.shape}"
            assert dkv.dtype == kv.dtype, f"dkv dtype mismatch: expected {kv.dtype}, got {dkv.dtype}"
            assert dkv.device == device, f"dkv device mismatch: expected {device}, got {dkv.device}"
            assert dkv.is_contiguous(), "dkv must be contiguous"
            dkv.fill_(0)
        d_sink = torch.zeros_like(attn_sink)

    problem_shape = (total_S_q, total_S_kv, head_dim, (num_head, batch_size))

    dtype = torch2cute_dtype_map[q.dtype]

    has_topk_length = topk_length is not None
    max_topk = topk_idxs.shape[1]
    compile_key = (dtype, head_dim, head_dim_v, num_head, block_tile, max_topk, has_topk_length, deterministic)

    if compile_key not in flash_attn_bwd_sm100.compile_cache:
        q_tensor = to_cute_tensor(q, divisibility=head_dim)
        kv_tensor = to_cute_tensor(kv, divisibility=head_dim)
        out_tensor = to_cute_tensor(out, divisibility=head_dim_v)
        dout_tensor = to_cute_tensor(dout, divisibility=head_dim_v)
        lse_tensor = to_cute_tensor(lse, assumed_align=8)
        attn_sink_tensor = to_cute_tensor(attn_sink)
        topk_idxs_tensor = to_cute_tensor(topk_idxs)
        topk_length_tensor = to_cute_tensor(topk_length) if has_topk_length else None
        dq_tensor = to_cute_tensor(dq, divisibility=head_dim)
        dkv_tensor = to_cute_tensor(dkv, divisibility=head_dim)
        d_sink_tensor = to_cute_tensor(d_sink)
        # Workspace extents vary with Q/KV length but are not part of the
        # plan-time compile key. Keep all modes dynamic; the kernel consumes
        # only the iterators, and the multidimensional shape avoids a single
        # byte extent exceeding Int32 for the multi-GiB deterministic buffer.
        workspace_LSE_OdO_tensor = to_cute_tensor(workspace_LSE_OdO, fully_dynamic=True)
        workspace_dKV_tensor = to_cute_tensor(workspace_dKV, fully_dynamic=True)

        if backend == "h16_m128":
            from .dsa_bwd_sm100_h16 import FlashAttentionDSABackwardSm100H16

            kernel_obj = FlashAttentionDSABackwardSm100H16(
                element_dtype=dtype,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                block_tile=block_tile,
                max_topk=max_topk,
            )
        elif backend == "h32_m64":
            from .dsa_bwd_sm100_h32 import FlashAttentionDSABackwardSm100H32

            kernel_obj = FlashAttentionDSABackwardSm100H32(
                element_dtype=dtype,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                block_tile=block_tile,
                max_topk=max_topk,
            )
        else:
            # Keep this constructor and class byte-for-byte on the tuned H64
            # path; embedding H16 conditionals in the same CuTe DSL class
            # measurably perturbs H64 code generation.
            kernel_cls = FlashAttentionDSABackwardSm100Deterministic if deterministic else FlashAttentionDSABackwardSm100
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
