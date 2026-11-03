# Copyright (c) 2026, Jerry Chen
# SPDX-License-Identifier: MIT
import math
from functools import lru_cache
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import device_capability as _device_capability, resolve_stream
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor
from cudnn.frost.buffers import cutedsl_requirement_error, memset_zero_async
from ._interface_sm100_common import (
    _BLACKWELL_CAPABILITIES,
    _align_workspace_bytes,
    _requires_wide_data_addressing,
    _workspace_shapes_sm100,
    flash_attn_bwd_sm100_workspace_size,
)
from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100
from .dsa_bwd_sm100_deterministic import (
    FlashAttentionDSABackwardSm100Deterministic,
    _deterministic_kernel_class_for_config,
    deterministic_kernel_class,
)

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

_SM100_FAMILY_ARCH_FLAGS = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
_DETERMINISTIC_HEAD_COUNTS = (16, 32, 64, 96, 128)
_STATIC_TOPK_STRIDE_BACKENDS = ("h16_m128", "h32_m64", "h96_h64_h32", "h96_d576_2cta_m64")
_H96_D576_TWO_CTA_TOPKS = (128, 512, 1024, 1152, 2048, 2049, 2050, 2051)


def _sm100_gpu_arch_flag(device_capability: Tuple[int, int]) -> str:
    """Return the architecture-specific CuTe target for a supported device."""
    gpu_arch = _SM100_FAMILY_ARCH_FLAGS.get(device_capability)
    if gpu_arch is None:
        raise ValueError(f"SM100 DSA backward does not support compute capability {device_capability}")
    return gpu_arch


def _topk_indices_compile_tensor(topk_idxs: torch.Tensor, max_topk: int, backend: str):
    """Keep Q dynamic while specializing the physical row stride for tuned backends."""
    if backend in _STATIC_TOPK_STRIDE_BACKENDS:
        return cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(), max_topk), stride_order=(1, 0), assumed_align=16)
    return to_cute_tensor(topk_idxs)


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


def _select_sm100_backend(
    num_heads: int,
    head_dim: int,
    *,
    head_dim_v: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    max_topk: int = 0,
    device_capability: Optional[Tuple[int, int]] = None,
    deterministic: bool = False,
    is_contiguous: bool = True,
) -> Tuple[str, int]:
    """Return the tuned SM100 kernel variant and its sparse-row tile size.

    The two-CTA specializations are deliberately fail-closed.  They are
    selected only for the Blackwell SM100/SM103 BF16 envelopes validated by
    their kernels (H128 D512, and contiguous H96/H128 D576 with D_v512); every other
    supported configuration retains the established backend.
    H96/H128 D576 execution uses the two-CTA Tensor Core kernel inside the
    validated envelope; other deterministic envelopes use the established
    generic M64 kernel.
    """
    if (
        device_capability in _BLACKWELL_CAPABILITIES
        and dtype == torch.bfloat16
        and num_heads == 128
        and head_dim == 512
        and head_dim_v == 512
        and max_topk in (128, 512, 1024, 1152, 2048)
        and (not deterministic or is_contiguous)
    ):
        return "h128_2cta_m64", 64
    if (
        device_capability in _BLACKWELL_CAPABILITIES
        and dtype == torch.bfloat16
        and num_heads == 128
        and head_dim == 576
        and head_dim_v == 512
        and max_topk in (128, 512, 1024, 1152, 2048)
        and is_contiguous
    ):
        # The D576 two-CTA plan launches without copies, so it only accepts
        # contiguous inputs; strided inputs keep the generic path below.
        return "h128_d576_2cta_m64", 64
    if (
        device_capability in _BLACKWELL_CAPABILITIES
        and dtype == torch.bfloat16
        and num_heads == 96
        and head_dim == 576
        and head_dim_v == 512
        and max_topk in _H96_D576_TWO_CTA_TOPKS
        and is_contiguous
    ):
        return "h96_d576_2cta_m64", 64
    if num_heads == 96 and head_dim == 576:
        # Unsupported H96 layouts, top-k widths, and devices compose H64 with
        # the tuned H32 tail.
        return "h96_h64_h32", 64
    if deterministic:
        return "generic_m64", 64
    if num_heads == 16 and head_dim == 576:
        # The H16 KV-major specialization can use the full M128 UMMA tile,
        # halving the top-k loop count while keeping one CTA per query token.
        return "h16_m128", 128
    if num_heads == 32 and head_dim == 576:
        return "h32_m64", 64
    return "generic_m64", 64


def _get_sm100_kernel_class(
    backend: str,
    deterministic: bool = False,
    total_s_q: Optional[int] = None,
    total_s_kv: Optional[int] = None,
    head_dim: Optional[int] = None,
    num_heads: int = 64,
    deterministic_relay_cluster_size: Optional[int] = None,
):
    """Load the selected implementation without perturbing other variants.

    Each variant lives in its own CuTe DSL class: embedding variant
    conditionals in a shared class measurably perturbs the code generation
    of the tuned paths.
    """
    if deterministic_relay_cluster_size is not None:
        if not deterministic:
            raise ValueError("deterministic_relay_cluster_size requires deterministic=True")
        if backend != "generic_m64":
            raise ValueError("deterministic_relay_cluster_size is supported only by the deterministic generic_m64 backend")
    if backend == "h128_2cta_m64":
        from .dsa_bwd_sm100_h128_2cta import FlashAttentionDSABackwardSm100H128TwoCTA, FlashAttentionDSABackwardSm100H128TwoCTADeterministic

        return FlashAttentionDSABackwardSm100H128TwoCTADeterministic if deterministic else FlashAttentionDSABackwardSm100H128TwoCTA
    if backend in ("h96_d576_2cta_m64", "h128_d576_2cta_m64"):
        from .dsa_bwd_sm100_h128_d576_2cta import FlashAttentionDSABackwardSm100H128D576TwoCTA, FlashAttentionDSABackwardSm100H128D576TwoCTADeterministic

        return FlashAttentionDSABackwardSm100H128D576TwoCTADeterministic if deterministic else FlashAttentionDSABackwardSm100H128D576TwoCTA
    if backend == "h16_m128":
        from .dsa_bwd_sm100_h16 import FlashAttentionDSABackwardSm100H16

        return FlashAttentionDSABackwardSm100H16
    if backend == "h32_m64":
        from .dsa_bwd_sm100_h32 import FlashAttentionDSABackwardSm100H32

        return FlashAttentionDSABackwardSm100H32
    if backend == "h96_h64_h32":
        from .dsa_bwd_sm100_h96 import FlashAttentionDSABackwardSm100H96, FlashAttentionDSABackwardSm100H96Deterministic

        return FlashAttentionDSABackwardSm100H96Deterministic if deterministic else FlashAttentionDSABackwardSm100H96
    if deterministic:
        if total_s_q is None or total_s_kv is None or head_dim is None:
            raise ValueError("deterministic SM100 kernel selection requires total_s_q, total_s_kv, and head_dim")
        kernel_cls = deterministic_kernel_class(
            total_s_q,
            total_s_kv,
            head_dim,
            num_heads,
            deterministic_relay_cluster_size,
        )
        if getattr(kernel_cls, "dkv_relay_cluster_size", 1) > 1:
            dsl_error = cutedsl_requirement_error("deterministic DSA backward cluster relay")
            if dsl_error is not None:
                raise RuntimeError(dsl_error)
        return kernel_cls
    return FlashAttentionDSABackwardSm100


def _fake_compact_tensor(dtype, shape, alignment=16):
    """Construct a storage-free row-major tensor for plan-time compilation."""
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=alignment,
    )


def _fake_runtime_tensor(dtype, rank, *, divisibility=1, alignment=16):
    """Mirror the dynamic DLPack tensor type rebound by the plan executor."""
    shape = [cute.sym_int() for _ in range(rank)]
    if divisibility != 1:
        shape[-1] = cute.sym_int(divisibility=divisibility)
    stride = [cute.sym_int64(divisibility=divisibility) for _ in range(rank - 1)] + [1]
    return cute.runtime.make_fake_tensor(dtype, tuple(shape), tuple(stride), assumed_align=alignment)


def _compile_sm100_artifact(
    device_index: int,
    device_capability: Tuple[int, int],
    element_dtype,
    head_dim: int,
    head_dim_v: int,
    num_heads: int,
    max_topk: int,
    has_topk_length: bool,
    *,
    kernel_cls,
    block_tile: int,
    dkv_planes: int,
    wide_addressing: bool,
):
    """Build an SM100 launchable from tensor metadata without allocating storage."""
    total_s_q, total_s_kv = cute.sym_int(), cute.sym_int()
    if wide_addressing:
        q = _fake_runtime_tensor(element_dtype, 3, divisibility=head_dim)
        kv = _fake_runtime_tensor(element_dtype, 2, divisibility=head_dim)
        out = _fake_runtime_tensor(element_dtype, 3, divisibility=head_dim_v)
        dout = _fake_runtime_tensor(element_dtype, 3, divisibility=head_dim_v)
        lse = _fake_runtime_tensor(cutlass.Float32, 2, alignment=8)
        sink = _fake_runtime_tensor(cutlass.Float32, 1)
        lengths = _fake_runtime_tensor(cutlass.Int32, 1, alignment=4) if has_topk_length else None
        dq = _fake_runtime_tensor(element_dtype, 3, divisibility=head_dim)
        dkv = _fake_runtime_tensor(element_dtype, 2, divisibility=head_dim)
        d_sink = _fake_runtime_tensor(cutlass.Float32, 1)
        workspace_lse = _fake_runtime_tensor(cutlass.Uint8, 4)
        workspace_dkv = _fake_runtime_tensor(cutlass.Uint8, 4)
        indices = cute.runtime.make_fake_tensor(
            cutlass.Int32,
            (cute.sym_int(), max_topk),
            (cute.sym_int64(divisibility=max(max_topk, 1)), 1),
            assumed_align=16,
        )
    else:
        q = _fake_compact_tensor(element_dtype, (total_s_q, num_heads, head_dim))
        kv = _fake_compact_tensor(element_dtype, (total_s_kv, head_dim))
        out = _fake_compact_tensor(element_dtype, (total_s_q, num_heads, head_dim_v))
        dout = out
        lse = _fake_compact_tensor(cutlass.Float32, (total_s_q, num_heads), alignment=8)
        sink = _fake_compact_tensor(cutlass.Float32, (num_heads,))
        lengths = _fake_compact_tensor(cutlass.Int32, (total_s_q,), alignment=4) if has_topk_length else None
        dq, dkv, d_sink = q, kv, sink
        workspace_lse = _fake_compact_tensor(cutlass.Uint8, (1, num_heads, cute.sym_int(), 8))
        workspace_dkv = _fake_compact_tensor(cutlass.Uint8, (1, dkv_planes, cute.sym_int(), head_dim * 4))
        indices = _fake_compact_tensor(cutlass.Int32, (total_s_q, max_topk))
    problem_shape = (
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(head_dim),
        (cutlass.Int32(num_heads), cutlass.Int32(1)),
    )
    kernel = kernel_cls(
        element_dtype=element_dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        block_tile=block_tile,
        max_topk=max_topk,
    )
    with torch.cuda.device(device_index):
        return cute.compile(
            kernel,
            problem_shape,
            q,
            kv,
            out,
            dout,
            lse,
            sink,
            indices,
            lengths,
            dq,
            dkv,
            d_sink,
            workspace_lse,
            workspace_dkv,
            cutlass.Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options=compile_options(gpu_arch=_sm100_gpu_arch_flag(device_capability)),
        )


@lru_cache(maxsize=None)
def _compile_deterministic_sm100(
    device_index: int,
    device_capability: Tuple[int, int],
    backend: str,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    num_heads: int,
    max_topk: int,
    has_topk_length: bool,
    deterministic_shards: int,
    deterministic_relay_cluster_size: int,
    wide_addressing: bool = False,
):
    """Compile one device-owned, shape-dynamic artifact from plan metadata.

    Query and KV extents remain symbolic, while dtype, head geometry, top-k
    width, optional-length presence, plane count, relay width, and address
    width are compile-time facts.
    The returned launchable belongs to ``device_index``'s CUDA context and may
    be reused only for plans with the same cached signature.
    """
    if device_capability not in _SM100_FAMILY_ARCH_FLAGS:
        raise ValueError(f"deterministic SM100 DSA backward requires an SM100-family device, got {device_capability}")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported deterministic SM100 dtype {dtype}")

    if backend == "generic_m64":
        kernel_cls = _deterministic_kernel_class_for_config(deterministic_shards, deterministic_relay_cluster_size)
    else:
        kernel_cls = _get_sm100_kernel_class(backend, True)
        required_shards = kernel_cls.num_dkv_shards
        required_relay_cluster_size = getattr(kernel_cls, "dkv_relay_cluster_size", 1)
        if required_shards != deterministic_shards:
            raise ValueError(f"deterministic backend {backend!r} requires {required_shards} workspace planes, got {deterministic_shards}")
        if required_relay_cluster_size != deterministic_relay_cluster_size:
            raise ValueError(
                f"deterministic backend {backend!r} requires relay cluster size {required_relay_cluster_size}, " f"got {deterministic_relay_cluster_size}"
            )

    return _compile_sm100_artifact(
        device_index,
        device_capability,
        torch2cute_dtype_map[dtype],
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_heads=num_heads,
        max_topk=max_topk,
        has_topk_length=has_topk_length,
        kernel_cls=kernel_cls,
        block_tile=64,
        dkv_planes=deterministic_shards,
        wide_addressing=wide_addressing,
    )


@lru_cache(maxsize=None)
def _compile_nondeterministic_sm100(
    device_index: int,
    device_capability: Tuple[int, int],
    backend: str,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    num_heads: int,
    max_topk: int,
    has_topk_length: bool,
    block_tile: int,
    wide_addressing: bool = False,
):
    """Compile one ordinary, device-owned, shape-dynamic plan artifact."""
    if backend in ("h96_d576_2cta_m64", "h128_d576_2cta_m64"):
        raise ValueError("the D576 two-CTA backend has its own prepared compiler")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported SM100 dtype {dtype}")
    return _compile_sm100_artifact(
        device_index,
        device_capability,
        torch2cute_dtype_map[dtype],
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_heads=num_heads,
        max_topk=max_topk,
        has_topk_length=has_topk_length,
        kernel_cls=_get_sm100_kernel_class(backend),
        block_tile=block_tile,
        dkv_planes=1,
        wide_addressing=wide_addressing,
    )


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
    *,
    d_sink: Optional[torch.Tensor] = None,
    deterministic_relay_cluster_size: Optional[int] = None,
    _compiled_kernel=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention (DSA) Backward Pass for Blackwell (SM100), with K=V.

    Accepts flat (unbatched) contiguous tensors with global topk indices.
    Internally wraps as batch=1 for the CuTe DSL kernel and never repacks.
    ``dq``/``dkv``/``d_sink`` must be caller-provided (``None`` raises
    ``ValueError``). Prepared plans pass their compiled launchable; direct
    callers retain lazy compilation on first use. The D576 direct entry point
    may allocate omitted scratch; other routes require caller-provided
    workspace.

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
        dq: preallocated (total_S_q, nheads, headdim), same dtype as q
        dkv: preallocated (total_S_kv, headdim), same dtype as kv
        d_sink: preallocated (nheads,) float32
        deterministic: use ordered bounded-wave scratch accumulation; large
            generic-M64 plans automatically use row-aware four-, eight-, or
            sixteen-CTA cluster relays, while supported BF16 H96/H128
            envelopes retain their two-CTA kernels
        deterministic_relay_cluster_size: optional exact 4-, 8-, 16-, 32-, or
            64-CTA logical relay for a deterministic generic-M64 large-context
            launch; 32/64 use two/four ordered 16-CTA physical launches; invalid or
            inapplicable requests fail instead of falling back
        current_stream: CUDA stream handle for all initialization, main, and
            reduction launches; defaults to the current CUDA stream (the plan
            API selects Q's device context before entering this function)
        workspace: reusable uint8 scratch sized by
            ``flash_attn_bwd_sm100_workspace_size``
        _compiled_kernel: internal plan-owned launchable passed only after the
            plan validates its runtime tensor contract; direct callers leave
            this unset to retain low-level validation and lazy compilation

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
    # Prepared plans validate their exact runtime contract before passing the
    # launchable; direct low-level calls retain the checks below.
    plan_inputs_validated = _compiled_kernel is not None

    if not plan_inputs_validated:
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
    if not plan_inputs_validated:
        assert kv.ndim == 2 and kv.shape[1] == head_dim, f"kv shape mismatch: expected (total_S_kv, {head_dim}), got {tuple(kv.shape)}"
        expected_o_shape = (total_S_q, num_head, head_dim_v)
        assert out.shape == expected_o_shape, f"out shape mismatch: expected {expected_o_shape}, got {tuple(out.shape)}"
        assert dout.shape == expected_o_shape, f"dout shape mismatch: expected {expected_o_shape}, got {tuple(dout.shape)}"
        assert lse.shape == (total_S_q, num_head), f"lse shape mismatch: expected {(total_S_q, num_head)}, got {tuple(lse.shape)}"
        assert attn_sink.shape == (num_head,), f"attn_sink shape mismatch: expected {(num_head,)}, got {tuple(attn_sink.shape)}"
        assert (
            topk_idxs.ndim == 2 and topk_idxs.shape[0] == total_S_q
        ), f"topk_idxs shape mismatch: expected ({total_S_q}, topk_max), got {tuple(topk_idxs.shape)}"
        if topk_length is not None:
            assert topk_length.dtype == torch.int32, f"topk_length dtype mismatch: expected torch.int32, got {topk_length.dtype}"
            assert topk_length.shape == (total_S_q,), f"topk_length shape mismatch: expected {(total_S_q,)}, got {tuple(topk_length.shape)}"

    # check_support() declines strided inputs; execute re-validates (Rule 1) and never repacks (Rule 2).
    if not plan_inputs_validated:
        for name, tensor in zip(("q", "kv", "out", "dout", "lse", "attn_sink", "topk_idxs", "topk_length"), tensors_to_check):
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous; got shape {tuple(tensor.shape)} strides {tuple(tensor.stride())}")
        if lse.data_ptr() % 8 != 0:
            raise ValueError(f"lse must be 8-byte aligned for the SM100 FP32 pair-copy path; got data_ptr=0x{lse.data_ptr():x}")

    # The compile cache is keyed without output strides, so a caller-provided
    # output must match the contiguous layout the kernel was compiled for (it
    # is not copied: that would break out-parameter identity).
    for name, tensor in (("dq", dq), ("dkv", dkv), ("d_sink", d_sink)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be preallocated; use sparse_attention_backward_wrapper for automatic output allocation")
    assert dq.shape == q.shape, f"dq shape mismatch: expected {q.shape}, got {dq.shape}"
    assert dq.dtype == q.dtype, f"dq dtype mismatch: expected {q.dtype}, got {dq.dtype}"
    assert dq.device == device, f"dq device mismatch: expected {device}, got {dq.device}"
    assert dq.is_contiguous(), "dq must be contiguous"
    expected_dkv_shape = (total_S_kv, head_dim)
    assert dkv.shape == expected_dkv_shape, f"dkv shape mismatch: expected {expected_dkv_shape}, got {dkv.shape}"
    assert dkv.dtype == kv.dtype, f"dkv dtype mismatch: expected {kv.dtype}, got {dkv.dtype}"
    assert dkv.device == device, f"dkv device mismatch: expected {device}, got {dkv.device}"
    assert dkv.is_contiguous(), "dkv must be contiguous"
    assert d_sink.shape == (num_head,), f"d_sink shape mismatch: expected {(num_head,)}, got {d_sink.shape}"
    assert d_sink.dtype == torch.float32, f"d_sink dtype mismatch: expected torch.float32, got {d_sink.dtype}"
    assert d_sink.device == device, f"d_sink device mismatch: expected {device}, got {d_sink.device}"
    assert d_sink.is_contiguous(), "d_sink must be contiguous"

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    max_topk = topk_idxs.shape[1]
    device_capability = _device_capability(q.device)
    backend, block_tile = _select_sm100_backend(
        num_head,
        head_dim,
        head_dim_v=head_dim_v,
        dtype=q.dtype,
        max_topk=max_topk,
        device_capability=device_capability,
        deterministic=deterministic,
    )
    kernel_cls = _get_sm100_kernel_class(
        backend,
        deterministic,
        total_S_q,
        total_S_kv,
        head_dim,
        num_head,
        deterministic_relay_cluster_size,
    )
    if backend in ("h96_d576_2cta_m64", "h128_d576_2cta_m64"):
        # The D576 two-CTA kernel has its own plan-time compile and launch
        # contract (see ``_interface_sm100_d576``); it shares this entry point
        # only so direct callers get the same selection as the plan.
        from ._interface_sm100_d576 import flash_attn_bwd_sm100_d576_2cta

        return flash_attn_bwd_sm100_d576_2cta(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            dq=dq,
            dkv=dkv,
            d_sink=d_sink,
            current_stream=current_stream,
            workspace=workspace,
            deterministic=deterministic,
        )
    deterministic_shards = kernel_cls.num_dkv_shards if deterministic else 0
    selected_relay_cluster_size = getattr(kernel_cls, "dkv_relay_cluster_size", 1) if deterministic else 0
    uses_h128_two_cta = backend == "h128_2cta_m64"
    batch_size = 1

    current_stream = resolve_stream(current_stream)

    workspace_lse_odo_shape, workspace_dkv_shape = _workspace_shapes_sm100(
        total_S_q,
        total_S_kv,
        head_dim,
        num_head,
        deterministic,
        backend,
        deterministic_relay_cluster_size,
    )
    workspace_LSE_OdO, workspace_dKV = _carve_workspace_sm100(
        workspace,
        device,
        workspace_lse_odo_shape,
        workspace_dkv_shape,
    )
    if uses_h128_two_cta:
        # The two-CTA kernel shares the generic workspace geometry, which keeps
        # ``flash_attn_bwd_sm100_workspace_size`` backend-agnostic for callers.
        assert kernel_cls._get_workspace_size_LSE_OdO(total_S_q, head_dim, num_head, batch_size, cutlass.Float32) == tuple(workspace_lse_odo_shape)
        assert kernel_cls._get_workspace_size_dKV(total_S_kv, head_dim, batch_size, cutlass.Float32) == tuple(workspace_dkv_shape)

    # Every route clears its FP32 dKV accumulator in-kernel and its finalizer
    # overwrites every public dKV element, so dkv needs no reset. Ordinary
    # sum_dSink uses atomics and needs this reset; deterministic sum_dSink and
    # the H128 two-CTA zero_init overwrite/clear d_sink themselves.
    if not uses_h128_two_cta and not deterministic:
        memset_zero_async(d_sink.data_ptr(), d_sink.numel() * d_sink.element_size(), int(current_stream))

    problem_shape = (total_S_q, total_S_kv, head_dim, (num_head, batch_size))

    dtype = torch2cute_dtype_map[q.dtype]

    has_topk_length = topk_length is not None
    # Backend, deterministic mode, and the finite shard specialization select
    # the kernel class. The tail stays ``(max_topk, has_topk_length)`` for the
    # dispatch tests that inspect it.
    compile_key = (
        backend,
        deterministic,
        dtype,
        head_dim,
        head_dim_v,
        num_head,
        block_tile,
        deterministic_shards,
        selected_relay_cluster_size,
        device.index,
        max_topk,
        has_topk_length,
    )

    compiled_kernel = _compiled_kernel
    if compiled_kernel is None and compile_key not in flash_attn_bwd_sm100.compile_cache:
        q_tensor = to_cute_tensor(q, divisibility=head_dim)
        kv_tensor = to_cute_tensor(kv, divisibility=head_dim)
        out_tensor = to_cute_tensor(out, divisibility=head_dim_v)
        dout_tensor = to_cute_tensor(dout, divisibility=head_dim_v)
        lse_tensor = to_cute_tensor(lse, assumed_align=8)
        attn_sink_tensor = to_cute_tensor(attn_sink)
        topk_idxs_tensor = _topk_indices_compile_tensor(topk_idxs, max_topk, backend)
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
                options=compile_options(gpu_arch=_sm100_gpu_arch_flag(device_capability)),
            )
    if compiled_kernel is None:
        compiled_kernel = flash_attn_bwd_sm100.compile_cache[compile_key]

    with torch.cuda.nvtx.range(f"flash_attn_bwd_sm100_kernel[{backend}]"):
        compiled_kernel(
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
