#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple, Optional

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Float16, BFloat16

from ._kernels.hstu_fwd import HSTUAttentionForwardSm100
from ._kernels.hstu_bwd import HSTUAttentionBackwardSm100
from ._kernels.hstu_bwd_q1 import HSTUAttentionBackwardQlen1Sm100
from ._kernels.block_sparse_builder import (
    build_hstu_k2q_block_sparse,
    build_hstu_q2k_block_sparse,
)
from ._kernels.block_sparsity import HSTUBlockSparseTensors


def _cutlass_dsl_version() -> tuple[int, int, int]:
    """Return the installed CUTLASS DSL version without another dependency."""
    version = getattr(cutlass, "__version__", None)
    try:
        parts = str(version).split(".")
        parsed = []
        for index in range(3):
            digits = ""
            for character in parts[index]:
                if not character.isdigit():
                    break
                digits += character
            if not digits:
                raise ValueError
            parsed.append(int(digits))
        return tuple(parsed)
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse CUTLASS DSL version {version!r}") from exc


def _normalize_scaling_seqlen(
    scaling_seqlen: Optional[float],
    max_seqlen_q: int,
) -> float:
    """Resolve the public HSTU sequence-normalization factor."""
    value = float(max_seqlen_q if scaling_seqlen is None else scaling_seqlen)
    if value <= 0.0:
        raise ValueError(f"scaling_seqlen must be positive, got {value}")
    return value


def _mark_dynamic_tensor(
    tensor: torch.Tensor,
    leading_dim: int,
    *,
    compact: bool = False,
    stride_order=(2, 3, 0, 4, 1),
):
    if tensor.data_ptr() % 16 != 0:
        raise ValueError("HSTU CuTe tensor storage must be 16-byte aligned")
    cute_tensor = from_dlpack(tensor.detach(), assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading_dim)
    if compact:
        cute_tensor = cute_tensor.mark_compact_shape_dynamic(
            mode=1,
            stride_order=stride_order,
            divisibility=64,
        )
    return cute_tensor


def _make_q1_dynamic_thd_tensor(
    tensor: torch.Tensor,
    total_tokens,
):
    """Build a qlen=1 THD compile descriptor with a dynamic token extent.

    ``mark_layout_dynamic`` only makes strides dynamic.  Packed token totals
    vary between continuous-batching steps, so the first shape mode must be a
    symbol as well.  Keep heads and head dim static, while preserving the
    qlen=1 contract's independently dynamic, 128-bit-aligned token/head
    strides.
    """
    if tensor.data_ptr() % 16 != 0:
        raise ValueError("HSTU CuTe tensor storage must be 16-byte aligned")
    element_type = Float16 if tensor.dtype == torch.float16 else BFloat16
    return cute.runtime.make_fake_tensor(
        element_type,
        (total_tokens, tensor.shape[1], tensor.shape[2]),
        (
            cute.sym_int64(divisibility=8),
            cute.sym_int64(divisibility=8),
            1,
        ),
        assumed_align=16,
    )


def _mark_optional_tensor(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return _mark_dynamic_tensor(tensor, tensor.ndim - 1)


def _mark_block_sparse_tensors(tensors):
    if tensors is None:
        return None
    return HSTUBlockSparseTensors(*(_mark_dynamic_tensor(tensor, tensor.ndim - 1) for tensor in tensors[:6]))


def _runtime_block_sparse_tensors(tensors):
    if tensors is None:
        return None
    return tuple(tensors[:6])


def _is_head_major_compact(t: torch.Tensor) -> bool:
    if t.dim() != 3:
        return False
    total_tokens, _, head_dim = t.shape
    return t.stride() == (head_dim, total_tokens * head_dim, 1)


def _as_bwd_compact_layout(t: torch.Tensor) -> torch.Tensor:
    if _is_head_major_compact(t):
        head_major = t.permute(1, 0, 2)
    else:
        head_major = t.permute(1, 0, 2).clone(memory_format=torch.contiguous_format)
    return head_major.permute(1, 2, 0).unsqueeze(3).unsqueeze(2)


def _empty_bwd_compact_layout_like(t: torch.Tensor) -> torch.Tensor:
    total_tokens, num_heads, head_dim = t.shape
    head_major = torch.empty(
        (num_heads, total_tokens, head_dim),
        dtype=t.dtype,
        device=t.device,
    )
    return head_major.permute(1, 2, 0).unsqueeze(3).unsqueeze(2)


def _as_bwd_original_qkv_layout(t: torch.Tensor) -> torch.Tensor:
    total_tokens, num_heads, head_dim = t.shape
    stride_token, stride_head, stride_dim = t.stride()
    return t.as_strided(
        (total_tokens, head_dim, 1, num_heads, 1),
        (
            stride_token,
            stride_dim,
            stride_head * num_heads,
            stride_head,
            stride_dim,
        ),
    )


def _supports_bwd_original_qkv_layout(t: torch.Tensor) -> bool:
    if t.dim() != 3 or t.stride(2) != 1:
        return False
    if t.data_ptr() % 16 != 0:
        return False
    dimensions = sorted((int(stride), int(size)) for size, stride in zip(t.shape, t.stride()) if size > 1)
    covered_span = 1
    for stride, size in dimensions:
        if stride < covered_span:
            return False
        covered_span += (size - 1) * stride
    # The backward kernel uses 128-bit global copy/TMA paths. For bf16/fp16,
    # token and head offsets must stay 8-element aligned.
    return t.stride(0) % 8 == 0 and t.stride(1) % 8 == 0


def _supports_bwd_direct_grad_layout(t: torch.Tensor) -> bool:
    """Return whether the fused epilogue can write directly to ``t``.

    The D32/D64/D128 epilogue uses 128-bit stores, so the unit-stride head
    dimension and the token/head offsets must remain 16-byte aligned.
    The kernel restores those dynamic-stride divisibility assumptions before
    constructing its output views.
    """
    # Keep zero strides out of this dynamic-layout cache variant. DLPack/CuTe
    # represents broadcast strides as static zero, so mixing them with a
    # previously compiled nonzero-stride descriptor would not be type-safe.
    return _supports_bwd_original_qkv_layout(t) and t.stride(0) != 0 and t.stride(1) != 0


def _select_q1_bwd_num_threads(
    capability: tuple[int, int],
    batch_size: int,
    num_heads: int,
    split_kv: int,
    head_dim: int,
) -> int:
    """Select the block size paired with the qlen=1 backward split."""
    base_ctas = batch_size * num_heads
    if split_kv == 1 and base_ctas >= 2048:
        # Once the unsplit grid already has enough CTAs, a small block avoids
        # spending 12--16 warps on each short sequence. Split kernels already
        # reach the same four-warp floor below. The unsplit dQ reduction writes
        # one element per thread, so its block must cover the full head.
        return max(128, head_dim)
    if capability in ((10, 0), (10, 3)):
        # Paired B200/B300 sweeps show the same block-size trend, so SM100 and
        # SM103 share one policy. Five 12-warp CTAs fit per B300 SM and win
        # once the grid is large; 16 warps expose more latency-hiding work for
        # smaller grids.
        num_threads = 384 if batch_size >= 448 else 512
    elif capability == (10, 7):
        num_threads = 512
    else:
        num_threads = 256
    if split_kv > 1:
        # Shrink each split CTA, but keep at least four warps so the per-CTA
        # dQ reduction and short KV loop still have enough parallel work.
        num_threads = max(128, (num_threads // split_kv // 32) * 32)
    return num_threads


def _hstu_varlen_bwd_q1_direct(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    alpha: float,
    scaling_seqlen: float,
    is_local: bool,
    window_size_left: int,
    window_size_right: int,
    split_kv: int,
    rows_per_warp: int,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    *,
    _compile_only: bool,
):
    no_preallocated_grads = dq is None and dk is None and dv is None
    if not no_preallocated_grads and (dq is None or dk is None or dv is None):
        raise ValueError("dq, dk, and dv must either all be supplied or all be omitted")
    if no_preallocated_grads:
        dq, dk, dv = [torch.empty_like(tensor, memory_format=torch.preserve_format) for tensor in (q, k, v)]
    assert dq is not None and dk is not None and dv is not None
    if dq.shape != q.shape or dk.shape != k.shape or dv.shape != v.shape:
        raise ValueError("HSTU gradient outputs must match their corresponding inputs")

    batch_size = cu_seqlens_q.shape[0] - 1
    capability = _get_q1_device_capability(q.device)
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_threads = _select_q1_bwd_num_threads(capability, batch_size, num_heads, split_kv, head_dim)
    compile_key = (q.device, q.dtype, head_dim, num_heads, num_threads, is_local, split_kv, rows_per_warp)
    if compile_key not in _hstu_varlen_bwd_q1_direct.compile_cache:
        total_q = cute.sym_int(divisibility=1)
        total_k = cute.sym_int(divisibility=1)
        batch_plus_one = cute.sym_int(divisibility=1)
        q_tensor = _make_q1_dynamic_thd_tensor(q, total_q)
        do_tensor = _make_q1_dynamic_thd_tensor(do, total_q)
        dq_tensor = _make_q1_dynamic_thd_tensor(dq, total_q)
        k_tensor = _make_q1_dynamic_thd_tensor(k, total_k)
        v_tensor = _make_q1_dynamic_thd_tensor(v, total_k)
        dk_tensor = _make_q1_dynamic_thd_tensor(dk, total_k)
        dv_tensor = _make_q1_dynamic_thd_tensor(dv, total_k)
        for tensor in (cu_seqlens_q, cu_seqlens_k):
            if tensor.data_ptr() % 16 != 0:
                raise ValueError("HSTU CuTe tensor storage must be 16-byte aligned")
        cu_q_tensor = cute.runtime.make_fake_compact_tensor(
            Int32,
            (batch_plus_one,),
            stride_order=(0,),
            assumed_align=16,
        )
        cu_k_tensor = cute.runtime.make_fake_compact_tensor(
            Int32,
            (batch_plus_one,),
            stride_order=(0,),
            assumed_align=16,
        )
        kernel = HSTUAttentionBackwardQlen1Sm100(
            element_dtype=Float16 if q.dtype == torch.float16 else BFloat16,
            head_dim=head_dim,
            num_threads=num_threads,
            split_kv=split_kv,
            rows_per_warp=rows_per_warp,
            is_local=is_local,
        )
        compile_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        with torch.cuda.nvtx.range("hstu_varlen_bwd_q1_kernel"):
            _hstu_varlen_bwd_q1_direct.compile_cache[compile_key] = cute.compile(
                kernel,
                q_tensor,
                k_tensor,
                v_tensor,
                do_tensor,
                dq_tensor,
                dk_tensor,
                dv_tensor,
                cu_q_tensor,
                cu_k_tensor,
                Int32(window_size_left),
                Int32(window_size_right),
                Int32(batch_size),
                Int32(q.shape[1]),
                alpha,
                scaling_seqlen,
                compile_stream,
                options="--enable-tvm-ffi",
            )

    if not _compile_only:
        with torch.cuda.nvtx.range("hstu_varlen_bwd_q1_kernel"):
            if split_kv > 1:
                dq.zero_()
            _hstu_varlen_bwd_q1_direct.compile_cache[compile_key](
                q,
                k,
                v,
                do,
                dq,
                dk,
                dv,
                cu_seqlens_q,
                cu_seqlens_k,
                Int32(window_size_left),
                Int32(window_size_right),
                Int32(batch_size),
                Int32(q.shape[1]),
                alpha,
                scaling_seqlen,
            )
    return dq, dk, dv


_hstu_varlen_bwd_q1_direct.compile_cache = {}


_q1_device_capability_cache: dict[torch.device, tuple[int, int]] = {}


def _get_q1_device_capability(device: torch.device) -> tuple[int, int]:
    capability = _q1_device_capability_cache.get(device)
    if capability is None:
        capability = torch.cuda.get_device_capability(device)
        _q1_device_capability_cache[device] = capability
    return capability


class _Q1FwdKernelConfig(NamedTuple):
    """Compile-time qlen=1 knobs shared with the standalone tuning benchmark."""

    block_m: int = 128
    block_n: int = 128
    split_kv: int = 1
    single_warp_epilogue: bool = False
    m64_silu_warps: int = 0
    m64_inplace_silu: bool = False
    m64_16dp_silu: bool = False
    m64_tail_branch: bool = False
    m64_kv_stage: int = 0


_Q1_FWD_DEFAULT_CONFIG = _Q1FwdKernelConfig()
_Q1_FWD_D64_D128_CONFIG = _Q1FwdKernelConfig(
    block_m=64,
    m64_inplace_silu=True,
    m64_16dp_silu=True,
    m64_tail_branch=True,
    m64_kv_stage=5,
)
_Q1_FWD_D256_CONFIG = _Q1FwdKernelConfig(
    block_m=64,
    m64_inplace_silu=True,
    m64_16dp_silu=True,
    m64_tail_branch=True,
)


def _select_q1_fwd_config(
    capability: tuple[int, int],
    supported: bool,
    head_dim: int,
    is_local: bool = False,
    window_size_left: Optional[int] = None,
) -> _Q1FwdKernelConfig:
    """Select one of the two measured qlen=1 production configurations."""
    if not supported or capability not in ((10, 0), (10, 3), (10, 7)):
        return _Q1_FWD_DEFAULT_CONFIG
    # The M128 baseline is consistently faster for local D256 with at most 256
    # active KV tokens. M64 remains beneficial for wider windows and D64/D128.
    if is_local and head_dim == 256 and window_size_left is not None and window_size_left <= 255:
        return _Q1_FWD_DEFAULT_CONFIG
    return _Q1_FWD_D256_CONFIG if head_dim == 256 else _Q1_FWD_D64_D128_CONFIG


_Q1_BWD_DIRECT_SPLITS = {
    "direct": 1,
    "direct-split2": 2,
    "direct-split4": 4,
    "direct-split8": 8,
    "direct-split16": 16,
    "direct-split22": 22,
    "direct-split26": 26,
    "direct-split32": 32,
    "direct-split64": 64,
    "direct-pair": 1,
    "direct-pair-split2": 2,
    "direct-pair-split4": 4,
    "direct-pair-split8": 8,
    "direct-pair-split13": 13,
    "direct-pair-split16": 16,
}


def _select_q1_bwd_split_kv(
    requested: str,
    capability: tuple[int, int],
    supported: bool,
    *,
    batch_size: Optional[int] = None,
    num_heads: Optional[int] = None,
    total_kv: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> int:
    """Resolve the cheap workload-aware qlen=1 backward split."""
    if requested in _Q1_BWD_DIRECT_SPLITS:
        return _Q1_BWD_DIRECT_SPLITS[requested]
    if requested != "auto" or not supported:
        return 1

    # Paired B200/B300 sweeps show no stable architecture-specific split
    # crossover, so SM100 and SM103 share one default.
    split_kv = {(10, 0): 8, (10, 3): 8, (10, 7): 13}.get(capability, 1)
    if split_kv == 1:
        return 1

    # Compare integer totals, avoiding a device read or a floating-point
    # average. Scaling KV by D/128 accounts for the kernel's D64/D128/D256
    # row-packing factors. D256 keeps splitting at medium grid sizes because
    # one warp handles only one row; D64/D128 can stop earlier. Once 4096 base
    # CTAs already exist, the small unsplit block remains worthwhile through a
    # somewhat larger per-sequence workload on both targets.
    has_workload = (
        batch_size is not None
        and batch_size > 0
        and num_heads is not None
        and num_heads > 0
        and total_kv is not None
        and total_kv >= 0
        and head_dim is not None
        and head_dim > 0
    )
    if has_workload:
        assert batch_size is not None
        assert num_heads is not None
        assert total_kv is not None
        assert head_dim is not None
        scaled_work = total_kv * head_dim
        base_ctas = batch_size * num_heads
        if scaled_work < batch_size * 128 * 256 and (head_dim < 256 or base_ctas <= 256):
            return 1
        if base_ctas >= 4096 and scaled_work < batch_size * 128 * 768:
            return 1

    # Causal and local use the same thresholds. Local backward still writes
    # zeros over the complete packed KV range, so total KV—not window width—is
    # the relevant work estimate.
    return split_kv


def hstu_varlen_fwd_100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    func: torch.Tensor,
    paged_kv: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    page_indptrs: Optional[torch.Tensor] = None,
    scaling_seqlen: Optional[float] = None,
    *,
    out: Optional[torch.Tensor] = None,
    _compile_only: bool = False,
    _q1_fwd_tuning_config: Optional[_Q1FwdKernelConfig] = None,
):
    scaling_seqlen = _normalize_scaling_seqlen(scaling_seqlen, max_seqlen_q)
    q_dtype = q.dtype
    assert q_dtype == torch.bfloat16 or q_dtype == torch.float16, "Only support bf16 and fp16"
    assert k.dtype == q_dtype, "k and q must have the same dtype"
    assert v.dtype == q_dtype, "v and q must have the same dtype"

    head_dim = q.shape[2]
    head_dim_v = v.shape[2]
    batch_size = cu_seqlens_q.shape[0] - 1
    assert head_dim == head_dim_v, "head_dim and head_dim_v must be equal"
    assert head_dim in (32, 64, 128, 256), "Only support head_dim 32, 64, 128 and 256"

    is_q_len_one = max_seqlen_q == 1
    window_size_left = max_seqlen_k if window_size_left < 0 or window_size_left > max_seqlen_k else window_size_left
    window_size_right = max_seqlen_k if window_size_right < 0 or window_size_right > max_seqlen_k else window_size_right
    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_local = (window_size_left < max_seqlen_k or window_size_right < max_seqlen_k) and not is_causal
    is_arbitrary = func is not None
    use_auto_block_metadata = is_arbitrary
    func_num = func.shape[-2] if func is not None else 0
    is_paged = paged_kv is not None
    q1_dynamic_thd = is_q_len_one and (is_causal or is_local) and not is_arbitrary and not is_paged and q.shape[1] == k.shape[1] == v.shape[1]
    q1_m64_supported = q1_dynamic_thd and head_dim in (64, 128, 256)
    q1_split_supported = q1_m64_supported and is_causal and q_dtype == torch.bfloat16
    capability = _get_q1_device_capability(q.device)
    # Production dispatch has two measured configurations. The private config
    # override is only for the standalone tuning benchmark, which owns the
    # larger candidate set and translates its human-readable names into knobs.
    q1_fwd_config = (
        _q1_fwd_tuning_config
        if _q1_fwd_tuning_config is not None
        else _select_q1_fwd_config(capability, q1_m64_supported, head_dim, is_local, window_size_left)
    )
    kBlockM = q1_fwd_config.block_m
    kBlockN = q1_fwd_config.block_n
    q1_split_kv = q1_fwd_config.split_kv
    q1_single_warp_epilogue = q1_fwd_config.single_warp_epilogue
    q1_m64_silu_warps = q1_fwd_config.m64_silu_warps
    q1_m64_inplace_silu = q1_fwd_config.m64_inplace_silu
    q1_m64_16dp_silu = q1_fwd_config.m64_16dp_silu
    q1_m64_tail_branch = q1_fwd_config.m64_tail_branch
    q1_m64_kv_stage = q1_fwd_config.m64_kv_stage
    # Rubin's two-CTA path supplies useful occupancy for the small qlen=1
    # launch; larger batches have enough query-head CTAs and favor one CTA.
    use_2cta_instrs = (
        (not is_q_len_one or batch_size < 256)
        and q1_split_kv == 1
        and kBlockM == 128
        and not q1_single_warp_epilogue
        and capability == (10, 7)
        and head_dim == 128
        and is_causal
        and not is_local
        and not is_arbitrary
        and not is_paged
        and q.shape[1] == k.shape[1] == v.shape[1]
    )
    # The 4.5.x PipelineAsync descriptor path is nondeterministic when CUDA
    # contexts contend for an SM. Keep CLC scheduling, but use its direct
    # work-coordinate path until the corrected 4.6 pipeline implementation.
    use_clc_descriptor = _cutlass_dsl_version() >= (4, 6, 0)
    if is_paged:
        assert is_causal, "Paged KV is True, but causal mask is False, this is not supported."
        assert not is_local, "Paged KV is True, but local mask is True, this is not supported."
        assert not is_arbitrary, "Paged KV is True, but arbitrary mask is True, this is not supported."
        assert page_ids is not None and page_indptrs is not None, "Paged KV is True, but page metadata is missing."
        assert paged_kv.dim() == 5 and paged_kv.shape[0] > 0 and paged_kv.shape[2] == 128, "Only accept a non-empty 5-D paged KV table with page_size=128"

    # Keep the public output in the standard contiguous (T, H, D) layout so
    # downstream callers can flatten it with view() without an extra copy.
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    else:
        if out.shape != q.shape:
            raise ValueError(f"out must have shape {tuple(q.shape)}, got {tuple(out.shape)}")
        if out.dtype != q.dtype or out.device != q.device:
            raise ValueError("out must have the same dtype and device as q")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
    if q1_split_kv > 1 and not q1_split_supported:
        raise ValueError("The split-KV qlen=1 forward algorithms require causal BF16 qlen=1 with D=64/128/256 and matching Q/K/V heads")
    if kBlockM == 64 and not q1_m64_supported:
        raise ValueError("The M64 qlen=1 forward algorithms require causal or local BF16/FP16 qlen=1 with D=64/128/256 and matching Q/K/V heads")
    if q1_single_warp_epilogue and not q1_split_supported:
        raise ValueError("The single-warp qlen=1 forward experiment requires causal BF16 qlen=1 with D=64/128/256 and matching Q/K/V heads")
    compile_key = (
        q.device,
        q_dtype,
        head_dim,
        kBlockM,
        kBlockN,
        is_causal,
        is_local,
        is_arbitrary,
        is_paged,
        func_num,
        use_auto_block_metadata,
        use_2cta_instrs,
        use_clc_descriptor,
        q1_split_kv,
        # Head count is plan-time metadata. Packed token and batch extents are
        # deliberately absent: the qlen=1 compile descriptors below represent
        # them with SymInt so one artifact re-binds every runtime batch.
        q.shape[1],
        q1_single_warp_epilogue,
        q1_m64_silu_warps,
        q1_m64_inplace_silu,
        q1_m64_16dp_silu,
        q1_m64_tail_branch,
        q1_m64_kv_stage,
    )

    block_sparse_tensors = None
    if use_auto_block_metadata:
        q2k_block_size = (
            kBlockM if head_dim == 256 or is_q_len_one else 2 * kBlockM,
            kBlockN,
        )
        with torch.cuda.nvtx.range("hstu_q2k_block_sparse_builder"):
            block_sparse_tensors = build_hstu_q2k_block_sparse(
                func,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                block_size=q2k_block_size,
                compile_only=_compile_only,
            )

    if _compile_only and compile_key in hstu_varlen_fwd_100.compile_cache:
        return out, None

    # The forward kernel only needs a contiguous last dim (q/k/v are passed via
    # mark_layout_dynamic(leading_dim=ndim-1)); full contiguity is not required.
    # When the (T,H,D) inputs already have a unit-stride last dim and 128-bit
    # aligned token/head strides, feed them in their original layout and skip the
    # contiguous copy. Non-aligned execution inputs use a real D2D clone; a
    # compile-only miss only needs matching empty layout samples.
    needs_contiguous_inputs = not (_supports_bwd_original_qkv_layout(q) and _supports_bwd_original_qkv_layout(k) and _supports_bwd_original_qkv_layout(v))
    if needs_contiguous_inputs:
        if _compile_only:
            q, k, v = [torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device) for tensor in (q, k, v)]
        else:
            q = q.clone(memory_format=torch.contiguous_format)
            k = k.clone(memory_format=torch.contiguous_format)
            v = v.clone(memory_format=torch.contiguous_format)

    paged_kv_flat = None
    if is_paged:
        paged_kv_flat = paged_kv.view(-1, paged_kv.shape[-2], paged_kv.shape[-1])

    if compile_key not in hstu_varlen_fwd_100.compile_cache:
        if q1_dynamic_thd:
            total_q = cute.sym_int(divisibility=1)
            total_k = cute.sym_int(divisibility=1)
            batch_plus_one = cute.sym_int(divisibility=1)
            q_tensor = _make_q1_dynamic_thd_tensor(q, total_q)
            k_tensor = _make_q1_dynamic_thd_tensor(k, total_k)
            v_tensor = _make_q1_dynamic_thd_tensor(v, total_k)
            o_tensor = _make_q1_dynamic_thd_tensor(out, total_q)
            for tensor in (cu_seqlens_q, cu_seqlens_k):
                if tensor.data_ptr() % 16 != 0:
                    raise ValueError("HSTU CuTe tensor storage must be 16-byte aligned")
            cu_seqlens_q_tensor = cute.runtime.make_fake_compact_tensor(
                Int32,
                (batch_plus_one,),
                stride_order=(0,),
                assumed_align=16,
            )
            cu_seqlens_k_tensor = cute.runtime.make_fake_compact_tensor(
                Int32,
                (batch_plus_one,),
                stride_order=(0,),
                assumed_align=16,
            )
        else:
            q_tensor, k_tensor, v_tensor, o_tensor = [_mark_dynamic_tensor(tensor, tensor.ndim - 1) for tensor in (q, k, v, out)]
            cu_seqlens_q_tensor, cu_seqlens_k_tensor = [_mark_dynamic_tensor(tensor, tensor.ndim - 1) for tensor in (cu_seqlens_q, cu_seqlens_k)]
        func_tensor = _mark_optional_tensor(func)
        paged_kv_tensor, page_ids_tensor, page_indptrs_tensor = [_mark_optional_tensor(tensor) for tensor in (paged_kv_flat, page_ids, page_indptrs)]
        block_sparse_cute = _mark_block_sparse_tensors(block_sparse_tensors)
        compile_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        hstu_fwd_sm100 = HSTUAttentionForwardSm100(
            head_dim=head_dim,
            is_causal=is_causal,
            is_local=is_local,
            is_arbitrary=is_arbitrary,
            is_paged=is_paged,
            func_num=func_num,
            kBlockM=kBlockM,
            kBlockN=kBlockN,
            use_auto_block_metadata=use_auto_block_metadata,
            use_2cta_instrs=use_2cta_instrs,
            use_clc_descriptor=use_clc_descriptor,
            is_q_len_one=is_q_len_one and not use_2cta_instrs,
            q1_split_kv=q1_split_kv,
            q1_single_warp_epilogue=q1_single_warp_epilogue,
            q1_m64_silu_warps=q1_m64_silu_warps,
            q1_m64_inplace_silu=q1_m64_inplace_silu,
            q1_m64_16dp_silu=q1_m64_16dp_silu,
            q1_m64_tail_branch=q1_m64_tail_branch,
            q1_m64_kv_stage=q1_m64_kv_stage,
        )
        with torch.cuda.nvtx.range("hstu_varlen_fwd_kernel"):
            hstu_varlen_fwd_100.compile_cache[compile_key] = cute.compile(
                hstu_fwd_sm100,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                Int32(max_seqlen_q),
                Int32(max_seqlen_k),
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                alpha,
                scaling_seqlen,
                compile_stream,
                window_size_left,
                window_size_right,
                func_tensor,
                paged_kv_tensor,
                page_ids_tensor,
                page_indptrs_tensor,
                block_sparse_cute,
                options="--enable-tvm-ffi",
            )

    if _compile_only:
        return out, None

    with torch.cuda.nvtx.range("hstu_varlen_fwd_kernel"):
        if q1_split_kv > 1:
            out.zero_()
        compiled_fwd = hstu_varlen_fwd_100.compile_cache[compile_key]
        compiled_fwd(
            q,
            k,
            v,
            out,
            Int32(max_seqlen_q),
            Int32(max_seqlen_k),
            cu_seqlens_q,
            cu_seqlens_k,
            alpha,
            scaling_seqlen,
            window_size_left,
            window_size_right,
            func,
            paged_kv_flat,
            page_ids,
            page_indptrs,
            _runtime_block_sparse_tensors(block_sparse_tensors),
        )

    return out, None


hstu_varlen_fwd_100.compile_cache = {}


def hstu_varlen_bwd_100(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    func: torch.Tensor,
    deterministic: bool,
    scaling_seqlen: Optional[float] = None,
    *,
    _compile_only: bool = False,
    _q1_bwd_algorithm: str = "auto",
):
    scaling_seqlen = _normalize_scaling_seqlen(scaling_seqlen, max_seqlen_q)
    if deterministic:
        raise NotImplementedError("deterministic HSTU backward is not supported")
    # asserts
    q_dtype = q.dtype
    assert q_dtype == torch.bfloat16 or q_dtype == torch.float16, "Only support bf16 and fp16"
    assert k.dtype == q_dtype, "k and q must have the same dtype"
    assert v.dtype == q_dtype, "v and q must have the same dtype"
    assert do.dtype == q_dtype, "do and q must have the same dtype"
    assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must have dtype int32"
    assert cu_seqlens_k.dtype == torch.int32, "cu_seqlens_k must have dtype int32"

    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_heads_k = k.shape[1]

    assert head_dim in (32, 64, 128, 256), "Only support head_dim 32, 64, 128 and 256"
    assert num_heads == num_heads_k, "Number of heads in key/value and query must be equal"
    assert k.shape[2] == head_dim, "k and q must have the same head_dim"
    assert v.shape[2] == head_dim, "v and q must have the same head_dim"
    assert do.shape == q.shape, "do and q must have the same shape"

    is_q_len_one_supported = max_seqlen_q == 1 and head_dim in (64, 128, 256)
    m_block_size = 128
    n_block_size = 128
    window_size_left = max_seqlen_k if window_size_left < 0 or window_size_left > max_seqlen_k else window_size_left
    window_size_right = max_seqlen_k if window_size_right < 0 or window_size_right > max_seqlen_k else window_size_right
    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_local = (window_size_left < max_seqlen_k or window_size_right < max_seqlen_k) and not is_causal
    is_arbitrary = func is not None
    func_num = func.shape[-2] if func is not None else 0
    use_2cta_instrs = head_dim == 128 and not is_arbitrary and not is_q_len_one_supported
    q1_inputs_direct = all(_supports_bwd_original_qkv_layout(tensor) for tensor in (q, k, v, do))
    q1_outputs_direct = (dq is None and dk is None and dv is None) or (
        dq is not None and dk is not None and dv is not None and all(_supports_bwd_direct_grad_layout(tensor) for tensor in (dq, dk, dv))
    )
    q1_bwd_algorithms = ("auto", *_Q1_BWD_DIRECT_SPLITS)
    if _q1_bwd_algorithm not in q1_bwd_algorithms:
        raise ValueError(f"Unsupported qlen=1 backward algorithm: {_q1_bwd_algorithm}")
    q1_direct_supported = is_q_len_one_supported and (is_causal or is_local) and not is_arbitrary and q1_inputs_direct and q1_outputs_direct
    if _q1_bwd_algorithm in _Q1_BWD_DIRECT_SPLITS and not q1_direct_supported:
        raise ValueError(f"The {_q1_bwd_algorithm} qlen=1 backward algorithm requires causal or local qlen=1 with D=64/128/256 and direct layouts")
    if _Q1_BWD_DIRECT_SPLITS.get(_q1_bwd_algorithm, 1) > 1 and q_dtype != torch.bfloat16:
        raise ValueError("The split-KV qlen=1 backward algorithms currently require BF16")
    capability = _get_q1_device_capability(q.device)
    q1_split_supported = q1_direct_supported and q_dtype == torch.bfloat16
    q1_bwd_split_kv = _select_q1_bwd_split_kv(
        _q1_bwd_algorithm,
        capability,
        q1_split_supported,
        batch_size=batch_size,
        num_heads=num_heads,
        total_kv=k.shape[0],
        head_dim=head_dim,
    )
    if q1_direct_supported:
        selected_q1_bwd_algorithm = "direct-pair" if _q1_bwd_algorithm == "auto" else _q1_bwd_algorithm
        # Keep every lane on one aligned 128-bit vector. A warp consequently packs
        # four D64 rows, two D128 rows, or one D256 row.
        q1_rows_per_warp = 256 // head_dim if selected_q1_bwd_algorithm.startswith("direct-pair") else 1
        return _hstu_varlen_bwd_q1_direct(
            do,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            alpha,
            scaling_seqlen,
            is_local,
            window_size_left,
            window_size_right,
            q1_bwd_split_kv,
            q1_rows_per_warp,
            dq,
            dk,
            dv,
            _compile_only=_compile_only,
        )
    if head_dim == 256:
        # The fused one-CTA kernel's live TMEM ranges exceed the SM100
        # 512-column capacity at D=256. Use the dedicated two-kernel path:
        # dQ first, followed by dK/dV.
        from ._kernels.hstu_bwd_256_cute import hstu_varlen_bwd_256_cute

        return hstu_varlen_bwd_256_cute(
            do,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dq,
            dk,
            dv,
            window_size_left,
            window_size_right,
            alpha,
            scaling_seqlen,
            func=func,
            _compile_only=_compile_only,
        )

    use_auto_block_metadata = is_arbitrary
    block_sparse_tensors = None
    if use_auto_block_metadata:
        # Build on every execution so in-place func updates, including CUDA
        # Graph replay updates, are visible to the consumer.  The private K2Q
        # layout is fixed by the fused D32/D64/D128 backward tile contract.
        block_sparse_tensors = build_hstu_k2q_block_sparse(
            func,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_size=(m_block_size, n_block_size),
            compile_only=_compile_only,
        )

    q_orig, k_orig, v_orig = q, k, v
    dq_orig, dk_orig, dv_orig = dq, dk, dv
    use_original_qkv_layout = _supports_bwd_original_qkv_layout(q) and _supports_bwd_original_qkv_layout(k) and _supports_bwd_original_qkv_layout(v)
    use_original_do_layout = _supports_bwd_original_qkv_layout(do)
    no_preallocated_grads = dq is None and dk is None and dv is None
    implicit_direct_grads = no_preallocated_grads and all(_supports_bwd_direct_grad_layout(tensor) for tensor in (q_orig, k_orig, v_orig))
    preallocated_direct_grads = (
        dq is not None
        and dk is not None
        and dv is not None
        and _supports_bwd_direct_grad_layout(dq)
        and _supports_bwd_direct_grad_layout(dk)
        and _supports_bwd_direct_grad_layout(dv)
    )
    # Gradient stores are independent of whether q/k/v use their original
    # layouts or compact read-only staging buffers.
    use_original_grad_layout = implicit_direct_grads or preallocated_direct_grads
    compile_key = (
        q_orig.device,
        q_dtype,
        head_dim,
        m_block_size,
        n_block_size,
        use_original_qkv_layout,
        use_original_do_layout,
        use_original_grad_layout,
        is_causal,
        is_local,
        is_arbitrary,
        func_num,
        use_auto_block_metadata,
        use_2cta_instrs,
    )
    if _compile_only and compile_key in hstu_varlen_bwd_100.compile_cache:
        if no_preallocated_grads:
            dq_orig, dk_orig, dv_orig = [torch.empty_like(tensor, memory_format=torch.preserve_format) for tensor in (q_orig, k_orig, v_orig)]
        return dq_orig, dk_orig, dv_orig

    if use_original_qkv_layout:
        q = _as_bwd_original_qkv_layout(q)
        k = _as_bwd_original_qkv_layout(k)
        v = _as_bwd_original_qkv_layout(v)
    elif _compile_only:
        q, k, v = [_empty_bwd_compact_layout_like(tensor) for tensor in (q, k, v)]
    else:
        q = _as_bwd_compact_layout(q)
        k = _as_bwd_compact_layout(k)
        v = _as_bwd_compact_layout(v)

    # Preserve an aligned dO layout and avoid a compact staging copy.
    if use_original_do_layout:
        do = _as_bwd_original_qkv_layout(do)
    elif _compile_only:
        do = _empty_bwd_compact_layout_like(do)
    else:
        do = _as_bwd_compact_layout(do)

    if use_original_grad_layout:
        if no_preallocated_grads:
            dq_orig, dk_orig, dv_orig = [
                torch.empty_strided(
                    tensor.shape,
                    tensor.stride(),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                for tensor in (q_orig, k_orig, v_orig)
            ]
        dq, dk, dv = [_as_bwd_original_qkv_layout(tensor) for tensor in (dq_orig, dk_orig, dv_orig)]
    elif use_original_qkv_layout:
        dq = _empty_bwd_compact_layout_like(q_orig)
        dk = _empty_bwd_compact_layout_like(k_orig)
        dv = _empty_bwd_compact_layout_like(v_orig)
    else:
        dq = torch.empty_strided(q.shape, q.stride(), dtype=q.dtype, device=q.device)
        dk = torch.empty_strided(k.shape, k.stride(), dtype=k.dtype, device=k.device)
        dv = torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device=v.device)

    workspace_head_dim = (head_dim + 7) // 8 * 8
    # Allocate and initialize the accumulation workspace directly on the GPU.
    workspace_padding_rows = batch_size * m_block_size if use_2cta_instrs else 0
    workspace_torch = torch.empty(
        (
            num_heads,
            q.shape[0] + workspace_padding_rows,
            workspace_head_dim,
        ),
        dtype=torch.float32,
        device=q.device,
    )
    if not _compile_only:
        workspace_torch.zero_()
    problem_shape = (
        Int32(max_seqlen_q),
        Int32(max_seqlen_k),
        Int32(head_dim),
        ((Int32(1), Int32(num_heads)), Int32(batch_size)),
    )
    if compile_key not in hstu_varlen_bwd_100.compile_cache:
        q_tensor, k_tensor, v_tensor = [
            _mark_dynamic_tensor(
                tensor,
                1,
                compact=not use_original_qkv_layout,
            )
            for tensor in (q, k, v)
        ]
        do_tensor = _mark_dynamic_tensor(
            do,
            1,
            compact=not use_original_do_layout,
        )
        dq_tensor, dk_tensor, dv_tensor = [
            _mark_dynamic_tensor(
                tensor,
                1,
                compact=not use_original_grad_layout,
            )
            for tensor in (dq, dk, dv)
        ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor = [_mark_dynamic_tensor(tensor, tensor.ndim - 1) for tensor in (cu_seqlens_q, cu_seqlens_k)]
        func_tensor = _mark_optional_tensor(func)
        workspace = _mark_dynamic_tensor(
            workspace_torch,
            workspace_torch.ndim - 1,
        )
        block_sparse_cute = _mark_block_sparse_tensors(block_sparse_tensors)
        compile_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        hstu_bwd_sm100 = HSTUAttentionBackwardSm100(
            element_dtype=Float16 if q_dtype == torch.float16 else BFloat16,
            head_dim=head_dim,
            tile_m=m_block_size,
            tile_n=n_block_size,
            is_causal=is_causal,
            is_local=is_local,
            is_arbitrary=is_arbitrary,
            func_num=func_num,
            use_auto_block_metadata=use_auto_block_metadata,
            use_2cta_instrs=use_2cta_instrs,
        )
        with torch.cuda.nvtx.range("hstu_varlen_bwd_kernel"):
            hstu_varlen_bwd_100.compile_cache[compile_key] = cute.compile(
                hstu_bwd_sm100,
                problem_shape,
                q_tensor,
                k_tensor,
                v_tensor,
                dq_tensor,
                dk_tensor,
                dv_tensor,
                do_tensor,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                Int32(window_size_left),
                Int32(window_size_right),
                func_tensor,
                alpha,
                scaling_seqlen,
                workspace,
                block_sparse_cute,
                compile_stream,
                options="--enable-tvm-ffi",
            )

    if _compile_only:
        return dq_orig, dk_orig, dv_orig

    with torch.cuda.nvtx.range("hstu_varlen_bwd_kernel"):
        compiled_bwd = hstu_varlen_bwd_100.compile_cache[compile_key]
        compiled_bwd(
            problem_shape,
            q,
            k,
            v,
            dq,
            dk,
            dv,
            do,
            cu_seqlens_q,
            cu_seqlens_k,
            Int32(window_size_left),
            Int32(window_size_right),
            func,
            alpha,
            scaling_seqlen,
            workspace_torch,
            _runtime_block_sparse_tensors(block_sparse_tensors),
        )

    if use_original_grad_layout:
        return dq_orig, dk_orig, dv_orig

    dq = dq.squeeze(4).squeeze(2).permute(0, 2, 1)
    dk = dk.squeeze(4).squeeze(2).permute(0, 2, 1)
    dv = dv.squeeze(4).squeeze(2).permute(0, 2, 1)

    if dq_orig is not None:
        dq_orig.copy_(dq)
        dq = dq_orig
    if dk_orig is not None:
        dk_orig.copy_(dk)
        dk = dk_orig
    if dv_orig is not None:
        dv_orig.copy_(dv)
        dv = dv_orig

    return dq, dk, dv


hstu_varlen_bwd_100.compile_cache = {}
