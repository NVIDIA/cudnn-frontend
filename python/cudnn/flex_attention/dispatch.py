# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# CuTe DSL arbitrary-mask dispatch for Hopper and Blackwell.

import math
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from cudnn.flex_attention.plan.kernels import (
    BlockSparseTensorsTorch,
    normalize_arbitrary_block_sparse_config,
    normalize_arbitrary_block_sparse_config_bwd,
    to_cute_block_sparse_tensors,
)
from cudnn.flex_attention.plan.mask_plan import (
    canonical_blackwell_arch_family,
    validate_arbitrary_attention_plan,
    validate_arbitrary_plan_runtime_binding,
    validate_arbitrary_plan_signature,
)
from cudnn.flex_attention.plan.validation import (
    SUPPORTED_HEAD_DIM_RULE,
    is_supported_head_dims,
)
from cudnn.flex_attention.kernels.common import device_utils as utils
from cudnn.flex_attention.kernels.common.backward_postprocess import FlexAttentionBackwardPostprocess
from cudnn.flex_attention.kernels.common.backward_preprocess import FlexAttentionBackwardPreprocess
from cudnn.flex_attention.kernels.sm90.bwd.backward import FlexAttentionBackwardSm90
from cudnn.flex_attention.kernels.sm90.bwd.backward_config import resolve_sm90_bwd_consumer_config
from cudnn.flex_attention.kernels.sm90.fwd.forward import FlexAttentionForwardSm90
from cudnn.flex_attention.kernels.sm90.fwd.forward_config import (
    _ResolvedSm90FwdConsumerConfig,
    resolve_sm90_fwd_consumer_config,
)
from cudnn.flex_attention.kernels.sm100.bwd.backward import FlexAttentionBackwardSm100
from cudnn.flex_attention.kernels.sm100.bwd.backward_config import resolve_sm100_bwd_consumer_config
from cudnn.flex_attention.kernels.sm100.bwd.backward_config_hd256 import (
    _ResolvedSm100Hd256DkdvConsumerConfig,
    _ResolvedSm100Hd256DqConsumerConfig,
    resolve_sm100_hd256_dkdv_consumer_config,
    resolve_sm100_hd256_dq_consumer_config,
)
from cudnn.flex_attention.kernels.sm100.bwd.backward_hd256 import (
    BlackwellFusedMultiHeadAttentionBackward,
)
from cudnn.flex_attention.kernels.sm100.fwd.forward_qstage1 import FlexAttentionForwardQStage1Sm100
from cudnn.flex_attention.kernels.sm100.fwd.forward_qstage2 import FlexAttentionForwardSm100
from cudnn.flex_attention.kernels.sm100.fwd.forward_config import (
    _ResolvedSm100FwdConsumerConfig,
    resolve_sm100_fwd_consumer_config,
    resolve_sm100_fwd_qstage1_1cta_consumer_config,
    resolve_sm100_fwd_qstage1_2cta_consumer_config,
)
from cudnn.flex_attention.kernels.sm100.fwd.forward_config_hd256 import (
    _ResolvedSm100Hd256FwdConsumerConfig,
    resolve_sm100_hd256_fwd_consumer_config,
)
from cudnn.flex_attention.kernels.sm100.fwd.forward_hd256 import (
    BlackwellFusedMultiHeadAttentionForward,
)
from cudnn.flex_attention.runtime.arch import SUPPORTED_ARCHES, get_device_arch
from cudnn.flex_attention.runtime.compile_cache import get_jit_cache
from cudnn.flex_attention.runtime.dsl_utils import (
    get_broadcast_dims,
    maybe_contiguous,
    to_cute_tensor,
)
from cudnn.flex_attention.runtime.fake_tensor import is_fake_mode

if os.environ.get("CUTE_DSL_PTXAS_PATH") is not None:
    from cudnn.flex_attention.runtime import ptxas  # noqa: F401

    ptxas.patch()


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


def _compile_with_timing(*args, **kwargs):
    """Compile one CuTe callable and report host compilation latency."""

    started_at = time.perf_counter()
    compiled = cute.compile(*args, **kwargs)
    print(f"Compiled FlexAttention kernel in {time.perf_counter() - started_at:.1f}s")
    return compiled


def _parse_arch_str(arch_str: str) -> int:
    """Parse an SM90/SM100/SM103 architecture override."""

    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if match is None:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch() -> int:
    """Return a supported architecture, honoring the test-only override."""

    arch_override = os.environ.get("FLEX_ATTENTION_ARCH")
    arch = _parse_arch_str(arch_override) if arch_override is not None else get_device_arch()
    if arch not in SUPPORTED_ARCHES:
        raise NotImplementedError(f"cudnn.flex_attention supports SM90, SM100, and SM103; got SM{arch}")
    return arch


def _validate_head_dims(head_dim: int, head_dim_v: int, alignment: int) -> None:
    if not is_supported_head_dims(head_dim, head_dim_v):
        raise ValueError(f"supported head dimensions are {SUPPORTED_HEAD_DIM_RULE}; got " f"({head_dim}, {head_dim_v})")
    if head_dim % alignment != 0 or head_dim_v % alignment != 0:
        raise ValueError(f"head dimensions must be divisible by {alignment}")


def _validate_tensor(
    tensor,
    name: str,
    expected_shape,
    expected_dtype,
    expected_device,
) -> None:
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} != expected {tuple(expected_shape)}")
    if tensor.dtype != expected_dtype:
        raise TypeError(f"{name} dtype {tensor.dtype} != expected {expected_dtype}")
    if tensor.device != expected_device:
        raise ValueError(f"{name} device {tensor.device} != expected {expected_device}")
    if not is_fake_mode() and not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA")


def _block_sparse_runtime_tuple(tensors):
    if tensors is None:
        return None
    return (
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        tensors.full_block_cnt,
        tensors.full_block_idx,
        tensors.cu_total_m_blocks,
        tensors.dq_write_order,
        tensors.dq_write_order_full,
        tensors.mask_block_offset,
        tensors.full_block_offset,
        tensors.mask_block_masks,
        tensors.sequence_desc,
        tensors.fwd_work_desc,
    )


def _validate_mode_geometry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
):
    is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None
    if (cu_seqlens_q is None) != (cu_seqlens_k is None):
        raise ValueError("true-varlen attention requires both cu_seqlens_q and cu_seqlens_k")
    if is_varlen and (max_seqlen_q is None or max_seqlen_k is None):
        raise ValueError("true-varlen attention requires max_seqlen_q and max_seqlen_k")

    num_q_heads, head_dim = q.shape[-2:]
    num_kv_heads = k.shape[-2]
    head_dim_v = v.shape[-1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    if is_varlen:
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("varlen Q/K/V must use THD layout")
        batch_size = cu_seqlens_q.shape[0] - 1
        if cu_seqlens_k.shape != (batch_size + 1,):
            raise ValueError("cu_seqlens_q and cu_seqlens_k batch sizes must match")
        if q.shape != (q.shape[0], num_q_heads, head_dim):
            raise ValueError("invalid varlen Q shape")
        if k.shape != (k.shape[0], num_kv_heads, head_dim):
            raise ValueError("invalid varlen K shape")
        if v.shape != (k.shape[0], num_kv_heads, head_dim_v):
            raise ValueError("invalid varlen V shape")
        total_q = q.shape[0]
        total_k = k.shape[0]
        seqlen_q = max_seqlen_q
        seqlen_k = max_seqlen_k
    else:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("fixed-length Q/K/V must use BSHD layout")
        batch_size, seqlen_q = q.shape[:2]
        if k.shape[:2] != (batch_size, k.shape[1]):
            raise ValueError("fixed-length Q/K batch sizes must match")
        seqlen_k = k.shape[1]
        if k.shape != (batch_size, seqlen_k, num_kv_heads, head_dim):
            raise ValueError("invalid fixed-length K shape")
        if v.shape != (batch_size, seqlen_k, num_kv_heads, head_dim_v):
            raise ValueError("invalid fixed-length V shape")
        total_q = batch_size * seqlen_q
        total_k = batch_size * seqlen_k

    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
    ):
        if tensor is not None:
            if tensor.dtype != torch.int32:
                raise TypeError(f"{name} must be torch.int32")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if tensor.device != q.device:
                raise ValueError(f"{name} must be on the Q device")

    return (
        is_varlen,
        batch_size,
        seqlen_q,
        seqlen_k,
        total_q,
        total_k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        head_dim_v,
    )


def _validate_plan_binding(
    plan,
    *,
    arch: int,
    is_varlen: bool,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    total_q: int,
    total_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cu_seqlens_q,
    cu_seqlens_k,
    context: str,
):
    signature = validate_arbitrary_attention_plan(
        block_sparse_tensors=plan,
    )
    expected_arch_family = "sm90" if arch == 90 else canonical_blackwell_arch_family(arch)
    if signature.arch_family != expected_arch_family:
        raise ValueError(f"{context} arch_family mismatch: expected " f"{expected_arch_family!r}, got {signature.arch_family!r}")
    validate_arbitrary_plan_runtime_binding(
        plan.topology_tensors.runtime_binding,
        is_varlen=is_varlen,
        batch_size=batch_size,
        seqlen_q=None if is_varlen else seqlen_q,
        seqlen_k=None if is_varlen else seqlen_k,
        total_q=total_q,
        total_k=total_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        context=context,
    )
    return signature


def _resolve_forward_config(
    *,
    arch: int,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    num_q_heads: int,
    num_kv_heads: int,
    is_varlen: bool,
    hmask: int,
    pack_gqa: Optional[bool],
    kernel_family: str,
):
    if arch == 90:
        return resolve_sm90_fwd_consumer_config(
            arch=arch,
            dtype=dtype,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            is_varlen=is_varlen,
            hmask=hmask,
            pack_gqa=pack_gqa,
        )
    if kernel_family == "sm100_qstage1_1cta_fwd":
        return resolve_sm100_fwd_qstage1_1cta_consumer_config(
            arch=arch,
            dtype=dtype,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            is_varlen=is_varlen,
            hmask=hmask,
            pack_gqa=pack_gqa,
        )
    if kernel_family == "sm100_qstage1_2cta_fwd":
        return resolve_sm100_fwd_qstage1_2cta_consumer_config(
            arch=arch,
            dtype=dtype,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            is_varlen=is_varlen,
            hmask=hmask,
            pack_gqa=pack_gqa,
        )
    if head_dim == 256 and head_dim_v == 256:
        supported_hd256_families = (
            "sm100_hd256_fwd",
            "sm100_hd256_qstage1_2cta_fwd",
        )
        if kernel_family not in supported_hd256_families:
            raise ValueError(f"unsupported D256 forward kernel family: {kernel_family}")
        return resolve_sm100_hd256_fwd_consumer_config(
            arch=arch,
            dtype=dtype,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            is_varlen=is_varlen,
            hmask=hmask,
            pack_gqa=pack_gqa,
            cta_group_size=(2 if kernel_family == "sm100_hd256_qstage1_2cta_fwd" else 1),
        )
    return resolve_sm100_fwd_consumer_config(
        arch=arch,
        dtype=dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        is_varlen=is_varlen,
        hmask=hmask,
        pack_gqa=pack_gqa,
    )


@dataclass(frozen=True)
class _FwdDispatch:
    arch: int
    is_varlen: bool
    batch_size: int
    seqlen_q: int
    seqlen_k: int
    total_q: int
    total_k: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    head_dim_v: int
    resolved_max_q: int
    resolved_max_k: int
    qhead_per_kvhead: int
    pack_gqa: bool
    config: object
    normalized_plan: BlockSparseTensorsTorch
    output_shape: tuple[int, ...]
    lse_shape: tuple[int, ...]
    use_hd256: bool
    use_qstage1_1cta: bool
    use_qstage1_2cta: bool
    use_smem_mask_pipeline: bool
    num_sms: int
    compile_key: tuple


def _prepare_flex_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    pack_gqa: Optional[bool] = None,
    block_sparse_tensors: BlockSparseTensorsTorch = None,
    has_lse: bool = False,
    sm90_use_smem_mask_pipeline: bool = True,
) -> _FwdDispatch:
    """Resolve and validate one forward launch without compiling or allocating."""

    if type(sm90_use_smem_mask_pipeline) is not bool:
        raise TypeError("sm90_use_smem_mask_pipeline must be a bool")

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Q/K/V must be FP16 or BF16")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("Q/K/V must have the same dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same device")
    if not is_fake_mode() and not q.is_cuda:
        raise ValueError("Q/K/V must be CUDA tensors")

    (
        is_varlen,
        batch_size,
        seqlen_q,
        seqlen_k,
        total_q,
        total_k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        head_dim_v,
    ) = _validate_mode_geometry(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )
    arch = _get_device_arch()
    use_smem_mask_pipeline = sm90_use_smem_mask_pipeline and arch == 90
    _validate_head_dims(head_dim, head_dim_v, 16 // q.element_size())
    resolved_max_q = max_seqlen_q if is_varlen else seqlen_q
    resolved_max_k = max_seqlen_k if is_varlen else seqlen_k
    qhead_per_kvhead = num_q_heads // num_kv_heads

    plan_signature = _validate_plan_binding(
        block_sparse_tensors,
        arch=arch,
        is_varlen=is_varlen,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        total_q=total_q,
        total_k=total_k,
        max_seqlen_q=resolved_max_q,
        max_seqlen_k=resolved_max_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        context="arbitrary forward plan",
    )
    if block_sparse_tensors.mask_block_cnt.ndim != 2:
        raise ValueError("arbitrary forward requires a rank-2 compact Q2K plan")
    config = _resolve_forward_config(
        arch=arch,
        dtype=q.dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        is_varlen=is_varlen,
        hmask=block_sparse_tensors.mask_block_cnt.shape[0],
        pack_gqa=pack_gqa,
        kernel_family=plan_signature.kernel_family,
    )
    validate_arbitrary_plan_signature(
        plan_signature,
        config.plan_signature,
        context="arbitrary forward plan after dispatch",
    )
    pack_gqa = config.pack_gqa
    tile_m, tile_n = config.tile_m, config.tile_n
    plan_tile_m, plan_tile_n = config.block_size
    q_stage = getattr(config, "q_stage", 2)
    cta_group_size = getattr(config, "cta_group_size", 1)
    qratio_plan = qhead_per_kvhead if pack_gqa else 1
    normalized_plan = normalize_arbitrary_block_sparse_config(
        block_sparse_tensors,
        device=q.device,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        is_varlen=is_varlen,
        block_size=(plan_tile_m, plan_tile_n),
        pack_gqa=pack_gqa,
        physical_subtiles=config.physical_subtiles,
        num_mask_payload_groups=config.num_mask_payload_groups,
        payload_padded_words=config.payload_padded_words,
        expected_fixed_total_m_blocks=(None if is_varlen else batch_size * math.ceil(seqlen_q * qratio_plan / plan_tile_m)),
    )
    use_qstage1_2cta = q_stage == 1 and cta_group_size == 2
    use_qstage1_1cta = q_stage == 1 and cta_group_size == 1
    qstage1_overlap_pv_with_k_wait = use_qstage1_1cta and not normalized_plan.narrow_workset

    output_shape = (total_q, num_q_heads, head_dim_v) if is_varlen else (batch_size, seqlen_q, num_q_heads, head_dim_v)
    lse_shape = (num_q_heads, total_q) if is_varlen else (batch_size, num_q_heads, seqlen_q)
    use_hd256 = head_dim == 256 and head_dim_v == 256 and arch in (100, 103)
    num_sms = 132 if is_fake_mode() else torch.cuda.get_device_properties(q.device).multi_processor_count

    compile_key = (
        arch,
        q.dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        pack_gqa,
        is_varlen,
        not has_lse,
        plan_signature.compile_key,
        get_broadcast_dims(q),
        get_broadcast_dims(k),
        get_broadcast_dims(v),
        use_hd256,
        q_stage,
        cta_group_size,
        qstage1_overlap_pv_with_k_wait,
        use_smem_mask_pipeline,
    )

    return _FwdDispatch(
        arch=arch,
        is_varlen=is_varlen,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        total_q=total_q,
        total_k=total_k,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        resolved_max_q=resolved_max_q,
        resolved_max_k=resolved_max_k,
        qhead_per_kvhead=qhead_per_kvhead,
        pack_gqa=pack_gqa,
        config=config,
        normalized_plan=normalized_plan,
        output_shape=output_shape,
        lse_shape=lse_shape,
        use_hd256=use_hd256,
        use_qstage1_1cta=use_qstage1_1cta,
        use_qstage1_2cta=use_qstage1_2cta,
        use_smem_mask_pipeline=use_smem_mask_pipeline,
        num_sms=num_sms,
        compile_key=compile_key,
    )


def _compile_flex_attn_fwd(
    dispatch: _FwdDispatch,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor],
    softmax_scale: float,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    scheduler_tile_counter: Optional[torch.Tensor],
):
    """Compile or reuse the forward callable for a resolved launch."""

    kernel_compile_key = dispatch.compile_key + (
        get_broadcast_dims(out),
        get_broadcast_dims(lse) if lse is not None else None,
    )
    if kernel_compile_key in _flex_attn_fwd.compile_cache:
        return _flex_attn_fwd.compile_cache[kernel_compile_key]

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    q_tensor, k_tensor, v_tensor, o_tensor = [to_cute_tensor(tensor) for tensor in (q, k, v, out)]
    lse_tensor = to_cute_tensor(lse, assumed_align=4) if lse is not None else None
    cu_q_tensor, cu_k_tensor = [
        to_cute_tensor(tensor, assumed_align=4, leading_dim=0) if tensor is not None else None for tensor in (cu_seqlens_q, cu_seqlens_k)
    ]
    sparse_tensor = to_cute_block_sparse_tensors(dispatch.normalized_plan)
    config = dispatch.config

    if dispatch.arch == 90:
        assert isinstance(config, _ResolvedSm90FwdConsumerConfig)
        kernel = FlexAttentionForwardSm90(
            torch2cute_dtype_map[q.dtype],
            dispatch.head_dim,
            dispatch.head_dim_v,
            dispatch.qhead_per_kvhead,
            pack_gqa=dispatch.pack_gqa,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            mma_pv_is_rs=config.mma_pv_is_rs,
            use_smem_mask_pipeline=dispatch.use_smem_mask_pipeline,
            num_mask_payload_groups=config.num_mask_payload_groups,
        )
        scheduler_counter_tensor = (
            to_cute_tensor(
                scheduler_tile_counter,
                assumed_align=4,
                leading_dim=0,
            )
            if scheduler_tile_counter is not None
            else None
        )
        compile_args = [
            kernel,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            cu_q_tensor,
            cu_k_tensor,
            sparse_tensor,
            scheduler_counter_tensor,
            Int32(0),
            current_stream,
        ]
        compile_options = "--enable-tvm-ffi " "--ptxas-options='--register-usage-level=0'"
    elif dispatch.use_hd256:
        assert isinstance(config, _ResolvedSm100Hd256FwdConsumerConfig)
        kernel = BlackwellFusedMultiHeadAttentionForward(
            use_2cta_instrs=config.cta_group_size == 2,
        )
        compile_args = [
            kernel,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            cu_q_tensor,
            cu_k_tensor,
            sparse_tensor,
            current_stream,
        ]
        compile_options = "--enable-tvm-ffi"
    else:
        assert isinstance(config, _ResolvedSm100FwdConsumerConfig)
        kernel_cls = FlexAttentionForwardQStage1Sm100 if dispatch.use_qstage1_1cta or dispatch.use_qstage1_2cta else FlexAttentionForwardSm100
        kernel_kwargs = (
            {
                "use_2cta_instrs": dispatch.use_qstage1_2cta,
                "overlap_pv_with_k_wait": dispatch.use_qstage1_1cta and not dispatch.normalized_plan.narrow_workset,
            }
            if dispatch.use_qstage1_1cta or dispatch.use_qstage1_2cta
            else {}
        )
        kernel = kernel_cls(
            dispatch.head_dim,
            dispatch.head_dim_v,
            qhead_per_kvhead=dispatch.qhead_per_kvhead,
            pack_gqa=dispatch.pack_gqa,
            is_varlen_q=dispatch.is_varlen,
            **kernel_kwargs,
        )
        compile_args = [
            kernel,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            cu_q_tensor,
            cu_k_tensor,
            sparse_tensor,
            current_stream,
        ]
        compile_options = "--enable-tvm-ffi"
    compiled_kernel = _compile_with_timing(
        *compile_args,
        options=compile_options,
    )
    _flex_attn_fwd.compile_cache[kernel_compile_key] = compiled_kernel
    return compiled_kernel


def _launch_flex_attn_fwd(
    dispatch: _FwdDispatch,
    compiled_kernel,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor],
    softmax_scale: float,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    scheduler_tile_counter: Optional[torch.Tensor],
) -> None:
    """Launch a previously compiled forward callable."""

    if not is_fake_mode():
        if scheduler_tile_counter is not None:
            scheduler_tile_counter.zero_()
        sparse_args = _block_sparse_runtime_tuple(dispatch.normalized_plan)
        if dispatch.arch == 90:
            call_args = [
                q.detach(),
                k.detach(),
                v.detach(),
                out.detach(),
                lse,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                sparse_args,
                scheduler_tile_counter,
                dispatch.num_sms,
            ]
        elif dispatch.use_hd256:
            call_args = [
                q.detach(),
                k.detach(),
                v.detach(),
                out.detach(),
                lse,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                sparse_args,
            ]
        else:
            call_args = [
                q.detach(),
                k.detach(),
                v.detach(),
                out.detach(),
                lse,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                sparse_args,
            ]
        compiled_kernel(*call_args)


def _flex_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    pack_gqa: Optional[bool] = None,
    block_sparse_tensors: BlockSparseTensorsTorch = None,
    return_lse: bool = False,
    sm90_use_smem_mask_pipeline: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocating convenience path for packed arbitrary-mask forward."""

    q, k, v = [maybe_contiguous(tensor) for tensor in (q, k, v)]
    needs_lse = q.requires_grad or k.requires_grad or v.requires_grad or return_lse
    dispatch = _prepare_flex_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        pack_gqa=pack_gqa,
        block_sparse_tensors=block_sparse_tensors,
        has_lse=needs_lse,
        sm90_use_smem_mask_pipeline=sm90_use_smem_mask_pipeline,
    )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dispatch.head_dim)
    out = torch.empty(dispatch.output_shape, dtype=q.dtype, device=q.device)
    lse = torch.empty(dispatch.lse_shape, dtype=torch.float32, device=q.device) if needs_lse else None
    if dispatch.total_q == 0 or dispatch.total_k == 0:
        out.zero_()
        if lse is not None:
            lse.fill_(float("-inf"))
        return out, lse
    scheduler_tile_counter = torch.zeros((1,), dtype=torch.int32, device=q.device) if dispatch.arch == 90 else None
    compiled_kernel = _compile_flex_attn_fwd(
        dispatch,
        q,
        k,
        v,
        out,
        lse,
        softmax_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        scheduler_tile_counter,
    )
    _launch_flex_attn_fwd(
        dispatch,
        compiled_kernel,
        q,
        k,
        v,
        out,
        lse,
        softmax_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        scheduler_tile_counter,
    )
    return out, lse


_flex_attn_fwd.compile_cache = get_jit_cache("fwd")


def _symbolic_fake_strides(rank: int, divisibility: int):
    """Build a stride-1 inner mode with symbolic aligned outer strides."""
    return tuple(cute.sym_int64(divisibility=divisibility) for _ in range(rank - 1)) + (1,)


def _make_fake_bwd_aux_tensors(dtype, is_varlen: bool):
    sym = cute.sym_int
    divisibility = 128 // dtype.width
    batch, seqlen_q = sym(), sym()
    num_q_heads, head_dim, head_dim_v = sym(), sym(), sym()
    total_q = sym()
    q_rounded = sym()
    q_d_rounded = sym()
    q_prefix = (total_q,) if is_varlen else (batch, seqlen_q)
    out_shape = (*q_prefix, num_q_heads, head_dim_v)
    dq_shape = (*q_prefix, num_q_heads, head_dim)

    out, dout, dq = (
        cute.runtime.make_fake_tensor(
            dtype,
            shape,
            stride=_symbolic_fake_strides(len(shape), divisibility),
            assumed_align=divisibility * dtype.width // 8,
        )
        for shape in (out_shape, out_shape, dq_shape)
    )
    if is_varlen:
        lse_shape = (num_q_heads, total_q)
        lse_log2_shape = (num_q_heads, q_rounded)
        dpsum_shape = (num_q_heads, q_rounded)
        dq_accum_shape = (num_q_heads, q_d_rounded)
    else:
        lse_shape = (batch, num_q_heads, seqlen_q)
        lse_log2_shape = (batch, num_q_heads, q_rounded)
        dpsum_shape = (batch, num_q_heads, q_rounded)
        dq_accum_shape = (batch, num_q_heads, q_d_rounded)
    lse, lse_log2, dpsum, dq_accum = (
        cute.runtime.make_fake_tensor(
            Float32,
            shape,
            stride=_symbolic_fake_strides(len(shape), divisibility),
            assumed_align=divisibility * Float32.width // 8,
        )
        for shape, divisibility in (
            (lse_shape, 1),
            (lse_log2_shape, 4),
            (dpsum_shape, 4),
            (dq_accum_shape, 4),
        )
    )
    return out, dout, dq, lse, lse_log2, dpsum, dq_accum


def _compile_bwd_preprocess(
    dtype,
    head_dim,
    head_dim_v,
    tile_m,
    is_varlen,
    has_dlse,
    has_dq_accum,
    use_padded_offsets,
):
    out, dout, _, lse, lse_log2, dpsum, dq_accum = _make_fake_bwd_aux_tensors(dtype, is_varlen)
    cu_q = cute.runtime.make_fake_tensor(Int32, (cute.sym_int(),), stride=(1,), assumed_align=4) if is_varlen else None
    dlse = cute.runtime.make_fake_tensor(Float32, lse.shape, stride=_symbolic_fake_strides(len(lse.shape), 1), assumed_align=4) if has_dlse else None
    kernel = FlexAttentionBackwardPreprocess(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        use_padded_offsets=use_padded_offsets,
    )
    return _compile_with_timing(
        kernel,
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        dq_accum if has_dq_accum else None,
        cu_q,
        dlse,
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess(
    out,
    dout,
    dpsum,
    lse,
    lse_log2,
    dq_accum,
    cu_seqlens_q,
    dlse,
    dtype,
    head_dim,
    head_dim_v,
    tile_m,
    max_seqlen_q,
    use_padded_offsets,
):
    compiled_kernel = _get_bwd_preprocess_kernel(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        cu_seqlens_q is not None,
        dlse is not None,
        dq_accum is not None,
        use_padded_offsets,
    )
    if not is_fake_mode():
        compiled_kernel(
            out,
            dout,
            dpsum,
            lse,
            lse_log2,
            dq_accum,
            cu_seqlens_q,
            dlse,
            max_seqlen_q,
        )


def _get_bwd_preprocess_kernel(
    dtype,
    head_dim,
    head_dim_v,
    tile_m,
    is_varlen,
    has_dlse,
    has_dq_accum,
    use_padded_offsets,
):
    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        is_varlen,
        has_dlse,
        has_dq_accum,
        use_padded_offsets,
    )
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(*compile_key)
    return _bwd_preprocess.compile_cache[compile_key]


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre")


def _compile_bwd_postprocess(
    dtype,
    head_dim,
    tile_m,
    num_threads,
    atom_layout,
    swap_ab,
    accum_row_major,
    is_varlen,
    use_2cta_instrs,
    cluster_size,
    arch,
):
    _, _, output, _, _, _, accum = _make_fake_bwd_aux_tensors(dtype, is_varlen)
    cu_q = cute.runtime.make_fake_tensor(Int32, (cute.sym_int(),), stride=(1,), assumed_align=4) if is_varlen else None
    kernel = FlexAttentionBackwardPostprocess(
        dtype,
        head_dim,
        arch,
        tile_m,
        num_threads,
        atom_layout,
        swap_ab,
        accum_row_major,
        use_2cta_instrs=use_2cta_instrs,
        cluster_size=cluster_size,
    )
    return _compile_with_timing(
        kernel,
        accum,
        output,
        Float32(0.0),
        cu_q,
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_convert(
    accum,
    output,
    scale,
    cu_seqlens,
    max_seqlen,
    arch,
    dtype,
    head_dim,
    tile_m,
    num_threads,
    atom_layout,
    swap_ab,
    *,
    accum_row_major=False,
    use_2cta_instrs=False,
    cluster_size=1,
):
    compiled_kernel = _get_bwd_postprocess_kernel(
        dtype,
        head_dim,
        tile_m,
        num_threads,
        atom_layout,
        swap_ab,
        accum_row_major,
        cu_seqlens is not None,
        use_2cta_instrs,
        cluster_size,
        arch,
    )
    if not is_fake_mode():
        compiled_kernel(
            accum,
            output,
            scale,
            cu_seqlens,
            max_seqlen,
        )


def _get_bwd_postprocess_kernel(
    dtype,
    head_dim,
    tile_m,
    num_threads,
    atom_layout,
    swap_ab,
    accum_row_major,
    is_varlen,
    use_2cta_instrs,
    cluster_size,
    arch,
):
    compile_key = (
        dtype,
        head_dim,
        tile_m,
        num_threads,
        atom_layout,
        swap_ab,
        accum_row_major,
        is_varlen,
        use_2cta_instrs,
        cluster_size,
        arch,
    )
    if compile_key not in _bwd_postprocess_convert.compile_cache:
        _bwd_postprocess_convert.compile_cache[compile_key] = _compile_bwd_postprocess(*compile_key)
    return _bwd_postprocess_convert.compile_cache[compile_key]


_bwd_postprocess_convert.compile_cache = get_jit_cache("bwd_post")


def _resolve_bwd_compile_options(
    *,
    arch: int,
    use_2cta: bool,
    use_hd256: bool,
) -> str:
    options = "--enable-tvm-ffi"
    if (arch == 103 and use_2cta) or (arch == 100 and use_2cta and use_hd256):
        options += " --opt-level 2"
    return options


@dataclass
class _BwdPreallocated:
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor
    dq_accum: Optional[torch.Tensor]
    dpsum: torch.Tensor
    lse_log2: torch.Tensor
    dk_accum: Optional[torch.Tensor]
    dv_accum: Optional[torch.Tensor]
    dq_semaphore: Optional[torch.Tensor]
    dk_semaphore: Optional[torch.Tensor]
    dv_semaphore: Optional[torch.Tensor]


@dataclass(frozen=True)
class _CompiledBwd:
    compile_key: tuple
    preprocess: object
    main: object
    post_dq: Optional[object]
    post_dk: Optional[object]
    post_dv: Optional[object]
    workspace_specs: tuple


def _checked_preallocated(tensor, name: str, shape, dtype, device):
    if shape is None:
        if tensor is not None:
            raise ValueError(f"{name} must be None for this compiled configuration")
        return None
    if tensor is None:
        raise ValueError(f"{name} is required for this compiled configuration")
    _validate_tensor(tensor, name, shape, dtype, device)
    return tensor


def _flex_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    deterministic: bool = False,
    block_sparse_tensors: BlockSparseTensorsTorch = None,
    dlse: Optional[torch.Tensor] = None,
    _preallocated: Optional[_BwdPreallocated] = None,
    _compiled: Optional[_CompiledBwd] = None,
    _compile_only: bool = False,
    _native_inputs: bool = False,
    _validate_only: bool = False,
    _compile_outputs: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, _CompiledBwd]:
    """Run packed arbitrary-mask backward."""

    use_hd256_input_abi = q.shape[-1] == 256 and v.shape[-1] == 256
    input_align_bytes = 128 if use_hd256_input_abi else 16
    if not _native_inputs:
        q, k, v, out, dout = [maybe_contiguous(tensor, align_bytes=input_align_bytes) for tensor in (q, k, v, out, dout)]
        lse, dlse = [maybe_contiguous(tensor) for tensor in (lse, dlse)]
    (
        is_varlen,
        batch_size,
        seqlen_q,
        seqlen_k,
        total_q,
        total_k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        head_dim_v,
    ) = _validate_mode_geometry(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Q/K/V/O/dO must be FP16 or BF16")
    if not (q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype):
        raise TypeError("Q/K/V/O/dO must have the same dtype")
    if lse.dtype != torch.float32:
        raise TypeError("LSE must be FP32")
    if not is_fake_mode() and not q.is_cuda:
        raise ValueError("backward inputs must be CUDA tensors")

    arch = _get_device_arch()
    _validate_head_dims(head_dim, head_dim_v, 16 // q.element_size())
    resolved_max_q = max_seqlen_q if is_varlen else seqlen_q
    resolved_max_k = max_seqlen_k if is_varlen else seqlen_k
    qhead_per_kvhead = num_q_heads // num_kv_heads
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    expected_out_shape = (total_q, num_q_heads, head_dim_v) if is_varlen else (batch_size, seqlen_q, num_q_heads, head_dim_v)
    expected_lse_shape = (num_q_heads, total_q) if is_varlen else (batch_size, num_q_heads, seqlen_q)
    _validate_tensor(out, "out", expected_out_shape, q.dtype, q.device)
    _validate_tensor(dout, "dout", expected_out_shape, q.dtype, q.device)
    _validate_tensor(lse, "lse", expected_lse_shape, torch.float32, q.device)
    if dlse is not None:
        _validate_tensor(dlse, "dlse", expected_lse_shape, torch.float32, q.device)

    outer_signature = _validate_plan_binding(
        block_sparse_tensors,
        arch=arch,
        is_varlen=is_varlen,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        total_q=total_q,
        total_k=total_k,
        max_seqlen_q=resolved_max_q,
        max_seqlen_k=resolved_max_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        context="arbitrary outer forward plan for backward",
    )
    if block_sparse_tensors.mask_block_cnt.ndim != 2:
        raise ValueError("arbitrary backward requires a rank-2 outer Q2K plan")
    hmask = block_sparse_tensors.mask_block_cnt.shape[0]
    forward_config = _resolve_forward_config(
        arch=arch,
        dtype=q.dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        is_varlen=is_varlen,
        hmask=hmask,
        pack_gqa=block_sparse_tensors.pack_gqa,
        kernel_family=outer_signature.kernel_family,
    )
    if arch != 90:
        validate_arbitrary_plan_signature(
            outer_signature,
            forward_config.plan_signature,
            context="arbitrary outer forward plan",
        )

    nested_plan = block_sparse_tensors.bwd_tensors
    if nested_plan is None:
        raise ValueError("arbitrary backward requires a MaskPlan built with backward metadata")
    nested_signature = _validate_plan_binding(
        nested_plan,
        arch=arch,
        is_varlen=is_varlen,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        total_q=total_q,
        total_k=total_k,
        max_seqlen_q=resolved_max_q,
        max_seqlen_k=resolved_max_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        context="arbitrary nested backward plan",
    )
    if nested_plan.topology_tensors.runtime_binding is not block_sparse_tensors.topology_tensors.runtime_binding:
        raise ValueError("outer and nested backward plans must share the runtime binding")

    use_hd256 = arch in (100, 103) and head_dim == 256 and head_dim_v == 256
    dq_plan = None
    dq_signature = None
    dq_config = None
    if use_hd256:
        if dlse is not None:
            raise NotImplementedError("SM100 head-dim 256 backward does not support an LSE gradient")
        if utils.get_disable_2cta_default():
            raise NotImplementedError("SM100 head-dim 256 arbitrary backward requires 2CTA")
        dq_plan = block_sparse_tensors.dq_tensors
        if dq_plan is None:
            raise ValueError("SM100 head-dim 256 backward requires an independent dQ plan")
        dq_signature = _validate_plan_binding(
            dq_plan,
            arch=arch,
            is_varlen=is_varlen,
            batch_size=batch_size,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            total_q=total_q,
            total_k=total_k,
            max_seqlen_q=resolved_max_q,
            max_seqlen_k=resolved_max_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            context="arbitrary head-dim 256 dQ plan",
        )
        if dq_plan.topology_tensors.runtime_binding is not block_sparse_tensors.topology_tensors.runtime_binding:
            raise ValueError("outer and dQ plans must share the runtime binding")
        dq_config = resolve_sm100_hd256_dq_consumer_config(
            arch=arch,
            dtype=q.dtype,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            is_varlen=is_varlen,
            hmask=hmask,
            pack_gqa=False,
            use_2cta_instrs=True,
            deterministic=deterministic,
        )
        bwd_config = resolve_sm100_hd256_dkdv_consumer_config(
            arch=arch,
            dtype=q.dtype,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            is_varlen=is_varlen,
            hmask=hmask,
            pack_gqa=False,
            use_2cta_instrs=True,
            deterministic=deterministic,
        )
        validate_arbitrary_plan_signature(
            dq_signature,
            dq_config.plan_signature,
            context="arbitrary head-dim 256 dQ plan",
        )
        validate_arbitrary_plan_signature(
            nested_signature,
            bwd_config.plan_signature,
            context="arbitrary head-dim 256 dKdV plan",
        )
        use_2cta = True
        cluster_size = 2
    else:
        bwd_config = (
            resolve_sm90_bwd_consumer_config(
                arch=arch,
                dtype=q.dtype,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                is_varlen=is_varlen,
            )
            if arch == 90
            else resolve_sm100_bwd_consumer_config(
                arch=arch,
                dtype=q.dtype,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                is_varlen=is_varlen,
            )
        )
        if arch != 90:
            validate_arbitrary_plan_signature(
                nested_signature,
                bwd_config.plan_signature,
                context="arbitrary nested backward plan",
            )
        use_2cta = arch != 90 and bwd_config.cta_group_size == 2
        if use_2cta and utils.get_disable_2cta_default():
            raise NotImplementedError("the resolved arbitrary backward plan requires 2CTA")
        cluster_size = 2 if use_2cta else 1

    if arch == 90:
        tile_m = bwd_config.tile_m
        tile_n = bwd_config.tile_n
        num_stages_q = bwd_config.num_stages_q
        num_stages_do = bwd_config.num_stages_do
        num_stages_pds = bwd_config.num_stages_pds
        sdp_swap_ab = bwd_config.sdp_swap_ab
        dkv_swap_ab = bwd_config.dkv_swap_ab
        atom_layout_n_dkv = bwd_config.atom_layout_n_dkv
        atom_layout_m_dq = bwd_config.atom_layout_m_dq
        dq_single_wg = bwd_config.dq_single_wg
    else:
        tile_m = 128
        tile_n = 128
        num_stages_q = num_stages_do = num_stages_pds = 2
        sdp_swap_ab = False
        dkv_swap_ab = False
        atom_layout_n_dkv = 1
        atom_layout_m_dq = 1
        dq_single_wg = False

    if _validate_only:
        return None

    if _preallocated is not None and _compile_outputs is not None:
        raise ValueError("_preallocated and _compile_outputs are mutually exclusive")
    if _preallocated is None:
        if _compile_outputs is None:
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
        else:
            dq = _checked_preallocated(_compile_outputs[0], "dq", tuple(q.shape), q.dtype, q.device)
            dk = _checked_preallocated(_compile_outputs[1], "dk", tuple(k.shape), k.dtype, k.device)
            dv = _checked_preallocated(_compile_outputs[2], "dv", tuple(v.shape), v.dtype, v.device)
    else:
        dq = _checked_preallocated(_preallocated.dq, "dq", tuple(q.shape), q.dtype, q.device)
        dk = _checked_preallocated(_preallocated.dk, "dk", tuple(k.shape), k.dtype, k.device)
        dv = _checked_preallocated(_preallocated.dv, "dv", tuple(v.shape), v.dtype, v.device)
    if total_q == 0 or total_k == 0:
        if _compile_only:
            raise ValueError("cannot compile Flex Attention backward from zero-token sample tensors")
        dq.zero_()
        dk.zero_()
        dv.zero_()
        return dq, dk, dv

    seqlen_q_rounded = math.ceil(seqlen_q / tile_m) * tile_m
    seqlen_k_rounded = math.ceil(seqlen_k / tile_n) * tile_n
    if cluster_size == 2 and (seqlen_k_rounded // tile_n) % 2:
        seqlen_k_rounded += tile_n

    # Generic SM90/SM100 kernels pad MMA head dimensions to 16 elements.
    # Keep the flattened accumulator workspace on the same row stride.
    head_dim_rounded = math.ceil(head_dim / 16) * 16
    head_dim_v_rounded = math.ceil(head_dim_v / 16) * 16
    if is_varlen:
        total_q_padded = (total_q + cu_seqlens_q.shape[0] * tile_m - 1) // tile_m * tile_m
        dq_accum_shape = None if use_hd256 else (num_q_heads, total_q_padded * head_dim_rounded)
        dpsum_shape = (num_q_heads, total_q_padded)
    else:
        dq_accum_shape = None if use_hd256 else (batch_size, num_q_heads, seqlen_q_rounded * head_dim_rounded)
        dpsum_shape = (batch_size, num_q_heads, seqlen_q_rounded)

    if _preallocated is None:
        dq_accum = torch.empty(dq_accum_shape, dtype=torch.float32, device=q.device) if dq_accum_shape is not None else None
        dpsum = torch.empty(dpsum_shape, dtype=torch.float32, device=q.device)
        lse_log2 = torch.empty_like(dpsum)
    else:
        dq_accum = _checked_preallocated(_preallocated.dq_accum, "dq_accum", dq_accum_shape, torch.float32, q.device)
        dpsum = _checked_preallocated(_preallocated.dpsum, "dpsum", dpsum_shape, torch.float32, q.device)
        lse_log2 = _checked_preallocated(_preallocated.lse_log2, "lse_log2", dpsum_shape, torch.float32, q.device)

    # The SM100 direct dK/dV epilogue stores full 16-element MMA columns.  Route
    # padded MHA dimensions through the FP32 workspace as well, so the shared
    # postprocess owns the final head-dimension predicate.
    dkv_postprocess = not use_hd256 and (qhead_per_kvhead > 1 or (arch in (100, 103) and (head_dim_rounded != head_dim or head_dim_v_rounded != head_dim_v)))
    dk_accum_shape = dv_accum_shape = None
    if dkv_postprocess:
        if is_varlen:
            cluster_tile_n = cluster_size * tile_n
            total_k_padded = (total_k + cu_seqlens_k.shape[0] * cluster_tile_n - 1) // cluster_tile_n * cluster_tile_n
            dk_accum_shape = (num_kv_heads, total_k_padded * head_dim_rounded)
            dv_accum_shape = (num_kv_heads, total_k_padded * head_dim_v_rounded)
        else:
            dk_accum_shape = (batch_size, num_kv_heads, seqlen_k_rounded * head_dim_rounded)
            dv_accum_shape = (batch_size, num_kv_heads, seqlen_k_rounded * head_dim_v_rounded)

    if _preallocated is None:
        dk_accum = torch.zeros(dk_accum_shape, dtype=torch.float32, device=q.device) if dk_accum_shape is not None else None
        dv_accum = torch.zeros(dv_accum_shape, dtype=torch.float32, device=q.device) if dv_accum_shape is not None else None
    else:
        dk_accum = _checked_preallocated(_preallocated.dk_accum, "dk_accum", dk_accum_shape, torch.float32, q.device)
        dv_accum = _checked_preallocated(_preallocated.dv_accum, "dv_accum", dv_accum_shape, torch.float32, q.device)

    dtype = torch2cute_dtype_map[q.dtype]
    preprocess_kernel = (
        _compiled.preprocess
        if _compiled is not None
        else _get_bwd_preprocess_kernel(
            dtype,
            head_dim,
            head_dim_v,
            tile_m,
            cu_seqlens_q is not None,
            dlse is not None,
            dq_accum is not None,
            use_hd256,
        )
    )
    normalized_bwd = normalize_arbitrary_block_sparse_config_bwd(
        nested_plan,
        device=q.device,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        is_varlen=is_varlen,
        block_size=bwd_config.block_size,
        subtile_factor=bwd_config.physical_subtiles,
        num_mma_threads=bwd_config.num_mma_threads,
        payload_padded_words=bwd_config.payload_padded_words,
        expected_hmask=hmask,
        expected_spt=bwd_config.spt,
        expected_fixed_total_n_blocks=(None if is_varlen else batch_size * math.ceil(seqlen_k / bwd_config.block_size[1])),
        require_dq_write_order=not use_hd256,
    )
    normalized_dq = None
    if use_hd256:
        assert isinstance(dq_config, _ResolvedSm100Hd256DqConsumerConfig)
        normalized_dq = normalize_arbitrary_block_sparse_config(
            dq_plan,
            device=q.device,
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            is_varlen=is_varlen,
            block_size=dq_config.block_size,
            pack_gqa=False,
            physical_subtiles=dq_config.physical_subtiles,
            num_mask_payload_groups=dq_config.num_mask_payload_groups,
            payload_padded_words=dq_config.payload_padded_words,
            expected_fixed_total_m_blocks=(None if is_varlen else batch_size * math.ceil(seqlen_q / dq_config.block_size[0])),
        )
    spt = deterministic and not use_hd256
    dq_semaphore_shape = (batch_size, num_q_heads, seqlen_q_rounded // tile_m, cluster_size) if deterministic and not use_hd256 else None
    dk_semaphore_shape = (batch_size, num_kv_heads, seqlen_k_rounded // tile_n, 2) if deterministic and qhead_per_kvhead > 1 and not use_hd256 else None
    if _preallocated is None:
        dQ_semaphore = torch.zeros(dq_semaphore_shape, dtype=torch.int32, device=q.device) if dq_semaphore_shape is not None else None
        dK_semaphore = torch.zeros(dk_semaphore_shape, dtype=torch.int32, device=q.device) if dk_semaphore_shape is not None else None
        dV_semaphore = torch.zeros_like(dK_semaphore) if dK_semaphore is not None else None
    else:
        dQ_semaphore = _checked_preallocated(_preallocated.dq_semaphore, "dq_semaphore", dq_semaphore_shape, torch.int32, q.device)
        dK_semaphore = _checked_preallocated(_preallocated.dk_semaphore, "dk_semaphore", dk_semaphore_shape, torch.int32, q.device)
        dV_semaphore = _checked_preallocated(_preallocated.dv_semaphore, "dv_semaphore", dk_semaphore_shape, torch.int32, q.device)

    compile_key = (
        arch,
        q.dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        deterministic,
        spt,
        is_varlen,
        nested_signature.compile_key,
        dq_signature.compile_key if dq_signature is not None else None,
        tile_m,
        tile_n,
        num_stages_q,
        num_stages_do,
        sdp_swap_ab,
        dkv_swap_ab,
        atom_layout_n_dkv,
        atom_layout_m_dq,
        dq_single_wg,
        use_2cta,
        cluster_size,
        bwd_config.subtile_factor,
        get_broadcast_dims(q),
        get_broadcast_dims(k),
        get_broadcast_dims(v),
        get_broadcast_dims(dout),
        get_broadcast_dims(dq),
        get_broadcast_dims(dk),
        get_broadcast_dims(dv),
        use_hd256,
    )
    api_compile_key = compile_key + (dlse is not None,)
    if _compiled is not None and _compiled.compile_key != api_compile_key:
        raise ValueError("runtime tensors, plan, or options do not match the compiled Flex Attention backward configuration")
    if _compiled is None and compile_key not in _flex_attn_bwd.compile_cache:
        current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [to_cute_tensor(tensor) for tensor in (q, k, v, dout, dq, dk, dv)]
        lse_log2_tensor = to_cute_tensor(lse_log2)
        dpsum_tensor = to_cute_tensor(dpsum)
        dq_accum_tensor = dq_tensor if use_hd256 else to_cute_tensor(dq_accum)
        dk_output_tensor = to_cute_tensor(dk_accum) if dkv_postprocess else dk_tensor
        dv_output_tensor = to_cute_tensor(dv_accum) if dkv_postprocess else dv_tensor
        cu_q_tensor, cu_k_tensor = [to_cute_tensor(tensor, assumed_align=4) if tensor is not None else None for tensor in (cu_seqlens_q, cu_seqlens_k)]
        sem_tensors = []
        for semaphore in (dQ_semaphore, dK_semaphore, dV_semaphore):
            if semaphore is None:
                sem_tensors.append(None)
            elif is_fake_mode():
                sem_tensors.append(
                    to_cute_tensor(
                        semaphore,
                        assumed_align=4,
                        leading_dim=3,
                    )
                )
            else:
                # A singleton block axis lets DLPack canonicalize its stride to
                # one, which would make the cached ABI reject longer sequences.
                # Compile from an equivalent non-singleton view so the static
                # stage layout is stable while the block extent remains dynamic.
                compile_semaphore = semaphore
                if semaphore.shape[2] == 1:
                    compile_shape = list(semaphore.shape)
                    compile_shape[2] = 2
                    compile_semaphore = torch.empty(
                        compile_shape,
                        dtype=semaphore.dtype,
                        device=semaphore.device,
                    )
                sem_tensors.append(utils.convert_semaphore_from_dlpack(compile_semaphore.detach()))
        dq_sem_tensor, dk_sem_tensor, dv_sem_tensor = sem_tensors
        sparse_tensor = to_cute_block_sparse_tensors(normalized_bwd)
        sparse_dq_tensor = to_cute_block_sparse_tensors(normalized_dq) if normalized_dq is not None else None

        if arch == 90:
            kernel = FlexAttentionBackwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                deterministic=deterministic,
                spt=spt,
                tile_m=tile_m,
                tile_n=tile_n,
                Q_stage=num_stages_q,
                dO_stage=num_stages_do,
                PdS_stage=num_stages_pds,
                SdP_swapAB=sdp_swap_ab,
                dKV_swapAB=dkv_swap_ab,
                AtomLayoutNdKV=atom_layout_n_dkv,
                AtomLayoutMdQ=atom_layout_m_dq,
                dQ_single_wg=dq_single_wg,
            )
            compile_args = [
                kernel,
                q_tensor,
                k_tensor,
                v_tensor,
                do_tensor,
                lse_log2_tensor,
                dpsum_tensor,
                dq_accum_tensor,
                dk_output_tensor,
                dv_output_tensor,
                softmax_scale,
                cu_q_tensor,
                cu_k_tensor,
                dq_sem_tensor,
                dk_sem_tensor,
                dv_sem_tensor,
                sparse_tensor,
                Int32(0),
                Int32(0),
                current_stream,
            ]
        elif use_hd256:
            assert isinstance(bwd_config, _ResolvedSm100Hd256DkdvConsumerConfig)
            kernel = BlackwellFusedMultiHeadAttentionBackward(
                qhead_per_kvhead,
            )
            compile_args = [
                kernel,
                q_tensor,
                k_tensor,
                v_tensor,
                do_tensor,
                lse_log2_tensor,
                dpsum_tensor,
                dq_accum_tensor,
                dk_output_tensor,
                dv_output_tensor,
                softmax_scale,
                cu_q_tensor,
                cu_k_tensor,
                sparse_dq_tensor,
                sparse_tensor,
                Int32(0),
                Int32(0),
                current_stream,
            ]
        else:
            kernel = FlexAttentionBackwardSm100(
                head_dim,
                head_dim_v,
                qhead_per_kvhead=qhead_per_kvhead,
                use_2cta_instrs=use_2cta,
                deterministic=deterministic,
                spt=spt,
            )
            compile_args = [
                kernel,
                q_tensor,
                k_tensor,
                v_tensor,
                do_tensor,
                lse_log2_tensor,
                dpsum_tensor,
                dq_accum_tensor,
                dk_output_tensor,
                dv_output_tensor,
                softmax_scale,
                cu_q_tensor,
                cu_k_tensor,
                dq_sem_tensor,
                dk_sem_tensor,
                dv_sem_tensor,
                sparse_tensor,
                current_stream,
            ]
        _flex_attn_bwd.compile_cache[compile_key] = _compile_with_timing(
            *compile_args,
            options=_resolve_bwd_compile_options(
                arch=arch,
                use_2cta=use_2cta,
                use_hd256=use_hd256,
            ),
        )

    main_kernel = _compiled.main if _compiled is not None else _flex_attn_bwd.compile_cache[compile_key]

    if not _compile_only and not is_fake_mode():
        if _preallocated is not None:
            for tensor in (dk_accum, dv_accum, dQ_semaphore, dK_semaphore, dV_semaphore):
                if tensor is not None:
                    tensor.zero_()
        preprocess_kernel(
            out,
            dout,
            dpsum,
            lse,
            lse_log2,
            dq_accum,
            cu_seqlens_q,
            dlse,
            resolved_max_q,
        )
        dq_accum_arg = dq if use_hd256 else dq_accum
        dk_arg = dk_accum if dkv_postprocess else dk
        dv_arg = dv_accum if dkv_postprocess else dv
        if arch == 90:
            call_args = [
                q.detach(),
                k.detach(),
                v.detach(),
                dout,
                lse_log2,
                dpsum,
                dq_accum_arg,
                dk_arg,
                dv_arg,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                dQ_semaphore,
                dK_semaphore,
                dV_semaphore,
                _block_sparse_runtime_tuple(normalized_bwd),
                resolved_max_q,
                resolved_max_k,
            ]
        elif use_hd256:
            call_args = [
                q.detach(),
                k.detach(),
                v.detach(),
                dout,
                lse_log2,
                dpsum,
                dq_accum_arg,
                dk_arg,
                dv_arg,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                _block_sparse_runtime_tuple(normalized_dq),
                _block_sparse_runtime_tuple(normalized_bwd),
                resolved_max_q,
                resolved_max_k,
            ]
        else:
            call_args = [
                q.detach(),
                k.detach(),
                v.detach(),
                dout,
                lse_log2,
                dpsum,
                dq_accum_arg,
                dk_arg,
                dv_arg,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                dQ_semaphore,
                dK_semaphore,
                dV_semaphore,
                _block_sparse_runtime_tuple(normalized_bwd),
            ]
        main_kernel(*call_args)

    post_dq_kernel = post_dk_kernel = post_dv_kernel = None
    if not use_hd256:
        num_threads_post_q = 128 if arch == 90 and dq_single_wg else 256 if arch == 90 else 128
        num_threads_post_kv = 256 if arch == 90 else 128
        post_dq_kernel = (
            _compiled.post_dq
            if _compiled is not None
            else _get_bwd_postprocess_kernel(
                dtype,
                head_dim,
                tile_m,
                num_threads_post_q,
                atom_layout_m_dq,
                False,
                False,
                is_varlen,
                use_2cta,
                1,
                arch,
            )
        )
        if not _compile_only and not is_fake_mode():
            post_dq_kernel(dq_accum, dq, softmax_scale, cu_seqlens_q, resolved_max_q)
        if dkv_postprocess:
            accum_row_major = arch == 90 and atom_layout_n_dkv == 1
            post_dk_kernel = (
                _compiled.post_dk
                if _compiled is not None
                else _get_bwd_postprocess_kernel(
                    dtype,
                    head_dim,
                    tile_n,
                    num_threads_post_kv,
                    atom_layout_n_dkv,
                    dkv_swap_ab,
                    accum_row_major,
                    is_varlen,
                    False,
                    cluster_size,
                    arch,
                )
            )
            post_dv_kernel = (
                _compiled.post_dv
                if _compiled is not None
                else _get_bwd_postprocess_kernel(
                    dtype,
                    head_dim_v,
                    tile_n,
                    num_threads_post_kv,
                    atom_layout_n_dkv,
                    dkv_swap_ab,
                    accum_row_major,
                    is_varlen,
                    False,
                    cluster_size,
                    arch,
                )
            )
            if not _compile_only and not is_fake_mode():
                post_dk_kernel(dk_accum, dk, softmax_scale, cu_seqlens_k, resolved_max_k)
                post_dv_kernel(dv_accum, dv, 1.0, cu_seqlens_k, resolved_max_k)

    compiled_bundle = _CompiledBwd(
        compile_key=api_compile_key,
        preprocess=preprocess_kernel,
        main=main_kernel,
        post_dq=post_dq_kernel,
        post_dk=post_dk_kernel,
        post_dv=post_dv_kernel,
        workspace_specs=(
            ("dq_accum", dq_accum_shape, torch.float32),
            ("dpsum", dpsum_shape, torch.float32),
            ("lse_log2", dpsum_shape, torch.float32),
            ("dk_accum", dk_accum_shape, torch.float32),
            ("dv_accum", dv_accum_shape, torch.float32),
            ("dq_semaphore", dq_semaphore_shape, torch.int32),
            ("dk_semaphore", dk_semaphore_shape, torch.int32),
            ("dv_semaphore", dk_semaphore_shape, torch.int32),
        ),
    )
    if _compile_only:
        return dq, dk, dv, compiled_bundle
    return dq, dk, dv


_flex_attn_bwd.compile_cache = get_jit_cache("bwd")


__all__ = ["_flex_attn_fwd", "_flex_attn_bwd"]
