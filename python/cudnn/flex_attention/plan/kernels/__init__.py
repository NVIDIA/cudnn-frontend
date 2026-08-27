# SPDX-License-Identifier: BSD-3-Clause
"""Runtime tensor schemas and validation for compiled arbitrary-mask plans."""

from __future__ import annotations

from typing import NamedTuple

import cutlass.cute as cute
import torch

from cudnn.flex_attention.plan.mask_plan import (
    ArbitraryPlanSignature,
    ArbitraryTopologyTensors,
)
from cudnn.flex_attention.runtime.dsl_utils import to_cute_tensor
from cudnn.flex_attention.runtime.fake_tensor import is_fake_mode


class BlockSparseTensors(NamedTuple):
    """CuTe view of one compact arbitrary-mask consumer plan."""

    mask_block_cnt: cute.Tensor
    mask_block_idx: cute.Tensor
    full_block_cnt: cute.Tensor | None = None
    full_block_idx: cute.Tensor | None = None
    cu_total_m_blocks: cute.Tensor | None = None
    dq_write_order: cute.Tensor | None = None
    dq_write_order_full: cute.Tensor | None = None
    mask_block_offset: cute.Tensor | None = None
    full_block_offset: cute.Tensor | None = None
    mask_block_masks: cute.Tensor | None = None
    sequence_desc: cute.Tensor | None = None
    fwd_work_desc: cute.Tensor | None = None

    def __new_from_mlir_values__(self, values):
        new_fields = []
        value_idx = 0
        for original in self:
            if original is None:
                new_fields.append(None)
            else:
                new_fields.append(values[value_idx])
                value_idx += 1
        return BlockSparseTensors(*new_fields)


class BlockSparseTensorsTorch(NamedTuple):
    """Torch storage owned by a compiled arbitrary-mask consumer plan."""

    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor | None = None
    full_block_idx: torch.Tensor | None = None
    cu_total_m_blocks: torch.Tensor | None = None
    block_size: tuple[int, int] | None = None
    dq_write_order: torch.Tensor | None = None
    dq_write_order_full: torch.Tensor | None = None
    spt: bool | None = None
    mask_block_offset: torch.Tensor | None = None
    full_block_offset: torch.Tensor | None = None
    mask_block_masks: torch.Tensor | None = None
    pack_gqa: bool | None = None
    bwd_tensors: "BlockSparseTensorsTorch | None" = None
    plan_signature: ArbitraryPlanSignature | None = None
    topology_tensors: ArbitraryTopologyTensors | None = None
    dq_tensors: "BlockSparseTensorsTorch | None" = None
    narrow_workset: bool | None = None
    sequence_desc: torch.Tensor | None = None
    fwd_work_desc: torch.Tensor | None = None


def _validate_required_tensors(
    required: dict[str, tuple[torch.Tensor | None, torch.dtype, int]],
    *,
    device: torch.device,
    context: str,
) -> None:
    for name, (tensor, dtype, ndim) in required.items():
        if tensor is None:
            raise ValueError(f"{context} requires {name}")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}; got {tensor.dtype}")
        if tensor.ndim != ndim:
            raise ValueError(f"{name} must have rank {ndim}; got shape {tuple(tensor.shape)}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}; got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _validate_common_compact_plan(
    tensors: BlockSparseTensorsTorch,
    *,
    device: torch.device,
    num_q_heads: int,
    block_size: tuple[int, int],
    expected_hmask: int | None,
    context: str,
) -> tuple[int, int]:
    if not isinstance(tensors, BlockSparseTensorsTorch):
        raise TypeError(f"{context} requires a plan returned by create_mask_plan")
    if tensors.block_size != block_size:
        raise ValueError(f"{context} block_size={tensors.block_size} does not match " f"the resolved consumer block_size={block_size}")

    _validate_required_tensors(
        {
            "mask_block_cnt": (tensors.mask_block_cnt, torch.int32, 2),
            "mask_block_offset": (tensors.mask_block_offset, torch.int32, 1),
            "mask_block_idx": (tensors.mask_block_idx, torch.int32, 1),
            "full_block_cnt": (tensors.full_block_cnt, torch.int32, 2),
            "full_block_offset": (tensors.full_block_offset, torch.int32, 1),
            "full_block_idx": (tensors.full_block_idx, torch.int32, 1),
            "mask_block_masks": (tensors.mask_block_masks, torch.uint32, 4),
        },
        device=device,
        context=context,
    )

    mask_cnt = tensors.mask_block_cnt
    full_cnt = tensors.full_block_cnt
    assert full_cnt is not None
    hmask, total_blocks = mask_cnt.shape
    if hmask not in (1, num_q_heads):
        raise ValueError(f"{context} Hmask must be 1 or Hq ({num_q_heads}); got {hmask}")
    if expected_hmask is not None and hmask != expected_hmask:
        raise ValueError(f"{context} Hmask={hmask} does not match forward plan Hmask={expected_hmask}")
    if tuple(full_cnt.shape) != tuple(mask_cnt.shape):
        raise ValueError("mask_block_cnt and full_block_cnt must have identical shapes")

    num_plan_rows = hmask * total_blocks
    assert tensors.mask_block_offset is not None
    assert tensors.full_block_offset is not None
    if tensors.mask_block_offset.numel() != num_plan_rows + 1:
        raise ValueError("mask_block_offset must have Hmask * total_blocks + 1 elements")
    if tensors.full_block_offset.numel() != num_plan_rows + 1:
        raise ValueError("full_block_offset must have Hmask * total_blocks + 1 elements")
    if not is_fake_mode() and tensors.mask_block_masks.data_ptr() % 16 != 0:
        raise ValueError("mask_block_masks must be 16-byte aligned")
    return hmask, total_blocks


def _validate_varlen_prefix(
    tensors: BlockSparseTensorsTorch,
    *,
    device: torch.device,
    batch_size: int,
    is_varlen: bool,
    context: str,
) -> None:
    cu_total_m_blocks = tensors.cu_total_m_blocks
    if not is_varlen:
        if cu_total_m_blocks is not None:
            raise ValueError(f"fixed {context} must not provide cu_total_m_blocks")
        return
    if cu_total_m_blocks is None:
        raise ValueError(f"varlen {context} requires cu_total_m_blocks")
    if (
        cu_total_m_blocks.dtype != torch.int32
        or cu_total_m_blocks.device != device
        or cu_total_m_blocks.ndim != 1
        or cu_total_m_blocks.numel() != batch_size + 1
        or not cu_total_m_blocks.is_contiguous()
    ):
        raise ValueError("cu_total_m_blocks must be contiguous int32 CUDA [B + 1]")


def normalize_arbitrary_block_sparse_config(
    tensors: BlockSparseTensorsTorch,
    *,
    device: torch.device,
    batch_size: int,
    num_q_heads: int,
    is_varlen: bool,
    block_size: tuple[int, int],
    pack_gqa: bool,
    physical_subtiles: int,
    num_mask_payload_groups: int,
    payload_padded_words: int,
    expected_fixed_total_m_blocks: int | None = None,
) -> BlockSparseTensorsTorch:
    """Validate a compact Q-to-K arbitrary-mask plan without device sync."""

    if type(physical_subtiles) is not int or physical_subtiles <= 0:
        raise ValueError("physical_subtiles must be a positive int")
    _, total_m_blocks = _validate_common_compact_plan(
        tensors,
        device=device,
        num_q_heads=num_q_heads,
        block_size=block_size,
        expected_hmask=None,
        context="arbitrary forward plan",
    )
    if tensors.pack_gqa is None or tensors.pack_gqa != pack_gqa:
        raise ValueError(f"arbitrary plan pack_gqa={tensors.pack_gqa} does not match " f"the resolved consumer pack_gqa={pack_gqa}")
    if type(tensors.narrow_workset) is not bool:
        raise TypeError("arbitrary forward plan requires a bool narrow_workset hint")
    if expected_fixed_total_m_blocks is not None and total_m_blocks != expected_fixed_total_m_blocks:
        raise ValueError(f"fixed arbitrary plan has total_m_blocks={total_m_blocks}; " f"expected {expected_fixed_total_m_blocks}")

    assert tensors.mask_block_masks is not None
    expected_payload_shape = (
        tensors.mask_block_idx.numel(),
        physical_subtiles,
        num_mask_payload_groups,
        payload_padded_words,
    )
    if tuple(tensors.mask_block_masks.shape) != expected_payload_shape:
        raise ValueError(
            "mask_block_masks has incompatible consumer layout: expected " f"{expected_payload_shape}, got {tuple(tensors.mask_block_masks.shape)}"
        )
    _validate_varlen_prefix(
        tensors,
        device=device,
        batch_size=batch_size,
        is_varlen=is_varlen,
        context="arbitrary forward plan",
    )
    is_plan_scheduled_forward = (
        tensors.plan_signature is not None and tensors.plan_signature.direction == "forward" and tensors.plan_signature.arch_family in ("sm90", "sm100")
    )
    if is_plan_scheduled_forward:
        _validate_required_tensors(
            {
                "fwd_work_desc": (tensors.fwd_work_desc, torch.int32, 2),
            },
            device=device,
            context="arbitrary forward schedule",
        )
        assert tensors.fwd_work_desc is not None
        if tensors.fwd_work_desc.shape[1] != 4:
            raise ValueError("fwd_work_desc must have shape [num_forward_tasks, 4]")
        num_scheduled_heads = num_q_heads // tensors.plan_signature.qhead_per_kvhead if pack_gqa else num_q_heads
        expected_tasks = total_m_blocks * num_scheduled_heads
        if tensors.fwd_work_desc.shape[0] != expected_tasks:
            raise ValueError("fwd_work_desc task count does not match compact Q rows and scheduled heads")
        if not is_fake_mode() and tensors.fwd_work_desc.data_ptr() % 16 != 0:
            raise ValueError("fwd_work_desc must be 16-byte aligned")
        needs_sequence_desc = is_varlen or tensors.plan_signature.kernel_family.startswith("sm100_hd256")
        if needs_sequence_desc:
            _validate_required_tensors(
                {
                    "sequence_desc": (tensors.sequence_desc, torch.int32, 2),
                },
                device=device,
                context="arbitrary varlen forward schedule",
            )
            assert tensors.sequence_desc is not None
            if tuple(tensors.sequence_desc.shape) != (batch_size, 8):
                raise ValueError("sequence_desc must have shape [B, 8]")
            if not is_fake_mode() and tensors.sequence_desc.data_ptr() % 16 != 0:
                raise ValueError("sequence_desc must be 16-byte aligned")
        elif tensors.sequence_desc is not None:
            raise ValueError("fixed generic forward must not provide sequence_desc")
    elif tensors.sequence_desc is not None or tensors.fwd_work_desc is not None:
        raise ValueError("forward schedule descriptors require a supported FWD plan")
    for name in (
        "dq_write_order",
        "dq_write_order_full",
        "spt",
    ):
        if getattr(tensors, name) is not None:
            raise ValueError(f"arbitrary forward plan requires {name}=None")
    return tensors


def normalize_arbitrary_block_sparse_config_bwd(
    tensors: BlockSparseTensorsTorch,
    *,
    device: torch.device,
    batch_size: int,
    num_q_heads: int,
    is_varlen: bool,
    block_size: tuple[int, int],
    subtile_factor: int,
    num_mma_threads: int,
    payload_padded_words: int,
    expected_hmask: int | None = None,
    expected_spt: bool = False,
    expected_fixed_total_n_blocks: int | None = None,
    require_dq_write_order: bool = True,
) -> BlockSparseTensorsTorch:
    """Validate a compact K-to-Q arbitrary-mask backward plan."""

    _, total_n_blocks = _validate_common_compact_plan(
        tensors,
        device=device,
        num_q_heads=num_q_heads,
        block_size=block_size,
        expected_hmask=expected_hmask,
        context="arbitrary backward plan",
    )
    if tensors.pack_gqa is not None:
        raise ValueError("arbitrary backward plan requires pack_gqa=None")
    if tensors.bwd_tensors is not None:
        raise ValueError("nested arbitrary backward plans are not supported")
    if expected_fixed_total_n_blocks is not None and total_n_blocks != expected_fixed_total_n_blocks:
        raise ValueError(f"fixed arbitrary backward plan has total_n_blocks={total_n_blocks}; " f"expected {expected_fixed_total_n_blocks}")

    if require_dq_write_order:
        _validate_required_tensors(
            {
                "dq_write_order": (tensors.dq_write_order, torch.int32, 1),
                "dq_write_order_full": (tensors.dq_write_order_full, torch.int32, 1),
            },
            device=device,
            context="arbitrary backward plan",
        )
        assert tensors.dq_write_order is not None
        assert tensors.dq_write_order_full is not None
        if tensors.dq_write_order.shape != tensors.mask_block_idx.shape:
            raise ValueError("dq_write_order must be parallel to mask_block_idx")
        if tensors.dq_write_order_full.shape != tensors.full_block_idx.shape:
            raise ValueError("dq_write_order_full must be parallel to full_block_idx")
    elif tensors.dq_write_order is not None or tensors.dq_write_order_full is not None:
        raise ValueError("this arbitrary backward consumer requires dq_write_order=None")

    assert tensors.mask_block_masks is not None
    expected_payload_shape = (
        tensors.mask_block_idx.numel(),
        subtile_factor,
        num_mma_threads,
        payload_padded_words,
    )
    if tuple(tensors.mask_block_masks.shape) != expected_payload_shape:
        raise ValueError(
            "mask_block_masks has incompatible backward consumer layout: expected " f"{expected_payload_shape}, got {tuple(tensors.mask_block_masks.shape)}"
        )
    _validate_varlen_prefix(
        tensors,
        device=device,
        batch_size=batch_size,
        is_varlen=is_varlen,
        context="arbitrary backward plan",
    )
    if require_dq_write_order:
        if tensors.spt is None or tensors.spt != expected_spt:
            raise ValueError(f"arbitrary backward plan spt={tensors.spt} does not match " f"consumer spt={expected_spt}")
    elif tensors.spt is not None:
        raise ValueError("this arbitrary backward consumer requires spt=None")
    return tensors


def to_cute_block_sparse_tensors(
    tensors: BlockSparseTensorsTorch,
) -> BlockSparseTensors:
    """Convert one validated arbitrary-mask plan from Torch to CuTe tensors."""

    def convert(tensor, *, align: int = 4, leading_dim: int = -1):
        return (
            to_cute_tensor(
                tensor,
                assumed_align=align,
                leading_dim=leading_dim,
            )
            if tensor is not None
            else None
        )

    return BlockSparseTensors(
        convert(tensors.mask_block_cnt),
        convert(tensors.mask_block_idx),
        convert(tensors.full_block_cnt),
        convert(tensors.full_block_idx),
        convert(tensors.cu_total_m_blocks, leading_dim=0),
        convert(tensors.dq_write_order),
        convert(tensors.dq_write_order_full),
        convert(tensors.mask_block_offset, leading_dim=0),
        convert(tensors.full_block_offset, leading_dim=0),
        convert(tensors.mask_block_masks, align=16),
        convert(tensors.sequence_desc, align=16),
        convert(tensors.fwd_work_desc, align=16),
    )
