# SPDX-License-Identifier: BSD-3-Clause
"""Shared SM100 forward consumer configuration and layout constructors.

The resolved dataclass contains host-side, hashable metadata only.  The CuTe
objects below are reconstructed by both the attention consumer and the plan
materializer so the mask payload is derived from the real TMEM load
partition instead of a hand-written lane formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import torch
from cutlass import Float32, cute
from cutlass.cute.nvgpu import tcgen05
from cudnn.flex_attention.plan.mask_plan import (
    ArbitraryPlanSignature,
    ArbitraryPlanTopology,
    canonical_blackwell_arch_family,
    resolve_arbitrary_pack_gqa,
)

_SM100_GENERIC_FWD_TILE_M = 128
_SM100_GENERIC_FWD_TILE_N = 128
_SM100_GENERIC_FWD_Q_STAGE = 2
_SM100_QSTAGE1_1CTA_Q_STAGE = 1
_SM100_QSTAGE1_1CTA_CTA_GROUP_SIZE = 1
_SM100_QSTAGE1_2CTA_Q_STAGE = 1
_SM100_QSTAGE1_2CTA_CTA_GROUP_SIZE = 2
_SM100_SOFTMAX_THREADS_PER_SUBTILE = 128
_SM100_TMEM_LOAD_ATOM_ID = "tcgen05.ld32x32b.r32"
_SM100_FWD_PAYLOAD_VALUES_PER_THREAD = 128
SM100_FWD_MASK_PAYLOAD_WORDS = 4


def sm100_generic_fwd_shape_is_supported(head_dim: int, head_dim_v: int) -> bool:
    """Return whether the generic SM100 forward family owns this head shape."""

    is_standard_shape = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128 and head_dim % 8 == 0 and head_dim_v % 8 == 0
    return is_standard_shape or (head_dim == 192 and head_dim_v == 128)


def _validate_sm100_generic_fwd_shape(
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
) -> None:
    """Validate the head shapes shared by qstage2 and qstage1 kernels."""

    if not sm100_generic_fwd_shape_is_supported(head_dim, head_dim_v):
        raise NotImplementedError("generic SM100 arbitrary forward supports D/Dv in [8, 128] " "or (192, 128); " f"got ({head_dim}, {head_dim_v})")
    alignment = 16 // torch.empty((), dtype=dtype).element_size()
    if head_dim % alignment != 0 or head_dim_v % alignment != 0:
        raise ValueError(f"head_dim and head_dim_v must be divisible by {alignment} for {dtype}")


@dataclass(frozen=True)
class _ResolvedSm100FwdConsumerConfig:
    arch: int
    dtype: torch.dtype
    head_dim: int
    head_dim_v: int
    num_q_heads: int
    num_kv_heads: int
    qhead_per_kvhead: int
    is_varlen: bool
    pack_gqa: bool
    tile_m: int
    tile_n: int
    q_stage: int
    cta_group_size: int
    cluster_axis: str
    physical_subtiles: int
    softmax_threads_per_subtile: int
    num_mask_payload_groups: int
    payload_values_per_thread: int
    payload_valid_words: int
    payload_padded_words: int
    tmem_load_atom_id: str
    payload_layout_id: str

    @property
    def topology(self) -> ArbitraryPlanTopology:
        return ArbitraryPlanTopology(
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            q_stage=self.q_stage,
            cta_group_size=self.cta_group_size,
            pack_gqa=self.pack_gqa,
            qhead_per_kvhead=self.qhead_per_kvhead,
            cluster_axis=self.cluster_axis,
        )

    @property
    def block_size(self) -> tuple[int, int]:
        return self.topology.block_size

    @property
    def topology_planner_compile_key(self) -> tuple:
        qhead_ratio = self.qhead_per_kvhead if self.pack_gqa else 1
        return (
            canonical_blackwell_arch_family(self.arch),
            self.tile_m,
            self.tile_n,
            self.is_varlen,
            self.pack_gqa,
            qhead_ratio,
            self.q_stage,
            self.cta_group_size,
            self.cluster_axis,
        )

    @property
    def payload_planner_compile_key(self) -> tuple:
        return (
            self.topology_planner_compile_key,
            self.dtype,
            self.tmem_load_atom_id,
            self.payload_layout_id,
            self.num_mask_payload_groups,
            self.payload_valid_words,
            self.payload_padded_words,
        )

    @property
    def plan_signature(self) -> ArbitraryPlanSignature:
        topology = self.topology
        is_qstage1_1cta = self.q_stage == 1 and self.cta_group_size == 1
        is_qstage1_2cta = self.q_stage == 1 and self.cta_group_size == 2
        kernel_family = "sm100_qstage1_1cta_fwd" if is_qstage1_1cta else ("sm100_qstage1_2cta_fwd" if is_qstage1_2cta else "sm100_generic_fwd")
        return ArbitraryPlanSignature(
            arch_family=canonical_blackwell_arch_family(self.arch),
            direction="forward",
            kernel_family=kernel_family,
            tile_m=topology.tile_m,
            tile_n=topology.tile_n,
            q_stage=topology.q_stage,
            cta_group_size=topology.cta_group_size,
            pack_gqa=topology.pack_gqa,
            qhead_per_kvhead=topology.qhead_per_kvhead,
            mma_atom_layout_id=(f"tcgen05_f32_ss_qk_cta{self.cta_group_size}" f"_m{self.tile_m * self.cta_group_size}" f"n{self.tile_n}_major_kk"),
            swap_ab=False,
            payload_layout_id=self.payload_layout_id,
            dq_order_format="none",
            cluster_axis=topology.cluster_axis,
            scheduler_layout_id="plan_fwd_work_desc_i32x4_v1",
        )


def resolve_sm100_fwd_consumer_config(
    *,
    arch: int,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    num_q_heads: int,
    num_kv_heads: int,
    is_varlen: bool,
    hmask: int,
    pack_gqa: bool | None,
) -> _ResolvedSm100FwdConsumerConfig:
    """Resolve the generic SM100 arbitrary-forward 1CTA consumer signature."""

    if arch not in (100, 103):
        raise NotImplementedError("SM100 arbitrary forward config supports SM100/SM103 only")
    if dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("SM100 arbitrary forward supports FP16 and BF16 only")
    if type(is_varlen) is not bool:
        raise TypeError("is_varlen must be a bool")
    _validate_sm100_generic_fwd_shape(dtype, head_dim, head_dim_v)
    tile_m = _SM100_GENERIC_FWD_TILE_M
    tile_n = _SM100_GENERIC_FWD_TILE_N
    effective_pack_gqa, qhead_per_kvhead = resolve_arbitrary_pack_gqa(
        requested_pack_gqa=pack_gqa,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        hmask=hmask,
        tile_m=tile_m,
    )
    # Keep the generic SM100 forward topology independent of runtime sequence
    # lengths and GQA geometry. Each CTA owns two M128 Q subtiles.
    q_stage = _SM100_GENERIC_FWD_Q_STAGE
    cta_group_size = 1
    cluster_axis = "m"
    physical_subtiles = q_stage

    tmem_load_atom_id = _SM100_TMEM_LOAD_ATOM_ID
    payload_layout_id = (
        "sm100_tcgen05_qk" f"_ld32x32b_r32_t{_SM100_SOFTMAX_THREADS_PER_SUBTILE}" f"_v{_SM100_FWD_PAYLOAD_VALUES_PER_THREAD}_w{SM100_FWD_MASK_PAYLOAD_WORDS}_v1"
    )

    return _ResolvedSm100FwdConsumerConfig(
        arch=arch,
        dtype=dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qhead_per_kvhead=qhead_per_kvhead,
        is_varlen=is_varlen,
        pack_gqa=effective_pack_gqa,
        tile_m=tile_m,
        tile_n=tile_n,
        q_stage=q_stage,
        cta_group_size=cta_group_size,
        cluster_axis=cluster_axis,
        physical_subtiles=physical_subtiles,
        softmax_threads_per_subtile=_SM100_SOFTMAX_THREADS_PER_SUBTILE,
        num_mask_payload_groups=_SM100_SOFTMAX_THREADS_PER_SUBTILE,
        payload_values_per_thread=_SM100_FWD_PAYLOAD_VALUES_PER_THREAD,
        payload_valid_words=SM100_FWD_MASK_PAYLOAD_WORDS,
        payload_padded_words=SM100_FWD_MASK_PAYLOAD_WORDS,
        tmem_load_atom_id=tmem_load_atom_id,
        payload_layout_id=payload_layout_id,
    )


def resolve_sm100_fwd_qstage1_2cta_consumer_config(
    *,
    arch: int,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    num_q_heads: int,
    num_kv_heads: int,
    is_varlen: bool,
    hmask: int,
    pack_gqa: bool | None,
) -> _ResolvedSm100FwdConsumerConfig:
    """Resolve the generic 1-q-stage, 2CTA forward candidate."""

    if arch not in (100, 103):
        raise NotImplementedError("generic qstage1 2CTA arbitrary forward supports SM100/SM103 only")
    if dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("generic qstage1 2CTA arbitrary forward supports FP16 and BF16 only")
    if type(is_varlen) is not bool:
        raise TypeError("is_varlen must be a bool")
    _validate_sm100_generic_fwd_shape(dtype, head_dim, head_dim_v)

    tile_m = _SM100_GENERIC_FWD_TILE_M
    tile_n = _SM100_GENERIC_FWD_TILE_N
    effective_pack_gqa, qhead_per_kvhead = resolve_arbitrary_pack_gqa(
        requested_pack_gqa=pack_gqa,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        hmask=hmask,
        tile_m=tile_m,
    )
    q_stage = _SM100_QSTAGE1_2CTA_Q_STAGE
    cta_group_size = _SM100_QSTAGE1_2CTA_CTA_GROUP_SIZE
    payload_layout_id = (
        "sm100_tcgen05_qk"
        f"_ld32x32b_r32_t{_SM100_SOFTMAX_THREADS_PER_SUBTILE}"
        f"_v{_SM100_FWD_PAYLOAD_VALUES_PER_THREAD}_w{SM100_FWD_MASK_PAYLOAD_WORDS}"
        f"_m{tile_m * cta_group_size}n{tile_n}"
        f"_ctarank{cta_group_size}_axis_m_qstage1_v1"
    )
    return _ResolvedSm100FwdConsumerConfig(
        arch=arch,
        dtype=dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qhead_per_kvhead=qhead_per_kvhead,
        is_varlen=is_varlen,
        pack_gqa=effective_pack_gqa,
        tile_m=tile_m,
        tile_n=tile_n,
        q_stage=q_stage,
        cta_group_size=cta_group_size,
        cluster_axis="m",
        physical_subtiles=q_stage * cta_group_size,
        softmax_threads_per_subtile=_SM100_SOFTMAX_THREADS_PER_SUBTILE,
        num_mask_payload_groups=_SM100_SOFTMAX_THREADS_PER_SUBTILE,
        payload_values_per_thread=_SM100_FWD_PAYLOAD_VALUES_PER_THREAD,
        payload_valid_words=SM100_FWD_MASK_PAYLOAD_WORDS,
        payload_padded_words=SM100_FWD_MASK_PAYLOAD_WORDS,
        tmem_load_atom_id="tcgen05.ld32x32b.r32.cta_group2",
        payload_layout_id=payload_layout_id,
    )


def resolve_sm100_fwd_qstage1_1cta_consumer_config(
    *,
    arch: int,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    num_q_heads: int,
    num_kv_heads: int,
    is_varlen: bool,
    hmask: int,
    pack_gqa: bool | None,
) -> _ResolvedSm100FwdConsumerConfig:
    """Resolve the generic 1-q-stage, 1CTA forward candidate."""

    if arch not in (100, 103):
        raise NotImplementedError("generic qstage1 1CTA arbitrary forward supports SM100/SM103 only")
    if dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("generic qstage1 1CTA arbitrary forward supports FP16 and BF16 only")
    if type(is_varlen) is not bool:
        raise TypeError("is_varlen must be a bool")
    _validate_sm100_generic_fwd_shape(dtype, head_dim, head_dim_v)

    tile_m = _SM100_GENERIC_FWD_TILE_M
    tile_n = _SM100_GENERIC_FWD_TILE_N
    effective_pack_gqa, qhead_per_kvhead = resolve_arbitrary_pack_gqa(
        requested_pack_gqa=pack_gqa,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        hmask=hmask,
        tile_m=tile_m,
    )
    q_stage = _SM100_QSTAGE1_1CTA_Q_STAGE
    cta_group_size = _SM100_QSTAGE1_1CTA_CTA_GROUP_SIZE
    payload_layout_id = (
        "sm100_tcgen05_qk"
        f"_ld32x32b_r32_t{_SM100_SOFTMAX_THREADS_PER_SUBTILE}"
        f"_v{_SM100_FWD_PAYLOAD_VALUES_PER_THREAD}_w{SM100_FWD_MASK_PAYLOAD_WORDS}"
        f"_m{tile_m}n{tile_n}_cta1_axis_m_qstage1_v1"
    )
    return _ResolvedSm100FwdConsumerConfig(
        arch=arch,
        dtype=dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qhead_per_kvhead=qhead_per_kvhead,
        is_varlen=is_varlen,
        pack_gqa=effective_pack_gqa,
        tile_m=tile_m,
        tile_n=tile_n,
        q_stage=q_stage,
        cta_group_size=cta_group_size,
        cluster_axis="m",
        physical_subtiles=1,
        softmax_threads_per_subtile=_SM100_SOFTMAX_THREADS_PER_SUBTILE,
        num_mask_payload_groups=_SM100_SOFTMAX_THREADS_PER_SUBTILE,
        payload_values_per_thread=_SM100_FWD_PAYLOAD_VALUES_PER_THREAD,
        payload_valid_words=SM100_FWD_MASK_PAYLOAD_WORDS,
        payload_padded_words=SM100_FWD_MASK_PAYLOAD_WORDS,
        tmem_load_atom_id=_SM100_TMEM_LOAD_ATOM_ID,
        payload_layout_id=payload_layout_id,
    )


@cute.jit
def make_sm100_fwd_tiled_mma_qk(
    dtype: type[cutlass.Numeric],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    cta_group_size: cutlass.Constexpr[int] = 1,
):
    """Build the QK MMA layout shared by planner and forward consumers."""

    if cutlass.const_expr(cta_group_size not in (1, 2)):
        raise ValueError("SM100 forward cta_group_size must be one or two")
    cta_group = tcgen05.CtaGroup.TWO if cutlass.const_expr(cta_group_size == 2) else tcgen05.CtaGroup.ONE

    return sm100_utils_basic.make_trivial_tiled_mma(
        dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        Float32,
        cta_group,
        (tile_m * cta_group_size, tile_n),
    )


@cute.jit
def make_sm100_fwd_tmem_load(
    tSAcc: cute.Tensor,
    tidx: cutlass.Int32,
    use_ldred: cutlass.Constexpr[bool] = False,
):
    """Build the score TMEM-to-register copy used by softmax.

    The plan materializer keeps the default load because ``ld.red`` preserves
    the score fragment layout and only adds a reduction side output.
    """

    tmem_load_op = (
        tcgen05.copy.LdRed32x32bOp(tcgen05.copy.Repetition(32)) if cutlass.const_expr(use_ldred) else tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))
    )
    tmem_load_atom = cute.make_copy_atom(tmem_load_op, Float32)
    return tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)


__all__ = [
    "SM100_FWD_MASK_PAYLOAD_WORDS",
    "_ResolvedSm100FwdConsumerConfig",
    "make_sm100_fwd_tiled_mma_qk",
    "make_sm100_fwd_tmem_load",
    "resolve_sm100_fwd_consumer_config",
    "resolve_sm100_fwd_qstage1_1cta_consumer_config",
    "resolve_sm100_fwd_qstage1_2cta_consumer_config",
    "sm100_generic_fwd_shape_is_supported",
]
