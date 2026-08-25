# SPDX-License-Identifier: BSD-3-Clause
"""Shared SM100 backward consumer configuration for mask planning.

The arbitrary mask payload must follow the score TMEM-to-register ownership
used by the eight compute warps in :mod:`flash_bwd_sm100`.  Keeping the layout
constructors here lets the planner rebuild the native consumer partition
without duplicating a lane formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import torch
from cutlass import Float32, cute
from cutlass.cute.nvgpu import tcgen05
from cudnn.flex_attention.kernels.common import copy_utils
from cudnn.flex_attention.plan.mask_plan import (
    ArbitraryPlanSignature,
    ArbitraryPlanTopology,
    canonical_blackwell_arch_family,
)

_SM100_GENERIC_BWD_TILE_M = 128
_SM100_GENERIC_BWD_TILE_N = 128
_SM100_BWD_COMPUTE_WARPS = 8
_SM100_BWD_CONSUMER_THREADS = _SM100_BWD_COMPUTE_WARPS * 32
_SM100_BWD_TMEM_LOAD_ATOM_ID = "tcgen05.ld32x32b.r32.wg2"
SM100_BWD_MASK_PAYLOAD_WORDS = 2


@dataclass(frozen=True)
class _ResolvedSm100BwdConsumerConfig:
    arch: int
    dtype: torch.dtype
    head_dim: int
    head_dim_v: int
    num_q_heads: int
    num_kv_heads: int
    qhead_per_kvhead: int
    is_varlen: bool
    tile_m: int
    tile_n: int
    sparse_tile_m: int
    sparse_tile_n: int
    subtile_factor: int
    physical_subtiles: int
    cta_group_size: int
    cluster_axis: str
    num_wg: int
    num_mma_threads: int
    attention_num_threads: int
    payload_values_per_thread: int
    payload_valid_words: int
    payload_padded_words: int
    tmem_load_atom_id: str
    payload_layout_id: str
    spt: bool

    @property
    def topology(self) -> ArbitraryPlanTopology:
        return ArbitraryPlanTopology(
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            q_stage=self.subtile_factor,
            cta_group_size=self.cta_group_size,
            pack_gqa=False,
            qhead_per_kvhead=self.qhead_per_kvhead,
            cluster_axis=self.cluster_axis,
        )

    @property
    def block_size(self) -> tuple[int, int]:
        return self.topology.block_size

    @property
    def topology_planner_compile_key(self) -> tuple:
        return (
            canonical_blackwell_arch_family(self.arch),
            self.tile_m,
            self.tile_n,
            self.is_varlen,
            False,
            self.qhead_per_kvhead,
            self.subtile_factor,
            self.cta_group_size,
            self.cluster_axis,
            self.sparse_tile_n,
        )

    @property
    def planner_compile_key(self) -> tuple:
        return (
            self.arch,
            self.topology_planner_compile_key,
            self.dtype,
            self.head_dim,
            self.head_dim_v,
            self.tmem_load_atom_id,
            self.payload_layout_id,
            self.num_mma_threads,
            self.payload_valid_words,
            self.payload_padded_words,
            self.spt,
        )

    @property
    def plan_signature(self) -> ArbitraryPlanSignature:
        topology = self.topology
        return ArbitraryPlanSignature(
            arch_family=canonical_blackwell_arch_family(self.arch),
            direction="backward",
            kernel_family="sm100_generic_bwd",
            tile_m=topology.tile_m,
            tile_n=topology.tile_n,
            q_stage=topology.q_stage,
            cta_group_size=topology.cta_group_size,
            pack_gqa=topology.pack_gqa,
            qhead_per_kvhead=topology.qhead_per_kvhead,
            mma_atom_layout_id=(f"tcgen05_f32_ss_kq_cta{topology.cta_group_size}" f"_m{self.sparse_tile_n}n{self.tile_m}_major_kk"),
            swap_ab=True,
            payload_layout_id=self.payload_layout_id,
            dq_order_format="rank_only",
            cluster_axis=self.cluster_axis,
        )


def resolve_sm100_bwd_consumer_config(
    *,
    arch: int,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    num_q_heads: int,
    num_kv_heads: int,
    is_varlen: bool,
    subtile_factor: int | None = None,
) -> _ResolvedSm100BwdConsumerConfig:
    """Resolve a generic SM100 backward topology.

    Exact D128 and (D, Dv) == (192, 128) use the cooperative topology.
    Smaller D/Dv configurations retain the native 1CTA topology.
    """

    if canonical_blackwell_arch_family(arch) != "sm100":
        raise NotImplementedError("SM100/SM103 arbitrary backward config supports SM100/SM103 only")
    if dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("SM100/SM103 arbitrary backward supports FP16 and BF16 only")
    is_standard_shape = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128
    is_d192_v128 = head_dim == 192 and head_dim_v == 128
    if not (is_standard_shape or is_d192_v128):
        raise NotImplementedError("generic SM100 arbitrary backward supports D/Dv <= 128 or (192, 128); " f"got ({head_dim}, {head_dim_v})")
    alignment = 16 // torch.empty((), dtype=dtype).element_size()
    if head_dim % alignment != 0 or head_dim_v % alignment != 0:
        raise ValueError(f"head_dim and head_dim_v must be divisible by {alignment} for {dtype}")
    if num_kv_heads <= 0 or num_q_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if subtile_factor is None:
        subtile_factor = 1
    if subtile_factor != 1:
        raise NotImplementedError("SM100/SM103 arbitrary backward currently requires subtile_factor=1")

    is_d128 = head_dim == 128 and head_dim_v == 128
    requires_cooperative_ctas = is_d192_v128 or is_d128
    tile_m = _SM100_GENERIC_BWD_TILE_M
    tile_n = _SM100_GENERIC_BWD_TILE_N
    cta_group_size = 2 if requires_cooperative_ctas else 1
    cluster_axis = "n"
    sparse_tile_n = tile_n * cta_group_size
    payload_values_per_thread, remainder = divmod(
        tile_m * sparse_tile_n,
        _SM100_BWD_CONSUMER_THREADS * cta_group_size,
    )
    if remainder:
        raise AssertionError("SM100 backward score values must partition evenly across compute threads")
    payload_valid_words = math.ceil(payload_values_per_thread / 32)
    payload_padded_words = SM100_BWD_MASK_PAYLOAD_WORDS
    if payload_valid_words > payload_padded_words:
        raise AssertionError("SM100 backward payload exceeds the packed vector")
    payload_layout_id = (
        "sm100_tcgen05_bwd_sdp"
        f"_ld32x32b_r32_wg2_t{_SM100_BWD_CONSUMER_THREADS}"
        f"_v{payload_values_per_thread}_w{payload_padded_words}"
        + (f"_k{sparse_tile_n}_ctarank{cta_group_size}_axis{cluster_axis}_v2" if requires_cooperative_ctas else "_v1")
    )

    return _ResolvedSm100BwdConsumerConfig(
        arch=arch,
        dtype=dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qhead_per_kvhead=num_q_heads // num_kv_heads,
        is_varlen=is_varlen,
        tile_m=tile_m,
        tile_n=tile_n,
        sparse_tile_m=tile_m,
        sparse_tile_n=sparse_tile_n,
        subtile_factor=subtile_factor,
        physical_subtiles=subtile_factor * cta_group_size,
        cta_group_size=cta_group_size,
        cluster_axis=cluster_axis,
        num_wg=2,
        num_mma_threads=_SM100_BWD_CONSUMER_THREADS,
        attention_num_threads=512,
        payload_values_per_thread=payload_values_per_thread,
        payload_valid_words=payload_valid_words,
        payload_padded_words=payload_padded_words,
        tmem_load_atom_id=_SM100_BWD_TMEM_LOAD_ATOM_ID,
        payload_layout_id=payload_layout_id,
        # Deterministic mode uses the native descending-K scheduler.
        spt=True,
    )


@cute.jit
def make_sm100_bwd_tiled_mma_sdp(
    dtype: type[cutlass.Numeric],
    tile_m: cutlass.Constexpr[int],
    sparse_tile_n: cutlass.Constexpr[int],
    cta_group_size: cutlass.Constexpr[int] = 1,
):
    """Build the KQ MMA layout shared by the planner and resolved consumer."""

    return sm100_utils_basic.make_trivial_tiled_mma(
        dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        Float32,
        tcgen05.CtaGroup.TWO if cta_group_size == 2 else tcgen05.CtaGroup.ONE,
        (sparse_tile_n, tile_m),
    )


@cute.jit
def make_sm100_bwd_tmem_load(
    consumer_tidx: cutlass.Int32,
    num_wg: cutlass.Constexpr[int] = 2,
):
    """Build the exact score TMEM-to-register copy used by compute_loop."""

    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
        Float32,
    )
    return copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(consumer_tidx)


__all__ = [
    "SM100_BWD_MASK_PAYLOAD_WORDS",
    "_ResolvedSm100BwdConsumerConfig",
    "make_sm100_bwd_tiled_mma_sdp",
    "make_sm100_bwd_tmem_load",
    "resolve_sm100_bwd_consumer_config",
]
