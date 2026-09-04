# SPDX-License-Identifier: BSD-3-Clause
"""Resolved consumer contract for the dedicated SM100 hd256 forward kernel.

The dedicated kernel supports one- and two-CTA ``M128 x N128`` physical score
subtiles.  Each CTA uses 128 softmax threads for its TMEM slice. ``q_stage``
remains one; ``cta_group_size`` selects a logical M128 or M256 consumer tile.
The two D256/K128 MMA K phases are internal and do not add payload planes.

Only hashable host metadata lives in the resolved dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cudnn.flex_attention.plan.mask_plan import (
    ArbitraryPlanSignature,
    ArbitraryPlanTopology,
    canonical_blackwell_arch_family,
)

_SM100_HD256_FWD_TILE_M = 128
_SM100_HD256_FWD_TILE_N = 128
_SM100_HD256_FWD_Q_STAGE = 1
_SM100_HD256_SOFTMAX_THREADS_PER_SUBTILE = 128
_SM100_HD256_PAYLOAD_VALUES_PER_THREAD = 128
SM100_HD256_FWD_MASK_PAYLOAD_WORDS = 4


@dataclass(frozen=True)
class _ResolvedSm100Hd256FwdConsumerConfig:
    """Hashable contract for one hd256 arbitrary-mask consumer."""

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
    def kernel_family(self) -> str:
        return "sm100_hd256_qstage1_2cta_fwd" if self.cta_group_size == 2 else "sm100_hd256_fwd"

    @property
    def topology_planner_compile_key(self) -> tuple:
        """Compilation key for the family-compatible Q2K classifier."""

        return (
            canonical_blackwell_arch_family(self.arch),
            self.tile_m,
            self.tile_n,
            self.is_varlen,
            self.pack_gqa,
            self.qhead_per_kvhead,
            self.q_stage,
            self.cta_group_size,
            self.cluster_axis,
        )

    @property
    def payload_planner_compile_key(self) -> tuple:
        """Compilation key for the consumer-native payload materializer."""

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
        return ArbitraryPlanSignature(
            arch_family=canonical_blackwell_arch_family(self.arch),
            direction="forward",
            kernel_family=self.kernel_family,
            tile_m=topology.tile_m,
            tile_n=topology.tile_n,
            q_stage=topology.q_stage,
            cta_group_size=topology.cta_group_size,
            pack_gqa=topology.pack_gqa,
            qhead_per_kvhead=topology.qhead_per_kvhead,
            mma_atom_layout_id=(f"tcgen05_f32_ss_qk_cta{topology.cta_group_size}" f"_m{topology.block_size[0]}" f"n{topology.block_size[1]}_major_kk"),
            swap_ab=False,
            payload_layout_id=self.payload_layout_id,
            dq_order_format="none",
            cluster_axis=topology.cluster_axis,
            scheduler_layout_id="plan_fwd_work_desc_i32x4_v1",
        )


def resolve_sm100_hd256_fwd_consumer_config(
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
    cta_group_size: int = 1,
) -> _ResolvedSm100Hd256FwdConsumerConfig:
    """Resolve the SM100/SM103 hd256 forward consumer."""

    if arch not in (100, 103):
        raise NotImplementedError("dedicated hd256 arbitrary forward supports SM100/SM103 only")
    if dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("dedicated hd256 arbitrary forward supports FP16 and BF16 only")
    if head_dim != 256 or head_dim_v != 256:
        raise NotImplementedError(f"dedicated hd256 arbitrary forward requires D=Dv=256; got ({head_dim}, {head_dim_v})")
    if type(is_varlen) is not bool:
        raise TypeError("is_varlen must be a bool")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if hmask not in (1, num_q_heads):
        raise ValueError(f"Hmask must be 1 or Hq ({num_q_heads}); got {hmask}")
    if pack_gqa not in (None, False):
        raise NotImplementedError("dedicated hd256 arbitrary forward requires pack_gqa=False")
    if cta_group_size not in (1, 2):
        raise ValueError("dedicated hd256 arbitrary forward requires cta_group_size in (1, 2)")

    tile_m = _SM100_HD256_FWD_TILE_M
    tile_n = _SM100_HD256_FWD_TILE_N
    q_stage = _SM100_HD256_FWD_Q_STAGE
    physical_subtiles = q_stage * cta_group_size

    payload_layout_id = (
        "sm100_hd256_tcgen05_qk"
        f"_ld32x32b_r32_rep32_t{_SM100_HD256_SOFTMAX_THREADS_PER_SUBTILE}"
        f"_v{_SM100_HD256_PAYLOAD_VALUES_PER_THREAD}_w{SM100_HD256_FWD_MASK_PAYLOAD_WORDS}"
        f"_m{tile_m}n{tile_n}"
        f"_ctarank{cta_group_size}_axis_m_v2"
    )

    return _ResolvedSm100Hd256FwdConsumerConfig(
        arch=arch,
        dtype=dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qhead_per_kvhead=num_q_heads // num_kv_heads,
        is_varlen=is_varlen,
        pack_gqa=False,
        tile_m=tile_m,
        tile_n=tile_n,
        q_stage=q_stage,
        cta_group_size=cta_group_size,
        cluster_axis="m",
        physical_subtiles=physical_subtiles,
        softmax_threads_per_subtile=_SM100_HD256_SOFTMAX_THREADS_PER_SUBTILE,
        num_mask_payload_groups=_SM100_HD256_SOFTMAX_THREADS_PER_SUBTILE,
        payload_values_per_thread=_SM100_HD256_PAYLOAD_VALUES_PER_THREAD,
        payload_valid_words=SM100_HD256_FWD_MASK_PAYLOAD_WORDS,
        payload_padded_words=SM100_HD256_FWD_MASK_PAYLOAD_WORDS,
        tmem_load_atom_id=(f"tcgen05.ld32x32b.r32.rep32.cta_group{cta_group_size}"),
        payload_layout_id=payload_layout_id,
    )


__all__ = [
    "SM100_HD256_FWD_MASK_PAYLOAD_WORDS",
    "_ResolvedSm100Hd256FwdConsumerConfig",
    "resolve_sm100_hd256_fwd_consumer_config",
]
