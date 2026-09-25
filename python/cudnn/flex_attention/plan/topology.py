# SPDX-License-Identifier: BSD-3-Clause
"""Consumer-specific topology adapters for arbitrary-mask planning."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cudnn.flex_attention.kernels.sm90.bwd.backward_config import _ResolvedSm90BwdConsumerConfig
from cudnn.flex_attention.kernels.sm100.bwd.backward_config import _ResolvedSm100BwdConsumerConfig
from cudnn.flex_attention.kernels.sm100.bwd.backward_config_hd256 import (
    _ResolvedSm100Hd256DkdvConsumerConfig,
    _ResolvedSm100Hd256DqConsumerConfig,
)
from cudnn.flex_attention.kernels.sm100.fwd.forward_config import _ResolvedSm100FwdConsumerConfig
from cudnn.flex_attention.kernels.sm100.fwd.forward_config_hd256 import (
    _ResolvedSm100Hd256FwdConsumerConfig,
)
from cudnn.flex_attention.plan.mask_plan import ArbitraryPlanSignature


def _consumer_plan_signature(config) -> ArbitraryPlanSignature:
    """Return versioned metadata for one resolved consumer config."""

    signature = getattr(config, "plan_signature", None)
    if signature is not None:
        return signature
    if isinstance(config, _ResolvedSm90BwdConsumerConfig):
        return ArbitraryPlanSignature(
            arch_family="sm90",
            direction="backward",
            kernel_family="sm90_generic_bwd",
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            q_stage=config.physical_subtiles,
            cta_group_size=1,
            pack_gqa=False,
            qhead_per_kvhead=config.qhead_per_kvhead,
            mma_atom_layout_id=f"sm90_wgmma_f32_ss_sdp_m{config.tile_m}n{config.tile_n}_atom_m1_wg2_major_kk",
            swap_ab=config.sdp_swap_ab,
            payload_layout_id=(f"sm90_wgmma_sdp_t256_s{config.subtile_factor}_swap{int(config.sdp_swap_ab)}_w{config.payload_padded_words}_v1"),
            dq_order_format="rank_only",
            cluster_axis="m",
        )
    raise TypeError(f"unsupported arbitrary consumer config: {type(config).__name__}")


@dataclass(frozen=True)
class _ResolvedSm90BwdTopologyConfig:
    """Adapt the backward sparse Q tile to the shared topology classifier."""

    consumer: _ResolvedSm90BwdConsumerConfig

    @property
    def arch(self) -> int:
        return self.consumer.arch

    @property
    def dtype(self) -> torch.dtype:
        return self.consumer.dtype

    @property
    def tile_m(self) -> int:
        return self.consumer.sparse_tile_m

    @property
    def tile_n(self) -> int:
        return self.consumer.tile_n

    @property
    def pack_gqa(self) -> bool:
        return False

    @property
    def qhead_per_kvhead(self) -> int:
        return self.consumer.qhead_per_kvhead

    @property
    def num_mma_threads(self) -> int:
        return self.consumer.num_mma_threads

    @property
    def num_mask_payload_groups(self) -> int:
        return self.consumer.num_mma_threads

    @property
    def payload_values_per_thread(self) -> int:
        return self.consumer.payload_values_per_thread

    @property
    def payload_valid_words(self) -> int:
        return self.consumer.payload_valid_words

    @property
    def payload_padded_words(self) -> int:
        return self.consumer.payload_padded_words

    @property
    def is_varlen(self) -> bool:
        return self.consumer.is_varlen

    @property
    def topology_planner_compile_key(self) -> tuple:
        return (
            self.consumer.arch,
            self.tile_m,
            self.tile_n,
            self.is_varlen,
            False,
            1,
        )


@dataclass(frozen=True)
class _ResolvedSm100BwdTopologyConfig:
    """Adapt the SM100 K2Q tile to the shared topology classifier."""

    consumer: _ResolvedSm100BwdConsumerConfig

    @property
    def arch(self) -> int:
        return self.consumer.arch

    @property
    def dtype(self) -> torch.dtype:
        return self.consumer.dtype

    @property
    def tile_m(self) -> int:
        return self.consumer.sparse_tile_m

    @property
    def tile_n(self) -> int:
        # Backward 2CTA cooperation expands the K/N axis.  The topology
        # classifier and K2Q row lookup therefore operate on the cluster-union
        # K tile rather than either CTA's physical K128 half.
        return self.consumer.sparse_tile_n

    @property
    def pack_gqa(self) -> bool:
        return False

    @property
    def qhead_per_kvhead(self) -> int:
        return self.consumer.qhead_per_kvhead

    @property
    def num_mma_threads(self) -> int:
        return self.consumer.num_mma_threads

    @property
    def num_mask_payload_groups(self) -> int:
        return self.consumer.num_mma_threads

    @property
    def payload_values_per_thread(self) -> int:
        return self.consumer.payload_values_per_thread

    @property
    def payload_valid_words(self) -> int:
        return self.consumer.payload_valid_words

    @property
    def payload_padded_words(self) -> int:
        return self.consumer.payload_padded_words

    @property
    def is_varlen(self) -> bool:
        return self.consumer.is_varlen

    @property
    def topology_planner_compile_key(self) -> tuple:
        return self.consumer.topology_planner_compile_key


@dataclass(frozen=True)
class _ResolvedSm100FwdTopologyConfig:
    """Expose the q-stage union tile to the architecture-neutral classifier."""

    consumer: _ResolvedSm100FwdConsumerConfig

    @property
    def arch(self) -> int:
        return self.consumer.arch

    @property
    def dtype(self) -> torch.dtype:
        return self.consumer.dtype

    @property
    def tile_m(self) -> int:
        return self.consumer.block_size[0]

    @property
    def tile_n(self) -> int:
        return self.consumer.tile_n

    @property
    def pack_gqa(self) -> bool:
        return self.consumer.pack_gqa

    @property
    def qhead_per_kvhead(self) -> int:
        return self.consumer.qhead_per_kvhead

    @property
    def num_mma_threads(self) -> int:
        return self.consumer.softmax_threads_per_subtile

    @property
    def num_mask_payload_groups(self) -> int:
        return self.consumer.num_mask_payload_groups

    @property
    def payload_values_per_thread(self) -> int:
        return self.consumer.payload_values_per_thread

    @property
    def payload_valid_words(self) -> int:
        return self.consumer.payload_valid_words

    @property
    def payload_padded_words(self) -> int:
        return self.consumer.payload_padded_words

    @property
    def is_varlen(self) -> bool:
        return self.consumer.is_varlen

    @property
    def topology_planner_compile_key(self) -> tuple:
        return self.consumer.topology_planner_compile_key


@dataclass(frozen=True)
class _ResolvedSm100Hd256FwdTopologyConfig:
    """Expose the dedicated 1CTA Q128 tile to the classifier."""

    consumer: _ResolvedSm100Hd256FwdConsumerConfig

    @property
    def arch(self) -> int:
        return self.consumer.arch

    @property
    def dtype(self) -> torch.dtype:
        return self.consumer.dtype

    @property
    def tile_m(self) -> int:
        return self.consumer.block_size[0]

    @property
    def tile_n(self) -> int:
        return self.consumer.tile_n

    @property
    def pack_gqa(self) -> bool:
        return self.consumer.pack_gqa

    @property
    def qhead_per_kvhead(self) -> int:
        return self.consumer.qhead_per_kvhead

    @property
    def num_mma_threads(self) -> int:
        return self.consumer.softmax_threads_per_subtile

    @property
    def num_mask_payload_groups(self) -> int:
        return self.consumer.num_mask_payload_groups

    @property
    def payload_values_per_thread(self) -> int:
        return self.consumer.payload_values_per_thread

    @property
    def payload_valid_words(self) -> int:
        return self.consumer.payload_valid_words

    @property
    def payload_padded_words(self) -> int:
        return self.consumer.payload_padded_words

    @property
    def is_varlen(self) -> bool:
        return self.consumer.is_varlen

    @property
    def topology_planner_compile_key(self) -> tuple:
        return self.consumer.topology_planner_compile_key


@dataclass(frozen=True)
class _ResolvedSm100Hd256DqTopologyConfig:
    """Expose the dedicated dQ Q256 union tile to the classifier."""

    consumer: _ResolvedSm100Hd256DqConsumerConfig

    @property
    def arch(self) -> int:
        return self.consumer.arch

    @property
    def dtype(self) -> torch.dtype:
        return self.consumer.dtype

    @property
    def tile_m(self) -> int:
        return self.consumer.block_size[0]

    @property
    def tile_n(self) -> int:
        return self.consumer.tile_n

    @property
    def pack_gqa(self) -> bool:
        return self.consumer.pack_gqa

    @property
    def qhead_per_kvhead(self) -> int:
        return self.consumer.qhead_per_kvhead

    @property
    def num_mma_threads(self) -> int:
        return self.consumer.num_mma_threads

    @property
    def num_mask_payload_groups(self) -> int:
        return self.consumer.num_mask_payload_groups

    @property
    def payload_values_per_thread(self) -> int:
        return self.consumer.payload_values_per_thread

    @property
    def payload_valid_words(self) -> int:
        return self.consumer.payload_valid_words

    @property
    def payload_padded_words(self) -> int:
        return self.consumer.payload_padded_words

    @property
    def is_varlen(self) -> bool:
        return self.consumer.is_varlen

    @property
    def topology_planner_compile_key(self) -> tuple:
        return self.consumer.topology_planner_compile_key


@dataclass(frozen=True)
class _ResolvedSm100Hd256DkdvTopologyConfig:
    """Expose the dedicated dKdV Q256 x K128 tile to K2Q planning."""

    consumer: _ResolvedSm100Hd256DkdvConsumerConfig

    @property
    def arch(self) -> int:
        return self.consumer.arch

    @property
    def dtype(self) -> torch.dtype:
        return self.consumer.dtype

    @property
    def tile_m(self) -> int:
        return self.consumer.sparse_tile_m

    @property
    def tile_n(self) -> int:
        return self.consumer.sparse_tile_n

    @property
    def pack_gqa(self) -> bool:
        return self.consumer.pack_gqa

    @property
    def qhead_per_kvhead(self) -> int:
        return self.consumer.qhead_per_kvhead

    @property
    def num_mma_threads(self) -> int:
        return self.consumer.num_mma_threads

    @property
    def num_mask_payload_groups(self) -> int:
        return self.consumer.num_mask_payload_groups

    @property
    def payload_values_per_thread(self) -> int:
        return self.consumer.payload_values_per_thread

    @property
    def payload_valid_words(self) -> int:
        return self.consumer.payload_valid_words

    @property
    def payload_padded_words(self) -> int:
        return self.consumer.payload_padded_words

    @property
    def is_varlen(self) -> bool:
        return self.consumer.is_varlen

    @property
    def topology_planner_compile_key(self) -> tuple:
        return self.consumer.topology_planner_compile_key
