# SPDX-License-Identifier: BSD-3-Clause
"""Consumer-native mask contracts for the dedicated SM100 hd256 backward kernels.

The hd256 dQ and dKdV kernels consume score masks through different TMEM
partitions.  Their payloads are therefore intentionally described by separate
resolved configs even though both kernels operate on the same logical Q256 x
K128 sparse tile.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, cute
from cutlass.cute.nvgpu import tcgen05
from cudnn.flex_attention.plan.mask_plan import (
    ArbitraryPlanSignature,
    ArbitraryPlanTopology,
    canonical_blackwell_arch_family,
)
from cudnn.flex_attention.kernels.sm100.bwd.backward_dkdv_hd256 import (
    split_wg as _split_hd256_dkdv_wg,
)

_HD256_HEAD_DIM = 256
_HD256_CTA_GROUP_SIZE = 2
_HD256_ACC_DTYPE = Float32

_HD256_DQ_TILE_M = 128
_HD256_DQ_TILE_N = 128
_HD256_DQ_Q_STAGE = 1
_HD256_DQ_CONSUMER_THREADS = 4 * 32
_HD256_DQ_PAYLOAD_VALUES = 128
_HD256_DQ_PAYLOAD_VALID_WORDS = 4
_HD256_DQ_PAYLOAD_PADDED_WORDS = 4
_HD256_DQ_TMEM_LOAD_ATOM_ID = "tcgen05.ld32x32b.r16.hd256_dq"
_HD256_DQ_PAYLOAD_LAYOUT_ID = "sm100_hd256_dq_tcgen05_qk_ld32x32b_r16_t128_v128_w4_ctarank2_axism_v1"

_HD256_DKDV_TILE_M = 128
_HD256_DKDV_TILE_N = 64
_HD256_DKDV_Q_STAGE = 2
_HD256_DKDV_NUM_WG = 2
_HD256_DKDV_CONSUMER_THREADS = 8 * 32
_HD256_DKDV_PAYLOAD_VALUES = 32
_HD256_DKDV_PAYLOAD_VALID_WORDS = 1
_HD256_DKDV_PAYLOAD_PADDED_WORDS = 1
_HD256_DKDV_TMEM_LOAD_ATOM_ID = "tcgen05.ld32x32b.r16.wg2.hd256_dkdv"
_HD256_DKDV_PAYLOAD_LAYOUT_ID = "sm100_hd256_dkdv_tcgen05_kq_ld32x32b_r16_wg2_t256_v32_w1_qsub2_ctarank2_axisn_v1"


def _validate_hd256_resolved_geometry(
    *,
    arch: int,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    num_q_heads: int,
    num_kv_heads: int,
    qhead_per_kvhead: int,
    hmask: int,
    is_varlen: bool,
    pack_gqa: bool,
) -> None:
    """Reject any route not implemented by the hd256 arbitrary backward slice."""

    if type(arch) is not int:
        raise TypeError("arch must be an int")
    if arch not in (100, 103):
        raise NotImplementedError("SM100 hd256 arbitrary backward supports SM100/SM103 only")
    if dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("SM100 hd256 arbitrary backward supports FP16 and BF16 only")
    if type(head_dim) is not int or type(head_dim_v) is not int:
        raise TypeError("head_dim and head_dim_v must be ints")
    if head_dim != _HD256_HEAD_DIM or head_dim_v != _HD256_HEAD_DIM:
        raise NotImplementedError("SM100 hd256 arbitrary backward requires D=Dv=256")
    if type(num_q_heads) is not int or type(num_kv_heads) is not int:
        raise TypeError("num_q_heads and num_kv_heads must be ints")
    if type(qhead_per_kvhead) is not int:
        raise TypeError("qhead_per_kvhead must be an int")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads != 0 or qhead_per_kvhead != num_q_heads // num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if type(hmask) is not int:
        raise TypeError("hmask must be an int")
    if hmask not in (1, num_q_heads):
        raise ValueError(f"Hmask must be 1 or Hq ({num_q_heads}); got {hmask}")
    if type(is_varlen) is not bool:
        raise TypeError("is_varlen must be a bool")
    if type(pack_gqa) is not bool:
        raise TypeError("pack_gqa must be a bool")
    if pack_gqa:
        raise NotImplementedError("SM100 hd256 arbitrary backward does not support PackGQA")


def _validate_hd256_route_options(
    *,
    use_2cta_instrs: bool,
    deterministic: bool,
) -> None:
    if type(use_2cta_instrs) is not bool:
        raise TypeError("use_2cta_instrs must be a bool")
    if not use_2cta_instrs:
        raise NotImplementedError("SM100 hd256 arbitrary backward requires 2CTA instructions")
    if type(deterministic) is not bool:
        raise TypeError("deterministic must be a bool")


@dataclass(frozen=True)
class _ResolvedSm100Hd256DqConsumerConfig:
    """Static contract for the dedicated Q2K dQ score consumer."""

    arch: int
    dtype: torch.dtype
    head_dim: int
    head_dim_v: int
    num_q_heads: int
    num_kv_heads: int
    qhead_per_kvhead: int
    hmask: int
    is_varlen: bool
    pack_gqa: bool

    def __post_init__(self) -> None:
        _validate_hd256_resolved_geometry(
            arch=self.arch,
            dtype=self.dtype,
            head_dim=self.head_dim,
            head_dim_v=self.head_dim_v,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            qhead_per_kvhead=self.qhead_per_kvhead,
            hmask=self.hmask,
            is_varlen=self.is_varlen,
            pack_gqa=self.pack_gqa,
        )

    @property
    def tile_m(self) -> int:
        return _HD256_DQ_TILE_M

    @property
    def tile_n(self) -> int:
        return _HD256_DQ_TILE_N

    @property
    def q_stage(self) -> int:
        # This is the plan's number of Q subtiles per CTA, not the kernel's
        # head-dimension pipeline stage count.
        return _HD256_DQ_Q_STAGE

    @property
    def cta_group_size(self) -> int:
        return _HD256_CTA_GROUP_SIZE

    @property
    def cluster_axis(self) -> str:
        return "m"

    @property
    def physical_subtiles(self) -> int:
        return self.q_stage * self.cta_group_size

    @property
    def num_mask_payload_groups(self) -> int:
        return _HD256_DQ_CONSUMER_THREADS

    @property
    def num_mma_threads(self) -> int:
        return self.num_mask_payload_groups

    @property
    def payload_values_per_thread(self) -> int:
        return _HD256_DQ_PAYLOAD_VALUES

    @property
    def payload_valid_words(self) -> int:
        return _HD256_DQ_PAYLOAD_VALID_WORDS

    @property
    def payload_padded_words(self) -> int:
        return _HD256_DQ_PAYLOAD_PADDED_WORDS

    @property
    def tmem_load_atom_id(self) -> str:
        return _HD256_DQ_TMEM_LOAD_ATOM_ID

    @property
    def payload_layout_id(self) -> str:
        return _HD256_DQ_PAYLOAD_LAYOUT_ID

    @property
    def topology_direction(self) -> str:
        return "q2k"

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
        return (
            canonical_blackwell_arch_family(self.arch),
            self.topology_direction,
            self.topology.compile_key,
            self.is_varlen,
        )

    @property
    def payload_planner_compile_key(self) -> tuple:
        return (
            self.topology_planner_compile_key,
            self.dtype,
            self.head_dim,
            self.head_dim_v,
            self.tmem_load_atom_id,
            self.payload_layout_id,
            self.num_mask_payload_groups,
            self.payload_values_per_thread,
            self.payload_valid_words,
            self.payload_padded_words,
        )

    @property
    def planner_compile_key(self) -> tuple:
        # CUBINs remain exact-arch even though SM100 and SM103 share a plan signature.
        return (self.arch, self.payload_planner_compile_key)

    @property
    def plan_signature(self) -> ArbitraryPlanSignature:
        topology = self.topology
        return ArbitraryPlanSignature(
            arch_family=canonical_blackwell_arch_family(self.arch),
            direction="backward",
            kernel_family="sm100_hd256_dq",
            tile_m=topology.tile_m,
            tile_n=topology.tile_n,
            q_stage=topology.q_stage,
            cta_group_size=topology.cta_group_size,
            pack_gqa=topology.pack_gqa,
            qhead_per_kvhead=topology.qhead_per_kvhead,
            mma_atom_layout_id=(f"tcgen05_f32_ss_qk_cta2_m{topology.block_size[0]}" f"n{topology.block_size[1]}_major_kk"),
            swap_ab=False,
            payload_layout_id=self.payload_layout_id,
            dq_order_format="none",
            cluster_axis=topology.cluster_axis,
        )


@dataclass(frozen=True)
class _ResolvedSm100Hd256DkdvConsumerConfig:
    """Static contract for the dedicated K2Q dKdV score consumer."""

    arch: int
    dtype: torch.dtype
    head_dim: int
    head_dim_v: int
    num_q_heads: int
    num_kv_heads: int
    qhead_per_kvhead: int
    hmask: int
    is_varlen: bool
    pack_gqa: bool

    def __post_init__(self) -> None:
        _validate_hd256_resolved_geometry(
            arch=self.arch,
            dtype=self.dtype,
            head_dim=self.head_dim,
            head_dim_v=self.head_dim_v,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            qhead_per_kvhead=self.qhead_per_kvhead,
            hmask=self.hmask,
            is_varlen=self.is_varlen,
            pack_gqa=self.pack_gqa,
        )

    @property
    def tile_m(self) -> int:
        return _HD256_DKDV_TILE_M

    @property
    def tile_n(self) -> int:
        return _HD256_DKDV_TILE_N

    @property
    def q_stage(self) -> int:
        return _HD256_DKDV_Q_STAGE

    @property
    def subtile_factor(self) -> int:
        return self.q_stage

    @property
    def cta_group_size(self) -> int:
        return _HD256_CTA_GROUP_SIZE

    @property
    def cluster_axis(self) -> str:
        return "n"

    @property
    def physical_subtiles(self) -> int:
        return self.q_stage * self.cta_group_size

    @property
    def sparse_tile_m(self) -> int:
        return self.block_size[0]

    @property
    def sparse_tile_n(self) -> int:
        return self.block_size[1]

    @property
    def num_wg(self) -> int:
        return _HD256_DKDV_NUM_WG

    @property
    def num_mask_payload_groups(self) -> int:
        return _HD256_DKDV_CONSUMER_THREADS

    @property
    def num_mma_threads(self) -> int:
        return self.num_mask_payload_groups

    @property
    def payload_values_per_thread(self) -> int:
        return _HD256_DKDV_PAYLOAD_VALUES

    @property
    def payload_valid_words(self) -> int:
        return _HD256_DKDV_PAYLOAD_VALID_WORDS

    @property
    def payload_padded_words(self) -> int:
        return _HD256_DKDV_PAYLOAD_PADDED_WORDS

    @property
    def tmem_load_atom_id(self) -> str:
        return _HD256_DKDV_TMEM_LOAD_ATOM_ID

    @property
    def payload_layout_id(self) -> str:
        return _HD256_DKDV_PAYLOAD_LAYOUT_ID

    @property
    def topology_direction(self) -> str:
        return "k2q"

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
        return (
            canonical_blackwell_arch_family(self.arch),
            self.topology_direction,
            self.topology.compile_key,
            self.is_varlen,
        )

    @property
    def payload_planner_compile_key(self) -> tuple:
        return (
            self.topology_planner_compile_key,
            self.dtype,
            self.head_dim,
            self.head_dim_v,
            self.tmem_load_atom_id,
            self.payload_layout_id,
            self.num_mask_payload_groups,
            self.payload_values_per_thread,
            self.payload_valid_words,
            self.payload_padded_words,
        )

    @property
    def planner_compile_key(self) -> tuple:
        return (self.arch, self.payload_planner_compile_key)

    @property
    def spt(self) -> None:
        # The dedicated kernel has no dQ rank/semaphore ordering contract.
        return None

    @property
    def plan_signature(self) -> ArbitraryPlanSignature:
        topology = self.topology
        return ArbitraryPlanSignature(
            arch_family=canonical_blackwell_arch_family(self.arch),
            direction="backward",
            kernel_family="sm100_hd256_dkdv",
            tile_m=topology.tile_m,
            tile_n=topology.tile_n,
            q_stage=topology.q_stage,
            cta_group_size=topology.cta_group_size,
            pack_gqa=topology.pack_gqa,
            qhead_per_kvhead=topology.qhead_per_kvhead,
            mma_atom_layout_id=(f"tcgen05_f32_ss_kq_cta2_m{topology.block_size[1]}" f"n{topology.block_size[0]}_major_kk"),
            swap_ab=True,
            payload_layout_id=self.payload_layout_id,
            dq_order_format="none",
            cluster_axis=topology.cluster_axis,
        )


def resolve_sm100_hd256_dq_consumer_config(
    *,
    arch: int,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    num_q_heads: int,
    num_kv_heads: int,
    is_varlen: bool,
    hmask: int = 1,
    pack_gqa: bool = False,
    use_2cta_instrs: bool = True,
    deterministic: bool = False,
) -> _ResolvedSm100Hd256DqConsumerConfig:
    """Resolve the hd256 dQ arbitrary-mask consumer."""

    _validate_hd256_route_options(
        use_2cta_instrs=use_2cta_instrs,
        deterministic=deterministic,
    )
    qhead_per_kvhead = num_q_heads // num_kv_heads if type(num_q_heads) is int and type(num_kv_heads) is int and num_kv_heads > 0 else 0
    return _ResolvedSm100Hd256DqConsumerConfig(
        arch=arch,
        dtype=dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qhead_per_kvhead=qhead_per_kvhead,
        hmask=hmask,
        is_varlen=is_varlen,
        pack_gqa=pack_gqa,
    )


def resolve_sm100_hd256_dkdv_consumer_config(
    *,
    arch: int,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    num_q_heads: int,
    num_kv_heads: int,
    is_varlen: bool,
    hmask: int = 1,
    pack_gqa: bool = False,
    use_2cta_instrs: bool = True,
    deterministic: bool = False,
) -> _ResolvedSm100Hd256DkdvConsumerConfig:
    """Resolve the hd256 dKdV arbitrary-mask consumer."""

    _validate_hd256_route_options(
        use_2cta_instrs=use_2cta_instrs,
        deterministic=deterministic,
    )
    qhead_per_kvhead = num_q_heads // num_kv_heads if type(num_q_heads) is int and type(num_kv_heads) is int and num_kv_heads > 0 else 0
    return _ResolvedSm100Hd256DkdvConsumerConfig(
        arch=arch,
        dtype=dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qhead_per_kvhead=qhead_per_kvhead,
        hmask=hmask,
        is_varlen=is_varlen,
        pack_gqa=pack_gqa,
    )


@cute.jit
def make_sm100_hd256_dkdv_tiled_mma_kq(dtype: type[cutlass.Numeric]):
    """Build the exact 2CTA K128 x Q128 score MMA used by the dKdV kernel."""

    return sm100_utils.make_trivial_tiled_mma(
        dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        _HD256_ACC_DTYPE,
        tcgen05.CtaGroup.TWO,
        (
            _HD256_DKDV_TILE_N * _HD256_CTA_GROUP_SIZE,
            _HD256_DKDV_TILE_M,
        ),
    )


@cute.jit
def make_sm100_hd256_dkdv_tmem_load(tSAcc: cute.Tensor, dp_idx: cutlass.Int32):
    """Build the dKdV kernel's native Rep16 score TMEM load slice."""

    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
        _HD256_ACC_DTYPE,
    )
    return tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(dp_idx)


@cute.jit
def make_sm100_hd256_dkdv_score_ownership(
    tiled_mma_kq: cute.TiledMma,
    payload_group_idx: cutlass.Int32,
):
    """Return the 32 score coordinates owned by one dKdV payload group.

    ``payload_group_idx`` follows the consumer's exact grouping:
    ``wg_idx * 128 + dp_idx``.  CTA rank and Q subtile remain explicit payload
    axes, so this helper returns the CTA-local K64 x Q128 ownership only.
    """

    score_shape = (
        _HD256_DKDV_TILE_N * _HD256_CTA_GROUP_SIZE,
        _HD256_DKDV_TILE_M,
    )
    score_fragment_shape = tiled_mma_kq.partition_shape_C(score_shape)
    tSTtST = tiled_mma_kq.make_fragment_C(score_fragment_shape)
    tSAcc = tSTtST[(None, None), 0, 0]

    dp_idx = payload_group_idx % Int32(128)
    wg_idx = payload_group_idx // Int32(128)
    thr_tmem_load = make_sm100_hd256_dkdv_tmem_load(tSAcc, dp_idx)
    cST = cute.make_identity_tensor((_HD256_DKDV_TILE_N, _HD256_DKDV_TILE_M))
    tTR_cST = thr_tmem_load.partition_D(cST)
    return _split_hd256_dkdv_wg(tTR_cST, Int32(_HD256_DKDV_NUM_WG), wg_idx)


__all__ = [
    "_ResolvedSm100Hd256DkdvConsumerConfig",
    "_ResolvedSm100Hd256DqConsumerConfig",
    "make_sm100_hd256_dkdv_score_ownership",
    "make_sm100_hd256_dkdv_tiled_mma_kq",
    "make_sm100_hd256_dkdv_tmem_load",
    "resolve_sm100_hd256_dkdv_consumer_config",
    "resolve_sm100_hd256_dq_consumer_config",
]
