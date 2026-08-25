# SPDX-License-Identifier: BSD-3-Clause
"""SM90 forward consumer configuration for attention and mask planning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import warpgroup

from cudnn.flex_attention.plan.mask_plan import ArbitraryPlanSignature


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool
    num_stages: int = 2


@dataclass(frozen=True)
class _ResolvedSm90FwdConsumerConfig:
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
    mma_pv_is_rs: bool
    intra_wg_overlap: bool
    swap_ab: bool
    physical_subtiles: int
    num_mma_threads: int
    num_mask_payload_groups: int
    attention_num_threads: int
    num_stages: int
    payload_values_per_thread: int
    payload_valid_words: int
    payload_padded_words: int

    @property
    def block_size(self) -> tuple[int, int]:
        return (self.tile_m, self.tile_n)

    @property
    def topology_planner_compile_key(self) -> tuple:
        qhead_ratio = self.qhead_per_kvhead if self.pack_gqa else 1
        return (
            self.arch,
            self.tile_m,
            self.tile_n,
            self.is_varlen,
            self.pack_gqa,
            qhead_ratio,
        )

    @property
    def payload_planner_compile_key(self) -> tuple:
        return (
            self.topology_planner_compile_key,
            self.dtype,
            self.num_mma_threads,
            self.payload_values_per_thread,
            self.payload_valid_words,
            self.payload_padded_words,
        )

    @property
    def plan_signature(self) -> ArbitraryPlanSignature:
        return ArbitraryPlanSignature(
            arch_family="sm90",
            direction="forward",
            kernel_family="sm90_generic_fwd",
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            q_stage=1,
            cta_group_size=1,
            pack_gqa=self.pack_gqa,
            qhead_per_kvhead=self.qhead_per_kvhead,
            mma_atom_layout_id=(f"sm90_wgmma_f32_ss_qk_m{self.tile_m}n{self.tile_n}" f"_t{self.num_mma_threads}_major_kk"),
            swap_ab=self.swap_ab,
            payload_layout_id=(f"sm90_wgmma_qk_t{self.num_mma_threads}" f"_v{self.payload_values_per_thread}" f"_w{self.payload_padded_words}_v1"),
            dq_order_format="none",
            cluster_axis="m",
            scheduler_layout_id="plan_fwd_work_desc_i32x4_v1",
        )


def _tile_size_fwd_sm90(
    head_dim: int,
    head_dim_v: int,
    sparse_block_size_q: int | None = None,
) -> FwdConfig:
    """Return the native SM90 forward tile configuration."""

    if head_dim <= 64:
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, True, True)
        return FwdConfig(192, 128, True, True)
    if head_dim <= 96:
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, False, True)
        return FwdConfig(192, 128, False, True)
    if head_dim <= 128:
        return FwdConfig(128, 128, True, True)
    if head_dim <= 192:
        tile_n = 128 if head_dim_v <= 128 else 112
        return FwdConfig(128, tile_n, True, True)
    return FwdConfig(128, 64, True, True)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _native_sm90_fwd_smem_bytes(
    head_dim: int,
    head_dim_v: int,
    config: FwdConfig,
) -> int:
    """Return the safe SM90 forward dynamic shared-memory requirement."""

    tile_hdim = math.ceil(head_dim / 16) * 16
    tile_hdim_v = math.ceil(head_dim_v / 16) * 16
    num_mma_threads = 128 * (config.m_block_size // 64)
    payload_values_per_thread = config.m_block_size * config.n_block_size // num_mma_threads
    payload_words = math.ceil(payload_values_per_thread / 32)
    mask_pipeline_bytes = num_mma_threads * payload_words * 2 * 4

    # Preserve the conservative native-QKV estimate while accounting for the
    # actual SharedStorageQKV lifetime layout used by the mask pipeline.
    native_offset = (2 + 2 * config.num_stages + 2 * config.num_stages) * 8
    native_fields = (
        config.n_block_size * tile_hdim_v * config.num_stages * 2,
        config.m_block_size * max(tile_hdim, tile_hdim_v) * 2,
        config.n_block_size * tile_hdim * config.num_stages * 2,
        0 if config.mma_pv_is_rs else config.m_block_size * config.n_block_size * 2,
    )
    for size in native_fields:
        native_offset = _align_up(native_offset, 1024) + size

    pipeline_offset = (2 + 2 * config.num_stages + 2 * config.num_stages + 2 * 2) * 8
    pipeline_offset = _align_up(pipeline_offset, 16) + mask_pipeline_bytes
    pipeline_fields = (
        max(
            config.n_block_size * tile_hdim_v * config.num_stages,
            config.m_block_size * tile_hdim_v,
        )
        * 2,
        config.m_block_size * tile_hdim * 2,
        config.n_block_size * tile_hdim * config.num_stages * 2,
        0 if config.mma_pv_is_rs else config.m_block_size * config.n_block_size * 2,
    )
    for size in pipeline_fields:
        pipeline_offset = _align_up(pipeline_offset, 1024) + size

    return max(
        _align_up(native_offset, 1024),
        _align_up(pipeline_offset, 1024),
    )


def sm90_native_fwd_can_implement(head_dim: int, head_dim_v: int) -> bool:
    """Match native SM90 public-forward resource and codegen coverage."""

    config = _tile_size_fwd_sm90(head_dim, head_dim_v)
    tile_hdim_v = math.ceil(head_dim_v / 16) * 16
    # The 3-WG RS+overlap path reserves 160 registers for each MMA thread.
    # A value accumulator wider than 128 exceeds that allocation.
    if head_dim <= 64 and tile_hdim_v > 128:
        return False
    return _native_sm90_fwd_smem_bytes(head_dim, head_dim_v, config) <= 232448


def _resolve_pack_gqa(
    *,
    requested_pack_gqa: bool | None,
    num_q_heads: int,
    num_kv_heads: int,
    hmask: int,
    tile_m: int,
) -> bool:
    """Resolve PackGQA for the selected SM90 forward tile."""

    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if hmask not in (1, num_q_heads):
        raise ValueError(f"Hmask must be 1 or Hq ({num_q_heads}); got {hmask}")
    qhead_per_kvhead = num_q_heads // num_kv_heads
    qratio_is_power_of_two = (qhead_per_kvhead & (qhead_per_kvhead - 1)) == 0
    can_pack = hmask == 1 and qratio_is_power_of_two and tile_m % qhead_per_kvhead == 0
    if requested_pack_gqa is True:
        if hmask != 1:
            raise ValueError("pack_gqa=True requires Hmask=1 for arbitrary attention")
        if not qratio_is_power_of_two:
            raise ValueError(f"pack_gqa=True requires qratio=Hq/Hkv to be a power of two; got {qhead_per_kvhead}")
        if tile_m % qhead_per_kvhead != 0:
            raise ValueError(f"pack_gqa=True requires tile_m ({tile_m}) to be divisible by " f"qratio ({qhead_per_kvhead})")
        return True
    if requested_pack_gqa is False:
        return False
    return num_q_heads > num_kv_heads and can_pack


def _num_sm90_fwd_mask_payload_groups(
    *,
    num_mma_threads: int,
    qhead_per_kvhead: int,
    pack_gqa: bool,
) -> int:
    """Return the number of consumer-native forward payload equivalence classes."""

    if not pack_gqa:
        return num_mma_threads
    if qhead_per_kvhead <= 8:
        return num_mma_threads // qhead_per_kvhead
    return 2 * num_mma_threads // qhead_per_kvhead


@cute.jit
def _sm90_fwd_mask_payload_group_idx(
    consumer_tidx: cutlass.Int32,
    qhead_per_kvhead: cutlass.Constexpr[int],
) -> cutlass.Int32:
    """Map one WGMMA consumer thread to its forward mask payload group."""

    group_idx = consumer_tidx
    if const_expr(qhead_per_kvhead != 1):
        warp_group_idx = consumer_tidx // Int32(128)
        tidx_in_warp_group = consumer_tidx - warp_group_idx * Int32(128)
        a = tidx_in_warp_group % Int32(4)
        b = (tidx_in_warp_group // Int32(4)) % Int32(8)
        c = tidx_in_warp_group // Int32(32)
        if const_expr(qhead_per_kvhead <= 8):
            groups_per_warp_group = 128 // qhead_per_kvhead
            group_idx = warp_group_idx * Int32(groups_per_warp_group) + c * Int32(32 // qhead_per_kvhead) + (b // Int32(qhead_per_kvhead)) * Int32(4) + a
        else:
            logical_q = (warp_group_idx * Int32(64) + c * Int32(16) + b) // Int32(qhead_per_kvhead)
            group_idx = logical_q * Int32(4) + a
    return group_idx


@cute.jit
def _sm90_fwd_mask_payload_representative_tidx(
    group_idx: cutlass.Int32,
    qhead_per_kvhead: cutlass.Constexpr[int],
) -> cutlass.Int32:
    """Return a representative WGMMA thread for one forward payload group."""

    consumer_tidx = group_idx
    if const_expr(qhead_per_kvhead != 1):
        a = group_idx % Int32(4)
        if const_expr(qhead_per_kvhead <= 8):
            groups_per_warp_group = 128 // qhead_per_kvhead
            warp_group_idx = group_idx // Int32(groups_per_warp_group)
            group_in_warp_group = group_idx - warp_group_idx * Int32(groups_per_warp_group)
            groups_per_c = 32 // qhead_per_kvhead
            c = group_in_warp_group // Int32(groups_per_c)
            group_in_c = group_in_warp_group - c * Int32(groups_per_c)
            b = (group_in_c // Int32(4)) * Int32(qhead_per_kvhead)
            consumer_tidx = warp_group_idx * Int32(128) + c * Int32(32) + b * Int32(4) + a
        else:
            logical_q = group_idx // Int32(4)
            physical_q = logical_q * Int32(qhead_per_kvhead)
            warp_group_idx = physical_q // Int32(64)
            q_in_warp_group = physical_q - warp_group_idx * Int32(64)
            c = q_in_warp_group // Int32(16)
            consumer_tidx = warp_group_idx * Int32(128) + c * Int32(32) + a
    return consumer_tidx


def resolve_sm90_fwd_consumer_config(
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
) -> _ResolvedSm90FwdConsumerConfig:
    """Resolve the exact SM90 forward consumer signature used by a packed plan."""

    if arch // 10 != 9:
        raise NotImplementedError("Arbitrary attention currently supports SM90 only")
    if dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("SM90 arbitrary attention supports FP16 and BF16 only")
    if not (8 <= head_dim <= 256 and 8 <= head_dim_v <= 256):
        raise ValueError("SM90 head_dim and head_dim_v must be in [8, 256]")
    alignment = 16 // torch.empty((), dtype=dtype).element_size()
    if head_dim % alignment != 0 or head_dim_v % alignment != 0:
        raise ValueError(f"head_dim and head_dim_v must be divisible by {alignment} for {dtype}")
    if not sm90_native_fwd_can_implement(head_dim, head_dim_v):
        raise NotImplementedError(
            "SM90 arbitrary forward only supports (head_dim, head_dim_v) "
            "signatures implemented by native SM90 CuTe DSL forward; "
            f"got ({head_dim}, {head_dim_v})"
        )

    # Match the native SM90 consumer configuration. Arbitrary masking changes
    # only the block traversal and mask payload, not the dimension policy.
    fwd = _tile_size_fwd_sm90(head_dim, head_dim_v)
    effective_pack_gqa = _resolve_pack_gqa(
        requested_pack_gqa=pack_gqa,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        hmask=hmask,
        tile_m=fwd.m_block_size,
    )
    qhead_per_kvhead = num_q_heads // num_kv_heads
    num_mma_threads = 128 * (fwd.m_block_size // 64)
    num_mask_payload_groups = _num_sm90_fwd_mask_payload_groups(
        num_mma_threads=num_mma_threads,
        qhead_per_kvhead=qhead_per_kvhead,
        pack_gqa=effective_pack_gqa,
    )
    payload_values_per_thread, remainder = divmod(fwd.m_block_size * fwd.n_block_size, num_mma_threads)
    if remainder:
        raise AssertionError("QK accumulator values must partition evenly across MMA threads")
    payload_valid_words = math.ceil(payload_values_per_thread / 32)
    # Keep the consumer-native payload compact. The common 1/2/4-word rows are
    # naturally vectorized; uncommon widths avoid carrying unused words.
    payload_padded_words = payload_valid_words

    return _ResolvedSm90FwdConsumerConfig(
        arch=arch,
        dtype=dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qhead_per_kvhead=qhead_per_kvhead,
        is_varlen=is_varlen,
        pack_gqa=effective_pack_gqa,
        tile_m=fwd.m_block_size,
        tile_n=fwd.n_block_size,
        mma_pv_is_rs=fwd.mma_pv_is_rs,
        intra_wg_overlap=fwd.intra_wg_overlap,
        swap_ab=False,
        physical_subtiles=1,
        num_mma_threads=num_mma_threads,
        num_mask_payload_groups=num_mask_payload_groups,
        attention_num_threads=128 + num_mma_threads,
        num_stages=fwd.num_stages,
        payload_values_per_thread=payload_values_per_thread,
        payload_valid_words=payload_valid_words,
        payload_padded_words=payload_padded_words,
    )


@cute.jit
def make_sm90_fwd_tiled_mma_qk(
    dtype: type[cutlass.Numeric],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
):
    """Build the QK tiled MMA layout shared by the planner and consumer."""

    return sm90_utils_basic.make_trivial_tiled_mma(
        dtype,
        dtype,
        warpgroup.OperandMajorMode.K,
        warpgroup.OperandMajorMode.K,
        Float32,
        atom_layout_mnk=(tile_m // 64, 1, 1),
        tiler_mn=(64, tile_n),
    )


@cute.jit
def make_sm90_fwd_tiled_mma(
    dtype: type[cutlass.Numeric],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    tile_hdimv: cutlass.Constexpr[int],
    mma_pv_is_rs: cutlass.Constexpr[bool],
):
    """Build the shared QK/PV tiled MMA layouts for planner and consumer."""

    tiled_mma_qk = make_sm90_fwd_tiled_mma_qk(dtype, tile_m, tile_n)
    tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
        dtype,
        dtype,
        warpgroup.OperandMajorMode.K,
        warpgroup.OperandMajorMode.MN,
        Float32,
        atom_layout_mnk=(tile_m // 64, 1, 1),
        tiler_mn=(64, tile_hdimv),
        a_source=(warpgroup.OperandSource.RMEM if mma_pv_is_rs else warpgroup.OperandSource.SMEM),
    )
    return tiled_mma_qk, tiled_mma_pv


__all__ = [
    "FwdConfig",
    "_ResolvedSm90FwdConsumerConfig",
    "_num_sm90_fwd_mask_payload_groups",
    "_resolve_pack_gqa",
    "_sm90_fwd_mask_payload_group_idx",
    "_sm90_fwd_mask_payload_representative_tidx",
    "_tile_size_fwd_sm90",
    "make_sm90_fwd_tiled_mma",
    "make_sm90_fwd_tiled_mma_qk",
    "resolve_sm90_fwd_consumer_config",
    "sm90_native_fwd_can_implement",
]
