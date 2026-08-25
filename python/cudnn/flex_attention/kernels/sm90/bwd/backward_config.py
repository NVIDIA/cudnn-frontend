# SPDX-License-Identifier: BSD-3-Clause
"""Shared SM90 backward consumer configuration for attention and mask planning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch
from cutlass import Float32, const_expr
from cutlass.cute.nvgpu import warpgroup

from cudnn.flex_attention.kernels.sm90.fwd.forward_config import sm90_native_fwd_can_implement


@dataclass(frozen=True)
class BwdConfig:
    m_block_size: int
    n_block_size: int
    num_stages_Q: int
    num_stages_dO: int
    num_stages_PdS: int
    SdP_swapAB: bool
    dKV_swapAB: bool
    dQ_swapAB: bool
    AtomLayoutMSdP: int
    AtomLayoutNdKV: int
    AtomLayoutMdQ: int
    num_wg: int = 2
    dQ_single_wg: bool = False


@dataclass(frozen=True)
class _ResolvedSm90BwdConsumerConfig:
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
    subtile_factor: int
    num_stages_q: int
    num_stages_do: int
    num_stages_pds: int
    sdp_swap_ab: bool
    dkv_swap_ab: bool
    dq_swap_ab: bool
    atom_layout_m_sdp: int
    atom_layout_n_dkv: int
    atom_layout_m_dq: int
    num_wg: int
    dq_single_wg: bool
    spt: bool
    physical_subtiles: int
    num_mma_threads: int
    attention_num_threads: int
    payload_values_per_thread: int
    payload_valid_words: int
    payload_padded_words: int

    @property
    def block_size(self) -> tuple[int, int]:
        return (self.sparse_tile_m, self.tile_n)

    @property
    def planner_compile_key(self) -> tuple:
        return (
            self.arch,
            self.dtype,
            self.head_dim,
            self.head_dim_v,
            self.qhead_per_kvhead,
            self.is_varlen,
            self.tile_m,
            self.tile_n,
            self.subtile_factor,
            self.sdp_swap_ab,
            self.atom_layout_m_sdp,
            self.num_wg,
            self.spt,
            self.num_mma_threads,
            self.payload_valid_words,
            self.payload_padded_words,
        )


def _tile_size_bwd_sm90(
    head_dim: int,
    head_dim_v: int,
    sparse_block_size_q: int | None = None,
) -> BwdConfig:
    """Return the native SM90 backward tile configuration."""

    if head_dim <= 64:
        return BwdConfig(
            m_block_size=128,
            n_block_size=128,
            num_stages_Q=2,
            num_stages_dO=2,
            num_stages_PdS=2,
            SdP_swapAB=True,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=2,
            AtomLayoutMdQ=2,
        )
    if head_dim <= 96:
        return BwdConfig(
            m_block_size=64,
            n_block_size=128,
            num_stages_Q=2,
            num_stages_dO=2,
            num_stages_PdS=2,
            SdP_swapAB=True,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=2,
            AtomLayoutMdQ=1,
            dQ_single_wg=True,
        )
    if head_dim <= 128:
        m_block_size = 64
        if sparse_block_size_q is not None and sparse_block_size_q % m_block_size != 0:
            m_block_size = 64
        return BwdConfig(
            m_block_size=m_block_size,
            n_block_size=128,
            num_stages_Q=2,
            num_stages_dO=2,
            num_stages_PdS=2,
            SdP_swapAB=True,
            dKV_swapAB=False,
            dQ_swapAB=m_block_size % 64 != 0,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=2,
            AtomLayoutMdQ=1,
        )
    if head_dim <= 192:
        hdimv128 = head_dim_v <= 128
        return BwdConfig(
            m_block_size=64,
            n_block_size=96,
            num_stages_Q=2,
            num_stages_dO=2 if hdimv128 else 1,
            num_stages_PdS=1,
            SdP_swapAB=False,
            dKV_swapAB=True,
            dQ_swapAB=False,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=2,
            AtomLayoutMdQ=1,
            num_wg=2,
        )
    return BwdConfig(
        m_block_size=64,
        n_block_size=64,
        num_stages_Q=1,
        num_stages_dO=1,
        num_stages_PdS=1,
        SdP_swapAB=False,
        dKV_swapAB=False,
        dQ_swapAB=False,
        AtomLayoutMSdP=1,
        AtomLayoutNdKV=1,
        AtomLayoutMdQ=1,
    )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _native_sm90_bwd_smem_bytes(
    head_dim: int,
    head_dim_v: int,
    config: BwdConfig,
) -> int:
    """Return the exact native SM90 backward dynamic shared-memory size."""

    tile_hdim = math.ceil(head_dim / 16) * 16
    tile_hdim_v = math.ceil(head_dim_v / 16) * 16
    tile_m = config.m_block_size
    tile_n = config.n_block_size
    mma_dkv_is_rs = config.AtomLayoutMSdP == 1 and config.AtomLayoutNdKV == config.num_wg and config.SdP_swapAB and not config.dKV_swapAB
    offset = (config.num_stages_Q * 2 + config.num_stages_dO * 2) * 8
    fields = (
        (_align_up(tile_m, 64) * config.num_stages_Q * 4, 128),
        (_align_up(tile_m, 64) * config.num_stages_dO * 4, 128),
        (tile_m * tile_hdim * config.num_stages_Q * 2, 1024),
        (tile_n * tile_hdim_v * 2, 1024),
        (tile_n * tile_hdim * 2, 1024),
        (tile_m * tile_hdim_v * config.num_stages_dO * 2, 1024),
        (
            0 if mma_dkv_is_rs else tile_m * tile_n * config.num_stages_PdS * 2,
            1024,
        ),
        (tile_m * tile_n * config.num_stages_PdS * 2, 1024),
        (tile_m * tile_hdim * 4, 1024),
    )
    for size, alignment in fields:
        offset = _align_up(offset, alignment) + size
    return _align_up(offset, 1024)


def sm90_native_bwd_can_implement(
    head_dim: int,
    head_dim_v: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> bool:
    """Match native SM90 arbitrary-forward/backward coverage."""

    if not sm90_native_fwd_can_implement(head_dim, head_dim_v):
        return False
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        return False
    if num_q_heads != num_kv_heads and head_dim != head_dim_v:
        return False

    tile_hdim = math.ceil(head_dim / 16) * 16
    tile_hdim_v = math.ceil(head_dim_v / 16) * 16
    if 128 < head_dim <= 192 and (tile_hdim % 64 != 0 or tile_hdim_v % 64 != 0):
        return False
    if head_dim > 192 and (tile_hdim != 256 or tile_hdim_v not in (64, 128, 256)):
        return False

    config = _tile_size_bwd_sm90(head_dim, head_dim_v)
    return _native_sm90_bwd_smem_bytes(head_dim, head_dim_v, config) <= 232448


def resolve_sm90_bwd_consumer_config(
    *,
    arch: int,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    num_q_heads: int,
    num_kv_heads: int,
    is_varlen: bool,
    subtile_factor: int | None = None,
) -> _ResolvedSm90BwdConsumerConfig:
    """Resolve the arbitrary SM90 backward consumer before inspecting a plan."""

    if arch // 10 != 9:
        raise NotImplementedError("Arbitrary backward currently supports SM90 only")
    if dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("SM90 arbitrary backward supports FP16 and BF16 only")
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if not (8 <= head_dim <= 256 and 8 <= head_dim_v <= 256):
        raise ValueError("SM90 head_dim and head_dim_v must be in [8, 256]")
    alignment = 16 // torch.empty((), dtype=dtype).element_size()
    if head_dim % alignment != 0 or head_dim_v % alignment != 0:
        raise ValueError(f"head_dim and head_dim_v must be divisible by {alignment} for {dtype}")
    if not sm90_native_bwd_can_implement(
        head_dim,
        head_dim_v,
        num_q_heads,
        num_kv_heads,
    ):
        raise NotImplementedError(
            "SM90 arbitrary backward only supports signatures implemented by "
            "native SM90 CuTe DSL public forward/backward; "
            f"got D={head_dim}, Dv={head_dim_v}, "
            f"Hq={num_q_heads}, Hkv={num_kv_heads}"
        )

    # Arbitrary masking changes K2Q traversal and payload application without
    # changing the dimension-specific tile or pipeline configuration.
    cfg = _tile_size_bwd_sm90(head_dim, head_dim_v)
    if subtile_factor is None:
        # Build K2Q at the physical consumer tile. Grouping multiple consumer
        # tiles can introduce fully masked physical subtiles and false dQ
        # semaphore contributors at arbitrary boundaries.
        subtile_factor = 1
    if subtile_factor <= 0:
        raise ValueError("subtile_factor must be positive")
    num_mma_threads = cfg.num_wg * 128
    payload_values_per_thread, remainder = divmod(cfg.m_block_size * cfg.n_block_size, num_mma_threads)
    if remainder:
        raise AssertionError("SdP accumulator values must partition evenly across MMA threads")
    payload_valid_words = math.ceil(payload_values_per_thread / 32)
    # Keep the consumer-native bit payload compact. Per-thread rows remain
    # naturally aligned to their power-of-two word width, so 1/2/4-word rows
    # compile to 32/64/128-bit coalesced accesses without padding every row to
    # 128 bits.
    payload_padded_words = payload_valid_words

    return _ResolvedSm90BwdConsumerConfig(
        arch=arch,
        dtype=dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qhead_per_kvhead=num_q_heads // num_kv_heads,
        is_varlen=is_varlen,
        tile_m=cfg.m_block_size,
        tile_n=cfg.n_block_size,
        sparse_tile_m=subtile_factor * cfg.m_block_size,
        subtile_factor=subtile_factor,
        num_stages_q=cfg.num_stages_Q,
        num_stages_do=cfg.num_stages_dO,
        num_stages_pds=cfg.num_stages_PdS,
        sdp_swap_ab=cfg.SdP_swapAB,
        dkv_swap_ab=cfg.dKV_swapAB,
        dq_swap_ab=cfg.dQ_swapAB,
        atom_layout_m_sdp=cfg.AtomLayoutMSdP,
        atom_layout_n_dkv=cfg.AtomLayoutNdKV,
        atom_layout_m_dq=cfg.AtomLayoutMdQ,
        num_wg=cfg.num_wg,
        dq_single_wg=cfg.dQ_single_wg,
        # A fixed descending K traversal provides deterministic dQ reduction order
        # and avoids the long-tail schedule of ascending K-major sparse rows.
        spt=True,
        physical_subtiles=subtile_factor,
        num_mma_threads=num_mma_threads,
        attention_num_threads=(cfg.num_wg + 1) * 128,
        payload_values_per_thread=payload_values_per_thread,
        payload_valid_words=payload_valid_words,
        payload_padded_words=payload_padded_words,
    )


@cute.jit
def make_sm90_bwd_tiled_mma_sdp(
    dtype: type[cutlass.Numeric],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    num_wg_mma: cutlass.Constexpr[int],
    atom_layout_m_sdp: cutlass.Constexpr[int],
    sdp_swap_ab: cutlass.Constexpr[bool],
):
    """Build the shared SdP MMA layout used by planner and backward consumer."""

    atom_layout = (atom_layout_m_sdp, num_wg_mma // atom_layout_m_sdp, 1)
    tiler_mn = (tile_m // atom_layout[0], tile_n // atom_layout[1])
    if const_expr(sdp_swap_ab):
        atom_layout = (atom_layout[1], atom_layout[0], atom_layout[2])
    return sm90_utils_basic.make_trivial_tiled_mma(
        dtype,
        dtype,
        warpgroup.OperandMajorMode.K,
        warpgroup.OperandMajorMode.K,
        Float32,
        atom_layout_mnk=atom_layout,
        tiler_mn=(64, tiler_mn[1] if not sdp_swap_ab else tiler_mn[0]),
    )


__all__ = [
    "BwdConfig",
    "_ResolvedSm90BwdConsumerConfig",
    "_tile_size_bwd_sm90",
    "make_sm90_bwd_tiled_mma_sdp",
    "resolve_sm90_bwd_consumer_config",
    "sm90_native_bwd_can_implement",
]
