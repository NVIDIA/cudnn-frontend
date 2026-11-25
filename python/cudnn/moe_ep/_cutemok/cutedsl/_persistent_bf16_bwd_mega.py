# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""SM100 / SM103 BF16 backward persistent megakernel with configured ring geometry.

One ``@cute.kernel`` and one cluster launch run the whole BF16 backward with runtime geometry.  The configured
communication CTAs run the dynamic task stream (``_bwd_dynamic_task_stream``): replay X, reverse-combine peer dY
(with the router score) and reverse-dispatch dX plus dRouter.  The compute CTAs are CLC scheduled:

    shared Down (SwiGLU-backward fused into its epilogue) -> shared Gate+Up dX -> shared Wgrads
    -> per-minibatch Replay (after saved ring slots) -> routed Down (swap-AB, fused SwiGLU-backward)
    -> routed Gate+Up dX (swap-AB, N = align16(valid rows of the block)) -> completed
    expert/macro-segment Wgrads (Down tiles, then Gate and Up tile t of an expert next to each other).

With zero or one true macrobatch all minibatch tasks come before the dense per-expert Wgrad suffix (no replay or
retirement); there the Down dgrad of minibatch s + 1 runs ahead of the Gate+Up dgrad of s (Down(0), then
[Down(s + 1), Gate/Up(s)]) so the next Down's MMA covers this minibatch's fused epilogue.

All ordinary GEMMs share a six-stage A/B and two-stage accumulator pipeline.  Each compute CTA's CLC warp decodes a
task once and publishes the decoded record through a shared-memory task-info pipeline; the epilogue, UMMA and
AB-load warps read the record (TASK_INFO_FIELDS) instead of decoding it themselves.  Replay Gate+Up is paired: one X
tile is consumed with both K-major expert weights through a compact three-stage A/B view, two FP32 TMEM slots in one
accumulator epoch, and two sequential BF16 epilogues.  Gate and Up dX enqueue I/K64 runtime tiles each into one FP32
TMEM accumulator and perform one BF16 epilogue.

The routed dgrad tiles run swap-AB (``_swapab_fwd``): the expert's weights are operand A (M = this block's 256
output features, MN-major: [Wg; Wu] for the Gate/Up dX, Wd for the Down dH), the block's token rows are operand B
with a runtime N from the expert's valid rows in the block (align16 for the Gate/Up dX, rounded up to the Wgrad K
block for the Down dH, _routed_down_n), so only each expert's last row block shrinks and its padding rows are neither
multiplied nor stored (the reverse dispatch never pushes them, the Wgrads never read them); the Gate/Up dX epilogue
stores the M-major accumulator transposed into the dX ring, the Down epilogue stages it transposed through shared
memory for the fused SwiGLU-backward.

Routed Wgrad dW tiles are written entirely by this kernel (no external zero kernel): an expert's first macrobatch
segment stores, later segments reduce-add in macrobatch order behind a per-tile wgrad_segment_done counter
(deterministic accumulation), and experts without rows store zero tiles from macrobatch 0.  Both routed buffer
lifetimes retire before a physical ring slot is reused.

Router weight w: dY is reverse-combined unscaled and w enters in exactly one place, the routed Down dgrad epilogue
(the fused SwiGLU-backward), for every minibatch alike.  It takes dH * w for the Gate/Up gradients, the unscaled dH
for the router gradient, and stores the Hidden it recomputes anyway back as w*H (the forward context's Hidden is
never read; there is no replay SwiGLU task, Replay is Gate+Up only).  Wgrad-down consumes w*H with the unscaled dY
and waits for the row block's Down dgrad epilogues (d_gate_up_ready) like Wgrad-gate/up, so the communication CTAs
have no per-row scaling loop.

Runtime vs compile time: token and expert counts, hidden / intermediate sizes, top-k, macrobatch / minibatch sizes,
the number of communication CTAs and the EP size are runtime values.  The variant flags of the kernel class and the
task decode method select code at compile time.

Private device module; the public entry is ``persistent_bf16_bwd.backward_bf16``.
"""

from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum, block_copy

from ._compat import copy_utils
from ._compat.activation import dswiglu
from ._compat.gemm_sm100 import GemmDefaultSm100, NamedBarrierGemm, VarlenArguments
from ._compat.pipeline import PipelineTmaUmma, PipelineUmmaAsync

from ._mxfp8_bwd_math import situ_bwd


from .persistent_bf16_bwd_contract import (
    AB_LOAD_WARP,
    BF16_IDLE_WARP,
    CLC_DEPTH,
    CLC_DRAIN_DEPTH,
    CLC_WARP,
    BWD_EXPERT_TABLE_CAP,
    CLUSTER_SIZE,
    COMM_TILE_SHAPES,
    DEFAULT_COMM_TILE_SHAPE,
    CommTileShape,
    EPILOGUE_WARPS,
    GEMM_ARENA_BYTES as _CONTRACT_GEMM_ARENA_BYTES,
    MLP_ACC_STAGES,
    MLP_EPILOGUE_PIECES,
    MLP_LOAD_STAGES,
    MLP_TILE_K,
    MLP_TILE_M,
    MLP_TILE_N,
    REUSABLE_ARENA_BYTES,
    SWIGLU_TILE_N,
    THREADS_PER_CTA,
    TaskDecode,
    UMMA_WARP,
    WARPS_PER_CTA,
)
from ._arch import COMPILE_OPTIONS
from ._host_common import TAIL_WIDTH, peer_address_table, row_major_view

# Helpers shared with the forward mega (search bound, storage member sizes).
from . import routed_descriptors as routed_descriptors_pkg
from ._persistent_bf16_fwd_mega import (
    _cu_end_upper_bound,
    _routed_descriptors_or_placeholder,
    _routed_tables_or_placeholder,
    _storage_member_bytes,
    routed_weight_view,
)
from . import _bwd_dynamic_task_stream, _swapab_fwd
from ._bwd_device_helpers import (
    named_barrier_sync_unaligned,
    _mbar_spin_wait,
    _store_global_zero16,
    _counter_wait_gpu,
    _counter_arrive_gpu,
    _ceil_div_i32,
    _swizzled_tile_i32,
)

CLUSTER_SHAPE = (CLUSTER_SIZE, 1, 1)
MLP_EPILOGUE_N = 32
MLP_BF16_D_TILES = 3
PAIRED_GATE_UP_STAGES = 3
PAIRED_GATE_UP_TRANSACTION_BYTES = 98304
PAIRED_ACC_STAGES = 1
CLC_RESPONSE_BYTES = 16
# The CLC response is read by the CLC warp of each CTA only (one arrival per CTA on rank 0's
# empty barrier).  That warp decodes the task once per CTA and publishes the decoded record to
# the other warps through the task-info pipeline (same form as the forward mega).
CLC_COMPLETION_WARPS = CLUSTER_SIZE
# Task-info record: Int32 words written by the CLC warp, read by the four epilogue, UMMA, idle and
# AB-load warps (TASK_INFO_CONSUMER_WARPS, one elected arrival each).  "active" is the loop
# condition of every warp (0 = stop); "needs_drain" is the CLC capacity-tail state of the final
# record; "kind" and "generation" identify the task for the role dispatch and the readiness waits;
# the ten words from "valid" to "tile_valid" are the _gemm_geometry tuple in its order.  One stage:
# the CLC warp requests the next task only after every consumer has taken the current record
# (CLC depth 1: the cluster holds at most the task it runs plus one in flight), so a second
# stage could never be filled before the first is released.
TASK_INFO_STAGES = 1
TASK_INFO_FIELDS = (
    "active",
    "needs_drain",
    "kind",
    "generation",
    "valid",
    "expert",
    "row",
    "column",
    "k_begin",
    "k_tiles",
    "wgrad_first",
    "wgrad_has_next",
    "wgrad_empty",
    "tile_valid",
)
TASK_INFO_GEOMETRY_FIELDS = TASK_INFO_FIELDS[4:]  # the _gemm_geometry tuple
TASK_INFO_WORDS = 16  # 64 B per stage, 16-byte aligned
TASK_INFO_CONSUMER_WARPS = WARPS_PER_CTA - 1
assert len(TASK_INFO_FIELDS) == len(set(TASK_INFO_FIELDS)) <= TASK_INFO_WORDS
assert len(TASK_INFO_GEOMETRY_FIELDS) == 10
# Every warp but the CLC warp consumes each task record (selected by the physical warp index: the
# epilogue warps, UMMA, AB-load and idle); the role constants must cover every warp of the CTA.
assert len({*EPILOGUE_WARPS, UMMA_WARP, AB_LOAD_WARP, BF16_IDLE_WARP, CLC_WARP}) == WARPS_PER_CTA
assert TASK_INFO_CONSUMER_WARPS == len(EPILOGUE_WARPS) + 3
_TI = {name: index for index, name in enumerate(TASK_INFO_FIELDS)}

GEMM_A_BYTES = MLP_LOAD_STAGES * 128 * MLP_TILE_K * 2
GEMM_B_BYTES = MLP_LOAD_STAGES * 128 * MLP_TILE_K * 2
GEMM_D_BYTES = MLP_BF16_D_TILES * 128 * MLP_EPILOGUE_N * 2
GEMM_ARENA_BYTES = GEMM_A_BYTES + GEMM_B_BYTES + GEMM_D_BYTES
# Barriers fill the first 1 KiB; the arena starts at 1024 B (230400 B total, 2 KiB under the
# 232448 B SM103 opt-in maximum).
DYNAMIC_SHARED_MEMORY_BYTES = 1024 + REUSABLE_ARENA_BYTES
assert DYNAMIC_SHARED_MEMORY_BYTES <= 232448  # SM103 opt-in dynamic shared-memory maximum

assert (GEMM_A_BYTES, GEMM_B_BYTES, GEMM_D_BYTES) == (98304, 98304, 24576)
assert GEMM_ARENA_BYTES == 221184 == _CONTRACT_GEMM_ARENA_BYTES
assert _bwd_dynamic_task_stream.ARENA_BYTES <= REUSABLE_ARENA_BYTES  # the communication buffers alias the arena
assert (MLP_LOAD_STAGES, MLP_ACC_STAGES) == (6, 2)
assert PAIRED_GATE_UP_STAGES * 2 == MLP_LOAD_STAGES
assert PAIRED_GATE_UP_TRANSACTION_BYTES == 3 * 128 * MLP_TILE_K * 2 * 2
assert PAIRED_ACC_STAGES == 1
assert (MLP_EPILOGUE_PIECES, MLP_BF16_D_TILES) == (8, 3)
assert (CLC_DEPTH, CLC_DRAIN_DEPTH) == (1, 8)
# Routed dgrad, swap-AB: operand A of each CTA is its 128 output features of the expert's weights ([Wg; Wu]
# for the Gate/Up dX, Wd for the Down dH), staged as two 64-feature x 64-K MN-major slabs per A stage
# (_swapab_fwd.A_MN_SLAB_*, 8 KiB each); the epilogues handle the accumulator transposed one 32-token
# subtile at a time through the D stages: the Gate/Up dX epilogue stores it, the Down epilogue stages the
# BF16 dH of a subtile as (32 tokens, 128 features) rows for the fused SwiGLU-backward (two staging
# tiles alternate; the rows are padded by 16 B so different tokens' 128-bit read-backs fall in different
# banks -- the chunks c and c + 2 of one token are 128 B apart and still share banks, a 2-way conflict).
_SWAP_SLAB_ROWS = _swapab_fwd.A_MN_SLAB_ROWS
_SWAP_SLAB_BYTES = _swapab_fwd.A_MN_SLAB_BYTES
_SWAP_EPI_ROWS = MLP_EPILOGUE_N
_SWAP_CTA_FEATURES = MLP_TILE_M // CLUSTER_SIZE
_SWAP_FEATURE_CHUNK = 32  # features per epilogue thread of the Down read-back (four chunks per token)
_SWAP_FEATURE_GROUP = 16  # features of a chunk per iteration of the fused SwiGLU-backward's group loop
_SWAP_FEATURE_GROUPS = _SWAP_FEATURE_CHUNK // _SWAP_FEATURE_GROUP
_SWAP_DH_ROW_STRIDE = _SWAP_CTA_FEATURES + 8
_SWAP_DH_STAGING_TILES = 2
assert _SWAP_SLAB_BYTES == _SWAP_SLAB_ROWS * MLP_TILE_K * 2
assert 2 * _SWAP_SLAB_ROWS == _SWAP_CTA_FEATURES  # the CTA's 128 features
assert 2 * _SWAP_SLAB_BYTES * MLP_LOAD_STAGES == GEMM_A_BYTES
assert _SWAP_EPI_ROWS * _SWAP_CTA_FEATURES * 2 * MLP_BF16_D_TILES == GEMM_D_BYTES
# One FP32 Wgrad staging tile where the BF16 epilogue keeps three: it has to fit the same region.
assert 128 * MLP_EPILOGUE_N * 4 <= GEMM_D_BYTES
assert _SWAP_EPI_ROWS * 4 == 32 * len(EPILOGUE_WARPS)  # one token x one chunk per epilogue thread
assert _SWAP_FEATURE_CHUNK * 4 == _SWAP_CTA_FEATURES and _SWAP_DH_ROW_STRIDE % 8 == 0
# A chunk is whole groups and a group whole 128-bit vectors of BF16 (the group loop's loads and stores).
assert _SWAP_FEATURE_GROUPS * _SWAP_FEATURE_GROUP == _SWAP_FEATURE_CHUNK and _SWAP_FEATURE_GROUP % 8 == 0
assert _SWAP_EPI_ROWS * _SWAP_DH_ROW_STRIDE * 2 * _SWAP_DH_STAGING_TILES <= GEMM_D_BYTES
assert MLP_TILE_K % _SWAP_EPI_ROWS == 0 and (MLP_TILE_K // _SWAP_EPI_ROWS) % 2 == 0  # Down pieces come in pairs
# The has_tail kernels' tail instantiations are written for a tail of exactly TAIL_WIDTH: half a tile in M and
# in N, which is one CTA's rows or features of a cluster tile and one SwiGLU column tile, and a whole number
# of epilogue pieces.  Checked here, when the module is loaded, so another tile geometry fails before any
# kernel is compiled; the host admits only H / I that are whole TAIL_WIDTH halves (_host_common.has_tail).
assert TAIL_WIDTH == 128 and MLP_TILE_M == MLP_TILE_N == 2 * TAIL_WIDTH
assert MLP_TILE_M // CLUSTER_SIZE == MLP_TILE_N // CLUSTER_SIZE == SWIGLU_TILE_N == TAIL_WIDTH
assert TAIL_WIDTH % (MLP_TILE_N // MLP_EPILOGUE_PIECES) == 0
_TAIL_PIECES = TAIL_WIDTH // (MLP_TILE_N // MLP_EPILOGUE_PIECES)  # the valid epilogue pieces of a shared tail tile


# Device family IDs mirror ``persistent_bf16_bwd_contract.TaskFamily``; 1 and 8 are unused: the shared
# and routed SwiGLU backward run in the Down dgrad epilogues (_run_down_swiglu_epilogue,
# _run_routed_down_swapab_epilogue).
SHARED_DOWN_DGRAD = 0
SHARED_GATE_UP_DGRAD = 2
SHARED_WGRAD_DOWN = 3
SHARED_WGRAD_GATE = 4
SHARED_WGRAD_UP = 5
REPLAY_GATE_UP = 6
# No replay SwiGLU kind: the routed Down dgrad epilogue writes w*Hidden itself.
ROUTED_DOWN_DGRAD = 7
ROUTED_GATE_UP_DGRAD = 9
ROUTED_WGRAD_DOWN = 10
ROUTED_WGRAD_GATE = 11
ROUTED_WGRAD_UP = 12


# Backward variant of the TMA block-load atom (Wgrad boundary-wait hook).
@dsl_user_op
def _make_tma_block_load_fn(
    atom: cute.CopyAtom,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    *,
    tma_desc_ptr=None,
    loc=None,
    ip=None,
):
    """Bind one CUTLASS-4.5 block TMA copy to a staged operand.

    ``tma_desc_ptr`` overrides the descriptor the atom carries, which is how a routed weight held
    in split per-expert allocations is read: the atom is built over one representative expert and
    ``routed_descriptors`` supplies the rest, since the allocations differ only in base address.
    Leaving it ``None`` uses the atom's own descriptor, which is every other load here.
    """

    source = cute.group_modes(src_tensor, 0, cute.rank(src_tensor) - 1, loc=loc, ip=ip)
    destination = cute.group_modes(dst_tensor, 0, cute.rank(dst_tensor) - 1, loc=loc, ip=ip)

    @dsl_user_op
    def copy_tma(src_idx, dst_idx, *, loc=None, ip=None, **kwargs):
        if tma_desc_ptr is not None:
            kwargs.setdefault("tma_desc_ptr", tma_desc_ptr)
        block_copy(
            atom,
            source[None, src_idx],
            destination[None, dst_idx],
            loc=loc,
            ip=ip,
            **kwargs,
        )

    return copy_tma


# The task->expert decode method (TaskDecode, contract) is a compile-time attribute of the kernel:
# SMEM_BINARY (E <= BWD_EXPERT_TABLE_CAP, one table entry per expert, no global read in the decode),
# SMEM_GROUP (E > BWD_EXPERT_TABLE_CAP, one entry per group of ceil(E / cap) experts, at most stride - 1
# global reads inside the group) and LINEAR_SCAN (a walk over tokens_per_expert, no table; never selected
# by the host).  The expert count, the group stride and every size stay runtime values;
# BWD_EXPERT_TABLE_CAP is the only static capacity (the backward's own 512-entry table; the forward keeps
# its EXPERT_TABLE_CAP).


@cute.struct
class _PersistentBwdSharedStorage:
    clc_mbarriers: cute.struct.MemRange[cutlass.Int64, 2 * CLC_DEPTH]
    clc_response: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, CLC_DEPTH * CLC_RESPONSE_BYTES // 4], 16]
    clc_drain_mbarriers: cute.struct.MemRange[cutlass.Int64, 2 * CLC_DRAIN_DEPTH]
    clc_drain_responses: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, CLC_DRAIN_DEPTH * CLC_RESPONSE_BYTES // 4], 16]
    gemm_ab_mbarriers: cute.struct.MemRange[cutlass.Int64, 2 * MLP_LOAD_STAGES]
    gemm_acc_mbarriers: cute.struct.MemRange[cutlass.Int64, 2 * MLP_ACC_STAGES]
    tmem_dealloc_mbarrier: cutlass.Int64
    tmem_holding_buffer: cutlass.Int32
    # Communication CTAs: one transaction barrier per stream buffer (one per warp: its 32 KiB buffer).
    comm_mbarriers: cute.struct.MemRange[cutlass.Int64, _bwd_dynamic_task_stream.MBARRIERS]
    # Compute CTAs: paired Gate/Up replay pipeline (AB stages) and its accumulator stage.
    paired_gate_up_mbarriers: cute.struct.MemRange[cutlass.Int64, 2 * PAIRED_GATE_UP_STAGES + 2 * PAIRED_ACC_STAGES]
    # Decoded-task records (TASK_INFO_FIELDS) with their full/empty barrier pair per stage.
    task_info_mbarriers: cute.struct.MemRange[cutlass.Int64, 2 * TASK_INFO_STAGES]
    task_info: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, TASK_INFO_STAGES * TASK_INFO_WORDS], 16]
    # One word per warp: the sum of the record words a consumer warp loaded, stored before it releases
    # the record (_load_task_info).  Written only.
    task_info_taken: cute.struct.MemRange[cutlass.Int32, WARPS_PER_CTA]
    # The expert table (expert_valid_end, EXPERT_TABLE_ARENA_OFFSET) lives in the arena's tail.
    arena: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, REUSABLE_ARENA_BYTES], 1024]


# expert_valid_end[g] = valid end (in routed tokens) of the last expert of group g, i.e. the first
# padding row of that expert's MLP_TILE_M-padded segment (its padded end when it has no padding; the
# valid end of an empty expert is its begin).  Group g covers local experts
# [g * stride, min((g + 1) * stride, E)); g < expert_groups = ceil(E / stride) <= BWD_EXPERT_TABLE_CAP.
# SMEM_BINARY: stride 1; SMEM_GROUP: stride = ceil(E / cap); unused by LINEAR_SCAN.  Every index is
# < expert_groups.
# One table serves both lookups: the padded end is the valid end rounded up to MLP_TILE_M
# (padding is at most MLP_TILE_M - 1 rows, so both lie in the same MLP_TILE_M block), and a task
# target row is a multiple of MLP_TILE_M, so `padded_end > target` <=> `valid_end > target`:
# the owner search compares the valid ends directly.  The routed Wgrad K range stops at the
# valid end rounded up to MLP_TILE_K (padding rows are zero, so skipping whole padding K blocks is
# exact), and the swap-AB Gate/Up dgrad tile takes its N from the valid rows left in its block.
# It sits in the compute CTAs' arena after everything a compute role stages there (GEMM A/B/D,
# 221184 B), in the 8 KiB only the communication slots use: communication CTAs build it too (one
# code path) but never read it, and they start using their slots only after the build's closing CTA
# barrier.
EXPERT_TABLE_ARENA_OFFSET = GEMM_ARENA_BYTES
assert EXPERT_TABLE_ARENA_OFFSET + BWD_EXPERT_TABLE_CAP * 4 <= REUSABLE_ARENA_BYTES


assert _PersistentBwdSharedStorage.size_in_bytes() == DYNAMIC_SHARED_MEMORY_BYTES


def shared_storage_layout() -> tuple[tuple[str, int, int], ...]:
    """(name, byte offset, byte size) of every _PersistentBwdSharedStorage member, in order."""
    offsets = _PersistentBwdSharedStorage._offsets
    annotations = _PersistentBwdSharedStorage._annotations
    return tuple((name, int(offsets[name]), _storage_member_bytes(annotations[name])) for name in annotations)


_STORAGE_LAYOUT = {name: (offset, size) for name, offset, size in shared_storage_layout()}
_ARENA_OFFSET = _STORAGE_LAYOUT["arena"][0]
_TASK_INFO_OFFSET, _TASK_INFO_BYTES = _STORAGE_LAYOUT["task_info"]
assert _ARENA_OFFSET % 1024 == 0 and _ARENA_OFFSET + REUSABLE_ARENA_BYTES == DYNAMIC_SHARED_MEMORY_BYTES
# The record ring lies with the barriers in the first KiB, before the arena.
assert _TASK_INFO_BYTES == TASK_INFO_STAGES * TASK_INFO_WORDS * 4 and _TASK_INFO_OFFSET % 16 == 0
assert _TASK_INFO_OFFSET + _TASK_INFO_BYTES <= _ARENA_OFFSET


class _BwdCollective(GemmDefaultSm100):
    """Raw BF16 M256xN256xK64 collective shared by ABt and AtB."""

    #: What ``configure`` requires of D.  The AtB collective writes FP32 when the weight gradients
    #: accumulate into float32 main_grad buffers.
    d_dtype_expected = BFloat16
    #: Epilogue staging tiles.  One FP32 tile is smaller than three BF16 ones, so it fits the same
    #: arena region.
    epi_stages = MLP_BF16_D_TILES

    @classmethod
    def _compute_stages(cls, *args, **kwargs):
        del args, kwargs
        return MLP_ACC_STAGES, MLP_LOAD_STAGES, cls.epi_stages, 0

    def __init__(self) -> None:
        super().__init__(
            acc_dtype=Float32,
            a_dtype=BFloat16,
            mma_tiler_mnk=(MLP_TILE_M, MLP_TILE_N, MLP_TILE_K),
            cluster_shape_mnk=CLUSTER_SHAPE,
            use_clc_persistence=False,
            use_pdl=False,
        )
        self.mma_warp_id = UMMA_WARP
        self.scheduler_warp_id = CLC_WARP
        self.epi_load_warp_id = BF16_IDLE_WARP
        self.ab_load_warp_id = AB_LOAD_WARP
        assert self.epilog_warp_id == (0, 1, 2, 3)
        assert self.threads_per_cta == THREADS_PER_CTA

    def configure(
        self,
        tensor_a: cute.Tensor,
        tensor_b: cute.Tensor,
        tensor_d: cute.Tensor,
    ) -> None:
        self.a_dtype = tensor_a.element_type
        self.b_dtype = tensor_b.element_type
        self.a_mma_dtype = self.a_dtype
        self.b_mma_dtype = self.b_dtype
        self.a_unpack = False
        self.b_unpack = False
        self.a_smem_dtype = self.a_dtype
        self.b_smem_dtype = self.b_dtype
        self.d_dtype = tensor_d.element_type
        self.c_dtype = None
        self.sf_dtype = None
        self.a_layout = LayoutEnum.from_tensor(tensor_a)
        self.b_layout = LayoutEnum.from_tensor(tensor_b)
        self.d_layout = LayoutEnum.from_tensor(tensor_d)
        self.c_layout = None
        self.a_major_mode = self.a_layout.mma_major_mode()
        self.b_major_mode = self.b_layout.mma_major_mode()
        self.varlen_m = False
        self.varlen_k = False
        self._setup_attributes(self.EpilogueArguments(), VarlenArguments())
        assert self.a_dtype is BFloat16
        assert self.b_dtype is BFloat16
        assert self.d_dtype is self.d_dtype_expected
        assert self.use_2cta_instrs
        assert self.cta_tile_shape_mnk == (128, 256, 64)
        assert cute.cosize(self.a_smem_layout_staged.outer) == 49152
        assert cute.cosize(self.b_smem_layout_staged.outer) == 49152


class _BwdCollectiveFp32D(_BwdCollective):
    """The AtB collective writing FP32 weight gradients straight into main_grad.

    The Tensor Core accumulator is FP32 already, so accumulating into a float32 main_grad never
    rounds to bfloat16.  One staging tile, not three: four live FP32 epilogue tiles spill, and one
    fits where the BF16 three do.
    """

    d_dtype_expected = Float32
    epi_stages = 1


@cute.jit
def _is_wgrad_kind(kind: Int32) -> Int32:
    result = Int32(0)
    if kind >= Int32(SHARED_WGRAD_DOWN) and kind <= Int32(SHARED_WGRAD_UP):
        result = Int32(1)
    elif kind >= Int32(ROUTED_WGRAD_DOWN):
        result = Int32(1)
    return result


@cute.jit
def _round_up_i32(value: Int32, unit: cutlass.Constexpr) -> Int32:
    """``value`` (>= 0) rounded up to the next multiple of the power-of-two ``unit``."""
    return (value + Int32(unit - 1)) & Int32(-unit)


@cute.jit
def _routed_down_n(tile_valid: Int32) -> Int32:
    """Swap-AB N of a routed Down dgrad tile: the block's valid rows (>= 1) rounded up to the MLP_TILE_K-row
    Wgrad K block (a multiple of the instruction's 16).  The routed Wgrads read the dGate / dUp / w*Hidden
    rows up to that rounded end (_expert_k_range_rt), so the fused epilogue writes exactly those rows, each
    as the unshrunk tile wrote it; the rows past N are read by nothing."""
    return cutlass.max(_round_up_i32(tile_valid, MLP_TILE_K), Int32(MLP_TILE_K))


@cute.jit
def _store_task_info(
    s_task_info: cute.Tensor,
    stage: Int32,
    active: Int32,
    needs_drain: Int32,
    kind: Int32,
    generation: Int32,
    valid: Int32,
    expert: Int32,
    row: Int32,
    column: Int32,
    k_begin: Int32,
    k_tiles: Int32,
    wgrad_first: Int32,
    wgrad_has_next: Int32,
    wgrad_empty: Int32,
    tile_valid: Int32,
) -> None:
    """Write one decoded-task record (TASK_INFO_FIELDS order) into stage ``stage``; one thread."""
    base = stage * Int32(TASK_INFO_WORDS)
    s_task_info[base + _TI["active"]] = active
    s_task_info[base + _TI["needs_drain"]] = needs_drain
    s_task_info[base + _TI["kind"]] = kind
    s_task_info[base + _TI["generation"]] = generation
    s_task_info[base + _TI["valid"]] = valid
    s_task_info[base + _TI["expert"]] = expert
    s_task_info[base + _TI["row"]] = row
    s_task_info[base + _TI["column"]] = column
    s_task_info[base + _TI["k_begin"]] = k_begin
    s_task_info[base + _TI["k_tiles"]] = k_tiles
    s_task_info[base + _TI["wgrad_first"]] = wgrad_first
    s_task_info[base + _TI["wgrad_has_next"]] = wgrad_has_next
    s_task_info[base + _TI["wgrad_empty"]] = wgrad_empty
    s_task_info[base + _TI["tile_valid"]] = tile_valid


@cute.jit
def _load_task_info(
    s_task_info: cute.Tensor, stage: Int32, s_task_taken: cute.Tensor, warp: Int32
) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32]:
    """Read the record of stage ``stage`` (TASK_INFO_FIELDS order); whole warp.  Every word is
    marked warp-uniform: the values drive the role dispatch, MMA/TMA descriptors and loop bounds,
    and a plain shared-memory load is not provably uniform to the compiler.

    The caller releases the stage right after this call and uses the values after the release; the
    CLC warp then rewrites the record.  A shared-memory load whose result no instruction has read
    when the release executes can return the next record's word, and the compiler moves the uses
    (the compares, the warp-uniform shuffles) behind a release that does not depend on them.  So
    the consumption is made explicit here: the sum of every loaded word is stored to this warp's
    word of ``s_task_taken``.  The store needs every load, and a shared-memory store stays in front
    of the mbarrier arrive that follows it (the order every producer relies on).  Nothing reads the
    stored word."""
    base = stage * Int32(TASK_INFO_WORDS)
    words = [s_task_info[base + index] for index in range(len(TASK_INFO_FIELDS))]
    s_task_taken[warp] = sum(words[1:], words[0])
    return tuple([cute.arch.make_warp_uniform(word) for word in words])


@cute.jit
def _swiglu_backward_element(
    gate: Float32,
    up: Float32,
    d_out: Float32,
    swiglu_limit: Float32,
    is_clamped: cutlass.Constexpr,
    is_situ: cutlass.Constexpr = False,
    situ_beta1=None,
    situ_beta2=None,
) -> tuple[Float32, Float32, Float32]:
    """One element of the fused SwiGLU-backward: (d_gate, d_up, hidden) from the saved gate / up and the
    router-weighted dH (quack's dswiglu).  ``is_clamped`` clamps the inputs and zeroes the gradient of a
    clamped one.  ``is_situ``: SiTU-GLU's backward instead (never clamped)."""
    if cutlass.const_expr(is_situ):
        return situ_bwd(gate, up, d_out, situ_beta1, situ_beta2)
    gate_grad = Boolean(True)
    up_grad = Boolean(True)
    if cutlass.const_expr(is_clamped):
        gate_grad = gate <= swiglu_limit
        up_grad = up >= -swiglu_limit and up <= swiglu_limit
        gate = cute.arch.fmin(gate, swiglu_limit)
        up = cute.arch.fmin(cute.arch.fmax(up, -swiglu_limit), swiglu_limit)
    d_gate, d_up, hidden = dswiglu(gate, up, d_out)
    if cutlass.const_expr(is_clamped):
        if not gate_grad:
            d_gate = Float32(0.0)
        if not up_grad:
            d_up = Float32(0.0)
    return d_gate, d_up, hidden


@cute.jit
def _padded_segment_valid_end(schedule_peer_rank: cute.Tensor, segment_begin: Int32, segment_end: Int32) -> Int32:
    """Valid end of one expert's routed segment [segment_begin, segment_end) (MLP_TILE_M-padded,
    the padding rows carry schedule peer rank -1 and are the tail of the last row block): the first
    padding row, found by a binary search over that last row block (log2(MLP_TILE_M) dependent
    global reads).  An empty segment returns its begin."""
    valid_end = segment_end
    if segment_end > segment_begin:
        low = segment_end - Int32(MLP_TILE_M)
        high = segment_end
        while low < high:
            middle = Int32(Uint32(low + high) // Uint32(2))
            if schedule_peer_rank[middle] < Int32(0):
                high = middle
            else:
                low = middle + Int32(1)
        valid_end = low
    return valid_end


@cute.jit
def _expert_tile_from_segment(
    expert: Int32,
    segment_begin: Int32,
    segment_end: Int32,
    mini_begin: Int32,
    mini_end: Int32,
    column_blocks: Int32,
    local_task: Int32,
) -> tuple[Int32, Int32, Int32, Int32]:
    """Tile of ``local_task`` inside the owner expert's block grid in this minibatch: the expert
    owns the rows [max(segment_begin, mini_begin), min(segment_end, mini_end)) of the minibatch
    (every bound is a multiple of MLP_TILE_M) and ``row_blocks * column_blocks`` consecutive tasks,
    swizzled inside that grid; the experts before it fill the minibatch rows up to its first row."""
    first = cutlass.max(segment_begin, mini_begin)
    last = cutlass.min(segment_end, mini_end)
    row_blocks = Int32(Uint32(last - first) // Uint32(MLP_TILE_M))
    remaining = local_task - Int32(Uint32(first - mini_begin) // Uint32(MLP_TILE_M)) * column_blocks
    local_row, column_tile = _swizzled_tile_i32(row_blocks, column_blocks, remaining)
    return Int32(1), expert, Int32(Uint32(first) // Uint32(MLP_TILE_M)) + local_row, column_tile


@cute.jit
def _decode_linear_scan(
    tokens_per_expert: cute.Tensor,
    expert_valid_end: cute.Tensor,
    expert_stride: Int32,
    expert_groups: Int32,
    num_tokens: Int32,
    generation: Int32,
    local_task: Int32,
    column_blocks: Int32,
    minibatch_size: Int32,
) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32]:
    """TaskDecode.LINEAR_SCAN: walk tokens_per_expert from expert 0 until the expert whose task
    range inside the minibatch holds ``local_task`` (only the expert, its token offset, the remaining
    task count and a found flag are loop-carried).  The shared table is not used.  The host never
    selects it; it is reachable only through the internal selector ``select_task_decode``."""
    num_experts = Int32(tokens_per_expert.shape[0])
    mini_begin = generation * minibatch_size
    mini_end = mini_begin + cutlass.min(minibatch_size, num_tokens - mini_begin)
    valid = Int32(0)
    owner = Int32(0)
    row_tile = Int32(0)
    column_tile = Int32(0)
    segment_begin = Int32(0)
    segment_end = Int32(0)
    expert = Int32(0)
    offset = Int32(0)
    remaining = local_task
    found = Int32(0)
    while expert < num_experts and found == Int32(0):
        expert_end = offset + Int32(tokens_per_expert[expert])
        first = cutlass.max(offset, mini_begin)
        last = cutlass.min(expert_end, mini_end)
        row_blocks = Int32(Uint32(cutlass.max(Int32(0), last - first)) // Uint32(MLP_TILE_M))
        tasks = row_blocks * column_blocks
        if remaining < tasks:
            found = Int32(1)
        else:
            remaining = remaining - tasks
            offset = expert_end
            expert = expert + Int32(1)
    if found != Int32(0):
        segment_begin = offset
        segment_end = offset + Int32(tokens_per_expert[expert])
        valid, owner, row_tile, column_tile = _expert_tile_from_segment(expert, offset, segment_end, mini_begin, mini_end, column_blocks, local_task)
    return valid, owner, row_tile, column_tile, segment_begin, segment_end


@cute.jit
def _decode_smem_binary(
    tokens_per_expert: cute.Tensor,
    expert_valid_end: cute.Tensor,
    expert_stride: Int32,
    expert_groups: Int32,
    num_tokens: Int32,
    generation: Int32,
    local_task: Int32,
    column_blocks: Int32,
    minibatch_size: Int32,
    search_begin: Int32,
) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32]:
    """TaskDecode.SMEM_BINARY: the owner is the first expert e with ``valid_end[e] > target``
    over the shared table of all E experts (stride 1, expert_groups = E <= BWD_EXPERT_TABLE_CAP; see
    the table comment for why the valid ends can be compared with the MLP_TILE_M-aligned target);
    its padded segment is [round_up(valid_end[e - 1]), round_up(valid_end[e])) (begin 0 for e = 0).
    No global read.  Selected by the host whenever E <= BWD_EXPERT_TABLE_CAP.  The search covers
    [search_begin, E): the caller passes an expert whose predecessors all end at or before the target
    (0 when it knows none)."""
    mini_begin = generation * minibatch_size
    mini_end = mini_begin + cutlass.min(minibatch_size, num_tokens - mini_begin)
    valid = Int32(0)
    owner = Int32(0)
    row_tile = Int32(0)
    column_tile = Int32(0)
    segment_begin = Int32(0)
    segment_end = Int32(0)
    target = mini_begin + Int32(Uint32(local_task) // Uint32(column_blocks)) * Int32(MLP_TILE_M)
    if target < mini_end:
        if target < expert_valid_end[expert_groups - Int32(1)]:
            expert = search_begin
            high = expert_groups
            while expert < high:
                middle = Int32(Uint32(expert + high) // Uint32(2))
                if expert_valid_end[middle] > target:
                    high = middle
                else:
                    expert = middle + Int32(1)
            if expert > Int32(0):
                segment_begin = _round_up_i32(expert_valid_end[expert - Int32(1)], MLP_TILE_M)
            segment_end = _round_up_i32(expert_valid_end[expert], MLP_TILE_M)
            valid, owner, row_tile, column_tile = _expert_tile_from_segment(expert, segment_begin, segment_end, mini_begin, mini_end, column_blocks, local_task)
    return valid, owner, row_tile, column_tile, segment_begin, segment_end


@cute.jit
def _decode_smem_group(
    tokens_per_expert: cute.Tensor,
    expert_valid_end: cute.Tensor,
    expert_stride: Int32,
    expert_groups: Int32,
    num_tokens: Int32,
    generation: Int32,
    local_task: Int32,
    column_blocks: Int32,
    minibatch_size: Int32,
) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32]:
    """TaskDecode.SMEM_GROUP: the table holds one entry per group of expert_stride =
    ceil(E / BWD_EXPERT_TABLE_CAP) experts (expert_groups = ceil(E / stride) entries; group g covers
    experts [g * stride, min((g + 1) * stride, E)), so the last group may be short).  The owner
    group is the first g with ``valid_end[g] > target``; the owner expert is found by walking the
    group's experts in global tokens_per_expert, at most stride - 1 reads, because the group's
    last expert ends at round_up(valid_end[g]).  Selected by the host when E > BWD_EXPERT_TABLE_CAP."""
    num_experts = Int32(tokens_per_expert.shape[0])
    mini_begin = generation * minibatch_size
    mini_end = mini_begin + cutlass.min(minibatch_size, num_tokens - mini_begin)
    valid = Int32(0)
    owner = Int32(0)
    row_tile = Int32(0)
    column_tile = Int32(0)
    segment_begin = Int32(0)
    segment_end = Int32(0)
    target = mini_begin + Int32(Uint32(local_task) // Uint32(column_blocks)) * Int32(MLP_TILE_M)
    if target < mini_end:
        if target < expert_valid_end[expert_groups - Int32(1)]:
            group = _cu_end_upper_bound(expert_valid_end, expert_groups, target)
            offset = Int32(0)
            if group > Int32(0):
                offset = _round_up_i32(expert_valid_end[group - Int32(1)], MLP_TILE_M)
            group_end = _round_up_i32(expert_valid_end[group], MLP_TILE_M)
            expert = group * expert_stride
            last_expert = cutlass.min(expert + expert_stride, num_experts) - Int32(1)
            expert_tokens = Int32(0)
            found = Int32(0)
            while expert < last_expert and found == Int32(0):
                expert_tokens = Int32(tokens_per_expert[expert])
                if target < offset + expert_tokens:
                    found = Int32(1)
                else:
                    offset = offset + expert_tokens
                    expert = expert + Int32(1)
            if found == Int32(0):
                expert_tokens = group_end - offset  # the group's last expert (target < group_end)
            segment_begin = offset
            segment_end = offset + expert_tokens
            valid, owner, row_tile, column_tile = _expert_tile_from_segment(expert, segment_begin, segment_end, mini_begin, mini_end, column_blocks, local_task)
    return valid, owner, row_tile, column_tile, segment_begin, segment_end


# Slots of the `runtime` tuple handed to every task body. 0..19 are bound by
# __call__; the kernel appends num_tokens/num_experts (read from device tensors),
# the shared expert table with its group geometry, the peer count, a counter and
# the schedule peer ranks.
_RT_MACROBATCH_SIZE = 0
_RT_MINIBATCH_SIZE = 1
_RT_NUM_COMM_SMS = 2
_RT_MINIBATCHES_PER_MACROBATCH = 3
_RT_TASK_OWNER_BUCKET_SHIFT = 4
_RT_ROUTED_TASKS = 5
_RT_REPLAY_GATE_UP_TASKS = 6
_RT_REPLAY_TASKS = 7
_RT_ROUTED_BUFFERS_DONE_REQUIRED = 8
_RT_ROUTED_HIDDEN_SIZE = 9
_RT_INTERMEDIATE_SIZE = 10
_RT_TOPK = 11
_RT_I_TILES = 12
_RT_ROUTED_H_TILES = 13
_RT_S_TILES = 14
_RT_ROUTED_WGRAD_TILES = 15
_RT_HIDDEN_READY_REQUIRED = 16
_RT_SWIGLU_LIMIT = 17
# The shared chain adds one slot: its H tiles.  Its K tiles and Wgrad tiles (I tiles x H tiles) are derived in
# the task bodies, as the routed ones are.
_RT_SHARED_H_TILES = 18
_RT_NUM_TOKENS = 19
_RT_NUM_EXPERTS = 20
_RT_EXPERT_VALID_END = 21  # shared table (arena tail, EXPERT_TABLE_ARENA_OFFSET), built by _build_expert_valid_end
_RT_EXPERT_STRIDE = 22  # experts per table entry (1 for SMEM_BINARY / LINEAR_SCAN)
_RT_EXPERT_GROUPS = 23  # table entries in use (<= BWD_EXPERT_TABLE_CAP)
_RT_EP_SIZE = 24  # runtime peer count = peer table length
_RT_REVERSE_DISPATCH_DONE = 25  # counter: dX tiles of generation g pushed out of the d_x ring (one arrive per tile)
_RT_SCHEDULE_PEER_RANK = 26  # schedule peer ranks (-1 = padding row): valid end of an expert not in the table
# Variant-only slots, None in every other variant (the kernel takes them as arguments only there).
_RT_SHARED_HIDDEN_SIZE = 27  # has_tail: a 128-wide H tail is two K tiles, not a whole H tile
_RT_SITU_BETA1 = 28  # is_situ: SiTU-GLU's betas
_RT_SITU_BETA2 = 29


@cute.jit
def _runtime_expert_owner_rt(
    runtime: tuple,
    tokens_per_expert: cute.Tensor,
    task_decode: cutlass.Constexpr,
    num_tokens: Int32,
    generation: Int32,
    local_task: Int32,
    column_blocks: Int32,
    minibatch_size: Int32,
    cached_expert: Int32,
    cached_begin: Int32,
    cached_end: Int32,
) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32]:
    """Map a minibatch-local routed task index to (valid, expert, row tile, column tile) with the
    compile-time decode method (TaskDecode): exactly one of the three helpers is compiled into a
    kernel variant.  Tasks are expert-major inside the minibatch; see the helpers.
    The warp keeps the padded token segment [cached_begin, cached_end) of the last expert it
    decoded (the table is fixed for the launch, so the segment stays valid across minibatches and
    task kinds): a task whose target row lies in it belongs to that expert (the segments partition
    the padded rows) and skips the lookup.  A target past the cached segment (the usual case: a
    warp's tasks ascend within a task kind) starts the SMEM_BINARY search after the cached expert
    instead of at 0; a target before it (the next kind or minibatch restarts at the minibatch's
    first row) searches the whole table.  Returns the tile and the updated (expert, begin, end)."""
    mini_begin = generation * minibatch_size
    mini_end = mini_begin + cutlass.min(minibatch_size, num_tokens - mini_begin)
    target = mini_begin + Int32(Uint32(local_task) // Uint32(column_blocks)) * Int32(MLP_TILE_M)
    valid = Int32(0)
    owner = Int32(0)
    row_tile = Int32(0)
    column_tile = Int32(0)
    segment_begin = cached_begin
    segment_end = cached_end
    if target < mini_end and target >= cached_begin and target < cached_end:
        valid, owner, row_tile, column_tile = _expert_tile_from_segment(
            cached_expert, cached_begin, cached_end, mini_begin, mini_end, column_blocks, local_task
        )
    else:
        found_begin = Int32(0)
        found_end = Int32(0)
        if cutlass.const_expr(task_decode == TaskDecode.LINEAR_SCAN):
            valid, owner, row_tile, column_tile, found_begin, found_end = _decode_linear_scan(
                tokens_per_expert,
                runtime[_RT_EXPERT_VALID_END],
                runtime[_RT_EXPERT_STRIDE],
                runtime[_RT_EXPERT_GROUPS],
                num_tokens,
                generation,
                local_task,
                column_blocks,
                minibatch_size,
            )
        elif cutlass.const_expr(task_decode == TaskDecode.SMEM_BINARY):
            # Every expert up to the cached one ends at or before cached_end <= target.
            search_begin = Int32(0)
            if cached_end > Int32(0) and target >= cached_end:
                search_begin = cached_expert + Int32(1)
            valid, owner, row_tile, column_tile, found_begin, found_end = _decode_smem_binary(
                tokens_per_expert,
                runtime[_RT_EXPERT_VALID_END],
                runtime[_RT_EXPERT_STRIDE],
                runtime[_RT_EXPERT_GROUPS],
                num_tokens,
                generation,
                local_task,
                column_blocks,
                minibatch_size,
                search_begin,
            )
        else:
            valid, owner, row_tile, column_tile, found_begin, found_end = _decode_smem_group(
                tokens_per_expert,
                runtime[_RT_EXPERT_VALID_END],
                runtime[_RT_EXPERT_STRIDE],
                runtime[_RT_EXPERT_GROUPS],
                num_tokens,
                generation,
                local_task,
                column_blocks,
                minibatch_size,
            )
        if valid != Int32(0):
            cached_expert = owner
            segment_begin = found_begin
            segment_end = found_end
    return valid, owner, row_tile, column_tile, cached_expert, segment_begin, segment_end


@cute.jit
def _expert_k_range_rt(runtime: tuple, tokens_per_expert: cute.Tensor, expert: Int32, task_decode: cutlass.Constexpr) -> tuple[Int32, Int32, Int32]:
    """Routed Wgrad K range of ``expert``: (padded token begin, padded token end, valid end: the
    expert's first padding row, not rounded), with
    the compile-time decode method.  SMEM_BINARY: two table reads (one for expert 0).  SMEM_GROUP: the group's
    begin from the table plus at most stride - 1 global token counts before the expert and the
    expert's own; the valid end is the table's for the group's last expert and otherwise a bounded
    search of schedule_peer_rank.  LINEAR_SCAN: an O(E) sum and the bounded search."""
    expert_valid_end = runtime[_RT_EXPERT_VALID_END]
    begin = Int32(0)
    end = Int32(0)
    valid_end = Int32(0)
    if cutlass.const_expr(task_decode == TaskDecode.SMEM_BINARY):
        if expert > Int32(0):
            begin = _round_up_i32(expert_valid_end[expert - Int32(1)], MLP_TILE_M)
        valid_end = expert_valid_end[expert]
        end = _round_up_i32(valid_end, MLP_TILE_M)
    else:
        index = Int32(0)
        last_expert = Int32(-1)  # the expert whose valid end the table holds (none for LINEAR_SCAN)
        if cutlass.const_expr(task_decode == TaskDecode.SMEM_GROUP):
            expert_stride = runtime[_RT_EXPERT_STRIDE]
            group = Int32(Uint32(expert) // Uint32(expert_stride))
            if group > Int32(0):
                begin = _round_up_i32(expert_valid_end[group - Int32(1)], MLP_TILE_M)
            index = group * expert_stride
            last_expert = cutlass.min(index + expert_stride, runtime[_RT_NUM_EXPERTS]) - Int32(1)
            valid_end = expert_valid_end[group]
        while index < expert:
            begin = begin + Int32(tokens_per_expert[index])
            index = index + Int32(1)
        end = begin + Int32(tokens_per_expert[expert])
        if expert != last_expert:
            valid_end = _padded_segment_valid_end(runtime[_RT_SCHEDULE_PEER_RANK], begin, end)
    return begin, end, valid_end


@cute.jit
def _expert_valid_end_rt(runtime: tuple, expert: Int32, segment_begin: Int32, segment_end: Int32, task_decode: cutlass.Constexpr) -> Int32:
    """Valid end (first padding row) of ``expert``, whose padded segment [segment_begin, segment_end)
    the task decode already found, with the compile-time decode method: SMEM_BINARY reads the table;
    SMEM_GROUP reads it for the group's last expert and otherwise searches schedule_peer_rank over the
    segment's last row block (_padded_segment_valid_end); LINEAR_SCAN always searches."""
    valid_end = Int32(0)
    if cutlass.const_expr(task_decode == TaskDecode.SMEM_BINARY):
        valid_end = runtime[_RT_EXPERT_VALID_END][expert]
    elif cutlass.const_expr(task_decode == TaskDecode.SMEM_GROUP):
        expert_stride = runtime[_RT_EXPERT_STRIDE]
        group = Int32(Uint32(expert) // Uint32(expert_stride))
        last_expert = cutlass.min((group + Int32(1)) * expert_stride, runtime[_RT_NUM_EXPERTS]) - Int32(1)
        if expert == last_expert:
            valid_end = runtime[_RT_EXPERT_VALID_END][group]
        else:
            valid_end = _padded_segment_valid_end(runtime[_RT_SCHEDULE_PEER_RANK], segment_begin, segment_end)
    else:
        valid_end = _padded_segment_valid_end(runtime[_RT_SCHEDULE_PEER_RANK], segment_begin, segment_end)
    return valid_end


@cute.jit
def _build_expert_valid_end(
    tokens_per_expert: cute.Tensor,
    schedule_peer_rank: cute.Tensor,
    expert_valid_end: cute.Tensor,
    task_decode: cutlass.Constexpr,
    thread: Int32,
) -> tuple[Int32, Int32]:
    """Build the per-CTA shared expert_valid_end table and return (expert_stride, expert_groups).
    SMEM_BINARY: stride 1, one entry per expert (the host admits E <= BWD_EXPERT_TABLE_CAP);
    SMEM_GROUP: stride = ceil(E / BWD_EXPERT_TABLE_CAP), expert_groups = ceil(E / stride) <=
    BWD_EXPERT_TABLE_CAP; LINEAR_SCAN: no table (returns (1, E)).  Every thread sums the padded token
    counts of its groups from global tokens_per_expert (indices < expert_groups), thread 0 turns
    them into the inclusive prefix (at most BWD_EXPERT_TABLE_CAP dependent shared-memory adds per CTA
    per launch), then every thread replaces its groups' padded ends by the valid end of the
    group's last expert.  Ends with a CTA barrier, so the table is visible to every role."""
    num_experts = Int32(tokens_per_expert.shape[0])
    expert_stride = Int32(1)
    expert_groups = num_experts
    if cutlass.const_expr(task_decode == TaskDecode.SMEM_GROUP):
        expert_stride = _ceil_div_i32(num_experts, Int32(BWD_EXPERT_TABLE_CAP))
        expert_groups = _ceil_div_i32(num_experts, expert_stride)
    if cutlass.const_expr(task_decode != TaskDecode.LINEAR_SCAN):
        group = thread
        while group < expert_groups:
            expert = group * expert_stride
            group_end = cutlass.min(expert + expert_stride, num_experts)
            tokens = Int32(0)
            while expert < group_end:
                tokens = tokens + Int32(tokens_per_expert[expert])
                expert = expert + Int32(1)
            expert_valid_end[group] = tokens
            group = group + Int32(THREADS_PER_CTA)
        cute.arch.sync_threads()
        if thread == Int32(0):
            index = Int32(1)
            while index < expert_groups:
                expert_valid_end[index] = expert_valid_end[index] + expert_valid_end[index - Int32(1)]
                index = index + Int32(1)
        cute.arch.sync_threads()
        group = thread
        while group < expert_groups:
            padded_end = expert_valid_end[group]
            last_expert = cutlass.min((group + Int32(1)) * expert_stride, num_experts) - Int32(1)
            last_begin = padded_end - Int32(tokens_per_expert[last_expert])
            expert_valid_end[group] = _padded_segment_valid_end(schedule_peer_rank, last_begin, padded_end)
            group = group + Int32(THREADS_PER_CTA)
    cute.arch.sync_threads()
    return expert_stride, expert_groups


@cute.jit
def _minibatch_task_prefix_i32(
    generation: Int32,
    completed_segment_offsets: cute.Tensor,
    saved_minibatches: Int32,
    routed_tasks: Int32,
    replay_tasks: Int32,
    routed_wgrad_tiles: Int32,
) -> Int32:
    replay_minibatches = cutlass.max(Int32(0), generation - Int32(saved_minibatches))
    return (
        generation * Int32(routed_tasks)
        + replay_minibatches * Int32(replay_tasks)
        + Int32(3) * routed_wgrad_tiles * Int32(completed_segment_offsets[generation])
    )


def _epilog_gmem_copy_fn(collective, atom, mD_mn, tile_shape_mn, epi_tile, sD, tile_coord, tma_desc_ptr=None):
    """QuACK's ``epilog_gmem_copy_and_partition`` with a descriptor override.

    The same partitioning, plus the ``tma_desc_ptr`` that method's signature does not take: a
    split main_grad buffer is one expert's own allocation, named by a descriptor rather than
    sliced out of an L mode.  ``None`` leaves the atom's own descriptor, which is every other
    epilogue here.
    """
    del collective
    gD = cute.local_tile(mD_mn, tile_shape_mn, tile_coord[:2])
    tDgD = cute.zipped_divide(gD, epi_tile)
    is_s2g = isinstance(atom.op, (cpasync.CopyBulkTensorTileS2GOp, cpasync.CopyReduceBulkTensorTileS2GOp))
    source, destination = (sD, tDgD) if is_s2g else (tDgD, sD)
    return copy_utils.tma_get_copy_fn(
        atom,
        cta_coord=0,
        cta_layout=cute.make_layout(1),
        src_tensor=source,
        dst_tensor=destination,
        tma_desc_ptr=tma_desc_ptr,
    )


def routed_weight_hie(tensor):
    """A routed weight with the output feature contiguous: ``[H, I, E]``, or ``[H, I]`` split.

    The dgrad reads the weights MN-major.  Grouped, that is the ``[E, I, H]`` parameter permuted
    with E last; split, it is one expert's own allocation, which has no E mode to move.  The rank
    is what tells the two apart, here as in the forward.

    A host helper over torch views, deliberately not a ``@cute.jit`` one -- it runs in the
    argument builder, before anything is traced.
    """
    view = row_major_view(tensor)
    return view.permute(1, 0) if view.ndim == 2 else view.permute(2, 1, 0)


@cute.jit
def _aligned_bf16_tensor(tensor: cute.Tensor, leading_dim: cutlass.Constexpr) -> cute.Tensor:
    # Only the actual unit-stride axis is static. All other strides retain
    # runtime 64-bit metadata with the host-validated TMA alignment assumption.
    strides = tuple(1 if axis == leading_dim else cute.assume(tensor.stride[axis], divby=8) for axis in range(cute.rank(tensor)))
    return cute.make_tensor(tensor.iterator, cute.make_layout(tensor.shape, stride=strides))


class _PersistentBf16BwdMega:
    def __init__(
        self,
        is_clamped: bool,
        task_decode: TaskDecode,
        split_weights: bool = False,
        accumulate_wgrad: bool = False,
        bf16_main_grad: bool = False,
        split_main_grads: bool = False,
        is_situ: bool = False,
        has_shared: bool = True,
        has_tail: bool = False,
        recompute_forward_context: bool = False,
        comm_tile_shape: CommTileShape = DEFAULT_COMM_TILE_SHAPE,
    ) -> None:
        # The communication tile shape (contract CommTileShape, CuteMoKConfig.comm_tile_shape): the one compile-time
        # parameter of the communication CTAs, like the compute tile sizes.  A communication warp moves one tile at a
        # time through its 32 KiB buffer; the tile counts derived from it use the runtime sizes.
        if comm_tile_shape not in COMM_TILE_SHAPES:
            raise ValueError(f"comm_tile_shape must be one of {COMM_TILE_SHAPES}")
        self.comm_tile_shape = comm_tile_shape
        # The clamp is compile time; its numerical limit is a runtime Float32.
        self.is_clamped = is_clamped
        # The forward context holds no routed activations: the first macrobatch's X / Gate / Up are replayed
        # like every later one's (X pulled by the communication CTAs, Gate / Up recomputed), and every
        # routed row takes the per-generation task stream.  The context's routed tensors are scratch.
        self.recompute_forward_context = bool(recompute_forward_context)
        # False for a routed-only layer (no shared expert): its shared tensors are placeholders
        # the kernel builds atoms over and never touches -- there are no shared tasks.
        self.has_shared = bool(has_shared)
        # H or I has a 128-wide tail (size % 256 != 0): one more tile, clipped where it is not a TMA access.
        # Compile time because H and I are model parameters and the flag only selects whether the tail code is
        # generated: an aligned model runs loops without predicates or tail branches.  It doubles this kernel's
        # executors (one for every aligned shape, one for every shape with a tail) as one more entry of the
        # executor key, which a model keeps for its whole run.  The values of H and I stay runtime arguments:
        # nothing is specialised on a size.
        self.has_tail = bool(has_tail)
        # SiTU-GLU instead of SwiGLU, likewise compile time; its betas are runtime Float32s.  Never with a clamp.
        self.is_situ = bool(is_situ)
        if self.is_situ and self.is_clamped:
            raise ValueError("SiTU cannot be combined with swiglu_limit")
        # Task->expert decode method, likewise compile-time (contract TaskDecode).
        self.task_decode = TaskDecode(task_decode)
        # The routed experts may be independent per-expert allocations instead of one grouped
        # [E, ...] parameter.  The dgrad then reads one expert's own matrix, named by a TMA
        # descriptor, instead of slicing the parameter's L mode -- in two views each, MN-major for
        # the swap-AB dgrad and K-major for the replay.  Compile-time, because it changes the
        # operands' rank.
        self.split_weights = bool(split_weights)
        # Fused gradient accumulation: the weight gradients go into main_grad buffers, which already
        # hold earlier microbatches' contributions, so every tile is additive and nothing is ever
        # zero-filled.  bf16_main_grad picks the buffer's dtype: bfloat16 reuses this epilogue
        # unchanged, float32 keeps the accumulator's own precision all the way to memory.
        self.accumulate_wgrad = bool(accumulate_wgrad)
        self.bf16_main_grad = bool(bf16_main_grad)
        if self.bf16_main_grad and not self.accumulate_wgrad:
            raise ValueError("bf16_main_grad is only meaningful for fused wgrad accumulation")
        #: Whether the Wgrad epilogue stores FP32 rather than rounding to bfloat16.
        self.fp32_main_grad = self.accumulate_wgrad and not self.bf16_main_grad
        # Whether the main_grad buffers are per-expert allocations too.  Independent of split_weights:
        # the dgrad can read split weights while the gradients go to one grouped buffer.
        self.split_main_grads = bool(split_main_grads)
        if self.split_main_grads and not self.accumulate_wgrad:
            raise ValueError("split main_grad buffers only exist for fused accumulation")
        # The three collectives alias one AB6/ACC2 barrier ring, one arena,
        # and one FP32 TMEM allocation.  ABt covers the dgrad family, AtB
        # covers Wgrad, and Replay is independently configured for the
        # forward-style NKE weight major.
        self.abt = _BwdCollective()
        # The Wgrad collective writes FP32 when it accumulates into float32 main_grad buffers.
        self.atb = _BwdCollectiveFp32D() if self.fp32_main_grad else _BwdCollective()
        self.replay = _BwdCollective()

    def _saved_minibatches(self, minibatches_per_macrobatch):
        """Generations whose routed X / Gate / Up the forward context holds: the first macrobatch's, or none
        with recompute_forward_context.  Generation g is replayed when g >= this (trace time: the argument itself
        otherwise)."""
        return Int32(0) if self.recompute_forward_context else minibatches_per_macrobatch

    def _dense_wgrad_limit(self, macrobatch_size):
        """Routed row count up to which the task stream is the dense one (every minibatch's dgrad tasks, then
        the per-expert Wgrad grid, no replay): one macrobatch, or no routed row at all with recompute_forward_context,
        whose replays need the per-generation stream (trace time: the argument itself otherwise)."""
        return Int32(0) if self.recompute_forward_context else macrobatch_size

    @cute.jit
    def __call__(
        self,
        x_peer_table: cute.Tensor,
        d_y_peer_table: cute.Tensor,
        d_x_peer_table: cute.Tensor,
        router_peer_table: cute.Tensor,
        d_router_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens_tensor: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        minibatch_expert_offsets: cute.Tensor,
        completed_segment_offsets: cute.Tensor,
        completed_segment_experts: cute.Tensor,
        task_owner_buckets: cute.Tensor,
        x_shared: cute.Tensor,
        x_routed: cute.Tensor,
        gate_shared: cute.Tensor,
        gate_routed: cute.Tensor,
        up_shared: cute.Tensor,
        up_routed: cute.Tensor,
        hidden_shared: cute.Tensor,
        hidden_routed: cute.Tensor,
        d_y_shared: cute.Tensor,
        w_shared_down_ih: cute.Tensor,
        w_routed_down_ihe: cute.Tensor,
        w_shared_gate_hi: cute.Tensor,
        w_routed_gate_hie: cute.Tensor,
        w_shared_up_hi: cute.Tensor,
        w_routed_up_hie: cute.Tensor,
        w_routed_gate_nke: cute.Tensor,
        w_routed_up_nke: cute.Tensor,
        x_shared_ht: cute.Tensor,
        x_routed_ht: cute.Tensor,
        hidden_shared_it: cute.Tensor,
        hidden_routed_it: cute.Tensor,
        d_y_shared_ht: cute.Tensor,
        d_y_routed_ht: cute.Tensor,
        d_gate_shared_it: cute.Tensor,
        d_gate_routed_it: cute.Tensor,
        d_up_shared_it: cute.Tensor,
        d_up_routed_it: cute.Tensor,
        d_x_shared: cute.Tensor,
        d_x_routed: cute.Tensor,
        d_gate_shared: cute.Tensor,
        d_gate_routed: cute.Tensor,
        d_up_shared: cute.Tensor,
        d_up_routed: cute.Tensor,
        d_y_routed: cute.Tensor,
        d_w_shared_down: cute.Tensor,
        d_w_shared_gate: cute.Tensor,
        d_w_shared_up: cute.Tensor,
        d_w_routed_down_hie: cute.Tensor,
        d_w_routed_gate_ihe: cute.Tensor,
        d_w_routed_up_ihe: cute.Tensor,
        router_weights: cute.Tensor,
        d_router_weight_partials: cute.Tensor,
        d_y_routed_ready: cute.Tensor,
        d_gate_up_ready: cute.Tensor,
        d_x_routed_ready: cute.Tensor,
        replayed_x_routed_ready: cute.Tensor,
        replayed_gate_up_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        wgrad_read_consumed: cute.Tensor,
        wgrad_segment_done: cute.Tensor,
        reverse_dispatch_done: cute.Tensor,
        num_local_tokens: Int32,
        schedule_capacity: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        num_comm_sms: Int32,
        topk: Int32,
        task_owner_bucket_shift: Int32,
        swiglu_limit: Float32,
        # (5, num_local_experts) int64 base addresses -- the Gate, Up and Down matrices the dgrad
        # reads MN-major, then the Gate and Up it reads K-major -- and the bytes their descriptors
        # live in.  One-element placeholders unless split_weights is set.  ``up_row_offset`` is the
        # Up half's first row inside a combined FC1 parameter, 0 otherwise: split allocations have
        # no E mode and a retargeted descriptor describes the whole [2I, H], so the half is chosen
        # by the coordinate -- an element offset, which stays right when I % 256 != 0 puts Up
        # mid-way through a 256-row tile.
        routed_storage_tables: cute.Tensor,
        routed_descriptors: cute.Tensor,
        # Nonzero when routed_descriptors has to be built by this launch: once per set of address
        # tables (persistent_bf16_fwd.RoutedDescriptors).
        build_descriptors: Int32,
        up_row_offset: Int32,
        # The Up half's first row inside a combined FC1 main_grad buffer.  Separate from the
        # weights' own offset: whether Gate and Up share one main_grad buffer is a property of those
        # buffers, not of the weights.
        main_grad_up_row_offset: Int32,
        # SiTU-GLU's two betas; unread unless is_situ.
        situ_beta1: Float32,
        situ_beta2: Float32,
        stream: cuda.CUstream,
    ) -> None:

        x_shared = _aligned_bf16_tensor(x_shared, 1)
        x_routed = _aligned_bf16_tensor(x_routed, 1)
        gate_shared = _aligned_bf16_tensor(gate_shared, 1)
        gate_routed = _aligned_bf16_tensor(gate_routed, 1)
        up_shared = _aligned_bf16_tensor(up_shared, 1)
        up_routed = _aligned_bf16_tensor(up_routed, 1)
        hidden_shared = _aligned_bf16_tensor(hidden_shared, 1)
        hidden_routed = _aligned_bf16_tensor(hidden_routed, 1)
        d_y_shared = _aligned_bf16_tensor(d_y_shared, 1)
        w_shared_down_ih = _aligned_bf16_tensor(w_shared_down_ih, 0)
        w_routed_down_ihe = _aligned_bf16_tensor(w_routed_down_ihe, 0)
        w_shared_gate_hi = _aligned_bf16_tensor(w_shared_gate_hi, 0)
        w_routed_gate_hie = _aligned_bf16_tensor(w_routed_gate_hie, 0)
        w_shared_up_hi = _aligned_bf16_tensor(w_shared_up_hi, 0)
        w_routed_up_hie = _aligned_bf16_tensor(w_routed_up_hie, 0)
        w_routed_gate_nke = _aligned_bf16_tensor(w_routed_gate_nke, 1)
        w_routed_up_nke = _aligned_bf16_tensor(w_routed_up_nke, 1)
        if cutlass.const_expr(not self.split_weights):
            # The Gate/Up dgrad reads Up MN-major, where a combined FC1's 2I is the contraction axis and its Up half
            # starts up_row_offset = I elements along it (0 for separate weights, which keep their shape).  Up gets
            # its own (H, I, E) view here, once per launch, so the dgrad's K loop tiles it from 0 as it tiles
            # separate weights, instead of adding the offset to every K tile's coordinate.  (Split weights have
            # no such view -- a retargeted descriptor describes one expert's whole [2I, H] -- and keep the
            # coordinate, see _run_tma_gemm_task.)
            w_routed_up_hie = cute.make_tensor(
                w_routed_up_hie.iterator + Int64(up_row_offset) * Int64(w_routed_up_hie.stride[1]),
                cute.make_layout(
                    (w_routed_up_hie.shape[0], Int32(w_routed_up_hie.shape[1]) - up_row_offset, w_routed_up_hie.shape[2]), stride=w_routed_up_hie.stride
                ),
            )
        x_shared_ht = _aligned_bf16_tensor(x_shared_ht, 0)
        x_routed_ht = _aligned_bf16_tensor(x_routed_ht, 0)
        hidden_shared_it = _aligned_bf16_tensor(hidden_shared_it, 0)
        hidden_routed_it = _aligned_bf16_tensor(hidden_routed_it, 0)
        d_y_shared_ht = _aligned_bf16_tensor(d_y_shared_ht, 0)
        d_y_routed_ht = _aligned_bf16_tensor(d_y_routed_ht, 0)
        d_gate_shared_it = _aligned_bf16_tensor(d_gate_shared_it, 0)
        d_gate_routed_it = _aligned_bf16_tensor(d_gate_routed_it, 0)
        d_up_shared_it = _aligned_bf16_tensor(d_up_shared_it, 0)
        d_up_routed_it = _aligned_bf16_tensor(d_up_routed_it, 0)
        d_x_shared = _aligned_bf16_tensor(d_x_shared, 1)
        d_x_routed = _aligned_bf16_tensor(d_x_routed, 1)
        d_gate_shared = _aligned_bf16_tensor(d_gate_shared, 1)
        d_gate_routed = _aligned_bf16_tensor(d_gate_routed, 1)
        d_up_shared = _aligned_bf16_tensor(d_up_shared, 1)
        d_up_routed = _aligned_bf16_tensor(d_up_routed, 1)
        d_y_routed = _aligned_bf16_tensor(d_y_routed, 1)
        d_w_shared_down = _aligned_bf16_tensor(d_w_shared_down, 1)
        d_w_shared_gate = _aligned_bf16_tensor(d_w_shared_gate, 1)
        d_w_shared_up = _aligned_bf16_tensor(d_w_shared_up, 1)
        d_w_routed_down_hie = _aligned_bf16_tensor(d_w_routed_down_hie, 1)
        d_w_routed_gate_ihe = _aligned_bf16_tensor(d_w_routed_gate_ihe, 1)
        d_w_routed_up_ihe = _aligned_bf16_tensor(d_w_routed_up_ihe, 1)
        if cutlass.const_expr(not self.accumulate_wgrad):
            # Fresh dW of a combined FC1 is one (2I, H, E) buffer under both names, Up main_grad_up_row_offset = I
            # rows down.  Its stores (and, past the first macrobatch, reduce-adds) see each half as its own
            # (I, H, E) matrix: an I % 256 != 0 tail tile's half past I is then clipped by the TMA instead of
            # storing zeros over Up's first rows.  Separate Gate / Up have offset 0 and keep their own shapes.
            # (Accumulating into main_grad adds those zeros, which leaves Up as it is.)
            fc1_rows = Int32(d_w_routed_gate_ihe.shape[0]) - main_grad_up_row_offset
            d_w_routed_gate_ihe = cute.make_tensor(
                d_w_routed_gate_ihe.iterator, cute.make_layout((fc1_rows, *d_w_routed_gate_ihe.shape[1:]), stride=d_w_routed_gate_ihe.stride)
            )
            d_w_routed_up_ihe = cute.make_tensor(
                d_w_routed_up_ihe.iterator + Int64(main_grad_up_row_offset) * Int64(d_w_routed_up_ihe.stride[0]),
                cute.make_layout((fc1_rows, *d_w_routed_up_ihe.shape[1:]), stride=d_w_routed_up_ihe.stride),
            )
        abt = self.abt
        atb = self.atb
        replay = self.replay
        # The D operand fixes only the epilogue dtype and major mode (BF16, row-major) of the ABt TMA
        # store target d_x_shared; the routed dX (swap-AB) and Down dgrad epilogues store through their own atoms.
        abt.configure(d_y_shared, w_shared_down_ih, d_x_shared)
        atb.configure(d_y_shared_ht, hidden_shared_it, d_w_shared_down)
        # ``x_routed`` and ``gate_routed`` are N-major [M, K]/[M, N].
        # The NKE expert slice is [N, K] K-major because it is the zero-copy
        # (E, I, H)->(I, H, E) forward-weight view.  It must not inherit the
        # N-major ABt B collective configured from the Down dgrad operand.
        # One expert's matrix: split allocations are that already, grouped ones slice the L mode.
        w_routed_gate_nke_0 = w_routed_gate_nke
        if cutlass.const_expr(not self.split_weights):
            w_routed_gate_nke_0 = w_routed_gate_nke[None, None, 0]
        replay.configure(x_routed, w_routed_gate_nke_0, gate_routed)
        assert abt.num_tmem_alloc_cols == atb.num_tmem_alloc_cols

        abt_a_layout = cute.slice_(abt.a_smem_layout_staged, (None, None, None, 0))
        abt_b_layout = cute.slice_(abt.b_smem_layout_staged, (None, None, None, 0))
        atb_a_layout = cute.slice_(atb.a_smem_layout_staged, (None, None, None, 0))
        atb_b_layout = cute.slice_(atb.b_smem_layout_staged, (None, None, None, 0))
        replay_a_layout = cute.slice_(replay.a_smem_layout_staged, (None, None, None, 0))
        replay_b_layout = cute.slice_(replay.b_smem_layout_staged, (None, None, None, 0))
        replay_epi_layout = replay.epi_smem_layout_staged
        tma_op_abt = cpasync.CopyBulkTensorTileG2SOp(abt.cta_group)
        tma_op_atb = cpasync.CopyBulkTensorTileG2SOp(atb.cta_group)
        tma_op_replay = cpasync.CopyBulkTensorTileG2SOp(replay.cta_group)

        abt_a_pairs = [
            cute.nvgpu.make_tiled_tma_atom_A(
                tma_op_abt,
                tensor,
                abt_a_layout,
                abt.mma_tiler,
                abt.tiled_mma,
                abt.cluster_layout_vmnk.shape,
            )
            for tensor in (
                d_y_shared,
                d_gate_shared,
                d_up_shared,
            )
        ]
        abt_b_pairs = [
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_op_abt,
                tensor,
                abt_b_layout,
                abt.mma_tiler,
                abt.tiled_mma,
                abt.cluster_layout_vmnk.shape,
            )
            for tensor in (
                w_shared_down_ih,
                w_shared_gate_hi,
                w_shared_up_hi,
            )
        ]
        # Routed dgrad, swap-AB (_swapab_fwd).  Operand A: this CTA's 128 output features of the expert's
        # Wg and Wu (K = I; the (H, I, E) views) or Wd (K = H; the (I, H, E) view), MN-major (the output
        # feature contiguous), loaded as two 64-feature x 64-K slabs per stage into the A ring with the
        # ring's 128 B swizzle (tiled per expert in the kernel).  Operand B: the block's dGate / dUp or dY
        # rows, K-major, staged like the Replay collective's K-major B.  D of the Gate/Up dX: dX as the
        # M-major accumulator (H features, ring rows), 128 features x 32 tokens per store; the Down dH
        # never leaves the CTA (fused SwiGLU-backward).
        swap_slab_layout = cute.make_composed_layout(abt_a_layout.inner, 0, cute.make_layout((_SWAP_SLAB_ROWS, MLP_TILE_K), stride=(1, _SWAP_SLAB_ROWS)))
        swap_a_pairs = [
            cpasync.make_tiled_tma_atom(tma_op_abt, tensor, swap_slab_layout, (_SWAP_SLAB_ROWS, MLP_TILE_K))
            for tensor in (w_routed_gate_hie, w_routed_up_hie, w_routed_down_ihe)
        ]
        swap_b_pairs = [
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_op_replay,
                tensor,
                replay_b_layout,
                replay.mma_tiler,
                replay.tiled_mma,
                replay.cluster_layout_vmnk.shape,
            )
            for tensor in (d_gate_routed, d_up_routed, d_y_routed)
        ]
        d_x_routed_features_major = cute.make_tensor(d_x_routed.iterator, cute.select(d_x_routed.layout, mode=[1, 0]))
        swap_d_atom, swap_d_tensor = abt._make_tma_epi_atoms_and_tensors(
            d_x_routed_features_major,
            _swapab_fwd.d_smem_layout(abt.epi_tile, MLP_BF16_D_TILES),
            abt.epi_tile,
            op_type="store",
        )
        replay_a_pairs = [
            cute.nvgpu.make_tiled_tma_atom_A(
                tma_op_replay,
                x_routed,
                replay_a_layout,
                replay.mma_tiler,
                replay.tiled_mma,
                replay.cluster_layout_vmnk.shape,
            )
        ]
        replay_b_pairs = [
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_op_replay,
                tensor,
                replay_b_layout,
                replay.mma_tiler,
                replay.tiled_mma,
                replay.cluster_layout_vmnk.shape,
            )
            for tensor in (
                w_routed_gate_nke,
                w_routed_up_nke,
            )
        ]
        atb_a_pairs = [
            cute.nvgpu.make_tiled_tma_atom_A(
                tma_op_atb,
                tensor,
                atb_a_layout,
                atb.mma_tiler,
                atb.tiled_mma,
                atb.cluster_layout_vmnk.shape,
            )
            for tensor in (
                d_y_shared_ht,
                d_gate_shared_it,
                d_up_shared_it,
                d_y_routed_ht,
                d_gate_routed_it,
                d_up_routed_it,
            )
        ]
        atb_b_pairs = [
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_op_atb,
                tensor,
                atb_b_layout,
                atb.mma_tiler,
                atb.tiled_mma,
                atb.cluster_layout_vmnk.shape,
            )
            for tensor in (
                hidden_shared_it,
                x_shared_ht,
                hidden_routed_it,
                x_routed_ht,
            )
        ]
        abt_d_pairs = [
            abt._make_tma_epi_atoms_and_tensors(
                tensor,
                abt.epi_smem_layout_staged,
                abt.epi_tile,
                op_type="store",
            )
            for tensor in (d_x_shared,)
        ]
        replay_d_pairs = [
            replay._make_tma_epi_atoms_and_tensors(
                tensor,
                replay_epi_layout,
                replay.epi_tile,
                op_type="store",
            )
            for tensor in (
                gate_routed,
                up_routed,
            )
        ]
        atb_d_pairs = [
            atb._make_tma_epi_atoms_and_tensors(
                tensor,
                atb.epi_smem_layout_staged,
                atb.epi_tile,
                op_type="store",
            )
            for tensor in (
                d_w_shared_down,
                d_w_shared_gate,
                d_w_shared_up,
                d_w_routed_down_hie,
                d_w_routed_gate_ihe,
                d_w_routed_up_ihe,
            )
        ]
        atb_d_add_pairs = [
            atb._make_tma_epi_atoms_and_tensors(
                tensor,
                atb.epi_smem_layout_staged,
                atb.epi_tile,
                op_type="add",
            )
            for tensor in (
                d_w_routed_down_hie,
                d_w_routed_gate_ihe,
                d_w_routed_up_ihe,
                # 3..5: the shared weight gradients, which accumulate into main_grad too -- only in
                # that variant, the one that names them.
                *((d_w_shared_down, d_w_shared_gate, d_w_shared_up) if self.accumulate_wgrad else ()),
            )
        ]
        abt.num_tma_load_bytes = (cute.size_in_bytes(BFloat16, abt_a_layout) + cute.size_in_bytes(BFloat16, abt_b_layout)) * 2
        atb.num_tma_load_bytes = (cute.size_in_bytes(BFloat16, atb_a_layout) + cute.size_in_bytes(BFloat16, atb_b_layout)) * 2
        replay.num_tma_load_bytes = (cute.size_in_bytes(BFloat16, replay_a_layout) + cute.size_in_bytes(BFloat16, replay_b_layout)) * 2
        assert abt.num_tma_load_bytes == atb.num_tma_load_bytes
        # Replay differs only in operand major mode.  Preserve the exact
        # shared execution envelope before aliasing the AB6/ACC2 resources.
        assert replay.cta_tile_shape_mnk == abt.cta_tile_shape_mnk
        assert replay.cluster_shape_mnk == abt.cluster_shape_mnk
        assert replay.cluster_layout_vmnk.shape == abt.cluster_layout_vmnk.shape
        assert cute.cosize(replay.a_smem_layout_staged.outer) == cute.cosize(abt.a_smem_layout_staged.outer)
        assert cute.cosize(replay.b_smem_layout_staged.outer) == cute.cosize(abt.b_smem_layout_staged.outer)
        assert cute.cosize(replay.epi_smem_layout_staged.outer) == cute.cosize(abt.epi_smem_layout_staged.outer)
        assert replay.num_tma_load_bytes == abt.num_tma_load_bytes
        assert replay.num_mcast_ctas_a == abt.num_mcast_ctas_a
        assert replay.num_mcast_ctas_b == abt.num_mcast_ctas_b
        assert replay.num_tmem_alloc_cols == abt.num_tmem_alloc_cols

        # The shared chain works in x_shared's hidden, the routed chain (rings,
        # communication, routed tasks and Wgrads) in x_routed's; equal for an ordinary MoE.
        shared_hidden_size = Int32(x_shared.shape[1])
        routed_hidden_size = Int32(x_routed.shape[1])
        # Off the routed context, which a routed-only layer has too.
        intermediate_size = Int32(gate_routed.shape[1])
        # The intermediate extent may have a 128-wide tail (I % 256 != 0).  Count it as a whole
        # tile: the host task counts, the metadata kernel's per-minibatch task count and every
        # readiness-counter index below must use the same ceil, or a consumer of the tail tile
        # waits on a counter slot nobody arrives at.
        i_tiles = _ceil_div_i32(intermediate_size, Int32(MLP_TILE_N))
        # H may have a 128-wide tail too (H % 256 != 0): one more tile, the same ceil everywhere H
        # is counted (task counts, readiness targets, Wgrad grids, the metadata kernel).
        shared_h_tiles = _ceil_div_i32(shared_hidden_size, Int32(MLP_TILE_N))
        if cutlass.const_expr(not self.has_shared):
            # The kernel reads num_local_tokens as the shared expert's rows alone (its task counts
            # and the readiness counters' offset past the shared row blocks), and the shared
            # Wgrad grid is I x shared-H tiles: with both zero there is no shared task.
            num_local_tokens = Int32(0)
            shared_h_tiles = Int32(0)
        routed_h_tiles = _ceil_div_i32(routed_hidden_size, Int32(MLP_TILE_N))
        s_tiles = intermediate_size // Int32(SWIGLU_TILE_N)
        shared_wgrad_tiles = i_tiles * shared_h_tiles
        routed_wgrad_tiles = i_tiles * routed_h_tiles
        hidden_ready_required = Int32(2) * s_tiles
        minibatches_per_macrobatch = macrobatch_size // minibatch_size
        segment_capacity = _ceil_div_i32(schedule_capacity, macrobatch_size) + Int32(tokens_per_expert.shape[0]) - Int32(1)
        shared_down_tasks = (num_local_tokens // MLP_TILE_M) * i_tiles
        shared_gate_up_tasks = (num_local_tokens // MLP_TILE_M) * shared_h_tiles
        # SwiGLU-backward is fused into the Down dgrad epilogues: neither chain has SwiGLU tasks.
        shared_tasks = shared_down_tasks + shared_gate_up_tasks + Int32(3) * shared_wgrad_tiles
        routed_down_tasks = (minibatch_size // MLP_TILE_M) * i_tiles
        routed_gate_up_tasks = (minibatch_size // MLP_TILE_M) * routed_h_tiles
        routed_tasks = routed_down_tasks + routed_gate_up_tasks
        replay_gate_up_tasks = routed_down_tasks
        replay_tasks = replay_gate_up_tasks  # Gate/Up replay only; no replay SwiGLU
        # Slot reuse waits only for the compute-side readers of dY/X.  The comm CTAs' own
        # dX reads are published on reverse_dispatch_done and awaited by the next writer of
        # that slot (the fused Down dgrad epilogue / Gate-Up dgrad of generation g + mpm), so the pull of
        # g + mpm does not wait for every comm CTA to finish pushing g.
        routed_buffers_done_required = Int32(CLUSTER_SIZE) * routed_tasks
        capacity_minibatches = _ceil_div_i32(schedule_capacity, minibatch_size)
        saved_minibatches = self._saved_minibatches(minibatches_per_macrobatch)
        capacity_compute_tasks = (
            shared_tasks
            + capacity_minibatches * routed_tasks
            + cutlass.max(Int32(0), capacity_minibatches - saved_minibatches) * replay_tasks
            + segment_capacity * Int32(3) * routed_wgrad_tiles
        )
        comm_clusters = num_comm_sms // CLUSTER_SIZE
        capacity_clusters = comm_clusters + capacity_compute_tasks
        scheduler_params = utils.ClcDynamicPersistentTileSchedulerParams((capacity_clusters * CLUSTER_SIZE, 1, 1), CLUSTER_SHAPE)

        runtime = (  # slot order = _RT_* constants
            macrobatch_size,  # _RT_MACROBATCH_SIZE
            minibatch_size,  # _RT_MINIBATCH_SIZE
            num_comm_sms,  # _RT_NUM_COMM_SMS
            minibatches_per_macrobatch,  # _RT_MINIBATCHES_PER_MACROBATCH
            task_owner_bucket_shift,  # _RT_TASK_OWNER_BUCKET_SHIFT
            routed_tasks,  # _RT_ROUTED_TASKS
            replay_gate_up_tasks,  # _RT_REPLAY_GATE_UP_TASKS
            replay_tasks,  # _RT_REPLAY_TASKS
            routed_buffers_done_required,  # _RT_ROUTED_BUFFERS_DONE_REQUIRED
            routed_hidden_size,  # _RT_ROUTED_HIDDEN_SIZE
            intermediate_size,  # _RT_INTERMEDIATE_SIZE
            topk,  # _RT_TOPK
            i_tiles,  # _RT_I_TILES
            routed_h_tiles,  # _RT_ROUTED_H_TILES
            s_tiles,  # _RT_S_TILES
            routed_wgrad_tiles,  # _RT_ROUTED_WGRAD_TILES
            hidden_ready_required,  # _RT_HIDDEN_READY_REQUIRED
            swiglu_limit,  # _RT_SWIGLU_LIMIT
            shared_h_tiles,  # _RT_SHARED_H_TILES
        )
        # One descriptor array per routed-weight atom, filled by its own launch before the mega
        # kernel: that one is persistent, so every CTA would otherwise race to write the same
        # descriptors.  Order: the three the dgrad reads MN-major, then the two the replay reads
        # K-major -- the same tensors in two views, so two schemas and two arrays.
        # The kernel takes the variant-only arguments only in their variants: None is a compile-time constant,
        # not a kernel parameter, so every other variant's kernel is the one without the feature.
        descriptors = None
        kernel_main_grad_up_row_offset = None
        kernel_shared_hidden_size = None
        kernel_situ_beta1 = None
        kernel_situ_beta2 = None
        if cutlass.const_expr(self.accumulate_wgrad):
            kernel_main_grad_up_row_offset = main_grad_up_row_offset
        if cutlass.const_expr(self.has_tail):
            kernel_shared_hidden_size = shared_hidden_size
        if cutlass.const_expr(self.is_situ):
            kernel_situ_beta1 = situ_beta1
            kernel_situ_beta2 = situ_beta2
        if cutlass.const_expr(self.split_weights):
            descriptors = cute.make_ptr(
                cutlass.Uint8, routed_descriptors.iterator.toint(), cute.AddressSpace.gmem, assumed_align=routed_descriptors_pkg.BYTES_PER_DESCRIPTOR
            )
            weight_atoms = (swap_a_pairs[0][0], swap_a_pairs[1][0], swap_a_pairs[2][0], replay_b_pairs[0][0], replay_b_pairs[1][0])
            if cutlass.const_expr(self.split_main_grads):
                # The reduce-add atoms of the three main_grad buffers; the plain store atoms
                # describe the same tensors, and split buffers always accumulate.
                weight_atoms = (*weight_atoms, atb_d_add_pairs[0][0], atb_d_add_pairs[1][0], atb_d_add_pairs[2][0])
            if build_descriptors != Int32(0):
                routed_descriptors_pkg.build_routed_descriptors(
                    weight_atoms, routed_storage_tables, descriptors, Int32(cute.size(routed_storage_tables.shape[1])), stream
                )
        self.kernel(
            runtime,
            abt.tiled_mma,
            atb.tiled_mma,
            replay.tiled_mma,
            [pair[0] for pair in abt_a_pairs],
            [pair[1] for pair in abt_a_pairs],
            [pair[0] for pair in abt_b_pairs],
            [pair[1] for pair in abt_b_pairs],
            [pair[0] for pair in swap_a_pairs],
            [pair[1] for pair in swap_a_pairs],
            [pair[0] for pair in swap_b_pairs],
            [pair[1] for pair in swap_b_pairs],
            [pair[0] for pair in replay_a_pairs],
            [pair[1] for pair in replay_a_pairs],
            [pair[0] for pair in replay_b_pairs],
            [pair[1] for pair in replay_b_pairs],
            [pair[0] for pair in atb_a_pairs],
            [pair[1] for pair in atb_a_pairs],
            [pair[0] for pair in atb_b_pairs],
            [pair[1] for pair in atb_b_pairs],
            [pair[0] for pair in abt_d_pairs],
            [pair[1] for pair in abt_d_pairs],
            swap_d_atom,
            swap_d_tensor,
            [pair[0] for pair in replay_d_pairs],
            [pair[1] for pair in replay_d_pairs],
            [pair[0] for pair in atb_d_pairs],
            [pair[1] for pair in atb_d_pairs],
            [pair[0] for pair in atb_d_add_pairs],
            [pair[1] for pair in atb_d_add_pairs],
            abt.cluster_layout_vmnk,
            abt.a_smem_layout_staged,
            abt.b_smem_layout_staged,
            abt.epi_smem_layout_staged,
            abt.epi_tile,
            replay.a_smem_layout_staged,
            replay.b_smem_layout_staged,
            replay_epi_layout,
            replay.epi_tile,
            atb.a_smem_layout_staged,
            atb.b_smem_layout_staged,
            atb.epi_smem_layout_staged,
            atb.epi_tile,
            x_peer_table,
            d_y_peer_table,
            d_x_peer_table,
            router_peer_table,
            d_router_peer_table,
            schedule_peer_rank,
            schedule_peer_token_idx,
            num_tokens_tensor,
            tokens_per_expert,
            minibatch_expert_offsets,
            completed_segment_offsets,
            completed_segment_experts,
            task_owner_buckets,
            x_shared,
            x_routed,
            gate_shared,
            gate_routed,
            up_shared,
            up_routed,
            hidden_shared,
            hidden_routed,
            d_y_shared,
            d_x_shared,
            d_x_routed,
            d_gate_shared,
            d_gate_routed,
            d_up_shared,
            d_up_routed,
            d_y_routed,
            d_w_shared_down,
            d_w_shared_gate,
            d_w_shared_up,
            d_w_routed_down_hie,
            d_w_routed_gate_ihe,
            d_w_routed_up_ihe,
            router_weights,
            d_router_weight_partials,
            d_y_routed_ready,
            d_gate_up_ready,
            d_x_routed_ready,
            replayed_x_routed_ready,
            replayed_gate_up_ready,
            routed_buffers_done,
            wgrad_read_consumed,
            wgrad_segment_done,
            reverse_dispatch_done,
            scheduler_params,
            comm_clusters,
            shared_down_tasks,
            shared_gate_up_tasks,
            shared_tasks,
            routed_down_tasks,
            routed_gate_up_tasks,
            routed_tasks,
            num_local_tokens,
            minibatch_size,
            num_comm_sms,
            up_row_offset,
            descriptors,
            kernel_main_grad_up_row_offset,
            kernel_shared_hidden_size,
            kernel_situ_beta1,
            kernel_situ_beta2,
        ).launch(
            grid=(capacity_clusters * CLUSTER_SIZE, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            cluster=CLUSTER_SHAPE,
            smem=DYNAMIC_SHARED_MEMORY_BYTES,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _decode_compute_task(
        self,
        runtime: tuple,
        compute_index: Int32,
        completed_segment_offsets: cute.Tensor,
        completed_segment_experts: cute.Tensor,
        task_owner_buckets: cute.Tensor,
        shared_down_tasks: Int32,
        shared_gate_up_tasks: Int32,
        shared_tasks: Int32,
        routed_down_tasks: Int32,
        routed_gate_up_tasks: Int32,
        routed_tasks: Int32,
    ) -> tuple[Int32, Int32, Int32, Int32]:
        saved_minibatches = self._saved_minibatches(runtime[_RT_MINIBATCHES_PER_MACROBATCH])
        task_owner_bucket_shift = runtime[_RT_TASK_OWNER_BUCKET_SHIFT]
        replay_gate_up_tasks = runtime[_RT_REPLAY_GATE_UP_TASKS]
        replay_tasks = runtime[_RT_REPLAY_TASKS]
        routed_wgrad_tiles = runtime[_RT_ROUTED_WGRAD_TILES]
        shared_wgrad_tiles = runtime[_RT_I_TILES] * runtime[_RT_SHARED_H_TILES]
        kind = Int32(-1)
        generation = Int32(-1)
        local = compute_index
        expert = Int32(-1)
        if local < Int32(shared_down_tasks):
            kind = Int32(SHARED_DOWN_DGRAD)
        elif local < Int32(shared_down_tasks + shared_gate_up_tasks):
            kind = Int32(SHARED_GATE_UP_DGRAD)
            local = local - Int32(shared_down_tasks)
        elif local < Int32(shared_tasks - 2 * shared_wgrad_tiles):
            kind = Int32(SHARED_WGRAD_DOWN)
            local = local - Int32(shared_tasks - 3 * shared_wgrad_tiles)
        elif local < Int32(shared_tasks - shared_wgrad_tiles):
            kind = Int32(SHARED_WGRAD_GATE)
            local = local - Int32(shared_tasks - 2 * shared_wgrad_tiles)
        elif local < Int32(shared_tasks):
            kind = Int32(SHARED_WGRAD_UP)
            local = local - Int32(shared_tasks - shared_wgrad_tiles)
        else:
            if runtime[_RT_NUM_TOKENS] <= self._dense_wgrad_limit(runtime[_RT_MACROBATCH_SIZE]):
                # At most one macrobatch: every minibatch's dgrad tasks, then the dense per-expert Wgrads.
                routed = local - Int32(shared_tasks)
                minis = _ceil_div_i32(runtime[_RT_NUM_TOKENS], runtime[_RT_MINIBATCH_SIZE])
                if routed < minis * routed_tasks:
                    # Down dgrad of minibatch s + 1 ahead of Gate/Up dgrad of s: Down(0), then per s
                    # [Down(s + 1), Gate/Up(s)] (the last block has Gate/Up only), so the MMA of the
                    # next minibatch's Down covers this minibatch's Down epilogue (fused SwiGLU).
                    # Same task count; one macrobatch has no ring-slot gates, so no new wait cycle.
                    if routed < routed_down_tasks:
                        generation = Int32(0)
                        local = routed
                        kind = Int32(ROUTED_DOWN_DGRAD)
                    else:
                        shifted = routed - routed_down_tasks
                        generation = shifted // routed_tasks
                        local = shifted % routed_tasks
                        kind = Int32(ROUTED_GATE_UP_DGRAD)
                        if generation + Int32(1) < minis:
                            if local < routed_down_tasks:
                                kind = Int32(ROUTED_DOWN_DGRAD)
                                generation = generation + Int32(1)
                            else:
                                local = local - routed_down_tasks
                else:
                    generation = Int32(0)
                    local = routed - minis * routed_tasks
                    matrix_tasks = runtime[_RT_NUM_EXPERTS] * routed_wgrad_tiles
                    # Down tiles first, then Gate and Up tile t of an expert next to each other (they
                    # read the same X slab): pair p -> Gate if p is even, Up if odd, tile p // 2.
                    kind = Int32(ROUTED_WGRAD_DOWN)
                    if local >= matrix_tasks:
                        pair = local - matrix_tasks
                        kind = Int32(ROUTED_WGRAD_GATE) + (pair & Int32(1))
                        local = pair >> Int32(1)
                    expert = local // routed_wgrad_tiles
                    local = local % routed_wgrad_tiles
            else:
                routed = local - Int32(shared_tasks)
                owner_bucket = routed >> Int32(task_owner_bucket_shift)
                generation = Int32(task_owner_buckets[owner_bucket])
                if routed >= _minibatch_task_prefix_i32(
                    generation + Int32(1),
                    completed_segment_offsets,
                    saved_minibatches,
                    routed_tasks,
                    replay_tasks,
                    routed_wgrad_tiles,
                ):
                    generation = generation + Int32(1)
                local = routed - _minibatch_task_prefix_i32(
                    generation,
                    completed_segment_offsets,
                    saved_minibatches,
                    routed_tasks,
                    replay_tasks,
                    routed_wgrad_tiles,
                )
                if generation >= Int32(saved_minibatches):
                    if local < Int32(replay_gate_up_tasks):
                        kind = Int32(REPLAY_GATE_UP)
                    else:
                        local = local - Int32(replay_tasks)
                if kind == Int32(-1):
                    if local < Int32(routed_down_tasks):
                        kind = Int32(ROUTED_DOWN_DGRAD)
                    elif local < Int32(routed_tasks):
                        kind = Int32(ROUTED_GATE_UP_DGRAD)
                        local = local - Int32(routed_down_tasks)
                    else:
                        local = local - Int32(routed_tasks)
                        first_segment = Int32(completed_segment_offsets[generation])
                        segment_count = Int32(completed_segment_offsets[generation + Int32(1)]) - first_segment
                        matrix_tasks = segment_count * routed_wgrad_tiles
                        segment_lane = Int32(0)
                        if local < matrix_tasks:
                            kind = Int32(ROUTED_WGRAD_DOWN)
                            segment_lane = local // routed_wgrad_tiles
                        else:
                            # Gate and Up tile t of a segment next to each other (same X slab).
                            pair = local - matrix_tasks
                            kind = Int32(ROUTED_WGRAD_GATE) + (pair & Int32(1))
                            local = pair >> Int32(1)
                            segment_lane = local // routed_wgrad_tiles
                        expert = Int32(completed_segment_experts[first_segment + segment_lane])
                        local = local % routed_wgrad_tiles
        return kind, generation, local, expert

    @cute.jit
    def _gemm_geometry(
        self,
        runtime: tuple,
        kind: Int32,
        generation: Int32,
        local: Int32,
        expert_from_task: Int32,
        num_tokens: Int32,
        tokens_per_expert: cute.Tensor,
        expert_counts: cute.Tensor,
        num_local_tokens: Int32,
        cached_expert: Int32,
        cached_begin: Int32,
        cached_end: Int32,
    ) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32]:
        """Task geometry of one GEMM task (device function, registers only): validity,
        expert, output tile (row, column), K range (k_begin, k_tiles), for routed
        Wgrad the segment facts (first segment / has a later segment / empty expert), and for the
        routed Down and Gate/Up dgrads the valid rows of their 256-row block (tile_valid, the swap-AB N
        source; 0 for other kinds), followed by the warp's updated owner cache (expert, segment begin,
        segment end).  Reads the shared expert table through _runtime_expert_owner_rt /
        _expert_k_range_rt / _expert_valid_end_rt."""
        macrobatch_size = runtime[_RT_MACROBATCH_SIZE]
        minibatch_size = runtime[_RT_MINIBATCH_SIZE]
        minibatches_per_macrobatch = runtime[_RT_MINIBATCHES_PER_MACROBATCH]
        intermediate_size = runtime[_RT_INTERMEDIATE_SIZE]
        i_tiles = runtime[_RT_I_TILES]
        routed_h_tiles = runtime[_RT_ROUTED_H_TILES]
        shared_h_tiles = runtime[_RT_SHARED_H_TILES]
        valid = Int32(1)
        expert = Int32(0)
        row = Int32(0)
        column = Int32(0)
        k_begin = Int32(0)
        k_tiles = Int32(0)
        # Routed Wgrad segment facts (0 for other kinds): first/last macrobatch segment of
        # the expert and an expert without rows.  Computed here, where the ranges are
        # already known, so the epilogue dispatch adds no divisions on the common path.
        wgrad_first = Int32(1)
        wgrad_has_next = Int32(0)
        wgrad_empty = Int32(0)
        tile_valid = Int32(0)
        if kind == Int32(SHARED_DOWN_DGRAD):
            row, column = _swizzled_tile_i32(Int32(num_local_tokens // MLP_TILE_M), i_tiles, local)
            k_tiles = shared_h_tiles * Int32(MLP_TILE_N // MLP_TILE_K)
            if cutlass.const_expr(self.has_tail):
                k_tiles = runtime[_RT_SHARED_HIDDEN_SIZE] // Int32(MLP_TILE_K)
        elif kind == Int32(SHARED_GATE_UP_DGRAD):
            row, column = _swizzled_tile_i32(Int32(num_local_tokens // MLP_TILE_M), shared_h_tiles, local)
            k_tiles = Int32(intermediate_size // MLP_TILE_K)
        elif kind == Int32(SHARED_WGRAD_DOWN):
            row, column = _swizzled_tile_i32(shared_h_tiles, i_tiles, local)
            k_tiles = Int32(num_local_tokens // MLP_TILE_K)
        elif kind == Int32(SHARED_WGRAD_GATE) or kind == Int32(SHARED_WGRAD_UP):
            row, column = _swizzled_tile_i32(i_tiles, shared_h_tiles, local)
            k_tiles = Int32(num_local_tokens // MLP_TILE_K)
        elif kind == Int32(REPLAY_GATE_UP):
            valid, expert, row, column, cached_expert, cached_begin, cached_end = _runtime_expert_owner_rt(
                runtime,
                expert_counts,
                self.task_decode,
                num_tokens,
                generation,
                local,
                i_tiles,
                minibatch_size,
                cached_expert,
                cached_begin,
                cached_end,
            )
            row = row % Int32(macrobatch_size // MLP_TILE_M)
            k_tiles = routed_h_tiles * Int32(MLP_TILE_N // MLP_TILE_K)
            if cutlass.const_expr(self.has_tail):
                k_tiles = runtime[_RT_ROUTED_HIDDEN_SIZE] // Int32(MLP_TILE_K)
        elif kind == Int32(ROUTED_DOWN_DGRAD):
            valid, expert, row, column, cached_expert, cached_begin, cached_end = _runtime_expert_owner_rt(
                runtime,
                expert_counts,
                self.task_decode,
                num_tokens,
                generation,
                local,
                i_tiles,
                minibatch_size,
                cached_expert,
                cached_begin,
                cached_end,
            )
            # Swap-AB N source (as for the Gate/Up dgrad below; the Down's N rounds it to the Wgrad K block).
            if valid != Int32(0):
                tile_valid = _swapab_fwd.tile_valid_rows(
                    _expert_valid_end_rt(runtime, expert, cached_begin, cached_end, self.task_decode), row * Int32(MLP_TILE_M)
                )
            row = row % Int32(macrobatch_size // MLP_TILE_M)
            k_tiles = routed_h_tiles * Int32(MLP_TILE_N // MLP_TILE_K)
            if cutlass.const_expr(self.has_tail):
                k_tiles = runtime[_RT_ROUTED_HIDDEN_SIZE] // Int32(MLP_TILE_K)
        elif kind == Int32(ROUTED_GATE_UP_DGRAD):
            valid, expert, row, column, cached_expert, cached_begin, cached_end = _runtime_expert_owner_rt(
                runtime,
                expert_counts,
                self.task_decode,
                num_tokens,
                generation,
                local,
                routed_h_tiles,
                minibatch_size,
                cached_expert,
                cached_begin,
                cached_end,
            )
            # Swap-AB N source: the owner's valid rows left in this (global) row block, from the
            # expert's valid end and the segment the decode found; at least 1 for a valid tile.
            if valid != Int32(0):
                tile_valid = _swapab_fwd.tile_valid_rows(
                    _expert_valid_end_rt(runtime, expert, cached_begin, cached_end, self.task_decode), row * Int32(MLP_TILE_M)
                )
            row = row % Int32(macrobatch_size // MLP_TILE_M)
            k_tiles = Int32(intermediate_size // MLP_TILE_K)
        elif kind >= Int32(ROUTED_WGRAD_DOWN):
            expert = expert_from_task
            expert_begin, expert_end, valid_end = _expert_k_range_rt(runtime, expert_counts, expert, self.task_decode)
            macro_begin = (generation // Int32(minibatches_per_macrobatch)) * Int32(macrobatch_size)
            macro_end = macro_begin + cutlass.min(Int32(macrobatch_size), num_tokens - macro_begin)
            segment_begin = cutlass.max(expert_begin, macro_begin)
            segment_end = cutlass.min(expert_end, macro_end)
            if expert_begin < macro_begin:
                wgrad_first = Int32(0)
            if expert_end > macro_begin + Int32(macrobatch_size):
                wgrad_has_next = Int32(1)
            if expert_end == expert_begin:
                wgrad_empty = Int32(1)
                wgrad_has_next = Int32(0)
            if segment_begin < segment_end:
                # Skip whole padding K blocks at the end of the expert segment, but keep at
                # least one K block in the segment and in its last minibatch so the
                # wgrad_read_consumed retire arrivals (counted from the padded ranges by the
                # comm CTAs) are unchanged.
                last_minibatch_begin = (segment_end - Int32(1)) // Int32(minibatch_size) * Int32(minibatch_size)
                keep_end = cutlass.max(segment_begin, last_minibatch_begin) + Int32(MLP_TILE_K)
                segment_end = cutlass.min(segment_end, cutlass.max(_round_up_i32(valid_end, MLP_TILE_K), keep_end))
            k_begin = segment_begin % Int32(macrobatch_size)
            k_tiles = (segment_end - segment_begin) // Int32(MLP_TILE_K)
            if k_tiles <= Int32(0):
                valid = Int32(0)
            if kind == Int32(ROUTED_WGRAD_DOWN):
                row, column = _swizzled_tile_i32(routed_h_tiles, i_tiles, local)
            else:
                row, column = _swizzled_tile_i32(i_tiles, routed_h_tiles, local)
        else:
            valid = Int32(0)
        return (valid, expert, row, column, k_begin, k_tiles, wgrad_first, wgrad_has_next, wgrad_empty, tile_valid, cached_expert, cached_begin, cached_end)

    @cute.jit
    def _decode_task_record(
        self,
        runtime: tuple,
        work_valid: Int32,
        logical_cluster: Int32,
        true_clusters: Int32,
        comm_clusters: Int32,
        completed_segment_offsets: cute.Tensor,
        completed_segment_experts: cute.Tensor,
        task_owner_buckets: cute.Tensor,
        shared_down_tasks: Int32,
        shared_gate_up_tasks: Int32,
        shared_tasks: Int32,
        routed_down_tasks: Int32,
        routed_gate_up_tasks: Int32,
        routed_tasks: Int32,
        num_tokens: Int32,
        tokens_per_expert: cute.Tensor,
        num_local_tokens: Int32,
        cached_expert: Int32,
        cached_begin: Int32,
        cached_end: Int32,
    ) -> tuple[
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
    ]:
        """Decode one CLC work item into the task-info record (TASK_INFO_FIELDS order), followed by
        the CLC warp's updated owner cache (expert, segment begin, segment end).

        ``active`` is 1 for a valid response naming a compute cluster (comm_clusters <= cluster <
        true_clusters); a valid response at or past true_clusters is the capacity tail
        (``needs_drain`` = 1: the warps stop and drain) and an invalid response stops the warps
        without a drain.  Only an active record is decoded (_decode_compute_task, then
        _gemm_geometry with the owner cache); a stop record has kind = generation = -1 and valid = 0,
        and no role runs a body for it.
        """
        active = Int32(0)
        needs_drain = Int32(0)
        kind = Int32(-1)
        generation = Int32(-1)
        valid = Int32(0)
        expert = Int32(0)
        row = Int32(0)
        column = Int32(0)
        k_begin = Int32(0)
        k_tiles = Int32(0)
        wgrad_first = Int32(0)
        wgrad_has_next = Int32(0)
        wgrad_empty = Int32(0)
        tile_valid = Int32(0)
        if work_valid != Int32(0):
            if logical_cluster >= Int32(comm_clusters):
                if logical_cluster < true_clusters:
                    active = Int32(1)
                else:
                    needs_drain = Int32(1)
        if active != Int32(0):
            kind, generation, local, expert_task = self._decode_compute_task(
                runtime,
                logical_cluster - Int32(comm_clusters),
                completed_segment_offsets,
                completed_segment_experts,
                task_owner_buckets,
                shared_down_tasks,
                shared_gate_up_tasks,
                shared_tasks,
                routed_down_tasks,
                routed_gate_up_tasks,
                routed_tasks,
            )
            (
                valid,
                expert,
                row,
                column,
                k_begin,
                k_tiles,
                wgrad_first,
                wgrad_has_next,
                wgrad_empty,
                tile_valid,
                cached_expert,
                cached_begin,
                cached_end,
            ) = self._gemm_geometry(
                runtime,
                kind,
                generation,
                local,
                expert_task,
                num_tokens,
                tokens_per_expert,
                tokens_per_expert,
                num_local_tokens,
                cached_expert,
                cached_begin,
                cached_end,
            )
        return (
            active,
            needs_drain,
            kind,
            generation,
            valid,
            expert,
            row,
            column,
            k_begin,
            k_tiles,
            wgrad_first,
            wgrad_has_next,
            wgrad_empty,
            tile_valid,
            cached_expert,
            cached_begin,
            cached_end,
        )

    @cute.jit
    def _wgrad_segment_index(
        self,
        runtime: tuple,
        generation: Int32,
        expert: Int32,
        expert_counts: cute.Tensor,
    ) -> Int32:
        """k for segment k > 0 of a split expert: the number of macrobatch boundaries
        between the expert's first row and this macrobatch (rare path only; the common
        path uses the division-free facts _gemm_geometry returns)."""
        macrobatch_size = runtime[_RT_MACROBATCH_SIZE]
        minibatches_per_macrobatch = runtime[_RT_MINIBATCHES_PER_MACROBATCH]
        expert_begin, _, _ = _expert_k_range_rt(runtime, expert_counts, expert, self.task_decode)
        macro_begin = (generation // Int32(minibatches_per_macrobatch)) * Int32(macrobatch_size)
        return (macro_begin - expert_begin + Int32(macrobatch_size) - Int32(1)) // Int32(macrobatch_size)

    @cute.jit
    def _reverse_dispatch_done_required(self, runtime: tuple, generation: Int32) -> Int32:
        """reverse_dispatch_done[generation] count meaning "every dX row of the generation has left
        the d_x ring": one arrival per pushed tile."""
        return self._comm_tiles(runtime, generation)

    @cute.jit
    def _comm_tiles(self, runtime: tuple, generation: Int32) -> Int32:
        """Communication tiles of one generation per direction (rows / tile rows x column blocks): the count of
        d_y_routed_ready, replayed_x_routed_ready and reverse_dispatch_done for it."""
        minibatch_size = runtime[_RT_MINIBATCH_SIZE]
        rows = cutlass.min(Int32(minibatch_size), runtime[_RT_NUM_TOKENS] - generation * Int32(minibatch_size))
        return _bwd_dynamic_task_stream.comm_blocks(rows, runtime[_RT_ROUTED_HIDDEN_SIZE], self.comm_tile_shape)

    @cute.jit
    def _run_comm_prefix(
        self,
        runtime: tuple,
        x_peer_table: cute.Tensor,
        d_y_peer_table: cute.Tensor,
        d_x_peer_table: cute.Tensor,
        router_peer_table: cute.Tensor,
        d_router_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens: Int32,
        router_weights: cute.Tensor,
        x_routed: cute.Tensor,
        d_y_routed: cute.Tensor,
        d_x_routed: cute.Tensor,
        d_router_partials: cute.Tensor,
        minibatch_expert_offsets: cute.Tensor,
        d_y_routed_ready: cute.Tensor,
        d_x_routed_ready: cute.Tensor,
        replayed_x_routed_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        wgrad_read_consumed: cute.Tensor,
        comm_mbarriers,
        arena,
        cluster_index: Int32,
        cta_rank: Int32,
        num_comm_sms: Int32,
    ) -> None:
        """Communication CTAs: the backward dynamic task stream prefix (_bwd_dynamic_task_stream)."""
        Bo = _bwd_dynamic_task_stream.StreamOperand
        operands = [None] * Bo.COUNT  # the stream hooks' operands, by position
        operands[Bo.X_PEER_TABLE] = x_peer_table
        operands[Bo.D_Y_PEER_TABLE] = d_y_peer_table
        operands[Bo.D_X_PEER_TABLE] = d_x_peer_table
        operands[Bo.ROUTER_PEER_TABLE] = router_peer_table
        operands[Bo.D_ROUTER_PEER_TABLE] = d_router_peer_table
        operands[Bo.X_ROUTED] = x_routed
        operands[Bo.D_Y_ROUTED] = d_y_routed
        operands[Bo.D_X_ROUTED] = d_x_routed
        operands[Bo.ROUTER_WEIGHTS] = router_weights
        operands[Bo.D_ROUTER_PARTIALS] = d_router_partials
        operands[Bo.MINIBATCH_EXPERT_OFFSETS] = minibatch_expert_offsets
        operands[Bo.D_Y_ROUTED_READY] = d_y_routed_ready
        operands[Bo.D_X_ROUTED_READY] = d_x_routed_ready
        operands[Bo.REPLAYED_X_ROUTED_READY] = replayed_x_routed_ready
        operands[Bo.ROUTED_BUFFERS_DONE] = routed_buffers_done
        operands[Bo.WGRAD_READ_CONSUMED] = wgrad_read_consumed
        operands[Bo.REVERSE_DISPATCH_DONE] = runtime[_RT_REVERSE_DISPATCH_DONE]
        operands[Bo.NUM_TOKENS] = num_tokens
        operands[Bo.MACROBATCH_SIZE] = runtime[_RT_MACROBATCH_SIZE]
        operands[Bo.MINIBATCH_SIZE] = runtime[_RT_MINIBATCH_SIZE]
        operands[Bo.MPM] = runtime[_RT_MINIBATCHES_PER_MACROBATCH]
        operands[Bo.HIDDEN_SIZE] = runtime[_RT_ROUTED_HIDDEN_SIZE]
        operands[Bo.TOPK] = runtime[_RT_TOPK]
        operands[Bo.S_TILES] = runtime[_RT_S_TILES]
        operands[Bo.SLOT_REUSE_REQUIRED] = runtime[_RT_ROUTED_BUFFERS_DONE_REQUIRED]
        operands[Bo.WGRAD_TILES] = runtime[_RT_ROUTED_WGRAD_TILES]
        operands[Bo.EP_RANKS] = runtime[_RT_EP_SIZE]
        _bwd_dynamic_task_stream.BackwardStream(self.has_tail, self.recompute_forward_context, self.comm_tile_shape).run(
            operands, schedule_peer_rank, schedule_peer_token_idx, arena, comm_mbarriers, cluster_index * Int32(CLUSTER_SIZE) + cta_rank, num_comm_sms
        )

    @cute.jit
    def _wait_gemm_inputs(
        self,
        runtime: tuple,
        kind: Int32,
        generation: Int32,
        row: Int32,
        d_y_routed_ready: cute.Tensor,
        d_gate_up_ready: cute.Tensor,
        replayed_x_routed_ready: cute.Tensor,
        num_local_tokens: Int32,
    ) -> None:
        minibatch_size = runtime[_RT_MINIBATCH_SIZE]
        minibatches_per_macrobatch = runtime[_RT_MINIBATCHES_PER_MACROBATCH]
        hidden_ready_required = runtime[_RT_HIDDEN_READY_REQUIRED]
        if kind == Int32(REPLAY_GATE_UP):
            _counter_wait_gpu(replayed_x_routed_ready, generation, self._comm_tiles(runtime, generation))
        elif kind == Int32(ROUTED_DOWN_DGRAD):
            _counter_wait_gpu(d_y_routed_ready, generation, self._comm_tiles(runtime, generation))
        elif kind == Int32(SHARED_GATE_UP_DGRAD):
            _counter_wait_gpu(d_gate_up_ready, row, hidden_ready_required)
        elif kind == Int32(ROUTED_GATE_UP_DGRAD):
            global_row = generation * Int32(minibatch_size // MLP_TILE_M) + row % Int32(minibatch_size // MLP_TILE_M)
            _counter_wait_gpu(
                d_gate_up_ready,
                Int32(num_local_tokens // MLP_TILE_M) + global_row,
                hidden_ready_required,
            )
            # This task's epilogue overwrites dX slot (g mod mpm): every comm CTA must have
            # pushed dX(g - mpm) out of it first (writer waits, FWD y_routed_done pattern).
            if generation >= Int32(minibatches_per_macrobatch):
                _counter_wait_gpu(
                    runtime[_RT_REVERSE_DISPATCH_DONE],
                    generation - Int32(minibatches_per_macrobatch),
                    self._reverse_dispatch_done_required(runtime, generation - Int32(minibatches_per_macrobatch)),
                )

    # Descriptor prefetch: a TMA descriptor is prefetched where it is used, once per task.  The kernel takes 38
    # descriptors and a task uses two to four of them (two or three loads, one store), so each loader prefetches its
    # load descriptors right before its K loop and each epilogue task its store descriptor: prefetch, then use.
    # A prefetch of every descriptor at the kernel entry would be far from most uses in a persistent kernel (the whole
    # task loop lies in between), and a CTA does not run every task kind.
    # The placement also decides the code of the K loop under CuTe DSL 4.8.  A TMA copy issued from a
    # kernel-parameter descriptor needs the parameter's generic address; 4.8 takes it from the prefetch right in
    # front of the loop, and otherwise forms it again in the block of every issue, where (under the copy's lane
    # election) it costs 7 instructions per load instead of 2: 15-19 instructions per K64 block of every load loop,
    # 2-7 % of the backward.  CuTe DSL 4.5.2 forms a descriptor's address once wherever it is prefetched; without any
    # prefetch it re-forms the address at every issue as well.  The prefetch is a hint and changes no data or
    # synchronization.
    @cute.jit
    def _load_one_abt_task(
        self,
        runtime: tuple,
        tiled_mma: cute.TiledMma,
        a_atom: cute.CopyAtom,
        a_tensor: cute.Tensor,
        b_atom: cute.CopyAtom,
        b_tensor: cute.Tensor,
        row: Int32,
        column: Int32,
        cta_rank: Int32,
        k_tiles: Int32,
        sA: cute.Tensor,
        sB: cute.Tensor,
        ab_pipeline,
        producer_state,
    ):
        cpasync.prefetch_descriptor(a_atom)  # once per task, see "Descriptor prefetch" above
        cpasync.prefetch_descriptor(b_atom)
        thr_mma = tiled_mma.get_slice(cta_rank)
        a_tile = cute.local_tile(
            a_tensor,
            cute.select(self.abt.mma_tiler, [0, 2]),
            (row, None),
        )
        b_tile = cute.local_tile(
            b_tensor,
            cute.select(self.abt.mma_tiler, [1, 2]),
            (column, None),
        )
        copy_a = _make_tma_block_load_fn(a_atom, thr_mma.partition_A(a_tile), sA)
        copy_b = _make_tma_block_load_fn(b_atom, thr_mma.partition_B(b_tile), sB)
        return self.abt.load_tma(ab_pipeline, producer_state, [copy_a, copy_b], k_tiles)

    @cute.jit
    def _load_swapab_operand(
        self,
        tiled_mma: cute.TiledMma,
        a_atom: cute.CopyAtom,
        a_tensor: cute.Tensor,
        b_atom: cute.CopyAtom,
        b_tensor: cute.Tensor,
        expert: Int32,
        row: Int32,
        column: Int32,
        tile_n: Int32,
        cta_rank: Int32,
        k_tiles: Int32,
        sA_slab0: cute.Tensor,
        sA_slab1: cute.Tensor,
        sB: cute.Tensor,
        ab_pipeline,
        producer_state,
        weights: cute.Tensor = None,
        a_desc_ptr=None,
        k_offset=None,
    ):
        """One operand pair of a routed swap-AB dgrad tile (k_tiles stages).  A = this CTA's 128 output
        features (column block ``column``, half ``cta_rank``) of the expert's weights as two 64-feature
        slabs per stage; B = the block's token rows (the Gate/Up dX's dGate / dUp, the Down's dY) with the
        instruction's N ``tile_n``.  cta_group::2 splits N at N / 2, so the non-leader CTA's B half
        starts at token N / 2 instead of 128 (the 128-row box stays inside the 256-row block; rows past N
        are loaded but not read by the MMA)."""
        # Once per task, see "Descriptor prefetch" above.  A split weight is read through its expert's own
        # descriptor (a_desc_ptr), not through the atom's.
        if cutlass.const_expr(a_desc_ptr is None):
            cpasync.prefetch_descriptor(a_atom)
        cpasync.prefetch_descriptor(b_atom)
        b_shift = Int32(0)
        if cta_rank != Int32(0):
            b_shift = _swapab_fwd.nonleader_b_shift(tile_n)
        thr_mma = tiled_mma.get_slice(cta_rank)
        feature_slab = (column * Int32(CLUSTER_SIZE) + cta_rank) * Int32(2)
        # A combined FC1's halves differ along I, and this operand is the weight MN-major -- rows
        # are the output feature H, so I is the *contraction* axis here.
        if cutlass.const_expr(k_offset is not None):
            weights = cute.domain_offset((Int32(0), k_offset), weights)
        a_tile0 = cute.local_tile(weights, (_SWAP_SLAB_ROWS, MLP_TILE_K), (feature_slab, None))
        a_tile1 = cute.local_tile(weights, (_SWAP_SLAB_ROWS, MLP_TILE_K), (feature_slab + Int32(1), None))
        b_tile = cute.local_tile(
            cute.domain_offset((b_shift, Int32(0)), b_tensor),
            cute.select(self.replay.mma_tiler, [1, 2]),
            (row, None),
        )
        copy_a0 = _make_tma_block_load_fn(a_atom, a_tile0, sA_slab0, tma_desc_ptr=a_desc_ptr)
        copy_a1 = _make_tma_block_load_fn(a_atom, a_tile1, sA_slab1, tma_desc_ptr=a_desc_ptr)
        copy_b = _make_tma_block_load_fn(b_atom, thr_mma.partition_B(b_tile), sB)
        return self.abt.load_tma(ab_pipeline, producer_state, [copy_a0, copy_a1, copy_b], k_tiles)

    @cute.jit
    def _main_grad_descriptor(self, descriptors: cute.Pointer, num_experts: Int32, operand: cutlass.Constexpr[int], expert: Int32):
        """One expert's main_grad descriptor, or None when the buffers are grouped.

        The write-back counterpart of ``_routed_operand``: split main_grad buffers are
        independent per-expert allocations, so the epilogue names one instead of slicing an L
        mode.  Rows 5..7 of the address table are the Down, Gate and Up buffers.
        """
        if cutlass.const_expr(not self.split_main_grads):
            return None
        return routed_descriptors_pkg.descriptor_argument(routed_descriptors_pkg.operand_descriptors(descriptors, num_experts, operand), expert)

    @cute.jit
    def _routed_operand(self, tensor: cute.Tensor, descriptors: cute.Pointer, num_experts: Int32, operand: cutlass.Constexpr[int], expert: Int32):
        """One routed weight for one expert: the view to tile, and the descriptor to read it with.

        Grouped: the L mode is sliced and the atom's own descriptor is used.  Split: the operand
        *is* one expert's allocation, so there is no L mode and the expert is named by its
        descriptor.  ``operand`` indexes the address table's rows.
        """
        if cutlass.const_expr(self.split_weights):
            return tensor, routed_descriptors_pkg.descriptor_argument(routed_descriptors_pkg.operand_descriptors(descriptors, num_experts, operand), expert)
        return tensor[None, None, expert], None

    @cute.jit
    def _load_paired_replay_gate_up(
        self,
        runtime: tuple,
        tiled_mma: cute.TiledMma,
        a_atom: cute.CopyAtom,
        a_tensor: cute.Tensor,
        gate_b_atom: cute.CopyAtom,
        gate_b_tensor: cute.Tensor,
        up_b_atom: cute.CopyAtom,
        up_b_tensor: cute.Tensor,
        row: Int32,
        column: Int32,
        cta_rank: Int32,
        k_tiles: Int32,
        sA: cute.Tensor,
        sB: cute.Tensor,
        paired_pipeline,
        paired_producer_state,
        gate_b_desc_ptr=None,
        up_b_desc_ptr=None,
        up_row_offset=None,
    ):
        """Load X once and Gate/Up independently into a compact AB3 ring.

        ``up_row_offset``: a combined FC1's Up half starts that many rows into the parameter.  It
        shifts the coordinates (as the forward's and the Gate/Up dgrad's loads do), not the tile
        index: with I % 256 != 0 the Up half starts mid-way through a replay N tile, and a tile
        offset (I // 256) would read Up from the wrong rows."""

        # Once per task, see "Descriptor prefetch" above.  Split weights are read through their expert's own
        # descriptors (gate_b_desc_ptr / up_b_desc_ptr), not through the atoms'.
        cpasync.prefetch_descriptor(a_atom)
        if cutlass.const_expr(gate_b_desc_ptr is None):
            cpasync.prefetch_descriptor(gate_b_atom)
        if cutlass.const_expr(up_b_desc_ptr is None):
            cpasync.prefetch_descriptor(up_b_atom)
        thr_mma = tiled_mma.get_slice(cta_rank)
        a_tile = cute.local_tile(
            a_tensor,
            cute.select(self.replay.mma_tiler, [0, 2]),
            (row, None),
        )
        gate_b_tile = cute.local_tile(
            gate_b_tensor,
            cute.select(self.replay.mma_tiler, [1, 2]),
            (column, None),
        )
        if cutlass.const_expr(up_row_offset is not None):
            up_b_tensor = cute.domain_offset((up_row_offset, Int32(0)), up_b_tensor)
        up_b_tile = cute.local_tile(
            up_b_tensor,
            cute.select(self.replay.mma_tiler, [1, 2]),
            (column, None),
        )
        copy_a = _make_tma_block_load_fn(a_atom, thr_mma.partition_A(a_tile), sA)
        copy_gate_b = _make_tma_block_load_fn(gate_b_atom, thr_mma.partition_B(gate_b_tile), sB, tma_desc_ptr=gate_b_desc_ptr)
        copy_up_b = _make_tma_block_load_fn(up_b_atom, thr_mma.partition_B(up_b_tile), sB, tma_desc_ptr=up_b_desc_ptr)
        for k_tile in cutlass.range(k_tiles, unroll=1):
            paired_pipeline.producer_acquire(paired_producer_state)
            stage = paired_producer_state.index
            tma_barrier = paired_pipeline.producer_get_barrier(paired_producer_state)
            copy_a(k_tile, stage, tma_bar_ptr=tma_barrier)
            copy_gate_b(k_tile, stage, tma_bar_ptr=tma_barrier)
            copy_up_b(
                k_tile,
                stage + Int32(PAIRED_GATE_UP_STAGES),
                tma_bar_ptr=tma_barrier,
            )
            paired_pipeline.producer_commit(paired_producer_state)
            paired_producer_state.advance()
        return paired_producer_state

    @cute.jit
    def _load_one_atb_task(
        self,
        runtime: tuple,
        tiled_mma: cute.TiledMma,
        a_atom: cute.CopyAtom,
        a_tensor: cute.Tensor,
        b_atom: cute.CopyAtom,
        b_tensor: cute.Tensor,
        row: Int32,
        column: Int32,
        k_begin: Int32,
        cta_rank: Int32,
        k_tiles: Int32,
        sA: cute.Tensor,
        sB: cute.Tensor,
        ab_pipeline,
        producer_state,
        kind: Int32,
        generation: Int32,
        d_y_routed_ready: cute.Tensor,
        d_gate_up_ready: cute.Tensor,
        replayed_x_routed_ready: cute.Tensor,
        num_local_tokens: Int32,
    ):
        macrobatch_size = runtime[_RT_MACROBATCH_SIZE]
        minibatch_size = runtime[_RT_MINIBATCH_SIZE]
        minibatches_per_macrobatch = runtime[_RT_MINIBATCHES_PER_MACROBATCH]
        hidden_ready_required = runtime[_RT_HIDDEN_READY_REQUIRED]
        thr_mma = tiled_mma.get_slice(cta_rank)
        a_base = cute.domain_offset((row * Int32(MLP_TILE_M), k_begin), a_tensor)
        b_base = cute.domain_offset((column * Int32(MLP_TILE_N), k_begin), b_tensor)
        a_tile = cute.local_tile(
            a_base,
            cute.select(self.atb.mma_tiler, [0, 2]),
            (Int32(0), None),
        )
        b_tile = cute.local_tile(
            b_base,
            cute.select(self.atb.mma_tiler, [1, 2]),
            (Int32(0), None),
        )
        copy_a = _make_tma_block_load_fn(a_atom, thr_mma.partition_A(a_tile), sA)
        copy_b = _make_tma_block_load_fn(b_atom, thr_mma.partition_B(b_tile), sB)
        # Wait only when a K64 load first touches a producing M256 row block or a minibatch.
        # Compute the task's global K origin once and track the next boundary tile with
        # adds/compares; the TMA producer loop must not carry per-tile runtime divisions.
        global_k = k_begin
        if kind >= Int32(ROUTED_WGRAD_DOWN):
            global_k = global_k + (generation // Int32(minibatches_per_macrobatch)) * Int32(macrobatch_size)
        row_block_tiles = Int32(MLP_TILE_M // MLP_TILE_K)
        minibatch_tiles = Int32(minibatch_size) // Int32(MLP_TILE_K)
        origin_tile = global_k // Int32(MLP_TILE_K)
        next_row_block_tile = row_block_tiles - origin_tile % row_block_tiles
        next_minibatch_tile = minibatch_tiles - origin_tile % minibatch_tiles
        for k_tile in cutlass.range(k_tiles, unroll=1):
            # Peek unguarded: under a 'has next' guard the guard predicate is read last by the try_wait, and ptxas then
            # places the loop-exit compare behind that try_wait, stalling every K64 block.
            peek_empty_status = ab_pipeline.producer_try_acquire(producer_state)
            wait_row_block = Int32(0)
            wait_minibatch = Int32(0)
            if k_tile == Int32(0):
                wait_row_block = Int32(1)
                wait_minibatch = Int32(1)
            if k_tile == next_row_block_tile:
                wait_row_block = Int32(1)
                next_row_block_tile = next_row_block_tile + row_block_tiles
            if k_tile == next_minibatch_tile:
                wait_minibatch = Int32(1)
                next_minibatch_tile = next_minibatch_tile + minibatch_tiles

            if wait_row_block != Int32(0):
                if kind == Int32(SHARED_WGRAD_GATE) or kind == Int32(SHARED_WGRAD_UP):
                    with cute.arch.elect_one():
                        _counter_wait_gpu(
                            d_gate_up_ready,
                            global_k // Int32(MLP_TILE_M),
                            hidden_ready_required,
                        )
                    cute.arch.sync_warp()
                elif kind >= Int32(ROUTED_WGRAD_DOWN):
                    # Routed Wgrad-down/gate/up all wait for the row block's Down dgrad epilogues
                    # (fused SwiGLU-backward), which store d_gate, d_up and w*Hidden together
                    # (every macrobatch).
                    with cute.arch.elect_one():
                        _counter_wait_gpu(
                            d_gate_up_ready,
                            Int32(num_local_tokens // MLP_TILE_M) + global_k // Int32(MLP_TILE_M),
                            hidden_ready_required,
                        )
                    cute.arch.sync_warp()

            if wait_minibatch != Int32(0):
                minibatch = global_k // Int32(minibatch_size)
                required = self._comm_tiles(runtime, minibatch)
                if kind == Int32(ROUTED_WGRAD_DOWN):
                    with cute.arch.elect_one():
                        _counter_wait_gpu(d_y_routed_ready, minibatch, required)
                    cute.arch.sync_warp()
                elif kind == Int32(ROUTED_WGRAD_GATE) or kind == Int32(ROUTED_WGRAD_UP):
                    if minibatch >= Int32(self._saved_minibatches(minibatches_per_macrobatch)):
                        with cute.arch.elect_one():
                            _counter_wait_gpu(
                                replayed_x_routed_ready,
                                minibatch,
                                required,
                            )
                        cute.arch.sync_warp()

            if not peek_empty_status:
                _mbar_spin_wait(ab_pipeline.sync_object_empty.get_barrier(producer_state.index), producer_state.phase)
            ab_pipeline.producer_acquire(producer_state, Boolean(True))
            tma_barrier = ab_pipeline.producer_get_barrier(producer_state)
            stage = producer_state.index
            # The two descriptors are prefetched here, in the block of their issues, not once in front of the K loop
            # (see "Descriptor prefetch" above).  This loop is the load loop without slack.  Under CuTe DSL 4.8 a
            # prefetch in front of the loop leaves the descriptor base in a general register across the loop and
            # every issue moves it lane-wise; next to the issues, 4.8 forms the addresses in 3.9 instead of 6.0
            # instructions per issue (9 fewer per K64 step).  CuTe DSL 4.5.2 emits the same binary for both
            # placements.  The prefetch is a hint: no data, no synchronization.
            cpasync.prefetch_descriptor(a_atom)
            cpasync.prefetch_descriptor(b_atom)
            copy_a(k_tile, stage, tma_bar_ptr=tma_barrier)
            copy_b(k_tile, stage, tma_bar_ptr=tma_barrier)
            ab_pipeline.producer_commit(producer_state)
            producer_state.advance()
            global_k = global_k + Int32(MLP_TILE_K)
        return producer_state

    @cute.jit
    def _run_tma_gemm_task(
        self,
        runtime: tuple,
        geometry: tuple,
        kind: Int32,
        generation: Int32,
        cta_rank: Int32,
        tiled_mma_abt: cute.TiledMma,
        tiled_mma_atb: cute.TiledMma,
        tiled_mma_replay: cute.TiledMma,
        abt_a_atoms,
        abt_a_tensors,
        abt_b_atoms,
        abt_b_tensors,
        swap_a_atoms,
        swap_a_tensors,
        swap_b_atoms,
        swap_b_tensors,
        atb_a_atoms,
        atb_a_tensors,
        atb_b_atoms,
        atb_b_tensors,
        sA_abt: cute.Tensor,
        sB_abt: cute.Tensor,
        sA_swap0: cute.Tensor,
        sA_swap1: cute.Tensor,
        sB_swap: cute.Tensor,
        sA_atb: cute.Tensor,
        sB_atb: cute.Tensor,
        ab_pipeline,
        producer_state,
        d_y_routed_ready: cute.Tensor,
        d_gate_up_ready: cute.Tensor,
        replayed_x_routed_ready: cute.Tensor,
        num_local_tokens: Int32,
        routed_descriptors,
        num_experts_rt: Int32,
        up_row_offset,
    ):
        valid, expert, row, column, k_begin, k_tiles, _, _, _, tile_valid = geometry
        if valid != Int32(0):
            # Wgrad readiness is intentionally incremental in the K64 load
            # loop below; only non-Wgrad GEMMs wait once before their segment.
            if _is_wgrad_kind(kind) == Int32(0):
                with cute.arch.elect_one():
                    self._wait_gemm_inputs(
                        runtime,
                        kind,
                        generation,
                        row,
                        d_y_routed_ready,
                        d_gate_up_ready,
                        replayed_x_routed_ready,
                        num_local_tokens,
                    )
                cute.arch.sync_warp()
            if kind == Int32(SHARED_DOWN_DGRAD):
                producer_state = self._load_one_abt_task(
                    runtime,
                    tiled_mma_abt,
                    abt_a_atoms[0],
                    abt_a_tensors[0],
                    abt_b_atoms[0],
                    abt_b_tensors[0],
                    row,
                    column,
                    cta_rank,
                    k_tiles,
                    sA_abt,
                    sB_abt,
                    ab_pipeline,
                    producer_state,
                )
            elif kind == Int32(SHARED_GATE_UP_DGRAD):
                producer_state = self._load_one_abt_task(
                    runtime,
                    tiled_mma_abt,
                    abt_a_atoms[1],
                    abt_a_tensors[1],
                    abt_b_atoms[1],
                    abt_b_tensors[1],
                    row,
                    column,
                    cta_rank,
                    k_tiles,
                    sA_abt,
                    sB_abt,
                    ab_pipeline,
                    producer_state,
                )
                producer_state = self._load_one_abt_task(
                    runtime,
                    tiled_mma_abt,
                    abt_a_atoms[2],
                    abt_a_tensors[2],
                    abt_b_atoms[2],
                    abt_b_tensors[2],
                    row,
                    column,
                    cta_rank,
                    k_tiles,
                    sA_abt,
                    sB_abt,
                    ab_pipeline,
                    producer_state,
                )
            elif kind == Int32(ROUTED_DOWN_DGRAD):
                # Swap-AB: Wd's 128 I features of this CTA, the block's dY rows, N = _routed_down_n.
                down_weights, down_desc = self._routed_operand(swap_a_tensors[2], routed_descriptors, num_experts_rt, 2, expert)
                producer_state = self._load_swapab_operand(
                    tiled_mma_replay,
                    swap_a_atoms[2],
                    swap_a_tensors[2],
                    swap_b_atoms[2],
                    swap_b_tensors[2],
                    expert,
                    row,
                    column,
                    _routed_down_n(tile_valid),
                    cta_rank,
                    k_tiles,
                    sA_swap0,
                    sA_swap1,
                    sB_swap,
                    ab_pipeline,
                    producer_state,
                    weights=down_weights,
                    a_desc_ptr=down_desc,
                )
            elif kind == Int32(ROUTED_GATE_UP_DGRAD):
                # Swap-AB: the Gate K tiles, then the Up K tiles (2 x k_tiles stages into one
                # accumulation), N = align16(tile_valid).
                for operand in cutlass.range_constexpr(2):
                    fc1_weights, fc1_desc = self._routed_operand(swap_a_tensors[operand], routed_descriptors, num_experts_rt, operand, expert)
                    # A combined FC1's Up half starts up_row_offset rows into the parameter, which is
                    # this operand's K axis (see _load_swapab_operand).  Grouped weights were given
                    # Up's own view at the launch; split ones take the offset as a coordinate.
                    fc1_k_offset = None
                    if cutlass.const_expr(operand == 1 and self.split_weights):
                        fc1_k_offset = up_row_offset
                    producer_state = self._load_swapab_operand(
                        tiled_mma_replay,
                        swap_a_atoms[operand],
                        swap_a_tensors[operand],
                        swap_b_atoms[operand],
                        swap_b_tensors[operand],
                        expert,
                        row,
                        column,
                        _swapab_fwd.mma_n(tile_valid),
                        cta_rank,
                        k_tiles,
                        sA_swap0,
                        sA_swap1,
                        sB_swap,
                        ab_pipeline,
                        producer_state,
                        weights=fc1_weights,
                        a_desc_ptr=fc1_desc,
                        k_offset=fc1_k_offset,
                    )
            elif kind == Int32(SHARED_WGRAD_DOWN):
                producer_state = self._load_one_atb_task(
                    runtime,
                    tiled_mma_atb,
                    atb_a_atoms[0],
                    atb_a_tensors[0],
                    atb_b_atoms[0],
                    atb_b_tensors[0],
                    row,
                    column,
                    k_begin,
                    cta_rank,
                    k_tiles,
                    sA_atb,
                    sB_atb,
                    ab_pipeline,
                    producer_state,
                    kind,
                    generation,
                    d_y_routed_ready,
                    d_gate_up_ready,
                    replayed_x_routed_ready,
                    num_local_tokens,
                )
            elif kind == Int32(SHARED_WGRAD_GATE):
                producer_state = self._load_one_atb_task(
                    runtime,
                    tiled_mma_atb,
                    atb_a_atoms[1],
                    atb_a_tensors[1],
                    atb_b_atoms[1],
                    atb_b_tensors[1],
                    row,
                    column,
                    k_begin,
                    cta_rank,
                    k_tiles,
                    sA_atb,
                    sB_atb,
                    ab_pipeline,
                    producer_state,
                    kind,
                    generation,
                    d_y_routed_ready,
                    d_gate_up_ready,
                    replayed_x_routed_ready,
                    num_local_tokens,
                )
            elif kind == Int32(SHARED_WGRAD_UP):
                producer_state = self._load_one_atb_task(
                    runtime,
                    tiled_mma_atb,
                    atb_a_atoms[2],
                    atb_a_tensors[2],
                    atb_b_atoms[1],
                    atb_b_tensors[1],
                    row,
                    column,
                    k_begin,
                    cta_rank,
                    k_tiles,
                    sA_atb,
                    sB_atb,
                    ab_pipeline,
                    producer_state,
                    kind,
                    generation,
                    d_y_routed_ready,
                    d_gate_up_ready,
                    replayed_x_routed_ready,
                    num_local_tokens,
                )
            elif kind == Int32(ROUTED_WGRAD_DOWN):
                producer_state = self._load_one_atb_task(
                    runtime,
                    tiled_mma_atb,
                    atb_a_atoms[3],
                    atb_a_tensors[3],
                    atb_b_atoms[2],
                    atb_b_tensors[2],
                    row,
                    column,
                    k_begin,
                    cta_rank,
                    k_tiles,
                    sA_atb,
                    sB_atb,
                    ab_pipeline,
                    producer_state,
                    kind,
                    generation,
                    d_y_routed_ready,
                    d_gate_up_ready,
                    replayed_x_routed_ready,
                    num_local_tokens,
                )
            elif kind == Int32(ROUTED_WGRAD_GATE):
                producer_state = self._load_one_atb_task(
                    runtime,
                    tiled_mma_atb,
                    atb_a_atoms[4],
                    atb_a_tensors[4],
                    atb_b_atoms[3],
                    atb_b_tensors[3],
                    row,
                    column,
                    k_begin,
                    cta_rank,
                    k_tiles,
                    sA_atb,
                    sB_atb,
                    ab_pipeline,
                    producer_state,
                    kind,
                    generation,
                    d_y_routed_ready,
                    d_gate_up_ready,
                    replayed_x_routed_ready,
                    num_local_tokens,
                )
            elif kind == Int32(ROUTED_WGRAD_UP):
                producer_state = self._load_one_atb_task(
                    runtime,
                    tiled_mma_atb,
                    atb_a_atoms[5],
                    atb_a_tensors[5],
                    atb_b_atoms[3],
                    atb_b_tensors[3],
                    row,
                    column,
                    k_begin,
                    cta_rank,
                    k_tiles,
                    sA_atb,
                    sB_atb,
                    ab_pipeline,
                    producer_state,
                    kind,
                    generation,
                    d_y_routed_ready,
                    d_gate_up_ready,
                    replayed_x_routed_ready,
                    num_local_tokens,
                )
        return producer_state

    @cute.jit
    def _mma_paired_replay_gate_up(
        self,
        runtime: tuple,
        tiled_mma: cute.TiledMma,
        cta_rank: Int32,
        k_tiles: Int32,
        sA: cute.Tensor,
        sB: cute.Tensor,
        acc_tmem_ptr,
        paired_ab_pipeline,
        paired_acc_pipeline,
        paired_ab_consumer_state,
        paired_acc_producer_state,
    ):
        """Consume X twice and commit Gate+Up as one accumulator epoch."""

        is_leader = cta_rank == Int32(0)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.replay.mma_tiler[:2])
        fake = tiled_mma.make_fragment_C(cute.append(acc_shape, MLP_ACC_STAGES))
        base = cute.make_tensor(acc_tmem_ptr, fake.layout)

        if is_leader:
            paired_acc_pipeline.producer_acquire(paired_acc_producer_state)
        gate_accumulator = base[None, None, None, Int32(0)]
        up_accumulator = base[None, None, None, Int32(1)]

        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range(k_tiles, unroll=1):
            if is_leader:
                _mbar_spin_wait(paired_ab_pipeline.sync_object_full.get_barrier(paired_ab_consumer_state.index), paired_ab_consumer_state.phase)
                paired_ab_pipeline.consumer_wait(paired_ab_consumer_state, Boolean(True))
                for k_block in cutlass.range(num_k_blocks, unroll_full=True):
                    coordinate = (
                        None,
                        None,
                        k_block,
                        paired_ab_consumer_state.index,
                    )
                    up_coordinate = (
                        None,
                        None,
                        k_block,
                        paired_ab_consumer_state.index + Int32(PAIRED_GATE_UP_STAGES),
                    )
                    if k_tile == Int32(0):
                        if k_block == Int32(0):
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        else:
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    else:
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(
                        tiled_mma,
                        gate_accumulator,
                        tCrA[coordinate],
                        tCrB[coordinate],
                        gate_accumulator,
                    )
                    if k_tile == Int32(0):
                        if k_block == Int32(0):
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        else:
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    else:
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(
                        tiled_mma,
                        up_accumulator,
                        tCrA[coordinate],
                        tCrB[up_coordinate],
                        up_accumulator,
                    )
                paired_ab_pipeline.consumer_release(paired_ab_consumer_state)
            paired_ab_consumer_state.advance()

        if is_leader:
            paired_acc_pipeline.producer_commit(paired_acc_producer_state)
        paired_acc_producer_state.advance()
        return paired_ab_consumer_state, paired_acc_producer_state, tiled_mma

    @cute.jit
    def _mma_one_atb_task(
        self,
        runtime: tuple,
        tiled_mma: cute.TiledMma,
        kind: Int32,
        generation: Int32,
        k_begin: Int32,
        k_tiles: Int32,
        cta_rank: Int32,
        num_tokens: Int32,
        sA: cute.Tensor,
        sB: cute.Tensor,
        acc_tmem_ptr,
        ab_pipeline,
        acc_pipeline,
        ab_consumer_state,
        acc_producer_state,
        wgrad_read_consumed: cute.Tensor,
    ):
        macrobatch_size = runtime[_RT_MACROBATCH_SIZE]
        minibatch_size = runtime[_RT_MINIBATCH_SIZE]
        minibatches_per_macrobatch = runtime[_RT_MINIBATCHES_PER_MACROBATCH]
        """Consume AtB K64 stages into the accumulator pipeline."""

        is_leader = cta_rank == Int32(0)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.atb.mma_tiler[:2])
        fake = tiled_mma.make_fragment_C(cute.append(acc_shape, MLP_ACC_STAGES))
        base = cute.make_tensor(acc_tmem_ptr, fake.layout)
        accumulator = base[None, None, None, acc_producer_state.index]

        peek_full_status = Boolean(True)
        if Int32(0) < k_tiles and is_leader:
            peek_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        if is_leader:
            acc_pipeline.producer_acquire(acc_producer_state)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])
        # wgrad_read_consumed: one arrive per minibatch intersected by this task's K range whose
        # ring slot has a later generation, issued once the last K64 block of that minibatch has
        # been waited on.  The boundary tiles are computed once here instead of dividing on every
        # K tile.
        retire_first = Int32(0)
        retire_last = Int32(-1)
        retire_limit = Int32(0)
        retire_next_tile = Int32(0x7FFFFFFF)
        retire_step = Int32(minibatch_size) // Int32(MLP_TILE_K)
        if kind >= Int32(ROUTED_WGRAD_DOWN):
            global_k_begin = (generation // Int32(minibatches_per_macrobatch)) * Int32(macrobatch_size) + k_begin
            global_k_end = global_k_begin + k_tiles * Int32(MLP_TILE_K)
            retire_first = global_k_begin // Int32(minibatch_size)
            retire_last = (global_k_end - Int32(1)) // Int32(minibatch_size)
            retire_limit = _ceil_div_i32(num_tokens, Int32(minibatch_size)) - Int32(minibatches_per_macrobatch)
            retire_next_tile = ((retire_first + Int32(1)) * Int32(minibatch_size) - global_k_begin) // Int32(MLP_TILE_K)
        retire_minibatch = retire_first
        for k_tile in cutlass.range(k_tiles, unroll=1):
            if is_leader:
                if not peek_full_status:
                    _mbar_spin_wait(ab_pipeline.sync_object_full.get_barrier(ab_consumer_state.index), ab_consumer_state.phase)
                ab_pipeline.consumer_wait(ab_consumer_state, Boolean(True))
                if cta_rank == Int32(0):
                    if k_tile + Int32(1) == retire_next_tile or k_tile + Int32(1) == k_tiles:
                        if retire_minibatch <= retire_last:
                            if retire_minibatch < retire_limit:
                                with cute.arch.elect_one():
                                    _counter_arrive_gpu(wgrad_read_consumed, retire_minibatch)
                            retire_minibatch = retire_minibatch + Int32(1)
                            retire_next_tile = retire_next_tile + retire_step
                for k_block in cutlass.range(num_k_blocks, unroll_full=True):
                    coordinate = (
                        None,
                        None,
                        k_block,
                        ab_consumer_state.index,
                    )
                    cute.gemm(
                        tiled_mma,
                        accumulator,
                        tCrA[coordinate],
                        tCrB[coordinate],
                        accumulator,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()
            peek_full_status = Boolean(True)
            if k_tile + Int32(1) < k_tiles and is_leader:
                peek_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        if is_leader:
            acc_pipeline.producer_commit(acc_producer_state)
        acc_producer_state.advance()
        return ab_consumer_state, acc_producer_state, tiled_mma

    @cute.jit
    def _run_umma_gemm_task(
        self,
        runtime: tuple,
        geometry: tuple,
        kind: Int32,
        generation: Int32,
        cta_rank: Int32,
        num_tokens: Int32,
        sA_abt: cute.Tensor,
        sB_abt: cute.Tensor,
        sA_atb: cute.Tensor,
        sB_atb: cute.Tensor,
        a_ring_base: Int32,
        b_ring_base: Int32,
        acc_tmem_ptr,
        ab_pipeline,
        acc_pipeline,
        ab_consumer_state,
        acc_producer_state,
        mma_op_abt,
        mma_op_atb,
        wgrad_read_consumed: cute.Tensor,
    ):
        valid, _, _, _, k_begin, k_tiles, _, _, _, tile_valid = geometry
        if valid != Int32(0):
            is_leader = cta_rank == Int32(0)
            is_abt = Int32(0)
            is_swap = Int32(0)
            if kind == Int32(SHARED_DOWN_DGRAD):
                is_abt = Int32(1)
            elif kind == Int32(SHARED_GATE_UP_DGRAD):
                is_abt = Int32(1)
            elif kind == Int32(ROUTED_DOWN_DGRAD):
                is_swap = Int32(1)
            elif kind == Int32(ROUTED_GATE_UP_DGRAD):
                is_swap = Int32(1)
            if is_swap != Int32(0):
                # Swap-AB: raw tcgen05 with a runtime N from the block's valid rows and the MN-major A
                # (_swapab_fwd).  Down dgrad: k_tiles with N rounded to the Wgrad K block; Gate/Up dgrad:
                # the Gate K tiles then the Up K tiles into one accumulator stage with N = align16.
                total_k = k_tiles
                tile_n = _routed_down_n(tile_valid)
                if kind == Int32(ROUTED_GATE_UP_DGRAD):
                    total_k = Int32(2) * k_tiles
                    tile_n = _swapab_fwd.mma_n(tile_valid)
                acc_shape = mma_op_abt.partition_shape_C(self.abt.mma_tiler[:2])
                fake = mma_op_abt.make_fragment_C(cute.append(acc_shape, MLP_ACC_STAGES))
                base = cute.make_tensor(acc_tmem_ptr, fake.layout)
                accumulator = base[None, None, None, acc_producer_state.index]
                ab_consumer_state, acc_producer_state = _swapab_fwd.mma_dynamic_n(
                    ab_pipeline,
                    acc_pipeline,
                    ab_consumer_state,
                    acc_producer_state,
                    a_ring_base,
                    b_ring_base,
                    GEMM_A_BYTES // MLP_LOAD_STAGES,
                    Int32(accumulator.iterator.toint()),
                    total_k,
                    tile_n,
                    is_leader,
                    a_mn_major=True,
                )
            elif is_abt != Int32(0):
                tiled_mma = mma_op_abt
                tCrA = tiled_mma.make_fragment_A(sA_abt)
                tCrB = tiled_mma.make_fragment_B(sB_abt)
                acc_shape = tiled_mma.partition_shape_C(self.abt.mma_tiler[:2])
                fake = tiled_mma.make_fragment_C(cute.append(acc_shape, MLP_ACC_STAGES))
                base = cute.make_tensor(acc_tmem_ptr, fake.layout)
                accumulator = base[None, None, None, acc_producer_state.index]
                total_k = k_tiles
                if kind == Int32(SHARED_GATE_UP_DGRAD):
                    total_k = Int32(2) * k_tiles
                (
                    ab_consumer_state,
                    acc_producer_state,
                    mma_op_abt,
                ) = self.abt.mma(
                    ab_pipeline,
                    acc_pipeline,
                    ab_consumer_state,
                    acc_producer_state,
                    mma_op_abt,
                    tCrA,
                    tCrB,
                    accumulator,
                    total_k,
                    is_leader,
                    cta_rank,
                )
            else:
                (
                    ab_consumer_state,
                    acc_producer_state,
                    mma_op_atb,
                ) = self._mma_one_atb_task(
                    runtime,
                    mma_op_atb,
                    kind,
                    generation,
                    k_begin,
                    k_tiles,
                    cta_rank,
                    num_tokens,
                    sA_atb,
                    sB_atb,
                    acc_tmem_ptr,
                    ab_pipeline,
                    acc_pipeline,
                    ab_consumer_state,
                    acc_producer_state,
                    wgrad_read_consumed,
                )
        return ab_consumer_state, acc_producer_state, mma_op_abt, mma_op_atb

    @cute.jit
    def _wgrad_counter_index(self, runtime: tuple, kind: Int32, expert: Int32, row: Int32, column: Int32) -> Int32:
        """wgrad_segment_done slot of one routed dW tile: (matrix, expert, tile)."""
        tile_columns = runtime[_RT_ROUTED_H_TILES]  # gate/up tiles: (i_tiles rows) x (routed_h_tiles columns)
        if kind == Int32(ROUTED_WGRAD_DOWN):
            tile_columns = runtime[_RT_I_TILES]  # down tiles: (routed_h_tiles rows) x (i_tiles columns)
        return ((kind - Int32(ROUTED_WGRAD_DOWN)) * runtime[_RT_NUM_EXPERTS] + expert) * runtime[_RT_ROUTED_WGRAD_TILES] + row * tile_columns + column

    @cute.jit
    def _run_one_epilogue_task(
        self,
        runtime: tuple,
        collective,
        tiled_mma: cute.TiledMma,
        epi_tile: cute.Tile,
        thread: Int32,
        warp: Int32,
        cta_rank: Int32,
        kind: Int32,
        generation: Int32,
        row: Int32,
        column: Int32,
        expert: Int32,
        d_atom: cute.CopyAtom,
        d_tensor: cute.Tensor,
        sD: cute.Tensor,
        acc_tmem_ptr,
        acc_pipeline,
        acc_consumer_state,
        routed_wgrad: cutlass.Constexpr,
        store_add: cutlass.Constexpr,
        replayed_gate_up_ready: cute.Tensor,
        wgrad_segment_done: cute.Tensor = None,
        segment_index: Int32 = Int32(0),
        publish_segment: Int32 = Int32(0),
        ordered_wait: cutlass.Constexpr = False,
        acc_slot: cutlass.Constexpr = -1,
        wait_for_acc: cutlass.Constexpr = True,
        acc_release_idx: cutlass.Constexpr = MLP_EPILOGUE_PIECES - 1,
        manual_acc_release: cutlass.Constexpr = False,
        advance_acc_state: cutlass.Constexpr = True,
        replay_ready_increment: cutlass.Constexpr = 1,
        d_desc_ptr=None,
        d_row_tile_offset=None,
    ):
        # The task stores through one descriptor: prefetch it here, once per task (see "Descriptor prefetch").
        # A split main_grad is written through its expert's own descriptor (d_desc_ptr), not through the atom's.
        if cutlass.const_expr(d_desc_ptr is None):
            cpasync.prefetch_descriptor(d_atom)
        minibatch_size = runtime[_RT_MINIBATCH_SIZE]
        i_tiles = runtime[_RT_I_TILES]
        acc_shape = tiled_mma.partition_shape_C(collective.mma_tiler[:2])
        fake = tiled_mma.make_fragment_C(cute.append(acc_shape, MLP_ACC_STAGES))
        base = cute.make_tensor(acc_tmem_ptr, fake.layout)
        tiled_copy_t2r, tTR_tAcc, tTR_rAcc = collective.epilog_tmem_copy_and_partition(thread, base, epi_tile, collective.use_2cta_instrs)
        tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, Float32)
        # The destination's dtype comes from the collective: the Wgrad one writes FP32 straight
        # into a float32 main_grad, every other epilogue rounds to bfloat16.
        d_dtype = collective.d_dtype_expected
        tiled_copy_r2s, tRS_rD, tRS_sD = collective.epilog_smem_store_and_partition(
            tiled_copy_t2r,
            collective.d_layout,
            d_dtype,
            tTR_rD,
            sD,
            thread,
        )
        rD_out = cute.make_rmem_tensor_like(tRS_rD, d_dtype)
        tile_coord = (
            row * Int32(CLUSTER_SIZE) + cta_rank,
            column,
            Int32(0),
            Int32(0),
        )
        if cutlass.const_expr(routed_wgrad):
            # QuACK's epilogue helper tiles only the first two modes.  As
            # in its batched GEMM path, select E before partitioning so
            # this ordinary TMA store owns exactly one expert matrix.
            #
            # Split main_grad buffers have no E mode: the expert is named by a descriptor.  A
            # combined FC1 shares one buffer -- grouped or split -- whose Up half starts
            # d_row_tile_offset tiles down.
            if cutlass.const_expr(not self.split_main_grads):
                d_tensor = d_tensor[None, None, expert]
            if cutlass.const_expr(d_row_tile_offset is not None):
                tile_coord = (tile_coord[0] + d_row_tile_offset, *tile_coord[1:])
        if cutlass.const_expr(d_desc_ptr is not None):
            copy_d, _, _ = _epilog_gmem_copy_fn(
                collective,
                d_atom,
                d_tensor,
                collective.cta_tile_shape_mnk[:2],
                epi_tile,
                sD,
                tile_coord,
                tma_desc_ptr=d_desc_ptr,
            )
        else:
            copy_d, _, _ = collective.epilog_gmem_copy_and_partition(
                d_atom,
                d_tensor,
                collective.cta_tile_shape_mnk[:2],
                epi_tile,
                sD,
                tile_coord,
            )
        acc_index = acc_consumer_state.index
        if cutlass.const_expr(acc_slot >= 0):
            acc_index = Int32(acc_slot)
        tTR_tAcc = tTR_tAcc[None, None, None, None, None, acc_index]
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        if cutlass.const_expr(wait_for_acc):
            acc_pipeline.consumer_wait(acc_consumer_state)
        store_pipeline = collective.make_epi_store_pipeline()
        is_store_thread = Boolean(warp == Int32(0))
        if cutlass.const_expr(ordered_wait):
            # Split expert, segment k > 0: reduce-add only after the k earlier segments of
            # this dW tile were stored (macrobatch order => deterministic sum).  Both CTAs
            # of a cluster publish once per segment (each owns half the tile), so k
            # complete segments == k * CLUSTER_SIZE arrivals.
            if is_store_thread:
                with cute.arch.elect_one():
                    _counter_wait_gpu(wgrad_segment_done, self._wgrad_counter_index(runtime, kind, expert, row, column), segment_index * Int32(CLUSTER_SIZE))
                cute.arch.sync_warp()
        epi_shape = cute.zipped_divide(cute.make_layout(collective.cta_tile_shape_mnk[:2]), epi_tile).shape[1]
        epi_layout = cute.make_ordered_layout(epi_shape, order=(0, 1))
        for piece in cutlass.range_constexpr(MLP_EPILOGUE_PIECES):
            coordinate = epi_layout.get_hier_coord(piece)
            collective.epi_load_acc_subtile(
                tiled_copy_t2r,
                tiled_copy_r2s,
                tTR_tAcc,
                tTR_rAcc,
                tRS_rD,
                coordinate,
                acc_pipeline,
                acc_consumer_state,
                acc_release_idx,
            )
            if cutlass.const_expr(manual_acc_release):
                if cutlass.const_expr(piece == MLP_EPILOGUE_PIECES - 1):
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
            rD_out.store(tRS_rD.load().to(d_dtype))
            if is_store_thread:
                store_pipeline.producer_acquire()
            named_barrier_sync_unaligned(collective.epilogue_barrier)
            buffer = piece % collective.epi_stages
            cute.copy(
                tiled_copy_r2s,
                rD_out,
                tRS_sD[None, None, None, buffer],
            )
            cute.arch.fence_view_async_shared()
            named_barrier_sync_unaligned(collective.epilogue_barrier)
            if is_store_thread:
                copy_d(src_idx=buffer, dst_idx=coordinate)
                store_pipeline.producer_commit()
        if cutlass.const_expr(advance_acc_state):
            acc_consumer_state.advance()
        if is_store_thread:
            cute.arch.cp_async_bulk_wait_group(0, read=False)
            store_pipeline.producer_tail()
            with cute.arch.elect_one():
                # Down dgrad tasks never come here (their epilogues are _run_down_swiglu_epilogue and
                # _run_routed_down_swapab_epilogue), nor routed Gate/Up dgrad tasks
                # (_run_routed_dx_swapab_epilogue).
                if kind == Int32(REPLAY_GATE_UP):
                    if cutlass.const_expr(replay_ready_increment > 0):
                        global_row = generation * Int32(minibatch_size // MLP_TILE_M) + row % Int32(minibatch_size // MLP_TILE_M)
                        _counter_arrive_gpu(
                            replayed_gate_up_ready,
                            global_row * i_tiles + column,
                            Int32(replay_ready_increment),
                        )
        named_barrier_sync_unaligned(collective.epilogue_barrier)
        if cutlass.const_expr(routed_wgrad):
            # Split expert with a later segment: publish this CTA's half of the tile (the
            # store group was drained before the CTA barrier above); one arrival per CTA.
            if publish_segment != Int32(0):
                if thread == Int32(0):
                    _counter_arrive_gpu(wgrad_segment_done, self._wgrad_counter_index(runtime, kind, expert, row, column))
        return acc_consumer_state

    @cute.jit
    def _run_routed_dx_swapab_epilogue(
        self,
        runtime: tuple,
        collective,
        tiled_mma: cute.TiledMma,
        epi_tile: cute.Tile,
        thread: Int32,
        warp: Int32,
        cta_rank: Int32,
        generation: Int32,
        row: Int32,
        column: Int32,
        tile_valid: Int32,
        d_atom: cute.CopyAtom,
        d_tensor: cute.Tensor,
        sDT: cute.Tensor,
        acc_tmem_ptr,
        acc_pipeline,
        acc_consumer_state,
        d_x_routed_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        num_tokens: Int32,
    ):
        """Routed Gate/Up dgrad epilogue, swap-AB: TMEM lanes are this CTA's 128 H features, the
        columns the block's 256 ring rows, so each CTA writes all 256 rows of the block for its 128
        features (the slot-reuse wait on reverse_dispatch_done covers the whole generation and sits in
        the load warp).  Every 32-token subtile below the tile's N = align16(tile_valid) goes
        M-major (features contiguous) into a D stage with stmatrix .trans and is stored by TMA through
        ``d_tensor``, the (H, ring rows) view of dX; subtiles at or past N hold padding rows only, which
        the reverse dispatch never pushes, so they are staged like the others (uniform barriers) but not
        stored.  Rows in [tile_valid, N) are stored from padding dGate / dUp rows exactly as the unshrunk
        tile stored them; a stored subtile's rows in [N, align32(N)) get stale TMEM columns, which no
        consumer reads either (the forward Down store does the same).  After its last store the task
        arrives on d_x_routed_ready, and on routed_buffers_done when the minibatch one macrobatch later
        exists (it reuses this ring slot)."""
        # The task stores through one descriptor: prefetch it here, once per task (see "Descriptor prefetch").
        cpasync.prefetch_descriptor(d_atom)
        minibatch_size = runtime[_RT_MINIBATCH_SIZE]
        minibatches_per_macrobatch = runtime[_RT_MINIBATCHES_PER_MACROBATCH]
        acc_shape = tiled_mma.partition_shape_C(collective.mma_tiler[:2])
        fake = tiled_mma.make_fragment_C(cute.append(acc_shape, MLP_ACC_STAGES))
        base = cute.make_tensor(acc_tmem_ptr, fake.layout)
        (
            tiled_copy_t2r,
            tTR_tAcc,
            tTR_rAcc,
            tiled_copy_r2s,
            tRS_rD,
            tRS_sD,
        ) = _swapab_fwd.epilogue_copies(collective.cta_tile_shape_mnk, base, epi_tile, collective.use_2cta_instrs, sDT, thread)
        rD_bf16 = cute.make_rmem_tensor_like(tRS_rD, BFloat16)
        tTR_tAcc = tTR_tAcc[None, None, None, None, None, acc_consumer_state.index]
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        tile_n = _swapab_fwd.mma_n(tile_valid)
        acc_pipeline.consumer_wait(acc_consumer_state)
        store_pipeline = collective.make_epi_store_pipeline()
        is_store_thread = Boolean(warp == Int32(0))
        epi_shape = cute.zipped_divide(cute.make_layout(collective.cta_tile_shape_mnk[:2]), epi_tile).shape[1]
        epi_layout = cute.make_ordered_layout(epi_shape, order=(0, 1))
        for piece in cutlass.range_constexpr(MLP_EPILOGUE_PIECES):
            coordinate = epi_layout.get_hier_coord(piece)
            collective.epi_load_acc_subtile(
                tiled_copy_t2r,
                tiled_copy_r2s,
                tTR_tAcc,
                tTR_rAcc,
                tRS_rD,
                coordinate,
                acc_pipeline,
                acc_consumer_state,
                MLP_EPILOGUE_PIECES - 1,
            )
            # Pieces past the valid rows are staged like the others (the barriers stay uniform and no
            # Python object enters a dynamic branch); only their TMA store is skipped.
            live = Int32(Int32(piece * _SWAP_EPI_ROWS) < tile_n)
            rD_bf16.store(tRS_rD.load().to(BFloat16))
            if is_store_thread:
                store_pipeline.producer_acquire()
            named_barrier_sync_unaligned(collective.epilogue_barrier)
            buffer = piece % MLP_BF16_D_TILES
            cute.copy(tiled_copy_r2s, rD_bf16, tRS_sD[None, None, None, buffer])
            cute.arch.fence_view_async_shared()
            named_barrier_sync_unaligned(collective.epilogue_barrier)
            if is_store_thread:
                if (Int32(thread == Int32(0)) & live) != Int32(0):
                    _swapab_fwd.store_tile(
                        d_atom,
                        d_tensor,
                        (MLP_TILE_M // CLUSTER_SIZE, _SWAP_EPI_ROWS),
                        (column * Int32(CLUSTER_SIZE) + cta_rank, row * Int32(MLP_EPILOGUE_PIECES) + Int32(piece)),
                        sDT[None, None, buffer],
                    )
                store_pipeline.producer_commit()
        acc_consumer_state.advance()
        if is_store_thread:
            cute.arch.cp_async_bulk_wait_group(0, read=False)
            store_pipeline.producer_tail()
            with cute.arch.elect_one():
                _counter_arrive_gpu(d_x_routed_ready, generation)
                if generation + Int32(minibatches_per_macrobatch) < _ceil_div_i32(num_tokens, Int32(minibatch_size)):
                    _counter_arrive_gpu(routed_buffers_done, generation)
        named_barrier_sync_unaligned(collective.epilogue_barrier)
        return acc_consumer_state

    @cute.jit
    def _run_down_swiglu_epilogue(
        self,
        runtime: tuple,
        collective,
        tiled_mma: cute.TiledMma,
        epi_tile: cute.Tile,
        thread: Int32,
        warp: Int32,
        cta_rank: Int32,
        row: Int32,
        column: Int32,
        acc_tmem_ptr,
        acc_pipeline,
        acc_consumer_state,
        fused_swiglu: tuple,
        tail: cutlass.Constexpr[bool] = False,
    ):
        """Shared Down dgrad epilogue with the shared SwiGLU-backward fused in (no dH round trip
        through HBM, no SwiGLU task; rows = local tokens, no ring, router weight 1, no w*Hidden, no
        router partial, inputs from the forward context, no comm waits).  Each epilogue thread holds one
        row of this CTA's 128 x 256 dH tile per 32-column piece (the TMEM load layout); it reads the
        row's Gate/Up values at the same positions with plain global loads (partitioned like the
        accumulator) and writes d_gate = dGate(dH), d_up = dUp(dH) with plain global stores.  dH is
        rounded to BF16 first, so every output is bit-identical to a separate BF16 dH store followed
        by a SwiGLU-backward pass.  The routed Down dgrad's fused epilogue is
        _run_routed_down_swapab_epilogue.

        ``tail`` (compile time): the instantiation for a tail tile (the last column block of an I that is not a
        multiple of 256).  Its valid columns are the first TAIL_WIDTH, a static count of pieces, and it arrives
        for one SwiGLU half."""
        _, _, _, _, _, _, _, _, d_gate_up_ready, _, gate_shared, up_shared, d_gate_shared, d_up_shared = fused_swiglu
        swiglu_limit = runtime[_RT_SWIGLU_LIMIT]
        situ_beta1 = runtime[_RT_SITU_BETA1]
        situ_beta2 = runtime[_RT_SITU_BETA2]
        acc_shape = tiled_mma.partition_shape_C(collective.mma_tiler[:2])
        fake = tiled_mma.make_fragment_C(cute.append(acc_shape, MLP_ACC_STAGES))
        base = cute.make_tensor(acc_tmem_ptr, fake.layout)
        tiled_copy_t2r, tTR_tAcc, tTR_rAcc = collective.epilog_tmem_copy_and_partition(thread, base, epi_tile, collective.use_2cta_instrs)
        tTR_tAcc = tTR_tAcc[None, None, None, None, None, acc_consumer_state.index]
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        thr_copy_t2r = tiled_copy_t2r.get_slice(thread)
        cta_tile = collective.cta_tile_shape_mnk[:2]
        row_tile = row * Int32(CLUSTER_SIZE) + cta_rank

        def partition_rows(tensor):
            tile = cute.local_tile(tensor, cta_tile, (row_tile, column))
            return thr_copy_t2r.partition_D(cute.flat_divide(tile, epi_tile))

        tTR_gGate = partition_rows(gate_shared)
        tTR_gUp = partition_rows(up_shared)
        tTR_gDGate = partition_rows(d_gate_shared)
        tTR_gDUp = partition_rows(d_up_shared)
        named_barrier_sync_unaligned(collective.epilogue_barrier)
        acc_pipeline.consumer_wait(acc_consumer_state)

        # Two pieces per (dynamic) iteration with register double buffering: the Gate/Up loads of the
        # next piece are in flight while this piece computes (one load round trip per piece would
        # serialize eight latencies per tile on four warps), and the binary holds two 32-column bodies.
        r_gate = (cute.make_rmem_tensor(tTR_rAcc.shape, BFloat16), cute.make_rmem_tensor(tTR_rAcc.shape, BFloat16))
        r_up = (cute.make_rmem_tensor(tTR_rAcc.shape, BFloat16), cute.make_rmem_tensor(tTR_rAcc.shape, BFloat16))
        if cutlass.const_expr(tail):
            # A tail tile owns the first TAIL_WIDTH columns of its 256-column CTA tile: _TAIL_PIECES pieces.
            # The Gate/Up accesses are plain vectorised global copies, not TMA, so nothing would drop a piece
            # past them: on a row-major (tokens, I) tensor it aliases the NEXT row's leading columns.  The
            # loop is static over the valid pieces, one 32-column body without the double buffering; the
            # last of them releases the accumulator, and the pieces past the tail are neither loaded nor
            # written.
            # (Names of its own: the loop below assigns its piece, buffers and results too.)
            for tail_piece in cutlass.range(_TAIL_PIECES, unroll=1):
                cute.autovec_copy(tTR_gGate[None, None, None, 0, tail_piece], r_gate[0])
                cute.autovec_copy(tTR_gUp[None, None, None, 0, tail_piece], r_up[0])
                cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, (0, tail_piece)], tTR_rAcc)
                if tail_piece == Int32(_TAIL_PIECES - 1):
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                tail_gate = r_gate[0]
                tail_up = r_up[0]
                for index in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                    tail_d_gate, tail_d_up, tail_hidden = _swiglu_backward_element(
                        Float32(tail_gate[index]),
                        Float32(tail_up[index]),
                        Float32(tTR_rAcc[index].to(BFloat16)),
                        swiglu_limit,
                        self.is_clamped,
                        self.is_situ,
                        situ_beta1,
                        situ_beta2,
                    )
                    tail_gate[index] = tail_d_gate.to(BFloat16)
                    tail_up[index] = tail_d_up.to(BFloat16)
                cute.autovec_copy(tail_gate, tTR_gDGate[None, None, None, 0, tail_piece])
                cute.autovec_copy(tail_up, tTR_gDUp[None, None, None, 0, tail_piece])
            acc_consumer_state.advance()
            # d_gate/d_up are read by TMA (async proxy) in other CTAs after d_gate_up_ready.
            cute.arch.fence_proxy("async.global")
            named_barrier_sync_unaligned(collective.epilogue_barrier)
            if warp == Int32(0):
                with cute.arch.elect_one():
                    # The tail tile's one 128-wide SwiGLU half: one arrival where a full tile makes two (a
                    # second one would release the consumer before the row block's d_gate/d_up are written).
                    _counter_arrive_gpu(d_gate_up_ready, row, Int32(TAIL_WIDTH // SWIGLU_TILE_N))
            return acc_consumer_state
        cute.autovec_copy(tTR_gGate[None, None, None, 0, 0], r_gate[0])
        cute.autovec_copy(tTR_gUp[None, None, None, 0, 0], r_up[0])
        for pair in cutlass.range(MLP_EPILOGUE_PIECES // 2, unroll=1):
            for half in cutlass.range_constexpr(2):
                piece = pair * Int32(2) + Int32(half)
                # Prefetch the next piece into the other buffer (piece 7 has none).
                if cutlass.const_expr(half == 0):
                    cute.autovec_copy(tTR_gGate[None, None, None, 0, piece + Int32(1)], r_gate[1])
                    cute.autovec_copy(tTR_gUp[None, None, None, 0, piece + Int32(1)], r_up[1])
                else:
                    if piece + Int32(1) < Int32(MLP_EPILOGUE_PIECES):
                        cute.autovec_copy(tTR_gGate[None, None, None, 0, piece + Int32(1)], r_gate[0])
                        cute.autovec_copy(tTR_gUp[None, None, None, 0, piece + Int32(1)], r_up[0])
                cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, (0, piece)], tTR_rAcc)
                if cutlass.const_expr(half == 1):
                    if piece == Int32(MLP_EPILOGUE_PIECES - 1):
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                gate_in = r_gate[half]
                up_in = r_up[half]
                for element in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                    d_gate, d_up, _ = _swiglu_backward_element(
                        Float32(gate_in[element]),
                        Float32(up_in[element]),
                        Float32(tTR_rAcc[element].to(BFloat16)),
                        swiglu_limit,
                        self.is_clamped,
                        self.is_situ,
                        situ_beta1,
                        situ_beta2,
                    )
                    gate_in[element] = d_gate.to(BFloat16)
                    up_in[element] = d_up.to(BFloat16)
                cute.autovec_copy(gate_in, tTR_gDGate[None, None, None, 0, piece])
                cute.autovec_copy(up_in, tTR_gDUp[None, None, None, 0, piece])
        acc_consumer_state.advance()
        # d_gate/d_up are read by TMA (async proxy) in other CTAs after d_gate_up_ready.
        cute.arch.fence_proxy("async.global")
        named_barrier_sync_unaligned(collective.epilogue_barrier)
        if warp == Int32(0):
            with cute.arch.elect_one():
                # Two 128 x 128 SwiGLU tiles per CTA tile: 2 x s_tiles arrivals per row block.
                swiglu_halves = Int32(2)
                _counter_arrive_gpu(d_gate_up_ready, row, swiglu_halves)
        return acc_consumer_state

    @cute.jit
    def _run_routed_down_swapab_epilogue(
        self,
        runtime: tuple,
        collective,
        tiled_mma: cute.TiledMma,
        epi_tile: cute.Tile,
        thread: Int32,
        warp: Int32,
        cta_rank: Int32,
        generation: Int32,
        row: Int32,
        column: Int32,
        tile_valid: Int32,
        sCT: cute.Tensor,
        acc_tmem_ptr,
        acc_pipeline,
        acc_consumer_state,
        fused_swiglu: tuple,
        replayed_gate_up_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        num_tokens: Int32,
        num_local_tokens: Int32,
        tail: cutlass.Constexpr[bool] = False,
    ):
        """Routed Down dgrad epilogue with the routed SwiGLU-backward fused in, swap-AB: TMEM lane m is
        feature column * 256 + cta_rank * 128 + m (this CTA's 128 features are exactly SwiGLU column tile
        column * 2 + cta_rank), the columns are the block's 256 ring rows.  Per 32-token piece below the
        tile's N (_routed_down_n: the valid rows rounded up to the Wgrad K block): the piece's dH is
        rounded to BF16 and staged transposed (tokens as rows) into one of two tiles in the D stages
        (``sCT``); one barrier; every thread takes one token and 32 consecutive features, reads dH back
        as 128-bit vectors and runs the SwiGLU-backward (Gate/Up read with plain loads,
        d_gate = dGate(dH * w), d_up = dUp(dH * w) and w * Hidden written with plain stores), so every
        written value is bit-identical to the unshrunk tile's.  A thread works through its 32 features in a
        runtime loop over groups of _SWAP_FEATURE_GROUP (register lifetimes, see the loop).  The router
        partial <dH, Hidden> of a token is summed per chunk: every thread sums its 32 features one by one
        from zero, the odd-chunk threads add the sum of the even neighbour (one shuffle; chunk 0 + chunk 1,
        chunk 2 + chunk 3) and the last chunk's thread adds half 0 + half 1 and stores
        d_router_partials[ring row, SwiGLU tile].  Rows in [tile_valid, N) are padding rows written from
        the padding dY (zero) / Gate / Up rows (zero router partial); rows at or past N
        are padding rows no routed Wgrad or reverse dispatch reads, so their pieces are neither loaded nor
        written (the last live piece releases the accumulator).  The waits and arrivals (d_y_routed_ready,
        replayed_gate_up_ready, reverse_dispatch_done; d_gate_up_ready += 2 per CTA, routed_buffers_done)
        are made here.

        ``tail`` (compile time): the instantiation for the CTA of a tail tile (the last feature block of an I
        that is not a multiple of 256) whose 128 features are past I, the cluster's second one: it makes the
        waits, releases the accumulator and arrives on routed_buffers_done, and reads and writes nothing.
        The first CTA of a tail tile owns all of its features and runs the code of a full tile."""
        (
            gate_routed,
            up_routed,
            d_gate_routed,
            d_up_routed,
            hidden_routed,
            router_weights,
            schedule_peer_rank,
            d_router_partials,
            d_gate_up_ready,
            d_y_routed_ready,
            _,
            _,
            _,
            _,
        ) = fused_swiglu
        minibatch_size = runtime[_RT_MINIBATCH_SIZE]
        minibatches_per_macrobatch = runtime[_RT_MINIBATCHES_PER_MACROBATCH]
        i_tiles = runtime[_RT_I_TILES]
        swiglu_limit = runtime[_RT_SWIGLU_LIMIT]
        situ_beta1 = runtime[_RT_SITU_BETA1]
        situ_beta2 = runtime[_RT_SITU_BETA2]
        ep_size = runtime[_RT_EP_SIZE]
        acc_shape = tiled_mma.partition_shape_C(collective.mma_tiler[:2])
        fake = tiled_mma.make_fragment_C(cute.append(acc_shape, MLP_ACC_STAGES))
        base = cute.make_tensor(acc_tmem_ptr, fake.layout)
        tiled_copy_t2r, tTR_tAcc, tTR_rAcc = collective.epilog_tmem_copy_and_partition(thread, base, epi_tile, collective.use_2cta_instrs)
        tTR_tAcc = tTR_tAcc[None, None, None, None, None, acc_consumer_state.index]
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        # (lane m, column n) of every register element of piece 0 (QuACK's partition of the CTA tile's
        # identity tensor): m is this thread's feature in the CTA, n the token within the piece.
        piece_coords = tiled_copy_t2r.get_slice(thread).partition_D(cute.flat_divide(cute.make_identity_tensor(collective.cta_tile_shape_mnk[:2]), epi_tile))[
            None, None, None, 0, 0
        ]
        feature_in_cta = piece_coords[0][0]
        epilogue_barrier = collective.epilogue_barrier
        rows_per_minibatch = minibatch_size // Int32(MLP_TILE_M)
        global_row_block = generation * rows_per_minibatch + row % rows_per_minibatch
        feature_tile = column * Int32(CLUSTER_SIZE) + cta_rank  # SwiGLU column tile of this CTA's features
        token_of_thread = thread // Int32(4)
        chunk_of_thread = thread % Int32(4)
        feature_chunk = feature_tile * Int32(4) + chunk_of_thread  # 32-feature chunk of I
        feature_group = feature_chunk * Int32(_SWAP_FEATURE_GROUPS)  # its first group of I
        staged_group = chunk_of_thread * Int32(_SWAP_FEATURE_GROUPS)  # the same group in a staged dH row
        pieces = _routed_down_n(tile_valid) // Int32(_SWAP_EPI_ROWS)  # live pieces, an even number

        # Inputs of the fused SwiGLU: the router scores of generation g (written by the communication
        # CTAs before d_y_routed_ready; the load warp's acquire of that counter does not order these
        # threads' plain loads, so acquire it here too -- already complete, no wait), the replayed
        # Gate/Up of this tile for a replayed minibatch, and the reverse dispatch of g - mpm, which
        # read the d_router partials slot this epilogue overwrites.
        if warp == Int32(0):
            with cute.arch.elect_one():
                _counter_wait_gpu(d_y_routed_ready, generation, self._comm_tiles(runtime, generation))
                if cutlass.const_expr(self.recompute_forward_context):
                    # Every generation is replayed; only a later macrobatch's has a slot predecessor.
                    _counter_wait_gpu(replayed_gate_up_ready, global_row_block * i_tiles + column, Int32(4))
                    if generation >= Int32(minibatches_per_macrobatch):
                        _counter_wait_gpu(
                            runtime[_RT_REVERSE_DISPATCH_DONE],
                            generation - Int32(minibatches_per_macrobatch),
                            self._reverse_dispatch_done_required(runtime, generation - Int32(minibatches_per_macrobatch)),
                        )
                else:
                    if generation >= Int32(minibatches_per_macrobatch):
                        _counter_wait_gpu(replayed_gate_up_ready, global_row_block * i_tiles + column, Int32(4))
                        _counter_wait_gpu(
                            runtime[_RT_REVERSE_DISPATCH_DONE],
                            generation - Int32(minibatches_per_macrobatch),
                            self._reverse_dispatch_done_required(runtime, generation - Int32(minibatches_per_macrobatch)),
                        )
        named_barrier_sync_unaligned(epilogue_barrier)
        acc_pipeline.consumer_wait(acc_consumer_state)
        if cutlass.const_expr(tail):
            # None of the tile's valid features is this CTA's: its Gate / Up chunks, d_gate / d_up / w*Hidden
            # chunks and router partial would be past I (plain accesses, which alias the NEXT ring row), and
            # its SwiGLU column tile does not exist, so it makes no d_gate_up_ready arrival.  What is left is
            # the accumulator's release and the row block's routed_buffers_done arrival.
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()
            named_barrier_sync_unaligned(epilogue_barrier)
            if warp == Int32(0):
                with cute.arch.elect_one():
                    if generation + Int32(minibatches_per_macrobatch) < _ceil_div_i32(num_tokens, Int32(minibatch_size)):
                        _counter_arrive_gpu(routed_buffers_done, generation)
            return acc_consumer_state
        r_dh_lane = cute.make_rmem_tensor_like(tTR_rAcc, BFloat16)
        # One group of the thread's features each (the group loop below).
        r_gate = cute.make_rmem_tensor((_SWAP_FEATURE_GROUP,), BFloat16)  # Gate in, d_gate out
        r_up = cute.make_rmem_tensor((_SWAP_FEATURE_GROUP,), BFloat16)  # Up in, d_up out
        r_gate_next = cute.make_rmem_tensor((_SWAP_FEATURE_GROUP,), BFloat16)  # the next group's Gate / Up, in flight
        r_up_next = cute.make_rmem_tensor((_SWAP_FEATURE_GROUP,), BFloat16)
        r_dh = cute.make_rmem_tensor((_SWAP_FEATURE_GROUP,), BFloat16)  # the group's staged dH
        r_hidden_bf16 = cute.make_rmem_tensor((_SWAP_FEATURE_GROUP,), BFloat16)  # w * Hidden
        # One runtime loop over the live pieces: the staging tile is the runtime value piece & 1, so the binary holds
        # one body.  As a runtime pair loop over two compile-time halves it held two, and the second half's group loop
        # ran with the lane predicates of the router partial (chunk parity, last chunk), first formed behind the first
        # half's loop, still live: one free predicate register for the clamp compares of the clamped SwiGLU, which
        # then go through it one after another.
        for piece in cutlass.range(pieces, unroll=1):
            half = piece & Int32(1)  # staging tile
            token_in_block = piece * Int32(_SWAP_EPI_ROWS) + token_of_thread
            ring_row = row * Int32(MLP_TILE_M) + token_in_block
            # This thread's token (four threads share one) and the first group of its 32 Gate/Up
            # features, issued before the staging so their latency overlaps the TMEM load, the staging
            # and the barrier.  The score of ring row r is rewritten every mpm minibatches by the
            # communication CTAs of this kernel: read it only after the d_y_routed_ready acquire above,
            # at gpu scope.
            peer_rank = schedule_peer_rank[global_row_block * Int32(MLP_TILE_M) + token_in_block]
            routed_row = peer_rank >= Int32(0) and peer_rank < ep_size
            router_weight = cute.arch.load(router_weights.iterator + ring_row, Float32, sem="relaxed", scope="gpu")
            cute.autovec_copy(cute.local_tile(gate_routed[ring_row, None], (_SWAP_FEATURE_GROUP,), (feature_group,)), r_gate_next)
            cute.autovec_copy(cute.local_tile(up_routed[ring_row, None], (_SWAP_FEATURE_GROUP,), (feature_group,)), r_up_next)
            # The piece's dH: TMEM -> registers (this thread's feature x 32 tokens), BF16, scattered
            # transposed into staging tile ``half``; the last live piece releases the accumulator.
            cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, (0, piece)], tTR_rAcc)
            if piece + Int32(1) == pieces:
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
            r_dh_lane.store(tTR_rAcc.load().to(BFloat16))
            for value in cutlass.range_constexpr(cute.size(r_dh_lane)):
                _, token = piece_coords[value]
                sCT[Int32(token), feature_in_cta, half] = r_dh_lane[value]
            named_barrier_sync_unaligned(epilogue_barrier)
            # The thread's 32 features go through a runtime loop over groups of _SWAP_FEATURE_GROUP instead
            # of one straight-line block.  Register lifetimes: the d_gate / d_up / w * Hidden of a group
            # are converted to BF16 and stored before the next group is computed, so one group's inputs
            # and results are live at a time and not all 32 features' (this block was the register peak
            # of the kernel).  It has to be a loop: within one basic block the assembler may schedule
            # every element's math first and all stores last, whatever the order written here; it does
            # not move work across the loop's back edge.  Gate / Up of a group are loaded one group
            # ahead (the first before the TMEM load above), so their latency overlaps the staging and the
            # barrier or the previous group's math.  The register buffers are indexed with constants
            # only; the runtime group index selects the global / staging tile.  chunk_partial is this
            # thread's share of the router partial, its 32 features one by one from zero.
            chunk_partial = Float32(0.0)
            for group in cutlass.range(_SWAP_FEATURE_GROUPS, unroll=1):
                for element in cutlass.range_constexpr(_SWAP_FEATURE_GROUP):  # register moves
                    r_gate[element] = r_gate_next[element]
                    r_up[element] = r_up_next[element]
                if group + Int32(1) < Int32(_SWAP_FEATURE_GROUPS):
                    cute.autovec_copy(cute.local_tile(gate_routed[ring_row, None], (_SWAP_FEATURE_GROUP,), (feature_group + group + Int32(1),)), r_gate_next)
                    cute.autovec_copy(cute.local_tile(up_routed[ring_row, None], (_SWAP_FEATURE_GROUP,), (feature_group + group + Int32(1),)), r_up_next)
                cute.autovec_copy(cute.local_tile(sCT[token_of_thread, None, half], (_SWAP_FEATURE_GROUP,), (staged_group + group,)), r_dh)
                for element in cutlass.range_constexpr(_SWAP_FEATURE_GROUP):
                    d_gate, d_up, hidden = _swiglu_backward_element(
                        Float32(r_gate[element]),
                        Float32(r_up[element]),
                        Float32(r_dh[element]) * router_weight,
                        swiglu_limit,
                        self.is_clamped,
                        self.is_situ,
                        situ_beta1,
                        situ_beta2,
                    )
                    r_gate[element] = d_gate.to(BFloat16)
                    r_up[element] = d_up.to(BFloat16)
                    r_hidden_bf16[element] = (hidden * router_weight).to(BFloat16)
                    chunk_partial = chunk_partial + Float32(r_dh[element]) * hidden
                cute.autovec_copy(r_gate, cute.local_tile(d_gate_routed[ring_row, None], (_SWAP_FEATURE_GROUP,), (feature_group + group,)))
                cute.autovec_copy(r_up, cute.local_tile(d_up_routed[ring_row, None], (_SWAP_FEATURE_GROUP,), (feature_group + group,)))
                cute.autovec_copy(r_hidden_bf16, cute.local_tile(hidden_routed[ring_row, None], (_SWAP_FEATURE_GROUP,), (feature_group + group,)))
            # Router partial of the token over this SwiGLU tile: chunks 0 and 2 are their own sums, chunks 1
            # and 3 add the sum of the lane below (features 0..63, 64..127), chunk 3 adds half 0 (two lanes
            # below) + half 1 and stores.  A half is (chunk sum) + (chunk sum), each chunk summed from zero, not
            # one running sum over its 64 features.  Padding rows (not routed) add nothing and store 0.
            if not routed_row:
                chunk_partial = Float32(0.0)
            router_partial = chunk_partial
            below = cute.arch.shuffle_sync_up(chunk_partial, 1, mask=0xFFFFFFFF, mask_and_clamp=0)
            if chunk_of_thread % Int32(2) == Int32(1):
                router_partial = below + chunk_partial
            router_half = cute.arch.shuffle_sync_up(router_partial, 2, mask=0xFFFFFFFF, mask_and_clamp=0)
            store_partial = chunk_of_thread == Int32(3)
            if store_partial:
                d_router_partials[ring_row, feature_tile] = router_half + router_partial
        acc_consumer_state.advance()
        # d_gate/d_up/w*Hidden are read by TMA (async proxy) in other CTAs after d_gate_up_ready.
        cute.arch.fence_proxy("async.global")
        named_barrier_sync_unaligned(epilogue_barrier)
        if warp == Int32(0):
            with cute.arch.elect_one():
                # One 256 x 128 SwiGLU tile per CTA, two per cluster tile: the per-row-block
                # requirement is 2 x s_tiles.  This CTA owns SwiGLU column tile column * 2 + cta_rank.
                _counter_arrive_gpu(d_gate_up_ready, Int32(num_local_tokens // MLP_TILE_M) + global_row_block, Int32(2))
                if generation + Int32(minibatches_per_macrobatch) < _ceil_div_i32(num_tokens, Int32(minibatch_size)):
                    _counter_arrive_gpu(routed_buffers_done, generation)
        return acc_consumer_state

    @cute.jit
    def _store_zero_wgrad_tile(
        self,
        thread: Int32,
        cta_rank: Int32,
        row: Int32,
        column: Int32,
        expert: Int32,
        base: Int64,
        row_stride: Int64,
        expert_stride: Int64,
        tail: cutlass.Constexpr[bool] = False,
    ):
        """dW tile of an expert without rows (no MMA ran): the 128 epilogue threads write
        this CTA's 128 x 256 bf16 tile as zeros with plain 16 B stores, one row per
        thread.  base/strides (elements) describe the (rows, columns, expert) dW view
        with unit column stride; one instantiation serves all three matrices.  ``tail`` (compile
        time): the instantiation for a tile in the last column block of a column extent that is not a
        multiple of 256, which writes the first TAIL_WIDTH columns of each row -- plain stores are
        not clipped as the TMA ones are, and past the extent lies the next row.  (A CTA whose rows
        are past the row extent is not sent here at all.)"""
        tile_row = (row * Int32(CLUSTER_SIZE) + cta_rank) * Int32(MLP_TILE_M // CLUSTER_SIZE) + thread
        address = base + (Int64(expert) * expert_stride + Int64(tile_row) * row_stride + Int64(column) * Int64(MLP_TILE_N)) * Int64(2)
        if cutlass.const_expr(tail):
            if thread < Int32(MLP_TILE_M // CLUSTER_SIZE):
                for chunk in cutlass.range(TAIL_WIDTH * 2 // 16, unroll=4):
                    _store_global_zero16(address + Int64(chunk) * Int64(16))
        else:
            if thread < Int32(MLP_TILE_M // CLUSTER_SIZE):
                for chunk in cutlass.range(MLP_TILE_N * 2 // 16, unroll=4):
                    _store_global_zero16(address + Int64(chunk) * Int64(16))

    @cute.jit
    def _write_routed_wgrad_tile(
        self,
        runtime: tuple,
        tiled_mma_atb: cute.TiledMma,
        atb_epi_tile: cute.Tile,
        thread: Int32,
        warp: Int32,
        cta_rank: Int32,
        kind: Int32,
        generation: Int32,
        row: Int32,
        column: Int32,
        expert: Int32,
        expert_counts: cute.Tensor,
        store_atom: cute.CopyAtom,
        store_tensor: cute.Tensor,
        add_atom: cute.CopyAtom,
        add_tensor: cute.Tensor,
        sD_atb: cute.Tensor,
        acc_tmem_ptr,
        acc_pipeline,
        acc_consumer_state,
        replayed_gate_up_ready: cute.Tensor,
        wgrad_segment_done: cute.Tensor,
        wgrad_first: Int32,
        wgrad_has_next: Int32,
        d_desc_ptr=None,
        d_row_tile_offset=None,
    ):
        """One dW tile write-back: the expert's first macrobatch segment stores, later
        segments reduce-add after the earlier stores were published (ordered variant);
        every segment but the expert's last publishes.

        Accumulating into main_grad makes even the first segment additive -- the buffer already
        holds earlier microbatches' contributions -- so only the segment ordering is left, and
        that is unchanged: the sum stays in macrobatch order and so stays bit-reproducible."""
        if cutlass.const_expr(self.accumulate_wgrad):
            # The segment index is <= 0 for an expert's first segment, so clamping gives it a
            # zero-arrival wait -- the same no-op the non-accumulating path gets by passing 0.
            # Its own name: the DSL hoists a variable assigned in two branches to None first, and
            # then refuses the Int32 that a dynamic `if` would give it.
            accumulate_segment = cutlass.max(Int32(0), self._wgrad_segment_index(runtime, generation, expert, expert_counts))
            return self._run_one_epilogue_task(
                runtime,
                self.atb,
                tiled_mma_atb,
                atb_epi_tile,
                thread,
                warp,
                cta_rank,
                kind,
                generation,
                row,
                column,
                expert,
                add_atom,
                add_tensor,
                sD_atb,
                acc_tmem_ptr,
                acc_pipeline,
                acc_consumer_state,
                True,
                True,
                replayed_gate_up_ready,
                wgrad_segment_done,
                accumulate_segment,
                wgrad_has_next,
                ordered_wait=True,
                d_desc_ptr=d_desc_ptr,
                d_row_tile_offset=d_row_tile_offset,
            )
        # Both stores of a tile land where the accumulating path's do: a combined FC1's Up rows
        # are d_row_tile_offset tiles down the one buffer.
        if wgrad_first != Int32(0):
            acc_consumer_state = self._run_one_epilogue_task(
                runtime,
                self.atb,
                tiled_mma_atb,
                atb_epi_tile,
                thread,
                warp,
                cta_rank,
                kind,
                generation,
                row,
                column,
                expert,
                store_atom,
                store_tensor,
                sD_atb,
                acc_tmem_ptr,
                acc_pipeline,
                acc_consumer_state,
                True,
                False,
                replayed_gate_up_ready,
                wgrad_segment_done,
                Int32(0),
                wgrad_has_next,
                d_desc_ptr=d_desc_ptr,
                d_row_tile_offset=d_row_tile_offset,
            )
        else:
            segment = self._wgrad_segment_index(runtime, generation, expert, expert_counts)
            acc_consumer_state = self._run_one_epilogue_task(
                runtime,
                self.atb,
                tiled_mma_atb,
                atb_epi_tile,
                thread,
                warp,
                cta_rank,
                kind,
                generation,
                row,
                column,
                expert,
                add_atom,
                add_tensor,
                sD_atb,
                acc_tmem_ptr,
                acc_pipeline,
                acc_consumer_state,
                True,
                True,
                replayed_gate_up_ready,
                wgrad_segment_done,
                segment,
                wgrad_has_next,
                ordered_wait=True,
                d_desc_ptr=d_desc_ptr,
                d_row_tile_offset=d_row_tile_offset,
            )
        return acc_consumer_state

    @cute.jit
    def _run_routed_wgrad_epilogue(
        self,
        runtime: tuple,
        tiled_mma_atb: cute.TiledMma,
        atb_epi_tile: cute.Tile,
        thread: Int32,
        warp: Int32,
        cta_rank: Int32,
        kind: Int32,
        generation: Int32,
        row: Int32,
        column: Int32,
        expert: Int32,
        expert_counts: cute.Tensor,
        atb_d_atoms,
        atb_d_tensors,
        atb_d_add_atoms,
        atb_d_add_tensors,
        sD_atb: cute.Tensor,
        acc_tmem_ptr,
        acc_pipeline,
        acc_consumer_state,
        replayed_gate_up_ready: cute.Tensor,
        wgrad_segment_done: cute.Tensor,
        d_w_routed_down_hie: cute.Tensor,
        d_w_routed_gate_ihe: cute.Tensor,
        d_w_routed_up_ihe: cute.Tensor,
        wgrad_first: Int32 = Int32(1),
        wgrad_has_next: Int32 = Int32(0),
        zero_fill: cutlass.Constexpr = False,
        routed_descriptors=None,
        num_experts_rt: Int32 = Int32(0),
        main_grad_up_row_offset=None,
    ):
        """Routed Wgrad dW write-back by matrix (down, gate, up = kind - ROUTED_WGRAD_DOWN;
        store atoms 3..5, reduce-add atoms 0..2).  zero_fill: an empty expert's tile,
        written with plain stores through the (rows, columns, expert) dW views."""
        if cutlass.const_expr(zero_fill):
            base = Int64(d_w_routed_down_hie.iterator.toint())
            row_stride = Int64(d_w_routed_down_hie.stride[0])
            expert_stride = Int64(d_w_routed_down_hie.stride[2])
            # (Fresh dW only: a combined FC1's Gate / Up are already their own (I, H, E) halves, see __call__.)
            if kind == Int32(ROUTED_WGRAD_GATE):
                base = Int64(d_w_routed_gate_ihe.iterator.toint())
                row_stride = Int64(d_w_routed_gate_ihe.stride[0])
                expert_stride = Int64(d_w_routed_gate_ihe.stride[2])
            elif kind == Int32(ROUTED_WGRAD_UP):
                base = Int64(d_w_routed_up_ihe.iterator.toint())
                row_stride = Int64(d_w_routed_up_ihe.stride[0])
                expert_stride = Int64(d_w_routed_up_ihe.stride[2])
            if cutlass.const_expr(self.has_tail):
                # The matrix's extents, which only a tail tile clips to.  Down is (H, I, E); Gate / Up are (I, H, E).
                rows = Int32(d_w_routed_down_hie.shape[0])
                columns = Int32(d_w_routed_down_hie.shape[1])
                if kind != Int32(ROUTED_WGRAD_DOWN):
                    rows = Int32(d_w_routed_gate_ihe.shape[0])
                    columns = Int32(d_w_routed_gate_ihe.shape[1])
                # A tail tile is TAIL_WIDTH rows or columns short.  Past the row extent lies the whole row
                # block of the cluster's second CTA, which writes nothing (the next expert's rows, or the end
                # of the buffer, are there); past the column extent lies the second half of every row, which
                # the tail instantiation's static loop leaves out.  One CTA-uniform branch per task for each.
                if (row * Int32(CLUSTER_SIZE) + cta_rank + Int32(1)) * Int32(MLP_TILE_M // CLUSTER_SIZE) <= rows:
                    if (column + Int32(1)) * Int32(MLP_TILE_N) > columns:
                        self._store_zero_wgrad_tile(thread, cta_rank, row, column, expert, base, row_stride, expert_stride, tail=True)
                    else:
                        self._store_zero_wgrad_tile(thread, cta_rank, row, column, expert, base, row_stride, expert_stride)
            else:
                self._store_zero_wgrad_tile(thread, cta_rank, row, column, expert, base, row_stride, expert_stride)
        elif kind == Int32(ROUTED_WGRAD_DOWN):
            acc_consumer_state = self._write_routed_wgrad_tile(
                runtime,
                tiled_mma_atb,
                atb_epi_tile,
                thread,
                warp,
                cta_rank,
                kind,
                generation,
                row,
                column,
                expert,
                expert_counts,
                atb_d_atoms[3],
                atb_d_tensors[3],
                atb_d_add_atoms[0],
                atb_d_add_tensors[0],
                sD_atb,
                acc_tmem_ptr,
                acc_pipeline,
                acc_consumer_state,
                replayed_gate_up_ready,
                wgrad_segment_done,
                wgrad_first,
                wgrad_has_next,
                d_desc_ptr=self._main_grad_descriptor(routed_descriptors, num_experts_rt, 5, expert),
                d_row_tile_offset=None,
            )
        elif kind == Int32(ROUTED_WGRAD_GATE):
            acc_consumer_state = self._write_routed_wgrad_tile(
                runtime,
                tiled_mma_atb,
                atb_epi_tile,
                thread,
                warp,
                cta_rank,
                kind,
                generation,
                row,
                column,
                expert,
                expert_counts,
                atb_d_atoms[4],
                atb_d_tensors[4],
                atb_d_add_atoms[1],
                atb_d_add_tensors[1],
                sD_atb,
                acc_tmem_ptr,
                acc_pipeline,
                acc_consumer_state,
                replayed_gate_up_ready,
                wgrad_segment_done,
                wgrad_first,
                wgrad_has_next,
                d_desc_ptr=self._main_grad_descriptor(routed_descriptors, num_experts_rt, 6, expert),
                d_row_tile_offset=None,
            )
        else:
            acc_consumer_state = self._write_routed_wgrad_tile(
                runtime,
                tiled_mma_atb,
                atb_epi_tile,
                thread,
                warp,
                cta_rank,
                kind,
                generation,
                row,
                column,
                expert,
                expert_counts,
                atb_d_atoms[5],
                atb_d_tensors[5],
                atb_d_add_atoms[2],
                atb_d_add_tensors[2],
                sD_atb,
                acc_tmem_ptr,
                acc_pipeline,
                acc_consumer_state,
                replayed_gate_up_ready,
                wgrad_segment_done,
                wgrad_first,
                wgrad_has_next,
                d_desc_ptr=self._main_grad_descriptor(routed_descriptors, num_experts_rt, 7, expert),
                d_row_tile_offset=(main_grad_up_row_offset // Int32(MLP_TILE_M // CLUSTER_SIZE) if self.accumulate_wgrad else None),
            )
        return acc_consumer_state

    @cute.jit
    def _run_epilogue_gemm_task(
        self,
        runtime: tuple,
        geometry: tuple,
        tiled_mma_abt: cute.TiledMma,
        tiled_mma_atb: cute.TiledMma,
        tiled_mma_replay: cute.TiledMma,
        abt_epi_tile: cute.Tile,
        atb_epi_tile: cute.Tile,
        replay_epi_tile: cute.Tile,
        thread: Int32,
        warp: Int32,
        cta_rank: Int32,
        kind: Int32,
        generation: Int32,
        num_tokens: Int32,
        expert_counts: cute.Tensor,
        abt_d_atoms,
        abt_d_tensors,
        swap_d_atom: cute.CopyAtom,
        swap_d_tensor: cute.Tensor,
        replay_d_atoms,
        replay_d_tensors,
        atb_d_atoms,
        atb_d_tensors,
        atb_d_add_atoms,
        atb_d_add_tensors,
        sD_abt: cute.Tensor,
        sD_replay: cute.Tensor,
        sD_atb: cute.Tensor,
        sDT: cute.Tensor,
        sCT: cute.Tensor,
        acc_tmem_ptr,
        acc_pipeline,
        acc_consumer_state,
        paired_acc_pipeline,
        paired_acc_consumer_state,
        d_x_routed_ready: cute.Tensor,
        replayed_gate_up_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        wgrad_segment_done: cute.Tensor,
        num_local_tokens: Int32,
        d_w_routed_down_hie: cute.Tensor,
        d_w_routed_gate_ihe: cute.Tensor,
        d_w_routed_up_ihe: cute.Tensor,
        fused_swiglu: tuple,
        routed_descriptors=None,
        num_experts_rt: Int32 = Int32(0),
        main_grad_up_row_offset=None,
    ):
        valid, expert, row, column, k_begin, k_tiles, wgrad_first, wgrad_has_next, wgrad_empty, tile_valid = geometry
        if valid != Int32(0):
            if kind == Int32(SHARED_DOWN_DGRAD):
                if cutlass.const_expr(self.has_tail):
                    # One branch per task, the same for the whole CTA: the tail tile (the last I column block
                    # when I % 256 != 0) runs the instantiation for its TAIL_WIDTH valid columns, every full
                    # tile the one for all 256.
                    if runtime[_RT_S_TILES] * Int32(SWIGLU_TILE_N) < (column + Int32(1)) * Int32(MLP_TILE_N):
                        acc_consumer_state = self._run_down_swiglu_epilogue(
                            runtime,
                            self.abt,
                            tiled_mma_abt,
                            abt_epi_tile,
                            thread,
                            warp,
                            cta_rank,
                            row,
                            column,
                            acc_tmem_ptr,
                            acc_pipeline,
                            acc_consumer_state,
                            fused_swiglu,
                            tail=True,
                        )
                    else:
                        acc_consumer_state = self._run_down_swiglu_epilogue(
                            runtime,
                            self.abt,
                            tiled_mma_abt,
                            abt_epi_tile,
                            thread,
                            warp,
                            cta_rank,
                            row,
                            column,
                            acc_tmem_ptr,
                            acc_pipeline,
                            acc_consumer_state,
                            fused_swiglu,
                        )
                else:
                    acc_consumer_state = self._run_down_swiglu_epilogue(
                        runtime,
                        self.abt,
                        tiled_mma_abt,
                        abt_epi_tile,
                        thread,
                        warp,
                        cta_rank,
                        row,
                        column,
                        acc_tmem_ptr,
                        acc_pipeline,
                        acc_consumer_state,
                        fused_swiglu,
                    )
            elif kind == Int32(SHARED_GATE_UP_DGRAD):
                acc_consumer_state = self._run_one_epilogue_task(
                    runtime,
                    self.abt,
                    tiled_mma_abt,
                    abt_epi_tile,
                    thread,
                    warp,
                    cta_rank,
                    kind,
                    generation,
                    row,
                    column,
                    expert,
                    abt_d_atoms[0],
                    abt_d_tensors[0],
                    sD_abt,
                    acc_tmem_ptr,
                    acc_pipeline,
                    acc_consumer_state,
                    False,
                    False,
                    replayed_gate_up_ready,
                )
            elif kind == Int32(REPLAY_GATE_UP):
                paired_acc_consumer_state = self._run_one_epilogue_task(
                    runtime,
                    self.replay,
                    tiled_mma_replay,
                    replay_epi_tile,
                    thread,
                    warp,
                    cta_rank,
                    kind,
                    generation,
                    row,
                    column,
                    expert,
                    replay_d_atoms[0],
                    replay_d_tensors[0],
                    sD_replay,
                    acc_tmem_ptr,
                    paired_acc_pipeline,
                    paired_acc_consumer_state,
                    False,
                    False,
                    replayed_gate_up_ready,
                    acc_slot=0,
                    wait_for_acc=True,
                    acc_release_idx=MLP_EPILOGUE_PIECES,
                    manual_acc_release=False,
                    advance_acc_state=False,
                    replay_ready_increment=0,
                )
                paired_acc_consumer_state = self._run_one_epilogue_task(
                    runtime,
                    self.replay,
                    tiled_mma_replay,
                    replay_epi_tile,
                    thread,
                    warp,
                    cta_rank,
                    kind,
                    generation,
                    row,
                    column,
                    expert,
                    replay_d_atoms[1],
                    replay_d_tensors[1],
                    sD_replay,
                    acc_tmem_ptr,
                    paired_acc_pipeline,
                    paired_acc_consumer_state,
                    False,
                    False,
                    replayed_gate_up_ready,
                    acc_slot=1,
                    wait_for_acc=False,
                    acc_release_idx=MLP_EPILOGUE_PIECES,
                    manual_acc_release=True,
                    advance_acc_state=True,
                    replay_ready_increment=2,
                )
            elif kind == Int32(ROUTED_DOWN_DGRAD):
                if cutlass.const_expr(self.has_tail):
                    # One CTA-uniform branch per task.  The tail tile (the last I feature block when
                    # I % 256 != 0) holds TAIL_WIDTH features, one CTA's: the cluster's first CTA owns all of
                    # its 128 and runs the code of a full tile, the second owns none and runs the instantiation
                    # that only synchronises.
                    if cta_rank != Int32(0) and (runtime[_RT_S_TILES] * Int32(SWIGLU_TILE_N) < (column + Int32(1)) * Int32(MLP_TILE_N)):
                        acc_consumer_state = self._run_routed_down_swapab_epilogue(
                            runtime,
                            self.abt,
                            tiled_mma_abt,
                            abt_epi_tile,
                            thread,
                            warp,
                            cta_rank,
                            generation,
                            row,
                            column,
                            tile_valid,
                            sCT,
                            acc_tmem_ptr,
                            acc_pipeline,
                            acc_consumer_state,
                            fused_swiglu,
                            replayed_gate_up_ready,
                            routed_buffers_done,
                            num_tokens,
                            num_local_tokens,
                            tail=True,
                        )
                    else:
                        acc_consumer_state = self._run_routed_down_swapab_epilogue(
                            runtime,
                            self.abt,
                            tiled_mma_abt,
                            abt_epi_tile,
                            thread,
                            warp,
                            cta_rank,
                            generation,
                            row,
                            column,
                            tile_valid,
                            sCT,
                            acc_tmem_ptr,
                            acc_pipeline,
                            acc_consumer_state,
                            fused_swiglu,
                            replayed_gate_up_ready,
                            routed_buffers_done,
                            num_tokens,
                            num_local_tokens,
                        )
                else:
                    acc_consumer_state = self._run_routed_down_swapab_epilogue(
                        runtime,
                        self.abt,
                        tiled_mma_abt,
                        abt_epi_tile,
                        thread,
                        warp,
                        cta_rank,
                        generation,
                        row,
                        column,
                        tile_valid,
                        sCT,
                        acc_tmem_ptr,
                        acc_pipeline,
                        acc_consumer_state,
                        fused_swiglu,
                        replayed_gate_up_ready,
                        routed_buffers_done,
                        num_tokens,
                        num_local_tokens,
                    )
            elif kind == Int32(ROUTED_GATE_UP_DGRAD):
                acc_consumer_state = self._run_routed_dx_swapab_epilogue(
                    runtime,
                    self.abt,
                    tiled_mma_abt,
                    abt_epi_tile,
                    thread,
                    warp,
                    cta_rank,
                    generation,
                    row,
                    column,
                    tile_valid,
                    swap_d_atom,
                    swap_d_tensor,
                    sDT,
                    acc_tmem_ptr,
                    acc_pipeline,
                    acc_consumer_state,
                    d_x_routed_ready,
                    routed_buffers_done,
                    num_tokens,
                )
            elif kind == Int32(SHARED_WGRAD_DOWN):
                # Accumulating: the shared weight gradient is added into main_grad, like the
                # routed ones, so it takes the reduce-add atom over the same tensor.
                shared_wgrad_atom = atb_d_atoms[0]
                shared_wgrad_tensor = atb_d_tensors[0]
                if cutlass.const_expr(self.accumulate_wgrad):
                    shared_wgrad_atom = atb_d_add_atoms[0 + 3]
                    shared_wgrad_tensor = atb_d_add_tensors[0 + 3]
                acc_consumer_state = self._run_one_epilogue_task(
                    runtime,
                    self.atb,
                    tiled_mma_atb,
                    atb_epi_tile,
                    thread,
                    warp,
                    cta_rank,
                    kind,
                    generation,
                    row,
                    column,
                    expert,
                    shared_wgrad_atom,
                    shared_wgrad_tensor,
                    sD_atb,
                    acc_tmem_ptr,
                    acc_pipeline,
                    acc_consumer_state,
                    False,
                    cutlass.const_expr(self.accumulate_wgrad),
                    replayed_gate_up_ready,
                )
            elif kind == Int32(SHARED_WGRAD_GATE):
                # Accumulating: the shared weight gradient is added into main_grad, like the
                # routed ones, so it takes the reduce-add atom over the same tensor.
                shared_wgrad_atom = atb_d_atoms[1]
                shared_wgrad_tensor = atb_d_tensors[1]
                if cutlass.const_expr(self.accumulate_wgrad):
                    shared_wgrad_atom = atb_d_add_atoms[1 + 3]
                    shared_wgrad_tensor = atb_d_add_tensors[1 + 3]
                acc_consumer_state = self._run_one_epilogue_task(
                    runtime,
                    self.atb,
                    tiled_mma_atb,
                    atb_epi_tile,
                    thread,
                    warp,
                    cta_rank,
                    kind,
                    generation,
                    row,
                    column,
                    expert,
                    shared_wgrad_atom,
                    shared_wgrad_tensor,
                    sD_atb,
                    acc_tmem_ptr,
                    acc_pipeline,
                    acc_consumer_state,
                    False,
                    cutlass.const_expr(self.accumulate_wgrad),
                    replayed_gate_up_ready,
                )
            elif kind == Int32(SHARED_WGRAD_UP):
                # Accumulating: the shared weight gradient is added into main_grad, like the
                # routed ones, so it takes the reduce-add atom over the same tensor.
                shared_wgrad_atom = atb_d_atoms[2]
                shared_wgrad_tensor = atb_d_tensors[2]
                if cutlass.const_expr(self.accumulate_wgrad):
                    shared_wgrad_atom = atb_d_add_atoms[2 + 3]
                    shared_wgrad_tensor = atb_d_add_tensors[2 + 3]
                acc_consumer_state = self._run_one_epilogue_task(
                    runtime,
                    self.atb,
                    tiled_mma_atb,
                    atb_epi_tile,
                    thread,
                    warp,
                    cta_rank,
                    kind,
                    generation,
                    row,
                    column,
                    expert,
                    shared_wgrad_atom,
                    shared_wgrad_tensor,
                    sD_atb,
                    acc_tmem_ptr,
                    acc_pipeline,
                    acc_consumer_state,
                    False,
                    cutlass.const_expr(self.accumulate_wgrad),
                    replayed_gate_up_ready,
                )
            elif kind >= Int32(ROUTED_WGRAD_DOWN):
                acc_consumer_state = self._run_routed_wgrad_epilogue(
                    runtime,
                    tiled_mma_atb,
                    atb_epi_tile,
                    thread,
                    warp,
                    cta_rank,
                    kind,
                    generation,
                    row,
                    column,
                    expert,
                    expert_counts,
                    atb_d_atoms,
                    atb_d_tensors,
                    atb_d_add_atoms,
                    atb_d_add_tensors,
                    sD_atb,
                    acc_tmem_ptr,
                    acc_pipeline,
                    acc_consumer_state,
                    replayed_gate_up_ready,
                    wgrad_segment_done,
                    d_w_routed_down_hie,
                    d_w_routed_gate_ihe,
                    d_w_routed_up_ihe,
                    wgrad_first,
                    wgrad_has_next,
                    routed_descriptors=routed_descriptors,
                    num_experts_rt=num_experts_rt,
                    main_grad_up_row_offset=main_grad_up_row_offset,
                )
        elif kind >= Int32(ROUTED_WGRAD_DOWN):
            # No K range: an expert without rows on this rank (registered once, in
            # macrobatch 0, by the metadata builder) gets its dW tile written as zeros
            # here instead of by an external zero kernel.  Experts merely absent from
            # this macrobatch (dense single-macrobatch grid) are skipped.
            #
            # Accumulating into main_grad skips it entirely: an expert with no rows contributes
            # nothing, and writing zeros would erase the earlier microbatches' gradients instead.
            # Its own statement, not an `and`: the DSL rewrites `if` from the AST and would
            # trace a dynamic condition's body even when the compile-time half is False.
            if cutlass.const_expr(not self.accumulate_wgrad):
                if wgrad_empty != Int32(0) and generation < runtime[_RT_MINIBATCHES_PER_MACROBATCH]:
                    acc_consumer_state = self._run_routed_wgrad_epilogue(
                        runtime,
                        tiled_mma_atb,
                        atb_epi_tile,
                        thread,
                        warp,
                        cta_rank,
                        kind,
                        generation,
                        row,
                        column,
                        expert,
                        expert_counts,
                        atb_d_atoms,
                        atb_d_tensors,
                        atb_d_add_atoms,
                        atb_d_add_tensors,
                        sD_atb,
                        acc_tmem_ptr,
                        acc_pipeline,
                        acc_consumer_state,
                        replayed_gate_up_ready,
                        wgrad_segment_done,
                        d_w_routed_down_hie,
                        d_w_routed_gate_ihe,
                        d_w_routed_up_ihe,
                        zero_fill=True,
                        main_grad_up_row_offset=main_grad_up_row_offset,
                    )
        return acc_consumer_state, paired_acc_consumer_state

    @cute.jit
    def _drain_clc_capacity_tail(
        self,
        runtime: tuple,
        needs_drain: Int32,
        clc_drain_pipeline,
        drain_consumer_state,
        drain_producer_state,
        drain_scheduler,
        cta_rank: Int32,
    ) -> None:
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        # Drain, lane 0 per warp: each CTA arms its own full barrier, rank 0 issues the
        # CLC query, both wait the multicast response, test .success only, release rank 0's
        # empty barrier; rank 0 waits it before re-issuing. Tight spins (latency-bound).
        if needs_drain != Int32(0):
            if cute.arch.lane_idx() == Int32(0):
                full_bar = clc_drain_pipeline.sync_object_full.get_barrier(drain_consumer_state.index)
                empty_bar = clc_drain_pipeline.sync_object_empty.get_barrier(drain_producer_state.index)
                response = drain_scheduler._clc_response_ptr
                full_phase = drain_consumer_state.phase
                empty_phase = Int32(0)
                cute.arch.mbarrier_arrive_and_expect_tx(full_bar, Int32(CLC_RESPONSE_BYTES))
                if cta_rank == Int32(0):
                    cute.arch.issue_clc_query(full_bar, response)
                active = Int32(1)
                while active != Int32(0):
                    _mbar_spin_wait(full_bar, full_phase)
                    full_phase = full_phase ^ Int32(1)
                    _, _, _, drained_valid = cute.arch.clc_response(response)
                    # The generic-proxy read of the response must be ordered before the next
                    # async-proxy CLC write into the same shared memory.
                    cute.arch.fence_proxy("async.shared", space="cta")
                    clc_drain_pipeline.sync_object_empty.arrive(drain_consumer_state.index, clc_drain_pipeline.consumer_mask)
                    if cta_rank == Int32(0):
                        _mbar_spin_wait(empty_bar, empty_phase)
                        empty_phase = empty_phase ^ Int32(1)
                    active = Int32(drained_valid)
                    if active != Int32(0):
                        cute.arch.mbarrier_arrive_and_expect_tx(full_bar, Int32(CLC_RESPONSE_BYTES))
                        if cta_rank == Int32(0):
                            cute.arch.issue_clc_query(full_bar, response)
            cute.arch.sync_warp()

    @cute.kernel
    def kernel(
        self,
        runtime: tuple,
        tiled_mma_abt: cute.TiledMma,
        tiled_mma_atb: cute.TiledMma,
        tiled_mma_replay: cute.TiledMma,
        abt_a_atoms: list[cute.CopyAtom],
        abt_a_tensors: list[cute.Tensor],
        abt_b_atoms: list[cute.CopyAtom],
        abt_b_tensors: list[cute.Tensor],
        swap_a_atoms: list[cute.CopyAtom],
        swap_a_tensors: list[cute.Tensor],
        swap_b_atoms: list[cute.CopyAtom],
        swap_b_tensors: list[cute.Tensor],
        replay_a_atoms: list[cute.CopyAtom],
        replay_a_tensors: list[cute.Tensor],
        replay_b_atoms: list[cute.CopyAtom],
        replay_b_tensors: list[cute.Tensor],
        atb_a_atoms: list[cute.CopyAtom],
        atb_a_tensors: list[cute.Tensor],
        atb_b_atoms: list[cute.CopyAtom],
        atb_b_tensors: list[cute.Tensor],
        abt_d_atoms: list[cute.CopyAtom],
        abt_d_tensors: list[cute.Tensor],
        swap_d_atom: cute.CopyAtom,
        swap_d_tensor: cute.Tensor,
        replay_d_atoms: list[cute.CopyAtom],
        replay_d_tensors: list[cute.Tensor],
        atb_d_atoms: list[cute.CopyAtom],
        atb_d_tensors: list[cute.Tensor],
        atb_d_add_atoms: list[cute.CopyAtom],
        atb_d_add_tensors: list[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        abt_a_layout: cute.ComposedLayout,
        abt_b_layout: cute.ComposedLayout,
        abt_epi_layout: cute.ComposedLayout,
        abt_epi_tile: cute.Tile,
        replay_a_layout: cute.ComposedLayout,
        replay_b_layout: cute.ComposedLayout,
        replay_epi_layout: cute.ComposedLayout,
        replay_epi_tile: cute.Tile,
        atb_a_layout: cute.ComposedLayout,
        atb_b_layout: cute.ComposedLayout,
        atb_epi_layout: cute.ComposedLayout,
        atb_epi_tile: cute.Tile,
        x_peer_table: cute.Tensor,
        d_y_peer_table: cute.Tensor,
        d_x_peer_table: cute.Tensor,
        router_peer_table: cute.Tensor,
        d_router_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens_tensor: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        minibatch_expert_offsets: cute.Tensor,
        completed_segment_offsets: cute.Tensor,
        completed_segment_experts: cute.Tensor,
        task_owner_buckets: cute.Tensor,
        x_shared: cute.Tensor,
        x_routed: cute.Tensor,
        gate_shared: cute.Tensor,
        gate_routed: cute.Tensor,
        up_shared: cute.Tensor,
        up_routed: cute.Tensor,
        hidden_shared: cute.Tensor,
        hidden_routed: cute.Tensor,
        d_y_shared: cute.Tensor,
        d_x_shared: cute.Tensor,
        d_x_routed: cute.Tensor,
        d_gate_shared: cute.Tensor,
        d_gate_routed: cute.Tensor,
        d_up_shared: cute.Tensor,
        d_up_routed: cute.Tensor,
        d_y_routed: cute.Tensor,
        d_w_shared_down: cute.Tensor,
        d_w_shared_gate: cute.Tensor,
        d_w_shared_up: cute.Tensor,
        d_w_routed_down_hie: cute.Tensor,
        d_w_routed_gate_ihe: cute.Tensor,
        d_w_routed_up_ihe: cute.Tensor,
        router_weights: cute.Tensor,
        d_router_weight_partials: cute.Tensor,
        d_y_routed_ready: cute.Tensor,
        d_gate_up_ready: cute.Tensor,
        d_x_routed_ready: cute.Tensor,
        replayed_x_routed_ready: cute.Tensor,
        replayed_gate_up_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        wgrad_read_consumed: cute.Tensor,
        wgrad_segment_done: cute.Tensor,
        reverse_dispatch_done: cute.Tensor,
        scheduler_params: utils.ClcDynamicPersistentTileSchedulerParams,
        comm_clusters: Int32,
        shared_down_tasks: Int32,
        shared_gate_up_tasks: Int32,
        shared_tasks: Int32,
        routed_down_tasks: Int32,
        routed_gate_up_tasks: Int32,
        routed_tasks: Int32,
        num_local_tokens: Int32,
        minibatch_size: Int32,
        num_comm_sms: Int32,
        # Where Up starts in the tensor passed as the routed Up weights, which the tile coordinate carries: the
        # rows of Gate when Gate and Up are one combined FC1 parameter, 0 when they are separate weights.
        up_row_offset: Int32,
        # Variant-only arguments, None in every other variant -- hence no annotation: the DSL casts an
        # annotated argument, and None is what keeps them out of the kernel's parameters.
        # routed_descriptors (split_weights): the per-expert TMA descriptors of the routed weights, one
        # array per operand in the address table's order.  main_grad_up_row_offset (accumulate_wgrad).
        # shared_hidden_size (has_tail).  situ_beta1 / situ_beta2 (is_situ).
        routed_descriptors,
        main_grad_up_row_offset,
        shared_hidden_size,
        situ_beta1,
        situ_beta2,
    ) -> None:
        minibatches_per_macrobatch = runtime[_RT_MINIBATCHES_PER_MACROBATCH]
        replay_tasks = runtime[_RT_REPLAY_TASKS]
        routed_wgrad_tiles = runtime[_RT_ROUTED_WGRAD_TILES]
        _ = x_shared
        _ = x_routed
        _ = hidden_shared
        _ = d_y_shared
        _ = d_w_shared_down
        _ = d_w_shared_gate
        _ = d_w_shared_up
        thread = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_index = cute.arch.make_warp_uniform(cute.arch.cluster_idx()[0])
        num_tokens = Int32(num_tokens_tensor[0])
        num_experts_rt = Int32(tokens_per_expert.shape[0])
        runtime = (*runtime, num_tokens, num_experts_rt)
        if cutlass.const_expr(self.split_weights):
            # Once per CTA, before any task: the routed descriptors are built by the launch that
            # precedes this kernel and do not change while it runs, so the loads only have to name
            # one.  Doing it here keeps an inline-asm fence out of the mainloop.
            # Five weight operands the dgrad reads, then -- when they are split too -- the
            # three main_grad buffers it writes.
            for operand in cutlass.range_constexpr(8 if self.split_main_grads else 5):
                routed_descriptors_pkg.acquire_descriptors(
                    routed_descriptors_pkg.operand_descriptors(routed_descriptors, num_experts_rt, operand), num_experts_rt
                )
        true_minibatches = _ceil_div_i32(num_tokens, Int32(minibatch_size))
        replay_minibatches = cutlass.max(Int32(0), true_minibatches - Int32(self._saved_minibatches(minibatches_per_macrobatch)))
        completed_segments = Int32(completed_segment_offsets[true_minibatches])
        true_compute_tasks = (
            Int32(shared_tasks)
            + true_minibatches * Int32(routed_tasks)
            + replay_minibatches * Int32(replay_tasks)
            + completed_segments * Int32(3) * routed_wgrad_tiles
        )
        if num_tokens <= self._dense_wgrad_limit(runtime[_RT_MACROBATCH_SIZE]):
            # Dense per-expert Wgrad grid, also with no routed rows at all: every expert
            # is then empty and its Wgrad tasks write the zero dW tiles.
            true_compute_tasks = Int32(shared_tasks) + true_minibatches * Int32(routed_tasks) + Int32(3) * runtime[_RT_NUM_EXPERTS] * routed_wgrad_tiles
        true_clusters = Int32(comm_clusters) + true_compute_tasks

        smem = utils.SmemAllocator()
        storage = smem.allocate(_PersistentBwdSharedStorage)
        clc_mbarriers_ptr = storage.clc_mbarriers.data_ptr()
        clc_response_ptr = storage.clc_response.data_ptr()
        clc_drain_mbarriers_ptr = storage.clc_drain_mbarriers.data_ptr()
        clc_drain_responses_ptr = storage.clc_drain_responses.data_ptr()
        gemm_ab_mbarriers_ptr = storage.gemm_ab_mbarriers.data_ptr()
        gemm_acc_mbarriers_ptr = storage.gemm_acc_mbarriers.data_ptr()
        tmem_dealloc_mbarrier_ptr = storage.tmem_dealloc_mbarrier.ptr
        tmem_holding_buffer_ptr = storage.tmem_holding_buffer.ptr
        comm_mbarriers_ptr = storage.comm_mbarriers.data_ptr()
        paired_gate_up_mbarriers_ptr = storage.paired_gate_up_mbarriers.data_ptr()
        task_info_mbarriers_ptr = storage.task_info_mbarriers.data_ptr()
        task_info_ptr = storage.task_info.data_ptr()
        s_task_taken = cute.make_tensor(storage.task_info_taken.data_ptr(), cute.make_layout(WARPS_PER_CTA))
        expert_counts = tokens_per_expert
        arena = storage.arena.data_ptr().align(min_align=1024)
        # Per-CTA shared expert table for the compile-time decode method (no-op for LINEAR_SCAN).
        s_expert_valid_end = cute.make_tensor(cute.recast_ptr(arena + EXPERT_TABLE_ARENA_OFFSET, dtype=Int32), cute.make_layout(BWD_EXPERT_TABLE_CAP))
        expert_stride, expert_groups = _build_expert_valid_end(tokens_per_expert, schedule_peer_rank, s_expert_valid_end, self.task_decode, thread)
        ep_size = Int32(x_peer_table.shape[0])
        # Operands of the routed and shared SwiGLU-backward fused into the Down dgrad epilogues.
        fused_swiglu = (
            gate_routed,
            up_routed,
            d_gate_routed,
            d_up_routed,
            hidden_routed,
            router_weights,
            schedule_peer_rank,
            d_router_weight_partials,
            d_gate_up_ready,
            d_y_routed_ready,
            gate_shared,
            up_shared,
            d_gate_shared,
            d_up_shared,
        )
        runtime = (
            *runtime,
            s_expert_valid_end,
            expert_stride,
            expert_groups,
            ep_size,
            reverse_dispatch_done,
            schedule_peer_rank,
            shared_hidden_size,
            situ_beta1,
            situ_beta2,
        )
        if cluster_index < Int32(comm_clusters):
            self._run_comm_prefix(
                runtime,
                x_peer_table,
                d_y_peer_table,
                d_x_peer_table,
                router_peer_table,
                d_router_peer_table,
                schedule_peer_rank,
                schedule_peer_token_idx,
                num_tokens,
                router_weights,
                x_routed,
                d_y_routed,
                d_x_routed,
                d_router_weight_partials,
                minibatch_expert_offsets,
                d_y_routed_ready,
                d_x_routed_ready,
                replayed_x_routed_ready,
                routed_buffers_done,
                wgrad_read_consumed,
                comm_mbarriers_ptr,
                arena,
                cluster_index,
                cta_rank,
                num_comm_sms,
            )
        else:
            # CuTe DSL forbids a runtime-region early return.  A real dynamic else keeps the
            # comm/compute split strict without letting communication CTAs construct CLC, TMEM,
            # AB or accumulator pipeline state.
            role_warp = Int32(WARPS_PER_CTA)
            if cluster_index >= Int32(comm_clusters):
                if cluster_index < true_clusters:
                    role_warp = warp

            sA_abt = cute.make_tensor(
                cute.recast_ptr(arena, abt_a_layout.inner, dtype=BFloat16),
                abt_a_layout.outer,
            )
            sB_abt = cute.make_tensor(
                cute.recast_ptr(arena + GEMM_A_BYTES, abt_b_layout.inner, dtype=BFloat16),
                abt_b_layout.outer,
            )
            sD_abt = cute.make_tensor(
                cute.recast_ptr(
                    arena + GEMM_A_BYTES + GEMM_B_BYTES,
                    abt_epi_layout.inner,
                    dtype=BFloat16,
                ),
                abt_epi_layout.outer,
            )
            # The Replay collective is layout-distinct but storage-identical to
            # ABt.  These views deliberately alias the one compute arena.
            sA_replay = cute.make_tensor(
                cute.recast_ptr(arena, replay_a_layout.inner, dtype=BFloat16),
                replay_a_layout.outer,
            )
            sB_replay = cute.make_tensor(
                cute.recast_ptr(
                    arena + GEMM_A_BYTES,
                    replay_b_layout.inner,
                    dtype=BFloat16,
                ),
                replay_b_layout.outer,
            )
            sD_replay = cute.make_tensor(
                cute.recast_ptr(
                    arena + GEMM_A_BYTES + GEMM_B_BYTES,
                    replay_epi_layout.inner,
                    dtype=BFloat16,
                ),
                replay_epi_layout.outer,
            )
            sA_atb = cute.make_tensor(
                cute.recast_ptr(arena, atb_a_layout.inner, dtype=BFloat16),
                atb_a_layout.outer,
            )
            sB_atb = cute.make_tensor(
                cute.recast_ptr(arena + GEMM_A_BYTES, atb_b_layout.inner, dtype=BFloat16),
                atb_b_layout.outer,
            )
            sD_atb = cute.make_tensor(
                cute.recast_ptr(
                    arena + GEMM_A_BYTES + GEMM_B_BYTES,
                    atb_epi_layout.inner,
                    dtype=Float32 if cutlass.const_expr(self.fp32_main_grad) else BFloat16,
                ),
                atb_epi_layout.outer,
            )
            # Routed dgrad, swap-AB views of the same arena: the two MN-major A slabs of every A stage
            # (the sA swizzle), the ring bases of the raw MMA descriptors, and the Gate/Up dX's transposed
            # staging in the D stages (_swapab_fwd.d_smem_layout); the B stages are sB_replay (K-major).
            swap_slab_stages = cute.make_layout(
                (_SWAP_SLAB_ROWS, MLP_TILE_K, MLP_LOAD_STAGES),
                stride=(1, _SWAP_SLAB_ROWS, (MLP_TILE_M // CLUSTER_SIZE) * MLP_TILE_K),
            )
            sA_swap0 = cute.make_tensor(cute.recast_ptr(arena, abt_a_layout.inner, dtype=BFloat16), swap_slab_stages)
            sA_swap1 = cute.make_tensor(cute.recast_ptr(arena + _SWAP_SLAB_BYTES, abt_a_layout.inner, dtype=BFloat16), swap_slab_stages)
            a_ring_base = Int32(arena.toint())
            b_ring_base = a_ring_base + Int32(GEMM_A_BYTES)
            swap_d_layout = _swapab_fwd.d_smem_layout(abt_epi_tile, MLP_BF16_D_TILES)
            assert cute.cosize(swap_d_layout) * 2 == GEMM_D_BYTES
            sDT = cute.make_tensor(
                cute.recast_ptr(arena + GEMM_A_BYTES + GEMM_B_BYTES, swap_d_layout.inner, dtype=BFloat16),
                swap_d_layout.outer,
            )
            # Routed Down dgrad, swap-AB: the transposed BF16 dH staging of the fused SwiGLU-backward in the
            # same D stages, (32 tokens, 128 features + 16 B pad) rows, two tiles alternating.
            sCT = cute.make_tensor(
                cute.recast_ptr(arena + GEMM_A_BYTES + GEMM_B_BYTES, dtype=BFloat16),
                cute.make_layout(
                    (_SWAP_EPI_ROWS, _SWAP_DH_ROW_STRIDE, _SWAP_DH_STAGING_TILES), stride=(_SWAP_DH_ROW_STRIDE, 1, _SWAP_EPI_ROWS * _SWAP_DH_ROW_STRIDE)
                ),
            )
            ab_pipeline = PipelineTmaUmma.create(
                num_stages=MLP_LOAD_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.abt.num_mcast_ctas_a + self.abt.num_mcast_ctas_b - 1,
                ),
                tx_count=self.abt.num_tma_load_bytes,
                barrier_storage=gemm_ab_mbarriers_ptr,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            paired_gate_up_pipeline = PipelineTmaUmma.create(
                num_stages=PAIRED_GATE_UP_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.replay.num_mcast_ctas_a + self.replay.num_mcast_ctas_b - 1,
                ),
                tx_count=PAIRED_GATE_UP_TRANSACTION_BYTES,
                barrier_storage=paired_gate_up_mbarriers_ptr,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            acc_pipeline = PipelineUmmaAsync.create(
                num_stages=MLP_ACC_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4 * CLUSTER_SIZE),
                barrier_storage=gemm_acc_mbarriers_ptr,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
                # QuACK 0.5's epilogue owns the accumulator-release election.
            )
            paired_acc_pipeline = PipelineUmmaAsync.create(
                num_stages=PAIRED_ACC_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4 * CLUSTER_SIZE),
                barrier_storage=(paired_gate_up_mbarriers_ptr + 2 * PAIRED_GATE_UP_STAGES),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            tmem_barrier = pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierGemm.TmemPtr),
                num_threads=5 * cute.arch.WARP_SIZE,
            )
            tmem = utils.TmemAllocator(
                tmem_holding_buffer_ptr,
                barrier_for_retrieve=tmem_barrier,
                allocator_warp_id=0,
                is_two_cta=True,
                two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbarrier_ptr,
            )
            if role_warp <= Int32(EPILOGUE_WARPS[-1]):
                tmem.allocate(self.abt.num_tmem_alloc_cols)
            acc_tmem_ptr = cute.make_ptr(Float32, 0, cute.AddressSpace.tmem, assumed_align=16)
            if role_warp <= Int32(UMMA_WARP):
                # The epilogue and MMA warps only, so unaligned and on the barrier itself (tmem.wait_for_alloc()
                # is the aligned bar.sync): the allocation is visible to all five.
                named_barrier_sync_unaligned(tmem_barrier)
                acc_tmem_ptr = tmem.retrieve_ptr(Float32)

            clc_layout = cute.make_layout((1, CLUSTER_SIZE, 1, 1))
            clc_pipeline = pipeline.PipelineClcFetchAsync.create(
                barrier_storage=clc_mbarriers_ptr,
                num_stages=CLC_DEPTH,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, CLC_COMPLETION_WARPS),
                tx_count=CLC_RESPONSE_BYTES,
                cta_layout_vmnk=clc_layout,
                defer_sync=True,
            )
            clc_drain_pipeline = pipeline.PipelineClcFetchAsync.create(
                barrier_storage=clc_drain_mbarriers_ptr,
                num_stages=CLC_DRAIN_DEPTH,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, CLUSTER_SIZE),
                tx_count=CLC_RESPONSE_BYTES,
                cta_layout_vmnk=clc_layout,
                defer_sync=True,
            )
            # Decoded-task records: producer = the CLC warp (one elected thread writes and commits),
            # consumers = the other seven warps (one elected arrival each).  Per CTA, not
            # cluster-wide: each CTA's CLC warp receives the multicast CLC response and decodes the
            # same work itself (nothing in the record depends on the CTA rank).
            task_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=task_info_mbarriers_ptr,
                num_stages=TASK_INFO_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, TASK_INFO_CONSUMER_WARPS),
                defer_sync=True,
            )
            s_task_info = cute.make_tensor(task_info_ptr, cute.make_layout(TASK_INFO_STAGES * TASK_INFO_WORDS))
            drain_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, CLC_DRAIN_DEPTH)
            drain_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.ProducerConsumer, CLC_DRAIN_DEPTH)
            drain_stage = Int32(0)
            while drain_stage < warp:
                drain_consumer_state.advance()
                drain_producer_state.advance()
                drain_stage = drain_stage + Int32(1)
            drain_response = (clc_drain_responses_ptr + warp * Int32(4)).align(min_align=16)
            drain_scheduler = utils.ClcDynamicPersistentTileScheduler.create(
                scheduler_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                drain_response,
            )
            pipeline_init_arrive(cluster_shape_mn=CLUSTER_SHAPE, is_relaxed=True)
            pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE)

            scheduler = utils.ClcDynamicPersistentTileScheduler.create(
                scheduler_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work = scheduler.initial_work_tile_info()
            clc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, CLC_DEPTH)
            clc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.ProducerConsumer, CLC_DEPTH)
            task_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, TASK_INFO_STAGES)
            task_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, TASK_INFO_STAGES)
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, MLP_LOAD_STAGES)
            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, MLP_LOAD_STAGES)
            paired_ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, PAIRED_GATE_UP_STAGES)
            paired_ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, PAIRED_GATE_UP_STAGES)
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, MLP_ACC_STAGES)
            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, MLP_ACC_STAGES)
            paired_acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, PAIRED_ACC_STAGES)
            paired_acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, PAIRED_ACC_STAGES)
            generic_ab_live = Int32(0)
            paired_ab_live = Int32(0)
            generic_acc_live = Int32(0)
            paired_acc_live = Int32(0)
            mma_op_abt = tiled_mma_abt
            mma_op_atb = tiled_mma_atb
            mma_op_replay = tiled_mma_replay

            logical_cluster = work.tile_idx[0] // Int32(CLUSTER_SIZE)
            work_valid = Int32(work.is_valid_tile)
            # CLC warp: whether the current work item came from a CLC response that is still to be
            # released (the initial item comes from the block index, not from a response).
            response_pending = Int32(0)
            # Owner cache of the CLC warp, the only warp that decodes: padded token segment of the
            # last routed expert it decoded (empty at start); see _runtime_expert_owner_rt.
            owner_cache_expert = Int32(0)
            owner_cache_begin = Int32(0)
            owner_cache_end = Int32(0)
            # The record of the current task (TASK_INFO_FIELDS): decoded by the CLC warp, read back
            # by every other warp.  The loop runs until a stop record (active = 0), whose
            # needs_drain carries the CLC capacity-tail state for the drain.
            active = Int32(1)
            needs_drain = Int32(0)
            kind = Int32(-1)
            generation = Int32(-1)
            valid = Int32(0)
            expert = Int32(0)
            row = Int32(0)
            column = Int32(0)
            k_begin = Int32(0)
            k_tiles = Int32(0)
            wgrad_first = Int32(0)
            wgrad_has_next = Int32(0)
            wgrad_empty = Int32(0)
            tile_valid = Int32(0)
            while active != Int32(0):
                # The physical warp index selects the scheduler warp: a capacity-tail CTA (every
                # role_warp = WARPS_PER_CTA) still needs its CLC warp to publish the stop record.
                if warp == Int32(CLC_WARP):
                    # Scheduler warp of this CTA: decode the work item once (the whole warp runs the
                    # decode uniformly) and publish the record.  The CLC
                    # query for the next item is issued only after every consumer has taken the
                    # current record and both CTAs have released the response (CLC depth 1: the
                    # cluster holds at most the task it runs plus one in flight): the request for
                    # task N + 1 goes out once every warp of both CTAs has started task N.
                    (
                        active,
                        needs_drain,
                        kind,
                        generation,
                        valid,
                        expert,
                        row,
                        column,
                        k_begin,
                        k_tiles,
                        wgrad_first,
                        wgrad_has_next,
                        wgrad_empty,
                        tile_valid,
                        owner_cache_expert,
                        owner_cache_begin,
                        owner_cache_end,
                    ) = self._decode_task_record(
                        runtime,
                        work_valid,
                        logical_cluster,
                        true_clusters,
                        comm_clusters,
                        completed_segment_offsets,
                        completed_segment_experts,
                        task_owner_buckets,
                        shared_down_tasks,
                        shared_gate_up_tasks,
                        shared_tasks,
                        routed_down_tasks,
                        routed_gate_up_tasks,
                        routed_tasks,
                        num_tokens,
                        tokens_per_expert,
                        num_local_tokens,
                        owner_cache_expert,
                        owner_cache_begin,
                        owner_cache_end,
                    )
                    task_pipeline.producer_acquire(task_producer_state)
                    with cute.arch.elect_one():
                        _store_task_info(
                            s_task_info,
                            task_producer_state.index,
                            active,
                            needs_drain,
                            kind,
                            generation,
                            valid,
                            expert,
                            row,
                            column,
                            k_begin,
                            k_tiles,
                            wgrad_first,
                            wgrad_has_next,
                            wgrad_empty,
                            tile_valid,
                        )
                        task_pipeline.producer_commit(task_producer_state)
                    task_producer_state.advance()
                    # Every consumer has taken this record (one stage: the phase the commit opened);
                    # for the final record this is also the pipeline's tail.
                    task_pipeline.producer_acquire(task_producer_state)
                    if response_pending != Int32(0):
                        with cute.arch.elect_one():
                            clc_pipeline.consumer_release(clc_consumer_state)
                        clc_consumer_state.advance()
                    if active != Int32(0):
                        if cta_rank == Int32(0):
                            clc_pipeline.producer_acquire(clc_producer_state)
                            scheduler.advance_to_next_work(clc_pipeline.producer_get_barrier(clc_producer_state))
                            clc_producer_state.advance()
                        clc_pipeline.consumer_wait(clc_consumer_state)
                        next_work = scheduler.get_current_work()
                        cute.arch.sync_warp()
                        logical_cluster = next_work.tile_idx[0] // Int32(CLUSTER_SIZE)
                        work_valid = Int32(next_work.is_valid_tile)
                        response_pending = Int32(1)
                else:
                    task_pipeline.consumer_wait(task_consumer_state)
                    (
                        active,
                        needs_drain,
                        kind,
                        generation,
                        valid,
                        expert,
                        row,
                        column,
                        k_begin,
                        k_tiles,
                        wgrad_first,
                        wgrad_has_next,
                        wgrad_empty,
                        tile_valid,
                    ) = _load_task_info(s_task_info, task_consumer_state.index, s_task_taken, warp)
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        task_pipeline.consumer_release(task_consumer_state)
                    task_consumer_state.advance()
                # Current task dispatch: the record's _gemm_geometry tuple, shared by every role's
                # runner.  A stop record runs no role body.
                geometry = (valid, expert, row, column, k_begin, k_tiles, wgrad_first, wgrad_has_next, wgrad_empty, tile_valid)
                if active != Int32(0):
                    if role_warp == Int32(AB_LOAD_WARP):
                        if kind == Int32(REPLAY_GATE_UP):
                            (
                                replay_valid,
                                replay_expert,
                                replay_row,
                                replay_column,
                                _,
                                replay_k_tiles,
                                _,
                                _,
                                _,
                                _,
                            ) = geometry
                            if replay_valid != Int32(0):
                                with cute.arch.elect_one():
                                    self._wait_gemm_inputs(
                                        runtime,
                                        kind,
                                        generation,
                                        replay_row,
                                        d_y_routed_ready,
                                        d_gate_up_ready,
                                        replayed_x_routed_ready,
                                        num_local_tokens,
                                    )
                                cute.arch.sync_warp()
                                if generic_ab_live != Int32(0):
                                    ab_drain_state = ab_producer_state.clone()
                                    for _ in cutlass.range_constexpr(ab_pipeline.num_stages - 1):
                                        ab_drain_state.advance()
                                    ab_pipeline.sync_object_empty.wait(
                                        ab_drain_state.index,
                                        ab_drain_state.phase,
                                    )
                                    generic_ab_live = Int32(0)
                                replay_gate_b, replay_gate_desc = self._routed_operand(
                                    replay_b_tensors[0], routed_descriptors, num_experts_rt, 3, replay_expert
                                )
                                replay_up_b, replay_up_desc = self._routed_operand(replay_b_tensors[1], routed_descriptors, num_experts_rt, 4, replay_expert)
                                paired_ab_producer_state = self._load_paired_replay_gate_up(
                                    runtime,
                                    tiled_mma_replay,
                                    replay_a_atoms[0],
                                    replay_a_tensors[0],
                                    replay_b_atoms[0],
                                    replay_gate_b,
                                    replay_b_atoms[1],
                                    replay_up_b,
                                    replay_row,
                                    replay_column,
                                    cta_rank,
                                    replay_k_tiles,
                                    sA_replay,
                                    sB_replay,
                                    paired_gate_up_pipeline,
                                    paired_ab_producer_state,
                                    gate_b_desc_ptr=replay_gate_desc,
                                    up_b_desc_ptr=replay_up_desc,
                                    up_row_offset=up_row_offset,
                                )
                                paired_ab_live = Int32(1)
                        else:
                            if paired_ab_live != Int32(0):
                                paired_ab_drain_state = paired_ab_producer_state.clone()
                                for _ in cutlass.range_constexpr(paired_gate_up_pipeline.num_stages - 1):
                                    paired_ab_drain_state.advance()
                                paired_gate_up_pipeline.sync_object_empty.wait(
                                    paired_ab_drain_state.index,
                                    paired_ab_drain_state.phase,
                                )
                                paired_ab_live = Int32(0)
                            ab_producer_state = self._run_tma_gemm_task(
                                runtime,
                                geometry,
                                kind,
                                generation,
                                cta_rank,
                                tiled_mma_abt,
                                tiled_mma_atb,
                                tiled_mma_replay,
                                abt_a_atoms,
                                abt_a_tensors,
                                abt_b_atoms,
                                abt_b_tensors,
                                swap_a_atoms,
                                swap_a_tensors,
                                swap_b_atoms,
                                swap_b_tensors,
                                atb_a_atoms,
                                atb_a_tensors,
                                atb_b_atoms,
                                atb_b_tensors,
                                sA_abt,
                                sB_abt,
                                sA_swap0,
                                sA_swap1,
                                sB_replay,
                                sA_atb,
                                sB_atb,
                                ab_pipeline,
                                ab_producer_state,
                                d_y_routed_ready,
                                d_gate_up_ready,
                                replayed_x_routed_ready,
                                num_local_tokens,
                                routed_descriptors,
                                num_experts_rt,
                                up_row_offset,
                            )
                            generic_ab_live = Int32(1)
                    elif role_warp == Int32(UMMA_WARP):
                        if kind == Int32(REPLAY_GATE_UP):
                            (
                                replay_valid,
                                _,
                                _,
                                _,
                                _,
                                replay_k_tiles,
                                _,
                                _,
                                _,
                                _,
                            ) = geometry
                            if replay_valid != Int32(0):
                                if generic_acc_live != Int32(0):
                                    if cta_rank == Int32(0):
                                        generic_acc_drain_state = acc_producer_state.clone()
                                        for _ in cutlass.range_constexpr(acc_pipeline.num_stages - 1):
                                            generic_acc_drain_state.advance()
                                        acc_pipeline.sync_object_empty.wait(
                                            generic_acc_drain_state.index,
                                            generic_acc_drain_state.phase,
                                        )
                                    generic_acc_live = Int32(0)
                                (
                                    paired_ab_consumer_state,
                                    paired_acc_producer_state,
                                    mma_op_replay,
                                ) = self._mma_paired_replay_gate_up(
                                    runtime,
                                    mma_op_replay,
                                    cta_rank,
                                    replay_k_tiles,
                                    sA_replay,
                                    sB_replay,
                                    acc_tmem_ptr,
                                    paired_gate_up_pipeline,
                                    paired_acc_pipeline,
                                    paired_ab_consumer_state,
                                    paired_acc_producer_state,
                                )
                                paired_acc_live = Int32(1)
                        else:
                            generic_valid, _, _, _, _, _, _, _, _, _ = geometry
                            if generic_valid != Int32(0) and paired_acc_live != Int32(0):
                                if cta_rank == Int32(0):
                                    paired_acc_drain_state = paired_acc_producer_state.clone()
                                    for _ in cutlass.range_constexpr(paired_acc_pipeline.num_stages - 1):
                                        paired_acc_drain_state.advance()
                                    paired_acc_pipeline.sync_object_empty.wait(
                                        paired_acc_drain_state.index,
                                        paired_acc_drain_state.phase,
                                    )
                                paired_acc_live = Int32(0)
                            (
                                ab_consumer_state,
                                acc_producer_state,
                                mma_op_abt,
                                mma_op_atb,
                            ) = self._run_umma_gemm_task(
                                runtime,
                                geometry,
                                kind,
                                generation,
                                cta_rank,
                                num_tokens,
                                sA_abt,
                                sB_abt,
                                sA_atb,
                                sB_atb,
                                a_ring_base,
                                b_ring_base,
                                acc_tmem_ptr,
                                ab_pipeline,
                                acc_pipeline,
                                ab_consumer_state,
                                acc_producer_state,
                                mma_op_abt,
                                mma_op_atb,
                                wgrad_read_consumed,
                            )
                            if generic_valid != Int32(0):
                                generic_acc_live = Int32(1)
                    elif role_warp <= Int32(EPILOGUE_WARPS[-1]):
                        (
                            acc_consumer_state,
                            paired_acc_consumer_state,
                        ) = self._run_epilogue_gemm_task(
                            runtime,
                            geometry,
                            tiled_mma_abt,
                            tiled_mma_atb,
                            tiled_mma_replay,
                            abt_epi_tile,
                            atb_epi_tile,
                            replay_epi_tile,
                            thread,
                            warp,
                            cta_rank,
                            kind,
                            generation,
                            num_tokens,
                            expert_counts,
                            abt_d_atoms,
                            abt_d_tensors,
                            swap_d_atom,
                            swap_d_tensor,
                            replay_d_atoms,
                            replay_d_tensors,
                            atb_d_atoms,
                            atb_d_tensors,
                            atb_d_add_atoms,
                            atb_d_add_tensors,
                            sD_abt,
                            sD_replay,
                            sD_atb,
                            sDT,
                            sCT,
                            acc_tmem_ptr,
                            acc_pipeline,
                            acc_consumer_state,
                            paired_acc_pipeline,
                            paired_acc_consumer_state,
                            d_x_routed_ready,
                            replayed_gate_up_ready,
                            routed_buffers_done,
                            wgrad_segment_done,
                            num_local_tokens,
                            d_w_routed_down_hie,
                            d_w_routed_gate_ihe,
                            d_w_routed_up_ihe,
                            fused_swiglu,
                            routed_descriptors=routed_descriptors,
                            num_experts_rt=num_experts_rt,
                            main_grad_up_row_offset=main_grad_up_row_offset,
                        )
                # No task-boundary cluster synchronization: only a CTA-local SwiGLU-backward task would
                # need one, and the SwiGLU-backward is fused into the Down dgrad epilogues.

            if role_warp == Int32(AB_LOAD_WARP):
                if generic_ab_live != Int32(0):
                    ab_pipeline.producer_tail(ab_producer_state)
                elif paired_ab_live != Int32(0):
                    paired_gate_up_pipeline.producer_tail(paired_ab_producer_state)
            if role_warp == Int32(UMMA_WARP):
                # no more MMA: the epilogue warps may free TMEM (unaligned, as every use of this barrier)
                tmem_barrier.arrive_unaligned()
                acc_pipeline.producer_tail(acc_producer_state)
                paired_acc_pipeline.producer_tail(paired_acc_producer_state)
            if role_warp <= Int32(EPILOGUE_WARPS[-1]):
                cute.arch.cp_async_bulk_wait_group(0, read=False)
                tmem.relinquish_alloc_permit()
                named_barrier_sync_unaligned(tmem_barrier)  # the MMA warp has issued its last MMA
                tmem_for_free = utils.TmemAllocator(
                    tmem_holding_buffer_ptr,
                    barrier_for_retrieve=tmem_barrier,
                    allocator_warp_id=0,
                    is_two_cta=True,
                    num_allocated_columns=self.abt.num_tmem_alloc_cols,
                    two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbarrier_ptr,
                    initialize_mbarrier=False,
                )
                tmem_for_free.free(acc_tmem_ptr)
            if role_warp == Int32(CLC_WARP) and cta_rank == Int32(0):
                clc_pipeline.producer_tail(clc_producer_state)

            self._drain_clc_capacity_tail(
                runtime,
                needs_drain,
                clc_drain_pipeline,
                drain_consumer_state,
                drain_producer_state,
                drain_scheduler,
                cta_rank,
            )


def _cute_tensor(tensor, leading_dim=None):
    return from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
        leading_dim=len(tensor.shape) - 1 if leading_dim is None else leading_dim
    )


def make_bwd_mega_args(
    inputs,
    state,
    *,
    config,
    stream,
    launch: bool = False,
    peer_tables=None,
    routed_storage_tables=None,
    routed_descriptors=None,
    up_row_offset: int = 0,
    main_grad_up_row_offset: int = 0,
    situ_beta=(0.0, 0.0),
    build_descriptors: int = 1,
):
    """Create the fixed descriptor set; every view is zero-copy.

    ``launch=False`` builds the CuTe tensors and DSL scalars that ``cute.compile``
    traces.  ``launch=True`` builds the arguments of a call to the compiled TVM FFI
    function: the same torch views and plain Python scalars, which the FFI entry
    converts natively (no ``from_dlpack`` per tensor and call), as the FWD host does.
    ``peer_tables`` is the workspace-owned cache of peer address tables (``None``: build them).
    """

    tables = {} if peer_tables is None else peer_tables
    if launch:
        wrap = lambda tensor, leading_dim=None: tensor  # noqa: E731
        Int32, Float32 = int, float  # noqa: N806 -- plain scalars at the FFI call
    else:
        wrap = _cute_tensor
        from cutlass import Float32, Int32  # noqa: N806

    raw = tuple(row_major_view(tensor) for tensor in state.raw13)
    counters = tuple(row_major_view(tensor) for tensor in state.counters)
    decoder_metadata = tuple(row_major_view(tensor) for tensor in state.metadata)
    d_x_shared, d_x_routed = raw[0], raw[1]
    d_gate_shared, d_gate_routed = raw[2], raw[3]
    d_up_shared, d_up_routed = raw[4], raw[5]
    d_y_routed = raw[6]
    d_w_shared_gate, d_w_routed_gate = raw[7], raw[8]
    d_w_shared_up, d_w_routed_up = raw[9], raw[10]
    d_w_shared_down, d_w_routed_down = raw[11], raw[12]

    # The routed dgrad reads the weights as [H,I,E] views.  The paired replay instead indexes
    # its B operand as [N,K,E], so retain a second zero-copy view instead of reusing the dgrad
    # descriptor with incompatible axes.
    w_shared_down_ih = row_major_view(inputs.w_shared_down).T
    w_routed_down_ihe = routed_weight_hie(inputs.w_routed_down)
    w_shared_gate_hi = row_major_view(inputs.w_shared_gate).T
    w_routed_gate_hie = routed_weight_hie(inputs.w_routed_gate)
    w_shared_up_hi = row_major_view(inputs.w_shared_up).T
    w_routed_up_hie = routed_weight_hie(inputs.w_routed_up)
    w_routed_gate_nke = routed_weight_view(inputs.w_routed_gate)
    w_routed_up_nke = routed_weight_view(inputs.w_routed_up)

    # AtB uses the literal A.T/B.T zero-copy views.  Routed outputs put E last
    # so a plain batch-coordinate TMA store owns one expert matrix.
    cuda_stream = stream if isinstance(stream, cuda.CUstream) else cuda.CUstream(stream.cuda_stream)
    args = (
        wrap(peer_address_table(inputs.x_peer_ptrs, inputs.x_shared.device, tables)),
        wrap(peer_address_table(inputs.d_y_peer_ptrs, inputs.x_shared.device, tables)),
        wrap(peer_address_table(inputs.d_x_peer_ptrs, inputs.x_shared.device, tables)),
        wrap(peer_address_table(inputs.router_peer_ptrs, inputs.x_shared.device, tables)),
        wrap(peer_address_table(inputs.d_router_peer_ptrs, inputs.x_shared.device, tables)),
        wrap(row_major_view(inputs.schedule_peer_rank)),
        wrap(row_major_view(inputs.schedule_peer_token_idx)),
        wrap(row_major_view(inputs.num_tokens)),
        wrap(row_major_view(inputs.tokens_per_expert)),
        *(wrap(item) for item in decoder_metadata),
        wrap(row_major_view(inputs.x_shared)),
        wrap(row_major_view(inputs.x_routed)),
        wrap(row_major_view(inputs.gate_shared)),
        wrap(row_major_view(inputs.gate_routed)),
        wrap(row_major_view(inputs.up_shared)),
        wrap(row_major_view(inputs.up_routed)),
        wrap(row_major_view(inputs.hidden_shared)),
        wrap(row_major_view(inputs.hidden_routed)),
        wrap(row_major_view(inputs.d_y_shared)),
        wrap(w_shared_down_ih, 0),
        wrap(w_routed_down_ihe, 0),
        wrap(w_shared_gate_hi, 0),
        wrap(w_routed_gate_hie, 0),
        wrap(w_shared_up_hi, 0),
        wrap(w_routed_up_hie, 0),
        wrap(w_routed_gate_nke, 1),
        wrap(w_routed_up_nke, 1),
        wrap(row_major_view(inputs.x_shared).T, 0),
        wrap(row_major_view(inputs.x_routed).T, 0),
        wrap(row_major_view(inputs.hidden_shared).T, 0),
        wrap(row_major_view(inputs.hidden_routed).T, 0),
        wrap(row_major_view(inputs.d_y_shared).T, 0),
        wrap(d_y_routed.T, 0),
        wrap(d_gate_shared.T, 0),
        wrap(d_gate_routed.T, 0),
        wrap(d_up_shared.T, 0),
        wrap(d_up_routed.T, 0),
        wrap(d_x_shared),
        wrap(d_x_routed),
        wrap(d_gate_shared),
        wrap(d_gate_routed),
        wrap(d_up_shared),
        wrap(d_up_routed),
        wrap(d_y_routed),
        wrap(d_w_shared_down),
        wrap(d_w_shared_gate),
        wrap(d_w_shared_up),
        wrap(routed_weight_view(d_w_routed_down), 1),
        wrap(routed_weight_view(d_w_routed_gate), 1),
        wrap(routed_weight_view(d_w_routed_up), 1),
        wrap(row_major_view(state.router_weights)),
        wrap(row_major_view(state.d_router_weight_partials)),
        *(wrap(counter) for counter in counters),
        Int32(config.num_local_tokens),
        Int32(config.schedule_capacity),
        Int32(config.macrobatch_size),
        Int32(config.minibatch_size),
        Int32(config.num_comm_sms),
        Int32(config.topk),
        Int32(config.task_owner_bucket_shift),
        Float32(0.0 if config.swiglu_limit is None else config.swiglu_limit),
        wrap(_routed_tables_or_placeholder(routed_storage_tables, inputs.x_shared.device), 1),
        wrap(_routed_descriptors_or_placeholder(routed_descriptors, inputs.x_shared.device), 0),
        Int32(build_descriptors),
        Int32(up_row_offset),
        Int32(main_grad_up_row_offset),
        Float32(situ_beta[0]),
        Float32(situ_beta[1]),
        cuda_stream,
    )
    return args


#: build_descriptors' position, counted from the end (before up_row_offset, main_grad_up_row_offset,
#: the two SiTU betas and the stream): per call, whether this launch builds the routed descriptors.
_BUILD_DESCRIPTORS_SLOT = -6


def fill_bwd_launch_slots(args, x_shared, d_y_shared, schedule, weights, context, raw13, stream, metadata_count, alias_slots, build=0):
    """Fill the per-call slots of a launch-argument list built by make_bwd_mega_args(launch=True): the
    caller-owned tensors (x_shared / d_y_shared, the caller's shared inputs or the workspace x / dY
    buffers, and their transposes; the four schedule tensors
    and their num_tokens aliases in the decoder metadata; the six weights' views; the seven
    forward-context activations and their transposes), the call's fresh ``raw13`` (the seven scratch
    tensors 0-6, five of them also as transposes, and the six weight-gradient outputs 7-12, the routed
    ones as [N,K,E] views) and the stream.  ``schedule`` is (peer_rank, peer_token_idx, num_tokens,
    tokens_per_expert) and ``weights`` (shared/routed gate, up, down).  The template keeps only the
    workspace preparation's buffers (readiness counters and decoder metadata, schedule-capacity
    sized; router weights and partials, macrobatch-sized), peer tables and scalars.  Positions follow
    make_bwd_mega_args."""

    # row_major_view only where a singleton extent can carry an arbitrary stride: num_tokens ([1]) and
    # the routed weights ([E, ...] with E = 1 possible).  The shared inputs, the forward context and the
    # shared weights were validated contiguous with every extent >= 256 when the plan was built, and a
    # call reaches this filler only after the plan's source check (weights) and layouts_match (shared
    # inputs, context) found the same shapes and strides: their canonical views are the tensors themselves.
    rmv = row_major_view
    args[5] = rmv(schedule[0])
    args[6] = rmv(schedule[1])
    args[7] = num_tokens = rmv(schedule[2])
    args[8] = rmv(schedule[3])
    for slot in alias_slots:
        args[slot] = num_tokens
    b = 9 + metadata_count  # after 5 peer tables, 4 schedule tensors, the decoder metadata
    args[b] = x_shared
    args[b + 1 : b + 8] = context
    args[b + 8] = d_y_shared
    args[b + 9] = weights[4].T
    args[b + 10] = routed_weight_hie(weights[5])
    args[b + 11] = weights[0].T
    args[b + 12] = routed_weight_hie(weights[1])
    args[b + 13] = weights[2].T
    args[b + 14] = routed_weight_hie(weights[3])
    args[b + 15] = routed_weight_view(weights[1])
    args[b + 16] = routed_weight_view(weights[3])
    args[b + 17] = x_shared.T
    args[b + 18] = context[0].T  # x_routed
    args[b + 19] = context[5].T  # hidden_shared
    args[b + 20] = context[6].T  # hidden_routed
    args[b + 21] = d_y_shared.T
    # raw13 is the call's own allocation (persistent_bf16_bwd._allocate_raw13): freshly allocated
    # tensors carry canonical row-major strides whatever their extents (also for E = 1 in the routed
    # dW), so no row_major_view normalization.
    args[b + 22] = raw13[6].T  # d_y_routed
    args[b + 23] = raw13[2].T  # d_gate_shared
    args[b + 24] = raw13[3].T  # d_gate_routed
    args[b + 25] = raw13[4].T  # d_up_shared
    args[b + 26] = raw13[5].T  # d_up_routed
    args[b + 27 : b + 34] = raw13[:7]  # d_x_shared, d_x_routed, d_gate_*, d_up_*, d_y_routed
    args[b + 34] = raw13[11]  # d_w_shared_down
    args[b + 35] = raw13[7]  # d_w_shared_gate
    args[b + 36] = raw13[9]  # d_w_shared_up
    args[b + 37] = routed_weight_view(raw13[12])  # d_w_routed_down
    args[b + 38] = routed_weight_view(raw13[8])  # d_w_routed_gate
    args[b + 39] = routed_weight_view(raw13[10])  # d_w_routed_up
    args[_BUILD_DESCRIPTORS_SLOT] = build
    args[-1] = stream
    return args


def bwd_per_call_slots(metadata_count, alias_slots):
    """Slots fill_bwd_launch_slots fills on every call."""

    b = 9 + metadata_count
    return (
        (5, 6, 7, 8)
        + tuple(alias_slots)
        + tuple(b + i for i in range(0, 9))
        + tuple(b + i for i in range(9, 17))
        + tuple(b + i for i in range(17, 27))
        + tuple(b + i for i in range(27, 40))
        + (_BUILD_DESCRIPTORS_SLOT, -1)
    )


def prepare_bwd_mega(
    inputs,
    state,
    *,
    config,
    stream,
    task_decode: TaskDecode,
    split_weights: bool = False,
    accumulate_wgrad: bool = False,
    bf16_main_grad: bool = False,
    split_main_grads: bool = False,
    is_situ: bool = False,
    has_shared: bool = True,
    has_tail: bool = False,
    recompute_forward_context: bool = False,
    **kwargs,
):
    """Compile the sole launch for one (clamp mode, task decode, routed layout, wgrad
    destination, communication tile shape of ``config``) variant and return the executor; all geometry scalars
    and tensor layouts stay runtime arguments."""

    args = make_bwd_mega_args(inputs, state, config=config, stream=stream, **kwargs)
    mega = _PersistentBf16BwdMega(
        config.swiglu_limit is not None,
        task_decode,
        split_weights=split_weights,
        accumulate_wgrad=accumulate_wgrad,
        bf16_main_grad=bf16_main_grad,
        split_main_grads=split_main_grads,
        is_situ=is_situ,
        has_shared=has_shared,
        has_tail=has_tail,
        recompute_forward_context=recompute_forward_context,
        comm_tile_shape=config.comm_tile_shape,
    )
    return cute.compile(mega, *args, options=COMPILE_OPTIONS)


__all__ = [
    "bwd_per_call_slots",
    "fill_bwd_launch_slots",
    "make_bwd_mega_args",
    "prepare_bwd_mega",
]
