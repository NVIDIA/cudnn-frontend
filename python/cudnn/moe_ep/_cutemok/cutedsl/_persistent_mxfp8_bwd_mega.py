# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""The MXFP8 backward megakernel: dispatch, MLP, SwiGLU and combine backward in one persistent kernel.

The routed experts run on e4m3 operands with e8m0 scales; the shared expert stays BF16 (its GEMM task is
the base class's, ``_mxfp8_bwd_base``).  Two kernel classes, by the routed Gate / Up context the forward
saved: ``_PersistentMxfp8BwdMegaE4m3Context`` reads it as e4m3 tiles with scales (the backward of a
``gate_up_context="mxfp8"`` forward); ``_PersistentMxfp8BwdMega`` reads it as BF16, so its routed SwiGLU
backward sees Gate / Up unquantized.

* Communication CTAs run one dynamic task stream per macrobatch (``_backward_stream``, the forward
  stream's shape): the reverse-combine of dY, in a replayed macrobatch the replay dispatch of x, and the
  reverse-dispatch of dX.
* The router weight enters in the reverse-combine of dY (each row scaled in place before its
  quantization), and the routed SwiGLU backward divides by it for the router-weight partial.
* Every replayed macrobatch reruns Gate, Up and SwiGLU (``REPLAY_SWIGLU``), since Wgrad-down reads the
  transposed e4m3 Hidden.
* ``recompute_forward_context`` (a compile-time variant flag): macrobatch 0 is a replayed macrobatch as well -- its
  stream pulls x, its minibatches run the replay tasks and its tasks wait on the replay's products -- so the routed
  context holds nothing on entry and the replay writes it before anything reads it.  The shared expert's context is
  the forward's either way.
* The compute task grid is fixed by the runtime sizes (3 x E x weight tiles of Wgrad per macrobatch).  A
  routed Wgrad task's K range stops at its expert's last valid row, rounded up to the K tile, when the
  per-CTA expert table has room for the valid ends (``_write_expert_valid_ends``: SMEM_BINARY and
  2 E <= EXPERT_TABLE_CAP); otherwise it is the 256-padded range.  The skipped tiles hold only
  dispatch-zeroed rows, so the two ranges give the same dW up to the sign of a zero (a -0 partial sum
  that the extra +0 tiles would turn into +0).

All sizes (tokens, hiddens, intermediate, experts, top-k, batch sizes, comm SMs) are runtime values; the
compile-time variant flags of the kernel class select code.

This module holds the device side -- task decode and geometry first, then the routed GEMMs, the SwiGLU
tasks and the communication prefix -- and the host half (``_launch``, the argument builder and the
compile entry).  A backward is two launches on the stream: a one-warp prologue (``_write_tensormaps``)
copies the routed GEMMs' TMA descriptors into the state's ``tensormaps`` buffer, and the mega's producer
warps read them from global memory (``ROUTED_TMA_POSITIONS``) instead of the kernel parameters.

Named barriers (ids 0 and 3 are the CuTe pipelines' / cluster's): 1 the GEMM epilogue's 128 threads,
2 the TMEM allocator's 256 threads, 4 the D-store tail's 128 (``_persistent_mxfp8_fwd_gemm`` holds the
table).  The SwiGLU tasks and their quantization use block syncs only.
"""

import enum
import inspect
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Float8E4M3FN, Float8E8M0FNU, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum, TensorMapManager, TensorMapUpdateMode

from ._compat.pipeline import PipelineTmaUmma, PipelineUmmaAsync

from . import _mxfp8_bwd_swiglu as swiglu_bwd
from . import _mxfp8_gemm_base as gemm
from . import routed_descriptors as routed_descriptors_pkg
from . import _persistent_mxfp8_fwd_gemm as mxgemm
from ._bwd_device_helpers import (
    _ceil_div_i32,
    _counter_wait_gpu,
    _issue_swiglu_tma_load,
    _issue_swiglu_tma_store,
    _load_peer_float,
    _peer_address,
    _store_peer_float,
    _swizzled_tile_i32,
)
from ._arch import COMPILE_OPTIONS
from ._host_common import peer_address_table, row_major_view
from ._mxfp8_bwd_base import (
    AB_TMA_PRODUCER_WARP,
    BF16_IDLE_WARP,
    CLC_PRODUCER_WARP,
    EPILOGUE_WARP_LAST,
    REPLAY_GATE,
    REPLAY_UP,
    ROUTED_DOWN_DGRAD,
    ROUTED_GATE_UP_DGRAD,
    ROUTED_SWIGLU_BWD,
    ROUTED_WGRAD_DOWN,
    ROUTED_WGRAD_GATE,
    SHARED_DOWN_DGRAD,
    SHARED_GATE_UP_DGRAD,
    SHARED_SWIGLU_BWD,
    SHARED_WGRAD_DOWN,
    SHARED_WGRAD_GATE,
    SHARED_WGRAD_UP,
    SWIGLU_TILE_M,
    SWIGLU_TILE_N,
    SWIGLU_TMA_TRANSACTION_BYTES,
    THREADS_PER_CTA,
    UMMA_PRODUCER_WARP,
    _Mxfp8BwdBase,
    _zero_empty_routed_wgrads,
)
from ._mxfp8_bwd_math import add_ftz, inverse_router_weight, situ_bwd, swiglu_bwd_pair, swiglu_bwd_shared
from ._mxfp8_fwd_base import DYNAMIC_SHARED_MEMORY_BYTES
from ._mxfp8_quant import _load_shared_v4, _store_shared_v4_words, cvt_bf16x2
from ._persistent_bf16_bwd_mega import _epilog_gmem_copy_fn, _padded_segment_valid_end
from ._persistent_bf16_fwd_mega import _build_expert_cu_end, _routed_expert_tile_i32
from ._persistent_mxfp8_fwd_mega import (
    STREAM_BUFFERS,
    STREAM_COLUMNS,
    STREAM_GROUP_BLOCKS,
    STREAM_PULL,
    STREAM_PUSH,
    STREAM_ROWS,
    STREAM_WARP_BYTES,
    STREAM_WARPS,
    _counter_arrive_release_gpu,
    _divide_u32,
    _fence_proxy_async_global,
    _PersistentMxfp8FwdMega,
    _reciprocal_u32,
    _stream_row_bytes,
)
from ._tma_1d import tma_load_1d_raw, tma_store_1d_raw
from .mxfp8_contract import ROUTED_SWIGLU as FWD_ROUTED_SWIGLU_KIND
from .persistent_bf16_fwd_contract import EXPERT_TABLE_CAP, MLP_TILE_ROWS, TaskDecode
from .persistent_bf16_fwd_contract import THREADS_PER_CTA as _FWD_THREADS_PER_CTA
from .persistent_mxfp8_bwd import ROUTED_TENSORMAPS, TENSORMAP_BYTES
from ._persistent_bf16_fwd_mega import (
    _routed_descriptors_or_placeholder,
    _routed_tables_or_placeholder,
    routed_weight_view,
)
from .persistent_mxfp8_fwd import dynamic_layout_tensor

# The kind the base module leaves free: replayed forward SwiGLU, quantizing
# Hidden into the e4m3 context Wgrad-down reads.
REPLAY_SWIGLU = 8
SF_PRODUCER_WARP = BF16_IDLE_WARP  # issues the routed GEMMs' scale loads
MLP_TILE_M = 256  # GEMM tile rows
# The routed GEMM tasks decode their expert through the BF16 forward's row-block table helpers
# (_build_expert_cu_end / _routed_expert_tile_i32): the backward's row tile and CTA size must be the forward's.
assert MLP_TILE_M == MLP_TILE_ROWS
assert THREADS_PER_CTA == _FWD_THREADS_PER_CTA
MLP_TILE_N = 256  # GEMM tile columns
FP8_K_TILE = 128  # K tile of the e4m3 GEMMs
SWIGLU_TILE = 128  # SwiGLU tile rows and columns
SWIGLU_FWD_STAGES = 3  # tiles per CTA of a replayed forward SwiGLU task
SWIGLU_BWD_STAGES = 2  # tiles per CTA of a SwiGLU backward task
CLUSTER_SIZE = 2
SF_TILE_BYTES = 512  # one 128 x 128 tile's e8m0 scales
FP8_TILE_BYTES = SWIGLU_TILE * SWIGLU_TILE
BF16_TILE_BYTES = 2 * FP8_TILE_BYTES
# The routed SwiGLU backward's shared memory with a BF16 Gate / Up context: per stage d_hidden | gate | up (BF16
# 128 x 128 each, the task's two tiles loaded at once), then per stage the d_gate / d_up normal scale tiles, the
# transposed scale tiles and the router partials.  BF16 d_gate / d_up go back over gate / up; the transposed e4m3
# tiles over the stage's d_hidden once it is read; the normal e4m3 rows leave from registers.
SWIGLU_BWD_D_HIDDEN = 0
SWIGLU_BWD_GATE = SWIGLU_BWD_D_HIDDEN + 2 * BF16_TILE_BYTES
SWIGLU_BWD_UP = SWIGLU_BWD_GATE + 2 * BF16_TILE_BYTES
SWIGLU_BWD_D_SC = SWIGLU_BWD_UP + 2 * BF16_TILE_BYTES
SWIGLU_BWD_D_T_SC = SWIGLU_BWD_D_SC + 2 * 2 * SF_TILE_BYTES
SWIGLU_BWD_ROUTER = SWIGLU_BWD_D_T_SC + 2 * 2 * SF_TILE_BYTES
SWIGLU_BWD_BYTES = SWIGLU_BWD_ROUTER + 2 * SWIGLU_TILE * 4
SWIGLU_BWD_TRANSACTION_BYTES = 3 * BF16_TILE_BYTES
assert SWIGLU_BWD_BYTES <= mxgemm.MXFP8_ARENA_BYTES and 2 * FP8_TILE_BYTES == BF16_TILE_BYTES
# gate_up_context="mxfp8" (the e4m3 / scale Gate / Up context): the routed SwiGLU backward's shared memory:
# d_hidden[2] (BF16) | gate[2] | up[2] (e4m3) | gate_sc[2] | up_sc[2] | d_up BF16 staging | d_t[2] (e4m3) |
# d_t_sc[2] | router partials[2][128].
E4M3_CONTEXT_SWIGLU_BWD_D_HIDDEN = 0
E4M3_CONTEXT_SWIGLU_BWD_GATE = E4M3_CONTEXT_SWIGLU_BWD_D_HIDDEN + 2 * BF16_TILE_BYTES
E4M3_CONTEXT_SWIGLU_BWD_UP = E4M3_CONTEXT_SWIGLU_BWD_GATE + 2 * FP8_TILE_BYTES
E4M3_CONTEXT_SWIGLU_BWD_GATE_SC = E4M3_CONTEXT_SWIGLU_BWD_UP + 2 * FP8_TILE_BYTES
E4M3_CONTEXT_SWIGLU_BWD_UP_SC = E4M3_CONTEXT_SWIGLU_BWD_GATE_SC + 2 * SF_TILE_BYTES
E4M3_CONTEXT_SWIGLU_BWD_D_UP = E4M3_CONTEXT_SWIGLU_BWD_UP_SC + 2 * SF_TILE_BYTES
E4M3_CONTEXT_SWIGLU_BWD_D_T = E4M3_CONTEXT_SWIGLU_BWD_D_UP + BF16_TILE_BYTES
E4M3_CONTEXT_SWIGLU_BWD_D_T_SC = E4M3_CONTEXT_SWIGLU_BWD_D_T + 2 * FP8_TILE_BYTES
E4M3_CONTEXT_SWIGLU_BWD_ROUTER = E4M3_CONTEXT_SWIGLU_BWD_D_T_SC + 2 * SF_TILE_BYTES
E4M3_CONTEXT_SWIGLU_BWD_BYTES = E4M3_CONTEXT_SWIGLU_BWD_ROUTER + 2 * SWIGLU_TILE * 4
E4M3_CONTEXT_SWIGLU_BWD_TRANSACTION_BYTES = BF16_TILE_BYTES + 2 * FP8_TILE_BYTES + 2 * SF_TILE_BYTES
assert E4M3_CONTEXT_SWIGLU_BWD_BYTES <= mxgemm.MXFP8_ARENA_BYTES and E4M3_CONTEXT_SWIGLU_BWD_D_UP % 1024 == 0


# The backward stream's third queue (the forward's two are STREAM_PULL / STREAM_PUSH): a replayed macrobatch's
# pull of x, the forward's dispatch pull (no row scaling), quantized into the replayed-x buffers.
STREAM_PULL_X = 2
assert STREAM_PULL_X not in (STREAM_PULL, STREAM_PUSH)
assert STREAM_WARPS * STREAM_WARP_BYTES <= mxgemm.MXFP8_ARENA_BYTES  # the stream warps' buffers fit the arena


class _B(enum.IntEnum):
    """Positions in the kernel's ``atoms`` / ``tensors`` lists (each atom with its TMA tensor).

    ``SH_*``: the shared expert's BF16 GEMM (ABt dgrad, AtB Wgrad) and SwiGLU-backward
    operands.  ``A_*`` / ``B_*``: the routed e4m3 operands (K-major),
    ``SFA_*`` / ``SFB_*`` their scales at the same offsets, ``D_*`` the routed BF16 outputs
    (``*_ADD``: reduce-add for later Wgrad contributions), ``T_*`` e4m3 tile stores and
    ``L_*`` tile loads (SwiGLU-backward and the replay SwiGLU).
    """

    SH_A_D_Y = 0
    SH_A_D_GATE = 1
    SH_A_D_UP = 2
    SH_B_DOWN = 3
    SH_B_GATE = 4
    SH_B_UP = 5
    SH_D_D_HIDDEN = 6
    SH_D_D_X = 7
    SH_WA_D_Y = 8
    SH_WA_D_GATE = 9
    SH_WA_D_UP = 10
    SH_WB_HIDDEN = 11
    SH_WB_X = 12
    SH_WD_DOWN = 13
    SH_WD_GATE = 14
    SH_WD_UP = 15
    SH_L_GATE = 16
    SH_L_UP = 17
    SH_L_D_HIDDEN = 18
    SH_T_D_GATE = 19
    SH_T_D_UP = 20
    A_X = 21
    A_D_Y = 22
    A_D_GATE = 23
    A_D_UP = 24
    A_D_Y_T = 25
    A_D_GATE_T = 26
    A_D_UP_T = 27
    B_GATE = 28
    B_UP = 29
    B_DOWN_T = 30
    B_GATE_T = 31
    B_UP_T = 32
    B_HIDDEN_T = 33
    B_X_T = 34
    SFA_X = 35
    SFB_GATE = 42
    D_D_HIDDEN = 49
    D_D_X = 50
    D_GATE = 51
    D_UP = 52
    D_W_DOWN = 53
    D_W_GATE = 54
    D_W_UP = 55
    D_W_DOWN_ADD = 56
    D_W_GATE_ADD = 57
    D_W_UP_ADD = 58
    T_HIDDEN = 59
    T_HIDDEN_T = 60
    T_D_GATE_T = 61
    T_D_UP_T = 62
    L_D_HIDDEN = 63
    L_GATE = 64
    L_UP = 65
    L_REPLAY_GATE = 66
    L_REPLAY_UP = 67
    # The shared weight gradients' reduce-add atoms, used instead of the plain stores when the
    # gradients accumulate into main_grad -- the routed ones' *_ADD, for the shared expert.  After
    # every other position: only the accumulate_wgrad variant's lists carry them.
    SH_WD_DOWN_ADD = 68
    SH_WD_GATE_ADD = 69
    SH_WD_UP_ADD = 70
    COUNT = 71


# SFA / SFB of a routed operand: ``_B.SFA_X + (operand - _B.A_X)``, ``_B.SFB_GATE + (operand - _B.B_GATE)``.
SFA_OF = _B.SFA_X - _B.A_X
SFB_OF = _B.SFB_GATE - _B.B_GATE

#: Which row of the split address table -- and so which descriptor array -- a routed weight
#: operand reads.  The five the dgrad needs, then their five scales, in the order
#: ``functional.backward`` hands them over: Gate, Up, Gate transposed, Up transposed, Down
#: transposed.  (These positions are the same in ``_BE4m3Context``.)
SPLIT_ROWS = {
    _B.B_GATE: 0,
    _B.B_UP: 1,
    _B.B_GATE_T: 2,
    _B.B_UP_T: 3,
    _B.B_DOWN_T: 4,
    _B.SFB_GATE: 5,
    _B.SFB_GATE + (_B.B_UP - _B.B_GATE): 6,
    _B.SFB_GATE + (_B.B_GATE_T - _B.B_GATE): 7,
    _B.SFB_GATE + (_B.B_UP_T - _B.B_GATE): 8,
    _B.SFB_GATE + (_B.B_DOWN_T - _B.B_GATE): 9,
}
#: How many descriptor arrays the split backward builds, and so how many rows its table has.
SPLIT_OPERANDS = len(SPLIT_ROWS)
#: The same operands in table-row order, which is the order the descriptor arrays are built in.
SPLIT_ORDER = tuple(sorted(SPLIT_ROWS, key=SPLIT_ROWS.__getitem__))

#: Which row of the address table a split main_grad buffer's reduce-add atom reads, after the
#: ten the dgrad needs.  Down, Gate, Up, as ``functional.backward`` hands the tables over.
MAIN_GRAD_ROWS = {
    _B.D_W_DOWN_ADD: SPLIT_OPERANDS,
    _B.D_W_GATE_ADD: SPLIT_OPERANDS + 1,
    _B.D_W_UP_ADD: SPLIT_OPERANDS + 2,
}
#: Every operand with a descriptor array when both the weights and the main_grads are split.
SPLIT_WITH_MAIN_GRADS = SPLIT_OPERANDS + len(MAIN_GRAD_ROWS)
assert _B.SFA_X + (_B.A_D_UP_T - _B.A_X) < _B.SFB_GATE and _B.SFB_GATE + (_B.B_X_T - _B.B_GATE) < _B.D_D_HIDDEN
# The routed GEMMs' TMA descriptors -- the e4m3 A / B operands and their scales, atom positions A_X .. B_X_T and
# SFA_X .. SFB_X_T -- live in the state's ``tensormaps`` buffer for the duration of a call (slot = position - A_X):
# ``_write_tensormaps`` (one warp, launched right before the mega) copies them there from the kernel-parameter
# atoms, and the producer warps' K loops pass each descriptor's global address to every bulk tensor copy.
# With the descriptors read from the kernel parameters, ptxas re-forms the parameter-space address of the
# tensormap before every TMA issue in the producer K loops (PTX ``cvta.param``: a constant load and two adds), on a
# single-thread issue path that already fills most of the K-tile period.  One loop-invariant global pointer per
# descriptor keeps the issue path short.
# The slot of a position is the same for both context variants: the position-agreement assert below
# ``_BE4m3Context`` covers every position under ``T_HIDDEN``, which includes all of ``ROUTED_TMA_POSITIONS``.
# Visibility: the prologue launch releases the descriptors to the tensormap proxy, but the tensormap proxy is not
# acquired from the generic proxy at kernel start (PTX ISA, memory consistency model, API synchronization), so
# stream order alone does not make the writes visible to the TMA unit.  Each producer warp therefore issues
# ``fence.proxy.tensormap::generic.acquire`` for its 14 descriptors once per CTA before its task loop: the fence is
# per thread block and must precede the block's first use of the map, and the elected lane that issues it is the
# lane that issues every TMA of that warp, so the warp sync after it is only for the lanes that never touch the
# descriptors.  Once per CTA, not once per task: a per-task acquire invalidates the TMA unit's descriptor cache
# before every task.
AB_TMA_POSITIONS = tuple(range(_B.A_X, _B.B_X_T + 1))  # the A / B operands: the AB producer warp's descriptors
SF_TMA_POSITIONS = tuple(range(_B.SFA_X, _B.SFB_GATE + (_B.B_X_T - _B.B_GATE) + 1))  # their scales: the SF producer's
ROUTED_TMA_POSITIONS = AB_TMA_POSITIONS + SF_TMA_POSITIONS
assert (
    len(ROUTED_TMA_POSITIONS) == ROUTED_TENSORMAPS
), "the routed atom positions changed: update persistent_mxfp8_bwd.ROUTED_TENSORMAPS (it sizes the state buffer)"
assert ROUTED_TMA_POSITIONS == tuple(range(_B.A_X, _B.SFB_GATE + (_B.B_X_T - _B.B_GATE) + 1))  # one slot range
_TENSORMAP_MANAGER = TensorMapManager(TensorMapUpdateMode.GMEM, TENSORMAP_BYTES)


@cute.jit
def _tensormap_ptr(tensormaps: cute.Tensor, position: cutlass.Constexpr[int], address_space: cutlass.Constexpr = cute.AddressSpace.generic) -> cute.Pointer:
    """The global copy of the TMA descriptor of atom position ``position`` (one of ``ROUTED_TMA_POSITIONS``): the
    generic pointer the bulk tensor copies take, or with ``address_space`` gmem the pointer ``copy_tensormap``
    writes through."""
    return _TENSORMAP_MANAGER.get_tensormap_ptr(tensormaps.iterator + (position - _B.A_X) * TENSORMAP_BYTES, address_space)


class _S(enum.IntEnum):
    """Positions in the kernel's ``scales`` list: 512 B e8m0 tiles moved by bulk copies, then the e4m3
    d_gate / d_up the routed SwiGLU backward stores from registers (addressed by pointer), then the plain e4m3
    dY and replayed-x outputs (normal, transposed) the stream's per-warp quantization stores to directly."""

    D_Y = 0
    D_Y_T = 1
    X = 2
    X_T = 3
    HIDDEN = 4
    HIDDEN_T = 5
    D_GATE = 6
    D_UP = 7
    D_GATE_T = 8
    D_UP_T = 9
    D_GATE_FP8 = 10
    D_UP_FP8 = 11
    D_Y_FP8 = 12
    D_Y_FP8_T = 13
    X_FP8 = 14
    X_FP8_T = 15
    COUNT = 16


class _BE4m3Context(enum.IntEnum):
    """``gate_up_context="mxfp8"`` (the e4m3 / scale Gate / Up context): positions in the kernel's
    ``atoms`` / ``tensors`` lists (each atom with its TMA tensor).  The groups are ``_B``'s; the e4m3
    Gate / Up context adds the ``T_GATE`` / ``T_UP`` and ``T_D_GATE`` / ``T_D_UP`` tile stores.
    """

    SH_A_D_Y = 0
    SH_A_D_GATE = 1
    SH_A_D_UP = 2
    SH_B_DOWN = 3
    SH_B_GATE = 4
    SH_B_UP = 5
    SH_D_D_HIDDEN = 6
    SH_D_D_X = 7
    SH_WA_D_Y = 8
    SH_WA_D_GATE = 9
    SH_WA_D_UP = 10
    SH_WB_HIDDEN = 11
    SH_WB_X = 12
    SH_WD_DOWN = 13
    SH_WD_GATE = 14
    SH_WD_UP = 15
    SH_L_GATE = 16
    SH_L_UP = 17
    SH_L_D_HIDDEN = 18
    SH_T_D_GATE = 19
    SH_T_D_UP = 20
    A_X = 21
    A_D_Y = 22
    A_D_GATE = 23
    A_D_UP = 24
    A_D_Y_T = 25
    A_D_GATE_T = 26
    A_D_UP_T = 27
    B_GATE = 28
    B_UP = 29
    B_DOWN_T = 30
    B_GATE_T = 31
    B_UP_T = 32
    B_HIDDEN_T = 33
    B_X_T = 34
    SFA_X = 35
    SFB_GATE = 42
    D_D_HIDDEN = 49
    D_D_X = 50
    D_GATE = 51
    D_UP = 52
    D_W_DOWN = 53
    D_W_GATE = 54
    D_W_UP = 55
    D_W_DOWN_ADD = 56
    D_W_GATE_ADD = 57
    D_W_UP_ADD = 58
    T_GATE = 59
    T_UP = 60
    T_HIDDEN = 61
    T_HIDDEN_T = 62
    T_D_GATE = 63
    T_D_UP = 64
    T_D_GATE_T = 65
    T_D_UP_T = 66
    L_D_HIDDEN = 67
    L_GATE = 68
    L_UP = 69
    L_REPLAY_GATE = 70
    L_REPLAY_UP = 71
    # The shared weight gradients' reduce-add atoms, as in ``_B``: only the accumulate_wgrad variant's lists
    # carry them.
    SH_WD_DOWN_ADD = 72
    SH_WD_GATE_ADD = 73
    SH_WD_UP_ADD = 74
    COUNT = 75


class _SE4m3Context(enum.IntEnum):
    """``gate_up_context="mxfp8"``: positions in the kernel's ``scales`` list: 512 B e8m0 tiles moved by bulk
    copies, then the plain e4m3 dY and replayed-x outputs (normal, transposed) the stream's per-warp quantization
    stores to directly."""

    D_Y = 0
    D_Y_T = 1
    X = 2
    X_T = 3
    GATE = 4
    UP = 5
    HIDDEN = 6
    HIDDEN_T = 7
    D_GATE = 8
    D_UP = 9
    D_GATE_T = 10
    D_UP_T = 11
    D_Y_FP8 = 12
    D_Y_FP8_T = 13
    X_FP8 = 14
    X_FP8_T = 15
    COUNT = 16


# The positions below ``_B.T_HIDDEN`` (operands, scales, epilogues) are the same in both context variants.
assert all(_B[name] == _BE4m3Context[name] for name in _B.__members__ if _B[name] < _B.T_HIDDEN)


@cute.struct
class _Mxfp8BwdSharedStorage:
    clc_mbarriers: cute.struct.MemRange[cutlass.Int64, 2]
    clc_response: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
    clc_drain_mbarriers: cute.struct.MemRange[cutlass.Int64, 16]
    clc_drain_responses: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 32], 16]
    ab_full: cute.struct.MemRange[cutlass.Int64, gemm.AB_STAGES]
    ab_empty: cute.struct.MemRange[cutlass.Int64, gemm.AB_STAGES]
    sf_full: cute.struct.MemRange[cutlass.Int64, gemm.AB_STAGES]  # the routed scales' own ring
    sf_empty: cute.struct.MemRange[cutlass.Int64, gemm.AB_STAGES]
    acc_full: cutlass.Int64
    acc_empty: cutlass.Int64
    swiglu_fwd_mbarriers: cute.struct.MemRange[cutlass.Int64, SWIGLU_FWD_STAGES]
    swiglu_bwd_mbarriers: cute.struct.MemRange[cutlass.Int64, SWIGLU_BWD_STAGES]
    stream_mbarriers: cute.struct.MemRange[cutlass.Int64, STREAM_WARPS * STREAM_BUFFERS]  # one per stream buffer
    tmem_dealloc: cutlass.Int64
    tmem_address: cutlass.Int32
    # The BF16 forward's per-CTA expert table (TaskDecode): inclusive row-block prefix per expert (or group).
    expert_cu_end: cute.struct.MemRange[cutlass.Int32, EXPERT_TABLE_CAP]
    # The task arena (1024-aligned): every task's staging lives in it.
    arena: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, mxgemm.MXFP8_ARENA_BYTES], 1024]


MXFP8_BWD_SHARED_MEMORY_BYTES = _Mxfp8BwdSharedStorage.size_in_bytes()
# The fields before the arena fit in its 1024 B alignment padding, so the struct is the arena plus one padded
# kilobyte block; the assert is the whole budget.
assert MXFP8_BWD_SHARED_MEMORY_BYTES <= DYNAMIC_SHARED_MEMORY_BYTES


@cute.jit
def _backward_stream_task(task: Int32, tiles: Int32, pull_tasks: Int32) -> tuple[Int32, Int32]:
    """Stream task ``task`` of one macrobatch -> (queue, 32-row tile).  Tasks ``0 .. pull_tasks - 1`` are the
    pulls: macrobatch 0's dY pulls in tile order (``pull_tasks == tiles``), or a replayed macrobatch's x pull and
    dY pull of each tile in turn (``pull_tasks == 2 tiles``: even = x, odd = dY; the compute side's first tasks
    of a replayed macrobatch, the replay Gate / Up, read x, and its Down dgrad reads dY only after them).  Tasks
    ``pull_tasks .. pull_tasks + tiles - 1`` are the dX pushes in tile order.  No division."""
    queue = Int32(STREAM_PUSH)
    tile = task - pull_tasks
    if task < pull_tasks:
        queue = Int32(STREAM_PULL)
        tile = task
        if pull_tasks > tiles:
            tile = task >> Int32(1)
            if (task & Int32(1)) == Int32(0):
                queue = Int32(STREAM_PULL_X)
    return queue, tile


@enum.unique
class _Bs(enum.IntEnum):
    """Positions in the backward stream hooks' operand list (``_backward_stream`` builds it per macrobatch,
    ``_run_stream`` hands it to every hook)."""

    D_Y_ROUTED_READY = 0
    REPLAYED_X_ROUTED_READY = 1
    D_X_ROUTED_READY = 2
    ROUTER_WEIGHTS = 3
    D_X_ROUTED = 4
    D_ROUTER_PARTIALS = 5
    D_Y_PEER_TABLE = 6
    X_PEER_TABLE = 7
    D_X_PEER_TABLE = 8
    D_ROUTER_PEER_TABLE = 9
    D_Y_BASE = 10
    D_Y_T_BASE = 11
    D_Y_SC_BASE = 12
    D_Y_SC_T_BASE = 13
    X_BASE = 14
    X_T_BASE = 15
    X_SC_BASE = 16
    X_SC_T_BASE = 17
    NUM_TOKENS = 18
    HIDDEN_SIZE = 19
    TOPK = 20
    MACROBATCH_SIZE = 21
    MINIBATCH_SIZE = 22
    MACRO = 23
    TILES = 24
    PULL_TASKS = 25
    RECIPROCAL_MINI = 26
    RECIPROCAL_TOPK = 27
    HIDDEN_BLOCKS = 28
    SWIGLU_COLUMNS = 29
    DONE_ADDRESS = 30
    MACRO_OFFSET = 31
    COUNT = 32


@cute.jit
def _is_cta_local(kind: Int32) -> Int32:
    """1 for the SwiGLU tasks, which need no cluster peer."""
    result = Int32(0)
    if kind == Int32(SHARED_SWIGLU_BWD) or kind == Int32(REPLAY_SWIGLU) or kind == Int32(ROUTED_SWIGLU_BWD):
        result = Int32(1)
    return result


def _tile_blocks(size: Int32, has_tail: bool) -> Int32:
    """The 256-wide tiles of an H / I extent.  ``has_tail`` (compile time): the extent may end in a
    128-wide tail, which is one more tile."""
    if has_tail:
        return _ceil_div_i32(size, Int32(MLP_TILE_N))
    return size // Int32(MLP_TILE_N)


def _d_tile_coord(row: Int32, cta_rank: Int32, column: Int32, row_tile_offset=None):
    """The D tile this CTA stores of cluster tile ``(row, column)``.  ``row_tile_offset`` (CTA tiles, None
    without one): the Up half of a combined FC1 main_grad."""
    cta_row = row * Int32(2) + cta_rank
    if row_tile_offset is not None:
        cta_row = cta_row + row_tile_offset
    return (cta_row, column, Int32(0), Int32(0))


def _wgrad_add_copy(collective, atom, tensor, epi_tile, sD, coord, desc_ptr=None):
    """A Wgrad's reduce-add of one D tile: the collective's own partition, or -- ``desc_ptr``, a split
    main_grad buffer -- the same partition through that expert's descriptor."""
    if desc_ptr is None:
        return gemm._Bf16Gemm.epilog_gmem_copy_and_partition(collective, atom, tensor, collective.cta_tile_shape_mnk[:2], epi_tile, sD, coord)[0]
    return _epilog_gmem_copy_fn(collective, atom, tensor, collective.cta_tile_shape_mnk[:2], epi_tile, sD, coord, tma_desc_ptr=desc_ptr)[0]


def _task_counts(
    num_local_tokens: Int32,
    shared_hidden_size: Int32,
    routed_hidden_size: Int32,
    intermediate_size: Int32,
    num_local_experts: Int32,
    minibatch_size: Int32,
    has_shared: bool = True,
    has_tail: bool = False,
):
    """The task counts (runtime Int32), as a tuple a jit function unpacks at its top level -- a
    Python object referenced inside a dynamic region would be flattened.

    ``has_shared=False`` (a routed-only layer): no shared task at all -- the kernel then passes
    0 shared rows, and the three shared Wgrad grids are dropped here.
    ``has_tail`` (compile time): the H / I tile counts include a 128-wide tail tile (``_tile_blocks``).

    ``(shared_down, shared_swiglu, shared_gate_up, shared, routed_down, routed_swiglu,
    routed_gate_up, routed, replay_gate_up, replay, weight_tiles, wgrad_matrix, wgrad)``

    ``shared_hidden_size`` is the shared expert's hidden (its Gate / Up dgrad tiles and ``weight_tiles``, the shared
    Wgrad matrices), ``routed_hidden_size`` the routed chain's (the routed Gate / Up dgrad tiles and the routed
    Wgrad matrices); the ordinary MoE passes the same value for both.
    """
    intermediate_blocks = _tile_blocks(intermediate_size, has_tail)
    shared_hidden_blocks = _tile_blocks(shared_hidden_size, has_tail)
    routed_hidden_blocks = _tile_blocks(routed_hidden_size, has_tail)
    swiglu_columns = intermediate_size // Int32(SWIGLU_TILE)
    weight_tiles = intermediate_blocks * shared_hidden_blocks
    routed_weight_tiles = intermediate_blocks * routed_hidden_blocks
    shared_row_blocks = num_local_tokens // Int32(MLP_TILE_M)
    minibatch_row_blocks = minibatch_size // Int32(MLP_TILE_M)
    minibatch_swiglu_tiles = minibatch_size // Int32(SWIGLU_TILE) * swiglu_columns
    shared_down = shared_row_blocks * intermediate_blocks
    shared_swiglu = _ceil_div_i32(num_local_tokens // Int32(SWIGLU_TILE) * swiglu_columns, Int32(CLUSTER_SIZE * SWIGLU_BWD_STAGES))
    shared_gate_up = shared_row_blocks * shared_hidden_blocks
    shared = shared_down + shared_swiglu + shared_gate_up + Int32(3) * weight_tiles
    if not has_shared:
        shared = Int32(0)
    routed_down = minibatch_row_blocks * intermediate_blocks
    routed_swiglu = _ceil_div_i32(minibatch_swiglu_tiles, Int32(CLUSTER_SIZE * SWIGLU_BWD_STAGES))
    routed_gate_up = minibatch_row_blocks * routed_hidden_blocks
    routed = routed_down + routed_swiglu + routed_gate_up
    replay_gate_up = minibatch_row_blocks * intermediate_blocks
    replay = Int32(2) * replay_gate_up + _ceil_div_i32(minibatch_swiglu_tiles, Int32(CLUSTER_SIZE * SWIGLU_FWD_STAGES))
    wgrad_matrix = num_local_experts * routed_weight_tiles
    return (
        shared_down,
        shared_swiglu,
        shared_gate_up,
        shared,
        routed_down,
        routed_swiglu,
        routed_gate_up,
        routed,
        replay_gate_up,
        replay,
        weight_tiles,
        wgrad_matrix,
        Int32(3) * wgrad_matrix,
    )


@cute.jit
def _expert_token_range(
    tokens_per_expert: cute.Tensor, expert_cu_end: cute.Tensor, expert_stride: Int32, task_decode: cutlass.Constexpr, expert: Int32
) -> tuple[Int32, Int32]:
    """``(begin, end)`` padded token rows of ``expert`` (the padded Wgrad K range), from the per-CTA row-block
    table: SMEM_BINARY reads the two neighbouring prefix entries; SMEM_GROUP reads the group's begin and
    sums the group's experts before this one from global ``tokens_per_expert``; LINEAR_SCAN sums
    ``tokens_per_expert`` from expert 0.  The counts are multiples of ``MLP_TILE_M``, so row blocks x
    MLP_TILE_M are exact token rows."""
    begin = Int32(0)
    if cutlass.const_expr(task_decode == TaskDecode.SMEM_BINARY):
        if expert > Int32(0):
            begin = expert_cu_end[expert - Int32(1)] * Int32(MLP_TILE_M)
        end = expert_cu_end[expert] * Int32(MLP_TILE_M)
    else:
        index = Int32(0)
        if cutlass.const_expr(task_decode == TaskDecode.SMEM_GROUP):
            group = expert // expert_stride
            if group > Int32(0):
                begin = expert_cu_end[group - Int32(1)] * Int32(MLP_TILE_M)
            index = group * expert_stride
        while index < expert:
            begin = begin + Int32(tokens_per_expert[index])
            index = index + Int32(1)
        end = begin + Int32(tokens_per_expert[expert])
    return begin, end


@cute.jit
def _write_expert_valid_ends(
    schedule_peer_rank: cute.Tensor, expert_cu_end: cute.Tensor, num_local_experts: Int32, task_decode: cutlass.Constexpr, thread: Int32
) -> Int32:
    """Per-expert end of the valid (non-padding) token rows, rounded up to the MXFP8 Wgrad K tile, written
    into the unused tail of the per-CTA expert table: entry ``num_local_experts + e``.  Returns that offset,
    or -1 when the table cannot hold a second set of entries (SMEM_GROUP or LINEAR_SCAN, or more than
    EXPERT_TABLE_CAP / 2 experts), in which case the Wgrad K range stays the padded range.  Entries are
    token rows, not row blocks.

    The valid end of one expert's segment is the BF16 backward's ``_padded_segment_valid_end`` (a binary
    search of the segment's last 256-row block for the first padding row), rounded up to the 128-row
    MXFP8 K tile.  The skipped K tiles hold only dispatch-zeroed rows of dY, x and the SwiGLU gradients,
    so dW is the padded range's up to the sign of a zero: a -0 partial sum stays -0 where the all-zero
    tiles would turn it into +0.  Ends with a CTA barrier."""
    offset = Int32(-1)
    if cutlass.const_expr(task_decode == TaskDecode.SMEM_BINARY):
        if num_local_experts * Int32(2) <= Int32(EXPERT_TABLE_CAP):
            offset = num_local_experts
            if thread < num_local_experts:
                begin = Int32(0)
                if thread > Int32(0):
                    begin = expert_cu_end[thread - Int32(1)] * Int32(MLP_TILE_M)
                end = expert_cu_end[thread] * Int32(MLP_TILE_M)
                valid_end = _padded_segment_valid_end(schedule_peer_rank, begin, end)
                expert_cu_end[num_local_experts + thread] = (valid_end + Int32(FP8_K_TILE - 1)) // Int32(FP8_K_TILE) * Int32(FP8_K_TILE)
    cute.arch.sync_threads()
    return offset


class _PersistentMxfp8BwdMega(_Mxfp8BwdBase):
    """The MXFP8 backward: shared GEMM tasks from the base class, routed tasks on e4m3."""

    # The forward's routed SwiGLU stage, reused by the replay.
    _swiglu_stage = _PersistentMxfp8FwdMega._swiglu_stage
    # All of the backward's communication runs on the forward's stream loop (``_run_stream``, called per macrobatch
    # by ``_backward_stream``) with this class's hooks (_stream_decode / _stream_schedule_row / _stream_ready_lane0
    # / _stream_issue / _stream_arrival / _stream_process; a third queue, STREAM_PULL_X), and reuses the forward's
    # pull of one block row per lane and per-warp quantization of a pulled block.
    _run_stream = _PersistentMxfp8FwdMega._run_stream
    _stream_pull = _PersistentMxfp8FwdMega._stream_pull
    _stream_quantize = _PersistentMxfp8FwdMega._stream_quantize
    # The transposed pass's store addresses in the form the backward is faster with (the forward's class comment).
    stream_lane_terms_in_32bit = False
    _quantize_hidden_tile = _PersistentMxfp8FwdMega._quantize_hidden_tile
    _store_hidden_bf16 = _PersistentMxfp8FwdMega._store_hidden_bf16
    # The routed Gate / Up context: BF16 here; e4m3 / scale pairs in _PersistentMxfp8BwdMegaE4m3Context,
    # whose kernel lists and scale list use the _BE4m3Context / _SE4m3Context positions.
    E4M3_CONTEXT = False
    POSITIONS = (_B, _S)

    def __init__(
        self,
        *,
        is_clamped: bool,
        task_decode: TaskDecode,
        native_columnwise: bool = False,
        split_weights: bool = False,
        accumulate_wgrad: bool = False,
        bf16_main_grad: bool = False,
        split_main_grads: bool = False,
        is_situ: bool = False,
        has_shared: bool = True,
        has_tail: bool = False,
        recompute_forward_context: bool = False,
    ) -> None:
        super().__init__(is_clamped=is_clamped)
        # H or I has a 128-wide tail (size % 256 != 0): one more tile in every count over it.
        # Compile time because H and I are model parameters and the flag only selects whether the tail code is
        # generated: an aligned model runs loops without predicates or tail branches.  It doubles this kernel's
        # executors (one for every aligned shape, one for every shape with a tail) as one more entry of the
        # executor key, which a model keeps for its whole run.  The values of H and I stay runtime arguments:
        # nothing is specialised on a size.
        self.has_tail = bool(has_tail)
        # Macrobatch 0 is replayed like every later macrobatch instead of read from the forward's routed context
        # (x, Gate, Up, Hidden), which is then uninitialized scratch on entry.  What a previous occupant of a ring
        # slot decides (the retirement waits and counts) is unchanged: macrobatch 0 still has none.
        self.recompute_forward_context = bool(recompute_forward_context)
        # False for a routed-only layer, whose shared tensors are placeholders the kernel builds
        # atoms over and never touches -- there are no shared tasks.
        self.has_shared = bool(has_shared)
        # SiTU-GLU instead of SwiGLU: the replay's forward activation and both backward ones; the
        # betas are runtime arguments.  Never with a clamp.
        self.is_situ = bool(is_situ)
        if self.is_situ and self.is_clamped:
            raise ValueError("SiTU cannot be combined with swiglu_limit")
        # Task -> expert decode, a compile-time attribute as in the BF16 forward (contract TaskDecode).
        self.task_decode = TaskDecode(task_decode)
        # The two hiddens are runtime tensor extents, as in the BF16 backward: the shared chain (x_shared,
        # d_y_shared, d_x_shared, the shared Wgrads) works in x_shared's width, the routed chain -- the dY /
        # replayed-x dispatch rows and their quantized layouts, the dX rows pushed back, the routed GEMMs' K / N,
        # the routed Wgrad matrices -- in d_x_routed's (equal for an ordinary MoE, different for the latent MoE of
        # functional.backward_latent_layer).  One compiled kernel serves both.
        self.routed = mxgemm._Mxfp8Gemm()
        # The routed dgrads read the weights transposed.  MOK's own quantizer transposes the payload,
        # so that operand is K-major; TransformerEngine leaves its columnwise payload in the
        # parameter's [E, M, K] order, so the same operand is MN-major instead.  A second collective
        # supplies the MN-major B's shared-memory layout, MMA atom and TMA atoms; everything else
        # (the A/B ring, the transaction count, the scales) is shared.
        self.native_columnwise = bool(native_columnwise)
        # The routed experts are independent per-expert allocations instead of one grouped
        # parameter.  The dgrad then reads one expert's own matrix and scale, each named by a TMA
        # descriptor, instead of slicing an [E, ...] parameter's L mode -- ten operands,
        # SPLIT_ROWS.  Compile-time: it changes their rank.
        self.split_weights = bool(split_weights)
        # As in the BF16 backward: the weight gradients go straight into the caller's main_grad
        # (float32, or BF16 with bf16_main_grad; per-expert buffers with split_main_grads), which
        # already holds earlier microbatches, so every tile is additive and an expert without rows
        # is left alone.
        self.accumulate_wgrad = bool(accumulate_wgrad)
        self.bf16_main_grad = bool(bf16_main_grad)
        self.split_main_grads = bool(split_main_grads)
        if (self.bf16_main_grad or self.split_main_grads) and not self.accumulate_wgrad:
            raise ValueError("a main_grad destination only exists for fused accumulation")
        self.fp32_main_grad = self.accumulate_wgrad and not self.bf16_main_grad
        if self.fp32_main_grad:
            self.atb = gemm._Bf16GemmFp32D()
        self.routed_mn = mxgemm._Mxfp8Gemm() if self.native_columnwise else self.routed

    @cute.jit
    def _preload_router_weights(
        self,
        router_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        router_weights: cute.Tensor,
        router_weights_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        num_tokens: Int32,
        macro: Int32,
        comm_cta: Int32,
        previous_required: Int32,
        macrobatch_size: Int32,
        num_comm_sms: Int32,
    ) -> None:
        """Loads macrobatch ``macro``'s router weights (0 for padding rows) once the previous
        macrobatch's buffers have retired (``previous_required`` arrivals), then a barrier over
        every communication CTA (the reverse-combine of dY reads all of them)."""
        thread = cute.arch.thread_idx()[0]
        if thread == Int32(0) and macro > Int32(0):
            _counter_wait_gpu(routed_buffers_done, macro - Int32(1), previous_required)
        cute.arch.sync_threads()
        rows = cutlass.min(Int32(macrobatch_size), num_tokens - macro * Int32(macrobatch_size))
        row = comm_cta * Int32(THREADS_PER_CTA) + thread
        while row < rows:
            global_row = macro * Int32(macrobatch_size) + row
            peer = schedule_peer_rank[global_row]
            value = Float32(0.0)
            if peer >= Int32(0):
                value = _load_peer_float(router_peer_table, peer, schedule_peer_token_idx[global_row])
            router_weights[row] = value
            row = row + num_comm_sms * Int32(THREADS_PER_CTA)
        cute.arch.sync_threads()
        if thread == Int32(0):
            _counter_arrive_release_gpu(router_weights_ready, macro)
            _counter_wait_gpu(router_weights_ready, macro, num_comm_sms)
        cute.arch.sync_threads()

    @cute.jit
    def _stream_decode(self, operands: list, task: Int32) -> tuple[Int32, Int32, Int32]:
        """``_run_stream``'s decode hook: stream task -> (queue, this macrobatch, 32-row tile),
        ``_backward_stream_task``."""
        queue, tile = _backward_stream_task(task, operands[_Bs.TILES], operands[_Bs.PULL_TASKS])
        return queue, operands[_Bs.MACRO], tile

    @cute.jit
    def _stream_schedule_row(self, operands: list, macro: Int32, row: Int32) -> Int32:
        """``_run_stream``'s schedule-row hook: this macrobatch's offset plus ``row`` (``macro`` is the stream's
        one macrobatch, an operand; the loop's copy is not read)."""
        return operands[_Bs.MACRO_OFFSET] + row

    @cute.jit
    def _stream_ready_lane0(self, operands: list, queue: Int32, macro: Int32, row: Int32) -> Int32:
        """``_run_stream``'s readiness hook, lane 0's answer whether a block waits for nothing more (the other
        lanes return 1): a pull of dY or of x never (the macrobatch's router weights are preloaded and the previous
        macrobatch's buffers have retired before its stream starts), a dX push for its minibatch's Gate / Up dgrad
        epilogues (``d_x_routed_ready``)."""
        ready = Int32(1)
        if cute.arch.lane_idx() == Int32(0) and queue == Int32(STREAM_PUSH):
            minibatch_size = operands[_Bs.MINIBATCH_SIZE]
            mini = _divide_u32(self._stream_schedule_row(operands, macro, row), minibatch_size, operands[_Bs.RECIPROCAL_MINI])
            mini_rows = cutlass.min(Int32(minibatch_size), operands[_Bs.NUM_TOKENS] - mini * minibatch_size)
            required = _ceil_div_i32(mini_rows, Int32(256)) * (operands[_Bs.HIDDEN_BLOCKS] * Int32(2))
            if cute.arch.load(operands[_Bs.D_X_ROUTED_READY].iterator + mini, Int32, sem="acquire", scope="gpu") < required:
                ready = Int32(0)
        return ready

    @cute.jit
    def _stream_issue(
        self, operands: list, queue: Int32, row: Int32, column_tile: Int32, buffer: cute.Pointer, mbarrier: cute.Pointer, lane: Int32, peer: Int32, route: Int32
    ) -> None:
        """``_run_stream``'s issue hook: a block into ``buffer`` on ``mbarrier`` (STREAM_ROWS arrivals per phase)
        with the lane's schedule entries (``peer``, ``route`` of the block's row ``lane``) already read.  A pull is
        the forward's (``_stream_pull``: one bulk copy of the peer's row ``route / topk`` per lane, a padding lane
        zero-filling its row) from the peer's dY buffer (STREAM_PULL) or x buffer (STREAM_PULL_X).  A push: the
        local dX row ``row + lane`` (``row`` macrobatch-local) by one bulk copy per valid lane; a padding lane
        arrives with no bytes."""
        hidden_size = operands[_Bs.HIDDEN_SIZE]
        if queue == Int32(STREAM_PUSH):
            chunk_bytes = _stream_row_bytes(hidden_size, column_tile, self.has_tail)
            if peer >= Int32(0):
                d_x_routed = operands[_Bs.D_X_ROUTED]
                source = d_x_routed.iterator.toint() + Int64(row + lane) * Int64(hidden_size) * Int64(2) + Int64(column_tile) * Int64(STREAM_COLUMNS * 2)
                cute.arch.mbarrier_arrive_and_expect_tx(mbarrier, chunk_bytes)
                tma_load_1d_raw(buffer + lane * Int32(STREAM_COLUMNS * 2), source, mbarrier, chunk_bytes)
            else:
                cute.arch.mbarrier_arrive(mbarrier)
        else:
            peer_buffer = Int64(0)  # a padding lane's (peer -1) is never read
            if peer >= Int32(0):
                if queue == Int32(STREAM_PULL):
                    peer_buffer = Int64(operands[_Bs.D_Y_PEER_TABLE][peer])
                else:
                    peer_buffer = Int64(operands[_Bs.X_PEER_TABLE][peer])
            # A padding lane's route is -1, outside _divide_u32's domain; its quotient is unused.
            self._stream_pull(
                buffer, mbarrier, lane, column_tile, peer, _divide_u32(route, operands[_Bs.TOPK], operands[_Bs.RECIPROCAL_TOPK]), hidden_size, peer_buffer
            )

    @cute.jit
    def _stream_arrival(self, operands: list, queue: Int32, macro: Int32, row: Int32) -> Int64:
        """``_run_stream``'s arrival-word hook: ``d_y_routed_ready[minibatch]`` (dY pull),
        ``replayed_x_routed_ready[minibatch]`` (x pull), ``routed_buffers_done[macrobatch]`` for a push before the
        last macrobatch (0 in the last)."""
        address = operands[_Bs.DONE_ADDRESS]
        if queue != Int32(STREAM_PUSH):
            mini = _divide_u32(self._stream_schedule_row(operands, macro, row), operands[_Bs.MINIBATCH_SIZE], operands[_Bs.RECIPROCAL_MINI])
            address = (operands[_Bs.REPLAYED_X_ROUTED_READY].iterator + mini).toint()
            if queue == Int32(STREAM_PULL):
                address = (operands[_Bs.D_Y_ROUTED_READY].iterator + mini).toint()
        return address

    @cute.jit
    def _stream_process(
        self,
        operands: list,
        queue: Int32,
        macro: Int32,
        row: Int32,
        column: Int32,
        buffer: cute.Pointer,
        lane: Int32,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
    ) -> None:
        """``_run_stream``'s process hook, the block's bytes landed in ``buffer``.  A push block: one bulk store per
        lane to the peer's dX row ``route`` (column block 0 also sums the row's router-weight partials into the
        peer's d_router entry), the buffer free after ``wait_group read``.  A dY pull block: the lane's row scaled
        in place by its router weight (``bf16(x * w)``), then the forward's per-warp quantization
        (``_stream_quantize``, direct global stores into the dY tiles).  An x pull block: the quantization alone,
        into the replayed-x tiles."""
        hidden_size = operands[_Bs.HIDDEN_SIZE]
        macrobatch_size = operands[_Bs.MACROBATCH_SIZE]
        if queue == Int32(STREAM_PUSH):
            schedule_row = self._stream_schedule_row(operands, macro, row) + lane
            peer = schedule_peer_rank[schedule_row]
            if peer >= Int32(0):
                route = schedule_peer_token_idx[schedule_row]
                chunk_bytes = _stream_row_bytes(hidden_size, column, self.has_tail)
                destination = (
                    _peer_address(operands[_Bs.D_X_PEER_TABLE], peer) + Int64(route) * Int64(hidden_size) * Int64(2) + Int64(column) * Int64(STREAM_COLUMNS * 2)
                )
                tma_store_1d_raw(destination, buffer + lane * Int32(STREAM_COLUMNS * 2), chunk_bytes)
                if column == Int32(0):
                    d_router_partials = operands[_Bs.D_ROUTER_PARTIALS]
                    d_router = Float32(0.0)
                    for partial in cutlass.range(operands[_Bs.SWIGLU_COLUMNS]):
                        d_router = d_router + Float32(d_router_partials[row + lane, partial])
                    _store_peer_float(operands[_Bs.D_ROUTER_PEER_TABLE], peer, route, d_router)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            if queue == Int32(STREAM_PULL):
                # The lane's row times its router weight, in place (rounded to BF16: bf16(x * w)); the 16 B
                # chunks staggered by lane so the 512 B row stride stays conflict-free.  Eight chunks are
                # loaded before any of them is scaled and stored: the shared-memory accessors are ordered
                # asm, so a load-scale-store chain per chunk would serialize on the load latency.
                weight = Float32(operands[_Bs.ROUTER_WEIGHTS][row + lane])
                row_address = buffer.toint() + lane * Int32(STREAM_COLUMNS * 2)
                for batch in cutlass.range_constexpr(STREAM_COLUMNS * 2 // 16 // 8):
                    loaded = []
                    for j in cutlass.range_constexpr(8):
                        chunk = (Int32(batch * 8 + j) + lane) & Int32(STREAM_COLUMNS * 2 // 16 - 1)
                        loaded.append(_load_shared_v4(row_address + chunk * Int32(16)))
                    for j in cutlass.range_constexpr(8):
                        chunk = (Int32(batch * 8 + j) + lane) & Int32(STREAM_COLUMNS * 2 // 16 - 1)
                        scaled = [
                            cvt_bf16x2((w_ << Uint32(16)).bitcast(Float32) * weight, (w_ & Uint32(0xFFFF0000)).bitcast(Float32) * weight) for w_ in loaded[j]
                        ]
                        _store_shared_v4_words(row_address + chunk * Int32(16), scaled)
                # Generic stores into a buffer the warp's next issue refills by bulk copy (async proxy): the
                # proxy fence the zero-filled rows get (_stream_pull).
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if cutlass.const_expr(self.has_tail):
                    # One branch per block, the same for the whole warp: the last column block of a hidden with
                    # a 128-wide tail quantizes its 128 columns, every full block all STREAM_COLUMNS (as the
                    # forward's stream, and for the x pull below).
                    if (column + Int32(1)) * Int32(STREAM_COLUMNS) > hidden_size:
                        self._stream_quantize(
                            buffer,
                            lane,
                            row // Int32(STREAM_ROWS),
                            column,
                            hidden_size,
                            macrobatch_size,
                            [operands[_Bs.D_Y_BASE], operands[_Bs.D_Y_T_BASE], operands[_Bs.D_Y_SC_BASE], operands[_Bs.D_Y_SC_T_BASE]],
                            tail=True,
                        )
                    else:
                        self._stream_quantize(
                            buffer,
                            lane,
                            row // Int32(STREAM_ROWS),
                            column,
                            hidden_size,
                            macrobatch_size,
                            [operands[_Bs.D_Y_BASE], operands[_Bs.D_Y_T_BASE], operands[_Bs.D_Y_SC_BASE], operands[_Bs.D_Y_SC_T_BASE]],
                        )
                else:
                    self._stream_quantize(
                        buffer,
                        lane,
                        row // Int32(STREAM_ROWS),
                        column,
                        hidden_size,
                        macrobatch_size,
                        [operands[_Bs.D_Y_BASE], operands[_Bs.D_Y_T_BASE], operands[_Bs.D_Y_SC_BASE], operands[_Bs.D_Y_SC_T_BASE]],
                    )
            else:
                if cutlass.const_expr(self.has_tail):
                    if (column + Int32(1)) * Int32(STREAM_COLUMNS) > hidden_size:
                        self._stream_quantize(
                            buffer,
                            lane,
                            row // Int32(STREAM_ROWS),
                            column,
                            hidden_size,
                            macrobatch_size,
                            [operands[_Bs.X_BASE], operands[_Bs.X_T_BASE], operands[_Bs.X_SC_BASE], operands[_Bs.X_SC_T_BASE]],
                            tail=True,
                        )
                    else:
                        self._stream_quantize(
                            buffer,
                            lane,
                            row // Int32(STREAM_ROWS),
                            column,
                            hidden_size,
                            macrobatch_size,
                            [operands[_Bs.X_BASE], operands[_Bs.X_T_BASE], operands[_Bs.X_SC_BASE], operands[_Bs.X_SC_T_BASE]],
                        )
                else:
                    self._stream_quantize(
                        buffer,
                        lane,
                        row // Int32(STREAM_ROWS),
                        column,
                        hidden_size,
                        macrobatch_size,
                        [operands[_Bs.X_BASE], operands[_Bs.X_T_BASE], operands[_Bs.X_SC_BASE], operands[_Bs.X_SC_T_BASE]],
                    )

    @cute.jit
    def _backward_stream(
        self,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        d_y_routed_ready: cute.Tensor,
        replayed_x_routed_ready: cute.Tensor,
        d_x_routed_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        router_weights: cute.Tensor,
        d_x_routed: cute.Tensor,
        d_router_partials: cute.Tensor,
        d_y_peer_table: cute.Tensor,
        x_peer_table: cute.Tensor,
        d_x_peer_table: cute.Tensor,
        d_router_peer_table: cute.Tensor,
        arena: cute.Pointer,
        stream_mbarriers: cute.Pointer,
        num_tokens: Int32,
        hidden_size: Int32,
        intermediate_size: Int32,
        topk: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        macro: Int32,
        last_macro: Int32,
        comm_cta: Int32,
        num_comm_sms: Int32,
        d_y_bases: list,
        x_bases: list,
        phase_bits: Int32,
    ) -> Int32:
        """One macrobatch's communication -- the reverse-combine of dY, in a replayed macrobatch the replay
        dispatch of x, and the reverse-dispatch of dX (with the router-weight gradient) -- on the forward's stream
        loop (``_run_stream``) with the backward's hooks; called once per macrobatch by ``_run_mxfp8_comm_prefix``
        after the macrobatch's router weights are preloaded (which waits for the previous macrobatch's buffers to
        retire).  ``phase_bits`` carries the warp's buffer-barrier phases from one macrobatch to the next (the
        barriers are initialized once, by the prefix).

        Stream tasks (``_backward_stream_task``): the pulls first (macrobatch 0: its dY pulls in tile order; a
        replayed macrobatch, which macrobatch 0 is with ``recompute_forward_context``: tile ``t``'s x pull then its dY
        pull), then the dX pushes in tile order; the head word is this macrobatch's among the extra last elements of
        ``d_y_routed_ready``.  A dY pull block arrives on ``d_y_routed_ready[minibatch]``, an x pull block on
        ``replayed_x_routed_ready[minibatch]`` (the arrivals batch a minibatch's worth in a macrobatch that pulls
        dY alone, one group's worth in a replayed macrobatch, where x and dY blocks alternate); before the last
        macrobatch each push block counts on ``routed_buffers_done[macrobatch]``.
        Every warp has released its pending arrivals when this returns: thread 0 waits for routed_buffers_done in
        the next macrobatch's router-weight preload, and this CTA's own push arrivals count there."""
        macro_offset = macro * Int32(macrobatch_size)
        rows = cutlass.min(Int32(macrobatch_size), num_tokens - macro_offset)
        # 32-row tiles: num_tokens and the macrobatch are multiples of STREAM_ROWS (the schedule pads every expert to
        # 256 rows, the macrobatch is a multiple of 256), so the floor here and the consumers' ceil agree.
        tiles = rows // Int32(STREAM_ROWS)
        pull_tasks = tiles
        if cutlass.const_expr(self.recompute_forward_context):
            pull_tasks = Int32(2) * tiles  # macrobatch 0 pulls its x as well
        else:
            if macro > Int32(0):
                pull_tasks = Int32(2) * tiles
        column_blocks = _ceil_div_i32(hidden_size, Int32(STREAM_COLUMNS))
        group_blocks = Int32(STREAM_GROUP_BLOCKS)
        groups = _ceil_div_i32(column_blocks, group_blocks)
        blocks = (pull_tasks + tiles) * groups
        head = cute.make_ptr(
            Int32, (d_y_routed_ready.iterator + (d_y_routed_ready.shape[0] - Int32(1) - macro)).toint(), cute.AddressSpace.gmem, assumed_align=4
        )
        reciprocal_groups = _reciprocal_u32(groups)
        reciprocal_group = _reciprocal_u32(group_blocks)
        reciprocal_mini = _reciprocal_u32(minibatch_size)
        reciprocal_topk = _reciprocal_u32(topk)
        hidden_blocks = _tile_blocks(hidden_size, self.has_tail)  # the dX push waits for an H tail's dgrad tile too
        swiglu_columns = intermediate_size // Int32(128)
        done_address = Int64(0)  # the push blocks' arrival word: routed_buffers_done[macro] before the last macrobatch
        if macro < last_macro:
            done_address = (routed_buffers_done.iterator + macro).toint()
        operands = [None] * _Bs.COUNT  # the hooks' operands, by position
        operands[_Bs.D_Y_ROUTED_READY] = d_y_routed_ready
        operands[_Bs.REPLAYED_X_ROUTED_READY] = replayed_x_routed_ready
        operands[_Bs.D_X_ROUTED_READY] = d_x_routed_ready
        operands[_Bs.ROUTER_WEIGHTS] = router_weights
        operands[_Bs.D_X_ROUTED] = d_x_routed
        operands[_Bs.D_ROUTER_PARTIALS] = d_router_partials
        operands[_Bs.D_Y_PEER_TABLE] = d_y_peer_table
        operands[_Bs.X_PEER_TABLE] = x_peer_table
        operands[_Bs.D_X_PEER_TABLE] = d_x_peer_table
        operands[_Bs.D_ROUTER_PEER_TABLE] = d_router_peer_table
        operands[_Bs.D_Y_BASE], operands[_Bs.D_Y_T_BASE], operands[_Bs.D_Y_SC_BASE], operands[_Bs.D_Y_SC_T_BASE] = d_y_bases
        operands[_Bs.X_BASE], operands[_Bs.X_T_BASE], operands[_Bs.X_SC_BASE], operands[_Bs.X_SC_T_BASE] = x_bases
        operands[_Bs.NUM_TOKENS] = num_tokens
        operands[_Bs.HIDDEN_SIZE] = hidden_size
        operands[_Bs.TOPK] = topk
        operands[_Bs.MACROBATCH_SIZE] = macrobatch_size
        operands[_Bs.MINIBATCH_SIZE] = minibatch_size
        operands[_Bs.MACRO] = macro
        operands[_Bs.TILES] = tiles
        operands[_Bs.PULL_TASKS] = pull_tasks
        operands[_Bs.RECIPROCAL_MINI] = reciprocal_mini
        operands[_Bs.RECIPROCAL_TOPK] = reciprocal_topk
        operands[_Bs.HIDDEN_BLOCKS] = hidden_blocks
        operands[_Bs.SWIGLU_COLUMNS] = swiglu_columns
        operands[_Bs.DONE_ADDRESS] = done_address
        operands[_Bs.MACRO_OFFSET] = macro_offset
        assert all(operand is not None for operand in operands)
        return self._run_stream(
            operands,
            schedule_peer_rank,
            schedule_peer_token_idx,
            head,
            arena,
            stream_mbarriers,
            blocks,
            groups,
            reciprocal_groups,
            reciprocal_group,
            column_blocks,
            comm_cta,
            num_comm_sms,
            phase_bits,
        )

    @cute.jit
    def _run_mxfp8_comm_prefix(
        self,
        x_peer_table: cute.Tensor,
        d_y_peer_table: cute.Tensor,
        d_x_peer_table: cute.Tensor,
        router_peer_table: cute.Tensor,
        d_router_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens: Int32,
        router_weights: cute.Tensor,
        d_x_routed: cute.Tensor,
        d_router_partials: cute.Tensor,
        router_weights_ready: cute.Tensor,
        d_y_routed_ready: cute.Tensor,
        d_x_routed_ready: cute.Tensor,
        replayed_x_routed_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        arena: cute.Pointer,
        stream_mbarriers: cute.Pointer,
        cluster_index: Int32,
        cta_rank: Int32,
        d_y_quant: list,
        x_quant: list,
        num_local_tokens: Int32,
        shared_hidden_size: Int32,
        routed_hidden_size: Int32,
        intermediate_size: Int32,
        num_local_experts: Int32,
        topk: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        num_comm_sms: Int32,
    ) -> None:
        """The communication CTAs' branch of the kernel: per macrobatch, its router weights
        (``_preload_router_weights``: after the previous macrobatch's routed buffers retire, then a barrier over
        every communication CTA), then its communication on the dynamic task stream (``_backward_stream``: the
        reverse-combine of dY, in a replayed macrobatch the replay dispatch of x, and the reverse-dispatch of
        dX).  Every row moved here is a routed-chain row (``routed_hidden_size``
        columns); ``shared_hidden_size`` only sizes the shared task counts.  ``d_y_quant`` / ``x_quant``: the plain
        row-major e4m3 outputs (normal, transposed) and the two scale tensors of dY / of the replayed x."""
        _, _, _, _, _, _, _, routed, _, _, _, _, wgrad = _task_counts(
            num_local_tokens, shared_hidden_size, routed_hidden_size, intermediate_size, num_local_experts, minibatch_size, self.has_shared, self.has_tail
        )
        thread = cute.arch.thread_idx()[0]
        comm_cta = cluster_index * Int32(CLUSTER_SIZE) + cta_rank
        if thread == Int32(0):
            for i in cutlass.range_constexpr(STREAM_WARPS * STREAM_BUFFERS):
                cute.arch.mbarrier_init(stream_mbarriers + Int32(i), STREAM_ROWS)  # one arrival per pulled row
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        d_y_bases = [tensor.iterator.toint() for tensor in d_y_quant]
        x_bases = [tensor.iterator.toint() for tensor in x_quant]
        column_blocks = _ceil_div_i32(routed_hidden_size, Int32(STREAM_COLUMNS))
        macro_count = _ceil_div_i32(num_tokens, Int32(macrobatch_size))
        last_macro = macro_count - Int32(1)
        phase_bits = Int32(0)
        for macro in cutlass.range(macro_count, unroll=1):
            # The previous macrobatch's routed buffers retire when its routed and Wgrad tasks (CLUSTER_SIZE arrivals
            # each) and its stream's push blocks (one each) have all arrived on routed_buffers_done.
            required = Int32(0)
            if macro > Int32(0):
                previous_rows = cutlass.min(Int32(macrobatch_size), num_tokens - (macro - Int32(1)) * Int32(macrobatch_size))
                required = (
                    Int32(CLUSTER_SIZE) * (_ceil_div_i32(previous_rows, Int32(minibatch_size)) * routed + wgrad)
                    + previous_rows // Int32(STREAM_ROWS) * column_blocks
                )
            self._preload_router_weights(
                router_peer_table,
                schedule_peer_rank,
                schedule_peer_token_idx,
                router_weights,
                router_weights_ready,
                routed_buffers_done,
                num_tokens,
                macro,
                comm_cta,
                required,
                macrobatch_size,
                num_comm_sms,
            )
            phase_bits = self._backward_stream(
                schedule_peer_rank,
                schedule_peer_token_idx,
                d_y_routed_ready,
                replayed_x_routed_ready,
                d_x_routed_ready,
                routed_buffers_done,
                router_weights,
                d_x_routed,
                d_router_partials,
                d_y_peer_table,
                x_peer_table,
                d_x_peer_table,
                d_router_peer_table,
                arena,
                stream_mbarriers,
                num_tokens,
                routed_hidden_size,
                intermediate_size,
                topk,
                macrobatch_size,
                minibatch_size,
                macro,
                last_macro,
                comm_cta,
                num_comm_sms,
                d_y_bases,
                x_bases,
                phase_bits,
            )

    @cute.jit
    def _run_replay_swiglu(
        self,
        local: Int32,
        generation: Int32,
        cta_rank: Int32,
        num_tokens: Int32,
        load_atoms: list,
        load_tensors: list,
        smem_layout: cute.ComposedLayout,
        arena: cute.Pointer,
        barriers: cute.Pointer,
        phase_bits: Int32,
        replayed_gate_up_ready: cute.Tensor,
        replayed_hidden_ready: cute.Tensor,
        intermediate_size: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        swiglu_limit: Float32,
        quant_outputs: list,
        situ_beta1=None,
        situ_beta2=None,
    ) -> Int32:
        """The REPLAY_SWIGLU task of a replayed macrobatch: the forward's routed SwiGLU stage on the
        replayed BF16 Gate / Up ring, with the backward's counters (base 0).  Waits for the tile's
        replayed Gate / Up epilogues; arrives on ``replayed_hidden_ready`` after its stores complete.
        """
        thread = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        swiglu_columns = intermediate_size // Int32(SWIGLU_TILE)
        sg = cute.make_tensor(cute.recast_ptr(arena, smem_layout.inner, dtype=BFloat16), smem_layout.outer)
        su = cute.make_tensor(cute.recast_ptr(arena + SWIGLU_FWD_STAGES * BF16_TILE_BYTES, smem_layout.inner, dtype=BFloat16), smem_layout.outer)
        hidden_layout = cute.select(smem_layout, mode=[0, 1])
        sh = cute.make_tensor(cute.recast_ptr(arena + 2 * SWIGLU_FWD_STAGES * BF16_TILE_BYTES, hidden_layout.inner, dtype=BFloat16), hidden_layout.outer)
        base_tile = (local * Int32(CLUSTER_SIZE) + cta_rank) * Int32(SWIGLU_FWD_STAGES)
        local_limit = Int32(minibatch_size // SWIGLU_TILE) * swiglu_columns
        global_base = generation * local_limit
        for stage in cutlass.range_constexpr(SWIGLU_FWD_STAGES):
            tile = base_tile + Int32(stage)
            global_row = (global_base + tile) // swiglu_columns
            column = (global_base + tile) % swiglu_columns
            valid = tile < local_limit and global_row * Int32(SWIGLU_TILE) < num_tokens
            if valid and thread == Int32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(barriers + stage, Int32(2 * BF16_TILE_BYTES))
                _counter_wait_gpu(
                    replayed_gate_up_ready,
                    (global_row // Int32(2)) * _tile_blocks(intermediate_size, self.has_tail) + column // Int32(2),
                    Int32(2 * CLUSTER_SIZE),
                )
            if valid and warp == Int32(0):
                cute.arch.sync_warp()
                row = global_row % Int32(macrobatch_size // SWIGLU_TILE)
                _issue_swiglu_tma_load(load_atoms[0], load_tensors[0], row, column, sg[None, None, stage], barriers + stage)
                _issue_swiglu_tma_load(load_atoms[1], load_tensors[1], row, column, su[None, None, stage], barriers + stage)
        for stage in cutlass.range(SWIGLU_FWD_STAGES, unroll=1):
            phase_bits = self._swiglu_stage(
                True,
                stage,
                Int32(FWD_ROUTED_SWIGLU_KIND),
                base_tile,
                global_base,
                local_limit,
                num_tokens,
                barriers,
                phase_bits,
                sg,
                su,
                sh,
                arena,
                thread,
                swiglu_limit,
                intermediate_size,
                macrobatch_size,
                [],
                [],
                quant_outputs,
                situ_beta1,
                situ_beta2,
            )
        if thread == Int32(0):
            cute.arch.cp_async_bulk_wait_group(0, read=False)
            for stage in cutlass.range_constexpr(SWIGLU_FWD_STAGES):
                tile = base_tile + Int32(stage)
                global_row = (global_base + tile) // swiglu_columns
                if tile < local_limit and global_row * Int32(SWIGLU_TILE) < num_tokens:
                    _counter_arrive_release_gpu(replayed_hidden_ready, global_row // Int32(2))
        return phase_bits

    @cute.jit
    def _minibatches_of(self, num_tokens: Int32, macro: Int32, macrobatch_size: Int32, minibatch_size: Int32) -> Int32:
        rows = cutlass.min(Int32(macrobatch_size), num_tokens - macro * Int32(macrobatch_size))
        return _ceil_div_i32(rows, Int32(minibatch_size))

    @cute.jit
    def _true_clusters(
        self,
        comm_clusters: Int32,
        num_tokens: Int32,
        num_local_tokens: Int32,
        shared_hidden_size: Int32,
        routed_hidden_size: Int32,
        intermediate_size: Int32,
        num_local_experts: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
    ) -> Int32:
        """The clusters that have work for the actual token count (comm + shared + routed + replay +
        Wgrad); the launch grid is sized for the schedule's capacity."""
        _, _, _, shared, _, _, _, routed, _, replay, _, _, wgrad = _task_counts(
            num_local_tokens, shared_hidden_size, routed_hidden_size, intermediate_size, num_local_experts, minibatch_size, self.has_shared, self.has_tail
        )
        minis = _ceil_div_i32(num_tokens, Int32(minibatch_size))
        macros = _ceil_div_i32(num_tokens, Int32(macrobatch_size))
        if cutlass.const_expr(self.recompute_forward_context):
            replay_minis = minis  # macrobatch 0's minibatches run the replay tasks too
        else:
            replay_minis = minis - self._minibatches_of(num_tokens, Int32(0), macrobatch_size, minibatch_size)
        return comm_clusters + shared + minis * routed + replay_minis * replay + macros * wgrad

    @cute.jit
    def _task_table(
        self,
        num_tokens: Int32,
        num_local_tokens: Int32,
        shared_hidden_size: Int32,
        routed_hidden_size: Int32,
        intermediate_size: Int32,
        num_local_experts: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
    ):
        """The per-launch constants of the task table, computed once per CTA before the task loop:
        the task counts of ``_task_counts``, the minibatches per macrobatch, the number of
        macrobatches, the minibatches of the first and of the last macrobatch, and the tasks of
        macrobatch 0 and of a full replayed macrobatch.  ``_decode_mxfp8_task`` then spends at most
        two runtime divisions per task.  With ``recompute_forward_context`` no macrobatch has the saved layout
        (``saved_tasks`` is 0): the replayed macrobatches start at macrobatch 0."""
        shared_down, shared_swiglu, _, shared, routed_down, routed_swiglu, _, routed, replay_gate_up, replay, weight_tiles, wgrad_matrix, wgrad = _task_counts(
            num_local_tokens, shared_hidden_size, routed_hidden_size, intermediate_size, num_local_experts, minibatch_size, self.has_shared, self.has_tail
        )
        minibatches_per_macrobatch = macrobatch_size // minibatch_size
        num_macros = _ceil_div_i32(num_tokens, Int32(macrobatch_size))
        first_minis = self._minibatches_of(num_tokens, Int32(0), macrobatch_size, minibatch_size)
        last_minis = self._minibatches_of(num_tokens, cutlass.max(num_macros - Int32(1), Int32(0)), macrobatch_size, minibatch_size)
        saved_tasks = Int32(0) if cutlass.const_expr(self.recompute_forward_context) else first_minis * routed + wgrad
        replayed_tasks = minibatches_per_macrobatch * (replay + routed) + wgrad
        return (
            shared_down,
            shared_swiglu,
            shared,
            routed_down,
            routed_swiglu,
            routed,
            replay_gate_up,
            replay,
            weight_tiles,
            wgrad_matrix,
            minibatches_per_macrobatch,
            num_macros,
            first_minis,
            last_minis,
            saved_tasks,
            replayed_tasks,
        )

    @cute.jit
    def _decode_mxfp8_task(
        self,
        task: Int32,
        shared_down: Int32,
        shared_swiglu: Int32,
        shared: Int32,
        routed_down: Int32,
        routed_swiglu: Int32,
        routed: Int32,
        replay_gate_up: Int32,
        replay: Int32,
        weight_tiles: Int32,
        wgrad_matrix: Int32,
        minibatches_per_macrobatch: Int32,
        num_macros: Int32,
        first_minis: Int32,
        last_minis: Int32,
        saved_tasks: Int32,
        replayed_tasks: Int32,
    ):
        """``(kind, macro, generation, local)`` of compute task ``task``, from the constants of
        ``_task_table``.

        Task order: the shared tasks; then macrobatch 0's routed backward and Wgrad; then per
        replayed macrobatch its minibatches' replay (Gate, Up, SwiGLU), their routed backward, and
        its Wgrad.  With ``recompute_forward_context`` macrobatch 0 is the first replayed macrobatch (``saved_tasks``
        is 0).  ``generation`` is the global minibatch.  Each quotient's remainder is formed by
        a multiply and subtract, so a task costs at most two runtime divisions.
        """
        kind = Int32(SHARED_DOWN_DGRAD)
        macro = Int32(0)
        generation = Int32(0)
        local = task
        if task < shared_down:
            kind = Int32(SHARED_DOWN_DGRAD)
        elif task < shared_down + shared_swiglu:
            kind = Int32(SHARED_SWIGLU_BWD)
            local = task - shared_down
        elif task < shared - Int32(3) * weight_tiles:
            kind = Int32(SHARED_GATE_UP_DGRAD)
            local = task - shared_down - shared_swiglu
        elif task < shared:
            local = task - (shared - Int32(3) * weight_tiles)
            matrix = local // weight_tiles
            kind = Int32(SHARED_WGRAD_DOWN) + matrix
            local = local - matrix * weight_tiles
        else:
            index = task - shared
            macro_minis = first_minis
            replay_count = Int32(0)
            if index >= saved_tasks:
                index = index - saved_tasks
                step = index // replayed_tasks
                macro = (Int32(0) if cutlass.const_expr(self.recompute_forward_context) else Int32(1)) + step
                index = index - step * replayed_tasks
                # Every macrobatch but the last is full (macrobatch_size is a multiple of minibatch_size).
                macro_minis = minibatches_per_macrobatch
                if macro == num_macros - Int32(1):
                    macro_minis = last_minis
                replay_count = macro_minis * replay
            if index < replay_count:
                mini = index // replay
                generation = macro * minibatches_per_macrobatch + mini
                local = index - mini * replay
                if local < replay_gate_up:
                    kind = Int32(REPLAY_GATE)
                elif local < Int32(2) * replay_gate_up:
                    kind = Int32(REPLAY_UP)
                    local = local - replay_gate_up
                else:
                    kind = Int32(REPLAY_SWIGLU)
                    local = local - Int32(2) * replay_gate_up
            else:
                index = index - replay_count
                routed_count = macro_minis * routed
                if index < routed_count:
                    mini = index // routed
                    generation = macro * minibatches_per_macrobatch + mini
                    local = index - mini * routed
                    if local < routed_down:
                        kind = Int32(ROUTED_DOWN_DGRAD)
                    elif local < routed_down + routed_swiglu:
                        kind = Int32(ROUTED_SWIGLU_BWD)
                        local = local - routed_down
                    else:
                        kind = Int32(ROUTED_GATE_UP_DGRAD)
                        local = local - routed_down - routed_swiglu
                else:
                    local = index - routed_count
                    matrix = local // wgrad_matrix
                    kind = Int32(ROUTED_WGRAD_DOWN) + matrix
                    local = local - matrix * wgrad_matrix
        return kind, macro, generation, local

    @cute.jit
    def _routed_gemm_geometry(
        self,
        kind: Int32,
        macro: Int32,
        generation: Int32,
        local: Int32,
        num_tokens: Int32,
        tokens_per_expert: cute.Tensor,
        expert_cu_end: cute.Tensor,
        expert_stride: Int32,
        expert_groups: Int32,
        hidden_size: Int32,
        intermediate_size: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        valid_end_offset: Int32,
    ):
        """``(valid, expert, row, column, k_start, k_end, first)`` of a routed GEMM task.

        The expert comes from the BF16 forward's compile-time decode (``TaskDecode``) over the
        per-CTA row-block table.  Replay / dgrad tasks: the minibatch's row blocks
        that fall in each expert, in order, each a swizzled (row, column) grid; ``row`` is
        macrobatch-local.  Wgrad: ``local`` = expert x (row_blocks x col_blocks); K is the expert's
        token rows inside this macrobatch -- the padded range, cut at the expert's valid end
        (``_write_expert_valid_ends``, whole K tiles of zero padding rows; a non-empty expert keeps at
        least one tile) when ``valid_end_offset`` >= 0; the first macrobatch an expert appears in
        stores, later ones accumulate.
        """
        intermediate_blocks = _tile_blocks(intermediate_size, self.has_tail)
        hidden_blocks = _tile_blocks(hidden_size, self.has_tail)
        valid = Int32(0)
        expert = Int32(-1)
        row = Int32(0)
        column = Int32(0)
        k_start = Int32(0)
        k_end = Int32(0)
        first = Int32(0)
        if kind >= Int32(ROUTED_WGRAD_DOWN):
            # Wgrad-down: M = hidden, N = intermediate; Wgrad-gate / -up: M = intermediate, N = hidden.
            row_blocks = intermediate_blocks
            col_blocks = hidden_blocks
            if kind == Int32(ROUTED_WGRAD_DOWN):
                row_blocks = hidden_blocks
                col_blocks = intermediate_blocks
            expert = local // (row_blocks * col_blocks)
            expert_begin, expert_end = _expert_token_range(tokens_per_expert, expert_cu_end, expert_stride, self.task_decode, expert)
            if valid_end_offset >= Int32(0):
                # The valid-end entries are token rows (the first entries are 256-row blocks): the K range
                # ends at the expert's last valid row rounded up to the K tile.  The padding rows skipped
                # are dispatch-zeroed, so the sum differs from the padded range's only on signed zero (a
                # -0 partial sum stays -0 where the +0 tiles would turn it into +0).  The min keeps one K
                # tile for a non-empty expert; the round-up already guarantees it (defensive).
                expert_end = cutlass.max(expert_cu_end[valid_end_offset + expert], cutlass.min(expert_begin + Int32(FP8_K_TILE), expert_end))
            k_start = cutlass.max(expert_begin, macro * Int32(macrobatch_size))
            k_end = cutlass.min(expert_end, cutlass.min((macro + Int32(1)) * Int32(macrobatch_size), num_tokens))
            first = Int32(k_start == expert_begin)
            if k_start < k_end:
                valid = Int32(1)
                tile_row, tile_column = _swizzled_tile_i32(row_blocks, col_blocks, local % (row_blocks * col_blocks))
                row = tile_row
                column = tile_column
        else:
            col_blocks = intermediate_blocks
            if kind == Int32(ROUTED_GATE_UP_DGRAD):
                col_blocks = hidden_blocks
            mini_rows = Int32(minibatch_size // MLP_TILE_M)
            owned, owner, row_block, tile_column, _ = _routed_expert_tile_i32(
                tokens_per_expert, expert_cu_end, expert_stride, expert_groups, self.task_decode, generation * mini_rows, mini_rows, col_blocks, local
            )
            if owned != Int32(0):
                valid = Int32(1)
                expert = owner
                row = row_block - macro * Int32(macrobatch_size // MLP_TILE_M)
                column = tile_column
        return valid, expert, row, column, k_start, k_end, first

    @cute.jit
    def _wait_routed_wgrad_operands(
        self,
        kind: Int32,
        macro: Int32,
        wait_row_block: Int32,
        wait_minibatch: Int32,
        row: Int32,
        num_tokens: Int32,
        d_y_routed_ready: cute.Tensor,
        d_gate_up_ready: cute.Tensor,
        replayed_x_routed_ready: cute.Tensor,
        replayed_hidden_ready: cute.Tensor,
        num_local_tokens: Int32,
        hidden_size: Int32,
        intermediate_size: Int32,
        minibatch_size: Int32,
    ):
        """The operand waits of a routed Wgrad K tile (``wait_row_block`` / ``wait_minibatch``: the
        tile is the first to touch a 256-row block / a minibatch).

        ``row`` is the global token row the K tile starts at.  Wgrad-down: dY's minibatch, and
        in a replayed macrobatch the replayed Hidden's row block; Wgrad-gate / -up: the row
        block's SwiGLU-backward, and in a replayed macrobatch the replayed x's minibatch.
        With ``recompute_forward_context`` macrobatch 0 is a replayed macrobatch.
        """
        swiglu_columns = intermediate_size // Int32(SWIGLU_TILE)
        if wait_row_block != Int32(0):
            if kind == Int32(ROUTED_WGRAD_DOWN):
                if cutlass.const_expr(self.recompute_forward_context):
                    _counter_wait_gpu(replayed_hidden_ready, row // Int32(MLP_TILE_M), swiglu_columns * Int32(2))
                else:
                    if macro > Int32(0):
                        _counter_wait_gpu(replayed_hidden_ready, row // Int32(MLP_TILE_M), swiglu_columns * Int32(2))
            else:
                _counter_wait_gpu(d_gate_up_ready, Int32(num_local_tokens // MLP_TILE_M) + row // Int32(MLP_TILE_M), swiglu_columns * Int32(2))
        if wait_minibatch != Int32(0):
            # dY and the replayed x arrive once per STREAM_ROWS x STREAM_COLUMNS block of the stream.
            mini = row // Int32(minibatch_size)
            rows = cutlass.min(Int32(minibatch_size), num_tokens - mini * Int32(minibatch_size))
            required = _ceil_div_i32(rows, Int32(STREAM_ROWS)) * _ceil_div_i32(hidden_size, Int32(STREAM_COLUMNS))
            if cutlass.const_expr(self.recompute_forward_context):
                if kind == Int32(ROUTED_WGRAD_DOWN):
                    _counter_wait_gpu(d_y_routed_ready, mini, required)
                else:
                    _counter_wait_gpu(replayed_x_routed_ready, mini, required)
            else:
                if kind == Int32(ROUTED_WGRAD_DOWN):
                    _counter_wait_gpu(d_y_routed_ready, mini, required)
                elif macro > Int32(0):
                    _counter_wait_gpu(replayed_x_routed_ready, mini, required)

    @cute.jit
    def _wait_task_operands(
        self,
        kind: Int32,
        macro: Int32,
        generation: Int32,
        row: Int32,
        num_tokens: Int32,
        d_y_ready,
        d_gate_up_ready,
        replayed_x_ready,
        *,
        num_local_tokens: Int32,
        hidden_size: Int32,
        intermediate_size: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
    ):
        """The base class's ``_wait_task_operands`` with one difference: dY (ROUTED_DOWN_DGRAD) and the replayed x
        (REPLAY_GATE / REPLAY_UP) arrive once per STREAM_ROWS x STREAM_COLUMNS block of the backward stream, not per
        128 x 512 dispatch task; every other case is the base method's."""
        if kind == Int32(ROUTED_DOWN_DGRAD) or kind == Int32(REPLAY_GATE) or kind == Int32(REPLAY_UP):
            rows = cutlass.min(Int32(minibatch_size), num_tokens - generation * Int32(minibatch_size))
            required = _ceil_div_i32(rows, Int32(STREAM_ROWS)) * _ceil_div_i32(hidden_size, Int32(STREAM_COLUMNS))
            if kind == Int32(ROUTED_DOWN_DGRAD):
                _counter_wait_gpu(d_y_ready, generation, required)
            else:
                _counter_wait_gpu(replayed_x_ready, generation, required)
        else:
            _Mxfp8BwdBase._wait_task_operands(
                self,
                kind,
                macro,
                generation,
                row,
                num_tokens,
                d_y_ready,
                d_gate_up_ready,
                replayed_x_ready,
                num_local_tokens=num_local_tokens,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                macrobatch_size=macrobatch_size,
                minibatch_size=minibatch_size,
            )

    @cute.jit
    def _routed_operand_waits(
        self,
        kind: Int32,
        macro: Int32,
        generation: Int32,
        row: Int32,
        k: Int32,
        k_start: Int32,
        num_tokens: Int32,
        d_y_routed_ready,
        d_gate_up_ready,
        replayed_x_routed_ready,
        replayed_hidden_ready,
        num_local_tokens: Int32,
        hidden_size: Int32,
        intermediate_size: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        is_wgrad: cutlass.Constexpr[bool],
    ):
        """The waits before K tile ``k`` of a routed task, on one lane (both producer warps).

        Dgrad / replay: every operand before the first tile.  Wgrad: as a
        tile first touches a 256-row block or a minibatch of the token-major operands.
        """
        if cutlass.const_expr(is_wgrad):
            operand_row = k_start + k * Int32(FP8_K_TILE)
            wait_row_block = Int32(k == Int32(0) or operand_row % Int32(MLP_TILE_M) == Int32(0))
            wait_minibatch = Int32(k == Int32(0) or operand_row % Int32(minibatch_size) == Int32(0))
            if wait_row_block + wait_minibatch != Int32(0):
                self._wait_routed_wgrad_operands(
                    kind,
                    macro,
                    wait_row_block,
                    wait_minibatch,
                    operand_row,
                    num_tokens,
                    d_y_routed_ready,
                    d_gate_up_ready,
                    replayed_x_routed_ready,
                    replayed_hidden_ready,
                    num_local_tokens,
                    hidden_size,
                    intermediate_size,
                    minibatch_size,
                )
        else:
            if k == Int32(0):
                self._wait_task_operands(
                    kind,
                    macro,
                    generation,
                    row,
                    num_tokens,
                    d_y_routed_ready,
                    d_gate_up_ready,
                    replayed_x_routed_ready,
                    num_local_tokens=num_local_tokens,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    macrobatch_size=macrobatch_size,
                    minibatch_size=minibatch_size,
                )

    @cute.jit
    def _run_routed_gemm(
        self,
        epi_tiled_mma: cute.TiledMma,
        routed_mma: cute.TiledMma,
        b_mma: cute.TiledMma,
        routed_mma_sfb: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_atom: cute.CopyAtom,
        a_tensor: cute.Tensor,
        b_atom: cute.CopyAtom,
        b_tensor: cute.Tensor,
        sfa_atom: cute.CopyAtom,
        sfa: cute.Tensor,
        sfb_atom: cute.CopyAtom,
        sfb: cute.Tensor,
        a2_atom: cute.CopyAtom,
        a2_tensor: cute.Tensor,
        b2_atom: cute.CopyAtom,
        b2_tensor: cute.Tensor,
        sfa2_atom: cute.CopyAtom,
        sfa2: cute.Tensor,
        sfb2_atom: cute.CopyAtom,
        sfb2: cute.Tensor,
        d_atom: cute.CopyAtom,
        d_tensor: cute.Tensor,
        quant: Optional[list[object]],
        sA: cute.Tensor,
        sB: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sD: cute.Tensor,
        epi_tile: cute.Tile,
        arena: cute.Pointer,
        acc_ptr: cute.Pointer,
        ab_pipeline,
        sf_pipeline,
        acc_pipeline,
        ab_producer_state,
        ab_consumer_state,
        sf_producer_state,
        sf_consumer_state,
        acc_producer_state,
        acc_consumer_state,
        thread: Int32,
        warp: Int32,
        cta_rank: Int32,
        kind: Int32,
        macro: Int32,
        generation: Int32,
        row: Int32,
        column: Int32,
        k_start: Int32,
        k_end: Int32,
        first: Int32,
        num_tokens: Int32,
        d_y_routed_ready: cute.Tensor,
        d_gate_up_ready: cute.Tensor,
        replayed_x_routed_ready: cute.Tensor,
        replayed_hidden_ready: cute.Tensor,
        d_hidden_ready: cute.Tensor,
        d_x_routed_ready: cute.Tensor,
        replayed_gate_up_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        num_local_tokens: Int32,
        hidden_size: Int32,
        intermediate_size: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        is_wgrad: cutlass.Constexpr[bool],
        second: cutlass.Constexpr[bool],
        tensormaps: cute.Tensor,
        a_position: cutlass.Constexpr[int],
        b_position: cutlass.Constexpr[int],
        a2_position: cutlass.Constexpr[int],
        b2_position: cutlass.Constexpr[int],
        d_add_atom: Optional[cute.CopyAtom] = None,
        d_add_tensor: Optional[cute.Tensor] = None,
        b_mn_major: cutlass.Constexpr[bool] = False,
        b_desc_ptr=None,
        sfb_desc_ptr=None,
        b2_desc_ptr=None,
        sfb2_desc_ptr=None,
        d_desc_ptr=None,
        d_row_tile_offset=None,
    ):
        """One routed MXFP8 GEMM task.

        The AB producer warp loads A / B (128-K e4m3 tiles), the SF producer warp the scales (its
        own ring), the UMMA warp of the leader CTA issues the block-scaled MMAs, the epilogue
        warps store D: in the e4m3-context kernel the Gate / Up replay goes through the MXFP8
        epilogue (BF16 ring + e4m3 context, ``quant``); otherwise D is a plain BF16 store (the Gate
        / Up replay into the BF16 context; Wgrad stores its first contribution and accumulates the
        rest).
        ``second``: dgrad Gate+Up, whose second operand pair continues the same accumulator.  Wgrad K runs over the
        expert's token rows of this macrobatch (``k_start`` .. ``k_end``, global rows).

        ``b_mma`` partitions B and ``sB`` is its stage: for an MN-major B (``b_mn_major``) both come
        from the second collective, while A, the scales and the ring bytes are the shared ones.
        ``*_desc_ptr``: a split operand's own descriptor, None for the global tensormap slot.
        ``d_row_tile_offset`` (Int32 CTA tiles, None without one): the Up half of a combined FC1 main_grad.
        """
        intermediate_blocks = _tile_blocks(intermediate_size, self.has_tail)
        k_tiles = Int32(cute.size(a_tensor, mode=[1])) // Int32(FP8_K_TILE)
        first_tiles = k_tiles
        if cutlass.const_expr(second):
            k_tiles = k_tiles * Int32(2)
        k_offset = Int32(0)
        if cutlass.const_expr(is_wgrad):
            k_tiles = (k_end - k_start) // Int32(FP8_K_TILE)
            k_offset = k_start - macro * Int32(macrobatch_size)
        #
        # A / B operand TMA producer warp
        #
        if warp == Int32(AB_TMA_PRODUCER_WARP):
            thr_mma = routed_mma.get_slice(cta_rank)
            thr_b_mma = thr_mma
            if cutlass.const_expr(b_mn_major):
                thr_b_mma = b_mma.get_slice(cta_rank)
            a_tile = cute.local_tile(cute.domain_offset((Int32(0), k_offset), a_tensor), (MLP_TILE_M, FP8_K_TILE), (row, None))
            b_tile = cute.local_tile(cute.domain_offset((Int32(0), k_offset), b_tensor), (MLP_TILE_N, FP8_K_TILE), (column, None))
            copy_A = gemm._make_tma_block_load_fn(a_atom, thr_mma.partition_A(a_tile), sA)
            copy_B = gemm._make_tma_block_load_fn(b_atom, thr_b_mma.partition_B(b_tile), sB, tma_desc_ptr=b_desc_ptr)
            copy_A2 = copy_A
            copy_B2 = copy_B
            if cutlass.const_expr(second):
                copy_A2 = gemm._make_tma_block_load_fn(a2_atom, thr_mma.partition_A(cute.local_tile(a2_tensor, (MLP_TILE_M, FP8_K_TILE), (row, None))), sA)
                copy_B2 = gemm._make_tma_block_load_fn(
                    b2_atom, thr_b_mma.partition_B(cute.local_tile(b2_tensor, (MLP_TILE_N, FP8_K_TILE), (column, None))), sB, tma_desc_ptr=b2_desc_ptr
                )
            # The descriptors' global copies (written by the prologue launch, acquired once per CTA at the kernel's
            # start); the K loop passes them to every load.
            a_desc = _tensormap_ptr(tensormaps, a_position)
            b_desc = _tensormap_ptr(tensormaps, b_position)
            if cutlass.const_expr(b_desc_ptr is not None):  # split weights: this expert's own descriptor
                b_desc = b_desc_ptr
            a2_desc = a_desc
            b2_desc = b_desc
            if cutlass.const_expr(second):
                a2_desc = _tensormap_ptr(tensormaps, a2_position)
                b2_desc = _tensormap_ptr(tensormaps, b2_position)
                if cutlass.const_expr(b2_desc_ptr is not None):
                    b2_desc = b2_desc_ptr
            for k in cutlass.range(k_tiles, unroll=1):
                with cute.arch.elect_one():
                    self._routed_operand_waits(
                        kind,
                        macro,
                        generation,
                        row,
                        k,
                        k_start,
                        num_tokens,
                        d_y_routed_ready,
                        d_gate_up_ready,
                        replayed_x_routed_ready,
                        replayed_hidden_ready,
                        num_local_tokens,
                        hidden_size,
                        intermediate_size,
                        macrobatch_size,
                        minibatch_size,
                        is_wgrad,
                    )
                cute.arch.sync_warp()
                ab_pipeline.producer_acquire(ab_producer_state)
                tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
                stage = ab_producer_state.index
                if cutlass.const_expr(second):
                    if k < first_tiles:
                        copy_A(src_idx=k, dst_idx=stage, tma_bar_ptr=tma_bar_ptr, tma_desc_ptr=a_desc)
                        copy_B(src_idx=k, dst_idx=stage, tma_bar_ptr=tma_bar_ptr, tma_desc_ptr=b_desc)
                    else:
                        copy_A2(src_idx=k - first_tiles, dst_idx=stage, tma_bar_ptr=tma_bar_ptr, tma_desc_ptr=a2_desc)
                        copy_B2(src_idx=k - first_tiles, dst_idx=stage, tma_bar_ptr=tma_bar_ptr, tma_desc_ptr=b2_desc)
                else:
                    copy_A(src_idx=k, dst_idx=stage, tma_bar_ptr=tma_bar_ptr, tma_desc_ptr=a_desc)
                    copy_B(src_idx=k, dst_idx=stage, tma_bar_ptr=tma_bar_ptr, tma_desc_ptr=b_desc)
                ab_pipeline.producer_commit(ab_producer_state)
                ab_producer_state.advance()
        #
        # Scale-factor TMA producer warp
        #
        elif warp == Int32(SF_PRODUCER_WARP):
            copy_sfa, copy_sfb = mxgemm.make_sf_load_fns(
                self.routed,
                routed_mma,
                routed_mma_sfb,
                cluster_layout_vmnk,
                cluster_layout_sfb_vmnk,
                sfa_atom,
                cute.domain_offset((Int32(0), k_offset), sfa),
                sfb_atom,
                cute.domain_offset((Int32(0), k_offset), sfb),
                sSFA,
                sSFB,
                row,
                column,
                cta_rank,
            )
            copy_sfa2 = copy_sfa
            copy_sfb2 = copy_sfb
            if cutlass.const_expr(second):
                copy_sfa2, copy_sfb2 = mxgemm.make_sf_load_fns(
                    self.routed,
                    routed_mma,
                    routed_mma_sfb,
                    cluster_layout_vmnk,
                    cluster_layout_sfb_vmnk,
                    sfa2_atom,
                    sfa2,
                    sfb2_atom,
                    sfb2,
                    sSFA,
                    sSFB,
                    row,
                    column,
                    cta_rank,
                )
            sfa_desc = _tensormap_ptr(tensormaps, a_position + SFA_OF)
            sfb_desc = _tensormap_ptr(tensormaps, b_position + SFB_OF)
            if cutlass.const_expr(sfb_desc_ptr is not None):  # split weights: this expert's own scale descriptor
                sfb_desc = sfb_desc_ptr
            sfa2_desc = sfa_desc
            sfb2_desc = sfb_desc
            if cutlass.const_expr(second):
                sfa2_desc = _tensormap_ptr(tensormaps, a2_position + SFA_OF)
                sfb2_desc = _tensormap_ptr(tensormaps, b2_position + SFB_OF)
                if cutlass.const_expr(sfb2_desc_ptr is not None):
                    sfb2_desc = sfb2_desc_ptr
            for k in cutlass.range(k_tiles, unroll=1):
                with cute.arch.elect_one():
                    self._routed_operand_waits(
                        kind,
                        macro,
                        generation,
                        row,
                        k,
                        k_start,
                        num_tokens,
                        d_y_routed_ready,
                        d_gate_up_ready,
                        replayed_x_routed_ready,
                        replayed_hidden_ready,
                        num_local_tokens,
                        hidden_size,
                        intermediate_size,
                        macrobatch_size,
                        minibatch_size,
                        is_wgrad,
                    )
                cute.arch.sync_warp()
                sf_pipeline.producer_acquire(sf_producer_state)
                tma_bar_ptr = sf_pipeline.producer_get_barrier(sf_producer_state)
                stage = sf_producer_state.index
                if cutlass.const_expr(second):
                    if k < first_tiles:
                        copy_sfa(src_idx=k, dst_idx=stage, tma_bar_ptr=tma_bar_ptr, tma_desc_ptr=sfa_desc)
                        copy_sfb(src_idx=k, dst_idx=stage, tma_bar_ptr=tma_bar_ptr, tma_desc_ptr=sfb_desc)
                    else:
                        copy_sfa2(src_idx=k - first_tiles, dst_idx=stage, tma_bar_ptr=tma_bar_ptr, tma_desc_ptr=sfa2_desc)
                        copy_sfb2(src_idx=k - first_tiles, dst_idx=stage, tma_bar_ptr=tma_bar_ptr, tma_desc_ptr=sfb2_desc)
                else:
                    copy_sfa(src_idx=k, dst_idx=stage, tma_bar_ptr=tma_bar_ptr, tma_desc_ptr=sfa_desc)
                    copy_sfb(src_idx=k, dst_idx=stage, tma_bar_ptr=tma_bar_ptr, tma_desc_ptr=sfb_desc)
                sf_pipeline.producer_commit(sf_producer_state)
                sf_producer_state.advance()
        #
        # UMMA issuing warp (cluster leader)
        #
        elif warp == Int32(UMMA_PRODUCER_WARP) and cta_rank == Int32(0):
            ab_consumer_state, sf_consumer_state, acc_producer_state = mxgemm.mma_blockscaled(
                ab_pipeline, sf_pipeline, acc_pipeline, ab_consumer_state, sf_consumer_state, acc_producer_state, arena, acc_ptr, k_tiles, b_mn_major=b_mn_major
            )
        #
        # Epilogue warpgroup: TMEM -> registers -> quantize / store
        #
        elif warp <= Int32(EPILOGUE_WARP_LAST):
            # The BF16 ABt collective's epilogue partitions: the same 2-CTA 256 x 256 FP32
            # accumulator and BF16 D tile as every routed task.  A Wgrad accumulating into main_grad
            # partitions with the Wgrad collective instead, the one that owns its D atoms there: FP32
            # with one staging tile for a float32 main_grad, which never rounds the accumulator.
            collective = self.atb if is_wgrad and self.accumulate_wgrad else self.abt
            fake = epi_tiled_mma.make_fragment_C(cute.append(epi_tiled_mma.partition_shape_C(collective.mma_tiler[:2]), 1))
            acc_tensor = cute.make_tensor(acc_ptr, fake.layout)
            t2r, acc_partition, r_acc = gemm._Bf16Gemm.epilog_tmem_copy_and_partition(collective, thread, acc_tensor, epi_tile, True)
            r_d = cute.make_rmem_tensor(r_acc.shape, Float32)
            # The destination's dtype comes from the collective: BFloat16, or Float32 for that main_grad.
            d_dtype = collective.d_dtype_expected
            r2s, r_d_partition, s_d_partition = gemm._Bf16Gemm.epilog_smem_store_and_partition(collective, t2r, collective.d_layout, d_dtype, r_d, sD, thread)
            acc_partition = cute.group_modes(acc_partition[None, None, None, None, None, 0], 3, 5)
            copy_D, _, _ = gemm._Bf16Gemm.epilog_gmem_copy_and_partition(
                collective, d_atom, d_tensor, collective.cta_tile_shape_mnk[:2], epi_tile, sD, _d_tile_coord(row, cta_rank, column, d_row_tile_offset)
            )
            if cutlass.const_expr(quant is not None):
                # Replay Gate / Up: BF16 ring for the replayed SwiGLU plus the e4m3 context
                # (gate / up fp8 + scales) dgrad Gate+Up reads; D stage 2 stages the e4m3 tile.
                fp8_atom, fp8_tensor, scales = quant
                acc_consumer_state = mxgemm.store_acc_mxfp8(
                    collective,
                    t2r,
                    r2s,
                    acc_partition,
                    r_acc,
                    r_d_partition,
                    s_d_partition,
                    copy_D,
                    fp8_atom,
                    fp8_tensor,
                    scales.iterator,
                    intermediate_size // Int32(SWIGLU_TILE),
                    arena + mxgemm.MXFP8_D_OFFSET + 2 * gemm.CTA_M * gemm.EPILOGUE_N * 2,
                    row * Int32(2) + cta_rank,
                    column,
                    acc_pipeline,
                    acc_consumer_state,
                    thread,
                    has_tail=self.has_tail,
                )
            else:
                acc_pipeline.consumer_wait(acc_consumer_state)
                if cutlass.const_expr(d_dtype is Float32):
                    # Straight from the accumulator into main_grad, one subtile live at a time.
                    assert d_add_atom is not None, "a float32 D is a main_grad, always accumulated"
                    copy_D_add = _wgrad_add_copy(
                        collective, d_add_atom, d_add_tensor, epi_tile, sD, _d_tile_coord(row, cta_rank, column, d_row_tile_offset), d_desc_ptr
                    )
                    gemm.store_acc_fp32(
                        collective, t2r, r2s, acc_partition, r_acc, r_d_partition, s_d_partition, copy_D_add, thread, acc_pipeline, acc_consumer_state
                    )
                    acc_consumer_state.advance()
                else:
                    registers = cute.make_rmem_tensor(cute.append(r_d_partition.shape, 8), d_dtype)
                    for subtile in cutlass.range_constexpr(8):
                        collective.epi_load_acc_subtile(t2r, r2s, acc_partition, r_acc, r_d_partition, (0, subtile), acc_pipeline, acc_consumer_state, 7)
                        registers[None, None, None, subtile].store(r_d_partition.load().to(d_dtype))
                    acc_consumer_state.advance()
                    if cutlass.const_expr(d_add_atom is not None):
                        copy_D_add = _wgrad_add_copy(
                            collective, d_add_atom, d_add_tensor, epi_tile, sD, _d_tile_coord(row, cta_rank, column, d_row_tile_offset), d_desc_ptr
                        )
                        # Accumulating into main_grad makes even the first contribution additive.
                        contribution = Int32(0) if cutlass.const_expr(self.accumulate_wgrad) else first
                        gemm.store_acc_all(collective, registers, r2s, s_d_partition, copy_D, thread, copy_D_add=copy_D_add, first_contribution=contribution)
                    else:
                        gemm.store_acc_all(collective, registers, r2s, s_d_partition, copy_D, thread)
            # Publish the task's readiness arrivals only after its stores complete.
            retires = num_tokens > Int32(macrobatch_size) and kind >= Int32(ROUTED_DOWN_DGRAD)
            if cutlass.const_expr(d_add_atom is not None):
                if warp == Int32(0):
                    cute.arch.cp_async_bulk_wait_group(0, read=False)
                    cute.arch.sync_warp()
            else:
                gemm.finish_store(thread)
            if thread == Int32(0):
                global_row = macro * Int32(macrobatch_size // MLP_TILE_M) + row
                if kind == Int32(ROUTED_DOWN_DGRAD):
                    _counter_arrive_release_gpu(
                        d_hidden_ready, Int32(num_local_tokens // MLP_TILE_M) * intermediate_blocks + global_row * intermediate_blocks + column
                    )
                elif kind == Int32(ROUTED_GATE_UP_DGRAD):
                    _counter_arrive_release_gpu(d_x_routed_ready, generation)
                elif kind == Int32(REPLAY_GATE) or kind == Int32(REPLAY_UP):
                    _counter_arrive_release_gpu(replayed_gate_up_ready, global_row * intermediate_blocks + column)
                if retires:
                    _counter_arrive_release_gpu(routed_buffers_done, macro)
        return (ab_producer_state, ab_consumer_state, sf_producer_state, sf_consumer_state, acc_producer_state, acc_consumer_state)

    @cute.jit
    def _run_shared_swiglu_bwd(
        self,
        role_warp: Int32,
        cta_rank: Int32,
        local: Int32,
        d_hidden_ready: cute.Tensor,
        d_gate_up_ready: cute.Tensor,
        swiglu_load_atoms: list,
        swiglu_load_tensors: list,
        swiglu_store_atoms: list,
        swiglu_store_tensors: list,
        s_d_hidden: cute.Tensor,
        s_gate: cute.Tensor,
        s_up: cute.Tensor,
        swiglu_mbarriers,
        phase_bits: Int32,
        *,
        num_local_tokens: Int32,
        intermediate_size: Int32,
        swiglu_limit: Float32,
        situ_beta1=None,
        situ_beta2=None,
    ) -> Int32:
        """The shared expert's SwiGLU backward, one task of two 128 x 128 tiles (warp 0 loads, every
        thread computes four elements at a time, warp 0 stores, thread 0 retires the 256-row block)
        with the element math of ``_mxfp8_bwd_math``: ``swiglu_bwd_shared`` when unclamped,
        ``swiglu_bwd_pair`` when clamped, ``situ_bwd`` for SiTU (QuACK's ``dswiglu`` rounds d_gate
        differently and is not used).  ``swiglu_load_atoms`` / ``_tensors``: Gate, Up, d_hidden;
        ``swiglu_store_atoms`` / ``_tensors``: d_gate, d_up.  The routed task is ``_run_routed_swiglu_bwd``.
        """
        intermediate_blocks = _tile_blocks(intermediate_size, self.has_tail)
        swiglu_columns = intermediate_size // Int32(SWIGLU_TILE_N)
        thread = cute.arch.thread_idx()[0]
        # Prefetch both stages, compute and store both, then drain and retire.
        for pass_index in cutlass.range_constexpr(3):
            if cutlass.const_expr(pass_index == 2):
                if role_warp == Int32(0):
                    cute.arch.cp_async_bulk_wait_group(0, read=False)
                cute.arch.sync_threads()
            for stage in cutlass.range_constexpr(SWIGLU_BWD_STAGES):
                global_tile = (local * Int32(CLUSTER_SIZE) + cta_rank) * Int32(SWIGLU_BWD_STAGES) + Int32(stage)
                global_row_tile = global_tile // swiglu_columns
                column_tile = global_tile - global_row_tile * swiglu_columns
                if global_row_tile * Int32(SWIGLU_TILE_M) < num_local_tokens:
                    stage_index = Int32(stage)
                    tma_barrier = swiglu_mbarriers + stage_index
                    s_d_hidden_stage = cute.slice_(s_d_hidden, (None, None, stage_index))
                    s_gate_stage = cute.slice_(s_gate, (None, None, stage_index))
                    s_up_stage = cute.slice_(s_up, (None, None, stage_index))
                    if cutlass.const_expr(pass_index == 0):
                        if role_warp == Int32(0):
                            with cute.arch.elect_one():
                                _counter_wait_gpu(
                                    d_hidden_ready, (global_row_tile // Int32(2)) * intermediate_blocks + column_tile // Int32(2), Int32(CLUSTER_SIZE)
                                )
                                cute.arch.mbarrier_arrive_and_expect_tx(tma_barrier, Int32(SWIGLU_TMA_TRANSACTION_BYTES))
                            cute.arch.sync_warp()
                            _issue_swiglu_tma_load(swiglu_load_atoms[0], swiglu_load_tensors[0], global_row_tile, column_tile, s_gate_stage, tma_barrier)
                            _issue_swiglu_tma_load(swiglu_load_atoms[1], swiglu_load_tensors[1], global_row_tile, column_tile, s_up_stage, tma_barrier)
                            _issue_swiglu_tma_load(swiglu_load_atoms[2], swiglu_load_tensors[2], global_row_tile, column_tile, s_d_hidden_stage, tma_barrier)
                    elif cutlass.const_expr(pass_index == 1):
                        cute.arch.mbarrier_wait(tma_barrier, (phase_bits >> stage_index) & Int32(1))
                        phase_bits = phase_bits ^ Int32(1 << stage)
                        cute.arch.sync_threads()
                        row_in_tile = thread % Int32(SWIGLU_TILE_M)
                        column_half = thread // Int32(SWIGLU_TILE_M)
                        copy_64b = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=64)
                        r_gate = cute.make_rmem_tensor((4,), BFloat16)
                        r_up = cute.make_rmem_tensor((4,), BFloat16)
                        r_d_hidden = cute.make_rmem_tensor((4,), BFloat16)
                        for group in cutlass.range(SWIGLU_TILE_N // 8, unroll=1):
                            base_col = column_half * Int32(SWIGLU_TILE_N // 2) + group * Int32(4)
                            s_gate_group = cute.coalesce(cute.local_tile(s_gate_stage, (1, 4), (row_in_tile, base_col // Int32(4))))
                            s_up_group = cute.coalesce(cute.local_tile(s_up_stage, (1, 4), (row_in_tile, base_col // Int32(4))))
                            s_d_hidden_group = cute.coalesce(cute.local_tile(s_d_hidden_stage, (1, 4), (row_in_tile, base_col // Int32(4))))
                            cute.copy(copy_64b, s_gate_group, r_gate)
                            cute.copy(copy_64b, s_up_group, r_up)
                            cute.copy(copy_64b, s_d_hidden_group, r_d_hidden)
                            for element in cutlass.range_constexpr(4):
                                gate = Float32(r_gate[element])
                                up = Float32(r_up[element])
                                d_hidden = Float32(r_d_hidden[element])
                                if cutlass.const_expr(self.is_situ):
                                    d_gate, d_up, _ = situ_bwd(gate, up, d_hidden, situ_beta1, situ_beta2)
                                elif cutlass.const_expr(self.is_clamped):
                                    d_gate, d_up, _ = swiglu_bwd_pair(gate, up, d_hidden, swiglu_limit, True)
                                else:
                                    d_gate, d_up = swiglu_bwd_shared(gate, up, d_hidden)
                                r_gate[element] = d_gate.to(BFloat16)
                                r_up[element] = d_up.to(BFloat16)
                            cute.copy(copy_64b, r_gate, s_gate_group)
                            cute.copy(copy_64b, r_up, s_up_group)
                        cute.arch.fence_view_async_shared()
                        cute.arch.sync_threads()
                        if role_warp == Int32(0):
                            _issue_swiglu_tma_store(swiglu_store_atoms[0], swiglu_store_tensors[0], global_row_tile, column_tile, s_gate_stage)
                            _issue_swiglu_tma_store(swiglu_store_atoms[1], swiglu_store_tensors[1], global_row_tile, column_tile, s_up_stage)
                            cute.arch.cp_async_bulk_commit_group()
                        cute.arch.sync_threads()
                    else:
                        if thread == Int32(0):
                            _counter_arrive_release_gpu(d_gate_up_ready, global_row_tile // Int32(2))
                        cute.arch.sync_threads()
        cute.arch.sync_threads()
        return phase_bits

    @cute.jit
    def _run_routed_swiglu_bwd(
        self,
        load_atoms: list,
        load_tensors: list,
        store_atoms: list,
        store_tensors: list,
        scale_tensors: list,
        arena: cute.Pointer,
        barriers: cute.Pointer,
        phase_bits: Int32,
        schedule_peer_rank: cute.Tensor,
        router_weights: cute.Tensor,
        d_router_partials: cute.Tensor,
        d_hidden_ready: cute.Tensor,
        replayed_gate_up_ready: cute.Tensor,
        d_gate_up_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        cta_rank: Int32,
        generation: Int32,
        macro: Int32,
        local: Int32,
        num_tokens: Int32,
        num_local_tokens: Int32,
        intermediate_size: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        swiglu_limit: Float32,
        situ_beta1=None,
        situ_beta2=None,
    ) -> Int32:
        """The routed SwiGLU backward with a BF16 Gate / Up context, one task (2 x 128 x 128 tiles).

        ``load_atoms`` / ``load_tensors``: d_hidden_routed, gate / up context (all BF16);
        ``store_atoms`` / ``store_tensors``: d_gate / d_up e4m3 transposed;
        ``scale_tensors``: d_gate_sc, d_up_sc, d_gate_sc_t, d_up_sc_t (512 B tiles, bulk copies), then
        the e4m3 d_gate / d_up (normal layout, stored from registers by ``_mxfp8_bwd_swiglu.compute``).
        """
        thread = cute.arch.thread_idx()[0]
        intermediate_blocks = _tile_blocks(intermediate_size, self.has_tail)
        col_blocks = intermediate_size // Int32(SWIGLU_TILE)
        macro_row_tiles = macro * Int32(macrobatch_size // SWIGLU_TILE)
        num_tiles = num_tokens // Int32(SWIGLU_TILE) * col_blocks
        tiles_per_minibatch = Int32(minibatch_size // SWIGLU_TILE) * col_blocks
        first_tile = generation * tiles_per_minibatch + local * Int32(CLUSTER_SIZE * SWIGLU_BWD_STAGES) + cta_rank * Int32(SWIGLU_BWD_STAGES)
        tile_end = cutlass.min(num_tiles, (generation + Int32(1)) * tiles_per_minibatch)
        retires = num_tokens > Int32(macrobatch_size)
        d_gate_sc, d_up_sc, d_gate_sc_t, d_up_sc_t, d_gate_fp8, d_up_fp8 = scale_tensors
        sc_columns = col_blocks  # scale tiles per macrobatch row tile (normal layout)
        sc_rows_t = Int32(macrobatch_size // SWIGLU_TILE)  # scale tiles per column tile (transposed)
        plain = cute.make_layout((SWIGLU_TILE, SWIGLU_TILE), stride=(SWIGLU_TILE, 1))
        if first_tile < tile_end:
            first_row = first_tile // col_blocks
            first_col = first_tile - first_row * col_blocks
            # Per stage, thread 0 arms the barrier and waits for the producers; then warp 0 as a whole,
            # after a warp sync, issues the stage's three BF16 tile loads, because the DSL's TMA tensor
            # copy elects its issuing lane itself (a single thread calling it deadlocks).
            for stage in cutlass.range_constexpr(SWIGLU_BWD_STAGES):
                if first_tile + Int32(stage) < tile_end:
                    row = first_row
                    col = first_col + Int32(stage)
                    if col >= col_blocks:
                        row = row + Int32(1)
                        col = col - col_blocks
                    barrier = barriers + Int32(stage)
                    local_row = row - macro_row_tiles
                    if thread == Int32(0):
                        cute.arch.mbarrier_arrive_and_expect_tx(barrier, Int32(SWIGLU_BWD_TRANSACTION_BYTES))
                        parent = (row // Int32(2)) * intermediate_blocks + col // Int32(2)
                        _counter_wait_gpu(d_hidden_ready, Int32(num_local_tokens // MLP_TILE_M) * intermediate_blocks + parent, Int32(CLUSTER_SIZE))
                        # A replayed macrobatch's Gate / Up are its replay's (recompute_forward_context: macrobatch 0's
                        # too).
                        if cutlass.const_expr(self.recompute_forward_context):
                            _counter_wait_gpu(replayed_gate_up_ready, parent, Int32(2 * CLUSTER_SIZE))
                        else:
                            if macro > Int32(0):
                                _counter_wait_gpu(replayed_gate_up_ready, parent, Int32(2 * CLUSTER_SIZE))
                    if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == Int32(0):
                        cute.arch.sync_warp()
                        _issue_swiglu_tma_load(
                            load_atoms[0],
                            load_tensors[0],
                            local_row,
                            col,
                            cute.make_tensor(cute.recast_ptr(arena + SWIGLU_BWD_D_HIDDEN + stage * BF16_TILE_BYTES, dtype=BFloat16), plain),
                            barrier,
                        )
                        _issue_swiglu_tma_load(
                            load_atoms[1],
                            load_tensors[1],
                            local_row,
                            col,
                            cute.make_tensor(cute.recast_ptr(arena + SWIGLU_BWD_GATE + stage * BF16_TILE_BYTES, dtype=BFloat16), plain),
                            barrier,
                        )
                        _issue_swiglu_tma_load(
                            load_atoms[2],
                            load_tensors[2],
                            local_row,
                            col,
                            cute.make_tensor(cute.recast_ptr(arena + SWIGLU_BWD_UP + stage * BF16_TILE_BYTES, dtype=BFloat16), plain),
                            barrier,
                        )
            for stage in cutlass.range_constexpr(SWIGLU_BWD_STAGES):
                if first_tile + Int32(stage) < tile_end:
                    row = first_row
                    col = first_col + Int32(stage)
                    if col >= col_blocks:
                        row = row + Int32(1)
                        col = col - col_blocks
                    cute.arch.mbarrier_wait(barriers + Int32(stage), (phase_bits >> Int32(stage)) & Int32(1))
                    phase_bits = phase_bits ^ Int32(1 << stage)
                    local_row = row - macro_row_tiles
                    tile_row = thread & Int32(SWIGLU_TILE - 1)
                    local_token = local_row * Int32(SWIGLU_TILE) + tile_row
                    peer_rank = schedule_peer_rank[row * Int32(SWIGLU_TILE) + tile_row]
                    inverse = inverse_router_weight(router_weights[local_token])
                    d_hidden = arena + SWIGLU_BWD_D_HIDDEN + stage * BF16_TILE_BYTES
                    gate = arena + SWIGLU_BWD_GATE + stage * BF16_TILE_BYTES
                    up = arena + SWIGLU_BWD_UP + stage * BF16_TILE_BYTES
                    d_scales = arena + SWIGLU_BWD_D_SC + stage * 2 * SF_TILE_BYTES
                    # This thread's row of the tile in the e4m3 outputs (macrobatch rows x intermediate).
                    output_offset = (Int64(local_row) * Int64(SWIGLU_TILE) + Int64(tile_row)) * Int64(intermediate_size) + Int64(col) * Int64(SWIGLU_TILE)
                    partial = swiglu_bwd.compute(
                        d_hidden,
                        gate,
                        up,
                        d_gate_fp8.iterator.toint() + output_offset,
                        d_up_fp8.iterator.toint() + output_offset,
                        d_scales,
                        thread,
                        inverse,
                        swiglu_limit,
                        self.is_clamped,
                        self.is_situ,
                        situ_beta1,
                        situ_beta2,
                    )
                    # The e4m3 rows (generic stores) precede the readiness release after the task: each thread
                    # orders its own against the async proxy the dgrad GEMM's TMA loads use.
                    _fence_proxy_async_global()
                    cute.arch.sync_threads()  # d_hidden read, BF16 d_gate / d_up and the scale tiles written
                    scale_offset = (Int64(local_row) * Int64(sc_columns) + Int64(col)) * Int64(SF_TILE_BYTES)
                    if thread == Int32(0):
                        cute.arch.fence_view_async_shared()
                        tma_store_1d_raw(d_gate_sc.iterator.toint() + scale_offset, d_scales, Int32(SF_TILE_BYTES))
                        tma_store_1d_raw(d_up_sc.iterator.toint() + scale_offset, d_scales + SF_TILE_BYTES, Int32(SF_TILE_BYTES))
                        cute.arch.cp_async_bulk_commit_group()
                    # The transposed tiles over this stage's d_hidden (read by every thread before the sync).
                    d_t = d_hidden
                    d_t_sc = arena + SWIGLU_BWD_D_T_SC + stage * 2 * SF_TILE_BYTES
                    swiglu_bwd.transpose(gate, up, d_t, d_t_sc, d_t + FP8_TILE_BYTES, d_t_sc + SF_TILE_BYTES, thread)
                    router = cute.make_tensor(
                        cute.recast_ptr(arena + SWIGLU_BWD_ROUTER + stage * SWIGLU_TILE * 4, dtype=Float32), cute.make_layout((SWIGLU_TILE,))
                    )
                    if thread >= Int32(SWIGLU_TILE):
                        router[tile_row] = partial
                    cute.arch.sync_threads()  # transposed tiles complete, partials stored
                    if thread == Int32(0):
                        cute.arch.fence_view_async_shared()
                        scale_offset_t = (Int64(col) * Int64(sc_rows_t) + Int64(local_row)) * Int64(SF_TILE_BYTES)
                        _issue_swiglu_tma_store(
                            store_atoms[0], store_tensors[0], col, local_row, cute.make_tensor(cute.recast_ptr(d_t, dtype=Float8E4M3FN), plain)
                        )
                        tma_store_1d_raw(d_gate_sc_t.iterator.toint() + scale_offset_t, d_t_sc, Int32(SF_TILE_BYTES))
                        _issue_swiglu_tma_store(
                            store_atoms[1], store_tensors[1], col, local_row, cute.make_tensor(cute.recast_ptr(d_t + FP8_TILE_BYTES, dtype=Float8E4M3FN), plain)
                        )
                        tma_store_1d_raw(d_up_sc_t.iterator.toint() + scale_offset_t, d_t_sc + SF_TILE_BYTES, Int32(SF_TILE_BYTES))
                        cute.arch.cp_async_bulk_commit_group()
                    if thread < Int32(SWIGLU_TILE) and peer_rank >= Int32(0):
                        d_router_partials[local_token, col] = add_ftz(partial, router[tile_row])
                    cute.arch.sync_threads()
            if thread == Int32(0):
                cute.arch.cp_async_bulk_wait_group(0, read=False)
                for stage in cutlass.range_constexpr(SWIGLU_BWD_STAGES):
                    if first_tile + Int32(stage) < tile_end:
                        row = first_row
                        if first_col + Int32(stage) >= col_blocks:
                            row = row + Int32(1)
                        _counter_arrive_release_gpu(d_gate_up_ready, Int32(num_local_tokens // MLP_TILE_M) + row // Int32(2))
        if thread == Int32(0) and retires:
            _counter_arrive_release_gpu(routed_buffers_done, macro)
        return phase_bits

    @cute.jit
    def _run_routed_swiglu_bwd_e4m3_context(
        self,
        load_atoms: list,
        load_tensors: list,
        store_atoms: list,
        store_tensors: list,
        scale_tensors: list,
        arena: cute.Pointer,
        barriers: cute.Pointer,
        phase_bits: Int32,
        schedule_peer_rank: cute.Tensor,
        router_weights: cute.Tensor,
        d_router_partials: cute.Tensor,
        d_hidden_ready: cute.Tensor,
        replayed_gate_up_ready: cute.Tensor,
        d_gate_up_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        cta_rank: Int32,
        generation: Int32,
        macro: Int32,
        local: Int32,
        num_tokens: Int32,
        num_local_tokens: Int32,
        intermediate_size: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        swiglu_limit: Float32,
        situ_beta1=None,
        situ_beta2=None,
    ) -> Int32:
        """The routed SwiGLU backward with the e4m3 / scale Gate / Up context, one task (2 x 128 x 128
        tiles).

        ``load_atoms`` / ``load_tensors``: d_hidden_routed (BF16), gate / up e4m3 context;
        ``store_atoms`` / ``store_tensors``: d_gate / d_up e4m3 (normal), then transposed;
        ``scale_tensors``: gate_sc, up_sc, d_gate_sc, d_up_sc, d_gate_sc_t, d_up_sc_t (512 B
        tiles, bulk copies).  The compute is ``_mxfp8_bwd_swiglu.compute_e4m3_context``.
        """
        thread = cute.arch.thread_idx()[0]
        intermediate_blocks = _tile_blocks(intermediate_size, self.has_tail)
        col_blocks = intermediate_size // Int32(SWIGLU_TILE)
        macro_row_tiles = macro * Int32(macrobatch_size // SWIGLU_TILE)
        num_tiles = num_tokens // Int32(SWIGLU_TILE) * col_blocks
        tiles_per_minibatch = Int32(minibatch_size // SWIGLU_TILE) * col_blocks
        first_tile = generation * tiles_per_minibatch + local * Int32(CLUSTER_SIZE * SWIGLU_BWD_STAGES) + cta_rank * Int32(SWIGLU_BWD_STAGES)
        tile_end = cutlass.min(num_tiles, (generation + Int32(1)) * tiles_per_minibatch)
        retires = num_tokens > Int32(macrobatch_size)
        gate_sc, up_sc, d_gate_sc, d_up_sc, d_gate_sc_t, d_up_sc_t = scale_tensors
        sc_columns = col_blocks  # scale tiles per macrobatch row tile (normal layout)
        sc_rows_t = Int32(macrobatch_size // SWIGLU_TILE)  # scale tiles per column tile (transposed)
        plain = cute.make_layout((SWIGLU_TILE, SWIGLU_TILE), stride=(SWIGLU_TILE, 1))
        if first_tile < tile_end:
            first_row = first_tile // col_blocks
            first_col = first_tile - first_row * col_blocks
            # Per stage, thread 0 arms the barrier, waits for the producers and issues the two 512 B
            # scale copies (raw bulk copies); then warp 0 as a whole, after a warp sync, issues the three
            # tensor loads, because the DSL's TMA tensor copy elects its issuing lane itself (a single
            # thread calling it deadlocks).  All five complete on the same barrier, so the order among
            # them does not matter.
            for stage in cutlass.range_constexpr(SWIGLU_BWD_STAGES):
                if first_tile + Int32(stage) < tile_end:
                    row = first_row
                    col = first_col + Int32(stage)
                    if col >= col_blocks:
                        row = row + Int32(1)
                        col = col - col_blocks
                    barrier = barriers + Int32(stage)
                    local_row = row - macro_row_tiles
                    if thread == Int32(0):
                        cute.arch.mbarrier_arrive_and_expect_tx(barrier, Int32(E4M3_CONTEXT_SWIGLU_BWD_TRANSACTION_BYTES))
                        parent = (row // Int32(2)) * intermediate_blocks + col // Int32(2)
                        _counter_wait_gpu(d_hidden_ready, Int32(num_local_tokens // MLP_TILE_M) * intermediate_blocks + parent, Int32(CLUSTER_SIZE))
                        # A replayed macrobatch's Gate / Up are its replay's (recompute_forward_context: macrobatch 0's
                        # too).
                        if cutlass.const_expr(self.recompute_forward_context):
                            _counter_wait_gpu(replayed_gate_up_ready, parent, Int32(2 * CLUSTER_SIZE))
                        else:
                            if macro > Int32(0):
                                _counter_wait_gpu(replayed_gate_up_ready, parent, Int32(2 * CLUSTER_SIZE))
                        scale_offset = (Int64(local_row) * Int64(sc_columns) + Int64(col)) * Int64(SF_TILE_BYTES)
                        tma_load_1d_raw(
                            arena + E4M3_CONTEXT_SWIGLU_BWD_GATE_SC + stage * SF_TILE_BYTES,
                            gate_sc.iterator.toint() + scale_offset,
                            barrier,
                            Int32(SF_TILE_BYTES),
                        )
                        tma_load_1d_raw(
                            arena + E4M3_CONTEXT_SWIGLU_BWD_UP_SC + stage * SF_TILE_BYTES, up_sc.iterator.toint() + scale_offset, barrier, Int32(SF_TILE_BYTES)
                        )
                    if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == Int32(0):
                        cute.arch.sync_warp()
                        _issue_swiglu_tma_load(
                            load_atoms[0],
                            load_tensors[0],
                            local_row,
                            col,
                            cute.make_tensor(cute.recast_ptr(arena + E4M3_CONTEXT_SWIGLU_BWD_D_HIDDEN + stage * BF16_TILE_BYTES, dtype=BFloat16), plain),
                            barrier,
                        )
                        _issue_swiglu_tma_load(
                            load_atoms[1],
                            load_tensors[1],
                            local_row,
                            col,
                            cute.make_tensor(cute.recast_ptr(arena + E4M3_CONTEXT_SWIGLU_BWD_GATE + stage * FP8_TILE_BYTES, dtype=Float8E4M3FN), plain),
                            barrier,
                        )
                        _issue_swiglu_tma_load(
                            load_atoms[2],
                            load_tensors[2],
                            local_row,
                            col,
                            cute.make_tensor(cute.recast_ptr(arena + E4M3_CONTEXT_SWIGLU_BWD_UP + stage * FP8_TILE_BYTES, dtype=Float8E4M3FN), plain),
                            barrier,
                        )
            for stage in cutlass.range_constexpr(SWIGLU_BWD_STAGES):
                if first_tile + Int32(stage) < tile_end:
                    row = first_row
                    col = first_col + Int32(stage)
                    if col >= col_blocks:
                        row = row + Int32(1)
                        col = col - col_blocks
                    cute.arch.mbarrier_wait(barriers + Int32(stage), (phase_bits >> Int32(stage)) & Int32(1))
                    phase_bits = phase_bits ^ Int32(1 << stage)
                    local_row = row - macro_row_tiles
                    tile_row = thread & Int32(SWIGLU_TILE - 1)
                    local_token = local_row * Int32(SWIGLU_TILE) + tile_row
                    peer_rank = schedule_peer_rank[row * Int32(SWIGLU_TILE) + tile_row]
                    inverse = inverse_router_weight(router_weights[local_token])
                    d_hidden = arena + E4M3_CONTEXT_SWIGLU_BWD_D_HIDDEN + stage * BF16_TILE_BYTES
                    gate = arena + E4M3_CONTEXT_SWIGLU_BWD_GATE + stage * FP8_TILE_BYTES
                    up = arena + E4M3_CONTEXT_SWIGLU_BWD_UP + stage * FP8_TILE_BYTES
                    gate_scales = arena + E4M3_CONTEXT_SWIGLU_BWD_GATE_SC + stage * SF_TILE_BYTES
                    up_scales = arena + E4M3_CONTEXT_SWIGLU_BWD_UP_SC + stage * SF_TILE_BYTES
                    partial = swiglu_bwd.compute_e4m3_context(
                        d_hidden,
                        gate,
                        up,
                        gate_scales,
                        up_scales,
                        arena + E4M3_CONTEXT_SWIGLU_BWD_D_UP,
                        thread,
                        inverse,
                        swiglu_limit,
                        self.is_clamped,
                        self.is_situ,
                        situ_beta1,
                        situ_beta2,
                    )
                    cute.arch.sync_threads()  # normal outputs and BF16 stagings visible
                    scale_offset = (Int64(local_row) * Int64(sc_columns) + Int64(col)) * Int64(SF_TILE_BYTES)
                    if thread == Int32(0):
                        cute.arch.fence_view_async_shared()
                        _issue_swiglu_tma_store(
                            store_atoms[0], store_tensors[0], local_row, col, cute.make_tensor(cute.recast_ptr(gate, dtype=Float8E4M3FN), plain)
                        )
                        tma_store_1d_raw(d_gate_sc.iterator.toint() + scale_offset, gate_scales, Int32(SF_TILE_BYTES))
                        _issue_swiglu_tma_store(
                            store_atoms[1], store_tensors[1], local_row, col, cute.make_tensor(cute.recast_ptr(up, dtype=Float8E4M3FN), plain)
                        )
                        tma_store_1d_raw(d_up_sc.iterator.toint() + scale_offset, up_scales, Int32(SF_TILE_BYTES))
                        cute.arch.cp_async_bulk_commit_group()
                    d_t = arena + E4M3_CONTEXT_SWIGLU_BWD_D_T
                    d_t_sc = arena + E4M3_CONTEXT_SWIGLU_BWD_D_T_SC
                    swiglu_bwd.transpose(d_hidden, arena + E4M3_CONTEXT_SWIGLU_BWD_D_UP, d_t, d_t_sc, d_t + FP8_TILE_BYTES, d_t_sc + SF_TILE_BYTES, thread)
                    router = cute.make_tensor(
                        cute.recast_ptr(arena + E4M3_CONTEXT_SWIGLU_BWD_ROUTER + stage * SWIGLU_TILE * 4, dtype=Float32), cute.make_layout((SWIGLU_TILE,))
                    )
                    if thread >= Int32(SWIGLU_TILE):
                        router[tile_row] = partial
                    cute.arch.sync_threads()  # transposed tiles complete, partials stored
                    if thread == Int32(0):
                        cute.arch.fence_view_async_shared()
                        scale_offset_t = (Int64(col) * Int64(sc_rows_t) + Int64(local_row)) * Int64(SF_TILE_BYTES)
                        _issue_swiglu_tma_store(
                            store_atoms[2], store_tensors[2], col, local_row, cute.make_tensor(cute.recast_ptr(d_t, dtype=Float8E4M3FN), plain)
                        )
                        tma_store_1d_raw(d_gate_sc_t.iterator.toint() + scale_offset_t, d_t_sc, Int32(SF_TILE_BYTES))
                        _issue_swiglu_tma_store(
                            store_atoms[3], store_tensors[3], col, local_row, cute.make_tensor(cute.recast_ptr(d_t + FP8_TILE_BYTES, dtype=Float8E4M3FN), plain)
                        )
                        tma_store_1d_raw(d_up_sc_t.iterator.toint() + scale_offset_t, d_t_sc + SF_TILE_BYTES, Int32(SF_TILE_BYTES))
                        cute.arch.cp_async_bulk_commit_group()
                        # The next stage reuses the d_up staging and the transposed tiles.
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    if thread < Int32(SWIGLU_TILE) and peer_rank >= Int32(0):
                        d_router_partials[local_token, col] = add_ftz(partial, router[tile_row])
                    cute.arch.sync_threads()
            if thread == Int32(0):
                cute.arch.cp_async_bulk_wait_group(0, read=False)
                for stage in cutlass.range_constexpr(SWIGLU_BWD_STAGES):
                    if first_tile + Int32(stage) < tile_end:
                        row = first_row
                        if first_col + Int32(stage) >= col_blocks:
                            row = row + Int32(1)
                        _counter_arrive_release_gpu(d_gate_up_ready, Int32(num_local_tokens // MLP_TILE_M) + row // Int32(2))
        if thread == Int32(0) and retires:
            _counter_arrive_release_gpu(routed_buffers_done, macro)
        return phase_bits

    @cute.jit
    def __call__(
        self,
        x_peer_table: cute.Tensor,  # int64 (ep_size,): every rank's x buffer
        d_y_peer_table: cute.Tensor,  # int64 (ep_size,): dY buffers
        d_x_peer_table: cute.Tensor,  # int64 (ep_size,): routed dX buffers
        router_peer_table: cute.Tensor,  # int64 (ep_size,): router weights
        d_router_peer_table: cute.Tensor,  # int64 (ep_size,): router-weight gradient partials
        schedule_peer_rank: cute.Tensor,  # int32 (capacity,)
        schedule_peer_token_idx: cute.Tensor,  # int32 (capacity,)
        num_tokens: cute.Tensor,  # int32 (1,)
        tokens_per_expert: cute.Tensor,  # int32 (E,)
        x_shared: cute.Tensor,  # bf16 (T, H)
        x_shared_ht: cute.Tensor,  # bf16 (H, T) view of x_shared
        d_y_shared: cute.Tensor,  # bf16 (T, H)
        d_y_shared_ht: cute.Tensor,  # bf16 (H, T) view of d_y_shared
        gate_shared: cute.Tensor,  # bf16 (T, I) context
        up_shared: cute.Tensor,  # bf16 (T, I) context
        hidden_shared_it: cute.Tensor,  # bf16 (I, T) view of the Hidden context
        w_shared_gate_hi: cute.Tensor,  # bf16 (H, I) view of the (I, H) weight
        w_shared_up_hi: cute.Tensor,  # bf16 (H, I) view of the (I, H) weight
        w_shared_down_ih: cute.Tensor,  # bf16 (I, H) view of the (H, I) weight
        d_hidden_shared: cute.Tensor,  # bf16 (T, I) out
        d_gate_shared: cute.Tensor,  # bf16 (T, I) out
        d_up_shared: cute.Tensor,  # bf16 (T, I) out
        d_gate_shared_it: cute.Tensor,  # bf16 (I, T) view of d_gate_shared
        d_up_shared_it: cute.Tensor,  # bf16 (I, T) view of d_up_shared
        d_x_shared: cute.Tensor,  # bf16 (T, H) out
        d_w_shared_gate: cute.Tensor,  # bf16 (I, H) out
        d_w_shared_up: cute.Tensor,  # bf16 (I, H) out
        d_w_shared_down: cute.Tensor,  # bf16 (H, I) out
        w_routed_gate: cute.Tensor,  # e4m3 (I, H, E) view of the (E, I, H) Gate weights
        w_routed_gate_sc: cute.Tensor,  # e8m0 scales of the Gate weights
        w_routed_up: cute.Tensor,  # e4m3 (I, H, E) view of the (E, I, H) Up weights
        w_routed_up_sc: cute.Tensor,  # e8m0 scales of the Up weights
        w_routed_gate_t: cute.Tensor,  # e4m3 (H, I, E) view of the (E, H, I) transposed Gate weights
        w_routed_gate_t_sc: cute.Tensor,  # e8m0 scales of the transposed Gate weights
        w_routed_up_t: cute.Tensor,  # e4m3 (H, I, E) view of the (E, H, I) transposed Up weights
        w_routed_up_t_sc: cute.Tensor,  # e8m0 scales of the transposed Up weights
        w_routed_down_t: cute.Tensor,  # e4m3 (I, H, E) view of the (E, I, H) transposed Down weights
        w_routed_down_t_sc: cute.Tensor,  # e8m0 scales of the transposed Down weights
        x_fp8_routed: cute.Tensor,  # e4m3 (M, H) replay dispatch out
        x_sc_routed: cute.Tensor,  # e8m0 scales of x_fp8_routed
        x_fp8_t_routed: cute.Tensor,  # e4m3 (H, M) context; a replayed macrobatch's stream rewrites it in place
        x_sc_t_routed: cute.Tensor,  # e8m0 scales of x_fp8_t_routed
        gate_routed: cute.Tensor,  # bf16 (M, I) Gate context (the forward's; each replay overwrites it)
        up_routed: cute.Tensor,  # bf16 (M, I) Up context (the forward's; each replay overwrites it)
        hidden_fp8_routed: cute.Tensor,  # e4m3 (M, I) replay SwiGLU out
        hidden_sc_routed: cute.Tensor,  # e8m0 scales of hidden_fp8_routed
        hidden_fp8_t_routed: cute.Tensor,  # e4m3 (I, M) context
        hidden_sc_t_routed: cute.Tensor,  # e8m0 scales of hidden_fp8_t_routed
        d_y_fp8_routed: cute.Tensor,  # e4m3 (M, H) reverse-combine out
        d_y_sc_routed: cute.Tensor,  # e8m0 scales of d_y_fp8_routed
        d_y_fp8_t_routed: cute.Tensor,  # e4m3 (H, M)
        d_y_sc_t_routed: cute.Tensor,  # e8m0 scales of d_y_fp8_t_routed
        d_hidden_routed: cute.Tensor,  # bf16 (M, I) out
        d_x_routed: cute.Tensor,  # bf16 (M, H) out
        d_gate_fp8_routed: cute.Tensor,  # e4m3 (M, I) out
        d_gate_sc_routed: cute.Tensor,  # e8m0 scales of d_gate_fp8_routed
        d_gate_fp8_t_routed: cute.Tensor,  # e4m3 (I, M)
        d_gate_sc_t_routed: cute.Tensor,  # e8m0 scales of d_gate_fp8_t_routed
        d_up_fp8_routed: cute.Tensor,  # e4m3 (M, I) out
        d_up_sc_routed: cute.Tensor,  # e8m0 scales of d_up_fp8_routed
        d_up_fp8_t_routed: cute.Tensor,  # e4m3 (I, M)
        d_up_sc_t_routed: cute.Tensor,  # e8m0 scales of d_up_fp8_t_routed
        d_w_routed_gate: cute.Tensor,  # bf16 (I, H, E) view of the (E, I, H) out
        d_w_routed_up: cute.Tensor,  # bf16 (I, H, E) view of the (E, I, H) out
        d_w_routed_down: cute.Tensor,  # bf16 (H, I, E) view of the (E, H, I) out
        router_weights: cute.Tensor,  # f32 (M,): the macrobatch's router weights
        d_router_weight_partials: cute.Tensor,  # f32 (M, I / 128)
        router_weights_ready: cute.Tensor,  # int32 counter (macrobatches,)
        d_y_routed_ready: cute.Tensor,  # int32 counter (minibatches + macrobatches,): + the stream's head words
        d_hidden_ready: cute.Tensor,  # int32 counter (row blocks x column blocks,)
        d_gate_up_ready: cute.Tensor,  # int32 counter (row blocks,)
        d_x_routed_ready: cute.Tensor,  # int32 counter (minibatches,)
        replayed_x_routed_ready: cute.Tensor,  # int32 counter (minibatches,)
        replayed_gate_up_ready: cute.Tensor,  # int32 counter (routed row blocks x column blocks,)
        replayed_hidden_ready: cute.Tensor,  # int32 counter (routed row blocks,)
        routed_buffers_done: cute.Tensor,  # int32 counter (macrobatches,)
        tensormaps: cute.Tensor,  # uint8 (ROUTED_TENSORMAPS x 128,): the routed TMA descriptors (_write_tensormaps)
        schedule_capacity: Int32,
        num_local_tokens: Int32,  # T
        intermediate_size: Int32,  # I
        num_local_experts: Int32,  # E
        topk: Int32,
        macrobatch_size: Int32,  # M
        minibatch_size: Int32,
        num_comm_sms: Int32,
        swiglu_limit: Float32,  # clamp limit, 0 when unclamped
        # SiTU-GLU's two betas; unread unless is_situ.
        situ_beta1: Float32,
        situ_beta2: Float32,
        # The Up half's first row when Gate and Up are one combined FC1 passed whole (payloads
        # and scales), 0 otherwise.
        up_row_offset: Int32,
        # (10, num_local_experts) int64 base addresses -- the five weights the dgrad reads
        # (Gate, Up, Gate transposed, Up transposed, Down transposed) and then their five
        # scales -- and the bytes their descriptors live in.  One-element placeholders unless
        # split_weights is set.
        routed_storage_tables: cute.Tensor,
        routed_descriptors: cute.Tensor,
        # Nonzero when routed_descriptors has to be built by this launch: once per set of
        # address tables (persistent_bf16_fwd.RoutedDescriptors).
        build_descriptors: Int32,
        # The Up half's first row inside a combined FC1 weight-gradient buffer -- the caller's
        # main_grad, or this call's own -- 0 when Gate and Up have separate buffers.
        main_grad_up_row_offset: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """The backward's launches on ``stream``: the tensormap prologue, the split-weight descriptor
        build when asked for, the mega kernel, then the zeroing of empty experts' weight gradients
        (skipped when accumulating into main_grad).

        Arguments, in groups (``make_mega_args_mxfp8_bwd`` builds them in this order; T / I / E / M
        are the local tokens, intermediate, local experts and macrobatch size, H the shared hidden and
        Hr the routed hidden -- both read from the tensors, not arguments -- "view" a torch view of the
        named buffer, "out" an output):

        * the five peer address tables (int64 ``(ep_size,)``) of x, dY, routed dX, the router
          weights and their gradient partials, then the schedule (``peer_rank``,
          ``peer_token_idx``, ``num_tokens``, ``tokens_per_expert``);
        * the shared expert: x and dY with their transposes, the context's Gate / Up / Hidden,
          the three BF16 weights, the six BF16 outputs (d_hidden / d_gate / d_up with the two
          transposed gradient views, d_x) and the three weight gradients;
        * the routed weights: Gate / Up e4m3 with scales, their transposes, the transposed Down;
        * the routed MXFP8 context and scratch (x, Gate, Up, Hidden, dY in e4m3 + e8m0, normal and
          transposed), d_hidden / d_x, the quantized d_gate / d_up and the three weight gradients;
        * the router weights of the macrobatch and the router-weight partials;
        * the nine counters (``persistent_mxfp8_bwd._COUNTERS`` order) and the tensormap buffer;
        * the runtime scalars (schedule capacity, geometry, batch sizes, comm SMs, clamp limit, the
          SiTU betas, the combined-FC1 row offsets), the split-weight tables and descriptors, and
          the stream.
        """
        self._launch(
            x_peer_table,
            d_y_peer_table,
            d_x_peer_table,
            router_peer_table,
            d_router_peer_table,
            schedule_peer_rank,
            schedule_peer_token_idx,
            num_tokens,
            tokens_per_expert,
            x_shared,
            x_shared_ht,
            d_y_shared,
            d_y_shared_ht,
            gate_shared,
            up_shared,
            hidden_shared_it,
            w_shared_gate_hi,
            w_shared_up_hi,
            w_shared_down_ih,
            d_hidden_shared,
            d_gate_shared,
            d_up_shared,
            d_gate_shared_it,
            d_up_shared_it,
            d_x_shared,
            d_w_shared_gate,
            d_w_shared_up,
            d_w_shared_down,
            w_routed_gate,
            w_routed_gate_sc,
            w_routed_up,
            w_routed_up_sc,
            w_routed_gate_t,
            w_routed_gate_t_sc,
            w_routed_up_t,
            w_routed_up_t_sc,
            w_routed_down_t,
            w_routed_down_t_sc,
            x_fp8_routed,
            x_sc_routed,
            x_fp8_t_routed,
            x_sc_t_routed,
            gate_routed,
            up_routed,
            hidden_fp8_routed,
            hidden_sc_routed,
            hidden_fp8_t_routed,
            hidden_sc_t_routed,
            d_y_fp8_routed,
            d_y_sc_routed,
            d_y_fp8_t_routed,
            d_y_sc_t_routed,
            d_hidden_routed,
            d_x_routed,
            d_gate_fp8_routed,
            d_gate_sc_routed,
            d_gate_fp8_t_routed,
            d_gate_sc_t_routed,
            d_up_fp8_routed,
            d_up_sc_routed,
            d_up_fp8_t_routed,
            d_up_sc_t_routed,
            d_w_routed_gate,
            d_w_routed_up,
            d_w_routed_down,
            router_weights,
            d_router_weight_partials,
            router_weights_ready,
            d_y_routed_ready,
            d_hidden_ready,
            d_gate_up_ready,
            d_x_routed_ready,
            replayed_x_routed_ready,
            replayed_gate_up_ready,
            replayed_hidden_ready,
            routed_buffers_done,
            tensormaps,
            schedule_capacity,
            num_local_tokens,
            intermediate_size,
            num_local_experts,
            topk,
            macrobatch_size,
            minibatch_size,
            num_comm_sms,
            swiglu_limit,
            situ_beta1,
            situ_beta2,
            up_row_offset,
            routed_storage_tables,
            routed_descriptors,
            build_descriptors,
            main_grad_up_row_offset,
            stream,
        )

    def _launch(
        self,
        x_peer_table: cute.Tensor,
        d_y_peer_table: cute.Tensor,
        d_x_peer_table: cute.Tensor,
        router_peer_table: cute.Tensor,
        d_router_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        x_shared: cute.Tensor,
        x_shared_ht: cute.Tensor,
        d_y_shared: cute.Tensor,
        d_y_shared_ht: cute.Tensor,
        gate_shared: cute.Tensor,
        up_shared: cute.Tensor,
        hidden_shared_it: cute.Tensor,
        w_shared_gate_hi: cute.Tensor,
        w_shared_up_hi: cute.Tensor,
        w_shared_down_ih: cute.Tensor,
        d_hidden_shared: cute.Tensor,
        d_gate_shared: cute.Tensor,
        d_up_shared: cute.Tensor,
        d_gate_shared_it: cute.Tensor,
        d_up_shared_it: cute.Tensor,
        d_x_shared: cute.Tensor,
        d_w_shared_gate: cute.Tensor,
        d_w_shared_up: cute.Tensor,
        d_w_shared_down: cute.Tensor,
        w_routed_gate: cute.Tensor,
        w_routed_gate_sc: cute.Tensor,
        w_routed_up: cute.Tensor,
        w_routed_up_sc: cute.Tensor,
        w_routed_gate_t: cute.Tensor,
        w_routed_gate_t_sc: cute.Tensor,
        w_routed_up_t: cute.Tensor,
        w_routed_up_t_sc: cute.Tensor,
        w_routed_down_t: cute.Tensor,
        w_routed_down_t_sc: cute.Tensor,
        x_fp8_routed: cute.Tensor,
        x_sc_routed: cute.Tensor,
        x_fp8_t_routed: cute.Tensor,
        x_sc_t_routed: cute.Tensor,
        gate_routed: cute.Tensor,
        up_routed: cute.Tensor,
        hidden_fp8_routed: cute.Tensor,
        hidden_sc_routed: cute.Tensor,
        hidden_fp8_t_routed: cute.Tensor,
        hidden_sc_t_routed: cute.Tensor,
        d_y_fp8_routed: cute.Tensor,
        d_y_sc_routed: cute.Tensor,
        d_y_fp8_t_routed: cute.Tensor,
        d_y_sc_t_routed: cute.Tensor,
        d_hidden_routed: cute.Tensor,
        d_x_routed: cute.Tensor,
        d_gate_fp8_routed: cute.Tensor,
        d_gate_sc_routed: cute.Tensor,
        d_gate_fp8_t_routed: cute.Tensor,
        d_gate_sc_t_routed: cute.Tensor,
        d_up_fp8_routed: cute.Tensor,
        d_up_sc_routed: cute.Tensor,
        d_up_fp8_t_routed: cute.Tensor,
        d_up_sc_t_routed: cute.Tensor,
        d_w_routed_gate: cute.Tensor,
        d_w_routed_up: cute.Tensor,
        d_w_routed_down: cute.Tensor,
        router_weights: cute.Tensor,
        d_router_weight_partials: cute.Tensor,
        router_weights_ready: cute.Tensor,
        d_y_routed_ready: cute.Tensor,
        d_hidden_ready: cute.Tensor,
        d_gate_up_ready: cute.Tensor,
        d_x_routed_ready: cute.Tensor,
        replayed_x_routed_ready: cute.Tensor,
        replayed_gate_up_ready: cute.Tensor,
        replayed_hidden_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        tensormaps: cute.Tensor,
        schedule_capacity: Int32,
        num_local_tokens: Int32,
        intermediate_size: Int32,
        num_local_experts: Int32,
        topk: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        num_comm_sms: Int32,
        swiglu_limit: Float32,
        situ_beta1: Float32,
        situ_beta2: Float32,
        up_row_offset: Int32,
        routed_storage_tables: cute.Tensor,
        routed_descriptors: cute.Tensor,
        build_descriptors: Int32,
        main_grad_up_row_offset: Int32,
        stream: cuda.CUstream,
        e4m3_context: Optional[tuple] = None,
    ) -> None:
        """Host half: collectives, TMA atoms (``_B`` / ``_S`` lists), the capacity grid, the launches.

        ``e4m3_context``: ``(gate_e4m3, gate_scales, up_e4m3, up_scales)`` in the e4m3-context variant (whose
        ``gate_routed`` / ``up_routed`` are the BF16 replay ring), ``None`` with the BF16 context.

        Shared-expert views are BF16 (``*_ht`` / ``*_it``: transposed views, ``*_hi``
        / ``*_ih``: the weights as its B operands); routed weights and Wgrads are ``(N, K, E)``
        / ``(M, N, E)`` views of the ``(E, ., .)`` tensors; scales are 512 B tiles.

        A combined FC1 hands Gate and Up the same scale tensors and tells them apart by offset.
        The halves sit on the rowwise scale's M-block axis, as in the forward, but on the
        transposed scale's K-block axis -- transposing the weight makes the combined 2I the
        contraction dimension -- so the two scales take different offsets.
        """
        _B, _S = self.POSITIONS
        abt, atb, routed = self.abt, self.atb, self.routed
        if cutlass.const_expr(not self.has_shared):
            # The kernel reads num_local_tokens as the shared expert's rows alone: its task counts
            # and the readiness counters' offset past the shared row blocks.  A routed-only layer
            # has none.
            num_local_tokens = Int32(0)
        abt.configure(d_y_shared, w_shared_down_ih, d_hidden_shared)
        atb.configure(d_y_shared_ht, hidden_shared_it, d_w_shared_down)
        # w_routed_gate is the rowwise weight, K-major whichever layout the transposed ones are in.
        # One expert's matrix: split allocations are that already, grouped ones slice the L mode.
        gate_one = w_routed_gate
        down_t_one = w_routed_down_t
        if cutlass.const_expr(not self.split_weights):
            gate_one = w_routed_gate[None, None, 0]
            down_t_one = w_routed_down_t[None, None, 0]
        routed.configure(d_y_fp8_routed, gate_one, d_hidden_routed)
        if cutlass.const_expr(self.native_columnwise):
            self.routed_mn.configure(d_y_fp8_routed, down_t_one, d_hidden_routed)
        abt_a_layout = cute.slice_(abt.a_smem_layout_staged, (None, None, None, 0))
        abt_b_layout = cute.slice_(abt.b_smem_layout_staged, (None, None, None, 0))
        atb_a_layout = cute.slice_(atb.a_smem_layout_staged, (None, None, None, 0))
        atb_b_layout = cute.slice_(atb.b_smem_layout_staged, (None, None, None, 0))
        abt.num_tma_load_bytes = (cute.size_in_bytes(BFloat16, abt_a_layout) + cute.size_in_bytes(BFloat16, abt_b_layout)) * CLUSTER_SIZE
        atb.num_tma_load_bytes = (cute.size_in_bytes(BFloat16, atb_a_layout) + cute.size_in_bytes(BFloat16, atb_b_layout)) * CLUSTER_SIZE
        # One A/B ring, one transaction count: BF16 64-K and e4m3 128-K tiles are both 16 KiB.
        assert abt.num_tma_load_bytes == atb.num_tma_load_bytes == routed.num_tma_load_bytes
        assert abt.num_tmem_alloc_cols == atb.num_tmem_alloc_cols
        # The shared reduce-add positions are the lists' tail, in the accumulating variant only.
        pairs = [None] * (_B.COUNT if self.accumulate_wgrad else _B.SH_WD_DOWN_ADD)

        def a_pair(collective, layout, tensor):
            return cute.nvgpu.make_tiled_tma_atom_A(
                cpasync.CopyBulkTensorTileG2SOp(collective.cta_group),
                tensor,
                layout,
                collective.mma_tiler,
                collective.tiled_mma,
                collective.cluster_layout_vmnk.shape,
            )

        def b_pair(collective, layout, tensor):
            return cute.nvgpu.make_tiled_tma_atom_B(
                cpasync.CopyBulkTensorTileG2SOp(collective.cta_group),
                tensor,
                layout,
                collective.mma_tiler,
                collective.tiled_mma,
                collective.cluster_layout_vmnk.shape,
            )

        def epi_pair(collective, tensor, op_type):
            return collective._make_tma_epi_atoms_and_tensors(tensor, collective.epi_smem_layout_staged, collective.epi_tile, op_type=op_type)

        def tile_pair(op, tensor, columns):
            return cpasync.make_tiled_tma_atom(op, tensor, cute.make_layout((SWIGLU_TILE, columns), stride=(columns, 1)), (SWIGLU_TILE, columns))

        for position, tensor in ((_B.SH_A_D_Y, d_y_shared), (_B.SH_A_D_GATE, d_gate_shared), (_B.SH_A_D_UP, d_up_shared)):
            pairs[position] = a_pair(abt, abt_a_layout, tensor)
        for position, tensor in ((_B.SH_B_DOWN, w_shared_down_ih), (_B.SH_B_GATE, w_shared_gate_hi), (_B.SH_B_UP, w_shared_up_hi)):
            pairs[position] = b_pair(abt, abt_b_layout, tensor)
        pairs[_B.SH_D_D_HIDDEN] = epi_pair(abt, d_hidden_shared, "store")
        pairs[_B.SH_D_D_X] = epi_pair(abt, d_x_shared, "store")
        for position, tensor in ((_B.SH_WA_D_Y, d_y_shared_ht), (_B.SH_WA_D_GATE, d_gate_shared_it), (_B.SH_WA_D_UP, d_up_shared_it)):
            pairs[position] = a_pair(atb, atb_a_layout, tensor)
        pairs[_B.SH_WB_HIDDEN] = b_pair(atb, atb_b_layout, hidden_shared_it)
        pairs[_B.SH_WB_X] = b_pair(atb, atb_b_layout, x_shared_ht)
        for position, tensor in ((_B.SH_WD_DOWN, d_w_shared_down), (_B.SH_WD_GATE, d_w_shared_gate), (_B.SH_WD_UP, d_w_shared_up)):
            pairs[position] = epi_pair(atb, tensor, "store")
        if self.accumulate_wgrad:
            for position, tensor in ((_B.SH_WD_DOWN_ADD, d_w_shared_down), (_B.SH_WD_GATE_ADD, d_w_shared_gate), (_B.SH_WD_UP_ADD, d_w_shared_up)):
                pairs[position] = epi_pair(atb, tensor, "add")
        # The SwiGLU-backward tiles (swizzled 128 x 128, two stages).
        swiglu_bwd_layout = utils.make_smem_layout_epi(BFloat16, LayoutEnum.ROW_MAJOR, (SWIGLU_TILE, SWIGLU_TILE), SWIGLU_BWD_STAGES)
        swiglu_tile_layout = cute.select(swiglu_bwd_layout, mode=[0, 1])
        for position, tensor, op in (
            (_B.SH_L_GATE, gate_shared, cpasync.CopyBulkTensorTileG2SOp()),
            (_B.SH_L_UP, up_shared, cpasync.CopyBulkTensorTileG2SOp()),
            (_B.SH_L_D_HIDDEN, d_hidden_shared, cpasync.CopyBulkTensorTileG2SOp()),
            (_B.SH_T_D_GATE, d_gate_shared, cpasync.CopyBulkTensorTileS2GOp()),
            (_B.SH_T_D_UP, d_up_shared, cpasync.CopyBulkTensorTileS2GOp()),
        ):
            pairs[position] = cpasync.make_tiled_tma_atom(op, tensor, swiglu_tile_layout, (SWIGLU_TILE, SWIGLU_TILE))
        # The forward's swizzled SwiGLU tiles (three stages) for the replayed Gate / Up ring.
        swiglu_fwd_layout = utils.make_smem_layout_epi(BFloat16, LayoutEnum.ROW_MAJOR, (SWIGLU_TILE, SWIGLU_TILE), SWIGLU_FWD_STAGES)
        for position, tensor in ((_B.L_REPLAY_GATE, gate_routed), (_B.L_REPLAY_UP, up_routed)):
            pairs[position] = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), tensor, cute.select(swiglu_fwd_layout, mode=[0, 1]), (SWIGLU_TILE, SWIGLU_TILE)
            )
        # Routed e4m3 operands (K-major) and their scales.
        routed_a_layout = cute.slice_(routed.a_smem_layout_staged, (None, None, None, 0))
        routed_b_layout = cute.slice_(routed.b_smem_layout_staged, (None, None, None, 0))
        sfa_layout = cute.slice_(routed.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_layout = cute.slice_(routed.sfb_smem_layout_staged, (None, None, None, 0))
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(routed.cluster_shape_mnk, routed.tiled_mma.thr_id)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(routed.cluster_shape_mnk, routed.tiled_mma.thr_id)
        for position, tensor, scales in (
            (_B.A_X, x_fp8_routed, x_sc_routed),
            (_B.A_D_Y, d_y_fp8_routed, d_y_sc_routed),
            (_B.A_D_GATE, d_gate_fp8_routed, d_gate_sc_routed),
            (_B.A_D_UP, d_up_fp8_routed, d_up_sc_routed),
            (_B.A_D_Y_T, d_y_fp8_t_routed, d_y_sc_t_routed),
            (_B.A_D_GATE_T, d_gate_fp8_t_routed, d_gate_sc_t_routed),
            (_B.A_D_UP_T, d_up_fp8_t_routed, d_up_sc_t_routed),
        ):
            pairs[position] = a_pair(routed, routed_a_layout, tensor)
            pairs[position + SFA_OF] = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op,
                mxgemm.scale_operand(scales, tensor.shape),
                sfa_layout,
                routed.mma_tiler,
                routed.tiled_mma,
                routed.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
        # The transposed weights are the MN-major operands when the columnwise payload is native;
        # the scales are the same either way, so only the data atom changes collective.
        routed_mn_b_layout = routed_b_layout
        if cutlass.const_expr(self.native_columnwise):
            routed_mn_b_layout = cute.slice_(self.routed_mn.b_smem_layout_staged, (None, None, None, 0))
        transposed_weights = (_B.B_DOWN_T, _B.B_GATE_T, _B.B_UP_T)
        for position, tensor, scales in (
            (_B.B_GATE, w_routed_gate, w_routed_gate_sc),
            (_B.B_UP, w_routed_up, w_routed_up_sc),
            (_B.B_DOWN_T, w_routed_down_t, w_routed_down_t_sc),
            (_B.B_GATE_T, w_routed_gate_t, w_routed_gate_t_sc),
            (_B.B_UP_T, w_routed_up_t, w_routed_up_t_sc),
            (_B.B_HIDDEN_T, hidden_fp8_t_routed, hidden_sc_t_routed),
            (_B.B_X_T, x_fp8_t_routed, x_sc_t_routed),
        ):
            # The whole scale: a combined FC1's Up half is a coordinate in the kernel.
            scale_view = mxgemm.scale_operand(scales, tensor.shape)
            if cutlass.const_expr(position == _B.B_UP_T and not self.split_weights):
                # Except the Up weight the dgrad reads transposed, where the combined 2I is the contraction
                # axis and Up starts up_row_offset = I columns along it (0 for separate weights, which keep
                # their shape): payload and scale get Up's own (H, I, E) view here, once per launch, so the
                # dgrad's K loop tiles them from 0 as it tiles separate weights, instead of adding the offset
                # to every K tile's coordinate.  MOK's transposed payload and TransformerEngine's native one
                # alike: the offset runs along the operand's K mode, whatever its stride.  (Split weights
                # have no such view and keep the coordinate, see _routed_gemm.)
                scale_view = mxgemm.scale_operand(scales, tensor.shape, k_offset=up_row_offset)
                tensor = cute.make_tensor(
                    tensor.iterator + Int64(up_row_offset) * Int64(tensor.stride[1]),
                    cute.make_layout((tensor.shape[0], Int32(tensor.shape[1]) - up_row_offset, tensor.shape[2]), stride=tensor.stride),
                )
            if cutlass.const_expr(self.native_columnwise) and position in transposed_weights:
                pairs[position] = b_pair(self.routed_mn, routed_mn_b_layout, tensor)
            else:
                pairs[position] = b_pair(routed, routed_b_layout, tensor)
            pairs[position + SFB_OF] = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op, scale_view, sfb_layout, routed.mma_tiler_sfb, routed.tiled_mma_sfb, routed.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16
            )
        # Routed BF16 outputs through the ABt collective's D3 epilogue.
        for position, tensor, op_type in (
            (_B.D_D_HIDDEN, d_hidden_routed, "store"),
            (_B.D_D_X, d_x_routed, "store"),
            (_B.D_GATE, gate_routed, "store"),
            (_B.D_UP, up_routed, "store"),
        ):
            pairs[position] = epi_pair(abt, tensor, op_type)
        # The routed weight gradients' destinations.  When they accumulate into main_grad their atoms are
        # built with the Wgrad collective, as the shared ones are: its D follows the destination, so a
        # float32 main_grad gets an FP32 epilogue instead of a bfloat16 one.
        w_gate_dst, w_up_dst = d_w_routed_gate, d_w_routed_up
        if cutlass.const_expr(not self.accumulate_wgrad):
            # Fresh gradients of one combined FC1 buffer (Up main_grad_up_row_offset = I rows down; 0 for
            # separate Gate and Up buffers, which keep their shapes): each half gets its own I-row view, so
            # a tail tile's TMA store is clipped at its half instead of running into the next one (Gate's
            # last tile would otherwise overwrite Up's first rows).  Accumulating keeps the whole buffer
            # and the row tile offset: its main_grad is the caller's, and I has to be a multiple of the
            # tile there.
            fc1_rows = Int32(d_w_routed_gate.shape[0]) - main_grad_up_row_offset
            w_gate_dst = cute.make_tensor(d_w_routed_gate.iterator, cute.make_layout((fc1_rows, *d_w_routed_gate.shape[1:]), stride=d_w_routed_gate.stride))
            w_up_dst = cute.make_tensor(
                d_w_routed_up.iterator + Int64(main_grad_up_row_offset) * Int64(d_w_routed_up.stride[0]),
                cute.make_layout((fc1_rows, *d_w_routed_up.shape[1:]), stride=d_w_routed_up.stride),
            )
        for position, tensor, op_type in (
            (_B.D_W_DOWN, d_w_routed_down, "store"),
            (_B.D_W_GATE, w_gate_dst, "store"),
            (_B.D_W_UP, w_up_dst, "store"),
            (_B.D_W_DOWN_ADD, d_w_routed_down, "add"),
            (_B.D_W_GATE_ADD, w_gate_dst, "add"),
            (_B.D_W_UP_ADD, w_up_dst, "add"),
        ):
            # The Wgrad collective when accumulating into main_grad; otherwise the ABt collective's
            # bfloat16 tile, as every other routed task.
            pairs[position] = epi_pair(atb if self.accumulate_wgrad else abt, tensor, op_type)
        store, load = cpasync.CopyBulkTensorTileS2GOp(), cpasync.CopyBulkTensorTileG2SOp()
        if self.E4M3_CONTEXT:
            gate_fp8_routed, gate_sc_routed, up_fp8_routed, up_sc_routed = e4m3_context
            # e4m3 tile stores and SwiGLU-backward tile loads.
            for position, op, tensor, columns in (
                (_B.T_GATE, store, gate_fp8_routed, gemm.EPILOGUE_N),
                (_B.T_UP, store, up_fp8_routed, gemm.EPILOGUE_N),
                (_B.T_HIDDEN, store, hidden_fp8_routed, SWIGLU_TILE),
                (_B.T_HIDDEN_T, store, hidden_fp8_t_routed, SWIGLU_TILE),
                (_B.T_D_GATE, store, d_gate_fp8_routed, SWIGLU_TILE),
                (_B.T_D_UP, store, d_up_fp8_routed, SWIGLU_TILE),
                (_B.T_D_GATE_T, store, d_gate_fp8_t_routed, SWIGLU_TILE),
                (_B.T_D_UP_T, store, d_up_fp8_t_routed, SWIGLU_TILE),
                (_B.L_D_HIDDEN, load, d_hidden_routed, SWIGLU_TILE),
                (_B.L_GATE, load, gate_fp8_routed, SWIGLU_TILE),
                (_B.L_UP, load, up_fp8_routed, SWIGLU_TILE),
            ):
                pairs[position] = tile_pair(op, tensor, columns)
        else:
            # e4m3 tile stores and SwiGLU-backward tile loads (the BF16 Gate / Up context).
            for position, op, tensor, columns in (
                (_B.T_HIDDEN, store, hidden_fp8_routed, SWIGLU_TILE),
                (_B.T_HIDDEN_T, store, hidden_fp8_t_routed, SWIGLU_TILE),
                (_B.T_D_GATE_T, store, d_gate_fp8_t_routed, SWIGLU_TILE),
                (_B.T_D_UP_T, store, d_up_fp8_t_routed, SWIGLU_TILE),
                (_B.L_D_HIDDEN, load, d_hidden_routed, SWIGLU_TILE),
                (_B.L_GATE, load, gate_routed, SWIGLU_TILE),
                (_B.L_UP, load, up_routed, SWIGLU_TILE),
            ):
                pairs[position] = tile_pair(op, tensor, columns)
        assert all(pair is not None for pair in pairs)
        scales = [None] * _S.COUNT
        if self.E4M3_CONTEXT:
            for position, tensor in (
                (_S.D_Y, d_y_sc_routed),
                (_S.D_Y_T, d_y_sc_t_routed),
                (_S.X, x_sc_routed),
                (_S.X_T, x_sc_t_routed),
                (_S.GATE, gate_sc_routed),
                (_S.UP, up_sc_routed),
                (_S.HIDDEN, hidden_sc_routed),
                (_S.HIDDEN_T, hidden_sc_t_routed),
                (_S.D_GATE, d_gate_sc_routed),
                (_S.D_UP, d_up_sc_routed),
                (_S.D_GATE_T, d_gate_sc_t_routed),
                (_S.D_UP_T, d_up_sc_t_routed),
                (_S.D_Y_FP8, d_y_fp8_routed),
                (_S.D_Y_FP8_T, d_y_fp8_t_routed),
                (_S.X_FP8, x_fp8_routed),
                (_S.X_FP8_T, x_fp8_t_routed),
            ):
                scales[position] = tensor
        else:
            for position, tensor in (
                (_S.D_Y, d_y_sc_routed),
                (_S.D_Y_T, d_y_sc_t_routed),
                (_S.X, x_sc_routed),
                (_S.X_T, x_sc_t_routed),
                (_S.HIDDEN, hidden_sc_routed),
                (_S.HIDDEN_T, hidden_sc_t_routed),
                (_S.D_GATE, d_gate_sc_routed),
                (_S.D_UP, d_up_sc_routed),
                (_S.D_GATE_T, d_gate_sc_t_routed),
                (_S.D_UP_T, d_up_sc_t_routed),
                (_S.D_GATE_FP8, d_gate_fp8_routed),
                (_S.D_UP_FP8, d_up_fp8_routed),
                (_S.D_Y_FP8, d_y_fp8_routed),
                (_S.D_Y_FP8_T, d_y_fp8_t_routed),
                (_S.X_FP8, x_fp8_routed),
                (_S.X_FP8_T, x_fp8_t_routed),
            ):
                scales[position] = tensor
        # The two hiddens are the shared input's and d_x_routed's widths (the kernel reads the same tensors), the
        # BF16 backward's shared_hidden_size / routed_hidden_size.  The BF16 backward reads the routed one from
        # x_routed; the MXFP8 backward has no BF16-layout x_routed (its routed x is the e4m3 / scale pair, the
        # transposed context), so the routed dX buffer, the same width, stands in.
        shared_hidden_size = Int32(x_shared.shape[1])
        routed_hidden_size = Int32(d_x_routed.shape[1])
        # The launch grid covers the schedule's capacity; the kernel computes the clusters that have a task
        # from the actual token count (_true_clusters) and the rest drain.
        _, _, _, shared, _, _, _, routed_tasks, _, replay, _, _, wgrad = _task_counts(
            num_local_tokens, shared_hidden_size, routed_hidden_size, intermediate_size, num_local_experts, minibatch_size, self.has_shared, self.has_tail
        )
        capacity_minibatches = _ceil_div_i32(schedule_capacity, minibatch_size)
        capacity_macrobatches = _ceil_div_i32(schedule_capacity, macrobatch_size)
        if cutlass.const_expr(self.recompute_forward_context):
            # Every minibatch of the capacity runs the replay tasks, macrobatch 0's included.
            capacity_clusters = num_comm_sms // Int32(CLUSTER_SIZE) + shared + capacity_minibatches * (routed_tasks + replay) + capacity_macrobatches * wgrad
        else:
            capacity_clusters = (
                num_comm_sms // Int32(CLUSTER_SIZE)
                + shared
                + capacity_minibatches * routed_tasks
                + (capacity_macrobatches - Int32(1)) * (macrobatch_size // minibatch_size) * replay
                + capacity_macrobatches * wgrad
            )
        scheduler_params = utils.ClcDynamicPersistentTileSchedulerParams((capacity_clusters * CLUSTER_SIZE, 1, 1), gemm.CLUSTER_SHAPE)
        # The routed TMA descriptors into their global slots, read by the mega's producer K loops
        # (ROUTED_TMA_POSITIONS).
        self._write_tensormaps([pairs[position][0] for position in ROUTED_TMA_POSITIONS], tensormaps).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
        # One descriptor array per routed operand the dgrad reads, filled by its own launch
        # before the mega kernel: that one is persistent, so every CTA would otherwise race to
        # write the same descriptors.
        # The kernel takes the variant-only arguments only in their variants: None is a compile-time constant,
        # not a kernel parameter, so every other variant's kernel is the one without the feature.
        descriptors = None
        kernel_main_grad_up_row_offset = None
        kernel_situ_beta1 = None
        kernel_situ_beta2 = None
        kernel_routed_mn_mma = None
        kernel_routed_mn_b_layout = None
        if cutlass.const_expr(self.accumulate_wgrad):
            kernel_main_grad_up_row_offset = main_grad_up_row_offset
        if cutlass.const_expr(self.is_situ):
            kernel_situ_beta1 = situ_beta1
            kernel_situ_beta2 = situ_beta2
        if cutlass.const_expr(self.native_columnwise):
            kernel_routed_mn_mma = self.routed_mn.tiled_mma
            kernel_routed_mn_b_layout = self.routed_mn.b_smem_layout_staged
        if cutlass.const_expr(self.split_weights):
            descriptors = cute.make_ptr(
                cutlass.Uint8, routed_descriptors.iterator.toint(), cute.AddressSpace.gmem, assumed_align=routed_descriptors_pkg.BYTES_PER_DESCRIPTOR
            )
            # The ten operands the dgrad reads, then -- split main_grads -- the three reduce-add
            # destinations of the Wgrad, rows MAIN_GRAD_ROWS of the address table.
            templates = tuple(pairs[operand][0] for operand in SPLIT_ORDER)
            if cutlass.const_expr(self.split_main_grads):
                templates = templates + tuple(pairs[operand][0] for operand in sorted(MAIN_GRAD_ROWS, key=MAIN_GRAD_ROWS.get))
            # _launch is not rewritten by the DSL: the runtime branch has to be built explicitly.
            routed_descriptors_pkg.build_routed_descriptors_if(
                build_descriptors, templates, routed_storage_tables, descriptors, Int32(cute.size(routed_storage_tables.shape[1])), stream
            )
        self.kernel(
            abt.tiled_mma,
            atb.tiled_mma,
            routed.tiled_mma,
            routed.tiled_mma_sfb,
            abt.cluster_layout_vmnk,
            routed.cluster_layout_sfb_vmnk,
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            scales,
            abt.a_smem_layout_staged,
            abt.b_smem_layout_staged,
            abt.epi_smem_layout_staged,
            abt.epi_tile,
            atb.a_smem_layout_staged,
            atb.b_smem_layout_staged,
            atb.epi_smem_layout_staged,
            atb.epi_tile,
            routed.a_smem_layout_staged,
            routed.b_smem_layout_staged,
            routed.sfa_smem_layout_staged,
            routed.sfb_smem_layout_staged,
            swiglu_bwd_layout,
            swiglu_fwd_layout,
            scheduler_params,
            x_peer_table,
            d_y_peer_table,
            d_x_peer_table,
            router_peer_table,
            d_router_peer_table,
            schedule_peer_rank,
            schedule_peer_token_idx,
            num_tokens,
            tokens_per_expert,
            router_weights,
            d_router_weight_partials,
            d_x_routed,
            router_weights_ready,
            d_y_routed_ready,
            d_hidden_ready,
            d_gate_up_ready,
            d_x_routed_ready,
            replayed_x_routed_ready,
            replayed_gate_up_ready,
            replayed_hidden_ready,
            routed_buffers_done,
            tensormaps,
            num_local_tokens,
            shared_hidden_size,
            intermediate_size,
            num_local_experts,
            topk,
            macrobatch_size,
            minibatch_size,
            num_comm_sms,
            swiglu_limit,
            up_row_offset,
            descriptors,
            kernel_main_grad_up_row_offset,
            kernel_situ_beta1,
            kernel_situ_beta2,
            kernel_routed_mn_mma,
            kernel_routed_mn_b_layout,
        ).launch(
            grid=(capacity_clusters * CLUSTER_SIZE, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            cluster=gemm.CLUSTER_SHAPE,
            smem=MXFP8_BWD_SHARED_MEMORY_BYTES,
            stream=stream,
            min_blocks_per_mp=1,
        )
        # Experts without tokens get zero gradients (routed hidden).  Accumulating into main_grad
        # skips it: an expert with no rows contributes nothing, and zeroing would erase the earlier
        # microbatches.
        if cutlass.const_expr(not self.accumulate_wgrad):
            # One combined FC1 buffer under both names has 2I rows per expert.
            _zero_empty_routed_wgrads(
                d_w_routed_gate,
                d_w_routed_up,
                d_w_routed_down,
                tokens_per_expert,
                routed_hidden_size,
                intermediate_size,
                num_local_experts,
                intermediate_size + main_grad_up_row_offset,
            ).launch(grid=(128, num_local_experts, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def _write_tensormaps(self, atoms: list[cute.CopyAtom], tensormaps: cute.Tensor):
        """The prologue launch (one warp, right before the mega on the same stream): copies the routed TMA
        descriptors (``atoms``, in ``ROUTED_TMA_POSITIONS`` order) into their ``tensormaps`` slots and releases them
        to the tensormap proxy; the mega's producer warps acquire them once per CTA before their task loop (see
        ``ROUTED_TMA_POSITIONS``)."""
        with cute.arch.elect_one():  # the launch is one warp
            for slot, position in enumerate(ROUTED_TMA_POSITIONS):
                cpasync.copy_tensormap(atoms[slot], _tensormap_ptr(tensormaps, position, cute.AddressSpace.gmem))
            cpasync.fence_tma_desc_release()

    @cute.jit
    def _routed_operand(self, tensors: list, operand: cutlass.Constexpr[int], descriptors, num_experts: Int32, expert: Int32):
        """One routed operand for one expert: the view to tile, and the descriptor to read it with.
        ``descriptors``: the kernel's ``routed_descriptors``, None unless split_weights.

        Grouped: the L mode is sliced and the atom's own descriptor is used.  Split: the operand
        *is* one expert's allocation, so there is no L mode and the expert is named by its
        descriptor.
        """
        if cutlass.const_expr(self.split_weights and operand in SPLIT_ROWS):
            return tensors[operand], routed_descriptors_pkg.descriptor_argument(
                routed_descriptors_pkg.operand_descriptors(descriptors, num_experts, SPLIT_ROWS[operand]), expert
            )
        return tensors[operand][None, None, expert], None

    @cute.jit
    def _routed_gemm(
        self,
        atoms: list[cute.CopyAtom],
        tensors: list[cute.Tensor],
        a: cutlass.Constexpr[int],
        b: cutlass.Constexpr[int],
        d: cutlass.Constexpr[int],
        expert: Int32,
        ctx: list[object],
        task: list[object],
        states: list[object],
        quant: Optional[list[object]] = None,
        a2: cutlass.Constexpr[int] = -1,
        b2: cutlass.Constexpr[int] = -1,
        d_add: cutlass.Constexpr[int] = -1,
        is_wgrad: cutlass.Constexpr[bool] = False,
        b_weight: cutlass.Constexpr[bool] = False,
    ):
        """One routed GEMM call site: operand ``a`` x ``b`` (expert slice when the B operand is a
        weight) into ``d``; ``a2`` / ``b2``: dgrad Gate+Up's second pair; ``d_add``: Wgrad's
        reduce-add for later contributions.  ``ctx``: the kernel's fixed arguments of
        ``_run_routed_gemm``; ``task``: ``(kind, macro, generation, row, column, k_start, k_end,
        first)``; ``states``: the six pipeline states, returned advanced.  ``b_weight``: B is a
        transposed routed weight, so it is MN-major when the columnwise payload is
        TransformerEngine's native one."""
        (
            epi_mma,
            routed_mma,
            routed_mma_sfb,
            cluster_layout_vmnk,
            cluster_layout_sfb_vmnk,
            sA,
            sB,
            sSFA,
            sSFB,
            sD,
            epi_tile,
            arena,
            acc_ptr,
            ab_pipeline,
            sf_pipeline,
            acc_pipeline,
            thread,
            warp,
            cta_rank,
            num_tokens,
            d_y_routed_ready,
            d_gate_up_ready,
            replayed_x_routed_ready,
            replayed_hidden_ready,
            d_hidden_ready,
            d_x_routed_ready,
            replayed_gate_up_ready,
            routed_buffers_done,
            num_local_tokens,
            hidden_size,
            intermediate_size,
            macrobatch_size,
            minibatch_size,
            tensormaps,
            # The variants' own: None wherever the kernel does not take the argument.
            descriptors,
            num_experts,
            atb_mma,
            sD_atb,
            atb_epi_tile,
            up_row_offset,
            main_grad_up_row_offset,
            routed_mn_mma,
            sB_mn,
        ) = ctx
        kind, macro, generation, row, column, k_start, k_end, first = task
        if cutlass.const_expr(is_wgrad and self.accumulate_wgrad):
            # Accumulating into main_grad: a Wgrad's D belongs to the Wgrad collective, whose atoms
            # the host built, so its epilogue partitions with that one (a float32 main_grad gets an
            # FP32 tile).  Otherwise the ABt collective's bfloat16 tile, as every other routed task.
            epi_mma, sD, epi_tile = atb_mma, sD_atb, atb_epi_tile
        d_tensor = tensors[d]
        b_desc = None
        sfb_desc = None
        if cutlass.const_expr(is_wgrad):
            # A Wgrad's B operand is an activation, never expert-indexed.  Its D is the weight
            # gradient: MOK's own [E, N, K] buffer, or one expert's main_grad when those are
            # split -- in which case only the reduce-add atom is ever used.
            b_tensor = tensors[b]
            sfb = tensors[b + SFB_OF]
            if cutlass.const_expr(not self.split_main_grads):
                d_tensor = d_tensor[None, None, expert]
        else:
            b_tensor, b_desc = self._routed_operand(tensors, b, descriptors, num_experts, expert)
            sfb, sfb_desc = self._routed_operand(tensors, b + SFB_OF, descriptors, num_experts, expert)
            if cutlass.const_expr(b == _B.B_UP):
                # Replay Up of a combined FC1: up_row_offset rows down the rowwise payload and its
                # scale (the scale's coordinates are the operand's rows); 0 for separate weights.  A
                # coordinate, so a retargeted descriptor keeps it.
                b_tensor = cute.domain_offset((up_row_offset, Int32(0)), b_tensor)
                sfb = cute.domain_offset((up_row_offset, Int32(0)), sfb)
        a2_atom, a2_tensor, b2_atom, b2_tensor = atoms[a], tensors[a], atoms[b], b_tensor
        sfa2_atom, sfa2, sfb2_atom, sfb2 = atoms[a + SFA_OF], tensors[a + SFA_OF], atoms[b + SFB_OF], sfb
        b2_desc, sfb2_desc = b_desc, sfb_desc
        if cutlass.const_expr(a2 >= 0):
            a2_atom, a2_tensor = atoms[a2], tensors[a2]
            b2_atom = atoms[b2]
            b2_tensor, b2_desc = self._routed_operand(tensors, b2, descriptors, num_experts, expert)
            sfa2_atom, sfa2 = atoms[a2 + SFA_OF], tensors[a2 + SFA_OF]
            sfb2_atom = atoms[b2 + SFB_OF]
            sfb2, sfb2_desc = self._routed_operand(tensors, b2 + SFB_OF, descriptors, num_experts, expert)
            if cutlass.const_expr(b2 == _B.B_UP_T and self.split_weights):
                # Dgrad Up of a combined FC1: the weight read transposed puts the combined 2I on
                # K, for MOK's transposed payload and TransformerEngine's native one alike (both
                # are (H, 2I) operand views), and on the transposed scale's K blocks.  Split weights
                # only: a retargeted descriptor describes one expert's whole operand, so the half is
                # a coordinate; grouped weights were given Up's own view at the launch.
                b2_tensor = cute.domain_offset((Int32(0), up_row_offset), b2_tensor)
                sfb2 = cute.domain_offset((Int32(0), up_row_offset), sfb2)
        d_add_atom = None
        d_add_tensor = None
        d_desc = None
        d_row_tile_offset = None
        if cutlass.const_expr(d_add >= 0):
            d_add_atom = atoms[d_add]
            if cutlass.const_expr(self.split_main_grads):
                # Split main_grad buffers have no E mode: the expert is named by a descriptor.
                d_add_tensor = tensors[d_add]
                d_desc = routed_descriptors_pkg.descriptor_argument(
                    routed_descriptors_pkg.operand_descriptors(descriptors, num_experts, MAIN_GRAD_ROWS[d_add]), expert
                )
            else:
                d_add_tensor = tensors[d_add][None, None, expert]
            if cutlass.const_expr(d_add == _B.D_W_UP_ADD and self.accumulate_wgrad):
                # A combined FC1's gradient is one buffer whose Up half starts I rows down, grouped
                # or split.  In CTA tiles.  Fresh gradients already address the Up half through its
                # own view (host side).
                d_row_tile_offset = main_grad_up_row_offset // Int32(gemm.CTA_M)
        ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer = states
        b_mn_major = b_weight and self.native_columnwise
        b_mma = routed_mn_mma if b_mn_major else routed_mma
        sB_operand = sB_mn if b_mn_major else sB
        return self._run_routed_gemm(
            epi_mma,
            routed_mma,
            b_mma,
            routed_mma_sfb,
            cluster_layout_vmnk,
            cluster_layout_sfb_vmnk,
            atoms[a],
            tensors[a],
            atoms[b],
            b_tensor,
            atoms[a + SFA_OF],
            tensors[a + SFA_OF],
            atoms[b + SFB_OF],
            sfb,
            a2_atom,
            a2_tensor,
            b2_atom,
            b2_tensor,
            sfa2_atom,
            sfa2,
            sfb2_atom,
            sfb2,
            atoms[d],
            d_tensor,
            quant,
            sA,
            sB_operand,
            sSFA,
            sSFB,
            sD,
            epi_tile,
            arena,
            acc_ptr,
            ab_pipeline,
            sf_pipeline,
            acc_pipeline,
            ab_producer,
            ab_consumer,
            sf_producer,
            sf_consumer,
            acc_producer,
            acc_consumer,
            thread,
            warp,
            cta_rank,
            kind,
            macro,
            generation,
            row,
            column,
            k_start,
            k_end,
            first,
            num_tokens,
            d_y_routed_ready,
            d_gate_up_ready,
            replayed_x_routed_ready,
            replayed_hidden_ready,
            d_hidden_ready,
            d_x_routed_ready,
            replayed_gate_up_ready,
            routed_buffers_done,
            num_local_tokens,
            hidden_size,
            intermediate_size,
            macrobatch_size,
            minibatch_size,
            is_wgrad,
            a2 >= 0,
            tensormaps,
            a,
            b,
            a2,
            b2,
            d_add_atom,
            d_add_tensor,
            b_mn_major,
            b_desc_ptr=b_desc,
            sfb_desc_ptr=sfb_desc,
            b2_desc_ptr=b2_desc,
            sfb2_desc_ptr=sfb2_desc,
            d_desc_ptr=d_desc,
            d_row_tile_offset=d_row_tile_offset,
        )

    @cute.kernel
    def kernel(
        self,
        abt_mma: cute.TiledMma,
        atb_mma: cute.TiledMma,
        routed_mma: cute.TiledMma,
        routed_mma_sfb: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        atoms: list[cute.CopyAtom],
        tensors: list[cute.Tensor],
        scales: list[cute.Tensor],
        abt_a_layout: cute.ComposedLayout,
        abt_b_layout: cute.ComposedLayout,
        abt_epi_layout: cute.ComposedLayout,
        abt_epi_tile: cute.Tile,
        atb_a_layout: cute.ComposedLayout,
        atb_b_layout: cute.ComposedLayout,
        atb_epi_layout: cute.ComposedLayout,
        atb_epi_tile: cute.Tile,
        routed_a_layout: cute.ComposedLayout,
        routed_b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        swiglu_bwd_layout: cute.ComposedLayout,
        swiglu_fwd_layout: cute.ComposedLayout,
        scheduler_params: utils.ClcDynamicPersistentTileSchedulerParams,
        x_peer_table: cute.Tensor,
        d_y_peer_table: cute.Tensor,
        d_x_peer_table: cute.Tensor,
        router_peer_table: cute.Tensor,
        d_router_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens_tensor: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        router_weights: cute.Tensor,
        d_router_weight_partials: cute.Tensor,
        d_x_routed: cute.Tensor,
        router_weights_ready: cute.Tensor,
        d_y_routed_ready: cute.Tensor,
        d_hidden_ready: cute.Tensor,
        d_gate_up_ready: cute.Tensor,
        d_x_routed_ready: cute.Tensor,
        replayed_x_routed_ready: cute.Tensor,
        replayed_gate_up_ready: cute.Tensor,
        replayed_hidden_ready: cute.Tensor,
        routed_buffers_done: cute.Tensor,
        tensormaps: cute.Tensor,
        num_local_tokens: Int32,
        shared_hidden_size: Int32,
        intermediate_size: Int32,
        num_local_experts: Int32,
        topk: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        num_comm_sms: Int32,
        swiglu_limit: Float32,
        # Where Up starts in the tensors passed as the routed Up weight and scale: the rows of Gate when Gate
        # and Up are one combined FC1 parameter, 0 when they are separate weights.
        up_row_offset: Int32,
        # Variant-only arguments, None in every other variant -- hence no annotation: the DSL casts an
        # annotated argument, and None is what keeps them out of the kernel's parameters.
        # routed_descriptors (split_weights): the per-expert TMA descriptors of the routed operands the
        # dgrad reads, one array per operand in SPLIT_ROWS order.  main_grad_up_row_offset
        # (accumulate_wgrad): Int32.  situ_beta1 / situ_beta2 (is_situ): Float32.
        # routed_mn_mma / routed_mn_b_layout (native_columnwise): the MN-major B collective's.
        routed_descriptors,
        main_grad_up_row_offset,
        situ_beta1,
        situ_beta2,
        routed_mn_mma,
        routed_mn_b_layout,
    ):
        """The mega kernel.  The first ``num_comm_sms / 2`` clusters run the communication prefix
        (``_run_mxfp8_comm_prefix``); every other cluster takes compute tasks from the CLC scheduler
        (``_decode_mxfp8_task``) while the scheduled cluster index is below ``_true_clusters``, then
        drains the capacity tail of the grid.  Tasks wait on the readiness counters for their operands and
        arrive on them after their stores complete."""
        _B, _S = self.POSITIONS
        thread = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_index = cute.arch.make_warp_uniform(cute.arch.cluster_idx()[0])
        num_tokens = Int32(num_tokens_tensor[0])
        # d_x_routed's width is the routed chain's hidden (``shared_hidden_size`` the shared chain's); both are runtime.
        routed_hidden_size = Int32(d_x_routed.shape[1])
        comm_clusters = num_comm_sms // Int32(CLUSTER_SIZE)
        true_clusters = self._true_clusters(
            comm_clusters,
            num_tokens,
            num_local_tokens,
            shared_hidden_size,
            routed_hidden_size,
            intermediate_size,
            num_local_experts,
            macrobatch_size,
            minibatch_size,
        )
        storage = utils.SmemAllocator().allocate(_Mxfp8BwdSharedStorage)
        arena = storage.arena.data_ptr().align(min_align=1024)
        swiglu_fwd_barriers = storage.swiglu_fwd_mbarriers.data_ptr()
        swiglu_bwd_barriers = storage.swiglu_bwd_mbarriers.data_ptr()
        stream_mbarriers = storage.stream_mbarriers.data_ptr()
        clc_drain_responses = storage.clc_drain_responses.data_ptr()
        # Dynamic regions capture typed pointers, not the Python storage object.
        ab_mbarriers = storage.ab_full.data_ptr()
        acc_mbarriers = storage.acc_full.ptr
        sf_mbarriers = storage.sf_full.data_ptr()
        clc_mbarriers = storage.clc_mbarriers.data_ptr()
        clc_drain_mbarriers = storage.clc_drain_mbarriers.data_ptr()
        clc_response = storage.clc_response.data_ptr()
        tmem_address = storage.tmem_address.ptr
        tmem_dealloc_mbarrier = storage.tmem_dealloc.ptr
        # The BF16 forward's per-CTA expert table (_persistent_bf16_fwd_mega, TaskDecode): the routed GEMM
        # tasks' owner decode and Wgrad K ranges read it instead of scanning tokens_per_expert in global
        # memory.  The MXFP8 backward's rows are 256-aligned, so the forward's row-block table serves
        # the backward as well.
        s_expert_cu_end = cute.make_tensor(storage.expert_cu_end.data_ptr(), cute.make_layout(EXPERT_TABLE_CAP))
        expert_stride, expert_groups = _build_expert_cu_end(tokens_per_expert, s_expert_cu_end, self.task_decode, thread)
        if cutlass.const_expr(self.split_weights):
            # Once per CTA, before any task: the routed descriptors are built by the launch that
            # precedes this kernel and do not change while it runs.
            for operand in cutlass.range_constexpr(SPLIT_WITH_MAIN_GRADS if self.split_main_grads else SPLIT_OPERANDS):
                routed_descriptors_pkg.acquire_descriptors(
                    routed_descriptors_pkg.operand_descriptors(routed_descriptors, num_local_experts, operand), num_local_experts
                )
        # The Wgrad K ranges stop at each expert's valid rows (table tail; -1 = padded range).
        valid_end_offset = _write_expert_valid_ends(schedule_peer_rank, s_expert_cu_end, num_local_experts, self.task_decode, thread)
        if cluster_index < comm_clusters:
            self._run_mxfp8_comm_prefix(
                x_peer_table,
                d_y_peer_table,
                d_x_peer_table,
                router_peer_table,
                d_router_peer_table,
                schedule_peer_rank,
                schedule_peer_token_idx,
                num_tokens,
                router_weights,
                d_x_routed,
                d_router_weight_partials,
                router_weights_ready,
                d_y_routed_ready,
                d_x_routed_ready,
                replayed_x_routed_ready,
                routed_buffers_done,
                arena,
                stream_mbarriers,
                cluster_index,
                cta_rank,
                [scales[_S.D_Y_FP8], scales[_S.D_Y_FP8_T], scales[_S.D_Y], scales[_S.D_Y_T]],
                [scales[_S.X_FP8], scales[_S.X_FP8_T], scales[_S.X], scales[_S.X_T]],
                num_local_tokens,
                shared_hidden_size,
                routed_hidden_size,
                intermediate_size,
                num_local_experts,
                topk,
                macrobatch_size,
                minibatch_size,
                num_comm_sms,
            )
        elif cluster_index < true_clusters:
            ab_pipeline = PipelineTmaUmma.create(
                num_stages=gemm.AB_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.abt.num_mcast_ctas_a + self.abt.num_mcast_ctas_b - 1),
                tx_count=self.abt.num_tma_load_bytes,
                barrier_storage=ab_mbarriers,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            acc_pipeline = PipelineUmmaAsync.create(
                num_stages=gemm.ACC_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.abt.epilog_warp_id) * CLUSTER_SIZE),
                barrier_storage=acc_mbarriers,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            # The routed scales' own ring.
            sf_pipeline = PipelineTmaUmma.create(
                num_stages=gemm.AB_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=self.routed.num_sf_load_bytes,
                barrier_storage=sf_mbarriers,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            if thread == Int32(0):
                for stage in cutlass.range_constexpr(SWIGLU_FWD_STAGES):
                    cute.arch.mbarrier_init(swiglu_fwd_barriers + stage, 1)
                for stage in cutlass.range_constexpr(SWIGLU_BWD_STAGES):
                    cute.arch.mbarrier_init(swiglu_bwd_barriers + stage, 1)
            clc_layout = cute.make_layout((1, CLUSTER_SIZE, 1, 1))
            clc_pipeline = pipeline.PipelineClcFetchAsync.create(
                barrier_storage=clc_mbarriers,
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 16),
                tx_count=16,
                cta_layout_vmnk=clc_layout,
                defer_sync=True,
            )
            drain_pipeline = pipeline.PipelineClcFetchAsync.create(
                barrier_storage=clc_drain_mbarriers,
                num_stages=8,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2),
                tx_count=16,
                cta_layout_vmnk=clc_layout,
                defer_sync=True,
            )
            # All of the TMEM columns: the accumulator, then the scale slots.
            tmem = utils.TmemAllocator(
                tmem_address,
                barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=mxgemm.TMEM_ALLOC_BARRIER_ID, num_threads=THREADS_PER_CTA),
                allocator_warp_id=0,
                is_two_cta=True,
                two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbarrier,
            )
            tmem.allocate(mxgemm.MXFP8_TMEM_COLUMNS)
            # The aligned barrier of wait_for_alloc() is valid here: all 256 threads of the CTA reach this
            # instruction (the enclosing condition has one value per cluster), before any warp-role branch.
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(Float32)
            pipeline_init_arrive(cluster_shape_mn=gemm.CLUSTER_SHAPE, is_relaxed=True)
            pipeline_init_wait(cluster_shape_mn=gemm.CLUSTER_SHAPE)
            # The shared expert's BF16 views of the arena, then the routed e4m3 ones.
            d_offset = mxgemm.MXFP8_D_OFFSET
            sA_abt = cute.make_tensor(cute.recast_ptr(arena, abt_a_layout.inner, dtype=BFloat16), abt_a_layout.outer)
            sB_abt = cute.make_tensor(cute.recast_ptr(arena + gemm.A_BYTES, abt_b_layout.inner, dtype=BFloat16), abt_b_layout.outer)
            sD_abt = cute.make_tensor(cute.recast_ptr(arena + d_offset, abt_epi_layout.inner, dtype=BFloat16), abt_epi_layout.outer)
            sA_atb = cute.make_tensor(cute.recast_ptr(arena, atb_a_layout.inner, dtype=BFloat16), atb_a_layout.outer)
            sB_atb = cute.make_tensor(cute.recast_ptr(arena + gemm.A_BYTES, atb_b_layout.inner, dtype=BFloat16), atb_b_layout.outer)
            # The Wgrad staging tile follows the collective: FP32 straight into a float32
            # main_grad, bfloat16 otherwise.  One FP32 tile fits where the three BF16 ones do.
            atb_d_dtype = Float32 if cutlass.const_expr(self.fp32_main_grad) else BFloat16
            sD_atb = cute.make_tensor(cute.recast_ptr(arena + d_offset, atb_epi_layout.inner, dtype=atb_d_dtype), atb_epi_layout.outer)
            sA = cute.make_tensor(cute.recast_ptr(arena, routed_a_layout.inner, dtype=Float8E4M3FN), routed_a_layout.outer)
            sB = cute.make_tensor(cute.recast_ptr(arena + gemm.A_BYTES, routed_b_layout.inner, dtype=Float8E4M3FN), routed_b_layout.outer)
            # The same ring bytes seen as MN-major stages: the tile, its swizzle and its size are
            # identical, only MN and K swap which one is contiguous.
            sB_mn = sB
            if cutlass.const_expr(self.native_columnwise):
                sB_mn = cute.make_tensor(cute.recast_ptr(arena + gemm.A_BYTES, routed_mn_b_layout.inner, dtype=Float8E4M3FN), routed_mn_b_layout.outer)
            sSFA = cute.make_tensor(cute.recast_ptr(arena + gemm.A_BYTES + gemm.B_BYTES, dtype=Float8E8M0FNU), sfa_layout)
            sSFB = cute.make_tensor(cute.recast_ptr(arena + gemm.A_BYTES + gemm.B_BYTES + mxgemm.SFA_BYTES, dtype=Float8E8M0FNU), sfb_layout)
            s_swiglu_d_hidden = cute.make_tensor(cute.recast_ptr(arena, swiglu_bwd_layout.inner, dtype=BFloat16), swiglu_bwd_layout.outer)
            s_swiglu_gate = cute.make_tensor(cute.recast_ptr(arena + 2 * BF16_TILE_BYTES, swiglu_bwd_layout.inner, dtype=BFloat16), swiglu_bwd_layout.outer)
            s_swiglu_up = cute.make_tensor(cute.recast_ptr(arena + 4 * BF16_TILE_BYTES, swiglu_bwd_layout.inner, dtype=BFloat16), swiglu_bwd_layout.outer)
            shared_swiglu_loads = [_B.SH_L_GATE, _B.SH_L_UP, _B.SH_L_D_HIDDEN]
            shared_swiglu_stores = [_B.SH_T_D_GATE, _B.SH_T_D_UP]
            # Accumulating: the shared weight gradients are added into main_grad, like the routed
            # ones, so each task carries the reduce-add atom over the same tensor and no
            # contribution ever stores plainly -- there is one contribution per layer, and
            # main_grad already holds the earlier microbatches.
            shared_wgrad_add = [(None, None)] * 3
            shared_wgrad_first = Int32(1)
            if cutlass.const_expr(self.accumulate_wgrad):
                shared_wgrad_add = [(atoms[position], tensors[position]) for position in (_B.SH_WD_DOWN_ADD, _B.SH_WD_GATE_ADD, _B.SH_WD_UP_ADD)]
                shared_wgrad_first = Int32(0)
            routed_ctx = [
                abt_mma,
                routed_mma,
                routed_mma_sfb,
                cluster_layout_vmnk,
                cluster_layout_sfb_vmnk,
                sA,
                sB,
                sSFA,
                sSFB,
                sD_abt,
                abt_epi_tile,
                arena,
                acc_ptr,
                ab_pipeline,
                sf_pipeline,
                acc_pipeline,
                thread,
                warp,
                cta_rank,
                num_tokens,
                d_y_routed_ready,
                d_gate_up_ready,
                replayed_x_routed_ready,
                replayed_hidden_ready,
                d_hidden_ready,
                d_x_routed_ready,
                replayed_gate_up_ready,
                routed_buffers_done,
                num_local_tokens,
                routed_hidden_size,
                intermediate_size,
                macrobatch_size,
                minibatch_size,
                tensormaps,  # the routed tasks' hidden
                routed_descriptors,
                num_local_experts,
                atb_mma,
                sD_atb,
                atb_epi_tile,
                up_row_offset,
                main_grad_up_row_offset,
                routed_mn_mma,
                sB_mn,
            ]
            scheduler = utils.ClcDynamicPersistentTileScheduler.create(scheduler_params, cute.arch.block_idx(), cute.arch.grid_dim(), clc_response)
            clc_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            clc_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.ProducerConsumer, 1)
            ab_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, gemm.AB_STAGES)
            ab_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, gemm.AB_STAGES)
            sf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, gemm.AB_STAGES)
            sf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, gemm.AB_STAGES)
            acc_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, gemm.ACC_STAGES)
            acc_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, gemm.ACC_STAGES)
            swiglu_fwd_phases = Int32(0)
            swiglu_bwd_phases = Int32(0)
            work = scheduler.initial_work_tile_info()
            logical_cluster = work.tile_idx[0] // Int32(CLUSTER_SIZE)
            active = Int32(work.is_valid_tile and logical_cluster >= comm_clusters and logical_cluster < true_clusters)
            # The task table's constants once per CTA; the per-task decode reads them.
            task_table = self._task_table(
                num_tokens, num_local_tokens, shared_hidden_size, routed_hidden_size, intermediate_size, num_local_experts, macrobatch_size, minibatch_size
            )
            # The routed TMA descriptors (the prologue launch's global copies) into the tensormap proxy once per CTA,
            # before this CTA's first use of them: in each producer warp the elected lane, which issues every TMA of
            # the warp, fences the descriptors it issues copies with; the warp syncs after (see ROUTED_TMA_POSITIONS).
            if warp == Int32(AB_TMA_PRODUCER_WARP):
                with cute.arch.elect_one():
                    for position in AB_TMA_POSITIONS:
                        cpasync.fence_tma_desc_acquire(_tensormap_ptr(tensormaps, position))
                cute.arch.sync_warp()
            elif warp == Int32(SF_PRODUCER_WARP):
                with cute.arch.elect_one():
                    for position in SF_TMA_POSITIONS:
                        cpasync.fence_tma_desc_acquire(_tensormap_ptr(tensormaps, position))
                cute.arch.sync_warp()
            while active != Int32(0):
                kind, macro, generation, local = self._decode_mxfp8_task(logical_cluster - comm_clusters, *task_table)
                if warp == Int32(CLC_PRODUCER_WARP) and cta_rank == Int32(0):
                    clc_pipeline.producer_acquire(clc_producer)
                    scheduler.advance_to_next_work(clc_pipeline.producer_get_barrier(clc_producer))
                    clc_producer.advance()
                #
                # CTA-local tasks: the SwiGLU backward (shared, routed) and the replayed SwiGLU
                #
                if kind == Int32(SHARED_SWIGLU_BWD):
                    swiglu_bwd_phases = self._run_shared_swiglu_bwd(
                        warp,
                        cta_rank,
                        local,
                        d_hidden_ready,
                        d_gate_up_ready,
                        [atoms[i] for i in shared_swiglu_loads],
                        [tensors[i] for i in shared_swiglu_loads],
                        [atoms[i] for i in shared_swiglu_stores],
                        [tensors[i] for i in shared_swiglu_stores],
                        s_swiglu_d_hidden,
                        s_swiglu_gate,
                        s_swiglu_up,
                        swiglu_bwd_barriers,
                        swiglu_bwd_phases,
                        num_local_tokens=num_local_tokens,
                        intermediate_size=intermediate_size,
                        swiglu_limit=swiglu_limit,
                        situ_beta1=situ_beta1,
                        situ_beta2=situ_beta2,
                    )
                elif kind == Int32(ROUTED_SWIGLU_BWD):
                    if cutlass.const_expr(self.E4M3_CONTEXT):
                        swiglu_bwd_phases = self._run_routed_swiglu_bwd_e4m3_context(
                            [atoms[_B.L_D_HIDDEN], atoms[_B.L_GATE], atoms[_B.L_UP]],
                            [tensors[_B.L_D_HIDDEN], tensors[_B.L_GATE], tensors[_B.L_UP]],
                            [atoms[_B.T_D_GATE], atoms[_B.T_D_UP], atoms[_B.T_D_GATE_T], atoms[_B.T_D_UP_T]],
                            [tensors[_B.T_D_GATE], tensors[_B.T_D_UP], tensors[_B.T_D_GATE_T], tensors[_B.T_D_UP_T]],
                            [scales[_S.GATE], scales[_S.UP], scales[_S.D_GATE], scales[_S.D_UP], scales[_S.D_GATE_T], scales[_S.D_UP_T]],
                            arena,
                            swiglu_bwd_barriers,
                            swiglu_bwd_phases,
                            schedule_peer_rank,
                            router_weights,
                            d_router_weight_partials,
                            d_hidden_ready,
                            replayed_gate_up_ready,
                            d_gate_up_ready,
                            routed_buffers_done,
                            cta_rank,
                            generation,
                            macro,
                            local,
                            num_tokens,
                            num_local_tokens,
                            intermediate_size,
                            macrobatch_size,
                            minibatch_size,
                            swiglu_limit,
                            situ_beta1,
                            situ_beta2,
                        )
                    else:
                        swiglu_bwd_phases = self._run_routed_swiglu_bwd(
                            [atoms[_B.L_D_HIDDEN], atoms[_B.L_GATE], atoms[_B.L_UP]],
                            [tensors[_B.L_D_HIDDEN], tensors[_B.L_GATE], tensors[_B.L_UP]],
                            [atoms[_B.T_D_GATE_T], atoms[_B.T_D_UP_T]],
                            [tensors[_B.T_D_GATE_T], tensors[_B.T_D_UP_T]],
                            [scales[_S.D_GATE], scales[_S.D_UP], scales[_S.D_GATE_T], scales[_S.D_UP_T], scales[_S.D_GATE_FP8], scales[_S.D_UP_FP8]],
                            arena,
                            swiglu_bwd_barriers,
                            swiglu_bwd_phases,
                            schedule_peer_rank,
                            router_weights,
                            d_router_weight_partials,
                            d_hidden_ready,
                            replayed_gate_up_ready,
                            d_gate_up_ready,
                            routed_buffers_done,
                            cta_rank,
                            generation,
                            macro,
                            local,
                            num_tokens,
                            num_local_tokens,
                            intermediate_size,
                            macrobatch_size,
                            minibatch_size,
                            swiglu_limit,
                            situ_beta1,
                            situ_beta2,
                        )
                elif kind == Int32(REPLAY_SWIGLU):
                    swiglu_fwd_phases = self._run_replay_swiglu(
                        local,
                        generation,
                        cta_rank,
                        num_tokens,
                        [atoms[_B.L_REPLAY_GATE], atoms[_B.L_REPLAY_UP]],
                        [tensors[_B.L_REPLAY_GATE], tensors[_B.L_REPLAY_UP]],
                        swiglu_fwd_layout,
                        arena,
                        swiglu_fwd_barriers,
                        swiglu_fwd_phases,
                        replayed_gate_up_ready,
                        replayed_hidden_ready,
                        intermediate_size,
                        macrobatch_size,
                        minibatch_size,
                        swiglu_limit,
                        [atoms[_B.T_HIDDEN], tensors[_B.T_HIDDEN], atoms[_B.T_HIDDEN_T], tensors[_B.T_HIDDEN_T], scales[_S.HIDDEN], scales[_S.HIDDEN_T]],
                        situ_beta1,
                        situ_beta2,
                    )
                #
                # Shared-expert GEMM tasks (BF16, the base class's task functions)
                #
                elif kind <= Int32(SHARED_WGRAD_UP):
                    valid, _, row, column, k_start, k_end, _ = self._gemm_geometry(
                        kind,
                        macro,
                        generation,
                        local,
                        num_tokens,
                        tokens_per_expert,
                        tokens_per_expert,
                        tokens_per_expert,
                        Int32(0),
                        num_local_tokens=num_local_tokens,
                        hidden_size=shared_hidden_size,
                        intermediate_size=intermediate_size,
                        num_local_experts=num_local_experts,
                        macrobatch_size=macrobatch_size,
                        minibatch_size=minibatch_size,
                    )
                    if valid != Int32(0):
                        if kind == Int32(SHARED_DOWN_DGRAD):
                            ab_producer, ab_consumer, acc_producer, acc_consumer = self._run_one_gemm(
                                self.abt,
                                abt_mma,
                                atoms[_B.SH_A_D_Y],
                                tensors[_B.SH_A_D_Y],
                                atoms[_B.SH_B_DOWN],
                                tensors[_B.SH_B_DOWN],
                                atoms[_B.SH_A_D_Y],
                                tensors[_B.SH_A_D_Y],
                                atoms[_B.SH_B_DOWN],
                                tensors[_B.SH_B_DOWN],
                                atoms[_B.SH_D_D_HIDDEN],
                                tensors[_B.SH_D_D_HIDDEN],
                                sA_abt,
                                sB_abt,
                                sD_abt,
                                abt_epi_tile,
                                acc_ptr,
                                ab_pipeline,
                                acc_pipeline,
                                ab_producer,
                                ab_consumer,
                                acc_producer,
                                acc_consumer,
                                thread,
                                warp,
                                cta_rank,
                                kind,
                                macro,
                                generation,
                                row,
                                column,
                                k_start,
                                k_end,
                                num_tokens,
                                d_y_routed_ready,
                                d_gate_up_ready,
                                replayed_x_routed_ready,
                                d_hidden_ready,
                                d_x_routed_ready,
                                replayed_gate_up_ready,
                                routed_buffers_done,
                                False,
                                False,
                                num_local_tokens=num_local_tokens,
                                hidden_size=shared_hidden_size,
                                intermediate_size=intermediate_size,
                                macrobatch_size=macrobatch_size,
                                minibatch_size=minibatch_size,
                            )
                        elif kind == Int32(SHARED_GATE_UP_DGRAD):
                            ab_producer, ab_consumer, acc_producer, acc_consumer = self._run_one_gemm(
                                self.abt,
                                abt_mma,
                                atoms[_B.SH_A_D_GATE],
                                tensors[_B.SH_A_D_GATE],
                                atoms[_B.SH_B_GATE],
                                tensors[_B.SH_B_GATE],
                                atoms[_B.SH_A_D_UP],
                                tensors[_B.SH_A_D_UP],
                                atoms[_B.SH_B_UP],
                                tensors[_B.SH_B_UP],
                                atoms[_B.SH_D_D_X],
                                tensors[_B.SH_D_D_X],
                                sA_abt,
                                sB_abt,
                                sD_abt,
                                abt_epi_tile,
                                acc_ptr,
                                ab_pipeline,
                                acc_pipeline,
                                ab_producer,
                                ab_consumer,
                                acc_producer,
                                acc_consumer,
                                thread,
                                warp,
                                cta_rank,
                                kind,
                                macro,
                                generation,
                                row,
                                column,
                                k_start,
                                k_end,
                                num_tokens,
                                d_y_routed_ready,
                                d_gate_up_ready,
                                replayed_x_routed_ready,
                                d_hidden_ready,
                                d_x_routed_ready,
                                replayed_gate_up_ready,
                                routed_buffers_done,
                                False,
                                True,
                                num_local_tokens=num_local_tokens,
                                hidden_size=shared_hidden_size,
                                intermediate_size=intermediate_size,
                                macrobatch_size=macrobatch_size,
                                minibatch_size=minibatch_size,
                            )
                        elif kind == Int32(SHARED_WGRAD_DOWN):
                            ab_producer, ab_consumer, acc_producer, acc_consumer = self._run_one_gemm(
                                self.atb,
                                atb_mma,
                                atoms[_B.SH_WA_D_Y],
                                tensors[_B.SH_WA_D_Y],
                                atoms[_B.SH_WB_HIDDEN],
                                tensors[_B.SH_WB_HIDDEN],
                                atoms[_B.SH_WA_D_Y],
                                tensors[_B.SH_WA_D_Y],
                                atoms[_B.SH_WB_HIDDEN],
                                tensors[_B.SH_WB_HIDDEN],
                                atoms[_B.SH_WD_DOWN],
                                tensors[_B.SH_WD_DOWN],
                                sA_atb,
                                sB_atb,
                                sD_atb,
                                atb_epi_tile,
                                acc_ptr,
                                ab_pipeline,
                                acc_pipeline,
                                ab_producer,
                                ab_consumer,
                                acc_producer,
                                acc_consumer,
                                thread,
                                warp,
                                cta_rank,
                                kind,
                                macro,
                                generation,
                                row,
                                column,
                                k_start,
                                k_end,
                                num_tokens,
                                d_y_routed_ready,
                                d_gate_up_ready,
                                replayed_x_routed_ready,
                                d_hidden_ready,
                                d_x_routed_ready,
                                replayed_gate_up_ready,
                                routed_buffers_done,
                                True,
                                False,
                                d_add_atom=shared_wgrad_add[0][0],
                                d_add_tensor=shared_wgrad_add[0][1],
                                first_contribution=shared_wgrad_first,
                                num_local_tokens=num_local_tokens,
                                hidden_size=shared_hidden_size,
                                intermediate_size=intermediate_size,
                                macrobatch_size=macrobatch_size,
                                minibatch_size=minibatch_size,
                            )
                        elif kind == Int32(SHARED_WGRAD_GATE):
                            ab_producer, ab_consumer, acc_producer, acc_consumer = self._run_one_gemm(
                                self.atb,
                                atb_mma,
                                atoms[_B.SH_WA_D_GATE],
                                tensors[_B.SH_WA_D_GATE],
                                atoms[_B.SH_WB_X],
                                tensors[_B.SH_WB_X],
                                atoms[_B.SH_WA_D_GATE],
                                tensors[_B.SH_WA_D_GATE],
                                atoms[_B.SH_WB_X],
                                tensors[_B.SH_WB_X],
                                atoms[_B.SH_WD_GATE],
                                tensors[_B.SH_WD_GATE],
                                sA_atb,
                                sB_atb,
                                sD_atb,
                                atb_epi_tile,
                                acc_ptr,
                                ab_pipeline,
                                acc_pipeline,
                                ab_producer,
                                ab_consumer,
                                acc_producer,
                                acc_consumer,
                                thread,
                                warp,
                                cta_rank,
                                kind,
                                macro,
                                generation,
                                row,
                                column,
                                k_start,
                                k_end,
                                num_tokens,
                                d_y_routed_ready,
                                d_gate_up_ready,
                                replayed_x_routed_ready,
                                d_hidden_ready,
                                d_x_routed_ready,
                                replayed_gate_up_ready,
                                routed_buffers_done,
                                True,
                                False,
                                d_add_atom=shared_wgrad_add[1][0],
                                d_add_tensor=shared_wgrad_add[1][1],
                                first_contribution=shared_wgrad_first,
                                num_local_tokens=num_local_tokens,
                                hidden_size=shared_hidden_size,
                                intermediate_size=intermediate_size,
                                macrobatch_size=macrobatch_size,
                                minibatch_size=minibatch_size,
                            )
                        else:
                            ab_producer, ab_consumer, acc_producer, acc_consumer = self._run_one_gemm(
                                self.atb,
                                atb_mma,
                                atoms[_B.SH_WA_D_UP],
                                tensors[_B.SH_WA_D_UP],
                                atoms[_B.SH_WB_X],
                                tensors[_B.SH_WB_X],
                                atoms[_B.SH_WA_D_UP],
                                tensors[_B.SH_WA_D_UP],
                                atoms[_B.SH_WB_X],
                                tensors[_B.SH_WB_X],
                                atoms[_B.SH_WD_UP],
                                tensors[_B.SH_WD_UP],
                                sA_atb,
                                sB_atb,
                                sD_atb,
                                atb_epi_tile,
                                acc_ptr,
                                ab_pipeline,
                                acc_pipeline,
                                ab_producer,
                                ab_consumer,
                                acc_producer,
                                acc_consumer,
                                thread,
                                warp,
                                cta_rank,
                                kind,
                                macro,
                                generation,
                                row,
                                column,
                                k_start,
                                k_end,
                                num_tokens,
                                d_y_routed_ready,
                                d_gate_up_ready,
                                replayed_x_routed_ready,
                                d_hidden_ready,
                                d_x_routed_ready,
                                replayed_gate_up_ready,
                                routed_buffers_done,
                                True,
                                False,
                                d_add_atom=shared_wgrad_add[2][0],
                                d_add_tensor=shared_wgrad_add[2][1],
                                first_contribution=shared_wgrad_first,
                                num_local_tokens=num_local_tokens,
                                hidden_size=shared_hidden_size,
                                intermediate_size=intermediate_size,
                                macrobatch_size=macrobatch_size,
                                minibatch_size=minibatch_size,
                            )
                #
                # Routed GEMM tasks (e4m3 + e8m0): replay Gate / Up, dgrad, Wgrad
                #
                else:
                    valid, expert, row, column, k_start, k_end, first = self._routed_gemm_geometry(
                        kind,
                        macro,
                        generation,
                        local,
                        num_tokens,
                        tokens_per_expert,
                        s_expert_cu_end,
                        expert_stride,
                        expert_groups,
                        routed_hidden_size,
                        intermediate_size,
                        macrobatch_size,
                        minibatch_size,
                        valid_end_offset,
                    )
                    if valid == Int32(0):
                        # An empty task still retires its buffers.
                        if thread == Int32(0) and kind >= Int32(ROUTED_DOWN_DGRAD) and num_tokens > macrobatch_size:
                            _counter_arrive_release_gpu(routed_buffers_done, macro)
                    else:
                        task = [kind, macro, generation, row, column, k_start, k_end, first]
                        if kind == Int32(REPLAY_GATE):
                            if cutlass.const_expr(self.E4M3_CONTEXT):
                                ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer = self._routed_gemm(
                                    atoms,
                                    tensors,
                                    _B.A_X,
                                    _B.B_GATE,
                                    _B.D_GATE,
                                    expert,
                                    routed_ctx,
                                    task,
                                    [ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer],
                                    quant=[atoms[_B.T_GATE], tensors[_B.T_GATE], scales[_S.GATE]],
                                )
                            else:
                                ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer = self._routed_gemm(
                                    atoms,
                                    tensors,
                                    _B.A_X,
                                    _B.B_GATE,
                                    _B.D_GATE,
                                    expert,
                                    routed_ctx,
                                    task,
                                    [ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer],
                                )
                        elif kind == Int32(REPLAY_UP):
                            if cutlass.const_expr(self.E4M3_CONTEXT):
                                ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer = self._routed_gemm(
                                    atoms,
                                    tensors,
                                    _B.A_X,
                                    _B.B_UP,
                                    _B.D_UP,
                                    expert,
                                    routed_ctx,
                                    task,
                                    [ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer],
                                    quant=[atoms[_B.T_UP], tensors[_B.T_UP], scales[_S.UP]],
                                )
                            else:
                                ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer = self._routed_gemm(
                                    atoms,
                                    tensors,
                                    _B.A_X,
                                    _B.B_UP,
                                    _B.D_UP,
                                    expert,
                                    routed_ctx,
                                    task,
                                    [ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer],
                                )
                        elif kind == Int32(ROUTED_DOWN_DGRAD):
                            ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer = self._routed_gemm(
                                atoms,
                                tensors,
                                _B.A_D_Y,
                                _B.B_DOWN_T,
                                _B.D_D_HIDDEN,
                                expert,
                                routed_ctx,
                                task,
                                [ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer],
                                b_weight=True,
                            )
                        elif kind == Int32(ROUTED_GATE_UP_DGRAD):
                            ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer = self._routed_gemm(
                                atoms,
                                tensors,
                                _B.A_D_GATE,
                                _B.B_GATE_T,
                                _B.D_D_X,
                                expert,
                                routed_ctx,
                                task,
                                [ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer],
                                a2=_B.A_D_UP,
                                b2=_B.B_UP_T,
                                b_weight=True,
                            )
                        elif kind == Int32(ROUTED_WGRAD_DOWN):
                            ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer = self._routed_gemm(
                                atoms,
                                tensors,
                                _B.A_D_Y_T,
                                _B.B_HIDDEN_T,
                                _B.D_W_DOWN,
                                expert,
                                routed_ctx,
                                task,
                                [ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer],
                                d_add=_B.D_W_DOWN_ADD,
                                is_wgrad=True,
                            )
                        elif kind == Int32(ROUTED_WGRAD_GATE):
                            ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer = self._routed_gemm(
                                atoms,
                                tensors,
                                _B.A_D_GATE_T,
                                _B.B_X_T,
                                _B.D_W_GATE,
                                expert,
                                routed_ctx,
                                task,
                                [ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer],
                                d_add=_B.D_W_GATE_ADD,
                                is_wgrad=True,
                            )
                        else:
                            ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer = self._routed_gemm(
                                atoms,
                                tensors,
                                _B.A_D_UP_T,
                                _B.B_X_T,
                                _B.D_W_UP,
                                expert,
                                routed_ctx,
                                task,
                                [ab_producer, ab_consumer, sf_producer, sf_consumer, acc_producer, acc_consumer],
                                d_add=_B.D_W_UP_ADD,
                                is_wgrad=True,
                            )
                clc_pipeline.consumer_wait(clc_consumer)
                # get_current_work() ends with the scheduler's own fence.proxy.async.shared::cta (CuTe DSL 4.8,
                # insert_fence, on unless create() turns it off): every lane's read of the response is ordered
                # before the next CLC write into it, which the release below admits.  ``work`` is used after the
                # release, so that fence must stay; _drain_clc_capacity_tail issues the same fence by hand.
                work = scheduler.get_current_work()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    clc_pipeline.consumer_release(clc_consumer)
                clc_consumer.advance()
                next_logical = work.tile_idx[0] // Int32(CLUSTER_SIZE)
                next_active = Int32(work.is_valid_tile and next_logical >= comm_clusters and next_logical < true_clusters)
                # A CTA-local (SwiGLU) task followed by a cluster task (GEMM) needs a cluster-wide sync.
                if _is_cta_local(kind) != Int32(0) and work.is_valid_tile:
                    next_local = Int32(0)
                    if next_active != Int32(0):
                        next_kind, _, _, _ = self._decode_mxfp8_task(next_logical - comm_clusters, *task_table)
                        next_local = _is_cta_local(next_kind)
                    if next_local == Int32(0):
                        cute.arch.cluster_arrive()
                        cute.arch.cluster_wait()
                logical_cluster = next_logical
                active = next_active
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            drain_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 8)
            drain_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.ProducerConsumer, 8)
            drain_stage = Int32(0)
            while drain_stage < warp:
                drain_consumer.advance()
                drain_producer.advance()
                drain_stage = drain_stage + Int32(1)
            drain_scheduler = utils.ClcDynamicPersistentTileScheduler.create(
                scheduler_params, cute.arch.block_idx(), cute.arch.grid_dim(), (clc_drain_responses + warp * Int32(4)).align(min_align=16)
            )
            needs_drain = Int32(work.is_valid_tile and logical_cluster >= true_clusters)
            self._drain_clc_capacity_tail(needs_drain, drain_pipeline, drain_consumer, drain_producer, drain_scheduler, cta_rank)
            tmem.relinquish_alloc_permit()
            tmem.free(acc_ptr)


class _PersistentMxfp8BwdMegaE4m3Context(_PersistentMxfp8BwdMega):
    """The backward of a ``gate_up_context="mxfp8"`` forward: the routed Gate / Up context is
    e4m3 with its scale tiles (dequantized by the routed SwiGLU backward, rewritten by the replay through
    ``mxgemm.store_acc_mxfp8`` next to the BF16 replay ring ``gate_routed`` / ``up_routed``).  The context's
    parameters, the ``_BE4m3Context`` / ``_SE4m3Context`` positions, the routed SwiGLU backward's shared
    memory and the replay epilogue differ; the rest of the kernel is shared."""

    E4M3_CONTEXT = True
    POSITIONS = (_BE4m3Context, _SE4m3Context)

    @cute.jit
    def __call__(
        self,
        x_peer_table: cute.Tensor,  # int64 (ep_size,): every rank's x buffer
        d_y_peer_table: cute.Tensor,  # int64 (ep_size,): dY buffers
        d_x_peer_table: cute.Tensor,  # int64 (ep_size,): routed dX buffers
        router_peer_table: cute.Tensor,  # int64 (ep_size,): router weights
        d_router_peer_table: cute.Tensor,  # int64 (ep_size,): router-weight gradient partials
        schedule_peer_rank: cute.Tensor,  # int32 (capacity,)
        schedule_peer_token_idx: cute.Tensor,  # int32 (capacity,)
        num_tokens: cute.Tensor,  # int32 (1,)
        tokens_per_expert: cute.Tensor,  # int32 (E,)
        x_shared: cute.Tensor,  # bf16 (T, H)
        x_shared_ht: cute.Tensor,  # bf16 (H, T) view of x_shared
        d_y_shared: cute.Tensor,  # bf16 (T, H)
        d_y_shared_ht: cute.Tensor,  # bf16 (H, T) view of d_y_shared
        gate_shared: cute.Tensor,  # bf16 (T, I) context
        up_shared: cute.Tensor,  # bf16 (T, I) context
        hidden_shared_it: cute.Tensor,  # bf16 (I, T) view of the Hidden context
        w_shared_gate_hi: cute.Tensor,  # bf16 (H, I) view of the (I, H) weight
        w_shared_up_hi: cute.Tensor,  # bf16 (H, I) view of the (I, H) weight
        w_shared_down_ih: cute.Tensor,  # bf16 (I, H) view of the (H, I) weight
        d_hidden_shared: cute.Tensor,  # bf16 (T, I) out
        d_gate_shared: cute.Tensor,  # bf16 (T, I) out
        d_up_shared: cute.Tensor,  # bf16 (T, I) out
        d_gate_shared_it: cute.Tensor,  # bf16 (I, T) view of d_gate_shared
        d_up_shared_it: cute.Tensor,  # bf16 (I, T) view of d_up_shared
        d_x_shared: cute.Tensor,  # bf16 (T, H) out
        d_w_shared_gate: cute.Tensor,  # bf16 (I, H) out
        d_w_shared_up: cute.Tensor,  # bf16 (I, H) out
        d_w_shared_down: cute.Tensor,  # bf16 (H, I) out
        w_routed_gate: cute.Tensor,  # e4m3 (I, H, E) view of the (E, I, H) Gate weights
        w_routed_gate_sc: cute.Tensor,  # e8m0 scales of the Gate weights
        w_routed_up: cute.Tensor,  # e4m3 (I, H, E) view of the (E, I, H) Up weights
        w_routed_up_sc: cute.Tensor,  # e8m0 scales of the Up weights
        w_routed_gate_t: cute.Tensor,  # e4m3 (H, I, E) view of the (E, H, I) transposed Gate weights
        w_routed_gate_t_sc: cute.Tensor,  # e8m0 scales of the transposed Gate weights
        w_routed_up_t: cute.Tensor,  # e4m3 (H, I, E) view of the (E, H, I) transposed Up weights
        w_routed_up_t_sc: cute.Tensor,  # e8m0 scales of the transposed Up weights
        w_routed_down_t: cute.Tensor,  # e4m3 (I, H, E) view of the (E, I, H) transposed Down weights
        w_routed_down_t_sc: cute.Tensor,  # e8m0 scales of the transposed Down weights
        x_fp8_routed: cute.Tensor,  # e4m3 (M, H) replay dispatch out
        x_sc_routed: cute.Tensor,  # e8m0 scales of x_fp8_routed
        x_fp8_t_routed: cute.Tensor,  # e4m3 (H, M) context; a replayed macrobatch's stream rewrites it in place
        x_sc_t_routed: cute.Tensor,  # e8m0 scales of x_fp8_t_routed
        gate_routed: cute.Tensor,  # bf16 (M, I) replay Gate
        up_routed: cute.Tensor,  # bf16 (M, I) replay Up
        gate_fp8_routed: cute.Tensor,  # e4m3 (M, I) context
        gate_sc_routed: cute.Tensor,  # e8m0 scales of gate_fp8_routed
        up_fp8_routed: cute.Tensor,  # e4m3 (M, I) context
        up_sc_routed: cute.Tensor,  # e8m0 scales of up_fp8_routed
        hidden_fp8_routed: cute.Tensor,  # e4m3 (M, I) replay SwiGLU out
        hidden_sc_routed: cute.Tensor,  # e8m0 scales of hidden_fp8_routed
        hidden_fp8_t_routed: cute.Tensor,  # e4m3 (I, M) context
        hidden_sc_t_routed: cute.Tensor,  # e8m0 scales of hidden_fp8_t_routed
        d_y_fp8_routed: cute.Tensor,  # e4m3 (M, H) reverse-combine out
        d_y_sc_routed: cute.Tensor,  # e8m0 scales of d_y_fp8_routed
        d_y_fp8_t_routed: cute.Tensor,  # e4m3 (H, M)
        d_y_sc_t_routed: cute.Tensor,  # e8m0 scales of d_y_fp8_t_routed
        d_hidden_routed: cute.Tensor,  # bf16 (M, I) out
        d_x_routed: cute.Tensor,  # bf16 (M, H) out
        d_gate_fp8_routed: cute.Tensor,  # e4m3 (M, I) out
        d_gate_sc_routed: cute.Tensor,  # e8m0 scales of d_gate_fp8_routed
        d_gate_fp8_t_routed: cute.Tensor,  # e4m3 (I, M)
        d_gate_sc_t_routed: cute.Tensor,  # e8m0 scales of d_gate_fp8_t_routed
        d_up_fp8_routed: cute.Tensor,  # e4m3 (M, I) out
        d_up_sc_routed: cute.Tensor,  # e8m0 scales of d_up_fp8_routed
        d_up_fp8_t_routed: cute.Tensor,  # e4m3 (I, M)
        d_up_sc_t_routed: cute.Tensor,  # e8m0 scales of d_up_fp8_t_routed
        d_w_routed_gate: cute.Tensor,  # bf16 (I, H, E) view of the (E, I, H) out
        d_w_routed_up: cute.Tensor,  # bf16 (I, H, E) view of the (E, I, H) out
        d_w_routed_down: cute.Tensor,  # bf16 (H, I, E) view of the (E, H, I) out
        router_weights: cute.Tensor,  # f32 (M,): the macrobatch's router weights
        d_router_weight_partials: cute.Tensor,  # f32 (M, I / 128)
        router_weights_ready: cute.Tensor,  # int32 counter (macrobatches,)
        d_y_routed_ready: cute.Tensor,  # int32 counter (minibatches + macrobatches,): + the stream's head words
        d_hidden_ready: cute.Tensor,  # int32 counter (row blocks x column blocks,)
        d_gate_up_ready: cute.Tensor,  # int32 counter (row blocks,)
        d_x_routed_ready: cute.Tensor,  # int32 counter (minibatches,)
        replayed_x_routed_ready: cute.Tensor,  # int32 counter (minibatches,)
        replayed_gate_up_ready: cute.Tensor,  # int32 counter (routed row blocks x column blocks,)
        replayed_hidden_ready: cute.Tensor,  # int32 counter (routed row blocks,)
        routed_buffers_done: cute.Tensor,  # int32 counter (macrobatches,)
        tensormaps: cute.Tensor,  # uint8 (ROUTED_TENSORMAPS x 128,): the routed TMA descriptors (_write_tensormaps)
        schedule_capacity: Int32,
        num_local_tokens: Int32,  # T
        intermediate_size: Int32,  # I
        num_local_experts: Int32,  # E
        topk: Int32,
        macrobatch_size: Int32,  # M
        minibatch_size: Int32,
        num_comm_sms: Int32,
        swiglu_limit: Float32,  # clamp limit, 0 when unclamped
        situ_beta1: Float32,  # SiTU-GLU's betas, as in the base kernel
        situ_beta2: Float32,
        up_row_offset: Int32,  # a combined FC1's Up row offset, as in the base kernel
        routed_storage_tables: cute.Tensor,  # split weights' address tables and descriptor bytes, as in the base
        routed_descriptors: cute.Tensor,
        build_descriptors: Int32,  # 1: this launch builds the split descriptors, as in the base kernel
        main_grad_up_row_offset: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """The base kernel's launches and argument groups (``_PersistentMxfp8BwdMega.__call__``), with
        the e4m3 Gate / Up context: ``gate_routed`` / ``up_routed`` are the BF16 replay ring, and the
        context's e4m3 tiles and scales (``gate_fp8_routed`` ... ``up_sc_routed``) follow them."""
        self._launch(
            x_peer_table,
            d_y_peer_table,
            d_x_peer_table,
            router_peer_table,
            d_router_peer_table,
            schedule_peer_rank,
            schedule_peer_token_idx,
            num_tokens,
            tokens_per_expert,
            x_shared,
            x_shared_ht,
            d_y_shared,
            d_y_shared_ht,
            gate_shared,
            up_shared,
            hidden_shared_it,
            w_shared_gate_hi,
            w_shared_up_hi,
            w_shared_down_ih,
            d_hidden_shared,
            d_gate_shared,
            d_up_shared,
            d_gate_shared_it,
            d_up_shared_it,
            d_x_shared,
            d_w_shared_gate,
            d_w_shared_up,
            d_w_shared_down,
            w_routed_gate,
            w_routed_gate_sc,
            w_routed_up,
            w_routed_up_sc,
            w_routed_gate_t,
            w_routed_gate_t_sc,
            w_routed_up_t,
            w_routed_up_t_sc,
            w_routed_down_t,
            w_routed_down_t_sc,
            x_fp8_routed,
            x_sc_routed,
            x_fp8_t_routed,
            x_sc_t_routed,
            gate_routed,
            up_routed,
            hidden_fp8_routed,
            hidden_sc_routed,
            hidden_fp8_t_routed,
            hidden_sc_t_routed,
            d_y_fp8_routed,
            d_y_sc_routed,
            d_y_fp8_t_routed,
            d_y_sc_t_routed,
            d_hidden_routed,
            d_x_routed,
            d_gate_fp8_routed,
            d_gate_sc_routed,
            d_gate_fp8_t_routed,
            d_gate_sc_t_routed,
            d_up_fp8_routed,
            d_up_sc_routed,
            d_up_fp8_t_routed,
            d_up_sc_t_routed,
            d_w_routed_gate,
            d_w_routed_up,
            d_w_routed_down,
            router_weights,
            d_router_weight_partials,
            router_weights_ready,
            d_y_routed_ready,
            d_hidden_ready,
            d_gate_up_ready,
            d_x_routed_ready,
            replayed_x_routed_ready,
            replayed_gate_up_ready,
            replayed_hidden_ready,
            routed_buffers_done,
            tensormaps,
            schedule_capacity,
            num_local_tokens,
            intermediate_size,
            num_local_experts,
            topk,
            macrobatch_size,
            minibatch_size,
            num_comm_sms,
            swiglu_limit,
            situ_beta1,
            situ_beta2,
            up_row_offset,
            routed_storage_tables,
            routed_descriptors,
            build_descriptors,
            main_grad_up_row_offset,
            stream,
            e4m3_context=(gate_fp8_routed, gate_sc_routed, up_fp8_routed, up_sc_routed),
        )


def make_mega_args_mxfp8_bwd(
    *,
    workspace,
    schedule,
    context: dict,
    weights: dict,
    state,
    peer_tables: dict,
    num_local_tokens: int,
    schedule_capacity: int,
    intermediate_size: int,
    num_local_experts: int,
    topk: int,
    macrobatch_size: int,
    minibatch_size: int,
    num_comm_sms: int,
    swiglu_limit: float,
    stream,
    launch: bool = False,
    x_shared=None,
    d_y_shared=None,
    native_columnwise: bool = False,
    up_row_offset: int = 0,
    routed_storage_tables=None,
    routed_descriptors=None,
    build_descriptors: int = 1,
    main_grad_up_row_offset: int = 0,
    situ_beta=(0.0, 0.0),
) -> dict:
    """``__call__``'s arguments, in order: ``launch=False`` builds the CuTe
    tensors and DSL scalars ``cute.compile`` traces, ``launch=True`` the arguments of a call to the
    compiled TVM FFI function (this call's torch views, plain scalars and the ``CUstream``).

    ``context``: the MXFP8 forward context by name (``x_fp8_t_routed`` ... ``hidden_sc_t_routed``,
    ``gate_shared``, ``up_shared``, ``hidden_shared``; the routed Gate / Up context as ``gate_routed_bf16`` /
    ``up_routed_bf16``, or as ``gate_fp8_routed`` / ``gate_sc_routed`` / ``up_fp8_routed`` / ``up_sc_routed``
    for the e4m3-context kernel, whose ``state`` then holds the BF16 replay ring); ``weights``: the shared BF16
    weights and the routed MXFP8 tuples' parts by name; ``state``: the call's allocations and counters
    (``persistent_mxfp8_bwd``).  Weights and Wgrads become ``(N, K, E)`` / ``(M, N, E)`` views.
    ``x_shared`` / ``d_y_shared``: the shared chain's input and output gradient in the shared hidden (the latent
    MoE; the workspace x / dY buffers, then in the routed hidden, by default).  The kernel reads the shared hidden
    from ``x_shared`` and the routed hidden from ``d_x_routed``, as the BF16 backward.

    ``native_columnwise``: the three transposed routed weights arrive in TransformerEngine's native
    columnwise layout -- the parameter's own ``(E, K, N)``, not MOK's transposed ``(E, N, K)``.  The
    operand view is ``(N, K, E)`` either way; what differs is which permutation reaches it, and
    hence whether N or K is contiguous.  This is host-only and is not one of ``__call__``'s
    arguments: the kernel learns it from its compile-time variant.

    ``up_row_offset``: where Up starts when Gate and Up are one combined FC1 passed whole.

    ``d_y_shared``: the shared expert chain's own output gradient, ``dY * G`` under a shared output
    gate; None reads ``dY`` from the workspace as the routed chain does.
    """
    import torch

    def dynamic_tensor(tensor):
        return tensor if launch else dynamic_layout_tensor(tensor)

    scalar_i, scalar_f = (int, float) if launch else (Int32, Float32)
    cuda_stream = stream if isinstance(stream, cuda.CUstream) else cuda.CUstream(stream.cuda_stream)
    device = workspace.device
    x = row_major_view(workspace.x_buffer if x_shared is None else x_shared)
    d_y = row_major_view(workspace.d_y_buffer if d_y_shared is None else d_y_shared)
    ctx = {name: row_major_view(tensor) for name, tensor in context.items()}
    w = {name: row_major_view(tensor) for name, tensor in weights.items()}

    def transposed(weight):
        """A transposed routed weight as its ``(N, K, E)`` operand view, or ``(N, K)`` split.

        MOK's quantizer gives ``(E, N, K)`` and the operand is K-major; TransformerEngine's native
        columnwise payload is the parameter's ``(E, K, N)`` and the same operand is MN-major.  Both
        describe the same matrix, so only the permutation differs.  Split allocations are one
        expert's own matrix and have no E mode to move, so the rank picks the permutation -- as
        ``routed_weight_view`` does for the rowwise payloads.
        """
        if weight.ndim == 2:
            return weight.permute(1, 0) if native_columnwise else weight
        return weight.permute(2, 1, 0) if native_columnwise else weight.permute(1, 2, 0)

    def wgrad_view(gradient):
        return gradient if gradient.ndim == 2 else gradient.permute(1, 2, 0)

    args = dict(
        x_peer_table=peer_address_table(workspace.x_buffer_ptrs, device, peer_tables),
        d_y_peer_table=peer_address_table(workspace.d_y_buffer_ptrs, device, peer_tables),
        d_x_peer_table=peer_address_table(workspace.d_x_routed_buffer_ptrs, device, peer_tables),
        router_peer_table=peer_address_table(workspace.router_weight_buffer_ptrs, device, peer_tables),
        d_router_peer_table=peer_address_table(workspace.d_router_weight_buffer_ptrs, device, peer_tables),
        schedule_peer_rank=row_major_view(schedule.peer_rank),
        schedule_peer_token_idx=row_major_view(schedule.peer_token_idx),
        num_tokens=row_major_view(schedule.num_tokens),
        tokens_per_expert=row_major_view(schedule.tokens_per_expert),
        x_shared=x,
        x_shared_ht=x.permute(1, 0),
        d_y_shared=d_y,
        d_y_shared_ht=d_y.permute(1, 0),
        gate_shared=ctx["gate_shared"],
        up_shared=ctx["up_shared"],
        hidden_shared_it=ctx["hidden_shared"].permute(1, 0),
        w_shared_gate_hi=w["shared_gate"].permute(1, 0),
        w_shared_up_hi=w["shared_up"].permute(1, 0),
        w_shared_down_ih=w["shared_down"].permute(1, 0),
        d_hidden_shared=state.d_hidden_shared,
        d_gate_shared=state.d_gate_shared,
        d_up_shared=state.d_up_shared,
        d_gate_shared_it=state.d_gate_shared.permute(1, 0),
        d_up_shared_it=state.d_up_shared.permute(1, 0),
        d_x_shared=state.d_x_shared,
        d_w_shared_gate=state.d_w_shared_gate,
        d_w_shared_up=state.d_w_shared_up,
        d_w_shared_down=state.d_w_shared_down,
        w_routed_gate=routed_weight_view(w["routed_gate"]),
        w_routed_gate_sc=w["routed_gate_sc"],
        w_routed_up=routed_weight_view(w["routed_up"]),
        w_routed_up_sc=w["routed_up_sc"],
        w_routed_gate_t=transposed(w["routed_gate_t"]),
        w_routed_gate_t_sc=w["routed_gate_t_sc"],
        w_routed_up_t=transposed(w["routed_up_t"]),
        w_routed_up_t_sc=w["routed_up_t_sc"],
        w_routed_down_t=transposed(w["routed_down_t"]),
        w_routed_down_t_sc=w["routed_down_t_sc"],
        x_fp8_routed=state.x_fp8_routed,
        x_sc_routed=state.x_sc_routed,
        x_fp8_t_routed=ctx["x_fp8_t_routed"],
        x_sc_t_routed=ctx["x_sc_t_routed"],
        **(
            dict(
                gate_routed=state.gate_routed,
                up_routed=state.up_routed,
                gate_fp8_routed=ctx["gate_fp8_routed"],
                gate_sc_routed=ctx["gate_sc_routed"],
                up_fp8_routed=ctx["up_fp8_routed"],
                up_sc_routed=ctx["up_sc_routed"],
            )
            if "gate_fp8_routed" in ctx
            else dict(gate_routed=ctx["gate_routed_bf16"], up_routed=ctx["up_routed_bf16"])
        ),
        hidden_fp8_routed=state.hidden_fp8_routed,
        hidden_sc_routed=state.hidden_sc_routed,
        hidden_fp8_t_routed=ctx["hidden_fp8_t_routed"],
        hidden_sc_t_routed=ctx["hidden_sc_t_routed"],
        d_y_fp8_routed=state.d_y_fp8_routed,
        d_y_sc_routed=state.d_y_sc_routed,
        d_y_fp8_t_routed=state.d_y_fp8_t_routed,
        d_y_sc_t_routed=state.d_y_sc_t_routed,
        d_hidden_routed=state.d_hidden_routed,
        d_x_routed=state.d_x_routed,
        d_gate_fp8_routed=state.d_gate_fp8_routed,
        d_gate_sc_routed=state.d_gate_sc_routed,
        d_gate_fp8_t_routed=state.d_gate_fp8_t_routed,
        d_gate_sc_t_routed=state.d_gate_sc_t_routed,
        d_up_fp8_routed=state.d_up_fp8_routed,
        d_up_sc_routed=state.d_up_sc_routed,
        d_up_fp8_t_routed=state.d_up_fp8_t_routed,
        d_up_sc_t_routed=state.d_up_sc_t_routed,
        # (M, N, E) views of the [E, M, N] gradients; a split main_grad is one expert's (M, N).
        d_w_routed_gate=wgrad_view(state.d_w_routed_gate),
        d_w_routed_up=wgrad_view(state.d_w_routed_up),
        d_w_routed_down=wgrad_view(state.d_w_routed_down),
        router_weights=state.router_weights,
        d_router_weight_partials=state.d_router_weight_partials,
    )
    args.update(state.counters)
    args["tensormaps"] = state.tensormaps
    args = {name: dynamic_tensor(value) if isinstance(value, torch.Tensor) else value for name, value in args.items()}
    args.update(
        schedule_capacity=scalar_i(schedule_capacity),
        num_local_tokens=scalar_i(num_local_tokens),
        intermediate_size=scalar_i(intermediate_size),
        num_local_experts=scalar_i(num_local_experts),
        topk=scalar_i(topk),
        macrobatch_size=scalar_i(macrobatch_size),
        minibatch_size=scalar_i(minibatch_size),
        num_comm_sms=scalar_i(num_comm_sms),
        swiglu_limit=scalar_f(swiglu_limit),
        situ_beta1=scalar_f(situ_beta[0]),
        situ_beta2=scalar_f(situ_beta[1]),
        up_row_offset=scalar_i(up_row_offset),
        # dynamic_layout_tensor picks the unit-stride dimension itself: the experts for the
        # (operands, E) table, the only one for the descriptor bytes.
        routed_storage_tables=dynamic_tensor(_routed_tables_or_placeholder(routed_storage_tables, device)),
        routed_descriptors=dynamic_tensor(_routed_descriptors_or_placeholder(routed_descriptors, device)),
        build_descriptors=scalar_i(build_descriptors),
        main_grad_up_row_offset=scalar_i(main_grad_up_row_offset),
        stream=cuda_stream,
    )
    return args


def mega_class_bwd(gate_up_context: str = "mxfp8") -> type:
    """The backward kernel class of a forward context's Gate / Up precision (``"mxfp8"``, the default, or
    ``"bf16"``)."""
    return {"bf16": _PersistentMxfp8BwdMega, "mxfp8": _PersistentMxfp8BwdMegaE4m3Context}[gate_up_context]


def prepare_mega_mxfp8_bwd(
    *,
    is_clamped: bool,
    task_decode: TaskDecode,
    native_columnwise: bool = False,
    is_situ: bool = False,
    split_weights: bool = False,
    accumulate_wgrad: bool = False,
    bf16_main_grad: bool = False,
    split_main_grads: bool = False,
    has_shared: bool = True,
    has_tail: bool = False,
    recompute_forward_context: bool = False,
    **kwargs,
):
    """Compile one (clamp mode, task decode, Gate / Up context, columnwise layout, routed layout) variant; the context's
    precision follows the ``context`` keys.  ``recompute_forward_context``: the variant that replays macrobatch 0 (its
    routed ``context`` entries are then scratch).  The executor's positional ABI is ``__call__``'s order."""
    cute_kwargs = make_mega_args_mxfp8_bwd(launch=False, native_columnwise=native_columnwise, **kwargs)
    kernel_class = mega_class_bwd("mxfp8" if "gate_fp8_routed" in kwargs["context"] else "bf16")
    parameters = tuple(inspect.signature(kernel_class.__call__).parameters)[1:]
    if tuple(cute_kwargs) != parameters:
        raise RuntimeError("make_mega_args_mxfp8_bwd keys must match __call__ parameters in order")
    return cute.compile(
        kernel_class(
            is_clamped=is_clamped,
            task_decode=task_decode,
            native_columnwise=native_columnwise,
            split_weights=split_weights,
            accumulate_wgrad=accumulate_wgrad,
            bf16_main_grad=bf16_main_grad,
            split_main_grads=split_main_grads,
            is_situ=is_situ,
            has_shared=has_shared,
            has_tail=has_tail,
            recompute_forward_context=recompute_forward_context,
        ),
        **cute_kwargs,
        options=COMPILE_OPTIONS,
    )


__all__ = ["REPLAY_SWIGLU", "make_mega_args_mxfp8_bwd", "mega_class_bwd", "prepare_mega_mxfp8_bwd"]
