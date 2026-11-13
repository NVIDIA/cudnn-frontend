# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications are licensed under Apache-2.0. Pre-existing code retains the terms
# under which FlashInfer distributes it (the BSD-3-Clause notice below, in an
# Apache-2.0 project); see LICENSING.md and THIRD_PARTY_LICENSES.txt.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Derived from FlashInfer (https://github.com/flashinfer-ai/flashinfer), tag v0.7.2.post1,
# commit beabf5004585b03da60e927bbb803acbaa62fe36:
#   flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100.py         (Sm100BlockScaledPersistentDenseGemmKernel)
#   flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_common.py  (the shared launch and copy-partition helpers)
# Both upstream files carry the notice above. dense_blockscaled_gemm_sm100.py states that it
# was ported from TensorRT-LLM's dense_blockscaled_gemm_persistent.py (https://github.com/NVIDIA/TensorRT-LLM).
# FlashInfer is distributed under Apache-2.0. FlashInfer's NOTICE:
#   FlashInfer
#   Copyright 2025-2026 NVIDIA
#   Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
# Local changes: see the module docstring.

"""The gate / up product of an FP4 MLP for decode rows with SiLU(gate) x up and the FP4 re-quantisation fused into the epilogue: a
persistent block-scaled tcgen05 GEMM whose token rows sit on the narrow N tile.  Config and ledgers in ``swiglu_fp4_decode_config.py``,
the host entry in ``swiglu_decode/swiglu_fp4_decode.py``.

OPERATION (``T`` token rows 1 .. 64, ``K`` = HIDDEN, ``I`` = INTERMEDIATE; fp32 accumulation in tensor memory)::

    gate[t, c] = bf16(alpha * sum_k activation[t, k] * weight[row_gate(c), k])       up[t, c] likewise with row_up(c)
    v[t, c]    = bf16(silu(gate[t, c]) * up[t, c])            silu(x) = x / (1 + 2^(x * fp32(-log2 e)))
    block b = channels 16 b .. 16 b + 15 of row t:   amax = max |v|;   byte = E4M3(gs * (amax * fp32(1/6)));   value = fp32(byte)
    encode = 0 if value == 0 else rcp(value * rcp(gs));     out[t, c] = E2M1(v[t, c] * encode);     out_scale[t, b] = byte

``weight`` ``[2 I, K]`` FP4 codes (two per byte) with one E4M3 scale per 16 K elements, rows INTERLEAVED in tiles of 128: rows
``128 j .. 128 j + 63`` are gate channels ``64 j .. 64 j + 63``, rows ``128 j + 64 .. 128 j + 127`` the up channels of the same 64.
``activation`` ``[T, K]`` in the same format.  ``alpha`` and ``gs`` (the output's global scale) one fp32 each.  ``out`` ``[T, I / 2]``
bytes (low nibble = even channel), ``out_scale`` one E4M3 byte per (row, 16-channel block) in the 128 x 4 scale-atom order.
Kernel coordinates: M = the 2 I weight rows (128 per CTA tile), N = the token rows (one tile of 8 / 16 / 32 / 64 per call),
K in stages of 512 elements (8 MMA instructions of 64).  The product itself is never stored: C is a null-pointer tensor that carries
the shape and the bf16 staging dtype.

ROUNDING, in order.  (1) fp32 chain per element in ascending K, no split, no reduction across rows or blocks; (2) ``alpha *`` then
round-to-nearest-even to bf16 (gate and up), staged in shared memory and widened back exactly; (3) SiLU as ``mul`` by ``fp32(-log2 e)``,
``ex2.approx.ftz``, ``+ 1``, ``div.approx``; (4) ``* up``, the second bf16 rounding; (5) the block maximum by compare + select (a NaN
propagates) and one masked exchange between the two lanes that own a block; (6) the block-scale chain of
``tile_dsl.pointwise.nvfp4_block_scale_global`` (the 1/6 as bits, an E4M3 rounding saturating at 448, a zero rule, two approximate
reciprocals); (7) E2M1 round-to-nearest-even saturating at 6.  No fused multiply-add.  A row's bytes depend on that row's operands
only: the token tile, the row's position and every ``Cfg`` field leave them unchanged.

PIPELINE, 192 threads = 6 warps (cluster 1 x 1, one CTA per SM by shared memory):

* warp 5, TMA loader: per work tile and K stage the weight tile (2 boxes), its scales (1), the activation tile (2) and its scale atom
  (1) into one ring stage under one transaction barrier.  With ``PREWAIT_STAGES`` the weight and weight-scale loads of the first
  stages of its first work tile are issued BEFORE its dependent-launch wait.
* warp 4, MMA: per stage 16 scale copies shared -> tensor memory and 8 block-scaled MMA instructions, then one commit that frees the
  stage; after the last stage of a work tile one commit that publishes the accumulator.
* warps 0-3, epilogue: per sub-tile of up to 32 tokens: accumulator -> registers, ``* alpha``, bf16, ``stmatrix`` into a staging stage;
  a named barrier; :func:`swiglu_quant_epilogue` on the staged tile (two global stores per thread); a named barrier.

BARRIER TABLE (``S`` operand stages = 5 / 4 / 4 / 3 at token tiles 8 / 16 / 32 / 64; SUM(issuing lanes) == init on every row):

  ab_full[s], s < S     init 1    warp 5, ONE elected lane, arrive + expect_tx(stage bytes); the bytes by 6 TMA loads.  A pre-wait
                                  stage gets the same full transaction count in the pre-wait block and the main loop then issues
                                  only its activation loads: one arrive per stage per lap.            waiter: warp 4
  ab_empty[s]           init 1    warp 4, ONE elected tcgen05.commit behind the stage's MMAs.         waiter: warp 5 (next lap, drain)
  acc_full[a], a < 2    init 1    warp 4, ONE elected tcgen05.commit behind the tile's last stage.    waiters: the 128 epilogue threads
  acc_empty[a]          init 4    warps 0-3, ONE elected lane each, behind the tile's last sub-tile.  waiter: warp 4 (next tile, drain)
  named 0               192       every thread, above the role dispatch (the pipelines' init sync and one explicit sync)
  named 2               160       the MMA warp and the four epilogue warps: the tensor-memory base word is published behind it
  named 1               128       the four epilogue warps: after a sub-tile's staging stores, after its quantise, before the dealloc
  init                            warp 0, one elected lane, every stage of the four rings; a fence and an all-thread sync follow

Named barriers 1 and 2 are reached from warp-role branches by a subset of the CTA.

SMEM BUFFER TABLE (allocation order; totals 231,424 / 231,424 / 231,424 / 230,400 B = what the launch reserves):

  mbarrier rings          (2 S + 5) x 8 B   section above (one 8-byte slot is unused)
  tmem_holding_buf        4 B               written once by tcgen05.alloc, read behind named barrier 2 and for the dealloc
  sC staging ring         C x 128 x sub-tile x 2 B, C = 7 / 12 / 4 / 7.  Written by the epilogue's stmatrix, read back by the same
                          128 threads with scalar loads through the layout's swizzle.  No descriptor reads it, so the bank argument
                          stands alone: a token's 8 lanes are 128 / 256 / 512 B apart -> conflict-free at tile 8, 2-way at tile 16,
                          4-way at tiles 32 / 64 per load.  REFILLED C sub-tiles later; released by the sub-tile's second named
                          barrier; every value loaded from it reaches a global store in front of that barrier.
  sA / sB rings           S x 32,768 B / S x token tile x 256 B.  TMA writes, the MMA descriptor reads: one layout object makes both
                          maps, so the swizzle matches by construction.  Released by the MMA warp's commit behind the MMAs.
  sSFA / sSFB rings       S x 4,096 B each (+ 512 B pad below tile 64).  TMA writes, the shared -> tensor memory copy reads; no swizzle.

TENSOR MEMORY: one allocation of 512 columns by warp 0 below the wait, released with the same constant at exit; used 80 / 96 / 128 / 192
(two accumulator stages of one token tile, 32 + 32 scale columns).  An accumulator stage is released by the four ``acc_empty``
arrives; what was read from it is cast and stored to the staging ring in front of them.

DEPENDENT LAUNCH.  Both ``griddepcontrol`` instructions are always emitted; ``Cfg.PDL`` is the launch attribute.  AS A DEPENDENT:
warps 0-4 wait right behind the all-thread sync, the loader after its pre-wait stages.  Read above a wait: ``alpha`` (every thread,
prologue) and the pre-staged weight and weight-scale tiles -- so no launch that can still be running when this kernel starts may write
``weight``, ``weight_scale`` or ``alpha``.  The activation, its scales and the output's global scale are read below the reading
role's wait, and both global stores are below the epilogue's.  The tensor-memory allocation is below the wait: a waiting CTA holds
shared memory and no tensor memory.  AS A PREDECESSOR: ONE ``launch_dependents`` statement stands behind the three role branches, so a
CTA signals when its FIRST role finishes -- the loader warp, once the MMA warp has consumed the CTA's last operand stage and BEFORE
the CTA's last tile is written.  A dependent launched with the attribute must issue its own wait before it reads ``out`` or
``out_scale``.  That trigger is always below the CTA's own tensor-memory allocation (the loader cannot drain before the MMA warp ran
behind named barrier 2).

DERIVED from FlashInfer's persistent block-scaled dense GEMM (the file header names the source and its terms).  Local changes:
(1) the store of the product is replaced by the SwiGLU + block-quantising epilogue; three operands are added (the packed output, its
block scales, the output's global scale); (2) the loader issues the first weight stages ahead of the dependent-launch wait
(``PREWAIT_STAGES``), and a CTA without work still waits; (3) one config object replaces the constructor's keyword list; class, kernel
and barrier names; the cuDNN symbol prefix with the form at the head of the symbol; (4) the epilogue is written on
``cudnn.frost.tile_dsl`` (``silu_ex2_div_approx``, ``nvfp4_block_scale_global``, ``fp32_to_e2m1x2``, ``st_global``,
``sf_atom_offset``); (5) C is bound to a null pointer in the host function: no buffer is allocated for a product that is never
stored; (6) every operand extent is a run-time value, so one compiled form serves every row count its token tile holds, launched on
an explicit stream; (7) the base class's launch and copy-partition helpers are methods of this class; (8) the stage rule's capacity
is a constant checked against the DSL, and the traced layout is checked against the config's mirror; (9) DSL names that moved
between releases are bound through ``cudnn._cutlass_compat``.
"""

from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import PipelineTmaUmma, PipelineUmmaAsync

from cudnn._cutlass_compat import LayoutEnum, SmemAllocator, get_smem_capacity_in_bytes
from cudnn.frost.buffers import cutedsl_state
from cudnn.frost.tile_dsl import pointwise, sf_layout, tma

from .swiglu_fp4_decode_config import (
    CTA_SYNC_BARRIER,
    EPILOG_SYNC_BARRIER,
    EPILOGUE_WARPS,
    FLAT_ALIGN_BYTES,
    K_TILES,
    MMA_WARP,
    NAME,
    PACKED_ALIGN_BYTES,
    SCALAR_ALIGN_BYTES,
    SF_VEC,
    STAGE_RULE_CAPACITY_BYTES,
    THREADS_PER_CTA,
    TMA_WARP,
    TMEM_ALLOC_COLS,
    TMEM_PTR_SYNC_BARRIER,
    WEIGHT_TILE_ROWS,
    Cfg,
    epilogue_tile,
    smem_bytes,
    stages,
    tx_bytes,
    validate_cfg,
)


def _dsl_text() -> str:
    _, version = cutedsl_state()
    return " ".join(version) if version else "unknown version"


def _per_token_fragment_plan(shape, stride, token_axis):
    """Static analysis of a t2r fragment of the identity tensor over C.

    ``shape``/``stride`` are the (hierarchical, static) layout of the
    per-thread fragment; strides are ``ScaledBasis`` values ``k@axis`` or 0.
    Returns ``(offsets, distinct)``: for every fragment element (in the
    colexicographic order the fragment is indexed with) its token-axis
    offset relative to element 0, and the sorted distinct offsets. Returns
    ``None`` when the layout is not fully static, so the caller falls back
    to one load per element.
    """
    leaves = []

    def walk(sh, st):
        if isinstance(sh, (tuple, list)):
            if not isinstance(st, (tuple, list)) or len(sh) != len(st):
                return False
            return all(walk(a, b) for a, b in zip(sh, st, strict=True))
        if not isinstance(sh, int):
            return False
        mode = getattr(st, "mode", None)
        value = getattr(st, "value", st)
        if mode is None:
            if not isinstance(value, int) or value != 0:
                return sh == 1 or value == 0
            leaves.append((sh, 0))
            return True
        if not isinstance(value, int):
            return False
        leaves.append((sh, value if list(mode)[0] == token_axis else 0))
        return True

    if not walk(shape, stride):
        return None
    offsets = []
    total = 1
    for extent, _ in leaves:
        total *= extent
    for i in range(total):
        rem = i
        off = 0
        for extent, scale in leaves:
            off += (rem % extent) * scale
            rem //= extent
        offsets.append(off)
    return offsets, sorted(set(offsets))


@dsl_user_op
def _l2_cache_policy(evict: str, *, loc=None, ip=None) -> cutlass.Int64:
    """Runtime ``createpolicy`` L2 eviction policy (fraction 1.0) for TMA loads.

    ``evict`` is ``"evict_first"`` (stream-once operand) or ``"evict_last"``
    (operand re-read by every CTA). Kept a runtime instruction on purpose: a
    constant policy makes ptxas emit an invalid encoding on some toolchains.
    """
    return cutlass.Int64(
        llvm.inline_asm(
            cutlass.Int64.mlir_type,
            [],
            f"createpolicy.fractional.L2::{evict}.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def swiglu_quant_epilogue(shared, out, out_scale, global_scale, tid, buffer, channel_base, token_base, rows, inter, epi_n: cutlass.Constexpr):
    """One epilogue sub-tile: read the staged bf16 gate / up values back from shared memory, ``bf16(silu(gate) * up)``, the block
    scale of 16 channels and the 4-bit codes; one 32-bit store of eight codes per thread, one scale byte per block (even lanes)."""
    col_local = (tid % 8) * 8
    # Four epilogue warps cover 16 token rows per pass; a 32-token sub-tile needs two.
    for row_pass in cutlass.range_constexpr((epi_n + 15) // 16):
        row_local = tid // 8 + row_pass * 16
        row = token_base + row_local
        col = channel_base + col_local
        if row_local < epi_n and row < rows:
            vals = cute.make_rmem_tensor((8,), cutlass.Float32)
            amax = cutlass.Float32(0)
            for p in cutlass.range_constexpr(8):
                gate = shared[col_local + p, row_local, buffer].to(cutlass.Float32)
                up = shared[64 + col_local + p, row_local, buffer].to(cutlass.Float32)
                value = (pointwise.silu_ex2_div_approx(gate) * up).to(cutlass.BFloat16).to(cutlass.Float32)
                vals[p] = value
                amax = cutlass.max(amax, cute.math.abs(value))  # compare + select: a NaN propagates (never fmax_f32)
            # A token owns eight lanes; the exchange must not name the lanes of another token.
            token_mask = cutlass.Uint32(255) << cutlass.Uint32((tid % 32) // 8 * 8)
            amax = cutlass.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=1, mask=token_mask))
            byte, encode_scale = pointwise.nvfp4_block_scale_global(amax, global_scale[0])
            payload = cutlass.Int32(0)
            for p in cutlass.range_constexpr(4):
                payload = payload | (pointwise.fp32_to_e2m1x2(vals[2 * p] * encode_scale, vals[2 * p + 1] * encode_scale) << cutlass.Int32(p * 8))
            tma.st_global((out.iterator + (row * (inter // 2) + col // 2)).toint(), payload, cutlass.Int32)
            if tid % 2 == 0:
                block = col // 16
                out_scale[(row // 128) * ((inter // 64) * 512) + sf_layout.sf_atom_offset(row % 128, block)] = byte.to(cutlass.Uint8)


class SwigluFp4DecodeKernel:
    """The kernel object of one form (``Cfg``): the persistent block-scaled tcgen05 GEMM with the tokens on the N tile and the SwiGLU +
    block-quantising epilogue.  ``__call__`` is the host launch (called from :func:`launch`), ``frost_swiglu_fp4_decode`` the device
    kernel.  ``str(kernel)`` is the form's compact text followed by the class name: the DSL writes it into the kernel symbol right
    behind the kernel's name, so the symbol's head names the form."""

    # Largest kernel N a narrow (< 64) N tile may run at. A narrow tile is a
    # sub-tile of the 128-token SFB tile: the mainloop shifts the MMA's SFB
    # TMEM column by tok_off // 32, and an odd shift makes tcgen05.mma fault,
    # so every sub-tile must stay within the first 32 tokens. See can_implement.
    NARROW_TILE_MAX_N = 32

    @classmethod
    def narrow_tile_ok(cls, tile_n: int, kernel_n: int) -> bool:
        """Whether an MMA N tile of width ``tile_n`` may run at ``kernel_n``.

        Tiles of 64 or wider are unrestricted; narrow tiles must stay within
        ``NARROW_TILE_MAX_N`` kernel-N columns (kernel N is the token count
        under swap_ab). Shared by can_implement and by the mm_mxfp8 / mm_fp4
        runners, which re-check a tactic before launching it.
        """
        return tile_n >= 64 or kernel_n <= cls.NARROW_TILE_MAX_N

    def __init__(self, cfg: Cfg):
        validate_cfg(cfg)
        self.cfg = cfg
        self.epilogue_mode = "quant"
        # Weight (A/SFA) stages the TMA warp loads before the PDL wait (0: none).
        self.prewait_stages = cfg.PREWAIT_STAGES
        self.per_token_alpha = None
        # MMA K instructions per pipeline stage: 8 (K tile 512, a 256 B TMA row per operand row).
        self.mma_inst_tile_k = K_TILES
        self.a_l2_evict_last = False
        self.a_l2_evict_first = cfg.L2_EVICT_FIRST
        self.b_l2_evict_last = False
        self.b_l2_evict_first = False
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = SF_VEC
        mma_tiler_mn = (WEIGHT_TILE_ROWS, cfg.TOKEN_TILE)
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = (1, 1)
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_prefetch = False
        self.enable_pdl = cfg.PDL
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = EPILOGUE_WARPS
        self.mma_warp_id = MMA_WARP
        self.tma_warp_id = TMA_WARP
        self.threads_per_cta = THREADS_PER_CTA
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_bar_id = CTA_SYNC_BARRIER
        self.epilog_sync_bar_id = EPILOG_SYNC_BARRIER
        self.tmem_ptr_sync_bar_id = TMEM_PTR_SYNC_BARRIER
        # The stage rule's capacity is ONE constant on every part (the config's docstring): the DSL's SM100 figure.  A DSL that
        # reports another figure would lay another kernel out, so it is refused by name instead of traced.
        self.smem_capacity = STAGE_RULE_CAPACITY_BYTES
        if get_smem_capacity_in_bytes("sm_100") != STAGE_RULE_CAPACITY_BYTES:
            raise RuntimeError(
                f"{NAME} lays its shared memory out for a per-block capacity of {STAGE_RULE_CAPACITY_BYTES} B; the installed CuTe DSL "
                f"({_dsl_text()}) reports {get_smem_capacity_in_bytes('sm_100')} B for sm_100"
            )
        self.num_tmem_alloc_cols = TMEM_ALLOC_COLS

    def __str__(self) -> str:
        return f"{self.cfg}_{type(self).__name__}"

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # Compute mma instruction shapes
        mma_inst_bits_k = 256
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_bits_k // self.a_dtype.width,
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk_sfb = (
            self.mma_inst_shape_mnk[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mnk[1], 128),
            self.mma_inst_shape_mnk[2],
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_tile_k = self.mma_inst_tile_k
        self.mma_tiler = (
            self.mma_inst_shape_mnk[0],
            self.mma_inst_shape_mnk[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mnk_sfb[0],
            self.mma_inst_shape_mnk_sfb[1],
            self.mma_inst_shape_mnk_sfb[2] * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )
        # Number of CTA N tiles covered by one (128-wide) SFB tile
        self.sfb_sub_tiles_per_tile = max(1, self.cta_tile_shape_mnk_sfb[1] // self.cta_tile_shape_mnk[1])
        # The S2T copy of a sub-tile starts up to 31 token rows (16 B each) into
        # the 512 B SF block and reads the same length, so pad sSFB by one block.
        self.sfb_smem_pad_bytes = 512 if self.cta_tile_shape_mnk[1] < 64 else 0

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        if self.cta_tile_shape_mnk[0] != WEIGHT_TILE_ROWS or self.use_2cta_instrs or cute.size(self.epi_tile[0]) != WEIGHT_TILE_ROWS:
            raise RuntimeError(
                f"{NAME} runs {WEIGHT_TILE_ROWS} weight rows per thread block on the one-block MMA and reads whole weight tiles in its "
                f"epilogue; the installed CuTe DSL ({_dsl_text()}) lays out a CTA tile of {self.cta_tile_shape_mnk[0]} rows and an "
                f"epilogue tile of {cute.size(self.epi_tile[0])} rows"
            )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.a_major_mode,
            self.b_dtype,
            self.b_major_mode,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            1024 + self.sfb_smem_pad_bytes,
        )

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        self.overlapping_accum = self.num_acc_stage == 1
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage if not self.overlapping_accum else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )

        # Only when overlapping_accum is enabled, we need to release accumulator buffer early in epilogue
        self.iter_acc_early_release_in_epilogue = self.num_sf_tmem_cols // self.epi_tile_n

        # TODO: [alel] Currently set prefetch dist to num_ab_stage, we may have more options for prefetch dist auto tuning
        self.prefetch_dist = self.num_ab_stage

    def _prepare_tma(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
    ):
        """Build the shared block-scaled MMA and TMA descriptors."""
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = LayoutEnum.from_tensor(c_tensor)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout)

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

        return (
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
        )

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        a_major_mode: cute.nvgpu.OperandMajorMode,
        b_dtype: Type[cutlass.Numeric],
        b_major_mode: cute.nvgpu.OperandMajorMode,
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
        reserved_smem_bytes: int,
    ) -> Tuple[int, int, int]:
        """Computes the optimal number of pipeline stages for operands.

        This method uses heuristics to determine the number of stages for the
        accumulator (ACC), A/B operands, and C operand based on the kernel
        configuration and available shared memory.

        Args:
            tiled_mma (cute.TiledMma): The tiled MMA object.
            mma_tiler_mnk (Tuple[int, int, int]): The shape (M, N, K) of the
                MMA tiler.
            a_dtype (Type[cutlass.Numeric]): Data type of operand A.
            a_major_mode (cute.nvgpu.OperandMajorMode): Layout of operand A.
            b_dtype (Type[cutlass.Numeric]): Data type of operand B.
            b_major_mode (cute.nvgpu.OperandMajorMode): Layout of operand B.
            epi_tile (cute.Tile): The epilogue tile shape.
            c_dtype (Type[cutlass.Numeric]): Data type of operand C.
            c_layout (LayoutEnum): Layout of operand C.
            sf_dtype (Type[cutlass.Numeric]): Data type of the scale factors.
            sf_vec_size (int): Vector size of the scale factors.
            smem_capacity (int): Total available shared memory in bytes.
            occupancy (int): Target number of CTAs per SM.
            reserved_smem_bytes (int): Per-CTA bytes reserved outside the staged operands and output.

        Returns:
            Tuple[int, int, int]: A tuple containing the number of stages for
                (ACC, A/B, C).
        """
        # ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default C stages
        num_c_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (smem_capacity // occupancy - (reserved_smem_bytes + c_bytes)) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (smem_capacity - occupancy * ab_bytes_per_stage * num_ab_stage - occupancy * (reserved_smem_bytes + c_bytes)) // (
            occupancy * c_bytes_per_stage
        )

        return num_acc_stage, num_ab_stage, num_c_stage

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        Args:
            sSF (cute.Tensor): The scale factor tensor in smem
            tSF (cute.Tensor): The scale factor tensor in tmem

        Returns:
            A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
                - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
                - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
                - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        Args:
            tidx (cutlass.Int32): The thread index in epilogue warp groups
            tAcc (cute.Tensor): The accumulator tensor to be copied and partitioned
            gC_mnl (cute.Tensor): The global tensor C
            epi_tile (cute.Tile): The epilogue tiler
            use_2cta_instrs (bool): Whether use_2cta_instrs is enabled

        Returns:
            A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
                - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
                - tTR_tAcc: The partitioned accumulator tensor
                - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        Args:
            tiled_copy_t2r (cute.TiledCopy): The tiled copy operation for tmem to register copy(t2r)
            tTR_rC (cute.Tensor): The partitioned accumulator tensor
            tidx (cutlass.Int32): The thread index in epilogue warp groups
            sC (cute.Tensor): The shared memory tensor to be copied and partitioned
            sepi (cute.Tensor):

        Returns:
            A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
                - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
                - tRS_rC: The partitioned tensor C (register source)
                - tRS_sC: The partitioned tensor C (smem destination)
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for global memory store, then use it to:
        partition shared memory (source) and global memory (destination) for TMA store version.

        Args:
            tidx (cutlass.Int32): The thread index in epilogue warp groups
            atom (Union[cute.CopyAtom, cute.TiledCopy]): The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
            gC_mnl (cute.Tensor): The global tensor C
            epi_tile (cute.Tile): The epilogue tiler
            sC (cute.Tensor): The shared memory tensor to be copied and partitioned

        Returns:
            A tuple containing (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        alpha: cute.Tensor,  # (1,) scalar alpha, or (m,)/(n,) per-token alpha
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        out: cute.Tensor,
        out_scale: cute.Tensor,
        out_global_scale: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        Args:
            a_tensor (cute.Tensor): Input tensor A
            b_tensor (cute.Tensor): Input tensor B
            sfa_tensor (cute.Tensor): Scale factor tensor A
            sfb_tensor (cute.Tensor): Scale factor tensor B
            c_tensor (cute.Tensor): Output tensor C
            max_active_clusters (cutlass.Constexpr): Maximum number of active clusters
            stream (cuda.CUstream): CUDA stream for asynchronous execution
            epilogue_op (cutlass.Constexpr): Optional elementwise lambda function to apply to the output tensor

        Raises:
            TypeError: If input data types are incompatible with the MMA instruction.
        """
        (
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
        ) = self._prepare_tma(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
        )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype,
                    cute.cosize(self.sfb_smem_layout_staged) + self.sfb_smem_pad_bytes,
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.frost_swiglu_fp4_decode(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            alpha,
            out,
            out_scale,
            out_global_scale,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.enable_pdl,
        )
        return

    # GPU device kernel
    @cute.kernel
    def frost_swiglu_fp4_decode(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
        out: cute.Tensor,
        out_scale: cute.Tensor,
        out_global_scale: cute.Tensor,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        # Keep alpha in FP32 for precision: the accumulator is in FP32 and alpha
        # may be a very small scaling factor. Converting to c_dtype (e.g., FP16)
        # before multiplication could cause overflow when acc values are large.
        # In per-token mode the epilogue folds the scale into the accumulators
        # once the output coordinate is known, so this stays neutral.
        if cutlass.const_expr(self.per_token_alpha is not None):
            alpha_value = cutlass.Float32(1.0)
        else:
            alpha_value = alpha[0].to(cutlass.Float32)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.ptr
        tmem_holding_buf = storage.tmem_holding_buf.ptr

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Tensor memory dealloc barrier init
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads)
        cute.arch.mbarrier_init_fence()

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed(aligned=True)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1)

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        k_block_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        if cutlass.const_expr(self.per_token_alpha is not None):
            # Identity tensor over C, partitioned like tCgC, so the epilogue can
            # recover each accumulator element's (m, n) coordinate.
            cC_mnl = cute.local_tile(
                cute.make_identity_tensor(mC_mnl.shape),
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
            tCcC = thr_mma.partition_C(cC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage_overlapped))
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta)

        # PDL bookend: griddepcontrol_wait is always emitted; the actual PDL
        # behavior is controlled by use_pdl= in .launch(). The griddepcontrol
        # instructions are effectively no-ops when PDL is not enabled at
        # the driver level.
        if cutlass.const_expr(self.prewait_stages > 0):
            # Only the TMA warp may read independent weights early.
            if warp_idx != self.tma_warp_id:
                griddepcontrol_wait()
        else:
            griddepcontrol_wait()

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)

            # L2 eviction policies for the operand streams (None = default).
            a_cache_policy = None
            b_cache_policy = None
            if cutlass.const_expr(self.a_l2_evict_first):
                a_cache_policy = _l2_cache_policy("evict_first")
            if cutlass.const_expr(self.a_l2_evict_last):
                a_cache_policy = _l2_cache_policy("evict_last")
            if cutlass.const_expr(self.b_l2_evict_last):
                b_cache_policy = _l2_cache_policy("evict_last")
            if cutlass.const_expr(self.b_l2_evict_first):
                b_cache_policy = _l2_cache_policy("evict_first")

            first_work_tile = cutlass.Boolean(True)
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] < 64):
                    # Several narrow N tiles share one 128-wide SFB tile; the MMA
                    # reads its sub-tile through a shifted TMEM address below.
                    slice_n = mma_tile_coord_mnl[1] // self.sfb_sub_tiles_per_tile
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                if cutlass.const_expr(self.use_prefetch):
                    # Prefetch both A and B (default behavior)
                    for pf_k_block in cutlass.range(0, min(self.prefetch_dist, k_block_cnt), unroll=1):
                        cute.prefetch(
                            tma_atom_a,
                            tAgA_slice[(None, pf_k_block)],
                        )
                        cute.prefetch(
                            tma_atom_b,
                            tBgB_slice[(None, pf_k_block)],
                        )
                        cute.prefetch(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, pf_k_block)],
                        )
                        cute.prefetch(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, pf_k_block)],
                        )

                if cutlass.const_expr(self.prewait_stages > 0):
                    # No new storage: prime only initial empty stages.
                    prewait_count = min(self.prewait_stages, self.num_ab_stage, k_block_cnt)
                    if first_work_tile:
                        prewait_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
                        for prewait_block in cutlass.range(0, prewait_count, 1, unroll=1):
                            ab_pipeline.producer_acquire(prewait_state)
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, prewait_state.count)],
                                tAsA[(None, prewait_state.index)],
                                tma_bar_ptr=ab_pipeline.producer_get_barrier(prewait_state),
                                mcast_mask=a_full_mcast_mask,
                                cache_policy=a_cache_policy,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, prewait_state.count)],
                                tAsSFA[(None, prewait_state.index)],
                                tma_bar_ptr=ab_pipeline.producer_get_barrier(prewait_state),
                                mcast_mask=sfa_full_mcast_mask,
                                cache_policy=a_cache_policy,
                            )
                            prewait_state.advance()
                        griddepcontrol_wait()

                # Peek (try_wait) AB buffer empty for k_block = prefetch_k_block_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_block_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                #
                # Tma load loop
                #
                for k_block in cutlass.range(0, k_block_cnt, 1, unroll=1):
                    weight_ops = cutlass.Boolean(True)
                    if cutlass.const_expr(self.prewait_stages > 0):
                        # Primed stages already own their full-byte barrier.
                        weight_ops = not (first_work_tile and ab_producer_state.count < prewait_count)
                    if weight_ops:
                        # Conditionally wait for AB buffer empty
                        ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, ab_producer_state.count)],
                            tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=a_full_mcast_mask,
                            cache_policy=a_cache_policy,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, ab_producer_state.count)],
                            tAsSFA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=sfa_full_mcast_mask,
                            cache_policy=a_cache_policy,
                        )

                    # Activations remain strictly after the PDL wait.
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                        cache_policy=b_cache_policy,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                        cache_policy=b_cache_policy,
                    )

                    # Prefetch: Rolling prefetch for next tiles
                    if cutlass.const_expr(self.use_prefetch):
                        if k_block < k_block_cnt - self.prefetch_dist:
                            future_k_block = ab_producer_state.count + self.prefetch_dist
                            # Prefetch both A and B (default behavior)
                            cute.prefetch(
                                tma_atom_a,
                                tAgA_slice[(None, future_k_block)],
                            )
                            cute.prefetch(
                                tma_atom_b,
                                tBgB_slice[(None, future_k_block)],
                            )
                            cute.prefetch(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, future_k_block)],
                            )
                            cute.prefetch(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, future_k_block)],
                            )

                    # Peek (try_wait) AB buffer empty for k_block = prefetch_k_block_cnt + k_block + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_block_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                #
                # Advance to next tile
                #
                first_work_tile = cutlass.Boolean(False)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            if cutlass.const_expr(self.prewait_stages > 0):
                if first_work_tile:
                    griddepcontrol_wait()
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            # Make accumulator tmem tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            #
            # Partition for S2T copy of SFA/SFB
            #
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Peek (try_wait) AB buffer full for k_block = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_block_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB_mma = tCtSFB
                tCsSFB_s2t_tile = tCsSFB_compact_s2t
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the TMEM start address for cta_tile_shape_n=192 case by two words (ignores first 64 columns of SFB)
                    offset = cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] < 64):
                    # Sub-tile of the 128-wide SFB tile. SF for token 32*c + l
                    # lives in TMEM column c, lane l: shift the MMA's SFB address
                    # by one column per 32 tokens and start the S2T copy's smem
                    # source at the remaining token row (16 B per row of the
                    # 32x4 SF block) so the sub-tile's first token lands in lane 0.
                    # can_implement limits narrow tiles to a kernel-N extent of
                    # 32, so the column shift below is always 0 in practice: an
                    # odd column shift makes tcgen05.mma fault (misaligned
                    # address), the 64-wide tile above shifts by two columns.
                    tok_off = (mma_tile_coord_mnl[1] % self.sfb_sub_tiles_per_tile) * self.cta_tile_shape_mnk[1]
                    offset = cutlass.Int32(tok_off // 32)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                    sSFB_shifted = cute.make_tensor(
                        sSFB.iterator + cutlass.Int32(tok_off % 32) * 16,
                        sSFB.layout,
                    )
                    _, tCsSFB_s2t_tile, _ = self.mainloop_s2t_copy_and_partition(sSFB_shifted, tCtSFB)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for _k_block in range(k_block_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_s2t_tile[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kphases = cute.size(tCrA, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            kphase_coord = (
                                None,
                                None,
                                kphase_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kphase_coord = (None, None, kphase_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kphase_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kphase_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kphase_coord],
                                tCrB[kphase_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kphase
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_block = k_block + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_block_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)
        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs)

            if cutlass.const_expr(self.per_token_alpha is not None):
                # Partitioned like tCgC, so this mirrors the t2r register
                # fragment element for element.
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
                tTR_cC_partitioned = tiled_copy_t2r.get_slice(epi_tidx).partition_D(cute.flat_divide(tCcC[((None, None), 0, 0, None, None, None)], epi_tile))
                alpha_extent = mC_mnl.shape[0] if cutlass.const_expr(self.per_token_alpha == "m") else mC_mnl.shape[1]
                token_axis = 0 if self.per_token_alpha == "m" else 1
                # Static plan: which fragment elements share a token, and
                # whether the token is constant across the epilogue subtiles.
                _frag_layout = tTR_cC_partitioned.layout
                _frag_shape = _frag_layout.shape
                _frag_stride = _frag_layout.stride
                alpha_plan = _per_token_fragment_plan(tuple(_frag_shape[0:3]), tuple(_frag_stride[0:3]), token_axis)
                _subtile_plan = _per_token_fragment_plan(tuple(_frag_shape[3:5]), tuple(_frag_stride[3:5]), token_axis)
                alpha_token_per_tile = alpha_plan is not None and len(alpha_plan[1]) == 1 and _subtile_plan is not None and _subtile_plan[1] == [0]

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rC, epi_tidx, sC)
            tma_atom_c, bSG_sC, bSG_gC_partitioned = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c, tCgC, epi_tile, sC)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                if cutlass.const_expr(self.per_token_alpha is not None):
                    # (T2R, T2R_M, T2R_N, (EPI_M, EPI_N))
                    tTR_cC = tTR_cC_partitioned[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            *mma_tile_coord_mnl,
                        )
                    ]
                    tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))
                    if cutlass.const_expr(alpha_token_per_tile):
                        # Every element this thread holds in this tile has
                        # the same token: one clamped load per tile, applied
                        # through the scalar alpha multiply below.
                        tile_token = tTR_cC[(0, 0, 0, 0)][token_axis]
                        tile_token = cutlass.min(tile_token, alpha_extent - 1)
                        alpha_value = alpha[tile_token].to(cutlass.Float32)

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                else:
                    acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # Store accumulator to global memory in sub-tiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt

                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = self.cta_tile_shape_mnk[1] // self.epi_tile_n - 1 - subtile_idx
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # Async arrive accumulator buffer empty earlier when overlapping_accum is enabled
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            # Fence for TMEM load
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()
                    #
                    # Fold the per-token scale into the accumulators
                    #
                    if cutlass.const_expr(self.per_token_alpha is not None and not alpha_token_per_tile):
                        tTR_cC_subtile = tTR_cC[(None, None, None, real_subtile_idx)]
                        if cutlass.const_expr(alpha_plan is not None):
                            # The fragment's token offsets are static: load
                            # each distinct token once per subtile (clamped so
                            # the rows past the token extent, which the TMA
                            # store drops, never read alpha out of bounds).
                            offsets, distinct = alpha_plan
                            token0 = tTR_cC_subtile[0][token_axis]
                            loaded = {}
                            for d in distinct:
                                tok = cutlass.min(token0 + d, alpha_extent - 1)
                                loaded[d] = alpha[tok].to(cutlass.Float32)
                            for i in cutlass.range_constexpr(len(offsets)):
                                tTR_rAcc[i] = tTR_rAcc[i] * loaded[offsets[i]]
                        else:
                            for i in cutlass.range_constexpr(cute.size(tTR_cC_subtile)):
                                coord = tTR_cC_subtile[i]
                                token = coord[token_axis]
                                token = cutlass.min(token, alpha_extent - 1)
                                tTR_rAcc[i] = tTR_rAcc[i] * alpha[token].to(cutlass.Float32)

                    #
                    # Convert to C type
                    #
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    # Multiply alpha in FP32 before converting to c_dtype to
                    # avoid overflow when c_dtype is FP16 and acc values are large.
                    acc_vec = epilogue_op((alpha_value * acc_vec).to(self.c_dtype))
                    tRS_rC.store(acc_vec)

                    #
                    # Store C to shared memory
                    #
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    epilog_threads = 32 * len(self.epilog_warp_id)
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                    #
                    # TMA store C to global memory
                    #
                    if cutlass.const_expr(self.epilogue_mode == "plain"):
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, real_subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                    elif cutlass.const_expr(self.epilogue_mode == "quant"):
                        # One CTA contains [gate64, up64]. The first BF16
                        # rounding was performed by the ordinary GEMM epilogue.
                        # Quantization preserves the second BF16 rounding after SiLU.
                        swiglu_quant_epilogue(
                            sC,
                            out,
                            out_scale,
                            out_global_scale,
                            tidx,
                            c_buffer,
                            mma_tile_coord_mnl[0] * 64,
                            mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1] + real_subtile_idx * self.epi_tile_n,
                            mC_mnl.shape[1],
                            mC_mnl.shape[0] // 2,
                            self.epi_tile_n,
                        )
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                #
                # Async arrive accumulator buffer empty
                #
                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Dealloc the tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            epilog_threads = 32 * len(self.epilog_warp_id)
            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads)
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs)
            #
            # Wait for C store complete
            #
            if cutlass.const_expr(self.epilogue_mode == "plain"):
                c_pipeline.producer_tail()

        griddepcontrol_launch_dependents()

    frost_swiglu_fp4_decode.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Computes the grid size for the kernel launch.

        This method uses a persistent tile scheduler to determine the grid
        dimensions based on the output tensor shape and hardware constraints.

        Args:
            c (cute.Tensor): The output tensor C.
            cta_tile_shape_mnk (Tuple[int, int, int]): The shape (M, N, K) of the
                CTA tile.
            cluster_shape_mn (Tuple[int, int]): The shape of a CTA cluster.
            max_active_clusters (cutlass.Constexpr): The maximum number of
                active clusters supported by the hardware.

        Returns:
            A tuple containing the tile scheduler parameters and the computed
            grid shape.
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnl)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """Checks if the data types and scale factor vector size are a valid combination.

        Args:
            ab_dtype (Type[cutlass.Numeric]): Data type of operands A and B.
            sf_dtype (Type[cutlass.Numeric]): Data type of the scale factors.
            sf_vec_size (int): Vector size of the scale factors.
            c_dtype (Type[cutlass.Numeric]): Data type of the output tensor C.

        Returns:
            bool: True if the combination is valid, False otherwise.
        """
        is_valid = True

        # Check valid ab_dtype
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Checks if the tensor layouts are valid for the given data types.

        Args:
            ab_dtype (Type[cutlass.Numeric]): Data type of operands A and B.
            c_dtype (Type[cutlass.Numeric]): Data type of the output tensor C.
            a_major (str): The major layout of tensor A ('k' or 'm').
            b_major (str): The major layout of tensor B ('k' or 'n').
            c_major (str): The major layout of tensor C ('n' or 'm').

        Returns:
            bool: True if the layouts are valid, False otherwise.
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Checks if the MMA tiler and cluster shape are a valid combination.

        Args:
            mma_tiler_mn (Tuple[int, int]): The (M, N) shape of the MMA tiler.
            cluster_shape_mn (Tuple[int, int]): The (M, N) shape of the CTA cluster.

        Returns:
            bool: True if the combination is valid, False otherwise.
        """
        is_valid = True
        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [8, 16, 32, 64, 128, 192, 256]:
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        _is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not _is_power_of_2(cluster_shape_mn[0])
            or not _is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Checks if the tensor dimensions are valid for memory alignment.

        Args:
            m (cutlass.Int64): The M dimension of the GEMM problem.
            n (cutlass.Int64): The N dimension of the GEMM problem.
            k (cutlass.Int64): The K dimension of the GEMM problem.
            l (cutlass.Int64): The batch dimension (L) of the GEMM problem.
            ab_dtype (Type[cutlass.Numeric]): Data type of operands A and B.
            c_dtype (Type[cutlass.Numeric]): Data type of the output tensor C.
            a_major (str): The major layout of tensor A ('k' or 'm').
            b_major (str): The major layout of tensor B ('k' or 'n').
            c_major (str): The major layout of tensor C ('n' or 'm').

        Returns:
            bool: True if the tensor alignment is valid, False otherwise.
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @classmethod
    def can_implement(
        cls,
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Checks if the kernel can implement the given GEMM problem.

        This method aggregates checks for data types, layouts, tile shapes,
        and alignment to determine if the configuration is supported.

        Args:
            ab_dtype (Type[cutlass.Numeric]): Data type of operands A and B.
            sf_dtype (Type[cutlass.Numeric]): Data type of the scale factors.
            sf_vec_size (int): Vector size of the scale factors.
            c_dtype (Type[cutlass.Numeric]): Data type of the output tensor C.
            mma_tiler_mn (Tuple[int, int]): The (M, N) shape of the MMA tiler.
            cluster_shape_mn (Tuple[int, int]): The (M, N) shape of the CTA
                cluster.
            m (cutlass.Int64): The M dimension of the GEMM problem.
            n (cutlass.Int64): The N dimension of the GEMM problem.
            k (cutlass.Int64): The K dimension of the GEMM problem.
            l (cutlass.Int64): The batch dimension (L) of the GEMM problem.
            a_major (str): The major layout of tensor A ('k' or 'm').
            b_major (str): The major layout of tensor B ('k' or 'n').
            c_major (str): The major layout of tensor C ('n' or 'm').

        Returns:
            bool: True if the configuration is supported, False otherwise.
        """
        can_implement = True
        # Skip unsupported types
        if not cls.is_valid_dtypes_and_scale_factor_vec_size(ab_dtype, sf_dtype, sf_vec_size, c_dtype):
            can_implement = False
        # Skip unsupported layouts
        if not cls.is_valid_layouts(ab_dtype, c_dtype, a_major, b_major, c_major):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not cls.is_valid_mma_tiler_and_cluster_shape(mma_tiler_mn, cluster_shape_mn):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not cls.is_valid_tensor_alignment(m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major):
            can_implement = False

        # Narrow N tiles (8/16/32) read their tokens' scale factors out of the
        # 128-wide SFB tile through the sub-tile addressing in the mainloop:
        # the S2T copy's smem source moves by whole token rows inside the
        # first 32-token group and the MMA's SFB TMEM address moves by one
        # column per 32 tokens. tcgen05.mma faults with a misaligned address
        # when that column shift is odd (offsets 1 and 3 fault; the 64-wide
        # tile's two-column shift is fine), so every sub-tile has to stay in
        # the first 32-token group: the kernel-N extent is limited to 32 and
        # the SFB tile is never multicast. Wider N takes the 64-wide or
        # 128-wide tiles. narrow_tile_ok holds the 32-column bound, shared with
        # the mm_mxfp8 / mm_fp4 runners' pre-launch re-check of replayed tactics.
        if not cls.narrow_tile_ok(mma_tiler_mn[1], n) or (mma_tiler_mn[1] < 64 and cluster_shape_mn[1] > 1):
            can_implement = False
        return can_implement


@cute.jit
def launch(
    gemm: cutlass.Constexpr,
    weight,
    activation,
    weight_scale,
    activation_scale,
    alpha,
    out,
    out_scale,
    out_global_scale,
    max_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """The host function of one launch: recast the byte operands to their element types and call the kernel object."""
    n = weight.shape[0]
    k = weight.shape[1] * 2
    m = activation.shape[0]
    a = cute.make_tensor(cute.recast_ptr(weight.iterator, dtype=cutlass.Float4E2M1FN), cute.make_ordered_layout((n, k, 1), order=(1, 0, 2)))
    b = cute.make_tensor(cute.recast_ptr(activation.iterator, dtype=cutlass.Float4E2M1FN), cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)))
    # The product is never stored: C is a NULL-pointer tensor that carries the shape (2 x INTERMEDIATE, rows) and the bf16 staging
    # dtype only.  No buffer stands behind it.
    c = cute.make_tensor(cute.make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16), cute.make_ordered_layout((n, m, 1), order=(0, 1, 2)))
    sa = cute.make_tensor(cute.recast_ptr(weight_scale.iterator, dtype=cutlass.Float8E4M3FN), cute.make_layout(weight_scale.shape))
    sb = cute.make_tensor(cute.recast_ptr(activation_scale.iterator, dtype=cutlass.Float8E4M3FN), cute.make_layout(activation_scale.shape))
    gemm(a, b, sa, sb, c, alpha, max_clusters, stream, out, out_scale, out_global_scale)


def traced_figures(kernel: SwigluFp4DecodeKernel) -> dict:
    """The layout figures of a TRACED kernel object (after :func:`compile_kernel`), under the names of the config's mirror."""
    return dict(
        operand_stages=int(kernel.num_ab_stage),
        staging_stages=int(kernel.num_c_stage),
        epilogue_tile=int(kernel.epi_tile_n),
        smem_bytes=int(kernel.shared_storage.size_in_bytes()),
        tx_bytes=int(kernel.num_tma_load_bytes),
    )


def mirror_figures(cfg: Cfg) -> dict:
    """The same figures from the config's host arithmetic (what the entry validated and what its launch facts state)."""
    operand, staging = stages(cfg)
    return dict(operand_stages=operand, staging_stages=staging, epilogue_tile=epilogue_tile(cfg), smem_bytes=smem_bytes(cfg), tx_bytes=tx_bytes(cfg))


def compile_kernel(cfg: Cfg):
    """Trace-compile one form (no operands: storage-free declarations).  The caller pinned the CUDA device and ran the DSL gate.  The
    weight and activation extents and the four flat operands are symbolic, so one compiled form serves every geometry and row count
    its token tile holds.  ``--opt-level 2`` is the level the kernel's bytes were qualified with.  Refuses (``RuntimeError``, naming
    the DSL) a trace whose layout differs from the config's mirror: the host entry validated the mirror."""
    fake = cute.runtime.make_fake_compact_tensor
    sym_m, sym_n, sym_k = cute.sym_int(), cute.sym_int(), cute.sym_int()
    weight = fake(cutlass.Uint8, (sym_n, sym_k), stride_order=(1, 0), assumed_align=PACKED_ALIGN_BYTES)
    activation = fake(cutlass.Uint8, (sym_m, sym_k), stride_order=(1, 0), assumed_align=PACKED_ALIGN_BYTES)

    def flat():  # only the pointer of a flat operand is used
        return fake(cutlass.Uint8, (cute.sym_int(),), assumed_align=FLAT_ALIGN_BYTES)

    scalar = fake(cutlass.Float32, (1,), assumed_align=SCALAR_ALIGN_BYTES)
    kernel = SwigluFp4DecodeKernel(cfg)
    compiled = cute.compile(
        launch, kernel, weight, activation, flat(), flat(), scalar, flat(), flat(), scalar, cfg.MAX_CLUSTERS, make_fake_stream(), options="--opt-level 2"
    )
    traced, mirror = traced_figures(kernel), mirror_figures(cfg)
    if traced != mirror:
        raise RuntimeError(
            f"{NAME}: the installed CuTe DSL ({_dsl_text()}) lays the form {cfg} out as {traced}; the configuration this entry " f"validated is {mirror}"
        )
    return compiled


def launch_args(weight, activation, weight_scale_u8, activation_scale_u8, alpha, out, out_scale_u8, out_global_scale):
    """The DSL views of one launch's operands, in :func:`launch`'s order: host objects only, no device work."""

    def dyn(t, align):
        return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic()

    def one(t):
        return from_dlpack(t.detach().reshape(1), assumed_align=SCALAR_ALIGN_BYTES)

    return (
        dyn(weight, PACKED_ALIGN_BYTES),
        dyn(activation, PACKED_ALIGN_BYTES),
        dyn(weight_scale_u8.reshape(-1), FLAT_ALIGN_BYTES),
        dyn(activation_scale_u8.reshape(-1), FLAT_ALIGN_BYTES),
        one(alpha),
        dyn(out.reshape(-1), FLAT_ALIGN_BYTES),
        dyn(out_scale_u8.reshape(-1), FLAT_ALIGN_BYTES),
        one(out_global_scale),
    )
