# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the subchannel-scaled dSwiGLU grouped GEMM tests.

Builds two-level NVFP4 inputs (reusing the unfused subchannel-scaled test
utils) plus the BYTE-EXACT torch/DSL reference of the fused kernel's op:

    d2n = dSwiGLU(alpha^2 * subchannel_scaled_gemm(A, B), beta * C, prob)

with fused dprob (the kernel's exact reduction tree), optional per-expert
dbias column sums (the kernel's tilewise scheme), second-level output
descales sfd2 (per-(sgm x sgn) block amax of the deinterleaved dy_gate /
dy_up halves, divided by the NVFP4 two-level encode range with the DSL's
approximate '/'), and the optional NVFP4 quantization of d (the kernel's
quant_sfd_row op sequence, replayed by a standalone DSL kernel).

The op-order-sensitive pieces (sigmoid, SFD2 divide, NVFP4 quantize) are
ported verbatim from the bs_ggemm_harness reference modules
(``references/activation/dglu.py``, ``references/activation/dsl.py``,
``references/reduction/torch.py``, ``references/quantization/dsl.py``,
``references/fc2_dgrad.py``) so the comparisons stay bit-exact — do not
"simplify" them into plain torch ops.

FE ``n`` convention: ``n`` is the GEMM/weight width (the harness reference's
``f``); the n-axis outputs cover ``2n`` (the harness reference's ``n``).
"""

from typing import Optional, Tuple

import pytest
import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, math, vector, arith
from cutlass.cute.typing import Float32, Int32
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T

from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import fmin

from fe_api.test_fe_api_utils import ceil_div
from fe_api.grouped_gemm.test_grouped_gemm_unfused_subchannel_scaled_utils import (
    NVFP4_MAX_REPRESENTABLE,
    _grouped_gemm_second_level_scaled,
    _group_ranges,
    _pack_fp4x2,
    _quantize_two_level,
    _second_level_block_amax,
    _sf2_strided,
    _sf_to_mma,
    build_discrete_pointers,
)

# Thread-block sizes of the ported DSL reference kernels (each file's
# module-level _THREADS: 256 in references/activation/dsl.py, 128 in
# references/quantization/dsl.py).
_SIGMOID_THREADS = 256
_LOG2_E = 1.4426950408889634
_THREADS = 128

# Compile cache of the DSL test kernels (one entry per shape/config).
_compiled: dict = {}


def _stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


# ---------------------------------------------------------------------------
# Private port of the harness kernels/common/device/moe_kernel_helpers.py
# cvt_f32_to_e4m3_byte_and_f32 (not present in the FE shared
# cudnn.gemm.cutedsl.grouped.moe_kernel_helpers module).
# ---------------------------------------------------------------------------


def _cvt_f32_to_e4m3_byte_and_f32(fp32x1, loc=None, ip=None):
    """Scalar f32 -> e4m3 (rn, satfinite), returning (encoded byte as Int32,
    exact f32 decode). Pure PTX so it compiles on both DSL wheels — the public
    4.5 wheel rejects scalar narrow-precision `.to()` conversions
    (nvgpu.cvt_fptrunc/cvt_fpext require vector operands). Bit-identical to the
    generic `.to(Float8E4M3FN)` / `.to(Float32)` pair: same cvt.rn.satfinite
    encode, and e4m3 decodes exactly in f16."""
    src_fp32 = Float32(fp32x1).ir_value(loc=loc, ip=ip)
    asm_tmpl = (
        "{\n"
        "  .reg .b16 q;\n"
        "  .reg .b32 up;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 q, 0f00000000, $2;\n"
        "  cvt.rn.f16x2.e4m3x2 up, q;\n"
        "  cvt.u32.u16 $0, q;\n"
        "  mov.b32 $1, up;\n"
        "}"
    )
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [src_fp32],
        asm_tmpl,
        "=r,=r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    byte_i32 = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    f16x2_i32 = llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)
    vec_f16_ty = ir.Type.parse("vector<2xf16>")
    f16x2 = llvm.bitcast(vec_f16_ty, f16x2_i32, loc=loc, ip=ip)
    h0 = vector.extract(f16x2, [], [0], loc=loc, ip=ip)
    decoded = Float32(arith.extf(Float32.mlir_type, h0, loc=loc, ip=ip))
    return Int32(byte_i32), decoded


# ---------------------------------------------------------------------------
# DSL sigmoid — port of references/activation/dsl.py (the fused kernels' EXACT
# sigmoid: rcp.approx(exp2.approx(x * -log2e) + 1); the approx instructions are
# device-generation-dependent, so the reference replays them on-device).
# ---------------------------------------------------------------------------


class _Sigmoid:
    """out[i] = rcp.approx(exp2.approx(x[i] * -log2e) + 1) — the kernels' sigmoid."""

    def __init__(self, n):
        self.n = n

    @cute.kernel
    def kernel(self, x: cute.Tensor, out: cute.Tensor):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        i = bidx * _SIGMOID_THREADS + tidx
        if i < self.n:
            t = x[i] * cutlass.Float32(-_LOG2_E)
            e = cute.math.exp2(t, fastmath=True)
            out[i] = cute.arch.rcp_approx(e + cutlass.Float32(1.0))

    @cute.jit
    def __call__(self, x: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        self.kernel(x, out).launch(
            grid=((self.n + _SIGMOID_THREADS - 1) // _SIGMOID_THREADS, 1, 1),
            block=(_SIGMOID_THREADS, 1, 1),
            stream=stream,
        )


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """The fused kernels' sigmoid over any-shape CUDA tensor; returns float32 of
    the same shape. Compiled once per element count."""
    assert x.is_cuda, "dsl sigmoid runs on the GPU (same device as the kernels)"
    x32 = x.detach().to(torch.float32).contiguous().view(-1)
    out = torch.empty_like(x32)
    n = x32.numel()
    key = ("sigmoid", n)
    args = (from_dlpack(x32, assumed_align=4), from_dlpack(out, assumed_align=4), _stream())
    if key not in _compiled:
        _compiled[key] = cute.compile(_Sigmoid(n), *args, options="--generate-line-info")
    _compiled[key](*args)
    return out.view(x.shape)


# ---------------------------------------------------------------------------
# dGLU torch reference — port of references/activation/dglu.py (dswiglu path).
# ---------------------------------------------------------------------------


def _split_gate_up(c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deinterleave the 32-feature gate/up bands: (m, n) -> gate, up (m, n/2)."""
    m, n2 = c.shape
    assert n2 % 64 == 0, f"gate/up 32-feature bands need n % 64 == 0, got n={n2}"
    bands = c.reshape(m, n2 // 64, 2, 32)
    return bands[:, :, 0, :].reshape(m, -1), bands[:, :, 1, :].reshape(m, -1)


def _merge_gate_up(dy1: torch.Tensor, dy2: torch.Tensor) -> torch.Tensor:
    """Inverse of _split_gate_up: (m, n/2) x2 -> (m, n) interleaved bands."""
    m, n = dy1.shape
    return torch.stack((dy1.reshape(m, n // 32, 32), dy2.reshape(m, n // 32, 32)), dim=2).reshape(m, 2 * n)


def _seq_sum(t: torch.Tensor) -> torch.Tensor:
    """Strictly sequential f32 sum over the LAST dim (torch.sum's reduction
    order is unspecified; the kernel's is not)."""
    s = torch.zeros_like(t[..., 0])
    for i in range(t.shape[-1]):
        s = s + t[..., i]
    return s


def dprob_rowsum(x: torch.Tensor, tile_n: int = 128) -> torch.Tensor:
    """Row sum replaying the dgrad_dglu kernel's dprob reduction tree exactly:
    per 32-column chunk, sequential even-lane and odd-lane pair sums are added
    (p0 + p1); chunk totals accumulate sequentially within each tile_n n-tile;
    tile partials combine in ascending tile order (the kernel's per-tile f32
    atomics). tile_n is the mma tiler N (128 — the kernel's only supported
    tiler). Bit-exact vs the kernel for up to TWO n-tiles (two f32 adds onto
    a zeroed buffer commute); with 3+ tiles the kernel's atomic arrival order
    is nondeterministic and only tolerance-level agreement is possible.
    (m, n) -> (m, 1)."""
    m, n = x.shape
    assert n % 32 == 0, f"32-wide dprob chunks need n % 32 == 0, got n={n}"
    x = x.float()
    chunks = x.reshape(m, n // 32, 32)
    chunk_tot = _seq_sum(chunks[..., 0::2]) + _seq_sum(chunks[..., 1::2])
    # Ragged last tile is fine: sequential accumulation per tile slice.
    tile_tots = [_seq_sum(chunk_tot[:, t0 : t0 + tile_n // 32]) for t0 in range(0, n // 32, tile_n // 32)]
    return _seq_sum(torch.stack(tile_tots, dim=-1)).reshape(m, 1)


def dswiglu(
    g: torch.Tensor,  # (m, n/2) f32, upstream gradient (gemm result, alpha applied)
    c: torch.Tensor,  # (m, n) f32, forward pre-activations (RAW — beta applies inside)
    prob: torch.Tensor,  # (m,) f32, per-token multiplier
    beta: Optional[torch.Tensor] = None,  # (m,) f32, per-row C scale (applied AFTER the clamps)
    glu_alpha: Optional[float] = None,  # extra upstream-gradient scale
    glu_clamp_max: Optional[float] = None,  # gate/up clamps on the RAW C values,
    glu_clamp_min: Optional[float] = None,  # gate max-only, up both sides
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """dSwiGLU: gradients through d = up * (gate * sigmoid(gate)) * prob,
    replaying the kernel's op order: glu_alpha on the accumulator, clamps on
    the raw C values (gate upper-only, up both sides), then beta."""
    m = g.shape[0]
    x1, x2 = _split_gate_up(c.float())
    g = g.float()
    p = prob.reshape(m, 1).float().to(g.device)

    if glu_alpha is not None and float(glu_alpha) != 1.0:
        g = g * float(glu_alpha)
    if glu_clamp_max is not None and glu_clamp_min is not None:
        x1 = x1.clamp(max=float(glu_clamp_max))
        x2 = x2.clamp(max=float(glu_clamp_max)).clamp(min=float(glu_clamp_min))
    if beta is not None:
        b = beta.reshape(m, 1).float().to(g.device)
        x1 = x1 * b
        x2 = x2 * b

    sig = sigmoid(x1)
    swish = x1 * sig

    dprob = dprob_rowsum(swish * x2 * g)
    dy1 = g * p * x2 * sig * (1 + x1 * (1 - sig))
    dy2 = g * p * swish
    return _merge_gate_up(dy1, dy2), dprob, dy1, dy2


# ---------------------------------------------------------------------------
# dbias column sums — port of references/reduction/torch.py::colsum_tilewise.
# ---------------------------------------------------------------------------


def colsum_tilewise(x: torch.Tensor, acc_dtype: torch.dtype = torch.float32, tile: Tuple[int, int] = (128, 256)) -> torch.Tensor:
    """Column sums with the kernel's tiling/accumulation: (rows, cols) ->
    (cols,) in acc_dtype. Only tile[0] (rows per partial) is numeric; tile[1]
    is the DSL kernel's column blocking, with no effect here (columns are
    independent)."""
    assert x.dim() == 2, f"expected 2-D input, got {tuple(x.shape)}"
    tile_m = tile[0]
    assert tile_m % 32 == 0, f"tile_m ({tile_m}) must be a multiple of 32 (warp chunks)"
    rows, cols = x.shape
    assert rows % tile_m == 0, f"rows ({rows}) must be {tile_m}-tileable"
    x = x.float()
    # f32 in-tile sums: (tiles, chunks, 32, cols) reduced sequentially over the
    # 32 rows, then sequentially over the chunks.
    chunks = x.reshape(rows // tile_m, tile_m // 32, 32, cols)
    csum = torch.zeros(chunks.shape[0], chunks.shape[1], cols, dtype=torch.float32, device=x.device)
    for r in range(32):
        csum = csum + chunks[:, :, r]
    partials = torch.zeros(chunks.shape[0], cols, dtype=torch.float32, device=x.device)
    for c in range(csum.shape[1]):
        partials = partials + csum[:, c]
    # Accumulate tile partials in acc_dtype, ascending tile order.
    acc = torch.zeros(cols, dtype=acc_dtype, device=x.device)
    for t in range(partials.shape[0]):
        acc = acc + partials[t].to(acc_dtype)
    return acc


# ---------------------------------------------------------------------------
# Band (de)interleaving — port of references/fc2_dgrad.py lines 99-118.
# ---------------------------------------------------------------------------


def _interleave_bands(a: torch.Tensor, b: torch.Tensor, band: int) -> torch.Tensor:
    """Interleave two (m, c) tensors in `band`-sized chunks -> (m, 2c):
    a[:band], b[:band], a[band:2band], ... — the block-granularity analogue of
    dglu._merge_gate_up (which interleaves 32-feature gate/up bands of c/d)."""
    m, c = a.shape
    return torch.stack((a.reshape(m, c // band, band), b.reshape(m, c // band, band)), dim=2).reshape(m, 2 * c)


def _deinterleave_bands(x: torch.Tensor, band: int, dim: int = 1) -> torch.Tensor:
    """Undo the gate/up interleave at `band`-sized chunks along `dim`: even
    chunks (gate) first, odd chunks (up) second — the layout of the
    deinterleaved kernel's n-axis outputs (interleaved 32-col band b ->
    deinterleaved band position 32*(b//2) + (b%2)*(n/2))."""
    x = x.movedim(dim, -1)
    lead = x.shape[:-1]
    c = x.shape[-1]
    x = x.reshape(*lead, c // (2 * band), 2, band)
    x = torch.cat((x[..., 0, :], x[..., 1, :]), dim=-2).reshape(*lead, c)
    return x.movedim(-1, dim).contiguous()


# ---------------------------------------------------------------------------
# DSL NVFP4 quantize + second-level descale — port of
# references/quantization/dsl.py (rowwise NVFP4-e4m3 path only; the e5m3 scale
# branches are dropped — they need FloatNV8E5M3FNU, absent from this repo).
# ---------------------------------------------------------------------------


class _QuantNvfp4:
    """Blockwise NVFP4 quantization at global encode scale norm_const. One
    thread per two-block chunk, replicating the fused kernels'
    flashinfer-derived op sequence (hadamard_utils._nvfp4_quant_row /
    dgrad_dglu quant_sfd_row) exactly: packed f32x2 scale multiplies, satfinite
    e4m3/e2m1 converts, rcp_approx + fmin. Input bf16 or f32 (the dgrad_dglu
    kernel quantizes its raw f32 accumulator values). block is the format's
    first-level vec size (a ctor param — the kernel takes one int, like the
    fused kernels)."""

    def __init__(self, m, f, norm_const=1.0, two_level=False, rowwise=True, block=None, sf_fmt="e4m3", sf_max=448.0):
        self.m, self.f = m, f
        self.block = int(block)
        self.norm_const = float(norm_const)
        # This port keeps the e4m3 scale format only (the NVFP4 recipe).
        assert str(sf_fmt) == "e4m3", f"only e4m3 scale bytes are ported, got {sf_fmt}"
        self.sf_fmt = str(sf_fmt)
        self.sf_max = float(sf_max)
        # When set, an extra per-first-level-block f32 second-level descale
        # (sf2) is folded into the first-level scale, mirroring the dgrad_dglu
        # kernels' quant_sfd_row / colwise-RHT quant: pvscale *=
        # rcp_approx(sf2) BEFORE the encode cvt, while the data multiply stays
        # on the un-prescaled values. The sf2 map shares the sf grid layout
        # ((m, f/block) rowwise, (f, m/block) colwise).
        self.two_level = bool(two_level)
        # Rowwise: (1, 16) feature blocks, one CTA per token row. Colwise:
        # (16, 1) token blocks, one CTA per 16-token slab, a thread quantizes
        # two consecutive feature COLUMNS — identical per-block math, only the
        # addressing differs. Scales: (m, f/16) rowwise, (f, m/16) colwise.
        # Data bytes always land in the (m, f) orientation (a byte pairs two
        # adjacent features of one token), like the fused kernels store them.
        self.rowwise = bool(rowwise)

    @cute.kernel
    def kernel(self, x: cute.Tensor, q_u8: cute.Tensor, sf_u8: cute.Tensor, sf2: cute.Tensor):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        q_ptr = cute.recast_ptr(q_u8.iterator, dtype=cutlass.Float4E2M1FN)
        q = cute.make_tensor(q_ptr, cute.make_layout((self.m, self.f), stride=(self.f, 1)))
        if cutlass.const_expr(self.rowwise):
            nblk = self.f // self.block
            sf = cute.make_tensor(sf_u8.iterator, cute.make_layout((self.m, nblk), stride=(nblk, 1)))
            nchunk = self.f // (2 * self.block)
        else:
            nblk = self.m // self.block
            sf = cute.make_tensor(sf_u8.iterator, cute.make_layout((self.f, nblk), stride=(nblk, 1)))
            nchunk = self.f // 2
        norm_const = cutlass.Float32(self.norm_const)
        for it in cutlass.range((nchunk + _THREADS - 1) // _THREADS, unroll_full=True):
            cidx = it * _THREADS + tidx
            if cidx < nchunk:
                tCompute = cute.make_rmem_tensor((2 * self.block,), cutlass.Float32)
                if cutlass.const_expr(self.rowwise):
                    x_row = cute.zipped_divide(x[(bidx, None)], (2 * self.block,))
                    x_rmem = cute.make_rmem_tensor((2 * self.block,), x.element_type)
                    cute.autovec_copy(cute.slice_(x_row, ((None,), cidx)), x_rmem)
                    for i in cutlass.range_constexpr(2 * self.block):
                        tCompute[i] = x_rmem[i].to(cutlass.Float32)
                else:
                    # Block 0 = feature 2*cidx, block 1 = feature 2*cidx + 1,
                    # each over this CTA's block-many tokens.
                    for t in cutlass.range_constexpr(self.block):
                        tCompute[t] = x[(bidx * self.block + t, 2 * cidx)].to(cutlass.Float32)
                        tCompute[self.block + t] = x[(bidx * self.block + t, 2 * cidx + 1)].to(cutlass.Float32)

                num_vecs = 2
                tTR_rAcc_frg = cute.logical_divide(tCompute, cute.make_layout(self.block))
                acc_frg = tTR_rAcc_frg.load()
                abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)
                tCrSFC_pvscale = cute.make_rmem_tensor((num_vecs,), cutlass.Float32)
                for vi in cutlass.range_constexpr(num_vecs):
                    tCrSFC_pvscale[vi] = abs_acc_frg[None, vi].reduce(
                        cute.ReductionOp.MAX,
                        cutlass.Float32(0.0),
                        0,
                    )
                for vi in cutlass.range_constexpr(0, num_vecs, 2):
                    tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = cute.arch.mul_packed_f32x2(
                        (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
                        (cutlass.Float32(1 / 6.0), cutlass.Float32(1 / 6.0)),
                    )
                    tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = cute.arch.mul_packed_f32x2(
                        (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
                        (norm_const, norm_const),
                    )
                # Fold the second-level descale into the first-level scale
                # (quant_sfd_row: pvscale *= rcp_approx(sfd2_descale)), so the
                # encoded SF absorbs the second-level block magnitude exactly
                # as the fused kernels do. The data multiply below is
                # unchanged. sf2 is addressed exactly like the sf store below:
                # (row, block) rowwise, (feature, token-block) colwise.
                if cutlass.const_expr(self.two_level):
                    for vi in cutlass.range_constexpr(num_vecs):
                        if cutlass.const_expr(self.rowwise):
                            sfd2_descale = sf2[(bidx, num_vecs * cidx + vi)]
                        else:
                            sfd2_descale = sf2[(2 * cidx + vi, bidx)]
                        sfd2_scale = cute.arch.rcp_approx(sfd2_descale)
                        tCrSFC_pvscale[vi] = tCrSFC_pvscale[vi] * sfd2_scale
                # PTX e4m3 encode + exact decode in one shot — scalar f32/f8
                # `.to()` conversions are rejected by the public 4.5 wheel.
                # Same cvt.rn.satfinite.e4m3x2.f32 the generic conversion emits.
                tCrSFC_up = cute.make_rmem_tensor((num_vecs,), cutlass.Float32)
                for vi in cutlass.range_constexpr(num_vecs):
                    sf_byte_i32, sf_up = _cvt_f32_to_e4m3_byte_and_f32(tCrSFC_pvscale[vi])
                    sf_byte = cutlass.Uint8(sf_byte_i32)
                    if cutlass.const_expr(self.rowwise):
                        sf[(bidx, num_vecs * cidx + vi)] = sf_byte
                    else:
                        sf[(2 * cidx + vi, bidx)] = sf_byte
                    tCrSFC_up[vi] = sf_up

                fp32_max = cutlass.Float32(3.40282346638528859812e38)
                for vi in cutlass.range_constexpr(0, num_vecs, 2):
                    acc_scale = cute.arch.mul_packed_f32x2(
                        (
                            cute.arch.rcp_approx(tCrSFC_up[vi]),
                            cute.arch.rcp_approx(tCrSFC_up[vi + 1]),
                        ),
                        (norm_const, norm_const),
                    )
                    acc_scale_min0 = fmin(acc_scale[0], fp32_max, nan=True)
                    acc_scale_min1 = fmin(acc_scale[1], fp32_max, nan=True)
                    vec0 = tTR_rAcc_frg[None, vi]
                    vec1 = tTR_rAcc_frg[None, vi + 1]
                    for ei in cutlass.range_constexpr(self.block):
                        vec0[ei], vec1[ei] = cute.arch.mul_packed_f32x2(
                            (vec0[ei], vec1[ei]),
                            (acc_scale_min0, acc_scale_min1),
                        )

                if cutlass.const_expr(self.rowwise):
                    fp4_rmem = cute.make_rmem_tensor((2 * self.block,), cutlass.Float4E2M1FN)
                    fp4_rmem.store(tCompute.load().to(cutlass.Float4E2M1FN))
                    q_row = cute.zipped_divide(q[(bidx, None)], (2 * self.block,))
                    cute.autovec_copy(fp4_rmem, cute.slice_(q_row, ((None,), cidx)))
                else:
                    # Interleave the two feature columns so each byte packs the
                    # (2*cidx, 2*cidx + 1) feature pair of one token, then one
                    # byte store per token row.
                    tInter = cute.make_rmem_tensor((2 * self.block,), cutlass.Float32)
                    for t in cutlass.range_constexpr(self.block):
                        tInter[2 * t] = tCompute[t]
                        tInter[2 * t + 1] = tCompute[self.block + t]
                    fp4_rmem = cute.make_rmem_tensor((2 * self.block,), cutlass.Float4E2M1FN)
                    fp4_rmem.store(tInter.load().to(cutlass.Float4E2M1FN))
                    fp4_pairs = cute.logical_divide(fp4_rmem, cute.make_layout(2))
                    for t in cutlass.range_constexpr(self.block):
                        q_pair_row = cute.zipped_divide(q[(bidx * self.block + t, None)], (2,))
                        cute.autovec_copy(fp4_pairs[(None, t)], cute.slice_(q_pair_row, ((None,), cidx)))

    @cute.jit
    def __call__(self, x: cute.Tensor, q_u8: cute.Tensor, sf_u8: cute.Tensor, sf2: cute.Tensor, stream: cuda.CUstream):
        grid_x = self.m if self.rowwise else self.m // self.block
        self.kernel(x, q_u8, sf_u8, sf2).launch(grid=(grid_x, 1, 1), block=(_THREADS, 1, 1), stream=stream)


class _SecondLevelDescale:
    """Elementwise x / divisor via the DSL '/' operator — the fused dgrad_dglu
    kernel's exact SFD2 formula (thread_tile_amax / (max(d_dtype)*max(sf_dtype))).
    A dedicated kernel is needed because the DSL division is approximate
    (div.full-style), not IEEE div.rn, so a torch divide/multiply does not
    reproduce it bit-for-bit."""

    def __init__(self, n, divisor):
        self.n = n
        self.divisor = float(divisor)

    @cute.kernel
    def kernel(self, x: cute.Tensor, o: cute.Tensor):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        i = bidx * _THREADS + tidx
        if i < self.n:
            o[i] = x[i] / self.divisor

    @cute.jit
    def __call__(self, x: cute.Tensor, o: cute.Tensor, stream: cuda.CUstream):
        nblocks = (self.n + _THREADS - 1) // _THREADS
        self.kernel(x, o).launch(grid=(nblocks, 1, 1), block=(_THREADS, 1, 1), stream=stream)


def second_level_descale(x: torch.Tensor, divisor: float) -> torch.Tensor:
    """x (any shape) f32 -> x / divisor, computed with the DSL '/' so it matches
    the fused kernel's SFD2 (thread_tile_amax / (max(d_dtype)*max(sf_dtype)))
    bit-for-bit (the DSL division is approximate, not IEEE div.rn). divisor is
    the product of the two second-level target format maxima."""
    assert x.is_cuda and x.dtype == torch.float32
    flat = x.contiguous().reshape(-1)
    n = flat.numel()
    out = torch.empty_like(flat)
    key = ("sldescale", n, float(divisor))
    args = (from_dlpack(flat), from_dlpack(out), _stream())
    if key not in _compiled:
        _compiled[key] = cute.compile(_SecondLevelDescale(n, divisor), *args, options="--generate-line-info")
    _compiled[key](*args)
    torch.cuda.synchronize()
    return out.reshape(x.shape)


def _quant_nvfp4(
    x: torch.Tensor, norm_const: float, rowwise: bool, block: int, sf_fmt: str = "e4m3", sf_max: float = 448.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blockwise-NVFP4-quantize a bf16 or f32 (m, f) tensor (global encode
    scale norm_const). Blocks run along the last dim rowwise ((1, block)
    features) or along the first dim colwise ((block, 1) tokens); the packed
    data bytes come out in the (m, f) orientation either way. Returns
    (packed e2m1 uint8 (m, f/2), e4m3 scale uint8 — (m, f/block) rowwise,
    (f, m/block) colwise)."""
    assert x.is_cuda and x.dtype in (torch.bfloat16, torch.float32) and x.dim() == 2
    assert x.is_contiguous()
    m, f = x.shape
    assert f % (2 * block) == 0
    if not rowwise:
        assert m % block == 0, f"colwise ({block}, 1) token blocks need m % {block} == 0, got m={m}"
    q = torch.empty((m, f // 2), dtype=torch.uint8, device=x.device)
    sf_shape = (m, f // block) if rowwise else (f, m // block)
    sf = torch.empty(sf_shape, dtype=torch.uint8, device=x.device)
    key = ("quant", m, f, x.dtype, float(norm_const), rowwise, block, sf_fmt)
    # sf is passed twice: the second slot is the (unused) sf2 placeholder for the
    # one-level path — the kernel never reads it when two_level is False.
    args = (from_dlpack(x, assumed_align=16), from_dlpack(q, assumed_align=16), from_dlpack(sf, assumed_align=16), from_dlpack(sf, assumed_align=16), _stream())
    if key not in _compiled:
        _compiled[key] = cute.compile(
            _QuantNvfp4(m, f, norm_const, rowwise=rowwise, block=block, sf_fmt=sf_fmt, sf_max=sf_max),
            *args,
            options="--generate-line-info",
        )
    _compiled[key](*args)
    torch.cuda.synchronize()
    return q, sf


def _quant_nvfp4_two_level(
    x: torch.Tensor,
    norm_const: float,
    sf2_descale: torch.Tensor,
    block: int,
    sf_fmt: str = "e4m3",
    sf_max: float = 448.0,
    rowwise: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two-level NVFP4 quant (dgrad_dglu quant_sfd_row / dgrad_dglu_rht colwise
    quant): like _quant_nvfp4, but the first-level block scale is folded by a
    per-block second-level descale sf2_descale (f32 on the scale grid — (m,
    f/block) rowwise, (f, m/block) colwise — the stored SFD2 value broadcast to
    each block), pvscale *= rcp_approx(sf2_descale) before the encode cvt. The
    data multiply stays on the raw values, so dequant is data * SF (single
    level); the fold only reshapes how the SF magnitude is split. Returns
    (packed e2m1 uint8 (m, f/2), scale uint8 on the same grid as sf2)."""
    assert x.is_cuda and x.dtype in (torch.bfloat16, torch.float32) and x.dim() == 2
    assert x.is_contiguous()
    m, f = x.shape
    assert f % (2 * block) == 0
    if not rowwise:
        assert m % block == 0, f"colwise ({block}, 1) token blocks need m % {block} == 0, got m={m}"
    sf_shape = (m, f // block) if rowwise else (f, m // block)
    assert sf2_descale.shape == sf_shape, f"sf2_descale must be {sf_shape}, got {tuple(sf2_descale.shape)}"
    assert sf2_descale.dtype == torch.float32 and sf2_descale.is_contiguous()
    q = torch.empty((m, f // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty(sf_shape, dtype=torch.uint8, device=x.device)
    key = ("quant2", m, f, x.dtype, float(norm_const), rowwise, block, sf_fmt)
    args = (
        from_dlpack(x, assumed_align=16),
        from_dlpack(q, assumed_align=16),
        from_dlpack(sf, assumed_align=16),
        from_dlpack(sf2_descale, assumed_align=16),
        _stream(),
    )
    if key not in _compiled:
        _compiled[key] = cute.compile(
            _QuantNvfp4(m, f, norm_const, two_level=True, rowwise=rowwise, block=block, sf_fmt=sf_fmt, sf_max=sf_max),
            *args,
            options="--generate-line-info",
        )
    _compiled[key](*args)
    torch.cuda.synchronize()
    return q, sf


def quantize(x: torch.Tensor, fmt=None, sf2: Optional[torch.Tensor] = None, norm_const: float = 1.0, rowwise: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """THE quantization entry point (harness references/quantization/dsl.py,
    trimmed to the rowwise NVFP4-e4m3 case — the only one these tests use).
    Blockwise-quantize a bf16 or f32 (m, f) tensor at global encode scale
    norm_const with (1, 16) feature blocks; data bytes come out (m, f)-packed.

    sf2 (f32 on the (m, f/16) scale grid) folds a GIVEN per-block second-level
    descale into the encoded scale — the dgrad_dglu kernel's quant_sfd_row.
    Second-level GENERATION is not done here.

    Returns (packed data bytes, scale bytes)."""
    assert fmt is None, "only the NVFP4 (e2m1 data, e4m3 scales) format is ported"
    assert rowwise, "only the rowwise block orientation is ported"
    block = 16  # NVFP4 first-level vec size
    sf_fmt = "e4m3"
    sf_max = 448.0
    if sf2 is not None:
        return _quant_nvfp4_two_level(x, norm_const, sf2, block, sf_fmt, sf_max, rowwise=rowwise)
    return _quant_nvfp4(x, norm_const, rowwise, block, sf_fmt, sf_max)


# ---------------------------------------------------------------------------
# SF-atom unpack (the kernel's MMA-tiled row-scale view -> flat scales).
# ---------------------------------------------------------------------------


def _sf_atom_unpack(sf_view: torch.Tensor, valid_m: int, n2: int) -> torch.Tensor:
    """Gather the flat (valid_m, n2//16) row scale factors out of the kernel's
    MMA-tiled SF view (32, 4, ceil(valid_m/128), 4, rest, 1).

    Canonical SF-atom mapping (the create_sf_layout_tensor /
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L scatter used for SFA/SFB inputs):

        mma[m % 32, (m // 32) % 4, m // 128, ksf % 4, ksf // 4, 0] == flat[m, ksf]

    Implemented directly from that mapping with advanced indexing, so it is
    layout-independent (works on any permuted/strided view of the backing).
    Validated against _sf_to_mma by round-trip in the test file."""
    assert sf_view.dim() == 6, f"expected the 6-D MMA-tiled view, got {tuple(sf_view.shape)}"
    ksf = n2 // 16
    dev = sf_view.device
    m_idx = torch.arange(valid_m, device=dev).unsqueeze(1).expand(valid_m, ksf)
    k_idx = torch.arange(ksf, device=dev).unsqueeze(0).expand(valid_m, ksf)
    zero = torch.zeros_like(m_idx)
    return sf_view[m_idx % 32, (m_idx // 32) % 4, m_idx // 128, k_idx % 4, k_idx // 4, zero]


# ---------------------------------------------------------------------------
# Problem generator.
# ---------------------------------------------------------------------------


def make_dswiglu_subchannel_problem(
    m_per_expert: int,
    n: int,
    k: int,
    l: int,
    block2_shape: Tuple[int, int, int] = (1, 256, 256),
    with_dbias: bool = False,
    dbias_dtype: torch.dtype = torch.bfloat16,
    d_quant: bool = False,
    norm_const: float = 0.5,
    d_deinterleaved: bool = True,
    seed: int = 0,
):
    """Build FE-format inputs plus the byte-exact references for the dSwiGLU
    subchannel-scaled wrapper. FE n = GEMM/weight width (harness f); the
    n-axis outputs cover 2n. d_deinterleaved is only effective with d_quant
    (the kernel's harness-validated combination); the returned dict's
    "d_deinterleaved" key holds the effective value to pass to the wrapper.

    Reference mirrors the harness references/fc2_dgrad.py dswiglu path exactly
    (op order matters for byte-exactness)."""
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("Current torch version does not support float4_e2m1fn_x2")
    torch.manual_seed(seed)
    sgm, sgn, sgk = block2_shape
    assert m_per_expert % 256 == 0 and m_per_expert % sgm == 0
    assert k % sgk == 0 and k % 32 == 0
    assert n % 128 == 0 and sgn % 128 == 0
    deint = bool(d_deinterleaved) and bool(d_quant)
    mt = l * m_per_expert
    n2 = 2 * n
    group_sizes = [m_per_expert] * l

    a_master = torch.randn((mt, k), dtype=torch.bfloat16, device="cuda")
    b_master = torch.randn((l * n, k), dtype=torch.bfloat16, device="cuda")

    a_vals, a_sf, a_sf2 = _quantize_two_level(a_master.float(), sgm, sgk)
    # B second-level blocks must not straddle experts: quantize per expert.
    b_parts = [_quantize_two_level(b_master[e * n : (e + 1) * n].float(), sgn, sgk) for e in range(l)]
    b_vals = torch.stack([p[0] for p in b_parts])  # (l, n, k)
    b_sf = torch.stack([p[1] for p in b_parts])  # (l, n, k/16)
    b_sf2 = torch.stack([p[2] for p in b_parts])  # (l, ceil(n/sgn), k/sgk)

    # Forward FC1 pre-activations (gate/up interleaved in 32-col bands); the
    # harness generator's c_init_scale = 1 / harness_n with harness_n = 2n.
    c = torch.randn((mt, n2), dtype=torch.bfloat16, device="cuda") * (1.0 / n2)

    alpha = (0.75 + 0.25 * (torch.arange(l) % 4)).float().cuda()
    beta = (0.75 + 0.25 * ((torch.arange(l) + 2) % 4)).float().cuda()
    prob = torch.randint(-2, 2, (mt, 1, 1), dtype=torch.float32, device="cuda").float()

    # ---- FE-format tensors ----
    a_packed = _pack_fp4x2(a_vals)  # (mt, k/2)
    a_tensor = a_packed.reshape(1, mt, k // 2).permute(1, 2, 0)
    b_packed = torch.stack([_pack_fp4x2(p[0]) for p in b_parts])  # (l, n, k/2)
    b_tensor = b_packed.permute(1, 2, 0)

    sfa_tensor = _sf_to_mma(a_sf.reshape(1, mt, k // 16))
    sfb_tensor = _sf_to_mma(b_sf)

    sfa2_tensor = _sf2_strided(a_sf2.reshape(1, mt // sgm, k // sgk))
    sfb2_tensor = _sf2_strided(b_sf2)

    c_tensor = c.reshape(1, mt, n2).permute(1, 2, 0)  # (mt, 2n, 1) n-major

    padded_offsets = torch.tensor(
        [sum(group_sizes[: e + 1]) for e in range(l)],
        dtype=torch.int32,
        device="cuda",
    )
    norm_const_tensor = torch.tensor([norm_const], dtype=torch.float32, device="cuda") if d_quant else None

    # ---- reference (fc2_dgrad.py dswiglu path, in FE-n terms) ----
    a_f32 = a_vals * a_sf.repeat_interleave(16, dim=-1)
    b_f32 = b_vals * b_sf.repeat_interleave(16, dim=-1)
    g = _grouped_gemm_second_level_scaled(a_f32, b_f32, a_sf2, b_sf2, group_sizes, sgm, sgn, sgk)

    # Alpha lands on both the A and B contributions, applied as ONE multiply
    # by the f32 pre-squared alpha — exactly as the kernel does (harness
    # apply_alpha(g, alpha * alpha, group_sizes)).
    alpha2 = alpha * alpha
    for e, (r0, r1) in enumerate(_group_ranges(group_sizes)):
        g[r0:r1] *= float(alpha2[e])

    # Beta applies per row INSIDE dswiglu (after the clamps). Clamp defaults
    # match the wrapper/API defaults (glu_clamp_max=7.0, glu_clamp_min=-7.0).
    beta_row = beta.repeat_interleave(m_per_expert)
    d, dprob, dy1, dy2 = dswiglu(
        g,
        c.float(),
        prob.reshape(mt),
        beta=beta_row,
        glu_alpha=None,
        glu_clamp_max=7.0,
        glu_clamp_min=-7.0,
    )

    ref_dbias = None
    if with_dbias:
        # The kernel's exact dbias scheme: f32 per-M-tile partials,
        # dbias_dtype-matched atomics across tiles.
        ref_dbias = torch.stack([colsum_tilewise(d[r0:r1].contiguous(), acc_dtype=dbias_dtype) for r0, r1 in _group_ranges(group_sizes)])  # (l, 2n)

    # SFD2: NVFP4 max_representable in both the quant and bf16-D modes (the
    # d_quant_format is NVFP4 either way); the divide replays the kernel's
    # approximate DSL '/' (not IEEE div.rn).
    sf2_norm = NVFP4_MAX_REPRESENTABLE  # 2688.0
    ref_sfd2_gate = second_level_descale(_second_level_block_amax(dy1, sgm, sgn), sf2_norm)
    ref_sfd2_up = second_level_descale(_second_level_block_amax(dy2, sgm, sgn), sf2_norm)

    ref_d_quant = ref_d_quant_sf = None
    if d_quant:
        # The kernel quantizes the bf16-rounded dGLU output (quant_sfd_row)
        # with norm_const as the global encode scale, folding the per-block
        # second-level descale into the e4m3 row scales. Build the fold map on
        # d's interleaved (32-feature-band) layout: broadcast each
        # deinterleaved gate/up sfd2 block over its sgn columns (clamped to the
        # real n/16 width for partial blocks), then re-interleave gate/up in
        # NVFP4-block-sized bands to match d.
        d_src = d.to(torch.bfloat16)
        d_sfn = 16  # D blocks run along N
        bpb = sgn // d_sfn
        g16 = ref_sfd2_gate.repeat_interleave(bpb, dim=1)[:, : n // d_sfn]
        u16 = ref_sfd2_up.repeat_interleave(bpb, dim=1)[:, : n // d_sfn]
        sf2_d = _interleave_bands(g16, u16, 2).contiguous()  # band = 32 // d_sfn
        ref_d_quant, ref_d_quant_sf = quantize(d_src.contiguous(), sf2=sf2_d, norm_const=norm_const)

    if deint:
        # Deinterleaved layout: permute every n-axis output so gate bands come
        # first, up bands second. The sfd2 refs are already deinterleaved.
        if ref_dbias is not None:
            ref_dbias = _deinterleave_bands(ref_dbias, 32, dim=1)
        ref_d_quant = _deinterleave_bands(ref_d_quant, 16, dim=1)  # 32 e2m1 = 16 B
        ref_d_quant_sf = _deinterleave_bands(ref_d_quant_sf, 2, dim=1)  # 2 SF/band

    ref_d = None if d_quant else d.to(torch.bfloat16)

    return {
        # wrapper inputs
        "a_tensor": a_tensor,
        "sfa_tensor": sfa_tensor,
        "sfa2_tensor": sfa2_tensor,
        "c_tensor": c_tensor,
        "b_tensor": b_tensor,
        "b_packed": b_packed,  # (l, n, k/2) contiguous backing for discrete pointers
        "sfb_tensor": sfb_tensor,
        "sfb2_tensor": sfb2_tensor,
        "padded_offsets": padded_offsets,
        "alpha_tensor": alpha,
        "beta_tensor": beta,
        "prob_tensor": prob,
        "norm_const_tensor": norm_const_tensor,
        "block2_shape": block2_shape,
        "d_quant": d_quant,
        "with_dbias": with_dbias,
        "dbias_dtype": dbias_dtype,
        "d_deinterleaved": deint,
        "norm_const": norm_const,
        # geometry
        "valid_m": mt,
        "n": n,
        "k": k,
        "l": l,
        "nd": ceil_div(n, sgn),
        "sfd2_rows": ceil_div(mt, sgm),
        # references
        "ref_d": ref_d,  # bf16 (mt, 2n), bf16-D mode only
        "ref_d_quant": ref_d_quant,  # uint8 (mt, n) packed e2m1, quant mode
        "ref_d_quant_sf": ref_d_quant_sf,  # uint8 (mt, 2n/16), quant mode
        "ref_dprob": dprob,  # f32 (mt, 1)
        "ref_dbias": ref_dbias,  # (l, 2n) dbias_dtype or None
        "ref_sfd2_gate": ref_sfd2_gate,  # f32 (sfd2_rows, nd)
        "ref_sfd2_up": ref_sfd2_up,  # f32 (sfd2_rows, nd)
    }


def build_discrete_pointers_dswiglu(problem) -> dict:
    """Per-expert int64 pointer arrays into the dense backings; the problem
    dict provides the same "b_packed"/"sfb_tensor"/"sfb2_tensor"/"l" keys the
    sibling helper expects, so reuse it directly."""
    return build_discrete_pointers(problem)
