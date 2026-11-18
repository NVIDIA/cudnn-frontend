# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The attention gate + quantize DECODE kernel (bf16 in, E4M3 out, one launch): ``out = e4m3( bf16( attn / (1 + exp(-gate)) ) x (1 / scale) )``
per element; config and knobs in ``gate_quant_decode_config.py``, the host entry in ``attention_gate/frost/gate_quant_decode.py``.
Arch-portable PTX (global loads / stores, ``cvt``, ``div.rn``, ``griddepcontrol``); the arch axis is an admission fact of the host entry
(cc 10.x), not a directory.

PIPELINE, in order, identical for every thread (an element-wise map; no warp role, no election):

1. ``griddepcontrol.launch_dependents`` under ``cfg.PDL_RELEASE`` (the PDL-sites paragraph below).
2. ``idx = (block x THREADS + thread) x VEC``; a thread with ``idx >= N`` exits (only threads of the last CTA).  Signed 32-bit: the host
   entry keeps ``N`` at or below ``ELEMENT_LIMIT``.
3. Two 128-bit global loads (``VEC`` bf16 values of ``attn`` and of ``gate`` at element ``idx``) and one 32-bit load of ``scale[0]``; the
   bases are 16-byte aligned by the host check.
4. ``inv = div.rn.f32(1, scale)``: the correctly rounded fp32 reciprocal (every thread computes the same value).
5. Per element: widen bf16 -> fp32 (exact, a bit move); ``den = 1 + exp(-g)`` with ``cute.math.exp`` at its DEFAULT flags (the precise
   lowering: range reduction, ``ex2.approx.ftz.f32``, the scale by ``2^n``); ``q = div.rn.f32(a, den)``; ``v = bf16(q)`` round-to-nearest-even
   (the FIRST rounding of record), widened back exactly; ``v x inv`` (one fp32 multiply); ``cvt.rn.satfinite.e4m3x2.f32`` round-to-nearest-even
   to E4M3, saturating at +-448 (the SECOND rounding of record); a one-byte global store of the code at ``out + idx + j``.  One convert
   and one store per element, each issued as soon as its element is done.

THE STORE FORM IS PERFORMANCE-ONLY.  Packing the eight codes into one 64-bit store (four two-value converts) would write the same bytes
with fewer instructions; this body keeps one convert and one byte store per element, issued as each element completes.  A change of the
store form moves no byte: it is a performance choice, and lands only as a config field with its measured table per part, like ``THREADS``.

THE ROUNDING CHAIN IS THE CONTRACT.  The gated value is spelled as the DIVISION ``a / (1 + exp(-g))`` and rounded to bf16 before the scale:
the convention of a chain that stores the gated attention output as bf16 and quantizes it for a projection whose static per-tensor
dequantization scale is ``scale``.  Never substitute an approximate sigmoid (``tanh.approx``), a ``fastmath`` exponential, a
reciprocal-multiply for either division, or the product ``a x sigmoid(g)`` -- each is another function with other bytes on some elements.  The
test module's rendering cell pins the instructions (``div.rn.f32``, ``ex2.approx.ftz.f32``, both ``cvt``); its reference cells pin the bytes.

BATCH INVARIANCE AND REPLAY.  No reduction, no atomics, no shared state: byte ``(r, c)`` is a function of ``attn[r, c]``, ``gate[r, c]``
and ``scale`` computed by the unrolled expression of lane position ``c mod VEC`` (``width`` is a multiple of ``VEC``, so a column keeps
its lane position at every row count and row position).  ``THREADS`` only regroups threads into CTAs and ``PDL_RELEASE`` adds one
instruction above the arithmetic: neither moves a byte.

BARRIER TABLE: none.  The kernel has no mbarrier, no named barrier, no ``barrier.cta.sync``, no cluster, no ``cp.async`` group, no wait and
no loop with a runtime trip count; ``SUM(issuing lanes) == init count`` is ``0 == 0``.

SHARED-MEMORY / TENSOR-MEMORY TABLE: none.  Nothing is allocated (no shared memory, no tensor memory, no descriptor); the operands are
read from and written to global memory by the thread that owns the elements.

PDL SITES: one, the release at the kernel's entry under ``cfg.PDL_RELEASE``.  There is no ``griddepcontrol.wait`` and the launch never
carries the programmatic-stream-serialization attribute, so this kernel starts when its stream predecessor has completed.  The release
creates an obligation for the dependents, not for this kernel: every grid that can start before this one completes issues its own wait
before it reads ``out`` and before it writes ``attn``, ``gate`` or ``scale`` (this kernel reads them after the release).  No allocation of
this kernel can be starved by a co-resident dependent (it holds no shared or tensor memory).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_times_inv_to_e4m3, rcp_div_rn
from cudnn.frost.tile_dsl.tma import ld_global_v4

from .gate_quant_decode_config import VEC

BF16 = cutlass.BFloat16
FP32 = cutlass.Float32
INT32 = cutlass.Int32
WORDS = VEC // 2  # 32-bit words (two bf16 values each) of a thread's 128-bit load


def _gate_bf16(a, g, one):
    """``bf16(a / (1 + exp(-g)))``: the gated value rounded ONCE to bf16 (nearest-even).  ``cute.math.exp`` at its default flags (the
    precise lowering), ``/`` = ``div.rn.f32``; see the module docstring before changing either."""
    return (a / (one + cute.math.exp(-g))).to(BF16)


@cute.kernel
def frost_attention_gate_quant(cfg: cutlass.Constexpr, attn: cute.Tensor, gate: cute.Tensor, scale: cute.Tensor, out: cute.Tensor, N: cutlass.Int32):
    if cutlass.const_expr(cfg.PDL_RELEASE):
        launch_dependent_grids()  # the only PDL site; this kernel holds no wait (the module docstring's PDL-sites paragraph)
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    idx = (bid * cfg.THREADS + tid) * VEC
    if idx < N:
        av = ld_global_v4((attn.iterator + idx).toint(), INT32)  # VEC bf16 = 16 B
        gv = ld_global_v4((gate.iterator + idx).toint(), INT32)
        inv = rcp_div_rn(scale[0])
        one = FP32(1)
        for w in cutlass.range_constexpr(WORDS):  # one 32-bit word = two bf16 elements of each operand
            a_lo, a_hi = f16x2_to_f32(av[w], dtype=BF16)
            g_lo, g_hi = f16x2_to_f32(gv[w], dtype=BF16)
            v_lo = _gate_bf16(a_lo, g_lo, one)
            v_hi = _gate_bf16(a_hi, g_hi, one)
            out[idx + 2 * w] = fp32_times_inv_to_e4m3(v_lo.to(FP32), inv)  # the bf16 value widened back exactly, scaled, rounded to E4M3
            out[idx + 2 * w + 1] = fp32_times_inv_to_e4m3(v_hi.to(FP32), inv)


frost_attention_gate_quant.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch(cfg: cutlass.Constexpr, attn: cute.Tensor, gate: cute.Tensor, scale: cute.Tensor, out: cute.Tensor, N: cutlass.Int32, stream: cuda.CUstream):
    """One ordinary launch (never the programmatic attribute): grid ``(ceil(N / VEC / THREADS), 1, 1)``, ``THREADS`` threads.  ``N`` (the
    element count ``rows x width``) is a runtime argument; the config is the compile key."""
    blocks = (N // VEC + cfg.THREADS - 1) // cfg.THREADS
    frost_attention_gate_quant(cfg, attn, gate, scale, out, N).launch(grid=(blocks, 1, 1), block=(cfg.THREADS, 1, 1), stream=stream)
