# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The block's per-tensor FP8 quantize pass (``kernels/quantize.py``).

Bit-exactness against torch's own saturating RNE cast is the contract: a
quantizer that rounds differently from the framework's reference would show up
as a numerics drift in every FP8 stage downstream, blamed on the wrong kernel.

The fp8 ``cvt`` needs sm_89+; on an A100 the numeric tests skip (the shape
algebra still runs).

The quantized BACKWARD's gradient quantization rides on the same module: the amax
pass and the scalar-block init need no fp8 instruction and are pinned bitwise on
every CUDA device; the scale-from-amax / publish arm of the quantize kernel is
pinned against the CPU formula (``grad_scale_from_amax``) on sm_89+.
"""

import math
import os
import subprocess
import sys
import textwrap
from fractions import Fraction

import numpy as np
import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block.kernels.quantize import (  # noqa: E402
    AMAX_CTAS_PER_SM,
    AMAX_SOURCES,
    ELEMS_PER_LANE,
    FP8_E4M3_MAX,
    MAX_ALPHA,
    MAX_INIT_CONSTS,
    SCALE_SOURCES,
    AmaxPartialsRecipe,
    AmaxRecipe,
    InitScalarsRecipe,
    QuantizeRecipe,
    amax_moved_bytes,
    check_scalar_slot,
    compile_amax,
    compile_amax_partials,
    compile_init_scalars,
    compile_quantize,
    grad_scale_from_amax,
    lanes_per_row,
    moved_bytes,
    n_partials_for,
    resolve_amax_src,
    run_amax,
    run_amax_partials,
    run_init_scalars,
    run_quantize,
    validate_shape,
)


def _fp8_cvt_available() -> bool:
    return torch.cuda.is_available() and tuple(torch.cuda.get_device_capability()) >= (8, 9)


requires_fp8 = pytest.mark.skipif(not _fp8_cvt_available(), reason="the fp8 cvt.rn.satfinite instruction needs sm_89+")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

N_QKVG = 17408  # the 397B slab width; V's block starts at 16896
V_OFFSET = 16896


def _reference(x: torch.Tensor, scale: float) -> torch.Tensor:
    # Clamp first so the reference is right even on a torch whose fp8 cast
    # returns NaN on overflow (the pinned torch saturates, matching the PTX).
    return torch.clamp(x.float() * scale, -FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)


def _run(src, h, d, scale_val):
    r = compile_quantize(dtype_in=src.dtype, h=h, d=d)
    dst = torch.empty(int(src.shape[0]), h, d, dtype=torch.float8_e4m3fn, device="cuda")
    scale = torch.tensor([scale_val], dtype=torch.float32, device="cuda")
    run_quantize(r, src, dst, scale, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return dst


# ---------------------------------------------------------------------------
# Shape algebra -- no GPU
# ---------------------------------------------------------------------------


def test_lane_layout_at_d256_is_half_a_warp_per_row():
    assert lanes_per_row(256) == 16 and ELEMS_PER_LANE == 16
    validate_shape(256, 128)
    validate_shape(512, 128)  # a full warp per row
    validate_shape(1024, 128)  # two chunks per lane


def test_validate_shape_declines_what_the_lanes_cannot_cover():
    with pytest.raises(ValueError, match="multiple of 16"):
        validate_shape(200, 128)
    with pytest.raises(ValueError, match="divide a warp"):
        validate_shape(80, 128)  # 5 lanes/row
    with pytest.raises(ValueError, match="threads_per_cta"):
        validate_shape(256, 40)


def test_moved_bytes_counts_the_bf16_read_and_the_fp8_write():
    assert moved_bytes(1000, 32, 256) == 1000 * 32 * 256 * 3
    assert moved_bytes(7, 2, 256, src_elem_bytes=2) == 7 * 2 * 256 * 3


def test_compile_declines_a_fp32_source():
    with pytest.raises(ValueError, match="bf16/f16"):
        compile_quantize(dtype_in=torch.float32, h=2, d=256)


# ---------------------------------------------------------------------------
# Numerics -- sm_89+
# ---------------------------------------------------------------------------


@requires_fp8
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("t, h", [(1000, 32), (4096, 2), (1000, 2)])
def test_compact_source_is_bit_exact_vs_torch(dtype, t, h):
    """Compact ``[T, H, D]`` in, compact fp8 out; includes a tail (T=1000 is not a multiple of the rows per CTA)."""
    torch.manual_seed(0)
    d = 256
    # Values spanning the whole E4M3 range incl. saturation and subnormals: N(0, 100) at scale 3.0 saturates ~13 %.
    x = (torch.randn(t, h, d, device="cuda") * 100.0).to(dtype)
    scale_val = 3.0
    got = _run(x, h, d, scale_val)
    ref = _reference(x, scale_val)
    assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8)), "fp8 bit pattern differs from torch's saturating RNE cast"
    assert (got.float().abs() == FP8_E4M3_MAX).float().mean().item() > 0.05, "the test data did not exercise saturation"


@requires_fp8
@pytest.mark.parametrize("col_off", [0, V_OFFSET])
def test_strided_slab_slice_source(col_off):
    """The source is a column slice of the fused ``[T, 17408]`` projection slab (token stride 17408), as the block hands it."""
    torch.manual_seed(1)
    t, h, d = 1000, 2, 256
    slab = torch.randn(t, N_QKVG, device="cuda").to(torch.bfloat16)
    src = torch.as_strided(slab, (t, h, d), (N_QKVG, d, 1), storage_offset=col_off)
    assert src.stride(0) == N_QKVG and not src.is_contiguous()
    got = _run(src, h, d, 0.75)
    ref = _reference(src, 0.75)
    assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
    # The slice past the destination must be untouched by construction: dst is its own compact buffer.
    assert got.is_contiguous()


@requires_fp8
def test_scale_is_read_from_the_device_tensor():
    torch.manual_seed(2)
    t, h, d = 256, 2, 256
    x = torch.randn(t, h, d, device="cuda").to(torch.bfloat16)
    a = _run(x, h, d, 0.5)
    b = _run(x, h, d, 2.0)
    assert not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    assert torch.equal(a.view(torch.uint8), _reference(x, 0.5).view(torch.uint8))
    assert torch.equal(b.view(torch.uint8), _reference(x, 2.0).view(torch.uint8))
    # Changing the tensor's VALUE without recompiling changes the output: no baked scale.
    r = compile_quantize(dtype_in=torch.bfloat16, h=h, d=d)
    scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    dst = torch.empty(t, h, d, dtype=torch.float8_e4m3fn, device="cuda")
    run_quantize(r, x, dst, scale, stream=torch.cuda.current_stream().cuda_stream)
    scale.fill_(2.0)
    run_quantize(r, x, dst, scale, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(dst.view(torch.uint8), b.view(torch.uint8))


@requires_fp8
def test_run_declines_a_mismatched_binding():
    r = compile_quantize(dtype_in=torch.bfloat16, h=2, d=256)
    x = torch.randn(64, 2, 256, device="cuda").to(torch.bfloat16)
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    st = torch.cuda.current_stream().cuda_stream
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        run_quantize(r, x, torch.empty(64, 2, 256, dtype=torch.bfloat16, device="cuda"), scale, stream=st)
    with pytest.raises(ValueError, match="H=2"):
        run_quantize(
            r, torch.randn(64, 4, 256, device="cuda").to(torch.bfloat16), torch.empty(64, 4, 256, dtype=torch.float8_e4m3fn, device="cuda"), scale, stream=st
        )
    with pytest.raises(ValueError, match="1-element fp32"):
        run_quantize(
            r, x, torch.empty(64, 2, 256, dtype=torch.float8_e4m3fn, device="cuda"), torch.tensor([1.0], device="cuda", dtype=torch.float64), stream=st
        )


def _time_ms(fn, *, iters: int, warmup: int, flush=None) -> float:
    """Median of `iters` CUDA-event-timed launches, L2-FLUSHED between them when
    `flush` is given -- without the flush, iteration 2 onward reads whatever of the
    working set fits in L2 and the GB/s is an L2/HBM blend (frost-tile-dsl.md, "A
    bandwidth number above the ceiling means your NUMERATOR is L2-fed")."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for a, b in ev:
        if flush is not None:
            flush()
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    return sorted(a.elapsed_time(b) for a, b in ev)[iters // 2]


class _L2Flusher:
    """Write a buffer comfortably larger than L2 so the next kernel starts cold."""

    def __init__(self, l2_bytes: int, factor: float = 3.0):
        self.buf = torch.empty(int(l2_bytes * factor) // 4, dtype=torch.int32, device="cuda")

    def __call__(self) -> None:
        self.buf.random_(0, 1 << 30)


@requires_fp8
def test_bandwidth_sign_check_prints_a_finite_positive_rate():
    """GB/s at T=32768 H=32 D=256, L2-flushed -- a SIGN check that PRINTS the rate on
    whatever box runs it. The only assertion is that the measurement is finite and
    positive: a GB/s floor depends on the GPU model and on shared load, so none is
    asserted in the default L0 selection. Perf numbers come from the perf node."""
    from cudnn.frost.device import l2_cache_bytes

    t, h, d = 32768, 32, 256
    x = torch.randn(t, h, d, device="cuda").to(torch.bfloat16)
    dst = torch.empty(t, h, d, dtype=torch.float8_e4m3fn, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    r = compile_quantize(dtype_in=torch.bfloat16, h=h, d=d)
    st = torch.cuda.current_stream().cuda_stream
    flush = _L2Flusher(l2_cache_bytes(torch.cuda.current_device()))
    ms = _time_ms(lambda: run_quantize(r, x, dst, scale, stream=st), iters=20, warmup=5, flush=flush)
    gbs = moved_bytes(t, h, d) / (ms * 1e-3) / 1e9
    print(f"\nquantize fp8: T={t} H={h} D={d}  {ms:.4f} ms  {gbs:.0f} GB/s (L2-flushed, {torch.cuda.get_device_name()})")
    assert math.isfinite(gbs) and gbs > 0, f"the timing did not produce a usable rate: {ms} ms, {gbs} GB/s"


# ---------------------------------------------------------------------------
# The quantized backward's gradient quantization: the "current" scale formula, the amax pass, the publish arm, the init
# ---------------------------------------------------------------------------


def _floor_log2(q: Fraction) -> int:
    """``floor(log2(q))`` for a positive rational, EXACTLY (no float log)."""
    k = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** k > q:
        k -= 1
    assert Fraction(2) ** k <= q < Fraction(2) ** (k + 1)
    return k


def _e4m3_positive_values():
    codes = torch.arange(1, 128, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    return [float(v) for v in codes.tolist() if math.isfinite(v) and v > 0]


def test_grad_scale_from_amax_is_the_exact_formula_and_never_saturates():
    """``scale = 2 ** (floor(log2(448 / amax)) - margin)``, derived in EXACT rational arithmetic here (the helper uses frexp on
    the exponent and never divides); clamped to a finite normal power of two; ``amax * scale <= 448`` for every finite amax
    (nothing the kernel quantizes at that scale saturates); ``1.0`` at ``amax == 0``.  The sweep covers every positive e4m3
    code, the ``m == 0.875`` boundary (``0.875 * 2**k``, where the floor steps) and its fp32 neighbours, 448 / 500, a
    subnormal fp32 amax (the ``2**127`` clamp) and the largest finite fp32 (``2**-120``: the low clamp is unreachable for a finite fp32)."""
    boundary = [math.ldexp(0.875, k) for k in range(-30, 40)]
    neighbours = [float(np.nextafter(np.float32(b), np.float32(np.inf))) for b in boundary] + [
        float(np.nextafter(np.float32(b), np.float32(0))) for b in boundary
    ]
    amaxes = sorted(
        set(_e4m3_positive_values() + boundary + neighbours + [FP8_E4M3_MAX, 500.0, 1e-3, 1e-30, 1e-40, 3.0, 1.0, 0.5, float(np.finfo(np.float32).max)])
    )
    assert 1e-40 < float(np.finfo(np.float32).tiny), "1e-40 must be an fp32 SUBNORMAL for the clamp cell"
    for margin in (0, 2):
        for amax in amaxes:
            scale = grad_scale_from_amax(amax, margin)
            k_exact = _floor_log2(Fraction(FP8_E4M3_MAX) / Fraction(amax)) - margin
            k = max(min(k_exact, 127), -126)
            assert Fraction(scale) == Fraction(2) ** k, (amax, margin, scale, k)
            assert amax * scale <= FP8_E4M3_MAX, (amax, margin, scale)
            if k == k_exact:
                # the floor is TIGHT: one more doubling would overshoot the (margin-reduced) range
                assert Fraction(amax) * Fraction(scale) * 2 > Fraction(FP8_E4M3_MAX) / Fraction(2) ** margin
            assert math.isfinite(1.0 / scale) and 1.0 / scale == math.ldexp(1.0, -k), "the reciprocal is exact"
    assert grad_scale_from_amax(0.0) == 1.0 and grad_scale_from_amax(0.0, 2) == 1.0
    assert grad_scale_from_amax(FP8_E4M3_MAX) == 1.0 and grad_scale_from_amax(500.0) == 0.5 and grad_scale_from_amax(224.0) == 2.0
    assert grad_scale_from_amax(1e-40) == math.ldexp(1.0, 127), "a subnormal amax lands on the 2**127 clamp"
    fmax = float(np.finfo(np.float32).max)  # 448 / 3.4e38 ~= 2**-120: the -126 clamp is UNREACHABLE for a finite fp32 amax (a guard, not a case)
    assert grad_scale_from_amax(fmax) == math.ldexp(1.0, _floor_log2(Fraction(FP8_E4M3_MAX) / Fraction(fmax))) == math.ldexp(1.0, -120)
    for bad in (-1.0, float("nan"), float("inf"), "0.5", True):
        with pytest.raises(ValueError, match="amax"):
            grad_scale_from_amax(bad)
    with pytest.raises(ValueError, match="margin_log2"):
        grad_scale_from_amax(1.0, 1.5)


def test_amax_moved_bytes_is_the_source_read():
    assert amax_moved_bytes(1000, 32, 256) == 1000 * 32 * 256 * 2
    assert amax_moved_bytes(7, 2, 256, src_elem_bytes=2) == 7 * 2 * 256 * 2


def test_recipe_defaults_keep_the_old_constructions_valid():
    """Appended, defaulted fields: a recipe built the old way is today's artifact's."""
    r = QuantizeRecipe(compiled=None, h=2, d=256, rows_per_cta=16, dtype_in=torch.bfloat16)
    assert (r.scale_src, r.n_alpha, r.margin_log2, r.publish, r.amax_src) == ("given", 0, 0, False, None)
    assert SCALE_SOURCES == ("given", "amax") and MAX_ALPHA >= 2 and AMAX_SOURCES == ("none", "slot", "partials") and AMAX_CTAS_PER_SM >= 1
    # the pre-partials constructions resolve as they always read: a "given" artifact reads no amax, an "amax" one the slot
    assert resolve_amax_src("given", None) == "none" and resolve_amax_src("amax", None) == "slot" and resolve_amax_src("given", "partials") == "partials"
    with pytest.raises(ValueError, match="amax_src"):
        compile_quantize(dtype_in=torch.bfloat16, h=2, d=256, amax_src="atomics")
    with pytest.raises(ValueError, match="amax_src must be 'slot' or 'partials'"):
        compile_quantize(dtype_in=torch.bfloat16, h=2, d=256, scale_src="amax", amax_src="none")
    with pytest.raises(ValueError, match="threads_per_cta % 32"):
        compile_quantize(dtype_in=torch.bfloat16, h=2, d=128, amax_src="partials", threads_per_cta=48)
    with pytest.raises(ValueError, match="scale_src"):
        compile_quantize(dtype_in=torch.bfloat16, h=2, d=256, scale_src="caller")
    with pytest.raises(ValueError, match="n_alpha"):
        compile_quantize(dtype_in=torch.bfloat16, h=2, d=256, n_alpha=-1)
    with pytest.raises(ValueError, match=f"exceeds the {MAX_ALPHA}"):
        compile_quantize(dtype_in=torch.bfloat16, h=2, d=256, n_alpha=MAX_ALPHA + 1)
    with pytest.raises(ValueError, match="margin_log2"):
        compile_quantize(dtype_in=torch.bfloat16, h=2, d=256, scale_src="amax", margin_log2=1.5)
    with pytest.raises(ValueError, match="publish"):
        compile_quantize(dtype_in=torch.bfloat16, h=2, d=256, publish=1)
    with pytest.raises(ValueError, match="multiple of 32"):
        compile_amax(dtype_in=torch.bfloat16, h=2, d=128, threads_per_cta=48)
    with pytest.raises(ValueError, match="bf16/f16"):
        compile_amax(dtype_in=torch.float32, h=2, d=256)
    with pytest.raises(ValueError, match="n_slots"):
        compile_init_scalars(0)
    # appended: the plan-time-constant arguments (defaults = no constants, today's artifact)
    assert (InitScalarsRecipe(compiled=None, n_slots=3).const_slot0, InitScalarsRecipe(compiled=None, n_slots=3).n_consts) == (0, 0)
    assert MAX_INIT_CONSTS >= 14
    with pytest.raises(ValueError, match="const_slot0"):
        compile_init_scalars(15, const_slot0=-1)
    with pytest.raises(ValueError, match="n_consts"):
        compile_init_scalars(15, n_consts=True)
    with pytest.raises(ValueError, match=f"exceeds the {MAX_INIT_CONSTS}"):
        compile_init_scalars(64, 0, MAX_INIT_CONSTS + 1)
    with pytest.raises(ValueError, match="do not fit"):
        compile_init_scalars(15, 14, 2)


@requires_cuda
def test_compile_quantize_declines_a_pre_ada_device_by_name():
    """The fp8 ``cvt`` needs sm_89+: a pre-Ada device gets a typed decline NAMING the arch at compile (AGENTS.md Rule 7),
    never the ptxas error the trace would die with.  On sm_89+ the compile goes through (the other cells pin it)."""
    if _fp8_cvt_available():
        pytest.skip("sm_89+: the fp8 cvt is available, the decline cannot fire here")
    cc = torch.cuda.get_device_capability()
    with pytest.raises(NotImplementedError, match=rf"sm_89\+.*sm_{cc[0]}{cc[1]}"):
        compile_quantize(dtype_in=torch.bfloat16, h=2, d=256)


def _slot_in_block(offset_words: int = 1):
    """A 1-element fp32 view at a 4-BYTE (not 16-byte) offset into a NaN-poisoned block, zeroed: the packed slot stride the
    scalar block uses (``assumed_align=4``)."""
    block = torch.full((64,), float("nan"), device="cuda", dtype=torch.float32)
    slot = block[offset_words : offset_words + 1]
    slot.zero_()
    assert slot.data_ptr() % 16 == 4 * (offset_words % 4) and slot.data_ptr() % 4 == 0
    return block, slot


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("t, h, d", [(1000, 32, 256), (37, 4, 128), (1, 2, 64), (257, 3, 256)])
def test_amax_is_bitwise_the_fp32_max_of_abs(dtype, t, h, d):
    """``amax == x.float().abs().amax()`` EXACTLY (the max is order-free; the int32 atomicMax of non-negative fp32 patterns IS
    the fp32 max), on every CUDA device: compact sources, ragged T (a partial last CTA whose clamped rows are selected out),
    the max planted negative and in the LAST row, a slot at a 4-byte offset of a poisoned block (nothing else written), two
    runs bitwise.  The slot is PRE-ZEROED by the caller: a second launch over smaller data can only RAISE it -- a poisoned
    slot is the caller's bug, documented in ``run_amax``."""
    torch.manual_seed(t * h + d)
    x = (torch.randn(t, h, d, device="cuda") * 100.0).to(dtype)
    x[-1, -1, -1] = -(x.float().abs().amax().item() * 1.5)  # the max is negative, in the very last element
    want = x.float().abs().amax()
    block, slot = _slot_in_block(1)
    r = compile_amax(dtype_in=dtype, h=h, d=d)
    st = torch.cuda.current_stream().cuda_stream
    run_amax(r, x, slot, stream=st)
    torch.cuda.synchronize()
    assert torch.equal(slot[0], want), f"amax {slot.item()!r} != max|x| {want.item()!r}"
    assert torch.isnan(block[0]) and torch.isnan(block[2:]).all(), "the launch wrote outside its 4-byte slot"
    block2, slot2 = _slot_in_block(3)
    run_amax(r, x, slot2, stream=st)
    torch.cuda.synchronize()
    assert torch.equal(slot2, slot), "two runs differ"
    # a smaller tensor cannot LOWER a slot (atomicMax): the pre-zero is the caller's contract
    run_amax(r, (x.float() * 0.25).to(dtype), slot, stream=st)
    torch.cuda.synchronize()
    assert torch.equal(slot[0], want)
    slot.zero_()
    run_amax(r, (x.float() * 0.25).to(dtype), slot, stream=st)
    torch.cuda.synchronize()
    assert torch.equal(slot[0], (x.float() * 0.25).to(dtype).float().abs().amax())


@requires_cuda
@pytest.mark.parametrize("col_off", [0, V_OFFSET])
def test_amax_over_a_strided_slab_slice(col_off):
    """The source is a column slice of the fused ``[T, 17408]`` slab (token stride 17408) -- the dqkvg / dY bands the backward
    folds: the max is over the slice only."""
    torch.manual_seed(11)
    t, h, d = 1000, 2, 256
    slab = (torch.randn(t, N_QKVG, device="cuda") * 50.0).to(torch.bfloat16)
    slab[:, (col_off + 3 * 256) % N_QKVG] = 1e4  # a column OUTSIDE the slice is huge (unless it falls inside: then it is the max)
    src = torch.as_strided(slab, (t, h, d), (N_QKVG, d, 1), storage_offset=col_off)
    assert src.stride(0) == N_QKVG and not src.is_contiguous()
    slot = torch.zeros(1, device="cuda", dtype=torch.float32)
    run_amax(compile_amax(dtype_in=torch.bfloat16, h=h, d=d), src, slot, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(slot[0], src.float().abs().amax())


@requires_cuda
def test_amax_contract_is_typed():
    """Host checks before any launch: dtype / shape / strides against the artifact, the slot contract (fp32, 1 element, CUDA,
    4-byte aligned), one device, no empty source."""
    r = AmaxRecipe(compiled=None, h=2, d=256, rows_per_cta=16, dtype_in=torch.bfloat16)
    x = torch.randn(64, 2, 256, device="cuda").to(torch.bfloat16)
    slot = torch.zeros(1, device="cuda")
    st = torch.cuda.current_stream().cuda_stream
    with pytest.raises(ValueError, match="compiled for torch.bfloat16"):
        run_amax(r, x.to(torch.float16), slot, stream=st)
    with pytest.raises(ValueError, match="H=2"):
        run_amax(r, torch.randn(64, 4, 256, device="cuda").to(torch.bfloat16), slot, stream=st)
    with pytest.raises(ValueError, match="head stride"):
        run_amax(r, torch.empty(64, 256, 2, device="cuda", dtype=torch.bfloat16).transpose(1, 2), slot, stream=st)  # strides (512, 1, 2)
    with pytest.raises(ValueError, match="T == 0"):
        run_amax(r, x[:0], slot, stream=st)
    with pytest.raises(ValueError, match="amax must be a 1-element fp32 CUDA tensor"):
        run_amax(r, x, torch.zeros(1, device="cuda", dtype=torch.float64), stream=st)
    with pytest.raises(ValueError, match="amax must be a 1-element fp32 CUDA tensor"):
        run_amax(r, x, torch.zeros(2, device="cuda"), stream=st)
    with pytest.raises(ValueError, match="amax must be a 1-element fp32 CUDA tensor"):
        run_amax(r, x, torch.zeros(1), stream=st)
    with pytest.raises(ValueError, match="one CUDA device"):
        run_amax(r, x.cpu(), slot, stream=st)
    check_scalar_slot("ok", torch.zeros(8, device="cuda")[1:2])  # a 4-byte-offset slot is LEGAL
    with pytest.raises(ValueError, match="contiguous"):
        check_scalar_slot("slots", torch.zeros(32, device="cuda")[::2], numel=16)


def _quant_recipe(**kw):
    base = dict(compiled=None, h=2, d=256, rows_per_cta=16, dtype_in=torch.bfloat16)
    return QuantizeRecipe(**{**base, **kw})


@requires_cuda
def test_quantize_publish_contract_is_typed():
    """Both directions of every slot against the recipe (Rule 1), before any launch: ``amax`` on a given artifact, ``scale``
    on an amax one, a missing ``amax`` / ``scale_out`` / ``descale``, ``scale_out`` on an artifact without the publish arm,
    an alpha tuple of the wrong length, a CPU / fp64 slot, a published slot aliasing a read slot or another published slot."""
    x = torch.randn(64, 2, 256, device="cuda").to(torch.bfloat16)
    dst = torch.empty(64, 2, 256, dtype=torch.float8_e4m3fn, device="cuda")
    st = torch.cuda.current_stream().cuda_stream
    blk = torch.zeros(16, device="cuda", dtype=torch.float32)
    s = [blk[i : i + 1] for i in range(16)]  # 16 distinct 4-byte slots
    given = _quant_recipe()
    given_pub = _quant_recipe(publish=True)
    amax2 = _quant_recipe(scale_src="amax", n_alpha=2, publish=True)
    with pytest.raises(ValueError, match="passing amax would silently ignore it"):
        run_quantize(given, x, dst, s[0], stream=st, amax=s[1])
    with pytest.raises(ValueError, match="passing scale_out / descale would silently ignore"):
        run_quantize(given, x, dst, s[0], stream=st, scale_out=s[1])
    with pytest.raises(ValueError, match="passing scale_out / descale would silently ignore"):
        run_quantize(given, x, dst, s[0], stream=st, descale=s[1])
    with pytest.raises(ValueError, match="scale_out AND descale"):
        run_quantize(given_pub, x, dst, s[0], stream=st, scale_out=s[1])
    with pytest.raises(ValueError, match="scale_out AND descale"):
        run_quantize(given_pub, x, dst, s[0], stream=st, descale=s[2])
    with pytest.raises(ValueError, match="passing a caller scale would silently ignore it"):
        run_quantize(amax2, x, dst, s[0], stream=st, amax=s[1], scale_out=s[2], descale=s[3], alpha_consts=(s[4], s[5]), alpha_outs=(s[6], s[7]))
    with pytest.raises(ValueError, match="amax .*must be bound"):
        run_quantize(amax2, x, dst, None, stream=st, scale_out=s[2], descale=s[3], alpha_consts=(s[4], s[5]), alpha_outs=(s[6], s[7]))
    with pytest.raises(ValueError, match="scale_out .*must be bound"):
        run_quantize(amax2, x, dst, None, stream=st, amax=s[1], descale=s[3], alpha_consts=(s[4], s[5]), alpha_outs=(s[6], s[7]))
    with pytest.raises(ValueError, match="scale_out AND descale"):
        run_quantize(amax2, x, dst, None, stream=st, amax=s[1], scale_out=s[2], alpha_consts=(s[4], s[5]), alpha_outs=(s[6], s[7]))
    with pytest.raises(ValueError, match="n_alpha=2 alpha products; got 1 alpha_consts and 2 alpha_outs"):
        run_quantize(amax2, x, dst, None, stream=st, amax=s[1], scale_out=s[2], descale=s[3], alpha_consts=(s[4],), alpha_outs=(s[6], s[7]))
    with pytest.raises(ValueError, match=r"alpha_outs\[1\] must be a 1-element fp32 CUDA tensor"):
        run_quantize(amax2, x, dst, None, stream=st, amax=s[1], scale_out=s[2], descale=s[3], alpha_consts=(s[4], s[5]), alpha_outs=(s[6], torch.zeros(1)))
    with pytest.raises(ValueError, match="amax must be a 1-element fp32 CUDA tensor"):
        run_quantize(amax2, x, dst, None, stream=st, amax=s[1].double(), scale_out=s[2], descale=s[3], alpha_consts=(s[4], s[5]), alpha_outs=(s[6], s[7]))
    with pytest.raises(ValueError, match="scale_out aliases amax"):
        run_quantize(amax2, x, dst, None, stream=st, amax=s[1], scale_out=s[1], descale=s[3], alpha_consts=(s[4], s[5]), alpha_outs=(s[6], s[7]))
    with pytest.raises(ValueError, match=r"alpha_outs\[0\] aliases alpha_consts\[1\]"):
        run_quantize(amax2, x, dst, None, stream=st, amax=s[1], scale_out=s[2], descale=s[3], alpha_consts=(s[4], s[5]), alpha_outs=(s[5], s[7]))
    with pytest.raises(ValueError, match="descale and alpha_outs\\[1\\] are the same slot"):
        run_quantize(amax2, x, dst, None, stream=st, amax=s[1], scale_out=s[2], descale=s[3], alpha_consts=(s[4], s[5]), alpha_outs=(s[6], s[3]))
    with pytest.raises(ValueError, match="scale_out aliases scale"):
        run_quantize(given_pub, x, dst, s[0], stream=st, scale_out=s[0], descale=s[3])
    with pytest.raises(ValueError, match="T == 0"):
        run_quantize(given, x[:0], dst[:0], s[0], stream=st)
    assert torch.equal(blk, torch.zeros_like(blk)), "no launch happened: nothing wrote a slot"


@requires_cuda
def test_run_quantize_refuses_operands_off_the_launch_device():
    """``src``, ``dst``, ``scale`` and every scalar slot are dereferenced in-kernel through device pointers: an operand on the
    host, or on ANOTHER GPU (which the slot contract's ``is_cuda`` cannot see), passes the dtype / shape / slot checks and
    would be a wild read or write -- an illegal-address fault at the next synchronize, sticky for the process, or silent
    peer traffic.  Refused by name -- the operand, the launch device and where it actually is -- before the artifact is
    touched (``compiled=None``: no launch can happen).  ``src`` anchors the device; a CPU ``dst`` (no earlier device check)
    reaches the one-device check on any host; a slot on a SECOND GPU is asserted where two devices are visible."""
    x = torch.randn(64, 2, 256, device="cuda").to(torch.bfloat16)
    dst = torch.empty(64, 2, 256, dtype=torch.float8_e4m3fn, device="cuda")
    blk = torch.zeros(16, device="cuda", dtype=torch.float32)
    s = [blk[i : i + 1] for i in range(16)]
    st = torch.cuda.current_stream().cuda_stream
    dev = x.device
    given = _quant_recipe()
    amax2 = _quant_recipe(scale_src="amax", n_alpha=2, publish=True)
    amax_kw = dict(amax=s[1], scale_out=s[2], descale=s[3], alpha_consts=(s[4], s[5]), alpha_outs=(s[6], s[7]))
    with pytest.raises(ValueError, match="src must be a CUDA tensor"):
        run_quantize(given, x.cpu(), dst, s[0], stream=st)
    with pytest.raises(ValueError, match=rf"dst must be on {dev} with src, got cpu"):
        run_quantize(given, x, dst.cpu(), s[0], stream=st)
    with pytest.raises(ValueError, match=rf"dst must be on {dev} with src, got cpu"):
        run_quantize(amax2, x, dst.cpu(), None, stream=st, **amax_kw)
    if torch.cuda.device_count() >= 2:
        other = torch.device("cuda", 1 if dev.index == 0 else 0)
        far = torch.zeros(1, device=other, dtype=torch.float32)
        check_scalar_slot("far", far)  # a VALID slot (fp32, 1 element, CUDA, 4-byte aligned) -- on the wrong GPU
        with pytest.raises(ValueError, match=rf"scale must be on {dev} with src, got {other}"):
            run_quantize(given, x, dst, far, stream=st)
        for name in ("amax", "scale_out", "descale"):
            with pytest.raises(ValueError, match=rf"{name} must be on {dev} with src, got {other}"):
                run_quantize(amax2, x, dst, None, stream=st, **{**amax_kw, name: far})
        with pytest.raises(ValueError, match=rf"alpha_consts\[1\] must be on {dev} with src, got {other}"):
            run_quantize(amax2, x, dst, None, stream=st, **{**amax_kw, "alpha_consts": (s[4], far)})
        with pytest.raises(ValueError, match=rf"alpha_outs\[0\] must be on {dev} with src, got {other}"):
            run_quantize(amax2, x, dst, None, stream=st, **{**amax_kw, "alpha_outs": (far, s[7])})
    assert torch.equal(blk, torch.zeros_like(blk)), "no launch happened: nothing wrote a slot"


@requires_cuda
@pytest.mark.parametrize("scale_val", [3.0, 2.0**-7, 7.25, 1.0 / 3.0], ids=["3", "2^-7", "7.25", "1/3"])
def test_init_scalars_zeroes_and_reciprocates(scale_val):
    """15 NaN-poisoned slots -> exact 0; slot 14 (a view INTO the block) -> ``1 / scale_dp`` BITWISE the IEEE RN division
    (``np.float32(1) / np.float32(scale)``: exact for a power of two, correctly rounded for 7.25 and 1/3 -- a reciprocal
    multiply would be one ulp off); the poison past the 15 slots untouched; a ``descale_dp_out`` OUTSIDE the block works too;
    ``scale_dp`` INSIDE the block is refused (it would be read as 0)."""
    block = torch.full((64,), float("nan"), device="cuda", dtype=torch.float32)
    slots = block[:15]
    scale_dp = torch.tensor([scale_val], device="cuda", dtype=torch.float32)
    r = compile_init_scalars(15)
    assert r.n_slots == 15 and compile_init_scalars(15).compiled is r.compiled
    st = torch.cuda.current_stream().cuda_stream
    run_init_scalars(r, slots, scale_dp, slots[14:15], stream=st)
    torch.cuda.synchronize()
    assert torch.equal(slots[:14], torch.zeros(14, device="cuda"))
    want = np.float32(1.0) / np.float32(scale_val)
    assert slots[14].view(torch.int32).item() == int(np.array([want], dtype=np.float32).view(np.int32)[0]), (slots[14].item(), float(want))
    assert torch.isnan(block[15:]).all(), "the launch wrote past its n_slots"
    outside = torch.full((1,), float("nan"), device="cuda", dtype=torch.float32)
    slots.fill_(float("nan"))
    run_init_scalars(r, slots, scale_dp, outside, stream=st)
    torch.cuda.synchronize()
    assert torch.equal(slots, torch.zeros(15, device="cuda")) and outside.view(torch.int32).item() == int(np.array([want], dtype=np.float32).view(np.int32)[0])
    with pytest.raises(ValueError, match="scale_dp lies inside the slot block"):
        run_init_scalars(r, slots, slots[3:4], slots[14:15], stream=st)
    with pytest.raises(ValueError, match="slots must be a 15-element fp32 CUDA tensor"):
        run_init_scalars(r, block[:16], scale_dp, slots[14:15], stream=st)
    with pytest.raises(ValueError, match="scale_dp must be a 1-element fp32 CUDA tensor"):
        run_init_scalars(r, slots, torch.tensor([scale_val]), slots[14:15], stream=st)
    with pytest.raises(ValueError, match="descale_dp_out must be a 1-element fp32 CUDA tensor"):
        run_init_scalars(r, slots, scale_dp, slots[13:15], stream=st)
    assert InitScalarsRecipe(compiled=None, n_slots=3).n_slots == 3


@requires_cuda
@pytest.mark.parametrize("consts", [(3.0, 2.0**-7, 7.25, 1.0 / 3.0, 256.0, 2.0**-8, 1.0), (0.1,) * 14], ids=["7-mixed", "14-tenths"])
def test_init_scalars_stores_the_plan_time_constants_from_its_arguments(consts):
    """``compile_init_scalars(n_slots, const_slot0, n_consts)``: after the zeroing and the reciprocal, slots ``[const_slot0,
    const_slot0 + n_consts)`` hold ``consts[i]`` BITWISE as ``np.float32(consts[i])`` (the kernel-argument path rounds a Python
    float like ``torch.full``'s fp32 fill), every other slot is zero (the reciprocal's apart), the poison past ``n_slots`` is
    untouched; the SAME compiled artifact serves different VALUES (two launches, two value sets, one ``compiled``: the constants
    are runtime arguments, never trace-time constants that would fork the artifact); the contract is typed both ways -- a
    ``consts`` length other than ``n_consts`` (the plain ``compile_init_scalars(n)`` artifact refuses any), a tensor or a non-finite
    entry, and a ``descale_dp_out`` inside the constants' range."""
    n_consts, slot0, n_slots = len(consts), 15, 15 + len(consts)
    block = torch.full((64,), float("nan"), device="cuda", dtype=torch.float32)
    slots = block[:n_slots]
    scale_dp = torch.tensor([4.0], device="cuda", dtype=torch.float32)
    r = compile_init_scalars(n_slots, slot0, n_consts)
    assert (r.n_slots, r.const_slot0, r.n_consts) == (n_slots, slot0, n_consts) and compile_init_scalars(n_slots, slot0, n_consts).compiled is r.compiled
    assert r.compiled is not compile_init_scalars(n_slots).compiled, "the slot layout is part of the artifact key"
    st = torch.cuda.current_stream().cuda_stream
    run_init_scalars(r, slots, scale_dp, slots[14:15], consts, stream=st)
    torch.cuda.synchronize()
    want = torch.tensor(np.array(consts, dtype=np.float32), device="cuda")
    assert torch.equal(slots[slot0 : slot0 + n_consts].view(torch.int32), want.view(torch.int32)), (slots[slot0:].tolist(), consts)
    assert torch.equal(slots[:14], torch.zeros(14, device="cuda")) and slots[14].item() == 0.25
    assert torch.isnan(block[n_slots:]).all(), "the launch wrote past its n_slots"
    other = tuple(2.0 * c for c in consts)
    run_init_scalars(r, slots, scale_dp, slots[14:15], other, stream=st)  # the same artifact, other values
    torch.cuda.synchronize()
    assert torch.equal(slots[slot0 : slot0 + n_consts], torch.tensor(np.array(other, dtype=np.float32), device="cuda"))
    with pytest.raises(ValueError, match=f"stores n_consts={n_consts}"):
        run_init_scalars(r, slots, scale_dp, slots[14:15], consts[:-1], stream=st)
    with pytest.raises(ValueError, match="stores n_consts=0"):
        run_init_scalars(compile_init_scalars(n_slots), slots, scale_dp, slots[14:15], consts, stream=st)
    with pytest.raises(ValueError, match=r"consts\[0\] must be a finite Python number"):
        run_init_scalars(r, slots, scale_dp, slots[14:15], (torch.ones(1, device="cuda"),) + consts[1:], stream=st)
    with pytest.raises(ValueError, match=r"consts\[1\] must be a finite Python number"):
        run_init_scalars(r, slots, scale_dp, slots[14:15], (consts[0], float("inf")) + consts[2:], stream=st)
    with pytest.raises(ValueError, match="descale_dp_out lies inside"):
        run_init_scalars(r, slots, scale_dp, slots[slot0 : slot0 + 1], consts, stream=st)
    assert torch.equal(slots[slot0 : slot0 + n_consts], torch.tensor(np.array(other, dtype=np.float32), device="cuda")), "a refused call launched"


@requires_cuda
def test_init_scalars_without_a_dp_scale():
    """``compile_init_scalars(n, const_slot0, n_consts, descale_dp=False)`` -- the arm of a pipeline whose attention backward
    has no dP scalar: 15 NaN-poisoned slots -> exact 0 and the constants stored BITWISE, with NO reciprocal anywhere (every
    slot below the constants is 0, nothing outside ``n_slots`` is touched); the artifact is keyed apart from the dividing one
    and its defaults keep the old constructions; the contract is typed BOTH ways -- a ``scale_dp`` / ``descale_dp_out`` handed
    to the non-dividing artifact is refused (nothing would read or write them), the dividing artifact refuses their absence,
    and a refused call launches nothing."""
    consts = (3.0, 2.0**-7, 7.25, 1.0 / 3.0, 256.0, 2.0**-8, 1.0)
    n_consts, slot0, n_slots = len(consts), 15, 15 + len(consts)
    block = torch.full((64,), float("nan"), device="cuda", dtype=torch.float32)
    slots = block[:n_slots]
    r = compile_init_scalars(n_slots, slot0, n_consts, descale_dp=False)
    assert r.descale_dp is False and (r.n_slots, r.const_slot0, r.n_consts) == (n_slots, slot0, n_consts)
    assert compile_init_scalars(n_slots, slot0, n_consts, descale_dp=False).compiled is r.compiled
    dividing = compile_init_scalars(n_slots, slot0, n_consts)
    assert dividing.descale_dp is True and dividing.compiled is not r.compiled, "descale_dp is part of the artifact key"
    assert InitScalarsRecipe(compiled=None, n_slots=3).descale_dp is True, "the appended field defaults to today's artifact"
    st = torch.cuda.current_stream().cuda_stream
    run_init_scalars(r, slots, consts=consts, stream=st)
    torch.cuda.synchronize()
    want = torch.tensor(np.array(consts, dtype=np.float32), device="cuda")
    assert torch.equal(slots[:slot0], torch.zeros(slot0, device="cuda")), "every slot below the constants is zero: no reciprocal landed anywhere"
    assert torch.equal(slots[slot0 : slot0 + n_consts].view(torch.int32), want.view(torch.int32)), (slots[slot0:].tolist(), consts)
    assert torch.isnan(block[n_slots:]).all(), "the launch wrote past its n_slots"
    r0 = compile_init_scalars(15, descale_dp=False)  # the plain artifact: zeroing only
    slots15 = block[:15]
    slots15.fill_(float("nan"))
    run_init_scalars(r0, slots15, stream=st)
    torch.cuda.synchronize()
    assert torch.equal(slots15, torch.zeros(15, device="cuda"))
    scale_dp = torch.tensor([4.0], device="cuda", dtype=torch.float32)
    outside = torch.full((1,), float("nan"), device="cuda", dtype=torch.float32)
    slots.fill_(float("nan"))
    with pytest.raises(ValueError, match="descale_dp=False"):
        run_init_scalars(r, slots, scale_dp, outside, consts, stream=st)
    with pytest.raises(ValueError, match="descale_dp=False"):
        run_init_scalars(r, slots, descale_dp_out=outside, consts=consts, stream=st)
    with pytest.raises(ValueError, match="pass scale_dp= and descale_dp_out="):
        run_init_scalars(dividing, slots, consts=consts, stream=st)
    with pytest.raises(ValueError, match="pass scale_dp= and descale_dp_out="):
        run_init_scalars(dividing, slots, scale_dp, None, consts, stream=st)
    with pytest.raises(ValueError, match="descale_dp must be a bool"):
        compile_init_scalars(15, descale_dp=1)
    torch.cuda.synchronize()
    assert torch.isnan(slots).all() and torch.isnan(outside).all(), "a refused call launched"
    run_init_scalars(dividing, slots, scale_dp, slots[14:15], consts, stream=st)  # today's positional form, unchanged
    torch.cuda.synchronize()
    assert slots[14].item() == 0.25 and torch.equal(slots[:14], torch.zeros(14, device="cuda"))


_GATE_BWD_TRIPLE_PROBE = textwrap.dedent("""
    import torch
    import cudnn.gated_attention_block.kernels.sigmoid_gate_bwd as sgb
    # the e4m3 arm's compile-time gate names the CURRENT device (Rule 7); this is a trace-compile for sm_107a on whatever box runs it
    sgb.require_fp8_cvt = lambda who: None
    r = sgb.compile_sigmoid_gate_bwd(
        dtype=torch.bfloat16, h=8, d=256, has_og=True, has_seq_lens=False, has_delta=True, og_fp8=True, has_amax_do=False, has_amax_dg=False
    )
    print("TRACED", r.og_fp8, r.has_delta, r.has_amax_do, r.has_amax_dg, r.n_ctas_cap)
    """)


def _sm107a_known_to_the_dsl() -> bool:
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string("sm_107a")
        return True
    except Exception:
        return False


def test_gate_bwd_fp8_arm_traces_without_the_amax_folds(tmp_path):
    """The gate backward's recipe of a BLOCK-SCALED pipeline -- ``og_fp8=True`` with the ``delta`` output and NEITHER amax
    fold (no per-tensor dO / dQKVG amax exists there) -- traces and compiles for sm_107a: the e4m3 O_gated arm must not lean
    on the folds' SMEM array or their partials.  The gate backward has no compile-options knob, so the arch comes from
    ``CUTE_DSL_ARCH`` in a subprocess (read at the first cutlass import); the per-device fp8 gate is stood down there.
    SKIPS (never fails) where the DSL predates ``sm_107a``."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    env = dict(os.environ, CUTE_DSL_ARCH="sm_107a", CUTE_DSL_DUMP_DIR=str(tmp_path))
    proc = subprocess.run([sys.executable, "-c", _GATE_BWD_TRIPLE_PROBE], env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"the recipe triple did not trace-compile:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    assert "TRACED True True False False 0" in proc.stdout, proc.stdout[-500:]


def _amax_arm_run(x, h, d, *, margin, consts):
    """The backward's amax pass + scale-from-amax quantize over one tensor: returns (dst, scale_out, descale, alpha_outs)."""
    st = torch.cuda.current_stream().cuda_stream
    blk = torch.full((16,), float("nan"), device="cuda", dtype=torch.float32)
    amax, scale_out, descale = blk[0:1], blk[4:5], blk[5:6]
    alpha_outs = tuple(blk[10 + i : 11 + i] for i in range(len(consts)))
    amax.zero_()
    run_amax(compile_amax(dtype_in=x.dtype, h=h, d=d), x, amax, stream=st)
    r = compile_quantize(dtype_in=x.dtype, h=h, d=d, scale_src="amax", n_alpha=len(consts), margin_log2=margin)
    assert r.publish is True and r.scale_src == "amax" and r.n_alpha == len(consts) and r.margin_log2 == margin
    dst = torch.empty(int(x.shape[0]), h, d, dtype=torch.float8_e4m3fn, device="cuda")
    alpha_consts = tuple(torch.tensor([c], device="cuda", dtype=torch.float32) for c in consts)
    run_quantize(r, x, dst, None, stream=st, amax=amax, scale_out=scale_out, descale=descale, alpha_consts=alpha_consts, alpha_outs=alpha_outs)
    torch.cuda.synchronize()
    assert torch.isnan(blk[1:4]).all() and torch.isnan(blk[6:10]).all() and torch.isnan(blk[10 + len(consts) :]).all(), "a slot outside the ABI was written"
    return dst, amax, scale_out, descale, alpha_outs


def _bits(t: torch.Tensor) -> int:
    return int(t.view(torch.int32).item())


def _np_bits(v) -> int:
    return int(np.array([v], dtype=np.float32).view(np.int32)[0])


@requires_fp8
@pytest.mark.parametrize("margin", [0, 2])
@pytest.mark.parametrize("amp", [100.0, 1e-3, 3e4], ids=["amp100", "amp1e-3", "amp3e4"])
def test_scale_from_amax_arm_publishes_the_formula_bitwise(margin, amp):
    """The kernel's scale is BITWISE ``grad_scale_from_amax(amax, margin)`` for the amax the pass produced; ``descale`` is the
    exact reciprocal; ``alpha_i`` is ONE fp32 RN multiply ``np.float32(descale) * np.float32(c_i)``; ``dst`` is bitwise torch's
    saturating cast at that scale -- and ``max |x| * scale <= 448`` on the data, so no element saturates (the margin halves
    the range twice).  Three amplitudes walk the exponent range (a 3e4 bf16 amax at margin 0 lands on a scale below 1)."""
    torch.manual_seed(int(amp) + margin)
    t, h, d = 1000, 4, 256
    x = (torch.randn(t, h, d, device="cuda") * amp).to(torch.bfloat16)
    consts = (1.0 / 3.0, 0.7)
    dst, amax, scale_out, descale, alpha_outs = _amax_arm_run(x, h, d, margin=margin, consts=consts)
    assert torch.equal(amax[0], x.float().abs().amax())
    want_scale = grad_scale_from_amax(amax.item(), margin)
    assert _bits(scale_out) == _np_bits(want_scale), (scale_out.item(), want_scale)
    assert _bits(descale) == _np_bits(np.float32(1.0) / np.float32(want_scale)) and descale.item() == 1.0 / want_scale
    for a, c in zip(alpha_outs, consts):
        assert _bits(a) == _np_bits(np.float32(1.0 / want_scale) * np.float32(c)), (a.item(), c)
    assert torch.equal(dst.view(torch.uint8), _reference(x, want_scale).view(torch.uint8)), "dst differs from torch's cast at the kernel's scale"
    assert (x.float().abs() * want_scale).max().item() <= FP8_E4M3_MAX / (2**margin)
    assert not (dst.float().abs() == FP8_E4M3_MAX).any() or (x.float().abs() * want_scale).max().item() == FP8_E4M3_MAX
    # the derivation is per launch, from the slot: a second pass over 4x the data moves the scale down by exactly 2 bits
    dst2, amax2, scale2, _, _ = _amax_arm_run((x.float() * 4.0).to(torch.bfloat16), h, d, margin=margin, consts=consts)
    assert scale2.item() == want_scale / 4.0 or amax2.item() * (want_scale / 4.0) > FP8_E4M3_MAX / (2**margin)  # bf16 re-rounding can cross a boundary


@requires_fp8
def test_given_arm_publishes_the_caller_scale():
    """The "delayed" twin: a ``scale_src="given"`` artifact with the publish arm, fed the scale the "current" run derived,
    writes the SAME ``dst`` and the SAME slots -- the bitwise-replay pin at kernel level; a given artifact WITHOUT alpha
    products but with ``publish=True`` (the dO launch under the delayed recipe) publishes scale and descale only."""
    torch.manual_seed(21)
    t, h, d = 512, 4, 256
    x = (torch.randn(t, h, d, device="cuda") * 50.0).to(torch.bfloat16)
    consts = (0.125, 2.0 / 3.0)
    dst, amax, scale_out, descale, alpha_outs = _amax_arm_run(x, h, d, margin=0, consts=consts)
    st = torch.cuda.current_stream().cuda_stream
    r = compile_quantize(dtype_in=torch.bfloat16, h=h, d=d, scale_src="given", n_alpha=2)
    assert r.publish is True, "n_alpha > 0 implies the publish arm"
    blk = torch.full((8,), float("nan"), device="cuda", dtype=torch.float32)
    dst2 = torch.empty_like(dst)
    alpha_consts = tuple(torch.tensor([c], device="cuda", dtype=torch.float32) for c in consts)
    run_quantize(r, x, dst2, scale_out.clone(), stream=st, scale_out=blk[0:1], descale=blk[1:2], alpha_consts=alpha_consts, alpha_outs=(blk[2:3], blk[3:4]))
    torch.cuda.synchronize()
    assert torch.equal(dst2.view(torch.uint8), dst.view(torch.uint8))
    assert _bits(blk[0:1]) == _bits(scale_out) and _bits(blk[1:2]) == _bits(descale)
    assert _bits(blk[2:3]) == _bits(alpha_outs[0]) and _bits(blk[3:4]) == _bits(alpha_outs[1]) and torch.isnan(blk[4:]).all()
    # publish=True on a given artifact with no alpha: scale + descale only (the delayed recipe's dO launch)
    r0 = compile_quantize(dtype_in=torch.bfloat16, h=h, d=d, publish=True)
    assert r0.publish is True and r0.n_alpha == 0 and r0.compiled is not r.compiled
    blk.fill_(float("nan"))
    sc = torch.tensor([0.75], device="cuda", dtype=torch.float32)  # NOT a power of two: descale is the RN quotient
    run_quantize(r0, x, dst2, sc, stream=st, scale_out=blk[0:1], descale=blk[1:2])
    torch.cuda.synchronize()
    assert blk[0].item() == 0.75 and _bits(blk[1:2]) == _np_bits(np.float32(1.0) / np.float32(0.75)) and torch.isnan(blk[2:]).all()
    assert torch.equal(dst2.view(torch.uint8), _reference(x, 0.75).view(torch.uint8))


@requires_fp8
def test_quantize_default_knobs_share_the_forward_artifact():
    """The defaults (``scale_src="given"``, ``n_alpha=0``, ``margin_log2=0``, ``publish=False``) are ONE artifact -- the forward's
    -- and every new knob is a distinct one (the cache keys on all of them)."""
    kw = dict(dtype_in=torch.bfloat16, h=2, d=256)
    a = compile_quantize(**kw)
    assert compile_quantize(**kw, scale_src="given", n_alpha=0, margin_log2=0, publish=False).compiled is a.compiled
    assert a.publish is False and a.n_alpha == 0 and a.scale_src == "given"
    seen = {id(a.compiled)}
    for other in (
        compile_quantize(**kw, publish=True),
        compile_quantize(**kw, n_alpha=1),
        compile_quantize(**kw, scale_src="amax"),
        compile_quantize(**kw, scale_src="amax", margin_log2=2),
        compile_quantize(**kw, scale_src="amax", n_alpha=2),
    ):
        assert id(other.compiled) not in seen, "two different knob sets share an artifact"
        seen.add(id(other.compiled))


# ---------------------------------------------------------------------------
# The amax PARTIALS pass and the quantize kernel's partials arm (the fused prologue's form of the gradient amax)
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("t, h", [(1000, 16), (4000, 16), (37, 3), (1, 2)], ids=["1000x16", "4000x16-persistent", "37x3-ragged", "1x2"])
def test_amax_partials_pass_is_bitwise_the_slot_pass(t, h):
    """``partials[c] = max |x|`` over persistent CTA ``c``'s row groups, EVERY one written (no slot to pre-zero), their ``max`` ==
    the slot pass's ``atomicMax`` fold == ``x.float().abs().amax()`` bitwise; ``n_partials_for`` = min(row groups, SMs x 8) (the
    4000 x 16 cell strides every CTA over several groups); the words past ``n_partials`` are never touched; the pass runs on
    every CUDA device (no fp8 instruction)."""
    d = 256
    g = torch.Generator(device="cuda").manual_seed(t * 7 + h)
    x = (torch.randn(t, h, d, generator=g, device="cuda") * 3.0).to(torch.bfloat16)
    st = torch.cuda.current_stream().cuda_stream
    slot = torch.zeros(1, device="cuda")
    run_amax(compile_amax(dtype_in=torch.bfloat16, h=h, d=d), x, slot, stream=st)
    r = compile_amax_partials(dtype_in=torch.bfloat16, h=h, d=d)
    assert isinstance(r, AmaxPartialsRecipe) and r.n_ctas_cap == torch.cuda.get_device_properties(0).multi_processor_count * AMAX_CTAS_PER_SM
    groups = (t * h + r.rows_per_cta - 1) // r.rows_per_cta
    assert n_partials_for(r, t) == max(1, min(groups, r.n_ctas_cap))
    if t == 4000:
        assert groups > r.n_ctas_cap, "the persistent cell must stride every CTA over several row groups"
    partials = torch.full((r.n_ctas_cap + 5,), float("nan"), device="cuda")
    n = run_amax_partials(r, x, partials, stream=st)
    torch.cuda.synchronize()
    assert n == n_partials_for(r, t)
    assert not torch.isnan(partials[:n]).any() and torch.isnan(partials[n:]).all(), "a partial was skipped, or a word past n_partials was written"
    assert (partials[:n] >= 0).all() and torch.equal(partials[:n].max(), x.float().abs().amax()) and torch.equal(partials[:n].max(), slot[0])
    with pytest.raises(ValueError, match="n_partials must be an int in"):
        run_amax_partials(r, x, partials[: n - 1], stream=st)  # too short for this T
    with pytest.raises(ValueError, match="contiguous fp32"):
        run_amax_partials(r, x, partials[::2], stream=st)


@requires_fp8
def test_quantize_partials_arm_is_bitwise_the_slot_arm():
    """The quantize kernel fed the partials (``amax_src="partials"``) reduces them in every CTA's prologue, derives the SAME scale as
    the slot arm, writes the SAME e4m3 bytes and the SAME scale / descale / alphas, and PUBLISHES the reduced amax into ``amax_out``
    -- the slot pass's slot value, re-created; the "delayed" form (``scale_src="given"`` + partials) reads the caller's scale and
    still publishes the amax."""
    torch.manual_seed(5)
    t, h, d = 1000, 4, 256
    x = (torch.randn(t, h, d, device="cuda") * 40.0).to(torch.bfloat16)
    st = torch.cuda.current_stream().cuda_stream
    consts = (torch.tensor([0.5], device="cuda"), torch.tensor([1.0 / 3.0], device="cuda"))
    slot = torch.zeros(1, device="cuda")
    run_amax(compile_amax(dtype_in=torch.bfloat16, h=h, d=d), x, slot, stream=st)
    rp = compile_amax_partials(dtype_in=torch.bfloat16, h=h, d=d)
    partials = torch.empty(rp.n_ctas_cap, device="cuda")
    n = run_amax_partials(rp, x, partials, stream=st)
    blk_s = torch.full((16,), float("nan"), device="cuda")
    dst_s = torch.empty(t, h, d, dtype=torch.float8_e4m3fn, device="cuda")
    r_slot = compile_quantize(dtype_in=torch.bfloat16, h=h, d=d, scale_src="amax", n_alpha=2)
    run_quantize(
        r_slot, x, dst_s, None, stream=st, amax=slot, scale_out=blk_s[4:5], descale=blk_s[5:6], alpha_consts=consts, alpha_outs=(blk_s[10:11], blk_s[11:12])
    )
    r_part = compile_quantize(dtype_in=torch.bfloat16, h=h, d=d, scale_src="amax", n_alpha=2, amax_src="partials")
    assert r_part.amax_src == "partials" and r_part.publish is True and r_part.compiled is not r_slot.compiled
    blk_p = torch.full((16,), float("nan"), device="cuda")
    dst_p = torch.empty_like(dst_s)
    run_quantize(
        r_part,
        x,
        dst_p,
        None,
        stream=st,
        partials=partials,
        n_partials=n,
        amax_out=blk_p[0:1],
        scale_out=blk_p[4:5],
        descale=blk_p[5:6],
        alpha_consts=consts,
        alpha_outs=(blk_p[10:11], blk_p[11:12]),
    )
    torch.cuda.synchronize()
    assert torch.equal(dst_p.view(torch.uint8), dst_s.view(torch.uint8)), "the partials arm's bytes differ from the slot arm's"
    assert torch.equal(blk_p[0], slot[0]) and torch.equal(blk_p[4:6], blk_s[4:6]) and torch.equal(blk_p[10:12], blk_s[10:12])
    assert torch.isnan(blk_p[1:4]).all() and torch.isnan(blk_p[6:10]).all() and torch.isnan(blk_p[12:]).all(), "a slot outside the ABI was written"
    # the delayed form: the caller's scale, the amax still published from the partials
    r_given = compile_quantize(dtype_in=torch.bfloat16, h=h, d=d, scale_src="given", n_alpha=2, amax_src="partials")
    assert r_given.publish is True and r_given.compiled is not r_part.compiled
    blk_g = torch.full((16,), float("nan"), device="cuda")
    dst_g = torch.empty_like(dst_s)
    run_quantize(
        r_given,
        x,
        dst_g,
        blk_s[4:5].clone(),
        stream=st,
        partials=partials,
        n_partials=n,
        amax_out=blk_g[0:1],
        scale_out=blk_g[4:5],
        descale=blk_g[5:6],
        alpha_consts=consts,
        alpha_outs=(blk_g[10:11], blk_g[11:12]),
    )
    torch.cuda.synchronize()
    assert torch.equal(dst_g.view(torch.uint8), dst_s.view(torch.uint8)) and torch.equal(blk_g[0], slot[0]) and torch.equal(blk_g[4:6], blk_s[4:6])


@requires_fp8
def test_persistent_quantize_is_bitwise_the_one_block_per_group_grid():
    """``compile_quantize(persistent=True)`` strides a grid of ``min(row groups, SMs x AMAX_CTAS_PER_SM)`` CTAs over the row groups
    (``quantize_grid``) and writes the SAME e4m3 bytes and the SAME published slots as the default grid, under the slot arm and
    the partials arm (where it exists to pay the per-CTA reduce once per resident CTA); the default recipe keeps ``persistent=False``,
    ``n_ctas_cap == 0`` and its one-row-group-per-block grid; the knob is typed."""
    from cudnn.gated_attention_block.kernels.quantize import quantize_grid

    torch.manual_seed(6)
    t, h, d = 4000, 16, 256  # 4000 row groups > the 204-SM cap of 1632: every persistent CTA strides over several groups
    x = (torch.randn(t, h, d, device="cuda") * 20.0).to(torch.bfloat16)
    st = torch.cuda.current_stream().cuda_stream
    consts = (torch.tensor([0.5], device="cuda"), torch.tensor([1.0 / 3.0], device="cuda"))
    slot = torch.zeros(1, device="cuda")
    run_amax(compile_amax(dtype_in=torch.bfloat16, h=h, d=d), x, slot, stream=st)
    rp = compile_amax_partials(dtype_in=torch.bfloat16, h=h, d=d)
    partials = torch.empty(rp.n_ctas_cap, device="cuda")
    n = run_amax_partials(rp, x, partials, stream=st)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for amax_src in ("slot", "partials"):
        r0 = compile_quantize(dtype_in=torch.bfloat16, h=h, d=d, scale_src="amax", n_alpha=2, amax_src=amax_src)
        r1 = compile_quantize(dtype_in=torch.bfloat16, h=h, d=d, scale_src="amax", n_alpha=2, amax_src=amax_src, persistent=True)
        assert (r0.persistent, r0.n_ctas_cap) == (False, 0) and (r1.persistent, r1.n_ctas_cap) == (True, sms * AMAX_CTAS_PER_SM)
        assert r1.compiled is not r0.compiled
        groups = (t * h + r0.rows_per_cta - 1) // r0.rows_per_cta
        assert quantize_grid(r0, t) == groups and quantize_grid(r1, t) == min(groups, r1.n_ctas_cap) < groups
        outs = []
        for r in (r0, r1):
            blk = torch.full((16,), float("nan"), device="cuda")
            dst = torch.full((t, h, d), 0x7F, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
            kw = dict(stream=st, scale_out=blk[4:5], descale=blk[5:6], alpha_consts=consts, alpha_outs=(blk[10:11], blk[11:12]))
            if amax_src == "slot":
                run_quantize(r, x, dst, None, amax=slot, **kw)
            else:
                run_quantize(r, x, dst, None, partials=partials, n_partials=n, amax_out=blk[0:1], **kw)
            torch.cuda.synchronize()
            outs.append((dst, blk))
        (dst0, blk0), (dst1, blk1) = outs
        assert torch.equal(dst1.view(torch.uint8), dst0.view(torch.uint8)), f"{amax_src}: the persistent grid's bytes differ"
        assert not (dst1.view(torch.uint8) == 0x7F).any(), "a row group was skipped (the poison survived)"
        assert torch.equal(blk1[4:6], blk0[4:6]) and torch.equal(blk1[10:12], blk0[10:12])
        if amax_src == "partials":
            assert torch.equal(blk1[0], slot[0]) and torch.equal(blk0[0], slot[0])
    with pytest.raises(ValueError, match="persistent must be a bool"):
        compile_quantize(dtype_in=torch.bfloat16, h=h, d=d, persistent=1)


@requires_cuda
def test_cta_max_of_partials_pair_is_the_max_over_both_arrays():
    """The host-side contract of the pair reduce's inputs is the partials contract; the device reduce itself is pinned through the
    fused epilogue (test_fp8_bwd_fused.py) -- here the two helpers' bookkeeping: ``quantize_grid`` on a recipe without the knob."""
    from cudnn.gated_attention_block.kernels.quantize import QuantizeRecipe, quantize_grid

    r = QuantizeRecipe(compiled=None, h=4, d=256, rows_per_cta=16, dtype_in=torch.bfloat16)
    assert (r.persistent, r.n_ctas_cap) == (False, 0) and quantize_grid(r, 1000) == 250 and quantize_grid(r, 1) == 1
    rp = r._replace(persistent=True, n_ctas_cap=64)
    assert quantize_grid(rp, 1000) == 64 and quantize_grid(rp, 100) == 25 and quantize_grid(rp, 1) == 1


@requires_cuda
def test_quantize_partials_contract_is_typed():
    """Both directions against the recipe (Rule 1), before any launch: partials / n_partials / amax_out missing on a partials
    artifact, an amax slot passed to it, partials passed to a slot artifact or to a plain one, a published slot inside the
    partials, a too-short / strided / misaligned partials array."""
    x = torch.randn(64, 2, 256, device="cuda").to(torch.bfloat16)
    dst = torch.empty(64, 2, 256, dtype=torch.float8_e4m3fn, device="cuda")
    st = torch.cuda.current_stream().cuda_stream
    blk = torch.zeros(16, device="cuda", dtype=torch.float32)
    s = [blk[i : i + 1] for i in range(16)]
    parts = torch.zeros(64, device="cuda")
    part = _quant_recipe(scale_src="amax", n_alpha=0, publish=True, amax_src="partials")
    slot = _quant_recipe(scale_src="amax", n_alpha=0, publish=True)
    given = _quant_recipe()
    kw = dict(stream=st, scale_out=s[4], descale=s[5])
    with pytest.raises(ValueError, match="partials .*n_partials .*amax_out .*must all be bound"):
        run_quantize(part, x, dst, None, partials=parts, n_partials=8, **kw)
    with pytest.raises(ValueError, match="passing an amax slot would silently ignore it"):
        run_quantize(part, x, dst, None, amax=s[0], partials=parts, n_partials=8, amax_out=s[1], **kw)
    with pytest.raises(ValueError, match="passing partials / n_partials / amax_out would silently ignore"):
        run_quantize(slot, x, dst, None, amax=s[0], partials=parts, n_partials=8, **kw)
    with pytest.raises(ValueError, match="reads no amax .*would silently ignore"):
        run_quantize(given, x, dst, s[0], stream=st, partials=parts, n_partials=8)
    with pytest.raises(ValueError, match="n_partials must be an int in"):
        run_quantize(part, x, dst, None, partials=parts, n_partials=65, amax_out=s[1], **kw)
    with pytest.raises(ValueError, match="n_partials must be an int in"):
        run_quantize(part, x, dst, None, partials=parts, n_partials=0, amax_out=s[1], **kw)
    with pytest.raises(ValueError, match="contiguous fp32"):
        run_quantize(part, x, dst, None, partials=parts[::2], n_partials=8, amax_out=s[1], **kw)
    with pytest.raises(ValueError, match="16-byte-aligned base"):
        run_quantize(part, x, dst, None, partials=parts[1:], n_partials=8, amax_out=s[1], **kw)
    with pytest.raises(ValueError, match="lies inside the partials"):
        run_quantize(part, x, dst, None, partials=parts, n_partials=8, amax_out=parts[2:3], **kw)
    with pytest.raises(ValueError, match="amax_out and alpha_outs\\[0\\] are the same slot|scale_out aliases"):
        run_quantize(
            _quant_recipe(scale_src="amax", n_alpha=1, publish=True, amax_src="partials"),
            x,
            dst,
            None,
            partials=parts,
            n_partials=8,
            amax_out=s[6],
            alpha_consts=(s[7],),
            alpha_outs=(s[6],),
            **kw,
        )
    assert torch.equal(blk, torch.zeros_like(blk)) and torch.equal(parts, torch.zeros_like(parts)), "no launch happened"
