# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The block's per-tensor FP8 quantize pass (``kernels/quantize.py``).

Bit-exactness against torch's own saturating RNE cast is the contract: a
quantizer that rounds differently from the framework's reference would show up
as a numerics drift in every FP8 stage downstream, blamed on the wrong kernel.

The fp8 ``cvt`` needs sm_89+; on an A100 the numeric tests skip (the shape
algebra still runs).
"""

import math

import pytest
import torch

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block.kernels.quantize import (  # noqa: E402
    ELEMS_PER_LANE,
    FP8_E4M3_MAX,
    compile_quantize,
    lanes_per_row,
    moved_bytes,
    run_quantize,
    validate_shape,
)


def _fp8_cvt_available() -> bool:
    return torch.cuda.is_available() and tuple(torch.cuda.get_device_capability()) >= (8, 9)


requires_fp8 = pytest.mark.skipif(not _fp8_cvt_available(), reason="the fp8 cvt.rn.satfinite instruction needs sm_89+")

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
