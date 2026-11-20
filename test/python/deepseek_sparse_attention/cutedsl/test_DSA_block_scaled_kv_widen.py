# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The device widening of block-scaled KV codes (``sparse_attention_forward/_block_scaled_kv_widen.py``) against the torch reference.

Every code byte is widened with every scale byte, with each code byte at each of the four byte positions of a code word, and the
BF16 result is compared bit for bit with ``block_scaled_kv_reference.widen_*`` (NaN-ness only where the reference is NaN: the PTX
conversions define no NaN payload). One probe kernel per format: a lane widens one 32-bit code word twice, through each half of a
scale pair whose other half is another byte (``*_scale_pair`` with a run-time selector), with the scale byte at each byte position of
its word and, for NVFP4, in either word. The probe is compiled for the sparse attention forward's own target, so it emits the
instructions the forward kernel will.
"""

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("the block-scaled KV widening tests (sparse_attention_forward/_block_scaled_kv_widen.py)")
if requirement_error is not None:
    pytest.skip(requirement_error, allow_module_level=True)
if not all(hasattr(torch, name) for name in ("float8_e8m0fnu", "float4_e2m1fn_x2")):
    pytest.skip("torch build lacks float8_e8m0fnu / float4_e2m1fn_x2", allow_module_level=True)

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import make_fake_stream  # noqa: E402

from cudnn.deepseek_sparse_attention.sparse_attention_forward import _interface_sm100  # noqa: E402
from cudnn.deepseek_sparse_attention.sparse_attention_forward._block_scaled_kv_widen import (  # noqa: E402
    mxfp8_scale_pair,
    nvfp4_scale_pair,
    widen_mxfp8_word,
    widen_nvfp4_word,
)
from cudnn.frost.tile_dsl.tma import ld_global, st_global_v4  # noqa: E402
from deepseek_sparse_attention.cutedsl.block_scaled_kv_reference import widen_mxfp8, widen_nvfp4  # noqa: E402

pytestmark = pytest.mark.L0

_CC = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
requires_sm100 = pytest.mark.skipif(_CC is None or _CC[0] != 10, reason=f"the sparse attention forward targets SM100-family GPUs (found {_CC})")
_THREADS = 128


@cute.kernel
def _probe_mxfp8(mCodes: cute.Tensor, mScales: cute.Tensor, mOut: cute.Tensor, n: cutlass.Int32):
    """Lane ``l`` < ``n``: four E4M3 codes widened by the UE8M0 byte ``p = l % 4`` of its scale word, once as the low and once
    as the high half of a scale pair (the other half: byte ``(p + 1) % 4``) -> 2 x two ``bf16x2`` words."""
    lane = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(_THREADS) + cutlass.Int32(cute.arch.thread_idx()[0])
    if lane < n:
        lane64 = lane.to(cutlass.Int64)
        codes = ld_global(mCodes.iterator.toint() + lane64 * cutlass.Int64(4), cutlass.Int32)
        scale_word = ld_global(mScales.iterator.toint() + lane64 * cutlass.Int64(8), cutlass.Int32)
        p = lane % cutlass.Int32(4)
        other = (p + cutlass.Int32(1)) % cutlass.Int32(4)
        w0, w1 = widen_mxfp8_word(codes, mxfp8_scale_pair(scale_word, p + other * cutlass.Int32(16)), 0)
        w2, w3 = widen_mxfp8_word(codes, mxfp8_scale_pair(scale_word, other + p * cutlass.Int32(16)), 1)
        st_global_v4(mOut.iterator.toint() + lane64 * cutlass.Int64(16), (w0, w1, w2, w3), cutlass.Int32)


@cute.kernel
def _probe_nvfp4(mCodes: cute.Tensor, mScales: cute.Tensor, mOut: cute.Tensor, n: cutlass.Int32):
    """Lane ``l`` < ``n``: eight E2M1 codes widened by the E4M3 byte ``p = l % 4`` of its two scale words, once as the low half of
    a pair (picked from the first word; the high half: byte ``(p + 1) % 4`` of the second) and once as the high half (picked from
    the second word) -> 2 x four ``bf16x2`` words."""
    lane = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(_THREADS) + cutlass.Int32(cute.arch.thread_idx()[0])
    if lane < n:
        lane64 = lane.to(cutlass.Int64)
        codes = ld_global(mCodes.iterator.toint() + lane64 * cutlass.Int64(4), cutlass.Int32)
        word_a = ld_global(mScales.iterator.toint() + lane64 * cutlass.Int64(8), cutlass.Int32)
        word_b = ld_global(mScales.iterator.toint() + lane64 * cutlass.Int64(8) + cutlass.Int64(4), cutlass.Int32)
        p = lane % cutlass.Int32(4)
        other = (p + cutlass.Int32(1)) % cutlass.Int32(4)
        out = mOut.iterator.toint() + lane64 * cutlass.Int64(32)
        low = widen_nvfp4_word(codes, nvfp4_scale_pair(word_a, word_b, p + (other + cutlass.Int32(4)) * cutlass.Int32(16)), 0)
        st_global_v4(out, low, cutlass.Int32)
        high = widen_nvfp4_word(codes, nvfp4_scale_pair(word_a, word_b, other + (p + cutlass.Int32(4)) * cutlass.Int32(16)), 1)
        st_global_v4(out + cutlass.Int64(16), high, cutlass.Int32)


_probe_mxfp8.set_name_prefix("cudnn", remove_cutlass_symbol=True)
_probe_nvfp4.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _launch_mxfp8(codes: cute.Tensor, scales: cute.Tensor, out: cute.Tensor, n: cutlass.Int32, n_blocks: cutlass.Int32, stream: cuda.CUstream):
    _probe_mxfp8(codes, scales, out, n).launch(grid=(n_blocks, 1, 1), block=(_THREADS, 1, 1), stream=stream)


@cute.jit
def _launch_nvfp4(codes: cute.Tensor, scales: cute.Tensor, out: cute.Tensor, n: cutlass.Int32, n_blocks: cutlass.Int32, stream: cuda.CUstream):
    _probe_nvfp4(codes, scales, out, n).launch(grid=(n_blocks, 1, 1), block=(_THREADS, 1, 1), stream=stream)


def _fake_i32():
    return cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=16)


def _run(launch, code_rows: torch.Tensor, scale_bytes: torch.Tensor, out_words_per_lane: int):
    """Widen ``code_rows`` (``uint8 [R, 256]``, one scale byte per row) on the device; returns the BF16 outputs ``[R, 256 *
    channels/byte]`` through the low and the high half of the scale pair."""
    rows = code_rows.shape[0]
    codes = code_rows.cuda().contiguous().view(torch.int32).view(-1)
    lanes = codes.numel()
    # Two scale words per lane; lane l's scale byte sits at byte l % 4 of both, the other bytes carry a varying pattern.
    lane = torch.arange(lanes, dtype=torch.int64)
    position = 8 * (lane % 4)
    pattern = torch.stack([(lane * 2654435761 + k * 40503) & 0xFFFFFFFF for k in range(2)], dim=1)
    scale = scale_bytes.to(torch.int64).repeat_interleave(lanes // rows)
    scale_words = (pattern & ~(0xFF << position)[:, None]) | (scale << position)[:, None]
    scales = scale_words.to(torch.uint32).view(torch.int32).cuda().contiguous().view(-1)
    out = torch.full((lanes * 2 * out_words_per_lane,), 0x7FFF7FFF, dtype=torch.int32, device="cuda")
    arch = _interface_sm100._gpu_arch_flag(torch.device("cuda"))
    fn = cute.compile(
        launch,
        _fake_i32(),
        _fake_i32(),
        _fake_i32(),
        cutlass.Int32(0),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
    )
    n_blocks = -(-lanes // _THREADS)
    fn(codes, scales, out, cutlass.Int32(lanes), cutlass.Int32(n_blocks), cuda.CUstream(int(torch.cuda.current_stream().cuda_stream)))
    torch.cuda.synchronize()
    both = out.view(lanes, 2, out_words_per_lane).cpu()
    return tuple(both[:, half].contiguous().view(torch.bfloat16).view(rows, -1) for half in range(2))


def _every_byte_at_every_position():
    """``uint8 [4 * 256, 256]``: row ``(r, s)`` lists the 256 byte values rotated by ``r`` (so each value sits at each byte position of
    a 32-bit word once), and is widened with scale byte ``s``."""
    rotations = torch.stack([torch.roll(torch.arange(256), -r) for r in range(4)]).to(torch.uint8)  # [4, 256]
    code_rows = rotations[:, None, :].expand(4, 256, 256).reshape(1024, 256)
    scale_bytes = torch.arange(256).repeat(4).to(torch.uint8)
    return code_rows.contiguous(), scale_bytes


def _assert_bitwise(got: torch.Tensor, want: torch.Tensor, label: str):
    nan = want.isnan()
    assert bool(got[nan].isnan().all()), f"{label}: a NaN code or scale widened to a number"
    same = got.view(torch.int16) == want.view(torch.int16)
    bad = ~(same | nan)
    assert not bool(bad.any()), (
        f"{label}: {int(bad.sum())} of {want.numel()} values differ; first (row, channel): {bad.nonzero()[:4].tolist()}, "
        f"got {got[bad][:4].tolist()}, want {want[bad][:4].tolist()}"
    )


@requires_sm100
def test_mxfp8_word_widening_is_bitwise_the_reference_for_every_code_and_scale_byte():
    code_rows, scale_bytes = _every_byte_at_every_position()
    kv_sf = scale_bytes.view(torch.float8_e8m0fnu)[:, None].repeat(1, 256 // 32)
    want = widen_mxfp8(code_rows.view(torch.float8_e4m3fn), kv_sf)
    for half, got in enumerate(_run(_launch_mxfp8, code_rows, scale_bytes, out_words_per_lane=2)):
        assert got.shape == want.shape == (1024, 256)
        _assert_bitwise(got, want, f"MXFP8 scale-pair half {half}")
    assert bool(torch.isinf(want).any()) and bool((want == 0).any()), "the table covers the overflow and the zero results"


@requires_sm100
def test_nvfp4_word_widening_is_bitwise_the_reference_for_every_code_and_scale_byte():
    code_rows, scale_bytes = _every_byte_at_every_position()
    kv_sf = scale_bytes.view(torch.float8_e4m3fn)[:, None].repeat(1, 512 // 16)
    want = widen_nvfp4(code_rows.view(torch.float4_e2m1fn_x2), kv_sf)
    for half, got in enumerate(_run(_launch_nvfp4, code_rows, scale_bytes, out_words_per_lane=4)):
        assert got.shape == want.shape == (1024, 512)
        _assert_bitwise(got, want, f"NVFP4 scale-pair half {half}")
        # Signed zeros survive: -0 codes, and +0 codes under negative scales.
        zero = want == 0
        assert torch.equal(torch.signbit(got[zero]), torch.signbit(want[zero])) and bool(torch.signbit(want[zero]).any())
