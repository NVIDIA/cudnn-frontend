# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tile_dsl.pointwise.bf16_words_to_e4m3_words`` -- the e4m3 rounding OF A PACKED bf16 ROW: four ``bf16x2`` words (8 elements) into two
``e4m3x4`` words, each bf16 widened to fp32 EXACTLY (a bit move) and rounded ONCE by ``cvt.rn.satfinite.e4m3x2.f32``, element order preserved.
The contract: the two words are bitwise ``bf16_row.to(torch.float8_e4m3fn)`` -- the convention of the indexer's e4m3 compressed-key cache
(the serving stacks' ``x.to(bf16).to(e4m3)`` chain and the prefill form's per-tensor cast of the bf16 rows) -- incl. the saturation band
(``|x| > 448`` and ``+-inf`` -> ``+-448``: ``satfinite``; torch's cast is clamped to the same bound first so the oracle stays right on a torch
whose cast overflows to NaN), the exact midpoints (round-to-nearest-even), subnormals and signed zeros.  The fp8 ``cvt`` is an sm_89+
instruction: the device cell skips below it (UNCOVERED there -- the callers decline older parts by name at compile time).

One probe kernel in this module: lane = one 8-element bf16 chunk (one ``ld.global.v4`` of the packed words, the helper, one ``st.global.v2``)."""

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("the bf16 -> e4m3 words helper's tests (tile_dsl/pointwise.py)")
if requirement_error is not None:
    pytest.skip(requirement_error, allow_module_level=True)

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import make_fake_stream  # noqa: E402

from cudnn.frost.tile_dsl.pointwise import bf16_words_to_e4m3_words  # noqa: E402
from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global_v2  # noqa: E402

pytestmark = pytest.mark.L0

_CC = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
requires_sm89 = pytest.mark.skipif(_CC is None or _CC < (8, 9), reason=f"cvt.rn.satfinite.e4m3x2.f32 needs sm_89+ (found {_CC})")
E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0


@cute.kernel
def _probe(mSrc: cute.Tensor, mDst: cute.Tensor, n: cutlass.Int32):
    """Lane ``l`` < ``n``: the four packed bf16 words of row ``l`` (16 B) -> the two e4m3 words (8 B) of row ``l``."""
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(128) + tidx
    if lane < n:
        lane64 = lane.to(cutlass.Int64)
        w0, w1, w2, w3 = ld_global_v4(mSrc.iterator.toint() + lane64 * cutlass.Int64(16), cutlass.Int32)
        u0, u1 = bf16_words_to_e4m3_words(w0, w1, w2, w3)
        st_global_v2(mDst.iterator.toint() + lane64 * cutlass.Int64(8), (u0, u1), cutlass.Int32)


_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _launch(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32, n_blocks: cutlass.Int32, stream: cuda.CUstream):
    _probe(src, dst, n).launch(grid=(n_blocks, 1, 1), block=(128, 1, 1), stream=stream)


def _fake_i32():
    return cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=16)


def _rows():
    """The probe rows ``[N, 8]`` bf16: random N(0, 1) and N(0, 100), the saturation band (|x| in (448, 4e4]), +-inf, exact e4m3
    midpoints (round-to-nearest-even both ways), e4m3 subnormals and the values below them, signed zeros, the largest bf16 below 448."""
    g = torch.Generator(device="cuda").manual_seed(11)
    rows = [
        torch.randn(2048, 8, generator=g, device="cuda"),
        torch.randn(2048, 8, generator=g, device="cuda") * 100.0,
        (torch.rand(512, 8, generator=g, device="cuda") * 4e4 + 449.0) * torch.where(torch.rand(512, 8, generator=g, device="cuda") < 0.5, -1.0, 1.0),
        torch.tensor([[float("inf"), float("-inf"), 448.0, -448.0, 464.0, -464.0, 480.0, 1e30]], device="cuda"),
        torch.tensor(
            [[0.0, -0.0, 2.0**-9, -(2.0**-9), 2.0**-10, 3 * 2.0**-10, 2.0**-6, 1.0 + 2.0**-4]], device="cuda"
        ),  # subnormals / below / the smallest normal
        torch.tensor([[1.0625, 1.1875, 1.3125, 1.4375, 2.125, 2.375, 3.0625, 3.1875]], device="cuda"),  # exact midpoints between e4m3 neighbours
        torch.tensor([[440.0, -440.0, 416.0, 432.0, 0.5, -0.5, 7.5, 15.5]], device="cuda"),
    ]
    return torch.cat(rows, 0).to(torch.bfloat16).contiguous()


def _reference(src16: torch.Tensor) -> torch.Tensor:
    """``bf16_row.to(e4m3)`` with the saturation band clamped to +-448 first (``satfinite``'s result; torch's cast agrees inside the range)."""
    return src16.float().clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)


@requires_sm89
def test_bf16_words_to_e4m3_words_is_bitwise_the_torch_cast_incl_the_saturation_band():
    src16 = _rows()
    n = int(src16.shape[0])
    assert src16.shape == (n, 8)
    src = src16.view(torch.int32).contiguous()  # [N, 4]
    dst = torch.full((n, 8), 0x7D, dtype=torch.uint8, device="cuda").view(torch.int32).contiguous()  # [N, 2]; the fill = a NaN-free sentinel
    fn = cute.compile(
        _launch, _fake_i32(), _fake_i32(), cutlass.Int32(0), cutlass.Int32(1), make_fake_stream(use_tvm_ffi_env_stream=False), options="--enable-tvm-ffi"
    )
    n_blocks = -(-n // 128)
    for _ in range(2):  # two launches: bitwise
        fn(src.view(-1), dst.view(-1), cutlass.Int32(n), cutlass.Int32(n_blocks), cuda.CUstream(int(torch.cuda.current_stream().cuda_stream)))
        torch.cuda.synchronize()
        got = dst.view(torch.uint8).view(n, 8)
        want = _reference(src16).view(torch.uint8)
        bad = got != want
        assert not bool(bad.any()), (
            f"{int(bad.sum())} of {got.numel()} bytes differ from bf16_row.to(float8_e4m3fn); first rows: "
            f"{src16[bad.any(-1)][:4].float().tolist()} -> got {got.view(E4M3)[bad.any(-1)][:4].float().tolist()}, want {want.view(E4M3)[bad.any(-1)][:4].float().tolist()}"
        )
    got8 = got.view(E4M3).float()
    assert bool((got8[src16.float() > E4M3_MAX] == E4M3_MAX).all()) and bool((got8[src16.float() < -E4M3_MAX] == -E4M3_MAX).all()), "the band does not saturate"
    assert not torch.isnan(got8).any(), "a finite (or infinite) input rounded to NaN"
    assert float(got8[0, 0]) == float(src16[0, 0].float().to(E4M3).float())  # the plain case, as a smoke of the comparison itself
    print(f"bf16 -> e4m3 words: {n} rows x 8 elements bitwise the torch cast (saturation band, midpoints, subnormals included)")


def test_helper_is_the_documented_chain_and_the_callers_import_it():
    """Static: the helper's PTX widens bf16 by a bit move (``mov.b32 f, {0, l}``), rounds with ``cvt.rn.satfinite.e4m3x2.f32`` -- never
    ``cvt.rn.f32.bf16`` and never a second rounding -- and packs two ``.b16`` halves; the three indexer store sites (the compress kernels,
    the fused prepare) reach it through ONE shared store helper."""
    import inspect

    from cudnn.gated_attention_block.kernels import qsa_compress, qsa_prepare

    src = inspect.getsource(bf16_words_to_e4m3_words)
    ptx = src[src.index("inline_ptx(") :]  # the PTX string (the docstring names the instruction once more)
    assert ptx.count("cvt.rn.satfinite.e4m3x2.f32 e") == 4 and ptx.count("mov.b32 f0, {0, lo0}") == 2 and "cvt.rn.f32.bf16" not in src
    assert hasattr(qsa_compress, "_store_row_chunk_e4m3") and qsa_prepare._store_row_chunk_e4m3 is qsa_compress._store_row_chunk_e4m3
    ksrc = inspect.getsource(qsa_compress._store_row_chunk_e4m3)
    assert "bf16_words_to_e4m3_words" in ksrc and "st_global_v2" in ksrc
