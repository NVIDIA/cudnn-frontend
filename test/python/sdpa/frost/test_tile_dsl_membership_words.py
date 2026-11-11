# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tile_dsl.mask.apply_membership_words`` / ``apply_membership_chunk`` -- the per-cell MEMBERSHIP mask of a list-gathered
KV tile: cell ``c`` keeps ``S[c]`` iff bit ``c % 32`` of word ``c // 32`` is set, else becomes ``-inf`` (a true ``-inf``, so a
fully-masked chunk's max is ``-inf`` and the consumer's empty-row substitute applies).

Pinned against a torch ``where`` on random words -- including the all-zero word (nothing kept), the all-one word
(everything kept, kept cells bit-exact), a single-cell word, bit 31 of an ``Int32`` word (the signed encoding
``-2**31``), a NaN payload passing through a kept cell -- for the 64- and 32-wide two-word form and the 128-wide
four-word form.  Pure ALU: runs on ANY CUDA device the DSL targets (the trace-time contract tests need no device).
"""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_sm80

pytestmark = [pytest.mark.L0, requires_dsl]


def _stream():
    """The current torch stream as a ``CUstream`` for the compiled probe."""
    import cuda.bindings.driver as cuda

    return cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))


def _probe():
    """The probe kernels, built lazily (the cutlass imports must not run at collection on a box without the DSL); the
    kernel bodies live in this module so the DSL preprocessor can read their source."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    from cudnn.frost.tile_dsl.mask import apply_membership_chunk, apply_membership_words

    @cute.kernel
    def membership_kernel(scores: cute.Tensor, words: cute.Tensor, out: cute.Tensor, N: cutlass.Constexpr[int], NW: cutlass.Constexpr[int]):
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])
        row = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(128) + tidx
        elems = []
        for i in cutlass.range_constexpr(N):
            elems.append(scores[row * N + i])
        vec = cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)
        if cutlass.const_expr(NW <= 2):
            res = apply_membership_chunk(vec, words[row * NW], words[row * NW + 1] if cutlass.const_expr(NW == 2) else words[row * NW], n=N)
        else:
            ws = []
            for s in cutlass.range_constexpr(NW):
                ws.append(words[row * NW + s])
            res = apply_membership_words(vec, tuple(ws), n_cols=N)
        for i in cutlass.range_constexpr(N):
            out[row * N + i] = res[i]

    @cute.jit
    def membership_host(scores, words, out, n_blocks, N: cutlass.Constexpr[int], NW: cutlass.Constexpr[int], stream: cuda.CUstream):
        membership_kernel(scores, words, out, N, NW).launch(grid=(n_blocks, 1, 1), block=(128, 1, 1), stream=stream)

    def fake_1d(dtype):
        """A fake 1-D tensor of ``dtype`` (symbolic length, 16-B aligned) for ``cute.compile``."""
        return make_fake_tensor(dtype, (cute.sym_int(),), (1,), assumed_align=16)

    def compile_membership(N: int, NW: int):
        """Compile the probe for an ``N``-wide chunk over ``NW`` words."""
        return cute.compile(
            membership_host,
            fake_1d(cutlass.Float32),
            fake_1d(cutlass.Int32),
            fake_1d(cutlass.Float32),
            cutlass.Int32(0),
            N,
            NW,
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    return compile_membership


# ============================================================================ trace-time contracts (no device)
def test_apply_membership_chunk_rejects_a_chunk_wider_than_two_words():
    """``apply_membership_chunk`` serves 1..64 columns (two words); a wider chunk is refused by value."""
    from cudnn.frost.tile_dsl.mask import apply_membership_chunk

    with pytest.raises(ValueError, match="n must be in 1..64"):
        apply_membership_chunk(None, None, None, n=65)


def test_apply_membership_words_rejects_too_few_words_for_the_chunk():
    """``apply_membership_words`` refuses a word list that covers fewer columns than the chunk."""
    from cudnn.frost.tile_dsl.mask import apply_membership_words

    with pytest.raises(ValueError, match="2 word\\(s\\) cover 64 columns, chunk has 96"):
        apply_membership_words(None, (None, None), n_cols=96)


# ============================================================================ numerics (any CUDA device)
def _reference(scores: torch.Tensor, words: torch.Tensor, N: int) -> torch.Tensor:
    """torch twin of the membership mask: keep ``scores[c]`` iff bit ``c % 32`` of word ``c // 32`` is set, else ``-inf``."""
    bits = torch.arange(N, device=scores.device)
    w = words.to(torch.int64) & 0xFFFFFFFF  # [rows, NW] as unsigned
    sel = w[:, bits // 32]  # [rows, N]
    keep = ((sel >> (bits % 32)) & 1) == 1
    return torch.where(keep, scores, torch.full_like(scores, float("-inf")))


@requires_sm80
@pytest.mark.parametrize("N, NW", [(64, 2), (32, 1), (128, 4)], ids=["chunk64", "chunk32", "words128"])
def test_apply_membership_matches_torch(N, NW):
    """Cell ``i`` keeps ``S[i]`` iff bit ``i`` of the word table is set, else ``-inf``; planted rows: nothing kept, everything
    kept, cell 0 only, the top cell only (bit 31 of the last word, signed), a NaN payload in a kept cell."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    n_blocks, rows = 3, 3 * 128
    scores = torch.randn(rows, N, device=dev, dtype=torch.float32)
    words = torch.randint(-(2**31), 2**31 - 1, (rows, NW), device=dev, dtype=torch.int64).to(torch.int32)
    words[0] = 0  # nothing kept
    words[1] = -1  # everything kept (all ones)
    words[2] = 0
    words[2, 0] = 1  # only cell 0
    words[3] = 0
    words[3, NW - 1] = -(2**31)  # only cell N - 1 (bit 31 of the last word, the signed encoding)
    words[4] = -1
    scores[4, 5] = float("nan")  # a kept NaN passes through bit-exactly
    out = torch.full_like(scores, 12345.0)
    compile_membership = _probe()
    art = compile_membership(N, NW)
    import cutlass

    art(scores.reshape(-1), words.reshape(-1), out.reshape(-1), cutlass.Int32(n_blocks), _stream())
    torch.cuda.synchronize()
    want = _reference(scores, words, N)
    assert torch.equal(out.view(torch.int32), want.view(torch.int32)), "bitwise, NaN payloads compared as bit patterns"
    assert bool(torch.isinf(out[0]).all()) and bool((out[0] < 0).all())
    assert torch.equal(out[1].view(torch.int32), scores[1].view(torch.int32))
    assert out[2, 0] == scores[2, 0] and bool(torch.isinf(out[2, 1:]).all())
    assert out[3, N - 1] == scores[3, N - 1] and bool(torch.isinf(out[3, : N - 1]).all())
    assert bool(torch.isnan(out[4, 5]))
