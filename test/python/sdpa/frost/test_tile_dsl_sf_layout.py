# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pins ``tile_dsl.sf_layout`` -- the ONE spelling of cuDNN's F8_128x4 atom byte rule that the
block-scale quantize kernels address scale-factor bytes with -- against the torch oracle
(``test/python/sdpa/mxfp8_quant.py::_swizzle_128x4``) on host ints, against the MXFP8 quantizer's
host twins, and on TRACED ``cutlass.Int32`` through a probe kernel (any SM80+ device: it is
integer arithmetic, so the A100 box that trace-compiles Rubin kernels also runs this)."""

import os
import sys

import pytest
import torch

from cudnn.frost.tile_dsl.sf_layout import SF_ATOM_BYTES, SF_ATOM_COLS, SF_ATOM_LINE_BYTES, SF_ATOM_LINE_ROWS, SF_ATOM_ROWS, sf_atom_byte, sf_atom_offset
from frost_test_utils import _dsl_installed, requires_dsl

pytestmark = pytest.mark.L0

try:
    from sdpa.mxfp8_quant import _swizzle_128x4  # importable when pytest runs from test/python (how the SDPA suites spell it)
except ImportError:  # a standalone driver run from elsewhere gets the test/python root added
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    from sdpa.mxfp8_quant import _swizzle_128x4

if _dsl_installed():
    import cuda.bindings.driver as _cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.base_dsl.typing import Pointer
    from cutlass.cute.runtime import from_dlpack

    @cute.kernel
    def _probe_kernel(out_t: cute.Tensor, n_cols: cutlass.Constexpr[int]):
        """One thread per (r, c) of a 128-row band with ``n_cols`` scale columns: out[r*n_cols + c] = (sf_atom_offset, sf_atom_byte)."""
        bx, _, _ = cute.arch.block_idx()  # r
        tidx, _, _ = cute.arch.thread_idx()  # c
        r = cutlass.Int32(bx)
        c = cutlass.Int32(tidx)
        base = Pointer(out_t.iterator.raw_ptr(), dtype=cutlass.Int32) + (r * cutlass.Int32(n_cols) + c) * cutlass.Int32(2)
        base.store(sf_atom_offset(r, c), alignment=4)
        (base + cutlass.Int32(1)).store(sf_atom_byte(r % cutlass.Int32(SF_ATOM_LINE_ROWS * 4), c % cutlass.Int32(SF_ATOM_COLS)), alignment=4)

    _probe_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def _probe_host(out_t, n_cols: cutlass.Constexpr[int], stream):
        _probe_kernel(out_t, n_cols).launch(grid=(SF_ATOM_ROWS, 1, 1), block=(n_cols, 1, 1), stream=stream)


def test_atom_geometry_constants():
    assert (SF_ATOM_ROWS, SF_ATOM_COLS, SF_ATOM_BYTES) == (128, 4, 512)
    assert SF_ATOM_LINE_ROWS * SF_ATOM_LINE_BYTES == SF_ATOM_BYTES and SF_ATOM_LINE_BYTES == 16 and SF_ATOM_LINE_ROWS == 32


def test_atom_byte_is_a_bijection_on_one_atom():
    """Every (r, c4) of a 128 x 4 atom lands on a distinct byte in [0, 512): the rule is a permutation, not a hash."""
    seen = sorted(sf_atom_byte(r, c4) for r in range(SF_ATOM_ROWS) for c4 in range(SF_ATOM_COLS))
    assert seen == list(range(SF_ATOM_BYTES))


def test_atom_offset_is_atom_byte_at_the_atom_base():
    """``sf_atom_offset(r, c)`` == the atom-local byte of ``(r, c % 4)`` at base ``(c // 4) * 512``, on every (r, c) of a wide row."""
    for r in range(SF_ATOM_ROWS):
        for c in range(64):  # 16 atoms wide (D=2048 at block 32)
            assert sf_atom_offset(r, c) == (c // SF_ATOM_COLS) * SF_ATOM_BYTES + sf_atom_byte(r, c % SF_ATOM_COLS), (r, c)


@pytest.mark.parametrize("rows, cols", [(128, 4), (128, 8), (384, 8), (256, 32), (128, 64)])
def test_atom_offset_matches_the_oracle_swizzle(rows, cols):
    """The oracle lays atoms out row-major over ``(rows/128) x (cols/4)``; within a 128-row band the helper is the byte address."""
    g = torch.Generator().manual_seed(0)
    m = torch.randint(0, 256, (rows, cols), dtype=torch.uint8, generator=g)
    sw = _swizzle_128x4(m).reshape(-1)
    band_bytes = SF_ATOM_ROWS * cols
    for r in range(rows):
        for c in range(cols):
            off = (r // SF_ATOM_ROWS) * band_bytes + sf_atom_offset(r % SF_ATOM_ROWS, c)
            assert int(sw[off]) == int(m[r, c]), (r, c)


def test_mxfp8_host_twins_reduce_to_the_shared_helper():
    """The quantizer's ``sf_byte_rowwise`` / ``sf_byte_columnwise`` are the helper plus a per-unit atom base -- pinned here so a
    future edit to either cannot fork the spelling silently."""
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import SF_BLOCK, sf_byte_columnwise, sf_byte_rowwise, sf_tile_bytes

    d, n_heads, n_tiles, batch = 256, 3, 5, 2
    g = torch.Generator().manual_seed(1)
    for _ in range(3000):
        b, h = int(torch.randint(0, batch, (1,), generator=g)), int(torch.randint(0, n_heads, (1,), generator=g))
        s, d_idx = int(torch.randint(0, n_tiles * SF_ATOM_ROWS, (1,), generator=g)), int(torch.randint(0, d, (1,), generator=g))
        tile = (b * n_heads + h) * n_tiles + s // SF_ATOM_ROWS
        assert sf_byte_rowwise(b, h, s, d_idx, n_heads=n_heads, n_tiles=n_tiles, d=d) == tile * sf_tile_bytes(d) + sf_atom_offset(
            s % SF_ATOM_ROWS, d_idx // SF_BLOCK
        )
        plane, dm = d_idx // SF_ATOM_ROWS, d_idx % SF_ATOM_ROWS
        expect = sf_atom_byte(dm, (s % SF_ATOM_ROWS) // SF_BLOCK, base=plane * (batch * n_heads * n_tiles * SF_ATOM_BYTES) + tile * SF_ATOM_BYTES)
        assert sf_byte_columnwise(b, h, s, d_idx, n_heads=n_heads, n_tiles=n_tiles, batch=batch) == expect


@requires_dsl
@pytest.mark.parametrize("n_cols", [4, 8, 16])
def test_helper_traces_on_int32_and_matches_the_host(n_cols):
    """The same plain-Python body on traced ``cutlass.Int32`` (Python-int constants folded by the DSL) equals the host evaluation."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to launch the probe")
    dev = torch.device("cuda")
    out = torch.full((SF_ATOM_ROWS * n_cols * 2,), -7, dtype=torch.int32, device=dev)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    cute.compile(_probe_host, from_dlpack(out, assumed_align=16), n_cols, stream)(from_dlpack(out, assumed_align=16), stream)
    torch.cuda.synchronize()
    got = out.cpu().view(SF_ATOM_ROWS, n_cols, 2)
    for r in range(SF_ATOM_ROWS):
        for c in range(n_cols):
            assert int(got[r, c, 0]) == sf_atom_offset(r, c), (r, c)
            assert int(got[r, c, 1]) == sf_atom_byte(r, c % SF_ATOM_COLS), (r, c)
