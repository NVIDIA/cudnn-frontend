# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pins ``tile_dsl.mask.compute_q_loop_bounds`` / ``swa_kv_lo_tile`` (the SM80
backward's Q-loop bounds and deterministic-relay origin) against a host replica
of the arithmetic they replaced, over every mask family, alignment and the
bottom-right / right-band variants.  A probe kernel evaluates the helpers for
every kv tile; the reference is plain Python on the same inputs."""

import pytest
import torch

from cudnn.frost.tile_dsl.constants import MASK_CAUSAL, MASK_NONE, MASK_SWA
from frost_test_utils import _dsl_installed, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

if _dsl_installed():
    import cuda.bindings.driver as _cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.base_dsl.typing import Pointer
    from cutlass.cute.runtime import from_dlpack

    from cudnn.frost.tile_dsl.mask import compute_q_loop_bounds, swa_kv_lo_tile

    @cute.kernel
    def _probe_kernel(
        out_t: cute.Tensor,
        seqlen_q: cutlass.Int32,
        seq_kv_len: cutlass.Int32,
        n_q_tiles: cutlass.Int32,
        window_right: cutlass.Int32,
        mask_flags: cutlass.Constexpr[int],
        window_left: cutlass.Constexpr[int],
        tile_q: cutlass.Constexpr[int],
        tile_kv: cutlass.Constexpr[int],
        bottom_right: cutlass.Constexpr[bool],
    ):
        bx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == cutlass.Int32(0):
            kv_base = bx * cutlass.Int32(tile_kv)
            b = compute_q_loop_bounds(kv_base, seqlen_q, seq_kv_len, n_q_tiles, window_left, mask_flags, tile_q, tile_kv, bottom_right, window_right)
            # Relay origin for the q tile with the same index as this kv tile
            # (any q row works; this keeps the probe one launch).
            anchor = bx * cutlass.Int32(tile_q)
            if cutlass.const_expr(bottom_right):
                anchor = anchor + (seq_kv_len - seqlen_q)
            kv_first = swa_kv_lo_tile(anchor, window_left, tile_kv)
            base = Pointer(out_t.iterator.raw_ptr(), dtype=cutlass.Int32) + bx * cutlass.Int32(3)
            base.store(b.lo, alignment=4)
            (base + cutlass.Int32(1)).store(b.hi, alignment=4)
            (base + cutlass.Int32(2)).store(kv_first, alignment=4)

    _probe_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def _probe_host(
        out_t,
        seqlen_q,
        seq_kv_len,
        n_q_tiles,
        window_right,
        n_kv,
        mask_flags: cutlass.Constexpr[int],
        window_left: cutlass.Constexpr[int],
        tile_q: cutlass.Constexpr[int],
        tile_kv: cutlass.Constexpr[int],
        bottom_right: cutlass.Constexpr[bool],
        stream,
    ):
        _probe_kernel(out_t, seqlen_q, seq_kv_len, n_q_tiles, window_right, mask_flags, window_left, tile_q, tile_kv, bottom_right).launch(
            grid=(n_kv, 1, 1), block=(32, 1, 1), stream=stream
        )


def _ref_bounds(kv_base, *, s_q, s_kv, n_q_tiles, window_left, mask_flags, tile_q, tile_kv, bottom_right, window_right):
    """The SM80 backward's original inline arithmetic (PR #866 form)."""
    diag = (s_kv - s_q) if bottom_right else 0
    lo = 0
    if mask_flags & MASK_CAUSAL:
        lo = max(((kv_base - diag) if bottom_right else kv_base) - window_right, -(10**9)) // tile_q
        lo = max(lo, 0)
    n_iters = max(n_q_tiles - lo, 0)
    if mask_flags & MASK_SWA:
        q_hi_abs = kv_base + tile_kv + window_left
        if bottom_right:
            q_hi_abs -= diag
        q_hi_abs = max(q_hi_abs, 0)
        q_hi_t = min(-(-q_hi_abs // tile_q), n_q_tiles)
        n_iters = max(q_hi_t - lo, 0)
    return lo, n_iters


def _ref_kv_first(q_row0, *, window_left, tile_kv):
    return max((q_row0 - window_left) // tile_kv, 0)


_CASES = [
    # (mask_flags, window_left, bottom_right, window_right, tile_q, tile_kv, s_q, s_kv)
    (MASK_NONE, 0, False, 0, 64, 64, 512, 512),
    (MASK_CAUSAL, 0, False, 0, 64, 64, 512, 512),
    (MASK_CAUSAL, 0, False, 40, 64, 64, 448, 640),  # band widening, unaligned q
    (MASK_CAUSAL, 0, True, 0, 64, 64, 512, 768),  # bottom-right, S_kv > S_q
    (MASK_CAUSAL, 0, True, 0, 64, 64, 768, 512),  # bottom-right, S_kv < S_q (negative diagonal)
    (MASK_CAUSAL | MASK_SWA, 128, False, 0, 64, 64, 1024, 1024),
    (MASK_CAUSAL | MASK_SWA, 128, True, 0, 64, 64, 1024, 1536),
    (MASK_CAUSAL | MASK_SWA, 96, True, 16, 128, 64, 1152, 1408),  # gptoss tile, odd window, band
    (MASK_SWA, 200, False, 0, 64, 64, 704, 704),  # window without causal
]


@pytest.mark.parametrize("case", _CASES, ids=[f"m{c[0]}-W{c[1]}-br{int(c[2])}-R{c[3]}-tq{c[4]}-{c[6]}x{c[7]}" for c in _CASES])
def test_q_loop_bounds_match_reference(case):
    mask_flags, window_left, bottom_right, window_right, tile_q, tile_kv, s_q, s_kv = case
    dev = torch.device("cuda")
    n_kv = -(-s_kv // tile_kv)
    n_q_tiles = -(-s_q // tile_q)
    out = torch.full((n_kv * 3,), -7, dtype=torch.int32, device=dev)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    args = (
        from_dlpack(out, assumed_align=16),
        cutlass.Int32(s_q),
        cutlass.Int32(s_kv),
        cutlass.Int32(n_q_tiles),
        cutlass.Int32(window_right),
        cutlass.Int32(n_kv),
        int(mask_flags),
        int(window_left),
        int(tile_q),
        int(tile_kv),
        bool(bottom_right),
        stream,
    )
    # The compiled host takes only the runtime arguments; the Constexpr ones are
    # baked in at compile time.
    runtime = tuple(a for a in args if not isinstance(a, (int, bool)) or isinstance(a, cutlass.Int32))
    cute.compile(_probe_host, *args)(*runtime)
    torch.cuda.synchronize()
    got = out.cpu().tolist()
    for kv_tile in range(n_kv):
        lo, n_iters = _ref_bounds(
            kv_tile * tile_kv,
            s_q=s_q,
            s_kv=s_kv,
            n_q_tiles=n_q_tiles,
            window_left=window_left,
            mask_flags=mask_flags,
            tile_q=tile_q,
            tile_kv=tile_kv,
            bottom_right=bottom_right,
            window_right=window_right,
        )
        g_lo, g_hi, g_first = got[kv_tile * 3 : kv_tile * 3 + 3]
        assert g_lo == lo, (kv_tile, g_lo, lo)
        assert max(g_hi - g_lo, 0) == n_iters, (kv_tile, g_hi, g_lo, n_iters)
        q_row0 = kv_tile * tile_q + ((s_kv - s_q) if bottom_right else 0)
        assert g_first == _ref_kv_first(q_row0, window_left=window_left, tile_kv=tile_kv), (kv_tile, g_first)
