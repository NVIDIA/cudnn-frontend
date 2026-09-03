# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``build_thd_bwd_setup_kernel``: the SM100 backward's THD setup launch.

Unit-level on purpose.  Everything this kernel produces is consumed by kernels
that do not exist yet, and both of its outputs fail SILENTLY downstream:

* the metadata words -- a reader offset and a writer offset that disagree by one
  make units decode garbage batches rather than raising;
* the per-sequence output descriptors -- their whole job is that stage 3's last
  M tile of a sequence, which OVERSHOOTS into the next sequence's rows with a
  live accumulator behind it, gets clipped by hardware.  ``compute-sanitizer``
  cannot see a TMA store that is in bounds for its descriptor but out of bounds
  for the sequence, so the descriptor extent IS the guard and a sentinel spike
  is the only thing that proves it (frost-gotchas.md, "Validation diagnostics").

Testing it here rather than through the chain means a failure names the setup
kernel instead of surfacing as wrong gradients three kernels later.
"""

from __future__ import annotations

import pytest
import torch

from frost_test_utils import _dsl_installed, requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_pre_rubin_blackwell, requires_dsl]

# Sequence lengths chosen so every interesting case is present at once: not a
# multiple of the block granularity (70, 33, 5), exactly one block (128), and a
# length spanning two blocks (130 kv).
_LENS_Q = (70, 33, 128, 5)
_LENS_KV = (64, 40, 130, 8)
_GRAN = 128  # the blocked workspace's row granularity: stage 2's per-CTA store box

# The DSL imports and the probe kernel sit at MODULE level, behind the same
# probe `requires_dsl` uses.  A `@cute.kernel` defined inside a test function
# resolves its names from the DEFINING MODULE's globals, not from the enclosing
# function's locals, so a nested one dies at trace time with
# `NameError: name 'cute' is not defined`.
if _dsl_installed():
    import cuda.bindings.driver as _cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.experimental import primitives as nvvm
    from cutlass.experimental.cuda import tensor_map as tmap

    from cudnn.frost.tile_dsl.thd import THD_BWD_META_WORDS, THD_SETUP_THREADS
    from cudnn.sdpa.bwd.kernels.thd_helpers import build_thd_bwd_setup_kernel

    @cute.jit
    def _setup_host(meta_t, ql, kl, stream):
        build_thd_bwd_setup_kernel(
            meta_t,
            ql,
            kl,
            cutlass.Int32(0),  # lens_form: both sides arrive as per-batch lengths
            cutlass.Int32(1),  # n_qh
            cutlass.Int32(len(_LENS_Q)),
            cutlass.Int32(_GRAN),  # ws_gran
            cutlass.Int32(_GRAN),  # cga_tile_m
            cutlass.Int32(8),  # n_clusters
        ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)


def _build(dev):
    """Run the setup kernel over ``_LENS_*``; returns the metadata as a list."""
    b = len(_LENS_Q)
    q_lens = torch.tensor(_LENS_Q, dtype=torch.int32, device=dev)
    kv_lens = torch.tensor(_LENS_KV, dtype=torch.int32, device=dev)
    meta = torch.zeros(THD_BWD_META_WORDS(b), dtype=torch.int32, device=dev)

    t = lambda x: from_dlpack(x, assumed_align=16)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    args = (t(meta), t(q_lens), t(kv_lens), stream)
    cute.compile(_setup_host, *args)(*args)
    torch.cuda.synchronize()
    return meta.cpu().tolist()


def test_thd_bwd_setup_metadata():
    """Every metadata word, against a host-side reference."""
    from cudnn.frost.tile_dsl.thd import THD_CTR_OFF, THD_LIVE_OFF, THD_REMAP_OFF, THD_ROWOFF_OFF

    b = len(_LENS_Q)
    meta = _build("cuda")

    cu_q = [0]
    for s in _LENS_Q:
        cu_q.append(cu_q[-1] + s)
    cu_k = [0]
    for s in _LENS_KV:
        cu_k.append(cu_k[-1] + s)

    assert meta[0:b] == list(_LENS_KV), "seq_kv_lens"
    assert meta[b : 2 * b + 1] == cu_q, "cu_seqlens_q"
    assert meta[2 * b + 1 : 3 * b + 2] == cu_k, "cu_seqlens_k"

    # batch_remap: descending Q length, ties on the lower original index.
    want = sorted(range(b), key=lambda i: (-_LENS_Q[i], i))
    assert meta[THD_REMAP_OFF(b) : THD_REMAP_OFF(b) + b] == want, "batch_remap"

    # One unit per (q-block, head); n_qh == 1 here.
    live = sum(-(-s // _GRAN) for s in _LENS_Q)
    assert meta[THD_LIVE_OFF(b)] == live, "live unit total"
    assert meta[THD_CTR_OFF(b)] == 8, "claim counter seeded at n_clusters"

    # row_off: the BLOCKED workspace prefix, each block padded up to _GRAN.
    row = [0]
    for s in _LENS_Q:
        row.append(row[-1] + -(-s // _GRAN) * _GRAN)
    assert meta[THD_ROWOFF_OFF(b) : THD_ROWOFF_OFF(b) + b + 1] == row, "row_off"
