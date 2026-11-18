# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A probe kernel for ``cudnn.frost.tile_dsl.pointwise.rcp_div_rn``: ``dst[i] = rcp_div_rn(src[i])`` over a flat fp32 vector.  Not a test
module; imported by ``test_gate_quant_decode.py`` AFTER its device and DSL gates (this module imports the CuTe DSL at top level)."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from cudnn.frost.tile_dsl.pointwise import rcp_div_rn

THREADS = 256


@cute.kernel
def frost_rcp_div_rn_probe(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < n:
        dst[i] = rcp_div_rn(src[i])


frost_rcp_div_rn_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _launch(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
    frost_rcp_div_rn_probe(src, dst, n).launch(grid=((n + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1), stream=stream)


_COMPILED = {}


def run(scales):
    """The reciprocals of a contiguous 1-D fp32 CUDA tensor, as a new tensor (launched on torch's current stream of its device)."""
    out = torch.empty_like(scales)
    views = (from_dlpack(scales, assumed_align=4), from_dlpack(out, assumed_align=4))
    n = scales.numel()
    if scales.device not in _COMPILED:
        with torch.cuda.device(scales.device):
            _COMPILED[scales.device] = cute.compile(_launch, *views, n, make_fake_stream())
    _COMPILED[scales.device](*views, n, cuda.CUstream(torch.cuda.current_stream(scales.device).cuda_stream))
    return out
