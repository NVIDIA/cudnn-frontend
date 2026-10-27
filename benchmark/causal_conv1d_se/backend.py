# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit backend C ABI used only by the comparison harness."""

import ctypes
from pathlib import Path

import torch


class Backend:
    def __init__(self, library):
        """Load the explicitly selected cuDNN library for direct SE calls."""
        self.path = str(Path(library).resolve(strict=True))
        self.lib = ctypes.CDLL(self.path)

    def forward(self, x, w, bias, y, *, channel_last, silu):
        """Run NCW or NWH forward into the supplied output on the current CUDA stream."""
        B, D, L = (x.shape[0], x.shape[2], x.shape[1]) if channel_last else x.shape
        K = w.shape[0] if channel_last else w.shape[1]
        name = "cudnnCausalConv1dNwhForward" if channel_last else "cudnnCausalConv1dForward"
        fn = getattr(self.lib, name)
        fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 6
        fn.restype = ctypes.c_int
        dtype = {torch.float32: 0, torch.float64: 1, torch.float16: 2, torch.bfloat16: 9}[x.dtype]
        status = fn(torch.cuda.current_stream().cuda_stream, x.data_ptr(), w.data_ptr(), bias.data_ptr(), y.data_ptr(), B, D, L, K, dtype, int(silu))
        if status:
            raise RuntimeError(f"{name} returned {status} from {self.path}")

    def backward(self, x, w, bias, dy, dx, dw, db, *, channel_last, silu):
        """Clear parameter gradients and run NCW or NWH backward into supplied outputs."""
        B, D, L = (x.shape[0], x.shape[2], x.shape[1]) if channel_last else x.shape
        K = w.shape[0] if channel_last else w.shape[1]
        name = "cudnnCausalConv1dNwhBackward" if channel_last else "cudnnCausalConv1dBackward"
        fn = getattr(self.lib, name)
        fn.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 7
        fn.restype = ctypes.c_int
        dtype = {torch.float32: 0, torch.float64: 1, torch.float16: 2, torch.bfloat16: 9}[x.dtype]
        dw.zero_()
        db.zero_()
        status = fn(
            torch.cuda.current_stream().cuda_stream,
            *(t.data_ptr() for t in (x, w, bias, dy, dx, dw, db)),
            B,
            D,
            L,
            K,
            dtype,
            1 if dw.dtype == torch.float64 else 0,
            int(silu),
        )
        if status:
            raise RuntimeError(f"{name} returned {status} from {self.path}")
