# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit backend C ABI used only by the comparison harness."""

import ctypes
import hashlib
import os
from pathlib import Path

import torch


def verify_loaded_library(library, backend_version):
    """Require one mapped SE library matching the request and record its on-disk identity.

    Run after the first untimed frontend call, so lazy loading has completed.
    This Linux-only benchmark deliberately rejects ambiguous or deleted mappings.
    """
    requested = Path(library).resolve(strict=True)
    mappings = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not Path(fields[5]).name.startswith("libcudnn_ext.so"):
            continue
        path = Path(fields[5]).resolve(strict=True)
        major, minor = (int(part, 16) for part in fields[3].split(":"))
        mappings[path] = (os.makedev(major, minor), int(fields[4]))
    if len(mappings) != 1 or not requested.samefile(next(iter(mappings))):
        raise RuntimeError(f"Requested backend {requested} does not uniquely match loaded SE libraries: {list(map(str, mappings))}")
    loaded = next(iter(mappings))
    stat = loaded.stat()
    if mappings[loaded] != (stat.st_dev, stat.st_ino):
        raise RuntimeError(f"Loaded backend was replaced on disk: {loaded}")
    digest = hashlib.sha256()
    with loaded.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "requested_path": str(requested),
        "loaded_path": str(loaded),
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "frontend_backend_version": backend_version,
    }


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
