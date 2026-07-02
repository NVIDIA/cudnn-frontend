"""
Shared helpers for the GDN (Gated DeltaNet) test suite.

Availability guards, run dispatcher, and comparison tolerances shared by the
fprop and bprop test files.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch

_HAS_CUDA = torch.cuda.is_available()

try:
    from cudnn.experimental.ops import gated_delta_net
    from cudnn.experimental.ops.linear_attention._gdn_chunk_cutile import (
        chunk_gated_delta_rule,
    )

    _HAS_CUTILE = True
except ImportError:
    _HAS_CUTILE = False

GDN_MARKS = [
    pytest.mark.L0,
    pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA"),
    pytest.mark.skipif(not _HAS_CUTILE, reason="needs the cuda.tile runtime"),
]

# RMS(out - ref) / RMS(ref) thresholds per compute dtype. Forward outputs go
# through one chunked pass; gradients chain several kernels, so they get a
# looser bound. The fp64 recurrent reference is exact at these scales.
FWD_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
STATE_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
BWD_TOL = {torch.bfloat16: 4e-2, torch.float16: 2e-2}

# (H, HV) pairs: H = q/k heads, HV = v/g/beta heads (GVA when HV > H).
HEAD_CONFIGS = [(1, 1), (3, 3), (1, 2), (2, 4), (16, 32), (16, 64)]
HEAD_CONFIGS_BWD = [(1, 1), (3, 3), (1, 4), (2, 4), (16, 32)]
HEAD_CONFIGS_SMALL = [(1, 1), (2, 4)]


def run_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """Run GDN through the public custom op when it supports the config
    (dense batch, H == HV), otherwise through the kernel-module entry point
    (varlen via cu_seqlens, GVA via HV != H). Returns ``(o, final_state)``
    with ``final_state`` normalized to ``None`` when not requested."""
    H, HV = q.shape[2], v.shape[2]
    if cu_seqlens is None and H == HV:
        o, fs = gated_delta_net(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        if fs is None or fs.numel() == 0:
            fs = None
        return o, fs
    o, fs = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return o, fs
