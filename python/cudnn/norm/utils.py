# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level CuTe/DLPack helpers shared by the sm_100 norm kernels.

Mirrors the role of ``cudnn.sdpa.utils``: DLPack -> cute.Tensor conversion and a
couple of small launch-side helpers. The heavy device-side primitives (cp.async
staging, block reductions) live in each pass's ``kernels/_common_sm100.py`` and
build on ``cudnn.frost.tile_dsl`` + ``cutlass.primitives``.
"""

from __future__ import annotations

from cutlass.cute.runtime import from_dlpack


def dyn(t):
    """Wrap a torch tensor as a dynamic-layout cute tensor (16-byte aligned)."""
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic()


def sm_count():
    """Multiprocessor count of the current device (cached)."""
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT


_SM_COUNT = None
_COOP_OCC = {}
COOP_OCC_MAX = 4


def run_coop(key, run):
    """Run ``run(occ)`` at the largest CTAs-per-SM the cooperative launch accepts.

    A cooperative grid must be fully co-resident, so its size is capped by the
    kernel's real occupancy -- which depends on its register count and therefore
    cannot be computed host-side. Sizing the grid to ONE CTA per SM (the obvious
    safe choice) leaves the machine at a fraction of its threads and costs ~20% on
    the NHWC norms; guessing higher trips ``CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE``.

    So try the biggest grid first and halve until it is accepted, remembering the
    answer per kernel shape. The oversize error comes from launch-time validation
    rather than from the device, so a rejected attempt leaves the context usable.
    """
    occ = _COOP_OCC.get(key)
    if occ is not None:
        return run(occ)
    occ = COOP_OCC_MAX
    while True:
        try:
            out = run(occ)
            _COOP_OCC[key] = occ
            return out
        except Exception as e:  # noqa: BLE001 - the DSL wraps the CUDA error type
            if occ <= 1 or "COOPERATIVE" not in str(e).upper():
                raise
            occ //= 2
