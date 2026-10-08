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


_SMEM_CAPACITY = None


def smem_capacity():
    """Per-block shared memory budget for the CURRENT architecture, via the CuTe DSL.

    ``cutlass.memory.get_smem_capacity_in_bytes()`` resolves the running arch through
    ``CuTeDSL.get_arch_enum()`` and returns the per-BLOCK opt-in limit, which is what a
    launch is actually validated against. Two reasons not to hardcode it:

    * it is architecture-dependent -- 232448 on sm_90/sm_100, but 101376 on sm_120, so
      a Blackwell constant does not merely mis-tune elsewhere, it over-requests and the
      launch fails;
    * it is NOT ``shared_memory_per_multiprocessor`` (233472 on sm_100). Sizing against
      the per-SM figure leaves a 1 KB window where the driver rejects the launch
      instead of us rejecting it first.

    Queried once and cached; the call itself is ~4 us.
    """
    global _SMEM_CAPACITY
    if _SMEM_CAPACITY is None:
        from cutlass.memory import get_smem_capacity_in_bytes

        _SMEM_CAPACITY = int(get_smem_capacity_in_bytes())
    return _SMEM_CAPACITY


def smem_budget(preferred_bytes):
    """A measured-on-sm_100 smem budget, clamped to what this architecture has.

    Deliberately a clamp rather than a rescale: where the tuned value fits it is used
    unchanged, so sm_100 behaviour is bit-identical, and on a smaller-smem part the
    kernel degrades to something launchable instead of being silently retuned by a
    ratio nobody measured.
    """
    return min(int(preferred_bytes), smem_capacity())


_SMEM_PER_SM = None


def smem_per_sm():
    """Total shared memory per SM -- for PINNING occupancy, not for sizing one block.

    Distinct from :func:`smem_capacity`: the BatchNorm kernels deliberately request
    more than ``smem_per_sm() // (occ + 1)`` so that only ``occ`` CTAs can co-reside,
    which is what makes the per-CTA TMEM budget enforceable. The CuTe DSL exposes only
    the per-BLOCK limit (232448 on sm_100) and this is the per-SM total (233472), so
    it comes from the device properties -- still queried, not hardcoded.
    """
    global _SMEM_PER_SM
    if _SMEM_PER_SM is None:
        import torch

        prop = torch.cuda.get_device_properties(0)
        _SMEM_PER_SM = int(getattr(prop, "shared_memory_per_multiprocessor", 0)) or smem_capacity()
    return _SMEM_PER_SM
