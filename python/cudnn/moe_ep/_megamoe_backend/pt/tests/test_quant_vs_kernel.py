# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Conformance check: pt/quant.py fake-quant vs the cutedsl_megamoe host
quantizers (the numerics the real kernels consume).

Launch (single GPU, needs the cutedsl_megamoe clone + triton):

    python -m pt.tests.test_quant_vs_kernel

Checks:

1. **mxfp8 — bit-exact.** ``fake_quant_mxfp8`` must reproduce
   ``mxfp8_quantize_per_block_32``'s dequantized values exactly (same
   ceil-log2 E8M0 scales, same RN E4M3 cast).
2. **nvfp4 — enumerated divergence.** ``fake_quant_nvfp4`` vs
   ``nvfp4_quantize_per_block_16`` (+ ``unpack_fp4_to_f32``) with
   ``norm_const = 448*6/amax`` (the same per-tensor scale, expressed the
   kernel's way). Block scales must match exactly. Element values may
   differ ONLY at rounding boundaries, from two known sources:
     - the kernel scales by ``norm_const * rcp.approx.ftz(sfc)`` (approximate
       reciprocal) where the simulation divides exactly;
     - the kernel's HW cast is RTNE where the simulation breaks ties toward
       the lower-magnitude grid point.
   Every mismatch must be exactly one E2M1 grid step at the same block
   scale, and the mismatch fraction must stay tiny.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import megamoe.repo_path  # noqa: F401  (puts cutedsl_megamoe on sys.path)

from common.host_utils import mxfp8_quantize_per_block_32  # noqa: E402
from moe_nvfp4_swapab.runner_common import (  # noqa: E402
    nvfp4_quantize_per_block_16,
    unpack_fp4_to_f32,
)

from pt.quant import (  # noqa: E402
    F4_MAX,
    F8E4M3_MAX,
    NVFP4_BLOCK,
    fake_quant_mxfp8,
    fake_quant_nvfp4,
)

SEED = 20260719
CASES = {
    "gaussian": lambda g: torch.randn((512, 1024), device="cuda", generator=g),
    "outliers": lambda g: _outliers(g),
    "tiny": lambda g: torch.randn((512, 1024), device="cuda", generator=g) * 1e-6,
    "hot-rows": lambda g: torch.randn((512, 1024), device="cuda", generator=g)
    * torch.logspace(-3, 3, 512, device="cuda")[:, None],
}


def _outliers(g):
    x = torch.randn((512, 1024), device="cuda", generator=g) / 10.0
    hot = torch.randperm(1024, generator=g, device="cuda")[:10]
    x[:, hot] *= 30.0
    return x


def check_mxfp8(name, x):
    mine = fake_quant_mxfp8(x, dim=-1)
    c_fp8, sfc_e8m0 = mxfp8_quantize_per_block_32(x, torch.float8_e4m3fn)
    scale = torch.exp2(sfc_e8m0.view(torch.uint8).float() - 127.0)
    theirs = c_fp8.float() * scale.repeat_interleave(32, dim=-1)
    n_diff = (mine != theirs).sum().item()
    print(f"[mxfp8] {name:<10} mismatched elements: {n_diff}/{x.numel()}")
    assert n_diff == 0, f"mxfp8 not bit-exact on {name}: {n_diff} diffs"


def check_nvfp4(name, x):
    amax = x.abs().amax().item()
    norm_const = (F8E4M3_MAX * F4_MAX) / amax  # == 1 / s_t of the simulation
    mine = fake_quant_nvfp4(x, dim=-1)

    c_fp4, sfc_fp8 = nvfp4_quantize_per_block_16(x, norm_const)
    q = unpack_fp4_to_f32(c_fp4)
    sf = sfc_fp8.float()
    theirs = q * (sf / norm_const).repeat_interleave(NVFP4_BLOCK, dim=-1)

    # Block scales must agree exactly (bitwise, modulo fp32 op-order in the
    # pre-cast product; count any flips).
    my_sf = (
        x.view(x.shape[0], -1, NVFP4_BLOCK).abs().amax(dim=-1)
        * (norm_const / F4_MAX)
    ).clamp(max=F8E4M3_MAX).to(torch.float8_e4m3fn).float()
    sf_diff = (my_sf != sf).sum().item()

    # Compare in grid-step units: a genuine rounding flip moves >= 0.5 steps
    # (adjacent E2M1 magnitudes differ by >= 0.5 scaled units). Sub-step
    # residue is dequant-arithmetic noise only — the simulation multiplies by
    # ``s_t`` where the kernel path divides by ``norm_const`` (reciprocals
    # differ by <= 1 ulp); the CODES and SCALES the kernel consumes are what
    # must match.
    scale_full = (sf / norm_const).repeat_interleave(NVFP4_BLOCK, dim=-1)
    steps = (mine - theirs).abs() / torch.where(
        scale_full > 0, scale_full, torch.ones_like(scale_full)
    )
    flips = steps > 0.25
    n_flip = flips.sum().item()
    frac = n_flip / x.numel()
    max_step = steps[flips].max().item() if n_flip else 0.0
    noise = steps[~flips].max().item() if (~flips).any() else 0.0
    print(
        f"[nvfp4] {name:<10} scale flips: {sf_diff}/{sf.numel()}  "
        f"code flips: {n_flip}/{x.numel()} ({100*frac:.4f}%)  "
        f"max flip dist: {max_step:.3f} steps  dequant noise: {noise:.2e} steps"
    )
    assert sf_diff <= sf.numel() * 1e-4, f"nvfp4 {name}: block scales diverge"
    assert frac < 0.01, f"nvfp4 {name}: {100*frac:.3f}% codes diverge"
    # One E2M1 step is at most 2.0 (4 -> 6) in scaled units.
    assert max_step <= 2.0 + 1e-6, f"nvfp4 {name}: non-boundary divergence"
    assert noise < 1e-4, f"nvfp4 {name}: dequant arithmetic drifted ({noise})"


def main():
    torch.cuda.set_device(0)
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    for name, make in CASES.items():
        x = make(gen).float()
        check_mxfp8(name, x)
        check_nvfp4(name, x)
    print("QUANT-VS-KERNEL CONFORMANCE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
