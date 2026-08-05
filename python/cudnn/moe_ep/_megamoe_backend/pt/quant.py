# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NVFP4 / MXFP8 fake-quantization + TurboQuant rotation for the pt path.

Simulated (quantize-dequantize) block formats, mirroring the megamoe
kernels' schemes:

- **nvfp4**: E2M1 values in per-16 blocks, FP8-E4M3 block scales, one fp32
  per-tensor scale (cutedsl_megamoe/moe_nvfp4_swapab/runner_common.py's
  ``nvfp4_quantize_per_block_16``);
- **mxfp8**: E4M3 values in per-32 blocks, E8M0 shared scales with the same
  ceil-log2 convention as cutedsl_megamoe's ``mxfp8_quantize_per_block_32``.

GEMMs run in fp32 on dequantized operands,
so this path answers the ACCURACY question of fp4 training only — it is
slower than the bf16 path, not faster; the speed arrives with a real fp4
bprop megakernel, for which this path is the numerics spec.

Deliberate differences vs the kernel:

- the per-tensor scale is the per-GEMM-call amax (no cross-rank MIN-reduced
  global norm_const) — the outer scale is fp32, so this has negligible
  accuracy effect;
- round-to-nearest ties break toward the lower-magnitude grid point
  (hardware E2M1 casts are ties-to-even; ties are measure-zero here).

Backward modes (:class:`QuantConfig`):

- ``quant_bprop=False`` (default): straight-through — grads are the EXACT
  gradients of the quantized forward, computed on the dequantized forward
  operands (``dX = dY @ Q(W)``, ``dW = dY^T @ Q(X)``).
- ``quant_bprop=True``: additionally NVFP4-quantize every backward GEMM's
  operands along that GEMM's contraction dim (``dY`` along out-features for
  dgrad; ``dY`` and ``X`` along the token dim for wgrad — the transposed
  requant a real bprop kernel must do). Token-dim blocks are zero-padded
  to 16, as a kernel would pad.
- ``stochastic_rounding_grads``: stochastic instead of nearest rounding when
  quantizing ``dY`` (the NVFP4 pretraining recipe's gradient treatment;
  uses the global torch RNG — seed for reproducibility, disable for parity
  tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

F4_MAX = 6.0  # E2M1 max magnitude
F8E4M3_MAX = 448.0
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

FORMATS = ("nvfp4", "mxfp8")


@dataclass
class QuantConfig:
    """Knobs for the simulated-quantized pt path.

    ``fprop_fmt`` quantizes both forward GEMM operands; ``bprop_fmt`` (used
    only with ``quant_bprop=True``) quantizes the backward GEMM operands and
    defaults to ``fprop_fmt``. The interesting mixed recipe is
    ``fprop_fmt="nvfp4", bprop_fmt="mxfp8"`` — 4-bit fprop speed with 8-bit
    gradient fidelity.
    """

    fprop_fmt: str = "nvfp4"  # "nvfp4" | "mxfp8"
    bprop_fmt: str | None = None  # None -> same as fprop_fmt
    turboquant: bool = False  # randomized-Hadamard rotation on fc1's hidden dim
    rotation_block: int = 128
    rotation_seed: int = 20260712
    quant_bprop: bool = False  # quantize backward GEMM operands too
    stochastic_rounding_grads: bool = False  # SR on dY quant (needs quant_bprop)

    def __post_init__(self) -> None:
        if self.fprop_fmt not in FORMATS:
            raise ValueError(f"fprop_fmt must be one of {FORMATS}, got {self.fprop_fmt}")
        if self.bprop_fmt is None:
            self.bprop_fmt = self.fprop_fmt
        if self.bprop_fmt not in FORMATS:
            raise ValueError(f"bprop_fmt must be one of {FORMATS}, got {self.bprop_fmt}")


# ---------------------------------------------------------------------------
# TurboQuant rotation (same construction as megamoe/turboquant.py, but
# device-agnostic and applied IN-GRAPH so master weights stay unrotated and
# autograd supplies the exact adjoint).
# ---------------------------------------------------------------------------


def hadamard_matrix(n: int, device="cpu") -> torch.Tensor:
    """Sylvester Hadamard matrix (n a power of two), entries +-1, fp32."""
    if n & (n - 1):
        raise ValueError(f"hadamard_matrix needs a power of two, got {n}.")
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def make_rotation(block: int = 128, seed: int = 20260712, device="cpu") -> torch.Tensor:
    """Orthogonal randomized-Hadamard block: Q = diag(signs) @ H / sqrt(b)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (block,), generator=gen) * 2 - 1).float().to(device)
    return (signs[:, None] * hadamard_matrix(block, device)) / math.sqrt(block)


def rotate_trailing(t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate the trailing dim block-diagonally by q (b x b). Differentiable;
    fp32 math, cast back to ``t.dtype``."""
    b = q.shape[0]
    *lead, h = t.shape
    if h % b:
        raise ValueError(f"trailing dim ({h}) must be a multiple of block ({b}).")
    out = (t.reshape(-1, b).float() @ q).reshape(*lead, h)
    return out.to(t.dtype)


# ---------------------------------------------------------------------------
# NVFP4 quantize-dequantize
# ---------------------------------------------------------------------------


def _round_e2m1(y: torch.Tensor, stochastic: bool) -> torch.Tensor:
    """Round block-scaled values (|y| expected <= 6) onto the E2M1 grid."""
    grid = torch.tensor(_E2M1_GRID, device=y.device, dtype=torch.float32)
    sign = torch.sign(y)
    a = y.abs().clamp(max=F4_MAX)
    last = len(_E2M1_GRID) - 1
    lo = torch.bucketize(a, grid, right=True).sub_(1).clamp_(0, last)
    hi = torch.clamp(lo + 1, max=last)
    glo, ghi = grid[lo], grid[hi]
    if stochastic:
        span = ghi - glo
        p = (a - glo) / torch.where(span > 0, span, torch.ones_like(span))
        q = torch.where(torch.rand_like(a) < p, ghi, glo)
    else:
        q = torch.where(a - glo <= ghi - a, glo, ghi)
    return sign * q


def fake_quant_nvfp4(
    t: torch.Tensor, dim: int = -1, *, stochastic: bool = False
) -> torch.Tensor:
    """Quantize-dequantize ``t`` to NVFP4 along ``dim`` (per-16 E2M1 blocks,
    FP8-E4M3 block scales, fp32 per-tensor scale). Returns fp32. ``dim`` is
    zero-padded to a multiple of 16 (padding stripped on return)."""
    if t.numel() == 0:
        return t.float()
    x = t.float().movedim(dim, -1)
    n = x.shape[-1]
    pad = (-n) % NVFP4_BLOCK
    if pad:
        x = F.pad(x, (0, pad))
    blocks = x.reshape(-1, NVFP4_BLOCK)
    amax = blocks.abs().amax()
    if amax == 0:
        return torch.zeros_like(t, dtype=torch.float32)
    s_t = amax / (F8E4M3_MAX * F4_MAX)  # per-tensor scale, block scale <= 448
    sf = (blocks.abs().amax(dim=-1, keepdim=True) / (F4_MAX * s_t)).clamp(max=F8E4M3_MAX)
    sf = sf.to(torch.float8_e4m3fn).float()  # RN cast + roundtrip = decode value
    scale = sf * s_t
    inv = torch.where(scale > 0, scale.reciprocal(), torch.zeros_like(scale))
    q = _round_e2m1(blocks * inv, stochastic)
    out = (q * scale).reshape(x.shape)
    if pad:
        out = out[..., :n]
    return out.movedim(-1, dim)


def _round_e4m3(y: torch.Tensor, stochastic: bool) -> torch.Tensor:
    """Round block-scaled values (|y| <= 448) to E4M3 via the native cast.

    Stochastic mode uses ulp-wide uniform dither before the RN cast, which
    equals true stochastic rounding for values between two representables one
    ulp apart (approximate only where the exponent steps)."""
    y = y.clamp(-F8E4M3_MAX, F8E4M3_MAX)
    if stochastic:
        # e4m3: 3 mantissa bits, min normal 2^-6 (subnormal ulp 2^-9).
        e = torch.floor(torch.log2(y.abs().clamp(min=2.0**-9))).clamp(min=-6.0)
        ulp = torch.exp2(e - 3.0)
        y = (y + (torch.rand_like(y) - 0.5) * ulp).clamp(-F8E4M3_MAX, F8E4M3_MAX)
    return y.to(torch.float8_e4m3fn).float()


def fake_quant_mxfp8(
    t: torch.Tensor, dim: int = -1, *, stochastic: bool = False
) -> torch.Tensor:
    """Quantize-dequantize ``t`` to MXFP8 along ``dim`` (per-32 E4M3 blocks,
    E8M0 shared scales; same ceil-log2 scale convention as
    cutedsl_megamoe's ``mxfp8_quantize_per_block_32``). Returns fp32."""
    if t.numel() == 0:
        return t.float()
    x = t.float().movedim(dim, -1)
    n = x.shape[-1]
    pad = (-n) % MXFP8_BLOCK
    if pad:
        x = F.pad(x, (0, pad))
    blocks = x.reshape(-1, MXFP8_BLOCK)
    absmax = blocks.abs().amax(dim=-1, keepdim=True)
    scale_exp = torch.ceil(torch.log2(absmax.clamp(min=1e-30) / F8E4M3_MAX))
    scale = torch.exp2(scale_exp.clamp(-127.0, 127.0))
    q = _round_e4m3(blocks / scale, stochastic)
    out = torch.where(absmax > 0, q * scale, torch.zeros_like(blocks)).reshape(x.shape)
    if pad:
        out = out[..., :n]
    return out.movedim(-1, dim)


def quant_mxfp8_tensors(
    t: torch.Tensor, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """REAL MXFP8 quantization along ``dim`` (must be a multiple of 32):
    returns ``(fp8_e4m3 data, e8m0 scales)`` — the tensors an actual fp8 GEMM
    (e.g. ``torch._scaled_grouped_mm``) consumes, same ceil-log2 scale
    convention as :func:`fake_quant_mxfp8` (dequant of these reproduces it
    bit-exactly)."""
    x = t.float().movedim(dim, -1).contiguous()
    n = x.shape[-1]
    if n % MXFP8_BLOCK:
        raise ValueError(f"quant dim ({n}) must be a multiple of {MXFP8_BLOCK}")
    blocks = x.reshape(-1, MXFP8_BLOCK)
    absmax = blocks.abs().amax(dim=-1, keepdim=True)
    scale_exp = torch.ceil(torch.log2(absmax.clamp(min=1e-30) / F8E4M3_MAX))
    e_u8 = (scale_exp + 127.0).clamp(0.0, 254.0).to(torch.uint8)
    e_u8 = torch.where(absmax == 0, torch.zeros_like(e_u8), e_u8)
    scale = torch.exp2(e_u8.float() - 127.0)
    data = (
        (blocks / scale).clamp(-F8E4M3_MAX, F8E4M3_MAX).to(torch.float8_e4m3fn)
    ).reshape(x.shape)
    scales = e_u8.view(torch.float8_e8m0fnu).reshape(*x.shape[:-1], n // MXFP8_BLOCK)
    return data.movedim(-1, dim), scales


_FAKE_QUANT = {"nvfp4": fake_quant_nvfp4, "mxfp8": fake_quant_mxfp8}


def fake_quant(
    t: torch.Tensor, fmt: str, dim: int = -1, *, stochastic: bool = False
) -> torch.Tensor:
    return _FAKE_QUANT[fmt](t, dim, stochastic=stochastic)


# ---------------------------------------------------------------------------
# Quantized GEMM (y = Q(x) @ Q(w)^T) with configurable backward
# ---------------------------------------------------------------------------


class QuantGemmT(torch.autograd.Function):
    """``y = Q(x) @ Q(w)^T`` with fake-quant along each GEMM's contraction
    dim; fp32 accumulate, output cast back to ``x.dtype``. Forward and
    backward operand formats are independent (fprop_fmt / bprop_fmt)."""

    @staticmethod
    def forward(ctx, x, w, fprop_fmt, bprop_fmt, quant_bprop, sr_grads):
        ctx.save_for_backward(x, w)
        ctx.fprop_fmt = fprop_fmt
        ctx.bprop_fmt = bprop_fmt
        ctx.quant_bprop = quant_bprop
        ctx.sr_grads = sr_grads
        xq = fake_quant(x, fprop_fmt, dim=-1)
        wq = fake_quant(w, fprop_fmt, dim=-1)
        return (xq @ wq.t()).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        gy = gy.float()
        if ctx.quant_bprop:
            # dgrad contracts over out-features; wgrad over tokens. Each
            # operand is requantized along that GEMM's K dim.
            fmt, sr = ctx.bprop_fmt, ctx.sr_grads
            dx = fake_quant(gy, fmt, dim=-1, stochastic=sr) @ fake_quant(w, fmt, dim=0)
            gw = fake_quant(gy, fmt, dim=0, stochastic=sr).t() @ fake_quant(x, fmt, dim=0)
        else:
            # Exact-STE: true gradient of the quantized forward.
            dx = gy @ fake_quant(w, ctx.fprop_fmt, dim=-1)
            gw = gy.t() @ fake_quant(x, ctx.fprop_fmt, dim=-1)
        return dx.to(x.dtype), gw.to(w.dtype), None, None, None, None
