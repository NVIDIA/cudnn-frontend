# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TurboQuant-style incoherence processing for the MegaMoE forward.

Randomized block-Hadamard rotation of the hidden dimension (DRIVE/EDEN/
QuaRot/TurboQuant lineage): rotate activations by an orthogonal Q before
MXFP8 quantization and fold Q into the fc1 weight rows offline, so
``(x Q) @ (W1^T Q)^T == x @ W1^T`` exactly.  The rotation spreads
per-channel outliers across each 128-wide block, so the per-32 E8M0 block
scales waste fewer bits on a single hot channel — better quant SNR on
BOTH the dispatch wire payload and the fc1 GEMM A-operand, at the cost of
one small block-diagonal bmm on the activation per forward.

The rotation lives entirely outside the kernel: only the bytes fed to
``DataPreprocess`` / ``load_weights`` change.  fc2 and the combine path are
untouched (the SwiGLU intermediate is requantized inside the kernel, and
the fc2 output leaves the rotated basis by construction).

Q = diag(rademacher_signs) @ H_b / sqrt(b), block-diagonal over
``hidden / b`` blocks with the same b x b block (b=128 aligns with 4
adjacent 32-wide scale blocks). Seed-deterministic, identical on every
rank (weights and activations must share Q).
"""

import math

import torch

import megamoe.repo_path  # noqa: F401

from megamoe.forward import MegaMoeMxfp8Forward, MegaMoeForwardConfig  # noqa: F401


def hadamard_matrix(n: int, device="cuda") -> torch.Tensor:
    """Sylvester Hadamard matrix (n a power of two), entries +-1, fp32."""
    if n & (n - 1):
        raise ValueError(f"hadamard_matrix needs a power of two, got {n}.")
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h


def make_rotation(block: int = 128, seed: int = 20260712, device="cuda") -> torch.Tensor:
    """Orthogonal randomized-Hadamard block: Q = diag(signs) @ H / sqrt(b)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (block,), generator=gen) * 2 - 1).float().to(device)
    return (signs[:, None] * hadamard_matrix(block, device)) / math.sqrt(block)


def rotate_hidden(t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate the trailing (hidden) dim block-diagonally by q (b x b), fp32 math."""
    b = q.shape[0]
    *lead, hidden = t.shape
    if hidden % b:
        raise ValueError(f"hidden ({hidden}) must be a multiple of block ({b}).")
    out = t.reshape(-1, hidden // b, b).float() @ q
    return out.reshape(*lead, hidden)


class TurboQuantMixin:
    """fc1-side randomized-Hadamard incoherence, format-agnostic.

    The rotation is FUSED into the staging write: instead of copy_(x), the
    staging buffer is produced by one (T*hidden/b, b) x (b, b) bf16 GEMM with
    ``out=`` — no extra pass over the activation, so the marginal cost over
    the plain forward is just GEMM-vs-copy on the same bytes.
    load_weights rotates w13's hidden dim (one-time, fp32).
    Compose left of a concrete forward class (mxfp8 / nvfp4).
    """

    def __init__(self, cfg, *, rank, world_size, rotation_block: int = 128,
                 rotation_seed: int = 20260712, **kwargs):
        if cfg.hidden % rotation_block:
            raise ValueError(
                f"hidden ({cfg.hidden}) must be a multiple of rotation_block "
                f"({rotation_block})."
            )
        super().__init__(cfg, rank=rank, world_size=world_size, **kwargs)
        self.q_fp32 = make_rotation(rotation_block, rotation_seed)
        self.q_bf16 = self.q_fp32.bfloat16()
        self.rotation_block = rotation_block

    def load_weights(self, w13: torch.Tensor, w2: torch.Tensor) -> None:
        # Fold Q into fc1's K(hidden) axis; w2 is untouched (its K is the
        # intermediate dim, whose quant lives inside the kernel).
        super().load_weights(rotate_hidden(w13, self.q_fp32), w2)

    def _stage_input(self, x, T):
        b = self.rotation_block
        torch.matmul(
            x.to(torch.bfloat16).reshape(-1, b), self.q_bf16,
            out=self.x_staging[:T].view(-1, b),
        )


class MegaMoeTurboQuantForward(TurboQuantMixin, MegaMoeMxfp8Forward):
    """MXFP8 forward + incoherence."""


from megamoe.forward_nvfp4 import MegaMoeNvfp4Forward  # noqa: E402


class MegaMoeTurboQuantNvfp4Forward(TurboQuantMixin, MegaMoeNvfp4Forward):
    """NVFP4 (4-bit dispatch) forward + incoherence — the combination
    turboquant exists for: 4-bit outlier damage is catastrophic without
    rotation."""
