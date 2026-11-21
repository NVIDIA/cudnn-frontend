# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Device algorithm source: pkuwangh/mlsys@72c2384a5c325198d6600906d5835d1ef6faabac
# mlsys/gpu/kernels/wan_vae/mega_wan_vae/kernels/_conv_config.py

"""Convolution geometry and specialization policy, independent of the DSL runtime."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvSpecialization:
    """Shape-independent channel/tile selection and persistent grid limit."""

    ci: int
    co: int
    ctas: int
    precision: str = "bf16"

    @property
    def tile_n(self) -> int:
        return _tile_n(self.co)

    @property
    def packed_weight_shape(self) -> tuple[int, ...]:
        return (
            self.co,
            3,
            3,
            3,
            packed_channels(self.ci, self.precision),
        )


def packed_channels(channels: int, precision: str) -> int:
    """Physical filter channels for each arithmetic policy."""
    tile = 32 if precision == "tf32" else 64
    return (channels + tile - 1) // tile * tile


def _tile_n(channels: int) -> int:
    # Exact N192 tiles halve repeated input loads relative to N96.
    if channels in (192, 384):
        return 192
    return 96 if channels == 96 else (256 if channels in (256, 1024) else 160)


SUPPORTED_CHANNEL_PAIRS = frozenset(
    {
        (160, 160),
        (160, 320),
        (320, 320),
        (320, 640),
        (640, 640),
        (512, 256),
        (256, 256),
        (48, 1024),
        (96, 96),
        (96, 192),
        (192, 192),
        (192, 384),
        (384, 384),
        (16, 384),
    }
)


@dataclass(frozen=True)
class ConvConfig:
    """Padded NTHWC dimensions for the 3x3x3, 2CTA kernel."""

    n: int = 1
    t: int = 4
    h: int = 8
    w: int = 8
    ci: int = 160
    co: int = 160
    ctas: int = 148  # Static persistent grid limit.
    precision: str = "bf16"

    @property
    def tile_n(self) -> int:
        """Use exact channel divisors; N192 halves repeated input loads vs N96."""
        return _tile_n(self.co)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.n, self.t, self.h, self.w, self.ci

    @property
    def packed_weight_shape(self) -> tuple[int, ...]:
        """OTRSC filters, padded/packed according to the explicit arithmetic policy."""
        return (
            self.co,
            3,
            3,
            3,
            packed_channels(self.ci, self.precision),
        )

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.n, self.t - 2, self.h - 2, self.w - 2, self.co


def needs_wide_index(config: ConvConfig) -> bool:
    """Bound coordinate/vector indices; byte addresses always use 64 bits.

    Include the final M tile and the largest padded/history vector space.
    This is independent of allocation and can be checked before compilation.
    """
    n, z, h, w, c = config.output_shape
    return max(n * z * h * w + 256, n * (z + 2) * (h + 2) * (w + 2) * (c // 8)) >= 2**31


@dataclass(frozen=True)
class LaunchPolicy:
    """One source for warp roles, pipeline depths, and register redistribution."""

    epilogue_warps: int
    producer_warps: int  # Includes MMA and TMA (or the C12 gather producers).
    auxiliary_warps: int
    ab_stages: int
    c_stages: int
    initial_registers: int = 0
    epilogue_registers: int = 0
    helper_registers: int = 0

    def __post_init__(self) -> None:
        if self.epilogue_warps not in (4, 8, 16) or self.producer_warps < 2 or self.auxiliary_warps < 0:
            raise ValueError("Invalid warp-role partition")
        if not (0 < self.threads <= 1024 and 0 < self.ab_stages <= 8 and 0 < self.c_stages <= 8):
            raise ValueError("Invalid launch or pipeline dimensions")
        budgets = (self.initial_registers, self.epilogue_registers, self.helper_registers)
        if any(budgets):
            if any(value < 24 or value > 256 or value % 8 for value in budgets):
                raise ValueError("Register budgets must be multiples of eight in [24, 256]")
            if self.epilogue_warps % 4 or (self.producer_warps + self.auxiliary_warps) % 4:
                raise ValueError("Register redistribution requires complete warpgroups")
            used = self.epilogue_warps * 32 * self.epilogue_registers
            used += (self.threads - self.epilogue_warps * 32) * self.helper_registers
            if used > self.threads * self.initial_registers or used > 65536:
                raise ValueError("Register redistribution exceeds the CTA budget")

    @property
    def threads(self) -> int:
        return 32 * (self.epilogue_warps + self.producer_warps + self.auxiliary_warps)

    @property
    def auxiliary_start_warp(self) -> int:
        return self.epilogue_warps + self.producer_warps

    @classmethod
    def for_conv(cls, config: ConvSpecialization, *, fuse_norm=False, has_residual=False, spatial_output=False, prepare_output=True):
        if fuse_norm and spatial_output:
            raise ValueError("Prepared and spatial output modes are exclusive")
        epilogue = (16 if has_residual or config.co in (192, 256, 320) else 8) if fuse_norm else 4
        auxiliary = (4 if config.co == 320 else 2) if fuse_norm else (4 if spatial_output else 0)
        ab, c = (6 if config.tile_n == 256 else 7 if config.tile_n == 192 else 8), 2
        if fuse_norm and config.ci == config.co == 320:
            ab, c = 6, 6
        if config.precision in ("tf32"):
            ab, c = (6 if config.tile_n in (192, 256) else 7 if config.tile_n == 160 else 8), 2
            if fuse_norm and config.co == 320:
                ab, c, auxiliary = 6, 4, 6
            budgets = ((80, 104, 32) if config.co == 320 else (96, 104, 64)) if fuse_norm else (0, 0, 0)
            if config.precision == "tf32" and fuse_norm and config.co == 160 and not has_residual:
                # C160 normalization spills at 104 registers; 144 removes that traffic.
                budgets = (128, 144, 64)
            return cls(epilogue, 2, auxiliary if prepare_output or any(budgets) or spatial_output else 0, ab, c, *budgets)
        # Reassign helper registers to the BF16 C192 residual epilogue,
        # keeping the same 640*96-register CTA budget.
        budgets = (0, 0, 0)
        if fuse_norm and config.co == 192:
            budgets = (96, 112, 32) if has_residual else (96, 104, 64)
        return cls(epilogue, 2, auxiliary if prepare_output or any(budgets) or spatial_output else 0, ab, c, *budgets)

    @classmethod
    def for_input(cls):
        return cls(4, 5, 0, 5, 2)
