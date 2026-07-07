# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral RMSNorm + RHT + per-CTA amax operation."""

from __future__ import annotations

from typing import Any, Optional

from .. import data_type
from .._op import Op
from .._tensor_desc import TensorDesc

HAD_BLOCK = 16

DEFAULT_NUM_THREADS_BY_N = {
    2048: 128,
    4096: 256,
    7168: 128,
    8192: 512,
    16384: 1024,
    32768: 512,
}
RPC_CANDIDATES = (2, 4, 8)
TARGET_MIN_CTAS = 148


def best_num_threads(n: int) -> Optional[int]:
    for num_threads in (1024, 512, 256, 128, 64):
        if n % num_threads != 0:
            continue
        ept = n // num_threads
        if ept >= 8 and ept % 8 == 0:
            return num_threads
    return None


def pick_rows_per_cta(m: int) -> int:
    for rows_per_cta in reversed(RPC_CANDIDATES):
        if m % rows_per_cta != 0:
            continue
        if m // rows_per_cta >= TARGET_MIN_CTAS:
            return rows_per_cta
    return RPC_CANDIDATES[0]


class RmsNormRhtAmaxSm100Op(Op):
    """Complete logical signature and launch configuration for the SM100 op."""

    def __init__(
        self,
        *,
        x: TensorDesc[Any],
        weight: TensorDesc[Any],
        output: TensorDesc[Any],
        amax: TensorDesc[Any],
        eps: float = 1e-5,
        num_threads: Optional[int] = None,
        rows_per_cta: Optional[int] = None,
    ) -> None:
        for name, desc in (("x", x), ("weight", weight), ("output", output), ("amax", amax)):
            if not isinstance(desc, TensorDesc):
                raise TypeError(f"{name} must be a TensorDesc, got {type(desc).__name__}")

        self.x = x
        self.weight = weight
        self.output = output
        self.amax = amax
        self.eps = eps
        self.requested_num_threads = num_threads
        self.requested_rows_per_cta = rows_per_cta

        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.num_threads: Optional[int] = None
        self.rows_per_cta: Optional[int] = None

    def check_support(self) -> bool:
        """Validate the complete signature and resolve the launch configuration."""

        self.m = None
        self.n = None
        self.num_threads = None
        self.rows_per_cta = None

        self._check_input_descriptors()
        m, n = self.x.shape

        num_threads = self.requested_num_threads
        if num_threads is None:
            num_threads = DEFAULT_NUM_THREADS_BY_N.get(n, best_num_threads(n))
        if num_threads is None:
            raise ValueError(f"No valid num_threads found for N={n}")

        rows_per_cta = self._resolve_rows_per_cta(m, self.requested_rows_per_cta)
        self._validate_launch_configuration(n, num_threads, rows_per_cta, m=m)
        self._check_output_descriptors(m, n, rows_per_cta)

        self.m = m
        self.n = n
        self.num_threads = num_threads
        self.rows_per_cta = rows_per_cta
        return True

    def _check_input_descriptors(self) -> None:
        if self.x.ndim != 2:
            raise ValueError(f"X must have rank 2, got shape {self.x.shape}")
        if self.weight.ndim != 1:
            raise ValueError(f"W must have rank 1, got shape {self.weight.shape}")
        if self.x.cudnn_dtype != data_type.BFLOAT16:
            raise ValueError(f"X must have dtype bfloat16, got {self.x.dtype}")
        if self.weight.cudnn_dtype != data_type.BFLOAT16:
            raise ValueError(f"W must have dtype bfloat16, got {self.weight.dtype}")

        m, n = self.x.shape
        if m <= 0:
            raise ValueError(f"M must be positive, got {m}")
        if n <= 0:
            raise ValueError(f"N must be positive, got {n}")
        if self.weight.shape != (n,):
            raise ValueError(f"W must have shape {(n,)}, got {self.weight.shape}")
        if self.x.stride != (n, 1) or self.x.stride_order != (1, 0):
            raise ValueError(f"X must be row-major contiguous, got stride {self.x.stride} and stride order {self.x.stride_order}")
        if self.weight.stride != (1,) or self.weight.stride_order != (0,):
            raise ValueError(f"W must be contiguous, got stride {self.weight.stride} and stride order {self.weight.stride_order}")
        if n % HAD_BLOCK != 0:
            raise ValueError(f"N must be divisible by {HAD_BLOCK} for the Hadamard block size, got {n}")

    def _check_output_descriptors(self, m: int, n: int, rows_per_cta: int) -> None:
        if self.output.ndim != 2:
            raise ValueError(f"O must have rank 2, got shape {self.output.shape}")
        if self.amax.ndim != 1:
            raise ValueError(f"Amax must have rank 1, got shape {self.amax.shape}")
        if self.output.cudnn_dtype != data_type.BFLOAT16:
            raise ValueError(f"O must have dtype bfloat16, got {self.output.dtype}")
        if self.amax.cudnn_dtype != data_type.FLOAT:
            raise ValueError(f"Amax must have dtype float32, got {self.amax.dtype}")
        if self.output.shape != (m, n):
            raise ValueError(f"O must have shape {(m, n)}, got {self.output.shape}")
        expected_amax_shape = (m // rows_per_cta,)
        if self.amax.shape != expected_amax_shape:
            raise ValueError(f"Amax must have shape {expected_amax_shape}, got {self.amax.shape}")
        if self.output.stride != (n, 1) or self.output.stride_order != (1, 0):
            raise ValueError(f"O must be row-major contiguous, got stride {self.output.stride} and stride order {self.output.stride_order}")
        if self.amax.stride == (0,):
            raise ValueError("Amax stride must be positive")

    @staticmethod
    def _resolve_rows_per_cta(m: int, rows_per_cta: Optional[int]) -> int:
        if rows_per_cta is None:
            rows_per_cta = pick_rows_per_cta(m)
        if rows_per_cta <= 0:
            raise ValueError(f"rows_per_cta must be positive, got {rows_per_cta}")
        if m % rows_per_cta != 0:
            raise ValueError(f"M must be divisible by rows_per_cta, got M={m}, rows_per_cta={rows_per_cta}")
        return rows_per_cta

    @staticmethod
    def _validate_launch_configuration(
        n: int,
        num_threads: int,
        rows_per_cta: int,
        *,
        m: Optional[int] = None,
    ) -> None:
        if n <= 0:
            raise ValueError(f"N must be positive, got {n}")
        if num_threads <= 0:
            raise ValueError(f"num_threads must be positive, got {num_threads}")
        if num_threads % 32 != 0:
            raise ValueError(f"num_threads must be warp-aligned, got {num_threads}")
        if num_threads > 1024:
            raise ValueError(f"num_threads must not exceed the CUDA block size limit, got {num_threads}")
        if n % num_threads != 0:
            raise ValueError(f"N={n} must be divisible by num_threads={num_threads}")

        ept = n // num_threads
        if ept < 8 or ept % 8 != 0:
            raise ValueError(f"EPT={ept} must be >= 8 and divisible by 8")
        if rows_per_cta <= 0:
            raise ValueError(f"rows_per_cta must be positive, got {rows_per_cta}")
        if m is not None and m % rows_per_cta != 0:
            raise ValueError(f"M must be divisible by rows_per_cta, got M={m}, rows_per_cta={rows_per_cta}")


__all__ = [
    "DEFAULT_NUM_THREADS_BY_N",
    "RPC_CANDIDATES",
    "TARGET_MIN_CTAS",
    "RmsNormRhtAmaxSm100Op",
    "best_num_threads",
    "pick_rows_per_cta",
]
