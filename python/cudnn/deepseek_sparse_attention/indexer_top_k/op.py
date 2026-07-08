# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral validation for the DeepSeek indexer top-K operation."""

from __future__ import annotations

from typing import Any

from ... import data_type
from ..._op import Op
from ..._tensor_desc import TensorDesc

INT32_MAX = (1 << 31) - 1
SUPPORTED_COMPUTE_CAPABILITIES = (90, 100, 103, 107)
SUPPORTED_INPUT_DTYPES = (data_type.HALF, data_type.BFLOAT16, data_type.FLOAT)


def bucket_num_cols(num_cols: int) -> int:
    if num_cols <= 0:
        return 1
    return 1 << (num_cols - 1).bit_length()


def _dtype_bits(dtype: data_type) -> int:
    return 32 if dtype == data_type.FLOAT else 16


def _require_compact(desc: TensorDesc[Any], label: str) -> None:
    expected_order = tuple(reversed(range(desc.ndim)))
    if not desc.is_compact(expected_order):
        raise ValueError(f"{label} must be row-major contiguous, got stride {desc.stride}")


def _num_threads_per_cta(
    *,
    dtype: data_type,
    max_num_cols: int,
    num_copy_bits: int,
    large_occupancy: bool,
) -> int:
    if large_occupancy:
        return 512
    vec_size = num_copy_bits // _dtype_bits(dtype)
    if dtype == data_type.FLOAT:
        if max_num_cols >= vec_size * 1024:
            return 1024
        if 2048 < max_num_cols < 8192:
            return 512
        return 256
    if max_num_cols >= 43008:
        return 1024
    if 4096 < max_num_cols < 43008:
        return 512
    return 256


class IndexerTopKOp(Op):
    """Complete input, output, and workspace signature for indexer top-K."""

    def __init__(
        self,
        *,
        input_values: TensorDesc[Any],
        seq_lens: TensorDesc[Any],
        output_indices: TensorDesc[Any],
        output_values: TensorDesc[Any] | None,
        workspace: TensorDesc[Any],
        top_k: int,
        next_n: int = 1,
        return_val: bool = True,
        num_copy_bits: int = 256,
        target_compute_capability: int = 90,
    ) -> None:
        for name, desc in (
            ("input_values", input_values),
            ("seq_lens", seq_lens),
            ("output_indices", output_indices),
            ("workspace", workspace),
        ):
            if not isinstance(desc, TensorDesc):
                raise TypeError(f"{name} must be a TensorDesc, got {type(desc).__name__}")
        if output_values is not None and not isinstance(output_values, TensorDesc):
            raise TypeError(f"output_values must be a TensorDesc or None, got {type(output_values).__name__}")

        self.input_values = input_values
        self.seq_lens = seq_lens
        self.output_indices = output_indices
        self.output_values = output_values
        self.workspace = workspace
        self.top_k = int(top_k)
        self.next_n = int(next_n)
        self.return_val = bool(return_val)
        self.num_copy_bits = int(num_copy_bits)
        self.target_compute_capability = int(target_compute_capability)

        self.num_rows: int | None = None
        self.num_cols: int | None = None
        self.batch_size: int | None = None
        self.buffer_count: int | None = None
        self.dtype_bits: int | None = None
        self.max_num_cols: int | None = None
        self.large_occupancy: bool | None = None
        self.num_threads_per_cta: int | None = None

    def check_support(self) -> bool:
        if self.target_compute_capability not in SUPPORTED_COMPUTE_CAPABILITIES:
            raise ValueError("target_compute_capability must be one of " f"{SUPPORTED_COMPUTE_CAPABILITIES}, got {self.target_compute_capability}")
        if self.input_values.ndim != 2:
            raise ValueError(f"input_values must be 2-D, got {self.input_values.shape}")
        if self.seq_lens.ndim != 1:
            raise ValueError(f"seq_lens must be 1-D, got {self.seq_lens.shape}")
        if self.input_values.cudnn_dtype not in SUPPORTED_INPUT_DTYPES:
            raise ValueError(f"input_values must have dtype float16, bfloat16, or float32, got {self.input_values.dtype}")
        if self.seq_lens.cudnn_dtype != data_type.INT32:
            raise ValueError(f"seq_lens must have dtype int32, got {self.seq_lens.dtype}")
        _require_compact(self.input_values, "input_values")
        _require_compact(self.seq_lens, "seq_lens")

        num_rows, num_cols = self.input_values.shape
        (batch_size,) = self.seq_lens.shape
        if num_rows <= 0 or num_cols <= 0:
            raise ValueError(f"input_values dimensions must be positive, got {(num_rows, num_cols)}")
        if batch_size <= 0:
            raise ValueError("seq_lens must not be empty")
        if self.next_n <= 0:
            raise ValueError(f"next_n must be positive, got {self.next_n}")
        if num_rows != batch_size * self.next_n:
            raise ValueError(f"num_rows ({num_rows}) must equal seq_lens.size * next_n " f"({batch_size} * {self.next_n} = {batch_size * self.next_n})")
        if self.top_k <= 0 or self.top_k > min(2048, num_cols):
            raise ValueError(f"top_k must be in (0, min(2048, num_cols={num_cols})], got {self.top_k}")
        if self.num_copy_bits <= 0 or self.num_copy_bits % 8:
            raise ValueError(f"num_copy_bits must be a positive whole-byte width, got {self.num_copy_bits}")
        copy_bytes = self.num_copy_bits // 8
        if copy_bytes & (copy_bytes - 1):
            raise ValueError(f"num_copy_bits must describe a power-of-two byte alignment, got {self.num_copy_bits}")

        dtype_bits = _dtype_bits(self.input_values.cudnn_dtype)
        if self.num_copy_bits % dtype_bits:
            raise ValueError(f"num_copy_bits ({self.num_copy_bits}) must be divisible by input dtype width ({dtype_bits})")

        output_shape = (num_rows, self.top_k)
        if self.output_indices.shape != output_shape or self.output_indices.cudnn_dtype != data_type.INT32:
            raise ValueError(f"output_indices must be int32 with shape {output_shape}, got {self.output_indices.shape}")
        _require_compact(self.output_indices, "output_indices")
        if self.return_val:
            if self.output_values is None:
                raise ValueError("output_values is required when return_val=True")
            if self.output_values.shape != output_shape or self.output_values.cudnn_dtype != self.input_values.cudnn_dtype:
                raise ValueError(f"output_values must have shape {output_shape} and match input_values dtype")
            _require_compact(self.output_values, "output_values")
        elif self.output_values is not None:
            raise ValueError("output_values must be None when return_val=False")

        buffer_count = 2 if self.input_values.cudnn_dtype == data_type.FLOAT else 1
        workspace_shape = (num_rows, buffer_count, num_cols)
        if self.workspace.shape != workspace_shape or self.workspace.cudnn_dtype != data_type.INT32:
            raise ValueError(f"workspace must be int32 with shape {workspace_shape}, got {self.workspace.shape}")
        _require_compact(self.workspace, "workspace")
        max_num_cols = bucket_num_cols(num_cols)
        large_occupancy = num_rows > 148
        num_threads_per_cta = _num_threads_per_cta(
            dtype=self.input_values.cudnn_dtype,
            max_num_cols=max_num_cols,
            num_copy_bits=self.num_copy_bits,
            large_occupancy=large_occupancy,
        )
        vector_size = min(
            self.top_k,
            (self.top_k + num_threads_per_cta - 1) // num_threads_per_cta,
            self.num_copy_bits // dtype_bits,
            2,
        )
        if self.top_k % vector_size:
            raise ValueError(f"top_k ({self.top_k}) must be divisible by the selected output vector width ({vector_size})")

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.batch_size = batch_size
        self.buffer_count = buffer_count
        self.dtype_bits = dtype_bits
        self.max_num_cols = max_num_cols
        self.large_occupancy = large_occupancy
        self.num_threads_per_cta = num_threads_per_cta
        return True


__all__ = [
    "INT32_MAX",
    "IndexerTopKOp",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "SUPPORTED_INPUT_DTYPES",
    "bucket_num_cols",
]
