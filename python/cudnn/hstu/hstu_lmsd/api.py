# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal compile/execute implementations for the HSTU LMSD functions."""

from __future__ import annotations

import math
from typing import Optional

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream
import torch

from cudnn.api_base import APIBase, TensorDesc

from ._runtime import record_streams, stream_handle
from ._kernels._common import (
    ALIGNMENT_BYTES,
    HIDDEN_SIZE_GRANULARITY,
    MAX_FLAT_ELEMENTS,
    MAX_HIDDEN_SIZE_EXCLUSIVE,
    is_supported_hidden_size,
    keep_threshold32,
    normalize_dropout_ratio,
)
from ._kernels._config import HSTULMSDBwdConfig, HSTULMSDFwdConfig, HSTULMSDGradReduceConfig
from ._kernels.hstu_lmsd_fwd import HSTULMSDForward
from ._kernels.hstu_lmsd_bwd import HSTULMSDBackward, HSTULMSDGradReduce


def _require_cuda_tensor(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.data_ptr() % ALIGNMENT_BYTES != 0:
        raise ValueError(f"{name} storage must be {ALIGNMENT_BYTES}-byte aligned")


def _require_same_device_desc(reference: TensorDesc, tensors) -> None:
    if reference.device.type != "cuda":
        raise ValueError(f"{reference.name} must be on a CUDA device, got {reference.device}")
    for name, desc in tensors:
        if desc.device.type != "cuda":
            raise ValueError(f"{name} must be on a CUDA device, got {desc.device}")
        if desc.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}, got {desc.device}")


def _require_matrix_layout(
    tensor: torch.Tensor | TensorDesc,
    name: str,
    *,
    row_stride: Optional[int] = None,
) -> None:
    shape = tuple(tensor.shape)
    stride_attr = tensor.stride
    stride = tuple(stride_attr() if callable(stride_attr) else stride_attr)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank 2, got shape {shape}")
    if stride[1] != 1:
        raise ValueError(f"{name} must have a unit innermost stride")
    if row_stride is not None and stride[0] != row_stride:
        raise ValueError(f"{name} row stride must be {row_stride}, got {stride[0]}")
    if stride[0] < shape[1]:
        raise ValueError(f"{name} rows must not overlap")
    if tensor.dtype == torch.bfloat16 and stride[0] % 8 != 0:
        raise ValueError(f"{name} row starts must remain 16-byte aligned")


def _require_vector(desc: TensorDesc, name: str, length: int) -> None:
    if desc.shape != (length,) or desc.stride != (1,):
        raise ValueError(f"{name} must be contiguous with shape ({length},), got " f"shape {tuple(desc.shape)} and stride {tuple(desc.stride)}")


def _check_runtime_tensor(
    tensor: torch.Tensor,
    desc: TensorDesc,
    name: str,
    *,
    num_rows: Optional[int] = None,
    dynamic_row_stride: bool = False,
) -> None:
    expected_shape = tuple(desc.shape) if num_rows is None else (num_rows, *desc.shape[1:])
    stride_mismatch = not dynamic_row_stride and tuple(tensor.stride()) != tuple(desc.stride)
    if expected_shape != tuple(tensor.shape) or stride_mismatch or tensor.dtype != desc.dtype or tensor.device != desc.device:
        raise ValueError(f"{name} specification changed after compilation")
    if dynamic_row_stride:
        _require_matrix_layout(tensor, name)
    _require_cuda_tensor(tensor, name)


def _storage_span(tensor: torch.Tensor) -> tuple[int, int]:
    start = tensor.data_ptr()
    offset = sum((int(size) - 1) * int(stride) for size, stride in zip(tensor.shape, tensor.stride()) if size > 0)
    return start, start + (offset + 1) * tensor.element_size()


def _require_disjoint(writes, reads) -> None:
    all_tensors = tuple(reads) + tuple(writes)
    for write_name, write_tensor in writes:
        write_begin, write_end = _storage_span(write_tensor)
        for other_name, other_tensor in all_tensors:
            if write_name == other_name and write_tensor is other_tensor:
                continue
            if write_tensor.device != other_tensor.device:
                continue
            other_begin, other_end = _storage_span(other_tensor)
            if write_begin < other_end and other_begin < write_end:
                raise ValueError(f"{write_name} storage must not overlap {other_name} storage")


class _HSTULMSDBase(APIBase):
    """Validation and dynamic fake-tensor helpers shared by forward/backward."""

    def _init_common(
        self,
        *,
        sample_x: torch.Tensor | TensorDesc,
        sample_u: torch.Tensor | TensorDesc,
        sample_weight: torch.Tensor | TensorDesc,
        sample_bias: torch.Tensor | TensorDesc,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self._warn_experimental_api()
        self.x_desc = self._make_tensor_desc(sample_x, name="x")
        self.u_desc = self._make_tensor_desc(sample_u, name="u")
        self.weight_desc = self._make_tensor_desc(sample_weight, name="weight")
        self.bias_desc = self._make_tensor_desc(sample_bias, name="bias")
        self.dropout_ratio = float(dropout_ratio)
        if self.x_desc.ndim != 2:
            raise ValueError("x must be rank 2")
        self.num_rows = int(self.x_desc.shape[0])
        self.hidden_size = int(self.x_desc.shape[1])

    def _check_common(self) -> None:
        x = self.x_desc
        u = self.u_desc
        weight = self.weight_desc
        bias = self.bias_desc
        _require_same_device_desc(
            x,
            (("u", u), ("weight", weight), ("bias", bias)),
        )
        if x.dtype != torch.bfloat16:
            raise ValueError(f"x must have dtype torch.bfloat16, got {x.dtype}")
        if u.dtype != x.dtype or weight.dtype != x.dtype or bias.dtype != x.dtype:
            raise ValueError("x, u, weight, and bias must have the same dtype")
        if x.shape != u.shape:
            raise ValueError(f"u must have shape {tuple(x.shape)}, got {tuple(u.shape)}")
        if not is_supported_hidden_size(self.hidden_size):
            raise ValueError(
                f"HSTU LMSD supports D divisible by {HIDDEN_SIZE_GRANULARITY} in "
                f"[{HIDDEN_SIZE_GRANULARITY}, {MAX_HIDDEN_SIZE_EXCLUSIVE}), got D={self.hidden_size}"
            )
        self.max_num_rows = MAX_FLAT_ELEMENTS // self.hidden_size
        if not 1 <= self.num_rows <= self.max_num_rows:
            raise ValueError(f"x row count must be in [1, {self.max_num_rows}] for D={self.hidden_size}, got {self.num_rows}")
        _require_matrix_layout(x, "x")
        _require_matrix_layout(u, "u")
        _require_vector(weight, "weight", self.hidden_size)
        _require_vector(bias, "bias", self.hidden_size)
        self.dropout_ratio = normalize_dropout_ratio(self.dropout_ratio)
        major, minor = torch.cuda.get_device_capability(x.device)
        if major != 10:
            raise RuntimeError(f"HSTU LMSD requires an SM10x GPU; found SM{major}{minor}")

    def _check_runtime_common(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> int:
        if x.ndim != 2 or x.shape[0] <= 0:
            raise ValueError(f"x must be rank 2 with at least one row, got shape {tuple(x.shape)}")
        num_rows = int(x.shape[0])
        if num_rows > self.max_num_rows:
            raise ValueError(f"x row count must be in [1, {self.max_num_rows}] for D={self.hidden_size}, got {num_rows}")
        for tensor, desc, name, dynamic_rows, dynamic_row_stride in (
            (x, self.x_desc, "x", True, True),
            (u, self.u_desc, "u", True, True),
            (weight, self.weight_desc, "weight", False, False),
            (bias, self.bias_desc, "bias", False, False),
        ):
            _check_runtime_tensor(
                tensor,
                desc,
                name,
                num_rows=num_rows if dynamic_rows else None,
                dynamic_row_stride=dynamic_row_stride,
            )
        return num_rows

    def _fake_matrix(self, desc, rows, *, dynamic_row_stride: bool = False) -> cute.Tensor:
        stride = (cute.sym_int64(divisibility=8), 1) if dynamic_row_stride else desc.stride
        return self._make_fake_cute_tensor(
            dtype=desc.dtype,
            shape=(rows, desc.shape[1]),
            stride=stride,
            assumed_align=ALIGNMENT_BYTES,
        )

    def _fake_vector(self, desc, length=None) -> cute.Tensor:
        return self._make_fake_cute_tensor(
            dtype=desc.dtype,
            shape=(desc.shape[0] if length is None else length,),
            stride=desc.stride,
            assumed_align=ALIGNMENT_BYTES,
        )


class HSTULMSDFwd(_HSTULMSDBase):
    """Explicit compile/execute API for HSTU LMSD forward on SM10x."""

    def __init__(
        self,
        sample_x: torch.Tensor | TensorDesc,
        sample_u: torch.Tensor | TensorDesc,
        sample_weight: torch.Tensor | TensorDesc,
        sample_bias: torch.Tensor | TensorDesc,
        sample_y: torch.Tensor | TensorDesc,
        sample_mean: torch.Tensor | TensorDesc,
        sample_rstd: torch.Tensor | TensorDesc,
        sample_mask: torch.Tensor | TensorDesc | None,
        eps: float = 1e-6,
        dropout_ratio: float = 0.1,
        apply_u_silu: bool = True,
        concat_u: bool = True,
        concat_x: bool = True,
    ) -> None:
        self._init_common(
            sample_x=sample_x,
            sample_u=sample_u,
            sample_weight=sample_weight,
            sample_bias=sample_bias,
            dropout_ratio=dropout_ratio,
        )
        self.y_desc = self._make_tensor_desc(sample_y, name="y")
        self.mean_desc = self._make_tensor_desc(sample_mean, name="mean")
        self.rstd_desc = self._make_tensor_desc(sample_rstd, name="rstd")
        self.mask_desc = self._make_tensor_desc(sample_mask, name="mask") if sample_mask is not None else None
        self.eps = float(eps)
        self.apply_u_silu = bool(apply_u_silu)
        self.concat_u = bool(concat_u)
        self.concat_x = bool(concat_x)
        self.output_segments = 1 + int(self.concat_u) + int(self.concat_x)
        self._kernel_config = HSTULMSDFwdConfig.from_hidden_size(self.hidden_size)

    def check_support(self) -> bool:
        if self._is_supported:
            return True
        self._check_common()
        self.has_dropout = self.dropout_ratio > 0.0
        self._threshold = keep_threshold32(self.dropout_ratio) if self.has_dropout else 0
        n, d = self.num_rows, self.hidden_size
        y = self.y_desc
        mean = self.mean_desc
        rstd = self.rstd_desc
        mask = self.mask_desc
        _require_same_device_desc(
            self.x_desc,
            (("y", y), ("mean", mean), ("rstd", rstd)) + ((("mask", mask),) if mask is not None else ()),
        )
        output_width = self.output_segments * d
        if y.shape != (n, output_width) or y.dtype != self.x_desc.dtype:
            raise ValueError(f"y must have shape ({n}, {output_width}) and dtype {self.x_desc.dtype}")
        _require_matrix_layout(y, "y", row_stride=output_width)
        for tensor, name in ((mean, "mean"), (rstd, "rstd")):
            _require_vector(tensor, name, n)
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name} must have dtype torch.float32")
        if self.has_dropout:
            if mask is None:
                raise ValueError("mask must be provided when dropout_ratio is nonzero")
            if mask.shape != (n, d) or mask.dtype != torch.int8:
                raise ValueError(f"mask must have shape ({n}, {d}) and dtype torch.int8")
            _require_matrix_layout(mask, "mask", row_stride=d)
        elif mask is not None:
            raise ValueError("mask must be None when dropout_ratio is zero")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(f"eps must be positive and finite, got {self.eps}")
        self._multiprocessor_count = torch.cuda.get_device_properties(self.x_desc.device).multi_processor_count
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        rows = cute.sym_int()
        fake_x = self._fake_matrix(self.x_desc, rows, dynamic_row_stride=True)
        fake_u = self._fake_matrix(self.u_desc, rows, dynamic_row_stride=True)
        fake_weight = self._fake_vector(self.weight_desc)
        fake_bias = self._fake_vector(self.bias_desc)
        fake_output_segment = self._make_fake_cute_tensor(
            dtype=self.y_desc.dtype,
            shape=(rows, self.hidden_size),
            stride=self.y_desc.stride,
            assumed_align=ALIGNMENT_BYTES,
        )
        # Every materialized segment has the same plan-time tensor contract.
        fake_silu_output = fake_output_segment
        fake_x_output = fake_output_segment
        fake_lmsd_output = fake_output_segment
        fake_mask = self._fake_matrix(self.mask_desc, rows) if self.has_dropout else fake_x
        fake_mean = self._fake_vector(self.mean_desc, rows)
        fake_rstd = self._fake_vector(self.rstd_desc, rows)
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        self._compiled_kernel = cute.compile(
            HSTULMSDForward(
                self.hidden_size,
                apply_u_silu=self.apply_u_silu,
                concat_u=self.concat_u,
                concat_x=self.concat_x,
                has_dropout=self.has_dropout,
                config=self._kernel_config,
            ),
            fake_x,
            fake_u,
            fake_weight,
            fake_bias,
            fake_silu_output,
            fake_x_output,
            fake_lmsd_output,
            fake_mask,
            fake_mean,
            fake_rstd,
            cutlass.Int64(0),
            cutlass.Int32(1),
            cutlass.Int32(self.hidden_size),
            cutlass.Float32(self.eps),
            cutlass.Float32(self.dropout_ratio),
            cutlass.Uint32(self._threshold),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            fake_stream,
            options="--enable-tvm-ffi",
        )

    def execute(
        self,
        x_tensor: torch.Tensor,
        u_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        bias_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        mean_tensor: torch.Tensor,
        rstd_tensor: torch.Tensor,
        mask_tensor: Optional[torch.Tensor],
        seed: int,
        current_stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    ) -> None:
        if self._compiled_kernel is None:
            raise RuntimeError("HSTULMSDFwd kernel is not compiled")
        n = self._check_runtime_common(x_tensor, u_tensor, weight_tensor, bias_tensor)
        for tensor, desc, name in (
            (y_tensor, self.y_desc, "y"),
            (mean_tensor, self.mean_desc, "mean"),
            (rstd_tensor, self.rstd_desc, "rstd"),
        ):
            _check_runtime_tensor(tensor, desc, name, num_rows=n)
        if self.has_dropout:
            if mask_tensor is None:
                raise ValueError("mask_tensor is required by the compiled dropout configuration")
            _check_runtime_tensor(mask_tensor, self.mask_desc, "mask", num_rows=n)
        elif mask_tensor is not None:
            raise ValueError("mask_tensor must be None for the compiled no-dropout configuration")
        if not -(1 << 63) <= int(seed) < (1 << 63):
            raise ValueError("seed must fit in a signed 64-bit integer")
        writes = (("y", y_tensor), ("mean", mean_tensor), ("rstd", rstd_tensor))
        if mask_tensor is not None:
            writes += (("mask", mask_tensor),)
        _require_disjoint(
            writes,
            (("x", x_tensor), ("u", u_tensor), ("weight", weight_tensor), ("bias", bias_tensor)),
        )
        stream = stream_handle(current_stream, x_tensor.device)
        d = self.hidden_size
        num_row_blocks = (n + self._kernel_config.rows_per_cta - 1) // self._kernel_config.rows_per_cta
        grid_size = self._kernel_config.launch_grid(n, self._multiprocessor_count)
        num_iterations = (num_row_blocks + grid_size - 1) // grid_size
        segment = 0
        silu_output = y_tensor[:, segment * d : (segment + 1) * d] if self.concat_u else None
        segment += int(self.concat_u)
        x_output = y_tensor[:, segment * d : (segment + 1) * d] if self.concat_x else None
        segment += int(self.concat_x)
        lmsd_output = y_tensor[:, segment * d : (segment + 1) * d]
        self._compiled_kernel(
            x_tensor,
            u_tensor,
            weight_tensor,
            bias_tensor,
            silu_output if silu_output is not None else lmsd_output,
            x_output if x_output is not None else lmsd_output,
            lmsd_output,
            mask_tensor if mask_tensor is not None else x_tensor,
            mean_tensor,
            rstd_tensor,
            cutlass.Int64(seed),
            cutlass.Int32(n),
            cutlass.Int32(d),
            cutlass.Float32(self.eps),
            cutlass.Float32(self.dropout_ratio),
            cutlass.Uint32(self._threshold),
            cutlass.Int32(num_row_blocks),
            cutlass.Int32(num_iterations),
            cutlass.Int32(grid_size),
            stream,
        )
        record_streams(
            tuple(t for t in (x_tensor, u_tensor, weight_tensor, bias_tensor, y_tensor, mean_tensor, rstd_tensor, mask_tensor) if t is not None),
            current_stream,
            x_tensor.device,
        )


class HSTULMSDBwd(_HSTULMSDBase):
    """Explicit LMSD backward without forward-output recomputation."""

    def __init__(
        self,
        sample_dy: torch.Tensor | TensorDesc,
        sample_x: torch.Tensor | TensorDesc,
        sample_u: torch.Tensor | TensorDesc,
        sample_weight: torch.Tensor | TensorDesc,
        sample_bias: torch.Tensor | TensorDesc,
        sample_mean: torch.Tensor | TensorDesc,
        sample_rstd: torch.Tensor | TensorDesc,
        sample_mask: torch.Tensor | TensorDesc | None,
        sample_dx: torch.Tensor | TensorDesc,
        sample_du: torch.Tensor | TensorDesc,
        sample_dweight: torch.Tensor | TensorDesc | None,
        sample_dbias: torch.Tensor | TensorDesc,
        sample_dweight_workspace: torch.Tensor | TensorDesc | None,
        sample_dbias_workspace: torch.Tensor | TensorDesc,
        dropout_ratio: float = 0.1,
        apply_u_silu: bool = True,
        concat_u: bool = True,
        concat_x: bool = True,
        compute_dweight: bool = True,
    ) -> None:
        self._init_common(
            sample_x=sample_x,
            sample_u=sample_u,
            sample_weight=sample_weight,
            sample_bias=sample_bias,
            dropout_ratio=dropout_ratio,
        )
        samples = {
            "dy": sample_dy,
            "mean": sample_mean,
            "rstd": sample_rstd,
            "dx": sample_dx,
            "du": sample_du,
            "dbias": sample_dbias,
            "dbias_workspace": sample_dbias_workspace,
        }
        for name, tensor in samples.items():
            setattr(self, f"{name}_desc", self._make_tensor_desc(tensor, name=name))
        self.mask_desc = self._make_tensor_desc(sample_mask, name="mask") if sample_mask is not None else None
        self.dweight_desc = self._make_tensor_desc(sample_dweight, name="dweight") if sample_dweight is not None else None
        self.dweight_workspace_desc = (
            self._make_tensor_desc(sample_dweight_workspace, name="dweight_workspace") if sample_dweight_workspace is not None else None
        )
        self.apply_u_silu = bool(apply_u_silu)
        self.concat_u = bool(concat_u)
        self.concat_x = bool(concat_x)
        self.compute_dweight = bool(compute_dweight)
        self.output_segments = 1 + int(self.concat_u) + int(self.concat_x)
        self._kernel_config = HSTULMSDBwdConfig.from_hidden_size(self.hidden_size)
        self._reduce_config = HSTULMSDGradReduceConfig()

    def check_support(self) -> bool:
        if self._is_supported:
            return True
        self._check_common()
        self._multiprocessor_count = torch.cuda.get_device_properties(self.x_desc.device).multi_processor_count
        self._workspace_rows = self._kernel_config.workspace_rows(self._multiprocessor_count)
        self.has_dropout = self.dropout_ratio > 0.0
        n, d = self.num_rows, self.hidden_size
        x = self.x_desc
        tensors = [
            ("dy", self.dy_desc),
            ("mean", self.mean_desc),
            ("rstd", self.rstd_desc),
            ("dx", self.dx_desc),
            ("du", self.du_desc),
            ("dbias", self.dbias_desc),
            ("dbias_workspace", self.dbias_workspace_desc),
        ]
        if self.mask_desc is not None:
            tensors.append(("mask", self.mask_desc))
        if self.dweight_desc is not None:
            tensors.append(("dweight", self.dweight_desc))
        if self.dweight_workspace_desc is not None:
            tensors.append(("dweight_workspace", self.dweight_workspace_desc))
        _require_same_device_desc(x, tensors)
        output_width = self.output_segments * d
        if self.dy_desc.shape != (n, output_width) or self.dy_desc.dtype != x.dtype:
            raise ValueError(f"dy must have shape ({n}, {output_width}) and dtype {x.dtype}")
        _require_matrix_layout(self.dy_desc, "dy")
        for desc, name in ((self.mean_desc, "mean"), (self.rstd_desc, "rstd")):
            _require_vector(desc, name, n)
            if desc.dtype != torch.float32:
                raise ValueError(f"{name} must have dtype torch.float32")
        if self.has_dropout:
            if self.mask_desc is None:
                raise ValueError("mask must be provided when dropout_ratio is nonzero")
            if self.mask_desc.shape != (n, d) or self.mask_desc.dtype != torch.int8:
                raise ValueError(f"mask must have shape ({n}, {d}) and dtype torch.int8")
            _require_matrix_layout(self.mask_desc, "mask", row_stride=d)
        elif self.mask_desc is not None:
            raise ValueError("mask must be None when dropout_ratio is zero")
        for desc, name in ((self.dx_desc, "dx"), (self.du_desc, "du")):
            if desc.shape != (n, d) or desc.dtype != x.dtype:
                raise ValueError(f"{name} must have shape ({n}, {d}) and dtype {x.dtype}")
            _require_matrix_layout(desc, name, row_stride=d)
        if self.compute_dweight:
            if self.dweight_desc is None or self.dweight_workspace_desc is None:
                raise ValueError("dweight and dweight_workspace are required when compute_dweight is True")
            _require_vector(self.dweight_desc, "dweight", d)
            if self.dweight_desc.dtype != x.dtype:
                raise ValueError(f"dweight must have dtype {x.dtype}")
        elif self.dweight_desc is not None or self.dweight_workspace_desc is not None:
            raise ValueError("dweight and dweight_workspace must be None when compute_dweight is False")
        _require_vector(self.dbias_desc, "dbias", d)
        if self.dbias_desc.dtype != x.dtype:
            raise ValueError(f"dbias must have dtype {x.dtype}")
        workspaces = [(self.dbias_workspace_desc, "dbias_workspace")]
        if self.compute_dweight:
            workspaces.append((self.dweight_workspace_desc, "dweight_workspace"))
        for desc, name in workspaces:
            if desc.shape != (self._workspace_rows, d) or desc.dtype != torch.float32:
                raise ValueError(f"{name} must have shape ({self._workspace_rows}, {d}) and dtype torch.float32")
            _require_matrix_layout(desc, name, row_stride=d)
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        d = self.hidden_size
        fake_weight = self._fake_vector(self.weight_desc)
        fake_bias = self._fake_vector(self.bias_desc)
        fake_dbp = self._make_fake_cute_tensor_from_desc(self.dbias_workspace_desc, assumed_align=ALIGNMENT_BYTES)
        fake_db = self._make_fake_cute_tensor_from_desc(self.dbias_desc, assumed_align=ALIGNMENT_BYTES)
        fake_dwp = self._make_fake_cute_tensor_from_desc(self.dweight_workspace_desc, assumed_align=ALIGNMENT_BYTES) if self.compute_dweight else fake_dbp
        fake_dw = self._make_fake_cute_tensor_from_desc(self.dweight_desc, assumed_align=ALIGNMENT_BYTES) if self.compute_dweight else fake_db
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        # Compile the row count symbolically so one plan serves every supported N.
        rows = cute.sym_int()
        fake_dy_segment = self._make_fake_cute_tensor(
            dtype=self.dy_desc.dtype,
            shape=(rows, d),
            stride=(cute.sym_int64(divisibility=8), 1),
            assumed_align=ALIGNMENT_BYTES,
        )
        # Every materialized dY segment has the same plan-time tensor contract.
        fake_dy_silu = fake_dy_segment
        fake_dy_x = fake_dy_segment
        fake_dy_lmsd = fake_dy_segment
        fake_x = self._fake_matrix(self.x_desc, rows, dynamic_row_stride=True)
        fake_u = self._fake_matrix(self.u_desc, rows, dynamic_row_stride=True)
        fake_mask = self._fake_matrix(self.mask_desc, rows) if self.has_dropout else fake_x
        fake_dx = self._fake_matrix(self.dx_desc, rows)
        fake_du = self._fake_matrix(self.du_desc, rows)
        fake_mean = self._fake_vector(self.mean_desc, rows)
        fake_rstd = self._fake_vector(self.rstd_desc, rows)
        main = cute.compile(
            HSTULMSDBackward(
                d,
                apply_u_silu=self.apply_u_silu,
                concat_u=self.concat_u,
                concat_x=self.concat_x,
                has_dropout=self.has_dropout,
                compute_dweight=self.compute_dweight,
                config=self._kernel_config,
            ),
            fake_dy_silu,
            fake_dy_x,
            fake_dy_lmsd,
            fake_x,
            fake_u,
            fake_weight,
            fake_bias,
            fake_mask,
            fake_dx,
            fake_du,
            fake_mean,
            fake_rstd,
            fake_dwp,
            fake_dbp,
            cutlass.Float32(self.dropout_ratio),
            cutlass.Int32(d),
            cutlass.Int32(1),
            cutlass.Int32(self._workspace_rows),
            fake_stream,
            options="--enable-tvm-ffi",
        )

        reduce = cute.compile(
            HSTULMSDGradReduce(self.compute_dweight, config=self._reduce_config),
            fake_dwp,
            fake_dbp,
            fake_dw,
            fake_db,
            cutlass.Int32(self._workspace_rows),
            cutlass.Int32(d),
            fake_stream,
            options="--enable-tvm-ffi",
        )
        self._compiled_kernel = (main, reduce)

    def execute(
        self,
        dy_tensor: torch.Tensor,
        x_tensor: torch.Tensor,
        u_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        bias_tensor: torch.Tensor,
        mean_tensor: torch.Tensor,
        rstd_tensor: torch.Tensor,
        mask_tensor: Optional[torch.Tensor],
        dx_tensor: torch.Tensor,
        du_tensor: torch.Tensor,
        dweight_tensor: Optional[torch.Tensor],
        dbias_tensor: torch.Tensor,
        dweight_workspace: Optional[torch.Tensor],
        dbias_workspace: torch.Tensor,
        current_stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    ) -> None:
        if self._compiled_kernel is None:
            raise RuntimeError("HSTULMSDBwd kernels are not compiled")
        n = self._check_runtime_common(x_tensor, u_tensor, weight_tensor, bias_tensor)
        runtime = [
            (dy_tensor, self.dy_desc, "dy", True, True),
            (mean_tensor, self.mean_desc, "mean", True, False),
            (rstd_tensor, self.rstd_desc, "rstd", True, False),
            (dx_tensor, self.dx_desc, "dx", True, False),
            (du_tensor, self.du_desc, "du", True, False),
            (dbias_tensor, self.dbias_desc, "dbias", False, False),
            (dbias_workspace, self.dbias_workspace_desc, "dbias_workspace", False, False),
        ]
        if self.has_dropout:
            if mask_tensor is None:
                raise ValueError("mask_tensor is required by the compiled dropout configuration")
            runtime.append((mask_tensor, self.mask_desc, "mask", True, False))
        elif mask_tensor is not None:
            raise ValueError("mask_tensor must be None for the compiled no-dropout configuration")
        if self.compute_dweight:
            if dweight_tensor is None or dweight_workspace is None:
                raise ValueError("dweight_tensor and dweight_workspace are required by the compiled configuration")
            runtime.extend(
                (
                    (dweight_tensor, self.dweight_desc, "dweight", False, False),
                    (dweight_workspace, self.dweight_workspace_desc, "dweight_workspace", False, False),
                )
            )
        elif dweight_tensor is not None or dweight_workspace is not None:
            raise ValueError("dweight_tensor and dweight_workspace must be None when compute_dweight is False")
        for tensor, desc, name, dynamic_rows, dynamic_row_stride in runtime:
            _check_runtime_tensor(
                tensor,
                desc,
                name,
                num_rows=n if dynamic_rows else None,
                dynamic_row_stride=dynamic_row_stride,
            )
        writes = [
            ("dx", dx_tensor),
            ("du", du_tensor),
            ("dbias", dbias_tensor),
            ("dbias_workspace", dbias_workspace),
        ]
        if self.compute_dweight:
            writes.extend((("dweight", dweight_tensor), ("dweight_workspace", dweight_workspace)))
        reads = [
            ("dy", dy_tensor),
            ("x", x_tensor),
            ("u", u_tensor),
            ("weight", weight_tensor),
            ("bias", bias_tensor),
            ("mean", mean_tensor),
            ("rstd", rstd_tensor),
        ]
        if self.has_dropout:
            reads.append(("mask", mask_tensor))
        _require_disjoint(
            writes,
            reads,
        )
        stream = stream_handle(current_stream, x_tensor.device)
        main, reduce = self._compiled_kernel
        d = self.hidden_size
        segment = 0
        dy_silu = dy_tensor[:, segment * d : (segment + 1) * d] if self.concat_u else None
        segment += int(self.concat_u)
        dy_x = dy_tensor[:, segment * d : (segment + 1) * d] if self.concat_x else None
        segment += int(self.concat_x)
        dy_lmsd = dy_tensor[:, segment * d : (segment + 1) * d]
        grid_size = self._kernel_config.launch_grid(n, self._multiprocessor_count)
        main(
            dy_silu if dy_silu is not None else dy_lmsd,
            dy_x if dy_x is not None else dy_lmsd,
            dy_lmsd,
            x_tensor,
            u_tensor,
            weight_tensor,
            bias_tensor,
            mask_tensor if mask_tensor is not None else x_tensor,
            dx_tensor,
            du_tensor,
            mean_tensor,
            rstd_tensor,
            dweight_workspace if dweight_workspace is not None else dbias_workspace,
            dbias_workspace,
            cutlass.Float32(self.dropout_ratio),
            cutlass.Int32(d),
            cutlass.Int32(n),
            cutlass.Int32(grid_size),
            stream,
        )
        reduce(
            dweight_workspace if dweight_workspace is not None else dbias_workspace,
            dbias_workspace,
            dweight_tensor if dweight_tensor is not None else dbias_tensor,
            dbias_tensor,
            cutlass.Int32(grid_size),
            cutlass.Int32(d),
            stream,
        )
        record_streams(
            tuple(tensor for tensor, _, _, _, _ in runtime) + (x_tensor, u_tensor, weight_tensor, bias_tensor),
            current_stream,
            x_tensor.device,
        )
