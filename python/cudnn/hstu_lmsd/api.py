# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit compile/execute APIs for the HSTU LMSD CuTe DSL kernels."""

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
from .cutedsl._common import ALIGNMENT_BYTES, keep_threshold32, normalize_dropout_ratio
from .cutedsl.cute_dsl_ln_mul_dropout import BLOCKS_PER_SM as FWD_BLOCKS_PER_SM
from .cutedsl.cute_dsl_ln_mul_dropout import ROWS_PER_CTA as FWD_ROWS_PER_CTA
from .cutedsl.cute_dsl_ln_mul_dropout import LnMulDropoutForward
from .cutedsl.cute_dsl_ln_mul_dropout_bwd import (
    MAX_NUM_ROWS,
    TARGET_TILES,
    LnMulDropoutBackward,
    LnMulDropoutGradReduce,
)


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
    desc: TensorDesc,
    name: str,
    *,
    row_stride: Optional[int] = None,
) -> None:
    if desc.ndim != 2:
        raise ValueError(f"{name} must be rank 2, got shape {tuple(desc.shape)}")
    if desc.stride[1] != 1:
        raise ValueError(f"{name} must have a unit innermost stride")
    if row_stride is not None and desc.stride[0] != row_stride:
        raise ValueError(f"{name} row stride must be {row_stride}, got {desc.stride[0]}")
    if desc.stride[0] < desc.shape[1]:
        raise ValueError(f"{name} rows must not overlap")
    if desc.dtype == torch.bfloat16 and desc.stride[0] % 8 != 0:
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
) -> None:
    expected_shape = tuple(desc.shape) if num_rows is None else (num_rows, *desc.shape[1:])
    if expected_shape != tuple(tensor.shape) or tuple(tensor.stride()) != tuple(desc.stride) or tensor.dtype != desc.dtype or tensor.device != desc.device:
        raise ValueError(f"{name} specification changed after compilation")
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

    hidden_size = 512

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
        if not 1 <= self.num_rows <= MAX_NUM_ROWS:
            raise ValueError(f"x row count must be in [1, {MAX_NUM_ROWS}], got {self.num_rows}")
        # Supported matrix size: [N, D], with safe tiled offsets and D = 512.
        if self.hidden_size != 512:
            raise ValueError(f"HSTU LMSD currently supports D=512, got D={self.hidden_size}")
        _require_matrix_layout(x, "x", row_stride=self.hidden_size)
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
        if num_rows > MAX_NUM_ROWS:
            raise ValueError(f"x row count must be in [1, {MAX_NUM_ROWS}], got {num_rows}")
        for tensor, desc, name, dynamic_rows in (
            (x, self.x_desc, "x", True),
            (u, self.u_desc, "u", True),
            (weight, self.weight_desc, "weight", False),
            (bias, self.bias_desc, "bias", False),
        ):
            _check_runtime_tensor(tensor, desc, name, num_rows=num_rows if dynamic_rows else None)
        return num_rows

    def _fake_matrix(self, desc, rows) -> cute.Tensor:
        return self._make_fake_cute_tensor(
            dtype=desc.dtype,
            shape=(rows, desc.shape[1]),
            stride=desc.stride,
            assumed_align=ALIGNMENT_BYTES,
        )

    def _fake_vector(self, desc, length=None) -> cute.Tensor:
        return self._make_fake_cute_tensor(
            dtype=desc.dtype,
            shape=(desc.shape[0] if length is None else length,),
            stride=desc.stride,
            assumed_align=ALIGNMENT_BYTES,
        )


class HSTULMSDFwdSm100(_HSTULMSDBase):
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
        sample_mask: torch.Tensor | TensorDesc,
        eps: float = 1e-6,
        dropout_ratio: float = 0.1,
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
        self.mask_desc = self._make_tensor_desc(sample_mask, name="mask")
        self.eps = float(eps)

    def check_support(self) -> bool:
        if self._is_supported:
            return True
        self._check_common()
        self._threshold = keep_threshold32(self.dropout_ratio)
        n, d = self.num_rows, self.hidden_size
        y = self.y_desc
        mean = self.mean_desc
        rstd = self.rstd_desc
        mask = self.mask_desc
        _require_same_device_desc(
            self.x_desc,
            (("y", y), ("mean", mean), ("rstd", rstd), ("mask", mask)),
        )
        if y.shape != (n, 3 * d) or y.dtype != self.x_desc.dtype:
            raise ValueError(f"y must have shape ({n}, {3 * d}) and dtype {self.x_desc.dtype}")
        _require_matrix_layout(y, "y", row_stride=3 * d)
        for tensor, name in ((mean, "mean"), (rstd, "rstd")):
            _require_vector(tensor, name, n)
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name} must have dtype torch.float32")
        if mask.shape != (n, d) or mask.dtype != torch.int8:
            raise ValueError(f"mask must have shape ({n}, {d}) and dtype torch.int8")
        _require_matrix_layout(mask, "mask", row_stride=d)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(f"eps must be positive and finite, got {self.eps}")
        self._grid_cap = torch.cuda.get_device_properties(self.x_desc.device).multi_processor_count * FWD_BLOCKS_PER_SM
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        rows = cute.sym_int()
        fake_x = self._fake_matrix(self.x_desc, rows)
        fake_u = self._fake_matrix(self.u_desc, rows)
        fake_weight = self._fake_vector(self.weight_desc)
        fake_bias = self._fake_vector(self.bias_desc)
        fake_output_segment = self._make_fake_cute_tensor(
            dtype=self.y_desc.dtype,
            shape=(rows, self.hidden_size),
            stride=self.y_desc.stride,
            assumed_align=ALIGNMENT_BYTES,
        )
        # All three output segments have the same plan-time tensor contract.
        fake_silu_output = fake_x_output = fake_lmsd_output = fake_output_segment
        fake_mask = self._fake_matrix(self.mask_desc, rows)
        fake_mean = self._fake_vector(self.mean_desc, rows)
        fake_rstd = self._fake_vector(self.rstd_desc, rows)
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        self._compiled_kernel = cute.compile(
            LnMulDropoutForward(),
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
        mask_tensor: torch.Tensor,
        seed: int,
        current_stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    ) -> None:
        if self._compiled_kernel is None:
            raise RuntimeError("HSTULMSDFwdSm100 kernel is not compiled")
        n = self._check_runtime_common(x_tensor, u_tensor, weight_tensor, bias_tensor)
        for tensor, desc, name in (
            (y_tensor, self.y_desc, "y"),
            (mean_tensor, self.mean_desc, "mean"),
            (rstd_tensor, self.rstd_desc, "rstd"),
            (mask_tensor, self.mask_desc, "mask"),
        ):
            _check_runtime_tensor(tensor, desc, name, num_rows=n)
        if not -(1 << 63) <= int(seed) < (1 << 63):
            raise ValueError("seed must fit in a signed 64-bit integer")
        _require_disjoint(
            (("y", y_tensor), ("mean", mean_tensor), ("rstd", rstd_tensor), ("mask", mask_tensor)),
            (("x", x_tensor), ("u", u_tensor), ("weight", weight_tensor), ("bias", bias_tensor)),
        )
        stream = stream_handle(current_stream, x_tensor.device)
        d = self.hidden_size
        num_row_blocks = (n + FWD_ROWS_PER_CTA - 1) // FWD_ROWS_PER_CTA
        grid_size = min(num_row_blocks, self._grid_cap)
        num_iterations = (num_row_blocks + grid_size - 1) // grid_size
        silu_output = y_tensor[:, :d]
        x_output = y_tensor[:, d : 2 * d]
        lmsd_output = y_tensor[:, 2 * d :]
        self._compiled_kernel(
            x_tensor,
            u_tensor,
            weight_tensor,
            bias_tensor,
            silu_output,
            x_output,
            lmsd_output,
            mask_tensor,
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
            (x_tensor, u_tensor, weight_tensor, bias_tensor, y_tensor, mean_tensor, rstd_tensor, mask_tensor),
            current_stream,
            x_tensor.device,
        )


class HSTULMSDBwdSm100(_HSTULMSDBase):
    """Explicit LMSD backward; the shipping path does not recompute y."""

    def __init__(
        self,
        sample_dy: torch.Tensor | TensorDesc,
        sample_x: torch.Tensor | TensorDesc,
        sample_u: torch.Tensor | TensorDesc,
        sample_weight: torch.Tensor | TensorDesc,
        sample_bias: torch.Tensor | TensorDesc,
        sample_mean: torch.Tensor | TensorDesc,
        sample_rstd: torch.Tensor | TensorDesc,
        sample_mask: torch.Tensor | TensorDesc,
        sample_dx: torch.Tensor | TensorDesc,
        sample_du: torch.Tensor | TensorDesc,
        sample_dweight: torch.Tensor | TensorDesc,
        sample_dbias: torch.Tensor | TensorDesc,
        sample_dweight_workspace: torch.Tensor | TensorDesc,
        sample_dbias_workspace: torch.Tensor | TensorDesc,
        dropout_ratio: float = 0.1,
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
            "mask": sample_mask,
            "dx": sample_dx,
            "du": sample_du,
            "dweight": sample_dweight,
            "dbias": sample_dbias,
            "dweight_workspace": sample_dweight_workspace,
            "dbias_workspace": sample_dbias_workspace,
        }
        for name, tensor in samples.items():
            setattr(self, f"{name}_desc", self._make_tensor_desc(tensor, name=name))

    def check_support(self) -> bool:
        if self._is_supported:
            return True
        self._check_common()
        n, d = self.num_rows, self.hidden_size
        x = self.x_desc
        tensors = (
            ("dy", self.dy_desc),
            ("mean", self.mean_desc),
            ("rstd", self.rstd_desc),
            ("mask", self.mask_desc),
            ("dx", self.dx_desc),
            ("du", self.du_desc),
            ("dweight", self.dweight_desc),
            ("dbias", self.dbias_desc),
            ("dweight_workspace", self.dweight_workspace_desc),
            ("dbias_workspace", self.dbias_workspace_desc),
        )
        _require_same_device_desc(x, tensors)
        if self.dy_desc.shape != (n, 3 * d) or self.dy_desc.dtype != x.dtype:
            raise ValueError(f"dy must have shape ({n}, {3 * d}) and dtype {x.dtype}")
        _require_matrix_layout(self.dy_desc, "dy")
        for desc, name in ((self.mean_desc, "mean"), (self.rstd_desc, "rstd")):
            _require_vector(desc, name, n)
            if desc.dtype != torch.float32:
                raise ValueError(f"{name} must have dtype torch.float32")
        if self.mask_desc.shape != (n, d) or self.mask_desc.dtype != torch.int8:
            raise ValueError(f"mask must have shape ({n}, {d}) and dtype torch.int8")
        _require_matrix_layout(self.mask_desc, "mask", row_stride=d)
        for desc, name in ((self.dx_desc, "dx"), (self.du_desc, "du")):
            if desc.shape != (n, d) or desc.dtype != x.dtype:
                raise ValueError(f"{name} must have shape ({n}, {d}) and dtype {x.dtype}")
            _require_matrix_layout(desc, name, row_stride=d)
        for desc, name in ((self.dweight_desc, "dweight"), (self.dbias_desc, "dbias")):
            _require_vector(desc, name, d)
            if desc.dtype != x.dtype:
                raise ValueError(f"{name} must have dtype {x.dtype}")
        for desc, name in (
            (self.dweight_workspace_desc, "dweight_workspace"),
            (self.dbias_workspace_desc, "dbias_workspace"),
        ):
            if desc.shape != (TARGET_TILES, d) or desc.dtype != torch.float32:
                raise ValueError(f"{name} must have shape ({TARGET_TILES}, {d}) and dtype torch.float32")
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
        fake_dwp = self._make_fake_cute_tensor_from_desc(self.dweight_workspace_desc, assumed_align=ALIGNMENT_BYTES)
        fake_dbp = self._make_fake_cute_tensor_from_desc(self.dbias_workspace_desc, assumed_align=ALIGNMENT_BYTES)
        fake_dw = self._make_fake_cute_tensor_from_desc(self.dweight_desc, assumed_align=ALIGNMENT_BYTES)
        fake_db = self._make_fake_cute_tensor_from_desc(self.dbias_desc, assumed_align=ALIGNMENT_BYTES)
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        # N is a runtime extent shared by every row-major operand. Compile one
        # full-N kernel for all supported row counts with the same D/layout.
        # Wide-stride dY/U operands use i64 row rebasing; MAX_NUM_ROWS keeps
        # the direct X/mask/DX/DU tiled offsets within signed Int32.
        rows = cute.sym_int()
        fake_dy_segment = self._make_fake_cute_tensor(
            dtype=self.dy_desc.dtype,
            shape=(rows, d),
            stride=self.dy_desc.stride,
            assumed_align=ALIGNMENT_BYTES,
        )
        # All three dY segments have the same plan-time tensor contract.
        fake_dy_silu = fake_dy_x = fake_dy_lmsd = fake_dy_segment
        fake_x = self._fake_matrix(self.x_desc, rows)
        fake_u = self._fake_matrix(self.u_desc, rows)
        fake_mask = self._fake_matrix(self.mask_desc, rows)
        fake_dx = self._fake_matrix(self.dx_desc, rows)
        fake_du = self._fake_matrix(self.du_desc, rows)
        fake_mean = self._fake_vector(self.mean_desc, rows)
        fake_rstd = self._fake_vector(self.rstd_desc, rows)
        main = cute.compile(
            LnMulDropoutBackward(),
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
            cutlass.Int32(TARGET_TILES),
            fake_stream,
            options="--enable-tvm-ffi",
        )

        reduce = cute.compile(
            LnMulDropoutGradReduce(),
            fake_dwp,
            fake_dbp,
            fake_dw,
            fake_db,
            cutlass.Int32(TARGET_TILES),
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
        mask_tensor: torch.Tensor,
        dx_tensor: torch.Tensor,
        du_tensor: torch.Tensor,
        dweight_tensor: torch.Tensor,
        dbias_tensor: torch.Tensor,
        dweight_workspace: torch.Tensor,
        dbias_workspace: torch.Tensor,
        current_stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    ) -> None:
        if self._compiled_kernel is None:
            raise RuntimeError("HSTULMSDBwdSm100 kernels are not compiled")
        n = self._check_runtime_common(x_tensor, u_tensor, weight_tensor, bias_tensor)
        runtime = (
            (dy_tensor, self.dy_desc, "dy", True),
            (mean_tensor, self.mean_desc, "mean", True),
            (rstd_tensor, self.rstd_desc, "rstd", True),
            (mask_tensor, self.mask_desc, "mask", True),
            (dx_tensor, self.dx_desc, "dx", True),
            (du_tensor, self.du_desc, "du", True),
            (dweight_tensor, self.dweight_desc, "dweight", False),
            (dbias_tensor, self.dbias_desc, "dbias", False),
            (dweight_workspace, self.dweight_workspace_desc, "dweight_workspace", False),
            (dbias_workspace, self.dbias_workspace_desc, "dbias_workspace", False),
        )
        for tensor, desc, name, dynamic_rows in runtime:
            _check_runtime_tensor(tensor, desc, name, num_rows=n if dynamic_rows else None)
        _require_disjoint(
            (
                ("dx", dx_tensor),
                ("du", du_tensor),
                ("dweight", dweight_tensor),
                ("dbias", dbias_tensor),
                ("dweight_workspace", dweight_workspace),
                ("dbias_workspace", dbias_workspace),
            ),
            (
                ("dy", dy_tensor),
                ("x", x_tensor),
                ("u", u_tensor),
                ("weight", weight_tensor),
                ("bias", bias_tensor),
                ("mean", mean_tensor),
                ("rstd", rstd_tensor),
                ("mask", mask_tensor),
            ),
        )
        stream = stream_handle(current_stream, x_tensor.device)
        main, reduce = self._compiled_kernel
        d = self.hidden_size
        dy_silu = dy_tensor[:, :d]
        dy_x = dy_tensor[:, d : 2 * d]
        dy_lmsd = dy_tensor[:, 2 * d :]
        main(
            dy_silu,
            dy_x,
            dy_lmsd,
            x_tensor,
            u_tensor,
            weight_tensor,
            bias_tensor,
            mask_tensor,
            dx_tensor,
            du_tensor,
            mean_tensor,
            rstd_tensor,
            dweight_workspace,
            dbias_workspace,
            cutlass.Float32(self.dropout_ratio),
            cutlass.Int32(d),
            cutlass.Int32(n),
            cutlass.Int32(TARGET_TILES),
            stream,
        )
        reduce(
            dweight_workspace,
            dbias_workspace,
            dweight_tensor,
            dbias_tensor,
            cutlass.Int32(TARGET_TILES),
            cutlass.Int32(d),
            stream,
        )
        record_streams(
            tuple(tensor for tensor, _, _, _ in runtime) + (x_tensor, u_tensor, weight_tensor, bias_tensor),
            current_stream,
            x_tensor.device,
        )
