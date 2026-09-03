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

from cudnn.api_base import APIBase

from .cutedsl.cute_dsl_ln_mul_dropout import (
    ALIGN,
    LnMulDropoutForward,
    _keep_threshold32,
)
from .cutedsl.cute_dsl_ln_mul_dropout_bwd import (
    TARGET_TILES,
    LnMulDropoutBackward,
    LnMulDropoutGradReduce,
)


def _require_cuda_tensor(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.data_ptr() % ALIGN != 0:
        raise ValueError(f"{name} storage must be {ALIGN}-byte aligned")


def _require_same_device(reference: torch.Tensor, tensors) -> None:
    for name, tensor in tensors:
        _require_cuda_tensor(tensor, name)
        if tensor.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}, got {tensor.device}")


def _require_matrix_layout(
    tensor: torch.Tensor,
    name: str,
    *,
    row_stride: Optional[int] = None,
) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank 2, got shape {tuple(tensor.shape)}")
    if tensor.stride(1) != 1:
        raise ValueError(f"{name} must have a unit innermost stride")
    if row_stride is not None and tensor.stride(0) != row_stride:
        raise ValueError(f"{name} row stride must be {row_stride}, got {tensor.stride(0)}")
    if tensor.stride(0) < tensor.shape[1]:
        raise ValueError(f"{name} rows must not overlap")
    if tensor.element_size() == 2 and tensor.stride(0) % 8 != 0:
        raise ValueError(f"{name} row starts must remain 16-byte aligned")


def _require_vector(tensor: torch.Tensor, name: str, length: int) -> None:
    if tensor.shape != (length,) or tensor.stride() != (1,):
        raise ValueError(f"{name} must be contiguous with shape ({length},), got " f"shape {tuple(tensor.shape)} and stride {tuple(tensor.stride())}")


def _check_runtime_tensor(tensor: torch.Tensor, desc, name: str) -> None:
    if tuple(tensor.shape) != tuple(desc.shape) or tuple(tensor.stride()) != tuple(desc.stride) or tensor.dtype != desc.dtype or tensor.device != desc.device:
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


def _stream_handle(
    stream: Optional[cuda.CUstream | torch.cuda.Stream],
    device: torch.device,
) -> cuda.CUstream:
    if stream is None:
        return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    if isinstance(stream, torch.cuda.Stream):
        if stream.device != device:
            raise ValueError(f"stream must be on {device}, got {stream.device}")
        return cuda.CUstream(stream.cuda_stream)
    return stream


def _record_streams(tensors, stream, device: torch.device) -> None:
    if stream is None:
        return
    if isinstance(stream, torch.cuda.Stream):
        torch_stream = stream
    elif int(stream) == 0:
        torch_stream = torch.cuda.default_stream(device)
    else:
        torch_stream = torch.cuda.ExternalStream(int(stream), device=device)
    for tensor in tensors:
        tensor.record_stream(torch_stream)


class _HSTULMSDBase(APIBase):
    """Validation and dynamic fake-tensor helpers shared by forward/backward."""

    hidden_size = 512

    def _init_common(
        self,
        *,
        sample_x: torch.Tensor,
        sample_u: torch.Tensor,
        sample_weight: torch.Tensor,
        sample_bias: torch.Tensor,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self._warn_experimental_api()
        self._sample_x = sample_x
        self._sample_u = sample_u
        self._sample_weight = sample_weight
        self._sample_bias = sample_bias
        self.x_desc = self._make_tensor_desc(sample_x, name="x")
        self.u_desc = self._make_tensor_desc(sample_u, name="u")
        self.weight_desc = self._make_tensor_desc(sample_weight, name="weight")
        self.bias_desc = self._make_tensor_desc(sample_bias, name="bias")
        self.dropout_ratio = float(dropout_ratio)
        if sample_x.ndim != 2:
            raise ValueError("x must be rank 2")
        self.num_rows = int(sample_x.shape[0])
        self.hidden_size = int(sample_x.shape[1])

    def _check_common(self) -> None:
        x = self._sample_x
        u = self._sample_u
        weight = self._sample_weight
        bias = self._sample_bias
        _require_same_device(
            x,
            (("u", u), ("weight", weight), ("bias", bias)),
        )
        if x.dtype != torch.bfloat16:
            raise ValueError(f"x must have dtype torch.bfloat16, got {x.dtype}")
        if u.dtype != x.dtype or weight.dtype != x.dtype or bias.dtype != x.dtype:
            raise ValueError("x, u, weight, and bias must have the same dtype")
        if x.shape != u.shape:
            raise ValueError(f"u must have shape {tuple(x.shape)}, got {tuple(u.shape)}")
        if self.num_rows <= 0:
            raise ValueError("x must contain at least one row")
        # Supported matrix size: [N, D], with N > 0 and D = 512.
        if self.hidden_size != 512:
            raise ValueError(f"HSTU LMSD currently supports D=512, got D={self.hidden_size}")
        _require_matrix_layout(x, "x", row_stride=self.hidden_size)
        _require_matrix_layout(u, "u")
        _require_vector(weight, "weight", self.hidden_size)
        _require_vector(bias, "bias", self.hidden_size)
        if not math.isfinite(self.dropout_ratio) or not (0.0 <= self.dropout_ratio < 1.0):
            raise ValueError("dropout_ratio must be finite and in [0, 1), got " f"{self.dropout_ratio}")
        major, minor = torch.cuda.get_device_capability(x.device)
        if major != 10:
            raise RuntimeError(f"HSTU LMSD requires an SM10x GPU; found SM{major}{minor}")

    def _check_runtime_common(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        for tensor, desc, name in (
            (x, self.x_desc, "x"),
            (u, self.u_desc, "u"),
            (weight, self.weight_desc, "weight"),
            (bias, self.bias_desc, "bias"),
        ):
            _check_runtime_tensor(tensor, desc, name)

    def _fake_matrix(self, desc, rows) -> cute.Tensor:
        return self._make_fake_cute_tensor(
            dtype=desc.dtype,
            shape=(rows, desc.shape[1]),
            stride=desc.stride,
            assumed_align=ALIGN,
        )

    def _fake_vector(self, desc, length=None) -> cute.Tensor:
        return self._make_fake_cute_tensor(
            dtype=desc.dtype,
            shape=(desc.shape[0] if length is None else length,),
            stride=desc.stride,
            assumed_align=ALIGN,
        )

    def _release_common_samples(self) -> None:
        self._sample_x = None
        self._sample_u = None
        self._sample_weight = None
        self._sample_bias = None


class HSTULMSDFwdSm100(_HSTULMSDBase):
    """Explicit compile/execute API for HSTU LMSD forward on SM10x."""

    def __init__(
        self,
        sample_x: torch.Tensor,
        sample_u: torch.Tensor,
        sample_weight: torch.Tensor,
        sample_bias: torch.Tensor,
        sample_y: torch.Tensor,
        sample_mean: torch.Tensor,
        sample_rstd: torch.Tensor,
        sample_mask: torch.Tensor,
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
        self._sample_y = sample_y
        self._sample_mean = sample_mean
        self._sample_rstd = sample_rstd
        self._sample_mask = sample_mask
        self.y_desc = self._make_tensor_desc(sample_y, name="y")
        self.mean_desc = self._make_tensor_desc(sample_mean, name="mean")
        self.rstd_desc = self._make_tensor_desc(sample_rstd, name="rstd")
        self.mask_desc = self._make_tensor_desc(sample_mask, name="mask")
        self.eps = float(eps)
        self._threshold = _keep_threshold32(self.dropout_ratio)

    def check_support(self) -> bool:
        if self._is_supported:
            return True
        self._check_common()
        n, d = self.num_rows, self.hidden_size
        y = self._sample_y
        mean = self._sample_mean
        rstd = self._sample_rstd
        mask = self._sample_mask
        _require_same_device(
            self._sample_x,
            (("y", y), ("mean", mean), ("rstd", rstd), ("mask", mask)),
        )
        if y.shape != (n, 3 * d) or y.dtype != self._sample_x.dtype:
            raise ValueError(f"y must have shape ({n}, {3 * d}) and dtype {self._sample_x.dtype}")
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
        _require_disjoint(
            (("y", y), ("mean", mean), ("rstd", rstd), ("mask", mask)),
            (
                ("x", self._sample_x),
                ("u", self._sample_u),
                ("weight", self._sample_weight),
                ("bias", self._sample_bias),
            ),
        )
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
        fake_y = self._make_fake_cute_tensor(
            dtype=self.y_desc.dtype,
            shape=(rows, self.hidden_size),
            stride=self.y_desc.stride,
            assumed_align=ALIGN,
        )
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
            fake_y,
            fake_y,
            fake_y,
            fake_mask,
            fake_mean,
            fake_rstd,
            cutlass.Int64(0),
            cutlass.Int32(self.num_rows),
            cutlass.Int32(0),
            cutlass.Float32(self.eps),
            cutlass.Float32(self.dropout_ratio),
            cutlass.Uint32(self._threshold),
            cutlass.Int32(self.hidden_size),
            fake_stream,
            options="--enable-tvm-ffi",
        )
        self._release_common_samples()
        self._sample_y = None
        self._sample_mean = None
        self._sample_rstd = None
        self._sample_mask = None

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
        self._check_runtime_common(x_tensor, u_tensor, weight_tensor, bias_tensor)
        for tensor, desc, name in (
            (y_tensor, self.y_desc, "y"),
            (mean_tensor, self.mean_desc, "mean"),
            (rstd_tensor, self.rstd_desc, "rstd"),
            (mask_tensor, self.mask_desc, "mask"),
        ):
            _check_runtime_tensor(tensor, desc, name)
        if not -(1 << 63) <= int(seed) < (1 << 63):
            raise ValueError("seed must fit in a signed 64-bit integer")
        _require_disjoint(
            (("y", y_tensor), ("mean", mean_tensor), ("rstd", rstd_tensor), ("mask", mask_tensor)),
            (("x", x_tensor), ("u", u_tensor), ("weight", weight_tensor), ("bias", bias_tensor)),
        )
        stream = _stream_handle(current_stream, x_tensor.device)
        d = self.hidden_size
        self._compiled_kernel(
            x_tensor,
            u_tensor,
            weight_tensor,
            bias_tensor,
            y_tensor[:, :d],
            y_tensor[:, d : 2 * d],
            y_tensor[:, 2 * d :],
            mask_tensor,
            mean_tensor,
            rstd_tensor,
            cutlass.Int64(seed),
            cutlass.Int32(self.num_rows),
            cutlass.Int32(0),
            cutlass.Float32(self.eps),
            cutlass.Float32(self.dropout_ratio),
            cutlass.Uint32(self._threshold),
            cutlass.Int32(d),
            stream,
        )
        _record_streams(
            (x_tensor, u_tensor, weight_tensor, bias_tensor, y_tensor, mean_tensor, rstd_tensor, mask_tensor),
            current_stream,
            x_tensor.device,
        )


class HSTULMSDBwdSm100(_HSTULMSDBase):
    """Explicit LMSD backward; the shipping path does not recompute y."""

    def __init__(
        self,
        sample_dy: torch.Tensor,
        sample_x: torch.Tensor,
        sample_u: torch.Tensor,
        sample_weight: torch.Tensor,
        sample_bias: torch.Tensor,
        sample_mean: torch.Tensor,
        sample_rstd: torch.Tensor,
        sample_mask: torch.Tensor,
        sample_dx: torch.Tensor,
        sample_du: torch.Tensor,
        sample_dweight: torch.Tensor,
        sample_dbias: torch.Tensor,
        sample_dweight_workspace: torch.Tensor,
        sample_dbias_workspace: torch.Tensor,
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
            setattr(self, f"_sample_{name}", tensor)
            setattr(self, f"{name}_desc", self._make_tensor_desc(tensor, name=name))

    def check_support(self) -> bool:
        if self._is_supported:
            return True
        self._check_common()
        n, d = self.num_rows, self.hidden_size
        x = self._sample_x
        tensors = (
            ("dy", self._sample_dy),
            ("mean", self._sample_mean),
            ("rstd", self._sample_rstd),
            ("mask", self._sample_mask),
            ("dx", self._sample_dx),
            ("du", self._sample_du),
            ("dweight", self._sample_dweight),
            ("dbias", self._sample_dbias),
            ("dweight_workspace", self._sample_dweight_workspace),
            ("dbias_workspace", self._sample_dbias_workspace),
        )
        _require_same_device(x, tensors)
        if self._sample_dy.shape != (n, 3 * d) or self._sample_dy.dtype != x.dtype:
            raise ValueError(f"dy must have shape ({n}, {3 * d}) and dtype {x.dtype}")
        _require_matrix_layout(self._sample_dy, "dy")
        for tensor, name in ((self._sample_mean, "mean"), (self._sample_rstd, "rstd")):
            _require_vector(tensor, name, n)
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name} must have dtype torch.float32")
        if self._sample_mask.shape != (n, d) or self._sample_mask.dtype != torch.int8:
            raise ValueError(f"mask must have shape ({n}, {d}) and dtype torch.int8")
        _require_matrix_layout(self._sample_mask, "mask", row_stride=d)
        for tensor, name in ((self._sample_dx, "dx"), (self._sample_du, "du")):
            if tensor.shape != (n, d) or tensor.dtype != x.dtype:
                raise ValueError(f"{name} must have shape ({n}, {d}) and dtype {x.dtype}")
            _require_matrix_layout(tensor, name, row_stride=d)
        for tensor, name in ((self._sample_dweight, "dweight"), (self._sample_dbias, "dbias")):
            _require_vector(tensor, name, d)
            if tensor.dtype != x.dtype:
                raise ValueError(f"{name} must have dtype {x.dtype}")
        for tensor, name in (
            (self._sample_dweight_workspace, "dweight_workspace"),
            (self._sample_dbias_workspace, "dbias_workspace"),
        ):
            if tensor.shape != (TARGET_TILES, d) or tensor.dtype != torch.float32:
                raise ValueError(f"{name} must have shape ({TARGET_TILES}, {d}) and dtype torch.float32")
            _require_matrix_layout(tensor, name, row_stride=d)
        writes = (
            ("dx", self._sample_dx),
            ("du", self._sample_du),
            ("dweight", self._sample_dweight),
            ("dbias", self._sample_dbias),
            ("dweight_workspace", self._sample_dweight_workspace),
            ("dbias_workspace", self._sample_dbias_workspace),
        )
        reads = (
            ("dy", self._sample_dy),
            ("x", x),
            ("u", self._sample_u),
            ("weight", self._sample_weight),
            ("bias", self._sample_bias),
            ("mean", self._sample_mean),
            ("rstd", self._sample_rstd),
            ("mask", self._sample_mask),
        )
        _require_disjoint(writes, reads)
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        d = self.hidden_size
        fake_weight = self._fake_vector(self.weight_desc)
        fake_bias = self._fake_vector(self.bias_desc)
        fake_dwp = self._make_fake_cute_tensor_from_desc(self.dweight_workspace_desc, assumed_align=ALIGN)
        fake_dbp = self._make_fake_cute_tensor_from_desc(self.dbias_workspace_desc, assumed_align=ALIGN)
        fake_dw = self._make_fake_cute_tensor_from_desc(self.dweight_desc, assumed_align=ALIGN)
        fake_db = self._make_fake_cute_tensor_from_desc(self.dbias_desc, assumed_align=ALIGN)
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        # N is fixed by this API object's runtime descriptors. Compile one
        # concrete full-N kernel; device-side i64 rebasing handles tensors
        # whose addressable span exceeds 4 GiB.
        n = self.num_rows
        fake_dy = self._make_fake_cute_tensor(
            dtype=self.dy_desc.dtype,
            shape=(n, d),
            stride=self.dy_desc.stride,
            assumed_align=ALIGN,
        )
        fake_x = self._fake_matrix(self.x_desc, n)
        fake_u = self._fake_matrix(self.u_desc, n)
        fake_mask = self._fake_matrix(self.mask_desc, n)
        fake_dx = self._fake_matrix(self.dx_desc, n)
        fake_du = self._fake_matrix(self.du_desc, n)
        fake_mean = self._fake_vector(self.mean_desc, n)
        fake_rstd = self._fake_vector(self.rstd_desc, n)
        main = cute.compile(
            LnMulDropoutBackward(compute_y=False),
            fake_dy,
            fake_dy,
            fake_dy,
            fake_x,
            fake_u,
            fake_weight,
            fake_bias,
            fake_mask,
            fake_dx,
            fake_du,
            fake_dx,
            fake_dx,
            fake_dx,
            fake_mean,
            fake_rstd,
            fake_dwp,
            fake_dbp,
            cutlass.Float32(self.dropout_ratio),
            cutlass.Int32(d),
            cutlass.Int32(n),
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
        self._release_common_samples()
        for name in (
            "dy",
            "mean",
            "rstd",
            "mask",
            "dx",
            "du",
            "dweight",
            "dbias",
            "dweight_workspace",
            "dbias_workspace",
        ):
            setattr(self, f"_sample_{name}", None)

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
        self._check_runtime_common(x_tensor, u_tensor, weight_tensor, bias_tensor)
        runtime = (
            (dy_tensor, self.dy_desc, "dy"),
            (mean_tensor, self.mean_desc, "mean"),
            (rstd_tensor, self.rstd_desc, "rstd"),
            (mask_tensor, self.mask_desc, "mask"),
            (dx_tensor, self.dx_desc, "dx"),
            (du_tensor, self.du_desc, "du"),
            (dweight_tensor, self.dweight_desc, "dweight"),
            (dbias_tensor, self.dbias_desc, "dbias"),
            (dweight_workspace, self.dweight_workspace_desc, "dweight_workspace"),
            (dbias_workspace, self.dbias_workspace_desc, "dbias_workspace"),
        )
        for tensor, desc, name in runtime:
            _check_runtime_tensor(tensor, desc, name)
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
        stream = _stream_handle(current_stream, x_tensor.device)
        main, reduce = self._compiled_kernel
        d = self.hidden_size
        n = self.num_rows
        main(
            dy_tensor[:, :d],
            dy_tensor[:, d : 2 * d],
            dy_tensor[:, 2 * d :],
            x_tensor,
            u_tensor,
            weight_tensor,
            bias_tensor,
            mask_tensor,
            dx_tensor,
            du_tensor,
            dx_tensor,
            dx_tensor,
            dx_tensor,
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
        _record_streams(
            tuple(tensor for tensor, _, _ in runtime) + (x_tensor, u_tensor, weight_tensor, bias_tensor),
            current_stream,
            x_tensor.device,
        )
