# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""PyTorch FE API for fused RMSNorm + RHT + per-CTA amax."""

import logging
from typing import Optional

from cuda.bindings import driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import ApiBaseTorch, TensorDesc, TupleDict

from .config import (
    best_num_threads,
    pick_rows_per_cta,
    validate_rmsnorm_rht_amax,
)
from .kernel import RMSNormRHTAmaxKernel


class RmsNormRhtAmaxSm100(ApiBaseTorch):
    """Class API for the RMSNorm + RHT + amax kernel."""

    def __init__(
        self,
        sample_x: torch.Tensor,
        sample_w: torch.Tensor,
        sample_o: torch.Tensor,
        sample_amax: torch.Tensor,
        eps: float = 1e-5,
        num_threads: Optional[int] = None,
        rows_per_cta: Optional[int] = None,
    ):
        super().__init__()

        self._warn_experimental_api()

        self.x_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_x, name="sample_x"), 2, "sample_x")
        self.w_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_w, name="sample_w"), 1, "sample_w")
        self.o_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_o, name="sample_o"), 2, "sample_o")
        self.amax_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_amax, name="sample_amax"), 1, "sample_amax")

        self.eps = eps
        self.requested_num_threads = num_threads
        self.requested_rows_per_cta = rows_per_cta
        self.num_threads = None
        self.rows_per_cta = None
        self.n = None

    def check_support(self) -> bool:
        plan = validate_rmsnorm_rht_amax(
            self.x_desc,
            self.w_desc,
            output=self.o_desc,
            amax=self.amax_desc,
            num_threads=self.requested_num_threads,
            rows_per_cta=self.requested_rows_per_cta,
        )

        self._check_tensor_stride(
            self.x_desc,
            stride=(plan.n, 1),
            name="X",
            extra_error_msg="X must be row-major contiguous",
        )
        self._check_tensor_stride(self.w_desc, stride=(1,), name="W", extra_error_msg="W must be contiguous")
        self._check_tensor_stride(
            self.o_desc,
            stride=(plan.n, 1),
            name="O",
            extra_error_msg="O must be row-major contiguous",
        )

        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        major, minor = torch.cuda.get_device_capability(self.x_desc.device)
        compute_capability = major * 10 + minor
        self._runtime_error_if(
            compute_capability < 100,
            f"RmsNormRhtAmaxSm100 requires SM100+, found SM{compute_capability}",
        )

        self.num_threads = plan.num_threads
        self.rows_per_cta = plan.rows_per_cta
        self.n = plan.n
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        kernel = RMSNormRHTAmaxKernel(
            n=self.n,
            num_threads=self.num_threads,
            eps=self.eps,
            rows_per_cta=self.rows_per_cta,
        )

        valid_m = cute.sym_int(divisibility=self.rows_per_cta)

        fake_x_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.x_desc.dtype,
            shape=(valid_m, self.n),
            stride_order=self.x_desc.stride_order,
            dynamic_mode=None,
            divisibility=self.rows_per_cta,
        )
        fake_w_tensor = self._make_fake_cute_tensor_from_desc(self.w_desc, assumed_align=16)
        fake_o_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.o_desc.dtype,
            shape=(valid_m, self.n),
            stride_order=self.o_desc.stride_order,
            dynamic_mode=None,
            divisibility=self.rows_per_cta,
        )
        fake_num_ctas = cute.sym_int()
        fake_amax_tensor = self._make_fake_cute_tensor(
            dtype=self.amax_desc.dtype,
            shape=(fake_num_ctas,),
            stride=self.amax_desc.stride,
            assumed_align=16,
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        compiled_kernel = cute.compile(
            kernel,
            fake_x_tensor,
            fake_w_tensor,
            fake_o_tensor,
            fake_amax_tensor,
            Float32(self.eps),
            fake_stream,
            options="--enable-tvm-ffi",
        )

        def tensor_api(
            x_tensor: torch.Tensor,
            w_tensor: torch.Tensor,
            o_tensor: torch.Tensor,
            amax_tensor: torch.Tensor,
            stream: cuda.CUstream,
        ) -> None:
            compiled_kernel(
                x_tensor,
                w_tensor,
                o_tensor,
                amax_tensor,
                Float32(self.eps),
                stream,
            )

        self._compiled_kernel = tensor_api

    def execute(
        self,
        x_tensor: torch.Tensor,
        w_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        amax_tensor: torch.Tensor,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._runtime_error_if(
            self._compiled_kernel is None,
            "RmsNormRhtAmaxSm100 kernel not compiled; call compile() first",
        )

        x_tensor = self._unpad_tensor_to_ndim(x_tensor, 2, "x_tensor")
        w_tensor = self._unpad_tensor_to_ndim(w_tensor, 1, "w_tensor")
        o_tensor = self._unpad_tensor_to_ndim(o_tensor, 2, "o_tensor")
        amax_tensor = self._unpad_tensor_to_ndim(amax_tensor, 1, "amax_tensor")

        if current_stream is None:
            current_stream = cuda.CUstream(torch.cuda.current_stream(x_tensor.device).cuda_stream)

        self._compiled_kernel(
            x_tensor=x_tensor,
            w_tensor=w_tensor,
            o_tensor=o_tensor,
            amax_tensor=amax_tensor,
            stream=current_stream,
        )


_logger = logging.getLogger(__name__)
_cache_of_RmsNormRhtAmaxSm100Objects = {}


def rmsnorm_rht_amax_wrapper_sm100(
    x_tensor: torch.Tensor,
    w_tensor: torch.Tensor,
    eps: float = 1e-5,
    num_threads: Optional[int] = None,
    rows_per_cta: Optional[int] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper for the RMSNorm + RHT + per-CTA amax kernel."""

    x_tensor = x_tensor.squeeze(-1) if x_tensor.ndim == 3 and x_tensor.shape[-1] == 1 else x_tensor
    w_tensor = w_tensor.squeeze(-1) if w_tensor.ndim == 2 and w_tensor.shape[-1] == 1 else w_tensor

    plan = validate_rmsnorm_rht_amax(
        TensorDesc(dtype=x_tensor.dtype, shape=tuple(x_tensor.shape), name="X"),
        TensorDesc(dtype=w_tensor.dtype, shape=tuple(w_tensor.shape), name="W"),
        num_threads=num_threads,
        rows_per_cta=rows_per_cta,
    )

    o_tensor = torch.empty_like(x_tensor)
    amax_tensor = torch.full(plan.amax_shape, float("-inf"), dtype=torch.float32, device=x_tensor.device)

    cache_key = (
        plan.n,
        x_tensor.dtype,
        w_tensor.dtype,
        o_tensor.dtype,
        tuple(x_tensor.stride()),
        tuple(w_tensor.stride()),
        tuple(o_tensor.stride()),
        eps,
        plan.num_threads,
        plan.rows_per_cta,
    )

    if cache_key in _cache_of_RmsNormRhtAmaxSm100Objects:
        api = _cache_of_RmsNormRhtAmaxSm100Objects[cache_key]
    else:
        api = RmsNormRhtAmaxSm100(
            sample_x=x_tensor,
            sample_w=w_tensor,
            sample_o=o_tensor,
            sample_amax=amax_tensor,
            eps=eps,
            num_threads=plan.num_threads,
            rows_per_cta=plan.rows_per_cta,
        )
        assert api.check_support(), "Unsupported configuration"
        api.compile()
        _cache_of_RmsNormRhtAmaxSm100Objects[cache_key] = api

    api.execute(
        x_tensor=x_tensor,
        w_tensor=w_tensor,
        o_tensor=o_tensor,
        amax_tensor=amax_tensor,
        current_stream=current_stream,
    )

    return TupleDict(o_tensor=o_tensor, amax_tensor=amax_tensor)


def rmsnorm_rht_amax_sm100(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-5,
    num_threads: Optional[int] = None,
    rows_per_cta: Optional[int] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Apply RMSNorm + RHT + amax through the aligned Torch API.

    This additive facade shares its operation name, semantic operands, options,
    and result roles with ``cudnn.jax.rmsnorm_rht_amax_sm100``.  The existing
    ``rmsnorm_rht_amax_wrapper_sm100`` API remains unchanged for compatibility.
    """

    result = rmsnorm_rht_amax_wrapper_sm100(
        x_tensor=x,
        w_tensor=weight,
        eps=eps,
        num_threads=num_threads,
        rows_per_cta=rows_per_cta,
        current_stream=current_stream,
    )
    return TupleDict(output=result["o_tensor"], amax=result["amax_tensor"])


__all__ = [
    "RmsNormRhtAmaxSm100",
    "best_num_threads",
    "pick_rows_per_cta",
    "rmsnorm_rht_amax_sm100",
    "rmsnorm_rht_amax_wrapper_sm100",
]
