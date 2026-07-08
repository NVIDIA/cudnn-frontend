# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FE API for fused RMSNorm + RHT + per-CTA amax."""

import logging
from typing import Optional

from cuda.bindings import driver as cuda
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import APIBase, TupleDict

from .kernel import RMSNormRHTAmaxKernel
from .op import (
    DEFAULT_NUM_THREADS_BY_N,
    RmsNormRhtAmaxSm100Op,
    best_num_threads,
    pick_rows_per_cta,
)

_TENSOR_ALIGNMENT = RMSNormRHTAmaxKernel.COPY_BITS // 8


class RmsNormRhtAmaxSm100(APIBase):
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

        self._op = RmsNormRhtAmaxSm100Op(
            x=self.x_desc,
            weight=self.w_desc,
            output=self.o_desc,
            amax=self.amax_desc,
            eps=eps,
            num_threads=num_threads,
            rows_per_cta=rows_per_cta,
        )

    def check_support(self) -> bool:
        self._is_supported = False
        self._op.check_support()

        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        major, minor = torch.cuda.get_device_capability(self.x_desc.device)
        compute_capability = major * 10 + minor
        self._runtime_error_if(
            compute_capability < 100,
            f"RmsNormRhtAmaxSm100 requires SM100+, found SM{compute_capability}",
        )

        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        kernel = RMSNormRHTAmaxKernel(
            n=self._op.n,
            num_threads=self._op.num_threads,
            eps=self._op.eps,
            rows_per_cta=self._op.rows_per_cta,
        )

        valid_m = cute.sym_int(divisibility=self._op.rows_per_cta)

        fake_x_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.x_desc.dtype,
            shape=(valid_m, self._op.n),
            stride_order=self.x_desc.stride_order,
            assumed_align=_TENSOR_ALIGNMENT,
            dynamic_mode=None,
            divisibility=self._op.rows_per_cta,
        )
        fake_w_tensor = self._make_fake_cute_tensor_from_desc(self.w_desc, assumed_align=_TENSOR_ALIGNMENT)
        fake_o_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.o_desc.dtype,
            shape=(valid_m, self._op.n),
            stride_order=self.o_desc.stride_order,
            assumed_align=_TENSOR_ALIGNMENT,
            dynamic_mode=None,
            divisibility=self._op.rows_per_cta,
        )
        fake_num_ctas = cute.sym_int()
        fake_amax_tensor = self._make_fake_cute_tensor(
            dtype=self.amax_desc.dtype,
            shape=(fake_num_ctas,),
            stride=self.amax_desc.stride,
            assumed_align=_TENSOR_ALIGNMENT,
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        compiled_kernel = cute.compile(
            kernel,
            fake_x_tensor,
            fake_w_tensor,
            fake_o_tensor,
            fake_amax_tensor,
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
        self._runtime_error_if(self._compiled_kernel is None, "RmsNormRhtAmaxSm100 kernel not compiled; call compile() first")

        x_tensor = self._unpad_tensor_to_ndim(x_tensor, 2, "x_tensor")
        w_tensor = self._unpad_tensor_to_ndim(w_tensor, 1, "w_tensor")
        o_tensor = self._unpad_tensor_to_ndim(o_tensor, 2, "o_tensor")
        amax_tensor = self._unpad_tensor_to_ndim(amax_tensor, 1, "amax_tensor")

        # TVM-FFI validates the compiled tensor ABI. The amax extent is an
        # independent symbol, so its relationship to M is checked here.
        if x_tensor.ndim == 2:
            self._check_tensor_shape(amax_tensor, (x_tensor.shape[0] // self._op.rows_per_cta,), "Amax")

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

    m, n = x_tensor.shape
    resolved_num_threads = num_threads if num_threads is not None else DEFAULT_NUM_THREADS_BY_N.get(n, best_num_threads(n))
    if resolved_num_threads is None:
        raise ValueError(f"No valid num_threads found for N={n}")
    resolved_rows_per_cta = rows_per_cta if rows_per_cta is not None else pick_rows_per_cta(m)
    if resolved_rows_per_cta <= 0:
        raise ValueError(f"rows_per_cta must be positive, got {resolved_rows_per_cta}")
    if m % resolved_rows_per_cta != 0:
        raise ValueError(f"M must be divisible by rows_per_cta, got M={m}, rows_per_cta={resolved_rows_per_cta}")

    o_tensor = torch.empty_like(x_tensor)
    amax_tensor = torch.full((m // resolved_rows_per_cta,), float("-inf"), dtype=torch.float32, device=x_tensor.device)

    cache_key = (
        n,
        x_tensor.dtype,
        w_tensor.dtype,
        o_tensor.dtype,
        tuple(x_tensor.stride()),
        tuple(w_tensor.stride()),
        tuple(o_tensor.stride()),
        x_tensor.device,
        eps,
        resolved_num_threads,
        resolved_rows_per_cta,
    )

    api = _cache_of_RmsNormRhtAmaxSm100Objects.get(cache_key)
    if api is None:
        api = RmsNormRhtAmaxSm100(
            sample_x=x_tensor,
            sample_w=w_tensor,
            sample_o=o_tensor,
            sample_amax=amax_tensor,
            eps=eps,
            num_threads=resolved_num_threads,
            rows_per_cta=resolved_rows_per_cta,
        )
        api.check_support()
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
