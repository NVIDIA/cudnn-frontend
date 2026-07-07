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

from .kernel import (
    DEFAULT_NUM_THREADS_BY_N,
    RPC_CANDIDATES,
    TARGET_MIN_CTAS,
    RMSNormRHTAmaxKernel,
    best_num_threads,
    pick_rows_per_cta,
)

_TENSOR_ALIGNMENT = RMSNormRHTAmaxKernel.TENSOR_ALIGNMENT


def _validate_torch_tensors(*named_tensors: tuple[str, torch.Tensor]) -> None:
    """Validate framework properties that are absent from TensorDesc."""

    if not named_tensors:
        return
    expected_device = named_tensors[0][1].device
    for name, tensor in named_tensors:
        if tensor.device != expected_device:
            raise ValueError(f"{name} must be on {expected_device}, got {tensor.device}")
        if tensor.data_ptr() % _TENSOR_ALIGNMENT != 0:
            raise ValueError(f"{name} must be {_TENSOR_ALIGNMENT}-byte aligned")


class RmsNormRhtAmaxSm100(APIBase):
    """Class API for the RMSNorm + RHT + amax kernel.

    Logical validation, output inference, and resolved launch state are owned
    by :attr:`kernel`. Output samples remain optional for callers that want
    the framework adapter to allocate inferred outputs.
    """

    def __init__(
        self,
        sample_x: torch.Tensor,
        sample_w: torch.Tensor,
        sample_o: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
        num_threads: Optional[int] = None,
        rows_per_cta: Optional[int] = None,
    ):
        super().__init__()

        self._warn_experimental_api()
        if (sample_o is None) != (sample_amax is None):
            raise ValueError("sample_o and sample_amax must either both be provided or both be omitted")

        named_tensors = [("sample_x", sample_x), ("sample_w", sample_w)]
        if sample_o is not None and sample_amax is not None:
            named_tensors.extend((("sample_o", sample_o), ("sample_amax", sample_amax)))
        _validate_torch_tensors(*named_tensors)

        self.x_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_x, name="sample_x"), 2, "sample_x")
        self.w_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_w, name="sample_w"), 1, "sample_w")
        self.o_desc = None if sample_o is None else self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_o, name="sample_o"), 2, "sample_o")
        self.amax_desc = None if sample_amax is None else self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_amax, name="sample_amax"), 1, "sample_amax")

        self.kernel = RMSNormRHTAmaxKernel(
            x=self.x_desc,
            weight=self.w_desc,
            output=self.o_desc,
            amax=self.amax_desc,
            eps=eps,
            num_threads=num_threads,
            rows_per_cta=rows_per_cta,
        )

    @property
    def eps(self) -> float:
        return self.kernel.eps

    @eps.setter
    def eps(self, value: float) -> None:
        self.kernel.eps = value
        self._is_supported = False
        self._compiled_kernel = None

    @property
    def requested_num_threads(self) -> Optional[int]:
        return self.kernel.requested_num_threads

    @requested_num_threads.setter
    def requested_num_threads(self, value: Optional[int]) -> None:
        self.kernel.requested_num_threads = value
        self._is_supported = False
        self._compiled_kernel = None

    @property
    def requested_rows_per_cta(self) -> Optional[int]:
        return self.kernel.requested_rows_per_cta

    @requested_rows_per_cta.setter
    def requested_rows_per_cta(self, value: Optional[int]) -> None:
        self.kernel.requested_rows_per_cta = value
        self._is_supported = False
        self._compiled_kernel = None

    @property
    def num_threads(self) -> Optional[int]:
        return self.kernel.num_threads

    @property
    def rows_per_cta(self) -> Optional[int]:
        return self.kernel.rows_per_cta

    @property
    def n(self) -> Optional[int]:
        return self.kernel.n

    def _materialize_outputs(
        self,
        *,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer and allocate outputs using this API's kernel instance."""

        self._ensure_support_checked()
        output_desc, amax_desc = self.kernel.infer_output()
        o_tensor = self._materialize_tensor_desc(output_desc, device=self.x_desc.device, stream=current_stream)
        amax_tensor = self._materialize_tensor_desc(amax_desc, device=self.x_desc.device, stream=current_stream)

        if self.o_desc is None and self.amax_desc is None:
            self.o_desc = self._to_tensor_desc(o_tensor, "sample_o")
            self.amax_desc = self._to_tensor_desc(amax_tensor, "sample_amax")
            self.kernel.output = self.o_desc
            self.kernel.amax = self.amax_desc

        return o_tensor, amax_tensor

    def check_support(self) -> bool:
        self._is_supported = False
        self.kernel.check_support()

        for name, desc in (("W", self.w_desc), ("O", self.o_desc), ("Amax", self.amax_desc)):
            if desc is not None:
                self._value_error_if(desc.device != self.x_desc.device, f"{name} must be on {self.x_desc.device}, got {desc.device}")

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
        self._runtime_error_if(self.o_desc is None or self.amax_desc is None, "Output tensors must be provided or materialized before compile()")

        valid_m = cute.sym_int(divisibility=self.kernel.rows_per_cta)

        fake_x_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.x_desc.dtype,
            shape=(valid_m, self.kernel.n),
            stride_order=self.x_desc.stride_order,
            assumed_align=_TENSOR_ALIGNMENT,
            dynamic_mode=None,
            divisibility=self.kernel.rows_per_cta,
        )
        fake_w_tensor = self._make_fake_cute_tensor_from_desc(self.w_desc, assumed_align=_TENSOR_ALIGNMENT)
        fake_o_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.o_desc.dtype,
            shape=(valid_m, self.kernel.n),
            stride_order=self.o_desc.stride_order,
            assumed_align=_TENSOR_ALIGNMENT,
            dynamic_mode=None,
            divisibility=self.kernel.rows_per_cta,
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
            self.kernel,
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
        _validate_torch_tensors(
            ("x_tensor", x_tensor),
            ("w_tensor", w_tensor),
            ("o_tensor", o_tensor),
            ("amax_tensor", amax_tensor),
        )

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
    _validate_torch_tensors(("x_tensor", x_tensor), ("w_tensor", w_tensor))

    cache_key = (
        tuple(x_tensor.shape),
        tuple(w_tensor.shape),
        x_tensor.dtype,
        w_tensor.dtype,
        tuple(x_tensor.stride()),
        tuple(w_tensor.stride()),
        x_tensor.device,
        eps,
        num_threads,
        rows_per_cta,
    )

    api = _cache_of_RmsNormRhtAmaxSm100Objects.get(cache_key)
    cache_miss = api is None
    if cache_miss:
        api = RmsNormRhtAmaxSm100(
            sample_x=x_tensor,
            sample_w=w_tensor,
            eps=eps,
            num_threads=num_threads,
            rows_per_cta=rows_per_cta,
        )
        api.check_support()

    o_tensor, amax_tensor = api._materialize_outputs(current_stream=current_stream)
    if cache_miss:
        api.compile()
        _cache_of_RmsNormRhtAmaxSm100Objects[cache_key] = api

    if current_stream is None and x_tensor.is_cuda:
        current_stream = cuda.CUstream(torch.cuda.current_stream(x_tensor.device).cuda_stream)

    api.execute(
        x_tensor=x_tensor,
        w_tensor=w_tensor,
        o_tensor=o_tensor,
        amax_tensor=amax_tensor,
        current_stream=current_stream,
    )

    return TupleDict(o_tensor=o_tensor, amax_tensor=amax_tensor)
