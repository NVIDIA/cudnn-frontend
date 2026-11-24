# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private BF16 API for the SM100 grouped GEMM + weighted squared-ReLU backward (dSReLU) kernel."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import make_fake_stream, make_ptr

from cudnn.api_base import APIBase
from cudnn._torch_stream import device_context
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.frost.compiled_cache import dsl_compile_lock
from cudnn.frost.workspace import Workspace
from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor
from cudnn.tensor_adapter import (
    default_stream,
    detect_framework,
    get_data_ptr,
    get_device,
    get_shape,
    is_torch_tensor,
)

from ..backend_utils import debug_validate_offsets, debug_validate_pointer_values, validate_workspace_aliases
from ..canonical import validate_once
from ..moe_utils import MoEWeightMode
from ..srelu._bf16_api import Bf16SreluLaunchMixin, check_bf16_srelu_device, mn1_view, row_scalar_view
from ..srelu._bf16_options import resolve_bf16_srelu_options
from ._blockscaled_api import _dprob_n_slots
from ._dprob_fold import compile_dprob_fold, fold_dprob_slots
from .moe_grouped_gemm_dsrelu import MoEGroupedGemmDsreluBf16Kernel

# TMA extents and the batch-mode strides of the compact operands are 32-bit in this kernel family.
_INT32_LIMIT = 2**31


class GroupedGemmDsreluBf16API(Bf16SreluLaunchMixin, APIBase):
    """Descriptor-first lifecycle API for the BF16 grouped GEMM + weighted squared-ReLU backward.

    ``dH = A @ B_e^T`` (FP32, never rounded) fused with the derivative of ``D = p * srelu(C)``
    evaluated at the saved BF16 pre-activation ``C``: ``d_row = bf16(dH * p * srelu'(C))``,
    ``dprob += sum_n srelu(C) * dH`` and, optionally, ``d_srelu = bf16(p * srelu(C))``.
    ``tanh_clamp_scale`` and ``tanh_mode`` must be the forward's: the kernel recomputes ``t`` and
    ``srelu(C)`` from ``C``.

    The API owns no device memory: ``execute()`` carves the kernel's descriptor and scheduler
    scratch from the caller's ``workspace`` (``scratch_workspace_bytes()`` bytes). The
    deterministic dprob slots are a separate caller buffer (``sample_dprob_workspace``).
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d_row: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_prob: torch.Tensor,
        sample_dprob: torch.Tensor,
        sample_b: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        sample_d_srelu: Optional[torch.Tensor] = None,
        sample_dprob_workspace: Optional[torch.Tensor] = None,
        acc_dtype: Optional[torch.dtype] = None,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        m_aligned: int = 256,
        b_major: str = "k",
        use_dynamic_sched: bool = False,
        deterministic: bool = False,
        tanh_clamp_scale: Optional[float] = None,
        tanh_mode: str = "approx",
    ) -> None:
        if acc_dtype is None:
            acc_dtype = cutlass.Float32
        super().__init__()
        self._warn_experimental_api()
        self._framework = detect_framework(sample_a)
        if self._framework != "torch":
            raise ValueError(f"GroupedGemmDsreluSm100 BF16 supports torch tensors only, got {self._framework}")

        if sample_b is not None and num_experts is None:
            self.weight_mode = MoEWeightMode.DENSE
        elif sample_b is None and num_experts is not None:
            self.weight_mode = MoEWeightMode.DISCRETE
            if b_shape is None or b_dtype is None:
                raise ValueError("b_shape and b_dtype are required in discrete mode")
        else:
            raise ValueError("Provide sample_b for dense mode or (num_experts, b_shape, b_dtype) for discrete mode, but not both")

        self._init_launch_state()
        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a", canonical=True)
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c", canonical=True)
        self.d_row_desc = self._make_tensor_desc(sample_d_row, name="sample_d_row", canonical=True)
        self.d_srelu_desc = self._make_tensor_desc(sample_d_srelu, name="sample_d_srelu", canonical=True)
        self.padded_offsets_desc = self._make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets", canonical=True)
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob", canonical=True)
        self.dprob_desc = self._make_tensor_desc(sample_dprob, name="sample_dprob", canonical=True)
        self.dprob_workspace_desc = self._make_tensor_desc(sample_dprob_workspace, name="sample_dprob_workspace", canonical=True)
        self.b_desc = self._make_tensor_desc(sample_b, name="sample_b", canonical=True)

        self._sample_data_ptrs = {
            name: get_data_ptr(tensor)
            for name, tensor in (
                ("sample_a", sample_a),
                ("sample_b", sample_b),
                ("sample_c", sample_c),
                ("sample_d_row", sample_d_row),
                ("sample_d_srelu", sample_d_srelu),
                ("sample_padded_offsets", sample_padded_offsets),
                ("sample_prob", sample_prob),
                ("sample_dprob", sample_dprob),
                ("sample_dprob_workspace", sample_dprob_workspace),
            )
            if tensor is not None
        }

        self.expert_cnt = self.b_desc.shape[2] if self.weight_mode == MoEWeightMode.DENSE and self.b_desc.ndim == 3 else int(num_experts or 0)
        self.b_shape = tuple(b_shape) if b_shape is not None else None
        self.b_dtype = _convert_to_cutlass_data_type(b_dtype) if b_dtype is not None else self.b_desc.dtype
        self.acc_dtype = _convert_to_cutlass_data_type(acc_dtype)
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.use_2cta_instrs = self.mma_tiler_mn[0] == 256
        self.cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if self.use_2cta_instrs else (1, 1)))
        self.m_aligned = m_aligned
        self.b_major = b_major
        self.use_dynamic_sched = use_dynamic_sched
        self.deterministic = bool(deterministic)
        self.tanh_clamp_scale = tanh_clamp_scale
        self.tanh_mode = tanh_mode
        self.generate_d_srelu = sample_d_srelu is not None
        self._dense_b_major = None
        self._kernel = MoEGroupedGemmDsreluBf16Kernel
        self._raw_compiled = None
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))

    def _n(self) -> int:
        return self.b_desc.shape[0] if self.weight_mode == MoEWeightMode.DENSE else self.b_shape[0]

    def _dprob_slots(self) -> int:
        """Columns of the buffer the kernel writes dprob into: one per N-tile slot, or one (atomic)."""
        return _dprob_n_slots(self._n(), self.mma_tiler_mn, self.cluster_shape_mn, self.deterministic)

    def check_support(self) -> bool:
        for desc, name in (
            (self.c_desc, "sample_c"),
            (self.d_row_desc, "sample_d_row"),
            (self.padded_offsets_desc, "sample_padded_offsets"),
            (self.prob_desc, "sample_prob"),
            (self.dprob_desc, "sample_dprob"),
        ):
            if desc is None:
                raise ValueError(f"{name} is required by the BF16 backend")
        if self.a_desc.ndim != 3:
            raise ValueError(f"sample_a must be rank-3 (m, k, 1), got {self.a_desc.shape}")
        tensor_m, k, one = self.a_desc.shape
        if one != 1:
            raise ValueError(f"sample_a trailing dimension must be 1, got {one}")

        if self.weight_mode == MoEWeightMode.DENSE:
            if self.b_desc.ndim != 3:
                raise ValueError(f"sample_b must be rank-3 (n, k, experts), got {self.b_desc.shape}")
            n, b_k, experts = self.b_desc.shape
            if b_k != k:
                raise ValueError(f"sample_b K dimension ({b_k}) must match sample_a ({k})")
            if experts != self.expert_cnt:
                raise ValueError("sample_b expert dimension is inconsistent")
            for major, stride in (("k", (k, 1, n * k)), ("n", (1, n, n * k))):
                try:
                    self._expect_stride(self.b_desc, stride, "sample_b")
                except ValueError:
                    continue
                self._dense_b_major = major
                break
            else:
                raise ValueError(f"sample_b must be K-major with stride {(k, 1, n * k)} or N-major with stride {(1, n, n * k)}, got {self.b_desc.stride}")
        else:
            if len(self.b_shape) not in (2, 3):
                raise ValueError(f"b_shape must be rank-2 or rank-3, got {self.b_shape}")
            n, b_k = self.b_shape[:2]
            if len(self.b_shape) == 3 and self.b_shape[2] != 1:
                raise ValueError(f"b_shape trailing dimension must be 1, got {self.b_shape}")
            if b_k != k:
                raise ValueError(f"b_shape K dimension ({b_k}) must match sample_a ({k})")
        if n <= 0 or n % 32 != 0:
            raise ValueError(f"N must be a positive multiple of 32 (the epilogue's 32-column subtiles), got {n}")
        if k <= 0 or k % 8 != 0:
            raise ValueError(f"K must be a positive multiple of 8 (16-byte TMA rows), got {k}")
        slots = self._dprob_slots()

        self._expect_shape(self.c_desc, (tensor_m, n, 1), "sample_c")
        self._expect_shape(self.d_row_desc, (tensor_m, n, 1), "sample_d_row")
        self._expect_shape(self.padded_offsets_desc, (self.expert_cnt,), "sample_padded_offsets")
        self._expect_shape(self.prob_desc, (tensor_m, 1, 1), "sample_prob")
        self._expect_shape(self.dprob_desc, (tensor_m, 1, 1), "sample_dprob")
        self._expect_stride(self.a_desc, (k, 1, tensor_m * k), "sample_a")
        self._expect_stride(self.c_desc, (n, 1, tensor_m * n), "sample_c")
        self._expect_stride(self.d_row_desc, (n, 1, tensor_m * n), "sample_d_row")
        self._expect_stride(self.padded_offsets_desc, (1,), "sample_padded_offsets")
        self._expect_stride(self.prob_desc, (1, 1, 1), "sample_prob")
        self._expect_stride(self.dprob_desc, (1, 1, 1), "sample_dprob")
        if self.d_srelu_desc is not None:
            self._expect_shape(self.d_srelu_desc, (tensor_m, n, 1), "sample_d_srelu")
            self._expect_stride(self.d_srelu_desc, (n, 1, tensor_m * n), "sample_d_srelu")
            self._check_dtype(self.d_srelu_desc, cutlass.BFloat16, "sample_d_srelu")
        if self.deterministic:
            if self.dprob_workspace_desc is None:
                raise ValueError(
                    "sample_dprob_workspace is required when deterministic=True: shape dprob_workspace_shape(valid_m, n), float32, zero-initialized"
                )
            self._expect_shape(self.dprob_workspace_desc, (tensor_m, slots, 1), "sample_dprob_workspace")
            self._expect_stride(self.dprob_workspace_desc, (slots, 1, tensor_m * slots), "sample_dprob_workspace")
            self._check_dtype(self.dprob_workspace_desc, cutlass.Float32, "sample_dprob_workspace")
        elif self.dprob_workspace_desc is not None:
            raise ValueError("sample_dprob_workspace must be omitted unless deterministic=True")

        self._check_dtype(self.a_desc, cutlass.BFloat16, "sample_a")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(self.b_desc, cutlass.BFloat16, "sample_b")
        self._check_dtype(self.b_dtype, cutlass.BFloat16, "b_dtype")
        self._check_dtype(self.c_desc, cutlass.BFloat16, "sample_c")
        self._check_dtype(self.d_row_desc, cutlass.BFloat16, "sample_d_row")
        self._check_dtype(self.padded_offsets_desc, cutlass.Int32, "sample_padded_offsets")
        self._check_dtype(self.prob_desc, cutlass.Float32, "sample_prob")
        self._check_dtype(self.dprob_desc, cutlass.Float32, "sample_dprob")

        device = self.a_desc.device
        for desc, name in (
            (self.b_desc, "sample_b"),
            (self.c_desc, "sample_c"),
            (self.d_row_desc, "sample_d_row"),
            (self.d_srelu_desc, "sample_d_srelu"),
            (self.padded_offsets_desc, "sample_padded_offsets"),
            (self.prob_desc, "sample_prob"),
            (self.dprob_desc, "sample_dprob"),
            (self.dprob_workspace_desc, "sample_dprob_workspace"),
        ):
            if desc is not None:
                self._expect_device(desc, device, name)
        for name, data_ptr in self._sample_data_ptrs.items():
            if data_ptr % 16 != 0:
                raise ValueError(f"{name} data pointer must be 16-byte aligned")

        if self.acc_dtype is not cutlass.Float32:
            raise ValueError(f"acc_dtype must be torch.float32, got {self.acc_dtype}")
        if self.m_aligned != MoEGroupedGemmDsreluBf16Kernel.FIX_PAD_SIZE:
            raise ValueError(f"m_aligned must be {MoEGroupedGemmDsreluBf16Kernel.FIX_PAD_SIZE}, got {self.m_aligned}")
        if self.b_major not in ("k", "n"):
            raise ValueError(f"b_major must be 'k' or 'n', got {self.b_major}")
        if self.expert_cnt <= 0 or self.expert_cnt > 1024:
            raise ValueError(f"expert count must be in [1, 1024], got {self.expert_cnt}")
        if tensor_m % MoEGroupedGemmDsreluBf16Kernel.FIX_PAD_SIZE != 0:
            raise ValueError(f"sample_a M dimension must be a multiple of {MoEGroupedGemmDsreluBf16Kernel.FIX_PAD_SIZE}, got {tensor_m}")
        if tensor_m * max(n, k, slots) >= _INT32_LIMIT or self.expert_cnt * n * k >= _INT32_LIMIT:
            raise ValueError(f"M*N, M*K, M*slots and E*N*K must each be below 2**31 elements, got M={tensor_m}, N={n}, K={k}, E={self.expert_cnt}")
        self.tanh_clamp_scale, self.tanh_mode = resolve_bf16_srelu_options(self.tanh_clamp_scale, self.tanh_mode)

        if not self._kernel.can_implement(
            self.use_2cta_instrs, self.mma_tiler_mn, self.cluster_shape_mn, max(tensor_m, MoEGroupedGemmDsreluBf16Kernel.FIX_PAD_SIZE), n, k, self.expert_cnt
        ):
            raise ValueError(
                f"Unsupported BF16 grouped GEMM dSReLU tile/cluster configuration mma_tiler_mn={self.mma_tiler_mn}, cluster_shape_mn={self.cluster_shape_mn}"
            )

        check_bf16_srelu_device("GroupedGemmDsreluSm100", self.a_desc.device.index)
        self._is_supported = True
        return True

    def _kernel_instance(self):
        if self._kernel_obj is None:
            self._kernel_obj = self._kernel(
                acc_dtype=cutlass.Float32,
                use_2cta_instrs=self.use_2cta_instrs,
                mma_tiler_mn=self.mma_tiler_mn,
                cluster_shape_mn=self.cluster_shape_mn,
                expert_cnt=self.expert_cnt,
                clamp_scale=self.tanh_clamp_scale,
                tanh_mode=self.tanh_mode,
                generate_h=self.generate_d_srelu,
                weight_mode=self.weight_mode,
                use_dynamic_sched=self.use_dynamic_sched,
                dprob_atomic=not self.deterministic,
            )
        return self._kernel_obj

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        kernel = self._kernel_instance()

        # Compile-time stand-ins carry type and alignment only (Rule 8: compile() owns no device
        # memory); the live addresses arrive at execute as plain ints.
        gmem = cute.AddressSpace.gmem
        workspace_fake = make_ptr(cutlass.Uint8, 128, gmem, assumed_align=128)

        _, k, _ = self.a_desc.shape
        n = self._n()
        e = self.expert_cnt

        def compact(dtype, cols):
            # (M, cols, 1) with strides (cols, 1, M * cols); M symbolic and a multiple of 256.
            return cute.runtime.make_fake_compact_tensor(
                dtype=dtype,
                shape=(cute.sym_int(divisibility=MoEGroupedGemmDsreluBf16Kernel.FIX_PAD_SIZE), cols, 1),
                stride_order=(1, 0, 2),
                assumed_align=16,
            )

        a_fake = compact(cutlass.BFloat16, k)
        c_fake = compact(cutlass.BFloat16, n)
        d_fake = compact(cutlass.BFloat16, n)
        h_fake = compact(cutlass.BFloat16, n) if self.generate_d_srelu else None
        valid_m = cute.sym_int(divisibility=MoEGroupedGemmDsreluBf16Kernel.FIX_PAD_SIZE)
        prob_fake = cute.runtime.make_fake_tensor(dtype=cutlass.Float32, shape=(valid_m, 1, 1), stride=(1, 1, 1), assumed_align=16)
        dprob_fake = cute.runtime.make_fake_compact_tensor(
            dtype=cutlass.Float32,
            shape=(valid_m, self._dprob_slots(), 1),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )
        offsets_fake = cute.runtime.make_fake_tensor(dtype=cutlass.Int32, shape=(e,), stride=(1,), assumed_align=16)

        if self.weight_mode == MoEWeightMode.DISCRETE:
            b_fake = make_ptr(cutlass.Int64, 16, gmem, assumed_align=8)
            n_value = cutlass.Int32(n)
            k_value = cutlass.Int32(k)
            b_stride = cutlass.Int64(k if self.b_major == "k" else n)
            b_major_mode = OperandMajorMode.K if self.b_major == "k" else OperandMajorMode.MN
        else:
            b_stride_dense = (k, 1, n * k) if self._dense_b_major == "k" else (1, n, n * k)
            b_fake = cute.runtime.make_fake_tensor(dtype=cutlass.BFloat16, shape=(n, k, e), stride=b_stride_dense, assumed_align=16)
            n_value = cutlass.Int32(0)
            k_value = cutlass.Int32(0)
            b_stride = cutlass.Int64(0)
            b_major_mode = OperandMajorMode.K  # ignored in dense mode (derived from the tensor)

        # Under the operands' device whichever device is current (Rule 4: a compiled launchable
        # belongs to the CUDA context it is compiled in, and the occupancy query runs in that context
        # too), and under the process-wide DSL compile lock, as compile_dprob_fold does: the
        # occupancy query compiles a probe kernel, and concurrent traces corrupt the DSL's state.
        with dsl_compile_lock(), device_context(self.a_desc.device):
            hardware_info = cutlass.utils.HardwareInfo(self.a_desc.device.index)
            max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1]) - self.num_cluster_overlap_margin
            if max_active_clusters <= 0:
                raise ValueError("max_active_clusters must be > 0 after applying CUDNNFE_CLUSTER_OVERLAP_MARGIN")
            raw_compiled = cute.compile(
                kernel,
                a=a_fake,
                b=b_fake,
                n=n_value,
                k=k_value,
                b_stride_size=b_stride,
                b_major_mode=b_major_mode,
                workspace_ptr=workspace_fake,
                c=c_fake,
                d=d_fake,
                h=h_fake,
                padded_offsets=offsets_fake,
                prob=prob_fake,
                dprob=dprob_fake,
                max_active_clusters=max_active_clusters,
                stream=make_fake_stream(use_tvm_ffi_env_stream=False),
                options="--enable-tvm-ffi",
            )
        self._raw_compiled = raw_compiled
        dense = self.weight_mode == MoEWeightMode.DENSE

        def tensor_api(
            a_tensor: torch.Tensor,
            c_tensor: torch.Tensor,
            d_row_tensor: torch.Tensor,
            d_srelu_tensor: Optional[torch.Tensor],
            padded_offsets: torch.Tensor,
            prob_tensor: torch.Tensor,
            dprob_kernel_tensor: torch.Tensor,
            b_tensor: Optional[torch.Tensor],
            b_ptrs: Optional[torch.Tensor],
            workspace_address: int,
            stream: cuda.CUstream,
        ) -> None:
            raw_compiled(
                a_tensor,
                b_tensor if dense else int(get_data_ptr(b_ptrs)),
                n_value,
                k_value,
                b_stride,
                workspace_address,
                c_tensor,
                d_row_tensor,
                d_srelu_tensor,
                padded_offsets,
                prob_tensor,
                dprob_kernel_tensor,
                stream,
            )

        self._compiled_kernel = tensor_api
        if self.deterministic:
            # Plan time too, so execute() only launches the fold.
            compile_dprob_fold(self.a_desc.device)

    def validate_live_operands(
        self, a_tensor, c_tensor, d_row_tensor, padded_offsets, prob_tensor, dprob_tensor, b_tensor, b_ptrs, d_srelu_tensor, dprob_workspace_tensor
    ) -> None:
        """Every metadata-only execute check; execute() memoizes it on the operands' metadata."""
        if self.generate_d_srelu != (d_srelu_tensor is not None):
            raise ValueError("d_srelu_tensor must be passed exactly when the API was built with sample_d_srelu")
        if self.deterministic != (dprob_workspace_tensor is not None):
            raise ValueError("dprob_workspace_tensor must be passed exactly when the API was built with deterministic=True")

        a_desc = self._validate_live_tensor(a_tensor, self.a_desc, "a_tensor", dynamic_m=True)
        c_desc = self._validate_live_tensor(c_tensor, self.c_desc, "c_tensor", dynamic_m=True)
        d_row_desc = self._validate_live_tensor(d_row_tensor, self.d_row_desc, "d_row_tensor", dynamic_m=True)
        prob_desc = self._validate_live_tensor(prob_tensor, self.prob_desc, "prob_tensor", dynamic_m=True)
        dprob_desc = self._validate_live_tensor(dprob_tensor, self.dprob_desc, "dprob_tensor", dynamic_m=True)
        self._validate_live_tensor(padded_offsets, self.padded_offsets_desc, "padded_offsets")

        tensor_m, k, _ = a_desc.shape
        n = c_desc.shape[1]
        slots = self._dprob_slots()
        if tensor_m % MoEGroupedGemmDsreluBf16Kernel.FIX_PAD_SIZE != 0:
            raise ValueError(f"a_tensor M dimension must be a multiple of {MoEGroupedGemmDsreluBf16Kernel.FIX_PAD_SIZE}, got {tensor_m}")
        for desc, expected_shape, expected_stride, name in (
            (a_desc, (tensor_m, k, 1), (k, 1, tensor_m * k), "a_tensor"),
            (c_desc, (tensor_m, n, 1), (n, 1, tensor_m * n), "c_tensor"),
            (d_row_desc, (tensor_m, n, 1), (n, 1, tensor_m * n), "d_row_tensor"),
            (prob_desc, (tensor_m, 1, 1), (1, 1, 1), "prob_tensor"),
            (dprob_desc, (tensor_m, 1, 1), (1, 1, 1), "dprob_tensor"),
        ):
            self._expect_shape(desc, expected_shape, name)
            self._expect_stride(desc, expected_stride, name)
        if d_srelu_tensor is not None:
            d_srelu_desc = self._validate_live_tensor(d_srelu_tensor, self.d_srelu_desc, "d_srelu_tensor", dynamic_m=True)
            self._expect_shape(d_srelu_desc, (tensor_m, n, 1), "d_srelu_tensor")
            self._expect_stride(d_srelu_desc, (n, 1, tensor_m * n), "d_srelu_tensor")
        if dprob_workspace_tensor is not None:
            ws_desc = self._validate_live_tensor(dprob_workspace_tensor, self.dprob_workspace_desc, "dprob_workspace_tensor", dynamic_m=True)
            self._expect_shape(ws_desc, (tensor_m, slots, 1), "dprob_workspace_tensor")
            self._expect_stride(ws_desc, (slots, 1, tensor_m * slots), "dprob_workspace_tensor")
        if tensor_m * max(n, k, slots) >= _INT32_LIMIT:
            raise ValueError(f"M*N, M*K and M*slots must be below 2**31 elements, got M={tensor_m}, N={n}, K={k}")

        if self.weight_mode == MoEWeightMode.DENSE:
            if b_tensor is None or b_ptrs is not None:
                raise ValueError("Dense execution requires b_tensor and forbids b_ptrs")
            self._validate_live_tensor(b_tensor, self.b_desc, "b_tensor")
        else:
            if b_tensor is not None or b_ptrs is None:
                raise ValueError("Discrete execution requires b_ptrs and forbids b_tensor")
            _validate_pointer_tensor(b_ptrs, "b_ptrs", self.expert_cnt)
            if get_device(b_ptrs) != self.a_desc.device:
                raise ValueError(f"b_ptrs must be on the same device as a_tensor ({self.a_desc.device}), got {get_device(b_ptrs)}")

    def execute(
        self,
        a_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_row_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        prob_tensor: torch.Tensor,
        dprob_tensor: torch.Tensor,
        b_tensor: Optional[torch.Tensor] = None,
        b_ptrs: Optional[torch.Tensor] = None,
        d_srelu_tensor: Optional[torch.Tensor] = None,
        dprob_workspace_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        *,
        workspace=None,
    ) -> None:
        """Launch the kernel on ``current_stream``; ``dprob_tensor`` is accumulated into (zero-initialize it).

        With ``deterministic=True`` the kernel writes one FP32 partial per (row, N tile) into
        ``dprob_workspace_tensor`` (zero-initialized) and execute() then adds their left fold to
        ``dprob_tensor`` and zeroes the workspace again. The block-scaled follow-up,
        ``reduce_dprob_workspace(slots, dprob_tensor)`` and then ``slots.zero_()``, therefore adds
        zero here; the return-value form ``reduce_dprob_workspace(slots)`` returns zeros.

        ``workspace``: caller-owned, 128-byte-aligned, contiguous device buffer of at least
        ``scratch_workspace_bytes()`` bytes on the operands' device (torch tensor or any DLPack
        buffer); never allocated here, and required and checked even when ``a_tensor`` has no rows.
        Each launch rewrites the per-expert B descriptors and resets the dynamic-scheduler counter
        in it, so launches that can overlap (on other streams, or in CUDA graphs replayed
        concurrently) need distinct workspaces.
        """
        if current_stream is None:
            # Torch's current stream on the operands' device, not on whichever device is current.
            with device_context(self.a_desc.device):
                current_stream = default_stream(detect_framework(a_tensor))
        if self._compiled_kernel is None:
            raise RuntimeError("Kernel not compiled; call compile() first")
        if prob_tensor is None or dprob_tensor is None:
            raise ValueError("prob_tensor and dprob_tensor are required")

        operands = (a_tensor, c_tensor, d_row_tensor, padded_offsets, prob_tensor, dprob_tensor, b_tensor, b_ptrs, d_srelu_tensor, dprob_workspace_tensor)
        validate_once(self._validated_signatures, self.validate_live_operands, operands)

        # Per call: data pointers change even when the metadata does not.
        for tensor, name in (
            (a_tensor, "a_tensor"),
            (c_tensor, "c_tensor"),
            (d_row_tensor, "d_row_tensor"),
            (padded_offsets, "padded_offsets"),
            (prob_tensor, "prob_tensor"),
            (dprob_tensor, "dprob_tensor"),
            (d_srelu_tensor, "d_srelu_tensor"),
            (dprob_workspace_tensor, "dprob_workspace_tensor"),
            (b_tensor, "b_tensor"),
        ):
            if tensor is not None and get_data_ptr(tensor) % 16 != 0:
                raise ValueError(f"{name} data pointer must be 16-byte aligned")
        if b_ptrs is not None and get_data_ptr(b_ptrs) % 8 != 0:
            raise ValueError("b_ptrs data pointer must be 8-byte aligned")

        # The caller-owned scratch contract (R2) holds whatever M is, so it is checked before the
        # empty-input return.
        nbytes = self.scratch_workspace_bytes()
        ws_view = Workspace(workspace, nbytes, type(self).__name__, device=self.a_desc.device.index).take(nbytes, "uint8")
        validate_workspace_aliases(
            ws_view.data_ptr(),
            nbytes,
            a_tensor=a_tensor,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            d_srelu_tensor=d_srelu_tensor,
            padded_offsets=padded_offsets,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            dprob_workspace_tensor=dprob_workspace_tensor,
            b_tensor=b_tensor,
            b_ptrs=b_ptrs,
        )

        tensor_m, k, _ = get_shape(a_tensor)
        if tensor_m == 0:
            return
        n = get_shape(c_tensor)[1]
        slots = self._dprob_slots()
        # After the empty-input return: the padded-offsets check rejects a last offset of 0, which
        # every M == 0 call has.
        debug_validate_offsets(padded_offsets, expert_cnt=self.expert_cnt, limit=tensor_m, mode="padded", stream=current_stream)
        if b_ptrs is not None:
            debug_validate_pointer_values(b_ptrs, "b_ptrs", stream=current_stream)
            if is_torch_tensor(b_ptrs):
                self._record_pointer_stream(b_ptrs, current_stream)
        # Views with exactly the compiled strides; extent-1 strides are free in the descriptors.
        if self.deterministic:
            dprob_kernel_tensor = mn1_view(dprob_workspace_tensor, tensor_m, slots)
        else:
            dprob_kernel_tensor = mn1_view(dprob_tensor, tensor_m, 1)
        # The launchables belong to the operands' device (compile() pinned them): launch there and
        # restore the caller's device (python/cudnn/AGENTS.md).
        with device_context(self.a_desc.device):
            self._compiled_kernel(
                mn1_view(a_tensor, tensor_m, k),
                mn1_view(c_tensor, tensor_m, n),
                mn1_view(d_row_tensor, tensor_m, n),
                None if d_srelu_tensor is None else mn1_view(d_srelu_tensor, tensor_m, n),
                padded_offsets,
                row_scalar_view(prob_tensor, tensor_m),
                dprob_kernel_tensor,
                b_tensor,
                b_ptrs,
                ws_view.data_ptr(),
                current_stream,
            )
            if self.deterministic:
                fold_dprob_slots(dprob_kernel_tensor, dprob_tensor, current_stream)
