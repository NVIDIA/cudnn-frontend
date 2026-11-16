# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private BF16 API for the SM100 grouped GEMM + weighted squared-ReLU forward kernel."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import make_fake_stream, make_ptr

from cudnn.api_base import APIBase, TensorDesc
from cudnn._torch_stream import as_torch_stream, device_context
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.frost.compiled_cache import dsl_compile_lock
from cudnn.frost.workspace import Workspace, align_up
from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor
from cudnn.tensor_adapter import (
    canonicalize_unit_dim_strides,
    cuda_is_available,
    default_stream,
    detect_framework,
    get_compute_capability,
    get_data_ptr,
    get_device,
    get_shape,
    is_torch_tensor,
)

from ..backend_utils import debug_validate_offsets, debug_validate_pointer_values, validate_workspace_aliases
from ..canonical import validate_once
from ..moe_utils import MoEWeightMode
from ._bf16_options import ACT_INPUTS, resolve_bf16_srelu_options
from .moe_grouped_gemm_srelu import MoEGroupedGemmSreluBf16Kernel

# The BF16 squared-ReLU kernels are tcgen05 kernels sized for the SM100/SM103 shared memory and
# validated there; every other architecture (including Rubin, SM107) is declined.
BF16_SRELU_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))

# TMA extents and the batch-mode strides of the compact operands are 32-bit in this kernel family.
_INT32_LIMIT = 2**31


def check_bf16_srelu_device(api_name: str, device_index: Optional[int] = None) -> None:
    """Raise unless the operands' GPU (``device_index``; the current GPU when None) is one the BF16
    squared-ReLU kernels support."""
    if not cuda_is_available():
        raise RuntimeError("CUDA is not available")
    major, minor = get_compute_capability(device_index)
    if (major, minor) not in BF16_SRELU_COMPUTE_CAPABILITIES:
        supported = ", ".join(f"SM{a}{b}" for a, b in BF16_SRELU_COMPUTE_CAPABILITIES)
        raise RuntimeError(f"{api_name} BF16 supports {supported} only, found SM{major}{minor}")


def mn1_view(tensor: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """``(rows, cols, 1)`` view with the batch-outermost strides ``(cols, 1, rows * cols)`` the kernels are compiled for."""
    return tensor.as_strided((rows, cols, 1), (cols, 1, rows * cols))


def row_scalar_view(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    """``(rows, 1, 1)`` view with unit strides, the layout the per-row prob operand is compiled for."""
    return tensor.as_strided((rows, 1, 1), (1, 1, 1))


class Bf16SreluLaunchMixin:
    """Launch-time state shared by the BF16 squared-ReLU forward and backward APIs.

    Live-operand validation memoized per operand signature, the launch stream a per-expert
    pointer array is recorded on, and the caller-owned scratch contract (recipe R2).
    """

    def _init_launch_state(self) -> None:
        # execute() checks operand metadata once per distinct signature; see validate_live_operands.
        self._validated_signatures: dict = {}
        self._launch_stream_cache: dict = {}
        self._kernel_obj = None

    @staticmethod
    def _expect_shape(desc: TensorDesc, expected: Tuple[int, ...], name: str) -> None:
        if desc.shape != expected:
            raise ValueError(f"{name} shape mismatch: expected {expected}, got {desc.shape}")

    @staticmethod
    def _expect_stride(desc: TensorDesc, expected: Tuple[int, ...], name: str) -> None:
        if desc.stride != canonicalize_unit_dim_strides(desc.shape, expected):
            raise ValueError(f"{name} must use the layout with stride {expected}, got {desc.stride}")

    @staticmethod
    def _expect_device(desc: TensorDesc, device: torch.device, name: str) -> None:
        if desc.device != device:
            raise ValueError(f"{name} must be on {device}, got {desc.device}")

    def _validate_live_tensor(self, tensor: torch.Tensor, sample: TensorDesc, name: str, *, dynamic_m: bool = False) -> TensorDesc:
        desc = self._make_tensor_desc(tensor, name=name, canonical=True)
        if desc.dtype != sample.dtype:
            raise ValueError(f"{name} dtype mismatch: expected {sample.dtype}, got {desc.dtype}")
        if desc.device != sample.device:
            raise ValueError(f"{name} device mismatch: expected {sample.device}, got {desc.device}")
        if dynamic_m:
            if desc.shape[1:] != sample.shape[1:]:
                raise ValueError(f"{name} shape suffix mismatch: expected {sample.shape[1:]}, got {desc.shape[1:]}")
        elif desc.shape != sample.shape or desc.stride != sample.stride:
            raise ValueError(f"{name} descriptor mismatch: expected shape/stride {sample.shape}/{sample.stride}, got {desc.shape}/{desc.stride}")
        return desc

    def _record_pointer_stream(self, b_ptrs: torch.Tensor, current_stream: cuda.CUstream) -> None:
        # The helper kernel reads b_ptrs on the launch stream: keep the caching allocator from
        # handing its memory out before that read, whichever stream the caller frees it on.
        # record_stream only reads the stream's handle, so caching by (device, handle) is safe.
        key = (b_ptrs.device, int(current_stream))
        launch_stream = self._launch_stream_cache.get(key)
        if launch_stream is None:
            launch_stream = as_torch_stream(int(current_stream), b_ptrs.device)
            self._launch_stream_cache[key] = launch_stream
        b_ptrs.record_stream(launch_stream)

    def scratch_workspace_bytes(self) -> int:
        """Caller-provided scratch ``execute(workspace=)`` carves (recipe R2): the per-expert
        TMA-descriptor slots and the dynamic-scheduler counter, 128-byte aligned, never 0."""
        self._ensure_support_checked()
        if not hasattr(self, "_scratch_nbytes"):
            self._scratch_nbytes = max(align_up(self._kernel_instance().get_workspace_bytes(), 128), 128)
        return self._scratch_nbytes


class GroupedGemmSreluBf16API(Bf16SreluLaunchMixin, APIBase):
    """Descriptor-first lifecycle API for the BF16 grouped GEMM + weighted squared-ReLU forward.

    ``C = bf16(A @ B_e^T)`` (the saved pre-activation) and ``D = bf16(p * srelu(Z))`` with ``Z``
    the FP32 accumulator (``act_input="acc"``) or ``float(C)`` (``act_input="saved"``), where
    ``srelu`` is ``relu(z)**2`` or, with ``tanh_clamp_scale=s``, ``(s * tanh(relu(z) / s))**2``.

    The API owns no device memory: ``execute()`` carves the kernel's descriptor and scheduler
    scratch from the caller's ``workspace`` (``scratch_workspace_bytes()`` bytes).
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_prob: torch.Tensor,
        sample_b: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        acc_dtype: Optional[torch.dtype] = None,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        m_aligned: int = 256,
        b_major: str = "k",
        use_dynamic_sched: bool = False,
        tanh_clamp_scale: Optional[float] = None,
        tanh_mode: str = "approx",
        act_input: str = "acc",
    ) -> None:
        if acc_dtype is None:
            acc_dtype = cutlass.Float32
        super().__init__()
        self._warn_experimental_api()
        self._framework = detect_framework(sample_a)
        if self._framework != "torch":
            raise ValueError(f"GroupedGemmSreluSm100 BF16 supports torch tensors only, got {self._framework}")

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
        self.d_desc = self._make_tensor_desc(sample_d, name="sample_d", canonical=True)
        self.padded_offsets_desc = self._make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets", canonical=True)
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob", canonical=True)
        self.b_desc = self._make_tensor_desc(sample_b, name="sample_b", canonical=True)

        self._sample_data_ptrs = {
            name: get_data_ptr(tensor)
            for name, tensor in (
                ("sample_a", sample_a),
                ("sample_b", sample_b),
                ("sample_c", sample_c),
                ("sample_d", sample_d),
                ("sample_padded_offsets", sample_padded_offsets),
                ("sample_prob", sample_prob),
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
        self.tanh_clamp_scale = tanh_clamp_scale
        self.tanh_mode = tanh_mode
        self.act_input = act_input
        self._kernel = MoEGroupedGemmSreluBf16Kernel
        self._raw_compiled = None
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))

    def check_support(self) -> bool:
        for desc, name in (
            (self.c_desc, "sample_c"),
            (self.d_desc, "sample_d"),
            (self.padded_offsets_desc, "sample_padded_offsets"),
            (self.prob_desc, "sample_prob"),
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
            self._expect_stride(self.b_desc, (k, 1, n * k), "sample_b")
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

        self._expect_shape(self.c_desc, (tensor_m, n, 1), "sample_c")
        self._expect_shape(self.d_desc, (tensor_m, n, 1), "sample_d")
        self._expect_shape(self.padded_offsets_desc, (self.expert_cnt,), "sample_padded_offsets")
        self._expect_shape(self.prob_desc, (tensor_m, 1, 1), "sample_prob")

        self._expect_stride(self.a_desc, (k, 1, tensor_m * k), "sample_a")
        self._expect_stride(self.c_desc, (n, 1, tensor_m * n), "sample_c")
        self._expect_stride(self.d_desc, (n, 1, tensor_m * n), "sample_d")
        self._expect_stride(self.padded_offsets_desc, (1,), "sample_padded_offsets")
        self._expect_stride(self.prob_desc, (1, 1, 1), "sample_prob")

        self._check_dtype(self.a_desc, cutlass.BFloat16, "sample_a")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(self.b_desc, cutlass.BFloat16, "sample_b")
        self._check_dtype(self.b_dtype, cutlass.BFloat16, "b_dtype")
        self._check_dtype(self.c_desc, cutlass.BFloat16, "sample_c")
        self._check_dtype(self.d_desc, cutlass.BFloat16, "sample_d")
        self._check_dtype(self.padded_offsets_desc, cutlass.Int32, "sample_padded_offsets")
        self._check_dtype(self.prob_desc, cutlass.Float32, "sample_prob")

        device = self.a_desc.device
        for desc, name in (
            (self.c_desc, "sample_c"),
            (self.d_desc, "sample_d"),
            (self.padded_offsets_desc, "sample_padded_offsets"),
            (self.prob_desc, "sample_prob"),
        ):
            self._expect_device(desc, device, name)
        if self.b_desc is not None:
            self._expect_device(self.b_desc, device, "sample_b")
        for name, data_ptr in self._sample_data_ptrs.items():
            if data_ptr % 16 != 0:
                raise ValueError(f"{name} data pointer must be 16-byte aligned")

        if self.acc_dtype is not cutlass.Float32:
            raise ValueError(f"acc_dtype must be torch.float32, got {self.acc_dtype}")
        if self.m_aligned != MoEGroupedGemmSreluBf16Kernel.FIX_PAD_SIZE:
            raise ValueError(f"m_aligned must be {MoEGroupedGemmSreluBf16Kernel.FIX_PAD_SIZE}, got {self.m_aligned}")
        if self.b_major not in ("k", "n"):
            raise ValueError(f"b_major must be 'k' or 'n', got {self.b_major}")
        if self.expert_cnt <= 0 or self.expert_cnt > 1024:
            raise ValueError(f"expert count must be in [1, 1024], got {self.expert_cnt}")
        if tensor_m % MoEGroupedGemmSreluBf16Kernel.FIX_PAD_SIZE != 0:
            raise ValueError(f"sample_a M dimension must be a multiple of {MoEGroupedGemmSreluBf16Kernel.FIX_PAD_SIZE}, got {tensor_m}")
        if tensor_m * max(n, k) >= _INT32_LIMIT or self.expert_cnt * n * k >= _INT32_LIMIT:
            raise ValueError(f"M*N, M*K and E*N*K must each be below 2**31 elements, got M={tensor_m}, N={n}, K={k}, E={self.expert_cnt}")
        if self.act_input not in ACT_INPUTS:
            raise ValueError(f"act_input must be one of {ACT_INPUTS}, got {self.act_input!r}")
        self.tanh_clamp_scale, self.tanh_mode = resolve_bf16_srelu_options(self.tanh_clamp_scale, self.tanh_mode)

        if not self._kernel.can_implement(
            self.use_2cta_instrs,
            self.mma_tiler_mn,
            self.cluster_shape_mn,
            max(tensor_m, MoEGroupedGemmSreluBf16Kernel.FIX_PAD_SIZE),
            n,
            k,
            self.expert_cnt,
        ):
            raise ValueError(
                f"Unsupported BF16 grouped GEMM SReLU tile/cluster configuration mma_tiler_mn={self.mma_tiler_mn}, cluster_shape_mn={self.cluster_shape_mn}"
            )

        check_bf16_srelu_device("GroupedGemmSreluSm100", self.a_desc.device.index)
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
                act_input=self.act_input,
                tanh_mode=self.tanh_mode,
                weight_mode=self.weight_mode,
                use_dynamic_sched=self.use_dynamic_sched,
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
        if self.weight_mode == MoEWeightMode.DENSE:
            n = self.b_desc.shape[0]
        else:
            n = self.b_shape[0]
        pad = MoEGroupedGemmSreluBf16Kernel.FIX_PAD_SIZE

        # Row-major (M, row, 1) operands with a dynamic M (a multiple of 256); stride_order (1, 0, 2)
        # lists the modes from fastest to slowest.
        def compact_bf16(cols):
            return cute.runtime.make_fake_compact_tensor(
                dtype=cutlass.BFloat16,
                shape=(cute.sym_int(divisibility=pad), cols, 1),
                stride_order=(1, 0, 2),
                assumed_align=16,
            )

        a_fake = compact_bf16(k)
        c_fake = compact_bf16(n)
        d_fake = compact_bf16(n)
        # One FP32 prob per row is read with scalar loads, so 4-byte alignment is all the kernel assumes.
        prob_fake = cute.runtime.make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(divisibility=pad), 1, 1), stride=(1, 1, 1), assumed_align=4)
        offsets_fake = cute.runtime.make_fake_tensor(dtype=cutlass.Int32, shape=(self.expert_cnt,), stride=(1,), assumed_align=16)

        if self.weight_mode == MoEWeightMode.DENSE:
            b_fake = cute.runtime.make_fake_tensor(dtype=cutlass.BFloat16, shape=(n, k, self.expert_cnt), stride=(k, 1, n * k), assumed_align=16)
            n_value = cutlass.Int32(0)
            k_value = cutlass.Int32(0)
            b_stride = cutlass.Int64(0)
            b_major_mode = OperandMajorMode.K
        else:
            b_fake = make_ptr(cutlass.Int64, 16, gmem, assumed_align=8)
            n_value = cutlass.Int32(n)
            k_value = cutlass.Int32(k)
            b_stride = cutlass.Int64(k if self.b_major == "k" else n)
            b_major_mode = OperandMajorMode.K if self.b_major == "k" else OperandMajorMode.MN

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
                padded_offsets=offsets_fake,
                prob=prob_fake,
                max_active_clusters=max_active_clusters,
                stream=make_fake_stream(use_tvm_ffi_env_stream=False),
                options="--enable-tvm-ffi",
            )
        self._raw_compiled = raw_compiled
        dense = self.weight_mode == MoEWeightMode.DENSE

        def tensor_api(
            a_tensor: torch.Tensor,
            c_tensor: torch.Tensor,
            d_tensor: torch.Tensor,
            padded_offsets: torch.Tensor,
            prob_tensor: torch.Tensor,
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
                d_tensor,
                padded_offsets,
                prob_tensor,
                stream,
            )

        self._compiled_kernel = tensor_api

    def validate_live_operands(self, a_tensor, c_tensor, d_tensor, padded_offsets, prob_tensor, b_tensor, b_ptrs) -> None:
        """Every metadata-only execute check; execute() memoizes it on the operands' metadata."""
        a_desc = self._validate_live_tensor(a_tensor, self.a_desc, "a_tensor", dynamic_m=True)
        c_desc = self._validate_live_tensor(c_tensor, self.c_desc, "c_tensor", dynamic_m=True)
        d_desc = self._validate_live_tensor(d_tensor, self.d_desc, "d_tensor", dynamic_m=True)
        prob_desc = self._validate_live_tensor(prob_tensor, self.prob_desc, "prob_tensor", dynamic_m=True)
        self._validate_live_tensor(padded_offsets, self.padded_offsets_desc, "padded_offsets")

        tensor_m, k, _ = a_desc.shape
        n = c_desc.shape[1]
        if tensor_m % MoEGroupedGemmSreluBf16Kernel.FIX_PAD_SIZE != 0:
            raise ValueError(f"a_tensor M dimension must be a multiple of {MoEGroupedGemmSreluBf16Kernel.FIX_PAD_SIZE}, got {tensor_m}")
        self._expect_shape(c_desc, (tensor_m, n, 1), "c_tensor")
        self._expect_shape(d_desc, (tensor_m, n, 1), "d_tensor")
        self._expect_shape(prob_desc, (tensor_m, 1, 1), "prob_tensor")
        self._expect_stride(a_desc, (k, 1, tensor_m * k), "a_tensor")
        self._expect_stride(c_desc, (n, 1, tensor_m * n), "c_tensor")
        self._expect_stride(d_desc, (n, 1, tensor_m * n), "d_tensor")
        self._expect_stride(prob_desc, (1, 1, 1), "prob_tensor")
        if tensor_m * max(n, k) >= _INT32_LIMIT:
            raise ValueError(f"M*N and M*K must be below 2**31 elements, got M={tensor_m}, N={n}, K={k}")

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
        d_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        prob_tensor: torch.Tensor,
        b_tensor: Optional[torch.Tensor] = None,
        b_ptrs: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        *,
        workspace=None,
    ) -> None:
        """Launch the kernel on ``current_stream``.

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
        if prob_tensor is None:
            raise ValueError("prob_tensor is required")

        operands = (a_tensor, c_tensor, d_tensor, padded_offsets, prob_tensor, b_tensor, b_ptrs)
        validate_once(self._validated_signatures, self.validate_live_operands, operands)

        # Per call: data pointers change even when the metadata does not.
        for tensor, name in (
            (a_tensor, "a_tensor"),
            (c_tensor, "c_tensor"),
            (d_tensor, "d_tensor"),
            (padded_offsets, "padded_offsets"),
            (prob_tensor, "prob_tensor"),
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
            d_tensor=d_tensor,
            padded_offsets=padded_offsets,
            prob_tensor=prob_tensor,
            b_tensor=b_tensor,
            b_ptrs=b_ptrs,
        )

        tensor_m, k, _ = get_shape(a_tensor)
        if tensor_m == 0:
            return
        n = get_shape(c_tensor)[1]
        # After the empty-input return: the padded-offsets check rejects a last offset of 0, which
        # every M == 0 call has.
        debug_validate_offsets(padded_offsets, expert_cnt=self.expert_cnt, limit=tensor_m, mode="padded", stream=current_stream)
        if b_ptrs is not None:
            debug_validate_pointer_values(b_ptrs, "b_ptrs", stream=current_stream)
            if is_torch_tensor(b_ptrs):
                self._record_pointer_stream(b_ptrs, current_stream)
        # Pass views with exactly the compiled strides; extent-1 strides are free in the descriptors.
        # The launchable belongs to the operands' device (compile() pinned it): launch there and restore
        # the caller's device (python/cudnn/AGENTS.md).
        with device_context(self.a_desc.device):
            self._compiled_kernel(
                mn1_view(a_tensor, tensor_m, k),
                mn1_view(c_tensor, tensor_m, n),
                mn1_view(d_tensor, tensor_m, n),
                padded_offsets,
                row_scalar_view(prob_tensor, tensor_m),
                b_tensor,
                b_ptrs,
                ws_view.data_ptr(),
                current_stream,
            )
