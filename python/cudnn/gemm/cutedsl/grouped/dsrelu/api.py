# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified API for Grouped GEMM dSReLU Backward Kernel (SM100+)

This module provides a single API class that supports both contiguous (dense)
and discrete weight modes for block-scaled grouped GEMM with dSReLU activation
gradient in MoE (Mixture of Experts) workloads.

Dense mode
    All expert weights are packed contiguously in a 3-D tensor (N, K, L).
    Callers supply ``sample_b`` and ``sample_sfb``.

Discrete mode
    Each expert has its own memory allocation.  Callers supply
    ``num_experts``, ``b_shape``, ``b_dtype``, and per-expert pointer arrays
    at execution time.
"""

from __future__ import annotations

from .moe_blockscaled_grouped_gemm_dsrelu_quant import (
    BlockScaledMoEGroupedGemmQuantBwdKernel,
    EpilogueType,
)
from ..backend_utils import _torch_stream_context
from ..moe_utils import MoEWeightMode
from cuda.bindings import driver as cuda
import logging
import os
from typing import Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from cudnn.datatypes import _convert_to_cutlass_data_type, _convert_to_cutlass_data_type_or_none
from cudnn.api_base import APIBase, TupleDict, ceil_div, is_power_of_2
from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor
from cudnn.tensor_adapter import (
    allocate_byte_workspace,
    canonicalize_unit_dim_strides,
    cuda_is_available,
    default_stream,
    detect_framework,
    framework_dtype,
    get_compute_capability,
    get_data_ptr,
    get_shape,
    get_strides,
    is_torch_tensor,
)


def _resolve_cluster_shape_mn(mma_tiler_mn: Tuple[int, int], cluster_shape_mn: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """The cluster shape the kernel will actually run with, defaults applied.

    Shared with ``GroupedGemmDsreluSm100.__init__`` so the wrapper cannot resolve a
    different default than the class it is about to construct. M=256 selects the 2-SM
    tcgen05 MMA atom (M=128 is 1-SM), and that CTA pair has to share a cluster along M --
    hence (2, 1). check_support asserts both halves of the pairing.
    """
    if cluster_shape_mn is not None:
        return cluster_shape_mn
    use_2cta_instrs = mma_tiler_mn[0] == 256
    return (2, 1) if use_2cta_instrs else (1, 1)


def _dprob_n_slots(n_out: int, mma_tiler_mn: Tuple[int, int], cluster_shape_mn: Optional[Tuple[int, int]], deterministic: bool) -> int:
    """Extent of dprob's dim 1.

    Non-deterministic (default): 1 -- every N-tile atomically accumulates into the same
    ``dprob[token]``, so the fp32 summation order follows tile scheduling.

    Deterministic: one slot per N-tile, so each ``(token, tile_n)`` pair has exactly one
    writer and the caller can reduce over dim 1 in a fixed order.

    The extent must cover every ``tile_n_idx`` the scheduler can emit, which is *not*
    ``ceil_div(n_out, tile_n)``: MoEPersistentTileScheduler counts whole clusters and then
    expands to CTAs (``cta_tile_n_idx = cluster_tile_n_idx * cluster_n + cta_id_in_cluster``,
    over ``ceil_div(n_out, tile_n * cluster_n)`` clusters), so the bound rounds up to a
    multiple of ``cluster_n``. The two agree only at ``cluster_n == 1``; using the narrower
    form would index past the end of the workspace for any wider cluster.
    """
    if not deterministic:
        return 1
    cluster_n = _resolve_cluster_shape_mn(mma_tiler_mn, cluster_shape_mn)[1]
    return ceil_div(n_out, mma_tiler_mn[1] * cluster_n) * cluster_n


def _cta_tile_m(mma_tiler_mn: Tuple[int, int]) -> int:
    """Rows one CTA's epilogue owns, and so the granularity of a deterministic dbias slot.

    Mirrors the kernel's ``cta_tile_shape_mnk[0] = mma_tiler[0] // thr_id_size``. Shared with
    ``GroupedGemmDsreluSm100.__init__`` for the same reason ``_resolve_cluster_shape_mn`` is:
    the wrapper must not derive a different value than the class it is about to construct.
    """
    return mma_tiler_mn[0] // (2 if mma_tiler_mn[0] == 256 else 1)


def _reduce_dprob_slots(dprob_workspace, dprob_tensor, current_stream):
    """Collapse the per-N-tile dprob slots into the caller's ``(valid_m, 1, 1)`` output.

    ``dprob_workspace`` is None whenever the kernel wrote dprob directly -- the non-deterministic
    path -- and then there is nothing to collapse. Every token owns the same number of slots, so
    this is a plain sum over dim 1; dbias needs a segment sum instead. Accumulates onto the
    output, matching what the atomic does on the non-deterministic path.

    Issued on ``current_stream``, the stream execute() ran the kernel on. On torch's current
    stream instead, a caller who passes a different stream would have the reduction read the
    workspace before the kernel finished writing it.
    """
    if dprob_workspace is None:
        return dprob_tensor
    import torch

    with _torch_stream_context(current_stream, dprob_workspace.device):
        reduced = torch.sum(dprob_workspace, dim=1, keepdim=True)
        return reduced if dprob_tensor is None else dprob_tensor.add_(reduced)


def _reduce_dbias_slots(dbias_workspace, dbias_tensor, current_stream, padded_offsets, cta_tile_m):
    """Segment-sum the per-M-block dbias slots into a per-expert ``(expert, n, 1)`` bf16 tensor.

    ``dbias_workspace`` is None whenever the kernel wrote dbias directly -- no dbias at all, or
    the non-deterministic path -- and then there is nothing to collapse.

    Expert ``e`` owns the contiguous block range whose first token falls in
    ``[padded_offsets[e-1], padded_offsets[e])``. Done as a one-hot matmul because ``index_add_``
    and ``scatter_add_`` are themselves non-deterministic on CUDA, and slicing per expert would
    need ``padded_offsets`` on the host -- a sync in the training loop. A cumsum-and-subtract
    would be deterministic too but cancels catastrophically across experts of different
    magnitude. Comparing in token space rather than block space saves the division.
    """
    if dbias_workspace is None:
        return dbias_tensor
    import torch

    m_blocks, n_out = dbias_workspace.shape[0], dbias_workspace.shape[1]
    with _torch_stream_context(current_stream, dbias_workspace.device):
        block_tok = torch.arange(0, m_blocks * cta_tile_m, cta_tile_m, device=dbias_workspace.device, dtype=padded_offsets.dtype).unsqueeze(0)
        starts = torch.cat((padded_offsets.new_zeros(1), padded_offsets[:-1])).unsqueeze(1)
        # Built in the workspace dtype so the matmul is a straight bf16 GEMM: cuBLAS accumulates
        # it in fp32 internally and rounds once, which is still better than the default path
        # rounding on every M-tile, at half the memory and one store instead of two.
        onehot = ((block_tok >= starts) & (block_tok < padded_offsets.unsqueeze(1))).to(dbias_workspace.dtype)
        reduced = (onehot @ dbias_workspace.reshape(m_blocks, n_out)).reshape(-1, n_out, 1)
        # Accumulate, like dprob and like the atomic on the non-deterministic path: a caller
        # summing dbias across micro-batches into one buffer must not get overwrite semantics
        # from the flag alone.
        return reduced if dbias_tensor is None else dbias_tensor.add_(reduced)


def _reinterpret_raw_grouped_fp4_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if _convert_to_cutlass_data_type_or_none(tensor.dtype) is cutlass.Uint8:
        cute_tensor = from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1)
        cute_tensor.element_type = cutlass.Float4E2M1FN
        return cute_tensor
    return tensor


class GroupedGemmDsreluSm100(APIBase):
    """Unified API for grouped GEMM dSReLU backward operation on SM100+ GPUs.

    This kernel performs block-scaled grouped GEMM with dSReLU activation
    gradient (dSReLU), designed for MoE workloads.  It supports
    both dense (contiguous) and discrete (per-expert pointer) weight layouts
    through the ``BlockScaledMoEGroupedGemmQuantBwdKernel``.

    Weight mode is auto-detected from the constructor arguments:

    - **Dense**: provide ``sample_b`` and ``sample_sfb``.
    - **Discrete**: provide ``num_experts``, ``b_shape``, and ``b_dtype``.

    Example::

        # Dense mode
        api = GroupedGemmDsreluSm100(
            sample_a=a, sample_c=c,
            sample_d_row=d_row, sample_d_col=d_col,
            sample_sfa=sfa, sample_padded_offsets=offsets,
            sample_alpha=alpha,
            sample_prob=prob, sample_dprob=dprob,
            sample_b=b, sample_sfb=sfb,
        )

        # Discrete mode
        api = GroupedGemmDsreluSm100(
            sample_a=a, sample_c=c,
            sample_d_row=d_row, sample_d_col=d_col,
            sample_sfa=sfa, sample_padded_offsets=offsets,
            sample_alpha=alpha,
            sample_prob=prob, sample_dprob=dprob,
            num_experts=8, b_shape=(n, k), b_dtype=torch.uint8,
        )

        api.check_support()
        api.compile()
        api.execute(...)
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        # Dense mode (contiguous) -- provide these. sample_dbias is optional:
        sample_b: Optional[torch.Tensor] = None,
        sample_c: Optional[torch.Tensor] = None,
        sample_d_row: Optional[torch.Tensor] = None,
        sample_d_col: Optional[torch.Tensor] = None,
        sample_d_srelu: Optional[torch.Tensor] = None,
        sample_sfa: Optional[torch.Tensor] = None,
        sample_sfb: Optional[torch.Tensor] = None,
        sample_padded_offsets: Optional[torch.Tensor] = None,
        sample_alpha: Optional[torch.Tensor] = None,
        sample_prob: Optional[torch.Tensor] = None,
        sample_dprob: Optional[torch.Tensor] = None,
        sample_dbias: Optional[torch.Tensor] = None,
        sample_dprob_workspace: Optional[torch.Tensor] = None,
        sample_dbias_workspace: Optional[torch.Tensor] = None,
        # Discrete mode -- provide these instead:
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        # Optional quantization output arguments
        sample_sfd_row: Optional[torch.Tensor] = None,
        sample_sfd_col: Optional[torch.Tensor] = None,
        sample_sfd_col_d_srelu: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        # Configuration
        acc_dtype: Optional[torch.dtype] = None,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
        b_major: str = "k",
        use_dynamic_sched: bool = False,
        use_dsrelu_reuse: bool = False,
        deterministic: bool = False,
    ):
        """Initialize the GroupedGemmDsreluSm100 API.

        :param sample_a: Sample A tensor (valid_m, k, 1)
        :param sample_c: Sample C tensor -- forward activations (valid_m, n, 1)
        :param sample_d_row: Sample D row output tensor (valid_m, n, 1)
        :param sample_d_col: Sample D col output tensor (valid_m, n, 1)
        :param sample_sfa: Sample scale factor A tensor
        :param sample_padded_offsets: End offset for each expert after padding
        :param sample_alpha: Per-group alpha scaling factors
        :param sample_prob: Per-row probability tensor (valid_m, 1, 1)
        :param sample_dprob: Gradient of probability tensor (valid_m, 1, 1), must be zero-initialized
        :param sample_b: (Dense) Sample B tensor (n, k, l)
        :param sample_sfb: (Dense) Sample scale factor B tensor
        :param sample_dbias: Optional dbias output tensor (expert_cnt, n, 1) bfloat16, in both modes
        :param sample_dprob_workspace: Determinism scratch, required iff ``deterministic``:
            ``dprob_workspace_shape(valid_m)``, float32
        :param sample_dbias_workspace: Determinism scratch, required iff ``deterministic`` and
            ``sample_dbias``: ``dbias_workspace_shape(valid_m, n)``, bfloat16
        :param num_experts: (Discrete) Number of experts
        :param b_shape: (Discrete) Shape of a single expert B tensor, e.g. (n, k)
        :param b_dtype: (Discrete) Data type of B tensors
        :param sample_sfd_row: Optional row scale factor for D
        :param sample_sfd_col: Optional column scale factor for D
        :param sample_amax: Optional amax tensor for quantization, shape (expert_cnt, 1)
        :param sample_norm_const: Optional normalization constant
        :param acc_dtype: Accumulator data type
        :param mma_tiler_mn: MMA tiler shape (M, N)
        :param cluster_shape_mn: Cluster shape (M, N)
        :param sf_vec_size: Scale factor vector size
        :param vector_f32: Use vectorized f32 operations
        :param m_aligned: Alignment for group M dimension
        :param discrete_col_sfd: Generate discrete col-major scale factor tensor
        :param b_major: Major dimension for B tensor, one of "k" or "n"
        :param use_dynamic_sched: Enable dynamic tile scheduling for load balancing
        :param use_dsrelu_reuse: Reuse relu(C)^2 between d_srelu and dprob
        :param deterministic: Compute dprob and dbias run-to-run bit-exactly. dprob and dbias keep
            their shapes and dtypes; what the flag adds is the two scratch tensors above, which the
            kernel writes instead, and a reduction the caller runs after execute():
            ``reduce_dprob_workspace`` and ``reduce_dbias_workspace``.
            grouped_gemm_dsrelu_wrapper_sm100 allocates and reduces both for you
        """
        framework = detect_framework(sample_a)
        if sample_a is not None and framework not in ("torch", "jax"):
            raise ValueError(f"Unsupported tensor framework '{framework}' for GroupedGemmDsreluSm100; pass torch tensors or JAX arrays")
        if framework == "jax":
            if sample_b is not None:
                raise ValueError(
                    "Dense weight mode (sample_b/sample_sfb) is not expressible as JAX arrays "
                    "(the expert-outermost strided B layout (n, k, l) has no row-major equivalent); "
                    "use discrete mode (num_experts, b_shape, b_dtype) with per-expert weight pointers"
                )
            if _convert_to_cutlass_data_type_or_none(getattr(sample_a, "dtype", None)) in (cutlass.Float4E2M1FN, cutlass.Uint8) or (
                b_dtype is not None and _convert_to_cutlass_data_type_or_none(b_dtype) in (cutlass.Float4E2M1FN, cutlass.Uint8)
            ):
                raise ValueError(
                    "Packed fp4 A/B tensors (float4_e2m1fn / raw uint8) are not expressible as JAX arrays "
                    "(JAX has no packed fp4 dtype); use fp8 inputs from JAX, or torch tensors for fp4"
                )
        if acc_dtype is None:
            acc_dtype = cutlass.Float32
        super().__init__()
        self._framework = framework

        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        # ---- Weight mode auto-detection ----
        if sample_b is not None and num_experts is None:
            self.weight_mode = MoEWeightMode.DENSE
            if sample_sfb is None:
                raise ValueError("sample_sfb is required when sample_b is provided (dense mode)")
        elif num_experts is not None and sample_b is None:
            self.weight_mode = MoEWeightMode.DISCRETE
            if b_shape is None or b_dtype is None:
                raise ValueError("b_shape and b_dtype are required in discrete mode")
        else:
            raise ValueError("Provide either (sample_b, sample_sfb) for dense mode " "or (num_experts, b_shape, b_dtype) for discrete mode, but not both.")

        self._sample_a_tensor = sample_a
        self._sample_b_tensor = sample_b

        # ---- Common tensor descriptors ----
        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a", interpret_uint8_as_fp4x2=False, canonical=True)
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c", canonical=True)
        self.d_row_desc = self._make_tensor_desc(sample_d_row, name="sample_d_row", canonical=True)
        self.d_col_desc = self._make_tensor_desc(sample_d_col, name="sample_d_col", canonical=True)
        self.d_srelu_desc = self._make_tensor_desc(sample_d_srelu, name="sample_d_srelu", canonical=True)
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa", canonical=True)
        self.padded_offsets_desc = self._make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets", canonical=True)
        self.alpha_desc = self._make_tensor_desc(sample_alpha, name="sample_alpha", canonical=True)
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob", canonical=True)
        self.dprob_desc = self._make_tensor_desc(sample_dprob, name="sample_dprob", canonical=True)
        self.dbias_desc = self._make_tensor_desc(sample_dbias, name="sample_dbias", canonical=True)
        # Determinism scratch, kept out of the output descriptors so dprob/dbias mean the same
        # thing in both modes. The kernel writes whichever of the pair is active.
        self.dprob_workspace_desc = self._make_tensor_desc(sample_dprob_workspace, name="sample_dprob_workspace", canonical=True)
        self.dbias_workspace_desc = self._make_tensor_desc(sample_dbias_workspace, name="sample_dbias_workspace", canonical=True)

        self.sfd_row_desc = self._make_tensor_desc(sample_sfd_row, name="sample_sfd_row", canonical=True)
        self.sfd_col_desc = self._make_tensor_desc(sample_sfd_col, name="sample_sfd_col", canonical=True)
        self.sfd_col_d_srelu_desc = self._make_tensor_desc(sample_sfd_col_d_srelu, name="sample_sfd_col_d_srelu", canonical=True)
        self.amax_desc = self._make_tensor_desc(sample_amax, name="sample_amax", canonical=True)
        self.norm_const_desc = self._unpad_tensor_to_ndim(
            self._make_tensor_desc(sample_norm_const, name="sample_norm_const", canonical=True),
            1,
            "norm_const",
        )

        # ---- Mode-specific state ----
        if self.weight_mode == MoEWeightMode.DENSE:
            self.b_desc = self._make_tensor_desc(sample_b, name="sample_b", interpret_uint8_as_fp4x2=False, canonical=True)
            self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb", canonical=True)
            self.expert_cnt = self.padded_offsets_desc.shape[0]
        else:
            self._value_error_if(num_experts == 0, "num_experts must be > 0")
            self.expert_cnt = num_experts
            self.b_shape = b_shape
            self.b_dtype = _convert_to_cutlass_data_type(b_dtype) if b_dtype is not None else None
            self.b_major = b_major
            self._value_error_if(
                self.padded_offsets_desc.shape[0] != self.expert_cnt,
                f"padded_offsets length ({self.padded_offsets_desc.shape[0]}) " f"must equal num_experts ({self.expert_cnt})",
            )

        # ---- Configuration ----
        self.acc_dtype = _convert_to_cutlass_data_type(acc_dtype)
        self.mma_tiler_mn = mma_tiler_mn
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cta_tile_m = _cta_tile_m(mma_tiler_mn)
        self.cluster_shape_mn = _resolve_cluster_shape_mn(mma_tiler_mn, cluster_shape_mn)
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = vector_f32
        self.m_aligned = m_aligned
        self.discrete_col_sfd = discrete_col_sfd
        if self.weight_mode == MoEWeightMode.DENSE:
            self.b_major = b_major  # stored for both modes

        self.use_dynamic_sched = use_dynamic_sched
        self.use_dsrelu_reuse = use_dsrelu_reuse
        self.deterministic = deterministic

        self._interpret_uint8_as_fp4x2 = True
        self._has_dbias = self.dbias_desc is not None
        self._generate_d_srelu = self.d_srelu_desc is not None
        self._kernel = BlockScaledMoEGroupedGemmQuantBwdKernel

        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        self._logger.debug(f"setting num_cluster_overlap_margin: {self.num_cluster_overlap_margin}")

        self._workspace = None
        self._live_ptrs = None
        self._compile_b_ptrs = None
        self._compile_sfb_ptrs = None

        self._logger.debug("__init__ completed")

    @staticmethod
    def _sf_desc_is_physical(sf_desc) -> bool:
        """True if the SF descriptor is the physical C-contiguous (L, MN', K', 32, 4, 4)
        allocation rather than the torch-style permuted (32, 4, MN', 4, K', L) atom view.

        Frameworks that cannot express the permuted strided view (e.g. JAX, whose arrays
        are row-major) pass the physical form. The kernel rebuilds the SF layout from the
        GEMM shapes via ``tile_atom_to_shape_SF`` and consumes only the SF base pointer,
        so the two forms are byte-identical.
        """
        shape = sf_desc.shape
        return not (len(shape) == 6 and shape[0] == 32 and shape[1] == 4 and shape[3] == 4)

    def _check_sf_shape(self, sf_desc, mn128: int, rest: int, l: int, name: str) -> None:
        """Validate an SF tensor shape and strides, accepting the permuted atom view or (in
        discrete weight mode) the physical C-contiguous form -- see ``_sf_desc_is_physical``.

        The kernel consumes only the SF base pointer and rebuilds the layout from the GEMM
        shapes, so both forms must be exactly the C-contiguous physical allocation in memory:
        strides are validated too (a shape-matching but differently-strided tensor would
        silently produce wrong results).
        """
        if sf_desc is None:
            return
        self._value_error_if(len(sf_desc.shape) != 6, f"{name} tensor must be 6-D, got shape {sf_desc.shape}")
        if not self._sf_desc_is_physical(sf_desc):
            self._check_tensor_shape(sf_desc, (32, 4, mn128, 4, rest, l), name)
            _ = self._check_tensor_stride(
                sf_desc,
                stride=[canonicalize_unit_dim_strides((32, 4, mn128, 4, rest, l), (16, 4, rest * 512, 1, 512, mn128 * rest * 512))],
                name=name,
                extra_error_msg=f"{name} atom view must be the (3, 4, 1, 5, 2, 0) permutation of a C-contiguous (L, MN', K', 32, 4, 4) allocation",
            )
            return
        self._value_error_if(
            self.weight_mode != MoEWeightMode.DISCRETE,
            f"{name} physical (L, MN', K', 32, 4, 4) form is only supported in discrete weight mode; " "provide the permuted (32, 4, MN', 4, K', L) atom view",
        )
        self._check_tensor_shape(sf_desc, (l, mn128, rest, 32, 4, 4), name)
        _ = self._check_tensor_stride(
            sf_desc,
            stride=[canonicalize_unit_dim_strides((l, mn128, rest, 32, 4, 4), (mn128 * rest * 512, rest * 512, 512, 16, 4, 1))],
            name=name,
            extra_error_msg=f"{name} in the physical (L, MN', K', 32, 4, 4) form must be C-contiguous",
        )

    @staticmethod
    def dprob_workspace_shape(valid_m: int, n: int, mma_tiler_mn: Tuple[int, int] = (256, 256), cluster_shape_mn: Optional[Tuple[int, int]] = None):
        """Shape of ``sample_dprob_workspace`` -- one fp32 slot per (token, N-tile)."""
        return (valid_m, _dprob_n_slots(n, mma_tiler_mn, cluster_shape_mn, True), 1)

    @staticmethod
    def dbias_workspace_shape(valid_m: int, n: int, mma_tiler_mn: Tuple[int, int] = (256, 256)):
        """Shape of ``sample_dbias_workspace`` -- one bf16 slot per (CTA M-block, n)."""
        return (ceil_div(valid_m, _cta_tile_m(mma_tiler_mn)), n, 1)

    @staticmethod
    def reduce_dprob_workspace(dprob_workspace, dprob_tensor=None, current_stream=None):
        """Collapse the dprob slots into ``(valid_m, 1, 1)``; accumulates onto dprob_tensor if given."""
        return _reduce_dprob_slots(dprob_workspace, dprob_tensor, current_stream)

    @staticmethod
    def reduce_dbias_workspace(dbias_workspace, padded_offsets, dbias_tensor=None, mma_tiler_mn: Tuple[int, int] = (256, 256), current_stream=None):
        """Segment-sum the dbias slots per expert; accumulates onto dbias_tensor if given.

        Public because a plain sum over dim 0 is wrong here and gets it wrong quietly: the slots
        have to be grouped by the expert ranges ``padded_offsets`` marks off. Accumulates rather
        than overwrites, matching both reduce_dprob_workspace and the atomic the kernel uses when
        the flag is off.
        """
        return _reduce_dbias_slots(dbias_workspace, dbias_tensor, current_stream, padded_offsets, _cta_tile_m(mma_tiler_mn))

    @property
    def _kernel_dprob_desc(self):
        """The dprob descriptor the kernel writes: the per-N-tile scratch under determinism."""
        return self.dprob_workspace_desc if self.deterministic else self.dprob_desc

    @property
    def _kernel_dbias_desc(self):
        """The dbias descriptor the kernel writes: the M-block scratch under determinism."""
        return self.dbias_workspace_desc if self.deterministic and self.dbias_desc is not None else self.dbias_desc

    def _make_dbias_fake(self):
        """Fake dbias tensor for compilation.

        The determinism scratch is indexed by M-block, so its dim 0 scales with the token count
        and has to stay symbolic -- baked static it would bind the compiled kernel to one valid_m
        and silently reuse it for another, the trap _dprob_n_slots documents for dprob's slots.
        Divisibility 2 because valid_m is a multiple of m_aligned (256) and a block is
        cta_tile_m (128) rows. The (expert, n, 1) output has no token-dependent extent.
        """
        desc = self._kernel_dbias_desc
        if desc is None or not self.deterministic:
            return self._make_fake_cute_tensor_from_desc(desc, assumed_align=16)
        return self._make_fake_cute_compact_tensor(
            dtype=desc.dtype,
            shape=desc.shape,
            stride_order=desc.stride_order,
            dynamic_mode=0,
            divisibility=2,
        )

    def check_support(self) -> bool:
        """Check if the kernel configuration is supported.

        :return: True if supported, raises exception otherwise
        """
        self._logger.debug("Entering check_support")

        # ---- SFD group validation ----
        all_none = all(x is None for x in [self.sfd_row_desc, self.sfd_col_desc, self.norm_const_desc])
        all_provided = all(x is not None for x in [self.sfd_row_desc, self.sfd_col_desc, self.norm_const_desc])
        self._value_error_if(
            not (all_none or all_provided),
            "sfd_row_desc, sfd_col_desc, and norm_const_desc must be all None or all not None",
        )
        self._user_requested_sfd = all_provided

        # ---- Shapes and strides ----
        self._logger.debug("Checking tensor shapes and strides")
        tensor_m, k, _one = self._tensor_shape(self.a_desc, name="sample_a")

        if self.weight_mode == MoEWeightMode.DENSE:
            n, _, l = self._tensor_shape(self.b_desc, name="sample_b")
        else:
            # Discrete: extract n, k from b_shape
            if len(self.b_shape) == 2:
                n, b_k = self.b_shape
            else:
                n, b_k, _ = self.b_shape
            self._value_error_if(b_k != k, f"B K dimension ({b_k}) must match A K dimension ({k})")
            l = self.expert_cnt  # for shape checks that use l

        n_out = n

        self._check_tensor_shape(self.a_desc, (tensor_m, k, 1), "A")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.b_desc, (n, k, l), "B")
        self._check_tensor_shape(self.c_desc, (tensor_m, n_out, 1), "C")
        self._check_tensor_shape(self.d_row_desc, (tensor_m, n_out, 1), "D_row")
        self._check_tensor_shape(self.d_col_desc, (tensor_m, n_out, 1), "D_col")
        self._check_tensor_shape(self.d_srelu_desc, (tensor_m, n_out, 1), "D_srelu")

        rest_k = ceil_div(ceil_div(k, self.sf_vec_size), 4)
        self._check_sf_shape(self.sfa_desc, ceil_div(tensor_m, 128), rest_k, 1, "SFA")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.sfb_desc, (32, 4, ceil_div(n, 128), 4, rest_k, l), "SFB")

        rest_n_out = ceil_div(ceil_div(n_out, self.sf_vec_size), 4)
        self._check_sf_shape(self.sfd_row_desc, ceil_div(tensor_m, 128), rest_n_out, 1, "SFD_row")
        rest_m = ceil_div(ceil_div(tensor_m, self.sf_vec_size), 4)
        self._check_sf_shape(self.sfd_col_desc, ceil_div(n_out, 128), rest_m, 1, "SFD_col")
        self._check_sf_shape(self.sfd_col_d_srelu_desc, ceil_div(n_out, 128), rest_m, 1, "SFD_col_d_srelu")

        self._check_tensor_shape(self.alpha_desc, (self.expert_cnt,), "alpha")
        self._check_tensor_shape(self.prob_desc, (tensor_m, 1, 1), "prob")
        self._check_tensor_shape(self.dprob_desc, (tensor_m, 1, 1), "dprob")
        self._check_tensor_shape(self.dbias_desc, (self.expert_cnt, n_out, 1), "dbias")
        # The determinism scratch. Separate tensors rather than a reshaped dprob/dbias, so the
        # output contract does not depend on the flag; the kernel writes these instead.
        self._check_tensor_shape(
            self.dprob_workspace_desc, (tensor_m, _dprob_n_slots(n_out, self.mma_tiler_mn, self.cluster_shape_mn, True), 1), "dprob_workspace"
        )
        self._check_tensor_shape(self.dbias_workspace_desc, (ceil_div(tensor_m, self.cta_tile_m), n_out, 1), "dbias_workspace")
        self._check_tensor_shape(self.amax_desc, (self.expert_cnt, 1), "amax")
        self._check_tensor_shape(self.norm_const_desc, (1,), "norm_const")
        self._check_tensor_shape(self.padded_offsets_desc, (self.expert_cnt,), "padded_offsets")

        # Strides
        _ = self._check_tensor_stride(
            self.a_desc,
            stride=[(k, 1, tensor_m * k)],
            extra_error_msg="A must have k-major layout",
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            if self._is_fp8(self.a_desc):
                _ = self._check_tensor_stride(
                    self.b_desc,
                    stride=[(k, 1, n * k), (1, n, n * k)],
                    extra_error_msg="For fp8 ab_dtype, B must have k- or n-major layout",
                )
            else:
                _ = self._check_tensor_stride(
                    self.b_desc,
                    stride=[(k, 1, n * k)],
                    extra_error_msg="For fp4 ab_dtype, B must have k-major layout",
                )
        _ = self._check_tensor_stride(
            self.c_desc,
            stride=[(n_out, 1, tensor_m * n_out)],
            extra_error_msg="C must have n-major layout",
        )
        _ = self._check_tensor_stride(
            self.d_row_desc,
            stride=[(n_out, 1, tensor_m * n_out)],
            extra_error_msg="D_row must have n-major layout",
        )
        _ = self._check_tensor_stride(
            self.d_col_desc,
            stride=[(n_out, 1, tensor_m * n_out)],
            extra_error_msg="D_col must have n-major layout",
        )

        # ---- Data types ----
        self._logger.debug("Checking data types")
        self.ab_dtype = self._check_dtype(
            self.a_desc,
            dtype=[
                cutlass.Float4E2M1FN,
                cutlass.Uint8,
                cutlass.Float8E5M2,
                cutlass.Float8E4M3FN,
            ],
            name="A/B",
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(
                self.b_desc,
                dtype=self.ab_dtype,
                name="B",
                extra_error_msg="B must have the same dtype as A",
            )
        else:
            self._value_error_if(
                self.b_dtype != self.ab_dtype,
                f"b_dtype ({self.b_dtype}) must match A dtype ({self.ab_dtype})",
            )

        self.sf_dtype = self._check_dtype(
            self.sfa_desc,
            dtype=[cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN],
            name="SFA/SFB/SFD",
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(
                self.sfb_desc,
                dtype=self.sf_dtype,
                name="SFB",
                extra_error_msg="SFB must have the same dtype as SFA",
            )
        self._check_dtype(
            self.sfd_row_desc,
            dtype=self.sf_dtype,
            name="SFD_row",
            extra_error_msg="SFD_row must have the same dtype as SFA",
        )
        self._check_dtype(
            self.sfd_col_desc,
            dtype=self.sf_dtype,
            name="SFD_col",
            extra_error_msg="SFD_col must have the same dtype as SFA",
        )

        self._value_error_if(
            self.sf_vec_size not in [16, 32],
            f"sf_vec_size must be 16 or 32, got {self.sf_vec_size}",
        )
        self._value_error_if(
            self.sf_dtype is cutlass.Float8E4M3FN and self.sf_vec_size == 32,
            f"sf_dtype {self.sf_dtype} and sf_vec_size {self.sf_vec_size} combination is not supported",
        )
        self._value_error_if(
            self._is_fp8(self.ab_dtype) and self.sf_vec_size == 16,
            f"ab_dtype {self.ab_dtype} and sf_vec_size {self.sf_vec_size} combination is not supported",
        )

        self._check_dtype(
            self.acc_dtype,
            dtype=cutlass.Float32,
            name="Accumulator",
            extra_error_msg="Accumulator must be float32",
        )
        self._check_dtype(
            self.prob_desc,
            dtype=cutlass.Float32,
            name="Prob",
            extra_error_msg="Prob must be float32",
        )
        self._check_dtype(
            self.dprob_desc,
            dtype=cutlass.Float32,
            name="Dprob",
            extra_error_msg="Dprob must be float32",
        )
        self._check_dtype(self.dbias_desc, dtype=cutlass.BFloat16, name="Dbias", extra_error_msg="dbias must be bfloat16")
        # Scratch matches the output it stands in for: the slots only need a single writer and a
        # fixed-order reduction to be reproducible, not a wider accumulator.
        self._check_dtype(self.dprob_workspace_desc, dtype=cutlass.Float32, name="Dprob workspace", extra_error_msg="dprob workspace must be float32")
        self._check_dtype(self.dbias_workspace_desc, dtype=cutlass.BFloat16, name="Dbias workspace", extra_error_msg="dbias workspace must be bfloat16")
        self.c_dtype = self._check_dtype(
            self.c_desc,
            dtype=[cutlass.Float32, cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2],
            name="C",
        )
        if self._is_fp8(self.c_dtype) and self.vector_f32:
            raise ValueError("Invalid configuration: fp8 c_dtype and vector_f32 is not supported. " "Please use vector_f32=False or c_dtype=bfloat16 instead")

        if self._is_fp4x2(self.ab_dtype):
            self.d_dtype = self._check_dtype(
                self.d_row_desc,
                dtype=[cutlass.Float16, cutlass.BFloat16, cutlass.Float32],
                name="D_row",
                extra_error_msg="D_row must be fp16, bf16, or float32 when ab_dtype is fp4",
            )
        elif self._is_fp8(self.ab_dtype):
            self.d_dtype = self._check_dtype(
                self.d_row_desc,
                dtype=[
                    cutlass.Float8E4M3FN,
                    cutlass.Float8E5M2,
                ],
                name="D_row",
                extra_error_msg="D_row must be fp8 dtype when ab_dtype is fp8",
            )
        else:
            raise NotImplementedError(f"Invalid ab_dtype: {self.ab_dtype}, expected fp4 or fp8")
        self._check_dtype(
            self.d_col_desc,
            dtype=self.d_dtype,
            name="D_col",
            extra_error_msg="D_col must have the same dtype as D_row",
        )
        self._check_dtype(
            self.d_srelu_desc,
            dtype=self.d_dtype,
            name="D_srelu",
            extra_error_msg="D_srelu must have the same dtype as D_row",
        )

        # ---- SFD generation logic ----
        kernel_generate_sfd = self._is_fp8(self.ab_dtype) and self.sf_dtype is cutlass.Float8E8M0FNU and self._is_fp8(self.d_dtype)
        self._value_error_if(
            kernel_generate_sfd and not self._user_requested_sfd,
            "sfd_row, sfd_col, and norm_const are required for FP8 input/FP8 output with sf_dtype=torch.float8_e8m0fnu",
        )
        if not kernel_generate_sfd and self._user_requested_sfd:
            self._logger.warning(
                "sfd_row/sfd_col/norm_const were provided, but this configuration does not generate SFD outputs; " "the tensors will be ignored by the kernel",
            )
        self.generate_sfd = kernel_generate_sfd
        self._value_error_if(
            self._generate_d_srelu and self.generate_sfd and self.sfd_col_d_srelu_desc is None,
            "sfd_col_d_srelu is required when d_srelu is generated for FP8 output",
        )
        self._check_dtype(
            self.sfd_col_d_srelu_desc,
            dtype=self.sf_dtype,
            name="SFD_col_d_srelu",
            extra_error_msg="SFD_col_d_srelu must have the same dtype as SFA",
        )
        if self.discrete_col_sfd and not self.generate_sfd:
            self._logger.warning("discrete_col_sfd is True but generate_sfd is False, discrete_col_sfd will be ignored")
            self.discrete_col_sfd = False

        # ---- Activation function validation ----
        # ---- Discrete-mode-specific validation ----
        if self.weight_mode == MoEWeightMode.DISCRETE:
            self._value_error_if(
                self.b_major not in ["k", "n"],
                f"b_major must be 'k' or 'n', got {self.b_major}",
            )
            self._value_error_if(
                self._is_fp4x2(self.ab_dtype) and self.b_major != "k",
                "b_major must be 'k' when ab_dtype is fp4",
            )

        # ---- MMA tile / cluster shape ----
        self._logger.debug("Checking MMA tile shape and cluster shape")
        self._value_error_if(
            not self.use_2cta_instrs and self.mma_tiler_mn[0] != 128,
            f"MMA tiler M must be 128 when use_2cta_instrs=False, got {self.mma_tiler_mn[0]}",
        )
        self._value_error_if(
            self.use_2cta_instrs and self.mma_tiler_mn[0] != 256,
            f"MMA tiler M must be 256 when use_2cta_instrs=True, got {self.mma_tiler_mn[0]}",
        )
        self._value_error_if(
            self.mma_tiler_mn[1] != 256,
            f"MMA tiler N must be 256, got {self.mma_tiler_mn[1]}",
        )
        self._value_error_if(
            self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0,
            f"cluster_shape_mn[0] must be divisible by 2 when use_2cta_instrs=True, got {self.cluster_shape_mn[0]}",
        )
        self._value_error_if(
            not (
                self.cluster_shape_mn[0] * self.cluster_shape_mn[1] <= 16
                and self.cluster_shape_mn[0] > 0
                and self.cluster_shape_mn[1] > 0
                and self.cluster_shape_mn[0] <= 4
                and self.cluster_shape_mn[1] <= 4
                and is_power_of_2(self.cluster_shape_mn[0])
                and is_power_of_2(self.cluster_shape_mn[1])
            ),
            f"Invalid cluster shape: expected values to be powers of 2 and product <= 16, got {self.cluster_shape_mn}",
        )
        cluster_tiler_m = (self.cluster_shape_mn[0] // (2 if self.use_2cta_instrs else 1)) * self.mma_tiler_mn[0]
        self._value_error_if(
            cluster_tiler_m not in [128, 256],
            f"Invalid cluster tiler shape: expected cluster_tiler_m in {{128, 256}}, got {cluster_tiler_m}",
        )
        self._value_error_if(
            self.m_aligned % self.mma_tiler_mn[0] != 0,
            f"m_aligned must be divisible by mma_tiler_mn[0], got {self.m_aligned} % {self.mma_tiler_mn[0]} != 0",
        )
        self._value_error_if(
            self.m_aligned != BlockScaledMoEGroupedGemmQuantBwdKernel.FIX_PAD_SIZE,
            f"m_aligned must be {BlockScaledMoEGroupedGemmQuantBwdKernel.FIX_PAD_SIZE} (FIX_PAD_SIZE), got {self.m_aligned}",
        )

        # ---- Tensor alignment ----
        self._logger.debug("Checking tensor alignment")

        def check_contiguous_16B_alignment(dtype, stride_order, tensor_shape):
            is_mode0_major = stride_order == (0, 1, 2)
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // (_convert_to_cutlass_data_type(dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2).width)
            return num_major_elements % num_contiguous_elements == 0

        if self.weight_mode == MoEWeightMode.DENSE:
            b_stride_order_for_check = self.b_desc.stride_order
            b_shape_for_check = (n, k, l)
        else:
            b_stride_order_for_check = (0, 1, 2) if self.b_major == "n" else (1, 0, 2)
            b_shape_for_check = (n, k, 1)

        self._value_error_if(
            not (
                check_contiguous_16B_alignment(self.ab_dtype, self.a_desc.stride_order, (tensor_m, k, l))
                and check_contiguous_16B_alignment(self.ab_dtype, b_stride_order_for_check, b_shape_for_check)
                and check_contiguous_16B_alignment(self.d_dtype, self.d_row_desc.stride_order, (tensor_m, n_out, 1))
            ),
            "Invalid tensor alignment: tensors must be 16B aligned",
        )

        # ---- Expert count limit ----
        self._value_error_if(
            self.expert_cnt > 1024,
            f"expert_cnt must be <= 1024, got {self.expert_cnt}",
        )

        # ---- Determinism scratch must be present exactly when the flag is on ----
        self._value_error_if(
            self.deterministic and self.dprob_workspace_desc is None,
            "deterministic=True requires sample_dprob_workspace of shape dprob_workspace_shape(valid_m, n), float32",
        )
        self._value_error_if(
            self.deterministic and self.dbias_desc is not None and self.dbias_workspace_desc is None,
            "deterministic=True with dbias requires sample_dbias_workspace of shape dbias_workspace_shape(valid_m, n), bfloat16",
        )
        self._value_error_if(
            not self.deterministic and (self.dprob_workspace_desc is not None or self.dbias_workspace_desc is not None),
            "the determinism workspaces are only used when deterministic=True",
        )

        # ---- Disabled configurations ----
        # Deterministic dbias keys its workspace by absolute M-block,
        # token_offset // cta_tile_m + tile_m_idx, which is single-writer only if the scheduler
        # emits no tile past an expert's range. It emits
        # ceil_div(tokens_e, cta_tile_m * cluster_m) * cluster_m tiles per expert, so that ceil
        # must be exact: m_aligned, which every tokens_e is a multiple of, has to divide by the
        # cluster's M footprint. True for every supported shape (m_aligned is pinned to 256,
        # cta_tile_m * cluster_m is 128 or 256), but a wider cluster would alias one expert's
        # phantom tiles onto the next expert's slots.
        self._not_implemented_error_if(
            self.deterministic and self.dbias_desc is not None and self.m_aligned % (self.cta_tile_m * self.cluster_shape_mn[0]) != 0,
            f"Invalid configuration: deterministic dbias needs m_aligned ({self.m_aligned}) to be a multiple of "
            f"cta_tile_m * cluster_m ({self.cta_tile_m} * {self.cluster_shape_mn[0]}).",
        )

        self._not_implemented_error_if(
            self.dbias_desc is None and self._is_fp4x2(self.ab_dtype) and self.sf_vec_size == 16 and self.d_dtype is cutlass.Float32,
            "Invalid configuration: fp4 ab_dtype, sf_vec_size 16, d_dtype float32 is not supported. " "Please use sf_vec_size 32 or d_dtype bf16 instead",
        )

        # ---- SM100+ check ----
        if not cuda_is_available():
            raise RuntimeError("CUDA is not available")
        major, minor = get_compute_capability()
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmDsrelu requires SM100+ compute capability, " f"but found SM{compute_capability}")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        """Compile the kernel."""
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            self._logger.debug("Kernel already compiled; skipping recompilation")
            return
        if self.a_desc.shape[0] == 0:
            self._logger.debug("sample valid_m is zero, skipping kernel compilation")
            return

        gemm_dsrelu = self._kernel(
            sf_vec_size=self.sf_vec_size,
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vectorized_f32=self.vector_f32,
            generate_sfd=self.generate_sfd,
            discrete_col_sfd=self.discrete_col_sfd,
            expert_cnt=self.expert_cnt,
            weight_mode=self.weight_mode,
            use_dynamic_sched=self.use_dynamic_sched,
            epilogue_type=EpilogueType.DSRELU.value,
            generate_dbias=self._has_dbias,
            generate_d_srelu=self._generate_d_srelu,
            use_dsrelu_reuse=self.use_dsrelu_reuse,
            deterministic=self.deterministic,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        max_active_clusters -= self.num_cluster_overlap_margin
        self._value_error_if(
            max_active_clusters <= 0,
            "max_active_clusters must be > 0 after applying overlap margin; reduce CUDNNFE_CLUSTER_OVERLAP_MARGIN",
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        self._use_full_dynamic_mnkl = os.environ.get("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1") != "0"

        workspace_bytes = gemm_dsrelu.get_workspace_bytes()
        # Internal scratch in the caller's framework allocator; kernels write through its
        # raw pointer and it is never surfaced as a framework array.
        self._workspace = allocate_byte_workspace(self._framework, workspace_bytes, self.a_desc.device)

        if self.weight_mode == MoEWeightMode.DENSE:
            self._compile_dense(gemm_dsrelu, max_active_clusters, fake_stream)
        else:
            self._compile_discrete(gemm_dsrelu, max_active_clusters, fake_stream)

        self._logger.debug("Kernel compiled successfully")

    def _compile_dense(self, gemm_dsrelu, max_active_clusters, fake_stream) -> None:
        """Compile for dense (contiguous) weight mode."""
        self._logger.debug("Compiling grouped_gemm_dsrelu kernel")
        use_full_dynamic = self._use_full_dynamic_mnkl

        fake_workspace_ptr = cute.runtime.nullptr(
            dtype=cutlass.Uint8,
            assumed_align=128,
        )

        if not use_full_dynamic:
            valid_m = cute.sym_int(divisibility=256)

            a_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.a_desc.dtype,
                shape=(valid_m, *self.a_desc.shape[1:]),
                stride_order=self.a_desc.stride_order,
            )
            b_cute_fake = self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16)
            c_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.c_desc.dtype,
                shape=(valid_m, *self.c_desc.shape[1:]),
                stride_order=self.c_desc.stride_order,
            )
            d_row_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_row_desc.dtype,
                shape=(valid_m, *self.d_row_desc.shape[1:]),
                stride_order=self.d_row_desc.stride_order,
            )
            d_col_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_col_desc.dtype,
                shape=(valid_m, *self.d_col_desc.shape[1:]),
                stride_order=self.d_col_desc.stride_order,
            )
            d_srelu_cute_fake = None
            if self.d_srelu_desc is not None:
                d_srelu_cute_fake = self._make_fake_cute_compact_tensor(
                    dtype=self.d_srelu_desc.dtype,
                    shape=(valid_m, *self.d_srelu_desc.shape[1:]),
                    stride_order=self.d_srelu_desc.stride_order,
                )

            tensor_m_128 = cute.sym_int()
            stride_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfa_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfa_desc.dtype,
                shape=(32, 4, tensor_m_128, 4, self.sfa_desc.shape[4], 1),
                stride=(16, 4, self.sfa_desc.stride[2], 1, 512, stride_tensor_m_128),
            )

            sfb_cute_fake = self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16)

            prob_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.prob_desc.dtype,
                shape=(valid_m, 1, 1),
                stride_order=self.prob_desc.stride_order,
            )
            dprob_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self._kernel_dprob_desc.dtype,
                shape=(valid_m, *self._kernel_dprob_desc.shape[1:]),
                stride_order=self._kernel_dprob_desc.stride_order,
            )

            sfd_row_fake = None
            sfd_col_fake = None
            sfd_col_d_srelu_fake = None
            if self.sfd_row_desc is not None:
                stride_sfd_m = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_row_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_row_desc.dtype,
                    shape=(32, 4, tensor_m_128, 4, self.sfd_row_desc.shape[4], 1),
                    stride=(16, 4, self.sfd_row_desc.stride[2], 1, 512, stride_sfd_m),
                )
            if self.sfd_col_desc is not None:
                rest_m = cute.sym_int(divisibility=1)
                stride_sfd_n = cute.sym_int(divisibility=32 * 4 * 4)
                stride_rest_m = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_col_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_col_desc.dtype,
                    shape=(32, 4, self.sfd_col_desc.shape[2], 4, rest_m, 1),
                    stride=(16, 4, stride_rest_m, 1, 512, stride_sfd_n),
                )
                if self.sfd_col_d_srelu_desc is not None:
                    sfd_col_d_srelu_fake = self._make_fake_cute_tensor(
                        dtype=self.sfd_col_d_srelu_desc.dtype,
                        shape=(32, 4, self.sfd_col_d_srelu_desc.shape[2], 4, rest_m, 1),
                        stride=(16, 4, stride_rest_m, 1, 512, stride_sfd_n),
                    )
        else:
            valid_m = cute.sym_int(divisibility=256)
            n_sym = cute.sym_int()
            n_out_sym = cute.sym_int()
            k_sym = cute.sym_int()
            l_sym = cute.sym_int()

            a_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.a_desc.dtype,
                shape=(valid_m, k_sym, 1),
                stride_order=self.a_desc.stride_order,
                dynamic_mode=self.a_desc.stride_order[0],
                divisibility=32 if self._is_fp4x2(self.ab_dtype) else 16,
            )
            b_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.b_desc.dtype,
                shape=(n_sym, k_sym, l_sym),
                stride_order=self.b_desc.stride_order,
                dynamic_mode=self.b_desc.stride_order[0],
                divisibility=32 if self._is_fp4x2(self.ab_dtype) else 16,
            )

            c_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.c_desc.dtype,
                shape=(valid_m, n_out_sym, 1),
                stride_order=self.c_desc.stride_order,
                dynamic_mode=self.c_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.c_desc.dtype) else 16,
            )

            d_row_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_row_desc.dtype,
                shape=(valid_m, n_out_sym, 1),
                stride_order=self.d_row_desc.stride_order,
                dynamic_mode=self.d_row_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.d_row_desc.dtype) else 16,
            )

            d_col_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_col_desc.dtype,
                shape=(valid_m, n_out_sym, 1),
                stride_order=self.d_col_desc.stride_order,
                dynamic_mode=self.d_col_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.d_col_desc.dtype) else 16,
            )
            d_srelu_cute_fake = None
            if self.d_srelu_desc is not None:
                d_srelu_cute_fake = self._make_fake_cute_compact_tensor(
                    dtype=self.d_srelu_desc.dtype,
                    shape=(valid_m, n_out_sym, 1),
                    stride_order=self.d_srelu_desc.stride_order,
                    dynamic_mode=self.d_srelu_desc.stride_order[0],
                    divisibility=8 if self._is_f16(self.d_srelu_desc.dtype) else 16,
                )

            tensor_m_128 = cute.sym_int()
            rest_k = cute.sym_int()
            stride_rest_k = cute.sym_int(divisibility=32 * 4 * 4)
            stride_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfa_shape = list(self.sfa_desc.shape)
            sfa_shape[2] = tensor_m_128
            sfa_shape[4] = rest_k
            sfa_stride = list(self.sfa_desc.stride)
            sfa_stride[2] = stride_rest_k
            sfa_stride[5] = stride_tensor_m_128
            sfa_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfa_desc.dtype,
                shape=tuple(sfa_shape),
                stride=tuple(sfa_stride),
            )

            tensor_n_128 = cute.sym_int()
            stride_sfb_rest_k = cute.sym_int(divisibility=32 * 4 * 4)
            stride_sfb_tensor_n_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfb_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfb_desc.dtype,
                shape=(32, 4, tensor_n_128, 4, rest_k, l_sym),
                stride=(16, 4, stride_sfb_tensor_n_128, 1, 512, stride_sfb_rest_k),
            )

            prob_cute_fake = self._make_fake_cute_tensor(
                dtype=self.prob_desc.dtype,
                shape=(valid_m, *self.prob_desc.shape[1:]),
                stride=self.prob_desc.stride,
            )
            dprob_cute_fake = self._make_fake_cute_tensor(
                dtype=self._kernel_dprob_desc.dtype,
                shape=(valid_m, *self._kernel_dprob_desc.shape[1:]),
                stride=self._kernel_dprob_desc.stride,
            )

            sfd_row_fake = None
            sfd_col_fake = None
            sfd_col_d_srelu_fake = None
            if self.sfd_row_desc is not None:
                rest_n_out = cute.sym_int()
                stride_sfd_rest_n_out = cute.sym_int(divisibility=32 * 4 * 4)
                stride_sfd_rest_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_row_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_row_desc.dtype,
                    shape=(32, 4, tensor_m_128, 4, rest_n_out, 1),
                    stride=(16, 4, stride_sfd_rest_n_out, 1, 512, stride_sfd_rest_tensor_m_128),
                )
            if self.sfd_col_desc is not None:
                tensor_n_out_128 = cute.sym_int()
                rest_m_dyn = cute.sym_int()
                stride_sfd_rest_m = cute.sym_int(divisibility=32 * 4 * 4)
                stride_sfd_n_out = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_col_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_col_desc.dtype,
                    shape=(32, 4, tensor_n_out_128, 4, rest_m_dyn, 1),
                    stride=(16, 4, stride_sfd_rest_m, 1, 512, stride_sfd_n_out),
                )
                if self.sfd_col_d_srelu_desc is not None:
                    sfd_col_d_srelu_fake = self._make_fake_cute_tensor(
                        dtype=self.sfd_col_d_srelu_desc.dtype,
                        shape=(32, 4, tensor_n_out_128, 4, rest_m_dyn, 1),
                        stride=(16, 4, stride_sfd_rest_m, 1, 512, stride_sfd_n_out),
                    )

        dbias_fake = self._make_dbias_fake()

        _compiled_kernel = cute.compile(
            gemm_dsrelu,
            a=_reinterpret_raw_grouped_fp4_tensor(self._sample_a_tensor) if self.a_desc.dtype is cutlass.Uint8 else a_cute_fake,
            b=_reinterpret_raw_grouped_fp4_tensor(self._sample_b_tensor) if self.b_desc.dtype is cutlass.Uint8 else b_cute_fake,
            sfb=sfb_cute_fake,
            n=cutlass.Int32(0),
            k=cutlass.Int32(0),
            b_stride_size=cutlass.Int64(0),
            b_major_mode=OperandMajorMode.K,
            workspace_ptr=fake_workspace_ptr,
            c=c_cute_fake,
            d=d_row_cute_fake,
            d_col=d_col_cute_fake,
            sfa=sfa_cute_fake,
            sfd_row_tensor=sfd_row_fake,
            sfd_col_tensor=sfd_col_fake,
            amax_tensor=self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16),
            norm_const_tensor=self._make_fake_cute_tensor_from_desc(self.norm_const_desc, assumed_align=16),
            padded_offsets=self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc, assumed_align=16),
            alpha=self._make_fake_cute_tensor_from_desc(self.alpha_desc, assumed_align=16),
            prob=prob_cute_fake,
            dprob=dprob_cute_fake,
            dbias_tensor=dbias_fake,
            d_srelu=d_srelu_cute_fake,
            sfd_col_d_srelu_tensor=sfd_col_d_srelu_fake,
            max_active_clusters=max_active_clusters,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )

        cached_workspace_ptr = from_dlpack(self._workspace, assumed_align=128).iterator

        def tensor_api(
            a_tensor: torch.Tensor,
            b_tensor: torch.Tensor,
            c_tensor: torch.Tensor,
            d_row_tensor: torch.Tensor,
            d_col_tensor: Optional[torch.Tensor],
            d_srelu_tensor: Optional[torch.Tensor],
            sfa_tensor: torch.Tensor,
            sfb_tensor: torch.Tensor,
            sfd_row_tensor: Optional[torch.Tensor],
            sfd_col_tensor: Optional[torch.Tensor],
            sfd_col_d_srelu_tensor: Optional[torch.Tensor],
            amax_tensor: Optional[torch.Tensor],
            norm_const_tensor: Optional[torch.Tensor],
            padded_offsets: torch.Tensor,
            alpha_tensor: torch.Tensor,
            prob_tensor: torch.Tensor,
            dprob_tensor: torch.Tensor,
            dbias_tensor: Optional[torch.Tensor],
            stream: cuda.CUstream,
        ) -> None:
            norm_const_tensor = self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const")
            _compiled_kernel(
                _reinterpret_raw_grouped_fp4_tensor(a_tensor),
                _reinterpret_raw_grouped_fp4_tensor(b_tensor),
                sfb_tensor,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int64(0),
                cached_workspace_ptr,
                c_tensor,
                d_row_tensor,
                d_col_tensor,
                sfa_tensor,
                sfd_row_tensor,
                sfd_col_tensor,
                amax_tensor,
                norm_const_tensor,
                padded_offsets,
                alpha_tensor,
                prob_tensor,
                dprob_tensor,
                dbias_tensor,
                d_srelu_tensor,
                sfd_col_d_srelu_tensor,
                stream,
            )

        self._compiled_kernel = tensor_api

    def _compile_discrete(self, gemm_dsrelu, max_active_clusters, fake_stream) -> None:
        """Compile for discrete (per-expert pointer) weight mode."""
        if len(self.b_shape) == 2:
            n, k = self.b_shape
        else:
            n, k, _ = self.b_shape

        b_major_mode = OperandMajorMode.K if self.b_major == "k" else OperandMajorMode.MN
        if self.b_major == "k":
            b_stride_size = k
        else:
            b_stride_size = n

        ab_cutlass_dtype = _convert_to_cutlass_data_type(self.a_desc.dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2)
        align = 32 if ab_cutlass_dtype.width == 4 else 16

        valid_m = cute.sym_int(divisibility=256)
        a_tensor = self._make_fake_cute_tensor(
            dtype=self.a_desc.dtype,
            shape=(valid_m, *self.a_desc.shape[1:]),
            stride=(self.a_desc.stride[0], *self.a_desc.stride[1:]),
            assumed_align=align,
        )
        c_tensor = self._make_fake_cute_tensor(
            dtype=self.c_desc.dtype,
            shape=(valid_m, *self.c_desc.shape[1:]),
            stride=(self.c_desc.stride[0], *self.c_desc.stride[1:]),
        )
        d_row_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.d_row_desc.dtype,
            shape=(valid_m, *self.d_row_desc.shape[1:]),
            stride_order=self.d_row_desc.stride_order,
        )
        d_col_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.d_col_desc.dtype,
            shape=(valid_m, *self.d_col_desc.shape[1:]),
            stride_order=self.d_col_desc.stride_order,
        )
        d_srelu_tensor = None
        if self.d_srelu_desc is not None:
            d_srelu_tensor = self._make_fake_cute_compact_tensor(
                dtype=self.d_srelu_desc.dtype,
                shape=(valid_m, *self.d_srelu_desc.shape[1:]),
                stride_order=self.d_srelu_desc.stride_order,
            )

        tensor_m_128 = cute.sym_int()
        stride_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
        if self._sf_desc_is_physical(self.sfa_desc):
            # Physical C-contiguous (1, M', K', 32, 4, 4) form (e.g. JAX): the kernel rebuilds
            # the SF layout from the GEMM shapes and consumes only the base pointer. The
            # extent-1 L dim's (M-dependent) stride is symbolic; the ABI cannot observe it.
            rest_k_sfa = self.sfa_desc.shape[2]
            sfa_tensor = self._make_fake_cute_tensor(
                dtype=self.sfa_desc.dtype,
                shape=(1, tensor_m_128, rest_k_sfa, 32, 4, 4),
                stride=(stride_tensor_m_128, rest_k_sfa * 512, 512, 16, 4, 1),
                assumed_align=16,
            )
        else:
            sfa_shape = list(self.sfa_desc.shape)
            sfa_shape[2] = tensor_m_128
            sfa_stride = list(self.sfa_desc.stride)
            sfa_stride[5] = stride_tensor_m_128
            sfa_tensor = self._make_fake_cute_tensor(
                dtype=self.sfa_desc.dtype,
                shape=tuple(sfa_shape),
                stride=tuple(sfa_stride),
                assumed_align=16,
            )
        sfd_row_tensor = None
        if self.sfd_row_desc is not None:
            stride_sfd_m = cute.sym_int(divisibility=32 * 4 * 4)
            if self._sf_desc_is_physical(self.sfd_row_desc):
                rest_n_sfd = self.sfd_row_desc.shape[2]
                sfd_row_tensor = self._make_fake_cute_tensor(
                    dtype=self.sfd_row_desc.dtype,
                    shape=(1, tensor_m_128, rest_n_sfd, 32, 4, 4),
                    stride=(stride_sfd_m, rest_n_sfd * 512, 512, 16, 4, 1),
                    assumed_align=16,
                )
            else:
                sfd_row_tensor = self._make_fake_cute_tensor(
                    dtype=self.sfd_row_desc.dtype,
                    shape=(32, 4, tensor_m_128, 4, self.sfd_row_desc.shape[4], 1),
                    stride=(16, 4, self.sfd_row_desc.stride[2], 1, 512, stride_sfd_m),
                    assumed_align=16,
                )
        sfd_col_tensor = None
        sfd_col_d_srelu_tensor = None
        if self.sfd_col_desc is not None:
            rest_m = cute.sym_int(divisibility=1)
            stride_sfd_n = cute.sym_int(divisibility=32 * 4 * 4)
            stride_rest_m = cute.sym_int(divisibility=32 * 4 * 4)
            if self._sf_desc_is_physical(self.sfd_col_desc):
                # Physical (1, N', M_rest, 32, 4, 4): both outer strides are M-dependent.
                n_out_128 = self.sfd_col_desc.shape[1]
                sfd_col_tensor = self._make_fake_cute_tensor(
                    dtype=self.sfd_col_desc.dtype,
                    shape=(1, n_out_128, rest_m, 32, 4, 4),
                    stride=(stride_sfd_n, stride_rest_m, 512, 16, 4, 1),
                    assumed_align=16,
                )
                if self.sfd_col_d_srelu_desc is not None:
                    sfd_col_d_srelu_tensor = self._make_fake_cute_tensor(
                        dtype=self.sfd_col_d_srelu_desc.dtype,
                        shape=(1, n_out_128, rest_m, 32, 4, 4),
                        stride=(stride_sfd_n, stride_rest_m, 512, 16, 4, 1),
                        assumed_align=16,
                    )
            else:
                sfd_col_tensor = self._make_fake_cute_tensor(
                    dtype=self.sfd_col_desc.dtype,
                    shape=(32, 4, self.sfd_col_desc.shape[2], 4, rest_m, 1),
                    stride=(16, 4, stride_rest_m, 1, 512, stride_sfd_n),
                    assumed_align=16,
                )
                if self.sfd_col_d_srelu_desc is not None:
                    sfd_col_d_srelu_tensor = self._make_fake_cute_tensor(
                        dtype=self.sfd_col_d_srelu_desc.dtype,
                        shape=(32, 4, self.sfd_col_d_srelu_desc.shape[2], 4, rest_m, 1),
                        stride=(16, 4, stride_rest_m, 1, 512, stride_sfd_n),
                        assumed_align=16,
                    )
        amax_tensor = self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16)
        norm_const_tensor_cute = self._make_fake_cute_tensor_from_desc(self.norm_const_desc, assumed_align=16)
        padded_offsets_tensor = self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc, assumed_align=16)
        alpha_tensor = self._make_fake_cute_tensor_from_desc(self.alpha_desc, assumed_align=16)
        prob_tensor = self._make_fake_cute_tensor(
            dtype=self.prob_desc.dtype,
            shape=(valid_m, *self.prob_desc.shape[1:]),
            stride=self.prob_desc.stride,
            assumed_align=16,
        )
        dprob_tensor = self._make_fake_cute_tensor(
            dtype=self._kernel_dprob_desc.dtype,
            shape=(valid_m, *self._kernel_dprob_desc.shape[1:]),
            stride=self._kernel_dprob_desc.stride,
            assumed_align=16,
        )
        dbias_tensor = self._make_dbias_fake()

        # Compile-time placeholders for the pointer-array arguments: real device bytes
        # (fake tensors have dummy iterators) allocated in the caller's framework,
        # retyped to Int64 via the element_type override.
        self._compile_b_ptrs = allocate_byte_workspace(self._framework, 8 * self.expert_cnt, self.a_desc.device)
        self._compile_sfb_ptrs = allocate_byte_workspace(self._framework, 8 * self.expert_cnt, self.a_desc.device)
        b_ptrs_placeholder = from_dlpack(self._compile_b_ptrs, assumed_align=8)
        b_ptrs_placeholder.element_type = cutlass.Int64
        b_ptrs_cute = b_ptrs_placeholder.iterator
        sfb_ptrs_placeholder = from_dlpack(self._compile_sfb_ptrs, assumed_align=8)
        sfb_ptrs_placeholder.element_type = cutlass.Int64
        sfb_ptrs_cute = sfb_ptrs_placeholder.iterator

        workspace_ptr_cute = from_dlpack(self._workspace, assumed_align=128).iterator

        self._logger.debug("Compiling discrete grouped GEMM dSReLU kernel")
        _compiled_kernel = cute.compile(
            gemm_dsrelu,
            a_tensor,
            b_ptrs_cute,
            sfb_ptrs_cute,
            cutlass.Int32(n),
            cutlass.Int32(k),
            cutlass.Int64(b_stride_size),
            b_major_mode,
            workspace_ptr_cute,
            c_tensor,
            d_row_tensor,
            d_col_tensor,
            sfa_tensor,
            sfd_row_tensor,
            sfd_col_tensor,
            amax_tensor,
            norm_const_tensor_cute,
            padded_offsets_tensor,
            alpha_tensor,
            prob_tensor,
            dprob_tensor,
            dbias_tensor,
            d_srelu_tensor,
            sfd_col_d_srelu_tensor,
            max_active_clusters,
            fake_stream,
            options="--enable-tvm-ffi",
        )

        self._n = n
        self._k = k
        self._b_stride_size = b_stride_size

        cached_workspace_ptr = from_dlpack(self._workspace, assumed_align=128).iterator
        cached_n = cutlass.Int32(self._n)
        cached_k = cutlass.Int32(self._k)
        cached_b_stride = cutlass.Int64(self._b_stride_size)

        def tensor_api(
            a_tensor: torch.Tensor,
            b_ptrs_device: torch.Tensor,
            sfb_ptrs_device: torch.Tensor,
            c_tensor: torch.Tensor,
            d_row_tensor: torch.Tensor,
            d_col_tensor: Optional[torch.Tensor],
            d_srelu_tensor: Optional[torch.Tensor],
            sfa_tensor: torch.Tensor,
            sfd_row_tensor: Optional[torch.Tensor],
            sfd_col_tensor: Optional[torch.Tensor],
            sfd_col_d_srelu_tensor: Optional[torch.Tensor],
            amax_tensor: Optional[torch.Tensor],
            norm_const_tensor: Optional[torch.Tensor],
            padded_offsets: torch.Tensor,
            alpha_tensor: torch.Tensor,
            prob_tensor: torch.Tensor,
            dprob_tensor: torch.Tensor,
            dbias_tensor: Optional[torch.Tensor],
            stream: cuda.CUstream,
        ) -> None:
            norm_const_tensor = self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const")
            b_ptrs_addr = int(get_data_ptr(b_ptrs_device))
            sfb_ptrs_addr = int(get_data_ptr(sfb_ptrs_device))

            _compiled_kernel(
                a_tensor,
                b_ptrs_addr,
                sfb_ptrs_addr,
                cached_n,
                cached_k,
                cached_b_stride,
                cached_workspace_ptr,
                c_tensor,
                d_row_tensor,
                d_col_tensor,
                sfa_tensor,
                sfd_row_tensor,
                sfd_col_tensor,
                amax_tensor,
                norm_const_tensor,
                padded_offsets,
                alpha_tensor,
                prob_tensor,
                dprob_tensor,
                dbias_tensor,
                d_srelu_tensor,
                sfd_col_d_srelu_tensor,
                stream,
            )

        self._compiled_kernel = tensor_api

    def execute(
        self,
        a_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_row_tensor: torch.Tensor,
        d_col_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        prob_tensor: torch.Tensor,
        dprob_tensor: torch.Tensor,
        # Dense mode:
        b_tensor: Optional[torch.Tensor] = None,
        sfb_tensor: Optional[torch.Tensor] = None,
        dbias_tensor: Optional[torch.Tensor] = None,
        # Determinism scratch, required iff the op was built with deterministic=True:
        dprob_workspace_tensor: Optional[torch.Tensor] = None,
        dbias_workspace_tensor: Optional[torch.Tensor] = None,
        # Discrete mode:
        b_ptrs: Optional[torch.Tensor] = None,
        sfb_ptrs: Optional[torch.Tensor] = None,
        # Optional:
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        d_srelu_tensor: Optional[torch.Tensor] = None,
        sfd_col_d_srelu_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Execute the compiled kernel.

        For dense mode, supply ``b_tensor`` and ``sfb_tensor``.
        For discrete mode, supply ``b_ptrs`` and ``sfb_ptrs``.

        :param a_tensor: Input A tensor (gradient input)
        :param c_tensor: Forward activations input
        :param d_row_tensor: Output D row tensor
        :param d_col_tensor: Output D column tensor
        :param sfa_tensor: Scale factor A
        :param padded_offsets: End offset per expert after padding
        :param alpha_tensor: Per-group alpha scaling factors
        :param prob_tensor: Per-row probability (from forward)
        :param dprob_tensor: Gradient of probability (output, must be zero-initialized)
        :param b_tensor: (Dense) Input B tensor (weights)
        :param sfb_tensor: (Dense) Scale factor B
        :param dbias_tensor: Optional dbias output tensor.
        :param dprob_workspace_tensor: Determinism scratch, required iff deterministic=True
        :param dbias_workspace_tensor: Determinism scratch, required iff deterministic=True and dbias
        :param b_ptrs: (Discrete) 1-D int64 device tensor of per-expert B data pointers
        :param sfb_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB data pointers
        :param sfd_row_tensor: Optional row scale factor D
        :param sfd_col_tensor: Optional column scale factor D
        :param amax_tensor: Optional amax tensor
        :param norm_const_tensor: Optional normalization constant
        :param current_stream: CUDA stream
        """
        self._logger.debug("Entering execute")
        if current_stream is None:
            # torch inputs stay ordered with the caller's current torch stream;
            # other frameworks (e.g. JAX) default to the CUDA legacy default stream.
            current_stream = default_stream(detect_framework(a_tensor))

        if a_tensor.shape[0] == 0:
            self._logger.debug("execute: valid_m is zero, skipping kernel execution")
            return
        self._runtime_error_if(
            self._compiled_kernel is None,
            "Kernel not compiled; call compile() first",
        )

        self._logger.debug("Executing grouped GEMM dSReLU kernel")
        if self._has_dbias:
            self._value_error_if(
                dbias_tensor is None,
                "dbias_tensor is required when GroupedGemmDsreluSm100 is configured with sample_dbias",
            )
        # Under determinism the kernel fills the scratch and the caller reduces it into
        # dprob/dbias afterwards; dprob_tensor/dbias_tensor are otherwise written directly.
        if self.deterministic:
            self._value_error_if(
                dprob_workspace_tensor is None,
                "dprob_workspace_tensor is required when the op was built with deterministic=True",
            )
            self._value_error_if(
                self._has_dbias and dbias_workspace_tensor is None,
                "dbias_workspace_tensor is required when the op was built with deterministic=True and sample_dbias",
            )
            dprob_tensor = dprob_workspace_tensor
            dbias_tensor = dbias_workspace_tensor if self._has_dbias else dbias_tensor

        if self.weight_mode == MoEWeightMode.DENSE:
            self._compiled_kernel(
                a_tensor=a_tensor,
                b_tensor=b_tensor,
                c_tensor=c_tensor,
                d_row_tensor=d_row_tensor,
                d_col_tensor=d_col_tensor,
                d_srelu_tensor=d_srelu_tensor,
                sfa_tensor=sfa_tensor,
                sfb_tensor=sfb_tensor,
                sfd_row_tensor=sfd_row_tensor,
                sfd_col_tensor=sfd_col_tensor,
                sfd_col_d_srelu_tensor=sfd_col_d_srelu_tensor,
                amax_tensor=amax_tensor,
                norm_const_tensor=norm_const_tensor,
                padded_offsets=padded_offsets,
                alpha_tensor=alpha_tensor,
                prob_tensor=prob_tensor,
                dprob_tensor=dprob_tensor,
                dbias_tensor=dbias_tensor,
                stream=current_stream,
            )
        else:
            if not is_torch_tensor(b_ptrs):
                # No record_stream equivalent for immutable frameworks (e.g. JAX): keep the
                # pointer arrays referenced until the next execute so their buffers outlive
                # the asynchronous launch.
                self._live_ptrs = (b_ptrs, sfb_ptrs)
            self._compiled_kernel(
                a_tensor=a_tensor,
                b_ptrs_device=b_ptrs,
                sfb_ptrs_device=sfb_ptrs,
                c_tensor=c_tensor,
                d_row_tensor=d_row_tensor,
                d_col_tensor=d_col_tensor,
                d_srelu_tensor=d_srelu_tensor,
                sfa_tensor=sfa_tensor,
                sfd_row_tensor=sfd_row_tensor,
                sfd_col_tensor=sfd_col_tensor,
                sfd_col_d_srelu_tensor=sfd_col_d_srelu_tensor,
                amax_tensor=amax_tensor,
                norm_const_tensor=norm_const_tensor,
                padded_offsets=padded_offsets,
                alpha_tensor=alpha_tensor,
                prob_tensor=prob_tensor,
                dprob_tensor=dprob_tensor,
                dbias_tensor=dbias_tensor,
                stream=current_stream,
            )

        self._logger.debug("Execute completed")


_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmDsreluSm100Objects = {}


def grouped_gemm_dsrelu_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: Optional[torch.Tensor] = None,
    c_tensor: Optional[torch.Tensor] = None,
    sfa_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    padded_offsets: Optional[torch.Tensor] = None,
    alpha_tensor: Optional[torch.Tensor] = None,
    prob_tensor: Optional[torch.Tensor] = None,
    dprob_tensor: Optional[torch.Tensor] = None,
    # generate_dbias is optional in both modes:
    generate_dbias: bool = False,
    # Discrete mode:
    b_ptrs: Optional[torch.Tensor] = None,
    sfb_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    # Common:
    norm_const_tensor: Optional[torch.Tensor] = None,
    acc_dtype: Optional[torch.dtype] = None,
    d_dtype: Optional[torch.dtype] = None,
    cd_major: str = "n",
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    use_dynamic_sched: bool = False,
    use_dsrelu_reuse: bool = False,
    deterministic: Optional[bool] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convenience wrapper for grouped GEMM dSReLU backward operation.

    Auto-detects dense vs. discrete mode based on which weight arguments
    are provided.

    Dense mode: provide ``b_tensor`` and ``sfb_tensor``.
    Discrete mode: provide ``b_ptrs``, ``sfb_ptrs``, ``n``, and ``b_dtype``.

    Compiled kernels are cached for reuse when called with the same configuration.

    Args:
        a_tensor: Input A tensor (valid_m, k, 1) -- gradient input
        c_tensor: Forward activations input (valid_m, n_out, 1)
        sfa_tensor: Scale factor A
        padded_offsets: End offset per expert after padding
        alpha_tensor: Per-group alpha scaling
        prob_tensor: Per-row probability (from forward)
        dprob_tensor: Gradient of probability (output, must be zero-initialized)
        b_tensor: (Dense) Weight B tensor (n, k, l)
        sfb_tensor: (Dense) Scale factor B
        generate_dbias: Optional flag to allocate and return dbias output
        b_ptrs: (Discrete) 1-D int64 device tensor of per-expert B data pointers
        sfb_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB data pointers
        n: (Discrete) B weight N dimension
        b_dtype: (Discrete) B weight data type
        b_major: (Discrete) B tensor major dimension ("k" or "n")
        norm_const_tensor: Optional normalization constant
        acc_dtype: Accumulator data type
        d_dtype: Output D tensor data type
        cd_major: CD major dimension (only "n" supported)
        mma_tiler_mn: MMA tiler shape
        cluster_shape_mn: Cluster shape
        sf_vec_size: Scale factor vector size
        vector_f32: Use vectorized f32
        m_aligned: M alignment (must be 256)
        discrete_col_sfd: Generate discrete col-major scale factor tensor
        use_dynamic_sched: Enable dynamic tile scheduling for load balancing
        use_dsrelu_reuse: Reuse relu(C)^2 between d_srelu and dprob
        deterministic: Make dprob and dbias bit-exact run to run; raises NotImplementedError
            for any framework other than torch. Every returned tensor keeps its usual shape and
            dtype -- the slot workspaces and the fixed-order reductions into them are internal to
            this wrapper. ``None`` (default) follows ``torch.use_deterministic_algorithms``.
            See docs/fe-oss-apis/gemm_fusions/grouped_gemm_dsrelu.md#deterministic-dprob.
        current_stream: CUDA stream

    Returns:
        TupleDict with keys: d_row_tensor, d_col_tensor, dprob_tensor,
            dbias_tensor, amax_tensor, sfd_row_tensor, sfd_col_tensor
    """
    framework = detect_framework(a_tensor)
    if framework not in ("torch", "jax"):
        raise ValueError(f"Unsupported tensor framework '{framework}' for grouped_gemm_dsrelu_wrapper_sm100; pass torch tensors or JAX arrays")

    acc_dtype = _convert_to_cutlass_data_type(acc_dtype) if acc_dtype is not None else cutlass.Float32
    d_dtype = _convert_to_cutlass_data_type(d_dtype) if d_dtype is not None else cutlass.BFloat16
    b_dtype = _convert_to_cutlass_data_type(b_dtype) if b_dtype is not None else None

    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None

    if is_dense and is_discrete:
        raise ValueError("Provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs), not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs)")

    if framework == "jax":
        if is_dense:
            raise ValueError(
                "Dense weight mode (b_tensor/sfb_tensor) is not expressible as JAX arrays "
                "(the expert-outermost strided B layout (n, k, l) has no row-major equivalent); "
                "use discrete mode (b_ptrs/sfb_ptrs) with per-expert weight pointers"
            )
        if _convert_to_cutlass_data_type(a_tensor.dtype) in (cutlass.Float4E2M1FN, cutlass.Uint8) or b_dtype in (cutlass.Float4E2M1FN, cutlass.Uint8):
            raise ValueError(
                "Packed fp4 A/B tensors (float4_e2m1fn / raw uint8) are not expressible as JAX arrays "
                "(JAX has no packed fp4 dtype); use fp8 inputs from JAX, or torch tensors for fp4"
            )
    if framework == "torch":
        import torch

    valid_m, k_physical, _ = get_shape(a_tensor)

    if is_dense:
        weight_mode = MoEWeightMode.DENSE
        n_weight, _, l = b_tensor.shape
    else:
        weight_mode = MoEWeightMode.DISCRETE
        num_experts = _validate_pointer_tensor(b_ptrs, "b_ptrs")
        _validate_pointer_tensor(sfb_ptrs, "sfb_ptrs", num_experts)
        if n is None or b_dtype is None:
            raise ValueError("n and b_dtype are required for discrete mode")
        n_weight = n
        k_logical = k_physical * 2 if b_dtype in (cutlass.Float4E2M1FN, cutlass.Uint8) else k_physical
        b_shape = (n_weight, k_logical)
        l = num_experts

    n_out = n_weight

    _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: Creating output tensors")

    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    if framework == "jax":
        import jax.numpy as jnp

    def _jax_alloc(builder):
        import jax

        # The kernel writes into these buffers on the launch stream, outside XLA's
        # tracking; materialize them before their pointers are taken.
        return jax.block_until_ready(builder())

    if framework == "torch":
        d_torch_dtype = framework_dtype(d_dtype, "torch")
        # Inside the context, like the accumulators: torch's allocator reuses a block freed on the
        # allocating stream on the strength of that stream's ordering, which says nothing about a
        # kernel writing the block on current_stream. record_stream cannot fix this direction --
        # it defers reuse after *this* tensor is freed, not the handout of an already-freed block.
        with _torch_stream_context(current_stream, a_tensor.device):
            d_row_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_torch_dtype, device=a_tensor.device)
            d_col_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_torch_dtype, device=a_tensor.device)
            d_srelu_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_torch_dtype, device=a_tensor.device)
    else:
        # n-major C-contiguous; the extent-1 batch dim's stride is unobservable by the kernel.
        d_jax_dtype = framework_dtype(d_dtype, "jax")
        d_row_tensor = _jax_alloc(lambda: jnp.empty((valid_m, n_out, 1), dtype=d_jax_dtype, device=a_tensor.device))
        d_col_tensor = _jax_alloc(lambda: jnp.empty((valid_m, n_out, 1), dtype=d_jax_dtype, device=a_tensor.device))
        d_srelu_tensor = _jax_alloc(lambda: jnp.empty((valid_m, n_out, 1), dtype=d_jax_dtype, device=a_tensor.device))

    sfd_row_tensor = None
    sfd_col_tensor = None
    sfd_col_d_srelu_tensor = None
    amax_tensor = None
    dbias_tensor = None

    # Deterministic mode gives dprob one slot per N-tile so each (token, tile_n) pair has a
    # single writer; the reduction back to (valid_m, 1, 1) happens after execute(). Resolved
    # here, with the other arguments, so `deterministic` is a plain bool from this point on.
    if deterministic is None:
        # Follows torch's process-wide flag. jax has no equivalent global, so there the opt-in has
        # to be explicit rather than inherited. Written as a nested branch, not
        # `framework == "torch" and torch...`: `torch` is only bound by the conditional import
        # above, so that form is one operand-reorder away from a NameError under jax.
        deterministic = False
        if framework == "torch":
            deterministic = torch.are_deterministic_algorithms_enabled()
    if deterministic and framework != "torch":
        # The slot reductions below are torch ops on torch streams. Rather than half-support it,
        # say so: a caller who asked for reproducibility should not get it silently unenforced.
        raise NotImplementedError(f"deterministic=True is only implemented for the torch framework, got {framework}")
    dprob_n_slots = _dprob_n_slots(n_out, mma_tiler_mn, cluster_shape_mn, deterministic)
    cta_tile_m = _cta_tile_m(mma_tiler_mn)
    if dprob_tensor is None:
        if framework == "torch":
            with _torch_stream_context(current_stream, a_tensor.device):
                dprob_tensor = torch.zeros((valid_m, 1, 1), dtype=torch.float32, device=a_tensor.device)
        else:
            dprob_tensor = _jax_alloc(lambda: jnp.zeros((valid_m, 1, 1), dtype=jnp.float32, device=a_tensor.device))
    # Determinism scratch, allocated here so the wrapper's own signature does not change with
    # the flag. torch-only, rejected for jax above.
    dprob_workspace = None
    if deterministic:
        with _torch_stream_context(current_stream, a_tensor.device):
            dprob_workspace = torch.zeros((valid_m, dprob_n_slots, 1), dtype=torch.float32, device=a_tensor.device)

    if _convert_to_cutlass_data_type(a_tensor.dtype) in (
        cutlass.Float8E4M3FN,
        cutlass.Float8E5M2,
    ) and _convert_to_cutlass_data_type(
        sfa_tensor.dtype
    ) in (cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN):
        _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: Detected fp8 config, constructing sfd tensors")

        sf_dtype = sfa_tensor.dtype
        sf_k_row = ceil_div(n_out, sf_vec_size)
        mma_shape_row = (1, ceil_div(valid_m, 128), ceil_div(sf_k_row, 4), 32, 4, 4)
        sf_k_col = ceil_div(valid_m, sf_vec_size)
        mma_shape_col = (1, ceil_div(n_out, 128), ceil_div(sf_k_col, 4), 32, 4, 4)
        if framework == "torch":
            mma_permute_order = (3, 4, 1, 5, 2, 0)
            with _torch_stream_context(current_stream, a_tensor.device):
                sfd_row_tensor = torch.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)
                sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)
                sfd_col_d_srelu_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)
        else:
            # Physical C-contiguous atom allocations: JAX cannot express the permuted view;
            # the kernel rebuilds the SF layout from the GEMM shapes and consumes only the
            # SF base pointer, so the physical form is byte-identical.
            sfd_row_tensor = _jax_alloc(lambda: jnp.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device))
            sfd_col_tensor = _jax_alloc(lambda: jnp.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device))
            sfd_col_d_srelu_tensor = _jax_alloc(lambda: jnp.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device))

    if d_dtype in (cutlass.BFloat16, cutlass.Float16):
        _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: Constructing amax_tensor")
        if framework == "torch":
            with _torch_stream_context(current_stream, a_tensor.device):
                amax_tensor = torch.full((l, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)
        else:
            amax_tensor = _jax_alloc(lambda: jnp.full((l, 1), float("-inf"), dtype=jnp.float32, device=a_tensor.device))
    # Deterministic dbias is a separate buffer, not a reshaped one: the kernel writes fp32
    # dbias keeps its (expert, n, 1) bf16 shape in both modes; under determinism the kernel fills
    # the M-block scratch instead and _reduce_dbias_slots segment-sums it into this tensor.
    dbias_workspace = None
    if generate_dbias:
        if framework == "torch":
            with _torch_stream_context(current_stream, a_tensor.device):
                dbias_tensor = torch.zeros((l, n_out, 1), dtype=torch.bfloat16, device=a_tensor.device)
                if deterministic:
                    dbias_workspace = torch.zeros((ceil_div(valid_m, cta_tile_m), n_out, 1), dtype=torch.bfloat16, device=a_tensor.device)
        else:
            dbias_tensor = _jax_alloc(lambda: jnp.zeros((l, n_out, 1), dtype=framework_dtype(cutlass.BFloat16, "jax"), device=a_tensor.device))
    dbias_kernel_tensor = dbias_workspace if dbias_workspace is not None else dbias_tensor

    if valid_m == 0:
        _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: valid_m is zero, skipping kernel execution")
        return TupleDict(
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            d_srelu_tensor=d_srelu_tensor,
            # Reduced here too, so a zero-token call returns the documented (valid_m, 1, 1)
            # shape rather than leaking the slot dimension.
            dprob_tensor=_reduce_dprob_slots(dprob_workspace, dprob_tensor, current_stream),
            dbias_tensor=_reduce_dbias_slots(dbias_workspace, dbias_tensor, current_stream, padded_offsets, cta_tile_m),
            amax_tensor=amax_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            sfd_col_d_srelu_tensor=sfd_col_d_srelu_tensor,
        )

    # ---- Build cache key ----
    def stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(i for i, s in sorted(enumerate(get_strides(tensor)), key=lambda x: x[1]))

    def tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return get_shape(tensor), get_strides(tensor), _convert_to_cutlass_data_type(tensor.dtype)

    def dynamic_tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return None, stride_order(tensor), _convert_to_cutlass_data_type(tensor.dtype)

    def dynamic_m_tensor_signature(
        tensor: Optional[torch.Tensor], static_shape_suffix: Optional[Tuple[int, ...]], dynamic_stride_dims: Tuple[int, ...] = ()
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        stride_signature = tuple(None if i in dynamic_stride_dims else s for i, s in enumerate(get_strides(tensor)))
        return static_shape_suffix, stride_signature, _convert_to_cutlass_data_type(tensor.dtype)

    def _sf_is_physical(tensor) -> bool:
        shape = get_shape(tensor)
        return not (len(shape) == 6 and shape[0] == 32 and shape[1] == 4 and shape[3] == 4)

    def dynamic_m_sf_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        """M-independent signature of an SF tensor in either atom form (see _sf_desc_is_physical)."""
        if tensor is None:
            return None, None, None
        shape = get_shape(tensor)
        if not _sf_is_physical(tensor):
            # torch-style permuted view: M' at dim 2, M-dependent stride at dim 5
            return dynamic_m_tensor_signature(tensor, (shape[4], 1), dynamic_stride_dims=(5,))
        # physical C-contiguous form (e.g. JAX): M' at dim 1, M-dependent stride at dim 0
        static_shape = (shape[0], None, *shape[2:])
        return dynamic_m_tensor_signature(tensor, static_shape, dynamic_stride_dims=(0,))

    def dynamic_sfd_col_tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        shape = get_shape(tensor)
        if not _sf_is_physical(tensor):
            static_shape = (shape[0], shape[1], shape[2], shape[3], shape[5])
            return dynamic_m_tensor_signature(tensor, static_shape, dynamic_stride_dims=(2, 5))
        # physical C-contiguous form (e.g. JAX): M_rest at dim 2, M-dependent strides at dims 0-1
        static_shape = (shape[0], shape[1], None, *shape[3:])
        return dynamic_m_tensor_signature(tensor, static_shape, dynamic_stride_dims=(0, 1))

    # Deterministic dbias indexes by M-block, so its dim 0 follows valid_m and keying on the full
    # shape would recompile for every distinct token count -- the trap _dprob_n_slots documents
    # for dprob's slot extent, and _make_dbias_fake already keeps that extent symbolic. Key it
    # the way dprob is keyed: M dynamic, the rest static. Strides are (n_out, 1, 1) either way,
    # so they carry no M. Non-deterministic dbias is (expert_cnt, n_out, 1) and static already.
    dbias_signature = (
        dynamic_m_tensor_signature(dbias_kernel_tensor, tuple(get_shape(dbias_kernel_tensor)[1:]))
        if deterministic and dbias_kernel_tensor is not None
        else tensor_signature(dbias_kernel_tensor)
    )
    # Used on both arms, not just the non-full-dynamic one. Every other tensor may drop its shape
    # under full dynamic because the kernel compiles with symbolic extents, but _make_dbias_fake
    # frees only dim 0 -- dbias's n stays baked -- and nothing else in the dense key carries n
    # (b_tensor.shape[2] is l, and dprob_n_slots is too coarse: n=384 and n=512 both give 2).
    # Verified: with the shape dropped, n=384 then n=512 reuses the first kernel and CuTeDSL
    # rejects the call with "Mismatched dbias_tensor.shape[1] ... expected to be 384" (job 469460).
    # The kernel writes the scratch under determinism, so that is what the key has to describe.
    dprob_kernel_tensor = dprob_workspace if dprob_workspace is not None else dprob_tensor

    use_full_dynamic = is_dense and os.environ.get("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1") != "0"

    if is_dense:
        cache_key = (
            weight_mode,
            use_full_dynamic,
            a_tensor.shape[1:] if not use_full_dynamic else None,
            b_tensor.shape[2] if use_full_dynamic else tuple(b_tensor.shape),
            c_tensor.shape[1:] if not use_full_dynamic else None,
            a_tensor.dtype,
            b_tensor.dtype,
            c_tensor.dtype,
            stride_order(a_tensor),
            stride_order(b_tensor),
            stride_order(c_tensor),
            *(
                dynamic_tensor_signature(sfa_tensor)
                if use_full_dynamic
                else dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1) if sfa_tensor is not None else None, dynamic_stride_dims=(5,))
            ),
            *tensor_signature(alpha_tensor),
            *(dynamic_m_tensor_signature(prob_tensor, (1, 1)) if not use_full_dynamic else dynamic_tensor_signature(prob_tensor)),
            *(
                dynamic_m_tensor_signature(dprob_kernel_tensor, tuple(dprob_kernel_tensor.shape[1:]))
                if not use_full_dynamic
                else dynamic_tensor_signature(dprob_kernel_tensor)
            ),
            # Explicit, because the signature above does not carry it under use_full_dynamic:
            # that path drops the shape entirely, yet the slot count is baked into the
            # compiled descriptor. n_out=512 and n_out=513 give 2 and 3 slots with identical
            # stride order, so without this the second call would reuse a kernel built for
            # the wrong dprob extent.
            dprob_n_slots,
            *dbias_signature,
            *(dynamic_m_tensor_signature(d_srelu_tensor, (n_out, 1)) if not use_full_dynamic else dynamic_tensor_signature(d_srelu_tensor)),
            *(dynamic_tensor_signature(sfb_tensor) if use_full_dynamic else tensor_signature(sfb_tensor)),
            *(dynamic_tensor_signature(sfd_col_d_srelu_tensor) if use_full_dynamic else tensor_signature(sfd_col_d_srelu_tensor)),
            norm_const_tensor.shape if norm_const_tensor is not None else None,
            norm_const_tensor.stride() if norm_const_tensor is not None else None,
            norm_const_tensor.dtype if norm_const_tensor is not None else None,
            tuple(padded_offsets.shape),
            tuple(padded_offsets.stride()),
            padded_offsets.dtype,
            acc_dtype,
            d_dtype,
            cd_major,
            mma_tiler_mn,
            cluster_shape_mn,
            sf_vec_size,
            vector_f32,
            m_aligned,
            discrete_col_sfd,
            use_dynamic_sched,
            use_dsrelu_reuse,
            deterministic,
        )
    else:
        cache_key = (
            weight_mode,
            *dynamic_m_tensor_signature(a_tensor, get_shape(a_tensor)[1:], dynamic_stride_dims=(2,)),
            b_shape,
            b_dtype,
            *dynamic_m_tensor_signature(c_tensor, get_shape(c_tensor)[1:], dynamic_stride_dims=(2,)),
            *dynamic_m_sf_signature(sfa_tensor),
            *tensor_signature(alpha_tensor),
            *dynamic_m_tensor_signature(prob_tensor, (1, 1)),
            *dynamic_m_tensor_signature(dprob_kernel_tensor, tuple(dprob_kernel_tensor.shape[1:])),
            *dbias_signature,
            *dynamic_m_tensor_signature(d_srelu_tensor, (n_out, 1), dynamic_stride_dims=(2,)),
            *dynamic_sfd_col_tensor_signature(sfd_col_d_srelu_tensor),
            *tensor_signature(norm_const_tensor),
            *tensor_signature(b_ptrs),
            *tensor_signature(sfb_ptrs),
            *tensor_signature(padded_offsets),
            acc_dtype,
            d_dtype,
            cd_major,
            mma_tiler_mn,
            cluster_shape_mn,
            sf_vec_size,
            vector_f32,
            m_aligned,
            discrete_col_sfd,
            use_dynamic_sched,
            use_dsrelu_reuse,
            deterministic,
            b_major,
            num_experts,
        )

    # ---- Cache lookup or create + compile ----
    if cache_key in _cache_of_GroupedGemmDsreluSm100Objects:
        _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: Using cached object")
        api = _cache_of_GroupedGemmDsreluSm100Objects[cache_key]
    else:
        _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: Creating new object")
        if is_dense:
            api = GroupedGemmDsreluSm100(
                sample_a=a_tensor,
                sample_c=c_tensor,
                sample_d_row=d_row_tensor,
                sample_d_col=d_col_tensor,
                sample_d_srelu=d_srelu_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_prob=prob_tensor,
                sample_dprob=dprob_tensor,
                sample_dbias=dbias_tensor,
                sample_dprob_workspace=dprob_workspace,
                sample_dbias_workspace=dbias_workspace,
                sample_b=b_tensor,
                sample_sfb=sfb_tensor,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_sfd_col_d_srelu=sfd_col_d_srelu_tensor,
                sample_amax=amax_tensor,
                sample_norm_const=norm_const_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                use_dynamic_sched=use_dynamic_sched,
                use_dsrelu_reuse=use_dsrelu_reuse,
                deterministic=deterministic,
            )
        else:
            api = GroupedGemmDsreluSm100(
                sample_a=a_tensor,
                sample_c=c_tensor,
                sample_d_row=d_row_tensor,
                sample_d_col=d_col_tensor,
                sample_d_srelu=d_srelu_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_prob=prob_tensor,
                sample_dprob=dprob_tensor,
                sample_dbias=dbias_tensor,
                sample_dprob_workspace=dprob_workspace,
                sample_dbias_workspace=dbias_workspace,
                num_experts=num_experts,
                b_shape=b_shape,
                b_dtype=b_dtype,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_sfd_col_d_srelu=sfd_col_d_srelu_tensor,
                sample_amax=amax_tensor,
                sample_norm_const=norm_const_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                b_major=b_major,
                use_dynamic_sched=use_dynamic_sched,
                use_dsrelu_reuse=use_dsrelu_reuse,
                deterministic=deterministic,
            )

        if not api.check_support():
            raise RuntimeError("Unsupported configuration")
        api.compile()
        _cache_of_GroupedGemmDsreluSm100Objects[cache_key] = api

    # ---- Execute ----
    if is_dense:
        api.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            d_srelu_tensor=d_srelu_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            dbias_tensor=dbias_tensor,
            dprob_workspace_tensor=dprob_workspace,
            dbias_workspace_tensor=dbias_workspace,
            b_tensor=b_tensor,
            sfb_tensor=sfb_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            sfd_col_d_srelu_tensor=sfd_col_d_srelu_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            current_stream=current_stream,
        )
    else:
        api.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            d_srelu_tensor=d_srelu_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            dbias_tensor=dbias_tensor,
            dprob_workspace_tensor=dprob_workspace,
            dbias_workspace_tensor=dbias_workspace,
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            sfd_col_d_srelu_tensor=sfd_col_d_srelu_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            current_stream=current_stream,
        )

    return TupleDict(
        d_row_tensor=d_row_tensor,
        d_col_tensor=d_col_tensor,
        d_srelu_tensor=d_srelu_tensor,
        dprob_tensor=_reduce_dprob_slots(dprob_workspace, dprob_tensor, current_stream),
        dbias_tensor=_reduce_dbias_slots(dbias_workspace, dbias_tensor, current_stream, padded_offsets, cta_tile_m),
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
        sfd_col_d_srelu_tensor=sfd_col_d_srelu_tensor,
    )
