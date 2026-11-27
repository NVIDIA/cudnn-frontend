# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""
MoE BF16 Grouped GEMM + GLU Kernel for Hopper (SM90).

The SM90 counterpart of moe_grouped_gemm_glu_bias.py: the SM90 grouped GEMM with a fused
SwiGLU / GeGLU epilogue. B and C hold the gate/up branches in interleaved 32-column blocks
(``glu_layout="interleaved_32"``) or as gate columns [0, N/2) then up columns [N/2, N)
(``glu_layout="gate_then_up"``, each branch's N a multiple of 128); D has half of C's columns.

Supports:
    - Dynamic persistent tile scheduling only, dense K-major weights, no bias
    - Routing-probability (prob) fusion and optional pre-activation C output (generate_c)
"""

from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from ..glu_layout import validate_glu_layout
from ..moe_utils import MoEWeightMode
from ..unfused.moe_grouped_gemm_sm90 import MoEGroupedGemmBf16KernelSm90


class MoEGroupedGemmGluBf16KernelSm90(MoEGroupedGemmBf16KernelSm90):
    """BF16 grouped GEMM with a fused GLU for Hopper; takes the SM100 GLU kernel's arguments.

    :param act_func: "swiglu" or "geglu".
    :param glu_layout: "interleaved_32" or "gate_then_up". gate_then_up loads each tile's gate and up weight rows as
        two TMA boxes into the stage's halves and pairs accumulator column j with j + N_tile/2; C stores go to the
        branch's columns. The math per output element is the same as interleaved_32's.
    :param pingpong: The pingpong warp-group schedule (see the base kernel).
    :param streaming_signal_capacity, device_layout, use_pdl, launch_pdl_dependents_at_entry: a streaming input producer
        of A (see the base kernel).
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        generate_c: bool,
        enable_bias: bool,
        expert_cnt: int,
        weight_mode: MoEWeightMode = MoEWeightMode.DENSE,
        use_dynamic_sched: bool = True,
        act_func: str = "swiglu",
        glu_layout: str = "interleaved_32",
        pingpong: bool = False,
        streaming_signal_capacity: Optional[int] = None,
        device_layout: bool = False,
        use_pdl: bool = False,
        launch_pdl_dependents_at_entry: bool = False,
    ):
        if act_func not in ("swiglu", "geglu"):
            raise ValueError(f"act_func must be 'swiglu' or 'geglu', got {act_func!r}")
        validate_glu_layout(glu_layout)
        super().__init__(
            acc_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            vectorized_f32,
            generate_c,
            enable_bias,
            expert_cnt,
            weight_mode,
            use_dynamic_sched,
            act_func=act_func,
            pingpong=pingpong,
            streaming_signal_capacity=streaming_signal_capacity,
            device_layout=device_layout,
            use_pdl=use_pdl,
            launch_pdl_dependents_at_entry=launch_pdl_dependents_at_entry,
        )
        self.glu_layout = glu_layout
        self.gate_then_up = glu_layout == "gate_then_up"

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        cd_major: str,
        m_aligned: int,
        pingpong: bool = False,
    ) -> bool:
        # C carries the accumulator's N, D half of it: both must fit the base kernel's rules.
        args = (ab_dtype, c_dtype, d_dtype, acc_dtype, use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, m)
        layout = (k, l, a_major, b_major, cd_major, m_aligned, pingpong)
        return (
            b_major == "k"
            and mma_tiler_mn[1] % 64 == 0
            and n % 64 == 0
            and MoEGroupedGemmBf16KernelSm90.can_implement(*args, n, *layout)
            and MoEGroupedGemmBf16KernelSm90.can_implement(*args, n // 2, *layout)
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        n: Int32,
        k: Int32,
        b_stride_size: cutlass.Int64,
        b_major_mode: cutlass.Constexpr,
        workspace_ptr,
        c: cute.Tensor,
        d: cute.Tensor,
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        prob: cute.Tensor,
        bias: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        linear_offset: cutlass.Float32 = 0.0,
    ):
        """Execute the GEMM; ``prob`` comes before ``bias`` as in the SM100 GLU kernel.

        ``n``, ``k``, ``b_stride_size`` and ``bias`` are unused; ``linear_offset`` is the GeGLU up offset.
        """
        self._launch(a, b, b_major_mode, workspace_ptr, c, d, padded_offsets, alpha, prob, max_active_clusters, stream, linear_offset)
