# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry point for grouped GEMM GLU + Hadamard, built on
:func:`cudnn.jax.call`.

``grouped_gemm_glu_hadamard_jax_forward`` is the canonical forward with dense weights
for packed FP4 inputs: A ``(m, k / 2)`` and B ``(experts, n, k / 2)`` as uint8, and
already MMA-packed scale bytes. ``padded_offsets`` values cannot be host-validated
under tracing; malformed offsets are the caller's responsibility here (the eager
wrapper validates them).
"""

import os

import jax.numpy as jnp
import ml_dtypes
import numpy as np

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.nvgpu import OperandMajorMode

from cudnn.api_base import TupleDict, ceil_div
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.jax import zeros_init
from cudnn.tensor_adapter import get_compute_capability
from ..canonical import kernel_facing_fp4
from ..canonical_jax import check_block_scaled_operands, check_jax_inputs, check_packed_sf_size, grouped_call, output_type, sf_array
from ..moe_utils import MoEWeightMode
from .hadamard_utils import HADAMARD_SIZE
from .moe_blockscaled_grouped_gemm_glu_hadamard import BlockScaledMoEGroupedGemmGluHadamardKernel

kernel_cache = {}
validated_configs = set()


def hadamard_array():
    """The fixed 16x16 Sylvester Hadamard matrix the kernel applies, as bfloat16."""
    matrix = np.ones((1, 1), np.float32)
    while matrix.shape[0] < HADAMARD_SIZE:
        matrix = np.kron(matrix, np.array([[1, 1], [1, -1]], np.float32))
    return jnp.asarray(matrix.T.astype(ml_dtypes.bfloat16))


@cute.jit
def grouped_glu_hadamard_adapter(stream, a, b, sfa, sfb, padded_offsets, alpha, prob, hadamard, c, d, amax, post_rht_amax, *, kernel, mac, linear_offset):
    kernel(
        a=kernel_facing_fp4(a),
        b=kernel_facing_fp4(b),
        sfa=sfa,
        sfb=sfb,
        n=cutlass.Int32(0),
        k=cutlass.Int32(0),
        b_stride_size=cutlass.Int64(0),
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=cute.make_ptr(cutlass.Uint8, 0, cute.AddressSpace.gmem, assumed_align=128),
        c=c,
        d=d,
        amax_tensor=amax,
        post_rht_amax_tensor=post_rht_amax,
        padded_offsets=padded_offsets,
        alpha=alpha,
        prob=prob,
        hadamard_tensor=hadamard,
        bias=None,
        max_active_clusters=mac,
        stream=stream,
        linear_offset=cutlass.Float32(linear_offset),
    )


def glu_hadamard_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn, sf_vec_size, act_func):
    a, b = inputs["a"], inputs["b"]
    m, k = a.shape
    experts, n, b_k = b.shape
    if m <= 0 or experts <= 0 or k != b_k:
        raise ValueError("Expected nonempty A/B with matching K and at least one expert")
    if m % 256:
        raise ValueError("A rows must be padded to a multiple of 256")
    ab_dtype, sf_dtype = check_block_scaled_operands(a, b, inputs["sfa"], inputs["sfb"], sf_vec_size)
    if ab_dtype is not cutlass.Float4E2M1FN:
        raise ValueError("a and b must be packed FP4 as uint8; this fusion is wired for FP4 inputs only")
    k *= 2
    if act_func not in ("swiglu", "geglu", "srelu"):
        raise ValueError(f"act_func must be 'swiglu', 'geglu' or 'srelu' for the JAX path, got {act_func!r}")
    if tuple(inputs["alpha"].shape) != (experts,) or _convert_to_cutlass_data_type(inputs["alpha"].dtype) is not cutlass.Float32:
        raise ValueError(f"alpha must have shape {(experts,)} and dtype float32")
    if inputs["prob"].shape != (m,) or _convert_to_cutlass_data_type(inputs["prob"].dtype) not in (cutlass.Float32, cutlass.BFloat16):
        raise ValueError("prob must have shape (m,) and dtype float32 or bfloat16")
    if inputs["padded_offsets"].shape != (experts,) or _convert_to_cutlass_data_type(inputs["padded_offsets"].dtype) is not cutlass.Int32:
        raise ValueError("padded_offsets must have shape (experts,) and dtype int32")
    d_dtype = _convert_to_cutlass_data_type(outputs["d"].dtype)
    if d_dtype not in (cutlass.BFloat16, cutlass.Float16):
        raise ValueError("d_dtype must be BF16 or FP16")
    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))
    margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    config = (experts, tuple(mma_tiler_mn), cluster_shape_mn, margin, sf_vec_size, act_func)
    validation_key = (config, tuple((name, tuple(t.shape), str(t.dtype)) for name, t in (*inputs.items(), *outputs.items())))
    if validation_key not in validated_configs:
        check_packed_sf_size(inputs, (("sfa", m, 1), ("sfb", n, experts)), ceil_div(ceil_div(k, sf_vec_size), 4))
        if experts > 1024:
            raise ValueError(f"expert count must be <= 1024, got {experts}")
        if not BlockScaledMoEGroupedGemmGluHadamardKernel.can_implement(
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            cutlass.Float32,
            d_dtype,
            use_2cta_instrs,
            tuple(mma_tiler_mn),
            cluster_shape_mn,
            m,
            n,
            k,
            experts,
            "k",
            "k",
            "n",
            BlockScaledMoEGroupedGemmGluHadamardKernel.FIX_PAD_SIZE,
        ):
            raise ValueError("Unsupported grouped GEMM GLU hadamard dtype, tile, cluster, alignment, or layout configuration")
        validated_configs.add(validation_key)
    if config not in kernel_cache:
        major, minor = get_compute_capability()
        if major < 10:
            raise RuntimeError(f"Grouped GEMM GLU hadamard requires SM100+, found SM{major}{minor}")
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - margin
        if mac <= 0:
            raise ValueError("CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves no active clusters")
        kernel = BlockScaledMoEGroupedGemmGluHadamardKernel(
            sf_vec_size=sf_vec_size,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=False,
            expert_cnt=experts,
            weight_mode=MoEWeightMode.DENSE,
            act_func=act_func,
        )
        kernel_cache[config] = (kernel, mac)
    return kernel_cache[config]


def grouped_gemm_glu_hadamard_jax_forward(
    a_tensor,
    b_tensor,
    sfa_tensor,
    sfb_tensor,
    padded_offsets,
    alpha_tensor,
    prob_tensor,
    *,
    c_dtype=cutlass.BFloat16,
    d_dtype=cutlass.BFloat16,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=None,
    sf_vec_size=16,
    act_func="swiglu",
):
    """Canonical FP4 grouped GEMM + GLU + Hadamard forward, eagerly or under jax.jit.

    Packed FP4 A (m,k/2) and dense B (experts,n,k/2) as uint8 with two values
    per byte, explicit alpha (experts,) fp32, prob (m,) fp32/bf16, and int32
    padded_offsets (experts,). Offsets must be nondecreasing multiples of 256
    within [0,m]; m is padded to 256. SF buffers contain packed MMA-tiled
    bytes at any dense rank: E4M3 with ``sf_vec_size=16`` for NVFP4, or E8M0.
    ``act_func`` is "swiglu", "geglu" or "srelu". Returns a TupleDict with the
    eager wrapper's keys: the pre-activation C (m,n), D (m,n/2), or (m,n) for
    "srelu", in a BF16/FP16 ``d_dtype``, and the per-expert amax and
    post-Hadamard amax (experts,1), which are zero for experts without rows.
    Rows at or past padded_offsets[-1] are unspecified, as in the torch path.
    SM100+, as in Torch, with the default 256x256 tile.
    """
    inputs = dict(a=a_tensor, b=b_tensor, sfa=sfa_tensor, sfb=sfb_tensor, padded_offsets=padded_offsets, alpha=alpha_tensor, prob=prob_tensor)
    check_jax_inputs(inputs)
    inputs["sfa"] = sf_array(sfa_tensor)
    inputs["sfb"] = sf_array(sfb_tensor)
    inputs["hadamard"] = hadamard_array()
    m = a_tensor.shape[0]
    experts, n = b_tensor.shape[:2]
    amax = output_type((experts, 1), cutlass.Float32)
    outputs = dict(c=output_type((m, n), c_dtype), d=output_type((m, n if act_func == "srelu" else n // 2), d_dtype), amax=amax, post_rht_amax=amax)
    kernel, mac = glu_hadamard_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn, sf_vec_size, act_func)
    result = grouped_call(
        grouped_glu_hadamard_adapter,
        kernel,
        mac,
        tuple(output_type(t.shape, t.dtype) for t in inputs.values()),
        tuple(outputs.values()),
        # Experts without rows launch no CTAs, so their amax must already hold the empty value.
        initialized=((2, zeros_init), (3, zeros_init)),
        linear_offset=1.0 if act_func == "geglu" else 0.0,
    )(*inputs.values())
    return TupleDict(c_tensor=result[0], d_tensor=result[1], amax_tensor=result[2], post_rht_amax_tensor=result[3])
