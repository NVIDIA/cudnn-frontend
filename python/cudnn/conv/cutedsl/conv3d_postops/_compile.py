# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache executables by device and specialization, independently of geometry."""

import math

import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from cudnn.frost.compiled_cache import compile_cached

from ._config import ConvSpecialization

OPTIONS = "--enable-tvm-ffi --generate-line-info --divmod-semantics=python"


def fake_flat(dtype=cutlass.BFloat16, alignment=16):
    return make_fake_compact_tensor(dtype, (cute.sym_int64(),), assumed_align=alignment)


def fake_geometry(wide=False):
    integer = cutlass.Int64 if wide else cutlass.Int32
    return tuple(integer(v) for v in (1, 3, 3, 3))


def compile_conv(device, spec: ConvSpecialization, mode, residual, residual_bias, wide):
    from .kernel import Conv3dPostOpsLaunch

    dtype = cutlass.BFloat16 if spec.precision == "bf16" else cutlass.Float32
    flat = lambda: fake_flat(dtype)
    parameter = make_fake_compact_tensor(dtype, (spec.co,), assumed_align=16)
    weight = make_fake_compact_tensor(dtype, (math.prod(spec.packed_weight_shape),), assumed_align=16)
    norm = mode in ("norm_silu", "norm_silu_pad")
    launch = Conv3dPostOpsLaunch(
        spec,
        fuse_norm=norm,
        has_residual=residual,
        has_residual_bias=residual_bias,
        spatial_output=mode == "bias_residual_pad",
        wide_index=wide,
        prepare_output=mode == "norm_silu_pad",
    )
    with torch.cuda.device(device):
        return compile_cached(
            launch,
            flat(),
            weight,
            flat(),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            parameter if mode != "raw" else None,
            parameter if norm else None,
            flat(),
            flat() if mode == "norm_silu_pad" else None,
            flat() if residual else None,
            parameter if residual_bias else None,
            flat() if norm and residual else None,
            geometry=fake_geometry(wide),
            history=cutlass.Int32(0),
            cache_key=f"conv3d_postops:conv:{spec}:{mode}:{residual}:{residual_bias}:{wide}",
            symbol="conv3d_postops_conv",
            options=OPTIONS,
        )


def compile_round(device):
    from ._tf32 import RoundInputLaunch

    with torch.cuda.device(device):
        return compile_cached(
            RoundInputLaunch(),
            fake_flat(cutlass.Float32),
            fake_flat(cutlass.Float32),
            cutlass.Int64(1),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            cache_key="conv3d_postops:round_tf32",
            symbol="conv3d_postops_round",
            options=OPTIONS,
        )


def compile_norm(device, storage_dtype, channels, wide, input_bias, residual, residual_bias, residual_output, previous):
    from .rmsnorm_silu_pad import RmsNormSiluPadLaunch

    dtype = cutlass.BFloat16 if storage_dtype == torch.bfloat16 else cutlass.Float32
    flat = lambda: fake_flat(dtype)
    parameter = make_fake_compact_tensor(dtype, (channels,), assumed_align=16)
    with torch.cuda.device(device):
        return compile_cached(
            RmsNormSiluPadLaunch(channels, wide, input_bias, residual, residual_bias),
            flat(),
            parameter,
            flat(),
            flat(),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            parameter if input_bias else None,
            flat() if residual else None,
            parameter if residual_bias else None,
            flat() if residual_output else None,
            flat() if previous else None,
            geometry=fake_geometry(wide),
            history=cutlass.Int32(0),
            cache_key=f"conv3d_postops:norm:{storage_dtype}:{channels}:{wide}:{input_bias}:{residual}:{residual_bias}:{residual_output}:{previous}",
            symbol="conv3d_postops_norm",
            options=OPTIONS,
        )


def compile_causal_pack(device, storage_dtype, previous):
    from .causal import CausalConv3dPackLaunch

    dtype = cutlass.BFloat16 if storage_dtype == torch.bfloat16 else cutlass.Float32
    with torch.cuda.device(device):
        return compile_cached(
            CausalConv3dPackLaunch(),
            fake_flat(dtype, 2),
            fake_flat(dtype),
            fake_flat(dtype),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            fake_flat(dtype, 8) if previous else None,
            geometry=fake_geometry(),
            input_strides=tuple(cutlass.Int64(1) for _ in range(5)),
            history=cutlass.Int32(0),
            cache_key=f"conv3d_postops:causal_pack:{storage_dtype}:{previous}",
            symbol="conv3d_postops_causal_pack",
            options=OPTIONS,
        )


def compile_causal_conv(device, ctas):
    from .causal import CausalConv3dLaunch

    with torch.cuda.device(device):
        return compile_cached(
            CausalConv3dLaunch(ctas),
            fake_flat(),
            make_fake_compact_tensor(cutlass.BFloat16, (160 * 448,), assumed_align=16),
            fake_flat(),
            fake_geometry(),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            cache_key=f"conv3d_postops:causal_conv:{ctas}",
            symbol="conv3d_postops_causal_conv",
            options=OPTIONS,
        )
