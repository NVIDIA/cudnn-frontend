# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compilation and ABI metadata for Rubin MXFP8 dGLU backward."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Any

import torch

from ..._contracts import ForwardConfig
from .._plan import PreparedResources
from .._workspace import WorkspaceRequirements
from ._compile import (
    _pre_reduced_sf_workspace_metadata,
    _pre_reduced_workspace_metadata,
)
from ._config import Mxfp8KernelConfig
from ._formats import combine_wire_format
from ._launch import _to_cute, _to_cute_ptr


@dataclass(frozen=True)
class PreparedMxfp8BackwardKernel:
    config: Mxfp8KernelConfig
    device: torch.device
    architecture: tuple[int, int]
    kernel: Any
    launch_cluster_count: int
    workspace_requirements: WorkspaceRequirements
    pool_token_capacity: int
    pre_reduced_activation_offset: int
    pre_reduced_activation_bytes_per_token: int
    pre_reduced_activation_sf_offset: int | None
    pre_reduced_activation_sf_bytes_per_token: int
    local_workspace_zero_bytes: int
    shared_workspace_zero_bytes: int
    dfc2_recompute: bool
    dfc2_col_output: bool
    enable_grad_y2_col_quant: bool


@dataclass(frozen=True)
class Mxfp8BackwardLaunchInputs:
    grad_out: torch.Tensor
    grad_out_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    fc1_weight: torch.Tensor
    fc1_weight_sf: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_sf: torch.Tensor
    beta: torch.Tensor
    fc1_preact: torch.Tensor
    output_activation: torch.Tensor
    overflow_flag: torch.Tensor
    dprob: torch.Tensor
    fc1_recompute: torch.Tensor
    fc1_recompute_sf: torch.Tensor
    fc1_col_output: torch.Tensor
    fc1_col_output_sf: torch.Tensor
    grad_y2: torch.Tensor
    grad_y2_sf: torch.Tensor
    local_workspace: torch.Tensor
    shared_workspace: torch.Tensor
    token_count: int


@dataclass(frozen=True)
class CompiledMxfp8BackwardKernel:
    key: tuple
    callable: Any


_COMPILE_LOCK = threading.RLock()
_COMPILE_CACHE: dict[tuple, CompiledMxfp8BackwardKernel] = {}


def prepare_backward_kernel(
    forward_config: ForwardConfig,
    config: Mxfp8KernelConfig,
    device: torch.device,
) -> PreparedMxfp8BackwardKernel:
    """Instantiate the fixed Rubin dGLU specialization."""

    torch.cuda.set_device(device)
    architecture = torch.cuda.get_device_capability(device)
    if architecture != (10, 7):
        raise RuntimeError(
            "Rubin MXFP8 backward requires compute capability (10, 7), "
            f"got {architecture}"
        )
    configured_architecture = os.environ.get("CUTE_DSL_ARCH")
    if configured_architecture is None:
        os.environ["CUTE_DSL_ARCH"] = "sm_107a"
    elif configured_architecture not in ("sm_107", "sm_107a"):
        raise RuntimeError(
            "CUTE_DSL_ARCH must target SM107 for the Rubin MXFP8 backward"
        )
    import cutlass
    import cutlass.utils as utils

    from ..cutedsl_src.kernel_src.rubin.training.mega.bwd_dglu import (
        Sm107MegaMoEMxfp8DgluKernel,
    )
    from ..cutedsl_src.quant_def import CombineFormat

    launch_cluster_count = int(
        utils.HardwareInfo().get_max_active_clusters(config.cluster_size)
    )
    if launch_cluster_count <= 0:
        raise RuntimeError(
            "hardware occupancy query returned no launchable Rubin clusters"
        )
    group_hint = (
        launch_cluster_count
        if config.group_hint is None
        else config.group_hint
    )
    operands_mode = forward_config.backward_wgrad_mode == "operands"
    dfc2_recompute = operands_mode
    dfc2_col_output = operands_mode
    enable_grad_y2_col_quant = operands_mode
    kernel = Sm107MegaMoEMxfp8DgluKernel.from_kwargs(
        mma_tiler_mnk=config.mma_tiler_mnk,
        cluster_shape_mnk=config.cluster_shape_mnk,
        use_2cta_instrs=config.use_2cta_instrs,
        group_hint=group_hint,
        token_padding_block=config.token_padding_block,
        sf_padding_block=config.sf_padding_block,
        load_balance_mode=config.load_balance_mode,
        static_expert_shape=(
            config.num_experts,
            config.intermediate,
            config.hidden,
        ),
        force_static_sched=config.force_static_sched,
        clc_bundle_size=config.clc_bundle_size,
        num_sched_stages=config.num_sched_stages,
        ab_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=config.sf_vec_size,
        world_size=config.world_size,
        local_rank=0,
        num_topk=config.top_k,
        max_tokens_per_rank=config.max_tokens_per_rank,
        max_recv_size_per_rank=config.max_recv_size_per_rank,
        hidden=config.hidden,
        launch_cluster_count=launch_cluster_count,
        drop_on_overflow=config.drop_on_overflow,
        fc2_in_kernel_topk_reduce=False,
        token_back_mode="epi_warps",
        epi_flag_batch=config.epi_flag_batch,
        flag_batch=config.flag_batch,
        combine_format=CombineFormat.parse(
            combine_wire_format(forward_config.combine_format)
        ),
        act_func=config.act_func,
        gate_up_clamp=config.gate_up_clamp,
        dfc2_recompute=dfc2_recompute,
        dfc2_col_output=dfc2_col_output,
        enable_grad_y2_col_quant=enable_grad_y2_col_quant,
        num_ctas_grad_y2_col_quant=config.col_quant_num_ctas,
    )
    local_bytes, shared_bytes = kernel.get_workspace_sizes()
    local_zero, shared_zero = kernel.require_zero_workspace_leading_bytes
    device_workspace = kernel._mega_device_workspace
    pool_capacity = int(kernel.pool_token_capacity)
    fc1_preact_shape = tuple(
        int(extent) for extent in kernel.get_fc1_preact_shape()
    )
    expected_preact_shape = (
        pool_capacity,
        2 * config.intermediate,
    )
    if fc1_preact_shape != expected_preact_shape:
        raise RuntimeError(
            "Rubin dGLU fc1_preact shape mismatch: "
            f"{fc1_preact_shape} != {expected_preact_shape}"
        )
    aux_shapes = {
        name: tuple(int(extent) for extent in shape)
        for name, shape in kernel.get_aux_output_shapes().items()
    }
    fc1_preact_bytes = (
        math.prod(fc1_preact_shape) * torch.bfloat16.itemsize
    )
    dprob_bytes = math.prod(aux_shapes["dprob"]) * torch.float32.itemsize
    aux_data_bytes = max(
        math.prod(aux_shapes["fc1_recompute"]),
        math.prod(aux_shapes["fc1_col_output"]),
        math.prod(aux_shapes["grad_y2"]),
    ) * torch.float8_e4m3fn.itemsize
    aux_scale_bytes = max(
        math.prod(aux_shapes["fc1_recompute_sf"]),
        math.prod(aux_shapes["fc1_col_output_sf"]),
        math.prod(aux_shapes["grad_y2_sf"]),
    ) * torch.float8_e8m0fnu.itemsize
    requirements = WorkspaceRequirements.for_mxfp8(
        forward_config,
        kernel_local_workspace_bytes=local_bytes,
        kernel_shared_workspace_bytes=shared_bytes,
        backward_fc1_preact_bytes=fc1_preact_bytes,
        backward_dprob_bytes=dprob_bytes,
        backward_aux_data_bytes=aux_data_bytes,
        backward_aux_scale_bytes=aux_scale_bytes,
    )
    pre_reduced_offset, pre_reduced_bytes_per_token = (
        _pre_reduced_workspace_metadata(
            device_workspace,
            config,
            shared_bytes,
        )
    )
    if pre_reduced_offset is None or pre_reduced_bytes_per_token <= 0:
        raise RuntimeError(
            "Rubin MXFP8 backward requires standalone pre-reduced activation"
        )
    pre_reduced_sf_offset, pre_reduced_sf_bytes_per_token = (
        _pre_reduced_sf_workspace_metadata(
            device_workspace,
            config,
            shared_bytes,
        )
    )
    return PreparedMxfp8BackwardKernel(
        config=config,
        device=torch.device(device),
        architecture=architecture,
        kernel=kernel,
        launch_cluster_count=launch_cluster_count,
        workspace_requirements=requirements,
        pool_token_capacity=pool_capacity,
        pre_reduced_activation_offset=pre_reduced_offset,
        pre_reduced_activation_bytes_per_token=pre_reduced_bytes_per_token,
        pre_reduced_activation_sf_offset=pre_reduced_sf_offset,
        pre_reduced_activation_sf_bytes_per_token=(
            pre_reduced_sf_bytes_per_token
        ),
        local_workspace_zero_bytes=int(local_zero),
        shared_workspace_zero_bytes=int(shared_zero),
        dfc2_recompute=dfc2_recompute,
        dfc2_col_output=dfc2_col_output,
        enable_grad_y2_col_quant=enable_grad_y2_col_quant,
    )


def _layout_signature(inputs: Mxfp8BackwardLaunchInputs) -> tuple:
    tensors = tuple(
        value
        for value in inputs.__dict__.values()
        if isinstance(value, torch.Tensor)
    )
    return tuple(
        (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        for tensor in tensors
    )


def build_backward_runtime_kwargs(
    inputs: Mxfp8BackwardLaunchInputs,
    resources: PreparedResources,
) -> dict[str, Any]:
    import cuda.bindings.driver as cuda

    stream = resources.runtime.current_stream()
    return {
        "grad_out": _to_cute(inputs.grad_out),
        "grad_out_sf": _to_cute(inputs.grad_out_sf),
        "topk_idx": _to_cute(inputs.topk_idx),
        "topk_weights": _to_cute(inputs.topk_weights, assumed_align=4),
        "fc1_weight": _to_cute(inputs.fc1_weight),
        "fc1_weight_sf": _to_cute(inputs.fc1_weight_sf),
        "fc2_weight": _to_cute(inputs.fc2_weight),
        "fc2_weight_sf": _to_cute(inputs.fc2_weight_sf),
        "beta": _to_cute(inputs.beta, assumed_align=4),
        "fc1_preact": _to_cute(
            inputs.fc1_preact,
            assumed_align=128,
            dynamic_layout=False,
        ),
        "output_activation": _to_cute(inputs.output_activation),
        "overflow_flag": _to_cute(
            inputs.overflow_flag,
            assumed_align=4,
            dynamic_layout=False,
        ),
        "dprob": _to_cute(inputs.dprob, dynamic_layout=False),
        "fc1_recompute": _to_cute(
            inputs.fc1_recompute,
            assumed_align=128,
            dynamic_layout=False,
        ),
        "fc1_recompute_sf": _to_cute(
            inputs.fc1_recompute_sf,
            assumed_align=128,
            dynamic_layout=False,
        ),
        "fc1_col_output": _to_cute(
            inputs.fc1_col_output,
            assumed_align=128,
            dynamic_layout=False,
        ),
        "fc1_col_output_sf": _to_cute(
            inputs.fc1_col_output_sf,
            assumed_align=128,
            dynamic_layout=False,
        ),
        "grad_y2": _to_cute(
            inputs.grad_y2,
            assumed_align=128,
            dynamic_layout=False,
        ),
        "grad_y2_sf": _to_cute(
            inputs.grad_y2_sf,
            dynamic_layout=False,
        ),
        "local_workspace": _to_cute_ptr(inputs.local_workspace),
        "shared_workspace": _to_cute_ptr(inputs.shared_workspace),
        "peer_rank_ptr_mapper_host": (
            resources.workspace.peer_mapping.to_sym_buffer_host()
        ),
        "stream": cuda.CUstream(stream.cuda_stream),
    }


def compile_backward_or_get(
    prepared: PreparedMxfp8BackwardKernel,
    inputs: Mxfp8BackwardLaunchInputs,
    resources: PreparedResources,
) -> CompiledMxfp8BackwardKernel:
    signature = _layout_signature(inputs)
    key = (
        prepared.config,
        prepared.device.index,
        prepared.architecture,
        prepared.launch_cluster_count,
        prepared.dfc2_recompute,
        prepared.dfc2_col_output,
        prepared.enable_grad_y2_col_quant,
        signature,
    )
    with _COMPILE_LOCK:
        cached = _COMPILE_CACHE.get(key)
        if cached is not None:
            return cached
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MXFP8 backward kernel must be compiled before capture"
            )
        import cutlass.cute as cute

        runtime_kwargs = build_backward_runtime_kwargs(inputs, resources)
        compiled = CompiledMxfp8BackwardKernel(
            key=key,
            callable=cute.compile(prepared.kernel, **runtime_kwargs),
        )
        _COMPILE_CACHE[key] = compiled
        return compiled


__all__ = [
    "CompiledMxfp8BackwardKernel",
    "Mxfp8BackwardLaunchInputs",
    "PreparedMxfp8BackwardKernel",
    "build_backward_runtime_kwargs",
    "compile_backward_or_get",
    "prepare_backward_kernel",
]
