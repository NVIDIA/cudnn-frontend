# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""In-process JIT cache for the vendored Rubin SM107 MXFP8 kernel."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

import torch

from ..._contracts import ForwardConfig
from .._plan import PreparedResources
from .._workspace import WorkspaceRequirements
from ._adapter import Mxfp8LaunchInputs
from ._config import Mxfp8KernelConfig
from ._fingerprint import build_kernel_fingerprint
from ._launch import build_runtime_kwargs, layout_signature


@dataclass(frozen=True)
class PreparedMxfp8Kernel:
    config: Mxfp8KernelConfig
    device: torch.device
    architecture: tuple[int, int]
    kernel: Any
    launch_cluster_count: int
    workspace_requirements: WorkspaceRequirements
    pool_token_capacity: int
    col_quant_data_rows: int
    col_quant_sf_elements: int
    token_src_metadata_offset: int
    token_src_metadata_bytes: int
    col_quant_sizes_offset: int | None
    col_quant_sizes_bytes: int
    pre_reduced_activation_offset: int | None
    pre_reduced_activation_bytes_per_token: int
    pre_reduced_activation_sf_offset: int | None
    pre_reduced_activation_sf_bytes_per_token: int
    local_workspace_zero_bytes: int
    shared_workspace_zero_bytes: int


@dataclass(frozen=True)
class CompiledMxfp8Kernel:
    key: tuple
    callable: Any
    fingerprint: dict[str, Any]


_COMPILE_LOCK = threading.RLock()
_COMPILE_CACHE: dict[tuple, CompiledMxfp8Kernel] = {}
_TOKEN_SRC_METADATA_REGION = "nvlink.token_comm.token_src_metadata"
_COL_QUANT_SIZES_REGION = (
    "rubin.glu_mxfp8.mega.col_quant_expert_token_sizes"
)
_PRE_REDUCED_ACTIVATION_REGION = (
    "nvlink.token_comm.pre_reduced_activation"
)
_PRE_REDUCED_ACTIVATION_SF_REGION = (
    "nvlink.token_comm.pre_reduced_activation_sf"
)


def _compile_kernel(kernel: Any, compile_kwargs: dict[str, Any]) -> Any:
    """Import CuTeDSL only on a cache miss and compile one callable."""

    import cutlass.cute as cute

    return cute.compile(kernel, **compile_kwargs)


def _pre_reduced_workspace_metadata(
    device_workspace: Any,
    config: Mxfp8KernelConfig,
    shared_bytes: int,
) -> tuple[int | None, int]:
    """Describe the standalone combine plane, if this kernel has one."""
    if config.fc2_in_kernel_topk_reduce:
        return None, 0

    region = device_workspace.region(_PRE_REDUCED_ACTIVATION_REGION)
    if region.buffer_space != "shared":
        raise RuntimeError(
            "Rubin pre_reduced_activation must reside in shared workspace"
        )
    offset = int(
        device_workspace.offset(_PRE_REDUCED_ACTIVATION_REGION)
    )
    nbytes = int(
        device_workspace.nbytes(_PRE_REDUCED_ACTIVATION_REGION)
    )
    wire_bits_per_element = {
        "bf16": 16,
        "32e4m3xe8m0": 8,
    }
    try:
        element_bits = wire_bits_per_element[config.combine_format]
    except KeyError as exc:
        raise ValueError(
            f"unsupported combine wire format {config.combine_format!r}"
        ) from exc
    wire_bits_per_token = config.top_k * config.hidden * element_bits
    if wire_bits_per_token % 8:
        raise RuntimeError("combine wire row is not byte aligned")
    bytes_per_token = wire_bits_per_token // 8
    expected_bytes = config.max_tokens_per_rank * bytes_per_token
    if nbytes != expected_bytes:
        raise RuntimeError(
            "Rubin pre_reduced_activation size does not match "
            f"combine_format={config.combine_format!r}: {nbytes} bytes, "
            f"expected {expected_bytes}"
        )
    if offset + nbytes > shared_bytes:
        raise RuntimeError(
            "Rubin pre_reduced_activation region exceeds shared workspace"
        )
    return offset, bytes_per_token


def _pre_reduced_sf_workspace_metadata(
    device_workspace: Any,
    config: Mxfp8KernelConfig,
    shared_bytes: int,
) -> tuple[int | None, int]:
    """Describe the standalone quantized-combine scale plane, if present."""
    if config.fc2_in_kernel_topk_reduce or config.combine_format == "bf16":
        return None, 0

    region = device_workspace.region(_PRE_REDUCED_ACTIVATION_SF_REGION)
    if region.buffer_space != "shared":
        raise RuntimeError(
            "Rubin pre_reduced_activation_sf must reside in shared workspace"
        )
    offset = int(device_workspace.offset(_PRE_REDUCED_ACTIVATION_SF_REGION))
    nbytes = int(device_workspace.nbytes(_PRE_REDUCED_ACTIVATION_SF_REGION))
    if nbytes % config.max_tokens_per_rank:
        raise RuntimeError(
            "Rubin pre_reduced_activation_sf size is not token aligned"
        )
    if offset + nbytes > shared_bytes:
        raise RuntimeError(
            "Rubin pre_reduced_activation_sf region exceeds shared workspace"
        )
    return offset, nbytes // config.max_tokens_per_rank


def prepare_kernel(
    forward_config: ForwardConfig,
    config: Mxfp8KernelConfig,
    device: torch.device,
) -> PreparedMxfp8Kernel:
    """Instantiate the kernel and derive exact allocation requirements."""

    torch.cuda.set_device(device)
    architecture = torch.cuda.get_device_capability(device)
    if architecture != (10, 7):
        raise RuntimeError(
            "Rubin MXFP8 kernel preparation requires compute capability "
            f"(10, 7), got {architecture}"
        )
    configured_architecture = os.environ.get("CUTE_DSL_ARCH")
    if configured_architecture is None:
        os.environ["CUTE_DSL_ARCH"] = "sm_107a"
    elif configured_architecture not in ("sm_107", "sm_107a"):
        raise RuntimeError(
            "CUTE_DSL_ARCH must target SM107 for the Rubin MXFP8 backend, "
            f"got {configured_architecture!r}"
        )

    import cutlass
    import cutlass.utils as utils

    from ..cutedsl_src.kernel_src.rubin.training.mega.fwd_glu import (
        Sm107MegaMoEMxfp8GluKernel,
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
    kernel_kwargs = dict(
        mma_tiler_mnk=config.mma_tiler_mnk,
        cluster_shape_mnk=config.cluster_shape_mnk,
        use_2cta_instrs=config.use_2cta_instrs,
        group_hint=group_hint,
        token_padding_block=config.token_padding_block,
        sf_padding_block=config.sf_padding_block,
        load_balance_mode=config.load_balance_mode,
        static_expert_shape=(
            config.num_experts,
            config.fc1_out,
            config.hidden,
        ),
        force_static_sched=config.force_static_sched,
        clc_bundle_size=config.clc_bundle_size,
        num_sched_stages=config.num_sched_stages,
        ab_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=config.sf_vec_size,
        world_size=config.world_size,
        # Runtime rank is carried by SymmetricBufferHost. Keeping this
        # descriptor rank-independent allows every EP rank to compile the
        # same Rubin kernel.
        local_rank=0,
        num_topk=config.top_k,
        max_tokens_per_rank=config.max_tokens_per_rank,
        max_recv_size_per_rank=config.max_recv_size_per_rank,
        hidden=config.hidden,
        launch_cluster_count=launch_cluster_count,
        drop_on_overflow=config.drop_on_overflow,
        fc2_in_kernel_topk_reduce=config.fc2_in_kernel_topk_reduce,
        token_back_mode=config.token_back_mode,
        epi_flag_batch=config.epi_flag_batch,
        flag_batch=config.flag_batch,
        gate_up_clamp=config.gate_up_clamp,
        generate_c=config.generate_c,
        combine_format=CombineFormat.parse(config.combine_format),
        act_func=config.act_func,
        fc2_use_bulk=config.fc2_use_bulk,
        fc2_tma_stages=config.fc2_tma_stages,
        enable_col_quant=config.enable_col_quant,
        col_quant_num_ctas=config.col_quant_num_ctas,
    )
    kernel = Sm107MegaMoEMxfp8GluKernel.from_kwargs(**kernel_kwargs)
    local_bytes, shared_bytes = kernel.get_workspace_sizes()
    local_zero_bytes, shared_zero_bytes = (
        kernel.require_zero_workspace_leading_bytes
    )
    for name, zero_bytes, total_bytes in (
        ("local", local_zero_bytes, local_bytes),
        ("shared", shared_zero_bytes, shared_bytes),
    ):
        if zero_bytes < 0 or zero_bytes > total_bytes:
            raise RuntimeError(
                f"Rubin kernel {name} zero prefix {zero_bytes} exceeds "
                f"workspace size {total_bytes}"
            )
    device_workspace = kernel._mega_device_workspace
    metadata_region = device_workspace.region(_TOKEN_SRC_METADATA_REGION)
    if metadata_region.buffer_space != "shared":
        raise RuntimeError(
            "Rubin token_src_metadata must reside in shared workspace"
        )
    token_src_metadata_offset = int(
        device_workspace.offset(_TOKEN_SRC_METADATA_REGION)
    )
    token_src_metadata_bytes = int(
        device_workspace.nbytes(_TOKEN_SRC_METADATA_REGION)
    )
    if token_src_metadata_offset + token_src_metadata_bytes > shared_bytes:
        raise RuntimeError(
            "Rubin token_src_metadata region exceeds shared workspace"
        )
    pool_token_capacity = int(kernel.pool_token_capacity)
    if token_src_metadata_bytes != pool_token_capacity * 8:
        raise RuntimeError(
            "Rubin token_src_metadata must contain one Int64 per pool token"
        )
    col_quant_data_rows = (
        pool_token_capacity if config.enable_col_quant else 0
    )
    col_quant_sf_elements = (
        int(kernel.token_comm.worst_case_sf_token_count)
        * (config.hidden // config.sf_vec_size)
        if config.enable_col_quant
        else 0
    )
    if config.enable_col_quant:
        col_quant_sizes_region = device_workspace.region(
            _COL_QUANT_SIZES_REGION
        )
        if col_quant_sizes_region.buffer_space != "local":
            raise RuntimeError(
                "Rubin col-quant expert-size snapshot must reside in "
                "local workspace"
            )
        col_quant_sizes_offset = int(
            device_workspace.offset(_COL_QUANT_SIZES_REGION)
        )
        col_quant_sizes_bytes = int(
            device_workspace.nbytes(_COL_QUANT_SIZES_REGION)
        )
        expected_sizes_bytes = config.num_experts * torch.int32.itemsize
        if col_quant_sizes_bytes != expected_sizes_bytes:
            raise RuntimeError(
                "Rubin col-quant expert-size snapshot has "
                f"{col_quant_sizes_bytes} bytes, expected "
                f"{expected_sizes_bytes}"
            )
        if col_quant_sizes_offset + col_quant_sizes_bytes > local_bytes:
            raise RuntimeError(
                "Rubin col-quant expert-size snapshot exceeds local workspace"
            )
    else:
        col_quant_sizes_offset = None
        col_quant_sizes_bytes = 0
    requirements = WorkspaceRequirements.for_mxfp8(
        forward_config,
        kernel_local_workspace_bytes=local_bytes,
        kernel_shared_workspace_bytes=shared_bytes,
        col_quant_data_bytes=col_quant_data_rows * config.hidden,
        col_quant_sf_bytes=col_quant_sf_elements,
    )
    (
        pre_reduced_activation_offset,
        pre_reduced_activation_bytes_per_token,
    ) = _pre_reduced_workspace_metadata(
        device_workspace,
        config,
        shared_bytes,
    )
    (
        pre_reduced_activation_sf_offset,
        pre_reduced_activation_sf_bytes_per_token,
    ) = _pre_reduced_sf_workspace_metadata(
        device_workspace,
        config,
        shared_bytes,
    )
    return PreparedMxfp8Kernel(
        config=config,
        device=torch.device(device),
        architecture=architecture,
        kernel=kernel,
        launch_cluster_count=launch_cluster_count,
        workspace_requirements=requirements,
        pool_token_capacity=pool_token_capacity,
        col_quant_data_rows=col_quant_data_rows,
        col_quant_sf_elements=col_quant_sf_elements,
        token_src_metadata_offset=token_src_metadata_offset,
        token_src_metadata_bytes=token_src_metadata_bytes,
        col_quant_sizes_offset=col_quant_sizes_offset,
        col_quant_sizes_bytes=col_quant_sizes_bytes,
        pre_reduced_activation_offset=pre_reduced_activation_offset,
        pre_reduced_activation_bytes_per_token=(
            pre_reduced_activation_bytes_per_token
        ),
        pre_reduced_activation_sf_offset=pre_reduced_activation_sf_offset,
        pre_reduced_activation_sf_bytes_per_token=(
            pre_reduced_activation_sf_bytes_per_token
        ),
        local_workspace_zero_bytes=int(local_zero_bytes),
        shared_workspace_zero_bytes=int(shared_zero_bytes),
    )


def compile_or_get(
    prepared: PreparedMxfp8Kernel,
    inputs: Mxfp8LaunchInputs,
    resources: PreparedResources,
) -> CompiledMxfp8Kernel:
    signature = layout_signature(inputs)
    key = (
        *prepared.config.compile_key(
            prepared.device,
            prepared.architecture,
            prepared.launch_cluster_count,
            signature,
        ),
    )
    with _COMPILE_LOCK:
        cached = _COMPILE_CACHE.get(key)
        if cached is not None:
            return cached
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MXFP8 kernel must be compiled before CUDA graph capture")

        compile_kwargs = build_runtime_kwargs(
            inputs,
            resources,
        )
        compiled = CompiledMxfp8Kernel(
            key=key,
            callable=_compile_kernel(prepared.kernel, compile_kwargs),
            fingerprint=build_kernel_fingerprint(
                prepared,
                signature,
            ),
        )
        _COMPILE_CACHE[key] = compiled
        return compiled


def clear_compile_cache() -> None:
    """Drop process-local compiled callable references."""

    with _COMPILE_LOCK:
        _COMPILE_CACHE.clear()


__all__ = [
    "CompiledMxfp8Kernel",
    "PreparedMxfp8Kernel",
    "clear_compile_cache",
    "compile_or_get",
    "prepare_kernel",
]
