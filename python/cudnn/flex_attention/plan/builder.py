# SPDX-License-Identifier: BSD-3-Clause
"""Compile arbitrary interval masks into compact architecture-native plans."""

from __future__ import annotations

import math
import os
import re
import time
from typing import Literal

import cutlass.cute as cute
from cutlass import Int32

import torch

from cudnn.flex_attention.kernels.sm90.bwd.backward_config import (
    _ResolvedSm90BwdConsumerConfig,
    resolve_sm90_bwd_consumer_config,
)
from cudnn.flex_attention.kernels.sm90.fwd.forward_config import (
    _ResolvedSm90FwdConsumerConfig,
    resolve_sm90_fwd_consumer_config,
)
from cudnn.flex_attention.kernels.sm100.bwd.backward_config import (
    _ResolvedSm100BwdConsumerConfig,
    resolve_sm100_bwd_consumer_config,
)
from cudnn.flex_attention.kernels.sm100.bwd.backward_config_hd256 import (
    _ResolvedSm100Hd256DkdvConsumerConfig,
    _ResolvedSm100Hd256DqConsumerConfig,
    resolve_sm100_hd256_dkdv_consumer_config,
    resolve_sm100_hd256_dq_consumer_config,
)
from cudnn.flex_attention.kernels.sm100.fwd.forward_config import (
    _ResolvedSm100FwdConsumerConfig,
    resolve_sm100_fwd_consumer_config,
    resolve_sm100_fwd_qstage1_1cta_consumer_config,
    resolve_sm100_fwd_qstage1_2cta_consumer_config,
)
from cudnn.flex_attention.kernels.sm100.fwd.forward_config_hd256 import (
    _ResolvedSm100Hd256FwdConsumerConfig,
    resolve_sm100_hd256_fwd_consumer_config,
)
from cudnn.flex_attention.plan.kernels import BlockSparseTensorsTorch
from cudnn.flex_attention.plan.kernels.k2q_count import _ArbitraryPlanK2QCountSm90
from cudnn.flex_attention.plan.kernels.materialize_sm90 import (
    _ArbitraryPlanK2QMaterializeSm90,
    _ArbitraryPlanMaterializeSm90,
)
from cudnn.flex_attention.plan.kernels.materialize_sm100 import (
    _ArbitraryPlanK2QMaterializeSm100,
    _ArbitraryPlanMaterializeSm100,
)
from cudnn.flex_attention.plan.kernels.q2k_classify import _ArbitraryPlanClassifySm90
from cudnn.flex_attention.plan.kernels.scan_header import (
    FixedScanHeader,
    VarlenCompactMetadata,
    VarlenGeometry,
    VarlenScanHeader,
)
from cudnn.flex_attention.plan.kernels.schedule import ForwardScheduleOrder, ForwardSchedulePlan
from cudnn.flex_attention.plan.kernels.workspace import PlanWorkspaceInit
from cudnn.flex_attention.plan.mask_plan import (
    ArbitraryPlanRuntimeBinding,
    ArbitraryTopologyTensors,
    MaskPlan,
)
from cudnn.flex_attention.plan.topology import (
    _ResolvedSm90BwdTopologyConfig,
    _ResolvedSm100BwdTopologyConfig,
    _ResolvedSm100FwdTopologyConfig,
    _ResolvedSm100Hd256DkdvTopologyConfig,
    _ResolvedSm100Hd256DqTopologyConfig,
    _ResolvedSm100Hd256FwdTopologyConfig,
    _consumer_plan_signature,
)
from cudnn.flex_attention.plan.validation import validate_create_mask_plan_inputs
from cudnn.flex_attention.runtime.compile_cache import get_jit_cache
from cudnn.flex_attention.runtime.dsl_utils import to_cute_tensor
from cudnn.flex_attention.runtime.fake_tensor import is_fake_mode

_QSTAGE1_OVERLAP_MIN_AVERAGE_BLOCKS = 7
_SM100_GENERIC_DEFAULT_FWD_VARIANT = "qstage1_2cta"
_Sm100FwdVariant = Literal[
    "qstage2_1cta",
    "qstage1_1cta",
    "qstage1_2cta",
]

_CLASSIFY_COMPILE_CACHE = get_jit_cache("arbitrary_plan_classify")
_MATERIALIZE_COMPILE_CACHE = get_jit_cache("arbitrary_plan_materialize")
_K2Q_COUNT_COMPILE_CACHE = get_jit_cache("arbitrary_plan_k2q_count")
_K2Q_MATERIALIZE_COMPILE_CACHE = get_jit_cache("arbitrary_plan_k2q_materialize")
_FWD_SCHEDULE_COMPILE_CACHE = get_jit_cache("arbitrary_plan_fwd_schedule")
_FWD_SCHEDULE_ORDER_COMPILE_CACHE = get_jit_cache("arbitrary_plan_fwd_schedule_order")
_FIXED_SCAN_HEADER_COMPILE_CACHE = get_jit_cache("arbitrary_plan_fixed_scan_header")
_VARLEN_SCAN_HEADER_COMPILE_CACHE = get_jit_cache("arbitrary_plan_varlen_scan_header")
_VARLEN_COMPACT_METADATA_COMPILE_CACHE = get_jit_cache("arbitrary_plan_varlen_compact_metadata")
_VARLEN_GEOMETRY_COMPILE_CACHE = get_jit_cache("arbitrary_plan_varlen_geometry")
_PLAN_WORKSPACE_INIT_COMPILE_CACHE = get_jit_cache("arbitrary_plan_workspace_init")


def _get_plan_builder_arch(device: torch.device) -> int:
    """Resolve fake compilation targets without silently producing an SM90 plan."""

    if not is_fake_mode():
        major, minor = torch.cuda.get_device_capability(device)
        return major * 10 + minor
    arch_override = os.environ.get("FLEX_ATTENTION_ARCH") or os.environ.get("CUTE_DSL_ARCH")
    if arch_override is None:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(device)
            return major * 10 + minor
        # Preserve the existing no-GPU SM90 planner test default.  CPU-only
        # compilation for another architecture must provide an override.
        return 90
    match = re.fullmatch(r"(?:sm_?)?(\d+)(\d)[af]?", arch_override, re.IGNORECASE)
    if match is None:
        raise ValueError(f"invalid fake arbitrary-plan architecture: {arch_override!r}")
    return int(match.group(1)) * 10 + int(match.group(2))


def _exclusive_offsets(counts: torch.Tensor) -> torch.Tensor:
    offsets = torch.empty((counts.numel() + 1,), dtype=torch.int32, device=counts.device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts.reshape(-1), dim=0, dtype=torch.int32)
    return offsets


def _exclusive_offsets_from_scan(
    counts: torch.Tensor,
    inclusive_scan: torch.Tensor | None,
) -> torch.Tensor:
    """Reuse a full-row inclusive scan when exact compact rows are known."""

    if inclusive_scan is None:
        return _exclusive_offsets(counts)
    offsets = torch.empty((counts.numel() + 1,), dtype=torch.int32, device=counts.device)
    offsets[0] = 0
    if counts.numel() > 0:
        hmask, compact_rows = counts.shape
        upper_rows = inclusive_scan.numel() // hmask
        offsets[1:].view(hmask, compact_rows).copy_(inclusive_scan.view(hmask, upper_rows)[:, :compact_rows])
    return offsets


def _compact_counts(
    counts: torch.Tensor,
    total_rows: int,
    *,
    is_varlen: bool,
) -> torch.Tensor:
    """Retain exact fixed counts and compact only data-dependent Varlen rows."""

    if not is_varlen:
        assert total_rows == counts.shape[1]
        return counts
    return counts[:, :total_rows].clone()


def _to_cute_optional(
    tensor: torch.Tensor | None,
    *,
    leading_dim: int = 0,
    assumed_align: int = 4,
):
    return to_cute_tensor(tensor, assumed_align=assumed_align, leading_dim=leading_dim) if tensor is not None else None


def _classify_compile_key(
    config: (
        _ResolvedSm90FwdConsumerConfig
        | _ResolvedSm90BwdTopologyConfig
        | _ResolvedSm100BwdTopologyConfig
        | _ResolvedSm100FwdTopologyConfig
        | _ResolvedSm100Hd256FwdTopologyConfig
        | _ResolvedSm100Hd256DqTopologyConfig
        | _ResolvedSm100Hd256DkdvTopologyConfig
    ),
) -> tuple:
    """Keep family-compatible signatures separate from exact-arch CUBINs."""

    return (
        "arbitrary_plan_classify_v7",
        config.arch,
        config.topology_planner_compile_key,
    )


def _materialize_compile_key(
    config: _ResolvedSm90FwdConsumerConfig | _ResolvedSm100FwdConsumerConfig | _ResolvedSm100Hd256FwdConsumerConfig | _ResolvedSm100Hd256DqConsumerConfig,
) -> tuple:
    """Return the exact-target payload materializer compilation key."""

    return (
        (
            "arbitrary_plan_materialize_hd256_dq_v3"
            if isinstance(config, _ResolvedSm100Hd256DqConsumerConfig)
            else (
                "arbitrary_plan_materialize_hd256_fwd_v3"
                if isinstance(config, _ResolvedSm100Hd256FwdConsumerConfig)
                else ("arbitrary_plan_sm90_materialize_v6" if isinstance(config, _ResolvedSm90FwdConsumerConfig) else "arbitrary_plan_materialize_v4")
            )
        ),
        config.arch,
        _consumer_plan_signature(config).arch_family,
        config.payload_planner_compile_key,
    )


def _compile_classify(
    config: (
        _ResolvedSm90FwdConsumerConfig
        | _ResolvedSm90BwdTopologyConfig
        | _ResolvedSm100BwdTopologyConfig
        | _ResolvedSm100FwdTopologyConfig
        | _ResolvedSm100Hd256FwdTopologyConfig
        | _ResolvedSm100Hd256DqTopologyConfig
        | _ResolvedSm100Hd256DkdvTopologyConfig
    ),
    arbitrary_func: torch.Tensor,
    visible_bits: torch.Tensor,
    full_bits: torch.Tensor,
    partial_counts: torch.Tensor,
    full_counts: torch.Tensor,
    interval_invalid: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    cu_total_m_blocks: torch.Tensor | None,
):
    key = _classify_compile_key(config)
    if key not in _CLASSIFY_COMPILE_CACHE:
        kernel = _ArbitraryPlanClassifySm90(config)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        started_at = time.perf_counter()
        compiled = cute.compile(
            kernel,
            to_cute_tensor(arbitrary_func, assumed_align=4, leading_dim=2),
            to_cute_tensor(visible_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(full_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(partial_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(full_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(interval_invalid, assumed_align=4, leading_dim=0),
            _to_cute_optional(cu_seqlens_q),
            _to_cute_optional(cu_seqlens_k),
            _to_cute_optional(cu_total_m_blocks),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
        print(f"Compiled mask-plan classify in {time.perf_counter() - started_at:.1f}s")
        _CLASSIFY_COMPILE_CACHE[key] = compiled
    return _CLASSIFY_COMPILE_CACHE[key]


def _compile_plan_workspace_init(
    arch: int,
    visible_bits: torch.Tensor,
    full_bits: torch.Tensor,
    partial_counts: torch.Tensor,
    full_counts: torch.Tensor,
    interval_invalid: torch.Tensor,
    schedule_histogram: torch.Tensor,
    schedule_section_cost: torch.Tensor,
    bwd_visible_bits: torch.Tensor | None,
    bwd_full_bits: torch.Tensor | None,
    bwd_q_partial_counts: torch.Tensor | None,
    bwd_q_full_counts: torch.Tensor | None,
    bwd_partial_counts: torch.Tensor | None,
    bwd_full_counts: torch.Tensor | None,
):
    """Compile the architecture-neutral planner workspace initializer."""

    build_backward = bwd_visible_bits is not None
    key = ("arbitrary_plan_workspace_init_v1", arch, build_backward)
    if key not in _PLAN_WORKSPACE_INIT_COMPILE_CACHE:
        kernel = PlanWorkspaceInit(build_backward=build_backward)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        started_at = time.perf_counter()
        compiled = cute.compile(
            kernel,
            to_cute_tensor(visible_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(full_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(partial_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(full_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(interval_invalid, assumed_align=4, leading_dim=0),
            to_cute_tensor(schedule_histogram, assumed_align=4, leading_dim=1),
            to_cute_tensor(schedule_section_cost, assumed_align=8, leading_dim=0),
            _to_cute_optional(bwd_visible_bits, leading_dim=2, assumed_align=16),
            _to_cute_optional(bwd_full_bits, leading_dim=2, assumed_align=16),
            _to_cute_optional(bwd_q_partial_counts, leading_dim=1),
            _to_cute_optional(bwd_q_full_counts, leading_dim=1),
            _to_cute_optional(bwd_partial_counts, leading_dim=1),
            _to_cute_optional(bwd_full_counts, leading_dim=1),
            stream,
            options="--enable-tvm-ffi",
        )
        print(f"Compiled mask-plan workspace init in {time.perf_counter() - started_at:.1f}s")
        _PLAN_WORKSPACE_INIT_COMPILE_CACHE[key] = compiled
    return _PLAN_WORKSPACE_INIT_COMPILE_CACHE[key]


def _compile_materialize(
    config: _ResolvedSm90FwdConsumerConfig | _ResolvedSm100FwdConsumerConfig | _ResolvedSm100Hd256FwdConsumerConfig | _ResolvedSm100Hd256DqConsumerConfig,
    arbitrary_func: torch.Tensor,
    visible_bits: torch.Tensor,
    full_bits: torch.Tensor,
    partial_offsets: torch.Tensor,
    partial_indices: torch.Tensor,
    partial_masks: torch.Tensor,
    partial_work_desc: torch.Tensor | None,
    full_offsets: torch.Tensor,
    full_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    cu_total_m_blocks: torch.Tensor | None,
):
    key = _materialize_compile_key(config)
    if key not in _MATERIALIZE_COMPILE_CACHE:
        kernel = (
            _ArbitraryPlanMaterializeSm100(config)
            if isinstance(
                config,
                (
                    _ResolvedSm100FwdConsumerConfig,
                    _ResolvedSm100Hd256FwdConsumerConfig,
                    _ResolvedSm100Hd256DqConsumerConfig,
                ),
            )
            else _ArbitraryPlanMaterializeSm90(config)
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        started_at = time.perf_counter()
        compile_args = (
            to_cute_tensor(arbitrary_func, assumed_align=4, leading_dim=2),
            to_cute_tensor(visible_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(full_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(partial_offsets, assumed_align=4, leading_dim=0),
            to_cute_tensor(partial_indices, assumed_align=4, leading_dim=0),
            to_cute_tensor(partial_masks, assumed_align=16, leading_dim=3),
        )
        if partial_work_desc is not None:
            compile_args += (to_cute_tensor(partial_work_desc, assumed_align=16, leading_dim=1),)
        compile_args += (
            to_cute_tensor(full_offsets, assumed_align=4, leading_dim=0),
            to_cute_tensor(full_indices, assumed_align=4, leading_dim=0),
            _to_cute_optional(cu_seqlens_q),
            _to_cute_optional(cu_seqlens_k),
            _to_cute_optional(cu_total_m_blocks),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream,
        )
        compiled = cute.compile(
            kernel,
            *compile_args,
            options="--enable-tvm-ffi",
        )
        print(f"Compiled mask-plan materialize in {time.perf_counter() - started_at:.1f}s")
        _MATERIALIZE_COMPILE_CACHE[key] = compiled
    return _MATERIALIZE_COMPILE_CACHE[key]


def _compile_fixed_scan_header(
    arch: int,
    partial_counts: torch.Tensor,
    full_counts: torch.Tensor,
    partial_offsets: torch.Tensor,
    full_offsets: torch.Tensor,
    bwd_partial_counts: torch.Tensor | None,
    bwd_full_counts: torch.Tensor | None,
    bwd_partial_offsets: torch.Tensor | None,
    bwd_full_offsets: torch.Tensor | None,
    dq_partial_counts: torch.Tensor | None,
    dq_full_counts: torch.Tensor | None,
    dq_partial_offsets: torch.Tensor | None,
    dq_full_offsets: torch.Tensor | None,
    interval_invalid: torch.Tensor,
    header: torch.Tensor,
):
    """Compile the architecture-neutral fixed scan and allocation header."""

    build_backward = bwd_partial_counts is not None
    build_dq = dq_partial_counts is not None
    key = ("arbitrary_plan_fixed_scan_header_v3", arch, build_backward, build_dq)
    if key not in _FIXED_SCAN_HEADER_COMPILE_CACHE:
        kernel = FixedScanHeader(build_backward=build_backward, build_dq=build_dq)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        started_at = time.perf_counter()
        compiled = cute.compile(
            kernel,
            to_cute_tensor(partial_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(full_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(partial_offsets, assumed_align=4, leading_dim=0),
            to_cute_tensor(full_offsets, assumed_align=4, leading_dim=0),
            (to_cute_tensor(bwd_partial_counts, assumed_align=4, leading_dim=1) if bwd_partial_counts is not None else None),
            (to_cute_tensor(bwd_full_counts, assumed_align=4, leading_dim=1) if bwd_full_counts is not None else None),
            _to_cute_optional(bwd_partial_offsets),
            _to_cute_optional(bwd_full_offsets),
            _to_cute_optional(dq_partial_counts, leading_dim=1),
            _to_cute_optional(dq_full_counts, leading_dim=1),
            _to_cute_optional(dq_partial_offsets),
            _to_cute_optional(dq_full_offsets),
            to_cute_tensor(interval_invalid, assumed_align=4, leading_dim=0),
            to_cute_tensor(header, assumed_align=8, leading_dim=0),
            Int32(1),
            Int32(1),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
        print(f"Compiled mask-plan fixed scan/header in {time.perf_counter() - started_at:.1f}s")
        _FIXED_SCAN_HEADER_COMPILE_CACHE[key] = compiled
    return _FIXED_SCAN_HEADER_COMPILE_CACHE[key]


def _compile_varlen_scan_header(
    arch: int,
    partial_counts: torch.Tensor,
    full_counts: torch.Tensor,
    partial_scan: torch.Tensor,
    full_scan: torch.Tensor,
    bwd_partial_counts: torch.Tensor | None,
    bwd_full_counts: torch.Tensor | None,
    bwd_partial_scan: torch.Tensor | None,
    bwd_full_scan: torch.Tensor | None,
    dq_partial_counts: torch.Tensor | None,
    dq_full_counts: torch.Tensor | None,
    dq_partial_scan: torch.Tensor | None,
    dq_full_scan: torch.Tensor | None,
    cu_total_m_blocks: torch.Tensor,
    cu_total_bwd_m_blocks: torch.Tensor | None,
    cu_total_bwd_n_blocks: torch.Tensor | None,
    metadata_invalid: torch.Tensor,
    interval_invalid: torch.Tensor,
    header: torch.Tensor,
):
    """Compile the architecture-neutral Varlen scan and allocation header."""

    build_backward = bwd_partial_counts is not None
    build_dq = dq_partial_counts is not None
    key = ("arbitrary_plan_varlen_scan_header_v2", arch, build_backward, build_dq)
    if key not in _VARLEN_SCAN_HEADER_COMPILE_CACHE:
        kernel = VarlenScanHeader(build_backward=build_backward, build_dq=build_dq)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        started_at = time.perf_counter()
        compiled = cute.compile(
            kernel,
            to_cute_tensor(partial_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(full_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(partial_scan, assumed_align=8, leading_dim=0),
            to_cute_tensor(full_scan, assumed_align=8, leading_dim=0),
            _to_cute_optional(bwd_partial_counts, leading_dim=1),
            _to_cute_optional(bwd_full_counts, leading_dim=1),
            _to_cute_optional(bwd_partial_scan),
            _to_cute_optional(bwd_full_scan),
            _to_cute_optional(dq_partial_counts, leading_dim=1),
            _to_cute_optional(dq_full_counts, leading_dim=1),
            _to_cute_optional(dq_partial_scan),
            _to_cute_optional(dq_full_scan),
            to_cute_tensor(cu_total_m_blocks, assumed_align=4, leading_dim=0),
            _to_cute_optional(cu_total_bwd_m_blocks),
            _to_cute_optional(cu_total_bwd_n_blocks),
            to_cute_tensor(metadata_invalid, assumed_align=1, leading_dim=0),
            to_cute_tensor(interval_invalid, assumed_align=4, leading_dim=0),
            to_cute_tensor(header, assumed_align=8, leading_dim=0),
            stream,
            options="--enable-tvm-ffi",
        )
        print(f"Compiled mask-plan Varlen scan/header in {time.perf_counter() - started_at:.1f}s")
        _VARLEN_SCAN_HEADER_COMPILE_CACHE[key] = compiled
    return _VARLEN_SCAN_HEADER_COMPILE_CACHE[key]


def _compile_varlen_geometry(
    arch: int,
    *,
    fwd_tile_m: int,
    fwd_tile_n: int,
    qhead_per_kvhead: int,
    bwd_tile_m: int,
    bwd_tile_n: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_total_m_blocks: torch.Tensor,
    cu_total_fwd_n_blocks: torch.Tensor,
    cu_total_bwd_m_blocks: torch.Tensor | None,
    cu_total_bwd_n_blocks: torch.Tensor | None,
    metadata_invalid: torch.Tensor,
):
    """Compile architecture-neutral Varlen block-prefix materialization."""

    build_backward = cu_total_bwd_m_blocks is not None
    key = (
        "arbitrary_plan_varlen_geometry_v1",
        arch,
        fwd_tile_m,
        fwd_tile_n,
        qhead_per_kvhead,
        build_backward,
        bwd_tile_m,
        bwd_tile_n,
    )
    if key not in _VARLEN_GEOMETRY_COMPILE_CACHE:
        kernel = VarlenGeometry(
            fwd_tile_m=fwd_tile_m,
            fwd_tile_n=fwd_tile_n,
            qhead_per_kvhead=qhead_per_kvhead,
            build_backward=build_backward,
            bwd_tile_m=bwd_tile_m,
            bwd_tile_n=bwd_tile_n,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        started_at = time.perf_counter()
        compiled = cute.compile(
            kernel,
            to_cute_tensor(cu_seqlens_q, assumed_align=4, leading_dim=0),
            to_cute_tensor(cu_seqlens_k, assumed_align=4, leading_dim=0),
            to_cute_tensor(cu_total_m_blocks, assumed_align=4, leading_dim=0),
            to_cute_tensor(cu_total_fwd_n_blocks, assumed_align=4, leading_dim=0),
            _to_cute_optional(cu_total_bwd_m_blocks),
            _to_cute_optional(cu_total_bwd_n_blocks),
            to_cute_tensor(metadata_invalid, assumed_align=1, leading_dim=0),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
        print(f"Compiled mask-plan Varlen geometry in {time.perf_counter() - started_at:.1f}s")
        _VARLEN_GEOMETRY_COMPILE_CACHE[key] = compiled
    return _VARLEN_GEOMETRY_COMPILE_CACHE[key]


def _compile_varlen_compact_metadata(
    arch: int,
    partial_counts_tmp: torch.Tensor,
    full_counts_tmp: torch.Tensor,
    partial_scan: torch.Tensor,
    full_scan: torch.Tensor,
    partial_counts: torch.Tensor,
    full_counts: torch.Tensor,
    partial_offsets: torch.Tensor,
    full_offsets: torch.Tensor,
    bwd_partial_counts_tmp: torch.Tensor | None,
    bwd_full_counts_tmp: torch.Tensor | None,
    bwd_partial_scan: torch.Tensor | None,
    bwd_full_scan: torch.Tensor | None,
    bwd_partial_counts: torch.Tensor | None,
    bwd_full_counts: torch.Tensor | None,
    bwd_partial_offsets: torch.Tensor | None,
    bwd_full_offsets: torch.Tensor | None,
    dq_partial_counts_tmp: torch.Tensor | None,
    dq_full_counts_tmp: torch.Tensor | None,
    dq_partial_scan: torch.Tensor | None,
    dq_full_scan: torch.Tensor | None,
    dq_partial_counts: torch.Tensor | None,
    dq_full_counts: torch.Tensor | None,
    dq_partial_offsets: torch.Tensor | None,
    dq_full_offsets: torch.Tensor | None,
):
    """Compile exact Varlen count/offset materialization."""

    build_backward = bwd_partial_counts_tmp is not None
    build_dq = dq_partial_counts_tmp is not None
    key = ("arbitrary_plan_varlen_compact_metadata_v1", arch, build_backward, build_dq)
    if key not in _VARLEN_COMPACT_METADATA_COMPILE_CACHE:
        kernel = VarlenCompactMetadata(build_backward=build_backward, build_dq=build_dq)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        started_at = time.perf_counter()
        compiled = cute.compile(
            kernel,
            to_cute_tensor(partial_counts_tmp, assumed_align=4, leading_dim=1),
            to_cute_tensor(full_counts_tmp, assumed_align=4, leading_dim=1),
            to_cute_tensor(partial_scan, assumed_align=8, leading_dim=0),
            to_cute_tensor(full_scan, assumed_align=8, leading_dim=0),
            to_cute_tensor(partial_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(full_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(partial_offsets, assumed_align=4, leading_dim=0),
            to_cute_tensor(full_offsets, assumed_align=4, leading_dim=0),
            _to_cute_optional(bwd_partial_counts_tmp, leading_dim=1),
            _to_cute_optional(bwd_full_counts_tmp, leading_dim=1),
            _to_cute_optional(bwd_partial_scan),
            _to_cute_optional(bwd_full_scan),
            _to_cute_optional(bwd_partial_counts, leading_dim=1),
            _to_cute_optional(bwd_full_counts, leading_dim=1),
            _to_cute_optional(bwd_partial_offsets),
            _to_cute_optional(bwd_full_offsets),
            _to_cute_optional(dq_partial_counts_tmp, leading_dim=1),
            _to_cute_optional(dq_full_counts_tmp, leading_dim=1),
            _to_cute_optional(dq_partial_scan),
            _to_cute_optional(dq_full_scan),
            _to_cute_optional(dq_partial_counts, leading_dim=1),
            _to_cute_optional(dq_full_counts, leading_dim=1),
            _to_cute_optional(dq_partial_offsets),
            _to_cute_optional(dq_full_offsets),
            stream,
            options="--enable-tvm-ffi",
        )
        print(f"Compiled mask-plan Varlen compact metadata in {time.perf_counter() - started_at:.1f}s")
        _VARLEN_COMPACT_METADATA_COMPILE_CACHE[key] = compiled
    return _VARLEN_COMPACT_METADATA_COMPILE_CACHE[key]


def _compile_forward_schedule(
    config: _ResolvedSm90FwdConsumerConfig | _ResolvedSm100FwdConsumerConfig | _ResolvedSm100Hd256FwdConsumerConfig,
    partial_counts: torch.Tensor,
    full_counts: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    cu_total_m_blocks: torch.Tensor | None,
    sequence_desc: torch.Tensor | None,
    work_desc: torch.Tensor,
    task_cost: torch.Tensor,
    section_id: torch.Tensor,
):
    key = (
        "arbitrary_plan_fwd_schedule_v3",
        config.arch,
        config.block_size,
        config.tile_n,
        config.qhead_per_kvhead,
        config.pack_gqa,
        config.is_varlen,
        # Fixed HD256 uses sequence descriptors while fixed generic kernels do
        # not.  This changes the TVM FFI signature even when their topology
        # fields otherwise match.
        sequence_desc is not None,
    )
    if key not in _FWD_SCHEDULE_COMPILE_CACHE:
        planner = ForwardSchedulePlan(
            plan_tile_m=config.block_size[0],
            tile_n=config.tile_n,
            qhead_per_kvhead=config.qhead_per_kvhead,
            pack_gqa=config.pack_gqa,
            is_varlen=config.is_varlen,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        started_at = time.perf_counter()
        compiled = cute.compile(
            planner,
            to_cute_tensor(partial_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(full_counts, assumed_align=4, leading_dim=1),
            _to_cute_optional(cu_seqlens_q),
            _to_cute_optional(cu_seqlens_k),
            _to_cute_optional(cu_total_m_blocks),
            (to_cute_tensor(sequence_desc, assumed_align=16, leading_dim=1) if sequence_desc is not None else None),
            to_cute_tensor(work_desc, assumed_align=16, leading_dim=1),
            to_cute_tensor(task_cost, assumed_align=4, leading_dim=0),
            to_cute_tensor(section_id, assumed_align=4, leading_dim=0),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
        print(f"Compiled mask-plan forward schedule in {time.perf_counter() - started_at:.1f}s")
        _FWD_SCHEDULE_COMPILE_CACHE[key] = compiled
    return _FWD_SCHEDULE_COMPILE_CACHE[key]


def _compile_forward_schedule_order(
    config: _ResolvedSm90FwdConsumerConfig | _ResolvedSm100FwdConsumerConfig | _ResolvedSm100Hd256FwdConsumerConfig,
    work_desc: torch.Tensor,
    task_cost: torch.Tensor,
    section_id: torch.Tensor,
    histogram: torch.Tensor,
    section_cost: torch.Tensor,
    section_order: torch.Tensor,
    positive_base: torch.Tensor,
    zero_base: torch.Tensor,
    sorted_work_desc: torch.Tensor,
    cu_total_m_blocks: torch.Tensor | None,
):
    key = (
        "arbitrary_plan_fwd_schedule_order_v2",
        config.arch,
        config.qhead_per_kvhead,
        config.pack_gqa,
        config.is_varlen,
    )
    if key not in _FWD_SCHEDULE_ORDER_COMPILE_CACHE:
        planner = ForwardScheduleOrder(
            qhead_per_kvhead=config.qhead_per_kvhead,
            pack_gqa=config.pack_gqa,
            is_varlen=config.is_varlen,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        started_at = time.perf_counter()
        compiled = cute.compile(
            planner,
            to_cute_tensor(work_desc, assumed_align=16, leading_dim=1),
            to_cute_tensor(task_cost, assumed_align=4, leading_dim=0),
            to_cute_tensor(section_id, assumed_align=4, leading_dim=0),
            to_cute_tensor(histogram, assumed_align=4, leading_dim=1),
            to_cute_tensor(section_cost, assumed_align=8, leading_dim=0),
            to_cute_tensor(section_order, assumed_align=4, leading_dim=0),
            to_cute_tensor(positive_base, assumed_align=4, leading_dim=0),
            to_cute_tensor(zero_base, assumed_align=4, leading_dim=0),
            to_cute_tensor(sorted_work_desc, assumed_align=16, leading_dim=1),
            _to_cute_optional(cu_total_m_blocks),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
        print(f"Compiled mask-plan forward schedule order in {time.perf_counter() - started_at:.1f}s")
        _FWD_SCHEDULE_ORDER_COMPILE_CACHE[key] = compiled
    return _FWD_SCHEDULE_ORDER_COMPILE_CACHE[key]


def _build_forward_schedule(
    config: _ResolvedSm90FwdConsumerConfig | _ResolvedSm100FwdConsumerConfig | _ResolvedSm100Hd256FwdConsumerConfig,
    partial_counts: torch.Tensor,
    full_counts: torch.Tensor,
    *,
    batch_size: int,
    total_m_blocks: int,
    max_m_blocks: int,
    max_task_cost: int,
    seqlen_q_fixed: int,
    seqlen_k_fixed: int,
    head_dim: int,
    head_dim_v: int,
    element_size: int,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    cu_total_m_blocks: torch.Tensor | None,
    schedule_histogram: torch.Tensor,
    schedule_section_cost: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Build the immutable architecture-neutral FWD work queue owned by a plan."""

    num_scheduled_heads = config.num_kv_heads if config.pack_gqa else config.num_q_heads
    num_forward_tasks = total_m_blocks * num_scheduled_heads
    cta_group_size = getattr(config, "cta_group_size", 1)
    num_backend_work_items = num_forward_tasks * cta_group_size
    if num_backend_work_items > torch.iinfo(torch.int32).max:
        raise ValueError("forward schedule exceeds the int32 work-item range")

    device = partial_counts.device
    needs_sequence_desc = config.is_varlen or isinstance(config, _ResolvedSm100Hd256FwdConsumerConfig)
    sequence_desc = torch.empty((batch_size, 8), dtype=torch.int32, device=device) if needs_sequence_desc else None
    work_desc = torch.empty(
        (num_forward_tasks, 4),
        dtype=torch.int32,
        device=device,
    )
    if is_fake_mode() or num_forward_tasks == 0:
        if needs_sequence_desc and not is_fake_mode():
            assert sequence_desc is not None
            if config.is_varlen:
                assert cu_seqlens_q is not None
                assert cu_seqlens_k is not None
                assert cu_total_m_blocks is not None
                q_offset = cu_seqlens_q[:-1]
                k_offset = cu_seqlens_k[:-1]
                q_len = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                k_len = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
                q_plan_row_begin = cu_total_m_blocks[:-1]
            else:
                q_len = torch.full((batch_size,), seqlen_q_fixed, dtype=torch.int32, device=device)
                k_len = torch.full((batch_size,), seqlen_k_fixed, dtype=torch.int32, device=device)
                batch_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
                q_offset = batch_idx * seqlen_q_fixed
                k_offset = batch_idx * seqlen_k_fixed
                q_plan_row_begin = batch_idx * max_m_blocks
            physical_q_len = q_len * config.qhead_per_kvhead if config.pack_gqa else q_len
            q_plan_count = torch.div(
                physical_q_len + config.block_size[0] - 1,
                config.block_size[0],
                rounding_mode="floor",
            )
            num_k_blocks = torch.div(
                k_len + config.tile_n - 1,
                config.tile_n,
                rounding_mode="floor",
            )
            sequence_desc.copy_(
                torch.stack(
                    (
                        q_offset,
                        k_offset,
                        q_len,
                        k_len,
                        q_plan_row_begin,
                        q_plan_count,
                        num_k_blocks,
                        torch.zeros_like(q_len),
                    ),
                    dim=1,
                ).to(torch.int32)
            )
        return sequence_desc, work_desc

    task_cost = torch.empty(
        (num_forward_tasks,),
        dtype=torch.int32,
        device=device,
    )
    section_id = torch.empty_like(task_cost)
    schedule = _compile_forward_schedule(
        config,
        partial_counts,
        full_counts,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_total_m_blocks,
        sequence_desc,
        work_desc,
        task_cost,
        section_id,
    )
    schedule(
        partial_counts,
        full_counts,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_total_m_blocks,
        sequence_desc,
        work_desc,
        task_cost,
        section_id,
        Int32(batch_size),
        Int32(num_scheduled_heads),
        Int32(config.num_kv_heads),
        Int32(seqlen_q_fixed),
        Int32(seqlen_k_fixed),
        Int32(max_m_blocks),
        Int32(head_dim),
        Int32(head_dim_v),
        Int32(element_size),
    )
    num_sections = batch_size * config.num_kv_heads
    section_order = torch.empty((num_sections,), dtype=torch.int32, device=device)
    positive_base = torch.empty_like(section_order)
    zero_base = torch.empty_like(section_order)
    sorted_work_desc = torch.empty_like(work_desc)
    order_schedule = _compile_forward_schedule_order(
        config,
        work_desc,
        task_cost,
        section_id,
        schedule_histogram,
        schedule_section_cost,
        section_order,
        positive_base,
        zero_base,
        sorted_work_desc,
        cu_total_m_blocks,
    )
    order_schedule(
        work_desc,
        task_cost,
        section_id,
        schedule_histogram,
        schedule_section_cost,
        section_order,
        positive_base,
        zero_base,
        sorted_work_desc,
        cu_total_m_blocks,
        Int32(num_forward_tasks),
        Int32(batch_size),
        Int32(num_scheduled_heads),
        Int32(config.num_kv_heads),
        Int32(max_m_blocks),
        Int32(max_task_cost),
    )
    return sequence_desc, sorted_work_desc


def _compile_k2q_count(
    config: _ResolvedSm90BwdConsumerConfig | _ResolvedSm100BwdConsumerConfig | _ResolvedSm100Hd256DkdvConsumerConfig,
    visible_bits: torch.Tensor,
    full_bits: torch.Tensor,
    partial_counts: torch.Tensor,
    full_counts: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    cu_total_q_blocks: torch.Tensor | None,
    cu_total_k_blocks: torch.Tensor | None,
):
    if isinstance(config, _ResolvedSm100BwdConsumerConfig):
        topology_config = _ResolvedSm100BwdTopologyConfig(config)
    elif isinstance(config, _ResolvedSm100Hd256DkdvConsumerConfig):
        topology_config = _ResolvedSm100Hd256DkdvTopologyConfig(config)
    else:
        topology_config = _ResolvedSm90BwdTopologyConfig(config)
    key = (
        ("arbitrary_plan_hd256_k2q_count_v1" if isinstance(config, _ResolvedSm100Hd256DkdvConsumerConfig) else "arbitrary_plan_k2q_count_v3"),
        config.arch,
        topology_config.topology_planner_compile_key,
    )
    if key not in _K2Q_COUNT_COMPILE_CACHE:
        kernel = _ArbitraryPlanK2QCountSm90(config)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        started_at = time.perf_counter()
        compiled = cute.compile(
            kernel,
            to_cute_tensor(visible_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(full_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(partial_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(full_counts, assumed_align=4, leading_dim=1),
            _to_cute_optional(cu_seqlens_q),
            _to_cute_optional(cu_seqlens_k),
            _to_cute_optional(cu_total_q_blocks),
            _to_cute_optional(cu_total_k_blocks),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
        print(f"Compiled mask-plan K2Q count in {time.perf_counter() - started_at:.1f}s")
        _K2Q_COUNT_COMPILE_CACHE[key] = compiled
    return _K2Q_COUNT_COMPILE_CACHE[key]


def _compile_k2q_materialize(
    config: _ResolvedSm90BwdConsumerConfig | _ResolvedSm100BwdConsumerConfig | _ResolvedSm100Hd256DkdvConsumerConfig,
    arbitrary_func: torch.Tensor,
    visible_bits: torch.Tensor,
    full_bits: torch.Tensor,
    q_partial_counts: torch.Tensor,
    q_full_counts: torch.Tensor,
    partial_offsets: torch.Tensor,
    partial_indices: torch.Tensor,
    partial_masks: torch.Tensor,
    partial_work_desc: torch.Tensor | None,
    partial_dq_order: torch.Tensor,
    full_offsets: torch.Tensor,
    full_indices: torch.Tensor,
    full_dq_order: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    cu_total_q_blocks: torch.Tensor | None,
    cu_total_k_blocks: torch.Tensor | None,
):
    key = (
        (
            "arbitrary_plan_hd256_k2q_materialize_v3"
            if isinstance(config, _ResolvedSm100Hd256DkdvConsumerConfig)
            else ("arbitrary_plan_sm90_k2q_materialize_v7" if isinstance(config, _ResolvedSm90BwdConsumerConfig) else "arbitrary_plan_k2q_materialize_v5")
        ),
        config.arch,
        _consumer_plan_signature(config).arch_family,
        config.planner_compile_key,
    )
    if key not in _K2Q_MATERIALIZE_COMPILE_CACHE:
        kernel = (
            _ArbitraryPlanK2QMaterializeSm100(config)
            if isinstance(
                config,
                (_ResolvedSm100BwdConsumerConfig, _ResolvedSm100Hd256DkdvConsumerConfig),
            )
            else _ArbitraryPlanK2QMaterializeSm90(config)
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        started_at = time.perf_counter()
        compile_args = (
            to_cute_tensor(arbitrary_func, assumed_align=4, leading_dim=2),
            to_cute_tensor(visible_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(full_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(q_partial_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(q_full_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(partial_offsets, assumed_align=4, leading_dim=0),
            to_cute_tensor(partial_indices, assumed_align=4, leading_dim=0),
            to_cute_tensor(partial_masks, assumed_align=16, leading_dim=3),
        )
        if partial_work_desc is not None:
            compile_args += (to_cute_tensor(partial_work_desc, assumed_align=16, leading_dim=1),)
        compile_args += (
            to_cute_tensor(partial_dq_order, assumed_align=4, leading_dim=0),
            to_cute_tensor(full_offsets, assumed_align=4, leading_dim=0),
            to_cute_tensor(full_indices, assumed_align=4, leading_dim=0),
            to_cute_tensor(full_dq_order, assumed_align=4, leading_dim=0),
            _to_cute_optional(cu_seqlens_q),
            _to_cute_optional(cu_seqlens_k),
            _to_cute_optional(cu_total_q_blocks),
            _to_cute_optional(cu_total_k_blocks),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream,
        )
        compiled = cute.compile(
            kernel,
            *compile_args,
            options="--enable-tvm-ffi",
        )
        print(f"Compiled mask-plan K2Q materialize in {time.perf_counter() - started_at:.1f}s")
        _K2Q_MATERIALIZE_COMPILE_CACHE[key] = compiled
    return _K2Q_MATERIALIZE_COMPILE_CACHE[key]


def _validate_builder_inputs(
    arbitrary_func: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
):
    varlen_values = (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )
    is_varlen = all(value is not None for value in varlen_values)
    if not is_varlen and any(value is not None for value in varlen_values):
        raise ValueError("cu_seqlens_q, cu_seqlens_k, max_seqlen_q, and max_seqlen_k must be provided together")
    expected_rank = 3 if is_varlen else 4
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must have rank {expected_rank} in this mode")
        if tensor.device != q.device:
            raise ValueError("q, k, and v must be on the same device")
        if tensor.dtype != q.dtype:
            raise TypeError("q, k, and v must have the same dtype")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous in the last dimension")
    if not q.is_cuda and not is_fake_mode():
        raise ValueError("q, k, and v must be CUDA tensors")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if k.shape[-3] != v.shape[-3]:
        raise ValueError("k and v must have the same sequence extent")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same number of heads")
    if q.shape[-2] % k.shape[-2] != 0:
        raise ValueError("Hq must be divisible by Hkv")

    if arbitrary_func.ndim != 3:
        raise ValueError("arbitrary_func must have shape [Hmask, nfunc, total_q]")
    if arbitrary_func.dtype != torch.int32:
        raise TypeError("arbitrary_func must have dtype torch.int32")
    if arbitrary_func.device != q.device:
        raise ValueError("arbitrary_func must be on the same device as q")
    if not arbitrary_func.is_contiguous():
        raise ValueError("arbitrary_func must be contiguous")
    hmask, nfunc, func_q_extent = arbitrary_func.shape
    if hmask not in (1, q.shape[-2]):
        raise ValueError(f"Hmask must be 1 or Hq ({q.shape[-2]}); got {hmask}")
    if nfunc <= 0 or nfunc % 2 == 0:
        raise ValueError("nfunc must be a positive odd runtime value")

    if is_varlen:
        if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
            raise ValueError("cu_seqlens_q/k must be rank-1")
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape")
        for name, tensor in (
            ("cu_seqlens_q", cu_seqlens_q),
            ("cu_seqlens_k", cu_seqlens_k),
        ):
            if tensor.dtype != torch.int32 or tensor.device != q.device:
                raise ValueError(f"{name} must be int32 on the same device as q")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        batch_size = cu_seqlens_q.numel() - 1
        total_q = q.shape[0]
        total_k = k.shape[0]
        if not isinstance(max_seqlen_q, int) or not isinstance(max_seqlen_k, int):
            raise TypeError("max_seqlen_q/k must be Python ints")
        if max_seqlen_q < 0 or max_seqlen_k < 0:
            raise ValueError("max_seqlen_q/k must be non-negative")
        seqlen_q_fixed = 0
        seqlen_k_fixed = 0
    else:
        batch_size, seqlen_q_fixed = q.shape[:2]
        if k.shape[0] != batch_size or v.shape[0] != batch_size:
            raise ValueError("fixed q, k, and v must have the same batch size")
        seqlen_k_fixed = k.shape[1]
        total_q = batch_size * seqlen_q_fixed
        total_k = batch_size * seqlen_k_fixed
        max_seqlen_q = seqlen_q_fixed
        max_seqlen_k = seqlen_k_fixed
    if func_q_extent != total_q:
        raise ValueError(f"arbitrary_func last extent must equal total_q ({total_q})")
    return {
        "is_varlen": is_varlen,
        "batch_size": batch_size,
        "seqlen_q_fixed": seqlen_q_fixed,
        "seqlen_k_fixed": seqlen_k_fixed,
        "total_q": total_q,
        "total_k": total_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "hmask": hmask,
        "nfunc": nfunc,
    }


def _build_packed_mask_plan(
    arbitrary_func: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    pack_gqa: bool | None = None,
    build_backward: bool = False,
    _fwd_variant: _Sm100FwdVariant | None = None,
) -> BlockSparseTensorsTorch:
    """Build the internal consumer-specific packed-mask payloads."""

    if _fwd_variant not in (
        None,
        "qstage2_1cta",
        "qstage1_1cta",
        "qstage1_2cta",
    ):
        raise ValueError(f"unknown internal forward variant: {_fwd_variant!r}")

    metadata = _validate_builder_inputs(
        arbitrary_func,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )
    runtime_binding = ArbitraryPlanRuntimeBinding.capture(
        is_varlen=metadata["is_varlen"],
        batch_size=metadata["batch_size"],
        seqlen_q=(None if metadata["is_varlen"] else metadata["seqlen_q_fixed"]),
        seqlen_k=(None if metadata["is_varlen"] else metadata["seqlen_k_fixed"]),
        total_q=metadata["total_q"],
        total_k=metadata["total_k"],
        max_seqlen_q=metadata["max_seqlen_q"],
        max_seqlen_k=metadata["max_seqlen_k"],
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
    )
    device = q.device
    arch = _get_plan_builder_arch(device)
    dq_config = None
    if arch == 90:
        fwd_config = resolve_sm90_fwd_consumer_config(
            arch=arch,
            dtype=q.dtype,
            head_dim=q.shape[-1],
            head_dim_v=v.shape[-1],
            num_q_heads=q.shape[-2],
            num_kv_heads=k.shape[-2],
            is_varlen=metadata["is_varlen"],
            hmask=metadata["hmask"],
            pack_gqa=pack_gqa,
        )
        bwd_config = (
            resolve_sm90_bwd_consumer_config(
                arch=arch,
                dtype=q.dtype,
                head_dim=q.shape[-1],
                head_dim_v=v.shape[-1],
                num_q_heads=q.shape[-2],
                num_kv_heads=k.shape[-2],
                is_varlen=metadata["is_varlen"],
            )
            if build_backward
            else None
        )
        fwd_topology_config = fwd_config
    elif arch in (100, 103):
        use_hd256_consumer = q.shape[-1] == 256 and v.shape[-1] == 256
        if use_hd256_consumer and _fwd_variant == "qstage2_1cta":
            raise NotImplementedError("dedicated SM100 D256 forward does not support qstage2")
        if use_hd256_consumer:
            fwd_config = resolve_sm100_hd256_fwd_consumer_config(
                arch=arch,
                dtype=q.dtype,
                head_dim=q.shape[-1],
                head_dim_v=v.shape[-1],
                num_q_heads=q.shape[-2],
                num_kv_heads=k.shape[-2],
                is_varlen=metadata["is_varlen"],
                hmask=metadata["hmask"],
                pack_gqa=pack_gqa,
                cta_group_size=(2 if _fwd_variant == "qstage1_2cta" else 1),
            )
            if build_backward:
                dq_config = resolve_sm100_hd256_dq_consumer_config(
                    arch=arch,
                    dtype=q.dtype,
                    head_dim=q.shape[-1],
                    head_dim_v=v.shape[-1],
                    num_q_heads=q.shape[-2],
                    num_kv_heads=k.shape[-2],
                    is_varlen=metadata["is_varlen"],
                    hmask=metadata["hmask"],
                    pack_gqa=False,
                    use_2cta_instrs=True,
                    deterministic=False,
                )
        generic_fwd_variant = _fwd_variant or _SM100_GENERIC_DEFAULT_FWD_VARIANT
        if not use_hd256_consumer and generic_fwd_variant == "qstage1_1cta":
            fwd_config = resolve_sm100_fwd_qstage1_1cta_consumer_config(
                arch=arch,
                dtype=q.dtype,
                head_dim=q.shape[-1],
                head_dim_v=v.shape[-1],
                num_q_heads=q.shape[-2],
                num_kv_heads=k.shape[-2],
                is_varlen=metadata["is_varlen"],
                hmask=metadata["hmask"],
                pack_gqa=pack_gqa,
            )
        elif not use_hd256_consumer and generic_fwd_variant == "qstage1_2cta":
            fwd_config = resolve_sm100_fwd_qstage1_2cta_consumer_config(
                arch=arch,
                dtype=q.dtype,
                head_dim=q.shape[-1],
                head_dim_v=v.shape[-1],
                num_q_heads=q.shape[-2],
                num_kv_heads=k.shape[-2],
                is_varlen=metadata["is_varlen"],
                hmask=metadata["hmask"],
                pack_gqa=pack_gqa,
            )
        elif not use_hd256_consumer:
            fwd_config = resolve_sm100_fwd_consumer_config(
                arch=arch,
                dtype=q.dtype,
                head_dim=q.shape[-1],
                head_dim_v=v.shape[-1],
                num_q_heads=q.shape[-2],
                num_kv_heads=k.shape[-2],
                is_varlen=metadata["is_varlen"],
                hmask=metadata["hmask"],
                pack_gqa=pack_gqa,
            )
        if build_backward and use_hd256_consumer:
            bwd_config = resolve_sm100_hd256_dkdv_consumer_config(
                arch=arch,
                dtype=q.dtype,
                head_dim=q.shape[-1],
                head_dim_v=v.shape[-1],
                num_q_heads=q.shape[-2],
                num_kv_heads=k.shape[-2],
                is_varlen=metadata["is_varlen"],
                hmask=metadata["hmask"],
                pack_gqa=False,
                use_2cta_instrs=True,
                deterministic=False,
            )
        else:
            bwd_config = (
                resolve_sm100_bwd_consumer_config(
                    arch=arch,
                    dtype=q.dtype,
                    head_dim=q.shape[-1],
                    head_dim_v=v.shape[-1],
                    num_q_heads=q.shape[-2],
                    num_kv_heads=k.shape[-2],
                    is_varlen=metadata["is_varlen"],
                )
                if build_backward
                else None
            )
        fwd_topology_config = _ResolvedSm100Hd256FwdTopologyConfig(fwd_config) if use_hd256_consumer else _ResolvedSm100FwdTopologyConfig(fwd_config)
    else:
        raise NotImplementedError("arbitrary plan building supports SM90/SM100/SM103 only")

    qratio = fwd_config.qhead_per_kvhead if fwd_config.pack_gqa else 1
    fwd_plan_tile_m = fwd_config.block_size[0]
    fwd_max_m_blocks = math.ceil(metadata["max_seqlen_q"] * qratio / fwd_plan_tile_m)
    fwd_max_n_blocks = math.ceil(metadata["max_seqlen_k"] / fwd_config.tile_n)
    fwd_upper_total_m_blocks = metadata["batch_size"] * fwd_max_m_blocks
    fwd_num_words = max(1, math.ceil(fwd_max_n_blocks / 32))

    bwd_max_m_blocks = math.ceil(metadata["max_seqlen_q"] / bwd_config.sparse_tile_m) if bwd_config is not None else 0
    bwd_sparse_tile_n = (
        bwd_config.sparse_tile_n
        if isinstance(
            bwd_config,
            (_ResolvedSm100BwdConsumerConfig, _ResolvedSm100Hd256DkdvConsumerConfig),
        )
        else bwd_config.tile_n if bwd_config is not None else 0
    )
    bwd_max_n_blocks = math.ceil(metadata["max_seqlen_k"] / bwd_sparse_tile_n) if bwd_config is not None else 0
    bwd_upper_total_m_blocks = metadata["batch_size"] * bwd_max_m_blocks
    bwd_upper_total_n_blocks = metadata["batch_size"] * bwd_max_n_blocks
    bwd_num_words = max(1, math.ceil(bwd_max_n_blocks / 32))

    cu_total_m_blocks = None
    cu_total_fwd_n_blocks = None
    cu_total_bwd_m_blocks = None
    cu_total_bwd_n_blocks = None
    metadata_invalid = None
    if metadata["is_varlen"]:
        assert cu_seqlens_q is not None
        assert cu_seqlens_k is not None
        cu_total_m_blocks = torch.empty((metadata["batch_size"] + 1,), dtype=torch.int32, device=device)
        cu_total_fwd_n_blocks = torch.empty_like(cu_total_m_blocks)
        if bwd_config is not None:
            cu_total_bwd_m_blocks = torch.empty((metadata["batch_size"] + 1,), dtype=torch.int32, device=device)
            cu_total_bwd_n_blocks = torch.empty_like(cu_total_bwd_m_blocks)
        metadata_invalid = torch.empty((1,), dtype=torch.bool, device=device)
        if is_fake_mode():
            q_lengths_raw = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            k_lengths_raw = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            q_lengths = q_lengths_raw.clamp(min=0, max=metadata["max_seqlen_q"])
            k_lengths = k_lengths_raw.clamp(min=0, max=metadata["max_seqlen_k"])
            m_counts = torch.div(
                q_lengths * qratio + fwd_plan_tile_m - 1,
                fwd_plan_tile_m,
                rounding_mode="floor",
            ).to(torch.int32)
            cu_total_m_blocks[0] = 0
            cu_total_m_blocks[1:] = torch.cumsum(m_counts, dim=0, dtype=torch.int32)
            fwd_n_counts = torch.div(
                k_lengths + fwd_config.tile_n - 1,
                fwd_config.tile_n,
                rounding_mode="floor",
            ).to(torch.int32)
            cu_total_fwd_n_blocks[0] = 0
            cu_total_fwd_n_blocks[1:] = torch.cumsum(fwd_n_counts, dim=0, dtype=torch.int32)
            if bwd_config is not None:
                assert cu_total_bwd_m_blocks is not None
                assert cu_total_bwd_n_blocks is not None
                bwd_m_counts = torch.div(
                    q_lengths + bwd_config.sparse_tile_m - 1,
                    bwd_config.sparse_tile_m,
                    rounding_mode="floor",
                ).to(torch.int32)
                bwd_n_counts = torch.div(
                    k_lengths + bwd_sparse_tile_n - 1,
                    bwd_sparse_tile_n,
                    rounding_mode="floor",
                ).to(torch.int32)
                cu_total_bwd_m_blocks[0] = 0
                cu_total_bwd_n_blocks[0] = 0
                cu_total_bwd_m_blocks[1:] = torch.cumsum(bwd_m_counts, dim=0, dtype=torch.int32)
                cu_total_bwd_n_blocks[1:] = torch.cumsum(bwd_n_counts, dim=0, dtype=torch.int32)
            metadata_invalid[0] = (
                (cu_seqlens_q[0] != 0)
                | (cu_seqlens_k[0] != 0)
                | (cu_seqlens_q[-1] != metadata["total_q"])
                | (cu_seqlens_k[-1] != metadata["total_k"])
                | torch.any(q_lengths_raw < 0)
                | torch.any(k_lengths_raw < 0)
                | torch.any(q_lengths_raw > metadata["max_seqlen_q"])
                | torch.any(k_lengths_raw > metadata["max_seqlen_k"])
            )
        else:
            varlen_geometry = _compile_varlen_geometry(
                arch,
                fwd_tile_m=fwd_plan_tile_m,
                fwd_tile_n=fwd_config.tile_n,
                qhead_per_kvhead=qratio,
                bwd_tile_m=bwd_config.sparse_tile_m if bwd_config is not None else 1,
                bwd_tile_n=bwd_sparse_tile_n if bwd_config is not None else 1,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                cu_total_m_blocks=cu_total_m_blocks,
                cu_total_fwd_n_blocks=cu_total_fwd_n_blocks,
                cu_total_bwd_m_blocks=cu_total_bwd_m_blocks,
                cu_total_bwd_n_blocks=cu_total_bwd_n_blocks,
                metadata_invalid=metadata_invalid,
            )
            varlen_geometry(
                cu_seqlens_q,
                cu_seqlens_k,
                cu_total_m_blocks,
                cu_total_fwd_n_blocks,
                cu_total_bwd_m_blocks,
                cu_total_bwd_n_blocks,
                metadata_invalid,
                Int32(metadata["total_q"]),
                Int32(metadata["total_k"]),
                Int32(metadata["max_seqlen_q"]),
                Int32(metadata["max_seqlen_k"]),
            )

    visible_bits = torch.empty(
        (metadata["hmask"], fwd_upper_total_m_blocks, fwd_num_words),
        dtype=torch.uint32,
        device=device,
    )
    full_bits = torch.empty_like(visible_bits)
    partial_counts_tmp = torch.empty(
        (metadata["hmask"], fwd_upper_total_m_blocks),
        dtype=torch.int32,
        device=device,
    )
    full_counts_tmp = torch.empty_like(partial_counts_tmp)
    interval_invalid = torch.empty((1,), dtype=torch.uint32, device=device)
    num_schedule_sections = metadata["batch_size"] * fwd_config.num_kv_heads
    schedule_histogram = torch.empty(
        (num_schedule_sections, fwd_max_n_blocks + 1),
        dtype=torch.int32,
        device=device,
    )
    schedule_section_cost = torch.empty(
        (num_schedule_sections,),
        dtype=torch.int64,
        device=device,
    )

    bwd_visible_bits = None
    bwd_full_bits = None
    bwd_q_partial_counts_tmp = None
    bwd_q_full_counts_tmp = None
    bwd_partial_counts_tmp = None
    bwd_full_counts_tmp = None
    if bwd_config is not None:
        bwd_visible_bits = torch.empty(
            (metadata["hmask"], bwd_upper_total_m_blocks, bwd_num_words),
            dtype=torch.uint32,
            device=device,
        )
        bwd_full_bits = torch.empty_like(bwd_visible_bits)
        bwd_q_partial_counts_tmp = torch.empty(
            (metadata["hmask"], bwd_upper_total_m_blocks),
            dtype=torch.int32,
            device=device,
        )
        bwd_q_full_counts_tmp = torch.empty_like(bwd_q_partial_counts_tmp)
        bwd_partial_counts_tmp = torch.empty(
            (metadata["hmask"], bwd_upper_total_n_blocks),
            dtype=torch.int32,
            device=device,
        )
        bwd_full_counts_tmp = torch.empty_like(bwd_partial_counts_tmp)

    if not is_fake_mode():
        workspace_init = _compile_plan_workspace_init(
            arch,
            visible_bits,
            full_bits,
            partial_counts_tmp,
            full_counts_tmp,
            interval_invalid,
            schedule_histogram,
            schedule_section_cost,
            bwd_visible_bits,
            bwd_full_bits,
            bwd_q_partial_counts_tmp,
            bwd_q_full_counts_tmp,
            bwd_partial_counts_tmp,
            bwd_full_counts_tmp,
        )
        workspace_init(
            visible_bits,
            full_bits,
            partial_counts_tmp,
            full_counts_tmp,
            interval_invalid,
            schedule_histogram,
            schedule_section_cost,
            bwd_visible_bits,
            bwd_full_bits,
            bwd_q_partial_counts_tmp,
            bwd_q_full_counts_tmp,
            bwd_partial_counts_tmp,
            bwd_full_counts_tmp,
        )

    if fwd_upper_total_m_blocks > 0 and fwd_max_n_blocks > 0:
        classify = _compile_classify(
            fwd_topology_config,
            arbitrary_func,
            visible_bits,
            full_bits,
            partial_counts_tmp,
            full_counts_tmp,
            interval_invalid,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_total_m_blocks,
        )
        if not is_fake_mode():
            classify(
                arbitrary_func,
                visible_bits,
                full_bits,
                partial_counts_tmp,
                full_counts_tmp,
                interval_invalid,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_total_m_blocks,
                Int32(metadata["batch_size"]),
                Int32(metadata["seqlen_q_fixed"]),
                Int32(metadata["seqlen_k_fixed"]),
                Int32(metadata["total_q"]),
                Int32(metadata["total_k"]),
                Int32(fwd_max_m_blocks),
                Int32(fwd_max_n_blocks),
                Int32(metadata["nfunc"]),
            )
    if bwd_config is not None and bwd_upper_total_m_blocks > 0 and bwd_max_n_blocks > 0:
        if isinstance(bwd_config, _ResolvedSm100BwdConsumerConfig):
            bwd_topology_config = _ResolvedSm100BwdTopologyConfig(bwd_config)
        elif isinstance(bwd_config, _ResolvedSm100Hd256DkdvConsumerConfig):
            bwd_topology_config = _ResolvedSm100Hd256DkdvTopologyConfig(bwd_config)
        else:
            bwd_topology_config = _ResolvedSm90BwdTopologyConfig(bwd_config)
        bwd_classify = _compile_classify(
            bwd_topology_config,
            arbitrary_func,
            bwd_visible_bits,
            bwd_full_bits,
            bwd_q_partial_counts_tmp,
            bwd_q_full_counts_tmp,
            interval_invalid,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_total_bwd_m_blocks,
        )
        if not is_fake_mode():
            bwd_classify(
                arbitrary_func,
                bwd_visible_bits,
                bwd_full_bits,
                bwd_q_partial_counts_tmp,
                bwd_q_full_counts_tmp,
                interval_invalid,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_total_bwd_m_blocks,
                Int32(metadata["batch_size"]),
                Int32(metadata["seqlen_q_fixed"]),
                Int32(metadata["seqlen_k_fixed"]),
                Int32(metadata["total_q"]),
                Int32(metadata["total_k"]),
                Int32(bwd_max_m_blocks),
                Int32(bwd_max_n_blocks),
                Int32(metadata["nfunc"]),
            )
    if bwd_config is not None and bwd_upper_total_n_blocks > 0 and bwd_max_m_blocks > 0:
        k2q_count = _compile_k2q_count(
            bwd_config,
            bwd_visible_bits,
            bwd_full_bits,
            bwd_partial_counts_tmp,
            bwd_full_counts_tmp,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_total_bwd_m_blocks,
            cu_total_bwd_n_blocks,
        )
        if not is_fake_mode():
            k2q_count(
                bwd_visible_bits,
                bwd_full_bits,
                bwd_partial_counts_tmp,
                bwd_full_counts_tmp,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_total_bwd_m_blocks,
                cu_total_bwd_n_blocks,
                Int32(metadata["batch_size"]),
                Int32(metadata["seqlen_q_fixed"]),
                Int32(metadata["seqlen_k_fixed"]),
                Int32(metadata["total_q"]),
                Int32(metadata["total_k"]),
                Int32(bwd_max_n_blocks),
            )
    partial_scan = None
    full_scan = None
    bwd_partial_scan = None
    bwd_full_scan = None
    dq_partial_scan = None
    dq_full_scan = None
    fixed_partial_offsets = None
    fixed_full_offsets = None
    fixed_bwd_partial_offsets = None
    fixed_bwd_full_offsets = None
    fixed_dq_partial_offsets = None
    fixed_dq_full_offsets = None
    use_fixed_scan_header = not metadata["is_varlen"] and fwd_max_n_blocks > 0
    use_varlen_scan_header = metadata["is_varlen"] and fwd_max_n_blocks > 0
    if is_fake_mode():
        total_m_blocks = fwd_upper_total_m_blocks
        partial_nnz = metadata["hmask"] * fwd_upper_total_m_blocks * fwd_max_n_blocks
        full_nnz = 0
        bwd_total_m_blocks = bwd_upper_total_m_blocks
        bwd_total_n_blocks = bwd_upper_total_n_blocks
        bwd_partial_nnz = metadata["hmask"] * bwd_upper_total_n_blocks * bwd_max_m_blocks
        bwd_full_nnz = 0
    elif use_fixed_scan_header:
        fixed_partial_offsets = torch.empty((partial_counts_tmp.numel() + 1,), dtype=torch.int32, device=device)
        fixed_full_offsets = torch.empty_like(fixed_partial_offsets)
        if bwd_config is not None:
            assert bwd_partial_counts_tmp is not None
            assert bwd_full_counts_tmp is not None
            fixed_bwd_partial_offsets = torch.empty(
                (bwd_partial_counts_tmp.numel() + 1,),
                dtype=torch.int32,
                device=device,
            )
            fixed_bwd_full_offsets = torch.empty_like(fixed_bwd_partial_offsets)
        if dq_config is not None:
            assert bwd_q_partial_counts_tmp is not None
            fixed_dq_partial_offsets = torch.empty(
                (bwd_q_partial_counts_tmp.numel() + 1,),
                dtype=torch.int32,
                device=device,
            )
            fixed_dq_full_offsets = torch.empty_like(fixed_dq_partial_offsets)
        header = torch.empty((9,), dtype=torch.int64, device=device)
        fixed_scan_header = _compile_fixed_scan_header(
            arch,
            partial_counts_tmp,
            full_counts_tmp,
            fixed_partial_offsets,
            fixed_full_offsets,
            bwd_partial_counts_tmp,
            bwd_full_counts_tmp,
            fixed_bwd_partial_offsets,
            fixed_bwd_full_offsets,
            bwd_q_partial_counts_tmp if dq_config is not None else None,
            bwd_q_full_counts_tmp if dq_config is not None else None,
            fixed_dq_partial_offsets,
            fixed_dq_full_offsets,
            interval_invalid,
            header,
        )
        fixed_scan_header(
            partial_counts_tmp,
            full_counts_tmp,
            fixed_partial_offsets,
            fixed_full_offsets,
            bwd_partial_counts_tmp,
            bwd_full_counts_tmp,
            fixed_bwd_partial_offsets,
            fixed_bwd_full_offsets,
            bwd_q_partial_counts_tmp if dq_config is not None else None,
            bwd_q_full_counts_tmp if dq_config is not None else None,
            fixed_dq_partial_offsets,
            fixed_dq_full_offsets,
            interval_invalid,
            header,
            Int32(fwd_upper_total_m_blocks),
            Int32(bwd_upper_total_m_blocks),
            Int32(bwd_upper_total_n_blocks),
        )
        (
            total_m_blocks,
            partial_nnz,
            full_nnz,
            bwd_total_m_blocks,
            bwd_total_n_blocks,
            bwd_partial_nnz,
            bwd_full_nnz,
            interval_invalid_value,
            seqlens_invalid_value,
        ) = (int(value) for value in header.cpu().tolist())
    elif use_varlen_scan_header:
        partial_scan = torch.empty((partial_counts_tmp.numel(),), dtype=torch.int64, device=device)
        full_scan = torch.empty_like(partial_scan)
        if bwd_config is not None:
            assert bwd_partial_counts_tmp is not None
            assert bwd_full_counts_tmp is not None
            bwd_partial_scan = torch.empty((bwd_partial_counts_tmp.numel(),), dtype=torch.int64, device=device)
            bwd_full_scan = torch.empty_like(bwd_partial_scan)
        if dq_config is not None:
            assert bwd_q_partial_counts_tmp is not None
            assert bwd_q_full_counts_tmp is not None
            dq_partial_scan = torch.empty((bwd_q_partial_counts_tmp.numel(),), dtype=torch.int64, device=device)
            dq_full_scan = torch.empty_like(dq_partial_scan)
        assert cu_total_m_blocks is not None
        assert metadata_invalid is not None
        header = torch.empty((9,), dtype=torch.int64, device=device)
        varlen_scan_header = _compile_varlen_scan_header(
            arch,
            partial_counts_tmp,
            full_counts_tmp,
            partial_scan,
            full_scan,
            bwd_partial_counts_tmp,
            bwd_full_counts_tmp,
            bwd_partial_scan,
            bwd_full_scan,
            bwd_q_partial_counts_tmp if dq_config is not None else None,
            bwd_q_full_counts_tmp if dq_config is not None else None,
            dq_partial_scan,
            dq_full_scan,
            cu_total_m_blocks,
            cu_total_bwd_m_blocks,
            cu_total_bwd_n_blocks,
            metadata_invalid,
            interval_invalid,
            header,
        )
        varlen_scan_header(
            partial_counts_tmp,
            full_counts_tmp,
            partial_scan,
            full_scan,
            bwd_partial_counts_tmp,
            bwd_full_counts_tmp,
            bwd_partial_scan,
            bwd_full_scan,
            bwd_q_partial_counts_tmp if dq_config is not None else None,
            bwd_q_full_counts_tmp if dq_config is not None else None,
            dq_partial_scan,
            dq_full_scan,
            cu_total_m_blocks,
            cu_total_bwd_m_blocks,
            cu_total_bwd_n_blocks,
            metadata_invalid,
            interval_invalid,
            header,
        )
        (
            total_m_blocks,
            partial_nnz,
            full_nnz,
            bwd_total_m_blocks,
            bwd_total_n_blocks,
            bwd_partial_nnz,
            bwd_full_nnz,
            interval_invalid_value,
            seqlens_invalid_value,
        ) = (int(value) for value in header.cpu().tolist())
    else:
        partial_scan = torch.cumsum(partial_counts_tmp.reshape(-1), dim=0, dtype=torch.int64)
        full_scan = torch.cumsum(full_counts_tmp.reshape(-1), dim=0, dtype=torch.int64)
        total_m_tensor = (
            cu_total_m_blocks[-1].to(torch.int64) if cu_total_m_blocks is not None else torch.tensor(fwd_upper_total_m_blocks, dtype=torch.int64, device=device)
        )
        partial_total = partial_scan[-1] if partial_scan.numel() else torch.zeros((), dtype=torch.int64, device=device)
        full_total = full_scan[-1] if full_scan.numel() else torch.zeros((), dtype=torch.int64, device=device)
        zero = torch.zeros((), dtype=torch.int64, device=device)
        bwd_partial_scan = torch.cumsum(bwd_partial_counts_tmp.reshape(-1), dim=0, dtype=torch.int64) if bwd_config is not None else None
        bwd_full_scan = torch.cumsum(bwd_full_counts_tmp.reshape(-1), dim=0, dtype=torch.int64) if bwd_config is not None else None
        bwd_total_m_tensor = (
            cu_total_bwd_m_blocks[-1].to(torch.int64)
            if cu_total_bwd_m_blocks is not None
            else (torch.tensor(bwd_upper_total_m_blocks, dtype=torch.int64, device=device) if bwd_config is not None else zero)
        )
        bwd_total_n_tensor = (
            cu_total_bwd_n_blocks[-1].to(torch.int64)
            if cu_total_bwd_n_blocks is not None
            else (torch.tensor(bwd_upper_total_n_blocks, dtype=torch.int64, device=device) if bwd_config is not None else zero)
        )
        bwd_partial_total = bwd_partial_scan[-1] if bwd_partial_scan is not None and bwd_partial_scan.numel() else zero
        bwd_full_total = bwd_full_scan[-1] if bwd_full_scan is not None and bwd_full_scan.numel() else zero
        interval_invalid_header = interval_invalid[0].to(torch.int64)
        if fwd_upper_total_m_blocks > 0 and fwd_max_n_blocks == 0:
            interval_invalid_header = torch.maximum(
                interval_invalid_header,
                torch.any(arbitrary_func != 0).to(torch.int64),
            )
        seqlens_invalid_header = metadata_invalid[0].to(torch.int64) if metadata_invalid is not None else zero
        header = torch.stack(
            (
                total_m_tensor,
                partial_total,
                full_total,
                bwd_total_m_tensor,
                bwd_total_n_tensor,
                bwd_partial_total,
                bwd_full_total,
                interval_invalid_header,
                seqlens_invalid_header,
            )
        )
        (
            total_m_blocks,
            partial_nnz,
            full_nnz,
            bwd_total_m_blocks,
            bwd_total_n_blocks,
            bwd_partial_nnz,
            bwd_full_nnz,
            interval_invalid_value,
            seqlens_invalid_value,
        ) = (int(value) for value in header.cpu().tolist())
    if not is_fake_mode():
        if seqlens_invalid_value:
            raise ValueError("cu_seqlens_q/k must start at zero, be nondecreasing, end at " "total_q/total_k, and respect max_seqlen_q/k")
        if interval_invalid_value:
            raise ValueError("mask_func endpoints must lie in each row's local-K range and be " "nondecreasing for every Q row")

    use_varlen_compact_metadata = not is_fake_mode() and metadata["is_varlen"] and use_varlen_scan_header
    if use_varlen_compact_metadata:
        partial_counts = torch.empty((metadata["hmask"], total_m_blocks), dtype=torch.int32, device=device)
        full_counts = torch.empty_like(partial_counts)
        partial_offsets = torch.empty((partial_counts.numel() + 1,), dtype=torch.int32, device=device)
        full_offsets = torch.empty_like(partial_offsets)
    else:
        partial_counts = _compact_counts(
            partial_counts_tmp,
            total_m_blocks,
            is_varlen=metadata["is_varlen"],
        )
        full_counts = _compact_counts(
            full_counts_tmp,
            total_m_blocks,
            is_varlen=metadata["is_varlen"],
        )
        partial_offsets = fixed_partial_offsets if fixed_partial_offsets is not None else _exclusive_offsets_from_scan(partial_counts, partial_scan)
        full_offsets = fixed_full_offsets if fixed_full_offsets is not None else _exclusive_offsets_from_scan(full_counts, full_scan)
    partial_indices = torch.empty((partial_nnz,), dtype=torch.int32, device=device)
    full_indices = torch.empty((full_nnz,), dtype=torch.int32, device=device)
    partial_masks = torch.empty(
        (
            partial_nnz,
            fwd_config.physical_subtiles,
            fwd_config.num_mask_payload_groups,
            fwd_config.payload_padded_words,
        ),
        dtype=torch.uint32,
        device=device,
    )
    partial_work_desc = torch.empty(
        (partial_nnz, 4),
        dtype=torch.int32,
        device=device,
    )
    dq_partial_counts = None
    dq_full_counts = None
    dq_partial_offsets = None
    dq_full_offsets = None
    dq_partial_indices = None
    dq_full_indices = None
    dq_partial_masks = None
    dq_partial_work_desc = None
    if dq_config is not None:
        assert isinstance(bwd_config, _ResolvedSm100Hd256DkdvConsumerConfig)
        assert dq_config.block_size == bwd_config.block_size
        assert bwd_q_partial_counts_tmp is not None
        assert bwd_q_full_counts_tmp is not None
        if use_varlen_compact_metadata:
            dq_partial_counts = torch.empty(
                (metadata["hmask"], bwd_total_m_blocks),
                dtype=torch.int32,
                device=device,
            )
            dq_full_counts = torch.empty_like(dq_partial_counts)
            dq_partial_offsets = torch.empty((dq_partial_counts.numel() + 1,), dtype=torch.int32, device=device)
            dq_full_offsets = torch.empty_like(dq_partial_offsets)
        else:
            dq_partial_counts = _compact_counts(
                bwd_q_partial_counts_tmp,
                bwd_total_m_blocks,
                is_varlen=metadata["is_varlen"],
            )
            dq_full_counts = _compact_counts(
                bwd_q_full_counts_tmp,
                bwd_total_m_blocks,
                is_varlen=metadata["is_varlen"],
            )
            dq_partial_offsets = (
                fixed_dq_partial_offsets if fixed_dq_partial_offsets is not None else _exclusive_offsets_from_scan(dq_partial_counts, dq_partial_scan)
            )
            dq_full_offsets = fixed_dq_full_offsets if fixed_dq_full_offsets is not None else _exclusive_offsets_from_scan(dq_full_counts, dq_full_scan)
        # Q-major and K-major views enumerate the same classified sparse pairs.
        dq_partial_indices = torch.empty((bwd_partial_nnz,), dtype=torch.int32, device=device)
        dq_full_indices = torch.empty((bwd_full_nnz,), dtype=torch.int32, device=device)
        dq_partial_masks = torch.empty(
            (
                bwd_partial_nnz,
                dq_config.physical_subtiles,
                dq_config.num_mask_payload_groups,
                dq_config.payload_padded_words,
            ),
            dtype=torch.uint32,
            device=device,
        )
        dq_partial_work_desc = torch.empty((bwd_partial_nnz, 4), dtype=torch.int32, device=device)
    bwd_partial_counts = None
    bwd_full_counts = None
    bwd_partial_offsets = None
    bwd_full_offsets = None
    bwd_plan = None
    if bwd_config is not None:
        if use_varlen_compact_metadata:
            bwd_partial_counts = torch.empty(
                (metadata["hmask"], bwd_total_n_blocks),
                dtype=torch.int32,
                device=device,
            )
            bwd_full_counts = torch.empty_like(bwd_partial_counts)
            bwd_partial_offsets = torch.empty((bwd_partial_counts.numel() + 1,), dtype=torch.int32, device=device)
            bwd_full_offsets = torch.empty_like(bwd_partial_offsets)
        else:
            bwd_partial_counts = _compact_counts(
                bwd_partial_counts_tmp,
                bwd_total_n_blocks,
                is_varlen=metadata["is_varlen"],
            )
            bwd_full_counts = _compact_counts(
                bwd_full_counts_tmp,
                bwd_total_n_blocks,
                is_varlen=metadata["is_varlen"],
            )
            bwd_partial_offsets = (
                fixed_bwd_partial_offsets if fixed_bwd_partial_offsets is not None else _exclusive_offsets_from_scan(bwd_partial_counts, bwd_partial_scan)
            )
            bwd_full_offsets = fixed_bwd_full_offsets if fixed_bwd_full_offsets is not None else _exclusive_offsets_from_scan(bwd_full_counts, bwd_full_scan)
        bwd_partial_indices = torch.empty((bwd_partial_nnz,), dtype=torch.int32, device=device)
        bwd_full_indices = torch.empty((bwd_full_nnz,), dtype=torch.int32, device=device)
        bwd_partial_dq_order = torch.empty_like(bwd_partial_indices)
        bwd_full_dq_order = torch.empty_like(bwd_full_indices)
        bwd_partial_masks = torch.empty(
            (
                bwd_partial_nnz,
                bwd_config.physical_subtiles,
                bwd_config.num_mma_threads,
                bwd_config.payload_padded_words,
            ),
            dtype=torch.uint32,
            device=device,
        )
        bwd_partial_work_desc = torch.empty(
            (bwd_partial_nnz, 4),
            dtype=torch.int32,
            device=device,
        )
    if use_varlen_compact_metadata:
        assert partial_scan is not None
        assert full_scan is not None
        compact_metadata = _compile_varlen_compact_metadata(
            arch,
            partial_counts_tmp,
            full_counts_tmp,
            partial_scan,
            full_scan,
            partial_counts,
            full_counts,
            partial_offsets,
            full_offsets,
            bwd_partial_counts_tmp,
            bwd_full_counts_tmp,
            bwd_partial_scan,
            bwd_full_scan,
            bwd_partial_counts,
            bwd_full_counts,
            bwd_partial_offsets,
            bwd_full_offsets,
            bwd_q_partial_counts_tmp if dq_config is not None else None,
            bwd_q_full_counts_tmp if dq_config is not None else None,
            dq_partial_scan,
            dq_full_scan,
            dq_partial_counts,
            dq_full_counts,
            dq_partial_offsets,
            dq_full_offsets,
        )
        compact_metadata(
            partial_counts_tmp,
            full_counts_tmp,
            partial_scan,
            full_scan,
            partial_counts,
            full_counts,
            partial_offsets,
            full_offsets,
            bwd_partial_counts_tmp,
            bwd_full_counts_tmp,
            bwd_partial_scan,
            bwd_full_scan,
            bwd_partial_counts,
            bwd_full_counts,
            bwd_partial_offsets,
            bwd_full_offsets,
            bwd_q_partial_counts_tmp if dq_config is not None else None,
            bwd_q_full_counts_tmp if dq_config is not None else None,
            dq_partial_scan,
            dq_full_scan,
            dq_partial_counts,
            dq_full_counts,
            dq_partial_offsets,
            dq_full_offsets,
        )
    if fwd_upper_total_m_blocks > 0 and partial_nnz + full_nnz > 0:
        materialize = _compile_materialize(
            fwd_config,
            arbitrary_func,
            visible_bits,
            full_bits,
            partial_offsets,
            partial_indices,
            partial_masks,
            partial_work_desc,
            full_offsets,
            full_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_total_m_blocks,
        )
        if not is_fake_mode():
            materialize_args = (
                arbitrary_func,
                visible_bits,
                full_bits,
                partial_offsets,
                partial_indices,
                partial_masks,
            )
            if partial_work_desc is not None:
                materialize_args += (partial_work_desc,)
            materialize_args += (
                full_offsets,
                full_indices,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_total_m_blocks,
                Int32(metadata["batch_size"]),
                Int32(metadata["seqlen_q_fixed"]),
                Int32(metadata["seqlen_k_fixed"]),
                Int32(metadata["total_q"]),
                Int32(metadata["total_k"]),
                Int32(fwd_max_m_blocks),
                Int32(fwd_max_n_blocks),
                Int32(metadata["nfunc"]),
            )
            materialize(*materialize_args)
    if dq_config is not None and dq_partial_masks is not None and bwd_upper_total_m_blocks > 0 and bwd_partial_nnz + bwd_full_nnz > 0:
        assert bwd_visible_bits is not None
        assert bwd_full_bits is not None
        assert dq_partial_offsets is not None
        assert dq_partial_indices is not None
        assert dq_partial_work_desc is not None
        assert dq_full_offsets is not None
        assert dq_full_indices is not None
        dq_materialize = _compile_materialize(
            dq_config,
            arbitrary_func,
            bwd_visible_bits,
            bwd_full_bits,
            dq_partial_offsets,
            dq_partial_indices,
            dq_partial_masks,
            dq_partial_work_desc,
            dq_full_offsets,
            dq_full_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_total_bwd_m_blocks,
        )
        if not is_fake_mode():
            dq_materialize_args = (
                arbitrary_func,
                bwd_visible_bits,
                bwd_full_bits,
                dq_partial_offsets,
                dq_partial_indices,
                dq_partial_masks,
                dq_partial_work_desc,
                dq_full_offsets,
                dq_full_indices,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_total_bwd_m_blocks,
                Int32(metadata["batch_size"]),
                Int32(metadata["seqlen_q_fixed"]),
                Int32(metadata["seqlen_k_fixed"]),
                Int32(metadata["total_q"]),
                Int32(metadata["total_k"]),
                Int32(bwd_max_m_blocks),
                Int32(bwd_max_n_blocks),
                Int32(metadata["nfunc"]),
            )
            dq_materialize(*dq_materialize_args)
    if bwd_config is not None and bwd_upper_total_n_blocks > 0 and bwd_partial_nnz + bwd_full_nnz > 0:
        bwd_materialize = _compile_k2q_materialize(
            bwd_config,
            arbitrary_func,
            bwd_visible_bits,
            bwd_full_bits,
            bwd_q_partial_counts_tmp,
            bwd_q_full_counts_tmp,
            bwd_partial_offsets,
            bwd_partial_indices,
            bwd_partial_masks,
            bwd_partial_work_desc,
            bwd_partial_dq_order,
            bwd_full_offsets,
            bwd_full_indices,
            bwd_full_dq_order,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_total_bwd_m_blocks,
            cu_total_bwd_n_blocks,
        )
        if not is_fake_mode():
            bwd_materialize_args = (
                arbitrary_func,
                bwd_visible_bits,
                bwd_full_bits,
                bwd_q_partial_counts_tmp,
                bwd_q_full_counts_tmp,
                bwd_partial_offsets,
                bwd_partial_indices,
                bwd_partial_masks,
            )
            if bwd_partial_work_desc is not None:
                bwd_materialize_args += (bwd_partial_work_desc,)
            bwd_materialize_args += (
                bwd_partial_dq_order,
                bwd_full_offsets,
                bwd_full_indices,
                bwd_full_dq_order,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_total_bwd_m_blocks,
                cu_total_bwd_n_blocks,
                Int32(metadata["batch_size"]),
                Int32(metadata["seqlen_q_fixed"]),
                Int32(metadata["seqlen_k_fixed"]),
                Int32(metadata["total_q"]),
                Int32(metadata["total_k"]),
                Int32(bwd_max_n_blocks),
                Int32(metadata["nfunc"]),
            )
            bwd_materialize(*bwd_materialize_args)
    sequence_desc = None
    fwd_work_desc = None
    if isinstance(
        fwd_config,
        (
            _ResolvedSm90FwdConsumerConfig,
            _ResolvedSm100FwdConsumerConfig,
            _ResolvedSm100Hd256FwdConsumerConfig,
        ),
    ):
        sequence_desc, fwd_work_desc = _build_forward_schedule(
            fwd_config,
            partial_counts,
            full_counts,
            batch_size=metadata["batch_size"],
            total_m_blocks=total_m_blocks,
            max_m_blocks=fwd_max_m_blocks,
            max_task_cost=fwd_max_n_blocks,
            seqlen_q_fixed=metadata["seqlen_q_fixed"],
            seqlen_k_fixed=metadata["seqlen_k_fixed"],
            head_dim=q.shape[-1],
            head_dim_v=v.shape[-1],
            element_size=q.element_size(),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_total_m_blocks=cu_total_m_blocks,
            schedule_histogram=schedule_histogram,
            schedule_section_cost=schedule_section_cost,
        )
    plan_rows = metadata["hmask"] * total_m_blocks
    # Full and partial blocks have equal scheduling cost.  The overlapped
    # qstage1 schedule wins consistently from seven blocks per plan row.
    narrow_workset = plan_rows > 0 and partial_nnz + full_nnz < _QSTAGE1_OVERLAP_MIN_AVERAGE_BLOCKS * plan_rows

    q2k_topology = ArbitraryTopologyTensors(
        direction="q2k",
        partial_count=partial_counts,
        partial_offset=partial_offsets,
        partial_index=partial_indices,
        full_count=full_counts,
        full_offset=full_offsets,
        full_index=full_indices,
        cu_total_q_plan_rows=cu_total_m_blocks,
        cu_total_k_plan_rows=cu_total_fwd_n_blocks,
        runtime_binding=runtime_binding,
    )
    dq_plan = None
    if dq_config is not None:
        assert dq_partial_masks is not None
        assert dq_partial_counts is not None
        assert dq_full_counts is not None
        assert dq_partial_offsets is not None
        assert dq_full_offsets is not None
        assert dq_partial_indices is not None
        assert dq_full_indices is not None
        dq_topology = ArbitraryTopologyTensors(
            direction="q2k",
            partial_count=dq_partial_counts,
            partial_offset=dq_partial_offsets,
            partial_index=dq_partial_indices,
            full_count=dq_full_counts,
            full_offset=dq_full_offsets,
            full_index=dq_full_indices,
            cu_total_q_plan_rows=cu_total_bwd_m_blocks,
            cu_total_k_plan_rows=cu_total_bwd_n_blocks,
            runtime_binding=runtime_binding,
        )
        dq_plan_rows = metadata["hmask"] * bwd_total_m_blocks
        dq_narrow_workset = dq_plan_rows > 0 and bwd_partial_nnz + bwd_full_nnz <= 8 * dq_plan_rows
        dq_plan = BlockSparseTensorsTorch(
            mask_block_cnt=dq_partial_counts,
            mask_block_idx=dq_partial_indices,
            full_block_cnt=dq_full_counts,
            full_block_idx=dq_full_indices,
            cu_total_m_blocks=(cu_total_bwd_m_blocks if metadata["is_varlen"] else None),
            block_size=dq_config.block_size,
            dq_write_order=None,
            dq_write_order_full=None,
            spt=None,
            mask_block_offset=dq_partial_offsets,
            full_block_offset=dq_full_offsets,
            mask_block_masks=dq_partial_masks,
            pack_gqa=False,
            bwd_tensors=None,
            plan_signature=_consumer_plan_signature(dq_config),
            topology_tensors=dq_topology,
            narrow_workset=dq_narrow_workset,
        )

    if bwd_config is not None:
        hd256_dkdv = isinstance(bwd_config, _ResolvedSm100Hd256DkdvConsumerConfig)
        exposed_partial_dq_order = None if hd256_dkdv else bwd_partial_dq_order
        exposed_full_dq_order = None if hd256_dkdv else bwd_full_dq_order
        bwd_plan = BlockSparseTensorsTorch(
            mask_block_cnt=bwd_partial_counts,
            mask_block_idx=bwd_partial_indices,
            full_block_cnt=bwd_full_counts,
            full_block_idx=bwd_full_indices,
            cu_total_m_blocks=(cu_total_bwd_n_blocks if metadata["is_varlen"] else None),
            block_size=bwd_config.block_size,
            dq_write_order=exposed_partial_dq_order,
            dq_write_order_full=exposed_full_dq_order,
            spt=bwd_config.spt,
            mask_block_offset=bwd_partial_offsets,
            full_block_offset=bwd_full_offsets,
            mask_block_masks=bwd_partial_masks,
            pack_gqa=None,
            bwd_tensors=None,
            plan_signature=_consumer_plan_signature(bwd_config),
            topology_tensors=ArbitraryTopologyTensors(
                direction="k2q",
                partial_count=bwd_partial_counts,
                partial_offset=bwd_partial_offsets,
                partial_index=bwd_partial_indices,
                full_count=bwd_full_counts,
                full_offset=bwd_full_offsets,
                full_index=bwd_full_indices,
                cu_total_q_plan_rows=cu_total_bwd_m_blocks,
                cu_total_k_plan_rows=cu_total_bwd_n_blocks,
                runtime_binding=runtime_binding,
                dq_write_order=exposed_partial_dq_order,
                dq_write_order_full=exposed_full_dq_order,
            ),
        )

    plan = BlockSparseTensorsTorch(
        mask_block_cnt=partial_counts,
        mask_block_idx=partial_indices,
        full_block_cnt=full_counts,
        full_block_idx=full_indices,
        cu_total_m_blocks=cu_total_m_blocks,
        block_size=fwd_config.block_size,
        dq_write_order=None,
        dq_write_order_full=None,
        spt=None,
        mask_block_offset=partial_offsets,
        full_block_offset=full_offsets,
        mask_block_masks=partial_masks,
        pack_gqa=fwd_config.pack_gqa,
        bwd_tensors=bwd_plan,
        plan_signature=_consumer_plan_signature(fwd_config),
        topology_tensors=q2k_topology,
        dq_tensors=dq_plan,
        narrow_workset=narrow_workset,
        sequence_desc=sequence_desc,
        fwd_work_desc=fwd_work_desc,
    )

    return plan


def create_mask_plan(
    mask_func: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    pack_gqa: bool | None = None,
    build_backward: bool | None = None,
    _fwd_variant: _Sm100FwdVariant | None = None,
) -> MaskPlan:
    """Create a reusable architecture-native interval-mask plan.

    ``mask_func`` is a contiguous CUDA INT32 tensor shaped
    ``[Hmask, nfunc, total_q]``. ``Hmask`` is 1 or ``Hq`` and ``nfunc`` is
    positive and odd. For each query, endpoints encode visible sample-local
    key intervals as ``[0,F0) U [F1,F2) U ...``.

    Fixed-length Q/K/V use BSHD layouts. Supplying all four variable-length
    arguments switches to flattened THD layouts; the plan clones the prefix
    tensors and owns that geometry. ``build_backward=None`` builds backward
    payloads when autograd is enabled and any sample input requires gradients.
    """

    if pack_gqa is not None and type(pack_gqa) is not bool:
        raise TypeError("pack_gqa must be a bool or None")
    if build_backward is not None and type(build_backward) is not bool:
        raise TypeError("build_backward must be a bool or None")
    geometry = validate_create_mask_plan_inputs(
        mask_func,
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    if build_backward is None:
        build_backward = torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v))
    packed_plan = _build_packed_mask_plan(
        mask_func,
        q,
        k,
        v,
        cu_seqlens_q=geometry.cu_seqlens_q,
        cu_seqlens_k=geometry.cu_seqlens_k,
        max_seqlen_q=(geometry.max_seqlen_q if geometry.is_varlen else None),
        max_seqlen_k=(geometry.max_seqlen_k if geometry.is_varlen else None),
        pack_gqa=pack_gqa,
        build_backward=build_backward,
        _fwd_variant=_fwd_variant,
    )
    return MaskPlan(
        packed_plan=packed_plan,
        geometry=geometry,
        dtype=q.dtype,
        device=q.device,
    )


__all__ = ["create_mask_plan"]
