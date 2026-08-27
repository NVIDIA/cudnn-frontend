# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Large-tensor convolution regression tests.

Tests fprop, dgrad, and wgrad with large tensors and filters. FPROP and WGRAD
target C*R*S > 2^27 boundary coverage when the per-worker memory budget
permits; DGRAD remains conservatively runtime-bounded while exercising large
problems. All cases execute the kernel and compare against a PyTorch float32
reference.

Tune DEFAULT_NUM_TESTS_L0 / DEFAULT_NUM_TESTS_L1 to adjust local or downstream
runtime targets:
  L0: default smoke slice
  L1: expansion slice for longer runs

Developer diagnostic graph-engine filter examples:
  CUDNN_FUZZ_ENGINE_OP=fprop CUDNN_FUZZ_GRAPH_ENGINE_INDICES=0 pytest ...
  CUDNN_FUZZ_ENGINE_OP=fprop CUDNN_FUZZ_GRAPH_ENGINE_INDICES=0 \
    CUDNN_FUZZ_REGEN_ON_UNSUPPORTED=1 CUDNN_FUZZ_REGEN_ATTEMPTS=50 pytest ...

When regeneration advances past attempt 1, the pytest node ID still describes
the initial config; diagnostics identify the active regenerated candidate.
"""

import json
import math
import os
import random
import shlex
import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import cudnn
import pytest
import torch

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

_WORKER_MEM_FRACTION = 0.80  # fraction of (total_gpu / workers) to budget per test
_GRAPH_ENGINE_OP_ENV = "CUDNN_FUZZ_ENGINE_OP"  # graph-specific op selector: fprop/dgrad/wgrad
_GRAPH_ENGINE_INDICES_ENV = "CUDNN_FUZZ_GRAPH_ENGINE_INDICES"  # public operation-graph engine indices
_REGEN_ON_UNSUPPORTED_ENV = "CUDNN_FUZZ_REGEN_ON_UNSUPPORTED"  # retry unsupported engine picks
_REGEN_ATTEMPTS_ENV = "CUDNN_FUZZ_REGEN_ATTEMPTS"  # retry cap for regeneration
_RUNTIME_WORK_BUDGET_ENV = "CUDNN_FUZZ_RUNTIME_WORK_BUDGET"  # override generated test runtime-work cap
_NUM_TESTS_L0_ENV = "CUDNN_FUZZ_NUM_TESTS_L0"  # override L0 generated count
_NUM_TESTS_L1_ENV = "CUDNN_FUZZ_NUM_TESTS_L1"  # override L1 generated count
_REPRO_CONFIG_ENV = "CUDNN_FUZZ_REPRO_CONFIG"  # exact configuration JSON payload
_REPRO_FILE_ENV = "CUDNN_FUZZ_REPRO_FILE"  # path to exact configuration JSON payload
_EVENTS_DIR_ENV = "CUDNN_FUZZ_EVENTS_DIR"  # optional per-worker JSONL diagnostic output
_REPRO_SCHEMA_VERSION = 1
_EVENT_SCHEMA_VERSION = 1
_REPRO_MESSAGE_LIMIT = 4096
_EVENT_PREFIX = "CUDNN_FUZZ_EVENT"
_DEFAULT_RUNTIME_WORK_BUDGET = 100_000_000_000_000
_MAX_CONFIG_GENERATION_ATTEMPTS = 50
_DEFAULT_ENGINE_REGEN_ATTEMPTS = 50
_SPARSE_INTEGER_REDUCTION_THRESHOLD = 1 << 20  # switch very large reductions off dense random data
_SPARSE_INTEGER_TARGET_NONZERO = 32  # max nonzero filter values per output channel
_SPARSE_INTEGER_RNG_SALT = 0x51A25EED  # split sparse-index RNG from tensor-value RNG
_GRAPH_ENGINE_RNG_SALT = 0xE61A5EED  # split engine selection from config and tensor RNGs
_OP_SCHEDULE_RNG_SALT = 0x0F5C4ED  # split operation ordering from config generation
_NVRTC_COMPILATION_STATUS = "CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED"
_PLAN_BUILD_ERROR_LIMIT = 1024
_MISMATCH_SAMPLE_LIMIT = 8


class _EngineFilterNotSupported(Exception):
    """Selected graph engine is unsupported, not a cuDNN execution failure."""

    pass


@dataclass(frozen=True)
class _PlanIdentity:
    execution_plan_index: int
    graph_engine_index: int
    knob_choices: Tuple[Tuple[str, int], ...]

    def as_dict(self) -> dict:
        return {
            "execution_plan_index": self.execution_plan_index,
            "graph_engine_index": self.graph_engine_index,
            "knob_choices": dict(self.knob_choices),
        }


@dataclass(frozen=True)
class _PlanBuildSelection:
    selected_plan: Optional[_PlanIdentity]
    candidate_count: int
    nvrtc_failures: Tuple[str, ...]

    @property
    def selected_plan_index(self) -> Optional[int]:
        return self.selected_plan.execution_plan_index if self.selected_plan is not None else None


@dataclass(frozen=True)
class _CudnnRunResult:
    ok: bool
    message: str
    selected_plan: Optional[_PlanIdentity] = None
    nvrtc_failures: Tuple[str, ...] = ()

    @property
    def selected_plan_index(self) -> Optional[int]:
        return self.selected_plan.execution_plan_index if self.selected_plan is not None else None


def _pytest_worker_count() -> int:
    try:
        return max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))
    except ValueError:
        return 1


def _device_memory_budget(fraction: float = _WORKER_MEM_FRACTION) -> int:
    """Return per-worker tensor memory budget in bytes.

    Queries the GPU that will execute the tests so the generator never
    produces configs whose tensor footprint alone exceeds the device.
    cuDNN workspace is on top of this and is not predictable at generation
    time; OOM-on-workspace cases are caught at runtime and skipped.

    Falls back to a conservative 7 GiB default when CUDA is unavailable
    (e.g. dry-run collection on a CPU-only node).
    """
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        return int(total / _pytest_worker_count() * fraction)
    return 7 * (1 << 30)


def _graph_engine_indices() -> List[int]:
    """Parse the optional comma-separated public graph engine indices."""
    raw = os.environ.get(_GRAPH_ENGINE_INDICES_ENV, "").strip()
    op = os.environ.get(_GRAPH_ENGINE_OP_ENV, "").strip()
    if raw and not op:
        raise ValueError(f"{_GRAPH_ENGINE_OP_ENV} must be set when {_GRAPH_ENGINE_INDICES_ENV} is set")
    if not raw:
        return []

    try:
        indices = [int(v.strip()) for v in raw.split(",") if v.strip()]
    except ValueError as e:
        raise ValueError(f"{_GRAPH_ENGINE_INDICES_ENV} must be a comma-separated list of graph engine indices") from e
    if not indices:
        raise ValueError(f"{_GRAPH_ENGINE_INDICES_ENV} must contain at least one graph engine index")
    duplicates = [value for value, count in Counter(indices).items() if count > 1]
    if duplicates:
        raise ValueError(f"{_GRAPH_ENGINE_INDICES_ENV} contains duplicate graph engine indices: {duplicates}")
    return indices


def _graph_engine_filter_op() -> Optional["ConvType"]:
    """Parse the convolution op paired with the graph engine index filter."""
    raw = os.environ.get(_GRAPH_ENGINE_OP_ENV, "").strip().lower()
    indices = os.environ.get(_GRAPH_ENGINE_INDICES_ENV, "").strip()
    if indices and not raw:
        raise ValueError(f"{_GRAPH_ENGINE_OP_ENV} must be set when {_GRAPH_ENGINE_INDICES_ENV} is set")
    if not raw:
        return None

    try:
        return _CONV_TYPE_ALIASES[raw]
    except KeyError as e:
        raise ValueError(f"{_GRAPH_ENGINE_OP_ENV} must be one of fprop, dgrad, or wgrad") from e


def _env_flag(name: str) -> bool:
    """Interpret common boolean env-var values."""
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("", "0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be one of 1, true, yes, on, 0, false, no, or off")


def _regen_on_unsupported() -> bool:
    """Return whether graph-engine filter runs should retry unsupported configs."""
    enabled = _env_flag(_REGEN_ON_UNSUPPORTED_ENV)
    if enabled and not _graph_engine_indices():
        raise ValueError(f"{_REGEN_ON_UNSUPPORTED_ENV} requires {_GRAPH_ENGINE_OP_ENV} and " f"{_GRAPH_ENGINE_INDICES_ENV}")
    return enabled


def _regen_attempts() -> int:
    """Return the retry cap for regenerating unsupported configs."""
    raw = os.environ.get(_REGEN_ATTEMPTS_ENV, "").strip()
    if not raw:
        return _DEFAULT_ENGINE_REGEN_ATTEMPTS

    try:
        return max(1, int(raw))
    except ValueError as e:
        raise ValueError(f"{_REGEN_ATTEMPTS_ENV} must be a positive integer") from e


def _runtime_work_budget() -> int:
    """Return the per-config conservative runtime-work budget."""
    raw = os.environ.get(_RUNTIME_WORK_BUDGET_ENV, "").strip()
    if not raw:
        return _DEFAULT_RUNTIME_WORK_BUDGET

    try:
        value = int(float(raw))
    except ValueError as e:
        raise ValueError(f"{_RUNTIME_WORK_BUDGET_ENV} must be a positive number") from e

    if value <= 0:
        raise ValueError(f"{_RUNTIME_WORK_BUDGET_ENV} must be a positive number")
    return value


def _num_tests(env_name: str, default: int) -> int:
    """Return a generated test count, optionally overridden by an env var."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"{env_name} must be a positive integer") from e

    if value <= 0:
        raise ValueError(f"{env_name} must be a positive integer")
    return value


def _kernel_cfg_choices(graph, engine_index: int) -> List[dict]:
    """Expand KERNEL_CFG knob values for one public graph engine index."""
    knobs = graph.get_knobs_for_engine(engine_index)
    kernel_cfg_knobs = [knob for knob in knobs if knob.type == cudnn.knob_type.KERNEL_CFG]
    if not kernel_cfg_knobs:
        return [{}]
    if len(kernel_cfg_knobs) > 1:
        detail = "an execution plan can represent only one value per knob type"
        raise RuntimeError(f"graph engine index {engine_index} reports multiple KERNEL_CFG knobs, but {detail}")

    knob = kernel_cfg_knobs[0]
    if knob.stride <= 0:
        raise RuntimeError(f"graph engine index {engine_index} reports invalid KERNEL_CFG stride {knob.stride}")
    return [{knob.type: value} for value in range(knob.min_value, knob.max_value + 1, knob.stride)]


def _validate_graph_engine_indices(graph, graph_engine_indices: List[int]) -> None:
    """Validate requested public indices against this operation graph."""
    engine_count = graph.get_engine_count()
    invalid_indices = [index for index in graph_engine_indices if index < 0 or index >= engine_count]
    if invalid_indices:
        valid_range = f"0..{engine_count - 1}" if engine_count else "empty because the graph reports no engines"
        invalid = ",".join(str(index) for index in invalid_indices)
        noun = "index" if len(invalid_indices) == 1 else "indices"
        raise ValueError(f"graph engine {noun} {invalid} is outside valid range {valid_range}")


def _select_graph_engine_index(cfg: "LargeTensorConfig", graph_engine_indices: List[int]) -> int:
    """Select one requested engine deterministically for this testcase."""
    if not graph_engine_indices:
        raise ValueError("Cannot select a graph engine from an empty index list")
    rng = random.Random(cfg.rng_seed ^ _GRAPH_ENGINE_RNG_SALT)
    return rng.choice(graph_engine_indices)


def _create_filtered_execution_plans(graph, graph_engine_index: int) -> None:
    """Create knob-derived plans for one selected public graph engine index."""
    created_count = 0

    try:
        knob_choices = _kernel_cfg_choices(graph, graph_engine_index)
    except cudnn.cudnnGraphNotSupportedError as e:
        raise _EngineFilterNotSupported(f"graph engine index {graph_engine_index} knobs unavailable: {e}") from e

    failures = []
    for knob_choice in knob_choices:
        try:
            graph.create_execution_plan(graph_engine_index, knob_choice)
            created_count += 1
        except cudnn.cudnnGraphNotSupportedError as e:
            failures.append(f"knob choice {knob_choice} rejected: {e}")

    if created_count == 0:
        details = "; ".join(failures[:3]) if failures else "no candidate plans created"
        raise _EngineFilterNotSupported(f"graph engine index {graph_engine_index} created no execution plans ({details})")


MEMORY_BUDGET_BYTES = _device_memory_budget()
RUNTIME_WORK_BUDGET = _runtime_work_budget()
WORKSPACE_OVERHEAD = 0.15  # fraction added for cuDNN workspace + allocator slack
DEFAULT_NUM_TESTS_L0 = _num_tests(_NUM_TESTS_L0_ENV, 64)  # default smoke slice
DEFAULT_NUM_TESTS_L1 = _num_tests(_NUM_TESTS_L1_ENV, 448)  # longer expansion slice
DEFAULT_SEED_L0 = 42
DEFAULT_SEED_L1 = 12345
INT32_MAX = (1 << 31) - 1  # 2_147_483_647
_WORKSPACE_POISON_DIVISOR = 1000
_MAX_WORKSPACE_POISON_COUNT = 1_000_000
_WORKSPACE_POISON_BYTE = 0xFF

# Baseline accumulation depth and rtol before sqrt(accum) scaling.
BASE_ACCUM = 128
_STANDARD_RTOL_AT_BASE_ACCUM = 1e-2
_FP32_GRAD_ATOL_AT_BASE_ACCUM = 1.5e-2
_FP32_FPROP_ATOL_AT_BASE_ACCUM = 3e-2
_MAX_DENSE_RANDOM_RTOL = 0.10

# Integer-policy inputs use fixed dtype-level bounds rather than reduction scaling.
# FP32 keeps a small absolute floor for fused bias/ReLU rounding near zero.
_INTEGER_DATA_TOLERANCES = {
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float32: (1.3e-6, 1e-3),
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConvType(Enum):
    FPROP = auto()
    DGRAD = auto()
    WGRAD = auto()


_CONV_TYPE_ALIASES = {
    "fprop": ConvType.FPROP,
    "fp": ConvType.FPROP,
    "dgrad": ConvType.DGRAD,
    "dg": ConvType.DGRAD,
    "wgrad": ConvType.WGRAD,
    "wg": ConvType.WGRAD,
}


class ShapeFamily(Enum):
    NARROW = "nar"  # tiny input spatial, large filter -> maximises C*R*S vs memory
    BALANCED = "bal"  # H ~ W ~ R ~ S, typical real-world shapes
    DOWNSAMPLE = "dwn"  # stride > 1 with non-trivial output spatial
    BATCHED = "bat"  # N > 1, valid conv (exercises reduction-axis indexing)
    # Input strictly larger than filter, producing output spatial > 1x1 with
    # pad=0 stride=1. Distinguishes from the families where input is sized
    # exactly to make output 1x1. Use a compact value in generated pytest IDs.
    NON_UNIT_OUTPUT = "nvc"
    RANDOM = "rnd"  # uniform random within budget


_TEST_ID_DTYPE = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
}
_TEST_ID_CONV_TYPE = {
    ConvType.FPROP: "fp",
    ConvType.DGRAD: "dg",
    ConvType.WGRAD: "wg",
}

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class LargeTensorConfig:
    conv_type: ConvType
    spatial_dims: int
    dtype: torch.dtype
    shape_family: ShapeFamily
    n: int
    k: int
    c: int
    input_spatial: List[int]  # [H, W] or [D, H, W]
    filter_spatial: List[int]  # [R, S] or [T, R, S]
    padding: List[int]
    stride: List[int]
    dilation: List[int]
    epilogue: str = "none"  # "none" | "relu" | "bias_relu"  (FPROP only)
    rng_seed: int = 0

    @property
    def x_shape(self) -> List[int]:
        return [self.n, self.c] + list(self.input_spatial)

    @property
    def w_shape(self) -> List[int]:
        return [self.k, self.c] + list(self.filter_spatial)

    @property
    def y_shape(self) -> List[int]:
        out = []
        for inp, flt, pad, st, dil in zip(self.input_spatial, self.filter_spatial, self.padding, self.stride, self.dilation):
            eff = dil * (flt - 1) + 1
            out.append((inp + 2 * pad - eff) // st + 1)
        return [self.n, self.k] + out

    @property
    def filter_elements(self) -> int:
        return math.prod(self.w_shape)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cudnn_dtype(t: torch.dtype):
    dtype_map = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
    }
    if t not in dtype_map:
        raise ValueError(f"Unsupported tensor dtype: {t}")
    return dtype_map[t]


def _estimate_bytes(cfg: LargeTensorConfig) -> int:
    """Conservative peak GPU bytes for one test's tensor/reference path."""
    elem = 2 if cfg.dtype in (torch.float16, torch.bfloat16) else 4
    x = math.prod(cfg.x_shape)
    w = math.prod(cfg.w_shape)
    y = math.prod(cfg.y_shape)

    base = (x + w + y) * elem

    if cfg.conv_type == ConvType.FPROP:
        actual = y * elem
        reference = x * 4 + w * 4 + y * 4
        compare = y * 4
    elif cfg.conv_type == ConvType.DGRAD:
        actual = x * elem
        # Reference keeps dYf, Wf, dX, fwd, dX.grad, and the returned clone.
        reference = y * 4 + w * 4 + x * 4 + y * 4 + x * 4 + x * 4
        compare = x * 4
    else:  # WGRAD
        actual = w * elem
        # Reference keeps Xf, dYf, dW, fwd, dW.grad, and the returned clone.
        reference = x * 4 + y * 4 + w * 4 + y * 4 + w * 4 + w * 4
        compare = w * 4

    total = base + actual + reference + compare
    return int(total * (1.0 + WORKSPACE_OVERHEAD))


# The default cap mainly filters huge-filter DGRAD cases, which can be slow even with small outputs.
def _estimate_runtime_work(cfg: LargeTensorConfig) -> int:
    """Estimate generated-config work for runtime budgeting."""
    input_spatial = math.prod(cfg.input_spatial)
    filter_spatial = math.prod(cfg.filter_spatial)
    output_spatial = math.prod(cfg.y_shape[2:])

    if cfg.conv_type == ConvType.FPROP:
        return 2 * cfg.n * cfg.k * output_spatial * cfg.c * filter_spatial
    if cfg.conv_type == ConvType.DGRAD:
        return 2 * cfg.n * cfg.c * input_spatial * cfg.k * filter_spatial
    return 2 * cfg.k * cfg.c * filter_spatial * cfg.n * output_spatial


def _fprop_reduction_size(cfg: LargeTensorConfig) -> int:
    """Return the FPROP per-output convolution accumulation depth."""
    return cfg.c * math.prod(cfg.filter_spatial)


def _effective_reduction_size(cfg: LargeTensorConfig) -> int:
    """Return a useful per-output accumulation estimate for diagnostics.

    FPROP reduces across C * filter_spatial. DGRAD reduces across K and the
    contributing dY positions, while WGRAD reduces across N * output_spatial.
    Keep this separate from the large-filter data-policy trigger.
    """
    output_spatial = math.prod(cfg.y_shape[2:])

    if cfg.conv_type == ConvType.FPROP:
        return _fprop_reduction_size(cfg)
    if cfg.conv_type == ConvType.DGRAD:
        return cfg.k * output_spatial
    return cfg.n * output_spatial


def _uses_integer_data(cfg: LargeTensorConfig) -> bool:
    """Return whether this config should avoid dense random large reductions.

    The policy is keyed to the FPROP-style C * filter_spatial reduction. For
    DGRAD/WGRAD, it is a large-filter policy rather than the operation's exact
    per-output accumulation depth.
    """
    return _fprop_reduction_size(cfg) >= _SPARSE_INTEGER_REDUCTION_THRESHOLD


def _uses_sparse_filter(cfg: LargeTensorConfig) -> bool:
    """Return whether W is a sparse integer input, not an output prefill."""
    return _uses_integer_data(cfg) and cfg.conv_type in (ConvType.FPROP, ConvType.DGRAD)


def _data_policy_name(cfg: LargeTensorConfig) -> str:
    if not _uses_integer_data(cfg):
        return "dense_random"
    if _uses_sparse_filter(cfg):
        return "sparse_filter_integer"
    return "dense_integer"


def _tolerance_policy_name(cfg: LargeTensorConfig) -> str:
    return "integer_dtype_fixed" if _uses_integer_data(cfg) else "dense_random_scaled"


def _comparison_context(cfg: LargeTensorConfig, rtol: float, atol: float) -> str:
    return (
        f"data_policy={_data_policy_name(cfg)}, "
        f"tolerance_policy={_tolerance_policy_name(cfg)}, "
        f"integer_data={_uses_integer_data(cfg)}, "
        f"sparse_filter={_uses_sparse_filter(cfg)}, "
        f"fprop_reduction={_fprop_reduction_size(cfg)}, "
        f"effective_reduction={_effective_reduction_size(cfg)}, "
        f"sparse_threshold={_SPARSE_INTEGER_REDUCTION_THRESHOLD}, "
        f"sparse_target_nonzero={_SPARSE_INTEGER_TARGET_NONZERO}, "
        f"rtol={rtol}, atol={atol}"
    )


def _tolerances(cfg: LargeTensorConfig) -> Tuple[float, float]:
    """Return comparison tolerances for the selected data policy.

    Integer-policy inputs use fixed per-dtype bounds. Dense random inputs scale
    with the operation's effective reduction, with relative error capped at 10%.
    FP32 operations use operation-specific absolute margins for cancellation
    near zero.
    """
    if _uses_integer_data(cfg):
        return _INTEGER_DATA_TOLERANCES[cfg.dtype]

    accum = _effective_reduction_size(cfg)
    scale = max(1.0, math.sqrt(accum / BASE_ACCUM))
    std_tol = _STANDARD_RTOL_AT_BASE_ACCUM * scale
    if cfg.dtype != torch.float32:
        atol_base = _STANDARD_RTOL_AT_BASE_ACCUM
    elif cfg.conv_type == ConvType.FPROP:
        atol_base = _FP32_FPROP_ATOL_AT_BASE_ACCUM
    else:
        atol_base = _FP32_GRAD_ATOL_AT_BASE_ACCUM
    atol = atol_base * scale
    rtol = min(std_tol, _MAX_DENSE_RANDOM_RTOL)
    return rtol, atol


def _test_id(param, prefix="lt") -> str:
    """Encode the generated conv problem into the pytest parameter name."""
    test_num, _, _, cfg = param
    dtype_s = _TEST_ID_DTYPE[cfg.dtype]
    conv_s = _TEST_ID_CONV_TYPE[cfg.conv_type]
    flt_s = "x".join(str(v) for v in cfg.filter_spatial)
    dil_s = "_dil" + "x".join(str(d) for d in cfg.dilation) if any(d != 1 for d in cfg.dilation) else ""
    stride_s = "_s" + "x".join(str(s) for s in cfg.stride) if any(s != 1 for s in cfg.stride) else ""
    epi_s = f"_{cfg.epilogue}" if cfg.epilogue != "none" else ""
    return (
        f"{prefix}{test_num}"
        f"_N{cfg.n}_C{cfg.c}K{cfg.k}"
        f"_R{flt_s}"
        f"_{dtype_s}_{cfg.spatial_dims}d_{conv_s}_{cfg.shape_family.value}{stride_s}{dil_s}{epi_s}"
    )


def _poison_workspace(workspace: torch.Tensor) -> None:
    """Initialize workspace and sprinkle poison bytes into random locations."""
    ws_size = workspace.numel()
    if ws_size == 0:
        return

    workspace.random_(0, 256)
    poison_count = min(
        ws_size,
        max(1, ws_size // _WORKSPACE_POISON_DIVISOR),
        _MAX_WORKSPACE_POISON_COUNT,
    )
    poison_indices = torch.randint(ws_size, (poison_count,), device=workspace.device)
    workspace[poison_indices] = _WORKSPACE_POISON_BYTE


_DTYPE_TO_REPRO_NAME = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
_REPRO_NAME_TO_DTYPE = {
    "float16": torch.float16,
    "torch.float16": torch.float16,
    "half": torch.float16,
    "fp16": torch.float16,
    "f16": torch.float16,
    "bfloat16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "torch.float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "f32": torch.float32,
}


def _config_to_repro_config(cfg: LargeTensorConfig) -> dict:
    """Serialize a config into stable JSON-compatible fields."""
    return {
        "conv_type": cfg.conv_type.name.lower(),
        "spatial_dims": cfg.spatial_dims,
        "dtype": _DTYPE_TO_REPRO_NAME[cfg.dtype],
        "shape_family": cfg.shape_family.name.lower(),
        "n": cfg.n,
        "k": cfg.k,
        "c": cfg.c,
        "input_spatial": list(cfg.input_spatial),
        "filter_spatial": list(cfg.filter_spatial),
        "padding": list(cfg.padding),
        "stride": list(cfg.stride),
        "dilation": list(cfg.dilation),
        "epilogue": cfg.epilogue,
        "rng_seed": cfg.rng_seed,
    }


def _parse_conv_type(value) -> ConvType:
    raw = str(value).strip().lower().removeprefix("convtype.")
    if raw in _CONV_TYPE_ALIASES:
        return _CONV_TYPE_ALIASES[raw]
    raise ValueError(f"Unsupported conv_type in repro config: {value!r}")


def _parse_shape_family(value) -> ShapeFamily:
    raw = str(value).strip().lower().removeprefix("shapefamily.")
    aliases = {}
    for family in ShapeFamily:
        aliases[family.name.lower()] = family
        aliases[family.value] = family
    if raw in aliases:
        return aliases[raw]
    raise ValueError(f"Unsupported shape_family in repro config: {value!r}")


def _parse_dtype(value) -> torch.dtype:
    raw = str(value).strip().lower()
    if raw in _REPRO_NAME_TO_DTYPE:
        return _REPRO_NAME_TO_DTYPE[raw]
    raise ValueError(f"Unsupported dtype in repro config: {value!r}")


def _int_list(config: dict, key: str, length: int) -> List[int]:
    try:
        raw_values = config[key]
    except KeyError as e:
        raise ValueError(f"Missing {key!r} in repro config") from e

    if not isinstance(raw_values, list):
        raise ValueError(f"{key!r} in repro config must be a list")
    try:
        values = [int(v) for v in raw_values]
    except (TypeError, ValueError) as e:
        raise ValueError(f"{key!r} in repro config must contain integers") from e

    if len(values) != length:
        raise ValueError(f"{key!r} length {len(values)} does not match spatial_dims={length}")
    return values


def _parse_spatial_dims(config: dict) -> int:
    try:
        spatial_dims = int(config["spatial_dims"])
    except KeyError as e:
        raise ValueError("Missing 'spatial_dims' in repro config") from e
    except (TypeError, ValueError) as e:
        raise ValueError("'spatial_dims' in repro config must be 2 or 3") from e

    if spatial_dims not in (2, 3):
        raise ValueError("'spatial_dims' in repro config must be 2 or 3")
    return spatial_dims


def _config_from_repro_payload(payload: dict) -> LargeTensorConfig:
    """Parse either the full emitted repro payload or just its config object."""
    if not isinstance(payload, dict):
        raise ValueError("Repro payload must be a JSON object")

    if "config" in payload:
        schema = payload.get("schema")
        if schema != _REPRO_SCHEMA_VERSION:
            raise ValueError(f"Unsupported repro schema {schema!r}; expected {_REPRO_SCHEMA_VERSION}")
        config = payload["config"]
    else:
        config = payload

    if not isinstance(config, dict):
        raise ValueError("Repro payload 'config' must be a JSON object")

    try:
        spatial_dims = _parse_spatial_dims(config)
        epilogue = str(config.get("epilogue", "none"))
        if epilogue not in ("none", "relu", "bias_relu"):
            raise ValueError(f"Unsupported epilogue in repro config: {epilogue!r}")
        return LargeTensorConfig(
            conv_type=_parse_conv_type(config["conv_type"]),
            spatial_dims=spatial_dims,
            dtype=_parse_dtype(config["dtype"]),
            shape_family=_parse_shape_family(config["shape_family"]),
            n=int(config["n"]),
            k=int(config["k"]),
            c=int(config["c"]),
            input_spatial=_int_list(config, "input_spatial", spatial_dims),
            filter_spatial=_int_list(config, "filter_spatial", spatial_dims),
            padding=_int_list(config, "padding", spatial_dims),
            stride=_int_list(config, "stride", spatial_dims),
            dilation=_int_list(config, "dilation", spatial_dims),
            epilogue=epilogue,
            rng_seed=int(config["rng_seed"]),
        )
    except KeyError as e:
        raise ValueError(f"Missing {e.args[0]!r} in repro config") from e


def _load_repro_payload() -> Optional[dict]:
    raw = os.environ.get(_REPRO_CONFIG_ENV, "").strip()
    source = _REPRO_CONFIG_ENV
    if not raw:
        repro_file = os.environ.get(_REPRO_FILE_ENV, "").strip()
        if not repro_file:
            return None
        source = f"{_REPRO_FILE_ENV}={repro_file}"
        try:
            with open(repro_file, "r", encoding="utf-8") as f:
                raw = f.read()
        except OSError as e:
            raise ValueError(f"Could not read repro file {repro_file!r}: {e}") from e

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{source} must contain valid JSON: {e}") from e


def _load_repro_config() -> Optional[LargeTensorConfig]:
    payload = _load_repro_payload()
    if payload is None:
        return None
    return _config_from_repro_payload(payload)


def _trim_repro_message(message: Optional[str]) -> Optional[str]:
    if message is None or len(message) <= _REPRO_MESSAGE_LIMIT:
        return message
    dropped = len(message) - _REPRO_MESSAGE_LIMIT
    return f"{message[:_REPRO_MESSAGE_LIMIT]}... <truncated {dropped} chars>"


def _repro_payload(
    cfg: LargeTensorConfig,
    *,
    test_num: Optional[int] = None,
    total_tests: Optional[int] = None,
    config_seed: Optional[int] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    message: Optional[str] = None,
    attempt: Optional[int] = None,
    selected_plan: Optional[_PlanIdentity] = None,
) -> dict:
    test_id = None
    if test_num is not None and total_tests is not None and config_seed is not None:
        test_id = _test_id((test_num, total_tests, config_seed, cfg))
    graph_engine_op = _FORCED_CONV_TYPE
    graph_engine_indices = _COLLECTED_GRAPH_ENGINE_INDICES
    requested_graph_engine_index = _select_graph_engine_index(cfg, graph_engine_indices) if graph_engine_indices else None

    metadata = {
        "test_id": test_id,
        "test_num": test_num,
        "total_tests": total_tests,
        "config_seed": config_seed,
        "attempt": attempt,
        "message": _trim_repro_message(message),
        "x_shape": cfg.x_shape,
        "w_shape": cfg.w_shape,
        "y_shape": cfg.y_shape,
        "memory_budget_bytes": MEMORY_BUDGET_BYTES,
        "worker_count": _pytest_worker_count(),
        "runtime_work_budget": RUNTIME_WORK_BUDGET,
        "estimated_bytes": _estimate_bytes(cfg),
        "estimated_runtime_work": _estimate_runtime_work(cfg),
        "fprop_reduction": _fprop_reduction_size(cfg),
        "effective_reduction": _effective_reduction_size(cfg),
        "sparse_threshold": _SPARSE_INTEGER_REDUCTION_THRESHOLD,
        "sparse_target_nonzero": _SPARSE_INTEGER_TARGET_NONZERO,
        "integer_data": _uses_integer_data(cfg),
        "sparse_filter": _uses_sparse_filter(cfg),
        "data_policy": _data_policy_name(cfg),
        "tolerance_policy": _tolerance_policy_name(cfg),
        "rtol": rtol,
        "atol": atol,
        "graph_engine_op": graph_engine_op.name.lower() if graph_engine_op is not None else None,
        "requested_graph_engine_indices": graph_engine_indices,
        "requested_graph_engine_index": requested_graph_engine_index,
        "execution_plan_index": selected_plan.execution_plan_index if selected_plan is not None else None,
        "graph_engine_index": selected_plan.graph_engine_index if selected_plan is not None else None,
        "knob_choices": dict(selected_plan.knob_choices) if selected_plan is not None else None,
    }
    return {
        "schema": _REPRO_SCHEMA_VERSION,
        "config": _config_to_repro_config(cfg),
        "metadata": metadata,
    }


def _repro_json(payload: dict, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(payload, indent=2, sort_keys=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _emit_diagnostic_event(event_type: str, payload: dict, *, human_message: Optional[str] = None) -> None:
    event = {
        "schema": _EVENT_SCHEMA_VERSION,
        "event": event_type,
        "payload": payload,
    }
    encoded = json.dumps(event, sort_keys=True, separators=(",", ":"))
    if human_message is not None:
        print(human_message, flush=True)
    print(f"{_EVENT_PREFIX} {encoded}", flush=True)

    events_dir = os.environ.get(_EVENTS_DIR_ENV, "").strip()
    if events_dir:
        os.makedirs(events_dir, exist_ok=True)
        worker = os.environ.get("PYTEST_XDIST_WORKER", f"pid{os.getpid()}")
        events_path = os.path.join(events_dir, f"events_{worker}.jsonl")
        with open(events_path, "a", encoding="ascii") as events_file:
            events_file.write(encoded + "\n")


def _emit_outcome_event(
    event_type: str,
    status: str,
    cfg: LargeTensorConfig,
    *,
    test_num: Optional[int] = None,
    total_tests: Optional[int] = None,
    config_seed: Optional[int] = None,
    message: Optional[str] = None,
    attempt: Optional[int] = None,
) -> None:
    rtol, atol = _tolerances(cfg)
    repro = _repro_payload(
        cfg,
        test_num=test_num,
        total_tests=total_tests,
        config_seed=config_seed,
        rtol=rtol,
        atol=atol,
        message=message,
        attempt=attempt,
    )
    _emit_diagnostic_event(event_type, {"status": status, "repro": repro})


def _repro_command(payload: dict) -> str:
    metadata = payload.get("metadata", {})
    graph_engine_indices = metadata.get("requested_graph_engine_indices", [])
    graph_filter = ""
    if graph_engine_indices:
        graph_engine_op = metadata["graph_engine_op"]
        indices = ",".join(str(index) for index in graph_engine_indices)
        graph_filter = f"{_GRAPH_ENGINE_OP_ENV}={shlex.quote(graph_engine_op)} " f"{_GRAPH_ENGINE_INDICES_ENV}={shlex.quote(indices)} "
    return (
        f"{graph_filter}{_REPRO_CONFIG_ENV}={shlex.quote(_repro_json(payload))} "
        f"{shlex.quote(sys.executable)} -m pytest test/python/test_conv_large_tensor_fuzzer.py "
        "-o addopts= -k test_conv_large_tensor_repro -s"
    )


def _format_repro_context(
    cfg: LargeTensorConfig,
    *,
    test_num: Optional[int] = None,
    total_tests: Optional[int] = None,
    config_seed: Optional[int] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    message: Optional[str] = None,
    attempt: Optional[int] = None,
    selected_plan: Optional[_PlanIdentity] = None,
) -> str:
    if rtol is None or atol is None:
        rtol, atol = _tolerances(cfg)
    payload = _repro_payload(
        cfg,
        test_num=test_num,
        total_tests=total_tests,
        config_seed=config_seed,
        rtol=rtol,
        atol=atol,
        message=message,
        attempt=attempt,
        selected_plan=selected_plan,
    )
    return (
        "\n\nLarge tensor fuzzer repro command:\n"
        f"{_repro_command(payload)}\n\n"
        "Large tensor fuzzer configuration JSON:\n"
        f"{_repro_json(payload, pretty=True)}\n\n"
        "Large tensor fuzzer context: "
        f"{_comparison_context(cfg, rtol, atol)}"
    )


# ---------------------------------------------------------------------------
# Config generator
# ---------------------------------------------------------------------------


class LargeTensorConfigGenerator:
    """Generates large-tensor conv configs biased toward large C*R*S."""

    _SHAPE_FAMILIES = [
        ShapeFamily.NARROW,
        ShapeFamily.BALANCED,
        ShapeFamily.DOWNSAMPLE,
        ShapeFamily.BATCHED,
        ShapeFamily.NON_UNIT_OUTPUT,
        ShapeFamily.RANDOM,
    ]
    _SHAPE_FAMILY_WEIGHTS = [0.25, 0.17, 0.17, 0.17, 0.17, 0.07]
    _OPS = [ConvType.FPROP, ConvType.DGRAD, ConvType.WGRAD]
    _OP_WEIGHTS = [0.50, 0.25, 0.25]
    _DTYPES = [torch.float16, torch.bfloat16, torch.float32]
    _POW2 = [1, 2, 4, 8, 16, 32, 64]

    def __init__(
        self,
        seed: int,
        allow_unaligned: bool = False,
        include_extras: bool = False,
        forced_conv_type: Optional[ConvType] = None,
        force_plain_fprop: bool = False,
    ):
        self.rng = random.Random(seed)
        self.allow_unaligned = allow_unaligned
        self.include_extras = include_extras
        self.forced_conv_type = forced_conv_type
        self.force_plain_fprop = force_plain_fprop

    def _ch(self, lo: int, hi: int) -> int:
        """Pick a channel count; aligned (power of 2) for L0, arbitrary for L1."""
        if self.allow_unaligned:
            return self.rng.randint(lo, hi)
        opts = [v for v in self._POW2 if lo <= v <= hi]
        return self.rng.choice(opts) if opts else lo

    def _extras(self, sdims: int) -> Tuple[List[int], str]:
        if not self.include_extras:
            return [1] * sdims, "none"
        dil = [self.rng.choice([1, 2])] * sdims
        epi = self.rng.choice(["none", "relu", "bias_relu"])
        return dil, epi

    def _raw2d(self, shape_family: ShapeFamily) -> dict:
        """Pick 2D conv knobs for one shape family before deriving input size."""
        rng = self.rng
        dil, epi = self._extras(2)
        # Defaults; per-shape-family branches override.
        n = 1
        stride = [1, 1]
        out_spatial = None  # None -> output 1x1 (valid conv)

        if shape_family == ShapeFamily.NARROW:
            # Large asymmetric filters can dominate DGRAD runtime. Bound R and S
            # here, then use C to preserve the target C*R*S coverage.
            r, s = rng.randint(512, 2048), rng.randint(512, 2048)
            c, k = self._ch(8, 64), self._ch(1, 4)
        elif shape_family == ShapeFamily.BALANCED:
            dim = rng.randint(256, 2048)
            r = s = dim
            c, k = self._ch(8, 64), self._ch(1, 4)
        elif shape_family == ShapeFamily.DOWNSAMPLE:
            # stride > 1 with non-trivial output. Filter ranges shrunk so total
            # ref compute (~ K*C*R*S * out_spatial^2) stays tractable.
            r, s = rng.randint(256, 1024), rng.randint(256, 1024)
            c, k = self._ch(8, 32), self._ch(1, 4)
            st = rng.choice([2, 4])
            stride = [st, st]
            out_spatial = rng.choice([4, 8])
        elif shape_family == ShapeFamily.BATCHED:
            # N > 1 to exercise reduction-axis indexing in B_offset.
            r, s = rng.randint(256, 1024), rng.randint(256, 1024)
            c, k = self._ch(8, 32), self._ch(1, 4)
            n = rng.choice([2, 4])
        elif shape_family == ShapeFamily.NON_UNIT_OUTPUT:
            # Output > 1x1 with stride=1; smaller filters so K*C*R*S * out^2 fits.
            r, s = rng.randint(128, 768), rng.randint(128, 768)
            c, k = self._ch(8, 32), self._ch(1, 4)
            out_spatial = rng.choice([4, 8, 16])
        else:  # RANDOM
            r, s = rng.randint(64, 2048), rng.randint(64, 2048)
            c, k = self._ch(1, 16), self._ch(1, 4)

        # Input spatial chosen so output spatial matches out_spatial (or 1 if None).
        out_h = 1 if out_spatial is None else out_spatial
        out_w = 1 if out_spatial is None else out_spatial
        h = stride[0] * (out_h - 1) + dil[0] * (r - 1) + 1
        w = stride[1] * (out_w - 1) + dil[1] * (s - 1) + 1
        return dict(n=n, k=k, c=c, input_spatial=[h, w], filter_spatial=[r, s], padding=[0, 0], stride=stride, dilation=dil, epilogue=epi)

    def _raw3d(self, shape_family: ShapeFamily) -> dict:
        """Pick 3D conv knobs for one shape family before deriving input size."""
        rng = self.rng
        dil, epi = self._extras(3)
        n = 1
        stride = [1, 1, 1]
        out_spatial = None

        if shape_family == ShapeFamily.NARROW:
            t, r, s = rng.randint(64, 256), rng.randint(64, 256), rng.randint(64, 256)
            c, k = self._ch(8, 64), self._ch(1, 2)
        elif shape_family == ShapeFamily.BALANCED:
            dim = rng.randint(32, 128)
            t = r = s = dim
            c, k = self._ch(4, 32), self._ch(1, 4)
        elif shape_family == ShapeFamily.DOWNSAMPLE:
            t, r, s = rng.randint(32, 96), rng.randint(32, 96), rng.randint(32, 96)
            c, k = self._ch(4, 32), self._ch(1, 2)
            st = rng.choice([2, 4])
            stride = [st, st, st]
            out_spatial = rng.choice([2, 4])
        elif shape_family == ShapeFamily.BATCHED:
            t, r, s = rng.randint(32, 96), rng.randint(32, 96), rng.randint(32, 96)
            c, k = self._ch(4, 16), self._ch(1, 2)
            n = rng.choice([2, 4])
        elif shape_family == ShapeFamily.NON_UNIT_OUTPUT:
            t, r, s = rng.randint(16, 64), rng.randint(16, 64), rng.randint(16, 64)
            c, k = self._ch(4, 16), self._ch(1, 2)
            out_spatial = rng.choice([2, 4])
        else:  # RANDOM
            t, r, s = rng.randint(16, 128), rng.randint(16, 128), rng.randint(16, 128)
            c, k = self._ch(4, 32), self._ch(1, 4)

        out_d = 1 if out_spatial is None else out_spatial
        out_h = 1 if out_spatial is None else out_spatial
        out_w = 1 if out_spatial is None else out_spatial
        d = stride[0] * (out_d - 1) + dil[0] * (t - 1) + 1
        h = stride[1] * (out_h - 1) + dil[1] * (r - 1) + 1
        w = stride[2] * (out_w - 1) + dil[2] * (s - 1) + 1
        return dict(n=n, k=k, c=c, input_spatial=[d, h, w], filter_spatial=[t, r, s], padding=[0, 0, 0], stride=stride, dilation=dil, epilogue=epi)

    def generate(self) -> LargeTensorConfig:
        rng = self.rng
        conv_type = self.forced_conv_type
        if conv_type is None:
            conv_type = rng.choices(self._OPS, weights=self._OP_WEIGHTS)[0]

        for _ in range(_MAX_CONFIG_GENERATION_ATTEMPTS):
            dtype = rng.choice(self._DTYPES)
            shape_family = rng.choices(self._SHAPE_FAMILIES, weights=self._SHAPE_FAMILY_WEIGHTS)[0]
            sdims = rng.choices([2, 3], weights=[0.75, 0.25])[0]

            raw = self._raw2d(shape_family) if sdims == 2 else self._raw3d(shape_family)

            # Epilogues only apply to FPROP, and engine-index filtering targets
            # the plain convolution operation graph.
            if conv_type != ConvType.FPROP or self.force_plain_fprop:
                raw["epilogue"] = "none"

            cfg = LargeTensorConfig(
                conv_type=conv_type,
                spatial_dims=sdims,
                dtype=dtype,
                shape_family=shape_family,
                rng_seed=rng.randint(0, 2**31 - 1),
                **raw,
            )

            if cfg.filter_elements > INT32_MAX:
                continue
            if any(s <= 0 for s in cfg.y_shape[2:]):
                continue
            if _estimate_runtime_work(cfg) > RUNTIME_WORK_BUDGET:
                continue
            if _estimate_bytes(cfg) <= MEMORY_BUDGET_BYTES:
                return cfg

        raise RuntimeError(
            f"Could not generate a fitting config in {_MAX_CONFIG_GENERATION_ATTEMPTS} attempts. "
            f"Adjust MEMORY_BUDGET_BYTES, {_RUNTIME_WORK_BUDGET_ENV}, "
            "or generator shape ranges."
        )


def _conv_type_schedule(num_tests: int, rng_seed: int) -> List[ConvType]:
    """Return a deterministic schedule matching the configured op weights."""
    total_weight = sum(LargeTensorConfigGenerator._OP_WEIGHTS)
    exact_counts = [num_tests * weight / total_weight for weight in LargeTensorConfigGenerator._OP_WEIGHTS]
    counts = [math.floor(count) for count in exact_counts]
    remainder = num_tests - sum(counts)
    largest_remainders = sorted(
        range(len(counts)),
        key=lambda index: (exact_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in largest_remainders[:remainder]:
        counts[index] += 1

    schedule = [conv_type for conv_type, count in zip(LargeTensorConfigGenerator._OPS, counts) for _ in range(count)]
    random.Random(rng_seed ^ _OP_SCHEDULE_RNG_SALT).shuffle(schedule)
    return schedule


def tlist_with_configs(
    *,
    num_tests: int,
    rng_seed: int,
    allow_unaligned: bool = False,
    include_extras: bool = False,
    forced_conv_type: Optional[ConvType] = None,
    force_plain_fprop: bool = False,
) -> list:
    rng = random.Random(rng_seed)
    conv_types = [forced_conv_type] * num_tests if forced_conv_type is not None else _conv_type_schedule(num_tests, rng_seed)
    out = []
    for i, conv_type in enumerate(conv_types):
        config_seed = rng.randint(65536, 2**31 - 1)
        gen = LargeTensorConfigGenerator(
            config_seed,
            allow_unaligned=allow_unaligned,
            include_extras=include_extras,
            forced_conv_type=conv_type,
            force_plain_fprop=force_plain_fprop,
        )
        out.append((i + 1, num_tests, config_seed, gen.generate()))
    return out


# ---------------------------------------------------------------------------
# PyTorch float32 reference (cuDNN disabled to avoid self-comparison)
# ---------------------------------------------------------------------------


def _reference(cfg: LargeTensorConfig, X: torch.Tensor, W: torch.Tensor, Y: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    f32 = torch.float32
    fn = torch.nn.functional.conv2d if cfg.spatial_dims == 2 else torch.nn.functional.conv3d
    # Explicit deletes release large temporaries that exception tracebacks may otherwise retain.

    with torch.backends.cudnn.flags(enabled=False):
        if cfg.conv_type == ConvType.FPROP:
            Xf = X.to(f32).contiguous()
            Wf = W.to(f32).contiguous()
            try:
                ref = fn(Xf, Wf, padding=cfg.padding, stride=cfg.stride, dilation=cfg.dilation)
                if bias is not None:
                    ref = ref + bias.to(f32)
                if cfg.epilogue in ("relu", "bias_relu"):
                    ref = torch.relu(ref)
                return ref  # kept in f32; caller compares actual.float() vs ref
            finally:
                del Xf
                del Wf

        elif cfg.conv_type == ConvType.DGRAD:
            dYf = Y.to(f32).contiguous()
            Wf = W.to(f32).contiguous()
            dX = torch.zeros(cfg.x_shape, device="cuda", dtype=f32, requires_grad=True)
            fwd = None
            try:
                fwd = fn(dX, Wf, padding=cfg.padding, stride=cfg.stride, dilation=cfg.dilation)
                fwd.backward(dYf)
                return dX.grad.clone()  # f32
            finally:
                del dYf
                del Wf
                del dX
                if fwd is not None:
                    del fwd

        else:  # WGRAD
            Xf = X.to(f32).contiguous()
            dYf = Y.to(f32).contiguous()
            dW = torch.zeros(cfg.w_shape, device="cuda", dtype=f32, requires_grad=True)
            fwd = None
            try:
                fwd = fn(Xf, dW, padding=cfg.padding, stride=cfg.stride, dilation=cfg.dilation)
                fwd.backward(dYf)
                return dW.grad.clone()  # f32
            finally:
                del Xf
                del dYf
                del dW
                if fwd is not None:
                    del fwd


def _json_float(value: float) -> object:
    value = float(value)
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def _flat_index_to_coordinates(flat_index: int, shape: torch.Size) -> List[int]:
    coordinates = []
    for size in reversed(shape):
        coordinates.append(flat_index % size)
        flat_index //= size
    return list(reversed(coordinates))


def _top_ranked_indices(score: torch.Tensor, mismatch: torch.Tensor, count: int) -> List[int]:
    if count == 0:
        return []
    ranked = torch.where(mismatch, score, torch.full_like(score, -math.inf))
    return [int(index) for index in torch.topk(ranked, k=count).indices.cpu().tolist()]


def _mismatch_diagnostics(actual: torch.Tensor, reference: torch.Tensor, rtol: float, atol: float) -> dict:
    actual_flat = actual.reshape(-1)
    reference_flat = reference.reshape(-1)
    absolute_error = (actual_flat - reference_flat).abs()
    reference_abs = reference_flat.abs()
    allowed_error = atol + rtol * reference_abs
    finite_pair = torch.isfinite(actual_flat) & torch.isfinite(reference_flat)
    mismatch = (finite_pair & (absolute_error > allowed_error)) | (~finite_pair & (actual_flat != reference_flat))
    mismatch_count = int(mismatch.sum().item())
    candidate_count = min(_MISMATCH_SAMPLE_LIMIT, mismatch_count)

    absolute_rank = torch.nan_to_num(absolute_error, nan=math.inf, posinf=math.inf, neginf=math.inf)
    relative_error = absolute_error / reference_abs
    relative_error = torch.where((absolute_error == 0) & (reference_abs == 0), 0.0, relative_error)
    relative_rank = torch.nan_to_num(relative_error, nan=math.inf, posinf=math.inf, neginf=math.inf)
    absolute_indices = _top_ranked_indices(absolute_rank, mismatch, candidate_count)
    relative_indices = _top_ranked_indices(relative_rank, mismatch, candidate_count)

    selected_indices = []
    selection_basis = {}
    for rank in range(candidate_count):
        for basis, candidates in (("absolute_error", absolute_indices), ("relative_error", relative_indices)):
            index = candidates[rank]
            if index not in selection_basis:
                selection_basis[index] = basis
                selected_indices.append(index)
            if len(selected_indices) == candidate_count:
                break
        if len(selected_indices) == candidate_count:
            break

    selected = torch.tensor(selected_indices, device=actual.device, dtype=torch.long)
    actual_values = actual_flat[selected].cpu().tolist()
    reference_values = reference_flat[selected].cpu().tolist()
    absolute_values = absolute_error[selected].cpu().tolist()
    relative_values = relative_error[selected].cpu().tolist()
    samples = [
        {
            "flat_index": flat_index,
            "index": _flat_index_to_coordinates(flat_index, actual.shape),
            "selected_by": selection_basis[flat_index],
            "actual": _json_float(actual_value),
            "reference": _json_float(reference_value),
            "absolute_error": _json_float(absolute_value),
            "relative_error": _json_float(relative_value),
        }
        for flat_index, actual_value, reference_value, absolute_value, relative_value in zip(
            selected_indices,
            actual_values,
            reference_values,
            absolute_values,
            relative_values,
        )
    ]
    return {
        "mismatch_count": mismatch_count,
        "sample_limit": _MISMATCH_SAMPLE_LIMIT,
        "samples": samples,
    }


def _emit_mismatch_event(plan: _PlanIdentity, rng_seed: int, rtol: float, atol: float, diagnostics: dict) -> None:
    payload = {
        "rng_seed": rng_seed,
        **plan.as_dict(),
        "rtol": rtol,
        "atol": atol,
        **diagnostics,
    }
    samples = json.dumps(diagnostics, sort_keys=True, separators=(",", ":"))
    human_message = f"Large tensor fuzzer mismatch samples rng_seed={rng_seed} {_plan_identity_text(plan)}: {samples}"
    _emit_diagnostic_event("comparison_mismatch", payload, human_message=human_message)


# ---------------------------------------------------------------------------
# cuDNN graph execution
# ---------------------------------------------------------------------------


def _plan_identity(graph, plan_index: int) -> _PlanIdentity:
    graph_engine_index, knob_choices = graph.get_engine_and_knobs_at_index(plan_index)
    serialized_knobs = tuple(sorted((knob.name, int(value)) for knob, value in knob_choices.items()))
    return _PlanIdentity(plan_index, int(graph_engine_index), serialized_knobs)


def _plan_identity_text(plan: _PlanIdentity) -> str:
    knobs = json.dumps(dict(plan.knob_choices), sort_keys=True, separators=(",", ":"))
    return f"plan_index={plan.execution_plan_index} graph_engine_index={plan.graph_engine_index} knobs={knobs}"


def _bounded_cudnn_error_detail(error: Exception) -> str:
    backend_error = cudnn.get_last_error_string().strip()
    detail = "\n".join(part for part in (str(error).strip(), backend_error) if part)
    detail = " ".join(detail.split())
    if len(detail) > _PLAN_BUILD_ERROR_LIMIT:
        detail = f"{detail[:_PLAN_BUILD_ERROR_LIMIT]}..."
    return detail


def _nvrtc_plan_build_failure(plan: _PlanIdentity, detail: str) -> Optional[str]:
    """Return a bounded diagnostic when the immediate plan error is NVRTC."""
    if _NVRTC_COMPILATION_STATUS not in detail:
        return None
    return f"{_plan_identity_text(plan)}: {detail}"


def _emit_plan_event(
    event_type: str, plan: _PlanIdentity, rng_seed: int, outcome: str, *, detail: Optional[str] = None, workspace_bytes: Optional[int] = None
) -> None:
    payload = {
        "rng_seed": rng_seed,
        **plan.as_dict(),
        "outcome": outcome,
        "detail": detail,
        "workspace_bytes": workspace_bytes,
    }
    human_message = f"Large tensor fuzzer plan rng_seed={rng_seed} {_plan_identity_text(plan)}: {outcome}"
    _emit_diagnostic_event(event_type, payload, human_message=human_message)


def _build_first_supported_plan(graph, rng_seed: int) -> _PlanBuildSelection:
    """Build plans in priority order, preserving any NVRTC failures."""
    candidate_count = graph.get_execution_plan_count()
    nvrtc_failures = []

    for plan_index in range(candidate_count):
        plan = _plan_identity(graph, plan_index)
        try:
            graph.build_plan_at_index(plan_index)
        except cudnn.cudnnGraphNotSupportedError as e:
            detail = _bounded_cudnn_error_detail(e)
            nvrtc_failure = _nvrtc_plan_build_failure(plan, detail)
            if nvrtc_failure is not None:
                nvrtc_failures.append(nvrtc_failure)
                outcome = "NVRTC compilation failure"
            else:
                outcome = "unsupported"
            _emit_plan_event("plan_build", plan, rng_seed, outcome, detail=detail)
            continue
        except RuntimeError as e:
            detail = _bounded_cudnn_error_detail(e)
            nvrtc_failure = _nvrtc_plan_build_failure(plan, detail)
            if nvrtc_failure is None:
                raise
            nvrtc_failures.append(nvrtc_failure)
            _emit_plan_event("plan_build", plan, rng_seed, "NVRTC compilation failure", detail=detail)
            continue

        _emit_plan_event("plan_build", plan, rng_seed, "built successfully; selected for execution")
        return _PlanBuildSelection(plan, candidate_count, tuple(nvrtc_failures))

    print(
        f"Large tensor fuzzer plan rng_seed={rng_seed}: exhausted {candidate_count} candidates without a buildable plan",
        flush=True,
    )
    return _PlanBuildSelection(None, candidate_count, tuple(nvrtc_failures))


def _run_cudnn(cfg: LargeTensorConfig, X: torch.Tensor, W: torch.Tensor, Y: torch.Tensor, bias: Optional[torch.Tensor], handle) -> _CudnnRunResult:
    selection = _PlanBuildSelection(None, 0, ())
    try:
        cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
        io_dt = _cudnn_dtype(cfg.dtype)

        graph = cudnn.pygraph(
            handle=handle,
            io_data_type=io_dt,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        if cfg.conv_type == ConvType.FPROP:
            X_t = graph.tensor(name="X", dim=list(X.size()), stride=list(X.stride()), data_type=io_dt)
            W_t = graph.tensor(name="W", dim=list(W.size()), stride=list(W.stride()), data_type=io_dt)
            out = graph.conv_fprop(image=X_t, weight=W_t, padding=cfg.padding, stride=cfg.stride, dilation=cfg.dilation)
            if bias is not None:
                B_t = graph.tensor(name="B", dim=list(bias.size()), stride=list(bias.stride()), data_type=io_dt)
                out = graph.bias(name="bias", input=out, bias=B_t)
            if cfg.epilogue in ("relu", "bias_relu"):
                out = graph.relu(name="relu", input=out)
            out.set_output(True).set_data_type(io_dt)
            vpack = {X_t: X, W_t: W, out: Y}
            if bias is not None:
                vpack[B_t] = bias

        elif cfg.conv_type == ConvType.DGRAD:
            dY_t = graph.tensor(name="dY", dim=list(Y.size()), stride=list(Y.stride()), data_type=io_dt)
            W_t = graph.tensor(name="W", dim=list(W.size()), stride=list(W.stride()), data_type=io_dt)
            out = graph.conv_dgrad(loss=dY_t, filter=W_t, padding=cfg.padding, stride=cfg.stride, dilation=cfg.dilation)
            out.set_output(True).set_dim(list(X.size())).set_stride(list(X.stride()))
            vpack = {dY_t: Y, W_t: W, out: X}

        else:  # WGRAD
            X_t = graph.tensor(name="X", dim=list(X.size()), stride=list(X.stride()), data_type=io_dt)
            dY_t = graph.tensor(name="dY", dim=list(Y.size()), stride=list(Y.stride()), data_type=io_dt)
            out = graph.conv_wgrad(image=X_t, loss=dY_t, padding=cfg.padding, stride=cfg.stride, dilation=cfg.dilation)
            out.set_output(True).set_dim(list(W.size())).set_stride(list(W.stride()))
            vpack = {X_t: X, dY_t: Y, out: W}

        graph.validate()
        graph.build_operation_graph()
        filtered_op = _FORCED_CONV_TYPE
        filtered_graph_engine_indices = _COLLECTED_GRAPH_ENGINE_INDICES
        if filtered_graph_engine_indices:
            if cfg.conv_type != filtered_op:
                raise ValueError(f"{_GRAPH_ENGINE_OP_ENV}={filtered_op.name.lower()} does not match " f"test op {cfg.conv_type.name.lower()}")
            _validate_graph_engine_indices(graph, filtered_graph_engine_indices)
            requested_graph_engine_index = _select_graph_engine_index(cfg, filtered_graph_engine_indices)
            _create_filtered_execution_plans(graph, requested_graph_engine_index)
        else:
            graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        selection = _build_first_supported_plan(graph, cfg.rng_seed)
        if selection.selected_plan is None:
            if selection.nvrtc_failures:
                detail = "; ".join(selection.nvrtc_failures)
                return _CudnnRunResult(
                    False,
                    f"error: NVRTC compilation failed and no fallback plan built: {detail}",
                    nvrtc_failures=selection.nvrtc_failures,
                )
            return _CudnnRunResult(False, f"not_supported: no buildable execution plan among {selection.candidate_count} candidates")

        plan = selection.selected_plan
        plan_index = plan.execution_plan_index
        ws_size = graph.get_workspace_size_plan_at_index(plan_index)
        workspace = torch.empty(ws_size, device="cuda", dtype=torch.uint8)
        _poison_workspace(workspace)

        graph.execute_plan_at_index(vpack, workspace, index=plan_index, handle=handle)
        torch.cuda.synchronize()
        _emit_plan_event("plan_execution", plan, cfg.rng_seed, "execution passed", workspace_bytes=ws_size)
        return _CudnnRunResult(True, "ok", plan, selection.nvrtc_failures)

    except _EngineFilterNotSupported as e:
        return _CudnnRunResult(False, f"not_supported: {e}", nvrtc_failures=selection.nvrtc_failures)
    except cudnn.cudnnGraphNotSupportedError as e:
        if selection.selected_plan is not None:
            _emit_plan_event("plan_execution", selection.selected_plan, cfg.rng_seed, "execution failed", detail=_bounded_cudnn_error_detail(e))
        if selection.nvrtc_failures:
            prior_failures = "; ".join(selection.nvrtc_failures)
            return _CudnnRunResult(
                False,
                f"error: fallback plan index {selection.selected_plan_index} became unsupported after "
                f"NVRTC compilation failure(s): {e}; prior plan-build failures: {prior_failures}",
                selection.selected_plan,
                selection.nvrtc_failures,
            )
        return _CudnnRunResult(False, f"not_supported: {e}", nvrtc_failures=selection.nvrtc_failures)
    except torch.cuda.OutOfMemoryError as e:
        if selection.selected_plan is not None:
            _emit_plan_event("plan_execution", selection.selected_plan, cfg.rng_seed, "out of memory", detail=str(e))
        if selection.nvrtc_failures:
            prior_failures = "; ".join(selection.nvrtc_failures)
            return _CudnnRunResult(
                False,
                f"error: fallback plan index {selection.selected_plan_index} ran out of memory after "
                f"NVRTC compilation failure(s): {e}; prior plan-build failures: {prior_failures}",
                selection.selected_plan,
                selection.nvrtc_failures,
            )
        raise
    except (RuntimeError, OSError) as e:
        if selection.selected_plan is not None:
            _emit_plan_event("plan_execution", selection.selected_plan, cfg.rng_seed, "execution failed", detail=_bounded_cudnn_error_detail(e))
        prior_failures = f"; prior plan-build failures: {'; '.join(selection.nvrtc_failures)}" if selection.nvrtc_failures else ""
        return _CudnnRunResult(
            False,
            f"error: {type(e).__name__}: {e}{prior_failures}",
            selection.selected_plan,
            selection.nvrtc_failures,
        )


# ---------------------------------------------------------------------------
# Input tensor initialization
# ---------------------------------------------------------------------------


def _dense_random_cpu_tensor(shape: List[int], dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, dtype=dtype, generator=gen)


def _dense_small_integer_cpu_tensor(shape: List[int], dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    values = torch.randint(0, 2, shape, dtype=torch.int8, generator=gen)
    values.mul_(2).sub_(1)
    return values.to(dtype=dtype)


def _sparse_filter_cpu_tensor(cfg: LargeTensorConfig, rng: random.Random) -> torch.Tensor:
    """Create a mostly-zero filter with bounded nonzeros per output channel."""
    reduction_size = _fprop_reduction_size(cfg)
    nonzero_count = min(_SPARSE_INTEGER_TARGET_NONZERO, reduction_size)
    W = torch.zeros(cfg.w_shape, dtype=cfg.dtype)
    if nonzero_count == 0:
        return W

    flat = W.view(cfg.k, reduction_size)
    for k_idx in range(cfg.k):
        positions = rng.sample(range(reduction_size), nonzero_count)
        signs = rng.choices((-1, 1), k=nonzero_count)
        indices = torch.tensor(positions, dtype=torch.long)
        values = torch.tensor(signs, dtype=cfg.dtype)
        flat[k_idx, indices] = values

    return W


def _to_cuda_conv_layout(cpu_tensor: torch.Tensor, memory_format) -> torch.Tensor:
    return cpu_tensor.to("cuda", non_blocking=True).contiguous(memory_format=memory_format)


def _make_x_cpu_tensor(cfg: LargeTensorConfig, gen: torch.Generator) -> torch.Tensor:
    if _uses_integer_data(cfg) and cfg.conv_type in (ConvType.FPROP, ConvType.WGRAD):
        return _dense_small_integer_cpu_tensor(cfg.x_shape, cfg.dtype, gen)
    return _dense_random_cpu_tensor(cfg.x_shape, cfg.dtype, gen)


def _make_w_cpu_tensor(cfg: LargeTensorConfig, gen: torch.Generator, sparse_rng: random.Random) -> torch.Tensor:
    if _uses_sparse_filter(cfg):
        return _sparse_filter_cpu_tensor(cfg, sparse_rng)
    return _dense_random_cpu_tensor(cfg.w_shape, cfg.dtype, gen)


def _make_dy_cpu_tensor(cfg: LargeTensorConfig, gen: torch.Generator) -> torch.Tensor:
    if _uses_integer_data(cfg):
        return _dense_small_integer_cpu_tensor(cfg.y_shape, cfg.dtype, gen)
    return _dense_random_cpu_tensor(cfg.y_shape, cfg.dtype, gen)


def _make_bias_cpu_tensor(cfg: LargeTensorConfig, gen: torch.Generator) -> torch.Tensor:
    shape = [1, cfg.k] + [1] * cfg.spatial_dims
    if _uses_integer_data(cfg):
        return _dense_small_integer_cpu_tensor(shape, cfg.dtype, gen)
    return _dense_random_cpu_tensor(shape, cfg.dtype, gen)


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------


def _run_single_config(
    cfg: LargeTensorConfig,
    cudnn_handle,
    *,
    test_num: Optional[int] = None,
    total_tests: Optional[int] = None,
    config_seed: Optional[int] = None,
    attempt: Optional[int] = None,
) -> Tuple[bool, str]:
    # Generate inputs on CPU so values are bit-identical across GPU
    # architectures. CUDA random generation depends on launch geometry and can
    # make the same seed produce different values on different architectures.
    #
    # Convert transferred tensors to channels-last memory format (NHWC / NDHWC)
    # so graph strides are correct for both unit and non-unit output spatial.
    X = None
    W = None
    Y = None
    bias = None
    actual = None
    actual_f32 = None
    ref = None
    cudnn_result = None
    phase = "input setup"
    try:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(cfg.rng_seed)
        sparse_rng = random.Random(cfg.rng_seed ^ _SPARSE_INTEGER_RNG_SALT)

        if cfg.spatial_dims == 2:
            memory_format = torch.channels_last
        else:
            memory_format = torch.channels_last_3d

        phase = "X allocation"
        X = _to_cuda_conv_layout(_make_x_cpu_tensor(cfg, gen), memory_format)
        phase = "W allocation"
        W = _to_cuda_conv_layout(_make_w_cpu_tensor(cfg, gen, sparse_rng), memory_format)
        phase = "Y allocation"
        Y = torch.empty(cfg.y_shape, device="cuda", dtype=cfg.dtype).contiguous(memory_format=memory_format)

        if cfg.epilogue == "bias_relu":
            phase = "bias allocation"
            bias = _make_bias_cpu_tensor(cfg, gen).to("cuda", non_blocking=True)

        # Y holds the upstream gradient (dY) for backward passes.
        if cfg.conv_type in (ConvType.DGRAD, ConvType.WGRAD):
            phase = "upstream-gradient allocation"
            Y_cpu = _make_dy_cpu_tensor(cfg, gen)
            Y.copy_(Y_cpu, non_blocking=True)
            del Y_cpu

        phase = "output prefill"
        if cfg.conv_type == ConvType.FPROP:
            Y.fill_(float("nan"))
        elif cfg.conv_type == ConvType.DGRAD:
            X.fill_(float("nan"))
        else:
            W.fill_(float("nan"))

        rtol, atol = _tolerances(cfg)
        active_payload = _repro_payload(
            cfg,
            test_num=test_num,
            total_tests=total_tests,
            config_seed=config_seed,
            rtol=rtol,
            atol=atol,
            attempt=attempt,
        )
        active_json = _repro_json(active_payload)
        _emit_diagnostic_event("active_config", active_payload, human_message=f"Large tensor fuzzer active configuration: {active_json}")

        phase = "cuDNN execution"
        cudnn_result = _run_cudnn(cfg, X, W, Y, bias, cudnn_handle)
        if not cudnn_result.ok:
            return False, cudnn_result.message

        phase = "result clone"
        actual = Y.clone() if cfg.conv_type == ConvType.FPROP else X.clone() if cfg.conv_type == ConvType.DGRAD else W.clone()

        # _reference returns f32; compare actual upcast to f32 vs f32 truth.
        # Avoids self-comparison artifacts from casting the reference back to
        # the compute dtype.
        phase = "PyTorch reference"
        ref = _reference(cfg, X, W, Y, bias)
        phase = "result comparison"
        actual_f32 = actual.to(torch.float32)
        try:
            torch.testing.assert_close(actual_f32, ref, rtol=rtol, atol=atol)
        except AssertionError as e:
            selected_plan = cudnn_result.selected_plan
            if selected_plan is not None:
                try:
                    diagnostics = _mismatch_diagnostics(actual_f32, ref, rtol, atol)
                except (RuntimeError, OSError) as diagnostic_error:
                    diagnostics = {
                        "mismatch_count": None,
                        "sample_limit": _MISMATCH_SAMPLE_LIMIT,
                        "samples": [],
                        "diagnostic_error": f"{type(diagnostic_error).__name__}: {diagnostic_error}",
                    }
                _emit_mismatch_event(selected_plan, cfg.rng_seed, rtol, atol, diagnostics)
                _emit_plan_event("comparison", selected_plan, cfg.rng_seed, "numerical comparison failed")
            context = _format_repro_context(
                cfg,
                test_num=test_num,
                total_tests=total_tests,
                config_seed=config_seed,
                rtol=rtol,
                atol=atol,
                message=str(e),
                attempt=attempt,
                selected_plan=selected_plan,
            )
            if cudnn_result.nvrtc_failures:
                plan_failures = "\n".join(cudnn_result.nvrtc_failures)
                raise AssertionError(
                    f"NVRTC compilation failure(s) occurred before fallback plan index "
                    f"{cudnn_result.selected_plan_index} executed:\n{plan_failures}\n"
                    f"The fallback plan also failed numerical comparison:\n{e}{context}"
                ) from e
            raise AssertionError(f"{e}{context}") from e

        selected_plan = cudnn_result.selected_plan
        if selected_plan is not None:
            _emit_plan_event("comparison", selected_plan, cfg.rng_seed, "numerical comparison passed")
        if cudnn_result.nvrtc_failures:
            plan_failures = "; ".join(cudnn_result.nvrtc_failures)
            return (
                False,
                f"error: NVRTC compilation failure(s) occurred before fallback plan index "
                f"{cudnn_result.selected_plan_index} executed and passed numerical comparison: {plan_failures}",
            )
        return True, "ok"

    except torch.cuda.OutOfMemoryError as e:
        if cudnn_result is not None and cudnn_result.nvrtc_failures:
            plan_failures = "; ".join(cudnn_result.nvrtc_failures)
            return (
                False,
                f"error: {phase} ran out of memory after NVRTC compilation failure(s): {e}; " f"prior plan-build failures: {plan_failures}",
            )
        return False, f"insufficient_memory: {phase}: {e}"
    finally:
        if X is not None:
            del X
        if W is not None:
            del W
        if Y is not None:
            del Y
        if bias is not None:
            del bias
        if actual is not None:
            del actual
        if actual_f32 is not None:
            del actual_f32
        if ref is not None:
            del ref
        torch.cuda.empty_cache()


def _run_test(cfg: LargeTensorConfig, cudnn_handle, test_num: int, total_tests: int, config_seed: int, allow_unaligned: bool, include_extras: bool) -> None:
    if cudnn_handle is None:
        pytest.skip("cuDNN handle not available")

    if _REGEN_ON_UNSUPPORTED_ENABLED:
        # Replay from config_seed so regeneration can advance past unsupported
        # candidates; cfg is the first candidate from the same generator.
        _run_test_with_regen(
            config_seed=config_seed,
            test_num=test_num,
            total_tests=total_tests,
            allow_unaligned=allow_unaligned,
            include_extras=include_extras,
            cudnn_handle=cudnn_handle,
        )
        return

    try:
        ok, msg = _run_single_config(cfg, cudnn_handle, test_num=test_num, total_tests=total_tests, config_seed=config_seed)
    except AssertionError as e:
        _emit_outcome_event(
            "failure",
            "numeric_mismatch",
            cfg,
            test_num=test_num,
            total_tests=total_tests,
            config_seed=config_seed,
            message=str(e),
        )
        raise
    except (RuntimeError, OSError) as e:
        _emit_outcome_event(
            "failure",
            "frontend_error",
            cfg,
            test_num=test_num,
            total_tests=total_tests,
            config_seed=config_seed,
            message=str(e),
        )
        raise
    if ok:
        _emit_outcome_event(
            "test_complete",
            "passed",
            cfg,
            test_num=test_num,
            total_tests=total_tests,
            config_seed=config_seed,
        )
        return
    if msg.startswith("insufficient_memory"):
        _emit_outcome_event(
            "skip",
            "insufficient_memory",
            cfg,
            test_num=test_num,
            total_tests=total_tests,
            config_seed=config_seed,
            message=msg,
        )
        context = _format_repro_context(cfg, test_num=test_num, total_tests=total_tests, config_seed=config_seed, message=msg)
        pytest.skip(f"Insufficient GPU memory: {msg}{context}")
    if msg.startswith("not_supported"):
        _emit_outcome_event(
            "skip",
            "not_supported",
            cfg,
            test_num=test_num,
            total_tests=total_tests,
            config_seed=config_seed,
            message=msg,
        )
        pytest.skip(f"Graph not supported on this arch: {msg}")
    _emit_outcome_event(
        "failure",
        "cudnn_execution_error",
        cfg,
        test_num=test_num,
        total_tests=total_tests,
        config_seed=config_seed,
        message=msg,
    )
    context = _format_repro_context(cfg, test_num=test_num, total_tests=total_tests, config_seed=config_seed, message=msg)
    pytest.fail(f"cuDNN execution error: {msg}{context}")


def _run_test_with_regen(*, config_seed: int, test_num: int, total_tests: int, allow_unaligned: bool, include_extras: bool, cudnn_handle) -> None:
    filtered_op = _FORCED_CONV_TYPE
    if filtered_op is None:
        raise ValueError(f"{_REGEN_ON_UNSUPPORTED_ENV} requires {_GRAPH_ENGINE_OP_ENV}")

    gen = LargeTensorConfigGenerator(
        config_seed,
        allow_unaligned=allow_unaligned,
        include_extras=include_extras,
        forced_conv_type=filtered_op,
        force_plain_fprop=bool(_COLLECTED_GRAPH_ENGINE_INDICES),
    )
    max_attempts = _REGEN_ATTEMPT_LIMIT
    last_unsupported = "no configs generated"

    for attempt in range(1, max_attempts + 1):
        candidate = gen.generate()

        candidate_id = _test_id((test_num, total_tests, config_seed, candidate))
        try:
            ok, msg = _run_single_config(candidate, cudnn_handle, test_num=test_num, total_tests=total_tests, config_seed=config_seed, attempt=attempt)
        except AssertionError as e:
            _emit_outcome_event(
                "failure",
                "numeric_mismatch",
                candidate,
                test_num=test_num,
                total_tests=total_tests,
                config_seed=config_seed,
                message=str(e),
                attempt=attempt,
            )
            raise
        except (RuntimeError, OSError) as e:
            _emit_outcome_event(
                "failure",
                "frontend_error",
                candidate,
                test_num=test_num,
                total_tests=total_tests,
                config_seed=config_seed,
                message=str(e),
                attempt=attempt,
            )
            details = f"regenerated candidate={candidate_id}, attempt={attempt}; pytest node ID identifies attempt 1"
            raise RuntimeError(f"{e} [{details}]") from e

        if ok:
            _emit_outcome_event(
                "test_complete",
                "passed",
                candidate,
                test_num=test_num,
                total_tests=total_tests,
                config_seed=config_seed,
                attempt=attempt,
            )
            if attempt > 1:
                print(
                    f"{_REGEN_ON_UNSUPPORTED_ENV}: selected {candidate_id} " f"after {attempt} attempts; last unsupported: {last_unsupported}",
                    flush=True,
                )
            return
        if msg.startswith("insufficient_memory"):
            _emit_outcome_event(
                "skip",
                "insufficient_memory",
                candidate,
                test_num=test_num,
                total_tests=total_tests,
                config_seed=config_seed,
                message=msg,
                attempt=attempt,
            )
            context = _format_repro_context(
                candidate,
                test_num=test_num,
                total_tests=total_tests,
                config_seed=config_seed,
                message=msg,
                attempt=attempt,
            )
            pytest.skip(f"Insufficient GPU memory for {candidate_id}: {msg}{context}")
        if not msg.startswith("not_supported"):
            _emit_outcome_event(
                "failure",
                "cudnn_execution_error",
                candidate,
                test_num=test_num,
                total_tests=total_tests,
                config_seed=config_seed,
                message=msg,
                attempt=attempt,
            )
            context = _format_repro_context(candidate, test_num=test_num, total_tests=total_tests, config_seed=config_seed, message=msg, attempt=attempt)
            identity = f"attempt {attempt}; pytest node ID identifies attempt 1"
            failure = f"cuDNN execution error for regenerated candidate {candidate_id} ({identity}): {msg}{context}"
            pytest.fail(failure)
        last_unsupported = f"attempt {attempt}: {candidate_id}: {msg}"

    _emit_outcome_event(
        "skip",
        "not_supported_after_regeneration",
        candidate,
        test_num=test_num,
        total_tests=total_tests,
        config_seed=config_seed,
        message=last_unsupported,
        attempt=max_attempts,
    )
    pytest.skip(
        f"No supported {_GRAPH_ENGINE_OP_ENV}={filtered_op.name.lower()} config "
        f"found for selected graph engine index after {max_attempts} attempts; "
        f"last unsupported: {last_unsupported}"
    )


# ---------------------------------------------------------------------------
# Pre-generated parameter lists (built at collection time)
# ---------------------------------------------------------------------------

_FORCED_CONV_TYPE = _graph_engine_filter_op()
_COLLECTED_GRAPH_ENGINE_INDICES = _graph_engine_indices()
_REGEN_ON_UNSUPPORTED_ENABLED = _regen_on_unsupported()
_REGEN_ATTEMPT_LIMIT = _regen_attempts()

TEST_PARAMS_L0 = tlist_with_configs(
    num_tests=DEFAULT_NUM_TESTS_L0,
    rng_seed=DEFAULT_SEED_L0,
    allow_unaligned=False,
    include_extras=False,
    forced_conv_type=_FORCED_CONV_TYPE,
    force_plain_fprop=bool(_COLLECTED_GRAPH_ENGINE_INDICES),
)
TEST_PARAMS_L1 = tlist_with_configs(
    num_tests=DEFAULT_NUM_TESTS_L1,
    rng_seed=DEFAULT_SEED_L1,
    allow_unaligned=True,
    include_extras=True,
    forced_conv_type=_FORCED_CONV_TYPE,
    force_plain_fprop=bool(_COLLECTED_GRAPH_ENGINE_INDICES),
)

# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_conv_large_tensor_repro(cudnn_handle):
    cfg = _load_repro_config()
    if cfg is None:
        pytest.skip(f"Set {_REPRO_CONFIG_ENV} or {_REPRO_FILE_ENV} to run an exact configuration")
    if cudnn_handle is None:
        pytest.skip("cuDNN handle not available")

    try:
        ok, msg = _run_single_config(cfg, cudnn_handle)
    except AssertionError as e:
        _emit_outcome_event("failure", "numeric_mismatch", cfg, message=str(e))
        raise
    except (RuntimeError, OSError) as e:
        _emit_outcome_event("failure", "frontend_error", cfg, message=str(e))
        raise
    if ok:
        _emit_outcome_event("test_complete", "passed", cfg)
        return
    if msg.startswith("insufficient_memory"):
        _emit_outcome_event("skip", "insufficient_memory", cfg, message=msg)
        pytest.skip(f"Insufficient GPU memory: {msg}" f"{_format_repro_context(cfg, message=msg)}")
    if msg.startswith("not_supported"):
        _emit_outcome_event("skip", "not_supported", cfg, message=msg)
        pytest.skip(f"Graph not supported on this arch: {msg}")
    _emit_outcome_event("failure", "cudnn_execution_error", cfg, message=msg)
    pytest.fail(f"cuDNN execution error: {msg}" f"{_format_repro_context(cfg, message=msg)}")


@pytest.mark.L0
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", TEST_PARAMS_L0, ids=[_test_id(p) for p in TEST_PARAMS_L0])
def test_conv_large_tensor_L0(test_num, total_tests, config_seed, config, cudnn_handle):
    _run_test(config, cudnn_handle, test_num, total_tests, config_seed, allow_unaligned=False, include_extras=False)


@pytest.mark.L1
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", TEST_PARAMS_L1, ids=[_test_id(p) for p in TEST_PARAMS_L1])
def test_conv_large_tensor_L1(test_num, total_tests, config_seed, config, cudnn_handle):
    _run_test(config, cudnn_handle, test_num, total_tests, config_seed, allow_unaligned=True, include_extras=True)
