"""
Large-tensor convolution regression tests.

Tests fprop, dgrad, and wgrad with large tensors and filters. FPROP and WGRAD
include C*R*S > 2^27 boundary coverage; DGRAD remains conservatively
runtime-bounded while exercising large problems. All cases execute the kernel
and compare against a PyTorch float32 reference.

Tune DEFAULT_NUM_TESTS_L0 / DEFAULT_NUM_TESTS_L1 to adjust local or downstream
runtime targets:
  L0: default smoke slice
  L1: expansion slice for longer runs

Developer diagnostic engine-filter examples:
  CUDNN_FUZZ_ENGINE_OP=fprop CUDNN_FUZZ_ENGINE_IDS=0 pytest ...
  CUDNN_FUZZ_ENGINE_OP=fprop CUDNN_FUZZ_ENGINE_IDS=0 \
    CUDNN_FUZZ_REGEN_ON_UNSUPPORTED=1 CUDNN_FUZZ_REGEN_ATTEMPTS=50 pytest ...
"""

import math
import os
import random
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
_ENGINE_FILTER_OP_ENV = "CUDNN_FUZZ_ENGINE_OP"  # local-only op filter: fprop/dgrad/wgrad
_ENGINE_FILTER_ENV = "CUDNN_FUZZ_ENGINE_IDS"  # local-only backend engine ids
_REGEN_ON_UNSUPPORTED_ENV = "CUDNN_FUZZ_REGEN_ON_UNSUPPORTED"  # retry unsupported engine picks
_REGEN_ATTEMPTS_ENV = "CUDNN_FUZZ_REGEN_ATTEMPTS"  # retry cap for regeneration
_WORK_BUDGET_FLOPS_ENV = "CUDNN_FUZZ_WORK_BUDGET_FLOPS"  # override generated test work cap
_NUM_TESTS_L0_ENV = "CUDNN_FUZZ_NUM_TESTS_L0"  # override L0 generated count
_NUM_TESTS_L1_ENV = "CUDNN_FUZZ_NUM_TESTS_L1"  # override L1 generated count
_BACKEND_ENGINE_MOD = 100  # backend enum suffix used by cudnnTest/backendEngine
_DEFAULT_WORK_BUDGET_FLOPS = 100_000_000_000_000  # 100 TFLOP per generated test


class _EngineFilterNotSupported(Exception):
    """Selected-engine support miss, not a cuDNN execution failure."""

    pass


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
        total = torch.cuda.get_device_properties(0).total_memory
        return int(total / _pytest_worker_count() * fraction)
    return 7 * (1 << 30)


def _engine_filter_ids() -> List[int]:
    """Parse the optional comma-separated backend engine id filter."""
    raw = os.environ.get(_ENGINE_FILTER_ENV, "").strip()
    op = os.environ.get(_ENGINE_FILTER_OP_ENV, "").strip()
    if op and not raw:
        raise ValueError(f"{_ENGINE_FILTER_ENV} must be set when {_ENGINE_FILTER_OP_ENV} is set")
    if not raw:
        return []

    try:
        return [int(v.strip()) for v in raw.split(",") if v.strip()]
    except ValueError as e:
        raise ValueError(f"{_ENGINE_FILTER_ENV} must be a comma-separated list of engine ids") from e


def _engine_filter_op() -> Optional["ConvType"]:
    """Parse the convolution op paired with the engine-id filter."""
    raw = os.environ.get(_ENGINE_FILTER_OP_ENV, "").strip().lower()
    ids = os.environ.get(_ENGINE_FILTER_ENV, "").strip()
    if ids and not raw:
        raise ValueError(f"{_ENGINE_FILTER_OP_ENV} must be set when {_ENGINE_FILTER_ENV} is set")
    if not raw:
        return None

    aliases = {
        "fprop": ConvType.FPROP,
        "fp": ConvType.FPROP,
        "dgrad": ConvType.DGRAD,
        "dg": ConvType.DGRAD,
        "wgrad": ConvType.WGRAD,
        "wg": ConvType.WGRAD,
    }
    try:
        return aliases[raw]
    except KeyError as e:
        raise ValueError(f"{_ENGINE_FILTER_OP_ENV} must be one of fprop, dgrad, or wgrad") from e


def _env_flag(name: str) -> bool:
    """Interpret common truthy env-var values."""
    raw = os.environ.get(name, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _regen_on_unsupported() -> bool:
    """Return whether local engine-filter runs should retry unsupported configs."""
    enabled = _env_flag(_REGEN_ON_UNSUPPORTED_ENV)
    if enabled and not _engine_filter_ids():
        raise ValueError(f"{_REGEN_ON_UNSUPPORTED_ENV} requires {_ENGINE_FILTER_OP_ENV} and " f"{_ENGINE_FILTER_ENV}")
    return enabled


def _regen_attempts() -> int:
    """Return the retry cap for regenerating unsupported configs."""
    raw = os.environ.get(_REGEN_ATTEMPTS_ENV, "").strip()
    if not raw:
        return MAX_REGEN_ATTEMPTS

    try:
        return max(1, int(raw))
    except ValueError as e:
        raise ValueError(f"{_REGEN_ATTEMPTS_ENV} must be a positive integer") from e


def _work_budget_flops() -> int:
    """Return the per-config estimated work cap in FLOPs."""
    raw = os.environ.get(_WORK_BUDGET_FLOPS_ENV, "").strip()
    if not raw:
        return _DEFAULT_WORK_BUDGET_FLOPS

    try:
        value = int(float(raw))
    except ValueError as e:
        raise ValueError(f"{_WORK_BUDGET_FLOPS_ENV} must be a positive number") from e

    if value <= 0:
        raise ValueError(f"{_WORK_BUDGET_FLOPS_ENV} must be a positive number")
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


def _kernel_cfg_choices(graph, engine: int) -> List[dict]:
    """Expand KERNEL_CFG knob values for one frontend engine."""
    knobs = graph.get_knobs_for_engine(engine)
    kernel_cfg_knobs = [knob for knob in knobs if knob.type == cudnn.knob_type.KERNEL_CFG]
    if not kernel_cfg_knobs:
        return [{}]

    choices = [{}]
    for knob in kernel_cfg_knobs:
        next_choices = []
        for value in range(knob.min_value, knob.max_value + 1, knob.stride):
            for choice in choices:
                candidate = dict(choice)
                candidate[knob.type] = value
                next_choices.append(candidate)
        choices = next_choices
    return choices


def _create_filtered_execution_plans(graph, backend_engine_ids: List[int]) -> None:
    """Create execution plans only for the requested backend engine ids."""
    engine_count = graph.get_engine_count()
    failures = []
    created_count = 0

    for backend_engine_id in backend_engine_ids:
        engine = backend_engine_id % _BACKEND_ENGINE_MOD
        if engine >= engine_count:
            failures.append(f"backend engine {backend_engine_id} -> frontend engine {engine} " f"outside available range 0..{engine_count - 1}")
            continue

        try:
            knob_choices = _kernel_cfg_choices(graph, engine)
        except RuntimeError as e:
            failures.append(f"backend engine {backend_engine_id} knobs unavailable: {e}")
            continue

        for knob_choice in knob_choices:
            try:
                graph.create_execution_plan(engine, knob_choice)
                created_count += 1
            except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as e:
                failures.append(f"backend engine {backend_engine_id} knob choice {knob_choice} rejected: {e}")

    if created_count == 0:
        details = "; ".join(failures[:3]) if failures else "no candidate plans created"
        raise _EngineFilterNotSupported(f"{_ENGINE_FILTER_ENV}={','.join(str(v) for v in backend_engine_ids)} " f"created no execution plans ({details})")


MEMORY_BUDGET_BYTES = _device_memory_budget()
WORK_BUDGET_FLOPS = _work_budget_flops()
WORKSPACE_OVERHEAD = 0.15  # fraction added for cuDNN workspace + allocator slack
DEFAULT_NUM_TESTS_L0 = _num_tests(_NUM_TESTS_L0_ENV, 48)  # default smoke slice
DEFAULT_NUM_TESTS_L1 = _num_tests(_NUM_TESTS_L1_ENV, 64)  # longer expansion slice
DEFAULT_SEED_L0 = 42
DEFAULT_SEED_L1 = 12345
MAX_REGEN_ATTEMPTS = 50
INT32_MAX = (1 << 31) - 1  # 2_147_483_647
_WORKSPACE_POISON_DIVISOR = 1000
_MAX_WORKSPACE_POISON_COUNT = 1_000_000
_WORKSPACE_POISON_BYTE = 0xFF

# Baseline accumulation depth and rtol before sqrt(accum) scaling.
BASE_ACCUM = 128
_STANDARD_RTOL_AT_BASE_ACCUM = 1e-2

# Machine epsilon per dtype, used for catastrophic-cancellation detection.
_DTYPE_EPS = {
    torch.float16: 9.77e-4,
    torch.bfloat16: 7.81e-3,
    torch.float32: 1.19e-7,
}
# Safety multiplier on the expected RMS rounding error (sqrt(accum) * eps).
# Defines the dtype's physical accuracy floor used as a tolerance fallback.
_CANCEL_SAFETY_FACTOR = 20.0

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConvType(Enum):
    FPROP = auto()
    DGRAD = auto()
    WGRAD = auto()


class ShapeFamily(Enum):
    NARROW = "nar"  # tiny input spatial, large filter -> maximises C*R*S vs memory
    BALANCED = "bal"  # H ~ W ~ R ~ S, typical real-world shapes
    DOWNSAMPLE = "dwn"  # stride > 1 with non-trivial output spatial
    BATCHED = "bat"  # N > 1, valid conv (exercises reduction-axis indexing)
    # Input strictly larger than filter, producing output spatial > 1x1 with
    # pad=0 stride=1. Distinguishes from the families where input is sized
    # exactly to make output 1x1. Keep the "nvc" value stable for pytest IDs.
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
    return {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
    }[t]


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


def _estimate_work_flops(cfg: LargeTensorConfig) -> int:
    """Approximate convolution work as multiply-add FLOPs.

    The memory budget keeps resident tensors tractable; this cap rejects cases
    whose implicit-GEMM work is too large for CI even when the tensors fit.
    """
    input_spatial = math.prod(cfg.input_spatial)
    filter_spatial = math.prod(cfg.filter_spatial)
    output_spatial = math.prod(cfg.y_shape[2:])

    if cfg.conv_type == ConvType.FPROP:
        return 2 * cfg.n * cfg.k * output_spatial * cfg.c * filter_spatial
    if cfg.conv_type == ConvType.DGRAD:
        return 2 * cfg.n * cfg.c * input_spatial * cfg.k * filter_spatial
    return 2 * cfg.k * cfg.c * filter_spatial * cfg.n * output_spatial


def _tolerances(cfg: LargeTensorConfig) -> Tuple[float, float]:
    """Return (rtol, atol) for the cuDNN-vs-reference comparison.

    Uses the looser of two bounds:
      - Standard: _STANDARD_RTOL_AT_BASE_ACCUM * sqrt(accum / BASE_ACCUM)
      - Dtype-rounding floor: sqrt(accum) * dtype_eps * _CANCEL_SAFETY_FACTOR

    The second bound is the physical limit on f16/bf16 accuracy at this
    accumulation depth; any cuDNN engine producing a result within it is as
    accurate as the dtype can represent. Previous versions gated the wider
    bound on a "reference near zero" check, but that proved unreliable across
    architectures because `torch.randn` on CUDA is not bit-identical across
    GPU arches (parallel PRNG depends on thread launch geometry) and cuDNN's
    engine selection differs per arch — both effects shift the |ref| value
    around the detection threshold.
    """
    accum = cfg.c * math.prod(cfg.filter_spatial)
    scale = max(1.0, math.sqrt(accum / BASE_ACCUM))
    std_tol = _STANDARD_RTOL_AT_BASE_ACCUM * scale
    eps = _DTYPE_EPS[cfg.dtype]
    cancel_atol = math.sqrt(float(accum)) * eps * _CANCEL_SAFETY_FACTOR

    atol = max(std_tol, cancel_atol)
    rtol = std_tol
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

    def __init__(self, seed: int, allow_unaligned: bool = False, include_extras: bool = False):
        self.rng = random.Random(seed)
        self.allow_unaligned = allow_unaligned
        self.include_extras = include_extras

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
            # Cap dims at 2048 and raise C to still hit C*R*S > 2^27 (the int32
            # overflow threshold). NOTE: empirically on H100 NVL (cuDNN 9.2.3),
            # 2D dgrad with any filter dim > ~2048 (e.g. R=2729, S=7023, f32)
            # takes >900 s — cuDNN dgrad kernels appear to scale very poorly for
            # large asymmetric 2D filter sizes. This is worth investigating in
            # the cuDNN backend.
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
        for _ in range(MAX_REGEN_ATTEMPTS):
            dtype = rng.choice(self._DTYPES)
            conv_type = rng.choices(self._OPS, weights=self._OP_WEIGHTS)[0]
            shape_family = rng.choices(self._SHAPE_FAMILIES, weights=self._SHAPE_FAMILY_WEIGHTS)[0]
            sdims = rng.choices([2, 3], weights=[0.75, 0.25])[0]

            raw = self._raw2d(shape_family) if sdims == 2 else self._raw3d(shape_family)

            # Epilogues only apply to FPROP
            if conv_type != ConvType.FPROP:
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
            if _estimate_work_flops(cfg) > WORK_BUDGET_FLOPS:
                continue
            if _estimate_bytes(cfg) <= MEMORY_BUDGET_BYTES:
                return cfg

        raise RuntimeError(
            f"Could not generate a fitting config in {MAX_REGEN_ATTEMPTS} attempts. "
            f"Adjust MEMORY_BUDGET_BYTES, {_WORK_BUDGET_FLOPS_ENV}, "
            "or generator shape ranges."
        )


def tlist_with_configs(*, num_tests: int, rng_seed: int, allow_unaligned: bool = False, include_extras: bool = False) -> list:
    rng = random.Random(rng_seed)
    out = []
    for i in range(num_tests):
        config_seed = rng.randint(65536, 2**31 - 1)
        gen = LargeTensorConfigGenerator(config_seed, allow_unaligned=allow_unaligned, include_extras=include_extras)
        out.append((i + 1, num_tests, config_seed, gen.generate()))
    return out


def _filter_params_for_engine_op(params: list) -> list:
    filtered_op = _engine_filter_op()
    if filtered_op is None:
        return params

    _engine_filter_ids()  # validate that engine ids were supplied with the op
    return [param for param in params if param[3].conv_type == filtered_op]


# ---------------------------------------------------------------------------
# PyTorch float32 reference (cuDNN disabled to avoid self-comparison)
# ---------------------------------------------------------------------------


def _reference(cfg: LargeTensorConfig, X: torch.Tensor, W: torch.Tensor, Y: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    f32 = torch.float32
    fn = torch.nn.functional.conv2d if cfg.spatial_dims == 2 else torch.nn.functional.conv3d

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


# ---------------------------------------------------------------------------
# cuDNN graph execution
# ---------------------------------------------------------------------------


def _run_cudnn(cfg: LargeTensorConfig, X: torch.Tensor, W: torch.Tensor, Y: torch.Tensor, bias: Optional[torch.Tensor], handle) -> Tuple[bool, str]:
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
        filtered_op = _engine_filter_op()
        filtered_backend_engines = _engine_filter_ids()
        if filtered_backend_engines:
            if cfg.conv_type != filtered_op:
                raise _EngineFilterNotSupported(f"{_ENGINE_FILTER_OP_ENV}={filtered_op.name.lower()} does not match " f"test op {cfg.conv_type.name.lower()}")
            _create_filtered_execution_plans(graph, filtered_backend_engines)
        else:
            graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        if filtered_backend_engines:
            if graph.get_execution_plan_count() <= 0:
                raise _EngineFilterNotSupported(
                    f"{_ENGINE_FILTER_ENV}={','.join(str(v) for v in filtered_backend_engines)} " "had no supported execution plans after check_support"
                )
            ws_size = graph.get_workspace_size_plan_at_index(0)
        else:
            ws_size = graph.get_workspace_size()
        workspace = torch.empty(ws_size, device="cuda", dtype=torch.uint8)
        _poison_workspace(workspace)

        if filtered_backend_engines:
            graph.execute_plan_at_index(vpack, workspace, index=0, handle=handle)
        else:
            graph.execute(vpack, workspace, handle=handle)
        torch.cuda.synchronize()
        return True, "ok"

    except _EngineFilterNotSupported as e:
        return False, f"not_supported: {e}"
    except cudnn.cudnnGraphNotSupportedError as e:
        return False, f"not_supported: {e}"
    except torch.cuda.OutOfMemoryError as e:
        # cuDNN requested a workspace larger than available GPU memory.
        # This is configuration/arch dependent (e.g. wgrad workspace for very
        # large filter sizes on some cuDNN builds).  Treat as not-supported so
        # the test skips rather than failing.
        return False, f"not_supported: OOM {e}"
    except (RuntimeError, OSError) as e:
        return False, f"error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------


def _run_single_config(cfg: LargeTensorConfig, cudnn_handle) -> Tuple[bool, str]:
    # Generate inputs on CPU then move to GPU so the values are bit-identical
    # across GPU architectures. `torch.randn` on a CUDA generator is not
    # bit-identical across arches because the parallel PRNG depends on thread
    # launch geometry; that arch-dependence makes failures non-reproducible
    # across H100 / A100 / B100.
    #
    # After transfer, convert to channels-last memory format (NHWC / NDHWC),
    # which is what cuDNN's heuristic engines expect for conv kernels. With
    # default NCHW layout, the OLD valid-conv tests (output spatial 1x1) worked
    # by accident because spatial strides collapse when spatial==1, but tests
    # with non-trivial output spatial produce scrambled cuDNN output.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.rng_seed)

    if cfg.spatial_dims == 2:
        memory_format = torch.channels_last
    else:
        memory_format = torch.channels_last_3d

    X = torch.randn(cfg.x_shape, dtype=cfg.dtype, generator=gen).to("cuda", non_blocking=True).contiguous(memory_format=memory_format)
    W = torch.randn(cfg.w_shape, dtype=cfg.dtype, generator=gen).to("cuda", non_blocking=True).contiguous(memory_format=memory_format)
    Y = torch.empty(cfg.y_shape, device="cuda", dtype=cfg.dtype).contiguous(memory_format=memory_format)
    bias = None

    if cfg.epilogue == "bias_relu":
        bias = torch.randn([1, cfg.k] + [1] * cfg.spatial_dims, dtype=cfg.dtype, generator=gen).to("cuda", non_blocking=True)

    # Y holds the upstream gradient (dY) for backward passes
    if cfg.conv_type in (ConvType.DGRAD, ConvType.WGRAD):
        Y_cpu = torch.randn(cfg.y_shape, dtype=cfg.dtype, generator=gen)
        Y.copy_(Y_cpu, non_blocking=True)
        del Y_cpu

    actual = None
    ref = None
    try:
        ok, msg = _run_cudnn(cfg, X, W, Y, bias, cudnn_handle)
        if not ok:
            return False, msg

        actual = Y.clone() if cfg.conv_type == ConvType.FPROP else X.clone() if cfg.conv_type == ConvType.DGRAD else W.clone()

        # _reference returns f32; compare actual upcast to f32 vs f32 truth.
        # Avoids self-comparison artifacts from casting the reference back to
        # the compute dtype.
        ref = _reference(cfg, X, W, Y, bias)
        rtol, atol = _tolerances(cfg)
        torch.testing.assert_close(actual.to(torch.float32), ref, rtol=rtol, atol=atol)
        return True, "ok"

    finally:
        del X
        del W
        del Y
        if bias is not None:
            del bias
        if actual is not None:
            del actual
        if ref is not None:
            del ref
        torch.cuda.empty_cache()


def _run_test(cfg: LargeTensorConfig, cudnn_handle, test_num: int, total_tests: int, config_seed: int, allow_unaligned: bool, include_extras: bool) -> None:
    if _regen_on_unsupported():
        _run_test_with_regen(
            config_seed=config_seed,
            test_num=test_num,
            total_tests=total_tests,
            allow_unaligned=allow_unaligned,
            include_extras=include_extras,
            cudnn_handle=cudnn_handle,
        )
        return

    ok, msg = _run_single_config(cfg, cudnn_handle)
    if ok:
        return
    if msg.startswith("not_supported"):
        pytest.skip(f"Graph not supported on this arch: {msg}")
    pytest.fail(f"cuDNN execution error: {msg}")


def _run_test_with_regen(*, config_seed: int, test_num: int, total_tests: int, allow_unaligned: bool, include_extras: bool, cudnn_handle) -> None:
    filtered_op = _engine_filter_op()
    if filtered_op is None:
        raise ValueError(f"{_REGEN_ON_UNSUPPORTED_ENV} requires {_ENGINE_FILTER_OP_ENV}")

    gen = LargeTensorConfigGenerator(config_seed, allow_unaligned=allow_unaligned, include_extras=include_extras)
    max_attempts = _regen_attempts()
    last_unsupported = "no configs generated"

    for attempt in range(1, max_attempts + 1):
        candidate = gen.generate()
        if candidate.conv_type != filtered_op:
            last_unsupported = f"attempt {attempt}: generated {candidate.conv_type.name.lower()} " f"while {_ENGINE_FILTER_OP_ENV}={filtered_op.name.lower()}"
            continue

        candidate_id = _test_id((test_num, total_tests, config_seed, candidate))
        try:
            ok, msg = _run_single_config(candidate, cudnn_handle)
        except (AssertionError, RuntimeError, OSError) as e:
            raise type(e)(f"{e} [candidate={candidate_id}, attempt={attempt}]") from e

        if ok:
            if attempt > 1:
                print(
                    f"{_REGEN_ON_UNSUPPORTED_ENV}: selected {candidate_id} " f"after {attempt} attempts; last unsupported: {last_unsupported}",
                    flush=True,
                )
            return
        if not msg.startswith("not_supported"):
            pytest.fail(f"cuDNN execution error for {candidate_id} " f"(regen attempt {attempt}): {msg}")
        last_unsupported = f"attempt {attempt}: {candidate_id}: {msg}"

    pytest.skip(
        f"No supported {_ENGINE_FILTER_OP_ENV}={filtered_op.name.lower()} config "
        f"found for selected engine after {max_attempts} attempts; "
        f"last unsupported: {last_unsupported}"
    )


# ---------------------------------------------------------------------------
# Pre-generated parameter lists (built at collection time)
# ---------------------------------------------------------------------------

TEST_PARAMS_L0 = tlist_with_configs(
    num_tests=DEFAULT_NUM_TESTS_L0,
    rng_seed=DEFAULT_SEED_L0,
    allow_unaligned=False,
    include_extras=False,
)
TEST_PARAMS_L0 = _filter_params_for_engine_op(TEST_PARAMS_L0)
TEST_PARAMS_L1 = tlist_with_configs(
    num_tests=DEFAULT_NUM_TESTS_L1,
    rng_seed=DEFAULT_SEED_L1,
    allow_unaligned=True,
    include_extras=True,
)
TEST_PARAMS_L1 = _filter_params_for_engine_op(TEST_PARAMS_L1)

# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", TEST_PARAMS_L0, ids=[_test_id(p) for p in TEST_PARAMS_L0])
def test_conv_large_tensor_L0(test_num, total_tests, config_seed, config, cudnn_handle):
    _run_test(config, cudnn_handle, test_num, total_tests, config_seed, allow_unaligned=False, include_extras=False)


@pytest.mark.L1
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", TEST_PARAMS_L1, ids=[_test_id(p) for p in TEST_PARAMS_L1])
def test_conv_large_tensor_L1(test_num, total_tests, config_seed, config, cudnn_handle):
    _run_test(config, cudnn_handle, test_num, total_tests, config_seed, allow_unaligned=True, include_extras=True)
