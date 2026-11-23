# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures and Torch references for the BF16 grouped GEMM squared-ReLU APIs (srelu and dsrelu).

Every output is compared three ways:

* the candidate, from the BF16 kernels under test;
* an FP32 reference: the same BF16 operands upcast to FP32, GEMMs in FP32 with TF32 off, the
  activation and its derivative in FP32, nothing rounded in between;
* the native path, what a model computes without the fusion: ``Z = bf16(A @ B^T)`` from a BF16
  GEMM with FP32 accumulation followed by the eager PyTorch activation of ``float(Z)``, and in the
  backward ``dH`` rounded to BF16 before the eager activation backward.

The fused kernels must not be less accurate than the native path, measured against the FP32
reference. Rows at or past the routed total are never written by the kernels and are left out.
MUFU ``tanh.approx`` (``tanh_mode="approx"`` with a clamp scale, the public default) is held to the
same criterion with no allowance; every such comparison prints the measured ratio as one
``APPROX_RATIO <test-id> <output> <ratio>`` line (see :func:`report_approx_ratio`).

Exact-valued problems (``make_problem(..., exact=True)``, :func:`ternary_weights`,
:func:`exact_prob`, :func:`exact_z`) hold small integers and signed powers of two, so every GEMM,
the plain squared-ReLU and its derivative are exact in FP32 in any evaluation order and the one
rounding to BF16 is the same everywhere: :func:`exact_forward_reference` and
:func:`exact_backward_reference` are independent float64 references for exact comparisons.

The class-API helpers (:func:`srelu_api`, :func:`dsrelu_api` and their ``*_execute``) build the
public facades over caller-owned outputs; ``execute()`` takes caller-owned scratch of
``scratch_workspace_bytes()`` bytes as ``workspace=`` (recipe R2), see ``gemm/cutedsl/_workspace.py``.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import subprocess
import sys
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import pytest
import torch

CLAMP_SCALE = 16.0
# L1 expert shapes: Nemotron 3 per rank (16 local experts, hidden 2688, expert FFN 1856, so N is not a
# multiple of the 256-column tile) and a small-K shape whose N is a whole number of 256-column tiles.
NEMOTRON3_SHAPE = {"experts": 16, "k": 2688, "n": 1856}
SMALL_K_SHAPE = {"experts": 8, "k": 512, "n": 3584}
# Relative slack on the native rel L2 error (different FP32 accumulation orders flip a few BF16
# roundings either way).
NATIVE_MARGIN = 0.05
# Near-native BF16 outputs: at most one BF16 ulp apart, on at most this fraction of the elements.
NEAR_NATIVE_FRACTION = 1e-3
FP32_UNIT_ROUNDOFF = 2.0**-24
# Pass as tanh_mode to leave the argument out of the call, so the public default is what runs.
PUBLIC_DEFAULT = object()
# The process-wide switch of the debug-only host checks of offset / pointer-table values.
DEVICE_VALUES_ENV = "CUDNN_FE_GROUPED_GEMM_VALIDATE_DEVICE_VALUES"


def require_bf16_srelu_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip(f"the BF16 grouped srelu kernels need SM100/SM103, found SM{capability[0]}{capability[1]}")
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed:
        pytest.skip("the BF16 grouped srelu kernels need CuTe DSL, which is not installed")
    from cudnn.gemm.cutedsl.grouped.srelu._bf16_options import DSL_FLOOR

    if cutedsl_too_old(version, DSL_FLOOR):
        floor = ".".join(str(x) for x in DSL_FLOOR)
        pytest.skip(f"the BF16 grouped srelu kernels need CuTe DSL >= {floor}, the version they are qualified on, found {version}")


def device_values_validated() -> bool:
    """True when this process reads offset / pointer-table values back to the host (debug switch set at import)."""
    from cudnn.gemm.cutedsl.grouped import backend_utils

    return bool(backend_utils.DEBUG_VALIDATE_DEVICE_VALUES)


def require_no_host_value_reads() -> None:
    """Skip a no-sync / capture detector in a process that opted into the blocking host checks."""
    if device_values_validated():
        pytest.skip(f"{DEVICE_VALUES_ENV}=1 makes execute() read device values to the host by design")


@contextlib.contextmanager
def fp32_highest_precision() -> Iterator[None]:
    """IEEE FP32 GEMMs (TF32 off) for the duration."""
    prev_precision = torch.get_float32_matmul_precision()
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = prev_cudnn
        torch.set_float32_matmul_precision(prev_precision)


@contextlib.contextmanager
def bf16_gemm_fp32_reduction() -> Iterator[None]:
    """BF16 GEMMs that reduce split-K partials in FP32, as an FP32-accumulating grouped GEMM does."""
    matmul = torch.backends.cuda.matmul
    prev = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_bf16_reduced_precision_reduction = False
    try:
        yield
    finally:
        matmul.allow_bf16_reduced_precision_reduction = prev


@contextlib.contextmanager
def sync_debug_error() -> Iterator[None]:
    """``torch.cuda.set_sync_debug_mode("error")`` for the duration; the caller's previous mode is
    restored on every exit. Inside a capture, enter this after ``torch.cuda.graph`` (test/AGENTS.md)."""
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


def allocator_state() -> Tuple[int, int]:
    """``(bytes allocated, allocations ever made)`` of the current device's caching allocator (recipe R9).

    The second number also moves for a temporary allocated and freed in between, which the first misses.
    """
    return torch.cuda.memory_allocated(), torch.cuda.memory_stats().get("allocation.all.allocated", 0)


def offsets_from_counts(counts: Sequence[int], device: str = "cuda") -> torch.Tensor:
    ends, total = [], 0
    for count in counts:
        total += count
        ends.append(total)
    return torch.tensor(ends, dtype=torch.int32, device=device)


def expert_ranges(counts: Sequence[int]) -> Iterator[Tuple[int, int, int]]:
    begin = 0
    for expert, count in enumerate(counts):
        yield expert, begin, begin + count
        begin += count


def mn1(tensor_2d: torch.Tensor) -> torch.Tensor:
    """``(rows, cols, 1)`` view with strides ``(cols, 1, rows * cols)`` of a contiguous 2-D tensor."""
    rows, cols = tensor_2d.shape
    return tensor_2d.as_strided((rows, cols, 1), (cols, 1, rows * cols))


def bf16_rows(rows: int, cols: int, fill: Optional[float] = None) -> torch.Tensor:
    """A fresh ``(rows, cols, 1)`` BF16 operand in the kernels' row-major layout (contiguous storage)."""
    storage = torch.empty((rows, cols), dtype=torch.bfloat16, device="cuda")
    if fill is not None:
        storage.fill_(fill)
    return mn1(storage)


def store_weights(logical: torch.Tensor, b_major: str) -> torch.Tensor:
    """Per-expert storage of logical ``(E, N, K)`` weights: ``[N, K]`` row-major (``"k"``) or ``[K, N]`` (``"n"``)."""
    if b_major == "k":
        return logical.contiguous()
    if b_major == "n":
        return logical.transpose(1, 2).contiguous()
    raise ValueError(f"b_major must be 'k' or 'n', got {b_major}")


def pointer_values(storage: torch.Tensor) -> torch.Tensor:
    """Host int64 tensor of the per-expert base addresses of ``storage`` (E leading)."""
    return torch.tensor([storage[e].data_ptr() for e in range(storage.shape[0])], dtype=torch.int64)


def make_problem(
    counts: Sequence[int],
    n: int,
    k: int,
    *,
    extra_rows: int = 0,
    discrete: bool = False,
    b_major: str = "k",
    seed: int = 0,
    a_scale: float = 1.0,
    identity: bool = False,
    exact: bool = False,
) -> Dict[str, Any]:
    """Grouped GEMM operands ``A`` ``(M, K, 1)`` BF16 and ``B_e`` ``(N, K)`` BF16 per expert.

    ``counts`` are the per-expert (256-padded) row counts, zeros allowed; ``extra_rows`` rows past
    the routed total are appended (never read). ``b_major`` selects how each expert's B is stored:
    ``"k"`` as ``[N, K]`` row-major (an FC1 weight), ``"n"`` as ``[K, N]`` row-major (an FC2 weight
    used for its dgrad). ``identity`` uses ``B_e = I`` (``N == K``), which makes the FP32
    accumulator equal to ``A`` exactly. ``exact`` draws ``A`` in {-1, +1} with its last column 0
    and every ``B_e`` in {-1, +1}: each accumulator sums ``K - 1`` terms of +-1, so it is an odd
    integer of magnitude below ``K`` -- never zero, exact in FP32 and, for ``K <= 257``, in BF16.
    """
    if any(count < 0 or count % 256 for count in counts):
        raise ValueError(f"counts must be non-negative multiples of 256, got {counts}")
    rows = sum(counts) + extra_rows
    if rows % 256:
        raise ValueError("extra_rows must keep M a multiple of 256")
    experts = len(counts)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if exact:
        signs = torch.randint(0, 2, (1, rows, k), generator=generator, device="cuda") * 2 - 1
        signs[..., k - 1] = 0
        a = signs.to(torch.bfloat16).permute(1, 2, 0)
    else:
        a = (torch.randn((1, rows, k), generator=generator, device="cuda") * a_scale).to(torch.bfloat16).permute(1, 2, 0)
    if identity:
        if n != k:
            raise ValueError("identity weights need N == K")
        b_logical = torch.eye(n, dtype=torch.bfloat16, device="cuda").expand(experts, n, n).contiguous()
    elif exact:
        b_logical = (torch.randint(0, 2, (experts, n, k), generator=generator, device="cuda") * 2 - 1).to(torch.bfloat16)
    else:
        b_logical = (torch.randn((experts, n, k), generator=generator, device="cuda") / math.sqrt(k)).to(torch.bfloat16)
    b_storage = store_weights(b_logical, b_major)
    if b_major == "k":
        b_dense = b_storage.permute(1, 2, 0)  # (N, K, E), strides (K, 1, N * K)
    else:
        b_dense = b_storage.permute(2, 1, 0)  # (N, K, E), strides (1, N, N * K)
    b_ptrs = pointer_values(b_storage).cuda()
    return {
        "a": a,
        "b": None if discrete else b_dense,
        "b_ptrs": b_ptrs if discrete else None,
        "b_storage": b_storage,
        "b_logical": b_logical,
        "b_major": b_major,
        "discrete": discrete,
        "offsets": offsets_from_counts(counts),
        "counts": list(counts),
        "rows": rows,
        "routed": sum(counts),
        "n": n,
        "k": k,
        "experts": experts,
    }


def ternary_weights(experts: int, n: int, k: int, *, seed: int) -> torch.Tensor:
    """Fresh logical ``(E, N, K)`` BF16 weights in {-1, 0, 1} (``randint(-1, 2)``): exact in any GEMM."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(-1, 2, (experts, n, k), generator=generator, device="cuda").to(torch.bfloat16)


def b_kwargs(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper weight arguments of a problem: the dense tensor or the per-expert pointer array."""
    if problem["discrete"]:
        return {"b_ptrs": problem["b_ptrs"], "n": problem["n"], "b_dtype": torch.bfloat16, "b_major": problem["b_major"]}
    return {"b_tensor": problem["b"]}


def make_prob(rows: int, *, seed: int, low: float = -1.0, high: float = 3.3) -> torch.Tensor:
    """Per-row routing probabilities ``(M, 1, 1)`` FP32; any sign and > 1 allowed by the contract."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand((rows, 1, 1), generator=generator, device="cuda") * (high - low) + low


def exact_prob(rows: int, *, seed: int, positive: bool = False) -> torch.Tensor:
    """Per-row probabilities ``(M, 1, 1)`` FP32 in {+-1/2, +-1, +-2} (positive only with ``positive``): exact factors."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    magnitude = torch.pow(2.0, torch.randint(-1, 2, (rows, 1, 1), generator=generator, device="cuda").float())
    if positive:
        return magnitude
    return magnitude * (torch.randint(0, 2, (rows, 1, 1), generator=generator, device="cuda").float() * 2 - 1)


def make_z(rows: int, n: int, *, seed: int, scale: float = 4.0) -> torch.Tensor:
    """BF16 pre-activations ``(M, N, 1)`` uniform in ``[-5, 5] * scale``."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    z = ((torch.rand((rows, n), generator=generator, device="cuda") * 10.0 - 5.0) * scale).to(torch.bfloat16)
    return mn1(z)


def exact_z(rows: int, n: int, *, seed: int, positive: bool = False) -> torch.Tensor:
    """BF16 saved pre-activations ``(M, N, 1)`` in {-3, ..., 3}, or {1, 2, 3} with ``positive``."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return mn1(torch.randint(1 if positive else -3, 4, (rows, n), generator=generator, device="cuda").to(torch.bfloat16))


def srelu_fp32(z: torch.Tensor, clamp_scale: Optional[float]) -> torch.Tensor:
    """``relu(z)**2`` or ``(s * tanh(relu(z) / s))**2`` with unfused ops in ``z``'s dtype."""
    if clamp_scale is None:
        r = torch.relu(z)
        return r * r
    c = clamp_scale * torch.tanh(torch.relu(z) / clamp_scale)
    return c * c


def eager_forward(z: torch.Tensor, prob: torch.Tensor, clamp_scale: Optional[float]) -> torch.Tensor:
    """The native activation: FP32 math on ``float(z)``, one rounding to BF16. ``prob`` is ``(M, 1)``."""
    zf = z.float()
    if clamp_scale is None:
        r = torch.relu(zf)
        return ((r * r) * prob).to(torch.bfloat16)
    clamped = clamp_scale * torch.tanh(torch.relu(zf) / clamp_scale)
    return (torch.pow(clamped, 2) * prob).to(torch.bfloat16)


def eager_backward(dh: torch.Tensor, z: torch.Tensor, prob: torch.Tensor, clamp_scale: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """The native activation backward from BF16 ``dh`` and ``z``: ``(dz BF16, dprob FP32 [M])``."""
    zf = z.float()
    g = dh.float()
    if clamp_scale is None:
        r = torch.relu(zf)
        dz = ((g * prob) * 2) * r
        dprob = torch.sum((r * r) * g, dim=-1)
        return dz.to(torch.bfloat16), dprob
    t = torch.tanh(torch.relu(zf) / clamp_scale)
    clamped = clamp_scale * t
    dz = (1 - torch.pow(t, 2)) * (2 * clamped) * g * prob
    dprob = torch.sum(torch.pow(clamped, 2) * g, dim=-1)
    return dz.to(torch.bfloat16), dprob


def grouped_matmul(problem: Dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
    """``A_e @ B_e^T`` per expert over the routed rows in ``dtype`` (rows past the total stay 0)."""
    a = problem["a"][:, :, 0].to(dtype)
    out = torch.zeros((problem["rows"], problem["n"]), dtype=dtype, device="cuda")
    for expert, begin, end in expert_ranges(problem["counts"]):
        if end > begin:
            out[begin:end] = a[begin:end] @ problem["b_logical"][expert].to(dtype).t()
    return out


def reference_forward(problem: Dict[str, Any], prob: torch.Tensor, clamp_scale: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP32 ``Z`` and ``H = p * srelu(Z)``, both ``(M, N)``."""
    with fp32_highest_precision():
        z = grouped_matmul(problem, torch.float32)
    return z, srelu_fp32(z, clamp_scale) * prob[:, :, 0]


def native_forward(problem: Dict[str, Any], prob: torch.Tensor, clamp_scale: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """BF16 ``Z = bf16(A @ B^T)`` and the eager activation of it, both ``(M, N)``."""
    with bf16_gemm_fp32_reduction():
        z = grouped_matmul(problem, torch.bfloat16)
    return z, eager_forward(z, prob[:, :, 0], clamp_scale)


def reference_backward(problem: Dict[str, Any], z_save: torch.Tensor, prob: torch.Tensor, clamp_scale: Optional[float]) -> Dict[str, torch.Tensor]:
    """FP32 ``dH = G @ B^T`` (never rounded) and the activation backward at ``float(Z_save)``.

    ``problem["a"]`` is ``G``. Returns ``dh``, ``dz``, ``dprob`` and ``h`` (the forward at
    ``Z_save``), FP32.
    """
    with fp32_highest_precision():
        dh = grouped_matmul(problem, torch.float32)
    z = z_save[:, :, 0].float()
    p = prob[:, :, 0]
    if clamp_scale is None:
        r = torch.relu(z)
        dz = 2.0 * r * dh * p
    else:
        t = torch.tanh(torch.relu(z) / clamp_scale)
        dz = (1.0 - t * t) * (2.0 * clamp_scale * t) * dh * p
    a = srelu_fp32(z, clamp_scale)
    return {"dh": dh, "dz": dz, "dprob": (a.double() * dh.double()).sum(dim=-1).float(), "h": a * p}


def native_backward(problem: Dict[str, Any], z_save: torch.Tensor, prob: torch.Tensor, clamp_scale: Optional[float]) -> Dict[str, torch.Tensor]:
    """BF16 ``dH = bf16(G @ B^T)`` and the eager backward and forward at ``Z_save``."""
    with bf16_gemm_fp32_reduction():
        dh = grouped_matmul(problem, torch.bfloat16)
    z = z_save[:, :, 0]
    p = prob[:, :, 0]
    dz, dprob = eager_backward(dh, z, p, clamp_scale)
    return {"dh": dh, "dz": dz, "dprob": dprob, "h": eager_forward(z, p, clamp_scale)}


def exact_grouped_matmul(a: torch.Tensor, b_logical: torch.Tensor, counts: Sequence[int]) -> torch.Tensor:
    """``A_e @ B_e^T`` per expert in float64 over the routed rows: ``(sum(counts), N)``."""
    a64 = a[:, :, 0].double()
    blocks = [a64[begin:end] @ b_logical[expert].double().t() for expert, begin, end in expert_ranges(counts) if end > begin]
    if not blocks:
        return torch.zeros((0, b_logical.shape[1]), dtype=torch.float64, device=a.device)
    return torch.cat(blocks)


def exact_forward_reference(a: torch.Tensor, b_logical: torch.Tensor, counts: Sequence[int], prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The plain squared-ReLU forward of an exact-valued problem: ``(C, D)``, BF16 ``(routed, N)``.

    float64 throughout and exact before the one rounding of each output to BF16 (FP32 -> BF16,
    round to nearest even, as the kernels' ``cvt.rn``), so every FP32 order gives these values.
    """
    acc = exact_grouped_matmul(a, b_logical, counts)
    p = prob[: acc.shape[0], 0, 0].double()[:, None]
    r = torch.relu(acc)
    return acc.float().to(torch.bfloat16), ((r * r) * p).float().to(torch.bfloat16)


def exact_backward_reference(
    a: torch.Tensor, b_logical: torch.Tensor, counts: Sequence[int], z_save: torch.Tensor, prob: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """The plain squared-ReLU backward of an exact-valued problem (``a`` is ``G``), float64 until one rounding.

    ``d_row`` and ``d_srelu`` BF16 ``(routed, N)``; ``dprob`` FP32 ``(routed,)``, an integer sum below
    ``2**24``, so every FP32 reduction order gives it exactly.
    """
    dh = exact_grouped_matmul(a, b_logical, counts)
    routed = dh.shape[0]
    z = z_save[:routed, :, 0].double()
    p = prob[:routed, 0, 0].double()[:, None]
    r = torch.relu(z)
    return {
        "d_row": (((dh * p) * 2.0) * r).float().to(torch.bfloat16),
        "d_srelu": ((r * r) * p).float().to(torch.bfloat16),
        "dprob": ((r * r) * dh).sum(dim=-1).float(),
    }


def rel_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    candidate = candidate.double()
    reference = reference.double()
    return float((candidate - reference).norm() / reference.norm().clamp_min(1e-30))


def approx_tanh(clamp_scale: Optional[float], tanh_mode: Any) -> bool:
    """MUFU tanh.approx is evaluated (a clamp scale with tanh_mode="approx"): its comparisons report their ratio."""
    return clamp_scale is not None and tanh_mode == "approx"


def current_test_id() -> str:
    """The running pytest node id without its phase suffix (``path::test[params]``)."""
    return os.environ.get("PYTEST_CURRENT_TEST", "unknown-test").rsplit(" (", 1)[0].replace(" ", "_")


def report_approx_ratio(output: str, ratio: float) -> str:
    """Print ``APPROX_RATIO <test-id> <output> <ratio>``: the candidate's rel L2 error over the native one.

    The line is printed (visible with ``-s`` or ``-rP``) and recorded through
    ``frost_routing.measured``, so a passing run carries it into the job log's ``measured`` section.
    """
    token = re.sub(r"\s+", "", output) or "output"
    line = f"APPROX_RATIO {current_test_id()} {token} {ratio:.4f}"
    print(line, flush=True)
    try:
        import frost_routing
    except ModuleNotFoundError as exc:
        if exc.name != "frost_routing":
            raise
        # Standalone reference checks have no pytest measurement collector; stdout remains available.
    else:
        frost_routing.measured(line, "approx/native rel L2 ratio")
    return line


def assert_not_worse_than_native(
    name: str, candidate: torch.Tensor, reference: torch.Tensor, native: torch.Tensor, *, routed: int, report_ratio: bool = False
) -> None:
    """rel L2 error of ``candidate`` vs ``reference`` at most the native one (+margin).

    ``report_ratio`` (the MUFU tanh.approx cells) reports the measured ratio first.
    """
    candidate = candidate.reshape(candidate.shape[0], -1)[:routed].float()
    reference = reference.reshape(reference.shape[0], -1)[:routed].float()
    native = native.reshape(native.shape[0], -1)[:routed].float()
    assert torch.isfinite(candidate).all(), f"{name}: non-finite candidate values"
    error, native_error = rel_l2(candidate, reference), rel_l2(native, reference)
    if report_ratio:
        report_approx_ratio(name, error / native_error if native_error > 0 else math.inf)
    assert error <= native_error * (1.0 + NATIVE_MARGIN) + 1e-7, f"{name}: rel L2 {error:.4e} vs native {native_error:.4e}"


def assert_values_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Exact value equality (``+0 == -0``, a NaN never equal): the check against an exact reference."""
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda message: f"{name}: {message}")


def bitwise(tensor: torch.Tensor) -> torch.Tensor:
    # Flatten first: a byte view needs a unit last stride, which the (rows, cols, 1) layouts lack.
    return tensor.contiguous().reshape(-1).view(torch.uint8)


_BITS_DTYPE = {1: torch.uint8, 2: torch.int16, 4: torch.int32}


def assert_bitwise_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual, expected = actual.contiguous(), expected.contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, f"{name}: {actual.shape}/{actual.dtype} vs {expected.shape}/{expected.dtype}"
    if not torch.equal(bitwise(actual), bitwise(expected)):
        bits_dtype = _BITS_DTYPE[actual.element_size()]
        differs = actual.reshape(-1).view(bits_dtype) != expected.reshape(-1).view(bits_dtype)
        raise AssertionError(f"{name}: {int(differs.sum())} of {differs.numel()} elements differ bitwise")


def _ordered_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """BF16 bit patterns mapped to integers that are ordered like the values (+0 and -0 both 0)."""
    bits = tensor.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def assert_near_native(name: str, candidate: torch.Tensor, native: torch.Tensor, *, max_ulp: int = 1) -> None:
    """BF16 ``candidate`` at most ``max_ulp`` BF16 ulps from ``native``, on a small fraction of elements."""
    candidate, native = candidate.contiguous(), native.contiguous()
    assert torch.equal(torch.isnan(candidate), torch.isnan(native)), f"{name}: NaN positions differ from native"
    finite = ~torch.isnan(native)
    ulps = (_ordered_bf16(candidate) - _ordered_bf16(native)).abs()[finite]
    limit = max(4, int(NEAR_NATIVE_FRACTION * ulps.numel()))
    worst, differing = int(ulps.max()) if ulps.numel() else 0, int((ulps != 0).sum())
    assert worst <= max_ulp, f"{name}: {worst} BF16 ulps from native (allowed {max_ulp})"
    assert differing <= limit, f"{name}: {differing} elements differ from native (allowed {limit})"


def dprob_sum_bound(z_save: torch.Tensor, dh: torch.Tensor, clamp_scale: Optional[float], tanh_mode: str) -> torch.Tensor:
    """Per-row bound on the difference of two FP32 sums of the same ``N`` terms ``A(z) * dh``.

    ``(4 sqrt(N) u + term_rel) * sum_n |A * dh|`` with ``u = 2^-24``: the summation-order term plus a
    per-term allowance for the tanh implementation (32 u for the accurate tanh, 2^-9 for MUFU
    tanh.approx).
    """
    z = z_save.reshape(z_save.shape[0], -1).float()
    terms = srelu_fp32(z, clamp_scale) * dh.reshape(dh.shape[0], -1).float()
    n_terms = z.shape[1]
    term_rel = 32.0 * FP32_UNIT_ROUNDOFF if tanh_mode == "accurate" else 2.0**-9
    return terms.abs().sum(dim=-1) * (4.0 * math.sqrt(n_terms) * FP32_UNIT_ROUNDOFF + term_rel)


def assert_dprob_within_bound(name: str, candidate: torch.Tensor, reference: torch.Tensor, bound: torch.Tensor, *, routed: int) -> None:
    candidate = candidate.reshape(-1)[:routed].double()
    reference = reference.reshape(-1)[:routed].double()
    bound = bound.reshape(-1)[:routed].double()
    for label, values in (("candidate", candidate), ("reference", reference), ("bound", bound)):
        assert bool(torch.isfinite(values).all()), f"{name}: non-finite {label} values"
    assert bool((bound >= 0).all()), f"{name}: negative summation bound"
    error = (candidate - reference).abs()
    over = error > bound
    assert not bool(
        over.any()
    ), f"{name}: {int(over.sum())} rows over the FP32 summation bound, worst ratio {float((error / bound.clamp_min(1e-30)).max()):.3f}"


# --------------------------------------------------------------------------- class API


def _weight_declaration(problem: Dict[str, Any]) -> Dict[str, Any]:
    if problem["discrete"]:
        return {"num_experts": problem["experts"], "b_shape": (problem["n"], problem["k"]), "b_dtype": torch.bfloat16, "b_major": problem["b_major"]}
    return {"sample_b": problem["b"]}


def _weight_binding(problem: Dict[str, Any]) -> Dict[str, Any]:
    return {"b_ptrs": problem["b_ptrs"]} if problem["discrete"] else {"b_tensor": problem["b"]}


def srelu_api(problem: Dict[str, Any], prob: torch.Tensor, c: torch.Tensor, d: torch.Tensor, **config):
    """``GroupedGemmSreluSm100`` over ``problem`` and the caller-owned outputs ``c`` / ``d`` (not checked or compiled)."""
    import cudnn

    return cudnn.GroupedGemmSreluSm100(
        sample_a=problem["a"],
        sample_c=c,
        sample_d=d,
        sample_padded_offsets=problem["offsets"],
        sample_prob=prob,
        **_weight_declaration(problem),
        **config,
    )


def srelu_execute(api, problem: Dict[str, Any], prob: torch.Tensor, c: torch.Tensor, d: torch.Tensor, *, workspace, **overrides) -> None:
    """``api.execute`` on ``problem``'s operands; ``overrides`` replaces any keyword argument."""
    kwargs = dict(
        a_tensor=problem["a"],
        sfa_tensor=None,
        padded_offsets=problem["offsets"],
        alpha_tensor=None,
        c_tensor=c,
        d_tensor=d,
        prob_tensor=prob,
        workspace=workspace,
        **_weight_binding(problem),
    )
    kwargs.update(overrides)
    api.execute(**kwargs)


def dsrelu_api(
    problem: Dict[str, Any],
    z_save: torch.Tensor,
    prob: torch.Tensor,
    d_row: torch.Tensor,
    dprob: torch.Tensor,
    *,
    d_srelu: Optional[torch.Tensor] = None,
    dprob_workspace: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    **config,
):
    """``GroupedGemmDsreluSm100`` over ``problem`` (``a`` is ``G``) and caller-owned outputs (not checked or compiled)."""
    import cudnn

    return cudnn.GroupedGemmDsreluSm100(
        sample_a=problem["a"],
        sample_c=z_save,
        sample_d_row=d_row,
        sample_d_srelu=d_srelu,
        sample_padded_offsets=problem["offsets"],
        sample_prob=prob,
        sample_dprob=dprob,
        sample_dprob_workspace=dprob_workspace,
        deterministic=deterministic,
        **_weight_declaration(problem),
        **config,
    )


def dsrelu_execute(
    api,
    problem: Dict[str, Any],
    z_save: torch.Tensor,
    prob: torch.Tensor,
    d_row: torch.Tensor,
    dprob: torch.Tensor,
    *,
    workspace,
    d_srelu: Optional[torch.Tensor] = None,
    dprob_workspace: Optional[torch.Tensor] = None,
    **overrides,
) -> None:
    """``api.execute`` on ``problem``'s operands; ``overrides`` replaces any keyword argument."""
    kwargs = dict(
        a_tensor=problem["a"],
        c_tensor=z_save,
        d_row_tensor=d_row,
        d_col_tensor=None,
        sfa_tensor=None,
        padded_offsets=problem["offsets"],
        alpha_tensor=None,
        prob_tensor=prob,
        dprob_tensor=dprob,
        d_srelu_tensor=d_srelu,
        dprob_workspace_tensor=dprob_workspace,
        workspace=workspace,
        **_weight_binding(problem),
    )
    kwargs.update(overrides)
    api.execute(**kwargs)


def count_wrapper_compiles(monkeypatch, module, facade_name: str, cache_name: str) -> list:
    """Empty a wrapper's plan cache for this test and record every facade ``compile()`` (the compile-count pattern of
    ``test_grouped_gemm_srelu.py``); returns the list the compiles are appended to."""
    monkeypatch.setattr(module, cache_name, {})
    facade = getattr(module, facade_name)
    original = facade.compile
    compiles = []

    def counted(self):
        compiles.append(self)
        return original(self)

    monkeypatch.setattr(facade, "compile", counted)
    return compiles


def dprob_slots(rows: int, n: int, **tiling) -> torch.Tensor:
    """A zeroed deterministic dprob slot buffer ``dprob_workspace_shape(rows, n)`` FP32."""
    import cudnn

    return torch.zeros(cudnn.GroupedGemmDsreluSm100.dprob_workspace_shape(rows, n, **tiling), dtype=torch.float32, device="cuda")


def aliasing_views(tensor: torch.Tensor, nbytes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(storage, disjoint, overlapping)``: one allocation holding a workspace of ``nbytes`` at its start,
    and two views with ``tensor``'s shape, strides and dtype -- one right after the workspace, one over it."""
    elements = 1 + sum((extent - 1) * stride for extent, stride in zip(tensor.shape, tensor.stride()))
    span = elements * tensor.element_size()
    storage = torch.empty(nbytes + span + 128, dtype=torch.uint8, device=tensor.device)

    def view(offset: int) -> torch.Tensor:
        return storage[offset : offset + span].view(tensor.dtype).as_strided(tuple(tensor.shape), tuple(tensor.stride()))

    return storage, view(nbytes), view(0)


class CompileSpy:
    """Stands in for ``cute.compile``: records ``(current CUDA device, DSL compile lock held)`` at every call in
    ``calls``, and the compiled object's type name in ``kinds``, then compiles. Subscription
    (compile options) goes to the real object."""

    def __init__(self, real):
        self.real = real
        self.calls = []
        self.kinds = []

    def __call__(self, *args, **kwargs):
        from cudnn.frost.compiled_cache import dsl_compile_lock

        self.calls.append((torch.cuda.current_device(), dsl_compile_lock()._is_owned()))
        self.kinds.append(type(args[0]).__name__ if args else "")
        return self.real(*args, **kwargs)

    def __getitem__(self, item):
        return self.real[item]


def persistent_clusters(cluster_shape_mn: Tuple[int, int], device_index: Optional[int] = None) -> int:
    """The persistent grid's cluster count the BF16 kernels launch: the occupancy query their ``compile()`` runs."""
    import cutlass.utils

    from cudnn.frost.compiled_cache import dsl_compile_lock

    index = torch.cuda.current_device() if device_index is None else device_index
    size = cluster_shape_mn[0] * cluster_shape_mn[1]
    with dsl_compile_lock(), torch.cuda.device(index):
        clusters = cutlass.utils.HardwareInfo(index).get_max_active_clusters(size)
    return int(clusters) - int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))


def other_bf16_device() -> Tuple[int, int]:
    """``(current, other)`` CUDA ordinals with ``other`` an SM100/SM103 GPU that is not current; skips otherwise."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two visible GPUs")
    caller = torch.cuda.current_device()
    other = (caller + 1) % torch.cuda.device_count()
    if torch.cuda.get_device_capability(other) not in ((10, 0), (10, 3)):
        pytest.skip(f"cuda:{other} is not SM100/SM103")
    return caller, other


def sm_count() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


# --------------------------------------------------------------------------- fresh-process checks

# Child prelude: arm the faulthandler before the parent's deadline, import the parent's cudnn package and prove it.
# argv: [package root, parent's cudnn.__file__, faulthandler deadline in seconds].
CHILD_PRELUDE = """
import faulthandler, os, re, sys
faulthandler.dump_traceback_later(float(sys.argv[3]), exit=True)
sys.path.insert(0, sys.argv[1])
import torch
import cudnn
assert os.path.realpath(cudnn.__file__) == os.path.realpath(sys.argv[2]), f"child imported {cudnn.__file__}, parent {sys.argv[2]}"


def expect_value_error(pattern, call):
    try:
        call()
    except ValueError as exc:
        assert re.search(pattern, str(exc)), f"{pattern!r} not in {exc}"
    else:
        raise AssertionError(f"no ValueError matching {pattern!r}")
"""


def _text(stream) -> str:
    if stream is None:
        return ""
    return stream.decode(errors="replace") if isinstance(stream, bytes) else stream


def run_python_child(body: str, *, env: Dict[str, str], cwd, marker: str, timeout_s: float = 900.0) -> str:
    """Run ``CHILD_PRELUDE + body`` in a fresh interpreter (``python -c``) with ``env`` added to this process's.

    The child imports the parent's ``cudnn`` (its package root goes first on ``sys.path``; the child checks the path),
    runs from ``cwd`` (no test-tree modules on its path) and dumps its stacks before the parent's deadline.
    The caller has checked the GPU: a ``pytest.skip`` cannot cross the process boundary.
    """
    import cudnn

    cudnn_file = os.path.abspath(cudnn.__file__)
    argv = [sys.executable, "-c", CHILD_PRELUDE + body, os.path.dirname(os.path.dirname(cudnn_file)), cudnn_file, str(timeout_s - 60.0)]
    try:
        result = subprocess.run(argv, env={**os.environ, **env}, cwd=str(cwd), capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"fresh-process check timed out after {timeout_s:.0f} s\n--- stdout\n{_text(exc.stdout)}\n--- stderr\n{_text(exc.stderr)}")
    assert (
        result.returncode == 0 and marker in result.stdout
    ), f"fresh-process check failed (rc {result.returncode})\n--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}"
    return result.stdout


# --------------------------------------------------------------------------- Rule 11 rendering pins

RENDER_OUT_ENV = "BF16_SRELU_RENDER_OUT"

# Prepended to a rendering child's body: ``save_ptx`` writes a compiled BF16 kernel's PTX module to RENDER_OUT_ENV
# (``__ptx__`` names the dumped file rather than holding the text when the DSL dumps it).
RENDER_CHILD_PRELUDE = r"""
import pathlib


def mn1(t):
    rows, cols = t.shape
    return t.as_strided((rows, cols, 1), (cols, 1, rows * cols))


def save_ptx(api, name):
    value = str(api._implementation._raw_compiled.__ptx__)
    if "\n" not in value.strip() and os.path.isfile(value.strip()):
        value = pathlib.Path(value.strip()).read_text()
    assert ".entry" in value, f"no PTX for {name}: {value[:200]!r}"
    (pathlib.Path(os.environ["BF16_SRELU_RENDER_OUT"]) / f"{name}.ptx").write_text(value)
"""

# Every warp-subset named barrier of the BF16 kernels, (id, threads): the epilogue warps (2, 128), the TMEM-allocation
# gate of the MMA and epilogue warps (3, 160) and the scheduler warp (4, 32). The CTA-wide barrier is the aligned one.
SUBSET_BARRIERS = {("2", "128"), ("3", "160"), ("4", "32")}


def render_env(tmp_path) -> Tuple[Dict[str, str], Any]:
    """``(env, ptx_dir)`` for a rendering child: the DSL compiles every kernel (no file cache) and keeps its PTX."""
    dump, out = tmp_path / "dump", tmp_path / "ptx"
    dump.mkdir()
    out.mkdir()
    return {"CUTE_DSL_KEEP": "ptx", "CUTE_DSL_DUMP_DIR": str(dump), "CUTE_DSL_DISABLE_FILE_CACHING": "1", RENDER_OUT_ENV: str(out)}, out


def main_kernel_ptx(ptx: str) -> str:
    """The module's largest ``.entry``: the main kernel (the discrete descriptor helper is a few dozen lines)."""
    entries = re.split(r"(?m)^(?=[ \t]*(?:\.visible[ \t]+|\.weak[ \t]+)?\.entry[ \t])", ptx)[1:]
    assert entries, "no .entry in the PTX"
    return max(entries, key=len)


def assert_release_rendering(label: str, ptx: str, *, sites: int, wide_words: Optional[int], acquire: bool) -> None:
    """python/cudnn/AGENTS.md Rule 11's rendering pin on a BF16 kernel's PTX, with its barrier and tensormap forms.

    * Exactly ``sites`` arrives are computed from loaded shared-memory registers through an ``and`` with the kernel's
      LAST parameter (the release-zero), each from at least the four tile-info words
      (warm_stress.assert_release_dependence); with ``wide_words``, exactly one of them (K2's Z_save stage) from at
      least that many.
    * Every warp-subset named barrier is the unaligned ``barrier.sync id, n`` (SUBSET_BARRIERS).
    * The per-expert tensormap acquire (``fence.proxy.tensormap::generic.acquire``) appears exactly when the weights
      are discrete.
    """
    import warm_stress

    kernel = main_kernel_ptx(ptx)
    dependent = [site for site in warm_stress.assert_release_dependence(kernel, sites=sites, words=4) if site.dependent]
    listing = "\n".join(f"  line {s.line}: loaded registers={s.loaded_registers} param={s.parameter} :: {s.text}" for s in dependent)
    entry = re.search(r"\.entry\s+(\w+)", kernel).group(1)
    parameter = re.compile(r"\b" + re.escape(entry) + r"_param_(\d+)\b")
    last = f"{entry}_param_{max(int(index) for index in parameter.findall(kernel))}"
    assert all(site.parameter == last for site in dependent), f"{label}: the mask is not the kernel's last parameter ({last}):\n{listing}"
    if wide_words is not None:
        wide = [site for site in dependent if site.loaded_registers >= wide_words]
        assert len(wide) == 1, f"{label}: expected one arrive computed from at least {wide_words} loaded registers:\n{listing}"
    aligned = re.findall(r"(?m)^\s*(?:@!?%p\d+\s+)?(?:bar\.sync|barrier\.sync\.aligned)\s+(\d+)\s*,\s*(\d+)", kernel)
    assert not aligned, f"{label}: aligned warp-subset barriers (id, threads) {aligned}"
    subsets = set(re.findall(r"(?m)^\s*(?:@!?%p\d+\s+)?barrier\.sync\s+(\d+)\s*,\s*(\d+)\s*;", kernel))
    assert subsets == SUBSET_BARRIERS, f"{label}: warp-subset barriers {sorted(subsets)}, expected {sorted(SUBSET_BARRIERS)}"
    acquires = len(re.findall(r"fence\.proxy\.tensormap::generic\.acquire", kernel))
    assert (acquires > 0) == acquire, f"{label}: {acquires} tensormap acquire(s); expected {'at least one' if acquire else 'none'}"
