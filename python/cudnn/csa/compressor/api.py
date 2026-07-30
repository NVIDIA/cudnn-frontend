# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase wrappers for the fused CSA/HCA Compressor gated-pooling CuTe-DSL kernels.

``CSACompressorForward`` / ``CSACompressorBackward`` wrap the forward and backward
kernels in ``compressor_sm100.py`` (ported from Megatron-LM, see
https://github.com/NVIDIA/Megatron-LM/pull/5984 and
https://github.com/NVIDIA/Megatron-LM/issues/5968). The kernels fuse the gated-softmax
pooling region of the CSA/HCA ``Compressor`` for the THD packed layout:

    out[b, j] = sum_k kv[w(b, k), c(k, j)] * softmax_k(score[w(b, k), c(k, j)] + ape[k % ratio, c(k, j)])

over the per-block window ``w``: ``2 * ratio`` entries for the overlapping ``coff == 2``
form (the previous block's half-window is invalid for each segment's first block), or
``ratio`` entries for the own-block ``coff == 1`` form (no overlap, every window valid).
The framework-side autograd wiring stays in the caller (e.g. a ``torch.autograd.Function``
that calls the forward wrapper in ``forward()`` and the backward wrapper in
``backward()``); these APIs are pure kernels-plus-validation.

Validated envelope (``check_support``): compute capability 10.0, BF16 ``kv``/``score``/
``out``, FP32 ``ape``, int32 ``cu_seqlens``/``cu_seqlens_comp``, int32 flat offsets
(``total_tokens * coff * head_dim < 2**31``), and per ratio:

- ``ratio == 4``, ``coff in {1, 2}`` (``coff == 2`` is the production CSA/HCA
  configuration, ``coff == 1`` the own-block window form) — served by the generic
  kernels in ``compressor_sm100.py`` (whole window in registers; optimal at small
  ratios, register-bound beyond ``ratio = 32``);
- ``ratio == 128``, ``coff in {1, 2}``, ``head_dim in {128, 512}`` — served by the
  dedicated kernels in ``compressor_sm100_r128.py`` (bucketed-schedule chunked-softmax
  forward; staged smem backward with fused per-chunk reductions). The wrappers route
  by ``ratio`` transparently. NOTE: the numerics contracts differ per family — see
  below.

Numerics contract (see the kernel modules and docs/fe-oss-apis/csa.md for details):
fp32 arithmetic with one final bf16 rounding, ``mul.rn``/``fma.rn`` pinned in PTX.
Forward, ``dKV`` and ``dScore`` are bitwise run-to-run deterministic in BOTH families.
At ``ratio == 4`` dKV/dScore are additionally bit-identical to the fp32-intermediate
eager autograd; at ``ratio == 128`` the contract is faithfulness to that
fp32-intermediate eager reference: out/dKV/dScore match it within final-bf16 rounding
at the gate tolerances (differing elements <= max(1, 0.1%), max_abs <= 1.6e-2,
calibrated on the documented gate input distribution — absolute bf16 deviations scale
with the input magnitudes), inputs that overflow the reference's fp32 intermediates
reproduce its NaN/Inf propagation (both sides compute in fp32; gate-tested), and on
inputs whose fp32 intermediates stay finite those outputs additionally carry
fp64-oracle parity (at least as close to an fp64 oracle as the eager reference). That
is the approved deterministic tolerance contract (reduction reorders + fast-exp
buckets).
``dAPE`` uses one fp32 atomic per ``(k, dim)`` per CTA in both families and is not
run-to-run deterministic (the backward APIs refuse to run under
``torch.use_deterministic_algorithms(True)``).
"""

from __future__ import annotations

import threading
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import cuda.bindings.driver as cuda

from cudnn.api_base import APIBase, TupleDict

from .compressor_sm100 import (
    CU_ALIGN_BYTES,
    PTR_ALIGN_BYTES,
    SUPPORTED_COMPUTE_CAPABILITY,
    precompile_bwd,
    precompile_fwd,
    run_bwd,
    run_fwd,
)
from .compressor_sm100_r128 import (
    precompile_bwd_r128,
    precompile_fwd_r128,
    run_bwd_r128,
    run_fwd_r128,
)

# int32 flat offsets: every element offset the kernels compute must fit in int32.
_INT32_LIMIT = 2**31
# Forward launch schedule gridDim.y bound: at 128 threads per column CTA the largest
# launchable head_dim is 128 * 65535 (identical for the 64-thread vec == 2 path, which
# halves the column count).
_MAX_HEAD_DIM = 128 * 65535
# Bound + eviction policy follow python/cudnn/graph.py's graph_cache precedent.
_API_CACHE_MAXSIZE = 256


class _LruDict:
    """Bounded thread-safe LRU mapping (per ``graph.py``'s ``graph_cache`` precedent)."""

    def __init__(self, maxsize: int = _API_CACHE_MAXSIZE):
        """Create an empty LRU mapping evicting past ``maxsize`` entries."""
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def get(self, key, default=None):
        """Return the value for ``key`` (refreshing its recency) or ``default``."""
        with self._lock:
            if key not in self._data:
                return default
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key, value) -> None:
        """Insert ``key -> value`` as most recent, evicting past ``maxsize``."""
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)


def _resolve_stream_handle(current_stream: Optional[cuda.CUstream]) -> Optional[int]:
    """Integer stream handle for the launch path (None -> torch current stream)."""
    if current_stream is None:
        return None
    return int(current_stream)


@contextmanager
def _torch_stream_context(current_stream: Optional[cuda.CUstream], device: torch.device) -> Iterator[None]:
    """Run torch work on ``current_stream`` (device-tagged) when one is given."""
    if current_stream is None:
        yield
        return
    with torch.cuda.stream(torch.cuda.get_stream_from_external(int(current_stream), device)):
        yield


def _reject_deterministic_backward() -> None:
    """The backward accumulates ``dAPE`` with fp32 atomics and is not deterministic.

    Mirrors torch's deterministic-mode semantics: strict mode raises, warn-only mode
    warns and runs.
    """
    if torch.are_deterministic_algorithms_enabled():
        message = (
            "CSA compressor backward accumulates dAPE with fp32 atomics and is not "
            "deterministic; torch.use_deterministic_algorithms(True) is set. Use a "
            "deterministic (eager) implementation instead."
        )
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            raise RuntimeError(message)


class _CSACompressorBase(APIBase):
    """Shared descriptor plumbing and ``check_support`` for forward and backward."""

    def __init__(
        self,
        sample_kv: torch.Tensor,  # (total_tokens, coff * head_dim) BF16
        sample_score: torch.Tensor,  # (total_tokens, coff * head_dim) BF16
        sample_ape: torch.Tensor,  # (ratio, coff * head_dim) FP32
        sample_cu_seqlens: torch.Tensor,  # (B + 1,) INT32 token offsets
        sample_cu_seqlens_comp: torch.Tensor,  # (B + 1,) INT32 compressed-block offsets
        sample_out: torch.Tensor,  # (total_comp, head_dim) BF16 (forward output / backward grad_out)
        ratio: int = 4,
        coff: int = 2,
    ):
        """Capture tensor descriptors and the ``(ratio, coff)`` configuration.

        The ``sample_*`` tensors provide shape/dtype/stride/device metadata only (meta
        tensors are accepted); validation runs later in ``check_support`` and nothing
        is read or launched until ``execute``.
        """
        super().__init__()
        self._warn_experimental_api()

        self.kv_desc = self._make_tensor_desc(sample_kv, name="sample_kv")
        self.score_desc = self._make_tensor_desc(sample_score, name="sample_score")
        self.ape_desc = self._make_tensor_desc(sample_ape, name="sample_ape")
        self.cu_desc = self._make_tensor_desc(sample_cu_seqlens, name="sample_cu_seqlens")
        self.cuc_desc = self._make_tensor_desc(sample_cu_seqlens_comp, name="sample_cu_seqlens_comp")
        self.out_desc = self._make_tensor_desc(sample_out, name="sample_out")

        self.ratio = int(ratio)
        self.coff = int(coff)

        self.total_tokens = None
        self.total_comp = None
        self.head_dim = None
        self.n_seg = None
        self.target_device: Optional[torch.device] = None

    def check_support(self) -> bool:
        """Validate the configuration against the kernels' validated envelope.

        Malformed inputs and configurations outside the envelope raise ``ValueError``
        (device-capability failures raise ``RuntimeError``), mirroring the other FE-OSS
        APIs; there is no soft fallback path inside this API.
        """
        self._logger.debug("Entering check_support")
        if self.ratio == 4:
            self._value_error_if(
                self.coff not in (1, 2),
                f"CSA compressor at ratio=4 is validated for coff in {{1, 2}} (coff=2 is the production CSA/HCA form), got coff={self.coff}",
            )
        elif self.ratio == 128:
            self._value_error_if(
                self.coff not in (1, 2),
                f"CSA compressor at ratio=128 supports coff in {{1, 2}}, got coff={self.coff}",
            )
        else:
            self._value_error_if(
                True,
                f"CSA compressor is validated for ratio in {{4, 128}} only, got ratio={self.ratio}, coff={self.coff}",
            )
        self._value_error_if(
            self.kv_desc.ndim != 2,
            f"kv must be 2-D (total_tokens, coff * head_dim), got {self.kv_desc.shape}",
        )
        self._value_error_if(
            self.out_desc.ndim != 2,
            f"out/grad_out must be 2-D (total_comp, head_dim), got {self.out_desc.shape}",
        )
        total_tokens, width = self.kv_desc.shape
        total_comp, head_dim = self.out_desc.shape
        self._value_error_if(
            head_dim < 1 or width != self.coff * head_dim,
            f"kv width must equal coff * head_dim = {self.coff} * {head_dim}, got {width}",
        )
        self._value_error_if(
            self.score_desc.shape != self.kv_desc.shape,
            f"score shape {self.score_desc.shape} != kv shape {self.kv_desc.shape}",
        )
        self._value_error_if(
            self.ape_desc.shape != (self.ratio, width),
            f"ape must be (ratio, coff * head_dim) = ({self.ratio}, {width}), got {self.ape_desc.shape}",
        )
        self._value_error_if(
            self.cu_desc.ndim != 1 or self.cuc_desc.ndim != 1,
            "cu_seqlens and cu_seqlens_comp must be 1-D",
        )
        self._value_error_if(
            self.cu_desc.shape != self.cuc_desc.shape or self.cu_desc.shape[0] < 2,
            f"cu_seqlens and cu_seqlens_comp must both have B + 1 >= 2 entries, got {self.cu_desc.shape} and {self.cuc_desc.shape}",
        )

        self._check_dtype(self.kv_desc, torch.bfloat16, name="kv")
        self._check_dtype(self.score_desc, torch.bfloat16, name="score")
        self._check_dtype(self.ape_desc, torch.float32, name="ape")
        self._check_dtype(self.cu_desc, torch.int32, name="cu_seqlens")
        self._check_dtype(self.cuc_desc, torch.int32, name="cu_seqlens_comp")
        self._check_dtype(self.out_desc, torch.bfloat16, name="out/grad_out")

        # int32 flat offsets: the kernels index flat views with int32 arithmetic.
        self._value_error_if(
            total_tokens * width >= _INT32_LIMIT,
            f"total_tokens * coff * head_dim must be < 2**31 for int32 flat offsets, got {total_tokens} * {width}",
        )
        self._value_error_if(
            total_comp * head_dim >= _INT32_LIMIT,
            f"total_comp * head_dim must be < 2**31 for int32 flat offsets, got {total_comp} * {head_dim}",
        )
        # APE is indexed as (k % ratio) * width + col with the same int32 arithmetic
        # (only reachable at extreme head_dims, but cheap to pin explicitly).
        self._value_error_if(
            self.ratio * width >= _INT32_LIMIT,
            f"ratio * coff * head_dim must be < 2**31 for int32 APE offsets, got {self.ratio} * {width}",
        )
        # gridDim.y bound of the forward launch schedule (64/128-thread column groups):
        # head_dims beyond this cannot be launched (the pre-vectorization schedule had
        # the same 128 * 65535 envelope, just unchecked).
        self._value_error_if(
            head_dim > _MAX_HEAD_DIM,
            f"head_dim must be <= {_MAX_HEAD_DIM} (forward launch gridDim.y bound), got {head_dim}",
        )
        # The ratio=128 kernels are gated to the head_dims actually validated on
        # hardware (numerics gate + ptxas 0-spill + benchmark, see
        # compressor_sm100_r128.py); the kernels are generic and the gate can be
        # widened per head_dim once validated.
        if self.ratio == 128:
            self._value_error_if(
                head_dim not in (128, 512),
                f"CSA compressor at ratio=128 is validated for head_dim in {{128, 512}} only, got head_dim={head_dim}",
            )
        # Rows (including static-capacity padding rows) gather a window of `ratio`
        # tokens; the eager gather has the same requirement.
        self._value_error_if(
            total_comp > 0 and total_tokens < self.ratio,
            f"total_comp={total_comp} > 0 requires at least ratio={self.ratio} tokens, got {total_tokens}",
        )
        for desc, name in (
            (self.kv_desc, "kv"),
            (self.score_desc, "score"),
            (self.ape_desc, "ape"),
            (self.cu_desc, "cu_seqlens"),
            (self.cuc_desc, "cu_seqlens_comp"),
            (self.out_desc, "out/grad_out"),
        ):
            self._value_error_if(not desc.is_contiguous(), f"{name} must be contiguous")

        # Device resolution: all runtime (CUDA) descriptors on one device; meta
        # descriptors are metadata-only stand-ins and pin nothing.
        all_descs = (self.kv_desc, self.score_desc, self.ape_desc, self.cu_desc, self.cuc_desc, self.out_desc)
        devices = {desc.device for desc in all_descs}
        self._value_error_if(
            any(dev.type not in ("cuda", "meta") for dev in devices),
            f"all tensors must be CUDA tensors, got devices {sorted(str(dev) for dev in devices)}",
        )
        cuda_devices = {dev for dev in devices if dev.type == "cuda"}
        self._value_error_if(
            len(cuda_devices) > 1,
            f"all tensors must share one CUDA device, got {sorted(str(dev) for dev in cuda_devices)}",
        )
        self._runtime_error_if(not torch.cuda.is_available(), "CSA compressor requires CUDA")
        if cuda_devices:
            target = next(iter(cuda_devices))
            if target.index is None:
                target = torch.device("cuda", torch.cuda.current_device())
        else:
            target = torch.device("cuda", torch.cuda.current_device())

        capability = torch.cuda.get_device_capability(target)
        self._runtime_error_if(
            capability != SUPPORTED_COMPUTE_CAPABILITY,
            f"CSA compressor requires compute capability {SUPPORTED_COMPUTE_CAPABILITY} (the only validated architecture so far), found SM{capability[0]}.{capability[1]} on {target}",
        )

        self.total_tokens = total_tokens
        self.total_comp = total_comp
        self.head_dim = head_dim
        self.n_seg = self.cu_desc.shape[0] - 1
        self.target_device = target
        self._is_supported = True
        return True

    def _validate_runtime_tensor(self, tensor, name, shape, dtype, device, align):
        """Cheap per-call validation of one runtime tensor."""
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError(f"{name} must be a CUDA tensor on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride {tuple(tensor.stride())}")
        # Contiguity does not imply base-pointer alignment (storage-offset views); the
        # kernels' pointer wrappers assume it.
        if tensor.data_ptr() % align:
            raise ValueError(f"{name} base pointer must be {align}-byte aligned, got 0x{tensor.data_ptr():x}")

    @staticmethod
    def _record_streams(tensors, current_stream: Optional[cuda.CUstream], device: torch.device) -> None:
        """Keep tensor storages alive for work enqueued on an explicit external stream.

        The launch path takes raw pointers, so PyTorch's caching allocator does not know
        the kernel on ``current_stream`` still reads/writes these tensors: without
        ``record_stream`` it may recycle a storage freed by the caller while the kernel
        is pending. Only needed for explicit streams — with ``current_stream=None`` the
        launch lands on torch's current stream and ordinary stream semantics apply.
        """
        if current_stream is None:
            return
        consumer = torch.cuda.get_stream_from_external(int(current_stream), device)
        for t in tensors:
            t.record_stream(consumer)


class CSACompressorForward(_CSACompressorBase):
    """Fused CSA compressor forward: one kernel over the whole THD pack.

    Rows in ``[cu_seqlens_comp[-1], total_comp)`` are static-capacity padding (for
    CUDA-graph static shapes) and are computed with first-in-segment semantics from
    token 0, exactly like the eager gather.
    """

    def compile(self) -> None:
        """JIT-compile the forward kernel for this ``(ratio, head_dim, coff, device)`` (idempotent)."""
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        # Route by ratio: the dedicated ratio=128 kernels share the launch machinery
        # and cache pattern with the generic kernels (JIT cache keyed per
        # (ratio, head_dim, coff, device) plus the compile-time schedule); the
        # numerics contracts intentionally DIFFER per family — see the module
        # docstring and docs/fe-oss-apis/csa.md.
        if self.ratio == 128:
            precompile_fwd_r128(self.ratio, self.head_dim, self.coff, self.target_device)
            run = run_fwd_r128
        else:
            precompile_fwd(self.ratio, self.head_dim, self.coff, self.target_device)
            run = run_fwd

        ratio, head_dim, coff = self.ratio, self.head_dim, self.coff

        def tensor_api(kv, score, ape, cu_seqlens, cu_seqlens_comp, out, stream_handle):
            """Per-config closure: invoke the routed forward with the bound ``(ratio, head_dim, coff)``."""
            run(kv, score, ape, cu_seqlens, cu_seqlens_comp, out, out.shape[0], ratio, head_dim, coff, stream_handle=stream_handle)

        self._compiled_kernel = tensor_api
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        kv: torch.Tensor,  # (total_tokens, coff * head_dim) BF16, contiguous
        score: torch.Tensor,  # (total_tokens, coff * head_dim) BF16, contiguous
        ape: torch.Tensor,  # (ratio, coff * head_dim) FP32, contiguous
        cu_seqlens: torch.Tensor,  # (B + 1,) INT32, contiguous
        cu_seqlens_comp: torch.Tensor,  # (B + 1,) INT32, contiguous
        out: torch.Tensor,  # (total_comp, head_dim) BF16, contiguous
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Run the compiled forward kernel; ``out`` is fully overwritten."""
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise ValueError("CSACompressorForward kernel not compiled")
        device = self.target_device
        width = self.coff * self.head_dim
        self._validate_runtime_tensor(kv, "kv", (self.total_tokens, width), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(score, "score", (self.total_tokens, width), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(ape, "ape", (self.ratio, width), torch.float32, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(cu_seqlens, "cu_seqlens", (self.n_seg + 1,), torch.int32, device, CU_ALIGN_BYTES)
        self._validate_runtime_tensor(cu_seqlens_comp, "cu_seqlens_comp", (self.n_seg + 1,), torch.int32, device, CU_ALIGN_BYTES)
        self._validate_runtime_tensor(out, "out", (self.total_comp, self.head_dim), torch.bfloat16, device, PTR_ALIGN_BYTES)
        if out.numel() == 0:
            return
        self._compiled_kernel(kv, score, ape, cu_seqlens, cu_seqlens_comp, out, _resolve_stream_handle(current_stream))
        self._record_streams((kv, score, ape, cu_seqlens, cu_seqlens_comp, out), current_stream, device)


class CSACompressorBackward(_CSACompressorBase):
    """Fused CSA compressor backward: recompute window probs, write grads in one kernel.

    ``grad_kv``/``grad_score`` may be UNINITIALIZED: the kernel writes every position —
    consumed positions get their gradients (disjoint, atomic-free stores), and every
    never-consumed position (segment-tail tokens; for ``coff == 2`` the first-half
    columns of each segment's last block; tokens of segments shorter than ``ratio``;
    tokens beyond ``cu_seqlens[-1]`` when the buffers carry static token-capacity
    padding) gets an exact zero from its unique owning CTA — matching autograd without
    separate zero-fill kernels. Exception: when ``total_comp == 0`` the kernel is not
    launched and the buffers are left untouched, so a caller that needs autograd-exact
    zeros in that case must zero them itself (the high-level wrapper does).
    ``grad_ape`` must be zero-initialized by the caller before EVERY ``execute`` call
    and before every CUDA-graph replay that reuses the buffer: the kernel only
    ACCUMULATES into it (fp32 atomics, not bitwise run-to-run deterministic —
    ``grad_kv``/``grad_score`` are) and never clears it, so a reused buffer otherwise
    carries the previous invocation's sums. (The high-level wrapper allocates a fresh
    zeroed buffer per call; because that zero-fill is captured together with the
    kernel, wrapper graph replays re-zero automatically.)
    ``execute`` raises under ``torch.use_deterministic_algorithms(True)``. Incoming
    gradients on static-capacity padding rows (``[cu_seqlens_comp[-1], total_comp)``)
    are ignored.
    """

    def compile(self) -> None:
        """JIT-compile the backward kernel for this ``(ratio, head_dim, coff, device)`` (idempotent)."""
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        # Route by ratio, as in the forward.
        if self.ratio == 128:
            precompile_bwd_r128(self.ratio, self.head_dim, self.coff, self.target_device)
            run = run_bwd_r128
        else:
            precompile_bwd(self.ratio, self.head_dim, self.coff, self.target_device)
            run = run_bwd

        ratio, head_dim, coff = self.ratio, self.head_dim, self.coff

        def tensor_api(kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out, grad_kv, grad_score, grad_ape, stream_handle):
            """Per-config closure: invoke the routed backward with the bound ``(ratio, head_dim, coff)``."""
            run(
                kv,
                score,
                ape,
                cu_seqlens,
                cu_seqlens_comp,
                grad_out,
                grad_kv,
                grad_score,
                grad_ape,
                grad_out.shape[0],
                ratio,
                head_dim,
                coff,
                stream_handle=stream_handle,
            )

        self._compiled_kernel = tensor_api
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        kv: torch.Tensor,  # (total_tokens, coff * head_dim) BF16, contiguous
        score: torch.Tensor,  # (total_tokens, coff * head_dim) BF16, contiguous
        ape: torch.Tensor,  # (ratio, coff * head_dim) FP32, contiguous
        cu_seqlens: torch.Tensor,  # (B + 1,) INT32, contiguous
        cu_seqlens_comp: torch.Tensor,  # (B + 1,) INT32, contiguous
        grad_out: torch.Tensor,  # (total_comp, head_dim) BF16, contiguous
        grad_kv: torch.Tensor,  # (total_tokens, coff * head_dim) BF16 (may be uninitialized; fully written when total_comp > 0)
        grad_score: torch.Tensor,  # (total_tokens, coff * head_dim) BF16 (may be uninitialized; fully written when total_comp > 0)
        grad_ape: torch.Tensor,  # (ratio, coff * head_dim) FP32, zero-initialized before EVERY call/replay (kernel accumulates)
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Run the compiled backward kernel into the gradient buffers (see class docs)."""
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise ValueError("CSACompressorBackward kernel not compiled")
        _reject_deterministic_backward()
        device = self.target_device
        width = self.coff * self.head_dim
        self._validate_runtime_tensor(kv, "kv", (self.total_tokens, width), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(score, "score", (self.total_tokens, width), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(ape, "ape", (self.ratio, width), torch.float32, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(cu_seqlens, "cu_seqlens", (self.n_seg + 1,), torch.int32, device, CU_ALIGN_BYTES)
        self._validate_runtime_tensor(cu_seqlens_comp, "cu_seqlens_comp", (self.n_seg + 1,), torch.int32, device, CU_ALIGN_BYTES)
        self._validate_runtime_tensor(grad_out, "grad_out", (self.total_comp, self.head_dim), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(grad_kv, "grad_kv", (self.total_tokens, width), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(grad_score, "grad_score", (self.total_tokens, width), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(grad_ape, "grad_ape", (self.ratio, width), torch.float32, device, PTR_ALIGN_BYTES)
        if grad_out.numel() == 0:
            return
        self._compiled_kernel(kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out, grad_kv, grad_score, grad_ape, _resolve_stream_handle(current_stream))
        self._record_streams((kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out, grad_kv, grad_score, grad_ape), current_stream, device)


# module-level bounded LRU cache of compiled API instances (thread-safe):
#   (kind, ratio, head_dim, coff, total_tokens, total_comp, n_seg, device) -> api
# The compiled kernel underneath is shared per (ratio, head_dim, coff, device) through
# compressor_sm100's compile cache, so shape changes only rebuild the cheap wrapper
# object, never the JIT.
_api_cache = _LruDict()
# serializes verdict construction + JIT so concurrent same-config callers cannot
# compile the same kernel twice
_api_build_lock = threading.Lock()


def _get_api(kind, kv, score, ape, cu_seqlens, cu_seqlens_comp, out_shape, ratio, coff):
    """Build (or fetch) a compiled forward/backward API instance for these tensors."""
    key = (kind, int(ratio), int(coff), out_shape[1], tuple(kv.shape), out_shape[0], cu_seqlens.shape[0], kv.device.index)
    api = _api_cache.get(key)
    if api is not None:
        return api
    with _api_build_lock:
        api = _api_cache.get(key)
        if api is not None:
            return api
        sample_out = torch.empty(out_shape, dtype=torch.bfloat16, device="meta")
        cls = CSACompressorForward if kind == "fwd" else CSACompressorBackward
        api = cls(
            sample_kv=kv,
            sample_score=score,
            sample_ape=ape,
            sample_cu_seqlens=cu_seqlens,
            sample_cu_seqlens_comp=cu_seqlens_comp,
            sample_out=sample_out,
            ratio=ratio,
            coff=coff,
        )
        api.check_support()
        api.compile()
        _api_cache.put(key, api)
        return api


def _infer_head_dim(kv: torch.Tensor, head_dim: Optional[int], coff: int) -> int:
    """Infer ``head_dim`` from the packed kv width when not given explicitly."""
    if kv.ndim != 2:
        raise ValueError(f"kv must be 2-D (total_tokens, coff * head_dim), got {tuple(kv.shape)}")
    if head_dim is not None:
        return int(head_dim)
    width = kv.shape[1]
    if coff < 1 or width % coff != 0:
        raise ValueError(f"cannot infer head_dim from kv width {width} and coff {coff}")
    return width // coff


def csa_compressor_forward_wrapper(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    ratio: int = 4,
    head_dim: Optional[int] = None,
    coff: int = 2,
    total_comp: Optional[int] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level forward wrapper. Allocates and returns the pooled output.

    Args:
        kv: ``(total_tokens, coff * head_dim)`` BF16 gate values (THD packed).
        score: ``(total_tokens, coff * head_dim)`` BF16 gate scores.
        ape: ``(ratio, coff * head_dim)`` FP32 additive position embedding.
        cu_seqlens: ``(B + 1,)`` int32 cumulative token counts per segment.
        cu_seqlens_comp: ``(B + 1,)`` int32 cumulative compressed-block counts,
            ``cu_seqlens_comp[b + 1] - cu_seqlens_comp[b] == seqlen_b // ratio``.
        ratio: compression ratio (tokens per output block); validated envelope:
            {4, 128} (the wrappers route to the matching kernel family by ratio).
        head_dim: output feature dimension; inferred from ``kv`` width when omitted.
        coff: 1 for the own-block window form (window = ``ratio`` tokens, no overlap) or
            2 for the overlapping-window form (window = ``2 * ratio``); validated
            envelope: {1, 2} at both ratios (ratio=128 additionally requires head_dim
            in {128, 512}).
        total_comp: output row count. Defaults to ``cu_seqlens_comp[-1]`` (synchronizes);
            pass it explicitly (e.g. a static CUDA-graph capacity, which must be
            ``>= cu_seqlens_comp[-1]``) to stay capture-safe.
        stream: CUDA stream for allocation and kernel launch (None -> current stream).

    Returns:
        ``{'out': (total_comp, head_dim) BF16}`` pooled output (pre-RMSNorm).
    """
    head_dim = _infer_head_dim(kv, head_dim, coff)
    if total_comp is None:
        if cu_seqlens_comp.numel() < 1:
            raise ValueError("cu_seqlens_comp must have B + 1 >= 2 entries")
        total_comp = int(cu_seqlens_comp[-1].item())
    api = _get_api("fwd", kv, score, ape, cu_seqlens, cu_seqlens_comp, (int(total_comp), head_dim), ratio, coff)
    with torch.cuda.device(kv.device), _torch_stream_context(stream, kv.device):
        out = torch.empty(int(total_comp), head_dim, dtype=torch.bfloat16, device=kv.device)
    with torch.cuda.nvtx.range("csa_compressor_fwd_kernel"):
        api.execute(kv, score, ape, cu_seqlens, cu_seqlens_comp, out, current_stream=stream)
    return TupleDict(out=out)


def csa_compressor_backward_wrapper(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    grad_out: torch.Tensor,
    ratio: int = 4,
    head_dim: Optional[int] = None,
    coff: int = 2,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level backward wrapper. Allocates the grad buffers and fills them.

    ``grad_out`` is ``(total_comp, head_dim)`` BF16 (the incoming gradient of the
    forward wrapper's ``out``); gradients on static-capacity padding rows are ignored.
    ``grad_kv``/``grad_score`` are allocated UNINITIALIZED — the kernel writes every
    position, storing exact zeros to never-consumed positions itself, so no zero-fill
    kernels run; ``grad_ape`` is allocated zeroed (fp32 atomic accumulation). When
    ``total_comp == 0`` the kernel is not launched and all three grads are allocated as
    zeros instead, preserving autograd's exact-zero semantics.
    Raises ``RuntimeError`` under ``torch.use_deterministic_algorithms(True)`` because
    ``grad_ape`` is accumulated with fp32 atomics (``grad_kv``/``grad_score`` are
    deterministic and bitwise reproducible).

    Returns:
        ``{'grad_kv': (total_tokens, coff * head_dim) BF16,
           'grad_score': (total_tokens, coff * head_dim) BF16,
           'grad_ape': (ratio, coff * head_dim) FP32}``
    """
    head_dim = _infer_head_dim(kv, head_dim, coff)
    if grad_out.ndim != 2:
        raise ValueError(f"grad_out must be 2-D (total_comp, head_dim), got {tuple(grad_out.shape)}")
    api = _get_api("bwd", kv, score, ape, cu_seqlens, cu_seqlens_comp, tuple(grad_out.shape), ratio, coff)
    with torch.cuda.device(kv.device), _torch_stream_context(stream, kv.device):
        if grad_out.shape[0] == 0:
            # No kernel launch below -> the buffers must carry autograd's exact zeros.
            grad_kv = torch.zeros_like(kv)
            grad_score = torch.zeros_like(score)
        else:
            grad_kv = torch.empty_like(kv)
            grad_score = torch.empty_like(score)
        grad_ape = torch.zeros_like(ape, dtype=torch.float32)
    with torch.cuda.nvtx.range("csa_compressor_bwd_kernel"):
        api.execute(kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out, grad_kv, grad_score, grad_ape, current_stream=stream)
    return TupleDict(grad_kv=grad_kv, grad_score=grad_score, grad_ape=grad_ape)
