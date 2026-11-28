# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation, compilation and launch of the Rubin MXFP8 MoE weight-gradient kernel.

The kernel reads the WGrad operands that training forward/backward already
wrote, in place: no operand is copied, transposed or re-padded.

- ``fc1_a`` / ``fc1_b`` / ``fc2_a`` are token-major (MN-major in the GEMM view)
  and every expert segment is one contiguous block, so dW1 uses the
  ``tensor_ragged`` input order (per-expert TMA bounds).
- ``fc2_b`` is ``(P, H)`` with stride ``(1, P)``: K-major with the global pool
  pitch. Only ``tensor2d`` can address that, and ``tensor2d`` lets the last K
  tile of an expert run past the segment end. The K tile is therefore pinned
  to the token padding, so every segment is a whole number of K tiles.
- Every scale carrier is passed as a zero-copy ``(round_up(rows, 128), P / 32)``
  view of its leading storage; per-expert scale blocks start at
  ``round_up(rows, 128) * expert_begin / 32`` in both producers and the kernel.
- Tiles are chosen so that M and N are whole multiples of the tile and K is
  a whole multiple of the K tile. No TMA box is then clipped, which is the
  condition under which the kernel needs no readable tail after a plane.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import torch

from ..._config import ResolvedMoeEpConfig
from ._arch import architecture_for
from ._cutedsl import require_rubin_wgrad_cutedsl, rubin_wgrad_cutedsl_error

WGRAD_TILE_K = 128
_SF_VEC_SIZE = 32
_WORKSPACE_ALIGNMENT = 128
_MIN_WORKSPACE_BYTES = 128

# (tile_m, tile_n), most preferred first. M = 512 is the B-reuse tile.
_TILE_CANDIDATES = ((512, 256), (256, 256), (256, 128), (128, 128))

_OUTPUT_DTYPES = (torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class Mxfp8WgradGemm:
    """One per-expert WGrad GEMM: ``out[e] (m, n) = A[e] (m, K_e) @ B[e] (K_e, n)``."""

    name: str
    m: int
    n: int
    a_k_major: bool
    b_k_major: bool
    input_order: str
    mma_tiler_mnk: tuple[int, int, int]
    cluster_shape_mn: tuple[int, int]

    @property
    def use_2cta_instrs(self) -> bool:
        return self.mma_tiler_mnk[0] >= 256


@dataclass(frozen=True)
class Mxfp8WgradVariant:
    """Output specialization compiled into one kernel."""

    output_dtype: torch.dtype
    n_major: bool
    storage: str
    accumulate: bool


@dataclass(frozen=True)
class PreparedMxfp8Wgrad:
    device: torch.device
    architecture: tuple[int, int]
    experts: int
    pool_rows: int
    token_padding: int
    fc1: Mxfp8WgradGemm
    fc2: Mxfp8WgradGemm
    local_workspace_bytes: int
    local_workspace_zero_bytes: int

    def gemm(self, name: str) -> Mxfp8WgradGemm:
        return self.fc1 if name == "fc1" else self.fc2

    def abi(self) -> dict[str, object]:
        def gemm_abi(gemm: Mxfp8WgradGemm) -> dict[str, object]:
            return {
                "m": gemm.m,
                "n": gemm.n,
                "a_k_major": gemm.a_k_major,
                "b_k_major": gemm.b_k_major,
                "input_order": gemm.input_order,
                "mma_tiler_mnk": list(gemm.mma_tiler_mnk),
                "cluster_shape_mn": list(gemm.cluster_shape_mn),
            }

        return {
            "pool_rows": self.pool_rows,
            "token_padding": self.token_padding,
            "fc1": gemm_abi(self.fc1),
            "fc2": gemm_abi(self.fc2),
            "local_workspace_bytes": self.local_workspace_bytes,
            "local_workspace_zero_bytes": self.local_workspace_zero_bytes,
        }


@dataclass(frozen=True)
class CompiledMxfp8Wgrad:
    key: tuple
    callable: Any


_COMPILE_LOCK = threading.RLock()
_COMPILE_CACHE: dict[tuple, CompiledMxfp8Wgrad] = {}


def select_wgrad_tile(m: int, n: int) -> tuple[tuple[int, int, int], tuple[int, int]] | None:
    """Pick the first tile whose M and N divide the problem exactly."""

    for tile_m, tile_n in _TILE_CANDIDATES:
        if m % tile_m or n % tile_n:
            continue
        cluster = (2, 1) if tile_m >= 256 else (1, 1)
        return (tile_m, tile_n, WGRAD_TILE_K), cluster
    return None


def wgrad_input_order(a_k_major: bool, b_k_major: bool) -> str:
    """``tensor_ragged`` for per-expert contiguous blocks, else ``tensor2d``.

    ``tensor_ragged`` assumes a K-major operand's pitch is its expert's token
    count, so a K-major operand with the global pool pitch needs ``tensor2d``.
    """

    return "tensor2d" if (a_k_major or b_k_major) else "tensor_ragged"


def _gemm(name: str, m: int, n: int, *, a_k_major: bool, b_k_major: bool) -> Mxfp8WgradGemm:
    selected = select_wgrad_tile(m, n)
    if selected is None:
        raise ValueError(f"no Rubin WGrad tile divides the {name} GEMM ({m} x {n})")
    tile, cluster = selected
    return Mxfp8WgradGemm(
        name=name,
        m=m,
        n=n,
        a_k_major=a_k_major,
        b_k_major=b_k_major,
        input_order=wgrad_input_order(a_k_major, b_k_major),
        mma_tiler_mnk=tile,
        cluster_shape_mn=cluster,
    )


def _cutlass_dtype(dtype: torch.dtype):
    import cutlass

    if dtype is torch.bfloat16:
        return cutlass.BFloat16
    if dtype is torch.float32:
        return cutlass.Float32
    raise ValueError(f"WGrad output dtype must be one of {_OUTPUT_DTYPES}, got {dtype}")


def _build_kernel(
    gemm: Mxfp8WgradGemm,
    *,
    experts: int,
    token_padding: int,
    variant: Mxfp8WgradVariant,
):
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode

    from ...._cutlass_compat import LayoutEnum
    from ..cutedsl_src.api import ImplDesc, ProblemDesc
    from ..cutedsl_src.kernel_src.rubin.training.traditional.wgrad.block_scaled_moe_wgrad_kernel import (
        BlockScaledMoeWgradKernel,
    )

    problem = ProblemDesc(
        {
            "expert_count": experts,
            "hidden": gemm.m,
            "intermediate": gemm.n,
            "a_dtype": cutlass.Float8E4M3FN,
            "b_dtype": cutlass.Float8E4M3FN,
            "sf_dtype": cutlass.Float8E8M0FNU,
            "output_dtype": _cutlass_dtype(variant.output_dtype),
            "acc_dtype": cutlass.Float32,
            "ab_quant_kind": "mxfp8_e4m3",
            "a_major_mode": OperandMajorMode.K if gemm.a_k_major else OperandMajorMode.MN,
            "b_major_mode": OperandMajorMode.K if gemm.b_k_major else OperandMajorMode.MN,
            "output_layout": LayoutEnum.ROW_MAJOR if variant.n_major else LayoutEnum.COL_MAJOR,
            "input_order": gemm.input_order,
            "output_storage_mode": variant.storage,
            "global_scale_mode": "none",
            "accumulate_on_output": variant.accumulate,
            "fix_pad_size": token_padding,
        }
    )
    impl = ImplDesc(
        {
            "mma_tiler_mnk": gemm.mma_tiler_mnk,
            "cluster_shape_mn": gemm.cluster_shape_mn,
            "use_2cta_instrs": gemm.use_2cta_instrs,
        }
    )
    return BlockScaledMoeWgradKernel(problem, impl)


def wgrad_unavailable_reason(architecture: tuple[int, int]) -> str | None:
    """Return why this process cannot run MoeEp WGrad on ``architecture``, or None."""

    kernel_architecture = architecture_for(architecture)
    if not kernel_architecture.supports_training_wgrad:
        return f"MoeEp training WGrad is not supported on {kernel_architecture.name}; " "only the Rubin SM107 WGrad kernel is vendored"
    return rubin_wgrad_cutedsl_error()


def prepare_wgrad(
    resolved_config: ResolvedMoeEpConfig,
    *,
    pool_rows: int,
    token_padding: int,
    sf_padding: int,
    device: torch.device,
    architecture: tuple[int, int],
) -> PreparedMxfp8Wgrad:
    """Plan both WGrad GEMMs; raises when this configuration cannot run them."""

    # The K tile equals the token padding so that tensor2d never reads past a
    # segment; see the module docstring.
    if token_padding != WGRAD_TILE_K or sf_padding != WGRAD_TILE_K:
        raise ValueError(f"MoeEp WGrad requires token and SF padding {WGRAD_TILE_K}, " f"got {token_padding} and {sf_padding}")
    if pool_rows % token_padding:
        raise ValueError(f"WGrad pool rows {pool_rows} are not a multiple of the token padding {token_padding}")
    require_rubin_wgrad_cutedsl()
    model = resolved_config.public_config.model
    hidden = int(model.hidden_size)
    intermediate = int(model.intermediate_size)
    experts = int(resolved_config.topology.experts_per_rank)
    # dW1 = fc1_a^T @ fc1_b: A = (H, P) stride (1, H), B = (P, 2I) stride (2I, 1).
    fc1 = _gemm("fc1", hidden, 2 * intermediate, a_k_major=False, b_k_major=False)
    # dW2 = fc2_a^T @ fc2_b: A = (I, P) stride (1, I), B = (P, H) stride (1, P).
    fc2 = _gemm("fc2", intermediate, hidden, a_k_major=False, b_k_major=True)
    workspace_bytes = 0
    zero_bytes = 0
    default_variant = Mxfp8WgradVariant(
        output_dtype=torch.bfloat16,
        n_major=True,
        storage="dense",
        accumulate=False,
    )
    for gemm in (fc1, fc2):
        kernel = _build_kernel(
            gemm,
            experts=experts,
            token_padding=token_padding,
            variant=default_variant,
        )
        workspace_bytes = max(workspace_bytes, int(kernel.get_workspace_size()))
        zero_bytes = max(zero_bytes, int(kernel.require_zero_workspace_leading_bytes))
    return PreparedMxfp8Wgrad(
        device=torch.device(device),
        architecture=architecture,
        experts=experts,
        pool_rows=int(pool_rows),
        token_padding=int(token_padding),
        fc1=fc1,
        fc2=fc2,
        local_workspace_bytes=max(workspace_bytes, _MIN_WORKSPACE_BYTES),
        local_workspace_zero_bytes=zero_bytes,
    )


def compile_wgrad_or_get(
    prepared: PreparedMxfp8Wgrad,
    gemm: Mxfp8WgradGemm,
    variant: Mxfp8WgradVariant,
) -> CompiledMxfp8Wgrad:
    """Return the cached callable; compiling during capture is an error."""

    key = (
        gemm,
        variant,
        prepared.experts,
        prepared.token_padding,
        prepared.device.index,
        prepared.architecture,
    )
    with _COMPILE_LOCK:
        cached = _COMPILE_CACHE.get(key)
        if cached is not None:
            return cached
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MoeEp WGrad kernel must be compiled before CUDA graph capture; " "run training_wgrad eagerly once with the same outputs first")
        require_rubin_wgrad_cutedsl()
        kernel = _build_kernel(
            gemm,
            experts=prepared.experts,
            token_padding=prepared.token_padding,
            variant=variant,
        )
        if int(kernel.get_workspace_size()) > prepared.local_workspace_bytes:
            raise RuntimeError("Rubin WGrad workspace grew past the prepared region: " f"{kernel.get_workspace_size()} > {prepared.local_workspace_bytes}")
        # Dynamic token extent divisible by the token padding; every other
        # extent is static, so one artifact serves every routing distribution.
        compiled = CompiledMxfp8Wgrad(key=key, callable=kernel.aot_compile())
        _COMPILE_CACHE[key] = compiled
        return compiled


def scale_view(scale: torch.Tensor, rows: int, pool_rows: int) -> torch.Tensor:
    """Zero-copy ``(round_up(rows, 128), pool_rows / 32)`` view of a scale carrier."""

    rows_padded = (rows + 127) // 128 * 128
    columns = pool_rows // _SF_VEC_SIZE
    if tuple(scale.shape) == (rows_padded, columns) and scale.is_contiguous():
        return scale
    if not scale.is_contiguous():
        raise ValueError("WGrad scale carriers must be contiguous")
    flat = scale.view(-1)
    if flat.numel() < rows_padded * columns:
        raise ValueError(f"WGrad scale carrier holds {flat.numel()} elements; " f"{rows_padded * columns} are required")
    return flat[: rows_padded * columns].view(rows_padded, columns)


def discrete_output_view(
    pointers: torch.Tensor,
    dtype: torch.dtype,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """View an int64 pointer table as the kernel's zero-stride output template."""

    return torch.as_strided(pointers.view(dtype), shape, (0, 0, 0))


def launch_wgrad(
    compiled: CompiledMxfp8Wgrad,
    *,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    output: torch.Tensor,
    expert_offsets: torch.Tensor,
    workspace: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    import cuda.bindings.driver as cuda

    address = int(workspace.data_ptr())
    if address % _WORKSPACE_ALIGNMENT:
        raise ValueError(f"WGrad workspace address {address:#x} is not {_WORKSPACE_ALIGNMENT}-byte aligned")
    compiled.callable(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        output,
        expert_offsets,
        address,
        cuda.CUstream(stream.cuda_stream),
        None,
        None,
    )


__all__ = [
    "CompiledMxfp8Wgrad",
    "Mxfp8WgradGemm",
    "Mxfp8WgradVariant",
    "PreparedMxfp8Wgrad",
    "WGRAD_TILE_K",
    "compile_wgrad_or_get",
    "discrete_output_view",
    "launch_wgrad",
    "prepare_wgrad",
    "scale_view",
    "select_wgrad_tile",
    "wgrad_input_order",
    "wgrad_unavailable_reason",
]
