# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX custom-call entry points for the SM100 ratio-4 CSA compressor."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute

from cudnn.jax import call, zeros_init
from .compressor_sm100 import (
    _BWD_ROWS,
    _BWD_THREADS,
    _EXT,
    _compressor_bwd_kernel,
    _compressor_fwd_kernel,
    _fwd_schedule,
)


@cute.jit
def _compressor_fwd_adapter(stream, kv, score, ape, cu, cu_comp, out, *, ratio, head_dim, coff):
    flat = cute.make_layout(_EXT)
    tensors = [cute.make_tensor(x.iterator, flat) for x in (kv, score, ape, cu, cu_comp, out)]
    vec, rows_per_cta, threads = _fwd_schedule(head_dim)
    ncol = head_dim // vec
    nb_total = cute.size(out.shape[0])
    n_seq = cute.size(cu.shape[0]) - 1
    _compressor_fwd_kernel(
        *tensors,
        cutlass.Int32(nb_total),
        cutlass.Int32(n_seq),
        ratio,
        head_dim,
        coff,
        vec,
        rows_per_cta,
        threads,
    ).launch(
        grid=((nb_total + rows_per_cta - 1) // rows_per_cta, (ncol + threads - 1) // threads, 1),
        block=(threads, 1, 1),
        stream=stream,
    )


@cute.jit
def _compressor_bwd_adapter(
    stream, kv, score, ape, cu, cu_comp, grad_out, grad_kv, grad_score, grad_ape, *, ratio, head_dim, coff
):
    flat = cute.make_layout(_EXT)
    tensors = [
        cute.make_tensor(x.iterator, flat)
        for x in (kv, score, ape, cu, cu_comp, grad_out, grad_kv, grad_score, grad_ape)
    ]
    nb_total = cute.size(grad_out.shape[0])
    n_seq = cute.size(cu.shape[0]) - 1
    total_tokens = cute.size(kv.shape[0])
    _compressor_bwd_kernel(
        *tensors,
        cutlass.Int32(nb_total),
        cutlass.Int32(n_seq),
        cutlass.Int32(total_tokens),
        ratio,
        head_dim,
        coff,
        _BWD_ROWS,
        _BWD_THREADS,
    ).launch(
        grid=((nb_total + _BWD_ROWS - 1) // _BWD_ROWS, (head_dim + _BWD_THREADS - 1) // _BWD_THREADS, 1),
        block=(_BWD_THREADS, 1, 1),
        stream=stream,
    )


def _validate(kv: Any, score: Any, ape: Any, cu: Any, cu_comp: Any, ratio: int, coff: int) -> int:
    if ratio != 4 or coff != 2:
        raise ValueError("The JAX CSA compressor currently supports ratio=4 and coff=2 only")
    if kv.ndim != 2 or score.shape != kv.shape:
        raise ValueError(f"kv and score must have the same 2-D shape, got {kv.shape} and {score.shape}")
    if kv.dtype != jnp.bfloat16 or score.dtype != jnp.bfloat16:
        raise TypeError("kv and score must be bfloat16")
    width = kv.shape[1]
    if width % coff:
        raise ValueError(f"kv width {width} is not divisible by coff={coff}")
    head_dim = width // coff
    if ape.shape != (ratio, width) or ape.dtype != jnp.float32:
        raise ValueError(f"ape must have shape {(ratio, width)} and dtype float32")
    if cu.ndim != 1 or cu_comp.shape != cu.shape or cu.dtype != jnp.int32 or cu_comp.dtype != jnp.int32:
        raise ValueError("cu_seqlens and cu_seqlens_comp must be matching 1-D int32 arrays")
    return head_dim


def csa_compressor_forward_jax_sm100(
    kv: Any,
    score: Any,
    ape: Any,
    cu_seqlens: Any,
    cu_seqlens_comp: Any,
    *,
    total_comp: int,
    ratio: int = 4,
    coff: int = 2,
) -> Any:
    """Run ratio-4 overlapping CSA pooling on JAX arrays."""
    head_dim = _validate(kv, score, ape, cu_seqlens, cu_seqlens_comp, ratio, coff)
    if total_comp < 0:
        raise ValueError(f"total_comp must be non-negative, got {total_comp}")
    return call(
        _compressor_fwd_adapter,
        output_shape_dtype=jax.ShapeDtypeStruct((total_comp, head_dim), jnp.bfloat16),
        ratio=ratio,
        head_dim=head_dim,
        coff=coff,
    )(kv, score, ape, cu_seqlens, cu_seqlens_comp)


def csa_compressor_backward_jax_sm100(
    kv: Any,
    score: Any,
    ape: Any,
    cu_seqlens: Any,
    cu_seqlens_comp: Any,
    grad_out: Any,
    *,
    ratio: int = 4,
    coff: int = 2,
) -> tuple[Any, Any, Any]:
    """Run the explicit ratio-4 CSA compressor backward custom call."""
    head_dim = _validate(kv, score, ape, cu_seqlens, cu_seqlens_comp, ratio, coff)
    if grad_out.ndim != 2 or grad_out.shape[1] != head_dim or grad_out.dtype != jnp.bfloat16:
        raise ValueError(f"grad_out must be (total_comp, {head_dim}) bfloat16")
    shapes = (
        jax.ShapeDtypeStruct(kv.shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(score.shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(ape.shape, jnp.float32),
    )
    return call(
        _compressor_bwd_adapter,
        output_shape_dtype=shapes,
        initialized_outputs={0: zeros_init, 1: zeros_init, 2: zeros_init},
        ratio=ratio,
        head_dim=head_dim,
        coff=coff,
    )(kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out)


__all__ = [
    "csa_compressor_forward_jax_sm100",
    "csa_compressor_backward_jax_sm100",
]
