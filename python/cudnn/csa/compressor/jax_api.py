# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX custom-call entry points for the SM100 CSA (ratio 4) and HCA (ratio 128) compressors."""

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
from .compressor_sm100_r128 import (
    _bwd_schedule_r128,
    _compressor_bwd_r128_kernel,
    _compressor_fwd_r128_kernel,
    _fwd_schedule_r128,
    bwd_rows_per_cta_for_sm_count_r128,
)

# SM count assumed when the JAX device does not report one; only affects the
# ratio-128 backward rows-per-CTA choice (performance, not correctness).
_DEFAULT_SM_COUNT = 148


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
def _compressor_bwd_adapter(stream, kv, score, ape, cu, cu_comp, grad_out, grad_kv, grad_score, grad_ape, *, ratio, head_dim, coff):
    flat = cute.make_layout(_EXT)
    tensors = [cute.make_tensor(x.iterator, flat) for x in (kv, score, ape, cu, cu_comp, grad_out, grad_kv, grad_score, grad_ape)]
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


@cute.jit
def _compressor_fwd_r128_adapter(stream, kv, score, ape, cu, cu_comp, out, *, ratio, head_dim, coff, vec, tchunks, threads_x, twophase, fastexp):
    flat = cute.make_layout(_EXT)
    tensors = [cute.make_tensor(x.iterator, flat) for x in (kv, score, ape, cu, cu_comp, out)]
    nb_total = cute.size(out.shape[0])
    n_seq = cute.size(cu.shape[0]) - 1
    ncol = head_dim // vec
    _compressor_fwd_r128_kernel(
        *tensors,
        cutlass.Int32(n_seq),
        ratio,
        head_dim,
        coff,
        vec,
        tchunks,
        threads_x,
        twophase,
        fastexp,
    ).launch(
        grid=(nb_total, (ncol + threads_x - 1) // threads_x, 1),
        block=(threads_x, tchunks, 1),
        stream=stream,
    )


@cute.jit
def _compressor_bwd_r128_adapter(
    stream,
    kv,
    score,
    ape,
    cu,
    cu_comp,
    grad_out,
    grad_kv,
    grad_score,
    grad_ape,
    *,
    ratio,
    head_dim,
    coff,
    vec,
    tchunks,
    threads_x,
    fastexp,
    goreuse,
    rows_per_cta,
):
    flat = cute.make_layout(_EXT)
    tensors = [cute.make_tensor(x.iterator, flat) for x in (kv, score, ape, cu, cu_comp, grad_out, grad_kv, grad_score, grad_ape)]
    nb_total = cute.size(grad_out.shape[0])
    n_seq = cute.size(cu.shape[0]) - 1
    total_tokens = cute.size(kv.shape[0])
    ncol = head_dim // vec
    _compressor_bwd_r128_kernel(
        *tensors,
        cutlass.Int32(nb_total),
        cutlass.Int32(n_seq),
        cutlass.Int32(total_tokens),
        cutlass.Int32(rows_per_cta),
        ratio,
        head_dim,
        coff,
        vec,
        tchunks,
        threads_x,
        fastexp,
        goreuse,
    ).launch(
        grid=((nb_total + rows_per_cta - 1) // rows_per_cta, (ncol + threads_x - 1) // threads_x, 1),
        block=(threads_x, tchunks, 1),
        stream=stream,
    )


def _sm_count() -> int:
    try:
        device = jax.local_devices(backend="gpu")[0]
    except RuntimeError:
        return _DEFAULT_SM_COUNT
    return int(getattr(device, "core_count", None) or _DEFAULT_SM_COUNT)


def _validate(kv: Any, score: Any, ape: Any, cu: Any, cu_comp: Any, ratio: int, coff: int) -> int:
    if ratio == 4:
        if coff != 2:
            raise ValueError("The JAX CSA compressor supports coff=2 at ratio=4")
    elif ratio == 128:
        if coff not in (1, 2):
            raise ValueError(f"The JAX HCA compressor supports coff in {{1, 2}} at ratio=128, got coff={coff}")
    else:
        raise ValueError(f"The JAX compressor supports ratio in {{4, 128}}, got ratio={ratio}")
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
    if ratio == 128 and head_dim not in (128, 512):
        raise ValueError(f"The JAX HCA compressor is validated for head_dim in {{128, 512}}, got {head_dim}")
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
    """Run compressor pooling on JAX arrays.

    ``ratio=4`` is the overlapping CSA compressor (``coff=2``); ``ratio=128`` is the
    HCA compressor (``coff=1`` non-overlapping, or ``coff=2`` overlapping).
    """
    head_dim = _validate(kv, score, ape, cu_seqlens, cu_seqlens_comp, ratio, coff)
    if total_comp < 0:
        raise ValueError(f"total_comp must be non-negative, got {total_comp}")
    out_shape = jax.ShapeDtypeStruct((total_comp, head_dim), jnp.bfloat16)
    if total_comp == 0:
        return jnp.zeros(out_shape.shape, out_shape.dtype)
    if ratio == 128:
        vec, tchunks, threads_x, twophase, fastexp = _fwd_schedule_r128(ratio, head_dim, coff, total_comp)
        return call(
            _compressor_fwd_r128_adapter,
            output_shape_dtype=out_shape,
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
            vec=vec,
            tchunks=tchunks,
            threads_x=threads_x,
            twophase=bool(twophase),
            fastexp=bool(fastexp),
        )(kv, score, ape, cu_seqlens, cu_seqlens_comp)
    return call(
        _compressor_fwd_adapter,
        output_shape_dtype=out_shape,
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
    """Run the explicit compressor backward custom call (ratio 4 or 128)."""
    head_dim = _validate(kv, score, ape, cu_seqlens, cu_seqlens_comp, ratio, coff)
    if grad_out.ndim != 2 or grad_out.shape[1] != head_dim or grad_out.dtype != jnp.bfloat16:
        raise ValueError(f"grad_out must be (total_comp, {head_dim}) bfloat16")
    shapes = (
        jax.ShapeDtypeStruct(kv.shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(score.shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(ape.shape, jnp.float32),
    )
    nb_total = grad_out.shape[0]
    if nb_total == 0:
        return tuple(jnp.zeros(s.shape, s.dtype) for s in shapes)
    if ratio == 128:
        vec, tchunks, threads_x, fastexp, goreuse = _bwd_schedule_r128(ratio, head_dim, coff, nb_total)
        rows_per_cta = bwd_rows_per_cta_for_sm_count_r128(nb_total, ratio, head_dim, coff, _sm_count())
        return call(
            _compressor_bwd_r128_adapter,
            output_shape_dtype=shapes,
            # grad_kv / grad_score are fully written; grad_ape is accumulated.
            initialized_outputs={2: zeros_init},
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
            vec=vec,
            tchunks=tchunks,
            threads_x=threads_x,
            fastexp=bool(fastexp),
            goreuse=bool(goreuse),
            rows_per_cta=rows_per_cta,
        )(kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out)
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
