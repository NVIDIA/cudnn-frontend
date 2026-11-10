# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX training API backed by the shared CuTe kernels."""

from functools import partial
import math

import jax
import jax.numpy as jnp

from cudnn.frost.buffers import cutedsl_requirement_error

from cudnn.jax import call, gemm_operand_spec, zeros_init
from .cutedsl import jax_kernels as kernels


def _shape(shape, dtype):
    return jax.ShapeDtypeStruct(shape, dtype)


def _call(fn, config, **kwargs):
    return call(fn, compile_options=f"--gpu-arch={config[0]}", allow_cuda_graph=False, **kwargs)


def _configuration(arrays, topk, coefficient, triangular):
    if not isinstance(topk, int) or isinstance(topk, bool) or topk not in (512, 1024, 2048):
        raise ValueError("topk must be a static integer in {512, 1024, 2048}")
    if not isinstance(coefficient, (int, float)) or not math.isfinite(coefficient):
        raise ValueError("loss_coeff must be a finite static scalar")
    if not isinstance(triangular, bool):
        raise TypeError("triangular_scores must be a static bool")
    for x in arrays:
        if not isinstance(x, (jax.Array, jax.core.Tracer)):
            raise TypeError("All operands must be JAX arrays")
    q, k, v, iq, ik, starts, ends, valid, denominator = arrays
    if q.ndim != 3 or q.shape[1] not in (16, 32, 96) or q.shape[2] != 256 or min(q.shape) <= 0:
        raise ValueError("query must have nonempty shape [Tq, Hq, 256], with Hq in {16, 32, 96}")
    if k.ndim != 3 or k.shape[1:] != (1, 256) or k.shape[0] <= 0:
        raise ValueError("key must have nonempty shape [Tk, 1, 256]")
    rows, keys = q.shape[0], k.shape[0]
    expected = ((keys, 1, 256), (rows, 128), (keys, 128), (rows,), (rows,), (rows,), (1,))
    for name, x, shape in zip(("value", "q_idx", "k_idx", "key_starts", "key_ends", "query_valid", "loss_denominator"), arrays[2:], expected):
        if x.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {x.shape}")
    types = (jnp.bfloat16,) * 5 + (jnp.int32, jnp.int32, jnp.bool_, jnp.float32)
    if any(x.dtype != dtype for x, dtype in zip(arrays, types)):
        raise TypeError("Use BF16 activations, int32 bounds, bool query_valid, and FP32 loss_denominator")
    if triangular and rows != keys:
        raise ValueError("triangular_scores requires Tq == Tk")
    if max(rows * q.shape[1] * 256, keys * 256, rows * topk) >= 2**31:
        raise ValueError("Tensor extents exceed the kernels' int32 indexing")
    padded_rows, padded_keys = (rows + 127) // 128 * 128, (keys + 255) // 256 * 256
    if (triangular or keys < 131072) and padded_rows * padded_keys >= 2**31:
        raise ValueError("Dense score workspace exceeds int32 indexing")
    placed = {d for x in arrays if not isinstance(x, jax.core.Tracer) for d in x.devices()}
    if len(placed) > 1 or any(d.platform != "gpu" for d in placed):
        raise ValueError("All operands must be on one CUDA device")
    # A traced call can execute on any visible target, including captured-input jits.
    devices = jax.local_devices(backend="gpu")
    archs = {str(getattr(d, "compute_capability", "")) for d in devices}
    if jax.process_count() != 1 or len(archs) != 1 or not archs <= {"10.0", "10.3"}:
        raise ValueError("JAX simplified sparse attention requires one process with only SM100 or only SM103 GPUs visible")
    from .cutedsl.backend import _validate_production_environment

    _validate_production_environment()
    arch = {"10.0": "sm_100a", "10.3": "sm_103a"}[next(iter(archs))]
    return arch, min(d.core_count for d in devices)


def _exact_indexer(q, k, starts, ends, topk, config):
    rows, keys = q.shape[0], k.shape[0]
    qp, kp = (rows + 127) // 128 * 128, (keys + 255) // 256 * 256
    q = jnp.pad(q, ((0, qp - rows), (0, 0)))
    k = jnp.pad(k, ((0, kp - keys), (0, 0)))
    if keys >= 131072:
        gemm = kernels.module("dense_gemm_sm100").DenseGemmKernel(kernels.F32, False, (128, 256), (1, 1), True)
        scores = _call(
            kernels.repair_score,
            config,
            output_shape_dtype=_shape((qp, kp, 1), jnp.float32),
            input_spec=(gemm_operand_spec(), gemm_operand_spec()),
            output_spec=gemm_operand_spec(),
            kernel=gemm,
        )(q[..., None], k[..., None])[..., 0]
    else:
        scores = _call(kernels.score, config, output_shape_dtype=_shape((qp, kp), jnp.float32), kernel=kernels.score_kernel(qp, kp))(q, k)
    return _call(
        kernels.select,
        config,
        output_shape_dtype=(_shape((rows, topk), jnp.int32), _shape((rows, topk), jnp.float32)),
        kernel=kernels.topk_kernel(kp, topk),
        blocks=2 if 49152 <= kp < 131072 else 4,
    )(scores[:rows], starts, ends)


def _indexer(q, k, starts, ends, topk, triangular, config):
    if triangular or k.shape[0] < 131072:
        return _exact_indexer(q, k, starts, ends, topk, config)
    rows, keys = q.shape[0], k.shape[0]
    qp, kp = (rows + 127) // 128 * 128, (keys + 255) // 256 * 256
    q = jnp.pad(q, ((0, qp - rows), (0, 0)))
    k = jnp.pad(k, ((0, kp - keys), (0, 0)))
    starts, ends = (jnp.pad(x, ((0, qp - rows),)) for x in (starts, ends))
    sampler, collector, selector, width = kernels.filtered_kernels(qp, kp, topk, config[1])
    if qp * width >= 2**31:
        raise ValueError("Filtered candidate workspace exceeds int32 indexing")
    parts, counts = sampler.scratch_shapes(qp)
    shapes = (
        _shape((qp, topk), jnp.int32),
        _shape((qp, topk), jnp.float32),
        _shape((qp + 1,), jnp.int32),
        _shape((1, 256), jnp.int32),
        _shape((qp, 2), jnp.int32),
        _shape((qp,), jnp.float32),
        _shape((qp, 2), jnp.int32),
        _shape((qp, width), jnp.int32),
        _shape((qp, width), jnp.float32),
        _shape(parts, jnp.int32),
        _shape(counts, jnp.int32),
        _shape((1,), jnp.uint8),
    )
    indices, predict, flags, *_ = _call(
        kernels.filtered,
        config,
        output_shape_dtype=shapes,
        initialized_outputs={i: zeros_init for i in (0, 1, 2, 3, 4, 5, 6, 11)},
        sampler=sampler,
        collector=collector,
        selector=selector,
        rows=rows,
        cols=kp,
    )(q, k, starts, ends)

    def repair(state):
        offset, indices, predict = state
        ids = jax.lax.dynamic_slice(flags, (offset + 1,), (128,))
        live = offset + jnp.arange(128) < flags[0]
        fq = jnp.where(live[:, None], q[ids], 0)
        fs, fe = jnp.where(live, starts[ids], 0), jnp.where(live, ends[ids], 1)
        fi, fp = _exact_indexer(fq, k, fs, fe, topk, config)
        destinations = jnp.where(live, ids, qp)
        return offset + 128, indices.at[destinations].set(fi, mode="drop"), predict.at[destinations].set(fp, mode="drop")

    # Only failed rows enter exact repair. XLA owns all scratch and loop state.
    _, indices, predict = jax.lax.while_loop(lambda state: state[0] < flags[0], repair, (jnp.int32(0), indices, predict))
    return indices[:rows], predict[:rows]


def _auxiliary(q, k, indices, lengths, predict, teacher, denominator, topk, coefficient, triangular, config):
    aux = kernels.module("indexer_aux_grad_precompute_tcgen05")
    rows, keys = q.shape[0], k.shape[0]
    common = (_shape((2, rows, topk), jnp.bfloat16), _shape((rows,), jnp.float32), _shape((rows, 128), jnp.float32))
    if triangular or keys < aux.GATHER_TK_PER_TOPK * topk:
        outputs = (_shape((1,), jnp.float32), _shape(q.shape, q.dtype), _shape(k.shape, k.dtype), *common, _shape((keys, 128), jnp.float32))
        loss, dq, dk, *_ = _call(
            kernels.auxiliary, config, output_shape_dtype=outputs, launcher=aux._launch, topk=topk, coefficient=coefficient, rectangular=not triangular
        )(q, k, indices, lengths, predict, teacher, denominator)
        return loss[0], dq, dk
    pre_shapes = (*common, _shape(q.shape, q.dtype), _shape((keys,), jnp.int32))
    dz, partial, q_acc, dq, counts = _call(
        kernels.auxiliary_pre, config, output_shape_dtype=pre_shapes, launcher=aux._launch_gather_pre, topk=topk, coefficient=coefficient
    )(k, indices, lengths, predict, teacher, denominator)
    offsets = jnp.concatenate((jnp.zeros(1, jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)))
    # Cursor and dQ are mutable in the native post stage; donation gives them private output storage.
    outputs = (
        _shape((keys,), jnp.int32),
        _shape(q.shape, q.dtype),
        _shape(k.shape, k.dtype),
        _shape((1,), jnp.float32),
        _shape((rows * topk,), jnp.int32),
        _shape((rows * topk,), jnp.float32),
        _shape((1, 128), jnp.float32),
    )
    cursor, dq, dk, loss, *_ = _call(
        kernels.auxiliary_post, config, output_shape_dtype=outputs, input_output_aliases={7: 0, 8: 1}, launcher=aux._launch_gather_post, topk=topk
    )(q, indices, lengths, dz, offsets, partial, q_acc, offsets[:-1], dq)
    return loss[0], dq, dk


def _forward(q, k, v, iq, ik, starts, ends, valid, denominator, topk, coefficient, triangular, config):
    ends = jnp.where(valid, ends, starts)
    lengths = jnp.clip(ends - starts, 0, topk)
    attn_lengths = jnp.maximum(lengths, 1)
    indices, predict = _indexer(iq, ik, starts, ends, topk, triangular, config)
    indices = jnp.where(valid[:, None], indices, 0)
    rows, heads = q.shape[:2]
    from .cutedsl.backend import _ATTENTION_FORWARD_KERNEL_VARIANTS

    mod = kernels.module(_ATTENTION_FORWARD_KERNEL_VARIANTS[(heads, topk)][0])
    plan = mod._plan(heads, topk) if hasattr(mod, "_plan") else ()
    shapes = (
        _shape(q.shape, q.dtype),
        _shape((rows, heads), jnp.float32),
        _shape((rows, topk), jnp.float32),
        _shape((rows, heads), jnp.float32),
        _shape((rows, heads), jnp.float32),
    )
    output, lse, teacher, *_ = _call(
        kernels.attention, config, output_shape_dtype=shapes, launcher=mod._launch, heads=heads, topk=topk, rows=rows, keys=k.shape[0], sms=config[1], plan=plan
    )(q, k, v, indices, attn_lengths)
    output = jnp.where(valid[:, None, None], output, 0)
    loss, diq, dik = _auxiliary(iq, ik, indices, lengths, predict, teacher, denominator, topk, coefficient, triangular, config)
    diq = jnp.where(valid[:, None], diq, 0)
    return (output, loss), (q, k, v, output, lse, indices, attn_lengths, valid, diq, dik)


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12))
def _attention(q, k, v, iq, ik, starts, ends, valid, denominator, topk, coefficient, triangular, config):
    return _forward(q, k, v, iq, ik, starts, ends, valid, denominator, topk, coefficient, triangular, config)[0]


def _backward(topk, coefficient, triangular, config, state, cotangents):
    q, k, v, output, lse, indices, lengths, valid, diq, dik = state
    dout, dloss = cotangents
    dout = jnp.where(valid[:, None, None], dout, 0)
    mod = kernels.module("sparse_attention_backward_tcgen05")
    launcher = mod._launch if q.shape[1] == 96 else mod._stock__launch
    shapes = (_shape(q.shape, q.dtype), _shape(k.shape, k.dtype), _shape(v.shape, v.dtype), _shape((k.shape[0] * 512 + 8,), jnp.float32))
    dq, dk, dv, _ = _call(
        kernels.backward, config, output_shape_dtype=shapes, launcher=launcher, heads=q.shape[1], topk=topk, rows=q.shape[0], keys=k.shape[0], sms=config[1]
    )(q, k, v, output, dout, lse, indices, lengths)
    return dq, dk, dv, diq * dloss.astype(diq.dtype), dik * dloss.astype(dik.dtype), None, None, None, None


_attention.defvjp(_forward, _backward)


def simplified_sparse_attention_jax(
    query, key, value, q_idx, k_idx, key_starts, key_ends, query_valid, loss_denominator, *, topk=512, loss_coeff=0.1, triangular_scores=False
):
    """Return ``(output, indexer_loss)`` with JIT and first-order reverse-mode gradients.

    BF16 Q/K/V and indexer shapes match the PyTorch API. Bounds are int32 [Tq],
    validity is bool [Tq], and the positive global denominator is FP32 [1].
    Supply valid key bounds and finite indexer inputs. Static options select
    top-k and loss scaling; metadata and teacher targets are not differentiated.
    Inputs must use one SM100/SM103 GPU, with one architecture visible per process.

    XLA owns scratch and orders all kernel launches. No reservation or producer
    event is needed. External CUDA graph capture, vmap, and higher derivatives
    are unsupported. Long rectangular inputs use filtered selection and an XLA
    loop for exact repair of failed rows.
    """
    arrays = (query, key, value, q_idx, k_idx, key_starts, key_ends, query_valid, loss_denominator)
    error = cutedsl_requirement_error("simplified_sparse_attention_jax", floor=(4, 8, 0))
    if error:
        raise ImportError(error)
    config = _configuration(arrays, topk, loss_coeff, triangular_scores)
    return _attention(*arrays, topk, float(loss_coeff), triangular_scores, config)
