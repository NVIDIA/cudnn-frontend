# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX execution of the native Cake mHC training stages."""

from functools import lru_cache, partial
import math

import jax
import jax.numpy as jnp
import numpy as np


def _validate(value, shape, dtype, name):
    if value.shape != shape or value.dtype != jnp.dtype(dtype):
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")


def _geometry(x, head_backward=False):
    ndim = 2 if head_backward else 3
    if x.ndim != ndim or (ndim == 3 and x.shape[1] != 4):
        raise ValueError("mHC requires x[T,4,H], or grad_out[T,H] for head backward")
    t, h = x.shape[0], x.shape[-1]
    if t < 1 or h not in (4096, 5120, 7168) or t * 4 * h >= 2**31:
        raise NotImplementedError("mHC requires positive rows, H=4096/5120/7168, and int32 element offsets")
    _validate(x, x.shape, "bfloat16", "x")
    return t, h


def _constants(rms_eps, hc_eps, sinkhorn_iters):
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (rms_eps, hc_eps)):
        raise ValueError("rms_eps and hc_eps must be finite positive host scalars")
    if isinstance(sinkhorn_iters, bool) or not isinstance(sinkhorn_iters, int) or not 1 <= sinkhorn_iters <= 64:
        raise ValueError("sinkhorn_iters must be a host integer in [1,64]")
    return float(rms_eps), float(hc_eps), sinkhorn_iters


def _device(arrays):
    devices = {d for x in arrays if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer) for d in x.devices()}
    if len(devices) > 1 or jax.process_count() != 1:
        raise NotImplementedError("JAX mHC requires one GPU in one process")
    device = devices.pop() if devices else (jax.config.jax_default_device or jax.local_devices()[0])
    if device.platform != "gpu":
        raise NotImplementedError("JAX mHC requires a CUDA GPU")
    return device.local_hardware_id


@lru_cache(maxsize=128)
def _build(kind, metadata, t, h, config, device):
    from cudnn._cake import loader
    from cudnn._cake.jax import TensorRef, make_call

    arch = loader.device_arch(device)
    loader.check_toolkit(arch, "JAX mHC")
    if arch == "sm_107a" and h != 4096:
        raise NotImplementedError("SM107 mHC requires H=4096")
    head = kind.startswith("head_")
    b = loader.load_package("cake_mhc_head_det_train" if head else "cake_mhc_det_train")
    names = {
        "fwd_prepare": ("u", "stats"),
        "fwd_finish": ("y",),
        "bwd_prepare": ("db", "gstats"),
        "bwd_finish": ("dx", "dfn", "dscale", "dbase"),
        "head_fwd": ("y",),
        "head_bwd": ("dx",),
    }[kind]
    specs = {
        "u": ((t, h), "bfloat16"),
        "stats": ((t, 64), "float32"),
        "y": ((t, h) if head else (t, 4, h), "bfloat16"),
        "db": ((t, h), "bfloat16"),
        "gstats": ((t, 32), "float32"),
        "dx": ((t, 4, h), "bfloat16"),
        "dfn": ((24, 4 * h), "float32"),
        "dscale": ((3,), "float32"),
        "dbase": ((24,), "float32"),
    }
    values = {name: TensorRef(i, shape, dtype) for i, (name, shape, dtype) in enumerate(metadata)}
    count = len(metadata)
    outputs = [jax.ShapeDtypeStruct(*specs[name]) for name in names]
    values.update({name: TensorRef(count + i, *specs[name]) for i, name in enumerate(names)})
    values["rows"] = t
    zeros = ()
    if head:
        name = b._loader.ROLES[kind]
        record = b._loader.MODULES[name]
        grid = dict(zip(("grid_x", "grid_y", "grid_z"), b._loader.launch_grid(record, t, h)))
        values["hv"] = b._loader.vectors_per_token(record, h)
        args = tuple(grid[key] if argkind == "grid" else values[record["logical"][key]] for argkind, key in record["arg_plan"])
        stages = [(getattr(b._loader.load(name, arch), record["ffi_entry"]), args)]
    else:
        rms_eps, hc_eps, iters = config
        sm_count = b._loader.sm_count(device)
        values.update(b._loader.host_plan(t, sm_count, arch, h))
        values.update(rms_eps=rms_eps, hc_eps=hc_eps, sinkhorn_iters=iters, accumulate=0)
        for name in ("x", "dy", "dx"):
            if name in values:
                ref = values[name]
                values[name + "_flat"] = TensorRef(ref.index, (t, 4 * h), "bfloat16")
        if kind in ("fwd_prepare", "bwd_finish"):
            regions = b._loader.workspace_regions("fwd" if kind == "fwd_prepare" else "bwd", t, sm_count, arch, h)
            workspace_index = count + len(outputs)
            outputs.append(jax.ShapeDtypeStruct((regions["total_bytes"],), "uint8"))
            for name, region in regions["regions"].items():
                values[name] = TensorRef(workspace_index, tuple(region["shape"]), region["dtype"], offset=region["offset"])
            zeros = (TensorRef(workspace_index, (256,), "uint8"),)
            if kind == "fwd_prepare":
                values["fnTp"] = values["fnp"]
            else:
                values["fnp"] = values["fnTp"]
                ref = values["dzp"]
                values["dzp_rows"] = TensorRef(ref.index, (t, *ref.shape[1:]), ref.dtype, offset=ref.offset)
        path = b._PATH_FORWARD if kind.startswith("fwd") else b._PATH_BACKWARD
        stages = []
        for launch in b._route_plan(arch, kind, t, h, device, path, iters):
            args = list(launch.template)
            for i, name in launch.slots:
                args[i] = values[name]
            stages.append((launch.run, tuple(args)))
    return make_call(stages=stages, input_count=count, output_shape_dtype=tuple(outputs), device=device, zero_buffers=zeros), len(names)


def _execute(kind, values, config=(1e-5, 1e-6, 20)):
    x = next(iter(values.values()))
    t, h = _geometry(x, kind == "head_bwd")
    metadata = tuple((name, tuple(value.shape), np.dtype(value.dtype).name) for name, value in values.items())
    invoke, count = _build(kind, metadata, t, h, config, _device(values.values()))
    return tuple(invoke(*values.values()))[:count]


def mhc_prepare_jax(x, fn, scale, base, *, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20):
    """Return ``(branch_input, stats)``; use mhc_jax for automatic differentiation."""
    _, h = _geometry(x)
    for value, shape, name in ((fn, (24, 4 * h), "fn"), (scale, (3,), "scale"), (base, (24,), "base")):
        _validate(value, shape, "float32", name)
    config = _constants(rms_eps, hc_eps, sinkhorn_iters)
    return _execute("fwd_prepare", dict(x=x, fn=fn, scale=scale, base=base), config)


def mhc_finish_jax(x, branch, stats):
    """Return the mixed residual streams as a one-element tuple."""
    t, h = _geometry(x)
    _validate(branch, (t, h), "bfloat16", "branch")
    _validate(stats, (t, 64), "float32", "stats")
    return _execute("fwd_finish", dict(x=x, b=branch, stats=stats))


def mhc_backward_prepare_jax(x, branch, stats, grad_out):
    """Return ``(grad_branch, grad_stats)`` for explicit first-order backward."""
    t, h = _geometry(x)
    for value, shape, dtype, name in (
        (branch, (t, h), "bfloat16", "branch"),
        (stats, (t, 64), "float32", "stats"),
        (grad_out, x.shape, "bfloat16", "grad_out"),
    ):
        _validate(value, shape, dtype, name)
    return _execute("bwd_prepare", dict(x=x, b=branch, stats=stats, dy=grad_out))


def mhc_backward_finish_jax(x, grad_out, grad_u, fn, scale, base, stats, grad_stats, *, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20):
    """Return ``(dx, dfn, dscale, dbase)`` after the external branch backward."""
    t, h = _geometry(x)
    for value, shape, dtype, name in (
        (grad_out, x.shape, "bfloat16", "grad_out"),
        (grad_u, (t, h), "bfloat16", "grad_u"),
        (fn, (24, 4 * h), "float32", "fn"),
        (scale, (3,), "float32", "scale"),
        (base, (24,), "float32", "base"),
        (stats, (t, 64), "float32", "stats"),
        (grad_stats, (t, 32), "float32", "grad_stats"),
    ):
        _validate(value, shape, dtype, name)
    config = _constants(rms_eps, hc_eps, sinkhorn_iters)
    return _execute("bwd_finish", dict(x=x, dy=grad_out, du=grad_u, fn=fn, scale=scale, base=base, stats=stats, gstats=grad_stats), config)


def mhc_head_backward_jax(grad_out):
    """Return the BF16 broadcast gradient as a one-element tuple."""
    return _execute("head_bwd", dict(dy=grad_out))


@jax.custom_vjp
def mhc_head_forward_jax(x):
    """Return the four-stream mean as a one-element tuple; supports reverse AD."""
    return _execute("head_fwd", dict(x=x))


def _head_fwd(x):
    return _execute("head_fwd", dict(x=x)), None


def _head_bwd(residual, gradients):
    return mhc_head_backward_jax(gradients[0])


mhc_head_forward_jax.defvjp(_head_fwd, _head_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _connection(branch_fn, config, x, fn, scale, base, branch_args):
    return _connection_fwd(branch_fn, config, x, fn, scale, base, branch_args)[0]


def _connection_fwd(branch_fn, config, x, fn, scale, base, branch_args):
    opts = dict(zip(("rms_eps", "hc_eps", "sinkhorn_iters"), config))
    u, stats = mhc_prepare_jax(x, fn, scale, base, **opts)
    branch = branch_fn(u, *branch_args)
    (y,) = mhc_finish_jax(x, branch, stats)
    return y, (x, fn, scale, base, branch_args, u, branch, stats)


def _connection_bwd(branch_fn, config, residual, dy):
    x, fn, scale, base, branch_args, u, branch, stats = residual
    db, gs = mhc_backward_prepare_jax(x, branch, stats, dy)
    _, pullback = jax.vjp(branch_fn, u, *branch_args)
    du, *dargs = pullback(db)
    opts = dict(zip(("rms_eps", "hc_eps", "sinkhorn_iters"), config))
    dx, dfn, ds, dbase = mhc_backward_finish_jax(x, dy, du, fn, scale, base, stats, gs, **opts)
    return dx, dfn, ds, dbase, tuple(dargs)


_connection.defvjp(_connection_fwd, _connection_bwd)


def mhc_jax(x, fn, scale, base, branch_fn, *branch_args, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20):
    """Apply mHC around ``branch_fn(u, *branch_args) -> BF16[T,H]``.

    Supports jit and first-order reverse AD, including explicit branch arguments.
    The branch and scalar options are static. Forward AD, vmap, higher derivatives,
    sharding, and differentiation of the explicit stages are unsupported.
    """
    return _connection(branch_fn, _constants(rms_eps, hc_eps, sinkhorn_iters), x, fn, scale, base, branch_args)
