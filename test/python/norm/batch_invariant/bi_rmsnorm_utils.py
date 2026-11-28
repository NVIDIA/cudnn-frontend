# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graphs, inputs and the exact CPU model for the batch-invariant RMSNorm tests."""

import numpy as np
import torch

import cudnn
from cudnn.norm.batch_invariant import RmsNormClass, class_for

TAGS = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
EPS = 1e-6


def adversarial(rows, hidden, dtype, seed, device):
    """Values over a wide exponent range with cancellation-free squares, outliers, signed zeros and subnormals."""
    g = torch.Generator().manual_seed(seed)
    half = dtype == torch.float16  # keep values and their squares' scale inside the FP16 range
    e = (torch.rand(rows, hidden, generator=g) * 2 - 1) * (6 if half else 12)
    x = (torch.rand(rows, hidden, generator=g) * 2 - 1) * torch.exp2(e)
    x[torch.randint(0, 97, (rows, hidden), generator=g) == 0] *= 2.0 ** (4 if half else 10)
    z = torch.randint(0, 211, (rows, hidden), generator=g)
    x[z == 0] = 0.0
    x[z == 1] = -0.0
    x[z == 2] = 2.0**-20 if half else 2.0**-130
    return x.to(dtype).to(device)


def weight(hidden, dtype, seed, device):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(hidden, generator=g) * 1.5 + 0.25).to(dtype).to(device)


def _packed(shape):
    strides, step = [], 1
    for size in reversed(shape):
        strides.insert(0, step)
        step *= size
    return strides


def build(x, w, *, training=False, out_dtype=None, note=True, eps=EPS, override=False):
    """A built graph and ``run(rows=None) -> {"y", ["rstd"]}`` over freshly allocated, packed outputs.

    With ``override`` the graph is built for ``x`` and ``run(rows)`` normalizes the leading ``rows`` rows of a 2-D
    ``x`` through execute-time shape overrides of X, Y and the inverse RMS.
    """
    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=override,
    )
    xt = graph.tensor(name="X", dim=x.size(), stride=x.stride(), data_type=x.dtype)
    scale = w.view(*([1] * (x.dim() - 1)), -1)
    wt = graph.tensor(name="scale", dim=scale.size(), stride=scale.stride(), data_type=w.dtype)
    eps_cpu = torch.full((1,) * x.dim(), eps, dtype=torch.float32)
    et = graph.tensor_like(eps_cpu, name="epsilon")
    phase = cudnn.norm_forward_phase.TRAINING if training else cudnn.norm_forward_phase.INFERENCE
    y, inv = graph.rmsnorm(name="rmsnorm", norm_forward_phase=phase, input=xt, scale=wt, bias=None, epsilon=et)
    out_dtype = out_dtype or x.dtype
    y_strides = _packed(x.shape)
    y.set_output(True).set_data_type(out_dtype).set_dim(list(x.shape)).set_stride(y_strides)
    stats_shape = (*x.shape[:-1], 1)
    if inv is not None:
        inv.set_output(True).set_data_type(torch.float32).set_dim(list(stats_shape)).set_stride(_packed(stats_shape))
    if note:
        graph.select_numeric_notes([cudnn.numerical_note.BATCH_INVARIANT])
    graph.build([cudnn.heur_mode.A])

    def run(rows=None):
        shape, stats = x.shape, stats_shape
        overrides = {}
        if rows is not None:
            shape, stats = (rows, x.shape[1]), (rows, 1)
            ports = [xt, y] + ([inv] if inv is not None else [])
            overrides = dict(
                override_uids=[t.get_uid() for t in ports],
                override_shapes=[list(shape), list(shape)] + ([list(stats)] if inv is not None else []),
                override_strides=[list(x.stride()), _packed(shape)] + ([_packed(stats)] if inv is not None else []),
            )
        out = {"y": torch.empty_strided(shape, _packed(shape), dtype=out_dtype, device=x.device).fill_(float("nan"))}
        pack = {xt: x if rows is None else x[:rows], wt: scale, et: eps_cpu, y: out["y"]}
        if inv is not None:
            out["rstd"] = torch.full(stats, float("nan"), dtype=torch.float32, device=x.device)
            pack[inv] = out["rstd"]
        graph.execute(pack, torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device=x.device), **overrides)
        return out

    return graph, run


def exact(x, w, out_dtype=None, eps=EPS):
    """The class's exact result: ``y`` rounded to the output dtype and fp32 ``rstd``, both on the input's device."""
    hidden = x.shape[-1]
    cls = class_for(hidden, TAGS[x.dtype])
    y, rstd = reference(cls, x.reshape(-1, hidden).float().cpu().numpy(), w.float().cpu().numpy(), eps)
    out_dtype = out_dtype or x.dtype
    y = torch.from_numpy(y).to(out_dtype).reshape(x.shape).to(x.device)
    return y, torch.from_numpy(rstd).reshape(*x.shape[:-1], 1).to(x.device)


# The exact CPU model of the class (the arithmetic defined in ``cudnn.norm.batch_invariant._spec``).
#
# fp32 add, mul, div and sqrt are computed in fp64 and rounded once to fp32.
# fp64 has more than 2 * 24 + 2 significand bits, so the double rounding is
# innocuous and each result is the correctly rounded fp32 operation. fma is
# computed with an exact fp64 product and a TwoSum, and corrected where the
# fp64 sum lands exactly on an fp32 midpoint.


def _f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).astype(np.float32)


def add_rn(a, b) -> np.ndarray:
    return _f32(np.asarray(a, np.float64) + np.asarray(b, np.float64))


def mul_rn(a, b) -> np.ndarray:
    return _f32(np.asarray(a, np.float64) * np.asarray(b, np.float64))


def div_rn(a, b) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return _f32(np.asarray(a, np.float64) / np.asarray(b, np.float64))


def sqrt_rn(a) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return _f32(np.sqrt(np.asarray(a, np.float64)))


def fma_rn(a, b, c) -> np.ndarray:
    """Correctly rounded fp32 ``a * b + c``."""
    a64, b64, c64 = (np.asarray(v, np.float32).astype(np.float64) for v in (a, b, c))
    with np.errstate(over="ignore", invalid="ignore"):
        p = a64 * b64  # exact: two 24-bit significands
        s = p + c64
        bb = s - p
        err = (p - (s - bb)) + (c64 - bb)  # s + err == p + c exactly
        r = s.astype(np.float32)
        r64 = r.astype(np.float64)
        toward = np.nextafter(r, np.where(s > r64, np.float32(np.inf), np.float32(-np.inf)).astype(np.float32))
        mid = (r64 + toward.astype(np.float64)) * 0.5
        on_mid = np.isfinite(s) & (r64 != s) & (s == mid) & (err != 0)
        # At an exact fp32 midpoint the sign of the TwoSum error decides the rounding.
        up = toward > r
        pick_toward = on_mid & (((err > 0) & up) | ((err < 0) & ~up))
        r = np.where(pick_toward, toward, r)
        # s exactly representable as fp32 but err nonzero: the true value lies off the grid point by less than half an fp32 ulp.
    return r.astype(np.float32)


def sum_of_squares(cls: RmsNormClass, x: np.ndarray) -> np.ndarray:
    """``ssq`` of every row of ``x`` (fp32 values, shape ``[rows, H]``) in the class order."""
    rows = x.shape[0]
    xg = np.asarray(x, dtype=np.float32).reshape(rows, cls.granules, cls.vector)
    acc = np.zeros((rows, cls.lanes), dtype=np.float32)
    for j in range(cls.granules_per_lane):
        g = np.arange(cls.lanes) + j * cls.lanes
        valid = g < cls.granules
        gi = np.where(valid, g, 0)
        for e in range(cls.vector):
            v = xg[:, gi, e]
            acc = np.where(valid[None, :], fma_rn(v, v, acc), acc)
    lane = np.arange(cls.lanes)
    offset = 1
    while offset < cls.group:
        acc = add_rn(acc, acc[:, lane ^ offset])
        offset *= 2
    total = acc[:, 0]
    for k in range(1, cls.groups):
        total = add_rn(total, acc[:, k * cls.group])
    return total


def reference(cls: RmsNormClass, x: np.ndarray, w: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """fp32 output before the final rounding, and ``rstd``, for rows ``x`` ``[rows, H]`` and weight ``w`` ``[H]``.

    ``x`` and ``w`` hold the fp32 widenings of the operands; round the first
    result to the output dtype with round-to-nearest-even.
    """
    ssq = sum_of_squares(cls, x)
    mean = div_rn(ssq, np.float32(cls.hidden))
    rstd = div_rn(np.float32(1), sqrt_rn(add_rn(mean, np.float32(eps))))
    y = mul_rn(mul_rn(x, rstd[:, None]), np.asarray(w, np.float32)[None, :])
    return y, rstd
