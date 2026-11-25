# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward Conv3D/post-op replay, batch invariance, and persistent warm stress."""

import math
from functools import partial

import cudnn
import pytest
import torch
import torch.nn.functional as F

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from test_conv3d_postops import requires_cutedsl, requires_gpu
from warm_stress import StressCase, assert_warm_stress, stress_validation

pytestmark = [pytest.mark.L0, requires_cutedsl, requires_gpu]
DTYPES = [pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float32, id="fp32")]
# Distribute tile widths across API/dtype cases. Odd spatial sizes put batch
# boundaries inside partial M tiles.
CASES = [
    ("raw", 512, 256, torch.bfloat16),
    ("raw", 320, 640, torch.float32),
    ("norm", 96, 96, torch.bfloat16),
    ("norm", 192, 192, torch.float32),
    ("norm_pad", 192, 192, torch.bfloat16),
    ("norm_pad", 320, 320, torch.float32),
    ("spatial", 96, 96, torch.bfloat16),
    ("spatial", 192, 192, torch.float32),
    ("post", 1024, 1024, torch.bfloat16),
    ("post", 96, 96, torch.float32),
    ("causal", 12, 160, torch.bfloat16),
    ("causal", 12, 160, torch.float32),
]
GOLDENS = [
    ("raw", 96, 96),
    ("norm", 96, 96),
    ("norm_pad", 192, 192),
    ("spatial", 96, 96),
    ("post", 96, 96),
    ("causal", 12, 160),
]
BATCH_INPUTS = ("x", "residual", "previous")


def _inputs(kind, ci, co, dtype, *, shape=(3, 1, 7, 9), history=2):
    n, t, h, w = shape
    generator = torch.Generator().manual_seed(1049 + ci + co)

    def rand(shape, scale=0.1):
        # CPU generation keeps the data independent of GPU SM count.
        return (torch.randn(shape, generator=generator) * scale).to(device="cuda", dtype=dtype)

    xshape = (n, ci, t, h, w) if kind == "causal" else (n, t, h, w, ci) if kind == "post" else (n, t + 2, h + 2, w + 2, ci)
    data = {"x": rand(xshape), "previous": rand((n, history, h, w, ci if kind == "causal" else co))}
    if kind != "post":
        data["weight"] = rand((co, ci, 3, 3, 3), (27 * ci) ** -0.5)
        pack = cudnn.pack_causal_conv3d_weight_sm100 if kind == "causal" else cudnn.pack_conv3d_weight_sm100
        data["packed"] = pack(data["weight"], allow_tf32=dtype == torch.float32)
    if kind not in ("raw", "causal"):
        data.update(bias=rand((co,)), gamma=rand((co,), 1), residual=rand((n, t, h, w, co)), residual_bias=rand((co,)))
    return data


def _batch(data, order):
    return {name: value[order].clone() if name in BATCH_INPUTS else value for name, value in data.items()}


def _run(kind, data):
    x, previous = data["x"], data["previous"]
    tf32 = x.dtype == torch.float32
    if kind == "raw":
        return {"output": cudnn.conv3d_raw_wrapper_sm100(x, data["packed"], allow_tf32=tf32)}
    if kind == "causal":
        return dict(cudnn.causal_conv3d_with_cache_wrapper_sm100(x, data["packed"], previous, allow_tf32=tf32))
    bias, gamma, residual, residual_bias = (data[name] for name in ("bias", "gamma", "residual", "residual_bias"))
    if kind == "post":
        return dict(cudnn.rmsnorm_silu_pad_wrapper_sm100(x, gamma, bias, previous, residual, residual_bias))
    if kind == "spatial":
        return dict(cudnn.conv3d_bias_residual_pad_wrapper_sm100(x, data["packed"], bias, residual, residual_bias=residual_bias, allow_tf32=tf32))
    if kind == "norm":
        return dict(cudnn.conv3d_rmsnorm_silu_wrapper_sm100(x, data["packed"], bias, gamma, residual, residual_bias, allow_tf32=tf32))
    result = dict(
        cudnn.conv3d_rmsnorm_silu_pad_wrapper_sm100(
            x,
            data["packed"],
            bias,
            gamma,
            history_frames=previous.shape[1],
            residual=residual,
            residual_bias=residual_bias,
            allow_tf32=tf32,
        )
    )
    # Fused convolution deliberately leaves activated history owned by callers.
    # Fill only those regions; every other output byte must come from the kernel.
    history, frames = previous.shape[1], x.shape[1] - 2
    if history:
        result["padded_output"][:, 2 - history : 2, 1:-1, 1:-1].copy_(previous)
        if frames == 1:
            result["cache_output"][:, :1].copy_(previous[:, -1:])
    return result


def _tf32(value):
    bits = value.contiguous().view(torch.int32)
    return ((bits + 4095 + ((bits >> 13) & 1)) & -8192).view(torch.float32)


def _reference(kind, data):
    x, previous = data["x"], data["previous"]
    dtype = x.dtype
    if kind == "causal":
        joined = torch.cat((previous, x.permute(0, 2, 3, 4, 1)), dim=1)
        x = F.pad(joined, (0, 0, 1, 1, 1, 1, 2 - previous.shape[1], 0))
    if kind != "post":
        weight = data["weight"]
        if dtype == torch.float32:
            x, weight = _tf32(x), _tf32(weight)
        with torch.backends.cudnn.flags(allow_tf32=False):
            value = F.conv3d(x.permute(0, 4, 1, 2, 3).float(), weight.float()).permute(0, 2, 3, 4, 1).to(dtype)
    else:
        value = x
    if kind == "raw":
        return {"output": value}
    if kind == "causal":
        return {"output": value, "cache_output": joined[:, -2:]}
    value = (value.float() + data["bias"].float()).to(dtype)
    value = (value.float() + (data["residual"].float() + data["residual_bias"].float()).to(dtype).float()).to(dtype)
    if kind == "spatial":
        return {"padded_output": F.pad(value, (0, 0, 0, 1, 0, 1))}
    normalized = F.normalize(value.float(), dim=-1).to(dtype)
    affine = ((normalized.float() * math.sqrt(value.shape[-1])).to(dtype).float() * data["gamma"].float()).to(dtype)
    activated = F.silu(affine.float()).to(dtype)
    if kind == "norm":
        return {"output": activated, "residual_output": value}
    joined = torch.cat((previous, activated), dim=1)
    return {"padded_output": F.pad(joined, (0, 0, 1, 1, 1, 1, 2 - previous.shape[1], 0)), "cache_output": joined[:, -2:], "residual_output": value}


def _check_reference(kind, data, actual):
    expected = _reference(kind, data)
    assert actual.keys() == expected.keys()
    # Check numerical accuracy against the independent Torch reference.
    tol = 0.04 if data["x"].dtype == torch.bfloat16 else 2e-4
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=tol, rtol=tol)


@pytest.mark.parametrize("kind,ci,co,dtype", CASES, ids=[f"fprop-{k}-c{i}-{o}-{dtype}-t1-h7-w9" for k, i, o, dtype in CASES])
@torch.inference_mode()
def test_forward_replay_and_batch(kind, ci, co, dtype):
    data = _inputs(kind, ci, co, dtype)
    original = snapshot(data)
    expected = assert_replays(lambda: _run(kind, data))
    _check_reference(kind, data, expected)
    for order in ([0], [2, 0, 1], [0, 0, 2], [2, 0], [1, 0, 2, 0]):
        selected = _batch(data, order)
        before = snapshot(selected)
        actual = snapshot(_run(kind, selected))
        assert_bitwise(actual, {name: value[order] for name, value in expected.items()}, f"batch {order}")
        assert_bitwise(selected, before, "input mutation")
    changed = _batch(data, [0, 1, 2])
    for name in BATCH_INPUTS:
        if name in changed:
            changed[name][1:].mul_(-0.5)
    actual = snapshot(_run(kind, changed))
    assert_bitwise({name: value[:1] for name, value in actual.items()}, {name: value[:1] for name, value in expected.items()}, "changed neighbors")
    assert_bitwise(data, original, "input mutation")


@pytest.mark.parametrize("kind,ci,co", GOLDENS, ids=[f"fprop-{k}-c{i}-{o}-t3-h5-w7" for k, i, o in GOLDENS])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_forward_golden(kind, ci, co, dtype):
    data = _inputs(kind, ci, co, dtype, shape=(1, 3, 5, 7), history=1)
    outputs = assert_replays(lambda: _run(kind, data), repeats=2)
    _check_reference(kind, data, outputs)
    assert_golden(f"conv3d_postops_fprop_{kind}_{ci}_{co}_{str(dtype).split('.')[-1]}", data, outputs)


@pytest.mark.parametrize(
    "kind,dtype",
    [("raw", torch.float32), ("norm_pad", torch.bfloat16)],
    ids=["fprop-raw-tf32-c192-t3-h97-w121", "fprop-norm-pad-bf16-c192-t3-h97-w121"],
)
@torch.inference_mode()
def test_forward_persistent_warm_stress(kind, dtype):
    data = _inputs(kind, 192, 192, dtype, shape=(1, 3, 97, 121), history=0)
    x, weight = data["x"], data["packed"]
    workspace = torch.empty_like(x) if dtype == torch.float32 else None
    options = dict(allow_tf32=dtype == torch.float32, sample_workspace=workspace)
    if kind == "raw":
        outputs = {"output": torch.empty((1, 3, 97, 121, 192), device="cuda", dtype=dtype)}
        plan = cudnn.Conv3dRawSm100(x, weight, outputs["output"], **options)
        execute = partial(plan.execute, x, weight, outputs["output"], workspace=workspace)
    else:
        outputs = {
            "padded_output": torch.empty((1, 5, 99, 123, 192), device="cuda", dtype=dtype),
            "cache_output": torch.empty((1, 2, 97, 121, 192), device="cuda", dtype=dtype),
            "residual_output": torch.empty_like(data["residual"]),
        }
        args = (x, weight, data["bias"], data["gamma"], outputs["padded_output"], outputs["cache_output"])
        residuals = (data["residual"], data["residual_bias"], outputs["residual_output"])
        plan = cudnn.Conv3dRmsNormSiluPadSm100(*args, 0, *residuals, **options)
        execute = partial(plan.execute, *args, *residuals, workspace=workspace)
    plan.check_support()
    plan.compile()
    execute()
    _check_reference(kind, data, snapshot(outputs))
    if workspace is not None:
        outputs["rounded_input"] = workspace
        assert_bitwise({"rounded_input": workspace}, {"rounded_input": _tf32(x)})
    original = snapshot(data)
    # Match the shipped 2CTA launch geometry: at least two scheduler waves,
    # and enough reduction tiles to wrap the producer's shared-memory ring.
    from cudnn.conv.cutedsl.conv3d_postops._config import LaunchPolicy

    clusters = math.ceil(3 * 97 * 121 / 256)
    assert clusters > plan.config.ctas // 2
    policy = LaunchPolicy.for_conv(plan.config, fuse_norm=kind == "norm_pad", has_residual=kind == "norm_pad")
    k_tiles = 27 * math.ceil(192 / (32 if dtype == torch.float32 else 64))
    assert_warm_stress(
        StressCase(
            name=f"conv3d-{kind}-{dtype}",
            launch=execute,
            outputs=outputs,
            thread_blocks=plan.config.ctas,
            refills_per_launch=k_tiles - policy.ab_stages,
        ),
        validation=stress_validation(cluster=2),
    )
    assert_bitwise(data, original, "stress input mutation")


@torch.inference_mode()
def test_fprop_causal_bf16_t3_h97_w121_warm_stress():
    """C12 BF16 uses a separate 1CTA body with seven K tiles and five stages."""
    data = _inputs("causal", 12, 160, torch.bfloat16, shape=(1, 3, 97, 121))
    x, weight, previous = data["x"], data["packed"], data["previous"]
    padded = torch.empty((1, 5, 99, 123, 16), device="cuda", dtype=x.dtype)
    cache = torch.empty((1, 2, 97, 121, 12), device="cuda", dtype=x.dtype)
    output = torch.empty((1, 3, 97, 121, 160), device="cuda", dtype=x.dtype)
    plan = cudnn.CausalConv3dWithCacheSm100(x, weight, padded, cache, output, previous)
    plan.check_support()
    plan.compile()
    execute = partial(plan.execute, x, weight, padded, cache, output, previous)
    execute()
    _check_reference("causal", data, snapshot({"output": output, "cache_output": cache}))
    joined = torch.cat((previous, x.permute(0, 2, 3, 4, 1)), dim=1)
    assert_bitwise({"padded_input": padded}, {"padded_input": F.pad(joined, (0, 4, 1, 1, 1, 1))})
    original = snapshot(data)
    assert math.ceil(3 * 97 * 121 / 128) > plan.max_active_clusters
    assert_warm_stress(
        StressCase(
            name="conv3d-causal-bf16",
            launch=execute,
            outputs={"output": output, "cache_output": cache, "padded_input": padded},
            thread_blocks=plan.max_active_clusters,
            refills_per_launch=7 - 5,
        )
    )
    assert_bitwise(data, original, "stress input mutation")
