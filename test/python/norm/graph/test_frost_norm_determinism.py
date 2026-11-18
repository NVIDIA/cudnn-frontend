# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive norm replay and per-sample invariance, including the SM107 target.

LN/RMSNorm parameter gradients use FP32 atomics and have no bitwise guarantee;
this suite covers their forward outputs and DX, and every AdaIN forward/backward output.
Numerical backward coverage for all three variants lives in test_frost_norm_affine.
"""

import cudnn
import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from test_frost_norm_affine import _build, _graph, _loss_grad, _reference, _supported  # noqa: F401

pytestmark = pytest.mark.L0


def _case(handle, kind, phase, dtype, zero_centered_gamma, shape):
    torch.manual_seed(20261113)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    affine_shape = (shape[0], shape[1], 1) if kind == "in" else (shape[0], 1, shape[2])
    gamma = torch.randn(affine_shape, device="cuda", dtype=torch.float32)
    beta = torch.randn_like(gamma)
    yref, mean, rstd, leaves = _reference(kind, x, gamma, beta, zero_centered_gamma, torch.float32)
    dy = _loss_grad(yref, x)
    grads = torch.autograd.grad(yref, leaves, grad_outputs=dy.float())
    inputs = dict(x=x, gamma=gamma, beta=beta, dy=dy, mean=mean.detach(), rstd=rstd.detach())
    expected = dict(zip(("dx", "dgamma", "dbeta"), grads)) if phase == "bprop" else dict(y=yref, rstd=rstd)
    if phase == "fprop" and kind != "rms":
        expected["mean"] = mean

    def prepare(values):
        g, vp, outputs = _graph(
            handle,
            kind,
            values["x"],
            values["gamma"],
            values["beta"],
            phase=phase,
            adaptive=True,
            zero_centered_gamma=zero_centered_gamma,
            dy=values["dy"],
            mean=values["mean"],
            rstd=values["rstd"],
        )
        _build(g, phase)
        workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
        output_keys = {name: next(t for t, value in vp.items() if value is buf) for name, buf in outputs.items()}

        def run():
            fresh = {name: torch.full_like(value, float("nan")) for name, value in outputs.items()}
            rebound = dict(vp)
            rebound.update({output_keys[name]: value for name, value in fresh.items()})
            old = cudnn.get_stream(handle)
            cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
            try:
                g.execute(rebound, workspace, handle=handle)
            finally:
                cudnn.set_stream(handle, old)
            return {"dx": fresh["dx"]} if phase == "bprop" and kind != "in" else fresh

        return run

    original = snapshot(inputs)
    output = assert_replays(prepare(inputs))
    for name, value in output.items():
        torch.testing.assert_close(value, expected[name].detach(), check_dtype=False)
    assert_bitwise(inputs, original, "inputs after replay")
    return inputs, output, prepare


@pytest.mark.parametrize("kind,phase", [(kind, phase) for kind in ("ln", "rms", "in") for phase in ("fprop", "bprop")])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("zero_centered_gamma", [False, True], ids=["gamma", "zero-centered-gamma"])
@pytest.mark.parametrize("shape", [(3, 5, 33), (3, 1025, 96)], ids=["tail-d33", "multirow-d96"])
def test_frost_norm_replay_and_batch_invariance(cudnn_handle, kind, phase, dtype, zero_centered_gamma, shape):
    inputs, output, prepare = _case(cudnn_handle, kind, phase, dtype, zero_centered_gamma, shape)
    # Preserve the original DY and loss scale across isolation, reordering and duplication.
    for indices in ([0], [2, 0], [1, 0, 0]):
        values = {name: value[indices].contiguous() for name, value in inputs.items()}
        actual = snapshot(prepare(values)())
        expected = {name: value[indices] for name, value in output.items()}
        assert_bitwise(actual, expected, f"batch {indices}")
    changed = {name: value.clone() for name, value in inputs.items()}
    for value in changed.values():
        value[1:].mul_(3)
    actual = snapshot(prepare(changed)())
    assert_bitwise({name: value[:1] for name, value in actual.items()}, {name: value[:1] for name, value in output.items()}, "changed neighbors")


@pytest.mark.parametrize("phase", ["fprop", "bprop"])
@pytest.mark.parametrize("kind", ["ln", "rms", "in"])
def test_frost_norm_golden(cudnn_handle, kind, phase):
    from cudnn.frost.buffers import cutedsl_compiler_version

    compiler = cutedsl_compiler_version()
    flavor = "unknown" if compiler is None else "_".join(map(str, compiler))
    inputs, output, _ = _case(cudnn_handle, kind, phase, torch.float32, True, (3, 5, 33))
    assert_golden(f"frost_ada{kind}_{phase}_fp32_zero_centered_gamma_cu{flavor}", inputs, output)


@pytest.mark.parametrize("kind", ["ln", "rms"])
@pytest.mark.parametrize("phase,format", [("fprop", "e4m3"), ("bprop", "e4m3"), ("fprop", "e5m2"), ("bprop", "e5m2"), ("fprop", "nvfp4")])
def test_frost_norm_block_scale_replay_and_batch_invariance(cudnn_handle, kind, phase, format):
    from test_frost_norm_block_scale import _build as build_quant, _check_quantized, _quantized_graph, nvfp4_reference

    torch.manual_seed(20261113)
    x = torch.randn((3, 1024, 96), device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn((3, 1, 96), device="cuda")
    beta = torch.randn_like(gamma)
    y, mean, rstd, leaves = _reference(kind, x, gamma, beta, True, torch.float32)
    dy = _loss_grad(y, x)
    dx = torch.autograd.grad(y, leaves, grad_outputs=dy.float())[0]
    inputs = dict(x=x, gamma=gamma, beta=beta, dy=dy, mean=mean.detach(), rstd=rstd.detach())

    def prepare(values, reference=None):
        vx, vg, vb = (values[name] for name in ("x", "gamma", "beta"))
        if format == "nvfp4":
            g, vp, result = _graph(cudnn_handle, kind, vx, vg, vb, adaptive=True, zero_centered_gamma=True, inference=True)
            source = next(t for t, buf in vp.items() if buf is result["y"])
            del vp[source]
            source.set_output(False).set_data_type(torch.float32)
            yt, st = g.block_scale_quantize(input=source, block_size=16, axis=2)
            payload = torch.empty((*vx.shape[:-1], vx.shape[-1] // 2), device="cuda", dtype=torch.uint8)
            scale = torch.empty((*vx.shape[:-1], vx.shape[-1] // 16), device="cuda", dtype=torch.float8_e4m3fn)
            yt.set_output(True).set_data_type(cudnn.data_type.FP4_E2M1).set_dim(vx.shape).set_stride(vx.stride())
            st.set_output(True).set_data_type(scale.dtype).set_dim(scale.shape).set_stride(scale.stride())
            vp.update({yt: payload, st: scale})
            pairs = [(payload, scale)]
        else:
            dtype = torch.float8_e4m3fn if format == "e4m3" else torch.float8_e5m2
            g, vp, _, pairs, _ = _quantized_graph(
                cudnn_handle,
                kind,
                vx,
                vg,
                vb,
                phase,
                torch.float32,
                dtype,
                adaptive=True,
                zero_centered_gamma=True,
                dy=values["dy"],
                mean=values["mean"],
                rstd=values["rstd"],
            )
        ws = build_quant(g, phase)
        outputs = {f"{i}_{j}": value for i, pair in enumerate(pairs) for j, value in enumerate(pair)}
        keys = {name: next(t for t, value in vp.items() if value is buf) for name, buf in outputs.items()}
        checked = False

        def run():
            nonlocal checked
            fresh = {name: torch.empty_like(value) for name, value in outputs.items()}
            for value in fresh.values():
                value.view(torch.uint8).fill_(0xFF)
            rebound = dict(vp)
            rebound.update({keys[name]: value for name, value in fresh.items()})
            ws.fill_(0xA5)
            old = cudnn.get_stream(cudnn_handle)
            cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
            try:
                g.execute(rebound, ws, handle=cudnn_handle)
            finally:
                cudnn.set_stream(cudnn_handle, old)
            if not checked:
                staged = ws.view(torch.float32).reshape(vx.shape)
                if reference is not None:
                    torch.testing.assert_close(staged, reference.detach())
                if format == "nvfp4":
                    code, scale = nvfp4_reference(staged)
                    torch.testing.assert_close(fresh["0_0"], code, rtol=0, atol=0)
                    torch.testing.assert_close(fresh["0_1"].view(torch.uint8), scale, rtol=0, atol=0)
                else:
                    _check_quantized(staged, [(fresh[f"{i}_0"], fresh[f"{i}_1"]) for i in range(2)], dtype)
                checked = True
            return {name: value.view(torch.uint8) for name, value in fresh.items()}

        return run

    original = snapshot(inputs)
    output = assert_replays(prepare(inputs, y if phase == "fprop" else dx))
    assert_bitwise(inputs, original, "quantized inputs after replay")
    for indices in ([0], [2, 0], [1, 0, 0]):
        values = {name: value[indices].contiguous() for name, value in inputs.items()}
        actual = snapshot(prepare(values)())
        assert_bitwise(actual, {name: value[indices] for name, value in output.items()}, f"quantized batch {indices}")
    changed = {name: value.clone() for name, value in inputs.items()}
    for value in changed.values():
        value[1:].mul_(3)
    actual = snapshot(prepare(changed)())
    assert_bitwise({name: value[:1] for name, value in actual.items()}, {name: value[:1] for name, value in output.items()}, "quantized changed neighbors")
