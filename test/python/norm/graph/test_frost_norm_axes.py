# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frost norm must not accept graph axes its rowwise kernels cannot represent."""

import pytest
import torch

import cudnn

pytest.importorskip("cudnn.norm", reason="Frost norm engines require a built frontend")


@pytest.mark.L0
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_nontrailing_layernorm_axes_use_backend(cudnn_handle, monkeypatch, backward):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    cc = torch.cuda.get_device_capability()
    if cc not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip("Frost norm supports SM90 and SM100-class devices")

    # Axis 1 is contiguous in storage, so cuDNN supports it, but it is not
    # the trailing logical axis that Frost flattens into each row.
    x = torch.randn((2, 64, 32), device="cuda").transpose(1, 2)
    gamma = torch.randn((1, 32, 1), device="cuda")
    beta = torch.randn_like(gamma)
    eps = torch.full((1, 1, 1), 1e-5, dtype=torch.float32)
    mean = x.mean(dim=1, keepdim=True).contiguous()
    rstd = torch.rsqrt(x.var(dim=1, correction=0, keepdim=True) + 1e-5).contiguous()
    saved_stream = cudnn.get_stream(cudnn_handle)
    cudnn.set_stream(handle=cudnn_handle, stream=torch.cuda.current_stream().cuda_stream)
    try:
        graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=cudnn_handle)
        X = graph.tensor(name="X", dim=x.shape, stride=x.stride(), data_type=x.dtype)
        S = graph.tensor(name="scale", dim=gamma.shape, stride=gamma.stride(), data_type=gamma.dtype)
        if backward:
            dy = torch.randn_like(x)
            DY = graph.tensor(name="DY", dim=dy.shape, stride=dy.stride(), data_type=dy.dtype)
            MN = graph.tensor(name="mean", dim=mean.shape, stride=mean.stride(), data_type=mean.dtype)
            RS = graph.tensor(name="rstd", dim=rstd.shape, stride=rstd.stride(), data_type=rstd.dtype)
            DX, DS, DB = graph.layernorm_backward(grad=DY, input=X, scale=S, mean=MN, inv_variance=RS)
            DX.set_output(True).set_data_type(torch.float32)
            DS.set_output(True).set_data_type(torch.float32)
            DB.set_output(True).set_data_type(torch.float32)
            dx, ds, db = torch.empty_like(x), torch.empty_like(gamma), torch.empty_like(beta)
            bindings = {X: x, S: gamma, DY: dy, MN: mean, RS: rstd, DX: dx, DS: ds, DB: db}
        else:
            B = graph.tensor(name="bias", dim=beta.shape, stride=beta.stride(), data_type=beta.dtype)
            E = graph.tensor_like(eps, name="epsilon")
            Y, MN, RS = graph.layernorm(norm_forward_phase=cudnn.norm_forward_phase.TRAINING, input=X, scale=S, bias=B, epsilon=E)
            for t in (Y, MN, RS):
                t.set_output(True).set_data_type(torch.float32)
            y, mean_out, rstd_out = torch.empty_like(x), torch.empty_like(mean), torch.empty_like(rstd)
            bindings = {X: x, S: gamma, B: beta, E: eps, Y: y, MN: mean_out, RS: rstd_out}

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        assert all(p.engine_id < 20000 for p in graph.plans), "Frost must decline non-trailing normalization axes"
        graph.check_support()
        graph.build_plans()
        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        graph.execute(bindings, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()

        xr = x.detach().requires_grad_(True)
        gr = gamma.detach().requires_grad_(True)
        br = beta.detach().requires_grad_(True)
        refmean = xr.mean(dim=1, keepdim=True)
        refrstd = torch.rsqrt(xr.var(dim=1, correction=0, keepdim=True) + 1e-5)
        ref = (xr - refmean) * refrstd * gr + br
        if backward:
            refdx, refds, refdb = torch.autograd.grad(ref, (xr, gr, br), dy)
            torch.testing.assert_close(dx, refdx, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(ds, refds, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(db, refdb, atol=2e-5, rtol=2e-5)
        else:
            torch.testing.assert_close(y, ref, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(mean_out, refmean, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(rstd_out, refrstd, atol=2e-5, rtol=2e-5)
    finally:
        cudnn.set_stream(handle=cudnn_handle, stream=saved_stream)


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["layer", "rms"])
@pytest.mark.parametrize(
    "x_shape,scale_shape,stats_shape,accepted",
    [
        ((2, 32, 64), (1, 1, 64), (2, 32, 1), True),
        ((2, 32, 64), (1, 32, 64), (2, 1, 1), True),
        ((2, 32, 1, 1), (1, 32, 1, 1), (2, 1, 1, 1), True),
        ((2, 32, 1), (1, 1, 1), (2, 32, 1), True),
        ((2, 32, 64), (1, 32, 1), (2, 1, 64), False),
        ((2, 32, 64), (1, 1, 1), (2, 32, 64), False),
        ((2, 32, 64), (1, 1, 64), (2, 1, 1), False),
    ],
)
def test_rowwise_axis_eligibility(variant, x_shape, scale_shape, stats_shape, accepted):
    from types import SimpleNamespace

    from cudnn.norm.config_sm100 import NormVariant
    from cudnn.norm.graph_analyzer import NormGraphFacts, rowwise_axes_mismatch

    def tensor(shape):
        return SimpleNamespace(get_dim=lambda: shape)

    facts = NormGraphFacts(
        variant=NormVariant.LAYER_NORM if variant == "layer" else NormVariant.RMS_NORM,
        x_shape=x_shape,
        scale_t=tensor(scale_shape),
        inv_var_t=tensor(stats_shape),
    )
    assert (rowwise_axes_mismatch(facts) is None) is accepted
