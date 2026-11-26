# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 canonical dGLU with atomic dprob; deterministic dprob remains unsupported."""

import pytest
import torch

from gemm.cutedsl import test_grouped_gemm_dglu_gate_then_up as shared

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_rubin():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SM107 canonical dGLU qualification")


def compare_outputs(actual, expected, precision, cfg, *, canonicalize=False, discrete_col=False):
    convert = shared.unpair if canonicalize else lambda x, **kwargs: x
    shared.assert_bytes(actual["d_row_tensor"], convert(expected["d_row_tensor"]))
    # Atomic dprob is not a deterministic contract. This is the existing atomic
    # versus deterministic reference bound, not a relaxed payload tolerance.
    torch.testing.assert_close(actual["dprob_tensor"], expected["dprob_tensor"], rtol=1e-5, atol=1e-5)
    if precision == "mxfp8":
        shared.assert_bytes(actual["d_col_tensor"], convert(expected["d_col_tensor"]))
        m, n = sum(cfg["group_m_list"]), 2 * cfg["n"]
        shared.assert_bytes(
            shared.logical_scales(actual["sfd_row_tensor"], m, n),
            convert(shared.logical_scales(expected["sfd_row_tensor"], m, n), block=1),
        )
        shared.assert_bytes(
            shared.column_scales(actual["sfd_col_tensor"], n, cfg["group_m_list"], discrete_col),
            convert(shared.column_scales(expected["sfd_col_tensor"], n, cfg["group_m_list"], discrete_col), axis=0),
        )
    else:
        shared.assert_bytes(actual["amax_tensor"], expected["amax_tensor"])


@pytest.mark.parametrize("precision,b_major", [("mxfp8", "k"), ("mxfp8", "n"), ("nvfp4", "k"), ("nvfp4_ue5m3", "k")])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("n", [128, 256])
def test_rubin_dglu_reference_and_replay(precision, discrete, b_major, n):
    torch.manual_seed(401)
    cfg, inputs = shared.problem(n=n, group_m=(256, 512), precision=precision, b_major=b_major)
    legacy = shared.run(cfg, inputs, canonical=False, discrete=discrete, deterministic=False)
    shared.check_ref_grouped_gemm_dswiglu(inputs, legacy, {**cfg, "discrete_col_sfd": False})
    natural = {**inputs, "c_tensor": shared.unpair(inputs["c_tensor"]).squeeze(-1).contiguous()}
    actual = shared.run(cfg, natural, canonical=True, discrete=discrete, deterministic=False)
    compare_outputs(actual, legacy, precision, cfg, canonicalize=True)
    saved = {key: value.clone() if value is not None else None for key, value in actual.items()}
    for _ in range(3):
        actual = shared.run(cfg, natural, canonical=True, discrete=discrete, deterministic=False)
        compare_outputs(actual, saved, precision, cfg)


@pytest.mark.parametrize("precision", ["mxfp8", "nvfp4", "nvfp4_ue5m3"])
def test_rubin_empty_expert(precision):
    cfg, inputs = shared.problem(precision=precision)
    legacy = shared.run(cfg, inputs, canonical=False, deterministic=False)
    natural = {**inputs, "c_tensor": shared.unpair(inputs["c_tensor"]).squeeze(-1).contiguous()}
    actual = shared.run(cfg, natural, canonical=True, deterministic=False)
    compare_outputs(actual, legacy, precision, cfg, canonicalize=True)
    if precision.startswith("nvfp4"):
        # An expert with no tiles retains the reduction identity.
        assert torch.isneginf(actual["amax_tensor"][1]).all().item()


@pytest.mark.parametrize("discrete", [False, True])
def test_rubin_expert_column_scales(discrete):
    cfg, inputs = shared.problem()
    legacy = shared.run(cfg, inputs, canonical=False, discrete=discrete, discrete_col_sfd=True, deterministic=False)
    shared.check_ref_grouped_gemm_dswiglu(inputs, legacy, {**cfg, "discrete_col_sfd": True})
    natural = {**inputs, "c_tensor": shared.unpair(inputs["c_tensor"]).squeeze(-1).contiguous()}
    actual = shared.run(cfg, natural, canonical=True, discrete=discrete, discrete_col_sfd=True, deterministic=False)
    compare_outputs(actual, legacy, "mxfp8", cfg, canonicalize=True, discrete_col=True)


@pytest.mark.parametrize("precision", ["mxfp8", "nvfp4", "nvfp4_ue5m3"])
def test_rubin_changed_input_graph(precision):
    cfg, inputs = shared.problem(n=128, group_m=(256,), precision=precision)
    inputs["c_tensor"] = shared.unpair(inputs["c_tensor"]).squeeze(-1).contiguous()
    dprob = torch.zeros_like(inputs["prob_tensor"], dtype=torch.float32)
    shared.run(cfg, inputs, canonical=True, deterministic=False, dprob=dprob)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dprob.zero_()
        captured = shared.run(cfg, inputs, canonical=True, deterministic=False, dprob=dprob)
    for value in (0.5, -0.25, 0.5):
        inputs["c_tensor"].fill_(value)
        graph.replay()
        saved = {key: tensor.clone() if tensor is not None else None for key, tensor in captured.items()}
        expected = shared.run(cfg, inputs, canonical=True, deterministic=False)
        compare_outputs(saved, expected, precision, cfg)


@pytest.mark.parametrize("n", [128, 256])
def test_rubin_dbias(n):
    shared.test_gate_then_up_dglu_dbias(n)


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("b_major", ["k", "n"])
@pytest.mark.parametrize("dbias", [False, True])
def test_rubin_bf16(discrete, b_major, dbias):
    shared.test_gate_then_up_bf16_dglu(discrete, b_major, dbias)


def test_rubin_deterministic_dprob_rejected():
    cfg, inputs = shared.problem()
    with pytest.raises(NotImplementedError, match="deterministic dprob"):
        shared.run(cfg, inputs, canonical=True, deterministic=True)
