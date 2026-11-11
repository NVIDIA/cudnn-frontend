# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default interleaved GLU/RHT output bytes consumed as GEMM-blocked scales."""

import pytest
import torch

from determinism_utils import assert_bitwise
from gemm.cutedsl.test_grouped_gemm_glu_hadamard_quant import _check_colwise_rht, _rht_ref
from gemm.graph.test_low_precision_matmul import float4_e2m1fn_x2_to_float32

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Default E4M3 column-scale qualification requires SM100/103")


def _experts(n, k=512, counts=(256, 0, 512)):
    rng = torch.Generator().manual_seed(401)
    result = []
    for index, m in enumerate(counts):
        result.append(
            dict(
                a=torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, generator=rng).cuda(),
                b=torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=rng).cuda(),
                sa=torch.exp2(torch.randint(-5, -2, (m, k // 16), generator=rng).float()).to(torch.float8_e4m3fn).cuda(),
                sb=torch.exp2(torch.randint(-5, -2, (n, k // 16), generator=rng).float()).to(torch.float8_e4m3fn).cuda(),
                prob=torch.linspace(0.5, 0.9, m, device="cuda"),
                alpha=0.5 + index * 0.25,
            )
        )
    return result


def _sf_view(sf):
    e, rows, cols = sf.shape
    backing = sf.view(torch.uint8).reshape(e, rows // 128, 4, 32, cols // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous()
    return backing.permute(3, 4, 1, 5, 2, 0).view(sf.dtype)


def _pack(experts):
    counts = [e["a"].shape[0] for e in experts]
    a = torch.cat([e["a"] for e in experts]).view(torch.float4_e2m1fn_x2)
    b = torch.stack([e["b"] for e in experts]).permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
    sa = torch.cat([e["sa"].view(torch.uint8) for e in experts]).view(torch.float8_e4m3fn)
    sb = torch.stack([e["sb"].view(torch.uint8) for e in experts]).view(torch.float8_e4m3fn)
    m, packed_k = a.shape
    return dict(
        a_tensor=a.as_strided((m, packed_k, 1), (packed_k, 1, m * packed_k)),
        b_tensor=b,
        sfa_tensor=_sf_view(sa.unsqueeze(0)),
        sfb_tensor=_sf_view(sb),
        padded_offsets=torch.tensor(counts, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32),
        alpha_tensor=torch.tensor([e["alpha"] for e in experts], device="cuda"),
        prob_tensor=torch.cat([e["prob"] for e in experts]).reshape(m, 1, 1),
        rht_colwise_dtype=torch.float4_e2m1fn_x2,
        rht_norm_const=16.0,
    )


def _call(kwargs):
    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100

    return grouped_gemm_glu_hadamard_quant_wrapper_sm100(**kwargs)


def _live(outputs):
    return {name: value for name, value in outputs.items() if value is not None}


def _snapshot(outputs):
    return {name: value.reshape(-1).view(torch.uint8).clone() for name, value in _live(outputs).items()}


def _input_tensors(kwargs):
    return {
        name: value.view(torch.uint8) if value.dtype == torch.float4_e2m1fn_x2 else value for name, value in kwargs.items() if isinstance(value, torch.Tensor)
    }


def _check_reference(experts, outputs, rht_norm_const=16.0):
    reference = []
    start = 0
    for expert in experts:
        count, n = expert["a"].shape[0], expert["b"].shape[0]
        if not count:
            continue
        a = float4_e2m1fn_x2_to_float32(expert["a"].cpu()) * expert["sa"].cpu().float().repeat_interleave(16, dim=1)
        b = float4_e2m1fn_x2_to_float32(expert["b"].cpu()) * expert["sb"].cpu().float().repeat_interleave(16, dim=1)
        c = ((a.double() @ b.double().t()) * expert["alpha"]).float()
        torch.testing.assert_close(outputs["c_tensor"][start : start + count, :, 0].cpu().float(), c, atol=0.005, rtol=0.01)
        gate_up = c.reshape(count, n // 64, 2, 32)
        d = (torch.nn.functional.silu(gate_up[:, :, 0]) * gate_up[:, :, 1]).reshape(count, n // 2)
        d *= expert["prob"].cpu().reshape(count, 1)
        reference.append(d)
        start += count
    d_reference = torch.cat(reference).unsqueeze(-1)
    torch.testing.assert_close(outputs["d_tensor"].cpu().float(), d_reference, atol=0.005, rtol=0.015)
    rht_reference = _rht_ref(d_reference, rowwise=False)
    requested_scales = rht_reference.float().abs().reshape(start // 16, 16, -1).amax(dim=1) * (rht_norm_const / 6.0)
    assert torch.isfinite(requested_scales).all()
    assert requested_scales.max().item() <= torch.finfo(torch.float8_e4m3fn).max
    _check_colwise_rht({"aligned_group_m_list": [e["a"].shape[0] for e in experts]}, outputs, rht_reference, rht_norm_const, None)


def _expert_output(outputs, start, count, f):
    result = {}
    for name, value in _live(outputs).items():
        if name == "rht_colwise_tensor":
            result[name] = value.view(torch.uint8)[start * f // 2 : (start + count) * f // 2]
        elif name == "sfrht_colwise_tensor":
            result[name] = value[start * f // 16 : (start + count) * f // 16]
        else:
            result[name] = value[start : start + count]
    return result


@pytest.mark.parametrize("n", [256, 2048])
def test_colwise_scale_reference_replay_graph(n):
    experts = _experts(n)
    kwargs = _pack(experts)
    original_inputs = {name: value.clone() for name, value in _input_tensors(kwargs).items()}
    outputs = _call(kwargs)
    _check_reference(experts, outputs)
    expected = _snapshot(outputs)
    noise_stream = torch.cuda.Stream()
    noise = torch.ones(1024 * 1024, device="cuda")
    for _ in range(3):
        with torch.cuda.stream(noise_stream):
            noise.mul_(1.001).add_(0.001)
        assert_bitwise(_snapshot(_call(_pack(experts))), expected, "fresh allocations / concurrent work")
    torch.cuda.current_stream().wait_stream(noise_stream)
    assert_bitwise(_input_tensors(kwargs), original_inputs, "inputs preserved")
    _call(kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _call(kwargs)
    kwargs["b_tensor"].view(torch.uint8).bitwise_xor_(1)
    kwargs["sfb_tensor"].view(torch.uint8).bitwise_xor_(1)
    expected = _snapshot(_call(kwargs))
    graph.replay()
    assert_bitwise(_snapshot(captured), expected, "graph replay after weight and scale changes")


@pytest.mark.parametrize("n", [256, 2048])
def test_colwise_scale_batch_invariance(n):
    experts = _experts(n)
    expected = _snapshot(_call(_pack([experts[0]])))
    for indices in ((0, 1, 2), (2, 0, 1), (0, 0, 2), (0, 1)):
        outputs = _call(_pack([experts[i] for i in indices]))
        start = 0
        for index in indices:
            count = experts[index]["a"].shape[0]
            if index == 0:
                assert_bitwise(_snapshot(_expert_output(outputs, start, count, n // 2)), expected, str(indices))
            start += count
    experts[2]["a"].zero_()
    experts[2]["b"].zero_()
    outputs = _call(_pack(experts))
    assert_bitwise(_snapshot(_expert_output(outputs, 0, 256, n // 2)), expected, "changed neighboring expert")


def test_colwise_scale_warm_stress(monkeypatch):
    import cutlass
    from warm_stress import StressCase, assert_warm_stress

    monkeypatch.delenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", raising=False)
    # K=4096 crosses the input pipeline's staged ring within one GEMM tile.
    # Several output tiles also reuse the epilogue's shared D buffers.
    experts = _experts(2048, k=4096, counts=(1024, 0, 2048))
    kwargs = _pack(experts)
    kwargs["rht_norm_const"] = 1.0
    outputs = _call(kwargs)
    _check_reference(experts, outputs, rht_norm_const=1.0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _call(kwargs)
    clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
    assert_warm_stress(
        StressCase(
            "grouped GLU/RHT colwise scales M=[1024,0,2048] N=2048 K=4096",
            launch=graph.replay,
            outputs=_live(captured),
            thread_blocks=2 * clusters,
            refills_per_launch=1,  # Conservative lower bound, not a count of all ring refills.
        )
    )
