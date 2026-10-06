# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KDA graph normalization semantics, independent of scheduling heuristics."""

import importlib

import pytest
import torch

import cudnn
from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.linear_attention.graph_analyzer import analyze

from linear_attention.reference_kda import kda_reference, rms_ratio
from linear_attention.test_la import CUDNN_DTYPE, build_and_pin

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.fixture(autouse=True)
def blackwell():
    error = cutedsl_requirement_error("KDA additive normalization tests")
    if error:
        pytest.skip(error)
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family Frost KDA")


@pytest.fixture(autouse=True)
def restore_handle_stream(cudnn_handle):
    previous = cudnn.get_stream(cudnn_handle)
    try:
        yield
    finally:
        cudnn.set_stream(cudnn_handle, previous)


def make_inputs(total=128, heads=4, dtype=torch.bfloat16, lengths=None, *, dim=128, value_dim=None, head_counts=None):
    torch.manual_seed(821)
    hq, hk, hv = head_counts or (heads, heads, heads)
    heads = max(hq, hv)
    value_dim = dim if value_dim is None else value_dim
    q = torch.randn(total, hq, dim, dtype=dtype, device="cuda")
    k = torch.randn(total, hk, dim, dtype=dtype, device="cuda")
    v = torch.randn(total, hv, value_dim, dtype=dtype, device="cuda")
    # Zero rows and small norms expose addition-vs-clamping; ordinary rows
    # also exercise normalization where epsilon has negligible influence.
    for tensor in (q, k):
        tensor[0::4] = 0
        tensor[1::4] *= 1e-5
        tensor[2::4] *= 1e-4
    lengths = [total] if lengths is None else lengths
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
    return dict(
        q=q,
        k=k,
        v=v,
        g=torch.full((total, heads, dim), -0.1, dtype=torch.float32, device="cuda"),
        beta=torch.full((total, heads), 0.5, dtype=dtype, device="cuda"),
        cu_seqlens=cu,
        initial_state=torch.randn(len(lengths), heads, value_dim, dim, dtype=torch.float32, device="cuda") * 0.01,
    )


def graph_for(values, epsilon=None, *, normalize=True, invariant=False, overwrite=False, checkpoint=0, safe_gate=False):
    graph = cudnn.pygraph()
    ports = {name: graph.tensor(list(t.shape), stride=list(t.stride()), data_type=CUDNN_DTYPE[t.dtype], name=name) for name, t in values.items()}
    graph.kda(
        **ports,
        output_final_state=True,
        use_qk_l2norm=normalize,
        batch_invariant=invariant,
        overwrite_initial_state=overwrite,
        qk_l2norm_additive_epsilon=epsilon,
        checkpoint_every_n_tokens=checkpoint,
        safe_gate=safe_gate,
        use_beta_sigmoid=safe_gate,
        gate_lower_bound=-5.0 if safe_gate else None,
    )
    node = graph.nodes[0]
    output = torch.empty(tuple(node.outputs["O"].dim), dtype=values["q"].dtype, device=values["q"].device)
    state = values["initial_state"] if overwrite else torch.empty_like(values["initial_state"])
    for name, tensor in (("O", output), ("final_state", state)):
        node.outputs[name].set_output(True).set_data_type(CUDNN_DTYPE[tensor.dtype])
    pack = {**{ports[name]: value for name, value in values.items()}, node.outputs["O"]: output, node.outputs["final_state"]: state}
    if checkpoint:
        port = node.outputs["state_checkpoints"]
        port.set_output(True).set_data_type(CUDNN_DTYPE[values["q"].dtype])
        pack[port] = torch.full(tuple(port.dim), float("nan"), dtype=values["q"].dtype, device=values["q"].device)
    return graph, pack, output, state


def prepare(values, epsilon=None, *, normalize=True, invariant=False, overwrite=False, checkpoint=0, safe_gate=False, handle):
    graph, pack, output, state = graph_for(
        values, epsilon, normalize=normalize, invariant=invariant, overwrite=overwrite, checkpoint=checkpoint, safe_gate=safe_gate
    )
    # Unsupported environments skip above; an accidentally ignored attribute
    # or a missing implementation on this supported route must fail.
    from cudnn.linear_attention.frost.kda_engine import KdaFrostEngine

    graph.validate()
    KdaFrostEngine().check_support(graph)
    build_and_pin(graph, "kda_frost")
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
    plan = graph._compiled_plans[graph._plan_index].compiled
    print("KDA normalization route", dict(epsilon=epsilon, chain=plan.chain, dv_split=plan.dv_split, prep=plan.prep))

    def run():
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        graph.execute(pack, workspace, handle=handle)
        result = (output, state)
        return result + (pack[graph.nodes[0].outputs["state_checkpoints"]],) if checkpoint else result

    return run, graph, workspace


def normalized(values, epsilon):
    """Approximate external-normalization comparison, not a rounding contract."""
    result = dict(values)
    for name in ("q", "k"):
        value = values[name].float()
        inv = (value.square().sum(-1, keepdim=True) + epsilon).rsqrt()
        result[name] = (value * inv).to(values[name].dtype)
    return result


def assert_matches(actual, expected, *, tolerance=0.02):
    for got, want in zip(actual, expected):
        assert torch.isfinite(got).all()
        assert rms_ratio(got, want) < tolerance
        torch.testing.assert_close(got.float(), want.float(), rtol=tolerance, atol=tolerance * max(want.abs().max().item(), 1e-4))
    # Check epsilon-sensitive rows separately; ordinary rows must not hide them.
    for start in (1, 2):
        assert rms_ratio(actual[0][start::4], expected[0][start::4]) < tolerance
    assert torch.count_nonzero(actual[0][0::4]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(32, 4, True), (384, 4, False), (256, 64, False)])
def test_additive_normalization_matches_references(dtype, shape, cudnn_handle):
    total, heads, invariant = shape
    values = make_inputs(total, heads, dtype)
    run, graph, workspace = prepare(values, 1e-6, invariant=invariant, handle=cudnn_handle)
    reference, reference_graph, reference_workspace = prepare(normalized(values, 1e-6), normalize=False, invariant=invariant, handle=cudnn_handle)
    expected = tuple(x.clone() for x in reference())
    actual = run()
    torch.cuda.synchronize()
    assert_matches(actual, expected)
    if total == 32:
        # The independent mathematical oracle must not round normalized Q/K
        # to the IO dtype: a fused implementation need not materialize them.
        inputs = dict(values)
        for name in ("q", "k"):
            value = values[name].double()
            inputs[name] = value * (value.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        expected_math = kda_reference(
            *(inputs[name].unsqueeze(0) for name in ("q", "k", "v", "g", "beta")),
            initial_state=inputs["initial_state"],
            cu_seqlens=inputs["cu_seqlens"],
        )
        assert_matches(actual, (expected_math[0].squeeze(0), expected_math[1]), tolerance=0.03)


def test_additive_cache_keeps_distinct_epsilon_and_legacy_semantics(cudnn_handle):
    values = make_inputs()
    runs = []
    for epsilon in (None, 1e-6, 1e-2, 1e-6):
        run, graph, workspace = prepare(values, epsilon, handle=cudnn_handle)
        output, state = run()
        runs.append((output.clone(), state.clone(), graph, workspace))
    torch.cuda.synchronize()
    assert_matches(runs[1][:2], runs[3][:2], tolerance=1e-6)
    assert not torch.equal(runs[0][0][1::4], runs[1][0][1::4])
    for index, epsilon in ((1, 1e-6), (2, 1e-2)):
        reference, graph, workspace = prepare(normalized(values, epsilon), normalize=False, handle=cudnn_handle)
        assert_matches(runs[index][:2], reference())


def test_additive_ragged_and_chunk_continuation(cudnn_handle):
    values = make_inputs(384, lengths=[127, 257])
    run, graph, workspace = prepare(values, 1e-6, handle=cudnn_handle)
    reference, reference_graph, reference_workspace = prepare(normalized(values, 1e-6), normalize=False, handle=cudnn_handle)
    assert_matches(run(), reference())

    values = make_inputs(384)
    full, full_graph, full_workspace = prepare(values, 1e-6, invariant=True, handle=cudnn_handle)
    full_output, full_state = full()
    state = values["initial_state"]
    pieces = []
    for start, end in ((0, 128), (128, 384)):
        segment = {name: tensor[start:end] for name, tensor in values.items() if name not in ("cu_seqlens", "initial_state")}
        segment["cu_seqlens"] = torch.tensor([0, end - start], dtype=torch.int32, device="cuda")
        segment["initial_state"] = state
        run, graph, workspace = prepare(segment, 1e-6, invariant=True, handle=cudnn_handle)
        output, state = run()
        pieces.append(output.clone())
    assert_matches((torch.cat(pieces), state), (full_output, full_state), tolerance=0.03)


@pytest.mark.parametrize("overwrite", [False, True])
def test_additive_capture_reads_changed_input_and_state(overwrite, cudnn_handle):
    values = make_inputs(384)
    seed = values["initial_state"].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run, graph, workspace = prepare(values, 1e-6, overwrite=overwrite, handle=cudnn_handle)
        run()
        torch.cuda.synchronize()
        values["initial_state"].copy_(seed)
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture, stream=stream):
            captured = run()
        for delta in (0.0, 0.015625):
            values["q"][1::4].add_(delta * 1e-3)
            values["v"].add_(delta)
            seed.add_(delta)
            values["initial_state"].copy_(seed)
            reference, ref_graph, ref_workspace = prepare(normalized(values, 1e-6), normalize=False, handle=cudnn_handle)
            expected = tuple(x.clone() for x in reference())
            captured[0].fill_(float("nan"))
            if not overwrite:
                captured[1].fill_(float("nan"))
            capture.replay()
            stream.synchronize()
            assert_matches(captured, expected)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf"), 1e-50, 1e40, True, "1e-6"])
def test_additive_invalid_epsilon_declines(epsilon):
    graph, *_ = graph_for(make_inputs(16), epsilon)
    facts = analyze(graph)
    assert facts.invalid and "qk_l2norm_additive_epsilon" in facts.invalid


def test_additive_requires_normalization_and_is_forward_only():
    graph, *_ = graph_for(make_inputs(16), 1e-6, normalize=False)
    assert "requires use_qk_l2norm" in analyze(graph).invalid
    graph = cudnn.pygraph()
    with pytest.raises(TypeError, match="qk_l2norm_additive_epsilon"):
        graph.kda_bwd(qk_l2norm_additive_epsilon=1e-6)


@pytest.mark.parametrize(
    "module,engine",
    [
        ("cake.kda_engine", "KdaCakeEngine"),
        ("cutile.kda_engine", "KdaCuTileEngine"),
        ("hopper.kda_engine", "KdaHopperEngine"),
        ("hopper.cuda_engine", "KdaHopperCudaEngine"),
    ],
)
def test_additive_other_providers_decline(module, engine, monkeypatch):
    from cudnn.engines import manifest

    monkeypatch.setattr(manifest, "opt_in_engines_enabled", lambda: True)
    graph, *_ = graph_for(make_inputs(16), 1e-6)
    graph.validate()
    provider = getattr(importlib.import_module("cudnn.linear_attention." + module), engine)()
    with pytest.raises(NotImplementedError, match="qk_l2norm_additive_epsilon"):
        provider.check_support(graph)


@pytest.mark.parametrize(
    "dim,value_dim,head_counts,safe_gate,checkpoint,state_dtype,total",
    [
        (64, 64, (4, 4, 4), False, 0, torch.float32, 384),
        (64, 128, (4, 4, 2), True, 32, torch.bfloat16, 128),
        (128, 64, (2, 4, 4), True, 0, torch.float32, 96),
        (128, 128, (4, 4, 4), True, 32, torch.bfloat16, 384),
        (128, 128, (72, 72, 72), False, 0, torch.float32, 256),
    ],
)
def test_additive_forward_contracts(dim, value_dim, head_counts, safe_gate, checkpoint, state_dtype, total, cudnn_handle):
    """Selected rectangular/GQA, raw-gate and checkpoint contracts, not a tile sweep."""
    values = make_inputs(total, dim=dim, value_dim=value_dim, head_counts=head_counts)
    values["initial_state"] = values["initial_state"].to(state_dtype)
    if safe_gate:
        heads = max(head_counts[0], head_counts[2])
        values["g"].fill_(-2.0)
        values["a_log"] = torch.zeros(heads, device="cuda", dtype=torch.float32)
        values["dt_bias"] = torch.zeros(heads, dim, device="cuda", dtype=torch.float32)
    run, graph, workspace = prepare(values, 1e-6, checkpoint=checkpoint, safe_gate=safe_gate, handle=cudnn_handle)
    reference, ref_graph, ref_workspace = prepare(normalized(values, 1e-6), normalize=False, checkpoint=checkpoint, safe_gate=safe_gate, handle=cudnn_handle)
    expected = tuple(x.clone() for x in reference())
    actual = run()
    torch.cuda.synchronize()
    if checkpoint:
        # One sequence here: rows are incoming states at j * checkpoint,
        # excluding the sequence end. The remaining capacity is unspecified.
        valid_rows = (total + checkpoint - 1) // checkpoint
        actual = actual[:2] + (actual[2][:valid_rows],)
        expected = expected[:2] + (expected[2][:valid_rows],)
    assert_matches(actual, expected)
