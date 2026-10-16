# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDN-2 decay range, deterministic correction and state-rebinding regressions."""

import math

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from .test_la import (
    BWD_TOL,
    CUDNN_DTYPE,
    H_TOL,
    LEAF_NAMES,
    M_TOL,
    SEED,
    Backend,
    assert_bitwise,
    assert_bwd_parity,
    assert_fwd_parity,
    assert_rms_close,
    assert_state_close,
    build_and_pin,
    chain_grads,
    chain_reference_grads,
    clear_caches,
    make_case,
    op_args,
    op_tail,
    pinned_op,
    oracle_m,
    random_state,
    reference,
    run_fwd,
    run_summary_bwd,
    thd_tensors,
    state_pool,
)

pytestmark = [pytest.mark.L0, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


@pytest.fixture
def backend():
    clear_caches()
    try:
        yield Backend("frost")
    finally:
        clear_caches()


@pytest.fixture
def built_plans(monkeypatch, backend):
    from cudnn.linear_attention.frost import gdn2_engine

    build = gdn2_engine.build_gdn2
    plans = []

    def record(graph):
        plan = build(graph)
        plans.append(plan)
        return plan

    monkeypatch.setattr(gdn2_engine, "build_gdn2", record)
    return plans


def assert_schedules(plans, shape):
    """Verify coverage against the constructed plans, including the backward's earlier chain threshold."""
    assert {plan.node.node_type.name for plan in plans} == {"GDN2", "GDN2_BWD"}
    for plan in plans:
        if plan.node.node_type.name == "GDN2":
            assert plan.chain == (shape == "chain"), f"{shape}: forward chain={plan.chain}"
            assert plan.prep == (shape == "prep"), f"{shape}: forward prep={plan.prep}"
            assert plan.dv_split == (shape == "prep"), f"{shape}: forward dv_split={plan.dv_split}"
        else:
            assert plan.chain == (shape != "uncut"), f"{shape}: backward chain={plan.chain}"


def steep_gate(case, pattern):
    """A copy of the case whose log-decay on a few key channels drops far below the 16-token-tile range the
    exp2(g) / exp2(-g) operand split can represent (the in-tile log2 prefix below -126, i.e. a tile sum of g
    below -87.3).  Edits are relative to each sequence start, where the kernel tiles begin."""
    g = case.gates["g"].clone()  # [B, T, HO, K], natural log
    if case.varlen:
        spans = [(0, s, e) for s, e in zip(case.cu[:-1].tolist(), case.cu[1:].tolist())]
    else:
        spans = [(b, 0, case.T) for b in range(case.B)]
    for b, s, e in spans:
        seq = g[b, s:e]
        if pattern == "steady":  # one channel at -6 per token: every full tile sums to -96
            seq[:, 0, 7] = -6.0
        elif pattern == "spike":  # one -200 token per tile on an otherwise mild channel: two decay clusters per tile
            seq[5::16, 0, 3] = -200.0
        elif pattern == "huge_spike":  # small gates after this spike must survive prefix cancellation
            seq[:, 0, 3] = -1.0
            seq[5::16, 0, 3] = -1.0e8
        elif pattern == "two_spikes":  # two -150 tokens inside each tile: three clusters
            seq[3::16, case.HO - 1, 9] = -150.0
            seq[11::16, case.HO - 1, 9] = -150.0
        elif pattern == "offset_window":  # a 40-token run at -6 from offset 5: partial tiles at both ends, one full tile past the edge
            seq[5:45, 0, 11] = -6.0
        elif pattern == "extreme":  # an effectively infinite decay and a -50 per-token decay
            seq[:, 0, 21] = -1.0e4
            seq[:, case.HO - 1, 33] = -50.0
        elif pattern == "all_channels":  # every key channel of a head steep at once
            seq[:, 0, :] = -12.0
        elif pattern == "alternating":
            # Each scan group sees mild then steep tiles on successive iterations.
            # This exercises scratch reuse when the mild tile skips the correction barriers.
            tokens = torch.arange(e - s, device=g.device)
            seq[:, 0, 7] = torch.where((tokens // 32) % 2 == 0, -0.01, -12.0)
        else:
            raise ValueError(pattern)
    return case.clone(gates=dict(case.gates, g=g))


STEEP_PATTERNS = ("steady", "spike", "huge_spike", "two_spikes", "offset_window", "extreme", "all_channels", "alternating")
STEEP_SHAPES = {  # token geometry; steep_shape selects heads for the actual device
    "prep": dict(seq_lens=[517, 0, 91, 128, 1]),
    "uncut": dict(seq_lens=[96, 32, 160, 1]),
    "chain": dict(seq_lens=[2048]),
}


def steep_shape(shape, num_sm):
    """Pick head counts that exercise the named schedules, including on small GPU partitions."""
    from cudnn.linear_attention.frost.common.piece_chain import choose_pieces, is_dv_split
    from cudnn.linear_attention.frost.gdn2_engine import PREP_TILE_FRACTION

    lengths = STEEP_SHAPES[shape]["seq_lens"]
    for heads in (2, 1):
        geometry = dict(num_seqs=len(lengths), heads_out=heads, num_sm=num_sm, total_tokens=sum(lengths), b_t=16, expand_num=1)
        pieces, units = choose_pieces(**geometry, cadence_tokens=0, batch_invariant=False)
        reverse, _ = choose_pieces(**geometry, cadence_tokens=0, batch_invariant=False, reverse=True)
        split = not pieces and is_dv_split(**geometry, dim_v=128, unit_chunks=units)
        prep = split and len(lengths) * heads <= int(PREP_TILE_FRACTION * num_sm)
        if (bool(pieces), bool(split), bool(prep), bool(reverse)) == (
            shape == "chain",
            shape == "prep",
            shape == "prep",
            shape != "uncut",
        ):
            return dict(seq_lens=lengths, H=heads)
    raise ValueError(f"{num_sm} SMs cannot exercise the {shape} forward/backward schedules")


@pytest.fixture
def schedule_shape(shape):
    num_sm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    try:
        return steep_shape(shape, num_sm)
    except ValueError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("num_sm", [16, 24, 132, 148, 152, 192])
@pytest.mark.parametrize("shape", list(STEEP_SHAPES))
def test_steep_schedule_geometry(shape, num_sm):
    # The 16-SM case needs fewer heads for prep; larger devices keep the original workload.
    geometry = steep_shape(shape, num_sm)
    assert geometry["H"] == (1 if shape == "prep" and num_sm < 24 else 2)


def test_steep_schedule_geometry_rejects_missing_schedule(monkeypatch):
    from cudnn.linear_attention.frost.common import piece_chain

    monkeypatch.setattr(piece_chain, "is_dv_split", lambda **geometry: False)
    with pytest.raises(ValueError, match="148 SMs cannot exercise the prep"):
        steep_shape("prep", 148)


@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("shape", list(STEEP_SHAPES))
@pytest.mark.parametrize("pattern", STEEP_PATTERNS)
@pytest.mark.parametrize("variant", ("gdn2",))
def test_fwd_bwd_steep_channel_decay(backend, built_plans, variant, pattern, shape, K, schedule_shape):
    """Per-channel log-decay far past the 16-token tile's exp2 range: the forward output and every gradient stay
    finite and match the fp64 reference (a tile sum of g below -126 * ln 2 = -87.3 used to return NaN for the
    rest of the sequence)."""
    case = steep_gate(make_case(variant, torch.bfloat16, K=K, **schedule_shape), pattern)
    assert_bwd_parity(backend, case, l2norm=True)
    assert_schedules(built_plans, shape)


@pytest.mark.parametrize("variant", ("gdn2",))
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fwd_steep_channel_decay_initial_state(backend, variant, K, dtype):
    """The steep path with an initial state, a final state and no in-kernel L2 norm."""
    case = steep_gate(make_case(variant, dtype, K=K, seq_lens=[96, 32, 160, 1]), "spike")
    assert_fwd_parity(backend, case, use_initial_state=True)
    assert_bwd_parity(backend, case, use_initial_state=True, use_dfs=True)


@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("output_transition", [False, True])
def test_steep_summary_matches_reference(backend, K, output_transition):
    """Both summary wrappers retain the seeded state and transition through steep/mild tiles."""
    case = steep_gate(make_case("gdn2", torch.bfloat16, K=K, seq_lens=[96, 0, 32, 160]), "alternating")
    state = random_state(case)
    _, expected = reference(case, initial_state=state)
    final_state, transition = torch.ops.cudnn.gated_delta_net_v2_summary(
        *op_args(case)[1:],
        initial_state=state,
        output_transition=output_transition,
        batch_invariant=True,
        plan_name="gdn2_summary_frost",
    )
    assert_state_close("summary final_state", final_state, expected, H_TOL[case.dtype])
    if output_transition:
        assert_state_close("summary transition", transition, oracle_m(case), M_TOL[case.dtype])


@pytest.mark.parametrize("variant", ("gdn2",))
@pytest.mark.parametrize("pattern", ["steady", "huge_spike"])
def test_steep_channel_decay_beta_guard(backend, variant, pattern):
    """The steep path under the beta guard, whose decay-headroom sensor reads the per-token decay."""
    case = steep_gate(make_case(variant, torch.bfloat16, T=256), pattern)
    assert_fwd_parity(backend, case, l2norm=True, beta_guard=True)
    assert_bwd_parity(backend, case, l2norm=True, beta_guard=True)


@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("shape", list(STEEP_SHAPES))
@pytest.mark.parametrize("variant", ("gdn2",))
def test_fwd_bwd_steep_channel_decay_fp16(backend, built_plans, variant, shape, K, schedule_shape):
    """fp16 operands overflow much sooner: K inv = k * 2^-g passes the fp16 maximum once a tile's log-decay sums
    below about -11, which an alpha of 0.3 per token reaches."""
    case = steep_gate(make_case(variant, torch.float16, K=K, lo=0.3, **schedule_shape), "steady")
    assert_bwd_parity(backend, case, l2norm=True)
    assert_schedules(built_plans, shape)


@pytest.mark.parametrize("variant", ("gdn2",))
def test_steep_channel_decay_overwrite_initial_state(backend, variant):
    """An aliased state (``overwrite_initial_state``) is read from a snapshot, so the exact relaunch still sees the
    original: output, final state and every gradient bitwise the separate-buffer run's."""
    case = steep_gate(make_case(variant, torch.bfloat16, B=8, T=512, H=4), "spike")
    state0, d_final = random_state(case), random_state(case, scale=0.1, seed=SEED + 3)
    o_ref, fs_ref = run_fwd(backend, case, initial_state=state0.clone(), output_final_state=True, batch_invariant=True)
    state = state0.clone()
    o, _ = run_fwd(backend, case, initial_state=state, output_final_state=True, overwrite_initial_state=True)
    assert torch.isfinite(o).all()
    assert_bitwise("o", o, o_ref)
    assert_bitwise("final_state", state, fs_ref)

    _, _, grads, dO, _ = chain_grads(backend, case, initial_state=state0, d_final_state=d_final, batch_invariant=True)
    names = LEAF_NAMES[variant]
    inputs = dict(zip(names, op_args(case)[: len(names)]))
    inputs.update(cu_seqlens=case.cu, dO=dO, initial_state=state0, d_final_state=d_final)
    graph = cudnn.pygraph()
    ports = {name: graph.tensor(list(t.shape), data_type=CUDNN_DTYPE[t.dtype], name=name) for name, t in inputs.items()}
    outs = getattr(graph, f"{variant}_bwd")(**ports, scale=1.0 / math.sqrt(case.K), overwrite_initial_state=True, batch_invariant=True, name="bwd")
    out_ports = dict(zip(list(names) + ["initial_state"], outs))
    for name, port in out_ports.items():
        port.set_output(True).set_data_type(CUDNN_DTYPE[grads[name].dtype])
    build_and_pin(graph, f"{variant}_frost")
    d_state = d_final.clone()
    got = {name: torch.empty_like(grads[name]) for name in names}
    pack = {ports[name]: t for name, t in inputs.items()}
    pack[ports["d_final_state"]] = d_state
    pack.update({out_ports[name]: got[name] for name in names})
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    # Warm the SAME graph with separate state buffers, then bind an alias.
    separate_state = torch.empty_like(d_state)
    pack[out_ports["initial_state"]] = separate_state
    graph.execute(pack, workspace)
    torch.cuda.synchronize()
    assert_bitwise("warmup d_initial_state", separate_state, grads["initial_state"])
    pack[out_ports["initial_state"]] = d_state
    graph.execute(pack, workspace)
    torch.cuda.synchronize()
    for name in names:
        assert torch.isfinite(got[name]).all(), name
        assert_bitwise(f"d{name}", got[name], grads[name])
    assert_bitwise("d_initial_state", d_state, grads["initial_state"])


@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("steep", [False, True])
def test_read_only_state_pool_needs_no_snapshot(backend, built_plans, K, steep):
    """An indexed input pool without a final-state output stays unchanged and needs no seed snapshot."""
    case = make_case("gdn2", torch.bfloat16, B=8, T=64, H=4, K=K)
    if steep:
        case = steep_gate(case, "spike")
    state0 = random_state(case)
    o_ref, _ = run_fwd(backend, case, initial_state=state0, output_final_state=False, batch_invariant=True)
    pool, slots, _ = state_pool(case, state0, padded=True)
    pool_before = pool.clone()
    o, final_state = run_fwd(backend, case, initial_state=pool, state_indices=slots, output_final_state=False, batch_invariant=True)
    assert final_state.numel() == 0
    assert_bitwise("output", o, o_ref)
    assert_bitwise("read-only pool", pool, pool_before)
    indexed_plans = [plan for plan in built_plans if "state_indices" in plan.node.inputs]
    assert indexed_plans
    for plan in indexed_plans:
        assert not plan.chain and not plan.prep and not plan.has_final_state
        assert "state_snapshot" not in plan.carve_names


@pytest.mark.parametrize("K", [64, 128])
def test_strided_state_overwrite(backend, K):
    """The snapshot follows the state's strides: a padded ``initial_state`` overwritten in place matches the
    separate-buffer run bitwise through a steep relaunch, and its padding is untouched."""
    case = steep_gate(make_case("gdn2", torch.bfloat16, K=K, seq_lens=[96, 32, 160, 1]), "spike")
    state0 = random_state(case)
    o_ref, fs_ref = run_fwd(backend, case, initial_state=state0.clone(), output_final_state=True, batch_invariant=True)
    padded = torch.full((case.N, case.HO, case.V, K + 16), 7.0, device="cuda")
    state = padded[..., :K]
    state.copy_(state0)
    o, _ = run_fwd(backend, case, initial_state=state, output_final_state=True, overwrite_initial_state=True)
    assert_bitwise("o", o, o_ref)
    assert_bitwise("final_state", state, fs_ref)
    assert torch.equal(padded[..., K:], torch.full_like(padded[..., K:], 7.0))


@pytest.mark.parametrize("K", [64, 128])
def test_fp16_inverse_range_without_normalization(backend, K):
    """A moderate prefix can overflow K inv when unnormalized keys exceed one."""
    case = make_case("gdn2", torch.float16, T=32, K=K, H=2)
    q, k, v = torch.zeros_like(case.q), torch.zeros_like(case.k), torch.ones_like(case.v)
    q[..., 0] = 1.0
    k[..., 0] = 5.0
    gates = {name: value.clone() for name, value in case.gates.items()}
    gates["g"].fill_(-13.75 * math.log(2.0) / 16)
    gates["beta"].zero_()
    gates["w"].fill_(1.0)
    case = case.clone(q=q, k=k, v=v, gates=gates)
    assert_fwd_parity(backend, case)
    assert_bwd_parity(backend, case)


@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("shape", list(STEEP_SHAPES))
def test_multiple_steep_channels_are_repeatable(backend, built_plans, K, shape, schedule_shape):
    """The correction reduction has a fixed order in forward and backward."""
    case = steep_gate(make_case("gdn2", torch.bfloat16, K=K, **schedule_shape), "all_channels")
    state0 = random_state(case)
    d_final = random_state(case, seed=SEED + 3)
    o_ref, fs_ref, grads_ref, dO, _ = chain_grads(backend, case, initial_state=state0, d_final_state=d_final)
    for _ in range(8):
        o, fs, grads, _, _ = chain_grads(backend, case, initial_state=state0, d_final_state=d_final, dO=dO)
        assert_bitwise("output", o, o_ref)
        assert_bitwise("final_state", fs, fs_ref)
        for name in grads_ref:
            assert_bitwise(name, grads[name], grads_ref[name])
    assert_schedules(built_plans, shape)


@pytest.mark.parametrize("K", [64, 128])
def test_forward_state_alias_after_graph_warmup(backend, K):
    """An overwrite-capable graph can bind separate buffers before binding an alias."""
    case = steep_gate(make_case("gdn2", torch.bfloat16, B=8, T=64, H=4, K=K), "spike")
    state0 = random_state(case)
    inputs = dict(zip(LEAF_NAMES["gdn2"], op_args(case)))
    inputs.update(cu_seqlens=case.cu, initial_state=state0)
    graph = cudnn.pygraph()
    ports = {name: graph.tensor(list(t.shape), data_type=CUDNN_DTYPE[t.dtype], name=name) for name, t in inputs.items()}
    output, final_state = graph.gdn2(
        **ports,
        scale=1.0 / math.sqrt(K),
        output_final_state=True,
        overwrite_initial_state=True,
        batch_invariant=True,
        name="forward",
    )[:2]
    output.set_output(True).set_data_type(CUDNN_DTYPE[case.dtype])
    final_state.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    build_and_pin(graph, "gdn2_frost")
    o_ref, fs_ref = run_fwd(backend, case, initial_state=state0, output_final_state=True, batch_invariant=True)
    o = torch.empty_like(o_ref)
    separate_state = torch.empty_like(state0)
    state = state0.clone()
    pack = {ports[name]: t for name, t in inputs.items()}
    pack[ports["initial_state"]] = state
    pack.update({output: o, final_state: separate_state})
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    graph.execute(pack, workspace)
    torch.cuda.synchronize()
    assert_bitwise("warmup output", o, o_ref)
    assert_bitwise("warmup state", separate_state, fs_ref)
    pack[final_state] = state
    graph.execute(pack, workspace)
    torch.cuda.synchronize()
    assert_bitwise("rebound output", o, o_ref)
    assert_bitwise("rebound state", state, fs_ref)


@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("T,key_value", [(16, 0.0), (17, 0.0), (32, 0.0), (16, 1.0e-4), (17, 1.0e-4), (32, 1.0e-3)])
@pytest.mark.parametrize("V", [64, 128])
def test_fp16_steep_large_final_state_gradient(backend, K, T, key_value, V):
    """A finite FP32 seed may exceed FP16 even when every returned gradient fits.

    The old decayed state operand is about 0.96 after a full tile; storing its
    undecayed 70000 seed as FP16 produces inf, including inf * 0 for zero keys.
    Small keys also exercise nonzero state-to-token gradient contractions.
    The larger initial state and 32-token key keep beta gradients above the
    FP16 rounding floor, while the 16-token key exposes premature K narrowing.
    """
    case = make_case("gdn2", torch.float16, B=1, T=T, H=1, K=K, V=V)
    gates = {name: value.clone() for name, value in case.gates.items()}
    gates["g"].fill_(-0.7)
    gates["beta"].zero_()
    gates["w"].fill_(1.0)
    case = case.clone(q=torch.zeros_like(case.q), k=torch.full_like(case.k, key_value), v=torch.full_like(case.v, 1.0e-4), gates=gates)
    state0 = random_state(case, scale=50.0)
    d_final = torch.full_like(state0, 70000.0)
    d_o = torch.zeros(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)
    _, _, got, _, _ = chain_grads(backend, case, initial_state=state0, d_final_state=d_final, dO=d_o, batch_invariant=True)
    expected = chain_reference_grads(case, state0, d_o, d_final)
    for name in expected:
        assert_rms_close(f"large final-state seed d{name}", got[name], expected[name], BWD_TOL[case.dtype])
    summary = run_summary_bwd(case, d_o, d_final_state=d_final, batch_invariant=True)
    assert_rms_close("large final-state seed summary", summary, expected["initial_state"], BWD_TOL[case.dtype])


@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("V", [64, 128])
def test_fp16_steep_large_state_gradient_carry(backend, K, V):
    """Finite FP16 output gradients can accumulate a carry above FP16's range."""
    case = make_case("gdn2", torch.float16, B=1, T=32, H=1, K=K, V=V)
    gates = {name: value.clone() for name, value in case.gates.items()}
    gates["g"].zero_()
    gates["g"][:, :16].fill_(-0.7)
    gates["beta"].zero_()
    gates["w"].fill_(1.0)
    q = torch.zeros_like(case.q)
    q[:, 16:, :, 0] = 1.0
    case = case.clone(q=q, k=torch.zeros_like(case.k), v=torch.zeros_like(case.v), gates=gates)
    state0 = torch.zeros_like(random_state(case))
    d_o = torch.zeros(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)
    d_o[16:].fill_(60000.0)
    _, _, got, _, _ = chain_grads(backend, case, initial_state=state0, dO=d_o, batch_invariant=True)
    expected = chain_reference_grads(case, state0, d_o, torch.zeros_like(state0))
    for name in expected:
        assert_rms_close(f"large inter-tile carry d{name}", got[name], expected[name], BWD_TOL[case.dtype])
    summary = run_summary_bwd(case, d_o, batch_invariant=True)
    assert_rms_close("large inter-tile carry summary", summary, expected["initial_state"], BWD_TOL[case.dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("shape", list(STEEP_SHAPES))
def test_steep_cuda_graph_gate_transitions(backend, built_plans, dtype, K, shape, schedule_shape):
    """One captured forward/backward workspace must reset dispatch on every replay."""
    case = make_case("gdn2", dtype, K=K, **schedule_shape)
    leaves = [t.requires_grad_(True) for t in thd_tensors(case)]
    gate = leaves[LEAF_NAMES["gdn2"].index("g")]
    patterns = [case, steep_gate(case, "steady"), case, steep_gate(case, "alternating"), steep_gate(case, "huge_spike"), case]
    gate_inputs = [item.gates["g"].reshape_as(gate).clone() for item in patterns]
    state0 = random_state(case).requires_grad_(True)
    d_final = random_state(case, scale=0.1, seed=SEED + 3)
    d_o = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=dtype)

    def launch():
        o, fs = pinned_op(backend, "gdn2")(*leaves, *op_tail(case), initial_state=state0, output_final_state=True)
        return [o, fs, *torch.autograd.grad([o, fs], leaves + [state0], [d_o, d_final])]

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        captured = launch()
    for index, values in enumerate(gate_inputs):
        with torch.no_grad():
            gate.copy_(values)
        eager = launch()
        graph.replay()
        torch.cuda.synchronize()
        for output_index, (got, expected) in enumerate(zip(captured, eager)):
            assert torch.isfinite(got).all(), f"gate transition {index}, output {output_index} is nonfinite"
            assert_bitwise(f"gate transition {index}, output {output_index}", got, expected)
    assert_schedules(built_plans, shape)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("K", [64, 128])
def test_steep_narrow_gate_ragged_tail(backend, dtype, gate_dtype, K):
    """Low-precision gate loads and padded scan rows share the exact-path contract."""
    case = steep_gate(make_case("gdn2", dtype, K=K, gate_dtype=gate_dtype, seq_lens=[17, 0, 33, 1]), "steady")
    assert_bwd_parity(backend, case, use_initial_state=True, use_dfs=True, l2norm=True)
