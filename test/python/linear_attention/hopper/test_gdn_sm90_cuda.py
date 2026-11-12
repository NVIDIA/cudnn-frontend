# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerics, routing and CUDA-graph behaviour of the ``gdn_hopper_cuda`` sm90 engine.

The shared ``backend`` fixture in ``test_la.py`` (``hopper_cuda``) already runs
this engine through the GDN cases written for every backend. What is covered
here is what that fixture does not reach:

* the gate range. GDN's log decay has no floor -- ``-exp(A_log) * softplus(.)``
  with ``exp(A_log)`` up to 16 is what a freshly initialised model produces --
  and the per-channel sm90 kernels this engine is built from return NaN there,
  because they divide by the in-chunk decay. The scalar-gate build must not,
  so the gate regimes below run from "no decay at all" through "every token
  wipes the state" to hard resets (-1e8, -inf) in the middle of a chunk,
  against the fp64 recurrence, in both directions;
* every kernel variant the hosts can select: the forward with one piece per
  sequence and with many, the three backward scan tilings (16, 32 and 64 value
  columns), single-sequence and packed launches, with and without a partial
  last chunk;
* that the scalar-gate build computes the same function as the per-channel
  one wherever the latter is valid, with ``dG`` equal to its sum over channels;
* that the GDN heuristics hook makes this engine the default on Hopper without
  changing who is eligible, and that a pinned plan still wins;
* that the path captures into a CUDA graph and is run-to-run bitwise stable.
"""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

import cudnn  # noqa: F401 -- import-order requirement, see test/python/conftest.py
from cudnn.linear_attention import gated_delta_net, kimi_delta_attention
from linear_attention.reference_gdn import gdn_reference, rms_ratio

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

ENGINE = "gdn_hopper_cuda"
HOPPER = (9, 0)
DIM = 128
FWD_TOL = 2e-2  # rms_ratio, the bf16 tolerances test_la.py uses
BWD_TOL = 4e-2
STATE_TOL = 2e-2
STATE_GRAD_TOL = 6e-2
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")


def _hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == HOPPER


requires_hopper = pytest.mark.skipif(not _hopper(), reason="the sm90 CUDA GDN engine needs Hopper")


def _has_cutile() -> bool:
    """Whether the optional cuda.tile runtime (the ``[cutile]`` extra) is installed."""
    try:
        return importlib.util.find_spec("cuda.tile") is not None
    except ModuleNotFoundError:  # no ``cuda`` package at all
        return False


def make_gate(kind, total, heads, gen):
    """Log decay per (token, head). Every regime but ``growing`` is <= 0, the documented range."""
    x = torch.randn(total, heads, generator=gen, device="cuda", dtype=torch.float32)
    if kind == "init":  # a freshly initialised model: -A * softplus(alpha + 1), A ~ U(1, 16)
        a = 1 + 15 * torch.rand(heads, generator=gen, device="cuda", dtype=torch.float32)
        return (-a * F.softplus(0.5 * x + 1.0)).contiguous()
    if kind == "steep":  # exp(-20) per token: nothing survives a step
        return torch.full((total, heads), -20.0, device="cuda", dtype=torch.float32)
    if kind == "bounded":  # the KDA production range, (-5, 0)
        return (-5.0 * torch.sigmoid(x)).contiguous()
    if kind == "slow":  # the state outlives any warm-up window
        return (-0.02 * F.softplus(x)).contiguous()
    if kind == "none":  # no decay at all
        return torch.zeros(total, heads, device="cuda", dtype=torch.float32)
    if kind == "reset":  # ordinary decay with hard resets dropped into the middle of chunks
        g = -F.softplus(x)
        token = torch.arange(total, device="cuda")
        g[token % 37 == 5] = -1e8
        g[token % 151 == 20] = float("-inf")
        return g.contiguous()
    if kind == "floor_run":  # runs of floored gates followed by mild ones inside each 16- and 64-token chunk
        g = -0.01 * F.softplus(x)
        token = torch.arange(total, device="cuda")
        g[((token % 64) < 20) | ((token % 16) < 2)] = -1e8
        return g.contiguous()
    if kind == "growing":  # alpha > 1 on most tokens: out of contract, and still the same recurrence
        return (0.02 + 0.02 * x).contiguous()
    raise ValueError(kind)


GATES = ("init", "steep", "bounded", "slow", "none", "reset", "growing")


def make_case(lengths, heads=4, gate="init", seed=7, dim=DIM):
    """THD inputs for packed sequences of the given lengths, plus an upstream gradient."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(lengths)
    q = F.normalize(torch.randn(total, heads, dim, generator=gen, device="cuda", dtype=torch.float32), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(total, heads, dim, generator=gen, device="cuda", dtype=torch.float32), dim=-1).to(torch.bfloat16)
    v = torch.randn(total, heads, dim, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.5
    g = make_gate(gate, total, heads, gen)
    beta = torch.sigmoid(torch.randn(total, heads, generator=gen, device="cuda", dtype=torch.float32)).contiguous()
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    d_out = torch.randn(total, heads, dim, generator=gen, device="cuda", dtype=torch.bfloat16)
    return (q, k, v, g, beta), cu, d_out


def run_engine(tensors, cu, d_out=None, plan=ENGINE, scale=None):
    """``(o, grads)`` from the op; ``grads`` is None when no upstream gradient is given."""
    leaves = [t.clone().requires_grad_(d_out is not None) for t in tensors]
    out = gated_delta_net(*leaves, cu, plan_name=plan, scale=scale)[0]
    grads = torch.autograd.grad(out, leaves, grad_outputs=d_out) if d_out is not None else None
    return out.detach(), grads


def run_reference(tensors, cu, d_out, scale=None):
    leaves = [t.clone().requires_grad_(True) for t in tensors]
    out, _ = gdn_reference(*(t.unsqueeze(0) for t in leaves), cu_seqlens=cu, scale=scale)
    grads = torch.autograd.grad(out, leaves, grad_outputs=d_out.double().unsqueeze(0))
    return out.squeeze(0).detach(), grads


def assert_matches(got, ref, what):
    assert torch.isfinite(got[0].float()).all(), f"{what}: non-finite output"
    err = rms_ratio(got[0], ref[0])
    assert err <= FWD_TOL, f"{what}: o rms_ratio {err:.3e} > {FWD_TOL}"
    for name, mine, theirs in zip(GRAD_NAMES, got[1], ref[1]):
        assert torch.isfinite(mine.float()).all(), f"{what}: non-finite {name}"
        err = rms_ratio(mine, theirs)
        assert err <= BWD_TOL, f"{what}: {name} rms_ratio {err:.3e} > {BWD_TOL}"


@pytest.fixture
def launches(monkeypatch):
    """Records every launch the two sm90 hosts make, as ``(direction, scalar_gate)``."""
    from cudnn.linear_attention.hopper import cuda_bwd_host, cuda_host

    seen = []
    for module, direction in ((cuda_host, "fwd"), (cuda_bwd_host, "bwd")):
        real = module.launch

        def spy(*args, _real=real, _direction=direction, **kwargs):
            seen.append((_direction, bool(kwargs.get("scalar_gate", False))))
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, "launch", spy)
    return seen


# ----------------------------------------------------------------------------- numerics
# What each shape reaches. Forward: P = min(264 // (N * H), chunks) pieces per sequence. Backward scan:
# 64 value columns from 32 (sequence, head) pairs up, 16 for one sequence of exactly 8 heads, 32 otherwise.
SHAPES = [
    pytest.param([257], 2, id="one_seq"),  # many forward pieces; scan<32>, single sequence with a tail
    pytest.param([130, 1, 63, 64, 65], 3, id="ragged"),  # around both chunk sizes (16 forward, 64 backward); scan<32>, packed
    pytest.param([512], 1, id="full_chunks"),  # a multiple of both chunk sizes: the tail-free kernel variants
    pytest.param([200, 300], 16, id="two_seq"),  # exactly 32 pairs: scan<64>, packed
    pytest.param([64] * 40, 8, id="many_seq"),  # more pairs than the forward has CTAs to split with: P == 1
    pytest.param([320], 32, id="one_seq_wide_scan"),  # scan<64>, single sequence, tail-free
    pytest.param([300], 40, id="one_seq_wide_scan_tail"),  # the same with a partial last chunk
    pytest.param([320], 8, id="one_seq_eight_heads"),  # scan<16>, tail-free
    pytest.param([300], 8, id="one_seq_eight_heads_tail"),  # scan<16> with a partial last chunk
]


@requires_hopper
@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("lengths, heads", SHAPES)
def test_matches_the_fp64_recurrence(gate, lengths, heads):
    """Output and all five gradients against the token-by-token fp64 recurrence."""
    tensors, cu, d_out = make_case(lengths, heads, gate)
    got = run_engine(tensors, cu, d_out)
    ref = run_reference(tensors, cu, d_out)
    assert_matches(got, ref, f"gate={gate} lengths={lengths[:5]} H={heads}")


@requires_hopper
@pytest.mark.parametrize("gate", ["init", "slow"])
def test_split_forward_with_wide_scan_matches_the_fp64_recurrence(gate):
    """Four sequences of sixteen chunks at 32 (sequence, head) pairs: eight forward pieces each, and scan<64>.

    On ``init`` a piece warms up from one chunk of history; on ``slow`` no piece can truncate and
    every one walks back to the start of its sequence.
    """
    tensors, cu, d_out = make_case([256] * 4, 8, gate, seed=11)
    got = run_engine(tensors, cu, d_out)
    ref = run_reference(tensors, cu, d_out)
    assert_matches(got, ref, f"gate={gate} 4x256x8")


@requires_hopper
@pytest.mark.parametrize("lengths, heads", [([257], 2), ([130, 1, 63, 64, 65], 3), ([64] * 40, 8)], ids=["one_seq", "ragged", "many_seq"])
def test_agrees_with_the_per_channel_kernels_on_a_bounded_gate(lengths, heads):
    """Where the per-channel kernels are valid, the scalar build is the same function.

    The oracle is ``kda_hopper_cuda`` on the same problem with the gate broadcast
    over the key dimension; its ``dG`` summed over that dimension is GDN's.
    """
    tensors, cu, d_out = make_case(lengths, heads, "bounded")
    got = run_engine(tensors, cu, d_out)

    q, k, v, g, beta = leaves = [t.clone().requires_grad_(True) for t in tensors]
    wide = g.unsqueeze(-1).expand(-1, -1, DIM).contiguous()
    out = kimi_delta_attention(q, k, v, wide, beta, cu, plan_name="kda_hopper_cuda")[0]
    grads = torch.autograd.grad(out, leaves, grad_outputs=d_out)  # autograd sums dG over the broadcast
    assert_matches(got, (out.detach(), grads), f"vs per-channel, lengths={lengths[:5]} H={heads}")


@requires_hopper
def test_a_slow_gate_walks_back_to_the_start():
    """Pieces that cannot truncate their history must replay it, not drop it.

    One sequence and one head gives the forward as many pieces as it has
    chunks. With a gate this slow no piece ever accumulates enough decay to
    start from zero, so every one walks back to token 0; the answer has to be
    the one a single piece would give.
    """
    tensors, cu, _ = make_case([1024], 1, "slow", seed=3)
    split, _ = run_engine(tensors, cu)
    # The same tokens as 300 heads' worth of launch have no room to split: P == 1.
    wide = [t.repeat(1, 300, *([1] * (t.dim() - 2))).contiguous() for t in tensors]
    whole, _ = run_engine(wide, cu)
    torch.cuda.synchronize()
    assert rms_ratio(split[:, 0], whole[:, 0]) <= 2e-3


@requires_hopper
@pytest.mark.parametrize(
    "lengths, heads",
    [([100, 164], 3), ([70, 64, 130, 56], 8), ([320], 8), ([300], 8), ([320], 32), ([300], 32)],
    ids=["scan32", "scan64", "scan16_single", "scan16_single_tail", "scan64_single", "scan64_single_tail"],
)
def test_carried_state_and_its_gradients_match_the_fp64_recurrence(lengths, heads):
    """A non-zero initial_state in, final_state out, and gradients through both.

    The upstream gradient reaches the kernels as dO and d_final_state, and
    d_initial_state comes back: the state ports of all four backward kernels,
    which a zero seed and an unread scratch output would leave unverified.
    """
    tensors, cu, d_out = make_case(lengths, heads, "slow", seed=5)
    gen = torch.Generator(device="cuda").manual_seed(17)
    shape = (len(lengths), heads, DIM, DIM)
    state = (torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32) * 0.05).contiguous()
    d_final = torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32)

    leaves = [t.clone().requires_grad_(True) for t in (*tensors, state)]
    out, final = gated_delta_net(*leaves[:5], cu, initial_state=leaves[5], output_final_state=True, plan_name=ENGINE)
    grads = torch.autograd.grad([out, final], leaves, grad_outputs=[d_out, d_final])

    ref_leaves = [t.clone().requires_grad_(True) for t in (*tensors, state)]
    ref_out, ref_final = gdn_reference(*(t.unsqueeze(0) for t in ref_leaves[:5]), initial_state=ref_leaves[5], cu_seqlens=cu)
    ref_grads = torch.autograd.grad([ref_out, ref_final], ref_leaves, grad_outputs=[d_out.double().unsqueeze(0), d_final.double()])

    assert rms_ratio(out, ref_out.squeeze(0)) <= FWD_TOL
    assert rms_ratio(final, ref_final) <= STATE_TOL
    for name, mine, theirs in zip(GRAD_NAMES, grads, ref_grads):
        err = rms_ratio(mine, theirs)
        assert err <= BWD_TOL, f"{name} rms_ratio {err:.3e} > {BWD_TOL}"
    err = rms_ratio(grads[5], ref_grads[5])
    assert err <= STATE_GRAD_TOL, f"d_initial_state rms_ratio {err:.3e} > {STATE_GRAD_TOL}"


@requires_hopper
@pytest.mark.parametrize("scale", [0.5, 1.0])
@pytest.mark.parametrize("lengths, heads", [([130, 1, 63, 64, 65], 3), ([64] * 8, 8)], ids=["ragged", "wide_scan"])
def test_query_scale_matches_the_fp64_recurrence(scale, lengths, heads):
    """A non-default query scale reaches both directions: the backward takes it as an argument."""
    tensors, cu, d_out = make_case(lengths, heads, "init")
    assert_matches(run_engine(tensors, cu, d_out, scale=scale), run_reference(tensors, cu, d_out, scale=scale), f"scale={scale}")


@requires_hopper
@pytest.mark.parametrize("lengths, heads", [([1000, 37, 515], 4), ([64] * 8, 8)], ids=["ragged", "wide_scan"])
def test_floored_runs_then_mild_gates_match_the_fp64_recurrence(lengths, heads):
    """The in-chunk decays are differences of fp32 prefix sums behind a -128 floor.

    The case that bound is weakest on: a run of floored gates followed by mild
    ones inside the same 16-token (forward) and 64-token (backward) chunk, so
    the differences that matter are small numbers taken between large prefixes.
    """
    tensors, cu, d_out = make_case(lengths, heads, "floor_run")
    assert_matches(run_engine(tensors, cu, d_out), run_reference(tensors, cu, d_out), "floor_run")


@requires_hopper
def test_carried_state_composes():
    """Running a sequence in two calls through final_state / initial_state equals one call."""
    tensors, cu, _ = make_case([384], 4, "slow", seed=5)
    whole = gated_delta_net(*tensors, cu, output_final_state=True, plan_name=ENGINE)
    head = [t[:200].contiguous() for t in tensors]
    tail = [t[200:].contiguous() for t in tensors]
    first = gated_delta_net(*head, torch.tensor([0, 200], device="cuda", dtype=torch.int32), output_final_state=True, plan_name=ENGINE)
    second = gated_delta_net(*tail, torch.tensor([0, 184], device="cuda", dtype=torch.int32), initial_state=first[1], output_final_state=True, plan_name=ENGINE)
    torch.cuda.synchronize()
    assert rms_ratio(torch.cat([first[0], second[0]]), whole[0]) <= FWD_TOL
    assert rms_ratio(second[1], whole[1]) <= FWD_TOL


@requires_hopper
def test_strided_inputs_are_not_misread():
    """A padded q slice must not change the answer (the kernels never see a stride).

    The slice is handed to the op as it is, a view of a wider leaf, so the
    engine sees an operand whose outer stride is padded, in the forward and
    again as the saved q in the backward. The padding is NaN: a kernel or a
    repack that read across it would poison o and every gradient. (dQ itself is
    written to a packed buffer and scattered back into the leaf by autograd.)
    """
    tensors, cu, d_out = make_case([200, 312], 4, "init")
    q = tensors[0]
    ref = run_engine(tensors, cu, d_out)

    wide_q = torch.full((q.shape[0], q.shape[1] * 2, q.shape[2]), float("nan"), dtype=q.dtype, device=q.device)
    wide_q[:, : q.shape[1]] = q
    wide_q.requires_grad_(True)
    strided_q = wide_q[:, : q.shape[1]]
    assert not strided_q.is_contiguous()
    rest = [t.clone().requires_grad_(True) for t in tensors[1:]]
    out = gated_delta_net(strided_q, *rest, cu, plan_name=ENGINE)[0]
    grads = torch.autograd.grad(out, [wide_q, *rest], grad_outputs=d_out)
    torch.cuda.synchronize()

    assert_matches((out.detach(), [grads[0][:, : q.shape[1]], *grads[1:]]), ref, "strided q")


@requires_hopper
@pytest.mark.parametrize("heads", [8, 16], ids=["scan32", "scan64_aliased_workspace"])
def test_run_to_run_bitwise(heads):
    """Same inputs, same launch geometry: identical bytes, forward and backward.

    Sixteen heads puts the launch on the 64-column scan, where three scan
    outputs share storage with the inputs they supersede. This is a replay
    check, not a race test: a few agreeing launches do not show that no
    schedule could disagree (test/AGENTS.md). Batch invariance is not
    provided; the engine declines ``batch_invariant``.
    """

    def raw(t):
        return t.detach().contiguous().view(torch.uint8)

    tensors, cu, d_out = make_case([1000, 37, 515], heads, "init")
    first = run_engine(tensors, cu, d_out)
    for _ in range(3):
        again = run_engine(tensors, cu, d_out)
        torch.cuda.synchronize()
        assert torch.equal(raw(first[0]), raw(again[0])), "o differs between two identical calls"
        for name, a, b in zip(GRAD_NAMES, first[1], again[1]):
            assert torch.equal(raw(a), raw(b)), f"{name} differs between two identical calls"


# ----------------------------------------------------------------------------- routing
@requires_hopper
def test_is_the_default_on_hopper(launches):
    """An unpinned call picks this engine, not slot-1 cuTile, in both directions."""
    tensors, cu, d_out = make_case([512, 512], 4)
    run_engine(tensors, cu, d_out, plan=None)
    torch.cuda.synchronize()
    assert launches == [("fwd", True), ("bwd", True)], f"expected the scalar-gate sm90 kernels, got {launches}"


def test_the_hook_reorders_only_on_hopper(monkeypatch):
    """The ranking itself, independent of which engines this machine can offer.

    ``test_is_the_default_on_hopper`` would also pass with no hook at all on a
    machine where cuTile is not installed, since the sm90 engine is then the
    only one offered.

    The contract: on Hopper the sm90 engine comes first and nothing is added or
    dropped; elsewhere the offered order is returned unchanged. Ids come from
    the manifest, opt-in slots included.
    """
    from cudnn.engines.manifest import MANIFEST
    from cudnn.linear_attention import gdn_heuristics

    (family,) = [f for f in MANIFEST if f.name == "gdn"]
    offered = {name: family.engine_id + s.slot for name, s in family.slots.items()}
    assert ENGINE in offered
    in_order = list(offered.values())
    monkeypatch.setattr(gdn_heuristics, "_current_sm", lambda: 90)
    ranked = [p.engine_id for p in gdn_heuristics.recommend("A", None, offered)]
    assert ranked[0] == offered[ENGINE] and sorted(ranked) == sorted(in_order)
    for sm in (100, 103, 107, 80, 0):
        monkeypatch.setattr(gdn_heuristics, "_current_sm", lambda sm=sm: sm)
        assert [p.engine_id for p in gdn_heuristics.recommend("A", None, offered)] == in_order, f"order changed on sm{sm}"


@requires_hopper
@pytest.mark.skipif(not _has_cutile(), reason="pins gdn_cutile, which needs the optional cuda.tile runtime ([cutile] extra)")
def test_plan_name_still_overrides(launches):
    """Ranking is a default, not a lock: naming another engine still runs it."""
    tensors, cu, d_out = make_case([512], 4, "bounded")
    out, grads = run_engine(tensors, cu, d_out, plan="gdn_cutile")
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all() and all(torch.isfinite(x.float()).all() for x in grads)
    assert launches == [], f"plan_name ignored, the sm90 hosts launched {launches}"


@requires_hopper
def test_declined_graph_falls_through(launches):
    """A head dim the sm90 kernels do not serve must not reach them.

    Without the optional cuTile engine nothing on Hopper serves this graph, so
    the call must decline it, with the 128-only engine's own reason.
    """
    tensors, cu, _ = make_case([512], 4, "bounded", dim=64)
    if not _has_cutile():
        with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="gdn_hopper_cuda: .*head dims must be 128"):
            run_engine(tensors, cu, plan=None)
        assert launches == [], "the 128-only kernels were launched on a 64-dim graph"
        return
    out, _ = run_engine(tensors, cu, plan=None)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert launches == [], "the 128-only kernels served a 64-dim graph"


@requires_hopper
@pytest.mark.parametrize("kwargs", [{"use_qk_l2norm_in_kernel": True}, {"batch_invariant": True}], ids=["l2norm", "batch_invariant"])
def test_out_of_scope_graphs_are_declined(kwargs):
    """Pinning the engine on a graph outside its scope raises instead of mis-serving it."""
    tensors, cu, _ = make_case([256], 4, "bounded")
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        gated_delta_net(*tensors, cu, plan_name=ENGINE, **kwargs)


# ----------------------------------------------------------------------------- CUDA graphs
@requires_hopper
@pytest.mark.parametrize("lengths, heads", [([2048], 12), ([1024] * 4, 32)], ids=["2048x12", "4x1024x32"])
def test_cuda_graph_capture_replays(lengths, heads):
    """The forward captures, and a replay recomputes from the inputs as they are at replay time.

    Capture only works while steady state allocates and synchronises nothing,
    so this fails the moment a per-call allocation creeps back onto the path.
    The inputs are changed in place between capture and replay: a kernel that
    ran eagerly during capture instead of being recorded would leave the
    captured output holding the old answer.
    """
    tensors, cu, _ = make_case(lengths, heads, "init")
    call = lambda: gated_delta_net(*tensors, cu, plan_name=ENGINE)[0]  # noqa: E731

    for _ in range(5):  # warm everything lazy: NVRTC compile, graph build
        before = call().clone()
    torch.cuda.synchronize()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = call()

        tensors[2].mul_(-0.5)  # v, in place: same buffers, different problem
        graph.replay()
        torch.cuda.synchronize()
        after = call()
        torch.cuda.synchronize()
        assert torch.equal(captured, after), "the replay did not reproduce an eager call on the same inputs"
        assert not torch.equal(captured, before), "the replay did not recompute from the changed inputs"
    finally:
        graph.reset()
