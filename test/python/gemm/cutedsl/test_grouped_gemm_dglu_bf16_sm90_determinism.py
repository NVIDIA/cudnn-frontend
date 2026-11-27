# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Deterministic dprob for the SM90 (Hopper) BF16 grouped dGLU kernel.

The SM90 counterpart of test_grouped_gemm_dglu_bf16_determinism.py. By default each N tile adds its
``dprob[token]`` partial with an fp32 atomic, so dprob depends on tile arrival order once N spans more
than one tile (128 or 256 columns). ``deterministic=True`` gives each (token, N-tile) partial its own
fp32 slot and sums the slots in a fixed order on the launch stream.

The suite covers both activations, both SM90 tiles and the clusters that round the slot count up,
ragged and empty experts and capacity past the last expert. dprob and D are compared as raw bytes
across replays under concurrent work and across expert reordering, duplication and neighbor changes;
the Torch reference and the atomic path are separate numerical checks.
"""

from __future__ import annotations

import pytest
import torch

from cuda.bindings import driver as cuda

from cudnn.tensor_adapter import has_wgmma

from determinism_utils import assert_bitwise, assert_replays
from gemm.cutedsl.test_grouped_gemm_glu_sm90 import check_dglu, make_problem, run_dglu, to_gate_then_up

pytestmark = pytest.mark.L0

# Unequal experts with an empty one; TAIL rows of capacity past the last offset are never addressed.
COUNTS = (512, 0, 256, 768)
TAIL = 256


@pytest.fixture(autouse=True)
def require_sm90():
    if not torch.cuda.is_available() or not has_wgmma(torch.cuda.get_device_capability()[0]):
        pytest.skip("SM90 is required")


def problem_with_tail(counts=COUNTS, n=1024, k=256, seed=0, tail=TAIL):
    problem = make_problem(counts, n=n, k=k, seed=seed)
    if tail:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        extra = lambda like, cols: torch.randn((tail, *cols), generator=generator, device="cuda").to(like.dtype)  # noqa: E731
        problem.update(
            a=torch.cat([problem["a"], extra(problem["a"], (k,))]),
            c=torch.cat([problem["c"], extra(problem["c"], (2 * n,))]),
            prob=torch.cat([problem["prob"], torch.rand((tail,), generator=generator, device="cuda")]),
        )
    return problem


def outputs(result, valid_m):
    """dprob in full; D up to the last expert offset (capacity past it is unspecified)."""
    return {"dprob": result["dprob_tensor"], "d_row": result["d_row_tensor"][:valid_m]}


def compiled_slots():
    from cudnn.gemm.cutedsl.grouped.dglu import api as dglu_api

    plans = [o._implementation for o in dglu_api._cache_of_GroupedGemmDgluSm100Objects.values()]
    return {plan._dprob_slots for plan in plans if getattr(plan, "_deterministic", False)}


TILES = [
    ((128, 256), None, 4),  # default cluster (2, 1): four 256-wide N tiles at N=1024
    ((128, 128), (1, 1), 8),
    ((128, 256), (1, 2), 4),
    ((128, 128), (2, 2), 8),
]


@pytest.mark.parametrize("tile,cluster,slots", TILES, ids=["128x256-default", "128x128-c11", "128x256-c12", "128x128-c22"])
@pytest.mark.parametrize("act", ["dswiglu", "dgeglu"])
def test_sm90_deterministic_dprob(act, tile, cluster, slots):
    """Bit-exact dprob over replays, the same D as the atomic path, and the reference's values."""
    problem = problem_with_tail()
    valid_m = sum(COUNTS)
    launch = dict(act=act, mma_tiler_mn=tile, cluster_shape_mn=cluster)
    atomic = {name: t.clone() for name, t in outputs(run_dglu(problem, **launch, deterministic=False), valid_m).items()}
    # Eight runs on fresh outputs with concurrent work on another stream; raw bytes.
    det = assert_replays(lambda: outputs(run_dglu(problem, **launch, deterministic=True), valid_m))

    assert slots in compiled_slots(), "the deterministic slot path did not compile with one slot per SM90 N tile"
    assert torch.count_nonzero(det["dprob"][:valid_m]).item() > 0
    # D has one writer per element; the flag must leave it byte-identical.
    assert_bitwise({"d_row": det["d_row"]}, {"d_row": atomic["d_row"]}, "deterministic vs atomic")
    torch.testing.assert_close(det["dprob"], atomic["dprob"], rtol=1e-4, atol=1e-4)
    # Nothing is added past the last expert offset.
    assert_bitwise({"tail": det["dprob"][valid_m:]}, {"tail": torch.zeros_like(det["dprob"][valid_m:])})
    check_dglu(problem, {"d_row_tensor": det["d_row"], "dprob_tensor": det["dprob"]}, act)


@pytest.mark.parametrize(
    "n,tile,cluster,slots",
    [(640, (128, 256), (1, 2), 4), (96, (128, 256), None, 1), (416, (128, 128), (2, 2), 4)],
    ids=["N640-128x256-c12", "N96-default", "N416-128x128-c22"],
)
def test_sm90_deterministic_dprob_partial_tiles(n, tile, cluster, slots):
    """N off the tile grid: a (1, 2) cluster rounds the slots up to whole clusters (3 tiles -> 4 slots)."""
    problem = problem_with_tail(counts=(256, 512, 0, 256), n=n, k=192, seed=n)
    valid_m = sum(problem["counts"])
    launch = dict(mma_tiler_mn=tile, cluster_shape_mn=cluster)
    atomic = outputs(run_dglu(problem, **launch), valid_m)
    det = assert_replays(lambda: outputs(run_dglu(problem, **launch, deterministic=True), valid_m))
    assert slots in compiled_slots()
    assert_bitwise({"d_row": det["d_row"]}, {"d_row": atomic["d_row"]})
    torch.testing.assert_close(det["dprob"], atomic["dprob"], rtol=1e-4, atol=1e-4)
    check_dglu(problem, {"d_row_tensor": det["d_row"], "dprob_tensor": det["dprob"]})


@pytest.mark.parametrize(
    "n,tile,cluster",
    [(1024, (128, 256), None), (640, (128, 256), (1, 2)), (384, (128, 128), (2, 2))],
    ids=["N1024-128x256-default", "N640-128x256-c12", "N384-128x128-c22"],
)
@pytest.mark.parametrize("act", ["dswiglu", "dgeglu"])
def test_sm90_deterministic_dprob_gate_then_up(act, n, tile, cluster):
    """glu_layout="gate_then_up": bit-exact dprob and D over replays, the interleaved_32 bytes (D in branch order)."""
    problem = problem_with_tail(counts=(256, 512, 0, 256), n=n, k=192, seed=n + 1)
    valid_m = sum(problem["counts"])
    launch = dict(act=act, mma_tiler_mn=tile, cluster_shape_mn=cluster, deterministic=True)
    interleaved = {name: t.clone() for name, t in outputs(run_dglu(problem, **launch), valid_m).items()}
    gate_then_up = dict(problem, c=to_gate_then_up(problem["c"], 1))
    det = assert_replays(lambda: outputs(run_dglu(gate_then_up, glu_layout="gate_then_up", **launch), valid_m))
    assert_bitwise({"dprob": det["dprob"]}, {"dprob": interleaved["dprob"]}, "gate_then_up vs interleaved_32 dprob")
    d_row = lambda t: t.reshape(t.shape[0], -1)  # noqa: E731
    assert_bitwise({"d_row": d_row(det["d_row"])}, {"d_row": to_gate_then_up(d_row(interleaved["d_row"]), 1)}, "gate_then_up vs interleaved_32 D")


@pytest.mark.parametrize("act", ["dswiglu", "dgeglu"])
def test_sm90_deterministic_dprob_expert_batch_invariance(act):
    """An expert's dprob and D bytes do not depend on which experts share the launch."""
    counts = (512, 256, 768)
    problem = make_problem(counts, n=1024, k=256, seed=1)
    a, c, prob, weights = problem["a"], problem["c"], problem["prob"], problem["weights"]
    saved = {"a": a.clone(), "c": c.clone(), "weights": weights.clone(), "prob": prob.clone()}
    begins = [0, *torch.tensor(counts).cumsum(0).tolist()]

    def invoke(order, a=a, c=c):
        rows = torch.cat([torch.arange(begins[e], begins[e + 1], device="cuda") for e in order])
        sizes = [counts[e] for e in order]
        sub = dict(
            problem,
            a=a[rows].contiguous(),
            c=c[rows].contiguous(),
            prob=prob[rows].contiguous(),
            weights=weights[list(order)].contiguous(),
            alpha=problem["alpha"][list(order)].contiguous(),
            beta=problem["beta"][list(order)].contiguous(),
            offsets=torch.tensor(sizes, dtype=torch.int64).cumsum(0).to(torch.int32).cuda(),
            counts=tuple(sizes),
        )
        return {name: t.clone() for name, t in outputs(run_dglu(sub, act, deterministic=True), len(rows)).items()}, rows

    packed, _ = invoke((0, 1, 2))
    for order in ((2,), (2, 0, 1), (1, 1)):  # alone, reordered, duplicated
        result, rows = invoke(order)
        assert_bitwise(result, {name: t[rows] for name, t in packed.items()}, f"order {order}")
    # Neighbor isolation: change expert 1's tokens; experts 0 and 2 keep their bytes.
    neighbor_a, neighbor_c = a.clone(), c.clone()
    neighbor_a[begins[1] : begins[2]].zero_()
    neighbor_c[begins[1] : begins[2]].fill_(0.25)
    changed, _ = invoke((0, 1, 2), a=neighbor_a, c=neighbor_c)
    for e in (0, 2):
        rows = slice(begins[e], begins[e + 1])
        assert_bitwise({k: t[rows] for k, t in changed.items()}, {k: t[rows] for k, t in packed.items()}, f"expert {e}")
    for name, tensor in (("a", a), ("c", c), ("weights", weights), ("prob", prob)):
        assert_bitwise({name: tensor}, {name: saved[name]}, "inputs mutated")


@pytest.mark.parametrize("busy", ["current_stream", "launch_stream"])
def test_sm90_deterministic_dprob_on_launch_stream(busy):
    """The slots are zeroed and reduced on the launch stream. A busy current stream would let the
    zero-fill land after the kernel; a busy launch stream would let the reduction run before it."""
    problem = problem_with_tail(seed=2)
    expected = run_dglu(problem, deterministic=True)["dprob_tensor"].clone()  # also compiles
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        dprob = torch.zeros_like(problem["prob"])  # the caller orders its own dprob before the launch
        if busy == "launch_stream":
            torch.cuda._sleep(200_000_000)
    if busy == "current_stream":
        torch.cuda._sleep(200_000_000)
    actual = run_dglu(problem, dprob=dprob, deterministic=True, current_stream=cuda.CUstream(side.cuda_stream))["dprob_tensor"]
    torch.cuda.synchronize()
    assert_bitwise({"dprob": actual}, {"dprob": expected})


def test_sm90_deterministic_empty_experts():
    """All-empty experts and a rank with no routed tokens: dprob stays zero / empty under the flag."""
    problem = make_problem((0, 0, 0), n=512, k=256)
    result = run_dglu(problem, deterministic=True)
    torch.cuda.synchronize()
    assert torch.count_nonzero(result["dprob_tensor"]) == 0
    problem = make_problem((0, 0), n=448, k=192)
    problem.update(a=problem["a"][:0], prob=problem["prob"][:0], c=problem["c"][:0])
    result = run_dglu(problem, deterministic=True)
    torch.cuda.synchronize()
    assert tuple(result["d_row_tensor"].shape) == (0, 896) and result["dprob_tensor"].numel() == 0


def test_sm90_deterministic_rejects_dbias():
    problem = make_problem((256,), n=512, k=256)
    with pytest.raises(NotImplementedError, match="dbias"):
        run_dglu(problem, deterministic=True, generate_dbias=True)


def test_sm90_deterministic_rejects_jax():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    problem = make_problem((256,), n=512, k=256)

    def to_jax(tensor):
        host = tensor.cpu()
        return jnp.asarray(host.float().numpy()).astype(jnp.bfloat16) if host.dtype == torch.bfloat16 else jnp.asarray(host.numpy())

    arrays = {name: to_jax(problem[name]) for name in ("a", "c", "weights", "offsets", "alpha", "beta", "prob")}
    jax.block_until_ready(arrays["a"])
    import cudnn

    with pytest.raises(NotImplementedError, match="torch tensors"):
        cudnn.grouped_gemm_dglu_wrapper_sm90(
            a_tensor=arrays["a"],
            c_tensor=arrays["c"],
            sfa_tensor=None,
            padded_offsets=arrays["offsets"],
            alpha_tensor=arrays["alpha"],
            beta_tensor=arrays["beta"],
            prob_tensor=arrays["prob"],
            dprob_tensor=jnp.zeros(arrays["prob"].shape, dtype=jnp.float32),
            b_tensor=arrays["weights"],
            d_dtype="bfloat16",
            deterministic=True,
        )


@pytest.mark.L1
def test_sm90_atomic_dprob_varies_at_scale():
    """DeepSeek-V3 FC2 dgrad (N=2048, K=7168, 8 experts x 1024 tokens): eight 256-wide N tiles per token.

    The atomic dprob changes between runs at this shape on H100; the deterministic one does not, and D
    never depends on the flag. Run with -rP to see how many replays and rows the atomic path changed.
    """
    problem = make_problem((1024,) * 8, n=2048, k=7168, seed=3)
    valid_m = sum(problem["counts"])
    runs = 20
    first = outputs(run_dglu(problem), valid_m)
    first = {name: t.clone() for name, t in first.items()}
    changed_runs, changed_rows = 0, 0
    for _ in range(runs - 1):
        again = outputs(run_dglu(problem), valid_m)
        rows = (again["dprob"].view(torch.int32) != first["dprob"].view(torch.int32)).sum().item()
        changed_runs += rows > 0
        changed_rows = max(changed_rows, rows)
        assert_bitwise({"d_row": again["d_row"]}, {"d_row": first["d_row"]})
    print(f"atomic dprob changed in {changed_runs} of {runs - 1} replays, up to {changed_rows} of {valid_m} rows")
    det = assert_replays(lambda: outputs(run_dglu(problem, deterministic=True), valid_m), repeats=runs)
    assert_bitwise({"d_row": det["d_row"]}, {"d_row": first["d_row"]})
    torch.testing.assert_close(det["dprob"], first["dprob"], rtol=1e-3, atol=1e-3)
    assert changed_runs > 0, "the atomic dprob never changed, so this shape does not exercise tile reordering here"
