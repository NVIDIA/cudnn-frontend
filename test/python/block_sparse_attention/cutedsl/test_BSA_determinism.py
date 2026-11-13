# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise replay and batch invariance of the blk64 block-sparse attention API (``sparse_block_size=64``).

The suite drives the public ``block_sparse_attention_forward`` / ``block_sparse_attention_backward`` /
``block_sparse_attention_prepare_backward`` entry points on SM90 (the blk64 kernels this PR changes) and, through the
same API, the SM100/SM103 blk64 path. Every case is L0.

Forward (BF16 and FP16): four configurations, each run in both datatypes for replay and for batch invariance:
fixed-count lists; variable counts with empty rows (``allow_empty_block_nums=True``) and partial KV blocks through
``block_sizes``; fixed counts with partial blocks and ``kv_splits=2``; and a 512-query-tile grid that crosses the
device's scheduling waves over a KV length that is not a multiple of 64 (with ``block_sizes`` marking the final
block, as the contract requires). Replay compares O and LSE as raw bytes across fresh buffers and concurrent
side-stream work; batch checks compare a batch element alone with its slice of packed, reordered, duplicated and
head-permuted batches and with changed neighbor data and masks; the ``-inf`` LSE sentinel is allowed exactly on the
empty rows, whose O must be zero; an FP32 reference is a separate numerical check; one small golden per datatype
records the hashes. The SM100/SM103 blk64 forward serves BF16 only: its FP16 cases skip as Unsupported by contract,
each after the public API has rejected a tiny FP16 call (``skip_by_contract``), so a widened contract fails instead
of skipping.

Backward (BF16, head dimension 128, MHA by contract) on its routes: the regular kernel with direct dK/dV stores
(one Q group, whole KV blocks), the regular kernel with the FP32 dK/dV workspace (several Q groups), the paired-KV
kernel (forced through ``_BLK64_PAIRED_KV_MIN_KV_TASKS`` as the sibling tests do, several Q groups), and on SM90
the prepared-metadata route (``block_sparse_attention_prepare_backward`` + ``metadata=``) over the regular and the
paired kernels (off SM90 those cases skip by contract, again only after the API has rejected the prepare call).
Every route is asserted through the dispatcher's own decisions, checked against the FP32 reference, and its
empty-row dQ must be zero; the prepared route must match the default route numerically and, where the arithmetic
is order-free, bitwise.

Known gap, stated rather than tolerated (support page: Not guaranteed): the backward reduce-adds dQ in FP32 across
the CTAs of a query block's KV blocks (``cp.reduce.async.bulk.add.f32``), the FP32 dK/dV workspace and the paired
kernel's later arrivers reduce-add dK/dV the same way, and the K-to-Q CSR build orders each KV block's query list
with an atomic cursor, so even the direct dK/dV stores accumulate a KV block's query blocks in an order that changes
from call to call. The replay and batch cases therefore MEASURE the differing bytes of dQ, dK and dV
(``record_property``, the captured output and the run's ``measured`` summary through ``frost_routing.measured``, which
reaches the CI job log of a passing run; the coverage report records them) instead of hiding a failure behind
a skip or xfail, and assert bytes only where the arithmetic is order-free: the regular kernel's direct dK/dV stores
over prepared metadata (one CSR build; ``REPLAY_STABLE``) and the goldens' patterns (one query block per KV block,
at most two reduce-adds onto zero per dQ tile), which establish bytes for those patterns and nothing more.
"""

import importlib
import re
from dataclasses import dataclass, replace
from typing import Optional

import pytest
import torch

import frost_routing
from block_sparse_attention.cutedsl.bsa_reference import attention_reference, block_sparse_mask
from block_sparse_attention.cutedsl.bsa_utils import import_bsa
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from test_utils import bitwise_bits

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

BLOCK = 64
DIM = 128
POISON_INDEX = 2147483647  # inactive q2k_block_index slots: the contract says they are ignored
ALLOWED_INF = {"lse": -torch.inf}
ARCHS = (90, 100, 103)
# Q blocks per bucketed K2Q CSR group where the backward runs several Q groups (the regular kernel then reduce-adds
# dK/dV through the FP32 workspace); every paired case uses it so the paired kernel compiles once.
GROUP_BUCKET = 5
# Backward routes: the dispatcher decisions each one must take.
ROUTES = {
    "regular": dict(paired=False, direct=True, bucket=None),
    "grouped": dict(paired=False, direct=False, bucket=GROUP_BUCKET),
    "paired": dict(paired=True, direct=False, bucket=GROUP_BUCKET),
}
PREPARED_ROUTES = ("regular", "paired")
GRADIENTS = ("dq", "dk", "dv")
# Gradients whose replay bytes are asserted, per route. Only the regular kernel's direct dK/dV stores over PREPARED
# metadata qualify: the CSR is then built once, each KV block's single owner CTA accumulates its query blocks in that
# fixed order and stores the tile with no reduce-add. The default route rebuilds the CSR on every call (atomic scatter
# cursor), the FP32 workspace and the paired kernel reduce-add dK/dV, and dQ is reduce-added on every route.
REPLAY_STABLE = {"prepared-regular": ("dk", "dv")}
# Three batch elements, two heads, 8 x 8 blocks, four-block lists (base forward cases) and 16 x 16 blocks, six-block
# lists (backward measurement cases: every KV block collects several query blocks and every query block several KV
# blocks, so any order dependence of the reduce-adds shows). The stress case runs 512 query tiles over 40 KV blocks
# with ten-block lists and a 47-token KV tail (its ``block_sizes`` marks the final block, as the contract requires).
FORWARD_SHAPE = dict(batch=3, heads=2, q_blocks=8, kv_blocks=8, width=4)
STRESS_SHAPE = dict(batch=2, heads=4, q_blocks=64, kv_blocks=40, width=10, kv_tail=17)
BACKWARD_SHAPE = dict(batch=3, heads=2, q_blocks=16, kv_blocks=16, width=6)
# The forward configurations (shape, counts, block sizes, kv_splits); every one runs in both datatypes, for replay and
# for batch invariance, so each configuration the support page names carries both kinds of evidence per datatype.
FORWARD_CONFIGS = {
    "fixed-full": (FORWARD_SHAPE, "fixed", None, 1),
    "variable-partial": (FORWARD_SHAPE, "variable", "partial", 1),
    "fixed-partial-split2": (FORWARD_SHAPE, "fixed", "partial", 2),
    "variable-kvtail-512tiles": (STRESS_SHAPE, "variable", "tail", 1),
}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
FORWARD_CASES = {f"{name}-{config}": (DTYPES[name], *FORWARD_CONFIGS[config]) for name in DTYPES for config in FORWARD_CONFIGS}
# Off SM90 the suite skips a configuration as Unsupported by contract only after the public API has rejected a tiny
# instance of it: the text the rejection must carry, and the clause the skip names.
CONTRACT = {
    "fp16": ("BF16", "the SM100/SM103 blk64 forward serves BF16 only"),
    "prepared": ("SM90 blk64 backward only", "block_sparse_attention_prepare_backward serves the SM90 blk64 backward only"),
}


@dataclass(frozen=True)
class Problem:
    """One batch of sequences: per-element Q/K/V/dO and sparse metadata, with the batch's shared ``block_sizes``."""

    samples: tuple  # dicts with q, k, v, do (1, H, S, D), idx (1, H, nq, width) and counts (1, H, nq) or None
    sizes: Optional[torch.Tensor]  # rank-1 block_sizes shared by the batch, or None for full blocks
    width: int
    seqlen_q: int
    seqlen_k: int
    dtype: torch.dtype


@pytest.fixture
def api():
    """The BSA API on a supported blk64 device, with TF32 off for the FP32 references."""
    ops = import_bsa(
        archs=ARCHS, min_cutedsl=(4, 8, 0), feature="the blk64 determinism suite", reason="the blk64 determinism suite requires SM90, SM100 or SM103"
    )
    major, minor = torch.cuda.get_device_capability()
    # Not monkeypatch: torch resolves this flag in __getattr__/__setattr__, so monkeypatch records it as unset and
    # tries to delete it at teardown.
    matmul, previous = torch.backends.cuda.matmul, torch.backends.cuda.matmul.allow_tf32
    matmul.allow_tf32 = False
    try:
        yield ops, importlib.import_module("cudnn.block_sparse_attention._interface"), major * 10 + minor
    finally:
        matmul.allow_tf32 = previous


def make_problem(*, dtype, batch, heads, q_blocks, kv_blocks, width, counts, sizes, kv_tail=0, seed):
    """Build the batch once; ``counts`` is ``fixed`` or ``variable`` (empty rows, poisoned inactive slots) and ``sizes``
    is None, ``partial`` (random block sizes), ``tail`` (full blocks, the final one shortened to the KV tail) or a
    tensor to share with another problem."""
    rng = torch.Generator().manual_seed(seed)
    seqlen_q, seqlen_k = q_blocks * BLOCK, kv_blocks * BLOCK - kv_tail
    samples = []
    for _ in range(batch):
        sample = {
            "q": torch.randn(1, heads, seqlen_q, DIM, generator=rng).to(dtype),
            "k": torch.randn(1, heads, seqlen_k, DIM, generator=rng).to(dtype),
            "v": torch.randn(1, heads, seqlen_k, DIM, generator=rng).to(dtype),
            "do": torch.randn(1, heads, seqlen_q, DIM, generator=rng).to(dtype),
            # Distinct KV block ids per row: the prefix of a random permutation.
            "idx": torch.stack([torch.randperm(kv_blocks, generator=rng)[:width] for _ in range(heads * q_blocks)]).view(1, heads, q_blocks, width).int(),
            "counts": None,
        }
        if counts == "variable":
            nums = torch.randint(0, width + 1, (1, heads, q_blocks), generator=rng).int()
            nums[..., ::5] = 0  # empty rows
            nums[..., 1] = width  # at least one full list per head
            sample["idx"].masked_fill_(torch.arange(width) >= nums[..., None], POISON_INDEX)
            sample["counts"] = nums
        samples.append({name: value.cuda() if value is not None else None for name, value in sample.items()})
    if isinstance(sizes, torch.Tensor):
        block_sizes = sizes
    elif sizes == "partial":
        block_sizes = torch.randint(1, BLOCK + 1, (kv_blocks,), generator=rng).int()
        block_sizes[::3] = BLOCK
        block_sizes[-1].clamp_(max=seqlen_k - (kv_blocks - 1) * BLOCK)
        block_sizes = block_sizes.cuda()
    elif sizes == "tail":
        # The contract asks for block_sizes whenever the final block is partially valid (bsa.md, "Sparse metadata").
        block_sizes = torch.full((kv_blocks,), BLOCK, dtype=torch.int32)
        block_sizes[-1] = seqlen_k - (kv_blocks - 1) * BLOCK
        block_sizes = block_sizes.cuda()
    else:
        assert sizes is None
        block_sizes = None
    return Problem(tuple(samples), block_sizes, width, seqlen_q, seqlen_k, dtype)


def forward_problem(case, seed=20261109):
    dtype, shape, counts, sizes, _ = FORWARD_CASES[case]
    return make_problem(dtype=dtype, counts=counts, sizes=sizes, seed=seed, **shape)


def backward_problem(seed=20261110, sizes="partial"):
    return make_problem(dtype=torch.bfloat16, counts="variable", sizes=sizes, seed=seed, **BACKWARD_SHAPE)


def golden_problem(dtype, counts):
    """Small, order-free patterns: query block 0 attends KV blocks 0 and 1, query block 1 attends 2 and 3 (head 1
    attends block 0 only from query block 0, so block 1 is never visited there); with variable counts a third,
    empty query block. Every KV block has at most one query block and every dQ tile at most two reduce-adds."""
    problem = make_problem(
        dtype=dtype, batch=1, heads=2, q_blocks=3 if counts == "variable" else 2, kv_blocks=4, width=2, counts=counts, sizes=None, seed=20261111
    )
    sample = dict(problem.samples[0])
    sample["idx"] = torch.tensor([[[0, 1], [2, 3], [POISON_INDEX, POISON_INDEX]][: problem.seqlen_q // BLOCK]] * 2, dtype=torch.int32).view(1, 2, -1, 2).cuda()
    if counts == "variable":
        sample["counts"] = torch.tensor([[[2, 2, 0], [1, 2, 0]]], dtype=torch.int32).cuda()
        sample["idx"][0, 1, 0, 1] = POISON_INDEX
    return replace(problem, samples=(sample,), sizes=torch.tensor([64, 40, 64, 23], dtype=torch.int32).cuda())


def pack(problem, samples):
    """Concatenate the sequences along the batch axis; the metadata stays per element, ``block_sizes`` is shared."""
    inputs = {name: torch.cat([sample[name] for sample in samples]) for name in ("q", "k", "v", "do", "idx")}
    inputs["counts"] = None if samples[0]["counts"] is None else torch.cat([sample["counts"] for sample in samples])
    inputs["sizes"] = problem.sizes
    inputs["width"] = problem.width
    return inputs


def permute_heads(inputs, perm):
    return {**inputs, **{name: inputs[name][:, perm] for name in ("q", "k", "v", "do", "idx", "counts") if inputs[name] is not None}}


def piece(result, position):
    return {name: value[position : position + 1] for name, value in result.items()}


def tensors_of(inputs):
    return {name: value for name, value in inputs.items() if isinstance(value, torch.Tensor)}


def empty_rows(inputs):
    """Bool (B, H, S_q) mask of the query rows whose block has no KV blocks."""
    counts, seqlen_q = inputs["counts"], inputs["q"].shape[2]
    if counts is None:
        return torch.zeros(inputs["q"].shape[:3], dtype=torch.bool, device=inputs["q"].device)
    return (counts == 0).repeat_interleave(BLOCK, dim=2)[..., :seqlen_q]


def reference_sizes(inputs):
    if inputs["sizes"] is not None:
        return inputs["sizes"]
    seqlen_k = inputs["k"].shape[2]
    sizes = torch.full(((seqlen_k + BLOCK - 1) // BLOCK,), BLOCK, dtype=torch.int32, device="cuda")
    sizes[-1] = seqlen_k - (sizes.numel() - 1) * BLOCK
    return sizes


def reference_mask(inputs):
    return block_sparse_mask(inputs["idx"], inputs["width"], reference_sizes(inputs), inputs["q"].shape[2], inputs["k"].shape[2], BLOCK, inputs["counts"])


def run_forward(ops, inputs, *, kv_splits=1):
    """Run the public forward on clones; check the output names, the empty-row sentinels and that no input changed."""
    local = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in inputs.items()}
    before = snapshot(tensors_of(local))
    output = ops.block_sparse_attention_forward(
        local["q"],
        local["k"],
        local["v"],
        local["idx"],
        local["width"],
        local["sizes"],
        local["counts"],
        sparse_block_size=BLOCK,
        allow_empty_block_nums=local["counts"] is not None,
        kv_splits=kv_splits,
    )
    assert tuple(output.keys()) == ("o_tensor", "lse_tensor")
    result = {"o": output["o_tensor"], "lse": output["lse_tensor"]}
    assert result["o"].dtype == inputs["q"].dtype and result["lse"].dtype == torch.float32
    empty = empty_rows(local)
    assert torch.equal(result["lse"] == -torch.inf, empty), "lse: -inf is allowed exactly on the empty rows"
    assert torch.count_nonzero(result["o"][empty]).item() == 0, "o: an empty row produced a nonzero output"
    assert_bitwise(tensors_of(local), before, "kernel inputs changed")
    return snapshot(result, allowed_inf=ALLOWED_INF)


def check_forward_reference(inputs, actual):
    o_ref, lse_ref = attention_reference(inputs["q"], inputs["k"], inputs["v"], reference_mask(inputs))
    torch.testing.assert_close(actual["o"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual["lse"], lse_ref, atol=2e-3, rtol=2e-3)


def reference_backward(inputs):
    """FP32 autograd gradients; empty rows (all masked) get zero probabilities instead of NaN."""
    mask = reference_mask(inputs)
    q, k, v = [inputs[name].float().detach().requires_grad_() for name in ("q", "k", "v")]
    scores = q @ k.transpose(-1, -2) / DIM**0.5 + mask
    nonempty = torch.isfinite(mask).any(-1, keepdim=True)
    probabilities = torch.where(nonempty, scores, 0).softmax(-1).masked_fill(~torch.isfinite(mask), 0)
    (probabilities @ v).backward(inputs["do"].float())
    return dict(zip(GRADIENTS, (q.grad, k.grad, v.grad)))


def check_backward_reference(inputs, actual):
    for name, expected in reference_backward(inputs).items():
        torch.testing.assert_close(actual[name].float(), expected, atol=3e-2, rtol=3e-2)


class RouteProbe:
    """Records the dispatcher's own decisions: the paired-KV gate and whether the FP32 dK/dV workspace was allocated."""

    def __init__(self, monkeypatch, interface):
        self.paired, self.workspace = [], []
        gate, allocate = interface._blk64_use_paired_kv, interface._empty_bwd_workspace_with_zeroed_accum

        def spy_gate(*args, **kwargs):
            decision = gate(*args, **kwargs)
            self.paired.append(bool(decision))
            return decision

        def spy_allocate(**kwargs):
            self.workspace.append(bool(kwargs["include_dkv_accum"]))
            return allocate(**kwargs)

        monkeypatch.setattr(interface, "_blk64_use_paired_kv", spy_gate)
        monkeypatch.setattr(interface, "_empty_bwd_workspace_with_zeroed_accum", spy_allocate)

    def expect(self, *, paired, direct, prepared=None):
        assert self.workspace and self.workspace[-1] == (not direct), f"dK/dV workspace: {self.workspace}"
        if prepared is None:
            assert self.paired and self.paired[-1] == paired, f"paired-KV gate: {self.paired}"
        else:
            assert prepared.use_paired_kv == paired and prepared.direct_kv == direct, repr(prepared)


def force_paired(monkeypatch, interface):
    """Route eligible variable-count calls through the paired-KV kernel regardless of the task-count gate (as the
    sibling tests do); the dispatcher's decision is still asserted by the probe."""
    monkeypatch.setattr(interface, "_BLK64_PAIRED_KV_MIN_KV_TASKS", 0)


def prepare(ops, inputs, bucket):
    return ops.block_sparse_attention_prepare_backward(
        inputs["idx"],
        inputs["width"],
        inputs["sizes"],
        inputs["counts"],
        seqlen_q=inputs["q"].shape[2],
        seqlen_k=inputs["k"].shape[2],
        bucket_size_blocks=bucket,
    )


def run_backward(ops, inputs, forward, *, bucket=None, metadata=None):
    """Run the public backward into fresh NaN-poisoned buffers; check the names, the buffers, the empty-row dQ and
    that no input changed."""
    local = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in inputs.items()}
    local["o"], local["lse"] = forward["o"].clone(), forward["lse"].clone()
    # A prepared object pins the exact metadata tensors it was built from, so that route reads the caller's.
    metadata_tensors = {name: inputs[name] for name in ("idx", "sizes", "counts") if inputs[name] is not None}
    before = snapshot({**tensors_of(local), **{f"caller {name}": value for name, value in metadata_tensors.items()}}, allowed_inf=ALLOWED_INF)
    buffers = {name: torch.full_like(local[source], torch.nan) for name, source in zip(GRADIENTS, ("q", "k", "v"))}
    grads = ops.block_sparse_attention_backward(
        local["do"],
        local["q"],
        local["k"],
        local["v"],
        local["o"],
        local["lse"],
        inputs["idx"] if metadata is not None else local["idx"],
        local["width"],
        inputs["sizes"] if metadata is not None else local["sizes"],
        inputs["counts"] if metadata is not None else local["counts"],
        sparse_block_size=BLOCK,
        bucket_size_blocks=bucket,
        dq_tensor=buffers["dq"],
        dk_tensor=buffers["dk"],
        dv_tensor=buffers["dv"],
        metadata=metadata,
    )
    assert tuple(grads.keys()) == ("dq_tensor", "dk_tensor", "dv_tensor")
    result = dict(zip(GRADIENTS, grads))
    for name in GRADIENTS:
        assert result[name].data_ptr() == buffers[name].data_ptr(), f"{name}: the caller's buffer was not used"
    assert torch.count_nonzero(result["dq"][empty_rows(local)]).item() == 0, "dq: an empty query block produced a nonzero gradient"
    assert_bitwise({**tensors_of(local), **{f"caller {name}": value for name, value in metadata_tensors.items()}}, before, "kernel inputs changed")
    return snapshot(result)


def differing_bytes(actual, expected):
    return {name: torch.count_nonzero(bitwise_bits(actual[name].reshape(-1)) != bitwise_bits(expected[name].reshape(-1))).item() for name in expected}


def measure_replays(run, stable=(), repeats=8):
    """Replay with fresh buffers and queued side-stream work (as ``assert_replays``): the outputs named in ``stable``
    are asserted bitwise, and the differing byte counts of every output per replay are returned (the measurement of
    the reduce-add routes)."""
    expected = snapshot(run())
    stream = torch.cuda.Stream()
    noise = torch.ones(1024 * 1024, device="cuda")
    differing = {name: [] for name in expected}
    for repeat in range(1, repeats):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(32):
                noise.mul_(1.00001).add_(0.00001)
        try:
            actual = snapshot(run())
        finally:
            torch.cuda.current_stream().wait_stream(stream)
        assert actual.keys() == expected.keys()
        for name, count in differing_bytes(actual, expected).items():
            differing[name].append(count)
        assert_bitwise({name: actual[name] for name in stable}, {name: expected[name] for name in stable}, f"replay {repeat}")
    return expected, differing


def report(record_property, label, differing):
    """Record the measured byte differences: junit properties, the captured output and the run's ``measured`` summary
    (``frost_routing.measured``; a passing test's captured output never reaches the CI job log)."""
    record_property(label, {name: list(counts) for name, counts in differing.items()})
    summary = ", ".join(f"{name}: {counts}" for name, counts in differing.items())
    print(f"[bsa blk64 determinism] {label}: differing bytes {summary}")
    frost_routing.measured(f"bsa blk64 {label}", f"differing bytes {summary}")


def skip_by_contract(api, what):
    """Off SM90, skip an FP16 forward case or a prepared-route case as Unsupported by contract, but only after the public
    API has rejected a tiny instance of the configuration (``what`` is ``fp16`` or ``prepared``): a widened contract then
    fails the case instead of skipping it, so the support page cannot go stale unnoticed."""
    ops, _, arch = api
    if arch == 90:
        return
    expected, clause = CONTRACT[what]
    problem = golden_problem(torch.float16 if what == "fp16" else torch.bfloat16, "fixed")
    inputs = pack(problem, problem.samples)
    try:
        if what == "fp16":
            ops.block_sparse_attention_forward(inputs["q"], inputs["k"], inputs["v"], inputs["idx"], inputs["width"], inputs["sizes"], sparse_block_size=BLOCK)
        else:
            prepare(ops, inputs, None)
    except (NotImplementedError, AssertionError) as error:
        assert re.search(expected, str(error)), f"SM{arch} rejected the {what} call for another reason: {error}"
    else:
        pytest.fail(f"SM{arch} accepted the {what} call: the contract widened, so this case must run and the support page must change")
    pytest.skip(f"{clause} (Unsupported by contract; SM{arch} rejected the call)")


def route_setup(api, monkeypatch, route):
    """Skip prepared routes off SM90 (after the API rejected them), force the paired gate where the route needs it, and
    return the route's flags."""
    _, interface, _ = api
    prepared = route.startswith("prepared-")
    flags = ROUTES[route.removeprefix("prepared-")]
    if prepared:
        skip_by_contract(api, "prepared")
    if flags["paired"]:
        force_paired(monkeypatch, interface)
    return prepared, flags


class TestBlk64Forward:
    @pytest.mark.parametrize("case", list(FORWARD_CASES), ids=[f"fprop-{case}" for case in FORWARD_CASES])
    def test_replay(self, api, case):
        ops, _, _ = api
        dtype, _, _, _, kv_splits = FORWARD_CASES[case]
        if dtype == torch.float16:
            skip_by_contract(api, "fp16")
        problem = forward_problem(case)
        inputs = pack(problem, problem.samples)
        actual = assert_replays(lambda: run_forward(ops, inputs, kv_splits=kv_splits), allowed_inf=ALLOWED_INF)
        check_forward_reference(inputs, actual)

    @pytest.mark.parametrize("case", list(FORWARD_CASES), ids=[f"fprop-{case}" for case in FORWARD_CASES])
    def test_batch_invariance(self, api, case):
        ops, _, _ = api
        dtype, _, _, _, kv_splits = FORWARD_CASES[case]
        if dtype == torch.float16:
            skip_by_contract(api, "fp16")
        problem = forward_problem(case)
        target, samples = 0, problem.samples
        expected = run_forward(ops, pack(problem, [samples[target]]), kv_splits=kv_splits)
        count = len(samples)
        for order in (tuple(range(count)), tuple(reversed(range(count))), (target, target), (1, target, 1)):
            actual = run_forward(ops, pack(problem, [samples[index] for index in order]), kv_splits=kv_splits)
            for position, index in enumerate(order):
                if index == target:
                    assert_bitwise(piece(actual, position), expected, f"batch {order}, position {position}")
        perm = torch.arange(expected["o"].shape[1] - 1, -1, -1, device="cuda")
        actual = run_forward(ops, permute_heads(pack(problem, [samples[target]]), perm), kv_splits=kv_splits)
        assert_bitwise(actual, {name: value[:, perm] for name, value in expected.items()}, "heads reordered")
        # Other neighbors: new values and new masks around the same target data, mask and (shared) block sizes.
        neighbors = list(forward_problem(case, seed=20261112).samples[1:])
        for position, selected in ((0, [samples[target], *neighbors]), (count - 1, [*neighbors, samples[target]])):
            actual = run_forward(ops, pack(problem, selected), kv_splits=kv_splits)
            assert_bitwise(piece(actual, position), expected, f"changed neighbors, target at position {position}")

    @pytest.mark.parametrize("name", list(DTYPES), ids=[f"fprop-{name}" for name in DTYPES])
    def test_golden(self, api, name):
        ops, _, _ = api
        if DTYPES[name] == torch.float16:
            skip_by_contract(api, "fp16")
        problem = golden_problem(DTYPES[name], "variable")
        inputs = pack(problem, problem.samples)
        actual = assert_replays(lambda: run_forward(ops, inputs), repeats=3, allowed_inf=ALLOWED_INF)
        check_forward_reference(inputs, actual)
        recorded = {**tensors_of(inputs), "width": torch.tensor([inputs["width"]], dtype=torch.int32)}
        assert_golden(f"bsa-blk64-fprop-{name}", recorded, actual, allowed_inf=ALLOWED_INF)


class TestBlk64Backward:
    @pytest.mark.parametrize("route", [*ROUTES, *(f"prepared-{route}" for route in PREPARED_ROUTES)], ids=lambda route: f"bprop-bf16-{route}")
    def test_replay(self, record_property, api, monkeypatch, route):
        """Eight executions with fresh buffers and side-stream work; the gradients' bytes are measured (see the
        module docstring) and asserted only where ``REPLAY_STABLE`` says so; the reference, the route and the
        empty-row dQ are asserted."""
        ops, interface, _ = api
        prepared, flags = route_setup(api, monkeypatch, route)
        probe = RouteProbe(monkeypatch, interface)
        problem = backward_problem()
        inputs = pack(problem, problem.samples)
        forward = run_forward(ops, inputs)
        metadata = prepare(ops, inputs, flags["bucket"]) if prepared else None
        expected, differing = measure_replays(
            lambda: run_backward(ops, inputs, forward, bucket=flags["bucket"], metadata=metadata), stable=REPLAY_STABLE.get(route, ())
        )
        probe.expect(paired=flags["paired"], direct=flags["direct"], prepared=metadata)
        report(record_property, f"{route} replay", differing)
        check_backward_reference(inputs, expected)

    @pytest.mark.parametrize("route", ["regular", "paired"], ids=lambda route: f"bprop-bf16-{route}")
    def test_batch_invariance(self, record_property, api, monkeypatch, route):
        """A batch element alone against its slice of the packed, reversed, duplicated and changed-neighbor batches:
        the forward results are asserted bitwise, the gradients' bytes are measured, the route is asserted."""
        ops, interface, _ = api
        _, flags = route_setup(api, monkeypatch, route)
        probe = RouteProbe(monkeypatch, interface)
        problem = backward_problem()
        target, samples = 0, problem.samples
        alone = pack(problem, [samples[target]])
        forward = run_forward(ops, alone)
        expected = run_backward(ops, alone, forward, bucket=flags["bucket"])
        probe.expect(paired=flags["paired"], direct=flags["direct"])
        changed = backward_problem(seed=20261113, sizes=problem.sizes)
        batches = {
            "packed": [samples[0], samples[1], samples[2]],
            "reversed": [samples[2], samples[1], samples[0]],
            "duplicated": [samples[target], samples[target]],
            "changed neighbors": [changed.samples[1], samples[target], changed.samples[2]],
        }
        differing = {name: [] for name in GRADIENTS}
        for label, selected in batches.items():
            inputs = pack(problem, selected)
            packed_forward = run_forward(ops, inputs)
            actual = run_backward(ops, inputs, packed_forward, bucket=flags["bucket"])
            probe.expect(paired=flags["paired"], direct=flags["direct"])
            for position, sample in enumerate(selected):
                if sample is samples[target]:
                    assert_bitwise(piece(packed_forward, position), forward, f"{label} batch, position {position} forward")
                    for name, count in differing_bytes(piece(actual, position), expected).items():
                        differing[name].append(count)
        report(record_property, f"{route} batch invariance", differing)

    @pytest.mark.parametrize("route", PREPARED_ROUTES, ids=lambda route: f"bprop-bf16-prepared-{route}")
    def test_prepared_parity(self, api, monkeypatch, route):
        """The prepared route runs the default route's kernels over metadata built once: same values within the FP32
        reduce-add order noise, and the object is reusable."""
        ops, interface, _ = api
        _, flags = route_setup(api, monkeypatch, f"prepared-{route}")
        probe = RouteProbe(monkeypatch, interface)
        problem = backward_problem()
        inputs = pack(problem, problem.samples)
        forward = run_forward(ops, inputs)
        default = run_backward(ops, inputs, forward, bucket=flags["bucket"])
        probe.expect(paired=flags["paired"], direct=flags["direct"])
        metadata = prepare(ops, inputs, flags["bucket"])
        for _ in range(2):
            actual = run_backward(ops, inputs, forward, bucket=flags["bucket"], metadata=metadata)
            probe.expect(paired=flags["paired"], direct=flags["direct"], prepared=metadata)
            for name in GRADIENTS:
                torch.testing.assert_close(actual[name].float(), default[name].float(), atol=1e-2, rtol=1e-2)
        check_backward_reference(inputs, actual)

    def test_zero_upstream_gradient(self, api, monkeypatch):
        ops, interface, _ = api
        probe = RouteProbe(monkeypatch, interface)
        problem = backward_problem()
        inputs = pack(problem, problem.samples)
        inputs["do"] = torch.zeros_like(inputs["do"])
        forward = run_forward(ops, inputs)
        actual = assert_replays(lambda: run_backward(ops, inputs, forward))
        probe.expect(paired=False, direct=True)
        for name in GRADIENTS:
            assert torch.count_nonzero(actual[name]).item() == 0, f"{name}: zero upstream gradient produced a nonzero result"

    @pytest.mark.parametrize("route", ["regular", "paired"], ids=lambda route: f"bprop-bf16-{route}")
    def test_golden(self, api, monkeypatch, route):
        """Order-free patterns (see ``golden_problem``): three executions must match bitwise before the hashes are
        recorded; on SM90 the prepared route must give the same bytes."""
        ops, interface, arch = api
        _, flags = route_setup(api, monkeypatch, route)
        probe = RouteProbe(monkeypatch, interface)
        problem = golden_problem(torch.bfloat16, "variable" if flags["paired"] else "fixed")
        inputs = pack(problem, problem.samples)
        forward = run_forward(ops, inputs)
        actual = assert_replays(lambda: run_backward(ops, inputs, forward, bucket=flags["bucket"]), repeats=3)
        probe.expect(paired=flags["paired"], direct=flags["direct"])
        check_backward_reference(inputs, actual)
        if arch == 90:
            metadata = prepare(ops, inputs, flags["bucket"])
            assert_bitwise(run_backward(ops, inputs, forward, bucket=flags["bucket"], metadata=metadata), actual, "prepared route")
            probe.expect(paired=flags["paired"], direct=flags["direct"], prepared=metadata)
        recorded = {**tensors_of(inputs), "width": torch.tensor([inputs["width"]], dtype=torch.int32)}
        assert_golden(f"bsa-blk64-bprop-{route}", recorded, {**forward, **actual}, allowed_inf=ALLOWED_INF)
