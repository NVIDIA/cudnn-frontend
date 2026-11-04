# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 block sparse attention backward (with the forward it consumes) against the torch reference.

Tests without an architecture prefix run on every GPU the backward supports (SM90, SM100/SM103/SM110) at both sparse
block sizes; ``test_sm<arch>_blk<size>_*`` tests cover one kernel and skip on other GPUs.  The metadata kernels the
backward builds (K-to-Q CSR, KV pairing) are tested in ``test_BSA_backward_metadata.py``, argument checks in
``test_BSA_api.py``.
"""

import importlib

import pytest
import torch

from test_utils import torch_fork_set_rng
from block_sparse_attention.cutedsl.bsa_reference import attention_reference, block_sparse_mask
from block_sparse_attention.cutedsl.bsa_utils import (
    blk128_case,
    blk128_reference,
    check_blk128,
    cuda_graph,
    import_bsa,
    make_fixed_metadata,
    no_host_sync,
    require_arch,
)

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

BACKWARD_FAMILIES = (9, 10, 11)
SM90_BLK128 = "the native SM90 blk128 kernels are under test"


# ---- every architecture


@pytest.mark.L0
@pytest.mark.parametrize("block_size", (64, 128), ids=("blk64", "blk128"))
@torch_fork_set_rng(seed=2)
def test_fixed_blocks(block_size):
    """BHSD with allocated gradients, then BSHD into caller-provided gradient tensors."""
    ops = import_bsa(families=BACKWARD_FAMILIES)
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 2 * block_size, 4 * block_size, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    _, _, *grads_ref = attention_reference(q, k, v, mask, do=do)

    forward = ops.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num, block_sizes, sparse_block_size=block_size)
    backward = ops.block_sparse_attention_backward(
        do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k, block_sparse_num, None, sparse_block_size=block_size
    )
    for name, expected in zip(("dq_tensor", "dk_tensor", "dv_tensor"), grads_ref):
        torch.testing.assert_close(backward[name].float(), expected, atol=3e-2, rtol=3e-2)

    q_bshd, k_bshd, v_bshd, do_bshd = (tensor.transpose(1, 2) for tensor in (q, k, v, do))
    forward_bshd = ops.block_sparse_attention_forward(q_bshd, k_bshd, v_bshd, q2k, block_sparse_num, block_sizes, sparse_block_size=block_size, layout="bshd")
    outputs = dict(dq_tensor=torch.empty_like(q_bshd), dk_tensor=torch.empty_like(k_bshd), dv_tensor=torch.empty_like(v_bshd))
    backward_bshd = ops.block_sparse_attention_backward(
        do_bshd,
        q_bshd,
        k_bshd,
        v_bshd,
        forward_bshd["o_tensor"],
        forward_bshd["lse_tensor"],
        q2k,
        block_sparse_num,
        None,
        **outputs,
        sparse_block_size=block_size,
        layout="bshd",
    )
    for (name, given), expected in zip(outputs.items(), grads_ref):
        assert backward_bshd[name].data_ptr() == given.data_ptr()
        torch.testing.assert_close(backward_bshd[name].transpose(1, 2).float(), expected, atol=3e-2, rtol=3e-2)


# ---- blk64 (SM90 and SM100/SM110)


def _force_paired_kv(monkeypatch):
    """Route eligible variable-count calls through the paired-KV kernel regardless of the size gate.

    Returns a list that records each paired kernel the dispatcher builds, so a test can prove the route
    without reading compile-cache key positions.
    """
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface, "_BLK64_PAIRED_KV_MIN_KV_TASKS", 0)
    builds = []
    name = "MixedKvBackwardSm90" if torch.cuda.get_device_capability()[0] == 9 else "MixedKvBackwardSm100"
    paired_cls = getattr(interface, name)

    class RecordingMixedKvBackward(paired_cls):
        def __init__(self, *args, **kwargs):
            builds.append(args)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(interface, name, RecordingMixedKvBackward)
    return builds


@pytest.mark.L0
@torch_fork_set_rng(seed=5)
@pytest.mark.parametrize(
    ("bucket_size_blocks", "seqlen_k", "softmax_scale"),
    [
        # Two Q blocks fit in the default 1088-block bucket: direct dK/dV.
        (None, 4 * 64, 0.17),
        # One Q block per bucket forces two groups: FP32 atomic fallback.
        (1, 4 * 64, 0.17),
        # A physical K tail keeps the existing predicated fallback.
        (None, 4 * 64 - 13, None),
    ],
)
@pytest.mark.parametrize("variable_counts", [False, True])
def test_blk64_buckets_and_dkv_workspace(
    bucket_size_blocks,
    seqlen_k,
    softmax_scale,
    variable_counts,
    monkeypatch,
):
    """Direct dK/dV within one Q bucket; the FP32 dK/dV workspace for several buckets, paired-KV tasks and a KV tail."""
    ops = import_bsa(families=BACKWARD_FAMILIES)
    major, _ = torch.cuda.get_device_capability()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    if variable_counts:
        # Keep this small case on the paired-KV path, whose shared dK/dV needs the FP32 workspace.
        monkeypatch.setattr(interface, "_BLK64_PAIRED_KV_MIN_KV_TASKS", 0)
    workspace_modes = []
    allocate_workspace = interface._empty_bwd_workspace_with_zeroed_accum

    def track_workspace_mode(**kwargs):
        workspace_modes.append(kwargs.get("include_dkv_accum", True))
        return allocate_workspace(**kwargs)

    monkeypatch.setattr(
        interface,
        "_empty_bwd_workspace_with_zeroed_accum",
        track_workspace_mode,
    )
    block_size = 64
    batch, heads, seqlen_q, dim = 1, 2, 2 * block_size, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)
    counts = torch.full(q2k.shape[:-1], block_sparse_num, dtype=torch.int32, device=q.device) if variable_counts else None
    # Metadata selects KV blocks 0 and 2, leaving 1 and 3 empty.
    block_sizes[2] = block_size // 2
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    _, _, dq_ref, dk_ref, dv_ref = attention_reference(q, k, v, mask, softmax_scale, do=do)

    dq = torch.full_like(q, torch.nan)
    dk = torch.full_like(k, torch.nan)
    dv = torch.full_like(v, torch.nan)

    forward = ops.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=64,
        softmax_scale=softmax_scale,
        **({} if major == 9 else {"use_clc": False}),
        q2k_block_nums=counts,
    )
    backward = ops.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        block_sparse_num,
        block_sizes,
        dq_tensor=dq,
        dk_tensor=dk,
        dv_tensor=dv,
        bucket_size_blocks=bucket_size_blocks,
        sparse_block_size=64,
        softmax_scale=softmax_scale,
        q2k_block_nums=counts,
    )
    assert backward["dq_tensor"].data_ptr() == dq.data_ptr()
    assert backward["dk_tensor"].data_ptr() == dk.data_ptr()
    assert backward["dv_tensor"].data_ptr() == dv.data_ptr()
    # Paired and residual tasks share FP32 accumulation even with a single Q group.
    expect_dkv_workspace = variable_counts or bucket_size_blocks == 1 or seqlen_k % block_size != 0
    assert workspace_modes == [expect_dkv_workspace]
    torch.testing.assert_close(backward["dq_tensor"].float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dk_tensor"].float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dv_tensor"].float(), dv_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.L1
@pytest.mark.parametrize("clc,splits", [(False, 1), (False, 2), (True, 1), (True, 2)])
@pytest.mark.parametrize("variable", [False, True])
def test_blk64_signed_scale_and_compile_cache(monkeypatch, clc, splits, variable):
    """Positive, zero and negative softmax scales with partial KV blocks and empty rows, forward and backward."""
    bsa = import_bsa(archs=(90, 100, 103), reason="requires SM90 or SM100/SM103")
    sm90 = torch.cuda.get_device_capability()[0] == 9
    if sm90 and clc:
        pytest.skip("CLC scheduling is SM100-only")
    if variable:
        _force_paired_kv(monkeypatch)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface.bsa_attn_fwd_blk64_cutedsl, "compile_cache", {})
    monkeypatch.setattr(interface.bsa_attn_fwd, "compile_cache", {})
    monkeypatch.setattr(interface._bsa_attn_bwd_bucketed_k2q_csr, "compile_cache", {})
    torch.manual_seed(27191)
    # The SM90 forward needs whole query blocks; the backward partial-query tail is covered elsewhere.
    seqlen_q = 256 if sm90 else 193
    q = torch.randn((1, 2, seqlen_q, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 129, 128), device="cuda", dtype=torch.bfloat16)
    v, do = torch.randn_like(k), torch.randn_like(q)
    indices = torch.arange(3, device="cuda", dtype=torch.int32).view(1, 1, 1, 3).expand(1, 2, 4, 3).contiguous()
    counts = torch.tensor([3, 0, 1, 2], device="cuda", dtype=torch.int32).view(1, 1, 4).expand(1, 2, 4).contiguous() if variable else None
    sizes = torch.tensor([64, 37, 1], device="cuda", dtype=torch.int32)
    mask = block_sparse_mask(indices, 3, sizes, seqlen_q, 129, 64, counts)
    kwargs = dict(block_sparse_num=3, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=64)
    cache_sizes = []
    # The forward compiles one extra, separately specialized nonpositive-scale kernel so the positive-scale
    # softmax loop stays free of the masking-order branch. Changing the sign never recompiles the SM100
    # backward; SM90 specializes its backward the same way as its forward.
    clc_kwargs = {} if sm90 else dict(use_clc=clc)
    for scale in (128**-0.5, 0.0, -(128**-0.5), 128**-0.5):
        expected, lse_ref, *grads_ref = attention_reference(q, k, v, mask, scale, do=do, dtype=torch.float64)
        output, lse = bsa.block_sparse_attention_forward(
            q, k, v, indices, **kwargs, softmax_scale=scale, **clc_kwargs, kv_splits=splits, allow_empty_block_nums=True
        )
        gradients = bsa.block_sparse_attention_backward(do, q, k, v, output, lse, indices, **kwargs, softmax_scale=scale, bucket_size_blocks=3)
        torch.testing.assert_close(lse.double(), lse_ref, atol=0.03, rtol=0.03)
        for got, ref in zip((output, *gradients), (expected, *grads_ref)):
            torch.testing.assert_close(got.double(), ref, atol=0.03, rtol=0.03)
        forward_cache = interface.bsa_attn_fwd if sm90 else interface.bsa_attn_fwd_blk64_cutedsl
        cache_sizes.append((len(forward_cache.compile_cache), len(interface._bsa_attn_bwd_bucketed_k2q_csr.compile_cache)))
    forward_sizes, backward_sizes = zip(*cache_sizes)
    specialized = (forward_sizes[0], forward_sizes[0] + 1, forward_sizes[0] + 1, forward_sizes[0] + 1)
    assert forward_sizes == specialized
    if sm90:
        assert backward_sizes == (backward_sizes[0], backward_sizes[0] + 1, backward_sizes[0] + 1, backward_sizes[0] + 1)
    else:
        assert set(backward_sizes) == {backward_sizes[0]}


@pytest.mark.L0
@pytest.mark.parametrize("duplicates", [False, True])
@pytest.mark.parametrize("kv_blocks", [17, 35])
def test_blk64_paired_kv_duplicates_and_graph_replay(monkeypatch, duplicates, kv_blocks):
    """Duplicate slots count once per occurrence; the paired-KV path does not sync the host and replays in a graph."""
    bsa = import_bsa(archs=(90, 100, 103), reason="requires SM90 or SM100/SM103")
    paired_builds = _force_paired_kv(monkeypatch)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface._bsa_attn_bwd_bucketed_k2q_csr, "compile_cache", {})
    torch.manual_seed(27123)
    q = torch.randn((1, 2, 320, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, kv_blocks * 64, 128), device="cuda", dtype=torch.bfloat16)
    v, do = torch.randn_like(k), torch.randn_like(q)
    indices = torch.tensor([0, 2, 8, kv_blocks - 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 4).expand(1, 2, 5, 4).contiguous()
    counts = torch.full((1, 2, 5), 4, device="cuda", dtype=torch.int32)
    if duplicates:
        indices[:, :, 1, 1] = indices[:, :, 1, 0]
    kwargs = dict(q2k_block_nums=counts, sparse_block_size=64)

    def step():
        output, lse = bsa.block_sparse_attention_forward(q, k, v, indices, **kwargs)
        gradients = bsa.block_sparse_attention_backward(do, q, k, v, output, lse, indices, **kwargs, bucket_size_blocks=3)
        return output, *gradients

    eager = step()
    # Duplicate slots contribute repeatedly; a boolean mask would hide a lost edge.
    multiplicity = torch.zeros((1, 2, 320, kv_blocks * 64), device="cuda", dtype=torch.float32)
    for h in range(2):
        for row in range(5):
            for block in indices[0, h, row].tolist():
                multiplicity[0, h, row * 64 : (row + 1) * 64, block * 64 : (block + 1) * 64] += 1
    ref_output, _, *grads_ref = attention_reference(q, k, v, multiplicity.log(), do=do)
    for got, expected in zip(eager, (ref_output, *grads_ref)):
        torch.testing.assert_close(got.float(), expected, atol=0.03, rtol=0.03)
    assert paired_builds, "the dispatcher did not build the paired-KV kernel"
    torch.cuda.synchronize()
    with no_host_sync():
        step()
    with cuda_graph() as graph:
        with torch.cuda.graph(graph):
            captured = step()
        q.mul_(0.8)
        counts.fill_(2)
        graph.replay()
        for got, expected in zip(captured, step()):
            torch.testing.assert_close(got, expected, atol=0.03, rtol=0.03)


@pytest.mark.L1
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("sq,sk", [(1024, 1025), (1088, 2177), (2048, 2048)])
def test_blk64_paired_kv_partial_blocks_and_buckets(layout, sq, sk, monkeypatch):
    """Local KV windows give matched pairs real common edges; partial blocks, a KV tail and empty rows mask both halves."""
    bsa = import_bsa(archs=(90, 100, 103), reason="requires SM90 or SM100/SM103")
    paired_builds = _force_paired_kv(monkeypatch)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface._bsa_attn_bwd_bucketed_k2q_csr, "compile_cache", {})
    torch.manual_seed(28000 + sk)
    q = torch.randn((2, 2, sq, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 2, sk, 128), device="cuda", dtype=torch.bfloat16)
    v, do = torch.randn_like(k), torch.randn_like(q)
    nq, nk = sq // 64, (sk + 63) // 64
    width = 6
    centers = (torch.arange(nq, device="cuda") * nk // nq).clamp(max=nk - width)
    window = centers[:, None] + torch.arange(width, device="cuda")
    indices = window.expand(2, 2, nq, width).int().contiguous()
    counts = torch.randint(width - 2, width + 1, (2, 2, nq), device="cuda", dtype=torch.int32)
    counts[:, :, ::7] = 0
    sizes = torch.randint(1, 65, (nk,), device="cuda", dtype=torch.int32)
    sizes[::3] = 64
    sizes[-1].clamp_(max=sk - (nk - 1) * 64)
    mask = block_sparse_mask(indices, width, sizes, sq, sk, 64, counts)
    _, _, *grads_ref = attention_reference(q, k, v, mask, do=do)
    tensors = (q, k, v, do)
    if layout == "bshd":
        tensors = tuple(t.transpose(1, 2).contiguous() for t in tensors)
    q, k, v, do = tensors
    kwargs = dict(block_sparse_num=width, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=64, layout=layout)
    output, lse = bsa.block_sparse_attention_forward(q, k, v, indices, **kwargs, allow_empty_block_nums=True)
    for bucket_size in (3, nq):
        gradients = bsa.block_sparse_attention_backward(do, q, k, v, output, lse, indices, **kwargs, bucket_size_blocks=bucket_size)
        for got, ref in zip(gradients, grads_ref):
            got = got.transpose(1, 2) if layout == "bshd" else got
            torch.testing.assert_close(got.float(), ref, atol=0.03, rtol=0.03)
    assert paired_builds, "the dispatcher did not build the paired-KV kernel"


@pytest.mark.L1
def test_blk64_paired_kv_cache_rebinds_bitset_lengths(monkeypatch):
    """One compiled paired-KV kernel serves query lengths on both sides of the bitset word boundaries (exact oracle)."""
    bsa = import_bsa(archs=(90, 100, 103), reason="requires SM90 or SM100/SM103")
    paired_builds = _force_paired_kv(monkeypatch)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface._bsa_attn_bwd_bucketed_k2q_csr, "compile_cache", {})
    cache = interface._bsa_attn_bwd_bucketed_k2q_csr.compile_cache
    for sq in (8192, 8256, 262144, 262208, 8192):
        q = torch.zeros((1, 1, sq, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.zeros((1, 1, 128, 128), device="cuda", dtype=q.dtype)
        v, do = torch.full_like(k, 0.25), torch.ones_like(q)
        indices = torch.arange(2, device="cuda", dtype=torch.int32).view(1, 1, 1, 2).expand(1, 1, sq // 64, 2).contiguous()
        counts = torch.full((1, 1, sq // 64), 2, device="cuda", dtype=torch.int32)
        kwargs = dict(q2k_block_nums=counts, sparse_block_size=64)
        output, lse = bsa.block_sparse_attention_forward(q, k, v, indices, **kwargs)
        gradients = bsa.block_sparse_attention_backward(do, q, k, v, output, lse, indices, **kwargs, bucket_size_blocks=256)
        torch.testing.assert_close(output, torch.full_like(output, 0.25), atol=0, rtol=0)
        for i, gradient in enumerate(gradients):
            expected = torch.full_like(gradient, sq / 128 if i == 2 else 0)
            torch.testing.assert_close(gradient, expected, atol=0, rtol=0)
        assert len(cache) == 1
    assert len(paired_builds) == 1


# ---- SM90 blk64


def _one_edge_reference(q, k, v, do, q2k, block_sizes):
    """FP32 reference for problems where every Q block selects exactly one KV block, batched over Q blocks.

    Each Q block has exactly one selected KV block. Its softmax is independent
    of the other Q blocks, so dK/dV contributions can be summed by KV index.
    This avoids a large dense mask in the 385-Q-block bucket-boundary case.
    All tensors passed here use BHSD layout.
    """
    batch, heads, seqlen_q, dim = q.shape
    num_q_blocks, num_kv_blocks = seqlen_q // 64, k.shape[2] // 64
    selected = q2k[..., 0].long()
    kv_index = (selected.reshape(batch * heads, num_q_blocks) + torch.arange(batch * heads, device=q.device)[:, None] * num_kv_blocks).flatten()
    q_blocks = q.reshape(-1, 1, 64, dim)
    do_blocks = do.reshape_as(q_blocks)
    k_blocks = k.reshape(-1, 64, dim).index_select(0, kv_index).unsqueeze(1)
    v_blocks = v.reshape(-1, 64, dim).index_select(0, kv_index).unsqueeze(1)
    valid_sizes = block_sizes[selected].reshape(-1, 1, 1, 1)
    valid = torch.arange(64, device=q.device).reshape(1, 1, 1, 64) < valid_sizes
    mask = torch.zeros(valid.shape, dtype=torch.float32, device=q.device).masked_fill(~valid, float("-inf"))
    o, lse, dq, dk_per_edge, dv_per_edge = attention_reference(q_blocks, k_blocks, v_blocks, mask, do=do_blocks)
    dk = torch.zeros((batch * heads * num_kv_blocks, 64, dim), dtype=torch.float32, device=q.device)
    dv = torch.zeros_like(dk)
    dk.index_add_(0, kv_index, dk_per_edge.squeeze(1))
    dv.index_add_(0, kv_index, dv_per_edge.squeeze(1))
    return o.reshape_as(q), lse.reshape(batch, heads, seqlen_q), dq.reshape_as(q), dk.reshape_as(k), dv.reshape_as(v)


@pytest.mark.L0
@pytest.mark.parametrize(
    "batch,heads,kv_for_q,valid_sizes",
    [
        pytest.param(1, 3, (0,), (64,), id="single_incoming"),
        pytest.param(1, 3, (0, 0), (64,), id="two_incoming"),
        pytest.param(2, 2, (0, 0, 1), (17, 40, 64), id="multibatch_partial_blocks"),
        # Q0 and Q384 share KV0, with one incoming edge in each real 384-Q
        # bucket. Other KV blocks have one edge; no high-fan-in accumulation.
        pytest.param(1, 1, (*range(384), 0), (64,) * 384, id="bucket384_tail"),
    ],
)
def test_sm90_blk64_shared_v_lifetime(batch, heads, kv_for_q, valid_sizes):
    """One incoming edge per Q block: the two MMA warpgroups share each V tile in shared memory, and KV blocks
    without an edge (or the masked tail of a partial block) get exactly zero dK/dV."""
    sm90_bsa = import_bsa(families=(9,), reason="the SM90 blk64 backward is under test")
    num_q_blocks, num_kv_blocks = len(kv_for_q), len(valid_sizes)
    # Physical Q/KV lengths stay padded; block_sizes masks KV only.
    generator = torch.Generator(device="cpu").manual_seed(20261003)
    tensors = []
    for blocks in (num_q_blocks, num_kv_blocks, num_kv_blocks, num_q_blocks):
        tensor = torch.randn((batch, heads, blocks * 64, 128), generator=generator, dtype=torch.bfloat16)
        assert torch.isfinite(tensor).all()
        tensors.append(tensor.transpose(1, 2).contiguous().cuda())
    q, k, v, do = tensors
    q2k = torch.tensor(kv_for_q, device="cuda", dtype=torch.int32).view(1, 1, num_q_blocks, 1).expand(batch, heads, -1, -1).contiguous()
    counts = torch.ones((batch, heads, num_q_blocks), device="cuda", dtype=torch.int32)
    block_sizes = torch.tensor(valid_sizes, device="cuda", dtype=torch.int32)
    reference = _one_edge_reference(*(tensor.transpose(1, 2) for tensor in tensors), q2k, block_sizes)

    forward = sm90_bsa.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num=1,
        block_sizes=block_sizes,
        q2k_block_nums=counts,
        sparse_block_size=64,
        layout="bshd",
    )
    backward = sm90_bsa.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        block_sparse_num=1,
        block_sizes=block_sizes,
        q2k_block_nums=counts,
        bucket_size_blocks=384,
        sparse_block_size=64,
        layout="bshd",
    )
    actual = (
        forward["o_tensor"].transpose(1, 2),
        forward["lse_tensor"],
        backward["dq_tensor"].transpose(1, 2),
        backward["dk_tensor"].transpose(1, 2),
        backward["dv_tensor"].transpose(1, 2),
    )
    for name, result, expected in zip(("O", "LSE", "dQ", "dK", "dV"), actual, reference):
        assert torch.isfinite(result).all(), f"{name} contains nonfinite values"
        tolerance = 1e-5 if name == "LSE" else 3e-2
        torch.testing.assert_close(result.float(), expected, atol=tolerance, rtol=tolerance, msg=lambda message: f"{name}: {message}")
    # A tolerance comparison alone could hide small stale values in KV blocks
    # with no incoming edges, or in the masked tail of a partial block.
    for kv_block, valid_size in enumerate(valid_sizes):
        start = kv_block * 64 + (valid_size if kv_block in kv_for_q else 0)
        end = (kv_block + 1) * 64
        for gradient in actual[3:]:
            assert torch.count_nonzero(gradient[:, :, start:end]).item() == 0


# ---- SM90 blk128


@pytest.mark.L0
@pytest.mark.parametrize("layout,variable,sq,sk", [("bhsd", False, 256, 384), ("bshd", True, 129, 257)])
def test_sm90_blk128_forward_backward(layout, variable, sq, sk):
    require_arch(families=(9,), reason=SM90_BLK128)
    check_blk128(layout=layout, variable=variable, sq=sq, sk=sk, sizes_rank=2 if variable else 0)


@pytest.mark.L1
@pytest.mark.parametrize("sq,sk", [(1, 1), (63, 127), (64, 129), (127, 257), (129, 513), (257, 385)])
@pytest.mark.parametrize("variable", [False, True])
def test_sm90_blk128_tails(sq, sk, variable):
    require_arch(families=(9,), reason=SM90_BLK128)
    check_blk128(sq=sq, sk=sk, variable=variable, sizes_rank=1, bucket=2)


@pytest.mark.L1
@pytest.mark.parametrize("size", [1, 63, 64, 65, 127, 128])
@pytest.mark.parametrize(
    "scale, bucket",
    # The Q bucket only regroups the backward, independently of the KV boundary and the scale sign.
    [(0.0, 1), (-0.125, 1), (0.125, 1), (0.125, 192)],
    ids=["zero-bucket1", "negative-bucket1", "positive-bucket1", "positive-bucket192"],
)
def test_sm90_blk128_partial_boundary_scale(size, scale, bucket):
    """Every partial-KV boundary of the 64-row MMA halves, with zero, negative and positive softmax scales."""
    require_arch(families=(9,), reason=SM90_BLK128)
    check_blk128(sq=129, sk=257, sizes_rank=2, valid_size=size, scale=scale, splits=2, layout="bshd", bucket=bucket)


@pytest.mark.L1
@pytest.mark.parametrize("variable,bucket", [(False, 1), (False, 192), (True, 3)])
def test_sm90_blk128_pipeline_ring(variable, bucket):
    require_arch(families=(9,), reason=SM90_BLK128)
    check_blk128(sq=1025, sk=2049, active_count=16, variable=variable, sizes_rank=2, bucket=bucket)


@pytest.mark.L1
@pytest.mark.parametrize("sq", [65, 191, 193])
@pytest.mark.parametrize("bucket", [1, 256])
def test_sm90_blk128_q64_subtiles(sq, bucket):
    """One-token second Q64 halves and entirely absent second halves, without expanding the Q128 CSR; both dQ head
    partitions and KV buckets."""
    require_arch(families=(9,), reason=SM90_BLK128)
    check_blk128(sq=sq, sk=385, active_count=4, variable=True, sizes_rank=2, bucket=bucket, strided=True)


@pytest.mark.L1
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("bucket", [1, 192])
def test_sm90_blk128_strided_inputs_and_metadata(layout, bucket):
    require_arch(families=(9,), reason=SM90_BLK128)
    check_blk128(sq=129, sk=257, variable=True, sizes_rank=2, strided=True, layout=layout, bucket=bucket)


@pytest.mark.L0
def test_sm90_blk128_graph_mutable_metadata_no_recompile():
    """Graph replay with new counts and indices at the same addresses: no recompile, no host sync, empty rows reset."""
    ops = import_bsa(families=(9,), reason=SM90_BLK128)
    q, k, v, ids, counts, sizes, count = blk128_case(variable=True)
    from cudnn.block_sparse_attention import _interface as native

    do = torch.randn_like(q)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def run():
        f = ops.block_sparse_attention_forward(q, k, v, ids, count, q2k_block_nums=counts, allow_empty_block_nums=True, sparse_block_size=128)
        g = ops.block_sparse_attention_backward(do, q, k, v, f["o_tensor"], f["lse_tensor"], ids, count, q2k_block_nums=counts, sparse_block_size=128)
        return f, g

    with torch.cuda.stream(stream):
        run()
        run()
        sizes_before = len(native._bsa_attn_fwd_sm90_blk128.compile_cache), len(native._bsa_attn_bwd_sm90_blk128.compile_cache)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            f, g = run()
        # Same pointers now describe an entirely empty graph; all output gradients must be reset.
        counts.zero_()
        with no_host_sync():
            graph.replay()
            run()
    torch.cuda.current_stream().wait_stream(stream)
    assert sizes_before == (len(native._bsa_attn_fwd_sm90_blk128.compile_cache), len(native._bsa_attn_bwd_sm90_blk128.compile_cache))
    assert torch.count_nonzero(f["o_tensor"]) == 0
    assert torch.isneginf(f["lse_tensor"]).all()
    for t in g:
        assert torch.count_nonzero(t) == 0

    # Change both the graph and counts at the same addresses, then replay again.
    ids.copy_(torch.arange(count - 1, -1, -1, device=ids.device, dtype=ids.dtype))
    counts.fill_(count)
    graph.replay()
    ro, rl, rg = blk128_reference(q, k, v, ids, counts, None, count, 128**-0.5, "bhsd", do)
    torch.testing.assert_close(f["o_tensor"].double(), ro, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(f["lse_tensor"].double(), rl, atol=1e-4, rtol=1e-5)
    for got, expected in zip(g, rg):
        torch.testing.assert_close(got.double(), expected, atol=3e-2, rtol=3e-2)
    assert sizes_before == (len(native._bsa_attn_fwd_sm90_blk128.compile_cache), len(native._bsa_attn_bwd_sm90_blk128.compile_cache))
    graph.reset()


@pytest.mark.L1
@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("bucket", [1, 256])
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_sm90_blk128_dynamic_shape_compile_reuse(monkeypatch, variable, bucket, layout):
    require_arch(families=(9,), reason=SM90_BLK128)
    from cudnn.block_sparse_attention import _interface as native

    monkeypatch.setattr(native._bsa_attn_fwd_sm90_blk128, "compile_cache", {})
    monkeypatch.setattr(native._bsa_attn_bwd_sm90_blk128, "compile_cache", {})
    for b, h, sq, sk, count, strided in [(2, 4, 193, 385, 2, False), (3, 6, 321, 641, 3, False), (1, 2, 449, 769, 4, True), (2, 4, 577, 897, 5, True)]:
        check_blk128(b=b, h=h, sq=sq, sk=sk, active_count=count, variable=variable, bucket=bucket, strided=strided, layout=layout)
        # A strided metadata view has no unit stride and needs a distinct tensor
        # type; subsequent sizes within each layout must reuse the compiled code.
        assert len(native._bsa_attn_fwd_sm90_blk128.compile_cache) == (2 if strided else 1)
        assert len(native._bsa_attn_bwd_sm90_blk128.compile_cache) == 1


# ---- SM100/SM110 blk64


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
@pytest.mark.parametrize("seqlen", [33, 64])
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_sm100_blk64_sum_odo_fp32_product(seqlen, layout):
    """Casting after O*dO has already lost BF16 product bits needed by dS."""
    require_arch(families=(10, 11), reason="blk64 backward is specific to SM100/SM110")

    from cudnn.frost.buffers import cutedsl_requirement_error

    requirement_error = cutedsl_requirement_error("blk64 backward preprocessing test")
    if requirement_error is not None:
        pytest.skip(requirement_error)

    import_bsa()
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cudnn.block_sparse_attention.csrc.bwd.sm100_blk64.bsa_bwd_sm100 import BlockSparseAttnBackwardSm100Blk64
    from cudnn.block_sparse_attention.csrc.utils.cute_dsl_utils import to_cute_tensor

    @cute.jit
    def launch_sum_odo(op: cutlass.Constexpr, out, dout, delta, lse, scaled_lse, stream):
        n, d, h, b = out.shape
        op.sum_OdO(
            cute.group_modes(out, 2, 4),
            cute.group_modes(dout, 2, 4),
            cute.group_modes(delta, 1, 3),
            cute.group_modes(lse, 1, 3),
            cute.group_modes(scaled_lse, 1, 3),
            cutlass.Float32(-1.0),
            cutlass.Float32(-1.4426950408889634),
            (n, n, d, (h, b)),
        ).launch(
            grid=(cute.ceil_div(n, op.sum_OdO_block_q), h, b),
            block=(op.sum_OdO_num_threads_d, op.sum_OdO_num_threads_q, 1),
            stream=stream,
        )

    batch, heads, dim = 2, 3, 128
    out = torch.randn((batch, heads, seqlen, dim), device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(out)
    # 1.0625 is exactly BF16, but its square 1.12890625 is not.
    out[:, :, 0, :] = 1.0625
    dout[:, :, 0, :] = 1.0625
    if layout == "bshd":
        out, dout = (tensor.transpose(1, 2).contiguous().transpose(1, 2) for tensor in (out, dout))
    lse = torch.randn((batch, heads, seqlen), device="cuda", dtype=torch.float32)
    delta = torch.full_like(lse, torch.nan)
    scaled_lse = torch.full_like(lse, torch.nan)
    operands = [out.permute(2, 3, 1, 0), dout.permute(2, 3, 1, 0)]
    operands += [tensor.permute(2, 1, 0) for tensor in (delta, lse, scaled_lse)]
    compiled = cute.compile(
        launch_sum_odo,
        BlockSparseAttnBackwardSm100Blk64(64, False),
        *[to_cute_tensor(tensor, leading_dim=1 if i < 2 else 0) for i, tensor in enumerate(operands)],
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
    compiled(*operands, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    reference = -(out.double() * dout.double()).sum(dim=-1)
    torch.testing.assert_close(delta.double(), reference, atol=2e-5, rtol=1e-6)
    torch.testing.assert_close(scaled_lse, lse * -1.4426950408889634, atol=0, rtol=0)


@pytest.mark.L1
@pytest.mark.parametrize(
    "sq, sk, variable, layout",
    # The layout only changes the strides, independently of the tails and the count mode: it alternates over the
    # shape x count-mode grid, so that every shape and both count modes see both layouts.
    [
        (sq, sk, variable, "bshd" if (i + variable) % 2 else "bhsd")
        for i, (sq, sk) in enumerate([(193, 129), (248, 2177), (320, 1088), (513, 1025)])
        for variable in (False, True)
    ],
)
def test_sm100_blk64_partial_tiles_and_buckets(sq, sk, variable, layout, monkeypatch):
    """Random partial KV blocks, partial Q/KV tails and (paired-KV) empty rows over 1, 3 and all-Q buckets."""
    bsa = import_bsa(archs=(100, 103), reason="requires SM100/SM103")
    if variable:
        _force_paired_kv(monkeypatch)
    torch.manual_seed(27000 + sq)
    q = torch.randn((2, 3, sq, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 3, sk, 128), device="cuda", dtype=torch.bfloat16)
    v, do = torch.randn_like(k), torch.randn_like(q)
    nq, nk = (sq + 63) // 64, (sk + 63) // 64
    width = min(4, nk)
    indices = torch.rand((2, 3, nq, nk), device="cuda").argsort(-1)[..., :width].int().contiguous()
    counts = torch.randint(0, width + 1, (2, 3, nq), device="cuda", dtype=torch.int32) if variable else None
    sizes = torch.randint(1, 65, (nk,), device="cuda", dtype=torch.int32)
    sizes[-1].clamp_(max=sk - (nk - 1) * 64)
    mask = block_sparse_mask(indices, width, sizes, sq, sk, 64, counts)
    expected, _, *grads_ref = attention_reference(q, k, v, mask, do=do)
    tensors = (q, k, v, do)
    if layout == "bshd":
        tensors = tuple(t.transpose(1, 2).contiguous() for t in tensors)
    q, k, v, do = tensors
    kwargs = dict(block_sparse_num=width, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=64, layout=layout)
    output, lse = bsa.block_sparse_attention_forward(q, k, v, indices, **kwargs, allow_empty_block_nums=True)
    canonical_output = output.transpose(1, 2) if layout == "bshd" else output
    torch.testing.assert_close(canonical_output.float(), expected, atol=0.03, rtol=0.03)
    buffers = tuple(torch.empty_like(t) for t in (q, k, v))
    for bucket_size in (1, 3, nq + 1):
        for buffer in buffers:
            buffer.fill_(float("nan"))
        gradients = bsa.block_sparse_attention_backward(
            do,
            q,
            k,
            v,
            output,
            lse,
            indices,
            **kwargs,
            bucket_size_blocks=bucket_size,
            dq_tensor=buffers[0],
            dk_tensor=buffers[1],
            dv_tensor=buffers[2],
        )
        for got, buffer, ref in zip(gradients, buffers, grads_ref):
            assert got.data_ptr() == buffer.data_ptr()
            got = got.transpose(1, 2) if layout == "bshd" else got
            torch.testing.assert_close(got.float(), ref, atol=0.03, rtol=0.03)


# ---- SM100/SM110 blk128


@pytest.mark.L0
@torch_fork_set_rng(seed=8)
@pytest.mark.parametrize("num_q_blocks", [1, 4])
def test_sm100_blk128_dk_zero_init_accumulate_transition(num_q_blocks):
    """dK zero-init on the first Q block and accumulate on later Q blocks.

    num_q_blocks=1 exercises pure zero-initialization; num_q_blocks=4 makes
    several Q blocks write the same K blocks, exercising the runtime
    initialize-to-accumulate transition of the dK MMA predicate.
    """
    ops = import_bsa(families=(10, 11), reason="blk128 backward is specific to SM100/SM110")
    block_size = 128
    batch, heads, dim = 1, 1, 128
    seqlen_q = num_q_blocks * block_size
    seqlen_k = 4 * block_size
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)

    # Every Q block attends to the same two K blocks.
    num_kv_blocks = seqlen_k // block_size
    selected = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    q2k = selected.view(1, 1, 1, 2).expand(batch, heads, num_q_blocks, 2).contiguous()
    block_sparse_num = 2
    full_block_sizes = torch.full((num_kv_blocks,), block_size, dtype=torch.int32, device="cuda")
    mask = block_sparse_mask(q2k, block_sparse_num, full_block_sizes, seqlen_q, seqlen_k, block_size)
    _, _, dq_ref, dk_ref, dv_ref = attention_reference(q, k, v, mask, do=do)

    forward = ops.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num, None, sparse_block_size=block_size)
    backward = ops.block_sparse_attention_backward(
        do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k, block_sparse_num, None, sparse_block_size=block_size
    )
    torch.testing.assert_close(backward["dk_tensor"].float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dq_tensor"].float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dv_tensor"].float(), dv_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize(
    "sq, sk, dim, layout, bucket",
    # The Q bucket only regroups the backward, independently of the tails and head_dim: one shape takes both buckets.
    [(512, 512, 128, "bshd", 1), (389, 517, 128, "bhsd", 1), (389, 517, 128, "bhsd", 128), (257, 257, 64, "bshd", 1)],
)
def test_sm100_blk128_partial_blocks_and_buckets(sq, sk, dim, layout, bucket):
    """Partial KV blocks (strided block_sizes), empty rows, Q/KV tails and head_dim 64 over Q buckets of 1 and 128."""
    BSA = import_bsa(archs=(100, 103), reason="blk128 backward block_sizes are SM100/SM103 only")
    torch.manual_seed(13)
    q = torch.randn(2, sq, 2, dim, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(2, sk, 2, dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    do = torch.randn_like(q)
    nq, nk = (sq + 127) // 128, (sk + 127) // 128
    sizes = torch.tensor(([17, 128, 69, 1, 0])[:nk], device="cuda", dtype=torch.int32)
    sizes[-1] = min(1, sk - 128 * (nk - 1))
    sizes = torch.stack((sizes, torch.zeros_like(sizes)), dim=1)[:, 0]  # Strided metadata.
    ids = torch.arange(nk, device="cuda", dtype=torch.int32).view(1, 1, 1, nk).expand(2, 2, nq, nk).contiguous()
    counts = torch.tensor([1, nk, 0, nk][:nq], device="cuda", dtype=torch.int32).view(1, 1, nq).expand(2, 2, nq).contiguous()
    # A physical KV tail must be declared to forward even when other blocks are full.
    full_sizes = torch.full_like(sizes, 128)
    full_sizes[-1] = sk - 128 * (nk - 1)
    full_sizes = None if sk % 128 == 0 else full_sizes
    # None -> partial -> None on aligned inputs exercises separate specializations.
    for block_sizes in (full_sizes, sizes, full_sizes):
        mask = block_sparse_mask(ids, 0, block_sizes, sq, sk, 128, counts)
        ref, _, *ref_grads = attention_reference(*(t.transpose(1, 2) for t in (q, k, v)), mask, do=do.transpose(1, 2), dtype=torch.float64)
        ref, *ref_grads = (t.transpose(1, 2) for t in (ref, *ref_grads))
        inputs = [t.transpose(1, 2).contiguous() for t in (q, k, v, do)] if layout == "bhsd" else (q, k, v, do)
        qin, kin, vin, din = inputs
        out, lse = BSA.block_sparse_attention_forward(
            qin, kin, vin, ids, q2k_block_nums=counts, block_sizes=block_sizes, sparse_block_size=128, layout=layout, allow_empty_block_nums=True
        )
        grads = BSA.block_sparse_attention_backward(
            din, qin, kin, vin, out, lse, ids, q2k_block_nums=counts, block_sizes=block_sizes, sparse_block_size=128, layout=layout, bucket_size_blocks=bucket
        )
        for name, actual, expected in zip(("O", "dQ", "dK", "dV"), (out, *grads), (ref, *ref_grads)):
            actual = actual.transpose(1, 2) if layout == "bhsd" else actual
            torch.testing.assert_close(actual.double(), expected, atol=0.03, rtol=0.03, msg=lambda msg: name + ": " + msg)


@pytest.mark.L0
def test_sm100_blk128_kv_prefix_boundaries():
    """Partial KV prefixes must preserve all gradients across internal MMA boundaries."""
    BSA = import_bsa(archs=(100, 103), reason="blk128 backward block_sizes are SM100/SM103 only")
    torch.manual_seed(123)
    q, k, v, do = [torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 2).expand(1, 2, 2, 2).contiguous()
    for prefix in (0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 77, 78, 79, 80, 81, 88, 94, 95, 96, 97, 104, 111, 112, 113, 127, 128):
        sizes = torch.tensor([prefix, 128 - prefix], device="cuda", dtype=torch.int32)
        mask = block_sparse_mask(ids, 2, sizes, 256, 256, 128)
        ref, _, *ref_grads = attention_reference(*(t.transpose(1, 2) for t in (q, k, v)), mask, do=do.transpose(1, 2), dtype=torch.float64)
        ref, *ref_grads = (t.transpose(1, 2) for t in (ref, *ref_grads))
        out, lse = BSA.block_sparse_attention_forward(q, k, v, ids, 2, sizes, sparse_block_size=128, layout="bshd")
        grads = BSA.block_sparse_attention_backward(do, q, k, v, out, lse, ids, 2, sizes, sparse_block_size=128, layout="bshd")
        for name, actual, expected in zip(("O", "dQ", "dK", "dV"), (out, *grads), (ref, *ref_grads)):
            torch.testing.assert_close(actual.double(), expected, atol=0.03, rtol=0.03, msg=lambda msg: name + ": " + msg)


@pytest.mark.L0
def test_sm100_blk128_partial_blocks_graph_replay():
    """A captured forward + backward reads block_sizes at replay time."""
    BSA = import_bsa(archs=(100, 103), reason="blk128 backward block_sizes are SM100/SM103 only")
    torch.manual_seed(19)
    q, k, v, do = [torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 2).expand(1, 2, 2, 2).contiguous()
    sizes = torch.tensor([17, 69], device="cuda", dtype=torch.int32)

    def run():
        out, lse = BSA.block_sparse_attention_forward(q, k, v, ids, 2, sizes, sparse_block_size=128, layout="bshd")
        grads = BSA.block_sparse_attention_backward(do, q, k, v, out, lse, ids, 2, sizes, sparse_block_size=128, layout="bshd")
        return (out, *grads)

    run()
    torch.cuda.synchronize()
    with cuda_graph() as graph:
        with torch.cuda.graph(graph):
            captured = run()
        for lengths in ([128, 1], [17, 69]):
            sizes.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            graph.replay()
            for actual, expected in zip(captured, run()):
                torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)


@pytest.mark.L1
def test_sm100_blk128_auto_bucket_cache_transitions():
    """One visible token gives an exact oracle across single/multiple Q groups."""
    BSA = import_bsa(archs=(100, 103), reason="blk128 backward block_sizes are SM100/SM103 only")
    k = torch.zeros(1, 128, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.full_like(k, -7)
    v[:, 0] = 2
    sizes = torch.ones(1, device="cuda", dtype=torch.int32)
    # Reuse the same process and caches across the automatic bucket boundary.
    for blocks in (1499, 1500, 2049, 1500, 1499):
        q = torch.zeros(1, blocks * 128, 2, 128, device="cuda", dtype=torch.bfloat16)
        do = torch.ones_like(q)
        ids = torch.zeros(1, 2, blocks, 4, device="cuda", dtype=torch.int32)
        counts = torch.ones(1, 2, blocks, device="cuda", dtype=torch.int32)
        out, lse = BSA.block_sparse_attention_forward(q, k, v, ids, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=128, layout="bshd")
        dq, dk, dv = BSA.block_sparse_attention_backward(
            do, q, k, v, out, lse, ids, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=128, layout="bshd"
        )
        expected_dv = torch.zeros_like(v)
        expected_dv[:, 0] = blocks * 128
        for actual, expected in ((out, torch.full_like(q, 2)), (dq, torch.zeros_like(q)), (dk, torch.zeros_like(k)), (dv, expected_dv)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
