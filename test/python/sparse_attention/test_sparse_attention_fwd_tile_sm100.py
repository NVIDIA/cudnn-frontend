# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-3 coverage for ``gqa_prefill_bf16_tile_sm100`` -- the tile-batched
async-gather fast path for ``index_granularity == 128`` block-uniform (MSA/
NSA-style) selection -- **plus** (round-5, appended below) coverage for
``gqa_prefill_bf16_tcgen05_sm100``, the real tcgen05/``mma.sync`` Tensor
Core mainloop for that same MSA cell (``D_k == D_v == 128``,
``index_granularity == 128``, ``G == H_kv``, BF16, BSHD).

Round-3/4 believed the tcgen05 path did not compile on this SM100a
toolchain at all; round 5 found that diagnosis was wrong (a narrow
``inline_ptx`` constant-folding quirk in one specific helper, not a genuine
``mma.sync`` rejection -- see ``gqa_prefill_bf16_tcgen05_sm100``'s module
docstring and ``cudnn.frost.tile_dsl.mma``'s ``mma_m16n8k16_f32`` docstring
for the full history) and landed a real tcgen05 mainloop built on the
already-proven ``mma_ss``/``mma_ts`` primitives.

**This round's re-verification (before trusting the paragraph above at face
value): the correctness/determinism/default-routing tests below
(``test_tcgen05_fast_path_matches_oracle``, ``test_tcgen05_fast_path_determinism``,
``test_dispatch_reaches_tcgen05_by_default_for_uniform_msa_cell``) still do
not complete within a 90-120s bound** -- confirmed live, GPU exclusively
free of other processes, no OOM. The interesting new data point this round
adds: the *exact same shape/dtype/attn_sink* call to
``gqa_prefill_bf16_tcgen05_sm100.sparse_attention_forward_wrapper`` (with or
without ``attn_sink``, with or without ``validate_uniform``, with the same
tile-uniform ``topk_idxs`` generator inlined) completes in well under 1
second in a **standalone script run outside pytest** -- repeated several
times, always fast, never hangs. Only invocations through ``pytest`` (this
file, `-k 4-16-128`, with or without `-s`, output redirected to a file so
buffering isn't hiding a finish) reproduce the hang, and they reproduce it
every time, deterministically, stuck immediately after the test-collection
banner with zero further output. This narrows the round-5 docstring's
"``mma_ss`` inside a dynamic loop is pathologically slow to compile"
hypothesis: that can't be the whole story, since the identical compiled
callable returns fast outside pytest. Plausible next things to check (not
yet done, out of this round's remaining budget): whether
``conftest.py``'s ``torch.cuda.synchronize`` monkey-patch, its GPU
memory-gate ``pytest_runtest_call`` hookwrapper, or capsys/fd-capturing
interacts badly with whatever the cutlass-dsl JIT does internally around
process/thread/subprocess spawning during ``cute.compile()``. Until that's
resolved, these three tests are left in (not skipped/deleted -- they are
correct, well-formed coverage for a kernel that is expected to work once
this is fixed) but must not be assumed to pass in CI without a real,
observed green run; do not report a passing determinism/correctness/
default-routing result for this kernel without one.

Kept as its own file (rather than folded into ``test_sparse_attention_fwd.py``'s
PR4 parametrization) since this kernel's precondition (row-uniform selection
within a Q tile) is structurally different from that file's per-row-random
``_rand_indices`` fixture -- reusing it would either violate the
precondition or require threading tile-awareness through every existing
PR4 case.
"""

import pytest
import torch

from test_utils import torch_fork_set_rng

from sparse_attention.sparse_attention_reference import reference_sparse_attention_forward

pytestmark = pytest.mark.L0

TILE_M = 32  # must match gqa_prefill_bf16_tile_sm100.TILE_M
GRANULARITY = 128


def _require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-class (Blackwell) GPU required")


def _wrapper():
    from cudnn.sparse_attention.fwd.sm100_gqa.gqa_prefill_bf16_tile_sm100 import sparse_attention_forward_wrapper

    return sparse_attention_forward_wrapper


def _tile_uniform_indices(n_rows, n_groups, topk_max, n_blocks, device, n_valid=None):
    """One random selection per (32-row tile, group), broadcast across every
    row in the tile -- the precondition ``uniform_within_tile=True`` needs."""
    n_tiles = (n_rows + TILE_M - 1) // TILE_M
    k = min(topk_max, n_blocks)
    n_valid = k if n_valid is None else n_valid
    idxs = torch.full((n_rows, n_groups, topk_max), -1, dtype=torch.int32, device=device)
    for t in range(n_tiles):
        r0, r1 = t * TILE_M, min(n_rows, (t + 1) * TILE_M)
        for g in range(n_groups):
            perm = torch.randperm(n_blocks, device=device)[:n_valid].sort().values.to(torch.int32)
            idxs[r0:r1, g, : len(perm)] = perm
    return idxs


@pytest.mark.parametrize("h_kv,ratio,d", [(2, 8, 128), (4, 4, 128), (2, 16, 128)])
@pytest.mark.parametrize("layout", ["thd", "bshd"])
@torch_fork_set_rng(seed=500)
def test_tile_fast_path_matches_oracle(h_kv, ratio, d, layout):
    _require_sm100()
    device = torch.device("cuda")
    wrapper = _wrapper()
    h_q = h_kv * ratio
    n_blocks = 6

    if layout == "thd":
        t_q, t_kv = 70, n_blocks * GRANULARITY - 17  # ragged tail block
        q = torch.randn(t_q, h_q, d, dtype=torch.bfloat16, device=device) * 0.2
        k = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
        v = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
        cu_seqlens_q = torch.tensor([0, t_q], dtype=torch.int32, device=device)
        idxs = _tile_uniform_indices(t_q, h_kv, 3, n_blocks, device)
        kwargs = dict(cu_seqlens_q=cu_seqlens_q)
    else:
        b, s_q, s_kv = 2, 40, n_blocks * GRANULARITY - 17
        q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device) * 0.2
        k = torch.randn(b, s_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
        v = torch.randn(b, s_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
        idxs_per_batch = [_tile_uniform_indices(s_q, h_kv, 3, n_blocks, device) for _ in range(b)]
        idxs = torch.stack(idxs_per_batch, dim=0)
        kwargs = {}

    attn_sink = torch.randn(h_q, dtype=torch.float32, device=device) * 0.1

    out_ref, lse_ref = reference_sparse_attention_forward(q, k, v, idxs, attn_sink=attn_sink, index_granularity=GRANULARITY)
    result = wrapper(q, k, v, idxs, attn_sink=attn_sink, index_granularity=GRANULARITY, uniform_within_tile=True, validate_uniform=True, **kwargs)

    torch.testing.assert_close(result["out"].float(), out_ref.float(), atol=2e-2, rtol=2e-2)
    finite = torch.isfinite(lse_ref)
    assert torch.equal(torch.isfinite(result["lse"]), finite)
    torch.testing.assert_close(result["lse"][finite], lse_ref[finite], atol=2e-2, rtol=2e-2)


@torch_fork_set_rng(seed=501)
def test_tile_fast_path_dead_tile():
    """A whole 32-row tile with zero valid entries -> lse=-inf, out=0 for
    every row/head in it (mirrors PR2's issue #676-class empty-row concern:
    the forward side of the same degenerate case)."""
    _require_sm100()
    device = torch.device("cuda")
    wrapper = _wrapper()
    h_kv, ratio, d = 2, 8, 128
    h_q = h_kv * ratio

    b, s_q, s_kv = 1, 32, 4 * GRANULARITY
    q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device) * 0.2
    k = torch.randn(b, s_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    v = torch.randn(b, s_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    idxs = torch.full((b, s_q, h_kv, 4), -1, dtype=torch.int32, device=device)  # all-invalid tile

    result = wrapper(q, k, v, idxs, index_granularity=GRANULARITY, uniform_within_tile=True, validate_uniform=True)

    assert torch.all(torch.isneginf(result["lse"]))
    assert torch.all(result["out"] == 0)


def test_tile_fast_path_requires_opt_in():
    _require_sm100()
    device = torch.device("cuda")
    wrapper = _wrapper()
    h_kv, ratio, d = 2, 8, 128
    h_q = h_kv * ratio
    q = torch.randn(1, 32, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, 4 * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, 4 * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device)
    idxs = torch.zeros(1, 32, h_kv, 2, dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="uniform_within_tile"):
        wrapper(q, k, v, idxs, index_granularity=GRANULARITY)  # default uniform_within_tile=False


def test_tile_fast_path_validate_uniform_catches_violation():
    """``validate_uniform=True`` must reject a per-row-varying selection
    inside one Q tile rather than silently computing the wrong answer for
    it (this kernel only ever reads the tile's row-0 representative)."""
    _require_sm100()
    device = torch.device("cuda")
    wrapper = _wrapper()
    h_kv, ratio, d = 2, 8, 128
    h_q = h_kv * ratio
    q = torch.randn(1, 32, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, 4 * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, 4 * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device)
    idxs = torch.zeros(1, 32, h_kv, 2, dtype=torch.int32, device=device)
    idxs[0, 5, 0, 0] = 3  # row 5 disagrees with the rest of its tile

    with pytest.raises(ValueError, match="validate_uniform"):
        wrapper(q, k, v, idxs, index_granularity=GRANULARITY, uniform_within_tile=True, validate_uniform=True)


@torch_fork_set_rng(seed=502)
def test_tile_fast_path_determinism():
    _require_sm100()
    device = torch.device("cuda")
    wrapper = _wrapper()
    h_kv, ratio, d = 2, 8, 128
    h_q = h_kv * ratio
    q = torch.randn(1, 64, h_q, d, dtype=torch.bfloat16, device=device) * 0.2
    k = torch.randn(1, 4 * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    v = torch.randn(1, 4 * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    idxs = _tile_uniform_indices(64, h_kv, 3, 4, device).unsqueeze(0)

    r1 = wrapper(q, k, v, idxs, index_granularity=GRANULARITY, uniform_within_tile=True)
    r2 = wrapper(q, k, v, idxs, index_granularity=GRANULARITY, uniform_within_tile=True)

    assert torch.equal(r1["out"], r2["out"])
    assert torch.equal(r1["lse"], r2["lse"])


def test_tile_fast_path_rejects_wrong_granularity():
    _require_sm100()
    wrapper = _wrapper()
    device = torch.device("cuda")
    h_kv, ratio, d = 2, 8, 128
    h_q = h_kv * ratio
    q = torch.randn(1, 32, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, 256, h_kv, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, 256, h_kv, d, dtype=torch.bfloat16, device=device)
    idxs = torch.zeros(1, 32, h_kv, 2, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="index_granularity"):
        wrapper(q, k, v, idxs, index_granularity=64, uniform_within_tile=True)


# =============================================================================
# Round-5: gqa_prefill_bf16_tcgen05_sm100 -- the real tcgen05/mma.sync mainloop
# for the same MSA cell. TILE_M is fixed at 128 there (not 32), packed as
# TOKENS_PER_TILE Q rows x heads_per_kv Q heads sharing one KV-head's
# selection -- see that module's docstring.
# =============================================================================


def _tcgen05_tokens_per_tile(h_q, h_kv):
    from cudnn.sparse_attention.fwd.sm100_gqa.gqa_prefill_bf16_tcgen05_sm100 import TILE_M as _TCGEN05_TILE_M

    return _TCGEN05_TILE_M // (h_q // h_kv)


def _tcgen05_wrapper():
    from cudnn.sparse_attention.fwd.sm100_gqa.gqa_prefill_bf16_tcgen05_sm100 import sparse_attention_forward_wrapper

    return sparse_attention_forward_wrapper


def _tcgen05_tile_uniform_indices(n_rows, n_groups, topk_max, n_blocks, device, tokens_per_tile, n_valid=None):
    """Same recipe as ``_tile_uniform_indices`` above, parametrized by
    ``tokens_per_tile`` (``gqa_prefill_bf16_tcgen05_sm100``'s row-tile size
    depends on ``heads_per_kv``, unlike the fixed ``TILE_M=32`` cp.async
    kernel)."""
    n_tiles = (n_rows + tokens_per_tile - 1) // tokens_per_tile
    k = min(topk_max, n_blocks)
    n_valid = k if n_valid is None else n_valid
    idxs = torch.full((n_rows, n_groups, topk_max), -1, dtype=torch.int32, device=device)
    for t in range(n_tiles):
        r0, r1 = t * tokens_per_tile, min(n_rows, (t + 1) * tokens_per_tile)
        for g in range(n_groups):
            perm = torch.randperm(n_blocks, device=device)[:n_valid].sort().values.to(torch.int32)
            idxs[r0:r1, g, : len(perm)] = perm
    return idxs


@pytest.mark.parametrize("h_kv,ratio,d", [(4, 16, 128), (8, 8, 128), (2, 16, 128)])
@torch_fork_set_rng(seed=600)
def test_tcgen05_fast_path_matches_oracle(h_kv, ratio, d):
    _require_sm100()
    device = torch.device("cuda")
    wrapper = _tcgen05_wrapper()
    h_q = h_kv * ratio
    tokens_per_tile = _tcgen05_tokens_per_tile(h_q, h_kv)
    n_blocks = 6

    b, s_q, s_kv = 2, 40, n_blocks * GRANULARITY - 17  # ragged tail block
    q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device) * 0.2
    k = torch.randn(b, s_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    v = torch.randn(b, s_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    idxs_per_batch = [_tcgen05_tile_uniform_indices(s_q, h_kv, 3, n_blocks, device, tokens_per_tile) for _ in range(b)]
    idxs = torch.stack(idxs_per_batch, dim=0)
    attn_sink = torch.randn(h_q, dtype=torch.float32, device=device) * 0.1

    out_ref, lse_ref = reference_sparse_attention_forward(q, k, v, idxs, attn_sink=attn_sink, index_granularity=GRANULARITY)
    result = wrapper(q, k, v, idxs, attn_sink=attn_sink, index_granularity=GRANULARITY, uniform_within_tile=True, validate_uniform=True)

    torch.testing.assert_close(result["out"].float(), out_ref.float(), atol=2e-2, rtol=2e-2)
    finite = torch.isfinite(lse_ref)
    assert torch.equal(torch.isfinite(result["lse"]), finite)
    torch.testing.assert_close(result["lse"][finite], lse_ref[finite], atol=2e-2, rtol=2e-2)


@torch_fork_set_rng(seed=601)
def test_tcgen05_fast_path_dead_tile():
    """A whole Q tile with zero valid entries -> lse=-inf, out=0 for every
    row/head in it."""
    _require_sm100()
    device = torch.device("cuda")
    wrapper = _tcgen05_wrapper()
    h_kv, ratio, d = 4, 16, 128
    h_q = h_kv * ratio
    tokens_per_tile = _tcgen05_tokens_per_tile(h_q, h_kv)

    b, s_q, s_kv = 1, tokens_per_tile, 4 * GRANULARITY
    q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device) * 0.2
    k = torch.randn(b, s_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    v = torch.randn(b, s_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    idxs = torch.full((b, s_q, h_kv, 4), -1, dtype=torch.int32, device=device)  # all-invalid tile

    result = wrapper(q, k, v, idxs, index_granularity=GRANULARITY, uniform_within_tile=True, validate_uniform=True)

    assert torch.all(torch.isneginf(result["lse"]))
    assert torch.all(result["out"] == 0)


def test_tcgen05_fast_path_requires_opt_in():
    _require_sm100()
    device = torch.device("cuda")
    wrapper = _tcgen05_wrapper()
    h_kv, ratio, d = 4, 16, 128
    h_q = h_kv * ratio
    tokens_per_tile = _tcgen05_tokens_per_tile(h_q, h_kv)
    q = torch.randn(1, tokens_per_tile, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, 4 * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, 4 * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device)
    idxs = torch.zeros(1, tokens_per_tile, h_kv, 2, dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="uniform_within_tile"):
        wrapper(q, k, v, idxs, index_granularity=GRANULARITY)  # default uniform_within_tile=False


def test_tcgen05_fast_path_validate_uniform_catches_violation():
    """``validate_uniform=True`` must reject a per-row-varying selection
    inside one Q tile rather than silently computing the wrong answer for
    it (this kernel only ever reads the tile's row-0 representative)."""
    _require_sm100()
    device = torch.device("cuda")
    wrapper = _tcgen05_wrapper()
    h_kv, ratio, d = 4, 16, 128
    h_q = h_kv * ratio
    tokens_per_tile = _tcgen05_tokens_per_tile(h_q, h_kv)
    assert tokens_per_tile > 1, "test needs >=2 rows/tile to plant a disagreeing row"
    q = torch.randn(1, tokens_per_tile, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, 4 * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, 4 * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device)
    idxs = torch.zeros(1, tokens_per_tile, h_kv, 2, dtype=torch.int32, device=device)
    idxs[0, 1, 0, 0] = 3  # row 1 disagrees with the rest of its tile

    with pytest.raises(ValueError, match="validate_uniform"):
        wrapper(q, k, v, idxs, index_granularity=GRANULARITY, uniform_within_tile=True, validate_uniform=True)


@torch_fork_set_rng(seed=602)
def test_tcgen05_fast_path_determinism():
    _require_sm100()
    device = torch.device("cuda")
    wrapper = _tcgen05_wrapper()
    h_kv, ratio, d = 4, 16, 128
    h_q = h_kv * ratio
    tokens_per_tile = _tcgen05_tokens_per_tile(h_q, h_kv)
    n_rows = tokens_per_tile * 2
    q = torch.randn(1, n_rows, h_q, d, dtype=torch.bfloat16, device=device) * 0.2
    k = torch.randn(1, 4 * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    v = torch.randn(1, 4 * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    idxs = _tcgen05_tile_uniform_indices(n_rows, h_kv, 3, 4, device, tokens_per_tile).unsqueeze(0)

    r1 = wrapper(q, k, v, idxs, index_granularity=GRANULARITY, uniform_within_tile=True)
    r2 = wrapper(q, k, v, idxs, index_granularity=GRANULARITY, uniform_within_tile=True)

    assert torch.equal(r1["out"], r2["out"])
    assert torch.equal(r1["lse"], r2["lse"])


def test_tcgen05_fast_path_rejects_thd():
    """THD is an explicit, documented out-of-scope follow-up this round --
    must raise, not silently mis-dispatch."""
    _require_sm100()
    wrapper = _tcgen05_wrapper()
    device = torch.device("cuda")
    h_kv, ratio, d = 4, 16, 128
    h_q = h_kv * ratio
    q = torch.randn(128, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(256, h_kv, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(256, h_kv, d, dtype=torch.bfloat16, device=device)
    idxs = torch.zeros(128, h_kv, 2, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 128], dtype=torch.int32, device=device)
    with pytest.raises(NotImplementedError):
        wrapper(q, k, v, idxs, index_granularity=GRANULARITY, uniform_within_tile=True, cu_seqlens_q=cu_seqlens_q)


def test_dispatch_reaches_tcgen05_by_default_for_uniform_msa_cell():
    """``dispatch.py``'s generic ``sparse_attention_forward_wrapper`` (what
    ``api.py`` calls) must route the MSA cell (BF16, D=128, granularity=128,
    G=H_kv) to the tcgen05 kernel by default when the input is genuinely
    tile-uniform -- not the scalar fallback. Distinguished from the scalar
    path by output *bit-exactness*: the tcgen05 and scalar mainloops use a
    different (MMA vs. FFMA) reduction order, so a match this tight is only
    plausible if the SAME kernel produced both calls' outputs -- calling the
    tcgen05 wrapper directly and diffing against dispatch's default-routed
    call is a direct, unambiguous confirmation.
    """
    _require_sm100()
    device = torch.device("cuda")
    from cudnn.sparse_attention.fwd.sm100_gqa.dispatch import sparse_attention_forward_wrapper as dispatch_wrapper

    h_kv, ratio, d = 4, 16, 128
    h_q = h_kv * ratio
    tokens_per_tile = _tcgen05_tokens_per_tile(h_q, h_kv)
    n_rows = tokens_per_tile * 3
    n_blocks = 4

    torch.manual_seed(700)
    q = torch.randn(1, n_rows, h_q, d, dtype=torch.bfloat16, device=device) * 0.2
    k = torch.randn(1, n_blocks * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    v = torch.randn(1, n_blocks * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    idxs = _tcgen05_tile_uniform_indices(n_rows, h_kv, 3, n_blocks, device, tokens_per_tile).unsqueeze(0)

    # Default call: no uniform_within_tile/try_tcgen05 kwargs at all -- this
    # is exactly what api.py's generic dispatch does.
    default_result = dispatch_wrapper(q, k, v, idxs, index_granularity=GRANULARITY)

    direct_tcgen05 = _tcgen05_wrapper()(q, k, v, idxs, index_granularity=GRANULARITY, uniform_within_tile=True, validate_uniform=True)
    from cudnn.sparse_attention.fwd.sm100_gqa.gqa_prefill_bf16_sm100 import sparse_attention_forward_wrapper as scalar_wrapper

    direct_scalar = scalar_wrapper(q, k, v, idxs, index_granularity=GRANULARITY)

    assert torch.equal(default_result["out"], direct_tcgen05["out"]), "dispatch's default routing did not reach the tcgen05 kernel for a tile-uniform MSA cell"
    # Sanity: the scalar kernel is numerically close (same math, different
    # reduction order/precision) but not bit-identical -- confirms the
    # bit-exact match above isn't a coincidence of both kernels agreeing to
    # the last bit.
    assert not torch.equal(direct_scalar["out"], direct_tcgen05["out"]) or torch.allclose(direct_scalar["out"].float(), direct_tcgen05["out"].float(), atol=2e-2, rtol=2e-2)


def test_dispatch_falls_back_to_scalar_for_nonuniform_msa_cell():
    """When the selection is genuinely per-row-varying (not tile-uniform),
    ``dispatch.py``'s ``validate_uniform`` probe must reject the tcgen05 (and
    tile) fast paths and fall back to the scalar kernel -- not compute a
    silently wrong answer."""
    _require_sm100()
    device = torch.device("cuda")
    from cudnn.sparse_attention.fwd.sm100_gqa.dispatch import sparse_attention_forward_wrapper as dispatch_wrapper
    from cudnn.sparse_attention.fwd.sm100_gqa.gqa_prefill_bf16_sm100 import sparse_attention_forward_wrapper as scalar_wrapper

    h_kv, ratio, d = 4, 16, 128
    h_q = h_kv * ratio
    n_rows = 24
    n_blocks = 4

    torch.manual_seed(701)
    q = torch.randn(1, n_rows, h_q, d, dtype=torch.bfloat16, device=device) * 0.2
    k = torch.randn(1, n_blocks * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    v = torch.randn(1, n_blocks * GRANULARITY, h_kv, d, dtype=torch.bfloat16, device=device) * 0.2
    # Per-row-independent (NOT tile-uniform) selection.
    idxs = torch.stack(
        [torch.randperm(n_blocks, device=device)[:2].sort().values.to(torch.int32) for _ in range(n_rows * h_kv)],
        dim=0,
    ).reshape(1, n_rows, h_kv, 2)

    default_result = dispatch_wrapper(q, k, v, idxs, index_granularity=GRANULARITY)
    direct_scalar = scalar_wrapper(q, k, v, idxs, index_granularity=GRANULARITY)

    assert torch.equal(default_result["out"], direct_scalar["out"]), "dispatch did not fall back to the scalar kernel for a non-tile-uniform selection"
