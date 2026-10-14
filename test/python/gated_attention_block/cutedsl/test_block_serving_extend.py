# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's EXTEND (serving prefill) form: part one, the write-through of post-RoPE K / V (and the raw
indexer key) into paged pools at a slot mapping -- ``paged_kv_page_size``; part two (the second half of this module, "The EXTEND
form"), attention OVER the pools -- ``kv_source="paged"`` on a packed, bounded (``token_capacity``) declaration whose SDPA reads
every sequence's prefix and its new tokens from ``k_cache`` / ``v_cache`` through ``execute(block_table=, kv_lens=)`` on the
bottom-right diagonal: the declaration rows and the envelope that predicts them, the stream-order / no-PDL pin, the execute form
checks, and the Rubin cells -- the empty-prefix step vs the dense block, the 2-chunk extend and the prefix-hit step vs the
single-shot oracle on L3, L1 vs L3 bitwise, the degenerate matrix on serving buffers (padded entries with kv_len 0 and the NULL
block, a live-Q dead-KV entry, mid-page prefixes, tail tiles, n_kv 1 / 3 / 4 and the ring wrap, pads at the END and in the
MIDDLE, a row-sliced block table), the admitted spellings executed (the lengths form, the fused norm + RoPE and the out-of-place
pipelines, a full and a wider block table), the bounded-token contract (live rows bitwise the tight declaration, pad rows exactly
0 on a sentinel-filled out over a zeroed workspace), and a CUDA-graph replay over new lengths / tables / slots.

The pool contract a serving stack binds: a 4-D view ``[num_pages, H_kv, P, D]`` with ``D`` contiguous and the other strides
free -- HND compact pools, NHD storage by strides, and a stack's UNIFIED content row ``[num_pages, H_kv, P, 2D]`` whose two
halves (K at ``+0``, V at ``+D``, token stride ``2D``) are the two pool views.  Host cells (any GPU, the declaration cells
even on the CPU): the attribute's validator rows, the packed (``thd=True``) declaration ACCEPTED with the stage in pipeline
position, the mirror-image refusals of every cache input on a block declared without the attribute (under
``torch.cuda.set_sync_debug_mode("error")``), the declared block's required inputs and form checks (every non-innermost
pool stride a whole number of 16-byte vectors, a typed ``ValueError`` otherwise), the typed declines, the CuTe DSL floor
probe, the unchanged workspace carve, and the kernel / stage themselves -- a bitwise scatter against a torch reference into
all three pool layouts at pages 16 / 32 / 64 / 128 with int32 and int64 slots, padded (negative) and past-the-pool slots
included.  Rubin cells: the whole block, dense AND packed, bf16 AND f16, on every layout at every page size -- the live pool
rows bitwise the slab's post-RoPE K / V bands attention consumed and within the suite's bars of the per-sequence oracle,
padded tokens unwritten on a NaN-poisoned pool, ``out`` bitwise the block's without write-through, the workspace size
unchanged -- plus the other bf16 pipelines (fused norm + RoPE, dense and packed; out-of-place Q / K / V; the fused gate) and
the indexer band's raw-key pool.
"""

import contextlib
import inspect
import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import (
    KV_SOURCE_PAGED,
    KV_SOURCE_SELF,
    SERVING_PAGE_SIZES,
    SERVING_POOL_LAYOUTS,
    DecodeEnvelope,
    ExtendEnvelope,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    QsaSpec,
    QuantSpec,
    ServingEnvelope,
    index_k_raw_view,
)
from cudnn.gated_attention_block.api import _EXTEND_DECLINES, PAGED_KV_PAGE_ALIGN, _align_up, _CacheWrite, _cols, _Projection, _Sdpa, _SparseSdpa, _view
from cudnn.gated_attention_block.kernels.proj_gemm import build_proj_gemm

# Below FROST's CuTe DSL floor the block's stages cannot import their kernels: the module skips as a whole there (the sibling
# block modules' convention); the typed decline itself is pinned by ``test_the_stage_declines_a_dsl_below_the_floor_by_version``.
requirement_error = cutedsl_requirement_error("the gated attention block's serving tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107
# The capture cell (g) needs the projection GEMM driver's capture-safe route (``build_proj_gemm(capture_safe=True)``: the forced-tile JIT
# route, no cuDNN handle at the first execute).  Where the driver has no such route, a fresh declaration's first execute inside a capture
# is the graph route's handle creation, so the cell is skipped there instead of asserting what that tree cannot deliver; the serving
# declaration still marks both dense projection stages capture-safe (pinned below), which the route consumes once it is in the tree.
requires_capture_safe_gemm_route = pytest.mark.skipif(
    "capture_safe" not in inspect.signature(build_proj_gemm).parameters,
    reason="the projection GEMM driver has no capture_safe route on this tree (a fresh plan's first execute inside a capture takes the graph route)",
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import (  # noqa: E402
    RefGeometry,
    build_rope_tables,
    cu_seqlens_of,
    gated_attention_block_reference,
    gated_attention_block_reference_packed,
    make_inputs,
    make_packed_inputs,
    qk_norm_rope_reference,
)

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="the cache-write kernel needs a GPU (any arch)")

# A shrunk d256 geometry for the declaration / stage cells and the Rubin layout x page matrix; the 24 / 2 geometry of the
# block-sparse model at TP 1 for the indexer band's raw-key pool.
_SMALL_D256 = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)
_FLASH_NEXT = dict(d_model=2560, h_q=24, h_kv=2, d_head=256, rope_dim=64)
_BAND_BAR = dict(rtol=2.0**-7, atol=1e-3)  # the suite's bar for a slab band against the oracle (one bf16 rounding of the GEMM output)
# The norm suite's absolute term for the normed K against the oracle norm of the SAME input, per dtype (test_qk_norm_rope.py `_check`:
# a couple of ulps of fp32 op order) -- never the bf16 term on an f16 cell.
_NORM_ATOL = {torch.bfloat16: 8e-3, torch.float16: 1e-3}
_PAGES = (16, 32, 64, 128)  # the kernel block sizes the serving stacks allocate
_LAYOUTS = ("hnd", "nhd", "unified")  # HND compact / NHD by strides / the unified [P, H, page, 2D] content row (K at +0, V at +D)
# The packed form's sequences: a zero-length sequence in the MIDDLE (the general case), a tail tile on every live one, T = 512.
_PACKED_LENS = (300, 0, 128, 84)
_DENSE_BS = (2, 256)  # the dense form at the same T = 512


@contextlib.contextmanager
def _no_host_sync():
    """Every declaration-time and form check stays on the host without a device sync (Rule 3) -- enforced, not read."""
    if not torch.cuda.is_available():
        yield
        return
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev)


def _bits(x: torch.Tensor) -> torch.Tensor:
    """The 16-bit patterns of a bf16 / f16 tensor: a NaN-safe bitwise comparison (the sentinel is NaN)."""
    return x.contiguous().view(torch.int16)


def _pools(n_pages: int, h: int, page_size: int, d: int, layout: str, dtype=torch.bfloat16, device=_DEV, fill: float = float("nan")):
    """Two ``fill``-filled pools (NaN by default: the write-through cells' poison) as the kernel's 4-D ``[P, H, page_size, D]``
    views -- ``(k_cache, v_cache)``.

    ``hnd``: HND compact storage, one tensor per pool.  ``nhd``: NHD storage (``[P, page_size, H, D]`` in memory) permuted
    into the same logical order -- the layout is nothing but the strides.  ``unified``: ONE ``[P, H, page_size, 2D]``
    content row per token (K at ``+0``, V at ``+D``; token stride ``2D``), the two pools being its two halves."""
    if layout == "hnd":
        return tuple(torch.full((n_pages, h, page_size, d), fill, device=device, dtype=dtype) for _ in range(2))
    if layout == "nhd":
        return tuple(torch.full((n_pages, page_size, h, d), fill, device=device, dtype=dtype).permute(0, 2, 1, 3) for _ in range(2))
    assert layout == "unified"
    row = torch.full((n_pages, h, page_size, 2 * d), fill, device=device, dtype=dtype)
    return row[..., :d], row[..., d:]


def _slots(t: int, n_slots: int, gen: torch.Generator, dtype=torch.int32, pad_every: int = 0, device=_DEV) -> torch.Tensor:
    """A random injective slot mapping of ``t`` tokens into ``n_slots`` slots; every ``pad_every``-th token padded (-1)."""
    perm = torch.randperm(n_slots, generator=gen, device=device)[:t].to(dtype)
    if pad_every:
        perm[::pad_every] = -1
    return perm


def _scatter_ref(src: torch.Tensor, slot: torch.Tensor, pool: torch.Tensor, page_size: int) -> torch.Tensor:
    """The torch reference of the write: a NaN pool with ``src[t]`` at ``(slot // P, :, slot % P)`` for every live slot."""
    ref = torch.full(pool.shape, float("nan"), device=pool.device, dtype=pool.dtype)
    live = (slot >= 0) & (slot < pool.shape[0] * page_size)
    s = slot[live].long()
    ref[s // page_size, :, s % page_size, :] = src[live]
    return ref


def _rows_at(pool: torch.Tensor, slot: torch.Tensor, page_size: int) -> torch.Tensor:
    """The pool rows of the live slots, in token order: ``[T_live, H, D]``."""
    live = slot >= 0
    s = slot[live].long()
    return pool[s // page_size, :, s % page_size, :]


def _written_rows(pool: torch.Tensor) -> int:
    """How many ``[D]`` rows of a NaN-poisoned pool hold finite values (a row written by the kernel is finite end to end)."""
    return int((~torch.isnan(pool.float())).all(-1).sum().item())


def _make_block(geom_kw, batch, seq_len, dtype=torch.bfloat16, **blk_kw):
    """A declared (not compiled) DENSE block on fresh inputs: ``(blk, inp, out, ref_geom)``."""
    geom = GatedAttentionBlockGeometry(**geom_kw)
    ref_geom = RefGeometry(**geom_kw)
    inp = make_inputs(ref_geom, batch=batch, seq_len=seq_len, dtype=dtype, device=_DEV)
    out = torch.empty(batch, seq_len, geom.d_model, device=_DEV, dtype=dtype)
    blk = GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, geom, **blk_kw)
    return blk, inp, out, ref_geom


def _make_packed_block(geom_kw, lens, dtype=torch.bfloat16, **blk_kw):
    """A declared (not compiled) PACKED block (``thd=True``) on fresh inputs: ``(blk, inp, out, meta)``."""
    geom = GatedAttentionBlockGeometry(**geom_kw)
    inp, meta = make_packed_inputs(RefGeometry(**geom_kw), lens, dtype=dtype, device=_DEV)
    out = torch.empty(1, meta["t"], geom.d_model, device=_DEV, dtype=dtype)
    blk = GatedAttentionBlockFwd(
        inp["h"],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        out,
        geom,
        thd=True,
        num_sequences=meta["b"],
        max_seq_len=meta["max_seq_len"],
        **blk_kw,
    )
    return blk, inp, out, meta


def _exec(blk, inp, out, ws, **kw):
    blk.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws, **kw)


def _assert_stage_in_pipeline_position(blk, page_size):
    assert blk.paged_kv_page_size == page_size and isinstance(blk._cache_write, _CacheWrite)
    assert blk._cache_write.page_size == page_size and blk._cache_write.index_band is False
    stages = list(blk._stages)
    assert blk._cache_write in stages
    # After norm + RoPE (it writes the post-RoPE K), before the SDPA (the operands it copies are the SDPA's).
    assert stages.index(blk._norm_rope) < stages.index(blk._cache_write) < stages.index(blk._sdpa)


# ---------------------------------------------------------------------------
# The attribute: validator rows, declines, the stage list, the carve
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_page_size_align_is_the_serving_stacks_granularity():
    assert PAGED_KV_PAGE_ALIGN == 16 and PAGED_KV_PAGE_ALIGN % 4 == 0  # a 4-token sparse block never straddles a page


@pytest.mark.L0
@pytest.mark.parametrize("page_size", list(_PAGES) + [48, 256])
def test_a_legal_page_size_builds_the_stage_in_pipeline_position(page_size):
    blk, *_ = _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=page_size)
    _assert_stage_in_pipeline_position(blk, page_size)


@pytest.mark.L0
@pytest.mark.parametrize("page_size", _PAGES)
def test_a_packed_declaration_builds_the_same_stage(page_size):
    """The write-through under ``thd=True`` is SERVED (the former typed decline is inverted, not deleted): the stage
    addresses tokens, so the packed block -- ``B = 1, S = T`` inside -- builds the same stage at the same pipeline
    position, sized at the packed token count."""
    blk, inp, out, meta = _make_packed_block(_SMALL_D256, _PACKED_LENS, paged_kv_page_size=page_size)
    _assert_stage_in_pipeline_position(blk, page_size)
    assert blk.thd and blk.batch * blk.seq_len == meta["t"] == sum(_PACKED_LENS)
    assert blk._cache_write.batch * blk._cache_write.seq_len == meta["t"]


@pytest.mark.L0
def test_zero_is_off_and_builds_no_stage():
    blk, *_ = _make_block(_SMALL_D256, 1, 64)
    assert blk.paged_kv_page_size == 0 and blk._cache_write is None and all(not isinstance(st, _CacheWrite) for st in blk._stages)
    blk0, *_ = _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=0)
    assert blk0.paged_kv_page_size == 0 and blk0._cache_write is None
    blk_thd, *_ = _make_packed_block(_SMALL_D256, _PACKED_LENS)
    assert blk_thd.paged_kv_page_size == 0 and blk_thd._cache_write is None


@pytest.mark.L0
@pytest.mark.parametrize(
    "page_size, exc, match",
    [
        (15, ValueError, "multiple of 16"),
        (8, ValueError, "multiple of 16"),
        (4, ValueError, "multiple of 16"),
        (-16, ValueError, ">= 0"),
        (16.0, TypeError, "must be an int"),
        ("16", TypeError, "must be an int"),
        (True, TypeError, "must be an int"),
    ],
)
def test_page_size_validator_rows(page_size, exc, match):
    with pytest.raises(exc, match=match):
        _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=page_size)
    with pytest.raises(exc, match=match):
        _make_packed_block(_SMALL_D256, _PACKED_LENS, paged_kv_page_size=page_size)


@pytest.mark.L0
def test_page_size_must_hold_whole_sparse_blocks_on_a_qsa_geometry():
    # 16 % 4 == 0 is implied by the alignment row; the cross-field row still exists so the message names the block size
    # should either constant ever move.
    geom = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec())
    ref_geom = RefGeometry(**_SMALL_D256)
    inp = make_inputs(ref_geom, batch=1, seq_len=64, dtype=torch.bfloat16, device=_DEV)
    out = torch.empty(1, 64, geom.d_model, device=_DEV, dtype=torch.bfloat16)
    blk = GatedAttentionBlockFwd(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, geom, paged_kv_page_size=32
    )
    assert blk.paged_kv_page_size == 32 and isinstance(blk._sdpa, _SparseSdpa) and isinstance(blk._cache_write, _CacheWrite)


@pytest.mark.L0
def test_typed_declines_at_declaration_name_the_feature():
    with pytest.raises(NotImplementedError, match="save_for_backward"):
        _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16, save_for_backward=True, inplace_qkv=False)
    with pytest.raises(NotImplementedError, match="save_for_backward"):
        _make_packed_block(_SMALL_D256, _PACKED_LENS, paged_kv_page_size=16, save_for_backward=True, inplace_qkv=False)
    geom = GatedAttentionBlockGeometry(**_SMALL_D256)
    ref_geom = RefGeometry(**_SMALL_D256)
    inp = make_inputs(ref_geom, batch=1, seq_len=64, dtype=torch.bfloat16, device=_DEV)
    e4 = torch.float8_e4m3fn
    out = torch.empty(1, 64, geom.d_model, device=_DEV, dtype=torch.bfloat16)
    spec = QuantSpec(descale_h=1.0, descale_w_qkvg=1.0, descale_w_o=1.0, scale_q=1.0, scale_k=1.0, scale_v=1.0, scale_o=1.0)
    with pytest.raises(NotImplementedError, match="quant=QuantSpec"):
        GatedAttentionBlockFwd(
            inp["h"].to(e4),
            inp["w_qkvg"].to(e4),
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"].to(e4),
            out,
            geom,
            quant=spec,
            paged_kv_page_size=16,
        )
    # The packed form is SERVED (the former ``thd=True`` decline, inverted): rank-2 token matrices declare without complaint.
    h_tokens = inp["h"].view(64, geom.d_model)
    blk_thd = GatedAttentionBlockFwd(
        h_tokens,
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"].view(64, -1),
        inp["sin"].view(64, -1),
        inp["w_o"],
        out.view(64, geom.d_model),
        geom,
        thd=True,
        num_sequences=2,
        max_seq_len=64,
        paged_kv_page_size=16,
    )
    assert blk_thd.thd and isinstance(blk_thd._cache_write, _CacheWrite) and blk_thd._cache_write.page_size == 16
    # Served pipelines declare without complaint: fused norm + RoPE, out-of-place Q / K / V, the fused gate, fp16 -- dense and packed.
    for kw in (dict(fuse_norm_rope=True), dict(inplace_qkv=False), dict(fuse_gate=True), dict(fuse_norm_rope=True, fuse_gate=True)):
        blk, *_ = _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16, **kw)
        assert isinstance(blk._cache_write, _CacheWrite)
    for kw in (dict(fuse_norm_rope=True), dict(inplace_qkv=False), dict(fuse_gate=True), dict(fuse_norm_rope=True, fuse_gate=True)):
        blk, *_ = _make_packed_block(_SMALL_D256, _PACKED_LENS, paged_kv_page_size=16, **kw)
        assert isinstance(blk._cache_write, _CacheWrite)
    blk16, *_ = _make_block(_SMALL_D256, 1, 64, dtype=torch.float16, paged_kv_page_size=16)
    assert blk16._cache_write.dtype == torch.float16
    blk16p, *_ = _make_packed_block(_SMALL_D256, _PACKED_LENS, dtype=torch.float16, paged_kv_page_size=16)
    assert blk16p._cache_write.dtype == torch.float16


@pytest.mark.L0
def test_the_workspace_carve_is_unchanged_by_write_through():
    """The pools and the slot mapping are the caller's; the stage reserves nothing, so the carve is the dense block's --
    and the packed block's."""
    plain, *_ = _make_block(_SMALL_D256, 2, 128)
    wt, *_ = _make_block(_SMALL_D256, 2, 128, paged_kv_page_size=16)
    assert plain._layout() == wt._layout()
    assert not hasattr(wt._cache_write, "workspace_bytes") and not hasattr(wt._cache_write, "scratch_workspace_bytes")
    plain_p, *_ = _make_packed_block(_SMALL_D256, _PACKED_LENS)
    wt_p, *_ = _make_packed_block(_SMALL_D256, _PACKED_LENS, paged_kv_page_size=64)
    assert plain_p._layout() == wt_p._layout()


@pytest.mark.L0
def test_moved_bytes_counts_every_operand_once_in_and_once_out():
    from cudnn.gated_attention_block.kernels.cache_write import moved_bytes

    assert moved_bytes(1024, 2, 256, elem_bytes=2, n_ops=2, slot_bytes=4) == 2 * 2 * 1024 * 2 * 256 * 2 + 1024 * 4
    blk, *_ = _make_block(_SMALL_D256, 2, 64, paged_kv_page_size=16)
    t = 128
    assert blk._cache_write.moved_bytes() == 2 * 2 * t * 2 * 256 * 2 + t * 4
    assert blk._cache_write.moved_bytes(torch.int64) == 2 * 2 * t * 2 * 256 * 2 + t * 8


@pytest.mark.L0
def test_the_stage_declines_a_dsl_below_the_floor_by_version(monkeypatch):
    """Rule 7: the stage's ``check_support`` reads the installed CuTe DSL version BEFORE importing its kernel module and
    declines a wheel below FROST's floor with the typed message naming the version and the fix -- probed here by
    reporting an old public wheel to the version helper (this module runs at or above the floor, or it skipped)."""
    import cudnn.frost.buffers as buffers

    blk, *_ = _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16)
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.6.2")))
    with pytest.raises(NotImplementedError, match=r"paged KV-cache write-through requires nvidia-cutlass-dsl >= 4\.7\.0; found 4\.6\.2"):
        blk._cache_write.check_support()
    monkeypatch.undo()
    blk._cache_write.check_support()  # at or above the floor the same stage accepts


# ---------------------------------------------------------------------------
# execute(): the mirror-image refusals, the required inputs, the form checks -- host, no sync
# ---------------------------------------------------------------------------


def _cache_args(blk, geom_kw, page_size=16, n_pages=4, slot_dtype=torch.int32, layout="hnd"):
    t = blk.batch * blk.seq_len
    k, v = _pools(n_pages, geom_kw["h_kv"], page_size, geom_kw["d_head"], layout)
    slot = torch.arange(t, device=_DEV, dtype=slot_dtype)
    return dict(k_cache=k, v_cache=v, slot_mapping=slot)


@pytest.mark.L0
@pytest.mark.parametrize("name", ["k_cache", "v_cache", "slot_mapping", "index_k_raw", "block_table", "kv_lens"])
def test_a_block_declared_without_write_through_refuses_every_cache_input(name):
    blk, inp, out, _ = _make_block(_SMALL_D256, 1, 64)
    ws = torch.empty(16, dtype=torch.uint8, device=_DEV)
    args = _cache_args(blk, _SMALL_D256)
    args["index_k_raw"] = args["k_cache"][:, 0]
    args["block_table"] = torch.zeros(1, 4, device=_DEV, dtype=torch.int32)
    args["kv_lens"] = torch.zeros(1, device=_DEV, dtype=torch.int32)
    with _no_host_sync():
        with pytest.raises(ValueError, match=f"{name}.*paged_kv_page_size"):
            _exec(blk, inp, out, ws, **{name: args[name]})


@pytest.mark.L0
def test_a_declared_block_requires_the_pools_and_the_slot_mapping_and_refuses_the_extend_inputs():
    """A write-through-only block (``kv_source="self"``) attends over its own K / V: the EXTEND form's ``block_table`` / ``kv_lens``
    are REFUSED there with a ``ValueError`` naming the declaration that takes them -- the former typed "paged-READ mode is a
    follow-up" decline, INVERTED (the read mode landed as ``kv_source="paged"``; its own cells are below), never deleted."""
    blk, inp, out, _ = _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16)
    ws = torch.empty(16, dtype=torch.uint8, device=_DEV)
    args = _cache_args(blk, _SMALL_D256)
    with _no_host_sync():
        for missing in ("k_cache", "v_cache", "slot_mapping"):
            with pytest.raises(ValueError, match=f"needs {missing} at execute"):
                _exec(blk, inp, out, ws, **{k: v for k, v in args.items() if k != missing})
        with pytest.raises(ValueError, match="paged_kv_page_size=16 needs"):
            _exec(blk, inp, out, ws)
        for read in (dict(block_table=torch.zeros(1, 4, device=_DEV, dtype=torch.int32)), dict(kv_lens=torch.zeros(1, device=_DEV, dtype=torch.int32))):
            with pytest.raises(ValueError, match="belong to a block declared kv_source='paged'"):
                _exec(blk, inp, out, ws, **args, **read)
        # No indexer band on this geometry: a raw-key pool is refused, not ignored.
        with pytest.raises(ValueError, match="index_k_raw.*indexer band"):
            _exec(blk, inp, out, ws, **args, index_k_raw=args["k_cache"][:, 0])
        # The well-formed call reaches the plan check (the block is declared, not compiled).
        with pytest.raises(RuntimeError, match="compile"):
            _exec(blk, inp, out, ws, **args)


@pytest.mark.L0
def test_a_packed_block_takes_a_slot_per_packed_token():
    """Under ``thd=True`` the slot mapping has exactly T entries (one per packed token, pads included) -- the dense
    count rule at ``B = 1, S = T``; the lengths tensor stays the packed block's own requirement (``test_block_thd.py``)."""
    blk, inp, out, meta = _make_packed_block(_SMALL_D256, _PACKED_LENS, paged_kv_page_size=16)
    ws = torch.empty(16, dtype=torch.uint8, device=_DEV)
    t = meta["t"]
    k, v = _pools(-(-t // 16) + 1, 2, 16, 256, "hnd")
    with _no_host_sync():
        with pytest.raises(ValueError, match=f"exactly T={t}"):
            _exec(blk, inp, out, ws, seq_lens=meta["seq_lens"], k_cache=k, v_cache=v, slot_mapping=torch.arange(t - 1, device=_DEV, dtype=torch.int32))
        with pytest.raises(RuntimeError, match="compile"):
            _exec(blk, inp, out, ws, seq_lens=meta["seq_lens"], k_cache=k, v_cache=v, slot_mapping=torch.arange(t, device=_DEV, dtype=torch.int32))


@pytest.mark.L0
def test_a_band_geometry_requires_the_raw_key_pool():
    geom = GatedAttentionBlockGeometry(**_FLASH_NEXT, qsa=QsaSpec(index_band=True))
    ref_geom = RefGeometry(**_FLASH_NEXT)
    inp = make_inputs(ref_geom, batch=1, seq_len=32, dtype=torch.bfloat16, device=_DEV)
    w_qkvg = torch.cat([inp["w_qkvg"], torch.zeros(geom.qsa.index_band_cols, geom.d_model, device=_DEV, dtype=torch.bfloat16)])
    out = torch.empty(1, 32, geom.d_model, device=_DEV, dtype=torch.bfloat16)
    blk = GatedAttentionBlockFwd(inp["h"], w_qkvg, inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, geom, paged_kv_page_size=16)
    assert blk._cache_write.index_band is True
    ws = torch.empty(16, dtype=torch.uint8, device=_DEV)
    args = _cache_args(blk, _FLASH_NEXT)
    # The QsaSpec block's own execute contract comes first (block_ids REQUIRED, form-checked); a well-formed list lets the
    # cache-write checks run.
    args["block_ids"] = torch.full((32, geom.qsa.top_k), -1, device=_DEV, dtype=torch.int32)
    good = torch.full((4, 16, 128), float("nan"), device=_DEV, dtype=torch.bfloat16)
    with _no_host_sync():
        with pytest.raises(ValueError, match="needs index_k_raw at execute"):
            blk.execute(inp["h"], w_qkvg, inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws, **args)
        for bad, match in (
            (good[:, :, :64], r"index_k_raw must be \[num_pages"),  # D = 64
            (good[:, :8], r"index_k_raw must be \[num_pages"),  # page_size 8
            (good.unsqueeze(1), "3-D"),
            (good.float(), "must be torch.bfloat16"),
        ):
            with pytest.raises(ValueError, match=match):
                blk.execute(inp["h"], w_qkvg, inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws, **args, index_k_raw=bad)
        with pytest.raises(RuntimeError, match="compile"):
            blk.execute(inp["h"], w_qkvg, inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws, **args, index_k_raw=good)


def _bad_pool_forms(geom_kw, page_size=16):
    h, d = geom_kw["h_kv"], geom_kw["d_head"]
    good, _ = _pools(4, h, page_size, d, "hnd")
    yield good.float(), "must be torch.bfloat16"
    yield good[:, :1], r"must be \[num_pages"  # H
    yield good[:, :, :8], r"must be \[num_pages"  # page_size
    yield good[..., :128], r"must be \[num_pages"  # D
    yield good[:0], r"must be \[num_pages"  # num_pages 0
    # [P, H, D, page_size] storage viewed as [P, H, page_size, D]: the shape is right, D is strided by page_size.
    yield torch.full((4, h, d, page_size), float("nan"), device=_DEV, dtype=torch.bfloat16).permute(0, 1, 3, 2), "contiguous along D"
    # A row pitch that is not a whole number of 16-byte vectors: 520 B (D + 4 elements), and a unified content row of
    # 2D + 4 elements (1032 B) -- the typed stride rule, on the token stride.
    yield torch.full((4, h, page_size, d + 4), float("nan"), device=_DEV, dtype=torch.bfloat16)[..., :d], "stride\\[2\\].*multiple of 16 bytes"
    yield torch.full((4, h, page_size, 2 * d + 4), float("nan"), device=_DEV, dtype=torch.bfloat16)[..., :d], "stride\\[2\\].*multiple of 16 bytes"
    # The same rule on the head stride: a [P, H, page_size, D] view whose heads are H x page_size x D + 4 elements apart.
    yield torch.full(
        (
            4,
            h * page_size * d + 4 * h,
        ),
        float("nan"),
        device=_DEV,
        dtype=torch.bfloat16,
    ).as_strided((4, h, page_size, d), (h * page_size * d + 4 * h, page_size * d + 4, d, 1)), "stride\\[1\\].*multiple of 16 bytes"
    yield torch.full((4 * h * page_size * d + 1,), float("nan"), device=_DEV, dtype=torch.bfloat16)[1:].view(4, h, page_size, d), "16-byte aligned"
    yield "not a tensor", "must be a"


@pytest.mark.L0
@pytest.mark.parametrize("which", ["k_cache", "v_cache"])
def test_pool_form_checks(which):
    blk, inp, out, _ = _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16)
    ws = torch.empty(16, dtype=torch.uint8, device=_DEV)
    args = _cache_args(blk, _SMALL_D256)
    with _no_host_sync():
        for bad, match in _bad_pool_forms(_SMALL_D256):
            with pytest.raises(ValueError, match=f"{which}.*{match}" if match != "must be a" else match):
                _exec(blk, inp, out, ws, **{**args, which: bad})
    # NHD storage and the two halves of a unified content row pass the same checks (and so does a 1040-B unified pitch:
    # 2D + 8 elements IS a whole number of 16-byte vectors -- the rule is the vector width, not the head dim).
    with _no_host_sync():
        for layout in ("nhd", "unified"):
            k, v = _pools(4, 2, 16, 256, layout)
            with pytest.raises(RuntimeError, match="compile"):
                _exec(blk, inp, out, ws, **{**args, "k_cache": k, "v_cache": v})
        wide = torch.full((4, 2, 16, 2 * 256 + 8), float("nan"), device=_DEV, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="compile"):
            _exec(blk, inp, out, ws, **{**args, "k_cache": wide[..., :256], "v_cache": wide[..., 256:512]})


@pytest.mark.L0
def test_slot_mapping_form_checks():
    blk, inp, out, _ = _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16)
    ws = torch.empty(16, dtype=torch.uint8, device=_DEV)
    args = _cache_args(blk, _SMALL_D256)
    t = 64
    cpu_slot = torch.arange(t, dtype=torch.int32)
    bad = [
        (torch.arange(t, device=_DEV, dtype=torch.int16), "int32 or int64"),
        (torch.arange(t, device=_DEV, dtype=torch.float32), "int32 or int64"),
        (torch.arange(t - 1, device=_DEV, dtype=torch.int32), "exactly T=64"),
        (torch.arange(t + 1, device=_DEV, dtype=torch.int32), "exactly T=64"),
        (torch.arange(t, device=_DEV, dtype=torch.int32).view(2, 32), "1-D"),
        (torch.arange(2 * t, device=_DEV, dtype=torch.int32)[::2], "contiguous"),
        ("slots", "must be a"),
    ]
    if _DEV == "cuda":
        bad.append((cpu_slot, "must live on"))
    with _no_host_sync():
        for slot, match in bad:
            with pytest.raises(ValueError, match=match):
                _exec(blk, inp, out, ws, **{**args, "slot_mapping": slot})
        # int64 passes the form checks (its dtype is a compile key the stage carries both of).
        with pytest.raises(RuntimeError, match="compile"):
            _exec(blk, inp, out, ws, **{**args, "slot_mapping": torch.arange(t, device=_DEV, dtype=torch.int64)})


# ---------------------------------------------------------------------------
# The kernel and the stage on ANY GPU: a bitwise scatter against the torch reference
# ---------------------------------------------------------------------------


@pytest.mark.L2
@_GPU
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
def test_cache_write_binds_int64_pool_strides():
    """A real allocation beyond the 32-bit element offset must bind and preserve low-address canaries."""
    from cudnn.gated_attention_block.kernels.cache_write import compile_cache_write, run_cache_write

    page_size, d = 16, 256
    page_stride = (1 << 32) + page_size * d
    elements = page_stride + page_size * d
    free, _ = torch.cuda.mem_get_info()
    if free < elements * 2 + (1 << 30):
        pytest.skip("the wide-stride regression requires an 8 GiB allocation plus 1 GiB headroom")
    try:
        storage = torch.empty(elements, dtype=torch.bfloat16, device="cuda")
    except torch.OutOfMemoryError:
        pytest.skip("the wide-stride backing allocation no longer fits in available GPU memory")
    pool = torch.as_strided(storage, (2, 1, page_size, d), (page_stride, page_size * d, d, 1))
    pool.fill_(-7)
    wrapped = storage[page_size * d : page_size * d + d]
    wrapped.fill_(-9)
    src = torch.arange(2 * d, dtype=torch.float32, device="cuda").to(torch.bfloat16).view(2, 1, d)
    slots = torch.tensor([0, page_size], dtype=torch.int32, device="cuda")
    recipe = compile_cache_write(dtype=torch.bfloat16, h=1, d=d, page_size=page_size, two_operands=False)
    run_cache_write(recipe, src, None, pool, None, slots, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(pool[0, 0, 0], src[0, 0])
    assert torch.equal(pool[1, 0, 0], src[1, 0])
    assert bool((pool[:, :, 1:] == -7).all())
    assert bool((wrapped == -9).all())


@_GPU
@pytest.mark.parametrize(
    "dtype, h, d, page_size, t, n_pages, slot_dtype, layout",
    [
        pytest.param(
            dtype,
            h,
            d,
            page_size,
            t,
            n_pages,
            slot_dtype,
            layout,
            marks=pytest.mark.L0 if case_id == "bf16_h2_d256_p16" and slot_dtype == torch.int32 and layout == "hnd" else pytest.mark.L1,
            id=f"{case_id}-{slot_id}-{layout}",
        )
        for dtype, h, d, page_size, t, n_pages, case_id in [
            (torch.bfloat16, 2, 256, 16, 512, 40, "bf16_h2_d256_p16"),
            (torch.float16, 2, 256, 32, 300, 12, "f16_h2_d256_p32"),
            (torch.bfloat16, 1, 128, 64, 700, 20, "bf16_h1_d128_p64"),
            (torch.float16, 2, 256, 128, 520, 6, "f16_h2_d256_p128"),
        ]
        for slot_dtype, slot_id in [(torch.int32, "i32"), (torch.int64, "i64")]
        for layout in _LAYOUTS
    ],
)
def test_kernel_scatters_bitwise_with_padded_and_out_of_range_slots(dtype, h, d, page_size, t, n_pages, slot_dtype, layout):
    """Two operands (or one) as COLUMN SLICES of a wider slab -> pools at a random injective slot mapping; padded (-1)
    tokens, a slot past the pool and (int64) a slot >= 2^31 whose low word is in range write NOTHING; everything else is
    bitwise the source row; every unwritten slot keeps its sentinel -- on HND, NHD and the unified content row (whose K
    half and V half are written by the same launch and must not touch each other)."""
    from cudnn.gated_attention_block.kernels.cache_write import compile_cache_write, run_cache_write

    two = h > 1
    gen = torch.Generator(device="cuda").manual_seed(0)
    n = 2 * h * d + 3 * d
    slab = torch.randn(t, n, generator=gen, device="cuda", dtype=torch.float32).to(dtype)
    src_a = torch.as_strided(slab, (t, h, d), (n, d, 1), 0)
    src_b = torch.as_strided(slab, (t, h, d), (n, d, 1), h * d) if two else None
    n_slots = n_pages * page_size
    slot = _slots(t, n_slots, gen, slot_dtype, pad_every=7, device="cuda")
    slot[t - 1] = n_slots + 5  # past the pool: never a write past it
    if slot_dtype == torch.int64:
        slot[5] = (1 << 33) + 2  # the low word alone would alias slot 2
    pool_a, pool_b = _pools(n_pages, h, page_size, d, layout, dtype, "cuda")
    if not two:
        pool_b = None
    r = compile_cache_write(dtype=dtype, h=h, d=d, page_size=page_size, slot_dtype=slot_dtype, two_operands=two)
    run_cache_write(r, src_a, src_b, pool_a, pool_b, slot, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    live = (slot >= 0) & (slot < n_slots)
    assert int(live.sum()) < t  # the dead slots are exercised
    for src, pool in ((src_a, pool_a), (src_b, pool_b)):
        if src is None:
            continue
        ref = _scatter_ref(src, slot, pool, page_size)
        assert torch.equal(_bits(pool), _bits(ref))
        assert _written_rows(pool) == int(live.sum()) * h  # exactly the live rows, nothing else


@pytest.mark.L0
@_GPU
def test_kernel_runner_refuses_a_mismatched_binding():
    from cudnn.gated_attention_block.kernels.cache_write import compile_cache_write, run_cache_write

    r = compile_cache_write(dtype=torch.bfloat16, h=2, d=256, page_size=16, slot_dtype=torch.int32, two_operands=True)
    src = torch.zeros(8, 2, 256, device="cuda", dtype=torch.bfloat16)
    pool, _ = _pools(2, 2, 16, 256, "hnd", device="cuda")
    slot = torch.arange(8, device="cuda", dtype=torch.int32)
    s = torch.cuda.current_stream().cuda_stream
    with pytest.raises(ValueError, match="two operands"):
        run_cache_write(r, src, None, pool, None, slot, stream=s)
    with pytest.raises(ValueError, match="slot dtype is fixed"):
        run_cache_write(r, src, src, pool, pool, slot.to(torch.int64), stream=s)
    with pytest.raises(ValueError, match="dtype is fixed"):
        run_cache_write(r, src.float(), src, pool, pool, slot, stream=s)
    with pytest.raises(ValueError, match="H and D are fixed"):
        run_cache_write(r, src[:, :1], src, pool, pool, slot, stream=s)
    with pytest.raises(ValueError, match="same T"):
        run_cache_write(r, src, src[:4], pool, pool, slot, stream=s)
    with pytest.raises(ValueError, match="page_size=16"):
        run_cache_write(r, src, src, pool[:, :, :8], pool, slot, stream=s)
    with pytest.raises(ValueError, match="slot_mapping must be 1-D"):
        run_cache_write(r, src, src, pool, pool, slot[:4], stream=s)
    # The typed stride rule at the runner too: a unified row of 2D + 4 elements (1032 B) is refused on either pool.
    odd = torch.full((2, 2, 16, 2 * 256 + 4), float("nan"), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"pool_b stride\[2\].*multiple of 16 bytes"):
        run_cache_write(r, src, src, pool, odd[..., 256:512], slot, stream=s)
    with pytest.raises(ValueError, match="page_size must be >= 1"):
        compile_cache_write(dtype=torch.bfloat16, h=2, d=256, page_size=0)
    with pytest.raises(ValueError, match="bf16 / f16"):
        compile_cache_write(dtype=torch.float32, h=2, d=256, page_size=16)
    with pytest.raises(ValueError, match="int32 or int64"):
        compile_cache_write(dtype=torch.bfloat16, h=2, d=256, page_size=16, slot_dtype=torch.int16)


@pytest.mark.L0
@_GPU
def test_the_stage_writes_the_slab_bands_and_the_raw_key_pool():
    """The block's own stage object on a synthetic slab (the block's SDPA stage gates the arch, the stage itself runs
    anywhere): K / V bands -> pools bitwise (one unified content row for both, one NHD / HND pair); on the band geometry
    the raw indexer key -> its 3-D pool bitwise; a second execute with another mapping re-places every row."""
    gen = torch.Generator(device="cuda").manual_seed(1)
    for geom, label in (
        (GatedAttentionBlockGeometry(**_SMALL_D256), "dense"),
        (GatedAttentionBlockGeometry(**_FLASH_NEXT, qsa=QsaSpec(index_band=True)), "band"),
    ):
        t, ps, n_pages = 192, 16, 15
        st = _CacheWrite(geom, batch=1, seq_len=t, dtype=torch.bfloat16, page_size=ps)
        st.check_support()
        st.compile()
        assert set(st._recipes) == {torch.int32, torch.int64} and (st._recipes[torch.int32][1] is not None) is geom.index_band
        slab = torch.randn(t, geom.n_qkvg, generator=gen, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        o_k, o_v = geom.qkvg_offsets[2], geom.qkvg_offsets[3]
        ks, vs = _cols(slab, o_k, geom.h_kv, geom.d_head), _cols(slab, o_v, geom.h_kv, geom.d_head)
        kc, vc = _pools(n_pages, geom.h_kv, ps, geom.d_head, "unified" if label == "dense" else "nhd", device="cuda")
        ix = _cols(slab, geom.index_k_raw_offset, 1, geom.qsa.index_head_dim) if geom.index_band else None
        ic = torch.full((n_pages, ps, geom.qsa.index_head_dim), float("nan"), device="cuda", dtype=torch.bfloat16) if geom.index_band else None
        for slot_dtype in (torch.int32, torch.int64):
            slot = _slots(t, n_pages * ps, gen, slot_dtype, pad_every=11, device="cuda")
            for p in (kc, vc) + ((ic,) if ic is not None else ()):
                p.fill_(float("nan"))
            st.execute(ks, vs, kc, vc, slot, index_src=ix, index_cache=ic)
            torch.cuda.synchronize()
            assert torch.equal(_bits(kc), _bits(_scatter_ref(ks, slot, kc, ps))) and torch.equal(_bits(vc), _bits(_scatter_ref(vs, slot, vc, ps))), label
            if ic is not None:
                assert torch.equal(_bits(ic.unsqueeze(1)), _bits(_scatter_ref(ix, slot, ic.unsqueeze(1), ps))), label
                assert torch.equal(_bits(_rows_at(ic.unsqueeze(1), slot, ps)[:, 0]), _bits(index_k_raw_view(slab, geom, 1, t)[0, slot >= 0, 0])), label
        with pytest.raises(ValueError, match="index_src / index_cache"):
            st.execute(ks, vs, kc, vc, slot, index_src=None if geom.index_band else ks[:, :1, :128], index_cache=None if geom.index_band else ic)


# ---------------------------------------------------------------------------
# Rubin: the whole block with write-through -- dense and packed, bf16 and f16, every layout at every page size
# ---------------------------------------------------------------------------

_COMPILED = {}  # (form, dtype, page) -> the compiled block with write-through and everything the cells assert against


def _oracle_rows(form, inp, meta, ref_geom):
    """The per-sequence oracle's post-RoPE K / V as ``[T, H_kv, D]`` rows in token order (the dense oracle at B x S rows; the
    packed one per sequence, concatenated -- an empty sequence contributes no rows)."""
    if form == "dense":
        ref = gated_attention_block_reference(**inp, geom=ref_geom)
        return ref.k.reshape(-1, *ref.k.shape[2:]), ref.v.reshape(-1, *ref.v.shape[2:])
    refs = gated_attention_block_reference_packed(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], ref_geom, meta["lens"]
    )
    ks = [r.k[0] for r in refs if r is not None]
    vs = [r.v[0] for r in refs if r is not None]
    return torch.cat(ks, dim=0), torch.cat(vs, dim=0)


def _compiled(form, dtype, page_size, **blk_kw):
    """ONE compiled block per (form, dtype, page) -- the layout varies only the pools, never the plan -- plus its twin
    WITHOUT write-through executed once on the same inputs (the ``out`` the cells compare bitwise)."""
    key = (form, dtype, page_size, tuple(sorted(blk_kw.items())))
    if key in _COMPILED:
        return _COMPILED[key]
    if form == "dense":
        blk, inp, out, ref_geom = _make_block(_SMALL_D256, *_DENSE_BS, dtype=dtype, paged_kv_page_size=page_size, **blk_kw)
        meta, seq_lens = None, None
        plain = GatedAttentionBlockFwd(
            inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], torch.empty_like(out), blk.geom, **blk_kw
        )
    else:
        blk, inp, out, meta = _make_packed_block(_SMALL_D256, _PACKED_LENS, dtype=dtype, paged_kv_page_size=page_size, **blk_kw)
        ref_geom = RefGeometry(**_SMALL_D256)
        seq_lens = meta["seq_lens"]
        plain = GatedAttentionBlockFwd(
            inp["h"],
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"],
            torch.empty_like(out),
            blk.geom,
            thd=True,
            num_sequences=meta["b"],
            max_seq_len=meta["max_seq_len"],
            **blk_kw,
        )
    for b in (blk, plain):
        b.check_support()
        b.compile()
    assert blk.get_workspace_size() == plain.get_workspace_size()
    ws = torch.full((blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device="cuda")
    ws_plain = torch.full((plain.get_workspace_size(),), 0x7F, dtype=torch.uint8, device="cuda")
    out_plain = torch.empty_like(out)
    _exec(plain, inp, out_plain, ws_plain, seq_lens=seq_lens)
    torch.cuda.synchronize()
    ref_k, ref_v = _oracle_rows(form, inp, meta, ref_geom)
    rec = dict(blk=blk, inp=inp, out=out, meta=meta, seq_lens=seq_lens, ws=ws, out_plain=out_plain, ref_k=ref_k, ref_v=ref_v, plain=plain)
    _COMPILED[key] = rec
    return rec


def _slab_rows(rec):
    """The post-RoPE K / V rows attention consumed, as ``[T, H_kv, D]`` views of the workspace the block just ran with: the
    slab's bands in place (the default pipeline and the fused norm + RoPE one), the compact buffers out of place."""
    blk, ws = rec["blk"], rec["ws"]
    g, t, dt = blk.geom, blk.batch * blk.seq_len, blk.act_dtype
    lay = blk._layout()
    if not blk.inplace_qkv:
        return _view(ws, lay.k, (t, g.h_kv, g.d_head), dt), _view(ws, lay.v, (t, g.h_kv, g.d_head), dt)
    proj = _view(ws, lay.proj, (t, g.n_qkvg), dt)
    return _cols(proj, g.qkvg_offsets[2], g.h_kv, g.d_head), _cols(proj, g.qkvg_offsets[3], g.h_kv, g.d_head)


def _check_pools_against(k_cache, v_cache, slot, page_size, k_rows, v_rows, ref_k, ref_v, label, *, k_oracle=True):
    """The live pool rows == the block's own K / V rows bitwise; == the oracle's post-RoPE K / V within the suite's bars
    (V at the band bar; K -- a normed band -- at the band bar's relative term composed with the norm suite's absolute
    term for the pool's dtype, the two roundings the UNFUSED stages make: the GEMM's output rounded to the dtype, then the
    fp32 norm + RoPE rounded once);
    every unwritten slot still NaN.  ``k_oracle=False`` skips the K-vs-oracle bar: the fused norm + RoPE fork norms the
    fp32 ACCUMULATOR, a different rounding structure the caller checks against its own faithful oracle."""
    live = slot >= 0
    pk, pv = _rows_at(k_cache, slot, page_size), _rows_at(v_cache, slot, page_size)
    assert torch.equal(_bits(pk), _bits(k_rows[live])), f"{label}: the K pool is not the K attention consumed"
    assert torch.equal(_bits(pv), _bits(v_rows[live])), f"{label}: the V pool is not the V attention consumed"
    n_live = int(live.sum())
    assert 0 < n_live < int(live.numel())  # live rows AND padded tokens are exercised
    for pool in (k_cache, v_cache):
        written = _written_rows(pool)
        assert written == n_live * pool.shape[1], f"{label}: {written} written rows, expected {n_live * pool.shape[1]}"
    ref_k, ref_v = ref_k[live], ref_v[live]
    err_k = (pk.float() - ref_k.float()).abs().max().item()
    err_v = (pv.float() - ref_v.float()).abs().max().item()
    print(
        f"{label}: {n_live} live rows; pool K vs oracle max|err| {err_k:.3e} (|K| max {ref_k.float().abs().max().item():.3f}); pool V vs oracle max|err| {err_v:.3e}"
    )
    torch.testing.assert_close(pv.float(), ref_v.float(), **_BAND_BAR)
    if k_oracle:
        torch.testing.assert_close(pk.float(), ref_k.float(), rtol=_BAND_BAR["rtol"], atol=_NORM_ATOL[pk.dtype])


@requires_rubin
@pytest.mark.parametrize(
    "form, dtype, layout, page_size",
    [
        pytest.param(
            form,
            dtype,
            layout,
            page_size,
            marks=pytest.mark.L0 if dtype == torch.bfloat16 and layout == "hnd" and page_size == 16 else pytest.mark.L1,
            id=f"{form}-{dtype_id}-{layout}-{page_size}",
        )
        for form in ["dense", "packed"]
        for dtype, dtype_id in [(torch.bfloat16, "bf16"), (torch.float16, "f16")]
        for layout in _LAYOUTS
        for page_size in _PAGES
    ],
)
def test_write_through_matches_the_slab_and_the_oracle_on_every_layout_and_page(form, dtype, layout, page_size):
    """The 8 / 2 d256 geometry at T = 512 (dense B=2 x S=256; packed sequences 300 / 0 / 128 / 84), the default inference
    pipeline (in-place Q / K / V), on NaN-poisoned pools of the given layout and page size with every 9th token padded
    (-1): the pools hold the oracle's post-RoPE K / V at every live slot and are bitwise the slab bands attention consumed;
    padded tokens write nothing (every unwritten slot keeps NaN); ``out`` is bitwise the block's without write-through; the
    workspace size is unchanged; the slot dtype alternates with the page size (int32 at 16 / 64, int64 at 32 / 128)."""
    rec = _compiled(form, dtype, page_size)
    blk, inp, out, seq_lens, ws = rec["blk"], rec["inp"], rec["out"], rec["seq_lens"], rec["ws"]
    g, t = blk.geom, blk.batch * blk.seq_len
    n_pages = -(-t // page_size) + 3
    gen = torch.Generator(device="cuda").manual_seed(7 + page_size)
    slot_dtype = torch.int32 if page_size in (16, 64) else torch.int64
    slot = _slots(t, n_pages * page_size, gen, slot_dtype, pad_every=9, device="cuda")
    k_cache, v_cache = _pools(n_pages, g.h_kv, page_size, g.d_head, layout, dtype, "cuda")
    _exec(blk, inp, out, ws, seq_lens=seq_lens, k_cache=k_cache, v_cache=v_cache, slot_mapping=slot)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert torch.equal(_bits(out), _bits(rec["out_plain"])), "write-through changed the block's output"
    k_rows, v_rows = _slab_rows(rec)
    label = f"8/2 {form} T={t} p{page_size} {layout} {slot_dtype} {dtype}"
    _check_pools_against(k_cache, v_cache, slot, page_size, k_rows, v_rows, rec["ref_k"], rec["ref_v"], label)
    assert blk._stages.index(blk._cache_write) == rec["plain"]._stages.index(rec["plain"]._sdpa)  # the one extra stage sits where the SDPA was


@pytest.mark.L1
@requires_rubin
@pytest.mark.parametrize("pipeline", ["fuse_norm_rope", "out_of_place", "fuse_gate", "packed_fuse_norm_rope", "packed_out_of_place", "packed_fuse_gate"])
def test_write_through_on_the_other_bf16_pipelines(pipeline):
    """Fused norm + RoPE (the slab is written post-RoPE by the GEMM fork, which norms the fp32 ACCUMULATOR -- so the
    pool's K is checked against the oracle norm + RoPE of the fp32 projection rounded ONCE, where the only legal
    difference is a one-ulp flip from the two accumulation orders: ``rtol 2^-7`` = one bf16 ulp at the bottom of a binade,
    ``atol 1e-4`` = the fp32 noise floor on near-zero values), out-of-place Q / K / V (the compact buffers are the SDPA's
    operands -- and the slab keeps the PRE-norm K, so the oracle norm of the block's OWN pre-norm K is checked at the norm
    suite's exact bar), and the fused gate (K / V untouched by it), in dense and packed forms:
    pools bitwise the operands attention consumed, ``out`` bitwise the block's without write-through; then a second
    execute with another mapping on the SAME compiled block re-places every row."""
    form = "packed" if pipeline.startswith("packed_") else "dense"
    kw = {"fuse_norm_rope": dict(fuse_norm_rope=True), "out_of_place": dict(inplace_qkv=False), "fuse_gate": dict(fuse_gate=True)}[
        pipeline.replace("packed_", "")
    ]
    page_size, layout = 16, "unified"
    rec = _compiled(form, torch.bfloat16, page_size, **kw)
    blk, inp, out, seq_lens, ws = rec["blk"], rec["inp"], rec["out"], rec["seq_lens"], rec["ws"]
    g, t = blk.geom, blk.batch * blk.seq_len
    n_pages = -(-t // page_size) + 3
    gen = torch.Generator(device="cuda").manual_seed(3)
    slot = _slots(t, n_pages * page_size, gen, torch.int32, pad_every=9, device="cuda")
    k_cache, v_cache = _pools(n_pages, g.h_kv, page_size, g.d_head, layout, torch.bfloat16, "cuda")
    _exec(blk, inp, out, ws, seq_lens=seq_lens, k_cache=k_cache, v_cache=v_cache, slot_mapping=slot)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all() and torch.equal(_bits(out), _bits(rec["out_plain"]))
    k_rows, v_rows = _slab_rows(rec)
    lay = blk._layout()
    proj = _view(ws, lay.proj, (t, g.n_qkvg), torch.bfloat16)
    cos, sin = inp["cos"].reshape(1, t, -1), inp["sin"].reshape(1, t, -1)  # per-token tables (the packed form's are per token already)
    if pipeline.endswith("out_of_place"):
        k_pre = _cols(proj, g.qkvg_offsets[2], g.h_kv, g.d_head).reshape(1, t, g.h_kv, g.d_head)
        want_k, _ = qk_norm_rope_reference(k_pre, inp["w_k_norm"], cos, sin, g.rope_dim, g.qk_norm_eps)
        pk = _rows_at(k_cache, slot, page_size)
        torch.testing.assert_close(pk.float(), want_k.reshape(t, g.h_kv, g.d_head)[slot >= 0].float(), rtol=0.0, atol=_NORM_ATOL[pk.dtype])
    if pipeline.endswith("fuse_norm_rope"):
        # The fork-faithful oracle: norm + RoPE of the UNROUNDED fp32 projection, rounded once to bf16.
        o_k = g.qkvg_offsets[2]
        k_pre32 = (inp["h"].reshape(t, g.d_model).float() @ inp["w_qkvg"].float().t())[:, o_k : o_k + g.h_kv * g.d_head].reshape(1, t, g.h_kv, g.d_head)
        want_k32, _ = qk_norm_rope_reference(k_pre32, inp["w_k_norm"].float(), cos.float(), sin.float(), g.rope_dim, g.qk_norm_eps)
        want_k = want_k32.to(torch.bfloat16).reshape(t, g.h_kv, g.d_head)[slot >= 0]
        pk = _rows_at(k_cache, slot, page_size)
        err = (pk.float() - want_k.float()).abs().max().item()
        flips = (pk != want_k).sum().item()
        print(f"8/2 {pipeline}: pool K vs the fp32-accumulator oracle rounded once: max|err| {err:.3e}, {flips} / {pk.numel()} elements differ (one-ulp flips)")
        torch.testing.assert_close(pk.float(), want_k.float(), rtol=2.0**-7, atol=1e-4)
    _check_pools_against(
        k_cache, v_cache, slot, page_size, k_rows, v_rows, rec["ref_k"], rec["ref_v"], f"8/2 {pipeline}", k_oracle=not pipeline.endswith("fuse_norm_rope")
    )
    # A second execute with another mapping on the SAME compiled block re-places every row (the plan is reused, Rule 4).
    gen = torch.Generator(device="cuda").manual_seed(11)
    slot2 = _slots(t, k_cache.shape[0] * page_size, gen, torch.int64, device="cuda")
    for p in (k_cache, v_cache):
        p.fill_(float("nan"))
    _exec(blk, inp, out, ws, seq_lens=seq_lens, k_cache=k_cache, v_cache=v_cache, slot_mapping=slot2)
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(rec["out_plain"]))
    k_rows, v_rows = _slab_rows(rec)
    assert torch.equal(_bits(_rows_at(k_cache, slot2, page_size)), _bits(k_rows)) and torch.equal(_bits(_rows_at(v_cache, slot2, page_size)), _bits(v_rows))


@pytest.mark.L1
@requires_rubin
def test_two_launches_are_bitwise_on_the_pools_and_the_output():
    """The same packed block twice on fresh NaN pools: pools and ``out`` bitwise equal across the two launches (a
    first-launch race would show here first)."""
    rec = _compiled("packed", torch.bfloat16, 32)
    blk, inp, out, seq_lens, ws = rec["blk"], rec["inp"], rec["out"], rec["seq_lens"], rec["ws"]
    g, t = blk.geom, blk.batch * blk.seq_len
    n_pages = -(-t // 32) + 2
    slot = _slots(t, n_pages * 32, torch.Generator(device="cuda").manual_seed(5), torch.int32, pad_every=13, device="cuda")
    runs = []
    for _ in range(2):
        k_cache, v_cache = _pools(n_pages, g.h_kv, 32, g.d_head, "nhd", torch.bfloat16, "cuda")
        _exec(blk, inp, out, ws, seq_lens=seq_lens, k_cache=k_cache, v_cache=v_cache, slot_mapping=slot)
        torch.cuda.synchronize()
        runs.append((_bits(k_cache).clone(), _bits(v_cache).clone(), _bits(out).clone()))
    for a, b in zip(runs[0], runs[1]):
        assert torch.equal(a, b)


@pytest.mark.L1
@requires_rubin
def test_raw_indexer_key_pool_at_flash_next():
    """The band geometry's write-through stages -- the unfused projection over the five-band slab, norm + RoPE, the cache
    write -- are the stages AHEAD of the sparse attention core in the block's walk (``check_support`` accepts them and
    then declines at the sparse stage, which has not landed), so they run through the block's own stage objects: the raw
    indexer key's pool is bitwise ``index_k_raw_view`` at every slot, the K / V pools bitwise the post-RoPE bands, and the
    pre-norm raw key is untouched by norm + RoPE."""
    geom = GatedAttentionBlockGeometry(**_FLASH_NEXT, qsa=QsaSpec(index_band=True))
    b, s = 1, 512
    t, dm, ps = b * s, geom.d_model, 16
    gen = torch.Generator(device="cuda").manual_seed(0)
    rnd = lambda *shape, std=0.02: (torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32) * std).to(torch.bfloat16)  # noqa: E731
    h = rnd(b, s, dm, std=1.0)
    w_qkvg = rnd(geom.n_qkvg, dm)
    cos, sin = build_rope_tables(s, geom.rope_dim, batch=b, device="cuda", dtype=torch.bfloat16)
    wn = torch.ones(geom.d_head, device="cuda", dtype=torch.bfloat16)
    w_o = rnd(dm, geom.h_q * geom.d_head)
    out = torch.empty(b, s, dm, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream().cuda_stream
    blk = GatedAttentionBlockFwd(h, w_qkvg, wn, wn, cos, sin, w_o, out, geom, paged_kv_page_size=ps)
    assert isinstance(blk._sdpa, _SparseSdpa) and blk._cache_write.index_band
    stages = list(blk._stages)
    assert stages.index(blk._cache_write) < stages.index(blk._sdpa)
    with pytest.raises(NotImplementedError, match="sparse attention core"):
        blk.check_support()  # the stages ahead of the sparse core -- the write-through included -- accepted the band geometry
    for st in (blk._proj, blk._norm_rope, blk._cache_write):
        st.compile()
    lay = blk._layout()
    ws = torch.full((lay.total_bytes + _align_up(max(blk._proj.workspace_bytes(), 1)),), 0x7F, dtype=torch.uint8, device="cuda")
    proj = _view(ws, lay.proj, (t, geom.n_qkvg), torch.bfloat16)
    blk._proj.execute(h.view(t, dm), w_qkvg, proj, ws[lay.engine_scratch :], stream=stream)
    o_q, _o_g, o_k, o_v = geom.qkvg_offsets[:4]
    raw_before = index_k_raw_view(proj, geom, b, s).clone()
    blk._norm_rope.execute(
        _cols(proj, o_q, geom.h_q, geom.d_head), _cols(proj, o_k, geom.h_kv, geom.d_head), wn, wn, cos, sin, current_stream=stream, flat=True
    )
    n_pages = t // ps + 2
    slot = _slots(t, n_pages * ps, gen, torch.int32, pad_every=13, device="cuda")
    k_cache, v_cache = _pools(n_pages, geom.h_kv, ps, geom.d_head, "unified", device="cuda")
    index_cache = torch.full((n_pages, ps, geom.qsa.index_head_dim), float("nan"), device="cuda", dtype=torch.bfloat16)
    k_rows, v_rows = _cols(proj, o_k, geom.h_kv, geom.d_head), _cols(proj, o_v, geom.h_kv, geom.d_head)
    ix = _cols(proj, geom.index_k_raw_offset, geom.qsa.index_kv_heads, geom.qsa.index_head_dim)
    blk._cache_write.execute(k_rows, v_rows, k_cache, v_cache, slot, index_src=ix, index_cache=index_cache, current_stream=stream)
    torch.cuda.synchronize()
    live = slot >= 0
    assert torch.equal(_bits(_rows_at(k_cache, slot, ps)), _bits(k_rows[live])) and torch.equal(_bits(_rows_at(v_cache, slot, ps)), _bits(v_rows[live]))
    raw = index_k_raw_view(proj, geom, b, s)
    assert torch.equal(_bits(raw), _bits(raw_before)), "norm + RoPE must not touch the raw indexer key"
    pooled = _rows_at(index_cache.unsqueeze(1), slot, ps)[:, 0]
    assert torch.equal(_bits(pooled), _bits(raw[0, live, 0])), "the raw-key pool is not the slab's raw indexer key"
    assert _written_rows(index_cache) == int(live.sum())
    assert blk._cache_write.moved_bytes() == 2 * 2 * t * geom.h_kv * geom.d_head * 2 + t * 4 + 2 * t * 128 * 2 + t * 4


# ===========================================================================
# The EXTEND form: kv_source="paged" on a packed, bounded declaration -- attention OVER the pools
# ===========================================================================

_EXTEND_GEOM = dict(_SMALL_D256, causal_bottom_right=True)  # the 8 / 2 d256 geometry on the extend (bottom-right) diagonal
_STALE = 777.0  # every unaddressed pool row of a READ cell holds this FINITE value: a serving pool is zero-initialised and then holds stale rows, never NaN
_NULL_PAGE = 0  # the serving stacks' NULL block: a padded entry's table row points here; the kernel reads page -1 (TMA OOB zeros) past a sequence's live pages
_OUT_SENTINEL = {torch.bfloat16: 1.5e30, torch.float16: 6.0e4}  # a finite magnitude no correct output cell holds; a survivor is an unwritten / untouched cell
_MATRIX_PACKING = dict(q_lens=[128, 84, 0, 200], prefix_lens=[0, 300, 0, 77])
# T_live 412 of 512 (pads at the END); kv_lens 128 / 384 / 0 / 277: n_kv = 1 (one KV tile; B x H_kv > 1 with it), a 300-row prefix ending
# mid-page at every page size with kv_len a 128-multiple, a PADDED entry (kv_len 0, the NULL block, no rows), a 77-row prefix (mid-page)
# with kv_len 277 (a tail tile, n_kv = 3)
_SECOND_PACKING = dict(q_lens=[100, 80, 0, 120], prefix_lens=[128, 384, 0, 277])  # the next chunk of the same sequences: T_live 300


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _kernel_source(module_name):
    """The SOURCE TEXT of a kernel module -- read from its file, never imported (a kernel module builds its config and traces at import)."""
    import importlib.util

    return open(importlib.util.find_spec(module_name).origin).read()


def _extend_samples(page, n_pages, b_max, max_pages, layout="hnd", dtype=torch.bfloat16, row_stride=None, h_kv=2, d=256, device=_DEV):
    """The EXTEND form's three declaration samples -- and, in the cells, the LIVE tensors (a serving stack hands the block views of
    its one cache): two ``_STALE``-filled pools with the NULL page zeroed and a ``[B_max, max_pages]`` int32 block table -- a
    ``[:, :max_pages]`` row slice of a wider ``[B_max, row_stride]`` table when ``row_stride`` is given: the serving stacks slice their tables)."""
    k, v = _pools(n_pages, h_kv, page, d, layout, dtype, device, fill=_STALE)
    k[_NULL_PAGE].zero_()
    v[_NULL_PAGE].zero_()
    table = torch.zeros(b_max, max_pages if row_stride is None else row_stride, device=device, dtype=torch.int32)
    return dict(sample_k_cache=k, sample_v_cache=v, sample_block_table=table[:, :max_pages])


def _declare_extend_with(
    b_max, t_max, samples, *, page=16, kv_source=KV_SOURCE_PAGED, cu=True, token_capacity="T", geom_kw=_EXTEND_GEOM, dtype=torch.bfloat16, seed=0, **blk_kw
):
    """A declared (not compiled) EXTEND block over EXPLICIT samples (the pools' shape / dtype / strides, the block table).  ``inp``
    carries the WEIGHTS every step shares and the step's token tensors (``h`` / ``cos`` / ``sin`` as ``[T_max, .]`` matrices the
    steps rewrite in place).  Returns ``(blk, inp, out)``."""
    geom = GatedAttentionBlockGeometry(**geom_kw)
    ref_geom = RefGeometry(
        **{k: v for k, v in geom_kw.items() if k in ("d_model", "h_q", "h_kv", "d_head", "rope_dim")}
    )  # the dims; the oracle's mask is its own
    inp = make_inputs(ref_geom, batch=1, seq_len=t_max, dtype=dtype, device=_DEV, seed=seed)
    inp = dict(inp, h=inp["h"].view(t_max, geom.d_model), cos=inp["cos"].view(t_max, -1).contiguous(), sin=inp["sin"].view(t_max, -1).contiguous())
    out = torch.empty(t_max, geom.d_model, device=_DEV, dtype=dtype)
    blk = GatedAttentionBlockFwd(
        inp["h"],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        out,
        geom,
        thd=True,
        num_sequences=b_max,
        max_seq_len=t_max,
        cu_seqlens=cu,
        token_capacity=t_max if token_capacity == "T" else token_capacity,
        paged_kv_page_size=page,
        kv_source=kv_source,
        **samples,
        **blk_kw,
    )
    return blk, inp, out


def _declare_extend(b_max, t_max, *, max_kv_len, page=16, layout="hnd", dtype=torch.bfloat16, row_stride=None, n_pages=None, geom_kw=_EXTEND_GEOM, **kw):
    """A declared (not compiled) EXTEND block: a packed declaration at the bounded capacity ``t_max`` for ``b_max`` sequences whose
    logical KV length may reach ``max_kv_len`` (``max_pages = ceil(max_kv_len / page)`` table columns; pools of ``n_pages`` pages,
    default one full allotment per sequence plus the NULL page).  Returns ``(blk, inp, out, samples)``."""
    geom = GatedAttentionBlockGeometry(**geom_kw)
    max_pages = -(-max_kv_len // page)
    n_pages = n_pages if n_pages is not None else b_max * max_pages + 1
    samples = _extend_samples(page, n_pages, b_max, max_pages, layout, dtype, row_stride, geom.h_kv, geom.d_head, _DEV)
    blk, inp, out = _declare_extend_with(b_max, t_max, samples, page=page, geom_kw=geom_kw, dtype=dtype, **kw)
    return blk, inp, out, samples


def _declaration_facts(blk, samples):
    """What an EXTEND declaration must have recorded: the attribute, the paged SDPA stage with the samples' descriptors, the
    capture-safe route on both dense projections, the write-through stage in position."""
    assert blk.kv_source == KV_SOURCE_PAGED and blk.thd and blk.token_capacity == blk.batch * blk.seq_len
    sd = blk._sdpa
    assert isinstance(sd, _Sdpa) and sd.kv_source == KV_SOURCE_PAGED and sd.paged_page_size == blk.paged_kv_page_size
    for nm in ("k_cache", "v_cache", "block_table"):
        d = blk._descs[nm]
        assert tuple(d.shape) == tuple(samples[f"sample_{nm}"].shape) and tuple(d.stride) == tuple(samples[f"sample_{nm}"].stride()), nm
    assert sd.k_cache_desc is blk._descs["k_cache"] and sd.v_cache_desc is blk._descs["v_cache"] and sd.block_table_desc is blk._descs["block_table"]
    for st in (blk._proj, blk._out_proj):
        if isinstance(st, _Projection):
            assert st.capture_safe is True
    if blk._norm_rope is not None:
        _assert_stage_in_pipeline_position(blk, blk.paged_kv_page_size)
    else:  # fuse_norm_rope: the fork writes the post-RoPE slab; the write-through sits right after it, before the paged read
        stages = list(blk._stages)
        assert stages.index(blk._proj) < stages.index(blk._cache_write) < stages.index(blk._sdpa)


def _slot_table(lay, geom, t, act_dtype):
    """``{slot: (offset, nbytes)}`` of every RESERVED workspace slot of a carve (the host assertion of the bounded-token contract
    reads it): a slot may never overlap another, and ``o`` in particular is aliased by no other stage's scratch."""
    e = torch.empty((), dtype=act_dtype).element_size()
    sizes = dict(
        proj=t * geom.n_qkvg * e,
        q=t * geom.h_q * geom.d_head * e,
        k=t * geom.h_kv * geom.d_head * e,
        v=t * geom.h_kv * geom.d_head * e,
        o=t * geom.h_q * geom.d_head * e,
        o_gated=t * geom.h_q * geom.d_head * e,
        q8=t * geom.h_q * geom.d_head,
        k8=t * geom.h_kv * geom.d_head,
        v8=t * geom.h_kv * geom.d_head,
        o8=t * geom.h_q * geom.d_head,
        gate16=t * geom.h_q * geom.d_head * e,
    )
    table = {name: (getattr(lay, name), nbytes) for name, nbytes in sizes.items() if getattr(lay, name) >= 0}
    table["engine_scratch"] = (lay.engine_scratch, None)
    return table


# ---------------------------------------------------------------------------
# Declaration rows, the envelope, the stage facts -- host, any device
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_extend_declaration_builds_the_paged_sdpa_stage_and_the_capture_safe_projections():
    blk, inp, out, samples = _declare_extend(4, 512, max_kv_len=512, page=16)
    _declaration_facts(blk, samples)
    assert blk.num_sequences == 4 and blk.max_seq_len == 512 and blk.cu_seqlens
    # lengths form, dtype, layout, page, a row-sliced table, the other bf16 pipelines, fp16: every served spelling builds the same stage
    for kw in (
        dict(cu=False),
        dict(dtype=torch.float16),
        dict(layout="nhd"),
        dict(layout="unified"),
        dict(page=128),
        dict(row_stride=70),
        dict(fuse_norm_rope=True),
        dict(inplace_qkv=False),
    ):
        blk2, _, _, samples2 = _declare_extend(3, 256, max_kv_len=1024, **kw)
        _declaration_facts(blk2, samples2)
    # the carve is the plain THD block's (the pools, the table and the lengths are the caller's; the paged read reserves nothing)
    plain, *_ = _make_packed_block(_SMALL_D256, (200, 0, 184, 128))
    assert blk._layout() == plain._layout()


@pytest.mark.L0
def test_extend_o_slot_is_aliased_by_no_other_stage_scratch():
    """The bounded-token contract (pad rows of ``out`` exactly 0 over a once-zeroed workspace) rests on the ``o`` slot being written by
    the SDPA alone: ``o_gated`` IS ``o`` under inference (the gate's own in-place write), and no other reserved slot -- nor the
    engines' scratch -- overlaps a byte of it.  Asserted on the carve itself, so a future slot sharing flips this RED instead of
    turning pad rows into garbage."""
    for kw in (dict(), dict(inplace_qkv=False), dict(fuse_norm_rope=True)):
        blk, *_ = _declare_extend(4, 512, max_kv_len=512, **kw)
        lay = blk._layout()
        assert lay.o >= 0 and lay.o_gated == -1, "inference: o_gated aliases o in place; o is reserved"
        table = _slot_table(lay, blk.geom, blk.batch * blk.seq_len, blk.act_dtype)
        o_lo, o_n = table["o"]
        o_hi = o_lo + o_n
        for name, (lo, n) in table.items():
            if name == "o":
                continue
            hi = lo + n if n is not None else lay.total_bytes + 1  # the engine scratch starts at engine_scratch and runs to the end
            assert hi <= o_lo or lo >= o_hi, f"{name} [{lo}, {hi}) overlaps o [{o_lo}, {o_hi})"
        assert lay.engine_scratch >= o_hi


def _declare_dense_sparse_paged():
    """A DENSE paged declaration on a QsaSpec geometry: the serving table's `sparse` row (the packed form reaches the same row -- the
    block serves the packed sparse stage, so both paged forms meet the table's `sparse` row)."""
    geom = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec())
    mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
    return GatedAttentionBlockFwd(
        mk(1, 16, geom.d_model),  # S <= 32: the decode_rows row sits before the sparse row
        mk(geom.n_qkvg, geom.d_model),
        mk(geom.d_head),
        mk(geom.d_head),
        mk(1, 16, geom.rope_dim),
        mk(1, 16, geom.rope_dim),
        mk(geom.d_model, geom.h_q * geom.d_head),
        mk(1, 16, geom.d_model),  # S <= 32: the decode_rows row sits before the sparse row
        geom,
        paged_kv_page_size=16,
        kv_source=KV_SOURCE_PAGED,
    )


@pytest.mark.L0
def test_extend_typed_declines_match_the_envelope_row_by_row(monkeypatch):
    """Every row of the EXTEND declines table is reachable at declaration with exactly the message the envelope publishes for it --
    the CPU proof that no declaration-time decline escapes ``serving_envelope``."""
    import re
    from cudnn.frost import buffers

    env = GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**_EXTEND_GEOM), torch.bfloat16, _DEV)
    assert isinstance(env, ServingEnvelope) and isinstance(env.extend, ExtendEnvelope) and isinstance(env.decode, DecodeEnvelope)
    names = [name for name, *_ in _EXTEND_DECLINES]
    assert set(env.extend.declines) == set(names) and len(names) == len(set(names))
    assert env.extend.requires and env.extend.page_sizes == SERVING_PAGE_SIZES == (16, 32, 64, 128) and env.extend.pool_layouts == SERVING_POOL_LAYOUTS
    assert env.extend.inference_only and not env.extend.writes_lse and env.extend.dtypes == (torch.bfloat16, torch.float16)
    good = _extend_samples(16, 9, 2, 8)
    d128 = dict(_EXTEND_GEOM, d_head=128, d_model=256)
    trips = {
        "write_through": lambda: _declare_extend_with(2, 128, good, page=0),
        # a dense paged declaration IS the DECODE form now (test_block_serving_decode.py); past the decode tile's 32 rows it is this row
        "decode_rows": lambda: _make_block(_EXTEND_GEOM, 1, 64, paged_kv_page_size=16, kv_source=KV_SOURCE_PAGED),
        "diagonal": lambda: _declare_extend(2, 128, max_kv_len=128, geom_kw=_SMALL_D256),
        "sparse": _declare_dense_sparse_paged,
        "window": lambda: _declare_extend(2, 128, max_kv_len=128, geom_kw=dict(_EXTEND_GEOM, window_left=64)),
        "head_dim": lambda: _declare_extend(2, 128, max_kv_len=128, geom_kw=d128),
        "fuse_gate": lambda: _declare_extend(2, 128, max_kv_len=128, fuse_gate=True),
        "return_lse": lambda: _declare_extend(2, 128, max_kv_len=128, return_lse=True),
        "token_capacity_dense": lambda: _make_block(_SMALL_D256, 1, 64, token_capacity=64),
        "token_capacity_training": lambda: _make_packed_block(
            _SMALL_D256, _PACKED_LENS, token_capacity=sum(_PACKED_LENS), save_for_backward=True, inplace_qkv=False
        ),
    }
    assert set(trips) == set(names)
    for name, exc, _pred, msg in _EXTEND_DECLINES:
        with pytest.raises(exc, match=re.escape(msg[:80])) as e:
            trips[name]()
        assert str(e.value) == msg == env.extend.declines[name], name
    # the served verdict reads the device: Rubin serves the form, every other part names itself in the reason
    cc = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    assert env.device_cc == cc
    if cc == (10, 7):
        if not env.cutedsl_ok:
            assert not env.extend.served and env.forms == () and env.extend.reason == env.cutedsl_reason
        # The remaining assertions inspect geometry on a hypothetical supported target.
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
        env = GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**_EXTEND_GEOM), torch.bfloat16, _DEV)
        assert env.extend.served and env.forms == ("extend", "decode") and env.extend.reason is None and env.cutedsl_ok
        fp32 = GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**_EXTEND_GEOM), torch.float32, _DEV)
        assert not fp32.extend.served and "bf16 / f16" in fp32.extend.reason
        for geom_kw, row in ((dict(_EXTEND_GEOM, window_left=64), "window"), (_SMALL_D256, "diagonal"), (d128, "head_dim")):
            bad = GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**geom_kw), torch.bfloat16, _DEV)
            assert not bad.extend.served and bad.extend.reason == env.extend.declines[row] and bad.forms == (), row
    else:
        assert not env.extend.served and env.forms == () and "targets Rubin" in env.extend.reason
    # a sparse geometry: the block serves the packed sparse stage (the sparse declaration's THD row is record-driven and the record claims
    # the arm), so the THD-only form reaches the table like the dense one and meets its `sparse` row -- the envelope says so wherever the
    # device / dtype facts let it speak (on Rubin: the very message the block raises for that declaration)
    sparse = GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec()), torch.bfloat16, _DEV)
    assert not sparse.extend.served
    with pytest.raises(NotImplementedError, match="kv_source='paged' with geometry.qsa") as e:
        _declare_extend(2, 128, max_kv_len=128, geom_kw=dict(_SMALL_D256, qsa=QsaSpec()))
    assert str(e.value) == env.extend.declines["sparse"]
    if cc == (10, 7):
        assert sparse.extend.reason == str(e.value)
    # ... and the Rubin reading on ANY host (device_cc=): BOTH halves say the table's `sparse` row -- the pin the first node run exposed
    # (the envelope read the `diagonal` row for a QsaSpec geometry on the THD-only form)
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    rubin = GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec()), torch.bfloat16, _DEV, device_cc=(10, 7))
    assert rubin.cutedsl_ok and rubin.forms == ()
    assert rubin.extend.reason == str(e.value) == rubin.decode.reason == env.decode.declines["sparse"]


@pytest.mark.L0
def test_extend_argument_rows_are_typed():
    """The argument contract around the table: ``kv_source`` values, ``token_capacity``'s type / value / equality with the samples'
    rows, the three samples REQUIRED under ``kv_source="paged"`` and REFUSED under ``"self"``, each pool's form (``check_pool``'s
    rules), K and V of one layout kind, the block table's rank / dtype / shape / page-axis stride / device."""
    good = _extend_samples(16, 9, 2, 8)
    with pytest.raises(ValueError, match="kv_source must be one of"):
        _declare_extend_with(2, 128, good, kv_source="pool")
    with pytest.raises(TypeError, match="token_capacity must be an int"):
        _declare_extend_with(2, 128, good, token_capacity=128.0)
    with pytest.raises(TypeError, match="token_capacity must be an int"):
        _declare_extend_with(2, 128, good, token_capacity=True)
    with pytest.raises(ValueError, match="token_capacity must be >= 1"):
        _declare_extend_with(2, 128, good, token_capacity=0)
    with pytest.raises(ValueError, match="token_capacity=100 must equal the samples' packed token rows T=128"):
        _declare_extend_with(2, 128, good, token_capacity=100)
    # token_capacity alone (kv_source="self") is a served THD declaration; None keeps the tight contract on the EXTEND form
    blk, *_ = _make_packed_block(_SMALL_D256, _PACKED_LENS, token_capacity=sum(_PACKED_LENS))
    assert blk.token_capacity == sum(_PACKED_LENS) and blk.kv_source == KV_SOURCE_SELF
    blk, *_ = _declare_extend_with(2, 128, good, token_capacity=None)
    assert blk.token_capacity is None and blk.kv_source == KV_SOURCE_PAGED
    # the samples: refused on "self", required on "paged"
    for kw in (dict(sample_k_cache=good["sample_k_cache"]), dict(sample_block_table=good["sample_block_table"]), good):
        with pytest.raises(ValueError, match="belong to the EXTEND form"):
            _make_packed_block(_SMALL_D256, _PACKED_LENS, paged_kv_page_size=16, **kw)
    for missing in ("sample_k_cache", "sample_v_cache", "sample_block_table"):
        with pytest.raises(ValueError, match=f"needs {missing} at declaration"):
            _declare_extend_with(2, 128, {k2: v2 for k2, v2 in good.items() if k2 != missing})
    # pool forms: every check_pool row names the sample
    for bad, match in _bad_pool_forms(_SMALL_D256):
        with pytest.raises(ValueError, match=f"sample_k_cache.*{match}" if match != "must be a" else match):
            _declare_extend_with(2, 128, dict(good, sample_k_cache=bad))
    # one layout kind for K and V
    k_n, _ = _pools(9, 2, 16, 256, "nhd")
    with pytest.raises(ValueError, match="share one layout kind"):
        _declare_extend_with(2, 128, dict(good, sample_k_cache=k_n))
    # the block table rows
    for bad, match in (
        (torch.zeros(2, 4, device=_DEV, dtype=torch.int64), "2-D int32"),
        (torch.zeros(2, 4, 1, device=_DEV, dtype=torch.int32), "2-D int32"),
        (torch.zeros(3, 4, device=_DEV, dtype=torch.int32), r"num_sequences=2, max_pages >= 1"),
        (torch.zeros(2, 0, device=_DEV, dtype=torch.int32), r"num_sequences=2, max_pages >= 1"),
        (torch.zeros(4, 2, device=_DEV, dtype=torch.int32).t(), "page axis must be contiguous"),
        ("table", "2-D int32"),
    ):
        with pytest.raises(ValueError, match=match):
            _declare_extend_with(2, 128, dict(good, sample_block_table=bad))
    if _DEV == "cuda":
        with pytest.raises(ValueError, match="must live on"):
            _declare_extend_with(2, 128, dict(good, sample_block_table=torch.zeros(2, 4, dtype=torch.int32)))


@pytest.mark.L0
def test_extend_write_through_declines_still_fire_first():
    """``quant`` / ``save_for_backward`` with the attribute are the write-through's typed declines, heard before anything of the
    EXTEND form (the form requires the attribute): no row of the EXTEND table repeats them."""
    good = _extend_samples(16, 9, 2, 8)
    with pytest.raises(NotImplementedError, match="paged_kv_page_size with save_for_backward=True"):
        _declare_extend_with(2, 128, good, save_for_backward=True, inplace_qkv=False)
    assert not any("quant" in name or "save_for_backward" in name for name, *_ in _EXTEND_DECLINES)


@pytest.mark.L0
def test_extend_execute_form_checks_are_host_side_and_typed():
    """On a DECLARED (uncompiled) EXTEND block, under ``set_sync_debug_mode("error")``: ``block_table`` / ``kv_lens`` REQUIRED, the pools
    matching the declared samples exactly (another page count or layout is another plan), the block table's dtype / shape / strides /
    device, ``kv_lens``'s form; a wider table at the declared row stride passes; the well-formed call reaches the plan check."""
    blk, inp, out, samples = _declare_extend(3, 256, max_kv_len=512, page=16, row_stride=50)
    ws = torch.empty(16, dtype=torch.uint8, device=_DEV)
    k, v, bt = samples["sample_k_cache"], samples["sample_v_cache"], samples["sample_block_table"]
    t = blk.batch * blk.seq_len
    slot = torch.full((t,), -1, device=_DEV, dtype=torch.int32)
    kv_lens = torch.zeros(3, device=_DEV, dtype=torch.int32)
    seq_lens = torch.zeros(4, device=_DEV, dtype=torch.int32)
    base = dict(seq_lens=seq_lens, k_cache=k, v_cache=v, slot_mapping=slot, block_table=bt, kv_lens=kv_lens)
    with _no_host_sync():
        for missing in ("block_table", "kv_lens"):
            with pytest.raises(ValueError, match=f"kv_source='paged' needs {missing} at execute"):
                _exec(blk, inp, out, ws, **{kk: vv for kk, vv in base.items() if kk != missing})
        for nm, bad in (("k_cache", _pools(9, 2, 16, 256, "hnd", fill=0.0)[0]), ("v_cache", _pools(k.shape[0], 2, 16, 256, "nhd", fill=0.0)[1])):
            with pytest.raises(ValueError, match=f"{nm} must match sample_{nm} exactly"):
                _exec(blk, inp, out, ws, **{**base, nm: bad})
        for bad, match in (
            (bt.to(torch.int64), "2-D int32"),
            (bt[:2], r"\[num_sequences=3, >= max_pages=32\]"),
            (bt[:, :31], r"\[num_sequences=3, >= max_pages=32\]"),
            (bt.contiguous(), r"declared \(row stride 50, 1\)"),
            (torch.zeros(3, 2 * 32, device=_DEV, dtype=torch.int32)[:, ::2], r"declared \(row stride 50, 1\)"),
        ):
            with pytest.raises(ValueError, match=match):
                _exec(blk, inp, out, ws, **{**base, "block_table": bad})
        for bad, match in (
            (kv_lens.to(torch.int64), "contiguous 1-D int32"),
            (torch.zeros(4, device=_DEV, dtype=torch.int32), "contiguous 1-D int32"),
            (torch.zeros(6, device=_DEV, dtype=torch.int32)[::2], "contiguous 1-D int32"),
            (torch.zeros(3, 1, device=_DEV, dtype=torch.int32), "contiguous 1-D int32"),
        ):
            with pytest.raises(ValueError, match=match):
                _exec(blk, inp, out, ws, **{**base, "kv_lens": bad})
        if _DEV == "cuda":
            with pytest.raises(ValueError, match="must live on"):
                _exec(blk, inp, out, ws, **{**base, "block_table": torch.zeros(3, 50, dtype=torch.int32)[:, :32]})
            with pytest.raises(ValueError, match="contiguous 1-D int32 CUDA"):
                _exec(blk, inp, out, ws, **{**base, "kv_lens": torch.zeros(3, dtype=torch.int32)})
        wide = torch.zeros(3, 50, device=_DEV, dtype=torch.int32)[:, :40]  # >= max_pages at the declared row stride: a view the adapter binds
        with pytest.raises(RuntimeError, match="compile"):
            _exec(blk, inp, out, ws, **{**base, "block_table": wide})
        with pytest.raises(RuntimeError, match="compile"):
            _exec(blk, inp, out, ws, **base)


@pytest.mark.L0
def test_extend_pipelines_launch_without_pdl_and_the_sdpa_bodies_wait_on_no_dependent_grid(monkeypatch):
    """The ONLY ordering between the cache write and the paged read is STREAM order.  Pinned: (1) the write-through stage compiles
    its kernels with ``use_pdl`` False (the runner's default; the stage passes nothing else -- a spy on the compile entry records
    every call's kwargs), so the cache write never launches as a programmatic dependent launch; (2) no SDPA body the form can
    run -- the shared d256 paged body compiled for Rubin and the Rubin sibling -- waits on a dependent grid or launches with
    PDL (a static detector over their SOURCE TEXT: a kernel adopting PDL after a pool write must wait before its first K / V load);
    (3) the runner's compile entry defaults ``use_pdl`` to False."""
    import inspect

    import cudnn.gated_attention_block.kernels.cache_write as cw

    assert inspect.signature(cw.compile_cache_write).parameters["use_pdl"].default is False
    calls = []

    def spy(**kw):
        calls.append(dict(kw))
        return cw.CacheWriteRecipe(
            compiled=None,
            h=kw["h"],
            d=kw["d"],
            page_size=kw["page_size"],
            rows_per_cta=1,
            two_operands=kw["two_operands"],
            dtype=kw["dtype"],
            slot_dtype=kw["slot_dtype"],
        )

    monkeypatch.setattr(cw, "compile_cache_write", spy)
    blk, *_ = _declare_extend(2, 128, max_kv_len=128)
    blk._cache_write.compile()
    assert len(calls) == 2 and all(c.get("use_pdl", False) is False for c in calls), calls
    assert "use_pdl" not in inspect.getsource(_CacheWrite.compile)
    for mod in ("cudnn.sdpa.fwd.kernels.sm100.prefill_d256_f16", "cudnn.sdpa.fwd.kernels.sm107.prefill_d256_f16"):
        src = _kernel_source(mod)
        for token in ("griddepcontrol", "wait_on_dependent_grids", "launch_dependent_grids", "use_pdl=True"):
            assert token not in src, (mod, token)


# ---------------------------------------------------------------------------
# The step harness: histories, pages, slots, the pools' prefix rows, the per-sequence oracle
# ---------------------------------------------------------------------------


class _ExtendHarness:
    """A declared + compiled EXTEND block and the serving state the cells drive it with: per sequence the FULL token history
    (prefix + new tokens, drawn once and extended per step), its pages (distinct, from a shuffled pool; the NULL page for a padded
    entry), the pools (``_STALE`` everywhere nobody wrote, the NULL page zero), the once-zeroed workspace (the serving stack's
    convention under ``token_capacity``) and the step's PERSISTENT device tensors (``h`` / ``cos`` / ``sin`` / ``slot`` / ``table`` /
    ``kv_lens`` / ``seq_lens``: rewritten in place every step, so a CUDA graph captured once replays new steps through them)."""

    def __init__(
        self, b_max, t_max, *, max_kv_len, page=16, layout="hnd", dtype=torch.bfloat16, cu=True, row_stride=None, slot_dtype=torch.int32, seed=0, **blk_kw
    ):
        self.blk, self.inp, self.out, self.samples = _declare_extend(
            b_max, t_max, max_kv_len=max_kv_len, page=page, layout=layout, dtype=dtype, cu=cu, row_stride=row_stride, seed=seed, **blk_kw
        )
        self.b_max, self.t_max, self.page, self.layout, self.dtype, self.cu = b_max, t_max, page, layout, dtype, cu
        geom_kw = blk_kw.get("geom_kw", _EXTEND_GEOM)  # the oracle's mirror follows the DECLARED geometry (a shard's, or the matrix's 8 / 2)
        self.geom, self.ref_geom = self.blk.geom, RefGeometry(**{k: v for k, v in geom_kw.items() if k in ("d_model", "h_q", "h_kv", "d_head", "rope_dim")})
        self.k_cache, self.v_cache, self.table = self.samples["sample_k_cache"], self.samples["sample_v_cache"], self.samples["sample_block_table"]
        self.n_pages, self.max_pages = int(self.k_cache.shape[0]), int(self.table.shape[1])
        self.slot = torch.full((t_max,), -1, device="cuda", dtype=slot_dtype)
        self.kv_t = torch.zeros(b_max, device="cuda", dtype=torch.int32)
        self.seq_t = torch.zeros(b_max + 1 if cu else b_max, device="cuda", dtype=torch.int32)
        self.gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
        self.hist = [None] * b_max
        self.pages = [[] for _ in range(b_max)]
        self.free = torch.randperm(self.n_pages - 1, generator=self.gen, device="cuda").add(1).tolist()  # never the NULL page
        self.blk.check_support()
        self.blk.compile()
        self.ws = torch.zeros(self.blk.get_workspace_size(), dtype=torch.uint8, device="cuda")  # zeroed ONCE: pad rows read zeros forever

    # -- state --------------------------------------------------------------

    def reset(self):
        """Fresh sequences: forget the histories and the pages, re-fill the pools with the stale value (NULL page zero)."""
        self.hist = [None] * self.b_max
        self.free = self.free + [p for row in self.pages for p in row]
        self.pages = [[] for _ in range(self.b_max)]
        for pool in (self.k_cache, self.v_cache):
            pool.fill_(_STALE)
            pool[_NULL_PAGE].zero_()

    def refill_pools(self):
        """The pools back to the stale value (NULL page zero) with the histories and the pages KEPT: the next step re-runs the same
        sequences (the prefix rows are re-written by the step's prefill, the new tokens by the block)."""
        for pool in (self.k_cache, self.v_cache):
            pool.fill_(_STALE)
            pool[_NULL_PAGE].zero_()

    def adopt(self, other):
        """Take another harness's histories and pages (the tight twin of a bounded declaration runs the SAME sequences)."""
        self.hist = [None if h is None else h.clone() for h in other.hist]
        self.pages = [list(p) for p in other.pages]
        self.free = list(other.free)

    def _extend_history(self, b, kv_len):
        have = 0 if self.hist[b] is None else int(self.hist[b].shape[0])
        if kv_len > have:
            new = torch.randn(kv_len - have, self.geom.d_model, generator=self.gen, device="cuda", dtype=torch.float32).to(self.dtype)
            self.hist[b] = new if self.hist[b] is None else torch.cat([self.hist[b], new])
        need = -(-kv_len // self.page)
        while len(self.pages[b]) < need:
            self.pages[b].append(self.free.pop())

    def oracle(self, b):
        """The dense fp32 oracle over sequence ``b``'s FULL history (positions 0 .. kv_len - 1): its ``out`` / ``o_gated`` rows past the
        prefix are the step's expected rows, its ``k`` / ``v`` the pool rows."""
        h = self.hist[b]
        cos, sin = build_rope_tables(int(h.shape[0]), self.geom.rope_dim, base=self.ref_geom.rope_base, batch=1, device="cuda", dtype=self.dtype)
        return gated_attention_block_reference(
            h[None], self.inp["w_qkvg"], self.inp["w_q_norm"], self.inp["w_k_norm"], cos, sin, self.inp["w_o"], self.ref_geom
        )

    def slot_of(self, b, j):
        return self.pages[b][j // self.page] * self.page + j % self.page

    def pool_rows(self, b, lo, hi):
        """Logical rows ``[lo, hi)`` of sequence ``b`` read back through the stack's own views: ``(K [n, H_kv, D], V [n, H_kv, D])``."""
        s = torch.tensor([self.slot_of(b, j) for j in range(lo, hi)], device="cuda")
        return self.k_cache[s // self.page, :, s % self.page, :], self.v_cache[s // self.page, :, s % self.page, :]

    def write_pool_rows(self, b, lo, hi, k_rows, v_rows):
        """Write logical rows ``[lo, hi)`` of sequence ``b`` into the pools as a torch caller would (the layout proof's other half)."""
        s = torch.tensor([self.slot_of(b, j) for j in range(lo, hi)], device="cuda")
        self.k_cache[s // self.page, :, s % self.page, :] = k_rows
        self.v_cache[s // self.page, :, s % self.page, :] = v_rows

    # -- one step -----------------------------------------------------------

    def prepare(self, q_lens, *, prefix_lens=None, prefill="reference", kv_lens=None, out_fill=None, cos_pad=1.0):
        """Build ONE step's state IN PLACE (no execute): ``q_lens[b]`` new tokens per sequence (0 = a padded / absent entry) after a
        prefix of ``prefix_lens[b]`` (default: the sequence's current history = a continuing chunk); the pools' prefix rows from the
        torch reference (``"reference"``: the prefix-hit case), from the previous step (``"keep"``: the chunked case) or none
        (``"none"``, prefix 0); ``kv_lens`` overrides the logical lengths (the dead-KV cell); ``out_fill`` sentinel-fills ``out``.
        Returns the step record (the oracle per live sequence included)."""
        b_max, t_max, page = self.b_max, self.t_max, self.page
        prefix_lens = [0 if self.hist[b] is None else int(self.hist[b].shape[0]) for b in range(b_max)] if prefix_lens is None else list(prefix_lens)
        q_lens = list(q_lens)
        assert len(q_lens) == len(prefix_lens) == b_max
        logical = [p + q for p, q in zip(prefix_lens, q_lens)]
        kv_lens = logical if kv_lens is None else list(kv_lens)
        t_live = sum(q_lens)
        assert t_live <= t_max
        for b in range(b_max):
            if logical[b] > 0:
                self._extend_history(b, logical[b])
        refs = [self.oracle(b) if logical[b] > 0 else None for b in range(b_max)]
        if prefill == "reference":
            for b in range(b_max):
                if prefix_lens[b] > 0:
                    self.write_pool_rows(b, 0, prefix_lens[b], refs[b].k[0, : prefix_lens[b]], refs[b].v[0, : prefix_lens[b]])
        elif prefill != "keep":
            assert prefill == "none" and all(p == 0 for p in prefix_lens)
        h, cos, sin = self.inp["h"], self.inp["cos"], self.inp["sin"]
        h.zero_()
        cos.fill_(cos_pad)
        sin.zero_()
        slots = [-1] * t_max
        row = 0
        rows = []
        for b in range(b_max):
            q, p = q_lens[b], prefix_lens[b]
            rows.append((row, row + q))
            if q:
                h[row : row + q] = self.hist[b][p : p + q]
                c, s = build_rope_tables(
                    q,
                    self.geom.rope_dim,
                    base=self.ref_geom.rope_base,
                    batch=1,
                    device="cuda",
                    dtype=self.dtype,
                    positions=torch.arange(p, p + q, device="cuda")[None],
                )
                cos[row : row + q] = c[0]
                sin[row : row + q] = s[0]
                if kv_lens[b] >= p + q:  # the step's tokens have slots iff they are inside the logical range (a dead-KV entry writes nothing)
                    slots[row : row + q] = [self.slot_of(b, j) for j in range(p, p + q)]
            row += q
        self.slot.copy_(torch.tensor(slots, dtype=self.slot.dtype))
        self.table.zero_()
        for b in range(b_max):
            if kv_lens[b] > 0:
                n = -(-kv_lens[b] // page)
                self.table[b, :n] = torch.tensor(self.pages[b][:n], device="cuda", dtype=torch.int32)
        self.kv_t.copy_(torch.tensor(kv_lens, dtype=torch.int32))
        self.seq_t.copy_(torch.tensor(cu_seqlens_of(q_lens) if self.cu else q_lens, dtype=torch.int32))
        if out_fill is not None:
            self.out.fill_(out_fill)
        torch.cuda.synchronize()
        return dict(q_lens=q_lens, prefix_lens=prefix_lens, kv_lens=kv_lens, t_live=t_live, rows=rows, refs=refs)

    def execute(self, current_stream=None):
        """The block's execute over the prepared state (the ONE call a serving stack captures)."""
        self.blk.execute(
            self.inp["h"],
            self.inp["w_qkvg"],
            self.inp["w_q_norm"],
            self.inp["w_k_norm"],
            self.inp["cos"],
            self.inp["sin"],
            self.inp["w_o"],
            self.out,
            self.ws,
            seq_lens=self.seq_t,
            current_stream=current_stream,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            block_table=self.table,
            kv_lens=self.kv_t,
            slot_mapping=self.slot,
        )

    def step(self, q_lens, **kw):
        """``prepare`` + ``execute`` eagerly; the record carries ``out``."""
        rec = self.prepare(q_lens, **kw)
        self.execute()
        torch.cuda.synchronize()
        rec["out"] = self.out.clone()
        return rec

    # -- the checks ---------------------------------------------------------

    def slab_rows(self):
        """The post-RoPE K / V rows attention would consume, as ``[T_max, H_kv, D]`` views of the workspace (the slab bands in place,
        the compact buffers out of place)."""
        g, t, lay = self.geom, self.t_max, self.blk._layout()
        if not self.blk.inplace_qkv:
            return _view(self.ws, lay.k, (t, g.h_kv, g.d_head), self.dtype), _view(self.ws, lay.v, (t, g.h_kv, g.d_head), self.dtype)
        proj = _view(self.ws, lay.proj, (t, g.n_qkvg), self.dtype)
        return _cols(proj, g.qkvg_offsets[2], g.h_kv, g.d_head), _cols(proj, g.qkvg_offsets[3], g.h_kv, g.d_head)

    def o_gated(self):
        """The gated O the SDPA + gate left in the ``o`` slot (in place under inference), ``[T_max, H_q, D]``."""
        g, lay = self.geom, self.blk._layout()
        return _view(self.ws, lay.o, (self.t_max, g.h_q, g.d_head), self.dtype)

    def check(self, rec, *, label, out_cos=0.999, o_atol=2e-2):
        """Every assertion a step owes: ``isfinite(out)`` over EVERY row first; per live sequence the dense suite's budget vs the
        oracle over its full history (``out`` cos, gated ``O`` cos + atol 2e-2); a live-Q dead-KV sequence's rows exactly 0; the new
        tokens' pool rows bitwise the slab rows attention consumed; every pool row nobody addressed still stale (NULL page zero).
        Collects per-sequence verdicts, prints one line each, asserts at the end."""
        out, g = rec["out"], self.geom
        assert torch.isfinite(out.float()).all(), f"{label}: non-finite out rows: {int((~torch.isfinite(out.float())).any(-1).sum())}"
        og = self.o_gated()
        failures = []
        for b, ((lo, hi), ref) in enumerate(zip(rec["rows"], rec["refs"])):
            q, p = rec["q_lens"][b], rec["prefix_lens"][b]
            if q == 0:
                print(f"{label} seq {b}: no new tokens (kv_len {rec['kv_lens'][b]})")
                continue
            if rec["kv_lens"][b] == 0:
                dead = out[lo:hi]
                if not bool((dead == 0).all()):
                    failures.append(f"seq {b}: dead-KV rows not exactly 0 (max |out| {dead.float().abs().max().item():.3e})")
                else:
                    print(f"{label} seq {b}: live Q over kv_len 0 -> {q} out rows exactly 0")
                continue
            want_out, want_og = ref.out[0, p : p + q], ref.o_gated[0, p : p + q]
            c = _cos(out[lo:hi], want_out)
            co = _cos(og[lo:hi], want_og)
            err = (og[lo:hi].float() - want_og.float()).abs().max().item()
            print(f"{label} seq {b}: q {q} prefix {p} kv_len {rec['kv_lens'][b]}: out cos {c:.6f}, gated-O cos {co:.6f} max|err| {err:.3e}")
            if not (c > out_cos and co > out_cos and err <= o_atol and torch.isfinite(og[lo:hi].float()).all()):
                failures.append(f"seq {b}: out cos {c:.6f} gated-O cos {co:.6f} max|err| {err:.3e}")
        assert not failures, f"{label}: " + "; ".join(failures)
        # the write-through half on the step's tokens, then the stale rows
        slot = self.slot
        live = slot >= 0
        k_rows, v_rows = self.slab_rows()
        if bool(live.any()):
            assert torch.equal(_bits(_rows_at(self.k_cache, slot, self.page)), _bits(k_rows[live])), f"{label}: the K pool is not the K attention consumed"
            assert torch.equal(_bits(_rows_at(self.v_cache, slot, self.page)), _bits(v_rows[live])), f"{label}: the V pool is not the V attention consumed"
        addressed = sorted({self.slot_of(b, j) for b in range(self.b_max) for j in range(rec["kv_lens"][b])})
        mask = torch.ones(self.n_pages * self.page, dtype=torch.bool, device="cuda")
        if addressed:
            mask[torch.tensor(addressed, device="cuda")] = False
        mask[: self.page] = False  # the NULL page is zero, never stale
        for pool in (self.k_cache, self.v_cache):
            flat = pool.permute(0, 2, 1, 3).reshape(self.n_pages * self.page, g.h_kv, g.d_head)
            assert bool((flat[mask] == _STALE).all()), f"{label}: a pool row nobody addressed was written"
            assert bool((pool[_NULL_PAGE] == 0).all()), f"{label}: the NULL page was written"


_HARNESS = {}  # the declaration key -> _ExtendHarness: ONE compiled block per declaration, the cells vary the packing (device data)


def _harness(b_max, t_max, *, max_kv_len, page=16, layout="hnd", dtype=torch.bfloat16, cu=True, row_stride=None, slot_dtype=torch.int32, seed=0, **blk_kw):
    frozen = tuple(sorted((k, tuple(sorted(v.items())) if isinstance(v, dict) else v) for k, v in blk_kw.items()))  # geom_kw is a dict
    key = (dtype, page, layout, cu, b_max, t_max, max_kv_len, row_stride, slot_dtype, seed, frozen)
    if key not in _HARNESS:
        _HARNESS[key] = _ExtendHarness(
            b_max, t_max, max_kv_len=max_kv_len, page=page, layout=layout, dtype=dtype, cu=cu, row_stride=row_stride, slot_dtype=slot_dtype, seed=seed, **blk_kw
        )
    hz = _HARNESS[key]
    hz.reset()
    return hz


def _dense_block_on(hz, h_full):
    """The dense ``B = 1, S = kv_len`` block on a sequence's full history, on the SAME weights and the same (bottom-right) geometry:
    returns ``(out [S, d_model], o_gated [S, H_q, D])``."""
    s = int(h_full.shape[0])
    cos, sin = build_rope_tables(s, hz.geom.rope_dim, base=hz.ref_geom.rope_base, batch=1, device="cuda", dtype=hz.dtype)
    out = torch.empty(1, s, hz.geom.d_model, device="cuda", dtype=hz.dtype)
    blk = GatedAttentionBlockFwd(h_full[None], hz.inp["w_qkvg"], hz.inp["w_q_norm"], hz.inp["w_k_norm"], cos, sin, hz.inp["w_o"], out, hz.geom)
    blk.check_support()
    blk.compile()
    ws = torch.zeros(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    blk.execute(h_full[None], hz.inp["w_qkvg"], hz.inp["w_q_norm"], hz.inp["w_k_norm"], cos, sin, hz.inp["w_o"], out, ws)
    torch.cuda.synchronize()
    og = _view(ws, blk._layout().o, (s, hz.geom.h_q, hz.geom.d_head), hz.dtype).clone()
    return out[0].clone(), og


# ---------------------------------------------------------------------------
# Rubin: the EXTEND cells
# ---------------------------------------------------------------------------


@pytest.mark.L1
@requires_rubin
def test_extend_one_sequence_empty_prefix_at_a_tight_declaration_vs_the_dense_block():
    """(a) ONE sequence, EMPTY prefix, the TIGHT declaration (T_live = T_max = S = 384, a tail tile): the EXTEND step reads the step's
    own tokens back from the pool it just wrote and must equal the dense ``B = 1`` block on the same tokens -- the SAME bytes in the same
    tile order; the two legs are DIFFERENT bodies (the shared d256 paged THD body compiled for Rubin vs the Rubin sibling) and MEASURED
    bitwise on both ``out`` and the gated O (0 of 196 608 / 786 432 elements differ on the Rubin development node, 2026-10-13), so bitwise
    is pinned here -- a legitimately changed arithmetic order on either body relaxes this to the recorded one-rounding bound in the
    open, never silently; both legs stay within the dense suite's budget against the oracle."""
    hz = _harness(1, 384, max_kv_len=384, page=16, layout="hnd")
    rec = hz.step([384], prefix_lens=[0], prefill="none")
    hz.check(rec, label="(a) empty prefix tight")
    out_dense, og_dense = _dense_block_on(hz, hz.hist[0])
    out_ext, og_ext = rec["out"], hz.o_gated()
    d_out = (out_ext.float() - out_dense.float()).abs().max().item()
    d_og = (og_ext.float() - og_dense.float()).abs().max().item()
    bitwise = torch.equal(out_ext, out_dense) and torch.equal(og_ext, og_dense)
    print(
        f"(a) EXTEND vs dense B=1 S=384: bitwise={bitwise} max|diff| out {d_out:.3e} (|out| max {out_dense.float().abs().max().item():.3f}), "
        f"gated O {d_og:.3e} (|O| max {og_dense.float().abs().max().item():.3f})"
    )
    ref = rec["refs"][0]
    assert torch.isfinite(out_dense.float()).all() and _cos(out_dense, ref.out[0]) > 0.999 and _cos(out_ext, ref.out[0]) > 0.999
    torch.testing.assert_close(og_dense.float(), ref.o_gated[0].float(), rtol=0, atol=2e-2)
    assert bitwise, f"the EXTEND step is no longer bitwise the dense block: max|diff| out {d_out:.3e}, gated O {d_og:.3e} (recorded bitwise)"


@pytest.mark.L1
@requires_rubin
def test_extend_two_chunks_match_the_single_shot_dense_block_and_the_oracle():
    """(b) a 512-token sequence as TWO chunks on ONE compiled block (T_max 320 > both chunks: a bounded declaration with pad rows):
    chunk 1 = tokens [0, 300) with an empty prefix (kv_len 300), chunk 2 = tokens [300, 512) reading chunk 1's pool rows (kv_len 512).
    Each chunk's rows vs the single-shot dense block over [0, 512) and vs the oracle within the dense suite's budget (cos >= 0.999 on
    ``out``, atol 2e-2 on the gated O); the pool after both chunks within the write-through's bars of the oracle's K / V over all 512
    positions (the token-wise stages at the same positions).  MEASURED bitwise the single-shot block on both chunks' ``out`` and gated O
    (the Rubin development node, 2026-10-13): pinned, with the budget kept."""
    hz = _harness(1, 320, max_kv_len=512, page=16, layout="hnd")
    rec1 = hz.step([300], prefix_lens=[0], prefill="none")
    hz.check(rec1, label="(b) chunk 1")
    og1 = hz.o_gated()[:300].clone()
    rec2 = hz.step([212], prefill="keep")
    hz.check(rec2, label="(b) chunk 2 over chunk 1's pool rows")
    og2 = hz.o_gated()[:212].clone()
    out_dense, og_dense = _dense_block_on(hz, hz.hist[0])
    ref = rec2["refs"][0]
    for label, rec, og, lo, hi in (("chunk 1", rec1, og1, 0, 300), ("chunk 2", rec2, og2, 300, 512)):
        o = rec["out"][: hi - lo]
        c_d, c_r = _cos(o, out_dense[lo:hi]), _cos(o, ref.out[0, lo:hi])
        d = (o.float() - out_dense[lo:hi].float()).abs().max().item()
        d_og = (og.float() - og_dense[lo:hi].float()).abs().max().item()
        print(f"(b) {label} vs single-shot dense: out cos {c_d:.6f} max|diff| {d:.3e}, gated O max|diff| {d_og:.3e}; vs oracle out cos {c_r:.6f}")
        assert c_d > 0.999 and c_r > 0.999
        torch.testing.assert_close(og.float(), ref.o_gated[0, lo:hi].float(), rtol=0, atol=2e-2)
        assert torch.equal(o, out_dense[lo:hi]) and torch.equal(og, og_dense[lo:hi]), f"{label}: no longer bitwise the single-shot block (recorded bitwise)"
    k_all, v_all = hz.pool_rows(0, 0, 512)
    torch.testing.assert_close(v_all.float(), ref.v[0].float(), **_BAND_BAR)
    torch.testing.assert_close(k_all.float(), ref.k[0].float(), rtol=_BAND_BAR["rtol"], atol=_NORM_ATOL[hz.dtype])


@pytest.mark.L1
@requires_rubin
def test_extend_prefix_hit_pool_prefilled_by_the_torch_reference_on_l3():
    """(c) the prefix-hit step on the UNIFIED content row (L3): three sequences whose prefixes (200 / 0 / 77 rows) were written into
    the pool by the TORCH REFERENCE through the stack's own views (``kv_cache[page, :, offset, :D]`` / ``[..., D:]``) -- FROST reads
    what torch wrote (the step's rows vs the oracle over the full history), and torch reads back what FROST wrote (the new tokens'
    rows read through the same views are bitwise the slab rows, ``check``): the cross-implementation LAYOUT proof, both directions."""
    hz = _harness(3, 512, max_kv_len=512, page=16, layout="unified")
    assert hz.k_cache.data_ptr() + hz.geom.d_head * hz.k_cache.element_size() == hz.v_cache.data_ptr() and hz.k_cache.stride(2) == 2 * hz.geom.d_head
    rec = hz.step([56, 128, 300], prefix_lens=[200, 0, 77], prefill="reference")
    hz.check(rec, label="(c) prefix hit on L3")
    # the prefix rows torch wrote are untouched by the step (the SDPA reads, the write-through addresses the new tokens only)
    for b, p in ((0, 200), (2, 77)):
        k_p, v_p = hz.pool_rows(b, 0, p)
        assert torch.equal(_bits(k_p), _bits(rec["refs"][b].k[0, :p])) and torch.equal(_bits(v_p), _bits(rec["refs"][b].v[0, :p]))


@pytest.mark.L1
@requires_rubin
def test_extend_l1_and_l3_pools_holding_the_same_values_are_bitwise():
    """(d) the three pool layouts on the paged THD read: the same step (histories, prefixes, pages) on an L1 (HND compact), an L2 (NHD by strides) and
    an L3 (unified row) declaration -- three compiled plans (the pool layout is compiled in) over pools holding the same values --
    give a bitwise identical ``out``."""
    outs = {}
    for layout in SERVING_POOL_LAYOUTS:
        hz = _harness(3, 512, max_kv_len=512, page=32, layout=layout, seed=3)
        rec = hz.step([56, 128, 300], prefix_lens=[200, 0, 77], prefill="reference")
        hz.check(rec, label=f"(d) {layout}")
        outs[layout] = rec["out"].clone()
    for layout in ("nhd", "unified"):
        assert torch.equal(outs["hnd"], outs[layout]), f"(d) {layout} differs from hnd"


@requires_rubin
@pytest.mark.parametrize(
    "dtype, layout, page",
    [
        pytest.param(
            dtype,
            layout,
            page,
            marks=pytest.mark.L0 if dtype == torch.bfloat16 and layout == "hnd" and page == 16 else pytest.mark.L1,
            id=f"{dtype_id}-{layout}-{page}",
        )
        for dtype, dtype_id in [(torch.bfloat16, "bf16"), (torch.float16, "f16")]
        for layout in _LAYOUTS
        for page in _PAGES
    ],
)
def test_extend_matrix_on_every_layout_and_page(dtype, layout, page):
    """(e) the degenerate matrix on serving buffers, one declaration per (dtype, layout, page): four sequences at T_max 512 -- a
    128-token sequence with an empty prefix (n_kv = 1; B x H_kv > 1 with it), an 84-token step over a 300-row prefix ending mid-page
    (kv_len 384 = 3 tiles), a PADDED entry (kv_len 0, the NULL block, no rows), a 200-token step over a 77-row prefix (kv_len 277: a
    tail tile, n_kv = 3) -- T_live 412 < 512 (pads at the END, slot -1); isfinite(out) before any cosine; every live sequence within the
    dense suite's budget of the oracle; the pools bitwise the slab rows at the step's slots, stale everywhere else; the int64 slot
    dtype at pages 32 / 128."""
    hz = _harness(4, 512, max_kv_len=512, page=page, layout=layout, dtype=dtype, slot_dtype=torch.int32 if page in (16, 64) else torch.int64)
    rec = hz.step(_MATRIX_PACKING["q_lens"], prefix_lens=_MATRIX_PACKING["prefix_lens"], prefill="reference")
    assert rec["t_live"] == 412
    hz.check(rec, label=f"(e) 8/2 {dtype} {layout} p{page}")


@pytest.mark.L1
@requires_rubin
def test_extend_padded_entries_in_the_middle_and_a_live_q_over_kv_len_0():
    """(e) the general packing: a zero-length sequence in the MIDDLE (B_live < B_max), and -- the serving stacks' padded request in
    its THD form -- a LIVE Q range (5 new tokens) over kv_len 0 with the NULL block and slot -1: those 5 ``out`` rows are
    EXACTLY 0 (the dead-row SELECT, never a residue), nothing is written to the NULL block, the neighbours are unaffected."""
    hz = _harness(4, 512, max_kv_len=512, page=16, layout="hnd")
    rec = hz.step([100, 0, 60, 5], prefix_lens=[0, 0, 0, 0], prefill="none", kv_lens=[100, 0, 60, 0])
    hz.check(rec, label="(e) middle pad + dead KV")
    lo, hi = rec["rows"][3]
    assert bool((rec["out"][lo:hi] == 0).all()) and bool((hz.o_gated()[lo:hi] == 0).all())
    assert rec["t_live"] == 165 and bool((hz.slot[lo:hi] == -1).all())


@pytest.mark.L1
@requires_rubin
@pytest.mark.parametrize("n_kv", [1, 2, 3, 4, 8])
def test_extend_n_kv_tiles_and_the_ring_wrap(n_kv):
    """(e) one sequence of 64 new tokens over a prefix bringing kv_len to exactly 128 x n_kv (whole tiles: 1, 2, the ring depth 3 and 4,
    the wrap at 8) and to 128 x n_kv - 37 (a tail tile), on L2 at page 64 -- the prefix from the reference."""
    hz = _harness(1, 64, max_kv_len=1024, page=64, layout="nhd", seed=n_kv)
    for kv_len in (128 * n_kv, 128 * n_kv - 37):
        hz.reset()
        rec = hz.step([64], prefix_lens=[kv_len - 64], prefill="reference")
        hz.check(rec, label=f"(e) n_kv {n_kv} kv_len {kv_len}")


@pytest.mark.L1
@requires_rubin
def test_extend_block_table_row_slice_at_an_arbitrary_row_stride():
    """(e) a row-sliced block table through the BLOCK API: the block table is a ``[:, :max_pages]`` slice of a wider ``[B_max, 50]`` table (row stride 50,
    declared by the sample) -- the step over it equals the same step over a compact table bitwise."""
    hz_sliced = _harness(3, 256, max_kv_len=256, page=16, layout="hnd", row_stride=50, seed=5)
    rec_s = hz_sliced.step([100, 0, 156], prefix_lens=[60, 0, 100], prefill="reference")
    hz_sliced.check(rec_s, label="(e) row-sliced table")
    hz = _harness(3, 256, max_kv_len=256, page=16, layout="hnd", seed=5)
    rec = hz.step([100, 0, 156], prefix_lens=[60, 0, 100], prefill="reference")
    hz.check(rec, label="(e) compact table")
    assert torch.equal(rec_s["out"], rec["out"])


@pytest.mark.L1
@requires_rubin
def test_extend_finite_stale_rows_past_kv_len_do_not_change_out():
    """(e) a prefix ending mid-page: the rows of the last live page past kv_len hold another sequence's stale values -- FINITE, the
    serving pools' contract (zero-initialised, then stale rows) -- and never change the result: the step over a 777-filled tail is
    bitwise the step over a zero-filled tail (the masked columns' P is exactly 0, and 0 x finite = 0)."""
    hz = _harness(2, 256, max_kv_len=512, page=64, layout="hnd", seed=7)
    rec_stale = hz.step([40, 200], prefix_lens=[300, 77], prefill="reference")
    hz.check(rec_stale, label="(e) stale tail 777")
    for b in range(2):
        kv = rec_stale["kv_lens"][b]
        last = hz.pages[b][(kv - 1) // 64]
        hz.k_cache[last, :, kv % 64 :, :] = 0
        hz.v_cache[last, :, kv % 64 :, :] = 0
    rec_zero = hz.step([40, 200], prefix_lens=[300, 77], prefill="reference")
    assert torch.equal(rec_stale["out"], rec_zero["out"])


@pytest.mark.L1
@requires_rubin
def test_extend_bounded_tokens_live_rows_bitwise_the_tight_declaration_and_pad_rows_exactly_zero():
    """(f) the bounded token form: the matrix packing (T_live 412) at T_max 512 vs the SAME packing on a TIGHT declaration (T_max =
    412): live rows BITWISE; on a sentinel-filled ``out`` over the once-zeroed workspace the pad rows [412, 512) are EXACTLY 0 (an
    unwritten ``o`` row is 0, so ``0 x sigmoid(G_pad) = 0`` and ``0 @ W_o = 0``) and the pad tokens' slots (-1) write nothing; a
    SECOND step with fewer live tokens (300) on the same workspace leaves [300, 412) FINITE (stale O rows -- unspecified, as the
    contract says) and [412, 512) still exactly 0."""
    hz = _harness(4, 512, max_kv_len=512, page=16, layout="hnd", seed=11)
    sentinel = _OUT_SENTINEL[hz.dtype]
    rec = hz.step(_MATRIX_PACKING["q_lens"], prefix_lens=_MATRIX_PACKING["prefix_lens"], prefill="reference", out_fill=sentinel)
    hz.check(rec, label="(f) bounded T_live 412 / 512")
    out = rec["out"]
    assert bool((out[412:] == 0).all()), f"pad rows not exactly 0: {int((out[412:] != 0).any(-1).sum())} rows"
    assert not bool((out == sentinel).any())
    og_bounded = hz.o_gated()[:412].clone()
    tight = _harness(4, 412, max_kv_len=512, page=16, layout="hnd", seed=11)
    tight.adopt(hz)
    rec_t = tight.step(_MATRIX_PACKING["q_lens"], prefix_lens=_MATRIX_PACKING["prefix_lens"], prefill="reference", out_fill=sentinel)
    tight.check(rec_t, label="(f) tight T_live = T_max = 412")
    assert torch.equal(out[:412], rec_t["out"]), "live rows differ between the bounded and the tight declaration"
    assert torch.equal(_bits(og_bounded), _bits(tight.o_gated()))
    rec2 = hz.step(_SECOND_PACKING["q_lens"], prefix_lens=_SECOND_PACKING["prefix_lens"], prefill="keep", out_fill=sentinel)
    hz.check(rec2, label="(f) second step T_live 300")
    assert rec2["t_live"] == 300
    assert torch.isfinite(rec2["out"].float()).all() and bool((rec2["out"][412:] == 0).all()) and not bool((rec2["out"] == sentinel).any())


@pytest.mark.L1
@requires_rubin
def test_extend_two_launches_are_bitwise():
    """The SAME step twice (histories and pages kept, the pools re-filled with the stale value in between): ``out``, the gated O and the
    pools bitwise across the two launches (a first-launch race would show here first)."""
    hz = _harness(4, 512, max_kv_len=512, page=32, layout="unified", seed=13)
    runs = []
    for _ in range(2):
        hz.refill_pools()
        rec = hz.step(_MATRIX_PACKING["q_lens"], prefix_lens=_MATRIX_PACKING["prefix_lens"], prefill="reference")
        hz.check(rec, label="(e) two launches")
        runs.append((rec["out"], hz.o_gated().clone(), _bits(hz.k_cache).clone(), _bits(hz.v_cache).clone()))
    for a, b in zip(runs[0], runs[1]):
        assert torch.equal(a, b)


@pytest.mark.L1
@requires_rubin
@requires_capture_safe_gemm_route
def test_extend_capture_of_the_first_execute_replays_with_new_lengths_tables_and_slots():
    """(g) vLLM's defaults: a FRESH EXTEND declaration whose FIRST execute is captured inside ``torch.cuda.graph`` on a NEW stream
    with no eager warm-up, under ``set_sync_debug_mode("error")`` (no host sync in the capture); the capture launches nothing (the
    sentinel survives), the replay is bitwise a fresh eager run; then a NEW step -- new lengths, tables, slots and pool contents
    written through the captured tensors (the next chunk of the same sequences) -- replays to the eager result over it."""
    hz = _ExtendHarness(4, 512, max_kv_len=512, page=16, layout="hnd", seed=17)
    twin = _ExtendHarness(4, 512, max_kv_len=512, page=16, layout="hnd", seed=17)
    sentinel = _OUT_SENTINEL[hz.dtype]
    rec = hz.prepare(_MATRIX_PACKING["q_lens"], prefix_lens=_MATRIX_PACKING["prefix_lens"], prefill="reference", out_fill=sentinel)
    stream = torch.cuda.Stream()  # NEW: the block never ran on it, nothing ran on it
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                hz.execute()  # the FIRST execute of this declaration, captured
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert bool((hz.out == sentinel).all()), "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        rec["out"] = hz.out.clone()
        hz.check(rec, label="(g) replay of the captured first execute")
        rec_e = twin.step(_MATRIX_PACKING["q_lens"], prefix_lens=_MATRIX_PACKING["prefix_lens"], prefill="reference", out_fill=sentinel)
        twin.check(rec_e, label="(g) eager twin")
        assert torch.equal(rec["out"], rec_e["out"]), "the replay of the captured first execute differs from the eager run"
        # the next chunk: new lengths / tables / slots / pool contents through the captured tensors
        rec2 = hz.prepare(_SECOND_PACKING["q_lens"], prefix_lens=_SECOND_PACKING["prefix_lens"], prefill="keep", out_fill=sentinel)
        graph.replay()
        torch.cuda.synchronize()
        rec2["out"] = hz.out.clone()
        hz.check(rec2, label="(g) replay over the next chunk")
        rec2_e = twin.step(_SECOND_PACKING["q_lens"], prefix_lens=_SECOND_PACKING["prefix_lens"], prefill="keep", out_fill=sentinel)
        twin.check(rec2_e, label="(g) eager twin, next chunk")
        assert torch.equal(rec2["out"], rec2_e["out"]), "the replay over new lengths / tables / slots differs from the eager run over them"
    finally:
        graph.reset()


_ADMITTED_SPELLINGS = {
    # id: (harness kwargs -- b_max / t_max first, q_lens, prefix_lens, the EXECUTED block table's column count or None = the declared max_pages)
    "lengths_form_bf16_hnd_p16": (
        dict(b_max=4, t_max=512, max_kv_len=512, page=16, layout="hnd", cu=False, seed=21),
        _MATRIX_PACKING["q_lens"],
        _MATRIX_PACKING["prefix_lens"],
        None,
    ),
    "fuse_norm_rope_bf16_nhd_p32": (
        dict(b_max=4, t_max=512, max_kv_len=512, page=32, layout="nhd", seed=22, fuse_norm_rope=True),
        _MATRIX_PACKING["q_lens"],
        _MATRIX_PACKING["prefix_lens"],
        None,
    ),
    "out_of_place_qkv_f16_unified_p64": (
        dict(b_max=4, t_max=512, max_kv_len=512, page=64, layout="unified", dtype=torch.float16, seed=23, inplace_qkv=False),
        _MATRIX_PACKING["q_lens"],
        _MATRIX_PACKING["prefix_lens"],
        None,
    ),
    "full_table_row_on_a_sliced_table_lengths_form": (
        dict(b_max=2, t_max=256, max_kv_len=256, page=16, layout="hnd", row_stride=50, cu=False, seed=24),
        [128, 16],
        [128, 0],
        None,
    ),
    "wider_table_40_columns_at_row_stride_50_max_pages_16": (
        dict(b_max=2, t_max=256, max_kv_len=256, page=16, layout="hnd", row_stride=50, seed=25),
        [100, 156],
        [60, 100],
        40,
    ),
}


@pytest.mark.L1
@requires_rubin
@pytest.mark.parametrize("spelling", list(_ADMITTED_SPELLINGS), ids=list(_ADMITTED_SPELLINGS))
def test_extend_admitted_spellings_execute_within_the_budget(spelling):
    """The spellings the declaration cell ADMITS ("every served spelling builds the same stage") or the docs CLAIM at execute, each
    EXECUTED on the target body through the suite's own harness and ``check`` (the dense suite's budget, the pool bitwise the slab rows,
    the stale rows untouched -- nothing widened): the [B] lengths form (``cu_seqlens=False``) on the matrix packing; the fused norm +
    RoPE pipeline (the fork writes the post-RoPE slab the write-through reads); the out-of-place Q / K / V pipeline (f16); a FULL table
    row (kv_len = max_pages x P = 256) on a ``[:, :max_pages]`` slice of a wider table under the lengths form; and a block table WIDER
    than ``max_pages`` (``[2, 40]`` at the declared row stride 50, ``max_pages`` 16 -- the docs' ``[num_sequences, >= max_pages]``).
    MEASURED on the Rubin development node (2026-10-13): 5 / 5 within the budget, worst ``out`` cos 0.999988 (the fused pipeline),
    gated-O max|err| 3.906e-03 bf16 / 4.883e-04 f16."""
    kw, q_lens, prefix_lens, table_cols = _ADMITTED_SPELLINGS[spelling]
    kw = dict(kw)
    hz = _harness(kw.pop("b_max"), kw.pop("t_max"), **kw)
    declared = hz.table
    if table_cols is not None:  # the SAME storage with more columns than the declared max_pages, at the declared row stride
        assert table_cols > hz.max_pages
        hz.table = declared.as_strided((hz.b_max, table_cols), (declared.stride(0), 1))
    try:
        rec = hz.step(q_lens, prefix_lens=prefix_lens, prefill="reference")
        hz.check(rec, label=f"(admitted) {spelling}")
    finally:
        hz.table = declared
    if spelling.startswith("full_table_row"):
        assert rec["kv_lens"][0] == hz.max_pages * hz.page, rec["kv_lens"]  # the row addresses every one of its max_pages columns
    og = hz.o_gated()
    worst_cos, worst_err = 1.0, 0.0
    for b, ((lo, hi), ref) in enumerate(zip(rec["rows"], rec["refs"])):
        q, p = rec["q_lens"][b], rec["prefix_lens"][b]
        if q and rec["kv_lens"][b]:
            worst_cos = min(worst_cos, _cos(rec["out"][lo:hi], ref.out[0, p : p + q]))
            worst_err = max(worst_err, (og[lo:hi].float() - ref.o_gated[0, p : p + q].float()).abs().max().item())
    print(f"(admitted) {spelling}: t_live {rec['t_live']} kv_lens {rec['kv_lens']} worst out cos {worst_cos:.6f} gated-O max|err| {worst_err:.3e}")


# The 397B shards at their real width (d_model 4096; the layout contract pins n_qkvg 8704 / 4608 / 2560): the query heads divided across
# the ranks, the ONE replicated KV head from TP 4 on -- each on the kernel block size its deployment allocates (16 / 32 / 64), so the
# three cells also walk the pages.  (geometry, page)
_SHARD_GEOMS = {
    "q35_397b_tp2_16_1": (dict(d_model=4096, h_q=16, h_kv=1, d_head=256, rope_dim=64, causal_bottom_right=True), 16),
    "q35_397b_tp4_8_1": (dict(d_model=4096, h_q=8, h_kv=1, d_head=256, rope_dim=64, causal_bottom_right=True), 32),
    "q35_397b_tp8_4_1": (dict(d_model=4096, h_q=4, h_kv=1, d_head=256, rope_dim=64, causal_bottom_right=True), 64),
}


@pytest.mark.L1
@requires_rubin
@pytest.mark.parametrize("name", list(_SHARD_GEOMS), ids=list(_SHARD_GEOMS))
def test_extend_qwen35_shards_execute_within_the_budget(name):
    """One EXTEND step per 397B shard (16 / 1 at page 16, 8 / 1 at page 32, 4 / 1 at page 64; two sequences at ``T_max`` 256 = a tight
    packing): a 100-token step over a 60-row prefix (kv_len 160: a tail tile; the prefix ends mid-page at pages 32 / 64) and a
    156-token step over a 100-row prefix (kv_len 256: two KV tiles exactly, the table row FULL at every page), the prefix rows written
    by the torch reference, through the suite's own harness and ``check`` (the dense suite's budget on every live sequence, the new
    tokens' pool rows bitwise the slab rows attention consumed, the stale rows untouched -- nothing widened)."""
    geom_kw, page = _SHARD_GEOMS[name]
    hz = _harness(2, 256, max_kv_len=256, page=page, geom_kw=geom_kw, seed=31)
    assert hz.geom.d_model == 4096 and hz.geom.h_kv == 1 and hz.k_cache.shape[1] == 1 and hz.geom.n_qkvg % 256 == 0 and hz.max_pages * page == 256
    rec = hz.step([100, 156], prefix_lens=[60, 100], prefill="reference")
    hz.check(rec, label=f"(shard) {name} p{page}")
    assert rec["kv_lens"] == [160, 256] and rec["t_live"] == 256 == hz.t_max
    og = hz.o_gated()
    worst_cos, worst_err = 1.0, 0.0
    for b, ((lo, hi), ref) in enumerate(zip(rec["rows"], rec["refs"])):
        q, p = rec["q_lens"][b], rec["prefix_lens"][b]
        worst_cos = min(worst_cos, _cos(rec["out"][lo:hi], ref.out[0, p : p + q]))
        worst_err = max(worst_err, (og[lo:hi].float() - ref.o_gated[0, p : p + q].float()).abs().max().item())
    print(f"(shard) {name} p{page}: t_live {rec['t_live']} kv_lens {rec['kv_lens']} worst out cos {worst_cos:.6f} gated-O max|err| {worst_err:.3e}")
