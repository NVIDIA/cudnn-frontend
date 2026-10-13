# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's EXTEND (serving prefill) form, part one: the write-through of post-RoPE K / V (and the raw
indexer key) into paged pools at a slot mapping -- ``paged_kv_page_size``.

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
import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry, QsaSpec, QuantSpec, index_k_raw_view
from cudnn.gated_attention_block.api import PAGED_KV_PAGE_ALIGN, _align_up, _CacheWrite, _cols, _SparseSdpa, _view

# Below FROST's CuTe DSL floor the block's stages cannot import their kernels: the module skips as a whole there (the sibling
# block modules' convention); the typed decline itself is pinned by ``test_the_stage_declines_a_dsl_below_the_floor_by_version``.
requirement_error = cutedsl_requirement_error("the gated attention block's serving tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import (  # noqa: E402
    RefGeometry,
    build_rope_tables,
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


def _pools(n_pages: int, h: int, page_size: int, d: int, layout: str, dtype=torch.bfloat16, device=_DEV):
    """Two NaN-filled pools as the kernel's 4-D ``[P, H, page_size, D]`` views -- ``(k_cache, v_cache)``.

    ``hnd``: HND compact storage, one tensor per pool.  ``nhd``: NHD storage (``[P, page_size, H, D]`` in memory) permuted
    into the same logical order -- the layout is nothing but the strides.  ``unified``: ONE ``[P, H, page_size, 2D]``
    content row per token (K at ``+0``, V at ``+D``; token stride ``2D``), the two pools being its two halves."""
    if layout == "hnd":
        return tuple(torch.full((n_pages, h, page_size, d), float("nan"), device=device, dtype=dtype) for _ in range(2))
    if layout == "nhd":
        return tuple(torch.full((n_pages, page_size, h, d), float("nan"), device=device, dtype=dtype).permute(0, 2, 1, 3) for _ in range(2))
    assert layout == "unified"
    row = torch.full((n_pages, h, page_size, 2 * d), float("nan"), device=device, dtype=dtype)
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


def test_page_size_align_is_the_serving_stacks_granularity():
    assert PAGED_KV_PAGE_ALIGN == 16 and PAGED_KV_PAGE_ALIGN % 4 == 0  # a 4-token sparse block never straddles a page


@pytest.mark.parametrize("page_size", list(_PAGES) + [48, 256])
def test_a_legal_page_size_builds_the_stage_in_pipeline_position(page_size):
    blk, *_ = _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=page_size)
    _assert_stage_in_pipeline_position(blk, page_size)


@pytest.mark.parametrize("page_size", _PAGES)
def test_a_packed_declaration_builds_the_same_stage(page_size):
    """The write-through under ``thd=True`` is SERVED (the former typed decline is inverted, not deleted): the stage
    addresses tokens, so the packed block -- ``B = 1, S = T`` inside -- builds the same stage at the same pipeline
    position, sized at the packed token count."""
    blk, inp, out, meta = _make_packed_block(_SMALL_D256, _PACKED_LENS, paged_kv_page_size=page_size)
    _assert_stage_in_pipeline_position(blk, page_size)
    assert blk.thd and blk.batch * blk.seq_len == meta["t"] == sum(_PACKED_LENS)
    assert blk._cache_write.batch * blk._cache_write.seq_len == meta["t"]


def test_zero_is_off_and_builds_no_stage():
    blk, *_ = _make_block(_SMALL_D256, 1, 64)
    assert blk.paged_kv_page_size == 0 and blk._cache_write is None and all(not isinstance(st, _CacheWrite) for st in blk._stages)
    blk0, *_ = _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=0)
    assert blk0.paged_kv_page_size == 0 and blk0._cache_write is None
    blk_thd, *_ = _make_packed_block(_SMALL_D256, _PACKED_LENS)
    assert blk_thd.paged_kv_page_size == 0 and blk_thd._cache_write is None


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
    for kw in (dict(fuse_norm_rope=True), dict(inplace_qkv=False)):  # fuse_gate under thd is the block's own (unrelated) typed decline
        blk, *_ = _make_packed_block(_SMALL_D256, _PACKED_LENS, paged_kv_page_size=16, **kw)
        assert isinstance(blk._cache_write, _CacheWrite)
    blk16, *_ = _make_block(_SMALL_D256, 1, 64, dtype=torch.float16, paged_kv_page_size=16)
    assert blk16._cache_write.dtype == torch.float16
    blk16p, *_ = _make_packed_block(_SMALL_D256, _PACKED_LENS, dtype=torch.float16, paged_kv_page_size=16)
    assert blk16p._cache_write.dtype == torch.float16


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


def test_moved_bytes_counts_every_operand_once_in_and_once_out():
    from cudnn.gated_attention_block.kernels.cache_write import moved_bytes

    assert moved_bytes(1024, 2, 256, elem_bytes=2, n_ops=2, slot_bytes=4) == 2 * 2 * 1024 * 2 * 256 * 2 + 1024 * 4
    blk, *_ = _make_block(_SMALL_D256, 2, 64, paged_kv_page_size=16)
    t = 128
    assert blk._cache_write.moved_bytes() == 2 * 2 * t * 2 * 256 * 2 + t * 4
    assert blk._cache_write.moved_bytes(torch.int64) == 2 * 2 * t * 2 * 256 * 2 + t * 8


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


def test_a_declared_block_requires_the_pools_and_the_slot_mapping_and_declines_the_read_mode():
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
            with pytest.raises(NotImplementedError, match="paged-READ mode"):
                _exec(blk, inp, out, ws, **args, **read)
        # No indexer band on this geometry: a raw-key pool is refused, not ignored.
        with pytest.raises(ValueError, match="index_k_raw.*indexer band"):
            _exec(blk, inp, out, ws, **args, index_k_raw=args["k_cache"][:, 0])
        # The well-formed call reaches the plan check (the block is declared, not compiled).
        with pytest.raises(RuntimeError, match="compile"):
            _exec(blk, inp, out, ws, **args)


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


@_GPU
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
@pytest.mark.parametrize(
    "dtype, h, d, page_size, t, n_pages",
    [
        (torch.bfloat16, 2, 256, 16, 512, 40),
        (torch.float16, 2, 256, 32, 300, 12),
        (torch.bfloat16, 1, 128, 64, 700, 20),
        (torch.float16, 2, 256, 128, 520, 6),
    ],
    ids=["bf16_h2_d256_p16", "f16_h2_d256_p32", "bf16_h1_d128_p64", "f16_h2_d256_p128"],
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
@pytest.mark.parametrize("page_size", _PAGES)
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("form", ["dense", "packed"])
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


@requires_rubin
@pytest.mark.parametrize("pipeline", ["fuse_norm_rope", "out_of_place", "fuse_gate", "packed_fuse_norm_rope", "packed_out_of_place"])
def test_write_through_on_the_other_bf16_pipelines(pipeline):
    """Fused norm + RoPE (the slab is written post-RoPE by the GEMM fork, which norms the fp32 ACCUMULATOR -- so the
    pool's K is checked against the oracle norm + RoPE of the fp32 projection rounded ONCE, where the only legal
    difference is a one-ulp flip from the two accumulation orders: ``rtol 2^-7`` = one bf16 ulp at the bottom of a binade,
    ``atol 1e-4`` = the fp32 noise floor on near-zero values), out-of-place Q / K / V (the compact buffers are the SDPA's
    operands -- and the slab keeps the PRE-norm K, so the oracle norm of the block's OWN pre-norm K is checked at the norm
    suite's exact bar), and the fused gate (K / V untouched by it; dense only -- the packed block declines it on its own):
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
