# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's PACKED block-sparse EXTEND form (2026-10-30): ``thd=True`` + ``kv_source="paged"`` on a ``QsaSpec``
geometry with CALLER lists -- the sparse core's THD x PAGED x BOTTOM_RIGHT composition through the block.  ONE execute serves a mixed
serving step: decode rows beside prefill chunks of any length, in any sequence order, every sequence behind its cached prefix.  The
step's tokens are written through into the pools (the write-through stage, as on every packed form), then stage (4) is the sparse core
reading the pools: row ``j`` of sequence ``b`` at position ``kv_lens[b] - q_len_b + j`` (the serving stack's logical KV length AFTER the
step), one list per packed row (``block_ids [T, top_k]``, sequence-relative ids), unsplit (the packed form's claim-counter scheduler has
no split axis), the LSE written (``[1, H_q, T]``), the gate in the core's epilogue or its own launch.  ``execute(seq_lens=)`` is the Q
side (``[B_max]`` lengths or ``[B_max + 1]`` prefix sums), ``kv_lens`` the KV side, ``block_table`` exactly the declared ``[B_max,
max_pages]`` table (contiguous: the core binds the whole table); ``token_capacity`` optional -- the bounded form (the CAPACITY launch:
every stage over the ``T`` rows, the pad rows' slot ``-1``) is the serving contract, the live-count launch (``num_tokens_live``) where
the dense projections resolve a JIT tile.

Host cells (any device; the declaration cells even on the CPU): the declaration FACTS on every page / layout / dtype / Q-length form
(the sparse stage with ``thd`` and ``paged``, the per-row list, the tcgen05 body, split 1, no route, no indexer, the write-through in
position, the gate where declared); the static ``serving_envelope`` (``extend.served`` with ``sparse_core`` on a Rubin reading, the
appended tail ``index_sources`` / ``index_list_forms`` / ``sparse_tiles``, the LSE written, no PDL stage, the ``thd_paged_*`` rows
beside the table's; a dense geometry's record unchanged); every typed decline of the form raised on a REAL declaration with its text
EQUAL to the envelope's row (the step form of the indexer, the split, the lean tile, the shared list, PDL, the prepare fusion, the e4m3
pool, the in-block indexer; the table's record-driven ``sparse`` row through a record without the composition claim); the execute-time
form checks on an UNCOMPILED block under ``set_sync_debug_mode("error")``.

Rubin cells (``requires_rubin``): the P1 mixed step at test scale -- decode rows beside a chunk -- in ONE packed execute against the
TWO-RUN form (a dense ``[B_dec, S_q]`` paged QsaSpec block for the decodes and a dense ``[1, T]`` paged QsaSpec block for the chunk over
the SAME weights / pools / table rows / lists, the pools restored to their prefix-only state between runs): the chunk rows BITWISE
(``out``, the gated O, the LSE, the written pool rows) WHERE the control resolved ``decode_split_kv == 1`` (read back, never assumed) AND
its slab took the same projection route (both past the small-M bound), READING B otherwise -- the cell asserts the class the
resolution PREDICTS (a flip is asserted, never discovered) and, in a second arm, pins ``decode_split_kv=1`` on the control where bitwise
is the claim; the decode rows within READING B of the dense decode block; every row within the suite's bars of the fp32 oracle per
sequence (the dense block reference over the whole sequence gives post-norm / post-RoPE Q, K, V and the pre-sigmoid gate; the step
rows' attention through ``qsa_visible_mask`` at the per-row positions; ``cos(out) > 0.999``, the gated O within the repo's bf16 / f16
OUTPUT bound, the LSE within ``atol 2e-2`` -- the sparse modules' bars, never widened).  Then: any sequence order (every sequence's
rows bitwise across packings); the degenerate rows (``kv_lens 0`` with rows, ``q_len 0``, a sequence shorter than its chunk, every
prefix residue, prefix 0, ``-1`` table entries, stale FINITE pool rows bitwise the NaN-filled pools, the CAPACITY launch with pad rows
exactly 0 on the once-zeroed workspace, the live-count launch where the slab resolves a JIT tile); ``get_workspace_size()`` honest, no
allocation and no host sync on the execute path, the launch census by kernel NAME against the dense per-row control (+ the THD setup
launch, no combine, no memset / memcpy); CUDA-graph capture AFTER ONE eager warm-up on the capture stream (this form's capture rule),
the replay bitwise eager and over a NEW packing bitwise its eager twin; two launches bitwise with and without the LSE.
"""

import contextlib
import os
import sys
import types

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import (
    KV_SOURCE_PAGED,
    ExtendEnvelope,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    QsaSpec,
    index_k_raw_view,
)
from cudnn.gated_attention_block.api import _EXTEND_DECLINES, _cols, _qsa_thd_paged_decline_texts, _SparseSdpa, _view, sparse_decode_split_choice
from cudnn.gated_attention_block.kernels.proj_gemm import SMALL_M_MAX

requirement_error = cutedsl_requirement_error("the gated attention block's packed sparse extend tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_block_qsa_decode as _qsa  # noqa: E402  (the decode module's helpers: the pools, the oracle attention, the RoPE tables)
from gated_block_bounds import assert_output_close  # noqa: E402
from gated_block_census import cupti_kernels as _cupti_kernels  # noqa: E402
from gated_block_qsa_reference import qsa_visible_mask, random_block_ids  # noqa: E402
from gated_block_reference import RefGeometry, cu_seqlens_of, gated_attention_block_reference  # noqa: E402

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_SM107 = (10, 7)
BF16, F16 = torch.bfloat16, torch.float16
D = 256
ROPE = 64
BS = 4  # tokens per selectable block
COS_OUT = 0.999  # the end-to-end suite's bar on `out`
ATOL = 2e-2  # the sparse modules' bar on the LSE -- never widened here; the gated O is held to the repo's OUTPUT bound (gated_block_bounds)
SPARSE_TILE_TCGEN05 = "sparse_d256_f16"
GEOM_SMALL = dict(d_model=512, h_q=8, h_kv=2, d_head=D, rope_dim=ROPE)  # group 4; the slab N 5120 = 20 x 256: the forced tile past 128 rows
GEOM_FLASH_NEXT = dict(d_model=2560, h_q=24, h_kv=2, d_head=D, rope_dim=ROPE)  # Qwen3.8-Flash-Next TP 1 (group 12)
_THD_PAGED_ROWS = (
    "thd_paged_select_form",
    "thd_paged_split",
    "thd_paged_lean",
    "thd_paged_shared_list",
    "thd_paged_pdl",
    "thd_paged_fuse_prepare",
    "thd_paged_e4m3_pool",
    "thd_paged_indexer",
)

_no_host_sync, _bits, _cos, _sentinel, _pools, _rope_tables, _attend = (
    _qsa._no_host_sync,
    _qsa._bits,
    _qsa._cos,
    _qsa._sentinel,
    _qsa._pools,
    _qsa._rope_tables,
    _qsa._attend,
)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


def _geom(geom_kw, top_k=512, index_band=False, index_list_form=None, index_source="caller", **over):
    kw = dict(geom_kw, causal_bottom_right=True)
    kw.update(over)
    spec = QsaSpec(top_k=top_k, index_band=index_band, index_list_form=index_list_form, index_source=index_source)
    return GatedAttentionBlockGeometry(**kw, qsa=spec), RefGeometry(**{k: v for k, v in kw.items() if k in ("d_model", "h_q", "h_kv", "d_head", "rope_dim")})


def _row(name):
    return next(r for r in _EXTEND_DECLINES if r[0] == name)


def _dense_reference(h_full, w_qkvg_dense, w_q_norm, w_k_norm, cos_t, sin_t, w_o, ref_geom):
    """The dense block reference over ONE sequence's full history: its post-norm / post-RoPE ``q`` / ``k`` / ``v`` and pre-sigmoid ``gate``
    rows (the only outputs the step oracle reads; the attention is recomputed per row through the visible mask).  A ONE-token history is
    computed on two rows and sliced to the first: the per-row stages are bitwise independent of the second row, and the reference's own
    attention would otherwise issue a ``K = 1`` batched matmul that torch routes to a Triton outer-product kernel whose bundled ptxas does not
    know every part this suite runs on."""
    L = int(h_full.shape[1])
    if L >= 2:
        return gated_attention_block_reference(h_full, w_qkvg_dense, w_q_norm, w_k_norm, cos_t[:, :L], sin_t[:, :L], w_o, ref_geom)
    ref = gated_attention_block_reference(h_full.repeat(1, 2, 1), w_qkvg_dense, w_q_norm, w_k_norm, cos_t[:, :2], sin_t[:, :2], w_o, ref_geom)
    return types.SimpleNamespace(q=ref.q[:, :1], k=ref.k[:, :1], v=ref.v[:, :1], gate=ref.gate[:, :1])


@contextlib.contextmanager
def _declined_record(**over):
    """The sparse adapter's record with a claim DROPPED (``thd_paged_kv=False``: the composition this form needs), so the record-driven
    rows stay reachable (frost-engine-contract section 2)."""
    import dataclasses

    from cudnn.sdpa.fwd import sparse_gqa_sm107 as S

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "SPARSE_CAPABILITIES", dataclasses.replace(S.SPARSE_CAPABILITIES, **over))
        yield


def _declare(
    geom_kw,
    lens_q,
    *,
    page=16,
    layout="hnd",
    n_pages=8,
    max_pages=4,
    dtype=BF16,
    top_k=512,
    cu=False,
    token_capacity=None,
    table=None,
    pool_dtype=None,
    geom_over=None,
    index_band=False,
    index_list_form=None,
    index_source="caller",
    **blk_kw,
):
    """A declared (not compiled) PACKED sparse EXTEND block on EMPTY tensors: the declaration cells (any device).  ``lens_q`` is the
    packing (its sum the capacity unless ``token_capacity`` is given); returns ``(blk, aux)``."""
    geom, _ = _geom(geom_kw, top_k=top_k, index_band=index_band, index_list_form=index_list_form, index_source=index_source, **(geom_over or {}))
    T = sum(lens_q) if token_capacity is None else int(token_capacity)
    B = len(lens_q)
    mk = lambda *shape: torch.empty(*shape, dtype=dtype, device=_DEV)  # noqa: E731
    kc, vc = _pools(n_pages, geom.h_kv, page, layout, dtype if pool_dtype is None else pool_dtype)
    bt = torch.zeros(B, max_pages, dtype=torch.int32, device=_DEV) if table is None else table
    kw = dict(
        thd=True,
        num_sequences=B,
        max_seq_len=max(2, max(lens_q), -(-T // B)),  # the THD envelope: num_sequences x max_seq_len >= the capacity
        cu_seqlens=cu,
        token_capacity=token_capacity,
        paged_kv_page_size=page,
        kv_source=KV_SOURCE_PAGED,
        sample_k_cache=kc,
        sample_v_cache=vc,
        sample_block_table=bt,
    )
    kw.update(blk_kw)
    blk = GatedAttentionBlockFwd(
        mk(T, geom.d_model),
        mk(geom.n_qkvg, geom.d_model),
        torch.ones(D, dtype=dtype, device=_DEV),
        torch.ones(D, dtype=dtype, device=_DEV),
        mk(T, ROPE),
        mk(T, ROPE),
        mk(geom.d_model, geom.h_q * D),
        mk(T, geom.d_model),
        geom,
        **kw,
    )
    return blk, dict(k_cache=kc, v_cache=vc, block_table=bt, geom=geom, T=T, B=B)


# ---------------------------------------------------------------------------
# Host: the declaration facts and the envelope, the typed declines, the execute form checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fuse_gate", [False, True], ids=["gate_launch", "fuse_gate"])
@pytest.mark.parametrize("cu", [False, True], ids=["lens", "cu"])
def test_packed_sparse_extend_declaration_facts_and_the_envelope(fuse_gate, cu, monkeypatch):
    """A packed (``thd=True``) declaration on a bottom-right QsaSpec geometry over the pools CONSTRUCTS on every page / layout / dtype /
    Q-length form: stage (4) is the sparse core with ``thd`` and ``paged`` (unsplit, the per-row list, the tcgen05 body), no dense route,
    no KV bound, no indexer, the write-through right before it, the gate in the core's epilogue iff declared; the static envelope on a Rubin
    reading SERVES the EXTEND form with ``sparse_core`` and the appended tail (``index_sources`` / ``index_list_forms`` / ``sparse_tiles``),
    the LSE written, no PDL stage, the ``thd_paged_*`` rows beside the table's; the tail is append-only with defaults, and a dense
    geometry's record is unchanged."""
    import dataclasses

    from cudnn.frost import buffers

    with _no_host_sync():
        for page in (16, 32, 64, 128):
            for layout in ("hnd", "nhd", "unified"):
                for dtype in (BF16, F16):
                    blk, aux = _declare(GEOM_SMALL, [300, 0, 128, 84], page=page, layout=layout, dtype=dtype, cu=cu, fuse_gate=fuse_gate, return_lse=True)
                    sd = blk._sdpa
                    assert isinstance(sd, _SparseSdpa) and sd.thd and sd.paged and sd.split_kv == 1 and not sd.list_per_sequence, (page, layout, dtype)
                    assert sd.paged_page_size == page and sd.num_sequences == 4 and sd.cu_seqlens == cu and sd.fuse_gate == fuse_gate and sd.want_lse
                    assert blk.thd and blk.kv_source == KV_SOURCE_PAGED and (blk.batch, blk.seq_len) == (1, aux["T"]) and blk.token_capacity is None
                    assert blk.sparse_decode and blk.qsa_list_form == "per_row" and not blk.qsa_shared_list and not blk.indexer_decode
                    assert blk.decode_split_kv == 1 and blk.decode_route is None and blk.max_kv_len is None
                    assert blk.sparse_tile_form == "tcgen05" and blk.sparse_tile == SPARSE_TILE_TCGEN05 and blk.sparse_sub_tiles_per_cta is None
                    assert blk.use_pdl is False and blk._indexer is None and blk._indexer_decode is None
                    stages = list(blk._stages)
                    assert stages.index(blk._cache_write) == stages.index(sd) - 1, "the write-through sits right before the packed sparse SDPA"
                    assert (blk._gate is None) == fuse_gate and blk.gate_fused == fuse_gate
                    for nm in ("k_cache", "v_cache", "block_table"):
                        d = blk._descs[nm]
                        assert tuple(d.shape) == tuple(aux[nm].shape) and tuple(d.stride) == tuple(aux[nm].stride()), nm
        # the bounded form: the capacity is the samples' row count, the lengths may sum below it
        blk, aux = _declare(GEOM_SMALL, [300, 0, 128, 84], token_capacity=640, cu=cu, fuse_gate=fuse_gate)
        assert blk.token_capacity == 640 and (blk.batch, blk.seq_len) == (1, 640) and blk._sdpa.thd and blk._sdpa.paged
    # the envelope on a Rubin reading
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    geom, _ = _geom(GEOM_SMALL)
    for dtype in (BF16, F16):
        env = GatedAttentionBlockFwd.serving_envelope(geom, dtype, _DEV, device_cc=_SM107)
        ext = env.extend
        assert isinstance(ext, ExtendEnvelope) and ext.served and ext.reason is None and ext.sparse_core and env.forms == ("extend", "decode")
        assert ext.index_sources == ("caller",) and ext.index_list_forms == ("per_row",) and ext.sparse_tiles == (SPARSE_TILE_TCGEN05,)
        assert ext.writes_lse and ext.pdl_stages == () and ext.inference_only and ext.dtypes == (BF16, F16)
        assert set(ext.declines) == {name for name, *_ in _EXTEND_DECLINES} | set(_THD_PAGED_ROWS)
        req = "\n".join(ext.requires)
        for token in (
            "thd=True",
            "index_source='caller'",
            "token_capacity=None (the tight contract: the lengths sum to T) or T",
            "any sequence",
            "ONE eager warm-up",
            "decode_split_kv None",
        ):
            assert token in req, token
        assert env.decode.served and env.decode.sparse_core  # the DECODE mode of the same geometry is untouched
    dense = GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True), BF16, _DEV, device_cc=_SM107)
    assert dense.extend.served and not dense.extend.sparse_core and dense.extend.index_sources == () and dense.extend.index_list_forms == ()
    assert (
        dense.extend.sparse_tiles == ()
        and not dense.extend.writes_lse
        and dense.extend.pdl_stages
        and not any(k.startswith("thd_paged_") for k in dense.extend.declines)
    )
    assert set(dense.extend.declines) == {name for name, *_ in _EXTEND_DECLINES}
    # the tail is appended (the last four fields, every one with a default: an older reader of the record keeps working)
    fields = dataclasses.fields(ExtendEnvelope)
    assert [f.name for f in fields[-4:]] == ["sparse_core", "index_sources", "index_list_forms", "sparse_tiles"]
    assert [f.default for f in fields[-4:]] == [False, (), (), ()]
    # the record-driven table row: through a record without the composition claim the EXTEND form is declined with the `sparse` row
    with _declined_record(thd_paged_kv=False):
        declined = GatedAttentionBlockFwd.serving_envelope(geom, BF16, _DEV, device_cc=_SM107)
    assert declined.forms == ("decode",) and not declined.extend.served and declined.extend.reason == _row("sparse")[3] and declined.extend.sparse_core
    monkeypatch.undo()
    here = GatedAttentionBlockFwd.serving_envelope(geom, BF16, _DEV)
    assert here.extend.served is (_cc() == _SM107 and here.cutedsl_ok)


def test_packed_sparse_extend_typed_declines_equal_the_envelope_row_by_row(monkeypatch):
    """Every ``thd_paged_*`` row is reachable at declaration with EXACTLY the text the envelope publishes for it (one spelling: the
    constructor raises what ``serving_envelope`` lists), each a REAL declaration that differs from the served form in the one knob the row
    names; the table's ``sparse`` row through a record without the composition claim; the DECODE form's facts refused on the packed form;
    ``return_lse`` served (no row)."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    geom, _ = _geom(GEOM_SMALL)
    env = GatedAttentionBlockFwd.serving_envelope(geom, BF16, _DEV, device_cc=_SM107)
    texts = _qsa_thd_paged_decline_texts(geom, BF16)
    assert set(texts) == set(_THD_PAGED_ROWS) and all(env.extend.declines[k] == texts[k] for k in texts)
    lens = [300, 0, 128, 84]
    e4m3 = torch.float8_e4m3fn
    trips = {
        "thd_paged_select_form": (ValueError, lambda: _declare(GEOM_SMALL, lens, index_select_form="extend")),
        "thd_paged_split": (ValueError, lambda: _declare(GEOM_SMALL, lens, decode_split_kv=1)),
        "thd_paged_lean": (NotImplementedError, lambda: _declare(GEOM_SMALL, lens, sparse_tile_form="lean")),
        "thd_paged_shared_list": (NotImplementedError, lambda: _declare(GEOM_SMALL, lens, index_list_form="shared")),
        "thd_paged_pdl": (NotImplementedError, lambda: _declare(GEOM_SMALL, lens, use_pdl=True)),
        "thd_paged_fuse_prepare": (NotImplementedError, lambda: _declare(GEOM_SMALL, lens, fuse_prepare=True, inplace_qkv=True)),
        "thd_paged_e4m3_pool": (NotImplementedError, lambda: _declare(GEOM_SMALL, lens, pool_dtype=e4m3, kv_cache_dtype=e4m3)),
        "thd_paged_indexer": (NotImplementedError, lambda: _declare(GEOM_SMALL, lens, index_band=True, index_source="indexer")),
    }
    assert set(trips) == set(_THD_PAGED_ROWS)
    with _no_host_sync():
        for name, (exc, trip) in trips.items():
            with pytest.raises(exc) as e:
                trip()
            assert str(e.value) == texts[name] == env.extend.declines[name], name
        with pytest.raises(ValueError) as e:
            _declare(GEOM_SMALL, lens, decode_split_kv=4)
        assert str(e.value) == texts["thd_paged_split"]
        # the table's `sparse` row: record-driven (with the claim the declaration constructs; without it the row, its text the envelope's)
        name, exc, _pred, msg = _row("sparse")
        with _declined_record(thd_paged_kv=False):
            with pytest.raises(exc) as e:
                _declare(GEOM_SMALL, lens)
        assert str(e.value) == msg == env.extend.declines["sparse"] and "thd_paged_kv" in msg
        # the DECODE form's own facts are refused on the packed form (max_kv_len: the dense form's KV bound), the existing row
        with pytest.raises(ValueError, match="the DECODE form's facts"):
            _declare(GEOM_SMALL, lens, max_kv_len=64)
        # the top-left QsaSpec geometry meets the `diagonal` row like a dense one; the LSE is served; the e4m3 pools on a DENSE geometry keep
        # their own (attribute's) row -- the thd_paged_e4m3_pool text is the QsaSpec geometry's spelling
        with pytest.raises(ValueError) as e:
            _declare(GEOM_SMALL, lens, geom_over=dict(causal_bottom_right=False))
        assert str(e.value) == env.extend.declines["diagonal"]
        blk, _ = _declare(GEOM_SMALL, lens, return_lse=True)
        assert blk._sdpa.want_lse


def test_packed_sparse_extend_execute_form_checks():
    """On an UNCOMPILED block (a declaration-vs-argument contract: form only, no plan, no sync): ``kv_lens`` exactly ``[B]``,
    ``block_table`` EXACTLY the declared table (a wider table, a row slice of a wider table and a table of another row count refused),
    ``block_ids`` ``[T, top_k]`` or ``[1, T, top_k]`` (the decode mode's shared ``[B, top_k]`` spelling refused; int64 refused, never
    converted), ``slot_mapping`` ``[T]`` (or ``[T_live]`` under the live count), the pools exactly the samples, the live count's rows; every
    form check passed, the block stops at its own compile gate.  The packed LENGTHS' rows (``seq_lens`` REQUIRED in the declared form, another
    count refused) sit behind that gate on every THD form and are pinned on a compiled block by the Rubin cell
    ``test_packed_sparse_extend_length_rows_on_a_compiled_block``."""
    if not torch.cuda.is_available():
        pytest.skip("the execute form checks need CUDA tensors (the device is never touched)")
    lens = [2, 3, 3]
    with _no_host_sync():
        blk, aux = _declare(GEOM_SMALL, lens, max_pages=4, n_pages=8, token_capacity=16)
        cu_blk, _ = _declare(GEOM_SMALL, lens, max_pages=4, n_pages=8, cu=True)
    geom, T = aux["geom"], 16
    i32 = lambda *shape: torch.zeros(*shape, dtype=torch.int32, device=_DEV)  # noqa: E731
    z = lambda *shape: torch.zeros(*shape, dtype=BF16, device=_DEV)  # noqa: E731

    def run(b, t, *, ids=None, lens_=None, kv_lens=None, table=None, seq_lens="ok", slot=None, k_cache=None, live=None):
        return b.execute(
            z(t, geom.d_model),
            z(geom.n_qkvg, geom.d_model),
            torch.ones(D, dtype=BF16, device=_DEV),
            torch.ones(D, dtype=BF16, device=_DEV),
            z(t, ROPE),
            z(t, ROPE),
            z(geom.d_model, geom.h_q * D),
            z(t, geom.d_model),
            torch.empty(1 << 20, dtype=torch.uint8, device=_DEV),
            seq_lens=(i32(3) if b is blk else i32(4)) if seq_lens == "ok" else seq_lens,
            block_ids=i32(t, 512) if ids is None else ids,
            block_lens=lens_,
            k_cache=aux["k_cache"] if k_cache is None else k_cache,
            v_cache=aux["v_cache"],
            block_table=aux["block_table"] if table is None else table,
            kv_lens=i32(3) if kv_lens is None else kv_lens,
            slot_mapping=i32(t) if slot is None else slot,
            **({} if live is None else dict(num_tokens_live=live)),
        )

    with _no_host_sync():
        with pytest.raises(ValueError, match="kv_lens must be a contiguous 1-D int32 CUDA tensor of 3 entries"):
            run(blk, T, kv_lens=i32(4))
        with pytest.raises(ValueError, match="block_table must be exactly the declared sample_block_table"):
            run(blk, T, table=i32(3, 6))  # wider than the declared table: the sparse core binds exactly (B, max_pages)
        with pytest.raises(ValueError, match="block_table must be exactly the declared sample_block_table"):
            run(blk, T, table=i32(3, 8)[:, :4])  # a row slice of a wider table: another row stride
        with pytest.raises(ValueError, match="block_table must be"):
            run(blk, T, table=i32(1, 4))  # Q's batch extent, never the row count
        with pytest.raises(ValueError, match=r"block_ids must be \[16, 512\] or \[1, 16, 512\]"):
            run(blk, T, ids=i32(3, 512))  # the decode mode's shared spelling on the packed per-row form
        with pytest.raises(ValueError, match="int64"):
            run(blk, T, ids=torch.zeros(T, 512, dtype=torch.int64, device=_DEV))
        with pytest.raises(ValueError, match=r"block_lens must be \[16\]"):
            run(blk, T, lens_=i32(3))
        with pytest.raises(ValueError, match="slot_mapping"):
            run(blk, T, slot=i32(T + 1))
        with pytest.raises(ValueError, match="k_cache must match sample_k_cache exactly"):
            run(blk, T, k_cache=aux["k_cache"][:7])
        with pytest.raises(ValueError, match="num_tokens_live=17 exceeds the declared token_capacity=16"):
            run(blk, T, live=17)
        with pytest.raises(ValueError, match="num_tokens_live=8 belongs to a block declared with token_capacity"):
            run(cu_blk, 8, live=8)
        with pytest.raises(RuntimeError, match="compile"):
            run(blk, T)  # every form check passed: the uncompiled block stops at its own gate
        with pytest.raises(RuntimeError, match="compile"):
            run(cu_blk, 8)


# ---------------------------------------------------------------------------
# Rubin: the packed step against the two-run form and the oracle, the orders, the degenerate rows, the workspace, the capture
# ---------------------------------------------------------------------------


class _PackedCase:
    """One PACKED serving step over the pools: ``seqs`` = ``[(q_len, kv_len), ...]`` per SEQUENCE ID (``kv_len`` the logical length AFTER the
    step; ``kv_len - q_len`` the cached prefix, negative = a sequence shorter than its chunk whose first rows are dead; ``kv_len 0`` a dead
    entry whose rows -- if any -- are dead), packed in ``order`` (a permutation of the ids; the block's sequence index is the PACKING
    position: the table row, ``kv_lens[p]``, ``seq_lens[p]``).  Every sequence's data is drawn from ITS OWN generator (seeded by the id), so
    two cases that differ in ``order`` alone carry the same bytes per sequence.  The whole-sequence inputs per id give the dense block
    reference over the full history (post-norm / post-RoPE Q, K, V, the pre-sigmoid gate): the prefix rows fill the NaN pools, the step's
    rows are the packed ``h`` / RoPE rows at their positions, the per-row lists at the bottom-right positions, the oracle of ``out`` /
    the gated O / the LSE per live row; the step's tokens are written through by the block at ``slot_mapping``."""

    def __init__(
        self,
        geom_kw,
        seqs,
        *,
        order=None,
        page=16,
        layout="hnd",
        dtype=BF16,
        seed=0,
        top_k=512,
        block_lens=True,
        return_lse=True,
        fuse_gate=True,
        fuse_norm_rope=False,
        inplace_qkv=None,
        cu=False,
        token_capacity=None,
        n_pages=None,
        max_pages=None,
        index_band=False,
        shuffle=True,
        **blk_kw,
    ):
        self.seqs = [(int(q), int(kv)) for q, kv in seqs]
        B = len(self.seqs)
        self.order = list(range(B)) if order is None else [int(x) for x in order]
        assert sorted(self.order) == list(range(B))
        self.page, self.layout, self.dtype, self.top_k, self.seed, self.cu = page, layout, dtype, top_k, seed, cu
        self.block_lens_given, self.return_lse, self.fuse_gate, self.fuse_norm_rope, self.index_band = (
            bool(block_lens),
            bool(return_lse),
            bool(fuse_gate),
            bool(fuse_norm_rope),
            bool(index_band),
        )
        self.geom, self.ref_geom = _geom(geom_kw, top_k=top_k, index_band=index_band)
        geom, ref_geom = self.geom, self.ref_geom
        h_q, h_kv, d_model = geom.h_q, geom.h_kv, geom.d_model
        G = h_q // h_kv
        q_lens = [q for q, _ in self.seqs]
        kv_lens = [kv for _, kv in self.seqs]
        self.T_live = sum(q_lens)
        self.T_cap = self.T_live if token_capacity is None else int(token_capacity)
        assert self.T_cap >= self.T_live
        self.token_capacity = token_capacity
        gw = torch.Generator(device=_DEV).manual_seed(seed)

        def randn(g, *shape, std=0.02):
            return (torch.randn(*shape, generator=g, device=_DEV, dtype=torch.float32) * std).to(dtype)

        n_dense = sum(geom.qkvg_block_widths[:4])
        w_qkvg = randn(gw, geom.n_qkvg, d_model)
        self.w = dict(
            w_qkvg=w_qkvg, w_q_norm=torch.ones(D, dtype=dtype, device=_DEV), w_k_norm=torch.ones(D, dtype=dtype, device=_DEV), w_o=randn(gw, d_model, h_q * D)
        )
        L_max = max(max(kv_lens), max(q_lens), 1)
        cos_t, sin_t = _rope_tables(L_max, dtype, ref_geom)
        scale = geom.scale
        T_cap = self.T_cap
        self.h = torch.zeros(T_cap, d_model, dtype=dtype, device=_DEV)
        self.cos = cos_t[0, :1].expand(T_cap, ROPE).clone()
        self.sin = sin_t[0, :1].expand(T_cap, ROPE).clone()
        self.ref_out = torch.zeros(T_cap, d_model, dtype=dtype, device=_DEV)
        self.ref_o_gated = torch.zeros(T_cap, h_q, D, dtype=dtype, device=_DEV)
        self.ref_lse = torch.full((h_q, T_cap), float("-inf"), dtype=torch.float32, device=_DEV)
        self.live_rows = torch.zeros(T_cap, dtype=torch.bool, device=_DEV)
        self.block_ids = torch.full((T_cap, top_k), -1, dtype=torch.int32, device=_DEV)
        self.block_lens = torch.zeros(T_cap, dtype=torch.int32, device=_DEV)
        self.slot_mapping = torch.full((T_cap,), -1, dtype=torch.int32, device=_DEV)
        # the packing: rows per packing position, the per-position lengths the block reads
        self.rows = []
        lo = 0
        for p, b in enumerate(self.order):
            self.rows.append((lo, lo + q_lens[b]))
            lo += q_lens[b]
        self.rows_by_id = {b: self.rows[p] for p, b in enumerate(self.order)}
        self.q_lens_packed = [q_lens[b] for b in self.order]
        self.kv_lens_packed = [kv_lens[b] for b in self.order]
        # pages by sequence ID (order-independent): block 0 reserved (NULL), each live sequence ceil(L / P) pages from a permutation
        need = sum(-(-L // page) for L in kv_lens)
        self.n_pages = n_pages if n_pages is not None else need + 1 + 2
        assert self.n_pages >= need + 1, (self.n_pages, need)
        perm = (torch.randperm(self.n_pages - 1, generator=gw, device=_DEV) + 1).tolist()
        it = iter(perm)
        self.pages = [[next(it) for _ in range(-(-L // page))] for L in kv_lens]
        self.max_pages = max_pages if max_pages is not None else max(-(-max(kv_lens) // page), 1)
        self.block_table = torch.full((B, self.max_pages), -1, dtype=torch.int32, device=_DEV)
        self.k_cache, self.v_cache = _pools(self.n_pages, h_kv, page, layout, dtype)
        self.index_cache = torch.full((self.n_pages, page, geom.qsa.index_head_dim), float("nan"), dtype=dtype, device=_DEV) if index_band else None
        self.ref_rows = {}
        for p, b in enumerate(self.order):
            q, L = q_lens[b], kv_lens[b]
            lo, hi = self.rows[p]
            gb = torch.Generator(device=_DEV).manual_seed(seed * 7919 + 17 * b + 3)
            if L == 0:
                self.block_table[p] = 0  # the dead entry: the NULL block, slots -1, every row dead
                if q:
                    self.h[lo:hi] = randn(gb, q, d_model, std=1.0)  # finite inputs on dead rows
                continue
            pages = self.pages[b]
            self.block_table[p, : len(pages)] = torch.tensor(pages, dtype=torch.int32, device=_DEV)
            h_full = randn(gb, 1, L, d_model, std=1.0)
            ref = _dense_reference(h_full, w_qkvg[:n_dense], self.w["w_q_norm"], self.w["w_k_norm"], cos_t, sin_t, self.w["w_o"], ref_geom)
            self.ref_rows[b] = ref
            pos = torch.arange(q, device=_DEV) + (L - q)
            live = pos >= 0
            n_prefix = max(L - q, 0)
            if n_prefix:  # the PREFIX: the oracle's K / V rows into the pools
                s_ = torch.tensor([pages[t // page] * page + t % page for t in range(n_prefix)], device=_DEV)
                self.k_cache[s_ // page, :, s_ % page] = ref.k[0, :n_prefix]
                self.v_cache[s_ // page, :, s_ % page] = ref.v[0, :n_prefix]
            if q == 0:
                continue
            self.live_rows[lo:hi] = live
            p_live = pos[live]
            idx = lo + torch.nonzero(live).flatten()
            self.h[lo:hi] = randn(gb, q, d_model, std=1.0)  # the dead rows (a negative position) keep finite random inputs
            self.h[idx] = h_full[0, p_live]
            self.cos[idx] = cos_t[0, p_live]
            self.sin[idx] = sin_t[0, p_live]
            self.slot_mapping[idx] = torch.tensor([pages[t // page] * page + t % page for t in p_live.tolist()], dtype=torch.int32, device=_DEV)
            ids_b, lens_b = random_block_ids(pos, top_k, BS, generator=gb, kv_lens=torch.tensor([L], device=_DEV).expand(q), shuffle=shuffle)
            self.block_ids[lo:hi] = ids_b
            self.block_lens[lo:hi] = lens_b
            if live.any():
                allowed = qsa_visible_mask(ids_b, lens_b if block_lens else None, pos, L, L, BS, top_k=top_k)[live]
                og, lse = _attend(ref, allowed, p_live, scale, dtype, h_q, G)
                self.ref_o_gated[idx] = og
                self.ref_lse[:, idx] = lse
                self.ref_out[idx] = (og.reshape(-1, h_q * D).float() @ self.w["w_o"].float().t()).to(dtype)
        self.kv_lens_t = torch.tensor(self.kv_lens_packed, dtype=torch.int32, device=_DEV)
        self.seq_lens_t = torch.tensor(cu_seqlens_of(self.q_lens_packed) if cu else self.q_lens_packed, dtype=torch.int32, device=_DEV)
        self.prefix_bits = (_bits(self.k_cache).clone(), _bits(self.v_cache).clone())
        self.k_prefix, self.v_prefix = self.k_cache.clone(), self.v_cache.clone()
        kw = dict(
            thd=True,
            num_sequences=B,
            max_seq_len=max(2, max(q_lens), -(-T_cap // B)),  # the THD envelope: num_sequences x max_seq_len >= the capacity
            cu_seqlens=cu,
            token_capacity=token_capacity,
            paged_kv_page_size=page,
            kv_source=KV_SOURCE_PAGED,
            sample_k_cache=self.k_cache,
            sample_v_cache=self.v_cache,
            sample_block_table=self.block_table,
            fuse_gate=fuse_gate,
            fuse_norm_rope=fuse_norm_rope,
            inplace_qkv=inplace_qkv,
            return_lse=return_lse,
        )
        kw.update(blk_kw)
        self.blk_kw = kw
        self.out = torch.empty(T_cap, d_model, dtype=dtype, device=_DEV)
        self.lse = torch.empty(1, h_q, T_cap, dtype=torch.float32, device=_DEV) if return_lse else None
        self.blk = GatedAttentionBlockFwd(
            self.h, self.w["w_qkvg"], self.w["w_q_norm"], self.w["w_k_norm"], self.cos, self.sin, self.w["w_o"], self.out, geom, **kw
        )
        assert isinstance(self.blk._sdpa, _SparseSdpa) and self.blk._sdpa.thd and self.blk._sdpa.paged and self.blk.decode_split_kv == 1
        assert self.blk.qsa_list_form == "per_row" and self.blk.sparse_tile == SPARSE_TILE_TCGEN05 and (self.blk.batch, self.blk.seq_len) == (1, T_cap)
        self.ws = None

    # -- running ---------------------------------------------------------------

    def compile(self, ws_fill=None):
        """``check_support`` + ``compile``; the workspace zeroed ONCE (the serving convention under ``token_capacity``: a pad row's ``o`` reads 0
        forever) unless ``ws_fill`` names another byte."""
        assert self.blk.check_support()
        self.blk.compile()
        n = self.blk.get_workspace_size()
        self.ws = torch.zeros(n, dtype=torch.uint8, device=_DEV) if ws_fill is None else torch.full((n,), ws_fill, dtype=torch.uint8, device=_DEV)
        return self

    def execute(self, *, out=None, ws=None, lse=None, stream=None, num_tokens_live=None, live_rows_inputs=False):
        n = num_tokens_live
        if live_rows_inputs:
            assert n is not None
            h, cos, sin, slot, out_ = self.h[:n], self.cos[:n], self.sin[:n], self.slot_mapping[:n], (self.out if out is None else out)[:n]
        else:
            h, cos, sin, slot, out_ = self.h, self.cos, self.sin, self.slot_mapping, (self.out if out is None else out)
        self.blk.execute(
            h,
            self.w["w_qkvg"],
            self.w["w_q_norm"],
            self.w["w_k_norm"],
            cos,
            sin,
            self.w["w_o"],
            out_,
            self.ws if ws is None else ws,
            seq_lens=self.seq_lens_t,
            lse=(self.lse if lse is None else lse) if self.return_lse else None,
            block_ids=self.block_ids,
            block_lens=self.block_lens if self.block_lens_given else None,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            block_table=self.block_table,
            kv_lens=self.kv_lens_t,
            slot_mapping=slot,
            index_k_raw=self.index_cache,
            current_stream=stream,
            **({} if n is None else dict(num_tokens_live=n)),
        )
        return self.out if out is None else out

    def run(self):
        if self.ws is None:
            self.compile()
        self.out.fill_(_sentinel(self.dtype))
        if self.lse is not None:
            self.lse.fill_(_sentinel(self.dtype))
        self.execute()
        torch.cuda.synchronize()
        return self.out

    def restore_pools(self):
        """The pools back to their prefix-only state (the two-run control re-writes the step's rows itself)."""
        self.k_cache.copy_(self.k_prefix)
        self.v_cache.copy_(self.v_prefix)
        if self.index_cache is not None:
            self.index_cache.fill_(float("nan"))

    # -- views -----------------------------------------------------------------

    def o_gated(self, ws=None):
        g, lay = self.geom, self.blk._layout()
        return _view(self.ws if ws is None else ws, lay.o, (self.T_cap, g.h_q, g.d_head), self.dtype)

    def slab_rows(self, ws=None):
        """The post-RoPE K / V rows attention consumed, ``[T_cap, H_kv, D]`` views of the workspace (the slab bands in place, the compact
        buffers out of place), and the slab itself."""
        g, t, lay = self.geom, self.T_cap, self.blk._layout()
        ws = self.ws if ws is None else ws
        proj = _view(ws, lay.proj, (t, g.n_qkvg), self.dtype)
        if not self.blk.inplace_qkv:
            return _view(ws, lay.k, (t, g.h_kv, g.d_head), self.dtype), _view(ws, lay.v, (t, g.h_kv, g.d_head), self.dtype), proj
        return _cols(proj, g.qkvg_offsets[2], g.h_kv, g.d_head), _cols(proj, g.qkvg_offsets[3], g.h_kv, g.d_head), proj

    def label(self):
        return (
            f"[{self.geom.h_q}/{self.geom.h_kv} packed q={self.q_lens_packed} kv={self.kv_lens_packed} cap={self.T_cap} p{self.page} {self.layout} "
            f"{self.dtype} gate={'fused' if self.fuse_gate else 'launch'} lse={self.return_lse} cu={self.cu}]"
        )

    # -- the checks -------------------------------------------------------------

    def check(self, out=None, ws=None, lse=None, *, what="out"):
        """The standard assertions over the packed rows: finite, no sentinel on a live row, dead rows exact (``kv_len 0`` or a negative position:
        ``out = 0``, ``O = 0``, ``LSE = -inf``), within the budget vs the oracle (``cos(out) > 0.999`` over the live rows, the gated O slot within
        the OUTPUT bound, the LSE within ``atol 2e-2``), the pools' step rows BITWISE the slab's K / V bands at ``slot_mapping``, the prefix rows
        and the NULL block untouched; the raw indexer key's pool rows bitwise the slab's band where the geometry declares the band."""
        out = self.out if out is None else out
        ws = self.ws if ws is None else ws
        lse = self.lse if lse is None else lse
        sent = _sentinel(self.dtype)
        sent_stored = float(torch.full((), sent, dtype=self.dtype).float())  # the sentinel AS THE OUTPUT DTYPE STORES IT (1.5e30 is not a bf16 value)
        T = self.T_live
        live = self.live_rows[:T]
        dead = ~live
        k_rows, v_rows, proj = self.slab_rows(ws)
        o = self.o_gated(ws)
        assert torch.isfinite(out[:T].float()).all(), f"{what}: non-finite out"
        assert not (out[:T][live].float() == sent_stored).any(), f"{what}: a sentinel survived on a live row of out"
        assert torch.isfinite(o[:T].float()).all(), f"{what}: non-finite gated O (a row the sparse core never wrote keeps the workspace fill)"
        if dead.any():
            assert (out[:T][dead] == 0).all() and (o[:T][dead] == 0).all(), f"{what}: a dead row's out and O must be exactly 0"
        c = _cos(out[:T][live], self.ref_out[:T][live])
        d_out = float((out[:T][live].float() - self.ref_out[:T][live].float()).abs().max())
        d_o = float((o[:T][live].float() - self.ref_o_gated[:T][live].float()).abs().max())
        d_lse = 0.0
        if self.return_lse:
            assert lse is not None and not (lse[0, :, :T][:, live] == sent).any(), f"{what}: a sentinel survived in LSE"
            if dead.any():
                assert torch.isinf(lse[0, :, :T][:, dead]).all() and (lse[0, :, :T][:, dead] < 0).all(), f"{what}: a dead row's LSE must be -inf exactly"
            d_lse = float((lse[0, :, :T][:, live] - self.ref_lse[:, :T][:, live]).abs().max())
            assert d_lse <= ATOL, f"{what}: LSE off the oracle by {d_lse} (budget {ATOL})"
        assert c > COS_OUT, f"{what}: cos(out) {c} <= {COS_OUT}"
        ob = assert_output_close(o[:T][live], self.ref_o_gated[:T][live], f"{self.label()} {what}: gated O vs the oracle")
        # the write-through: the step's live rows are the slab's post-RoPE K / V rows, bitwise; nothing else moved
        slot = self.slot_mapping
        lv = slot >= 0
        s_ = slot[lv].long()
        assert torch.equal(_bits(self.k_cache[s_ // self.page, :, s_ % self.page, :]), _bits(k_rows[lv])), f"{what}: the K pool rows are not the slab's rows"
        assert torch.equal(_bits(self.v_cache[s_ // self.page, :, s_ % self.page, :]), _bits(v_rows[lv])), f"{what}: the V pool rows are not the slab's rows"
        written = set((s_ // self.page * self.page + s_ % self.page).tolist())
        for pool, before in zip((self.k_cache, self.v_cache), self.prefix_bits):
            touched = (_bits(pool) != before).any(dim=-1).any(dim=1)  # [n_pages, page]
            moved = touched.nonzero()
            assert all(int(pg) * self.page + int(off) in written for pg, off in moved), f"{what}: a pool row outside the step's slots moved"
            assert not touched[0].any(), f"{what}: the NULL block moved"
        if self.index_cache is not None:
            raw = index_k_raw_view(proj, self.geom, 1, self.T_cap)
            assert torch.equal(_bits(self.index_cache[s_ // self.page, s_ % self.page, :]), _bits(raw.reshape(-1, raw.shape[-1])[lv]))
        print(
            f"\n{self.label()} {what}: cos(out) {c:.6f} max|d out| {d_out:.4e} max|d O_gated| {d_o:.4e} ({ob['bound_ratio']:.3f} of the output bound) "
            f"max|d LSE| {d_lse:.2e} dead rows {int(dead.sum())} live rows {int(live.sum())}"
        )
        return dict(cos=c, d_out=d_out, d_o=d_o, d_lse=d_lse, dead=int(dead.sum()), o_bound_ratio=ob["bound_ratio"])


def _two_launches(case: _PackedCase, what: str):
    a = case.run().clone()
    o_a = case.o_gated().clone()
    lse_a = None if case.lse is None else case.lse.clone()
    b = case.run().clone()
    o_b = case.o_gated().clone()
    assert torch.equal(_bits(a), _bits(b)) and torch.equal(_bits(o_a), _bits(o_b)), f"{what}: two launches are not bitwise"
    if lse_a is not None:
        assert torch.equal(lse_a, case.lse), f"{what}: two launches are not bitwise on LSE"
    return case.check(what=what)


def _dense_control(case: _PackedCase, members, *, split=None, index_list_form="per_row", sparse_tile_form=None):
    """The TWO-RUN form's control for the packing positions ``members`` (all of one Q length ``S``): a DENSE ``[B_g, S]`` paged QsaSpec block
    over the SAME weights, pools (restored to the prefix-only state first), table rows (a contiguous copy), lists, lengths and slots, with
    the per-row list form (the packed form's) and ``decode_split_kv=split`` (``None`` = the fill rule the block resolves; read back by the
    caller, never assumed).  Returns ``dict(blk, out [B_g, S, d_model], o [B_g, S, H_q, D], lse [B_g, H_q, S] | None, rows)``."""
    g = case.geom
    S = {case.q_lens_packed[p] for p in members}
    assert len(S) == 1, "a control group shares one Q length"
    S = S.pop()
    rows = [case.rows[p] for p in members]
    B_g = len(members)
    geom_c, _ = _geom(
        dict(d_model=g.d_model, h_q=g.h_q, h_kv=g.h_kv, d_head=g.d_head, rope_dim=g.rope_dim),
        top_k=case.top_k,
        index_band=case.index_band,
        index_list_form=index_list_form,
    )
    h = torch.stack([case.h[lo:hi] for lo, hi in rows])
    cos = torch.stack([case.cos[lo:hi] for lo, hi in rows])
    sin = torch.stack([case.sin[lo:hi] for lo, hi in rows])
    ids = torch.stack([case.block_ids[lo:hi] for lo, hi in rows]).contiguous()
    lens = torch.stack([case.block_lens[lo:hi] for lo, hi in rows]).contiguous()
    table = case.block_table[members].contiguous()
    kv = case.kv_lens_t[members].contiguous()
    slot = torch.cat([case.slot_mapping[lo:hi] for lo, hi in rows]).contiguous()
    out = torch.empty(B_g, S, g.d_model, dtype=case.dtype, device=_DEV)
    lse = torch.empty(B_g, g.h_q, S, dtype=torch.float32, device=_DEV) if case.return_lse else None
    kw = dict(
        paged_kv_page_size=case.page,
        kv_source=KV_SOURCE_PAGED,
        sample_k_cache=case.k_cache,
        sample_v_cache=case.v_cache,
        sample_block_table=table,
        fuse_gate=case.fuse_gate,
        fuse_norm_rope=case.fuse_norm_rope,
        return_lse=case.return_lse,
        decode_split_kv=split,
    )
    if sparse_tile_form is not None:
        kw["sparse_tile_form"] = sparse_tile_form  # the body pinned (the census control); None = the block's "auto" as a stack would declare it
    blk = GatedAttentionBlockFwd(h, case.w["w_qkvg"], case.w["w_q_norm"], case.w["w_k_norm"], cos, sin, case.w["w_o"], out, geom_c, **kw)
    assert isinstance(blk._sdpa, _SparseSdpa) and not blk._sdpa.thd and blk._sdpa.paged and not blk.qsa_shared_list
    assert blk.check_support()
    blk.compile()
    ws = torch.zeros(blk.get_workspace_size(), dtype=torch.uint8, device=_DEV)
    case.restore_pools()
    out.fill_(_sentinel(case.dtype))
    if lse is not None:
        lse.fill_(_sentinel(case.dtype))
    blk.execute(
        h,
        case.w["w_qkvg"],
        case.w["w_q_norm"],
        case.w["w_k_norm"],
        cos,
        sin,
        case.w["w_o"],
        out,
        ws,
        lse=lse,
        block_ids=ids,
        block_lens=lens if case.block_lens_given else None,
        k_cache=case.k_cache,
        v_cache=case.v_cache,
        block_table=table,
        kv_lens=kv,
        slot_mapping=slot,
        index_k_raw=case.index_cache,
    )
    torch.cuda.synchronize()
    lay = blk._layout()
    o = _view(ws, lay.o, (B_g * S, g.h_q, g.d_head), case.dtype).view(B_g, S, g.h_q, g.d_head).clone()
    k_pool_rows = {}
    for p, (lo, hi) in zip(members, rows):
        sl = case.slot_mapping[lo:hi]
        lv = sl >= 0
        s_ = sl[lv].long()
        k_pool_rows[p] = (
            _bits(case.k_cache[s_ // case.page, :, s_ % case.page, :]).clone(),
            _bits(case.v_cache[s_ // case.page, :, s_ % case.page, :]).clone(),
        )
    return dict(blk=blk, out=out.clone(), o=o, lse=None if lse is None else lse.clone(), rows=rows, members=members, pool_rows=k_pool_rows)


def _same_slab_route(a: GatedAttentionBlockFwd, b: GatedAttentionBlockFwd) -> bool:
    """Whether the two blocks' dense projections resolve the SAME GEMM route (the small-M route at ``M <= SMALL_M_MAX``, else the forced
    256-wide tile where 256 divides N; the graph route otherwise): the per-row GEMM arithmetic is bitwise across M only on one route."""
    ja, jb = a._proj.resolves_a_jit_tile(), b._proj.resolves_a_jit_tile()
    return ja and jb and ((a._proj.m <= SMALL_M_MAX) == (b._proj.m <= SMALL_M_MAX))


def _parity_class(case: _PackedCase, ctl) -> str:
    """The class the RESOLUTION predicts for a control group: ``"bitwise"`` iff the control runs the TCGEN05 core unsplit (its resolved body
    and ``decode_split_kv`` read back; the lean tile the measured part picks for one-row sequences is another arithmetic) AND its slab took
    the packed block's projection route; ``"reading_b"`` otherwise."""
    c = ctl["blk"]
    return "bitwise" if (c.decode_split_kv == 1 and c.sparse_tile == SPARSE_TILE_TCGEN05 and _same_slab_route(case.blk, c)) else "reading_b"


def _predicted_split(case: _PackedCase, ctl, n_b: int, q: int) -> int:
    """The split the block's rule predicts for the control's RESOLVED body at this device's SM count (the tcgen05 fill / wave rule, or the lean
    split of the resolved ``k``), never a hard-coded number."""
    c = ctl["blk"]
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    return sparse_decode_split_choice(
        batch=n_b,
        h_kv=case.geom.h_kv,
        s_q=q,
        top_k=case.top_k,
        sm_count=sm,
        tile_form=c.sparse_tile_form,
        sub_tiles_per_cta=c.sparse_sub_tiles_per_cta,
        group=case.geom.gqa_ratio,
    )


def _assert_parity(case: _PackedCase, ctl, cls: str, what: str, *, packed=None):
    """The packed rows of every member against the control's: BITWISE (``out``, the gated O, the LSE, the written pool rows) or READING B
    (``cos(out) > 0.999``, the gated O within the OUTPUT bound of the control's, the LSE within ``atol 2e-2``)."""
    out, o, lse = packed if packed is not None else (case.out, case.o_gated(), case.lse)
    for i, (p, (lo, hi)) in enumerate(zip(ctl["members"], ctl["rows"])):
        lv = case.live_rows[lo:hi]
        po, co = out[lo:hi], ctl["out"][i]
        pg, cg = o[lo:hi], ctl["o"][i]
        if cls == "bitwise":
            assert torch.equal(_bits(po), _bits(co)) and torch.equal(_bits(pg), _bits(cg)), f"{what}: position {p} is not bitwise the two-run control"
            if lse is not None:
                assert torch.equal(lse[0, :, lo:hi], ctl["lse"][i]), f"{what}: position {p} LSE is not bitwise the two-run control"
        else:
            if lv.any():
                assert _cos(po[lv], co[lv]) > COS_OUT, f"{what}: position {p} out vs the two-run control"
                assert_output_close(pg[lv], cg[lv], f"{what}: position {p} gated O vs the two-run control")
                if lse is not None:
                    assert float((lse[0, :, lo:hi][:, lv] - ctl["lse"][i][:, lv]).abs().max()) <= ATOL, f"{what}: position {p} LSE vs the two-run control"
            if (~lv).any():
                assert (po[~lv] == 0).all() and (co[~lv] == 0).all()
    print(
        f"\n{what}: class {cls} (control body {ctl['blk'].sparse_tile_form} split {ctl['blk'].decode_split_kv}, control M {ctl['blk']._proj.m}, "
        f"packed M {case.blk._proj.m})"
    )


def _groups(case: _PackedCase):
    """The packing positions grouped by Q length (the two-run form runs one dense block per group), empty sequences left out."""
    by = {}
    for p, q in enumerate(case.q_lens_packed):
        if q:
            by.setdefault(q, []).append(p)
    return by


# the P1 mixed step at test scale: (id, geometry, seqs (q_len, kv_len) per sequence id, dtype, page, layout, return_lse, block_lens, seed)
_MIXED = [
    pytest.param(
        "decodes-and-a-300-chunk",
        GEOM_SMALL,
        [(1, 8192), (1, 2052), (1, 4097), (1, 7000), (1, 300), (1, 1), (1, 2051), (1, 8191), (300, 2351)],
        BF16,
        16,
        "hnd",
        True,
        True,
        11,
        id="8-decodes+chunk300-p16",
    ),
    pytest.param(
        "mtp-rows-and-a-128-chunk",
        GEOM_SMALL,
        [(4, 8192), (4, 300), (4, 4097), (4, 2052), (128, 6272)],
        BF16,
        64,
        "nhd",
        True,
        False,
        13,
        id="4-mtp+chunk128-p64-nhd-no-block-lens",
    ),
    pytest.param(
        "two-chunks-f16",
        GEOM_SMALL,
        [(300, 2351), (1, 8192), (200, 2252)],
        F16,
        16,
        "unified",
        False,
        True,
        17,
        id="2-chunks+decode-f16-unified-no-lse",
    ),
    pytest.param(
        "flash-next-geometry",
        GEOM_FLASH_NEXT,
        [(1, 8192), (1, 4097), (160, 2212), (1, 300)],
        BF16,
        16,
        "hnd",
        True,
        True,
        19,
        id="flash-next-24-2-decodes+chunk160",
    ),
]


@requires_rubin
@pytest.mark.parametrize("name, geom_kw, seqs, dtype, page, layout, return_lse, block_lens, seed", _MIXED)
def test_packed_sparse_extend_matches_the_two_run_form_per_sequence(name, geom_kw, seqs, dtype, page, layout, return_lse, block_lens, seed):
    """The P1 mixed step at test scale in ONE packed execute vs the TWO-RUN form (one dense paged QsaSpec block per Q-length group over the
    same weights / pools / table rows / lists): every row within the suite's bars of the fp32 oracle per sequence, two launches bitwise; per
    group the parity class the control's RESOLUTION predicts -- BITWISE where the control runs unsplit on the packed block's projection
    route, READING B otherwise -- asserted, never discovered; then, where bitwise is the claim, the control's split PINNED to 1 and the
    class asserted again."""
    case = _PackedCase(geom_kw, seqs, dtype=dtype, page=page, layout=layout, return_lse=return_lse, block_lens=block_lens, seed=seed)
    _two_launches(case, f"{name}: packed")
    packed = (case.out.clone(), case.o_gated().clone(), None if case.lse is None else case.lse.clone())
    classes = {}
    for q, members in sorted(_groups(case).items()):
        ctl = _dense_control(case, members)
        cls = _parity_class(case, ctl)
        classes[q] = (cls, ctl["blk"].decode_split_kv, ctl["blk"].sparse_tile_form)
        assert ctl["blk"].decode_split_kv == _predicted_split(case, ctl, len(members), q), (q, ctl["blk"].decode_split_kv, ctl["blk"].sparse_tile_form)
        _assert_parity(case, ctl, cls, f"{name}: group S_q={q} vs the two-run control (resolved split {ctl['blk'].decode_split_kv})", packed=packed)
        if cls == "bitwise":
            pinned = _dense_control(case, members, split=1)
            assert pinned["blk"].decode_split_kv == 1
            _assert_parity(case, pinned, "bitwise", f"{name}: group S_q={q} vs the control pinned at split 1", packed=packed)
        # the pool rows the two forms wrote for the group's step tokens: bitwise where the route agrees (the write-through reads the slab)
        for p in members:
            lo, hi = case.rows[p]
            sl = case.slot_mapping[lo:hi]
            lv = sl >= 0
            if lv.any() and _same_slab_route(case.blk, ctl["blk"]):
                s_ = sl[lv].long()
                k_now, v_now = ctl["pool_rows"][p]
                case.restore_pools()
                case.run()
                assert torch.equal(_bits(case.k_cache[s_ // page, :, s_ % page, :]), k_now) and torch.equal(
                    _bits(case.v_cache[s_ // page, :, s_ % page, :]), v_now
                )
    # the step's chunk (the longest Q length) is the bitwise class whenever its control resolves the core at split 1 on the same route; the
    # decode groups are the budget class by construction (the small-M route, the fill rule's split or the lean tile at one row per sequence)
    print(f"\n{name}: classes per group {classes}")


@requires_rubin
def test_packed_sparse_extend_fill_rule_boundary_cells_assert_their_class():
    """CRITIQUE G2 at the live SM count: three chunk lengths whose dense control resolves a DIFFERENT split under the fill / wave rule (the
    values READ from ``sparse_decode_split_choice`` at this device's SM count, never hard-coded): the class each predicts (bitwise at
    split 1 on the shared route, READING B otherwise) is asserted per cell, so a flip of the rule flips an assertion rather than a
    discovered mismatch."""
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    rows = []
    for T in (300, 128, 64):
        case = _PackedCase(GEOM_SMALL, [(T, T + 2051), (1, 4097)], seed=23 + T)
        _two_launches(case, f"boundary T={T}")
        ctl = _dense_control(case, [0])
        split = _predicted_split(case, ctl, 1, T)
        assert ctl["blk"].decode_split_kv == split and ctl["blk"].sparse_tile == SPARSE_TILE_TCGEN05, (
            T,
            ctl["blk"].decode_split_kv,
            ctl["blk"].sparse_tile_form,
        )
        cls = _parity_class(case, ctl)
        expect = "bitwise" if (split == 1 and (T > SMALL_M_MAX) == (case.T_live > SMALL_M_MAX)) else "reading_b"
        assert cls == expect, (T, cls, expect)
        _assert_parity(case, ctl, cls, f"boundary T={T} (split {split})")
        rows.append((T, split, cls))
    assert len({cls for _, _, cls in rows}) == 2 or len({s for _, s, _ in rows}) >= 2, f"the boundary cells did not straddle the rule at {sm} SMs: {rows}"
    print(f"\nfill-rule boundary at {sm} SMs: {rows}")


@requires_rubin
def test_packed_sparse_extend_serves_any_sequence_order():
    """The same step packed in three orders -- the chunk first, the decodes interleaved with a dead entry and an empty sequence in the MIDDLE,
    the identity -- every sequence's rows BITWISE across the packings (the GEMM rows, the norm + RoPE, the write-through and the sparse core
    are per row / per item; the position in the packing changes no bit), each within the budget vs the oracle."""
    seqs = [(1, 8192), (300, 2351), (0, 100), (4, 0), (17, 2052), (1, 1), (64, 67)]
    orders = [None, [1, 0, 3, 2, 5, 4, 6], [6, 5, 4, 3, 2, 1, 0]]
    cases = [_PackedCase(GEOM_SMALL, seqs, order=o, seed=29) for o in orders]
    for i, c in enumerate(cases):
        _two_launches(c, f"order {orders[i]}")
    base = cases[0]
    for c in cases[1:]:
        for b, (q, kv) in enumerate(seqs):
            if q == 0:
                continue
            lo, hi = base.rows_by_id[b]
            lo2, hi2 = c.rows_by_id[b]
            assert torch.equal(_bits(base.out[lo:hi]), _bits(c.out[lo2:hi2])), f"sequence {b}: out differs across packings"
            assert torch.equal(_bits(base.o_gated()[lo:hi]), _bits(c.o_gated()[lo2:hi2])), f"sequence {b}: gated O differs across packings"
            assert torch.equal(base.lse[0, :, lo:hi], c.lse[0, :, lo2:hi2]), f"sequence {b}: LSE differs across packings"
    print("\nany sequence order: every sequence bitwise across three packings")


@requires_rubin
@pytest.mark.parametrize("residue", [0, 1, 2, 3])
def test_packed_sparse_extend_degenerate_rows_through_the_block(residue):
    """The composed degenerate rows through the block at every prefix residue: ``kv_len 0`` with rows (out 0, O 0, LSE -inf, no pool write), an
    empty sequence, a sequence shorter than its chunk (the negative-position rows exact, the rest live), a chunk straddling a 4-token block
    behind a prefix of this residue, prefix 0 (the packed self-attention form over the pool), the identity bound, a one-token sequence,
    ``-1`` table entries past the live pages; the stale FINITE twin (every unwritten pool row 0 instead of NaN) BITWISE the NaN pools."""
    seqs = [(5, 0), (0, 100), (8, 5), (37, 37), (3, 2052), (1, 1), (64, 1000 + residue + 64), (2, 2 + 4 * 7 + residue), (4, 300)]
    case = _PackedCase(GEOM_SMALL, seqs, seed=31 + residue, cu=bool(residue % 2))
    r = _two_launches(case, f"degenerate rows (residue {residue})")
    assert r["dead"] == 5 + 3, "five rows of the dead entry and the three negative-position rows of the short sequence"
    assert (case.block_table[1] == -1).any() and (case.block_table[0] == 0).all()
    lo, hi = case.rows[0]
    assert (case.slot_mapping[lo:hi] == -1).all() and (case.out[lo:hi] == 0).all() and torch.isinf(case.lse[0, :, lo:hi]).all()
    # the stale FINITE twin: the same step over pools whose unwritten rows hold 0 -- the core gathers listed blocks and the open tail only
    twin = _PackedCase(GEOM_SMALL, seqs, seed=31 + residue, cu=bool(residue % 2))
    for pool in (twin.k_cache, twin.v_cache):
        pool.copy_(torch.nan_to_num(pool, nan=0.0))
    twin.prefix_bits = (_bits(twin.k_cache).clone(), _bits(twin.v_cache).clone())
    assert torch.isfinite(twin.k_cache.float()).all() and torch.isnan(case.k_cache.float()).any()
    twin.run()
    twin.check(what="stale finite twin")
    assert torch.equal(_bits(case.out), _bits(twin.out)) and torch.equal(case.lse, twin.lse) and torch.equal(_bits(case.o_gated()), _bits(twin.o_gated()))


@requires_rubin
def test_packed_sparse_extend_capacity_launch_pads_are_exactly_zero_and_the_live_count_launch_matches():
    """The BOUNDED form (``token_capacity=T_max``): a step of ``T_live < T_max`` tokens as the CAPACITY launch -- every stage over the ``T_max``
    rows, the pad rows' slot ``-1`` writing nothing, the pad rows of ``out`` EXACTLY 0 on the once-zeroed workspace and stale-finite after a
    longer earlier step (the dense form's pin, the sparse twin); the live rows BITWISE the tight declaration of the same step; the LIVE-COUNT
    launch (``num_tokens_live=T_live``: the dense projections over ``T_live`` rows) where the slab resolves a JIT tile (GEOM_SMALL's N 5120
    past the small-M bound): the live rows bitwise the capacity launch, the rows past them UNTOUCHED, the token tensors at exactly ``T_live``
    rows too."""
    seqs = [(1, 8192), (200, 2251), (0, 50), (4, 304), (1, 2052)]
    tight = _PackedCase(GEOM_SMALL, seqs, seed=37)
    _two_launches(tight, "tight declaration")
    T_live, T_max = tight.T_live, 320
    cap = _PackedCase(GEOM_SMALL, seqs, seed=37, token_capacity=T_max)
    assert cap.T_live == T_live and cap.T_cap == T_max and (cap.slot_mapping[T_live:] == -1).all()
    cap.compile()  # the workspace zeroed ONCE
    cap.out.fill_(float("nan"))
    cap.lse.fill_(float("nan"))
    cap.execute()
    torch.cuda.synchronize()
    cap.check(what="capacity launch")
    assert torch.equal(_bits(cap.out[:T_live]), _bits(tight.out)) and torch.equal(
        cap.lse[0, :, :T_live], tight.lse[0]
    ), "the live rows differ from the tight declaration"
    assert (cap.out[T_live:] == 0).all(), "a pad row's out must be exactly 0 on the once-zeroed workspace"
    assert torch.isfinite(cap.out.float()).all()
    # the LIVE-COUNT launch where the slab resolves a JIT tile: the rows past T_live untouched (a sentinel), the live rows bitwise
    if not cap.blk._proj.resolves_a_jit_tile():
        pytest.skip(f"the slab GEMM at M {cap.blk._proj.m} x N {cap.blk._proj.n} takes the graph route: the live-count launch is refused by name there")
    out2 = torch.full((T_max, cap.geom.d_model), _sentinel(cap.dtype), dtype=cap.dtype, device=_DEV)
    lse2 = torch.full_like(cap.lse, _sentinel(cap.dtype))
    cap.execute(out=out2, lse=lse2, num_tokens_live=T_live)
    torch.cuda.synchronize()
    assert torch.equal(_bits(out2[:T_live]), _bits(cap.out[:T_live])) and torch.equal(lse2[0, :, :T_live], cap.lse[0, :, :T_live])
    assert (out2[T_live:].float() == float(torch.full((), _sentinel(cap.dtype), dtype=cap.dtype).float())).all(), "a row past the live count was written"
    assert (lse2[0, :, T_live:] == _sentinel(cap.dtype)).all()
    out3 = torch.full((T_live, cap.geom.d_model), _sentinel(cap.dtype), dtype=cap.dtype, device=_DEV)
    cap.execute(out=out3, lse=lse2, num_tokens_live=T_live, live_rows_inputs=True)  # the token tensors at EXACTLY T_live rows
    torch.cuda.synchronize()
    assert torch.equal(_bits(out3), _bits(cap.out[:T_live]))
    # a shorter second step through the same block leaves the earlier step's rows past its live count stale-finite, never NaN
    short = _PackedCase(
        GEOM_SMALL, [(1, 8192), (100, 2151), (0, 50), (4, 304), (1, 2052)], seed=41, token_capacity=T_max, n_pages=cap.n_pages, max_pages=cap.max_pages
    )
    for nm in ("h", "cos", "sin", "slot_mapping", "block_ids", "block_lens", "block_table", "kv_lens_t", "seq_lens_t"):
        getattr(cap, nm).copy_(getattr(short, nm))
    cap.k_cache.copy_(short.k_cache)
    cap.v_cache.copy_(short.v_cache)
    cap.w["w_qkvg"].copy_(short.w["w_qkvg"])
    cap.w["w_o"].copy_(short.w["w_o"])
    cap.prefix_bits, cap.ref_out, cap.ref_o_gated, cap.ref_lse, cap.live_rows, cap.T_live = (
        short.prefix_bits,
        short.ref_out,
        short.ref_o_gated,
        short.ref_lse,
        short.live_rows,
        short.T_live,
    )
    cap.out.fill_(float("nan"))
    cap.lse.fill_(float("nan"))
    cap.execute()
    torch.cuda.synchronize()
    cap.check(what="a shorter second step at the capacity")
    assert torch.isfinite(cap.out.float()).all(), "a pad row of the shorter step reads the earlier step's finite rows, never NaN"


@requires_rubin
def test_packed_sparse_extend_workspace_is_honest_and_the_execute_path_allocates_nothing_and_never_syncs():
    """``get_workspace_size()`` honest (a 4 KiB sentinel past the carve untouched over three executes), no CUDA allocation and no host sync on
    the execute path after the first execute (``set_sync_debug_mode("error")``; the allocator's counters), and the launch census by kernel
    NAME against the dense per-row control of the same chunk pinned unsplit: the packed form = the control + ONE launch (the THD setup that
    writes the packed metadata), no split combine, no memset / memcpy."""
    from collections import Counter

    case = _PackedCase(GEOM_SMALL, [(1, 4097), (300, 2351), (1, 8192)], seed=43)
    case.compile()
    n = case.blk.get_workspace_size()
    ws = torch.full((n + 4096,), 0x5A, dtype=torch.uint8, device=_DEV)
    ws[:n].zero_()
    case.ws = ws[:n]
    case.run()
    case.check(what="sentinel workspace")
    for _ in range(2):
        case.run()
    assert (ws[n:] == 0x5A).all(), "a stage wrote past get_workspace_size()"
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            case.execute()
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "the execute path allocated"
    case.check(what="after the no-sync executes")
    # the census: the chunk alone as a packed step vs the dense per-row control pinned at split 1
    chunk = _PackedCase(GEOM_SMALL, [(300, 2351)], seed=43)
    chunk.compile()
    chunk.run()
    ctl = _dense_control(chunk, [0], split=1, sparse_tile_form="tcgen05")
    k_pk, m_pk = _cupti_kernels(lambda: chunk.execute())
    ctl_blk = ctl["blk"]
    # the control's operands built OUTSIDE the profiled call (a stack / zeros inside it would be the harness's memcpys, not the block's)
    h_c = torch.stack([chunk.h[lo:hi] for lo, hi in chunk.rows])
    cos_c = torch.stack([chunk.cos[lo:hi] for lo, hi in chunk.rows])
    sin_c = torch.stack([chunk.sin[lo:hi] for lo, hi in chunk.rows])
    ws_c = torch.zeros(ctl_blk.get_workspace_size(), dtype=torch.uint8, device=_DEV)
    ids_c, lens_c = chunk.block_ids[None].contiguous(), chunk.block_lens[None].contiguous()
    torch.cuda.synchronize()
    k_ct, m_ct = _cupti_kernels(
        lambda: ctl_blk.execute(
            h_c,
            chunk.w["w_qkvg"],
            chunk.w["w_q_norm"],
            chunk.w["w_k_norm"],
            cos_c,
            sin_c,
            chunk.w["w_o"],
            ctl["out"],
            ws_c,
            lse=ctl["lse"],
            block_ids=ids_c,
            block_lens=lens_c,
            k_cache=chunk.k_cache,
            v_cache=chunk.v_cache,
            block_table=chunk.block_table,
            kv_lens=chunk.kv_lens_t,
            slot_mapping=chunk.slot_mapping,
        )
    )
    if k_pk is None or k_ct is None:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the workspace and allocation cells stand")
    print(f"\npacked: {len(k_pk)} kernels + {len(m_pk)} memset/memcpy:\n  " + "\n  ".join(x[:140] for x in k_pk))
    print(f"\ncontrol: {len(k_ct)} kernels + {len(m_ct)} memset/memcpy:\n  " + "\n  ".join(x[:140] for x in k_ct))
    assert not m_pk and not m_ct, "a memset / memcpy on the execute path"
    assert not any("combine" in nm.lower() for nm in k_pk), "the packed form runs the sparse core unsplit: no combine"
    assert len(k_pk) == len(k_ct) + 1, (len(k_pk), len(k_ct))
    extra = Counter(k_pk) - Counter(k_ct)
    assert sum(extra.values()) == 1, f"the packed form adds exactly the THD setup launch: {extra}"


@requires_rubin
@pytest.mark.parametrize("band", [False, True], ids=["n5120-jit-route", "band-n5760"])
def test_packed_sparse_extend_capture_after_one_warm_up_replays_bitwise_and_over_a_new_packing(band):
    """CUDA-graph capture AFTER ONE eager warm-up on the capture stream (this form's capture rule: the first execute may create the
    projection route's per-stream state, which a capture refuses to create), the captured execute under ``set_sync_debug_mode("error")``:
    the capture launches nothing, the replay is BITWISE the eager run; a NEW packing of the same shapes (other Q lengths summing to the same
    total, other KV lengths / table / lists / slots / pools / inputs) through the captured tensors replays bitwise its eager twin and passes
    the sibling's oracle -- at GEOM_SMALL (N 5120: the forced tile past 128 rows) and on the indexer band's slab (N 5760, the graph route)."""
    seqs_a = [(1, 8192), (200, 2251), (0, 50), (4, 304), (1, 2052)]
    seqs_b = [(100, 2151), (1, 4097), (4, 0), (100, 4196), (1, 1)]
    case = _PackedCase(GEOM_SMALL, seqs_a, seed=53, index_band=band)
    sib = _PackedCase(GEOM_SMALL, seqs_b, seed=59, index_band=band, n_pages=case.n_pages, max_pages=case.max_pages)
    assert case.T_live == sib.T_live and case.block_table.shape == sib.block_table.shape and case.k_cache.shape == sib.k_cache.shape
    case.compile()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        case.run()  # the eager warm-up ON the capture stream
    eager_out, eager_lse, eager_o = case.out.clone(), case.lse.clone(), case.o_gated().clone()
    case.check(what="eager warm-up")
    case.restore_pools()
    case.out.fill_(float("nan"))
    case.lse.fill_(float("nan"))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.execute()
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(case.out).all(), "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        assert (
            torch.equal(_bits(case.out), _bits(eager_out)) and torch.equal(case.lse, eager_lse) and torch.equal(_bits(case.o_gated()), _bits(eager_o))
        ), "the replay differs from the eager run"
        case.check(what="replay")
        # the NEW packing through the captured tensors
        for nm in ("h", "cos", "sin", "slot_mapping", "block_ids", "block_lens", "block_table", "kv_lens_t", "seq_lens_t", "k_cache", "v_cache"):
            getattr(case, nm).copy_(getattr(sib, nm))
        if band:
            case.index_cache.copy_(sib.index_cache)
        for nm in ("w_qkvg", "w_o"):
            case.w[nm].copy_(sib.w[nm])
        case.prefix_bits, case.ref_out, case.ref_o_gated, case.ref_lse, case.live_rows = (
            sib.prefix_bits,
            sib.ref_out,
            sib.ref_o_gated,
            sib.ref_lse,
            sib.live_rows,
        )
        case.q_lens_packed, case.kv_lens_packed, case.rows = sib.q_lens_packed, sib.kv_lens_packed, sib.rows
        case.k_prefix, case.v_prefix = sib.k_prefix, sib.v_prefix
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        rep_out, rep_lse, rep_o = case.out.clone(), case.lse.clone(), case.o_gated().clone()
        case.check(what="replay over a new packing")
        sib.compile()
        sib.run()
        assert (
            torch.equal(_bits(rep_out), _bits(sib.out)) and torch.equal(rep_lse, sib.lse) and torch.equal(_bits(rep_o), _bits(sib.o_gated()))
        ), "the replay over a new packing differs from its eager twin"
    finally:
        graph.reset()
    print(
        f"\ncapture after one warm-up (band={band}, slab M {case.blk._proj.m} N {case.blk._proj.n}, jit route {case.blk._proj.resolves_a_jit_tile()}): replay bitwise eager and over a new packing"
    )


@requires_rubin
def test_packed_sparse_extend_length_rows_on_a_compiled_block():
    """The packed lengths' rows on a COMPILED block (they sit behind the compile gate on every THD form): ``seq_lens`` REQUIRED, exactly
    ``[B]`` lengths (``[B + 1]`` prefix sums under ``cu_seqlens``) -- another count, another dtype or a CPU tensor refused by name before any
    launch; the well-formed call then runs and matches the case's own execute bitwise."""
    case = _PackedCase(GEOM_SMALL, [(1, 4097), (64, 2115), (0, 10)], seed=71)
    cu = _PackedCase(GEOM_SMALL, [(1, 4097), (64, 2115), (0, 10)], seed=71, cu=True)
    case.compile()
    cu.compile()
    i32 = lambda *shape: torch.zeros(*shape, dtype=torch.int32, device=_DEV)  # noqa: E731

    def run(c, seq_lens):
        c.blk.execute(
            c.h,
            c.w["w_qkvg"],
            c.w["w_q_norm"],
            c.w["w_k_norm"],
            c.cos,
            c.sin,
            c.w["w_o"],
            c.out,
            c.ws,
            seq_lens=seq_lens,
            lse=c.lse,
            block_ids=c.block_ids,
            block_lens=c.block_lens,
            k_cache=c.k_cache,
            v_cache=c.v_cache,
            block_table=c.block_table,
            kv_lens=c.kv_lens_t,
            slot_mapping=c.slot_mapping,
        )

    with _no_host_sync():
        with pytest.raises(ValueError, match=r"execute\(seq_lens=\) is required"):
            run(case, None)
        with pytest.raises(ValueError, match="seq_lens must be a contiguous 1-D int32 CUDA tensor of 3 elements"):
            run(case, i32(4))  # the prefix-sum count on a lengths declaration
        with pytest.raises(ValueError, match="seq_lens must be a contiguous 1-D int32 CUDA tensor of 4 elements"):
            run(cu, i32(3))
        with pytest.raises(ValueError, match="seq_lens must be a contiguous 1-D int32 CUDA tensor"):
            run(case, torch.zeros(3, dtype=torch.int64, device=_DEV))
        with pytest.raises(ValueError, match="seq_lens must be a contiguous 1-D int32 CUDA tensor"):
            run(case, torch.zeros(3, dtype=torch.int32))
    _two_launches(case, "lengths form")
    _two_launches(cu, "prefix-sum form")
    assert torch.equal(_bits(case.out), _bits(cu.out)) and torch.equal(case.lse, cu.lse), "the two length forms are not bitwise"


@requires_rubin
@pytest.mark.parametrize("fuse_gate", [False, True], ids=["gate_launch", "fuse_gate"])
def test_packed_sparse_extend_two_launches_bitwise_with_and_without_the_lse(fuse_gate):
    """Two executes bitwise on ``out`` / the gated O / the LSE ``[1, H_q, T]``; ``return_lse=False`` declares a stage without the LSE (the
    presence is compiled in) and matches the LSE-writing twin bitwise on ``out`` and the gated O; the separate gate launch and the fused
    epilogue gate each within the budget."""
    seqs = [(1, 8192), (128, 2180), (4, 300), (0, 10), (37, 2089)]
    with_lse = _PackedCase(GEOM_SMALL, seqs, seed=67, fuse_gate=fuse_gate, return_lse=True)
    _two_launches(with_lse, f"gate={'fused' if fuse_gate else 'launch'} lse=True")
    no_lse = _PackedCase(GEOM_SMALL, seqs, seed=67, fuse_gate=fuse_gate, return_lse=False)
    assert not no_lse.blk._sdpa.want_lse and no_lse.lse is None
    _two_launches(no_lse, f"gate={'fused' if fuse_gate else 'launch'} lse=False")
    assert torch.equal(_bits(with_lse.out), _bits(no_lse.out)) and torch.equal(_bits(with_lse.o_gated()), _bits(no_lse.o_gated()))
