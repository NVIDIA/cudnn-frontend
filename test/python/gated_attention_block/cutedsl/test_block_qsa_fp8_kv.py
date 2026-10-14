# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's e4m3 KV cache (``kv_cache_dtype=torch.float8_e4m3fn``): the serving stacks' ``--kv-cache-dtype fp8``
recipe -- the pools hold e4m3 codes under per-tensor fp32 DEscales ``execute(k_scale=, v_scale=)`` (value = code x scale), a bf16 Q
attends over the dequantized rows.  Two served forms, one attribute: (a) the WRITE-THROUGH quantizes the post-RoPE K / V on the way
into the pools (two per-tensor casts + the bit copy through the pools' bf16 views; no new kernel) on every form that writes through
over the block's own K / V; (b) the BLOCK-SPARSE DECODE mode (``kv_source="paged"`` on a dense ``[B, S_q <= 4]`` QsaSpec declaration)
reads the pools through ``kernels/qsa_kv_upcast.py`` (the listed rows gathered and cast back to bf16 into the workspace scratch) and
the Rubin d256 DECODE tile over that scratch as a plain padded dense SDPA -- the sparse core untouched by fp8.  bf16 Q over an e4m3
pool only (no fp8-Q form).

Host cells (any device; the declaration cells even on the CPU): the declaration FACTS (the two e4m3 stages, the forced out-of-place
norm, the gate as its own launch, the split 1, the appended workspace slots at the END of an otherwise byte-identical carve, the
stage order, the write-through-only declaration); the typed declines (e5m2, another dtype, the attribute without the write-through,
``quant``, f16 activations, the dense DECODE form over e4m3 pools -- the gap --, the EXTEND form, ``S_q > 4``, ``return_lse``, a
split, ``fuse_norm_rope`` / ``inplace_qkv=True``, pools of the wrong dtype both ways); the execute-time form checks on an UNCOMPILED
block under ``set_sync_debug_mode("error")`` (the scales required under the attribute, refused without it, their form); the
envelope's ``kv_cache_dtypes``.

Rubin cells (``requires_rubin``; one compiled block per cell): the decode ladder against the fp32 oracle ON THE DEQUANTIZED POOL --
the dense block reference over the WHOLE sequence gives the post-norm / post-RoPE K / V; the PREFIX rows are quantized by torch with
the per-tensor scales (``scale = amax / 448``) into NaN-filled e4m3 pools, the STEP rows are the block's own quantizing write-through;
the oracle attends in fp32 over ``bf16(fp32(code) x scale)`` of every row (the step rows' codes computed the way the block's cast
computes them: ``sat_e4m3(row x (1 / scale))``) with the bf16 post-RoPE Q over ``qsa_visible_mask(..., pos0=)``, then the gate and
the out projection -- ``cos(out) > 0.999``, the gated O (the workspace slot) within ``atol 2e-2`` (the sparse module's bar, never
widened), a dead row (``kv_lens = 0``; a row at a negative position) EXACTLY 0, the pools' step rows BITWISE torch's quantization
of the block's own normed K rows and V band, the prefix rows and the NULL block untouched, two launches bitwise; REPORTED without a
gate: the distance to the PRE-quantization bf16 oracle (the serving recipe's own error).  The ladder: ``S_q`` 1 .. 4 (the shared
list), B 1 / 3 / 4, ``kv_lens`` up to 20000 incl. 2052, a dead entry and a sequence shorter than the step, pages 16 / 32 / 64 on
HND / unified / NHD pools, ``block_lens`` given / absent, the gate fused (= its own launch here) and separate, the indexer band's
raw-key write-through (bf16), the 24 / 2 Flash-Next geometry; the MTP residues on a short sequence; the write-through-only PREFILL
form (the sparse block over its own K / V with e4m3 pools: ``out`` BITWISE the bf16-pool twin's, the pools bitwise the torch cast);
``get_workspace_size()`` honest (a sentinel past the carve untouched; the scratch inside); CUDA-graph capture of the FIRST execute
on a NEW stream with no warm-up under ``set_sync_debug_mode("error")``, the replay bitwise eager, and the replay over NEW scales /
lengths / tables / lists / slots / inputs bitwise the eager run over them.
"""

import contextlib
import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import KV_SOURCE_PAGED, GatedAttentionBlockFwd, GatedAttentionBlockGeometry, QsaSpec, index_k_raw_view
from cudnn.gated_attention_block.api import KV_CACHE_E4M3, _cols, _plan_workspace, _QuantizedCacheWrite, _UpcastSparseSdpa, _view

requirement_error = cutedsl_requirement_error("the gated attention block's e4m3 KV-cache tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_qsa_reference import full_block_ids, qsa_visible_mask, random_block_ids  # noqa: E402
from gated_block_reference import RefGeometry, build_rope_tables, gated_attention_block_reference  # noqa: E402

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
BF16, F16, E4M3 = torch.bfloat16, torch.float16, torch.float8_e4m3fn
D = 256
ROPE = 64
BS = 4  # tokens per selectable block
COS_OUT = 0.999  # the end-to-end suite's bar on `out`
ATOL = 2e-2  # the sparse module's bar on O -- never widened here
E4M3_MAX = 448.0
GEOM_SMALL = dict(d_model=512, h_q=8, h_kv=2, d_head=D, rope_dim=ROPE)  # group 4, cheap projections
GEOM_FLASH_NEXT = dict(d_model=2560, h_q=24, h_kv=2, d_head=D, rope_dim=ROPE)  # Qwen3.8-Flash-Next TP 1 (group 12)


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


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _bits(x: torch.Tensor) -> torch.Tensor:
    """The bit patterns of a bf16 / f16 / e4m3 tensor: a NaN-safe bitwise comparison."""
    return x.contiguous().view(torch.int8 if x.element_size() == 1 else torch.int16)


def _sentinel(dtype):
    return float(torch.finfo(dtype).max) / 4


def _pools(n_pages, h, page, layout, dtype, device=_DEV, fill=float("nan")):
    """Two pools as ``[P, H, page, D]`` views: ``hnd`` (compact), ``nhd`` (by strides), ``unified`` (one ``[P, H, page, 2D]`` content
    row per token, K at ``+0``, V at ``+D``).  NaN-filled (an e4m3 NaN is the code 0x7f): a gathered unwritten row poisons O."""
    mk = lambda *shape: torch.full(shape, fill, device=device, dtype=torch.float32).to(dtype)  # noqa: E731
    if layout == "hnd":
        return tuple(mk(n_pages, h, page, D) for _ in range(2))
    if layout == "nhd":
        return tuple(mk(n_pages, page, h, D).permute(0, 2, 1, 3) for _ in range(2))
    assert layout == "unified"
    row = mk(n_pages, h, page, 2 * D)
    return row[..., :D], row[..., D:]


def _quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The block's quantizing write as torch computes it: ``sat_e4m3(fp32(row) x fp32(1 / scale))`` -- one fp32 product, one rounding
    (torch's cast rounds to nearest even like the kernel's ``cvt.rn``; the clamp is the ``satfinite`` the kernel applies)."""
    inv = torch.reciprocal(scale.float())
    return (x.float() * inv).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)


def _dequantize(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The gathered cast's arithmetic: ``bf16(fp32(code) x scale)`` -- one fp32 rounding."""
    return (codes.float() * scale.float()).to(BF16)


def _scale_of(x: torch.Tensor) -> torch.Tensor:
    """A per-tensor DEscale calibrated on ``x`` (the serving stacks' ``amax / 448``), a 1-element fp32 CUDA tensor."""
    amax = x.float().abs().amax().clamp_min(1e-12)
    return (amax / E4M3_MAX).reshape(1).to(torch.float32).contiguous()


def _geom(geom_kw, top_k=512, index_band=False, **over):
    kw = dict(geom_kw, causal_bottom_right=True)
    kw.update(over)
    return GatedAttentionBlockGeometry(**kw, qsa=QsaSpec(top_k=top_k, index_band=index_band)), RefGeometry(**kw)


def _declare(
    geom_kw, batch, s_q, *, page=16, layout="hnd", n_pages=8, dtype=BF16, pool_dtype=E4M3, top_k=512, index_band=False, geom_over=None, qsa=True, **blk_kw
):
    """A declared (not compiled) block on EMPTY tensors: the declaration cells (any device).  ``kv_cache_dtype`` rides ``blk_kw``."""
    kw0 = dict(geom_kw)
    kw0.update(geom_over or {})
    if qsa:
        geom, _ = _geom(geom_kw, top_k=top_k, index_band=index_band, **(geom_over or {}))
    else:
        geom = GatedAttentionBlockGeometry(**kw0)
    mk = lambda *shape: torch.empty(*shape, dtype=dtype, device=_DEV)  # noqa: E731
    kc, vc = _pools(n_pages, geom.h_kv, page, layout, pool_dtype)
    bt = torch.zeros(batch, n_pages, dtype=torch.int32, device=_DEV)
    kw = dict(paged_kv_page_size=page, kv_source=KV_SOURCE_PAGED, sample_k_cache=kc, sample_v_cache=vc, sample_block_table=bt, kv_cache_dtype=E4M3)
    kw.update(blk_kw)
    blk = GatedAttentionBlockFwd(
        mk(batch, s_q, geom.d_model),
        mk(geom.n_qkvg, geom.d_model),
        torch.ones(D, dtype=dtype, device=_DEV),
        torch.ones(D, dtype=dtype, device=_DEV),
        mk(batch, s_q, ROPE),
        mk(batch, s_q, ROPE),
        mk(geom.d_model, geom.h_q * D),
        mk(batch, s_q, geom.d_model),
        geom,
        **kw,
    )
    return blk, dict(k_cache=kc, v_cache=vc, block_table=bt, geom=geom)


# ---------------------------------------------------------------------------
# Host: the declaration facts, the typed declines, the execute form checks, the envelope, the carve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s_q", [1, 2, 3, 4])
def test_declaration_facts_of_the_e4m3_mode(s_q):
    """Over e4m3 pools the block-sparse decode declaration builds the quantizing write-through and the gathered-cast stage, norms out
    of place (``inplace_qkv=None`` reads as False), gates in its own launch whatever ``fuse_gate`` says, runs the tile unsplit, and
    carves the e4m3 slots at the END of the layout (the twin without the attribute is byte-identical to before)."""
    B = 3
    with _no_host_sync():
        blk, env = _declare(GEOM_SMALL, B, s_q, fuse_gate=True)
        twin, _ = _declare(GEOM_SMALL, B, s_q, fuse_gate=True, pool_dtype=BF16, kv_cache_dtype=None)
    assert blk.kv_cache_dtype == E4M3 and blk.kv_cache_e4m3 and blk.upcast_decode and blk.sparse_decode and blk.qsa_shared_list
    assert isinstance(blk._sdpa, _UpcastSparseSdpa) and isinstance(blk._cache_write, _QuantizedCacheWrite)
    assert blk.inplace_qkv is False and blk.fuse_gate is True and blk.gate_fused is False and blk._gate is not None and blk._compact_v is None
    assert blk.decode_split_kv == 1 and blk.seq_lens_present
    assert [st.name for st in blk._stages] == ["qkv_gate_proj", "qk_norm_rope", "cache_write_e4m3", "sdpa_sparse_e4m3", "sigmoid_gate", "out_proj"]
    items = B * s_q * env["geom"].h_kv
    lay, lay0 = blk._layout(), twin._layout()
    assert lay.kc8 >= 0 and lay.vc8 >= 0 and lay.kv_inv >= 0 and lay.kv_gathered >= 0 and lay.kv_len_t >= 0
    assert lay.kv_len_t - lay.kv_gathered >= items * 2 * 2176 * 256 * 2 and lay.total_bytes > lay.kv_len_t
    assert lay0.kc8 == lay0.vc8 == lay0.kv_inv == lay0.kv_gathered == lay0.kv_len_t == -1
    # the twin keeps the in-place carve (no compact q / k / v); the e4m3 mode's compact slots precede the appended e4m3 slots
    assert twin.inplace_qkv is True and lay0.q == -1 and lay.q >= 0 and lay.k >= 0 and lay.proj == lay0.proj == 0
    assert lay.kc8 > max(lay.q, lay.k, lay.v, lay.o)
    # the write-through-only declaration (the prefill form over the block's own K / V): the e4m3 write without the gathered scratch
    with _no_host_sync():
        pre, _ = _declare(
            GEOM_SMALL, 2, 8, kv_source="self", sample_k_cache=None, sample_v_cache=None, sample_block_table=None, geom_over=dict(causal_bottom_right=False)
        )
    assert pre.kv_cache_e4m3 and not pre.upcast_decode and isinstance(pre._cache_write, _QuantizedCacheWrite) and pre.inplace_qkv is True
    lp = pre._layout()
    assert lp.kc8 >= 0 and lp.vc8 >= 0 and lp.kv_inv >= 0 and lp.kv_gathered == -1 and lp.kv_len_t == -1
    # the dense (no QsaSpec) prefill with the e4m3 write-through is served too
    with _no_host_sync():
        dense, _ = _declare(GEOM_SMALL, 2, 8, kv_source="self", sample_k_cache=None, sample_v_cache=None, sample_block_table=None, qsa=False)
    assert dense.kv_cache_e4m3 and isinstance(dense._cache_write, _QuantizedCacheWrite) and dense.qsa is None


def test_the_carve_without_the_attribute_is_byte_identical_and_the_e4m3_slots_append():
    g, _ = _geom(GEOM_SMALL)
    base = _plan_workspace(g, 2, 4, BF16, False, False, True)
    same = _plan_workspace(g, 2, 4, BF16, False, False, True, kv_cache_e4m3=False, upcast_items=0)
    assert base == same and base.kc8 == -1 and base.kv_gathered == -1
    e4 = _plan_workspace(g, 2, 4, BF16, False, False, False, kv_cache_e4m3=True, upcast_items=2 * 4 * g.h_kv)
    oop = _plan_workspace(g, 2, 4, BF16, False, False, False)
    assert (e4.proj, e4.q, e4.k, e4.v, e4.o) == (oop.proj, oop.q, oop.k, oop.v, oop.o) and e4.kc8 == oop.total_bytes
    assert e4.vc8 - e4.kc8 >= 2 * 4 * g.h_kv * D and e4.kv_len_t - e4.kv_gathered == 2 * 4 * g.h_kv * 2 * 2176 * 256 * 2 and e4.total_bytes > e4.kv_len_t
    with pytest.raises(ValueError, match="form of the bf16 pipeline"):
        _plan_workspace(g, 2, 4, BF16, False, False, False, fp8=True, kv_cache_e4m3=True)
    with pytest.raises(ValueError, match="upcast_items belongs to the e4m3 KV cache"):
        _plan_workspace(g, 2, 4, BF16, False, False, False, upcast_items=4)


def test_typed_declines_of_the_e4m3_mode():
    """Every decline by name, device-free (the declaration never touches the device); the mirror images included."""
    dec = _declare
    # the attribute's own rows
    with pytest.raises(NotImplementedError, match="float8_e5m2.*e4m3 KV cache is the one served"):
        dec(GEOM_SMALL, 2, 1, kv_cache_dtype=torch.float8_e5m2)
    with pytest.raises(ValueError, match="kv_cache_dtype must be None, the activation dtype"):
        dec(GEOM_SMALL, 2, 1, kv_cache_dtype=torch.int8)
    with pytest.raises(TypeError, match="kv_cache_dtype must be a torch.dtype"):
        dec(GEOM_SMALL, 2, 1, kv_cache_dtype="e4m3")
    with pytest.raises(ValueError, match="differs from the activation dtype"):
        dec(GEOM_SMALL, 2, 1, kv_cache_dtype=F16, pool_dtype=BF16)
    with pytest.raises(ValueError, match="belongs to the paged KV-cache write-through"):
        dec(GEOM_SMALL, 2, 8, paged_kv_page_size=0, kv_source="self", sample_k_cache=None, sample_v_cache=None, sample_block_table=None, qsa=False)
    with pytest.raises(NotImplementedError, match="f16 activations over an e4m3 cache are a follow-up|activations: the gathered cast"):
        dec(GEOM_SMALL, 2, 1, dtype=F16)
    # the pools' dtype both ways (the declaration samples)
    with pytest.raises(ValueError, match="sample_k_cache must be torch.float8_e4m3fn"):
        dec(GEOM_SMALL, 2, 1, pool_dtype=BF16)
    with pytest.raises(ValueError, match="sample_k_cache must be torch.bfloat16"):
        dec(GEOM_SMALL, 2, 1, kv_cache_dtype=None)
    # quant / training: the write-through's own rows speak
    from cudnn.gated_attention_block import QuantSpec

    g, _ = _geom(GEOM_SMALL)
    mk8 = lambda *shape: torch.empty(*shape, dtype=E4M3, device=_DEV)  # noqa: E731
    with pytest.raises(NotImplementedError, match="paged_kv_page_size with quant=QuantSpec"):
        GatedAttentionBlockFwd(
            mk8(2, 8, 512),
            mk8(g.n_qkvg, 512),
            torch.ones(D, dtype=BF16, device=_DEV),
            torch.ones(D, dtype=BF16, device=_DEV),
            torch.empty(2, 8, ROPE, dtype=BF16, device=_DEV),
            torch.empty(2, 8, ROPE, dtype=BF16, device=_DEV),
            mk8(512, 8 * D),
            torch.empty(2, 8, 512, dtype=BF16, device=_DEV),
            GatedAttentionBlockGeometry(**GEOM_SMALL),
            quant=QuantSpec(descale_h=1.0, descale_w_qkvg=1.0, descale_w_o=1.0, scale_q=1.0, scale_k=1.0, scale_v=1.0, scale_o=1.0),
            paged_kv_page_size=16,
            kv_cache_dtype=E4M3,
        )
    with pytest.raises(NotImplementedError, match="paged_kv_page_size with save_for_backward=True"):
        dec(
            GEOM_SMALL,
            2,
            8,
            kv_source="self",
            sample_k_cache=None,
            sample_v_cache=None,
            sample_block_table=None,
            qsa=False,
            save_for_backward=True,
            inplace_qkv=False,
        )
    # over the pools: the dense DECODE form (the gap), the EXTEND form, S_q > 4, the LSE, a split, the in-place knobs
    with pytest.raises(NotImplementedError, match="dense geometry \\(the DECODE form\\).*in-kernel dequant arm"):
        dec(GEOM_SMALL, 2, 1, qsa=False, geom_over=dict(is_causal=True, causal_bottom_right=True))
    with pytest.raises(NotImplementedError, match="EXTEND form over e4m3 pools"):
        _declare_packed_extend()
    with pytest.raises(NotImplementedError, match="SHARED-list form, S_q <= 4"):
        dec(GEOM_SMALL, 2, 5)
    with pytest.raises(NotImplementedError, match="return_lse=True with kv_cache_dtype"):
        dec(GEOM_SMALL, 2, 2, return_lse=True)
    with pytest.raises(NotImplementedError, match="decode_split_kv=4 with kv_cache_dtype"):
        dec(GEOM_SMALL, 2, 2, decode_split_kv=4)
    with pytest.raises(NotImplementedError, match="fuse_norm_rope=True with kv_cache_dtype"):
        dec(GEOM_SMALL, 2, 2, fuse_norm_rope=True)
    with pytest.raises(NotImplementedError, match="inplace_qkv=True with kv_cache_dtype"):
        dec(GEOM_SMALL, 2, 2, inplace_qkv=True)
    # the split 1 and inplace_qkv=False are accepted spellings of the mode's own facts
    blk, _ = dec(GEOM_SMALL, 2, 2, decode_split_kv=1, inplace_qkv=False)
    assert blk.decode_split_kv == 1 and blk.inplace_qkv is False


def _declare_packed_extend():
    """A packed (thd=True) EXTEND declaration on a DENSE geometry over e4m3 pools: the table's rows pass (bottom-right, causal, d256)
    and the attribute's EXTEND row declines."""
    geom = GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True)
    T = 32
    mk = lambda *shape: torch.empty(*shape, dtype=BF16, device=_DEV)  # noqa: E731
    kc, vc = _pools(8, geom.h_kv, 16, "hnd", E4M3)
    return GatedAttentionBlockFwd(
        mk(T, geom.d_model),
        mk(geom.n_qkvg, geom.d_model),
        torch.ones(D, dtype=BF16, device=_DEV),
        torch.ones(D, dtype=BF16, device=_DEV),
        mk(T, ROPE),
        mk(T, ROPE),
        mk(geom.d_model, geom.h_q * D),
        mk(T, geom.d_model),
        geom,
        thd=True,
        num_sequences=2,
        max_seq_len=T,
        paged_kv_page_size=16,
        kv_source=KV_SOURCE_PAGED,
        sample_k_cache=kc,
        sample_v_cache=vc,
        sample_block_table=torch.zeros(2, 8, dtype=torch.int32, device=_DEV),
        kv_cache_dtype=E4M3,
    )


def test_execute_form_checks_of_the_e4m3_mode():
    """The scales are a DECLARATION-vs-argument contract checked before any plan exists: required (1-element fp32 CUDA on h's device)
    under the attribute, refused on a bf16-pool block; the pools in the pool dtype at execute -- every refusal on an UNCOMPILED block
    without a device sync (Rule 3)."""
    if not torch.cuda.is_available():
        pytest.skip("the execute form checks need CUDA tensors (the device is never touched)")
    B, s_q = 2, 2
    blk, env = _declare(GEOM_SMALL, B, s_q)
    twin, env2 = _declare(GEOM_SMALL, B, s_q, pool_dtype=BF16, kv_cache_dtype=None)
    g = env["geom"]
    mk = lambda *shape, dt=BF16: torch.zeros(*shape, dtype=dt, device=_DEV)  # noqa: E731
    one = torch.ones(1, dtype=torch.float32, device=_DEV)
    ids = torch.full((B, 512), -1, dtype=torch.int32, device=_DEV)
    common = dict(
        w_qkvg=mk(g.n_qkvg, g.d_model),
        w_q_norm=torch.ones(D, dtype=BF16, device=_DEV),
        w_k_norm=torch.ones(D, dtype=BF16, device=_DEV),
        cos=mk(B, s_q, ROPE),
        sin=mk(B, s_q, ROPE),
        w_o=mk(g.d_model, g.h_q * D),
        out=mk(B, s_q, g.d_model),
        workspace=mk(16, dt=torch.uint8),
        block_ids=ids,
        slot_mapping=torch.full((B * s_q,), -1, dtype=torch.int32, device=_DEV),
        kv_lens=torch.zeros(B, dtype=torch.int32, device=_DEV),
    )

    def run(b, e, **over):
        kw = dict(common, k_cache=e["k_cache"], v_cache=e["v_cache"], block_table=e["block_table"])
        kw.update(over)
        b.execute(mk(B, s_q, g.d_model), **kw)

    with _no_host_sync():
        with pytest.raises(ValueError, match="needs k_scale at execute"):
            run(blk, env, v_scale=one)
        with pytest.raises(ValueError, match="needs v_scale at execute"):
            run(blk, env, k_scale=one)
        with pytest.raises(ValueError, match="needs k_scale at execute"):
            run(blk, env, k_scale=one.to(BF16), v_scale=one)
        with pytest.raises(ValueError, match="needs k_scale at execute"):
            run(blk, env, k_scale=torch.ones(2, dtype=torch.float32, device=_DEV), v_scale=one)
        with pytest.raises(ValueError, match="needs k_scale at execute"):
            run(blk, env, k_scale=1.0, v_scale=one)
        with pytest.raises(ValueError, match="k_scale, v_scale: the per-tensor scales of an e4m3 KV cache"):
            run(twin, env2, k_scale=one, v_scale=one)
        with pytest.raises(ValueError, match="k_cache must be torch.float8_e4m3fn"):
            run(blk, env, k_cache=env2["k_cache"], v_cache=env2["v_cache"], k_scale=one, v_scale=one)
        with pytest.raises(ValueError, match="k_cache must be torch.bfloat16"):
            run(twin, env2, k_cache=env["k_cache"], v_cache=env["v_cache"])
        # past the form checks the uncompiled block stops at its own plan guard -- nothing above touched the device
        with pytest.raises(RuntimeError, match="call compile\\(\\) before execute\\(\\)"):
            run(blk, env, k_scale=one, v_scale=one)
        # the scales on a block without the write-through at all
        plain, _ = _declare(
            GEOM_SMALL,
            B,
            s_q,
            paged_kv_page_size=0,
            kv_source="self",
            sample_k_cache=None,
            sample_v_cache=None,
            sample_block_table=None,
            kv_cache_dtype=None,
            geom_over=dict(causal_bottom_right=False),
        )
        per_token_ids = torch.full((B * s_q, 512), -1, dtype=torch.int32, device=_DEV)  # the prefill form's list (the index check runs first)
        with pytest.raises(ValueError, match="k_scale, v_scale: the paged KV-cache inputs belong to a block declared with paged_kv_page_size > 0"):
            plain.execute(mk(B, s_q, g.d_model), **dict(common, block_ids=per_token_ids, slot_mapping=None, kv_lens=None, k_scale=one, v_scale=one))


def test_the_envelope_names_the_pool_dtypes():
    """``decode.kv_cache_dtypes``: the activation dtype always; e4m3 on a QsaSpec geometry with bf16 activations only."""
    g, _ = _geom(GEOM_SMALL)
    dense = GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True)
    for cc in ((10, 7), (10, 0)):
        env = GatedAttentionBlockFwd.serving_envelope(g, BF16, _DEV, device_cc=cc)
        assert env.decode.kv_cache_dtypes == (BF16, E4M3) and env.decode.sparse_core
        assert GatedAttentionBlockFwd.serving_envelope(g, F16, _DEV, device_cc=cc).decode.kv_cache_dtypes == (F16,)
        assert GatedAttentionBlockFwd.serving_envelope(dense, BF16, _DEV, device_cc=cc).decode.kv_cache_dtypes == (BF16,)
    assert any("kv_cache_dtype" in r for r in env.decode.requires)


# ---------------------------------------------------------------------------
# Rubin: the oracle on the dequantized pool
# ---------------------------------------------------------------------------


def _rope_tables(n_pos, dtype, ref_geom):
    cos, sin = build_rope_tables(n_pos, ROPE, base=ref_geom.rope_base, batch=1, device=_DEV, dtype=dtype)
    return cos, sin  # [1, n_pos, ROPE]


class _Fp8Case:
    """One e4m3-cache sparse DECODE step over a batch of sequences with lengths ``kv_lens`` (0 = a dead entry; a length below ``s_q``
    leaves the rows with a negative position dead): the whole-sequence bf16 reference per entry, per-tensor DEscales calibrated on
    the reference K / V of every live entry, the NaN-filled e4m3 pools holding the torch-quantized PREFIX rows, the block table (page
    0 = the NULL block), the slot mapping of the step's tokens, ONE list per sequence at the step-0 position, and two fp32 oracles
    of the step's gated O and ``out``: on the DEQUANTIZED rows (the gate) and on the pre-quantization bf16 rows (reported)."""

    def __init__(self, geom_kw, s_q, kv_lens, *, page=16, layout="hnd", seed=0, top_k=512, block_lens=True, fuse_gate=False, index_band=False, n_pages=None):
        self.s_q, self.kv_lens, self.page, self.layout, self.top_k = int(s_q), [int(x) for x in kv_lens], page, layout, top_k
        self.block_lens_given, self.fuse_gate, self.index_band = bool(block_lens), bool(fuse_gate), bool(index_band)
        self.geom, self.ref_geom = _geom(geom_kw, top_k=top_k, index_band=index_band)
        geom, ref_geom, dtype = self.geom, self.ref_geom, BF16
        self.dtype = dtype
        h_q, h_kv, d_model = geom.h_q, geom.h_kv, geom.d_model
        G = h_q // h_kv
        B = len(self.kv_lens)
        g = torch.Generator(device=_DEV).manual_seed(seed)
        self.gen = g

        def randn(*shape, std=0.02):
            return (torch.randn(*shape, generator=g, device=_DEV, dtype=torch.float32) * std).to(dtype)

        n_dense = sum(geom.qkvg_block_widths[:4])
        w_qkvg = randn(geom.n_qkvg, d_model)
        self.w = dict(
            w_qkvg=w_qkvg, w_q_norm=torch.ones(D, dtype=dtype, device=_DEV), w_k_norm=torch.ones(D, dtype=dtype, device=_DEV), w_o=randn(d_model, h_q * D)
        )
        L_max = max(max(self.kv_lens), s_q)
        cos_t, sin_t = _rope_tables(L_max, dtype, ref_geom)
        scale = geom.scale
        self.h = torch.zeros(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.cos = cos_t[0, :1].expand(s_q, ROPE).clone()[None].expand(B, s_q, ROPE).clone()
        self.sin = sin_t[0, :1].expand(s_q, ROPE).clone()[None].expand(B, s_q, ROPE).clone()
        self.live_rows = torch.zeros(B, s_q, dtype=torch.bool, device=_DEV)
        refs, ids_rows, lens_rows = {}, [], []
        for b, L in enumerate(self.kv_lens):
            pos = torch.arange(s_q, device=_DEV) + (L - s_q)
            live = pos >= 0
            self.live_rows[b] = live
            pos0 = torch.tensor([L - s_q], device=_DEV)
            ids_b, lens_b = random_block_ids(pos0, top_k, BS, generator=g, kv_lens=torch.tensor([L], device=_DEV), shuffle=True)
            ids_rows.append(ids_b)
            lens_rows.append(lens_b)
            if L == 0:
                continue  # a dead entry: no sequence, no reference, block 0, slots -1
            h_full = randn(1, L, d_model, std=1.0)
            ref = gated_attention_block_reference(
                h_full, w_qkvg[:n_dense], self.w["w_q_norm"], self.w["w_k_norm"], cos_t[:, :L], sin_t[:, :L], self.w["w_o"], ref_geom
            )
            refs[b] = (ref, pos, live, ids_b, lens_b)
            p_live = pos[live]
            self.h[b, live] = h_full[0, p_live]
            self.cos[b, live] = cos_t[0, p_live]
            self.sin[b, live] = sin_t[0, p_live]
        # the per-tensor DEscales, calibrated on every live entry's whole-sequence K / V (a static serving calibration)
        all_k = torch.cat([r[0].k[0].reshape(-1) for r in refs.values()]) if refs else torch.ones(1, device=_DEV)
        all_v = torch.cat([r[0].v[0].reshape(-1) for r in refs.values()]) if refs else torch.ones(1, device=_DEV)
        self.k_scale, self.v_scale = _scale_of(all_k), _scale_of(all_v)
        # the two oracles per entry: on the DEQUANTIZED rows (prefix AND step rows through the e4m3 codes) and on the bf16 rows
        self.ref_out = torch.zeros(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.ref_o_gated = torch.zeros(B, s_q, h_q, D, dtype=dtype, device=_DEV)
        self.ref_o_gated_bf16 = torch.zeros(B, s_q, h_q, D, dtype=dtype, device=_DEV)
        self.ref_codes = {}  # b -> (K codes [L, h_kv, D], V codes): the prefix rows go into the pools; the step rows pin the write-through
        for b, (ref, pos, live, ids_b, lens_b) in refs.items():
            L = self.kv_lens[b]
            k_codes, v_codes = _quantize(ref.k[0], self.k_scale), _quantize(ref.v[0], self.v_scale)
            self.ref_codes[b] = (k_codes, v_codes)
            kd, vd = _dequantize(k_codes, self.k_scale), _dequantize(v_codes, self.v_scale)
            p_live = pos[live]
            rows = ids_b.expand(s_q, top_k)
            lens_arg = lens_b.expand(s_q) if block_lens else None
            allowed = qsa_visible_mask(rows, lens_arg, pos, L, L, BS, top_k=top_k, pos0=L - s_q)[live]
            q_rows = ref.q[0][p_live].float()
            gate_rows = ref.gate[0][p_live].float()
            for which, (kk_all, vv_all, o_dst) in (("deq", (kd, vd, self.ref_o_gated)), ("bf16", (ref.k[0], ref.v[0], self.ref_o_gated_bf16))):
                o = torch.zeros(int(live.sum()), h_q, D, dtype=torch.float32, device=_DEV)
                for hh in range(h_q):
                    kk, vv = kk_all[:, hh // G].float(), vv_all[:, hh // G].float()
                    sc = (q_rows[:, hh] @ kk.t()) * scale
                    sc = sc.masked_fill(~allowed, float("-inf"))
                    rmax = sc.amax(dim=-1)
                    dead = torch.isinf(rmax) & (rmax < 0)
                    safe = torch.where(dead, torch.zeros_like(rmax), rmax)
                    p = torch.where(allowed, torch.exp(sc - safe[:, None]), torch.zeros_like(sc))
                    den = p.sum(dim=-1)
                    den_safe = torch.where(dead, torch.ones_like(den), den)
                    oc = (p @ vv) / den_safe[:, None]
                    o[:, hh] = torch.where(dead[:, None], torch.zeros_like(oc), oc)
                o16 = o.to(dtype)  # the oracle's single rounding of O, then the gate in fp32, then one rounding
                og = (o16.float() * torch.sigmoid(gate_rows)).to(dtype)
                o_dst[b, live] = og
                if which == "deq":
                    self.ref_out[b, live] = (og.reshape(-1, h_q * D).float() @ self.w["w_o"].float().t()).to(dtype)
        self.block_ids = torch.stack([x.reshape(top_k) for x in ids_rows]).contiguous()
        self.block_lens = torch.stack([x.reshape(()) for x in lens_rows]).contiguous()
        assert self.block_ids.dtype == torch.int32 and self.block_lens.dtype == torch.int32
        # pages: block 0 reserved (NULL); each live entry gets ceil(L / P) pages from a permutation of 1 .. n_pages - 1; -1 past them
        need = sum(-(-L // page) for L in self.kv_lens)
        self.n_pages = n_pages if n_pages is not None else need + 1 + 2
        assert self.n_pages >= need + 1, (self.n_pages, need)
        perm = (torch.randperm(self.n_pages - 1, generator=g, device=_DEV) + 1).tolist()
        self.max_pages = max(-(-L_max // page), 1)
        self.block_table = torch.full((B, self.max_pages), -1, dtype=torch.int32, device=_DEV)
        it = iter(perm)
        slots = torch.full((B * s_q,), -1, dtype=torch.int32, device=_DEV)
        self.k_cache, self.v_cache = _pools(self.n_pages, h_kv, page, layout, E4M3)
        self.index_cache = torch.full((self.n_pages, page, geom.qsa.index_head_dim), float("nan"), dtype=dtype, device=_DEV) if index_band else None
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                self.block_table[b] = 0  # the dead entry: the NULL block, slots -1
                continue
            pages = [next(it) for _ in range(-(-L // page))]
            self.block_table[b, : len(pages)] = torch.tensor(pages, dtype=torch.int32, device=_DEV)
            k_codes, v_codes = self.ref_codes[b]
            for p in range(max(L - s_q, 0)):  # the PREFIX: the torch-quantized codes into the pools
                pg, off = pages[p // page], p % page
                self.k_cache[pg, :, off] = k_codes[p]
                self.v_cache[pg, :, off] = v_codes[p]
            for j in range(s_q):  # the step's live tokens: written through (and quantized) by the block
                p = L - s_q + j
                if p >= 0:
                    slots[b * s_q + j] = pages[p // page] * page + p % page
        self.slot_mapping = slots
        self.kv_lens_t = torch.tensor(self.kv_lens, dtype=torch.int32, device=_DEV)
        self.prefix_bits = (_bits(self.k_cache).clone(), _bits(self.v_cache).clone())
        self.blk_kw = dict(
            paged_kv_page_size=page,
            kv_source=KV_SOURCE_PAGED,
            sample_k_cache=self.k_cache,
            sample_v_cache=self.v_cache,
            sample_block_table=self.block_table,
            fuse_gate=fuse_gate,
            kv_cache_dtype=E4M3,
        )
        self.out = torch.empty(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.blk = GatedAttentionBlockFwd(
            self.h, self.w["w_qkvg"], self.w["w_q_norm"], self.w["w_k_norm"], self.cos, self.sin, self.w["w_o"], self.out, self.geom, **self.blk_kw
        )
        assert isinstance(self.blk._sdpa, _UpcastSparseSdpa) and self.blk.upcast_decode and self.blk.inplace_qkv is False and self.blk.gate_fused is False
        self.ws = None

    def compile(self):
        assert self.blk.check_support()
        self.blk.compile()
        self.ws = torch.full((self.blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
        return self

    def execute(
        self,
        *,
        out=None,
        ws=None,
        h=None,
        cos=None,
        sin=None,
        kv_lens=None,
        block_table=None,
        slot_mapping=None,
        ids=None,
        lens=None,
        k_scale=None,
        v_scale=None,
        stream=None,
    ):
        out = self.out if out is None else out
        self.blk.execute(
            self.h if h is None else h,
            self.w["w_qkvg"],
            self.w["w_q_norm"],
            self.w["w_k_norm"],
            self.cos if cos is None else cos,
            self.sin if sin is None else sin,
            self.w["w_o"],
            out,
            self.ws if ws is None else ws,
            block_ids=self.block_ids if ids is None else ids,
            block_lens=(self.block_lens if lens is None else lens) if self.block_lens_given else None,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            block_table=self.block_table if block_table is None else block_table,
            kv_lens=self.kv_lens_t if kv_lens is None else kv_lens,
            slot_mapping=self.slot_mapping if slot_mapping is None else slot_mapping,
            index_k_raw=self.index_cache,
            current_stream=stream,
            k_scale=self.k_scale if k_scale is None else k_scale,
            v_scale=self.v_scale if v_scale is None else v_scale,
        )
        return out

    def run(self):
        if self.ws is None:
            self.compile()
        self.out.fill_(_sentinel(self.dtype))
        self.execute()
        torch.cuda.synchronize()
        return self.out

    def views(self, ws=None):
        """The compact normed Q and K (out of place), the slab's GATE and V bands, the O slot, the slab -- what the e4m3 mode writes."""
        ws = self.ws if ws is None else ws
        g, lay = self.geom, self.blk._ws
        B, S = self.blk.batch, self.blk.seq_len
        t = B * S
        proj = _view(ws, lay.proj, (t, g.n_qkvg), self.dtype)
        _o_q, o_g, _o_k, o_v = g.qkvg_offsets[:4]
        q = _view(ws, lay.q, (t, g.h_q, g.d_head), self.dtype).view(B, S, g.h_q, g.d_head)
        k = _view(ws, lay.k, (t, g.h_kv, g.d_head), self.dtype).view(B, S, g.h_kv, g.d_head)
        gate = _cols(proj, o_g, g.h_q, g.d_head).view(B, S, g.h_q, g.d_head)
        v = _cols(proj, o_v, g.h_kv, g.d_head).view(B, S, g.h_kv, g.d_head)
        o = _view(ws, lay.o, (t, g.h_q, g.d_head), self.dtype).view(B, S, g.h_q, g.d_head)
        return q, gate, k, v, o, proj

    def label(self):
        return f"[{self.geom.h_q}/{self.geom.h_kv} S_q={self.s_q} e4m3 p{self.page} {self.layout} gate={'fused->launch' if self.fuse_gate else 'launch'} lens={self.kv_lens}]"

    def check(self, out=None, ws=None, *, what="out"):
        """The standard assertions: finite, no sentinel, dead rows exact, within the budget vs the oracle on the DEQUANTIZED pool (out, the
        gated O slot), the pools' step rows BITWISE torch's quantization of the block's own K rows and V band, the prefix rows and
        the NULL block untouched; the distance to the bf16 oracle printed (no gate)."""
        out = self.out if out is None else out
        ws = self.ws if ws is None else ws
        sent = _sentinel(self.dtype)
        live = self.live_rows
        dead = ~live
        q, gate, k, v, o, proj = self.views(ws)
        assert torch.isfinite(out.float()).all(), f"{what}: non-finite out"
        assert not (out.float() == sent).any(), f"{what}: a sentinel survived in out"
        assert torch.isfinite(o.float()).all(), f"{what}: non-finite gated O (a row the decode tile never wrote keeps the workspace fill)"
        if dead.any():
            assert (out[dead] == 0).all() and (o[dead] == 0).all(), f"{what}: a dead row's out and O must be exactly 0"
        c = _cos(out[live], self.ref_out[live])
        d_out = float((out[live].float() - self.ref_out[live].float()).abs().max())
        d_o = float((o[live].float() - self.ref_o_gated[live].float()).abs().max())
        d_o_bf16 = float((o[live].float() - self.ref_o_gated_bf16[live].float()).abs().max())
        assert c > COS_OUT, f"{what}: cos(out) {c} <= {COS_OUT}"
        assert d_o <= ATOL, f"{what}: gated O off the oracle on the dequantized pool by {d_o} (budget {ATOL})"
        # the quantizing write-through: the step's live rows are torch's e4m3 codes of the block's own normed K rows / V band, bitwise
        slot = self.slot_mapping
        lv = slot >= 0
        s_ = slot[lv].long()
        kf, vf = k.reshape(-1, self.geom.h_kv, D), v.reshape(-1, self.geom.h_kv, D)
        assert torch.equal(
            _bits(self.k_cache[s_ // self.page, :, s_ % self.page, :]), _bits(_quantize(kf[lv], self.k_scale))
        ), f"{what}: the K pool rows are not the quantized K rows"
        assert torch.equal(
            _bits(self.v_cache[s_ // self.page, :, s_ % self.page, :]), _bits(_quantize(vf[lv], self.v_scale))
        ), f"{what}: the V pool rows are not the quantized V rows"
        for pool, before in zip((self.k_cache, self.v_cache), self.prefix_bits):
            now = _bits(pool)
            touched = (now != before).any(dim=-1).any(dim=1)  # [n_pages, page] rows whose bits moved
            rows_moved = touched.nonzero()
            written = set((int(p) for p in (s_ // self.page * self.page + s_ % self.page).tolist()))
            assert all(int(pg) * self.page + int(off) in written for pg, off in rows_moved), f"{what}: a pool row outside the step's slots moved"
            assert not touched[0].any(), f"{what}: the NULL block moved"
        if self.index_cache is not None:
            raw = index_k_raw_view(proj, self.geom, self.blk.batch, self.blk.seq_len)  # [B, S, 1, index_head_dim]
            assert torch.equal(_bits(self.index_cache[s_ // self.page, s_ % self.page, :]), _bits(raw.reshape(-1, raw.shape[-1])[lv]))
            assert (~torch.isnan(self.index_cache.float())).all(-1).sum().item() == int(lv.sum())
        print(
            f"\n{self.label()} {what}: cos(out) {c:.6f} max|d out| {d_out:.4e} max|d O_gated| vs the DEQUANTIZED oracle {d_o:.4e} (budget {ATOL}); "
            f"vs the PRE-quantization bf16 oracle {d_o_bf16:.4e} (reported, no gate); dead rows {int(dead.sum())}; k_scale {float(self.k_scale):.4e} v_scale {float(self.v_scale):.4e}"
        )
        return dict(cos=c, d_out=d_out, d_o=d_o, d_o_bf16=d_o_bf16, dead=int(dead.sum()))


def _two_launches(case: _Fp8Case, what: str):
    a = case.run().clone()
    o_a = case.views()[4].clone()
    b = case.run().clone()
    o_b = case.views()[4].clone()
    assert torch.equal(_bits(a), _bits(b)) and torch.equal(_bits(o_a), _bits(o_b)), f"{what}: two launches are not bitwise"
    return case.check(what=what)


# (geometry, S_q, kv_lens, page, layout, fuse_gate, block_lens, index_band, seed)
_LADDER = [
    pytest.param(GEOM_SMALL, 1, (2052, 8192, 0, 4097), 16, "hnd", False, True, False, 1, id="Sq1-B4-p16-hnd-dead-entry-gate-launch"),
    pytest.param(GEOM_SMALL, 4, (300, 0, 1000, 77), 64, "nhd", True, False, False, 2, id="Sq4-B4-p64-nhd-no-block-lens-fuse-gate"),
    pytest.param(
        GEOM_SMALL, 3, (2, 20000, 7), 32, "unified", True, True, False, 3, id="Sq3-B3-p32-unified-shorter-than-the-step-20000"
    ),  # a 2-token sequence under a 3-row step: one row at a negative position (a 1-token sequence would hand the fp32 reference a K = 1 matmul, which torch 2.10 routes to a Triton kernel this node cannot build for sm_107a)
    pytest.param(GEOM_FLASH_NEXT, 1, (4096,), 16, "hnd", True, True, False, 4, id="Sq1-B1-flash-next-24-2-p16"),
    pytest.param(GEOM_SMALL, 3, (8192, 2052), 16, "hnd", False, True, True, 5, id="Sq3-B2-p16-index-band-raw-key-write-through"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, s_q, kv_lens, page, layout, fuse_gate, block_lens, index_band, seed", _LADDER)
def test_e4m3_decode_matches_the_oracle_on_the_dequantized_pool(geom_kw, s_q, kv_lens, page, layout, fuse_gate, block_lens, index_band, seed):
    """The e4m3 decode ladder (module docstring): every cell within the budget vs the step-0-anchored oracle on the DEQUANTIZED pool,
    dead rows exact, the write-through rows the torch cast bitwise, two launches bitwise; the bf16-oracle distance reported."""
    case = _Fp8Case(geom_kw, s_q, kv_lens, page=page, layout=layout, seed=seed, fuse_gate=fuse_gate, block_lens=block_lens, index_band=index_band)
    case.compile()
    assert case.blk.decode_split_kv == 1 and case.blk._sdpa._impl.kernel_template == "decode_d256_f16"
    _two_launches(case, "ladder")


@requires_rubin
@pytest.mark.parametrize("r", [1, 3], ids=lambda r: f"pos0-mod4-{r}")
def test_mtp_rows_over_e4m3_pools_share_the_step0_list(r):
    """The speculative rows of a SHORT sequence at a residue of the step-0 position where a block completes after step 0 (``r = 1``)
    and where none does (``r = 3``): the e4m3 mode's rows within the budget of the step-0-anchored oracle on the dequantized pool."""
    s_q = 4
    L = 56 + r + s_q
    case = _Fp8Case(GEOM_SMALL, s_q, (L, L + 4), page=16, seed=10 + r, fuse_gate=True)
    case.compile()
    _two_launches(case, f"mtp r={r}")


@requires_rubin
def test_get_workspace_size_is_honest_and_holds_the_gathered_scratch():
    """The gathered scratch, the e4m3 staging rows and the inverse scales sit inside ``get_workspace_size()``; nothing past the size is
    written: a sentinel tail survives an execute; the decode tile's own scratch is 0 (unsplit)."""
    case = _Fp8Case(GEOM_SMALL, 2, (4096, 512, 0), page=16, seed=31, fuse_gate=True)
    case.compile()
    blk = case.blk
    lay = blk._layout()
    assert blk._sdpa.scratch_workspace_bytes() == 0 and blk.get_workspace_size() >= lay.total_bytes
    assert lay.kv_len_t - lay.kv_gathered == 3 * 2 * 2 * 2 * 2176 * 256 * 2
    need = blk.get_workspace_size()
    ws = torch.empty(need + 65536, dtype=torch.uint8, device=_DEV)
    ws[need:].fill_(0xA5)
    ws[:need].fill_(0x7F)
    out = torch.empty_like(case.out)
    case.execute(out=out, ws=ws)
    torch.cuda.synchronize()
    assert (ws[need:] == 0xA5).all(), "bytes past get_workspace_size() were written"
    case.check(out, ws, what="sentinel-tailed workspace")
    # the scratch was filled: every row past an item's count is 0.0 (zero_fill="full"), nothing kept the 0x7F7F fill
    items = 3 * 2 * case.geom.h_kv
    scratch = _view(ws, lay.kv_gathered, (items, 2, 2176, 256), BF16)
    kv_len_t = _view(ws, lay.kv_len_t, (items,), torch.int32)
    assert torch.isfinite(scratch.float()).all() and int(kv_len_t.max()) <= 2054 and int(kv_len_t.min()) == 0
    for i in range(items):
        n = int(kv_len_t[i])
        assert (scratch[i, :, n:] == 0).all(), f"item {i}: a row past kv_len_t={n} is not zero"
    inv = _view(ws, lay.kv_inv, (2,), torch.float32)
    assert torch.equal(inv, torch.reciprocal(torch.cat([case.k_scale, case.v_scale])))


@requires_rubin
def test_the_quantizing_write_through_on_the_prefill_form_matches_the_torch_cast_and_leaves_out_unchanged():
    """The write-through-only form (the sparse block over its OWN K / V with e4m3 pools, ``kv_source="self"``): the pools' rows are
    torch's e4m3 codes of the block's post-RoPE K / V BITWISE, and ``out`` is BITWISE the bf16-pool twin's (the write-through never
    changes attention); padded slots (-1) write nothing, every unwritten slot still NaN."""
    B, S, page = 2, 64, 16
    geom = GatedAttentionBlockGeometry(**GEOM_SMALL, qsa=QsaSpec(top_k=512))
    g = torch.Generator(device=_DEV).manual_seed(77)
    randn = lambda *shape, std=0.02: (torch.randn(*shape, generator=g, device=_DEV, dtype=torch.float32) * std).to(BF16)  # noqa: E731
    cos, sin = build_rope_tables(S, ROPE, base=10000.0, batch=B, device=_DEV, dtype=BF16)
    h, w_qkvg, w_o = randn(B, S, geom.d_model, std=1.0), randn(geom.n_qkvg, geom.d_model), randn(geom.d_model, geom.h_q * D)
    ones = torch.ones(D, dtype=BF16, device=_DEV)
    ids_b, lens_b = full_block_ids(torch.arange(S, device=_DEV), 512, BS)
    ids = ids_b[None].expand(B, S, 512).contiguous()
    lens = lens_b[None].expand(B, S).contiguous()
    slots = torch.arange(B * S, dtype=torch.int32, device=_DEV)
    slots[::9] = -1  # a padded token every ninth row writes nothing
    n_pages = -(-(B * S) // page) + 2
    k_scale, v_scale = torch.tensor([0.0123], device=_DEV), torch.tensor([0.0456], device=_DEV)
    runs = {}
    for tag, pool_dtype, attr, scales in (("e4m3", E4M3, E4M3, dict(k_scale=k_scale, v_scale=v_scale)), ("bf16", BF16, None, {})):
        kc, vc = _pools(n_pages, geom.h_kv, page, "hnd", pool_dtype)
        out = torch.empty(B, S, geom.d_model, dtype=BF16, device=_DEV)
        blk = GatedAttentionBlockFwd(h, w_qkvg, ones, ones, cos, sin, w_o, out, geom, paged_kv_page_size=page, kv_cache_dtype=attr)
        assert blk.check_support()
        blk.compile()
        ws = torch.full((blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
        blk.execute(h, w_qkvg, ones, ones, cos, sin, w_o, out, ws, block_ids=ids, block_lens=lens, k_cache=kc, v_cache=vc, slot_mapping=slots, **scales)
        torch.cuda.synchronize()
        lay = blk._ws
        proj = _view(ws, lay.proj, (B * S, geom.n_qkvg), BF16)
        _o_q, _o_g, o_k, o_v = geom.qkvg_offsets[:4]
        runs[tag] = (out.clone(), kc, vc, _cols(proj, o_k, geom.h_kv, D).clone(), _cols(proj, o_v, geom.h_kv, D).clone(), blk)
    out8, kc8, vc8, k_rows, v_rows, blk8 = runs["e4m3"]
    out16, kc16, vc16, k_rows16, v_rows16, _ = runs["bf16"]
    assert isinstance(blk8._cache_write, _QuantizedCacheWrite) and blk8.inplace_qkv is True and not blk8.upcast_decode
    assert torch.equal(_bits(out8), _bits(out16)), "the e4m3 write-through changed the block's own attention"
    assert torch.equal(_bits(k_rows), _bits(k_rows16)) and torch.equal(_bits(v_rows), _bits(v_rows16))
    lv = slots >= 0
    s_ = slots[lv].long()
    assert torch.equal(_bits(kc8[s_ // page, :, s_ % page, :]), _bits(_quantize(k_rows[lv], k_scale)))
    assert torch.equal(_bits(vc8[s_ // page, :, s_ % page, :]), _bits(_quantize(v_rows[lv], v_scale)))
    assert torch.equal(_bits(kc16[s_ // page, :, s_ % page, :]), _bits(k_rows[lv]))
    for pool in (kc8, vc8):
        assert int((~torch.isnan(pool.float())).all(-1).sum()) == int(lv.sum()) * geom.h_kv, "an unwritten or padded slot moved"
    print(f"\nprefill write-through e4m3: {int(lv.sum())} of {B * S} tokens written, out bitwise the bf16-pool twin, pools bitwise the torch cast")


def _written(pool):
    return int((~torch.isnan(pool.float())).all(-1).sum())


@requires_rubin
def test_capture_of_the_first_execute_replays_bitwise_and_takes_new_scales_lengths_lists_slots_and_inputs():
    """The serving stack's defaults: the FIRST execute of a fresh declaration captured on a NEW stream with no eager warm-up under
    ``set_sync_debug_mode("error")`` (no host sync in the path: scales, lengths, tables, lists and slots are device data; the inverse
    scales are a device op on the capture stream); the capture launches nothing, the replay is BITWISE a fresh eager run; new
    ``k_scale`` / ``v_scale`` / ``kv_lens`` / ``block_table`` / ``block_ids`` / ``block_lens`` / ``slot_mapping`` / ``h`` / RoPE rows
    through the captured tensors replay bitwise the eager run over them."""
    case = _Fp8Case(GEOM_SMALL, 2, (300, 0, 1000, 77), page=16, seed=41, fuse_gate=True)  # fresh: never executed
    case.blk.check_support()
    case.blk.compile()
    case.ws = torch.full((case.blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
    case.out.fill_(float("nan"))
    k_before = _written(case.k_cache)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.execute()  # the FIRST execute of this declaration, captured
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(case.out).all() and _written(case.k_cache) == k_before, "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        replayed = case.out.clone()
        case.check(replayed, what="replay")
        case.out.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(_bits(case.out), _bits(replayed)), "the replay of the captured first execute differs from the eager run"
        # a NEW step through the captured tensors: new scales (a recalibration), new lengths / table / lists / slots / h / RoPE rows -- the
        # oracle of the new step through a sibling case over the same shapes
        sib = _Fp8Case(GEOM_SMALL, 2, (301, 0, 1000, 77), page=16, seed=43, fuse_gate=True, n_pages=case.n_pages)
        assert sib.block_table.shape == case.block_table.shape and sib.k_cache.shape == case.k_cache.shape
        assert not torch.equal(sib.k_scale, case.k_scale)
        case.k_cache.copy_(sib.k_cache)
        case.v_cache.copy_(sib.v_cache)
        case.block_table.copy_(sib.block_table)
        case.kv_lens_t.copy_(sib.kv_lens_t)
        case.slot_mapping.copy_(sib.slot_mapping)
        case.block_ids.copy_(sib.block_ids)
        case.block_lens.copy_(sib.block_lens)
        case.k_scale.copy_(sib.k_scale)
        case.v_scale.copy_(sib.v_scale)
        case.h.copy_(sib.h)
        case.cos.copy_(sib.cos)
        case.sin.copy_(sib.sin)
        case.w["w_qkvg"].copy_(sib.w["w_qkvg"])
        case.w["w_o"].copy_(sib.w["w_o"])
        case.prefix_bits = (_bits(case.k_cache).clone(), _bits(case.v_cache).clone())
        case.ref_out, case.ref_o_gated, case.ref_o_gated_bf16, case.live_rows, case.kv_lens = (
            sib.ref_out,
            sib.ref_o_gated,
            sib.ref_o_gated_bf16,
            sib.live_rows,
            sib.kv_lens,
        )
        case.out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed2 = case.out.clone()
        case.check(replayed2, what="replay over new scales / lengths / table / lists / slots / inputs")
        case.out.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(
            _bits(case.out), _bits(replayed2)
        ), "the replay over new scales / lengths / tables / lists / slots differs from the eager run over them"
    finally:
        graph.reset()
