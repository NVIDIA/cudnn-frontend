# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm107_mxfp8`` THD / varlen: the packed d = 256 block-scale MXFP8 E4M3 backward on the Rubin line, end to end.

The twin of ``test_sdpa_bwd_thd_fp8_sm107.py`` for the MXFP8 row -- reuse by import, never copy: the tail sentinel, the stage-3
band twins and the case tables come from the bf16 THD suite, the per-sequence packing and the bitwise helper from the fp8 twin,
the tolerance table, the ``ds_policy`` fixture and the workspace read-back helpers from ``test_sdpa_bwd_mxfp8_sm107.py``.

Two surfaces:

* ``_run_mx_direct`` drives ``SdpaBwdDslSm107Mxfp8(thd=True, ...)`` DIRECTLY over PACKED e4m3 payloads (rowwise AND columnwise
  quantizations of Q / K / dO, rowwise V), the bf16 ``o_f16`` / ``dO_f16`` ports and the seven PACKED scale-factor tensors --
  the numerics of the chain one layer below the plan machinery, so a failure localises to the kernels.  Every accept cell of
  this module runs on this tier.
* ``_run_mx_graph`` goes through the ragged ``sdpa_mxfp8_backward`` GRAPH with the engine pinned (the quantized backward node
  declares its packed totals through ``max_total_seq_len_q/kv``).

The scale factors follow the FORWARD's packed convention -- per-sequence-TILE-padded, never token-indexed: every (head, 128-token
tile) of a sequence owns one F8_128x4 atom row (1024 B: two atoms, the two d-chunks of a rowwise tile or the two D-planes of a
columnwise one), the tiles of sequence ``b`` start at ``cu_sf[b] = SUM_{i<b} ceil(s_i / 128)`` and the columnwise tensors keep both
D-planes of a (head, tile) CONTIGUOUS (plane stride one atom) -- the dense D-plane-major order would read plane 1 from an
S-dependent wrong place.  The packed tile count is what the bound buffer's byte size says (``nbytes // (H * 1024)``), bounded by the
declared totals; the per-sequence pad rows / groups of the last tile hold whatever the producer left there (0xFF = E8M0 NaN in the
poisoned cells), and the chain's own device pre-pass zeroes the pads of the five tensors whose pad bytes reach an MMA (``sf_v``,
``sf_do``, ``sf_do_T``, ``sf_q_T``, ``sf_k_T``); ``sf_q`` / ``sf_k`` pads are harmless by construction (a NaN score on a select-dead
cell).  The RED-then-green pair below proves both halves per hazard tensor and per sequence.

The reference is the repo's MXFP8 backward oracle (``sdpa.mxfp8_ref.compute_ref_backward``) run PER SEQUENCE over the packed live
tokens, composing the chain's own dS rounding (``quantize_ds=True`` under the shipped block-scaled policy P-b, ``False`` under the
bf16-dS twin P-c), under the dense MXFP8 suite's recipe and never a looser one: dV under the fp8 recipe (``assert_close_fp8_grad``,
atol 0.08 / rtol 0.2 + the midpoint-flip budget), dK / dQ under the fp8 recipe on P-b and the bf16 row's (``_BF16_GRAD_TOL``) on P-c.
Stats come from the torch forward over the DEQUANTIZED operands (the exact natural-log log-sum-exp; there is no Rubin MXFP8 THD
forward to feed this row, so ``test_frost_forward_stats_feed_this_row`` stays dense in the sibling suite).  A fully masked row
carries the forward's convention (``O = 0``, ``LSE = -inf`` -> ``P = 0`` -> ``dQ = 0``) and the cells with such rows assert the exact
zeros on the OUTPUT directly.  No case table carries a length-1 sequence (a (1 x 1) matmul in the bf16 suite's fp64 oracle trips a
Triton codegen defect before any kernel runs); five rows is the shortest sequence here too.
"""

from __future__ import annotations

import inspect
import math
import re
from types import SimpleNamespace

import pytest
import torch

import cudnn
from frost_test_utils import cuda_launch_counts, cuda_launch_names, requires_dsl, requires_rubin, requires_sm80, select_engine
from test_sdpa_bwd_mxfp8_sm107 import (  # noqa: F401  (ds_policy: fixture by import)
    _BF16_GRAD_TOL,
    _GRAD_TOL,
    _active_policy,
    _atom_bytes_to_scales,
    _code_only,
    _def_body,
    _p_b,
    _p_c,
    _report,
    _workspace_region,
    ds_policy,
)
from test_sdpa_bwd_thd_fp8_sm107 import _StandaloneGraphShim, _bitwise, _cu, _finite, _graph_kw_to_direct
from test_sdpa_bwd_thd_sm107 import (
    _GQA_TWIN_CASES,
    _NO_KEY_ROW_CASES,
    _TAIL_SENTINEL,
    _TRIM_TWIN_CASES,
    _assert_empty_sequence_exactly_zero,
    _assert_tails_untouched,
    _plan_index,
    _rows_with_keys,
    _sentinel_tails,
    _spec_without_compiling,
)

pytestmark = [pytest.mark.L0, requires_dsl]

_ENGINE = "sdpa_bwd_sm107_mxfp8"
_D = 256
_RUBIN_CC = (10, 7)
_T_E4M3 = torch.float8_e4m3fn
_BF16 = torch.bfloat16
_MX_BLOCK = 32
_SF_ATOM_ROWS = 128
_SF_GROUPS = _D // _MX_BLOCK  # 8 scale groups per rowwise row
_SF_PLANES = _D // _SF_ATOM_ROWS  # 2 D-planes of a columnwise tile
_SF_TILE_BYTES = _SF_ATOM_ROWS * _SF_GROUPS  # 1024 B per (head, 128-token tile), rowwise and columnwise alike
_PAYLOADS = ("q", "q_T", "k", "k_T", "v", "do", "do_T")
_SF_ROWWISE = ("sf_q", "sf_k", "sf_v", "sf_do")
_SF_COLUMNWISE = ("sf_q_T", "sf_k_T", "sf_do_T")
_SF_ALL = ("sf_q", "sf_q_T", "sf_k", "sf_k_T", "sf_v", "sf_do", "sf_do_T")
# The five scale-factor tensors whose per-sequence PAD bytes reach an MMA (the chain's pre-pass zeroes them) and the two whose pads are
# harmless by construction (a NaN score on a cell the q band / row_dead already selects to zero).
_SF_HAZARD = ("sf_v", "sf_do", "sf_do_T", "sf_q_T", "sf_k_T")
_SF_HARMLESS = ("sf_q", "sf_k")


@pytest.fixture(autouse=True)
def _mock_target_for_cross_arch_contracts(monkeypatch):
    # The host rejects probe the Rubin row off the Rubin line (the analyzer's cc faked to 10.7); mismatch() carries the sm_107a DSL
    # gate, so the fake device gets a fake compiler target too -- and the block-scaled chain (P-b) declines typed off the Rubin line
    # at check_support (the stage-3 arm's 576-column exclusive TMEM), so the plan-level pins see the device query answer SM107, as the
    # dense MXFP8 suite arranges it.
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != _RUBIN_CC:
        from cudnn.frost import buffers
        from cudnn.sdpa.bwd import prepared_sm107

        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
        monkeypatch.setattr(prepared_sm107, "_sm", lambda api: 107)


def _spec(name=_ENGINE):
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    spec = next((s for s in ENGINE_SPECS if s.name == name), None)
    assert spec is not None, f"{name} is not in cudnn.sdpa.bwd.engines.ENGINE_SPECS"
    return spec


def _ceil128(n):
    return -(-n // _SF_ATOM_ROWS) * _SF_ATOM_ROWS


def _tiles(n):
    return -(-n // _SF_ATOM_ROWS)


def _block_scaled():
    from cudnn.sdpa.bwd import config_sm107 as cfg

    return _active_policy() == cfg.DS_SF_P_B


def _binding_declares_totals() -> bool:
    """Whether the NATIVE ``sdpa_mxfp8_backward`` binding takes ``max_total_seq_len_q/kv`` (the C++ replay forwards every captured
    kwarg to it; pybind writes the signature into the docstring)."""
    return "max_total_seq_len_q" in (cudnn._pybind_module.backend_graph.sdpa_mxfp8_backward.__doc__ or "")


# --------------------------------------------------------------------------- the per-sequence quantization in the packed SF convention


class _SeqQuant:
    """Rowwise + columnwise MXFP8 of ONE sequence ``[1, H, s, D]`` (fp32), the F8_128x4 atoms laid out as the PACKED per-sequence-TILE-padded
    convention wants them: ``pay_d`` / ``pay_s`` e4m3 payloads ``[1, H, s, D]`` (rowwise: scaled along D; columnwise: scaled along s),
    ``tiles_d`` / ``tiles_s`` uint8 ``[H, ceil(s/128), 1024]`` -- per (head, tile) the two rowwise d-chunk atoms, or the two columnwise
    D-plane atoms CONTIGUOUS -- and the oracle's views: ``ref_d`` / ``ref_s`` ``[1, H, s, D]`` e4m3 and ``sfref_d`` / ``sfref_s`` the per-element
    fp32 dequant scales ``[H, s, D]`` (``sdpa.mxfp8_ref._dequant`` views them as the payload's shape).

    ``poison_pads`` names which of the two forms get their PAD bytes -- the rows past ``s`` (rowwise) / the 32-groups wholly past ``s``
    (columnwise) of the LAST tile -- set to 0xFF (E8M0 NaN): the producer-defined bytes the chain's pre-pass must zero.  A sequence of
    length 0 owns no token and no tile."""

    def __init__(self, t_1hsd, s, *, poison_rows=False, poison_cols=False):
        from sdpa.mxfp8_quant import e8m0_to_float, quantize_mxfp8_2d, swizzle_sf_columnwise, swizzle_sf_rowwise

        h, d = t_1hsd.shape[1], t_1hsd.shape[3]
        self.s, self.h = s, h
        if s == 0:
            empty = t_1hsd.new_zeros((1, h, 0, d))
            self.pay_d = self.pay_s = self.ref_d = self.ref_s = empty.to(_T_E4M3)
            self.sfref_d = self.sfref_s = empty.float().reshape(h, 0, d)
            self.tiles_d = torch.zeros((h, 0, _SF_TILE_BYTES), dtype=torch.uint8, device=t_1hsd.device)
            self.tiles_s = self.tiles_d.clone()
            return
        s_pad = _ceil128(s)
        x = t_1hsd.float().reshape(h, s, d)
        if s_pad != s:
            x = torch.nn.functional.pad(x, (0, 0, 0, s_pad - s))
        row_data, row_e, col_data, col_e = quantize_mxfp8_2d(x.reshape(h * s_pad, d), _T_E4M3)
        # the oracle's scales BEFORE any poisoning (its live rows / groups only; a poisoned pad byte is never dequantized here)
        self.sfref_d = torch.repeat_interleave(e8m0_to_float(row_e).reshape(h, s_pad, d // _MX_BLOCK), _MX_BLOCK, dim=2)[:, :s].contiguous()
        self.sfref_s = torch.repeat_interleave(e8m0_to_float(col_e).reshape(h, s_pad // _MX_BLOCK, d), _MX_BLOCK, dim=1)[:, :s].contiguous()
        if poison_rows and s_pad != s:
            row_e.view(h, s_pad, d // _MX_BLOCK)[:, s:, :] = 0xFF
        if poison_cols and s_pad != s:
            col_e.view(h, s_pad // _MX_BLOCK, d)[:, -(-s // _MX_BLOCK) :, :] = 0xFF
        n_tiles = s_pad // _SF_ATOM_ROWS
        self.pay_d = row_data.reshape(1, h, s_pad, d)[:, :, :s].contiguous()
        self.pay_s = col_data.reshape(1, h, s_pad, d)[:, :, :s].contiguous()
        self.ref_d, self.ref_s = self.pay_d, self.pay_s
        # rowwise atoms: (h, tile) major, then the two d-chunk atoms of the tile -> [H, tiles, 1024] as the F8_128x4 reorder leaves them
        self.tiles_d = swizzle_sf_rowwise(row_e).contiguous().view(torch.uint8).reshape(h, n_tiles, _SF_TILE_BYTES)
        # columnwise atoms: the F8_128x4 reorder of the TRANSPOSED scale matrix is D-plane-major over the whole tensor (plane, (h, tile),
        # 512 B); the packed THD convention keeps a (head, tile)'s planes contiguous -> transpose the plane axis inside each tile
        self.tiles_s = (
            swizzle_sf_columnwise(col_e)
            .contiguous()
            .view(torch.uint8)
            .reshape(_SF_PLANES, h, n_tiles, _SF_TILE_BYTES // _SF_PLANES)
            .permute(1, 2, 0, 3)
            .contiguous()
            .reshape(h, n_tiles, _SF_TILE_BYTES)
        )

    def deq_d(self):
        return self.ref_d.float() * self.sfref_d.view(self.ref_d.shape)


def _pack_sf(tiles_per_seq, h, *, rowwise, tail_tiles=0, poison_tail=False, device="cuda"):
    """The packed scale-factor tensor of one operand over ``tiles_per_seq`` (a ``[H, n_b, 1024]`` per sequence, in cu_seqlens order):
    rowwise ``(1, H, 128 * T_sf, 8)``, columnwise ``(1, H, 4 * T_sf, 256)`` -- the dense ``(B, H, ceil128(S), 8)`` / ``(B, H, ceil128(S) // 32, 256)``
    forms with the batch folded into the packed tile axis; the byte count is ``H * 1024 * T_sf`` either way.  ``tail_tiles`` appends
    capacity tiles past ``cu_sf[B]`` (0xFF when ``poison_tail`` -- the clamped SF maps must never read them).  A side with no live tile
    at all (every sequence empty on it) still binds ONE tile -- a zero-capacity SF buffer cannot be bound -- and the kernel's packed
    extent follows ``cu_sf[B] == 0``, so it is never read."""
    live = torch.cat(tiles_per_seq, dim=1) if tiles_per_seq else torch.zeros((h, 0, _SF_TILE_BYTES), dtype=torch.uint8, device=device)
    n_live = live.shape[1]
    n_total = max(n_live + tail_tiles, 1)
    out = torch.full((h, n_total, _SF_TILE_BYTES), 0xFF if poison_tail else 0, dtype=torch.uint8, device=device)
    if n_live:
        out[:, :n_live] = live
    if rowwise:
        return out.reshape(1, h, n_total * _SF_ATOM_ROWS, _SF_GROUPS)
    return out.reshape(1, h, n_total * (_SF_ATOM_ROWS // _MX_BLOCK), _D)


def _sf_tile_count(sf):
    """The packed tile count the chain derives from a bound SF buffer: its byte count over ``H * 1024``."""
    return sf.numel() // (sf.shape[1] * _SF_TILE_BYTES)


def _seq_forward(q_deq, k_deq, v_deq, scale, *, causal, bottom_right, window_left):
    """The torch forward of ONE sequence over the DEQUANTIZED operands ``[H, s, D]`` (K / V broadcast to the Q heads) under its own
    mask: ``(lse [H, s_q] natural log, o [H, s_q, D] fp32)``; a fully masked row (bottom-right with ``s_q > s_kv``, a window past the
    keys) yields ``LSE = -inf`` and ``O = 0`` -- the forward's contract, applied by the oracle itself."""
    s_q, s_kv = q_deq.shape[1], k_deq.shape[1]
    dev = q_deq.device
    s_raw = torch.einsum("hqd,hkd->hqk", q_deq, k_deq)
    rel = torch.arange(s_kv, device=dev).view(1, s_kv) - torch.arange(s_q, device=dev).view(s_q, 1)
    diag = (s_kv - s_q) if (causal and bottom_right) else 0
    masked = torch.zeros(s_q, s_kv, dtype=torch.bool, device=dev)
    if causal:
        masked |= rel > diag
    if window_left is not None:
        masked |= rel <= diag - window_left
    s_scaled = (s_raw * scale).masked_fill(masked, float("-inf"))
    lse = torch.logsumexp(s_scaled, dim=-1)  # -inf on a fully masked row
    p = torch.exp(s_scaled - lse.unsqueeze(-1)).nan_to_num(0.0)
    o = torch.einsum("hqk,hkd->hqd", p, v_deq)
    return lse.contiguous(), o


def _thd_mx_case(
    lens_q,
    lens_kv,
    h,
    hkv=None,
    *,
    cap_q=None,
    cap_kv=None,
    poison=False,
    sf_tail_tiles=0,
    token_pad_heads=0,
    poison_sf_pads=(),
    poison_sf_seqs=None,
    seed=7,
    causal=False,
    bottom_right=False,
    window_left=None,
    quantize_ds=True,
    device="cuda",
    oracle=True,
):
    """Packed MXFP8 Q / Q_T / K / K_T / V / dO / dO_T (one ``_SeqQuant`` per sequence, concatenated in cu_seqlens order), the seven
    PACKED per-sequence-tile-padded scale-factor tensors, the bf16 ``o_f16`` / ``dO_f16`` ports and the packed natural-log Stats from
    the torch forward over the dequantized operands, per sequence; the per-sequence reference gradients through
    ``mxfp8_ref.compute_ref_backward`` (``quantize_ds`` = the dS rounding the chain composes).

    ``cap_*`` over-allocates the packed payload buffers past the real totals; with ``poison`` the slack is NaN (e4m3 has a NaN code) and
    the Stats / O tails NaN too; ``sf_tail_tiles`` appends that many SF tiles past ``cu_sf[B]`` per side (0xFF under ``poison``) -- an int
    for both sides, or a ``(q, kv)`` pair for the dense-capacity cells, which declare MORE tiles than the packed bound on purpose.
    ``token_pad_heads`` pads every packed payload's TOKEN stride: the buffers are allocated ``[1, cap, H + pad, D]`` and the port is their
    first ``H`` heads (token stride ``(H + pad) x D``, head stride ``D`` -- the padded packed rows the adapter admits).
    ``poison_sf_pads`` names the SF tensors whose per-sequence pad bytes are 0xFF, for the sequences in ``poison_sf_seqs`` (every
    sequence when None).  ``window_left`` is the graph's band bound = the number of keys a row keeps (the adapter takes
    ``window_size_left = window_left - 1``).  Unit-normal bf16-rounded inputs on a CPU generator (the dataset does not depend on the
    GPU's SM count).  ``oracle=False`` builds the GEOMETRY only (no forward, no reference) -- for a host probe that lowers the ragged
    graph and never runs it."""
    from sdpa.mxfp8_ref import compute_ref_backward

    dev, b = device, len(lens_q)
    lens_q, lens_kv = [int(x) for x in lens_q], [int(x) for x in lens_kv]
    assert len(lens_kv) == b
    assert all(n != 1 for n in lens_q + lens_kv), "never a length-1 sequence (the fp64 oracle's (1 x 1) matmul trips a Triton defect)"
    hkv = h if hkv is None else hkv
    t_q, t_kv = sum(lens_q), sum(lens_kv)
    cap_q, cap_kv = cap_q or t_q, cap_kv or t_kv
    assert cap_q >= t_q and cap_kv >= t_kv
    cu_q, cu_k = _cu(lens_q), _cu(lens_kv)
    poison_sf_pads = tuple(poison_sf_pads)
    assert all(name in _SF_ALL for name in poison_sf_pads), poison_sf_pads
    pad_seqs = set(range(b)) if poison_sf_seqs is None else set(int(i) for i in poison_sf_seqs)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scale = 1.0 / math.sqrt(_D)

    def draw(n, nh):  # bf16-rounded fp32 [1, nh, n, D] (the dense MXFP8 suite's dataset)
        return torch.randn(1, nh, max(n, 1), _D, generator=gen)[:, :, :n].to(_BF16).float().to(dev)

    def poisons(name, i):
        return i in pad_seqs and name in poison_sf_pads

    quant = {}  # per sequence: dict(q=_SeqQuant, k=..., v=..., do=...)
    lse_seq, o_seq, do_f16_seq = {}, {}, {}
    right = 0 if causal else None
    align = (cudnn.diagonal_alignment.BOTTOM_RIGHT if bottom_right else cudnn.diagonal_alignment.TOP_LEFT) if causal else None
    for i in range(b):
        sq_, sk_ = lens_q[i], lens_kv[i]
        q32, do32, k32, v32 = draw(sq_, h), draw(sq_, h), draw(sk_, hkv), draw(sk_, hkv)
        quant[i] = dict(
            q=_SeqQuant(q32, sq_, poison_rows=poisons("sf_q", i), poison_cols=poisons("sf_q_T", i)),
            k=_SeqQuant(k32, sk_, poison_rows=poisons("sf_k", i), poison_cols=poisons("sf_k_T", i)),
            v=_SeqQuant(v32, sk_, poison_rows=poisons("sf_v", i)),
            do=_SeqQuant(do32, sq_, poison_rows=poisons("sf_do", i), poison_cols=poisons("sf_do_T", i)),
        )
        do_f16_seq[i] = do32.to(_BF16)  # [1, h, s_q, D]: the bf16 dO port (the oracle's delta reads it)
        if not oracle or sq_ == 0:
            lse_seq[i] = torch.full((h, sq_), float("-inf"), device=dev)
            o_seq[i] = torch.zeros(h, sq_, _D, device=dev)
            continue
        if sk_ == 0:  # a sequence without keys: every row is dead (the forward's convention)
            lse_seq[i] = torch.full((h, sq_), float("-inf"), device=dev)
            o_seq[i] = torch.zeros(h, sq_, _D, device=dev)
            continue
        grp = h // hkv
        k_deq, v_deq = quant[i]["k"].deq_d()[0].repeat_interleave(grp, dim=0), quant[i]["v"].deq_d()[0].repeat_interleave(grp, dim=0)
        lse_seq[i], o_seq[i] = _seq_forward(quant[i]["q"].deq_d()[0], k_deq, v_deq, scale, causal=causal, bottom_right=bottom_right, window_left=window_left)

    def pack_tokens(parts, cap, nh, dt):  # [1, nh, s_i, D] per sequence -> [1, cap, nh, D], the live rows leading, the tail NaN (poison) or zero
        # ``token_pad_heads`` widens the slab: the port is the first nh heads of [1, cap, nh + pad, D] (token stride (nh + pad) * D)
        stor = torch.full((1, cap, nh + token_pad_heads, _D), float("nan") if poison else 0.0, device=dev).to(dt)[:, :, :nh]
        pieces = [x[0].permute(1, 0, 2) for x in parts if x.shape[2]]
        if pieces:
            live = torch.cat(pieces, dim=0)
            stor[0, : live.shape[0]] = live.to(dt)
        return stor

    q_p = pack_tokens([quant[i]["q"].pay_d for i in range(b)], cap_q, h, _T_E4M3)
    qT_p = pack_tokens([quant[i]["q"].pay_s for i in range(b)], cap_q, h, _T_E4M3)
    k_p = pack_tokens([quant[i]["k"].pay_d for i in range(b)], cap_kv, hkv, _T_E4M3)
    kT_p = pack_tokens([quant[i]["k"].pay_s for i in range(b)], cap_kv, hkv, _T_E4M3)
    v_p = pack_tokens([quant[i]["v"].pay_d for i in range(b)], cap_kv, hkv, _T_E4M3)
    do_p = pack_tokens([quant[i]["do"].pay_d for i in range(b)], cap_q, h, _T_E4M3)
    doT_p = pack_tokens([quant[i]["do"].pay_s for i in range(b)], cap_q, h, _T_E4M3)
    o_p = pack_tokens([o_seq[i][None] for i in range(b)], cap_q, h, _BF16)
    dof16_p = pack_tokens([do_f16_seq[i] for i in range(b)], cap_q, h, _BF16)
    lse = torch.full((1, h, cap_q), float("nan") if poison else float("-inf"), device=dev, dtype=torch.float32)
    for i in range(b):
        if lens_q[i]:
            lse[0, :, cu_q[i] : cu_q[i] + lens_q[i]] = lse_seq[i]
    tail_q, tail_kv = (sf_tail_tiles, sf_tail_tiles) if isinstance(sf_tail_tiles, int) else tuple(int(x) for x in sf_tail_tiles)
    sf = dict(
        sf_q=_pack_sf([quant[i]["q"].tiles_d for i in range(b)], h, rowwise=True, tail_tiles=tail_q, poison_tail=poison, device=dev),
        sf_q_T=_pack_sf([quant[i]["q"].tiles_s for i in range(b)], h, rowwise=False, tail_tiles=tail_q, poison_tail=poison, device=dev),
        sf_k=_pack_sf([quant[i]["k"].tiles_d for i in range(b)], hkv, rowwise=True, tail_tiles=tail_kv, poison_tail=poison, device=dev),
        sf_k_T=_pack_sf([quant[i]["k"].tiles_s for i in range(b)], hkv, rowwise=False, tail_tiles=tail_kv, poison_tail=poison, device=dev),
        sf_v=_pack_sf([quant[i]["v"].tiles_d for i in range(b)], hkv, rowwise=True, tail_tiles=tail_kv, poison_tail=poison, device=dev),
        sf_do=_pack_sf([quant[i]["do"].tiles_d for i in range(b)], h, rowwise=True, tail_tiles=tail_q, poison_tail=poison, device=dev),
        sf_do_T=_pack_sf([quant[i]["do"].tiles_s for i in range(b)], h, rowwise=False, tail_tiles=tail_q, poison_tail=poison, device=dev),
    )
    tiles_q, tiles_kv = sum(_tiles(n) for n in lens_q), sum(_tiles(n) for n in lens_kv)
    for name in _SF_ALL:
        kv_side = name in ("sf_k", "sf_k_T", "sf_v")
        side_tiles, tail, cap = (tiles_kv, tail_kv, cap_kv) if kv_side else (tiles_q, tail_q, cap_q)
        assert _sf_tile_count(sf[name]) == max(side_tiles + tail, 1), name
        if isinstance(sf_tail_tiles, int):
            # an int tail stays inside the packed bound (a mis-built case); an explicit per-side pair is a dense-capacity cell, which
            # declares MORE tiles than the bound on purpose -- the plan's capacity grows to the declared sample's own count
            assert _sf_tile_count(sf[name]) <= _tiles(cap) + b, f"{name}: the case exceeds the packed SF tile bound ceil(T_cap/128) + B"
    live = [i for i in range(b) if lens_q[i] > 0 and lens_kv[i] > 0] if oracle else []

    def ref_bwd(i, *, quantize_ds_=None):
        """The MXFP8 oracle on sequence ``i`` alone (its own quantization, Stats and mask), ``(dQ, dK, dV)`` as ``[1, H, s, D]`` bf16."""
        qq, kk, vv, dd = quant[i]["q"], quant[i]["k"], quant[i]["v"], quant[i]["do"]
        dq, dk, dv, _dsink = compute_ref_backward(
            qq.ref_d, qq.ref_s, kk.ref_d, kk.ref_s, vv.ref_d, o_seq[i][None].to(_BF16), do_f16_seq[i], dd.ref_d, dd.ref_s, scale,
            qq.sfref_d, qq.sfref_s, kk.sfref_d, kk.sfref_s, vv.sfref_d, dd.sfref_d, dd.sfref_s,
            torch_itype=_T_E4M3, torch_otype=_BF16, left_bound=window_left, right_bound=right, diag_align=align,
            stats=lse_seq[i][None, :, :, None], quantize_ds=quantize_ds if quantize_ds_ is None else quantize_ds_,
        )  # fmt: skip
        return dq, dk, dv

    refs = {i: dict(zip(("dQ", "dK", "dV"), ref_bwd(i))) for i in live}
    return SimpleNamespace(
        b=b, h=h, hkv=hkv, d=_D, dtype=_BF16, grad_dtype=_BF16, scale=scale, causal=causal, bottom_right=bottom_right, window_left=window_left,
        lens_q=lens_q, lens_kv=lens_kv, cu_q=cu_q, cu_k=cu_k, t_q=t_q, t_kv=t_kv, cap_q=cap_q, cap_kv=cap_kv, live=live,
        tiles_q=tiles_q, tiles_kv=tiles_kv, quant=quant, token_pad_heads=token_pad_heads,
        q=q_p, q_T=qT_p, k=k_p, k_T=kT_p, v=v_p, do=do_p, do_T=doT_p, o=o_p, do_f16=dof16_p, lse=lse, sf=sf,
        lse_seq=lse_seq, o_seq=o_seq, do_f16_seq=do_f16_seq, refs=refs, ref_bwd=ref_bwd, quantize_ds=quantize_ds,
    )  # fmt: skip


# --------------------------------------------------------------------------- the per-sequence comparison under the MXFP8 row's recipe


def _check_mx(case, dq, dk, dv):
    """Per-sequence comparison under the MXFP8 row's recipe, COLLECTED and asserted at the end (which gradient of which sequence is
    wrong is the attribution: dK and dQ come from the two stage-3 renderings, dV from the main kernel): dV under the fp8 recipe
    always; dK / dQ under the fp8 recipe on the block-scaled chain (both sides round dS to e4m3 per 32-block) and under the bf16 row's
    recipe on the bf16-dS twin.  A sequence empty on either side has no reference and is checked by its caller."""
    from sdpa.fp8 import assert_close_fp8_grad

    block_scaled = case.quantize_ds
    bad, verdicts = [], []
    for i in case.live:
        slq = slice(case.cu_q[i], case.cu_q[i] + case.lens_q[i])
        slk = slice(case.cu_k[i], case.cu_k[i] + case.lens_kv[i])
        for name, got, keys in (("dQ", dq[0, slq], case.lens_kv[i]), ("dK", dk[0, slk], case.lens_q[i]), ("dV", dv[0, slk], case.lens_q[i])):
            tag = f"seq {i} {name} (lens q={case.lens_q[i]} kv={case.lens_kv[i]})"
            g_ = got.permute(1, 0, 2)[None].float()  # [1, H, s, D]
            w_ = case.refs[i][name].float()
            if not _finite(g_):
                bad.append(f"{tag}: non-finite output ({int(torch.isnan(g_).sum())} NaN cells)")
                verdicts.append(bad[-1])
                continue
            fp8_recipe = name == "dV" or block_scaled
            tol = _GRAD_TOL if fp8_recipe else _BF16_GRAD_TOL
            _report(
                f"{tag} vs the {'e4m3-dS (1x32 both ways)' if block_scaled else 'fp32-dS'} oracle ({'fp8' if fp8_recipe else 'bf16'} recipe)",
                g_,
                w_,
                tol["atol"],
                tol["rtol"],
            )
            try:
                if fp8_recipe:
                    assert_close_fp8_grad(g_, w_, tol["atol"], tol["rtol"], tag=tag, keys=keys, budget=1e-5)
                else:
                    torch.testing.assert_close(g_, w_, **tol)
                verdicts.append(f"{tag}: ok")
            except AssertionError as exc:
                bad.append(f"{tag} vs the MXFP8 oracle: {str(exc).splitlines()[0]}")
                verdicts.append(bad[-1])
    assert not bad, "\n".join(verdicts)


def _check_mx_rows_with_keys(case, dq, dk, dv):
    """The oracle for a case with FULLY MASKED q rows (bottom-right with ``s_q > s_kv``, a top-left window past the keys): the rows
    without a key get ``dQ = 0`` EXACTLY -- asserted on the output directly (a diff against a reference proves nothing there) -- and
    contribute nothing to dK / dV; the whole sequence then passes the per-sequence oracle, which yields the same zeros on those rows
    (``LSE = -inf`` -> ``P = 0``)."""
    for i in case.live:
        r0, r1 = _rows_with_keys(case, i)
        q0, k0 = case.cu_q[i], case.cu_k[i]
        for lo, hi in ((0, r0), (r1, case.lens_q[i])):
            dead = dq[0, q0 + lo : q0 + hi].float()
            if dead.numel():
                assert (
                    _finite(dead) and not dead.any()
                ), f"seq {i}: dQ rows [{lo}, {hi}) have no key and must be exactly zero (max |.| {dead.abs().max().item():.3e})"
        if r1 <= r0:
            for name, got in (("dK", dk[0, k0 : k0 + case.lens_kv[i]]), ("dV", dv[0, k0 : k0 + case.lens_kv[i]])):
                assert _finite(got) and not got.float().any(), f"seq {i}: no row has a key, {name} must be exactly zero"
    _check_mx(case, dq, dk, dv)


# --------------------------------------------------------------------------- the direct adapter surface


def _envelope(n, s, nh, dt, dev="cuda", pad_heads=0):
    """A logical-BHSD view over BSHD-physical storage of the ENVELOPE (B, H, S_max, D) the adapter's SAMPLES declare; ``pad_heads`` widens
    the token stride to ``(nh + pad_heads) x D`` (the port = the first ``nh`` heads of a wider slab, head stride D)."""
    return torch.empty(1, n, s, nh + pad_heads, _D, device=dev, dtype=dt)[0, :, :, :nh].permute(0, 2, 1, 3)


def _build_direct_api(case, *, token_major_stats=False, envelope_q=None, envelope_kv=None, external_delta=False):
    """``SdpaBwdDslSm107Mxfp8(thd=True, ...)`` over envelope SAMPLES and the case's PACKED scale-factor tensors (the packed tile count is
    read off their byte sizes).  The packed buffers are allocated at ``cap_q`` / ``cap_kv`` tokens; the envelope must cover them
    (``B * S_max >= cap``) or the adapter tightens the plan's capacity below the buffers and the standalone surface refuses the
    runtime geometry.  The samples carry the case's token stride (``token_pad_heads``): the plan's geometry is the samples'."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Mxfp8

    b, h, hkv = case.b, case.h, case.hkv
    env_q = max(envelope_q or max(max(case.lens_q), 1), -(-case.cap_q // b))
    env_kv = max(envelope_kv or max(max(case.lens_kv), 1), -(-case.cap_kv // b))
    pad = int(getattr(case, "token_pad_heads", 0))

    def env(n, s, nh, dt):
        return _envelope(n, s, nh, dt, pad_heads=pad)

    api = SdpaBwdDslSm107Mxfp8(
        env(b, env_q, h, _T_E4M3),
        env(b, env_kv, hkv, _T_E4M3),
        env(b, env_kv, hkv, _T_E4M3),
        env(b, env_q, h, _BF16),  # the o_f16 port
        env(b, env_q, h, _T_E4M3),
        torch.empty(b, h, env_q, 1, device="cuda", dtype=torch.float32),
        env(b, env_q, h, _BF16),
        env(b, env_kv, hkv, _BF16),
        env(b, env_kv, hkv, _BF16),
        sample_q_T=env(b, env_q, h, _T_E4M3),
        sample_k_T=env(b, env_kv, hkv, _T_E4M3),
        sample_do_T=env(b, env_q, h, _T_E4M3),
        sample_do_f16=env(b, env_q, h, _BF16),
        **{f"sample_{name}": case.sf[name] for name in _SF_ALL},
        scale_softmax=case.scale,
        is_causal=case.causal,
        causal_bottom_right=case.bottom_right,
        window_size_left=None if case.window_left is None else case.window_left - 1,
        thd=True,
        max_total_seq_len_q=case.cap_q,
        max_total_seq_len_kv=case.cap_kv,
        thd_stats_token_major=token_major_stats,
        # head-major Stats allocated at the capacity names its own head stride (the graph path derives it from the port's stride);
        # token-major (T, H) Stats is compact and takes no stride (``_run_mx_direct`` slices it to the plan's cap)
        thd_stats_head_stride=None if token_major_stats else case.cap_q,
        external_delta=external_delta,
    )
    assert api.check_support()
    return api


def _run_mx_direct(
    lens_q,
    lens_kv,
    *,
    h=2,
    hkv=None,
    token_major_stats=False,
    causal=False,
    bottom_right=False,
    window_left=None,
    poison=False,
    pad_cap=0,
    sf_tail_tiles=0,
    poison_sf_pads=(),
    poison_sf_seqs=None,
    poison_outputs=False,
    runs=1,
    ws_fill=0xFF,
    check=None,
    seed=7,
    envelope_q=None,
    token_pad_heads=0,
    sf_dense_capacity=False,
    external_delta=False,
    delta=None,
):
    """Build the case, drive ``SdpaBwdDslSm107Mxfp8(thd=True)`` directly on PACKED views over a 0xFF-poisoned workspace (NaN in every
    dtype the chain stores, an E8M0 NaN in every atom: a stage reading a scratch region before writing it surfaces as NaN), the
    gradient tails past the packed totals under the finite sentinel; compare per sequence under the MXFP8 recipe.  The dS rounding
    the oracle composes follows the ``ds_policy`` fixture (``api_dsl_sm107.MXFP8_DS_SF_POLICY`` at construction).  Returns the run
    (case, gradients, every run's outputs, the api and its arguments) for the caller's extra assertions.  ``token_pad_heads`` binds
    every packed operand -- payloads, ports and gradients -- at a PADDED token stride ``(H + pad) x D`` (the pad heads of the gradient
    slabs hold the sentinel; ``grad_storage`` returns the slabs); ``sf_dense_capacity`` sizes the scale-factor buffers at the DENSE
    capacity ``B x ceil(S_max / 128)`` tiles per side (the envelope the adapter declares), above the packed bound for ragged lengths."""
    dev = "cuda"
    cap_q, cap_kv = sum(lens_q) + pad_cap, sum(lens_kv) + pad_cap
    if sf_dense_capacity:
        # the envelope ``_build_direct_api`` declares, per side; the plan's capacity grows to the declared sample's own tile count
        b_ = len(lens_q)
        env_q_, env_kv_ = max(envelope_q or max(max(lens_q), 1), -(-cap_q // b_)), max(max(max(lens_kv), 1), -(-cap_kv // b_))
        sf_tail_tiles = (b_ * _tiles(env_q_) - sum(_tiles(n) for n in lens_q), b_ * _tiles(env_kv_) - sum(_tiles(n) for n in lens_kv))
        assert min(sf_tail_tiles) >= 0, sf_tail_tiles
    case = _thd_mx_case(
        lens_q, lens_kv, h, hkv, cap_q=cap_q, cap_kv=cap_kv, poison=poison, sf_tail_tiles=sf_tail_tiles, token_pad_heads=token_pad_heads,
        poison_sf_pads=poison_sf_pads, poison_sf_seqs=poison_sf_seqs, seed=seed, causal=causal, bottom_right=bottom_right, window_left=window_left,
        quantize_ds=_block_scaled(),
    )  # fmt: skip
    api = _build_direct_api(case, token_major_stats=token_major_stats, envelope_q=envelope_q, external_delta=external_delta)
    assert delta is None or external_delta, "a delta binds on an external_delta plan only"
    view = lambda t: t.permute(0, 2, 1, 3)  # noqa: E731  [1,T,H,D] -> logical [1,H,T,D], the dense path's orientation
    fill = float("nan") if poison_outputs else 0.0

    def grad(cap, nh):  # [1, cap, nh, D]: the first nh heads of a [1, cap, nh + pad, D] slab under ``token_pad_heads``, pad heads = sentinel
        stor = torch.empty(1, cap, nh + token_pad_heads, _D, device=dev, dtype=_BF16)
        stor[:, :, nh:] = _TAIL_SENTINEL
        return stor, stor[:, :, :nh]

    (dq_stor, dq), (dk_stor, dk), (dv_stor, dv) = grad(case.cap_q, case.h), grad(case.cap_kv, case.hkv), grad(case.cap_kv, case.hkv)
    # TRANSPOSED, not reshaped: lse is head-major [1, H, T]; the compact token-major form holds exactly the plan's packed capacity
    stats = case.lse[0].transpose(0, 1).contiguous()[: api._t_q_cap] if token_major_stats else case.lse
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), dtype=torch.uint8, device=dev)
    lq = torch.tensor(case.lens_q, dtype=torch.int32, device=dev)
    lk = torch.tensor(case.lens_kv, dtype=torch.int32, device=dev)
    tensors = (view(case.q), view(case.k), view(case.v), view(case.o), view(case.do), stats, view(dq), view(dk), view(dv))
    kwargs = dict(
        workspace=ws, seq_q_lens=lq, seq_kv_lens=lk,
        q_T_tensor=view(case.q_T), k_T_tensor=view(case.k_T), do_T_tensor=view(case.do_T), do_f16_tensor=view(case.do_f16),
        **{name: case.sf[name] for name in _SF_ALL}, delta_tensor=delta,
    )  # fmt: skip
    outs = []
    for _ in range(runs):
        for x in (dq, dk, dv):
            x.fill_(fill)
        _sentinel_tails(case, dq, dk, dv)
        ws.fill_(ws_fill)
        api.execute(*tensors, **kwargs)
        torch.cuda.synchronize()
        outs.append(tuple(x.clone() for x in (dq, dk, dv)))
    _assert_tails_untouched(case, dq, dk, dv)
    if check is not False:
        for name, x, live in (("dQ", dq, case.t_q), ("dK", dk, case.t_kv), ("dV", dv, case.t_kv)):
            assert _finite(x[0, :live]), f"{name} has non-finite values in the packed region ({int(torch.isnan(x[0, :live].float()).sum())} NaN cells)"
        (check or _check_mx)(case, dq, dk, dv)
    return SimpleNamespace(
        case=case, dq=dq, dk=dk, dv=dv, outs=outs, api=api, tensors=tensors, kwargs=kwargs, ws=ws, stats=stats, grad_storage=(dq_stor, dk_stor, dv_stor)
    )


@requires_rubin
@pytest.mark.parametrize(
    "lens_q, lens_kv, causal", (((300, 128, 200), (300, 128, 200), True), ((256, 100), (180, 300), False)), ids=("self-causal", "cross-Tq356-Tkv480-dense")
)
def test_thd_mxfp8_external_delta_is_bitwise_the_rows_own_pre_pass(ds_policy, lens_q, lens_kv, causal):
    """A THD plan built with ``external_delta=True`` and fed the delta the row's OWN pre-pass wrote (the fp32 dot of the packed bf16
    ``o_f16`` / ``dO_f16`` ports, read back out of the sibling plan's ``R_DELTA`` region: the PACKED head-major ``[1, H_q, ceil128(T_q)]``,
    zeros past ``T_q``) returns dQ / dK / dV ``torch.equal`` the sibling's under BOTH dS policies -- the same artifact minus the ``dot``
    launch -- over three ragged sequences, GQA and a causal band.  Also pinned: the slot (22) binds on the external plan only, the carve
    lost exactly the delta region, one launch fewer (CUPTI, when available), the compiled plan's refusals fire with no launch.  The
    CROSS-attention cell has Q and KV token capacities that round to DIFFERENT tiles (356 -> 384 vs 480 -> 512): the slot's shape and the
    carve follow the Q capacity."""
    from dataclasses import replace

    from cudnn.sdpa.bwd import prepared_sm107
    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import R_DELTA
    from cudnn.sdpa.fwd.api_dsl import ws_align

    t_pad, kv_pad = (-(-sum(x) // 128) * 128 for x in (lens_q, lens_kv))
    assert (t_pad != kv_pad) is (lens_q != lens_kv), "the cross cell pins the Q capacity against a DIFFERENT KV one"
    own = _run_mx_direct(lens_q, lens_kv, h=4, hkv=2, causal=causal)
    offset, shape, _strides = prepared_sm107._regions(own.api, prepared_sm107._REGION_SLOTS_MXFP8)[0][R_DELTA]
    assert shape == own.api.external_delta_shape == (1, 4, t_pad), "the slot follows the Q token capacity"
    delta = own.ws[offset : offset + 4 * math.prod(shape)].view(torch.float32).view(*shape).clone()
    assert torch.isfinite(delta).all() and torch.equal(delta[:, :, own.case.t_q :], torch.zeros_like(delta[:, :, own.case.t_q :]))
    ext = _run_mx_direct(lens_q, lens_kv, h=4, hkv=2, causal=causal, external_delta=True, delta=delta)
    assert own.api._prepared.roles[22] == prepared_sm107.EXTERNAL_DELTA_ROLE and own.api._prepared.operands[22] is None
    assert ext.api._prepared.operands[22] is not None, "the delta slot binds on the external plan only"
    assert "delta" not in [n for n, _n, _d in ext.api._scratch_plan()] and "delta" in [n for n, _n, _d in own.api._scratch_plan()]
    assert own.api.scratch_workspace_bytes() - ext.api.scratch_workspace_bytes() == ws_align(4 * t_pad * 4), "the carve lost exactly the delta region"
    for name, x, y in zip(("dQ", "dK", "dV"), ext.outs[0], own.outs[0]):
        _bitwise(f"{name}: the external-delta plan vs the row's own pre-pass", x, y)
    counts = cuda_launch_counts(lambda: own.api.execute(*own.tensors, **own.kwargs), lambda: ext.api.execute(*ext.tensors, **ext.kwargs))
    if counts is None:
        print("\nlaunch count unverified here (no CUDA profiler activity: CUPTI unavailable)")
    else:
        assert counts[1] == counts[0] - 1, counts
        print(f"\nlaunches: own {counts[0]}, external delta {counts[1]}")
    launches = []
    ext.api._prepared = replace(ext.api._prepared, fn=lambda *args: launches.append(args))
    with pytest.raises(ValueError, match=r"CONTIGUOUS \[1, H_q, ceil128\(T_q\)\]"):
        ext.api.execute(
            *ext.tensors, **dict(ext.kwargs, delta_tensor=torch.zeros(len(lens_q), 4, -(-max(lens_q) // 128) * 128, device="cuda"))
        )  # the DENSE envelope shape
    with pytest.raises(ValueError, match="delta_tensor is required"):
        ext.api.execute(*ext.tensors, **dict(ext.kwargs, delta_tensor=None))
    assert not launches


@requires_rubin
def test_thd_mxfp8_self_attention(ds_policy):
    """Three sequences of unequal length, none a tile multiple -- on both dS policies of the row (P-c first, then the shipped P-b)."""
    _run_mx_direct((300, 128, 200), (300, 128, 200))


@requires_rubin
def test_thd_mxfp8_cross_attention(ds_policy):
    """Unequal Q and KV lengths, and unequal packed totals (and SF tile totals) with them."""
    _run_mx_direct((256, 100), (180, 300))


@requires_rubin
def test_thd_mxfp8_single_sequence_matches_dense_shape(ds_policy):
    """B == 1 is the degenerate packing: it must agree with the dense answer."""
    _run_mx_direct((512,), (512,))


@requires_rubin
def test_thd_mxfp8_stats_token_major():
    """The other packed Stats layout the forward can emit."""
    _run_mx_direct((300, 128), (300, 128), token_major_stats=True)


@requires_rubin
def test_thd_mxfp8_dead_units_run_one_masked_tile():
    """FEWER live units than clusters (one 256-row kv block of one head on an occupancy-sized grid): every other cluster's first unit
    is past the device live total and runs ONE forced fully-masked tile.  Poisoned outputs and workspace: a dead unit that stored
    anything into a live row, or a live unit that skipped a store, surfaces as NaN or a wrong sequence."""
    _run_mx_direct((256,), (256,), h=1, poison_outputs=True)


@requires_rubin
def test_thd_mxfp8_zero_length_sequence():
    """A sequence with no tokens on either side -- and no SF tile -- must not corrupt its neighbours (whose SF tiles follow it in the
    packed order); its own gradients are exact zeros."""
    run = _run_mx_direct((256, 0, 128), (256, 0, 128), poison_outputs=True)
    _assert_empty_sequence_exactly_zero(run.case, run.dq, run.dk, run.dv, 1)


@requires_rubin
@pytest.mark.parametrize("lens_q,lens_kv", (((0, 128, 256), (192, 128, 256)), ((192, 128, 256), (0, 128, 256))), ids=("empty_q_side", "empty_kv_side"))
def test_thd_mxfp8_one_sided_empty_sequence(lens_q, lens_kv):
    """A sequence empty on ONE side only.  Empty Q: its kv blocks still get units (one forced fully-masked q tile each -> dS = 0 into
    their own rows, dV = 0 stored to the sequence's rows), the dK GEMM's reduction is empty and must store zeros by SELECT (not
    residue), dQ has no rows; its Q-side SF tile prefix does not advance.  Empty KV: no unit at all, the dQ GEMM's reduction is empty
    -> zeros by select, dK / dV have no rows, its kv-side SF prefix does not advance.  Zero is the answer, not a convention; the
    poisoned outputs make an unwritten live row NaN."""
    run = _run_mx_direct(lens_q, lens_kv, poison_outputs=True)
    _assert_empty_sequence_exactly_zero(run.case, run.dq, run.dk, run.dv, 0)


@requires_rubin
def test_thd_mxfp8_nan_capacity_tail():
    """Declared totals larger than the live packing, with a NaN tail in every e4m3 payload, O, dO_f16 and Stats AND two 0xFF scale-factor
    tiles past ``cu_sf[B]`` per side: the kernels' device-side descriptor clamps keep the rows between ``cu_*[B]`` and the capacity out of
    every MMA, and the FIVE clamped SF maps keep the capacity tiles out of every UTCCP."""
    _run_mx_direct((256, 128), (256, 128), poison=True, pad_cap=384, sf_tail_tiles=2)


@requires_rubin
def test_thd_mxfp8_nan_capacity_tail_unaligned_last_sequence():
    """The capacity tail reached by a K-TILE OVERSHOOT (a 100-token last sequence: the block-scale arm's 128-element e4m3 K tile and
    its whole-atom SFB read reach 28 rows past the packed total) on both GEMM orientations, and by the main kernel's last kv block /
    q tile and its SF tile."""
    _run_mx_direct((256, 100), (256, 100), poison=True, pad_cap=384, sf_tail_tiles=1)


@requires_rubin
@pytest.mark.parametrize("hkv", (2, 1), ids=("gqa_group2", "mqa"))
def test_thd_mxfp8_gqa(hkv):
    """Packed GQA / MQA: bf16 dK / dV partials ONE PER Q HEAD over the packed kv axis, folded onto the KV heads by the bounded fold
    (``cu_k[B]`` on device); the reference SUMS a group's contributions."""
    _run_mx_direct((300, 128, 200), (300, 128, 200), h=4, hkv=hkv)


@requires_rubin
def test_thd_mxfp8_gqa_causal(ds_policy):
    _run_mx_direct((256, 100), (256, 100), h=4, hkv=2, causal=True)


@requires_rubin
def test_thd_mxfp8_gqa_cross_attention_and_zero_length():
    """GQA over unequal Q / KV totals with an empty sequence in the middle: the dK / dV partials are sized on the packed KV capacity
    while dQ rides the Q one."""
    run = _run_mx_direct((256, 0, 100), (180, 0, 300), h=4, hkv=2, poison_outputs=True)
    _assert_empty_sequence_exactly_zero(run.case, run.dq, run.dk, run.dv, 1)


@requires_rubin
def test_thd_mxfp8_gqa_capacity_tail_untouched():
    """GQA with declared totals past the live packing and a NaN tail on BOTH sides.  The per-Q-head partials past ``cu_k[B]`` are never
    written, so the fold must stop at the live kv total ON DEVICE: an unbounded fold reads the 0xFF-poisoned workspace there and writes
    NaN into the caller's dK / dV capacity tail -- which only the FINITE tail sentinel can see."""
    _run_mx_direct((300, 128, 200), (300, 128, 200), h=4, hkv=2, poison=True, pad_cap=384, sf_tail_tiles=2, poison_outputs=True)


@requires_rubin
def test_thd_mxfp8_every_sequence_empty_q_with_nan_in_dO_row_0():
    """No query anywhere (``cu_q[B] = 0``, ``cu_sf_q[B] = 0``), keys in every sequence, the Q / dO capacity all NaN and the one bound
    Q-side SF tile all 0xFF (a zero-capacity SF buffer cannot bind; the forward's convention).  The clamped Q / dO descriptors keep an
    extent of 1 (0 is invalid), so a forced tile that addressed ``cu_q[b] = 0`` would load the NaN row and the NaN scale: the kernel
    routes every load of a unit without query rows past the clamped extent instead (zero-filled) -- dK and dV exact zeros, nothing
    past the packed totals written."""
    run = _run_mx_direct((0, 0), (128, 256), poison=True, pad_cap=128, poison_outputs=True, envelope_q=128)
    for i in range(run.case.b):
        _assert_empty_sequence_exactly_zero(run.case, run.dq, run.dk, run.dv, i)


@requires_rubin
@pytest.mark.parametrize("lens", [(200, 512, 700, 1024, 2048)], ids=["kv-blocks-1-2-3-4-8"])
def test_thd_mxfp8_kv_blocks_per_sequence(lens):
    """kv blocks per sequence 1, 2, 3, 4 and 8 in ONE packed batch (sdpa-invariants s9: the parity reuse and the ring wrap of the
    per-sequence block walk), q tiles per block from 1 to 4 across them, SF tile prefixes 2 / 4 / 6 / 8 / 16 deep."""
    _run_mx_direct(lens, lens)


@requires_rubin
def test_thd_mxfp8_q_tiles_per_kv_block():
    """q tiles per kv block = 1, 2, 3 and 4 in one packed batch against one kv block each: the Q / dO rings and the P ring wrap at
    different depths per sequence, the SF rings with them."""
    _run_mx_direct((128, 256, 384, 512), (256, 256, 256, 256), h=1)


# --- causal family: the per-sequence diagonal is the whole risk, and stage 3 trims its K range per sequence over a workspace whose
# masked tiles the kernel never wrote (0xFF-poisoned here, never zero-filled: a read past the band lands NaN in dQ / dK).


@requires_rubin
def test_thd_mxfp8_causal(ds_policy):
    _run_mx_direct((300, 128, 200), (300, 128, 200), causal=True)


@requires_rubin
def test_thd_mxfp8_causal_zero_length_sequence():
    """Causal plus an empty sequence: the masked tiles the kernel never wrote and the zero-length sequence's missing unit / extent-1
    descriptor / missing SF tile are independent mechanisms, all live at once."""
    run = _run_mx_direct((256, 0, 128), (256, 0, 128), causal=True, poison_outputs=True)
    _assert_empty_sequence_exactly_zero(run.case, run.dq, run.dk, run.dv, 1)


@requires_rubin
def test_thd_mxfp8_causal_cross_attention():
    """Top-left aligned, so the diagonal does not move -- but S_kv[b] != S_q[b] makes the live kv blocks per q row differ per sequence,
    which a trim keyed on absolute workspace rows would get wrong."""
    _run_mx_direct((256, 100), (180, 300), causal=True)


@requires_rubin
def test_thd_mxfp8_causal_bottom_right():
    """Bottom-right alignment: the diagonal offset IS per sequence (56 and 200 here), read from the metadata per sequence by the kernel
    and by stage 3's trim."""
    _run_mx_direct((200, 100), (256, 300), causal=True, bottom_right=True)


@requires_rubin
def test_thd_mxfp8_causal_bottom_right_ragged_s_q(ds_policy):
    """Bottom-right at a RAGGED S_q per sequence (200 and 300: neither a q-tile multiple): the per-sequence ``s_q[b]`` feeds the body's
    diagonal and its q-tile trim; the last q tile's pad columns are select-masked by the THD q band (NOT the dense ``MASK_Q_PAD`` arm,
    which a THD plan never sets)."""
    _run_mx_direct((200, 300), (456, 300), causal=True, bottom_right=True)


@requires_rubin
def test_thd_mxfp8_causal_swa():
    """Sliding window: a LEFT bound on top of the causal right one -- the band's second per-sequence edge, through the kernel's q-tile
    trim from above and stage 3's per-sequence window edge, so the window-skipped tiles are never read."""
    _run_mx_direct((300, 128, 200), (300, 128, 200), causal=True, window_left=64)


@requires_rubin
def test_thd_mxfp8_causal_nan_capacity_tail():
    """Causal with a NaN tail past the declared totals and 0xFF SF tiles past ``cu_sf[B]``: a different set of workspace rows than the
    dense tail case, read through the per-sequence trim with no zero-fill in between."""
    _run_mx_direct((256, 128), (256, 128), poison=True, pad_cap=384, sf_tail_tiles=2, causal=True)


@requires_rubin
@pytest.mark.parametrize(
    "lens_q,lens_kv,kw",
    [
        ((0, 128, 256), (192, 128, 256), dict(causal=True, bottom_right=True)),
        ((192, 128, 256), (0, 128, 256), dict(causal=True, window_left=64)),
        ((256, 0, 128), (256, 0, 128), dict(causal=True, bottom_right=True, window_left=64)),
        ((300, 5, 200), (300, 5, 200), dict(causal=True)),
        ((300, 65, 200), (300, 129, 200), dict(causal=True, window_left=2)),  # s_q <= s_kv: every row keeps a key
    ],
    ids=["br-empty-q-side", "swa-empty-kv-side", "br-swa-zero-length", "five-row-sequence", "window-of-two-keys-odd-tiles"],
)
def test_thd_mxfp8_trimmed_degenerate_sequences(lens_q, lens_kv, kw):
    """The degenerate-input matrix under the trimmed stage 3 (sdpa-invariants): an empty q side (dK / dV rows exist, the reduction is
    empty -> exact zeros through the select), an empty kv side (no unit, no rows), a zero-length sequence inside the batch, a FIVE-row
    sequence (one q tile, one kv block, one SF tile each side -- never a one-row one), the narrowest served window (two keys) on odd
    q-tile counts -- all over a poisoned workspace and poisoned outputs, every live gradient held to the oracle."""
    run = _run_mx_direct(lens_q, lens_kv, poison_outputs=True, **kw)
    for i in range(run.case.b):
        if run.case.lens_q[i] == 0 or run.case.lens_kv[i] == 0:
            _assert_empty_sequence_exactly_zero(run.case, run.dq, run.dk, run.dv, i)


@requires_rubin
@pytest.mark.parametrize("case", list(_NO_KEY_ROW_CASES), ids=list(_NO_KEY_ROW_CASES))
def test_thd_mxfp8_rows_without_a_key_are_exactly_zero(case, monkeypatch, ds_policy):
    """Per-sequence geometries with FULLY MASKED q rows -- the trailing rows of a top-left window past ``s_kv + W`` and the leading rows
    of a bottom-right sequence with ``s_q > s_kv``.  Their dQ is exactly zero: the trim hands those M tiles an EMPTY K range (the
    select-zero store) instead of clamping onto an unwritten tile of the poisoned workspace; the rows with keys match the oracle.  The
    untrimmed, zero-filled twin (``STAGE3_CAUSAL_TRIM = False``) must agree bitwise -- on both dS policies (under P-b the fill zeroes
    the second payload and the two atom tensors with the first: an E8M0 byte 0 scales a zero payload to exactly zero)."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    kw = {k: v for k, v in _NO_KEY_ROW_CASES[case].items() if k not in ("lens_q", "lens_kv")}
    lens_q, lens_kv = _NO_KEY_ROW_CASES[case]["lens_q"], _NO_KEY_ROW_CASES[case]["lens_kv"]
    run_kw = dict(_graph_kw_to_direct(kw), poison_outputs=True, check=_check_mx_rows_with_keys)
    trimmed = _run_mx_direct(lens_q, lens_kv, **run_kw)
    monkeypatch.setattr(sm107, "STAGE3_CAUSAL_TRIM", False)
    untrimmed = _run_mx_direct(lens_q, lens_kv, **run_kw)
    for name, a, b in zip(("dQ", "dK", "dV"), (trimmed.dq, trimmed.dk, trimmed.dv), (untrimmed.dq, untrimmed.dk, untrimmed.dv)):
        _bitwise(f"{name} (trimmed vs untrimmed, rows without a key)", a, b)


@requires_rubin
@pytest.mark.parametrize("case", list(_TRIM_TWIN_CASES), ids=list(_TRIM_TWIN_CASES))
def test_thd_mxfp8_trimmed_stage3_is_bitwise_the_untrimmed_rendering_over_a_poisoned_workspace(case, monkeypatch, ds_policy):
    """The per-sequence K-trim reads ONLY dS tiles the main kernel wrote: the workspace is 0xFF-poisoned (NaN in bf16 and e4m3, an E8M0
    NaN in every atom) and the shipped chain runs WITHOUT the zero-fill, so a GEMM reaching a skipped tile lands NaN in dQ / dK.  The
    untrimmed twin (``STAGE3_CAUSAL_TRIM = False``: ``CAUSAL_K_NONE`` on both GEMMs, the fill back on) walks every k tile of the
    zero-filled workspace -- the skipped tiles are exact zeros (payloads AND atoms), so the two accumulate the same values in the same
    order: the gradients must be the SAME BITS.  Every mask arm the row serves, on tails that are no tile multiple, with an empty side
    on either axis, under both dS policies."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    kw = dict(_TRIM_TWIN_CASES[case])
    lens_q, lens_kv = kw.pop("lens_q"), kw.pop("lens_kv")
    assert sm107.STAGE3_CAUSAL_TRIM, "the per-sequence trim is what ships; the pin flips it OFF for the twin"
    trimmed = _run_mx_direct(lens_q, lens_kv, **_graph_kw_to_direct(kw))
    monkeypatch.setattr(sm107, "STAGE3_CAUSAL_TRIM", False)
    untrimmed = _run_mx_direct(lens_q, lens_kv, **_graph_kw_to_direct(kw))
    for name, a, b in zip(("dQ", "dK", "dV"), (trimmed.dq, trimmed.dk, trimmed.dv), (untrimmed.dq, untrimmed.dk, untrimmed.dv)):
        _bitwise(f"{name} (trimmed stage 3, no fill vs untrimmed, zero-filled)", a, b)


@requires_rubin
@pytest.mark.parametrize("case", list(_GQA_TWIN_CASES), ids=list(_GQA_TWIN_CASES))
def test_thd_mxfp8_single_launch_dq_is_bitwise_the_per_member_launches(case, monkeypatch, ds_policy):
    """Under GQA the THD dQ GEMM is ONE launch per head chunk on BOTH dS policies -- the dQ record takes ``b_head_group = group``:
    P-c through the plain THD rendering, P-b through the block-scale arm's THD leg, whose packed B = k_T AND its packed scale-factor
    planes are indexed by ``h // group`` (the per-sequence SF tile prefix is a token-side term, the B descriptor's clamp touches only
    the token extent) -- where it used to be one per group MEMBER: dQ the SAME BITS (and dK / dV, untouched).  ``DQ_SINGLE_LAUNCH =
    False`` is the per-member twin; both runs are held to the per-sequence fp64 oracle over a 0xFF-poisoned workspace.  Covers GQA
    32/2 and 64/8, tails that are no multiple of 256, dense / causal / bottom-right / a window, and a sequence empty on either side."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    kw = dict(_GQA_TWIN_CASES[case])
    lens_q, lens_kv = kw.pop("lens_q"), kw.pop("lens_kv")
    group = kw["h"] // kw["hkv"]
    assert sm107.DQ_SINGLE_LAUNCH, "one dQ launch per chunk is what ships; the pin flips it OFF for the twin"
    single = _run_mx_direct(lens_q, lens_kv, **_graph_kw_to_direct(kw))
    monkeypatch.setattr(sm107, "DQ_SINGLE_LAUNCH", False)
    members = _run_mx_direct(lens_q, lens_kv, **_graph_kw_to_direct(kw))
    # the dQ record's b_head_group, copied off it at compile: the group on the shipped arm, 1 on the twin -- on BOTH policies
    assert (single.api._ds_block_scaled, members.api._ds_block_scaled) == (_block_scaled(), _block_scaled())
    assert (int(single.api._dq_b_head_group), int(members.api._dq_b_head_group)) == (group, 1), (single.api._dq_b_head_group, members.api._dq_b_head_group)
    for name, a, b in zip(("dQ", "dK", "dV"), (single.dq, single.dk, single.dv), (members.dq, members.dk, members.dv)):
        _bitwise(f"{name} (single dQ launch vs per-member launches, {'P-b' if _block_scaled() else 'P-c'})", a, b)


@requires_rubin
def test_thd_mxfp8_two_launches_are_bitwise_and_race_free(ds_policy):
    """Launch 2 vs 1 = the two-launch race trick, launch 3 vs 2 = the determinism to show before claiming it."""
    run = _run_mx_direct((300, 128, 200), (300, 128, 200), h=4, hkv=2, causal=True, runs=3, poison_outputs=True)
    for which, i, j in (("launch 2 vs 1 (race)", 1, 0), ("launch 3 vs 2 (determinism)", 2, 1)):
        for name, a, b in zip(("dQ", "dK", "dV"), run.outs[i], run.outs[j]):
            _bitwise(f"{name} {which}", a, b)


@requires_rubin
@pytest.mark.parametrize("geometry", ["mha", "gqa_causal"])
def test_thd_mxfp8_padded_token_stride(geometry, ds_policy):
    """Every packed operand -- the e4m3 payloads q / k / v / dO, the transposed-quantization q_T / k_T / dO_T, the bf16 o_f16 / dO_f16
    ports and the three gradients -- bound at a PADDED token stride, ``(H + 1) x D`` elements (the port is the first H heads of a wider
    slab: what the adapter admits as packed BSHD rows, and the stride the plan's geometry carries), on both dS policies.  Under P-c this
    is the cell the per-token dequant of q_T / k_T owed: it walked the packed slab compactly and read the wrong tokens into dK / dQ
    (finite and wrong, dV exact) at any token stride above H x D; the TMA descriptors, the dot pre-pass and the GQA fold already took
    the strides.  The pad heads of the gradient slabs hold the sentinel and stay untouched; two launches bitwise."""
    kw = dict(h=2) if geometry == "mha" else dict(h=4, hkv=2, causal=True)
    run = _run_mx_direct((300, 128, 200), (300, 128, 200), token_pad_heads=1, runs=2, **kw)
    c, ops = run.case, run.api._prepared.operands
    for i, nh in ((0, c.h), (1, c.hkv), (11, c.h), (12, c.hkv)):  # q, k, q_T, k_T: the plan's packed geometry carries the padded stride
        assert ops[i].strides[1] == (nh + 1) * _D and ops[i].strides[2] == _D, (i, ops[i].strides)
    for name, stor, nh in zip(("dQ", "dK", "dV"), run.grad_storage, (c.h, c.hkv, c.hkv)):
        written = int((stor[:, :, nh:] != _TAIL_SENTINEL).sum())
        assert written == 0, f"{name}: {written} elements of the pad heads were written"
    for name, a, b in zip(("dQ", "dK", "dV"), run.outs[0], run.outs[1]):
        _bitwise(f"{name} differs between two launches at a padded token stride", a, b)


@requires_rubin
def test_thd_mxfp8_sf_declared_at_the_dense_capacity(ds_policy):
    """The seven scale-factor buffers sized at the DENSE capacity ``B x ceil(S_max / 128)`` tiles per head -- the layout a ragged graph
    declares and the forward emits -- bind, stage and compute exactly on both dS policies, although it is MORE tiles than the packed
    bound ``ceil(T_cap / 128) + B`` for these ragged lengths (9 vs 8 per side): the plan's capacity is the larger of the two, the
    staging copies and the operand span follow it, the clamped maps never read the tiles past the live total (0xFF here).  Before
    this the binder refused such a buffer after eligibility had passed."""
    lens = (300, 128, 200)  # B = 3, S_max = 300: 3 x 3 = 9 declared tiles per side vs ceil(628 / 128) + 3 = 8
    run = _run_mx_direct(lens, lens, h=2, sf_dense_capacity=True, poison=True)
    c, api = run.case, run.api
    declared = c.b * _tiles(max(lens))
    assert declared > _tiles(c.cap_kv) + c.b, "the cell must declare MORE tiles than the packed bound, or it proves nothing"
    for name in _SF_ALL:
        assert _sf_tile_count(c.sf[name]) == declared == api._thd_sf_tiles_cap(name), name


@requires_rubin
def test_thd_mxfp8_dv_is_bitwise_across_the_ds_policies(monkeypatch):
    """The two chains side by side on one packed input: dV does not depend on dS, so it is BITWISE across the policy (a difference is a
    main-kernel change, not a stage-3 change); dQ / dK each pass the recipe against their OWN oracle inside ``_run_mx_direct``."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107, config_sm107 as cfg

    runs = {}
    for policy in (cfg.DS_SF_P_C, cfg.DS_SF_P_B):
        with monkeypatch.context() as patch:
            patch.setattr(sm107, "MXFP8_DS_SF_POLICY", policy)
            runs[policy] = _run_mx_direct((300, 128, 200), (300, 128, 200), h=4, hkv=2, causal=True)
    _bitwise("dV across the dS policy", runs[cfg.DS_SF_P_C].dv, runs[cfg.DS_SF_P_B].dv)


@requires_rubin
@pytest.mark.L1
@pytest.mark.parametrize("batch", [33, 129])
def test_thd_mxfp8_batched_descriptors(batch):
    """More sequences than setup warps (the per-sequence dV descriptors, the SF tile prefixes and stage 3's descriptors are shared
    round-robin), empty sequences interleaved, a poisoned capacity tail."""
    lens_q = [[0, 17, 65, 129][i % 4] for i in range(batch)]
    lens_kv = [[33, 0, 127, 257][i % 4] for i in range(batch)]
    _run_mx_direct(lens_q, lens_kv, poison=True, pad_cap=256, sf_tail_tiles=1)


# --------------------------------------------------------------------------- the scale-factor pads: RED then green, per hazard tensor and per sequence


@requires_rubin
@pytest.mark.parametrize("lens_q,lens_kv", [((300, 200), (300, 200)), ((256, 100), (180, 300))], ids=["self", "cross"])
def test_thd_mxfp8_poisoned_sf_pads_of_every_tensor_are_zeroed_by_the_pre_pass(lens_q, lens_kv, ds_policy):
    """GREEN half: the per-sequence pad rows / groups of ALL SEVEN packed scale-factor tensors hold 0xFF (E8M0 NaN) in every sequence
    with a ragged tail, and the outputs are still finite and within the recipe -- the chain's device pre-pass zeroes the pads of the
    five hazard tensors into its staging copies, and the ``sf_q`` / ``sf_k`` pads are harmless by construction (a NaN score on a
    cell the q band / row_dead selects to zero).  The producers' pad bytes are undefined; a green suite over 0x00 pads proves
    nothing (the quantizer writes zeros there), so every cell here poisons them."""
    _run_mx_direct(lens_q, lens_kv, h=4, hkv=2, poison_sf_pads=_SF_ALL)


@requires_rubin
@pytest.mark.parametrize("seq", [0, 1], ids=["seq0", "seq1"])
@pytest.mark.parametrize("tensor", list(_SF_HAZARD))
def test_thd_mxfp8_poisoned_sf_pad_is_red_without_the_pre_pass_per_tensor_and_sequence(tensor, seq, monkeypatch):
    """RED half, per hazard tensor and per sequence: with the pad pre-pass switched OFF (``prepared_host.MXFP8_STAGE_SF_PADS``, a
    plan-time constant folded into the artifact) the poisoned pad bytes of ONE tensor in ONE sequence reach an MMA -- ``sf_v`` / ``sf_do``
    through dP (NaN -> dS -> dQ / dK of that sequence), ``sf_do_T`` through dV = P^T . dO_T (``0 x NaN``), ``sf_q_T`` / ``sf_k_T`` through
    the block-scale GEMMs' whole-atom SFB reads -- and SOME gradient of that sequence comes back non-finite.  A test that cannot fail
    for the right reason proves nothing; this is the failing half, one tensor at a time so a pre-pass that forgot one is named."""
    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host

    monkeypatch.setattr(prepared_host, "MXFP8_STAGE_SF_PADS", False)
    lens = (300, 200)
    run = _run_mx_direct(lens, lens, h=2, poison_sf_pads=(tensor,), poison_sf_seqs=(seq,), check=False)
    c = run.case
    slq = slice(c.cu_q[seq], c.cu_q[seq] + c.lens_q[seq])
    slk = slice(c.cu_k[seq], c.cu_k[seq] + c.lens_kv[seq])
    finite = dict(dQ=_finite(run.dq[0, slq]), dK=_finite(run.dk[0, slk]), dV=_finite(run.dv[0, slk]))
    print(f"\n{tensor} pads of sequence {seq} poisoned, pre-pass OFF: finite = {finite}")
    assert not all(
        finite.values()
    ), f"the RED twin came back finite for {tensor} / sequence {seq} -- the pre-pass switch is not reaching the artifact, or the pad bytes are never read: {finite}"


@requires_rubin
def test_thd_mxfp8_harmless_sf_pads_stay_exact_without_the_pre_pass(monkeypatch, ds_policy):
    """The two tensors the pre-pass does NOT stage (``sf_q``, ``sf_k``): their poisoned pads with the pre-pass switched OFF leave every
    gradient within the recipe -- a NaN score lands only on cells the q band / row_dead select to zero, which is what makes staging
    them unnecessary.  Pinned so the five-tensor list stays a design fact, not a guess."""
    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host

    monkeypatch.setattr(prepared_host, "MXFP8_STAGE_SF_PADS", False)
    _run_mx_direct((300, 200), (300, 200), h=2, poison_sf_pads=_SF_HARMLESS)


# --------------------------------------------------------------------------- the workspace contracts, read back at the plan's own offsets


def _row_off(case):
    """``row_off[b]``: where sequence ``b``'s kv-BLOCKED rows start in the dS workspace -- the setup kernel's prefix of the kv lengths
    rounded up to the 256-row block, in batch order (``tile_dsl.thd.write_thd_row_offsets``)."""
    off, out = 0, []
    for n in case.lens_kv:
        out.append(off)
        off += -(-n // 256) * 256
    return out + [off]


def _region(run, name, dtype):
    off, shape = _workspace_region(run.api, name)
    n = math.prod(shape)
    return run.ws[off : off + n * dtype.itemsize].view(dtype).view(*shape)


@requires_rubin
def test_thd_mxfp8_ds_pad_cells_are_exactly_zero(ds_policy):
    """The kv-blocked dS workspace read back at the plan's region offset after a DENSE (no mask) THD run of two ragged sequences: every
    written PAD cell -- the kv rows of a sequence's block past its own ``s_kv[b]`` (the main kernel's row_dead), the q columns past
    ``s_q[b]`` of the q pairs it wrote (the THD q band) -- is EXACTLY 0 and finite in every written payload (the bf16 workspace under P-c;
    ``ds_dk`` AND ``ds_dq`` under P-b), and no E8M0 NaN byte (255) sits in any atom of a written tile.  Under P-b a 32-block straddles
    its sequence's own pad cells only (never another sequence's: kv blocks are 256-aligned, q tiles sequence-local), and that is
    correct iff those cells are exactly 0 -- a finite non-zero or a NaN pad would move the block's scale and corrupt its LIVE cells.
    The live cells are finite too (and nonzero somewhere); cells outside the written q pairs keep the 0xFF poison."""
    lens_q, lens_kv = (300, 200), (300, 200)
    run = _run_mx_direct(lens_q, lens_kv, h=2, ws_fill=0xFF)
    c, api = run.case, run.api
    block_scaled = api._ds_block_scaled
    ds_dtype = _T_E4M3 if block_scaled else _BF16
    payloads = {"ds_ws": _region(run, "ds_ws", ds_dtype)}
    if block_scaled:
        payloads["ds_dq"] = _region(run, "ds_dq", ds_dtype)
    row_off = _row_off(c)
    for name, ws in payloads.items():
        assert ws.dim() == 4 and ws.shape[0] == 1, (name, ws.shape)
        for b in range(c.b):
            s_q, s_kv = c.lens_q[b], c.lens_kv[b]
            kv_blk, q_pairs = -(-s_kv // 256) * 256, -(-s_q // 256) * 256
            blk = ws[0, :, row_off[b] : row_off[b] + kv_blk, :q_pairs].float()  # [H, kv block rows, written q pairs]
            assert torch.isfinite(blk).all(), f"{name}: sequence {b}'s written block holds non-finite cells ({int(torch.isnan(blk).sum())})"
            live = blk[:, :s_kv, :s_q]
            assert live.abs().sum() > 0, f"{name}: sequence {b}'s live cells are all zero (nothing written?)"
            kv_pad = blk[:, s_kv:, :]
            q_pad = blk[:, :s_kv, s_q:]
            assert not kv_pad.any(), f"{name}: sequence {b}: {int(kv_pad.count_nonzero())} kv PAD cells (rows past s_kv={s_kv}) are not exactly zero"
            assert not q_pad.any(), f"{name}: sequence {b}: {int(q_pad.count_nonzero())} q PAD cells (columns past s_q={s_q}) are not exactly zero"
    if block_scaled:
        for name in ("sf_ds_dk", "sf_ds_dq"):
            atoms = _region(run, name, torch.uint8)  # [1, H, R/128, C/128, 512] (kv tiles, q tiles) or the transpose
            for b in range(c.b):
                s_q, s_kv = c.lens_q[b], c.lens_kv[b]
                kv_t0, kv_t1 = row_off[b] // 128, (row_off[b] + -(-s_kv // 256) * 256) // 128
                q_t1 = (-(-s_q // 256) * 256) // 128
                written = atoms[0, :, kv_t0:kv_t1, :q_t1] if name == "sf_ds_dk" else atoms[0, :, :q_t1, kv_t0:kv_t1]
                assert int(written.max()) <= 254, f"{name}: sequence {b}: an E8M0 NaN byte (255) in a written atom"


def _oracle_ds(case, i):
    """The oracle's fp32 dS of sequence ``i``, ``[1, H, s_q, s_kv]``: ``P (dP - delta) * scale`` from the fp32 P of the forward's Stats
    (the module's exact-LSE pin), over the dequantized operands."""
    log2e = math.log2(math.e)
    qq, kk, vv, dd = (case.quant[i][n] for n in ("q", "k", "v", "do"))
    grp = case.h // case.hkv
    q_deq, k_deq, v_deq, do_deq = qq.deq_d(), kk.deq_d().repeat_interleave(grp, dim=1), vv.deq_d().repeat_interleave(grp, dim=1), dd.deq_d()
    s_raw = torch.einsum("bhqd,bhkd->bhqk", q_deq, k_deq)
    s_q, s_kv = case.lens_q[i], case.lens_kv[i]
    rel = torch.arange(s_kv, device=s_raw.device).view(1, s_kv) - torch.arange(s_q, device=s_raw.device).view(s_q, 1)
    diag = (s_kv - s_q) if (case.causal and case.bottom_right) else 0
    masked = torch.zeros(s_q, s_kv, dtype=torch.bool, device=s_raw.device)
    if case.causal:
        masked |= rel > diag
    if case.window_left is not None:
        masked |= rel <= diag - case.window_left
    p = torch.pow(2.0, s_raw * (case.scale * log2e) - (case.lse_seq[i][None] * log2e).unsqueeze(-1)).masked_fill(masked, 0.0)
    dP = torch.einsum("bhqd,bhkd->bhqk", do_deq, v_deq)
    delta = (case.o_seq[i][None].to(_BF16).float() * case.do_f16_seq[i].float()).sum(-1)
    return p * (dP - delta.unsqueeze(-1)) * case.scale


@requires_rubin
def test_thd_mxfp8_p_b_payloads_and_atoms_dequantize_per_sequence(monkeypatch):
    """The P-b workspace contract under THD, read back at the plan's region offsets over a two-sequence packing: sequence ``b``'s
    ``ds_dk`` rows ``[row_off[b], row_off[b] + s_kv)`` x columns ``[0, s_q)`` (e4m3 scaled per 32-q block, the ``sf_ds_dk`` atoms at kv tile
    ``row_off[b] / 128 + t``) and its ``ds_dq`` (scaled per 32-kv block, ``sf_ds_dq``) -- dequantized, each must equal the oracle's own
    1x32 quantization of that sequence's fp32 dS along the same axis, the dense pin's verdicts: a WHOLE 32-block off is a scale-rule /
    atom-LAYOUT mismatch (an atom indexed from the wrong row offset or tile prefix) and FAILS; an ISOLATED cell is an e4m3 midpoint
    flip, budgeted at 1e-4 of the cells within one e4m3 step at the block's scale.  The kv pad cells of a block are the sequence's own
    zeros, so its 32-blocks quantize exactly as the unpadded oracle's."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107, config_sm107 as cfg

    monkeypatch.setattr(sm107, "MXFP8_DS_SF_POLICY", cfg.DS_SF_P_B)
    lens_q, lens_kv = (300, 200), (300, 200)
    run = _run_mx_direct(lens_q, lens_kv, h=2)
    c, api = run.case, run.api
    assert api._ds_block_scaled
    ds_dk, ds_dq = _region(run, "ds_ws", _T_E4M3).float(), _region(run, "ds_dq", _T_E4M3).float()  # [1, H, R_kv_cap, S_q_pad]
    r_cap, sq_pad = ds_dk.shape[2], ds_dk.shape[3]
    e_dk = _atom_bytes_to_scales(_region(run, "sf_ds_dk", torch.uint8), r_cap, sq_pad // 32)  # [1, H, R_kv_cap, S_q_pad/32]
    e_dq = _atom_bytes_to_scales(_region(run, "sf_ds_dq", torch.uint8), sq_pad, r_cap // 32)  # [1, H, S_q_pad, R_kv_cap/32]
    inv448 = torch.tensor(1.0 / 448.0, dtype=torch.float32, device="cuda")
    row_off = _row_off(c)

    def near_boundary(amax):
        x = amax.float() * inv448
        lg = torch.log2(x.clamp_min(2.0**-126))
        return (lg - lg.round()).abs() <= 2.0**-16

    def check(tag, kern_deq, ref_deq, e_kern, e_ref, amax_blocks, block_axis):
        diff = (kern_deq - ref_deq).abs()
        bad = diff > 0
        n_bad = int(bad.sum())
        e_bad = e_kern != e_ref
        cells_per_block = bad.unflatten(block_axis, (-1, 32)).sum(block_axis + 1)
        byte_off = e_bad & ~near_boundary(amax_blocks)
        blocks_off = int((byte_off | (cells_per_block >= 8)).sum())
        step = (
            (ref_deq.abs() * 2.0**-3).clamp_min(2.0**-9) * torch.pow(2.0, (e_ref.float() - 127.0)).repeat_interleave(32, dim=block_axis).clamp_min(1.0) * 1.001
        )
        flips_too_big = int((bad & (diff > step + 2.0**-9)).sum())
        print(
            f"\n{tag}: {n_bad} of {bad.numel()} dequantized cells differ from the oracle's 1x32 quantization; E8M0 bytes differing {int(e_bad.sum())} of "
            f"{e_bad.numel()} ({int((e_bad & near_boundary(amax_blocks)).sum())} at a power-of-two boundary); whole blocks off {blocks_off}; max |diff| {float(diff.max()):.4g}"
        )
        assert (
            blocks_off == 0
        ), f"{tag}: {blocks_off} whole 32-blocks off -- a scale-rule or atom-layout mismatch (row offset / tile prefix), not a rounding flip"
        assert flips_too_big == 0, f"{tag}: {flips_too_big} differing cells exceed one e4m3 step at the block's scale (not a midpoint flip)"
        assert n_bad <= max(1, int(bad.numel() * 1e-4)), f"{tag}: {n_bad} isolated flips exceed the 1e-4 budget"

    for b in range(c.b):
        s_q, s_kv, r0 = c.lens_q[b], c.lens_kv[b], row_off[b]
        s_q32, s_kv32 = -(-s_q // 32) * 32, -(-s_kv // 32) * 32  # whole 32-blocks: the pad cells inside them are the sequence's own zeros
        ds_ref = _oracle_ds(c, b)  # [1, H, s_q, s_kv]
        ds_ref32 = torch.nn.functional.pad(ds_ref, (0, s_kv32 - s_kv, 0, s_q32 - s_q))
        q_d, sf_d, _sw_d, q_s, sf_s, _sw_s = quantize_to_mxfp8(ds_ref32, 1, c.h, s_q32, s_kv32, block_size=32, fp8_dtype=_T_E4M3)
        ref_dq = (q_d.float() * sf_d).view(1, c.h, s_q32, s_kv32)  # scaled along kv: ds_dq's convention
        ref_dk = (q_s.float() * sf_s).view(1, c.h, s_q32, s_kv32)  # scaled along q: ds_dk's convention
        e_ref_dq = ((sf_d.view(1, c.h, s_q32, s_kv32)[..., ::32].contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)  # [1, H, s_q32, s_kv32/32]
        e_ref_dk = ((sf_s.view(1, c.h, s_q32, s_kv32)[:, :, ::32, :].contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)  # [1, H, s_q32/32, s_kv32]
        # ds_dq: kernel rows kv, columns q -> [1, H, s_q32, s_kv32]; blocks along kv; scales [1, H, S_q_pad, R/32] -> the sequence's window
        e_dq_b = e_dq[:, :, :s_q32, r0 // 32 : (r0 + s_kv32) // 32]
        kern_dq = ds_dq[:, :, r0 : r0 + s_kv32, :s_q32].transpose(-1, -2) * torch.pow(2.0, e_dq_b.float() - 127.0).repeat_interleave(32, dim=-1)
        amax_dq = ds_ref32.unflatten(-1, (-1, 32)).abs().amax(-1)
        check(f"seq {b} ds_dq (per 32-kv block)", kern_dq, ref_dq, e_dq_b, e_ref_dq, amax_dq, block_axis=3)
        # ds_dk: blocks along q; the kernel's scales [1, H, R, S_q_pad/32] -> the sequence's rows, transposed to [1, H, s_q32/32, s_kv32]
        e_dk_b = e_dk[:, :, r0 : r0 + s_kv32, : s_q32 // 32].transpose(-1, -2)
        kern_dk = ds_dk[:, :, r0 : r0 + s_kv32, :s_q32].transpose(-1, -2) * torch.pow(2.0, e_dk_b.float() - 127.0).repeat_interleave(32, dim=2)
        amax_dk = ds_ref32.unflatten(2, (-1, 32)).abs().amax(3)
        check(f"seq {b} ds_dk (per 32-q block)", kern_dk, ref_dk, e_dk_b, e_ref_dk, amax_dk, block_axis=2)


@requires_rubin
def test_thd_mxfp8_launch_census(monkeypatch):
    """The launch census of one THD execute per dS policy (torch.profiler / CUPTI): the bf16-dS twin P-c runs its two SF-aware dequant
    passes (packed q_T, k_T) ahead of the bf16 GEMMs and ONE dQ launch per head chunk under ``DQ_SINGLE_LAUNCH``; the block-scaled
    chain P-b runs NO dequant pass and the SAME single dQ launch per chunk (its dQ record takes ``b_head_group = group``: the packed
    B and its scale factors indexed by ``h // group``); both run the main kernel, the THD setup launches and the scale-factor pad
    pre-pass."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107, config_sm107 as cfg

    counts = {}
    for policy in (cfg.DS_SF_P_C, cfg.DS_SF_P_B):
        monkeypatch.setattr(sm107, "MXFP8_DS_SF_POLICY", policy)
        run = _run_mx_direct((300, 128, 200), (300, 128, 200), h=4, hkv=2, check=False)
        torch.cuda.synchronize()
        # CUDA activity collection unavailable (the profiler cannot start, or captures no CUDA event) -> an EXPLICIT skip, never a
        # missing-kernel finding; a failure raised by the execute propagates, and every count assertion below sits outside any handler
        names = cuda_launch_names(lambda: run.api.execute(*run.tensors, **run.kwargs))
        if names is None:
            pytest.skip("no CUDA activity captured (torch.profiler / CUPTI unavailable on this box): the launch census is unverified here, not failed")
        launches = names[0]
        counts[policy] = dict(
            dequant=sum("dequant_mxfp8_to_bf16" in k for k in launches),
            gemm=sum("bprop_matmul_bh_sm100_kernel" in k for k in launches),
            main=sum("__kernel_TensorMap" in k for k in launches),
            sf_pad=sum("_pad_sf" in k for k in launches),
            setup=sum("thd" in k.lower() and "setup" in k.lower() for k in launches),
        )
        print(f"\npolicy {policy}: {counts[policy]} from {sorted(set(launches))}")
    group = 4 // 2
    assert counts[cfg.DS_SF_P_C]["dequant"] == 2 and counts[cfg.DS_SF_P_B]["dequant"] == 0, counts
    assert counts[cfg.DS_SF_P_B]["gemm"] == 1 + (1 if sm107.DQ_SINGLE_LAUNCH else group), counts  # dK + ONE dQ launch per chunk (the block-scale arm too)
    assert counts[cfg.DS_SF_P_C]["gemm"] == 1 + (1 if sm107.DQ_SINGLE_LAUNCH else group), counts
    assert counts[cfg.DS_SF_P_B]["main"] == counts[cfg.DS_SF_P_C]["main"] >= 1, counts
    for policy in counts:
        assert counts[policy]["sf_pad"] >= 1, f"policy {policy}: no scale-factor pad pre-pass launch on a ragged packing"
        assert counts[policy]["setup"] >= 1, f"policy {policy}: no THD setup launch"


# --------------------------------------------------------------------------- the prepared plan: rebind, replay, length forms, artifact


@requires_rubin
@pytest.mark.parametrize("token_major", [False, True])
def test_prepared_thd_mxfp8_standalone_switches_length_and_prefix_form(token_major, monkeypatch):
    """The standalone surface takes ``(B,)`` lengths or ``(B+1,)`` prefixes per side without recompiling (the form is host metadata,
    derived from numel), and replays under CUDA-graph capture -- the four extra payloads and the seven scale-factor tensors rebound
    by NAME."""
    import cutlass.cute as cute

    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Mxfp8

    original = SdpaBwdDslSm107Mxfp8.execute

    def poison(args, kwargs):
        for tensor in args[6:9]:
            tensor.fill_(float("nan"))
        kwargs["workspace"].fill_(0xBD)

    def exercise(api, *args, **kwargs):
        original(api, *args, **kwargs)
        artifact = api._prepared.artifact
        for name in ("seq_q_lens", "seq_kv_lens"):
            lens = kwargs[name]
            kwargs[name] = torch.cat((torch.zeros(1, device=lens.device, dtype=torch.int32), lens.cumsum(0, dtype=torch.int32)))
        poison(args, kwargs)
        capture = torch.cuda.CUDAGraph()
        try:
            with monkeypatch.context() as patcher:
                patcher.setattr(cute, "compile", lambda *a, **k: pytest.fail("length/prefix form must reuse the compiled host"))
                original(api, *args, **kwargs)
                with torch.cuda.graph(capture):
                    original(api, *args, **kwargs)
            poison(args, kwargs)
            capture.replay()
            assert api._prepared.artifact is artifact
        finally:
            capture.reset()

    monkeypatch.setattr(SdpaBwdDslSm107Mxfp8, "execute", exercise)
    _run_mx_direct((129, 63, 97), (113, 75, 141), token_major_stats=token_major)


@requires_rubin
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_prepared_thd_mxfp8_rebind_lengths_and_replay(causal, monkeypatch, ds_policy):
    """The prepared THD plan rebinds new buffers AND new lengths (every length fact is a device value) without rebuilding tensor
    operands, allocating, synchronizing or compiling, and replays under CUDA-graph capture with changed lengths and data.  The new
    cases keep the SAME packed SF tile totals per side (the tile count is read off the bound buffers' byte sizes, so the buffers are
    reused as they are) while every sequence length moves; GQA with a padded capacity: the bounded fold must stop at every rebound
    live total."""
    import cutlass.cute as cute

    from cudnn.sdpa.bwd.api_dsl import WorkspaceCarver

    first = _run_mx_direct((129, 97, 63), (143, 83, 79), h=4, hkv=2, poison=True, pad_cap=64, causal=causal)
    api, cap_q, cap_kv = first.api, first.case.cap_q, first.case.cap_kv
    tiles_q, tiles_kv = first.case.tiles_q, first.case.tiles_kv
    grads = dict(dq=first.dq, dk=first.dk, dv=first.dv)
    buffers = {name: getattr(first.case, name) for name in ("q", "q_T", "k", "k_T", "v", "do", "do_T", "o", "do_f16", "lse")}
    sfs = first.case.sf
    lens = dict(q=first.kwargs["seq_q_lens"], kv=first.kwargs["seq_kv_lens"])
    ws = first.ws

    def load(lens_q, lens_kv):
        """A NEW case into the SAME buffers (copies, no new operands) with the same packed capacities and SF tile totals."""
        case = _thd_mx_case(lens_q, lens_kv, 4, 2, cap_q=cap_q, cap_kv=cap_kv, poison=True, causal=causal, quantize_ds=first.case.quantize_ds)
        assert (case.tiles_q, case.tiles_kv) == (tiles_q, tiles_kv), "the rebind keeps the packed SF tile totals (the buffers are reused)"
        for name, buf in buffers.items():
            buf.copy_(getattr(case, name))
        for name in _SF_ALL:
            sfs[name].copy_(case.sf[name])
        lens["q"].copy_(torch.tensor(lens_q, dtype=torch.int32, device="cuda"))
        lens["kv"].copy_(torch.tensor(lens_kv, dtype=torch.int32, device="cuda"))
        for x in grads.values():
            x.fill_(float("nan"))
        _sentinel_tails(case, grads["dq"], grads["dk"], grads["dv"])
        ws.fill_(0xBD)
        return case

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared THD rebuilt tensor operands, allocated, synchronized or compiled")

    def execute_guarded():
        with monkeypatch.context() as patcher:
            for name in ("view", "reshape", "as_strided", "permute", "transpose", "copy_", "zero_"):
                patcher.setattr(torch.Tensor, name, forbidden)
            for name in ("empty", "empty_like", "zeros", "zeros_like"):
                patcher.setattr(torch, name, forbidden)
            patcher.setattr(WorkspaceCarver, "__init__", forbidden)
            patcher.setattr(cute, "compile", forbidden)
            api.execute(*first.tensors, **first.kwargs)

    def verify(case):
        torch.cuda.synchronize()
        _assert_tails_untouched(case, grads["dq"], grads["dk"], grads["dv"])
        _check_mx(case, grads["dq"], grads["dk"], grads["dv"])

    # tiles (2, 1, 1) on both sides, as the first case, inside the first case's packed capacities (353 q / 369 kv tokens)
    case = load((200, 100, 50), (250, 30, 80))
    torch.cuda.set_sync_debug_mode("error")
    try:
        execute_guarded()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    verify(case)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            execute_guarded()
        case = load((131, 120, 38), (200, 60, 100))
        capture.replay()
        verify(case)
    finally:
        capture.reset()


def _prepared_mxfp8_thd_case():
    """One executed THD MXFP8 plan (GQA, causal, poisoned capacity, the shipped policy) as the reload child wants it: ``graph`` (the
    standalone shim), ``pack`` (the execute kwargs), ``workspace``, the live ``outs_t`` and the first run's bits in ``outs``."""
    run = _run_mx_direct((129, 97, 63), (143, 83, 79), h=4, hkv=2, poison=True, pad_cap=64, sf_tail_tiles=1, causal=True)
    names = ("q_tensor", "k_tensor", "v_tensor", "o_tensor", "do_tensor", "stats_tensor", "dq_tensor", "dk_tensor", "dv_tensor")
    pack = dict(zip(names, run.tensors))
    pack.update({k: v for k, v in run.kwargs.items() if k != "workspace" and v is not None})
    case = SimpleNamespace(graph=_StandaloneGraphShim(run.api), pack=pack, workspace=run.ws, outs_t=dict(dQ=run.dq, dK=run.dk, dV=run.dv))
    case.outs = {name: t.clone() for name, t in case.outs_t.items()}
    case.live = dict(dQ=run.case.t_q, dK=run.case.t_kv, dV=run.case.t_kv)  # packed token totals: the rows the chain writes
    return case


def _check_prepared_mxfp8_thd(case):
    """The live outputs are BITWISE the first run's (two-launch pin).  Live = the packed totals: the reload protocol NaN-fills the whole
    output tensors before a replay, and the capacity tail past the totals is never written (the tails' untouched pin is
    ``_assert_tails_untouched`` on the first run), so the comparison stops at the live total."""
    torch.cuda.synchronize()
    for name, t in case.outs_t.items():
        live = case.live[name]
        _bitwise(f"{name}: a re-execute / replay of the prepared THD launch changed the bits", t[:, :live], case.outs[name][:, :live])


@requires_rubin
def test_prepared_thd_mxfp8_artifact_reloads_in_fresh_process(tmp_path):
    from prepared_bwd_cache_utils import check_backward_artifact_reload

    check_backward_artifact_reload("sm107", "mxfp8_thd", "float8_e4m3fn", tmp_path)


# --------------------------------------------------------------------------- the graph path: ragged MXFP8 tensors -> lower_dsl_bwd_mxfp8 -> packed views

_GRAPH_SF = {
    "sf_q": "descale_q",
    "sf_q_T": "descale_q_T",
    "sf_k": "descale_k",
    "sf_k_T": "descale_k_T",
    "sf_v": "descale_v",
    "sf_do": "descale_dO",
    "sf_do_T": "descale_dO_T",
}


def _build_thd_mx_graph(case, *, stats_layout="head_major", declare_totals=True, envelope_q=None, **sdpa_kwargs):
    """A ragged ``sdpa_mxfp8_backward`` graph over ``case``'s packed buffers: the nine payload / half ports declared as the ENVELOPE
    (B, H, S_max, D) plus a per-tensor ragged offset -- how cuDNN spells a packed tensor -- the seven scale-factor tensors declared
    with the envelope's F8_128x4 dims (``(B, H, ceil128(S_max), 8)``: the forward's ragged convention; the PACKED buffer bound to them
    carries the packed tile count in its byte size), the Stats ragged in either packing.  ``declare_totals`` passes
    ``max_total_seq_len_q/kv`` (the node attribute the quantized backward bindings carry)."""
    b, h, hkv, d, dev = case.b, case.h, case.hkv, case.d, case.q.device
    e4m3, bf16, f32 = cudnn.data_type.FP8_E4M3, cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    s_max_q, s_max_kv = envelope_q or max(max(case.lens_q), 1), max(max(case.lens_kv), 1)
    st_q = [s_max_q * h * d, d, h * d, 1]
    st_kv = [s_max_kv * hkv * d, d, hkv * d, 1]
    g = cudnn.pygraph(io_data_type=e4m3, intermediate_data_type=f32, compute_data_type=f32)
    ro_q_t = (torch.tensor(case.cu_q, dtype=torch.int64, device=dev) * (h * d)).view(b + 1, 1, 1, 1)
    ro_k_t = (torch.tensor(case.cu_k, dtype=torch.int64, device=dev) * (hkv * d)).view(b + 1, 1, 1, 1)
    vp, t = {}, {}

    def _ragged(name, s_max, stride, nh, dt, ro_t):
        x = g.tensor(name=name, dim=[b, nh, s_max, d], stride=stride, data_type=dt)
        ro = g.tensor(name=f"{name}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
        x.set_ragged_offset(ro)
        vp[ro] = ro_t
        return x

    for n, dt in (("q", e4m3), ("q_T", e4m3), ("o_f16", bf16), ("dO_f16", bf16), ("dO", e4m3), ("dO_T", e4m3)):
        t[n] = _ragged(n, s_max_q, st_q, h, dt, ro_q_t)
    for n in ("k", "k_T", "v"):
        t[n] = _ragged(n, s_max_kv, st_kv, hkv, e4m3, ro_k_t)
    t_cap = max(64, -(-case.cap_q // 64) * 64)
    if stats_layout == "head_major":
        stats_stride = [h * t_cap, t_cap, 1, 1]
        stats_stor = torch.zeros(h * t_cap, dtype=torch.float32, device=dev)
        stats_stor.as_strided((1, h, case.cap_q), (h * t_cap, t_cap, 1)).copy_(case.lse[:, :, : case.cap_q])
        stats_ro_t = torch.tensor(case.cu_q, dtype=torch.int64, device=dev).view(b + 1, 1, 1, 1)
    else:
        stats_stride = [s_max_q * h, 1, h, 1]
        stats_stor = case.lse[0].transpose(0, 1).contiguous().reshape(-1)
        stats_ro_t = (torch.tensor(case.cu_q, dtype=torch.int64, device=dev) * h).view(b + 1, 1, 1, 1)
    st = g.tensor(name="stats", dim=[b, h, s_max_q, 1], stride=stats_stride, data_type=f32)
    st_ro = g.tensor(name="stats_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
    st.set_ragged_offset(st_ro)
    vp[st_ro], vp[st] = stats_ro_t, stats_stor
    for name, port in _GRAPH_SF.items():
        nh, s_max = (hkv, s_max_kv) if name in ("sf_k", "sf_k_T", "sf_v") else (h, s_max_q)
        dims = (b, nh, _ceil128(s_max), _SF_GROUPS)
        stride = [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1]
        t[port] = g.tensor(name=port, dim=list(dims), stride=stride, data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)
        # a packed buffer of EXACTLY the declared bytes binds in the declared shape (the dense-capacity cell); the smaller packed-count
        # form otherwise -- the engine reads the bound tensor's own byte size either way
        buf = case.sf[name]
        vp[t[port]] = buf.reshape(dims) if buf.numel() == math.prod(dims) else buf
    slq = torch.tensor(case.lens_q, dtype=torch.int32, device=dev).view(b, 1, 1, 1)
    slk = torch.tensor(case.lens_kv, dtype=torch.int32, device=dev).view(b, 1, 1, 1)
    tq_len = g.tensor(name="seq_len_q", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    tk_len = g.tensor(name="seq_len_kv", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    vp[tq_len], vp[tk_len] = slq, slk
    kw = dict(name="mb", attn_scale=case.scale, use_padding_mask=True, seq_len_q=tq_len, seq_len_kv=tk_len)
    if declare_totals:
        kw.update(max_total_seq_len_q=case.cap_q, max_total_seq_len_kv=case.cap_kv)
    kw.update(sdpa_kwargs)
    outs = g.sdpa_mxfp8_backward(
        q=t["q"], q_T=t["q_T"], k=t["k"], k_T=t["k_T"], v=t["v"], o_f16=t["o_f16"], dO_f16=t["dO_f16"], dO=t["dO"], dO_T=t["dO_T"], stats=st,
        **{port: t[port] for port in _GRAPH_SF.values()}, **kw,
    )  # fmt: skip
    t.update(zip(("dQ", "dK", "dV"), outs[:3]))
    for name, s_max, stride, nh, ro_t in (("dQ", s_max_q, st_q, h, ro_q_t), ("dK", s_max_kv, st_kv, hkv, ro_k_t), ("dV", s_max_kv, st_kv, hkv, ro_k_t)):
        t[name].set_output(True).set_data_type(bf16).set_dim([b, nh, s_max, d]).set_stride(stride)
        ro = g.tensor(name=f"{name}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
        t[name].set_ragged_offset(ro)
        vp[ro] = ro_t
    for a in outs[3:]:  # the amax ports stay virtual (the row declines amax); validate() wants their dims
        a.set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(f32)
    vp.update(
        {
            t["q"]: case.q,
            t["q_T"]: case.q_T,
            t["k"]: case.k,
            t["k_T"]: case.k_T,
            t["v"]: case.v,
            t["o_f16"]: case.o,
            t["dO_f16"]: case.do_f16,
            t["dO"]: case.do,
            t["dO_T"]: case.do_T,
        }
    )
    return g, vp, t


def _run_mx_graph(
    lens_q,
    lens_kv,
    *,
    h=2,
    hkv=None,
    stats_layout="head_major",
    poison=False,
    pad_cap=0,
    sf_tail_tiles=0,
    poison_outputs=False,
    runs=1,
    check=None,
    sf_dense_capacity=False,
    **kw,
):
    """Build the ragged MXFP8 graph, PIN the engine, execute, compare per sequence.  ``sf_dense_capacity`` sizes the scale-factor buffers
    at exactly the bytes the graph DECLARES, ``(B, H, ceil128(S_max), 8)`` = ``B x ceil(S_max / 128)`` packed tiles per side, and binds
    them in that shape (the forward's graph layout)."""
    if not _binding_declares_totals():
        pytest.skip("the native sdpa_mxfp8_backward binding does not declare max_total_seq_len_q/kv (a stale extension); the direct tier carries the numerics")
    if sf_dense_capacity:
        b_ = len(lens_q)
        sf_tail_tiles = (
            b_ * _tiles(max(max(lens_q), 1)) - sum(_tiles(n) for n in lens_q),
            b_ * _tiles(max(max(lens_kv), 1)) - sum(_tiles(n) for n in lens_kv),
        )
    case = _thd_mx_case(
        lens_q, lens_kv, h, hkv, cap_q=sum(lens_q) + pad_cap, cap_kv=sum(lens_kv) + pad_cap, poison=poison, sf_tail_tiles=sf_tail_tiles,
        causal=bool(kw.get("use_causal_mask") or kw.get("use_causal_mask_bottom_right")), bottom_right=bool(kw.get("use_causal_mask_bottom_right")),
        window_left=kw.get("left_bound"), quantize_ds=_block_scaled(),
    )  # fmt: skip
    g, vp, t = _build_thd_mx_graph(case, stats_layout=stats_layout, **kw)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    assert _plan_index(g, _ENGINE) is not None, f"{_ENGINE} not offered; plans = {[g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]}"
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    fill = float("nan") if poison_outputs else 0.0
    dev = "cuda"
    dq = torch.empty(1, case.cap_q, case.h, _D, device=dev, dtype=_BF16)
    dk, dv = (torch.empty(1, case.cap_kv, case.hkv, _D, device=dev, dtype=_BF16) for _ in range(2))
    vp.update({t["dQ"]: dq, t["dK"]: dk, t["dV"]: dv})
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    outs = []
    for _ in range(runs):
        for x in (dq, dk, dv):
            x.fill_(fill)
        _sentinel_tails(case, dq, dk, dv)
        ws.fill_(0xFF)
        g.execute(vp, ws)
        torch.cuda.synchronize()
        outs.append(tuple(x.clone() for x in (dq, dk, dv)))
    for name, x, live in (("dQ", dq, case.t_q), ("dK", dk, case.t_kv), ("dV", dv, case.t_kv)):
        assert _finite(x[0, :live]), f"{name} has non-finite values in the packed region"
    _assert_tails_untouched(case, dq, dk, dv)
    (check or _check_mx)(case, dq, dk, dv)
    return SimpleNamespace(case=case, dq=dq, dk=dk, dv=dv, outs=outs, graph=g)


@requires_rubin
def test_graph_thd_mxfp8_self_attention(ds_policy):
    """The whole point of the graph tier: a ragged MXFP8 graph declaring its packed totals reaches the kernels through the engine."""
    _run_mx_graph((300, 128, 200), (300, 128, 200))


@requires_rubin
@pytest.mark.parametrize("layout", ("head_major", "token_major"))
def test_graph_thd_mxfp8_stats_packings(layout):
    _run_mx_graph((300, 128), (300, 128), stats_layout=layout)


@requires_rubin
def test_graph_thd_mxfp8_causal_gqa_nan_tail():
    _run_mx_graph((300, 128, 200), (300, 128, 200), h=4, hkv=2, poison=True, pad_cap=384, sf_tail_tiles=2, use_causal_mask=True)


@requires_rubin
def test_graph_thd_mxfp8_two_launches_are_bitwise():
    run = _run_mx_graph((300, 128, 200), (300, 128, 200), h=4, hkv=2, use_causal_mask=True, runs=2)
    for name, a, b in zip(("dQ", "dK", "dV"), run.outs[0], run.outs[1]):
        _bitwise(f"{name} differs between two launches", a, b)


@requires_rubin
def test_graph_thd_mxfp8_sf_declared_at_the_dense_capacity():
    """The graph tier's form of the packed scale-factor contract: the seven tensors declared with the envelope's F8_128x4 dims
    ``(B, H, ceil128(S_max), 8)`` and bound with buffers of EXACTLY those bytes, in that shape -- ``B x ceil(S_max / 128)`` packed tiles
    per head, the forward's layout, above the packed bound for these lengths -- lower, bind and compute exactly through the engine."""
    run = _run_mx_graph((300, 128, 200), (300, 128, 200), sf_dense_capacity=True)
    c = run.case
    assert all(_sf_tile_count(c.sf[name]) == c.b * _tiles(300) > _tiles(c.cap_q) + c.b for name in _SF_ALL)


# --------------------------------------------------------------------------- rejects and host pins (no Rubin GPU needed)


def _desc(shape, dtype, name, stride=None):
    from cudnn.api_base import TensorDesc

    stride = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape))) if stride is None else tuple(stride)
    return TensorDesc(
        dtype=dtype,
        shape=tuple(shape),
        stride=stride,
        stride_order=TensorDesc._compute_stride_order(tuple(shape), stride),
        device=torch.device("cuda", 0),
        name=name,
    )


def _bshd_desc(b, nh, s, dtype, name):
    return _desc((b, nh, s, _D), dtype, name, stride=(s * nh * _D, _D, nh * _D, 1))


def _sf_desc(nh, tiles, *, rowwise, name, over=None):
    """A PACKED scale-factor TensorDesc of ``tiles`` per head: rowwise ``(1, H, 128 * T, 8)``, columnwise ``(1, H, 4 * T, 256)``."""
    shape = over or ((1, nh, _SF_ATOM_ROWS * tiles, _SF_GROUPS) if rowwise else (1, nh, (_SF_ATOM_ROWS // _MX_BLOCK) * tiles, _D))
    return _desc(shape, torch.int8, name)


def _thd_mx_adapter(b=2, h=2, hkv=None, s_max=256, *, tiles_q=None, tiles_kv=None, sf_over=None, **kw):
    """A THD MXFP8 adapter over the ENVELOPE (B, H, S_max, D) as ``TensorDesc``s with PACKED scale-factor descs of ``tiles_q`` /
    ``tiles_kv`` tiles per head (default: the envelope's ``B * ceil(S_max / 128)``) -- no buffer, no device: the host rejects check
    shapes, dtypes and plan facts, nothing executes.  ``sf_over`` replaces one SF desc's shape."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Mxfp8

    hkv = h if hkv is None else hkv
    tiles_q = b * _tiles(s_max) if tiles_q is None else tiles_q
    tiles_kv = b * _tiles(s_max) if tiles_kv is None else tiles_kv
    sf_over = sf_over or {}
    sf = dict(
        sf_q=_sf_desc(h, tiles_q, rowwise=True, name="sf_q", over=sf_over.get("sf_q")),
        sf_q_T=_sf_desc(h, tiles_q, rowwise=False, name="sf_q_T", over=sf_over.get("sf_q_T")),
        sf_k=_sf_desc(hkv, tiles_kv, rowwise=True, name="sf_k", over=sf_over.get("sf_k")),
        sf_k_T=_sf_desc(hkv, tiles_kv, rowwise=False, name="sf_k_T", over=sf_over.get("sf_k_T")),
        sf_v=_sf_desc(hkv, tiles_kv, rowwise=True, name="sf_v", over=sf_over.get("sf_v")),
        sf_do=_sf_desc(h, tiles_q, rowwise=True, name="sf_do", over=sf_over.get("sf_do")),
        sf_do_T=_sf_desc(h, tiles_q, rowwise=False, name="sf_do_T", over=sf_over.get("sf_do_T")),
    )
    return SdpaBwdDslSm107Mxfp8(
        _bshd_desc(b, h, s_max, _T_E4M3, "q"),
        _bshd_desc(b, hkv, s_max, _T_E4M3, "k"),
        _bshd_desc(b, hkv, s_max, _T_E4M3, "v"),
        _bshd_desc(b, h, s_max, _BF16, "o"),
        _bshd_desc(b, h, s_max, _T_E4M3, "dO"),
        _desc((b, h, s_max, 1), torch.float32, "stats"),
        _bshd_desc(b, h, s_max, _BF16, "dQ"),
        _bshd_desc(b, hkv, s_max, _BF16, "dK"),
        _bshd_desc(b, hkv, s_max, _BF16, "dV"),
        sample_q_T=_bshd_desc(b, h, s_max, _T_E4M3, "q_T"),
        sample_k_T=_bshd_desc(b, hkv, s_max, _T_E4M3, "k_T"),
        sample_do_T=_bshd_desc(b, h, s_max, _T_E4M3, "dO_T"),
        sample_do_f16=_bshd_desc(b, h, s_max, _BF16, "dO_f16"),
        **{f"sample_{n}": sf[n] for n in sf},
        scale_softmax=1.0 / math.sqrt(_D),
        thd=True,
        **kw,
    )


_TOTALS = dict(max_total_seq_len_q=400, max_total_seq_len_kv=400)


def test_capabilities_claim_thd_with_declared_totals():
    """The MXFP8 row claims THD with declared totals (its body carries the THD arm now: packed operands, the per-sequence
    tile-padded scale factors, the SF pad pre-pass) and bottom-right at ANY S_q, keeps declining the dense padding graph (it
    carries ``seq_len_q``, which no body threads), the forward-only ``cu_seq_len`` ports and every amax output.  Rule S2: the tracker
    rows change with these."""
    c = _spec().capabilities
    assert c.is_mxfp8 and c.thd and c.thd_declared_totals, "the MXFP8 row serves THD with declared packed totals"
    assert not c.cu_seq_len, "no BACKWARD node carries cu_seq_len_* (forward-only ports)"
    assert c.bottom_right_s_q_multiple == 1
    assert not c.padded, "the dense padding graph stays declined (seq_len_q by construction; the standalone per-batch kv lengths are served)"
    assert not c.amax_dgrad, "no amax in the MXFP8 row, THD or dense"


def test_mxfp8_thd_plan_facts_are_in_the_constructors_own_signature():
    """The engines' lowering forwards a plan-time fact to the adapter only when it appears in the constructor's OWN signature; the
    MXFP8 ctor re-declares the THD facts and ``external_delta`` next to its own keyword surface."""
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDsl
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Mxfp8

    own = inspect.signature(SdpaBwdDslSm107Mxfp8.__init__).parameters
    base = inspect.signature(SdpaBwdDsl.__init__).parameters
    for name in ("thd", "max_total_seq_len_q", "max_total_seq_len_kv", "thd_stats_token_major", "thd_stats_head_stride"):
        assert name in own and name in base, f"{name} must be in the MXFP8 row's own constructor signature (the lowering forwards only those)"
        assert own[name].default == base[name].default, name
    assert "external_delta" in own and "p_scale_log2" in own
    assert SdpaBwdDslSm107Mxfp8._THD_SUPPORTED is True


def test_mxfp8_thd_requires_declared_totals():
    """THD without ``max_total_seq_len_*`` is DECLINED, not silently mis-sized (the kv-blocked workspace, the packed delta, the SF
    staging copies and the partials are fixed at build time from the packed token capacity)."""
    with pytest.raises(ValueError, match="max_total_seq_len"):
        _thd_mx_adapter().check_support()
    assert _thd_mx_adapter(**_TOTALS).check_support()


def test_mxfp8_thd_refuses_the_dense_length_flags():
    """THD carries its lengths in the metadata buffer: ``seq_kv_lens_present`` / ``seq_q_lens_present`` with THD are refused (two
    sources of truth drift apart) -- while per-batch kv lengths alone are served on this row and per-batch Q lengths declined for
    their own reason."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _thd_mx_adapter(seq_kv_lens_present=True, **_TOTALS).check_support()
    with pytest.raises(ValueError, match="mutually exclusive"):
        _thd_mx_adapter(seq_q_lens_present=True, **_TOTALS).check_support()
    assert _thd_mx_adapter(**_TOTALS).check_support()


def test_thd_host_helpers_keep_their_forms():
    """The THD columnwise dequant kernel (``_dequant_mxfp8_to_bf16_thd``, a ``@cute.kernel``) traces two helpers that must stay
    ``@cute.jit`` -- ``_thd_prefix_bases`` (the token / tile prefix bases of one side) and ``_thd_seq_of`` (the sequence lookup) --
    while the THD hosts' delta-geometry helper ``_thd_delta_geometry`` is plain Python (int arithmetic on two ``config`` entries, the
    form of ``_dq_launches``).  Read from the SOURCE: ``cute.jit`` returns an ordinary function object, so a decorator displaced by an
    insertion between it and its ``def`` changes which helper is traced as device code and nothing at run time reports it."""
    import ast
    from pathlib import Path

    from cudnn.sdpa.bwd import prepared as prep

    path = Path(prep.__file__).parent / "kernels" / "sm107" / "prepared_host.py"
    forms = {n.name: [ast.unparse(d) for d in n.decorator_list] for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
    assert forms["_dequant_mxfp8_to_bf16_thd"] == ["cute.kernel"]
    assert forms["_thd_prefix_bases"] == ["cute.jit"] and forms["_thd_seq_of"] == ["cute.jit"], "the dequant kernel's helpers are traced as device code"
    assert forms["_thd_delta_geometry"] == [] and forms["_dq_launches"] == [], "plan-time int arithmetic stays plain Python"


def test_mxfp8_thd_serves_the_external_delta(monkeypatch):
    """A caller's delta is SERVED under THD on the MXFP8 row exactly as on the other two: the plan fact passes ``check_support``;
    its contract is the packed head-major ``[1, H_q, ceil128(T_q)]`` fp32 layout in TRUE units (bitwise the row's own ``dot_do_o``
    over the packed bf16 ``o_f16`` / ``dO_f16`` ports when a producer reproduces that order -- which is why the chain's own pre-pass
    is simply not launched); the THD carve drops its own ``delta`` region exactly while the default plan keeps it; the THD roles
    carry the delta LAST -- slot 22, after the two lengths at 9 / 10, the four payloads and the seven scale-factor blobs -- on the
    roles AND the attributes, standalone-only on the launch spec, whose ``geometry`` carries a trailing None for it (the host views
    the delta from ``config``), and the plan fact reaches the host compile and keys the artifact."""
    from cudnn.sdpa.bwd import prepared_sm107
    from cudnn.sdpa.bwd.prepared import Operand
    from cudnn.sdpa.bwd.prepared_sm107 import (
        ATTRIBUTES_MXFP8_THD,
        EXTERNAL_DELTA_ROLE,
        MXFP8_PAYLOADS,
        MXFP8_SF,
        MXFP8_SF_KV_SIDE,
        MXFP8_SF_Q_SIDE,
        ROLES,
        ROLES_MXFP8,
        ROLES_MXFP8_THD,
    )
    from cudnn.sdpa.fwd.api_dsl import ws_align

    h, t_pad = 2, -(-400 // 128) * 128
    ext = _thd_mx_adapter(h=h, external_delta=True, **_TOTALS)
    assert ext.check_support() and ext.external_delta is True
    api = _thd_mx_adapter(h=h, **_TOTALS)
    assert api.check_support() and api.external_delta is False
    assert ext.external_delta_shape == api.external_delta_shape == (1, h, t_pad)
    assert "delta" in [name for name, _n, _d in api._scratch_plan()], "the default THD plan keeps the chain's own (packed) delta region"
    assert "delta" not in [name for name, _n, _d in ext._scratch_plan()], "the external plan carves no delta region"
    assert api.scratch_workspace_bytes() - ext.scratch_workspace_bytes() == ws_align(h * t_pad * 4), "the carve lost exactly the delta region"
    assert ROLES_MXFP8[-2:] == ("seq_kv", EXTERNAL_DELTA_ROLE), "the dense MXFP8 roles: the lengths, then the delta (appended)"
    assert ROLES_MXFP8_THD == ROLES[:9] + ("seq_q", "seq_kv") + MXFP8_PAYLOADS + MXFP8_SF + (
        EXTERNAL_DELTA_ROLE,
    ), "the THD roles: the nine packed tensors, the two lengths at slots 9 / 10, the family's extras, the delta LAST (slot 22)"
    assert ATTRIBUTES_MXFP8_THD[9:11] == ("seq_len_q", "seq_len_kv") and len(ATTRIBUTES_MXFP8_THD) == len(ROLES_MXFP8_THD) == 23
    assert ATTRIBUTES_MXFP8_THD[-1] == EXTERNAL_DELTA_ROLE and ROLES_MXFP8_THD.index(EXTERNAL_DELTA_ROLE) == 22
    with pytest.raises(ValueError, match="external_delta=False"):
        api._check_external_delta(torch.zeros(1, h, t_pad))
    with pytest.raises(ValueError, match="delta_tensor is required"):
        ext._check_external_delta(None)
    # the launch spec (the real builder over a fake artifact entry): the slot, its specialization per plan, the standalone-only role, the
    # trailing geometry entry, the key
    own_spec, own_calls = _spec_without_compiling(monkeypatch, api, prepared_sm107.compile_plan_mxfp8_thd, "compile_host_mxfp8_thd")
    ext_spec, ext_calls = _spec_without_compiling(monkeypatch, ext, prepared_sm107.compile_plan_mxfp8_thd, "compile_host_mxfp8_thd")
    for spec, calls in ((own_spec, own_calls), (ext_spec, ext_calls)):
        assert not spec.native_binding and spec.length_form and spec.scale_log2 and spec.roles == ROLES_MXFP8_THD and len(spec.operands) == 23
        assert spec.standalone_only_roles == (EXTERNAL_DELTA_ROLE,), "no graph declares a delta: framed absent on the graph path"
        assert spec.packed_tile_groups == (MXFP8_SF_Q_SIDE, MXFP8_SF_KV_SIDE)
        geometry = calls[0][0][4]
        assert len(geometry) == 23 and geometry[-1] is None and geometry[9] is None and geometry[10] is None, "geometry[i] is operand i's layout"
    assert own_spec.operands[22] is None and ext_spec.operands[22] == Operand("float32", (1, h, t_pad), (h * t_pad, t_pad, 1), h * t_pad, 16, 4)
    assert own_calls[0][1]["external_delta"] is False and ext_calls[0][1]["external_delta"] is True and own_calls[0][0][7] != ext_calls[0][0][7]


def test_mxfp8_thd_execute_requires_both_lengths(monkeypatch):
    """A THD plan's execute needs ``seq_q_lens`` AND ``seq_kv_lens`` -- refused typed before compile, ahead of the operand checks."""
    api = _thd_mx_adapter(**_TOTALS)
    monkeypatch.setattr(api, "compile", lambda: pytest.fail("refused before compile"))
    lens = torch.zeros(2, dtype=torch.int32)
    dummy = torch.empty(1)
    extras = {name + "_tensor": dummy for name in ("q_T", "k_T", "do_T", "do_f16")}
    extras.update({name: dummy for name in _SF_ALL})
    with pytest.raises(ValueError, match="seq_q_lens AND seq_kv_lens"):
        api.execute(*([None] * 9), seq_kv_lens=lens, **extras)
    with pytest.raises(ValueError, match="seq_q_lens AND seq_kv_lens"):
        api.execute(*([None] * 9), seq_q_lens=lens, **extras)


def test_mxfp8_thd_sf_shape_pin_and_packed_tile_count_rejects():
    """The packed scale-factor contract at plan BUILD (TensorDescs, no device): every SF desc must hold WHOLE packed tile rows
    (``H * 1024`` bytes per 128-token tile across the side's heads), and ``sf_v`` keeps its ROWWISE shape pin in the packed forms
    ``(1, H_kv, 128 k, 8)`` rows / ``(1, H_kv, T_sf, 1024)`` tiles (a columnwise-shaped binding has the same byte count and would be a
    wrong dV) -- each a ValueError before any compile.  The LIVE tile count is NOT a build fact: a count mismatch between ``sf_k``
    and ``sf_v`` is admitted here and refused per call by ``bind`` on the bound buffers, and a sample ABOVE the packed bound
    ``ceil(T_cap / 128) + B`` raises the plan's capacity to its own count (a ragged graph declares the dense capacity), so only a
    BOUND buffer above that capacity is refused (``test_mxfp8_thd_bind_derives_the_packed_tile_count_per_call``,
    ``test_thd_mxfp8_execute_refuses_a_mismatched_or_oversized_packed_sf_count``)."""
    b, s_max = 2, 256
    cap_tiles = _tiles(_TOTALS["max_total_seq_len_kv"]) + b  # the bound: ceil(400 / 128) + 2 = 6
    ok = _thd_mx_adapter(b=b, s_max=s_max, tiles_q=3, tiles_kv=3, **_TOTALS)
    assert ok.check_support() and ok._compiled is None
    with pytest.raises(ValueError, match=r"whole packed SF tile rows"):
        _thd_mx_adapter(b=b, s_max=s_max, tiles_q=3, tiles_kv=3, sf_over=dict(sf_q=(1, 2, _SF_ATOM_ROWS * 3 + 64, _SF_GROUPS)), **_TOTALS).check_support()
    with pytest.raises(ValueError, match=r"sf_v must be the ROWWISE"):
        _thd_mx_adapter(b=b, s_max=s_max, tiles_q=3, tiles_kv=3, sf_over=dict(sf_v=(1, 2, (_SF_ATOM_ROWS // _MX_BLOCK) * 3, _D)), **_TOTALS).check_support()
    assert _thd_mx_adapter(
        b=b, s_max=s_max, tiles_q=3, tiles_kv=3, sf_over=dict(sf_v=(1, 2, 3, _SF_TILE_BYTES)), **_TOTALS
    ).check_support(), "the packer's tile form of sf_v is admitted"
    # per-call facts, admitted at build: the sample descs' counts may differ and exceed the cap (the plan's capacity bounds the BOUND buffer)
    assert _thd_mx_adapter(b=b, s_max=s_max, tiles_q=3, tiles_kv=3, sf_over=dict(sf_k=(1, 2, _SF_ATOM_ROWS * 4, _SF_GROUPS)), **_TOTALS).check_support()
    assert _thd_mx_adapter(b=b, s_max=s_max, tiles_q=3, tiles_kv=cap_tiles + 1, **_TOTALS).check_support()
    # the plan's capacity is the LARGER of the packed bound and the declared sample's own count, per side: a sample above the bound
    # grows the staging copies and the operand span to its count; one at or below it leaves the bound as is
    assert _thd_mx_adapter(b=b, s_max=s_max, tiles_q=3, tiles_kv=3, **_TOTALS)._thd_sf_tiles_cap("sf_k") == cap_tiles
    above = _thd_mx_adapter(b=b, s_max=s_max, tiles_q=3, tiles_kv=cap_tiles + 1, **_TOTALS)
    assert above._thd_sf_tiles_cap("sf_k") == cap_tiles + 1 and above._thd_sf_tiles_cap("sf_v") == cap_tiles + 1
    assert above._thd_sf_tiles_cap("sf_q") == cap_tiles and above._sf_capacity_bytes("sf_v") == (cap_tiles + 1) * 2 * _SF_TILE_BYTES


@requires_sm80
def test_mxfp8_thd_bind_derives_the_packed_tile_count_per_call():
    """``prepared.bind`` on a packed scale-factor operand (``Operand.packed_tile_bytes``, ``BwdLaunchSpec.packed_tile_groups``): the live tile
    count is the bound buffer's byte size over the tile-row bytes -- it must be WHOLE tile rows, at most the plan's capacity (the
    operand's ``span``), and every role of a group (``sf_k`` / ``sf_v``: one K / V SF extent) must agree; the count is framed once per
    group after the length form, a side with no live tile framed as 1.  A synthetic two-role spec over real CUDA buffers (any CUDA
    device: the facts, not a kernel)."""
    from cudnn.sdpa.bwd.prepared import BwdLaunchSpec, Operand, bind
    from cudnn.sdpa.fwd.prepared import facts_of_tensor

    hkv, cap_tiles = 2, 6
    row = hkv * _SF_TILE_BYTES
    op = Operand("int8", (row * cap_tiles,), (1,), row * cap_tiles, 16, 1, opaque_bytes=True, packed_tile_bytes=row)
    spec = BwdLaunchSpec(
        None,
        None,
        (op, op),
        16,
        0,
        0.125,
        name="probe",
        roles=("sf_k", "sf_v"),
        attributes=("sf_k", "sf_v"),
        packed_tile_groups=(("sf_k", "sf_v"),),
        native_binding=False,
    )
    ws = torch.empty(64, dtype=torch.uint8, device="cuda")

    def sf(tiles, extra_rows=0):
        return torch.zeros(1, hkv, _SF_ATOM_ROWS * tiles + extra_rows, _SF_GROUPS, dtype=torch.uint8, device="cuda")

    def frame(k, v):
        return bind(spec, {"sf_k": facts_of_tensor(k), "sf_v": facts_of_tensor(v)}, ws.data_ptr(), 7)

    assert frame(sf(3), sf(3))[-2:] == [3, 7], "ONE count per group after the scalars, then the stream"
    assert frame(sf(1), sf(1))[-2] == 1
    with pytest.raises(ValueError, match="must share one packed SF tile count"):
        frame(sf(4), sf(3))
    with pytest.raises(ValueError, match="whole packed SF tile rows"):
        frame(sf(3, extra_rows=64), sf(3))
    with pytest.raises(ValueError, match="above the plan's capacity"):
        frame(sf(cap_tiles + 1), sf(cap_tiles + 1))
    assert frame(sf(cap_tiles), sf(cap_tiles))[-2] == cap_tiles, "the capacity itself is admitted"
    # the workspace-overlap check measures a packed operand by its LIVE bytes, not the plan's capacity (6 tiles here): one blob
    # carries sf_k (3 tiles), a gap of the workspace's size and sf_v (3 tiles); a caller workspace in the gap binds, one 16 bytes
    # inside sf_k's live bytes is refused
    live, pad = 3 * row, max(16, spec.workspace_bytes)
    blob = torch.zeros(2 * live + pad, dtype=torch.uint8, device="cuda")
    k3 = blob[:live].view(1, hkv, _SF_ATOM_ROWS * 3, _SF_GROUPS)
    v3 = blob[live + pad :].view(1, hkv, _SF_ATOM_ROWS * 3, _SF_GROUPS)
    gap = blob.data_ptr() + live
    assert bind(spec, {"sf_k": facts_of_tensor(k3), "sf_v": facts_of_tensor(v3)}, gap, 7)[-2] == 3, "a workspace right after the live bytes is no overlap"
    with pytest.raises(ValueError, match="workspace overlaps sf_k"):
        bind(spec, {"sf_k": facts_of_tensor(k3), "sf_v": facts_of_tensor(v3)}, gap - 16, 7)


@requires_rubin
def test_thd_mxfp8_execute_refuses_a_mismatched_or_oversized_packed_sf_count():
    """The execute surface of a compiled THD plan: the K / V scale-factor buffers must agree on their packed tile count, every SF buffer
    must hold whole tile rows, and none may exceed the plan's capacity (the larger of ``ceil(T_cap / 128) + B`` tiles and the declared
    sample's own count; the packed bound here) -- each a ValueError from the binder with NO launch (the artifact entry replaced by a
    recorder)."""
    from dataclasses import replace

    run = _run_mx_direct((300, 200), (300, 200), h=2)
    api, c = run.api, run.case
    launches = []
    api._prepared = replace(api._prepared, fn=lambda *args: launches.append(args))
    kw = dict(run.kwargs)
    one_more = torch.zeros(1, c.hkv, _SF_ATOM_ROWS * (c.tiles_kv + 1), _SF_GROUPS, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="must share one packed SF tile count"):
        api.execute(*run.tensors, **dict(kw, sf_k=one_more))
    with pytest.raises(ValueError, match="whole packed SF tile rows"):
        api.execute(*run.tensors, **dict(kw, sf_q=torch.zeros(1, c.h, _SF_ATOM_ROWS * c.tiles_q + 64, _SF_GROUPS, dtype=torch.uint8, device="cuda")))
    assert api._thd_sf_tiles_cap("sf_k") == _tiles(c.cap_kv) + c.b, "the sample's own count (5 tiles) sits below the packed bound here"
    cap = api._thd_sf_tiles_cap("sf_k") + 1
    over = torch.zeros(1, c.hkv, _SF_ATOM_ROWS * cap, _SF_GROUPS, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="above the plan's capacity"):
        api.execute(*run.tensors, **dict(kw, sf_k=over, sf_v=over))
    assert not launches, "a refused packed scale-factor binding must not launch"


def test_mxfp8_thd_scratch_plan_is_the_packed_carve(ds_policy):
    """The THD workspace of the MXFP8 row, in carve order: the packed head-major delta, ONE head chunk of the kv-BLOCKED dS workspace
    (bf16 under P-c; the e4m3 ``ds_dk`` payload under P-b), the metadata + the main kernel's ``10 + B`` tensor maps (five clamped
    payload maps, five clamped scale-factor maps, one clipped dV map per sequence -- the body's own slot count through
    ``_thd_map_slots``), stage 3's (B + 1) descriptors, then the family's THD regions: the int32 ``sf_meta`` (the two per-sequence
    SF TILE prefixes, ``2 (B + 1)`` words), the packed staging copies of the five hazard tensors' scale factors, the P-b payload /
    atom tensors (``ds_dq``, ``sf_ds_dk``, ``sf_ds_dq``) at the kv-blocked geometry or the P-c dequantized bf16 ``q_T`` / ``k_T`` over the
    packed tokens, the per-Q-head partials under GQA.  No staging slab of the dense path (the packed path reads the caller's buffers).
    ``regions[seq_kv].numel`` is pinned to the family's map-slot count."""
    from cudnn.frost.tile_dsl.thd import THD_BWD_MAPS_META_WORDS
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107
    from cudnn.sdpa.fwd.api_dsl import ws_align

    b, h, hkv = 2, 4, 2
    api = _thd_mx_adapter(b=b, h=h, hkv=hkv, **_TOTALS)
    assert api.check_support()
    assert api._thd_map_slots(b) == 10 + b, "the MXFP8 body writes 5 clamped payload + 5 clamped SF maps + B clipped dV maps"
    assert sm107.SdpaBwdDslSm107._thd_map_slots(api, b) == 5 + b, "the half / fp8 bodies write 5 clamped maps + B clipped dV maps"
    plan = {name: (tuple(int(x) for x in shape), dt) for name, shape, dt in api._scratch_shapes()}
    names = [name for name, _s, _d in api._scratch_shapes()]
    tq, tkv = api._t_q_cap, api._t_kv_cap
    assert names[:4] == ["delta", "ds_ws", "seq_kv", "desc_words"], names
    assert plan["delta"] == ((1, h, -(-tq // 128) * 128), torch.float32)
    assert plan["ds_ws"][0] == (1, api._qh_chunk, api._ws_rows_cap, api._sq_pad) and plan["ds_ws"][1] == (_T_E4M3 if api._ds_block_scaled else _BF16)
    assert plan["seq_kv"] == ((THD_BWD_MAPS_META_WORDS(b, 10 + b),), torch.int32), "the metadata words + (10 + B) tensor maps"
    assert plan["desc_words"] == (((b + 1) * 16,), torch.int64)
    assert plan["sf_meta"] == ((2 * (b + 1),), torch.int32), "cu_sf_q(B+1) | cu_sf_k(B+1), in TILES"
    for name in ("sf_v_pad", "sf_do_pad", "sf_doT_pad"):
        # the packed staging copy of a hazard tensor's scale factors: a 1-D byte region at the plan's SF tile capacity, in the slot the
        # dense plan uses for its zero-filled pad slab of the same tensor
        assert name in plan and plan[name][1] == torch.uint8 and len(plan[name][0]) == 1, f"{name}: {plan.get(name)}"
        assert plan[name][0][0] % _SF_TILE_BYTES == 0 and plan[name][0][0] >= _SF_TILE_BYTES * (
            hkv if name == "sf_v_pad" else h
        ), f"{name}: whole packed tile rows"
        assert plan[name][0][0] == api._sf_capacity_bytes({"sf_v_pad": "sf_v", "sf_do_pad": "sf_do", "sf_doT_pad": "sf_do_T"}[name]), name
    if api._ds_block_scaled:
        assert plan["ds_dq"] == ((1, api._qh_chunk, api._ws_rows_cap, api._sq_pad), _T_E4M3)
        assert plan["sf_ds_dk"] == ((1, api._qh_chunk, api._ws_rows_cap // 128, api._sq_pad // 128, 512), torch.uint8)
        assert plan["sf_ds_dq"] == ((1, api._qh_chunk, api._sq_pad // 128, api._ws_rows_cap // 128, 512), torch.uint8)
        for name in ("sf_qT_pad", "sf_kT_pad"):
            assert (
                name in plan and plan[name][1] == torch.uint8 and len(plan[name][0]) == 1
            ), f"{name}: the block-scale GEMMs' whole-atom SFB reads need the staged columnwise pads"
        assert not any(n in plan for n in ("q_T_bf16", "k_T_bf16"))
    else:
        assert plan["q_T_bf16"] == ((1, tq, h, _D), _BF16) and plan["k_T_bf16"] == (
            (1, tkv, hkv, _D),
            _BF16,
        ), "P-c: the dequantized stage-3 B operands over the packed tokens"
        assert not any(
            n in plan for n in ("ds_dq", "sf_ds_dk", "sf_ds_dq", "sf_qT_pad", "sf_kT_pad")
        ), "P-c dequantizes per token and never reads a columnwise pad byte"
    assert plan["dv_part"] == ((1, tkv, h, _D), _BF16) and plan["dk_part"] == (
        (1, tkv, h, _D),
        torch.float32 if api._ds_block_scaled else _BF16,
    ), "GQA: the per-Q-head partials over the packed kv capacity -- dK fp32 on the block-scaled chain (the bounded fold rounds it once), dV bf16"
    assert api._dk_part_fp32 is api._ds_block_scaled
    assert not any(
        name in plan for name in ("q_pad", "do_pad", "lse_pad", "k_pad", "v_pad", "do_T_pad", "sf_q_pad", "sf_k_pad", "dk_fold", "dv_fold")
    ), "no dense staging slab under THD: the packed path reads the caller's buffers; the harmless sf_q / sf_k are never re-staged"
    assert api.scratch_workspace_bytes() == sum(ws_align(math.prod(s) * dt.itemsize) for s, dt in plan.values())
    # the external-delta plan: the SAME carve minus its first region, exactly
    ext = _thd_mx_adapter(b=b, h=h, hkv=hkv, external_delta=True, **_TOTALS)
    assert ext.check_support() and [n for n, _s, _d in ext._scratch_shapes()] == names[1:]
    assert api.scratch_workspace_bytes() - ext.scratch_workspace_bytes() == ws_align(math.prod(plan["delta"][0]) * 4)


def test_mxfp8_thd_stage3_records_are_the_thd_arm_of_each_policy(ds_policy):
    """The adapter's THD stage-3 records: under P-c the base THD arm (bf16 renderings over the bf16 dS, ``thd_varlen`` +
    ``thd_rows_kv``, EPI_NONE, ``causal_shift`` 0 with the per-sequence diagonal read from the metadata, the window KEPT, dQ's
    ``b_head_group`` the GQA group under the single launch); under P-b the block-scale arm with the same THD fields and the SAME
    ``b_head_group`` (one dQ launch per head chunk: the arm indexes its packed B scale factors by the grouped head too) -- both
    admitted by the template's validator."""
    import types

    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE, EPI_NONE, validate_matmul_params

    mod = types.SimpleNamespace(CFG=types.SimpleNamespace(TILE_M=128, CTA_MMA=2))
    api = _thd_mx_adapter(h=4, hkv=2, is_causal=True, causal_bottom_right=True, window_size_left=63, **_TOTALS)
    assert api.check_support()
    dk, dq = api._stage3_records(mod, (256, 256))
    for rec in (dk, dq):
        validate_matmul_params(rec)
        assert rec.thd_varlen and rec.thd_rows_kv and rec.thd_causal_bottom_right and rec.epi_mode == EPI_NONE
        assert rec.causal_shift == 0 and rec.causal_window == 63 and rec.causal_gran == 256
    assert (dk.causal_mode, dq.causal_mode) == (CAUSAL_K_LO, CAUSAL_K_HI)
    if api._ds_block_scaled:
        assert dk.block_scale and dq.block_scale and dk.dtype_qkv == dq.dtype_qkv == DTYPE_E4M3
        assert dq.b_head_group == (2 if sm107.DQ_SINGLE_LAUNCH else 1) and dk.b_head_group == 1, "the block-scale arm's dQ takes the group like the plain one"
    else:
        assert not dk.block_scale and not dq.block_scale and dk.dtype_qkv == dq.dtype_qkv == DTYPE_BF16
        assert dq.b_head_group == (2 if sm107.DQ_SINGLE_LAUNCH else 1) and dk.b_head_group == 1
    dense = _thd_mx_adapter(h=4, hkv=2, **_TOTALS)
    assert dense.check_support()
    dk, dq = dense._stage3_records(mod, (256, 256))
    assert (dk.causal_mode, dq.causal_mode) == (CAUSAL_K_NONE, CAUSAL_K_NONE) and not dk.thd_causal_bottom_right and dk.thd_varlen and dq.thd_varlen


def test_mxfp8_thd_host_is_a_sibling_artifact_with_the_packed_sf_pre_pass():
    """Source pins on the MXFP8 THD host: a SIBLING artifact (``host_mxfp8_thd`` / ``compile_host_mxfp8_thd``, like ``host_f16_thd``)
    that runs the chain's THD setup with the kv-blocked workspace, derives the per-sequence SF TILE prefixes (``sf_meta``), stages
    the five hazard tensors' pads through the packed pre-pass, forms the packed delta from the bf16 ports, launches the main kernel
    with the metadata AND the SF prefixes, renders stage 3 through the THD helpers of BOTH policies, and bounds the GQA fold at the
    live kv total on device.  A host that reused the dense helper's call shape would read a dense SF layout off a packed buffer."""
    from pathlib import Path

    from cudnn.sdpa.bwd import prepared_sm107
    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host

    assert hasattr(prepared_sm107, "compile_plan_mxfp8_thd") and hasattr(prepared_host, "compile_host_mxfp8_thd")
    src = Path(prepared_host.__file__).read_text(encoding="utf-8")
    assert "def host_mxfp8_thd(" in src, "the MXFP8 row's THD host is a sibling artifact (host_mxfp8_thd), like host_f16_thd / host_fp8_thd"
    body = _def_body(_code_only(src), "host_mxfp8_thd")
    assert "thd_bwd_setup_host(" in body and "kv_blocked=True" in body, "the chain's THD setup over the kv-blocked workspace"
    assert "sf_meta" in body, "the per-sequence SF tile prefixes reach the main kernel and the block-scale GEMMs"
    assert "pad_sf_atoms_thd_host(" in body, "the packed SF pad pre-pass over the hazard tensors (the chain's own kernel, per sequence)"
    assert "dot_do_o_host(" in body, "the packed delta is the dot over the packed bf16 o_f16 / dO_f16 ports"
    assert "_stage3_thd(" in body and "_stage3_block_scale_thd(" in body, "both policies' THD stage-3 helpers"
    assert "dkv_reduce_bounded_host(" in body and "THD_CU_K_TOTAL_OFF" in body, "the GQA fold is bounded at the live kv total on device"
    assert re.search(r"\bmain\(", body), "host_mxfp8_thd launches the main kernel"
    for call in re.findall(r"\bmain\((.*?)\n\s*\)", body, re.S):  # the multi-line form black emits: one positional per line
        args = [a.strip() for a in re.split(r",[ \t]*\n", call.strip().strip(",")) if a.strip()]
        assert args[-1] in ("stream", "stream=stream"), args
        assert any("meta" in a for a in args), f"the THD main launch takes the metadata buffer: {args}"
        assert any("sf_meta" in a for a in args), f"the THD main launch takes the per-sequence SF tile prefixes: {args}"


@requires_sm80
def test_graph_thd_mxfp8_served_through_the_declared_totals_and_declined_without(monkeypatch):
    """The ragged MXFP8 backward graph on the row (the analyzer's cc faked to 10.7 off the Rubin line): with the packed totals
    declared it is SERVED by the row's own ``mismatch()`` (``thd`` and ``thd_declared_totals`` claimed) across the mask arms and
    GQA; WITHOUT them it is a typed decline naming ``max_total_seq_len`` -- a caller gets a decline it can act on, never a mis-sized
    workspace.  Right-band widening stays declined under THD."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)

    def reason(lens_q=(256, 128), lens_kv=(256, 128), *, h=2, hkv=None, declare_totals=True, **kw):
        case = _thd_mx_case(lens_q, lens_kv, h, hkv, device="cpu", oracle=False)
        g, _vp, _t = _build_thd_mx_graph(case, declare_totals=declare_totals, **kw)
        try:
            g.validate()
            g.build_operation_graph()
        except cudnn.cudnnGraphNotSupportedError as exc:
            return f"refused by the node: {exc}"
        facts = ga.analyze(g)
        assert facts is not None and facts.is_mxfp8 and facts.thd, facts
        return mismatch(_spec().capabilities, facts)

    without = reason(declare_totals=False)
    assert without is not None and "max_total_seq_len" in without, without
    assert reason() is None
    assert reason(use_causal_mask=True) is None
    assert reason(use_causal_mask_bottom_right=True) is None
    assert reason(h=4, hkv=2, use_causal_mask=True, left_bound=64) is None
    widened = reason(diagonal_band_right_bound=16)
    assert widened is not None and "right-band" in widened, widened


@requires_sm80
def test_graph_thd_mxfp8_backend_replay_takes_the_totals(monkeypatch):
    """Below ``mismatch()``: the python-native graph captures ``max_total_seq_len_q/kv`` on the MXFP8 backward node and the C++ replay
    forwards every captured kwarg to the native ``sdpa_mxfp8_backward`` binding, which declares them -- so plan creation on a ragged
    MXFP8 graph gets past the binding: served on the Rubin line, a typed ``cudnnGraphNotSupportedError`` elsewhere (never a bare
    ``TypeError``).  A stale extension built before the attribute is named by the skip."""
    from cudnn.sdpa import graph_analyzer as ga

    if not _binding_declares_totals():
        pytest.skip("the native sdpa_mxfp8_backward binding predates max_total_seq_len_q/kv -- rebuild the extension")
    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    case = _thd_mx_case((256, 128), (256, 128), 2, device="cuda", oracle=False)
    g, _vp, _t = _build_thd_mx_graph(case, declare_totals=True)
    g.validate()
    g.build_operation_graph()
    try:
        g.create_execution_plans([cudnn.heur_mode.A])
    except cudnn.cudnnGraphNotSupportedError:
        assert torch.cuda.get_device_capability() != _RUBIN_CC, "a Rubin device must offer the MXFP8 THD plan"
        return
    assert _plan_index(g, _ENGINE) is not None, [g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]
