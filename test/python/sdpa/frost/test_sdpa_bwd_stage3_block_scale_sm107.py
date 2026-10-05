# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The stage-3 GEMM's BLOCK-SCALE arm (``bprop_matmul_blackwell.py`` at ``MatmulTemplateParams.block_scale=True``): the
MXFP8 d=256 backward's dK / dQ over an e4m3 dS workspace whose 32-element K blocks carry E8M0 scales, dequantized in the
K64 ``tcgen05.mma.block_scale`` (kind MXF8F6F4, BLOCK32) against the columnwise-quantized Q / K payload and its
D-plane-major scale factors.

Host (any box whose cutlass-dsl knows ``sm_107a``):
* the record validation (block_scale is the fp8 (256, 256) row's arm with EPI_NONE and nothing else);
* the arm's derived geometry (SF ring bytes, TMEM columns 512 + 4 + 8 in the 576-column exclusive allocation, the SF rings
  declared ahead of the operand rings, every descriptor root below 256 KiB) and the default rendering's untouched layout;
* ``block_scale=False`` folds the arm out of the template, pinned on the template's SOURCE: the field is read once through
  ``getattr`` with the default, every traced read of it is an ``if cutlass.const_expr(_BLOCK_SCALE):`` guard and every piece of the
  arm's machinery sits under one, the TMEM allocation's ``is_exclusive`` is the folded module constant, the host passes ``None``
  for both scale-factor tensor maps, and a record WITHOUT the field renders the default record's module constants;
* the sm_107a SASS census of both block-scale renderings: ``UTCCP`` == 1 SFA + ``num_blocks_n`` SFB atoms per K stage,
  ``UTCQMMA`` == the stage's k-blocks, STL == LDL == 0, no ``MEMBAR.ALL.GPU`` / ``CGAERRBAR``.

Rubin (``requires_rubin``): random e4m3 payloads and random E8M0 atoms (bytes 124..131 -- both sides of 128 -- plus the
amax == 0 byte) laid out per the workspace contract, against an fp64 reference of the dequantized products under a
one-bf16-ulp bound (the accumulator is fp32, the output bf16); and the RED twin -- one corrupted scale byte must fail
the same comparison.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from frost_test_utils import arch_known_to_the_dsl, nvdisasm_candidates, requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl]

_SMEM_DESC_V0_LIMIT = 1 << 18
_RUBIN_OVERSIZED_SMEM = 327 * 1024
_SF_ATOM_BYTES = 512
_MX_BLOCK = 32


def _load_stage3(**params):
    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.bwd.api_dsl import _SM100_MATMUL_FILE, _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm100 import MatmulTemplateParams

    return load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), MatmulTemplateParams(**params), tag="test_stage3_block_scale")


def _bs_record(a_is_m_major: bool, **extra) -> dict:
    from cudnn.frost.tile_dsl.constants import DTYPE_E4M3
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_NONE

    rec = dict(
        a_is_m_major=a_is_m_major,
        b_is_n_major=True,
        causal_mode=CAUSAL_K_NONE,
        causal_gran=256,
        causal_shift=0,
        vec_bytes_epi=32,
        dtype_qkv=DTYPE_E4M3,
        cgrp_tile_mn=(256, 256),
        block_scale=True,
    )
    rec.update(extra)
    return rec


# --------------------------------------------------------------------------------------------------------- host: the record


def test_block_scale_records_are_validated():
    """block_scale is the fp8 (256, 256) row's arm: EPI_NONE (the MMA dequantizes), a bf16 / fp16 output, and a THD leg over the
    packed per-sequence scale-factor tiles (bottom-right spelled ``thd_causal_bottom_right`` on a trimmed mode, no constant shift)."""
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP16
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, EPI_DESCALE, EPI_QUANT, MatmulTemplateParams, matmul_out_dtype, validate_matmul_params

    for ok in (
        _bs_record(False),
        _bs_record(True),
        _bs_record(False, causal_mode=CAUSAL_K_LO),
        _bs_record(True, causal_mode=CAUSAL_K_HI, causal_shift=512),
        _bs_record(False, causal_mode=CAUSAL_K_LO, causal_window=640),
        _bs_record(True, dtype_out=DTYPE_FP16),
        _bs_record(True, dtype_out=DTYPE_BF16),
        _bs_record(False, thd_varlen=True),  # the THD leg (per-sequence SF tile prefixes read from the appended sf_meta operand)
        _bs_record(True, causal_mode=CAUSAL_K_HI, thd_varlen=True, thd_rows_kv=True, thd_causal_bottom_right=True),
    ):
        validate_matmul_params(MatmulTemplateParams(**ok))
    assert matmul_out_dtype(MatmulTemplateParams(**_bs_record(False))) == DTYPE_BF16, "the inherited output of the arm is bf16 (true-unit)"
    for bad, why in (
        (_bs_record(False, dtype_qkv=DTYPE_BF16), "block_scale is the MXFP8 arm of the fp8 rendering"),
        (_bs_record(False, epi_mode=EPI_DESCALE), "epilogue is EPI_NONE"),
        (_bs_record(True, epi_mode=EPI_QUANT, dtype_out=DTYPE_E4M3), "epilogue is EPI_NONE"),
        (_bs_record(False, cgrp_tile_mn=(512, 512)), "rendered at the (256, 256) row only"),
        (_bs_record(False, dtype_out=DTYPE_E4M3), "needs EPI_QUANT"),
        (_bs_record(True, b_head_group=2), "keeps b_head_group == 1"),  # the arm launches dQ per GQA group member (SFB indexed per A / C head)
    ):
        with pytest.raises(ValueError, match=re.escape(why)):
            validate_matmul_params(MatmulTemplateParams(**bad))
    # Append-only: the field defaults False and the pre-arm records are unchanged -- it follows every field that existed before
    # it, and only the THD fields appended after it (``thd_rows_kv``, ``thd_causal_bottom_right``) come later.
    assert MatmulTemplateParams().block_scale is False
    fields = list(MatmulTemplateParams.__dataclass_fields__)
    assert fields.index("block_scale") > fields.index("b_head_group")
    assert fields[fields.index("block_scale") + 1 :] == ["thd_rows_kv", "thd_causal_bottom_right"], fields


def test_block_scale_thd_leg_sf_prefix_contract_and_operand_refusals():
    """The THD leg's scale-factor contract has ONE spelling, in the config module: the int32 ``sf_meta`` region is
    ``[cu_sf_q(B+1) | cu_sf_k(B+1)]`` in units of 128-token TILES (``2 * (B + 1)`` words, the k prefixes at ``B + 1``), and the
    SFB view over a PACKED columnwise SF tensor is the per-(head, tile) plane-contiguous one (atom, plane, tile, head, batch 1; plane
    stride one atom, tile stride the ``planes * 512``-byte slab) -- the dense D-plane-major view reads plane 1 from an S-dependent
    wrong place.  And ``_require_epi_operands`` refuses the operand set that does not match the rendering: a block-scale THD
    rendering without ``sf_meta_t``, a dense block-scale rendering with it, a plain THD rendering (no block scale) with it."""
    from cudnn.sdpa.bwd.config_sm100 import (
        EPI_QUANT,
        STAGE3_THD_SF_CU_K_OFF,
        STAGE3_THD_SF_CU_Q_OFF,
        STAGE3_THD_SF_META_WORDS,
        stage3_thd_sfb_layout,
    )

    assert STAGE3_THD_SF_CU_Q_OFF == 0
    for b in (1, 2, 3, 33):
        assert STAGE3_THD_SF_META_WORDS(b) == 2 * (b + 1) and STAGE3_THD_SF_CU_K_OFF(b) == b + 1
    t, h = 7, 3
    assert stage3_thd_sfb_layout(2, t, h) == ((512, 2, t, h, 1), (1, 512, 1024, t * 1024, h * t * 1024))
    assert stage3_thd_sfb_layout(1, t, h) == ((512, 1, t, h, 1), (1, 512, 512, t * 512, h * t * 512))

    some = object()
    thd_bs = _load_stage3(**_bs_record(False, thd_varlen=True))
    assert thd_bs._BLOCK_SCALE and thd_bs._THD_MM
    with pytest.raises(TypeError, match="sf_meta_t"):
        thd_bs._require_epi_operands(None, None, None, some, some, None)
    thd_bs._require_epi_operands(None, None, None, some, some, some)  # the complete THD block-scale operand set
    dense_bs = _load_stage3(**_bs_record(False))
    assert dense_bs._BLOCK_SCALE and not dense_bs._THD_MM
    with pytest.raises(TypeError, match="is the THD leg's operand"):
        dense_bs._require_epi_operands(None, None, None, some, some, some)
    dense_bs._require_epi_operands(None, None, None, some, some, None)
    plain_thd = _load_stage3(**_bs_record(False, block_scale=False, epi_mode=EPI_QUANT, thd_varlen=True, thd_rows_kv=True))
    assert not plain_thd._BLOCK_SCALE and plain_thd._THD_MM
    with pytest.raises(TypeError, match="does not block-scale"):
        plain_thd._require_epi_operands(some, some, some, None, None, some)
    plain_thd._require_epi_operands(some, some, some, None, None, None)


@pytest.mark.parametrize("a_is_m_major", (False, True), ids=("dK-Kmajor", "dQ-Mmajor"))
def test_block_scale_renderings_derive_their_sf_geometry(a_is_m_major):
    """Every SF constant of the arm is DERIVED from the tile row: one F8_128x4 atom per 128-B K stage per 128-row block
    (SFA 512 B / 4 TMEM columns, SFB num_blocks_n atoms = 1024 B / 8 columns, one UTCCP'd atom per num_k_blocks K64 MMAs),
    the SF TMEM past the two accumulator stages inside the 576-column exclusive allocation, the SF rings declared FIRST
    and every tcgen05 descriptor root below the version-0 line."""
    mod = _load_stage3(**_bs_record(a_is_m_major))
    assert mod._BLOCK_SCALE and mod._IS_FP8
    assert (mod.num_tmem_alloc_cols, mod.tmem_alloc_exclusive) == (576, True)
    assert mod.mma_inst_shape_mnk == (256, 256, 64) and mod.mma_k_dim == 1 and mod.mma_size_k == 2
    assert mod.sfa_smem_bytes == mod.cta_tile_mnk[0] // 128 * _SF_ATOM_BYTES == 512
    assert mod.num_blocks_n == mod.mma_inst_shape_mnk[1] // 128 == 2
    assert mod.sfb_smem_bytes == mod.num_blocks_n * _SF_ATOM_BYTES == 1024
    assert mod.sf_scales_per_inst == mod.mma_inst_shape_mnk[2] // _MX_BLOCK == 2
    assert mod.sf_insts_per_atom == mod.mma_size_k == 2, "one UTCCP'd atom (4 scales) serves the stage's two K64 MMAs (2 scales each)"
    assert (mod.sfa_tmem_cols, mod.sfb_tmem_cols) == (4, 8)
    assert mod.sfa_col_base == mod.acc_stages * 256 == 512 and mod.sfb_col_base == 516
    assert mod.sfb_col_base + mod.sfb_tmem_cols <= mod.num_tmem_alloc_cols
    layout = mod._smem_layout_bytes()
    print(f"\nblock-scale a_is_m_major={a_is_m_major}: SMEM layout {layout}")
    assert layout["smem_sfa_0"] < layout["smem_sfb_0"] < layout["smem_a_0"] < layout["smem_b_0"] < layout["smem_d"], "SF rings first, then A, B, staging"
    assert layout["smem_sfb_0"] - layout["smem_sfa_0"] == mod.ab_stages * mod.sfa_smem_bytes
    assert layout["smem_a_0"] - layout["smem_sfb_0"] == mod.ab_stages * mod.sfb_smem_bytes
    for key in ("smem_sfa_0", "smem_sfb_0", "smem_a_0", "smem_b_0"):
        assert layout[key] < _SMEM_DESC_V0_LIMIT, (key, layout)
    assert layout["total"] == 231424 + mod.ab_stages * (mod.sfa_smem_bytes + mod.sfb_smem_bytes) == 240640, "the fp8 arm's 226 KiB + the two SF rings"
    assert layout["total"] <= _RUBIN_OVERSIZED_SMEM


@pytest.mark.parametrize("a_is_m_major", (False, True), ids=("dK-Kmajor", "dQ-Mmajor"))
def test_block_scale_false_keeps_the_fp8_arm_untouched(a_is_m_major):
    """The default record renders the constants it always did: 512 non-exclusive TMEM columns, no SF ring in the layout, the
    226 KiB SMEM total, the dense-fp8 MMA kind."""
    mod = _load_stage3(**_bs_record(a_is_m_major, block_scale=False, epi_mode=1))
    assert not mod._BLOCK_SCALE
    assert (mod.num_tmem_alloc_cols, mod.tmem_alloc_exclusive) == (512, False)
    layout = mod._smem_layout_bytes()
    assert "smem_sfa_0" not in layout and "smem_sfb_0" not in layout
    assert layout["total"] == 231424
    assert mod.sfa_smem_bytes == 0 and mod.sfb_smem_bytes == 0


# ------------------------------------------------------- host: the default record folds the arm out of the template (its SOURCE)


def test_block_scale_default_folds_out_of_the_template():
    """``MatmulTemplateParams.block_scale`` at its default False renders what the template rendered before the field existed.  The
    rule, pinned on the template's SOURCE (no GPU; the shape of the ``b_head_group`` pin in ``test_sdpa_bwd_dsl_sm107``):

    * the field is read ONCE, at module level, through ``getattr(PARAMS, "block_scale", False)`` -- a record built before the field
      existed takes the default -- and never as a ``PARAMS.block_scale`` attribute;
    * every read of ``_BLOCK_SCALE`` inside a TRACED function (``@cute.kernel`` / ``@cute.jit``) is the test of an
      ``if cutlass.const_expr(_BLOCK_SCALE):`` -- a folded branch, never a staged ``if`` on the Python bool, which would trace BOTH
      arms; every other read is plain Python at template-load time: three ternaries whose default arm is the pre-arm literal (512
      TMEM columns, 0 SF ring bytes), ``tmem_alloc_exclusive = _BLOCK_SCALE``, the arm's validator, the SMEM layout's SF rows, the
      descriptor-reach check and the operand check of ``_require_epi_operands`` -- the census below names each;
    * every use of the arm's machinery inside a traced function -- the SF SMEM rings and their byte counts, the SF tensor maps'
      prefetch / TMA loads, the UTCCP source descriptors and ``tcgen05_cp`` sites, the Mx instruction descriptors and the
      ``_tcgen05_mma_block_scale`` calls, the SF TMEM pointers, the host's two SF tensor-map builds -- sits INSIDE one of those
      guard bodies (the bare ``tma_sf*_desc_0`` names are the launch arguments, ``sfa_0`` / ``sfb_0`` also reach the plain-Python
      operand check);
    * the TMEM allocation's ``is_exclusive`` is the module constant ``tmem_alloc_exclusive`` at its one alloc and two dealloc
      sites, next to ``cutlass.Int32(num_tmem_alloc_cols)``, and the ``tile_helpers`` wrappers emit the kwarg on their
      ``if is_exclusive:`` branch only -- the default allocation traces the pre-arm ``tcgen05.alloc`` of 512 columns;
    * the kernel's two SF tensor-map parameters are the LAST two, None-specialized away: the host's ``else`` arm passes ``None``.

    And a record WITHOUT the field (the dataclass minus ``block_scale``) renders the same module constants as the default record,
    on the bf16 and on the fp8 DESCALE (256, 256) rows.  The True arm is proven by the SASS census and the Rubin cells below."""
    import ast
    import dataclasses
    import inspect

    from cudnn.frost.template_loader import DIGEST_GLOBAL, PARAMS_GLOBAL, load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.gemm.frost import tile_helpers
    from cudnn.sdpa.bwd.api_dsl import _SM100_MATMUL_FILE, _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_NONE, EPI_DESCALE, MatmulTemplateParams

    path = _sm100_kernel_path(_SM100_MATMUL_FILE)
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def ancestors(node):
        while node in parents:
            node = parents[node]
            yield node

    def is_traced(fn):
        return any(
            isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name) and d.value.id == "cute" and d.attr in ("kernel", "jit") for d in fn.decorator_list
        )

    def traced_function(node):
        """The ``@cute.kernel`` / ``@cute.jit`` function the node sits in (its name), else None: module level or plain Python."""
        return next((a.name for a in ancestors(node) if isinstance(a, ast.FunctionDef) and is_traced(a)), None)

    def is_guard(node):
        """``cutlass.const_expr(_BLOCK_SCALE)``."""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "const_expr"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cutlass"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "_BLOCK_SCALE"
        )

    def loads(name):
        return [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]

    # (1) the field is read ONCE, at module level, through getattr with the default
    assert len(re.findall(r'^_BLOCK_SCALE = bool\(getattr\(PARAMS, "block_scale", False\)\)$', src, flags=re.M)) == 1
    assert sum(1 for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "block_scale") == 1, "the field is read through ONE getattr"
    assert not any(
        isinstance(n, ast.Attribute) and n.attr == "block_scale" for n in ast.walk(tree)
    ), "no PARAMS.block_scale read: a pre-field record takes the default"

    # (2) every read of _BLOCK_SCALE: a const_expr guard inside a traced function, plain Python at template load elsewhere
    def plain_form(n):
        p = parents[n]
        if isinstance(p, ast.IfExp) and p.test is n and isinstance(parents[p], ast.Assign):
            return f"ternary -> {ast.unparse(parents[p].targets[0])} else {ast.unparse(p.orelse)}"
        if isinstance(p, ast.Assign):
            return f"assign -> {ast.unparse(p.targets[0])}"
        if isinstance(p, ast.If) and p.test is n:
            return "if _BLOCK_SCALE"
        if isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.Not) and isinstance(parents[p], ast.BoolOp) and isinstance(parents[p].op, ast.And):
            return "if (not _BLOCK_SCALE and ...)"
        if isinstance(p, ast.BoolOp) and isinstance(p.op, ast.And):
            return "if (_BLOCK_SCALE and ...)"
        return f"UNCLASSIFIED {type(p).__name__} at line {n.lineno}"

    guards, plain = [], []
    for n in loads("_BLOCK_SCALE"):
        fn = traced_function(n)
        if fn is not None:
            p = parents[n]
            assert (
                is_guard(p) and isinstance(parents[p], ast.If) and parents[p].test is p
            ), f"line {n.lineno} ({fn}): a traced read of _BLOCK_SCALE outside an `if cutlass.const_expr(_BLOCK_SCALE):` guard"
            guards.append(parents[p])
        else:
            where = next((a.name for a in ancestors(n) if isinstance(a, ast.FunctionDef)), "<module>")
            plain.append((where, plain_form(n)))
    census = sorted(plain)
    print(f"\nblock_scale: plain-Python reads at template load {census}; const_expr guards at lines {sorted(g.lineno for g in guards)}")
    assert census == sorted(
        [
            ("<module>", "ternary -> num_tmem_alloc_cols else 512"),  # the pre-arm 512-column allocation
            ("<module>", "assign -> tmem_alloc_exclusive"),  # the non-exclusive allocation
            ("<module>", "ternary -> sfa_smem_bytes else 0"),  # no SF rings
            ("<module>", "ternary -> sfb_smem_bytes else 0"),
            ("<module>", "if _BLOCK_SCALE"),  # the arm's own validator
            ("<module>", "if (_BLOCK_SCALE and ...)"),  # the version-0 descriptor reach of the SF ring roots
            ("_smem_layout_bytes", "if _BLOCK_SCALE"),  # the SF rows of the SMEM layout
            ("_require_epi_operands", "if (_BLOCK_SCALE and ...)"),  # block_scale needs sfa_0 / sfb_0 ...
            ("_require_epi_operands", "if (not _BLOCK_SCALE and ...)"),  # ... and refuses them otherwise
        ]
    ), f"a read of block_scale outside the const_expr fold reached the template: {census}"
    assert len(guards) == src.count("if cutlass.const_expr(_BLOCK_SCALE):") >= 2, "every guard is a plain `if cutlass.const_expr(_BLOCK_SCALE):` statement"
    assert {traced_function(g) for g in guards} == {"_bprop_matmul_bh_sm100_kernel", "_host"}

    # (3) every use of the arm's machinery inside a traced function sits inside one of those guard bodies
    spans = [(g.body[0].lineno, g.body[-1].end_lineno) for g in guards]

    def guarded(node):
        return any(lo <= node.lineno and node.end_lineno <= hi for lo, hi in spans)

    machinery = {
        "smem_sfa", "smem_sfb", "sfa_smem_bytes", "sfb_smem_bytes", "num_blocks_n", "sf_scales_per_inst", "sf_scale_format",  # rings, atoms
        "desc_sfa_root", "desc_sfb_root", "desc_sfa_stage", "desc_sfb_stage", "_s2t_shape", "_s2t_multicast",  # UTCCP descriptors / shape
        "idesc_by_k", "mma_block_scale_kind", "scale_vec_size", "_tcgen05_mma_block_scale",  # the block-scale idesc / MMA
        "sfa_tmem_base", "sfb_tmem_base", "sfa_ptr", "sfb_ptr", "sfb_block_ptrs", "sfa_col_base", "sfb_col_base", "sfa_tmem_cols", "sfb_tmem_cols",  # SF TMEM
    }  # fmt: skip
    machinery_attrs = {"tcgen05_cp", "Tcgen05MxInstrDesc"}  # nvvm.tcgen05_cp(...), ...Tcgen05MxInstrDesc.build(...)
    seen = set()
    for n in ast.walk(tree):
        fn = traced_function(n)
        if fn is None:
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            p = parents[n]
            if n.id in machinery:
                assert guarded(n), f"line {n.lineno} ({fn}): {n.id} used outside a const_expr(_BLOCK_SCALE) body"
            elif n.id in ("tma_sfa_desc_0", "tma_sfb_desc_0"):
                if isinstance(p, ast.Attribute) and p.attr == "get_ptr":  # the descriptor's use: prefetch, TMA load
                    assert guarded(n), f"line {n.lineno} ({fn}): {n.id}.get_ptr() outside a const_expr(_BLOCK_SCALE) body"
                else:  # the bare name is the kernel launch argument (None at the default)
                    assert fn == "_host" and isinstance(p, ast.Call), f"line {n.lineno} ({fn}): {n.id} read outside its guard and not as the launch argument"
            elif n.id in ("sfa_0", "sfb_0"):
                if not (isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id == "_require_epi_operands"):
                    assert guarded(n), f"line {n.lineno} ({fn}): {n.id} used outside a const_expr(_BLOCK_SCALE) body"
            else:
                continue
            seen.add(n.id)
        elif isinstance(n, ast.Attribute) and n.attr in machinery_attrs:
            assert guarded(n), f"line {n.lineno} ({fn}): {n.attr} outside a const_expr(_BLOCK_SCALE) body"
            seen.add(n.attr)
    must_see = {
        "smem_sfa",
        "smem_sfb",
        "tcgen05_cp",
        "Tcgen05MxInstrDesc",
        "_tcgen05_mma_block_scale",
        "sfa_ptr",
        "sfb_ptr",
        "tma_sfa_desc_0",
        "tma_sfb_desc_0",
        "sfa_0",
        "sfb_0",
    }
    assert must_see <= seen, f"the census missed part of the arm's machinery: {sorted(must_see - seen)}"

    # (4) the TMEM allocation: the folded module constant at the one alloc + two dealloc sites; the wrappers trace the kwarg on True only
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("_tcgen05_alloc", "_tcgen05_dealloc")]
    assert sorted(c.func.id for c in calls) == ["_tcgen05_alloc", "_tcgen05_dealloc", "_tcgen05_dealloc"]
    for c in calls:
        assert {k.arg: ast.unparse(k.value) for k in c.keywords}.get("is_exclusive") == "tmem_alloc_exclusive", ast.unparse(c)
        assert any(ast.unparse(a) == "cutlass.Int32(num_tmem_alloc_cols)" for a in c.args), ast.unparse(c)
    assert (
        len([n for n in loads("tmem_alloc_exclusive") if traced_function(n)]) == 3
    ), "tmem_alloc_exclusive reaches the trace at the three alloc / dealloc sites only"
    stores = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "tmem_alloc_exclusive" for t in n.targets)]
    assert len(stores) == 1 and ast.unparse(stores[0].value) == "_BLOCK_SCALE"
    for helper in (tile_helpers.tcgen05_alloc, tile_helpers.tcgen05_dealloc):
        (branch,) = ast.parse(textwrap.dedent(inspect.getsource(helper))).body[0].body
        assert isinstance(branch, ast.If) and ast.unparse(branch.test) == "is_exclusive", inspect.getsource(helper)
        on, off = branch.body[0].value, branch.orelse[0].value
        assert {k.arg: ast.unparse(k.value) for k in on.keywords}.get("is_exclusive") == "True"
        assert "is_exclusive" not in {k.arg for k in off.keywords}, "the default allocation must not trace the kwarg (the pre-arm form, the 4.7.0 floor)"

    # (5) the two SF tensor maps are the kernel's LAST parameters (the THD leg's SF tile-prefix operand sits right ahead of them, and
    # is _host's LAST, defaulted parameter) and the host's else arm passes None for both maps
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_bprop_matmul_bh_sm100_kernel")
    assert [a.arg for a in kernel.args.args][-3:] == ["sf_meta_t", "tma_sfa_desc_0", "tma_sfb_desc_0"], "appended last (the kernel ABI is append-only)"
    host_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_host")
    assert host_fn.args.args[-1].arg == "sf_meta_t" and ast.unparse(host_fn.args.defaults[-1]) == "None", "sf_meta_t is _host's last, defaulted parameter"
    (host_guard,) = [g for g in guards if traced_function(g) == "_host"]
    assert [ast.unparse(s) for s in host_guard.orelse] == ["tma_sfa_desc_0 = None", "tma_sfb_desc_0 = None"], ast.unparse(host_guard)

    # (6) a record that never carried the field renders the same module constants as the default record
    records = {
        "bf16": MatmulTemplateParams(
            a_is_m_major=True, causal_mode=CAUSAL_K_NONE, causal_gran=256, causal_shift=0, vec_bytes_epi=32, dtype_qkv=DTYPE_BF16, cgrp_tile_mn=(256, 256)
        ),
        "fp8-descale": MatmulTemplateParams(**_bs_record(False, block_scale=False, epi_mode=EPI_DESCALE)),
    }
    legacy_fields = [(f.name, f.type, dataclasses.field(default=f.default)) for f in dataclasses.fields(MatmulTemplateParams) if f.name != "block_scale"]
    Legacy = dataclasses.make_dataclass("MatmulTemplateParamsBeforeBlockScale", legacy_fields, frozen=True)

    def consts(mod):
        return {
            k: v
            for k, v in vars(mod).items()
            if not k.startswith("__") and k not in (PARAMS_GLOBAL, DIGEST_GLOBAL) and isinstance(v, (bool, int, float, str, tuple, frozenset))
        }

    for name, record in records.items():
        legacy = Legacy(**{n: getattr(record, n) for n, _, _ in legacy_fields})
        assert not hasattr(legacy, "block_scale")
        default_mod, legacy_mod = (load_template(path, p, tag="test_stage3_block_scale_default_fold") for p in (record, legacy))
        assert default_mod._BLOCK_SCALE is False and legacy_mod._BLOCK_SCALE is False
        assert (default_mod.num_tmem_alloc_cols, default_mod.tmem_alloc_exclusive, default_mod.sfa_smem_bytes, default_mod.sfb_smem_bytes) == (512, False, 0, 0)
        assert consts(default_mod) == consts(legacy_mod), f"{name}: the default record and the record without the field render different module constants"


# ------------------------------------------------------------------------------ host: the sm_107a trace-compile probe

# One stage-3 rendering, trace-compiled for `arch` in a fresh interpreter: the operands, the fp8 arm's four epilogue scalars
# and the block-scale arm's two SF views (laid out per the workspace contract) are pointer arguments, exactly as the prepared
# hosts pass them.  Prints PTX_MD5 / CUBIN_MD5, PTX opcode counts, and -- with an nvdisasm that decodes the arch -- SASS counts.
_PROBE = textwrap.dedent(r"""
    import glob, hashlib, json, os, subprocess, sys
    dump, arch, params_json, cands = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_ARCH"] = arch
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    import cutlass
    import cutlass.cute as cute
    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.bwd.api_dsl import _SM100_MATMUL_FILE, _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm100 import EPI_NONE, MatmulTemplateParams, matmul_out_dtype
    kw = json.loads(params_json)
    if "cgrp_tile_mn" in kw:
        kw["cgrp_tile_mn"] = tuple(kw["cgrp_tile_mn"])
    params = MatmulTemplateParams(**kw)
    mod = load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), params, tag="stage3_probe")
    _DSL = {0: cutlass.Float8E4M3FN, 2: cutlass.BFloat16, 3: cutlass.Float16}
    io, out = _DSL[int(params.dtype_qkv)], _DSL[matmul_out_dtype(params)]
    epi = int(getattr(params, "epi_mode", EPI_NONE)) != EPI_NONE
    bs = bool(getattr(params, "block_scale", False))
    if bs:
        print("EXPECT_UTCCP", 1 + mod.num_blocks_n)
        print("EXPECT_BSMMA", mod.mma_size_k)
    S_Q, S_KV, H, B, D = 1024, 1024, 8, 1, 256
    a_m_major = bool(params.a_is_m_major)
    KT, MT = (S_KV // 128, S_Q // 128) if a_m_major else (S_Q // 128, S_KV // 128)
    PL = D // 128

    @cute.jit
    def probe(entry: cutlass.Constexpr, a_ptr: cute.Pointer, b_ptr: cute.Pointer, c_ptr: cute.Pointer, meta_ptr: cute.Pointer, desc_ptr: cute.Pointer,
              f_ptr: cute.Pointer, sfa_ptr: cute.Pointer, sfb_ptr: cute.Pointer, stream):
        if cutlass.const_expr(a_m_major):
            a = cute.make_tensor(a_ptr, cute.make_layout((S_Q, S_KV, H, B), stride=(1, S_Q, S_KV * S_Q, H * S_KV * S_Q)))
            b = cute.make_tensor(b_ptr, cute.make_layout((D, S_KV, H, B), stride=(1, H * D, D, S_KV * H * D)))
            c = cute.make_tensor(c_ptr, cute.make_layout((S_Q, D, H, B), stride=(H * D, 1, D, S_Q * H * D)))
        else:
            a = cute.make_tensor(a_ptr, cute.make_layout((S_KV, S_Q, H, B), stride=(S_Q, 1, S_KV * S_Q, H * S_KV * S_Q)))
            b = cute.make_tensor(b_ptr, cute.make_layout((D, S_Q, H, B), stride=(1, H * D, D, S_Q * H * D)))
            c = cute.make_tensor(c_ptr, cute.make_layout((S_KV, D, H, B), stride=(H * D, 1, D, S_KV * H * D)))
        meta = cute.make_tensor(meta_ptr, cute.make_layout((1,), stride=(1,)))
        desc = cute.make_tensor(desc_ptr, cute.make_layout((1,), stride=(1,)))
        problem = tuple(cutlass.Int64(x) for x in (a.shape[0], b.shape[0], a.shape[1], H, B, *a.stride, *b.stride, *c.stride, b.shape[1], a.shape[0], c.shape[0]))
        if cutlass.const_expr(bs):
            sfa = cute.make_tensor(sfa_ptr, cute.make_layout((512, KT, MT, H, B), stride=(1, 512, KT * 512, MT * KT * 512, H * MT * KT * 512)))
            sfb = cute.make_tensor(sfb_ptr, cute.make_layout((512, PL, KT, H, B), stride=(1, B * H * KT * 512, 512, KT * 512, H * KT * 512)))
            entry(problem, a, b, c, meta, desc, stream, None, None, None, None, sfa, sfb)
        elif cutlass.const_expr(epi):
            f = cute.make_tensor(f_ptr, cute.make_layout((1,), stride=(1,)))
            entry(problem, a, b, c, meta, desc, stream, f, f, f, f)
        else:
            entry(problem, a, b, c, meta, desc, stream)

    def ptr(t, align=16):
        return cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=align)

    cute.compile(probe, mod._host, ptr(io), ptr(io), ptr(out), ptr(cutlass.Int32, 4), ptr(cutlass.Int64, 8), ptr(cutlass.Float32, 4), ptr(cutlass.Uint8), ptr(cutlass.Uint8),
                 cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False), options=f"--enable-tvm-ffi --gpu-arch {arch}")
    cubins = sorted(glob.glob(os.path.join(dump, "*.cubin")), key=os.path.getmtime)
    ptxs = sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)
    if not cubins or not ptxs:
        print("FAIL no cubin / ptx dumped into", dump, os.listdir(dump)); sys.exit(3)
    print("PTX_MD5", hashlib.md5(open(ptxs[-1], "rb").read()).hexdigest())
    print("CUBIN_MD5", hashlib.md5(open(cubins[-1], "rb").read()).hexdigest())
    ptx = open(ptxs[-1]).read()
    print("PTX TCGEN05_CP", ptx.count("tcgen05.cp"))
    print("PTX BLOCK_SCALE_MMA", ptx.count("kind::mxf8f6f4"))
    nvd = None
    for c in cands:
        try:
            proc = subprocess.run([c, "-c", cubins[-1]], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            print("REJECT", c, "->", repr(exc)); continue
        if proc.returncode == 0 and proc.stdout.strip():
            nvd = c; print("NVDISASM", c); break
        print("REJECT", c, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if nvd is None:
        print("SKIP no nvdisasm candidate decodes the cubin"); sys.exit(0)
    sass = subprocess.run([nvd, "-c", cubins[-1]], capture_output=True, text=True, check=True).stdout.splitlines()
    def cnt(*subs):
        return sum(1 for ln in sass if all(sb in ln for sb in subs))
    for key, subs in {"STL": ("STL",), "LDL": ("LDL",), "MEMBAR_GPU": ("MEMBAR.ALL.GPU",), "CGAERRBAR": ("CGAERRBAR",), "UTCCP": ("UTCCP",),
                      "UTCQMMA": ("UTCQMMA",), "UTCHMMA": ("UTCHMMA",)}.items():
        print("SASS", key, cnt(*subs))
    print("SASS LINES", len(sass))
    """)


def _run_probe(tmp_path, arch: str, tag: str, params: dict, timeout: int = 900) -> dict:
    """Trace-compile one rendering for ``arch`` in a fresh interpreter; returns the probe's ``NAME value`` lines as a dict."""
    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"this cutlass-dsl has no {arch}")
    dump = tmp_path / f"{arch}_{tag}_{hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]}"
    dump.mkdir()
    script = dump / "probe.py"
    script.write_text(_PROBE)
    argv = [sys.executable, str(script), str(dump), arch, json.dumps(params), *nvdisasm_candidates()]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, f"{arch} trace-compile of {tag} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = {}
    for ln in proc.stdout.splitlines():
        parts = ln.split()
        if len(parts) == 2 and parts[0] in ("PTX_MD5", "CUBIN_MD5", "NVDISASM", "EXPECT_UTCCP", "EXPECT_BSMMA"):
            out[parts[0]] = parts[1]
        elif len(parts) == 3 and parts[0] in ("PTX", "SASS") and parts[2].lstrip("-").isdigit():
            out[f"{parts[0]}_{parts[1]}"] = int(parts[2])
        elif ln.startswith(("SKIP", "REJECT")):
            out.setdefault("SKIP", []).append(ln)
    return out


@pytest.mark.parametrize("a_is_m_major", (False, True), ids=("dK-Kmajor", "dQ-Mmajor"))
def test_block_scale_renderings_sass_census(tmp_path, a_is_m_major):
    """sm_107a trace-compile of the arm: per K stage one UTCCP per SF atom (1 SFA + num_blocks_n SFB) and one block-scale MMA
    (``UTCQMMA``) per k-block, no dense-fp8 MMA, no spills, no cluster-scope drains."""
    out = _run_probe(tmp_path, "sm_107a", f"block_scale_{'dq' if a_is_m_major else 'dk'}", _bs_record(a_is_m_major, cgrp_tile_mn=[256, 256]))
    print(f"\nblock-scale a_is_m_major={a_is_m_major}: {out}")
    utccp, bsmma = int(out["EXPECT_UTCCP"]), int(out["EXPECT_BSMMA"])
    assert (utccp, bsmma) == (3, 2)
    assert out["PTX_TCGEN05_CP"] == utccp and out["PTX_BLOCK_SCALE_MMA"] == bsmma
    if "SKIP" in out or "SASS_UTCCP" not in out:
        pytest.skip(f"compiled + PTX pinned; SASS half skipped: {out.get('SKIP')}")
    assert out["SASS_UTCCP"] == utccp, out
    assert out["SASS_UTCQMMA"] == bsmma and out["SASS_UTCHMMA"] == 0, out
    assert out["SASS_STL"] == 0 and out["SASS_LDL"] == 0, out
    assert out["SASS_MEMBAR_GPU"] == 0 and out["SASS_CGAERRBAR"] == 0, out


# ---------------------------------------------------------------------------------------------------- Rubin: numerics


def _f8_128x4_atoms(sf: torch.Tensor) -> torch.Tensor:
    """Logical scale matrix ``[..., R, C]`` (uint8, R % 128 == 0, C % 4 == 0) -> its F8_128x4 atoms ``[..., R/128, C/4, 512]``:
    scale (r, c) at byte ``(r % 32) * 16 + (r // 32) * 4 + c % 4`` of atom (r // 128, c // 4) -- the quantizer's rule."""
    *lead, R, C = sf.shape
    n = len(lead)
    v = sf.reshape(*lead, R // 128, 4, 32, C // 4, 4)  # (rt, rg, rr, ct, cc)
    v = v.permute(*range(n), n + 0, n + 3, n + 2, n + 1, n + 4)  # (rt, ct, rr, rg, cc)
    return v.contiguous().reshape(*lead, R // 128, C // 4, 512)


def _e8m0_bytes(shape, gen, dev) -> torch.Tensor:
    """Random E8M0 bytes 124..131 (scales 1/8 .. 16 -- bytes on both sides of 128, so a signed decode would be caught) with
    ~3 % of the blocks at the amax == 0 byte (scale 2^-127: the block contributes nothing)."""
    e = torch.randint(124, 132, shape, generator=gen).to(torch.uint8)
    zero = torch.rand(shape, generator=gen) < 0.03
    return torch.where(zero, torch.zeros_like(e), e).to(dev)


def _dequant(payload_u8: torch.Tensor, sf_log: torch.Tensor) -> torch.Tensor:
    """fp64 ``payload * 2^(e - 127)`` with the E8M0 byte broadcast over its 32-element K block (last dim)."""
    x = payload_u8.view(torch.float8_e4m3fn).double()
    e = sf_log.double().repeat_interleave(_MX_BLOCK, dim=-1) - 127.0
    return x * torch.pow(torch.tensor(2.0, dtype=torch.float64, device=x.device), e)


class _Case:
    """One block-scale GEMM problem laid out per the workspace contract.

    dK (a_is_m_major False): out[b, h, kv, d] = sum_q  ds[b, h, kv, q] * q_T[b, q, h, d]   A K-major,  SFA atoms [B, H, S_kv/128, S_q/128, 512]
    dQ (a_is_m_major True):  out[b, h, q, d]  = sum_kv ds[b, h, kv, q] * k_T[b, kv, h, d]  A M-major,  SFA atoms [B, H, S_q/128, S_kv/128, 512]
    B is the columnwise-quantized payload [B, S_K, H, D]; its SF is D-plane-major: [planes, B, H, S_K/128, 512].
    """

    def __init__(self, a_is_m_major: bool, bsz: int, h: int, s_q: int, s_kv: int, d: int = 256, seed: int = 11):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        dev = torch.device("cuda")
        self.a_is_m_major = a_is_m_major
        M, K = (s_q, s_kv) if a_is_m_major else (s_kv, s_q)
        self.bsz, self.h, self.s_q, self.s_kv, self.d, self.M, self.K = bsz, h, s_q, s_kv, d, M, K
        # e4m3 payloads as their bytes (random normal-ish values around 0.5 magnitude, like the GEMM suite's inputs)
        ds_f = torch.randn(bsz, h, s_kv, s_q, generator=gen) * 0.5
        self.ds_u8 = ds_f.to(torch.float8_e4m3fn).view(torch.uint8).to(dev).contiguous()  # [B, H, S_kv, S_q]
        pay_f = torch.randn(bsz, K, h, d, generator=gen) * 0.5
        self.pay_u8 = pay_f.to(torch.float8_e4m3fn).view(torch.uint8).to(dev).contiguous()  # [B, S_K, H, D]
        # logical scales: A [B, H, M, K/32]; B [B, H, D, K/32] (per d column, per 32 K)
        self.sfa_log = _e8m0_bytes((bsz, h, M, K // _MX_BLOCK), gen, dev)
        self.sfb_log = _e8m0_bytes((bsz, h, d, K // _MX_BLOCK), gen, dev)
        self.sfa_atoms = _f8_128x4_atoms(self.sfa_log).contiguous()  # [B, H, M/128, K/128, 512]
        self.sfb_atoms = _f8_128x4_atoms(self.sfb_log).permute(2, 0, 1, 3, 4).contiguous()  # [planes, B, H, K/128, 512]
        self.out = torch.full((bsz, M, h, d), float("nan"), dtype=torch.bfloat16, device=dev)  # [B, S_M, H, D]

    def views(self):
        """The template's operand orders: A (M, K, H, B), B (N, K, H, B), C (M, N, H, B), SFA (512, Kt, Mt, H, B), SFB (512, planes, Kt, H, B)."""
        if self.a_is_m_major:
            a = self.ds_u8.permute(3, 2, 1, 0)  # ds[b, h, kv, q] read as [M = q, K = kv]: M contiguous
        else:
            a = self.ds_u8.permute(2, 3, 1, 0)  # ds[b, h, kv, q] as [M = kv, K = q]: K contiguous
        b = self.pay_u8.permute(3, 1, 2, 0)
        c = self.out.permute(1, 3, 2, 0)
        sfa = self.sfa_atoms.permute(4, 3, 2, 1, 0)
        sfb = self.sfb_atoms.permute(4, 0, 3, 2, 1)
        return a, b, c, sfa, sfb

    def reference(self) -> torch.Tensor:
        """fp64 ``sum_k A_deq[b, h, m, k] * B_deq[b, h, n, k]`` -> [B, M, H, D] like ``out``."""
        if self.a_is_m_major:
            a = self.ds_u8.permute(0, 1, 3, 2)  # [B, H, M = q, K = kv]
        else:
            a = self.ds_u8  # [B, H, M = kv, K = q]
        a_deq = _dequant(a, self.sfa_log)  # [B, H, M, K]
        b_deq = _dequant(self.pay_u8.permute(0, 2, 3, 1), self.sfb_log)  # [B, H, D, K]
        ref = torch.einsum("bhmk,bhnk->bhmn", a_deq, b_deq)  # [B, H, M, D]
        return ref.permute(0, 2, 1, 3).contiguous()


_LAUNCHERS: dict = {}


def _launcher(a_is_m_major: bool):
    """The compiled ``_host`` wrapper of one block-scale rendering (compiled once per process)."""
    if a_is_m_major in _LAUNCHERS:
        return _LAUNCHERS[a_is_m_major]
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute

    mod = _load_stage3(**_bs_record(a_is_m_major))

    @cute.jit
    def launch(
        a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, meta: cute.Tensor, desc: cute.Tensor, sfa: cute.Tensor, sfb: cute.Tensor, stream: cuda_driver.CUstream
    ):
        problem = tuple(
            cutlass.Int64(x)
            for x in (a.shape[0], b.shape[0], a.shape[1], a.shape[2], a.shape[3], *a.stride, *b.stride, *c.stride, b.shape[1], a.shape[0], c.shape[0])
        )
        mod._host(problem, a, b, c, meta, desc, stream, None, None, None, None, sfa, sfb)

    _LAUNCHERS[a_is_m_major] = launch
    return launch


def _run(case: _Case, sfa_atoms=None) -> torch.Tensor:
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    a, b, c, sfa, sfb = case.views()
    if sfa_atoms is not None:
        sfa = sfa_atoms.permute(4, 3, 2, 1, 0)
    dev = torch.device("cuda")
    meta = torch.zeros(1, dtype=torch.int32, device=dev)
    desc = torch.zeros(1, dtype=torch.int64, device=dev)
    case.out.fill_(float("nan"))
    stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    args = tuple(from_dlpack(t, assumed_align=16) for t in (a, b, c, meta, desc, sfa, sfb))
    launch = _launcher(case.a_is_m_major)
    cute.compile(launch, *args, stream)(*args, stream)
    torch.cuda.synchronize()
    return case.out


def _check(case: _Case, out: torch.Tensor, tag: str) -> None:
    ref = case.reference()
    got = out.double()
    assert torch.isfinite(got).all(), f"{tag}: unwritten / non-finite output cells: {int((~torch.isfinite(got)).sum())}"
    err = (got - ref).abs()
    scale = ref.abs().mean().item()
    print(f"\n{tag}: max|err| {err.max().item():.3e} mean|err| {err.mean().item():.3e} mean|ref| {scale:.3e} max|ref| {ref.abs().max().item():.3e}")
    # One bf16 ulp: the accumulator is fp32 (K <= 640 products of |x| <= 448 * 16 -- exact to ~1e-7 relative), the output ONE
    # bf16 rounding (<= 2^-8 relative); the atol covers cancellation near zero at the same ulp of the tensor's mean magnitude.
    torch.testing.assert_close(got, ref, rtol=2.0**-7, atol=2.0**-7 * scale)


_SHAPES = [(1, 2, 512, 512), (2, 1, 384, 640)]  # (B, H, S_q, S_kv): dense square; two batch entries, non-square, M and K tails


@requires_rubin
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"B{s[0]}H{s[1]}Sq{s[2]}Skv{s[3]}")
@pytest.mark.parametrize("a_is_m_major", (False, True), ids=("dK-Kmajor", "dQ-Mmajor"))
def test_block_scale_gemm_matches_the_dequantized_fp64_reference(a_is_m_major, shape):
    case = _Case(a_is_m_major, *shape)
    out = _run(case)
    _check(case, out, f"block-scale {'dQ' if a_is_m_major else 'dK'} {shape}")


@requires_rubin
@pytest.mark.parametrize("a_is_m_major", (False, True), ids=("dK-Kmajor", "dQ-Mmajor"))
def test_block_scale_gemm_detects_a_wrong_scale_byte(a_is_m_major):
    """The RED twin: ONE scale byte of ONE atom raised by 3 (an 8x scale on one 32-element block of one A row) must fail the
    comparison the green test passes -- the pin that the SF bytes reach the MMA at the atom position the contract names."""
    case = _Case(a_is_m_major, 1, 2, 512, 512, seed=5)
    _check(case, _run(case), "green twin")
    bad = case.sfa_atoms.clone()
    flat = bad[0, 1, 1, 2]  # atom (b 0, h 1, m_tile 1, k_tile 2)
    idx = int((flat >= 124).nonzero()[0])  # a live block (not the amax == 0 byte, whose block contributes nothing either way)
    flat[idx] = flat[idx] + 3
    with pytest.raises(AssertionError):
        _check(case, _run(case, sfa_atoms=bad), "RED twin (must fail)")
