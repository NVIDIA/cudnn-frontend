# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM107 LEAN index-list sparse d256 DECODE tile (``sm107/sparse_lean_d256_f16.py``: ``mma.sync`` + ``cp.async``, 128 threads, no TMEM /
TMA / CLC) and the visible-set helper it shares with the tcgen05 tile (``sm107/_sparse_item_common.py``).

Host tier (any box): the config validator pins the body's geometry; the kernel source pins (no tcgen05 / TMEM / TMA / mbarrier token, the
denominator floor exactly twice and SELECTed, the running-max guard CALL, the PDL pair's order, the one cp.async spelling, the explicit ABI);
the item-helper LIFT keeps the tcgen05 module's cubins byte-identical (the committed md5 record, host sm_107a trace-compiles at
PYTHONHASHSEED=0); every S_q = 1 rendering trace-compiles for sm_107a within the register budget (REG <= 128 at k 1, STL 0, LDL 0) with the barrier /
copy counts the header's tables derive.

Rubin tier (cc 10.7): the ladder (one / two / three / 65 sub-tiles, B x H > 1, the identity bound, a dead sequence, no block_lens, f16, G 16,
G 1, k 2 / k 4 with dead trailing sub-tiles, dense == paged bitwise), the degenerate matrix row by row, the chain through the pinned combine
against the UNSPLIT tcgen05 core within the DERIVED budget, capture-safe replays, the PDL arm bitwise the plain arm.

Oracles are the sparse module's (``test_sdpa_fwd_sparse_sm107``: the fp32 reference anchored at the step-0 position, the derived
split-vs-unsplit budget, the paginator, the tcgen05 launcher) -- imported, never re-derived.  Tolerances: ATOL 2e-2 on O and LSE (the sparse
module's own budget), never widened; the lean and the tcgen05 tile are bitwise DIFFERENT by construction (a 4 x 64-d split-K sum in fixed
order vs one 256-d MMA chain, 32- vs 128-key tiles) and compared within the derived budget only.
"""

import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from frost_test_utils import nvdisasm_candidates, process_watchdog, requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl]

_HERE = Path(__file__).resolve().parent
_RENDERINGS = _HERE / "renderings"
D = 256
BS = 4
ATOL = 2e-2  # on O and LSE, no rtol: the sparse module's budget -- never widened here
_SENTINEL = 1.5e30


def _kernels_dir() -> Path:
    import cudnn.sdpa.fwd.config_sm107 as c107

    return Path(c107.__file__).resolve().parent / "kernels" / "sm107"


def _lean_source() -> str:
    return (_kernels_dir() / "sparse_lean_d256_f16.py").read_text()


def _common_source() -> str:
    return (_kernels_dir() / "_sparse_item_common.py").read_text()


def _tcgen05_source() -> str:
    return (_kernels_dir() / "sparse_d256_f16.py").read_text()


def _code(src: str) -> str:
    """The source without comment lines and without trailing `  # ...` comments (the pins are about code, not prose)."""
    return "\n".join(re.sub(r"\s+#.*$", "", ln) for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def _body(src: str) -> str:
    """The source without ANY docstring (module and function alike): the pins below are about CODE -- a header sentence naming a token
    ("no tcgen05", "the tcgen05 tile's contract") is not a use."""
    return re.sub(r'"""[\s\S]*?"""', "", src)


def _params(**kw):
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    base = dict(
        dtype_qkv=2,
        cta_mma=1,
        pack_gqa=True,
        qh_per_kh=12,
        qsa_block_topk=512,
        qsa_block_size=4,
        seq_kv_lens_present=True,
        bottom_right=True,
        qsa_list_per_sequence=True,
    )
    base.update(kw)
    return TemplateParams(**base)


# ============================================================================ host: the config pins the body's geometry
def test_lean_config_pins_the_body_geometry():
    from cudnn.sdpa.fwd.config_sm107 import (
        LEAN_SPARSE_MAX_SUB_TILES_PER_CTA,
        SMEM_STANDARD_CARVEOUT_BYTES,
        CfgD256SparseLean,
        _validate_cfg_d256_sparse_lean,
        lean_sparse_n_sub,
        lean_sparse_split,
        make_cfg_d256_sparse_lean,
    )
    from dataclasses import replace

    assert lean_sparse_n_sub(512) == 65 and lean_sparse_n_sub(8) == 2 and lean_sparse_n_sub(16) == 3 and lean_sparse_n_sub(256) == 33
    assert [lean_sparse_split(512, k) for k in (1, 2, 3, 4)] == [65, 33, 22, 17]
    cfg = make_cfg_d256_sparse_lean(_params())
    assert (cfg.SPLIT, cfg.SUB_TILES_PER_CTA, cfg.STAGES, cfg.THREADS, cfg.N_MT, cfg.SMEM_TOTAL_BYTES) == (65, 1, 1, 128, 1, 40960)
    assert cfg.SMEM_TOTAL_BYTES == 2 * cfg.STAGES * 32 * 512 + cfg.N_MT * 8192 <= SMEM_STANDARD_CARVEOUT_BYTES
    for k, split in ((2, 33), (3, 22), (4, 17), (8, 9)):
        c = make_cfg_d256_sparse_lean(_params(split_kv=split))
        assert (c.SUB_TILES_PER_CTA, c.SPLIT, c.STAGES, c.SMEM_TOTAL_BYTES) == (k, split, 2, 73728), (k, split)
        assert make_cfg_d256_sparse_lean(_params(), sub_tiles_per_cta=k) == c
    assert make_cfg_d256_sparse_lean(_params(split_kv=33), sub_tiles_per_cta=2).SUB_TILES_PER_CTA == 2
    # accepted arms: G 1 / 12 / 16, dense + paged (16 / 48 / 64), f16, block_lens is a compile() axis (not a config field), top_k 8 / 64 / 256 / 512
    for kw in (
        dict(qh_per_kh=1),
        dict(qh_per_kh=16),
        dict(paged_kv=True, page_size=16),
        dict(paged_kv=True, page_size=48),
        dict(paged_kv=True, page_size=64),
        dict(dtype_qkv=3),
        dict(qsa_block_topk=8),
        dict(qsa_block_topk=64),
        dict(qsa_block_topk=256),
    ):
        c = make_cfg_d256_sparse_lean(_params(**kw))
        assert c.THREADS == 128 and c.N_KEYS == 32, kw
    # typed declines, by name
    for kw, match in (
        (dict(split_kv=16), "not a lean split"),
        (dict(split_kv=33), None),  # accepted (k 2)
        (dict(qsa_block_topk=4), "multiple of 4 in \\[8, 512\\]"),
        (dict(qsa_block_topk=14), "multiple of 4 in \\[8, 512\\]"),
        (dict(qsa_block_topk=516), "multiple of 4 in \\[8, 512\\]"),
        (dict(qh_per_kh=17), "qh_per_kh"),
        (dict(dtype_qkv=0), "f16/bf16"),
        (dict(thd_varlen=True), "thd_varlen"),
        (dict(epilogue_gate=True), "epilogue_gate"),
        (dict(has_sink=True), "sink"),
        (dict(stats_log2=True), "stats_log2"),
        (dict(seq_q_lens_present=True), "seq_q_lens_present"),
        (dict(window_left=4), "window_left"),
        (dict(bottom_right=False), "DECODE form"),
        (dict(qsa_list_per_sequence=False), "DECODE form"),
        (dict(seq_kv_lens_present=False), "DECODE form"),
        (dict(qsa_include_open_block=False), "pure-list"),
        (dict(paged_kv=True, page_size=6), "page_size"),
        (dict(page_size=16), "page_size requires paged_kv"),
        (dict(pack_gqa=False), "pack_gqa"),
        (dict(cta_mma=2), "cta_mma"),
        (dict(qsa_softmax_groups=2), "other kernels"),
        (dict(qsa_block_topk=0), "not an index-list record"),
        (dict(qsa_block_size=8), "qsa_block_size"),
    ):
        if match is None:
            make_cfg_d256_sparse_lean(_params(**kw))
            continue
        with pytest.raises(ValueError, match=match):
            make_cfg_d256_sparse_lean(_params(**kw))
    with pytest.raises(ValueError, match="sub_tiles_per_cta"):
        make_cfg_d256_sparse_lean(_params(), sub_tiles_per_cta=LEAN_SPARSE_MAX_SUB_TILES_PER_CTA + 1)
    with pytest.raises(ValueError, match="n_keys"):
        make_cfg_d256_sparse_lean(_params(), n_keys=64)
    # the validator's own pins (the body's geometry): a record edited off the body is refused
    for bad, match in (
        (dict(THREADS=256), "THREADS must be 128"),
        (dict(N_KEYS=64, BLOCKS_PER_SUB=16), "32 keys"),
        (dict(STAGES=2), "STAGES must be min"),
        (dict(SPLIT=64), "SPLIT must be ceil"),
        (dict(SMEM_TOTAL_BYTES=40959), "SMEM_TOTAL_BYTES"),
        (dict(SEQ_KV_LENS_PRESENT=0), "DECODE form"),
        (dict(S_Q=2), "S_q = 1"),
    ):
        with pytest.raises(ValueError, match=match):
            _validate_cfg_d256_sparse_lean(replace(cfg, **bad))
    assert isinstance(cfg, CfgD256SparseLean)


# ============================================================================ host: the kernel source pins
def test_lean_kernel_source_pins():
    src = _lean_source()
    code = _code(_body(src))
    low = code.lower()
    for tok in (
        "tcgen05",
        "tmem",
        "clusterlaunchcontrol",
        "tensormap",
        "tma_gather",
        "tma_load",
        "mbarrier",
        "pipelinestate",
        "setmaxnreg",
        "atomic",
        "elect_sync",
    ):
        assert tok not in low, f"the lean body must not carry the token {tok!r}"
    # the denominator floor: ONE constant, used exactly twice (the log and the reciprocal of each row pair), each followed by the row_dead SELECT
    assert code.count("TINY = cutlass.Float32(1e-30)") == 1 and code.count("1e-30") == 1
    assert code.count("cute.math.max(l_tot_lo, TINY)") == 2 and code.count("cute.math.max(l_tot_hi, TINY)") == 2
    for frag in (
        "lse_lo = _select_f32(row_dead_lo, NEG_INF, lse_lo)",
        "lse_hi = _select_f32(row_dead_hi, NEG_INF, lse_hi)",
        "inv_lo = _select_f32(row_dead_lo, ZERO, inv_lo)",
        "inv_hi = _select_f32(row_dead_hi, ZERO, inv_hi)",
        "_select_f32(row_dead_lo, ZERO, o_acc[nb * 4 + 0] * inv_lo)",
        "_select_f32(row_dead_hi, ZERO, o_acc[nb * 4 + 3] * inv_hi)",
    ):
        assert frag in code, frag
    assert code.index("inv_hi = ONE / cute.math.max(l_tot_hi, TINY)") < code.index("lse_lo = _select_f32(row_dead_lo, NEG_INF, lse_lo)")
    # the running-max guard CALL (the family grep of sdpa-invariants section 10.9) and the dead-tile semantics through it
    assert code.count("row_max_for_exp2(") == 6, "ms_old / ms_new for both rows + the epilogue's m_nat for both rows"
    assert "alpha_lo = cute.math.exp2(cute.math.min(ms_lo_old - ms_lo, ZERO, ftz=True), fastmath=True)" in code
    # the accumulator initialisation: opaque zeros (no constant float into the mma asm), -inf running max, 0 sums
    assert code.count("opaque_f32_zero()") == 2 and "m_lo = NEG_INF" in code and "m_hi = NEG_INF" in code and "l_lo = ZERO" in code and "l_hi = ZERO" in code
    # the valid WORD mask (frost-tile-dsl 10d): one word per sub-tile, bits tested per lane; no apply_mask_chunk, no per-cell compare on key_abs
    assert "def _valid_word(" in code and code.count("apply_membership_fragment(") == 1 and "apply_mask_chunk" not in code
    assert "bits = word >>" not in code and "_select_f32(v0" not in code, "the membership mask is the shared tile_dsl op, never re-spelled per cell"
    assert "from cudnn.frost.tile_dsl.mask import apply_membership_fragment" in code
    # the gather: cp_async_gather_row16 is the ONLY cp.async spelling, one commit per sub-tile, the wait immediates fold on k
    assert "cp.async" not in code and code.count("cp_async_gather_row16(") == 2 and code.count("cp_async_commit()") == 1
    assert "cp_async_wait(1)" in code and "cp_async_wait(0)" in code
    # the barrier sites: A, X and the folded F (the refill's guard), nothing else; no `if` on a runtime value encloses one
    assert code.count("nvvm.barrier_cta_sync()") == 3
    i_f = code.index("if cutlass.const_expr(t + 2 < SUB_TILES_PER_CTA):\n            nvvm.barrier_cta_sync()")
    assert i_f > 0
    # the loops are compile-time (range_constexpr) -- no runtime trip count anywhere in the body
    assert code.count("cutlass.range_constexpr(") >= 12 and re.search(r"\bcutlass\.range\(", code) is None and re.search(r"\bwhile\b", code) is None
    # PDL: the wait textually before the first global read of the body, the release right after it, both under the constexpr
    kbody = code[code.index("def _kernel(") :]  # the helpers above the kernel hold the load SITES; the ORDER question is the kernel body's
    i_pdl = kbody.index("wait_on_dependent_grids()")
    assert kbody.index("launch_dependent_grids()") > i_pdl
    first_read = min(
        kbody.index(s) for s in ("visible_bounds(\n        zero", "ld_global(", "_issue_subtile(")
    )  # the lengths / counts, Q, the ids + table + pool rows
    assert i_pdl < first_read, "the PDL wait must precede every global read of the kernel body"
    assert "if cutlass.const_expr(use_pdl):" in kbody
    # the shared helper: visible_bounds + block_id_at are the only visible-set arithmetic (no local count / tail / dead derivation)
    assert "visible_bounds(" in code and "block_id_at(" in code and "n_sel_default" not in code and "tail_lo + rel" not in code
    assert "eff_seqlen_kv, n_vis = visible_bounds(" in code and "n_vis = cute.math" not in code, "n_vis comes from the helper, not a copy of its line"
    # the dense arm's extent clamp (a length past the tensor never reads past it; the TMA form clips the row to zeros) and the paged arm's page bound
    assert "live = (blk >= zero) & in_range & (tok < seqlen_kv)" in code and "live = (page >= zero) & in_range" in code
    # the zero-column table: the clamped table read is steered onto the ids row (a valid address) when max_pages == 0
    assert "table_base_i64 = _select_i64(max_pages > zero, cutlass.make_array_view(block_table_tensor).data_ptr().toint(), ids_base_i64)" in code
    assert "const_expr(True)" not in code, "no dead constexpr wrapper around a barrier site"
    # Rule S7: the partial batch is widened BEFORE the stride multiply
    assert "part_batch = cutlass.Int64(batch) + cutlass.Int64(chunk) * cutlass.Int64(n_batch)" in code
    # hygiene
    assert re.search(r"^assert\b", code, re.M) is None, "no module-level assert (the config validator raises)"
    assert "EXPLICIT_ABI = True" in code
    # the geometry is CFG's, never a literal: the body's constants derive from CFG fields
    for name in (
        "N_KEYS = CFG.N_KEYS",
        "SUB_TILES_PER_CTA = CFG.SUB_TILES_PER_CTA",
        "SPLIT = CFG.SPLIT",
        "STAGES = CFG.STAGES",
        "THREADS = CFG.THREADS",
        "G = CFG.QH_PER_KH",
    ):
        assert name in code, name


def test_lean_host_entry_has_no_defaulted_slot_ahead_of_stream():
    """frost-gotchas (PR #1649): a defaulted parameter ahead of ``stream`` breaks the compiled entry's call contract."""
    from cudnn.frost.template_loader import load_template

    mod = load_template(str(_kernels_dir() / "sparse_lean_d256_f16.py"), _params(), tag="lean_host_sig")
    sig = inspect.signature(mod._host)
    names = list(sig.parameters)
    assert names[-1] == "stream" and names[-2] == "use_pdl"
    for n in names[:-1]:
        assert sig.parameters[n].default is inspect.Parameter.empty, f"{n} has a default ahead of stream"
    assert mod.EXPLICIT_ABI is True and mod.CFG.SPLIT == 65


def test_shared_item_helper_carries_the_visible_set_and_the_tcgen05_module_calls_it():
    """The ONE helper for the visible set: the decode-form fragments live in ``_sparse_item_common.py`` (moved there with the lift) and the
    tcgen05 module's ``_item_bounds`` keeps its name, its tile / chunk cut and calls ``visible_bounds``; ``_block_id_at`` forwards."""
    common = _code(_body(_common_source()))
    for frag in (
        "pos = tok + (eff_seqlen_kv - eff_seqlen_q)",
        "anchor = eff_seqlen_kv - eff_seqlen_q",
        "n_tail = cute.math.max((n_vis + bs - one) // bs - tail_lo, zero)",
        "n_tail = _select_i32((n_vis % bs) != zero, 1, 0)",
        "tail = _select_i32((rel >= cutlass.Int32(0)) & (rel < n_tail), tail_lo + rel, cutlass.Int32(-1))",
        "tail = _select_i32((idx == count) & (n_tail != cutlass.Int32(0)), tail_lo, cutlass.Int32(-1))",
        "invalid = dead | (blk < cutlass.Int32(0)) | (blk > cutlass.Int32((2**31 - 1) // block_size))",
        "return pos, count, n_tail, tail_lo, dead, n_listed, eff_seqlen_kv, n_vis",
    ):
        assert frag in common, frag
    assert "FROST_TEMPLATE_PARAMS" not in common and "CFG" not in common, "the shared helper is not a template: every knob is a constexpr parameter"
    tc = _code(_body(_tcgen05_source()))
    assert "from cudnn.sdpa.fwd.kernels.sm107._sparse_item_common import _select_f32, _select_i32, block_id_at, visible_bounds" in tc
    i_ib = tc.index("def _item_bounds(")
    ib = tc[i_ib : tc.index("\ndef ", i_ib + 1)]
    for frag in (
        "= visible_bounds(",
        "seq_kv_lens_present=CFG.SEQ_KV_LENS_PRESENT",
        "include_open_block=CFG.INCLUDE_OPEN_BLOCK",
        "tile0 = split * per + cute.math.min(split, rem)",
        "dead = dead | (n_sp == zero)",
        "n_tiles = cute.math.max(cutlass.Int32(1), n_sp)",
        "return pos, count, n_tail, tail_lo, dead, n_tiles, eff_seqlen_kv, tile0",
    ):
        assert frag in ib, frag
    assert "n_sel_default" not in ib and "n_vis = cute.math" not in ib, "the visible-set arithmetic left the tcgen05 module"
    assert "eff_seqlen_kv, _n_vis = visible_bounds(" in ib, "the tcgen05 module unpacks the helper's 8-tuple (its n_vis unused there)"
    assert "return block_id_at(idx, loaded, count, n_tail, tail_lo, dead, list_per_sequence=LIST_PER_SEQUENCE, block_size=BLOCK_SIZE)" in tc


# ============================================================================ host: the item-helper lift keeps the tcgen05 cubins byte-identical
_LIFT_RECORD = _RENDERINGS / "md5_sparse_d256_item_lift_sm_107a.txt"
# The tcgen05 renderings of the record (TemplateParams kwargs + compile() kwargs); each a separate process at PYTHONHASHSEED=0.
_LIFT_RENDERINGS = {
    "prefill_dense_bf16_g12": (
        dict(dtype_qkv=2, cta_mma=1, pack_gqa=True, qh_per_kh=12, qsa_block_topk=512, qsa_block_size=4),
        dict(has_lse=True, has_block_lens=True),
    ),
    "prefill_dense_f16_g16_top4": (
        dict(dtype_qkv=3, cta_mma=1, pack_gqa=True, qh_per_kh=16, qsa_block_topk=4, qsa_block_size=4),
        dict(has_lse=True, has_block_lens=False),
    ),
    "prefill_gated_bf16_g12": (
        dict(dtype_qkv=2, cta_mma=1, pack_gqa=True, qh_per_kh=12, qsa_block_topk=512, qsa_block_size=4, epilogue_gate=True),
        dict(has_lse=False, has_block_lens=True),
    ),
    "prefill_thd_bf16_g12": (
        dict(dtype_qkv=2, cta_mma=1, pack_gqa=True, qh_per_kh=12, qsa_block_topk=512, qsa_block_size=4, thd_varlen=True, seq_kv_lens_present=True),
        dict(has_lse=True, has_block_lens=True),
    ),
    "decode_paged16_split16_bf16_g12": (
        dict(
            dtype_qkv=2,
            cta_mma=1,
            pack_gqa=True,
            qh_per_kh=12,
            qsa_block_topk=512,
            qsa_block_size=4,
            paged_kv=True,
            page_size=16,
            seq_kv_lens_present=True,
            bottom_right=True,
            qsa_list_per_sequence=True,
            split_kv=16,
        ),
        dict(has_lse=True, has_block_lens=True),
    ),
    "decode_paged16_split16_bf16_g12_pdl": (
        dict(
            dtype_qkv=2,
            cta_mma=1,
            pack_gqa=True,
            qh_per_kh=12,
            qsa_block_topk=512,
            qsa_block_size=4,
            paged_kv=True,
            page_size=16,
            seq_kv_lens_present=True,
            bottom_right=True,
            qsa_list_per_sequence=True,
            split_kv=16,
        ),
        dict(has_lse=True, has_block_lens=True, use_pdl=True),
    ),
    "decode_dense_split3_bf16_g1": (
        dict(
            dtype_qkv=2,
            cta_mma=1,
            pack_gqa=True,
            qh_per_kh=1,
            qsa_block_topk=64,
            qsa_block_size=4,
            seq_kv_lens_present=True,
            bottom_right=True,
            qsa_list_per_sequence=True,
            split_kv=3,
        ),
        dict(has_lse=True, has_block_lens=False),
    ),
    "prefill_paged64_bf16_g12_blocklens": (
        dict(dtype_qkv=2, cta_mma=1, pack_gqa=True, qh_per_kh=12, qsa_block_topk=256, qsa_block_size=4, paged_kv=True, page_size=64, seq_kv_lens_present=True),
        dict(has_lse=True, has_block_lens=True),
    ),
}
_RENDER_SRC = """
import glob, hashlib, json, os, re, subprocess, sys
dump, module_file, pjson, cjson, cands = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
os.environ["CUTE_DSL_DUMP_DIR"] = dump
os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
os.environ["CUTE_DSL_ARCH"] = "sm_107a"
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
os.environ["CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO"] = "0"
from cudnn.frost.template_loader import load_template
from cudnn.sdpa.fwd.config_sm100 import TemplateParams
import cudnn.sdpa.fwd.config_sm107 as c107
path = os.path.join(os.path.dirname(os.path.abspath(c107.__file__)), "kernels", "sm107", module_file)
mod = load_template(path, TemplateParams(**json.loads(pjson)), tag="render_gate")
ckw = json.loads(cjson)
(mod.compile.__wrapped__ if hasattr(mod.compile, "__wrapped__") else mod.compile)(**ckw)
cubins = sorted(glob.glob(os.path.join(dump, "*.cubin")), key=os.path.getmtime)
if not cubins:
    print("FAIL no cubin dumped into", dump, os.listdir(dump)); sys.exit(3)
print("CUBIN_MD5", hashlib.md5(open(cubins[-1], "rb").read()).hexdigest())
ptxs = sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)
if ptxs:
    ptx = open(ptxs[-1]).read()
    print("PTX_GRIDDEPCONTROL", len(re.findall(r"griddepcontrol", ptx)))
    print("PTX_REQNTID", re.findall(r"\\.reqntid\\s+(\\d+)", ptx)[:1])
    print("PTX_CPASYNC", len(re.findall(r"cp\\.async\\.cg", ptx)))
    print("PTX_MAXNREG", int(bool(re.search(r"\\.maxnreg", ptx))))
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
def cnt(sub): return sum(1 for ln in sass if sub in ln)
for key, sub in {"STL": " STL", "LDL": " LDL", "BAR.SYNC": "BAR.SYNC", "LDGSTS": "LDGSTS", "HMMA": "HMMA", "LDSM": "LDSM", "USETMAXREG": "USETMAXREG", "R2P": " R2P "}.items():
    print("SASS", key, cnt(sub))
print("SASS LINES", len(sass))
cuobj = os.path.join(os.path.dirname(nvd), "cuobjdump")
if os.path.isfile(cuobj):
    res = subprocess.run([cuobj, "-res-usage", cubins[-1]], capture_output=True, text=True).stdout
    regs = [int(m) for m in re.findall(r"REG:(\\d+)", res)]
    locs = [int(m) for m in re.findall(r"LOCAL:(\\d+)", res)]
    if regs:
        print("RES REG", max(regs))
        print("RES LOCAL", max(locs))
"""


def _render(tmp_path, module_file: str, pkw: dict, ckw: dict, tag: str, *, need_sass: bool) -> dict:
    """Host trace-compile one rendering in a fresh interpreter (PYTHONHASHSEED=0) and parse CUBIN_MD5 / SASS / RES lines."""
    from frost_test_utils import arch_known_to_the_dsl

    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a")
    cands = nvdisasm_candidates() if need_sass else []
    dump = tmp_path / f"{tag}_{hashlib.md5(json.dumps([pkw, ckw], sort_keys=True).encode()).hexdigest()[:8]}"
    dump.mkdir()
    env = dict(os.environ, PYTHONHASHSEED="0")
    proc = subprocess.run(
        [sys.executable, "-c", _RENDER_SRC, str(dump), module_file, json.dumps(pkw), json.dumps(ckw), *cands],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    assert proc.returncode == 0, f"sm_107a trace-compile of {tag} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    res = {"md5": next(ln.split()[1] for ln in out if ln.startswith("CUBIN_MD5 "))}
    res["skip"] = [ln for ln in out if ln.startswith(("SKIP", "REJECT"))]
    for ln in out:
        p = ln.split()
        if ln.startswith("SASS ") and len(p) == 3:
            res[p[1]] = int(p[2])
        elif ln.startswith("RES ") and len(p) == 3:
            res[p[1]] = int(p[2])
        elif ln.startswith("PTX_") and len(p) == 2:
            res[p[0]] = p[1]
    print(f"\n{tag}: {res}")
    return res


def _lift_record():
    assert _LIFT_RECORD.is_file(), _LIFT_RECORD
    dsl, want = None, {}
    for ln in _LIFT_RECORD.read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[len("dsl=") :].strip()
        m = re.match(r"sparse_d256_f16 sm_107a (\S+) cubin_md5=([0-9a-f]{32})", ln)
        if m:
            want[m.group(1)] = m.group(2)
    return dsl, want


def test_item_lift_md5_record_is_committed_and_complete():
    dsl, want = _lift_record()
    assert dsl and set(want) == set(_LIFT_RENDERINGS), (sorted(want), sorted(_LIFT_RENDERINGS))


@pytest.mark.parametrize("rendering", sorted(_LIFT_RENDERINGS))
def test_item_helper_lift_keeps_the_tcgen05_cubins_byte_identical(tmp_path, rendering):
    """The tcgen05 module renders BYTE-IDENTICAL cubins with the visible-set helper lifted into ``_sparse_item_common`` (every knob a
    constexpr parameter folds like the module global it replaced): the committed md5s were rendered on the tree BEFORE the lift at
    PYTHONHASHSEED=0 (frost-gotchas SASS: two images from one source are a hash-seed artefact; frost-kernels.md section 6).  A record stays
    valid for the DSL build it names."""
    from cudnn.frost.buffers import cutedsl_state

    dsl, want = _lift_record()
    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if dsl is not None and have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: the cubin is a function of the DSL build, re-render the record")
    pkw, ckw = _LIFT_RENDERINGS[rendering]
    res = _render(tmp_path, "sparse_d256_f16.py", pkw, ckw, f"lift_{rendering}", need_sass=False)
    assert (
        res["md5"] == want[rendering]
    ), f"{rendering}: cubin md5 {res['md5']} != the pre-lift record's {want[rendering]} -- the lift changed the tcgen05 rendering"


# ============================================================================ host: every S_q = 1 rendering within the register budget
def _lean_pkw(**kw):
    base = dict(
        dtype_qkv=2,
        cta_mma=1,
        pack_gqa=True,
        qh_per_kh=12,
        qsa_block_topk=512,
        qsa_block_size=4,
        seq_kv_lens_present=True,
        bottom_right=True,
        qsa_list_per_sequence=True,
    )
    base.update(kw)
    return base


def _lean_bar_syncs(k: int) -> int:
    """The header's barrier table: A and X per sub-tile, F only where a refill follows (t + 2 < k)."""
    return 2 * k + max(0, k - 2)


_LEAN_RENDERINGS = {
    "dense_k1_g12": (_lean_pkw(), dict(has_block_lens=True)),
    "dense_k1_g12_pdl": (_lean_pkw(), dict(has_block_lens=True, use_pdl=True)),
    "dense_k1_g12_nolens": (_lean_pkw(), dict(has_block_lens=False)),
    "paged16_k1_g12": (_lean_pkw(paged_kv=True, page_size=16), dict(has_block_lens=True)),
    "paged16_k1_g12_pdl": (_lean_pkw(paged_kv=True, page_size=16), dict(has_block_lens=True, use_pdl=True)),
    "dense_k2_g12": (_lean_pkw(split_kv=33), dict(has_block_lens=True)),
    "paged16_k2_g12": (_lean_pkw(paged_kv=True, page_size=16, split_kv=33), dict(has_block_lens=True)),
    "paged64_k4_g12": (_lean_pkw(paged_kv=True, page_size=64, split_kv=17), dict(has_block_lens=True)),
    "dense_k3_g12": (_lean_pkw(split_kv=22), dict(has_block_lens=True)),
    "dense_k1_g1": (_lean_pkw(qh_per_kh=1), dict(has_block_lens=True)),
    "dense_k1_g16": (_lean_pkw(qh_per_kh=16), dict(has_block_lens=True)),
    "dense_k1_g12_f16": (_lean_pkw(dtype_qkv=3), dict(has_block_lens=True)),
    "paged16_k1_g12_f16_nolens": (_lean_pkw(dtype_qkv=3, paged_kv=True, page_size=16), dict(has_block_lens=False)),
    "dense_k1_g12_top8": (_lean_pkw(qsa_block_topk=8), dict(has_block_lens=True)),
}


@pytest.mark.parametrize("rendering", sorted(_LEAN_RENDERINGS))
def test_lean_renderings_trace_compile_within_the_register_budget(tmp_path, rendering):
    """Host sm_107a trace-compile of every S_q = 1 rendering: REG <= 128 at k 1 (4 CTAs per SM by registers; the DSL emits .reqntid 128 and no .maxnreg,
    so ptxas sizes the flat body), STL 0 / LDL 0, no USETMAXREG, the header's counts -- BAR.SYNC = 2 k + max(0, k - 2), LDGSTS = 16 k (the
    K + V copies of every sub-tile), HMMA = 32 k (16 BMM1 + 16 BMM2 per sub-tile per warp), LDSM = 16 k, the bit-word mask (R2P > 0) --
    and the griddepcontrol pair present iff use_pdl (the OFF rendering carries no token).  Skips where no nvdisasm decodes the cubin."""
    from cudnn.sdpa.fwd.config_sm107 import make_cfg_d256_sparse_lean

    pkw, ckw = _LEAN_RENDERINGS[rendering]
    k = make_cfg_d256_sparse_lean(_params(**{kk: v for kk, v in pkw.items()})).SUB_TILES_PER_CTA
    res = _render(tmp_path, "sparse_lean_d256_f16.py", pkw, dict(has_lse=True, **ckw), f"lean_{rendering}", need_sass=True)
    assert res.get("PTX_REQNTID") == "['128']" and res.get("PTX_MAXNREG") == "0"
    assert int(res.get("PTX_CPASYNC", 0)) == 16 * k, res
    assert int(res.get("PTX_GRIDDEPCONTROL", -1)) == (2 if ckw.get("use_pdl") else 0), res
    if res["skip"]:
        pytest.skip(str(res["skip"]))
    # k = 1 (the launch-bound cells' rendering): REG <= 128 = 4 CTAs / SM by registers (measured 70 dense / 72 paged).  k >= 2: ptxas software-
    # pipelines the unrolled sub-tiles (the next sub-tile's fragments hoisted over the current one's chain) and takes 104..164 registers at
    # k 2..4 -- 3 CTAs / SM, the SMEM bound of the 72-KiB two-stage form anyway; the bound here is 168 (a -maxrregcount=128 build is a MEASURED
    # lever of the gate, never assumed).  k 8 (REG 204, 2 CTAs / SM) is outside the fill rule's range and not pinned.
    reg_bound = 128 if k == 1 else 168
    assert res["REG"] <= reg_bound, f"REG {res['REG']} > {reg_bound} at k {k}: the fragment plan no longer fits (the occupancy by registers drops)"
    assert res["LOCAL"] == 0 and res["STL"] == 0 and res["LDL"] == 0, f"spills: {res}"
    assert res["USETMAXREG"] == 0 and res["R2P"] > 0
    assert res["BAR.SYNC"] == _lean_bar_syncs(k) and res["LDGSTS"] == 16 * k and res["HMMA"] == 32 * k and res["LDSM"] == 16 * k, (k, res)


# ============================================================================ Rubin: the launchers and the oracles
def _sparse_suite():
    """The sparse module's oracles and launchers (imported, never re-derived)."""
    import test_sdpa_fwd_sparse_sm107 as S

    return S


def _stream():
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _pool_terms(pool):
    """(s_page, s_tok, col_head, rows_per_page, rows_per_head) of a ``[num_pages, KH, P, D]`` pool view (HND contiguous, or NHD storage permuted into it)."""
    s_page, s_head, s_tok, _ = pool.stride()
    if s_head >= s_tok and s_head % s_tok == 0:
        return s_page, s_tok, 0, s_page // s_tok, s_head // s_tok
    return s_page, s_tok, s_head, s_page // s_tok, 0


class LeanRun:
    """One compiled lean rendering bound to one set of operands: ``run()`` launches the kernel ALONE into the sentinel-filled fp32 slabs."""

    def __init__(self, q, kv, ids, lens, kv_lens, top_k, scale, *, k=1, pdl=False, sentinel=True):
        from cudnn.frost.template_loader import load_template
        from cudnn.sdpa.fwd.config_sm107 import lean_sparse_split

        import cutlass
        from cutlass.cute.runtime import make_ptr
        import cutlass.cute as cute

        self.q, self.ids, self.lens, self.top_k, self.scale = q, ids, lens, top_k, float(scale)
        dev = q.device
        B, S, H, _ = q.shape
        assert S == 1
        self.B, self.H = B, H
        self.paged = len(kv) == 4
        if self.paged:
            self.k_pool, self.v_pool, self.table, self.P = kv
            KH = self.k_pool.shape[1]
        else:
            self.k, self.v = kv
            KH = self.k.shape[2]
        self.KH = KH
        self.kv_lens = torch.tensor(kv_lens, device=dev, dtype=torch.int32)
        split = lean_sparse_split(top_k, k)
        self.split = split
        dtype_code = 2 if q.dtype == torch.bfloat16 else 3
        params = _params(
            dtype_qkv=dtype_code,
            qh_per_kh=H // KH,
            qsa_block_topk=top_k,
            paged_kv=self.paged,
            page_size=self.P if self.paged else 0,
            split_kv=split if split > 1 else 1,
        )
        self.mod = load_template(str(_kernels_dir() / "sparse_lean_d256_f16.py"), params, tag="sdpa_fwd_sm107_sparse_lean")
        assert self.mod.CFG.SPLIT == split and self.mod.CFG.SUB_TILES_PER_CTA == k, (self.mod.CFG.SPLIT, split, self.mod.CFG.SUB_TILES_PER_CTA, k)
        compile_kernel = getattr(self.mod.compile, "__wrapped__", self.mod.compile)
        self.fn = compile_kernel(has_lse=True, has_block_lens=lens is not None, use_pdl=pdl)
        fill = _SENTINEL if sentinel else 0.0
        self.o_part = torch.full((B * split, 1, H, D), fill, device=dev, dtype=torch.float32)
        self.lse_part = torch.full((B * split, H, 1), fill, device=dev, dtype=torch.float32)
        gmem = cute.AddressSpace.gmem
        half = cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float16

        def P_(t, dtype, align=16):
            return None if t is None else make_ptr(dtype, t.data_ptr(), gmem, assumed_align=align)

        if self.paged:
            ks = _pool_terms(self.k_pool)
            vs = _pool_terms(self.v_pool)
            num_pages, max_pages = self.k_pool.shape[0], self.table.shape[1]
            k_strides, v_strides = (ks[0], ks[1], ks[2]), (vs[0], vs[1], vs[2])
            paged_geom = (num_pages, max_pages, ks[3], ks[4], vs[3], vs[4])
            skv = max_pages * self.P
            k_ptr, v_ptr, t_ptr = P_(self.k_pool, half), P_(self.v_pool, half), P_(self.table, cutlass.Int32, 4)
        else:
            k_strides, v_strides = tuple(self.k.stride()[:3]), tuple(self.v.stride()[:3])
            paged_geom = (0, 0, 0, 0, 0, 0)
            skv = self.k.shape[1]
            k_ptr, v_ptr, t_ptr = P_(self.k, half), P_(self.v, half), None
        self._kw = dict(
            q_ptr=P_(q, half),
            k_ptr=k_ptr,
            v_ptr=v_ptr,
            o_ptr=P_(self.o_part, cutlass.Float32),
            lse_ptr=P_(self.lse_part, cutlass.Float32, 4),
            block_ids_ptr=P_(ids, cutlass.Int32, 16),
            block_lens_ptr=P_(lens, cutlass.Int32, 4),
            seq_kv_lens_ptr=P_(self.kv_lens, cutlass.Int32, 4),
            problem_size=(B, H, KH, 1, skv),
            q_strides=tuple(q.stride()[:3]),
            k_strides=k_strides,
            v_strides=v_strides,
            o_strides=(1 * H * D, H * D, D),
            lse_strides=(H * 1, 1, 1),
            scale_softmax_log2=cutlass.Float32(self.scale * math.log2(math.e)),
            block_table_ptr=t_ptr,
            paged_geom=paged_geom,
        )

    def run(self, stream=None):
        self.fn(**self._kw, stream=_stream() if stream is None else stream)

    def combine_exact(self):
        """The exact torch fp32 combine of the slab: O = sum_s w_s O_s / sum_s w_s, LSE = m + log(sum_s w_s) with w_s = exp(lse_s - m); all-dead
        rows 0 / -inf.  Returns (O [B, 1, H, D] fp32, LSE [B, H, 1] fp32)."""
        B, S_, H = self.B, self.split, self.H
        o = self.o_part.view(S_, B, 1, H, D).double()  # split-major: index b + s B -> [s, b]
        lse = self.lse_part.view(S_, B, H, 1).double()
        m = lse.amax(dim=0)  # [B, H, 1]
        dead = torch.isinf(m) & (m < 0)
        m_safe = torch.where(dead, torch.zeros_like(m), m)
        w = torch.exp(lse - m_safe[None])  # [S, B, H, 1]; exp(-inf) = 0 for dead chunks
        den = w.sum(dim=0)  # [B, H, 1]
        num = (o * w.permute(0, 1, 3, 2)[..., None]).sum(dim=0)  # [B, 1, H, D]
        den_safe = torch.where(dead, torch.ones_like(den), den)
        o_out = num / den_safe.permute(0, 2, 1)[..., None]
        o_out = torch.where(dead.permute(0, 2, 1)[..., None], torch.zeros_like(o_out), o_out)
        lse_out = torch.where(dead, torch.full_like(m, float("-inf")), m_safe + torch.log(den_safe))
        return o_out.float(), lse_out.float()

    def combine_via_kernel(self, dtype):
        """The pinned split combine (``split_combine.compile_ptr`` -> the positional entry, exactly as the sparse adapter launches it)."""
        from cudnn.frost.compiled_cache import positional_entry
        from cudnn.sdpa.fwd.kernels.sm100 import split_combine
        from cudnn.sdpa.fwd.sparse_gqa_sm107 import _sparse_combine_decode_grid

        B, H = self.B, self.H
        dev = self.q.device
        o = torch.full((B, 1, H, D), _SENTINEL if dtype == torch.bfloat16 else 6.0e4, device=dev, dtype=dtype)
        lse = torch.full((B, H, 1), _SENTINEL, device=dev, dtype=torch.float32)
        cc = torch.cuda.get_device_capability(dev)
        form = _sparse_combine_decode_grid(B, H, 1, self.split, cc)
        tag = "bf16" if dtype == torch.bfloat16 else "f16"
        compile_combine = getattr(split_combine.compile_ptr, "__wrapped__", split_combine.compile_ptr)
        owner = compile_combine(dtype_o=tag, dtype_partial="f32", has_lse=True, stats_log2=False, gate=False, dtype_gate=None, decode_grid=form, use_pdl=False)
        entry = positional_entry(owner)
        assert entry is not None
        args = [
            self.o_part.data_ptr(),
            self.lse_part.data_ptr(),
            o.data_ptr(),
            lse.data_ptr(),
            (B, H, 1, D),
            self.split,
            tuple(o.stride()[:4]),
            tuple(lse.stride()[:3]),
        ]
        entry(*args, int(_stream()))
        torch.cuda.synchronize()
        return o, lse, form


def _slab_audit(run: LeanRun, ref_lse):
    """Every (chunk, row) slot the combine reads was WRITTEN (no sentinel survives), every slot finite or exactly -inf, and a dead row's
    slots are exactly 0 / -inf on every chunk."""
    o, lse = run.o_part, run.lse_part
    assert not (o == _SENTINEL).any(), "an O partial slot was never written"
    assert not (lse == _SENTINEL).any(), "an LSE partial slot was never written"
    assert torch.isfinite(o).all(), "a non-finite O partial"
    assert (torch.isfinite(lse) | (torch.isinf(lse) & (lse < 0))).all(), "an LSE partial that is neither finite nor -inf"
    dead = torch.isinf(ref_lse)  # [B, H, 1]
    if dead.any():
        S_, B = run.split, run.B
        H_ = run.H
        o4 = o.view(S_, B, H_, D)  # tok 0 squeezed: [chunk, b, head, d]
        l4 = lse.view(S_, B, H_, 1)
        dm = dead.view(1, B, H_, 1).expand(S_, B, H_, 1)
        assert (l4[dm] == float("-inf")).all(), "a dead row's LSE partial must be -inf on every chunk"
        assert (o4[dm[..., 0]] == 0).all(), "a dead row's O partial must be 0 on every chunk"


def _check_outputs(o, lse, ref_o, ref_lse, *, what: str):
    """Finite, dead rows exact, within ATOL on the live rows; returns (max|dO|, max|dLSE|)."""
    assert torch.isfinite(o).all(), f"{what}: non-finite O"
    dead = torch.isinf(ref_lse)
    if dead.any():
        assert torch.isinf(lse[dead]).all() and (lse[dead] < 0).all(), f"{what}: a dead row's LSE must be -inf exactly"
        assert (o.permute(0, 2, 1, 3)[dead] == 0).all(), f"{what}: a dead row's O must be 0 exactly"
    live = ~dead
    max_o = float((o.float() - ref_o).abs().max())
    max_lse = float((lse[live] - ref_lse[live]).abs().max()) if live.any() else 0.0
    assert max_o <= ATOL, f"{what}: O off the reference by {max_o} (budget {ATOL})"
    assert max_lse <= ATOL, f"{what}: LSE off the reference by {max_lse} (budget {ATOL})"
    return max_o, max_lse


def _lean_cell(
    *,
    B,
    SKV,
    kv_lens,
    H=24,
    KH=2,
    dtype=torch.bfloat16,
    top_k=512,
    list_kind="shuffled",
    k=1,
    paged=None,
    seed=0,
    block_lens=True,
    scale=None,
    pdl=False,
    vs_core=True,
):
    """One lean cell: two launches bitwise, the slab audit, the exact combine within ATOL of the fp32 oracle, the pinned combine within ATOL
    too, and -- ``vs_core`` -- the lean O / LSE against the UNSPLIT tcgen05 core within the DERIVED budget (never bitwise across tiles); a
    paged twin (``paged = (P, hnd)``) BITWISE the dense slab."""
    S = _sparse_suite()
    q, kk, vv, g = S._inputs(B, 1, H, KH, dtype, seed, SKV=SKV)
    ids, lens = S._decode_lists(list_kind, B, 1, top_k, kv_lens, g)
    lens_arg = lens if block_lens else None
    scale = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    ref_o, ref_lse, pv_abs = S._reference_decode(q, kk, vv, ids, lens_arg, kv_lens, scale, top_k)
    with process_watchdog(300, f"lean cell B {B} SKV {SKV} k {k} compile + first dispatch"):
        run = LeanRun(q, (kk, vv), ids, lens_arg, kv_lens, top_k, scale, k=k, pdl=pdl)
        run.run()
        torch.cuda.synchronize()
    _slab_audit(run, ref_lse)
    o1, l1 = run.o_part.clone(), run.lse_part.clone()
    run.o_part.fill_(_SENTINEL)
    run.lse_part.fill_(_SENTINEL)
    run.run()
    torch.cuda.synchronize()
    assert torch.equal(o1, run.o_part) and torch.equal(l1, run.lse_part), "two launches must be bitwise"
    o_x, lse_x = run.combine_exact()
    max_o, max_lse = _check_outputs(o_x, lse_x, ref_o, ref_lse, what="lean (exact combine)")
    o_c, lse_c, form = run.combine_via_kernel(dtype)
    assert not (o_c.float() == float(torch.full((), _SENTINEL if dtype == torch.bfloat16 else 6.0e4, dtype=dtype).float())).any(dim=-1).any()
    c_o, c_lse = _check_outputs(o_c, lse_c, ref_o, ref_lse, what=f"lean + the pinned combine (form {form})")
    res = dict(max_o=max_o, max_lse=max_lse, combine_max_o=c_o, combine_max_lse=c_lse, combine_form=form, dead=int(torch.isinf(ref_lse).sum()), split=run.split)
    if paged is not None:
        P, hnd = paged
        kp, vp, table = S._paginate(kk, vv, kv_lens, P, hnd, "nan", g)
        with process_watchdog(300, f"lean paged cell P {P} compile + first dispatch"):
            pr = LeanRun(q, (kp, vp, table, P), ids, lens_arg, kv_lens, top_k, scale, k=k, pdl=pdl)
            pr.run()
            torch.cuda.synchronize()
        assert torch.equal(pr.o_part, run.o_part) and torch.equal(
            pr.lse_part, run.lse_part
        ), f"paged page {P} {'HND' if hnd else 'NHD'}: the slab differs from the dense read of the same tokens"
    if vs_core:
        core = S._launch_decode(q, (kk, vv), ids, lens_arg, kv_lens, top_k, scale, split=1, launches=1)
        co, cl = core[0][0], core[0][1]
        S._check([(co, cl)], ref_o, ref_lse)
        live = ~torch.isinf(ref_lse).permute(0, 2, 1)  # [B, S, H]
        d, ratio = S._split_vs_unsplit(o_x.to(dtype).float(), co.float(), dtype, pv_abs, live)
        lse_d = float((lse_x - cl)[~torch.isinf(ref_lse)].abs().max()) if live.any() else 0.0
        assert (
            ratio <= 1.0
        ), f"lean vs the unsplit tcgen05 core: {ratio:.3f} of the derived budget (max |dO| {d:.3e}); the two tiles differ beyond one rounding + the P quantisation"
        assert lse_d <= ATOL, f"lean vs core LSE differ by {lse_d}"
        res.update(core_max_d=d, core_ratio=ratio, core_lse_d=lse_d)
    return res, run


# ============================================================================ Rubin: the ladder
@requires_rubin
@pytest.mark.parametrize(
    "B, SKV, kv_lens, H, KH, dtype, top_k, list_kind, k, paged, block_lens",
    [
        pytest.param(1, 8192, [8192], 24, 2, torch.bfloat16, 8, "shuffled", 1, None, True, id="top8-two-sub-tiles-one-live"),
        pytest.param(1, 8192, [8192], 24, 2, torch.bfloat16, 16, "shuffled", 1, None, True, id="top16-three-sub-tiles"),
        pytest.param(1, 8192, [8192], 24, 2, torch.bfloat16, 24, "shuffled", 1, None, True, id="top24-four-sub-tiles"),
        pytest.param(1, 8192, [8192], 24, 2, torch.bfloat16, 512, "shuffled", 1, None, True, id="B1-8K-top512-65-sub-tiles"),
        pytest.param(2, 8192, [8192, 5000], 24, 2, torch.bfloat16, 512, "shuffled", 1, None, True, id="BxH-gt-1-different-lists"),
        pytest.param(2, 2052, [2052, 2052], 24, 2, torch.bfloat16, 512, "shuffled", 1, None, True, id="identity-bound-shuffled-B2"),
        pytest.param(3, 8192, [0, 8192, 300], 24, 2, torch.bfloat16, 512, "shuffled", 1, None, True, id="kv-lens-dead-sequence"),
        pytest.param(1, 8192, [8192], 24, 2, torch.bfloat16, 256, "shuffled", 1, None, False, id="top256-no-block-lens"),
        pytest.param(1, 8192, [8192], 24, 2, torch.float16, 512, "shuffled", 1, None, True, id="f16"),
        pytest.param(1, 8192, [8192], 16, 1, torch.bfloat16, 512, "shuffled", 1, None, True, id="group16-one-kv-head"),
        pytest.param(1, 8192, [8192], 2, 2, torch.bfloat16, 512, "shuffled", 1, None, True, id="group1-mha"),
        pytest.param(1, 8192, [8192], 24, 2, torch.bfloat16, 512, "shuffled", 2, None, True, id="k2-split33-dead-trailing"),
        pytest.param(1, 8192, [8192], 24, 2, torch.bfloat16, 512, "shuffled", 4, None, True, id="k4-split17-ring"),
        pytest.param(1, 8192, [8192], 24, 2, torch.bfloat16, 512, "shuffled", 3, None, True, id="k3-split22-one-refill"),
        pytest.param(4, 8192, [8192, 2052, 4097, 7000], 24, 2, torch.bfloat16, 512, "shuffled", 1, (16, True), True, id="B4-paged16-HND-bitwise-dense"),
        pytest.param(4, 8192, [8192, 2052, 4097, 7000], 24, 2, torch.bfloat16, 512, "shuffled", 2, (64, False), False, id="B4-paged64-NHD-k2-no-block-lens"),
        pytest.param(2, 8192, [8192, 300], 24, 2, torch.float16, 512, "shuffled", 1, (48, True), True, id="paged48-HND-f16"),
    ],
)
def test_lean_tile_ladder(B, SKV, kv_lens, H, KH, dtype, top_k, list_kind, k, paged, block_lens):
    res, _ = _lean_cell(B=B, SKV=SKV, kv_lens=kv_lens, H=H, KH=KH, dtype=dtype, top_k=top_k, list_kind=list_kind, k=k, paged=paged, block_lens=block_lens)
    print(
        f"\nlean ladder B={B} S_kv={SKV} lens={kv_lens} {H}/{KH} {dtype} top_k={top_k} k={k} split={res['split']} paged={paged}: exact-combine max|dO| {res['max_o']:.5f} max|dLSE| {res['max_lse']:.6f}; "
        f"pinned combine (form {res['combine_form']}) max|dO| {res['combine_max_o']:.5f}; vs unsplit tcgen05 core max|dO| {res['core_max_d']:.3e} = {res['core_ratio']:.3f} of the derived budget, |dLSE| {res['core_lse_d']:.2e}; dead rows {res['dead']}"
    )


# ============================================================================ Rubin: the degenerate matrix, row by row
def _lean_direct(q, kk, vv, ids, lens, kv_lens, top_k, scale, *, k=1):
    with process_watchdog(300, "lean degenerate cell compile + first dispatch"):
        run = LeanRun(q, (kk, vv), ids, lens, kv_lens, top_k, scale, k=k)
        run.run()
        torch.cuda.synchronize()
    return run


@requires_rubin
@pytest.mark.parametrize("L", [1, 2, 3, 4, 8, 6])
def test_rows_with_no_complete_block_and_rows_with_no_open_block(L):
    """L in {1, 2, 3}: 0 complete blocks (count 0), the tail only; L in {4, 8}: (pos_0 + 1) % 4 == 0 -> no open block, complete blocks only;
    L = 6: one complete block + a 2-key tail.  One sub-tile live, 64 dead."""
    S = _sparse_suite()
    B, H, KH, top_k, SKV = 1, 24, 2, 512, 16
    q, kk, vv, g = S._inputs(B, 1, H, KH, torch.bfloat16, 3, SKV=SKV)
    ids, lens = S._decode_lists("full", B, 1, top_k, [L], g)
    scale = 1.0 / math.sqrt(D)
    ref_o, ref_lse, _ = S._reference_decode(q, kk, vv, ids, lens, [L], scale, top_k)
    run = _lean_direct(q, kk, vv, ids, lens, [L], top_k, scale)
    _slab_audit(run, ref_lse)
    o, lse = run.combine_exact()
    max_o, max_lse = _check_outputs(o, lse, ref_o, ref_lse, what=f"L {L}")
    print(f"\nL={L}: count {int(lens[0])} complete blocks, tail {'yes' if L % 4 else 'no'}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
def test_minus_one_inside_the_count_and_entries_beyond_the_count_are_never_read():
    """A -1 at index < count is "no key" (its rows zero-filled and masked); the entries at index >= count are never read: garbage there (a
    valid-looking id, a huge id, -17) leaves the slab BITWISE unchanged."""
    S = _sparse_suite()
    B, H, KH, top_k, SKV = 1, 24, 2, 64, 2048
    q, kk, vv, g = S._inputs(B, 1, H, KH, torch.bfloat16, 5, SKV=SKV)
    ids, lens = S._decode_lists("shuffled", B, 1, top_k, [SKV], g)
    ids = ids.clone()
    lens = torch.tensor([40], dtype=torch.int32, device=ids.device)
    ids[0, 3] = -1
    ids[0, 17] = -1
    scale = 1.0 / math.sqrt(D)
    ref_o, ref_lse, _ = S._reference_decode(q, kk, vv, ids, lens, [SKV], scale, top_k)
    run = _lean_direct(q, kk, vv, ids, lens, [SKV], top_k, scale)
    o, lse = run.combine_exact()
    max_o, max_lse = _check_outputs(o, lse, ref_o, ref_lse, what="-1 inside the count")
    garbage = ids.clone()
    garbage[0, 40:] = torch.tensor([7, 2**30, -17, 511] * 6, dtype=torch.int32, device=ids.device)
    run2 = _lean_direct(q, kk, vv, garbage, lens, [SKV], top_k, scale)
    assert torch.equal(run.o_part, run2.o_part) and torch.equal(run.lse_part, run2.lse_part), "an entry beyond the count was read"
    print(f"\n-1 inside the count: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}; garbage past the count: bitwise invariant")


@requires_rubin
@pytest.mark.parametrize("bad_id", [2**29, 2**30, 2**31 - 1, -(2**31), 600, 2048])
def test_ids_past_the_length_and_extreme_ids_cannot_wrap_to_live_keys(bad_id):
    """A listed block past the sequence's length (600 at L 2051 = 512 complete blocks + 3 tail keys; 2048 = past the tensor) contributes
    nothing; an extreme id is bounded BEFORE the multiply by four (block_id_at's domain bound) -- never a live key, never an OOB read."""
    S = _sparse_suite()
    B, H, KH, top_k, SKV = 1, 24, 2, 64, 2052
    q, kk, vv, g = S._inputs(B, 1, H, KH, torch.bfloat16, 7, SKV=SKV)
    L = 2051
    ids, lens = S._decode_lists("shuffled", B, 1, top_k, [L], g)
    ids = ids.clone()
    ids[0, 5] = bad_id
    scale = 1.0 / math.sqrt(D)
    ref_o, ref_lse, _ = S._reference_decode(q, kk, vv, ids, lens, [L], scale, top_k)
    run = _lean_direct(q, kk, vv, ids, lens, [L], top_k, scale)
    o, lse = run.combine_exact()
    max_o, max_lse = _check_outputs(o, lse, ref_o, ref_lse, what=f"bad id {bad_id}")
    print(f"\nbad id {bad_id}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
def test_a_future_block_contributes_nothing_and_a_duplicate_counts_twice():
    S = _sparse_suite()
    oracle = S._oracle()
    B, H, KH, top_k, SKV = 1, 24, 2, 64, 1024
    L = 770  # pos_0 769: 192 complete blocks, tail keys 768, 769
    q, kk, vv, g = S._inputs(B, 1, H, KH, torch.bfloat16, 9, SKV=SKV)
    ids, lens = S._decode_lists("shuffled", B, 1, top_k, [L], g)
    ids = ids.clone()
    ids[0, 2] = 200  # a future block (keys 800..803 > pos_0): nothing
    ids[0, 9] = int(ids[0, 10])  # a duplicate: attended twice
    scale = 1.0 / math.sqrt(D)
    weights = S._visible_weights(ids.view(1, top_k), lens.view(1), torch.tensor([L - 1], device=ids.device), torch.tensor([L], device=ids.device), SKV, top_k)
    assert (weights == 2).any() and (weights[0, 800:804] == 0).all()
    run = _lean_direct(q, kk, vv, ids, lens, [L], top_k, scale)
    o, lse = run.combine_exact()
    # the multiplicity reference: softmax over the multiset
    kq = kk[0, :, 0].float()
    for h in range(H):
        sc = (q[0, 0, h].float() @ kk[0, :, h // (H // KH)].float().t()) * scale
        w = weights[0].double()
        p = torch.exp(sc.double() - sc.double()[w > 0].max()) * w
        ref = (p @ vv[0, :, h // (H // KH)].double()) / p.sum()
        assert float((o[0, 0, h].double() - ref).abs().max()) <= ATOL, h
        assert abs(float(lse[0, h, 0]) - float(sc.double()[w > 0].max() + torch.log(p.sum()))) <= ATOL, h
    del kq, oracle
    print("\nfuture block masked, duplicate counted twice: within ATOL of the multiplicity reference")


@requires_rubin
def test_block_lens_clamped_and_absent_and_a_negative_scale():
    S = _sparse_suite()
    B, H, KH, top_k, SKV = 2, 24, 2, 64, 4096
    q, kk, vv, g = S._inputs(B, 1, H, KH, torch.bfloat16, 11, SKV=SKV)
    kv_lens = [4096, 301]
    ids, lens = S._decode_lists("shuffled", B, 1, top_k, kv_lens, g)
    lens = torch.tensor([10**6, -5], dtype=torch.int32, device=ids.device)  # clamped to n_sel_default / 0 on device
    for scale in (1.0 / math.sqrt(D), -0.5):
        ref_o, ref_lse, _ = S._reference_decode(q, kk, vv, ids, lens, kv_lens, scale, top_k)
        run = _lean_direct(q, kk, vv, ids, lens, kv_lens, top_k, scale)
        o, lse = run.combine_exact()
        max_o, max_lse = _check_outputs(o, lse, ref_o, ref_lse, what=f"clamped block_lens scale {scale}")
        print(f"\nblock_lens clamped ([1e6, -5]) scale {scale}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")
    ref_o, ref_lse, _ = S._reference_decode(q, kk, vv, ids, None, kv_lens, 1.0 / math.sqrt(D), top_k)
    run = _lean_direct(q, kk, vv, ids, None, kv_lens, top_k, 1.0 / math.sqrt(D))
    o, lse = run.combine_exact()
    _check_outputs(o, lse, ref_o, ref_lse, what="block_lens absent")


@requires_rubin
def test_dead_chunks_and_dead_trailing_sub_tiles_write_the_combine_identity():
    """top_k 512 at L 100 (25 complete blocks, no tail: 4 live sub-tiles of 65): the 61 dead chunks (k 1) and, at k 4, the dead trailing sub-tiles
    of chunk 0 (sub-tiles 4..) plus the 16 dead chunks write exactly 0 / -inf into every slot; the live output within ATOL."""
    S = _sparse_suite()
    B, H, KH, top_k, SKV = 1, 24, 2, 512, 8192
    L = 100
    q, kk, vv, g = S._inputs(B, 1, H, KH, torch.bfloat16, 13, SKV=SKV)
    ids, lens = S._decode_lists("shuffled", B, 1, top_k, [L], g)
    scale = 1.0 / math.sqrt(D)
    ref_o, ref_lse, _ = S._reference_decode(q, kk, vv, ids, lens, [L], scale, top_k)
    for k, live_chunks in ((1, 4), (4, 1), (2, 2)):
        run = _lean_direct(q, kk, vv, ids, lens, [L], top_k, scale, k=k)
        _slab_audit(run, ref_lse)
        o5 = run.o_part.view(run.split, B, 1, H, D)
        l4 = run.lse_part.view(run.split, B, H, 1)
        assert (l4[live_chunks:] == float("-inf")).all() and (o5[live_chunks:] == 0).all(), f"k {k}: a dead chunk's slot is not the combine's identity"
        assert torch.isfinite(l4[:live_chunks]).all()
        o, lse = run.combine_exact()
        max_o, max_lse = _check_outputs(o, lse, ref_o, ref_lse, what=f"L 100 k {k}")
        print(f"\nL=100 top_k 512 k={k} split={run.split}: {run.split - live_chunks} dead chunks = identity; max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
@pytest.mark.parametrize("P, hnd", [(16, True), (64, False), (48, True)])
def test_paged_stale_rows_past_the_length_are_never_read(P, hnd):
    """The rows of a sequence's partially filled last page past its KV length hold NaN (``_paginate(..., "nan")``) and, under "other", ANOTHER
    sequence's tokens: the lean slab is BITWISE the dense read either way (zero-fill by the per-row predicate; parity with the sparse tile), and
    -1 table entries past the live pages are never dereferenced."""
    S = _sparse_suite()
    B, H, KH, top_k, SKV = 2, 24, 2, 512, 8192
    kv_lens = [8190, 2053]
    q, kk, vv, g = S._inputs(B, 1, H, KH, torch.bfloat16, 17, SKV=SKV)
    ids, lens = S._decode_lists("shuffled", B, 1, top_k, kv_lens, g)
    scale = 1.0 / math.sqrt(D)
    ref_o, ref_lse, _ = S._reference_decode(q, kk, vv, ids, lens, kv_lens, scale, top_k)
    dense = _lean_direct(q, kk, vv, ids, lens, kv_lens, top_k, scale)
    o, lse = dense.combine_exact()
    _check_outputs(o, lse, ref_o, ref_lse, what="dense")
    for fill in ("nan", "other"):
        kp, vp, table = S._paginate(kk, vv, kv_lens, P, hnd, fill, g)
        with process_watchdog(300, f"lean paged {P} {fill} compile + first dispatch"):
            pr = LeanRun(q, (kp, vp, table, P), ids, lens, kv_lens, top_k, scale)
            pr.run()
            torch.cuda.synchronize()
        assert torch.equal(pr.o_part, dense.o_part) and torch.equal(
            pr.lse_part, dense.lse_part
        ), f"paged {P} {'HND' if hnd else 'NHD'} fill {fill}: differs from the dense read"
    print(f"\npaged {P} {'HND' if hnd else 'NHD'}: NaN and other-sequence stale rows never reach the MMA (bitwise the dense slab)")


# ============================================================================ Rubin: capture / replay, the PDL arm
@requires_rubin
def test_lean_launch_is_capture_safe_and_the_pdl_arm_is_bitwise_the_plain_arm():
    S = _sparse_suite()
    B, H, KH, top_k, SKV = 2, 24, 2, 512, 8192
    kv_lens = [8192, 4097]
    q, kk, vv, g = S._inputs(B, 1, H, KH, torch.bfloat16, 19, SKV=SKV)
    ids, lens = S._decode_lists("shuffled", B, 1, top_k, kv_lens, g)
    scale = 1.0 / math.sqrt(D)
    ref_o, ref_lse, _ = S._reference_decode(q, kk, vv, ids, lens, kv_lens, scale, top_k)
    plain = _lean_direct(q, kk, vv, ids, lens, kv_lens, top_k, scale)
    o, lse = plain.combine_exact()
    _check_outputs(o, lse, ref_o, ref_lse, what="plain")
    with process_watchdog(300, "lean PDL arm compile + first dispatch"):
        pdl = LeanRun(q, (kk, vv), ids, lens, kv_lens, top_k, scale, pdl=True)
        pdl.run()
        torch.cuda.synchronize()
    assert torch.equal(pdl.o_part, plain.o_part) and torch.equal(pdl.lse_part, plain.lse_part), "the PDL arm differs from the plain arm"
    # the two-kernel chain under PDL: tile -> the pinned combine, captured and replayed with CHANGED inputs
    o_c, lse_c, _form = pdl.combine_via_kernel(torch.bfloat16)
    _check_outputs(o_c, lse_c, ref_o, ref_lse, what="pdl + combine")
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        plain.run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        plain.run()
    for value in (3.0, -1.0):
        vv.fill_(value)
        plain.o_part.fill_(-17.0)
        graph.replay()
        torch.cuda.synchronize()
        eager = _lean_direct(q, kk, vv, ids, lens, kv_lens, top_k, scale)
        assert torch.equal(plain.o_part, eager.o_part) and torch.equal(plain.lse_part, eager.lse_part), f"replay with V = {value} differs from the eager launch"
    print("\ncapture-safe (changed-input replays bitwise eager); PDL arm bitwise the plain arm")


@requires_rubin
def test_dense_rows_past_the_tensor_under_a_length_past_it_are_zero_filled_not_read():
    """Dense BSHD K / V with ``kv_lens`` PAST the tensor's extent on the LAST batch (a caller bug the device never checks: the lengths are device
    values).  The rows at or past S_kv are zero-filled by the extent clamp -- never read -- and attended as zero keys where the valid word marks
    them (keys below L_b): the oracle is the decode reference on the SAME tensors zero-padded to L_b, and the unsplit tcgen05 core (whose TMA
    clips those rows to zeros) agrees within the derived budget.  K / V are views into a buffer whose rows past S_kv hold NaN, so a read past
    the extent cannot hide in allocator padding.  A listed block fully past the tensor (2048) and the open tail block straddling it (2049:
    keys 8196, 8197 live) are both in the visible set."""
    S = _sparse_suite()
    B, H, KH, top_k, SKV = 2, 24, 2, 512, 8192
    L_over = SKV + 6  # pos_0 8197: 2049 complete blocks; the tail block 2049 = keys 8196, 8197 (past the tensor, below L_b)
    kv_lens = [4097, L_over]
    q, k0, v0, g = S._inputs(B, 1, H, KH, torch.bfloat16, 23, SKV=SKV)
    big_k = torch.full((B, SKV + 64, KH, D), float("nan"), device=q.device, dtype=k0.dtype)
    big_v = torch.full((B, SKV + 64, KH, D), float("nan"), device=q.device, dtype=v0.dtype)
    big_k[:, :SKV] = k0
    big_v[:, :SKV] = v0
    kk, vv = big_k[:, :SKV], big_v[:, :SKV]  # the tensor's extent is S_kv; the NaN rows sit exactly where an unclamped row read would land
    ids, lens = S._decode_lists("shuffled", B, 1, top_k, [4097, SKV], g)  # the builder's lists over the TENSOR's complete blocks
    ids = ids.clone()
    ids[1, 0] = 2048  # the block 8192..8195: past the tensor, below L_b (the builder never lists it: 2048 is not a block of the tensor)
    assert int((ids[1] == 2048).sum()) == 1 and int(lens[1]) == top_k
    scale = 1.0 / math.sqrt(D)
    pad_k = torch.zeros(B, L_over - SKV, KH, D, device=q.device, dtype=k0.dtype)
    pad_v = torch.zeros_like(pad_k)
    ref_o, ref_lse, pv_abs = S._reference_decode(q, torch.cat([k0, pad_k], 1), torch.cat([v0, pad_v], 1), ids, lens, kv_lens, scale, top_k)
    run = _lean_direct(q, kk, vv, ids, lens, kv_lens, top_k, scale)
    _slab_audit(run, ref_lse)
    o1, l1 = run.o_part.clone(), run.lse_part.clone()
    run.o_part.fill_(_SENTINEL)
    run.lse_part.fill_(_SENTINEL)
    run.run()
    torch.cuda.synchronize()
    assert torch.equal(o1, run.o_part) and torch.equal(l1, run.lse_part), "two launches must be bitwise"
    o, lse = run.combine_exact()
    max_o, max_lse = _check_outputs(o, lse, ref_o, ref_lse, what="dense kv_lens past the tensor (zero-padded reference)")
    core = S._launch_decode(q, (k0, v0), ids, lens, kv_lens, top_k, scale, split=1, launches=1)
    co, cl = core[0][0], core[0][1]
    live = ~torch.isinf(ref_lse).permute(0, 2, 1)
    d, ratio = S._split_vs_unsplit(o.to(torch.bfloat16).float(), co.float(), torch.bfloat16, pv_abs, live)
    assert ratio <= 1.0, f"lean vs the unsplit tcgen05 core on a length past the tensor: {ratio:.3f} of the derived budget (max |dO| {d:.3e})"
    lse_d = float((lse - cl)[~torch.isinf(ref_lse)].abs().max())
    assert lse_d <= ATOL, f"lean vs core LSE differ by {lse_d}"
    print(
        f"\ndense kv_lens {kv_lens} over S_kv {SKV}: rows past the tensor never read (NaN there); max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f} vs the "
        f"zero-padded reference; vs the tcgen05 core {ratio:.3f} of the derived budget, |dLSE| {lse_d:.2e}"
    )


@requires_rubin
def test_a_minus_one_table_entry_under_a_live_block_is_a_zero_key_attended():
    """Header contract: a -1 / out-of-range TABLE entry under a LIVE block (4 blk < L_b; a malformed table) is zero-filled but NOT masked -- the
    key is attended with score 0 over a zero V row (parity with the tcgen05 tile, whose valid word ignores the page too).  Pinned BITWISE
    against the same pool with that page's rows zeroed and the entry valid (one and the same program state), and within ATOL of the decode
    reference with those keys as zero rows."""
    S = _sparse_suite()
    B, H, KH, top_k, SKV, P = 1, 24, 2, 64, 1024, 16
    kv_lens = [1000]  # no tail; the full list = blocks 0..63 = keys 0..255 = table columns 0..15, all below L_b
    q, kk, vv, g = S._inputs(B, 1, H, KH, torch.bfloat16, 29, SKV=SKV)
    ids, lens = S._decode_lists("full", B, 1, top_k, kv_lens, g)
    scale = 1.0 / math.sqrt(D)
    kp, vp, table = S._paginate(kk, vv, kv_lens, P, True, "nan", g)
    hole = int(table[0, 3])  # the page of keys 48..63 (blocks 12..15: listed, live)
    bad_table = table.clone()
    bad_table[0, 3] = -1
    zero_kp, zero_vp = kp.clone(), vp.clone()
    zero_kp[hole] = 0
    zero_vp[hole] = 0
    with process_watchdog(300, "lean -1 table entry under a live block: compile + first dispatch"):
        a = LeanRun(q, (kp, vp, bad_table, P), ids, lens, kv_lens, top_k, scale)
        a.run()
        b = LeanRun(q, (zero_kp, zero_vp, table, P), ids, lens, kv_lens, top_k, scale)
        b.run()
        torch.cuda.synchronize()
    assert torch.equal(a.o_part, b.o_part) and torch.equal(a.lse_part, b.lse_part), "a -1 table entry under a live block must read as a page of zero rows"
    kz, vz = kk.clone(), vv.clone()
    kz[0, 48:64] = 0
    vz[0, 48:64] = 0
    ref_o, ref_lse, _ = S._reference_decode(q, kz, vz, ids, lens, kv_lens, scale, top_k)
    _slab_audit(a, ref_lse)
    o, lse = a.combine_exact()
    max_o, max_lse = _check_outputs(o, lse, ref_o, ref_lse, what="-1 table entry under a live block (zero-key reference)")
    print(f"\n-1 table entry under a live block: bitwise the zeroed page; max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f} vs the zero-key reference")


@requires_rubin
def test_a_zero_column_block_table_is_never_read_and_the_listed_blocks_are_zero_keys():
    """max_pages == 0 (a zero-column table -- its data pointer is null): the clamped table read is steered onto the ids row, every page reads -1
    (page_idx < max_pages fails) and the listed blocks are zero keys -- no out-of-bounds read, O exactly 0 and LSE = log(64) for the live
    sequence (64 valid zero keys: m = 0, l = 64), every slot written, two launches bitwise.  Kept LAST in the module: before the steering an
    unclamped read of address 0 poisons the CUDA context of the process."""
    S = _sparse_suite()
    B, H, KH, top_k, SKV, P = 1, 24, 2, 64, 256, 16
    kv_lens = [64]  # 16 complete blocks = count 16, no tail -> 64 valid keys
    q, kk, vv, g = S._inputs(B, 1, H, KH, torch.bfloat16, 31, SKV=SKV)
    ids, lens = S._decode_lists("full", B, 1, top_k, kv_lens, g)
    scale = 1.0 / math.sqrt(D)
    kp, vp, _table = S._paginate(kk, vv, kv_lens, P, True, "nan", g)
    empty = torch.empty((B, 0), dtype=torch.int32, device=q.device)
    print(f"\nzero-column table data_ptr = {empty.data_ptr()}")
    with process_watchdog(300, "lean zero-column table: compile + first dispatch"):
        run = LeanRun(q, (kp, vp, empty, P), ids, lens, kv_lens, top_k, scale)
        run.run()
        torch.cuda.synchronize()
    ref_lse = torch.full((B, H, 1), math.log(64.0), device=q.device, dtype=torch.float32)
    _slab_audit(run, ref_lse)
    o1, l1 = run.o_part.clone(), run.lse_part.clone()
    run.o_part.fill_(_SENTINEL)
    run.lse_part.fill_(_SENTINEL)
    run.run()
    torch.cuda.synchronize()
    assert torch.equal(o1, run.o_part) and torch.equal(l1, run.lse_part), "two launches must be bitwise"
    o, lse = run.combine_exact()
    assert (o == 0).all(), "every V row is zero-filled: O must be exactly 0"
    assert torch.allclose(lse, ref_lse, atol=1e-5, rtol=0), f"64 zero keys: LSE must be log(64), got {lse.flatten()[:4].tolist()}"
    print("zero-column table: no read of it, O = 0, LSE = log(64) on every row, slab complete, two launches bitwise")
