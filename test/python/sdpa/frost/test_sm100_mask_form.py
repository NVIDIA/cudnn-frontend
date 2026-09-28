# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sm100 masked softmax arms mask through ONE tile_dsl op, ``tile_dsl.mask.apply_mask_chunk``.

Every masked call site of the twelve sm100 prefill kernels, the d128 decode kernel and the d512 backward stage-2
kernel calls ``apply_mask_chunk(...)`` directly: one 32-column keep-word from two saturating shifts, then a
register-to-predicate ``R2P`` + one ``FSEL`` per cell, independent of the number of active terms.  It replaced a
per-cell compare + select (one ``IADD`` + ``ISETP`` + ``FSEL`` per cell per mask term) -- first behind a per-kernel
``MASK_FORM`` constant (#1197 on this arch line), then collapsed into the op itself once both arch lines shipped it.
Same masked set, same sentinel, same ``arith.select`` -> O / LSE (and, on the backward, S and every gradient
downstream) are bitwise identical; the forms differed ONLY in instruction count.  What is left to pin: every masked
site calls the op DIRECTLY (no dispatcher, no ``form=``, no module constant a kernel could flip back), and the
lowering is the register-to-predicate one.

Host-only where possible: the site scan loads each module through the adapter's own loaders (no GPU, no compile); the
SASS pins trace-compile two masked builds for sm_100a (``CUTE_DSL_ARCH`` needs no matching device) and skip when no
nvdisasm on ``$CUDA_PATH/bin`` or ``$PATH`` decodes the cubin.  The Rubin twins live in ``test_sdpa_fwd_dsl_sm107.py``.
"""

import re
from dataclasses import replace

import pytest

from frost_test_utils import MASK_SASS_OPCODE_COUNTS, assert_no_new_spills, requires_dsl, run_sass_probe, sass_probe_source

from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module, supported_cgas_for
from cudnn.sdpa.fwd.config_sm100 import TemplateParams, canonicalize_d192_lowering

pytestmark = [pytest.mark.L0, requires_dsl]

_E4M3, _BF16 = 0, 2

# Every sm100 kernel with a masked softmax arm: (quantization kind, flavor) -> (the kernel file, the number of
# apply_mask_chunk sites it carries).  "decode" is the d128 f16 flavor at TILE_CGA_M=1 (the decode tile,
# `sm100/decode_d128_f16.py`); "bwd" is the d512 backward's stage-2 kernel.  The site count is the migration table of
# the port: a count that moves means an arm was added or dropped and the table (and this pin) must say so.
_KERNELS = {
    ("f16", (128, 128)): ("prefill_d128_f16.py", 1),
    ("f16", (192, 128)): ("prefill_d192_d128_f16.py", 5),
    ("f16", (256, 256)): ("prefill_d256_f16.py", 2),
    ("f16", (512, 512)): ("prefill_d512_f16.py", 1),
    ("fp8", (128, 128)): ("prefill_d128_fp8.py", 1),
    ("fp8", (192, 128)): ("prefill_d192_d128_fp8.py", 2),
    ("fp8", (256, 256)): ("prefill_d256_fp8.py", 4),
    ("fp8", (512, 512)): ("prefill_d512_fp8.py", 1),
    ("mxfp8", (128, 128)): ("prefill_d128_mxfp8.py", 2),
    ("mxfp8", (192, 128)): ("prefill_d192_d128_mxfp8.py", 3),
    ("mxfp8", (256, 256)): ("prefill_d256_mxfp8.py", 3),
    ("mxfp8", (512, 512)): ("prefill_d512_mxfp8.py", 3),
    ("decode", (128, 128)): ("decode_d128_f16.py", 1),
    ("bwd", (512, 512)): ("bprop_d512_f16.py", 1),
}
_KERNEL_IDS = [f"{k}-d{f[0]}" if f[0] == f[1] else f"{k}-d{f[0]}x{f[1]}" for k, f in _KERNELS]

# The switch to the bit-word form, sm_100a listings of both forms (2026-09-22, cutlass-dsl 4.8.0): whole-kernel ISETP 598 -> 94
# on the fp8 d128 causal + padded build and 590 -> 86 on the f16 d512 causal + SWA 640 build, R2P 0 -> 32 on both, the masked
# KV-tile body 1191 -> 723 / 1172 -> 719 instructions per lane while the dense bodies keep their 552 / 530.  An A/B against
# the per-cell form is a branch that re-adds it to tile_dsl; the shipped tree has one op.


def _load(kind, flavor):
    """Load one sm100 kernel module through the loader the adapter itself uses for that kernel (no GPU, no compile)."""
    if kind == "bwd":
        from cudnn.frost.template_loader import load_template
        from cudnn.sdpa.bwd.api_dsl import _SM100_STAGE2_FILE, _sm100_kernel_path
        from cudnn.sdpa.bwd.config_sm100 import TemplateParams as BwdTemplateParams

        return load_template(_sm100_kernel_path(_SM100_STAGE2_FILE), BwdTemplateParams(), tag="sdpa_bwd_sm100_stage2")
    if kind == "decode":
        # TILE_CGA_M=1 on the d128 f16 / bf16 flavor IS the decode tile (config_sm100.CfgD128Decode; api_dsl routes it).
        return _load_sm100_kernel_module(flavor, TemplateParams(dtype_qkv=_BF16, dtype_o=_BF16, cta_mma=1), fp8=False, pertensor=False, rubin=False)
    fp8, pertensor = kind != "f16", kind == "fp8"
    # The prefill tile at the widest CTA geometry the adapter serves on cc 10.0: cga2 where both widths exist (cga1 on
    # the d128 f16 flavor would select the decode kernel above), cga1 where that is the only width (fp8 d256, mxfp8 d512).
    cta_mma = max(supported_cgas_for(flavor, fp8=fp8, device_cc=(10, 0), pertensor=pertensor))
    params = TemplateParams(dtype_qkv=_E4M3 if fp8 else _BF16, dtype_o=_BF16, cta_mma=cta_mma)
    return _load_sm100_kernel_module(flavor, params, fp8=fp8, pertensor=pertensor, rubin=False)


def _code_lines(src):
    """Source with whole-line comments dropped -- the prose in these modules quotes the spellings the scans look for."""
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


@pytest.mark.parametrize("kind,flavor", list(_KERNELS), ids=_KERNEL_IDS)
def test_sm100_every_mask_site_calls_apply_mask_chunk(kind, flavor):
    """Every masked call site is a direct `apply_mask_chunk(` call, counted against the migration table above (a count
    that moves means an arm was added or dropped and the table must say so), on the module the adapter really loads for
    that kernel (the file name is checked, so a loader routing to a sibling cannot pass).  No dispatcher, no `form=`
    kwarg and no module `MASK_FORM` constant: a reintroduced per-kernel selector would let one arm (the d256 and mxfp8
    kernels have 2-4 masked arms each) drift to a slower lowering with bitwise-identical output, which no numerics test
    sees; the lowering itself is held by test_sm100_masked_softmax_sass_is_register_to_predicate."""
    mod = _load(kind, flavor)
    file, n_expected = _KERNELS[(kind, flavor)]
    assert mod.__file__.endswith(file), f"{mod.__name__}: loaded {mod.__file__}, expected {file}"
    assert not hasattr(mod, "MASK_FORM"), f"{mod.__name__}: a MASK_FORM constant is back"
    with open(mod.__file__, encoding="utf-8") as fh:
        code = _code_lines(fh.read())
    n_sites = len(re.findall(r"\bapply_mask_chunk\(", code))
    assert n_sites == n_expected, f"{mod.__name__}: {n_sites} apply_mask_chunk site(s), the migration table says {n_expected} -- update both"
    for spelling in (r"\bapply_mask_chunk_form\b", r"\bapply_mask_chunk_bits\b", r"\bMASK_FORM", r"(?<!\w)form="):
        assert not re.search(spelling, code), f"{mod.__name__}: {spelling!r} -- the per-kernel mask-form selector was collapsed into apply_mask_chunk"


@pytest.mark.parametrize("rubin", [False, True], ids=["sm100", "sm107"])
def test_sm100_d192_dense_fp8_window_sentinel_is_inside_the_bits_domain(rubin):
    """`canonicalize_d192_lowering` lowers the DENSE per-tensor FP8 d192x128 plan as MASK_CAUSAL with a right band no
    sequence reaches (the DSL 4.7 MASK_NONE workaround).  That band is the compile-time `window_right` at the kernel's
    `apply_mask_chunk` sites, so it has to sit INSIDE the bit-word mask op's Int32 domain: from `MASK_BOUND_LIMIT` on the
    op raises at trace time, `engine.build_plan` turns that into a typed decline, and the FROST fp8 row silently drops
    out of every dense d192x128 graph on both arch lines (the sentinel used to be `1 << 30` == the limit).  Pinned on the
    canonicalized params and on the loaded module's CFG for the SM100 and the SM107 kernel file; no GPU, no compile."""
    from cudnn.frost.tile_dsl.mask import MASK_BOUND_LIMIT, MASK_CAUSAL

    dense = TemplateParams(dtype_qkv=_E4M3, dtype_o=_E4M3, cta_mma=2)
    params = canonicalize_d192_lowering(dense, pertensor=True, s_q=8192, s_kv=8192)
    assert (
        params.window_right is not None and params.window_right < MASK_BOUND_LIMIT
    ), f"window_right={params.window_right} is not below MASK_BOUND_LIMIT={MASK_BOUND_LIMIT}"
    assert params.window_right >= 1 << 28, f"window_right={params.window_right} no longer exceeds every dense d192 sequence a TMA coordinate can address"
    mod = _load_sm100_kernel_module((192, 128), params, fp8=True, pertensor=True, rubin=rubin)
    assert mod.__file__.endswith("prefill_d192_d128_fp8.py") and (("/sm107/" in mod.__file__) == rubin), mod.__file__
    assert int(mod.CFG.MASK_FLAGS) & MASK_CAUSAL, "the dense fp8 d192 plan is expected to lower through the causal arm"
    assert int(mod.CFG.WINDOW_RIGHT) == params.window_right < MASK_BOUND_LIMIT
    # a graph-stated right band is passed through untouched, and the canonicalization stays confined to the dense plan
    banded = canonicalize_d192_lowering(replace(dense, window_right=64), pertensor=True, s_q=8192, s_kv=8192)
    assert banded.window_right == 64


# ============================================================================ sm_100a SASS pin: the masked arm is R2P + FSEL, not ISETP + FSEL per cell
# Compiles a masked specialization for sm_100a (no Blackwell device needed) at the PRODUCTION geometry from the adapter itself:
# the CTA width supported_cgas_for serves for the flavor on cc 10.0 (one width for both pinned flavors -- if that ever widens,
# the unpack fails here and the row has to say which geometry it pins) and the cc 10.0 exp2-split record api_dsl.template_params()
# builds for the (kind, flavor).  The dtypes, the mask specialization (window_left / window_right / seq_kv_lens_present) and the
# compile() kwargs come from the row.
_SM100_MASK_SASS_PROBE = sass_probe_source(
    """
    from cudnn.sdpa.fwd.api_dsl import _exp2_fma_split_for
    kind, d, compile_kw = params_kw.pop("kind"), params_kw.pop("d"), params_kw.pop("compile")
    fp8, pertensor = kind != "f16", kind == "fp8"
    (cta_mma,) = supported_cgas_for((d, d), fp8=fp8, device_cc=(10, 0), pertensor=pertensor)
    params = TemplateParams(cta_mma=cta_mma, exp2_fma_split=_exp2_fma_split_for((10, 0), kind=kind, flavor=(d, d)), **params_kw)
    mod = _load_sm100_kernel_module((d, d), params, fp8=fp8, pertensor=pertensor, rubin=False)
    print("MASK_FLAGS", mod.CFG.MASK_FLAGS)
    mod.compile(**compile_kw)
    """,
    counts=MASK_SASS_OPCODE_COUNTS,
)

# The two builds the Rubin pin also covers, at their sm100 chart-layer geometry: the fp8 d128 causal + padded build (E4M3 in /
# E4M3 out, GQA 64/8, Stats + Amax_O, S=8K -- the llama layer) and the f16 d512 causal + SWA 640 build (bf16, S envelope 512,
# Stats -- the largest masked win of the Rubin study).  Rows: (kind, d, TemplateParams fields, compile() kwargs, pins).
# The pins are the counts MEASURED on the bit-word cubins (2026-09-22, sm_100a, cutlass-dsl 4.8.0 + the CUDA 13.5 ptxas); the
# per-cell cubins of the pre-port tree read, in the same order (ISETP / predicate-to-register moves / STL / LDL):
# fp8-d128-causal_padded 598 / 0 / 5 / 9 and f16-d512-causal_swa640 590 / 0 / 0 / 0, with 0 R2P on both (the d128 causal
# builds carry their 5 STL / 9 LDL on develop already -- that is the row's pre-existing count).  One masked arm falling back
# to per-cell compares adds >= 128 ISETP, so _ISETP_SLACK = 32 still catches a single arm; STL / LDL are bounds
# (frost_test_utils.SPILL_TOLERANCE), never literals.
_ISETP_SLACK = 32
_PRED2GPR_SLACK = 6
_SM100_MASK_SASS_ROWS = [
    pytest.param(
        "fp8",
        128,
        {"dtype_qkv": _E4M3, "dtype_o": _E4M3, "qh_per_kh": 8, "emit_amax_o": True, "window_right": 0, "seq_kv_lens_present": True},
        {"b": 1, "qh": 64, "kh": 8, "sq": 8192, "skv": 8192, "has_lse": True},
        {"ISETP": 94, "PRED2GPR": 0, "STL": 5, "LDL": 9},
        id="fp8-d128-causal_padded",
    ),
    pytest.param(
        "f16",
        512,
        {"dtype_qkv": _BF16, "dtype_o": _BF16, "window_right": 0, "window_left": 640},
        {"d_qk": 512, "d_v": 512, "has_lse": True},
        {"ISETP": 86, "PRED2GPR": 0, "STL": 0, "LDL": 0},
        id="f16-d512-causal_swa640",
    ),
]


@pytest.mark.parametrize("kind, d, fields, compile_kw, pins", _SM100_MASK_SASS_ROWS)
def test_sm100_masked_softmax_sass_is_register_to_predicate(tmp_path, kind, d, fields, compile_kw, pins):
    """The masked softmax arm masks through R2P + FSEL (the bit-word form of apply_mask_chunk), not one ISETP + FSEL per cell per term: R2P > 0,
    whole-kernel ISETP within the measured ceiling, no predicate-register spill storm (predicate-to-register moves) and no
    new stack spills (STL / LDL within SPILL_TOLERANCE of this toolchain's count).  Compiled for sm_100a at the production
    geometry; skips when no nvdisasm decodes the cubin."""
    probe = run_sass_probe(
        tmp_path, probe_src=_SM100_MASK_SASS_PROBE, arch="sm_100a", params={"kind": kind, "d": d, "compile": compile_kw, **fields}, tag=f"mask_{kind}_d{d}"
    )
    assert probe.expect["MASK_FLAGS"] != 0, "the probe params select the DENSE specialization -- no masked arm is compiled in"
    stats = probe.stats
    assert stats["R2P"] > 0, "no R2P: the masked arm is back to per-cell compare + select (tile_dsl.mask.apply_mask_chunk regressed)"
    assert (
        stats["ISETP"] <= pins["ISETP"] + _ISETP_SLACK
    ), f"{stats['ISETP']} ISETP > {pins['ISETP']} + {_ISETP_SLACK}: a masked arm is comparing per cell again"
    assert (
        stats["PRED2GPR"] <= pins["PRED2GPR"] + _PRED2GPR_SLACK
    ), f"{stats['PRED2GPR']} predicate-to-register moves > {pins['PRED2GPR']} + {_PRED2GPR_SLACK}: predicate registers are spilling into GPRs"
    assert_no_new_spills(stats, pins, f"[{kind} d={d}] ")
