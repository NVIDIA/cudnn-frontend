# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM90 native D128 forward template (issue #378), through the direct adapter.

``SdpaFwdDslSm90`` runs ``sm90/prefill_d128_f16.py`` for an exact (128, 128) dense declaration
it serves (``config_sm90.pick_flavor``). The engine row does not offer (128, 128) graphs yet, so
every case here builds the adapter directly and asserts the D128 flavor before it trusts a number.
O and Stats start as NaN, so a row the kernel never writes cannot pass. Tolerances are the SM90
suite's own (O 5e-2 / 3e-2, Stats 2e-2 / 2e-2 against the fp32 reference) except where the
arithmetic is the same and the comparison is ``torch.equal``.
"""

import math
import subprocess
import sys
import textwrap

import pytest
import torch

from test_utils import torch_fork_set_rng

from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
from frost_test_utils import _SM, assert_no_new_spills, nvdisasm_candidates, requires_dsl
from test_sdpa_fwd_dsl_sm100 import _bhsd, _ref_sdpa_full

pytestmark = [
    pytest.mark.skipif(_SM != 90, reason="needs an SM90 Hopper GPU, have " + ("none" if _SM is None else f"sm_{_SM}")),
    requires_dsl,
]


# Levels (test/AGENTS.md: L0 stays fast, big sweeps go to higher levels). Every test carries its own level mark, since a
# module-level L0 would select every parameter of a sweep. The sweeps below keep one or a few cells per record type in L0
# (the one-tile kernel, the persistent loop, the paired walk, the NATURAL stride, the claim counter) and run the rest of
# their forced-grid cells, the other SASS probes and the other watchdogged children at L1.
_L0_CROSS = frozenset({"ring-wrap", "causal-br", "gqa-causal"})
_L0_WALK = frozenset({"pairs-t10", "middle-light-lone"})
_L0_STRIDE = frozenset({("t1-tl-mha", 2), ("t1-tl-mha", 3), ("t8-br-gqa", 2), ("t8-br-gqa", 5)})
_L0_CLAIM = frozenset({"t8-br-gqa", "t32-tl-gqa"})
_L0_SASS = frozenset({"pc1-causal"})  # carries the claim ring's release-dependence pin
_L0_WATCHDOG = frozenset({"pa"})


def _level(l0):
    return pytest.mark.L0 if l0 else pytest.mark.L1


_D = 128
_SCALE = 1.0 / math.sqrt(_D)
_D128_FILE = "sm90/prefill_d128_f16.py"
_DTYPES = (torch.float16, torch.bfloat16)


def _inputs(b, h_q, h_kv, s_q, s_kv, dtype):
    """Q, K, V as BHSD views over BSHD storage (the notebook convention)."""
    return _bhsd(b, h_q, s_q, _D, dtype), _bhsd(b, h_kv, s_kv, _D, dtype), _bhsd(b, h_kv, s_kv, _D, dtype)


def _plan(q, k, v, *, stats=False, **kw):
    """One compiled D128 plan over NaN-filled O (and Stats); asserts the adapter chose D128."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    b, h_q, s_q, d = q.shape
    o = torch.full((b, s_q, h_q, d), float("nan"), dtype=q.dtype, device="cuda").transpose(1, 2)
    lse = torch.full((b, h_q, s_q), float("nan"), dtype=torch.float32, device="cuda") if stats else None
    api = SdpaFwdDslSm90(q, k, v, o, sample_lse=lse, **kw)
    assert api.check_support() and api.flavor == (128, 128), api.flavor
    api.compile()
    return api, o, lse


def _run(q, k, v, *, stats=False, **kw):
    """Plan, execute once, and return (O, Stats or None)."""
    api, o, lse = _plan(q, k, v, stats=stats, **kw)
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)
    torch.cuda.synchronize()
    return o, lse


def _check(o, lse, q, k, v, *, scale=_SCALE, is_causal=False, bottom_right=False):
    o_ref, lse_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=is_causal, bottom_right=bottom_right, return_stats=True)
    assert not torch.isnan(o).any(), "an O element kept its NaN prefill: a row or column was never stored"
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)
    if lse is not None:
        assert not torch.isnan(lse).any(), "a Stats row kept its NaN prefill"
        torch.testing.assert_close(lse, lse_ref, atol=2e-2, rtol=2e-2)


# --- Routing -------


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_d128_compiles_its_own_template(monkeypatch):
    """The direct adapter loads exactly the D128 file under its own tag, and the result is right."""
    from cudnn.sdpa.fwd import api_dsl

    loaded, load = [], api_dsl._load_kernel_template
    monkeypatch.setattr(
        api_dsl,
        "_load_kernel_template",
        lambda filename, params, tag="template", part=None: loaded.append((filename, tag)) or load(filename, params, tag, part=part),
    )
    q, k, v = _inputs(2, 4, 4, 96, 160, torch.float16)
    o, lse = _run(q, k, v)
    assert loaded == [(_D128_FILE, "sdpa_fwd_sm90_d128")], loaded
    _check(o, lse, q, k, v)


# --- Tiles: whole, partial and single, Q and KV -------
#
# A CTA is 128 query rows (two 64-row warpgroups) by 128 keys, and K and V ride a two-stage ring.
# The shapes cover a Q tile whose second warpgroup has no row (S_q <= 64), a partial row in either
# warpgroup, a KV walk of one tile (fewer tiles than stages), and partial last KV tiles.

_TILE_SHAPES = [
    (2, 4, 128, 128),
    (2, 4, 256, 384),
    (1, 2, 1, 1),
    (1, 2, 2, 63),
    (2, 2, 17, 65),
    (1, 2, 64, 128),
    (1, 2, 65, 129),
    (2, 2, 127, 200),
    (1, 2, 129, 1000),
    (1, 2, 255, 64),
    (1, 1, 1000, 1000),
    (2, 2, 70, 100),
]


@pytest.mark.L0
@pytest.mark.parametrize("i, shape", list(enumerate(_TILE_SHAPES)), ids=[f"b{b}h{h}-sq{sq}-skv{skv}" for b, h, sq, skv in _TILE_SHAPES])
@torch_fork_set_rng(seed=0)
def test_d128_unmasked_tiles_match_the_reference(i, shape):
    """Unmasked MHA; dtype and Stats rotate with the index. (2, 2, 70, 100) is the SM90 suite's
    folded-mask case: an unmasked partial KV tile must keep its padding P a runtime value."""
    b, h, s_q, s_kv = shape
    q, k, v = _inputs(b, h, h, s_q, s_kv, _DTYPES[i % 2])
    o, lse = _run(q, k, v, stats=bool(i // 2 % 2))
    _check(o, lse, q, k, v)


# --- Causal -------

_CAUSAL_CASES = [
    pytest.param(256, 256, False, id="top_left-square"),
    pytest.param(300, 200, False, id="top_left-sq_above_skv"),
    pytest.param(130, 700, True, id="bottom_right-sq_below_skv"),
    pytest.param(129, 129, True, id="bottom_right-square-partial"),
    pytest.param(1, 300, True, id="bottom_right-one_row"),
]


@pytest.mark.L0
@pytest.mark.parametrize("s_q, s_kv, bottom_right", _CAUSAL_CASES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp16", "bf16"])
@torch_fork_set_rng(seed=0)
def test_d128_causal_matches_the_reference(s_q, s_kv, bottom_right, dtype):
    """Causal with Stats: the reverse walk's masked right boundary, unmasked interior and the
    partial-Q-tile arm, for both diagonal anchors."""
    q, k, v = _inputs(2, 4, 4, s_q, s_kv, dtype)
    o, lse = _run(q, k, v, stats=True, is_causal=True, causal_bottom_right=bottom_right)
    # The adapter spells top-left on a dense square as bottom-right (identical there).
    _check(o, lse, q, k, v, is_causal=True, bottom_right=bottom_right or s_q == s_kv)


@pytest.mark.L0
def test_d128_unpadded_bottom_right_above_skv_still_declines():
    """D128 keeps the adapter's pending case: unpadded bottom-right with S_q > S_kv."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    q, k, v = _inputs(1, 2, 2, 200, 100, torch.float16)
    o = torch.empty_like(q)
    with pytest.raises(NotImplementedError, match="unpadded bottom-right S_q > S_kv is not yet supported on SM90"):
        SdpaFwdDslSm90(q, k, v, o, is_causal=True, causal_bottom_right=True).check_support()


# --- GQA and MQA: one KV-head division -------


@pytest.mark.L0
@pytest.mark.parametrize("h_q, h_kv", [(8, 2), (8, 1), (12, 3)], ids=["gqa4", "mqa", "gqa4-h12"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=0)
def test_d128_gqa_is_bitwise_repeated_mha(h_q, h_kv, causal):
    """Each CTA does the same arithmetic for GQA as for MHA over repeated K/V heads, so O and
    Stats are equal bit for bit; any difference is a KV-head mapping bug."""
    q, k, v = _inputs(2, h_q, h_kv, 200, 333, torch.bfloat16)
    o, lse = _run(q, k, v, stats=True, is_causal=causal)
    group = h_q // h_kv
    o_mha, lse_mha = _run(q, k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1), stats=True, is_causal=causal)
    assert torch.equal(o, o_mha) and torch.equal(lse, lse_mha)
    _check(o, lse, q, k, v, is_causal=causal)


# --- Stats and scales -------


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_d128_stats_log2_is_the_natural_log_times_log2e():
    """``stats_log2`` scales each natural-log row LSE by log2(e) once, at the store; O is unchanged."""
    q, k, v = _inputs(2, 4, 2, 150, 260, torch.float16)
    o, lse = _run(q, k, v, stats=True, is_causal=True)
    o2, lse2 = _run(q, k, v, stats=True, is_causal=True, stats_log2=True)
    assert torch.equal(o, o2)
    torch.testing.assert_close(lse2, lse * math.log2(math.e), atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize("scale", [-0.25, 0.0, 1.0], ids=["negative", "zero", "unit"])
@torch_fork_set_rng(seed=0)
def test_d128_scale_signs(scale):
    """Each sign compiles its own probability path: a negative scale anchors each row at its
    minimum and always takes the FP64 exponents, a zero scale weighs every visible key equally,
    and a unit scale makes the softmax sharp. Causal with Stats, so masked rows are covered."""
    q, k, v = _inputs(2, 4, 2, 130, 200, torch.float16)
    o, lse = _run(q, k, v, stats=True, is_causal=True, scale_softmax=scale)
    _check(o, lse, q, k, v, scale=scale, is_causal=True)


# --- Invariance: schedulers, repeats, layouts -------


@torch_fork_set_rng(seed=0)
def test_d128_schedulers_are_bitwise_equal():
    """NATURAL, LPT and LPT_L2 launch the same tiles in different orders over ~4 waves on 132 SMs;
    each tile's arithmetic is identical, so outputs are equal bit for bit."""
    q, k, v = _inputs(4, 16, 4, 1024, 1024, torch.bfloat16)
    runs = [_run(q, k, v, stats=True, is_causal=True, sched_policy=policy) for policy in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2)]
    for o, lse in runs[1:]:
        assert torch.equal(o, runs[0][0]) and torch.equal(lse, runs[0][1])
    _check(*runs[0], q, k, v, is_causal=True)


@torch_fork_set_rng(seed=0)
def test_d128_repeat_launches_are_bitwise_equal():
    """Two launches of one plan over the same inputs agree bit for bit (no race, no stale stage)."""
    q, k, v = _inputs(2, 8, 2, 700, 900, torch.float16)
    api, o, lse = _plan(q, k, v, stats=True, is_causal=True)
    outs = []
    for _ in range(2):
        o.fill_(float("nan"))
        lse.fill_(float("nan"))
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)
        torch.cuda.synchronize()
        outs.append((o.clone(), lse.clone()))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


_LAYOUTS = {
    "bhsd": (0, 1, 2, 3),
    "bshd": (0, 2, 1, 3),
    "sbhd": (2, 0, 1, 3),
}


def _laid_out(t, order):
    """The same BHSD values in storage whose axes run in ``order`` (outermost first)."""
    return t.permute(*order).contiguous().permute(*[order.index(axis) for axis in range(4)])


@pytest.mark.L0
@pytest.mark.parametrize("layout", list(_LAYOUTS), ids=list(_LAYOUTS))
@torch_fork_set_rng(seed=0)
def test_d128_layouts_are_bitwise_equal(layout):
    """Every dense_flex storage order binds natively (the maps carry the strides); the values,
    and so the arithmetic, are the same, so O and Stats equal the BHSD-contiguous run bit for bit."""
    q, k, v = _inputs(2, 4, 2, 100, 140, torch.float16)
    base = _run(*(_laid_out(t, _LAYOUTS["bhsd"]) for t in (q, k, v)), stats=True, is_causal=True)
    other = _run(*(_laid_out(t, _LAYOUTS[layout]) for t in (q, k, v)), stats=True, is_causal=True)
    assert torch.equal(base[0], other[0]) and torch.equal(base[1], other[1])


# --- Tight numerics against FP64 -------


def _ref64(q, k, v, *, scale, is_causal, bottom_right):
    """FP64 O and natural-log Stats (no TF32 anywhere: both matmuls run in double)."""
    b, h_q, s_q, _ = q.shape
    group = h_q // k.shape[1]
    kk, vv = (t.repeat_interleave(group, dim=1).double() for t in (k, v))
    scores = torch.matmul(q.double(), kk.transpose(-1, -2)) * scale
    if is_causal:
        s_kv = k.shape[2]
        rows = torch.arange(s_q, device=q.device).view(-1, 1) + (s_kv - s_q if bottom_right else 0)
        scores = scores.masked_fill(torch.arange(s_kv, device=q.device).view(1, -1) > rows, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), vv), torch.logsumexp(scores, dim=-1)


@pytest.mark.L0
@pytest.mark.parametrize("shape", [(2, 8, 2, 256, 256), (1, 8, 8, 2000, 1000)], ids=["b2-gqa-s256", "b1-mha-sq2000-skv1000"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp16", "bf16"])
@torch_fork_set_rng(seed=0)
def test_d128_fp64_bounds(shape, causal, dtype):
    """|O - O64| <= u * (|O64| + max|V|) with a 2x margin, u the input dtype's unit roundoff (P
    is rounded to it for the 16-bit WGMMA and O is stored in it); Stats within 1e-3 + 1e-5 rel."""
    b, h_q, h_kv, s_q, s_kv = shape
    q, k, v = _inputs(b, h_q, h_kv, s_q, s_kv, dtype)
    o, lse = _run(q, k, v, stats=True, is_causal=causal)
    o64, lse64 = _ref64(q, k, v, scale=_SCALE, is_causal=causal, bottom_right=s_q == s_kv)
    u = 2.0**-8 if dtype == torch.bfloat16 else 2.0**-11
    bound = 2 * u * (o64.abs() + v.double().abs().max())
    excess = (o.double() - o64).abs() - bound
    assert excess.max().item() <= 0, f"O exceeds its FP64 bound by up to {excess.max().item():.3e}"
    torch.testing.assert_close(lse.double(), lse64, atol=1e-3, rtol=1e-5)


# --- Large scores: the FP32 exponent rounding -------

# Q and K hold one bf16 constant a = 1.5 * 2**k, so every score is M = 128 * a * a: exact in bf16 products and in every
# FP32 partial sum (at most 11 significant bits), whatever order the WGMMA adds them in.
_LARGE_SCORE_AMPS = [1.5 * 2.0**k for k in (0, 6, 12, 14, 16, 17)]


def _row_max_id(a):
    mantissa, exponent = f"{128 * a * a:.1e}".split("e")
    return f"max-{mantissa}e{int(exponent)}"


@pytest.mark.L0
@pytest.mark.parametrize("a", _LARGE_SCORE_AMPS, ids=[_row_max_id(a) for a in _LARGE_SCORE_AMPS])
@torch_fork_set_rng(seed=0)
def test_d128_large_scores_keep_rows_finite_and_nonzero(a):
    """Every score equals M = 128 * a * a, so the softmax is uniform: O is the mean of V over the 256 keys and Stats is
    M * scale + ln(256). The FP32 path forms each exponent as one FFMA, s * k - c with k = scale * log2(e) and c = M * k
    rounded once per row, so the top score's exponent is c's rounding error instead of 0: about -16 at M = 4.8e9, -256 or
    less from M = 7.7e10 on, where every P underflows and normalize_row takes its empty-row branch (O = 0, Stats = -inf).
    With the row limit at 2**24 / |k| (about 1.3e8 here) every case from M = 4.8e9 on takes the FP64 exponents instead;
    with only the old overflow limit (2**126) the three largest cases fail. This test is the detector."""
    q, k, v = _inputs(1, 2, 2, 128, 256, torch.bfloat16)
    q.fill_(a)
    k.fill_(a)
    o, lse = _run(q, k, v, stats=True)
    o64, lse64 = _ref64(q, k, v, scale=_SCALE, is_causal=False, bottom_right=False)
    bad_o = (~torch.isfinite(o)).any(dim=-1)
    zero_o = (o == 0).all(dim=-1)
    bad_lse = ~torch.isfinite(lse)
    assert not (bad_o.any() or zero_o.any() or bad_lse.any()), (
        f"row max {128 * a * a:.3e}: {int(bad_o.sum())} rows with a non-finite O, {int(zero_o.sum())} all-zero O rows "
        f"(V is random, the exact O is its mean), {int(bad_lse.sum())} non-finite Stats of {lse.numel()} rows"
    )
    u = 2.0**-8
    excess = (o.double() - o64).abs() - 2 * u * (o64.abs() + v.double().abs().max())
    assert excess.max().item() <= 0, f"row max {128 * a * a:.3e}: O exceeds its FP64 bound by up to {excess.max().item():.3e}"
    torch.testing.assert_close(lse.double(), lse64, atol=0.0, rtol=1e-5)


# --- Code generation: the register split, spills and WGMMA serialization -------

_SASS_PROBE = textwrap.dedent(r"""
    import glob, os, subprocess, sys
    dump, ptxas, cands = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_ARCH"] = "sm_90a"  # unconditional: an inherited value would pin another target
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template
    from cudnn.sdpa.fwd.config_sm90 import TemplateParams
    # The heaviest D128 body: causal (three masked phases), GQA and Stats.
    params = TemplateParams(dtype_qkv=DTYPE_BF16, causal=True, has_lse=True, qh_per_kh=4)
    mod = _load_kernel_template("sm90/prefill_d128_f16.py", params, tag="sass_sm90_d128")
    b, h_q, h_kv, s = 2, 8, 2, 1024
    q, kv = (s * h_q * 128, 128, h_q * 128, 1), (s * h_kv * 128, 128, h_kv * 128, 1)
    mod.compile(b, h_q, h_kv, s, s, q, kv, kv, q, (h_q * s, s, 1))
    cubins = sorted(glob.glob(os.path.join(dump, "**", "*.cubin"), recursive=True), key=os.path.getmtime)
    ptxs = sorted(glob.glob(os.path.join(dump, "**", "*.ptx"), recursive=True), key=os.path.getmtime)
    if not cubins or not ptxs:
        print("FAIL no cubin/ptx dumped into", dump, os.listdir(dump)); sys.exit(3)
    nvd = None
    for c in cands:
        try:
            proc = subprocess.run([c, "-c", cubins[-1]], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            print("REJECT", c, "->", repr(exc)); continue
        if proc.returncode == 0 and proc.stdout.strip():
            nvd = c; break
        print("REJECT", c, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if nvd is None:
        print("SKIP no nvdisasm candidate decodes the cubin"); sys.exit(0)
    sass = subprocess.run([nvd, "-c", cubins[-1]], capture_output=True, text=True, check=True).stdout.splitlines()
    for key in ("USETMAXREG", "STL", "LDL", "HGMMA"):
        print("SASS", key, sum(1 for ln in sass if key in ln))
    alloc = [int(ln.split(",")[-1].strip(" ;"), 16) for ln in sass if "USETMAXREG.TRY_ALLOC" in ln]
    dealloc = [int(ln.split()[-2], 16) for ln in sass if "USETMAXREG.DEALLOC" in ln]
    print("SASS USETMAXREG_ALLOC", alloc[0] if alloc else -1)
    print("SASS USETMAXREG_DEALLOC", dealloc[0] if dealloc else -1)
    if ptxas == "-":
        print("PTXAS none"); sys.exit(0)
    asm = subprocess.run([ptxas, "-arch=sm_90a", "-v", "-o", os.path.join(dump, "probe.cubin"), ptxs[-1]], capture_output=True, text=True, timeout=900)
    log = asm.stdout + asm.stderr
    if asm.returncode != 0 and "Unsupported .version" in log:
        print("PTXAS too_old"); sys.exit(0)
    print("PTXAS rc", asm.returncode)
    print("PTXAS serialized", sum(1 for ln in log.splitlines() if "wgmma.mma_async instructions are serialized" in ln))
    for ln in log.splitlines():
        if "Used" in ln or "spill" in ln or "C75" in ln:
            print("PTXAS_LINE", ln.strip())
    """)
# A target, not yet a measurement: the plan expects no spills at the 240-register budget (design
# 3.5 estimates about 200). Replace it with the STL/LDL counts of the first sm_90a compile (S2).
_SPILL_PIN = {"STL": 0, "LDL": 0}


def _ptxas():
    import os
    import shutil

    cands = [os.path.join(os.environ["CUDA_PATH"], "bin", "ptxas")] if os.environ.get("CUDA_PATH") else []
    cands.append(shutil.which("ptxas") or "")
    return next((c for c in cands if c and os.path.isfile(c) and os.access(c, os.X_OK)), None)


def test_d128_register_split_no_spills_and_no_wgmma_serialization(tmp_path):
    """The (24, 240, 240) setmaxnreg split survives into SASS (min_blocks_per_mp=1 on the launch;
    python/cudnn/AGENTS.md: ptxas otherwise drops it silently), the compute body does not spill,
    and ptxas keeps the QK and PV WGMMA groups pipelined (no C7511 serialization advisory)."""
    cands = nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / "dump"
    dump.mkdir()
    proc = subprocess.run([sys.executable, "-c", _SASS_PROBE, str(dump), _ptxas() or "-", *cands], capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, f"sm_90a trace-compile of the D128 template failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ")}
    print("\nD128 SASS:", stats, [ln for ln in out if ln.startswith("PTXAS")])
    assert stats["USETMAXREG"] > 0, "no USETMAXREG: ptxas dropped the register split (C7508)"
    assert (stats["USETMAXREG_ALLOC"], stats["USETMAXREG_DEALLOC"]) == (240, 24), stats
    assert stats["HGMMA"] > 0, stats
    assert_no_new_spills(stats, _SPILL_PIN, tag="D128 sm_90a: ")
    ptxas = {ln.split()[1]: ln.split()[2] for ln in out if ln.startswith("PTXAS ") and len(ln.split()) == 3}
    if "serialized" in ptxas:
        assert ptxas["rc"] == "0" and ptxas["serialized"] == "0", [ln for ln in out if ln.startswith("PTXAS")]
