# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The shared split combine's FUSED entry: the gate AND a static-scale e4m3 store in ONE launch
(``sm100/split_combine.compile_ptr(gate=True, gate_e4m3=True, dtype_o="e4m3")`` -> ``_host_ptr_gated_e4m3``).

The entry reduces the fp32 split partials, gates the fp32 merged value with the existing tanh form (``h = acc * (inv_den / 2)``,
``O = h * tanh(g / 2) + h``, the dead-row SELECT after the fma), multiplies by ``inv = div.rn(1, scale)`` computed on device from the
1-element fp32 per-tensor DEQUANT scale (the serving stacks' convention: the o_proj input scale), and rounds ONCE to e4m3 with
``cvt.rn.satfinite.e4m3x2.f32`` (round-to-nearest-even, saturating at +-448).  The unfused production path -- the gated half entry of the
same form, then ``(O.float() * inv).to(float8_e4m3fn)`` -- rounds to bf16 / f16 first, so the two differ by at most ONE e4m3 ulp at the larger
magnitude (the unit named: the e4m3 spacing at ``max(|a|, |b|)``; the twin named: the gated half entry of the SAME kernel form plus torch's
cast), never bitwise, WHEREVER THE HALF TWIN DOES NOT UNDERFLOW BEFORE THE SCALE: below the half dtype's smallest normal the twin loses digits
and then flushes to zero (f16 from 2^-25 down) while the fused path keeps the fp32 magnitude -- the single rounding is the contract, the twin
is not a general bound, and the boundary cell below records the intended difference (48 on every form against the f16 twin's 0; the bf16
twin's 52 is within the ulp).  The fused store is deterministic (two launches bitwise), writes exactly 0 on a dead row whatever the gate holds,
and leaves the LSE bitwise the half entry's (the LSE never sees the gate or the scale).  Against the serving stacks' gate-and-quantize
convention (``e4m3(bf16(bf16(O) * sigmoid(G)) * inv)``: two half roundings, an exact sigmoid) the same one-ulp class holds under the same
proviso; the histogram is printed, not claimed bitwise.

What this module pins: (1) the ENTRY's host contract -- the keyword's typed declines, the positional frame (the scale slot right after the gate
strides, no defaulted slot ahead of ``stream``), the ``quant_divisor`` constexpr appended to every form and to the two launchers, and a host
trace-compile of the AUTO / OFF / LEAN_WALK forms for sm_100a and sm_107a whose PTX carries the ``div.rn.f32`` inverse, the
``cvt.rn.satfinite.e4m3x2.f32`` pack and the packed ``st.global.b32`` store; (2) the NUMERICS on the GPU (any SM100-line part) on every form the
entry can take -- AUTO / ON / OFF and the lean group walk --: within one ulp of the unfused twin and of the gate-and-quantize model, bitwise x2,
dead rows 0, the LSE bitwise, AUTO == ON on a decode grid, ON == OFF at <= 16 splits (the forms' bitwise window), the walk at two geometries the
cc 10.7 rule selects it for, the f16 underflow boundary of the twin claim (48 against 0 on every form, recorded as intended; the bf16 twin
within the ulp), the saturation contract (+-448, never NaN, where the scaled value exceeds the e4m3 range), the strided e4m3 output
(the element path: canaries kept, bytes bitwise the packed path's), the PDL arm bitwise the plain arm; (3) the ADAPTER's op attribute (``o``
float8_e4m3fn + ``out_scale`` + the epilogue gate on a split plan) with its declines by name, its combine-form rule on the Blackwell explicit-lean
parts (the cc 10.7 split-aware rule's shape on cc 10.0, measured on the GB200 class 2026-10-28; cc 10.3 keeps the module's own forms), the lean
route end to end on ANY SM100-line part (cc 10.0 / 10.3 serve the explicit lean tile; the combine form is the part's own rule) and the tcgen05
route's split plan on a Rubin part (the module's own forms inside the bound, the lean group walk on an overflow grid), each against the same
route with a bf16 ``o`` plus the torch cast.

THE SECOND ROUNDING CONVENTION (``compile_ptr(out_rounding="staged")``; the adapter's ``out_rounding="staged"`` attribute; the trailing
``quant_rounding`` constexpr): a serving stack's gate-and-quantize order, ``e4m3(rn, satfinite)(bf16(bf16(O32) / (1 + exp(-G))) * div.rn(1, s))``
-- the merged value rounded to bf16, the exact sigmoid as a DIVISION, the gated value rounded to bf16, the fp32 multiply by the same inverse,
the saturating cvt -- spelled with the DSL ops that stack's kernel spells, so a provider can serve bytes identical to that chain.  What this
module pins for it: (4) the host contract -- the keyword's vocabulary and its declines by name (``"staged"`` without the fused entry, with an f16
gate), the constexpr appended to the four kernels, the two launchers and the fused host entry (after ``use_pdl``: the positional frame
unchanged), the staged arm called at every gate site and absent from the packed kernel, the host trace-compile of every form whose PTX carries
the bf16 conversions, the ``div.rn.f32`` sigmoid divisions and NO approximate tanh in the staged mode (the single mode's PTX as before);
(5) the NUMERICS on the GPU (bf16; every form): the staged bytes against the torch model of the stack's convention built on the SAME form's
UNGATED bf16 output (= ``bf16(acc * inv_den)``, the value the staged arm rounds first) within one e4m3 code -- the exp lowering is the one term
that may part the two (torch's CUDA libm against the DSL's precise ``exp``, whose final scaling multiply the compiler contracts with the
``+ 1`` into one fma); the histogram is the record --, x2 bitwise, dead rows 0 under a NaN gate, the LSE bitwise the half entry's, AUTO == ON on
the decode grids, ON == OFF at <= 16 splits, the walk at its geometries, saturation, the strided element path, the PDL arm; the EXACT-MERGE
cells (four identical bf16-valued partials at partial LSE 0: the merged value IS the bf16 input, so the bytes are the arithmetic alone on
(a, g, s)) with the near-tie sub-cell (the two conventions part on a few percent of the elements and the staged bytes follow the model there);
the boundary cell reading 52 where the single mode reads 48; (6) the ADAPTER's attribute: the accept on the bf16 lean / tcgen05 routes, the
declines by name (a half ``o``, f16 operands, a value outside the vocabulary), the record field appended last, and the routes end to end
against the ungated bf16 route plus the torch model."""

import dataclasses
import inspect
import json
import math
import os
import subprocess
import sys

import pytest

import torch

from cudnn.frost.compiled_cache import positional_entry

from frost_test_utils import arch_known_to_the_dsl, process_watchdog, requires_blackwell, requires_dsl, requires_rubin
from test_sdpa_split_combine_decode_grid import _partials, _same

pytestmark = [pytest.mark.L0]
_gpu = [requires_dsl, requires_blackwell]

E4M3_MAX = 448.0


# --- helpers --------------------------------------------------------------------------------------------------------


def _gate(b, sq, h, d, dtype, seed):
    """An O-shaped half gate; row 0 (the all-dead row of the multi-row partial patterns) holds NaN, which must never reach the output."""
    gen = torch.Generator().manual_seed(seed)
    gate = (torch.randn(b, sq, h, d, generator=gen) * 3.0).to(dtype)
    if sq > 1:
        gate[:, 0] = torch.nan
    return gate.cuda()


_FORMS = ("auto", "on", "off", "walk")
"""Every form the entry can take: AUTO = the adapters' entry, ON / OFF the forced kernels, WALK the lean group walk (the cc 10.7 adapter's form
on a lean-grid-overflow plan; the entry obeys its selector whatever the grid)."""


_ROUNDINGS = ("single", "staged")
"""The fused store's two rounding conventions (``heuristics.COMBINE_OUT_ROUNDINGS``): one rounding of the fp32 gated value, and a serving stack's
gate-and-quantize order (bf16 only)."""

# Whether the staged bytes are asserted BITWISE the torch model of the stack's convention (``_staged_twin``), or only within one e4m3 code with
# the histogram printed.  The two could part only through the exp lowering: the DSL's precise ``exp`` ends in ``ex2.approx.ftz`` scaled by
# ``2^j`` through an fma whose ``+ 1`` (the sigmoid's denominator) the compiler contracts into it, where torch's CUDA ``exp`` rounds its result
# before the add -- but the scaling by a power of two is exact, so the fma rounds ``e + 1`` exactly as the separate add does, and the two libms
# are the same algorithm.  MEASURED bitwise on a cc 10.0 part on every cell of this module (74 histograms, 0 elements one code off); the bound
# is never widened past one code.
STAGED_TWIN_BITWISE = True


def _entries(comb, *, fused, tag, use_pdl=False, forms=_FORMS, rounding="single"):
    """The ``forms`` of the fused e4m3 entry (``rounding``: its convention) or of its unfused twin, the gated half entry of the same form
    (``dtype_o`` = the gate's half dtype; no convention)."""
    codes = {"auto": comb.DECODE_GRID_AUTO, "on": comb.DECODE_GRID_ON, "off": comb.DECODE_GRID_OFF, "walk": comb.DECODE_GRID_LEAN_WALK}
    kw = dict(dtype_partial="f32", has_lse=True, gate=True, dtype_gate=tag, use_pdl=use_pdl)
    kw.update(dict(dtype_o="e4m3", gate_e4m3=True, out_rounding=rounding) if fused else dict(dtype_o=tag))
    return {name: positional_entry(comb.compile_ptr(**kw, decode_grid=codes[name])) for name in forms}


def _plain_entries(comb, forms=_FORMS):
    """The ``forms`` of the UNGATED bf16 half entry: its store is ``bf16(acc * inv_den)``, the value the staged convention rounds first (the bf16
    attention output a serving stack's gate-and-quantize kernel reads)."""
    codes = {"auto": comb.DECODE_GRID_AUTO, "on": comb.DECODE_GRID_ON, "off": comb.DECODE_GRID_OFF, "walk": comb.DECODE_GRID_LEAN_WALK}
    return {name: positional_entry(comb.compile_ptr(dtype_o="bf16", dtype_partial="f32", has_lse=True, decode_grid=codes[name])) for name in forms}


def _launch(fn, op, lp, b, h, sq, d, splits, gate, *, scale=None, out_dtype=torch.bfloat16, o=None):
    """One launch: the fused entry when ``scale`` is given (the e4m3 output pre-filled with the e4m3 MAX as a canary), else the gated half
    entry (the half output pre-filled with -31) -- or, with ``gate`` None, the ungated half entry; the final LSE pre-filled with -31."""
    if o is None:
        o = torch.full((b, sq, h, d), E4M3_MAX if scale is not None else -31.0, device="cuda", dtype=torch.float8_e4m3fn if scale is not None else out_dtype)
    lse = torch.full((b, h, sq), -31.0, device="cuda", dtype=torch.float32)
    extra = ((gate.data_ptr(), gate.stride()) if gate is not None else ()) + ((scale.data_ptr(),) if scale is not None else ())
    fn(
        op.data_ptr(),
        lp.data_ptr(),
        o.data_ptr(),
        lse.data_ptr(),
        (b, h, sq, d),
        splits,
        o.stride(),
        lse.stride(),
        *extra,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return o, lse


def _ulp_e4m3(x):
    """The e4m3 spacing at |x| (fp32 tensor): ``2^(floor(log2 |x|) - 3)`` for a normal value (|x| >= 2^-6), the subnormal spacing 2^-9 below."""
    mag = x.abs().clamp_min(2.0**-6)
    return torch.exp2(torch.floor(torch.log2(mag)) - 3.0)


def _ulp_distance(a, b):
    """|a - b| in e4m3 ulps AT THE LARGER MAGNITUDE (a one-ulp claim names its unit and its twin); both fp32 tensors."""
    a, b = a.float(), b.float()
    return (a - b).abs() / _ulp_e4m3(torch.maximum(a.abs(), b.abs()))


def _histogram(dist):
    """Counts of exact / one-ulp / more-than-one-ulp differences (the number the report quotes, never a bitwise claim)."""
    exact = int((dist == 0).sum())
    one = int(((dist > 0) & (dist <= 1.0 + 1e-6)).sum())
    more = int((dist > 1.0 + 1e-6).sum())
    return dict(exact=exact, one_ulp=one, more=more, max_ulp=float(dist.max()) if dist.numel() else 0.0)


def _unfused_twin(o_half, scale):
    """The production path the fused entry replaces: the gated half output cast in torch at the same ``inv`` (fp32 ``1 / scale``, RN)."""
    inv = torch.ones((), dtype=torch.float32, device=o_half.device) / scale.float().reshape(())
    return (o_half.float() * inv).to(torch.float8_e4m3fn)


def _gate_quant_model(ref_o, gate, scale, half_dtype):
    """The serving stacks' gate-and-quantize convention on the float64 merged reference: the attention output as a half tensor (the attention kernel's
    own output), the gated value rounded to half, then ``* inv`` in fp32 and the e4m3 cast (an exact sigmoid; two half roundings)."""
    inv = torch.ones((), dtype=torch.float32) / scale.float().cpu().reshape(())
    attn = ref_o.float().to(half_dtype).float()
    g = gate.float().cpu()
    gated = (attn * torch.sigmoid(g)).to(half_dtype).float()
    return (gated * inv).to(torch.float8_e4m3fn)


def _staged_twin(attn_bf16, gate, scale):
    """The torch model of the STAGED convention, on the device, on the bf16 attention output ``attn_bf16`` (the ungated half entry's store of the
    same form = ``bf16(acc * inv_den)``, the value the staged arm rounds first): ``e4m3(rn, satfinite)(bf16(a / (1 + exp(-g))) * div.rn(1, s))`` --
    the sigmoid as a DIVISION (never ``a * sigmoid(g)``: a different rounding), torch's CUDA ``exp``, the bf16 rounding of the quotient, the fp32
    multiply by the RN inverse, and torch's e4m3 cast after a clamp to +-448 (torch reads NaN from 480 up where ``cvt.rn.satfinite`` saturates;
    the clamp makes the two one function on every finite value).  Differs from the kernel only where the exp lowerings part (``STAGED_TWIN_BITWISE``)."""
    inv = torch.ones((), dtype=torch.float32, device=attn_bf16.device) / scale.float().reshape(())
    a, g = attn_bf16.float(), gate.float()
    v = (a / (1.0 + torch.exp(-g))).to(torch.bfloat16).float()
    return (v * inv).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


def _single_model(attn_bf16, gate, scale):
    """The torch model of the SINGLE convention on a bf16-valued merged value: ``e4m3(a * sigmoid(g) * inv)``, one rounding of the fp32 gated
    value -- within one ulp of the kernel's tanh form (an approximate tanh), never bitwise; the exact-merge cell's second reference."""
    inv = torch.ones((), dtype=torch.float32, device=attn_bf16.device) / scale.float().reshape(())
    return (attn_bf16.float() * torch.sigmoid(gate.float()) * inv).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


def _check_staged_twin(hist, what):
    """The staged-vs-model verdict: within one code always (``_check_fused`` asserted it); bitwise when the record says the lowerings agree."""
    if STAGED_TWIN_BITWISE:
        assert hist["one_ulp"] == 0, f"{what}: {hist['one_ulp']} elements one code off the stack-convention model (expected bitwise): {hist}"


def _live_rows(ref_lse):
    """A [B, S_q, H] live mask from the float64 reference LSE ([B, H, S_q])."""
    return torch.isfinite(ref_lse).permute(0, 2, 1)


def _check_fused(fused, twin, lse_fused, lse_twin, ref_lse, *, what, max_ulp=1.0):
    """The fused bytes against a twin's bytes on the LIVE rows within ``max_ulp`` (unit: the e4m3 ulp at the larger magnitude), no NaN bytes,
    dead rows exactly 0 on the fused output, the LSE bitwise; returns the histogram."""
    live = _live_rows(ref_lse)
    f, t = fused.cpu().float(), twin.cpu().float()
    assert not torch.isnan(f).any(), f"{what}: a NaN byte in the fused output"
    dead = ~live
    if dead.any():
        assert torch.count_nonzero(f[dead.unsqueeze(-1).expand_as(f)]).item() == 0, f"{what}: a dead row is not exactly 0"
    lv = live.unsqueeze(-1).expand_as(f)
    dist = _ulp_distance(f[lv], t[lv])
    hist = _histogram(dist)
    assert hist["more"] == 0, f"{what}: {hist['more']} elements differ by more than {max_ulp} e4m3 ulp (max {hist['max_ulp']:.3f}): {hist}"
    if lse_twin is not None:
        assert torch.equal(lse_fused, lse_twin), f"{what}: the LSE must be bitwise the half entry's (it never sees the gate or the scale)"
    return hist


# --- (1) the host contract ----------------------------------------------------------------------------------------------


@requires_dsl
def test_compile_ptr_validates_the_fused_entry_by_name():
    """The fused keyword composes with the gate arm only: an e4m3 output, fp32 partials, no quantized / amax / multiplier-scale arms, no ragged /
    packed / sink placement (the gate's own refusals), and ``use_pdl`` served like the gated half entry's."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    base = dict(dtype_partial="f32", has_lse=True, gate=True, dtype_gate="bf16")
    with pytest.raises(ValueError, match="gate_e4m3 is a gate=True specialization"):
        comb.compile_ptr(dtype_o="e4m3", dtype_partial="f32", has_lse=True, gate_e4m3=True)
    with pytest.raises(ValueError, match="gate_e4m3 stores e4m3"):
        comb.compile_ptr(**dict(base, dtype_o="bf16"), gate_e4m3=True)
    with pytest.raises(ValueError, match="gate_e4m3 stores e4m3"):
        comb.compile_ptr(**dict(base, dtype_o="e5m2"), gate_e4m3=True)
    with pytest.raises(ValueError, match="fp32 partials"):
        comb.compile_ptr(**dict(base, dtype_partial="bf16", dtype_o="e4m3"), gate_e4m3=True)
    for arm in ("quantized", "has_amax", "has_scale_o"):
        with pytest.raises(ValueError, match="gate_e4m3 appends its own scale"):
            comb.compile_ptr(**dict(base, dtype_o="e4m3"), gate_e4m3=True, **{arm: True})
    # the e4m3 output without the fused keyword keeps the former refusals (the quantized entry is the ungated store; the gate never quantizes)
    with pytest.raises(ValueError, match="quantized pointer entry"):
        comb.compile_ptr(**dict(base, dtype_o="e4m3"))
    with pytest.raises(ValueError, match="dense half split launches"):
        comb.compile_ptr(**dict(base, dtype_o="e4m3"), quantized=True)
    # the host-level frame: the scale slot right after the gate strides, every slot ahead of ``stream`` REQUIRED (no defaulted runtime slot)
    params = list(inspect.signature(comb._host_ptr_gated_e4m3).parameters.values())
    names = [p.name for p in params]
    assert names[: names.index("stream")] == [
        "o_partial_ptr",
        "lse_partial_ptr",
        "o_out_ptr",
        "lse_out_ptr",
        "problem_size",
        "n_splits",
        "o_strides",
        "lse_strides",
        "gate_ptr",
        "gate_strides",
        "scale_ptr",
        "stats_log2",
        "decode_grid",
    ], names
    assert all(p.default is inspect.Parameter.empty for p in params[: names.index("stream")]), "a defaulted slot ahead of stream is the #1649 trap"
    assert inspect.signature(comb.compile_ptr).parameters["gate_e4m3"].default is False
    # the ROUNDING CONVENTION keyword (appended; "single" by default): the vocabulary by name, "staged" on the fused bf16 entry only; the host
    # entry's constexpr trails use_pdl (a Constexpr, never a runtime slot: the frame above is unchanged)
    from cudnn.sdpa.fwd.heuristics import COMBINE_OUT_ROUNDINGS
    from cudnn.sdpa.fwd.sparse_tile_rules import COMBINE_OUT_ROUNDINGS as _RULES_ROUNDINGS

    assert (
        COMBINE_OUT_ROUNDINGS is _RULES_ROUNDINGS
    ), "the vocabulary's home is the DSL-free rules module (a declaration's construction imports no DSL); the heuristics re-export it"
    assert COMBINE_OUT_ROUNDINGS == _ROUNDINGS == ("single", "staged")
    assert inspect.signature(comb.compile_ptr).parameters["out_rounding"].default == "single"
    assert (comb.QUANT_ROUNDING_SINGLE, comb.QUANT_ROUNDING_STAGED) == (0, 1)
    with pytest.raises(ValueError, match="out_rounding must be one of"):
        comb.compile_ptr(**dict(base, dtype_o="e4m3"), gate_e4m3=True, out_rounding="double")
    with pytest.raises(ValueError, match="belongs to the fused gated e4m3 entry"):
        comb.compile_ptr(**dict(base, dtype_o="bf16"), out_rounding="staged")  # the gated half entry takes no convention
    with pytest.raises(ValueError, match="belongs to the fused gated e4m3 entry"):
        comb.compile_ptr(dtype_o="bf16", dtype_partial="f32", has_lse=True, out_rounding="staged")  # nor the plain half entry
    with pytest.raises(ValueError, match="defined for the bf16 half dtype"):
        comb.compile_ptr(**dict(base, dtype_gate="f16", dtype_o="e4m3"), gate_e4m3=True, out_rounding="staged")
    with pytest.raises(ValueError, match="needs the gate dtype declared"):  # dtype_gate omitted: the missing declaration is named first
        comb.compile_ptr(dtype_o="e4m3", dtype_partial="f32", has_lse=True, gate=True, gate_e4m3=True, out_rounding="staged")
    hp = inspect.signature(comb._host_ptr_gated_e4m3).parameters
    assert list(hp)[-2:] == ["use_pdl", "quant_rounding"] and hp["quant_rounding"].default == comb.QUANT_ROUNDING_SINGLE


@requires_dsl
def test_quant_divisor_constexpr_is_appended_to_every_form_and_both_launchers():
    """The fused arithmetic is a trailing ``quant_divisor: cutlass.Constexpr[bool] = False`` on the FOUR kernels behind ``_launch_combine``'s
    five forms (the decode-grid kernel, the lean group walk, and ``_combine_kernel`` / ``_combine_decode_kernel`` = the module's own three forms)
    and on the two launchers -- a trace-time constant: every existing entry's traced code is unchanged --, the inverse is the IEEE ``div.rn.f32``
    and the store the saturating e4m3 pair conversion; ``_combine_packed_kernel`` (reached by the packed host entry only, never by
    ``_launch_combine``) carries NO fused arm.  Source pins, so a refactor that drops one -- or 'completes the five' -- is caught without a GPU."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    with open(comb.__file__) as fh:
        src = fh.read()
    assert src.count("quant_divisor: cutlass.Constexpr[bool] = False") == 6, "the four kernels behind _launch_combine + _launch_combine + _launch_own_forms"

    def body(kernel):
        b = src[src.index(f"def {kernel}(") :]
        return b[: b.index("set_name_prefix")]

    for kernel in ("_combine_kernel", "_combine_decode_kernel", "_combine_decode_grid_kernel", "_combine_lean_walk_kernel"):
        assert "quant_divisor" in body(kernel), kernel
        assert "_rcp_div_rn(" in body(kernel), f"{kernel}: the on-device inverse"
        assert "_store_e4m3_quad(" in body(kernel), f"{kernel}: the fused store"
    assert "quant_divisor" not in body("_combine_packed_kernel") and "_store_e4m3_quad(" not in body(
        "_combine_packed_kernel"
    ), "the packed kernel never takes the fused store"
    assert "div.rn.f32 $0, 0f3F800000, $1;" in src
    assert (
        src.count("cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;") == 1 and src.count("cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;") == 1
    ), "one pair conversion per half of a thread's four columns, element 0 in the low byte"
    # the decode-grid launch admits the scale ONLY on the fused entry (the quantized entries' multiplier scale keeps the module's own forms)
    launch = src[src.index("def _launch_combine(") :]
    assert "and (scale_o is None or quant_divisor)" in launch
    # the ROUNDING CONVENTION: a trailing ``quant_rounding: cutlass.Constexpr[int] = QUANT_ROUNDING_SINGLE`` on the same four kernels, the two
    # launchers AND the fused host entry (seven defaults), threaded at every launch site, read at every gate site (the staged arm, called on the
    # same gate loads; ``_combine_kernel`` has two sites), absent from the packed kernel; the staged arm spells the stack's ops -- the precise exp
    # (no fastmath, no approx), the division, two bf16 roundings -- and never a tanh
    assert src.count("quant_rounding: cutlass.Constexpr[int] = QUANT_ROUNDING_SINGLE") == 7, "four kernels + two launchers + the fused host entry"
    staged_call = "_staged_gate_quad(outs[0], outs[1], outs[2], outs[3], g0, g1, g2, g3, all_dead)"
    for kernel in ("_combine_kernel", "_combine_decode_kernel", "_combine_decode_grid_kernel", "_combine_lean_walk_kernel"):
        assert body(kernel).count(staged_call) == (2 if kernel == "_combine_kernel" else 1), f"{kernel}: the staged arm at every gate site"
        assert "if cutlass.const_expr(quant_rounding == QUANT_ROUNDING_STAGED):" in body(kernel), kernel
    assert "quant_rounding" not in body("_combine_packed_kernel") and "_staged_gate_quad(" not in body("_combine_packed_kernel")
    assert launch[: launch.index("def _host_ptr(")].count("quant_rounding,") == 4, "the walk kernel, the decode-grid kernel, the two own-forms calls"
    assert src[src.index("def _launch_own_forms(") : src.index("def _launch_combine(")].count("quant_rounding=quant_rounding") == 3
    host = src[src.index("def _host_ptr_gated_e4m3(") : src.index("def _host_ptr_packed(")]
    assert "quant_rounding: cutlass.Constexpr[int] = QUANT_ROUNDING_SINGLE" in host and host.count("quant_rounding,") == 1
    elem = src[src.index("def _staged_gate_elem(") : src.index("def _staged_gate_quad(")]
    code = elem[elem.index('"""', elem.index('"""') + 3) + 3 :]  # the body after the docstring (the prose names the ops it avoids)
    assert "cute.math.exp(-g)" in code and code.count(".to(cutlass.BFloat16).to(cutlass.Float32)") == 2 and "(a / den)" in code
    assert "tanh" not in code and "fastmath" not in code and "approx" not in code, "the staged arm uses the precise exp and the IEEE division"
    assert "if out_rounding == COMBINE_OUT_ROUNDING_STAGED else {}" in src, "compile_ptr maps the keyword onto the constexpr"


def _frame_adapter(
    *, B=1, SQ=1, H=24, KH=2, top_k=512, split=1, SKV=8192, cc=(10, 0), out_dtype=torch.float8_e4m3fn, out_scale=True, gate=True, dtype=torch.bfloat16, **kw
):
    """A storage-free declaration of the adapter's lean decode form with an e4m3 output and the static scale (the host pins only); ``dtype`` is
    the half dtype of Q / K / V and the gate."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107, SparseOperandDesc

    D = 256
    bshd = lambda b, s, h, dt=dtype: SparseOperandDesc((b, s, h, D), (s * h * D, h * D, D, 1), dt)  # noqa: E731
    return SparseGqaFwdDslSm107(
        q=bshd(B, SQ, H),
        k=bshd(B, SKV, KH),
        v=bshd(B, SKV, KH),
        o=bshd(B, SQ, H, out_dtype),
        lse=SparseOperandDesc((B, H, SQ), (H * SQ, SQ, 1), torch.float32),
        block_ids=SparseOperandDesc((B, top_k), (top_k, 1), torch.int32),
        block_lens=SparseOperandDesc((B,), (1,), torch.int32),
        seq_kv_lens=SparseOperandDesc((B,), (1,), torch.int32),
        top_k=top_k,
        bottom_right=True,
        list_per_sequence=True,
        split_kv=split,
        device_cc=cc,
        epilogue_gate=bshd(B, SQ, H) if gate else None,
        out_scale=SparseOperandDesc((1,), (1,), torch.float32) if out_scale else None,
        **kw,
    )


@requires_dsl
def test_adapter_declares_the_fused_store_by_operand_and_pins_the_combine_form_per_part():
    """``o`` float8_e4m3fn + ``out_scale`` + the epilogue gate on the lean route is the fused entry (an op attribute: the output operand's dtype
    and a scale operand, never a knob); on cc 10.0 the lean route's combine FORM follows the cc 10.7 split-aware rule's shape (the decode-grid
    kernel on the small grids from split 8, on the 384-row grid from split 32, the module's own forms otherwise), on cc 10.3 it keeps the module's
    own forms (unmeasured), on cc 10.7 nothing changes; the declines name the arm (``ValueError`` for a malformed declaration, ``NotImplementedError``
    for an unwired form); the tcgen05 core's split plan takes the fused store too (its forms: the module's own inside the bound, the lean group
    walk on an overflow grid on cc 10.7) and is declined on the explicit-lean parts; the record states the explicit part set and appends."""
    from cudnn.sdpa.fwd import sparse_tile_rules as H
    from cudnn.sdpa.fwd.heuristics import _DECODE_COMBINE_RULE_CC
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as sc
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SPARSE_CAPABILITIES, SparseGqaFwdDslSm107, _sparse_combine_decode_grid

    assert H.SPARSE_LEAN_COMBINE_RULE_CCS == ((10, 0),) and H._SPARSE_LEAN_COMBINE_RULE_SOURCE_CC == _DECODE_COMBINE_RULE_CC == (10, 7)
    assert H.sparse_lean_combine_rule_cc((10, 0), lean=True) == (10, 7) and H.sparse_lean_combine_rule_cc((10, 0), lean=False) == (10, 0)
    assert H.sparse_lean_combine_rule_cc((10, 3), lean=True) == (10, 3) and H.sparse_lean_combine_rule_cc((10, 7), lean=True) == (10, 7)
    assert SPARSE_CAPABILITIES.gated_e4m3_out is True
    assert list(inspect.signature(SparseGqaFwdDslSm107.__init__).parameters)[-2:] == [
        "out_scale",
        "out_rounding",
    ], "append-only: the two appended keywords, in order"
    assert inspect.signature(SparseGqaFwdDslSm107.__init__).parameters["out_rounding"].default == "single"
    # the rule function on the lean grids of the rules module's table (B x 24 heads x S_q; the lean splits 65 / 33 / 22 / 17 at top_k 512)
    for (b, s_q, split), want in {
        (1, 1, 65): "auto",
        (1, 1, 17): "auto",
        (1, 4, 65): "auto",
        (4, 1, 17): "auto",
        (4, 4, 65): "auto",
        (4, 4, 33): "auto",
        (4, 4, 22): "off",
        (4, 4, 17): "off",
    }.items():
        code = {"auto": sc.DECODE_GRID_AUTO, "off": sc.DECODE_GRID_OFF}[want]
        assert _sparse_combine_decode_grid(b, 24, s_q, split, (10, 0), lean=True) == code, (b, s_q, split, want)
        assert _sparse_combine_decode_grid(b, 24, s_q, split, (10, 7), lean=True) == code, "cc 10.7 reads the same rule"
        assert _sparse_combine_decode_grid(b, 24, s_q, split, (10, 3), lean=True) == sc.DECODE_GRID_OFF, "cc 10.3: unmeasured, the module's own forms"
        assert _sparse_combine_decode_grid(b, 24, s_q, split, (10, 0), lean=False) == sc.DECODE_GRID_OFF, "the tcgen05 body keeps the former reading"
        assert _sparse_combine_decode_grid(b, 24, s_q, split, (10, 0)) == sc.DECODE_GRID_OFF, "a caller of the former 5-argument form reads as before"
    # the declaration: the explicit lean route on cc 10.0 with the fused store
    for cc in ((10, 0), (10, 7)):
        a = _frame_adapter(cc=cc, tile_form="lean")
        assert a.check_support() and a.fused_e4m3 and a.tile_form == "lean" and (a.split_kv, a.sub_tiles_per_cta) == (65, 1)
        assert a.combine_form() == sc.DECODE_GRID_AUTO, cc
        a4 = _frame_adapter(cc=cc, tile_form="lean", B=4, SQ=4, sub_tiles_per_cta=4)
        assert a4.check_support() and a4.fused_e4m3 and a4.split_kv == 17 and a4.combine_form() == sc.DECODE_GRID_OFF, cc
    a3 = _frame_adapter(cc=(10, 3), tile_form="lean")
    assert a3.check_support() and a3.fused_e4m3 and a3.combine_form() == sc.DECODE_GRID_OFF
    plain = _frame_adapter(cc=(10, 0), tile_form="lean", out_dtype=torch.bfloat16, out_scale=False)
    assert (
        plain.check_support() and not plain.fused_e4m3 and plain.combine_form() == sc.DECODE_GRID_AUTO
    ), "the bf16 lean route takes the decode-grid form by the same rule"
    # the tcgen05 core's split plan (cc 10.7): the fused store too; the module's own forms at split 4 inside the bound (B 1), the lean group
    # walk on an overflow grid (B 8 x 24 heads x S_q 1 = 192 lean blocks at split <= 4); on the explicit-lean parts the core itself is declined
    t1 = _frame_adapter(cc=(10, 7), tile_form="tcgen05", split=4)
    assert t1.check_support() and t1.fused_e4m3 and t1.tile_form == "tcgen05" and t1.split_kv == 4 and t1.combine_form() == sc.DECODE_GRID_OFF
    t8 = _frame_adapter(cc=(10, 7), tile_form="tcgen05", split=4, B=8)
    assert t8.check_support() and t8.fused_e4m3 and t8.split_kv == 4 and t8.combine_form() == sc.DECODE_GRID_LEAN_WALK
    for cc in ((10, 0), (10, 3)):
        with pytest.raises(NotImplementedError, match="EXPLICIT lean tile only"):
            _frame_adapter(cc=cc, tile_form="tcgen05", split=4).check_support()
    # the record: the explicit part set is a field (the copy the adapter hands the part statement); the three new fields are APPENDED in order,
    # the rounding conventions pinned equal to the heuristics vocabulary
    from cudnn.sdpa.fwd.heuristics import COMBINE_OUT_ROUNDINGS

    assert SPARSE_CAPABILITIES.lean_explicit_ccs == H.SPARSE_LEAN_TILE_EXPLICIT_CCS == ((10, 0), (10, 3))
    assert SPARSE_CAPABILITIES.gated_e4m3_out_roundings == COMBINE_OUT_ROUNDINGS == _ROUNDINGS
    names = [f.name for f in dataclasses.fields(SPARSE_CAPABILITIES)]
    assert names.index("cutedsl_min_version") < names.index("lean_explicit_ccs") < names.index("gated_e4m3_out"), "append-only: after the former last field"
    assert names[-2:] == ["gated_e4m3_out", "gated_e4m3_out_roundings"], "append-only: the conventions field is the record's LAST field"
    # the ROUNDING CONVENTION attribute: accepted with the fused store on bf16 operands (both routes; the combine form unchanged), declined by name
    # otherwise -- a half `o` (nothing to apply it to) is a malformed declaration, f16 operands the typed decline, a bad spelling refused at construction
    st = _frame_adapter(cc=(10, 7), tile_form="lean", out_rounding="staged")
    assert st.check_support() and st.fused_e4m3 and st.out_rounding == "staged" and st.combine_form() == sc.DECODE_GRID_AUTO
    st_t = _frame_adapter(cc=(10, 7), tile_form="tcgen05", split=4, out_rounding="staged")
    assert st_t.check_support() and st_t.fused_e4m3 and st_t.out_rounding == "staged" and st_t.combine_form() == sc.DECODE_GRID_OFF
    st_b = _frame_adapter(cc=(10, 0), tile_form="lean", out_rounding="staged")
    assert (
        st_b.check_support() and st_b.out_rounding == "staged" and st_b.combine_form() == sc.DECODE_GRID_AUTO
    ), "the Blackwell explicit-lean part: the same attribute"
    assert _frame_adapter(cc=(10, 7), tile_form="lean").out_rounding == "single", "the default convention"
    with pytest.raises(ValueError, match="belongs to the fused e4m3 store"):
        _frame_adapter(cc=(10, 7), tile_form="lean", out_dtype=torch.bfloat16, out_scale=False, out_rounding="staged").check_support()
    with pytest.raises(NotImplementedError, match="bf16 operands only"):
        _frame_adapter(cc=(10, 7), tile_form="lean", dtype=torch.float16, out_rounding="staged").check_support()
    assert _frame_adapter(cc=(10, 7), tile_form="lean", dtype=torch.float16).check_support(), "the f16 fused store keeps the single convention"
    with pytest.raises(ValueError, match="out_rounding must be one of"):
        _frame_adapter(cc=(10, 7), tile_form="lean", out_rounding="double")
    # declines by name: ValueError for a MALFORMED declaration (a missing / misplaced / mis-shaped out_scale, like the adapter's other
    # operand-shape errors), NotImplementedError (the typed decline) for an UNWIRED form
    with pytest.raises(ValueError, match="out_scale"):
        _frame_adapter(cc=(10, 0), tile_form="lean", out_scale=False).check_support()  # e4m3 O without the scale
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        _frame_adapter(cc=(10, 0), tile_form="lean", out_dtype=torch.bfloat16).check_support()  # the scale without an e4m3 O
    with pytest.raises(NotImplementedError, match="epilogue_gate"):
        _frame_adapter(cc=(10, 0), tile_form="lean", gate=False).check_support()  # e4m3 O without the gate
    with pytest.raises(NotImplementedError, match="split"):
        _frame_adapter(cc=(10, 7), tile_form="tcgen05").check_support()  # the unsplit core has no combine
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseOperandDesc

    bad = _frame_adapter(cc=(10, 0), tile_form="lean")
    bad.out_scale = SparseOperandDesc((2,), (1,), torch.float32)
    with pytest.raises(ValueError, match="out_scale must be"):
        bad.check_support()  # two elements: not the 1-element fp32 per-tensor scale
    with pytest.raises(NotImplementedError, match="float8_e4m3fn"):
        _frame_adapter(cc=(10, 0), tile_form="lean", out_dtype=torch.float8_e5m2).check_support()  # e5m2 is not the convention served


_RENDER_SRC = r"""
import glob, json, os, re, sys
dump, arch, ckw = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
os.environ["CUTE_DSL_DUMP_DIR"] = dump
os.environ["CUTE_DSL_KEEP"] = "ptx"
os.environ["CUTE_DSL_ARCH"] = arch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
os.environ["CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO"] = "0"
from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb
form = {"auto": comb.DECODE_GRID_AUTO, "on": comb.DECODE_GRID_ON, "off": comb.DECODE_GRID_OFF, "walk": comb.DECODE_GRID_LEAN_WALK}[ckw.pop("form")]
comb.compile_ptr(**ckw, decode_grid=form)
ptx = "\n".join(open(p).read() for p in sorted(glob.glob(os.path.join(dump, "*.ptx"))))
if not ptx:
    print("FAIL no ptx dumped into", dump, os.listdir(dump)); sys.exit(3)
for key, pat in {
    "DIV_RN": r"div\.rn\.f32",
    "DIV_FULL": r"div\.full\.f32",
    "CVT_E4M3": r"cvt\.rn\.satfinite\.e4m3x2\.f32",
    "CVT_BF16": r"cvt\.rn\.bf16\.f32",
    "CVT_BF16X2": r"cvt\.rn\.bf16x2\.f32",
    "TANH": r"tanh\.approx",
    "ST_B32": r"st\.global\.b32",
    "GRIDDEP": r"griddepcontrol",
    "KERNELS": r"\.entry",
}.items():
    print("PTX", key, len(re.findall(pat, ptx)))
"""


@requires_dsl
@pytest.mark.parametrize("arch", ["sm_100a", "sm_107a"])
@pytest.mark.parametrize("form", ["auto", "off", "walk"])
@pytest.mark.parametrize("rounding", _ROUNDINGS)
def test_fused_entry_host_renders_with_the_inverse_and_the_packed_e4m3_store(tmp_path, arch, form, rounding):
    """A host trace-compile of the fused entry (no GPU: the DSL targets the named arch) in a fresh interpreter, under the AUTO / OFF / LEAN_WALK
    selectors (together they trace all four kernels) and under both rounding conventions: the PTX of every kernel the entry traces carries the
    ``div.rn.f32`` inverse, the saturating e4m3 pair conversion and the packed ``st.global.b32`` store, and no griddepcontrol (``use_pdl=False``);
    the SINGLE mode's gate is the approximate tanh form with no bf16 rounding, the STAGED mode's carries four ``div.rn.f32`` sigmoid divisions and
    eight ``cvt.rn.bf16.f32`` roundings per gate site (``_combine_kernel`` has two sites) and NO approximate tanh; neither carries a ``div.full``."""
    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"this cutlass-dsl has no {arch}")
    dump = tmp_path / f"{arch}_{form}_{rounding}"
    dump.mkdir()
    ckw = dict(dtype_o="e4m3", dtype_partial="f32", has_lse=True, gate=True, dtype_gate="bf16", gate_e4m3=True, out_rounding=rounding, form=form)
    proc = subprocess.run(
        [sys.executable, "-c", _RENDER_SRC, str(dump), arch, json.dumps(ckw)],
        capture_output=True,
        text=True,
        timeout=900,
        env=dict(os.environ, PYTHONHASHSEED="0"),
    )
    assert proc.returncode == 0, f"{arch} trace-compile of the fused entry ({form}, {rounding}) failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    got = {ln.split()[1]: int(ln.split()[2]) for ln in proc.stdout.splitlines() if ln.startswith("PTX ")}
    print(f"\n{arch} {form} {rounding}: {got}")
    assert got["KERNELS"] >= 1 and got["DIV_RN"] >= got["KERNELS"], got  # every traced kernel computes the inverse
    assert got["CVT_E4M3"] >= 2 * got["KERNELS"] and got["ST_B32"] >= got["KERNELS"], got  # the packed path in every form
    assert got["GRIDDEP"] == 0 and got["DIV_FULL"] == 0, got
    if rounding == "single":
        assert got["TANH"] >= 4 * got["KERNELS"] and got["CVT_BF16"] == 0 and got["CVT_BF16X2"] == 0, got  # the tanh form; no half rounding
    else:
        assert got["TANH"] == 0, got  # the staged arm never uses the approximate tanh
        assert got["DIV_RN"] >= 5 * got["KERNELS"], got  # the inverse + four IEEE sigmoid divisions per gate site
        assert got["CVT_BF16"] + 2 * got["CVT_BF16X2"] >= 8 * got["KERNELS"], got  # two bf16 roundings per element


# --- (2) the numerics on the GPU -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("splits", [4, 16, 17, 65], ids=lambda s: f"s{s}")
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 24, 1, 256), id="decode_b1_h24_sq1_d256", marks=_gpu),
        pytest.param((1, 24, 4, 256), id="mtp_b1_h24_sq4_d256", marks=_gpu),
        pytest.param((4, 24, 4, 256), id="mtp_b4_h24_sq4_d256", marks=_gpu),
        pytest.param((2, 3, 5, 160), id="b2_h3_sq5_d160", marks=_gpu),
    ],
)
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float16, id="f16")])
def test_fused_matches_the_unfused_path_within_one_ulp_on_every_form(shape, splits, dtype):
    """Every form of the fused entry against the SAME form's gated half entry plus torch's cast (one ulp at the larger magnitude: the fused path
    skips the half rounding), two launches bitwise, the dead row exactly 0 under a NaN gate, the LSE bitwise, AUTO == ON on the decode grids,
    ON == OFF bitwise at <= 16 splits, and the gate-and-quantize model within one ulp (histograms printed); the lean group walk form runs at every
    cell too (the entry obeys its selector whatever the grid)."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = shape
    tag = "f16" if dtype == torch.float16 else "bf16"
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    gate = _gate(b, sq, h, d, dtype, 11 + splits)
    scale = torch.tensor([0.02], dtype=torch.float32, device="cuda")  # inv 50: |O * sigmoid| ~ 2 -> ~100 << 448, no saturation here
    fused_fns = _entries(comb, fused=True, tag=tag)
    twin_fns = _entries(comb, fused=False, tag=tag)
    decode_grid = comb.decode_grid_selected(b, h, sq, d)
    gq = _gate_quant_model(ref_o, gate, scale, dtype)
    outs = {}
    for name in _FORMS:
        o1, l1 = _launch(fused_fns[name], op, lp, b, h, sq, d, splits, gate, scale=scale)
        o2, l2 = _launch(fused_fns[name], op, lp, b, h, sq, d, splits, gate, scale=scale)
        assert _same((o1.view(torch.uint8), l1), (o2.view(torch.uint8), l2)), f"{name}: two launches differ"
        ot, lt = _launch(twin_fns[name], op, lp, b, h, sq, d, splits, gate, out_dtype=dtype)
        hist = _check_fused(o1, _unfused_twin(ot, scale), l1, lt, ref_lse, what=f"{name} vs the unfused twin")
        hist_gq = _check_fused(o1, gq, l1, None, ref_lse, what=f"{name} vs the gate-and-quantize model")
        print(f"\n{shape} s{splits} {tag} {name}: vs unfused {hist}; vs gate-and-quantize {hist_gq}")
        assert hist["exact"] > 0
        outs[name] = (o1.view(torch.uint8), l1)
    if decode_grid:
        assert _same(outs["auto"], outs["on"]), "a decode-shaped grid takes the decode-grid kernel under AUTO"
    if splits <= 16:
        assert _same(outs["on"], outs["off"]), f"splits {splits} <= 16: the decode-grid bytes must be bitwise the lean kernel's"


def _staged_cell(comb, op, lp, b, h, sq, d, splits, gate, scale, ref_lse, name, *, what):
    """One form of the STAGED entry against its references: two launches bitwise; the bytes within one code of the stack-convention model built
    on the same form's UNGATED bf16 output (``_staged_twin``; bitwise under ``STAGED_TWIN_BITWISE``); dead rows 0; the LSE bitwise the gated half
    entry's (and the ungated entry's); within one code of the SINGLE mode's bytes on a non-empty set of flips.  Returns
    ``(bytes, lse, hist vs model, hist vs single, flip fraction)``."""
    staged = _entries(comb, fused=True, tag="bf16", rounding="staged", forms=(name,))[name]
    single = _entries(comb, fused=True, tag="bf16", forms=(name,))[name]
    plain = _plain_entries(comb, forms=(name,))[name]
    gated = _entries(comb, fused=False, tag="bf16", forms=(name,))[name]
    o1, l1 = _launch(staged, op, lp, b, h, sq, d, splits, gate, scale=scale)
    o2, l2 = _launch(staged, op, lp, b, h, sq, d, splits, gate, scale=scale)
    assert _same((o1.view(torch.uint8), l1), (o2.view(torch.uint8), l2)), f"{what} {name}: two launches differ"
    oa, la = _launch(plain, op, lp, b, h, sq, d, splits, None, out_dtype=torch.bfloat16)  # the ungated bf16 O of the same form: the model's input
    og, lg = _launch(gated, op, lp, b, h, sq, d, splits, gate, out_dtype=torch.bfloat16)  # the gated half entry: the LSE's twin
    assert torch.equal(la, lg), f"{what} {name}: the ungated and the gated half entries publish one LSE"
    hist = _check_fused(o1, _staged_twin(oa, gate, scale), l1, lg, ref_lse, what=f"{what} {name}: staged vs the stack-convention model")
    _check_staged_twin(hist, f"{what} {name}")
    os_, ls = _launch(single, op, lp, b, h, sq, d, splits, gate, scale=scale)
    hist_s = _check_fused(o1, os_, l1, ls, ref_lse, what=f"{what} {name}: staged vs single")
    live = _live_rows(ref_lse).unsqueeze(-1).expand_as(o1.cpu().float())
    flips = (o1.view(torch.uint8).cpu() != os_.view(torch.uint8).cpu())[live]
    frac = float(flips.float().mean()) if flips.numel() else 0.0
    print(f"\n{what} s{splits} {name} staged: vs model {hist}; vs single {hist_s}; conventions part on {frac:.4%} of the live elements")
    assert hist["exact"] > 0 and hist_s["exact"] > 0
    return (o1.view(torch.uint8), l1), hist, hist_s, frac


@pytest.mark.parametrize("splits", [4, 16, 17, 65], ids=lambda s: f"s{s}")
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 24, 1, 256), id="decode_b1_h24_sq1_d256", marks=_gpu),
        pytest.param((1, 24, 4, 256), id="mtp_b1_h24_sq4_d256", marks=_gpu),
        pytest.param((4, 24, 4, 256), id="mtp_b4_h24_sq4_d256", marks=_gpu),
        pytest.param((2, 3, 5, 160), id="b2_h3_sq5_d160", marks=_gpu),
    ],
)
def test_staged_rounding_matches_the_stack_convention_on_every_form(shape, splits):
    """The STAGED convention (bf16) on every form of the entry (AUTO / ON / OFF / the lean group walk) against the torch model of a serving
    stack's gate-and-quantize launch on the SAME form's UNGATED bf16 output (the division form of the sigmoid, two bf16 roundings, the fp32
    multiply by the RN inverse, the saturating cast) within one e4m3 code (bitwise under ``STAGED_TWIN_BITWISE``; the histogram is the record),
    two launches bitwise, the dead row exactly 0 under a NaN gate, the LSE bitwise the half entry's; against the SINGLE mode's bytes within one
    code on a non-empty set of flips (the mode is honoured); AUTO == ON on the decode grids, ON == OFF at <= 16 splits (the forms' bitwise window
    holds in the staged mode too)."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = shape
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    gate = _gate(b, sq, h, d, torch.bfloat16, 11 + splits)
    scale = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    decode_grid = comb.decode_grid_selected(b, h, sq, d)
    outs, fracs = {}, []
    for name in _FORMS:
        outs[name], _, _, frac = _staged_cell(comb, op, lp, b, h, sq, d, splits, gate, scale, ref_lse, name, what=f"{shape}")
        fracs.append(frac)
    assert max(fracs) > 0, "the two conventions never part on this cell: the mode is not honoured"
    if decode_grid:
        assert _same(outs["auto"], outs["on"]), "a decode-shaped grid takes the decode-grid kernel under AUTO"
    if splits <= 16:
        assert _same(outs["on"], outs["off"]), f"splits {splits} <= 16: the decode-grid bytes must be bitwise the lean kernel's"


@pytest.mark.parametrize(
    "shape",
    [pytest.param((1, 24, 1, 256), id="decode_b1_h24_sq1_d256", marks=_gpu), pytest.param((4, 24, 4, 256), id="mtp_b4_h24_sq4_d256", marks=_gpu)],
)
def test_staged_exact_merge_is_the_stack_convention_on_the_bf16_input_and_parts_from_single_on_the_near_ties(shape):
    """Four IDENTICAL splits of a bf16-valued fp32 partial at partial LSE 0: the merge is exact (den 4, inv_den 0.25, acc = 4 a without a
    rounding: a bf16 value times four fits fp32), so the merged value IS the bf16 input ``a`` (asserted on the ungated half entry) and the
    staged bytes are the arithmetic alone, ``e4m3(bf16(a / (1 + exp(-g))) * inv)`` on (a, g, s) -- the function a serving stack's
    gate-and-quantize kernel computes on its attention kernel's bf16 output; against the torch model within one code (bitwise under
    ``STAGED_TWIN_BITWISE``), x2 bitwise, every form.  The NEAR-TIE sub-cell: the two conventions part on a few percent of the elements (non-empty,
    below 15 %), and at those elements the staged bytes are the model's (the mode is honoured where it matters) while the single mode stays within
    one ulp of its own model."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = shape
    splits = 4
    gen = torch.Generator().manual_seed(31 + b)
    a = (torch.randn(b, sq, h, d, generator=gen) * 0.5).to(torch.bfloat16)
    op = a.float().unsqueeze(0).expand(splits, b, sq, h, d).reshape(splits * b, sq, h, d).contiguous().cuda()
    lp = torch.zeros((splits * b, h, sq), device="cuda", dtype=torch.float32)
    ref_lse = torch.full((b, h, sq), math.log(splits), dtype=torch.float64)  # every row live
    gate = (torch.randn(b, sq, h, d, generator=gen) * 3.0).to(torch.bfloat16).cuda()
    a = a.cuda()
    scale = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    twin, single_model = _staged_twin(a, gate, scale), _single_model(a, gate, scale)
    for name in _FORMS:
        oa, _ = _launch(_plain_entries(comb, forms=(name,))[name], op, lp, b, h, sq, d, splits, None, out_dtype=torch.bfloat16)
        assert torch.equal(oa, a), f"{name}: the exact merge must reproduce the bf16 input"
        (o_st, l_st), hist, hist_s, frac = _staged_cell(comb, op, lp, b, h, sq, d, splits, gate, scale, ref_lse, name, what=f"exact-merge {shape}")
        assert torch.allclose(l_st, torch.full((b, h, sq), math.log(splits), device="cuda"), atol=1e-5)
        o_si, _ = _launch(_entries(comb, fused=True, tag="bf16", forms=(name,))[name], op, lp, b, h, sq, d, splits, gate, scale=scale)
        hist_si = _check_fused(o_si, single_model, None, None, ref_lse, what=f"exact-merge {shape} {name}: single vs its model")
        flips = o_st.cpu() != o_si.view(torch.uint8).cpu()
        at_flips = int((o_st.cpu()[flips] == twin.view(torch.uint8).cpu()[flips]).sum())
        print(
            f"\nexact-merge {shape} {name}: flips {int(flips.sum())} / {flips.numel()} ({frac:.4%}); staged == model at {at_flips} of them; single vs model {hist_si}"
        )
        assert 0 < frac < 0.15, f"{name}: the two conventions part on {frac:.4%} of the elements (expected a few percent, non-empty)"
        want = int(flips.sum()) if STAGED_TWIN_BITWISE else math.ceil(0.9 * int(flips.sum()))
        assert (
            at_flips >= want
        ), f"{name}: the staged bytes must follow the stack-convention model where the two conventions part ({at_flips} of {int(flips.sum())})"


@pytest.mark.parametrize(
    "shape, splits",
    [
        pytest.param((8, 24, 4, 256), 4, id="overflow_b8_h24_sq4_d256-s4", marks=_gpu),  # 192 lean blocks, S_q 4: the serial-walk window
        pytest.param((32, 24, 1, 256), 3, id="overflow_b32_h24_sq1_d256-s3", marks=_gpu),  # 768 lean blocks: the measured B 32 geometry
    ],
)
def test_fused_walk_form_at_a_walk_selected_geometry(shape, splits):
    """The lean group walk form of the fused entry (``DECODE_GRID_LEAN_WALK``: the cc 10.7 adapter's form on a lean-grid-overflow plan) at two
    geometries the cc 10.7 rule selects it for -- past the decode-grid bound (>= 129 lean blocks) at a split inside the walk's up-front group
    (<= 4) --, in both half dtypes: against the gated half walk entry plus torch's cast and the gate-and-quantize model within one ulp, bitwise
    x2, dead rows 0, the LSE bitwise.  The kernel has no arch-specific lowering, so the cell runs on any SM100-line part (the adapter names the
    form on cc 10.7 only: pinned here on the host)."""
    from cudnn.sdpa.fwd.heuristics import decode_combine_lean_walk_selected
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = shape
    assert decode_combine_lean_walk_selected(b, h, sq, splits, device_cc=(10, 7)) and not decode_combine_lean_walk_selected(b, h, sq, splits, device_cc=(10, 0))
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    scale = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    for dtype, tag in ((torch.bfloat16, "bf16"), (torch.float16, "f16")):
        gate = _gate(b, sq, h, d, dtype, 23 + splits)
        fused = _entries(comb, fused=True, tag=tag, forms=("walk",))["walk"]
        twin = _entries(comb, fused=False, tag=tag, forms=("walk",))["walk"]
        o1, l1 = _launch(fused, op, lp, b, h, sq, d, splits, gate, scale=scale)
        o2, l2 = _launch(fused, op, lp, b, h, sq, d, splits, gate, scale=scale)
        assert _same((o1.view(torch.uint8), l1), (o2.view(torch.uint8), l2)), f"walk {tag}: two launches differ"
        ot, lt = _launch(twin, op, lp, b, h, sq, d, splits, gate, out_dtype=dtype)
        hist = _check_fused(o1, _unfused_twin(ot, scale), l1, lt, ref_lse, what=f"walk {tag} vs the unfused twin")
        hist_gq = _check_fused(o1, _gate_quant_model(ref_o, gate, scale, dtype), l1, None, ref_lse, what=f"walk {tag} vs the gate-and-quantize model")
        print(f"\n{shape} s{splits} {tag} walk: vs unfused {hist}; vs gate-and-quantize {hist_gq}")
        assert hist["exact"] > 0
    # the STAGED convention's walk form at the same geometries (bf16): the model on the ungated walk entry's bf16 O, x2 bitwise, the LSE bitwise
    _staged_cell(comb, op, lp, b, h, sq, d, splits, _gate(b, sq, h, d, torch.bfloat16, 23 + splits), scale, ref_lse, "walk", what=f"{shape} walk")


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_gpu)])
def test_fused_keeps_the_fp32_magnitude_where_the_f16_twin_underflows_before_the_scale(_m):
    """The BOUNDARY of the one-ulp twin claim, recorded as intended: B = H = S_q = 1, D 128, four identical fp32 partials of 1e-8 under a
    zero partial LSE (the merged value 1e-8, the final LSE log 4), a zero gate (sigmoid 1/2: the fp32 gated value 5e-9) and the positive
    normal scale 1e-10 (``inv`` 1e10: the scaled value is fp32 50, an e4m3 tie that rounds to even).  Every form of the fused entry stores
    48 -- the fp32 magnitude, rounded once -- with the LSE bitwise the half entry's.  The f16 twin underflows BEFORE the scale: 5e-9 is
    below 2^-25, half of f16's smallest subnormal, so the gated f16 entry stores 0 and its cast reads 0 -- 12 e4m3 ulp at the larger
    magnitude; the twin is not a general bound.  bf16 does not underflow there (its smallest normal is 2^-126): the gated bf16 entry stores
    bf16(5e-9) = 5.0059e-9 and its cast reads 52, within the one ulp the claim names.  The single rounding is the contract; the twin bounds
    it only where the half dtype keeps the magnitude."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = 1, 1, 1, 128, 4
    op = torch.full((splits * b, sq, h, d), 1e-8, device="cuda", dtype=torch.float32)
    lp = torch.zeros((splits * b, h, sq), device="cuda", dtype=torch.float32)
    ref_lse = torch.full((b, h, sq), math.log(splits), dtype=torch.float64)  # every row live
    scale = torch.tensor([1e-10], dtype=torch.float32, device="cuda")
    for dtype, tag, twin_want, ulp_want in ((torch.float16, "f16", 0.0, 12.0), (torch.bfloat16, "bf16", 52.0, 1.0)):
        gate = torch.zeros((b, sq, h, d), device="cuda", dtype=dtype)
        fused_fns = _entries(comb, fused=True, tag=tag)
        twin_fns = _entries(comb, fused=False, tag=tag)
        for name in _FORMS:
            o1, l1 = _launch(fused_fns[name], op, lp, b, h, sq, d, splits, gate, scale=scale)
            ot, lt = _launch(twin_fns[name], op, lp, b, h, sq, d, splits, gate, out_dtype=dtype)
            f, half_o, twin = o1.cpu().float(), ot.cpu().float(), _unfused_twin(ot, scale).cpu().float()
            dist = _ulp_distance(f, twin)
            print(
                f"\nboundary {tag} {name}: fused {f.unique().tolist()} half entry {half_o.unique().tolist()} twin {twin.unique().tolist()} "
                f"ulp {dist.unique().tolist()} lse {l1.flatten().tolist()}"
            )
            assert torch.all(f == 48.0), f"{tag} {name}: the fused store reads 48 everywhere (fp32 50 -> e4m3 ties to even); got {f.unique().tolist()}"
            assert torch.equal(l1, lt), f"{tag} {name}: the LSE is bitwise the half entry's"
            assert torch.allclose(l1.cpu(), torch.full((b, h, sq), math.log(splits)), atol=1e-5), l1  # the final LSE is log 4 (m 0, den 4)
            assert torch.all(twin == twin_want) and torch.all(dist == ulp_want), (tag, name, twin.unique().tolist(), dist.unique().tolist())
            if dtype == torch.float16:
                assert torch.all(half_o == 0.0), f"{name}: the gated f16 entry underflows before the scale"
            else:
                assert torch.all(half_o != 0.0), f"{name}: the gated bf16 entry keeps the magnitude"
                _check_fused(o1, _unfused_twin(ot, scale), l1, lt, ref_lse, what=f"boundary bf16 {name}")  # the claim holds where the twin keeps it
    # the STAGED convention on the same cell (bf16 only): the merged 1e-8 rounds to bf16 1.0012e-8, halved by the exact sigmoid of the zero gate
    # and rounded to bf16 5.0059e-9, times 1e10 = 50.06 -> e4m3 52 on every form (the bf16 twin's value: the mode is honoured where the two
    # conventions part), the LSE bitwise the half entry's
    gate = torch.zeros((b, sq, h, d), device="cuda", dtype=torch.bfloat16)
    staged_fns = _entries(comb, fused=True, tag="bf16", rounding="staged")
    twin_fns = _entries(comb, fused=False, tag="bf16")
    for name in _FORMS:
        o1, l1 = _launch(staged_fns[name], op, lp, b, h, sq, d, splits, gate, scale=scale)
        ot, lt = _launch(twin_fns[name], op, lp, b, h, sq, d, splits, gate, out_dtype=torch.bfloat16)
        f = o1.cpu().float()
        print(f"\nboundary staged bf16 {name}: fused {f.unique().tolist()} lse {l1.flatten().tolist()}")
        assert torch.all(
            f == 52.0
        ), f"staged bf16 {name}: the staged store reads 52 everywhere (bf16(bf16(1e-8) / 2) x 1e10 = 50.06); got {f.unique().tolist()}"
        assert torch.equal(l1, lt), f"staged bf16 {name}: the LSE is bitwise the half entry's"


@pytest.mark.parametrize("rounding", [pytest.param(r, marks=_gpu) for r in _ROUNDINGS])
def test_fused_store_saturates_at_the_e4m3_range_and_never_writes_nan(rounding):
    """A tiny scale pushes |O * sigmoid(G) / scale| past 448: ``cvt.rn.satfinite`` stores +-448 (torch's cast would give NaN from 480 up), the
    sign kept, no NaN byte; inside the range the twin's bytes stay within one ulp -- in both conventions (the staged twin's pre-cast value is the
    stack convention's ``bf16(a / (1 + exp(-g))) * inv`` on the ungated bf16 O)."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = 1, 24, 4, 256, 65
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    gate = _gate(b, sq, h, d, torch.bfloat16, 3)
    scale = torch.tensor([2e-4], dtype=torch.float32, device="cuda")  # inv 5000: |O * sigmoid(G)| above 0.096 saturates -- most live elements
    fused = _entries(comb, fused=True, tag="bf16", rounding=rounding, forms=("auto",))["auto"]
    twin = _entries(comb, fused=False, tag="bf16", forms=("auto",))["auto"]
    o, lse = _launch(fused, op, lp, b, h, sq, d, splits, gate, scale=scale)
    ot, lt = _launch(twin, op, lp, b, h, sq, d, splits, gate, out_dtype=torch.bfloat16)
    f = o.cpu().float()
    live = _live_rows(ref_lse).unsqueeze(-1).expand_as(f)
    assert not torch.isnan(f).any() and torch.equal(lse, lt)
    if rounding == "single":
        pre = ot.cpu().float() * (1.0 / scale.cpu().float().item())  # the twin's pre-cast value
    else:
        oa, la = _launch(_plain_entries(comb, forms=("auto",))["auto"], op, lp, b, h, sq, d, splits, None, out_dtype=torch.bfloat16)
        assert torch.equal(la, lt)
        inv = torch.ones((), dtype=torch.float32, device="cuda") / scale.reshape(())
        pre = ((oa.float() / (1.0 + torch.exp(-gate.float()))).to(torch.bfloat16).float() * inv).cpu()  # the stack convention's pre-cast value
    over = (pre.abs() >= 480.0) & live  # beyond torch's NaN threshold: ours must read exactly +-448
    inside = (pre.abs() < 400.0) & live
    assert over.sum() > 1000 and inside.sum() > 100, f"the cell must exercise both sides: {int(over.sum())} elements past 480, {int(inside.sum())} inside"
    assert torch.equal(f[over], torch.sign(pre[over]) * E4M3_MAX), "saturated elements read +-448 exactly"
    hist = _histogram(_ulp_distance(f[inside], pre[inside].to(torch.float8_e4m3fn).float()))
    print(f"\nsaturation cell ({rounding}): {int(over.sum())} saturated, inside the range {hist}")
    assert hist["more"] == 0
    assert torch.count_nonzero(f[~live]).item() == 0


@pytest.mark.parametrize("rounding", [pytest.param(r, marks=_gpu) for r in _ROUNDINGS])
def test_strided_e4m3_output_takes_the_element_path_and_keeps_its_canaries(rounding):
    """A strided e4m3 output (D stride 2) takes the element path (Rule S5: the caller's layout kept, padding canaries untouched) and its bytes
    are bitwise the packed path's on a compact output (one ``cvt.rn.satfinite`` convention on both paths) -- in both rounding conventions."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = 2, 3, 5, 160, 8
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    gate = _gate(b, sq, h, d, torch.bfloat16, 5)
    scale = torch.tensor([0.05], dtype=torch.float32, device="cuda")
    fn = _entries(comb, fused=True, tag="bf16", rounding=rounding, forms=("on",))["on"]
    head, token, batch = d * 2 + 8, h * (d * 2 + 8) + 16, sq * (h * (d * 2 + 8) + 16) + 32
    strides = (batch, token, head, 2)
    span = 1 + sum((n - 1) * s for n, s in zip((b, sq, h, d), strides))
    store = torch.full((span + 32,), E4M3_MAX, device="cuda", dtype=torch.float8_e4m3fn)
    view = store.as_strided((b, sq, h, d), strides, storage_offset=16)
    offsets = torch.zeros((b, sq, h, d), dtype=torch.int64)
    for axis, (n, s) in enumerate(zip((b, sq, h, d), strides)):
        dims = [1] * 4
        dims[axis] = n
        offsets += torch.arange(n).view(dims) * s
    used = torch.zeros(store.numel(), dtype=torch.bool)
    used[offsets.flatten() + 16] = True
    o_strided, lse_s = _launch(fn, op, lp, b, h, sq, d, splits, gate, scale=scale, o=view)
    o_packed, lse_p = _launch(fn, op, lp, b, h, sq, d, splits, gate, scale=scale)
    assert torch.all(store.view(torch.uint8).cpu()[~used] == torch.tensor(E4M3_MAX, dtype=torch.float8_e4m3fn).view(torch.uint8)), "canaries untouched"
    assert torch.equal(o_strided.contiguous().view(torch.uint8), o_packed.view(torch.uint8)) and torch.equal(
        lse_s, lse_p
    ), "element path == packed path bitwise"
    if rounding == "single":
        twin = _unfused_twin(_launch(_entries(comb, fused=False, tag="bf16", forms=("on",))["on"], op, lp, b, h, sq, d, splits, gate)[0], scale)
    else:
        twin = _staged_twin(_launch(_plain_entries(comb, forms=("on",))["on"], op, lp, b, h, sq, d, splits, None)[0], gate, scale)
    hist = _check_fused(o_packed, twin, lse_p, None, ref_lse, what=f"d160 packed ({rounding})")
    if rounding == "staged":
        _check_staged_twin(hist, "d160 packed (staged)")


@pytest.mark.parametrize("rounding", [pytest.param(r, marks=_gpu) for r in _ROUNDINGS])
def test_pdl_arm_of_the_fused_entry_is_bitwise_the_plain_arm(rounding):
    """``use_pdl=True`` on the fused entry traces the griddepcontrol pair under the same arithmetic: bitwise the plain arm (the standalone launch
    carries no programmatic predecessor, so the pair is inert) -- in both rounding conventions."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = 1, 24, 4, 256, 65
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    gate = _gate(b, sq, h, d, torch.bfloat16, 9)
    scale = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    plain = _entries(comb, fused=True, tag="bf16", rounding=rounding, forms=("auto", "off"))
    pdl = _entries(comb, fused=True, tag="bf16", rounding=rounding, use_pdl=True, forms=("auto", "off"))
    for name in ("auto", "off"):
        a = _launch(plain[name], op, lp, b, h, sq, d, splits, gate, scale=scale)
        p = _launch(pdl[name], op, lp, b, h, sq, d, splits, gate, scale=scale)
        assert _same((a[0].view(torch.uint8), a[1]), (p[0].view(torch.uint8), p[1])), name


# --- (3) the adapter end to end: the lean route on ANY SM100-line part (cc 10.0 / 10.3 serve the EXPLICIT lean tile; the combine form is the
# part's own rule), the tcgen05 route's split plan on a Rubin part (the module's own forms inside the bound, the lean group walk on an overflow grid)

_LEAN_ROUTE = [requires_dsl, requires_blackwell]  # the explicit lean tile: the core's parts and the admitted Blackwell parts (cc 10.0 / 10.3)
_TCGEN05_ROUTE = [requires_dsl, requires_rubin]  # the tcgen05 core: cc 10.7 .. 11.9 only


def _adapter_fused_cell(*, tile_form, split, B, SQ, kv_lens, paged, seed, out_rounding="single"):
    """The ``tile_form`` route (``split`` 0 = the body's default: the lean tile's 65 at k 1) with an e4m3 ``o`` + ``out_scale`` + the gate against
    the same route with a bf16 ``o`` and torch's cast: within one ulp on the live rows, dead rows 0, the LSE bitwise, the combine form the part's
    own rule names (read back), two launches bitwise.  Under ``out_rounding="staged"`` the reference is the stack-convention model on the same
    route's UNGATED bf16 ``o`` (one more run; its LSE bitwise the gated run's).  Returns ``(histogram, combine form)``."""
    import test_sdpa_fwd_sparse_sm107 as S
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107, _sparse_combine_decode_grid
    import cuda.bindings.driver as cuda

    D, H, KH, top_k, dtype = 256, 24, 2, 512, torch.bfloat16
    q, kk, vv, g = S._inputs(B, SQ, H, KH, dtype, seed, SKV=8192)
    ids, lens = S._decode_lists("shuffled", B, SQ, top_k, kv_lens, g)
    scale_sm = 1.0 / math.sqrt(D)
    ref_o, ref_lse, _ = S._reference_decode(q, kk, vv, ids, lens, kv_lens, scale_sm, top_k)
    gate = (torch.randn(B, SQ, H, D, generator=g, device=q.device, dtype=torch.float32) * 2.0).to(dtype)
    out_scale = torch.tensor([0.05], dtype=torch.float32, device=q.device)
    seq_kv = torch.tensor(kv_lens, device=q.device, dtype=torch.int32)
    kw = {}
    if paged is not None:
        P, hnd = paged
        kp, vp, table = S._paginate(kk, vv, kv_lens, P, hnd, "nan", g)
        kk, vv = kp, vp
        kw = dict(paged_kv=True, page_size=P, block_table=table)
    if split:
        kw["split_kv"] = split
    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))

    def run(o_dtype, launches, gated=True):
        outs, a = [], None
        fused_o = o_dtype == torch.float8_e4m3fn
        for _ in range(launches):
            o = torch.full((B, SQ, H, D), E4M3_MAX if fused_o else 1.5e30, device=q.device, dtype=o_dtype)
            lse = torch.full((B, H, SQ), 1.5e30, device=q.device, dtype=torch.float32)
            a = SparseGqaFwdDslSm107(
                q=q,
                k=kk,
                v=vv,
                o=o,
                lse=lse,
                block_ids=ids,
                block_lens=lens,
                seq_kv_lens=seq_kv,
                top_k=top_k,
                scale=scale_sm,
                list_per_sequence=True,
                bottom_right=True,
                epilogue_gate=gate if gated else None,
                tile_form=tile_form,
                out_scale=out_scale if fused_o else None,
                out_rounding=out_rounding if fused_o else "single",
                **kw,
            )
            assert a.check_support()
            with process_watchdog(300, f"adapter {tile_form} route (fused e4m3) compile + first dispatch"):
                a.compile()
                ws = torch.full((max(a.scratch_workspace_bytes(), 16) // 4,), 1.5e30, device=q.device, dtype=torch.float32)
                a.execute(stream=stream, workspace=ws)
                torch.cuda.synchronize()
            outs.append((o, lse))
        return outs, a

    fused, af = run(torch.float8_e4m3fn, 2)
    half, ah = run(torch.bfloat16, 1)
    cc = torch.cuda.get_device_capability(q.device)
    want_split = split or 65
    # the part's OWN rule: cc 10.7 and (for a lean plan) cc 10.0 the split-aware rule, cc 10.3 the module's own forms, the tcgen05 body as before
    want_form = _sparse_combine_decode_grid(B, H, SQ, want_split, cc, lean=(tile_form == "lean"))
    assert af.fused_e4m3 and not ah.fused_e4m3 and af.tile_form == ah.tile_form == tile_form and af.split_kv == ah.split_kv == want_split
    assert af.combine_form() == ah.combine_form() == want_form == af._combine_form, (cc, want_form)
    assert af.out_rounding == out_rounding and ah.out_rounding == "single"
    (o1, l1), (o2, l2) = fused
    assert torch.equal(o1.view(torch.uint8), o2.view(torch.uint8)) and torch.equal(l1, l2), "two launches bitwise"
    S._check([half[0]], ref_o * torch.sigmoid(gate.float()), ref_lse)  # the bf16 route against the gated oracle (the suite's own check)
    if out_rounding == "single":
        hist = _check_fused(o1, _unfused_twin(half[0][0], out_scale), l1, half[0][1], ref_lse.cpu(), what=f"{tile_form} route fused vs bf16 + cast")
    else:
        plain, ap = run(torch.bfloat16, 1, gated=False)  # the UNGATED bf16 route: the stack convention's input (the same partials, one LSE)
        assert not ap.fused_e4m3 and ap.split_kv == want_split and torch.equal(plain[0][1], half[0][1]), "the ungated and the gated routes publish one LSE"
        hist = _check_fused(o1, _staged_twin(plain[0][0], gate, out_scale), l1, half[0][1], ref_lse.cpu(), what=f"{tile_form} route staged vs the model")
        _check_staged_twin(hist, f"{tile_form} route staged")
    return hist, want_form


@pytest.mark.parametrize("rounding", _ROUNDINGS)
@pytest.mark.parametrize(
    "B, SQ, kv_lens, paged",
    [
        pytest.param(1, 1, [8192], None, id="B1-Sq1-dense", marks=_LEAN_ROUTE),
        pytest.param(2, 4, [8192, 2], (16, True), id="B2-Sq4-paged16-HND-one-sequence-shorter-than-the-step", marks=_LEAN_ROUTE),
    ],
)
def test_adapter_lean_route_fused_e4m3_store(B, SQ, kv_lens, paged, rounding):
    """The explicit lean route on the part under test -- cc 10.7 (the measured part) or an admitted Blackwell part (cc 10.0 / 10.3: the accept test
    of the explicit admission, which has no other GPU row) -- with the fused store in both rounding conventions; the combine form is the part's
    own rule (the decode-grid kernel at split 65 on cc 10.0 and cc 10.7, the module's own forms on cc 10.3)."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as sc

    hist, form = _adapter_fused_cell(tile_form="lean", split=0, B=B, SQ=SQ, kv_lens=kv_lens, paged=paged, seed=B + SQ, out_rounding=rounding)
    cc = torch.cuda.get_device_capability()
    assert form == (sc.DECODE_GRID_OFF if cc == (10, 3) else sc.DECODE_GRID_AUTO), (cc, form)
    print(f"\nadapter lean route fused e4m3 ({rounding}) B={B} S_q={SQ} lens={kv_lens} paged={paged} cc={cc} form={form}: vs the bf16 route's model {hist}")


@pytest.mark.parametrize("rounding", _ROUNDINGS)
@pytest.mark.parametrize(
    "B, form",
    [
        pytest.param(1, "off", id="B1-Sq1-split4-own-forms", marks=_TCGEN05_ROUTE),
        pytest.param(8, "walk", id="B8-Sq1-split4-lean-group-walk", marks=_TCGEN05_ROUTE),
    ],
)
def test_adapter_tcgen05_route_fused_e4m3_store(B, form, rounding):
    """The tcgen05 core's split plan (split 4) with the fused store on a Rubin part, in both rounding conventions: at B 1 x 24 heads x S_q 1 (24
    lean blocks, inside the bound, below the decode-grid rule's split 8) the module's own forms; at B 8 (192 lean blocks: past the bound at a split
    inside the walk's group) the LEAN GROUP WALK -- the fused arm of ``_combine_lean_walk_kernel`` through the adapter."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as sc

    hist, got = _adapter_fused_cell(tile_form="tcgen05", split=4, B=B, SQ=1, kv_lens=[8192] * B, paged=None, seed=40 + B, out_rounding=rounding)
    assert got == {"off": sc.DECODE_GRID_OFF, "walk": sc.DECODE_GRID_LEAN_WALK}[form], got
    print(f"\nadapter tcgen05 route fused e4m3 ({rounding}) B={B} split=4 form={form}: vs the bf16 route's model {hist}")
