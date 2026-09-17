# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-condition markers for the FROST SDPA suites.

One gate for the suite, aligned with what the ENGINES declare
(``Capabilities.sm_lo``/``sm_hi`` in sdpa/fwd/engines.py) rather than
re-derived per file. Five files each carried their own copy pinned to exactly
(10, 0), so every one skipped on sm103 -- and would have on Rubin and Thor --
while the engines they test serve the whole line.
"""

import pytest


def _active_sm():
    import torch

    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


_SM = _active_sm()

# Floor for anything that needs a backend SDPA plan or a DSL kernel: the
# backend declines SDPA below Ampere and the DSL has no sm_7x target, so a
# Turing card (the fallback GPU on a runner whose Ampere board has dropped
# out) must skip these rather than fail them.
requires_sm80 = pytest.mark.skipif(
    _SM is None or _SM < 80,
    reason="needs an SM80+ GPU, have " + ("none" if _SM is None else f"sm_{_SM}"),
)
requires_blackwell = pytest.mark.skipif(
    _SM is None or not (100 <= _SM <= 119),
    reason="needs an SM100-line GPU (100 <= SM <= 119), have " + ("none" if _SM is None else f"sm_{_SM}"),
)
# Pre-Rubin gate for the suites whose lowerings do not exist on the Rubin
# line (f16/bf16 and MXFP8 SM100 paths; Rubin serves per-tensor FP8 only) —
# these must SKIP on cc10.7 so the Rubin CI lane can run the whole frost
# directory (the lane's FROST_TEST_PATHS note asks exactly for this).
requires_pre_rubin_blackwell = pytest.mark.skipif(
    _SM is None or not (100 <= _SM <= 106),
    reason="needs a pre-Rubin SM100-line GPU (100 <= SM <= 106; no f16/MXFP8 Rubin lowering), have " + ("none" if _SM is None else f"sm_{_SM}"),
)
requires_blackwell_geforce = pytest.mark.skipif(
    _SM is None or not (120 <= _SM <= 129),
    reason="needs an SM120-line GPU, have " + ("none" if _SM is None else f"sm_{_SM}"),
)


def _dsl_usable():
    """``(usable, why_not)`` for the DSL these engines lower through.

    Version too, not just presence: the extra is deliberately NOT pinned to the
    floor (that would make cudnn-frontend incompatible with anything holding the
    DSL back -- quack-kernels pins ==4.6.0), so an environment can legitimately
    have an older one. The engines decline it; these tests must skip rather than
    fail, and for the same reason.
    """
    from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_state, cutedsl_too_old

    if _SM is not None and _SM < 80:
        return False, f"cutedsl has no sm_{_SM} target (needs SM80+)"
    installed, version = cutedsl_state()
    if not installed:
        return False, "needs the cutedsl extra (nvidia-cutlass-dsl)"
    if cutedsl_too_old(version):
        want = ".".join(str(v) for v in CUTEDSL_MIN_VERSION)
        return False, f"needs nvidia-cutlass-dsl >= {want}, have {version[1]}"
    return True, ""


_DSL_OK, _DSL_WHY = _dsl_usable()
requires_dsl = pytest.mark.skipif(not _DSL_OK, reason=_DSL_WHY or "cutedsl available")


def _dsl_installed() -> bool:
    """For the few call sites that gate inside a test body rather than on it."""
    return _DSL_OK


def _is_plan_for(plan_name, engine) -> bool:
    """A plan reads ``<engine>[<knobs>]``: the heuristics name a concrete config
    for every entry, so match on the engine, not on the whole plan name."""
    return plan_name == engine or plan_name.startswith(engine + "[")


def select_engine(graph, name, tiles=None, pack_gqa=None):
    """Pin the ranked entry for engine ``name`` (graph.plans holds the backend's
    plans and the python engines' in one list). A pin is strict: check_support /
    build_plans raise if that engine declines the graph.

    The FIRST entry for that engine is the heuristics' own best guess for this
    shape. ``tiles`` / ``pack_gqa`` pin a different one, so a test can run a
    config the best guess would not choose. Filters match the STRUCTURED knobs,
    not the rendered plan name: substring matching a name would let a request
    for tile_n=128 select a tile_n=1280 plan, and the test would pass having
    run something else. Every filter — and each ``tiles`` component — is
    None-transparent (matches any value), so a test can pin just kv_tile
    (the auto-tile_m graph_api cases) or nothing at all (the best guess).
    Returns the pinned ``PlanConfig`` so a test can read the knobs the
    heuristics filled in (``split_kv``, tiles, ...).
    """
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    want_m, want_n = tiles if tiles is not None else (None, None)

    def _wanted(i):
        if not _is_plan_for(names[i], name):
            return False
        return all(
            want is None or getattr(graph.plans[i].knobs, field, None) == want
            for field, want in (("tile_m", want_m), ("tile_n", want_n), ("pack_gqa", pack_gqa))
        )

    index = next((i for i in range(len(names)) if _wanted(i)), None)
    assert index is not None, f"no plan for engine {name!r} with tiles={tiles} pack_gqa={pack_gqa}; plans={names}"
    graph.select_plan(index)
    return graph.plans[index]


def offers_engine(graph, name) -> bool:
    """Whether any ranked entry is a plan for engine ``name``."""
    return any(_is_plan_for(graph.get_plan_name_at_index(i), name) for i in range(len(graph.plans)))


def make_dense_stats(batch: int, heads: int, sequence: int, layout: str):
    """Allocate dense Stats in compact or permuted-and-gapped storage."""
    import torch

    if layout == "contiguous":
        return torch.empty(batch, heads, sequence, 1, dtype=torch.float32, device="cuda")
    if layout == "strided":
        storage = torch.empty(sequence + 7, heads + 2, batch, dtype=torch.float32, device="cuda")
        stats = storage.permute(2, 1, 0)[:, :heads, :sequence].unsqueeze(-1)
        assert not stats.is_contiguous()
        return stats
    raise ValueError(f"unknown dense Stats layout {layout!r}")


_CUTE_DTYPE = {
    "torch.float16": "Float16",
    "torch.bfloat16": "BFloat16",
    "torch.float32": "Float32",
    "torch.int32": "Int32",
    "torch.int64": "Int64",
    "torch.int8": "Int8",
    "torch.float8_e4m3fn": "Float8E4M3FN",
    "torch.float8_e5m2": "Float8E5M2",
}


def launch_f16(
    fn,
    q,
    k,
    v,
    o,
    lse,
    sinks,
    seq_kv,
    o_desc,
    problem_size,
    scale_log2,
    units,
    seq_q_lens_addr,
    *,
    o_partial_f32=None,
    block_table_tensor=None,
    block_table_v_tensor=None,
    page_size=0,
    stream=None,
    host=None,
):
    """Drive an EXPLICIT_ABI f16 prefill host from BSHD torch tensors: the same operand
    list the old tensor entry took, translated to pointers plus (batch, seq, head) strides.
    Dense (padded) only — THD goes through the adapter."""
    import inspect

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    gmem = cute.AddressSpace.gmem

    def P(t, align=16):
        return None if t is None else make_ptr(getattr(cutlass, _CUTE_DTYPE[str(t.dtype)]), t.data_ptr(), gmem, assumed_align=align)

    paged = block_table_tensor is not None
    b, h, kh, sq, skv, _ = problem_size
    if paged:
        skv, n_pages = block_table_tensor.shape[1] * page_size, k.shape[0]
        k_st, v_st = (k.stride(0), k.stride(1), k.stride(2)), (v.stride(0), v.stride(1), v.stride(2))
        t_st = (block_table_tensor.stride(0), block_table_tensor.stride(1))
    else:
        n_pages = 0
        k_st, v_st = (k.stride(0), k.stride(1), k.stride(2)), (v.stride(0), v.stride(1), v.stride(2))
        t_st = (0, 0)
    kw = dict(
        q_ptr=P(q),
        k_ptr=P(k),
        v_ptr=P(v),
        o_ptr=P(o),
        lse_ptr=P(lse, 4),
        sinks_ptr=P(sinks),
        meta_ptr=P(seq_kv),
        o_desc_ptr=P(o_desc),
        problem_size=(b, h, kh, sq, skv, 0),
        q_strides=(q.stride(0), q.stride(1), q.stride(2)),
        k_strides=k_st,
        v_strides=v_st,
        o_strides=(o.stride(0), o.stride(1), o.stride(2)),
        lse_strides=tuple(lse.stride()) if lse is not None else (0, 0, 0),
        lse_ext=0,
        scale_softmax_log2=scale_log2,
        n_thd_units=units,
        seq_q_lens_addr=seq_q_lens_addr,
        thd_q_lens_ptr=None,
        thd_kv_lens_ptr=None,
        thd_lens_form=None,
        o_partial_ptr=P(o_partial_f32),
        block_table_ptr=P(block_table_tensor, 4),
        block_table_v_ptr=P(block_table_v_tensor, 4),
        table_strides=t_st,
        n_pages=n_pages,
    )
    params = set(inspect.signature(host if host is not None else fn).parameters)
    fn(**{name: value for name, value in kw.items() if name in params}, stream=stream)


def launch_combine(cfn, o_partial, lse_partial, o_out, lse_out, amax_o, scale_o, problem_size, n_splits, stream=None):
    """Drive the explicit split-combine host from torch tensors (same operand list the tensor entry took)."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    gmem = cute.AddressSpace.gmem

    def P(t, align=16):
        return None if t is None else make_ptr(getattr(cutlass, _CUTE_DTYPE[str(t.dtype)]), t.data_ptr(), gmem, assumed_align=align)

    cfn(
        P(o_partial),
        P(lse_partial),
        P(o_out),
        P(lse_out, 4),
        P(amax_o, 4),
        P(scale_o, 4),
        problem_size,
        n_splits,
        tuple(lse_out.stride()) if lse_out is not None else (0, 0, 0),
        stream=stream,
    )


def launch_quant(
    fn,
    *,
    host=None,
    q,
    k,
    v,
    o,
    lse,
    sinks,
    seq_kv,
    o_desc,
    problem_size,
    scale_log2,
    units,
    seq_q_lens_addr=0,
    o_partial_f32=None,
    descales=None,
    o_scale_fused=None,
    sf=None,
    amax_o=None,
    stream=None,
):
    """Drive an explicit-ABI FP8 / MXFP8 prefill host from dense BSHD torch tensors, by parameter name.
    ``descales`` = (descale_q, descale_k, descale_v, scale_o) for the per-tensor FP8 templates; ``sf`` = (sf_q, sf_k, sf_v)
    ``[B, H, tiles, SF_SMEM]`` int8 for MXFP8."""
    import inspect

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    gmem = cute.AddressSpace.gmem

    def P(t, align=16):
        return None if t is None else make_ptr(getattr(cutlass, _CUTE_DTYPE[str(t.dtype)]), t.data_ptr(), gmem, assumed_align=align)

    # the reloaded artifact exposes the host's parameter names; the in-process (*args, **kwargs) wrapper does not
    params = set(inspect.signature(host if host is not None else fn).parameters)
    kw = dict(
        q_ptr=P(q),
        k_ptr=P(k),
        v_ptr=P(v),
        o_ptr=P(o),
        lse_ptr=P(lse, 4),
        sinks_ptr=P(sinks),
        meta_ptr=P(seq_kv),
        o_desc_ptr=P(o_desc),
        problem_size=problem_size,
        q_strides=tuple(q.stride()[:3]),
        k_strides=tuple(k.stride()[:3]),
        v_strides=tuple(v.stride()[:3]),
        o_strides=tuple(o.stride()[:3]),
        lse_strides=tuple(lse.stride()) if lse is not None else (0, 0, 0),
        lse_ext=0,
        scale_softmax_log2=scale_log2,
        n_thd_units=units,
        seq_q_lens_addr=seq_q_lens_addr,
    )
    if "thd_q_lens_ptr" in params:
        kw.update(thd_q_lens_ptr=None, thd_kv_lens_ptr=None, thd_lens_form=None)
    if "o_partial_ptr" in params:
        kw["o_partial_ptr"] = P(o_partial_f32)
    if descales is not None:
        dq, dk, dv, so = descales
        kw.update(descale_q_ptr=P(dq, 4), descale_k_ptr=P(dk, 4), descale_v_ptr=P(dv, 4), scale_o_ptr=P(so, 4), o_scale_fused=o_scale_fused)
    if sf is not None:
        sfq, sfk, sfv = sf
        kw.update(sf_q_ptr=P(sfq), sf_k_ptr=P(sfk), sf_v_ptr=P(sfv))
        kw.update(
            dict(q_sf_tiles=sfq.shape[2], kv_sf_tiles=sfk.shape[2])
            if "q_sf_tiles" in params
            else dict(total_q_sf_tiles=sfq.shape[2], total_kv_sf_tiles=sfk.shape[2])
        )
    if amax_o is not None:
        kw["amax_o_ptr"] = P(amax_o, 4)
    fn(**kw, stream=stream)
