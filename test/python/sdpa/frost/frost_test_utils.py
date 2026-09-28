# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-condition markers for the FROST SDPA suites.

One gate for the suite, aligned with what the ENGINES declare
(``Capabilities.sm_lo``/``sm_hi`` in sdpa/fwd/engines.py) rather than
re-derived per file. Five files each carried their own copy pinned to exactly
(10, 0), so every one skipped on sm103 -- and would have on Rubin and Thor --
while the engines they test serve the whole line.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from typing import NamedTuple

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
# Rubin gate for the suites whose lowerings exist ONLY from cc 10.7 up (the
# sm107 kernel trees; their engine rows declare sm_lo=107 / sm_hi=119, so the
# marker mirrors that range rather than pinning (10, 7)).
requires_rubin = pytest.mark.skipif(
    _SM is None or not (107 <= _SM <= 119),
    reason="needs a Rubin-line GPU (107 <= SM <= 119), have " + ("none" if _SM is None else f"sm_{_SM}"),
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


def select_engine(graph, name, tiles=None, pack_gqa=None, split_kv=None):
    """Pin the ranked entry for engine ``name`` (graph.plans holds the backend's
    plans and the python engines' in one list). A pin is strict: check_support /
    build_plans raise if that engine declines the graph.

    The FIRST entry for that engine is the heuristics' own best guess for this
    shape. ``tiles`` / ``pack_gqa`` / ``split_kv`` pin a different one, so a test can run a
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
            for field, want in (("tile_m", want_m), ("tile_n", want_n), ("pack_gqa", pack_gqa), ("split_kv", split_kv))
        )

    index = next((i for i in range(len(names)) if _wanted(i)), None)
    assert index is not None, f"no plan for engine {name!r} with tiles={tiles} pack_gqa={pack_gqa} split_kv={split_kv}; plans={names}"
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
    """Drive an EXPLICIT_ABI f16 host from BSHD torch tensors: the same operand
    list the old tensor entry took, translated to pointers plus (batch, seq, head) strides.
    Dense (padded) only — THD goes through the adapter. Direct tests must honor the
    compiled head packing and decode tile; production validates these in the binder."""
    import inspect

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    gmem = cute.AddressSpace.gmem

    def P(t, align=16):
        return None if t is None else make_ptr(getattr(cutlass, _CUTE_DTYPE[str(t.dtype)]), t.data_ptr(), gmem, assumed_align=align)

    paged = block_table_tensor is not None
    b, h, kh, sq, skv, _ = problem_size
    if host is not None:
        namespace = inspect.unwrap(host).__globals__
        cfg = namespace["CFG"]
        if cfg.PACK_GQA and h != kh * cfg.QH_PER_KH:
            raise ValueError(f"PACK_GQA requires H_q == H_kv * {cfg.QH_PER_KH}; got H_q={h}, H_kv={kh}")
        if "N_Q" in namespace and sq * namespace["HEADS_PER_TILE"] > namespace["N_Q"]:
            raise ValueError(f"decode Q rows exceed the compiled {namespace['N_Q']}-row tile")
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
        # The d128 decode tile's ragged-Q leg slots (dense launches leave them
        # dead: RAGGED_Q is off in every direct test's params); filtered out for
        # hosts that do not carry them.
        ragged_q_addr=0,
        ragged_q_div=1,
    )
    params = set(inspect.signature(host if host is not None else fn).parameters)
    fn(**{name: value for name, value in kw.items() if name in params}, stream=stream)


# ---------------------------------------------------------------------------- SASS pins: trace-compile a kernel for a target arch and count opcodes
# A perf lever that is invisible to every numerics test (the exp2 MUFU / FMA split, the Amax_O FMNMX3 fold, a spill) has
# exactly one tripwire: the SASS.  These helpers compile ONE kernel module for ONE arch in a fresh subprocess
# (`CUTE_DSL_ARCH` needs no matching device -- the suite-level `requires_blackwell` is what confines the pins to the
# Blackwell-line lanes), dump the cubin, disassemble it with the first nvdisasm on $CUDA_PATH/bin or $PATH that decodes
# the arch, and hand the caller the opcode counts plus whatever module-derived expectations the probe body printed.
# SKIPS (never fails) when no nvdisasm decodes the cubin; a compile failure IS a failure.

# Opcode -> the substrings a listing line must all contain.  The leading space on the packed ops keeps `FFMA2` from
# matching a hypothetical `XFFMA2`; `MUFU.EX2` also matches `MUFU.EX2.F16x2`-style variants on purpose.  `BSSY` and the
# per-lane ` SYNCS.ARRIVE` (the leading space excludes the uniform `USYNCS.ARRIVE`) are the two opcodes that tell the
# scheduler credit arrive's lowerings apart: the lane-compare branch form costs one BSSY reconverge and `cga_size` arrives
# per call site, the predicated form none and one (`tile_dsl/scheduler.py::read_tile_id_arrive`).
# A count spec is a tuple of substrings a line must ALL contain, or ``("regex:", <pattern>)`` for the few opcodes whose
# name must not be spelled in this source (the CI guardword scan): the predicate-to-general-register move is counted through
# the pattern ``" P\dR "`` (P, a digit, R -- the reverse of R2P).
SASS_OPCODE_COUNTS = {
    "MUFU_EX2": ("MUFU.EX2",),
    "FFMA2": (" FFMA2",),
    "FADD2": (" FADD2",),
    "FSETP": ("FSETP",),
    "FSEL": ("FSEL",),
    "FMNMX3": ("FMNMX3",),
    "STL": ("STL",),
    "LDL": ("LDL",),
    "BSSY": ("BSSY",),
    "SYNCS_ARRIVE": (" SYNCS.ARRIVE",),
}
# The masked-softmax-arm pins (`tile_dsl.mask.apply_mask_chunk`, the bit-word form): every masked KV-tile body carries 4 R2P
# per 32-column keep-word and ~0.04 ISETP per cell; the per-cell compare + select form it replaced carried 0 R2P and one ISETP
# per cell per mask term, and a build that ran out of predicate registers spilled them into GPRs through predicate-to-register moves.
MASK_SASS_OPCODE_COUNTS = {
    **SASS_OPCODE_COUNTS,
    "R2P": (" R2P ",),
    "ISETP": (" ISETP",),
    "PRED2GPR": ("regex:", r" P\dR "),
}

_SASS_PROBE_TEMPLATE = """
import glob, hashlib, json, os, re, subprocess, sys
dump, arch, params_json, cands = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
os.environ["CUTE_DSL_DUMP_DIR"] = dump          # read once, at the first cutlass import
os.environ["CUTE_DSL_KEEP"] = "cubin"            # keep the cubin, disassemble it ourselves
os.environ["CUTE_DSL_ARCH"] = arch               # unconditional: an inherited value would pin the wrong target's SASS
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"  # a compiled-plan cache HIT skips ptxas and dumps no cubin
from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module, supported_cgas_for
from cudnn.sdpa.fwd.config_sm100 import TemplateParams
params_kw = json.loads(params_json)
%(body)s
cubins = sorted(glob.glob(os.path.join(dump, "*.cubin")), key=os.path.getmtime)
if not cubins:
    print("FAIL no cubin dumped into", dump, os.listdir(dump)); sys.exit(3)
print("CUBIN_MD5", hashlib.md5(open(cubins[-1], "rb").read()).hexdigest())
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
    if subs and subs[0] == "regex:":  # ("regex:", <pattern>): one match per line
        pat = re.compile(subs[1])
        return sum(1 for ln in sass if pat.search(ln))
    return sum(1 for ln in sass if all(sb in ln for sb in subs))
for key, subs in json.loads(%(counts)r).items():
    print("SASS", key, cnt(*subs))
print("SASS LINES", len(sass))
"""


def sass_probe_source(body: str, counts: dict = SASS_OPCODE_COUNTS) -> str:
    """The source of a SASS-probe subprocess.  ``body`` (dedented, column-0 statements) builds ``params`` from
    ``TemplateParams(..., **params_kw)``, loads the module with ``_load_sm100_kernel_module``, prints any
    module-derived expectation as ``<UPPER_NAME> <int>`` lines (e.g. ``EXPECT_MUFU_EX2 194``) and calls
    ``mod.compile(...)``; the template around it fixes the arch, dumps the cubin, picks an nvdisasm and prints one
    ``SASS <key> <count>`` line per entry of ``counts``."""
    return _SASS_PROBE_TEMPLATE % {"body": textwrap.dedent(body).strip("\n"), "counts": json.dumps(counts)}


def nvdisasm_candidates() -> list:
    """Every nvdisasm the pins may try, in preference order: $CUDA_PATH/bin first (an internal toolkit that decodes a
    newer arch than the wheel's), then $PATH.  Only existing executables."""
    cands = []
    if os.environ.get("CUDA_PATH"):
        cands.append(os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm"))
    on_path = shutil.which("nvdisasm")
    if on_path:
        cands.append(on_path)
    return [c for c in dict.fromkeys(cands) if os.path.isfile(c) and os.access(c, os.X_OK)]


def arch_known_to_the_dsl(arch: str) -> bool:
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string(arch)
        return True
    except Exception:
        return False


# Spill counts are a property of the TOOLCHAIN at a fixed geometry, so a pin on them is a BOUND, never a compiler-specific
# literal.  The causal specializations of the three exp2-split kernels read STL / LDL 0 / 0 under cutlass-dsl 4.8.0.dev0 +
# the CUDA 13.5 ptxas and 1 / 1 under the CI lane's 4.7.0 + CUDA 13.3 ptxas, on sm_100a and sm_103a alike, on develop AND
# on the branch (review on PR #1178) -- an exact zero there fails a lane on which nothing changed.  A pin records the count
# MEASURED on the branch's own toolchain and :func:`assert_no_new_spills` accepts up to SPILL_TOLERANCE more per opcode: a
# real regression (a fold site or an emulated exp2 pair spilling its operands into local memory) adds tens of STL / LDL,
# the DSL / ptxas jitter one or two.
SPILL_TOLERANCE = 4


def assert_no_new_spills(stats: dict, pins: dict, tag: str = "") -> None:
    """``stats["STL"]`` / ``stats["LDL"]`` at or below ``pins["STL"]`` / ``pins["LDL"]`` plus :data:`SPILL_TOLERANCE`."""
    for key in ("STL", "LDL"):
        assert stats[key] <= pins[key] + SPILL_TOLERANCE, (
            f"{tag}{key} {stats[key]} > {pins[key]} + {SPILL_TOLERANCE}: new spills -- the pin is the count measured on the branch's own "
            f"toolchain, the tolerance the DSL / ptxas jitter (SPILL_TOLERANCE): {stats}"
        )


class SassProbe(NamedTuple):
    stats: dict  # opcode key -> count, one per SASS_OPCODE_COUNTS entry (plus LINES)
    expect: dict  # every `<UPPER_NAME> <int>` line the probe body printed (module-derived expectations)
    cubin_md5: str


def run_sass_probe(tmp_path, *, probe_src: str, arch: str, params: dict, tag: str, timeout: int = 1500) -> SassProbe:
    """Run ``probe_src`` (from :func:`sass_probe_source`) for ``arch`` with the per-build ``params`` (the TemplateParams
    fields the adapter would derive for that device -- JSON-serialised into ``params_kw``) in a fresh interpreter and
    parse its output.  Skips when the DSL has no such arch or no nvdisasm decodes the cubin; a non-zero exit fails."""
    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"this cutlass-dsl has no {arch}")
    cands = nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    # One dump dir per (arch, tag, params): two probes of the same kernel that differ only in ``params`` (the dense and the
    # causal specialization of one pin) must not share a dir, or the second compile trips over the first's.
    dump = tmp_path / f"{arch}_{tag}_{hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]}"
    dump.mkdir()
    argv = [sys.executable, "-c", probe_src, str(dump), arch, json.dumps(params), *cands]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, f"{arch} trace-compile of {tag} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ") and len(ln.split()) == 3 and ln.split()[2].isdigit()}
    expect = {}
    for ln in out:
        parts = ln.split()
        if len(parts) == 2 and parts[0].isupper() and parts[0] not in ("SASS", "CUBIN_MD5", "NVDISASM") and parts[1].lstrip("-").isdigit():
            expect[parts[0]] = int(parts[1])
    md5 = next((ln.split()[1] for ln in out if ln.startswith("CUBIN_MD5 ")), "")
    print(f"\n{tag} {arch} {params} SASS: {stats}; module says {expect}; cubin md5 {md5}")
    return SassProbe(stats, expect, md5)
