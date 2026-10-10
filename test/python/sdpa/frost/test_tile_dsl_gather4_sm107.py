# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tile_dsl.tma.tma_gather4`` / ``ldg_int32x4`` / ``opaque_i64`` -- the index-list row gather of a d = 256 K/V tile.

The primitive lands four indirectly-addressed rows of a 2-D tensor map per issue in the SAME SWIZZLE_128B layout a tiled
box load produces, provided every quad sits at a 512-B-aligned offset inside a 1024-B-aligned sub-box.  That claim has no
symptom when it is wrong (the bytes are finite and plausible), so it is pinned through the STORE path: the roundtrip probe
(``tile_dsl_gather4_probe.py``) gathers 64 KiB tiles (32 blocks of 4 token rows x 4 column boxes of 128 B) from a
``[S, H_kv * 256]`` bf16 table with the head in the COLUMN coordinate, TMA-stores each tile through a plain tiled
(128, 64) SWIZZLE_128B box and the test compares with ``kv[4 blk + r, h * 256 : (h + 1) * 256]`` bitwise.

Three tiers:

* host, no device: the primitive imports nothing kernel-private and nothing of the pre-upstream DSL; the ``cta_group``
  contract; the CuTe DSL version gate (AGENTS.md Rule 7) -- a DSL below the floor and a DSL without the ``sm_107a``
  target are each refused by NAME, before any atom import (a controlled probe over the version state, no real downgrade);
  the HOST entry of that gate is ``tile_dsl.requirements`` (no cutlass import -- ``test_tile_dsl_requirements.py`` proves
  it importable below the floor in a fresh process, which the probes here, run after importing ``tma``, cannot).
* host, an ``sm_107a`` trace-compile (any box whose cutlass-dsl knows the arch): the PTX carries one
  ``tile::gather4 ... cta_group::N.L2::cache_hint`` per issue of the per-warp body, the ``ld.global.nc.v4.s32`` id load and
  the opaque ``mov.b64`` of the EVICT_LAST hint; with an nvdisasm that decodes the cubin, the SASS carries one
  ``UTMALDG.2D.GATHER4`` per issue (the ``.2CTA`` form under ``cta_group=2``), ``R2UR`` (the elected-lane issue model) and
  no spill.
* ``requires_rubin`` (a cc 10.7 device): the roundtrip, bitwise, with ``-1`` and past-the-end block ids coming back as
  four ZERO rows -- their bytes are credited (an uncredited OOB row hangs the ``expect_tx`` = the full share, exit 124
  under ``timeout``) and not over-credited (the same mbarrier serves a second tile on its next phase, bitwise too); the
  8-warp form of the sparse core, the 4-warp fallback, one CTA per SM, and the ``cta_group=2`` pair-leader routing (the
  non-leader's local mbarrier must NOT complete).  Skips typed below the DSL floor and on a DSL without ``sm_107a``.
"""

import importlib
import os
import re
import subprocess
import sys

import pytest
import torch

from frost_test_utils import arch_known_to_the_dsl, nvdisasm_candidates, process_watchdog, requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl]

_PROBE = "tile_dsl_gather4_probe"
_PROBE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PROBE + ".py")


def _probe():
    """The probe module, imported lazily (its cutlass imports must not run at collection on a box without the DSL)."""
    return importlib.import_module(_PROBE)


def _tile_dsl_source(name: str) -> str:
    import cudnn.frost.tile_dsl as tile_dsl

    with open(os.path.join(os.path.dirname(tile_dsl.__file__), name)) as f:
        return f.read()


def _code_lines(src: str) -> str:
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


# ============================================================================ host: the primitive's contract
def test_gather4_primitive_imports_nothing_private_and_nothing_of_the_old_dsl():
    """``tile_dsl`` is the shared library: the gather4 / membership primitives import no kernel-private module (the DSA
    bridge they replace, a block's kernels) and import nothing of the pre-upstream DSL -- the import allow-list below is
    the check."""
    for name in ("tma.py", "mask.py"):
        src = _tile_dsl_source(name)
        code = _code_lines(src)
        assert "deepseek_sparse_attention" not in code, name
        assert "sparse_attention_block" not in code, name
        for line in code.splitlines():
            if line.startswith(("import ", "from ")):
                assert line.startswith(("import cutlass", "from cutlass", "from .", "from typing", "from dataclasses", "import enum")), f"{name}: {line}"
    for line in _code_lines(_tile_dsl_source("requirements.py")).splitlines():
        if line.startswith(("import ", "from ")):
            assert "cutlass" not in line, f"requirements.py (the host gate) must import nothing from the DSL: {line}"
    tma_src = _tile_dsl_source("tma.py")
    assert "def tma_gather4(" in tma_src and "def ldg_int32x4(" in tma_src and "def opaque_i64(" in tma_src
    assert "def tma_gather4_requirement_error(" in _tile_dsl_source("requirements.py")
    assert "TMA_L2_EVICT_LAST = 0x14F0000000000000" in tma_src
    assert "def apply_membership_words(" in _tile_dsl_source("mask.py")


def test_tma_gather4_rejects_a_cta_group_outside_1_2():
    from cudnn.frost.tile_dsl.tma import tma_gather4

    with pytest.raises(ValueError, match="cta_group must be 1 or 2"):
        tma_gather4(None, None, None, 0, 0, 0, 0, 0, cta_group=3)


def test_tma_gather4_declines_a_dsl_below_the_floor_by_name(monkeypatch):
    """Rule 7: a DSL below ``CUTEDSL_MIN_VERSION`` is refused by the gate -- host-callable (``tile_dsl.requirements``, which
    ``tma`` re-exports) and at trace time, BEFORE the inline-asm atom import -- with a message naming the installed and the
    required version (never an ``AttributeError`` from inside the DSL).  The version state is substituted, not the wheel;
    the below-floor IMPORT of the host entry is ``test_tile_dsl_requirements.py``'s fresh-process pin."""
    import cudnn.frost.buffers as buffers
    import cudnn.frost.tile_dsl.tma as tma
    from cudnn.frost.tile_dsl.requirements import tma_gather4_requirement_error
    from cudnn.frost.tile_dsl.tma import tma_gather4

    assert tma.tma_gather4_requirement_error is tma_gather4_requirement_error, "tma re-exports the host module's gate"
    assert tma_gather4_requirement_error() is None, "the installed DSL satisfies the floor (requires_dsl)"
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
    msg = tma_gather4_requirement_error()
    assert msg is not None and "4.6.2" in msg and "4.7.0" in msg and "tma_gather4" in msg, msg
    with pytest.raises(RuntimeError, match=re.escape("found 4.6.2")):
        tma_gather4(None, None, None, 0, 0, 0, 0, 0)  # the gate fires before any operand is touched


def test_tma_gather4_declines_a_dsl_without_the_sm_107a_target_by_name(monkeypatch):
    """Rule 7, the arch half: on a cc 10.7 part a DSL whose ``Arch`` lacks ``sm_107a`` (every public wheel before 4.8.0)
    is refused by name through the host-callable gate the adapters run before importing their kernel module; the gate
    is silent for a part the DSL serves."""
    import cudnn.frost.buffers as buffers
    from cudnn.frost.tile_dsl.requirements import tma_gather4_requirement_error

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    msg = tma_gather4_requirement_error((10, 7))
    assert msg is not None and "sm_107a" in msg, msg
    assert tma_gather4_requirement_error((10, 0)) is None


# ============================================================================ host: the sm_107a form
@pytest.mark.parametrize("cta_group", [1, 2])
def test_sm107a_trace_compile_gather4_form(tmp_path, cta_group):
    """PTX: 16 ``tile::gather4`` per warp body (4 blocks x 4 boxes at 8 gather warps) in the requested ``cta_group`` form
    with the L2 cache hint, one ``ld.global.nc.v4.s32`` (the warp's quad of ids), the opaque EVICT_LAST ``mov.b64``; SASS
    (when an nvdisasm decodes sm_107a): 16 ``UTMALDG.2D.GATHER4`` (``.2CTA`` iff ``cta_group=2``), ``R2UR`` present, no spill."""
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    P = _probe()
    issues = P.blocks_per_warp(8) * P.N_BOX
    dump = tmp_path / f"dump_sm107a_cg{cta_group}"
    dump.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="ptx,cubin")
    proc = subprocess.run(
        [sys.executable, _PROBE_PATH, "sm_107a", str(dump), str(cta_group), *nvdisasm_candidates()], capture_output=True, text=True, timeout=900, env=env
    )
    assert proc.returncode == 0, f"sm_107a trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert f"COMPILED sm_107a cta_group {cta_group}" in lines, proc.stdout[-2000:]
    ptx = {ln.split()[1]: int(ln.split()[2]) for ln in lines if ln.startswith("PTX ")}
    assert ptx["GATHER4"] == issues, f"one gather4 per issue of the per-warp body -- {ptx}"
    assert ptx[f"GATHER4_CG{cta_group}"] == issues and ptx[f"GATHER4_CG{3 - cta_group}"] == 0, ptx
    assert ptx["LDG_NC_V4"] == 1, ptx
    assert ptx["HINT_MOV"] >= 1, f"the EVICT_LAST hint must reach the asm as a register (opaque mov.b64) -- {ptx}"
    if not nvdisasm_candidates():
        pytest.skip("compiled + PTX pinned; no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    if any(ln.startswith("SKIP") for ln in lines):
        pytest.skip(f"compiled + PTX pinned; SASS half skipped: {[ln for ln in lines if ln.startswith(('SKIP', 'REJECT'))]}")
    sass = {ln.split()[1]: int(ln.split()[2]) for ln in lines if ln.startswith("SASS ")}
    print(f"\ngather4 sm_107a cta_group={cta_group} PTX {ptx} SASS {sass}")
    assert sass["GATHER4"] == issues, sass
    assert sass["GATHER4_2CTA"] == (issues if cta_group == 2 else 0), sass
    assert sass["R2UR"] > 0, "row coordinates reach the uniform datapath (R2UR) -- the elected-lane issue model"
    assert sass["SPILL"] == 0, sass


# ============================================================================ Rubin: the roundtrip
def _stream():
    import cuda.bindings.driver as cuda

    return cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))


def _skip_unless_the_dsl_serves_this_part():
    from cudnn.frost.tile_dsl.requirements import tma_gather4_requirement_error

    msg = tma_gather4_requirement_error(tuple(torch.cuda.get_device_capability()))
    if msg is not None:
        pytest.skip(msg)


def _block_ids(n_tiles: int, n_blocks_kv: int, *, g: torch.Generator) -> torch.Tensor:
    """Random block ids per tile with the degenerate rows planted: ``-1`` (no key) every 7th slot, the past-the-end block
    ``n_blocks_kv`` on some of them, and one tile whose 32 slots are ALL ``-1`` (the clamped dead item of the sparse core)."""
    P = _probe()
    ids = torch.randint(0, n_blocks_kv, (n_tiles, P.N_BLOCKS), generator=g, dtype=torch.int64)
    flat = ids.reshape(-1)
    flat[::7] = -1
    flat[3::21] = n_blocks_kv
    ids[n_tiles // 2] = -1
    return ids.to(torch.int32)


def _expected(kv2d: torch.Tensor, ids: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
    """``out[tile * 128 + 4 blk + r] = kv2d[4 ids[tile, blk] + r, h * D : (h + 1) * D]`` when that row exists, else zeros."""
    P = _probe()
    S = kv2d.shape[0]
    n_tiles = ids.shape[0]
    rows = (ids.to(torch.int64) * P.BLOCK).unsqueeze(-1) + torch.arange(P.BLOCK, device=ids.device)  # [tiles, 32, 4]
    rows = rows.reshape(n_tiles, P.TILE_ROWS)
    live = (rows >= 0) & (rows < S)
    cols = heads.to(torch.int64).unsqueeze(-1) * P.D + torch.arange(P.D, device=ids.device)  # [tiles, D]
    gathered = kv2d[rows.clamp(0, S - 1).unsqueeze(-1), cols.unsqueeze(1)]  # [tiles, 128, D]
    want = torch.where(live.unsqueeze(-1), gathered, torch.zeros((), dtype=kv2d.dtype, device=kv2d.device))
    return want.reshape(n_tiles * P.TILE_ROWS, P.D)


_S = 4096  # table rows (tokens) of every roundtrip case: 1024 blocks of 4


def _run_roundtrip(h_kv: int, gw: int, cta_group: int, n_cta: int, n_tiles: int = 2, *, seed: int = 0):
    P = _probe()
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(seed)
    kv2d = torch.randn(_S, h_kv * P.D, generator=g).to(torch.bfloat16).to(dev)
    ids = _block_ids(n_cta * n_tiles, _S // P.BLOCK, g=g).to(dev)
    heads = (torch.arange(n_cta * n_tiles, device=dev) // n_tiles) % h_kv  # the probe's head = bidx % H_KV
    out = torch.full((n_cta * n_tiles * P.TILE_ROWS, P.D), 3.0, dtype=torch.bfloat16, device=dev)  # a sentinel no gathered row carries
    probe = torch.full((n_cta * n_tiles * 2,), -9, dtype=torch.int32, device=dev)
    art = P.compile_roundtrip(h_kv, gw, cta_group)
    import cutlass

    ids_flat = ids.reshape(-1).contiguous()  # the kernel reads the ids as one flat row-major [tiles * 32] int32 stream
    art(kv2d, out, ids_flat, probe, cutlass.Int32(n_cta), cutlass.Int32(n_tiles), _stream())
    torch.cuda.synchronize()
    want = _expected(kv2d, ids, heads)
    out2 = torch.zeros_like(out)
    probe2 = torch.zeros_like(probe)
    art(kv2d, out2, ids_flat, probe2, cutlass.Int32(n_cta), cutlass.Int32(n_tiles), _stream())
    torch.cuda.synchronize()
    return ids, out, want, probe.cpu().view(-1, 2), out2


def _assert_bitwise(out, want, ids, label):
    P = _probe()
    bad = (out != want).any(dim=1).nonzero().flatten()
    if bad.numel():
        first = bad[:8].tolist()
        blk = [(int(r) // P.TILE_ROWS, (int(r) % P.TILE_ROWS) // P.BLOCK) for r in first]
        pytest.fail(f"{label}: {bad.numel()} of {out.shape[0]} rows differ; first rows {first} = (tile, block) {blk}, ids {[int(ids[t, b]) for t, b in blk]}")


@requires_rubin
@pytest.mark.parametrize(
    "h_kv, gw, n_cta",
    [
        pytest.param(2, 8, 6, id="H_kv2-8warps"),  # the sparse core's form: 8 gather warps, the head in the column coordinate
        pytest.param(1, 4, 3, id="H_kv1-4warps"),  # the 4-warp fallback over a single-head table
        pytest.param(2, 8, 0, id="H_kv2-8warps-one-cta-per-sm"),  # n_cta = the SM count
    ],
)
def test_gather4_roundtrip_lands_the_tiled_swizzled_layout(h_kv, gw, n_cta):
    """gather4 -> the SWIZZLE_128B tile -> tiled TMA store == kv[ids] bitwise, over two tiles per CTA on ONE mbarrier.

    ``-1`` blocks and the past-the-end block come back as four ZERO rows each, with their bytes credited (the arm is the
    full share; an uncredited row wedges the wait, so a PASS inside the watchdog budget is the credit -- the watchdog ends
    the process with exit 70 rather than the whole run, as exit 124 under an outer ``timeout`` would) and not over-credited
    (tile 1 completes on the next phase of the same barrier and is bitwise too); one whole tile of ``-1`` ids is the sparse
    core's clamped dead item.  The probe pins that the waited phase is complete and the next one is not, and a second
    launch is bitwise the first."""
    _skip_unless_the_dsl_serves_this_part()
    P = _probe()
    if n_cta == 0:
        n_cta = torch.cuda.get_device_properties(torch.device("cuda")).multi_processor_count
    with process_watchdog(420, f"gather4 cta_group=1 roundtrip H_kv={h_kv} gw={gw} n_cta={n_cta}"):
        ids, out, want, probe, out2 = _run_roundtrip(h_kv, gw, 1, n_cta, seed=h_kv * 10 + gw)
    idl = ids.to(torch.int64)
    oob = (idl < 0) | (idl >= _S // P.BLOCK)
    assert bool(oob.any()) and bool((~oob).any()), "the case must exercise OOB AND live rows"
    assert bool(oob[ids.shape[0] // 2].all()), "one tile is all -1 (the dead item)"
    _assert_bitwise(out, want, ids, f"H_kv={h_kv} gw={gw} n_cta={n_cta}")
    zero_rows = (out == 0).all(dim=1).view(ids.shape[0], P.N_BLOCKS, P.BLOCK)
    assert bool(zero_rows[oob].all()), "every OOB block is four zero rows"
    assert bool((probe[:, 0] == 1).all()), f"every tile's phase completed: {probe.tolist()[:8]}"
    assert bool((probe[:, 1] == 0).all()), f"no tile's NEXT phase completed (no over-arrival / over-credit): {probe.tolist()[:8]}"
    assert torch.equal(out, out2), "second launch bitwise"
    print(f"\ngather4 roundtrip H_kv={h_kv} gw={gw} n_cta={n_cta}: {out.shape[0]} rows bitwise, {int(oob.sum())} OOB blocks zero, 204-SM-class device")


@requires_rubin
def test_gather4_roundtrip_cta_group_2_credits_the_pair_leader():
    """Under ``cta_group=2`` ((2,1,1) cluster) ONLY the pair leader arms ``2 x`` its warps' share and waits; both CTAs'
    tiles come back bitwise (the routing verdict: bytes credited to the leader's copy through the bit-24 clear) and the
    NON-leader's local mbarrier never completes phase 0 (its parity-1 test reads the fresh-barrier bootstrap TRUE, P2).  A
    mis-route is a HANG: the watchdog ends the process (exit 70)
    rather than the whole run; exit 124 under the outer ``timeout`` reads the same."""
    _skip_unless_the_dsl_serves_this_part()
    with process_watchdog(420, "gather4 cta_group=2 roundtrip"):
        ids, out, want, probe, out2 = _run_roundtrip(2, 8, 2, 4, seed=7)
    _assert_bitwise(out, want, ids, "cta_group=2")
    assert torch.equal(out, out2)
    tiles_per_cta = ids.shape[0] // 4
    tile_ix = torch.arange(ids.shape[0])
    leader = ((tile_ix // tiles_per_cta) % 2) == 0
    assert bool((probe[leader, 0] == 1).all()) and bool((probe[leader, 1] == 0).all()), f"the pair leaders' mbarriers complete each phase: {probe.tolist()}"
    # The probe columns are [try_wait.parity(phase), try_wait.parity(phase ^ 1)] with phase = t & 1.  A never-armed barrier stays in
    # its INITIAL state: the parity-0 test is false (phase 0 never completed) and the parity-1 test is TRUE (the fresh-barrier
    # bootstrap that lets a `PipelineState.start(phase=1)` wait return at once) -- so the non-leader reads [0, 1] on even tiles and
    # [1, 0] on odd ones; a completed phase 0 would flip the parity-0 test to 1.
    phase = (tile_ix % tiles_per_cta)[~leader] & 1
    nl = probe[~leader]
    parity0 = torch.where(phase == 0, nl[:, 0], nl[:, 1])
    parity1 = torch.where(phase == 0, nl[:, 1], nl[:, 0])
    assert bool((parity0 == 0).all()) and bool((parity1 == 1).all()), f"the non-leaders' LOCAL mbarriers must never complete phase 0: {probe.tolist()}"
