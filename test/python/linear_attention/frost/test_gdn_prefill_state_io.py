# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The state I/O build switches of the chunked GDN prefill kernel keep the op's bytes.

``STATE_IO_COALESCED`` and ``TILE_HANDOVER_EARLY`` change how compute group 1 moves the fp32 state between global and
tensor memory and when it reads the next work item's seed; ``STATE_SEED_COPY`` hands the seed to the load warp and the
copy engine.  None may change a word of the output or of a state.

The coalesced path moves a state through a per-warp shared-memory scratch in which a lane loads words other lanes of its
warp stored, and the seed copy keeps a warp's final-state keys in words every lane of the warp stores and loads.  Lanes
of a warp are not ordered against each other, so each such hand-over is held by a host-only pin of its rendering (a
trace-compile for ``sm_100a``; no device, nothing is launched): a warp synchronisation stands between the stores and the
loads and again between the loads and the next stores."""

import os
import re
import subprocess
import sys
import textwrap

import pytest
import torch

from cudnn import cpp_launcher
from cudnn.linear_attention.frost import gdn_engine
from linear_attention.test_la import (
    FWD_TOL,
    STATE_TOL,
    Backend,
    assert_bitwise,
    assert_rms_close,
    clear_caches,
    frost_la_unavailable,
    make_case,
    random_state,
    reference,
    run_fwd,
    state_pool,
)

gdn_prefill_f16 = pytest.importorskip("cudnn.linear_attention.frost.kernel.gdn_prefill_f16")
gdn_warmup_forward_f16 = pytest.importorskip("cudnn.linear_attention.frost.kernel.gdn_warmup_forward_f16")

# (STATE_IO_COALESCED, TILE_HANDOVER_EARLY, STATE_SEED_COPY): each switch acts only on top of the ones in front of it.
SWITCHES = {"coalesced": (True, False, False), "coalesced_early": (True, True, False), "coalesced_early_seed_copy": (True, True, True)}
OFF = (False, False, False)
SPARE_FILL = 3.14159


@pytest.fixture
def frost(monkeypatch):
    """``(backend, built)``: the FROST backend and the prefill configs the test's calls build.  The host cache starts empty,
    so every static configuration of the test is built inside it.  Every call runs the op: the C++ launcher keys a call
    by its arguments, not by a build switch, and would replay one setting's kernels for the next."""
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    backend = Backend("frost")
    built = []
    real_build_cfg = gdn_prefill_f16.build_cfg

    def build_cfg(*args, **kwargs):
        cfg = real_build_cfg(*args, **kwargs)
        built.append(cfg)
        return cfg

    monkeypatch.setattr(gdn_prefill_f16, "build_cfg", build_cfg)
    monkeypatch.setattr(gdn_warmup_forward_f16, "warmup_forward_cache", {})
    monkeypatch.setattr(cpp_launcher, "HOT_CALLS", sys.maxsize)  # no call is recorded, none is served by a recording
    clear_caches()
    yield backend, built
    clear_caches()


def set_switches(monkeypatch, coalesced, early, seed_copy):
    monkeypatch.setattr(gdn_prefill_f16, "STATE_IO_COALESCED", coalesced)
    monkeypatch.setattr(gdn_prefill_f16, "TILE_HANDOVER_EARLY", early)
    monkeypatch.setattr(gdn_prefill_f16, "STATE_SEED_COPY", seed_copy)
    clear_caches()


def cap_sm_count(monkeypatch, sm_cap):
    """Plan for ``sm_cap`` thread blocks, so each block runs several work items one behind another."""
    if sm_cap is not None:
        monkeypatch.setattr(gdn_engine, "multiprocessor_count", lambda device: sm_cap)


def pooled_forward(backend, case, packed, **pool_kw):
    """The op on a fresh pool holding ``packed`` at the table's rows: ``(o, pool, indices, spare)``."""
    pool, indices, spare = state_pool(case, packed, fill=SPARE_FILL, **pool_kw)
    o, final_state = run_fwd(backend, case, initial_state=pool, state_indices=indices, output_final_state=True)
    assert final_state is pool
    torch.cuda.synchronize()
    return o, pool, indices, spare


def assert_path_taken(built, coalesced, early, seed_copy):
    assert any(cfg.state_io_coalesced for cfg in built) == coalesced
    assert any(cfg.tile_handover_early for cfg in built) == early
    assert any(cfg.state_seed_copy for cfg in built) == seed_copy


@pytest.mark.L0
@pytest.mark.parametrize("switches", list(SWITCHES))
def test_state_io_switches_keep_pool_bytes(monkeypatch, frost, switches):
    """Ragged sequences on more (sequence, head) tiles than the device has SMs, so thread blocks hand over between work
    items: output and pool are bitwise the same with the switches on and off, and agree with the fp64 reference."""
    backend, built = frost
    case = make_case("gdn", torch.bfloat16, seq_lens=[257, 64, 320, 129, 200, 71], H=64, lo=0.9)
    packed = random_state(case)
    set_switches(monkeypatch, *OFF)
    o_off, pool_off, indices, spare = pooled_forward(backend, case, packed)
    assert_path_taken(built, *OFF)
    set_switches(monkeypatch, *SWITCHES[switches])
    o_on, pool_on, _, _ = pooled_forward(backend, case, packed)
    assert_path_taken(built, *SWITCHES[switches])
    assert_bitwise(f"o[{switches}]", o_on, o_off)
    assert_bitwise(f"pool[{switches}]", pool_on, pool_off)
    assert (pool_on[spare] == SPARE_FILL).all()
    o_ref, fs_ref = reference(case, initial_state=packed)
    assert_rms_close("o", o_on, o_ref, FWD_TOL[case.dtype])
    assert_rms_close("final_state", pool_on[indices.long()], fs_ref, STATE_TOL[case.dtype])


@pytest.mark.L0
@pytest.mark.parametrize("switches", ["coalesced_early", "coalesced_early_seed_copy"])
def test_shared_pool_row_seeds_the_next_sequence(monkeypatch, frost, switches):
    """One thread block, one head and consecutive sequences that name the same pool row: every sequence is seeded with
    the state its predecessor wrote, so the batch is bitwise the one concatenated sequence.  The early hand-over reads
    the next seed ahead of the final-state store and must not take that read when the seed is the row being written; the
    seed copy reads a seed ahead of the stores of the block's last work items and must load it again behind them."""
    backend, built = frost
    cap_sm_count(monkeypatch, 1)
    num_seqs, tokens = 4, 256
    whole = make_case("gdn", torch.bfloat16, seq_lens=[num_seqs * tokens], H=1, lo=0.9)
    batch = whole.clone(cu=torch.arange(0, num_seqs + 1, dtype=torch.int32, device="cuda") * tokens, N=num_seqs)
    seed = random_state(whole)
    set_switches(monkeypatch, *OFF)
    o_whole, fs_whole = run_fwd(backend, whole, initial_state=seed.clone(), output_final_state=True, batch_invariant=True)
    torch.cuda.synchronize()
    row = 1
    indices = torch.full((num_seqs,), row, dtype=torch.int32, device="cuda")

    def shared_row_forward():
        pool = torch.full((3,) + tuple(seed.shape[1:]), SPARE_FILL, device="cuda", dtype=seed.dtype)
        pool[row] = seed[0]
        o, _ = run_fwd(backend, batch, initial_state=pool, state_indices=indices, output_final_state=True, batch_invariant=True)
        torch.cuda.synchronize()
        return o, pool

    o_off, pool_off = shared_row_forward()
    assert_bitwise("o[shared row]", o_off, o_whole)
    assert_bitwise("final_state[shared row]", pool_off[row : row + 1], fs_whole)
    set_switches(monkeypatch, *SWITCHES[switches])
    o_on, pool_on = shared_row_forward()
    assert_path_taken(built, *SWITCHES[switches])
    assert_bitwise(f"o[shared row, {switches}]", o_on, o_whole)
    assert_bitwise(f"pool[shared row, {switches}]", pool_on, pool_off)


@pytest.mark.L0
def test_seed_copy_without_a_final_state_keeps_the_output(monkeypatch, frost):
    """A seeded call that returns no final state compiles the seed copy without the final-state store: every work item
    hands the state region over behind its last chunk.  The output is bitwise the switched-off call's."""
    backend, built = frost
    case = make_case("gdn", torch.bfloat16, seq_lens=[257, 64, 320, 129, 200, 71], H=64, lo=0.9)
    packed = random_state(case)

    def forward():
        o, _ = run_fwd(backend, case, initial_state=packed.clone(), output_final_state=False)
        torch.cuda.synchronize()
        return o

    set_switches(monkeypatch, *OFF)
    o_off = forward()
    set_switches(monkeypatch, *SWITCHES["coalesced_early_seed_copy"])
    o_on = forward()
    assert_path_taken(built, *SWITCHES["coalesced_early_seed_copy"])
    assert_bitwise("o[seed only, seed copy]", o_on, o_off)
    o_ref, _ = reference(case, initial_state=packed)
    assert_rms_close("o", o_on, o_ref, FWD_TOL[case.dtype])


@pytest.mark.L1
@pytest.mark.parametrize("tokens", [64, 192], ids=["one_chunk", "three_chunks"])
@pytest.mark.parametrize("period", [1, 2, 3, 4, 6])
def test_pool_rows_shared_at_a_distance_keep_the_block_order(monkeypatch, frost, period, tokens):
    """One thread block, one head and twelve sequences whose pool rows repeat every ``period`` sequences: a sequence is
    seeded with the state the sequence ``period`` work items before it wrote.  With the switches off every seed is read
    behind every earlier store of the block.  The seed copy reads a seed ahead of the stores of the block's last work
    items and loads it again when its row is one of theirs; a row written further back is read behind its store.  Output
    and pool are bitwise the switched-off call's for every distance."""
    backend, built = frost
    cap_sm_count(monkeypatch, 1)
    num_seqs = 12
    case = make_case("gdn", torch.bfloat16, seq_lens=[tokens] * num_seqs, H=1, lo=0.9)
    seed = random_state(case)
    indices = torch.arange(num_seqs, dtype=torch.int32, device="cuda") % period + 1

    def forward():
        builds = len(built)
        pool = torch.full((period + 2,) + tuple(seed.shape[1:]), SPARE_FILL, device="cuda", dtype=seed.dtype)
        pool[1 : period + 1] = seed[:period]
        o, _ = run_fwd(backend, case, initial_state=pool, state_indices=indices, output_final_state=True, batch_invariant=True)
        torch.cuda.synchronize()
        assert len(built) > builds, "the call built no prefill kernel: it was not run under its own switch setting"
        return o, pool

    set_switches(monkeypatch, *OFF)
    o_off, pool_off = forward()
    for switches in ("coalesced_early", "coalesced_early_seed_copy"):
        set_switches(monkeypatch, *SWITCHES[switches])
        o_on, pool_on = forward()
        assert_bitwise(f"o[period {period}, {switches}]", o_on, o_off)
        assert_bitwise(f"pool[period {period}, {switches}]", pool_on, pool_off)
    assert_path_taken(built, *SWITCHES["coalesced_early_seed_copy"])
    assert (pool_on[0] == SPARE_FILL).all() and (pool_on[period + 1] == SPARE_FILL).all()


FORMS = {
    "pool": dict(call="pool"),
    "pool_row_pad_int64": dict(call="pool", pool_kw=dict(padded=True, index_dtype=torch.int64)),
    "pool_head_v_pad": dict(call="pool", pool_kw=dict(padded=True, head_padded=True)),
    "pool_k64": dict(call="pool", K=64),
    "pool_one_long_of_nine": dict(call="pool", seq_lens=[17000, 64, 130, 1, 257, 0, 64, 64, 192]),
    "seed_and_final": dict(call="packed", op_kw=dict(output_final_state=True)),
    "seed_and_final_long": dict(call="packed", seq_lens=[4096, 130, 0, 64], op_kw=dict(output_final_state=True)),
    "seed_and_final_invariant": dict(call="packed", op_kw=dict(output_final_state=True, batch_invariant=True)),
    "seed_overwritten": dict(call="packed", op_kw=dict(output_final_state=True, overwrite_initial_state=True)),
    "seed_only": dict(call="packed", op_kw=dict(output_final_state=False)),
    "final_only": dict(call="unseeded", op_kw=dict(output_final_state=True)),
}


@pytest.mark.L1
@pytest.mark.parametrize("sm_cap", [None, 1, 3], ids=["device_sms", "one_block", "three_blocks"])
@pytest.mark.parametrize("form", list(FORMS))
def test_state_io_switches_keep_bytes_across_call_forms(monkeypatch, frost, form, sm_cap):
    """Every call form that hands a state in or out, on ragged lengths with an empty sequence and a length that is not
    a chunk multiple, planned for the device's SMs and for one and three thread blocks: output and states are bitwise
    the same with the switches off and with each switch added to the ones in front of it.  One sequence above 16,384
    tokens among eight short ones builds a plan whose first kernel may hand the whole batch to the chain group; the
    prefill then launches without a work item.  A long seeded sequence in the default call may be cut into pieces that
    start behind the seed.  A pool padded between rows, heads and value rows is read by the seed copy's tensor map."""
    backend, built = frost
    cap_sm_count(monkeypatch, sm_cap)
    spec = FORMS[form]
    case = make_case("gdn", torch.bfloat16, seq_lens=spec.get("seq_lens", [130, 0, 257, 64, 321, 1, 192]), H=4, K=spec.get("K", 128), lo=0.9)
    packed = random_state(case)

    def forward():
        builds = len(built)
        if spec["call"] == "pool":
            o, state, _, _ = pooled_forward(backend, case, packed, **spec.get("pool_kw", {}))
        else:
            state = packed.clone() if spec["call"] == "packed" else None
            o, final_state = run_fwd(backend, case, initial_state=state, **spec["op_kw"])
            torch.cuda.synchronize()
            state = final_state if final_state is not None else state
        assert len(built) > builds, "the call built no prefill kernel: it was not run under its own switch setting"
        return o, state

    set_switches(monkeypatch, *OFF)
    o_off, state_off = forward()
    for switches, setting in SWITCHES.items():
        set_switches(monkeypatch, *setting)
        o_on, state_on = forward()
        assert_bitwise(f"o[{form}, {switches}]", o_on, o_off)
        assert_bitwise(f"state[{form}, {switches}]", state_on, state_off)
    assert any(cfg.state_io_coalesced for cfg in built)
    # The early hand-over and the seed copy need a seed table.
    seeded = spec["call"] != "unseeded"
    assert any(cfg.tile_handover_early for cfg in built) == seeded
    assert any(cfg.state_seed_copy for cfg in built) == seeded


# ---- the rendering of the hand-overs from lane to lane through shared memory (host only) ----------------------------

RENDER_ARCH = "sm_100a"
RENDER_ROUNDS = 4  # a state of 128 key columns: four tcgen05 rounds of 32 columns, two exchanges through the scratch each
RENDER_KEYS = 3
HANDOVER_FORMS = ("seed", "final", "final_freed_inside", "keys")
HANDOVER = ("put", "sync", "take", "sync")  # one hand-over: the stores, the warp's synchronisation, the loads, the synchronisation

_PROBE_SRC = textwrap.dedent('''
    """Probe: one warp.  Every lane runs one hand-over of the chunked GDN prefill kernel with the kernel's own helpers and
    address maps -- seed: the global loads of a state in the coalesced order, through the warp's scratch, into tensor
    memory; final: out of tensor memory, through the scratch, into the coalesced global stores; final_freed_inside: the
    same where the seed copy takes the state region over inside the store; keys: a warp's final-state keys, stored,
    compared, stored and compared again."""
    import glob, os, sys
    from typing import NamedTuple

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.experimental.primitives as nvvm

    from cudnn.frost.tile_dsl.barrier import MBarrier, Producer
    from cudnn.linear_attention.frost.kernel import gdn_prefill_f16 as kernel

    LANES = 32
    ROUND_WORDS = 32
    SMEM = cutlass.AddressSpace.smem


    class Bars(NamedTuple):
        mb_state_free: MBarrier


    @cute.kernel
    def probe(mState: cute.Tensor, mOut: cute.Tensor, form: cutlass.Constexpr[str], rounds: cutlass.Constexpr[int], keys: cutlass.Constexpr[int], word: cutlass.Int32):
        lane_idx = cutlass.Int32(cute.arch.thread_idx()[0])
        if cutlass.const_expr(form == "keys"):
            final_keys = cutlass.Array(cutlass.Int32, keys, space=SMEM, alignment=16).data_ptr()
            kernel.final_key_put(final_keys, cutlass.Int32(0), word)
            first = kernel.final_key_held(final_keys, keys, word)
            kernel.final_key_put(final_keys, cutlass.Int32(1), word + cutlass.Int32(1))
            second = kernel.final_key_held(final_keys, keys, word + cutlass.Int32(1))
            mOut[lane_idx] = (cutlass.Int32(1) if first else cutlass.Int32(0)) + (cutlass.Int32(2) if second else cutlass.Int32(0))
        else:
            xfer_base = cutlass.Array(cutlass.Float32, kernel.STATE_XFER_ROWS * kernel.STATE_XFER_ROW_WORDS, space=SMEM, alignment=128).data_ptr()
            co_off, row_off = kernel.state_scratch_maps(lane_idx)
            rows_per_store = kernel.STATE_XFER_ROWS // 4
            pieces_per_row = kernel.STATE_XFER_ROW_WORDS // 4
            gmem_row = lane_idx // pieces_per_row
            gmem_col = (lane_idx % pieces_per_row) * 4
            if cutlass.const_expr(form == "seed"):
                pieces = kernel.coalesced_state_loads(mState, gmem_row, rows_per_store, gmem_col, cutlass.Int32(0), rounds, ROUND_WORDS)
                kernel.coalesced_state_to_tmem(pieces, xfer_base, co_off, row_off, word, rounds, ROUND_WORDS)
                nvvm.tcgen05_wait("store")
            else:
                bars = None
                if cutlass.const_expr(form == "final_freed_inside"):
                    bars = Bars(MBarrier(cutlass.Array(cutlass.Int64, 1, space=SMEM, alignment=16), try_wait=True, stages=1, init_count=LANES, producer=Producer.THREAD))
                    bars.mb_state_free[0].init()
                    nvvm.fence_mbarrier_init()
                    nvvm.barrier_cta_sync()
                gmem_rows = [(mState.iterator + mState.layout((gmem_row + g * rows_per_store, 0))).raw_ptr() for g in range(pieces_per_row)]
                kernel.coalesced_state_from_tmem(word, xfer_base, co_off, row_off, gmem_rows, gmem_col, rounds, ROUND_WORDS, bars)
                nvvm.tcgen05_wait("load")


    probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


    @cute.jit
    def launch(state: cute.Tensor, out: cute.Tensor, form: cutlass.Constexpr[str], rounds: cutlass.Constexpr[int], keys: cutlass.Constexpr[int], stream: cuda.CUstream):
        probe(state, out, form, rounds, keys, cutlass.Int32(0)).launch(grid=(1, 1, 1), block=(LANES, 1, 1), stream=stream)


    if __name__ == "__main__":
        # argv: <arch> <form> <rounds> <keys> <dump dir>; CUTE_DSL_DUMP_DIR is set by the caller (keeps the ptx).
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

        arch, form, rounds, keys, dump = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        state = make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(), rounds * ROUND_WORDS), stride=(rounds * ROUND_WORDS, 1), assumed_align=16)
        out = make_fake_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride=(1,), assumed_align=4)
        cute.compile(launch, state, out, form, rounds, keys, make_fake_stream(use_tvm_ffi_env_stream=False), options=f"--enable-tvm-ffi --opt-level 2 --gpu-arch {arch} --keep-ptx")
        print("COMPILED", arch, form)
        ptxs = glob.glob(os.path.join(dump, f"*.{arch}.ptx"))
        if len(ptxs) != 1:
            print("FAIL the dump holds", len(ptxs), "ptx files:", os.listdir(dump)); sys.exit(3)
        for line in open(ptxs[0]).read().splitlines():
            print("PTX", line.strip())
    ''')

# What a line of the rendering is for the pin: a store to shared memory, a load from it, the warp's synchronisation, an
# mbarrier arrive (the state region handed to the seed copy inside the final-state store).
_PTX_STEPS = (
    ("put", re.compile(r"^st\.shared\.")),
    ("take", re.compile(r"^ld\.shared\.")),
    ("sync", re.compile(r"^bar\.warp\.sync\b")),
    ("arrive", re.compile(r"^mbarrier\.arrive\b")),
)


def render_handover(tmp_path, form):
    """The shared-memory steps of the probe's rendering for ``form``, in the order of the PTX: ``put``, ``take``, ``sync``
    and ``arrive``, one entry per instruction."""
    probe_py = tmp_path / "state_handover_probe.py"
    probe_py.write_text(_PROBE_SRC)
    dump = tmp_path / f"dump_{form}"
    dump.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="ptx", CUDA_VISIBLE_DEVICES="")
    argv = [sys.executable, str(probe_py), RENDER_ARCH, form, str(RENDER_ROUNDS), str(RENDER_KEYS), str(dump)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, f"{RENDER_ARCH} trace-compile of the probe ({form}) failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = [line[len("PTX ") :] for line in proc.stdout.splitlines() if line.startswith("PTX ")]
    assert f"COMPILED {RENDER_ARCH} {form}" in proc.stdout and lines, proc.stdout[-2000:]
    steps = [name for line in lines for name, pattern in _PTX_STEPS if pattern.match(line)]
    print(f"\n{form}: {len(lines)} PTX lines; shared-memory steps {runs_of(steps)}")
    return steps


def runs_of(steps):
    """``steps`` as runs: ``[(step, how many in a row), ...]``."""
    runs = []
    for step in steps:
        if runs and runs[-1][0] == step:
            runs[-1] = (step, runs[-1][1] + 1)
        else:
            runs.append((step, 1))
    return runs


@pytest.fixture(scope="module", params=HANDOVER_FORMS)
def handover(request, tmp_path_factory):
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string(RENDER_ARCH)
    except Exception:
        pytest.skip(f"this cutlass-dsl has no {RENDER_ARCH}")
    return request.param, render_handover(tmp_path_factory.mktemp("state_handover"), request.param)


@pytest.mark.L0
def test_lanes_hand_over_through_shared_memory_behind_a_warp_synchronisation(handover):
    """The scratch of the coalesced state path and the final-state keys of the seed copy are written by lanes of a warp
    and read by other lanes of it.  In the rendering every run of stores to shared memory is followed by one warp
    synchronisation before the first load, and every run of loads by one before the next store: a state of 128 key
    columns is eight exchanges through the scratch in either direction, 16 synchronisations; the probe's keys are stored
    and compared twice, four synchronisations.  Where the seed copy takes the fp32 state region over inside the
    final-state store, the lanes' arrive stands in the seventh exchange behind the synchronisation of its stores.  How
    many instructions a run of stores or loads is belongs to the toolchain (it may fold neighbouring words into one)."""
    form, steps = handover
    runs = runs_of([step for step in steps if step != "arrive"])
    exchanges = 2 if form == "keys" else RENDER_ROUNDS * 2
    assert [step for step, _ in runs] == list(HANDOVER) * exchanges, f"{form}: {exchanges} hand-overs, each with its two synchronisations -- {runs}"
    assert all(count == 1 for step, count in runs if step == "sync") and steps.count("sync") == 2 * exchanges, f"{form}: {runs}"
    arrives = [at for at, step in enumerate(steps) if step == "arrive"]
    if form == "final_freed_inside":
        assert len(arrives) == 1, f"{form}: one arrive of the state region -- {runs_of(steps)}"
        before = steps[: arrives[0]]
        assert (
            before.count("sync") == 2 * (exchanges - 2) + 1 and before[-1] == "sync" and steps[arrives[0] + 1] == "take"
        ), f"{form}: the arrive stands in the last round's first exchange, between the synchronisation behind its stores and its loads -- {runs_of(steps)}"
    else:
        assert not arrives, f"{form}: {runs_of(steps)}"
