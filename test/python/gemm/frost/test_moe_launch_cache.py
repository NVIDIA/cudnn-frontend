# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replayed variant-pack calls of the grouped plans (cudnn.gemm.frost.launch_cache).

A grouped plan's call takes the full path -- resolve, check, view, fill, launch -- the first time its buffer signature
(each bound buffer's type, shape, strides, dtype, device and address residue, and which buffers share an address)
comes; the second time it takes it recorded and leaves an entry, which later calls with the signature replay on their
own buffers. A replayed call writes the bytes the full path writes, every buffer the full path refuses is refused with
the same error, and an entry holds up under CUDA-graph capture, threads, aliased bindings and the engine's execute. The
module's own rules (no GPU) come first."""

from __future__ import annotations

import concurrent.futures
import math
import threading
import types

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs recorder)
import pytest
import torch

from gemm_test_utils import Plan, graph_binding, requires_sm100
from cudnn.gemm.frost import launch_cache as LC
from cudnn.gemm.frost.dtypes import MAX_MEM_ACCESS_BYTES
from cudnn.gemm.frost.kernel_registry import select_template
from cudnn.gemm.frost.tile_config import by_name
from test_general_block_scale_gated_fp8 import _graph as _gated_graph
from test_general_moe_grouped_block_scale_matmul_fwd import _TEMPLATES, _TILE_128, _config, _graph as _grouped_graph, _offsets
from test_moe_grouped_matmul_fwd import _build_graph as _bf16_graph, _check_moe_quant_scale_k_padding_is_zeroed

pytestmark = pytest.mark.L0

_BLOCK = 128


# The module's rules (no GPU)


def _pool_views(nbytes=1 << 16):
    """``at(offset, shape, dtype)``: a view of one CPU buffer at ``offset`` bytes past a 256-byte boundary."""
    pool = torch.empty(nbytes + 256, dtype=torch.uint8)
    start = -pool.data_ptr() % 256

    def at(offset, shape, dtype=torch.float32):
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        return pool[start + offset : start + offset + size].view(dtype).view(shape)

    return at


def test_signature_separates_every_fact_the_checks_read():
    at = _pool_views()
    a = at(0, (4, 8))
    assert LC.signature([a]) == LC.signature([at(4096, (4, 8))]), "another address with the same residue"
    others = {
        "address residue": at(16, (4, 8)),
        "shape": at(0, (8, 4)),
        "strides": at(0, (8, 4)).t(),
        "dtype": at(0, (4, 8), torch.int32),
        "type": torch.nn.Parameter(at(0, (4, 8)), requires_grad=False),
    }
    if torch.cuda.is_available():
        others["device"] = torch.empty(4, 8, device="cuda")  # a fresh allocation: residue 0, as `a`'s
    for fact, other in others.items():
        assert LC.signature([other]) != LC.signature([a]), fact
    # Two bindings of one buffer are another signature than two buffers of the same facts.
    b = at(4096, (4, 8))
    assert LC.signature([a, b]) != LC.signature([a, a])
    assert LC.signature([a, a]) == LC.signature([b, b])
    # The residue settles every alignment test of up to MAX_MEM_ACCESS_BYTES bytes.
    assert LC.PTR_RESIDUE % MAX_MEM_ACCESS_BYTES == 0


def test_view_steps_rebuild_the_launch_views_and_nothing_else():
    at = _pool_views()
    views = {
        # A weight-by-token weight: the graph-order [E, K, N] view, permuted twice.
        "two permutes": (lambda off: at(off, (4, 8, 16)).transpose(1, 2), lambda w: w.permute(0, 2, 1).permute(1, 2, 0)),
        "reshape": (lambda off: at(off, (5, 1, 1), torch.int32), lambda o: o.reshape([5])),
        "permute and reshape": (lambda off: at(off, (1, 1, 8)), lambda a: a.permute(0, 2, 1).reshape((8, 1))),
        "none": (lambda off: at(off, (4, 8, 16)).transpose(1, 2), lambda w: w),
    }
    for what, (make, build) in views.items():
        src, twin = make(0), make(8192)
        steps = LC.view_steps(src, build(src))
        assert steps is not None, what
        # The steps give any buffer with src's facts the view the launch path builds of it.
        got, want = LC.view(twin, steps), build(twin)
        assert (type(got), got.dtype, tuple(got.shape), tuple(got.stride()), got.data_ptr()) == (
            type(want),
            want.dtype,
            tuple(want.shape),
            tuple(want.stride()),
            want.data_ptr(),
        ), what
    # Another dtype over the same bytes, another address, or a copy: no steps.
    plain = at(0, (4, 8))
    assert LC.view_steps(plain, plain.view(torch.int32)) is None
    assert LC.view_steps(plain, plain[:, 1:]) is None
    assert LC.view_steps(plain.t(), plain.t().reshape(32)) is None


class _Carve:
    """A stand-in for a workspace carve: launch_cache only compares it by identity."""


def _record(launch_args=None, *, carves=None, fills=(), problem=(1, 2, 3)):
    record = LC.LaunchRecord()
    carve = _Carve()
    record.carves = [(5, carve)] if carves is None else carves
    record.fills = list(fills)
    if launch_args is not None:
        record.launches = [(problem, False, tuple(carve if a is None else a for a in launch_args))]
    return record


def test_derive_keeps_only_what_an_entry_can_replay():
    at = _pool_views()
    token, weight = at(0, (1, 8, 16)), at(4096, (4, 16, 8))
    slots = [token, weight]
    entry = LC.derive(_record([token.permute(1, 2, 0), None, weight.permute(1, 2, 0)]), slots, 0)
    assert isinstance(entry, LC.LaunchEntry)
    assert entry.workspace_slots == 5 and entry.problem == (1, 2, 3) and entry.args[1] is None
    assert [a[0] for a in entry.args if a is not None] == [0, 1]
    # A launch argument no bound buffer's steps give, an address no bound buffer has, a fill the plan does not issue,
    # a second carve, a launch without the carve: those calls are not replayable.
    for record, fills in (
        (_record([token.view(torch.int32), None]), 0),
        (_record([at(8192, (1, 8, 16)), None]), 0),
        (_record([token, None], fills=[(token.data_ptr(), [(0, 1, 1, 1)], "zero", 4)]), 0),
        (_record([token, None]), 1),
        (_record([token, None], carves=[(5, _Carve()), (5, _Carve())]), 0),
        (_record([token, weight]), 0),
    ):
        assert LC.derive(record, slots, fills) is LC.UNCACHEABLE
    # Two bindings at one address: either gives the argument in every call of the signature, which says they share it.
    shared = LC.derive(_record([token.permute(1, 2, 0), None]), [token, token], 0)
    assert shared.args[0][0] in (0, 1)
    # Of two bindings at one address, the one whose steps give the argument.
    flat = token.view(8, 16)
    assert LC.derive(_record([token.permute(1, 2, 0), None]), [flat, token], 0).args[0][0] == 1
    # A call that launches nothing (an empty one) replays its carve and fills.
    fill = (weight.data_ptr(), [(0, 1, 8, 1)], "zero", 4)
    empty = LC.derive(_record(None, fills=[fill]), slots, 1)
    assert empty.problem is None and empty.fills == ((1, [(0, 1, 8, 1)], "zero", 4),)


def test_cache_bounds_its_entries_and_its_signatures_seen_once():
    cache = LC.LaunchCache(types.SimpleNamespace(bound_tensors=lambda: [object()]), 0)
    keys = [("signature", i) for i in range(LC.MAX_ENTRIES + 6)]
    # Signatures seen once: the oldest is dropped past the bound, and seen again it is a first sighting again.
    assert all(cache.first_sighting(key) for key in keys)
    assert len(cache.seen) == LC.MAX_ENTRIES and not cache.first_sighting(keys[-1]) and cache.first_sighting(keys[0])
    # Entries: the same bound, and an entry moves its signature out of the seen ones.
    for key in keys:
        cache.learn(key, _record(), [])
    assert list(cache.entries) == keys[6:]
    assert keys[0] not in cache.seen


def test_a_record_is_this_threads_alone():
    seen = []
    with LC.recording() as record:
        assert LC.active_record() is record
        worker = threading.Thread(target=lambda: seen.append(LC.active_record()))
        worker.start()
        worker.join()
    assert seen == [None] and LC.active_record() is None


# Replayed calls of compiled plans

_E, _S, _N, _K = 8, 512, 256, 512
_SRC = 300  # GATHER's token rows
_TOP_K = 2

_CASES = {
    "none": dict(kind="grouped", config=_TILE_128),
    "none_swap": dict(kind="grouped", config=_config(128, 32), swap=True),
    "gather": dict(kind="grouped", config=_TILE_128, gather=True),
    "gather_swap": dict(kind="grouped", config=_config(128, 32), swap=True, gather=True),
    "combine": dict(kind="grouped", config=_TILE_128, combine=True),
    "combine_swap": dict(kind="grouped", config=_config(128, 16), swap=True, combine=True),
    "gated_fp8_gather_swap": dict(kind="gated", config=_config(128, 16), swap=True),
    "bf16": dict(kind="bf16"),
    "bf16_amax": dict(kind="bf16", amax=True),
}


class _Case:
    """A grouped plan, and per ``buffers(seed)`` call new allocations of one signature: inputs drawn from ``seed``, outputs
    poisoned (NaN, or 0x7F bytes for FP8 codes)."""

    def __init__(self, name):
        spec = _CASES[name]
        self.kind, self.swap = spec["kind"], spec.get("swap", False)
        self.gather, self.combine, self.amax = spec.get("gather", False), spec.get("combine", False), spec.get("amax", False)
        if self.kind == "grouped":
            self.offsets = _offsets(_S, _E)
            g, self.t = _grouped_graph(
                _E, _SRC if self.gather else _S, _N, _K, _E, gather_rows=_S if self.gather else None, combine=_TOP_K if self.combine else None
            )
            self.plan = Plan(g, config=by_name(spec["config"]), swap_ab=self.swap)
            assert select_template(self.plan.chain, self.plan._compiled.config).file == _TEMPLATES[(self.gather, self.swap)]
        elif self.kind == "gated":
            self.offsets = _offsets(_S, _E)
            g, self.t = _gated_graph("gather", _E, _SRC, _N, _K, rows=_S, gemms=2, fp8=True)
            self.plan = Plan(g, config=by_name(spec["config"]), swap_ab=self.swap)
        else:
            self.offsets = [0, 64, 64, 184, 256]  # groups of 64, 0, 120 and 72 rows
            kw = dict(reduction_mode=cudnn.reduction_mode.AMAX, reduction_dims=(1, 1, 1)) if self.amax else {}
            g = _bf16_graph(4, 256, 128, 128, **kw)
            self.plan = Plan(g, config=by_name("CONFIG_sm100_128x256x128_128x256x32_cluster1x1"))
            self.binding = graph_binding(self.plan._compiled)
        # The rows a call writes: a COMBINE call zeroes and writes every token's row.
        self.used = None if self.combine else self.offsets[-1]

    def buffers(self, seed):
        dev = "cuda"
        gen = torch.Generator(device=dev).manual_seed(seed)

        def fp8(*shape):
            return (torch.randn(*shape, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)

        def scale(*shape):
            return torch.rand(*shape, device=dev, generator=gen) + 0.5

        kb, nb = _K // _BLOCK, _N // _BLOCK
        if self.kind == "bf16":
            S, N, K = 256, 128, 128
            token = torch.randn(1, S, K, device=dev, generator=gen).to(torch.bfloat16)
            weight = torch.randn(4, N, K, device=dev, generator=gen).to(torch.bfloat16)
            outs = [torch.full((1, S, N), float("nan"), device=dev, dtype=torch.bfloat16)]
            if self.amax:
                outs.append(torch.full((1, 1, 1), float("nan"), device=dev))
            b = self.binding
            pack = {b.a_operands[0]: token, b.b_operands[0]: weight, b.first_token_offset: torch.tensor(self.offsets, dtype=torch.int32, device=dev)}
            pack.update(zip(b.outputs, outs))
            return pack, outs
        t = self.t
        src = _SRC if self.gather or self.kind == "gated" else _S
        pack = {t["tok"]: fp8(1, src, _K), t["sfa"]: scale(1, src, kb), t["fto"]: torch.tensor(self.offsets, dtype=torch.int32, device=dev).view(-1, 1, 1)}
        if self.gather or self.kind == "gated":
            pack[t["idx"]] = torch.randint(0, src, (1, _S, 1), device=dev, generator=gen, dtype=torch.int32)
        if self.kind == "gated":
            for w, sf in zip(t["w"], t["sfb"]):
                pack[w], pack[sf] = fp8(_E, _N, _K).transpose(1, 2), scale(_E, nb, kb).transpose(1, 2)
            codes = torch.full((1, _S, _N), 0x7F, device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
            scales = torch.full((1, _S, _N // _BLOCK), float("nan"), device=dev)
            pack[t["q"]], pack[t["qs"]] = codes, scales
            return pack, [codes, scales]
        pack[t["w"]], pack[t["sfb"]] = fp8(_E, _N, _K).transpose(1, 2), scale(_E, nb, kb).transpose(1, 2)
        if self.combine:
            dest = torch.randperm(_S, device=dev, generator=gen).to(torch.int32)
            pack[t["idx"]], pack[t["ks"]] = (dest // _TOP_K).view(1, _S, 1), (dest % _TOP_K).view(1, _S, 1)
            pack[t["scores"]] = torch.rand(1, _S // _TOP_K, _TOP_K, device=dev, generator=gen)
            out = torch.full((1, _S // _TOP_K, _N), float("nan"), device=dev)
        else:
            out = torch.full((1, _S, _N), float("nan"), device=dev, dtype=torch.bfloat16)
        pack[t["out"]] = out
        return pack, [out]

    def warm(self, seeds=(100, 101)):
        """A signature's first call (the full path) and second (the full path, recorded): an entry for the calls after."""
        for seed in seeds:
            self.plan(self.buffers(seed)[0])
        torch.cuda.synchronize()

    def assert_same(self, got, want):
        """The bytes of the full path's outputs; COMBINE's fp32 sums to rounding (its atomics land in any order)."""
        for g, w in zip(got, want):
            written = g if self.used is None or g.shape[1] == 1 else g[:, : self.used]
            assert bool(torch.isfinite(written.float()).all()), "a written row kept its poison"
            if self.combine:
                torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-6)
            else:
                assert torch.equal(g.view(torch.uint8), w.view(torch.uint8))


def _full_path_calls(monkeypatch, plan):
    """A list that grows by one per call that takes the plan's full path (``_call_variant_pack``)."""
    cls = type(plan._compiled)
    real = cls._call_variant_pack
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(cls, "_call_variant_pack", counted)
    return calls


def _replay(monkeypatch, plan, on):
    monkeypatch.setattr(type(plan._compiled), "_replay_calls", on, raising=False)


@requires_sm100
@pytest.mark.parametrize("name", list(_CASES))
def test_replayed_call_writes_what_the_full_path_writes(name, monkeypatch):
    """The third call of a signature, with new buffers, does not take the full path, and writes the full path's bytes
    for those buffers into outputs poisoned before it (a COMBINE output and an amax seed are filled by the replay)."""
    case = _Case(name)
    full_path = _full_path_calls(monkeypatch, case.plan)
    case.warm()
    assert len(full_path) == 2
    pack, got = case.buffers(1)
    case.plan(pack)
    torch.cuda.synchronize()
    assert len(full_path) == 2, "a call with the signature of two earlier calls took the full path"
    _replay(monkeypatch, case.plan, False)
    pack, want = case.buffers(1)
    case.plan(pack)
    torch.cuda.synchronize()
    assert len(full_path) == 3
    case.assert_same(got, want)


class _CountingFill:
    """A plan's COMBINE output fill (sm100/combine_zero.CombineZero) that notes the start of each region it zeroes."""

    def __init__(self, fill):
        self.fill, self.bases = fill, []

    def takes(self, *region):
        return self.fill.takes(*region)

    def __call__(self, base, *rest):
        self.bases.append(base)
        return self.fill(base, *rest)


@requires_sm100
@pytest.mark.parametrize("name", ["combine", "combine_swap"])
def test_replayed_general_combine_zeroes_its_output_with_the_fill_kernel(name, monkeypatch):
    """A general grouped COMBINE plan zeroes its output with its programmatic fill (sm100/combine_zero.py), not a memset,
    and so does the replay of its calls: on the replayed call's own output, with the full path's bytes."""
    from cudnn.frost import buffers
    from cudnn.gemm.frost.sm100.combine_zero import CombineZero

    case = _Case(name)
    assert isinstance(case.plan._compiled.combine_zero, CombineZero)
    fill = _CountingFill(case.plan._compiled.combine_zero)
    monkeypatch.setattr(case.plan._compiled, "combine_zero", fill)
    monkeypatch.setattr(buffers, "apply_zero_fill_plan", lambda *args: pytest.fail("a general COMBINE call zeroed its output with the memset"))
    full_path = _full_path_calls(monkeypatch, case.plan)
    case.warm()
    pack, got = case.buffers(1)
    case.plan(pack)
    torch.cuda.synchronize()
    assert len(full_path) == 2, "a call with the signature of two earlier calls took the full path"
    assert len(fill.bases) == 3 and fill.bases[-1] == got[0].data_ptr()
    _replay(monkeypatch, case.plan, False)
    pack, want = case.buffers(1)
    case.plan(pack)
    torch.cuda.synchronize()
    case.assert_same(got, want)


def _misaligned(buf, offset=1):
    """A copy of ``buf`` at ``offset`` bytes past an allocation's start."""
    raw = torch.empty(buf.numel() * buf.element_size() + 256, device=buf.device, dtype=torch.uint8)
    copy = raw[offset : offset + buf.numel() * buf.element_size()].view(buf.dtype).view(buf.shape)
    copy.copy_(buf)
    return copy


def _replace(key, make):
    def bad(case, pack):
        pack = dict(pack)
        pack[case.t[key]] = make(pack[case.t[key]])
        return pack

    return bad


def _drop(key):
    def bad(case, pack):
        pack = dict(pack)
        del pack[case.t[key]]
        return pack

    return bad


def _add_unknown(case, pack):
    return {**pack, "no tensor of this graph": pack[case.t["tok"]]}


_BAD = {
    "token_misaligned": _replace("tok", _misaligned),
    "token_dtype": _replace("tok", lambda b: b.to(torch.bfloat16)),
    "token_k": _replace("tok", lambda b: b[:, :, : _K - _BLOCK]),
    "weight_scale_k_blocks": _replace("sfb", lambda b: b[:, :-1]),
    "token_index_dtype": _replace("idx", lambda b: b.to(torch.int64)),
    "token_index_rows": _replace("idx", lambda b: b[:, : _S // 2]),
    "output_dtype": _replace("out", lambda b: b.float()),
    "missing_key": _drop("sfa"),
    "unknown_key": _add_unknown,
}


@requires_sm100
@pytest.mark.parametrize("bad", list(_BAD))
@pytest.mark.parametrize("name", ["gather_swap", "gather"])
def test_replayed_plan_refuses_what_the_full_path_refuses(name, bad, monkeypatch):
    """With an entry for good buffers in place, a call that binds a bad one -- three times, as a signature takes to
    reach an entry -- raises the full path's error, the same type and message, and writes nothing."""
    case = _Case(name)
    case.warm()
    pack, outs = case.buffers(1)
    pack = _BAD[bad](case, pack)
    errors = []
    for replay in (True, True, True, False):
        _replay(monkeypatch, case.plan, replay)
        with pytest.raises(Exception) as info:
            case.plan(pack)
        torch.cuda.synchronize()
        errors.append((type(info.value), str(info.value)))
        assert all(bool(torch.isnan(out.float()).all()) for out in outs if out.dtype != torch.float8_e4m3fn)
    assert len(set(errors)) == 1, errors


@requires_sm100
def test_replayed_plan_still_checks_each_calls_workspace(monkeypatch):
    """The workspace is no part of a signature: each replayed call validates its own."""
    case = _Case("none_swap")
    case.warm()
    pack, outs = case.buffers(1)
    for replay in (True, False):
        _replay(monkeypatch, case.plan, replay)
        with pytest.raises(ValueError, match="needs a .*-byte workspace, got 16 bytes"):
            case.plan(pack, workspace=torch.empty(16, dtype=torch.uint8, device="cuda"))
        unaligned = torch.empty(case.plan.workspace_bytes + 256, dtype=torch.uint8, device="cuda")[64:]
        with pytest.raises(ValueError, match="must be 128-byte aligned"):
            case.plan(pack, workspace=unaligned)
    assert bool(torch.isnan(outs[0].float()).all())


@requires_sm100
@pytest.mark.no_workspace_shim
def test_replayed_plan_refuses_a_call_without_a_workspace(monkeypatch):
    case = _Case("gather")
    full_path = _full_path_calls(monkeypatch, case.plan)
    workspace = torch.empty(case.plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    for seed in (0, 1):
        case.plan._compiled(case.buffers(seed)[0], workspace=workspace)
    with pytest.raises(ValueError, match="requires a .*-byte workspace but execute\\(\\) received none"):
        case.plan._compiled(case.buffers(2)[0])
    assert len(full_path) == 2, "the call without a workspace was a replay"


@requires_sm100
@pytest.mark.parametrize("swap", [False, True], ids=["none", "none_swap"])
def test_replayed_empty_call_launches_and_writes_nothing(swap, monkeypatch):
    """An expert-parallel rank without routed rows: the replay of an empty call returns before any launch."""
    case = _Case("none_swap" if swap else "none")
    full_path = _full_path_calls(monkeypatch, case.plan)
    t, dev = case.t, "cuda"

    def empty_call():
        backing = torch.full((1, 8, _N), float("nan"), device=dev, dtype=torch.bfloat16)
        pack = {
            t["tok"]: torch.zeros(1, 8, _K, device=dev).to(torch.float8_e4m3fn)[:, :0],
            t["w"]: torch.ones(_E, _N, _K, device=dev).to(torch.float8_e4m3fn).transpose(1, 2),
            t["sfa"]: torch.ones(1, 8, _K // _BLOCK, device=dev)[:, :0],
            t["sfb"]: torch.ones(_E, _N // _BLOCK, _K // _BLOCK, device=dev).transpose(1, 2),
            t["fto"]: torch.zeros(_E + 1, 1, 1, dtype=torch.int32, device=dev),
            t["out"]: backing[:, :0],
        }
        assert case.plan(pack) is None
        torch.cuda.synchronize()
        assert bool(torch.isnan(backing.float()).all())

    empty_call()
    empty_call()

    def no_launch(*args, **kwargs):
        pytest.fail("a call with nothing to compute reached the launch")

    monkeypatch.setattr(case.plan._compiled, "_launchable", no_launch)
    empty_call()
    assert len(full_path) == 2


@requires_sm100
def test_an_address_residue_the_entry_has_not_seen_takes_the_full_path(monkeypatch):
    """A token buffer 16 bytes past an allocation (aligned enough, another residue) is another signature: the full path
    checks its first two calls, the third replays, each with the full path's bytes."""
    case = _Case("none_swap")
    full_path = _full_path_calls(monkeypatch, case.plan)
    case.warm()
    results = []
    for seed in (1, 2, 3):
        pack, outs = case.buffers(seed)
        pack[case.t["tok"]] = _misaligned(pack[case.t["tok"]], offset=16)
        case.plan(pack)
        torch.cuda.synchronize()
        results.append(outs[0].clone())
    assert len(full_path) == 4, "the first two calls at the new residue take the full path, the third replays"
    _replay(monkeypatch, case.plan, False)
    pack, want = case.buffers(3)
    pack[case.t["tok"]] = _misaligned(pack[case.t["tok"]], offset=16)
    case.plan(pack)
    torch.cuda.synchronize()
    case.assert_same([results[2]], want)


@requires_sm100
def test_replay_resolves_uid_and_name_keys(monkeypatch):
    """A pack keyed by tensor names (and one tensor object) resolves through graph_analyzer.resolve_variant_pack and
    replays like one keyed by the tensor objects."""
    case = _Case("gather")
    full_path = _full_path_calls(monkeypatch, case.plan)

    def by_name(pack):
        return {(t if t is case.t["out"] else t.get_name()): buf for t, buf in pack.items()}

    for seed in (100, 101):
        case.plan(by_name(case.buffers(seed)[0]))
    pack, got = case.buffers(1)
    case.plan(by_name(pack))
    torch.cuda.synchronize()
    assert len(full_path) == 2
    _replay(monkeypatch, case.plan, False)
    pack, want = case.buffers(1)
    case.plan(pack)
    torch.cuda.synchronize()
    case.assert_same(got, want)


@requires_sm100
def test_aliased_bindings_are_a_signature_of_their_own(monkeypatch):
    """A gated pair whose two weights (and scales) are one buffer is another signature than two buffers of the same
    facts: an entry of either serves only its own calls, each with the full path's bytes."""
    case = _Case("gated_fp8_gather_swap")
    full_path = _full_path_calls(monkeypatch, case.plan)
    t = case.t

    def aliased(seed):
        pack, outs = case.buffers(seed)
        pack[t["w"][1]], pack[t["sfb"][1]] = pack[t["w"][0]], pack[t["sfb"][0]]
        return pack, outs

    case.warm()
    for seed in (102, 103):
        case.plan(aliased(seed)[0])
    assert len(full_path) == 4, "the aliased binding replayed the entry of two buffers"
    got = []
    for make in (aliased, case.buffers):
        pack, outs = make(1)
        case.plan(pack)
        torch.cuda.synchronize()
        got.append(outs)
    assert len(full_path) == 4
    _replay(monkeypatch, case.plan, False)
    for make, outs in zip((aliased, case.buffers), got):
        pack, want = make(1)
        case.plan(pack)
        torch.cuda.synchronize()
        case.assert_same(outs, want)


@requires_sm100
@pytest.mark.parametrize("swap_ab", [False, True], ids=["token_by_weight", "weight_by_token"])
def test_replayed_call_keeps_the_k_padding_kernel(swap_ab, monkeypatch):
    """A call whose quantized feature axis ends a scale row inside a quad (5 blocks of 32 pad to 8) takes the plan's
    K-padding kernel, and so do its replays: each of four runs zeroes the padding slots of a 0xFF F8_128x4 blob."""
    from cudnn.gemm.frost.sm100 import compiler as sm100

    calls = []
    real = sm100.CompiledMoeGemm._call_variant_pack
    monkeypatch.setattr(sm100.CompiledMoeGemm, "_call_variant_pack", lambda self, *a, **kw: calls.append(1) or real(self, *a, **kw))
    _check_moe_quant_scale_k_padding_is_zeroed(by_name(_TILE_128), -1, swap_ab, [([100, 0, 120, 80], 160)] * 4)
    assert len(calls) == 2, "the runs after a signature's second took the full path"


@requires_sm100
@pytest.mark.parametrize("name", ["combine_swap", "bf16_amax", "gather"])
def test_replayed_call_captured_in_a_cuda_graph(name, monkeypatch):
    """A replayed call captured into a CUDA graph holds its output fills and its launch: each replay writes the eager
    full path's bytes into an output poisoned before it."""
    case = _Case(name)
    case.warm()
    pack, outs = case.buffers(1)
    workspace = torch.empty(case.plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    full_path = _full_path_calls(monkeypatch, case.plan)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                case.plan._compiled(pack, workspace=workspace, stream=torch.cuda.current_stream().cuda_stream)
        assert len(full_path) == 0, "the captured call took the full path"
        replays = []
        for _ in range(2):
            for out in outs:
                out.view(torch.uint8).fill_(0x7F) if out.dtype == torch.float8_e4m3fn else out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            replays.append([out.clone() for out in outs])
    finally:
        torch.cuda.synchronize()
        graph.reset()
    _replay(monkeypatch, case.plan, False)
    want_pack, want = case.buffers(1)
    case.plan(want_pack)
    torch.cuda.synchronize()
    for got in replays:
        case.assert_same(got, want)


@requires_sm100
def test_replayed_calls_from_threads(monkeypatch):
    """Threads replaying one plan's entry at once, each with its own buffers and workspace, each write the bytes of
    the full path for their buffers."""
    case = _Case("gather_swap")
    seeds = list(range(1, 9))
    _replay(monkeypatch, case.plan, False)
    want = {}
    for seed in seeds:
        pack, outs = case.buffers(seed)
        case.plan(pack)
        want[seed] = outs
    torch.cuda.synchronize()
    _replay(monkeypatch, case.plan, True)
    case.warm()
    full_path = _full_path_calls(monkeypatch, case.plan)

    def worker(seed):
        workspace = torch.empty(case.plan.workspace_bytes, dtype=torch.uint8, device="cuda")
        got = []
        for _ in range(3):
            pack, outs = case.buffers(seed)
            case.plan._compiled(pack, workspace=workspace)
            torch.cuda.synchronize()
            got.append(outs)
        return seed, got

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        results = list(pool.map(worker, seeds))
    assert len(full_path) == 0
    for seed, got in results:
        for outs in got:
            case.assert_same(outs, want[seed])


@requires_sm100
def test_engine_execute_replays(monkeypatch):
    """graph.execute through frost_gemm hands the plan the variant pack's operand buffers: a third execute with new
    buffers replays, and writes the full path's bytes."""
    case = _Case("gather")
    g, t = _grouped_graph(_E, _SRC, _N, _K, _E, gather_rows=_S)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    g.select_plan(next(i for i, n in enumerate(names) if n == "frost_gemm" or n.startswith("frost_gemm[")))
    g.check_support()
    g.build_plans()
    assert g.selected_engine.name == "frost_gemm"
    workspace = torch.empty(max(1, g.get_workspace_size()), dtype=torch.uint8, device="cuda")

    def execute(seed):
        pack, outs = case.buffers(seed)
        g.execute({t[key]: pack[case.t[key]] for key in ("tok", "w", "sfa", "sfb", "fto", "idx", "out")}, workspace)
        torch.cuda.synchronize()
        return outs

    from cudnn.gemm.frost.sm100 import compiler as sm100

    calls = []
    real = sm100.CompiledMoeBlockScaleGemm._call_variant_pack
    monkeypatch.setattr(sm100.CompiledMoeBlockScaleGemm, "_call_variant_pack", lambda self, *a, **kw: calls.append(1) or real(self, *a, **kw))
    execute(100)
    execute(101)
    got = execute(1)
    assert len(calls) == 2, "the third execute took the full path"
    monkeypatch.setattr(sm100.CompiledMoeBlockScaleGemm, "_replay_calls", False)
    case.assert_same(got, execute(1))
