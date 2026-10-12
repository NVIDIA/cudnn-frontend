# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The surface FlashInfer consumes, pinned name by name with its signature (host-only: no GPU, no DSL needed).

FlashInfer's cuDNN backend (``flashinfer/cudnn/{prefill,decode,utils}.py``) reads a handful of frontend facts before it
routes work to this package: the native SDPA binder's ``supports_*`` flags, ``graph.selected_engine`` (and its ``.name``
against the FROST engine names), ``graph.backend_plan_entries()``, the CuTe DSL gate in ``cudnn.frost.buffers``, the
torch dtype mapping, the unified attention implementation, the graph helpers and the two ``pygraph`` caller hints.
Every one of those is a caller boundary whatever its spelling (python/cudnn/AGENTS.md Rule 9), so this module asserts
each still resolves AS SPELLED and that its public twin exists -- a rename fails here, in this repository's CI, instead
of silently degrading FlashInfer to its fallback path.

The arch-aware probe gets the cell that motivated it: on a cc 10.7 part with a 4.7.x public DSL (no ``sm_107a`` target)
every FROST row declines, so ``cutedsl_runtime_available((10, 7))`` must read False while the device-blind form -- the
check FlashInfer spells today -- reads True.  Modelled by pinning the DSL state exactly where the engines read it;
nothing is imported from the DSL.
"""

from __future__ import annotations

import inspect
import json
import pathlib
import subprocess
import sys

import pytest

import cudnn
import cudnn.frost
from cudnn import datatypes
from cudnn.engines import manifest
from cudnn.frost import buffers

pytestmark = pytest.mark.L0

# The binder flags FlashInfer reads by name today (flashinfer/cudnn/prefill.py), then the rest the build declares.
BINDER_FLAGS_READ_BY_FLASHINFER = ("supports_stats_stride_override", "supports_nonpaged_packed_split")
BINDER_FLAGS = BINDER_FLAGS_READ_BY_FLASHINFER + (
    "supports_paged_packed_split",
    "supports_paged_split_sink",
    "supports_paged_d64_packed_split",
    "supports_paged_d256_packed_split",
    "supports_nonpaged_d256_packed_split",
    "supports_nonpaged_d128_packed_split",
)
# The engine names FlashInfer compares ``selected_engine.name`` against -> their manifest slots (forever: slots are
# append-only, never renumbered or reused).
ENGINE_NAMES = {"sdpa_fwd_prefill_sm100": 11, "sdpa_fwd_prefill_sm107": 15, "sdpa_fwd_prefill_sm120": 5}
PUBLIC_DSL = "nvidia-cutlass-dsl"


def _binder():
    binder = getattr(cudnn._pybind_module, "_SdpaThdBinder", None)
    assert binder is not None, "cudnn._pybind_module._SdpaThdBinder is gone: FlashInfer reads its supports_* flags by name"
    return binder


# ----------------------------------------------------------------------------------------------------------------------
# The binder flags and their public spelling
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("flag", BINDER_FLAGS)
def test_binder_flag_resolves_as_spelled(flag):
    """A static read-only property reading True on the CLASS (FlashInfer never instantiates the binder for it)."""
    binder = _binder()
    assert getattr(binder, flag, None) is True, f"{flag} is not a True static property of _SdpaThdBinder"
    assert getattr(binder, flag, False) is True  # the getattr(..., False) spelling FlashInfer uses


def test_sdpa_capabilities_is_the_public_spelling_of_the_binder_flags():
    caps = cudnn.sdpa_capabilities()
    assert isinstance(caps, frozenset)
    pinned = {flag[len("supports_") :] for flag in BINDER_FLAGS}
    missing = pinned - caps
    assert not missing, f"capabilities renamed or removed (append-only contract): {sorted(missing)}"
    binder = _binder()
    for cap in caps:
        assert isinstance(cap, str) and cap and not cap.startswith("supports_"), cap
        assert getattr(binder, "supports_" + cap, None) is True, f"{cap!r} has no True supports_{cap} twin on the binder"
    assert "sdpa_capabilities" in cudnn.__all__
    assert list(inspect.signature(cudnn.sdpa_capabilities).parameters) == []


def test_sdpa_capabilities_is_empty_without_the_binder(monkeypatch):
    """A build without the native binder reports no capability instead of raising."""

    class _NoBinder:
        pass

    monkeypatch.setattr(cudnn, "_pybind_module", _NoBinder())
    assert cudnn.sdpa_capabilities() == frozenset()


# ----------------------------------------------------------------------------------------------------------------------
# pygraph: selected_engine, backend_plan_entries, the two caller hints, the graph helpers
# ----------------------------------------------------------------------------------------------------------------------


def test_selected_engine_is_a_property_and_backend_plan_entries_a_query():
    assert isinstance(inspect.getattr_static(cudnn.pygraph, "selected_engine"), property)
    assert list(inspect.signature(cudnn.pygraph.backend_plan_entries).parameters) == ["self"]
    assert list(inspect.signature(cudnn.pygraph.get_plan_name_at_index).parameters) == ["self", "index"]
    assert list(inspect.signature(cudnn.pygraph.get_engine_and_knobs_at_index).parameters) == ["self", "index"]
    # The two names FlashInfer's utils probe with hasattr() before it builds a graph.
    assert hasattr(cudnn.pygraph, "backend_plan_entries")
    assert hasattr(cudnn.pygraph, "selected_engine")


def test_pygraph_takes_the_two_caller_hints_and_the_graph_helper_forwards_them():
    ctor = inspect.signature(cudnn.pygraph.__init__).parameters
    override = ctor["is_override_shape_enabled"]
    assert override.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and override.default is False
    replay = ctor["is_cuda_graph_replay_expected"]
    assert replay.kind is inspect.Parameter.KEYWORD_ONLY and replay.default is False
    # FlashInfer's cudnn_frontend_accepts_cuda_graph_replay_hint(): the keyword on the ctor AND **kwargs on cudnn.graph.
    helper = inspect.signature(cudnn.graph).parameters
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in helper.values()), "cudnn.graph no longer forwards graph kwargs"
    assert "handle" in helper and "io_data_type" in helper
    assert "heur_modes" in inspect.signature(cudnn.jit).parameters
    assert list(inspect.signature(cudnn.graph_cache).parameters) == ["key_fn", "maxsize"]


def test_the_engine_names_flashinfer_compares_against_are_stable_identifiers():
    family = next(f for f in manifest.MANIFEST if f.name == "frost_sdpa_fwd")
    for name, slot in ENGINE_NAMES.items():
        assert name in family.slots, f"{name} left the manifest: FlashInfer compares selected_engine.name against it"
        assert family.slots[name].slot == slot, f"{name} moved from slot {slot}: manifest slots are forever"
        engine = manifest.engine_for_id(family.engine_id + slot)
        assert engine is not None and engine.name == name, (name, engine)


def test_unified_implementation_graph_helpers_and_decline_type_resolve():
    assert cudnn.attention_implementation.UNIFIED is not None
    assert cudnn.heur_mode.A is not None
    assert issubclass(cudnn.cudnnGraphNotSupportedError, Exception)
    assert callable(cudnn.graph_cache) and callable(cudnn.jit) and callable(cudnn.graph)
    for name in ("graph_cache", "jit", "graph", "pygraph", "attention_implementation", "heur_mode", "cudnnGraphNotSupportedError"):
        assert name in cudnn.__all__, name


# ----------------------------------------------------------------------------------------------------------------------
# The CuTe DSL gate: the names FlashInfer reads today and the arch-aware public probe
# ----------------------------------------------------------------------------------------------------------------------


def test_dsl_gate_names_resolve_as_spelled():
    assert buffers.CUTEDSL_MIN_VERSION == (4, 7, 0)
    assert list(inspect.signature(buffers.cutedsl_state).parameters) == []
    assert list(inspect.signature(buffers.cutedsl_too_old).parameters) == ["version"]
    assert list(inspect.signature(buffers.cutedsl_arch_requirement_error).parameters) == ["device_cc"]
    assert list(inspect.signature(buffers.cutedsl_requirement_error).parameters) == ["what"]
    installed, version = buffers.cutedsl_state()
    assert isinstance(installed, bool)
    assert version is None or (isinstance(version, tuple) and len(version) == 2)


def test_cutedsl_runtime_available_signature_and_export():
    params = inspect.signature(cudnn.frost.cutedsl_runtime_available).parameters
    assert list(params) == ["device_cc"] and params["device_cc"].default is None
    assert "cutedsl_runtime_available" in cudnn.frost.__all__


def test_device_blind_probe_equals_the_version_gate_flashinfer_spells_today():
    """On this install, the device-blind form is exactly FlashInfer's current formula over the frontend's own gate."""
    installed, version = buffers.cutedsl_state()
    assert cudnn.frost.cutedsl_runtime_available() is (bool(installed) and not buffers.cutedsl_too_old(version))


@pytest.fixture
def dsl_state(monkeypatch):
    """Pin the DSL state where the engines read it: the metadata pair and the sm_107a target probe."""

    def pin(installed, version, has_sm107):
        monkeypatch.setattr(buffers, "_DSL_STATE", (installed, version))
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: has_sm107)

    return pin


@pytest.mark.parametrize(
    "label, installed, version, has_sm107, blind, cc100, cc107",
    [
        ("dsl absent", False, None, False, False, False, False),
        ("4.6.2 (the downstream floor, below ours)", True, (PUBLIC_DSL, "4.6.2"), False, False, False, False),
        ("4.7.1 without sm_107a", True, (PUBLIC_DSL, "4.7.1"), False, True, True, False),
        ("4.7.0rc1 without sm_107a", True, (PUBLIC_DSL, "4.7.0rc1"), False, True, True, False),
        ("4.8.0 with sm_107a", True, (PUBLIC_DSL, "4.8.0"), True, True, True, True),
        ("internal build without sm_107a", True, ("nvidia-cutlass-dsl-internal", "0.3.0+2026"), False, True, True, False),
    ],
)
def test_cutedsl_runtime_available_is_arch_aware(dsl_state, label, installed, version, has_sm107, blind, cc100, cc107):
    """The floor is 4.7.0 everywhere and 4.8.0 on cc 10.7: the (10, 7) + 4.7.x row is the cell a device-blind probe
    cannot see -- it reads True there while every FROST row declines."""
    dsl_state(installed, version, has_sm107)
    probe = cudnn.frost.cutedsl_runtime_available
    assert probe() is blind, label
    assert probe(None) is blind, label
    assert probe((8, 0)) is cc100, label  # no target gate below cc 10.7
    assert probe((10, 0)) is cc100, label
    assert probe((10, 7)) is cc107, label
    assert probe([10, 7]) is cc107, label  # a list spells the same part


def test_the_blind_spot_cell_cc107_with_a_47x_dsl(dsl_state):
    dsl_state(True, (PUBLIC_DSL, "4.7.1"), False)
    assert cudnn.frost.cutedsl_runtime_available() is True
    assert cudnn.frost.cutedsl_runtime_available((10, 7)) is False
    # The same install answers the engines' own gate the same way: the probe IS their decision.
    assert buffers.cutedsl_arch_requirement_error((10, 7)) is not None
    assert buffers.cutedsl_arch_requirement_error((10, 0)) is None


@pytest.mark.parametrize("bad", [(10, 7, 0), (10,), 107, "10.7", (10.5, 7)])
def test_cutedsl_runtime_available_refuses_a_malformed_cc(dsl_state, bad):
    """A cc that is not a pair of integers is refused, never silently read as 'not cc 10.7'."""
    dsl_state(True, (PUBLIC_DSL, "4.7.1"), False)
    with pytest.raises(TypeError):
        cudnn.frost.cutedsl_runtime_available(bad)


def test_arch_aware_probe_touches_the_dsl_only_for_a_cc107_part(dsl_state, monkeypatch):
    dsl_state(True, (PUBLIC_DSL, "4.7.1"), False)
    calls = []
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: calls.append(1) or False)
    assert cudnn.frost.cutedsl_runtime_available() is True
    assert cudnn.frost.cutedsl_runtime_available((10, 0)) is True
    assert calls == []  # neither the blind form nor another part asks the DSL
    assert cudnn.frost.cutedsl_runtime_available((10, 7)) is False
    assert calls == [1]


# ----------------------------------------------------------------------------------------------------------------------
# The torch dtype mapping
# ----------------------------------------------------------------------------------------------------------------------


def test_torch_dtype_mapping_public_and_private_spellings_are_one_function():
    assert datatypes.torch_to_cudnn_data_type is datatypes._torch_to_cudnn_data_type
    assert list(inspect.signature(datatypes.torch_to_cudnn_data_type).parameters) == ["torch_data_type"]
    assert datatypes.torch_to_cudnn_data_type(object()) is None  # an unmapped dtype is None, not an exception


def test_torch_dtype_mapping_values():
    torch = pytest.importorskip("torch")
    f = datatypes.torch_to_cudnn_data_type
    assert f(torch.bfloat16) is cudnn.data_type.BFLOAT16
    assert f(torch.float16) is cudnn.data_type.HALF
    assert f(torch.float32) is cudnn.data_type.FLOAT
    assert f(torch.int32) is cudnn.data_type.INT32
    assert f(torch.float8_e4m3fn) is cudnn.data_type.FP8_E4M3
    assert f(torch.complex64) is None


# ----------------------------------------------------------------------------------------------------------------------
# import cudnn stays cheap; the docs name every public spelling
# ----------------------------------------------------------------------------------------------------------------------

_HEAVY = ("torch", "cutlass", "nvidia_cutlass_dsl")


def _modules_after(code: str) -> set:
    probe = code + "\nimport json, sys; print('@@' + json.dumps(sorted(sys.modules)))"
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert run.returncode == 0, f"probe failed: {run.stderr.strip()[-600:]}"
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("@@"))
    return set(json.loads(line[2:]))


def test_import_cudnn_and_the_public_probes_pull_no_framework():
    """The new public names cost no torch / DSL import: the capability set is read off the binder, the device-blind
    probe and a non-cc-10.7 part read package metadata only (the cc 10.7 target check is the one DSL import)."""
    code = (
        "import cudnn\nimport cudnn.frost\ncudnn.sdpa_capabilities()\ncudnn.frost.cutedsl_runtime_available()\ncudnn.frost.cutedsl_runtime_available((10, 0))\n"
    )
    added = _modules_after(code) - _modules_after("")
    present = sorted(m for m in _HEAVY if m in added)
    assert not present, f"the public probes imported {present}; they must not"


def test_the_docs_name_every_public_spelling():
    docs = pathlib.Path(__file__).resolve().parents[4] / "docs" / "utilities" / "python_graph_and_execution_backends.md"
    if not docs.is_file():
        pytest.skip("the docs tree is not part of this checkout")
    text = docs.read_text(encoding="utf-8")
    assert "### Consumed by FlashInfer" in text
    for name in (
        "cudnn.sdpa_capabilities()",
        "cudnn.frost.cutedsl_runtime_available(device_cc=None)",
        "cudnn.datatypes.torch_to_cudnn_data_type",
        "graph.selected_engine",
        "graph.backend_plan_entries()",
        "is_cuda_graph_replay_expected",
        "4.8.0 on cc 10.7",
    ):
        assert name in text, name
