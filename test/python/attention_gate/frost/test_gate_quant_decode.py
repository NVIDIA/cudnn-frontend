# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The attention gate + quantize decode step (``cudnn.attention_gate.frost.gate_quant_decode``): on the host the config's limits and pick
table, every typed decline of the entry (CPU tensors), the DSL gates (version, target, compiler), the plan-time-only compile key, the lazy
export, and the kernel's RENDERING for the claimed targets (the instructions that define the rounding chain); on a cc 10.x device the bytes
against the fp64 reference (``gate_quant_utils.assert_reference``: the bracket, the true-nearest margin rule, the torch model of the
convention), the exact cells, the edge shapes up to the widest served slab, output ownership, the bit-neutral knob arms, CUDA-graph replay,
the warm path's allocation / synchronization discipline, and the appended reciprocal primitive.

No cell arms a timer of its own: the kernel holds no wait and no loop, and every device cell synchronizes under the suite's per-test
watchdog (``CUDNN_TEST_TIMEOUT`` in ``test/python/conftest.py``), which a second in-process timer would replace and then disarm.
The operand builders, the reference and the device gate live in ``gate_quant_utils.py`` beside this module."""

import glob
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from gate_quant_utils import (  # noqa: E402  (the operand builders, the reference and the device gate)
    SCALES,
    _device_or_skip,
    assert_reference,
    device_model,
    make_operands,
    product_form_model,
    quantize,
    scale_tensor,
)

from cudnn.attention_gate.frost import gate_quant_decode as entry  # noqa: E402
from cudnn.attention_gate.frost.kernel import gate_quant_decode_config as kcfg  # noqa: E402

pytestmark = [pytest.mark.L0]

REPO = Path(__file__).resolve().parents[4]
U8 = torch.uint8
E4M3 = torch.float8_e4m3fn
ROWS = (1, 2, 3, 8, 16, 17, 24, 32, 64, 160)


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the config
# ---------------------------------------------------------------------------------------------------------------------------
def test_config_constants_and_limits():
    """The limits with their derivation: eight elements per thread; the element index is a signed 32-bit value, and the largest index ANY
    thread computes for the largest admitted slab stays below 2^31 under every served thread count -- with room: the arithmetic ceiling
    ``ELEMENT_LIMIT`` is far above the widest slab the suite runs on a device (``MAX_ROWS x MAX_WIDTH``)."""
    assert (kcfg.VEC, kcfg.MAX_ROWS, kcfg.MAX_WIDTH, kcfg.SERVED_THREADS, kcfg.DEFAULT_THREADS) == (8, 160, 32768, (128, 256), 256)
    assert kcfg.ELEMENT_LIMIT == 2**31 - 8 * 256 and kcfg.MAX_ROWS * kcfg.MAX_WIDTH <= kcfg.ELEMENT_LIMIT
    assert kcfg.MAX_WIDTH % kcfg.VEC == 0
    for threads in kcfg.SERVED_THREADS:
        for elements in (8, 2048, 160 * 2048, kcfg.MAX_ROWS * kcfg.MAX_WIDTH, kcfg.ELEMENT_LIMIT // 8 * 8):
            blocks = kcfg.blocks_for(elements, threads)
            assert (blocks - 1) * threads * kcfg.VEC < elements <= blocks * threads * kcfg.VEC, "the grid covers the slab with no empty CTA"
            assert kcfg.last_index(elements, threads) < 2**31, (threads, elements)
    assert kcfg.blocks_for(2048, 256) == 1 and kcfg.blocks_for(160 * 6144, 256) == 480 and kcfg.blocks_for(160 * 32768, 256) == 2560
    assert kcfg.blocks_for(1032, 256) == 1 and kcfg.blocks_for(3 * 1032, 256) == 2 and kcfg.blocks_for(8, 128) == 1


def test_pick_table_is_pinned():
    """The pick is a function of plan-time facts (rows x width, the part's SM count and compute capability) and performance-only; today ONE
    row -- 256 threads, the release off -- on every part.  The table literal is pinned, so a new row is a visible diff next to its table."""
    assert kcfg.THREADS_PICKS == ((0, 999, kcfg.ELEMENT_LIMIT, 256),)
    for rows in (1, 8, 32, 160):
        for width in (2048, 6144):
            for sm_count in (None, 148, 152, 212):
                for cc in (None, (10, 0), (10, 3), (10, 7)):
                    assert kcfg.default_cfg(rows, width, sm_count, cc) == kcfg.Cfg(THREADS=256, PDL_RELEASE=False)
    assert kcfg.pick_threads(8) == 256 and kcfg.pick_threads(kcfg.ELEMENT_LIMIT, 152, (10, 3)) == 256
    with pytest.raises(NotImplementedError, match="160 rows"):
        kcfg.default_cfg(161, 2048)
    with pytest.raises(NotImplementedError, match="multiple of 8"):
        kcfg.default_cfg(1, 2052)


def test_cfg_knobs_validate_by_name():
    for threads in kcfg.SERVED_THREADS:
        assert kcfg.validate_cfg(kcfg.Cfg(THREADS=threads, PDL_RELEASE=True)).THREADS == threads
    for bad in (0, 100, 512, 256.0, True):
        with pytest.raises(ValueError, match="THREADS must be one of"):
            kcfg.validate_cfg(kcfg.Cfg(THREADS=bad))
    with pytest.raises(ValueError, match="PDL_RELEASE must be a bool"):
        kcfg.validate_cfg(kcfg.Cfg(PDL_RELEASE=1))
    for bad in (None, dict(THREADS=256), "Cfg"):
        with pytest.raises(TypeError, match="cfg must be a gate_quant_decode_config.Cfg"):
            kcfg.validate_cfg(bad)
    with pytest.raises(TypeError, match="cfg must be a gate_quant_decode_config.Cfg"):
        entry.AttentionGateQuantDecode("cpu", 2048, cfg=dict(THREADS=256))
    with pytest.raises(ValueError, match="THREADS must be one of"):
        entry.AttentionGateQuantDecode("cpu", 2048, cfg=kcfg.Cfg(THREADS=512))


@pytest.mark.parametrize("width", (4, 0, -8, 2052, kcfg.MAX_WIDTH + 8))
def test_declines_an_unserved_width_by_name(width):
    with pytest.raises(NotImplementedError, match=f"multiple of 8 elements up to {kcfg.MAX_WIDTH}.*got {width}"):
        entry.AttentionGateQuantDecode("cpu", width)


@pytest.mark.parametrize("width", (2048.0, "2048", None, True))
def test_declines_a_width_that_is_not_an_int(width):
    with pytest.raises(TypeError, match="width must be an int"):
        entry.AttentionGateQuantDecode("cpu", width)


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the entry's declines (CPU tensors)
# ---------------------------------------------------------------------------------------------------------------------------
def _e4m3(rows, width):
    return torch.zeros(rows, width, dtype=U8).view(E4M3)


def _host_operands(rows=3, width=2048, with_out=True):
    ops = dict(attn=torch.zeros(rows, width, dtype=torch.bfloat16), gate=torch.zeros(rows, width, dtype=torch.bfloat16), scale=torch.tensor([0.01]))
    if with_out:
        ops["out"] = _e4m3(rows, width)
    return ops


def _check(ops, **kw):
    return entry.check_support(ops["attn"], ops["gate"], ops["scale"], ops.get("out"), **{"device_cc": (10, 3), **kw})


def test_served_operands_pass_the_host_checks():
    for width in (8, 768, 1024, 1032, 1536, 2048, 3072, 6144, kcfg.MAX_WIDTH):
        for rows in (1, 17, 160):
            assert _check(_host_operands(rows, width), width=width) == rows
            assert _check(_host_operands(rows, width, with_out=False)) == rows, "the width defaults to attn's own"
    ops = _host_operands()
    ops["scale"] = torch.tensor(0.01)  # a 0-d scale is one element too
    assert _check(ops) == 3
    ops["out"] = _e4m3(7, 2048)  # a taller caller-owned output: rows past the call are never written
    assert _check(ops) == 3
    for cc in ((10, 0), (10, 3), (10, 7)):
        assert _check(_host_operands(), device_cc=cc) == 3


def _misaligned(dtype, shape):
    """A contiguous tensor of ``shape`` whose first byte sits one ELEMENT (one byte for uint8-backed E4M3, two for bf16) past a 16-byte
    aligned address."""
    n = 1
    for d in shape:
        n *= d
    if dtype == E4M3:
        return torch.zeros(n + 16, dtype=U8)[1 : 1 + n].view(E4M3).view(*shape)
    return torch.zeros(n + 16, dtype=dtype)[1 : 1 + n].view(*shape)


def _misaligned_scale():
    """One fp32 element at an address that is not a multiple of 4 (a byte buffer read at an odd offset)."""
    buf = bytearray(64)
    base = torch.frombuffer(buf, dtype=U8).data_ptr()
    return torch.frombuffer(buf, dtype=torch.float32, count=1, offset=1 + (-base) % 4)  # address % 4 == 1


def _overlap(which):
    """Operands whose ``out`` shares bytes with ``which``."""
    ops = _host_operands()
    n = 3 * 2048
    if which == "scale":
        store = torch.zeros(n + 16, dtype=U8)
        ops["out"] = store[:n].view(E4M3).view(3, 2048)
        ops["scale"] = store[n - 4 : n].view(torch.float32)
    else:
        store = torch.zeros(2 * n, dtype=U8)
        ops[which] = store.view(torch.bfloat16).view(3, 2048)
        ops["out"] = store[n:].view(E4M3).view(3, 2048)  # the second half of the bf16 slab's bytes
    return ops


def _mutations():
    wide = lambda dtype: torch.zeros(3, 4096, dtype=dtype)[:, ::2]  # noqa: E731  (a non-contiguous [3, 2048])
    return {
        "attn-not-a-tensor": (dict(attn=[[0.0] * 2048] * 3), TypeError, "attn must be a torch.Tensor"),
        "scale-not-a-tensor": (dict(scale=0.01), TypeError, "scale must be a torch.Tensor"),
        "attn-fp32": (dict(attn=torch.zeros(3, 2048)), TypeError, "attn must be torch.bfloat16"),
        "gate-fp16": (dict(gate=torch.zeros(3, 2048, dtype=torch.float16)), TypeError, "gate must be torch.bfloat16"),
        "scale-fp64": (dict(scale=torch.tensor([0.01], dtype=torch.float64)), TypeError, "scale must be torch.float32"),
        "out-uint8": (dict(out=torch.zeros(3, 2048, dtype=U8)), TypeError, "out must be torch.float8_e4m3fn"),
        "attn-1d": (dict(attn=torch.zeros(3 * 2048, dtype=torch.bfloat16)), ValueError, "attn must be .rows, width. with at least one row"),
        "attn-other-width": (dict(attn=torch.zeros(3, 2040, dtype=torch.bfloat16)), ValueError, "attn must be .rows, 2048. with at least one row"),
        "attn-no-rows": (
            dict(attn=torch.zeros(0, 2048, dtype=torch.bfloat16), gate=torch.zeros(0, 2048, dtype=torch.bfloat16)),
            ValueError,
            "at least one row",
        ),
        "gate-other-shape": (dict(gate=torch.zeros(2, 2048, dtype=torch.bfloat16)), ValueError, "gate must have attn's shape"),
        "too-many-rows": (
            dict(attn=torch.zeros(161, 2048, dtype=torch.bfloat16), gate=torch.zeros(161, 2048, dtype=torch.bfloat16), out=None),
            NotImplementedError,
            "at most 160 rows.*got 161",
        ),
        "attn-strided": (dict(attn=wide(torch.bfloat16)), ValueError, "attn must be contiguous"),
        "gate-strided": (dict(gate=wide(torch.bfloat16)), ValueError, "gate must be contiguous"),
        "out-strided": (dict(out=torch.zeros(3, 4096, dtype=U8).view(E4M3)[:, ::2]), ValueError, "out must be contiguous"),
        "scale-two-elements": (dict(scale=torch.tensor([0.01, 0.02])), ValueError, "scale must be one fp32 element"),
        "out-other-width": (dict(out=_e4m3(3, 2040)), ValueError, "out must be .>= 3, 2048."),
        "out-too-few-rows": (dict(out=_e4m3(2, 2048)), ValueError, "out must be .>= 3, 2048."),
        "attn-misaligned": (
            dict(attn=_misaligned(torch.bfloat16, (3, 2048))),
            ValueError,
            "attn must start at a 16-byte aligned address for its 128-bit loads",
        ),
        "gate-misaligned": (dict(gate=_misaligned(torch.bfloat16, (3, 2048))), ValueError, "gate must start at a 16-byte aligned address"),
        "out-misaligned": (dict(out=_misaligned(E4M3, (3, 2048))), ValueError, "out must start at a 16-byte aligned address"),
        "scale-misaligned": (dict(scale=_misaligned_scale()), ValueError, "scale must start at a 4-byte aligned address"),
        "out-overlaps-attn": (_overlap("attn"), ValueError, "out must not overlap attn"),
        "out-overlaps-gate": (_overlap("gate"), ValueError, "out must not overlap gate"),
        "out-overlaps-scale": (_overlap("scale"), ValueError, "out must not overlap scale"),
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_typed_declines_name_the_operand(name):
    """Every binding decline of the entry, by exception type and by the operand it names; each mutation changes ONE served operand set."""
    mutation, exc, token = _mutations()[name]
    ops = {**_host_operands(), **mutation}
    with pytest.raises(exc, match=token):
        _check(ops, width=2048)
    with pytest.raises(exc, match=token):  # the same decline from the binding checks a warm launch re-runs
        entry.check_binding(ops["attn"], ops["gate"], ops["scale"], ops.get("out"), width=2048)


def test_declines_an_unserved_compute_capability_by_name():
    for cc in ((8, 0), (9, 0), (12, 0), (12, 1), (11, 0)):
        with pytest.raises(NotImplementedError, match=rf"serves compute capability 10\.x.*this device is {cc[0]}\.{cc[1]}"):
            _check(_host_operands(), device_cc=cc)
    assert [entry.cc_supported(cc) for cc in ((10, 0), (10, 3), (10, 7), (9, 0), (12, 0), None)] == [True, True, True, False, False, False]


def test_declines_cpu_operands_when_no_capability_is_given():
    ops = _host_operands()
    with pytest.raises(NotImplementedError, match="serves CUDA tensors only"):
        entry.check_support(ops["attn"], ops["gate"], ops["scale"], ops["out"])
    k = entry.AttentionGateQuantDecode("cpu", 2048)  # a host entry runs every binding decline; its compile is the device's typed decline
    with pytest.raises(NotImplementedError, match="serves CUDA tensors only"):
        k.compile(ops["attn"], ops["gate"], ops["scale"])
    assert not k.compiled


def test_binding_check_requires_every_operand_on_the_given_device():
    ops = _host_operands()
    with pytest.raises(ValueError, match=r"every operand must live on one device, the entry's \(cuda:1\); attn is on cpu"):
        entry.check_binding(ops["attn"], ops["gate"], ops["scale"], ops["out"], device="cuda:1")
    assert entry.check_binding(ops["attn"], ops["gate"], ops["scale"], ops["out"], device="cpu") == 3


def test_launch_knob_applies_on_top_of_the_pick_and_of_a_given_cfg():
    """``pdl_release=None`` keeps the config's own field; a bool overrides it -- on the pick and on a caller's ``Cfg`` alike; the default
    is OFF.  Every served row count maps to a config resolved at construction."""
    k = entry.AttentionGateQuantDecode("cpu", 2048)
    assert k.cfgs == (kcfg.Cfg(THREADS=256, PDL_RELEASE=False),) and k.pdl_release is None and not k.cfg_given
    assert entry.AttentionGateQuantDecode("cpu", 2048, pdl_release=True).cfgs == (kcfg.Cfg(THREADS=256, PDL_RELEASE=True),)
    given = entry.AttentionGateQuantDecode("cpu", 2048, cfg=kcfg.Cfg(THREADS=128, PDL_RELEASE=True))
    assert given.cfgs == (kcfg.Cfg(THREADS=128, PDL_RELEASE=True),) and given.cfg_given
    assert entry.AttentionGateQuantDecode("cpu", 2048, kcfg.Cfg(THREADS=128, PDL_RELEASE=True), pdl_release=False).cfgs == (kcfg.Cfg(THREADS=128),)
    assert len(k.cfg_by_rows) == kcfg.MAX_ROWS and all(k.cfg_for(rows) == k.cfgs[0] for rows in range(1, kcfg.MAX_ROWS + 1))
    with pytest.raises(NotImplementedError, match="160 rows"):
        k.cfg_for(161)
    with pytest.raises(TypeError):
        entry.AttentionGateQuantDecode("cpu", 2048, None, True)  # the launch knobs are keyword-only


def test_compile_key_is_plan_time_only():
    """The key is the config alone: one key for every row count and (across entries) for every width; it moves with ``THREADS`` and
    ``PDL_RELEASE`` only.  The element count is a runtime argument of the kernel."""
    k = entry.AttentionGateQuantDecode("cpu", 2048)
    assert k.key(1) == k.key(17) == k.key(160) == (kcfg.Cfg(),)
    assert entry.AttentionGateQuantDecode("cpu", 6144).key(160) == k.key(1)
    assert entry.AttentionGateQuantDecode("cpu", 2048, pdl_release=True).key(1) != k.key(1)
    assert entry.AttentionGateQuantDecode("cpu", 2048, cfg=kcfg.Cfg(THREADS=128)).key(1) != k.key(1)


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the DSL gates (installed -> version floor -> the part's target -> the compiler that assembles it)
# ---------------------------------------------------------------------------------------------------------------------------
def test_declines_a_dsl_below_the_floor_by_version(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
    for msg in (entry._dsl_version_error(), entry._dsl_error((10, 3)), entry._dsl_error((10, 7))):
        assert "requires nvidia-cutlass-dsl >= 4.8.0" in msg and "found 4.7.1" in msg
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    assert entry._dsl_version_error() is None


def test_declines_a_missing_dsl(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (False, None))
    for msg in (entry._dsl_version_error(), entry._dsl_error((10, 3))):
        assert "requires nvidia-cutlass-dsl >= 4.8.0" in msg and "not installed" in msg


def test_declines_a_dsl_without_the_parts_target_by_name(monkeypatch):
    """A version floor does not guarantee the part's target: with the version gate passed and a DSL whose target list lacks ``sm_107a``
    (a controlled missing-target probe), cc 10.7 is declined naming the target and the DSL found, cc 10.3 is admitted.  The version gate
    runs first: a too-old DSL is reported as too old, and the target probe (the first gate that imports the DSL) is never reached."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    msg = entry._dsl_error((10, 7))
    assert msg is not None and "sm_107a" in msg and "nvidia-cutlass-dsl 4.8.0" in msg and msg.startswith(entry.NAME)
    assert entry._dsl_target_error((10, 3)) is None and entry._dsl_target_error((10, 0)) is None
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.0")))

    def unreachable():
        raise AssertionError("the target probe ran although the version gate declined")

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", unreachable)
    assert "found 4.7.0" in entry._dsl_error((10, 7))


def test_declines_a_compiler_that_rejects_the_parts_target(monkeypatch):
    """cc 10.7 under a DSL whose compiler is the CUDA 12 toolchain flavour: declined BEFORE JIT, naming the compiler found and the one
    required (the flavour rejects the ``sm_107a`` target at compile time although the DSL lists it).  cc 10.0 / 10.3 are served by both
    flavours (the rendering cell pins that their PTX is one text)."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    monkeypatch.setattr(entry, "dsl_compiler_cuda_version", lambda: (12, 9))
    msg = entry._dsl_error((10, 7))
    assert msg is not None and "compute capability 10.7" in msg and "CUDA 13.4 or later" in msg and "compiles with CUDA 12.9" in msg and "cu13" in msg
    assert entry._dsl_error((10, 3)) is None and entry._dsl_error((10, 0)) is None
    for served in ((13, 4), (13, 6), (14, 0)):
        monkeypatch.setattr(entry, "dsl_compiler_cuda_version", lambda v=served: v)
        assert entry._dsl_error((10, 7)) is None
    monkeypatch.setattr(entry, "dsl_compiler_cuda_version", lambda: (13, 3))
    assert "compiles with CUDA 13.3" in entry._dsl_error((10, 7))


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: exports and the import boundary
# ---------------------------------------------------------------------------------------------------------------------------
def test_cudnn_level_export_resolves_to_the_entry():
    import importlib

    assert cudnn.AttentionGateQuantDecode is entry.AttentionGateQuantDecode
    family, frost = importlib.import_module("cudnn.attention_gate"), importlib.import_module("cudnn.attention_gate.frost")
    assert family.AttentionGateQuantDecode is entry.AttentionGateQuantDecode and frost.AttentionGateQuantDecode is entry.AttentionGateQuantDecode
    assert frost.Cfg is kcfg.Cfg
    assert "AttentionGateQuantDecode" in cudnn._LAZY_OPTIONAL_IMPORTS and "AttentionGateQuantDecode" in cudnn._OPTIONAL_DEPENDENCY_INSTALL_HINTS
    with pytest.raises(AttributeError):
        family.NoSuchName


def test_import_cudnn_does_not_import_the_kernel_or_torch():
    """The lazy boundary, in fresh interpreters: importing cudnn loads neither the family's modules nor the DSL nor torch; importing the
    host entry and constructing a host entry (its declines included) still loads neither the DSL nor the kernel module."""
    code = "import sys, cudnn; print(sorted(m for m in sys.modules if m.startswith('cudnn.attention_gate') or m in ('cutlass', 'torch')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out
    code = (
        "import sys, cudnn\n"
        "k = cudnn.AttentionGateQuantDecode('cpu', 2048)\n"
        "print(sorted(m for m in sys.modules if m == 'cutlass' or m.endswith('gate_quant_decode_f16')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the rendering (the instructions that define the rounding chain), per claimed target
# ---------------------------------------------------------------------------------------------------------------------------
_ARMS = tuple((threads, release) for threads in kcfg.SERVED_THREADS for release in (False, True))

_RENDER_PROBE = textwrap.dedent("""
    import glob, os, sys
    dump, arch = sys.argv[1], sys.argv[2]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump  # read once, at the first cutlass import
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_ARCH"] = arch  # unconditional: an inherited value would render another target
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    import cutlass
    import cutlass.cute as cute
    from cutlass.base_dsl.version_info import CUDA_VERSION
    from cutlass.cute.runtime import make_fake_stream
    from cudnn.attention_gate.frost.kernel import gate_quant_decode_config as kcfg
    from cudnn.attention_gate.frost.kernel import gate_quant_decode_f16 as mod

    print("COMPILER", int(CUDA_VERSION.major), int(CUDA_VERSION.minor))
    half = lambda: cute.runtime.make_fake_tensor(dtype=cutlass.BFloat16, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
    scale = cute.runtime.make_fake_tensor(dtype=cutlass.Float32, shape=(1,), stride=(1,), assumed_align=4)
    out = cute.runtime.make_fake_tensor(dtype=cutlass.Uint8, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
    for spec in sys.argv[3:]:
        threads, release = spec.split(":")
        before = set(os.listdir(dump))
        cfg = kcfg.Cfg(THREADS=int(threads), PDL_RELEASE=release == "1")
        cute.compile(mod.launch, cfg, half(), half(), scale, out, cutlass.Int32(2048), make_fake_stream())
        for name in sorted(set(os.listdir(dump)) - before):
            print("KEPT", spec, name)
    print("COMPILED", arch)
    """)


def _nvdisasm_candidates():
    cands = []
    if os.environ.get("CUDA_PATH"):
        cands.append(os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm"))
    on_path = shutil.which("nvdisasm")
    if on_path:
        cands.append(on_path)
    return [c for c in dict.fromkeys(cands) if os.path.isfile(c) and os.access(c, os.X_OK)]


def _sass(cubin):
    """The disassembly of a cubin by the first nvdisasm that decodes it, or None (a toolkit older than the target decodes nothing)."""
    for nvd in _nvdisasm_candidates():
        try:
            proc = subprocess.run([nvd, "-c", cubin], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.splitlines()
    return None


# RECORDED per compiler major: how many of the eight ``1 + exp(-g)`` additions the compiler contracts into the exponential's last fused
# multiply-add (``fma.rn.f32 ..., 0f3F800000``).  The contraction changes no byte (the scale factor is a power of two, so the product is
# exact and the fused form rounds as the separate add does); the count is a compiler choice, recorded so a change is visible.  CuTe DSL 4.8.0:
# the CUDA 12.9 and the CUDA 13.4 flavours render ONE text (all eight contracted).  A new major is ADDED with its measurement.
_RECORDED_CONTRACTED_ADDS = {12: 8, 13: 8}


@pytest.mark.parametrize("arch, cc", (("sm_100a", (10, 0)), ("sm_103a", (10, 3)), ("sm_107a", (10, 7))))
def test_rendering_pins_the_convention(arch, cc, tmp_path):
    """Trace-compile the kernel for a claimed target on this box (fake tensors: no device needed) and pin, per arm, the instructions that ARE
    the contract: two 128-bit global loads and one 32-bit load; nine ``div.rn.f32`` (eight gate divisions and the reciprocal, whose
    numerator is the immediate 1.0); eight ``ex2.approx.ftz.f32`` (the precise exponential's core); eight ``cvt.rn.bf16.f32`` (the first
    rounding), eight ``mul.f32`` by the reciprocal and eight ``cvt.rn.satfinite.e4m3x2.f32`` (the second rounding), each with its one-byte
    global store; none of ``tanh`` / ``div.approx`` / ``div.full`` / ``rcp.approx`` (an approximate sigmoid or division), no shared-memory
    access, no barrier, no ``griddepcontrol.wait``; the release present exactly under ``PDL_RELEASE``; ``.reqntid`` = ``THREADS``; the
    cuDNN symbol prefix.  With an nvdisasm that decodes the cubin, also the SASS: nine ``FCHK`` (every ``div.rn.f32`` lowers to the
    checked reciprocal + Newton form), eight ``MUFU.EX2``, the 128-bit loads, no shared load / store, no spill.  A compile failure FAILS; a
    DSL that does not serve the target (version, target list, compiler) SKIPS with the gate's text."""
    msg = entry._dsl_error(cc)
    if msg is not None:
        pytest.skip(msg)
    probe = tmp_path / "render_probe.py"  # a FILE, not ``python -c``: the DSL preprocessor re-reads a kernel's source through inspect
    probe.write_text(_RENDER_PROBE)
    dump = tmp_path / "dump"
    dump.mkdir()
    specs = [f"{threads}:{int(release)}" for threads, release in _ARMS]
    proc = subprocess.run([sys.executable, str(probe), str(dump), arch, *specs], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"trace-compile for {arch} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert f"COMPILED {arch}" in lines, proc.stdout[-2000:]
    major = int(next(ln for ln in lines if ln.startswith("COMPILER")).split()[1])
    kept = {}
    for ln in lines:
        if ln.startswith("KEPT "):
            _, spec, name = ln.split(" ", 2)
            kept.setdefault(spec, []).append(str(dump / name))
    bodies = {}
    for (threads, release), spec in zip(_ARMS, specs):
        files = kept.get(spec, [])
        ptx_files = [f for f in files if f.endswith(".ptx")]
        assert len(ptx_files) == 1, f"{arch} {spec}: expected one PTX, kept {files}"
        ptx = Path(ptx_files[0]).read_text()
        count = lambda needle: sum(1 for ln in ptx.splitlines() if needle in ln)  # noqa: E731
        what = f"{arch} THREADS {threads} PDL_RELEASE {release} (compiler CUDA {major})"
        assert f".target {arch}" in ptx, what
        assert count("ld.global.v4.") == 2 and count("ld.global.") == 3, what
        assert count("div.rn.f32") == 9, what
        assert sum(1 for ln in ptx.splitlines() if "div.rn.f32" in ln and "0f3F800000," in ln) == 1, f"{what}: the reciprocal's immediate numerator"
        assert count("ex2.approx.ftz.f32") == 8, what
        assert count("cvt.rn.bf16.f32") == 8 and count("cvt.f32.bf16") == 8, what
        assert count("cvt.rn.satfinite.e4m3x2.f32") == 8 and sum(1 for ln in ptx.splitlines() if "mul.f32 v," in ln and "e4m3x2" in ln) == 8, what
        assert count("st.global.b8") == 8 and count("st.global.") == 8, what
        for banned in (
            "tanh",
            "div.approx",
            "div.full",
            "rcp.approx",
            "ld.shared",
            "st.shared",
            "mbarrier",
            "bar.sync",
            "barrier.",
            "griddepcontrol.wait",
            "cp.async",
            "tcgen05",
        ):
            assert count(banned) == 0, f"{what}: {banned} in the rendering"
        assert count("griddepcontrol.launch_dependents") == int(release), what
        assert f".reqntid {threads}, 1, 1" in ptx, what
        assert ".entry cudnn_kernel_frost_attention_gate_quant" in ptx, what
        contracted = sum(1 for ln in ptx.splitlines() if "fma.rn.f32" in ln and ln.rstrip().endswith("0f3F800000;"))
        explicit = sum(1 for ln in ptx.splitlines() if "add.f32" in ln and "0f3F800000" in ln)
        assert contracted + explicit == 8, f"{what}: the eight (1 + exp) additions: {contracted} contracted, {explicit} explicit"
        if major in _RECORDED_CONTRACTED_ADDS:
            assert (
                contracted == _RECORDED_CONTRACTED_ADDS[major]
            ), f"{what}: {contracted} contracted additions, the record says {_RECORDED_CONTRACTED_ADDS[major]}"
        bodies[(threads, release)] = ptx[ptx.index(".reqntid") :]
        cubins = [f for f in files if f.endswith(".cubin")]
        sass = _sass(cubins[0]) if cubins else None
        if sass is not None:
            n = lambda needle: sum(1 for ln in sass if needle in ln)  # noqa: E731
            assert n("FCHK") == 9 and n("MUFU.RCP") >= 9, f"{what}: the div.rn lowering in the SASS (FCHK {n('FCHK')}, MUFU.RCP {n('MUFU.RCP')})"
            assert n("LDS") == 0 and n("STS") == 0 and n(" STL") == 0 and n(" LDL") == 0, f"{what}: shared access or a spill in the SASS"
            assert n("STG.E.U8") == 8 and n("LDG.E.128") == 2 and n("MUFU.EX2") == 8, f"{what}: the load / store / exponential forms in the SASS"
            assert n("PREEXIT") == int(release), what
    # the release adds its one instruction and nothing else: the arithmetic text of the two release arms is one text per thread count
    for threads in kcfg.SERVED_THREADS:
        with_release = "\n".join(ln for ln in bodies[(threads, True)].splitlines() if "griddepcontrol" not in ln and "inline asm" not in ln)
        without = "\n".join(ln for ln in bodies[(threads, False)].splitlines() if "inline asm" not in ln)
        assert _strip_symbols(with_release) == _strip_symbols(without), f"{arch} THREADS {threads}: the release arm changes more than its one instruction"


def _strip_symbols(text):
    import re

    return re.sub(r"cudnn_kernel_frost_attention_gate_quant_\w+", "ENTRY", text)


# ---------------------------------------------------------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------------------------------------------------------
def _run(k, attn, gate, scale, out=None):
    y = k(attn, gate, scale, out)
    torch.cuda.synchronize()
    return y


def _bytes(t):
    return t.view(U8)


@pytest.mark.parametrize("kind", ("random", "adversarial"))
@pytest.mark.parametrize("width", (2048, 6144))
@pytest.mark.parametrize("rows", ROWS)
def test_matches_the_reference(rows, width, kind):
    """The bytes against the fp64 reference of the bf16 operands (the bracket, the true-nearest margin rule, the torch model of the
    convention) at every scale of the reference set; random draws and the adversarial pairs (saturation, signed zeros, tiny and huge
    magnitudes, gates on both sides of the exponential's overflow)."""
    dev = _device_or_skip()
    k = entry.AttentionGateQuantDecode(dev, width)
    attn, gate = make_operands(rows, width, dev, seed=1000 * width + rows, kind=kind)
    for value in SCALES:
        scale = scale_tensor(value, dev)
        out = _run(k, attn, gate, scale)
        assert out.shape == (rows, width) and out.dtype == E4M3
        assert_reference(out, attn, gate, scale, f"rows {rows} width {width} {kind} scale {value}")


def test_matches_the_reference_on_every_finite_gate_value():
    """One ``[128, 32768]`` call per scale holding EVERY finite bf16 gate value against 64 attention values (drawn once): the reference on
    the whole gate axis -- the exponential's range reduction, its underflow and overflow -- not only where a draw lands."""
    dev = _device_or_skip()
    rows, width = 128, 32768
    k = entry.AttentionGateQuantDecode(dev, width)
    gbits = torch.arange(65536, dtype=torch.int32, device=dev)
    gate = gbits.to(torch.int16).view(torch.bfloat16)
    gate = torch.where(torch.isfinite(gate.float()), gate, torch.zeros_like(gate)).repeat(64).reshape(rows, width).contiguous()
    gen = torch.Generator(device="cpu").manual_seed(5)
    attn = (torch.randn(64, generator=gen) * 4).to(dev).bfloat16().repeat_interleave(65536).reshape(rows, width).contiguous()
    for value in (0.01, 1.0):
        scale = scale_tensor(value, dev)
        assert_reference(_run(k, attn, gate, scale), attn, gate, scale, f"gate sweep scale {value}")


def test_exact_cells_are_bitwise_with_torch():
    """Operands whose gated value is EXACT, so the bytes are torch's with no bracket: gate 0 (the denominator is 2: half the attention
    value), gate >= 40 (the denominator is 1 in fp32: the static quantization of the attention value itself), gate <= -89 (the exponential
    overflows fp32: a signed zero), attention +-0 under any gate, and saturation at +-448 (codes 0x7E / 0xFE)."""
    dev = _device_or_skip()
    rows, width = 4, 2048
    k = entry.AttentionGateQuantDecode(dev, width)
    attn, _ = make_operands(rows, width, dev, seed=11)
    full = lambda value: torch.full((rows, width), value, dtype=torch.bfloat16, device=dev)  # noqa: E731
    for value in (0.01, 1.0):
        scale = scale_tensor(value, dev)
        assert torch.equal(_bytes(_run(k, attn, full(0.0), scale)), quantize((attn.float() * 0.5).bfloat16(), scale)), f"gate 0, scale {value}"
        for big in (40.0, 88.5, 700.0):
            assert torch.equal(_bytes(_run(k, attn, full(big), scale)), quantize(attn, scale)), f"gate {big}, scale {value}"
        for small in (-89.0, -104.0, -700.0):
            want = torch.where(attn.float() < 0, torch.full_like(attn, -0.0), torch.zeros_like(attn))
            got = _bytes(_run(k, attn, full(small), scale))
            assert torch.equal(got, quantize(want, scale)) and set(got.unique().tolist()) <= {0x00, 0x80}, f"gate {small}, scale {value}"
    scale = scale_tensor(0.01, dev)
    zeros = torch.zeros(rows, width, dtype=torch.bfloat16, device=dev)
    zeros[1::2] = -0.0
    _, gate = make_operands(rows, width, dev, seed=12)
    got = _bytes(_run(k, zeros, gate, scale))
    assert bool((got[0::2] == 0x00).all()) and bool((got[1::2] == 0x80).all()), "attention +-0 keeps its sign under any gate"
    sat = full(1e4)
    sat[1::2] = -1e4
    got = _bytes(_run(k, sat, full(40.0), scale))
    assert bool((got[0::2] == 0x7E).all()) and bool((got[1::2] == 0xFE).all()), "saturation at +-448"


def test_division_form_is_the_contract_on_recorded_pairs():
    """Two recorded (attn, gate) pairs at scale 1.0 on which the PRODUCT form ``attn * sigmoid(gate)`` rounds to another code than the
    step's division form: the step writes the division form's byte at every position of a thread's eight elements, the product model does
    not -- so a reference (or a kernel) must never be 'simplified' to the product."""
    dev = _device_or_skip()
    k = entry.AttentionGateQuantDecode(dev, 8)
    scale = scale_tensor(1.0, dev)
    for attn_bits, gate_bits, byte, product_byte in ((0x3D47, 0x3C77, 13, 12), (0x3CEC, 0x3CF3, 8, 7)):
        attn = torch.full((1, 8), attn_bits, dtype=torch.int16, device=dev).view(torch.bfloat16)
        gate = torch.full((1, 8), gate_bits, dtype=torch.int16, device=dev).view(torch.bfloat16)
        got = _bytes(_run(k, attn, gate, scale))
        assert got.tolist() == [[byte] * 8], (hex(attn_bits), hex(gate_bits), got.tolist())
        assert device_model(attn, gate, scale).tolist() == [[byte] * 8]
        assert product_form_model(attn, gate, scale).tolist() == [[product_byte] * 8], "the product form is another function on this pair"
        assert_reference(got, attn, gate, scale, "recorded pair")


@pytest.mark.parametrize("rows, width", ((1, 8), (3, 1032), (160, 1032), (160, 6144), (160, kcfg.MAX_WIDTH), (1, kcfg.MAX_WIDTH)))
def test_edge_shapes(rows, width):
    """One active thread (``1 x 8``), a partial last CTA with rows sharing CTAs and warps (width 1032), the tallest served slabs, and the
    widest: ``160 x 32768`` = 2,560 CTAs (several waves on any SM100-class part), ``1 x 32768``."""
    dev = _device_or_skip()
    k = entry.AttentionGateQuantDecode(dev, width)
    attn, gate = make_operands(rows, width, dev, seed=31 * width + rows, kind="adversarial")
    for value in (0.01, 1.0):
        scale = scale_tensor(value, dev)
        assert_reference(_run(k, attn, gate, scale), attn, gate, scale, f"rows {rows} width {width} scale {value}")


def test_one_sixty_one_rows_is_the_typed_decline_never_a_launch():
    dev = _device_or_skip()
    k = entry.AttentionGateQuantDecode(dev, 2048)
    attn, gate = make_operands(161, 2048, dev, seed=3)
    scale = scale_tensor(0.01, dev)
    _run(k, attn[:160].contiguous(), gate[:160].contiguous(), scale)
    before = _bytes(k.out_buffer).clone()
    with pytest.raises(NotImplementedError, match="at most 160 rows.*got 161"):
        k(attn, gate, scale)
    with pytest.raises(NotImplementedError, match="at most 160 rows.*got 161"):
        k.compile(attn, gate, scale)
    torch.cuda.synchronize()
    assert torch.equal(_bytes(k.out_buffer), before), "a declined call launches nothing"


def test_caller_owned_out_and_rows_past_the_call_stay_untouched():
    """A taller caller-owned ``out`` keeps its sentinel past the call's rows and is returned AS GIVEN; its bytes are the entry-owned
    buffer's; the inputs are unmodified; and a call with a caller-owned ``out`` leaves the entry-owned buffer's bytes as they were (only a
    later call WITHOUT ``out`` rewrites the entry-owned view)."""
    dev = _device_or_skip()
    rows, width = 5, 2048
    k = entry.AttentionGateQuantDecode(dev, width)
    attn, gate = make_operands(rows, width, dev, seed=2, kind="adversarial")
    scale = scale_tensor(0.02, dev)
    keep = (attn.clone(), gate.clone(), scale.clone())
    owned = _run(k, attn, gate, scale)
    assert owned.shape == (rows, width) and owned.dtype == E4M3 and owned.data_ptr() == k.out_buffer.data_ptr()
    assert k.output(rows).data_ptr() == owned.data_ptr() and k.output(rows).shape == owned.shape
    owned_bytes = _bytes(owned).clone()
    mine = torch.full((9, width), 0xA5, dtype=U8, device=dev).view(E4M3)
    other_attn, other_gate = make_operands(rows, width, dev, seed=99)
    assert _run(k, other_attn, other_gate, scale, mine) is mine
    assert torch.equal(_bytes(k.out_buffer)[:rows], owned_bytes), "a caller-owned call does not touch the entry-owned buffer"
    assert _run(k, attn, gate, scale, mine) is mine
    assert torch.equal(_bytes(mine)[:rows], owned_bytes) and bool((_bytes(mine)[rows:] == 0xA5).all()), "rows past the call are never written"
    for now, was, bits in zip((attn, gate, scale), keep, (torch.int16, torch.int16, torch.int32)):
        assert torch.equal(now.view(bits), was.view(bits)), "the inputs are unmodified"


BITWISE_ARMS = [dict(THREADS=128), dict(PDL_RELEASE=True), dict(THREADS=128, PDL_RELEASE=True)]


@pytest.mark.parametrize("width", (2048, 6144, 1032))
@pytest.mark.parametrize("rows", (1, 17, 160))
@pytest.mark.parametrize("arm", BITWISE_ARMS, ids=lambda a: ",".join(f"{k}={v}" for k, v in a.items()))
def test_bit_neutral_knob_arms_write_the_default_bytes(arm, rows, width):
    """``THREADS`` (the grid) and ``PDL_RELEASE`` move no byte: every arm against the default config on random and adversarial operands
    at two scales, width 1032 included (a partial last CTA under both thread counts).  The release is exercised STANDALONE here (no grid
    on the stream carries the programmatic attribute, so the trigger is inert); a chain cell with a dependent belongs to the suite of the
    step that consumes this output."""
    dev = _device_or_skip()
    base = entry.AttentionGateQuantDecode(dev, width)
    knob = entry.AttentionGateQuantDecode(dev, width, cfg=replace(kcfg.Cfg(), **arm))
    via_kwarg = entry.AttentionGateQuantDecode(dev, width, pdl_release=True) if arm == dict(PDL_RELEASE=True) else None
    for kind in ("random", "adversarial"):
        attn, gate = make_operands(rows, width, dev, seed=7 * width + rows, kind=kind)
        for value in (0.01, 1.0):
            scale = scale_tensor(value, dev)
            want = _bytes(_run(base, attn, gate, scale)).clone()
            assert torch.equal(_bytes(_run(knob, attn, gate, scale)), want), f"{arm} rows {rows} width {width} {kind} scale {value}"
            if via_kwarg is not None:
                assert torch.equal(_bytes(_run(via_kwarg, attn, gate, scale)), want), "the pdl_release keyword is the config field"


def test_one_compile_serves_every_row_count_and_scale_form(monkeypatch):
    """Compiled once at one row, the entry launches every row count, a 0-d and a ``[1]`` scale, an entry-owned and a caller-owned output
    with compilation FORBIDDEN (Rule 4: the row count and the element count are runtime arguments, never a key)."""
    dev = _device_or_skip()
    width = 2048
    k = entry.AttentionGateQuantDecode(dev, width)
    attn, gate = make_operands(160, width, dev, seed=21)
    scale = scale_tensor(0.01, dev)
    k.compile(attn[:1].contiguous(), gate[:1].contiguous(), scale)
    assert len(k.compiled) == 1
    import cutlass.cute as cute

    def forbidden(*args, **kwargs):
        raise AssertionError("a warm call compiled")

    monkeypatch.setattr(cute, "compile", forbidden)
    for rows in ROWS:
        a, g = attn[:rows].contiguous(), gate[:rows].contiguous()
        out = _run(k, a, g, scale)
        assert_reference(out, a, g, scale, f"rows {rows} after one compile")
        want = _bytes(out).clone()
        assert torch.equal(_bytes(_run(k, a, g, scale_tensor(0.01, dev, zero_d=True))), want), "a 0-d scale"
        mine = torch.empty(rows, width, dtype=E4M3, device=dev)
        assert torch.equal(_bytes(_run(k, a, g, scale, mine)), want)
    k.compile(attn, gate, scale)  # a second compile() is a no-op
    assert len(k.compiled) == 1


def test_compile_declines_a_dsl_below_the_floor_before_the_kernel_is_built(monkeypatch):
    dev = _device_or_skip()
    from cudnn.frost import buffers

    k = entry.AttentionGateQuantDecode(dev, 2048)
    attn, gate = make_operands(1, 2048, dev)
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
    with pytest.raises(NotImplementedError, match="requires nvidia-cutlass-dsl >= 4.8.0; found 4.7.1"):
        k.compile(attn, gate, scale_tensor(0.01, dev))
    with pytest.raises(NotImplementedError, match="found 4.7.1"):
        k(attn, gate, scale_tensor(0.01, dev))
    assert not k.compiled


def test_graph_capture_replays_bitwise_with_eager():
    """Several layers' launches captured into one CUDA graph after ``compile()``: the replay is bitwise the eager run, and after the
    operands and the scale change IN PLACE the replay is bitwise a fresh eager run on the new values (every tensor is a launch argument:
    the graph holds pointers, never values)."""
    dev = _device_or_skip()
    rows, width = 6, 2048
    k = entry.AttentionGateQuantDecode(dev, width)
    layers = [make_operands(rows, width, dev, seed=100 + i, kind="adversarial" if i == 1 else "random") for i in range(4)]
    scales = [scale_tensor(v, dev) for v in (0.01, 0.02, 1.0, 37.5)]
    outs = [torch.empty(rows, width, dtype=E4M3, device=dev) for _ in layers]
    eager = [_bytes(_run(k, a, g, s)).clone() for (a, g), s in zip(layers, scales)]  # compile + warm
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for (a, g), s, o in zip(layers, scales, outs):
            k(a, g, s, o)
    for o in outs:
        _bytes(o).fill_(0xA5)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for want, o in zip(eager, outs):
            assert torch.equal(_bytes(o), want), "graph replay bitwise with eager"
    for i, (a, g) in enumerate(layers):  # new values in the same storage
        na, ng = make_operands(rows, width, dev, seed=500 + i)
        a.copy_(na)
        g.copy_(ng)
        scales[i].fill_((0.02, 1.0, 37.5, 0.01)[i])
    graph.replay()
    torch.cuda.synchronize()
    replayed = [_bytes(o).clone() for o in outs]
    for (a, g), s, got in zip(layers, scales, replayed):
        assert torch.equal(_bytes(_run(k, a, g, s)), got), "the replay reads the operands' current values"
        assert_reference(got, a, g, s, "replay after an in-place change")


def test_warm_call_allocates_nothing_and_never_synchronizes():
    """Rules 1 and 3 on the warm path: with a caller-owned ``out`` three calls allocate no device memory, and a call under
    ``set_sync_debug_mode('error')`` raises nothing (no device read reaches the host; ``scale``'s value is never fetched)."""
    dev = _device_or_skip()
    rows, width = 8, 2048
    k = entry.AttentionGateQuantDecode(dev, width)
    attn, gate = make_operands(rows, width, dev, seed=41)
    scale = scale_tensor(0.01, dev)
    mine = torch.empty(rows, width, dtype=E4M3, device=dev)
    want = _bytes(_run(k, attn, gate, scale, mine)).clone()
    _run(k, attn, gate, scale)  # the entry-owned buffer exists too
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats(dev)["allocation.all.allocated"]
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            k(attn, gate, scale, mine)
            k(attn, gate, scale)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats(dev)["allocation.all.allocated"] == allocated, "a warm call allocated device memory"
    assert torch.equal(_bytes(mine), want)


def test_warm_entry_rejects_a_bad_binding_before_the_compiled_kernel_is_looked_up():
    dev = _device_or_skip()
    k = entry.AttentionGateQuantDecode(dev, 2048)
    attn, gate = make_operands(2, 2048, dev)
    scale = scale_tensor(0.01, dev)
    want = _bytes(_run(k, attn, gate, scale)).clone()
    with pytest.raises(TypeError, match="attn must be torch.bfloat16"):
        k(attn.float(), gate, scale)
    with pytest.raises(ValueError, match="every operand must live on one device"):
        k(attn.cpu(), gate, scale)
    with pytest.raises(ValueError, match="attn must be .rows, 2048."):
        k(attn[:, :1024].contiguous(), gate[:, :1024].contiguous(), scale)
    with pytest.raises(ValueError, match="scale must be one fp32 element"):
        k(attn, gate, torch.tensor([0.01, 0.02], device=dev))
    with pytest.raises(ValueError, match="out must not overlap attn"):
        k(attn, gate, scale, attn.view(U8).reshape(-1)[: 2 * 2048].view(E4M3).view(2, 2048))  # the first half of attn's own bytes
    torch.cuda.synchronize()
    assert torch.equal(_bytes(k.out_buffer)[:2], want), "a declined warm call launches nothing"


def test_reciprocal_primitive_is_the_correctly_rounded_division():
    """``pointwise.rcp_div_rn`` (appended with this step; the reciprocal of a static dequantization scale): a probe kernel stores
    ``rcp_div_rn(s)`` for a sweep of fp32 scales -- powers of two, the suite's scales, values next to them, a log-uniform draw over the
    normal range, both signs -- bitwise the host's IEEE ``1 / s``."""
    dev = _device_or_skip()
    import rcp_div_rn_probe as probe

    gen = torch.Generator(device="cpu").manual_seed(8)
    drawn = torch.exp(torch.empty(4000).uniform_(-80.0, 80.0, generator=gen))
    special = torch.tensor([1.0, 2.0, 0.5, 3.0, 7.0, 0.01, 0.02, 37.5, 2.0**-7, 2.0**-10, 1e-10, 3.0e38, 1.1754944e-38, 448.0, 1.0 / 3.0, 0.1])
    near = torch.cat([torch.nextafter(special, torch.full_like(special, float("inf"))), torch.nextafter(special, torch.zeros_like(special))])
    scales = torch.cat([special, near, drawn]).to(torch.float32)
    scales = torch.cat([scales, -scales]).to(dev).contiguous()
    got = probe.run(scales).cpu()
    want = torch.ones((), dtype=torch.float32) / scales.cpu()  # the host's IEEE division: independent of any GPU arithmetic
    diff = got.view(torch.int32) != want.view(torch.int32)
    assert not bool(
        diff.any()
    ), f"{int(diff.sum())} of {scales.numel()} reciprocals differ from the IEEE division (first: scale {float(scales.cpu()[diff][0])!r})"
