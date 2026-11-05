# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The d128 accumulation DIAGNOSTIC lever ``diag_delete_dq_reduce_add`` (default OFF) renders the shipped kernels byte-identically.

The lever makes the store warp of the THD body of ``kernels/sm100/bprop_d128_f16.py`` (shared by the grouped THD kernel) skip the four fp32 dQ
``tma_reduce_add`` issues per q tile while every barrier arrive / wait, ``tma_store_commit`` and ``tma_store_wait`` stay.
Armed, dQ stays zero BY DESIGN: it is a timing-only ceiling for the L2 reduce-add term, set on the adapter instance before
``compile()`` by a probe driver, never by a graph.  Pins (host trace-compiles for sm_100a, no device):

* the DEFAULT rendering of both THD hosts equals the committed pre-lever record (``renderings/md5_d128_accum_thd_default_sm100a.txt``);
* the ARMED rendering differs, carries ZERO ``cp.reduce.async.bulk`` where the default carries four, and keeps the
  ``mbarrier`` arrive / wait and bulk-group commit / wait instruction counts of the default (the barrier table is unchanged);
* the record field, the kernel, the host and the ``compile_host`` entries all default to ``False`` (no GPU, no render).
"""

import inspect
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from frost_test_utils import D128_THD_SHARED_KERNELS, arch_known_to_the_dsl, assert_kernel_record, ptx_kernel_md5s, read_kernel_record, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

FLAG = "diag_delete_dq_reduce_add"
_RECORD = Path(__file__).resolve().parent / "renderings" / "md5_d128_accum_thd_default_sm100a.txt"
_HOSTS = ("thd", "grouped_thd")
# The probe geometry (B, Hq, Hkv) of the record: the 16K / 32K cells' heads; a render is a function of the declared heads only.
_B, _H, _HK = 4, 32, 4
# PTX instruction census: the lever deletes REDUCE_ADD only; every other row must be equal between the two renderings.
_CENSUS = {
    "REDUCE_ADD": r"cp\.reduce\.async\.bulk",
    "MBARRIER_ARRIVE": r"mbarrier\.arrive",
    "MBARRIER_WAIT": r"mbarrier\.(try_wait|test_wait)",
    "BULK_COMMIT": r"cp\.async\.bulk\.commit_group",
    "BULK_WAIT": r"cp\.async\.bulk\.wait_group",
}
_REDUCE_ADDS_PER_Q_TILE = 4  # the four 16 KB fp32 dQ slices the store loop issues per q tile (one `.entry` renders them once)

_PROBE = textwrap.dedent(r"""
    import glob, hashlib, json, os, re, sys
    dump, host, flag, spec = sys.argv[1], sys.argv[2], sys.argv[3] == "1", json.loads(sys.argv[4])
    arch = sys.argv[6]  # the target: the host's ``--gpu-arch sm_{sm}a`` (the production selector, from the device cc) wins; CUTE_DSL_ARCH agrees
    sm = int(re.fullmatch(r"sm_(\d+)a", arch).group(1))
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "ptx"
    os.environ["CUTE_DSL_ARCH"] = arch
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    import cutlass
    B, H, HK = spec["b"], spec["h"], spec["hk"]
    ports = ("q", "k", "v", "o", "do", "dq", "dk", "dv", "stats")
    kv = ("k", "v", "dk", "dv")
    def spec_for(heads_q, heads_k):
        return tuple((i, 1, heads_q if p == "stats" else (heads_k if p in kv else heads_q) * 128, "int32") for i, p in enumerate(ports))
    dtypes = tuple("int32" for _ in ports)
    kw = {"diag_delete_dq_reduce_add": True} if flag else {}
    if host == "grouped_thd":
        from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_host import compile_host_grouped_thd as compile_host
        compile_host((B, H, HK, True), spec_for(H, HK), dtypes, cutlass.BFloat16, True, sm, **kw)
    else:
        from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_host import compile_host_thd as compile_host
        compile_host((B, H, True), spec_for(H, H), dtypes, cutlass.BFloat16, True, sm, **kw)
    ptxs = sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)
    if not ptxs:
        print("FAIL no ptx dumped into", dump, os.listdir(dump)); sys.exit(3)
    txt = open(ptxs[-1], "rb").read().decode()
    print("PTX_TARGET", re.search(r"^\.target\s+(\S+)", txt, re.M).group(1))
    print("PTX_MD5", hashlib.md5(txt.encode()).hexdigest())
    print("PTX_FILE", ptxs[-1])
    for key, pat in json.loads(sys.argv[5]).items():
        print("CENSUS", key, len(re.findall(pat, txt)))
    """)


_RECORD_TOKEN = "d128_accum"
# The main kernel of each THD host (one record row per kernel the host's chain OWNS; the grouped chain's group fold is shared with the pair
# chain: rendered, never pinned here -- frost_test_utils.D128_THD_SHARED_KERNELS).
_MAIN_KERNEL = {"thd": "bprop_d128_accum_thd_kernel", "grouped_thd": "bprop_d128_accum_grouped_thd_kernel"}


def _parse_record(f):
    """-> (dsl line or None, {host: {kernel: md5}}) of the per-kernel record (frost_test_utils.read_kernel_record)."""
    return read_kernel_record(f, _RECORD_TOKEN)


def _render_kernels(tmp_path, host, armed, arch="sm_100a"):
    """Host trace-compile ONE d128 accumulation THD host for ``arch`` (the record's sm_100a by default; sm_107a = the Rubin target the prepared
    module selects on cc 10.7) in a fresh process -> (md5, census, {kernel: md5} in entry order).  Skips BY NAME on a DSL without the target."""
    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"this cutlass-dsl has no {arch} target")
    dump = tmp_path / f"{arch}_d128_accum_{host}_{'armed' if armed else 'default'}"
    dump.mkdir()
    script = dump / "ptx_probe.py"
    script.write_text(_PROBE)
    spec = json.dumps(dict(b=_B, h=_H, hk=_HK))
    cmd = [sys.executable, str(script), str(dump), host, "1" if armed else "0", spec, json.dumps(_CENSUS), arch]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    what = f"{arch} trace-compile of the {host} host ({'armed' if armed else 'default'})"
    assert proc.returncode == 0, f"{what} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    assert re.search(r"^PTX_TARGET (\S+)$", proc.stdout, re.M).group(1) == arch, proc.stdout[-500:]
    md5, ptx_file = None, None
    census = {}
    for ln in proc.stdout.splitlines():
        if ln.startswith("PTX_MD5 "):
            md5 = ln.split()[1]
        elif ln.startswith("PTX_FILE "):
            ptx_file = ln.split(" ", 1)[1]
        elif ln.startswith("CENSUS "):
            _, key, n = ln.split()
            census[key] = int(n)
    assert md5 and ptx_file and set(census) == set(_CENSUS), proc.stdout[-2000:]
    with open(ptx_file) as f:
        kernels = ptx_kernel_md5s(f.read())
    return md5, census, kernels


def _render(tmp_path, host, armed):
    """Host trace-compile ONE d128 accumulation THD host for sm_100a in a fresh process -> (md5, census)."""
    return _render_kernels(tmp_path, host, armed)[:2]


def _skip_unless_recorded_dsl(dsl):
    from cudnn.frost.buffers import cutedsl_state

    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if dsl is not None and have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: PTX text differs by DSL build, re-render the record")


def test_record_is_committed_and_complete():
    """The pin's baseline is in the tree: a DSL line and, per THD host, one md5 per kernel the host's chain owns (its main kernel among them);
    the shared group fold has no row (it is pinned by its own module, not by every chain that launches it)."""
    assert _RECORD.is_file(), _RECORD
    dsl, want = _parse_record(_RECORD)
    assert dsl and dsl.startswith("nvidia-cutlass-dsl "), dsl
    assert set(want) == set(_HOSTS), (sorted(want), _HOSTS)
    for host in _HOSTS:
        assert _MAIN_KERNEL[host] in want[host] and not set(want[host]) & set(D128_THD_SHARED_KERNELS), (host, sorted(want[host]))


@pytest.mark.parametrize("host", _HOSTS)
def test_default_rendering_ptx_md5_is_unchanged(tmp_path, host):
    """The default rendering is PTX-IDENTICAL to the tree before the lever (the flag is a trace-time constant that folds to
    the shipped code; a kernel argument it may be, a knob, env var or runtime branch it is not)."""
    dsl, want = _parse_record(_RECORD)
    _skip_unless_recorded_dsl(dsl)
    got, census, kernels = _render_kernels(tmp_path, host, armed=False)
    print(f"\nSM100 d128 accum {host} default: PTX md5 {got} census {census} kernels {kernels} (record {want.get(host)})")
    assert census["REDUCE_ADD"] == _REDUCE_ADDS_PER_Q_TILE, census
    assert _MAIN_KERNEL[host] in kernels, kernels
    assert_kernel_record(
        kernels, want.get(host, {}), token=_RECORD_TOKEN, variant=host, excluded=D128_THD_SHARED_KERNELS, what=f"{host}: the default rendering"
    )


@pytest.mark.parametrize("host", _HOSTS)
def test_default_rendering_for_sm_107a_matches_the_record(tmp_path, host):
    """The Rubin target (cc 10.7 -> ``sm`` = 107 -> ``--gpu-arch sm_107a``; the cc 10.7 admission of the d128 family, 2026-11-03): each THD
    host's default rendering for sm_107a hashes, kernel by kernel, to the committed sm_100a record -- the bodies have no per-part branch, so
    only the module's `.target` line differs (the sm_103a precedent of the pair chain, PR 1964).  One record file serves; skips BY NAME on a DSL
    without the sm_107a target or on another DSL build -- never a silent pass."""
    dsl, want = _parse_record(_RECORD)
    _skip_unless_recorded_dsl(dsl)
    got, census, kernels = _render_kernels(tmp_path, host, armed=False, arch="sm_107a")
    print(f"\nSM107 d128 accum {host} default: PTX md5 {got} census {census} kernels {kernels} (record {want.get(host)})")
    assert census["REDUCE_ADD"] == _REDUCE_ADDS_PER_Q_TILE, census
    assert _MAIN_KERNEL[host] in kernels, kernels
    assert_kernel_record(
        kernels, want.get(host, {}), token=_RECORD_TOKEN, variant=host, excluded=D128_THD_SHARED_KERNELS, what=f"{host}: the default rendering for sm_107a"
    )


@pytest.mark.parametrize("host", _HOSTS)
def test_armed_rendering_deletes_only_the_reduce_add(tmp_path, host):
    """Armed: a different PTX with ZERO reduce-adds; the mbarrier arrive / wait and bulk-group commit / wait counts are the
    default's (every barrier site and the store-group protocol survive -- the barrier table is unchanged)."""
    default_md5, default = _render(tmp_path, host, armed=False)
    armed_md5, armed = _render(tmp_path, host, armed=True)
    print(f"\nSM100 d128 accum {host}: default {default_md5} {default} | armed {armed_md5} {armed}")
    assert armed_md5 != default_md5, "the armed rendering must differ from the default (the flag folded to nothing)"
    assert default["REDUCE_ADD"] == _REDUCE_ADDS_PER_Q_TILE and armed["REDUCE_ADD"] == 0, (default, armed)
    for key in _CENSUS:
        if key == "REDUCE_ADD":
            continue
        assert default[key] > 0, f"{key}: the census regex matched nothing in the default rendering"
        assert armed[key] == default[key], f"{key}: {armed[key]} armed vs {default[key]} default -- the lever moved a barrier / store-group site"


def test_flag_defaults_to_false_everywhere():
    """The record field is a typed ``bool`` default ``False``; the kernel, the host entries and ``compile_host`` default it too
    (a plain import renders the shipped kernel), and a non-bool on the record is a ``ValueError`` before any compile."""
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.kernels.sm100 import bprop_d128_f16 as packed  # the family module: the THD body (section 5) and the grouped bodies (section 7)
    from cudnn.sdpa.bwd.kernels.sm100 import prepared_d128_host as hosts
    from cudnn.sdpa.bwd.prepared_sm100_d128_thd import diag_delete_dq_reduce_add

    assert SdpaBwdDslSm100D128.diag_delete_dq_reduce_add is False
    assert SdpaBwdDslSm100D128.__annotations__[FLAG] is bool
    for fn in (
        packed.bprop_d128_accum_thd_kernel,
        packed.host_accum_thd,
        packed._store_loop_thd,
        packed.bprop_d128_accum_grouped_thd_kernel,
        packed.bprop_d128_accum_grouped_dense_kernel,
        packed.main_thd_host,
        packed.main_dense_host,
        hosts.compile_host_thd,
        hosts.compile_host_grouped_thd,
        hosts.compile_host_grouped_dense,
    ):
        param = inspect.signature(fn).parameters[FLAG]
        assert param.default is False, (fn, param)

    class _Record:
        _NAME = "probe"

    assert diag_delete_dq_reduce_add(_Record()) is False
    armed = _Record()
    armed.diag_delete_dq_reduce_add = True
    assert diag_delete_dq_reduce_add(armed) is True
    bad = _Record()
    bad.diag_delete_dq_reduce_add = 1
    with pytest.raises(ValueError, match=FLAG):
        diag_delete_dq_reduce_add(bad)
