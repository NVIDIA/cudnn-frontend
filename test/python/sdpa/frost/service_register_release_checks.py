# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile every service-role body and check the actual PTX release sites."""

import inspect
import json
import re
import subprocess
import sys

import pytest

from frost_test_utils import arch_known_to_the_dsl


def _service_entries(ptx):
    """Select the template's named _kernel, excluding reset/unscale helpers."""
    return [entry for entry in re.split(r"\.visible\s+\.entry\s+", ptx)[1:] if entry.startswith("cudnn_kernel__kernel_")]


def _assert_service_guard(entry, first_warp, n_warps):
    """Check the emitted guard and trace its operand back to each lane's tid.x."""
    guard = re.search(
        r"setp\.(?P<cmp>lt|ne)\.(?:s32|u32|b32)\s+(?P<pred>%p\d+),\s*(?P<reg>%r\d+),\s*(?P<bound>\d+);\s*"
        r"@(?P=pred)\s+bra\s+(?P<label>\$\w+);\s*"
        r"(?P<release>setmaxnreg\.dec\.sync\.aligned\.u32\s+\d+;)\s*(?P=label):",
        entry,
    )
    assert guard and guard.start("release") == entry.index("setmaxnreg."), "unrecognized common service-release guard"

    def values(reg, before):
        if reg == "%tid.x":
            return list(range(n_warps * 32))
        definitions = list(re.finditer(r"(?m)^\s*([\w.]+)\s+" + re.escape(reg) + r",\s*([^;]+);", entry[:before]))
        assert definitions, ("missing warp-id definition", reg)
        definition = definitions[-1]
        op = definition[1]
        args = [arg.strip() for arg in definition[2].split(",")]
        source = values(args[0], definition.start())
        if op in ("mov.u32", "mov.b32"):
            assert len(args) == 1, definition[0]
            return source
        if op == "shr.u32":
            assert args[1:] == ["5"], definition[0]
            return [value >> 5 for value in source]
        assert op == "shfl.sync.idx.b32" and args[1:] == ["0", "31", "-1"], ("unsupported warp-id definition", definition[0])
        return [source[(lane // 32) * 32] for lane in range(n_warps * 32)]

    warp_ids = values(guard["reg"], guard.start())
    bound = int(guard["bound"])
    # The branch skips the release. Check all lanes, including excluded compute warps.
    participants = [lane for lane, warp in enumerate(warp_ids) if not (warp < bound if guard["cmp"] == "lt" else warp != bound)]
    assert participants == list(range(first_warp * 32, (first_warp + 4) * 32)), ("service warp participants", participants, first_warp)


_PROBE = r"""
import faulthandler, json, os, pathlib, re, sys
faulthandler.dump_traceback_later(270, exit=True)
dump, filename, arch, kwargs, negative = sys.argv[1:]
os.environ["CUTE_DSL_DUMP_DIR"] = dump
os.environ["CUTE_DSL_KEEP"] = "ptx"
os.environ["CUTE_DSL_ARCH"] = arch
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
from cudnn.sdpa.fwd.api_dsl import _load_kernel_template
from cudnn.sdpa.fwd.config_sm100 import TemplateParams
params = TemplateParams(**json.loads(kwargs))
mod = _load_kernel_template(filename, params, tag="register_release")
assert pathlib.Path(mod.__file__).as_posix().endswith("/kernels/" + filename)
if negative == "single_warp":
    source = pathlib.Path(mod.__file__).read_text()
    anchor = "if warp_idx >= CFG.MMA_WARP_ID:"
    assert source.count(anchor) == 1
    private = pathlib.Path(dump).parent / "single_warp_release.py"
    private.write_text(source.replace(anchor, "if warp_idx == CFG.MMA_WARP_ID:"))
    mod = _load_kernel_template(str(private), params, tag="single_warp_release")
cfg = mod.CFG
assert cfg.MMA_WARP_ID % 4 == 0
assert (cfg.TMALDG_WARP_ID, cfg.TMASTG_WARP_ID, cfg.SCHED_WARP_ID) == tuple(cfg.MMA_WARP_ID + i for i in (1, 2, 3))
assert cfg.SCHED_WARP_ID + 1 == cfg.THREADS_PER_CTA // 32
print("COMPILING", filename, "service budget", cfg.OTHER_REGS, flush=True)
compile_fn = mod.compile_prepared if "fp8" in filename else mod.compile
compile_fn()
files = list(pathlib.Path(dump).glob("*.ptx"))
assert files, (dump, "no PTX emitted")
sites, entry_bodies = [], []
for path in files:
    ptx = path.read_text()
    # D512 MXFP8 emits a separate entry for each V half in one PTX module.
    if negative.startswith("missing_release_"):
        entries = _service_entries(ptx)
        assert len(entries) == 2, "missing-release control requires both V-half entries"
        entry = entries[int(negative.rsplit("_", 1)[1])]
        stripped = re.sub(r"\bsetmaxnreg\.(?:inc|dec)\.sync\.aligned\.u32\s+\d+\s*;", "", entry)
        assert stripped != entry
        ptx = ptx.replace(entry, stripped, 1)  # Inspect only; never launch invalid PTX.
    for entry in _service_entries(ptx):
        instructions = [(action, int(count)) for action, count in re.findall(r"\bsetmaxnreg\.(inc|dec)\.sync\.aligned\.u32\s+(\d+)\s*;", entry)]
        sites.append({"file": path.name, "entry": entry.split("(", 1)[0], "instructions": instructions})
        entry_bodies.append(entry)
expected_entries = 2 if filename == "sm100/prefill_d512_mxfp8.py" else 1
assert len(sites) == expected_entries, ("missing or unexpected service-kernel entries", sites)
result = {"template": filename, "arch": arch, "service_budget": cfg.OTHER_REGS, "sites": sites}
print("RELEASE_RESULT", json.dumps(result), flush=True)
for site, entry in zip(sites, entry_bodies):
    assert site["instructions"], ("missing register redistribution", site)
    _assert_service_guard(entry, cfg.MMA_WARP_ID, cfg.THREADS_PER_CTA // 32)
    instructions = site["instructions"]
    # The common service release precedes the role-local softmax acquisitions.
    assert instructions[0] == ("dec", cfg.OTHER_REGS), result
    releases = [count for action, count in instructions if action == "dec"]
    # D192 half uses the same 40-register budget for its correction warpgroup.
    correction_matches = getattr(cfg, "CORRECTION_REGS", None) == cfg.OTHER_REGS
    assert releases.count(cfg.OTHER_REGS) == 1 + correction_matches, result
    assert len(releases) <= 2, result
faulthandler.cancel_dump_traceback_later()
"""


def release_cases(arch):
    """One dense specialization per changed template, with its native CGA."""
    names = [f"prefill_{shape}_{dtype}.py" for shape in ("d128", "d192_d128", "d256", "d512") for dtype in ("f16", "fp8", "mxfp8")]
    names.append("prefill_d512_f16_2x2.py")
    if arch == "sm100":
        names.append("decode_d128_f16.py")
    else:
        names.remove("prefill_d128_f16.py")  # Already covered by TestNativeHalfRegisterRelease.
    cases = []
    for name in names:
        cga = 2
        if name.startswith("decode_") or (arch == "sm100" and name in ("prefill_d256_fp8.py", "prefill_d256_mxfp8.py", "prefill_d512_mxfp8.py")):
            cga = 1
        kwargs = dict(cta_mma=cga, dtype_qkv=2 if "f16" in name else 0, dtype_o=2, mma_2x2="2x2" in name)
        moved = {
            "prefill_d256_f16.py": "d256_f16_lookahead/kernel.py",
            "prefill_d512_f16.py": "d512_f16_role_split/kernel.py",
            "prefill_d512_f16_2x2.py": "d512_f16_2x2/kernel.py",
        }
        path = moved.get(name, name) if arch == "sm100" else name
        cases.append(pytest.param(f"{arch}/{path}", kwargs, id=name.removesuffix(".py")))
    return cases


def check_service_release(tmp_path, filename, params, *, negative=""):
    arch = filename.split("/")[0].replace("sm", "sm_") + "a"
    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"this cutlass-dsl has no {arch}")
    dump = tmp_path / ("ptx-" + negative if negative else "ptx")
    dump.mkdir()
    probe = inspect.getsource(_service_entries) + "\n" + inspect.getsource(_assert_service_guard) + "\n" + _PROBE
    argv = [sys.executable, "-c", probe, str(dump), filename, arch, json.dumps(params), negative]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"{filename} compile timeout:\n{exc.stdout}\n{exc.stderr}")
    if negative:
        reason = "service warp participants" if negative == "single_warp" else "missing register redistribution"
        assert proc.returncode != 0 and reason in proc.stderr, f"{negative} control was not rejected:\n{proc.stdout}\n{proc.stderr}"
        return
    assert proc.returncode == 0, f"{filename}:\n{proc.stdout[-6000:]}\n{proc.stderr[-6000:]}"
    assert "RELEASE_RESULT" in proc.stdout, proc.stdout
    print(proc.stdout)


@pytest.mark.parametrize("filename,params", release_cases("sm100"))
def test_sm100_service_register_release(tmp_path, filename, params):
    check_service_release(tmp_path, filename, params)
    if filename == "sm100/prefill_d128_fp8.py":
        # Compile only: never launch the deliberately invalid redistribution.
        check_service_release(tmp_path, filename, params, negative="single_warp")
    if filename == "sm100/prefill_d512_mxfp8.py":
        for half in range(2):
            check_service_release(tmp_path, filename, params, negative=f"missing_release_{half}")


@pytest.mark.parametrize("filename,params", release_cases("sm107"))
def test_sm107_service_register_release(tmp_path, filename, params):
    check_service_release(tmp_path, filename, params)
