# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile every service-role body and check the actual PTX release sites."""

import json
import subprocess
import sys

import pytest

from frost_test_utils import arch_known_to_the_dsl

_PROBE = r"""
import faulthandler, json, os, pathlib, re, sys
faulthandler.dump_traceback_later(270, exit=True)
dump, filename, arch, kwargs = sys.argv[1:]
os.environ["CUTE_DSL_DUMP_DIR"] = dump
os.environ["CUTE_DSL_KEEP"] = "ptx"
os.environ["CUTE_DSL_ARCH"] = arch
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
from cudnn.sdpa.fwd.api_dsl import _load_kernel_template
from cudnn.sdpa.fwd.config_sm100 import TemplateParams
params = TemplateParams(**json.loads(kwargs))
mod = _load_kernel_template(filename, params, tag="register_release")
assert pathlib.Path(mod.__file__).as_posix().endswith("/kernels/" + filename)
cfg = mod.CFG
assert cfg.MMA_WARP_ID % 4 == 0
assert (cfg.TMALDG_WARP_ID, cfg.TMASTG_WARP_ID, cfg.SCHED_WARP_ID) == tuple(cfg.MMA_WARP_ID + i for i in (1, 2, 3))
assert cfg.SCHED_WARP_ID + 1 == cfg.THREADS_PER_CTA // 32
print("COMPILING", filename, "service budget", cfg.OTHER_REGS, flush=True)
compile_fn = mod.compile_prepared if "fp8" in filename else mod.compile
compile_fn()
files = list(pathlib.Path(dump).glob("*.ptx"))
assert files, (dump, "no PTX emitted")
sites = []
for path in files:
    ptx = path.read_text()
    # D512 MXFP8 emits a separate entry for each V half in one PTX module.
    for entry in re.split(r"\.visible\s+\.entry\s+", ptx)[1:]:
        instructions = [(action, int(count)) for action, count in re.findall(r"\bsetmaxnreg\.(inc|dec)\.sync\.aligned\.u32\s+(\d+)\s*;", entry)]
        if instructions:
            sites.append({"file": path.name, "entry": entry.split("(", 1)[0], "instructions": instructions})
assert sites, "no register redistribution in emitted entries"
result = {"template": filename, "arch": arch, "service_budget": cfg.OTHER_REGS, "sites": sites}
print("RELEASE_RESULT", json.dumps(result), flush=True)
for site in sites:
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
        cases.append(pytest.param(f"{arch}/{name}", kwargs, id=name.removesuffix(".py")))
    return cases


def check_service_release(tmp_path, filename, params):
    arch = filename.split("/")[0].replace("sm", "sm_") + "a"
    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"this cutlass-dsl has no {arch}")
    dump = tmp_path / "ptx"
    dump.mkdir()
    argv = [sys.executable, "-c", _PROBE, str(dump), filename, arch, json.dumps(params)]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"{filename} compile timeout:\n{exc.stdout}\n{exc.stderr}")
    assert proc.returncode == 0, f"{filename}:\n{proc.stdout[-6000:]}\n{proc.stderr[-6000:]}"
    assert "RELEASE_RESULT" in proc.stdout, proc.stdout
    print(proc.stdout)


@pytest.mark.parametrize("filename,params", release_cases("sm100"))
def test_sm100_service_register_release(tmp_path, filename, params):
    check_service_release(tmp_path, filename, params)


@pytest.mark.parametrize("filename,params", release_cases("sm107"))
def test_sm107_service_register_release(tmp_path, filename, params):
    check_service_release(tmp_path, filename, params)
