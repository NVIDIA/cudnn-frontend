# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SASS pins for the sm120 FROST GEMM templates' register split (compile only, no launch).

Every sm120 template hands registers from its producer warps to its compute warps
with setmaxnreg (``PROD_REG_COUNT`` down, ``EPI_REG_COUNT`` up). ptxas honours that
only when the launch declares ``.minnctapersm 1`` (``min_blocks_per_mp=1``): with
``.reqntid`` alone it prints C7508 -- a message the DSL never surfaces -- and drops
every setmaxnreg, so the numerics stay green while the compute warps lose their
budget (the dual MoE kernels fell to 40 registers with 11 KB of spills). Once the
split is live the producer budget is a hard limit too: a GATHER producer whose
elected lane held the whole tile's token rows spilled at 24 registers and ran up to
1.45x slower.

A subprocess trace-compiles one plan per case with ``CUTE_DSL_KEEP=cubin`` and
disassembles it, as the frost SDPA SASS pins do. Spill counts are toolchain facts,
so the GATHER rows pin a bound, not an exact count: STL + LDL in the whole cubin,
set between today's count and the one-lane producer's (in parentheses below). The
block-scale GATHER kernels keep a few benign spills from their fully unrolled
scale-factor repack -- a rolled, spill-free repack measured 2.4% slower.

The MoE COMBINE cases also pin the store itself: where one lane's chunk is adjacent
features of one token row (token-by-weight, N-major output), the weighted top-k
scatter-add must reach the SASS as 16-byte vector reductions (REDG.E.ADD.BF16x8 /
F32x4) -- a refactor that falls back to per-element adds stays numerically green.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap

import pytest

from gemm_test_utils import requires_sm120

pytestmark = [pytest.mark.L0, requires_sm120]

# case -> (kind, combo, mode, dual, CTA tile + warp grid, swap_ab, spill bound or None);
# a bound's comment reads "today (one-lane producer)" in STL + LDL.
_CASES = {
    "dense_bf16": ("dense", "bf16", "none", False, "128x128_4x2", False, None),
    "dense_nvfp4": ("dense", "nvfp4", "none", False, "128x128_4x2", False, None),
    "moe_bf16": ("moe", "bf16", "none", False, "128x128_4x2", False, None),
    "moe_bf16_dual": ("moe", "bf16", "none", True, "128x128_4x2", False, None),
    "moe_bf16_gather": ("moe", "bf16", "gather", False, "128x128_4x2", False, 8),  # 0 (61)
    "moe_bf16_gather_dual": ("moe", "bf16", "gather", True, "128x128_4x2", False, 8),  # 0 (64)
    "moe_bf16_gather_swap": ("moe", "bf16", "gather", False, "128x64_4x2", True, 8),  # 0 (45)
    "moe_nvfp4": ("moe", "nvfp4", "none", False, "128x128_4x2", False, None),
    "moe_nvfp4_gather": ("moe", "nvfp4", "gather", False, "128x128_4x2", False, 64),  # 34 (110)
    "moe_nvfp4_gather_dual": ("moe", "nvfp4", "gather", True, "128x128_4x2", False, 120),  # 81 (163)
    "moe_nvfp4_gather_256": ("moe", "nvfp4", "gather", False, "256x128_2x4", False, 200),  # 108 (322)
    "moe_nvfp4_gather_swap": ("moe", "nvfp4", "gather", False, "128x64_4x2", True, 8),  # 0 (30)
    "moe_bf16_combine": ("moe", "bf16", "combine", False, "128x128_4x2", False, None),
    "moe_bf16_combine_f32": ("moe", "bf16", "combine_f32", False, "128x128_4x2", False, None),
    "moe_bf16_combine_swap": ("moe", "bf16", "combine", False, "128x64_4x2", True, None),
    "moe_nvfp4_combine": ("moe", "nvfp4", "combine", False, "128x128_4x2", False, None),
    "moe_nvfp4_combine_swap": ("moe", "nvfp4", "combine", False, "128x64_4x2", True, None),
}

# COMBINE case -> the 16-byte vector reduction its N-major token-by-weight store must emit.
_COMBINE_VECTOR_REDUCTION = {
    "moe_bf16_combine": "REDG.E.ADD.BF16x8",
    "moe_bf16_combine_f32": "REDG.E.ADD.F32x4",
    "moe_nvfp4_combine": "REDG.E.ADD.BF16x8",
}


def _graph(kind, combo, mode, dual):
    """The case's graph (shapes only matter for the compile, never launched)."""
    import cudnn

    dt = cudnn.data_type
    E, T, R, N, K = 4, 512, 1024, 256, 512
    g = cudnn.pygraph(io_data_type=dt.BFLOAT16, intermediate_data_type=dt.FLOAT, compute_data_type=dt.FLOAT)
    bs = combo != "bf16"
    bsz, a_dt, sf_dt = (16, dt.FP4_E2M1, dt.FP8_E4M3) if bs else (None, dt.BFLOAT16, None)
    sf_k = K // bsz if bs else 0
    if kind == "dense":
        a = g.tensor(name="A", dim=[1, R, K], stride=[R * K, K, 1], data_type=a_dt)
        b = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=a_dt)
        if bs:
            sfa = g.tensor(name="SFA", dim=[1, R, sf_k], stride=[R * sf_k, sf_k, 1], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4)
            sfb = g.tensor(name="SFB", dim=[1, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4)
            a = g.block_scale_dequantize(input=a, descale=sfa, block_size=[1, bsz])
            b = g.block_scale_dequantize(input=b, descale=sfb, block_size=[bsz, 1])
        y = g.matmul(A=a, B=b, compute_data_type=dt.FLOAT)
        y.set_output(True).set_data_type(dt.BFLOAT16)
        return g
    src = T if mode == "gather" else R
    tok = g.tensor(name="token", dim=[1, src, K], stride=[src * K, K, 1], data_type=a_dt)
    if bs:
        pitch = -(-sf_k // 16) * 16 if mode == "gather" else sf_k
        reorder = cudnn.tensor_reordering.NONE if mode == "gather" else cudnn.tensor_reordering.F8_128x4
        sfa = g.tensor(name="SFA", dim=[1, src, sf_k], stride=[src * pitch, pitch, 1], data_type=sf_dt, reordering_type=reorder)
        tok = g.block_scale_dequantize(input=tok, descale=sfa, block_size=[1, bsz])
    fto = g.tensor(name="fto", dim=[E + 1, 1, 1], stride=[1, 1, 1], data_type=dt.INT32)
    routing = {"mode": cudnn.moe_grouped_matmul_mode.NONE}
    if mode == "gather":
        idx = g.tensor(name="token_index", dim=[1, R, 1], stride=[R, 1, 1], data_type=dt.INT32)
        routing = {"token_index": idx, "mode": cudnn.moe_grouped_matmul_mode.GATHER}
    elif mode.startswith("combine"):
        idx = g.tensor(name="token_index", dim=[1, R, 1], stride=[R, 1, 1], data_type=dt.INT32)
        ks = g.tensor(name="token_ks", dim=[1, R, 1], stride=[R, 1, 1], data_type=dt.INT32)
        scores = g.tensor(name="top_k_scores", dim=[1, T, R // T], stride=[R, R // T, 1], data_type=dt.FLOAT)
        routing = {"token_index": idx, "token_ks": ks, "top_k_scores": scores, "top_k": R // T, "mode": cudnn.moe_grouped_matmul_mode.COMBINE}
    mms = []
    for i in range(2 if dual else 1):
        w = g.tensor(name=f"w{i}", dim=[E, K, N], stride=[K * N, 1, K], data_type=a_dt)
        if bs:
            sfb = g.tensor(name=f"SFB{i}", dim=[E, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4)
            w = g.block_scale_dequantize(input=w, descale=sfb, block_size=[bsz, 1])
        mms.append(g.moe_grouped_matmul(tok, w, fto, compute_data_type=dt.FLOAT, **routing))
    y = g.mul(a=g.swish(input=mms[0]), b=mms[1]) if dual else mms[0]
    y.set_output(True).set_data_type(dt.FLOAT if mode == "combine_f32" else dt.BFLOAT16)
    return g


def _compile_case(case):
    """Trace-compile ``case`` (subprocess side); returns the compiled plan."""
    from dataclasses import replace

    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
    from cudnn.gemm.frost.tile_config import by_name

    kind, combo, mode, dual, tile, swap, _bound = _CASES[case]
    cta, warps = tile.split("_")
    cfg = replace(by_name(f"CONFIG_sm120_{cta}x128_16x16x32_cluster1x1_warps{warps}"), swap_ab=swap)
    return jit_from_cudnn_graph(_graph(kind, combo, mode, dual), config=cfg)


_PROBE = textwrap.dedent(r"""
    import glob, json, os, re, subprocess, sys
    here, dump, case, cands = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "cubin"
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    sys.path.insert(0, here)
    import test_sm120_register_split as T
    compiled = T._compile_case(case)
    cubins = sorted(glob.glob(os.path.join(dump, "**", "*.cubin"), recursive=True))
    sass = None
    for c in cands:
        try:
            outs = [subprocess.run([c, "-c", cb], capture_output=True, text=True, timeout=300) for cb in cubins]
        except (OSError, subprocess.SubprocessError):
            continue
        if outs and all(p.returncode == 0 for p in outs):
            sass = "\n".join(p.stdout for p in outs).splitlines()
            break
    if sass is None:
        print("SKIP no nvdisasm decodes", cubins); sys.exit(0)
    dealloc = [int(ln.split()[-2], 16) for ln in sass if "USETMAXREG.DEALLOC" in ln]
    alloc = [int(ln.split(",")[-1].strip(" ;"), 16) for ln in sass if "USETMAXREG.TRY_ALLOC" in ln]
    stl = sum(1 for ln in sass if re.search(r"\bSTL(\.[0-9A-Z]+)*\s", ln))
    ldl = sum(1 for ln in sass if re.search(r"\bLDL(\.[0-9A-Z]+)*\s", ln))
    red = sorted({m for ln in sass for m in re.findall(r"\bREDG\.E\.ADD\.[0-9A-Za-z]+", ln)})
    print("RESULT " + json.dumps(dict(generated=str(compiled.generated_path), dealloc=dealloc, alloc=alloc, stl=stl, ldl=ldl, red=red, cubins=len(cubins))))
    """)


def _nvdisasm_candidates():
    """$CUDA_PATH/bin first (a toolkit that decodes a newer arch than the wheel's), then $PATH."""
    cands = [os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm")] if os.environ.get("CUDA_PATH") else []
    if shutil.which("nvdisasm"):
        cands.append(shutil.which("nvdisasm"))
    return [c for c in dict.fromkeys(cands) if os.path.isfile(c) and os.access(c, os.X_OK)]


def _rendered_reg_counts(path):
    """``(PROD_REG_COUNT, EPI_REG_COUNT)`` of the rendered kernel module."""
    src = open(path).read()
    return tuple(int(re.search(rf"^{name}\s*=\s*(\d+)", src, re.M).group(1)) for name in ("PROD_REG_COUNT", "EPI_REG_COUNT"))


@pytest.mark.parametrize("case", sorted(_CASES))
def test_sm120_register_split_reaches_the_sass(tmp_path, case):
    """The template's setmaxnreg pair survives ptxas (one warpgroup-uniform DEALLOC
    and one TRY_ALLOC, at the template's own counts), and a GATHER producer keeps
    its spills under the bound once the split is live."""
    cands = _nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable (CUDA_PATH unset and none on PATH)")
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, os.path.dirname(os.path.abspath(__file__)), str(tmp_path), case, *cands],
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert proc.returncode == 0, f"{case}: trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in lines):
        pytest.skip(next(ln for ln in lines if ln.startswith("SKIP")))
    res = json.loads(next(ln for ln in lines if ln.startswith("RESULT ")).split(" ", 1)[1])
    prod, epi = _rendered_reg_counts(res["generated"])
    assert res["dealloc"] and res["alloc"], f"{case}: no USETMAXREG in the SASS -- ptxas dropped the register split (C7508): {res}"
    assert (res["dealloc"], res["alloc"]) == ([prod], [epi]), f"{case}: the split is not the template's {prod}/{epi} at one site each: {res}"
    bound = _CASES[case][-1]
    if bound is not None:
        assert res["stl"] + res["ldl"] <= bound, (
            f"{case}: {res['stl']} STL + {res['ldl']} LDL > {bound}: the GATHER producer spills at {prod} registers "
            f"(keep its per-lane state small -- each TMA-warp lane reads its own rows and issues its own GATHER4s)"
        )
    if case in _COMBINE_VECTOR_REDUCTION:
        want = _COMBINE_VECTOR_REDUCTION[case]
        assert want in res["red"], f"{case}: the COMBINE store lost its 16-byte vector reduction {want}; SASS has {res['red']}"
