# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Release-order check of the SM90 BF16 grouped GEMM (forward and dgrad, cooperative and pingpong), compile-only.

Every consumer of a tile-info slot (TMA warp, MMA warp groups, pingpong's skip of the other warp group's tile) loads the
four words with one ``ld.shared.v4`` and releases the slot through ``tile_dsl.consume``: the arrive's address is computed
from the four loaded registers through an ``and`` with the kernel's last parameter, the release-zero argument, so every
loaded word is consumed before the slot is released. The kernels are compiled for sm_90a from fake tensors, so the check
needs no H100.
The disassembly proof (every loaded register read before the arrive, 0 spills) is in the PR text.
"""

from __future__ import annotations

import glob
import os
import re

import pytest

from warm_stress import assert_release_dependence

pytestmark = pytest.mark.L0

EXPERTS, ROWS = 8, 4096

# id: (direction, tile, cluster, pingpong, tile-info arrives in the rendering)
CASES = {
    "fwd-cooperative": ("fwd", (128, 256), (2, 1), False, 4),
    "fwd-pingpong": ("fwd", (128, 192), (2, 1), True, 6),
    "dgrad-cooperative": ("dgrad", (128, 256), (2, 1), False, 4),
    "dgrad-pingpong": ("dgrad", (128, 192), (2, 1), True, 6),
    "fwd-128x128-c11": ("fwd", (128, 128), (1, 1), False, 4),
}


def sm90_ptx(kernel, args, dump):
    """PTX of ``kernel(*args)`` compiled for sm_90a (any GPU or none)."""
    cute = pytest.importorskip("cutlass.cute")
    compiled = cute.compile(kernel, *args, options=f"--gpu-arch sm_90a --keep-ptx --dump-dir {dump}")
    ptx = compiled.__ptx__
    if not ptx:
        found = sorted(glob.glob(os.path.join(dump, "**", "*.sm_90a.ptx"), recursive=True), key=os.path.getmtime)
        assert found, f"no sm_90a PTX kept in {dump}"
        ptx = open(found[-1], encoding="utf-8").read()
    return ptx


def assert_tile_info_releases(ptx, sites):
    """``sites`` tile-info arrives, each computed from the 4 loaded words through the mask; the mask is the last parameter."""
    found = assert_release_dependence(ptx, sites=sites, words=4)
    dependent = [site for site in found if site.dependent]
    listing = "\n".join(f"  line {s.line}: registers={s.loaded_registers} param={s.parameter} :: {s.text}" for s in dependent)
    entry = next(name for name in re.findall(r"\.entry\s+(\w+)", ptx) if "helper_kernel" not in name)
    indices = [int(i) for i in re.findall(r"\b" + re.escape(entry) + r"_param_(\d+)\b", ptx)]
    last = f"{entry}_param_{max(indices)}"
    assert all(site.parameter == last for site in dependent), f"the mask is not the kernel's last parameter ({last}):\n{listing}"
    # Every consumer thread arrives on the tile-info slot: no lane predicate.
    assert all(site.loaded_registers == 4 and not site.text.startswith("@") for site in dependent), listing


@pytest.mark.parametrize("case", list(CASES))
def test_sm90_tile_info_release_depends_on_the_loaded_words(tmp_path, case):
    cutlass = pytest.importorskip("cutlass")
    from cutlass.cute import runtime
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.typing import AddressSpace
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode
    from cudnn.gemm.cutedsl.grouped.unfused.moe_grouped_gemm_sm90 import MoEGroupedGemmBf16KernelSm90

    direction, tile, cluster, pingpong, sites = CASES[case]
    n, k = (1536, 512) if direction == "fwd" else (512, 1536)

    def fake(dtype, shape, stride, align=16):
        return runtime.make_fake_tensor(dtype, shape, stride, assumed_align=align)

    bf16 = cutlass.BFloat16
    a = fake(bf16, (ROWS, k, 1), (k, 1, ROWS * k))
    b = fake(bf16, (n, k, EXPERTS), (k, 1, n * k)) if direction == "fwd" else fake(bf16, (n, k, EXPERTS), (1, n, n * k))
    c = fake(bf16, (ROWS, n, 1), (n, 1, ROWS * n))
    d = fake(bf16, (ROWS, n, 1), (n, 1, ROWS * n))
    offsets = fake(cutlass.Int32, (EXPERTS,), (1,), align=4)
    alpha = fake(cutlass.Float32, (EXPERTS,), (1,))
    prob = fake(cutlass.Float32, (ROWS, 1, 1), (1, 1, 1))
    workspace = runtime.make_ptr(cutlass.Uint8, 1 << 20, AddressSpace.gmem, assumed_align=128)
    major = OperandMajorMode.K if direction == "fwd" else OperandMajorMode.MN
    kernel = MoEGroupedGemmBf16KernelSm90(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=tile,
        cluster_shape_mn=cluster,
        vectorized_f32=False,
        generate_c=False,
        enable_bias=False,
        expert_cnt=EXPERTS,
        weight_mode=MoEWeightMode.DENSE,
        use_dynamic_sched=True,
        pingpong=pingpong,
    )
    args = [a, b, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int64(0), major, workspace, c, d, offsets, alpha, None, prob]
    ptx = sm90_ptx(kernel, args + [132 // (cluster[0] * cluster[1]), runtime.make_fake_stream()], str(tmp_path))
    assert_tile_info_releases(ptx, sites)
