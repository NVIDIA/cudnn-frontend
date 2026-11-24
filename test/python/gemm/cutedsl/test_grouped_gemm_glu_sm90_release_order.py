# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Release-order check of the SM90 BF16 grouped GEMM GLU (cooperative and pingpong, with C) and dGLU, compile-only.

Values loaded from a slot must be consumed before the slot is released. Tile-info slots: every consumer loads the four
words with one ``ld.shared.v4`` and releases the slot through ``tile_dsl.consume`` on every lane, the arrive's address
computed from the four loaded registers through an ``and`` with the kernel's last parameter, the release-zero. The GLU
inherits these sites from the forward kernel. dGLU's C stage: every epilogue warp loads its C subtile with two ``ldmatrix.x4`` (8 words per lane) and
its lane 0 releases the stage with the predicated arrive of ``consume_then_arrive(..., pred=)``, computed from all 8
words. The kernels are compiled for sm_90a from fake tensors, so the check needs no H100. The disassembly proof is in the
PR text.
"""

from __future__ import annotations

import glob
import os
import re

import pytest

from warm_stress import assert_release_dependence

pytestmark = pytest.mark.L0

EXPERTS, ROWS = 8, 4096

# id: (op, tile, cluster, pingpong, tile-info arrives, C-stage arrives, glu_layout) in the rendering
CASES = {
    "glu-cooperative": ("glu", (128, 256), (2, 1), False, 4, 0, "interleaved_32"),
    "glu-pingpong": ("glu", (128, 192), (2, 1), True, 6, 0, "interleaved_32"),
    "dglu-128x256-c21": ("dglu", (128, 256), (2, 1), False, 6, 16, "interleaved_32"),
    "dglu-128x128-c11": ("dglu", (128, 128), (1, 1), False, 6, 8, "interleaved_32"),
    "glu-cooperative-gate_then_up": ("glu", (128, 256), (2, 1), False, 4, 0, "gate_then_up"),
    "glu-pingpong-gate_then_up": ("glu", (128, 192), (2, 1), True, 6, 0, "gate_then_up"),
    "dglu-128x256-c21-gate_then_up": ("dglu", (128, 256), (2, 1), False, 6, 16, "gate_then_up"),
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


@pytest.mark.parametrize("case", list(CASES))
def test_sm90_glu_releases_depend_on_the_loaded_words(tmp_path, case):
    cutlass = pytest.importorskip("cutlass")
    from cutlass.cute import runtime
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.typing import AddressSpace
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    op, tile, cluster, pingpong, tile_info_sites, c_sites, glu_layout = CASES[case]
    n, k = 1536, 512

    def fake(dtype, shape, stride, align=16):
        return runtime.make_fake_tensor(dtype, shape, stride, assumed_align=align)

    bf16 = cutlass.BFloat16
    a = fake(bf16, (ROWS, k, 1), (k, 1, ROWS * k))
    b = fake(bf16, (n, k, EXPERTS), (k, 1, n * k))
    offsets = fake(cutlass.Int32, (EXPERTS,), (1,), align=4)
    alpha = fake(cutlass.Float32, (EXPERTS,), (1,))
    prob = fake(cutlass.Float32, (ROWS, 1, 1), (1, 1, 1))
    workspace = runtime.make_ptr(cutlass.Uint8, 1 << 20, AddressSpace.gmem, assumed_align=128)
    head = [a, b, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int64(0), OperandMajorMode.K, workspace]
    clusters = 132 // (cluster[0] * cluster[1])
    common = dict(acc_dtype=cutlass.Float32, use_2cta_instrs=False, mma_tiler_mn=tile, cluster_shape_mn=cluster, vectorized_f32=False)
    common.update(expert_cnt=EXPERTS, weight_mode=MoEWeightMode.DENSE, use_dynamic_sched=True)
    if op == "glu":
        from cudnn.gemm.cutedsl.grouped.glu.moe_grouped_gemm_glu_sm90 import MoEGroupedGemmGluBf16KernelSm90

        kernel = MoEGroupedGemmGluBf16KernelSm90(**common, generate_c=True, enable_bias=False, act_func="swiglu", pingpong=pingpong, glu_layout=glu_layout)
        c = fake(bf16, (ROWS, n, 1), (n, 1, ROWS * n))
        d = fake(bf16, (ROWS, n // 2, 1), (n // 2, 1, ROWS * n // 2))
        args = head + [c, d, offsets, alpha, prob, None, clusters, runtime.make_fake_stream(), cutlass.Float32(0.5)]
    else:
        from cudnn.gemm.cutedsl.grouped.dglu.moe_grouped_gemm_dglu_sm90 import MoEGroupedGemmDgluBf16KernelSm90

        kernel = MoEGroupedGemmDgluBf16KernelSm90(**common, act_func="dswiglu", round_dgrad_to_input_dtype=False, glu_layout=glu_layout)
        c = fake(bf16, (ROWS, 2 * n, 1), (2 * n, 1, ROWS * 2 * n))
        d = fake(bf16, (ROWS, 2 * n, 1), (2 * n, 1, ROWS * 2 * n))
        beta = fake(cutlass.Float32, (EXPERTS,), (1,))
        dprob = fake(cutlass.Float32, (ROWS, 1, 1), (1, 1, 1))
        args = head + [c, d, offsets, alpha, beta, prob, dprob, cutlass.Float32(0.5), None, clusters, runtime.make_fake_stream()]
    ptx = sm90_ptx(kernel, args, str(tmp_path))

    found = assert_release_dependence(ptx, sites=tile_info_sites + c_sites, words=4)
    dependent = [site for site in found if site.dependent]
    listing = "\n".join(f"  line {s.line}: registers={s.loaded_registers} param={s.parameter} :: {s.text}" for s in dependent)
    entry = next(name for name in re.findall(r"\.entry\s+(\w+)", ptx) if "helper_kernel" not in name)
    indices = [int(i) for i in re.findall(r"\b" + re.escape(entry) + r"_param_(\d+)\b", ptx)]
    last = f"{entry}_param_{max(indices)}"
    assert all(site.parameter == last for site in dependent), f"the mask is not the kernel's last parameter ({last}):\n{listing}"
    # Tile-info: every consumer lane arrives, from its 4 loaded words. C stage: one predicated lane per epilogue warp,
    # from the 8 words of its two ldmatrix.x4.
    tile_info = [site for site in dependent if not site.text.startswith("@")]
    c_stage = [site for site in dependent if site.text.startswith("@")]
    assert len(tile_info) == tile_info_sites and all(site.loaded_registers == 4 for site in tile_info), listing
    assert len(c_stage) == c_sites and all(site.loaded_registers == 8 and site.loads == 2 for site in c_stage), listing
