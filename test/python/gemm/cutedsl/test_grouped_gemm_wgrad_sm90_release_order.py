# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Release-order check of the SM90 BF16 grouped GEMM wgrad (cooperative and pingpong), compile-only.

Every consumer of a tile-info slot (TMA warp, MMA warp groups) loads the four words with one ``ld.shared.v4`` and
releases the slot through ``tile_dsl.consume``: the arrive's address is computed from the four loaded registers through
an ``and`` with the kernel's last parameter, the release-zero argument, so every loaded word is consumed before the slot
is released. The kernel is compiled for sm_90a from fake tensors, so the check needs no H100. The disassembly proof is in the PR text.
"""

from __future__ import annotations

import glob
import os
import re

import pytest

from warm_stress import assert_release_dependence

pytestmark = pytest.mark.L0

EXPERTS, TOKENS = 32, 65536

# id: (tile, cluster, pingpong, tile-info arrives in the rendering)
CASES = {
    "cooperative": ((128, 256), (2, 1), False, 4),
    "pingpong": ((128, 192), (2, 1), True, 4),
    "cooperative-128x128-c11": ((128, 128), (1, 1), False, 4),
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
def test_sm90_wgrad_tile_info_release_depends_on_the_loaded_words(tmp_path, case):
    cutlass = pytest.importorskip("cutlass")
    import cutlass.cute as cute
    from cutlass.cute import runtime
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode, WGradInputOrder
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad_sm90 import MoEGroupedGemmWgradBF16KernelSm90

    tile, cluster, pingpong, sites = CASES[case]
    m, n = 2048, 1536
    bf16 = cutlass.BFloat16

    def compact(dtype, shape, order, align=16, dynamic=None):
        if dynamic is not None:
            shape = shape[:dynamic] + (cute.sym_int(divisibility=256),) + shape[dynamic + 1 :]
        return runtime.make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=order, assumed_align=align)

    a = compact(bf16, (m, TOKENS), (0, 1), dynamic=1)
    b = compact(bf16, (TOKENS, n), (1, 0), dynamic=0)
    out = compact(bf16, (EXPERTS, m, n), (2, 1, 0))
    offsets = runtime.make_fake_tensor(cutlass.Int32, (EXPERTS,), (1,), assumed_align=4)
    workspace = runtime.make_fake_tensor(cutlass.Uint8, (128,), (1,), assumed_align=128)
    kernel = MoEGroupedGemmWgradBF16KernelSm90(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=tile,
        cluster_shape_mn=cluster,
        accumulate_on_output=False,
        expert_cnt=EXPERTS,
        weight_mode=MoEWeightMode.DENSE,
        input_order=WGradInputOrder.Tensor2D,
        **({"pingpong": True} if pingpong else {}),
    )
    args = [a, b, out, offsets, workspace, 132 // (cluster[0] * cluster[1]), runtime.make_fake_stream(), None]
    ptx = sm90_ptx(kernel, args, str(tmp_path))

    found = assert_release_dependence(ptx, sites=sites, words=4)
    dependent = [site for site in found if site.dependent]
    listing = "\n".join(f"  line {s.line}: registers={s.loaded_registers} param={s.parameter} :: {s.text}" for s in dependent)
    entry = next(name for name in re.findall(r"\.entry\s+(\w+)", ptx) if "helper_kernel" not in name)
    indices = [int(i) for i in re.findall(r"\b" + re.escape(entry) + r"_param_(\d+)\b", ptx)]
    last = f"{entry}_param_{max(indices)}"
    assert all(site.parameter == last for site in dependent), f"the mask is not the kernel's last parameter ({last}):\n{listing}"
    # Every consumer thread arrives on the tile-info slot: no lane predicate.
    assert all(site.loaded_registers == 4 and not site.text.startswith("@") for site in dependent), listing
