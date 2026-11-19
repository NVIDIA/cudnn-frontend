# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage reuse and paired-CTA SFB handoff for canonical NVFP4 Hadamard fprop."""

import importlib

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays, snapshot
from gemm.cutedsl.test_grouped_gemm_glu_branch_layout import canonicalize_c, make_problem, snapshot_inputs
from gemm.cutedsl.test_grouped_gemm_glu_hadamard_layout import invoke, prepare, reference

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Canonical Hadamard requires SM100/103")


@pytest.mark.parametrize("discrete,tmem", [(False, False), (True, False), (False, True)], ids=["dense-smem", "discrete-smem", "dense-tmem"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("k", [1792, 4096], ids=["K1792", "K4096"])
def test_hadamard_sfb_stage_reuse(discrete, tmem, dynamic, k):
    # Different K lengths exercise different ring phases at expert/tile changes.
    # Nonuniform scales distinguish stale stages and missing gate/up quarters.
    problem = make_problem(f=256, k=k, counts=(256, 0, 768), precision="nvfp4")
    original = snapshot_inputs(problem)
    args, owners = prepare(problem, discrete=discrete, tmem=tmem)
    args["use_dynamic_sched"] = dynamic
    legacy_args, legacy_owners = prepare(problem, layout="interleaved_32", discrete=discrete, tmem=tmem)
    legacy_args["use_dynamic_sched"] = dynamic
    expected = invoke(legacy_args)
    expected["c_tensor"] = canonicalize_c(expected["c_tensor"])
    actual = assert_replays(lambda: invoke(args), repeats=4)
    assert_bitwise(actual, expected)
    reference(problem, actual)
    assert_bitwise(snapshot_inputs(problem), original, "input bytes")


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("delayed_cta", [0, 1], ids=["leader-late", "follower-late"])
def test_hadamard_sfb_asymmetric_arrival(monkeypatch, discrete, delayed_cta):
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import dsl_user_op

    module = importlib.import_module("cudnn.gemm.cutedsl.grouped.glu_hadamard.moe_blockscaled_grouped_gemm_glu_hadamard")
    api = importlib.import_module("cudnn.gemm.cutedsl.grouped.glu_hadamard.api")
    original_interleave = module.interleave_sfb_in_place

    @dsl_user_op
    def pause(*, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [],
            "nanosleep.u32 20000;",
            "~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @cute.jit
    def delayed_interleave(
        scales: cute.Tensor, stage: cutlass.Int32, lane: cutlass.Int32, tile_k: cutlass.Constexpr, sf_vec_size: cutlass.Constexpr, stages: cutlass.Constexpr
    ):
        if cute.arch.block_idx_in_cluster() == delayed_cta:
            pause()
        original_interleave(scales, stage, lane, tile_k, sf_vec_size, stages)

    problem = make_problem(f=256, k=4096, counts=(256, 0, 768), precision="nvfp4")
    args, owners = prepare(problem, discrete=discrete)
    expected = snapshot(invoke(args))
    reference(problem, expected)
    # Isolate wrapper caches so a precompiled, uninstrumented kernel cannot
    # satisfy this test, and delayed kernels cannot escape into later cases.
    monkeypatch.setattr(api, "_cache_of_GroupedGemmGluHadamardSm100Objects", {})
    monkeypatch.setattr(module, "interleave_sfb_in_place", delayed_interleave)
    actual = assert_replays(lambda: invoke(args), repeats=4)
    assert_bitwise(actual, expected, "asymmetric CTA arrival")


@pytest.mark.parametrize("discrete,tmem", [(False, False), (True, False), (False, True)], ids=["dense-smem", "discrete-smem", "dense-tmem"])
def test_hadamard_sfb_warm_stress(monkeypatch, discrete, tmem):
    import cutlass
    from warm_stress import StressCase, assert_warm_stress

    monkeypatch.delenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", raising=False)
    # K=4096 refills the AB/SFB ring; the M/N grid also spans persistent tiles.
    problem = make_problem(f=1024, k=4096, counts=(1024, 0, 2048), precision="nvfp4")
    args, owners = prepare(problem, discrete=discrete, tmem=tmem)
    actual = invoke(args)
    reference(problem, actual)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = invoke(args)
    clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
    report = assert_warm_stress(
        StressCase(
            f"canonical Hadamard SFB M=[1024,0,2048] N=2048 K=4096 discrete={discrete} tmem={tmem}",
            launch=graph.replay,
            outputs=captured,
            thread_blocks=2 * clusters,
            refills_per_launch=1,
        )
    )
    print(report.text())
