# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Programmatic dependent launch (PDL) for the SM100 grouped GEMM wgrad kernels.

* ``use_pdl``: the main kernel launches as the programmatic dependent of the kernel before it on the stream and waits for
  that grid before its first global memory access. Rejected whenever the call launches the helper kernel right before the
  main kernel (TensorRagged inputs, discrete output, ``atomic_counter_claim``, every block-scaled call).
* ``launch_pdl_dependents_at_entry``: the main kernel lets its own dependent launch at its entry; allowed everywhere.

The stream neighbours come from ``pdl_dependent_copy`` (``test/python``): a predecessor that releases its dependents first and
only then writes, and a dependent that waits before it copies. The eager launches are warm ones: the first launch of a freshly
compiled kernel returns to the host only when the work in flight on the stream has completed, so it never starts early.
"""

import pytest
import torch

M, N = 512, 512
BF16_GROUPS = (256, 0, 512, 256, 0, 1024)
MXFP8_GROUPS = (128, 0, 384, 1024, 0, 128)
# (mma_tiler_mn, cluster_shape_mn): one-CTA and a (2, 1) pair.
CONFIGS = [((128, 256), (1, 1)), ((256, 256), (2, 1))]


def _config_id(config):
    (tm, tn), (cm, cn) = config
    return f"tile{tm}x{tn}-cluster{cm}x{cn}"


def _require_sm100():
    from cudnn.tensor_adapter import has_tcgen05

    if not torch.cuda.is_available() or not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM wgrad")


# ---------------------------------------------------------------------------
# Option checks (CPU only: the kernel constructors)
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_pdl_options_rejected_behind_helper_kernel():
    """use_pdl is rejected wherever the call launches the helper kernel right before the main kernel; the helper would be
    the main kernel's programmatic primary. launch_pdl_dependents_at_entry alone stays allowed there."""
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode, WGradInputOrder
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad import MoEGroupedGemmWgradBF16Kernel

    bf16 = dict(use_2cta_instrs=False, mma_tiler_mn=(128, 256), cluster_shape_mn=(1, 1), expert_cnt=4)
    helper_configs = [
        dict(weight_mode=MoEWeightMode.DISCRETE),
        dict(input_order=WGradInputOrder.TensorRagged),
        dict(atomic_counter_claim=True),
    ]
    for options in helper_configs:
        with pytest.raises(ValueError, match="helper_kernel"):
            MoEGroupedGemmWgradBF16Kernel(**bf16, **options, use_pdl=True)
        kernel = MoEGroupedGemmWgradBF16Kernel(**bf16, **options, launch_pdl_dependents_at_entry=True)
        assert kernel.launch_pdl_dependents_at_entry and not kernel.use_pdl
    kernel = MoEGroupedGemmWgradBF16Kernel(**bf16, use_pdl=True, launch_pdl_dependents_at_entry=True)
    assert kernel.use_pdl and kernel.launch_pdl_dependents_at_entry

    # The block-scaled call always launches the helper kernel (the scale-factor descriptors), SM100 and SM107.
    for kernel_class in (BlockScaledMoEGroupedGemmWgradKernel, BlockScaledMoEGroupedGemmWgradRubinKernel):
        for weight_mode in (MoEWeightMode.DENSE, MoEWeightMode.DISCRETE):
            scaled = dict(sf_vec_size=32, mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1), expert_cnt=4, weight_mode=weight_mode)
            with pytest.raises(ValueError, match="helper_kernel"):
                kernel_class(**scaled, use_pdl=True)
            assert kernel_class(**scaled, launch_pdl_dependents_at_entry=True).launch_pdl_dependents_at_entry


# ---------------------------------------------------------------------------
# On a device
# ---------------------------------------------------------------------------


def _bf16_problem(*, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(BF16_GROUPS)
    # K-major A (hidden, tokens) and B (tokens, intermediate) with B's token axis contiguous: B.t() is contiguous.
    a = (torch.randn((M, tokens), device="cuda", generator=generator) * 0.125).to(torch.bfloat16)
    b = (torch.randn((N, tokens), device="cuda", generator=generator) * 0.125).to(torch.bfloat16).t()
    offsets = torch.tensor(BF16_GROUPS, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    return {"a": a, "b": b, "offsets": offsets, "output": torch.empty((len(BF16_GROUPS), M, N), dtype=torch.bfloat16, device="cuda")}


def _op(problem, config, **options):
    import cudnn

    tile, cluster = config
    op = cudnn.GroupedGemmWgradSm100(
        sample_a=problem["a"],
        sample_b=problem["b"],
        sample_sfa=problem.get("sfa"),
        sample_sfb=problem.get("sfb"),
        sample_offsets=problem["offsets"],
        sample_wgrad=problem["output"],
        acc_dtype=torch.float32,
        mma_tiler_mn=tile,
        cluster_shape_mn=cluster,
        sf_vec_size=32 if "sfa" in problem else 16,
        **options,
    )
    assert op.check_support()
    op.compile()
    return op


def _execute(op, problem, a, output):
    op.execute(
        a_tensor=a,
        b_tensor=problem["b"],
        sfa_tensor=problem.get("sfa"),
        sfb_tensor=problem.get("sfb"),
        offsets_tensor=problem["offsets"],
        wgrad_tensor=output,
    )


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.uint8)


def _same(x, y):
    return torch.equal(_bits(x), _bits(y))


def _graph(chain):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain()
    return graph


@pytest.mark.L0
def test_pdl_rejected_by_class_api_behind_helper_kernel():
    """The class API forwards use_pdl; with atomic_counter_claim the kernel constructor rejects it at compile()."""
    _require_sm100()
    problem = _bf16_problem(seed=5)
    with pytest.raises(ValueError, match="helper_kernel"):
        _op(problem, CONFIGS[0], use_pdl=True, atomic_counter_claim=True)


@pytest.mark.L0
@pytest.mark.parametrize("config", CONFIGS, ids=_config_id)
@pytest.mark.parametrize("operand", ["a", "accumulated_output"])
def test_use_pdl_matches_plain_launch_behind_producers(config, operand):
    """use_pdl on versus off, bitwise, behind a predecessor that writes an operand: A, or the output that
    accumulate_on_output reduces into. The predecessor is an ordinary torch kernel (it releases its dependents only when it
    completes) or ``EarlyReleaseWriter`` (it releases them first, then writes slowly with one thread block): eager and
    replayed from a CUDA graph. A main kernel that touched the operand in front of its wait would read or reduce into other
    bytes. The positive control (a dependent copy without a wait behind the same writer) shows the early start reproduces."""
    _require_sm100()
    from pdl_dependent_copy import DependentCopy, EarlyReleaseWriter

    problem = _bf16_problem(seed=7)
    accumulate = operand == "accumulated_output"
    new = (torch.randn_like(problem["output"], dtype=torch.float32) * 0.25).to(torch.bfloat16) if accumulate else problem["a"]
    stale = (torch.randn_like(new, dtype=torch.float32) * 0.25).to(new.dtype)
    buf = stale.clone()
    a = buf if not accumulate else problem["a"]

    plain = _op(problem, config, accumulate_on_output=accumulate)
    pdl = _op(problem, config, accumulate_on_output=accumulate, use_pdl=True)
    expected = new.clone() if accumulate else torch.empty_like(problem["output"])
    _execute(plain, problem, problem["a"], expected)
    torch.cuda.synchronize()

    def output_for_run():
        return buf if accumulate else problem["output"]

    writer, unwaited = EarlyReleaseWriter(torch.device("cuda")), DependentCopy(torch.device("cuda"), wait=False)
    seen = torch.empty_like(new)

    def ordinary():
        torch.mul(new, 1, out=buf)  # an ordinary kernel: no early release
        _execute(pdl, problem, a, output_for_run())

    def early():
        writer(new, buf)
        _execute(pdl, problem, a, output_for_run())

    def control():
        writer(new, buf)
        unwaited(buf, seen)

    for chain in (ordinary, early, control):  # warm every launch
        chain()
    torch.cuda.synchronize()

    def runs(run, check, count=3):
        results = []
        for _ in range(count):
            buf.copy_(stale)
            problem["output"].fill_(float("nan"))
            torch.cuda.synchronize()
            run()
            torch.cuda.synchronize()
            results.append(check())
        return results

    def early_start():
        return not _same(seen, new)

    def matches():
        return _same(output_for_run(), expected)

    early_starts = sum(runs(control, early_start))
    differing = []
    for name, chain in (("ordinary", ordinary), ("early", early)):
        ok = runs(chain, matches)
        if not all(ok):
            differing.append(f"{name} producer, eager: {ok}")
    for name, chain in (("ordinary", ordinary), ("early", early), ("control", control)):
        graph = _graph(chain)
        try:
            if name == "control":
                early_starts += sum(runs(graph.replay, early_start))
            else:
                ok = runs(graph.replay, matches)
                if not all(ok):
                    differing.append(f"{name} producer, graph: {ok}")
        finally:
            graph.reset()
    assert not differing, f"use_pdl output differs from the plain launch: {differing}"
    if early_starts == 0:
        pytest.skip("the outputs match, but the early start of a dependent did not reproduce on this device (a coverage gap)")


def _mxfp8_problem():
    from gemm.cutedsl.test_grouped_gemm_wgrad_atomic_counter import _make_mxfp8_problem

    return _make_mxfp8_problem(MXFP8_GROUPS, M, N, seed=9)


@pytest.mark.L0
@pytest.mark.parametrize("config", CONFIGS, ids=_config_id)
@pytest.mark.parametrize("variant", ["bf16", "bf16_counter", "bf16_pdl", "mxfp8"])
def test_launch_dependents_at_entry_with_dependent_copy(config, variant):
    """launch_pdl_dependents_at_entry lets the next launch start at the wgrad kernel's entry: a dependent that waits before
    it copies the wgrad output reads the finished output, eager and replayed from a CUDA graph. Also behind the helper kernel
    (atomic_counter_claim, block-scaled) and together with use_pdl."""
    _require_sm100()
    from pdl_dependent_copy import DependentCopy

    problem = _mxfp8_problem() if variant == "mxfp8" else _bf16_problem(seed=11)
    if variant == "mxfp8" and config == CONFIGS[0]:
        config = ((128, 128), (1, 1))  # the one-CTA MXFP8 tile of the atomic-counter suite
    options = dict(atomic_counter_claim=variant == "bf16_counter", use_pdl=variant == "bf16_pdl")
    plain = _op(problem, config, atomic_counter_claim=options["atomic_counter_claim"])
    entry = _op(problem, config, launch_pdl_dependents_at_entry=True, **options)
    expected = torch.empty_like(problem["output"])
    _execute(plain, problem, problem["a"], expected)
    torch.cuda.synchronize()

    output = problem["output"]
    seen = torch.empty_like(output)
    dependent = DependentCopy(torch.device("cuda"))

    def chain():
        _execute(entry, problem, problem["a"], output)
        dependent(output, seen)

    chain()  # warm
    torch.cuda.synchronize()
    results = []
    graph = _graph(chain)
    try:
        for run in (chain, chain, graph.replay, graph.replay):
            output.fill_(float("nan"))
            seen.fill_(float("nan"))
            torch.cuda.synchronize()
            run()
            torch.cuda.synchronize()
            results.append((_same(output, expected), _same(seen, expected)))
    finally:
        graph.reset()
    assert all(o and s for o, s in results), f"(output, dependent copy) bitwise equal per run: {results}"
