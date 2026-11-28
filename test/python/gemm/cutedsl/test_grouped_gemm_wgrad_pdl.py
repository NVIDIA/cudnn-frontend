# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Programmatic dependent launch (PDL) for the SM100 grouped GEMM wgrad kernels.

* ``use_pdl``: the main kernel launches as the programmatic dependent of the kernel before it on the stream and waits for
  that grid before its first global memory access. When the call launches the helper kernel (every block-scaled call; BF16
  with TensorRagged inputs, discrete output or ``atomic_counter_claim``), the helper launches the same way and waits for its
  predecessor at its start, so the chain is: producer, helper, main kernel.
* ``launch_pdl_dependents_at_entry``: the main kernel lets its own dependent launch at its entry.
* ``pdl_mode`` (MXFP8): ``"wait_at_start"`` is the ``use_pdl`` placement above; ``"wait_then_trigger"`` waits, then lets the
  dependent launch; ``"trigger_then_wait_at_exit"`` lets it launch at entry and waits only at the end, so the caller
  guarantees that no input comes from the grid right before it.

MXFP8 with dense output at 2-CTA 256x256 tiles in a (2, 1) cluster runs the global-descriptor variant: no helper kernel
(only the counter reset with ``atomic_counter_claim``), and for an expert whose token count is not a multiple of the K tile
the load warp builds that expert's descriptors in the workspace, after its wait.

The stream neighbours come from ``pdl_dependent_copy`` (``test/python``): a predecessor that releases its dependents first and
only then writes, and a dependent that waits before it copies. The eager launches are warm ones: the first launch of a freshly
compiled kernel returns to the host only when the work in flight on the stream has completed, so it never starts early.
"""

import pytest
import torch

M, N = 512, 512
BF16_GROUPS = (256, 0, 512, 256, 0, 1024)
MXFP8_GROUPS = (128, 0, 384, 1024, 0, 128)
# Other valid group sizes with the same total: what a kernel that read the offsets in front of its wait would see.
# Experts that end inside a K tile (the global-descriptor variant builds their descriptors in the kernel), long enough on
# average that the API picks that variant for them.
MXFP8_PARTIAL_GROUPS = (1120, 0, 1696, 1376, 0, 736)
STALE_GROUPS = {"bf16": (1024, 256, 0, 512, 0, 256), "mxfp8": (1024, 128, 0, 384, 128, 0), "mxfp8_partial": (1696, 1120, 0, 736, 1376, 0)}
# The offsets operand sits at the end of a buffer of this many int32: a one-pass writer reaches it last.
OFFSETS_PREFIX = 1 << 20
# (mma_tiler_mn, cluster_shape_mn): one-CTA and a (2, 1) pair. MXFP8 takes the 128x128 tile for the one-CTA case.
CONFIGS = [((128, 256), (1, 1)), ((256, 256), (2, 1))]
# Without a helper kernel (bf16), and every way the call launches one in front of the main kernel.
VARIANTS = ["bf16", "bf16_ragged", "bf16_discrete", "bf16_counter", "mxfp8", "mxfp8_discrete", "mxfp8_counter"]


def _config_id(config):
    (tm, tn), (cm, cn) = config
    return f"tile{tm}x{tn}-cluster{cm}x{cn}"


def _require_sm100():
    from cudnn.tensor_adapter import has_tcgen05

    if not torch.cuda.is_available() or not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM wgrad")


# ---------------------------------------------------------------------------
# Options (CPU only: the kernel constructors)
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_pdl_options_accepted_by_every_kernel():
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode, WGradInputOrder
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad import MoEGroupedGemmWgradBF16Kernel

    pdl = dict(use_pdl=True, launch_pdl_dependents_at_entry=True)
    for options in ({}, dict(weight_mode=MoEWeightMode.DISCRETE), dict(input_order=WGradInputOrder.TensorRagged), dict(atomic_counter_claim=True)):
        kernel = MoEGroupedGemmWgradBF16Kernel(mma_tiler_mn=(128, 256), expert_cnt=4, **options, **pdl)
        assert kernel.use_pdl and kernel.launch_pdl_dependents_at_entry
        assert not MoEGroupedGemmWgradBF16Kernel(mma_tiler_mn=(128, 256), expert_cnt=4, **options).use_pdl
    for kernel_class in (BlockScaledMoEGroupedGemmWgradKernel, BlockScaledMoEGroupedGemmWgradRubinKernel):
        for weight_mode in (MoEWeightMode.DENSE, MoEWeightMode.DISCRETE):
            kernel = kernel_class(32, mma_tiler_mn=(128, 128), expert_cnt=4, weight_mode=weight_mode, **pdl)
            assert kernel.use_pdl and kernel.launch_pdl_dependents_at_entry


def _mxfp8_kernel_classes():
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense import BlockScaledMoEGroupedGemmWgradDenseKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense_rubin import BlockScaledMoEGroupedGemmWgradDenseRubinKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse import BlockScaledMoEGroupedGemmWgradReuseKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse_rubin import BlockScaledMoEGroupedGemmWgradReuseRubinKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel

    stock = (BlockScaledMoEGroupedGemmWgradKernel, BlockScaledMoEGroupedGemmWgradRubinKernel)
    descriptor_free = (
        BlockScaledMoEGroupedGemmWgradDenseKernel,
        BlockScaledMoEGroupedGemmWgradReuseKernel,
        BlockScaledMoEGroupedGemmWgradDenseRubinKernel,
        BlockScaledMoEGroupedGemmWgradReuseRubinKernel,
    )
    return stock, descriptor_free


@pytest.mark.L0
def test_pdl_mode_kernel_rejections():
    """pdl_mode on the MXFP8 kernel classes (SM100/SM103 and their Rubin subclasses): a known value; a non-default mode
    needs use_pdl and excludes launch_pdl_dependents_at_entry; trigger_then_wait_at_exit is refused by the stock kernels,
    whose descriptor helper always launches in front of the GEMM."""
    import cutlass

    stock, descriptor_free = _mxfp8_kernel_classes()
    common = dict(sf_vec_size=32, acc_dtype=cutlass.Float32, use_2cta_instrs=True, mma_tiler_mn=(256, 256), cluster_shape_mn=(2, 1), expert_cnt=4)
    for cls in stock + descriptor_free:
        assert cls(**common).pdl_mode == "wait_at_start"
        assert cls(**common, use_pdl=True, launch_pdl_dependents_at_entry=True).pdl_mode == "wait_at_start"
        with pytest.raises(ValueError, match="pdl_mode must be one of"):
            cls(**common, use_pdl=True, pdl_mode="exit")
        for mode in ("wait_then_trigger", "trigger_then_wait_at_exit"):
            with pytest.raises(ValueError, match="requires use_pdl"):
                cls(**common, pdl_mode=mode)
            with pytest.raises(ValueError, match="requires use_pdl"):
                cls(**common, use_pdl=True, launch_pdl_dependents_at_entry=True, pdl_mode=mode)
        assert cls(**common, use_pdl=True, pdl_mode="wait_then_trigger").pdl_mode == "wait_then_trigger"
    for cls in stock:
        with pytest.raises(ValueError, match="helper kernel"):
            cls(**common, use_pdl=True, pdl_mode="trigger_then_wait_at_exit")
    for cls in descriptor_free:
        assert cls(**common, use_pdl=True, pdl_mode="trigger_then_wait_at_exit").pdl_mode == "trigger_then_wait_at_exit"


def _cpu_mxfp8_sample(experts=4, m=256, n=256, tokens=1024):
    cols = -(-(-(-tokens // 32)) // 4) * 4
    return dict(
        sample_a=torch.zeros((m, tokens), dtype=torch.float8_e4m3fn),
        sample_b=torch.zeros((n, tokens), dtype=torch.float8_e4m3fn).t(),
        sample_sfa=torch.zeros((m, cols), dtype=torch.float8_e8m0fnu),
        sample_sfb=torch.zeros((n, cols), dtype=torch.float8_e8m0fnu),
        sample_offsets=torch.arange(1, experts + 1, dtype=torch.int32) * (tokens // experts),
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
    )


@pytest.mark.L0
def test_pdl_mode_and_scheduler_counter_api_rejections():
    """The class API refuses what the kernels would: at construction, a bad pdl_mode and a caller counter without
    atomic_counter_claim; when the backend is chosen (check_support), the BF16 backend, trigger_then_wait_at_exit where the
    stock kernel is selected (discrete output, accumulate_on_output), and trigger_then_wait_at_exit with
    atomic_counter_claim but no caller counter. No device needed."""
    import cudnn

    sample = _cpu_mxfp8_sample()
    dense = dict(sample, sample_wgrad=torch.empty((4, 256, 256), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="pdl_mode must be one of"):
        cudnn.GroupedGemmWgradSm100(**dense, use_pdl=True, pdl_mode="later")
    for mode in ("wait_then_trigger", "trigger_then_wait_at_exit"):
        with pytest.raises(ValueError, match="requires use_pdl"):
            cudnn.GroupedGemmWgradSm100(**dense, pdl_mode=mode)
        with pytest.raises(ValueError, match="requires use_pdl"):
            cudnn.GroupedGemmWgradSm100(**dense, use_pdl=True, launch_pdl_dependents_at_entry=True, pdl_mode=mode)
    with pytest.raises(ValueError, match="requires atomic_counter_claim"):
        cudnn.GroupedGemmWgradSm100(**dense, sample_scheduler_counter=torch.zeros(1, dtype=torch.int32))

    exit_mode = dict(use_pdl=True, pdl_mode="trigger_then_wait_at_exit")
    discrete = dict(
        sample, sample_wgrad_expert=torch.empty((256, 256), dtype=torch.bfloat16), num_experts=4, wgrad_shape=(256, 256), wgrad_dtype=torch.bfloat16
    )
    for kwargs, match in (
        (dict(dense, atomic_counter_claim=True, **exit_mode), "needs sample_scheduler_counter"),
        (dict(discrete, **exit_mode), "selects the stock kernel"),
        (dict(dense, accumulate_on_output=True, **exit_mode), "selects the stock kernel"),
    ):
        with pytest.raises(ValueError, match=match):
            cudnn.GroupedGemmWgradSm100(**kwargs).check_support()

    bf16 = dict(
        sample_a=torch.zeros((256, 1024), dtype=torch.bfloat16),
        sample_b=torch.zeros((1024, 256), dtype=torch.bfloat16),
        sample_sfa=None,
        sample_sfb=None,
        sample_offsets=torch.arange(1, 5, dtype=torch.int32) * 256,
        sample_wgrad=torch.empty((4, 256, 256), dtype=torch.bfloat16),
    )
    with pytest.raises(ValueError, match="require the block-scaled WGrad backend"):
        cudnn.GroupedGemmWgradSm100(**bf16, use_pdl=True, pdl_mode="wait_then_trigger").check_support()


# ---------------------------------------------------------------------------
# On a device
# ---------------------------------------------------------------------------


def _partial_mxfp8_problem(groups, m, n, *, seed):
    """MXFP8 operands for experts with partial K tiles; random scale-factor bytes of the size the API checks (the output
    is compared between two launches of the same kernel, not with a reference)."""
    from gemm.cutedsl.test_grouped_gemm_wgrad_utils import _wgrad_create_scale_tensor

    torch.manual_seed(seed)
    tokens = sum(groups)
    cols = -(-(-(-tokens // 32)) // 4) * 4
    return {
        "a": torch.randint(-1, 2, (m, tokens), device="cuda").to(torch.float8_e4m3fn),
        "b": torch.randint(-1, 2, (n, tokens), device="cuda").to(torch.float8_e4m3fn).t(),
        "sfa": _wgrad_create_scale_tensor((m, cols), torch.float8_e8m0fnu),
        "sfb": _wgrad_create_scale_tensor((n, cols), torch.float8_e8m0fnu),
        "offsets": torch.tensor(groups, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32),
        "output": torch.empty((len(groups), m, n), dtype=torch.bfloat16, device="cuda"),
    }


def _stale_key(variant):
    return "mxfp8_partial" if variant.startswith("mxfp8_partial") else "mxfp8" if variant.startswith("mxfp8") else "bf16"


def _problem(variant):
    if variant.startswith("mxfp8_partial"):
        return _partial_mxfp8_problem(MXFP8_PARTIAL_GROUPS, M, N, seed=9)
    if variant.startswith("mxfp8"):
        from gemm.cutedsl.test_grouped_gemm_wgrad_atomic_counter import _make_mxfp8_problem

        return _make_mxfp8_problem(MXFP8_GROUPS, M, N, seed=9)
    from gemm.cutedsl.test_grouped_gemm_wgrad_bf16_utils import make_grouped_gemm_wgrad_bf16_problem

    order = "tensor_ragged" if variant == "bf16_ragged" else "tensor2d"
    return make_grouped_gemm_wgrad_bf16_problem(group_k_list=BF16_GROUPS, m=M, n=N, input_order=order)


def _variant_config(variant, config):
    return ((128, 128), (1, 1)) if variant.startswith("mxfp8") and config == CONFIGS[0] else config


class _Op:
    """A compiled ``GroupedGemmWgradSm100`` of ``variant`` and its launch on caller buffers: ``out`` is the dense output, or
    for discrete output the per-expert buffers behind a fixed pointer array (no per-call pointer upload on the stream)."""

    def __init__(self, problem, variant, config, **options):
        import cudnn

        tile, cluster = _variant_config(variant, config)
        self.problem, self.discrete = problem, variant.endswith("discrete")
        output = problem["output"]
        kwargs = dict(
            sample_a=problem["a"],
            sample_b=problem["b"],
            sample_sfa=problem.get("sfa"),
            sample_sfb=problem.get("sfb"),
            sample_offsets=problem["offsets"],
            acc_dtype=torch.float32,
            mma_tiler_mn=tile,
            cluster_shape_mn=cluster,
            sf_vec_size=32 if "sfa" in problem else 16,
            atomic_counter_claim=variant.endswith("counter"),
            **options,
        )
        if self.discrete:
            kwargs.update(sample_wgrad_expert=output[0], num_experts=output.shape[0], wgrad_shape=tuple(output.shape[1:]), wgrad_dtype=output.dtype)
        else:
            kwargs["sample_wgrad"] = output
        self.op = cudnn.GroupedGemmWgradSm100(**kwargs)
        assert self.op.check_support()
        self.op.compile()
        self._ptrs = {}
        select = getattr(self.op._implementation, "_select_kernel", None)
        self.kernel_class = select() if select is not None else self.op._kernel

    def __call__(self, a, out, offsets=None, counter=None):
        if self.discrete and out.data_ptr() not in self._ptrs:
            self._ptrs[out.data_ptr()] = torch.tensor([e.data_ptr() for e in out], dtype=torch.int64, device="cuda")
        self.op.execute(
            a_tensor=a,
            b_tensor=self.problem["b"],
            sfa_tensor=self.problem.get("sfa"),
            sfb_tensor=self.problem.get("sfb"),
            offsets_tensor=self.problem["offsets"] if offsets is None else offsets,
            wgrad_tensor=None if self.discrete else out,
            wgrad_ptrs=self._ptrs.get(out.data_ptr()) if self.discrete else None,
            scheduler_counter_tensor=counter,
        )


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.uint8)


def _same(x, y):
    return torch.equal(_bits(x), _bits(y))


def _flat_storage(t):
    """A contiguous 1-D tensor over the whole storage of ``t`` (a ragged view's padding included)."""
    numel = t.untyped_storage().nbytes() // t.element_size()
    return torch.empty(0, dtype=t.dtype, device=t.device).set_(t.untyped_storage(), 0, (numel,), (1,))


def _random_like(t, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, 256, (t.numel() * t.element_size(),), dtype=torch.uint8, device="cuda", generator=g).view(t.dtype).view(t.shape)


def _check_global_descriptor_path(op, variant, config, *, exact=False):
    """MXFP8 at 2-CTA 256x256 tiles in a (2, 1) cluster with dense output and no accumulation runs the global-descriptor
    variant (on Rubin its Rubin subclass; or, with ``exact=False``, a variant built on it)."""
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense import BlockScaledMoEGroupedGemmWgradDenseKernel as Dense

    if variant.startswith("mxfp8") and not variant.endswith("discrete") and config == CONFIGS[1] and torch.cuda.get_device_capability()[0] == 10:
        from cudnn.api_base import get_device_type

        if op.op._implementation.accumulate_on_output:
            return
        picked = op.kernel_class
        if get_device_type() == "rubin":
            from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense_rubin import BlockScaledMoEGroupedGemmWgradDenseRubinKernel
            from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel

            ok = (
                (picked is BlockScaledMoEGroupedGemmWgradDenseRubinKernel)
                if exact
                else issubclass(picked, Dense) and issubclass(picked, BlockScaledMoEGroupedGemmWgradRubinKernel)
            )
        else:
            ok = (picked is Dense) if exact else issubclass(picked, Dense)
        assert ok, f"{variant}: the API picked {picked.__name__}"


def _graph(chain):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain()
    return graph


PRODUCER_CASES = (
    [(v, "a") for v in VARIANTS]
    + [("bf16", "accumulated_output"), ("mxfp8", "accumulated_output")]
    + [("bf16", "offsets"), ("bf16_ragged", "offsets"), ("mxfp8", "offsets")]
)


@pytest.mark.L0
@pytest.mark.parametrize("config", CONFIGS, ids=_config_id)
@pytest.mark.parametrize("variant,operand", PRODUCER_CASES, ids=[f"{v}-{o}" for v, o in PRODUCER_CASES])
def test_use_pdl_matches_plain_launch_behind_producers(variant, operand, config):
    """use_pdl on versus off, bitwise, behind a predecessor that writes an operand: A (its whole storage), the output that
    accumulate_on_output reduces into, or the offsets (which the helper kernel reads to build the per-expert descriptors of
    TensorRagged inputs and of the scale factors; they hold other valid offsets until the writer reaches them). The predecessor is an ordinary torch kernel (it releases its dependents only when it
    completes) or ``EarlyReleaseWriter`` (it releases them first, then writes slowly with one thread block): eager and
    replayed from a CUDA graph. With a helper kernel, the helper is the writer's dependent and the main kernel the helper's;
    a helper or main kernel that read in front of its wait would read other bytes. The positive control (a dependent copy
    without a wait behind the same writer) shows the early start reproduces."""
    _require_sm100()
    _check_behind_producers(variant, operand, config)


@pytest.mark.L0
@pytest.mark.parametrize("operand", ["a", "offsets"])
@pytest.mark.parametrize("variant", ["mxfp8_partial", "mxfp8_partial_counter"])
def test_use_pdl_global_descriptor_partial_k_tiles_behind_producers(variant, operand):
    """As above for the MXFP8 global-descriptor variant (on Rubin its Rubin subclass) with experts that end inside a K
    tile: its load warp builds those experts' descriptors from the offsets and A in the workspace, which must happen
    after its wait."""
    _require_sm100()
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("The global-descriptor wgrad is SM100/SM103/SM107 only")
    _check_behind_producers(variant, operand, CONFIGS[1], exact=True)


def _check_behind_producers(variant, operand, config, exact=False):
    from pdl_dependent_copy import DependentCopy, EarlyReleaseWriter

    problem = _problem(variant)
    accumulate = operand == "accumulated_output"
    plain = _Op(problem, variant, config, accumulate_on_output=accumulate)
    pdl = _Op(problem, variant, config, accumulate_on_output=accumulate, use_pdl=True)
    for op in (plain, pdl):
        _check_global_descriptor_path(op, variant, config, exact=exact)

    out = torch.empty_like(problem["output"])
    offsets = None
    if operand == "offsets":
        stale_offsets = torch.tensor(STALE_GROUPS[_stale_key(variant)], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
        new = torch.zeros(OFFSETS_PREFIX + stale_offsets.numel(), dtype=torch.int32, device="cuda")
        new[OFFSETS_PREFIX:] = problem["offsets"]
        stale = new.clone()
        stale[OFFSETS_PREFIX:] = stale_offsets
        buf, a = stale.clone(), problem["a"]
        offsets = buf[OFFSETS_PREFIX:]
        expected = torch.empty_like(out)
        plain(problem["a"], expected)
    elif accumulate:
        new = (torch.randn_like(out, dtype=torch.float32) * 0.25).to(out.dtype)
        buf, a = out, problem["a"]
        expected = new.clone()
        plain(problem["a"], expected)
    else:
        new = _flat_storage(problem["a"]).clone()
        buf = _random_like(new, seed=3)
        a = torch.as_strided(buf, problem["a"].size(), problem["a"].stride(), problem["a"].storage_offset())
        expected = torch.empty_like(out)
        plain(problem["a"], expected)
    if operand != "offsets":
        stale = _random_like(new, seed=5)
    torch.cuda.synchronize()

    # One pass for the offsets: every byte the writer stores is then the new one, and the offsets are stored last.
    writer = EarlyReleaseWriter(torch.device("cuda"), passes=1 if operand == "offsets" else 8)
    unwaited = DependentCopy(torch.device("cuda"), wait=False)
    seen = torch.empty_like(new)
    new_bytes, buf_bytes = new.view(-1).view(torch.uint8), buf.view(-1).view(torch.uint8)

    def ordinary():
        torch.bitwise_or(new_bytes, 0, out=buf_bytes)  # an ordinary kernel: no early release
        pdl(a, out, offsets)

    def early():
        writer(new, buf)
        pdl(a, out, offsets)

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
            if operand != "accumulated_output":
                out.fill_(float("nan"))
            torch.cuda.synchronize()
            run()
            torch.cuda.synchronize()
            results.append(check())
        return results

    def early_start():
        return not _same(seen, new)

    def matches():
        return _same(out, expected)

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


# Partial K tiles only at the (2, 1) pair: the global-descriptor variant's in-kernel descriptor build.
ENTRY_CASES = [(c, v) for v in ("bf16", "bf16_counter", "bf16_pdl", "mxfp8", "mxfp8_pdl") for c in CONFIGS] + [
    (CONFIGS[1], v) for v in ("mxfp8_partial", "mxfp8_partial_pdl")
]


@pytest.mark.L0
@pytest.mark.parametrize("config,variant", ENTRY_CASES, ids=[f"{_config_id(c)}-{v}" for c, v in ENTRY_CASES])
def test_launch_dependents_at_entry_with_dependent_copy(config, variant):
    """launch_pdl_dependents_at_entry lets the next launch start at the wgrad kernel's entry: a dependent that waits before
    it copies the wgrad output reads the finished output, eager and replayed from a CUDA graph. Also behind the helper kernel
    (atomic_counter_claim, block-scaled) and together with use_pdl."""
    _require_sm100()
    from pdl_dependent_copy import DependentCopy

    base = variant.removesuffix("_pdl")
    problem = _problem(base)
    plain = _Op(problem, base, config)
    entry = _Op(problem, base, config, launch_pdl_dependents_at_entry=True, use_pdl=variant.endswith("_pdl"))
    for op in (plain, entry):
        _check_global_descriptor_path(op, base, config)
    expected = torch.empty_like(problem["output"])
    plain(problem["a"], expected)
    torch.cuda.synchronize()

    output = problem["output"]
    seen = torch.empty_like(output)
    dependent = DependentCopy(torch.device("cuda"))

    def chain():
        entry(problem["a"], output)
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


# ---------------------------------------------------------------------------
# pdl_mode behind a producer of A (MXFP8)
# ---------------------------------------------------------------------------
#
# The kernel classes run through the split helpers of test_grouped_gemm_wgrad_atomic_counter (own grids, a caller-owned
# counter); "api" runs the same chains through GroupedGemmWgradSm100. The producer is EarlyReleaseWriter: it lets its
# dependents start at once and writes A slowly.

# The trigger_then_wait_at_exit GEMM may read A before the grid right before it ends, so A must not come from that grid.
EXIT_ARRANGEMENTS = {
    # producer -> first launch (wait_then_trigger) -> second launch (exit wait): the first launch releases the second
    # only after its wait for the producer, so A is final. The documented use.
    "producer_tail_base": "correct",
    # producer -> exit-wait launch: the producer is its direct predecessor (breaks the contract).
    "producer_base": "unsafe",
    # producer -> first launch that triggers at entry, before its wait -> exit-wait launch (breaks the contract).
    "producer_tail_at_entry_base": "unsafe",
}


class _ApiSplit:
    """The split chain through the class API on the problem of the class-level split tests."""

    def __init__(self):
        from gemm.cutedsl.test_grouped_gemm_wgrad_atomic_counter import split_problem

        problem = split_problem()
        self.counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        config = ((256, 256), (2, 1))
        ctr = dict(sample_scheduler_counter=self.counter)
        self.single = _Op(problem, "mxfp8", config)
        self.start = _Op(problem, "mxfp8", config, use_pdl=True)
        self.tail = _Op(problem, "mxfp8_counter", config, use_pdl=True, pdl_mode="wait_then_trigger", **ctr)
        self.base = _Op(problem, "mxfp8_counter", config, use_pdl=True, pdl_mode="trigger_then_wait_at_exit", **ctr)
        self.entry_tail = _Op(problem, "mxfp8_counter", config, use_pdl=True, launch_pdl_dependents_at_entry=True, **ctr)
        self.wtt = _Op(problem, "mxfp8", config, use_pdl=True, pdl_mode="wait_then_trigger")
        for op in (self.single, self.start, self.tail, self.base, self.wtt):
            _check_global_descriptor_path(op, "mxfp8", config)


_API_SPLIT = {}


def _launches(impl):
    """(reference dW, {role: launch(a, out, counter)}) for a kernel class name or "api"."""
    if impl == "api":
        if "s" not in _API_SPLIT:
            _API_SPLIT["s"] = _ApiSplit()
        api = _API_SPLIT["s"]
        expected = torch.full_like(api.single.problem["output"], float("nan"))
        api.single(api.single.problem["a"], expected)
        torch.cuda.synchronize()

        def counted(op):
            return lambda a, out, counter: op(a, out, counter=counter)

        roles = {
            "wait_at_start": lambda a, out, counter=None: api.start(a, out),
            "wait_then_trigger": lambda a, out, counter=None: api.wtt(a, out),
            "tail": counted(api.tail),
            "base": counted(api.base),
            "tail_at_entry": counted(api.entry_tail),
        }
        return expected, roles
    from gemm.cutedsl.test_grouped_gemm_wgrad_atomic_counter import TAIL_CLUSTERS, split_base, split_launch, split_reference, split_tail

    roles = {
        "wait_at_start": split_launch("pdl_start", impl, use_pdl=True),
        "wait_then_trigger": split_launch("pdl_wtt", impl, use_pdl=True, pdl_mode="wait_then_trigger"),
        "tail": split_tail(impl),
        "base": split_base(impl),
        "tail_at_entry": split_launch("tail_at_entry", impl, clusters=TAIL_CLUSTERS, counter=True, use_pdl=True, launch_pdl_dependents_at_entry=True),
    }
    return split_reference(impl), roles


@pytest.mark.L0
@pytest.mark.parametrize("mode", ["wait_at_start", "wait_then_trigger", "trigger_then_wait_at_exit"])
@pytest.mark.parametrize("impl", ["dense", "reuse", "api"])
def test_pdl_mode_behind_producer_of_a(impl, mode):
    """Each pdl_mode behind an early-releasing producer of A, eager and replayed from a CUDA graph; dW must be the
    single plain launch's every time.

    * wait_at_start: producer -> GEMM.
    * wait_then_trigger: producer -> GEMM -> a dependent copy of A that does NOT wait: it starts only when the GEMM
      triggers, which is after the GEMM's wait, so it must copy the final A.
    * trigger_then_wait_at_exit: producer -> first launch (wait_then_trigger) -> exit-wait launch, on one caller counter;
      the producer is not the exit-wait launch's direct predecessor, as the contract requires. Two arrangements that
      break the contract (the producer right before the exit-wait launch, or a first launch that triggers before its
      wait; the latter only on the kernel classes' small first grid) must read stale A whenever the positive control
      shows a dependent starting early.
    """
    _require_sm100()
    if impl != "api":
        from gemm.cutedsl.test_grouped_gemm_wgrad_atomic_counter import require_sm100_sm103

        require_sm100_sm103()
    elif torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("The global-descriptor wgrad is SM100/SM103/SM107 only")
    from pdl_dependent_copy import DependentCopy, EarlyReleaseWriter
    from gemm.cutedsl.test_grouped_gemm_wgrad_atomic_counter import split_problem

    expected, roles = _launches(impl)
    p = split_problem()
    a_final = p["a"].clone()
    a_buf = torch.empty_like(a_final)
    stale = torch.empty_like(a_final)
    stale.view(torch.uint8).copy_(~a_final.view(torch.uint8))  # zero-free and far from A
    writer = EarlyReleaseWriter(torch.device("cuda"))
    unwaited = DependentCopy(torch.device("cuda"), wait=False)
    seen = torch.empty_like(a_final)
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.empty_like(expected)

    def chain_for(arrangement):
        def chain():
            counter.zero_()
            writer(a_final, a_buf)
            if arrangement in ("wait_at_start", "wait_then_trigger"):
                roles[arrangement](a_buf, out)
                if arrangement == "wait_then_trigger":
                    unwaited(a_buf, seen)
                return
            if arrangement == "producer_tail_base":
                roles["tail"](a_buf, out, counter)
            elif arrangement == "producer_tail_at_entry_base":
                roles["tail_at_entry"](a_buf, out, counter)
            roles["base"](a_buf, out, counter)

        return chain

    def control():
        writer(a_final, a_buf)
        unwaited(a_buf, seen)

    arrangements = EXIT_ARRANGEMENTS if mode == "trigger_then_wait_at_exit" else {mode: "correct"}
    if impl == "api" and mode == "trigger_then_wait_at_exit":
        # The API sizes both launches to the whole device, so the exit-wait launch's CTAs find SMs only as CTAs of the
        # first launch exit, which happens after their wait for the producer: an early trigger there cannot show.
        arrangements = {k: v for k, v in arrangements.items() if k != "producer_tail_at_entry_base"}
    for arrangement in arrangements:  # warm every launch
        chain_for(arrangement)()
    control()
    torch.cuda.synchronize()

    def run(launch, check):
        a_buf.copy_(stale)
        out.fill_(float("nan"))
        seen.copy_(stale)
        torch.cuda.synchronize()
        launch()
        torch.cuda.synchronize()
        return check()

    def a_final_seen():
        return torch.equal(seen.view(torch.uint8), a_final.view(torch.uint8))

    def check(arrangement):
        if arrangement == "wait_then_trigger":
            return lambda: _same(out, expected) and a_final_seen()
        return lambda: _same(out, expected)

    early_starts = sum(run(control, lambda: not a_final_seen()) for _ in range(3))
    report = {}
    for arrangement in arrangements:
        chain = chain_for(arrangement)
        graph = _graph(chain)
        try:
            report[arrangement] = [run(launch, check(arrangement)) for launch in (chain, chain, chain, graph.replay, graph.replay)]
        finally:
            graph.reset()
    for arrangement, kind in arrangements.items():
        if kind == "correct":
            assert all(report[arrangement]), f"{arrangement}: differs from the single plain launch: {report}"
    if early_starts == 0:
        pytest.skip(f"no early start reproduced on this device (coverage gap); runs: {report}")
    for arrangement, kind in arrangements.items():
        if kind == "unsafe":
            assert not all(report[arrangement]), f"{arrangement}: no stale read although dependents start early: {report}"
