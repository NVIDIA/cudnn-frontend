# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared by the streaming signal tests of the BF16 grouped GEMM kernels: kernel construction, problems, and one
compile cache, so a kernel built the same way for the same argument layout compiles once per pytest process."""

import torch

ROWS_ALIGNMENT = 256


def frozen(value):
    """A hashable, order-independent stand-in for a kernel option (tuples, enums, plain config objects)."""
    if isinstance(value, dict):
        return tuple(sorted((key, frozen(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(frozen(item) for item in value)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return (type(value).__qualname__, frozen(vars(value)))
    return value


def make_kernel(glu, experts, one_cta=False, **options):
    """A plain or GLU BF16 grouped GEMM, cluster (2, 1) with the 2-CTA MMA or (1, 1) with the 1-CTA MMA."""
    import cutlass
    from cudnn.gemm.cutedsl.grouped.glu.moe_grouped_gemm_glu_bias import MoEGroupedGemmGluBiasBf16Kernel
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode
    from cudnn.gemm.cutedsl.grouped.unfused.moe_grouped_gemm import MoEGroupedGemmBf16Kernel

    common = dict(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=not one_cta,
        mma_tiler_mn=(128, 256) if one_cta else (256, 256),
        cluster_shape_mn=(1, 1) if one_cta else (2, 1),
        vectorized_f32=True,
        expert_cnt=experts,
        weight_mode=MoEWeightMode.DENSE,
    )
    if glu:
        kernel = MoEGroupedGemmGluBiasBf16Kernel(generate_c=True, **common, **options)
    else:
        kernel = MoEGroupedGemmBf16Kernel(generate_c=True, enable_bias=False, **common, **options)
    # Every constructor argument, so a compile is reused only for an identically built kernel.
    kernel.test_compile_key = (type(kernel).__qualname__, frozen({**common, **options}))
    return kernel


def padded(tokens):
    return -(-tokens // ROWS_ALIGNMENT) * ROWS_ALIGNMENT


def make_problem(tokens_per_expert, n, k, seed=0, rows=None):
    """Expert-major rows padded to ROWS_ALIGNMENT; padding rows hold data too (the kernel computes them).  One expert
    per entry of ``tokens_per_expert``; ``rows`` can exceed the experts' rows (the extra rows belong to no expert)."""
    experts = len(tokens_per_expert)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    blocks = [padded(tokens) for tokens in tokens_per_expert]
    rows = rows or max(sum(blocks), ROWS_ALIGNMENT)
    a = torch.randn(rows, k, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn(experts, n, k, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.25
    # A device layout reads layout[0] (expert 0's start) before padded_offsets = layout[1:E+1].
    layout = torch.zeros(experts + 1, device="cuda", dtype=torch.int32)
    layout[1:] = torch.tensor(blocks, device="cuda", dtype=torch.int64).cumsum(0).to(torch.int32)
    alpha = torch.rand(experts, generator=generator, device="cuda") + 0.5
    prob = torch.rand(rows, 1, 1, generator=generator, device="cuda")
    return {"a": a, "b": b, "layout": layout, "offsets": layout[1:], "alpha": alpha, "prob": prob, "rows": rows, "n": n, "k": k}


def as_mkl(x):
    m, k = x.shape
    return x.as_strided((m, k, 1), (k, 1, m * k))


def new_buffers(kernel, problem, glu):
    """NaN-filled C and D, and a zeroed workspace."""
    rows, n = problem["rows"], problem["n"]
    c = torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    d = torch.full((rows, n // 2 if glu else n), float("nan"), device="cuda", dtype=torch.bfloat16)
    workspace = torch.zeros(max(kernel.get_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    return c, d, workspace


def kernel_args(problem, glu, c, d, workspace):
    """The kernel's arguments; index 5 (the B major mode) is a compile-time constant, not passed at launch."""
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack

    def tensor(x):
        return from_dlpack(x, assumed_align=16)

    n, k = problem["n"], problem["k"]
    args = [
        tensor(as_mkl(problem["a"])),
        tensor(problem["b"].permute(1, 2, 0)),
        cutlass.Int32(n),
        cutlass.Int32(k),
        cutlass.Int64(k),
        OperandMajorMode.K,
        from_dlpack(workspace, assumed_align=128).iterator,
        tensor(as_mkl(c)),
        tensor(as_mkl(d)),
        from_dlpack(problem["offsets"], assumed_align=4),
        tensor(problem["alpha"]),
    ]
    prob = tensor(problem["prob"])
    return args + ([prob, None] if glu else [None, prob])


def launch_args(args):
    return [x for i, x in enumerate(args) if i != 5]


# One compile per (kernel construction, argument layout, max_active_clusters) in this process: tensors are passed with
# static layouts, so their shapes and strides are part of the key; their values and the launch buffers are not.
COMPILED = {}


def compile_once(kernel, problem, glu, buffers, max_active_clusters=None):
    """The compiled kernel for this construction and argument layout, compiling it on first use; also returns the
    ``max_active_clusters`` it was built for."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.utils import HardwareInfo

    if max_active_clusters is None:
        max_active_clusters = HardwareInfo().get_max_active_clusters(2)
    c, d, workspace = buffers
    layouts = [as_mkl(problem["a"]), problem["b"].permute(1, 2, 0), as_mkl(c), as_mkl(d), problem["offsets"], problem["alpha"], problem["prob"], workspace]
    key = (
        kernel.test_compile_key,
        glu,
        problem["n"],
        problem["k"],
        max_active_clusters,
        tuple((tuple(x.shape), tuple(x.stride()), str(x.dtype)) for x in layouts),
    )
    compiled = COMPILED.get(key)
    if compiled is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = COMPILED[key] = cute.compile(kernel, *kernel_args(problem, glu, *buffers), max_active_clusters, stream)
    return compiled


def spin_before_pdl_launch(kernel, stream):
    """With ``use_pdl`` the GEMM waits for the launch right before it on ``stream`` only at exit, after it has read the
    signal workspace, so that launch must not be the workspace reset (a torch op).  Enqueue a short spin kernel there:
    it starts once the reset is complete and writes nothing the GEMM reads, as the producer does in the contract's
    order (reset, producer, GEMM)."""
    if getattr(kernel, "use_pdl", False):
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1000)


def signal_ready(kernel):
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace

    return lambda workspace: reset_streaming_signal_workspace(workspace, kernel, all_complete=True)


# Rows per expert.  More than 32 experts take the batch opener several lane rounds; an expert of more than 16 M tiles
# (EXPERT_BATCH_MAX_M_TILES) needs several batches.
MANY_EXPERTS_64 = [(0, 1, 255, 256, 257, 700, 1300)[i % 7] for i in range(64)]
# 64 small experts of 0 to 3 M tiles: with every M tile ready, most lanes of a batch opening hold a run that ends
# their expert, so one opening publishes many batches.
SMALL_EXPERTS_64 = [(0, 1, 256, 300, 512, 600, 768)[i % 7] for i in range(64)]
BIG_EXPERT = [256, 5000, 0, 513, 1]
