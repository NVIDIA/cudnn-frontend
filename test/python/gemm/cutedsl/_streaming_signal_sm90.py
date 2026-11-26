# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared by the streaming signal tests of the SM90 BF16 grouped GEMM kernels: the plain and GLU forward kernels
(cooperative 128x256 and pingpong 128x192) and the dGLU (cooperative 128x256), all at cluster (2, 1), so one ``signal[]``
entry is a 256-row M tile.  Kernel construction, problems, buffers, launch arguments and one compile cache per pytest
process.

- "plain": A [rows, K] x B [E, N, K] -> C, D = prob * C.
- "glu": the same with a fused SwiGLU, N = 2 * FFN interleaved gate / up columns, D [rows, FFN].
- "dglu": dY [rows, K] x W2 [E, FFN, K] with the saved C [rows, 2 FFN] -> D [rows, 2 FFN] and dprob, one slot per
  (row, N tile) (``deterministic=True``), so every output is independent of the tile order.
"""

import pytest
import torch

from gemm.cutedsl._streaming_signal import BIG_EXPERT, MANY_EXPERTS_64, ROWS_ALIGNMENT, as_mkl, frozen, spin_before_pdl_launch
from gemm.cutedsl._streaming_signal import make_problem as make_base_problem

HIDDEN = 512
FFN = 512
KINDS = ("plain", "glu", "dglu")
KIND_IDS = {"plain": "fprop-bf16-plain", "glu": "fprop-bf16-glu", "dglu": "bprop-bf16-dglu"}
#: schedule -> (CTA tile, pingpong).  The dGLU has the cooperative schedule only.
SCHEDULES = {"cooperative": ((128, 256), False), "pingpong": ((128, 192), True)}
KIND_SCHEDULES = [(kind, schedule) for kind in KINDS for schedule in SCHEDULES if not (kind == "dglu" and schedule == "pingpong")]

# Rows per expert.  More than 32 experts take the batch opener several lane rounds; an expert of more than 16 M tiles
# needs several batches.
ROUTINGS = {
    "uneven": [700, 0, 256, 1],
    "many_experts_64": MANY_EXPERTS_64,
    "many_experts_128_small": [(256, 300, 512, 600, 768)[i % 5] for i in range(128)],
    "big_expert": BIG_EXPERT,
}


@pytest.fixture(autouse=True)
def require_sm90():
    """Every test here: an SM90 GPU and a CuTe DSL that is new enough and targets it."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old
    from cudnn.tensor_adapter import has_wgmma

    cc = torch.cuda.get_device_capability()
    if not has_wgmma(cc[0]):
        pytest.skip("an SM90 GPU is required")
    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip(f"the CuTe DSL is older than the library floor: {version}")
    reason = cutedsl_arch_requirement_error(cc)
    if reason is not None:
        pytest.skip(reason)


def levels(cases, l0, ids):
    """``cases`` as pytest params: L0 for those in ``l0``, L1 for the rest."""
    return [pytest.param(*case, marks=pytest.mark.L0 if case in l0 else pytest.mark.L1, id=ids(*case)) for case in cases]


def kernel_class(kind):
    if kind == "dglu":
        from cudnn.gemm.cutedsl.grouped.dglu.moe_grouped_gemm_dglu_sm90 import MoEGroupedGemmDgluBf16KernelSm90 as cls
    elif kind == "glu":
        from cudnn.gemm.cutedsl.grouped.glu.moe_grouped_gemm_glu_sm90 import MoEGroupedGemmGluBf16KernelSm90 as cls
    else:
        from cudnn.gemm.cutedsl.grouped.unfused.moe_grouped_gemm_sm90 import MoEGroupedGemmBf16KernelSm90 as cls
    return cls


def make_kernel(kind, experts, schedule="cooperative", **options):
    """The stock dynamic-scheduler kernel, or with ``streaming_signal_capacity`` (and the other options) the signal
    consumer; cluster (2, 1)."""
    import cutlass
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    tile, pingpong = SCHEDULES[schedule]
    common = dict(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=tile,
        cluster_shape_mn=(2, 1),
        vectorized_f32=False,
        expert_cnt=experts,
        weight_mode=MoEWeightMode.DENSE,
        use_dynamic_sched=True,
    )
    if kind == "dglu":
        assert not pingpong, "the SM90 dGLU has the cooperative schedule only"
        kernel = kernel_class(kind)(**common, act_func="dswiglu", deterministic=True, **options)
    elif kind == "glu":
        kernel = kernel_class(kind)(**common, generate_c=True, enable_bias=False, act_func="swiglu", pingpong=pingpong, **options)
    else:
        kernel = kernel_class(kind)(**common, generate_c=True, enable_bias=False, pingpong=pingpong, **options)
    # Every constructor argument, so a compile is reused only for an identically built kernel.
    kernel.test_kind = kind
    kernel.test_compile_key = (kind, schedule, frozen({**common, **options}))
    return kernel


def signal_options(kind, capacity, variant):
    """``variant``: "signal" (every M tile from signal[]), or "signal_layout_pdl" (also the PDL options; the forward
    kernels with the producer-written layout)."""
    options = dict(streaming_signal_capacity=capacity)
    if variant == "signal_layout_pdl":
        options.update(use_pdl=True, launch_pdl_dependents_at_entry=True)
        if kind != "dglu":
            options["device_layout"] = True
    return options


def gemm_n(kind):
    """The GEMM N: B's rows per expert."""
    return 2 * FFN if kind == "glu" else FFN


def make_problem(kind, tokens_per_expert, seed=0, rows=None):
    """Expert-major rows padded to ROWS_ALIGNMENT (padding rows hold data too); the dGLU also gets C and beta."""
    problem = make_base_problem(tokens_per_expert, gemm_n(kind), HIDDEN, seed=seed, rows=rows)
    if kind == "dglu":
        generator = torch.Generator(device="cuda").manual_seed(seed + 1000)
        experts = len(tokens_per_expert)
        # The forward's saved pre-activation (gate | up), wide enough to reach the activation's tails.
        problem["c"] = (torch.randn(problem["rows"], 2 * FFN, generator=generator, device="cuda") * 4).to(torch.bfloat16)
        problem["beta"] = torch.rand(experts, generator=generator, device="cuda") + 0.5
    problem["kind"] = kind
    return problem


def capacity_of(problem):
    return problem["rows"] // ROWS_ALIGNMENT


def valid_rows(problem):
    return int(problem["offsets"][-1].item())


def dprob_slots():
    """The dGLU's dprob slots per row: one per 256-column N tile of FFN."""
    return -(-FFN // SCHEDULES["cooperative"][0][1])


def new_buffers(kernel, problem):
    """The kernel's outputs, poisoned (NaN; dprob slots 0, as the kernel adds into them), and a zeroed workspace."""
    rows, kind = problem["rows"], problem["kind"]
    workspace = torch.zeros(max(kernel.get_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    if kind == "dglu":
        d = torch.full((rows, 2 * FFN), float("nan"), device="cuda", dtype=torch.bfloat16)
        dprob = torch.zeros((rows, dprob_slots(), 1), device="cuda", dtype=torch.float32)
        return {"d": d, "dprob": dprob, "workspace": workspace}
    n = problem["n"]
    c = torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    d = torch.full((rows, n // 2 if kind == "glu" else n), float("nan"), device="cuda", dtype=torch.bfloat16)
    return {"c": c, "d": d, "workspace": workspace}


def outputs(problem, buffers):
    """Every output over the experts' rows (rows past the last expert are not written)."""
    rows = valid_rows(problem)
    names = ("d", "dprob") if problem["kind"] == "dglu" else ("c", "d")
    return {name: buffers[name][:rows] for name in names}


def kernel_args(problem, buffers):
    """The kernel's arguments; index 5 (the B major mode) is a compile-time constant, not passed at launch."""
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack

    def tensor(x):
        return from_dlpack(x, assumed_align=16)

    n, k = problem["n"], problem["k"]
    head = [
        tensor(as_mkl(problem["a"])),
        tensor(problem["b"].permute(1, 2, 0)),
        cutlass.Int32(n),
        cutlass.Int32(k),
        cutlass.Int64(k),
        OperandMajorMode.K,
        from_dlpack(buffers["workspace"], assumed_align=128).iterator,
    ]
    offsets = from_dlpack(problem["offsets"], assumed_align=4)
    if problem["kind"] == "dglu":
        tail = [tensor(as_mkl(problem["c"])), tensor(as_mkl(buffers["d"])), offsets, tensor(problem["alpha"]), tensor(problem["beta"])]
        return head + tail + [tensor(problem["prob"]), from_dlpack(buffers["dprob"], assumed_align=4), cutlass.Float32(0.0), None]
    tail = [tensor(as_mkl(buffers["c"])), tensor(as_mkl(buffers["d"])), offsets, tensor(problem["alpha"])]
    prob = tensor(problem["prob"])
    return head + tail + ([prob, None] if problem["kind"] == "glu" else [None, prob])


def launch_args(args):
    return [x for i, x in enumerate(args) if i != 5]


# One compile per (kernel construction, argument layout, max_active_clusters) in this process.
COMPILED = {}


def compile_once(kernel, problem, buffers, max_active_clusters=None):
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.utils import HardwareInfo

    if max_active_clusters is None:
        max_active_clusters = HardwareInfo().get_max_active_clusters(2)
    tensors = [problem["a"], problem["b"], problem["offsets"], problem["alpha"], problem["prob"]] + list(buffers.values())
    tensors += [problem[name] for name in ("c", "beta") if problem["kind"] == "dglu"]
    key = (kernel.test_compile_key, max_active_clusters, tuple((tuple(x.shape), tuple(x.stride()), str(x.dtype)) for x in tensors))
    compiled = COMPILED.get(key)
    if compiled is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = COMPILED[key] = cute.compile(kernel, *kernel_args(problem, buffers), max_active_clusters, stream)
    return compiled


class Launcher:
    """One compile of ``kernel`` for a problem shape, launched any number of times on fresh poisoned buffers."""

    def __init__(self, kernel, problem, max_active_clusters=None):
        self.kernel = kernel
        self._compiled = compile_once(kernel, problem, self.buffers(problem), max_active_clusters)

    def buffers(self, problem):
        return new_buffers(self.kernel, problem)

    def launch(self, problem, buffers, stream=None):
        """Enqueue the GEMM on ``stream`` (default: current) over ``buffers``."""
        import cuda.bindings.driver as cuda

        stream = stream or torch.cuda.current_stream()
        spin_before_pdl_launch(self.kernel, stream)
        self._compiled(*launch_args(kernel_args(problem, buffers)), cuda.CUstream(stream.cuda_stream))

    def __call__(self, problem, prepare_workspace=None):
        """Fresh buffers, ``prepare_workspace(workspace)``, one launch; returns the outputs over the experts' rows."""
        buffers = self.buffers(problem)
        if prepare_workspace is not None:
            prepare_workspace(buffers["workspace"])
        self.launch(problem, buffers)
        return outputs(problem, buffers)


def signal_view(kernel, workspace):
    """``signal[]`` inside the kernel's workspace."""
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import streaming_signal_workspace_offsets

    offsets = streaming_signal_workspace_offsets(kernel)
    word = offsets["signal"] // 4
    return workspace.view(torch.int32)[word : word + offsets["streaming_signal_capacity"]]


def release_order(order, capacity):
    """The order in which the producer releases the M tiles."""
    if order == "in_order":
        return list(range(capacity))
    if order == "reverse":
        return list(reversed(range(capacity)))
    return torch.randperm(capacity, generator=torch.Generator().manual_seed(capacity)).tolist()


def run_with_producer(launcher, problem, final_a, orders, final_layout=None):
    """Per release order: the GEMM starts while every signal[] entry reads not-ready and A holds NaN (and, with
    ``final_layout``, the layout reads -1); a producer on another stream (launched first, after a sleep) writes the
    layout, then A one M tile at a time in that order, each followed by its signal[] entry.  Reading any tile early
    carries NaN into the outputs.  Returns {order: outputs}."""
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace

    kernel = launcher.kernel
    # One launch with every M tile ready loads the module: under lazy loading a first launch waits for running work.
    launcher(problem, lambda workspace: reset_streaming_signal_workspace(workspace, kernel, all_complete=True))
    torch.cuda.synchronize()
    gemm_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    results = {}
    for order in orders:
        buffers = launcher.buffers(problem)
        signal = signal_view(kernel, buffers["workspace"])
        problem["a"].fill_(float("nan"))
        if final_layout is not None:
            problem["layout"].fill_(-1)
        gemm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gemm_stream):
            reset_streaming_signal_workspace(buffers["workspace"], kernel)
        producer_stream.wait_stream(gemm_stream)
        # Producer launches go before the GEMM's, so no host launch waits for the spinning GEMM.
        with torch.cuda.stream(producer_stream):
            torch.cuda._sleep(20_000_000)
            if final_layout is not None:
                problem["layout"].copy_(final_layout)
            for tile in release_order(order, signal.numel()):
                rows_of_tile = slice(tile * ROWS_ALIGNMENT, (tile + 1) * ROWS_ALIGNMENT)
                problem["a"][rows_of_tile].copy_(final_a[rows_of_tile])
                signal[tile].fill_(tile)
        launcher.launch(problem, buffers, gemm_stream)
        torch.cuda.synchronize()
        results[order] = outputs(problem, buffers)
    problem["a"].copy_(final_a)
    if final_layout is not None:
        problem["layout"].copy_(final_layout)
    return results


def assert_same_bytes(got, want, label=""):
    for name, expected in want.items():
        actual = got[name]
        assert actual.shape == expected.shape, f"{label} {name}: shape {tuple(actual.shape)} vs {tuple(expected.shape)}"
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)), f"{label} {name}: bytes differ"
