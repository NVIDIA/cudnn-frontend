# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared by the streaming signal tests of the SM100 block-scaled grouped GEMM kernels, on top of #2810's BF16 helper:
problems, kernels, one compile cache per process, the N_CHUNK consumer and the fake input producer.

Kernel kinds (K = 512, cluster (2, 1) with the 2-CTA 256 x 256 tile unless ``one_cta``, dense weights):

- "quant": the MXFP8 MoE quant kernel (FC2), A [rows, K] x B [E, N, K] -> BF16 D [rows, N]; N_CHUNK output notify.
- "glu": the MXFP8 SwiGLU kernel (FC1), N = gate and up -> BF16 C [rows, N] and D [rows, N / 2]; the input signal.
- "dglu_bf16", "dglu": the BF16 and MXFP8 dSwiGLU kernels, dY [rows, K] x W2 [E, N, K] with the saved C [rows, 2N] ->
  BF16 D [rows, 2N] and dprob (``deterministic=True``: one FP32 slot per 256-wide N tile); the input signal.

MXFP8 operands are E4M3 codes with E8M0 exponents, kept row-major (``a_exp``, ``b_exp``) and as the swizzled SF atoms
the kernels read (``sfa``, ``sfb``); with ``compact_sfa_sf_warp`` the kernel reads ``a_exp`` and builds the atoms.
"""

import pytest
import torch

from gemm.cutedsl import _streaming_signal as bf16
from gemm.cutedsl._streaming_signal import ROWS_ALIGNMENT, frozen, padded, signal_ready
from gemm.cutedsl.test_grouped_gemm_streaming_signal import levels, release_order, require_sm10x  # noqa: F401  (re-exported)

HIDDEN = 512
SF_VEC_SIZE = 32
#: Scale-factor atom: 128 rows x 4 scales, 512 bytes (tcgen05 block-scaled MMA layout).
SF_ATOM_COLS = 4
#: NaN as an E4M3 code, in both bytes of a BF16 and as an E8M0 scale: a tile read before its producer wrote it carries
#: NaN into the outputs.
POISON = 0xFF
#: The default N of each kind: the GLU's gate and up, the dGLU's W2 rows.
N = {"quant": 1024, "glu": 1024, "dglu_bf16": 512, "dglu": 512}
#: Parameter ids: direction, recipe, kernel.
KIND_IDS = {"quant": "fprop-mxfp8-fc2", "glu": "fprop-mxfp8-glu", "dglu_bf16": "bprop-bf16-dglu", "dglu": "bprop-mxfp8-dglu"}
# More than 32 experts take the batch opener several lane rounds; an expert of more than 16 M tiles needs several batches.
ROUTINGS = {"uneven": [700, 0, 256, 1], "many_experts_64": bf16.MANY_EXPERTS_64, "big_expert": bf16.BIG_EXPERT}


def is_rubin():
    return tuple(torch.cuda.get_device_capability()) == (10, 7)


def require_kernel(kind):
    """Skip what this GPU's kernel of ``kind`` does not take."""
    if not is_rubin():
        return
    if kind == "quant":
        pytest.skip("N_CHUNK on the SM107 (Rubin) quant kernel is not wired yet")
    if kind != "dglu_bf16":
        pytest.skip("the SM100 kernel (the Rubin twin does not take a streaming input producer)")


def kernel_class(kind):
    if kind == "quant":
        from cudnn.gemm.cutedsl.grouped.quant.grouped_gemm_quant import BlockScaledMoEGroupedGemmQuantKernel as cls
    elif kind == "glu":
        from cudnn.gemm.cutedsl.grouped.glu.moe_blockscaled_grouped_gemm_glu_bias import BlockScaledMoEGroupedGemmGluBiasKernel as cls
    elif kind == "dglu_bf16":
        from cudnn.gemm.cutedsl.grouped.dglu.moe_grouped_gemm_dglu_dbias import MoEGroupedGemmDgluDbiasBf16Kernel as cls
    else:
        from cudnn.gemm.cutedsl.grouped.dglu.moe_blockscaled_grouped_gemm_dglu_dbias import BlockScaledMoEGroupedGemmDgluDbiasKernel as cls
    return cls


def make_kernel(kind, experts, one_cta=False, **options):
    """The kernel of ``kind``; an option naming a construction argument overrides it (the rejection tests)."""
    import cutlass
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    require_kernel(kind)
    common = dict(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=not one_cta,
        mma_tiler_mn=(128, 256) if one_cta else (256, 256),
        cluster_shape_mn=(1, 1) if one_cta else (2, 1),
        vectorized_f32=True,
        expert_cnt=experts,
        weight_mode=MoEWeightMode.DENSE,
    )
    if kind != "dglu_bf16":
        common.update(sf_vec_size=SF_VEC_SIZE, discrete_col_sfd=False)
    if kind in ("quant", "glu"):
        common["generate_sfd"] = False
    if kind == "quant":
        common["enable_bias"] = False
    if kind == "glu":
        common["act_func"] = "swiglu"
    if kind.startswith("dglu"):
        common["act_func"] = "dswiglu"
        options.setdefault("deterministic", True)
    # compact_sfa_sf_warp is passed only when on: the BF16 kernel has no such argument.
    if not options.get("compact_sfa_sf_warp"):
        options.pop("compact_sfa_sf_warp", None)
    common.update({name: options.pop(name) for name in list(options) if name in common})
    kernel = kernel_class(kind)(**common, **options)
    kernel.test_kind, kernel.test_cluster_size = kind, 1 if one_cta else 2
    kernel.test_compact = bool(options.get("compact_sfa_sf_warp"))
    kernel.test_signal = bool(options.get("streaming_signal_capacity"))
    kernel.test_device_layout = bool(options.get("device_layout"))
    kernel.test_deterministic = bool(options.get("deterministic"))
    kernel.test_compile_key = (type(kernel).__qualname__, frozen({**common, **options}))
    return kernel


def trigger_options(kind):
    """The GLU's device layout with the all-slots-ready PDL trigger; the dGLU's trigger at entry."""
    if kind == "glu":
        return dict(device_layout=True, use_pdl=True, launch_pdl_dependents_at_all_streaming_signal_slots_ready=True)
    return dict(use_pdl=True, launch_pdl_dependents_at_entry=True)


def variant_options(kind, problem, variant):
    """Options of an input-signal ``variant``: "signal" (one flag per M tile), "trigger" (the signal and
    ``trigger_options``), "entry" (the dGLU's entry trigger alone); a "compact" prefix adds ``compact_sfa_sf_warp``
    ("compact" alone: on the stock dynamic scheduler)."""
    options = dict(use_dynamic_sched=True)
    if variant.startswith("compact"):
        options["compact_sfa_sf_warp"] = True
    if variant.endswith(("signal", "trigger")):
        options["streaming_signal_capacity"] = problem["rows"] // ROWS_ALIGNMENT
    if variant.endswith(("trigger", "entry")):
        options.update(trigger_options(kind))
    return options


def stock_kernel(kind, experts, one_cta=False):
    return make_kernel(kind, experts, one_cta, use_dynamic_sched=True)


# ---- Problems ----


def swizzle_sf(scales):
    """Row-major scales [rows, cols] (rows % 128 == 0, cols % 4 == 0) -> the flat SF-atom bytes the kernel reads:
    128-row blocks outermost, then 4-column atoms, each atom 32 x (4 x 4) bytes."""
    rows, cols = scales.shape
    return scales.view(rows // 128, 4, 32, cols // 4, 4).permute(0, 3, 2, 1, 4).contiguous().view(-1)


def sf_cols(k):
    """Scale columns per row, padded to whole atoms."""
    return -(-k // (SF_VEC_SIZE * SF_ATOM_COLS)) * SF_ATOM_COLS


def mxfp8_codes(rows, k, generator):
    """E4M3 codes [rows, k] and E8M0 exponents [rows, sf_cols(k)] (2^-3 .. 2^3; padding columns 2^0)."""
    codes = (torch.randn(rows, k, generator=generator, device="cuda") * 2).to(torch.float8_e4m3fn)
    exponents = torch.full((rows, sf_cols(k)), 127, dtype=torch.uint8, device="cuda")
    exponents[:, : k // SF_VEC_SIZE] = torch.randint(124, 131, (rows, k // SF_VEC_SIZE), generator=generator, device="cuda", dtype=torch.uint8)
    return codes, exponents


def complete(problem):
    """The derived fields: offsets (``layout[1:]``, inclusive row ends), sizes and the swizzled SF atoms."""
    experts, n, k = problem["b"].shape
    problem.update(offsets=problem["layout"][1:], rows=problem["a"].shape[0], n=n, k=k)
    if "a_exp" in problem:
        problem["sfa"] = swizzle_sf(problem["a_exp"])
        problem["sfb"] = torch.cat([swizzle_sf(problem["b_exp"][e]) for e in range(experts)])
    return problem


def make_problem(kind, tokens_per_expert, n=None, k=HIDDEN, seed=0):
    """Expert-major rows padded to ROWS_ALIGNMENT (padding rows hold data too): A [rows, k], B [experts, n, k], alpha,
    prob [rows, 1, 1]; MXFP8 exponents ``a_exp`` [rows, sf_cols(k)] and ``b_exp`` [experts, n, sf_cols(k)]; the dGLU's C
    [rows, 2n] and beta [experts]."""
    n = n or N[kind]
    experts = len(tokens_per_expert)
    if kind == "dglu_bf16":
        problem = bf16.make_problem(tokens_per_expert, n, k, seed=seed)
    else:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        rows = max(sum(padded(tokens) for tokens in tokens_per_expert), ROWS_ALIGNMENT)
        a, a_exp = mxfp8_codes(rows, k, generator)
        b, b_exp = mxfp8_codes(experts * n, k, generator)
        layout = torch.zeros(experts + 1, device="cuda", dtype=torch.int32)
        layout[1:] = torch.tensor([padded(tokens) for tokens in tokens_per_expert], device="cuda", dtype=torch.int64).cumsum(0).to(torch.int32)
        alpha = torch.rand(experts, generator=generator, device="cuda") + 0.5
        prob = torch.rand(rows, 1, 1, generator=generator, device="cuda")
        problem = dict(a=a, a_exp=a_exp, b=b.view(experts, n, k), b_exp=b_exp.view(experts, n, -1), layout=layout, alpha=alpha, prob=prob)
    if kind.startswith("dglu"):
        generator = torch.Generator(device="cuda").manual_seed(seed + 1000)
        problem["c"] = (torch.randn(problem["a"].shape[0], 2 * n, generator=generator, device="cuda") * 2).to(torch.bfloat16)
        problem["beta"] = torch.rand(experts, generator=generator, device="cuda") + 0.5
    return complete(problem)


def valid_rows(problem):
    return int(problem["offsets"][-1].item())


# ---- Launches ----


def new_buffers(kernel, problem):
    """NaN-filled BF16 outputs, a zeroed dprob and a zeroed workspace."""
    rows, n, kind = problem["rows"], problem["n"], kernel.test_kind
    workspace = torch.zeros(max(kernel.get_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)

    def nan(cols):
        return torch.full((rows, cols), float("nan"), device="cuda", dtype=torch.bfloat16)

    if kind == "quant":
        return nan(n), workspace
    if kind == "glu":
        return nan(n), nan(n // 2), workspace
    dprob = torch.zeros(rows, n // 256 if kernel.test_deterministic else 1, device="cuda", dtype=torch.float32)
    return nan(2 * n), dprob, workspace


OUTPUTS = {"quant": ("d",), "glu": ("c", "d"), "dglu_bf16": ("d", "dprob"), "dglu": ("d", "dprob")}


def outputs(kernel, problem, buffers):
    """The output buffers by name, over the experts' rows."""
    rows = valid_rows(problem)
    return {name: value[:rows] for name, value in zip(OUTPUTS[kernel.test_kind], buffers)}


def assert_same_bytes(got, want, label=""):
    """Raw-byte equal outputs."""
    for name in want:
        assert torch.equal(got[name].view(torch.int16), want[name].view(torch.int16)), f"{label}: {name} differs from the stock scheduler"


#: Index of the amax tensor among the quant kernel's arguments.
AMAX_ARG = 13


def kernel_args(kernel, problem, *buffers):
    """The kernel's arguments before max_active_clusters; ``b_major_arg(kernel)`` is a compile-time constant."""
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack

    def tensor(x):
        return from_dlpack(x, assumed_align=16)

    def e8m0(x):
        return tensor(x.view(torch.float8_e8m0fnu))

    kind, n, k = kernel.test_kind, problem["n"], problem["k"]
    head = [tensor(problem["a"]), tensor(problem["b"].permute(1, 2, 0) if kind.startswith("dglu") else problem["b"])]
    head += [] if kind == "dglu_bf16" else [e8m0(problem["sfb"])]
    head += [cutlass.Int32(n), cutlass.Int32(k), cutlass.Int64(k), OperandMajorMode.K, from_dlpack(buffers[-1], assumed_align=128).iterator]
    offsets, alpha, prob = from_dlpack(problem["offsets"], assumed_align=4), tensor(problem["alpha"]), tensor(problem["prob"])
    d = tensor(buffers[1] if kind == "glu" else buffers[0])
    if kind == "dglu_bf16":
        # (c, d, padded_offsets, alpha, beta, prob, dprob, linear_offset, dbias)
        return head + [tensor(problem["c"]), d, offsets, alpha, tensor(problem["beta"]), prob, tensor(buffers[1].unsqueeze(-1)), cutlass.Float32(0.0), None]
    sfa = e8m0(problem["a_exp"] if kernel.test_compact else problem["sfa"])
    if kind == "quant":
        # (d, d_col, sfa, sfd_row, sfd_col, amax, norm_const, offsets, alpha, row_scale, bias, prob)
        return head + [d, d, sfa, None, None, None, None, offsets, alpha, None, None, prob]
    if kind == "glu":
        # (c, d, d_col, sfa, sfd_row, sfd_col, amax, norm_const, offsets, alpha, prob, bias)
        return head + [tensor(buffers[0]), d, d, sfa, None, None, None, None, offsets, alpha, prob, None]
    # (c, d, d_col, sfa, sfd_row, sfd_col, amax, norm_const, padded_offsets, alpha, beta, prob, dprob, dbias)
    tail = [tensor(problem["c"]), d, d, sfa, None, None, None, None, offsets, alpha, tensor(problem["beta"]), prob, tensor(buffers[1].unsqueeze(-1))]
    return head + tail + [None]


def b_major_arg(kernel):
    return 5 if kernel.test_kind == "dglu_bf16" else 6


def cu_stream(stream=None):
    import cuda.bindings.driver as cuda

    return cuda.CUstream((stream or torch.cuda.current_stream()).cuda_stream)


COMPILED = {}


class Launcher:
    """One compile of ``kernel`` per construction and argument layout in this process (tensors pass static layouts),
    launched on fresh NaN-filled outputs and a zeroed workspace."""

    def __init__(self, kernel, problem, max_active_clusters=None):
        import cutlass.cute as cute
        from cutlass.utils import HardwareInfo

        self.kernel = kernel
        if max_active_clusters is None:
            max_active_clusters = HardwareInfo().get_max_active_clusters(kernel.test_cluster_size)
        buffers = self.buffers(problem)
        tensors = [value for _, value in sorted(problem.items()) if isinstance(value, torch.Tensor)] + list(buffers)
        key = (kernel.test_compile_key, max_active_clusters, tuple((tuple(x.shape), tuple(x.stride()), str(x.dtype)) for x in tensors))
        if key not in COMPILED:
            COMPILED[key] = cute.compile(kernel, *kernel_args(kernel, problem, *buffers), max_active_clusters, cu_stream())
        self._compiled = COMPILED[key]

    def buffers(self, problem):
        return new_buffers(self.kernel, problem)

    def launch(self, problem, buffers, stream=None):
        skip = b_major_arg(self.kernel)
        self._compiled(*[x for i, x in enumerate(kernel_args(self.kernel, problem, *buffers)) if i != skip], cu_stream(stream))

    def __call__(self, problem, prepare_workspace=None):
        """Fresh buffers, ``prepare_workspace(workspace)`` (for a signal kernel by default: every M tile ready), one
        launch; returns (outputs over the experts' rows, workspace)."""
        buffers = self.buffers(problem)
        prepare_workspace = prepare_workspace or (signal_ready(self.kernel) if self.kernel.test_signal else None)
        if prepare_workspace is not None:
            prepare_workspace(buffers[-1])
        self.launch(problem, buffers)
        return outputs(self.kernel, problem, buffers), buffers[-1]


# ---- Batch invariance (the determinism suite) ----

# One target expert keeps its rows, codes, scales, B, alpha, prob (and the dGLU's C and beta) while its neighbors change.
# Four expert slots each (empty slots pad the count), so one compiled kernel serves every packing.
BANK_TOKENS = {"target": 700, "n300": 300, "n513": 513, "r1": 1, "r255": 255, "r257": 257}
PACKINGS = {
    "alone": ["target", None, None, None],
    "neighbors": ["n300", "target", "n513", None],
    "empty_neighbors": [None, "target", None, None],
    "ragged": ["r1", "r255", "target", "r257"],
    "reordered": ["n513", "n300", None, "target"],
    "duplicated": ["target", "n300", "target", None],
}
SLOTS = 4
BATCH_ROWS = max(sum(padded(BANK_TOKENS[key]) for key in packing if key) for packing in PACKINGS.values())
ROW_KEYS = ("a", "a_exp", "prob", "c")


def pack(bank, packing):
    """A problem of SLOTS experts in ``packing`` order over BATCH_ROWS rows, each copied from ``bank`` (a problem over
    BANK_TOKENS; empty slots: zero codes, unit scales); returns it and the target's row ranges."""
    ends = [0] + bank["offsets"].tolist()
    packed = {}
    for name in ROW_KEYS + ("b", "b_exp", "alpha", "beta"):
        if name in bank:
            value = torch.zeros((BATCH_ROWS if name in ROW_KEYS else SLOTS, *bank[name].shape[1:]), device="cuda", dtype=bank[name].dtype)
            packed[name] = value.fill_(127) if name.endswith("_exp") else value.fill_(1) if value.dtype == torch.float32 else value
    layout = torch.zeros(SLOTS + 1, device="cuda", dtype=torch.int32)
    begin, target_rows = 0, []
    for slot, key in enumerate(packing):
        if key is not None:
            e = list(BANK_TOKENS).index(key)
            end = begin + ends[e + 1] - ends[e]
            for name, value in packed.items():
                if name in ROW_KEYS:
                    value[begin:end] = bank[name][ends[e] : ends[e + 1]]
                else:
                    value[slot] = bank[name][e]
            if key == "target":
                target_rows.append(slice(begin, end))
            begin = end
        layout[slot + 1] = begin
    packed["layout"] = layout
    return complete(packed), target_rows


# ---- N_CHUNK output notify (FC2) ----


def notify_kernel(experts, n, chunk, one_cta=False, clusters_beside_consumer=1, caller_reset=False, **options):
    """The quant kernel with N_CHUNK output notify over D's ``n`` columns in chunks of ``chunk``; ``options`` override
    the dynamic scheduler and the dependent launch at entry (the rejection tests)."""
    from cudnn.gemm.cutedsl.grouped.moe_streaming_output_chunk_tile_scheduler import (
        StreamingOutputNotifyConfig,
        StreamingOutputNotifyMode,
        streaming_output_notify_capacity,
    )

    capacity = streaming_output_notify_capacity(chunk, n)
    notify = StreamingOutputNotifyConfig(
        StreamingOutputNotifyMode.N_CHUNK, chunk, capacity, caller_reset=caller_reset, clusters_beside_consumer=clusters_beside_consumer
    )
    options = {"use_dynamic_sched": True, "launch_pdl_dependents_at_entry": True, **options}
    return make_kernel("quant", experts, one_cta, streaming_output_notify=notify, **options)


def cluster_id(one_cta):
    return "cluster11" if one_cta else "cluster21"


def dequantize(codes, exponents):
    k = codes.shape[-1]
    # A lookup of exact powers of two (E8M0 exponent e is 2^(e - 127)).
    powers = torch.tensor([2.0**e for e in range(-127, 129)], device="cuda")
    return codes.float() * powers[exponents[..., : k // SF_VEC_SIZE].long()].repeat_interleave(SF_VEC_SIZE, dim=-1)


def quant_reference(problem):
    """float32 D = alpha[e] * prob * (SFA * A) @ (SFB * B[e])^T over the experts' rows, from the unswizzled codes and scales."""
    a = dequantize(problem["a"], problem["a_exp"])
    out, begin = torch.zeros(problem["rows"], problem["n"], device="cuda"), 0
    for expert, end in enumerate(problem["offsets"].tolist()):
        out[begin:end] = problem["alpha"][expert] * (a[begin:end] @ dequantize(problem["b"][expert], problem["b_exp"][expert]).T)
        begin = end
    return out * problem["prob"][:, 0, :]


def chunk_signal(kernel, workspace):
    """The kernel's ``signal[]``, one int32 per N chunk, as a view of ``workspace``."""
    from cudnn.gemm.cutedsl.grouped.moe_streaming_output_chunk_tile_scheduler import streaming_output_notify_offsets

    offsets = streaming_output_notify_offsets(kernel)
    return workspace[offsets["signal"] : offsets["signal"] + 4 * offsets["capacity"]].view(torch.int32)


def chunk_words(rows, n, chunk):
    """The int32 words of a row-major BF16 D [rows, n] in each N chunk: (word indices, CSR begin per chunk).  Each chunk
    lists its rows in the order its last tiles finish (even chunks walk M forward, so last row first; odd chunks first
    row first), so a consumer reads the most recently stored rows right after the publication."""
    indices, begins = [], [0]
    for u, first_col in enumerate(range(0, n, chunk)):
        cols = torch.arange(first_col // 2, min(first_col + chunk, n) // 2, device="cuda", dtype=torch.int64)
        row_order = torch.arange(rows, device="cuda", dtype=torch.int64)
        if u % 2 == 0:
            row_order = row_order.flip(0)
        indices.append((row_order[:, None] * (n // 2) + cols[None, :]).reshape(-1))
        begins.append(begins[-1] + indices[-1].numel())
    return torch.cat(indices).to(torch.int32), torch.tensor(begins, device="cuda", dtype=torch.int32)


CONSUMER_THREADS = 256


def _consumer_module():
    """The chunk consumer, built once: block u acquire-waits (thread 0) for ``signal[u] == u``, then copies chunk u's
    words of ``src`` into ``dst`` with gpu-scope loads, so every chunk is read the moment it is published."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Int32

    @cute.kernel
    def consume(signal: cute.Tensor, src: cute.Tensor, dst: cute.Tensor, words: cute.Tensor, begins: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        u, _, _ = cute.arch.block_idx()
        if tidx == 0:
            seen = Int32(-1)
            while seen != u:
                seen = cute.arch.load((signal.iterator + u).llvm_ptr, Int32, sem="acquire", scope="gpu")
        cute.arch.sync_threads()
        for i in cutlass.range(begins[u] + tidx, begins[u + 1], CONSUMER_THREADS, unroll=1):
            word = words[i]
            dst[word] = cute.arch.load((src.iterator + word).llvm_ptr, Int32, sem="relaxed", scope="gpu")

    @cute.jit
    def launch(
        signal: cute.Tensor,
        src: cute.Tensor,
        dst: cute.Tensor,
        words: cute.Tensor,
        begins: cute.Tensor,
        chunks: cutlass.Constexpr,
        stream,
        use_pdl: cutlass.Constexpr,
    ):
        consume(signal, src, dst, words, begins).launch(grid=(chunks, 1, 1), block=(CONSUMER_THREADS, 1, 1), stream=stream, use_pdl=use_pdl)

    return launch


_CONSUMERS = {}


def chunk_consumer(signal, d, words, begins, use_pdl=False):
    """A launcher of the chunk consumer over ``d``'s words; returns (launch(stream), snapshot): ``snapshot`` is the
    int32 copy each chunk's words land in, -1 bytes where nothing was copied."""
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    src = d.view(torch.int32).view(-1)
    snapshot = torch.full_like(src, -1)
    args = [from_dlpack(x, assumed_align=4) for x in (signal, src, snapshot, words, begins)]
    key = (use_pdl, tuple(tuple(x.shape) for x in (signal, src, words, begins)))
    if key not in _CONSUMERS:
        _CONSUMERS[key] = cute.compile(_consumer_module(), *args, begins.numel() - 1, cu_stream(), use_pdl)

    def launch(stream):
        _CONSUMERS[key](*args, cu_stream(stream))

    return launch, snapshot


# ---- The fake input producer: signal[] -> the GEMM ----


def signal_view(kernel, workspace):
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import streaming_signal_workspace_offsets

    offsets = streaming_signal_workspace_offsets(kernel)
    word = offsets["signal"] // 4
    return workspace.view(torch.int32)[word : word + offsets["streaming_signal_capacity"]]


def poison_inputs(problem, late):
    """NaN in the operands ``late`` names ("a", "sfa" or "both"; SFA in both forms, atoms and compact rows); returns
    their final bytes."""
    final = {name: problem[name].view(torch.uint8).clone() for name in ("a", "sfa", "a_exp") if name in problem}
    if late in ("a", "both"):
        problem["a"].view(torch.uint8).fill_(POISON)
    if late in ("sfa", "both"):
        problem["sfa"].fill_(POISON)
        problem["a_exp"].fill_(POISON)
    return final


def produce_tiles(problem, final, late, signal, order):
    """The fake producer, enqueued on the current stream: per M tile in ``order``, write the late operands' rows and SFA
    bytes (``[t * 256 * Kc, (t + 1) * 256 * Kc)`` of the atoms, and the compact rows), then ``signal[t] = t``."""
    sf_bytes = ROWS_ALIGNMENT * sf_cols(problem["k"])
    for tile in order:
        rows = slice(tile * ROWS_ALIGNMENT, (tile + 1) * ROWS_ALIGNMENT)
        if late in ("a", "both"):
            problem["a"].view(torch.uint8)[rows].copy_(final["a"][rows])
        if late in ("sfa", "both"):
            sf = slice(tile * sf_bytes, (tile + 1) * sf_bytes)
            problem["sfa"][sf].copy_(final["sfa"][sf])
            problem["a_exp"][rows].copy_(final["a_exp"][rows])
        signal[tile].fill_(tile)


def producer_case(kind, variant, routing):
    """The signal GEMM launched last on its stream with every M tile not ready, A (dY) and its scales poisoned and, with
    ``device_layout``, the layout -1; a producer stream writes them tile by tile in order, in reverse and in a seeded
    random order.  Returns (kernel, the stock scheduler's outputs, {order: outputs})."""
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace

    tokens = ROUTINGS[routing]
    problem = make_problem(kind, tokens, seed=7)
    expected, _ = Launcher(stock_kernel(kind, len(tokens)), problem)(problem)
    kernel = make_kernel(kind, len(tokens), **variant_options(kind, problem, variant))
    capacity, late = problem["rows"] // ROWS_ALIGNMENT, "a" if kind == "dglu_bf16" else "both"
    # A few clusters only, so the producer finds free SMs while the GEMM waits.
    launcher = Launcher(kernel, problem, max_active_clusters=4)
    launcher(problem)  # loads the module: under lazy loading a first launch waits for running work
    torch.cuda.synchronize()
    final_layout = problem["layout"].clone()
    gemm_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    results = {}
    for order in ("in_order", "reverse", "random"):
        buffers = launcher.buffers(problem)
        final = poison_inputs(problem, late)
        if kernel.test_device_layout:
            problem["layout"].fill_(-1)
        signal = signal_view(kernel, buffers[-1])
        gemm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gemm_stream):
            reset_streaming_signal_workspace(buffers[-1], kernel)
        producer_stream.wait_stream(gemm_stream)
        # Producer work goes before the GEMM's launch, so no host launch waits for the spinning GEMM; the sleep makes the
        # GEMM wait first.
        with torch.cuda.stream(producer_stream):
            torch.cuda._sleep(20_000_000)
            if kernel.test_device_layout:
                problem["layout"].copy_(final_layout)
            produce_tiles(problem, final, late, signal, release_order(order, capacity))
        launcher.launch(problem, buffers, gemm_stream)
        torch.cuda.synchronize()
        results[order] = {name: value.clone() for name, value in outputs(kernel, problem, buffers).items()}
    return kernel, expected, results
