# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming signal options of the SM100 block-scaled grouped GEMM kernels: N_CHUNK output notify on the MXFP8 quant
kernel (FC2); the input signal on the MXFP8 SwiGLU kernel (FC1), with stock SF atoms or ``compact_sfa_sf_warp``, and on
the BF16 and MXFP8 dSwiGLU kernels.  On SM107 the GLU and MXFP8 dGLU cases run the Rubin twins (stock SF atoms; the
compact and N_CHUNK cases skip there), whose dGLU dprob is an FP32 atomic sum compared within 8 FP32 eps of max |dprob|.

The options reorder tiles, wait and count; they never change the tile math, so every output must be raw-byte equal to
the stock dynamic scheduler's.  Input direction: the producer writes A (dY) and its scales, then releases ``signal[t] =
t`` for the 256-row M tile t (SFA atom rows 2t and 2t + 1, or the compact rows); until then they hold NaN, so a tile read
early carries NaN into D.  Output direction: a concurrent consumer copies each N chunk the moment ``signal[u] == u``; a
chunk published before its rows are in global memory carries the NaN prefill into that copy.

Replays, producer release orders, batch invariance and claim-slot warm stress are in the determinism file.  L0 keeps
one case per behaviour; the remaining shapes, routings and cluster shapes are L1.
"""

import pytest
import torch

from gemm.cutedsl import _streaming_signal_blockscaled as bs
from gemm.cutedsl._streaming_signal_blockscaled import require_sm10x  # noqa: F401  (autouse fixture)

# ---- N_CHUNK output notify (FC2) ----

# (D columns N, N chunk C): whole chunks; N < C (one chunk); N == C; a partial last chunk (columns 512..767).
OUTPUT_CHUNKS = {"n1024_c256": (1024, 256), "n768_c1024": (768, 1024), "n768_c768": (768, 768), "n768_c512": (768, 512)}
# (clusters beside the consumer, one CTA, N and chunk, routing): every chunk shape at both cluster shapes and both
# beside counts, every routing at both cluster shapes.  1 cluster beside the consumer: the extras stop early; 1000: never.
NOTIFY_CASES = [
    (1, False, "n768_c512", "many_experts_64"),
    (1, True, "n1024_c256", "uneven"),
    (1000, False, "n768_c1024", "uneven"),
    (1000, True, "n768_c768", "uneven"),
    (1, False, "n768_c768", "big_expert"),
    (1000, True, "n768_c512", "big_expert"),
    (1, True, "n768_c1024", "many_experts_64"),
    (1000, False, "n1024_c256", "big_expert"),
]


@pytest.mark.parametrize(
    "clusters_beside_consumer, one_cta, n_and_chunk, routing",
    bs.levels(NOTIFY_CASES, NOTIFY_CASES[:3], lambda beside, one_cta, chunk, routing: f"{beside}-{bs.cluster_id(one_cta)}-{chunk}-{routing}"),
)
def test_mxfp8_output_notify_publishes_every_chunk(clusters_beside_consumer, one_cta, n_and_chunk, routing):
    """Every chunk is published (``signal[u] == u``) and D equals the stock dynamic scheduler's bytes."""
    n, chunk = OUTPUT_CHUNKS[n_and_chunk]
    tokens = bs.ROUTINGS[routing]
    problem = bs.make_problem("quant", tokens, n, seed=3)
    stock, _ = bs.Launcher(bs.stock_kernel("quant", len(tokens), one_cta), problem)(problem)
    kernel = bs.notify_kernel(len(tokens), n, chunk, one_cta, clusters_beside_consumer)
    out, workspace = bs.Launcher(kernel, problem)(problem)
    torch.cuda.synchronize()
    bs.assert_same_bytes(out, stock, "N_CHUNK")
    assert bs.chunk_signal(kernel, workspace).tolist() == list(range(len(range(0, n, chunk))))


@pytest.mark.L0
def test_mxfp8_output_notify_matches_float_reference():
    """An independent check of the bytes compared above: D is within 2e-2 of a float32 reference computed from the
    dequantized E4M3 codes and E8M0 scales (partial last chunk, an empty expert, a one-row expert)."""
    tokens = bs.ROUTINGS["uneven"]
    problem = bs.make_problem("quant", tokens, 768, seed=3)
    out, _ = bs.Launcher(bs.notify_kernel(len(tokens), 768, 512), problem)(problem)
    torch.testing.assert_close(out["d"].float(), bs.quant_reference(problem)[: bs.valid_rows(problem)], atol=2e-2, rtol=2e-2)


# The consumer is the GEMM's PDL dependent on its stream (the contract's topology: a consumer on another stream could take
# an SM a not-yet-resident GEMM cluster needs).  full_grid: the GEMM takes every cluster, so the consumer overlaps only
# once extras stop early; spare_sms: the GEMM leaves SMs free, so the consumer overlaps either way.  The GEMM is the PDL
# dependent of its own reset (waits for it at entry), except with caller_reset, where the caller resets signal[].
# Every topology with both beside counts and both cluster shapes; L0 keeps one cluster beside the consumer, where a chunk
# published early is read soonest.
CONSUMER_CASES = [
    ("full_grid", 1, False),
    ("spare_sms", 1, True),
    ("caller_reset", 1, True),
    ("full_grid", 1000, True),
    ("spare_sms", 1000, False),
    ("caller_reset", 1000, False),
]
SPARE_CLUSTERS = 4


@pytest.mark.parametrize(
    "topology, clusters_beside_consumer, one_cta",
    bs.levels(CONSUMER_CASES, CONSUMER_CASES[:3], lambda topology, beside, one_cta: f"{topology}-{beside}-{bs.cluster_id(one_cta)}"),
)
def test_mxfp8_output_notify_consumer_reads_each_published_chunk(topology, clusters_beside_consumer, one_cta):
    """A consumer kernel copies chunk u's D words as soon as it acquires ``signal[u] == u``, while the GEMM still runs.
    Each copy equals the final D, so no chunk was published before every row of it was in global memory."""
    from cutlass.utils import HardwareInfo

    n, chunk = 1024, 256
    tokens = bs.ROUTINGS["many_experts_64"]
    problem = bs.make_problem("quant", tokens, n, seed=5)
    stock, _ = bs.Launcher(bs.stock_kernel("quant", len(tokens), one_cta), problem)(problem)
    caller_reset = topology == "caller_reset"
    kernel = bs.notify_kernel(len(tokens), n, chunk, one_cta, clusters_beside_consumer, caller_reset=caller_reset, use_pdl=not caller_reset)
    clusters = HardwareInfo().get_max_active_clusters(1 if one_cta else 2)
    launcher = bs.Launcher(kernel, problem, max_active_clusters=clusters if topology == "full_grid" else clusters - SPARE_CLUSTERS)
    rows = bs.valid_rows(problem)
    words, begins = bs.chunk_words(rows, n, chunk)
    stream = torch.cuda.Stream()
    for attempt in range(2):
        d, workspace = launcher.buffers(problem)
        signal = bs.chunk_signal(kernel, workspace)
        consume, copied = bs.chunk_consumer(signal, d, words, begins, use_pdl=True)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if caller_reset:
                signal.fill_(-1)
            launcher.launch(problem, (d, workspace), stream)
            consume(stream)
        torch.cuda.synchronize()
        bs.assert_same_bytes({"d": d[:rows]}, stock, f"attempt {attempt}")
        final = d.view(torch.int32).view(-1)
        early = torch.count_nonzero(copied[words.long()] != final[words.long()]).item()
        assert early == 0, f"attempt {attempt}: {early} of {words.numel()} D words read at publication differ from the final D"
        assert signal.tolist() == list(range(begins.numel() - 1))


@pytest.mark.L0
def test_mxfp8_output_notify_rejects_invalid():
    """N_CHUNK counts BF16 D's bulk stores only: SFD, a static scheduler, discrete weights and other cluster shapes are
    rejected at construction; amax and an FP8 D at trace time."""
    import cutlass.cute as cute
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode
    from cutlass.cute.runtime import from_dlpack

    with pytest.raises(ValueError, match="use_dynamic_sched"):
        bs.notify_kernel(4, 1024, 256, use_dynamic_sched=False)
    with pytest.raises(ValueError, match="BF16 D"):
        bs.notify_kernel(4, 1024, 256, generate_sfd=True)
    with pytest.raises(ValueError, match="dense weights"):
        bs.notify_kernel(4, 1024, 256, weight_mode=MoEWeightMode.DISCRETE)
    with pytest.raises(ValueError, match="cluster"):
        bs.notify_kernel(4, 1024, 256, cluster_shape_mn=(4, 1))
    problem = bs.make_problem("quant", [256, 256, 0, 0])
    kernel = bs.notify_kernel(4, 1024, 256)
    d, workspace = bs.new_buffers(kernel, problem)
    args = bs.kernel_args(kernel, problem, d, workspace)
    amax = from_dlpack(torch.zeros(4, 1, device="cuda"), assumed_align=16)
    with pytest.raises(Exception, match="no scheduler_counter, amax or SFD"):
        cute.compile(kernel, *args[: bs.AMAX_ARG], amax, *args[bs.AMAX_ARG + 1 :], 4, bs.cu_stream())
    d_fp8 = torch.zeros(problem["rows"], problem["n"], device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
    with pytest.raises(Exception, match="requires BF16 D"):
        cute.compile(kernel, *bs.kernel_args(kernel, problem, d_fp8, workspace), 4, bs.cu_stream())


NO_ROWS_CASES = [("quant", "cluster21"), ("glu", "trigger"), ("quant", "cluster11"), ("glu", "signal")]


@pytest.mark.parametrize("kind, variant", bs.levels(NO_ROWS_CASES, NO_ROWS_CASES[:2], lambda kind, variant: f"{bs.KIND_IDS[kind]}-{variant}"))
def test_with_no_rows_exits(kind, variant):
    """A GEMM with no rows exits; N_CHUNK publishes nothing (signal[] keeps its reset value)."""
    problem = bs.make_problem(kind, [0, 0, 0, 0])
    if kind == "quant":
        kernel = bs.notify_kernel(4, 1024, 256, variant == "cluster11")
    else:
        kernel = bs.make_kernel(kind, 4, **bs.variant_options(kind, problem, variant))
    _, workspace = bs.Launcher(kernel, problem)(problem)
    torch.cuda.synchronize()
    if kind == "quant":
        assert bs.chunk_signal(kernel, workspace).tolist() == [-1] * 4


# ---- The input signal (FC1 GLU, dGLU) ----

# (kind, variant, routing) beside the replay determinism cases (64 experts, a 5000-row expert): the uneven routing (an
# empty and a one-row expert), the compact SF warp on the stock dynamic scheduler, the dGLU's entry trigger alone, and
# 64 small experts on the signal scheduler (batch openings that publish several batches).
MATCH_CASES = [
    ("glu", "signal", "uneven"),
    ("glu", "trigger", "uneven"),
    ("glu", "compact", "big_expert"),
    ("glu", "compact", "many_experts_64"),
    ("glu", "compact", "uneven"),
    ("glu", "compact_signal", "uneven"),
    ("glu", "compact_trigger", "uneven"),
    ("dglu_bf16", "entry", "uneven"),
    ("dglu", "entry", "uneven"),
    ("dglu_bf16", "signal", "big_expert"),
    ("dglu_bf16", "trigger", "uneven"),
    ("dglu", "signal", "uneven"),
    ("dglu", "trigger", "big_expert"),
    ("dglu", "compact_signal", "big_expert"),
    ("dglu", "compact_trigger", "uneven"),
    ("glu", "signal", "small_experts_64"),
    ("glu", "compact_signal", "small_experts_64"),
    ("dglu_bf16", "signal", "small_experts_64"),
    ("dglu", "signal", "small_experts_64"),
    ("dglu", "compact_signal", "small_experts_64"),
]
MATCH_L0 = [
    ("glu", "trigger", "uneven"),
    ("glu", "compact", "big_expert"),
    ("dglu_bf16", "entry", "uneven"),
    ("dglu", "entry", "uneven"),
    ("dglu", "compact_trigger", "uneven"),
    ("glu", "signal", "small_experts_64"),
    ("dglu_bf16", "signal", "small_experts_64"),
    ("dglu", "signal", "small_experts_64"),
]


@pytest.mark.parametrize("kind, variant, routing", bs.levels(MATCH_CASES, MATCH_L0, lambda kind, variant, routing: f"{bs.KIND_IDS[kind]}-{variant}-{routing}"))
def test_signal_matches_dynamic(kind, variant, routing):
    """Every M tile ready at launch: the outputs equal the stock dynamic scheduler's bytes."""
    tokens = bs.ROUTINGS[routing]
    problem = bs.make_problem(kind, tokens, seed=len(routing))
    expected, _ = bs.Launcher(bs.stock_kernel(kind, len(tokens)), problem)(problem)
    kernel = bs.make_kernel(kind, len(tokens), **bs.variant_options(kind, problem, variant))
    got, _ = bs.Launcher(kernel, problem)(problem)
    torch.cuda.synchronize()
    bs.assert_same_bytes(got, expected, f"{kind} {variant}")
    assert not torch.isnan(got["d"].float()).any()


POISON_CASES = [
    ("glu", "a", False),
    ("glu", "sfa", False),
    ("glu", "sfa", True),
    ("dglu_bf16", "a", False),
    ("dglu", "a", False),
    ("dglu", "sfa", False),
    ("dglu", "sfa", True),
]


@pytest.mark.L0
@pytest.mark.parametrize("kind, late, compact", POISON_CASES, ids=[f"{bs.KIND_IDS[k]}-late_{late}{'-compact' * c}" for k, late, c in POISON_CASES])
def test_poison_reaches_d(kind, late, compact):
    """The detector works: with A (dY) codes, SFA atoms or compact SFA rows poisoned, the stock scheduler's D holds NaN,
    so the producer tests see an early read of any of them."""
    tokens = bs.ROUTINGS["uneven"]
    problem = bs.make_problem(kind, tokens, seed=1)
    bs.poison_inputs(problem, late)
    out, _ = bs.Launcher(bs.make_kernel(kind, len(tokens), use_dynamic_sched=True, compact_sfa_sf_warp=compact), problem)(problem)
    torch.cuda.synchronize()
    assert torch.isnan(out["d"].float()).any(), f"poisoned {late} left D finite"


@pytest.mark.L1
@pytest.mark.parametrize("kind", ["glu", "dglu"], ids=bs.KIND_IDS.get)
def test_signal_waits_for_producer_big_expert(kind):
    """The determinism file's producer cases on an expert of several batches (5000 rows): every release order gives the
    stock bytes."""
    _, expected, results = bs.producer_case(kind, "signal", "big_expert")
    for order, got in results.items():
        bs.assert_same_bytes(got, expected, f"{order} release")


def rejected(kind, compact):
    """(options, exception, message) of the rejected input-signal constructions of ``kind``."""
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    if compact:
        one_cta = dict(use_2cta_instrs=False, mma_tiler_mn=(128, 256), cluster_shape_mn=(1, 1))
        options = [one_cta] + ([dict(sf_vec_size=16), dict(glu_layout="gate_then_up")] if kind == "glu" else [])
        return [(dict(use_dynamic_sched=True, compact_sfa_sf_warp=True, **extra), ValueError, "compact_sfa_sf_warp") for extra in options]
    signal = dict(use_dynamic_sched=True, streaming_signal_capacity=4)
    cases = [
        (dict(streaming_signal_capacity=4), ValueError, "use_dynamic_sched"),
        (dict(signal, weight_mode=MoEWeightMode.DISCRETE), ValueError, "dense weights"),
        (dict(signal, cluster_shape_mn=(4, 1)), ValueError, "cluster"),
    ]
    slots_ready = "launch_pdl_dependents_at_all_streaming_signal_slots_ready"
    if kind == "glu":
        return cases + [(dict(use_dynamic_sched=True, **{name: True}), ValueError, name) for name in ("device_layout", slots_ready)]
    # The forward GLU's device layout and slots-ready trigger are not dGLU options.
    return cases + [(dict(signal, **{name: True}), TypeError, name) for name in ("device_layout", slots_ready)]


REJECT_CASES = [("glu", False), ("glu", True), ("dglu_bf16", False), ("dglu", False), ("dglu", True)]


@pytest.mark.L0
@pytest.mark.parametrize("kind, compact", REJECT_CASES, ids=[f"{bs.KIND_IDS[k]}{'-compact' * c}" for k, c in REJECT_CASES])
def test_input_signal_rejects_invalid(kind, compact):
    for options, error, message in rejected(kind, compact):
        with pytest.raises(error, match=message):
            bs.make_kernel(kind, 4, **options)
    if kind == "glu" and compact:
        # K = 384 is a whole number of SF atoms but not of the SF warp's 4-K-tile loads.
        import cutlass.cute as cute

        problem = bs.make_problem("glu", [256, 0, 0, 0], k=384)
        kernel = bs.make_kernel("glu", 4, use_dynamic_sched=True, compact_sfa_sf_warp=True)
        with pytest.raises(Exception, match="K % 512"):
            cute.compile(kernel, *bs.kernel_args(kernel, problem, *bs.new_buffers(kernel, problem)), 4, bs.cu_stream())


@pytest.mark.L0
def test_pdl_rejected_behind_helper_kernel():
    """use_pdl relaxes the dependency on the launch right before the GEMM, which waits for it only at exit.  When that
    launch is the GEMM's own helper_kernel (discrete B descriptors, dynamic claim counter reset) the kernel would read
    them unsynchronized: rejected at construction where the constructor can tell, else (the GLU and quant kernels, whose
    helper also depends on ``scheduler_counter``) at trace time.  The input signal and the quant kernel's own N_CHUNK
    reset keep use_pdl.  On SM107 the GLU and dGLU are the Rubin twins; the quant kernel takes no N_CHUNK there."""
    import cutlass.cute as cute
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    kinds = ("glu", "dglu_bf16", "dglu") if bs.is_rubin() else ("glu", "dglu_bf16", "dglu", "quant")
    discrete = dict(use_dynamic_sched=True, weight_mode=MoEWeightMode.DISCRETE, use_pdl=True)
    for kind in kinds:
        with pytest.raises(ValueError, match="helper_kernel"):
            bs.make_kernel(kind, 4, **discrete)
    for kind in ("dglu_bf16", "dglu"):
        with pytest.raises(ValueError, match="helper_kernel"):
            bs.make_kernel(kind, 4, use_dynamic_sched=True, use_pdl=True)
        bs.make_kernel(kind, 4, use_dynamic_sched=True, streaming_signal_capacity=4, use_pdl=True)
    if not bs.is_rubin():
        with pytest.raises(ValueError, match="helper_kernel"):
            bs.notify_kernel(4, 1024, 256, caller_reset=True, use_pdl=True)
        bs.notify_kernel(4, 1024, 256, use_pdl=True)
    bs.make_kernel("glu", 4, use_dynamic_sched=True, streaming_signal_capacity=4, use_pdl=True)
    # The dynamic scheduler with no scheduler_counter: the helper resets the claim counter, known only at trace time.
    for kind in [kind for kind in ("glu", "quant") if kind in kinds]:
        problem = bs.make_problem(kind, [256, 256, 0, 0])
        kernel = bs.make_kernel(kind, 4, use_dynamic_sched=True, use_pdl=True)
        with pytest.raises(ValueError, match="helper_kernel"):
            cute.compile(kernel, *bs.kernel_args(kernel, problem, *bs.new_buffers(kernel, problem)), 4, bs.cu_stream())
