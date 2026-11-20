# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the performance mode of the model samples (``samples/models/model_samples/perf.py``; ``--perf`` of
``samples/models/qwen_dense/00_decode.py``).

* every statistic on constructed time stamps: the time per output token, the rate, the nearest-rank percentiles, the run's
  value as the median over its timed requests, the time to first token, and the step and the loop's host time at the
  traced positions;
* the GPU side on constructed launch records: GPU busy as the union of a step's intervals, the step taken apart and the
  rule for printing launch overhead as a term, tracing's stretch; the rows of the per-kernel table by the place in the
  wiring, the time a launch adds to the GPU's timeline, the rows against GPU busy, the resolution of the stamps; the
  launch records of ``--dump-census`` with their start and end;
* the judged positions from the request and the page of the attention cache; the reference checks at judged positions
  through ``tie_out.run_checks``; the record class from its conditions;
* the protocol through a real ``Session`` on the CPU with stand-in providers and a stand-in profiler: no profiler pass
  before the last timed request, nothing but the step inside a timed request, the three statements of the ``checks``
  lines, and what a statement that does not hold does to the rate line and to the last line;
* the screen (no line wider than 110 characters, in every form) and the report file (strict JSON, a definition beside
  every statistic, no local path; its schema, and a file of another schema -- one of each of the two before it among
  them -- refused in one sentence that names both);
* the sample's flags, each refusal before the load, ``main()`` with ``--perf`` to its last line, and an ordinary run
  that prints what it printed and does not import the mode's module;
* the session of a run, built for its longest request: a warm-up request longer than the timed one runs to its report,
  the context is held against it before the load, and nothing a request computes moves with the session's size;
* the public text of the mode: a sentence of the READMEs, of the guide or of the script that says what the mode does is
  held beside the assertion that shows it (``_says``), and so are the definitions the report file carries
  (``_defines``); the words, the ``stage`` row, the flag table and the example of the report against the code;
* the observation of the GPU: on constructed samples the cut by the bounds of a request, the own-process test behind
  ``none seen``, the memory rule, every form of the clock line; through a stand-in for the driver's tool, written by the
  test, a reading, a tool that fails, and the sampler's child process whose loops end with its pipe; through a session,
  every call of the observer against the stamps of the requests, and what the file's ``protocol`` says of the sampler.
  No test asks the driver of the host it runs on.

No GPU, no weights on disk, no profiler."""

import ast
import contextlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cudnn")

pytestmark = [pytest.mark.L0]

MODELS = Path(__file__).resolve().parents[3] / "samples" / "models"
SCRIPT = MODELS / "qwen_dense" / "00_decode.py"
GIB = 1024**3
# kernel names as the profiler reports them (heads of the names of one traced default run; a name is matched by its head)
SYMBOL = (
    "cudnn_kernel_frost_gdn_decode_fused_CfgTHREADS256TPR4K_RESIDENTTrueWARPGROUP_SPLITTrueCTAS_PER_SM0STATE_LOAD_PATHbulkSTATE_STREAMFalse"
    "MIN_CTAS_PER_SM1PDLFalsePDL_RELEASEFalsePIN_ENTRY_L_0"
)
OTHER_SYMBOL = SYMBOL.replace("PDLFalsePDL_RELEASEFalsePIN_ENTRY_L_0", "PDLTruePDL_RELEASETruePIN_ENTRY_LOA_0")  # another form of the same kernel
GATHER = "void at::native::(anonymous namespace)::indexSelectSmallIndex<BFloat16, int, unsigned int, 2, 2, -2>(at::cuda::detail::TensorInfo<...>)"
ELEMENTWISE = "void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<double>, std::array<char*, 2ul> >(int, ...)"
ARGMAX = "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<BFloat16, at::native::ArgMaxOps<BFloat16>, unsigned int, long, 4, 4> >(...)"
SOFTMAX = "void (anonymous namespace)::softmax_warp_forward<double, double, double, 3, false, false, 32>(double*, double const*, int, int, int, ...)"
ARANGE = "void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(...)::{lambda(long)#1}>(int, ...)"
NVJET = "nvjet_sm100_tst_64x8_64x16_2x1_v_bz_splitK_TNT"
# two kernels a library launches for a row of the stack native: a cuBLASLt kernel of 8-bit operands, and the library's 4-bit multiply
NVJET_FP8 = "nvjet_sm100_qqtst_64x8_128x16_4x1_v_bz_TNT"
FI_MM_FP4 = "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100_splitkSm100BlockScaledSplitKGemmKernel_object_at__TiledMMA_0"
HOST_READ = "Memcpy DtoH (Device -> Pageable)"
NO_PROFILER = "this PyTorch build records no kernel activity"


@pytest.fixture(scope="module", autouse=True)
def helper_layer():
    """``samples/models`` on ``sys.path`` for this module only: the path and every ``model_samples`` module leave with it."""
    sys.path.insert(0, str(MODELS))
    yield
    sys.path.remove(str(MODELS))
    for name in [n for n in sys.modules if n == "model_samples" or n.startswith("model_samples.")]:
        del sys.modules[name]


_TEXTS = {"area": MODELS / "README.md", "guide": MODELS / "AGENTS.md", "family": MODELS / "qwen_dense" / "README.md", "script": SCRIPT}


def _flat(text):
    return " ".join(text.split())


def _says(where, *sentences):
    """The public text ``where`` holds every one of ``sentences``, white space aside (of the script: its adjacent string
    literals read as one).  Called beside the assertion that shows the code does what the sentence says."""
    text = _TEXTS[where].read_text(encoding="utf-8")
    text = _flat(re.sub(r'"\s*\n\s*f?"', "", text) if where == "script" else text)
    for sentence in sentences:
        assert _flat(sentence) in text, f"{_TEXTS[where].name} does not say: {sentence}"


def _defines(key, *phrases):
    """The definition the report file carries beside the statistic ``key`` holds every one of ``phrases``."""
    from model_samples import perf

    for phrase in phrases:
        assert phrase in perf.DEFINITIONS[key][1], f"{key}: {phrase}"


class _Nth:
    """True the n-th time it is asked and false every other time: a defect of one step (``defect={"token": _Nth(n)}``: the
    n-th token choice of a session returns another id)."""

    def __init__(self, n):
        self.n, self.asked = n, 0

    def __bool__(self):
        self.asked += 1
        return self.asked == self.n


def _request(first, tokens, launches=None, request=0.0, kind="timed", ids=None):
    """A request with constructed stamps, in seconds: ``t_request``, ``t_first``, ``t_token`` and, where given, ``t_launch``
    (default: each step is launched 1 ms behind the read before it)."""
    from model_samples import perf

    reads = [first, *tokens]
    launches = [read + 0.001 for read in reads[:-1]] if launches is None else launches
    return perf.Request(kind, request, first, list(launches), list(tokens), ids or [[7]] * len(reads))


# ---------------------------------------------------------------------------------------------- statistics on constructed stamps
def test_time_per_output_token_is_last_minus_first_over_tokens_minus_one():
    """(the last token's ids on the host - the first new token's) / (new tokens - 1): four forced steps are five new tokens
    and four intervals.  The time before the first new token, however long, is not in it."""
    from model_samples import perf

    request = _request(first=10.0, tokens=[10.010, 10.030, 10.060, 10.100], request=2.0)
    assert len(request.ids) == 5 and len(request.intervals_ms()) == 4
    assert request.time_per_output_token_ms() == pytest.approx((10.100 - 10.0) * 1e3 / 4)
    assert request.time_per_output_token_ms() == pytest.approx(sum(request.intervals_ms()) / 4)
    late = _request(first=60.0, tokens=[60.010, 60.030, 60.060, 60.100], request=2.0)
    assert late.time_per_output_token_ms() == pytest.approx(request.time_per_output_token_ms()), "a longer prompt moves no interval"
    stats = perf._statistics([request], 0, 1, "")
    assert stats["time_per_output_token"]["value"] == pytest.approx(25.0) and stats["time_per_output_token"]["events"] == "t_token[last] - t_first"
    _defines("time_per_output_token", "ids reach the host) / (new tokens - 1). Of a run: the median over its timed requests.")
    _says("area", "time per output token = (the last token's ids on the host - the first new token's ids on the host) / (new tokens - 1) of a timed request")


def test_rate_is_never_a_median_step():
    """The rate is 1000 / time per output token.  With one long step among short ones the median step would say 100 tokens
    a second; the request delivered its tokens at 40."""
    from model_samples import perf

    request = _request(first=0.0, tokens=[0.010, 0.020, 0.030, 0.100])  # intervals 10, 10, 10, 70 ms
    stats = perf._statistics([request], 0, 1, "")
    assert stats["token_to_token_interval"]["median"] == pytest.approx(10.0)
    assert stats["decode_rate"]["value"] == pytest.approx(1000 / 25.0) and stats["decode_rate"]["value"] != pytest.approx(1000 / 10.0)
    # not output tokens over the wall time of the request either: that figure holds the time to first token and is file-only
    slow_start = _request(first=1.0, tokens=[1.010, 1.020, 1.030, 1.100])
    wall = perf._statistics([slow_start], 0, 1, "")
    assert wall["decode_rate"]["value"] == pytest.approx(40.0) and wall["tokens_per_second_wall"]["value"] == pytest.approx(5 / 1.1)
    # several sequences: the total beside the rate per sequence; a rate that is not printed has neither
    assert perf._statistics([request], 0, 4, "")["decode_rate"]["total"] == pytest.approx(160.0)
    unprinted = perf._statistics([request], 0, 4, "not printed for a reduced model")
    missing, wall = unprinted["decode_rate"], unprinted["tokens_per_second_wall"]
    assert missing["value"] is None and missing["total"] is None and missing["reason"] == "not printed for a reduced model"
    assert wall["value"] is None and wall["reason"] == "not printed for a reduced model", "no figure of a rate where no rate is printed"
    _defines("decode_rate", "1000 / time per output token in ms. Never a median step turned into a rate.")
    _defines("tokens_per_second_wall", "new tokens / (time the last token's ids reach the host - time the request is handed to the text path)")
    _defines("tokens_per_second_wall", "It holds the time to first token and is not the decode rate; where no decode rate is printed it is null too.")
    _says("area", "| `decode rate` | 1000 / time per output token, in tokens per second per sequence;")
    _says(
        "area", "Never a median step turned into a rate, and not output tokens / wall time, which holds every time to first token (that figure is in the file)."
    )


def test_percentiles_nearest_rank():
    """The sample of rank ceil(p n / 100) of the n sorted samples: never an interpolated value."""
    from model_samples import perf

    samples = [float(n) for n in range(1, 11)]
    assert perf.nearest_rank(samples, 90) == 9.0 and perf.nearest_rank(samples, 99) == 10.0 and perf.nearest_rank(samples, 50) == 5.0
    assert perf.nearest_rank(list(reversed(samples)), 90) == 9.0, "sorted here"
    assert perf.nearest_rank([3.0], 99) == 3.0 and perf.nearest_rank([1.0, 2.0], 1) == 1.0
    hundred = [float(n) for n in range(100)]
    assert perf.nearest_rank(hundred, 99) == 98.0 and perf.nearest_rank(hundred, 100) == 99.0
    pooled = perf.summary([1.0, 2.0, 3.0, 4.0, 100.0])
    assert pooled == {"n": 5, "median": 3.0, "mean": 22.0, "minimum": 1.0, "maximum": 100.0, "p90": 100.0, "p99": 100.0}
    assert perf.summary([]) == {"n": 0}
    _defines("token_to_token_interval", "pooled over the run's timed requests; percentiles are nearest-rank.")
    _says("area", "pooled over the timed requests: median, p90 and p99 by nearest rank.")
    _defines("token_to_token_interval", "The intervals span a growing attention cache as well as jitter")
    _says("area", "Not a service's tail latency: the intervals span a growing attention cache as well as jitter")
    assert perf.DEFINITIONS["token_to_token_interval"][2] == "None: a percentile of this loop is not a service's tail latency."


def test_run_value_is_median_over_timed_requests():
    """Of a run: the median over its timed requests, never the pooled mean; each request and the spread are in the file."""
    from model_samples import perf

    timed = [_request(0.0, [0.010, 0.020]), _request(0.0, [0.012, 0.024]), _request(0.0, [0.050, 0.100])]  # 10, 12 and 50 ms per token
    stats = perf._statistics(timed, 0, 1, "")
    per_token = stats["time_per_output_token"]
    assert per_token["per_request"] == pytest.approx([10.0, 12.0, 50.0]) and per_token["value"] == pytest.approx(12.0) and per_token["n"] == 3
    assert per_token["mean"] == pytest.approx(24.0) and per_token["spread_percent"] == pytest.approx((50.0 - 10.0) / 12.0 * 100)
    assert stats["decode_rate"]["value"] == pytest.approx(1000 / 12.0)
    assert stats["token_to_token_interval"]["n"] == 6, "the intervals are pooled over the requests"
    _says("area", "of a timed request, and of a run the median over its timed requests.", "The line also gives how far the timed requests are apart.")
    _says("family", "the time per output token and the time to first token of a run are the median over them, the token-to-token intervals are pooled")


def test_time_to_first_token_starts_at_the_request():
    """From the request being handed to the text path to the first new token's ids on the host; of a run the median."""
    from model_samples import perf

    timed = [
        _request(first=5.8, tokens=[5.9], request=0.0),
        _request(first=16.0, tokens=[16.1], request=10.0),
        _request(first=27.0, tokens=[27.1], request=20.0),
    ]
    assert [request.time_to_first_token_ms() for request in timed] == pytest.approx([5800.0, 6000.0, 7000.0])
    first = perf._statistics(timed, 0, 1, "")["time_to_first_token"]
    assert (
        first["value"] == pytest.approx(6000.0) and first["events"] == "t_first - t_request" and first["per_request"] == pytest.approx([5800.0, 6000.0, 7000.0])
    )
    assert "The load is outside it" in first["definition"] and "without the network" in first["serving_counterpart"]
    _defines("time_to_first_token", "from the request being handed to the text path to the first new token's ids on the host", "Of a run: the median over")
    _says("area", "and the host read; the median over the timed requests. The load is outside it.")
    same = "prefills the prompt in one pass and adds its queue and its network path"
    assert f"not the same quantity: a serving engine {same}." in first["serving_counterpart"]
    _says("area", f"A serving engine {same}: the same pair of events, not the same quantity")


def test_step_is_the_median_interval_at_the_traced_positions():
    """step = the median token-to-token interval over the timed steps at the positions the traced request covers, all
    timed requests; host = from a read to the next step's first launch call, at the same positions.  Where no traced step
    was observed both are taken over every timed step."""
    from model_samples import perf

    # ten-millisecond steps, each launched 1 ms behind the read before it; the fourth step of each request is longer
    timed = [_request(0.0, [0.010, 0.020, 0.030, 0.090], launches=[0.001, 0.011, 0.021, 0.035]), _request(0.0, [0.010, 0.020, 0.030, 0.090])]
    three = perf._statistics(timed, 3, 1, "")
    assert three["step"]["value"] == pytest.approx(10.0) and three["step"]["n"] == 6 and three["step"]["maximum"] == pytest.approx(10.0)
    assert three["host"]["value"] == pytest.approx(1.0) and three["host"]["n"] == 6
    four = perf._statistics(timed, 4, 1, "")  # the long last step is among the traced positions now
    assert four["step"]["n"] == 8 and four["step"]["maximum"] == pytest.approx(60.0) and four["step"]["value"] == pytest.approx(10.0)
    assert four["host"]["n"] == 8 and four["host"]["maximum"] == pytest.approx(5.0), "the fourth step of the first request: 5 ms behind the read before it"
    every = perf._statistics(timed, 0, 1, "")
    assert every["step"]["n"] == 8 and every["step"]["value"] == every["token_to_token_interval"]["value"] == pytest.approx(10.0)
    assert three["step"]["events"] == "t_token[i] - t_token[i - 1] at the traced positions"
    assert three["gpu_busy"]["value"] is None and three["gpu_busy"]["reason"] == "GPU busy not measured (no traced step was observed)", "no record handed in"
    assert three["launch_overhead"]["value"] is None and three["launch_overhead"]["remainder_ms"] is None and three["tracing_stretch"]["value"] is None
    _defines("host", "From a token's ids on the host to the next step's first launch call", "the median over the timed steps at the traced positions")
    _defines("step", "The median token-to-token interval over the timed steps at the positions the traced request covers, all timed requests.")
    _says("area", "| `step` | the median token-to-token interval over the timed steps at the positions the traced request covers (the line names them).")
    _says("area", "host = the loop's own time from a token's ids on the host to the next step's first launch call;")
    assert three["step"]["clock"] == three["host"]["clock"] == perf.CLOCK and "clock" not in three["memory"] and "clock" not in three["launches_per_step"]


def test_judged_positions_from_page_and_request():
    """Positions 0 to 15, the two on either side of the first page boundary that has both inside the timed steps, and the
    last two timed positions; the default prompt of 58 tokens with 256 timed steps and a page of 64 positions gives
    0-15, 63, 64, 312, 313.  The family names the page."""
    from model_samples import perf, tie_out
    from model_samples.qwen_dense import QwenDense, configs

    def listed(prompt, steps, page=64):
        return tie_out.positions_text([position for position, _ in perf.judged_positions(prompt, steps, page)])

    assert QwenDense.attention_page == configs.ATTENTION_PAGE == 64
    assert listed(58, 256) == "0-15, 63, 64, 312, 313"
    assert listed(64, 256) == "0-15, 127, 128, 318, 319", "position 63 is a prompt position here: the next boundary has both sides timed"
    assert listed(63, 256) == "0-15, 63, 64, 317, 318" and listed(65, 256) == "0-15, 127, 128, 319, 320"
    assert listed(9, 8) == "0-16", "no boundary inside the timed steps 9 to 16: the first 16 and the last two"
    assert listed(58, 4) == "0-15, 60, 61" and listed(58, 256, None) == "0-15, 312, 313", "a family that names no page"
    assert listed(3, 2) == "0-4" and listed(62, 3) == "0-15, 63, 64", "the boundary's two and the last two are the same positions"
    why = dict(perf.judged_positions(58, 256, 64))
    assert why[0] == "one of the first 16 positions of the request" and why[313] == "one of the last two timed positions"
    assert why[63] == why[64] == "on either side of the first page boundary of the attention cache inside the timed steps (64 positions a page)"
    assert all(58 <= position <= 313 for position in why if position > 15), "every judged position behind the first 16 is a timed one"
    assert sum(position > 15 for position in why) == 4
    _says("script", "with the reference checks on it, also at four positions of the timed steps")
    _says("family", "one checked request with the reference checks on it (also at four positions of the timed steps)")
    _says("area", "The judged positions are the first 16 of the request, the two on either side of the first page boundary of the attention cache inside")
    _says("area", f"for the default prompt of 58 tokens, {listed(58, 256).replace(', 313', ' and 313')}.", f"A page is {configs.ATTENTION_PAGE} positions,")
    _says("area", "the attention blocks of this sample read one contiguous cache today, and the two positions are judged all the same")
    assert tie_out.positions_text([5, 3, 4, 9, 10, 12]) == "3-5, 9, 10, 12" and tie_out.positions_text([]) == ""


def test_record_class_reasons_in_the_order_of_the_conditions():
    """A run is a controlled run only when every condition is met; otherwise orientation, with every reason, in order."""
    from model_samples import perf

    kind, reasons = perf.record_class(True, True, "no target kernel set declared", False, "kernels not observed")
    assert kind == "orientation"
    assert reasons == [
        "reduced model",
        "fallback rows present",
        "no target kernel set declared",
        "launched from Python",
        "kernels not observed",
        "other processes not observed",
        "clocks not observed",
    ]
    today = perf.record_class(False, True, "kernel identity not claimed on this GPU class", False, "")
    assert today == (
        "orientation",
        [
            "fallback rows present",
            "kernel identity not claimed on this GPU class",
            "launched from Python",
            "other processes not observed",
            "clocks not observed",
        ],
    )
    assert perf.record_class(False, False, "", True, "") == ("orientation", ["other processes not observed", "clocks not observed"])
    assert perf.record_class(False, False, "", True, "a check did not hold", "", "")[1] == ["a check did not hold"]
    assert perf.record_class(False, False, "", True, "", "", "") == ("controlled", []), "every condition handed in as met"
    # the third reason, from the kernel facts of a run: no target declared; a declared target whose census line is not EQUAL
    census_line = "target census plain / cc 10.0 / 148 SMs (recorded <date>): "
    assert perf.identity_reason({"declared": False, "census": None}) == "no target kernel set declared"
    assert perf.identity_reason({"declared": True, "census": census_line + "EQUAL (three numbers, 0 by membership alone)"}) == ""
    for other in ("NOT CLAIMED (fallback rows present)", "NOT EQUAL (the kernel check did not hold)", "NOT OBSERVED", "not compared"):
        assert perf.identity_reason({"declared": True, "census": census_line + other}) == "kernel identity not claimed on this GPU class", other
    _says("area", "A run is a controlled run only when it observed every condition itself: a checkpoint, every block on a claimed target kernel,")
    _says("area", "Otherwise it is `orientation` and the line lists every reason, in that order, and the rate line reads `decode rate (orientation)`.")
    _says("area", "*Record class*: what a run may call its numbers, on the second line of its report: `orientation` with every reason, or a controlled run.")
    _says("family", "Every run of the mode is `orientation` today and says why on the second line of its report")
    _says("family", "its blocks are launched from Python, a fallback row is left in every stack, and whatever the run saw of the GPU's other")
    _says("family", "whatever the run saw of the GPU's other processes and of its clock that a controlled run may not have is a reason too.")
    _says("area", "and no stack of this area is captured: every run today is `orientation`.")


# ---------------------------------------------------------------------------------------------- a session on the CPU, stand-in providers
def _tiny_model():
    from model_samples.qwen_dense import Geometry, QwenDense

    geometry = Geometry(
        layer_kinds=("linear_attention", "full_attention"),
        hidden=64,
        vocab=272,
        mlp=96,
        la_key_heads=1,
        la_value_heads=2,
        la_head_dim=8,
        attn_heads=2,
        attn_kv_heads=1,
        attn_head_dim=8,
        rotary_dim=4,
    )
    return QwenDense(geometry, reductions=("tiny geometry",))


def _stand_ins(monkeypatch, events, profiler="records", defect=None):
    """The provider factories of a device that exists only as a log, for the rows of the stack ``fe-cublas``: every provider
    runs the reference's arithmetic and writes the launches a provider of its row would make into the log; a stand-in
    profiler (it replaces ``loop.trace_steps``) hands the log over as the launch records of a traced pass and writes
    ``"profiler"`` into ``events``.  ``profiler``: ``"records"``, or the reason it cannot run.  ``defect`` is a dict a
    test switches while the run goes on: ``"symbol"`` -- the fused step launches this kernel in the place of its own;
    ``"token"`` -- the token choice returns the id behind the greedy one; ``"at the first profiler pass"`` -- defects that
    begin with the traced steps."""
    from model_samples import loop
    from model_samples.census import Facts, Kernel, Launch
    from model_samples.providers import factories, torch_ref
    from model_samples.timing import Trace

    log, defect = [], ({} if defect is None else defect)

    def launch(name, shape=((1, 1, 1), (128, 1, 1), 0), kind="kernel"):
        at = float(len(log))
        log.append(Launch(name, kind, at, at + 0.5, bytes=4) if kind == "copy" else Launch(name, "kernel", at, at + 0.5, *shape, 32))

    def logging(base, *names):
        class StandIn(base):
            def __call__(self, *operands):
                for name in names:
                    launch(name)
                return super().__call__(*operands)

        return StandIn

    class Step(torch_ref.GdnDecodeStep):
        """The stand-in of the fused step: one entry instance, in the form the row names; it states its launch."""

        provider = "cuDNN FE"
        kernels = (Kernel("gdn_decode_fused", "cuDNN FE", "exact", launches=1, shape="entry"),)

        def launch_facts(self, rows, site=""):
            return Facts((SYMBOL,), dict(self.row.form), ((96, rows, 1), (256, 1, 1), 42112))

        def serving(self):
            return ("cudnn.GdnDecodeFused #1", tuple(self.row.form))

        def __call__(self, x, z, hidden, slot_ids, params, slots):
            launch(defect.get("symbol") or SYMBOL, ((96, x.shape[0], 1), (256, 1, 1), 42112))
            return super().__call__(x, z, hidden, slot_ids, params, slots)

    class Choice(torch_ref.TokenChoice):
        def __call__(self, logits):
            launch(ARGMAX)
            chosen = super().__call__(logits)
            return (chosen + 1) % logits.shape[-1] if defect.get("token") else chosen

    stand_ins = {
        "fe.gdn_decode_fused": Step,
        "cublas.bf16_dequantized": logging(torch_ref.Linear, ELEMENTWISE, NVJET, ELEMENTWISE),
        "cublas.bf16_dequantized+torch_ref.silu_mul_quantize": logging(torch_ref.GateUp, ELEMENTWISE, NVJET, ELEMENTWISE),
        "torch.index_select": logging(torch_ref.Embedding, GATHER),
        "torch.argmax": Choice,
        "torch_ref.add_norm_quantize": logging(torch_ref.AddNormQuantize, ELEMENTWISE),
        "torch_ref.attention_prepare": logging(torch_ref.AttentionPrepare, ARANGE),
        "torch_ref.attention": logging(torch_ref.Attention, SOFTMAX),
        "torch_ref.gate_quantize": logging(torch_ref.GateQuantize, ELEMENTWISE),
    }
    for key, stand_in in stand_ins.items():  # the stand-ins keep the census entries of the providers they stand for
        if not key.startswith("fe."):
            real = factories()[key]
            stand_in.kernels, stand_in.provider = real.kernels, real.provider

    def profiled(step, count, device=None, records=None):
        events.append("profiler")
        defect.update(defect.pop("at the first profiler pass", {}))  # a defect of the traced steps alone
        if profiler != "records":
            return None, profiler
        del log[:]
        for i in range(count):
            step(i)
            launch(HOST_READ, kind="copy")
        if records is not None:
            records.extend(log)
        return Trace(count, 1.0, 2.0, 3, 4, 5.0), ""

    monkeypatch.setattr(loop, "trace_steps", profiled)
    return dict(factories(), **stand_ins)


def _session(monkeypatch, events, sequences=1, capacity=40, watch=False, record_steps=None, **stand_in):
    """A loaded session of the tiny geometry on the CPU through the rows of ``fe-cublas``, on the stand-ins of
    :func:`_stand_ins`: ``(session, text)``.  Every decode step of it writes what it ran with into ``events``:
    ``("step", rows of the record, whether a tap is set, whether a call log is open)``.  With ``watch`` the events also
    hold every host read (``"read"``), every reset (``("reset", fresh, recorded)``), every call of the tokenizer
    (``"encode"``) and every device synchronisation of a stage (``"sync"``)."""
    from model_samples import Session, TextIO, blocks

    build = _stand_ins(monkeypatch, events, **stand_in)
    model = _tiny_model()
    text = TextIO.synthetic(model.geometry.vocab)
    session = Session(model, text, sequences=sequences, stack="fe-cublas", device="cpu", capacity=capacity, check_kernels=True, record_steps=record_steps)
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 0, 0)
    options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": True}
    session.plan = blocks.plan(model.provider_table, model.regimes, facts, sequences, 1, "fe-cublas", model=model, options=options, factories=build)
    session.load()
    advance = session._advance

    def watched(ids, *more, **named):
        events.append(("step", int(session.record.inputs.shape[0]), session.tap is not None, session._call_log is not None))
        return advance(ids, *more, **named)

    monkeypatch.setattr(session, "_advance", watched)
    if watch:
        read, reset, encode = session.read, session.reset, text.encode
        session.stages._sync = lambda: events.append("sync")
        monkeypatch.setattr(session, "read", lambda ids: events.append("read") or read(ids))
        monkeypatch.setattr(session, "reset", lambda fresh=True, recorded=True: events.append(("reset", fresh, recorded)) or reset(fresh, recorded))
        monkeypatch.setattr(text, "encode", lambda *words, **named: events.append("encode") or encode(*words, **named))
    return session, text


class _Sink:
    """Where the text of sequence 0 would go: it keeps what it is handed."""

    def __init__(self):
        self.written = []

    def reported(self, text=""):
        return "goes to stdout after this report"

    def write(self, text):
        self.written.append(text)


def _mode(monkeypatch, tmp_path, steps=6, requests=2, warmup=2, sequences=1, budget=160_000, real=False, on_say=None, eos=None, **session_options):
    """One run of the mode on the stand-ins: ``(outcome, the lines it said, events, sink)``.  ``events`` holds, in order,
    every stamp the mode took (``"stamp"``), every decode step (``_session``, which takes ``session_options``), every
    line (``("say", line)``), every decoder made for a text (``"decoder"``) and every profiler pass (``"profiler"``).
    ``on_say(line)`` is called with every line: a test switches a defect on at the line that says which phase comes
    next.  ``real`` runs the text path of a checkpoint (the text written once, to the sink), with ``eos`` as the text's
    end tokens."""
    from model_samples import Environment, blocks, perf

    events, lines, sink = [], [], _Sink()
    session, text = _session(monkeypatch, events, sequences, capacity=5 + steps + 2, **session_options)
    if real:  # the weights are loaded; the report names a checkpoint by its own summary, which seeded weights do not have
        session.checkpoint, text.eos_ids = None, tuple(eos or ())
    clock = perf.now

    def stamped():
        events.append("stamp")
        return clock()

    def say(*said, flush=False):
        for line in said:
            events.append(("say", line))
            lines.extend(str(line).split("\n"))
            if on_say is not None:
                on_say(str(line))

    decoder = text.decoder

    def decoded(prompt=None):
        events.append("decoder")
        return decoder(prompt)

    monkeypatch.setattr(perf, "now", stamped)
    monkeypatch.setattr(text, "decoder", decoded)
    outcome = perf.run(
        session,
        text,
        lambda: [text.encode("ab" + str(s) + "xy") for s in range(sequences)],
        perf.Protocol(warmup, steps, requests, budget),
        say,
        name="tiny model, reduced model",
        facts=blocks.DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB),
        environment=Environment("1.34.0", "1.34.0", True, "0123456789ab", "2.13.0", "4.8.0"),
        sink=sink,
        report=str(tmp_path / "report.json"),
        verdict=lambda passed, judged, unobserved: "FAIL" if not passed else (f"RAN (kernels not observed: {unobserved})" if unobserved else "PASS"),
        real=real,
        source="the seeded prompt",
        text_path={"render": 0.001, "tokenize": 0.002},
    )
    return outcome, lines, events, sink


def _line(lines, start):
    return next(line for line in lines if line.startswith(start))


def test_no_profiler_before_the_last_timed_request(monkeypatch, tmp_path):
    """The first profiler pass of the process starts behind the last stamp of the last timed request -- behind the traced
    request's own prompt and first read -- and PyTorch's profiler is constructed by nothing but that door."""
    import torch.profiler

    constructed = []
    monkeypatch.setattr(torch.profiler, "profile", lambda *args, **options: constructed.append(options) or pytest.fail("a profiler was constructed"))
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path, steps=6, requests=2, warmup=2)
    assert outcome.passed and not constructed
    stamps = [index for index, event in enumerate(events) if event == "stamp"]
    passes = [index for index, event in enumerate(events) if event == "profiler"]
    # the checked request, the warm-up request and two timed requests: t_request and t_first, then two stamps per forced step
    before = (2 + 2 * 6) + (2 + 2 * 2) + 2 * (2 + 2 * 6)
    assert len(passes) == 2 and passes[0] > stamps[before + 1], "the traced request's t_request and t_first come first too: its prompt is not traced"
    assert sum(event == "stamp" for event in events[: passes[0]]) == before + 2
    assert sum(event == "stamp" for event in events[passes[0] : passes[1]]) == 2, "the first pass is one step: its records size the rest"
    assert sum(event == "stamp" for event in events[passes[1] :]) == 2 * 5, "the second pass: the other five timed positions"
    assert outcome.data["protocol"]["profiler_first_started"] == "behind the last timed request, in the traced request"
    assert outcome.data["traced"]["steps"] == 6 and outcome.data["traced"]["records_of_the_sizing_step"] > 1
    # a budget that holds three steps: the first three timed positions are traced, and the report says which
    sizing = outcome.data["traced"]["records_of_the_sizing_step"]
    few, lines, events, _ = _mode(monkeypatch, tmp_path / "few", budget=3 * sizing + 1)
    assert (few.data["traced"]["steps"], few.data["traced"]["first_position"], few.data["traced"]["last_position"]) == (3, 5, 7)
    assert few.passed and few.data["statistics"]["step"]["n"] == 2 * 3, "the kernel check holds on both passes; the step is taken at those positions"
    at = lines.index(_line(lines, "step "))
    under = [line.strip() for line in lines[at + 1 : at + 6]]
    assert any(line.startswith("(GPU busy from 3 traced steps, positions 5 to 7; median ") for line in under), lines[at : at + 6]
    one, lines, events, _ = _mode(monkeypatch, tmp_path / "one", budget=1)  # a budget below one step: the sizing step is the traced request
    assert one.data["traced"]["steps"] == 1 and events.count("profiler") == 1 and one.passed
    _says("area", "The first profiler pass of the process starts behind the last timed request: the traced request feeds its prompt untraced")
    _says("area", "its first step alone -- whose launch records size the rest -- then as many further steps as `--perf-trace-records` holds")
    _says("family", "Its first step is traced alone and sizes the rest (it is always traced, also where its records alone pass N);")
    _says("family", "where the timed steps do not all fit, it traces the first steps that do, and the report says how many and at which positions")
    _says("guide", "and no profiler pass runs in the process before its last timed request.")


def test_trace_request_traces_steps_of_the_request_that_is_running(monkeypatch):
    """``Session.trace_request``: the caller's own loop body under the profiler, from where the state stands -- nothing is
    replayed or swapped --, one call log per traced step, closed behind the last one; a second call goes on from there.
    A session that verifies behind a prompt and a step of several tokens are refused."""
    from model_samples import Session, TextIO

    events = []
    session, text = _session(monkeypatch, events, capacity=12)
    state, ids, read = session.state, [session.prefill([text.encode("ab0xy")])], []

    def step(index):
        ids[0] = session.step(ids[0])
        read.append(session.read(ids[0]))

    session.read(ids[0])
    session.trace_request(step, 2)
    assert session.state is state and state.position == 7 and len(read) == 2, "the request's own state, two steps on"
    assert len(session.traced.call_log) == len(session.traced.steps) == 2 and session._call_log is None, "a log per step, closed behind the last"
    assert [event for event in events if event[0] == "step"][-2:] == [("step", 12, False, True)] * 2 and events.count("profiler") == 1
    session.step(ids[0])
    assert len(session.traced.call_log[-1]) == len(session.traced.call_log[0]), "a step behind the traced ones is logged nowhere"
    session.trace_request(step, 1)
    assert state.position == 9 and len(session.traced.steps) == 1 and events.count("profiler") == 2, "each call is a pass of its own"
    for other in (
        Session(_tiny_model(), TextIO.synthetic(272), stack="reference", device="cpu", draft_tokens=2),
        Session(_tiny_model(), TextIO.synthetic(272), stack="reference", device="cpu", tokens=3),
    ):
        with pytest.raises(ValueError, match="Session.trace_request"):
            other.trace_request(step, 1)


def test_timed_request_has_no_recorder_no_text(monkeypatch, tmp_path):
    """From the line that announces the warm-up request to the first profiler pass: every decode step runs on a record of 0
    steps, with no tap and no call log; no line is said, no text is decoded.  The checked request before it is recorded
    whole; the traced steps have their call log; the text is decoded once, behind the report."""
    from model_samples import perf

    (tmp_path / "x").mkdir()
    outcome, lines, events, sink = _mode(monkeypatch, tmp_path / "x", steps=6, requests=2, warmup=2)
    announced = next(index for index, event in enumerate(events) if event[0] == "say" and "a warm-up request, 2 timed requests" in event[1])
    first_pass = events.index("profiler")
    inside = events[announced + 1 : first_pass]
    steps = [event for event in inside if event[0] == "step"]
    assert len(steps) == (5 + 2) + 2 * (5 + 6) + 5, "the warm-up request, the timed requests, the traced request's prompt"
    assert all(event == ("step", 0, False, False) for event in steps), "nothing recorded, no tap, no call log"
    assert all(event == "stamp" or event[0] == "step" for event in inside), "no line and no text between the requests' stamps"
    checked = [event for event in events[:announced] if event[0] == "step"]
    assert checked[: 5 + 6] == [("step", 5 + 6 + 2, False, False)] * (5 + 6), "the checked request: every step recorded, none under a tap"
    assert [event[2] for event in checked[5 + 6 :]] == [True] * (5 + 6), "the replay of the checks: every one of these eleven positions is a judged one"
    traced = [event for event in events[first_pass:] if event[0] == "step"]
    assert traced == [("step", 0, False, True)] * 6, "the traced steps: the call log of the kernel check, still no recorder"
    report = next(index for index, event in enumerate(events) if event[0] == "say" and "decode performance report" in event[1])
    assert events.count("decoder") == 1 and events.index("decoder") > report, "the ids are detokenised once, behind the report"
    assert sink.written == [], "a seeded run writes its text nowhere"
    assert outcome.data["protocol"]["in_a_timed_request"] == {
        "recorder": "off (a record of 0 steps)",
        "call_log": "off",
        "profiler": "not started",
        "text": "none",
        "sampler": "none (no CUDA device)",  # this session is on the CPU: no sampler ran beside it, and the file says so
    }
    assert all(request["end_state_sha256"] == outcome.data["raw"]["requests"][0]["end_state_sha256"] for request in outcome.data["raw"]["requests"][2:4])
    # the count the run printed before the load: every decode step of the stack, the replay of the checks among them
    assert sum(event[0] == "step" for event in events) == perf.Protocol(2, 6, 2).decode_steps(5)
    _says("area", "the mode separates what is checked from what is timed, and what is timed from what is traced.")
    _says(
        "area", "A run makes, in this order: one checked request, launched as every other, with the recorder of the checks on, and the reference checks on it"
    )
    inside = "no recorder, no call log, no profiler, no text"
    _says("area", f"Inside a timed request there is nothing but the step: {inside}.", "with the recorder of the checks on")
    _says("guide", f"Nothing but the step runs inside a timed request: {inside};")
    _says("script", "forced decode steps (default 256, the end token ignored) with nothing but the step inside; only then the profiler")


def test_every_stamp_stands_where_its_definition_says(monkeypatch, tmp_path):
    """One monotonic clock, and every stamp of a timed request at its event: ``t_request`` in front of render and
    tokenize, ``t_first`` behind the first host read, ``t_launch`` in front of a step, ``t_token`` behind its one host
    read.  The state is zeroed in place in front of every request, and the device is synchronised between that zeroing
    and the request's first stamp, and not again before its last."""
    from model_samples import perf, timing

    assert perf.now is timing.now and perf.CLOCK.startswith("time.perf_counter")
    with monkeypatch.context() as patch:
        patch.setattr(timing.time, "perf_counter", lambda: 12345.5)
        assert perf.now() == 12345.5, "the clock of the stamps is the monotonic one"
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path, steps=6, requests=2, warmup=2, watch=True)
    kinds = [event if isinstance(event, str) else event[0] for event in events]
    zeroings = [index for index, event in enumerate(events) if event == ("reset", False, False)]
    assert len(zeroings) == 4 and ("reset", True, False) not in events, "in front of the warm-up, each timed and the traced request: in place"
    request = ["stamp", "encode", *["step"] * 5, "read", "stamp"]  # t_request, the text path, the prompt's steps, the first host read, t_first
    step = ["stamp", "step", "read", "stamp"]  # t_launch, the step, its one host read, t_token
    for start, stop in zip(zeroings[1:3], zeroings[2:4]):  # the two timed requests, each from its zeroing to the next one
        inside = [kind for kind in kinds[start + 1 : stop] if kind in ("stamp", "encode", "step", "read", "sync")]
        assert inside == ["sync", "sync", *request, *step * 6, "sync", "sync"], inside
    assert kinds[zeroings[0] + 1 : zeroings[1]].count("sync") == 4, "the warm-up request too: its zeroing and its own stage"
    assert outcome.passed and outcome.data["protocol"]["restart"].endswith("zeroed in place at position 0, the device synchronised behind the zeroing")
    assert outcome.data["startup"]["zeroing the state_s"] >= 0.0
    _defines("startup", "Import, describe and plan are host times; every later stage", "the zeroing of the", "has the device synchronised at both ends.")
    _defines("startup", "The stage of the timed requests spans the requests the statistics are taken from; no other stage enters a rate or the time to")
    _says("area", "timed requests, each from the state zeroed in place (the device is synchronised behind the zeroing, which is outside every statistic);")
    _says("area", "*Token-to-token interval*: the time between the host reads of two consecutive tokens of a timed request.")
    _says("area", "from the request being handed to the text path to the first new token's ids on the host: render, tokenize, the copy of the ids,")
    # the raw stamps of the file are seconds from the request, and they give the file's statistics
    data = json.loads(json.dumps(outcome.data))
    timed = [made for made in data["raw"]["requests"] if made["kind"] == "timed"]
    assert [(made["t_token_s"][-1] - made["t_first_s"]) * 1e3 / 6 for made in timed] == pytest.approx(
        data["statistics"]["time_per_output_token"]["per_request"]
    )
    assert [made["t_first_s"] * 1e3 for made in timed] == pytest.approx(data["statistics"]["time_to_first_token"]["per_request"])
    _says("area", "(every stamp of every request, relative to its start, and the hashes of its ids and of its end state)")


def test_the_end_state_digest_reads_every_byte(monkeypatch):
    """The digest of a state changes with one element of any one of its tensors, and comes back with it."""
    from model_samples import perf

    session, text = _session(monkeypatch, [], capacity=12)
    session.read(session.prefill([text.encode("ab0xy")]))
    before = perf.state_digest(session.state)
    tensors = [tensor for layer in session.state.layers for tensor in layer.tensors()]
    assert len(tensors) > 2
    for tensor in tensors:
        flat = tensor.view(-1)
        kept = flat[-1].clone()
        flat[-1] = kept + 1
        assert perf.state_digest(session.state) != before, "one element of one state tensor"
        flat[-1] = kept
    assert perf.state_digest(session.state) == before


def test_the_text_of_a_checkpoint_is_written_once_behind_the_report(monkeypatch, tmp_path):
    """The text path of a checkpoint: the text of sequence 0 up to the first end token (all of it where none occurs) goes
    to the sink once, behind the report, with one line break; the ``text out`` line says which token ended it."""
    for name in ("seeded", "ended", "whole"):
        (tmp_path / name).mkdir()
    produced = _mode(monkeypatch, tmp_path / "seeded")[0].produced[0]
    assert len(produced) == 7, "the first new token and six forced steps"
    end = produced[2]
    outcome, lines, events, sink = _mode(monkeypatch, tmp_path / "ended", real=True, eos=[end])
    at = produced.index(end) + 1
    assert outcome.passed and outcome.produced[0] == produced, "the end token is ignored by the forced steps: the same ids"
    assert _line(lines, "text out ") == f"text out              sequence 0, to its first end token (token {at} of 7): goes to stdout after this report"
    assert len(sink.written) == 1 and sink.written[0].endswith("\n") and events.count("decoder") == 1
    assert outcome.data["text_path"]["tokens_written"] == at and outcome.data["text_path"]["cut_at_an_end_token"] is True
    assert lines.index(_line(lines, "text out ")) > lines.index(_line(lines, "report file ")), "behind the report"
    assert events.index("decoder") > max(index for index, event in enumerate(events) if event == "profiler"), "and behind every request"
    outcome, lines, events, whole = _mode(monkeypatch, tmp_path / "whole", real=True, eos=[271])  # an id the run does not produce
    assert 271 not in produced
    assert _line(lines, "text out ") == "text out              sequence 0, all 7 new tokens (no end token): goes to stdout after this report"
    assert len(whole.written) == 1 and len(whole.written[0]) >= len(sink.written[0]) and outcome.data["text_path"]["tokens_written"] == 7
    _says("guide", "or once, after the report, with `--no-stream` and always in the performance mode")
    _says("script", "the text is written once, after the report.")
    _says("family", "and the text written once, after the report (as `--no-stream`)")


def test_checks_lines_say_what_was_judged(monkeypatch, tmp_path):
    """Three statements, each with its own verdict: the blocks at the judged positions, the timed requests' tokens and
    state, the kernels of the traced steps.  The first line of the checks names what was replayed and where the blocks
    were judged; the block check ran at the judged positions and nowhere else."""
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path, steps=62, requests=1, warmup=1)  # 5 prompt tokens: timed positions 5 to 66
    assert outcome.passed and outcome.judged > 0
    head = _line(lines, "checks (1 sequence, ")
    assert head == "checks (1 sequence, 67 steps replayed: 5 prompt + 62 decode; blocks judged at positions 0-15, 63-66; tokens on the first 16 steps)"
    step = _line(lines, "    linear_attention.step ")
    assert " 20 calls " in step and step.endswith("PASS"), "one call of the fused step at each of the 20 judged positions"
    screen = outcome.data["checks"]
    assert screen["blocks"] == "PASS" and screen["timed_requests"] == "PASS" and screen["kernels"] == "PASS"
    assert [entry["position"] for entry in screen["judged_positions"]] == [*range(16), 63, 64, 65, 66]
    assert _line(lines, "checks      ") == "checks      blocks against the reference at positions 0-15, 63-66: PASS"
    both = "            timed requests give the checked request's tokens and state: PASS | kernels, traced steps: PASS"
    assert lines[lines.index(_line(lines, "checks      ")) + 1] == both
    (kernel_line,) = [line for line in lines if line.startswith("  kernels           ")]
    assert kernel_line.endswith("    PASS"), "one kernel line, the traced request's, under the checks: none from the checks before"
    assert _line(lines, "kernels     ") == "kernels     no target kernel set is declared for this stack"
    after = lines[lines.index(_line(lines, "kernels     ")) + 1]
    assert re.fullmatch(r"            \d+ block calls on fallback rows; a step launches \d+ kernels and 1 copy", after), after
    assert screen["last_line"] == "[00] PASS" and outcome.data["record"]["performance_claim_eligible"] is False
    _says("area", "replay over the whole request, every provider block against the reference at the judged positions, the tokens on the first 16 steps;")
    _says("area", "The `checks` lines are three statements, each with its own verdict:", "on a stack that declares no target they say so.")
    _says("guide", "The report says at which positions the blocks were held against the reference, that every timed request gave the checked request's")
    _says("guide", "tokens and state, and that the kernels of the traced steps are the plan's;")
    # what is printed as judged is what the checks judged: they hand it back, and a position the run did not record is refused
    with pytest.raises(ValueError, match="the checked request is 67 steps and the session records 16"):
        _mode(monkeypatch, tmp_path / "short", steps=62, requests=1, warmup=1, record_steps=16)


def test_the_kernels_lines_are_three_numbers_from_the_plan_and_the_census(monkeypatch):
    """Statistic 16 on the family's own tables: the 64-layer model on compute capability 10.0 with 148 SMs, planned on
    stated facts -- how many launches of a step are on the target kernel set, by provider and as three numbers, and the
    block calls on fallback rows: on the cuDNN Frontend rows alone (``fe-entries``) and on ``native``, which has the
    rows a library serves beside them."""
    from model_samples import Session, TextIO, blocks, census, perf
    from model_samples.qwen_dense import QwenDense

    monkeypatch.setattr(census, "installed_versions", lambda packages: {"torch": "2.14.0+cu130", "nvidia-cutlass-dsl": "4.8.0"})
    model = QwenDense()
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)

    def planned(stack):
        session = Session(model, TextIO.synthetic(272), stack=stack, device="cpu", capacity=8)
        session.plan = blocks.plan(model.provider_table, model.regimes, facts, 1, 1, stack, model=model, options={"device": "cpu"})
        return perf.kernel_facts(session, "PASS")

    kernels, native = planned("fe-entries"), planned("native")
    assert (native["on_target"], native["launches"]) == (451, 484) and (native["code"], native["port"], native["stand_ins"]) == (178, 273, 0)
    assert native["by_provider"] == {"cuDNN Frontend": 321, "FlashInfer": 129, "PyTorch": 1}
    assert (native["fallback_calls"], native["absent"]) == (32, 1) and native["fallback_rows"], "451 + 32 + 1 = 484: the attention blocks are left"
    wide = perf.screen(_data(kernels=native))  # a third provider: the line breaks between two of them
    at = wide.index("kernels     451 of 484 launches per step are on the target kernel set: cuDNN Frontend 321, FlashInfer 129,")
    assert wide[at + 1 : at + 3] == [
        " " * 12 + "PyTorch 1",
        " " * 12 + "= 178 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results",
    ]
    assert kernels["declared"] and kernels["stated"] and kernels["fallback_rows"]
    assert (kernels["on_target"], kernels["launches"]) == (322, 484) and (kernels["code"], kernels["port"], kernels["stand_ins"]) == (49, 273, 0)
    assert kernels["by_provider"] == {"cuDNN Frontend": 321, "PyTorch": 1}
    assert (kernels["fallback_calls"], kernels["absent"]) == (161, 1), "322 + 161 + the one launch of the target that no block stands for = 484"
    assert kernels["kernels_per_step"] is None and "NOT CLAIMED" in kernels["census"], "nothing traced; a fallback row is left"
    data = _data(kernels={**kernels, "kernels_per_step": 2908, "copies_per_step": 200})
    shown = perf.screen(data)
    at = shown.index("kernels     322 of 484 launches per step are on the target kernel set: cuDNN Frontend 321, PyTorch 1")
    assert shown[at + 1 : at + 5] == [
        "            = 49 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results",
        "              + 0 declared stand-ins",
        "            not yet: 161 block calls on fallback rows + 1 launch of the target that has no block yet",
        "            a step also launches 2,586 kernels and 200 copies",
    ]
    unseen = perf.screen(_data(kernels=kernels))
    assert "            what a step also launches was not observed" in unseen
    whole = perf.screen(_data(kernels={**kernels, "absent": 0, "fallback_calls": 0, "kernels_per_step": 485, "copies_per_step": 1}))
    assert "            not yet: no block calls on fallback rows; a step also launches 163 kernels and 1 copy" in whole
    three = "the target's own compiled kernels, cuDNN Frontend entries with byte-equal results, declared stand-ins"
    _defines("launches_per_step", f"the launches of a step that are on the target kernel set, by provider and as three numbers -- {three}.")
    _says("area", "The `kernels` lines give the launches of a step that are on the target kernel set, by provider and as three numbers, and what else")


# ---------------------------------------------------------------------------------------------- the screen and the file
_BASE = {}


def _data(**changes):
    """The data of a report, as ``perf.run`` builds it on the stand-ins (made once, with pytest's own monkeypatch context),
    with sections replaced: ``_data(model={...})`` merges into the section ``model``."""
    import copy
    import tempfile

    if not _BASE:
        with pytest.MonkeyPatch.context() as patch, tempfile.TemporaryDirectory() as folder:
            _BASE["data"] = _mode(patch, Path(folder))[0].data
    data = copy.deepcopy(_BASE["data"])
    for section, values in changes.items():
        data[section] = {**data[section], **values} if isinstance(values, dict) and isinstance(data.get(section), dict) else values
    return data


def _checkpoint_form(**changes):
    """The data of a run on a checkpoint -- not a reduced model, a rate, 51 traced steps behind a prompt of 58 tokens --
    with sections replaced as in :func:`_data`.  The rate and the step are made up for the form -- 2.5 tokens per second
    and a step of 400 ms -- and are no run's."""
    base = {
        "model": {"name": "Qwen3.8-27B-NVFP4", "layers": 64, "reduced": False},
        "record": {"reasons": ["fallback rows present", "kernel identity not claimed on this GPU class", "launched from Python"]},
        "traced": {"steps": 51, "first_position": 58, "last_position": 108},
    }
    for section, values in changes.items():
        base[section] = {**base[section], **values} if isinstance(values, dict) and section in base else values
    data = _data(**base)
    data["statistics"]["decode_rate"].update(value=2.5, reason=None, total=2.5)
    data["statistics"]["step"]["value"] = 400.0
    data["protocol"]["prompt"]["tokens"] = 58
    return _gpu_side(data)


def _gpu_side(data, busy=80.0, term=True, host=40.0, stretch=5.0, why="", mean=410.0):
    """The GPU side of a report's data, set: the median GPU busy in ms (``None``: not measured, and ``why``), whether
    launch overhead is a term of the step, the loop's median host time, tracing's stretch in percent and the mean
    step.  The means of GPU busy and of host are set 2 ms and 1 ms above their medians, and the remainder is what they
    leave of the mean step: the terms add up, as a run's do.  The defaults are made up for the form -- the parts of a
    mean step of 410 ms beside a median of 400, the time per block call that of 700 calls -- and are no run's."""
    from model_samples import perf

    stats, measured = data["statistics"], busy is not None
    remainder = mean - (busy + 2.0) - (host + 1.0) if measured else None
    stats["step"]["mean"] = mean
    stats["gpu_busy"].update(value=busy, mean=busy + 2.0 if measured else None, reason=None if measured else f"GPU busy not measured ({why})")
    stats["host"].update(value=host, mean=host + 1.0)
    stats["launch_overhead"].update(value=remainder if term and measured else None, remainder_ms=remainder, label=perf.ESTIMATE if measured else None)
    stats["launch_overhead"]["host_time_per_block_call_us"] = remainder * 1e3 / 700 if measured else None
    stats["tracing_stretch"]["value"] = None if busy is None else stretch
    data["traced"].update(gpu_busy_ms=busy, gpu_busy_reason=why or None)
    if busy is None:
        data["per_kernel"] = {**data["per_kernel"], "rows": None, "reason": f"the kernels of the traced steps were not observed ({why})"}
    return data


def test_orientation_stands_in_the_rate_line():
    """While a run is orientation the class is on line 2 with every reason and inside the rate line; a reduced model prints
    no rate and prefixes every time; a controlled run -- every condition handed in as met -- prints neither."""
    from model_samples import perf

    lines = perf.screen(_checkpoint_form())
    assert lines[0] == "cuDNN Frontend model sample: decode performance report"
    assert lines[1] == "record class          orientation, not a number of record: fallback rows present;"
    assert lines[2] == "                      kernel identity not claimed on this GPU class; launched from Python", "a line ends where a reason ends"
    assert lines[3] == "scope       a decode-loop rate, not a served rate: one process, no request queue, no scheduler, no network"
    rate = _line(lines, "decode rate")
    assert rate.startswith("decode rate (orientation)   2.5  tokens/s per sequence   (") and " ms per token; 2 requests within " in rate
    assert not any(line.startswith("decode rate   ") for line in lines) and not any("reduced model:" in line for line in lines)
    assert "with draft tokens     not run by this script (plain decode)" in lines
    reduced = perf.screen(_data())
    assert "decode rate           not printed for a reduced model" in reduced and not any("tokens/s" in line for line in reduced)
    assert reduced[1] == "record class          orientation, not a number of record: reduced model; fallback rows present;"
    assert reduced[2:4] == [
        "                      no target kernel set declared; launched from Python; other processes not observed;",
        "                      clocks not observed",
    ]
    for start in ("                      token-to-token interval: ", "time to first token   ", "step                  "):
        assert _line(reduced, start).startswith(start + "reduced model: "), start
    _says("area", "every time then carries `reduced model:`, no rate is printed, and the text is written nowhere.")
    _says("script", "the same protocol runs on the reduced seeded model and prints no rate.")
    _says("guide", "The rate of a run that is not a controlled run says so in its own line, and line 2 says it of every other number.")
    _says("area", "Line 2, `record class`, says what the numbers of a run are.")
    _says("area", "The line `with draft tokens` says `not run by this script`: this script decodes without a drafter.")
    controlled = perf.screen(_checkpoint_form(record={"class": "controlled", "reasons": []}))
    assert controlled[1] == "record class          controlled run in one process; a number of record is the median of 3 such runs"
    assert _line(controlled, "decode rate").startswith("decode rate           2.5  tokens/s per sequence   (")
    several = perf.screen(_checkpoint_form(model={"name": "Qwen3.8-27B-NVFP4", "layers": 64, "reduced": False, "sequences": 4}))
    at = several.index("decode rate (orientation)   2.5  tokens/s per sequence (2.5 in all)")  # the total beside the rate: the bracket moves down
    assert several[at + 1].startswith(" " * 28 + "(") and several[at + 1].endswith(" %)") and "| 4 sequences | greedy" in _line(several, "model       ")
    # every number of the timing lines is the field of the file's data its label names: the run's value, never the mean
    stats = (labelled := _checkpoint_form(model={"name": "Qwen3.8-27B-NVFP4", "layers": 64, "reduced": False, "sequences": 4}))["statistics"]
    stats["decode_rate"].update(value=2.5, total=10.0)
    stats["time_per_output_token"].update(value=111.111, median=111.111, mean=222.222, minimum=1.0, maximum=999.0, spread_percent=3.3)
    stats["token_to_token_interval"].update(value=33.333, median=33.333, mean=44.444, p90=55.555, p99=66.666)
    stats["time_to_first_token"].update(value=7777.7, median=7777.7, mean=8888.8)
    said = " ".join(line.strip() for line in perf.screen(labelled))
    assert "decode rate (orientation)   2.5  tokens/s per sequence (10.0 in all) (111.111 ms per token; 2 requests within 3.3 %)" in said
    assert "token-to-token interval: median 33.333, p90 55.555, p99 66.666 ms time to first token   7777.7  ms for a 58-token prompt" in said


def test_the_step_line_says_what_was_not_measured():
    """Where GPU busy was not measured the ``step`` line names the positions the step was taken at and says so, with the
    reason: no line of the screen takes the step apart, and no table is printed."""
    from model_samples import perf

    cut = "the launch records could not be cut into the traced steps"
    lines = perf.screen(_gpu_side(_checkpoint_form(), busy=None, why=cut))
    at = lines.index(_line(lines, "step  "))
    said = lines[at] + " " + lines[at + 1].strip()
    assert said == f"step                  400.000  ms at the 51 traced positions, 58 to 108; GPU busy not measured ({cut})"
    assert _line(lines, "time to first token   ").endswith(" ms for a 58-token prompt (58 decode steps: this sample has no prefill kernel)")
    assert not any("launch overhead" in line or line.startswith(("note ", "per kernel, per step (")) for line in lines)
    _says("area", "Where the traced steps could not be observed the line says `GPU busy not measured`, with the reason, and no table is printed")
    one = perf.screen(_gpu_side(_checkpoint_form(traced={"steps": 1, "last_position": 58}), busy=None, why=cut))
    assert _line(one, "step  ").startswith("step                  400.000  ms at the traced position, 58; GPU busy not measured (the launch records ")
    why = "the profiler's trace could not be exported (No space left on device)"
    unmeasured = perf.screen(_gpu_side(_checkpoint_form(traced={"steps": 0}), busy=None, why=why))
    at = unmeasured.index("step                  400.000  ms over every timed step; GPU busy not measured (the profiler's trace could not")
    assert unmeasured[at + 1] == "                      be exported (No space left on device))"
    table = unmeasured.index("per kernel, per step  not printed: the kernels of the traced steps were not observed (the profiler's trace")
    assert unmeasured[table + 1] == "                      could not be exported (No space left on device))" and unmeasured[table - 1] == ""
    memory = lines.index(_line(lines, "memory                "))
    assert (
        " GiB | state and cache " in lines[memory] and lines[memory + 1] == "                      peak reserved not observed  (GiB = 2^30 bytes)"
    ), "on the CPU"
    held = perf.screen(_checkpoint_form(memory={"weights_gib": 37.3, "state_gib": 0.5, "peak_reserved_gib": 40.0}))
    assert "memory                weights 37.3 GiB | state and cache 0.50 GiB | peak reserved 40.0 GiB  (GiB = 2^30 bytes)" in held
    assert len(_line(held, "memory   ")) == 110, "one line at the screen's width"
    _defines("memory", "Peak reserved: PyTorch's peak between the emptied pool behind the checks and the end of the timed requests.")
    _says("area", "peak reserved: PyTorch's peak between the emptied pool behind the checks and the end of the timed requests; in GiB (2^30 bytes)")


def _wide_row():
    """A row of the per-kernel table whose every value is as wide as a run can make it."""
    row = {
        "label": "not yet: projection.fp8.out + 123 more (one row)",
        "relation": "fallback",
        "launches_per_step": 1234567.5,
        "us_per_launch_added": 1234567.891,
    }
    return {**row, "ms_per_step": 12345.678, "share_percent": 100.0}


def test_no_line_wider_than_110(monkeypatch, tmp_path):
    """No line of the screen is wider than 110 characters, in every form a run of this version prints: the reduced model,
    a checkpoint, a target stack with its three numbers, each form of the ``step`` line, the per-kernel table with and
    without its ``us/launch`` column and with large values, a failed statement, kernels that were not observed with a
    long reason, long names, six-digit values."""
    from model_samples import perf

    long_reason = "the profiler's trace could not be exported (" + "a reason that the exporter words at length " * 3 + ")"
    target = {"stated": True, "declared": True, "on_target": 322, "launches": 484, "code": 49, "port": 273, "stand_ins": 0}
    target.update(by_provider={"cuDNN Frontend": 321, "PyTorch": 1}, fallback_calls=162, kernels_per_step=2908, copies_per_step=200)
    forms = [
        _data(),
        _checkpoint_form(),
        _checkpoint_form(kernels=target),
        _checkpoint_form(kernels={**target, "stated": False, "not_stated": "kernel identity: not stated for this run (" + "no recorded section " * 6 + ")"}),
        _checkpoint_form(traced={"steps": 256, "first_position": 123456, "last_position": 123711}),
        _gpu_side(_checkpoint_form(traced={"steps": 256, "first_position": 123456, "last_position": 123711}), busy=123456.789, term=False, mean=234567.891),
        _gpu_side(_checkpoint_form(traced={"steps": 256, "first_position": 123456, "last_position": 123711}), busy=123456.789, host=1234.567, mean=987654.321),
        _gpu_side(_checkpoint_form(traced={"steps": 0}), busy=None, why=long_reason),
        _checkpoint_form(per_kernel={"us_per_launch_printed": True, "rows": [dict(_wide_row(), label="x" * perf.LABEL), _wide_row()]}),
        _checkpoint_form(per_kernel={"rows": [_wide_row()], "reason": None}),
        _checkpoint_form(per_kernel={"rows": None, "reason": perf.NOT_THE_PLAN}),
        _checkpoint_form(checks={"timed_requests": "FAIL (request 3, step 255)", "kernels": f"NOT OBSERVED ({long_reason})"}),
        _checkpoint_form(
            checks={
                "positions_text": ", ".join(str(n) for n in range(0, 400, 2)),
                "kernels": "FAIL (12 kinds of difference: the kernels lines above name them)",
            }
        ),
        _checkpoint_form(model={"name": "a-checkpoint-directory-with-a-very-long-name-" * 3, "layers": 64, "reduced": False, "sequences": 64}),
        _checkpoint_form(report_file="a-report-file-with-a-very-long-name-" * 4 + ".json"),
        _checkpoint_form(record={"reasons": ["reduced model", "fallback rows present", "kernel identity not claimed on this GPU class"] * 4}),
    ]
    forms[1]["statistics"]["decode_rate"].update(value=None, reason=f"not printed: {perf.NOT_COMPUTED}")
    forms[2]["statistics"]["decode_rate"].update(value=None, reason=f"not printed: {perf.NOT_THE_PLAN}")
    for data in forms:
        lines = perf.screen(data)
        assert max(len(line) for line in lines) <= perf.WIDTH == 110, max(lines, key=len)
        assert lines[0] == "cuDNN Frontend model sample: decode performance report" and lines[-1].startswith(("report file           ", " " * 22))
        assert sum(line.startswith("scope       a decode-loop rate, not a served rate: ") for line in lines) == 1, "in every form"
    assert "decode rate           not printed: the timed requests did not compute the checked request's tokens" in perf.screen(forms[1])
    assert "decode rate           not printed: the kernels of the traced steps are not the plan's" in perf.screen(forms[2])
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path)  # and the lines a run printed are those of its data
    said = lines[lines.index("cuDNN Frontend model sample: decode performance report") :]
    assert max(len(line) for line in said) <= 110, "the text out line behind the screen too"
    assert said[: len(perf.screen(outcome.data))] == perf.screen(outcome.data) and 34 <= len(perf.screen(outcome.data)) <= 40
    assert perf.screen(json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))) == perf.screen(outcome.data), "the file's data and nothing else"
    unwritten = perf.screen(outcome.data, "--perf-report late.json: No space left on device")
    assert (
        unwritten[-1] == "report file           not written: --perf-report late.json: No space left on device" and unwritten[:-1] == said[: len(unwritten) - 1]
    )
    _says("guide", f"The screen is rendered from the data the report file holds and from nothing else, at no more than {perf.WIDTH} characters a line;")
    _says("guide", "a line of it that would be longer breaks at a separator or between two words.")
    _says("area", "The `scope` line behind it is always there: a decode-loop rate is not a served rate.")


def test_file_strict_json_definitions_no_path(monkeypatch, tmp_path):
    """The report file: strict JSON (no NaN, no infinity), the sections of this version, a definition beside every statistic
    and the serving counterpart or null, units in the field names, UTC, no local path in any string, and never written
    over a file that exists."""
    from model_samples import perf

    outcome, lines, events, _ = _mode(monkeypatch, tmp_path, steps=6, requests=2, warmup=2)
    text = (tmp_path / "report.json").read_text(encoding="utf-8")

    def refuse(constant):
        raise AssertionError(f"the file holds {constant}")

    data = json.loads(text, parse_constant=refuse)
    assert data == json.loads(json.dumps(outcome.data)) and outcome.unwritten == ""
    sections = "schema kind written_utc report_file record model environment observation kernels protocol checks statistics raw traced per_kernel memory"
    sections += " startup text_path"
    assert list(data) == sections.split() and data["schema"] == "model-sample-performance/4" and data["kind"] == "run"
    assert "fingerprints" not in data and "report_text" not in data, "of later versions"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", data["written_utc"]) and data["report_file"] == "report.json"
    assert str(tmp_path) not in text and os.sep + "home" + os.sep not in text and str(MODELS) not in text, "a file is named by its own name"
    assert data["record"] == {
        "class": "orientation",
        "reasons": [
            "reduced model",
            "fallback rows present",
            "no target kernel set declared",
            "launched from Python",
            "other processes not observed",
            "clocks not observed",
        ],
        "timing_role": "orientation",
        "performance_claim_eligible": False,
    }
    assert set(data["statistics"]) == set(perf.DEFINITIONS), "one definition per statistic of the file, and no statistic without one"
    for key, (unit, definition, counterpart) in perf.DEFINITIONS.items():
        entry = data["statistics"][key]
        assert (entry["unit"], entry["definition"], entry["serving_counterpart"]) == (unit, definition, counterpart) and len(definition) > 40, key
        assert "value" in entry and (entry["value"] is not None or entry["reason"]), f"{key}: a value that does not exist is null with a reason"
    assert data["statistics"]["decode_rate"]["value"] is None and data["statistics"]["decode_rate"]["reason"] == "not printed for a reduced model"
    sentences = " ".join(f"{definition} {counterpart or ''}" for _, definition, counterpart in perf.DEFINITIONS.values())
    assert not re.search(
        r"\d\d", sentences.replace("1000 /", "").replace("2^30", "").replace("10^9", "")
    ), "a definition holds no figure of a run or of a record"
    kinds = [request["kind"] for request in data["raw"]["requests"]]
    assert kinds == ["checked", "warm-up", "timed", "timed", "traced"]
    for request, steps in zip(data["raw"]["requests"], (6, 2, 6, 6, 6)):
        assert request["t_request_s"] == 0.0 and len(request["t_launch_s"]) == len(request["t_token_s"]) == steps
        assert 0.0 < request["t_first_s"] < request["t_launch_s"][0] < request["t_token_s"][0] < request["t_token_s"][-1], "relative to the request"
    assert len({request["ids_sha256"] for request in data["raw"]["requests"] if request["kind"] != "warm-up"}) == 1, "the same ids in every request of 6 steps"
    protocol = data["protocol"]
    assert (protocol["warmup_steps"], protocol["timed_steps"], protocol["timed_requests"]) == (2, 6, 2) and protocol["launch_path"] == "python"
    assert (protocol["new_tokens_per_request"], protocol["intervals_per_request"], protocol["stop_rule"]) == (7, 6, "forced")
    assert (protocol["first_timed_position"], protocol["last_timed_position"]) == (5, 10) and protocol["prompt"]["tokens"] == 5
    assert set(data["memory"]) == {"weights_gib", "weights_gb", "state_gib", "peak_reserved_gib", "peak_allocated_gib", "process_peak_reserved_gib"}
    assert data["memory"]["weights_gb"] == pytest.approx(data["memory"]["weights_gib"] * 2**30 / 1e9) and data["memory"]["peak_reserved_gib"] is None
    stages = "the checked request_s", "the checks before_s", "the warm-up request_s", "the timed requests_s", "the traced request_s", "the checks after_s"
    assert all(stage in data["startup"] and data["startup"][stage] >= 0.0 for stage in stages) and "warm-up_s" in data["startup"]
    assert (
        data["observation"]["source"] == "none"
        and data["environment"]["driver"]["value"] is None
        and data["environment"]["cudnn_frontend"]["version"] == "1.34.0"
    )
    assert data["text_path"]["render_s"] == 0.001 and data["text_path"]["tokens_written"] == 0
    # a value that is no number is not written as one: the file is refused, and nothing is left behind a name that exists
    refusals = [perf.write_report(str(tmp_path / name), value) for name, value in (("report.json", {}), ("nan.json", {"value": float("nan")}))]
    assert refusals[0] == "--perf-report report.json: File exists" and refusals[1].startswith("--perf-report nan.json: ")
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == data, "the file that was there is the file that is there"
    missing = perf.write_report(str(tmp_path / "missing" / "r.json"), data)
    assert missing == "--perf-report r.json: No such file or directory"
    assert re.fullmatch(r"report-\d{8}T\d{6}Z\.json", perf.default_report_name())
    _says("area", "is strict JSON with units in its field names and no local path:", "`statistics` (per statistic its definition,")
    paragraph = _flat(_TEXTS["area"].read_text(encoding="utf-8")).split("**The report file**", 1)[1].split("The report, with dots", 1)[0]
    assert re.findall(r"`([a-z_]+)`(?= \(|,| and|\.)", paragraph.split("no local path:", 1)[1]) == sections.split()[4:], "the sections, in the file's order"
    _says("guide", "One definition per statistic, written in the report file, with a host test of its arithmetic on constructed time stamps.")
    _says("area", "**The statistics**, each with one definition, which the report file carries beside the number:")
    _says("area", "in GiB (2^30 bytes), where an ordinary run's `bytes per step` line is in GB")


def _schema_3(data):
    """The report file of the same run as the mode wrote it before it observed the GPU, schema
    ``model-sample-performance/3``: 18 sections and 16 statistics -- no statistic of the GPU and the host, no load of the
    host, an ``observation`` section of five entries and a driver entry that say the GPU was not observed, and nothing of
    a sampler among what a timed request ran with."""
    import copy

    older = copy.deepcopy(data)
    older["schema"] = "model-sample-performance/3"
    unobserved = "this version does not observe the GPU"
    del older["statistics"]["gpu_and_host"], older["environment"]["host"]["load"], older["protocol"]["in_a_timed_request"]["sampler"]
    older["environment"]["driver"] = {"value": None, "reason": f"not observed ({unobserved})"}
    older["observation"] = {"source": "none", "reason": unobserved, "reason_short": "this version", "other_compute_processes": None, "clocks": None}
    return older


def _schema_1(data):
    """From a file of schema ``model-sample-performance/2`` (``_schema_2``), the report file of the same run as the mode
    wrote it before it took the GPU side of a step from the traced request, schema ``model-sample-performance/1``: 17
    sections and 10 statistics -- no ``per_kernel`` section, none of the six statistics of the GPU side, a ``traced``
    section of seven entries that says GPU busy was not taken, and the definition of the step with the sentence that
    said so."""
    import copy

    older = copy.deepcopy(data)
    older["schema"] = "model-sample-performance/1"
    del older["per_kernel"]
    for key in ("gpu_busy", "launch_overhead", "tracing_stretch", "kernel_launches", "kernel_time_per_launch", "kernel_time_per_step"):
        del older["statistics"][key]
    older["statistics"]["step"]["definition"] += " This version does not take GPU busy from the traced request, so the step is not taken apart."
    for key in ("per_step", "tracing_stretch_percent", "time_resolution_us"):
        del older["traced"][key]
    older["traced"].update(gpu_busy_ms=None, gpu_busy_reason="this version")
    return older


def _schema_2(data):
    """The report file of the same run as the mode wrote it while launch overhead was the median step less the median GPU
    busy less the median host time, schema ``model-sample-performance/2``: the sections and the statistics of the
    file it is handed (this version's, or those of ``_schema_3``), and a ``launch_overhead`` of seven entries -- no
    label, no count, median, mean, extremes or percentiles of per-step remainders -- whose remainder is the difference
    of the three medians, under the definition that said so.  The same field names as that file's, and other numbers
    behind them."""
    import copy

    older = copy.deepcopy(data)
    older["schema"] = "model-sample-performance/2"
    stats = older["statistics"]
    overhead, busy = stats["launch_overhead"], stats["gpu_busy"]["value"]
    for key in ("label", "n", "median", "mean", "minimum", "maximum", "p90", "p99"):
        overhead.pop(key, None)
    remainder = None if busy is None else stats["step"]["value"] - busy - stats["host"]["value"]
    term = remainder is not None and remainder > busy
    overhead.update(value=remainder if term else None, remainder_ms=remainder)
    overhead["host_time_per_block_call_us"] = None if remainder is None else remainder * 1e3 / len(_tiny_model().wiring())
    overhead["definition"] = "step - GPU busy - host: time inside a step in which the GPU waited for the next launch, or the host for the read."
    return older


def test_a_file_of_another_schema_is_refused_in_one_sentence(monkeypatch, tmp_path):
    """The report file names its schema in its first field, and the screen is rendered from the data of a file of that
    schema alone.  Handed a file of another schema -- the file of the same run as the mode wrote it before it observed
    the GPU, as it was while launch overhead came from three medians, and as the mode wrote it before it took the GPU
    side of a step, each read back from disk; this version's own file with the launch overhead of three medians under
    that file's name, whose every field name this version reads; a file of this version's own shape under the next
    name and under a name of something else; a file that names no schema; something that is no report file --
    ``perf.screen`` renders nothing and says so in one sentence that names both schemas: a ``ValueError``, never the
    ``KeyError`` or the ``TypeError`` of an entry the other schema does not have."""
    from model_samples import perf

    outcome, lines, events, _ = _mode(monkeypatch, tmp_path)
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert list(data)[0] == "schema" and data["schema"] == perf.SCHEMA == "model-sample-performance/4"
    _says("area", f"The file's first field names its schema, `{perf.SCHEMA}`.")
    assert perf.screen(data) == perf.screen(outcome.data) and perf.screen(data)[0] in lines, "a file of this schema is rendered: the screen of its run"
    _says("area", "The screen is rendered from the data of a file of this schema (`screen` in `model_samples/perf.py`);")
    refusal = "a report file of schema {} is not read by this version of the performance mode, which reads and writes 'model-sample-performance/4'"

    def refused(file):
        with pytest.raises(ValueError) as said:  # a KeyError is not one: it would pass through and fail the test
            perf.screen(file)
        return str(said.value)

    # the files of the three versions before, each written and read back as a file: 18 sections and 16 statistics twice, and 17 and 10,
    # where this version has 18 and 17
    third = _schema_3(data)
    second = _schema_2(third)
    older = _schema_1(second)
    assert [(len(file), len(file["statistics"])) for file in (older, second, third, data)] == [(17, 10), (18, 16), (18, 16), (18, 17)]
    assert list(older["traced"])[-2:] == ["gpu_busy_ms", "gpu_busy_reason"] and len(older["traced"]) == 7
    assert list(third) == list(second) == list(data) and len(third["observation"]) == 5 and list(third["protocol"]["in_a_timed_request"])[-1] == "text"
    for number, file in ((1, older), (2, second), (3, third)):
        assert file["schema"] == f"model-sample-performance/{number}" and perf.write_report(str(tmp_path / f"schema_{number}.json"), file) == ""
        read = json.loads((tmp_path / f"schema_{number}.json").read_text(encoding="utf-8"))
        assert read == file and read["observation"]["other_compute_processes"] is None, "what the screen of this version would take apart"
        assert refused(read) == refusal.format(f"'model-sample-performance/{number}'")
        assert refused(read).count("model-sample-performance/") == 2 and "\n" not in refused(read), "one sentence, both names"
    assert "per_kernel" not in older and "gpu_busy" not in older["statistics"], "and what it would look for"
    # the file that took launch overhead from three medians: every section, every statistic and every field name this version reads is
    # there, and the numbers behind three of them are others -- it is refused by its name, not rendered under this version's
    as_two = _schema_2(data)
    assert (len(as_two), len(as_two["statistics"]), list(as_two)) == (18, 17, list(data)) and perf.write_report(str(tmp_path / "as_two.json"), as_two) == ""
    was, now = as_two["statistics"]["launch_overhead"], data["statistics"]["launch_overhead"]
    assert set(now) - set(was) == {"label", "n", "median", "mean", "minimum", "maximum", "p90", "p99"} and not set(was) - set(now)
    assert refused(json.loads((tmp_path / "as_two.json").read_text(encoding="utf-8"))) == refusal.format("'model-sample-performance/2'")
    form = _checkpoint_form()  # made-up parts: a median step of 400 ms with medians of 80 and 40 beside it, a mean step of 410 with means of 82 and 41
    was, now = _schema_2(form)["statistics"]["launch_overhead"], form["statistics"]["launch_overhead"]
    assert (was["remainder_ms"], was["value"]) == (280.0, 280.0) and (now["remainder_ms"], now["value"]) == (287.0, 287.0), "one field name, two numbers"
    assert refused(_schema_2(form)) == refusal.format("'model-sample-performance/2'") and perf.screen(form)
    # a name this version does not know, on a file of this version's own shape: the name decides, and nothing is rendered
    later = f"model-sample-performance/{int(perf.SCHEMA.rsplit('/', 1)[1]) + 1}"
    for name in (later, perf.SCHEMA + "0", "kernel-census/2", "model-sample-performance", 2):
        assert refused({**data, "schema": name}) == refusal.format(repr(name)), name
    assert refused({key: value for key, value in data.items() if key != "schema"}) == refusal.format("None"), "a file that names no schema"
    assert refused([]) == refused({}) == refusal.format("None"), "and what is no report file at all"
    _says("area", "handed a file of another schema, written by another version of the mode, it renders nothing and says so")
    _says("area", "it renders nothing and says so in one sentence that names both schemas.")
    _says("area", "never written over, its schema, and a file of another schema refused in one sentence that names both);")


def test_a_failed_check_prints_no_rate(monkeypatch, tmp_path):
    """A statement of the checks that does not hold: its FAIL on the line, the rate line says why no rate is printed, the
    reason stands in the record class, the file is written with the stamps, and the run does not pass.  A timed request
    that computes other tokens; traced steps that launch another kernel than the plan's; kernels that could not be
    observed (reported, never PASS); a reference check that fails before anything is timed (no file)."""
    from model_samples import perf

    defect = {}

    def other_tokens(line):  # from the warm-up request on, the token choice returns another id
        if "a warm-up request" in line:
            defect["token"] = True

    for name in ("tokens", "kernel", "unobserved", "whole", "traced", "last", "state", "weights", "replay", "before"):
        (tmp_path / name).mkdir()
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path / "tokens", defect=defect, on_say=other_tokens)
    assert not outcome.passed and outcome.data["checks"]["timed_requests"] == "FAIL (request 1, step 0)"
    at = lines.index("            timed requests give the checked request's tokens and state: FAIL (request 1, step 0)")
    assert lines[at + 1] == "            kernels, traced steps: PASS", "each statement keeps its own verdict"
    assert "decode rate           not printed: the timed requests did not compute the checked request's tokens" in lines
    assert "a check did not hold" in outcome.data["record"]["reasons"] and outcome.data["checks"]["last_line"] == "[00] FAIL"
    assert (tmp_path / "tokens" / "report.json").exists() and len(outcome.data["raw"]["requests"]) == 5, "the stamps are kept for the diagnosis"
    _says("area", "A statement that does not hold prints its `FAIL`, the rate line says why no rate is printed, and the run ends in `[00] FAIL`;")
    _says("area", "Not printed for a reduced model, nor where a statement of the `checks` lines does not hold |")
    _says("guide", "a run in which one of the three does not hold prints no rate, and kernels that could not be observed are reported as not observed,")
    form = _checkpoint_form(checks=outcome.data["checks"])
    form["statistics"]["decode_rate"].update(outcome.data["statistics"]["decode_rate"], reason=f"not printed: {perf.NOT_COMPUTED}")
    assert not any("tokens/s" in line for line in perf.screen(form)), "on a checkpoint too: a rate needs the three statements"

    defect.clear()

    def other_kernel(line):  # from the warm-up request on, the fused step launches another form's kernel
        if "a warm-up request" in line:
            defect["symbol"] = OTHER_SYMBOL

    outcome, lines, events, _ = _mode(monkeypatch, tmp_path / "kernel", defect=defect, on_say=other_kernel)
    assert not outcome.passed and outcome.data["checks"]["timed_requests"] == "PASS"
    assert outcome.data["checks"]["kernels"].startswith("FAIL (") and outcome.data["checks"]["kernels"].endswith(": the kernels lines above name them)")
    assert outcome.data["statistics"]["decode_rate"]["reason"] == f"not printed: {perf.NOT_THE_PLAN}"
    assert any(line.startswith("  kernels           ") and line.endswith("    FAIL") for line in lines) and outcome.data["checks"]["last_line"] == "[00] FAIL"
    form = _checkpoint_form(checks=outcome.data["checks"])
    form["statistics"]["decode_rate"].update(value=None, reason=f"not printed: {perf.NOT_THE_PLAN}")
    assert "decode rate           not printed: the kernels of the traced steps are not the plan's" in perf.screen(form)

    defect.clear()
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path / "unobserved", profiler=NO_PROFILER)
    assert outcome.passed and outcome.unobserved == NO_PROFILER and outcome.data["checks"]["kernels"] == f"NOT OBSERVED ({NO_PROFILER})"
    assert "kernels not observed" in outcome.data["record"]["reasons"] and "a check did not hold" not in outcome.data["record"]["reasons"]
    at = lines.index(_line(lines, "step "))
    step = lines[at] + " " + lines[at + 1].strip()
    assert step.startswith("step                  reduced model: ") and step.endswith(f"ms over every timed step; GPU busy not measured ({NO_PROFILER})")
    assert outcome.data["checks"]["last_line"] == f"[00] RAN (kernels not observed: {NO_PROFILER})" and events.count("profiler") == 1
    assert outcome.data["traced"]["steps"] == 0 and outcome.data["kernels"]["kernels_per_step"] is None
    assert "what a step launches was not observed" in lines[lines.index(_line(lines, "kernels     ")) + 1]
    traced = outcome.data["traced"]
    assert outcome.data["protocol"]["profiler_first_started"] is None and traced["first_position"] is None and traced["last_position"] is None
    # ... and on a model that is not a reduced one the rate stays, as orientation, with that reason on line 2
    whole = _tiny_model()
    whole.reductions = ()
    with monkeypatch.context() as patch:
        patch.setattr(sys.modules[__name__], "_tiny_model", lambda: whole)
        outcome, lines, events, _ = _mode(patch, tmp_path / "whole", profiler=NO_PROFILER)
    assert _line(lines, "decode rate (orientation)   ") and "kernels not observed" in " ".join(lines[lines.index(_line(lines, "record class ")) :][:3])
    assert outcome.data["checks"]["last_line"] == f"[00] RAN (kernels not observed: {NO_PROFILER})" and outcome.data["statistics"]["decode_rate"]["value"] > 0
    _says("area", "the run keeps its rate as `orientation` with the reason `kernels not observed`, and it ends in")
    _says("guide", "the rate staying `orientation` with that reason.")

    # the traced request alone computes other tokens: it is held against the checked request too
    defect.clear()
    defect["at the first profiler pass"] = {"token": True}
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path / "traced", defect=defect)
    assert not outcome.passed and outcome.data["checks"]["timed_requests"] == "FAIL (traced request, step 1)"
    assert outcome.data["statistics"]["decode_rate"]["reason"] == "not printed: the traced request did not compute the checked request's tokens"

    # a token that differs at the last step alone, of the first timed request: one token choice of the load's own first launch, then the
    # checked request's 5 + 6, the replay's 11, the warm-up request's 5 + 2, the request's 11
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path / "last", defect={"token": _Nth(1 + 11 + 11 + 7 + 11)})
    assert not outcome.passed and outcome.data["checks"]["timed_requests"] == "FAIL (request 1, step 6)"
    assert outcome.data["statistics"]["decode_rate"]["reason"] == f"not printed: {perf.NOT_COMPUTED}"

    # a timed request that gives the same tokens and leaves another state
    defect.clear()
    digest, taken = perf.state_digest, []

    def another_state(state):
        taken.append(digest(state))
        return "0" * 64 if len(taken) == 3 else taken[-1]  # the checked request, the warm-up request, then the first timed request

    monkeypatch.setattr(perf, "state_digest", another_state)
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path / "state")
    monkeypatch.setattr(perf, "state_digest", digest)
    assert not outcome.passed and outcome.data["checks"]["timed_requests"] == "FAIL (request 1, end state)"
    assert outcome.data["statistics"]["decode_rate"]["reason"] == "not printed: a timed request did not leave the checked request's state"

    # a stored weight that no longer holds the bytes of its load behind the timed requests (the checks before found it whole, twice)
    asked = []

    def written_to(weights):
        asked.append(weights)
        return ["mlp.down of layer 0"] if len(asked) > 2 else []

    monkeypatch.setattr(perf.tie_out, "changed_weights", written_to)
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path / "weights")
    assert not outcome.passed and outcome.data["checks"]["timed_requests"] == "FAIL (stored weights: mlp.down of layer 0)" and len(asked) == 3
    assert outcome.data["statistics"]["decode_rate"]["reason"] == "not printed: a stored weight no longer holds the bytes of its load"
    monkeypatch.undo()

    # a replay that gives another token at its last step alone, the 67th, far behind the 16 steps of the tokens' own pass: nothing is timed behind it
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path / "replay", steps=62, requests=1, warmup=1, defect={"token": _Nth(1 + 67 + 67)})
    assert not outcome.passed and outcome.data is None and any("  replay " in line and line.endswith("FAIL") for line in lines)
    assert sum(event[0] == "step" for event in events) == 2 * 67 and "profiler" not in events
    monkeypatch.undo()

    events_before = []
    session, text = _session(monkeypatch, events_before, capacity=13)
    choice = session.plan["token_choice"]
    calls = {"count": 0}
    real_call = type(choice).__call__

    def drifting(self, logits):  # another token every time the same step runs again: the replay cannot give the recorded ids
        calls["count"] += 1
        return (real_call(self, logits) + calls["count"]) % logits.shape[-1]

    monkeypatch.setattr(type(choice), "__call__", drifting)
    said = []
    failed = perf.run(
        session,
        text,
        lambda: [text.encode("ab0xy")],
        perf.Protocol(2, 6, 2),
        lambda *lines_said, flush=False: said.extend(lines_said),
        name="tiny model, reduced model",
        facts=None,
        environment=None,
        sink=_Sink(),
        report=str(tmp_path / "before" / "report.json"),
        verdict=lambda passed, judged, unobserved: "PASS" if passed else "FAIL",
        real=False,
    )
    assert not failed.passed and failed.data is None and not (tmp_path / "before" / "report.json").exists(), "nothing timed, no file"
    assert any("  replay " in line and line.endswith("FAIL") for line in "\n".join(said).split("\n")) and "profiler" not in events_before
    assert not any("decode performance report" in line for line in said)
    _says("area", "a check of the checked request that fails ends the run before anything is timed, and no file is written.")
    _says("area", "the timed requests and the traced request gave the checked request's token ids, step for step, every timed request left the")
    _says("area", "state in the same bytes and the stored weights still hold the bytes of their load; the kernels of the traced steps are the plan's.")


# ---------------------------------------------------------------------------------------------- the GPU side: GPU busy, the step taken apart, the per-kernel table
COPY_IN = "Memcpy HtoD (Pageable -> Device)"


def _records(sites, start=0.0, us=4.3, period=0.01, also=()):
    """Constructed launch records of one step for ``sites`` (``census.Site``), in start order: per site one record per planned
    launch of an exact entry, named after its kernel; one per family of a fallback or host entry; one copy behind a
    fallback site; then the host read.  A record lasts ``us`` microseconds and one starts every ``period`` ms from
    ``start``, so none overlaps.  ``also``: ``(index, name)`` -- another kernel, or another copy, launched behind the
    records of that site."""
    from model_samples.census import FAMILIES, Launch

    names = {"embedding_gather": GATHER, "gdn_decode_fused": SYMBOL, "pytorch": ELEMENTWISE, "cublas": NVJET}
    names.update({"cublaslt_fp8_gemm": NVJET_FP8, "fi_mm_fp4": FI_MM_FP4})  # the exact kernels of the rows a library serves
    names.update({kernel: pattern.pattern.lstrip("^") + "Cfg" for kernel, pattern in FAMILIES if pattern.pattern.startswith("^cudnn_kernel_frost_")})
    records = []

    def launch(name, copy=False):
        at = start + period * len(records)
        shape = () if copy else ((1, 1, 1), (128, 1, 1), 0)
        records.append(Launch(name, "copy" if copy else "kernel", at, at + us / 1e3, *shape, duration=us))

    for index, site in enumerate(sites):
        for kernel in site.kernels:
            for _ in range(kernel.launches or 1):
                launch(names[kernel.id])
        if any(kernel.relation == "fallback" for kernel in site.kernels):
            launch(COPY_IN, copy=True)
        for name in [name for at, name in also if at == index]:
            launch(name, copy=name.startswith("Memcpy"))
    launch(HOST_READ, copy=True)
    return records


def _walked(sites, steps, claim=None):
    """What a session keeps of a traced pass, from constructed records: the kernel check's walk over ``steps`` (each a list
    of launch records that ends in its host read) against ``sites``, and its verdict; and GPU busy of every step."""
    from model_samples import census, timing

    traced = census.Traced(list(sites), [], steps=[list(step) for step in steps], claim=claim)
    traced.result = census.compare(traced.steps, traced.sites)
    traced.status = census.FAIL if traced.result.failures else census.PASS
    return traced, [timing.step_facts([(launch.name, launch.start, launch.end) for launch in step])[0][0] for step in steps]


def _site(block, layer=-1, *kernels):
    """A block call of a step with census entries by name: ``pytorch`` and ``cublas`` are fallback families, ``host`` the
    host path, any other name an exact kernel of one launch per call."""
    from model_samples.census import Kernel, Site

    family = {"pytorch": Kernel("pytorch", "PyTorch"), "cublas": Kernel("cublas", "cuBLAS"), "host": Kernel("pytorch", "PyTorch", "host")}
    return Site(block, layer, tuple(family.get(name) or Kernel(name, "cuDNN FE", "exact", launches=1) for name in kernels), entry=f"entry of {block}")


def _planned(monkeypatch, stack="native"):
    """``(sites, claim)`` of a step of the 64-layer model on ``stack``, compute capability 10.0 with 148 SMs, planned on
    stated facts: the family's own wiring, each call with the census entries of the row that serves it."""
    from model_samples import Session, TextIO, blocks, census
    from model_samples.qwen_dense import QwenDense

    monkeypatch.setattr(census, "installed_versions", lambda packages: {"torch": "2.14.0+cu130", "nvidia-cutlass-dsl": "4.8.0"})
    model = QwenDense()
    session = Session(model, TextIO.synthetic(272), stack=stack, device="cpu", capacity=8)
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    session.plan = blocks.plan(model.provider_table, model.regimes, facts, 1, 1, stack, model=model, options={"device": "cpu"})
    parts = [(name, layer, at, session.plan.at(name, at)[1]) for (name, layer), at in zip(model.wiring(), census.call_sites(model))]
    return [census.Site(name, layer, tuple(block.kernels), block.entry, site=at) for name, layer, at, block in parts], session.claim()


def _fine_stamps(monkeypatch, behind=0):
    """For ``_mode(on_say=...)``: from the run's first line on, the stand-in profiler hands over records of 0.4375 us, one
    every microsecond -- stamps finer than a microsecond, and a GPU busy far below the step of a session on the CPU.
    ``behind``: the first so many profiler passes of the run keep the stand-in's own stamps, whole microseconds."""
    from model_samples import loop

    passes = []

    def install(line):
        coarse = loop.trace_steps
        if getattr(coarse, "fine", False):
            return

        def fine(step, count, device=None, records=None):
            mine = []
            made = coarse(step, count, device, mine)
            passes.append(count)
            if records is not None and len(passes) <= behind:
                records.extend(mine)
            elif records is not None:
                records.extend(record._replace(start=record.start / 1e3, end=record.start / 1e3 + 0.0004375, duration=0.4375) for record in mine)
            return made

        fine.fine = True
        monkeypatch.setattr(loop, "trace_steps", fine)

    return install


def _corresponds(key, *phrases):
    """The serving counterpart the report file carries beside the statistic ``key`` holds every one of ``phrases``; with
    none given, the statistic has no counterpart."""
    from model_samples import perf

    counterpart = perf.DEFINITIONS[key][2]
    assert (counterpart is None) == (not phrases), key
    for phrase in phrases:
        assert phrase in counterpart, f"{key}: {phrase}"


def test_added_time_partitions_the_union():
    """GPU busy of a step is the length of the union of the intervals of its kernel and copy records: a stretch in which
    two records run is counted once.  The time a launch adds to the GPU's timeline is its end minus the later of its
    start and the latest end of any earlier record, not below zero; the added times of a step are the terms of that
    union, so their sum IS the step's GPU busy."""
    from model_samples import perf, timing
    from model_samples.census import Launch

    spans = [(0.0, 1.0), (0.5, 1.5), (0.625, 0.875), (1.5, 2.0), (3.0, 3.25)]  # ms: overlapping, nested, touching, behind a gap
    step = [Launch(ELEMENTWISE, "kernel", start, end) for start, end in spans] + [Launch(HOST_READ, "copy", 4.0, 4.125)]
    assert perf.added_ms(step) == [1.0, 0.5, 0.0, 0.5, 0.25, 0.125], "the overlapped half once, the nested record nothing, the read its own time"
    ((busy, idle, kernels, copies),) = timing.step_facts([(launch.name, launch.start, launch.end) for launch in step])
    assert busy == sum(perf.added_ms(step)) == 2.375 and idle == 4.125 - 2.375 and (kernels, copies) == (5, 1)
    assert sum(end - start for start, end in spans) + 0.125 == 3.125, "the sum of the durations counts the overlapped stretches twice"
    late = [
        Launch(ELEMENTWISE, "kernel", 0.0, 2.0),
        Launch(NVJET, "kernel", 0.5, 1.0),
        Launch(ELEMENTWISE, "kernel", 1.5, 2.5),
        Launch(HOST_READ, "copy", 2.5, 2.75),
    ]
    assert perf.added_ms(late) == [2.0, 0.0, 0.5, 0.25], "the latest end of ANY earlier record, not of the one before"
    assert perf.added_ms([]) == [] and perf.added_ms(step[:1]) == [1.0]
    # two exact rows whose launches overlap: the second adds 2 us of the 7 it lasts, and the file holds both figures
    sites = [_site("x", 0, "gate_quant"), _site("y", 0, "swiglu_fp4")]
    kinds = _records(sites)
    overlapping = [kinds[0]._replace(start=0.0, end=0.010, duration=10.0), kinds[1]._replace(start=0.005, end=0.012, duration=7.0)]
    overlapping.append(kinds[2]._replace(start=0.020, end=0.021, duration=1.0))
    rows = perf.kernel_table(*_walked(sites, [overlapping]))["rows"]
    assert [(row["us_per_launch_added"], row["us_per_launch_recorded"]) for row in rows] == [(10.0, 10.0), (pytest.approx(2.0), 7.0), (pytest.approx(1.0), 1.0)]
    assert sum(row["ms_per_step"] for row in rows) == pytest.approx(0.013) and rows[1]["share_percent"] == pytest.approx(200 / 13)
    # the screen's us/launch is that added time, not the recorded duration: 2.5 us of the 7.5 us the second launch lasts
    finer = [overlapping[0], overlapping[1]._replace(end=0.0125, duration=7.5), overlapping[2]]
    printed = perf._kernel_lines(perf.kernel_table(*_walked(sites, [finer])), "")
    assert [line.split()[:3] for line in printed[3:5]] == [["x", "1", "10.00"], ["y", "1", "2.50"]], "launches, then us/launch"
    # a row of three launches a step, of 1, 2 and 9 us: the median over the row's launches, of the added and of the recorded time
    thrice = [_site("x", layer, "gate_quant") for layer in range(3)]
    uneven = [record._replace(end=record.start + us / 1e3, duration=us) for record, us in zip(_records(thrice), (1.0, 2.0, 9.0, 4.5))]
    row = perf.kernel_table(*_walked(thrice, [uneven]))["rows"][0]
    assert (row["launches_per_step"], row["us_per_launch_added"], row["us_per_launch_recorded"]) == (3, pytest.approx(2.0), 2.0), "not their mean, 4 us"
    _defines("gpu_busy", "the length of the union of the intervals of every kernel and copy between two host reads", "the median over the traced steps")
    _defines("gpu_busy", "An upper bound: tracing lengthens kernels.")
    _corresponds("gpu_busy", "The same method as the kernel-busy sum of a traced serving step.")
    _defines("kernel_time_per_launch", "the median over the row's launches in all traced steps.", "The file also holds the median recorded duration.")
    _says("area", "the latest end of any earlier launch of the step, the median over the row's launches;")
    _corresponds("kernel_time_per_launch", "The same method as the timeline-exclusive duration of a per-kernel analysis of a traced serving step;")
    _corresponds("kernel_time_per_launch", "comparable only on one GPU class, SM count, clock and launch path.")
    _says("area", "from kernel activity records, the median over the traced steps (an upper bound: tracing lengthens kernels);")
    _defines("kernel_time_per_launch", "its end minus the later of its start and the latest end of any earlier record of the step, not below zero")
    _defines("kernel_time_per_step", "Within one step the added times of all records sum exactly to that step's GPU busy.")
    _says("area", "GPU busy = the union of the intervals of every kernel and copy of a traced step, from kernel activity records")
    _says("area", "`us/launch`: the time a launch adds to the GPU's timeline: its end less the later of its start and the latest end of any earlier launch")


def test_step_equation_only_where_the_remainder_exceeds_gpu_busy():
    """step = the median interval at the traced positions; host = from a read to the next launch.  The remainder of a
    timed step = the step - GPU busy of the traced step at the same position - host; of a run the mean of them, which
    adds up with the means of the same steps.  Launch overhead is a term of the ``step`` line only where the remainder
    is larger than GPU busy, and is called an estimate there; otherwise the line gives the mean GPU busy as an upper
    bound and says that the rest is not resolved.  The file holds the remainder as computed, also where it is negative.
    The stretch of tracing is the traced request's time in a step over the timed requests' at the same positions,
    minus 1."""
    from model_samples import perf

    alike = ([[10.0] * 3], [[1.0] * 3])  # one timed request: three steps of 10 ms, 1 ms of each the loop's host time
    assert perf.step_split(*alike, [2.0] * 3, 9.0, 9.9) == {"remainder_ms": 7.0, "term": True, "stretch": pytest.approx(0.1), "remainders": [7.0] * 3}
    assert perf.step_split(*alike, [6.0] * 3)["remainder_ms"] == 3.0 and not perf.step_split(*alike, [6.0] * 3)["term"], "GPU busy is the larger part"
    assert perf.step_split(*alike, [4.5] * 3)["remainder_ms"] == 4.5 and not perf.step_split(*alike, [4.5] * 3)["term"], "equal is not larger"
    passed = {"remainder_ms": -2.0, "term": False, "stretch": None, "remainders": [-2.0] * 3}
    assert perf.step_split(*alike, [11.0] * 3) == passed, "an upper bound can pass the untraced step"
    assert perf.step_split(*alike, None) == {"remainder_ms": None, "term": False, "stretch": None, "remainders": []}
    # ten-millisecond steps, each launched 1 ms behind the read before it; the last step of each request is longer and not traced
    timed = [_request(0.0, [0.010, 0.020, 0.030, 0.090]), _request(0.0, [0.010, 0.020, 0.030, 0.090])]
    traced = _request(0.0, [0.0109, 0.0218, 0.0327], kind="traced", launches=[0.001, 0.0119, 0.0228])  # in a step: 9.9 ms against 9
    assert traced.in_step_ms() == pytest.approx([9.9, 9.9, 9.9]) and timed[0].in_step_ms() == pytest.approx([9.0, 9.0, 9.0, 59.0])
    stats = perf._statistics(timed, 3, 1, "", busy=[2.0, 2.5, 2.0], traced=traced, calls=35)
    assert stats["step"]["value"] == pytest.approx(10.0) and stats["host"]["value"] == pytest.approx(1.0)
    assert stats["gpu_busy"]["value"] == 2.0 and (stats["gpu_busy"]["n"], stats["gpu_busy"]["maximum"]) == (3, 2.5), "the median over the traced steps"
    overhead = stats["launch_overhead"]
    assert overhead["value"] == overhead["remainder_ms"] == overhead["mean"] == pytest.approx(41 / 6), "the mean of 7, 6.5 and 7 ms, in both timed requests"
    assert (overhead["n"], overhead["median"], overhead["minimum"]) == (6, pytest.approx(7.0), pytest.approx(6.5)), "every timed step at a traced position"
    assert overhead["host_time_per_block_call_us"] == pytest.approx(41 / 6 * 1e3 / 35) and overhead["label"] == perf.ESTIMATE
    assert stats["step"]["mean"] == pytest.approx(
        stats["gpu_busy"]["mean"] + overhead["remainder_ms"] + stats["host"]["mean"], rel=1e-12
    ), "10 = 13/6 + 41/6 + 1"
    assert stats["tracing_stretch"]["value"] == pytest.approx(10.0) and stats["tracing_stretch"]["clock"] == perf.CLOCK
    slow = _request(0.0, [0.010, 0.070, 0.130, 0.190], launches=[0.001, 0.011, 0.071, 0.131])  # in a step: 9 ms, then 59 ms three times
    longer = _request(0.0, [0.0109, 0.0518], kind="traced", launches=[0.001, 0.0119])  # in a step: 9.9 ms at the traced position, 39.9 ms behind it
    first = perf._statistics([slow, slow], 1, 1, "", busy=[2.0], traced=longer)
    assert first["tracing_stretch"]["value"] == pytest.approx(10.0), "both sides at the traced position, not over all their steps"
    assert first["launch_overhead"]["remainder_ms"] == pytest.approx(7.0), "the step at the traced position (10 ms), not the interval over every timed step"
    # the third traced position is a long step, in the traced request and in the timed ones: a median on both sides of the stretch
    long_last = _request(0.0, [0.0109, 0.0218, 0.0627], kind="traced", launches=[0.001, 0.0119, 0.0228])  # in a step: 9.9, 9.9 and 39.9 ms
    uneven = [_request(0.0, [0.010, 0.020, 0.060], launches=[0.001, 0.011, 0.021])] * 2  # in a step: 9, 9 and 39 ms
    medians = perf._statistics(uneven, 3, 1, "", busy=[2.0, 3.0, 9.0], traced=long_last)
    assert medians["tracing_stretch"]["value"] == pytest.approx(10.0), "9.9 ms over 9 ms: not the means (19.9 over 19), not one position"
    assert medians["gpu_busy"]["value"] == 3.0 and medians["gpu_busy"]["mean"] == pytest.approx(14 / 3), "the median over the traced steps"
    _defines("tracing_stretch", "The median time from a step's first launch to its ids on the host in the traced request,")
    upper = perf._statistics(timed, 3, 1, "", busy=[11.0, 11.0, 11.0], traced=traced, calls=35)["launch_overhead"]
    assert upper["value"] is None and upper["remainder_ms"] == pytest.approx(-2.0) and upper["host_time_per_block_call_us"] < 0
    assert upper["reason"] == "not resolved by this method: the remainder is not larger than the mean GPU busy" and upper["label"] == perf.ESTIMATE
    unseen = perf._statistics(timed, 0, 1, "", unmeasured="this PyTorch build records no kernel activity")
    assert unseen["gpu_busy"]["reason"] == unseen["launch_overhead"]["reason"] == "GPU busy not measured (this PyTorch build records no kernel activity)"
    assert unseen["tracing_stretch"]["reason"] == unseen["gpu_busy"]["reason"] and unseen["gpu_busy"]["n"] == 0
    assert (unseen["launch_overhead"]["remainder_ms"], unseen["launch_overhead"]["label"], unseen["launch_overhead"]["n"]) == (None, None, 0)
    instant = _request(1.0, [1.010, 1.020], launches=[1.010, 1.020])  # each step's ids are on the host at the stamp of its launch
    still = perf._statistics([instant], 2, 1, "", busy=[1.0, 1.0], traced=traced)["tracing_stretch"]
    assert still["value"] is None and still["reason"] == "not formed: the timed requests spend no time in a step"
    _defines("launch_overhead", "Printed as a term of the step, and called an estimate there, only where it is larger than the mean GPU busy;")
    _defines("launch_overhead", "otherwise the line says that launch overhead and host are not resolved by this method.")
    _defines("launch_overhead", "The file holds the remainder as computed, which can be negative, the count, the median, the extremes and the percentiles of")
    _defines("launch_overhead", "the per-step remainders, and the launch overhead per block call.")
    _defines("tracing_stretch", "in the traced request, divided by the same in the timed requests at the same positions, minus 1.")
    _defines("launch_overhead", "It stands for time inside a step in which the GPU waited for the next launch, or the host for the read,")
    _corresponds("launch_overhead")
    _corresponds("tracing_stretch")
    _says("area", "(the remainder as computed is in the file).")
    _says("area", "A line under it says how many steps were traced, at which positions, their median GPU busy, and by how much tracing lengthened a step.")
    # form A: the step, its mean taken apart, the line that calls launch overhead an estimate, the traced steps and the note
    lines = perf.screen(_checkpoint_form())
    at = lines.index("step                  400.000  ms; mean 410.000 = GPU busy 82.000 + launch overhead 287.000 + host 41.000")
    assert lines[at + 1 : at + 4] == [
        "                      launch overhead: an estimate (GPU busy is the traced request's and an upper bound)",
        "                      (GPU busy from 51 traced steps, positions 58 to 108; median 80.000 ms;",
        "                      tracing lengthens a step by 5.0 %)",
    ]
    assert 410.0 == 82.0 + 287.0 + 41.0 and lines[at + 1].endswith(perf.ESTIMATE), "the terms of the line add up, and the label is the file's"
    note = lines.index("note                  launch overhead (an estimate) 70 % of the mean step = 410 us per block call: time in")
    assert lines[note + 1] == "                      this sample's call path from Python, not kernel time" and lines.index(_line(lines, "memory  ")) < note
    assert lines[note + 2] == "" and round(287.0 / 410.0 * 100) == 70, "the remainder over the mean step, the step the equation takes apart"
    _says("area", "its share of the mean step and the time per block call, named as the estimate it is,")
    _says("area", "and what it stands for -- time in this sample's call path from Python, not kernel time |")
    one = perf.screen(_checkpoint_form(traced={"steps": 1, "last_position": 58}))
    assert "                      (GPU busy from the traced step, position 58; median 80.000 ms;" in one
    # form B: GPU busy is not the smaller part -- the step and its mean, the mean GPU busy as the upper bound it is, no term for the rest; no note
    bound = perf.screen(_gpu_side(_checkpoint_form(), busy=240.0, term=False))
    at = bound.index("step                  400.000  ms (not traced), mean 410.000")
    assert bound[at + 1 : at + 5] == [
        "                      mean GPU busy 242.000 ms in a traced request (an upper bound: tracing lengthens kernels)",
        "                      launch overhead and host together: not resolved by this method",
        "                      (GPU busy from 51 traced steps, positions 58 to 108; median 240.000 ms;",
        "                      tracing lengthens a step by 5.0 %)",
    ]
    assert not any(
        line.startswith("note ") or "launch overhead 127" in line or "estimate" in line for line in bound
    ), "the remainder is in the file, not on the screen"
    assert (
        bound[at + 5].startswith("with draft tokens ")
        and _gpu_side(_checkpoint_form(), busy=240.0, term=False)["statistics"]["launch_overhead"]["remainder_ms"] == 127.0
    )
    unformed = perf.screen(_gpu_side(_checkpoint_form(), stretch=None))
    at = unformed.index(
        "                      (GPU busy from 51 traced steps, positions 58 to 108; median 80.000 ms;"
    )  # wider than a line: it breaks at a separator
    assert unformed[at + 1] == "                      the stretch of tracing was not formed)"
    reduced = perf.screen(_gpu_side(_data(), busy=1.0, host=1.0, mean=14.0))
    assert _line(reduced, "note   ").startswith("note                  reduced model: launch overhead (an estimate) 64 % of the mean step = ")
    at = reduced.index(_line(reduced, "step  "))  # the prefix of a reduced model makes the line wider than the screen: it breaks behind the step
    assert reduced[at].startswith("step                  reduced model: ") and reduced[at].endswith("  ms;")
    assert reduced[at + 1] == "                      mean 14.000 = GPU busy 3.000 + launch overhead 9.000 + host 2.000"
    _says("area", "*Launch overhead*: the part of a step in which the GPU waited for the next launch, or the host for the read.")
    _says("area", "It is an estimate: a timed step less GPU busy of the traced step at the same position and less the loop's own host time.")
    _says("area", "It is printed as a term only where it is larger than the mean GPU busy;")
    _says("area", "otherwise the line gives the mean GPU busy as the upper bound it is and says that launch overhead and host are not resolved by this method")
    _says("area", "| `note` | while launch overhead is a term of the step:")


def _stepped(hosts, insides, kind="timed"):
    """A request from constructed parts, in ms: per forced step the loop's host time behind the read before it, then the
    time from the step's first launch call to its ids on the host.  The request and its first token are at 0."""
    from model_samples import perf

    made, read = perf.Request(kind), 0.0
    for host, inside in zip(hosts, insides):
        made.t_launch.append(read + host / 1e3)
        read = made.t_launch[-1] + inside / 1e3
        made.t_token.append(read)
    made.ids = [[7]] * (len(made.t_token) + 1)
    return made


def test_the_remainder_is_formed_step_by_step_before_it_is_aggregated():
    """Launch overhead is what a timed step leaves beside GPU busy of the traced step at the SAME position and beside its
    own host time, formed step by step and averaged afterwards -- never the median step less the median GPU busy less
    the median host time: three medians of three lists need not be one step's, and their difference can be a time
    that no step has.  Steps that leave nothing report nothing, and the line then has the form without the term; every
    term the line prints is the mean over the same steps, so the terms add up; and the remainder is called an estimate
    wherever it is stored or printed.  Every number of this test is constructed, none is a run's."""
    from model_samples import perf

    def terms(stats):
        return stats["step"]["mean"], stats["gpu_busy"]["mean"], stats["launch_overhead"]["remainder_ms"], stats["host"]["mean"]

    def run(timed, busy, covered, traced=None, calls=0):
        stats = perf._statistics(timed, covered, 1, "", busy=busy, traced=traced or _stepped([1.0] * covered, [1.0] * covered, "traced"), calls=calls)
        step, gpu, left, host = terms(stats)
        assert step == pytest.approx(gpu + left + host, rel=1e-12, abs=1e-12), "in every case of this test: the mean step is the sum of its three terms"
        return stats, stats["launch_overhead"], perf._step_lines(stats, {"steps": covered, "first_position": 0, "last_position": covered - 1}, "")

    def of_medians(stats):  # what the three medians would say: the subtraction this test keeps out
        return stats["step"]["value"] - stats["gpu_busy"]["value"] - stats["host"]["value"]

    # three steps, each exactly its host time and its GPU busy: the long host time and the long GPU busy are different steps'
    hosts, busy = [100.0, 1.0, 1.0], [1.0, 1.0, 100.0]
    timed = _stepped(hosts, busy)
    assert timed.intervals_ms() == pytest.approx([101.0, 2.0, 101.0]) and timed.host_ms() == pytest.approx(hosts) and timed.in_step_ms() == pytest.approx(busy)
    stats, overhead, lines = run([timed], busy, 3, _stepped(hosts, busy, "traced"), calls=1)
    assert overhead["remainder_ms"] == pytest.approx(0.0, abs=1e-9) and overhead["value"] is None, "no step leaves a remainder: not 101 - 1 - 1 = 99 ms"
    assert of_medians(stats) == pytest.approx(99.0), "the three medians are those of three different steps"
    assert (overhead["n"], overhead["median"]) == (3, pytest.approx(0.0, abs=1e-9)) and max(abs(overhead["minimum"]), abs(overhead["maximum"])) < 1e-9
    assert overhead["reason"] == "not resolved by this method: the remainder is not larger than the mean GPU busy" and overhead["label"] == perf.ESTIMATE
    assert overhead["host_time_per_block_call_us"] == pytest.approx(0.0, abs=1e-6) and terms(stats) == pytest.approx((68.0, 34.0, 0.0, 34.0), abs=1e-9)
    assert lines == [
        "step                  101.000  ms (not traced), mean 68.000",
        "                      mean GPU busy 34.000 ms in a traced request (an upper bound: tracing lengthens kernels)",
        "                      launch overhead and host together: not resolved by this method",
        "                      (GPU busy from 3 traced steps, positions 0 to 2; median 1.000 ms;",
        "                      tracing lengthens a step by 0.0 %)",
    ], "the form without the term, and the two means side by side: half of the mean step is GPU busy, not a hundredth of the median one"
    # the three candidates apart: one step of three leaves 6 ms, and another one has the long GPU busy
    stats, overhead, lines = run([_stepped([1.0, 1.0, 1.0], [8.0, 2.0, 20.0])], [2.0, 2.0, 20.0], 3)
    assert stats["step"]["value"] == pytest.approx(9.0) and of_medians(stats) == pytest.approx(6.0), "9 - 2 - 1: a term of the line, by the three medians"
    assert overhead["median"] == pytest.approx(0.0, abs=1e-9), "the median of the per-step remainders: most steps leave nothing"
    assert overhead["remainder_ms"] == pytest.approx(2.0) and overhead["value"] is None, "their mean, 2 ms beside a mean GPU busy of 8: not a term"
    assert terms(stats) == pytest.approx((11.0, 8.0, 2.0, 1.0)) and lines[0] == "step                  9.000  ms (not traced), mean 11.000"
    # skewed host times: one step waits 28 ms for its launch and leaves 16 ms beside its GPU busy -- the medians see neither
    skewed = _stepped([1.0, 1.0, 28.0], [5.0, 5.0, 20.0])
    stats, overhead, lines = run([skewed], [4.0, 4.0, 4.0], 3, calls=12)
    assert stats["step"]["value"] == pytest.approx(6.0) and of_medians(stats) == pytest.approx(1.0), "6 - 4 - 1: not a term, by the three medians"
    assert overhead["median"] == pytest.approx(1.0) and overhead["value"] == overhead["remainder_ms"] == pytest.approx(6.0), "the mean of 1, 1 and 16 ms"
    assert terms(stats) == pytest.approx((20.0, 4.0, 6.0, 10.0)) and overhead["host_time_per_block_call_us"] == pytest.approx(500.0)
    assert lines[:2] == [
        "step                  6.000  ms; mean 20.000 = GPU busy 4.000 + launch overhead 6.000 + host 10.000",
        "                      launch overhead: an estimate (GPU busy is the traced request's and an upper bound)",
    ]
    mean, gpu, launch, host = (float(number) for number in re.findall(r"(?:mean|GPU busy|launch overhead|host) (\d+\.\d{3})", lines[0]))
    assert mean == pytest.approx(gpu + launch + host, abs=0.002), "the printed terms relate as the line says, to the rounding of three decimals"
    # the form is chosen on the means as well: a remainder of 5 ms beside GPU busy of 2, 2 and 20 ms -- above their median, below their mean
    stats, overhead, lines = run([_stepped([1.0] * 3, [7.0, 7.0, 25.0])], [2.0, 2.0, 20.0], 3)
    assert overhead["remainder_ms"] == pytest.approx(5.0) and overhead["value"] is None and stats["gpu_busy"]["value"] == 2.0, "5 is not larger than 8"
    assert lines[1] == "                      mean GPU busy 8.000 ms in a traced request (an upper bound: tracing lengthens kernels)"
    # a traced request that is not the timed ones: two positions of four, its own stamps under the profiler; two timed requests that differ
    first, second = _stepped([1.0, 1.0, 50.0, 50.0], [9.0, 9.0, 500.0, 500.0]), _stepped([2.0, 2.0, 70.0, 70.0], [12.0, 12.0, 900.0, 900.0])
    under = _stepped([1.5, 1.5], [11.55, 11.55], "traced")
    stats, overhead, lines = run([first, second], [4.0, 6.0], 2, under, calls=11)
    assert (overhead["n"], overhead["minimum"], overhead["maximum"]) == (4, pytest.approx(3.0), pytest.approx(8.0)), "9 - 4, 9 - 6, 12 - 4, 12 - 6"
    assert overhead["value"] == pytest.approx(5.5) and overhead["host_time_per_block_call_us"] == pytest.approx(500.0)
    assert terms(stats) == pytest.approx((12.0, 5.0, 5.5, 1.5)), "the means of the four steps at the two traced positions"
    assert stats["step"]["maximum"] == pytest.approx(14.0), "the steps behind the traced positions are in no term"
    assert stats["tracing_stretch"]["value"] == pytest.approx(10.0), "the traced request's own stamps: in the stretch (11.55 over 10.5 ms) ..."
    other = run([first, second], [4.0, 6.0], 2, _stepped([30.0, 30.0], [40.0, 80.0], "traced"))[0]
    assert terms(other) == terms(stats), "... and in no term of the equation: its GPU busy is, position by position"
    moved = run([_stepped([1.0, 1.0], [9.0, 20.0]), _stepped([1.0, 1.0], [9.0, 20.0])], [4.0, 15.0], 2)[1]
    apart = run([_stepped([1.0, 1.0], [9.0, 20.0]), _stepped([1.0, 1.0], [9.0, 20.0])], [15.0, 4.0], 2)[1]
    assert (moved["minimum"], moved["maximum"]) == (pytest.approx(5.0), pytest.approx(5.0)), "GPU busy against the step at its own position: 9 - 4 and 20 - 15"
    assert (apart["minimum"], apart["maximum"]) == (pytest.approx(-6.0), pytest.approx(16.0)) and apart["remainder_ms"] == pytest.approx(5.0)
    # the edges: one step; steps that are all alike; a bound that passes the step; lists that cannot be set against each other
    stats, overhead, lines = run([_stepped([1.0], [9.0])], [2.0], 1, _stepped([1.0], [9.0], "traced"))
    assert (overhead["n"], overhead["value"]) == (1, pytest.approx(7.0)) and terms(stats) == pytest.approx((10.0, 2.0, 7.0, 1.0))
    assert lines[0] == "step                  10.000  ms; mean 10.000 = GPU busy 2.000 + launch overhead 7.000 + host 1.000"
    assert lines[2:] == [
        "                      (GPU busy from the traced step, position 0; median 2.000 ms;",
        "                      tracing lengthens a step by 0.0 %)",
    ]
    stats, overhead, lines = run([_stepped([1.0] * 3, [9.0] * 3)] * 2, [2.0] * 3, 3)
    assert overhead["value"] == pytest.approx(7.0) == pytest.approx(of_medians(stats)), "steps that are alike: the medians are one step's"
    assert overhead["n"] == 6 and overhead["minimum"] == pytest.approx(overhead["maximum"])
    stats, overhead, lines = run([_stepped([1.0] * 3, [9.0] * 3)], [11.0, 12.0, 13.0], 3)
    assert overhead["value"] is None and overhead["remainder_ms"] == pytest.approx(-3.0) and overhead["maximum"] == pytest.approx(-2.0), "stored as computed"
    assert terms(stats) == pytest.approx((10.0, 12.0, -3.0, 1.0)), "the terms add up where the remainder is negative too"
    assert lines[0] == "step                  10.000  ms (not traced), mean 10.000" and not any("estimate" in line for line in lines)
    unequal = ([[10.0, 10.0]], [[1.0, 1.0]], [2.0]), ([[10.0]], [[1.0]], [2.0, 2.0]), ([[10.0], [10.0, 10.0]], [[1.0], [1.0, 1.0]], [2.0])
    for steps, hosts, busy in (*unequal, ([[10.0], [10.0]], [[1.0]], [2.0])):  # the last: the host times of one request for the steps of two
        with pytest.raises(ValueError, match="cannot be paired with the steps of the timed requests at the traced positions"):
            perf.step_split(steps, hosts, busy)
    with pytest.raises(ValueError, match="GPU busy of 3 traced steps cannot be paired"):
        run([_stepped([1.0] * 4, [9.0] * 4)], [2.0] * 3, 2)  # GPU busy of three steps for two traced positions
    with pytest.raises(ValueError, match="GPU busy of 3 traced steps cannot be paired"):
        run([_stepped([1.0], [9.0])], [2.0] * 3, 3)  # a timed request that does not reach the traced positions
    _defines("launch_overhead", "Per timed step at a traced position: step - GPU busy of the traced step at the same position - host,")
    _defines("launch_overhead", "formed step by step before anything is aggregated; of a run the mean over those steps.")
    _defines("launch_overhead", "With the means of the step, of GPU busy and of host over the same steps, which this file holds, it adds up:")
    _defines("launch_overhead", "it adds up: step = GPU busy + launch overhead + host.")
    _defines("launch_overhead", "it is an estimate, not a measurement: a timed step has no GPU-side record, and GPU busy is the traced request's,")
    _defines("launch_overhead", "taken under the profiler, and is an upper bound.")
    _says("area", "launch overhead = step - GPU busy - host, formed for every timed step with GPU busy of the traced step at the same position,")
    _says("area", "before anything is aggregated, and of a run the mean over those steps.")
    _says("area", "The line gives the step's mean beside its median and takes the mean apart, `mean = GPU busy + launch overhead + host`:")
    _says("area", "every term is the mean over the same steps, so the terms add up, which the medians of three lists do not.")
    _says("area", "Launch overhead is an estimate, and the line under the equation says so:")
    _says("area", "a timed step has no GPU-side record, and GPU busy is the traced request's and an upper bound.")
    _says("guide", "A figure that sets one against the other -- launch overhead, a step less its GPU busy and its host time -- is formed step by step,")
    _says("guide", "at the same position, before anything is aggregated: the medians of three lists need not be one step's,")
    _says("guide", "and their difference can be a time no step has.")
    _says("guide", "Where its terms are printed as a sum they are the means over the same steps, so that a reader can add them up,")
    _says("guide", "and it is called an estimate: the two sides are two requests.")


def test_rows_by_wiring_position_not_by_name(monkeypatch):
    """A launch record gets its row from its place in the wiring, as the kernel check finds it, never from its kernel's
    name: two blocks that launch the same kernel are two rows, and a library kernel inside a fallback chain is that
    chain's row wherever else its name is launched.  On the family's own plans the exact rows hold the planned launches,
    and the row of a block a library serves is named by the library's entry and holds the launches of its kernel."""
    from model_samples import census, perf

    sites = [_site("first", 0, "gate_quant"), _site("chain", 0, "pytorch", "cublas"), _site("second", 0, "gate_quant"), _site("tail", -1, "cublas")]
    traced, busy = _walked(sites, [_records(sites), _records(sites, start=1.0)])
    assert traced.status == "PASS" and [row.blocks for row in traced.result.table] == [("first",), ("second",), ("chain",), ("tail",), ()]
    assert [row.relation for row in traced.result.table] == ["exact", "exact", "fallback", "fallback", "read"]
    assert (
        traced.result.attributed == [[0, 2, 2, 2, 1, 3, 3, 4]] * 2
    ), "the gate kernel of `second` is not a launch of `first`; the library kernel of `tail` not of `chain`"
    names = [record.name for record in traced.steps[0]]
    assert names[0] == names[4] and names[2] == names[5] == NVJET, "one kernel name in two rows, twice"
    rows = perf.kernel_table(traced, busy)["rows"]
    assert [row["label"] for row in rows] == ["first", "second", "fallback: chain", "fallback: tail", "host read"]
    assert [(row["launches_per_step"], row["kernels_per_step"], row["copies_per_step"]) for row in rows] == [
        (1, 1, 0),
        (1, 1, 0),
        (3, 2, 1),
        (2, 1, 1),
        (1, 0, 1),
    ]
    assert [row["planned_launches_per_step"] for row in rows] == [1, 1, None, None, None] and rows[2]["blocks"][0]["entry"] == "entry of chain"
    assert rows[0]["launch_shapes"] == [{"symbol": names[0], "grid": (1, 1, 1), "block": (128, 1, 1), "shared_memory": 0, "launches": 1}]
    assert rows[2]["launch_shapes"] is None and sum(row["launches_per_step"] for row in rows) == len(traced.steps[0]), "the rows hold every record of a step"
    twice = [
        _site("first", 0, "gate_quant"),
        census.Site("pair", 0, (census.Kernel("swiglu_fp4", "cuDNN FE", "exact", launches=2),)),
        _site("pair", 1, "swiglu_fp4"),
    ]
    pair = perf.kernel_table(*_walked(twice, [_records(twice)]))["rows"][1]
    assert (pair["launches_per_step"], pair["planned_launches_per_step"], pair["blocks"][0]["calls_per_step"]) == (3, 3, 2), "calls x launches per call"
    # a block that launches its exact kernel and a chain of its own: the kernel is its exact row; the chain's records are the row of the gap
    # they stand in, which says what it holds -- no block of it is on a fallback entry -- and prints 0 where nothing was launched there
    mixed = [_site("x", 0, "gate_quant"), _site("both", 0, "swiglu_fp4", "pytorch"), _site("y", 0, "gate_quant")]
    held = perf.kernel_table(*_walked(mixed, [_records(mixed)]))["rows"]
    assert [(row["label"], row["relation"], row["launches_per_step"]) for row in held] == [
        ("x", "exact", 1),
        ("both", "exact", 1),
        ("y", "exact", 1),
        ("beside its kernel: both", "beside", 2),
        ("host read", "read", 1),
    ]
    alone = [record for record in _records(mixed) if record.name not in (ELEMENTWISE, COPY_IN)]  # the chain launched nothing in this step
    quiet = perf.kernel_table(*_walked(mixed, [alone]))
    assert [quiet["rows"][3][key] for key in ("label", "launches_per_step", "ms_per_step", "share_percent", "us_per_launch_added")] == [
        "beside its kernel: both",
        0,
        0.0,
        0.0,
        None,
    ]
    assert perf._kernel_lines(quiet, "")[6] == f"  {'beside its kernel: both':<58}{0:>9}{'--':>11}{0.0:>9.3f}{'0.0 %':>9}", "a row of the plan, printed"
    # what serves a row is decided from the census entries of ALL its block calls: one library entry names the row; a block whose calls two
    # entries serve is named by neither, and a kernel of cuDNN Frontend is no library's -- at another call of the block, or in the same call
    library = (census.Kernel("cublaslt_fp8_gemm", "FlashInfer", "exact", launches=1),)
    second = ((census.Kernel("fi_mm_fp4", "FlashInfer", "exact", launches=1),), "another.library.call")
    own = (census.Kernel("fp8_projection", "cuDNN FE", "exact", launches=1),)
    cases = [[(library, "a.library.call")] * 2, [(library, "a.library.call"), second], [(library, "a.library.call"), (own, "an entry")]]
    cases += [[(library + own, "a.library.call")] * 2]
    named = []
    for calls in cases:
        served = [census.Site("p", layer, kernels, entry=name) for layer, (kernels, name) in enumerate(calls)]
        row = perf.kernel_table(*_walked(served, [_records(served)]))["rows"][0]
        named.append((row["label"], row["relation"], row["launches_per_step"], row["planned_launches_per_step"]))
    assert named == [("a.library.call: p", "library", 2, 2), ("p", "exact", 2, 2), ("p", "exact", 2, 2), ("p", "exact", 4, 4)]
    # the family's plan: 64 layers on the stack native -- the exact rows in the wiring order, each with its planned launches; the row of a block
    # a library serves carries the library's entry as the plan names it, and holds the launches of the library's kernel
    sites, claim = _planned(monkeypatch)
    traced, busy = _walked(sites, [_records(sites)], claim)
    rows = perf.kernel_table(traced, busy)["rows"]
    exact = [(row["label"], row["relation"], row["launches_per_step"], row["planned_launches_per_step"]) for row in rows[:9]]
    assert exact == [
        ("embedding", "exact", 1, 1),
        ("add_norm_quantize", "exact", 129, 129),
        ("projection.fp8.in", "exact", 64, 64),
        ("linear_attention.step", "exact", 48, 48),
        ("flashinfer.gemm.bmm_fp8: projection.fp8.out", "library", 64, 64),
        ("mlp.gate_up", "exact", 64, 64),
        ("flashinfer.gemm.mm_fp4: mlp.down", "library", 64, 64),
        ("attention.gate_quantize", "exact", 16, 16),
        ("torch_ref.quantize + flashinfer.gemm.mm_fp4: output_head", "library", 1, 1),
    ]
    assert not [row["label"] for row in rows[9:] if row["relation"] in ("exact", "library")]
    three = "the target's own compiled kernels + cuDNN Frontend entries with byte-equal results + declared stand-ins"
    numbers = claim.numbers()
    assert sum(planned for *_, planned in exact) == traced.result.exact == numbers["target_code"] + numbers["target_port"] == 178 + 273 + 0, three
    assert sum(planned for _, kind, _, planned in exact if kind == "library") == 64 + 64 + 1, "one launch of the library per call of its three blocks"
    assert rows[1]["blocks"] == [{"block": "add_norm_quantize", "calls_per_step": 129, "entry": "cudnn.AddNormQuantDecode", "identity": "target"}]
    assert rows[4]["blocks"] == [{"block": "projection.fp8.out", "calls_per_step": 64, "entry": "flashinfer.gemm.bmm_fp8", "identity": "target"}]
    # the file says which kernel the library launched for its row, by the name the profiler gave it, and how often in a step
    assert [(shape["symbol"], shape["launches"]) for shape in rows[4]["launch_shapes"]] == [(NVJET_FP8, 64)]
    assert [(shape["symbol"], shape["launches"]) for shape in rows[6]["launch_shapes"]] == [(FI_MM_FP4, 64)] and rows[8]["launch_shapes"][0]["launches"] == 1
    printed = perf._kernel_lines(perf.kernel_table(traced, busy), "")
    assert printed[9] == f"  {'flashinfer.gemm.mm_fp4: mlp.down':<58}{64:>9}{'4.30':>11}{0.275:>9.3f}{'9.7 %':>9}", "one kernel per launch: its time per launch"
    # the stack fe-entries: the cuDNN Frontend rows of native and no library -- no row of that kind, and the exact rows it had before
    sites, claim = _planned(monkeypatch, "fe-entries")
    traced, busy = _walked(sites, [_records(sites)], claim)
    rows = perf.kernel_table(traced, busy)["rows"]
    assert not [row["label"] for row in rows if row["relation"] in ("library", "beside")]
    assert [(row["label"], row["launches_per_step"], row["planned_launches_per_step"]) for row in rows if row["relation"] == "exact"] == [
        ("embedding", 1, 1),
        ("add_norm_quantize", 129, 129),
        ("projection.fp8.in", 64, 64),
        ("linear_attention.step", 48, 48),
        ("mlp.gate_up", 64, 64),
        ("attention.gate_quantize", 16, 16),
    ]
    assert traced.result.exact == 49 + 273 + 0, three
    assert rows[-1]["label"] == "host read" and rows[-1]["launches_per_step"] == 1 and rows[-1]["blocks"] == []
    _defines("kernel_launches", "the kernel and copy records of a traced step that the kernel check attributes to the row's blocks; the median per step.")
    _corresponds("kernel_launches", "The kernel census of a traced serving step.")
    _says("area", "the device time of every block's kernels, from the launch records of the traced request.")
    _says(
        "area", "`launches`: the kernel and copy records of a step the kernel check attributes to the row.", "the host read that ends a step is the last row."
    )
    _says("area", "A row is a block, found by its place in the wiring, as the kernel check finds it, never by a kernel's name")
    _says("guide", "A row of the per-kernel table is found by its place in the wiring, never by a kernel's name,")
    _says("area", "the row of a block a library serves carries the entry in front, as the plan names it (`flashinfer.gemm.mm_fp4: mlp.down`),")
    _says("area", "and holds the launches of the library's kernel, found by their place in the wiring like every exact kernel;")
    _says("guide", "The row of a block a library serves names the library's entry, as the plan does, never the block alone and never a fallback;")


def test_adjacent_fallback_blocks_share_a_row(monkeypatch):
    """The records between two exact kernels belong to the blocks the wiring has there on fallback or host entries: such
    blocks are one row together and the row says so; the same blocks in another layer are the same row.  A row of a
    stack that declares a target kernel set says ``not yet`` where the census says so of its block, and what a block
    launches beside a kernel of its own is a row that says so: no block of it is on a fallback entry."""
    from model_samples import census, perf

    sites = [_site("x", 0, "gate_quant"), _site("f", 0, "pytorch"), _site("g", 0, "pytorch", "cublas"), _site("y", 0, "swiglu_fp4"), _site("f", 1, "pytorch")]
    sites += [
        _site("x", 1, "gate_quant"),
        _site("f", 2, "pytorch"),
        _site("g", 2, "pytorch", "cublas"),
        _site("y", 2, "swiglu_fp4"),
        _site("choice", -1, "host"),
    ]
    traced, busy = _walked(sites, [_records(sites)])
    table = traced.result.table
    assert [(row.blocks, row.relation, row.sites) for row in table] == [
        (("x",), "exact", (0, 5)),
        (("y",), "exact", (3, 8)),
        (("f", "g"), "fallback", (1, 2, 6, 7)),
        (("f",), "fallback", (4,)),
        (("choice",), "host", (9,)),
        ((), "read", ()),
    ]
    rows = perf.kernel_table(traced, busy)["rows"]
    assert [row["label"] for row in rows] == ["x", "y", "fallback: f + g (one row)", "fallback: f", "host path: choice", "host read"]
    assert [row["relation"] for row in rows] == ["exact", "exact", "fallback", "fallback", "host", "read"], "what the file calls each row"
    assert [block["identity"] for row in rows for block in row["blocks"]] == [None] * 6, "no census: no identity is stated for a block"
    assert rows[2]["launches_per_step"] == 2 * (2 + 3) and [block["calls_per_step"] for block in rows[2]["blocks"]] == [2, 2]
    assert rows[3]["launches_per_step"] == 2 and rows[4]["launches_per_step"] == 1 and rows[2]["us_per_launch_added"] == pytest.approx(4.3)
    lines = perf._kernel_lines(perf.kernel_table(traced, busy), "")
    assert lines[5] == f"  {'fallback: f + g (one row)':<58}{10:>9}{'--':>11}{0.043:>9.3f}{'55.6 %':>9}", "no time per launch for a chain of many kernels"
    assert (
        lines[7] == f"  {'host path: choice':<58}{1:>9}{'--':>11}{0.004:>9.3f}{'5.6 %':>9}" and lines[8].startswith("  host read   ") and " 4.30 " in lines[8]
    )
    _says("area", "the median over the row's launches; not printed for a row of fallback or host blocks nor for one of launches beside a kernel,")
    _says("area", "chains of many kernels.")
    # with the identities a census states for the calls: a declared stand-in says so, a block the census calls not yet says so
    relations = ["stand-in", "not yet", "not yet", "target", "not yet", "stand-in", "not yet", "not yet", "target", "host"]
    stated = census.Claim("a census", "plain", "a device", 1, identities=[census.Identity(relation, reason="a chain") for relation in relations])
    named = [row["label"] for row in perf.kernel_table(*_walked(sites, [_records(sites)], stated))["rows"]]
    assert named == ["x (declared stand-in)", "y", "not yet: f + g (one row)", "not yet: f", "host path: choice", "host read"]
    _says("area", "A row's name says what it is: a row of blocks on fallback entries is marked `not yet:` where the kernel census says so of one of")
    _says("area", "of one of its blocks and `fallback:` where no kernel identity is stated, a row of host-path blocks alone `host path:`,")
    _says("area", "and an exact row on a declared stand-in `(declared stand-in)`;")
    # a list of blocks wider than the label column is shortened; the file holds every block
    many = [_site("exact", 0, "gate_quant")] + [_site(f"a_block_with_a_long_name_{n}", 0, "pytorch") for n in range(4)]
    wide = perf.kernel_table(*_walked(many, [_records(many)]))["rows"][1]
    assert wide["label"] == "fallback: a_block_with_a_long_name_0 + 3 more (one row)" and len(wide["blocks"]) == 4 and len(wide["label"]) <= perf.LABEL
    _says("area", "a list of blocks wider than the column is shortened to its first block and `+ N more`.")
    # a gap that holds a block on a fallback entry and what another block launches beside its kernel is a fallback row: the mark is the chain's
    shared = [_site("x", 0, "gate_quant"), _site("f", 0, "pytorch"), _site("both", 0, "swiglu_fp4", "pytorch"), _site("y", 0, "gate_quant")]
    assert [(row["label"], row["relation"]) for row in perf.kernel_table(*_walked(shared, [_records(shared)]))["rows"]] == [
        ("x", "exact"),
        ("both", "exact"),
        ("y", "exact"),
        ("fallback: f + both (one row)", "fallback"),
        ("beside its kernel: both", "beside"),
        ("host read", "read"),
    ]
    # the family's plan: the attention preparation and the attention core of a layer have no exact kernel between them.  On the stack native
    # three blocks are a library's: what their calls launch beside the library's kernel -- a further launch of the library's choice behind the
    # out-projection, the reference quantiser in front of the output head's multiply -- is the row of its place in the wiring, and says so
    sites, claim = _planned(monkeypatch)
    table = perf.kernel_table(*_walked(sites, [_records(sites)], claim))
    rows = table["rows"]
    behind = [(row["label"], row["relation"], [block["calls_per_step"] for block in row["blocks"]]) for row in rows[9:]]
    assert behind == [
        ("beside its kernel: projection.fp8.out", "beside", [64]),
        ("not yet: attention.prepare + attention.core (one row)", "fallback", [16, 16]),
        ("beside its kernel: output_head", "beside", [1]),
        ("beside its kernel: output_head + token_choice (one row)", "beside", [1, 1]),
        ("host read", "read", []),
    ]
    assert not [label for label, *_ in behind if label.startswith("fallback: ")], "no block of this plan is without a stated identity"
    assert rows[10]["blocks"][0]["identity"].startswith("not yet (") and rows[12]["blocks"] == [
        {"block": "output_head", "calls_per_step": 1, "entry": "torch_ref.quantize + flashinfer.gemm.mm_fp4", "identity": "target"},
        {"block": "token_choice", "calls_per_step": 1, "entry": "torch.argmax", "identity": "host path, not compared"},
    ]
    printed = perf._kernel_lines(table, "")
    assert printed[14] == f"  {'beside its kernel: output_head':<58}{1:>9}{'--':>11}{0.004:>9.3f}{'0.2 %':>9}", "no time per launch: a chain"
    assert sum(row["launches_per_step"] for row in rows) == len(_records(sites)), "the rows hold every record of the step, the library's among them"
    # the stack fe-entries: those three blocks are on their fallback rows there, and the census says `not yet` of them
    sites, claim = _planned(monkeypatch, "fe-entries")
    rows = perf.kernel_table(*_walked(sites, [_records(sites)], claim))["rows"]
    behind = [(row["label"], [block["calls_per_step"] for block in row["blocks"]]) for row in rows if row["relation"] != "exact"]
    assert behind == [
        ("not yet: projection.fp8.out", [64]),
        ("not yet: mlp.down", [64]),
        ("not yet: attention.prepare + attention.core (one row)", [16, 16]),
        ("not yet: output_head + token_choice (one row)", [1, 1]),
        ("host read", []),
    ]
    assert rows[8]["blocks"][0]["identity"].startswith("not yet (") and rows[9]["blocks"][1] == {
        "block": "token_choice",
        "calls_per_step": 1,
        "entry": "torch.argmax",
        "identity": "host path, not compared",
    }
    _says("area", "what a block's call launches beside a kernel of its own -- the reference quantiser in front of the output head's multiply,")
    _says("area", "a further launch of a library's choice -- is the row of its place in the wiring, marked `beside its kernel:` and shared with")
    _says("area", "whatever else the wiring has there (the records do not tell them apart), and it prints 0 where nothing was launched;")
    _says("guide", "what a block launches beside a kernel of its own is the row of its place in the wiring and says so.")
    _says("area", "The report file holds every block of every row, with its entry and the identity the census states for it.")
    _says("area", "the blocks served by fallback or host entries that no exact kernel separates share one row, and the row says so (`one row`)")


def test_shares_sum_to_gpu_busy(monkeypatch, tmp_path):
    """Within one step the added times of all records are that step's GPU busy, so over the traced steps the rows' mean
    time per step sums to the mean GPU busy to the rounding of a double, and the medians the screen prints sum to GPU
    busy up to the difference between a median of sums and a sum of medians: nothing where the steps are alike."""
    from model_samples import perf

    sites = [_site("x", 0, "gate_quant"), _site("f", 0, "pytorch"), _site("y", 0, "swiglu_fp4"), _site("g", 0, "pytorch"), _site("z", 0, "gate_quant")]
    alike, busy = _walked(sites, [_records(sites, start=at) for at in (0.0, 1.0, 2.0)])
    rows = perf.kernel_table(alike, busy)["rows"]
    assert sum(row["share_percent"] for row in rows) == pytest.approx(100.0) and sum(row["ms_per_step"] for row in rows) == pytest.approx(busy[0])
    assert [row["share_percent"] for row in rows] == pytest.approx(
        [12.5, 12.5, 12.5, 25.0, 25.0, 12.5]
    ), "x, y, z; f and g with their copy; the read: eight records"
    # steps that differ: the first chain launches a kernel and a copy more in the first step, the second chain two kernels more in the last
    steps = [_records(sites, also=[(1, ELEMENTWISE), (1, COPY_IN)]), _records(sites, start=1.0), _records(sites, start=2.0, also=[(3, ELEMENTWISE)] * 2)]
    traced, busy = _walked(sites, steps)
    rows = perf.kernel_table(traced, busy)["rows"]
    for step, held, of_step in zip(traced.steps, traced.result.attributed, busy):
        per_row = [sum(added for added, row in zip(perf.added_ms(step), held) if row == number) for number in range(len(rows))]
        assert sum(per_row) == pytest.approx(of_step, rel=1e-12), "the rows of one step partition its GPU busy"
    assert sum(row["ms_per_step_mean"] for row in rows) == pytest.approx(sum(busy) / 3, rel=1e-12), "whatever the steps are"
    assert [row["launches_per_step"] for row in rows] == [1, 1, 1, 2, 2, 1] and rows[3]["ms_per_step"] == pytest.approx(
        2 * 0.0043
    ), "the median step of each row"
    assert sum(row["ms_per_step"] for row in rows) == pytest.approx(8 * 0.0043) and sorted(busy)[1] == pytest.approx(10 * 0.0043)
    assert sum(row["share_percent"] for row in rows) == pytest.approx(
        80.0
    ), "a sum of medians is not the median of the sums: the screen's shares need not reach 100 %"
    assert (rows[3]["kernels_per_step"], rows[3]["copies_per_step"]) == (1, 1), "of the median step, not of the first one (2 kernels, 2 copies)"
    assert all(row["share_percent"] == pytest.approx(row["ms_per_step"] / sorted(busy)[1] * 100) for row in rows), "a row's median over the MEDIAN GPU busy"
    shown = perf._kernel_lines(perf.kernel_table(traced, busy), "")
    assert shown[6] == f"  {'fallback: f':<58}{2:>9}{'--':>11}{0.009:>9.3f}{'20.0 %':>9}", "the median over the steps (0.0086 ms), not their mean (0.0115 ms)"
    # a run: the same tie-out on what the file holds
    data = _mode(monkeypatch, tmp_path)[0].data
    means, of_steps = [row["ms_per_step_mean"] for row in data["per_kernel"]["rows"]], data["traced"]["per_step"]["gpu_busy_ms"]
    assert sum(means) == pytest.approx(sum(of_steps) / len(of_steps), rel=1e-12) and data["traced"]["gpu_busy_ms"] == data["statistics"]["gpu_busy"]["value"]
    assert sum(row["launches_per_step"] for row in data["per_kernel"]["rows"]) == data["traced"]["records_of_the_sizing_step"]
    _says("area", "The added times of a step partition the union, so the rows sum to GPU busy")
    _defines("kernel_time_per_step", "per traced step the sum of the row's added times, the median over the steps; its share is that over the median GPU busy.")
    _defines("kernel_time_per_step", "The medians of the rows need not sum to the median GPU busy, nor their shares to the whole of it:")
    _defines("kernel_time_per_step", "the file also holds each row's mean, and the means sum to the mean GPU busy.")
    _corresponds("kernel_time_per_step", "The same shares as in a per-kernel analysis of a traced serving step.")
    _says("area", "`ms/step` and its share of GPU busy: the row's added times of a step, the median over the traced steps,")
    _says("area", "and that median over the median GPU busy, which the line under the `step` line gives.")
    _says("area", "(the medians, and with them the shares, up to the difference between a median of sums and a sum of medians;")
    _says("area", "the means in the file to the rounding of a double).")


def test_undeclared_record_prints_no_table(monkeypatch, tmp_path):
    """A table is made only of traced steps whose every record the kernel check attributed.  A kernel no block of the plan
    launches, a copy between two exact rows, a step whose exact launches are not the plan's: the check fails, the record
    has no row, and the screen says why there is no table.  GPU busy, a union that needs no attribution, is still
    measured.  Kernels that were not observed: no table either, with the reason."""
    from model_samples import perf

    sites = [_site("x", 0, "gate_quant"), _site("y", 0, "swiglu_fp4"), _site("f", 0, "pytorch")]
    for index, extra in ((0, "a_kernel_no_pattern_names"), (0, COPY_IN), (0, ELEMENTWISE), (2, SYMBOL)):
        traced, busy = _walked(sites, [_records(sites), _records(sites, start=1.0, also=[(index, extra)])])
        assert traced.status == "FAIL" and -1 in traced.result.attributed[1] and -1 not in traced.result.attributed[0], extra
        table = perf.kernel_table(traced, busy)
        assert table["rows"] is None and table["reason"] == "the kernels of the traced steps are not the plan's", extra
        assert perf._kernel_lines(table, "") == ["per kernel, per step  not printed: the kernels of the traced steps are not the plan's"]
    clean, busy = _walked(sites, [_records(sites)])
    assert perf.kernel_table(clean, busy)["rows"] and perf.kernel_table(clean, [])["rows"] is None and perf.kernel_table(None, busy)["rows"] is None
    # through a run: traced steps that launch another kernel than the plan's -- no table, GPU busy all the same, and no rate
    defect = {"at the first profiler pass": {"symbol": OTHER_SYMBOL}}
    (tmp_path / "kernel").mkdir()
    outcome, lines, _, _ = _mode(monkeypatch, tmp_path / "kernel", defect=defect)
    assert not outcome.passed and outcome.data["per_kernel"]["rows"] is None and outcome.data["statistics"]["gpu_busy"]["value"] > 0
    assert "per kernel, per step  not printed: the kernels of the traced steps are not the plan's" in lines
    assert not any(line.startswith("per kernel, per step (") for line in lines)
    (tmp_path / "blind").mkdir()
    outcome, lines, _, _ = _mode(monkeypatch, tmp_path / "blind", profiler=NO_PROFILER)
    assert outcome.data["per_kernel"] == {
        **outcome.data["per_kernel"],
        "rows": None,
        "reason": f"the kernels of the traced steps were not observed ({NO_PROFILER})",
    }
    at = lines.index("per kernel, per step  not printed: the kernels of the traced steps were not observed (this PyTorch build")
    assert lines[at + 1] == "                      records no kernel activity)" and outcome.data["traced"]["per_step"]["gpu_busy_ms"] == []
    _says("area", "A table is printed only where the kernel check holds: a record that no block of the plan accounts for prints no table")
    _says("guide", "and a traced pass whose kernel check does not hold prints no table.")


def test_whole_microsecond_stamps_print_no_us_per_launch(monkeypatch, tmp_path):
    """The time one launch adds needs stamps finer than a microsecond.  The resolution is read from the durations the
    trace holds -- a launch record keeps its duration as the trace has it, which its start and end on the profiler's
    clock no longer show -- and where every one is a whole number of microseconds the ``us/launch`` column is not
    printed and the table says so; the time per step and the shares stay."""
    from model_samples import census, perf

    def lasting(*durations):
        return [[census.Launch(ELEMENTWISE, "kernel", float(index), index + us / 1e3, duration=us) for index, us in enumerate(durations)]]

    assert perf.time_resolution_us(lasting(5.0, 12.0, 300.0)) == 1.0 and perf.time_resolution_us(lasting(4.375, 12.0)) == pytest.approx(0.001)
    assert perf.time_resolution_us(lasting(5.6, 2.25)) == pytest.approx(0.01) and perf.time_resolution_us(lasting(0.5)) == pytest.approx(0.1)
    assert perf.time_resolution_us([]) is None and perf.time_resolution_us([[]]) is None
    bare = [[census.Launch(ELEMENTWISE, "kernel", 3.0, 3.0105), census.Launch(NVJET, "kernel", 4.0, 4.25)]]  # no duration: end - start, to the nanosecond
    assert perf.time_resolution_us(bare) == pytest.approx(0.1) and perf._recorded_us(bare[0][0]) == 10.5
    # the reader keeps the duration the trace holds; the stamps of a long-running clock have lost its last decimals
    trace = {"traceEvents": [{"ph": "X", "cat": "kernel", "name": NVJET, "ts": 3000000000000.999, "dur": 4.375, "args": {}}]}
    trace["traceEvents"].append({"ph": "X", "cat": "gpu_memcpy", "name": HOST_READ, "ts": 3000000000011, "dur": 7, "args": {"bytes": 4}})
    (tmp_path / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
    kernel, read = census.read_trace(str(tmp_path / "trace.json"))
    assert (kernel.duration, read.duration) == (4.375, 7.0) and isinstance(read.duration, float)
    assert (kernel.end - kernel.start) * 1e3 != 4.375 and (kernel.end - kernel.start) * 1e3 == pytest.approx(4.375, abs=2e-3)
    assert perf.time_resolution_us([[kernel, read]]) == pytest.approx(0.001) and perf.time_resolution_us([[read]]) == 1.0
    # the table of whole-microsecond stamps: no us/launch column, and the line that says why; of finer ones: the column
    sites = [_site("x", 0, "gate_quant"), _site("f", 0, "pytorch")]
    whole = perf.kernel_table(*_walked(sites, [_records(sites, us=7.0)]))
    assert whole["us_per_launch_printed"] is False and whole["rows"][0]["us_per_launch_added"] == pytest.approx(7.0), "computed and in the file"
    assert perf._kernel_lines(whole, "") == [
        f"{'per kernel, per step (traced request)':<60} launches  ms/step  share of GPU busy",
        "  launched from Python: not the time of a kernel inside a captured sequence",
        "  us/launch not printed: the profiler's time stamps are whole microseconds here",
        f"  {'x':<58}{1:>9}{0.007:>9.3f}{'25.0 %':>9}",
        f"  {'fallback: f':<58}{2:>9}{0.014:>9.3f}{'50.0 %':>9}",
        f"  {'host read':<58}{1:>9}{0.007:>9.3f}{'25.0 %':>9}",
    ]
    later = perf.kernel_table(*_walked(sites, [_records(sites, us=7.0), _records(sites, start=1.0)]))
    assert later["us_per_launch_printed"] is True, "the finest stamp of any traced step, not of the first one"
    fine = perf.kernel_table(*_walked(sites, [_records(sites)]))
    assert fine["us_per_launch_printed"] is True and perf._kernel_lines(fine, "reduced model: ") == [
        f"{'per kernel, per step (reduced model: traced request)':<60} launches  us/launch  ms/step  share of GPU busy",
        "  launched from Python: not the time of a kernel inside a captured sequence",
        "  us/launch = the time a launch adds to the GPU's timeline",
        f"  {'x':<58}{1:>9}{'4.30':>11}{0.004:>9.3f}{'25.0 %':>9}",
        f"  {'fallback: f':<58}{2:>9}{'--':>11}{0.009:>9.3f}{'50.0 %':>9}",
        f"  {'host read':<58}{1:>9}{'4.30':>11}{0.004:>9.3f}{'25.0 %':>9}",
    ]
    # a run: the stand-in profiler's stamps are whole microseconds, and the file says which resolution it found
    for name in ("whole", "fine", "mixed"):
        (tmp_path / name).mkdir()
    outcome, lines, _, _ = _mode(monkeypatch, tmp_path / "whole")
    assert outcome.data["traced"]["time_resolution_us"] == 1.0 and outcome.data["per_kernel"]["us_per_launch_printed"] is False
    assert "  us/launch not printed: the profiler's time stamps are whole microseconds here" in lines and not any(
        "us/launch  ms/step" in line for line in lines
    )
    outcome, lines, _, _ = _mode(monkeypatch, tmp_path / "fine", on_say=_fine_stamps(monkeypatch))
    assert outcome.data["traced"]["time_resolution_us"] == pytest.approx(0.0001) and outcome.data["per_kernel"]["us_per_launch_printed"] is True
    assert "  us/launch = the time a launch adds to the GPU's timeline" in lines and any(
        line.endswith("launches  us/launch  ms/step  share of GPU busy") for line in lines
    )
    # the first traced step on whole microseconds (the stand-in's records of half a millisecond), the five behind it on finer stamps
    outcome, lines, _, _ = _mode(monkeypatch, tmp_path / "mixed", on_say=_fine_stamps(monkeypatch, behind=1))
    traced, busy = outcome.data["traced"], outcome.data["traced"]["per_step"]["gpu_busy_ms"]
    assert traced["time_resolution_us"] == pytest.approx(0.0001) and outcome.data["per_kernel"]["us_per_launch_printed"] is True, "of every traced step"
    assert busy[0] == pytest.approx(39 * 0.5) and busy[1:] == pytest.approx([39 * 0.0004375] * 5), "per step, in the order of the positions"
    assert traced["gpu_busy_ms"] == outcome.data["statistics"]["gpu_busy"]["value"] == pytest.approx(39 * 0.0004375), "the median step, not the mean"
    _defines("kernel_time_per_launch", "It needs time stamps finer than a microsecond: the file records the resolution found, and where the profiler hands")
    _says("area", "where the profiler hands over whole microseconds the `us/launch` column is not printed and the table says so")


def test_table_of_a_python_step_carries_its_line(monkeypatch, tmp_path):
    """The per-kernel table of a run: behind the ``memory`` line (and the ``note``), in front of ``report file``; under its
    head line the sentence that a kernel launched from Python does not show its time inside a captured sequence; then
    the exact rows in the wiring order, the rows of the blocks between them, the host read.  With stamps finer than a
    microsecond and a GPU busy below the step, the ``step`` line is the equation and the ``note`` line is printed."""
    from model_samples import perf

    outcome, lines, _, _ = _mode(monkeypatch, tmp_path, on_say=_fine_stamps(monkeypatch))
    data = outcome.data
    assert outcome.passed and data["per_kernel"]["launch_path"] == data["protocol"]["launch_path"] == "python" and data["per_kernel"]["reason"] is None
    head = lines.index(f"{'per kernel, per step (reduced model: traced request)':<60} launches  us/launch  ms/step  share of GPU busy")
    assert (
        lines[head + 1]
        == "  launched from Python: not the time of a kernel inside a captured sequence"
        == "  " + perf.FROM_PYTHON
        == "  " + data["per_kernel"]["launch_path_note"]
    )
    labels = [row["label"] for row in data["per_kernel"]["rows"]]
    assert labels == ["embedding", "linear_attention.step", "fallback: add_norm_quantize + projection.fp8.in (one row)", labels[3], "host read"]
    assert labels[3] == "fallback: projection.fp8.out + 9 more (one row)" and len(data["per_kernel"]["rows"][3]["blocks"]) == 10
    assert [line[2:60].rstrip() for line in lines[head + 3 : head + 8]] == labels and lines[head + 8] == "" and lines[head - 1] == ""
    assert lines[head + 9].startswith("report file           ") and lines[head - 2].startswith(("note   ", " " * 22))
    exact = [row for row in data["per_kernel"]["rows"] if row["relation"] == "exact"]
    assert [(row["launches_per_step"], row["planned_launches_per_step"]) for row in exact] == [(1, 1), (1, 1)], "as the kernel check counted them"
    assert all(row["us_per_launch_added"] == pytest.approx(0.4375) and row["us_per_launch_recorded"] == 0.4375 for row in data["per_kernel"]["rows"])
    step, stats = lines.index(_line(lines, "step  ")), data["statistics"]
    said = " ".join(line.strip() for line in lines[step : step + 5])
    assert " ms; mean " in said and " = GPU busy 0.017 + launch overhead " in said, "the mean of the step taken apart"
    assert f" launch overhead: {perf.ESTIMATE} (GPU busy from 6 traced steps, positions 5 to 10; median 0.017 ms; tracing lengthens a step by " in said
    overhead, calls = stats["launch_overhead"], len(_tiny_model().wiring())
    assert overhead["value"] == overhead["remainder_ms"] == overhead["mean"] > 0.017 and overhead["label"] == perf.ESTIMATE
    assert overhead["n"] == stats["step"]["n"] == stats["host"]["n"] == 2 * 6, "every timed step at a traced position: two requests of six"
    assert stats["step"]["mean"] == pytest.approx(
        stats["gpu_busy"]["mean"] + overhead["remainder_ms"] + stats["host"]["mean"], rel=1e-9
    ), "a run's terms add up"
    printed = [float(number) for number in re.findall(r"(?:mean|GPU busy|launch overhead|host) (\d+\.\d{3})", said.split(" launch overhead: ")[0])]
    assert len(printed) == 4 and printed[0] == pytest.approx(sum(printed[1:]), abs=0.002), "and so do the four numbers the line prints, to their rounding"
    assert overhead["host_time_per_block_call_us"] == pytest.approx(overhead["remainder_ms"] * 1e3 / calls), "per block call of a step: the model's wiring"
    note = lines.index(_line(lines, "note   "))
    noted = _flat(" ".join(lines[note : note + 2]))
    assert noted.startswith(f"note reduced model: launch overhead (an estimate) {overhead['value'] / stats['step']['mean'] * 100:.0f} % of the mean step = ")
    assert f" % of the mean step = {overhead['host_time_per_block_call_us']:.0f} us per block call: " in noted, "the figure of the note line"
    assert (
        data["statistics"]["gpu_busy"]["value"] == pytest.approx(39 * 0.0004375)
        and data["traced"]["tracing_stretch_percent"] == data["statistics"]["tracing_stretch"]["value"]
    )
    assert data["statistics"]["gpu_busy"]["n"] == data["traced"]["steps"] == len(data["traced"]["per_step"]["gpu_busy_ms"]) == 6, "every traced step"
    _says("area", "It is taken apart with the launch records of the traced request: GPU busy = the union of the intervals")
    assert list(data["traced"])[-5:] == ["gpu_busy_ms", "gpu_busy_reason", "per_step", "tracing_stretch_percent", "time_resolution_us"]
    assert list(data["traced"]["per_step"]) == ["gpu_busy_ms", "idle_ms", "kernels", "copies"] and data["traced"]["per_step"]["copies"] == [1] * 6
    _says("area", "GPU busy, idle time, kernels and copies of every traced step; the resolution of the profiler's time stamps as found),")
    assert list(data["per_kernel"]) == ["launch_path", "launch_path_note", "us_per_launch_printed", "reason", "rows"]
    fields = "label relation blocks launches_per_step planned_launches_per_step kernels_per_step copies_per_step us_per_launch_added us_per_launch_recorded"
    assert all(list(row) == [*fields.split(), "ms_per_step", "ms_per_step_mean", "share_percent", "launch_shapes"] for row in data["per_kernel"]["rows"])
    _says("area", "`per_kernel` (every row of the table: its blocks, its launches against the planned ones, the added and the recorded time of a launch,")
    _says("area", "its time per step as median and as mean, its share, the launch shapes of an exact row and of a library's row, there with the")
    _says("area", "name of the kernel the library launched),")
    # what the GPU cell of the area holds of the table, and the tests of this module that the README's row names
    cells = (MODELS.parents[1] / "test" / "python" / "linear_attention" / "frost" / "test_model_sample_qwen_dense.py").read_text(encoding="utf-8")
    cell = cells.split("def test_the_performance_mode_of_the_seeded_run_reports_and_writes_its_file(", 1)[1].split("@pytest.mark.L1", 1)[0]
    kept = ('"the rows hold every record of a step"', 'row["launches_per_step"] == row["planned_launches_per_step"]', '"the rows partition GPU busy"')
    for held in (*kept, '"the terms of the step add up"', '"every timed step at a traced position"'):
        assert held in cell, held
    _says("area", "and in it the per-kernel table of real launches -- the rows hold every record of a step, an exact row its planned launches,")
    _says("area", "and the rows' mean time per step sums to the mean GPU busy")
    _says("area", "the GPU side on constructed launch records (GPU busy as the union of a step's intervals with an overlapped stretch counted once,")
    _says("area", "the step taken apart and the rule for printing launch overhead as a term, tracing's stretch, the rows of the per-kernel table by the")
    _says("area", "place in the wiring on stated sites and on the family's own plans of `native` and `fe-entries`, the rows of the blocks a library")
    _says("area", "serves and of what a block launches beside its kernel, blocks on fallback or host entries that share a row,")
    _says(
        "area", "the rows against GPU busy, a record without a row, whole-microsecond stamps, the launch records of `--dump-census` with their start and end);"
    )
    named = "added_time_partitions_the_union step_equation_only_where_the_remainder_exceeds_gpu_busy rows_by_wiring_position_not_by_name"
    named += " adjacent_fallback_blocks_share_a_row shares_sum_to_gpu_busy undeclared_record_prints_no_table whole_microsecond_stamps_print_no_us_per_launch"
    assert all(callable(globals()[f"test_{name}"]) for name in [*named.split(), "table_of_a_python_step_carries_its_line", "dump_census_schema_2"])
    assert callable(test_the_remainder_is_formed_step_by_step_before_it_is_aggregated), "the step taken apart: its second test"
    assert perf.screen(json.loads(json.dumps(data))) == lines[lines.index("cuDNN Frontend model sample: decode performance report") :][: len(perf.screen(data))]
    _says("area", "The table says of itself what it is: its second line says that it is launched from Python:")
    _says("area", "a kernel's time there is not its time inside a captured sequence.")
    _says("area", "An ordinary run prints no time per block; the performance mode prints the device time of every block's kernels.")
    _says("area", "Neither prints a limit (floor, speed-of-light figure)")
    _says("guide", "The token-to-token interval and the rate come from the host clock of requests that are not traced; GPU busy and the device time of a")
    _says("guide", "kernel come from kernel activity records of a traced request at the same positions, behind the last timed request.")
    _says("guide", "The two are never taken from one pass, and a report says which is which.")


def test_dump_census_schema_2(monkeypatch, capsys, tmp_path):
    """``--dump-census`` writes every launch record with its start and its end, in milliseconds on the profiler's clock, and
    says ``schema 2``: of an ordinary run and of the traced request of the performance mode."""
    sites = [_site("x", 0, "gate_quant"), _site("f", 0, "pytorch")]
    traced, _ = _walked(sites, [_records(sites), _records(sites, start=1.0)])
    dumped = traced.dump()
    assert traced.status == "PASS" and traced.result.attributed[0] == traced.result.attributed[1], "a second later: the same check, the same rows"
    _says("family", "Registers per thread, and the start and the end of every launch, are recorded in `--dump-census` and compared with nothing.")
    _says("family", "Nothing the check says is a timing.")
    assert dumped["fields"] == [
        "name",
        "kernel",
        "kind",
        "grid",
        "block",
        "shared_memory",
        "registers",
        "graph_id",
        "graph_node_id",
        "bytes",
        "start_ms",
        "end_ms",
    ]
    assert [record[-2:] for record in dumped["steps"][1]] == [[launch.start, launch.end] for launch in traced.steps[1]] and dumped["steps"][1][0][-2] == 1.0
    sample = _sample_on_the_cpu(monkeypatch, [])
    monkeypatch.chdir(tmp_path)
    for arguments, steps in (
        (["--sequences", "1", "--force-tokens", "4"], 32),
        (["--perf", "--perf-steps", "6", "--perf-requests", "1", "--perf-warmup", "1"], 6),
    ):
        code = sample.main([*arguments, "--dump-census", "census.json"])
        assert code == 0 and capsys.readouterr().err.splitlines()[-1] == "[00] PASS"
        written = json.loads((tmp_path / "census.json").read_text(encoding="utf-8"))
        (tmp_path / "census.json").unlink()
        assert (written["schema"], written["traced_steps"]) == (2, steps) and written["runs"][0]["fields"][-2:] == ["start_ms", "end_ms"]
        records = [record for step in written["runs"][0]["steps"] for record in step]
        assert records and all(len(record) == 12 and record[-2] <= record[-1] for record in records)
        assert all([record[-2] for record in step] == sorted(record[-2] for record in step) for step in written["runs"][0]["steps"]), "in start order"
    _says(
        "family",
        "per launch its kernel id, name, grid, block, shared memory, registers per thread, and its start and end in milliseconds on the profiler's clock",
    )
    _says("script", "(kernel, grid, block, shared memory, start and end per launch)")


# ---------------------------------------------------------------------------------------------- the flags and main()
def _sample_script(monkeypatch):
    """The sample script as a module, for its own functions; what loading it changes in the process leaves with the test."""
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script puts samples/models and its own folder on the path
    monkeypatch.setenv("KINETO_LOG_LEVEL", os.environ.get("KINETO_LOG_LEVEL", "6"))  # and sets a default for the profiler's log level
    spec = importlib.util.spec_from_file_location("model_sample_qwen_dense_00_decode_perf", SCRIPT)
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    return sample


def _sample_on_the_cpu(monkeypatch, events, loads=True, **stand_in):
    """The sample script as a module whose runs stay on the CPU: stated device facts (compute capability 10.0), the tiny
    geometry in the place of the reduced model, every session on the CPU and planned on the stand-ins of
    :func:`_stand_ins`, the environment check answered.  With ``loads`` false a load is an ``AssertionError``."""
    from model_samples import blocks
    from model_samples.qwen_dense import QwenDense

    model_samples = importlib.import_module("model_samples")  # the package as an object: its names are patched below
    build = _stand_ins(monkeypatch, events, **stand_in)
    stated = blocks.DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    real_session = model_samples.Session

    def session(model, text, **options):
        made = real_session(model, text, **dict(options, device="cpu"))
        plan_options = {"reference_precision": made.reference_precision, "device": "cpu", "check_kernels": made.check_kernels}
        table, regimes = model.provider_table, model.regimes
        made.plan = blocks.plan(table, regimes, stated, made.sequences, 1, made.stack, made.overrides, model=model, options=plan_options, factories=build)
        return made

    def no_load(self):
        raise AssertionError("reached a load")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (80 * GIB, 180 * GIB))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 0)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: stated))
    monkeypatch.setattr(model_samples, "check_environment", lambda: model_samples.Environment("1.34.0", "1.34.0", True, "0123456789ab", "2.13.0", "4.8.0"))
    if not loads:
        monkeypatch.setattr(real_session, "load", no_load)
    monkeypatch.setattr(model_samples, "Session", session)
    monkeypatch.setattr(QwenDense, "reduced", classmethod(lambda cls, layers=None: _tiny_model()))
    return _sample_script(monkeypatch)


def test_perf_flags_refused_before_load(monkeypatch, capsys, tmp_path):
    """Every refusal of the mode comes before the load: a flag of the mode without ``--perf``; ``--perf`` with the reference
    stack, with a limit of new tokens or a forced count of its own, with a list of sequence counts, with two prompts; a
    count that is not positive; a report file that exists, has an empty name or cannot be written; a request that does
    not fit the context.  With ``--perf`` alone the defaults are the protocol's, at one sequence count."""
    sample = _sample_on_the_cpu(monkeypatch, [], loads=False)

    def usage(arguments):
        with pytest.raises(SystemExit) as stop:
            sample.main(arguments)
        assert stop.value.code == 2
        return capsys.readouterr().err.splitlines()[-1].split(": error: ", 1)[1]

    for flag, value in (("--perf-warmup", "4"), ("--perf-steps", "8"), ("--perf-requests", "2"), ("--perf-trace-records", "9"), ("--perf-report", "r.json")):
        assert usage([flag, value]) == f"{flag}: a flag of the performance mode; it needs --perf"
    assert usage(["--perf-steps", "8", "--perf-report", "r.json"]) == "--perf-steps, --perf-report: a flag of the performance mode; it needs --perf"
    assert (
        usage(["--perf", "--stack", "reference"])
        == "--perf times a provider stack and holds it against the reference; --stack reference has no provider to time"
    )
    fixed = "--perf decodes a fixed count: the first new token and --perf-steps forced steps behind it, the end token ignored; "
    assert usage(["--perf", "--force-tokens", "8"]) == fixed + "--force-tokens does not apply beside it"
    assert usage(["--perf", "--model", "x", "--max-tokens", "8"]) == fixed + "--max-tokens does not apply beside it"
    assert usage(["--perf", "--model", "x", "--max-new-tokens", "8"]) == fixed + "--max-new-tokens does not apply beside it"
    assert usage(["--perf", "--sequences", "1,4"]) == "--perf times one sequence count; --sequences was given a list"
    assert usage(["--perf", "--model", "x", "--prompt", "a", "--prompt", "b"]) == "--perf times one prompt; 2 were given (--prompt, --messages)"
    counts = "--perf-warmup, --perf-requests and --perf-trace-records take positive integers, --perf-steps an integer from 2"
    for flag, value in (("--perf-warmup", "0"), ("--perf-steps", "1"), ("--perf-requests", "0"), ("--perf-trace-records", "0")):
        assert usage(["--perf", flag, value]) == counts

    def error(arguments):
        capsys.readouterr()
        code = sample.main(arguments)
        assert code == 1
        return capsys.readouterr().err.splitlines()[-1]

    (tmp_path / "there.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert error(["--perf", "--perf-report", "there.json"]) == "[00] ERROR --perf-report there.json: the file exists, and a report file is never written over"
    assert (tmp_path / "there.json").read_text() == "{}"
    assert error(["--perf", "--perf-report", ""]) == "[00] ERROR --perf-report: needs a file name; it was given an empty one"
    assert error(["--perf", "--perf-report", str(tmp_path / "missing" / "r.json")]) == "[00] ERROR --perf-report r.json: cannot be written there"
    for flag in ("--dump-tokens", "--dump-census"):  # written behind the report: the same file would be written over
        same = error(["--perf", "--perf-report", "same.json", flag, os.path.join(".", "same.json")])
        assert same == f"[00] ERROR --perf-report and {flag} name one file, same.json: a report file is never written over"
    assert not (tmp_path / "same.json").exists(), "nothing was written (--text-out is held by the same lines; the flag needs a checkpoint)"
    # a request beyond the context, in the mode's own words: 26 prompt tokens, the first new token and 256 timed steps against 40 positions
    # (the default warm-up of 32 steps does not fit the 13 that are left either: test_the_context_is_held_against_the_longest_request has the rule)
    late = error(["--perf", "--max-model-len", "40"])
    words = "the seeded prompt: 26 prompt tokens, the first new token and --perf-steps 256 (a timed request, the longest request of this run) are 283 "
    assert late == f"[00] ERROR {words}positions and --max-model-len is 40; take --perf-steps and --perf-warmup 13 or fewer, or raise --max-model-len"
    assert error(["--perf", "--max-model-len", "28"]).endswith(" of this run) are 283 positions and --max-model-len is 28; raise --max-model-len")
    body = sample.serving_options.Request([], None, {}, "--messages body.json", max_tokens=8, limit_field="max_tokens")
    fits = sample._arguments(["--perf", "--perf-steps", "8"])
    beside = '--messages body.json: its "max_tokens": 8 does not apply beside --perf, which decodes --perf-steps 8 forced steps behind the first new token'
    assert sample._perf_limit(fits, body, 26) == [beside] and sample._perf_limit(fits, sample.serving_options.Request([], None, {}, "a prompt"), 26) == []
    _says("family", "do not fit `--max-model-len` is an `ERROR` before the load, like every request beyond the context:")
    _says("family", "A file that exists, a file that `--dump-tokens`, `--dump-census` or `--text-out` names too, an empty name or a folder that cannot")
    _says("family", "be written is an `ERROR` before the load: a report file is never written over |")
    _says("area", "a file that exists, or that another output flag of the run names, is an `ERROR` before the load: a report file is never written over)")
    _says("script", "a file that exists, or that another flag of the run writes, is an error before the load")
    _says("family", "Not with `--stack reference`, `--max-tokens`, `--force-tokens`, a list of sequence counts or several prompts: each is refused before")
    _says("family", "Implies one sequence count and one prompt, the kernel check (as `--check-kernels`), forced steps with the end token ignored,")
    _says("family", "one token more than a serving request of 256 output tokens. At least 2 |")
    _says("area", "the decode performance report of this GPU: one process, one sequence count, one prompt.")
    _says("script", "Implies one sequence count, the kernel check, forced steps with the end token ignored, and the text written once, after the report;")
    _says("script", "not with --stack reference")
    # the defaults of the mode: the protocol's counts, one sequence count, the kernel check on, the forced count of a request
    args = sample._arguments(["--perf"])
    assert (args.perf_warmup, args.perf_steps, args.perf_requests, args.perf_trace_records) == (32, 256, 3, 160_000)
    assert args.sequences == [1] and args.check_kernels and args.force_tokens == 257 and args.perf_report is None
    assert sample._arguments(["--perf", "--sequences", "4", "--perf-steps", "8"]).sequences == [4]
    with pytest.raises(AssertionError, match="reached a load"):  # nothing above reached one; the mode itself does, behind its line
        sample.main(["--perf", "--perf-steps", "8"])
    before = capsys.readouterr().err.splitlines()
    # the count before the load: the checked request, its replay by the checks, three timed and the traced request of 26 + 8 steps, the warm-up's 26 + 32
    line = "[00] performance mode: at most 262 decode steps of this stack in all: one checked request and its replay by the checks, one warm-up request "
    line += "of 32 decode steps behind its first new token, 3 timed requests of 8 decode steps each, one traced request of up to 8; a prompt of 26 tokens "
    line += "is 26 decode steps in front of each; the reference's own passes are not counted"
    assert line in before, before[:8]
    assert before.index(line) < next(index for index, said in enumerate(before) if said.startswith("served by ("))
    _says("area", "Before the load the run prints how many decode steps of the stack all of this is at most, the replay of the checks among them, as a count;")
    assert not list(tmp_path.glob("report-*.json")), "a run that ended before its report wrote no file"


def test_perf_main_on_cpu(monkeypatch, capsys, tmp_path):
    """``main()`` with ``--perf`` on the CPU, the seeded tiny model on the stand-ins, to its last line: ``[00] PASS``, the
    report in its reduced-model form on standard error, the report file under its default name, standard output empty."""
    events = []
    sample = _sample_on_the_cpu(monkeypatch, events)
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()
    code = sample.main(["--perf", "--perf-steps", "6", "--perf-requests", "2", "--perf-warmup", "2"])
    out, err = capsys.readouterr()
    lines = err.splitlines()
    assert code == 0 and lines[-1] == "[00] PASS" and out == "", err[-4000:]
    table = next(index for index, line in enumerate(lines) if line.startswith("served by (decided before the load"))
    planned = next(index for index, line in enumerate(lines) if line.startswith("kernels, planned (decided before the load"))
    mode = next(index for index, line in enumerate(lines) if line.startswith("[00] performance mode: at most "))
    stage = next(index for index, line in enumerate(lines) if line.startswith("[00] stage "))
    assert mode < table < planned < stage, "the mode's line, the tables and with them the kernel table: before the load"
    report = lines.index("cuDNN Frontend model sample: decode performance report")
    head = "checks (1 sequence, 32 steps replayed: 26 prompt + 6 decode; blocks judged at positions 0-15, 30, 31; tokens on the first 16 steps)"
    checks = lines.index(head)
    assert stage < checks < report and "decode rate           not printed for a reduced model" in lines[report:]
    assert lines[report + 1].startswith("record class          orientation, not a number of record: reduced model; ")
    assert "model       qwen_dense, reduced model, 2 layers | 1 sequence | greedy" in lines[report:]
    gpu = lines.index("GPU         stated device, compute capability 10.0, 148 SMs")
    assert lines[gpu + 1] == "            other compute processes: not observed (nvidia-smi not found)"
    load = r"host load \d+\.\d on \d+ cores?" if hasattr(os, "getloadavg") else "host load not observed"  # read also where the driver cannot be asked
    assert re.fullmatch(r" {12}clocks observed: no \(nvidia-smi not found\) \| " + load, lines[gpu + 2]), lines[gpu + 2]
    assert "protocol    warm-up 2 steps | timed 6 steps x 2, end token ignored | launched from Python, not captured" in lines[report:]
    assert any(line.startswith("text out              sequence 0, 7 tokens (seeded weights: not language): ") for line in lines[report:])
    assert not any(
        line.startswith(("text path ", "first token ", "steps (", "traced pass ", "rate ")) for line in lines
    ), "the mode's report, not the plain one"
    (written,) = list(tmp_path.glob("report-*.json"))
    assert re.fullmatch(r"report-\d{8}T\d{6}Z\.json", written.name)
    named = lines.index(f"report file           {written.name}")  # with its time stamp the name leaves no room for the bracket on its line
    assert lines[named + 1] == "                      (every number, its definition, the protocol, the environment)"
    data = json.loads(written.read_text())
    assert data["checks"]["last_line"] == "[00] PASS" and data["model"]["name"] == "qwen_dense, reduced model" and data["protocol"]["stack"] == "fe-cublas"
    assert data["protocol"]["prompt"] == {"tokens": 26, "sha256": data["protocol"]["prompt"]["sha256"], "source": "the seeded prompt"}
    assert str(tmp_path) not in err and str(tmp_path) not in written.read_text()
    assert "[00] seeded weights: the output is not language; ids hash " in lines[-2] and lines[-3].startswith("[00] GPU memory: PyTorch peak ")
    _says("area", "# The report goes to standard error, a report file report-<UTC time>.json to the current directory, the text to standard output.")
    _says("script", "The report goes to standard error and, with every number, its definition, the protocol and the environment, into a report file")
    _says("script", "(``--perf-report FILE``; default ``report-<UTC time>.json`` in the current directory);")
    # the same command again: a second file beside the first, or an ERROR that keeps the first (the name holds the second)
    code = sample.main(
        ["--perf", "--perf-steps", "6", "--perf-requests", "2", "--perf-warmup", "2", "--perf-report", "mine.json", "--dump-census", "census.json"]
    )
    again = capsys.readouterr().err.splitlines()
    assert code == 0 and again[-1] == "[00] PASS" and len(list(tmp_path.glob("*.json"))) == 3
    _says("area", "default `report-<UTC time>.json` in the current directory, so one command can be run twice;")
    _says("family", "The default name carries a time stamp, so the same command can be run twice.")
    assert json.loads((tmp_path / "census.json").read_text())["traced_steps"] == 6, "the launch records of the traced request"
    # a report file that cannot be written after all: the run is done and keeps its verdict, and its last line is the ERROR
    perf = importlib.import_module("model_samples.perf")
    write = perf.write_report
    monkeypatch.setattr(perf, "write_report", lambda path, data: "--perf-report late.json: No space left on device")
    code = sample.main(["--perf", "--perf-steps", "6", "--perf-requests", "1", "--perf-warmup", "1", "--perf-report", "late.json"])
    monkeypatch.setattr(perf, "write_report", write)
    late = capsys.readouterr().err.splitlines()
    assert code == 1 and late[-1] == "[00] ERROR --perf-report late.json: No space left on device (the run itself: PASS)"
    assert "report file           not written: --perf-report late.json: No space left on device" in late, "the screen names no file that is not there"
    # kernels that cannot be observed: the mode's report with the reason, and the last line of a run that could not look
    for name in ("mine.json", "census.json"):
        (tmp_path / name).rename(tmp_path / f"kept_{name}")
    blind = _sample_on_the_cpu(monkeypatch, [], profiler=NO_PROFILER)
    code = blind.main(["--perf", "--perf-steps", "6", "--perf-requests", "1", "--perf-warmup", "1", "--perf-report", "blind.json"])
    unseen = capsys.readouterr().err.splitlines()
    assert code == 0 and unseen[-1] == f"[00] RAN fe-cublas stack (kernels not observed: {NO_PROFILER})"
    assert f"            kernels, traced steps: NOT OBSERVED ({NO_PROFILER})" in unseen
    _says("area", "kernels that could not be observed are `NOT OBSERVED` with the reason,", "`[00] RAN <stack> stack (kernels not observed: <reason>)`.")
    # a statement of the checks that does not hold: the script's own last line and exit code (the token choice of a timed request's first step)
    wrong = _sample_on_the_cpu(monkeypatch, [], defect={"token": _Nth(1 + 32 + 32 + 28 + 27)})
    code = wrong.main(["--perf", "--perf-steps", "6", "--perf-requests", "2", "--perf-warmup", "2", "--perf-report", "failed.json"])
    failed = capsys.readouterr().err.splitlines()
    assert (
        code == 1 and failed[-1] == "[00] FAIL" and "            timed requests give the checked request's tokens and state: FAIL (request 1, step 1)" in failed
    )
    assert "decode rate           not printed: the timed requests did not compute the checked request's tokens" in failed
    assert json.loads((tmp_path / "failed.json").read_text())["checks"]["last_line"] == "[00] FAIL", "the file's last line is the run's"
    # the GPU cells of the area run this form on the two stacks the repository's tests run, each in a fresh interpreter
    cells = (MODELS.parents[1] / "test" / "python" / "linear_attention" / "frost" / "test_model_sample_qwen_dense.py").read_text(encoding="utf-8")
    cell = cells.split("def test_the_performance_mode_of_the_seeded_run_reports_and_writes_its_file(", 1)
    assert cell[0].endswith('@pytest.mark.L0\n@pytest.mark.parametrize("stack", ["fe-cublas", "fe-entries"])\n') and "_fresh_interpreter(arguments)" in cell[1]
    _says(
        "area", "the performance mode of the seeded model (`--perf`) on both stacks, in a fresh interpreter: `[NN] PASS`, the report in its reduced-model form"
    )
    _says("area", "Without `--model` the same protocol runs on the reduced seeded model:", "The GPU test of this area runs that form.")


def _capacities(monkeypatch):
    """The capacity of every session the script builds from here on, in the order it builds them (a list that fills)."""
    package = importlib.import_module("model_samples")
    built, session = [], package.Session
    monkeypatch.setattr(package, "Session", lambda model, text, **options: built.append(options["capacity"]) or session(model, text, **options))
    return built


def test_the_session_is_built_for_the_longest_request(monkeypatch, capsys, tmp_path):
    """One session serves every request of a run of the mode, so it is built for the longest of them: the warm-up request
    where ``--perf-warmup`` is larger than ``--perf-steps``, a timed request otherwise.  A run whose warm-up is longer
    than its timed request -- the default warm-up of 32 steps behind 2 and behind 16 timed steps; a warm-up equal to, one
    above and far above the timed request -- runs to ``[00] PASS`` and writes its report, with every count as it was
    given; what the requests compute and launch does not move with the size of the session; and an ordinary run builds
    the session it built."""
    with monkeypatch.context() as patch:  # the runs themselves, to their last line
        sample = _sample_on_the_cpu(patch, [])
        built = _capacities(patch)
        patch.chdir(tmp_path)

        def run(name, *counts):
            """One run of the mode to its last line: ``(the data of its report file, its lines, the capacity of its session)``."""
            capsys.readouterr()
            del built[:]
            code = sample.main(["--perf", "--perf-requests", "1", *counts, "--perf-report", f"{name}.json"])
            out, err = capsys.readouterr()
            lines = err.splitlines()
            assert code == 0 and lines[-1] == "[00] PASS" and out == "", lines[-3:]
            (capacity,) = built
            return json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8")), lines, capacity

        # the default warm-up of 32 forced steps behind a timed request of 2 and of 16: to the last line, with the report file
        for steps in (2, 16):
            data, lines, capacity = run(f"default_warmup_behind_{steps}", "--perf-steps", str(steps))
            assert capacity == 59 and data["checks"]["last_line"] == "[00] PASS" and data["checks"]["timed_requests"] == "PASS"
            protocol = data["protocol"]
            counts = (protocol["warmup_steps"], protocol["timed_steps"], protocol["timed_requests"], protocol["new_tokens_per_request"])
            assert counts == (32, steps, 1, steps + 1), "the counts as they were given: the default warm-up, the timed steps, one new token more"
            assert f"protocol    warm-up 32 steps | timed {steps} steps x 1, end token ignored | launched from Python, not captured" in lines
            before = next(line for line in lines if line.startswith("[00] performance mode: at most "))
            assert f"one warm-up request of 32 decode steps behind its first new token, 1 timed request of {steps} decode steps each" in before
            requests = data["raw"]["requests"]
            assert [(made["kind"], len(made["t_token_s"])) for made in requests] == [("checked", steps), ("warm-up", 32), ("timed", steps), ("traced", steps)]
            assert requests[2]["ids_sha256"] == requests[0]["ids_sha256"] and requests[2]["end_state_sha256"] == requests[0]["end_state_sha256"]
            assert f"checks (1 sequence, {26 + steps} steps replayed: 26 prompt + {steps} decode; " in " ".join(lines)
        # a warm-up below, equal to, one above and far above a timed request of 6 steps: the session follows the longer of the two ...
        runs = {warmup: run(f"warmup_{warmup}_behind_6", "--perf-steps", "6", "--perf-warmup", str(warmup)) for warmup in (2, 6, 7, 300)}
        assert {warmup: capacity for warmup, (_, _, capacity) in runs.items()} == {2: 33, 6: 33, 7: 34, 300: 327}
        small, far = runs[2][0], runs[300][0]
        assert [len(made["t_token_s"]) for made in far["raw"]["requests"]] == [6, 300, 6, 6] and far["protocol"]["warmup_steps"] == 300
        assert "protocol    warm-up 300 steps | timed 6 steps x 1, end token ignored | launched from Python, not captured" in runs[300][1]
        # ... and nothing a request computes or launches moves with it: the ids of the checked, the timed and the traced request, the
        # kernels of a step, the traced steps and the checks are those of the small session, whatever the warm-up made of the session
        # (of the traced section every entry but tracing's stretch, which stays in the section: a ratio of two host times of the run)
        for warmup, (data, _, _) in runs.items():
            assert [data["raw"]["requests"][index]["ids_sha256"] for index in (0, 2, 3)] == [small["raw"]["requests"][0]["ids_sha256"]] * 3, warmup
            traced, of_small = ({key: value for key, value in one["traced"].items() if key != "tracing_stretch_percent"} for one in (data, small))
            assert data["kernels"] == small["kernels"] and traced == of_small and data["checks"] == small["checks"], warmup
            assert list(data["traced"]) == list(small["traced"]) and isinstance(data["traced"]["tracing_stretch_percent"], float), warmup
        # what does follow the session is its allocation: the attention cache has that many positions (here 2 tensors x 8 x 2 bytes each),
        # and the digest of an end state reads every byte of it -- equal sessions give equal digests, a larger one another
        assert (far["memory"]["state_gib"] - small["memory"]["state_gib"]) * GIB == pytest.approx(32 * (327 - 33))
        assert runs[6][0]["raw"]["requests"][2]["end_state_sha256"] == small["raw"]["requests"][2]["end_state_sha256"]
        assert far["raw"]["requests"][2]["end_state_sha256"] != small["raw"]["requests"][2]["end_state_sha256"]
        written = {"default_warmup_behind_2.json", "default_warmup_behind_16.json", *(f"warmup_{warmup}_behind_6.json" for warmup in runs)}
        assert {path.name for path in tmp_path.iterdir()} == written, "six runs, six report files, nothing else"
    with monkeypatch.context() as patch:  # the session alone, as a number: runs that end at their load
        planned = _sample_on_the_cpu(patch, [], loads=False)
        built = _capacities(patch)
        patch.chdir(tmp_path)

        def capacity(*arguments):
            del built[:]
            with pytest.raises(AssertionError, match="reached a load"):
                planned.main(list(arguments))
            return list(built)

        # the seeded prompt is 26 tokens; a request is its prompt, its first new token and its forced steps
        assert capacity("--perf") == [26 + 1 + 256], "the default protocol: a timed request is the longest, and the session is the one it was"
        assert capacity("--perf", "--perf-steps", "2") == capacity("--perf", "--perf-steps", "16") == [26 + 1 + 32], "the default warm-up is the longest"
        assert capacity("--perf", "--perf-steps", "16", "--perf-warmup", "4") == [26 + 1 + 16], "a shorter warm-up: the timed request"
        assert capacity("--perf", "--perf-steps", "6", "--perf-warmup", "300") == [26 + 1 + 300]
        assert capacity("--perf", "--perf-steps", "2", "--sequences", "4") == [59], "per sequence, whatever the sequence count"
        assert capacity() == [26 + 32, 26 + 32], "an ordinary run, at 1 and at 4 sequences: its prompt and its new tokens, as before"
        assert capacity("--sequences", "1", "--force-tokens", "4") == [26 + 4]
        capsys.readouterr()
        longest = planned._perf_longest
        assert longest(planned._arguments(["--perf"])) == (256, "--perf-steps", "a timed request")
        assert longest(planned._arguments(["--perf", "--perf-steps", "2"])) == (32, "--perf-warmup", "the warm-up request")
        assert longest(planned._arguments(["--perf", "--perf-steps", "32"])) == (32, "--perf-steps", "a timed request"), "equal counts: the timed request"
        rule = importlib.import_module("model_samples").StopRule((), max_new_tokens=7)
        assert planned._positions(planned._arguments(["--perf"]), 58, rule) == 58 + 257 and planned._positions(planned._arguments([]), 58, rule) == 58 + 7
        given = planned._arguments(["--perf", "--perf-steps", "2"])  # neither a count that was given nor a default is changed to make a run fit
        assert (given.perf_warmup, given.perf_steps, given.force_tokens) == (32, 2, 3)
    _says("family", "One session serves every request of a run and is built for the longest of them:")
    _says("family", "the warm-up request `--perf-warmup`, which may be the larger count.", "No count is changed to make a run fit.")


def test_the_context_is_held_against_the_longest_request(monkeypatch, capsys, tmp_path):
    """``--max-model-len`` is held against the longest request of a run of the mode before anything is planned or loaded:
    the warm-up request where it is the longer one.  The ``ERROR`` says which request is the longest and names every
    count that has to come down, and with it what to take; a run that fits exactly goes on to its load."""
    sample = _sample_on_the_cpu(monkeypatch, [], loads=False)
    monkeypatch.chdir(tmp_path)
    head = "[00] ERROR the seeded prompt: 26 prompt tokens, the first new token and "
    warmup, timed = "(the warm-up request, the longest request of this run)", "(a timed request, the longest request of this run)"

    def error(*flags):
        capsys.readouterr()
        assert sample.main(["--perf", *flags]) == 1
        lines = capsys.readouterr().err.splitlines()
        assert sum(line.startswith("[00] ERROR") for line in lines) == 1 and not any(line.startswith(("[00] planning", "served by")) for line in lines)
        return lines[-1]

    def fits(*flags):
        with pytest.raises(AssertionError, match="reached a load"):
            sample.main(["--perf", *flags])
        capsys.readouterr()

    # the warm-up request is the longest: the default 32 steps behind 2 timed steps are 59 positions
    assert error("--perf-steps", "2", "--max-model-len", "40") == (
        f"{head}--perf-warmup 32 {warmup} are 59 positions and --max-model-len is 40; take --perf-warmup 13 or fewer, or raise --max-model-len"
    )
    fits("--perf-steps", "2", "--max-model-len", "59")
    assert error("--perf-steps", "2", "--max-model-len", "58").endswith(
        " are 59 positions and --max-model-len is 58; take --perf-warmup 31 or fewer, or raise --max-model-len"
    )
    assert error("--perf-steps", "20", "--perf-warmup", "30", "--max-model-len", "40") == (
        f"{head}--perf-warmup 30 {warmup} are 57 positions and --max-model-len is 40; take --perf-steps and --perf-warmup 13 or fewer, or raise --max-model-len"
    )
    # at the default context: a warm-up that does not fit beside the prompt is refused here, not found by the run
    assert error("--perf-warmup", "4000") == (
        f"{head}--perf-warmup 4,000 {warmup} are 4,027 positions and --max-model-len is 2,051; take --perf-warmup 2,024 or fewer, or raise --max-model-len"
    )
    fits("--perf-warmup", "2024")
    # a timed request is the longest: a warm-up that fits the room is not named, one that does not is
    assert error("--perf-warmup", "13", "--max-model-len", "40") == (
        f"{head}--perf-steps 256 {timed} are 283 positions and --max-model-len is 40; take --perf-steps 13 or fewer, or raise --max-model-len"
    )
    assert error("--perf-warmup", "14", "--max-model-len", "40").endswith("; take --perf-steps and --perf-warmup 13 or fewer, or raise --max-model-len")
    assert error("--perf-steps", "32", "--max-model-len", "40") == (  # equal counts: the timed request names the longest
        f"{head}--perf-steps 32 {timed} are 59 positions and --max-model-len is 40; take --perf-steps and --perf-warmup 13 or fewer, or raise --max-model-len"
    )
    # the room that is left: 2 steps are the fewest a timed request has, so with less there is nothing to take but a larger context
    assert error("--perf-steps", "3", "--perf-warmup", "5", "--max-model-len", "29").endswith(
        f"--perf-warmup 5 {warmup} are 32 positions and --max-model-len is 29; take --perf-steps and --perf-warmup 2 or fewer, or raise --max-model-len"
    )
    assert error("--perf-steps", "2", "--perf-warmup", "5", "--max-model-len", "29").endswith("is 29; take --perf-warmup 2 or fewer, or raise --max-model-len")
    assert error("--perf-steps", "2", "--perf-warmup", "5", "--max-model-len", "28").endswith(
        f"--perf-warmup 5 {warmup} are 32 positions and --max-model-len is 28; raise --max-model-len"
    )
    fits("--perf-steps", "2", "--perf-warmup", "1", "--max-model-len", "29")
    assert not list(tmp_path.iterdir()), "a refusal writes nothing"
    _says("family", "A prompt whose tokens, its first new token and the forced steps of that longest request do not fit `--max-model-len`")
    _says("family", "it says which request is the longest and names every count to take lower (`--perf-steps`, `--perf-warmup`).")


def test_plain_report_unchanged(monkeypatch, capsys, tmp_path):
    """An ordinary run prints what it printed.  Without ``--perf``: the parser's defaults, no line of the mode, no report
    file, the plain report and the plain checks with their kernel line; ``Session.reset`` records as it did; and
    ``tie_out.run_checks`` without judged positions replays and judges every step, and still refuses to leave the kernel
    check of a target stack out."""
    from model_samples import Session, TextIO, tie_out

    events = []
    sample = _sample_on_the_cpu(monkeypatch, events)
    args = sample._arguments([])
    assert args.perf is False and args.sequences == [1, 4] and args.force_tokens == 32 and not args.check_kernels and args.perf_steps is None
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()
    package = importlib.import_module("model_samples")
    monkeypatch.delitem(sys.modules, "model_samples.perf", raising=False)  # another test of this module imported it:
    monkeypatch.delattr(package, "perf", raising=False)  # neither the module nor the package's name for it is there before the run
    code = sample.main(["--check-kernels", "--sequences", "1", "--force-tokens", "4"])
    out, err = capsys.readouterr()
    lines = err.splitlines()
    assert code == 0 and lines[-1] == "[00] PASS" and out == "" and not list(tmp_path.iterdir())
    assert "model_samples.perf" not in sys.modules and not hasattr(package, "perf"), "an ordinary run does not import the mode's module"
    assert (sample._record_steps(True, False), sample._record_steps(False, False)) == (32, None), "what an ordinary run records, as before"
    assert sample._record_steps(True, True) is None, "with a checkpoint the mode's checked request is recorded whole: the checks replay all of it"
    _says("guide", "An ordinary run prints what it printed before the mode existed and does not import the mode's module:")
    _says("guide", "the mode is a branch of its own in the script, and a pull request that touches the loop, the census, the checks or the timing shows")
    assert not any("performance" in line or line.startswith(("record class", "scope ", "decode rate")) for line in lines)
    for start in ("text path ", "first token ", "steps (", "traced pass (", "rate ", "bytes per step ", "note ", "text out "):
        assert sum(line.startswith(start) for line in lines) == 1, start
    assert "checks (1 sequence, 29 steps: 26 prompt + 3 decode)" in lines and events.count("profiler") == 1
    assert any(line.startswith("  kernels           29 traced steps: ") and line.endswith("    PASS") for line in lines)
    assert (
        "[00] decoding: 26 prompt tokens through the decode step, then 4 new tokens and a traced pass; the report and the text follow, the text is not streamed"
        in lines
    )
    # a session records unless it is told not to, and the record comes back with the next reset that records
    session, text = _session(monkeypatch, [], capacity=12)
    assert session.record.inputs.shape[0] == session.record_steps == 12
    ids = session.prefill([text.encode("ab0xy")])
    session.step(ids)
    session.reset(fresh=False)
    assert session.record.inputs.shape[0] == 12 and session.record.count == 0 and session.state.position == 0
    kept = session.state
    session.reset(fresh=False, recorded=False)
    assert session.record.inputs.shape[0] == 0 and session.state is kept, "nothing recorded; the state zeroed in place"
    session.step(session.prefill([text.encode("ab0xy")]))
    assert session.record.count == 6 and session.record.inputs.shape[0] == 0, "the steps were counted and nothing was copied"
    session.reset(fresh=False)
    assert session.record.inputs.shape[0] == 12 and session.state is kept
    session.reset()
    assert session.record.inputs.shape[0] == 12 and session.state is not kept
    # the checks of a recorded run, as they were: every step replayed and judged, the kernel line from the traced pass
    ids = session.prefill([text.encode("ab0xy")])
    for _ in range(3):
        ids = session.step(ids)
    session.trace(4)
    with pytest.raises(ValueError, match="judged positions 0, 8: the run recorded 8 steps, the positions 0 to 7"):
        tie_out.run_checks(session, prompt_tokens=5, positions=[0, 8])
    plain = tie_out.run_checks(session, prompt_tokens=5)
    assert plain.positions == (), "no judged positions were asked for: every step was judged"
    assert plain.lines[0] == "checks (1 sequence, 8 steps: 5 prompt + 3 decode)" and plain.ok and plain.kernels == "PASS"
    assert " 8 calls " in next(line for line in plain.lines if line.startswith("    linear_attention.step "))
    with pytest.raises(ValueError, match="judged positions are steps of a decode run"):
        tie_out.run_checks(Session(_tiny_model(), TextIO.synthetic(272), stack="reference", device="cpu", draft_tokens=2), prompt_tokens=0, positions=[0])
    target = Session(_tiny_model(), TextIO.synthetic(272), stack="fe-entries", device="cpu")
    with pytest.raises(ValueError, match="its kernel check cannot be switched off"):
        tie_out.run_checks(target, prompt_tokens=0, kernels=False)
    assert tie_out.DEFERRED == "DEFERRED", "the word a caller hands in when it traces after the checks"


# ---------------------------------------------------------------------------------------------- the public text
def test_the_public_text_of_the_mode_says_what_the_code_does(monkeypatch, tmp_path):
    """The READMEs, the guide and the script's own text against the code, where no other test of this module holds the
    sentence beside the behaviour (``_says``): the words, the ``stage`` row against the stages a load names, what the
    mode never prints, the flag table and the script's own text against the parser's defaults, and the README's example
    of the report against the rendered screen with every value replaced by dots."""
    from model_samples import perf

    area, guide = (MODELS / "README.md").read_text(encoding="utf-8"), (MODELS / "AGENTS.md").read_text(encoding="utf-8")
    family, script = (MODELS / "qwen_dense" / "README.md").read_text(encoding="utf-8"), SCRIPT.read_text(encoding="utf-8")
    # the words: each names something the mode prints or holds
    for word in ("*Timed request*", "*Judged position*", "*Token-to-token interval*", "*Record class*"):
        assert word in _flat(area), word
    steps = perf.Protocol().steps  # a timed request: forced steps, one token more than intervals
    _says("area", "*Timed request*: one prompt from a zero state, its first new token, then a fixed number of forced decode steps --")
    _says("area", f"with the default of {steps} steps, {steps + 1} new tokens and {steps} intervals, one token more than a serving request of {steps} output")
    _says("family", f"{steps + 1} new tokens and {steps} intervals at the default, one token more than a serving request of {steps} output tokens.")
    _says("script", f"timed decode steps of a request, behind its first new token; at the default {steps + 1} new tokens and {steps} intervals")
    _says("area", "*Judged position*: a position of a request at which every provider block is held against its reference block.")
    written_out = f"--perf-warmup {perf.Protocol().warmup} --perf-steps {steps} --perf-requests {perf.Protocol().requests} --perf-report my_gpu.json"
    _says("area", f"# The protocol written out (these are the defaults) and a file name of your own: python samples/models/qwen_dense/00_decode.py --model")
    _says("area", written_out)
    # the stage row names every stage a load of the family adds to the line
    row = next(line for line in area.splitlines() if line.startswith("| `stage` |"))
    loader = (MODELS / "model_samples" / "qwen_dense" / "model.py").read_text(encoding="utf-8")
    for stage in re.findall(r'stages\.stage\("([a-z ]+)"\)', loader):
        assert stage in row, f"the stage row does not name {stage!r}"
    assert "lay out" in row and "checksum" in row and "merge" in row
    assert "19. **The performance mode states what it observed, never what it assumes.**" in guide, "the rule of the mode: its sentences are held above"
    # what the mode never prints: not one line of a run of it, nor its file, holds these words, and what it does not observe it says so
    outcome, lines, _, _ = _mode(monkeypatch, tmp_path)
    printed = "\n".join(lines) + json.dumps(outcome.data)
    for word in ("locked", "speed-up", "baseline", "faster", "slower"):
        assert word not in printed, word
    assert "none seen" not in "\n".join(lines), "a run that could not ask the driver says so"
    assert "            other compute processes: not observed (no CUDA device)" in lines and outcome.data["observation"]["clocks"]["verdict"] == "no"
    _says("guide", "Another process, the clock and a limit reason are observed or printed as not observed;", "the report never prints the word 'locked'.")
    _says("guide", "`none seen` is printed only when the run's own process is in the driver's list;")
    _says("guide", "A report holds no speed-up, no baseline and no figure of a second GPU class.")
    _says("area", "What a run could not observe of the GPU's other processes or of its clock is `not observed`, never assumed,")
    # the flag table of the family README: every flag of the mode, with the parser's default
    sample_flags = dict(re.findall(r"^\| `(--perf[a-z-]*)(?: [A-Z]+)?` \| ([^|]*) \|", family, flags=re.MULTILINE))
    assert set(sample_flags) == {"--perf", "--perf-warmup", "--perf-steps", "--perf-requests", "--perf-report", "--perf-trace-records", "--perf-clock-lock"}
    assert (
        sample_flags["--perf-warmup"].strip() == "`32`" and sample_flags["--perf-steps"].strip() == "`256`" and sample_flags["--perf-requests"].strip() == "`3`"
    )
    assert sample_flags["--perf-trace-records"].strip() == "`160000`" and "report-<UTC time>.json" in sample_flags["--perf-report"]
    protocol = perf.Protocol()
    assert (protocol.warmup, protocol.steps, protocol.requests, protocol.trace_records) == (32, 256, 3, 160000)
    for flag in sample_flags:
        assert f'"{flag}"' in script and f"``{flag}" in script.split('"""', 2)[1], f"{flag}: a flag of the parser, named in the script's docstring"
    counts = _sample_script(monkeypatch).PERF_COUNTS  # the parser's defaults: the flag table's and the script's own text
    assert f"| `--perf-warmup N` | `{counts['--perf-warmup'][0]}` | {counts['--perf-warmup'][1]} |" in family
    assert counts["--perf-requests"][1] == "timed requests" and "| `--perf-requests N` | `3` | timed requests; " in family
    assert {flag: default for flag, (default, _) in counts.items()} == {flag: int(sample_flags[flag].strip(" `")) for flag in counts}
    _says(
        "script",
        f"(``--perf-warmup`` forced steps, default {counts['--perf-warmup'][0]});",
        f"``--perf-requests`` timed requests (default {counts['--perf-requests'][0]}),",
    )
    _says("script", f"``--perf-steps`` forced decode steps (default {counts['--perf-steps'][0]}, the end token ignored)")
    _says("script", f"``--perf-trace-records`` launch records hold (default {counts['--perf-trace-records'][0]}),")
    assert "--perf-show" not in family + area + script and "--perf-overwrite" not in family + area + script, "flags of a later version"
    # the section of the area README: its example is the screen, every value a run measures or counts replaced by dots
    section = area.split("## The performance mode\n", 1)[1].split("\n## ", 1)[0]
    example = section.split("```text\n", 1)[1].split("```", 1)[0].splitlines()
    # the stack the example names, native: its three numbers and its providers with the rows a library serves (the test of the kernels lines)
    target = {"stated": True, "declared": True, "on_target": 451, "launches": 484, "code": 178, "port": 273, "stand_ins": 0}
    target.update(by_provider={"cuDNN Frontend": 321, "FlashInfer": 129, "PyTorch": 1}, fallback_calls=32, kernels_per_step=2908, copies_per_step=200)
    today = ["fallback rows present", "kernel identity not claimed on this GPU class", "launched from Python"]
    sites, claim = _planned(monkeypatch)  # the family's own plan, 64 layers: the rows of the example's table
    table = perf.kernel_table(*_walked(sites, [_records(sites)], claim))
    form = _checkpoint_form(kernels={**target, "absent": 1}, checks={"positions_text": "0-15, 63, 64, 312, 313"}, record={"reasons": today}, per_kernel=table)
    form["observation"] = _alone()  # the example is a run that had its GPU to itself
    form["memory"].update(weights_gib=37.3, state_gib=0.5, peak_reserved_gib=40.0)
    shown = perf.screen({**form, "report_file": "report-<UTC time>.json"})

    def dotted(line):
        line = re.sub(r"positions [\d, -]+:", "positions ...:", line)
        line = line.replace("stated device", "<name>").replace("Qwen3.8-27B-NVFP4", "<checkpoint>").replace("2^30", "two^thirty")
        return " ".join(re.sub(r"(?<![\w^])\d[\d,.]*(?<![,.])", "...", line).replace("two^thirty", "2^30").split())

    assert [" ".join(line.split()) for line in example] == [dotted(line) for line in shown], "the example of the README is the screen of the code"
    assert not re.search(r"(?<![\w^])\d", "\n".join(example).replace("2^30", "")), "the example holds dots in place of every value"
    assert max(len(line) for line in shown) <= 110, "the screen the example stands for; three dots can be wider than the value they stand for"
    for said in (
        "A report holds no speed-up, no baseline and no figure of another GPU class",
        "`orientation`",
        "one checked request",
        "The first profiler pass of the process starts behind the last timed request",
    ):
        assert said in _flat(section), said
    assert "| `--perf`" in family and "The performance mode" in family


# ---------------------------------------------------------------------------------------------- the observation of the GPU
@pytest.fixture(autouse=True)
def driver_tool_not_found(monkeypatch):
    """No test of this module asks the driver of the host it runs on: the driver's tool is not found, unless a test brings
    its own stand-in for it (``_tool``)."""
    found = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *more, **named: None if name == "nvidia-smi" else found(name, *more, **named))


# a stand-in for the driver's tool: it answers the queries of the observation from the file scenario.json beside it, in
# loop mode a line every period until it is ended, and writes how it was called into calls.log and, where it ends because
# it was asked to, its process id into ends.log
_STAND_IN = """#!@PYTHON@ -IS
import json, os, signal, sys, time

here = os.path.dirname(os.path.abspath(__file__))


def scenario():
    with open(os.path.join(here, "scenario.json")) as file:
        return json.load(file)


def stamp():
    at = time.time()
    return time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(at)) + ".%03d" % int(at % 1 * 1000)


def asked(*_):
    with open(os.path.join(here, "ends.log"), "a") as log:
        log.write(str(os.getpid()) + "\\n")
    sys.exit(0)


now = scenario()
signal.signal(signal.SIGTERM, signal.SIG_IGN if now.get("deaf") else asked)
with open(os.path.join(here, "calls.log"), "a") as log:
    log.write(json.dumps({"pid": os.getpid(), "argv": sys.argv[1:]}) + "\\n")
if now.get("unbound"):
    import ctypes

    ctypes.CDLL(None).prctl(1, 0)
time.sleep(now.get("hang", 0))
if now.get("exit"):
    sys.stderr.write(now.get("stderr", ""))
    sys.exit(now["exit"])
query = next(word for word in sys.argv[1:] if word.startswith("--query-"))
loop = next((int(word.split("=")[1]) for word in sys.argv[1:] if word.startswith("--loop-ms=")), None)
while True:
    now = scenario()
    if query.startswith("--query-gpu=driver_version"):
        rows = [now["device"]]
    elif query.startswith("--query-gpu="):
        rows = [[stamp(), *now["clock"]]]
    else:
        rows = [([stamp()] if "timestamp" in query else []) + [str(value) for value in row] for row in now["processes"]]
    for row in rows + now.get("junk", []):
        print(", ".join(row), flush=True)
    if loop is None:
        break
    time.sleep(loop / 1000)
"""


def _scenario(folder, **changes):
    """What the stand-in answers from now on: the device's line of a reading (driver, SM clock in MHz, memory in use in
    MiB, utilisation), the fields of a clock sample behind its stamp, the list of compute processes (id, MiB) -- by
    default this process alone, at rest -- and ``exit`` / ``stderr`` / ``hang`` / ``junk`` for a tool that fails, ``deaf``
    for one that does not end when it is asked to, ``unbound`` for one that has cut itself loose from the process that
    started it (on Linux: it is not ended by the kernel with that process)."""
    made = {"device": ["000.00", "2000", "450", "9"], "clock": ["2000", "300.00", "0x0000000000000000"], "processes": [[os.getpid(), 400]], **changes}
    (folder / "scenario.next").write_text(json.dumps(made), encoding="utf-8")
    os.replace(folder / "scenario.next", folder / "scenario.json")  # a loop that reads meanwhile reads the old file or the new one


def _tool(monkeypatch, tmp_path, **scenario):
    """The stand-in for the driver's tool in a folder of the test, found as ``nvidia-smi``, and a GPU id for it: the
    folder.  It is a real program, so the readings and the sampler's child process run as they do on a GPU.  The host's
    load and its cores are stated too -- 4.5 at the first reading behind this call and 5.5 at every later one, on 70
    cores: what a line of the screen has room for does not depend on the host a test runs on."""
    from model_samples import observe

    folder = tmp_path / "driver tool"
    folder.mkdir(parents=True)
    tool = folder / "nvidia-smi"
    tool.write_text(_STAND_IN.replace("@PYTHON@", sys.executable), encoding="utf-8")
    tool.chmod(0o755)
    _scenario(folder, **scenario)
    found = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *more, **named: str(tool) if name == "nvidia-smi" else found(name, *more, **named))
    monkeypatch.setattr(observe, "target_of", lambda device: "GPU-00000000-stand-in")
    loads = iter([[4.5, 4.0, 3.5]])
    monkeypatch.setattr(observe, "host_load", lambda: next(loads, [5.5, 5.0, 4.5]))
    monkeypatch.setattr(observe, "host_cores", lambda: 70)
    return folder


def _calls(folder):
    """Every start of the stand-in so far: its process id and its arguments."""
    log = folder / "calls.log"
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []


class _Untouched:
    """In the place of what the timing process has no business with between the sampler's start and its stop -- the
    keeper's process object, the folder of its files: whatever is asked of it goes into ``order``."""

    def __init__(self, order, what):
        self._order, self._what = order, what

    def __getattr__(self, name):
        self._order.append((self._what, name))
        return lambda *more, **named: None


def _asked(folder):
    """The process ids of the stand-in's starts that ended because they were asked to: not those that were killed."""
    log = folder / "ends.log"
    return {int(line) for line in log.read_text(encoding="utf-8").split()} if log.exists() else set()


def _dense(monkeypatch, order, clock=((2000.0, 300.0, 0),), processes=None):
    """In the place of the sampler's child process: its samples are made behind the run, one every 0.1 ms from its start
    to its stop -- ``clock``: the (MHz, W, event bits) of the clock samples, taken in turn; ``processes``: the (id, MiB) of
    every line of the process list, default this process alone -- so that a request of a few milliseconds has samples.
    ``order`` gets ``"start"`` and ``"stop"``, and every use of the keeper or of its files (``_Untouched``)."""
    from model_samples import observe

    listed = [(os.getpid(), 400.0)] if processes is None else processes

    class Dense:
        def __init__(self, tool, target, period=observe.PERIOD_MS):
            self.period, self.child, self.folder = period, _Untouched(order, "the keeper"), _Untouched(order, "the files")
            self.clocks, self.processes, self.unread, self.started, self.stopped = [], [], 0, (0.0, 0.0), (0.0, 0.0)

        def start(self):
            order.append("start")
            self.started = (time.time(), observe.now())
            return ""

        def stop(self):
            order.append("stop")
            self.stopped = (time.time(), observe.now())
            for index in range(int((self.stopped[0] - self.started[0]) * 1e4) + 1):
                at = self.started[0] + index * 1e-4
                self.clocks.append(observe.Clock(at, *clock[index % len(clock)]))
                self.processes.extend(observe.Seen(at, pid, mib) for pid, mib in listed)

    monkeypatch.setattr(observe, "Sampler", Dense)


def _observed(monkeypatch, tmp_path, lock=None, dense=True, scenario=None, clock=((2000.0, 300.0, 0),), processes=None, **options):
    """One run of the mode (``_mode``, which takes ``options``) whose observer believes its session is on a GPU and asks
    the stand-in: ``(outcome, lines, order, the stand-in's folder)``.  ``order`` holds, as they happened, every stamp
    the mode took (``"stamp"``), every reading (``("read", when)``) and the start and the stop of the sampler; with
    ``dense`` false the sampler is the real one, on the stand-in's loops.  ``order`` also gets what a run may not do: a
    stamp taken while the process runs a thread that it did not run before the run (``("threads", how many)``: told
    by the thread itself, not by a count, which a thread that ends meanwhile would leave unchanged), and a call of the
    observer while a stage is open (``("inside the stage", name)``), whose seconds it would be counted into."""
    from model_samples import observe, perf, timing

    folder, order, stamp = _tool(monkeypatch, tmp_path, **(scenario or {})), [], perf.now
    if dense:
        _dense(monkeypatch, order, clock, processes)
    threads, opened, stage = set(threading.enumerate()), [], timing.Stages.stage

    def stamped():
        order.append("stamp")
        if set(threading.enumerate()) - threads:
            order.append(("threads", len(set(threading.enumerate()) - threads)))
        return stamp()

    @contextlib.contextmanager
    def staged(self, name):
        opened.append(name)
        try:
            with stage(self, name):
                yield
        finally:
            opened.pop()

    monkeypatch.setattr(perf, "now", stamped)
    monkeypatch.setattr(timing.Stages, "stage", staged)

    class OnAGpu(observe.Observer):
        def __init__(self, device, lock=lock):
            super().__init__(torch.device("cuda", 0), lock)

        def read(self, when):
            order.extend(("inside the stage", name) for name in opened)
            order.append(("read", when))
            super().read(when)

        def start(self):
            order.extend(("inside the stage", name) for name in opened)
            super().start()

        def stop(self):
            order.extend(("inside the stage", name) for name in opened)
            super().stop()

    monkeypatch.setattr(observe, "Observer", OnAGpu)
    outcome, lines, _, _ = _mode(monkeypatch, tmp_path, **options)
    return outcome, lines, order, folder


def _alone():
    """The observation of a run that had its GPU to itself: one reading at rest, twelve steady clock samples and this
    process alone in the list during its one timed request, on a host of 70 cores."""
    from model_samples import observe

    reading = observe.Reading("behind the last timed request", listed=True, own=400.0, used=410.0, load=[5.5, 5.0, 4.5])
    samples = [observe.Clock(1.0 + index / 100, 2000.0, 250.0, 0) for index in range(12)]
    return {**observe.summary([reading], samples, [observe.Seen(1.5, 4242, 400.0)], [("timed", 1.0, 2.0)], 4242), "host": {"load": 5.5, "cores": 70}}


def test_samples_cut_by_request_bounds(monkeypatch, tmp_path):
    """Statistic 18 on constructed samples: a sample belongs to the request whose bounds hold its stamp, both ends
    included, and to no request between two; per request the clock's count, minimum, median, mean and maximum, the share
    below a declared lock, the power median, the limit reasons, the idle samples and the process list; the screen's
    clock over the samples of the timed requests, pooled.  The bounds of a request come to the samples' clock through
    one pair of clock readings, and a second pair shows a wall clock that was set meanwhile."""
    from model_samples import observe

    own = 4242
    requests = [("warm-up", 10.0, 12.0), ("timed", 20.0, 30.0), ("timed", 40.0, 50.0)]
    clocks = [
        observe.Clock(9.999, 700.0, 90.0, 1),  # in front of the warm-up request: no request's
        observe.Clock(10.0, 1900.0, 200.0, 0),  # on a bound
        observe.Clock(12.0, 1930.0, 210.0, 0),  # on a bound
        observe.Clock(15.0, 500.0, 80.0, 1),  # between two requests
        observe.Clock(20.0, 2000.0, 250.0, 0),
        observe.Clock(22.5, 1880.0, 260.0, 0b100),  # the software power cap
        observe.Clock(25.0, 2000.0, None, 0b1),  # no power in this line; the GPU idled
        observe.Clock(30.0, 1820.0, 230.0, 0b1000100),  # the power cap and the hardware thermal slow-down
        observe.Clock(30.001, 100.0, 1.0, 0),  # behind the request
        observe.Clock(41.0, 2000.0, 255.0, 0),
        observe.Clock(45.0, None, 251.0, None),  # neither a clock nor event bits in this line
        observe.Clock(49.0, 1970.0, 249.0, 0),
    ]
    seen = [observe.Seen(21.0, own, 400.0), observe.Seen(21.0, 777, 900.0), observe.Seen(35.0, 888, 10.0), observe.Seen(44.0, own, 400.0)]
    reading = observe.Reading("before the warm-up request", listed=True, own=400.0, used=410.0, load=[4.5, 4.0, 3.5])
    made = observe.summary([reading], clocks, seen, requests, own, lock=1900)
    warm, first, second = made["requests"]
    assert [request["kind"] for request in made["requests"]] == ["warm-up", "timed", "timed"]
    assert warm["clock"] == {
        "samples": 2,
        "minimum_mhz": 1900.0,
        "median_mhz": 1915.0,
        "mean_mhz": 1915.0,
        "maximum_mhz": 1930.0,
        "share_below_declared_lock": 0.0,
    }
    assert first["clock"] == {
        "samples": 4,
        "minimum_mhz": 1820.0,
        "median_mhz": 1940.0,
        "mean_mhz": 1925.0,
        "maximum_mhz": 2000.0,
        "share_below_declared_lock": 0.5,
    }
    assert (first["power_median_w"], first["limit_reasons"], first["idle_samples"]) == (250.0, ["hw_thermal_slowdown", "sw_power_cap"], 1)
    assert (first["process_samples"], first["this_process_listed"], first["other_compute_processes"]) == (2, True, 1)
    assert second["clock"] == {
        "samples": 2,
        "minimum_mhz": 1970.0,
        "median_mhz": 1985.0,
        "mean_mhz": 1985.0,
        "maximum_mhz": 2000.0,
        "share_below_declared_lock": 0.0,
    }
    assert (second["clock_lines"], second["power_median_w"], second["limit_reasons"], second["idle_samples"]) == (3, 251.0, [], 0)
    assert (second["process_samples"], second["this_process_listed"], second["other_compute_processes"]) == (1, True, 0)
    # the screen's clock: the six samples of the two timed requests, pooled; the fewest samples of a timed request
    assert (made["clocks"]["median_mhz"], made["clocks"]["minimum_mhz"], made["clocks"]["maximum_mhz"], made["clocks"]["samples"]) == (
        1985.0,
        1820.0,
        2000.0,
        2,
    )
    assert made["other_compute_processes"] == {"verdict": "seen", "seen": 2, "reason": None}, "777 inside a request and 888 between two"
    assert made["host"]["load"] == 4.5 and made["clock_lock_declared_mhz"] == 1900 and made["sampler_period_ms"] == observe.PERIOD_MS == 50
    _defines("gpu_and_host", "Per timed request: the SM clock samples (count, minimum, median, mean, maximum), the power median and the limit reasons")
    _defines("gpu_and_host", "A sample belongs to a request when its time stamp lies between the request being handed to the text path and its last")
    _defines("gpu_and_host", "The screen's clock is the median and the minimum over the samples of all timed requests.")
    _says(
        "area", "A sample belongs to a request when its time stamp lies between the request being handed to the text path and its last token's ids on the host."
    )
    _says("area", "`yes` with the median and the minimum of the SM clock over the samples of all timed requests,")
    # a line of the tool: its stamp is local time, a field the driver does not give is none, another line is no sample
    line = observe.clock_sample("2001/02/03 04:05:06.789, 2000, 111.11, 0x0000000000000102")
    assert line == observe.Clock(time.mktime((2001, 2, 3, 4, 5, 6, 0, 0, -1)) + 0.789, 2000.0, 111.11, 0x102)
    assert observe.clock_sample("2001/02/03 04:05:06.789, [N/A], [Not Supported], [Unknown Error]")[1:] == (None, None, None)
    assert observe.clock_sample("not a line of the tool") is None and observe.clock_sample("2001-02-03, 2000, 111.11, 0x0") is None
    assert observe.process_sample("2001/02/03 04:05:06.812, 7654321, 12345")[1:] == (7654321, 12345.0)
    assert observe.process_sample("2001/02/03 04:05:06.812, a name, 12345") is None and observe.process_sample("") is None
    assert observe.limits_of(0x1FF) == list(observe.EVENTS[1:]) and observe.limits_of(1) == [] and observe.limits_of(1 << 12) == ["bit 12"]
    with monkeypatch.context() as patch:  # the stamp is the host's local time in whatever zone the host keeps: 04:05:06 in Tokyo is 19:05:06 UTC the day before
        patch.setenv("TZ", "Asia/Tokyo")
        time.tzset()
        try:
            elsewhere = observe.clock_sample("2001/02/03 04:05:06.789, 2000, 111.11, 0x0000000000000102")
        finally:
            patch.undo()
            time.tzset()
    assert elsewhere.at == pytest.approx(981140706.789, abs=1e-6)
    # a field that is no size -- not a number, not finite, below zero -- is a field the driver does not give; a clock of zero is no clock
    assert observe.clock_sample("2001/02/03 04:05:06.789, nan, inf, 0x0")[1:] == (None, None, 0)
    assert observe.clock_sample("2001/02/03 04:05:06.789, -1, -0.5, 0x0")[1:] == (None, None, 0)
    assert observe.clock_sample("2001/02/03 04:05:06.789, 0, 0.0, 0x0")[1:] == (None, 0.0, 0)
    assert observe.clock_sample("2001/02/03 04:05:06.789, 1e400, -inf, 0x0")[1:] == (None, None, 0)
    assert observe.process_sample("2001/02/03 04:05:06.812, 7654321, nan")[1:] == (7654321, None)
    unsized = [observe.clock_sample(f"2001/02/03 04:05:06.{index:03d}, nan, 250.00, 0x0") for index in range(12)]
    blind = observe.summary([reading], unsized, [], [("timed", unsized[0].at, unsized[-1].at)], own)
    assert observe.clocks_text(blind) == "no (the driver reports no SM clock)" and observe.clock_reasons(blind) == ["clocks not observed"]
    assert json.loads(json.dumps(blind, allow_nan=False)) == blind, "nothing that is not a number reaches the file"
    # the stamps of a request are on the clock of the run, the tool's on the wall clock: one pair of readings at the start maps them
    _tool(monkeypatch, tmp_path)
    watch = observe.Observer(torch.device("cuda", 0), 1900)
    assert watch.reason == "" and watch.pid == os.getpid()
    watch.readings, watch.pid = [reading], own
    watch.sampler = types.SimpleNamespace(
        started=(1000.0, 5.0), stopped=(1100.0, 105.0), clocks=[observe.Clock(1020.0, 2000.0, 250.0, 0)], processes=[], unread=3, period=20
    )
    mapped = watch.section([("timed", 25.0, 35.0), ("timed", 45.0, 55.0)])  # on the wall clock: 1020 to 1030, 1040 to 1050
    assert [request["clock"]["samples"] for request in mapped["requests"]] == [1, 0] and mapped["unread_lines"] == 3 and mapped["clock_offset_drift_ms"] == 0.0
    assert mapped["sampler_period_ms"] == 20, "the period the loops were started with"
    # a second pair at the stop: the two clocks may be a millisecond apart from the first pair, the resolution of the tool's stamp, and no more
    assert observe.DRIFT_LIMIT_MS == 1.0
    watch.sampler.stopped = (1100.0009, 105.0)
    held = watch.section([("timed", 25.0, 35.0)])
    assert held["requests"][0]["clock"]["samples"] == 1 and held["clocks"]["reason"] is None and held["clock_offset_drift_ms"] == pytest.approx(0.9)
    watch.sampler.stopped = (1100.0011, 105.0)  # the wall clock was set by more than that meanwhile: the cut is not made
    moved = watch.section([("timed", 25.0, 35.0)])
    assert moved["clocks"]["reason"] == moved["other_compute_processes"]["reason"] == "the host's clock moved during the timed requests"
    assert moved["requests"][0]["clock"]["samples"] == 0 and moved["clock_offset_drift_ms"] == pytest.approx(1.1)
    watch.sampler.stopped = (1099.9989, 105.0)  # ... in either direction
    back = watch.section([("timed", 25.0, 35.0)])
    assert back["clocks"]["reason"] == "the host's clock moved during the timed requests" and back["clock_offset_drift_ms"] == pytest.approx(-1.1)
    # a pair: the wall clock read between two readings of the stamps' clock, and of three such readings the closest -- here the second,
    # a thousandth of a second wide beside two that were put off for seconds
    ticks, walls = iter([0.0, 5.0, 10.0, 10.001, 20.0, 29.0]), iter([100.0, 110.0, 120.0])
    with monkeypatch.context() as patch:
        patch.setattr(observe, "now", lambda: next(ticks))
        patch.setattr(observe, "time", types.SimpleNamespace(time=lambda: next(walls)))
        pair = observe._pair()
    assert pair == (110.0, pytest.approx(10.0005)) and next(ticks, None) is None and next(walls, None) is None


def test_none_seen_needs_the_own_process():
    """An empty list of compute processes is not ``none seen``: this process itself must be in every reading and in a
    process sample of every timed request.  A list that hides processes therefore ends in ``not observed``."""
    from model_samples import observe

    def reading(**changes):
        return observe.Reading("before the warm-up request", **{"listed": True, "own": 400.0, "used": 410.0, **changes})

    def timed(**changes):
        return {"kind": "timed", "process_samples": 20, "this_process_listed": True, **changes}

    alone = observe.others([reading(), reading()], [timed(), timed()], 0)
    assert alone == {"verdict": "none seen", "seen": 0, "reason": None}
    assert observe.others_text({"other_compute_processes": alone}) == "none seen" and observe.others_reason({"other_compute_processes": alone}) == ""
    hidden = {"verdict": "not observed", "seen": None, "reason": "this process is not in the driver's list"}
    assert observe.others([reading(), reading(listed=False, own=None)], [timed()], 0) == hidden, "missing from one reading"
    assert observe.others([reading(listed=False, own=None), reading(), reading()], [timed()], 0) == hidden, "from the first of three readings"
    assert observe.others([reading(), reading(listed=False, own=None), reading()], [timed()], 0) == hidden, "from the one in the middle"
    assert observe.others([reading()], [timed(), timed(this_process_listed=False)], 0) == hidden, "missing from the samples of one timed request"
    assert observe.others([reading()], [timed(this_process_listed=False), timed()], 0) == hidden, "from those of the first timed request"
    # the warm-up request and the traced request are not asked: the verdict is of the readings and of the timed requests
    beside = [
        {"kind": "warm-up", "process_samples": 0, "this_process_listed": False},
        timed(),
        {"kind": "traced", "process_samples": 4, "this_process_listed": False},
    ]
    assert observe.others([reading()], beside, 0) == alone
    # a count of other processes is stated only of a list in which the run finds itself: another id, and this process missing from a
    # reading or from the samples of a timed request -- one of the others may be this process under another id
    assert observe.others([reading(), reading()], [timed()], 2) == {"verdict": "seen", "seen": 2, "reason": None}
    assert observe.others([reading(listed=False, own=None), reading()], [timed()], 2) == hidden
    assert observe.others([reading()], [timed(), timed(this_process_listed=False)], 1) == hidden
    failed = observe.Reading("behind the last timed request", observe.TOOL_ERROR)
    assert observe.others([reading(), failed], [timed(process_samples=0, this_process_listed=False)], 1)["verdict"] == "seen", "no list, nothing hidden"
    assert observe.others([reading(listed=False, own=None), failed], [timed()], 1) == hidden, "the reason is the list without this process"
    renamed = observe.summary([reading(listed=False, own=None, others={99001: 400.0})], [], [observe.Seen(1.5, 99001, 400.0)], [("timed", 1.0, 2.0)], 4242)
    assert observe.others_text(renamed) == "not observed (this process is not in the driver's list)"
    assert observe.others_reason(renamed) == "other processes not observed" and renamed["readings"][0]["other_compute_processes"] == 1
    assert observe.others([], [timed()], 0) == hidden, "no reading at all"
    assert observe.others_text({"other_compute_processes": hidden}) == "not observed (this process is not in the driver's list)"
    assert observe.others_reason({"other_compute_processes": hidden}) == "other processes not observed"
    unsampled = [timed(), timed(process_samples=0, this_process_listed=False)]
    assert observe.others([reading()], unsampled, 0)["reason"] == observe.others([reading()], [], 0)["reason"] == "no process sample inside a timed request"
    assert observe.others([reading(), observe.Reading("behind the last timed request", observe.TOOL_ERROR)], [timed()], 0)["reason"] == observe.TOOL_ERROR
    assert observe.others([reading()], [timed()], 0, observe.NO_SAMPLER)["reason"] == "the sampler could not be started"
    # the tool prints nothing for an empty list: a reading and a loop that saw no process at all
    empty = observe.summary([reading(listed=False, own=None, used=0.0)], [], [], [("timed", 1.0, 2.0)], 4242)
    assert empty["other_compute_processes"]["verdict"] == "not observed" and empty["readings"][0]["this_process_listed"] is False
    _defines("gpu_and_host", "none seen is stated only when this process itself is in the driver's list and the device's memory in use agrees with")
    _defines("gpu_and_host", "and is stated only when no reading and no timed request's samples showed a list without this process itself;")
    _defines("gpu_and_host", "of such a list no count is stated: not observed, with the reason.")
    _says("area", "`none seen` only when no other process was in the driver's list in a reading or in a sample, this process itself was in every reading")
    _says("area", "this process itself was in every reading and in a sample of every timed request, and the device's memory in use agreed with")
    _says("area", "A process list the run could not read, or in which it did not find itself, or beside a device memory in use that its own does not")
    _says("area", "a list that hides processes is never read as an empty one.", "a list is never assumed to be empty.")
    _says("area", "`<n> seen`, the distinct other processes over every reading and every sample -- the reading before the load and the time between")
    _says(
        "area", "and stated only when no reading and no timed request's samples showed a list without this process itself; or `not observed` with the reason --"
    )
    _says("guide", "A count of other processes is stated only of a list that shows the run's own entry too.")
    _says("guide", "`none seen` is printed only when the run's own process is in the driver's list;")
    _says("guide", "A list of processes counts as empty only beside the run's own entry and a device memory in use that the run's own explains.")


def test_own_process_is_not_another(monkeypatch, tmp_path):
    """This process's own entry of the driver's list is never counted as another process; every other id is, once, however
    often and wherever it was seen -- in a reading, inside a request or between two.  Of another process the file keeps
    its memory and nothing else."""
    from model_samples import observe

    own = 4242
    before = observe.Reading("before the load", listed=True, own=300.0, used=1200.0, others={777: 900.0})
    behind = observe.Reading("behind the last timed request", listed=True, own=400.0, used=410.0)
    samples = [observe.Seen(1.5, own, 400.0), observe.Seen(1.5, 777, 900.0), observe.Seen(1.6, 888, 5.0), observe.Seen(5.0, 888, 5.0)]
    made = observe.summary([before, behind], [], samples, [("timed", 1.0, 2.0)], own)
    assert made["other_compute_processes"] == {"verdict": "seen", "seen": 2, "reason": None}
    assert made["requests"][0]["other_compute_processes"] == 2 and made["requests"][0]["this_process_listed"]
    assert observe.others_text(made) == "2 seen" and observe.others_reason(made) == "shared GPU"
    _defines("gpu_and_host", "<n> seen is the number of distinct other compute processes over every reading and every sample, the reading before the")
    _defines("gpu_and_host", "the reading before the load and the time between two requests included,")
    assert made["readings"][0]["other_compute_processes"] == 1 and made["readings"][0]["other_memory_mib"] == [900.0]
    assert "777" not in json.dumps(made) and "888" not in json.dumps(made), "no id of another process in the file"
    early = observe.summary([before, behind], [], samples[:1], [("timed", 1.0, 2.0)], own)
    assert early["other_compute_processes"] == {"verdict": "seen", "seen": 1, "reason": None}, "a process that was in the reading before the load alone"
    late = observe.summary([behind, before], [], samples[:1], [("timed", 1.0, 2.0)], own)
    assert late["other_compute_processes"] == {"verdict": "seen", "seen": 1, "reason": None}, "a process that was in the last reading alone"
    only = observe.summary([behind], [], samples[:1], [("timed", 1.0, 2.0)], own)
    assert only["other_compute_processes"]["verdict"] == "none seen" and only["requests"][0]["other_compute_processes"] == 0
    # a reading through the tool: this process's line is its own, the other line is another process
    folder = _tool(monkeypatch, tmp_path, processes=[[os.getpid(), 400], [777, 900]], device=["000.00", "2000", "1310", "55"])
    read = observe.reading(str(folder / "nvidia-smi"), "GPU-00000000-stand-in", os.getpid(), "before the load")
    assert (read.reason, read.listed, read.own, read.others, read.used, read.clock, read.utilization) == ("", True, 400.0, {777: 900.0}, 1310.0, 2000.0, 55.0)
    assert read.driver == "000.00" and read.load == [4.5, 4.0, 3.5] and read.seconds > 0.0
    data = read.data()
    assert (data["this_process_mib"], data["other_compute_processes"], data["other_memory_mib"], data["memory_agrees"]) == (400.0, 1, [900.0], False)
    assert [call["argv"] for call in _calls(folder)] == [
        ["--id=GPU-00000000-stand-in", "--query-gpu=driver_version,clocks.sm,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        ["--id=GPU-00000000-stand-in", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
    ], "a reading is two queries: no name of a process is asked for"
    _says("area", "Another process on the GPU is the reason `shared GPU`.")
    _says("guide", "no process name is asked for, and the report keeps no id of another process.")


def test_memory_beyond_own_is_not_observed():
    """A list in which this process stands alone counts only beside a device memory in use that this process's own
    explains: more than the slack beyond it, in any reading, and the other processes are ``not observed``."""
    from model_samples import observe

    slack = observe.MEMORY_SLACK_MIB
    assert slack == 64, "MiB: pinned; a proposal until an exclusive run has sized it"

    def reading(used):
        return observe.Reading("behind the last timed request", listed=True, own=1000.0, used=used)

    timed = [{"kind": "timed", "process_samples": 3, "this_process_listed": True}]
    assert reading(1000.0 + slack).agrees() and not reading(1001.0 + slack).agrees() and not reading(None).agrees()
    assert observe.others([reading(1000.0 + slack)], timed, 0)["verdict"] == "none seen"
    beyond = observe.others([reading(1010.0), reading(1001.0 + slack)], timed, 0)
    assert beyond == {"verdict": "not observed", "seen": None, "reason": "device memory in use exceeds this process's own"}
    assert observe.others([reading(1001.0 + slack), reading(1010.0), reading(1010.0)], timed, 0) == beyond, "in the first of three readings"
    assert observe.others([reading(1010.0), reading(1001.0 + slack), reading(1010.0)], timed, 0) == beyond, "in the one in the middle"
    assert observe.others_text({"other_compute_processes": beyond}) == "not observed (device memory in use exceeds this process's own)"
    # a figure the driver does not give is not a figure beyond this process's own: the reason says which it was
    unknown = {"verdict": "not observed", "seen": None, "reason": "the driver reports no memory figure"}
    ownless = observe.Reading("before the warm-up request", listed=True, own=None, used=1010.0)
    assert observe.others([reading(1010.0), reading(None)], timed, 0) == unknown, "the device's memory in use"
    assert observe.others([ownless, reading(1010.0)], timed, 0) == unknown, "this process's own"
    assert observe.others_text({"other_compute_processes": unknown}) == "not observed (the driver reports no memory figure)"
    assert observe.others_reason({"other_compute_processes": unknown}) == "other processes not observed"
    assert observe.others_reason({"other_compute_processes": beyond}) == "other processes not observed"
    assert reading(1001.0 + slack).data()["memory_agrees"] is False and reading(1000.0).data()["device_memory_used_mib"] == 1000.0
    assert _alone()["rules"] == {"least_clock_samples": 10, "clock_floor_percent": 97, "clock_ceiling_percent": 103, "memory_slack_mib": slack}
    _says("area", "or beside a device memory in use that its own does not explain, is `other processes not observed`:")
    _says("area", "the device's memory in use agreed with this process's own;")
    _says("area", "the two verdicts of the `GPU` lines and the rules behind them), `kernels`,")


def test_clock_value_forms(monkeypatch):
    """Every form of the clock line: ``yes`` with the median and the minimum over the timed requests' samples, with a
    declared lock, with the limit reasons by the driver's names; ``thin``; ``no`` with its reason.  The reasons each gives
    the record class: a minimum below 97 % of the median, or of a declared lock; a maximum above 103 % of a declared
    lock; a limit reason.  That the GPU idled is no limit.  Every timed request is judged, and the timed requests
    alone.  No form holds the word 'locked'."""
    from model_samples import observe

    def watched(mhz, lock=None, events=0, more=(), failed=""):
        """The observation of a timed request with these clock samples (``more``: those of a second timed request)."""
        first = [observe.Clock(1.0 + index / 1000, value, 250.0, events) for index, value in enumerate(mhz)]
        second = [observe.Clock(3.0 + index / 1000, value, 250.0, 0) for index, value in enumerate(more)]
        return observe.summary([], first + second, [], [("timed", 1.0, 2.0), ("timed", 3.0, 4.0)] if more else [("timed", 1.0, 2.0)], 4242, lock, failed)

    def said(made):
        return observe.clocks_text(made), observe.clock_reasons(made)

    steady = [2000.0] * 12
    assert said(watched(steady)) == ("yes (SM 2000 MHz median, 2000 min; no limit reported)", [])
    # no lock declared: the minimum against 97 % of the median (1940 MHz): a sample at it is not below it
    assert said(watched(steady[:11] + [1940.0])) == ("yes (SM 2000 MHz median, 1940 min; no limit reported)", [])
    assert said(watched(steady[:11] + [1939.0])) == ("yes (SM 2000 MHz median, 1939 min; no limit reported)", ["SM clock fell to 1939 MHz"])
    # a declared lock: at the clock, above it, below it; the minimum and the maximum are held against it, not against the median
    assert said(watched(steady, lock=2000)) == ("yes (SM 2000 MHz median, 2000 min; lock 2000 MHz declared; no limit reported)", [])
    assert said(watched(steady, lock=2600)) == ("yes (SM 2000 MHz median, 2000 min; lock 2600 MHz declared; no limit reported)", ["below the declared lock"])
    assert said(watched(steady, lock=1600)) == ("yes (SM 2000 MHz median, 2000 min; lock 1600 MHz declared; no limit reported)", ["above the declared lock"])
    assert said(watched(steady[:11] + [1939.0], lock=2000))[1] == ["below the declared lock"]
    assert said(watched([1450.0] * 6 + [1750.0] * 6, lock=1600))[1] == ["below the declared lock", "above the declared lock"]
    assert said(watched([1553.0] * 6 + [1648.0] * 6, lock=1600))[1] == [], "inside 97 % and 103 % of the lock"
    assert said(watched([1551.0] * 6 + [1600.0] * 6, lock=1600))[1] == ["below the declared lock"], "one MHz under 97 %"
    assert said(watched([1600.0] * 6 + [1649.0] * 6, lock=1600))[1] == ["above the declared lock"], "one MHz over 103 %"
    assert watched(steady[:9] + [1900.0] * 3, lock=2000)["requests"][0]["clock"]["share_below_declared_lock"] == 0.25
    assert watched(steady)["requests"][0]["clock"]["share_below_declared_lock"] is None
    # limit reasons, by the driver's own names; that the GPU idled is counted and is none
    assert said(watched(steady, events=0b100)) == ("yes (SM 2000 MHz median, 2000 min; limit reported: sw_power_cap)", ["limit reported: sw_power_cap"])
    several = said(watched(steady, lock=2000, events=0b1001101))
    assert several[0] == "yes (SM 2000 MHz median, 2000 min; lock 2000 MHz declared; limit reported: hw_slowdown, hw_thermal_slowdown, sw_power_cap)"
    assert several[1] == ["limit reported: hw_slowdown, hw_thermal_slowdown, sw_power_cap"]
    idle = watched(steady, events=0b1)
    assert said(idle) == ("yes (SM 2000 MHz median, 2000 min; no limit reported)", []) and idle["requests"][0]["idle_samples"] == 12
    assert said(watched(steady, events=0b100, more=steady))[1] == ["limit reported: sw_power_cap"], "a limit in the first timed request, none in the last"
    _says("area", "a lock that was declared, and the limit reasons the driver reported in a timed request, by its own names, or `no limit reported`;")
    # thin: a timed request with fewer than ten clock samples; every reason that applies, in order
    assert observe.LEAST_SAMPLES == 10 and said(watched(steady[:9])) == ("thin (9 samples in a timed request)", ["too few clock samples"])
    assert said(watched(steady, more=steady[:3], lock=2000)) == ("thin (3 samples in a timed request; lock 2000 MHz declared)", ["too few clock samples"])
    assert said(watched(steady[:1]))[0] == "thin (1 sample in a timed request)"
    assert said(watched(steady[:3], more=steady)) == ("thin (3 samples in a timed request)", ["too few clock samples"]), "the first timed request, not the last"
    # the timed requests alone: a warm-up request and a traced request that were thin, low and limited are in the file and are no reason
    warm = [observe.Clock(0.1 + index / 1000, 1300.0, 250.0, 0b1000) for index in range(3)]
    traced = [observe.Clock(5.0 + index / 1000, 1300.0, 250.0, 0b100) for index in range(3)]
    timed = [observe.Clock(1.0 + index / 1000, 2000.0, 250.0, 0) for index in range(12)]
    beside = observe.summary([], warm + timed + traced, [], [("warm-up", 0.0, 0.5), ("timed", 1.0, 2.0), ("traced", 5.0, 6.0)], 4242)
    assert said(beside) == ("yes (SM 2000 MHz median, 2000 min; no limit reported)", [])
    assert [(made["kind"], made["clock"]["samples"], made["clock"]["minimum_mhz"], made["limit_reasons"]) for made in beside["requests"]] == [
        ("warm-up", 3, 1300.0, ["hw_slowdown"]),
        ("timed", 12, 2000.0, []),
        ("traced", 3, 1300.0, ["sw_power_cap"]),
    ]
    assert said(watched(steady[:5] + [1000.0], events=0b100))[1] == ["too few clock samples", "SM clock fell to 1000 MHz", "limit reported: sw_power_cap"]
    # no: nothing to judge, and why
    assert said(watched([])) == ("no (no clock sample inside a timed request)", ["clocks not observed"])
    assert said(watched([None] * 12)) == ("no (the driver reports no SM clock)", ["clocks not observed"])
    assert said(watched(steady, failed=observe.NO_SAMPLER)) == ("no (the sampler could not be started)", ["clocks not observed"])
    assert said(watched([], lock=2000))[0] == "no (no clock sample inside a timed request; lock 2000 MHz declared)"
    # the host's load beside the clocks, and the cores this process may use -- not those the host has
    with monkeypatch.context() as patch:
        patch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
        patch.setattr(os, "cpu_count", lambda: 64)
        cores = observe.host_cores()
    assert cores == 3
    assert observe.host_text({"host": {"load": 7.345, "cores": 70}}) == "host load 7.3 on 70 cores"
    assert (
        observe.host_text({"host": {"load": 0.0, "cores": 1}}) == "host load 0.0 on 1 core"
        and observe.host_text({"host": {"load": None, "cores": 8}}) == "host load not observed"
    )
    for made in (watched(steady, lock=2000), watched(steady[:3], lock=2000), watched([], lock=2000), watched(steady, lock=2600, events=0b10)):
        assert "locked" not in observe.clocks_text(made) + " ".join(observe.clock_reasons(made)) + json.dumps(made)
    _says("area", f"`thin` where a timed request has fewer than {observe.LEAST_SAMPLES} clock samples; `no` with the reason.")
    _says("area", "`yes` says that the clock was sampled in every timed request, not that it held: a clock that fell, that left a declared lock or that the")
    _says("area", "or that the driver limited is a reason of the `record class` line.", "The clocks give `clocks not observed`; `too few clock samples`;")
    _says("area", f"`SM clock fell to <n> MHz`, a sample below {observe.FLOOR * 100:.0f} % of the median where no lock was declared;")
    _says("area", f"a sample below {observe.FLOOR * 100:.0f} % or above {observe.CEILING * 100:.0f} % of a lock that was declared;")
    assert round((1 - observe.FLOOR) * 100) == round((observe.CEILING - 1) * 100) == 3, "the family README: more than 3 % below or above it"
    _says("area", "That the driver saw the GPU idle in a sample is counted in the file and is no limit.", "the report never prints the word 'locked'.")
    _says("area", "`host load`: the host's load average over the last minute, read behind the last timed request, and the cores this process may use:")
    _says("guide", "A lock is the operator's declaration: it is printed as declared, held against the observed clock in both directions, and never set.")


def test_the_record_class_takes_what_was_observed(monkeypatch, tmp_path):
    """Conditions C5 and C6: what the run observed of the other processes and of the clocks is a reason of line 2, in the
    order of the conditions, or none.  Nothing observed makes a run that is launched from Python a controlled run."""
    from model_samples import observe, perf

    alone = _alone()
    assert (observe.others_reason(alone), observe.clock_reasons(alone)) == ("", [])
    assert perf.record_class(False, False, "", True, "", observe.others_reason(alone), observe.clock_reasons(alone)) == ("controlled", [])
    assert perf.record_class(False, False, "", False, "", "", []) == ("orientation", ["launched from Python"]), "what every run of this tree is"
    loaded = {**alone, "host": {"load": 999.0, "cores": 1}}  # the host's load is printed and is no condition
    assert perf.record_class(False, False, "", True, "", observe.others_reason(loaded), observe.clock_reasons(loaded)) == ("controlled", [])
    assert observe.host_text(loaded) == "host load 999.0 on 1 core"
    _says("area", "and the cores this process may use: printed, not a condition |")
    several = ["too few clock samples", "SM clock fell to 1000 MHz", "limit reported: sw_power_cap"]
    assert perf.record_class(False, True, "", False, "", "shared GPU", several)[1] == ["fallback rows present", "launched from Python", "shared GPU", *several]
    assert perf.record_class(False, False, "", True, "", "other processes not observed", ["clocks not observed"])[1] == [
        "other processes not observed",
        "clocks not observed",
    ]
    # a run that had its GPU to itself: the two reasons of a run that could not look are gone, the others stay
    outcome, lines, _, _ = _observed(monkeypatch, tmp_path / "alone")
    assert outcome.data["record"]["reasons"] == ["reduced model", "fallback rows present", "no target kernel set declared", "launched from Python"]
    assert outcome.data["record"]["class"] == "orientation" and outcome.data["record"]["performance_claim_eligible"] is False
    assert lines[lines.index("cuDNN Frontend model sample: decode performance report") + 1 :][:2] == [
        "record class          orientation, not a number of record: reduced model; fallback rows present;",
        "                      no target kernel set declared; launched from Python",
    ]
    assert "GPU         stated device, compute capability 10.0, 148 SMs | other compute processes: none seen" in lines
    # another process in the list, in a reading and in the samples: seen, and the reason
    other = [(os.getpid(), 400.0), (777, 900.0)]
    shared, lines, _, _ = _observed(monkeypatch, tmp_path / "shared", scenario={"processes": [[os.getpid(), 400], [777, 900]]}, processes=other)
    assert shared.data["record"]["reasons"][-2:] == ["launched from Python", "shared GPU"]
    assert "GPU         stated device, compute capability 10.0, 148 SMs | other compute processes: 1 seen" in lines
    # a clock that fell in a sample, and a limit the driver reported
    dipped = ((2000.0, 300.0, 0),) * 9 + ((1600.0, 180.0, 0b100),)
    capped, lines, _, _ = _observed(monkeypatch, tmp_path / "capped", clock=dipped)
    assert capped.data["record"]["reasons"][-3:] == ["launched from Python", "SM clock fell to 1600 MHz", "limit reported: sw_power_cap"]
    clocks = lines.index("            clocks observed: yes (SM 2000 MHz median, 1600 min; limit reported: sw_power_cap)")
    assert lines[clocks + 1].startswith("            host load "), "the two parts do not fit one line: a line break in the place of the separator"
    _says("area", "None of this makes a run a controlled run while its launch sequence is not captured.")
    _says("area", "`below the declared lock` and `above the declared lock`,", "and `limit reported: <names>`, the driver's own names.")


def test_the_observer_is_called_outside_every_request(monkeypatch, tmp_path):
    """Where the timing process touches the observation: a reading and the start of the sampler behind the checks and in
    front of the warm-up request's first stamp; a reading behind the last stamp of the last timed request; the stop
    behind the last stamp of the traced request.  Between the first and the last stamp of the warm-up and of every timed
    request, and inside the traced request, it makes no call of the observer, asks nothing of the keeper or of its
    files and runs no thread; and no call of the observer falls into a stage, whose seconds the file holds.  The file's
    ``observation`` holds every reading, every request the sampler ran beside, the two verdicts and the rules; the
    screen is rendered from it."""
    from model_samples import observe, perf

    outcome, lines, order, folder = _observed(monkeypatch, tmp_path, steps=6, requests=2, warmup=2)
    marks = [index for index, event in enumerate(order) if event != "stamp"]
    # nothing else is in the order of the run: no use of the keeper or of its files, no stamp beside another thread, no call inside a stage
    assert [order[index] for index in marks] == [("read", "before the warm-up request"), "start", ("read", "behind the last timed request"), "stop"]
    checked, warm, timed, traced = 2 + 2 * 6, 2 + 2 * 2, 2 * (2 + 2 * 6), 2 + 2 * 6  # t_request and t_first, then two stamps a step
    assert order[: marks[0]].count("stamp") == checked and marks[1] == marks[0] + 1, "behind the checked request; the sampler starts behind the reading"
    assert order[marks[1] : marks[2]].count("stamp") == warm + timed, "the warm-up request and both timed requests: not one call among their stamps"
    assert order[marks[2] : marks[3]].count("stamp") == traced and order[marks[3] :].count("stamp") == 0, "the traced request, then the stop"
    # ... and the file says of the sampler what this order shows: nothing was asked of the keeper or of its files between the start and the stop
    beside = outcome.data["protocol"]["in_a_timed_request"]
    assert list(beside) == ["recorder", "call_log", "profiler", "text", "sampler"]
    assert beside["sampler"] == observe.SAMPLER == "a child process that the timing process neither reads nor polls between its start and its stop"
    _says("area", "`protocol` (the counts, the prompt's hash, where the profiler first started, what a timed request ran with and")
    _says("area", "what the sampler is beside it: a child process that the timing process neither reads nor polls between its start and its stop,")
    _says("guide", "The report file says what the sampler was to the timed requests of its run, or that none ran, with the reason (`protocol`).")
    watched = outcome.data["observation"]
    assert (watched["source"], watched["reason"], watched["sampler_period_ms"], watched["clock_lock_declared_mhz"]) == ("nvidia-smi", None, 50, None)
    assert [made["when"] for made in watched["readings"]] == ["before the warm-up request", "behind the last timed request"]
    for made in watched["readings"]:
        assert (made["reason"], made["this_process_listed"], made["this_process_mib"], made["device_memory_used_mib"]) == (None, True, 400.0, 450.0)
        assert (made["memory_agrees"], made["other_compute_processes"], made["sm_clock_mhz"], made["utilization_percent"]) == (True, 0, 2000.0, 9.0)
    assert [made["kind"] for made in watched["requests"]] == ["warm-up", "timed", "timed", "traced"], "every request the sampler ran beside"
    _says("area", "Beside the warm-up request, the timed requests and the traced request a child process samples the SM clock,")
    _says("area", "| `GPU` | what the run observed of its GPU and of its host. `other compute processes`:")
    assert all(made["clock"]["samples"] >= observe.LEAST_SAMPLES and made["this_process_listed"] for made in watched["requests"][1:3])
    assert watched["other_compute_processes"] == {"verdict": "none seen", "seen": 0, "reason": None}
    assert (watched["clocks"]["verdict"], watched["clocks"]["median_mhz"], watched["clocks"]["minimum_mhz"], watched["clocks"]["reasons"]) == (
        "yes",
        2000.0,
        2000.0,
        [],
    )
    # a request is cut from the request being handed to the text path to its last token: this sampler has a sample every 0.1 ms
    behind = [made for made in outcome.data["raw"]["requests"] if made["kind"] != "checked"]
    for made, request in zip(watched["requests"], behind):
        assert abs(made["clock"]["samples"] - request["t_token_s"][-1] * 1e4) <= 2 and made["process_samples"] == made["clock_lines"], made["kind"]
    assert set(watched["requests"][0]["clock"]) == {"samples", "minimum_mhz", "median_mhz", "mean_mhz", "maximum_mhz", "share_below_declared_lock"}
    assert {"readings", "requests", "other_compute_processes", "clocks", "rules"} <= set(watched), "counts and extremes: no sample itself is in the file"
    _says("area", "`observation` (every reading, the samples of every request behind the checked one as counts and extremes, the two verdicts of the `GPU`")
    _says("area", "it reads the GPU's list of compute processes, its memory in use, its utilisation and its SM clock, and the host's load.")
    gpu = lines.index("GPU         stated device, compute capability 10.0, 148 SMs | other compute processes: none seen")
    assert lines[gpu + 1] == "            clocks observed: yes (SM 2000 MHz median, 2000 min; no limit reported) | host load 5.5 on 70 cores"
    assert [made["host_load"] for made in watched["readings"]] == [[4.5, 4.0, 3.5], [5.5, 5.0, 4.5]]
    assert watched["host"] == {"load": 5.5, "cores": 70}, "the load over the last minute, of the reading behind the last timed request"
    environment = outcome.data["environment"]
    assert environment["driver"] == {"value": "000.00", "reason": None}
    assert list(environment["host"]["load"]) == ["before the warm-up request", "behind the last timed request"]
    entry = outcome.data["statistics"]["gpu_and_host"]
    assert (entry["definition"], entry["unit"], entry["value"]) == (perf.DEFINITIONS["gpu_and_host"][1], "MHz, W, MiB", None)
    assert entry["reason"] == "see the section observation of this file" and entry["serving_counterpart"].startswith("The same observations as a serving")
    # two readings of two queries each, and nothing else: the tool was asked, never told
    assert len(_calls(folder)) == 4 and all(call["argv"][1].startswith("--query-") and len(call["argv"]) == 3 for call in _calls(folder))
    text = (tmp_path / "report.json").read_text(encoding="utf-8")
    assert perf.screen(json.loads(text)) == perf.screen(outcome.data) and str(folder) not in text + "\n".join(lines), "the file's data, and no path"
    # perf.run without an observer of the caller's makes its own: no reading before the load, and the file says which there were
    assert "before the load" not in environment["host"]["load"]
    _says("area", "the process that times starts it in front of the warm-up request, makes no call that concerns it inside a request,")
    _says(
        "area",
        "and reads its files behind the traced request.",
        "The mode asks the driver through its `nvidia-smi` tool, in child processes, and adds no package.",
    )
    _says("area", "Where the run is at rest -- before the load, before the warm-up request and behind the last timed request -- it reads the GPU's list")
    _says("guide", "The GPU is observed through the driver's `nvidia-smi` tool in child processes, with no package added, and never by")
    _says("guide", "a call of the timing process inside a request: the sampler is started in front of the warm-up request and its files are read behind")
    _says("guide", "its files are read behind the traced request, and its loops end with the pipe the run holds:")
    _says("script", "The report's ``GPU`` lines say what the run observed through the driver's ``nvidia-smi`` tool, in child processes --")


def _wait(condition, limit=30.0):
    """Wait, bounded, for something a child process of the test does."""
    end = time.monotonic() + limit
    while not condition():
        assert time.monotonic() < end, "a child process of the test did not get there in time"
        time.sleep(0.01)


def _running(pid):
    """Whether a process is still at work: one that has ended and waits to be collected is not."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as file:
            return file.read().rpartition(") ")[2][:1] != "Z"
    except OSError:  # it went meanwhile -- or this host has no /proc, and then it is there
        return not os.path.exists("/proc/self/stat")


def test_the_sampler_is_a_child_process_whose_loops_end_with_its_pipe(monkeypatch, tmp_path):
    """The sampler: one child, the keeper, and under it two loops of the tool that write a line every period into files
    the timing process does not read before the stop.  The stop ends both loops and reads the files -- a line that is no
    sample is counted -- and removes them.  A run that is gone without a stop, or a keeper that is ended from outside,
    leaves no loop behind; nor does a tool that does not end when it is asked to, or -- on Linux -- a keeper that is
    killed: no loop outlives its keeper."""
    from model_samples import observe

    folder = _tool(monkeypatch, tmp_path, junk=[["not a line of the tool"]])
    tool = str(folder / "nvidia-smi")

    def loops():
        return [call for call in _calls(folder) if call["argv"][-1].startswith("--loop-ms=")]

    def begun(sampler, count):
        """The sampler started and both of its loops writing, the ``count``-th and the one before it: their process ids."""
        said = sampler.start()
        assert said == ""
        _wait(lambda: len(loops()) == count)
        files = [Path(sampler.folder.name) / name for name in ("clocks.csv", "processes.csv")]
        _wait(lambda: all(file.exists() and file.stat().st_size for file in files))
        return [call["pid"] for call in loops()][-2:]

    sampler = observe.Sampler(tool, "GPU-00000000-stand-in", 5)
    started = sampler.start()
    assert started == "" and sampler.child.poll() is None and sampler.child.stdout is None and sampler.child.stderr is None
    files = [Path(sampler.folder.name) / name for name in ("clocks.csv", "processes.csv")]
    _wait(lambda: all(file.exists() and file.read_text(encoding="utf-8").count("\n") >= 8 for file in files))  # the test looks; the sampler does not
    assert sampler.clocks == [] and sampler.processes == [] and len(loops()) == 2
    queries = ["--query-compute-apps=timestamp,pid,used_memory", "--query-gpu=timestamp,clocks.sm,power.draw,clocks_event_reasons.active"]
    expected = [["--id=GPU-00000000-stand-in", query, "--format=csv,noheader,nounits", "--loop-ms=5"] for query in queries]
    assert sorted(call["argv"] for call in loops()) == expected, "the two loops start side by side: either may write its line first"
    running = [call["pid"] for call in loops()]
    assert all(_running(pid) for pid in running)
    sampler.stop()
    assert not any(_running(pid) for pid in running) and not files[0].parent.exists(), "both loops ended, the files read and removed"
    assert _asked(folder) == set(running), "the keeper asked both loops to end and killed neither"
    assert len(sampler.clocks) >= 4 and {sample[1:] for sample in sampler.clocks} == {(2000.0, 300.0, 0)}
    assert len(sampler.processes) >= 4 and {sample[1:] for sample in sampler.processes} == {(os.getpid(), 400.0)}
    assert sampler.unread >= 8 and sampler.started[0] - 0.002 <= sampler.clocks[0].at <= sampler.clocks[-1].at <= sampler.stopped[0] + 0.5
    apart = (sampler.stopped[0] - sampler.stopped[1]) - (sampler.started[0] - sampler.started[1])
    assert abs(apart) * 1e3 < observe.DRIFT_LIMIT_MS, "the wall clock against the stamps' clock at the start and at the stop: nobody set a clock meanwhile"
    sampler.stop()  # a second stop finds nothing to do
    # the run is gone without a stop: its end of the pipe closes, and the keeper ends the loops
    gone = observe.Sampler(tool, "GPU-00000000-stand-in", 5)
    left = begun(gone, 4)  # both loops writing: a loop that is asked to end before it has its handler ends by the default action
    gone.child.stdin.close()
    _wait(lambda: not any(_running(pid) for pid in left))
    ended_with = gone.child.wait(timeout=30)
    assert ended_with == 0 and set(left) <= _asked(folder), "the keeper ended in order, and so did its loops"
    gone.folder.cleanup()
    # the keeper is ended from outside: the loops end with it, and a stop behind it reads what there is
    ended = observe.Sampler(tool, "GPU-00000000-stand-in", 5)
    last = begun(ended, 6)
    ended.child.terminate()
    _wait(lambda: not any(_running(pid) for pid in last))
    assert set(last) <= _asked(folder), "the keeper took the signal and ended its loops in order"
    ended.stop()
    assert ended.child is None and ended.unread >= 0
    # a tool that does not end when it is asked to, and that nothing else ends: the keeper kills it within its limit, and the stop leaves no loop
    assert 2 * observe.LOOP_LIMIT_S < observe.ASK_LIMIT_S, "the keeper is through with both loops before the stop gives up on it"
    _scenario(folder, deaf=True, unbound=True)
    with monkeypatch.context() as patch:
        patch.setattr(observe, "LOOP_LIMIT_S", 0.3)
        deaf = observe.Sampler(tool, "GPU-00000000-stand-in", 5)
        unasked = begun(deaf, 8)
        deaf.stop()
    assert not any(_running(pid) for pid in unasked) and not set(unasked) & _asked(folder), "killed by the keeper"
    assert len(deaf.clocks) >= 1 and len(deaf.processes) >= 1
    if sys.platform.startswith("linux"):  # a loop is bound to its keeper: none outlives a keeper that is killed, with a stop behind it or without
        _scenario(folder)
        killed = observe.Sampler(tool, "GPU-00000000-stand-in", 5)
        orphans = begun(killed, 10)
        os.kill(killed.child.pid, signal.SIGKILL)
        _wait(lambda: not any(_running(pid) for pid in orphans), 10.0)
        assert not set(orphans) & _asked(folder), "ended by the kernel with their keeper: a loop that takes no request to end goes too"
        killed.stop()
        assert killed.child is None and len(killed.clocks) >= 1
        _scenario(folder, deaf=True)  # ... also where the stop itself gives up on a keeper that is still waiting for a loop, and kills it
        with monkeypatch.context() as patch:
            patch.setattr(observe, "ASK_LIMIT_S", 0.3)
            patch.setattr(observe, "LOOP_LIMIT_S", 60.0)
            given_up = observe.Sampler(tool, "GPU-00000000-stand-in", 5)
            deserted = begun(given_up, 12)
            given_up.stop()
        _wait(lambda: not any(_running(pid) for pid in deserted), 10.0)
    _scenario(folder)
    for missing in (None, ""):  # no interpreter to run the keeper with (an embedded one names none): said, not raised
        with monkeypatch.context() as patch:
            patch.setattr(sys, "executable", missing)
            refused = observe.Sampler(tool, "GPU-00000000-stand-in").start()
        assert refused == "the sampler could not be started"
    with monkeypatch.context() as patch:  # an interpreter that cannot be started: said, and the folder of the files is not left behind
        patch.setattr(sys, "executable", str(tmp_path / "no interpreter"))
        unstarted = observe.Sampler(tool, "GPU-00000000-stand-in")
        refused = unstarted.start()
    assert refused == "the sampler could not be started" and not Path(unstarted.folder.name).exists()

    def no_folder(*_, **__):
        raise FileNotFoundError(2, f"No usable temporary directory found in ['{tmp_path}']")

    with monkeypatch.context() as patch:  # no folder for temporary files on this host: said, and nothing of the error's text
        patch.setattr(observe.tempfile, "TemporaryDirectory", no_folder)
        refused = observe.Sampler(tool, "GPU-00000000-stand-in").start()
    assert refused == "the sampler could not be started"
    _says("area", "a child process samples the SM clock, the power, the limit reasons the driver reports and the process list, twenty times a second,")
    assert 1000 / observe.PERIOD_MS == 20
    _says("guide", "and its loops end with the pipe the run holds: at the stop, or when the run is gone.")
    # no package is added: the module imports the standard library, the helper layer's clock and, where it asks for the GPU's id, PyTorch
    tree = ast.parse(Path(observe.__file__).read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {(node.module or "") if isinstance(node, ast.ImportFrom) else node.names[0].name for node in imports}
    standard = {"__future__", "math", "os", "shutil", "statistics", "subprocess", "sys", "tempfile", "time", "dataclasses", "typing"}
    assert names == standard | {"timing", "torch"}, names


def test_not_observed_names_no_path(monkeypatch, tmp_path):
    """A tool that fails, hangs, answers something else or is not there: the reason is one of a closed list of phrases.
    Neither the screen nor the file holds the tool's folder, its own words or an exception's text."""
    from model_samples import observe

    assert observe.Observer(torch.device("cpu")).reason == "no CUDA device", "a session that is not on a GPU"
    assert observe.Observer(torch.device("cpu")).in_a_request() == "none (no CUDA device)"
    with monkeypatch.context() as patch:  # the GPU is named to the tool by its UUID, with the prefix the tool takes
        patch.setattr(torch.cuda, "get_device_properties", lambda device: types.SimpleNamespace(uuid="0f9a1c2e-5b7d-4e6f-8a90-123456789abc"))
        named = observe.target_of(torch.device("cuda", 0))
    assert named == "GPU-0f9a1c2e-5b7d-4e6f-8a90-123456789abc"
    words = f"NVIDIA-SMI has failed because it could not open {tmp_path}/libnvidia-ml.so.1\n"
    folder = _tool(monkeypatch, tmp_path, exit=6, stderr=words)
    tool, pid = str(folder / "nvidia-smi"), os.getpid()
    failed = observe.reading(tool, "GPU-00000000-stand-in", pid, "before the load")
    assert failed.reason == "nvidia-smi ended with an error" and failed.data() == {
        "when": "before the load",
        "reason": "nvidia-smi ended with an error",
        "host_load": failed.load,
        "seconds": failed.seconds,
    }
    for scenario, reason in (
        ({"device": ["000.00", "2000"]}, "nvidia-smi's answer could not be read"),
        ({"processes": [["a name", 4]]}, "nvidia-smi's answer could not be read"),
        ({"device": []}, "nvidia-smi's answer could not be read"),
    ):
        _scenario(folder, **scenario)
        unread = observe.reading(tool, "GPU-00000000-stand-in", pid, "before the load")
        assert unread.reason == reason, scenario
    # fields the driver does not give, or gives as no size: the reading is made, and holds none of them
    _scenario(folder, device=["[N/A]", "0", "inf", "nan"])
    bare = observe.reading(tool, "GPU-00000000-stand-in", pid, "before the load")
    assert (bare.reason, bare.driver, bare.clock, bare.used, bare.utilization, bare.listed, bare.own) == ("", None, None, None, None, True, 400.0)
    asking = observe.Observer(torch.device("cuda", 0))
    unasked = asking.driver()
    asking.readings = [bare]
    assert unasked == {"value": None, "reason": "not observed (no reading was made)"}
    assert asking.driver() == {"value": None, "reason": "not observed (nvidia-smi's answer could not be read)"}
    with monkeypatch.context() as patch:
        patch.setattr(observe, "ASK_LIMIT_S", 0.5)
        _scenario(folder, hang=60)
        silent = observe.reading(tool, "GPU-00000000-stand-in", pid, "before the load")
    absent_tool = observe.reading(str(folder / "not there"), "GPU-00000000-stand-in", pid, "before the load")
    assert silent.reason == "nvidia-smi gave no answer" and absent_tool.reason == "nvidia-smi ended with an error"
    # a run whose tool fails at every call, with the real sampler: both lines say why, by the list
    outcome, lines, _, run = _observed(monkeypatch, tmp_path / "run", dense=False, scenario={"exit": 6, "stderr": words})
    gpu = lines.index("GPU         stated device, compute capability 10.0, 148 SMs")
    assert lines[gpu + 1] == "            other compute processes: not observed (nvidia-smi ended with an error)"
    assert lines[gpu + 2].startswith("            clocks observed: no (no clock sample inside a timed request) | host load ")
    assert outcome.passed and outcome.data["record"]["reasons"][-2:] == ["other processes not observed", "clocks not observed"]
    watched = outcome.data["observation"]
    assert outcome.data["environment"]["driver"] == {"value": None, "reason": "not observed (nvidia-smi ended with an error)"}
    assert {made["reason"] for made in watched["readings"]} == {"nvidia-smi ended with an error"}
    assert watched["sampler_period_ms"] == observe.PERIOD_MS == 50, "the period the run's loops were started with"
    assert outcome.data["protocol"]["in_a_timed_request"]["sampler"] == observe.SAMPLER, "the keeper was started: its loops gave nothing"
    asked = [call for call in _calls(run) if not call["argv"][-1].startswith("--loop-ms=")]
    assert len(asked) == 2, "one call a reading: a tool that failed is not asked for its list too (the two loops of the sampler start beside the run)"
    printed = "\n".join(lines) + json.dumps(outcome.data)
    assert str(tmp_path) not in printed and "libnvidia" not in printed and "NVIDIA-SMI" not in printed and "Traceback" not in printed
    assert watched["other_compute_processes"]["reason"] in observe.REASONS and watched["clocks"]["reason"] in observe.REASONS
    # the tool is not there at all: nothing is started, and the report says so
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "which", lambda name, *more, **named: None)
        absent = observe.Observer(torch.device("cuda", 0), 2000)
        absent.read("before the load")
        absent.start()
        absent.stop()
        missing = absent.section([("timed", 0.0, 1.0)])
    assert (absent.reason, absent.sampler, missing["source"], missing["reason"], missing["sampler_period_ms"]) == (
        "nvidia-smi not found",
        None,
        "none",
        "nvidia-smi not found",
        None,
    )
    assert (
        observe.others_text(missing) == "not observed (nvidia-smi not found)"
        and observe.clocks_text(missing) == "no (nvidia-smi not found; lock 2000 MHz declared)"
    )
    assert (
        absent.driver() == {"value": None, "reason": "not observed (nvidia-smi not found)"} and missing["readings"][0]["host_load"] == absent.readings[0].load
    )
    # what the file's protocol says where no sampler ran: none, with the reason -- the tool is not there, the session is not on a GPU, the
    # sampler was not asked for yet, its child could not be started
    assert absent.in_a_request() == "none (nvidia-smi not found)"
    with monkeypatch.context() as patch:
        patch.setattr(sys, "executable", "")
        early = asking.in_a_request()
        asking.start()
    assert (early, asking.failed, asking.in_a_request()) == ("none (not started)", observe.NO_SAMPLER, "none (the sampler could not be started)")
    _says("area", "its stop, or none, with the reason), `checks` (every")
    with monkeypatch.context() as patch:  # a PyTorch that names no id for the GPU
        patch.setattr(observe, "target_of", lambda device: device.uuid)
        assert observe.Observer(torch.device("cuda", 0)).reason == "PyTorch names no id for this GPU"
    assert len(set(observe.REASONS)) == len(observe.REASONS) == 15 and not any(os.sep in reason or "\\" in reason for reason in observe.REASONS)
    _says("area", "A `not observed` names its reason, one of a closed list, and never a path;")
    _says("guide", "A `not observed` carries a reason from a closed list, never the tool's path or its own words;")


def test_clock_lock_is_a_declaration(monkeypatch, capsys, tmp_path):
    """``--perf-clock-lock`` declares a clock that was fixed outside the sample: a flag of the mode, an integer from 1.  The
    run prints it in its header, on the clock line and -- where a sample lies below or above it -- as a reason of line
    2, and writes it into the file; it starts the driver's tool with queries and with nothing else, and the word
    'locked' is in no line of such a run.  The reading before the load is made by ``main()``, before anything is loaded."""
    folder = _tool(monkeypatch, tmp_path)
    with monkeypatch.context() as patch:  # up to the load, which is an error here: what is refused, and what is read before it
        sample = _sample_on_the_cpu(patch, [], loads=False)
        for arguments, refusal in (
            (["--perf-clock-lock", "2000"], "--perf-clock-lock: a flag of the performance mode; it needs --perf"),
            (["--perf", "--perf-clock-lock", "0"], "--perf-clock-lock takes the declared SM clock in MHz, a positive integer"),
        ):
            with pytest.raises(SystemExit) as stop:
                sample.main(arguments)
            assert stop.value.code == 2 and capsys.readouterr().err.splitlines()[-1].split(": error: ", 1)[1] == refusal
        assert sample._arguments(["--perf"]).perf_clock_lock is None and sample._arguments(["--perf", "--perf-clock-lock", "2000"]).perf_clock_lock == 2000
        assert _calls(folder) == [], "a refusal asks nothing of the driver"
        with pytest.raises(AssertionError, match="reached a load"):
            sample.main(["--perf", "--perf-steps", "8"])
        plain = capsys.readouterr().err.splitlines()
        assert "[00] clock not locked: timings are for orientation only | reference precision fp64" in plain, "without the flag: the header as it was"
        assert [call["argv"][1] for call in _calls(folder)] == [
            "--query-gpu=driver_version,clocks.sm,memory.used,utilization.gpu",
            "--query-compute-apps=pid,used_memory",
        ], "the reading before the load: made before the load was reached, two queries and no loop"
        with pytest.raises(AssertionError, match="reached a load"):
            sample.main(["--perf", "--perf-steps", "8", "--perf-clock-lock", "2600"])
        declared = capsys.readouterr().err.splitlines()
        assert "[00] clock lock declared: 2600 MHz, set outside this sample, which sets no clock | reference precision fp64" in declared
        assert not any("locked" in line for line in declared)
    # a whole run with a lock declared above the clock it observes
    _dense(monkeypatch, [])
    loaded = _sample_on_the_cpu(monkeypatch, [])
    monkeypatch.chdir(tmp_path)
    code = loaded.main(["--perf", "--perf-steps", "6", "--perf-requests", "2", "--perf-warmup", "2", "--perf-clock-lock", "2600", "--perf-report", "lock.json"])
    err = capsys.readouterr().err
    lines = err.splitlines()
    assert code == 0 and lines[-1] == "[00] PASS", err[-3000:]
    clocks = lines.index("            clocks observed: yes (SM 2000 MHz median, 2000 min; lock 2600 MHz declared; no limit reported)")
    assert lines[clocks + 1].startswith("            host load ") and lines[clocks - 1].endswith("| other compute processes: none seen")
    report = lines.index("cuDNN Frontend model sample: decode performance report")
    assert " ".join(line.strip() for line in lines[report + 1 : report + 4]).split(" scope ")[0].endswith("launched from Python; below the declared lock")
    text = (tmp_path / "lock.json").read_text(encoding="utf-8")
    data = json.loads(text)
    assert data["observation"]["clock_lock_declared_mhz"] == 2600 and data["record"]["reasons"][-1] == "below the declared lock"
    assert [made["when"] for made in data["observation"]["readings"]] == ["before the load", "before the warm-up request", "behind the last timed request"]
    assert list(data["environment"]["host"]["load"]) == ["before the load", "before the warm-up request", "behind the last timed request"]
    assert all(made["clock"]["share_below_declared_lock"] == 1.0 for made in data["observation"]["requests"])
    assert "locked" not in err and "locked" not in text, "neither the run nor its file"
    for call in _calls(folder):  # the tool is asked and never told: a GPU, a query, the format and, for the sampler, a period
        assert call["argv"][0] == "--id=GPU-00000000-stand-in" and call["argv"][1].startswith("--query-") and call["argv"][2] == "--format=csv,noheader,nounits"
        assert all(word.startswith("--loop-ms=") for word in call["argv"][3:])
    _says("family", "| `--perf-clock-lock MHZ` | none | the SM clock in MHz that an operator fixed outside the sample: a declaration.")
    _says("family", "The report prints it beside the clock it observed and names it as a reason where a sample lies more than 3 % below or above it;")
    _says("family", "the sample never sets a clock |")
    _says("area", "`--perf-clock-lock MHZ` declares a clock that an operator fixed outside the sample: the declaration is written into the report and held")
    _says("area", "against the clock the run observed, and nothing is set.")
    _says("area", "*Clock*: a sample never sets the GPU's clock; the performance mode observes it, and an operator may declare a lock applied outside.")
    _says("area", "# An operator who fixed the GPU's clock outside the sample declares it; the sample sets nothing:")
    _says("area", "--model /data/Qwen3.8-27B-NVFP4 --stack native --perf --perf-clock-lock <MHz>")
    _says("script", "``--perf-clock-lock MHZ`` declares a clock that an operator fixed outside this sample, which sets none: the report")
    _says("script", "holds the declaration against the clock it observed.")
    _says("script", "the other compute processes on the GPU, its SM clock, the limit reasons -- or that it could not, with the reason.")
    _says("script", "with --perf: the SM clock in MHz that an operator fixed outside this sample: a declaration, written into the report and held")
    _says("script", "a declaration, written into the report and held against the clock the run observes. The sample never sets a clock")


def test_every_form_of_the_gpu_lines_within_110():
    """No line of the screen is wider than 110 characters in any form of the two ``GPU`` lines: every reason of a ``not
    observed``, a lock, every limit the driver names, a long device name, a large host."""
    from model_samples import observe, perf

    def screen(gpu="stated device", **changes):
        form = _checkpoint_form(environment={"gpu": {"name": gpu, "compute_capability": "10.0", "sm_count": 148}})
        form["observation"] = {**_alone(), **changes}
        lines = perf.screen(form)
        assert max(len(line) for line in lines) <= perf.WIDTH == 110, max(lines, key=len)
        start = next(index for index, line in enumerate(lines) if line.startswith("GPU         "))
        return lines[start : next(index for index, line in enumerate(lines) if line.startswith("kernels     "))]

    alone = _alone()
    assert screen() == [
        "GPU         stated device, compute capability 10.0, 148 SMs | other compute processes: none seen",
        "            clocks observed: yes (SM 2000 MHz median, 2000 min; no limit reported) | host load 5.5 on 70 cores",
    ]
    assert len(screen()[1]) == 110, "one line at the screen's width"
    _says("area", "(a checkpoint on the stack `native`, on a GPU the run had to itself;")
    assert (
        screen(host={"load": None, "cores": 4})[1]
        == "            clocks observed: yes (SM 2000 MHz median, 2000 min; no limit reported) | host load not observed"
    )
    seen = {"verdict": "seen", "seen": 12, "reason": None}
    assert screen(other_compute_processes=seen)[0].endswith("148 SMs | other compute processes: 12 seen")
    for reason in observe.REASONS:  # every reason, in both lines, with a lock declared
        hidden = {"verdict": "not observed", "seen": None, "reason": reason}
        lines = screen(other_compute_processes=hidden, clocks={**alone["clocks"], "verdict": "no", "reason": reason}, clock_lock_declared_mhz=2000)
        assert f"other compute processes: not observed ({reason})" in " ".join(line.strip() for line in lines[:2])
        assert f"clocks observed: no ({reason}; lock 2000 MHz declared)" in " ".join(line.strip() for line in lines)
    every = [*observe.EVENTS[1:], "bit 12"]
    limited = screen(clocks={**alone["clocks"], "limit_reasons": every}, clock_lock_declared_mhz=2000, host={"load": 123.4, "cores": 256})
    assert " ".join(line.strip() for line in limited[1:]) == (
        "clocks observed: yes (SM 2000 MHz median, 2000 min; lock 2000 MHz declared; limit reported: " + ", ".join(every) + ") host load 123.4 on 256 cores"
    ), "a part that does not fit its line is wrapped by its words; the host's load stands on a line of its own"
    long_name = screen("a device with a name of " + "many " * 12 + "words")
    assert long_name[-2:] == [
        "            other compute processes: none seen",
        "            " + f"clocks observed: {observe.clocks_text(alone)} | {observe.host_text(alone)}",
    ]
    thin = screen(clocks={**alone["clocks"], "verdict": "thin", "samples": 3})
    assert thin[1].startswith("            clocks observed: thin (3 samples in a timed request) | host load ")


def test_an_ordinary_run_asks_no_driver(monkeypatch, capsys, tmp_path):
    """An ordinary run does not import the module that observes the GPU, starts the driver's tool not once, and prints the
    header it printed."""
    folder = _tool(monkeypatch, tmp_path)
    sample = _sample_on_the_cpu(monkeypatch, [])
    package = importlib.import_module("model_samples")
    for name in ("perf", "observe"):  # other tests of this module imported them: neither is there before the run
        monkeypatch.delitem(sys.modules, f"model_samples.{name}", raising=False)
        monkeypatch.delattr(package, name, raising=False)
    monkeypatch.chdir(tmp_path)
    code = sample.main(["--sequences", "1", "--force-tokens", "4"])
    lines = capsys.readouterr().err.splitlines()
    assert code == 0 and lines[-1] == "[00] PASS" and _calls(folder) == []
    assert "model_samples.observe" not in sys.modules and "model_samples.perf" not in sys.modules and not hasattr(package, "observe")
    assert "[00] clock not locked: timings are for orientation only | reference precision fp64" in lines
    assert not any("other compute processes" in line or "clocks observed" in line or "host load" in line for line in lines)
