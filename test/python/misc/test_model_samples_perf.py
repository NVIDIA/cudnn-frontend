# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the performance mode of the model samples (``samples/models/model_samples/perf.py``; ``--perf`` of
``samples/models/qwen_dense/00_decode.py``).

* every statistic on constructed time stamps: the time per output token, the rate, the nearest-rank percentiles, the run's
  value as the median over its timed requests, the time to first token, and the step and the loop's host time at the
  traced positions;
* the judged positions from the request and the page of the attention cache; the reference checks at judged positions
  through ``tie_out.run_checks``; the record class from its conditions;
* the protocol through a real ``Session`` on the CPU with stand-in providers and a stand-in profiler: no profiler pass
  before the last timed request, nothing but the step inside a timed request, the three statements of the ``checks``
  lines, and what a statement that does not hold does to the rate line and to the last line;
* the screen (no line wider than 110 characters, in every form) and the report file (strict JSON, a definition beside
  every statistic, no local path);
* the sample's flags, each refusal before the load, ``main()`` with ``--perf`` to its last line, and an ordinary run
  that prints what it printed;
* the public text of the mode: the words, the ``stage`` row, the rules of the guide, the flag table and the example of
  the report, each against the code.

No GPU, no weights on disk, no profiler."""

import importlib
import importlib.util
import json
import os
import re
import sys
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
    missing = perf._statistics([request], 0, 4, "not printed for a reduced model")["decode_rate"]
    assert missing["value"] is None and missing["total"] is None and missing["reason"] == "not printed for a reduced model"


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
    assert four["host"]["maximum"] == pytest.approx(5.0), "the fourth step of the first request was launched 5 ms behind the read before it"
    every = perf._statistics(timed, 0, 1, "")
    assert every["step"]["n"] == 8 and every["step"]["value"] == every["token_to_token_interval"]["value"] == pytest.approx(10.0)
    assert three["step"]["events"] == "t_token[i] - t_token[i - 1] at the traced positions"
    assert "gpu_busy" not in three and "launch_overhead" not in three and "tracing_stretch" not in three, "not statistics of this version"
    assert "does not take GPU busy from the traced request" in three["step"]["definition"]


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


def _session(monkeypatch, events, sequences=1, capacity=40, **stand_in):
    """A loaded session of the tiny geometry on the CPU through the rows of ``fe-cublas``, on the stand-ins of
    :func:`_stand_ins`: ``(session, text)``.  Every decode step of it writes what it ran with into ``events``:
    ``("step", rows of the record, whether a tap is set, whether a call log is open)``."""
    from model_samples import Session, TextIO, blocks

    build = _stand_ins(monkeypatch, events, **stand_in)
    model = _tiny_model()
    text = TextIO.synthetic(model.geometry.vocab)
    session = Session(model, text, sequences=sequences, stack="fe-cublas", device="cpu", capacity=capacity, check_kernels=True)
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 0, 0)
    options = {"reference_precision": "fp64", "device": "cpu", "check_kernels": True}
    session.plan = blocks.plan(model.provider_table, model.regimes, facts, sequences, 1, "fe-cublas", model=model, options=options, factories=build)
    session.load()
    advance = session._advance

    def watched(ids, *more, **named):
        events.append(("step", int(session.record.inputs.shape[0]), session.tap is not None, session._call_log is not None))
        return advance(ids, *more, **named)

    monkeypatch.setattr(session, "_advance", watched)
    return session, text


class _Sink:
    """Where the text of sequence 0 would go: it keeps what it is handed."""

    def __init__(self):
        self.written = []

    def reported(self, text=""):
        return "goes to stdout after this report"

    def write(self, text):
        self.written.append(text)


def _mode(monkeypatch, tmp_path, steps=6, requests=2, warmup=2, sequences=1, budget=160_000, real=False, on_say=None, eos=None, **stand_in):
    """One run of the mode on the stand-ins: ``(outcome, the lines it said, events, sink)``.  ``events`` holds, in order,
    every stamp the mode took (``"stamp"``), every decode step (``_session``), every line (``("say", line)``), every
    decoder made for a text (``"decoder"``) and every profiler pass (``"profiler"``).  ``on_say(line)`` is called with
    every line: a test switches a defect on at the line that says which phase comes next.  ``real`` runs the text path of
    a checkpoint (the text written once, to the sink), with ``eos`` as the text's end tokens."""
    from model_samples import Environment, blocks, perf

    events, lines, sink = [], [], _Sink()
    session, text = _session(monkeypatch, events, sequences, capacity=5 + steps + 2, **stand_in)
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
    assert (lines[at] + " " + lines[at + 1].strip()).endswith(" ms at the 3 traced positions, 5 to 7; GPU busy not measured (this version)")
    one, lines, events, _ = _mode(monkeypatch, tmp_path / "one", budget=1)  # a budget below one step: the sizing step is the traced request
    assert one.data["traced"]["steps"] == 1 and events.count("profiler") == 1 and one.passed


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
    }
    assert all(request["end_state_sha256"] == outcome.data["raw"]["requests"][0]["end_state_sha256"] for request in outcome.data["raw"]["requests"][2:4])


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
    assert (
        _line(lines, "text out ") == f"text out              sequence 0: the text up to the first end token (token {at} of 7), goes to stdout after this report"
    )
    assert len(sink.written) == 1 and sink.written[0].endswith("\n") and events.count("decoder") == 1
    assert outcome.data["text_path"]["tokens_written"] == at and outcome.data["text_path"]["cut_at_an_end_token"] is True
    assert lines.index(_line(lines, "text out ")) > lines.index(_line(lines, "report file ")), "behind the report"
    assert events.index("decoder") > max(index for index, event in enumerate(events) if event == "profiler"), "and behind every request"
    outcome, lines, events, whole = _mode(monkeypatch, tmp_path / "whole", real=True, eos=[271])  # an id the run does not produce
    assert 271 not in produced and _line(lines, "text out ").startswith(
        "text out              sequence 0: the text up to the first end token (no end token among the 7), "
    )
    assert len(whole.written) == 1 and len(whole.written[0]) >= len(sink.written[0]) and outcome.data["text_path"]["tokens_written"] == 7


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


def test_the_kernels_lines_are_three_numbers_from_the_plan_and_the_census(monkeypatch):
    """Statistic 16 on the family's own tables: the 64-layer model on the stack ``native``, compute capability 10.0 with
    148 SMs, planned on stated facts -- how many launches of a step are on the target kernel set, by provider and as
    three numbers, and the block calls on fallback rows."""
    from model_samples import Session, TextIO, blocks, census, perf
    from model_samples.qwen_dense import QwenDense

    monkeypatch.setattr(census, "installed_versions", lambda packages: {"torch": "2.14.0+cu130", "nvidia-cutlass-dsl": "4.8.0"})
    model = QwenDense()
    session = Session(model, TextIO.synthetic(272), stack="native", device="cpu", capacity=8)
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)
    session.plan = blocks.plan(model.provider_table, model.regimes, facts, 1, 1, "native", model=model, options={"device": "cpu"})
    kernels = perf.kernel_facts(session, "PASS")
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
    with sections replaced as in :func:`_data`."""
    base = {
        "model": {"name": "Qwen3.8-27B-NVFP4", "layers": 64, "reduced": False},
        "record": {"reasons": ["fallback rows present", "kernel identity not claimed on this GPU class", "launched from Python"]},
        "traced": {"steps": 51, "first_position": 58, "last_position": 108, "gpu_busy_reason": "this version"},
    }
    for section, values in changes.items():
        base[section] = {**base[section], **values} if isinstance(values, dict) and section in base else values
    data = _data(**base)
    data["statistics"]["decode_rate"].update(value=9.9, reason=None, total=9.9)
    data["statistics"]["step"]["value"] = 100.0
    data["protocol"]["prompt"]["tokens"] = 58
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
    assert rate.startswith("decode rate (orientation)   9.9  tokens/s per sequence   (") and " ms per token; 2 requests within " in rate
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
    controlled = perf.screen(_checkpoint_form(record={"class": "controlled", "reasons": []}))
    assert controlled[1] == "record class          controlled run in one process; a number of record is the median of 3 such runs"
    assert _line(controlled, "decode rate").startswith("decode rate           9.9  tokens/s per sequence   (")
    several = perf.screen(_checkpoint_form(model={"name": "Qwen3.8-27B-NVFP4", "layers": 64, "reduced": False, "sequences": 4}))
    at = several.index("decode rate (orientation)   9.9  tokens/s per sequence (9.9 in all)")  # the total beside the rate: the bracket moves down
    assert several[at + 1].startswith(" " * 28 + "(") and several[at + 1].endswith(" %)") and "| 4 sequences | greedy" in _line(several, "model       ")


def test_the_step_line_says_what_was_not_measured():
    """The ``step`` line names the positions it was taken at and says that GPU busy was not measured, with the reason: this
    version, or why no traced step was observed.  No line of the screen takes the step apart."""
    from model_samples import perf

    lines = perf.screen(_checkpoint_form())
    assert "step                  100.000  ms at the 51 traced positions, 58 to 108; GPU busy not measured (this version)" in lines
    assert _line(lines, "time to first token   ").endswith(" ms for a 58-token prompt (58 decode steps: this sample has no prefill kernel)")
    assert not any("launch overhead" in line or line.startswith("note ") for line in lines)
    one = perf.screen(_checkpoint_form(traced={"steps": 1, "last_position": 58}))
    assert "step                  100.000  ms at the traced position, 58; GPU busy not measured (this version)" in one
    why = "the profiler's trace could not be exported (No space left on device)"
    unmeasured = perf.screen(_checkpoint_form(traced={"steps": 0, "gpu_busy_reason": why}))
    at = unmeasured.index("step                  100.000  ms over every timed step; GPU busy not measured (the profiler's trace could not")
    assert unmeasured[at + 1] == "                      be exported (No space left on device))"
    memory = lines.index(_line(lines, "memory                "))
    assert (
        " GiB | state and cache " in lines[memory] and lines[memory + 1] == "                      peak reserved not observed  (GiB = 2^30 bytes)"
    ), "on the CPU"
    held = perf.screen(_checkpoint_form(memory={"weights_gib": 37.3, "state_gib": 0.5, "peak_reserved_gib": 40.0}))
    assert "memory                weights 37.3 GiB | state and cache 0.50 GiB | peak reserved 40.0 GiB  (GiB = 2^30 bytes)" in held
    assert len("memory                weights 37.3 GiB | state and cache 0.50 GiB | peak reserved 40.0 GiB  (GiB = 2^30 bytes)") == 110


def test_no_line_wider_than_110(monkeypatch, tmp_path):
    """No line of the screen is wider than 110 characters, in every form a run of this version prints: the reduced model,
    a checkpoint, a target stack with its three numbers, each form of the ``step`` line, a failed statement, kernels that
    were not observed with a long reason, long names, six-digit values."""
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
        _checkpoint_form(traced={"steps": 0, "gpu_busy_reason": long_reason}),
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
    assert "decode rate           not printed: the timed requests did not compute the checked request's tokens" in perf.screen(forms[1])
    assert "decode rate           not printed: the kernels of the traced steps are not the plan's" in perf.screen(forms[2])
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path)  # and the lines a run printed are those of its data
    said = lines[lines.index("cuDNN Frontend model sample: decode performance report") :]
    assert said[: len(perf.screen(outcome.data))] == perf.screen(outcome.data) and 24 <= len(perf.screen(outcome.data)) <= 30


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
    sections = "schema kind written_utc report_file record model environment observation kernels protocol checks statistics raw traced memory startup text_path"
    assert list(data) == sections.split() and data["schema"] == "model-sample-performance/1" and data["kind"] == "run"
    assert "per_kernel" not in data and "fingerprints" not in data and "report_text" not in data, "of later versions"
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
    for key, (unit, definition, counterpart) in perf.DEFINITIONS.items():
        entry = data["statistics"][key]
        assert (entry["unit"], entry["definition"], entry["serving_counterpart"]) == (unit, definition, counterpart), key
        assert "value" in entry and (entry["value"] is not None or entry["reason"]), f"{key}: a value that does not exist is null with a reason"
    assert sorted(set(data["statistics"]) - set(perf.DEFINITIONS)) == ["tokens_per_second_wall"]
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
    assert perf.write_report(str(tmp_path / "report.json"), data) == "--perf-report report.json: File exists"
    assert perf.write_report(str(tmp_path / "nan.json"), {"value": float("nan")}).startswith("--perf-report nan.json: ")
    assert perf.write_report(str(tmp_path / "missing" / "r.json"), data) == "--perf-report r.json: No such file or directory"
    assert re.fullmatch(r"report-\d{8}T\d{6}Z\.json", perf.default_report_name())


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

    for name in ("tokens", "kernel", "unobserved", "traced", "state", "weights", "before"):
        (tmp_path / name).mkdir()
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path / "tokens", defect=defect, on_say=other_tokens)
    assert not outcome.passed and outcome.data["checks"]["timed_requests"] == "FAIL (request 1, step 0)"
    at = lines.index("            timed requests give the checked request's tokens and state: FAIL (request 1, step 0)")
    assert lines[at + 1] == "            kernels, traced steps: PASS", "each statement keeps its own verdict"
    assert "decode rate           not printed: the timed requests did not compute the checked request's tokens" in lines
    assert "a check did not hold" in outcome.data["record"]["reasons"] and outcome.data["checks"]["last_line"] == "[00] FAIL"
    assert (tmp_path / "tokens" / "report.json").exists() and len(outcome.data["raw"]["requests"]) == 5, "the stamps are kept for the diagnosis"
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

    # the traced request alone computes other tokens: it is held against the checked request too
    defect.clear()
    defect["at the first profiler pass"] = {"token": True}
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path / "traced", defect=defect)
    assert not outcome.passed and outcome.data["checks"]["timed_requests"] == "FAIL (traced request, step 1)"
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

    # a stored weight that no longer holds the bytes of its load behind the timed requests (the checks before found it whole, twice)
    asked = []

    def written_to(weights):
        asked.append(weights)
        return ["mlp.down of layer 0"] if len(asked) > 2 else []

    monkeypatch.setattr(perf.tie_out, "changed_weights", written_to)
    outcome, lines, events, _ = _mode(monkeypatch, tmp_path / "weights")
    assert not outcome.passed and outcome.data["checks"]["timed_requests"] == "FAIL (stored weights: mlp.down of layer 0)" and len(asked) == 3
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
        assert sample.main(arguments) == 1
        return capsys.readouterr().err.splitlines()[-1]

    (tmp_path / "there.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert error(["--perf", "--perf-report", "there.json"]) == "[00] ERROR --perf-report there.json: the file exists, and a report file is never written over"
    assert (tmp_path / "there.json").read_text() == "{}"
    assert error(["--perf", "--perf-report", ""]) == "[00] ERROR --perf-report: needs a file name; it was given an empty one"
    assert error(["--perf", "--perf-report", str(tmp_path / "missing" / "r.json")]) == "[00] ERROR --perf-report r.json: cannot be written there"
    late = error(["--perf", "--max-model-len", "40"])  # 26 prompt tokens + 257 forced new tokens do not fit 40 positions
    assert late.startswith("[00] ERROR the seeded prompt: ") and "--max-model-len" in late
    # the defaults of the mode: the protocol's counts, one sequence count, the kernel check on, the forced count of a request
    args = sample._arguments(["--perf"])
    assert (args.perf_warmup, args.perf_steps, args.perf_requests, args.perf_trace_records) == (32, 256, 3, 160_000)
    assert args.sequences == [1] and args.check_kernels and args.force_tokens == 257 and args.perf_report is None
    assert sample._arguments(["--perf", "--sequences", "4", "--perf-steps", "8"]).sequences == [4]
    with pytest.raises(AssertionError, match="reached a load"):  # nothing above reached one; the mode itself does, behind its line
        sample.main(["--perf", "--perf-steps", "8"])
    before = capsys.readouterr().err.splitlines()
    line = "[00] performance mode: at most 228 decode steps in all: one checked request, one warm-up request of 32 decode steps behind its first new token, "
    line += "3 timed requests of 8 decode steps each, one traced request of up to 8; a prompt of 26 tokens is 26 decode steps in front of each; "
    assert (line + "the checks replay the checked request once") in before, before[:8]
    assert before.index(line + "the checks replay the checked request once") < next(
        index for index, said in enumerate(before) if said.startswith("served by (")
    )
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
    assert lines[gpu + 1 : gpu + 3] == [
        "            other compute processes: not observed (this version)",
        "            clocks observed: no (this version does not observe the GPU)",
    ]
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
    # the same command again: a second file beside the first, or an ERROR that keeps the first (the name holds the second)
    code = sample.main(
        ["--perf", "--perf-steps", "6", "--perf-requests", "2", "--perf-warmup", "2", "--perf-report", "mine.json", "--dump-census", "census.json"]
    )
    again = capsys.readouterr().err.splitlines()
    assert code == 0 and again[-1] == "[00] PASS" and len(list(tmp_path.glob("*.json"))) == 3
    assert json.loads((tmp_path / "census.json").read_text())["traced_steps"] == 6, "the launch records of the traced request"
    # a report file that cannot be written after all: the run is done and keeps its verdict, and its last line is the ERROR
    perf = importlib.import_module("model_samples.perf")
    write = perf.write_report
    monkeypatch.setattr(perf, "write_report", lambda path, data: "--perf-report late.json: No space left on device")
    code = sample.main(["--perf", "--perf-steps", "6", "--perf-requests", "1", "--perf-warmup", "1", "--perf-report", "late.json"])
    monkeypatch.setattr(perf, "write_report", write)
    late = capsys.readouterr().err.splitlines()
    assert code == 1 and late[-1] == "[00] ERROR --perf-report late.json: No space left on device (the run itself: PASS)"
    # kernels that cannot be observed: the mode's report with the reason, and the last line of a run that could not look
    for name in ("mine.json", "census.json"):
        (tmp_path / name).rename(tmp_path / f"kept_{name}")
    blind = _sample_on_the_cpu(monkeypatch, [], profiler=NO_PROFILER)
    code = blind.main(["--perf", "--perf-steps", "6", "--perf-requests", "1", "--perf-warmup", "1", "--perf-report", "blind.json"])
    unseen = capsys.readouterr().err.splitlines()
    assert code == 0 and unseen[-1] == f"[00] RAN fe-cublas stack (kernels not observed: {NO_PROFILER})"
    assert f"            kernels, traced steps: NOT OBSERVED ({NO_PROFILER})" in unseen


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
    code = sample.main(["--check-kernels", "--sequences", "1", "--force-tokens", "4"])
    out, err = capsys.readouterr()
    lines = err.splitlines()
    assert code == 0 and lines[-1] == "[00] PASS" and out == "" and not list(tmp_path.iterdir())
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
    plain = tie_out.run_checks(session, prompt_tokens=5)
    assert plain.lines[0] == "checks (1 sequence, 8 steps: 5 prompt + 3 decode)" and plain.ok and plain.kernels == "PASS"
    assert " 8 calls " in next(line for line in plain.lines if line.startswith("    linear_attention.step "))
    with pytest.raises(ValueError, match="judged positions are steps of a decode run"):
        tie_out.run_checks(Session(_tiny_model(), TextIO.synthetic(272), stack="reference", device="cpu", draft_tokens=2), prompt_tokens=0, positions=[0])
    target = Session(_tiny_model(), TextIO.synthetic(272), stack="fe-entries", device="cpu")
    with pytest.raises(ValueError, match="its kernel check cannot be switched off"):
        tie_out.run_checks(target, prompt_tokens=0, kernels=False)
    assert tie_out.DEFERRED == "DEFERRED", "the word a caller hands in when it traces after the checks"


# ---------------------------------------------------------------------------------------------- the public text
def _flat(text):
    return " ".join(text.split())


def test_the_public_text_of_the_mode_says_what_the_code_does():
    """The READMEs, the guide and the script's own text against the code: the words the mode prints, the ``stage`` row
    against the stages a load names, the rule about the text, the rule of the mode, the flag table against the parser,
    and the README's example of the report against the rendered screen with every value replaced by dots."""
    from model_samples import perf

    area, guide = (MODELS / "README.md").read_text(encoding="utf-8"), (MODELS / "AGENTS.md").read_text(encoding="utf-8")
    family, script = (MODELS / "qwen_dense" / "README.md").read_text(encoding="utf-8"), SCRIPT.read_text(encoding="utf-8")
    # the words: each names something the mode prints or holds
    for word in ("*Timed request*", "*Judged position*", "*Token-to-token interval*", "*Record class*"):
        assert word in _flat(area), word
    # the stage row names every stage a load of the family adds to the line
    row = next(line for line in area.splitlines() if line.startswith("| `stage` |"))
    loader = (MODELS / "model_samples" / "qwen_dense" / "model.py").read_text(encoding="utf-8")
    for stage in re.findall(r'stages\.stage\("([a-z ]+)"\)', loader):
        assert stage in row, f"the stage row does not name {stage!r}"
    assert "lay out" in row and "checksum" in row and "merge" in row
    # rule 17: where the text is written; rule 19: what the mode states
    assert "or once, after the report, with `--no-stream` and always in the performance mode" in _flat(guide)
    rule = _flat(guide.split("19. **The performance mode states what it observed, never what it assumes.**", 1)[1].split("\n## ", 1)[0])
    for sentence in (
        "One definition per statistic, written in the report file, with a host test of its arithmetic on constructed time stamps.",
        "Nothing but the step runs inside a timed request: no recorder, no call log, no profiler, no text; and no profiler pass runs in the process before its last timed request.",
        "a run in which one of the three does not hold prints no rate.",
        "the word 'locked' is never printed.",
        "A report holds no speed-up, no baseline and no figure of a second GPU class.",
    ):
        assert sentence in rule, sentence
    assert "locked" not in " ".join(perf.screen(_checkpoint_form())) and "locked" not in json.dumps(_data())
    # the flag table of the family README: every flag of the mode, with the parser's default
    sample_flags = dict(re.findall(r"^\| `(--perf[a-z-]*)(?: [A-Z]+)?` \| ([^|]*) \|", family, flags=re.MULTILINE))
    assert set(sample_flags) == {"--perf", "--perf-warmup", "--perf-steps", "--perf-requests", "--perf-report", "--perf-trace-records"}
    assert (
        sample_flags["--perf-warmup"].strip() == "`32`" and sample_flags["--perf-steps"].strip() == "`256`" and sample_flags["--perf-requests"].strip() == "`3`"
    )
    assert sample_flags["--perf-trace-records"].strip() == "`160000`" and "report-<UTC time>.json" in sample_flags["--perf-report"]
    protocol = perf.Protocol()
    assert (protocol.warmup, protocol.steps, protocol.requests, protocol.trace_records) == (32, 256, 3, 160000)
    for flag in sample_flags:
        assert f'"{flag}"' in script and f"``{flag}" in script.split('"""', 2)[1], f"{flag}: a flag of the parser, named in the script's docstring"
    assert "--perf-show" not in family + area + script and "--perf-overwrite" not in family + area + script, "flags of a later version"
    # the section of the area README: its example is the screen, every value a run measures or counts replaced by dots
    section = area.split("## The performance mode\n", 1)[1].split("\n## ", 1)[0]
    example = section.split("```text\n", 1)[1].split("```", 1)[0].splitlines()
    target = {"stated": True, "declared": True, "on_target": 322, "launches": 484, "code": 49, "port": 273, "stand_ins": 0}
    target.update(by_provider={"cuDNN Frontend": 321, "PyTorch": 1}, fallback_calls=162, kernels_per_step=2908, copies_per_step=200)
    today = [
        "fallback rows present",
        "kernel identity not claimed on this GPU class",
        "launched from Python",
        "other processes not observed",
        "clocks not observed",
    ]
    form = _checkpoint_form(kernels={**target, "absent": 1}, checks={"positions_text": "0-15, 63, 64, 312, 313"}, record={"reasons": today})
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
