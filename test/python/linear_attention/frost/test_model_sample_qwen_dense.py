# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke of the dense Qwen model sample (``samples/models/qwen_dense/00_decode.py``): the script is loaded by path and its
``main()`` is called as a user's default run -- the reduced seeded model, 1 and 4 sequences, the sample's own checks against its
fp64-accumulated reference (replay, every provider block teacher-forced, the greedy tokens, each sequence alone against the
same sequence in the batch).  The run must print ``[00] PASS`` and the environment line; on a GPU, a library version or a
free-memory budget the sample does not serve it prints ``[00] SKIP <requirement>`` and the cell is reported as skipped with
that sentence -- unless a check failed before it: a skip never covers a failure.  Every line of the run is read from
standard error, and standard output must stay empty: it is kept for generated text, which a seeded run does not have.
Each sequence count also reports its
traced pass (GPU busy, idle and launches per step from kernel activity records, or the reason there are none) and the
bytes of a step.  The performance mode of the same seeded model (``--perf``) runs on both stacks in a fresh interpreter:
its verdict, its report in the reduced-model form and its report file are held, none of its times.  The
verdict line, the stack, the versions and the wall time are recorded through ``frost_routing.measured``, so the result -- or
the skip -- reaches the job log of a passing run.  L0 runs the two-layer model (one linear-attention + one attention layer), L1
the four-layer one (three + one).

The ``fe-entries`` cell runs the same default run on the stack ``fe-entries``: every cuDNN Frontend row of the stack ``native``
(the fused linear-attention step in its wait + release form; on compute capability 10.x the attention gate + quantise
entry, the add + RMSNorm + quantise entry in the form of each call site and the fused 4-bit gate / up entry, each judged
against its twin) with every other block on its fallback row -- no package beyond cuDNN Frontend and PyTorch.
That stack always checks its kernels: the profiler's record of every launch of the traced steps is held against the plan,
the entries' own launch facts and, where the family's kernel census has a section for the GPU, the recorded launch shapes.
Where the profiler's records cannot be had the sample ends in ``[00] RAN fe-entries stack (kernels not observed: <reason>)``:
the cell then passes on the other checks and the recorded line says that the kernels were not observed -- uncovered
there, not verified.

The ``native`` cell runs the default run on the stack ``native`` itself, which has three rows more: the multiplies the target
runs on a library, FlashInfer, on the stored weights.  FlashInfer is an optional package of that stack alone, so the cell runs
where it is installed at a validated version and is skipped elsewhere -- by this module where the package is absent, by the
sample's own ``[00] SKIP`` line where its version is not validated.  A skipped cell is uncovered, not passing.

That cell runs the sample in a FRESH INTERPRETER, as a user does.  The kernel check compares the profiler's kernel names
with the symbols the entries state, and the profiler names a kernel wrongly after a kernel module was unloaded in the same
process: a kernel loaded where an unloaded one had been was recorded under the unloaded kernel's name (two raw CUDA
kernels loaded and unloaded in turn: about every second launch; none where every module stayed loaded; PyTorch 2.13.0).
The kernel that runs is the right one; its record's name is not.  A sample process unloads nothing, a test session does:
the cells of this module drop the sample's modules, and with them the entries of the stack that ran before.  Run in this
process after the default cell, the kernel check failed one run in three by the name of the fused step alone.

The verify cells run the step that feeds several tokens per sequence at once (``Session(tokens=T)``, ``Session.verify``): no
script reaches it, so a small program drives the helper layer in a fresh interpreter -- a decode session and a verify
session of the stack ``fe-entries`` on one load of the reduced seeded model; the decode run's token stream through verify
steps that keep every possible number of tokens once, against the decode steps (``tie_out.verify_checks``); then the
verify session's own checks with its kernel check (``tie_out.run_checks``).  The fused verify entry is judged, token by
token, against reference decode steps; its launch against the entry's own facts and, at one sequence of six tokens on a
GPU the census has a section for, against the recorded launch shape.  The out-projection of the linear-attention layer is
the FP8 projection entry's in such a step where a row of the stack holds the step's rows (compute capability 10.0: 2 to
16 rows; 10.3 and 10.7: 2 to 32), in the split of that row range, and is judged against the reference multiply under the
name of its call site; the attention layer's out-projection, and both beyond those rows, are the multiply over a copy.

The speculative cells run ``samples/models/qwen_dense/02_speculative_decode.py``: one request through the prompt's decode
steps and then steps of speculative decoding -- the seeded drafter's passes on reference blocks, one verify step, one host
read --, with the drafted tokens of every step scheduled, held line by line against decode steps of the same tokens and
then by the checks of every run.  The L0 cell is its default run, loaded by path and run in this process, with both
streams held; it launches two forms no other L0 cell launches -- the fused verify step and the fused decode step on token
0 of a state of six tokens per sequence, both in the form of the default stack -- and records what their first launches
took.  The L1 cell is the same run on the stack ``fe-entries`` in a fresh interpreter: its kernel check holds whole steps,
the lookups of the drafter's passes and the verify plan's rows.

The paged-cache cell runs no script: no provider row asks for the paged attention cache of a state yet
(``state.PagedKvCache``), so no run builds one.  A small program in a fresh interpreter runs the two reference chains of an
attention layer on a plain cache and on a paged one at the published widths, across the first page boundary, and holds
the bytes of the two against each other, the pool against index arithmetic of its own, the state's cos / sin table against
the chain's function on the device, and the kernels the chains launch on the pool to their families.  It runs where the
family serves the GPU and on compute capability 8.0, and is skipped by name on any other class; a host case holds that
condition on stated compute capabilities.

Every line of a run is read from standard error, which is not the sample's alone: a library may print there first, and
behind the run's last line.  A cell that reads the header of a run finds it by the script's tag (``_header``), and
one that reads the line in front of the verdict counts from the verdict (``_in_front_of_the_verdict``); host cases
hold both, and the cells of the speculative script read every run between two lines that are not the sample's."""

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cudnn")

import frost_routing  # noqa: E402

MODELS = Path(__file__).resolve().parents[4] / "samples" / "models"
SCRIPT = MODELS / "qwen_dense" / "00_decode.py"
SPECULATIVE = MODELS / "qwen_dense" / "02_speculative_decode.py"


def _loaded(monkeypatch, script):
    """A sample script as a module; its ``sys.path`` entries, every ``model_samples`` module and the default it sets for the
    profiler's log level leave with the test."""
    monkeypatch.setenv("KINETO_LOG_LEVEL", os.environ.get("KINETO_LOG_LEVEL", "6"))
    paths = list(sys.path)
    spec = importlib.util.spec_from_file_location("model_sample_qwen_dense_" + script.stem, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path[:] = paths
    for name in [n for n in sys.modules if n in ("model_samples", "serving_options") or n.startswith("model_samples.")]:
        del sys.modules[name]


@pytest.fixture()
def sample(monkeypatch):
    """The plain script, ``00_decode.py``."""
    yield from _loaded(monkeypatch, SCRIPT)


@pytest.fixture()
def speculative(monkeypatch):
    """The speculative script, ``02_speculative_decode.py``."""
    yield from _loaded(monkeypatch, SPECULATIVE)


FE_CUBLAS = "by provider: cuDNN FE 1 | fallbacks 8 (cuBLAS over copies 4, PyTorch chains 4) | PyTorch 2"
#: with the gate entry, the norm entry and the fused gate / up entry (compute capability 10.x) three more blocks are cuDNN Frontend's:
#: two reference chains and one multiply over a copy fewer are fallbacks
FE_ENTRIES = "by provider: cuDNN FE 4 | fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2) | PyTorch 2"
#: on compute capability 10.0 the in-projection is the FP8 projection entry's as well; the out-projection stays a multiply over a copy
FE_ENTRIES_CC100 = "by provider: cuDNN FE 5 | fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2) | PyTorch 2"
#: the stack native on compute capability 10.x: the three multiplies FlashInfer serves are no fallbacks, and no lane of this
#: repository runs them
NATIVE = "by provider: cuDNN FE 4 | FlashInfer 3 | fallbacks 3 (cuBLAS over copies 1, PyTorch chains 2) | PyTorch 2 | unverified in CI: 3"
NATIVE_CC100 = "by provider: cuDNN FE 5 | FlashInfer 3 | fallbacks 2 (PyTorch chains 2) | PyTorch 2 | unverified in CI: 3"


CHILD_SECONDS = 1200  # the fresh interpreter's deadline: imports, the compiles of both sequence counts and the run, far above what they take


def _fresh_interpreter(arguments, script=SCRIPT, tag="[00]"):
    """A sample script as a user runs it: in a fresh interpreter, which imports the ``cudnn`` package this process imported.
    Returns ``(exit code, the run's lines, what the run wrote to standard output)``: a sample writes every line of a run
    to standard error and keeps standard output for generated text.  The child arms its own stack dump before the
    parent's deadline; where the deadline passes, the failure carries what the child had printed."""
    import cudnn

    package_root = str(Path(cudnn.__file__).resolve().parent.parent)
    child = (
        "import faulthandler, runpy, sys\n"
        f"faulthandler.dump_traceback_later({CHILD_SECONDS - 60}, exit=True)\n"
        f"sys.path.insert(0, {package_root!r})\n"
        "import cudnn\n"
        f"assert cudnn.__file__ == {cudnn.__file__!r}, f'the child imported another cudnn: {{cudnn.__file__}}'\n"
        f"sys.argv = [{str(script)!r}, *{list(arguments)!r}]\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
    )
    try:
        done = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=CHILD_SECONDS)
    except subprocess.TimeoutExpired as late:
        shown = [part.decode(errors="replace") if isinstance(part, bytes) else (part or "") for part in (late.stdout, late.stderr)]
        pytest.fail(f"the sample did not end within {CHILD_SECONDS} s in a fresh interpreter\n--- stdout\n{shown[0][-6000:]}\n--- stderr\n{shown[1][-6000:]}")
    if f"{tag} " not in done.stderr:  # the child never reached the sample's own lines: an import or an environment failure
        pytest.fail(
            f"the sample printed no verdict in a fresh interpreter (exit code {done.returncode})\n--- stdout\n{done.stdout[-3000:]}\n--- stderr\n{done.stderr[-6000:]}"
        )
    return done.returncode, done.stderr, done.stdout


def _run(sample, capsys, arguments, stack="fe-cublas", by_provider=FE_CUBLAS, unobserved=False):
    """Run the sample's ``main()`` and hold its output; with ``sample`` None, run the script in a fresh interpreter instead.
    ``unobserved`` accepts, beside ``[00] PASS``, the verdict of a run whose kernel check could not look (``RAN <stack> stack
    (kernels not observed: ...)``, exit code 0): the line is recorded as it is, so a CI job without the profiler's records
    shows as such in its log."""
    start = time.perf_counter()
    if sample is None:
        code, out, written = _fresh_interpreter(arguments)
    else:
        code = sample.main(arguments)
        captured = capsys.readouterr()
        out, written = captured.err, captured.out  # the run's lines; and standard output, which is for generated text
    seconds = time.perf_counter() - start
    # A seeded run has no language: whatever its verdict, standard output stays empty.  A line there -- of the sample, or of
    # a library the stack calls (a kernel build, a log switched to standard output) -- would stand in front of a text that
    # a user pipes into a file.
    assert written == "", f"a seeded run writes nothing to standard output; it holds\n{written[-3000:]}"
    tagged = [line for line in out.splitlines() if line.startswith("[00] ")]
    versions = next((line[5:] for line in tagged if line.startswith("[00] cuDNN Frontend ")), "no environment line")
    device = next((line[5:].split(" | ")[0] for line in tagged if " SMs)" in line), "no device line")
    verdict = tagged[-1] if tagged else "no output"
    key = "model sample qwen_dense 00_decode " + (" ".join(arguments) or "(default)")
    frost_routing.measured(key, f"{verdict}; stack {stack}; {device}; {versions}; {seconds:.0f} s")
    if verdict.startswith("[00] SKIP"):
        assert " FAIL" not in out and "[00] ERROR" not in out, out[-6000:]  # a skip never covers a failed check
        pytest.skip(verdict)
    held = verdict == "[00] PASS" or (unobserved and verdict.startswith(f"[00] RAN {stack} stack (kernels not observed: "))
    assert code == 0 and held, out[-6000:]
    assert "distribution" in versions and "source hash" in versions, "the environment check ran and its line was printed"
    assert by_provider in out
    for check in ("replay", "blocks", "tokens"):
        assert out.count(f"  {check} ") == 2 and f"  {check} " in out, f"the {check} check ran at both sequence counts"
    assert "linear_attention.step equal (gated)" in out or "REPORTED" in out, "batch invariance was checked at 4 sequences"
    assert out.count("\ntraced pass ") == 2 and out.count("\nbytes per step ") == 2, "each count reports its traced pass (or why there is none) and its bytes"
    assert " FAIL" not in out
    return out


@pytest.mark.L0
def test_default_run_passes_its_own_checks(sample, capsys):
    out = _run(sample, capsys, [])
    assert "2 of 64 layers (1 linear-attention + 1 attention)" in out and "20 block calls per step" in out


@pytest.mark.L0
def test_default_run_on_the_fe_entries_stack_passes_its_own_checks_and_its_kernel_check(capsys):
    """The cuDNN Frontend rows of the stack ``native``, on real launches: the fused step in its wait + release form and, on
    compute capability 10.x, the gate + quantise entry, the add + RMSNorm + quantise entry (four of its forms on the
    two-layer model, each call served by the instance of its call site) and the fused 4-bit gate / up entry (the kernel
    stated whole, in the token tile of the row count) inside their bands against their twins; on compute capability 10.0
    the FP8 projection entry for the in-projections (one instance per weight shape) inside the band of the reference
    multiply, and on every other class that block on its multiply over a copy; the kernel check on every launch of the
    traced steps; the census line, which claims nothing while a row is a fallback.
    Where an entry that multiplies on the stored format serves both sequence counts -- the fused gate / up entry on
    compute capability 10.x, the projection entry on 10.0 -- the run holds those weights on their stored format alone, so
    the reference of the checks makes its own bf16 copies: the run says that the stored bytes they are made from are those
    of the load (``stored weights``) and what the copies cost (``on demand``), and PyTorch's peak stays below the memory
    sentence -- one copy at a time, where all of them together are many times that sentence.  On compute capability 12.x
    every weight has its copy and neither line is printed.
    In a fresh interpreter: the kernel check reads kernel names from the profiler, which names them wrongly in a process
    that has unloaded a kernel module (the module docstring)."""
    if not torch.cuda.is_available():  # decided here, before a child is started
        pytest.skip("needs a CUDA GPU")
    gate = torch.cuda.get_device_capability()[0] == 10  # the gate entry and the norm entry serve compute capability 10.x
    projection = tuple(torch.cuda.get_device_capability()) == (10, 0)  # the class the row of the FP8 projection entry claims
    by_provider = FE_ENTRIES_CC100 if projection else (FE_ENTRIES if gate else FE_CUBLAS)
    out = _run(None, capsys, ["--stack", "fe-entries"], "fe-entries", by_provider, unobserved=True)
    assert "2 of 64 layers (1 linear-attention + 1 attention)" in out and "stack fe-entries" in out
    assert "cudnn.GdnDecodeFused [PDL True, PDL_RELEASE True]" in out, "the fused step in the target's form"
    assert out.count("\nkernels, planned (decided before the load") == 2 and out.count("\n  kernels  ") == 2, "the kernel check ran at both sequence counts"
    assert out.count(": NOT CLAIMED (") == 2 and ": EQUAL" not in out, "nothing is claimed while a row of the stack is a fallback"
    if gate:
        assert "cudnn.AttentionGateQuantDecode [THREADS 256, PDL_RELEASE True]" in out
        assert out.count("[band: gate_fp8]  PASS") == 2, "the gate entry was judged against its twin at both sequence counts"
        assert "attention.gate_quantize equal (gated)" in out or "REPORTED" in out
        assert "cudnn.AddNormQuantDecode [PDL True, PDL_RELEASE True] + one of 6 forms, by call site" in out, "the norm entry, a form per call site"
        assert out.count("[band: add_norm_quant]  PASS") == 2, "the norm entry was judged against its twin at both sequence counts"
        assert "add_norm_quantize equal (gated)" in out or "REPORTED" in out
        served = [line for line in out.splitlines() if line.strip().startswith("add_norm_quantize, as the call log of a traced step has it (5 calls): ")]
        assert len(served) == 2 and all(
            line.count(" x cudnn.AddNormQuantDecode #") == 4 for line in served
        ), "five calls of a step on four forms, each from its instance"
        fused = "cudnn.SwigluFp4Decode [TOKEN_TILE 8, PREWAIT_STAGES 2, L2_EVICT_FIRST True, MAX_CLUSTERS 148, PDL True]"
        assert fused in out, "the fused gate / up entry, the kernel stated whole: 1 and 4 rows run the 8-token tile"
        assert out.count("[band: swiglu_fp4]  PASS") == 2, "judged against its twin at both sequence counts"
        assert "mlp.gate_up equal (gated)" in out or "REPORTED" in out
        stated = [line for line in out.splitlines() if line.strip().startswith(f"mlp.gate_up: {fused}, 2 launches per step") and "; the entry states " in line]
        assert len(stated) == 2 and all("block (192, 1, 1), shared memory 231,424 B" in line for line in stated), "one launch per MLP, at both sequence counts"
        # the weights the run keeps on their stored format alone: the gate / up weights, and on 10.0 the in-projections too
        kept, count = ("projection.fp8.in, mlp.gate_up", 4) if projection else ("mlp.gate_up", 2)
        made = [line for line in out.splitlines() if line.startswith("  on demand ")]
        assert len(made) == 2 and all(
            f"weights this run keeps on their stored format alone ({kept})" in line for line in made
        ), "no bf16 copy of these weights is held: the reference of the checks made its own, at both sequence counts"
        assert (
            "GiB of it for the checks (the reference's bf16 copy of one weight at a time, where the run keeps none)" in out
        ), "and the memory sentence says so"
        held = [line for line in out.splitlines() if line.startswith("  stored weights ")]
        assert len(held) == 2 and all(
            line.startswith(f"  stored weights    {count} of {count} weights this run keeps on their stored format alone hold the bytes of their load, ")
            and line.endswith("    PASS")
            for line in held
        ), "the copies were made from the bytes of the load: no block wrote into its weight, at both sequence counts"
        # one copy at a time: the copies of a count are tens of GiB in all, and what PyTorch held at its peak is below the sentence
        need = float(out.split("this stack needs about ", 1)[1].split(" GiB", 1)[0])
        peak = float(out.split("GPU memory: PyTorch peak ", 1)[1].split(" GiB allocated", 1)[0])
        assert 0.0 < peak < need, f"PyTorch held {peak} GiB at its peak; the memory sentence states {need} GiB"
    else:
        assert "\n  on demand " not in out and " for the checks (" not in out, "every weight has its bf16 copy: nothing is made on demand"
        assert "\n  stored weights " not in out, "and the reference reads the copies of the load: no stored bytes to hold"
        assert "cudnn.SwigluFp4Decode" not in out, "the rows claim compute capability 10.x only"
        assert any(line.split()[:1] == ["mlp.gate_up"] and line.endswith("fallback: bf16 copy of 4-bit weights") for line in out.splitlines())
    multiply = [line for line in out.splitlines() if line.split()[:1] == ["projection.fp8.in"] and "calls" in line and "[band: " in line]
    assert len(multiply) == 2 and all(line.endswith("[band: bf16_rounding]  PASS") for line in multiply), "the in-projection was judged at both sequence counts"
    if projection:
        form = "[SPLIT_K 1, MMA_M 128, STAGES 8, TMEM_COLS 32, PREWAIT_WEIGHTS True, HOLD_TMEM True, PDL True, PDL_RELEASE True]"
        assert f"cudnn.Fp8ProjectionDecode {form} + one of 2 forms, by call site" in out, "the FP8 projection entry, the kernel stated whole"
        assert "projection.fp8.in equal (gated)" in out or "REPORTED" in out
        served = [line for line in out.splitlines() if line.strip().startswith("projection.fp8.in, as the call log of a traced step has it (2 calls): ")]
        assert len(served) == 2 and all(line.count(" x cudnn.Fp8ProjectionDecode #") == 2 for line in served), "one instance per weight shape"
        stated = [line for line in out.splitlines() if line.strip().startswith("projection.fp8.in at ") and "; the entry states " in line]
        assert len(stated) == 4 and all("block (256, 1, 1), shared memory" in line for line in stated), "both shapes, at both sequence counts"
    else:
        assert "cudnn.Fp8ProjectionDecode" not in out, "the row claims compute capability 10.0 only"
        assert any(line.split()[:1] == ["projection.fp8.in"] and line.endswith("fallback: bf16 copy of 8-bit weights") for line in out.splitlines())
    observed = [line for line in out.splitlines() if line.startswith("  kernels  ")]
    frost_routing.measured("model sample qwen_dense 00_decode --stack fe-entries: the kernel check", "; ".join(line.split("    ")[-1] for line in observed))


@pytest.mark.L0
@pytest.mark.parametrize("stack", ["fe-cublas", "fe-entries"])
def test_the_performance_mode_of_the_seeded_run_reports_and_writes_its_file(stack, tmp_path):
    """``--perf`` on the reduced seeded model, on the two stacks the tests of this repository run, in a fresh interpreter
    (the kernel check reads kernel names from the profiler: the module docstring): the checked request and its reference
    checks, a warm-up request, two timed requests, the traced request and the kernel check on it.  The run ends in ``[00]
    PASS`` -- or, where the profiler's records cannot be had, in ``RAN <stack> stack (kernels not observed: ...)`` --
    prints the report in its reduced-model form (no rate; ``orientation`` with ``reduced model`` among its reasons) on
    standard error, leaves standard output empty and writes the report file.  Where the kernels were observed the file
    holds GPU busy of every traced step and the per-kernel table of real launches: its rows hold every record of a
    step, an exact row the launches the plan has for it, and the rows' mean time per step sums to the mean GPU busy;
    and the step taken apart: one remainder for every timed step at a traced position, whose mean adds up with the
    means of the step, of GPU busy and of the host time, and is labelled an estimate.
    What it observed of its GPU through the driver's tool -- or that it could not -- stands on the ``GPU`` lines and
    among the reasons, whatever a host shows.
    No time of it is held against anything."""
    import json

    if not torch.cuda.is_available():  # decided here, before a child is started
        pytest.skip("needs a CUDA GPU")
    report = tmp_path / "report.json"
    arguments = ["--stack", stack, "--perf", "--perf-steps", "16", "--perf-requests", "2", "--perf-warmup", "4", "--perf-report", str(report)]
    start = time.perf_counter()
    code, out, written = _fresh_interpreter(arguments)
    tagged = [line for line in out.splitlines() if line.startswith("[00] ")]
    verdict = tagged[-1] if tagged else "no output"
    frost_routing.measured(f"model sample qwen_dense 00_decode --stack {stack} --perf", f"{verdict}; {time.perf_counter() - start:.0f} s")
    if verdict.startswith("[00] SKIP"):
        assert " FAIL" not in out and "[00] ERROR" not in out, out[-6000:]  # a skip never covers a failed check
        pytest.skip(verdict)
    unobserved = verdict.startswith(f"[00] RAN {stack} stack (kernels not observed: ")
    assert code == 0 and (verdict == "[00] PASS" or unobserved) and " FAIL" not in out, out[-8000:]
    assert written == "", f"a seeded run writes nothing to standard output; it holds\n{written[-3000:]}"
    lines = out.splitlines()
    screen = lines[lines.index("cuDNN Frontend model sample: decode performance report") :]
    screen = screen[: next(index for index, line in enumerate(screen) if line.startswith("[00] "))]  # the report and its text out line
    assert max(len(line) for line in screen) <= 110, max(screen, key=len)
    assert any(line.startswith("report file           report.json") for line in screen) and any(line.startswith("text out  ") for line in screen)
    assert screen[1].startswith("record class          orientation, not a number of record: reduced model; ")
    assert "decode rate           not printed for a reduced model" in screen and not any("tokens/s" in line for line in screen)
    assert any(line.startswith("checks      blocks against the reference at positions 0-15, ") and line.endswith(": PASS") for line in screen)
    said = " ".join(line.strip() for line in screen)
    assert "timed requests give the checked request's tokens and state: PASS" in said
    assert ("kernels, traced steps: NOT OBSERVED (" if unobserved else "kernels, traced steps: PASS") in said
    assert "protocol    warm-up 4 steps | timed 16 steps x 2, end token ignored | launched from Python, not captured" in screen
    assert any(line.startswith("[00] performance mode: at most ") for line in tagged), "what the mode will run, before the load"

    def refuse(constant):
        raise AssertionError(f"the report file holds {constant}")

    data = json.loads(report.read_text(encoding="utf-8"), parse_constant=refuse)
    assert data["schema"] == "model-sample-performance/3" and data["record"]["class"] == "orientation" and data["checks"]["last_line"] == verdict
    assert data["protocol"]["stack"] == stack and str(tmp_path) not in report.read_text(encoding="utf-8"), "no path in the file"
    requests = data["raw"]["requests"]
    assert [request["kind"] for request in requests] == ["checked", "warm-up", "timed", "timed", "traced"]
    assert requests[2]["ids_sha256"] == requests[3]["ids_sha256"] == requests[0]["ids_sha256"], "every timed request: the checked request's ids"
    assert requests[2]["end_state_sha256"] == requests[3]["end_state_sha256"] == requests[0]["end_state_sha256"], "and its end state, byte for byte"
    assert data["protocol"]["profiler_first_started"] == (None if unobserved else "behind the last timed request, in the traced request")
    # the record class: every reason, in the order of the conditions; a stack that declares a target kernel set names its identity
    identity = "no target kernel set declared" if stack == "fe-cublas" else "kernel identity not claimed on this GPU class"
    observed = ["kernels not observed"] if unobserved else []
    fixed = ["reduced model", "fallback rows present", identity, "launched from Python", *observed]
    # what the run observed of its GPU: the reasons behind the fixed ones are those of its two verdicts, which the screen prints
    watched = data["observation"]
    seen, clocks = watched["other_compute_processes"], watched["clocks"]
    others = {"none seen": [], "seen": ["shared GPU"], "not observed": ["other processes not observed"]}[seen["verdict"]]
    assert data["record"]["reasons"] == [*fixed, *others, *clocks["reasons"]] and data["record"]["performance_claim_eligible"] is False
    assert (clocks["verdict"] == "no") == (clocks["reasons"] == ["clocks not observed"]) and (seen["verdict"] == "seen") == bool(seen["seen"])
    assert any("other compute processes: " in line for line in screen) and any(f"clocks observed: {clocks['verdict']} (" in line for line in screen)
    assert "locked" not in " ".join(screen), "the report never prints the word"
    if watched["source"] == "nvidia-smi":  # the driver's tool was found: a reading before the load, before the warm-up request and behind the timed requests
        assert [made["when"] for made in watched["readings"]] == ["before the load", "before the warm-up request", "behind the last timed request"]
        assert [made["kind"] for made in watched["requests"]] == ["warm-up", "timed", "timed", "traced"]
    else:
        assert watched["reason"] and seen["verdict"] == "not observed" and clocks["verdict"] == "no"
    assert all(entry["definition"] and entry["unit"] for entry in data["statistics"].values()), "a definition and a unit beside every statistic"
    memory = data["memory"]  # the peak of the timed part is PyTorch's own, between the settle and the last timed request: inside the process's
    assert (
        0 < memory["peak_reserved_gib"] <= memory["process_peak_reserved_gib"] and memory["weights_gib"] + memory["state_gib"] <= memory["peak_allocated_gib"]
    )
    kernels = data["kernels"]
    if not unobserved:
        assert 1 <= data["traced"]["steps"] <= 16 and requests[4]["ids_sha256"] == requests[0]["ids_sha256"], "the traced request too"
        # the GPU side: GPU busy of every traced step, and the per-kernel table of the launches the kernel check attributed
        busy, rows = data["traced"]["per_step"]["gpu_busy_ms"], data["per_kernel"]["rows"]
        assert len(busy) == data["traced"]["steps"] and min(busy) > 0 and data["statistics"]["gpu_busy"]["value"] == data["traced"]["gpu_busy_ms"] > 0
        assert any(line.startswith("per kernel, per step (reduced model: traced request)") for line in screen) and rows[-1]["label"] == "host read"
        assert sum(row["launches_per_step"] for row in rows) == data["traced"]["records_of_the_sizing_step"], "the rows hold every record of a step"
        exact = [row for row in rows if row["relation"] == "exact"]
        assert exact and all(row["launches_per_step"] == row["planned_launches_per_step"] for row in exact), "as the plan has them"
        assert abs(sum(row["ms_per_step_mean"] for row in rows) - sum(busy) / len(busy)) <= 1e-9 * sum(busy), "the rows partition GPU busy"
        # the step taken apart: a remainder per timed step at a traced position; its mean and the means of the same steps add up
        step, host, overhead = (data["statistics"][key] for key in ("step", "host", "launch_overhead"))
        assert overhead["n"] == step["n"] == host["n"] == 2 * data["traced"]["steps"], "every timed step at a traced position"
        assert overhead["remainder_ms"] == overhead["mean"] and overhead["label"].startswith("an estimate") and overhead["value"] in (None, overhead["mean"])
        of_gpu = data["statistics"]["gpu_busy"]["mean"]
        assert abs(of_gpu - sum(busy) / len(busy)) <= 1e-9 * of_gpu, "the mean of GPU busy of the traced steps"
        assert abs(step["mean"] - (of_gpu + overhead["remainder_ms"] + host["mean"])) <= 1e-9 * step["mean"], "the terms of the step add up"
        assert 0 < data["traced"]["time_resolution_us"] <= 1 and data["per_kernel"]["us_per_launch_printed"] == (data["traced"]["time_resolution_us"] < 1)
    if kernels["stated"]:  # a target stack on a GPU the census has a section for: the three numbers add up
        assert kernels["on_target"] == kernels["code"] + kernels["port"] + kernels["stand_ins"] and kernels["launches"] == 20
    else:
        assert (stack == "fe-cublas") == (not kernels["declared"]), "fe-cublas declares no target kernel set; fe-entries does"


@pytest.mark.L0
def test_default_run_on_the_native_stack_with_its_library_rows(capsys):
    """The stack ``native`` with the rows a library serves, on real launches, where FlashInfer is installed: the MLP's down
    multiply and the output head on ``flashinfer.gemm.mm_fp4`` and the out-projections on ``flashinfer.gemm.bmm_fp8``, each
    on the stored weight and each inside the band of the reference multiply at both sequence counts; the first use of the
    library announced while the run is planned; the 4-bit multiply outside any tuning context (``TUNED False`` in the call
    log, or the kernel check fails); every weight of a multiply on its stored format alone.  On the GPU class the family's
    census has a section for, the down multiply is held to the recorded kernel class and launch shape, and the kernel the
    library chose for the out-projection is printed beside the recorded pick.  Skipped where the package is absent."""
    if not torch.cuda.is_available():  # decided here, before a child is started
        pytest.skip("needs a CUDA GPU")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the library rows of the stack native claim compute capability 10.x")
    if importlib.util.find_spec("flashinfer") is None:  # looked up, not imported: the package is the child's to import
        pytest.skip("the stack native calls FlashInfer, which is not installed here (samples/models/requirements-flashinfer.txt): uncovered, not passing")
    cc100 = tuple(torch.cuda.get_device_capability()) == (10, 0)
    out = _run(None, capsys, ["--stack", "native"], "native", NATIVE_CC100 if cc100 else NATIVE, unobserved=True)
    assert "stack native" in out and out.count("[00] planning: first use of FlashInfer ") == 1, "the library's first use is announced once, while the run plans"
    for block in ("projection.fp8.out", "mlp.down", "output_head"):
        judged = [line for line in out.splitlines() if line.split()[:1] == [block] and "calls" in line and "[band: " in line]
        assert len(judged) == 2 and all(line.endswith("[band: bf16_rounding]  PASS") for line in judged), f"{block} was judged at both sequence counts"
    down = [
        line for line in out.splitlines() if line.strip().startswith("mlp.down: flashinfer.gemm.mm_fp4 [BACKEND cute-dsl, TUNED False], 2 launches per step")
    ]
    assert len(down) == 2, "one launch per MLP at both sequence counts, outside any tuning context"
    kept, count = (
        ("projection.fp8.in, projection.fp8.out, mlp.gate_up, mlp.down, output_head", 9)
        if cc100
        else ("projection.fp8.out, mlp.gate_up, mlp.down, output_head", 7)
    )
    made = [line for line in out.splitlines() if line.startswith("  on demand ")]
    assert len(made) == 2 and all(
        f"weights this run keeps on their stored format alone ({kept})" in line for line in made
    ), "no bf16 copy of a multiply's weight"
    held = [line for line in out.splitlines() if line.startswith("  stored weights ")]
    assert len(held) == 2 and all(line.startswith(f"  stored weights    {count} of {count} weights") and line.endswith("    PASS") for line in held)
    if cc100 and torch.cuda.get_device_properties(0).multi_processor_count == 148:  # the class the census has a section for, at 1 sequence
        assert "17 of 20 launches of a step are target kernels (7 the target's own compiled kernels + 10 cuDNN Frontend entries with byte-equal results)" in out
        assert sum("in the kernel class Sm100BlockScaledSplitKGemmKernel and the launch shape the census records" in line for line in down) == 1
        assert out.count("chosen by the library for projection.fp8.out at ") == 2, "the library's pick at both call sites: printed and compared, never gated"
    observed = [line for line in out.splitlines() if line.startswith("  kernels  ")]
    frost_routing.measured("model sample qwen_dense 00_decode --stack native: the kernel check", "; ".join(line.split("    ")[-1] for line in observed))


@pytest.mark.L1
def test_four_layer_run_passes_its_own_checks(sample, capsys):
    out = _run(sample, capsys, ["--layers", "4"])
    assert "4 of 64 layers (3 linear-attention + 1 attention)" in out and "34 block calls per step" in out


VERIFY_PROGRAM = """
import faulthandler, sys
faulthandler.dump_traceback_later({deadline}, exit=True)
sys.path.insert(0, {package_root!r})
import cudnn
assert cudnn.__file__ == {cudnn_file!r}, f'the child imported another cudnn: {{cudnn.__file__}}'
import torch
sys.path.insert(0, {models!r})
import model_samples
from model_samples import census, tie_out
from model_samples.qwen_dense import QwenDense


def say(text):
    print(text, file=sys.stderr, flush=True)


model = QwenDense.reduced()
text = model_samples.TextIO.synthetic(model.geometry.vocab)
common = dict(stack="fe-entries", device="cuda:0", capacity=40)
decode = model_samples.Session(model, text, sequences={sequences}, **common)
verify = model_samples.Session(model, text, sequences={sequences}, tokens={tokens}, share=decode, **common)
try:
    say(decode.plan_table([verify]))
    say(verify.resolve().table())
    say(census.table(verify.plan, verify.claim()))
except NotImplementedError as error:
    say(f"[verify] SKIP {{error}}")
    raise SystemExit(0)
say(f"[verify] {{model_samples.check_environment().header()}}")
decode.load()
verify.load()
generator = torch.Generator().manual_seed(0)
ids = decode.prefill([torch.randint(0, model.geometry.vocab, (8,), generator=generator, dtype=torch.int32) for _ in range({sequences})])
for _ in range(23):
    ids = decode.step(ids)
against = tie_out.verify_checks(verify, decode)
say(against.text())
verify.trace(32)
own = tie_out.run_checks(verify, prompt_tokens=0)
say(own.text())
passed = against.ok and own.ok
if passed and own.unobserved:
    say(f"[verify] RAN fe-entries stack (kernels not observed: {{own.unobserved}})")
else:
    say("[verify] PASS" if passed else "[verify] FAIL")
raise SystemExit(0 if passed else 1)
"""


def _program(program, tag, key, what, unobserved="RAN fe-entries stack (kernels not observed: ", **fields):
    """A small program that drives the helper layer in a fresh interpreter (``program``, formatted with ``fields``): the
    run's lines.  ``tag`` starts its own lines (``[verify]``), ``key`` names the run in the routing record, ``what`` in a
    failure.  Skips where the plan declines the GPU; fails where a check failed or standard output is not empty.
    ``unobserved``: how the program's last line begins, behind its tag, where its kernels could not be observed."""
    import cudnn

    if not torch.cuda.is_available():  # decided here, before a child is started
        pytest.skip("needs a CUDA GPU")
    program = program.format(
        deadline=CHILD_SECONDS - 60,
        package_root=str(Path(cudnn.__file__).resolve().parent.parent),
        cudnn_file=cudnn.__file__,
        models=str(MODELS),
        **fields,
    )
    start = time.perf_counter()
    try:
        done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=CHILD_SECONDS)
    except subprocess.TimeoutExpired as late:
        shown = [part.decode(errors="replace") if isinstance(part, bytes) else (part or "") for part in (late.stdout, late.stderr)]
        pytest.fail(f"the {what} did not end within {CHILD_SECONDS} s in a fresh interpreter\n--- stdout\n{shown[0][-6000:]}\n--- stderr\n{shown[1][-6000:]}")
    out = done.stderr
    tagged = [line for line in out.splitlines() if line.startswith(f"{tag} ")]
    verdict = tagged[-1] if tagged else f"no verdict (exit code {done.returncode})"
    versions = next((line[len(tag) + 1 :] for line in tagged if line.startswith(f"{tag} cuDNN Frontend ")), "no environment line")
    frost_routing.measured(key, f"{verdict}; {versions}; {time.perf_counter() - start:.0f} s")
    assert done.stdout == "", f"a run of the helper layer writes nothing to standard output; it holds\n{done.stdout[-3000:]}"
    if verdict.startswith(f"{tag} SKIP"):
        assert " FAIL" not in out, out[-6000:]  # a skip never covers a failed check
        pytest.skip(verdict)
    held = verdict == f"{tag} PASS" or verdict.startswith(f"{tag} {unobserved}")
    assert done.returncode == 0 and held and " FAIL" not in out, out[-8000:]
    return out


def _verify(sequences, tokens):
    """A decode session and a verify session of ``tokens`` tokens per sequence in a fresh interpreter (``VERIFY_PROGRAM``):
    the run's lines."""
    key = f"model sample qwen_dense verify step, {sequences} x {tokens} tokens, stack fe-entries"
    return _program(VERIFY_PROGRAM, "[verify]", key, "verify run", sequences=sequences, tokens=tokens)


def _verify_asserts(out, sequences, tokens):
    several = f"{sequences} sequence{'s' if sequences > 1 else ''} x {tokens} tokens per sequence"
    assert f"stack fe-entries, {several}" in out, "the plan of a step of several tokens per sequence"
    assert f"cudnn.GdnVerifyFused [PDL True, PDL_RELEASE True, TOKENS {tokens}]" in out, "the fused verify entry in the target's form, its tokens a field of it"
    assert f"verify against decode ({several}, " in out
    rows = [line for line in out.splitlines() if line.startswith("  tokens  ")]
    assert len(rows) == 2 and all(line.endswith("    PASS") for line in rows), "against the decode steps, and against the reference stack's verify steps"
    kept = next(line for line in out.splitlines() if line.startswith("  kept state  "))
    assert kept.endswith(("    PASS", "    REPORTED")), kept
    judged = [line for line in out.splitlines() if line.split()[:1] == ["linear_attention.step"] and "[band: gdn_step]" in line]
    assert len(judged) == 1 and judged[0].endswith("[band: gdn_step]  PASS") and "conv history equal; null slot untouched" in judged[0]
    assert "byte-identical on a second run from a fresh state    PASS" in out
    assert (
        "cudnn.GdnVerifyFused states replay-identical bytes and no batch invariance between one sequence and several" in out
    ), "what the entry states is printed"
    # the out-projection: the FP8 projection entry at the linear-attention layer's call where a row of the stack holds the rows of
    # this step on this class -- the kernel of that row range, stated whole -- and the multiply over a copy at the attention
    # layer's call; beyond those rows, and on a class the rows do not claim, the multiply over a copy at both call sites
    cc, rows = tuple(torch.cuda.get_device_capability()), sequences * tokens
    one, other = "projection.fp8.out at linear_attention.out", "projection.fp8.out at full_attention.out"
    if (cc == (10, 0) and rows <= 16) or (cc in ((10, 3), (10, 7)) and rows <= 32):
        split, stages = (2, 11) if cc == (10, 0) and rows <= 8 else (4, 7)
        tile = 8 if rows <= 8 else (16 if rows <= 16 else 32)
        kernel = (
            f"[SPLIT_K {split}, MMA_M 64, STAGES {stages}, TOKEN_TILE {tile}, TMEM_COLS 32, PREWAIT_WEIGHTS True, HOLD_TMEM True, PDL True, PDL_RELEASE True"
        )
        assert f"cudnn.Fp8ProjectionDecode {kernel}, OUT_FEATURES 5120, IN_FEATURES 6144]" in out, "the kernel of the step's rows, stated whole"
        for site in (one, other):
            judged = [line for line in out.splitlines() if line.strip().startswith(site + " ") and "[band: " in line]
            assert len(judged) == 1 and judged[0].endswith("[band: bf16_rounding]  PASS"), f"{site}: judged under the name of its call site"
        assert any(line.strip().startswith(other + " ") and line.endswith("fallback: bf16 copy of 8-bit weights") for line in out.splitlines())
        served = [line for line in out.splitlines() if line.strip().startswith("projection.fp8.out, as the call log of a traced step has it (1 calls): ")]
        assert len(served) == 1 and served[0].count(" x cudnn.Fp8ProjectionDecode #") == 1, "one call of a step from the entry's instance"
    else:
        assert "projection.fp8.out at " not in out, "one row at both call sites"
        assert any(line.split()[:1] == ["projection.fp8.out"] and line.endswith("fallback: bf16 copy of 8-bit weights") for line in out.splitlines())
    observed = [line for line in out.splitlines() if line.startswith("  kernels  ")]
    frost_routing.measured(
        f"model sample qwen_dense verify step, {sequences} x {tokens} tokens: the kernel check", "; ".join(line.split("    ")[-1] for line in observed)
    )


@pytest.mark.L0
def test_a_verify_step_on_the_fe_entries_stack_is_the_decode_steps_of_its_tokens_and_passes_its_kernel_check():
    """One sequence of six tokens: the step shape the census of the target has a recorded section for on compute capability
    10.0 with 148 SMs.  The decode run's 32 tokens go through verify steps that keep 0 to 5 tokens in turn; every row is
    compared with the decode step of its token, the state of the kept tokens with the decode steps' state, the verify
    entry token by token with reference decode steps, and every launch of the traced verify steps with the plan.  Six
    rows are a row of the FP8 projection entry for the linear-attention layer's out-projection on compute capability
    10.x: where the census holds the run, 13 launches of a step are target kernels."""
    out = _verify(1, 6)
    _verify_asserts(out, 1, 6)
    assert "tokens kept behind the first: 0, 1, 2, 3, 4, 5" in out, "every number of tokens is kept once"
    assert "batch invariance  not applicable at 1 sequence" in out
    if "kernel identity, as planned: " in out:  # a GPU the census has a section for: what the plan reaches of the target, with its split
        reached = "13 of 20 launches of a step are target kernels (2 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results)"
        assert f"kernel identity, as planned: {reached}; not yet: mlp.down 2, attention.prepare 1, attention.core 1, projection.fp8.out 1, output_head 1" in out


@pytest.mark.L1
def test_a_verify_step_of_four_sequences_runs_each_sequence_alone_with_its_tokens():
    """Four sequences of six tokens, the last step shape the row of six tokens claims: one thread block per value head and
    sequence, and in the batch-invariance check one sequence alone with its six rows -- for which the verify entry takes
    another form, as its determinism row says (printed, never gated).  At 24 rows the out-projection is the multiply over
    a copy at both call sites on compute capability 10.0 (the rows of the projection entry end at 16 there) and the
    entry's row of 17 to 32 rows on 10.3 and 10.7."""
    out = _verify(4, 6)
    _verify_asserts(out, 4, 6)
    assert "batch invariance  one sequence alone vs among 4 (step t: sequence t mod 4)" in out
    assert "linear_attention.step equal (gated)" not in out, "the entry states no batch invariance: nothing is gated on it"


def _header(lines, tag):
    """The header of a run among the lines of its standard error: the first line that begins with the script's tag.  Standard
    error is not the sample's alone -- a library may print a warning there before the sample's first line -- so the header
    is found, never taken to be line 0."""
    tagged = [line for line in lines if line.startswith(tag)]
    assert tagged, f"no line of the run begins with {tag}: the run printed no header"
    return tagged[0]


def _in_front_of_the_verdict(lines, tag):
    """The line of a run that stands directly in front of its verdict, the last line that begins with the script's tag.  As
    for the header, the place is counted from a line of the run and never from the end of standard error: a library
    may print there behind the sample's last line (at interpreter exit)."""
    verdict = max((index for index, line in enumerate(lines) if line.startswith(tag)), default=0)
    assert verdict, f"no line of the run stands in front of a line that begins with {tag}"
    return lines[verdict - 1]


#: what the cells of the speculative script read a run's lines between: a line in front of the run and one behind it, as a
#: library prints them on standard error at import and at interpreter exit
NOT_THE_SAMPLES = ("a line on standard error in front of the run: a library's, at import", "a line on standard error behind the run: a library's, at exit")


@pytest.mark.L0
def test_the_header_of_a_run_is_found_behind_a_line_a_library_printed_first():
    """A run's lines are read from standard error, and standard error is not the sample's alone: a library may print a
    warning there before the sample's first line (seen in a fresh interpreter on one host: the sample ended in ``PASS``
    and the cell that took line 0 for its header failed).  The header is the first line with the script's tag."""
    header = "[02] qwen_dense speculative decode | reduced model: 2 of 64 layers (1 linear-attention + 1 attention) | the drafter runs on reference blocks"
    run = [header, "[02] planning: an entry of this stack builds its kernel now", "served by (decided before the load)", "[02] PASS"]
    warning = ["/site-packages/a_library/__init__.py:12: UserWarning: a library says something at import", "  warnings.warn(text)"]
    assert _header(run, "[02] ") == header
    assert _header(warning + run, "[02] ") == header, "two lines of a library in front of the run"
    assert _header(run[:1] + warning + run[1:], "[02] ") == header and _header(["[00] a line of the other script", *run], "[02] ") == header
    assert _header(["a library's line that quotes [02] PASS in passing", *run], "[02] ") == header, "a line of the run BEGINS with the tag"
    with pytest.raises(AssertionError, match=r"no line of the run begins with \[02\] : the run printed no header"):
        _header(warning, "[02] ")


@pytest.mark.L0
def test_the_line_in_front_of_the_verdict_is_found_where_a_library_printed_last():
    """The other end of a run's lines.  A cell that holds the line in front of the verdict counts from the verdict -- the
    last line with the script's tag -- and not from the end of standard error, where a library may print behind the
    sample's last line (seen with an interpreter made to print its import times: the sample ended in ``PASS`` and the
    cell that took the second line from the end failed).  Directly in front: no line of the run is skipped."""
    last = "[02] the drafter ran on reference blocks: none of its blocks was compared with a provider (no provider serves them yet)"
    run = ["[02] qwen_dense speculative decode | reduced model", "  ids  ids hash 11bfb9f9bcf5, plain 11bfb9f9bcf5    PASS", last, "[02] PASS"]
    behind = ["a line a library prints when the interpreter exits", "  and its second line"]
    assert _in_front_of_the_verdict(run, "[02] ") == last
    assert _in_front_of_the_verdict(run + behind, "[02] ") == last, "two lines of a library behind the verdict"
    assert _in_front_of_the_verdict(behind + run + behind[:1], "[02] ") == last
    assert _in_front_of_the_verdict([*run[:-1], "a line between the two", run[-1]], "[02] ") == "a line between the two", "nothing is skipped"
    for nothing_in_front in (behind, run[-1:], []):
        with pytest.raises(AssertionError, match=r"no line of the run stands in front of a line that begins with \[02\] "):
            _in_front_of_the_verdict(nothing_in_front, "[02] ")


PAGED_PROGRAM = """
import faulthandler, sys
faulthandler.dump_traceback_later({deadline}, exit=True)
sys.path.insert(0, {package_root!r})
import cudnn
assert cudnn.__file__ == {cudnn_file!r}, f'the child imported another cudnn: {{cudnn.__file__}}'
import torch
sys.path.insert(0, {models!r})
import model_samples
from model_samples import census, timing
from model_samples.providers import AttentionParams, torch_ref
from model_samples.qwen_dense import QwenDense
from model_samples.state import DecodeState

PAGE, SEQUENCES, FIRST, LAST = 64, 2, 60, 68  # the traced steps: positions 60 to 68, across the first page boundary


def say(text):
    print(text, file=sys.stderr, flush=True)


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


failed = []


def hold(name, held, text):
    say(f"  {{name:<10}} {{text}}    {{'PASS' if held else 'FAIL'}}")
    if not held:
        failed.append(name)


device = torch.device("cuda:0")
model = QwenDense.reduced()  # the published widths: 24 query heads, 4 key / value heads of 256, a rotary part of 64
g, shapes, kinds = model.geometry, model.state_shapes(), ("full_attention",)
heads, width, capacity, pages = g.attn_kv_heads, g.attn_head_dim, LAST + 4, 2
generator = torch.Generator().manual_seed(0)


def draw(*shape, scale=1.0):
    return (scale * torch.randn(*shape, generator=generator)).to(torch.bfloat16).to(device)


params = AttentionParams(g.attn_heads, heads, width, g.rotary_dim, g.rope_theta, draw(width, scale=0.02), draw(width, scale=0.02))
rows = [draw(SEQUENCES, g.in_projection("full_attention")[0]) for _ in range(capacity)]  # an in-projection's row of every sequence, per position


def step(cache, position, tokens=1):
    # the two reference chains of an attention layer on one cache: the preparation writes the position, the core reads up to it
    if tokens > 1:
        row = torch.stack(rows[position : position + tokens], dim=1).flatten(0, 1)  # sequence by sequence
        q, gate = torch_ref.attention_prepare_tokens(row, cache, position, tokens, params, g.norm_eps, torch.float64)
        return q, gate, torch_ref.attention_tokens(q, cache, position, tokens, torch.float64)
    q, gate = torch_ref.attention_prepare(rows[position], cache, position, params, g.norm_eps, torch.float64)
    return q, gate, torch_ref.attention(q, cache, position, torch.float64)


plain = DecodeState(kinds, SEQUENCES, capacity, shapes, device)
paged = DecodeState(kinds, SEQUENCES, capacity, shapes, device, page=PAGE)
flat, cache, table = plain.layers[0], paged.layers[0], paged.table
say(f"[paged] {{model_samples.check_environment().header()}}")
say(f"[paged] the attention preparation and the attention core of the reference chain on a paged cache: {{SEQUENCES}} sequences, pool "
    f"{{list(cache.pool.shape)}}, positions 0 to {{LAST}}, the steps at {{FIRST}} to {{LAST}} under the profiler")
equal = sum(all(same(a, b) for a, b in zip(step(cache, position), step(flat, position))) for position in range(FIRST))
theirs = [step(flat, position) for position in range(FIRST, LAST + 1)]
mine, launches = [], []


def traced(index):
    mine.append(step(cache, FIRST + index))
    mine[-1][2][0, :4].tolist()  # the host read that ends a step


trace, missing = timing.trace_steps(traced, LAST + 1 - FIRST, device, records=launches)
equal += sum(all(same(a, b) for a, b in zip(own, other)) for own, other in zip(mine, theirs))
hold("outputs", equal == LAST + 1, f"queries, gates and attention outputs of {{equal}} of {{LAST + 1}} steps are the bytes of the same step on the plain cache")
# the pool, read by arithmetic written here: position p of sequence i is row p % 64 of page i * pages + p // 64, keys then values
rows_equal = behind = 0
for sequence in range(SEQUENCES):
    for position in range(pages * PAGE):
        row = cache.pool[sequence * pages + position // PAGE, :, position % PAGE]
        if position <= LAST:
            rows_equal += int(same(row[:, :width], flat.k[sequence, :, position]) and same(row[:, width:], flat.v[sequence, :, position]))
        else:
            behind += int(bool(row.any()))
hold("pool", rows_equal == SEQUENCES * (LAST + 1) and behind == 0,
     f"{{rows_equal}} of {{SEQUENCES * (LAST + 1)}} rows of the pool hold the plain cache's key and value row of their position; {{behind}} of the "
     f"{{SEQUENCES * (pages * PAGE - LAST - 1)}} rows behind position {{LAST}} hold anything")
reads = 0
for count in (1, 63, 64, 65, LAST + 1):
    for dtype in (torch.bfloat16, torch.float64):
        k = torch.stack([cache.pool[sequence * pages + p // PAGE, :, p % PAGE, :width] for sequence in range(SEQUENCES) for p in range(count)])
        v = torch.stack([cache.pool[sequence * pages + p // PAGE, :, p % PAGE, width:] for sequence in range(SEQUENCES) for p in range(count)])
        k, v = (t.view(SEQUENCES, count, heads, width).transpose(1, 2).to(dtype) for t in (k, v))
        reads += int(torch.equal(cache.keys(count, dtype), k) and torch.equal(cache.values(count, dtype), v))
hold("accessors", reads == 10, f"keys and values of the first 1, 63, 64, 65 and {{LAST + 1}} positions, in bf16 and in fp64: {{reads}} of 10 reads are the pool's rows")
chain = torch.stack([torch.cat(torch_ref.rope_cos_sin(position, g.rotary_dim, g.rope_theta, device)) for position in range(capacity)])
hold("cos / sin", same(table.cos_sin, chain), f"the table's row of each of {{capacity}} positions is what the chain forms for that position on this device, bit for bit")
# a step of three tokens per sequence across the boundary: positions 62 to 64, on states of that layout
several = [DecodeState(kinds, SEQUENCES, capacity, shapes, device, tokens=3, **layout) for layout in ({{}}, {{"page": PAGE}})]
for position in range(62):
    for state in several:
        step(state.decode_layers[0], position)
pair = [step(state.layers[0], 62, tokens=3) for state in several]
_, positions, slots = several[1].layers[0].at(62)
tables = positions.tolist() == [[62, 63, 64] * SEQUENCES] * 3 and slots.tolist() == [s * pages * PAGE + p for s in range(SEQUENCES) for p in (62, 63, 64)]
hold("verify", all(same(a, b) for a, b in zip(*pair)) and tables,
     "a step of 3 tokens per sequence at positions 62 to 64: the plain cache's bytes; its table rows name those positions and their slots")
ids = {{}}
for record in launches:
    ids[census.canonical(record)] = ids.get(census.canonical(record), 0) + 1
unnamed = sorted({{record.name[:80] for record in launches if not census.canonical(record)}})
if not launches:
    say(f"  kernels    not observed: {{missing}}")
else:
    counted = ", ".join(f"{{name or 'NO ID'}} {{count}}" for name, count in sorted(ids.items()))
    gathers, nameless = ids.get("embedding_gather", 0), ids.get("", 0)
    foreign = sum(count for name, count in ids.items() if name not in ("pytorch", "cublas", "copy", "embedding_gather", ""))
    hold("kernels", gathers == nameless == foreign == 0,
         f"{{len(launches)}} records of {{LAST + 1 - FIRST}} steps on the pool: {{counted}}; {{gathers}} with the id of a step's embedding lookup, "
         f"{{nameless}} unnamed{{'' if not unnamed else ' (' + '; '.join(unnamed) + ')'}}, {{foreign}} with an id outside the chains' families")
if failed:
    say("[paged] FAIL")
elif not launches:
    say(f"[paged] RAN (kernels not observed: {{missing}})")
else:
    say("[paged] PASS")
raise SystemExit(1 if failed else 0)
"""


@pytest.mark.L0
def test_the_reference_chains_on_a_paged_cache_give_the_plain_caches_bytes_and_launch_no_gather():
    """The paged attention cache of a state (``state.PagedKvCache``) under the two reference chains of an attention layer,
    on the device, at the published widths and two sequences -- no provider row asks for that layout yet, so no run of a
    script builds it.  A small program in a fresh interpreter runs the attention preparation and the attention core at
    positions 0 to 68 on a plain cache and on a paged one, the steps at 60 to 68 of the paged one under the profiler:

    * the queries, gates and attention outputs of every step are the plain cache's bytes;
    * the pool, read by index arithmetic written in the program, holds the plain cache's key and value row of every
      position, and the accessors return those rows;
    * the cos / sin table of the state is, row for row, what the chain forms for that position on this device;
    * a step of three tokens per sequence across positions 62 to 64 gives the plain cache's bytes;
    * no kernel the chains launch on the pool has the id of a step's embedding lookup, and none is unnamed: a gather
      through an index tensor would be counted as the embedding's launch by the kernel check of every stack.

    The chains are PyTorch's, so no row of the family declines a GPU here; the cell is gated all the same.  It runs on
    the classes the family serves -- compute capability 10.x and 12.x, where the other cells of this module run -- and
    on compute capability 8.0: a GPU the family serves and one of 8.0 are the two kinds it has run on, and it holds
    every kernel the chains launch.  On any other class it skips with a sentence that names the classes and the device.
    Where the profiler's records cannot be had the last line says so, and the kernels are uncovered there, not verified."""
    cc = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None  # no GPU: the runner of the program skips
    if cc is not None and cc[0] not in (10, 12) and cc != (8, 0):
        pytest.skip(
            "the cell of the paged cache runs where the family serves the GPU (compute capability 10.x or 12.x) and on compute capability 8.0; "
            f"this device is {cc[0]}.{cc[1]}, which the family does not serve: uncovered, not passing"
        )
    out = _program(
        PAGED_PROGRAM,
        "[paged]",
        "model sample: the reference chains on a paged attention cache",
        "run on a paged cache",
        unobserved="RAN (kernels not observed: ",
    )
    lines = out.splitlines()
    assert any(line.startswith("[paged] ") and line.endswith("positions 0 to 68, the steps at 60 to 68 under the profiler") for line in lines)
    for name in ("outputs", "pool", "accessors", "cos / sin", "verify"):
        (line,) = [line for line in lines if line.startswith(f"  {name} ")]
        assert line.endswith("    PASS"), line
    assert "queries, gates and attention outputs of 69 of 69 steps" in out and "138 of 138 rows of the pool" in out and "0 of the 118 rows behind" in out
    observed = [line for line in lines if line.startswith("  kernels ")]
    none = "; 0 with the id of a step's embedding lookup, 0 unnamed, 0 with an id outside the chains' families    PASS"  # the line states its counts
    assert len(observed) == 1 and (observed[0].endswith(none) or observed[0].startswith("  kernels    not observed: ")), observed
    frost_routing.measured("model sample: the reference chains on a paged attention cache, the kernels of the traced steps", observed[0].strip())


@pytest.mark.L0
def test_the_cell_of_the_paged_cache_runs_where_the_family_serves_the_gpu_and_on_compute_capability_8_0(sample, monkeypatch):
    """Where the cell of the paged cache runs, without a GPU: the cell is called on stated compute capabilities, with a
    stand-in for the runner of its program.  It reaches its program on every class the family serves -- read here from
    the family's own table: the classes on which the default stack has a row for every block, which is what lets the
    other cells of this module run -- and on compute capability 8.0, and on no other class.  There it skips, and the
    sentence names the classes it runs on and the device.  Without a GPU it asks for no class: the runner skips."""
    from model_samples.qwen_dense import configs  # on the path while the fixture holds the script

    def served(cc):  # the plan of the default stack declines a class on which one of its blocks has no row
        return all(any("fe-cublas" in row.stacks and row.serves_class(cc) for row in rows) for rows in configs.PROVIDERS.values())

    class Reached(Exception):
        pass

    def runner(*_, **__):
        raise Reached

    def cell(cc):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: cc)
        try:
            test_the_reference_chains_on_a_paged_cache_give_the_plain_caches_bytes_and_launch_no_gather()
        except Reached:
            return "runs"
        except pytest.skip.Exception as skipped:
            return skipped.msg

    monkeypatch.setattr(sys.modules[__name__], "_program", runner)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    classes = [(major, minor) for major in range(7, 14) for minor in range(10)]
    family = [cc for cc in classes if served(cc)]
    assert family == [cc for cc in classes if cc[0] in (10, 12)], "the classes the family serves: compute capability 10.x and 12.x"
    assert [cc for cc in classes if cell(cc) == "runs"] == sorted([(8, 0), *family]), "where the family serves the GPU, and on compute capability 8.0"
    elsewhere = (
        "the cell of the paged cache runs where the family serves the GPU (compute capability 10.x or 12.x) and on compute capability 8.0; "
        "this device is {}.{}, which the family does not serve: uncovered, not passing"
    )
    assert all(cell(cc) == elsewhere.format(*cc) for cc in classes if cc != (8, 0) and cc not in family), "a skip names the classes and the device"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cell(None) == "runs", "without a GPU the cell asks for no class: the runner of its program skips"


def _speculative_run(sample, capsys, arguments, stack="fe-cublas"):
    """Run ``main()`` of the speculative script and hold what every such run prints; with ``sample`` None, run the script in
    a fresh interpreter instead.  Standard output must stay empty (a seeded run has no language); the verdict is ``[02]
    PASS`` or, where the kernels were to be checked and the profiler's records could not be had, ``[02] RAN <stack> stack
    (kernels not observed: ...)`` -- recorded as it is.  A ``SKIP`` skips the cell, unless a check failed before it."""
    start = time.perf_counter()
    if sample is None:
        code, out, written = _fresh_interpreter(arguments, SPECULATIVE, "[02]")
    else:
        code = sample.main(arguments)
        captured = capsys.readouterr()
        out, written = captured.err, captured.out
    seconds = time.perf_counter() - start
    assert written == "", f"a seeded run writes nothing to standard output; it holds\n{written[-3000:]}"
    # Standard error is not the sample's alone: a library may print there in front of the run and behind it.  Every cell
    # that comes through here reads the run's lines with such a line at either end, so that nothing below can take a
    # line of the run by its place from the start or from the end of standard error.
    lines = [NOT_THE_SAMPLES[0], *out.splitlines(), NOT_THE_SAMPLES[1]]
    tagged = [line for line in lines if line.startswith("[02] ")]
    versions = next((line[5:] for line in tagged if line.startswith("[02] cuDNN Frontend ")), "no environment line")
    device = next((line[5:].split(" | ")[0] for line in tagged if " SMs)" in line), "no device line")
    verdict = tagged[-1] if tagged else "no output"
    key = "model sample qwen_dense 02_speculative_decode " + (" ".join(arguments) or "(default)")
    frost_routing.measured(key, f"{verdict}; stack {stack}; {device}; {versions}; {seconds:.0f} s")
    if verdict.startswith("[02] SKIP"):
        assert " FAIL" not in out and "[02] ERROR" not in out, out[-6000:]  # a skip never covers a failed check
        pytest.skip(verdict)
    assert code == 0 and (verdict == "[02] PASS" or verdict.startswith(f"[02] RAN {stack} stack (kernels not observed: ")), out[-6000:]
    assert " FAIL" not in out and "distribution" in versions and "source hash" in versions
    # before the load: the reductions, the configuration key by key, the seeded list, what a prompt and a step are
    header = _header(lines, "[02] ")
    assert "2 of 64 layers (1 linear-attention + 1 attention)" in header and header.endswith(" | the drafter runs on reference blocks")
    assert "acceptance is scheduled, not measured: the drafted tokens of every step are replaced by the plain run's tokens for 0, 1, .., 5 positions" in header
    load = next(index for index, line in enumerate(lines) if line.startswith("[02] stage "))
    before = (
        '[02] --speculative-config (the default): the speculative configuration of a serving engine. Read: method "mtp", num_speculative_tokens 5, '
        "use_local_argmax_reduction true.",
        "[02] draft vocabulary seeded list (seed 0): 3,880 ids of the model's 15,520, in ascending order (3,880 entries listed, none dropped)",
        "[02] prompt of 26 tokens = 26 decode steps and 25 passes of the drafter before the first token; then every step is 5 draft passes and one "
        "verify step of 6 tokens, and keeps 1 to 6 of them",
        "the prompt of this session goes through decode steps, one per prompt token, on these rows:",
        "the drafter: 18 block calls a pass; the first pass of a step behind a verify step runs at S x T rows, every other pass "
        "and every pass over the prompt at S rows; every row of this table serves any row count",
    )
    assert all(line in lines[:load] for line in before), [line for line in before if line not in lines[:load]]
    assert "with its drafter, one after the other" in out and out.count("this stack needs about ") == 1, "one memory sentence, the drafter in it"
    # the report, in front of the comparison: the schedule, no rate, the bytes of a step
    against = lines.index(
        "speculative against decode (1 sequence x 6 tokens per sequence, 10 engine steps behind a prompt of 26 tokens; "
        "tokens kept behind the first: 0, 1, 2, 3, 4, 5, 0, 1, 2, 3; 2 layers)"
    )
    report = lines.index("acceptance            scheduled, not measured (reduction): 10 steps; kept 1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1")
    assert (
        report < against and "rate                  not printed while the drafter runs on reference blocks (kept tokens per step: the acceptance line)" in lines
    )
    assert any(
        line.startswith("first token ") and "each but the first behind one pass of the drafter over the position in front of it" in line for line in lines
    )
    assert any(line.startswith("steps (10) ") and line.endswith(" | 110 block calls per step") for line in lines), "five passes of 18 calls and the model's 20"
    assert any(line.startswith("bytes per step ") and " + drafter 5 x " in line for line in lines[:against])
    assert any(line.startswith("text out              32 tokens in 10 steps (seeded weights: not language): ") for line in lines[:against])
    # the comparison with decode steps of the same tokens: every line of it, and the script's own line of the ids
    for name in ("tokens", "position", "prompt", "recorded", "rejected rows", "drafter cache", "draft proposals", "acceptance", "ids"):
        (line,) = [line for line in lines if line.startswith(f"  {name} ") and "greedy tokens equal to the reference stack's" not in line]
        assert line.endswith("    PASS"), line
    assert "ids and logits of 36 of 36 steps the session recorded of its own run (26 of its prompt, 10 verify steps) are this replay's bytes" in out
    assert "26 prompt tokens + the kept ones = 57; the session's own run ended there    PASS" in out
    assert "10 of 10 steps kept the count they were scheduled to keep; every count from 0 to 5 was kept at least once    PASS" in out
    kept = next(line for line in lines if line.startswith("  kept state  "))
    assert kept.endswith(("    PASS", "    REPORTED")), kept
    reported = next(line for line in lines if line.startswith("  bytes  "))
    assert reported.endswith("    REPORTED") and f"(reported: on stack {stack} the two kinds of step may launch other kernels for a block)" in reported
    ids = next(line for line in lines if line.startswith("  ids  "))
    hashes = ids.split("ids hash ")[1].split("    ")[0].split(", plain ")
    assert hashes[0] == hashes[1] and f"[02] seeded weights: the output is not language; ids hash {hashes[0]}" in lines
    # the checks of the run: replay with the proposals, the blocks of both kinds of step
    assert (
        "checks (1 sequence x 6 tokens per sequence, 10 verify steps; tokens kept behind the first: 0, 1, 2, 3, 4, 5, 0, 1, 2, 3; "
        "behind a prompt of 26 decode steps)"
    ) in lines
    assert "  replay            ids, logits and the drafter's proposals byte-identical on a second run from a fresh state    PASS" in lines
    judged = [line for line in lines if line.startswith("  blocks  ")]
    assert len(judged) == 2 and all(line.endswith("    PASS") for line in judged), "the prompt's decode steps and the verify steps, each against the reference"
    steps = [line for line in lines if line.split()[:1] == ["linear_attention.step"] and "[band: gdn_step]" in line]
    assert len(steps) == 2 and all(line.endswith("[band: gdn_step]  PASS") and "conv history equal; null slot untouched" in line for line in steps)
    last = "[02] the drafter ran on reference blocks: none of its blocks was compared with a provider (no provider serves them yet)"
    assert _in_front_of_the_verdict(lines, "[02] ") == last
    return out


@pytest.mark.L0
def test_the_speculative_default_run_passes_its_own_checks(speculative, capsys):
    """The default run of ``02_speculative_decode.py``, loaded by path, in this process: the stack ``fe-cublas``, both
    streams held.  It is the only L0 cell that launches the fused verify step and the fused decode step on token 0 of a
    state of six tokens per sequence in the form of the default stack; what their first launches took is recorded.  The
    kernels are not checked on this stack unless asked: no kernel line, and no statement about the drafter's kernels."""
    out = _speculative_run(speculative, capsys, [])
    lines = out.splitlines()
    assert "stack fe-cublas, 1 sequence x 6 tokens per sequence" in out and "cudnn.GdnVerifyFused " in out and "cudnn.GdnDecodeFused " in out
    assert "[PDL " not in out, "an entry's form is printed with the kernel table, which this stack prints on request"
    assert "\n  kernels  " not in out and "kernel identity" not in out, "the default stack declares no target and its kernels are checked on request"
    stages = next(line for line in lines if line.startswith("[02] the session with the drafter: stage  "))
    forms = [part.strip() for part in stages.split("stage  ", 1)[1].split(" | ") if part.strip().startswith("compile and first launch linear_attention.step ")]
    assert [form.rsplit(" ", 2)[0] for form in forms] == [
        "compile and first launch linear_attention.step (1 row)",
        "compile and first launch linear_attention.step (6 rows)",
    ], "the decode step on token 0 of the state of six tokens, then the verify step"
    # an entry builds its kernel at the plan (its first use) and launches it once at the load: both are recorded with their seconds
    plan = next(part.strip() for line in lines if line.startswith("[02] stage  ") for part in line.split(" | ") if part.strip().startswith("plan "))
    frost_routing.measured("model sample qwen_dense 02_speculative_decode (default): the two forms this cell alone launches", "; ".join([plan, *forms]))


@pytest.mark.L1
def test_the_speculative_run_on_the_fe_entries_stack_passes_its_kernel_check(capsys):
    """The default run of ``02_speculative_decode.py`` on the stack ``fe-entries``, in a fresh interpreter, both streams
    held: the cuDNN Frontend rows of ``native`` under the verify step and under the prompt's decode steps, the drafter's
    passes on reference blocks in front of every verify step.  The traced pass holds whole steps behind the replayed
    prompt, and the kernel check holds them: the lookups of the passes and the verify plan's rows one by one, the
    chains of the passes by family; under it the run says that the prompt's decode steps were not observed and that it
    states nothing of the drafter's kernels.  Where the census has a section for the GPU, the verify plan's share of
    the target kernel set is stated with its split."""
    if not torch.cuda.is_available():  # decided here, before a child is started
        pytest.skip("needs a CUDA GPU")
    out = _speculative_run(None, capsys, ["--stack", "fe-entries"], "fe-entries")
    lines = out.splitlines()
    assert "stack fe-entries, 1 sequence x 6 tokens per sequence" in out and "cudnn.GdnVerifyFused [PDL True, PDL_RELEASE True, TOKENS 6]" in out
    assert "cudnn.GdnDecodeFused " in out and "cudnn.GdnDecodeFused [" not in out, "the prompt's decode entry: a row of its plan; no form is printed for it"
    assert out.count("\nkernels, planned (decided before the load") == 1, "the kernel table of the verify plan"
    traced = [line for line in lines if line.startswith("traced pass (")]
    assert not traced or traced[0].endswith(" | behind the prompt's 26 decode steps, replayed untraced")
    if "kernel identity, as planned: " in out:  # a GPU the census has a section for: the verify plan's share of the target, with its split
        reached = "13 of 20 launches of a step are target kernels (2 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results)"
        assert f"kernel identity, as planned: {reached}; not yet: mlp.down 2, attention.prepare 1, attention.core 1, projection.fp8.out 1, output_head 1" in out
    observed = [line for line in lines if line.startswith("  kernels  ")]
    assert len(observed) == 1, "the kernel check of the traced steps ran, or said why it could not look"
    # the kernel check is of the verify steps: beside its verdict the run says what it did not look at, and what it does not state
    assert "    the prompt's decode steps: 26 replayed in front of the traced window, their kernels not observed (the check is of the verify steps)" in lines
    assert (
        "    kernel identity of the drafter's passes: not stated (the kernel census holds the verify step and the output head behind it; "
        "no section holds a draft pass)"
    ) in lines
    frost_routing.measured(
        "model sample qwen_dense 02_speculative_decode --stack fe-entries: the kernel check", "; ".join(line.split("    ")[-1] for line in observed)
    )
