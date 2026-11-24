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
the four-layer one (three + one), and a run of 85 sequences of the two-layer model, whose peak of allocated memory is held to the memory
sentence and to the count behind it.

The ``fe-entries`` cell runs the same default run on the stack ``fe-entries``: every cuDNN Frontend row of the stack ``native``
(the fused linear-attention step in its wait + release form; on compute capability 10.x the attention gate + quantise
entry, the add + RMSNorm + quantise entry in the form of each call site, the fused 4-bit gate / up entry and the attention
preparation entry, which writes a paged attention cache, each judged against its twin) with every other block on its
fallback row -- no package beyond cuDNN Frontend and PyTorch.
That stack always checks its kernels: the profiler's record of every launch of the traced steps is held against the plan
and against the entries' own launch facts.
Where the profiler's records cannot be had the sample ends in ``[00] RAN fe-entries stack (kernels not observed: <reason>)``:
the cell then passes on the other checks and the recorded line says that the kernels were not observed -- uncovered
there, not verified.

The ``native`` cell runs the default run on the stack ``native`` itself, which has four rows more: three multiplies of a
library, FlashInfer, on the stored weights, and the attention core on that library's paged decode attention.
FlashInfer is an optional package of that stack alone, so the cell runs where it is installed at a validated version with
its kernel-binary package and is skipped elsewhere -- by this module where the library is absent, by the sample's own
``[00] SKIP`` line where its version is not validated or a kernel binary is not on the machine.  A skipped cell is
uncovered, not passing.  A second cell of that stack sets the library's directory variable to an empty directory: the
run ends in ``[00] SKIP`` before anything is loaded, by one of the sample's own sentences, and neither stream holds a path.

The attention core's cells run no script: a small program in a fresh interpreter takes the block of the stack ``native``
for a step shape and calls it on a paged cache whose every row is drawn, judged in the row's band the way a run judges
a call.  One cell holds five drawn cases at one sequence; one runs the control arms, each of which must fail -- a
masked row let in, the newest row dropped, the last page exchanged with a full one, the scores at the library's default
scale, a twin that reads a value row from the next slot or lets every row of a step attend to all of them -- and the
adapter's own decline of 64-bit tables; one (L1) runs a step shape for each kernel form of the library and records the
kernel's name.

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
token, against reference decode steps; its launch against the entry's own facts.  The out-projection of the linear-attention layer is
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

The attention preparation's cell runs 80 plain steps of the stack ``fe-entries`` at two sequences, across the first page
boundary of the attention cache, and holds the pool its entry wrote against a plain cache that the entry's twin fills
from the rows the in-projection handed the entry: read by index arithmetic of its own -- value rows bitwise, key rows
inside the entry's band, every row behind the last position still zero --, the cache's accessors against the same
arithmetic, then the session's own checks over all 80 steps.  Two verify cells run steps of several tokens on that pool:
two sequences of six tokens with a step across the boundary and rejected rows on either side of it, and 16 sequences
of five tokens, 80 rows in one launch.

The paged-cache cell runs no script: it holds the two reference chains of an attention layer on the paged attention cache
of a state (``state.PagedKvCache``), without the entry.  A small program in a fresh interpreter runs the two reference chains of an
attention layer on a plain cache and on a paged one at the published widths, across the first page boundary, and holds
the bytes of the two against each other, the pool against index arithmetic of its own, the state's cos / sin table against
the chain's function on the device, and the kernels the chains launch on the pool to their families.  It runs where the
family serves the GPU and on compute capability 8.0, and is skipped by name on any other class; a host case holds that
condition on stated compute capabilities.

The recorded-step cells run no script either: a small program in a fresh interpreter builds two sessions of the stack
``native`` on one load -- one launched from Python, one that records its decode step once and replays it
(``Session(capture=True)``) -- and holds a replayed request to the same request launched from Python: the ids, the
logits byte for byte, the end-state digest, at 1 and at 4 sequences; then that a planted difference is found at its
step, that a replay past the capacity launches nothing, and that a declined recording leaves PyTorch's current stream
where it was.  They run where the library is installed, as the ``native`` cell does.

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


FE_CUBLAS = "by provider: cuDNN FE 1 | fallbacks 9 (cuBLAS over copies 4, PyTorch chains 5) | PyTorch 2"
#: with the gate entry, the norm entry, the fused gate / up entry, the attention preparation entry and the entry of the head's
#: quantise (compute capability 10.x) five more blocks are cuDNN Frontend's: four reference chains and one multiply over a copy
#: fewer are fallbacks
FE_ENTRIES = "by provider: cuDNN FE 6 | fallbacks 4 (cuBLAS over copies 3, PyTorch chains 1) | PyTorch 2"
#: on compute capability 10.0 the in-projection is the FP8 projection entry's as well; the out-projection stays a multiply over a copy
FE_ENTRIES_CC100 = "by provider: cuDNN FE 7 | fallbacks 4 (cuBLAS over copies 3, PyTorch chains 1) | PyTorch 2"
#: the stack native on compute capability 10.x: the three multiplies and the attention core FlashInfer serves are no fallbacks,
#: and no lane of this repository runs them
NATIVE = "by provider: cuDNN FE 6 | FlashInfer 4 | fallbacks 1 (cuBLAS over copies 1) | PyTorch 2 | unverified in CI: 4"
NATIVE_CC100 = "by provider: cuDNN FE 7 | FlashInfer 4 | PyTorch 2 | unverified in CI: 4"


CHILD_SECONDS = 1200  # the fresh interpreter's deadline: imports, the compiles of both sequence counts and the run, far above what they take


def _fresh_interpreter(arguments, script=SCRIPT, tag="[00]", environment=None):
    """A sample script as a user runs it: in a fresh interpreter, which imports the ``cudnn`` package this process imported.
    Returns ``(exit code, the run's lines, what the run wrote to standard output)``: a sample writes every line of a run
    to standard error and keeps standard output for generated text.  The child arms its own stack dump before the
    parent's deadline; where the deadline passes, the failure carries what the child had printed.  ``environment``:
    variables the child has beside this process's."""
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
        env = None if environment is None else {**os.environ, **environment}
        done = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=CHILD_SECONDS, env=env)
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
    assert "2 of 64 layers (1 linear-attention + 1 attention)" in out and "21 block calls per step" in out


@pytest.mark.L0
def test_default_run_on_the_fe_entries_stack_passes_its_own_checks_and_its_kernel_check(capsys):
    """The cuDNN Frontend rows of the stack ``native``, on real launches: the fused step in its wait + release form and, on
    compute capability 10.x, the gate + quantise entry, the add + RMSNorm + quantise entry (four of its forms on the
    two-layer model, each call served by the instance of its call site) and the fused 4-bit gate / up entry (the kernel
    stated whole, in the token tile of the row count) and the attention preparation entry (on a paged attention cache, the
    kernel stated whole) inside their bands against their twins; on compute capability 10.0
    the FP8 projection entry for the in-projections (one instance per weight shape) inside the band of the reference
    multiply, and on every other class that block on its multiply over a copy; the kernel check on every launch of the
    traced steps, and under each kernel table its count of a step.
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
    assert "cudnn.GdnDecodeFused [PDL True, PDL_RELEASE True]" in out, "the fused step in the form of the stack's row"
    assert out.count("\nkernels, planned (decided before the load") == 2 and out.count("\n  kernels  ") == 2, "the kernel check ran at both sequence counts"
    counted = [line for line in out.splitlines() if line.startswith("  per step: ") and " exact row" in line]
    assert len(counted) == 2 and counted[0] == counted[1], "the count of a step under the kernel table, the same at both sequence counts"
    if projection:  # every entry row of the stack serves this class: the four blocks a library serves in native are on their fallback rows
        assert counted[0].startswith("  per step: 14 launches on 8 exact rows, checked one by one in the wiring order; the launches of 6 fallback calls ")
    assert "; the library launched " not in out, "no row of this stack is a library's"
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
        assert "cudnn.RowQuantizeFp4Decode [WIDTH 5120, PDL True, PDL_RELEASE True]" in out, "the entry of the head's quantise, its whole config"
        quantised = [line for line in out.splitlines() if line.split()[:1] == ["output_head.quantize"] and "calls" in line and "[band: " in line]
        assert len(quantised) == 2 and all(
            line.endswith("; 0 outside  [band: row_quantize_fp4]  PASS") for line in quantised
        ), "the head's quantise was judged against its twin at both sequence counts"
        assert "output_head.quantize equal (gated)" in out or "REPORTED" in out
        assert (
            "cudnn.AttentionPrepareDecode [POSITIONS 3d, SECTION (11, 11, 10), PAGE_SIZE 64, POSITION_DTYPE int64, SLOT_DTYPE int64, PDL False, PDL_RELEASE False]"
            in out
        ), "the attention preparation entry, the kernel stated whole"
        prepared = [line for line in out.splitlines() if line.split()[:1] == ["attention.prepare"] and "[band: " in line]
        whole = (
            "0 query and 0 key elements outside their band; gate rows equal; value rows equal; the rest of the pool untouched  [band: attention_prepare]  PASS"
        )
        assert len(prepared) == 2 and all(
            line.endswith(whole) for line in prepared
        ), "judged against its twin, and its pool row by row, at both sequence counts"
        assert "attention.prepare equal (gated)" in out or "REPORTED" in out
        stated = [line for line in out.splitlines() if line.strip().startswith(f"mlp.gate_up: {fused}, 2 launches per step") and "; the entry states " in line]
        assert len(stated) == 2 and all("block (192, 1, 1), shared memory 231,424 B" in line for line in stated), "one launch per MLP, at both sequence counts"
        # the weights the run keeps on their stored format alone: the gate / up weights, and on 10.0 the in-projections too
        kept, count = ("projection.fp8.in, mlp.gate_up", 4) if projection else ("mlp.gate_up", 2)
        made = [line for line in out.splitlines() if line.startswith("  on demand ")]
        assert len(made) == 2 and all(
            f"weights this run keeps on their stored format alone ({kept})" in line for line in made
        ), "no bf16 copy of these weights is held: the reference of the checks made its own, at both sequence counts"
        assert (
            "GiB of it for the checks (the reference's bf16 copy of one weight at a time, where the run keeps none, and what they hold for the sequences "
            "beyond the first of a count)" in out
        ), "and the memory sentence says so, beside what the checks hold for the sequences beyond the first"
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
        assert "\n  on demand " not in out and "the reference's bf16 copy of one weight" not in out, "every weight has its bf16 copy: nothing is made on demand"
        assert (
            "GiB of it for the checks (what they hold for the sequences beyond the first of a count)" in out
        ), "the checks' part of the sentence is of the rows alone"
        assert "\n  stored weights " not in out, "and the reference reads the copies of the load: no stored bytes to hold"
        assert "cudnn.SwigluFp4Decode" not in out and "cudnn.AttentionPrepareDecode" not in out, "the rows claim compute capability 10.x only"
        assert "cudnn.RowQuantizeFp4Decode" not in out, "and so does the row of the head's quantise"
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
    assert data["schema"] == "model-sample-performance/5" and data["record"]["class"] == "orientation" and data["checks"]["last_line"] == verdict
    assert data["protocol"]["stack"] == stack and str(tmp_path) not in report.read_text(encoding="utf-8"), "no path in the file"
    requests = data["raw"]["requests"]
    assert [request["kind"] for request in requests] == ["checked", "warm-up", "timed", "timed", "traced"]
    assert requests[2]["ids_sha256"] == requests[3]["ids_sha256"] == requests[0]["ids_sha256"], "every timed request: the checked request's ids"
    assert requests[2]["end_state_sha256"] == requests[3]["end_state_sha256"] == requests[0]["end_state_sha256"], "and its end state, byte for byte"
    assert data["protocol"]["profiler_first_started"] == (None if unobserved else "behind the last timed request, in the traced request")
    # the record class: every reason, in the order of the conditions; both stacks have a block on a fallback row
    observed = ["kernels not observed"] if unobserved else []
    fixed = ["reduced model", "fallback rows present", "launched from Python", *observed]
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
        child = "a child process that the timing process neither reads nor polls between its start and its stop"
        assert data["protocol"]["in_a_timed_request"]["sampler"] in (child, "none (the sampler could not be started)")
    else:
        assert watched["reason"] and seen["verdict"] == "not observed" and clocks["verdict"] == "no"
        assert data["protocol"]["in_a_timed_request"]["sampler"] == f"none ({watched['reason']})"
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
    # the launches of a step as the plan states them: on exact rows, by provider, and the block calls on fallback rows -- an exact
    # row launches once per call, so the two add up to the 20 block calls of a step of the two-layer model that are no host path
    assert kernels["exact_launches_per_step"] == sum(kernels["by_provider"].values()) and kernels["fallback_rows"]
    assert kernels["exact_launches_per_step"] + kernels["fallback_calls"] == 20
    if stack == "fe-cublas":
        assert kernels["by_provider"] == {"PyTorch": 1, "cuDNN Frontend": 1}
    elif tuple(torch.cuda.get_device_capability()) == (10, 0):
        assert kernels["by_provider"] == {"cuDNN Frontend": 13, "PyTorch": 1}
    else:  # a row of an entry claims its GPU classes: off compute capability 10.0 fewer of the 13 are exact rows
        assert kernels["by_provider"]["PyTorch"] == 1 and 1 <= kernels["by_provider"]["cuDNN Frontend"] < 13 and len(kernels["by_provider"]) == 2
    at = next(index for index, line in enumerate(screen) if line.startswith("kernels     "))
    assert screen[at] == f"kernels     {kernels['exact_launches_per_step']} launches per step on exact rows: " + ", ".join(
        f"{provider} {count}" for provider, count in kernels["by_provider"].items()
    )
    assert screen[at + 1].startswith(f"            {kernels['fallback_calls']} block calls on fallback rows; ")
    assert (
        "scope       the decode loop of one process: no request queue, no scheduler, no network" in screen[2:5]
    ), "behind the record class, however many lines that takes"


@pytest.mark.L0
def test_default_run_on_the_native_stack_with_its_library_rows(capsys):
    """The stack ``native`` with the rows a library serves, on real launches, where FlashInfer is installed: the MLP's down
    multiply and the output head on ``flashinfer.gemm.mm_fp4`` and the out-projections on ``flashinfer.gemm.bmm_fp8``, each
    on the stored weight and each inside the band of the reference multiply at both sequence counts; the attention core
    on the library's paged decode attention, inside the band of its own row at both counts, the cache it reads left as
    it was; the two first uses of the library announced once each while the run is planned; the 4-bit multiply outside
    any tuning context (``TUNED False`` in the call log, or the kernel check fails); every weight of a multiply on its
    stored format alone.  Where the kernels of the traced steps were observed, the line of every row the library serves
    names the kernel the library launched, at both sequence counts.
    Skipped where the library is absent, and by the sample's own line where a kernel binary is."""
    _needs_the_library()
    cc100 = tuple(torch.cuda.get_device_capability()) == (10, 0)
    out = _run(None, capsys, ["--stack", "native"], "native", NATIVE_CC100 if cc100 else NATIVE, unobserved=True)
    assert "stack native" in out and out.count("[00] planning: first use of FlashInfer ") == 1, "the library's first use is announced once, while the run plans"
    assert out.count("[00] planning: first use of the paged decode attention of FlashInfer ") == 1, "and so is the first use of its attention"
    core = [line for line in out.splitlines() if line.split()[:1] == ["attention.core"] and "calls" in line and "[band: " in line]
    assert len(core) == 2 and all(
        "; 0 outside their bound; " in line and line.endswith("the cache untouched  [band: attention_bf16_weights]  PASS") for line in core
    ), "the attention core was judged in the band of its row at both sequence counts"
    served = "attention.core: flashinfer.decode.trtllm_batch_decode_with_kv_cache [KV_LAYOUT HND], 1 launch per step"
    assert sum(line.strip().startswith(served) for line in out.splitlines()) == 2, "one launch of the library's kernel per step, at both sequence counts"
    for block in ("projection.fp8.out", "mlp.down", "output_head"):
        judged = [line for line in out.splitlines() if line.split()[:1] == [block] and "calls" in line and "[band: " in line]
        assert len(judged) == 2 and all(line.endswith("[band: bf16_rounding]  PASS") for line in judged), f"{block} was judged at both sequence counts"
    quantised = [line for line in out.splitlines() if line.split()[:1] == ["output_head.quantize"] and "calls" in line and "[band: " in line]
    assert len(quantised) == 2 and all(
        line.endswith("; 0 outside  [band: row_quantize_fp4]  PASS") for line in quantised
    ), "the head's quantise, whose codes the library's head multiply reads as they are, was judged at both sequence counts"
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
    if cc100:  # every row of the stack serves this class: no block call of a step is on a fallback row
        counted = "  per step: 20 launches on 12 exact rows, checked one by one in the wiring order; the launches of 0 fallback calls "
        assert sum(line.startswith(counted) for line in out.splitlines()) == 2, "the count of a step under the kernel table, at both sequence counts"
    observed = [line for line in out.splitlines() if line.startswith("  kernels  ")]
    if len(observed) == 2 and all(line.endswith("    PASS") for line in observed):  # the kernels of the traced steps were observed
        launched = [line.strip().split(": ", 1)[0] for line in out.splitlines() if "; the library launched " in line]
        rows = ["projection.fp8.out at linear_attention.out", "mlp.down", "attention.core", "projection.fp8.out at full_attention.out", "output_head"]
        assert launched == rows * 2, "the line of every row the library serves names the kernel the library launched, at both sequence counts"
        named = [line for line in out.splitlines() if line.strip().startswith(served) and "; the library launched " in line]
        assert len(named) == 2 and all(line.endswith(" over the traced steps") and " fi_decode_attention by name; " in line for line in named)
    frost_routing.measured("model sample qwen_dense 00_decode --stack native: the kernel check", "; ".join(line.split("    ")[-1] for line in observed))


def _needs_the_library():
    """Skips by name where a cell of the library's rows cannot run: no GPU, another class than the rows claim, no library."""
    if not torch.cuda.is_available():  # decided here, before a child is started
        pytest.skip("needs a CUDA GPU")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the library rows of the stack native claim compute capability 10.x")
    if importlib.util.find_spec("flashinfer") is None:  # looked up, not imported: the package is the child's to import
        pytest.skip("the stack native calls FlashInfer, which is not installed here (samples/models/requirements-flashinfer.txt): uncovered, not passing")


@pytest.mark.L0
def test_the_native_stack_with_an_empty_kernel_binary_directory_is_a_skip_that_names_no_path(tmp_path):
    """``--stack native`` with the library's directory variable naming an empty directory: the run ends in ``[00] SKIP``
    while it is planned, by the sentence of the installation it finds -- with the kernel-binary package installed, that
    the library would read the directory in the place of the package; without it, that the directory does not hold the
    kernel binaries.  Exit code 0, standard output empty, nothing loaded, the directory's path on neither stream, and
    the directory still empty: nothing was fetched into it."""
    _needs_the_library()
    empty = tmp_path / "kernel_binaries"
    empty.mkdir()
    code, out, written = _fresh_interpreter(["--stack", "native"], environment={"FLASHINFER_CUBIN_DIR": str(empty)})
    verdict = [line for line in out.splitlines() if line.startswith("[00] ")][-1]
    frost_routing.measured("model sample qwen_dense 00_decode --stack native, an empty kernel-binary directory", verdict)
    if importlib.util.find_spec("flashinfer_cubin") is None:
        sentence = (
            "the directory FLASHINFER_CUBIN_DIR names does not hold the kernel binaries the library's table names for this call's data types, head width, "
            "paging and page on this GPU's class as their manifest lists them ("
        )
    else:
        sentence = "FLASHINFER_CUBIN_DIR is set and the kernel-binary package flashinfer-cubin is installed, "
    assert code == 0 and verdict.startswith("[00] SKIP ") and sentence in verdict, out[-6000:]
    assert written == "" and "\n[00] stage " not in out and "\nchecks" not in out, "nothing was loaded and nothing checked"
    assert str(tmp_path) not in out and list(empty.iterdir()) == [], "no path on either stream, and nothing written into the directory"


RECORDED_PROGRAM = """
import faulthandler, sys
faulthandler.dump_traceback_later({deadline}, exit=True)
sys.path.insert(0, {package_root!r})
import cudnn
assert cudnn.__file__ == {cudnn_file!r}, f'the child imported another cudnn: {{cudnn.__file__}}'
import torch
sys.path.insert(0, {models!r})
import model_samples
from model_samples import perf, tie_out
from model_samples.qwen_dense import QwenDense

SEQUENCES, PROMPT, STEPS, PLANTED = {sequences}, 26, 70, 31  # positions 0 to 95: across the first page boundary of the attention cache


def say(text):
    print(text, file=sys.stderr, flush=True)


failed = []


def hold(name, held, text):
    say(f"  {{name:<10}} {{text}}    {{'PASS' if held else 'FAIL'}}")
    if not held:
        failed.append(name)


model = QwenDense.reduced()
text = model_samples.TextIO.synthetic(model.geometry.vocab)
common = dict(sequences=SEQUENCES, stack="native", device="cuda:0", capacity=PROMPT + STEPS)  # the request fills the state: no position is left
eager = model_samples.Session(model, text, **common)
session = model_samples.Session(model, text, share=eager, capture=True, **common)
try:
    for note in dict.fromkeys(eager.resolve().announcements()):
        say(f"[recorded] planning: {{note}}")
    say(eager.plan_table([session]))
except NotImplementedError as error:
    say(f"[recorded] SKIP {{error}}")
    raise SystemExit(0)
say(f"[recorded] {{model_samples.check_environment().header()}}")
if session.recordable():  # a class on which a block of the stack is on a fallback row: its step is not recorded there
    say(f"[recorded] SKIP the step of the stack native is not recorded on this GPU: {{session.recordable()}}")
    raise SystemExit(0)
eager.load()
session.load()
generator = torch.Generator().manual_seed(0)
prompts = [torch.randint(0, model.geometry.vocab, (PROMPT,), generator=generator, dtype=torch.int32) for _ in range(SEQUENCES)]


def request(of, between=None, steps=STEPS):
    # a prompt, its first new token and ``steps`` steps, each ended by its host read: (the reads, ids fed, ids chosen, logits, end state)
    of.reset(fresh=False)
    ids = of.prefill(prompts)
    reads = [of.read(ids)]
    for index in range(steps):
        if between is not None:
            between(index)
        ids = of.step(ids)
        reads.append(of.read(ids))
    torch.cuda.synchronize()
    count = of.record.count
    return reads, of.record.inputs[:count].clone(), of.record.outputs[:count].clone(), of.record.logits[:count].clone(), perf.state_digest(of.state)


def equal(a, b):
    return a[0] == b[0] and all(tie_out.bytes_equal(x, y) for x, y in zip(a[1:4], b[1:4])) and a[4] == b[4]


recorded = session.recorded
hold("recorded", session.python_reason == "" and recorded is not None and "capture" in session.stages.seconds, f"the step was recorded by the load ({{session.python_reason or 'a stage of its own'}})")
say(f"[recorded] the sequence holds, as this installation states it: {{recorded.launches}}; its own pool {{recorded.pool_bytes}} bytes")
zero = perf.state_digest(session.state)
session.reset(fresh=True)
before = perf.state_digest(session.state)
session.record_step()
torch.cuda.synchronize()
recorded = session.recorded
hold("nothing", before == zero == perf.state_digest(session.state) and recorded.replays == 0, "recording the step launched nothing: the state holds the bytes it held")
want = request(eager)
again = request(eager)
hold("python", equal(want, again), f"the request launched from Python twice: the ids of {{len(want[0])}} reads, {{want[3].shape[0]}} steps of ids and logits, the end state")
got = [request(session) for _ in range(3)]
steps = want[3].shape[0]
hold("replayed", all(equal(want, each) for each in got) and recorded.replays == 3 * steps, f"three replayed requests against it, {{steps}} steps each: the ids of every read, the ids fed and chosen, the logits byte for byte, the end-state digest")
cursor = recorded.cursor
hold("tail", cursor.holds() and cursor.position == session.state.position == steps and got[-1][0][-1] == cursor.ids.tolist(), f"behind a request the ids buffer holds the last chosen ids and the position rows stand at position {{cursor.position}}")


def planted(index):
    if index == PLANTED - PROMPT:  # in front of the step at position PLANTED: the length the attention reads, one too small from there on
        cursor.lengths.sub_(1)


wrong = request(session, planted)
differs = [step for step in range(steps) if not tie_out.bytes_equal(wrong[3][step], want[3][step])]
hold("planted", differs[:1] == [PLANTED] and wrong[0][: PLANTED - PROMPT + 1] == want[0][: PLANTED - PROMPT + 1], f"a length one too small from position {{PLANTED}} on: the logits differ first at step {{differs[:1]}}, {{len(differs)}} steps differ")
hold("restored", equal(want, request(session)), "the next request, its cursor set from the tables at its first replay, is the request again")
# past the capacity: the step is refused and nothing is launched
ids, replays, position = recorded.next_ids, recorded.replays, session.state.position
try:
    session.step(ids)
    refused = ""
except ValueError as error:
    refused = str(error)
hold("capacity", "positions per sequence" in refused and (recorded.replays, session.state.position) == (replays, position), f"a replay past the capacity is refused and launches nothing ({{refused[:60]}})")
# a pass the installation declines: a block that reads the host inside the step
kind = type(session.plan["token_choice"])
call = kind.__call__


def reads_the_host(self, logits):
    out = call(self, logits)
    out.tolist()
    return out


session.reset(fresh=True)
stream = torch.cuda.current_stream()
kind.__call__ = reads_the_host
try:
    reason = session.record_step()
finally:
    kind.__call__ = call
say(f"[recorded] {{reason}}")
declined = reason.startswith("the capture of the step was declined on this installation (") and session.recorded is None
hold("declined", declined and torch.cuda.current_stream() == stream, "a pass with a host read inside is declined, no sequence is kept and PyTorch's current stream is the one it was")
hold("from python", equal(want, request(session)) and session.recorded is None, "the session then launches from Python and computes the request")
session.close()
say("[recorded] FAIL" if failed else "[recorded] PASS")
raise SystemExit(1 if failed else 0)
"""


@pytest.mark.L0
@pytest.mark.parametrize("sequences", [1, 4])
def test_a_recorded_request_equals_the_same_request_from_python(sequences):
    """One decode step of the stack ``native`` recorded as a launch sequence and replayed for a request, against the same
    request launched from Python (``Session(capture=True)`` beside a session without it, on one load): the reduced seeded
    model, a prompt of 26 tokens and 70 steps behind it, across the first page boundary of the attention cache, in a
    state of exactly those 96 positions.  A small program in a fresh interpreter holds, at 1 and at 4 sequences:

    * recording the step launches nothing: the state holds the bytes it held;
    * three replayed requests give the ids of every host read, the ids fed to and chosen by every step, the logits of
      every step byte for byte and the end-state digest of the request launched from Python;
    * behind a request the ids buffer holds the last chosen ids and the position rows stand at the last position;
    * a planted difference -- the length the attention reads, one too small from one position on -- moves the logits
      first at that step; the next request is the request again;
    * a replay past the capacity is refused and launches nothing;
    * a pass with a host read inside it is declined: no sequence is kept, PyTorch's current stream is the one it was,
      and the session computes the request from Python.

    The stack calls a library: skipped where that is absent, by the program's own line where the plan declines the GPU
    or a block of the stack is on a fallback row.  A skipped cell is uncovered, not passing."""
    _needs_the_library()
    key = f"model sample qwen_dense: a recorded decode step against the step launched from Python, {sequences} sequence(s), stack native"
    out = _program(RECORDED_PROGRAM, "[recorded]", key, "run of a recorded step", unobserved="RAN (never printed: ", sequences=sequences)
    lines = out.splitlines()
    for name in ("recorded", "nothing", "python", "replayed", "tail", "planted", "restored", "capacity", "declined", "from python"):
        (line,) = [line for line in lines if line.startswith(f"  {name} ")]
        assert line.endswith("    PASS"), line
    assert "the logits differ first at step [31]" in out and "three replayed requests against it, 96 steps each" in out
    stated = next(line for line in lines if line.startswith("[recorded] the sequence holds, as this installation states it: "))
    frost_routing.measured(key + ": the recorded sequence", stated[len("[recorded] ") :])


CORE_PROGRAM = """
import faulthandler, os, sys
faulthandler.dump_traceback_later({deadline}, exit=True)
os.environ.setdefault("KINETO_LOG_LEVEL", "6")  # as the sample's scripts set it: the profiler's own lines stay off standard error
sys.path.insert(0, {package_root!r})
import cudnn
assert cudnn.__file__ == {cudnn_file!r}, f'the child imported another cudnn: {{cudnn.__file__}}'
import torch
sys.path.insert(0, {models!r})
import model_samples
from types import SimpleNamespace
from model_samples import tie_out, timing
from model_samples.providers import torch_ref
from model_samples.qwen_dense import QwenDense
from model_samples.state import DecodeState, KvCache

DEVICE, PAGE, DEFAULT_LENGTH, BF16 = "cuda:0", 64, 2051, torch.bfloat16
model = QwenDense.reduced()
failed = []


def say(text):
    print(text, file=sys.stderr, flush=True)


def hold(name, held, text):
    say(f"  {{name:<52}} {{text}}    {{'PASS' if held else 'FAIL'}}")
    if not held:
        failed.append(name)


def drawn(sequences, tokens, live, limit, pool_seed, query_seed, scale=1.0, pages=None):
    # the block of the stack native for a step of this shape, drawn queries, a paged cache of `live` positions per sequence
    # whose every row is drawn, and the position the step starts at
    text = model_samples.TextIO.synthetic(model.geometry.vocab)
    session = model_samples.Session(model, text, sequences=sequences, tokens=tokens, stack="native", device=DEVICE, capacity=live, max_model_len=limit)
    block = session.resolve()["attention.core"]
    block.first_use(sequences, tokens)
    shapes, device = model.state_shapes(), torch.device(DEVICE)
    cache = DecodeState(("full_attention",), sequences, live, shapes, device, tokens=tokens, page=PAGE, pages=pages, max_model_len=limit).layers[0]
    cache.pool.copy_(torch.randn(cache.pool.shape, generator=torch.Generator("cpu").manual_seed(pool_seed)).to(BF16))
    rows = sequences * tokens, model.geometry.attn_heads, model.geometry.attn_head_dim
    queries = torch.randn(rows, generator=torch.Generator("cpu").manual_seed(query_seed)) * scale
    return block, queries.to(BF16).to(DEVICE), cache, live - tokens


def judged(block, queries, cache, position, twin=None):
    # one call of the block, judged in the band of its row the way a run judges a call; `twin`: a twin with a defect in the
    # place of the correct one.  With the statistics, the names of the library's attention kernels the profiler recorded
    correct = torch_ref.AttentionBound(None, None, None, {{}})
    reference, before, outputs, records = correct(queries, cache, position), cache.pool.clone(), [], []

    def step(index):
        outputs.append(block(queries, cache, position))
        outputs[0].float().sum().item()  # the host read a traced step ends in

    timing.trace_steps(step, 1, DEVICE, records=records)
    if not outputs:  # no kernel activity records on this build: the call alone
        step(0)
    names = sorted({{record.name for record in records if record.name.startswith("fmha")}})
    stats = tie_out.BlockStats("attention_bf16_weights")
    tie_out.BlockTap._judge(stats, (outputs[0],), (reference,), [cache.pool], [before], 0, twin or correct)
    return stats, ", ".join(names) or "kernel not observed"


def cases():
    # one sequence at the maximum length the scripts default to, in a pool of 39 pages: two live rows at a step of one
    # row -- the queries of the third draw four times as large --, six live rows at a step of six
    for pool_seed, query_seed, scale, tokens in ((1002, 2002, 1.0, 1), (2002, 4002, 1.0, 1), (3002, 6002, 4.0, 1), (3012, 4012, 1.0, 6), (6012, 8012, 1.0, 6)):
        live = 2 if tokens == 1 else 6
        stats, kernel = judged(*drawn(1, tokens, live, DEFAULT_LENGTH, pool_seed, query_seed, scale, pages=39))
        hold(f"1 x {{tokens}} rows, {{live}} live, seeds {{pool_seed}} / {{query_seed}}", stats.ok and stats.worst_share <= 1.0, f"{{stats.line()}}; {{kernel}}")


def forms():
    # a step shape for each kernel form of the library: sequences x tokens per sequence at a maximum length, 100 live rows
    # per sequence (a full page and 36 rows of a second), 37 where the maximum length is 40
    for sequences, tokens, limit in ((1, 1, DEFAULT_LENGTH), (1, 6, DEFAULT_LENGTH), (7, 1, DEFAULT_LENGTH), (1, 8, DEFAULT_LENGTH), (16, 5, DEFAULT_LENGTH), (4, 6, DEFAULT_LENGTH), (1, 1, 512), (1, 6, 40)):
        stats, kernel = judged(*drawn(sequences, tokens, min(100, limit - 3), limit, 7001 + sequences, 9001 + tokens))
        hold(f"{{sequences}} x {{tokens}} rows at a maximum length of {{limit}}", stats.ok and stats.worst_share <= 1.0, f"{{stats.line()}}; {{kernel}}")


def arms():
    # every arm must FAIL.  One sequence, 100 live rows: a full page and 36 rows of a second
    live = 100
    block, queries, cache, position = drawn(1, 1, live, DEFAULT_LENGTH, 7001, 9001)
    stats, kernel = judged(block, queries, cache, position)
    hold("no defect", stats.ok, f"{{stats.line()}}; {{kernel}}")
    table, lengths, blocks = cache.table, cache.table.lengths.clone(), cache.table.blocks.clone()

    def longer(by):
        table.lengths[position] += by

    def exchanged():
        table.blocks[0, :2] = table.blocks[0, :2].flip(0)

    def unscaled():  # the library's own default for the scale of the scores
        library = sys.modules["flashinfer"].decode
        real = library.trtllm_batch_decode_with_kv_cache
        library.trtllm_batch_decode_with_kv_cache = lambda **passed: real(**{{name: value for name, value in passed.items() if name != "bmm1_scale"}})
        return lambda: setattr(library, "trtllm_batch_decode_with_kv_cache", real)

    in_the_call = (
        ("the lengths one too long: a masked row let in", lambda: longer(1)),
        ("the lengths one too short: the newest row dropped", lambda: longer(-1)),
        ("the last attended page exchanged with a full one", exchanged),
        ("the scores at the library's default scale", unscaled),
    )
    for name, put in in_the_call:
        undo = put()
        stats, _ = judged(block, queries, cache, position)
        table.lengths.copy_(lengths), table.blocks.copy_(blocks), undo and undo()
        hold(name, not stats.ok and stats.finite and stats.outside > 0, stats.line())

    def twin_on(rows, values, attended):  # the terms of a twin that reads `values`, row `r` of the step attending to `attended[r]` positions
        terms = []
        for row, count in enumerate(attended):
            twin = torch_ref.AttentionBound(None, None, None, {{}})
            twin(rows[row : row + 1], KvCache(cache.keys(count, BF16), values[:, :, :count], 1), count - 1)
            terms.append(twin.bound)
        return SimpleNamespace(bound=tuple(torch.cat([term[part] for term in terms]) for part in (0, 1)))

    values = cache.values(live + 1, BF16)
    newest, oldest = values.clone(), values.clone()
    newest[:, :, live - 1], oldest[:, :, 0] = values[:, :, live], values[:, :, 1]
    for name, wrong in (("a twin: the newest value row read from the next slot", newest), ("a twin: the value row of position 0 read from slot 1", oldest)):
        stats, _ = judged(block, queries, cache, position, twin_on(queries, wrong, [live]))
        hold(name, not stats.ok and stats.finite and stats.outside > 0, stats.line())
    block, queries, cache, position = drawn(1, 6, live, DEFAULT_LENGTH, 7001, 9001)
    stats, _ = judged(block, queries, cache, position, twin_on(queries, cache.values(live, BF16), [live] * 6))
    hold("a twin: every row of a step of six attends to all rows", not stats.ok and stats.finite and stats.outside > 0, stats.line())
    # tables of 64-bit words are the adapter's own decline, at two sequences of two pages: the library would read them as 32-bit words
    block, queries, cache, position = drawn(2, 1, live, DEFAULT_LENGTH, 7001, 9001)
    for name, wide in (("lengths", dict(lengths=cache.table.lengths.long())), ("blocks", dict(blocks=cache.table.blocks.long()))):
        kept = {{field: getattr(cache.table, field) for field in wide}}
        vars(cache.table).update(wide)
        try:
            block(queries, cache, position)
            declined = "no error"
        except ValueError as error:
            declined = str(error)
        vars(cache.table).update(kept)
        hold(f"{{name}} of 64-bit words", "the kernel reads 32-bit words" in declined, f"the adapter declines: {{declined[:160]}}")


try:
    {{"cases": cases, "forms": forms, "arms": arms}}[{selected!r}]()
except NotImplementedError as error:
    say(f"[core] SKIP {{error}}")
    raise SystemExit(0)
say(f"[core] {{model_samples.check_environment().header()}}")
say("[core] FAIL" if failed else "[core] PASS")
raise SystemExit(1 if failed else 0)
"""


@pytest.mark.parametrize(
    "selected", [pytest.param("cases", marks=pytest.mark.L0), pytest.param("arms", marks=pytest.mark.L0), pytest.param("forms", marks=pytest.mark.L1)]
)
def test_the_librarys_attention_on_drawn_operands_is_inside_its_band_and_every_control_arm_fails(selected):
    """The block of the attention core on the stack ``native``, called on drawn operands and judged in the band of its
    row (``CORE_PROGRAM``, a fresh interpreter), where the library and its kernel binaries are installed.  ``cases``:
    five drawn cases at one sequence, each inside the band.  ``arms``: the same call with one thing wrong must be
    outside it -- in the call: the lengths one too long, one too short, the entry of the last attended page exchanged
    with a full one's, the scores at the library's default scale; in the twin: a value row read from the next slot, at
    the newest position and at position 0, and every row of a step of six attending to all rows -- and tables of 64-bit
    words are declined by the adapter.  ``forms``: a step shape for each kernel form of the library, each inside the
    band, the kernel the profiler recorded printed beside it."""
    _needs_the_library()
    out = _program(
        CORE_PROGRAM,
        "[core]",
        f"model sample qwen_dense: the library's attention on drawn operands, {selected}",
        "run of the attention core",
        selected=selected,
    )
    held = [line for line in out.splitlines() if line.startswith("  ") and line.endswith(("    PASS", "    FAIL"))]
    assert len(held) == {"cases": 5, "arms": 10, "forms": 8}[selected] and all(line.endswith("    PASS") for line in held), out[-6000:]
    if selected != "arms":  # the kernel the profiler recorded under each call, at the end of its line
        kernels = sorted({line.rsplit("; ", 1)[1].split("    ")[0] for line in held})
        frost_routing.measured(f"model sample qwen_dense: the library's attention on drawn operands, {selected}: kernels", "; ".join(kernels))


QUANTISE_PROGRAM = """
import faulthandler, os, sys
faulthandler.dump_traceback_later({deadline}, exit=True)
os.environ.setdefault("KINETO_LOG_LEVEL", "6")  # as the sample's scripts set it: the profiler's own lines stay off standard error
sys.path.insert(0, {package_root!r})
import cudnn
assert cudnn.__file__ == {cudnn_file!r}, f'the child imported another cudnn: {{cudnn.__file__}}'
import torch
sys.path.insert(0, {models!r})
import model_samples
from model_samples import tie_out, timing
from model_samples.providers import Quant, torch_ref
from model_samples.qwen_dense import QwenDense
from model_samples.weights import tile_scale_index

DEVICE, BF16, WIDTH = "cuda:0", torch.bfloat16, 5120
model = QwenDense.reduced()
failed = []


def say(text):
    print(text, file=sys.stderr, flush=True)


def hold(name, held, text):
    say(f"  {{name:<44}} {{text}}    {{'PASS' if held else 'FAIL'}}")
    if not held:
        failed.append(name)


def planned(stack, rows):
    # a session of `rows` sequences on `stack` and the block of the head's quantise in its plan, its first use made
    text = model_samples.TextIO.synthetic(model.geometry.vocab)
    session = model_samples.Session(model, text, sequences=rows, stack=stack, device=DEVICE, capacity=8)
    block = session.resolve()["output_head.quantize"]
    block.first_use(rows, 1)
    return session, block


def traced(block, x, quant):
    # one call of the block under the profiler: what it returned, and the records of its entry's kernel
    outputs, records = [], []

    def step(index):
        outputs.append(block(x, quant))
        outputs[0].codes.sum().item()  # the host read a traced step ends in

    timing.trace_steps(step, 1, DEVICE, records=records)
    if not outputs:  # no kernel activity records on this build: the call alone
        step(0)
    return outputs[0], [record for record in records if record.name.startswith("cudnn_kernel_frost_row_quantize_fp4_")]


def drawn(rows):
    # drawn rows under the seeded model's head scale and under one whose reciprocal is rounded: the block against the
    # quantisation at the kernel's rounding points, in the band of its row; and the launch against what the entry states
    _, block = planned("fe-entries", rows)
    facts = block.launch_facts(rows)
    for seed, scale in ((7, 8.0 / (448.0 * 6.0)), (11, 0.0205544)):
        x = (torch.randn((rows, WIDTH), generator=torch.Generator("cpu").manual_seed(seed + rows)) * 1.7).to(BF16).to(DEVICE)
        quant = Quant("nvfp4", scale)
        out, launches = traced(block, x, quant)
        twin = torch_ref.QuantizeBracket(None, None, None, {{}})
        expected = twin(x, quant)
        stats = tie_out.BlockStats("row_quantize_fp4")
        tie_out.BlockTap._judge(stats, out.tensors(), expected.tensors(), [], [], 0, twin)
        hold(f"{{rows}} rows, scale {{scale:.6g}}: the band", stats.ok and stats.outside == 0, stats.line())
        if launches:
            (launch,) = launches
            same = launch.name in facts.symbols and tuple(launch.shape) == tuple(facts.shape)
            hold(f"{{rows}} rows, scale {{scale:.6g}}: the launch", same, f"{{launch.name}} in grid {{launch.shape[0]}}, block {{launch.shape[1]}}, shared memory {{launch.shape[2]}} B")
        else:
            hold(f"{{rows}} rows, scale {{scale:.6g}}: the launch", True, "kernel not observed")


def exact():
    # a power-of-two scale and power-of-two block scales, rows that hold every rounding tie of the code grid at both signs:
    # the entry's codes are e2m1 of the exact quotient with ties to the even code, its block scales those powers of two
    rows, scale = 4, 2.0**-5
    even = {{0.25: 0, 0.75: 2, 1.25: 2, 1.75: 4, 2.5: 4, 3.5: 6, 5.0: 6}}  # the even code of the two around each midpoint
    x, want = torch.zeros((rows, WIDTH)), torch.zeros((rows, WIDTH), dtype=torch.uint8)
    bytes_wanted, ties = torch.zeros((rows, WIDTH // 16), dtype=torch.uint8), 0
    for r in range(rows):
        for block in range(WIDTH // 16):
            exponent = (block + r) % 9 - 6
            unit, first = 2.0**exponent, block * 16
            bytes_wanted[r, block] = (exponent + 5 + 7) * 8  # the E4M3 byte of unit x 32: a power of two, mantissa zero
            x[r, first], want[r, first] = 6.0 * unit, 7
            x[r, first + 1], want[r, first + 1] = 3.0 * unit, 5
            for j, middle in enumerate(even):
                for sign, at in ((1.0, 2 + 2 * j), (-1.0, 3 + 2 * j)):
                    x[r, first + at], want[r, first + at] = sign * middle * unit, even[middle] + (8 if sign < 0 else 0)
                    ties += 1
    _, block = planned("fe-entries", rows)
    out = block(x.to(BF16).to(DEVICE), Quant("nvfp4", scale))
    codes, scales = (tensor.cpu() for tensor in out.tensors())
    nibbles = torch.stack((codes & 0xF, codes >> 4), dim=-1).flatten(-2)
    off = int((nibbles != want).sum())
    hold("exact cells: the codes", off == 0 and ties == rows * (WIDTH // 16) * 14, f"{{ties:,}} ties judged, {{off:,}} codes not e2m1 of the exact quotient with ties to even")
    hold("exact cells: the block scales", torch.equal(scales, bytes_wanted), f"{{int((scales != bytes_wanted).sum()):,}} of {{scales.numel():,}} not their power of two")


def handover():
    # the two blocks of a session on the stack native, block by block: the head's logits with the padded rows of the scale
    # buffer as the quantise returned them, set to zero, and set to a pattern that is no finite value
    for rows in (1, 129):
        text = model_samples.TextIO.synthetic(model.geometry.vocab)
        session = model_samples.Session(model, text, sequences=rows, stack="native", device=DEVICE, capacity=8)
        session.plan_table()
        session.load()
        quantise, head, weights = session.plan["output_head.quantize"], session.plan["output_head"], session.weights
        x = (torch.randn((rows, WIDTH), generator=torch.Generator("cpu").manual_seed(100 + rows)) * 1.7).to(BF16).to(DEVICE)
        q = quantise(x, weights.head_quant)
        padded = torch.ones(q.tiled_scales.numel(), dtype=torch.bool, device=DEVICE)
        padded[tile_scale_index(rows, WIDTH // 16, DEVICE)] = False
        as_returned = head(q, weights.head).clone()
        results = []
        for pattern in (0x00, 0x7F):
            q.tiled_scales.view(-1)[padded] = pattern
            results.append(head(q, weights.head).clone())
        same = all(torch.equal(as_returned.view(torch.int16), other.view(torch.int16)) for other in results)
        finite = bool(torch.isfinite(as_returned.float()).all())
        hold(f"{{rows}} rows: the head behind the quantise", same and finite, f"{{int(padded.sum()):,}} padded scale bytes as returned, at zero and at 0x7F: {{as_returned.numel():,}} logits, the same bytes" if same else "the logits moved with the padded rows of the scale buffer")


try:
    selected = {selected!r}
    if selected.startswith("drawn"):
        drawn(int(selected.split()[1]))
        if selected == "drawn 1":
            exact()
    else:
        handover()
except NotImplementedError as error:
    say(f"[quantise] SKIP {{error}}")
    raise SystemExit(0)
say(f"[quantise] {{model_samples.check_environment().header()}}")
say("[quantise] FAIL" if failed else "[quantise] PASS")
raise SystemExit(1 if failed else 0)
"""


@pytest.mark.parametrize(
    "selected", [pytest.param("drawn 1", marks=pytest.mark.L0), pytest.param("drawn 6", marks=pytest.mark.L1), pytest.param("drawn 1024", marks=pytest.mark.L1)]
)
def test_the_head_quantise_on_drawn_operands_is_inside_its_band_and_launches_what_its_entry_states(selected):
    """The block of the output head's quantise on its entry, called on drawn operands (``QUANTISE_PROGRAM``, a fresh
    interpreter): at 1, 6 and 1,024 rows, under the seeded model's head scale and under one whose reciprocal is rounded,
    every block scale the byte of the kernel's rule and every code inside its bracket, and the one launch of a call in
    the shape and under the symbol the entry states.  In the cell of 1 row also the exact cells: under a power-of-two
    scale with power-of-two block scales, on four rows that hold every rounding tie of the code grid at both signs, the codes are
    ``e2m1`` of the exact quotient with ties to the even code -- written out in the program, not taken from the twin,
    whose low code is another byte at three of the seven midpoints."""
    if not torch.cuda.is_available():  # decided here, before a child is started
        pytest.skip("needs a CUDA GPU")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the row of the output head's quantise claims compute capability 10.x")
    out = _program(
        QUANTISE_PROGRAM,
        "[quantise]",
        f"model sample qwen_dense: the head's quantise on drawn operands, {selected}",
        "run of the head's quantise",
        selected=selected,
    )
    held = [line for line in out.splitlines() if line.startswith("  ") and line.endswith(("    PASS", "    FAIL"))]
    assert len(held) == (6 if selected == "drawn 1" else 4) and all(line.endswith("    PASS") for line in held), out[-6000:]
    if selected == "drawn 1":
        assert "17,920 ties judged, 0 codes not e2m1 of the exact quotient with ties to even" in out, "every tie of the built rows was judged"
    # what the profiler recorded under each call, or that it recorded nothing: a launch that was not observed is uncovered, and the record says so
    launches = sorted({line.split(": the launch", 1)[1].rsplit("    ", 1)[0].strip() for line in held if ": the launch" in line})
    frost_routing.measured(f"model sample qwen_dense: the head's quantise on drawn operands, {selected}: the launch", "; ".join(launches))


@pytest.mark.L0
def test_the_heads_logits_do_not_move_with_the_padded_rows_of_the_scale_buffer():
    """The hand-over of the stack ``native``, block by block: the quantise's entry writes the block scales of the live rows
    into a buffer of whole bands of 128 rows that the adapter does not initialise, and the library's head multiply is
    handed that buffer as it is.  At 1 row and at 129 rows the head's logits are the same bytes with the padded rows as
    the adapter returned them, set to zero, and set to a pattern that is no finite value: nothing reads them."""
    _needs_the_library()
    out = _program(
        QUANTISE_PROGRAM, "[quantise]", "model sample qwen_dense: the head behind the quantise's scale buffer", "run of the hand-over", selected="handover"
    )
    held = [line for line in out.splitlines() if line.startswith("  ") and line.endswith(("    PASS", "    FAIL"))]
    assert len(held) == 2 and all(line.endswith("    PASS") for line in held), out[-6000:]


@pytest.mark.L1
def test_the_memory_sentence_holds_what_a_run_of_85_sequences_takes(sample):
    """``--stack fe-entries --sequences 85`` in a fresh interpreter, on compute capability 10.x: a run of many sequences holds,
    beyond its weights and its state, what its checks keep for every sequence, and the memory sentence counts it.  The run
    passes its checks; PyTorch's peak of allocated bytes is below the sentence; and it is within 5 % of the weights the
    plan holds, the state and what the family counts for the checks (``check_row_bytes``), formed here from the model's own
    functions for the plan of this GPU.  This cell is what holds the judge's own tensors -- two fp64 copies of a pool,
    their difference and its square -- to the count: they are alive inside one call, where no host case looks.
    The run takes about 8 GiB of device memory, so the cell is L1: at L0 the count is held by the host cases alone, and
    its largest term on a GPU is uncovered there.  Skipped by name on a class the stack's entries do not claim and
    where the sentence declines the GPU."""
    import re

    if not torch.cuda.is_available():  # decided here, before a child is started
        pytest.skip("needs a CUDA GPU")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the runs this count is held to are of the rows of compute capability 10.x: uncovered on this class, not passing")
    start = time.perf_counter()
    code, out, written = _fresh_interpreter(["--stack", "fe-entries", "--sequences", "85"])
    tagged = [line for line in out.splitlines() if line.startswith("[00] ")]
    verdict = tagged[-1] if tagged else "no output"
    frost_routing.measured("model sample qwen_dense 00_decode --stack fe-entries --sequences 85", f"{verdict}; {time.perf_counter() - start:.0f} s")
    if verdict.startswith("[00] SKIP"):
        assert " FAIL" not in out and "[00] ERROR" not in out, out[-6000:]  # a skip never covers a failed check
        pytest.skip(verdict)
    unobserved = verdict.startswith("[00] RAN fe-entries stack (kernels not observed: ")
    assert code == 0 and (verdict == "[00] PASS" or unobserved) and " FAIL" not in out, out[-6000:]
    assert written == "", f"a seeded run writes nothing to standard output; it holds\n{written[-3000:]}"
    sentence = next(line.strip() for line in out.splitlines() if line.strip().startswith("this stack needs about "))
    assert "what they hold for the 84 sequences beyond the first); " in sentence, sentence
    need = float(sentence.split("this stack needs about ", 1)[1].split(" GiB", 1)[0])
    peak = float(out.split("GPU memory: PyTorch peak ", 1)[1].split(" GiB allocated", 1)[0])
    assert 0.0 < peak < need, f"PyTorch held {peak} GiB at its peak; the memory sentence states {need} GiB"
    # the arithmetic of the sentence's parts, without its allowances: the weights and the state of this GPU's plan, and the count
    import model_samples
    from model_samples.qwen_dense import QwenDense

    request = re.search(r"decoding: (\d+) prompt tokens through the decode step, then (\d+) new tokens", out)
    capacity = int(request.group(1)) + int(request.group(2))  # the positions of the run: its prompt and its new tokens
    model = QwenDense.reduced(2)
    text = model_samples.TextIO.synthetic(model.geometry.vocab)
    session = model_samples.Session(model, text, sequences=85, stack="fe-entries", device=f"cuda:{torch.cuda.current_device()}", capacity=capacity)
    session.resolve()  # the tables of this GPU: nothing is built or loaded
    held = model.copy_bytes(session.uncopied_weights()) + model.stored_bytes(session.stored_weights())
    page = session.cache_page()
    state = model.state_bytes(85, capacity, **({} if page is None else {"page": page}))
    counted = (held + state + model.check_row_bytes(85, capacity)) / 2**30
    frost_routing.measured(
        "model sample qwen_dense 00_decode --stack fe-entries --sequences 85: memory",
        f"the sentence {need} GiB; PyTorch's peak {peak} GiB; counted {counted:.2f} GiB",
    )
    assert abs(peak / counted - 1.0) <= 0.05, (
        f"PyTorch held {peak} GiB at its peak; the weights held, the state and what the checks hold are {counted:.2f} GiB by the count "
        f"({held / 2**30:.2f} + {state / 2**30:.2f} + {model.check_row_bytes(85, capacity) / 2**30:.2f})"
    )


@pytest.mark.L1
def test_four_layer_run_passes_its_own_checks(sample, capsys):
    out = _run(sample, capsys, ["--layers", "4"])
    assert "4 of 64 layers (3 linear-attention + 1 attention)" in out and "35 block calls per step" in out


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
common = dict(stack="fe-entries", device="cuda:0", capacity={capacity})
decode = model_samples.Session(model, text, sequences={sequences}, **common)
verify = model_samples.Session(model, text, sequences={sequences}, tokens={tokens}, share=decode, **common)
try:
    say(decode.plan_table([verify]))
    say(verify.resolve().table())
    say(census.table(verify.plan))
except NotImplementedError as error:
    say(f"[verify] SKIP {{error}}")
    raise SystemExit(0)
say(f"[verify] {{model_samples.check_environment().header()}}")
decode.load()
verify.load()
generator = torch.Generator().manual_seed(0)
ids = decode.prefill([torch.randint(0, model.geometry.vocab, (8,), generator=generator, dtype=torch.int32) for _ in range({sequences})])
for _ in range({steps}):
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


def _verify(sequences, tokens, steps=23, capacity=40):
    """A decode session and a verify session of ``tokens`` tokens per sequence in a fresh interpreter (``VERIFY_PROGRAM``):
    the run's lines.  The decode run is a prompt of 8 tokens and ``steps`` steps behind it, in a state of ``capacity``
    positions."""
    key = f"model sample qwen_dense verify step, {sequences} x {tokens} tokens, stack fe-entries"
    return _program(VERIFY_PROGRAM, "[verify]", key, "verify run", sequences=sequences, tokens=tokens, steps=steps, capacity=capacity)


def _verify_asserts(out, sequences, tokens):
    several = f"{sequences} sequence{'s' if sequences > 1 else ''} x {tokens} tokens per sequence"
    assert f"stack fe-entries, {several}" in out, "the plan of a step of several tokens per sequence"
    assert (
        f"cudnn.GdnVerifyFused [PDL True, PDL_RELEASE True, TOKENS {tokens}]" in out
    ), "the fused verify entry in the form of its row, its tokens a field of it"
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
    if cc[0] == 10:  # the attention preparation on its entry: one launch for all rows of the step, each at its own position and slot
        prepared = [line for line in out.splitlines() if line.split()[:1] == ["attention.prepare"] and "[band: " in line]
        assert len(prepared) == 1 and prepared[0].endswith("the rest of the pool untouched  [band: attention_prepare]  PASS"), prepared
        assert " 0 query and 0 key elements outside their band; gate rows equal; value rows equal; " in prepared[0]
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
    """One sequence of six tokens.  The decode run's 32 tokens go through verify steps that keep 0 to 5 tokens in turn; every row is
    compared with the decode step of its token, the state of the kept tokens with the decode steps' state, the verify
    entry token by token with reference decode steps, and every launch of the traced verify steps with the plan.  Six
    rows are a row of the FP8 projection entry for the linear-attention layer's out-projection on compute capability
    10.x: on compute capability 10.0 the kernel table counts 15 launches of a step on exact rows."""
    out = _verify(1, 6)
    _verify_asserts(out, 1, 6)
    assert "tokens kept behind the first: 0, 1, 2, 3, 4, 5" in out, "every number of tokens is kept once"
    assert "batch invariance  not applicable at 1 sequence" in out
    if tuple(torch.cuda.get_device_capability()) == (10, 0):  # every entry row of the stack serves this class
        counted = "  per step: 15 launches on 9 exact rows, checked one by one in the wiring order; the launches of 5 fallback calls and of the host path "
        assert counted + "are attributed by family\n" in out, "the count of a step under the kernel table of the verify plan"


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


@pytest.mark.L1
def test_verify_steps_on_either_side_of_a_page_boundary_of_the_attention_cache_and_across_it():
    """Two sequences of six tokens over a decode run of 72 tokens: the verify steps keep 0 to 5 tokens in turn, so one of
    them starts at position 63 -- its first row the last of a sequence's first page, its other five on the second -- and
    keeps that first row alone: five rejected rows are written to the second page and written again by the step behind
    it.  The step in front of it fills positions 57 to 62 and keeps them all, and the one that starts at 52 rejects
    position 57, on the first page.  Every row is held against the decode step of its token and the kept state against
    the decode steps', as in the other verify cells; on compute capability 10.x the attention preparation's entry writes
    all twelve rows of a step in one launch, each to the slot of its position, and its pool is judged row by row."""
    schedule, start = [], 0  # the comparison's schedule, by its rule: each step starts behind the kept tokens of the one before it
    while start + 6 <= 72:
        schedule.append((start, len(schedule) % 6))
        start += schedule[-1][1] + 1
    assert schedule[16:19] == [(52, 4), (57, 5), (63, 0)] and len(schedule) == 21, "rejected rows on both pages, and a step across the boundary"
    out = _verify(2, 6, steps=64, capacity=80)
    _verify_asserts(out, 2, 6)
    assert "21 verify steps over 72 tokens of a run of 72 decode steps; tokens kept behind the first: " + ", ".join(str(kept) for _, kept in schedule) in out


@pytest.mark.L1
def test_a_verify_step_of_16_sequences_of_five_tokens_prepares_its_80_rows_in_one_launch():
    """16 sequences of five tokens, 80 rows of a step: past the rows of both projection entries, which are multiplies over
    copies there, and inside the 1,024 rows of the attention preparation's entry on compute capability 10.x -- one launch
    per step for all 80 rows, each sequence's rows on its own pages of the pool."""
    out = _verify(16, 5)
    _verify_asserts(out, 16, 5)
    if torch.cuda.get_device_capability()[0] == 10:
        stated = [
            line for line in out.splitlines() if line.strip().startswith("attention.prepare: cudnn.AttentionPrepareDecode [") and "; the entry states " in line
        ]
        assert len(stated) == 1 and "1 launch per step (1 call x 1 per call); the entry states grid (80, 28, 1), block (128, 1, 1)" in stated[0], stated


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
# the unit of the attention preparation's band, on this device: the exact power of two for every binade of a bf16 value (a
# power function on a CUDA device is not: a unit that is one fp64 unit low refuses an element exactly one unit away)
binades = range(-125, 128)
units = torch_ref.bf16_unit(torch.tensor([2.0**e for e in binades], dtype=torch.float64).to(device)).cpu()
hold("unit", units.tolist() == [2.0 ** (e - 7) for e in binades], f"the bf16 unit of the band of the attention preparation is the exact power of two in each of {{len(binades)}} binades")
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
    on the device, at the published widths and two sequences -- the layout a run builds where the attention preparation's
    entry serves, here under the chains alone.  A small program in a fresh interpreter runs the attention preparation and the attention core at
    positions 0 to 68 on a plain cache and on a paged one, the steps at 60 to 68 of the paged one under the profiler:

    * the queries, gates and attention outputs of every step are the plain cache's bytes;
    * the pool, read by index arithmetic written in the program, holds the plain cache's key and value row of every
      position, and the accessors return those rows;
    * the cos / sin table of the state is, row for row, what the chain forms for that position on this device;
    * the bf16 unit of the attention preparation's band is the exact power of two on this device, in every binade;
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
    for name in ("outputs", "pool", "accessors", "cos / sin", "unit", "verify"):
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


PREPARE_PROGRAM = """
import faulthandler, sys
faulthandler.dump_traceback_later({deadline}, exit=True)
sys.path.insert(0, {package_root!r})
import cudnn
assert cudnn.__file__ == {cudnn_file!r}, f'the child imported another cudnn: {{cudnn.__file__}}'
import torch
sys.path.insert(0, {models!r})
import model_samples
from model_samples import tie_out
from model_samples.providers import torch_ref
from model_samples.qwen_dense import QwenDense
from model_samples.state import DecodeState

PAGE, SEQUENCES, STEPS = 64, 2, 80  # positions 0 to 79: a whole first page of every sequence and 16 rows of its second


def say(text):
    print(text, file=sys.stderr, flush=True)


def same(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


failed = []


def hold(name, held, text):
    say(f"  {{name:<10}} {{text}}    {{'PASS' if held else 'FAIL'}}")
    if not held:
        failed.append(name)


model = QwenDense.reduced()
session = model_samples.Session(model, model_samples.TextIO.synthetic(model.geometry.vocab), sequences=SEQUENCES, stack="fe-entries", device="cuda:0", capacity=STEPS + 8)
try:
    say(session.plan_table())
    block = session.resolve()["attention.prepare"]
except NotImplementedError as error:
    say(f"[prepare] SKIP {{error}}")
    raise SystemExit(0)
if session.cache_page() != PAGE:
    say(f"[prepare] SKIP no row of the stack fe-entries writes a paged attention cache on this GPU (attention.prepare is {{block.entry}}): uncovered, not passing")
    raise SystemExit(0)
say(f"[prepare] {{model_samples.check_environment().header()}}")
session.load()
fed, kind = [], type(block)
entry_call = kind.__call__


def recorded(self, *operands):  # what the in-projection handed the entry at every step: its row, the position and the layer's weights
    if self is block:
        fed.append((operands[0].clone(), operands[2], operands[3]))
    return entry_call(self, *operands)


kind.__call__ = recorded
generator = torch.Generator().manual_seed(0)
ids = session.prefill([torch.randint(0, model.geometry.vocab, (8,), generator=generator, dtype=torch.int32) for _ in range(SEQUENCES)])
for _ in range(STEPS - 8):
    ids = session.step(ids)
kind.__call__ = entry_call
hold("calls", [position for _, position, _ in fed] == list(range(STEPS)), f"the entry was called once per step, at positions 0 to {{STEPS - 1}}: {{len(fed)}} calls")
# the twin, teacher-forced on those rows, fills a plain cache; it keeps the band of every key element it wrote
cache = next(layer for layer in session.state.layers if hasattr(layer, "pool"))
plain = DecodeState(("full_attention",), SEQUENCES, STEPS + 8, model.state_shapes(), torch.device("cuda:0")).layers[0]
bands = [torch_ref.attention_prepare_rounded(row, plain, position, 1, params, model.geometry.norm_eps)[2].k_band for row, position, params in fed]
# the pool, read by arithmetic written here: position p of sequence i is row p % 64 of page i * pages + p // 64, keys then values
pool, heads = cache.pool, cache.pool.shape[1]
pages, width = pool.shape[0] // SEQUENCES, pool.shape[-1] // 2
keys, values, behind = [0, 0], [0, 0], 0  # per page of a sequence
for sequence in range(SEQUENCES):
    for position in range(pages * PAGE):
        row = pool[sequence * pages + position // PAGE, :, position % PAGE]
        if position >= STEPS:
            behind += int(bool(row.any()))
            continue
        got, want = row[:, :width].double().cpu(), plain.k[sequence, :, position].double().cpu()  # compared on the host
        larger = torch.maximum(got.abs(), want.abs())
        unit = torch.ldexp(torch.ones_like(larger), torch.frexp(larger).exponent - 8)  # one bf16 unit at the larger magnitude, formed on the host
        keys[position // PAGE] += int(bool(((got - want).abs() <= unit + bands[position][sequence].cpu()).all()))
        values[position // PAGE] += int(same(row[:, width:], plain.v[sequence, :, position]))
# the unit this program holds the key rows with is its own.  It is formed on the host and asked of no device: there it is the exact
# power of two in every binade of a bf16 value, which the line below holds (the sample's own unit on this device is the other cell's)
binades = range(-125, 128)
units = torch.ldexp(torch.ones(len(binades), dtype=torch.float64), torch.tensor([e - 7 for e in binades]))
hold("unit", units.tolist() == [2.0 ** (e - 7) for e in binades], f"the bf16 unit of this comparison, formed on the host, is the exact power of two in each of {{len(binades)}} binades")
first, second = SEQUENCES * PAGE, SEQUENCES * (STEPS - PAGE)
hold("pool", keys == values == [first, second] and behind == 0,
     f"{{keys[0]}} of {{first}} key rows of the first pages and {{keys[1]}} of {{second}} of the second inside the entry's band around the twin's; "
     f"{{values[0]}} of {{first}} and {{values[1]}} of {{second}} value rows the twin's bytes; {{behind}} of the {{SEQUENCES * (pages * PAGE - STEPS)}} rows "
     f"behind position {{STEPS - 1}} hold anything")
reads = 0
for count in (1, 63, 64, 65, STEPS):
    for dtype in (torch.bfloat16, torch.float64):
        k = torch.stack([pool[sequence * pages + p // PAGE, :, p % PAGE, :width] for sequence in range(SEQUENCES) for p in range(count)])
        v = torch.stack([pool[sequence * pages + p // PAGE, :, p % PAGE, width:] for sequence in range(SEQUENCES) for p in range(count)])
        k, v = (t.view(SEQUENCES, count, heads, width).transpose(1, 2).to(dtype) for t in (k, v))
        reads += int(torch.equal(cache.keys(count, dtype), k) and torch.equal(cache.values(count, dtype), v))
hold("accessors", reads == 10, f"keys and values of the first 1, 63, 64, 65 and {{STEPS}} positions, in bf16 and in fp64: {{reads}} of 10 reads are the pool's rows")
session.trace(STEPS)
own = tie_out.run_checks(session, prompt_tokens=8)
say(own.text())
if failed or not own.ok:
    say("[prepare] FAIL")
elif own.unobserved:
    say(f"[prepare] RAN fe-entries stack (kernels not observed: {{own.unobserved}})")
else:
    say("[prepare] PASS")
raise SystemExit(1 if failed or not own.ok else 0)
"""


@pytest.mark.L0
def test_the_attention_preparations_entry_writes_the_pool_its_twin_would_across_a_page_boundary():
    """80 plain steps of the stack ``fe-entries`` at two sequences (``PREPARE_PROGRAM``, a fresh interpreter): the attention
    preparation's entry writes positions 0 to 79 of a paged attention cache, 64 on a sequence's first page and 16 on its
    second.  The program keeps the row the in-projection handed the entry at every step and runs the entry's twin on
    those rows over a plain cache; then it reads the pool by index arithmetic of its own and holds, on both pages, every
    value row to the twin's bytes and every key row inside the entry's band around the twin's -- with a bf16 unit of its
    own, formed on the host, where it is an exact power of two -- every row behind
    position 79 still zero, and the cache's accessors to the same arithmetic.  The session's own checks follow, over
    all 80 steps: the entry against its twin at every step, the kernel check on every traced step -- the steps past
    position 63 among them, where the attention core reads two pages of a sequence.  On a class whose plan has no such
    row the program says so and the cell is skipped: uncovered, not passing."""
    out = _program(
        PREPARE_PROGRAM, "[prepare]", "model sample qwen_dense: the attention preparation's entry across a page boundary", "run of the attention preparation"
    )
    lines = out.splitlines()
    for name in ("calls", "unit", "pool", "accessors"):
        (line,) = [line for line in lines if line.startswith(f"  {name} ")]
        assert line.endswith("    PASS"), line
    assert "128 of 128 key rows of the first pages and 32 of 32 of the second inside the entry's band" in out and "0 of the 96 rows behind position 79" in out
    (judged,) = [line for line in lines if line.split()[:1] == ["attention.prepare"] and "[band: " in line]
    assert judged.split()[1:3] == ["80", "calls"] and judged.endswith("the rest of the pool untouched  [band: attention_prepare]  PASS"), judged
    observed = [line for line in lines if line.startswith("  kernels  ")]
    frost_routing.measured(
        "model sample qwen_dense: the attention preparation's entry across a page boundary, the kernel check",
        "; ".join(line.split("    ")[-1] for line in observed),
    )


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
        '[02] --speculative-config (the default): read: method "mtp", num_speculative_tokens 5, use_local_argmax_reduction true.',
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
        "speculative against decode (1 sequence x 6 tokens per sequence, 10 speculative steps behind a prompt of 26 tokens; "
        "tokens kept behind the first: 0, 1, 2, 3, 4, 5, 0, 1, 2, 3; 2 layers)"
    )
    report = lines.index("acceptance            scheduled, not measured (reduction): 10 steps; kept 1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1")
    assert (
        report < against and "rate                  not printed while the drafter runs on reference blocks (kept tokens per step: the acceptance line)" in lines
    )
    assert any(
        line.startswith("first token ") and "each but the first behind one pass of the drafter over the position in front of it" in line for line in lines
    )
    assert any(line.startswith("steps (10) ") and line.endswith(" | 111 block calls per step") for line in lines), "five passes of 18 calls and the model's 21"
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
    assert "\n  kernels  " not in out and "\nkernels, planned" not in out, "the kernels of the default stack are checked on request"
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
    chains of the passes by family; under it the run says that the prompt's decode steps were not observed.  The
    kernel table of the verify plan ends in its count of a step."""
    if not torch.cuda.is_available():  # decided here, before a child is started
        pytest.skip("needs a CUDA GPU")
    out = _speculative_run(None, capsys, ["--stack", "fe-entries"], "fe-entries")
    lines = out.splitlines()
    assert "stack fe-entries, 1 sequence x 6 tokens per sequence" in out and "cudnn.GdnVerifyFused [PDL True, PDL_RELEASE True, TOKENS 6]" in out
    assert "cudnn.GdnDecodeFused " in out and "cudnn.GdnDecodeFused [" not in out, "the prompt's decode entry: a row of its plan; no form is printed for it"
    assert out.count("\nkernels, planned (decided before the load") == 1, "the kernel table of the verify plan"
    traced = [line for line in lines if line.startswith("traced pass (")]
    assert not traced or traced[0].endswith(" | behind the prompt's 26 decode steps, replayed untraced")
    if tuple(torch.cuda.get_device_capability()) == (10, 0):  # every entry row of the stack serves this class
        counted = "  per step: 15 launches on 9 exact rows, checked one by one in the wiring order; the launches of 5 fallback calls and of the host path "
        assert counted + "are attributed by family\n" in out, "the count of a step under the kernel table of the verify plan"
    observed = [line for line in lines if line.startswith("  kernels  ")]
    assert len(observed) == 1, "the kernel check of the traced steps ran, or said why it could not look"
    # the kernel check is of the verify steps: beside its verdict the run says what it did not look at
    assert "    the prompt's decode steps: 26 replayed in front of the traced window, their kernels not observed (the check is of the verify steps)" in lines
    assert "; the library launched " not in out, "no row of this stack is a library's"
    frost_routing.measured(
        "model sample qwen_dense 02_speculative_decode --stack fe-entries: the kernel check", "; ".join(line.split("    ")[-1] for line in observed)
    )
