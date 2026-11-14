# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke of the dense Qwen model sample (``samples/models/qwen_dense/00_decode.py``): the script is loaded by path and its
``main()`` is called as a user's default run -- the reduced seeded model, 1 and 4 sequences, the sample's own checks against its
fp64-accumulated reference (replay, every provider block teacher-forced, the greedy tokens, each sequence alone against the
same sequence in the batch).  The run must print ``[00] PASS`` and the environment line; on a GPU, a library version or a
free-memory budget the sample does not serve it prints ``[00] SKIP <requirement>`` and the cell is reported as skipped with
that sentence -- unless a check failed before it: a skip never covers a failure.  Each sequence count also reports its
traced pass (GPU busy, idle and launches per step from kernel activity records, or the reason there are none) and the
bytes of a step.  The
verdict line, the stack, the versions and the wall time are recorded through ``frost_routing.measured``, so the result -- or
the skip -- reaches the job log of a passing run.  L0 runs the two-layer model (one linear-attention + one attention layer), L1
the four-layer one (three + one).

The ``fe-entries`` cell runs the same default run on the stack ``fe-entries``: every cuDNN Frontend row of the stack ``native``
(the fused linear-attention step in its wait + release form; on compute capability 10.x the attention gate + quantise
entry, judged against its twin) with every other block on its fallback row -- no package beyond cuDNN Frontend and PyTorch.
That stack always checks its kernels: the profiler's record of every launch of the traced steps is held against the plan,
the entries' own launch facts and, where the family's kernel census has a section for the GPU, the recorded launch shapes.
Where the profiler's records cannot be had the sample ends in ``[00] RAN fe-entries stack (kernels not observed: <reason>)``:
the cell then passes on the other checks and the recorded line says that the kernels were not observed -- uncovered
there, not verified.

That cell runs the sample in a FRESH INTERPRETER, as a user does.  The kernel check compares the profiler's kernel names
with the symbols the entries state, and the profiler names a kernel wrongly after a kernel module was unloaded in the same
process: a kernel loaded where an unloaded one had been was recorded under the unloaded kernel's name (two raw CUDA
kernels loaded and unloaded in turn: about every second launch; none where every module stayed loaded; PyTorch 2.13.0).
The kernel that runs is the right one; its record's name is not.  A sample process unloads nothing, a test session does:
the cells of this module drop the sample's modules, and with them the entries of the stack that ran before.  Run in this
process after the default cell, the kernel check failed one run in three by the name of the fused step alone."""

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


@pytest.fixture()
def sample(monkeypatch):
    """The sample script as a module; its ``sys.path`` entry, every ``model_samples`` module and the default it sets for the
    profiler's log level leave with the test."""
    monkeypatch.setenv("KINETO_LOG_LEVEL", os.environ.get("KINETO_LOG_LEVEL", "6"))
    paths = list(sys.path)
    spec = importlib.util.spec_from_file_location("model_sample_qwen_dense_00_decode", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path[:] = paths
    for name in [n for n in sys.modules if n == "model_samples" or n.startswith("model_samples.")]:
        del sys.modules[name]


FE_CUBLAS = "by provider: cuDNN FE 1 | fallbacks 8 (cuBLAS over copies 4, PyTorch chains 4) | PyTorch 2"
#: with the gate entry (compute capability 10.x) one more block is cuDNN Frontend's and one reference chain fewer is a fallback
FE_ENTRIES = "by provider: cuDNN FE 2 | fallbacks 7 (cuBLAS over copies 4, PyTorch chains 3) | PyTorch 2"


CHILD_SECONDS = 1200  # the fresh interpreter's deadline: imports, the compiles of both sequence counts and the run, far above what they take


def _fresh_interpreter(arguments):
    """The sample script as a user runs it: in a fresh interpreter, which imports the ``cudnn`` package this process imported.
    Returns ``(exit code, stdout)``.  The child arms its own stack dump before the parent's deadline; where the deadline
    passes, the failure carries what the child had printed."""
    import cudnn

    package_root = str(Path(cudnn.__file__).resolve().parent.parent)
    child = (
        "import faulthandler, runpy, sys\n"
        f"faulthandler.dump_traceback_later({CHILD_SECONDS - 60}, exit=True)\n"
        f"sys.path.insert(0, {package_root!r})\n"
        "import cudnn\n"
        f"assert cudnn.__file__ == {cudnn.__file__!r}, f'the child imported another cudnn: {{cudnn.__file__}}'\n"
        f"sys.argv = [{str(SCRIPT)!r}, *{list(arguments)!r}]\n"
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
    )
    try:
        done = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=CHILD_SECONDS)
    except subprocess.TimeoutExpired as late:
        shown = [part.decode(errors="replace") if isinstance(part, bytes) else (part or "") for part in (late.stdout, late.stderr)]
        pytest.fail(f"the sample did not end within {CHILD_SECONDS} s in a fresh interpreter\n--- stdout\n{shown[0][-6000:]}\n--- stderr\n{shown[1][-6000:]}")
    if "[00] " not in done.stdout:  # the child never reached the sample's own lines: an import or an environment failure
        pytest.fail(
            f"the sample printed no verdict in a fresh interpreter (exit code {done.returncode})\n--- stdout\n{done.stdout[-3000:]}\n--- stderr\n{done.stderr[-6000:]}"
        )
    return done.returncode, done.stdout


def _run(sample, capsys, arguments, stack="fe-cublas", by_provider=FE_CUBLAS, unobserved=False):
    """Run the sample's ``main()`` and hold its output; with ``sample`` None, run the script in a fresh interpreter instead.
    ``unobserved`` accepts, beside ``[00] PASS``, the verdict of a run whose kernel check could not look (``RAN <stack> stack
    (kernels not observed: ...)``, exit code 0): the line is recorded as it is, so a CI job without the profiler's records
    shows as such in its log."""
    start = time.perf_counter()
    if sample is None:
        code, out = _fresh_interpreter(arguments)
    else:
        code = sample.main(arguments)
        out = capsys.readouterr().out
    seconds = time.perf_counter() - start
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
    compute capability 10.x, the gate + quantise entry inside its band against its twin; the kernel check on every launch
    of the traced steps; the census line, which claims nothing while a row is a fallback.  In a fresh interpreter: the
    kernel check reads kernel names from the profiler, which names them wrongly in a process that has unloaded a kernel
    module (the module docstring)."""
    if not torch.cuda.is_available():  # decided here, before a child is started
        pytest.skip("needs a CUDA GPU")
    gate = torch.cuda.get_device_capability()[0] == 10  # the gate entry serves compute capability 10.x
    out = _run(None, capsys, ["--stack", "fe-entries"], "fe-entries", FE_ENTRIES if gate else FE_CUBLAS, unobserved=True)
    assert "2 of 64 layers (1 linear-attention + 1 attention)" in out and "stack fe-entries" in out
    assert "cudnn.GdnDecodeFused [PDL True, PDL_RELEASE True]" in out, "the fused step in the target's form"
    assert out.count("\nkernels, planned (decided before the load") == 2 and out.count("\n  kernels  ") == 2, "the kernel check ran at both sequence counts"
    assert out.count(": NOT CLAIMED (") == 2 and ": EQUAL" not in out, "nothing is claimed while a row of the stack is a fallback"
    if gate:
        assert "cudnn.AttentionGateQuantDecode [THREADS 256, PDL_RELEASE True]" in out
        assert out.count("[band: gate_fp8]  PASS") == 2, "the gate entry was judged against its twin at both sequence counts"
        assert "attention.gate_quantize equal (gated)" in out or "REPORTED" in out
    observed = [line for line in out.splitlines() if line.startswith("  kernels  ")]
    frost_routing.measured("model sample qwen_dense 00_decode --stack fe-entries: the kernel check", "; ".join(line.split("    ")[-1] for line in observed))


@pytest.mark.L1
def test_four_layer_run_passes_its_own_checks(sample, capsys):
    out = _run(sample, capsys, ["--layers", "4"])
    assert "4 of 64 layers (3 linear-attention + 1 attention)" in out and "34 block calls per step" in out
