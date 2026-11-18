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
bytes of a step.  The
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
name of its call site; the attention layer's out-projection, and both beyond those rows, are the multiply over a copy."""

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
#: with the gate entry, the norm entry and the fused gate / up entry (compute capability 10.x) three more blocks are cuDNN Frontend's:
#: two reference chains and one multiply over a copy fewer are fallbacks
FE_ENTRIES = "by provider: cuDNN FE 4 | fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2) | PyTorch 2"
#: on compute capability 10.0 the in-projection is the FP8 projection entry's as well; the out-projection stays a multiply over a copy
FE_ENTRIES_CC100 = "by provider: cuDNN FE 5 | fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2) | PyTorch 2"


CHILD_SECONDS = 1200  # the fresh interpreter's deadline: imports, the compiles of both sequence counts and the run, far above what they take


def _fresh_interpreter(arguments):
    """The sample script as a user runs it: in a fresh interpreter, which imports the ``cudnn`` package this process imported.
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
        f"sys.argv = [{str(SCRIPT)!r}, *{list(arguments)!r}]\n"
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
    )
    try:
        done = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=CHILD_SECONDS)
    except subprocess.TimeoutExpired as late:
        shown = [part.decode(errors="replace") if isinstance(part, bytes) else (part or "") for part in (late.stdout, late.stderr)]
        pytest.fail(f"the sample did not end within {CHILD_SECONDS} s in a fresh interpreter\n--- stdout\n{shown[0][-6000:]}\n--- stderr\n{shown[1][-6000:]}")
    if "[00] " not in done.stderr:  # the child never reached the sample's own lines: an import or an environment failure
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


def _program(program, tag, key, what, **fields):
    """A small program that drives the helper layer in a fresh interpreter (``program``, formatted with ``fields``): the
    run's lines.  ``tag`` starts its own lines (``[verify]``), ``key`` names the run in the routing record, ``what`` in a
    failure.  Skips where the plan declines the GPU; fails where a check failed or standard output is not empty."""
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
    held = verdict == f"{tag} PASS" or verdict.startswith(f"{tag} RAN fe-entries stack (kernels not observed: ")
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


SPECULATIVE_PROGRAM = """
import faulthandler, sys
faulthandler.dump_traceback_later({deadline}, exit=True)
sys.path.insert(0, {package_root!r})
import cudnn
assert cudnn.__file__ == {cudnn_file!r}, f'the child imported another cudnn: {{cudnn.__file__}}'
import torch
sys.path.insert(0, {models!r})
import model_samples
from model_samples import census, loop, sampling, tie_out
from model_samples.qwen_dense import QwenDense, QwenDenseDrafter


def say(text):
    print(text, file=sys.stderr, flush=True)


drafted, new, tokens = {drafted}, 32, {drafted} + 1
model = QwenDense.reduced()
drafter = QwenDenseDrafter.seeded(model)
text = model_samples.TextIO.synthetic(model.geometry.vocab)
common = dict(stack="fe-entries", device="cuda:0", capacity=8 + new + 2 * drafted)
plain = model_samples.Session(model, text, **common)
session = model_samples.Session(model, text, draft_tokens=drafted, share=plain, drafter=drafter, **common)
try:
    say(plain.plan_table([session]))
    say(session.resolve().table())
    say(loop.PROMPT_ROWS)
    say(session.prefill_plan.table())
    say(loop.DRAFTER_ROWS.format(calls=drafter.calls_per_pass()))
    say(session.draft_plan.table())
    say(census.table(session.prefill_plan))
    say(census.table(session.plan, session.claim()))
except NotImplementedError as error:
    say(f"[speculative] SKIP {{error}}")
    raise SystemExit(0)
say(f"[speculative] {{model_samples.check_environment().header()}}")
plain.load()
session.load()
generator = torch.Generator().manual_seed(0)
prompts = [torch.randint(0, model.geometry.vocab, (8,), generator=generator, dtype=torch.int32)]
ids = plain.prefill(prompts)
stream = plain.read(ids)
for _ in range(new + drafted - 1):
    ids = plain.step(ids)
    stream += plain.read(ids)
out = session.read(session.prefill(prompts))
scheduled = []
while len(out) < new:  # the loop is the caller's: the step, the one host read, the count, the keep; the drafted tokens are scheduled
    length = len(scheduled) % tokens
    force = torch.tensor([tie_out.forced_drafts(stream[len(out):], length, drafted, model.geometry.vocab)], dtype=torch.int32, device="cuda:0")
    ((window, greedy),) = session.read(session.speculate(force))
    kept = sampling.accepted(window, greedy)
    session.accept(kept)
    out += greedy[: kept + 1]
    scheduled.append(length)
same = out[:new] == stream[:new]
say(f"[speculative] the {{new}} new tokens are the plain session's: {{same}}; kept {{session.record.accepted}}")
against = tie_out.speculative_checks(session, plain, scheduled)
say(against.text())
session.trace(32)
say(session.report().text())
own = tie_out.run_checks(session, prompt_tokens=8)
say(own.text())
passed = same and against.ok and own.ok
if passed and own.unobserved:
    say(f"[speculative] RAN fe-entries stack (kernels not observed: {{own.unobserved}})")
else:
    say("[speculative] PASS" if passed else "[speculative] FAIL")
raise SystemExit(0 if passed else 1)
"""


@pytest.mark.L1
def test_a_session_with_a_drafter_speculates_behind_its_prompt_on_the_fe_entries_stack():
    """One session for a prompt and the steps of speculative decoding behind it, on real launches: one sequence, five
    drafted tokens, the seeded drafter of the seeded model.  The prompt goes through decode steps on token 0 of the state
    of six tokens per sequence -- the decode entry on that layout -- each but the first behind one pass of the drafter;
    every step behind it is the drafter's five passes on reference blocks, one verify step and one host read, with the
    verified drafted tokens scheduled to be kept 0 to 5 in turn (``speculate(force=)``).  The output is the plain
    session's; the comparison with decode steps holds its lines, the drafter's two among them; the traced pass holds
    whole steps behind the replayed prompt, and the kernel check holds them: the passes' gathers and chains, then the
    verify plan.  In a fresh interpreter, with both streams held: a run of the helper layer writes nothing to standard
    output."""
    key = "model sample qwen_dense session with a drafter, 1 x 6 tokens, stack fe-entries"
    out = _program(SPECULATIVE_PROGRAM, "[speculative]", key, "run of a session with a drafter", drafted=5)
    lines = out.splitlines()
    assert (
        "the drafter: 18 block calls a pass; the first pass of a step behind a verify step runs at S x T rows, every other pass "
        "and every pass over the prompt at S rows; every row of this table serves any row count"
    ) in lines
    assert "with its drafter, one after the other" in out and "draft.token_choice" in out, "the memory sentence and the tables name the drafter"
    assert "the prompt of this session goes through decode steps, one per prompt token, on these rows:" in lines
    assert "stack fe-entries, 1 sequence x 6 tokens per sequence" in out and "cudnn.GdnVerifyFused [PDL True, PDL_RELEASE True, TOKENS 6]" in out
    assert "cudnn.GdnDecodeFused [PDL True, PDL_RELEASE True]" in out, "the prompt's decode steps on the fused decode entry in the target's form"
    assert "[speculative] the 32 new tokens are the plain session's: True; kept [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]" in lines
    assert "speculative against decode (1 sequence x 6 tokens per sequence, 10 engine steps behind a prompt of 8 tokens; " in out
    for name in ("tokens", "position", "prompt", "recorded", "rejected rows", "drafter cache", "draft proposals", "acceptance"):
        (line,) = [line for line in lines if line.startswith(f"  {name} ") and "greedy tokens equal to the reference stack's" not in line]
        assert line.endswith("    PASS"), line
    # what the comparison replayed is the run this program's loop ran: its recorded steps, and where it ended
    assert "ids and logits of 18 of 18 steps the session recorded of its own run (8 of its prompt, 10 verify steps) are this replay's bytes" in out
    assert "8 prompt tokens + the kept ones = 39; the session's own run ended there    PASS" in out
    kept = next(line for line in lines if line.startswith("  kept state  "))
    assert kept.endswith(("    PASS", "    REPORTED")), kept
    reported = next(line for line in lines if line.startswith("  bytes  "))
    assert reported.endswith("    REPORTED") and "(reported: on stack fe-entries the two kinds of step may launch other kernels for a block)" in reported
    assert (
        "checks (1 sequence x 6 tokens per sequence, 10 verify steps; tokens kept behind the first: 0, 1, 2, 3, 4, 5, 0, 1, 2, 3; behind a prompt of 8 decode steps)"
        in lines
    )
    assert "  replay            ids, logits and the drafter's proposals byte-identical on a second run from a fresh state    PASS" in lines
    judged = [line for line in lines if line.startswith("  blocks  ")]
    assert len(judged) == 2 and all(line.endswith("    PASS") for line in judged), "the prompt's decode steps and the verify steps, each against the reference"
    steps = [line for line in lines if line.split()[:1] == ["linear_attention.step"] and "[band: gdn_step]" in line]
    assert len(steps) == 2 and all(line.endswith("[band: gdn_step]  PASS") and "conv history equal; null slot untouched" in line for line in steps)
    assert "rate                  not printed while the drafter runs on reference blocks (kept tokens per step: the acceptance line)" in lines
    assert "acceptance            scheduled, not measured (reduction): 10 steps; kept 1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1" in lines
    assert any(
        line.startswith("first token ") and "each but the first behind one pass of the drafter over the position in front of it" in line for line in lines
    )
    traced = [line for line in lines if line.startswith("traced pass (")]
    assert not traced or traced[0].endswith(" | behind the prompt's 8 decode steps, replayed untraced")
    if "kernel identity, as planned: " in out:  # a GPU the census has a section for: the verify plan's share of the target, with its split
        reached = "13 of 20 launches of a step are target kernels (2 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results)"
        assert f"kernel identity, as planned: {reached}; not yet: mlp.down 2, attention.prepare 1, attention.core 1, projection.fp8.out 1, output_head 1" in out
    observed = [line for line in lines if line.startswith("  kernels  ")]
    # the kernel check is of the verify steps: beside its verdict the run says that the prompt's decode steps were not observed
    assert (
        not observed
        or "    the prompt's decode steps: 8 replayed in front of the traced window, their kernels not observed (the check is of the verify steps)" in lines
    )
    assert (
        not observed
        or (
            "    kernel identity of the drafter's passes: not stated (the kernel census holds the verify step and the output head behind it; "
            "no section holds a draft pass)"
        )
        in lines
    )
    frost_routing.measured(
        "model sample qwen_dense session with a drafter, 1 x 6 tokens: the kernel check", "; ".join(line.split("    ")[-1] for line in observed)
    )
