# samples/models — Agent Guide

End-to-end model samples: one process, text or token ids in, text or token ids out, a printed table of who serves each block,
and a self-check against a plain-PyTorch reference. Start with [README.md](README.md); the repository rules are in
[../AGENTS.md](../AGENTS.md) and [../../AGENTS.md](../../AGENTS.md).

## The tripwire

A model sample has no socket, no request queue and no admission of work while a step runs. The set of sequences is fixed
before the first step. A model sample keeps no state between two prompt sets, has no cache keyed by content, and chooses
nothing at run time: every provider and route is a row of the printed table. The list of provider keys is closed; it changes
only together with a kernel. A change that needs any of these belongs in a serving engine, not here.

Also out: a `generate()` method on the helper layer (the loop is in the script), stop strings, constrained output, logit
processors, tool-call parsing of the output, automatic choice of a stack or a route.

## Rules

1. **Run or skip cleanly.** Check the GPU, the library versions and the free memory before any load, for every sequence
   count of the run (`Session.plan_table(later)`: one memory sentence for the whole run). Print `[NN] SKIP <requirement>` and
   exit 0; the sentence names the fix. After the plan nothing skips: a provider that declines the operands an adapter built
   is an error, never a skip (a faulty adapter must not look like an unsupported GPU), and no `SKIP` can follow a check that
   ran. A test that honours a `SKIP` looks for a failed check in the output first.
2. **State the environment before the first load.** `check_environment()` compares the installed distribution of cuDNN
   Frontend with `cudnn.__version__`, says whether `cudnn` is an editable checkout and prints a source hash (never a path). A
   mismatch is an error that names both. It runs after the plan, so a GPU the sample does not serve still skips.
3. **`import cudnn` before `import torch`.** The helper layer never sets the process-wide switch that enables the opt-in FROST
   graph engines; a sample that needs a graph engine pins it for its own graph. A version-gated sample checks both the
   headers the module was compiled with and the loaded cuDNN library.
4. **Public API only.** The one host read of a decode step (the chosen ids) is in the sample's loop, outside every block.
   A block's `__call__` launches on the current stream and neither reads the device nor synchronises.
5. **No downloads.** Weights and tokenizer files come from a local path and are read without pickle. Every reduction (depth,
   vocabulary, seeded weights) is printed; a seeded run prints a hash of the produced ids and "the output is not language".
6. **The stack is explicit.** `fe-cublas` unless `--stack` says otherwise; never chosen by what is installed. `--stack
   reference` reports `[NN] RAN reference stack (no provider checked)`, never `PASS`.
7. **Provider selection is data.** A family's `configs.py` holds ordered `ProviderRow`s per block; the first row whose GPU
   classes, sequence range, tokens-per-sequence range and stack contain the run serves the block. To add a provider: add its
   key to `providers.KEYS` with its factory, add its row above the rows it should win against, name its tie-out band, and
   regenerate the family README's table (`configs.readme_table`); a host test keeps the two in agreement. A third-party
   provider is imported inside its adapter, its version is checked there, and a missing or unvalidated version is a `SKIP`
   that names it.
8. **Bands are never widened in place.** The band of a block against the reference is named in its provider row
   (`tie_out.BANDS`). A row that is a reference chain or a multiply over a dequantised copy carries a `fallback` reason and is
   counted as a fallback. A provider row that no test of this repository runs is printed as `unverified in CI`. Every band
   asks for finite values and is judged element by element. A band that is a share over a run is also held per unit of
   failure of the kernel (the fused step: one value head of one sequence). A band is declared before the run, never taken
   from the run's own measurement: the family declares its logits band (`logits_band_units` in `configs.py`).
9. **A default run is a test.** Each sample's default run is loaded by path and run by a GPU test that records its verdict in
   the job log; a required run that skips is uncovered, not passing. Host-only behaviour is tested under `test/python/misc/`.
10. **The reference accumulates in fp64** and keeps every intermediate cast of the chain it models
    ([../../test/AGENTS.md](../../test/AGENTS.md): "fp32 operands and TF32 references"). The report prints the reference's
    precision.
11. **No performance table in a sample README.** Timings a sample prints are for orientation only; numbers are published
    through the `benchmark/` folder's formal mode. A licence header on every file; the repository's formatter settings; no
    absolute local path in any output.

## What the first GPU run of this area taught

- A chain of 8-bit and 4-bit quantisers is not a continuous function. Two correct stacks whose multiplies differ by one bf16
  rounding on a quarter of their elements produce different codes after every quantiser, and with unit-scale random weights
  their logits differ by tens of percent within one step. Verify a provider per block, teacher-forced on the same operands;
  an end-to-end token comparison can only be as sharp as the margins of the model.
- A multiply that returns bf16 and applies its scales afterwards differs from a once-rounded fp64 reference by exactly one
  bf16 unit on about a quarter of its elements. That is the band, not an error.
- A bf16 multiply through PyTorch may reduce its partial sums in bf16: cuBLAS has that licence by default
  (`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`) and uses it by shape, row count and GPU. The same
  call was within one bf16 unit at 1 row and more than a hundred units off at 4 rows on compute capability 12.0. A provider
  that claims fp32 accumulation switches the licence off for its own call; the band is not what gives.
- The fused linear-attention step against an fp64-accumulated reference: its 8-bit bytes agree on more than 99.9 % of a run,
  but its recurrent state is not within 1e-4 everywhere, because a one-unit difference in a gate or a key column moves one
  head by up to a few 1e-3. The band therefore has a share and a cap (`tie_out.py`).

## What the first review of this area taught

Each of these let a deliberately wrong provider print `PASS`; `test/python/misc/test_model_samples_plan.py` now holds the
defect that shows it.

- `max(x, nan)` is `x`, and every comparison with a NaN is false. A band written as "largest distance so far within the
  limit" passes a NaN; downstream both stacks carry it and compare equal byte for byte. Ask for `isfinite` and judge with
  `(distance <= limit).all()`.
- A share over a whole run hides a defect confined to one unit of the kernel. One value head of one sequence is 0.4 % of the
  fused step's bytes at 4 sequences: it could be wrong at every step inside a 99.9 % band. Hold each unit to a share of its
  own, and leave padding (the null slot) out of the denominator.
- A band derived from the run's own measurement cannot fail. "Tokens equal wherever the margin exceeds twice the largest
  logit distance of this run" is true of any two stacks. Declare the band as data.
- Checking the first row alone against the first row in the batch says nothing about the other rows. Rotate the row that
  runs alone with the step.
- A tie-out between two stacks that read the same weight copies and quantisers cannot see an error in them. Pin those
  against the format definitions in a test of their own.
- A `SKIP` printed after a check has failed turns a failure into a skip for whoever reads the last line. Plan everything
  that can decline before the first load.
