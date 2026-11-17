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
processors, tool-call parsing of the output, automatic choice of a stack or a route. A field of a request body that asks
for one of them is a row of the table of rule 18 and is refused there by name, never served in part. A field that table
does not hold is printed as not known and ignored, so a new name for one of them goes into the table.

## Rules

1. **Run or skip cleanly.** Check the GPU, the library versions and the free memory before any load, for every sequence
   count of the run (`Session.plan_table(later)`: one memory sentence for the whole run). Print `[NN] SKIP <requirement>` and
   exit 0; the sentence names the fix. After the plan nothing skips: a provider that declines the operands an adapter built
   is an error, never a skip (a faulty adapter must not look like an unsupported GPU), and no `SKIP` can follow a check that
   ran. A test that honours a `SKIP` looks for a failed check in the output first. An entry that declines the geometry or
   the row count of a row that claims the device is an `ERROR`: a wrong table. An adapter to an entry asks it at plan
   time, on the real device, for the launch of the run's row count, and builds the entry's kernel for that row count on
   scratch operands: an entry asks what it needs of the installed toolchain where it builds, not where it is
   constructed, and that answer is a `SKIP` in the entry's own words before the load. A decline at that point which no
   requirement the adapter can name explains is of the adapter's operands, and an `ERROR`.
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
   A checkpoint is described from its `config.json` and shard headers first (`Checkpoint.describe`, the family's
   `describe`): the plan, the memory sentence and every decline come before a tensor is read. No output names a local path:
   a directory or a file is printed by its own name. No tokenizer, template or weight file of a real checkpoint is committed;
   the tests generate what they read.
6. **The stack is explicit.** `fe-cublas` unless `--stack` says otherwise; never chosen by what is installed. `native` is
   defined by the target kernel set of the family's census, not by a list of what happens to be fast; `fe-entries` is its
   cuDNN Frontend rows with every other block on a fallback row, so that those rows run where no further package is
   installed. `--stack
   reference` reports `[NN] RAN reference stack (no provider checked)`, never `PASS`. So does a run whose `--provider`
   flags leave no provider block to judge (`[NN] RAN (no provider block was judged)`, the blocks line `REPORTED`):
   `PASS` is kept for a run in which provider blocks were compared with the reference and held.
7. **Provider selection is data.** A family's `configs.py` holds ordered `ProviderRow`s per block; the first row whose GPU
   classes, sequence range, tokens-per-sequence range and stack contain the run serves the block. An entry that serves
   some layer widths only has them in its row (`widths`: fields of the model's geometry); a model of the family with
   another width is then the next row's, and the run says that no kernel identity is stated for it -- never `ERROR`. To add a provider: add its
   key to `providers.KEYS` with its factory, add its row above the rows it should win against (and below a row of the
   same key that an existing command selects with `--provider`), name its tie-out band, and regenerate the family
   README's tables (`configs.readme_table`); a host test keeps them in agreement. A third-party
   provider is imported inside its adapter, its version is checked there, and a missing or unvalidated version is a `SKIP`
   that names it.
8. **Bands are never widened in place.** The band of a block against the reference is named in its provider row
   (`tie_out.BANDS`). A provider whose kernel rounds at other points than the reference chain is compared with its twin
   (`Block.twin`): a reference block written at the kernel's rounding points, in the reference's accumulation precision.
   The reference is matched to the kernel's documented arithmetic; the band is then the kernel's own contract, never a
   width that covers the distance between two chains. A band names the outputs it applies to; an output it does not name
   is judged by its dtype, and state a call updates byte for byte. An adapter launches its entry and nothing else:
   what a consumer on a fallback row needs beyond the kernel's bytes -- a gather out of the kernel's scale layout, a
   quantisation the target does with a launch of its own -- is launched by that consumer, when it reads its operand
   (`weights.dequantize`). A row that is a reference chain or a multiply over a dequantised copy carries a `fallback` reason and is
   counted as a fallback. A provider row that no test of this repository runs is printed as `unverified in CI`. Every band
   asks for finite values and is judged element by element. A band that is a share over a run is also held per unit of
   failure of the kernel (the fused step: one value head of one sequence). A band is declared before the run, never taken
   from the run's own measurement: the family declares its logits band (`logits_band_units` in `configs.py`). Where no
   band both passes two correct stacks and fails a wrong one, the family declares none (`None`) and the line is printed
   as `REPORTED`, gated on finite values only: a line that cannot fail on a finite value is never printed as `PASS`.
9. **A default run is a test.** Each sample's default run is loaded by path and run by a GPU test that records its verdict in
   the job log; a required run that skips is uncovered, not passing. Host-only behaviour is tested under `test/python/misc/`.
10. **The reference accumulates in fp64** and keeps every intermediate cast of the chain it models
    ([../../test/AGENTS.md](../../test/AGENTS.md): "fp32 operands and TF32 references"). The report prints the reference's
    precision.
11. **No performance table in a sample README.** Timings a sample prints are for orientation only; numbers are published
    through the `benchmark/` folder's formal mode. A licence header on every file; the repository's formatter settings; no
    absolute local path in any output.
12. **Sample-only packages stay at the edge.** `samples/models/requirements.txt` holds them with lower bounds, never pins,
    and a "Validated with" line; none is a dependency of the package. Each is imported inside the functions that need it
    (`TextIO.from_checkpoint` and the stream decoder: `tokenizers`, `jinja2`; `Checkpoint.describe` and the tensor read:
    `safetensors`), never at module import, and a missing one is a `SKIP` that names the requirement file. A lower bound
    is a version the host tests of the text and weight readers were run against; raise it with evidence, in the same
    change as the code that needs it (for `jinja2`, whose sandbox a checkpoint's template runs in, with a sandbox fix). The
    family README states the checkpoint (repository, size, licence field as on its card, date read, manifest hash), the
    package versions and the GPU class of the last run with a checkpoint. Dependency alerts come through the `pip` entry for
    `/samples/models` in `.github/dependabot.yml` and are handled by the repository's maintainers.
13. **The report's GPU side comes from kernel activity records, never from a pair of CUDA events.** An event is stamped
    when the stream reaches it, so a pair spans every gap in which the GPU waited for the host. The traced pass
    (`Session.trace`, `timing.trace_steps`) replays recorded steps under the profiler and takes a step to end at its one
    host read; a block that reads the device would show as an extra step, which the pass reports instead of guessing.
    Only the profiler's own start and stop -- and, where launch records are asked for, the export and the read of its
    trace -- are reported as a missing trace, in a sentence that names no path; an error of a step under it propagates.
14. **What a user can get wrong ends in one `[NN] ERROR` line, never in a traceback.** A path that is not a directory, a
    file that is missing or is not what its name says, a copy of a checkpoint that is not whole (a shard missing, or one
    that does not hold the bytes its header describes), an entry of a side file that holds another kind than it stands
    for, a request file that is not JSON, a field of a request body that would change the output, a request whose
    prompt and new tokens do not fit the context length, messages a template cannot render or renders to no token,
    an output file that cannot be written or a file flag that was given an empty name (`--text-out "$OUT"` with the
    variable unset: a flag is tested with `is not None`, never by its truth), a standard input or output that was
    closed before the process started (the interpreter then holds `None` where the stream would be): each is raised as
    a `ValueError` whose text names the file by its own name, and
    the sample prints it before the plan. A reader of standard output that goes away while the text is written is the
    user's too (`sample | head`): the write stops, the run goes on, and its last line is an `ERROR` that keeps the
    verdict. Whatever a checkpoint's chat template raises is its refusal. That template is
    untrusted input: it is compiled in `jinja2`'s immutable sandbox, and a host test holds templates that try to leave it.

15. **A kernel is declared, then observed; a name proves no form.** Every adapter declares its census entries (`kernels`,
    `model_samples/census.py`): the kernels one call launches, as `exact` (judged one by one: name, order, count, and the
    launch shape where the entry states one), or as the families of a `fallback` row or of the `host` path. A provider row
    names the form its entry is called in (`form`: fields of the entry's own form), the adapter passes exactly those, and
    the kernel check (`--check-kernels`) compares them with the entry's answer on the objects. A row of an entry that
    states its launch names at least one field: with none the entry runs on its own pick and nothing is compared, so
    the check fails it. The call log of a traced step is an observation: after each call the adapter says which entry
    instance it called and what that instance states for the row's fields (`serving()`), never a copy of the row. A
    traced pass keeps one such log per traced step, and a step without its log fails the check. The
    check then holds the profiler's record of every launch of the traced steps against the plan. Where those records
    cannot be had the run ends in `RAN <stack> stack (kernels not observed: <reason>)`, never in `PASS`; a failed check
    stays `FAIL`. A new adapter comes with its census entries, and a chain that gains an operator whose kernel no
    pattern of `census.FAMILIES` names adds the pattern in the same change: an unnamed kernel fails the check by name.
    The host tests of the check are in `test/python/misc/test_model_samples_census.py`; each constructed defect there
    must fail with its own sentence, and the last line of a run is held by running `main()` to its end.
    An entry that a step calls in several forms has them in its row, by call site (`sites`: the further fields of the
    form at each site); the family's wiring names the site of every call (`sites()`), and the step hands that name to
    the block. The adapter holds one entry instance per form and serves a call with the instance of its site; which
    form serves which call is never its choice. Every item of the check then holds per call site, and a call is found
    by its place in the wiring, not by its block and layer: a block can be called twice in a layer.
16. **A target kernel set is data, and a statement about it is three numbers.** A family's `kernel_census.json` holds the
    target kernel set of its decode step: per block call the kernel, the form the target calls it in, and per GPU class,
    SM count and row count the recorded launch shape. How a block stands to the target (`target`, `stand-in`, `not yet`,
    `not the target`) is decided against that file from the row's kernel and form, never written into an adapter or a
    row. A count of target launches is always given as three numbers -- the target's own compiled kernels, cuDNN
    Frontend entries with byte-equal results, declared stand-ins -- never as one, wherever it stands: a sentence of a
    README, a cell of a table, the line in which a run says what it compared. The three numbers of the set count what a
    complete stack launches; a text that gives them says how many of the entries a stack calls today. Byte equality
    with a target kernel is the entry's own record (the census names where it stands: `tie_out`, `differences`), never
    a measurement of the sample. A section is `claimed` only on a `recorded` basis and only when no row selected on it
    is a fallback; until then a run prints `NOT CLAIMED` with every reason that applies, and a section of one GPU
    class, SM count, row count or set of layer widths is never used for another. `EQUAL` says what was compared: a
    claimed section records the launch shape of every target kernel but a tuner's pick and a stand-in, and a fact the
    reader of the file does not know is refused, never read as "nothing recorded". A pull request
    that changes a form the kernel census names updates the census in the same commit. Public text of this area calls
    it "the target kernel set" and names no product, repository or person for it.
17. **Standard output is the generated text; every line of a run is standard error.** A sample prints its `[NN]` lines,
    tables, report and checks through one function, to standard error, in every run: the text can then be piped, and
    a test reads a run's lines from standard error. The text of sequence 0 of a checkpoint goes to standard output (or to
    the file of `--text-out`) and nothing else does; it is written piece by piece inside the loop, each piece flushed,
    between the host read of a step and the next launch, so its write is counted in that step's turnaround and never
    inside a block. Seeded weights give no language, and their text is written nowhere. A request gets one text: a run
    with a list of sequence counts writes the text of a prompt with the first count, and a later count reports whether
    its sequence 0 produced the same text. A line of a run never goes to standard output, also where standard error
    is closed (`print` writes to standard output when it is handed no stream). The GPU test of a default run holds
    both streams: the run's lines on standard error and an empty standard output, in this process and in a fresh
    interpreter.
18. **A request body is answered field by field, beside the script.** A sample that takes the body a serving engine's
    client sends parses it in one module beside the scripts of its family (`serving_options.py`), as data: the fields
    it reads, the ones it states or does not use, and for every field that would change the output the values at which
    it changes nothing. Such a field at another value is an `ERROR` that names the field, its value, the reason and
    what is accepted; a field the sample does not know is printed and ignored; no field is dropped in silence, and
    nothing is decoded differently from what a field of the table asks for without an error. The table holds every
    name under which a client asks for something a sample leaves out (the tripwire's list); the host test names them
    by feature. The line a run prints for a body holds the names of its fields and the values the table knows; it
    never prints back what somebody else wrote into the body: of `model` the last part (a serving engine that serves
    a directory is asked for it by its path), of a field that is not used its name, and a name that is not a plain
    word as a JSON string. The helper layer is handed
    messages, tool schemas and template options and never a request body. An option that exists in a serving engine
    keeps that engine's name and meaning here (`--max-tokens`, `--max-model-len`); a new field of the table comes with
    its case in `test/python/misc/test_model_samples_text.py`, which fails for a field without one. What a run says
    before the load to somebody who comes from a serving engine -- the decoding rule beside the checkpoint's own
    generation settings, the decode steps a prompt takes before the first new token -- states what is exact at that
    point, a rule and a count, never a time: the report measures the time.

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

## What the first run on a real checkpoint taught

- An end-to-end comparison of logits or tokens is weak on a model with 4-bit activations, and no band makes it sharp. On
  the published checkpoint two reference stacks that differ only in accumulation precision (fp64 against IEEE fp32) were up
  to 52 bf16 units of the largest logit apart within 80 steps and disagreed on the greedy token at 13 of them, at margins of
  up to several logits. Verify providers per block, teacher-forced; print a second reference's distance beside the
  provider's instead of pretending the token check gates more than it does.
- The first call of a tokenizer and of a compiled chat template pays for caches the libraries build, tens of times the
  repeated call. A sample that times one call times the caches; print the first call and the repeated call apart.
- A streaming decoder is a library feature with its own history: the incremental decoder of `tokenizers` 0.21.0 fails on
  inputs 0.21.1 decodes. A lower bound in the requirement file is a version the host tests ran against, not a guess.
- The profiler's tracing library logs to stderr at start and stop, and once per kernel it cannot rate on a GPU it does not
  know. The sample sets its log level before PyTorch is imported unless one is set in the environment; the report says itself when the
  records are missing.

## What the review of the checkpoint path taught

Each of these survived the first set of tests; the host tests now hold the defect or the input that shows it.

- A band in units of the run's largest value changes its absolute width with that value's binade. 128 bf16 units of the
  largest logit are 8 logits where that logit is below 16 and 16 where it is below 32, while two correct stacks were 8.7 to
  13.5 logits apart whatever the binade. Such a band fails a correct stack on the next prompt, and its token half is never
  reached: a token is decidable only above twice the band, which is more than the largest logit. A gate that can fail a
  correct stack and cannot fail a wrong token is worse than none; print the line as `REPORTED`.
- Count distinct inputs, not runs. A chat template that opens with a fixed preamble makes the first 16 steps of every
  plain prompt the same 16 tokens: four prompts were three checked windows. Say which steps a check covers.
- A setting no test fails without is a setting a refactor drops. The template sandbox, block trimming, encoding without
  added special tokens and the precedence of the template file each needed a template or a tokenizer that renders
  differently, or escapes, when the setting is gone. Inject the defect; if the tests stay green, write the test.
- An `except` around a loop that runs the workload reports the workload's failure as the tool's. Catch around the tool's
  own start and stop.
- "Installed by the README" means the README's lines and no other. One unlisted `pip install --upgrade pip` hid that the
  pip of a new virtual environment has no `--group` option.
- Both stacks of the tie-out read a checkpoint through the same loader, so `PASS` cannot see a wrong reading. Say what
  does verify it (the text, read by a person; a comparison of the dumped ids), and do not let the verdict line imply more.

## What the second review of the checkpoint path taught

Found by reading the run as somebody who had never seen it, and by feeding it what a download or a hand edit leaves
behind; the host tests hold each.

- A header that parses says nothing about the bytes behind it. A shard cut after its header passed the description, the
  binding and the plan and failed in the load, without a file name. Compare every shard's size with what its header
  describes, and the directory with what its index file lists, before the plan.
- A check with nothing to judge cannot fail. "0 of 0 blocks inside their bands" is the reference compared with itself;
  count what was judged, print the line as `REPORTED` and the run as `RAN` where the count is zero.
- A sentence about what bounds a step, printed as a constant, will one day stand under a measurement that says the
  opposite. Word it from the run's own figures.
- One label for two units misleads on the line that decides whether a run starts: 52.4 GiB printed as "GB" is 56.3 GB.
  Say GiB where the divisor is 2^30, and state an allowance that covers what a run was seen to take.
- A run that prints nothing for a minute reads as a hang, and a budget cuts it. Say what comes next before a phase that
  prints nothing, outside every timed interval, and flush the line.
- A file a library saved carries the settings of the call before the save. A `tokenizer.json` with truncation or padding
  switched on cuts or pads a prompt without a word; switch both off, and hold it with a tokenizer saved that way.
- A command block is typed as it is written. A step that stands in a sentence elsewhere, such as fetching the checkpoint,
  is a step the reader does not take: put it in the block, in order, with what it costs.

## What building the kernel check taught

- A generated kernel symbol is the kernel's name and the text of its compile-time arguments, cut at a fixed length. Two
  forms of one kernel can share a symbol, and a launch shape too: the grid, the block and the shared memory of a launch
  do not say whether the kernel waits for its predecessor. Compare the form on the objects (the row's fields against the
  entry's own form), and use the name and the launch shape for what they do show: whose kernel it is, and in which shape.
- In a stack with reference chains the kernel names of a step are not the same in every step, although their number is:
  the softmax kernel of the reference attention is instantiated by the length of the cache, and its fp64 multiplies
  over a cache of one position are served by other kernels than over a longer one. So only the kernels of an entry are
  judged one by one; a chain is attributed by family, and no test may assume one name sequence per step for a plan that
  has chains.
- The profiler's `events()` carry names and times; the grid, the block and the shared memory of a launch are in its
  exported trace. That trace grows with the launches of a step (32 steps of the two-layer model: about 30 MB), so it
  is read event by event and never held as a whole.
- A copy inside a launch graph that was captured before the process's first profiler pass was seen to be replayed as a
  kernel record, not as a copy record (PyTorch 2.14). Count a copy by either spelling, or a captured step has one more
  kernel than the same step launched from Python.
- A check that could not look must not read as one that looked and found nothing: a run without the profiler's records
  is `RAN`, and the reason is in its last line.

## What the review of the kernel check taught

Each of these passed every test of the first version; the host tests now hold the defect that shows it.

- A log line that is copied from the plan cannot disagree with the plan. The call log's form was the row's own tuple,
  so "every logged call carries the row's form" could fail on a hand-built line only. Write the line after the call,
  from the object that was called.
- An empty statement compares equal to anything. A row without a form passed the form check with nothing compared
  while its entry ran on its own pick. Fail an entry row that names no field.
- The last line of a run is code. The verdict function and the check were each tested; that `main()` hands the reason
  from one to the other was not, and one dropped line turned `RAN` into `PASS` with every test green. Run `main()` to
  its last line in a host test, with stand-in providers and a stand-in profiler.
- The text of an `OSError` carries the path of its file. A reason built from it put a temporary path into the last
  line of a run. Say the system's words for the error number, and name a file by its own name or not at all.

## What building the native stack taught

- How a row stands to a target kernel set is not a property of its adapter. One adapter serves the fused step in the
  entry's own pick and in the target's form; the kernel is the same and the form is not. Decide the identity from the
  row's kernel id and form against the census file, per run, and let a host test hold the rows of the stack to it.
- A kernel with no pattern of its own reads as a missing launch. The first run of the gate entry on a GPU failed with
  "did not launch gate_quant": its symbol had fallen to the pattern for any other cuDNN Frontend kernel. A new adapter
  adds its kernel's pattern in front of that one, and a host test holds every exactly judged kernel id of the provider
  list to a pattern of its own.
- An entry that returns a view of one buffer of its own hands back bytes that its next call overwrites. The tie-out
  calls a block again on one sequence alone, so the first call's result would no longer be the first call's. An adapter
  to such an entry passes an output buffer of its own per call.
- A recorded section holds for its row count. The default run decodes at 1 and at 4 sequences; the census was recorded
  at 1 row, so at 4 the run states no identity and says why. Do not carry the kernels of one row count, SM count or GPU
  class over to another: which kernel serves a block changes with each of them.
- Compare the versions of what the stack imports, not of what is installed. The first census line named two packages
  that sat in the environment and that no provider of the stack had loaded.
- A new row of an existing provider key changes what `--provider BLOCK=KEY` selects when it is placed first: the first
  declared row of that key that serves the run wins. Put it below the row an existing command selects.
- The profiler's name of a kernel is right only in a process that has unloaded no kernel module. The GPU test of the
  second stack failed its kernel check one run in three, by the name of the fused step alone: the test session had run
  the default stack first and dropped its entries, and the new form's kernel was recorded under the old form's symbol.
  Two raw CUDA kernels, loaded and unloaded in turn, showed the same on about every second launch, each writing its own
  output; nothing of it where every module stayed loaded. So the kernel that runs is right and its recorded name is not.
  A sample process never unloads an entry; a test that checks kernels runs the sample in a fresh interpreter. When a
  kernel check fails by one name only, and the name is another form of the same entry, look for an unload before the
  run before looking for a wrong launch -- and then look for the wrong launch.

## What the review of the native stack taught

Each of these passed every test and every GPU run of the first version; a host test now holds each.

- A count given as one number was attributed wrongly in the next clause: "two entries (65 launches)" where the two
  entries are 64 and the 65th is PyTorch's gather. The split is what makes a count checkable; a host test refuses a
  count of target launches in the READMEs that is not followed by it.
- "273 have an entry" was the number of the complete stack in the present tense: 257 of them were not called through
  an entry by any stack. Say whose number it is, and how much of it exists today.
- A record holds for what it was taken on, and the model is part of that. The census was keyed by GPU class, SM count
  and row count; a checkpoint of the family with other layer widths was told `target` and then failed on the 27B's
  grids. A section names the widths it holds for, and a launch shape that depends on one more field names that field.
- A decision taken in two places is taken in neither. "A target stack always checks its kernels" held in the script,
  which set its own flag; the helper's check function defaulted to off, so the next script of the stack would have
  printed `PASS` for a run that never looked. The decision is the session's, and the helper reads it there.
- `EQUAL` printed for a comparison that was not made. An entry without its launch shape, or with the key misspelt, was
  read as "nothing recorded" and the line still said `EQUAL`. Refuse the file; print what was compared.
- A comparison of three fields tested by varying one field holds one field. Weakening "grid, block and shared memory
  are equal" to the grid alone left every host test green. Vary each field in a case of its own.
- An entry gates the installed toolchain where it builds its kernel, not where it is constructed or asked for a launch
  shape. Planned without a build, a GPU whose target the installed CuTe DSL cannot assemble passed the plan, read 55 GiB
  of weights and ended in a traceback that blamed the adapter. The entry's first use belongs to the plan.

## What building the norm row taught

- One block, several forms. The target calls the add + RMSNorm + quantise kernel in five forms in a step, and a launch
  record cannot tell them apart: one kernel id, one grid, one block, one shared memory. So the form of a call is data
  at three places that a host test holds to each other -- the row (per call site), the wiring (the site of every
  call), the census (the form of every call) -- and it is observed in the call log, which says which entry
  instance served the call. A wrong form shows there and, for this entry, in the kernel's symbol, which carries the
  form's text; never in a launch shape. A check that reads the log must also fail without it: a traced pass with
  launch records and no call log passed until a host test asked.
- A block that is called twice in a layer is not found by its block and layer. The first version of the census dump
  looked a call up that way and reported, for a layer's input norm, the form of the layer's second norm call. A call
  is the one at its place in the wiring.
- A kernel that updates an operand in place changes what the next caller of the tie-out sees. The norm kernel adds
  into the residual stream; the reference block and the run of one sequence alone must each start from the stream as
  it was. Declare such an operand as state of the block (`state_args`), so that it is saved, restored and compared.
- A kernel breaks an exact rounding tie its own way. A bf16 value over a block scale times a scale such as 1/336 is
  an exact 4-bit midpoint more often than chance suggests (2 elements in 1,000 on random rows), and there the code
  follows the kernel's own reciprocal. The reference quantiser and a kernel's chain differed by a whole code on 6
  elements in 10,000. So the twin of such a kernel keeps a bracket, and the band is the kernel's documented
  contract; a tolerance on codes would have been the wrong tool, and "equal to the reference quantiser" the wrong
  claim. On rows 64 wide, in a host test, one such code moved a logit by seven bf16 units.
- What a kernel hands to a block that is not a kernel yet is not the adapter's to convert. A gather, a quantisation
  or a copy inside the adapter is a launch in the span of a row that declares one kernel; next to another exactly
  judged row the check fails it, and next to a fallback row it would pass as that row's. Hand the bytes over as they
  are and let the consumer convert when it reads them.
- Attribution by family cannot see everything. A launch more in front of the first norm of a step fails the check,
  because the gather before it is judged one by one; the same launch in front of a norm that follows a fallback row
  is counted as that row's, and a launch behind the norm's kernel is counted as the consumer's at every call site
  while that consumer is a fallback row. The check is exact where two exactly judged rows meet, and it gets sharper
  with every row that leaves its fallback. Until then "the adapter launches nothing but its entry" is held on the
  adapter itself: a host test finds, in what the adapter returns, the very tensors its entry wrote.
- A row says what its entry serves, also in the model's widths. The norm entry serves one hidden width. The first
  version of its row did not say so, and a checkpoint of the family with another width ended in "the provider table
  is wrong" where the README promised a run on the reference chain. The table was wrong: the width is a condition of
  the row.
- A sentence counted once per occurrence, not once per step, read "and 7 more of the 4 traced steps" as soon as a
  block could fail twice in a step.

## What building the serving-style options taught

- A line that warns of a cost before the load has no measurement of its own to give. The first plan for the
  first-token line was "about X s before the first token", with X taken from an earlier run. The same stack took a
  quarter of a second per step in one checkpoint run and 0.15 s in another, on another host, because the launch
  path is the host's: a recorded X would have stood above a report that says otherwise. The line gives the count of
  decode steps, which is exact, and says where the time is.
- `true == 1` in the language this is written in. A table of neutral values compared with `==` accepts `"n": true`
  and `"logprobs": 0`. Compare a boolean only with a boolean, and hold it with a case of each kind.
- A stand-in that fails forever cannot show a second attempt. The first test of "after a failed write nothing more is
  written" used a stream that raised at every later write, so a sink that kept trying looked the same. Fail once.
- A remainder nobody produces is a remainder nobody tests. What a stream still owes at its close (a sequence that
  stopped inside a character) was dropped by a one-line defect in two places with every test green: no test stream
  owed anything. Hand the loop a stream that does.
- A reader of a pipe that goes away fails twice: at the write, and again when the interpreter flushes at exit, with a
  message of its own and exit code 120 whatever the run returned. Catching the first is not enough; point the
  descriptor at the null device. A host test runs a child process whose standard output is a closed pipe.
- Moving a run's lines to another stream moves every test that reads them. Four test modules read the report from
  standard output; each read was one line, and a run that prints nothing where a test looks fails by an empty list,
  not by a wrong value. Print through one function, so that the stream is decided in one place.

## What the review of the serving-style options taught

Each of these passed every test and every GPU run of the first version.

- "Not known and ignored" is where a feature hides under another name. The first table refused `stop` and
  `response_format` and ignored `include_stop_str_in_output`, `structural_tag`, `logits_processors`, `bad_words` and
  the older `functions`: fields that ask for exactly what this guide lists as out, while the guide said such a request
  is never ignored. List the names by feature, not by what came to mind, and let a test hold the list.
- A body copied from client code carries the client library's keyword (`extra_body`), whose content the library
  would have merged into the top level. Ignored as an unknown field, the sampling settings and template options
  inside it were dropped and the run decoded another prompt than the client sends.
- A line that echoes a body echoes what is in it. `model` is a local path where an engine served a directory; `user`
  and `metadata` hold what somebody else wrote; a field's name can hold a line break, and a line `[00] PASS` then
  stands in the middle of a run. Print names and the values the table knows, and everything else as a JSON string.
- A standard stream that was closed before the process started is `None` in the interpreter, not a stream that
  fails. A write to it is an `AttributeError`, after the whole load; and `print(..., file=None)` writes to standard
  output, so a run with standard error closed put every line of its report into the text.
- An empty string is not "no flag". `if path:` let `--text-out ""` (an unset shell variable) through the check before
  the run; the whole run happened, the text was written nowhere, and the last line blamed standard output.
- One sink for every sequence count wrote the text of a request once per count, with nothing between the copies.
  A request is one; the counts are a loop. What the loop writes once and what it writes per count is decided where
  the loop is, and held by a run with two counts.
- A sentence about a stream is held on a device only by a test that reads that stream there. "The seeded run writes
  nothing to standard output" stood in three READMEs while the GPU test dropped standard output.
