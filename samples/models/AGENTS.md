# samples/models — Agent Guide

End-to-end model samples: one process, text or token ids in, text or token ids out, a printed table of who serves each block,
and a self-check against a plain-PyTorch reference. Start with [README.md](README.md); the repository rules are in
[../AGENTS.md](../AGENTS.md) and [../../AGENTS.md](../../AGENTS.md).

## The tripwire

A model sample has no socket, no request queue and no admission of work while a step runs. The set of sequences is fixed
before the first step. A model sample keeps no state between two prompt sets, has no cache keyed by content, and chooses
nothing at run time: every provider and route is a row of the printed table. The list of provider keys is closed; it changes
only together with a kernel, with the reference chain of a block that a kernel is to serve, or with a declared library
entry. A library's own tuner may choose among that library's kernels for a call a row names; the choice is printed with
the kernel check and is never a row. A change that needs any of these belongs in a serving engine, not here.

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
   requirement the adapter can name explains is of the adapter's operands, and an `ERROR`. An entry that is built
   from the row count alone, once the entry has stated a launch for that row count, is handed nothing a decline
   could be about: every typed decline of such a build is the entry's answer about the installation, and the `SKIP`.
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
   `describe`, its drafter's): the plan, the memory sentence and every decline come before a tensor is read. No output names a local path:
   a directory or a file is printed by its own name. No tokenizer, template or weight file of a real checkpoint is committed;
   the tests generate what they read. A third-party provider may build its own kernels at its first use; a sample makes
   that first use while the run is planned, before anything is loaded, and says so in one line.
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
   provider is imported inside its adapter, on a device; its version is read from the distribution's metadata there and
   held to the validated list, and a missing or unvalidated version is a `SKIP` that names it, the requirement file and
   the nearest command that runs without it. A plan on stated device facts imports no third-party package.
8. **Bands are never widened in place.** The band of a block against the reference is named in its provider row
   (`tie_out.BANDS`). A provider whose kernel rounds at other points than the reference chain is compared with its twin
   (`Block.twin`): a reference block written at the kernel's rounding points, in the reference's accumulation precision.
   The reference is matched to the kernel's documented arithmetic; the band is then the kernel's own contract, never a
   width that covers the distance between two chains. A band names the outputs it applies to; an output it does not name
   is judged by its dtype, and state a call updates byte for byte. An adapter launches its entry and nothing else:
   what a consumer on a fallback row needs beyond the kernel's bytes -- a gather out of the kernel's scale layout, a
   quantisation the target does with a launch of its own -- is launched by that consumer, when it reads its operand
   (`weights.dequantize`). An adapter that multiplies on the stored format of its weight says so (`stored_weights`):
   the load of the session then keeps that format for the block, as the loader had it, and keeps the bf16 copy of a
   weight only where a block of the run's plans reads one -- which forms a weight has is decided from the plans of
   every sequence count before the first load, never at run time, and a block that finds its form missing is an
   error that names it. The reference blocks of the checks read copies: where the load kept none the checks make
   theirs for one call at a time (`weights.OnDemand`), the memory sentence states that part, and the checks print
   what the copies cost. Such a copy is made from the bytes the weight's own block multiplies on, so it is a
   reference only while those are the bytes of the load: the loader takes a checksum of every weight it keeps on its
   stored format alone and the checks compare it before their first pass and after their last (`stored weights`: a
   weight that was written to fails the run by block and layer). An adapter never writes into its weight. Only the
   reference session of the checks makes copies per call; any other session is refused the maker. A kernel that
   reads its weight in a row order of its own gets it from the load in that order: the adapter names the layout
   (`stored_layout`, built by the entry's own helper), the session collects the layouts of its plans, and the load
   keeps that one stored form -- never a second stored tensor beside it, never a layout made inside a call. The
   reference's copy is made from that form, into the checkpoint's row order. And an operand that reaches a kernel in
   another layout than the kernel reads is brought into it by the kernel's adapter, with a launch of its own. A row that is a reference chain or a multiply over a dequantised copy carries a `fallback` reason and is
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
    package versions and the GPU class of the last run with a checkpoint. Version updates come through the `pip` entry
    for `/samples/models` in `.github/dependabot.yml` and are handled by the repository's maintainers. A library whose
    kernels a stack calls has a requirement file of its own, a lower bound with its "Validated with" line
    (`requirements-flashinfer.txt`), and `constraints-native.txt` holds the validated versions as exact pins: the
    versions the launch records of the family's census were recorded with. The maintainers of this area own that list; a
    version joins it with a recorded run, never with an update of the pin alone.
13. **The report's GPU side comes from kernel activity records, never from a pair of CUDA events.** An event is stamped
    when the stream reaches it, so a pair spans every gap in which the GPU waited for the host. The traced pass
    (`Session.trace`, `timing.trace_steps`) replays recorded steps under the profiler and takes a step to end at its one
    host read; a block that reads the device would show as an extra step, which the pass reports instead of guessing.
    Only the profiler's own start and stop -- and, where launch records are asked for, the export and the read of its
    trace -- are reported as a missing trace, in a sentence that names no path; an error of a step under it propagates.
    The token-to-token interval and the rate come from the host clock of requests that are not traced; GPU busy and
    the device time of a kernel come from kernel activity records of a traced request at the same positions, behind
    the last timed request. The two are never taken from one pass, and a report says which is which. A figure that
    sets one against the other -- launch overhead, a step less its GPU busy and its host time -- is formed step by
    step, at the same position, before anything is aggregated: the medians of three lists need not be one step's, and
    their difference can be a time no step has. Where its terms are printed as a sum they are the means over the same
    steps, so that a reader can add them up, and it is called an estimate: the two sides are two requests. The median
    of the per-step values is printed beside it: one stalled step moves a mean and not a median. A row of the
    per-kernel table is found by its place in the wiring, never by a kernel's name, and a traced pass whose kernel
    check does not hold prints no table. The row of a block a library serves names the library's entry, as the plan
    does, never the block alone and never a fallback; what a block launches beside a kernel of its own is the row of
    its place in the wiring and says so.
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
    by its place in the wiring, not by its block and layer: a block can be called twice in a layer. A site name
    belongs to one block (an adapter finds the sites of its row in the wiring by name), and what differs between the
    sites of a row may be the operand's shape as well as the kernel's form: an entry that is one weight shape per
    instance has the shape of each site in its row. A row may also serve some call sites of its block only: it names
    each other site without fields, which hands the calls there on to the next row of the stack that serves the run.
    The plan then holds one part per call site under the block's name (`blocks.BySite`; `Plan.at`, `Plan.parts`), and
    whatever reads a row or a block of a call -- the census entries, the form, the identity, the band, the batch
    invariance statement, the form of the weight a load keeps -- reads the part of that call's site. A site a row
    does not name at all stays a wrong table.
16. **A target kernel set is data, and a statement about it is three numbers.** A family's `kernel_census.json` holds the
    target kernel set of its decode step: per block call the kernel, the form the target calls it in, and per GPU class,
    SM count and row count the recorded launch shape. How a block stands to the target (`target`, `stand-in`, `not yet`,
    `not the target`) is decided against that file from the row's kernel and form, never written into an adapter or a
    row. A count of target launches is always given as three numbers -- the target's own compiled kernels, cuDNN
    Frontend entries with byte-equal results, declared stand-ins -- never as one, wherever it stands: a sentence of a
    README, a cell of a table, the line in which a run says what it compared. The three numbers of the set count what a
    complete stack launches; a text that gives them says how many of the entries a stack calls today. Byte equality
    with a target kernel is the entry's own record (the census names where it stands: `tie_out`, `differences`), never
    a measurement of the sample. A form may hold a field only the port has; the row and the census then name the value
    the port's byte comparison ran with, the census lists under `differences` what the target kernel does there, and
    the family README says so beside the row. A section is `claimed` only on a `recorded` basis and only when no row selected on it
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
    inside a block -- or once, after the report, with `--no-stream` and always in the performance mode. Seeded
    weights give no language, and their text is written nowhere. A request gets one text: a run with a list of
    sequence counts writes the text of a prompt with the first count, and a later count reports whether its
    sequence 0 produced the same text. A line of a run never goes to standard output, also where standard error is
    closed (`print` writes to standard output when it is handed no stream). The GPU test of a default run holds
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
    A configuration object of a serving engine that selects what is launched is answered key by key in the same
    module; a key it does not hold is an `ERROR` that names it. A key is read, accepted at the value that changes
    nothing, or refused with its reason, and each key of its tables has its case in the same test module.
19. **The performance mode states what it observed, never what it assumes.** One definition per statistic, written in
    the report file, with a host test of its arithmetic on constructed time stamps. Nothing but the step runs inside a
    timed request: no recorder, no call log, no profiler, no text; and no profiler pass runs in the process before its
    last timed request. The report says at which positions the blocks were held against the reference, that every
    timed request gave the checked request's tokens and state, and that the kernels of the traced steps are the
    plan's; a run in which one of the three does not hold prints no rate, and kernels that could not be observed are
    reported as not observed, the rate staying `orientation` with that reason. Another process, the clock and a
    limit reason are observed or printed as not observed; `none seen` is printed only when the run's own process is
    in the driver's list; the report never prints the word 'locked'. The rate of a run that is not a controlled run
    says so in its own line, and line 2 says it of every other number. A report holds no speed-up, no baseline and
    no figure of a second GPU class. The screen is rendered from the data the report file holds and from nothing
    else, at no more than 110 characters a line; a line of it that would be longer breaks at a separator or between
    two words. An ordinary run prints what it printed before the mode existed and does not import the mode's module:
    the mode is a branch of its own in the script, and a pull request that touches the loop, the census, the checks
    or the timing shows the area's plain runs line by line against the tree it started from
    (`test/python/misc/test_model_samples_perf.py`).
    The GPU is observed through the driver's `nvidia-smi` tool in child processes, with no package added, and never by
    a call of the timing process inside a request: the sampler is started in front of the warm-up request and its
    files are read behind the traced request, and its loops end with the pipe the run holds: at the stop, or when the
    run is gone. The report file says what the sampler was to the timed requests of its run, or that none ran, with
    the reason (`protocol`).
    A list of processes counts as empty only beside the run's own entry and a device memory in use that the run's own
    explains. A count of other processes is stated only of a list that shows the run's own entry too. A `not observed`
    carries a reason from a closed list, never the tool's path or its own words; no process name is asked for, and
    the report keeps no id of another process. A lock is the operator's declaration: it is printed as declared, held
    against the observed clock in both directions, and never set.

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

## What building the projection row taught

- The class of a row is where the target launches the kernel, not where the entry runs. The FP8 projection entry
  serves compute capability 10.x; the target launches it for the in-projections on 10.0 and a library multiply on the
  other classes, so the row claims 10.0 and up to 32 rows, and everywhere else the block prints its fallback reason.
- A row states the kernel whole. This entry picks its own form when none is passed, and the pick may move with the
  entry; the row names the split, the weight tile, the ring depth, the tensor-memory columns and their release and
  both launch knobs, the adapter passes the form as one object, and it refuses a row that leaves one of them out.
- One block, one form, two weight shapes. The in-projection's weight has another shape in each kind of layer and the
  entry is one shape per instance, so the shape is a field of the row per call site, as a form is for the norm block.
  Here a launch record does tell the sites apart (one thread block per 128 weight rows), and the census records both.
- A site name belongs to one block. The first draft named the in-projection's sites by the layer's kind, which the
  norm block's sites already are; the norm's adapter, which finds the sites of its row in the wiring by name, would
  have built a form no call of a two-layer model is made in. A host test that pins the forms a run builds holds it.
- A kernel on the stored format needs the stored format. The loader had the merged 8-bit codes in hand and freed them
  after the copy; it now keeps them for the blocks that say they multiply on them, asked by the plan of the session
  and of the sessions that follow on its weights. At that point the copy stayed for the reference (it no longer
  does: "What making the loader keep what the plans read taught" below), so the memory sentence grew, and the
  allocator's slack with it: the first checkpoint run took 0.4 GiB more than a sentence that had added the stored
  bytes alone. Measure the sentence against a run whenever what a stack holds changes.
- With a second exactly judged row behind it, a launch behind a layer's input norm fails the check: it now sits
  between two rows that are judged one by one. The same launch behind the in-projection fails in a linear-attention
  layer (the fused step follows) and is counted as the reference chain's in an attention layer. What the check
  cannot see shrank by those places and no further.
- A chain of dependent launches is a statement about three kernels. The norm releases at its wait, the in-projection
  starts on that release and waits before it reads the norm's codes, the fused step starts on the in-projection's
  release and waits before it reads the projection's rows. Each row says in a comment what the launch before it must
  not write and what the launch after it must wait for; replay and the block check on the device are what hold it.
- Run a row at the largest row count it claims. The row claims 1 to 32 rows and its first runs went to 16. From 17
  rows the kernel check of every stack failed on the embedding row: PyTorch's index-select launches another kernel
  there, which the row's pattern did not name. A pattern names every kernel its operator launches over the row counts
  a stack serves, and the run table of a row has its first and its last row count.
- What a report says of a stack follows the plan. The `bytes per step` line counted every weight as a bf16 copy after
  a block had begun to read its weight on the stored format; the model is now asked with the blocks of the session's
  own plan that multiply there, and the line says how many bytes are read in the stored format.
- A table rendered for one GPU class names that class. The second served-by table of the family README is the plan
  on compute capability 10.0; its heading said 10.x and the sentence under it kept the count of the other classes. A
  host test now holds the heading and the by-provider sentence to the plan's own line, per class.
- An option an adapter declines on is tested through the session that hands it over. The adapter's decline for a
  weight merged at one scale per row was tested on a block built by hand; without the option in the session's plan
  the decline would have come after the load, as an error about another plan.

## What building the drafter's loader taught

- A loader that no step runs yet has no tie-out behind it, and when a step does run, both stacks will read the same
  loaded bytes. So what a load makes is pinned where it is made: the 8-bit quantisation against a quantiser written
  from the format's definition, bytes and scales; the rows cut from the output head against the bytes of the shard
  file, read by the offsets of its header and not through the reader under test.
- A rule for codes is fixed down to its operation. Dividing by a scale and multiplying by its reciprocal agree on
  almost every element and differ where a quotient is exactly the midpoint of two codes. On the weights of the
  published checkpoint's drafter 2,021,676 of 424,673,280 quotients are such a midpoint, and the reciprocal would have
  given another code on 207,130 elements. Say the operation in the docstring, and hold it with an element that tells
  the two apart.
- A block of a README that shows what a run printed is held like a table. The manifest of the published checkpoint is
  rendered in a host test from the code that prints it, at the published widths; the run on the checkpoint then only
  has to show that the checkpoint's own headers give that list.
- A call that a README shows is a run, and it reads what it reads. The section's own call of the loader read the
  model's embedding table as well, 2.37 GiB at the published widths, under a manifest whose last line counts 0.57 GiB
  "besides" that table; nothing said that an argument hands the model's table over. Say under a shown call what it
  reads beyond the lines it prints. A host test runs the call as it is written and finds the table among what was
  read, and holds the size in the sentence to the tensor list.
- A section names a flag when a script has it. The first version named the flag that a script which does not exist
  is to take for the list. A host test refuses, in that section, a flag that no script of the folder has.

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

## What making the loader keep what the plans read taught

- A weight has the forms its readers read, and the readers are the plans of the whole run. The projection row kept
  the stored codes beside the copy for one block. The rule for every block is two questions asked of every plan
  before the first load: does some plan multiply on the stored format (keep it), does every plan (make no copy). A
  sequence count on a fallback row, or a row selected away, keeps the copy; nothing is decided at run time.
- The reference of the checks is a reader too, and the one reader that may come without its form. It gets a bf16
  copy for one call, made by the session of the reference and by the tap of the checked pass -- never inside a
  block, whose call neither reads the device nor synchronises. Such a copy is bit for bit the copy a load keeps as
  long as no block has written into its weight, and a host test loads the same plan both ways and compares every
  check line.
- A reference made from the provider's own operand is no reference until that operand is held still. With the copy
  gone, the reference's copy of a weight is made from the bytes its block multiplies on: a stand-in that zeroed half
  of its weight from its first call on passed replay, every block line (distance 0.00), the tokens and the kernel
  check, where the same stand-in in a run that kept its copies was 255 units off on half of its elements. The loader
  now takes a checksum of every weight it keeps on its stored format alone, the checks take it again before their
  first pass and after their last, and a changed weight fails the run by name. When a change takes an independent
  operand away from a check, give the check a way to see that the shared one has not moved -- and test the defect in
  both forms of the run.
- What a memory sentence rests on is tested on the thing itself. "One copy at a time" was held by one run's peak; a
  defect that kept every copy alive passed every host module, and a two-layer run would have held 34 GiB more
  without a failure. A host test now holds weak references to the copies' tensors (none of an earlier call alive
  when the next is made, none when the checks return), and the GPU cell holds PyTorch's peak below the sentence.
- An option that exists for one caller is refused for every other. The maker of copies was a public parameter of
  every session; on a checked session it put a cast and two synchronisations in front of blocks of its timed and
  traced steps, and the run passed. It is refused when the session is built, and such a session is not traced.
- On the reduced models the memory sentence is an estimate that runs miss by a few tenths of a GiB in either
  direction (a flat 2 GiB of working memory), with and without their copies; leaving copies out neither widened nor
  narrowed that in a way the runs agree on (2.3 GiB of room with the in-projections of four layers on their codes
  alone, 1.7 GiB with every multiplied weight of two layers). State such a sentence with the runs beside it, and do
  not fit a term to one run.
- The memory a run takes beyond its weights does not shrink with the copies it leaves out. Without the 9.7 GiB of
  in-projection copies the device grew by 10 GiB less, but the room beyond the weights held stayed where the stored
  tensors had put it: 6.6 GiB, against 6.9 GiB with the codes beside every copy and 5.3 GiB with copies alone. The
  first sentence, an eighth of what is held, was 0.5 GiB short of the run. The lesson of the row before, again:
  measure the sentence against a run whenever what a stack holds changes.
- A form that is missing fails where the plan is, not where the multiply is. A session on weights that were loaded
  for another plan failed at the adapter's first call, in the adapter's words, and a block that reads a copy would
  have ended in an error about `None`. The load of the session now names the block and the form before anything is
  compiled or stepped.
- Arithmetic that was right for one format was wrong for the next. The bytes of a step took a stored weight's bytes
  off its copy's, which is right at one byte per element and wrong at 0.5625. State such a formula in both forms: the
  copy's two bytes off, the stored bytes on.
- A line in GiB reads 0.00 on the geometry of a host test, so no test held its figures. The text of the cost line
  is a function, and a host test calls it on the figures of a run at the published widths.

## What building the gate / up row taught

- A kernel's row order of a weight is a form of the stored weight, and a load keeps one. `cudnn.SwigluFp4Decode`
  reads gate and up rows interleaved in tiles of 64, and its helper returns new tensors. Kept beside the merged
  tensor that would be the weight twice (12 GiB on the published model), made per call a permutation of 100 MB in
  front of every launch. The layout is asked of the provider while the run is planned and applied once per weight
  at load; the merged tensor is dropped. Two plans of one run that multiply on one weight in two row orders are a
  wrong table, named with the plans; a session on weights that another plan loaded in another row order is named
  before its first step, like a missing form.
- The reference's copy is made from the bytes the kernel multiplies on, whatever their order. With the weight kept
  interleaved, a copy made by un-permuting it into a second packed tensor would have put a stored tensor the
  checksum does not cover between the kernel's bytes and the reference's. `weights.dequantize` reads the layout a
  band of rows at a time and writes the checkpoint's row order; a host test holds the copy bit for bit to the copy
  of the merged weight, and the write-into-the-weight control fails the run on this weight as on the others.
- A form that follows the row count is a row per value. The entry's token tile is a floor: it runs the narrowest
  of four tiles that holds the rows of the call. One row with `TOKEN_TILE 8` would have stated at 9 rows a form the
  kernel does not run, and the form check compares the row with the entry's own form on the objects. The block
  has four rows of one key by row range, the census names the field of its section's row count, the README table
  shows the row of 1 to 8 rows and a sentence names the others, and a host test holds each row to the public
  entry's form at the first and the last row count it claims.
- Not every kernel has a bracket. The gate and the norm are compared with a twin that keeps the two bf16 values
  around an fp64 value; an fp32 sum over 5,120 products is not within a few units of the fp64 one, so this entry's
  contract is an interval: the fp64 chain with a propagated error bound. The band takes the terms of the entry's
  own suite, a host test holds them to that file, and it holds the reference pair's bytes and a kernel-like fp32
  chain inside the band and a wrong code and a wrong block scale outside. What the band cannot see is pinned as
  that: one bf16 unit of the product.
- The layout between two kernels is theirs; between a kernel and anything else it is the kernel's adapter's. The
  norm kernel writes block scales in the 128 x 4 layout and the gate / up kernel reads them: handed over as they
  are, no launch between the two. One row of a batch in the batch check, or the norm row selected away, hands
  block scales row by row, and the adapter brings them into the layout itself. Asking the producer for the
  layout would put a launch inside the norm's call, on a path only the checks take.
- The rounding chain of a fused kernel is its own, and a checkpoint run says so: the greedy tokens of the checkpoint
  run moved with the row, as they did with the norm row, and the token line stayed where two reference stacks are
  from each other. On the seeded model the ids hash is the default stack's. State both, and never read the first
  as a defect of the row or the second as its proof: the block check is what holds the kernel.
- Measure the sentence again. Without 21.2 GiB of gate / up copies the device grew by 15 GiB less; the room beyond
  the weights stayed (6.9 GiB), and the sentence held by 0.2 GiB of 37.1. On the one-layer model the same sentence
  is 0.5 GiB above what the run reserved: the flat allowance of the reduced models is an estimate in both
  directions, as the section before says.
- A rule that only fails together with another is not held. The band has two rules, and a block scale outside its
  interval also puts every code of its block outside its bound: the test of the scale rule passed with that rule
  switched off. Under zero codes every scale gives the value zero, and a wrong scale byte there fails by its own rule
  alone. One-line defects applied to the code found it, and found that the test of "half a code step" stepped two
  codes, and that the chunks of a copy made from an interleaved weight were only read at one chunk length.
- A count pinned in a test moves with the table. The split rule of the READMEs carried the old numbers in its own
  pattern (`= 49 + 209`), so the rule would have passed a README that kept them. A test that names numbers of the
  table is derived from the plan where it can be, and where it cannot, it is on the list of what a new row changes.

## What the review of the gate / up row taught

Each of these passed every host test, every one-line defect and every GPU run of the first version; a host test now
holds each.

- An interval has two ends, and a test of one of them holds one. The block-scale rule of the band was tested one
  binade up, one binade down and at a lone byte above a zero block; the case one binade down failed through the code
  rule, so with the lower end taken out every test stayed green -- and the lower end is the one rule of the band that
  fails buffers nobody wrote: under a zero scale every code reads as zero, and a code's bound admits the whole value
  below a scale that is too small. Buffers of zeros are a test of their own, and so is the byte just below the lower
  end with the codes of its block written under it. Ask of every rule which wrong output it alone would catch, and
  write that output.
- "The operands this adapter built" is a sentence about operands. The first use of this entry hands its build a
  row count and nothing else, and the page of the entry names a requirement no public probe answers (the names its
  kernel is written on). A decline by that requirement ended in an `ERROR` that blamed the adapter for operands it
  had not built, where rule 1 asks for the entry's own sentence as a `SKIP`. Where an adapter hands a build nothing
  of its own, it has nothing to be blamed for there: hold what it does hand over (the row count) in front of the
  build, and let the entry speak.
- A question asked inside an error path must not raise. The adapter read the compiler of the installed CuTe DSL to
  word a decline; a DSL that does not report one turned the entry's decline into an import error from inside the
  handler. The entry itself declines nothing on a value it could not read, and neither does its adapter.
- "Two of X are refused" is tested with X and with what stands beside X. Two providers' layouts of one weight were
  refused before the load; a provider's layout beside the checkpoint's row order -- a block that multiplies on the
  stored tensor as it is -- was not, and ended after the read in the sentence of a session that was not planned
  with the run, although it was. What a load keeps is one row order per weight, whoever asks for it: every reader of
  the stored form has an owner in that comparison, and both orders of the two plans are a case each.

## What the verify step taught

The verify step feeds several tokens per sequence in one launch sequence (`Session(tokens=T)`, `Session.verify`, the
key `fe.gdn_verify_fused`). What building it showed:

- A step shape is a field of the form. The fused verify kernel is compiled per tokens count, and its name is cut in
  front of that count and of its launch knobs: two kernels of two counts share a symbol and a grid, and differ in the
  shared memory of a thread block. So the row names `TOKENS`, one row per count; the adapter holds the slot table of
  every call to it; and the call log states the form for the step shape of the table that call handed the entry
  (`_served_stated`). An entry whose launch facts take a step shape says which shape a row count stands for
  (`EntryBlock._step_of`): that is the plan's question. The call log's is another one -- what was this call handed --
  and the first version answered it with the plan's: it asked the instance for the row's own tokens and logged the
  echo, so with the adapter's guard taken away a table of two sequences x three tokens was logged as `TOKENS 6`. A
  field of a form that is an argument of the question says nothing as the answer: read it from the operands.
- A band stated for one token per call is applied token by token, not loosened. The band of the fused step tolerates a
  rounding that falls on the other side of a midpoint as an event, seen once, because the reference starts every
  call from the kernel's state. A reference that runs the `T` tokens of a verify step on its own state sees such an
  event at every later token of the step, and a cell of one token over nine steps fails a share that a cell of the
  decode step passes over 32. The fix is in what is compared, never in the numbers: the reference starts every token
  from the state the kernel wrote for the token before it (`gdn_verify_step(anchor=)`), and a cell is one value head
  of one sequence over the tokens of its steps -- one block of the launch grid, as the band says.
- The layout of one token per sequence is the layout of several at `T = 1`. Slots numbered token by token keep token 0
  of a sequence in its plain slot and its history in that slot; one state class then serves both kinds of step, and a
  plan of one token per sequence builds the tensors it built before. Check that with the shapes, in a host test.
- Keeping tokens launches nothing. How many tokens a step kept is one number on the host; the kernel's operand is a
  row of a constant table built with the state. A copy or a fill between two steps would be the first record of the
  next step, where the kernel check declares nothing.
- An operand of a kernel starts 16-byte aligned, and a row of a table need not. The slot table of one sequence out of
  several (the batch-invariance check runs a sequence alone) is handed over in tensors of its own, and the rows of the
  kept-count table are padded to four elements.
- A decline names the row of the step that was asked for. A verify step no row holds was first declined in the words
  of the decode entry, the block's first row ("serves rows 1-65,535"). A plan of several tokens per sequence is
  declined by the block's row for that tokens count, else by the last of its rows for such steps.
- A census section holds for the step it was recorded with: the tokens per sequence beside the row count, the widths
  and the class. Six rows are one sequence of six tokens and two of three; only the first is the recorded step.
- Two runs that feed a block other rows cannot be compared on that block's state. The verify run against the decode
  run gates the recurrent state where both were fed the same rows (the history taps say so) and prints it elsewhere;
  what holds in every layer is the block check, teacher-forced. A line that gated nothing is `REPORTED`.
- A row is run in every form its entry takes inside the row's range, not only at the range's ends. The verify entry
  picks its form by the sequences and the SM count: on a 148-SM part two thread blocks per value head at one
  sequence, one thread block per SM at two and three, two per SM from four. The first runs were at 1, 4 and 16
  sequences and never launched the middle form, whose shared memory at eight tokens is 704 B below the part's limit.
  Ask the entry's launch facts for every step shape a row claims, group them by form, and run one of each.
- A session of another kind of step does not inherit the report of the first kind. The report of a verify session
  printed the decode step's bytes per step, a prefill route of 0 tokens and a rate that counted steps as tokens.
  Until a line is taught the new step, the report says that it is not stated, and a host test holds the sentence.
- Two sessions on one load share weights, never a state. Say so beside the snippet that builds them: a reader who
  prefills a prompt in one and steps the other gets the tokens after an empty context, and no error.

## What the out-projection of a verify step taught

The target serves one block of the wiring with two kernels in a verify step: the out-projection of a linear-attention
layer is the 8-bit projection kernel, the out-projection of an attention layer a library multiply, and both multiply by
a weight of one shape (rows of `fe.fp8_projection_decode` in the block `projection.fp8.out`, for steps of several tokens
per sequence alone). What building it showed:

- A block is not always one row. Splitting the block in two moves lines of every plan of one token per sequence -- a
  table row, the by-provider line, the coverage line -- for a difference only a verify step has. The call sites
  carry it instead: the row of the entry names the site it serves and hands the other on, and a plan prints the
  block once per call site only where two rows serve it. A plan of the decode step is the text it was; hold that
  with a run of both trees, line by line.
- What a row hands on is data, never the adapter's choice. An adapter that served one site and called the fallback
  chain itself at the other would be one row with two kinds of census entry, which the kernel check cannot judge: an
  exact kernel at one call and a family at the next. The plan resolves the next row; the adapter is never built or
  called at a site its row hands on.
- A split is a row range. The target chooses the K slices of the projection kernel by the rows of a step, and the
  entry fixes them per instance (another split is another byte class), so each range is a row and an entry instance
  of its own, selected by `rows`. The bytes of the block change where a step's rows cross from one range into the
  next; say so beside the rows, and gate batch invariance inside a range only.
- A field that follows the call is named, or the call log cannot show it. The projection adapter already asked the
  serving instance with the rows of the operand it was handed, but a form without a field that follows the rows
  logs the same text at every row count. The rows of this block name the token tile of their range, and the entry
  takes it as a floor: an instance handed the rows of another range states another tile, and item 0 names it.
- The forms of a weight follow the call sites. One block, two rows: the linear-attention layers' weights are read on
  their stored codes, the attention layers' as copies. A load is asked per call site (`<block> at <call site>`
  beside the block names), so a verify plan alone holds no copy of the first and no codes of the second, and a
  decode session on the same load, which reads every out-projection as a copy, has its copies. Test the loader
  with each plan alone and with both, and the check that names a missing form with a load made for the other.
- A decline of a weight's form is asked per weight. The projection adapter declined a load that keeps a scale per
  merged part for every row of its entry; the out-projection's weight is one stored tensor with one scale under
  every merge rule. The family says which weights a rule gives a scale per row (`scale_per_row`), per call site.
- Two sessions of one load that serve a block with two kernels are not byte-equal behind it. Up to the last row
  count of the entry's rows the verify session rounds the out-projection once and the decode session twice: their
  logits differ by a bf16 unit and an attention layer behind caches other bytes, printed and not gated. From the
  first row count on the fallback row both multiply over a copy: the bytes met again at three and four sequences
  and stayed within half a bf16 unit at nine and 16 sequences of two tokens (a library multiply is another kernel at
  another row count). Run both sides of that boundary, and say which lines of the comparison are reported there.
  And state a comparison for the step shapes where it gated: a sentence that begins "at every step shape that ran"
  is read as a gate on all of them, also on those whose line was `REPORTED`. The memory sentence likewise: "within a
  tenth of a GiB" is a statement about every run, and one step shape reserved half a GiB less than it asked for.

## What one session for a prompt and its verify steps taught

A session built with `draft_tokens=` feeds its prompt through decode steps and runs verify steps behind it on one state
(`Session.prefill`, `Session.verify`, `Session.accept`, `tie_out.speculative_checks`). What building it showed:

- A comparison of tokens is blind where a model's token does not follow its state. The seeded model's output head is a
  permutation of its embedding table: three wrong loops -- the position moved by the kept count alone, the prompt's
  attention cache not the one the verify steps read, a row that attends over the whole window -- kept every greedy token
  and stayed inside the logits band at the published widths. Hold what such a loop gets wrong by what it is: the
  position as an integer, the prompt byte for byte, the rejected rows by a second replay with other tokens behind the
  kept ones. Each of the three lines fails one of those loops, and a host test on the tiny geometry cannot show that
  the token line does not: there the same loops leave the band. Run the wrong loops where the gate runs.
- A line that cannot fail by rounding is gated on every stack; one that can is gated where the two runs are one chain.
  `position`, `prompt` and `rejected rows` compare integers, or two runs of the same kernels on the same rows. `bytes`
  compares a verify step with decode steps: the same chain on the reference stack, other kernels for some blocks on a
  provider stack -- gated on the first, `REPORTED` on the second, and never given a band of its own.
- The greedy rule is part of the token line. A decode session that is fed the tokens a loop kept agrees with that loop
  row by row, whatever the loop kept: a count taken from a schedule, or a drafted token that went out, passed a
  comparison of logits alone. The line also holds that every window starts with the greedy token kept last and that
  every kept drafted token is the greedy token in front of it.
- A kernel on a view is another compiled form, under the same name. The decode entry's compile key holds the strides of
  its history operand; token 0 of a pool of several tokens per sequence has the plain taps and another slot stride. A
  device run showed that form byte-equal to the plain pool's, the rest of the pool untouched and the launch the same;
  the kernel check cannot tell the two forms apart and has no need to. The adapter builds the form with the plan (the
  plan's option says the state's tokens), or the first prompt step would compile it.
- Hand a block the tensor its band was written for. Token 0 of the recurrent pool is handed over as the null slot and
  the plain slots, not as the whole pool: the band of the fused step takes its share over the slots of the pool it is
  handed, and the untouched slots of the later tokens would have made that share easier to reach.
- The method that is called selects the plan of a step. A session with two plans looks a block up in the plan of the
  step that is running, set where the step starts; the state says what layout it is, never which kind of step runs.
  A decode step on a state that stands behind a verify step is a `ValueError` with its own sentence, where it was an
  error of a tensor shape.
- A replay starts from zero, so the record of a run holds everything its steps stand on. The verify steps of such a
  session began on the state its prompt left: the session keeps the prompt whole, and replay, the block check, the
  reference's pass and the traced pass feed it first. The traced window holds the verify steps alone -- the prompt
  is replayed in front of the profiler, and the report's line says so.
- One count for several sequences is exactly greedy only if it is the smallest of the set. The helper that counts
  returns that one, and the method that takes the count says so.
- A comparison that replays a run judges the replay. `speculative_checks` drives its replays through the family's
  step on the session's plans and state, and its first version read the inputs of the run and nothing else of it: a
  session whose own methods were wrong -- a count never told to the state, the position moved by the count alone,
  the prompt fed to a state the verify steps never read -- passed every line, and where the defect sat in a method
  the session's own checks replay through as well, it passed those too. Tie a replay to the run: the ids and logits
  the run recorded, byte for byte against the replay (`recorded`), and where the run ended, as integers
  (`position`). And put a control defect where a caller reaches the code -- in the session's methods -- not only
  below the path the check itself takes.
- Hold as an integer what is one. Which token the state stands on after a step was seen through the kept state
  alone, a band that gates a layer only where both runs fed it the same rows: on a stack whose front rows state no
  batch invariance a forgotten count failed nothing that was gated. The kept token is a number the state holds; it
  is compared with the count the step kept, beside the position, on every stack.
- A number counted over the rows of a step is not a token's number. The kept rows of a step stand sequence by
  sequence, so the first row that differs is the first kept token that differs at one sequence only. A line that
  names a token keeps the token's number and its sequence for every row it compares.
- What a first launch reserves outside the pool is reserved once in a process. A second plan of one session adds no
  second reserve to the memory sentence. The class with such a reserve was not the class of the gate, so the
  sentence of that class is held by a host test on stated facts.
- A check says what it looked at. The traced window of such a session holds its verify steps; the prompt's decode
  steps launch a compiled form of their own that the kernel check never sees. The run says so beside the verdict, a
  line that compares nothing is `REPORTED`, and a kind of step of which no step was judged is named as that.

## What building the drafter's pass taught

The checkpoint's drafter has its draft pass on reference blocks (`QwenDenseDrafter.body`, `propose`, `draft`; the table
`configs.DRAFTER_PROVIDERS`; the keys `torch_ref.pre_fc_norm_quantize`, `torch_ref.gate_row_quantize`,
`torch_ref.silu_mul`, `torch_ref.quantize`, `torch.argmax+lookup`). What building it showed:

- A quantiser per row of a step is a third rule beside the loader's two, and it is pinned to its operation like them.
  The static scale of the model's activations multiplies by a reciprocal that is one number per operand, a constant
  of the checkpoint; the loader's
  weights divide by a scale per row; the rows of a drafter's step multiply by the reciprocal of a scale formed from
  the row itself. The stated row that tells the division from the reciprocal for the weights tells them apart here
  too, in the other direction: two elements whose quotient is the midpoint of two codes take the code above. Name the
  operation in the docstring and hold both rules on one row.
- A division by a host number is not a division on a GPU. PyTorch divides a CUDA tensor by a Python number as a
  multiplication by that number's reciprocal, and by a tensor as a division: on bf16 values over 448 the two differed
  in the last bit on more than half of the elements, on the device and never on the CPU. A rule that says "a division
  in fp32" forms its constants as tensors on the operand's device, and no host test can show that it does: the device
  run compares the codes and scales of a stated tensor with the CPU's. The first form of that comparison had seven
  rows and passed a quantiser that divides by a host number, its few scales agreeing by chance; with 2,048 rows the
  same quantiser failed on 954 scales. Run a check against the defect it is for before believing it.
- A block that is the reference chain is judged by nothing, so its structure is held where it is made. The block check
  compares a provider with the reference; a row whose provider IS the reference has no line, in any stack, and the
  proposals of seeded weights mean nothing that a comparison of tokens could hold. Four things hold such a pass,
  and each sees what the others do not: every function against a version written from its definition in the
  test (an error inside a function); every block of the plan, as the plan builds it, against the function it stands
  for, and one pass through the plan's blocks against the pass written out from the functions (an error inside a
  block's class: operands exchanged, another function called, an epsilon of its own); a pass over several rows
  against that many passes of one row, byte for byte (rows that read each other, a scale shared across a step, a
  cache position written twice or not at all); and a recording stand-in for the blocks, which is handed what each
  pass of a step reads -- the row the step kept, the pass's own last output and proposal, its position (an error
  between the blocks, which passes every numeric test with seeded weights because every row is as good as the next).
- A defect is put into every layer a call goes through. The first one-line defects of this pass were made in the
  functions, the table and the pass, and the tests held them; eight made inside the classes of the new blocks
  passed every test. The function tests call the functions, the stand-in replaces the blocks, and a pass compared
  with passes of itself moves with the defect. What a block declares about its operands for a sequence run alone
  was read by no test either: a block that no check taps yet is run through the tap in a test.
- A value that a function hands on only in quantised form can still be read. The gate with one rounding and the
  model's gate with two differ by one bf16 unit and give the same 8-bit code on almost every element; a row of ONE
  element has that element as its largest, so its scale states its bf16 value exactly.
- A formula with a transcendental function is not pinned to the last bit by a double. An fp32 sigmoid differs from
  the double's by a unit in the last place, which moves a bf16 rounding only next to a midpoint: judge every element
  that is clear of one, count them, and assert the count -- on bf16 inputs the values repeat, so "one in two
  thousand" was one in a hundred and sixty.
- A count of a pass is a count of block calls, and it is the wiring's: 18, of which the body is 15 and the head 3.
  One block may stand for more than one operation of another implementation (the attention preparation also writes
  the cache), so a count read off another table is not this one. This area states no count of the launches of a
  draft pass: no kernel census holds one.
- A call says which pass it belongs to. The model's step hands every block its layer; a drafter has one layer and a
  step has several passes, so `call(block, pass, *operands)` carries the pass where a step carries the layer, and
  the wiring lists one pass. Whatever records the calls of a step can then tell its passes apart without counting.
- A reference row names no form. A provider row that names call sites with fields is compared with what its entry
  instance states, and a reference chain states nothing: the five sites of the drafter's multiply and the two of its
  norm are data beside the table, handed to the blocks by name and read by the README, until a kernel's row takes them.
- An arm that takes an input away has to take it away. On the checkpoint the pass's first proposal was the model's
  token at 167 of 207 positions. With the two halves in front of the first multiply exchanged it was 0, with the
  hidden row replaced by zeros 8 -- and with the hidden row of the position before, 108: a row one position old is
  a row of the same text, and the token embeddings and the cache carry the rest. An arm that shifts an input is a
  measurement of how much the shift costs, not a collapse; before a figure is held against an arm, run the arm and
  see what it leaves.
- What a load keeps for today's blocks is said where the load is shown. The reference blocks multiply on bf16 copies
  that a kernel on the codes will not need; a default load makes them, the manifest's last line counts them, and the
  argument that leaves them out is named in the same sentence.

## What the speculative step taught

A session that is handed the drafter runs the whole step of speculative decoding (`Session(drafter=)`,
`Session.speculate`: the drafter's passes, one verify step, one host read; `tie_out.speculative_checks` with
`drafter cache` and `draft proposals`). What building it showed:

- The cut of a step follows the one host read. The count a step keeps is decided on the host, after the read, and the
  first pass of the drafter needs that count: so a step is passes, verify step, read, and the acceptance follows the
  read. The first step of a run then has no verify step in front of it and its first pass runs at one row per
  sequence; every text that states the rows of a first pass says which step it means.
- What the drafter reads of the step before it is state, and it goes where the state goes. The final-norm rows, the
  greedy ids, the kept count and the drafter's cache are reset, snapshotted and restored with the model's state; a
  cache kept beside the state would have carried the first prompt set into the second, and nothing reads a stale
  position as an error -- an unwritten or a stale position is read as what it holds.
- For the same reason a step that would leave the cache behind is refused, not allowed: `verify` on a session with a
  drafter would run the model's step without the passes in front of it, and the next first pass would then go on
  from a position nothing wrote. The step with the caller's drafted tokens is `speculate(force=)`, where the drafter
  runs all the same.
- A check that calls the function it checks holds nothing. The second run of the passes never calls `draft`: it
  writes the passes out at one row per call from its own books of kept tokens and rows. And because both runs call
  the same blocks, its two lines hold what a step hands its passes -- rows, tokens, positions, the row a step goes
  on from -- and no arithmetic; the text beside them says so.
- Each of the two lines sees what the other cannot. A first pass one position off, a prompt position without its
  pass, a drafted token where the kept one belongs and the rows of an older step fail `drafter cache`; the row
  select at row 0 and a later pass on the wrong row, the wrong proposal or the wrong position leave the cache up to
  the last kept position as it must be and fail `draft proposals` alone. Eight wrong passes, each in a loop that
  still emits the plain session's ids, hold that split in a host test.
- The window of a step is the session's. With `verify` the caller builds the window, so a loop that emits a drafted
  token feeds it back and the token line fails; with `speculate` the window starts with the session's own greedy
  token, the emitted token goes nowhere but the output, and no line of the comparison sees it: the run's ids
  against the plain run's do.
- A traced step is the step the loop runs. The window and the pair are a concatenation each, launched between the
  passes and the model's step and behind it; a traced pass that replayed the recorded window through the verify step
  alone would have held a step no loop runs. The traced step is `speculate`'s own code, fed the recorded drafted
  tokens, and its one host read is the pair's.
- An index into a wiring moves when the wiring grows in front. The census records launch shapes by the index of a
  site of the model's wiring; with the passes' sites in front of them an index that is not shifted names a site of a
  pass, where no exact launch stands, and the shape is then held against nothing -- silently. `Traced.first` shifts
  it, and the host test holds a launch in another shape at the model's site with both of its sentences.
- The comparison counts the launches of one kernel id; it does not place them. The embedding lookups of the passes
  and of the model are one id, so a lookup more is named with the count of the step and the plan, in front of the
  run that matches, and a lookup less is named at the first pass.
- An arm that shifts an input is not an arm that takes it away -- again. On the checkpoint, on the cuDNN Frontend
  rows, the loop kept 3.22 tokens a step; with zeros in the place of the hidden row 1.04, with the rows of another
  text 1.05, with the two halves in front of the first multiply exchanged 1.00; with a later pass one position off,
  3.25. Acceptance does not see that shift at all: the line that holds the positions of the later passes is
  `draft proposals`, on bytes and ids.
- Working memory is stated as arithmetic on what a pass holds at a time, then measured. A reference multiply
  accumulates a chunk of the weight in fp64; the allowance of a pass is twice the largest such chunk, 1.33 GiB at the
  published widths -- not a share of what the drafter holds. On the checkpoint, on the cuDNN Frontend rows, the run
  asked for about 43.3 GiB, reserved 40.0 GiB and grew the device by 40.2 GiB, the GPU otherwise empty.
- The counts of a run live with its record. The comparison resets the session and puts its record back; the kept
  counts of the acceptance lines are part of that record, so the acceptance lines of a report printed behind the
  comparison are still the run's. Its bytes of a step are not: they follow where the session stands, and the
  comparison leaves it at position 0 -- a run prints its report first. Every step counts in full, the last one
  included, whose tokens behind an end token are not emitted.
- No count of the launches of a pass is stated, in any text, while no census section holds a draft pass: the kernel
  check attributes them, the run says "kernel identity of the drafter's passes: not stated", and a count of block
  calls -- 18 a pass -- is the wiring's.
- What a step hands on by reference is held by identity, where it is handed on. The session and the second run of
  the comparison read the same attribute of the state, so a step that left another tensor of the same shape there --
  the rows in front of the final norm, a layer's normed rows, a copy -- passed every line of both checks and every
  host test. One test holds it through the session's tap: `state.hidden` is the tensor the final norm's call
  returned, the one whose quantised rows the head read, and `state.greedy` the one the token choice returned.
- A traced step is held to the recorded step, not to its shape alone. The kernel check reads no token value: a
  traced step whose window began with another token, or whose passes drafted on an empty cache, passed it. The host
  test with the stand-in for the profiler holds the window the traced verify step is fed and what its passes
  propose against the record.
- A refusal stands in front of everything the refused call would leave. A step asked for in front of the prompt was
  refused by its passes, behind the mark of a launch on the report's clock: the read that ends the prompt was then
  taken for a step's and the first-token line read 0.0 ms. And a refusal names the step of the session that was
  asked: on a session with a drafter that is `speculate`, also where the session refuses a decode step or a count.
- The names beside a snippet are run as the text says they may be. The README's loop said "a described checkpoint"
  and built no session with it: a session that was not handed the checkpoint drew seeded weights, and the drafter's
  with them, without a word (it now refuses the drafter of a checkpoint by name). The loop is now run as written on the seeded pair and on a checkpoint a test
  generates and describes, and the table it runs on is held against the file.
- What describes shared weights is read at the session that loaded them: a session that shares reads no checkpoint
  of its own, and a `seed=` says nothing about a checkpoint a session was handed. A drafter two links of `share=`
  behind the loader had nothing to load from, and the drafter of a model handed seed 11's checkpoint was drawn from
  seed 0 with every token still right, since the verify step decides the tokens; the load now follows `share=` to
  its end (`Session._weights_owner`) and draws a seeded drafter from the seed its checkpoint records.

## What building the speculative script taught

`02_speculative_decode.py` runs the speculative step for one request: its own loop, the speculative configuration of a
serving engine key by key, the comparison with decode steps and the checks of every run. What building it showed:

- A configuration that selects what is launched has no class "printed and ignored". A request body's unknown field is
  named and ignored; an unknown key of a speculative configuration would change which kernels run. So its tables are
  read, neutral and refused, a key outside them is an `ERROR`, and the keys that are read must be named: an engine's
  default for a missing one selects another launch set. What a refused key holds is not printed back.
- A script that compares where another checks nothing needs the verdict's order. `_verdict` answered the reference
  stack before it looked at the checks; the speculative script compares its run with decode steps on that stack
  too. A failed line is `FAIL` on every stack, in both scripts, and one host test holds the two functions to each
  other and to the README's table over every input.
- What the library's comparison cannot see, the script holds itself. The window of a step is the session's, so a
  drafted token a loop emits reaches no line of `speculative_checks`; the ids of both arms do, and the script
  compares them. Three wrong loops run through `main()`, on a host and on the device. A count kept beyond the
  compared one also fails the drafter's two lines: the first pass of the next step takes the greedy token of a kept
  row for the token that follows it, which is true of an accepted draft alone.
- A default follows what it is the default of. A scheduled run needs enough new tokens to keep every count once: 17
  at five drafted tokens, 38 at eight. A fixed default of 32 answered eight drafted tokens with an error about the
  token count; now the default follows the schedule, the plan declines by its row, and a typed count is still held.
- A flag that is declared in order to be refused is still a string of the script. A test that took a flag for a
  script's own when its string stood in a script of the folder passed for `--draft-vocab` whatever the speculative
  script did; it now asks each script whether it takes the flag.
- A refusal belongs where the object is built. A checkpoint's drafter on a session that was handed no checkpoint
  proposed from rows of a seeded head without a word; a drafter of another model's widths ended in a tensor size
  inside the warm-up step. Both are sentences of `Session` now, before a plan.
- A stand-in that rounds otherwise is not given the product's band. On the tiny geometry of the host tests the fp32
  chains of the stand-in providers leave the seeded model's logits band at six tokens per sequence: the test of the
  last line holds `PASS` at three drafted tokens and says, for the default, which line fails.

## What the rows a library serves taught

Three blocks of the target kernel set are multiplies of a library, and the stack `native` calls FlashInfer for them on
the stored weights (`providers/flashinfer_calls.py`). What building these rows showed:

- A library's pick of a kernel is a function of the call only until something tunes the operation. Outside a tuning
  context the 4-bit multiply launched the same kernel class in the same grid for a weight shape and a row count every
  time; inside one, the winner was not the same kernel in every process, and it outlived the context. So the row
  names what makes the pick a function of the call (`TUNED False`), the adapter reads the library's tuner after each
  call and the call log carries what it found, and the kernel class and launch shape of the untuned pick are census
  data that the kernel check holds exactly. A row that cannot name such a condition holds its launch by membership and
  prints the pick: the 8-bit multiply. Say what such a field sees: `TUNED` is the tuner's statement about this
  process, and a tuning store a caller attached is outside it.
- A launch held by membership is the same call, not the same kernel. In every run of these rows the library's untuned
  8-bit multiply was its own CUTLASS kernel, not the cuBLASLt kernel the census records as the target's pick. A count
  of target launches that includes such launches says how many are held to a recorded launch shape and how many by
  membership, wherever the count stands.
- The class and the grid of a pick follow the weight shape: the output head of a reduced vocabulary is another kernel
  launch than the published one's. The census entry of such a call names the widths it was recorded at (`at`), and a
  run of another width says that no launch shape is held for that block instead of failing.
- A library that builds kernels at its first use must make that first use while the run is planned. Its first build
  of a compiled module took more than a minute and printed nothing; inside a decode loop that would have been a
  stalled step. The first use is the call on scratch operands at every row count the run will call the block with --
  its own, and the rows of one sequence, which the batch-invariance check calls: the library builds per range of row
  counts -- after the memory sentence and before any load, announced in one line; a decline there is typed -- the
  adapter's operands an `ERROR`, the installation a `SKIP` with the type of the exception and never its text, which
  can hold a local path. The catch that makes the `SKIP` is around the library call alone: around the adapter's own
  code it would turn a fault of the adapter into advice about the installation, with exit code 0.
- A library kernel's name may reach the profiler mangled, and long: two kernels of one template differ far from the
  start of their names. The family pattern of such a kernel names what the declared call launches (the mainloop, the
  operand type), not the head all kernels of the template share; and a line that prints such a name prints its length
  and a digest with it.
- What a census records is recorded per call. A line that prints it for a block whose call sites two rows serve looks
  it up by call site; by block name it printed one row's recorded launch shape for the other row's kernel.
- A layout that a provider builds of a stored weight at load is held to the bytes it was made from, at load. The
  reference's copy of such a weight is made through the layout's own way back to the checkpoint's order, so a wrong
  layout agrees with its reference in every later check of the run.
- A third-party adapter is tested as it is, over a stand-in of the library: the adapter reaches the library through
  one method, and a host test replaces that method. A stand-in of the adapter would test the stand-in. The stand-in
  of the library has the library's own defaults and records what it is handed, or a call without the row's argument
  passes on the stand-in and runs another kernel on a device.
- A library's tuner may start worker processes that import the main module again. A script that calls a sample
  inside a tuning context of the library needs its `if __name__ == "__main__":` guard, or every worker runs the
  sample.
- When a stack gains rows, every dated run of that stack in a README becomes a run of another row set. Say which
  rows a dated run selected, and move the counts that the tree computes to the stack that still selects those rows.
- A figure of a greedy run on a checkpoint is a figure of a row set as well. A library's multiply is inside the band
  of the reference multiply and is not the bytes of the multiply it replaces, so the run follows its first other
  token: with the three library rows the plain run of the default prompt was another text, and the speculative loop
  kept 3.27 tokens a step where it had kept 3.22 on the cuDNN Frontend rows. Such a figure is stated per row set
  that ran, each with its versions; it is never carried over.
- A row a stack gains is a row of every script that plans the stack. The speculative script arrived while `native`
  had no library row. With the rows it made the library's first use without the line rule 5 asks for, its help text
  stated the memory of the other row set, and the runs its README dated on `native` were that row set's. When the rows
  of a stack change, every script of the family is run on that stack again, and what it says of the stack is read.

## What the attention preparation's row taught

The attention preparation is the first block on an entry that writes a cache: `cudnn.AttentionPrepareDecode` writes the
key and the value row of every row of a step into a pool of pages. What building the row showed:

- A kernel that writes a cache decides the cache's layout, and the layout belongs to the state, not to the block. The
  adapter names the page it needs (`cache_page`), the session builds one state in that layout, and every other block
  reaches the cache through its accessors -- views and host integers alone. An accessor that gathered through an index
  tensor would put a launch between two rows that are judged one by one, and the kernel check would name it.
- What tells a kernel where a row goes is a table the state builds once: its positions, its slots, its cos / sin rows.
  A judge that read the same tables would agree with a table that is one row late. The judge forms the rows of the
  pool from host integers, and the twin forms its cosines from the position it is handed: on a GPU a slot table and a
  position table shifted by one row, an exchanged rotary half and a write accessor that exchanges two pages each
  failed a run on the entry's line.
- A run's own checks hold what the kernel wrote, not what the next block reads. The accessor through which the
  attention core's reference chain reads the pool, with two pages exchanged, PASSED a seeded run on a GPU: no check of
  a run compares what that chain reads with anything, and the layer moves the seeded logits by less than their band.
  That accessor is held by tests -- the host case of a session on stand-ins, whose tokens differ, the host case of the
  paged cache against the plain one, and the cells of the GPU test that read the pool by index arithmetic of their
  own -- and a check on a device of what the attention core reads belongs to the row of the core's own kernel. A
  lesson says that a defect failed a run only of a run that was made.
- The twin of an entry is the twin of the entry's own suite, statement by statement, and the band is the contract on
  the entry's page. Where the page states the kernel against an evaluation in fp64, the twin evaluates in fp64 whatever
  precision the run's reference accumulates in: a twin in another precision is held to a contract nobody stated.
- A stand-in entry that computes the twin's own arithmetic shows nothing about a band. The stand-in of the host tests
  runs a kernel-like fp32 chain, so that some elements are another byte than the twin's and every one is inside; each
  defect is then placed alone -- an element two units off, a gate bit, a value bit, a key row at the neighbouring
  slot, a written row no position names, a NaN.
- The adapter passes every field of the row's form as an argument and leaves none to the entry's default, and a row
  without a field is refused by name: a default can move with the entry, and the form would still read as the target.
- A form is compared by value where a census file and a table meet: a list in the file and a tuple in the table are
  the same sections. And a width the kernel bakes into its name is a width of the row: a checkpoint with another
  rotary section stays on the reference chain, and its run states no kernel identity.
- The first use builds the entry before it allocates anything, and its scratch pool is one page: a typed decline of
  the installation costs no memory, and a decline of the adapter's operands is an error in the entry's own words.
- A row that changes the layout of a state changes what a run holds: a pool is whole pages, so a memory sentence can
  move by a tenth of a GiB where a host test pinned it, and a dated run keeps the figure of its own layout.
- A row a stack gains moves every table that follows the plan without a change of its code: the per-kernel table of
  the performance mode showed the two attention blocks as one fallback row, and shows an exact row and a fallback row.
