# samples/models — Agent Guide

End-to-end model samples: one process, text or token ids in, text or token ids out, a printed table of who serves each block,
and a self-check against a plain-PyTorch reference. Start with [README.md](README.md) and the family README
([qwen_dense/README.md](qwen_dense/README.md)); repository rules are in [../AGENTS.md](../AGENTS.md) and
[../../AGENTS.md](../../AGENTS.md). Host tests in `test/python/misc/test_model_samples_*.py` hold sentences of this file
verbatim: run them after editing it.

## The tripwire

A model sample has no socket, no request queue and no admission of work while a step runs. The set of sequences is fixed
before the first step. A model sample keeps no state between two prompt sets, has no cache keyed by content, and chooses
nothing at run time: every provider and route is a row of the printed table. The list of provider keys is closed; it changes
only together with a kernel, with the reference chain of a block that a kernel is to serve, or with a declared library
entry. A library's own tuner may choose among that library's kernels for a call a row names; the choice is printed with
the kernel check and is never a row. A change that needs any of these belongs in a serving engine.

Also out: a `generate()` method on the helper layer (the loop is in the script), stop strings, constrained output, logit
processors, tool-call parsing of the output, automatic choice of a stack or a route. A field of a request body that asks
for one of them is a row of the table of rule 18 and is refused there by name, never served in part. A field that table
does not hold is printed as not known and ignored, so a new name for one of them goes into the table.

## Rules

1. **Run or skip cleanly.** Before any load, check the GPU, the library versions and the free memory for every sequence
   count of the run (`Session.plan_table(later)`: one memory sentence for the whole run). Print `[NN] SKIP <requirement>`
   naming the fix and exit 0. After the plan nothing skips, and no `SKIP` follows a check that ran; a test that honours a
   `SKIP` looks for a failed check in the output first.
   - A provider that declines the operands an adapter built is an `ERROR`, never a `SKIP`. So is an entry that declines
     the geometry or row count of a row that claims the device (a wrong table).
   - An adapter asks its entry at plan time, on the real device, for the launch at the run's row count, and builds the
     entry's kernel for that row count on scratch operands. A toolchain requirement the entry names there is a `SKIP` in
     the entry's own words, before the load; a decline no nameable requirement explains is an `ERROR`.
   - An entry built from the row count alone, after stating a launch for it, is handed nothing a decline could be about:
     every typed decline of that build is the `SKIP`.
2. **State the environment before the first load.** `check_environment()` compares the installed cuDNN Frontend
   distribution with `cudnn.__version__`, says whether `cudnn` is an editable checkout and prints a source hash (never a
   path). A mismatch is an error naming both. It runs after the plan, so an unserved GPU still skips.
3. **`import cudnn` before `import torch`.** The helper layer never sets the process-wide switch for the opt-in FROST graph
   engines; a sample that needs one pins it for its own graph. A version-gated sample checks both the compile-time headers
   and the loaded cuDNN library.
4. **Public API only.** The one host read of a decode step (the chosen ids) is in the sample's loop, outside every block.
   A block's `__call__` launches on the current stream and neither reads the device nor synchronises.
5. **No downloads.** Weights and tokenizer files come from a local path, read without pickle. Print every reduction
   (depth, vocabulary, seeded weights); a seeded run prints a hash of the produced ids and "the output is not language".
   - Describe a checkpoint from `config.json` and shard headers first (`Checkpoint.describe`, the family's and its
     drafter's `describe`): the plan, the memory sentence and every decline come before a tensor is read.
   - No output names a local path; print a file or directory by its own name. Commit no tokenizer, template or weight
     file of a real checkpoint; tests generate what they read.
   - A third-party provider may build its own kernels at its first use; a sample makes that first use while the run is
     planned, before anything is loaded, and says so in one line.
     A provider's precompiled kernel binaries are not built: they come from a package of that provider at a pinned
     version (`constraints-native.txt`) or from a directory the user filled beforehand, never from a fetch inside a run
     -- a sample switches the provider's own download off before its first use, and a binary that is not on the machine
     is a `SKIP` that names what is missing and the fix.
6. **The stack is explicit.** `fe-cublas` unless `--stack` says otherwise; never chosen by what is installed. `native` is
   the target kernel set of the family's census; `fe-entries` is its cuDNN Frontend rows with every other block on a
   fallback row, so it runs without further packages. `--stack reference` reports `[NN] RAN reference stack (no provider
   checked)`, never `PASS`; so does a run whose `--provider` flags leave no provider block to judge (`[NN] RAN (no
   provider block was judged)`, blocks line `REPORTED`). `PASS` means provider blocks were compared and held.
7. **Provider selection is data.** A family's `configs.py` holds ordered `ProviderRow`s per block; the first row whose GPU
   classes, sequence range, tokens-per-sequence range and stack contain the run serves the block. An entry that serves
   some layer widths only names them in its row (`widths`); a model with another width falls to the next row and the run
   says no kernel identity is stated, never `ERROR`.
   - To add a provider: add its key to `providers.KEYS` with its factory; place its row above the rows it should win
     against and below any row of the same key that an existing `--provider` command selects; name its tie-out band;
     regenerate the family README tables (`configs.readme_table`). A host test keeps them in agreement.
   - A third-party provider is imported inside its adapter, on a device; its version is read from the distribution's
     metadata there and held to the validated list, and a missing or unvalidated version is a `SKIP` that names it, the
     requirement file and the nearest command that runs without it. A plan on stated device facts imports no third-party
     package.
8. **Bands are never widened in place.** The band of a block against the reference is named in its provider row
   (`tie_out.BANDS`).
   - A provider that rounds at other points than the reference chain is compared with its twin (`Block.twin`): a reference
     written at the kernel's rounding points, in the reference's accumulation precision. The band is the kernel's
     documented contract, never a width covering the distance between two chains.
     The one exception is a library kernel that documents no numerics contract and whose rounding points are not
     established: its twin is the reference chain itself, which also keeps the operand terms of a bound, and its band is
     that bound -- derived from a stated model of the kernel's arithmetic, declared with that model before any run of
     the row, held by control arms that must fail, and never taken from a measured distance; an element outside it is a
     finding, and the family README says beside the row that the row is not within one bf16 unit of the chain.
   - A band names the outputs it applies to; an unnamed output is judged by its dtype, and updated state byte for byte.
   - Every band asks for finite values and is judged element by element (`(distance <= limit).all()`; a running `max`
     passes NaN). A share over a run is also held per unit of failure of the kernel (fused step: one value head of one
     sequence), padding (the null slot) excluded from the denominator.
   - Declare a band before the run, never from the run's own measurement; the family declares its logits band
     (`logits_band_units` in `configs.py`). Where no band passes two correct stacks and fails a wrong one, declare `None`:
     the line prints `REPORTED`, gated on finite values only. A line that cannot fail on a finite value never prints `PASS`.
   - An adapter launches its entry and nothing else. Conversions a fallback consumer needs (a gather out of the kernel's
     scale layout, a quantisation) are launched by that consumer when it reads the operand (`weights.dequantize`). An
     operand that reaches a kernel in a layout it does not read is converted by that kernel's adapter, with its own launch.
   - Weight forms: an adapter that multiplies on the stored format says so (`stored_weights`); one that reads its own row
     order names it (`stored_layout`, built by the entry's helper). The load keeps one stored form per weight (never a
     second stored tensor, never a layout made inside a call) and a bf16 copy only where some plan of the run reads one,
     decided from the plans of every sequence count before the first load. A block whose form is missing is an error
     naming it.
   - Reference copies: where the load kept none, the checks make one per call (`weights.OnDemand`) from the stored form
     into the checkpoint's row order; the memory sentence states it and the checks print the cost. Only the reference
     session of the checks may make copies per call; any other session is refused the maker. The loader checksums every
     weight kept only in stored form, and the checks compare it before their first pass and after their last (`stored
     weights`: fails by block and layer). An adapter never writes into its weight.
   - A row that is a reference chain or a multiply over a dequantised copy carries a `fallback` reason. A row no test of
     this repository runs prints `unverified in CI`.
9. **A default run is a test.** A GPU test loads each sample's default run by path and records its verdict in the job log;
   a required run that skips is uncovered, not passing. Host-only behaviour is tested under `test/python/misc/`.
10. **The reference accumulates in fp64** and keeps every intermediate cast of the chain it models
    ([../../test/AGENTS.md](../../test/AGENTS.md): "fp32 operands and TF32 references"). The report prints its precision.
11. **No performance table in a sample README.** Printed timings are orientation only; publish numbers through the
    `benchmark/` folder's formal mode. Licence header on every file; repository formatter settings; no absolute local
    path in any output.
12. **Sample-only packages stay at the edge.** `samples/models/requirements.txt` holds them with lower bounds (never pins)
    and a "Validated with" line; none is a package dependency. Import each inside the functions that need it
    (`TextIO.from_checkpoint` and the stream decoder: `tokenizers`, `jinja2`; `Checkpoint.describe` and the tensor read:
    `safetensors`), never at module import; a missing one is a `SKIP` naming the requirement file. A lower bound is a
    version the host tests of the readers ran against; raise it with evidence, in the change that needs it (`jinja2`:
    with a sandbox fix). The family README states the checkpoint (repository, size, licence field as on its card, date
    read, manifest hash), package versions and GPU class of the last checkpoint run. Version updates come through the
    `pip` entry for `/samples/models` in `.github/dependabot.yml` and are handled by the repository's maintainers. A library whose
    kernels a stack calls has a requirement file of its own, a lower bound with its "Validated with" line
    (`requirements-flashinfer.txt`), and `constraints-native.txt` holds the validated versions as exact pins: the
    versions the launch records of the family's census were recorded with. The maintainers of this area own that list; a
    version joins it with a recorded run, never with an update of the pin alone.
    The one pin that is not a version of that record is the library's kernel-binary package
    (`requirements-flashinfer-cubin.txt`): the record's runs read kernel binaries the library had fetched, and the
    package joined the list when its files were held byte for byte to those files and a run on it launched the recorded
    kernels in the recorded launch shapes (the census names that run: `validated.kernel_binaries`).
13. **The report's GPU side comes from kernel activity records, never from a pair of CUDA events** (a pair spans every
    host wait). The traced pass (`Session.trace`, `timing.trace_steps`) replays recorded steps under the profiler; a step
    ends at its one host read, and a block that reads the device shows as an extra step, which the pass reports. Only the
    profiler's own start and stop (and, where launch records are asked for, the export and read of its trace) are
    reported as a missing trace, in a sentence naming no path; an error of a step under it propagates.
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
14. **A user error ends in one `[NN] ERROR` line, never a traceback.** Raise a `ValueError` whose text names the file by
    its own name, printed before the plan, for: a path that is not a directory; a missing file or one that is not what its
    name says; an incomplete checkpoint copy (missing shard, or a shard whose size differs from its header); a side-file
    entry of the wrong kind; a request file that is not JSON; a body field that would change the output; prompt plus new
    tokens over the context length; messages a template cannot render or renders to no token; an unwritable output file
    or an empty file-flag value (test flags with `is not None`, never truth); a standard stream closed before start (it is
    `None`). A reader of standard output that goes away (`sample | head`): stop writing, run on, last line an `ERROR`
    keeping the verdict. Whatever a checkpoint's chat template raises is its refusal; the template is untrusted, compiled
    in `jinja2`'s immutable sandbox, and a host test holds templates that try to leave it.
15. **A kernel is declared, then observed; a name proves no form.** Host tests: `test/python/misc/test_model_samples_census.py`;
    each constructed defect fails with its own sentence, and the last line of a run is held by running `main()` to its end.
    - Every adapter declares its census entries (`kernels`, `model_samples/census.py`): the kernels one call launches, as
      `exact` (name, order, count, launch shape where the entry states one) or as families of a `fallback` row or the
      `host` path. A new adapter comes with its entries; an operator whose kernel no pattern of `census.FAMILIES` names
      adds the pattern in the same change (an unnamed kernel fails the check by name).
    - A provider row names its entry's form (`form`); the adapter passes exactly those fields; `--check-kernels` compares
      them with the entry's answer on the objects. An entry row that states its launch must name at least one field.
    - The call log is an observation: after each call the adapter reports which instance it called and what that instance
      states for the row's fields (`serving()`), never a copy of the row. One log per traced step; a step without its log
      fails. The check then holds the profiler's record of every launch of the traced steps against the plan. Without those
      records the run ends `RAN <stack> stack (kernels not observed: <reason>)`, never `PASS`; a failed check stays `FAIL`.
    - Several forms per step: the row lists them by call site (`sites`); the wiring names every call's site (`sites()`) and
      the step hands it to the block; the adapter holds one instance per form and serves each call with its site's instance.
      Every check item holds per call site, found by its place in the wiring (a block can be called twice in a layer). A
      site name belongs to one block; sites of a row may differ in operand shape as well as form. A row may serve only some
      sites, naming the others without fields to hand them to the next row. The plan then holds one part per site
      (`blocks.BySite`; `Plan.at`, `Plan.parts`), and everything read per call (census entries, form, identity, band,
      batch invariance statement, stored weight form) reads that site's part. A site no row names is a wrong table.
16. **A target kernel set is data, and a statement about it is three numbers.** A family's `kernel_census.json` holds the
    target kernel set of its decode step: per block call the kernel, the target's form, and per GPU class, SM count and row
    count the recorded launch shape.
    - A block's standing (`target`, `stand-in`, `not yet`, `not the target`) is decided against that file from the row's
      kernel and form, never written into an adapter or row.
    - A count of target launches is always three numbers (the target's own compiled kernels, cuDNN Frontend entries with
      byte-equal results, declared stand-ins), in every README sentence, table cell and run line. The numbers count a
      complete stack; a text that gives them says how many entries a stack calls today.
    - Byte equality is the entry's own record (census `tie_out`, `differences`), never a sample measurement. A form field
      only the port has: row and census name the value the byte comparison ran with, `differences` lists what the target
      does, the family README says so.
    - A section is `claimed` only on a `recorded` basis with no fallback row selected on it; otherwise the run prints
      `NOT CLAIMED` with every reason. A section of one GPU class, SM count, row count or set of layer widths is never used
      for another. `EQUAL` says what was compared; a field the reader does not know is refused, never read as "nothing
      recorded".
    - A pull request that changes a form the census names updates the census in the same commit. Public text calls it "the
      target kernel set" and names no product, repository or person for it.
17. **Standard output is the generated text; every line of a run is standard error.** All `[NN]` lines, tables, report and
    checks go through one function to standard error. The text of sequence 0 of a checkpoint goes to standard output (or
    `--text-out`) and nothing else does: written piece by piece in the loop, flushed, between the host read of a step and
    the next launch (counted in that step's turnaround, never inside a block) -- or once, after the report, with `--no-stream` and always in the performance mode. Seeded weights write no text. A request gets
    one text: with several sequence counts the first count writes it and later counts report whether their sequence 0
    matched. No run line goes to standard output even when standard error is closed (`print` with no stream writes to
    standard output). The GPU test of a default run holds both streams, in this process and in a fresh interpreter.
18. **A request body is answered field by field, beside the script.** A sample that takes a serving engine client's body
    parses it in one module beside the family's scripts (`serving_options.py`), as data: fields read, fields stated or
    unused, and for every output-changing field the values at which it changes nothing.
    - Such a field at another value is an `ERROR` naming the field, value, reason and what is accepted. An unknown field is
      printed and ignored; nothing is dropped silently or decoded otherwise than the table says.
    - The table holds every client name for a feature the tripwire leaves out; the host test names them by feature.
    - The printed line holds field names and values the table knows; it never echoes what others wrote: of `model` the last
      part, of an unused field its name, a name that is not a plain word as a JSON string.
    - The helper layer gets messages, tool schemas and template options, never a body. A serving-engine option keeps its
      name and meaning (`--max-tokens`, `--max-model-len`). A new table field comes with its case in
      `test/python/misc/test_model_samples_text.py`.
    - Pre-load lines for serving-engine users (decoding rule beside the checkpoint's generation settings, decode steps
      before the first new token) state a rule and a count, never a time.
    - A configuration object of a serving engine that selects what is launched is answered key by key in the same
      module; a key it does not hold is an `ERROR` that names it. Each key is read, accepted at its neutral value, or
      refused with its reason, with its case in the same test module.
19. **The performance mode states what it observed, never what it assumes.** One definition per statistic, written in the
    report file, with a host test of its arithmetic on constructed time stamps. Nothing but the step runs inside a timed
    request: no recorder, no call log, no profiler, no text; and no profiler pass runs in the process before its last
    timed request. The report says at which positions the blocks were held against the reference, that every timed
    request gave the checked request's tokens and state, and that the kernels of the traced steps are the plan's; a run
    in which one of the three does not hold prints no rate, and kernels that could not be observed are reported as not
    observed, the rate staying `orientation` with that reason. Another process, the clock and a limit reason are observed
    or printed as not observed; `none seen` is printed only when the run's own process is in the driver's list; the
    report never prints the word 'locked'. The rate of a run that is not a controlled run says so in its own line, and
    line 2 says it of every other number. A report holds no speed-up, no baseline and no figure of a second GPU class.
    The screen is rendered from the data the report file holds and from nothing else, at no more than 110 characters a
    line; a line of it that would be longer breaks at a separator or between two words. An ordinary run prints what it
    printed before the mode existed and does not import the mode's module: the mode is a branch of its own in the
    script, and a pull request that touches the loop, the census, the checks or the timing shows the area's plain runs
    line by line against the tree it started from (`test/python/misc/test_model_samples_perf.py`).
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

## Checks, tests and kernels: lessons

Each item below once let a wrong provider or a wrong text pass every test; the host tests now hold the defect.

**Verification**
- Verify a provider per block, teacher-forced on the same operands. With 8-/4-bit quantiser chains and 4-bit activations
  an end-to-end logit or token comparison is only as sharp as the model's margins; print a second reference's distance
  beside the provider's instead of implying the token check gates more.
- A bf16 multiply through PyTorch may reduce in bf16 (`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`,
  by shape, row count and GPU). A provider that claims fp32 accumulation switches it off for its own call.
- A band in units of the run's largest value changes width with that value's binade. A gate that can fail a correct stack
  and cannot fail a wrong token is printed `REPORTED`.
- The batch check rotates which row runs alone. Count distinct inputs, not runs (a fixed template preamble makes prompts
  share steps); say which steps a check covers.
- Both stacks read the same weight copies, quantisers and loader, so the tie-out cannot see an error there. Pin each
  against the format definition in its own test (loader output against shard bytes read by header offsets, not through the
  reader under test). Say what does verify a checkpoint read (the text, dumped ids); the verdict line implies no more.
- A check with nothing to judge cannot fail: count what was judged; a zero count prints `REPORTED` and the run `RAN`.
- A kernel breaks exact rounding ties its own way (its reciprocal); the twin of such a kernel keeps a bracket. An fp32 sum
  over thousands of products has no bracket: its contract is an interval (fp64 chain with a propagated error bound, terms
  taken from the entry's own suite and held to it by a host test).
- Fix a rule for codes down to its operation (divide by a scale vs multiply by its reciprocal), name it in the docstring,
  and test it with an element that tells the two apart.

**Writing tests**
- A setting no test fails without is one a refactor drops. Inject one-line defects into every layer a call passes through
  (functions, block classes, tables, adapters, session methods); if tests stay green, write the test.
- Ask of every rule which wrong output it alone catches and write that output: test both ends of an interval, each field
  of a multi-field comparison separately, buffers of zeros.
- A stand-in that fails forever cannot show a second attempt: fail once. Hand a stream loop a stream that owes a remainder
  at close.
- The last line of a run is code: run `main()` to its end in a host test with stand-in providers and a stand-in profiler.
- Test a third-party adapter as it is, over a stand-in of the library (replace the one method the adapter reaches it
  through); the stand-in keeps the library's defaults and records what it is handed.
- A number pinned in a test moves with the table: derive it from the plan where possible, else list it among what a new
  row changes.
- A README call or command block is run as written by a host test; a section names only flags a script of the folder
  has; a README block showing a run's output is rendered from the printing code; say under a shown call what it reads
  beyond what it prints.

**Kernel check and census**
- A generated symbol is the kernel name plus truncated compile-time arguments; two forms can share symbol and launch
  shape. Compare the form on the objects; use name and launch shape only for whose kernel and which shape.
- Reference chains change kernel names from step to step; only entry kernels are judged one by one, chains by family. No
  test assumes one name sequence per step for a plan with chains.
- Read the profiler's exported trace event by event (it grows with launches). A copy inside a launch graph captured
  before the first profiler pass may appear as a kernel record: count a copy by either spelling.
- Write the call log after the call, from the object called. Fail an entry row that names no form field.
- A new adapter adds its kernel's pattern in front of the generic cuDNN Frontend pattern; a host test holds every exactly
  judged kernel id to a pattern of its own. A pattern names every kernel its operator launches over the row counts a stack
  serves (PyTorch's index-select changes kernel above 16 rows).
- Run a row at the first and last row count it claims, and in every form its entry takes inside that range (ask the
  entry's launch facts per step shape, group by form, run one of each).
- A process that unloads a kernel module can record later kernels under a stale name. A sample never unloads an entry;
  a test that checks kernels runs the sample in a fresh interpreter. A check failing by one name that is another form of
  the same entry: look for an unload first, then for the wrong launch.
- An entry that returns a view of its own buffer is passed an output buffer per call by its adapter. An operand a kernel
  updates in place is declared state of the block (`state_args`): saved, restored, compared.
- A recorded section holds for the GPU class, SM count, row count, layer widths and step shape it was recorded with (and
  any further field a launch shape depends on, named in the section). Never carry kernels across them.
- Compare versions of what the stack imports, not of what is installed.
- A decision taken in two places is taken in neither: whether a target stack checks its kernels is the session's, and
  helpers read it there.
- Refuse a census entry without its launch shape or with a misspelt key; never print `EQUAL` for a comparison not made.
- An entry gates the installed toolchain where it builds its kernel; its first use belongs to the plan.
- A row claims the GPU classes on which the target launches the kernel, which may be fewer than the entry serves. A row states the form whole (every
  field the entry would otherwise pick); the adapter refuses a row missing one. A row names the widths it serves.
- Each row of a chain of dependent launches says in a comment what the launch before it must not write and what the
  launch after it must wait for.
- Family attribution cannot see a stray launch next to a fallback row; while such rows remain, "the adapter launches nothing but its
  entry" is held on the adapter itself (a host test finds the entry's written tensors in what the adapter returns).
- Give every count of target launches its split and say whose number it is and how much of it exists today. Headings of
  a per-class table name that class.
- What a report says follows the session's own plan (bytes per step count stored-format reads). Test an adapter's
  decline through the session that hands the option over.

**Loads and memory**
- A weight has the forms its readers read, readers being every plan of the run plus the reference: keep the stored form if
  some plan multiplies on it; make no copy if every plan does. Decided before the first load.
- Copies for the reference are made by the reference session and the checked pass's tap, never inside a block, and
  equal bit for bit the copy a load keeps; a host test loads one plan both ways and compares every check line.
- When a change removes an independent operand from a check, give the check a way to see the shared one has not moved
  (the stored-weight checksum), and test the defect in both forms of the run.
- Hold "one copy at a time" with weak references in a host test; the GPU cell holds PyTorch's peak below the memory
  sentence. Measure the memory sentence against a run whenever what a stack holds changes; on reduced models it is an
  estimate in both directions; do not fit a term to one run. Say GiB where the divisor is 2^30.
- An option for one caller is refused for every other (the copy maker on a checked session).
- A missing form fails where the plan is: the session's load names block and form before compile or step.
- State a byte formula for both stored and copy forms (stored formats are not one byte per element). The text of the cost
  line is a function, tested on the figures of a run at the published widths.
- One row order per weight, whoever asks: two plans with two orders, or a provider layout beside the checkpoint order,
  are a wrong table named before the load. A layout is applied once at load, the merged tensor dropped;
  `weights.dequantize` reads the layout a band of rows at a time into the checkpoint's order.
- The layout between two kernels is theirs, handed over as is; between a kernel and anything else, the kernel's adapter
  converts. A form that follows the row count (a token tile floor) is one row per value.

**Host behaviour and text**
- Time a tokenizer's or compiled template's first call apart from repeated calls (library caches).
- Set the profiler tracing library's log level before importing PyTorch unless set in the environment.
- Catch around a tool's own start and stop, never around the loop that runs the workload.
- Check every shard's size against its header and the directory against its index file before the plan.
- Word sentences about what bounds a step from the run's own figures, not constants. Announce a silent phase, flushed,
  outside timed intervals.
- Switch off truncation and padding of a loaded `tokenizer.json`.
- Name a file by its own name; build an `OSError` reason from the system's words for the error number, never its text.
- Compare a boolean only with a boolean (`true == 1`).
- A closed pipe fails again at interpreter exit (exit code 120): point the descriptor at the null device; a host test runs
  a child with a closed standard output.
- A README command block is typed as written: put every step in it, in order, with what it costs. "Installed by the
  README" means its lines and no others.
- Request bodies: list the refused names by feature; refuse `extra_body` whole (its fields would be dropped); a closed
  standard stream is `None`; an empty string is not "no flag"; one request writes one text; a claim about a stream is held
  only by a device test that reads that stream.

## What the verify step taught

The verify step feeds several tokens per sequence in one launch sequence (`Session(tokens=T)`, `Session.verify`, the key
`fe.gdn_verify_fused`).

- A step shape is a field of the form: the row names `TOKENS`, one row per count. The adapter holds the slot table of every
  call; the call log states the form for the step shape it was handed (`_served_stated`), read from the operands, never
  echoed from the row. `EntryBlock._step_of` answers the plan's question (which shape a row count stands for).
- Apply a one-token band token by token: the reference starts every token from the state the kernel wrote for the token
  before (`gdn_verify_step(anchor=)`), and a cell is one value head of one sequence over its tokens. Never loosen the band.
- One token per sequence is the layout of several at `T = 1`; a host test checks the shapes.
- Keeping tokens launches nothing: the kept count is a host number, the kernel reads a row of a constant table.
- Kernel operands start 16-byte aligned: a sequence run alone gets its own slot tensors; kept-count rows pad to four.
- A decline names the row for the step asked (the block's row for that tokens count, else its last such row).
- A census section holds for its tokens per sequence as well as row count, widths and class.
- Compare two runs on a block's state only where both fed the same rows; elsewhere print, and a line that gated nothing
  is `REPORTED`.
- A session of another kind of step does not inherit the first kind's report lines; until taught, a line says it is not
  stated. Two sessions on one load share weights, never a state: say so beside the snippet.

## What the out-projection of a verify step taught

In a verify step the out-projection (`projection.fp8.out`) is `fe.fp8_projection_decode` in linear-attention layers and a
library multiply in attention layers.

- Express a per-site difference through call sites, not by splitting the block; a decode plan prints the text it did
  before (hold with a line-by-line run of both trees).
- What a row hands on is data: the plan resolves the next row; the adapter is never built or called at a handed-on site.
- A split-K choice by rows is a row range with its own entry instance (`rows`). Gate batch invariance inside a range only.
- A field that follows the call (the token tile) is named in the row, or the call log cannot show it.
- A load is asked per call site (`<block> at <call site>`); test the loader with each plan alone and both together.
  Weights given a scale per row are named per site (`scale_per_row`).
- Two sessions serving a block with two kernels are not byte-equal behind it: run both sides of the row-count boundary,
  say which comparison lines are reported there, and state memory and comparison sentences for the step shapes they
  hold for.

## What one session for a prompt and its verify steps taught

A session built with `draft_tokens=` feeds its prompt through decode steps and runs verify steps behind it on one state
(`Session.prefill`, `Session.verify`, `Session.accept`, `tie_out.speculative_checks`).

- Tokens are blind where a model's token does not follow its state (the seeded output head permutes the embedding): three wrong loops -- the position moved by the kept count alone, the prompt's attention cache not the one the verify steps read, a row that attends over the whole window -- kept every greedy token and stayed inside the logits band at the published widths. Hold each by what it is: the position as an integer, the prompt byte for byte, rejected rows by a second replay with other tokens. Run wrong loops where the gate runs.
- Gate on every stack a line that cannot fail by rounding (`position`, `prompt`, `rejected rows`); `bytes` is gated on the
  reference stack, `REPORTED` on a provider stack, with no band of its own.
- The token line also holds the greedy rule: every window starts with the greedy token kept last; every kept drafted
  token is the greedy token in front of it.
- A kernel on a view is another compiled form, under the same name. The adapter builds it with the plan.
- Hand a block the tensor its band was written for (token 0 of the pool as null slot plus plain slots).
- The method called selects the step's plan; a decode step on a state behind a verify step is a `ValueError`.
- A replay starts from zero, so the record holds the prompt; the traced window holds verify steps alone and the report says so.
- One count for several sequences is exactly greedy only as the smallest of the set.
- A comparison that replays a run judges the replay. Tie it to the run: recorded ids and logits byte for byte
  (`recorded`) and the end position as integers (`position`). Put control defects in the session's methods.
- Hold as an integer what is one. The kept token is compared with the step's count, beside the position, on every stack.
- A line that names a token keeps the token's number and its sequence.
- What a first launch reserves outside the pool is reserved once in a process. It is held by a host test on stated facts.
- A check says what it looked at. Prompt decode steps are outside the traced window; say so, and name a kind of step of
  which none was judged.

## What building the drafter's pass taught

The checkpoint's drafter has its draft pass on reference blocks (`QwenDenseDrafter.body`, `propose`, `draft`; the table
`configs.DRAFTER_PROVIDERS`; the keys `torch_ref.pre_fc_norm_quantize`, `torch_ref.gate_row_quantize`,
`torch_ref.silu_mul`, `torch_ref.quantize`, `torch.argmax+lookup`).

- Three quantiser rules, each pinned to its operation and held on one row: the activations' static scale multiplies by a reciprocal that is one number per operand, a constant of the checkpoint; the loader's weights divide by a scale per row; a drafter row multiplies by the reciprocal of a scale formed from the row.
- PyTorch divides a CUDA tensor by a Python number as a multiply by its reciprocal. A rule that says "a division in fp32"
  forms its constants as tensors on the operand's device; only a device run comparing codes and scales with the CPU's
  shows it (seven rows passed by chance; with 2,048 rows the same quantiser failed on 954 scales). Run a check against
  the defect it is for.
- A block that is the reference chain is judged by nothing, and seeded proposals mean nothing that a comparison of tokens could hold. Four things hold such a pass, and each sees what the others do not: every function against a version written from its definition (an error inside a function); every block of the plan against its function, and one pass through the plan's blocks against the pass written from the functions (an error inside a block's class: operands exchanged, another function called, an epsilon of its own); a pass over several rows against that many one-row passes, byte for byte; and a recording stand-in for the blocks (an error between the blocks).
- Put a defect into every layer a call goes through: the first defects were in functions, table and pass; eight made inside the classes of the new blocks passed every test. Run a block no check taps yet through the tap in a test.
- A row of one element states its bf16 value exactly through its scale.
- An fp32 transcendental differs from the double's by an ulp: judge elements clear of a midpoint and assert their count (on bf16 inputs values repeat, so "one in two thousand" was one in a hundred and sixty.)
- A count of a pass is a count of block calls, and it is the wiring's: 18, of which the body is 15 and the head 3. State no count of launches of a draft pass: no census holds one.
- A call says which pass it belongs to: `call(block, pass, *operands)`.
- A reference row names no form; the drafter's sites are data beside the table until a kernel's row takes them.
- An arm that takes an input away has to take it away. On the checkpoint the pass's first proposal was the model's token at 167 of 207 positions. With the two halves in front of the first multiply exchanged it was 0, with the hidden row replaced by zeros 8 -- and with the hidden row of the position before, 108: a one-position shift is a cost measurement, not a collapse. Run the arm before holding a figure against it.
- Where a load is shown, say what it keeps for today's blocks (bf16 copies) and name the argument that leaves them out.

## What the speculative step taught

A session that is handed the drafter runs the whole step of speculative decoding (`Session(drafter=)`,
`Session.speculate`: the drafter's passes, one verify step, one host read; `tie_out.speculative_checks` with
`drafter cache` and `draft proposals`).

- The cut of a step follows the one host read. A step is passes, verify step, read; acceptance follows the read. The
  first step has no verify step in front and its first pass runs at one row per sequence; say which step a text means.
- What the drafter reads of the step before (final-norm rows, greedy ids, kept count, drafter cache) is state: reset,
  snapshotted and restored with the model's; a cache kept beside the state would have carried the first prompt set into the second, and a stale position reads without error.
- `verify` on a session with a drafter is refused; the caller-drafted step is `speculate(force=)`.
- A check that calls the function it checks holds nothing. The second run of the passes never calls `draft`: it writes
  the passes out at one row per call from its own books; its lines hold what a step hands its passes, not arithmetic.
- `drafter cache` and `draft proposals` each catch defects the other misses. Eight wrong passes, each in a loop that still emits the plain session's ids, hold that split in a host test.
- The window of a step is the session's. An emitted drafted token reaches no comparison line; the run's ids against the
  plain run's do.
- A traced step is the step the loop runs. It is `speculate`'s own code fed the recorded drafted tokens.
- Census launch shapes are indexed by wiring site; passes' sites in front shift the index. `Traced.first` shifts it, and the host test holds a launch in another shape at the model's site with both of its sentences.
- The comparison counts launches of one kernel id without placing them (embedding lookups share an id).
- An arm that shifts an input is not one that takes it away. On the checkpoint, on the cuDNN Frontend rows, the loop kept 3.22 tokens a step; with zeros in the place of the hidden row 1.04, with the rows of another text 1.05, with the two halves in front of the first multiply exchanged 1.00; with a later pass one position off, 3.25. `draft proposals` is the line that holds later passes' positions.
- State working memory as arithmetic, then measure: the allowance of a pass is twice the largest such chunk, 1.33 GiB at the published widths (a reference multiply's fp64 chunk). On the checkpoint, on the cuDNN Frontend rows, the run asked for about 43.3 GiB, reserved 40.0 GiB and grew the device by 40.2 GiB, the GPU otherwise empty.
- The kept counts are part of the record, so the acceptance lines of a report printed behind the comparison are still the run's. Its bytes of a step are not: the comparison leaves the session at position 0, so a run prints its report first. Every step counts in full.
- No count of the launches of a pass is stated while no census section holds a draft pass; the run says "kernel identity of the drafter's passes: not stated", and a count of block calls -- 18 a pass -- is the wiring's.
- What a step hands on by reference is held by identity, where it is handed on. A host test holds through the session's tap that `state.hidden` is the tensor the final norm's call returned, the one whose quantised rows the head read, and `state.greedy` the one the token choice returned.
- A traced step is held to the recorded step, not to its shape alone. A host test with the profiler stand-in holds the
  window and the proposals against the record.
- A refusal stands in front of everything the refused call would leave. A refusal behind a launch mark broke the
  first-token line. A refusal names the session's step: on a session with a drafter that is `speculate`, also where the session refuses a decode step or a count.
- The names beside a snippet are run as the text says they may be. In the README loop a session that was not handed the checkpoint drew seeded weights, and the drafter's with them, without a word (it now refuses the drafter of a checkpoint by name). The README loop runs as written in a test.
- Shared weights are described at the session that loaded them: the load now follows `share=` to its end (`Session._weights_owner`) and draws a seeded drafter from the seed its checkpoint records.

## What building the speculative script taught

`02_speculative_decode.py` runs the speculative step for one request: its own loop, a serving engine's speculative
configuration key by key, the comparison with decode steps and the checks of every run.

- A configuration that selects what is launched has no class "printed and ignored". Unknown keys are `ERROR`; keys that
  are read must be named (an engine default selects another launch set); a refused key's value is not printed.
- A script that compares where another checks nothing needs the verdict's order. A failed line is `FAIL` on every stack,
  in both scripts; one host test holds both `_verdict` functions to each other and to the README table.
- What the library's comparison cannot see, the script holds itself. It compares the ids of both arms, with three wrong
  loops run through `main()` on host and device.
- A default follows what it is the default of. A scheduled run needs enough new tokens to keep every count once: 17 at five drafted tokens, 38 at eight. The default follows the schedule; a typed count is still held.
- A flag that is declared in order to be refused is still a string of the script. Tests ask each script whether it takes
  the flag.
- A refusal belongs where the object is built. `Session` refuses a checkpoint's drafter without a checkpoint and a
  drafter of other widths, before a plan.
- A stand-in that rounds otherwise is not given the product's band. The test says which line fails at the default.

## What the rows a library serves taught

Three blocks of the target kernel set are library multiplies; `native` calls FlashInfer for them on the stored weights
(`providers/flashinfer_calls.py`).

- A library's kernel pick is a function of the call only until something tunes the operation. The row names the
  condition (`TUNED False`); the adapter reads the tuner after each call into the call log; the untuned pick's class and
  launch shape are census data held exactly. A row without such a condition holds its launch by membership and prints
  the pick (the 8-bit multiply). `TUNED` covers this process's tuner only, not a caller-attached tuning store.
- A launch held by membership is the same call, not the same kernel: a count of target launches says how many are held to
  a recorded shape and how many by membership.
- Class and grid follow the weight shape: a census entry names the widths it was recorded at (`at`); another width says no
  launch shape is held.
- A library that builds kernels at first use makes that use while the run is planned, after the memory sentence and
  before any load, announced in one line: scratch operands at every row count the run calls (including one-sequence rows
  of the batch check). A decline there: adapter operands an `ERROR`; installation a `SKIP` with the exception type, never
  its text. Catch around the library call alone.
- A mangled library kernel name: the family pattern names what the declared call launches (mainloop, operand type); a
  printed name comes with its length and a digest.
- Look census records up by call site.
- A layout a provider builds at load is held to the bytes it was made from, at load (its reference copy goes back through
  the same layout).
- A library tuner's worker processes re-import the main module: a script calling a sample in a tuning context needs its
  `if __name__ == "__main__":` guard.
- When a stack gains rows, every dated README run of it is of another row set: name the rows it selected.
- A figure of a greedy run on a checkpoint is a figure of a row set as well. A library multiply is inside the reference band but not the replaced multiply's bytes: with the three library rows the plain run of the default prompt was another text, and the speculative loop kept 3.27 tokens a step where it had kept 3.22 on the cuDNN Frontend rows. Such a figure is stated per row set that ran, each with its versions; it is never carried over.
- A row a stack gains is a row of every script that plans the stack. The speculative script arrived while `native` had no library row. When the rows of a stack change, every script of the family is run on that stack again, and what it says of the stack is read.

## What the attention preparation's row taught

The attention preparation is the first block on an entry that writes a cache: `cudnn.AttentionPrepareDecode` writes the
key and the value row of every row of a step into a pool of pages.

- A kernel that writes a cache decides the cache's layout, and the layout belongs to the state, not to the block: the
  adapter names its page (`cache_page`), the session builds one state in that layout, and every other block reads it
  through accessors that launch nothing (views and host integers).
- The judge forms the rows of the pool from host integers and the twin its cosines from the position it is handed: a
  judge that read the state's own tables agrees with a table one row late.
- A run's own checks hold what the kernel wrote, not what the next block reads: a read accessor is held by tests that
  index the pool by arithmetic of their own, and a run checks what the next block reads only with the row of that
  block's own kernel. Say that a defect failed a run only of a run that was made.
- The twin of an entry is the twin of the entry's own suite, statement by statement, and the band is the contract on
  the entry's page: where the page states the kernel against fp64, the twin evaluates in fp64 whatever the reference
  accumulates in.
- A stand-in entry that computes the twin's own arithmetic shows nothing about a band: run a kernel-like fp32 chain
  (some elements another byte, every one inside) and place each defect alone.
- Compare a form by value where a census file and a table meet (a list and a tuple are the same sections). A width the
  kernel bakes into its name is a width of the row.
- The first use builds the entry before it allocates anything, and its scratch pool is one page.
- A pool is whole pages: a row that changes the layout of a state moves the memory sentence, and a row a stack gains
  moves every table that follows the plan (the per-kernel table of the performance mode).

## What counting the checks into the memory sentence taught

The memory sentence counts what the checks of a run hold for every sequence beyond the first (`check_row_bytes` of the
family's model, `Session.plan_table`, the pool's share `loop.CHECK_POOL`); the list of hashed files is closed under one
rule (`fe.SOURCE_FILES`); an override by key takes the row of the run's stack where that row names the same call.

- Count what is alive at the largest moment of the checks, term by term, and name the moment. Say of every term what
  holds it: a host test finds the records, the states and the saves among the live objects at the judge; what is alive
  inside one call is held on a GPU alone, by PyTorch's peak against the count.
- An allowance taken from runs holds what those runs held: the working memory was measured at one sequence, checks
  included, so the count is added for the sequences beyond the first and no sentence of one sequence moves.
- Arithmetic on tensors says nothing about an allocator's pool. Find the pool's share by capping the pool at two
  sequence counts where the cap binds: the smallest share with which both pass, as one named constant with its runs in
  the family README, never a factor read off one run. The share is in the need, not in the part the sentence names.
- A count has a reach: its docstring and the family README say where it ends and name what is not counted. Name a
  part of the sentence whenever it is not zero bytes, also where it prints as 0.00 GiB.
- A key is by stack only between rows that name the same call: the selection compares every field but the stacks and
  the reason.
- A hash names what ran only if an edit of what ran moves it: the list is closed to its fixed point, the `__init__.py`
  of every package on the way included, and a host test forms the fixed point again from the files. A module imported
  inside a function is outside the rule.
- "Dropped when the checks return" is a sentence about references: hold it with weak references while Python's
  collection of reference cycles is off, break a cycle where it is made (the stage clock is handed the device, not the
  session), and never hide one behind a collection. A census of live objects must not keep them alive: a list of the
  process's objects that a nested scope of the same function reads as a free variable (a generator expression, and
  before Python 3.12 a comprehension) is held by a cell which the list itself holds; a canary object shows whether a
  census holds what it lists.
- When a figure moves, read every passage that states it, in every text, and reword the dated ones as sentences of
  their day.

## What the attention core's library row taught

The attention core of `native` is a library's paged decode attention, a kernel that comes as precompiled binaries
(`providers/flashinfer_calls.py`: `DecodeAttentionBlock`).

- Switch a provider's own download off in the process before the provider is imported, also where the variable is
  set and empty. Hold it with a package of the provider's name that notes what it finds at its import.
- Decide at plan time, with nothing imported, the states of an installation in which the provider's import prints a
  path: ask the distribution's metadata and the module both, and decline where they disagree.
- A refusal that holds a local path is told by its first words and never printed. The `SKIP` names what is missing and
  the fix still to be made: no install command where the package is installed.
- A provider that lacks a file may refuse nothing: this one prints a path from its own code and launches another kernel
  where the missing binary is not the first it reads. Do not rely on the refusal. Before the first call, hold every
  binary the provider's own table names for the call's data types, head width, paging and page on the GPU's class to
  its manifest, and the manifest to the checksum the provider names for it: the set read from the provider's own files,
  never a list of the sample and never a model of the provider's selection; a missing or changed file is the `SKIP`.
- A provider's identifier stands where its interface is called and nowhere else: one constant for a value the call
  passes, and prose that says what it is.
- The class a row names is where the table selects it. Say in the texts which class ran; on a device the census has no
  section for, a run judges its blocks and claims no identity.
- The interpreter's report of an exception inside a provider's callback is dropped for that one exception, during the
  first use alone; every other report goes to the hook that was there.
- A kernel whose launch follows a fact of the run is recorded with that fact: held by class and shape at the recorded
  fact, by membership elsewhere, and the run prints the kernel that was launched.
- A library that checks nothing of a table it reads has its checks in the adapter, each naming the operand. Hold a
  defect at the smallest shape that shows it, and pin the shape that hides it.
- A control arm through a run needs a run that can fail it: two table columns that name one page are no arm.
- The first use makes one scratch call per step shape of the run: its sequences and one sequence alone, at the tokens
  per sequence of each plan.
