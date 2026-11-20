# Model samples

Each folder runs one open-weight model family end to end in a single process: text in, tokens, prefill, decode steps, text
out. The samples are not a serving engine. They show which kernel serves each block of the model, check the whole path
against a reference, and make the cost of every step visible. Every sample prints a block table before it runs: block,
provider (cuDNN Frontend, cuBLAS, FlashInfer or PyTorch) and the rows that provider serves. By default a sample builds the
model at its published layer widths with two layers and seeded weights; pass `--model /path/to/checkpoint` to run a real
checkpoint from local disk. The samples never download weights or tokenizer files. The cuDNN Frontend APIs used here are
experimental. Timings printed by a sample are for orientation only.

**Where this area stands.** The first family is the dense Qwen 3.5-architecture model ([qwen_dense](qwen_dense/)), and its
first sample is the plain decode: by default on the reduced seeded model, self-checking; with `--model`, text in and text
out on a checkpoint from local disk, with the checkpoint's own tokenizer and chat template; the request can be the body a
chat client sends, and the generated text goes to standard output while every line of the run goes to standard error.
A prompt is not prefilled by a kernel of its own: it goes through the decode step, one step per prompt token, so a
prompt of N tokens takes as long as N new tokens before the first new token appears (in the runs of 2026-11-17 with a
checkpoint, on compute capability 10.0: 4.3 s for a prompt of 29 tokens on the stack `fe-cublas`, 7.8 s for the 58
tokens of the default prompt on the cuDNN Frontend rows of the stack `native`, the rows `fe-entries` selects today;
orientation values of single runs). Both run by default on the
stack `fe-cublas`, which is a reference path: it holds every quantised weight as a bf16 copy and issues every launch from
Python. The second stack, `native`, is defined by a kernel set instead: the target kernel set of the model's decode step,
checked in as a recorded kernel census, per block the kernel and the form it is called in. It is built block by block;
today five cuDNN Frontend entries are on their target forms -- the fused linear-attention step, the residual add +
RMSNorm + quantise in its five forms, the 8-bit in-projection on the stored weights (on compute capability 10.0, where
the target launches that kernel), the fused 4-bit gate / up step of the MLP on the stored weights (on compute
capability 10.x), the attention gate + quantise: on compute capability 10.0, 321 of the 484 launches of a step of the
64-layer model, 48 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results (322 = 49 +
273 with PyTorch's gather; the 1 declared stand-in has no block yet). Three more blocks are multiplies the target runs
on a library, FlashInfer, on the stored weights -- the MLP's down multiply and the output head on the 4-bit format, the
out-projections on the 8-bit one -- and `native` calls the library for them on compute capability 10.x: with them it
launches 451 of the 484 (178 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results). Of
those, 387 (114 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results) are held to the
kernel and the launch shape the census records; the other 64 are the out-projections, whose kernel the library chooses
among its own: they are held by membership, and in every run so far the library, untuned, launched its own CUTLASS
multiply there, not the cuBLASLt kernel the census records as the target's pick -- the same call, another kernel.
FlashInfer is an optional package of that stack alone; the stack `fe-entries` runs the cuDNN Frontend rows without it.
The two attention blocks are still on their fallback rows, printed as `not yet`; their kernels and a captured decode
step (a step recorded once and replayed without Python between its launches) follow in that stack. Speculative decoding
has its first pieces: the loader of the checkpoint's drafter
(its tensors, its rows of the output head, its 8-bit weights); the drafter's draft pass on reference blocks, reached
from the library (`QwenDenseDrafter.body`, `propose`, `draft`: every block of it a reference chain in every stack, its
embedding lookup PyTorch's gather); and the verify step -- the step that feeds every sequence
the token it kept last and the tokens drafted behind it at once, on `cudnn.GdnVerifyFused` with a recurrent state per
token (`Session(tokens=T)`, `Session.verify`; reached from the helper layer and held against the decode steps of the same
tokens). On compute capability 10.0, at one sequence of six tokens, five entries -- `cudnn.GdnVerifyFused` in place of
the fused decode step, the other four the same, the 8-bit projection entry there for the out-projection of the
linear-attention layers as well -- serve 369 of the 484 launches of a verify step, 48 the target's own compiled kernels +
321 cuDNN Frontend entries with byte-equal results (with PyTorch's gather: 370 = 49 + 321). A session can feed a
prompt through decode steps and verify behind it (`Session(draft_tokens=k)`), the caller supplying the drafted tokens;
handed the drafter (`drafter=`, one sequence) it runs the whole step of speculative decoding: `Session.speculate` is
the drafter's passes on reference blocks, one verify step and one host read, and the caller says after that read how
many drafted tokens it keeps. The second sample, `02_speculative_decode.py`, runs that step for one request: by
default on the seeded model with its seeded drafter and scheduled drafts, self-checking against decode steps of the
same tokens; with `--model`, a draft vocabulary and the speculative configuration of a serving engine as a JSON
object, text out and the tokens a step kept. No block of the drafter is on a kernel yet, and the sample prints no
rate. The destination is a runnable
performance reference. Until a sample reaches it, its
README says what the sample measures and what it does not, and no README here carries a performance table: numbers are
published through the formal mode of the `benchmark/` folder.

## What a model sample is, and what it is not

| it is | it is not |
|---|---|
| one program per phase (decode, speculative decode, later prefill + decode), one process | a server: no socket, no request queue, no admission of work while a step runs |
| a fixed set of sequences, decided before the first step | continuous batching or a scheduler |
| a printed table of who serves every block, decided before anything is loaded | a stack chosen by what happens to be installed |
| a self-check against a plain-PyTorch reference that accumulates in fp64 | a timing comparison with a serving engine; the `native` stack is checked against a recorded kernel census |
| the decode loop in the script itself, with its one host read per step | a `generate()` method |

Rows and users: the sequences of a run start together and stay together. A service at the same number of users also runs
prompt tokens of new requests inside decode steps, waits for cache space and for the next step, and streams over a network:
its time per token is longer and its time to first token is dominated by waiting.

## Run

The decode entries are newer than the last released wheel, so install cuDNN Frontend from this checkout. Building it
needs a CUDA toolkit with its compiler, cuDNN 9 headers and libraries, CMake 3.26 or newer and a C++17 compiler. Install
PyTorch first so the build finds the cuDNN headers of its wheel, or set `CUDNN_PATH`; a CUDA toolkit outside its default
location is named with `CUDAToolkit_ROOT`. `pip install --group` reads the dependency groups of `pyproject.toml` and needs
pip 25.1 or newer; the pip a new virtual environment starts with is often older, so upgrade it first.

```bash
python -m pip install --upgrade pip                # --group needs pip 25.1 or newer
pip install --group torch
pip install -e .                                   # a minute or two; deprecation warnings of the build backend are expected
python samples/models/qwen_dense/00_decode.py      # 2 layers, seeded weights, self-check: no checkpoint, no extra package
python samples/models/qwen_dense/02_speculative_decode.py   # the same model with a seeded drafter: speculative decoding, self-check

# A checkpoint is your own download: the samples fetch nothing. 22 GB of disk; a GPU of compute capability 10.x or 12.x
# with about 57 GiB free; about two and a half minutes for the run below (qwen_dense/README.md has the details).
pip install -r samples/models/requirements.txt     # tokenizers, jinja2, safetensors: only for a checkpoint; brings the hf command
hf download nvidia/Qwen3.8-27B-NVFP4 --local-dir /data/Qwen3.8-27B-NVFP4
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --prompt "Why is the sky blue?" --max-tokens 256
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --messages request.json > answer.txt   # the request body of a chat client

# The stack native calls FlashInfer for three multiplies: an optional package of that stack alone. The constraints pin
# PyTorch 2.14.0, CuTe DSL 4.8.0, FlashInfer 0.7.2.post1 and apache-tvm-ffi 0.1.11, the versions its kernel census was
# recorded with: in an environment with another PyTorch the line replaces it and its cuDNN, so give native an
# environment of its own (run `pip install -e .` again if pip lists nvidia-cudnn-frontend among what it installed).
pip install -c samples/models/constraints-native.txt -r samples/models/requirements-flashinfer.txt
python samples/models/qwen_dense/00_decode.py --stack native           # without the package, on compute capability 10.x: SKIP before anything is loaded
```

PyTorch warns once on stderr that NumPy is missing (`Failed to initialize NumPy`); the samples need none.

A sample writes every line of a run -- the `[00]` lines, the tables, the report, the checks -- to standard error. Standard
output carries the generated text of a checkpoint run and nothing else, piece by piece as it is generated: `> answer.txt`
keeps the text, `2> run.log` keeps the run. A seeded run has no language and writes nothing to standard output. A request
gets one text: where a run has several sequence counts, the text of a prompt is written with the first. A user
who comes from a serving engine finds the option names and the request body of one in the family README ("Coming from a
serving engine"): what is read, what is refused because it would change the output, and what a sample deliberately
does not have.

`samples/models/requirements.txt` holds sample-only dependencies with lower bounds; none of them is a dependency of the
`nvidia-cudnn-frontend` package, and the default run of every sample needs none of them. They are imported only inside the
functions that read a checkpoint: `tokenizers` and `jinja2` where its text is read and streamed (`text_io.py`),
`safetensors` where its shards are described and read (`weights.py`); never at module import. Where one is
missing, a run with `--model` prints a `SKIP` that names the file. The family README states which checkpoint the sample
was last run with, how much disk and GPU memory it needs and where the checkpoint comes from; a sample reads it from the
directory it is given and from nowhere else.

A sample checks the GPU, the library versions and the free memory before it loads anything, for every sequence count of
the run at once: a `SKIP` comes before the first load or not at all. It ends in one of (the last line of standard
error). `[NN]` is the script's tag: `[00]` for `00_decode.py`, `[02]` for `02_speculative_decode.py`.

| last line | exit code | meaning |
|---|---|---|
| `[NN] PASS` | 0 | every check of the run held |
| `[NN] SKIP <requirement>` | 0 | this GPU, library version or free-memory budget is not served; the sentence names the fix. `02_speculative_decode.py` also skips for a count of drafted tokens no row of the stack holds, and for a checkpoint without a drafter |
| `[NN] RAN reference stack (no provider checked)` | 0 | `--stack reference`: every block ran on the reference, so no provider was verified. `00_decode.py` checks nothing there; `02_speculative_decode.py` compares its run with decode steps of the same tokens there too, and a line of that comparison that fails is `FAIL` |
| `[NN] RAN (no provider block was judged)` | 0 | `--provider` flags put every judged block on its reference row: the run compared the reference with itself, and its blocks line is `REPORTED` |
| `[NN] RAN <stack> stack (kernels not observed: <reason>)` | 0 | the kernels were to be checked (`--check-kernels`; always in the stacks `native` and `fe-entries`) and the profiler's launch records could not be had (no profiler library in the environment, another profiler attached, a trace that could not be exported or read, records without launch arguments): the other checks held, the kernel check observed nothing. Never printed as `PASS`; a check that failed is `FAIL` whatever was observed |
| `[NN] ERROR <reason>` | 1 | a contract was violated: an unknown provider key; a `--model` path that is not a directory (a hub name is not fetched), or a directory with a file that is missing or is not what its name says, or a copy of a checkpoint that is not whole (a shard missing, or shorter than its header describes); a `--messages` file, or standard input, that cannot be read or is not JSON (or that was closed before the process started); a field of a request body that would change the output (the line names the field, its value and what is accepted); a key of a speculative configuration that is not accepted or that the sample does not hold, a checkpoint's drafter without a draft vocabulary, or more than one sequence for the speculative script; a request whose prompt and new tokens do not fit `--max-model-len`; messages the checkpoint's chat template refuses or turns into no token; a `--dump-tokens`, `--dump-census` or `--text-out` file that cannot be written, or one of these flags with an empty name; a standard output that was closed before the process started while the text has no file to go to; generated text that could not be written (the reader of standard output went away: the line keeps the run's verdict); a provider table that is wrong (an entry declines the row count or the geometry of a row that claims this GPU; a kernel census that does not follow the model's wiring); or an installed distribution that disagrees with the imported `cudnn`. A file is named by its own name, never by its path |
| `[NN] FAIL` | 1 | a check did not hold, on any stack; the check lines above it say which |

The first lines of a run state the environment: the `cudnn` version, whether it is an editable checkout or a wheel, a hash of
the package files the sample's cuDNN Frontend provider runs, and the installed distribution's version. If the last two
disagree the sample stops with `ERROR`: an editable install with stale metadata lets a later `pip install` replace the
checkout with a released wheel. `pip install -e .` again fixes it. The GPU class comes before that check: on a GPU the
sample does not serve, the default run prints its `SKIP` whatever is installed. With `--model` the directory is read
before the plan, because the plan needs the model: a missing sample-only package (`SKIP`) or a directory that is not a
checkpoint (`ERROR`) is reported before the GPU class is.

## The helper layer

`model_samples/` is one importable package shared by every family. It is sample code: not installed, not part of the wheel,
and for a default run it imports nothing beyond cuDNN Frontend, PyTorch and the standard library.
FlashInfer is imported inside the adapters of the rows that call it, on a device, in the stack `native` alone; no other
stack and no plan on stated device facts imports it.

| module | what it holds |
|---|---|
| `text_io.py` | text in, token ids; token ids, streamed text out: a synthetic byte-level tokenizer for the seeded run; a checkpoint's own tokenizer and chat template (messages, tool schemas, template options) from local disk; a chat request read for what a template is given; the sampling settings a checkpoint's `generation_config.json` names, which nothing here applies |
| `weights.py` | the three storage formats (bf16; 8-bit E4M3 with a scale; packed 4-bit with block scales and a global scale, the NVFP4 of a checkpoint's name), the merge rule, dequantisation; the two containers that carry what a kernel on the stored formats writes to a block that is not one yet (block scales in the kernel's 128 x 4 layout; bf16 rows the consumer quantises itself), and the one of a stored 4-bit weight in the row order a kernel multiplies on (gate and up rows interleaved), with the copy the reference reads made from it; a weight in the forms the blocks of a run read it in (its bf16 copy, its stored form, or both), and the copies the reference of the checks makes for one call of a weight that has none; the 8-bit quantisation of a weight with one scale per row, made when a weight stored in bf16 is loaded, and the 8-bit quantisation of the rows of a step, each at a scale formed from the row (a drafter's operands); seeded checkpoints built in memory; a checkpoint on local disk, described from its shard headers and read tensor by tensor without pickle; a second tensor list (a drafter's) bound beside the model's |
| `state.py` | the key/value cache of the attention layers in one of two layouts -- two tensors with one run of positions per sequence, or one pool of pages (a fixed number of positions each, 64 in the target kernel set; a run of pages per sequence) with the tables a kernel that writes such a pool reads (the cos / sin row of every position, the positions and the pool slots of a step's rows) -- and the three accessors through which a reference chain reaches either; a session builds the pool where a block of its plans asks for one, and no provider row does yet, so every run builds the first layout; the recurrent-state and conv-history pool of the linear-attention layers, by slot id; for a step of several tokens per sequence a recurrent state per (sequence, token), the slot table such a step hands its block, and how many of its tokens a run keeps; token 0 of such a state in the layout of one token per sequence, which the decode steps of a prompt work on |
| `blocks.py` | provider rows as data (GPU classes, ranges, stack, the layer widths an entry serves where it serves some only, the form an entry is called in -- per call site, where a step calls it in several -- and the call sites a row hands on to the next row), the plan for (GPU class, sequences, tokens per sequence, stack), with one part per call site for a block whose calls two rows serve, the served-by table, the memory sentence, typed declines |
| `census.py` | the kernel census of a decode step: what a block declares to launch (its census entries), the reader of the profiler's exported trace, canonical ids of the kernel families, the kernel table, and the kernel check that holds every launch of the traced steps against the plan (`--check-kernels`); the reader of a family's census of a target kernel set (`kernel_census.json`), the identity of every block call against it, the three numbers and the census line |
| `providers/fe.py` | adapters to cuDNN Frontend entries: the fused linear-attention step, the attention gate + quantise, the residual add + RMSNorm + quantise (one entry instance per form, each call served by the instance of its call site), the 8-bit projection on the stored weight (one entry instance per weight shape; for the out-projection of a verify step one per split, at the call site its row serves), the fused 4-bit gate / up step on the stored weight in the kernel's own row order (the layout a load keeps the weight in is the adapter's) |
| `providers/library.py` | adapters to library kernels (cuBLAS through PyTorch today) |
| `providers/flashinfer_calls.py` | adapters to library kernels of FlashInfer on the stored weights: the 4-bit multiply (`flashinfer.gemm.mm_fp4`, outside any tuning context: the library's own pick of kernel class and grid is census data and is held exactly) for the MLP's down multiply and, behind the reference's quantiser, for the output head; the 8-bit multiply (`flashinfer.gemm.bmm_fp8`: the library chooses the kernel, which is printed and never gated) for the out-projections; the package check (a validated version, read from the distribution's metadata) and the first use of the library while a run is planned |
| `providers/torch_ref.py` | every block in plain PyTorch with fp64 accumulation: the reference of the tie-out; and the twin of a provider whose kernel rounds at other points than the reference chain (the gate and the norm at their kernels' rounding points; the gate / up step in fp64 with the error bound of its entry's contract); the blocks of a drafter's pass, which no kernel serves yet: the two norms in front of its first multiply with one quantisation per row over both, a multiply that applies a scale per row of the operand and per row of the weight, the gate rounded once, SiLU x up, the quantisation of a row at a scale of its own, the token choice over listed rows with the id of the chosen row |
| `loop.py` | `Session`: plan, load, prefill, step for a fixed set of sequences; at several tokens per sequence (`tokens=`) the verify step (`verify`); with `draft_tokens=` one session with two plans on one state -- its prompt through decode steps, verify steps behind it, the kept count told after the host read (`accept`); with `drafter=` (one sequence) the whole step of speculative decoding (`speculate`: the drafter's passes, one verify step, the pair of the window and its greedy ids for the one host read) and one pass of the drafter per prompt position; which forms of a weight the plans of a run read -- per call site, where two rows serve a block -- and in which layout a provider multiplies on it, asked of the family's loader before the first load; stop rules |
| `sampling.py` | greedy token choice on the GPU; how many drafted tokens a verify step keeps by the greedy rule, from its host read (`accepted`: the smallest count of the set of sequences) |
| `timing.py` | host timestamps, stage lines, the traced pass (GPU busy, idle, launches per step from kernel activity records; on request the launch records themselves), the report's text |
| `tie_out.py` | bands, judged per output; the replay check, the per-block check of every provider block against the reference block or its twin, the token check, batch invariance; the reference's own copies of the weights a run keeps on their stored format alone, the check that their stored bytes are those of the load, and what the copies cost; the line of the kernel check; a session that verifies behind its prompt against decode steps of the same tokens (`speculative_checks`: tokens, kept state, position, prompt, recorded, bytes, rejected rows, acceptance; for a session with a drafter also `drafter cache` and `draft proposals`, against the drafter's passes written out a second time at one row per call) and the drafted tokens of a scheduled step (`forced_drafts`) |
| `<family>/` | what is specific to a model: geometry, layer order, block wiring, the provider table and the regime table as data; the census of its target kernel set (`kernel_census.json`); the drafter of a checkpoint (`drafter.py`): its loader -- its tensor list, the rows of the output head at a draft vocabulary's ids, its 8-bit weights, the bf16 copies its reference blocks multiply on -- and its draft pass with a provider table of its own (`DRAFTER_PROVIDERS`: 18 block calls a pass, every block a reference chain but the embedding lookup; `body`, `propose`, and `draft`, the passes of one step of speculative decoding, on a cache of the drafter's own) |

What a user of a serving engine brings -- the request body of a chat client, the option names of a serving command line --
is not read in the helper layer. It is parsed beside the scripts of a family (`qwen_dense/serving_options.py`), which hand
the helper layer messages, tool schemas and template options.

## What a run reports

Every line below is printed, on standard error, by every run of a provider stack (`fe-cublas`, `native`, `fe-entries`);
`--stack reference` prints no `checks` and no `first launch` line. A run on the reduced model prefixes its times with "reduced model" and prints no rate. None of it is
a measurement of record: no clock is locked, and the stack is launched from Python.

Words a run and these pages use. *Sequence*: one text being generated. *Row*: one token position of a step; a plain decode
has one row per sequence. *Step*: one pass of the model over all rows. *Period*: from the first launch of one step to the
first launch of the next. *Captured*: recorded once and replayed without Python between the launches; these stacks are
launched from Python instead, one launch after the other. *Clock not locked*: the GPU's clock frequency is left as it
is, one reason no timing here is a measurement of record. *Teacher-forced*: a provider block and its reference block are
run on the same operands and the same state, call by call, so that an error cannot accumulate from block to block.

Words of the performance mode (`--perf`; "The performance mode" below). *Timed request*: one prompt from a zero state,
its first new token, then a fixed number of forced decode steps -- what a serving benchmark calls a request. *Judged
position*: a position of a request at which every provider block is held against its reference block. *Token-to-token
interval*: the time between the host reads of two consecutive tokens of a timed request. *Launch overhead*: the part
of a step in which the GPU waited for the next launch, or the host for the read. It is an estimate: a timed step
less GPU busy of the traced step at the same position and less the
loop's own host time. *Record class*: what a run may call its numbers, on the second line of its report:
`orientation` with every reason, or a controlled run.

| line | definition |
|---|---|
| header | the model (the reductions of the default run, or the checkpoint's directory name, shard count, size and manifest hash), the GPU, the sequence counts, the stack; the `cudnn` version, editable checkout or wheel, a source hash, the installed distribution; the reference's accumulation precision |
| `served by` | per block: calls per step, provider, entry, the rows the entry serves, and the entry's published record (the GPU class and rows of a published measurement of that entry, `none`, or the reason a row is a fallback: not a measurement of this run or this GPU); the declared regime; then the memory the run needs against what is free, in GiB (2^30 bytes) -- and, where the run keeps a weight on its stored format alone, the part of that need which is the checks' (the reference's bf16 copy of one such weight at a time). Decided and printed before anything is loaded |
| `stage` | seconds of import, describe (a checkpoint's tokenizer, chat template and shard headers; no weights), plan, read (or draw), merge, dequantise, lay out (a stored weight brought into the row order its kernel multiplies on) and checksum (of every weight the run keeps on its stored format alone) where the stack has such weights, compile and first launch of each compiled form, warm-up |
| `greedy decoding`, a request body's line, `first token:` (with a checkpoint, before anything is loaded) | the decoding rule beside the sampling settings the checkpoint's own `generation_config.json` names, which are not applied; for a request body, every field of it and what the sample does with it; and what the prompt costs before the first new token: one decode step per prompt token, as a count. The family README has the three ("Coming from a serving engine") |
| `decoding`, `checks on` | two lines that say what comes next, printed before the phases in which a run prints nothing on standard error: the prompt and the decode loop, and the checks. The first says where the text goes: with a checkpoint to standard output, piece by piece as it is generated (or once after the report with `--no-stream`, or to the file of `--text-out`); a seeded run writes its text nowhere |
| `text path` | render = chat messages to prompt text; tokenize = prompt text to ids, also per prompt token; ids to GPU = the copy of the prompt's ids to the device; text out = ids to text per produced token, inside the loop, with the write of the piece where the text is written as it is generated. With a checkpoint, the first call of the process and the same call repeated |
| `first token` | the prefill route and its time. While the prompt goes through the decode step, one step per token, this is not a time to first token of record |
| `steps` | period = from one step's first launch to the next step's, median and mean; in the step = from the first launch to the ids on the host; turnaround = from the ids on the host to the next launch (detokenise, the write of the text piece where the text is written as it is generated, stop check); block calls per step |
| `traced pass` | the first 32 steps again, from a zero state, under the PyTorch profiler with kernel activity records only. The period under the profiler comes first, with its stretch against the untraced one. GPU busy = the union of the intervals of the step's kernels and copies; idle inside the traced step = the step's span on the GPU less GPU busy (the two are parts of the traced step); kernel launches and copies per step; host-exposed = the untraced period - GPU busy. Where the records cannot be had the line says why |
| `rate` | tokens produced / wall time of the decode loop, per sequence |
| `bytes per step` | arithmetic on the model's tensor list: the bytes of the weights a step reads in their stored formats plus the state it reads and writes, against the same with the weights as this stack reads them (bf16 copies, and the stored format where a block of the stack multiplies on it); in GB (10^9 bytes), as a checkpoint's size in the header |
| `note` | what bounds a step of this run, from the traced pass: the launch path from Python where host-exposed time is the larger part of the period, the stack's GPU work where GPU busy is; "not measured" without a traced pass. An ordinary run prints no time per block; the performance mode prints the device time of every block's kernels. Neither prints a limit (floor, speed-of-light figure) |
| `text out` | how many tokens sequence 0 produced, why it ended and where its text went (standard output, or the file of `--text-out`; as it was generated, or after this report); the text itself is not repeated on standard error. For a seeded run, the first characters of what its ids decode to |
| `ids hash` | in the last line before the verdict: a SHA-256 over the produced ids of every sequence of the run, its first 12 hex digits; equal between two runs that produced the same tokens. The other two hashes of a run are in its header: the manifest hash of a checkpoint (its layout, not its values) and the source hash of the package files the cuDNN Frontend provider runs |
| `checks` | replay, blocks, tokens, batch invariance (the family README has the bands; a run prints each band's text once, under the block lines); with a checkpoint, on the first 16 steps of every prompt, and the token line is `REPORTED`, not judged against a band. Where the run keeps a weight without its bf16 copy, two lines under them. `stored weights`, a verdict: every such weight still holds the bytes of its load, before the checks and after them -- the reference's copy of it is made from the bytes its own block multiplies on, so a block that wrote into its weight fails the run here, by block and layer. `on demand`: how many copies the reference made for itself, the largest, their bytes in all and the seconds (no verdict: it is what the checks cost). With `--check-kernels`, and always in the stacks `native` and `fe-entries`, a fifth line, `kernels`: see below |
| `kernels, planned`, `kernels, as the entries state them` | with `--check-kernels`, and always in the stacks `native` and `fe-entries`. Before the load, from the plan alone: per block the kernels one call launches and how they are judged -- `name and launch shape, as the entry states them` (an entry of cuDNN Frontend: its own symbol, grid, block and shared memory), `name` (a kernel known by its name alone), `by family` (a fallback row: a reference chain or a multiply over a copy launches kernels of PyTorch and cuBLAS, how many is not a fact of the row), `host path, not compared`. After the load: what each entry states for this run's row count -- its launch shape and the symbol of the kernel it compiled. In a stack that declares a target kernel set the planned table has a last column, `identity` -- `target`, `stand-in`, `not yet (<fallback reason>)` or `not the target (<what differs>)`, decided against the family's kernel census -- and two lines under it: what the target is for this model, and how many launches of a step are target kernels, each as three numbers (the target's own compiled kernels, cuDNN Frontend entries with byte-equal results, declared stand-ins). Where the census has no section for the run's GPU class, SM count or row count, no identity is stated and the line says so. A row whose entry a step calls in several forms prints the form of every call site under the planned table, and after the load the launch each of those forms states |
| `kernels` (a line of `checks`) | only with `--check-kernels`: every launch of the traced steps against the plan. The form a provider row names is compared on the objects: with the entry's own form, and with what the entry instance that served each call of the traced steps states (the call log: one per traced step, and a step without its log fails); a row of an entry that states its launch and names no form fails. Every exact kernel must be there once per call, in the wiring order, under the entry's own symbol and with exactly the entry's grid, block and shared memory; every other launch must belong to a family that a fallback row or the host path at that place declares; copies are counted, and one between two exact rows is not declared. For a block whose entry is called in several forms all of this holds per call site: a call is held to the form of its own site and to the entry instance of that form, and a line says which instance served how many calls of a step. `PASS`, `FAIL` with one line per kind of difference, or `NOT OBSERVED` with the reason. The kernels a library chose inside the fallback rows are printed and not gated. In a stack that declares a target kernel set, every launch on a target kernel must also have the launch shape the census recorded for it on this GPU class, SM count and row count, and the lines under the check end with `target census <configuration> / <class> (recorded <date>): EQUAL` or `NOT CLAIMED (<every reason that applies>)` |
| `GPU memory`, `first launch` | PyTorch's peak and the growth on the device since the plan, in GiB; what the first launch of a compiled form took outside PyTorch's pool, in MiB |

## The performance mode

`--perf` turns a run of a decode sample into the decode performance report of this GPU: one process, one sequence
count, one prompt. An ordinary run measures once, while its checks record its first steps and its text is written
between its steps; the mode separates what is checked from what is timed, and what is timed from what is traced.

```bash
# The report goes to standard error, a report file report-<UTC time>.json to the current directory, the text to standard output.
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --stack native --perf

# The protocol written out (these are the defaults) and a file name of your own:
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --stack native --perf \
    --perf-warmup 32 --perf-steps 256 --perf-requests 3 --perf-report my_gpu.json
```

**The protocol.** A *timed request* is the prompt, its first new token and `--perf-steps` forced decode steps behind
it: with the default of 256 steps, 257 new tokens and 256 intervals, one token more than a serving request of 256
output tokens. A run makes, in this order: one checked request, launched as every other, with the recorder of the
checks on, and the reference checks on it -- replay over the whole request, every provider block against the
reference at the judged positions, the tokens on the first 16 steps; a warm-up request; `--perf-requests` timed
requests, each from the state zeroed in place (the device is synchronised behind the zeroing, which is outside every
statistic); one traced request; and the checks behind it. Inside a timed request there is nothing but the step: no
recorder, no call log, no profiler, no text. The first profiler pass of the process starts behind the last timed
request: the traced request feeds its prompt untraced and then traces the first timed positions, its first step
alone -- whose launch records size the rest -- then as many further steps as `--perf-trace-records` holds. The judged
positions are the first 16 of the request, the two on either side of the first page boundary of the attention cache
inside the timed steps, and the last two timed positions: for the default prompt of 58 tokens, 0-15, 63, 64, 312 and
313. A page is 64 positions, the page of the paged attention cache that the attention core of the target kernel set
reads; the attention blocks of this sample read one contiguous cache today, and the two positions are judged all the
same, so that the judged positions do not move when the paged cache arrives. Before the load the run prints how many
decode steps of the stack all of this is at most, the replay of the checks among them, as a count; a check of the
checked request that fails ends the run before anything is timed, and no file is written.

**The statistics**, each with one definition, which the report file carries beside the number:

| line | definition |
|---|---|
| `decode rate` | 1000 / time per output token, in tokens per second per sequence; time per output token = (the last token's ids on the host - the first new token's ids on the host) / (new tokens - 1) of a timed request, and of a run the median over its timed requests. Never a median step turned into a rate, and not output tokens / wall time, which holds every time to first token (that figure is in the file). The line also gives how far the timed requests are apart. Not printed for a reduced model, nor where a statement of the `checks` lines does not hold |
| `token-to-token interval` | the time between the host reads of two consecutive tokens, pooled over the timed requests: median, p90 and p99 by nearest rank. Not a service's tail latency: the intervals span a growing attention cache as well as jitter |
| `time to first token` | from the request being handed to the text path to the first new token's ids on the host: render, tokenize, the copy of the ids, the prompt's decode steps (one per prompt token: no stack of this sample has a prefill kernel) and the host read; the median over the timed requests. The load is outside it. A serving engine prefills the prompt in one pass and adds its queue and its network path: the same pair of events, not the same quantity |
| `step` | the median token-to-token interval over the timed steps at the positions the traced request covers (the line names them). It is taken apart with the launch records of the traced request: GPU busy = the union of the intervals of every kernel and copy of a traced step, from kernel activity records, the median over the traced steps (an upper bound: tracing lengthens kernels); host = the loop's own time from a token's ids on the host to the next step's first launch call; launch overhead = step - GPU busy - host, formed for every timed step with GPU busy of the traced step at the same position, before anything is aggregated, and of a run the mean over those steps. The line gives the step's mean beside its median and takes the mean apart, `mean = GPU busy + launch overhead + host`: every term is the mean over the same steps, so the terms add up, which the medians of three lists do not. Launch overhead is an estimate, and the line under the equation says so: a timed step has no GPU-side record, and GPU busy is the traced request's and an upper bound. It is printed as a term only where it is larger than the mean GPU busy; otherwise the line gives the mean GPU busy as the upper bound it is and says that launch overhead and host are not resolved by this method (the remainder as computed is in the file). A line under it says how many steps were traced, at which positions, their median GPU busy, and by how much tracing lengthened a step. Where the traced steps could not be observed the line says `GPU busy not measured`, with the reason, and no table is printed |
| `note` | while launch overhead is a term of the step: its share of the mean step and the time per block call, named as the estimate it is, and what it stands for -- time in this sample's call path from Python, not kernel time |
| `per kernel, per step` | the device time of every block's kernels, from the launch records of the traced request. A row is a block, found by its place in the wiring, as the kernel check finds it, never by a kernel's name; the blocks served by fallback or host entries that no exact kernel separates share one row, and the row says so (`one row`); the host read that ends a step is the last row. A row's name says what it is: a row of blocks on fallback entries is marked `not yet:` where the kernel census says so of one of its blocks and `fallback:` where no kernel identity is stated, a row of host-path blocks alone `host path:`, and an exact row on a declared stand-in `(declared stand-in)`; the row of a block a library serves carries the entry in front, as the plan names it (`flashinfer.gemm.mm_fp4: mlp.down`), and holds the launches of the library's kernel, found by their place in the wiring like every exact kernel; what a block's call launches beside a kernel of its own -- the reference quantiser in front of the output head's multiply, a further launch of a library's choice -- is the row of its place in the wiring, marked `beside its kernel:` and shared with whatever else the wiring has there (the records do not tell them apart), and it prints 0 where nothing was launched; a list of blocks wider than the column is shortened to its first block and `+ N more`. The report file holds every block of every row, with its entry and the identity the census states for it. `launches`: the kernel and copy records of a step the kernel check attributes to the row. `us/launch`: the time a launch adds to the GPU's timeline: its end less the later of its start and the latest end of any earlier launch of the step, the median over the row's launches; not printed for a row of fallback or host blocks nor for one of launches beside a kernel, chains of many kernels. `ms/step` and its share of GPU busy: the row's added times of a step, the median over the traced steps, and that median over the median GPU busy, which the line under the `step` line gives. The added times of a step partition the union, so the rows sum to GPU busy (the medians, and with them the shares, up to the difference between a median of sums and a sum of medians; the means in the file to the rounding of a double). The table says of itself what it is: its second line says that it is launched from Python: a kernel's time there is not its time inside a captured sequence. A table is printed only where the kernel check holds: a record that no block of the plan accounts for prints no table, and where the profiler hands over whole microseconds the `us/launch` column is not printed and the table says so |
| `memory` | weights, state and cache: the family's arithmetic over the tensors held on the device; peak reserved: PyTorch's peak between the emptied pool behind the checks and the end of the timed requests; in GiB (2^30 bytes), where an ordinary run's `bytes per step` line is in GB |

**The labels.** Line 2, `record class`, says what the numbers of a run are. A run is a controlled run only when it
observed every condition itself: a checkpoint, every block on a claimed target kernel, a captured launch sequence,
the three statements of the `checks` lines, no other process on the GPU, the clocks. Otherwise it is `orientation`
and the line lists every reason, in that order, and the rate line reads `decode rate (orientation)`. No version of
the mode observes the GPU's other processes or its clock yet (`not observed`, never assumed), and no stack of this
area is captured: every run today is `orientation`. The `scope` line behind it is always there: a decode-loop rate
is not a served rate. The `kernels` lines give the launches of a step that are on the target kernel set, by provider and as
three numbers, and what else a step launches; on a stack that declares no target they say so. The `checks` lines are
three statements, each with its own verdict: the blocks against the reference at the judged positions; the timed
requests and the traced request gave the checked request's token ids, step for step, every timed request left the
state in the same bytes and the stored weights still hold the bytes of their load; the kernels of the traced steps
are the plan's. A statement that does not hold prints its `FAIL`, the rate line says why no rate is printed, and the
run ends in `[00] FAIL`; kernels that could not be observed are `NOT OBSERVED` with the reason, the run keeps its
rate as `orientation` with the reason `kernels not observed`, and it ends in
`[00] RAN <stack> stack (kernels not observed: <reason>)`. The line `with draft tokens` says `not run by this script`:
this script decodes without a drafter. A report holds no speed-up, no baseline and no figure of another GPU class.

**The report file** (`--perf-report FILE`; default `report-<UTC time>.json` in the current directory, so one command
can be run twice; a file that exists, or that another output flag of the run names, is an `ERROR` before the load:
a report file is never written over) is strict JSON with units in its field names and no local path: `record` (the class and its reasons), `model`, `environment`,
`observation`, `kernels`, `protocol` (the counts, the prompt's hash, where the profiler first started), `checks` (every
judged position with its reason, the lines of the checks, the three verdicts, the run's last line),
`statistics` (per statistic its definition, what it corresponds to in a serving benchmark, its value and, where there
are samples, their count, median, mean, extremes and percentiles), `raw` (every stamp of every request, relative to
its start, and the hashes of its ids and of its end state), `traced` (the steps the traced request covered and their
positions; GPU busy, idle time, kernels and copies of every traced step; the resolution of the profiler's time
stamps as found), `per_kernel` (every row of the table: its blocks, its launches against the planned ones, the added
and the recorded time of a launch, its time per step as median and as mean, its share, the launch shapes of an exact
row and of a library's row, there with the name of the kernel the library launched), `memory`, `startup` (the seconds
of every stage) and `text_path`. The file's first field names its schema,
`model-sample-performance/3`. The screen is rendered from the data of a file of this schema (`screen` in
`model_samples/perf.py`); handed a file of another schema, written by another version of the mode, it renders nothing
and says so in one sentence that names both schemas.

The report, with dots where a run prints a value (a checkpoint on the stack `native`; the checks' own lines come
before it, the `text out` line and the last lines of a run after it):

```text
cuDNN Frontend model sample: decode performance report
record class          orientation, not a number of record: fallback rows present;
                      kernel identity not claimed on this GPU class; launched from Python;
                      other processes not observed; clocks not observed
scope       a decode-loop rate, not a served rate: one process, no request queue, no scheduler, no network
model       <checkpoint>, ... layers | ... sequence | greedy
GPU         <name>, compute capability ..., ... SMs
            other compute processes: not observed (this version)
            clocks observed: no (this version does not observe the GPU)
kernels     ... of ... launches per step are on the target kernel set: cuDNN Frontend ..., FlashInfer ...,
            PyTorch ...
            = ... the target's own compiled kernels + ... cuDNN Frontend entries with byte-equal results
              + ... declared stand-ins
            not yet: ... block calls on fallback rows + ... launch of the target that has no block yet
            a step also launches ... kernels and ... copies
protocol    warm-up ... steps | timed ... steps x ..., end token ignored | launched from Python, not captured
checks      blocks against the reference at positions ...: PASS
            timed requests give the checked request's tokens and state: PASS | kernels, traced steps: PASS

decode rate (orientation)   ...  tokens/s per sequence   (... ms per token; ... requests within ... %)
                      token-to-token interval: median ..., p90 ..., p99 ... ms
time to first token   ...  ms for a ...-token prompt (... decode steps: this sample has no prefill kernel)
step                  ...  ms; mean ... = GPU busy ... + launch overhead ... + host ...
                      launch overhead: an estimate (GPU busy is the traced request's and an upper bound)
                      (GPU busy from ... traced steps, positions ... to ...; median ... ms;
                      tracing lengthens a step by ... %)
with draft tokens     not run by this script (plain decode)
memory                weights ... GiB | state and cache ... GiB | peak reserved ... GiB  (GiB = 2^30 bytes)
note                  launch overhead (an estimate) ... % of the mean step = ... us per block call: time in
                      this sample's call path from Python, not kernel time

per kernel, per step (traced request)                        launches  us/launch  ms/step  share of GPU busy
  launched from Python: not the time of a kernel inside a captured sequence
  us/launch = the time a launch adds to the GPU's timeline
  embedding                                                       ...        ...      ...    ... %
  add_norm_quantize                                               ...        ...      ...    ... %
  projection.fp8.in                                               ...        ...      ...    ... %
  linear_attention.step                                           ...        ...      ...    ... %
  flashinfer.gemm.bmm_fp8: projection.fp8.out                     ...        ...      ...    ... %
  mlp.gate_up                                                     ...        ...      ...    ... %
  flashinfer.gemm.mm_fp4: mlp.down                                ...        ...      ...    ... %
  attention.gate_quantize                                         ...        ...      ...    ... %
  torch_ref.quantize + flashinfer.gemm.mm_fp4: output_head        ...        ...      ...    ... %
  beside its kernel: projection.fp8.out                           ...         --      ...    ... %
  not yet: attention.prepare + attention.core (one row)           ...         --      ...    ... %
  beside its kernel: output_head                                  ...         --      ...    ... %
  beside its kernel: output_head + token_choice (one row)         ...         --      ...    ... %
  host read                                                       ...        ...      ...    ... %

report file           report-<UTC time>.json (every number, its definition, the protocol, the environment)
```

Without `--model` the same protocol runs on the reduced seeded model: every time then carries `reduced model:`, no
rate is printed, and the text is written nowhere. The GPU test of this area runs that form.

## What is verified where

| what | by | where |
|---|---|---|
| the helper layer without a GPU: tables, plan, declines, memory sentence, environment check, token choice, stop rules, a decode of a tiny geometry on the CPU, also through the sample's own loop function; the sample's flags and its last line; the tie-out itself on stated tensors and through stand-in providers with deliberate defects (a NaN, one wrong value head, a row that reads its neighbour); the drafter as a model of a plan: its table in every stack on stated facts, every row at any row count, its call sites, the five provider keys of its pass in the closed list, and its two tables of the family README against the data | `test/python/misc/test_model_samples_plan.py` | any host, level L0 |
| the weight formats: the bf16 copies both stacks read against a decode written from the format definitions, the quantisers, both merge rules, the reference multiply, also where the operand brings a scale per row beside the weight's (against a loop over the code values), and the slice of such an operand by rows; the two containers a kernel's bytes travel in to a consumer on a fallback row, with the index of the 128 x 4 block-scale layout against the layout rule; a checkpoint directory generated in the test: described from its headers, bound through the family's name map, read back bit for bit, decoded on the CPU to the bytes the same tensors give from memory; the declines of a checkpoint the family does not read; the errors of a copy that is not whole and of what else a user can get wrong, each as the one line the sample prints; the memory sentence of `--model` before a tensor is read | `test/python/misc/test_model_samples_weights.py` | any host, level L0; the checkpoint tests skip visibly without `safetensors` |
| the text side: the synthetic tokenizer and its stream; chat requests from JSON files; a chat template with the constructs of a checkpoint's own (tool schemas, a reasoning block, tool calls and results, several turns, an option, a refusal) against stored strings; a tokenizer trained in the test, encoded, decoded and streamed, also when it was saved with truncation and padding; the requirement file. And what a user of a serving engine brings: a request body answered field by field, with one case for every field that would change the output (accepted at its neutral values, refused by name elsewhere) and one for every other refusal; the limit of new tokens of a request against `--max-model-len`; the two lines a run prints before the load; the sink of the generated text (pieces flushed, a file, a reader that went away); the sample's loop writing the text between its steps; and the sample's `main()` on the CPU to its last line, on a checkpoint directory generated in the test, with a request body from a file and from standard input -- standard output the text alone, standard error every line of the run | `test/python/misc/test_model_samples_text.py` | any host, level L0; the checkpoint tests skip visibly without `jinja2`, `tokenizers` or `safetensors` |
| the kernel check without a GPU: the canonical ids of the kernel families, the reader of an exported trace, the wiring order of a step and its call log, the form check on the objects, and the comparison on constructed launch records -- an extra launch, a missing one, a swapped pair, a kernel of a fallback family where none is declared, the expected name with another grid, block or shared memory, another symbol, an undeclared copy: each fails with its own sentence; through a session, stand-in providers whose launches a stand-in profiler records, among them one that declares one kernel and launches another and one whose calls another entry instance served; a traced pass without the call log of every step; the sample's `main()` run to its last line on the CPU: `PASS`, `FAIL`, and `RAN` without a profiler; the census file | `test/python/misc/test_model_samples_census.py` | any host, level L0 |
| the target kernel set without a GPU: the census file against its own three numbers and against the wiring of the two-layer, the four-layer and the 64-layer model; the identity of every block call and the coverage of the `native` plan; every reason of `NOT CLAIMED`, and `EQUAL` on a constructed complete census; a target launch in another shape than the recorded one fails with its sentence; "claimed implies no fallback row and a recorded basis"; the gate entry's adapter over a stand-in entry, its twin and its band on stated tensors (a byte outside the two codes, a byte off the nearest code beyond the margin, a NaN byte: each fails); the norm entry, which a step calls in several forms: the call sites of the wiring and the form the row and the census name for each, its adapter over a stand-in entry (one instance per form, the residual stream in place, buffers of its own, what it hands a fallback consumer), its twin and every rule of its band on stated tensors, and through a session a call served in the form of another site, a chain under the row's name, a launch more, a launch less, a launch in another shape than the recorded one, a launch under the symbol of another form, a traced pass without its call log and four kinds of wrong byte: each fails with its own sentence -- and a launch more behind the kernel, in front of a fallback row, which the check does not see, pinned as that; a row that names the layer widths its entry serves: a model of the family with another hidden width plans on the reference chain, with the real entries, and states no kernel identity; the fused gate / up entry: its adapter over a stand-in entry (the config stated whole, the weight in the kernel's row order, the two scalars, buffers of its own, a row-major operand brought into the layout by the adapter), its twin and both rules of its band on stated tensors, its four rows by row count, with the fallback row from 65 rows, on compute capability 12.x and at another MLP width, and through a session a call served in another form, a chain under the row's name, a launch more in front of its kernel, a launch less, a launch in another shape than the recorded one or than the entry states, a wrong code and a wrong block scale: each fails with its own sentence -- and a launch more behind the kernel, which the check does not see, pinned as that; the sample's `main()` with `--stack native` to its last line | `test/python/misc/test_model_samples_census.py` | any host, level L0 |
| the rows a library serves, without the library: the real adapters over a stand-in of FlashInfer through a session -- the clean run, and each control: another kernel class or another grid of the 4-bit multiply, the same call inside a tuning context of the library and after one, a launch more, the kernels of a multiply over a copy under the row's name (each fails the kernel check with its sentence), a wrong element (fails the block in its band); for the 8-bit multiply the recorded pick, a split pick and the library's own kernel (each passes, printed and compared), a kernel of neither family (fails); the first use while the run is planned and its two declines; a missing and an unvalidated package through the sample's last line; the stored 4-bit weight in the layout the library reads; the same rows in a step of speculative decoding, behind the drafter's passes: the kernel class and the launch shape of the 4-bit multiply and the recorded pick of the 8-bit one held and printed at the model's call sites; the requirement files against the census and the adapter; the ignore list of the Dependabot entry against the packages the constraints pin, and the form of its items (a name and the three update types of a version update, nothing else); the policy sentences | `test/python/misc/test_model_samples_census.py` | any host, level L0 |
| what a run holds of a weight, without a GPU: per multiply block the stored tensor the loader keeps when asked (8-bit codes with their scale; packed 4-bit codes with their block scales and global scale), bit for bit what the bf16 copy is made from, and the bytes the family states for each form against what a load holds; which forms the plans of a run ask for (a follower on a fallback row keeps the copy); a weight kept in the row order its provider multiplies on: the layout asked of the provider, the same bytes in that order and in no other stored form, the checksum taken of them, the copy made from them bit for bit the copy of the checkpoint's order, a layout without its stored weight named; a session on weights that were loaded for another plan, in both directions, and on a stored weight in another row order than its block multiplies on: an error that names the block before its first step; the reference's copies made for one call -- every check line equal to the same run with its copies kept, the copies counted, the run's weights untouched, a wrong element still failing its band; each such copy gone when its call returns (weak references to the tensors, through the checked pass and through the reference's own passes); a block that writes into its stored weight -- every block line holds, because its reference moved with it, and the run fails on the weight by block and layer, before the checks or during them, where the same block in a run with its copies fails its band; the checksum behind that on stated tensors (one byte, two rows exchanged, two bytes exchanged along a row, a scale); the maker of such copies refused for every session but the reference of the checks, which is not traced; and, with a test's own blocks on the stored format on every multiply (two provider rows multiply on a stored weight today, the in-projection's and the gate / up step's), a run without a copy of any multiplied weight through all its checks, with its memory sentence and its bytes of a step | `test/python/misc/test_model_samples_census.py` | any host, level L0 |
| the loader of a checkpoint's drafter without a GPU: the 8-bit quantisation with one scale per row against a quantiser written from the format's definition, bytes and scales -- a row of zeros, a row at the smallest scale, quotients exactly between two codes, and two elements that a multiplication by the reciprocal of the scale would code differently; the draft vocabulary (each id once, in ascending order, inside the model's vocabulary, with what was dropped counted) and every file its reader refuses; the family README's call of the loader and its recipe for a draft vocabulary, run as they are written; a checkpoint directory with a drafter, generated in the test: the manifest from the headers alone, the rows of the head against the bytes of the shard file read by the header's offsets, the 8-bit weights and the order of the parts of the merged ones, what a load reads; every drafter the loader declines, by name; two tensor lists bound to one checkpoint. And the drafter's draft pass, whose blocks are reference chains that no tie-out judges: the quantiser of a row of a step against a quantiser written from the format's definition (a row of zeros at the smallest scale, a product that saturates, two rows of two magnitudes, and the two elements that a division by the scale would code differently); the joint quantiser of the two normed halves, their order and their one scale; the gate with one rounding and SiLU x up with two against their formulas, element by element, with the model's own chains byte for byte what they were; the token choice over equal columns; a pass over 2 and over 6 rows of a sequence against that many passes of one row, rows and cache byte for byte, at one sequence and at two; through a recording stand-in for the blocks, what every pass of a step is handed for every kept count; every block of the drafter's plan, as the plan builds it, against the function it stands for, and one pass through the plan's blocks against the same pass written out from the functions; every block of a pass over two sequences through the tap of the tie-out, each sequence alone; the family README's draft step, run as it is written; the model's stored forms and the drafter with its copies loaded together from a checkpoint generated in the test, with `memory_bytes()` against what is held | `test/python/misc/test_model_samples_drafter.py` | any host, level L0; the checkpoint tests skip visibly without `safetensors` |
| a session with a drafter without a GPU, the seeded drafter of a tiny model on the CPU: the step through `speculate` with scheduled drafted tokens (the ids of the plain session, every count kept, the drafter's own proposals recorded) and on the drafter's own proposals; every line of the comparison with the two of the drafter; replay with the proposals; eight wrong passes, each failing the line that sees it; a second prompt set and a restored snapshot; the refusals; the kernel check of such a step on constructed records (the gathers, a copy between two exact rows, two host reads, a launch in another shape than the recorded one at the model's site, a kernel class and a recorded pick at the model's site); the report's acceptance lines and bytes; the memory sentence with the drafter at the published widths | `test/python/misc/test_model_samples_verify.py` | any host, level L0 |
| the performance mode without a GPU: every statistic on constructed time stamps (the time per output token, the rate, the nearest-rank percentiles, the run's value as the median over its timed requests, the time to first token, the step and the loop's host time at the traced positions); the GPU side on constructed launch records (GPU busy as the union of a step's intervals with an overlapped stretch counted once, the step taken apart and the rule for printing launch overhead as a term, tracing's stretch, the rows of the per-kernel table by the place in the wiring on stated sites and on the family's own plans of `native` and `fe-entries`, the rows of the blocks a library serves and of what a block launches beside its kernel, blocks on fallback or host entries that share a row, the rows against GPU busy, a record without a row, whole-microsecond stamps, the launch records of `--dump-census` with their start and end); the judged positions from the request and the family's page; the record class from its conditions; through a session on the CPU with stand-in providers and a stand-in profiler: no profiler pass before the last timed request, nothing but the step inside a timed request, the reference checks at the judged positions, the three statements of the `checks` lines, a timed request that computes other tokens, traced steps that launch another kernel, kernels that cannot be observed, a check that fails before anything is timed; the screen in every form at no more than 110 characters a line; the report file (strict JSON, a definition beside every statistic, no path, never written over, its schema, and a file of another schema refused in one sentence that names both); the three numbers of the `kernels` lines on the family's own tables; every refusal of the flags before the load; `main()` with `--perf` to its last line, also where a statement of the checks does not hold; an ordinary run that prints what it printed and does not import the mode's module; every stamp of a timed request at its event, with the device synchronised between the zeroing of the state and the request; each number of the screen against the field of the file its label names; and the public text of the mode -- the sentences of this page's section, of the family's flag table, of the guide's rule and of the script's own text, each asserted beside the behaviour it states, with this page's words, its `stage` row and its example of the report against the code | `test/python/misc/test_model_samples_perf.py` | any host, level L0 |
| the speculative script without a GPU: the speculative configuration key by key -- one case per key of its three tables: a key that is read held to what is accepted and required, a neutral key at its value and at `null`, a refused key by name with its reason, and a key no table holds; the script's flags and every refusal by its sentence; the line it prints before the load; its `main()` on the CPU to its last line on a checkpoint directory with a drafter that is generated in the test, a request body from a file and from standard input -- standard output the text alone, the tokens and the kept counts those of the library's loop; three wrong loops through `main()`, each failing its line; on stated facts the four plans and the one memory sentence before the load, and a count of drafted tokens no row holds as a `SKIP`; the last line of both scripts over every input against the table of last lines; the memory figures of its help text; on stand-in providers the stack `native` to its last line: `PASS`, `FAIL`, and `RAN` without a profiler; the two refusals of a drafter that is not the session's own; a snapshot behind a step that kept drafted tokens; the README's recipe on the files of both scripts | `test_model_samples_text.py`, `test_model_samples_plan.py`, `test_model_samples_census.py`, `test_model_samples_verify.py`, `test_model_samples_drafter.py` under `test/python/misc/` | any host, level L0; the checkpoint tests skip visibly without `jinja2`, `tokenizers` or `safetensors` |
| the paged attention cache of a state, which no provider row asks for yet, without a GPU: the pool a state builds and the pages a sequence owns; a write at the page and row of its position and nowhere else, and the reads of the first positions against the plain cache, by index arithmetic written in the test; the position and slot tables value by value, the cos / sin rows against the reference chain's own function, the views a kernel is handed, and a cache that refuses tables built for another pool; a session on a paged cache against the same session on the plain one, byte for byte, with a prompt longer than a page, across positions 63 and 64 and to the last row of the last page; the checks of a run on it (the block that asked for the pool judged, every sequence alone on its window of the pool); two layouts in one session, and a page that is no whole number of positions, as a wrong table; the memory sentence with the pool and the tables; verify steps that reject rows on both sides of a page boundary, the rejected rows filled with NaN and never read; a snapshot and its restore; the three wrong loops of a session behind its prompt, each failing its line | `test_model_samples_plan.py`, `test_model_samples_verify.py` under `test/python/misc/` | any host, level L0 |
| the default run of each sample: `[NN] PASS` with its own checks, both streams held; the same run on the stack `fe-entries`, in a fresh interpreter: the cuDNN Frontend rows of `native` with their tie-out and the kernel check on real launches -- for the speculative script whole steps: the drafter's passes, the verify step, the one host read; the performance mode of the seeded model (`--perf`) on both stacks, in a fresh interpreter: `[NN] PASS`, the report in its reduced-model form, the report file, and in it the per-kernel table of real launches -- the rows hold every record of a step, an exact row its planned launches, and the rows' mean time per step sums to the mean GPU busy; the two reference chains of an attention layer on a paged cache against the plain one at the published widths, with the kernels they launch on the pool held to their families (none with the id of a step's embedding lookup) | `test/python/linear_attention/frost/test_model_sample_qwen_dense.py` | one GPU the family serves; L0 runs two layers, L1 four; the cell of the paged cache runs there and on compute capability 8.0, and is skipped by name on any other class |
| the kernel check on the stack `fe-cublas` (`--check-kernels`), and the stack `native` itself | by hand; the family README states the GPU class of the last run | no requirement file of this repository's tests names the package `native` needs (FlashInfer), which is why `fe-entries` exists. The GPU test has an L0 cell for `native` that runs where the package is installed at a validated version and is skipped elsewhere: skipped is uncovered, not passing |
| a real checkpoint: text out, and the sample's checks on the first 16 steps of every prompt | by hand, with a run record; the family README states the checkpoint, the versions and the GPU class of the last run | a GPU with room for the stack; never in CI |

`[00] PASS` on a checkpoint says that the provider blocks agree with the reference blocks on the weights both were given.
It cannot see an error in how the checkpoint was read: the two stacks read the same weights through the same name map,
scales, merges and dequantisation, so a wrong reading passes replay, blocks and tokens alike. The reading of a real checkpoint is verified
by the text the run produces, read by a person, and by comparing the ids of `--dump-tokens` with another implementation;
the host tests verify the reader on a checkpoint they generate.

No tokenizer, template or weight file of a real checkpoint is committed. A default run that skips on a GPU class is
uncovered there, not passing. The family README lists, per GPU class, whether the default run has been verified.

Version updates for `samples/models/requirements.txt` come from the `pip` entry for `/samples/models` in
`.github/dependabot.yml`, as pull requests, and are handled by the maintainers of this repository, as for its other
requirement files. The packages `constraints-native.txt` pins are on the ignore list of that entry, each by its name
with the three update types of a version update and nothing else: the entry proposes no version update for them -- not
for a pin, and not for the lower bound of `flashinfer-python` in `requirements-flashinfer.txt` either, which is covered
through the exact pin of the same package --, and a security update is not ignored. The pins are the versions a
family's kernel census was recorded with, and a pin changes only together with a recorded run; a security update of one
fails the host tests until that run is made.
