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
out on a checkpoint from local disk, with the checkpoint's own tokenizer and chat template. Both run on the stack
`fe-cublas`, which is a reference path: it holds every quantised weight as a bf16 copy and issues every launch from Python.
Kernels on the stored 8-bit and 4-bit formats, FlashInfer where it has the block, and a captured decode step (a step
recorded once and replayed without Python between its launches) follow as a second stack (`native`). The destination is a runnable performance reference. Until a sample reaches it, its README says
what the sample measures and what it does not, and no README here carries a performance table: numbers are published
through the formal mode of the `benchmark/` folder.

## What a model sample is, and what it is not

| it is | it is not |
|---|---|
| one program per phase (decode, later prefill + decode and speculative decode), one process | a server: no socket, no request queue, no admission of work while a step runs |
| a fixed set of sequences, decided before the first step | continuous batching or a scheduler |
| a printed table of who serves every block, decided before anything is loaded | a stack chosen by what happens to be installed |
| a self-check against a plain-PyTorch reference that accumulates in fp64 | a comparison with a serving engine |
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

# A checkpoint is your own download: the samples fetch nothing. 22 GB of disk; a GPU of compute capability 10.x or 12.x
# with about 57 GiB free; about two and a half minutes for the run below (qwen_dense/README.md has the details).
pip install -r samples/models/requirements.txt     # tokenizers, jinja2, safetensors: only for a checkpoint; brings the hf command
hf download nvidia/Qwen3.8-27B-NVFP4 --local-dir /data/Qwen3.8-27B-NVFP4
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --prompt "Why is the sky blue?" --max-new-tokens 256
```

PyTorch warns once on stderr that NumPy is missing (`Failed to initialize NumPy`); the samples need none.

`samples/models/requirements.txt` holds sample-only dependencies with lower bounds; none of them is a dependency of the
`nvidia-cudnn-frontend` package, and the default run of every sample needs none of them. They are imported only inside the
functions that read a checkpoint: `tokenizers` and `jinja2` where its text is read and streamed (`text_io.py`),
`safetensors` where its shards are described and read (`weights.py`); never at module import. Where one is
missing, a run with `--model` prints a `SKIP` that names the file. The family README states which checkpoint the sample
was last run with, how much disk and GPU memory it needs and where the checkpoint comes from; a sample reads it from the
directory it is given and from nowhere else.

A sample checks the GPU, the library versions and the free memory before it loads anything, for every sequence count of
the run at once: a `SKIP` comes before the first load or not at all. It ends in one of:

| last line | exit code | meaning |
|---|---|---|
| `[00] PASS` | 0 | every check of the run held |
| `[00] SKIP <requirement>` | 0 | this GPU, library version or free-memory budget is not served; the sentence names the fix |
| `[00] RAN reference stack (no provider checked)` | 0 | `--stack reference`: every block ran on the reference, so nothing was verified |
| `[00] RAN (no provider block was judged)` | 0 | `--provider` flags put every judged block on its reference row: the run compared the reference with itself, and its blocks line is `REPORTED` |
| `[00] RAN <stack> stack (kernels not observed: <reason>)` | 0 | `--check-kernels` was given and the profiler's launch records could not be had (no profiler library in the environment, another profiler attached, records without launch arguments): the other checks held, the kernel check observed nothing. Never printed as `PASS`; a check that failed is `FAIL` whatever was observed |
| `[00] ERROR <reason>` | 1 | a contract was violated: an unknown provider key; a `--model` path that is not a directory (a hub name is not fetched), or a directory with a file that is missing or is not what its name says, or a copy of a checkpoint that is not whole (a shard missing, or shorter than its header describes); a `--messages` file that cannot be read or is not JSON; messages the checkpoint's chat template refuses or turns into no token; a `--dump-tokens` file that cannot be written; or an installed distribution that disagrees with the imported `cudnn`. A file is named by its own name, never by its path |
| `[00] FAIL` | 1 | a check did not hold; the check lines above it say which |

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

| module | what it holds |
|---|---|
| `text_io.py` | text in, token ids; token ids, streamed text out: a synthetic byte-level tokenizer for the seeded run; a checkpoint's own tokenizer and chat template (messages, tool schemas, template options) from local disk |
| `weights.py` | the three storage formats (bf16; 8-bit E4M3 with a scale; packed 4-bit with block scales and a global scale, the NVFP4 of a checkpoint's name), the merge rule, dequantisation; seeded checkpoints built in memory; a checkpoint on local disk, described from its shard headers and read tensor by tensor without pickle |
| `state.py` | the key/value cache of the attention layers; the recurrent-state and conv-history pool of the linear-attention layers, by slot id |
| `blocks.py` | provider rows as data (GPU classes, ranges, stack, the form an entry is called in), the plan for (GPU class, sequences, tokens per sequence, stack), the served-by table, the memory sentence, typed declines |
| `census.py` | the kernel census of a decode step: what a block declares to launch (its census entries), the reader of the profiler's exported trace, canonical ids of the kernel families, the kernel table, and the kernel check that holds every launch of the traced steps against the plan (`--check-kernels`) |
| `providers/fe.py` | adapters to cuDNN Frontend entries |
| `providers/library.py` | adapters to library kernels (cuBLAS through PyTorch today) |
| `providers/torch_ref.py` | every block in plain PyTorch with fp64 accumulation: the reference of the tie-out |
| `loop.py` | `Session`: plan, load, prefill, step for a fixed set of sequences; stop rules |
| `sampling.py` | greedy token choice on the GPU |
| `timing.py` | host timestamps, stage lines, the traced pass (GPU busy, idle, launches per step from kernel activity records; on request the launch records themselves), the report's text |
| `tie_out.py` | bands, the replay check, the per-block and token checks, batch invariance; the line of the kernel check |
| `<family>/` | what is specific to a model: geometry, layer order, block wiring, the provider table and the regime table as data |

## What a run reports

Every line below is printed by every run of the stack `fe-cublas`; `--stack reference` prints no `checks` and no
`first launch` line. A run on the reduced model prefixes its times with "reduced model" and prints no rate. None of it is
a measurement of record: no clock is locked, and the stack is launched from Python.

Words a run and these pages use. *Sequence*: one text being generated. *Row*: one token position of a step; a plain decode
has one row per sequence. *Step*: one pass of the model over all rows. *Period*: from the first launch of one step to the
first launch of the next. *Captured*: recorded once and replayed without Python between the launches; these stacks are
launched from Python instead, one launch after the other. *Clock not locked*: the GPU's clock frequency is left as it
is, one reason no timing here is a measurement of record. *Teacher-forced*: a provider block and its reference block are
run on the same operands and the same state, call by call, so that an error cannot accumulate from block to block.

| line | definition |
|---|---|
| header | the model (the reductions of the default run, or the checkpoint's directory name, shard count, size and manifest hash), the GPU, the sequence counts, the stack; the `cudnn` version, editable checkout or wheel, a source hash, the installed distribution; the reference's accumulation precision |
| `served by` | per block: calls per step, provider, entry, the rows the entry serves, and the entry's published record (the GPU class and rows of a published measurement of that entry, `none`, or the reason a row is a fallback: not a measurement of this run or this GPU); the declared regime; then the memory the run needs against what is free, in GiB (2^30 bytes). Decided and printed before anything is loaded |
| `stage` | seconds of import, describe (a checkpoint's tokenizer, chat template and shard headers; no weights), plan, read (or draw), merge, dequantise, compile and first launch of each compiled form, warm-up |
| `decoding`, `checks on` | two lines that say what comes next, printed before the phases in which a run prints nothing: the prompt and the decode loop, and the checks. The text is not streamed: it follows the report, so that printing is not part of a timed step |
| `text path` | render = chat messages to prompt text; tokenize = prompt text to ids, also per prompt token; ids to GPU = the copy of the prompt's ids to the device; text out = ids to text per produced token, inside the loop. With a checkpoint, the first call of the process and the same call repeated |
| `first token` | the prefill route and its time. While the prompt goes through the decode step, one step per token, this is not a time to first token of record |
| `steps` | period = from one step's first launch to the next step's, median and mean; in the step = from the first launch to the ids on the host; turnaround = from the ids on the host to the next launch (detokenise, stop check); block calls per step |
| `traced pass` | the first 32 steps again, from a zero state, under the PyTorch profiler with kernel activity records only. The period under the profiler comes first, with its stretch against the untraced one. GPU busy = the union of the intervals of the step's kernels and copies; idle inside the traced step = the step's span on the GPU less GPU busy (the two are parts of the traced step); kernel launches and copies per step; host-exposed = the untraced period - GPU busy. Where the records cannot be had the line says why |
| `rate` | tokens produced / wall time of the decode loop, per sequence |
| `bytes per step` | arithmetic on the model's tensor list: the bytes of the weights a step reads in their stored formats plus the state it reads and writes, against the same with the weights as this stack holds them; in GB (10^9 bytes), as a checkpoint's size in the header |
| `note` | what bounds a step of this run, from the traced pass: the launch path from Python where host-exposed time is the larger part of the period, the stack's GPU work where GPU busy is; "not measured" without a traced pass. No time per block and no limit (floor, speed-of-light figure) is printed by the stacks of this area so far |
| `text out` | the text of sequence 0 and why it ended |
| `ids hash` | in the last line before the verdict: a SHA-256 over the produced ids of every sequence of the run, its first 12 hex digits; equal between two runs that produced the same tokens. The other two hashes of a run are in its header: the manifest hash of a checkpoint (its layout, not its values) and the source hash of the package files the cuDNN Frontend provider runs |
| `checks` | replay, blocks, tokens, batch invariance (the family README has the bands; a run prints each band's text once, under the block lines); with a checkpoint, on the first 16 steps of every prompt, and the token line is `REPORTED`, not judged against a band. With `--check-kernels` a fifth line, `kernels`: see below |
| `kernels, planned`, `kernels, as the entries state them` | only with `--check-kernels`. Before the load, from the plan alone: per block the kernels one call launches and how they are judged -- `name and launch shape, as the entry states them` (an entry of cuDNN Frontend: its own symbol, grid, block and shared memory), `name` (a kernel known by its name alone), `by family` (a fallback row: a reference chain or a multiply over a copy launches kernels of PyTorch and cuBLAS, how many is not a fact of the row), `host path, not compared`. After the load: what each entry states for this run's row count -- its launch shape and the symbol of the kernel it compiled |
| `kernels` (a line of `checks`) | only with `--check-kernels`: every launch of the traced steps against the plan. The form a provider row names is compared with the entry's own form on the objects; every exact kernel must be there once per call, in the wiring order, under the entry's own symbol and with exactly the entry's grid, block and shared memory; every other launch must belong to a family that a fallback row or the host path at that place declares; copies are counted, and one between two exact rows is not declared. `PASS`, `FAIL` with one line per kind of difference, or `NOT OBSERVED` with the reason. The kernels a library chose inside the fallback rows are printed and not gated |
| `GPU memory`, `first launch` | PyTorch's peak and the growth on the device since the plan, in GiB; what the first launch of a compiled form took outside PyTorch's pool, in MiB |

## What is verified where

| what | by | where |
|---|---|---|
| the helper layer without a GPU: tables, plan, declines, memory sentence, environment check, token choice, stop rules, a decode of a tiny geometry on the CPU, also through the sample's own loop function; the sample's flags and its last line; the tie-out itself on stated tensors and through stand-in providers with deliberate defects (a NaN, one wrong value head, a row that reads its neighbour) | `test/python/misc/test_model_samples_plan.py` | any host, level L0 |
| the weight formats: the bf16 copies both stacks read against a decode written from the format definitions, the quantisers, both merge rules, the reference multiply; a checkpoint directory generated in the test: described from its headers, bound through the family's name map, read back bit for bit, decoded on the CPU to the bytes the same tensors give from memory; the declines of a checkpoint the family does not read; the errors of a copy that is not whole and of what else a user can get wrong, each as the one line the sample prints; the memory sentence of `--model` before a tensor is read | `test/python/misc/test_model_samples_weights.py` | any host, level L0; the checkpoint tests skip visibly without `safetensors` |
| the text side: the synthetic tokenizer and its stream; chat requests from JSON files; a chat template with the constructs of a checkpoint's own (tool schemas, a reasoning block, tool calls and results, several turns, an option, a refusal) against stored strings; a tokenizer trained in the test, encoded, decoded and streamed, also when it was saved with truncation and padding; the requirement file | `test/python/misc/test_model_samples_text.py` | any host, level L0; the checkpoint tests skip visibly without `jinja2` or `tokenizers` |
| the kernel check without a GPU: the canonical ids of the kernel families, the reader of an exported trace, the wiring order of a step and its call log, the form check on the objects, and the comparison on constructed launch records -- an extra launch, a missing one, a swapped pair, a kernel of a fallback family where none is declared, the expected name with another grid, block or shared memory, another symbol, an undeclared copy: each fails with its own sentence; through a session, stand-in providers whose launches a stand-in profiler records, among them one that declares one kernel and launches another; the verdict without a profiler | `test/python/misc/test_model_samples_census.py` | any host, level L0 |
| the default run of each sample: `[NN] PASS` with its own checks | `test/python/linear_attention/frost/test_model_sample_qwen_dense.py` | one GPU the family serves; L0 runs two layers, L1 four |
| the kernel check on a GPU (`--check-kernels`: the profiler's records of real launches against the plan) | by hand; the family README states the GPU class of the last run | no test of this repository runs it yet |
| a real checkpoint: text out, and the sample's checks on the first 16 steps of every prompt | by hand, with a run record; the family README states the checkpoint, the versions and the GPU class of the last run | a GPU with room for the stack; never in CI |

`[00] PASS` on a checkpoint says that the provider blocks agree with the reference blocks on the weights both were given.
It cannot see an error in how the checkpoint was read: the two stacks read the same bf16 copies through the same name map,
scales and merges, so a wrong reading passes replay, blocks and tokens alike. The reading of a real checkpoint is verified
by the text the run produces, read by a person, and by comparing the ids of `--dump-tokens` with another implementation;
the host tests verify the reader on a checkpoint they generate.

No tokenizer, template or weight file of a real checkpoint is committed. A default run that skips on a GPU class is
uncovered there, not passing. The family README lists, per GPU class, whether the default run has been verified.

Dependency alerts on `samples/models/requirements.txt` are raised by the `pip` entry for `/samples/models` in
`.github/dependabot.yml` and are handled by the maintainers of this repository, as for its other requirement files.
