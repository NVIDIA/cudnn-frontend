# Model samples

Each folder runs one open-weight model family end to end in a single process: text in, tokens, prefill, decode steps, text
out. The samples are not a serving engine. They show which kernel serves each block of the model, check the whole path
against a reference, and make the cost of every step visible. Every sample prints a block table before it runs: block,
provider (cuDNN Frontend, cuBLAS, FlashInfer or PyTorch) and the rows that provider serves. By default a sample builds the
model at its published layer widths with two layers and seeded weights. The samples never download weights or tokenizer
files. The cuDNN Frontend APIs used here are experimental. Timings printed by a sample are for orientation only.

**Where this area stands.** The first family is the dense Qwen 3.5-architecture model ([qwen_dense](qwen_dense/)), and its
first sample is the plain decode on the reduced seeded model: token ids in, token ids out, self-checking. Reading a real
checkpoint from local disk (`--model`, with the checkpoint's own tokenizer and chat template) is the next step; kernels on
the stored 8-bit and 4-bit formats, FlashInfer where it has the block, and a captured decode step follow as a second
stack (`native`). The destination is a runnable performance reference. Until a sample reaches it, its README says what the
sample measures and what it does not, and no README here carries a performance table: numbers are published through the
formal mode of the `benchmark/` folder.

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

Building cuDNN Frontend from this checkout needs a CUDA toolkit with its compiler, cuDNN 9 headers and libraries, CMake 3.26
or newer and a C++17 compiler. Install PyTorch first so the build finds the cuDNN headers of its wheel, or set `CUDNN_PATH`.

```bash
pip install --group torch
pip install -e .                                   # the decode entries are newer than the last released wheel
python samples/models/qwen_dense/00_decode.py      # 2 layers, seeded weights, self-check: no checkpoint, no extra package
```

A sample checks the GPU, the library versions and the free memory before it loads anything, for every sequence count of
the run at once: a `SKIP` comes before the first load or not at all. It ends in one of:

| last line | exit code | meaning |
|---|---|---|
| `[00] PASS` | 0 | every check of the run held |
| `[00] SKIP <requirement>` | 0 | this GPU, library version or free-memory budget is not served; the sentence names the fix |
| `[00] RAN reference stack (no provider checked)` | 0 | `--stack reference`: every block ran on the reference, so nothing was verified |
| `[00] ERROR <reason>` | 1 | a contract was violated: an unknown provider key, or an installed distribution that disagrees with the imported `cudnn` |
| `[00] FAIL` | 1 | a check did not hold; the check lines above it say which |

The first lines of a run state the environment: the `cudnn` version, whether it is an editable checkout or a wheel, a hash of
the package files the sample's cuDNN Frontend provider runs, and the installed distribution's version. If the last two
disagree the sample stops with `ERROR`: an editable install with stale metadata lets a later `pip install` replace the
checkout with a released wheel. `pip install -e .` again fixes it. The GPU comes first: on a GPU the sample does not serve
it prints its `SKIP`, whatever is installed.

## The helper layer

`model_samples/` is one importable package shared by every family. It is sample code: not installed, not part of the wheel,
and it imports nothing beyond cuDNN Frontend, PyTorch and the standard library.

| module | what it holds |
|---|---|
| `text_io.py` | text in, token ids; token ids, streamed text out (a synthetic byte-level tokenizer for the seeded run) |
| `weights.py` | the three storage formats (bf16; 8-bit E4M3 with a scale; packed 4-bit with block scales and a global scale), the merge rule, dequantisation, seeded checkpoints built in memory |
| `state.py` | the key/value cache of the attention layers; the recurrent-state and conv-history pool of the linear-attention layers, by slot id |
| `blocks.py` | provider rows as data, the plan for (GPU class, sequences, tokens per sequence, stack), the served-by table, the memory sentence, typed declines |
| `providers/fe.py` | adapters to cuDNN Frontend entries |
| `providers/library.py` | adapters to library kernels (cuBLAS through PyTorch today) |
| `providers/torch_ref.py` | every block in plain PyTorch with fp64 accumulation: the reference of the tie-out |
| `loop.py` | `Session`: plan, load, prefill, step for a fixed set of sequences; stop rules |
| `sampling.py` | greedy token choice on the GPU |
| `timing.py` | host timestamps, stage lines, the report's text |
| `tie_out.py` | bands, the replay check, the per-block and token checks, batch invariance |
| `<family>/` | what is specific to a model: geometry, layer order, block wiring, the provider table and the regime table as data |

## What is verified where

| what | by | where |
|---|---|---|
| the helper layer without a GPU: tables, plan, declines, memory sentence, environment check, token choice, stop rules, a decode of a tiny geometry on the CPU; the tie-out itself on stated tensors and through stand-in providers with deliberate defects (a NaN, one wrong value head, a row that reads its neighbour) | `test/python/misc/test_model_samples_plan.py` | any host, level L0 |
| the weight formats: the bf16 copies both stacks read against a decode written from the format definitions, the quantisers, both merge rules, the reference multiply | `test/python/misc/test_model_samples_weights.py` | any host, level L0 |
| the default run of each sample: `[NN] PASS` with its own checks | `test/python/linear_attention/frost/test_model_sample_qwen_dense.py` | one GPU the family serves; L0 runs two layers, L1 four |
| a real checkpoint | not yet read by any sample | -- |

A default run that skips on a GPU class is uncovered there, not passing. The family README lists, per GPU class, whether the
default run has been verified.
