# Dense Qwen 3.5-architecture model

Samples for the dense model of the Qwen 3.5 architecture at its published 27B-class layer widths: hidden 5,120; 64 layers,
three linear-attention (gated delta-rule, GDN) layers to one attention layer; linear attention with 16 key heads and 48
value heads of 128; attention with 24 query heads and 4 key/value heads of 256; MLP width 17,408; 8-bit projections, a
4-bit MLP and a 4-bit output head. `nvidia/Qwen3.8-27B-NVFP4` is a checkpoint of this architecture, and the one this
sample was last run with; the NVFP4 of its name is its 4-bit storage format (packed 4-bit codes with block scales and a
global scale). Area rules, the helper layer and the words a run uses: [../README.md](../README.md).

| script | what it runs | status |
|---|---|---|
| `00_decode.py` | plain greedy decode in one process: on the reduced seeded model, token ids in and out, self-checking; with `--model`, text in and text out on a checkpoint from local disk | this page |

```bash
python samples/models/qwen_dense/00_decode.py                          # 2 layers, 1 and 4 sequences, 32 tokens each
python samples/models/qwen_dense/00_decode.py --sequences 4 --layers 4
python samples/models/qwen_dense/00_decode.py --stack reference        # every block on the reference: RAN, never PASS
python samples/models/qwen_dense/00_decode.py --check-kernels          # also: every launch of the traced steps against the plan
python samples/models/qwen_dense/00_decode.py --stack native           # the target kernel set, as far as it is served here

# A checkpoint is your own download: the sample fetches nothing. 22 GB of disk; a GPU of compute capability 10.x or 12.x
# with about 57 GiB free (about 60 GiB on 12.x); about two and a half minutes for the first run below ("A checkpoint from local disk" has the details).
pip install -r samples/models/requirements.txt                         # tokenizers, jinja2, safetensors; brings the hf command
hf download nvidia/Qwen3.8-27B-NVFP4 --local-dir /data/Qwen3.8-27B-NVFP4
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --prompt "Why is the sky blue?" --max-new-tokens 256
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --messages chat.json --dump-tokens ids.json
```

| flag | default | meaning |
|---|---|---|
| `--model DIR` | -- | a checkpoint directory on local disk; never a hub name; nothing is downloaded |
| `--prompt TEXT` | `Why is the sky blue?` with `--model` | one user message, rendered by the checkpoint's chat template; repeat the flag for several prompts, run one after the other in one process |
| `--messages FILE` | -- | chat messages as JSON: a list of messages, or an object with `messages` and optionally `tools` (JSON tool schemas) and `chat_template_kwargs` (options the checkpoint's template understands); repeatable |
| `--max-new-tokens N` | `256` | with `--model`: stop after N new tokens unless an end token comes first. Without `--model` the flag is an error: the seeded run decodes `--force-tokens` |
| `--force-tokens N` | `32` without `--model`, off with it | decode exactly N tokens per sequence, ignoring the end token; not together with `--max-new-tokens` |
| `--sequences N[,N...]` | `1,4` without `--model`, `1` with it | sequences per run, fixed before the first step; a list repeats the run per count; with `--model` every sequence of a run gets the same prompt |
| `--layers N` | `2` | layers of the reduced model: the last N of the published order, so the model ends in an attention layer; not with `--model` |
| `--stack fe-cublas\|native\|fe-entries\|reference` | `fe-cublas` | `fe-cublas`: cuDNN Frontend + cuBLAS + PyTorch, checked against the reference. `native`: the target kernel set of the decode step ("The stack `native`" below); the kernel check is always on. `fe-entries`: the cuDNN Frontend rows of `native`, everything else on fallback rows. `reference`: every block in PyTorch, nothing checked. Explicit, never chosen by what is installed |
| `--provider BLOCK=KEY` | -- | selects one declared provider row of a block; the key list is closed. Block names are in the table a run prints, keys in the table below; an unknown block or key is an `ERROR` that lists the declared ones |
| `--reference-precision fp64\|ieee-fp32` | `fp64` | accumulation of the reference blocks; printed in the header. `ieee-fp32` sets `torch.backends.cuda.matmul.fp32_precision` for the rest of the process, which ends with the run |
| `--dump-tokens FILE` | -- | per prompt, the prompt ids and the produced ids of every sequence as JSON, for a comparison made elsewhere |
| `--check-kernels` | off; always on with `--stack native` and `--stack fe-entries` | adds the kernel check: prints the kernel table, keeps the profiler's record of every launch of the traced steps and holds them against the plan ("The kernel check" below). Not with `--stack reference`, which checks nothing |
| `--dump-census FILE` | -- | the launch records of the traced steps as JSON (per launch its kernel id, name, grid, block, shared memory, registers per thread), with the block calls of a step, what the entries state, and per call site the entry instance that served it and the form that instance states (`served`, beside the row's `form`), for a comparison made elsewhere; implies `--check-kernels`. In a stack that declares a target, also each call's identity and the run's three numbers (`target`) |

A `--messages` file (the `chat.json` of the last command above), here with a system turn and the reasoning switched off:

```json
{"messages": [{"role": "system", "content": "Answer in one short sentence."},
              {"role": "user", "content": "What is 17 times 23?"}],
 "chat_template_kwargs": {"enable_thinking": false}}
```

A list of messages alone is also a valid file, and `tools` takes JSON tool schemas. The options are those of the
checkpoint's chat template: for `nvidia/Qwen3.8-27B-NVFP4`, `enable_thinking: false` leaves out the reasoning the model
otherwise writes before its answer.

## What the default run is

The reduced model keeps every layer width and reduces what the run prints as reductions: 2 of 64 layers (one linear-attention
and one attention layer), a vocabulary of 15,520 ids, seeded weights and a synthetic byte-level tokenizer. The weights are drawn
on the CPU generator and stored in the model's three formats, so the 8-bit and 4-bit paths are exercised. Two properties of
the seeded weights are choices, not accidents: the projections that write back into the residual stream are drawn at a quarter
of the unit scale, and the output head is a seeded permutation of the embedding table. The second makes the greedy token
decidable by a wide margin, and it means the produced ids are a walk through that permutation: with seeded weights the layers
move the logits, not the choice. The layers are verified block by block.

The stack `fe-cublas` is a reference path, not a fast path. The linear-attention step runs on cuDNN Frontend's fused kernel;
every multiply runs on cuBLAS over a bf16 copy of the quantised weights, which moves 2 bytes per weight element where the
stored formats hold 1 or 0.56; the rest runs in PyTorch; and every launch is issued from Python, one block after the other.
It exists to prove the loader, the loop, the state handling and the tie-out, and it is the correctness oracle of the stacks
that follow. It claims no number.

## The stack `native`: the target kernel set

`--stack native` is defined by a kernel set, not by what is installed: the target kernel set of this model's decode
step, the kernels a specialised decode path launches for it, block by block. The set is checked in as data,
`model_samples/qwen_dense/kernel_census.json`: per block call of a step the kernel, the form it is called in, and per GPU
class, SM count and row count the launch shape that was recorded for it. For the plain step of the 64-layer model that is
484 launches. A statement about this stack is three numbers, never one: of those 484, 210 are the target's own compiled
kernels (the fused linear-attention step of cuDNN Frontend, library calls of FlashInfer and of cuBLASLt, a Triton kernel,
PyTorch's gather), 273 have a cuDNN Frontend entry with byte-equal results, and 1 is an operation that lives inside a
serving engine and gets a declared stand-in.

The stack is built block by block. A block whose target kernel is not served here yet stays on the fallback row it has in
`fe-cublas`, and a run prints it as `not yet (<fallback reason>)`. Today two entries are on their target forms:

| block | launches per step (64 layers) | what `native` launches | identity |
|---|---|---|---|
| token embedding lookup | 1 | PyTorch's gather | target: the target's own compiled kernel |
| linear-attention step | 48 | `cudnn.GdnDecodeFused` launched as a dependent of the launch before it and releasing the launch after it at entry (`PDL True`, `PDL_RELEASE True`) | target: the target's own compiled kernel |
| gate + quantise | 16 | `cudnn.AttentionGateQuantDecode` with 256 threads per thread block, releasing the launch after it at entry (`THREADS 256`, `PDL_RELEASE True`); compute capability 10.x | target: a cuDNN Frontend entry with byte-equal results |
| every other block of the table below | 418 | the fallback rows of `fe-cublas` | not yet |
| the 4-bit quantise in front of the output head | 1 | nothing: this sample has no block for it yet | the target's declared stand-in, not yet |

So a run with a checkpoint prints under its kernel table:

```text
the target kernel set of a step of this model, when a stack reproduces all of it: 484 launches per step = 210 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results + 1 declared stand-in
kernel identity, as planned: 65 of 484 launches of a step are target kernels (49 the target's own compiled kernels + 16 cuDNN Frontend entries with byte-equal results); not yet: add_norm_quantize 129, projection.fp8.in 64, projection.fp8.out 64, mlp.gate_up 64, mlp.down 64, attention.prepare 16, attention.core 16, output_head 1; 1 declared stand-in (not yet: output_head.quantize)
```

and with its checks the census line, `target census plain / cc 10.0 / 148 SMs (recorded 2026-11-12): NOT CLAIMED (fallback
rows present; ...)`. The default run on the two-layer model counts 3 of 20.

**Nothing is claimed yet.** A GPU class is claimed only when its section of the census says so, and that needs every
block of a step on its target kernel, a census recorded on that compute capability and SM count, and the package versions
of the record. Until then the line reads `NOT CLAIMED` with every reason that applies: a row selected with `--provider`
(`not the target set: <block> selected by --provider`), fallback rows present, a kernel of the target the plan has no
block for, no recorded section for this compute capability and SM count (the line names both) or for this row count,
versions other than the recorded ones. The census file has one section today: compute capability 10.0 with 148 SMs, at
1 row, recorded on 2026-11-12. On another GPU class, another SM count or another sequence count a run says `kernel
identity: not stated for this run`: the kernels of the target are recorded per class and are never copied from one class
to another.

**What is checked already**, in every run of this stack (its kernel check cannot be switched off), for every launch on a
target kernel: the provider row's form is the census's form of that kernel and the entry's own form, compared on the
objects; the launch carries the entry's own symbol and exactly the launch shape the entry states; and where the census
has a section for the run, the launch shape is the recorded one -- grid (96, 1, 1), block (256, 1, 1) and 42,112 B of
shared memory for the fused step, grid (3, 1, 1), block (256, 1, 1) and no shared memory for the gate, at 1 row on
compute capability 10.0 with 148 SMs. A difference fails the run with a sentence that names the block.

**What this stack is not, today.** It is not a fast path: it holds the same bf16 copies as `fe-cublas`, needs the same
memory and issues every launch from Python. No timing of it means anything. And a target kernel between fallback rows
does not run in the target's surroundings: the fused step waits for the launch before it and releases the launch after
it, the gate releases the launch after it, but the launches around them here are PyTorch's and cuBLAS's, which neither
release early nor start early, so both knobs are exercised and change nothing. With `--provider
linear_attention.step=fe.gdn_decode_fused` the stack runs the fused step in its `fe-cublas` form instead (no wait, no
release); the table then prints `not the target (...)` for the block and the census line says why the run is not the
target set.

`--stack fe-entries` is every cuDNN Frontend row of `native` with each block that a library or a temporary kernel will
serve there on its fallback row. It needs no package beyond cuDNN Frontend and PyTorch, and it is the stack the GPU test
of this area runs, so the rows of the entries stay verified wherever that test runs. Until `native` has a row a library
serves, the two stacks are the same rows.

On compute capability 12.x the gate entry does not serve; the block then stays on its reference chain (`not yet (reference
chain)`) in both stacks, and the census has no section for that class.

The stack was last run on compute capability 10.0 (2026-11-13, a 148-SM part, PyTorch 2.13.0, CuTe DSL 4.8.0; the census
line of every run names the PyTorch version as other than the recorded 2.14.0):

| run | result |
|---|---|
| the default run, `--stack native` | `[00] PASS` at 1 and at 4 sequences, with the ids hash of the default stack. A traced step of the two-layer model is 460 kernel launches and 37 copies; 3 are judged one by one. At 1 sequence the gather, the fused step and the gate have the recorded launch shapes; at 4 no identity is stated. The gate entry was inside its band with every byte the code of the nearest bf16 value (350,208 bytes at 1 sequence, 1,400,832 at 4), and equal alone and among 4 |
| the checkpoint, one sequence, the default prompt, 32 new tokens | `[00] PASS`: 65 launches of every one of the 32 traced steps on target kernels, in the wiring order (the gather, then three fused steps and one gate, 16 times), each in the recorded launch shape; 11,574 kernel launches and 968 copies per traced step in all. Replay held; 7 of 7 provider blocks inside their bands; the gate entry over 256 calls with all of its 1,572,864 bytes the code of the nearest bf16 value; the token line reported |
| the default run, `--stack fe-entries` | `[00] PASS`, the same lines; also as the L0 cell of the area's GPU test |
| `--stack native --provider linear_attention.step=fe.gdn_decode_fused`, 1 sequence | `[00] PASS`; `NOT CLAIMED (not the target set: linear_attention.step selected by --provider; ...)`; the 2 remaining target launches in the recorded shape |

The fused step has the same launch shape in both of its forms, grid (96, 1, 1), block (256, 1, 1), 42,112 B at 1 row:
that it waits and releases shows in its symbol and in its form, not in a launch record. The stack has not run on
another GPU class.

## A checkpoint from local disk

`--model DIR` runs a checkpoint of this architecture in the build with 8-bit projections and a 4-bit MLP and output head.

| you need | |
|---|---|
| GPU | compute capability 10.x or 12.x **and**, for the 27B checkpoint on the stack `fe-cublas`, about 57 GiB of free memory (about 60 GiB on compute capability 12.x, where the first launch of the fused step takes about 3 GiB more): in practice a data-center Blackwell GPU. A 32 GB card prints the memory sentence and exits 0. The checkpoint has run on compute capability 10.0 and 10.3 |
| disk | 22 GB for the checkpoint |
| packages | `pip install -r samples/models/requirements.txt` |
| time | about two and a half minutes for the default prompt with 256 new tokens (the seconds are below) |

The sample was last run with:

| | |
|---|---|
| checkpoint | `nvidia/Qwen3.8-27B-NVFP4`: 3 shards, 2,194 tensors, 21.9 GB; licence field on its model card `apache-2.0`, not gated (card read 2026-11-10) |
| manifest hash | `95049016a247`, printed in the first line of a run: a SHA-256 over every shard's name, size and header (tensor names, formats, shapes, offsets), not over the values |
| packages | tokenizers 0.23.3, jinja2 3.1.6, safetensors 0.8.0; the lower bounds of `samples/models/requirements.txt` (0.21.1, 3.1.6, 0.4.3) pass the host tests of the text and weight readers |
| run | 2026-11-12, compute capability 10.0 (aarch64), in two environments: PyTorch 2.14.0 with CuTe DSL 4.8.0, and a fresh environment built by the lines of this file with PyTorch 2.15.0 and CuTe DSL 4.8.1. Three prompts at one sequence (two plain user turns, one with a tool schema and the reasoning switched off) and one prompt at two sequences; text out, `[00] PASS` in every run, the token line reported. The run before it was on compute capability 10.3 (2026-11-11) |
| provider rows | every row of the stack `fe-cublas` ran in that run: `cudnn.GdnDecodeFused` of cuDNN Frontend 1.33.0 from this checkout, cuBLAS through PyTorch 2.14.0 and 2.15.0, the PyTorch chains |

The sample downloads nothing. Fetching the checkpoint is your own step, for example
`hf download nvidia/Qwen3.8-27B-NVFP4 --local-dir /data/Qwen3.8-27B-NVFP4` (the command comes with the packages of the
requirement file); it needs 22 GB of free disk, and the weights carry their own licence, which is not this
repository's. The files read: `config.json`, `tokenizer.json`,
`chat_template.jinja` (or the template in `tokenizer_config.json`), `generation_config.json` for the end tokens, and the
`*.safetensors` shards. A path that is not a directory is an `ERROR` that says nothing is downloaded; a directory that
lacks one of the files, or holds one that is not what its name says, is an `ERROR` that names the file; so is a copy
that is not whole: a shard that the index file (`model.safetensors.index.json`) or the shards' numbering asks for and
that is not there, or a shard that does not hold the bytes its header describes (a download that was cut). A checkpoint of
another architecture or in other storage formats is a `SKIP` that names the first tensor that does not fit.

What happens, in order: the tokenizer, the template and the shard headers are read (no weights); every prompt is rendered
and encoded; the plan is resolved and printed with the memory the run needs, and a GPU with less free memory is a `SKIP`
with exit code 0 before a tensor is read; then the weights are read, merged (query, key and value; `qkv` and `z`; the
beta and alpha rows, beta first; gate and up) and dequantised to bf16 copies, the fused step is compiled, and each prompt
is fed through the decode step and decoded greedily to an end token or to `--max-new-tokens`. The text of sequence 0 is
printed; reasoning the model writes before its answer is part of that text.

How long it takes, as orientation values of one run (compute capability 10.3, the default prompt, 256 new tokens, the
checkpoint in the page cache): about two and a half minutes. About 6 s until the first step (import, read, merge,
dequantise, compile, warm-up); 14 s for the 58 prompt tokens; 54 s for 225 new tokens, a quarter of a second per token on
this stack; 12 s for the traced pass; and about a minute for the checks. The text is not streamed: it is printed after
the report, so that printing is not part of a timed step. A run prints one line before the prompt and the decode loop
(`[00] decoding: ...`) and one before the checks (`[00] checks on ...`), and nothing while either of the two runs.

Memory, in GiB (2^30 bytes), as the memory lines of a run print it: the stack `fe-cublas`
holds every weight as a bf16 copy, 50.1 GiB for this checkpoint, and its plan asks for about 56.7 GiB free with one
sequence and the default prompt: the copies, the state and working memory, which is an eighth of the copies for a model
of this size (2 GiB for the default run). A run with this checkpoint held 51.5 GiB in PyTorch at its peak, reserved
55.2 GiB from the device, and the device grew by 55.4 GiB. A GPU with less free memory than the 56.7 GiB is a `SKIP`
before a tensor is read. A 32 GB GPU cannot run this stack with a checkpoint of this size; it prints the
memory sentence and exits 0. A stack on the stored formats, which needs less than half, arrives with the next step of
this area.

The prompt goes through the decode step, one step per token (the `first token` line says so), and a step issues 484 block
calls from Python: on the run above about twelve thousand kernel launches and a thousand small copies per step, with the
GPU busy for about a quarter of the period. That is what this stack is: it reads 51.6 GB per step where the model in its
stored formats needs 17.9 GB, and it is bound by the launch path, not by the GPU. The timings a run prints are orientation
values of that run and no more; no rate of this stack is a number to quote.

The checks of a run with a checkpoint cover the first 16 steps of every prompt: replay, every provider block
teacher-forced against the fp64-accumulated reference with the bands below, the tokens, and with more than one sequence
the batch invariance. The block check is what verifies the providers, and it holds on trained weights with the same bands
as on seeded ones.

The token line of a checkpoint is `REPORTED`: it fails on a logit that is not finite and is judged against no band. The
model quantises its MLP inputs to 4 bits and its projection inputs to 8, so a difference of one bf16 rounding in a
multiply becomes a different code in the next quantiser, 64 layers deep. Two reference stacks that differ only in their
accumulation precision (fp64 against IEEE fp32) were up to 52 bf16 units of the largest logit (13.0 logits) apart over 80
steps of one prompt and chose another greedy token at 13 of them; the provider stack was up to 54 units (13.5 logits)
from the fp64 reference over the same steps. In the three 16-step windows the runs above checked, the provider stack was
8.7 to 10.3 logits from the reference and a second reference 7.5 to 11.4. No declared band separates a correct stack
from a failed one here. A band in bf16 units of the largest logit is 8, 16 or 32 logits wide by that logit's binade, and
the distance of two correct stacks does not follow the binade: 128 units, beyond which a stack that returns zeros
passes, are 8 logits where a window's largest logit is below 16, which is less than two correct stacks were apart. And a
greedy token is decidable only where its margin exceeds twice the band, which no step of these runs reached. So the line
prints what it measured (the provider stack's distance, the number of greedy tokens equal to the reference's, and the
same two figures for a second reference with the other accumulation precision) and gates on finite logits only. A
defect of a provider shows in the block check. A defect in what both stacks share, the reading of the checkpoint
included, shows in neither: the area README says what verifies that.

Which steps are checked: the first 16 of the prompt. This checkpoint's chat template opens a conversation that has no
system turn with a fixed one of its own, and a plain `--prompt` reaches the text of the prompt only after it, at token 45 of
the 58 of the default prompt. The checked window of every plain prompt is therefore the same 16 tokens of that
preamble. A `--messages` file with its own system turn or with tool schemas has another window. The runs above cover
three distinct windows (the preamble, a system turn, a tool schema), not one per prompt, and none of them reaches a
decode step.

Comparing the produced ids with a serving engine is not part of this sample; `--dump-tokens` writes them for a
comparison made elsewhere, and greedy decoding is not the checkpoint's own default (its generation settings sample).

## Who serves each block (stack `fe-cublas`)

| block | calls per step (64 layers) | provider key | entry [form] |
|---|---|---|---|
| token embedding lookup | 1 | `torch.index_select` | torch.index_select |
| residual add + RMSNorm + quantise | 129 | `torch_ref.add_norm_quantize` | torch_ref.add_norm_quantize, fallback: reference chain |
| 8-bit in-projection (linear attention QKVZ / attention QKV) | 48 / 16 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 8-bit weights |
| linear-attention step | 48 | `fe.gdn_decode_fused` | cudnn.GdnDecodeFused [PDL False, PDL_RELEASE False] |
| 8-bit out-projection | 48 / 16 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 8-bit weights |
| MLP gate / up + SiLU x up + quantise | 64 | `cublas.bf16_dequantized+torch_ref.silu_mul_quantize` | torch.matmul over a bf16 copy + torch_ref.silu_mul_quantize, fallback: bf16 copy of 4-bit weights |
| MLP down | 64 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 4-bit weights |
| attention preparation | 16 | `torch_ref.attention_prepare` | torch_ref.attention_prepare, fallback: reference chain |
| attention core | 16 | `torch_ref.attention` | torch_ref.attention, fallback: reference chain over a plain per-layer cache |
| gate + quantise | 16 | `torch_ref.gate_quantize` | torch_ref.gate_quantize, fallback: reference chain |
| output head | 1 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 4-bit weights |
| token choice | 1 | `torch.argmax` | torch.argmax |

484 block calls per step on the full model, 20 on the two-layer model and 34 on the four-layer one. By provider: cuDNN
Frontend 1; fallbacks 8 (cuBLAS over copies 4, PyTorch chains 4); PyTorch as the natural provider 2.

## Who serves each block (stacks `native` and `fe-entries`, compute capability 10.x)

| block | calls per step (64 layers) | provider key | entry [form] |
|---|---|---|---|
| token embedding lookup | 1 | `torch.index_select` | torch.index_select |
| residual add + RMSNorm + quantise | 129 | `torch_ref.add_norm_quantize` | torch_ref.add_norm_quantize, fallback: reference chain |
| 8-bit in-projection (linear attention QKVZ / attention QKV) | 48 / 16 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 8-bit weights |
| linear-attention step | 48 | `fe.gdn_decode_fused` | cudnn.GdnDecodeFused [PDL True, PDL_RELEASE True] |
| 8-bit out-projection | 48 / 16 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 8-bit weights |
| MLP gate / up + SiLU x up + quantise | 64 | `cublas.bf16_dequantized+torch_ref.silu_mul_quantize` | torch.matmul over a bf16 copy + torch_ref.silu_mul_quantize, fallback: bf16 copy of 4-bit weights |
| MLP down | 64 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 4-bit weights |
| attention preparation | 16 | `torch_ref.attention_prepare` | torch_ref.attention_prepare, fallback: reference chain |
| attention core | 16 | `torch_ref.attention` | torch_ref.attention, fallback: reference chain over a plain per-layer cache |
| gate + quantise | 16 | `fe.attention_gate_quant_decode` | cudnn.AttentionGateQuantDecode [THREADS 256, PDL_RELEASE True] |
| output head | 1 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 4-bit weights |
| token choice | 1 | `torch.argmax` | torch.argmax |

By provider: cuDNN Frontend 2; fallbacks 7 (cuBLAS over copies 4, PyTorch chains 3); PyTorch as the natural provider 2.
On compute capability 12.x the gate + quantise row is the reference chain of `fe-cublas`.

| GPU classes | sequences | regime | note |
|---|---|---|---|
| any | 1 and more | launch-bound | every block of these stacks is launched from Python, one launch or more per block; no byte-bound form is declared, and the sequence count at which one would win is not measured on any class |

## The checks of a run

Against the plain-PyTorch reference of every block, which keeps the chain's casts and accumulates in fp64
(`model_samples/tie_out.py` has the definitions):

| check | what must hold |
|---|---|
| replay | the same inputs from a fresh state give byte-identical ids and logits |
| blocks, teacher-forced | every value finite; every multiply within one bf16 rounding of the reference, element by element; the linear-attention step inside its band (below); in the stacks `native` and `fe-entries`, the gate + quantise entry inside its band against its twin (below) |
| tokens | the logits of the two stacks, each run end to end on its own state, are finite. On the reduced seeded model they are also within the declared logits band (`configs.py`: 4 bf16 units of the run's largest reference logit), and the greedy token equals the reference stack's at every step whose margin exceeds twice that band; steps below it are counted and printed as near-ties. On a checkpoint no band is declared and the line is `REPORTED` (the section on checkpoints says why). The largest logit distance and the number of equal greedy tokens are printed either way |
| batch invariance | one sequence alone gives the bytes it has among 4; the sequence that runs alone rotates with the step, so every row of the batch is covered: a gate for the linear-attention step on compute capability 10.0, 10.3 and 12.0 (in its `native` form on 10.0, the class that form has run on) and for the gate + quantise entry on 10.0, a printed fact for every other block and class |

The band of the linear-attention step is judged twice. Over the whole run: more than 99.9 % of its 8-bit output bytes
equal, more than 99.9 % of its recurrent state within 1e-4 and all of it within two bf16 units of the state's largest
value, every value finite, the conv history byte-identical and the null slot untouched. And per cell, which is one value
head of one sequence in one layer (one block of the step's launch grid, so its natural unit of failure): over the run more
than 99 % of the cell's output bytes equal and more than 99.9 % of the cell's state within 1e-4.

The state band has a share and a cap because a state "within 1e-4 everywhere" does not hold against an fp64-accumulated
reference: the fused step rounds its gates and conv taps to bf16 as a serving chain does, and where its fp32 sum falls on the
other side of a rounding midpoint one head's state moves by up to a few 1e-3 while the rest agrees to fp32 rounding. The band
tolerates such a move of one head at one step. Measured over runs of this sample with two to eight layers, the worst cell of
a run had about 0.1 % of its output bytes different and about 0.01 % of its state outside 1e-4, a tenth of what the band
allows. The band still catches a value head of one sequence that is wrong at every step, from two of its 128 output bytes or
0.1 % of its state's elements on; a NaN anywhere; any error of the wiring or of an operand's layout; a write to the null
slot; and any element beyond the cap. One wrong output byte of one head at every step stays inside it. The step's line also
prints the state's rms ratio (the rms of the difference over the rms of the reference state, at the worst call) as a second
view of the same comparison; it is printed, not gated.

The gate + quantise entry is not compared with the reference chain of its block but with its twin, the same gate
written at the kernel's rounding points (`model_samples/providers/torch_ref.py`: `GateQuantizeQuotient`). The reference
chain rounds the sigmoid to bf16 and then the product; the kernel forms the gated value as the quotient
`attn / (1 + exp(-gate))`, rounds it once to bf16 and once to an 8-bit code, which is another function, one code apart on
rare elements. Its band (`gate_fp8`) is the numerics contract on the entry's page
(`docs/operations/attention/gate_quant_decode.md`), not a share: the twin forms the quotient in the reference's
accumulation precision and keeps the two bf16 values around it; every byte must be the code of one of the two, and the
code of the nearest one wherever the quotient is more than 2.5 fp32 units from their midpoint; every value finite. Inside
that margin the kernel's fp32 quotient may fall on the other side of the midpoint. The line prints how many bytes did;
one byte outside the two codes, or off the nearest one beyond the margin, fails the block.

The token check has a declared band because a band taken from the run's own logit distance cannot fail. With seeded
weights it guards the path, not the layers: the layers move the logits by little (the two stacks are 1.0 to 1.4 units apart
with two to eight layers, and a fused step with its two gate rows exchanged moved that by a third of a unit), so a
defect of one block shows in the block check, and the token check catches what breaks the run as a whole: the state between
steps, the position, the output head, the token choice. That band, 4 units, is declared for the reduced seeded model; a
checkpoint declares none, and the section on checkpoints above says why its token line is reported.

The multiplies of this stack ask cuBLAS for the fp32 accumulator throughout: `model_samples/providers/library.py` switches
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` off for its call and restores it. With PyTorch's
default, cuBLAS may reduce the partial sums of a bf16 multiply in bf16; on compute capability 12.0 it did so for two of
this model's multiplies at 4 sequences, which put them more than a hundred bf16 units from the reference.

Batch invariance of the fused step on compute capability 12.x: the determinism page (`docs/operations/determinism.md`) has
no column for that class; its settings section records the fused step entries as verified on compute capability 12.0 with
the same statuses as on 10.0 and 10.3, and this sample's run agrees: the gate held at 2, 3 and 4 sequences. On that class
the two projection multiplies (cuBLAS) of a sequence differ in bytes between 1 and 4 sequences, inside their band; the run
prints it. On 10.7 every block was equal.

The provider stack and the reference stack read the same bf16 copies of the weights and use the same quantisers, so the
checks above cannot see an error in either. `test/python/misc/test_model_samples_weights.py` pins them against a decode
written from the format definitions.

## The kernel check

`--check-kernels` adds a fifth check to a run. The other four compare the stack with the reference; this one compares
what was launched with what the provider table says is launched. It prints two tables and one line of `checks`:

| | printed | what it says |
|---|---|---|
| `kernels, planned` | before the load, from the plan alone | per block the kernels one call launches (its census entries) and how each is judged |
| `kernels, as the entries state them` | after the load | for every block that an entry of cuDNN Frontend serves: the launch shape the entry states for this run's row count (grid, block, shared memory of a thread block) and the symbol of the kernel it compiled |
| `kernels` | with the checks | the profiler's record of every launch of the traced steps (the first 32 steps of the run, replayed) against the two |

On the stack `fe-cublas` two blocks launch a kernel that is judged one by one:

| block | kernel | held |
|---|---|---|
| linear-attention step | `gdn_decode_fused`, the kernel of `cudnn.GdnDecodeFused` | one launch per layer and step, in the wiring order; its name is the symbol the entry compiled for this run's row count; grid, block and shared memory equal the entry's own launch shape exactly; and the form the provider row names (`PDL False`, `PDL_RELEASE False`: the entry's pick per row count, launched without a wait on its predecessor and without an early release of its successor) equals the entry's own form, compared on the objects, and the entry instance that served each call of the traced steps states the same (the call log, written after each call from the object that was called) |
| token embedding lookup | `embedding_gather`, PyTorch's index-select kernel | one launch per step, first in the step, known by its name; no launch shape is declared for it |

Every other block of this stack is a fallback row: a reference chain or a multiply over a bf16 copy. What such a row
launches is PyTorch's and cuBLAS's own choice by shape, row count and GPU, so its launches are attributed by family
(`pytorch`, `cublas`) to the fallback rows that sit between two judged kernels in the wiring order, and their number is
not a fact of the row. A launch that belongs to no declared family there, a kernel of an entry that no block declares,
and a copy between two judged kernels with no fallback row between them each fail the check by name. The kernels cuBLAS
chose are printed under the line and are not gated. The token choice is the sample's host path: listed, not compared.

What the check cannot say. A kernel name does not prove a form: a generated symbol is the kernel's name followed by the
text of its compile-time arguments, cut at a fixed length, so two forms can share a symbol and a launch shape. That is
why the form is compared on the objects and the name only against the entry's own symbol. Registers per thread are
recorded in `--dump-census` and compared with nothing. Nothing here is a timing.

The names themselves come from the profiler, and the profiler names a kernel wrongly in one situation: after a kernel
module was unloaded in the same process, a kernel loaded where the unloaded one had been can be recorded under the
unloaded kernel's name. Two raw CUDA kernels loaded, launched and unloaded in turn were recorded under each other's name
on about every second launch, while each wrote its own output; with every module kept loaded, never (PyTorch 2.13.0,
CUDA 13.0, a compute capability 8.0 GPU; the fused step in its two forms showed it on compute capability 10.0). The
kernel that runs is the right one; the name of its record is not. A run of this sample unloads nothing: its entries live
until the process ends, so its names are right. A longer process that builds and frees entries before it runs the check
-- a test session that runs two stacks one after the other -- can fail the check by name alone. It errs to that side
only: the expected symbols are read from the entry object that was called, so a stale name cannot make another kernel
pass. The area's GPU test runs the `fe-entries` sample in a fresh interpreter for this reason.

Where the profiler's launch records cannot be had (no profiler library in the environment, another profiler attached, a
trace that cannot be exported or read, records without launch arguments), the line reads `NOT OBSERVED` with the reason
and the run ends in `[00] RAN fe-cublas stack (kernels not observed: <reason>)` with exit code 0: its other checks
held, the one that was asked for observed nothing. The reason names no path. A failed check is `[00] FAIL` either way.

The default run with `--check-kernels` was last run on compute capability 10.0 (2026-11-13, a 148-SM part, PyTorch
2.13.0, CuTe DSL 4.8.0): `[00] PASS` at 1 and at 4 sequences, with the ids hash, the replay and the batch invariance of
the run without the flag. A traced step of the two-layer model is 471 kernel launches and 40 copies there. Two are
judged one by one: the gather, and the fused step, which the entry states and the profiler recorded as grid (96, 1, 1),
block (256, 1, 1) and 42,112 B of shared memory at 1 row, and grid (48, 4, 1), block (256, 1, 1) and 74,848 B at 4
rows. The other 469 are attributed by family: 16 or 17 of them are cuBLAS's, by the length of the cache, the rest
PyTorch's. The 40 copies are the host read and copies inside the fallback rows. The check has not been run on
another GPU class, and not with a checkpoint (where a traced step of this stack has about twelve thousand launches
and the trace the check reads grows accordingly): a class or a model it has not run on is uncovered, not passing.

## Support per GPU class

`cudnn.GdnDecodeFused` serves compute capability 10.x and 12.x and declines elsewhere; the sample then prints
`[00] SKIP cudnn.GdnDecodeFused serves compute capability 10.x or 12.x; this device is <cc>` and exits 0. The entry is
validated on compute capability 10.0 and 12.0 (`docs/operations/linear_attention/gdn_decode_flashinfer_backend.md`).
`nvidia-smi --query-gpu=name,compute_cap --format=csv` prints the compute capability of a GPU.

| compute capability | the default run of this sample |
|---|---|
| 10.0 (data-center Blackwell, for example a GB200) | `[00] PASS`, verified at 1 and 4 sequences; batch invariance of the fused step gated and equal; the checkpoint run of the section above. With `--check-kernels`: `[00] PASS` at 1 and 4 sequences, the only class the kernel check has run on. The stacks `native` and `fe-entries`: `[00] PASS` at 1 and 4 sequences and with the checkpoint, batch invariance of the fused step and of the gate entry gated and equal; the only class these stacks have run on |
| 10.3 (for example a GB300) | `[00] PASS`, verified at 1 and 4 sequences; batch invariance of the fused step gated and equal; a checkpoint run before the one of the section above |
| 10.7 (a VR200-class part) | `[00] PASS`; batch invariance of the fused step is printed on this class (`REPORTED`), not gated: every block was equal in the runs so far |
| 12.0 (for example an RTX 5090) | `[00] PASS`, verified at 1 and 4 sequences; batch invariance of the fused step gated and equal. The checkpoint does not fit a 32 GB card: on one, the sample printed the memory sentence (about 59.7 GiB needed) and exited 0 |
| any other | `[00] SKIP`, by design. `--stack reference` runs the reduced model there on PyTorch alone, with the whole report, and ends in `[00] RAN reference stack (no provider checked)`: nothing is verified (seen on compute capability 8.0) |

A class the current checks have not run on is uncovered, not passing. That holds for the stacks `native` and
`fe-entries` on every class but 10.0: the gate entry inside this sample, the fused step in its wait + release form, and
the kernel patterns of the check have run on 10.0 only.

Memory of the default run (2 layers, 1 and 4 sequences): 1.7 GiB of bf16 copies; PyTorch held 2.5 GiB at its peak and
reserved 3.7 GiB on a compute capability 10.0 GPU, so the sample asks for about 3.7 GiB free there. On compute capability 12.x
it adds an allowance of 3 GiB and asks for about 6.7 GiB: the first launch of the fused step reserves device memory outside
PyTorch's pool on that class. The need is stated once, for every sequence count of the run, before anything is loaded: the
copies, the working memory and the first launch are counted once, the state per count. The run measures that launch and
prints what it took. Measured by this sample on a compute
capability 12.0 GPU of 32 GB: 3,124 MiB at the first launch of the 1-sequence form, nothing at the 4-sequence form, and
6.9 GiB of growth on the device over the whole run.
The four-layer run (`--layers 4`) holds 3.1 GiB of copies. The whole default run takes well under a minute on a data-center
Blackwell GPU: about 5 s to draw the weights, about a second to compile the fused step for the first sequence count (0.5
to 2.4 s in the runs so far) and 0.3 to 0.4 s for the next, the rest for the decode and its checks. These are orientation
values of single runs, not measurements of record.
