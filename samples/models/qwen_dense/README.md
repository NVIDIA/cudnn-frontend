# Dense Qwen 3.5-architecture model

Samples for the dense model of the Qwen 3.5 architecture at its published 27B-class layer widths: hidden 5,120; 64 layers,
three linear-attention (gated delta-rule, GDN) layers to one attention layer; linear attention with 16 key heads and 48
value heads of 128; attention with 24 query heads and 4 key/value heads of 256; MLP width 17,408; 8-bit projections, a
4-bit MLP and a 4-bit output head. `nvidia/Qwen3.8-27B-NVFP4` is a checkpoint of this architecture, and the one this
sample was last run with; the NVFP4 of its name is its 4-bit storage format (packed 4-bit codes with block scales and a
global scale). Area rules, the helper layer and the words a run uses: [../README.md](../README.md).

| script | what it runs | status |
|---|---|---|
| `00_decode.py` | plain greedy decode in one process: on the reduced seeded model, token ids in and out, self-checking; with `--model`, text in and text out on a checkpoint from local disk, from a prompt, chat messages or the request body a chat client sends, with the generated text on standard output | this page |

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
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --prompt "Why is the sky blue?" --max-tokens 256
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --messages chat.json --dump-tokens ids.json
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --messages request.json > answer.txt   # a client's request body
```

Where the output goes: the generated text of sequence 0 goes to standard output, as it is generated, and nothing else goes
there; every other line of a run -- the `[00]` lines, the tables, the report, the checks -- goes to standard error. So
`> answer.txt` keeps the text and `2> run.log` keeps the run. The seeded default run has no language and writes nothing to
standard output. A request gets one text: with a list of sequence counts (`--sequences 1,4`) every prompt is decoded once
per count, its text is written with the first count, and the `text out` line of a later count says whether sequence 0
produced the same text there.

| flag | default | meaning |
|---|---|---|
| `--model DIR` | -- | a checkpoint directory on local disk; never a hub name; nothing is downloaded |
| `--prompt TEXT` | `Why is the sky blue?` with `--model` | one user message, rendered by the checkpoint's chat template; repeat the flag for several prompts, run one after the other in one process |
| `--messages FILE` | -- | a chat request as JSON, from a file or, with `-`, from standard input (once per run): a list of messages; an object with `messages` and optionally `tools` (JSON tool schemas) and `chat_template_kwargs` (options the checkpoint's template understands); or the whole request body a chat client sends, every field of which is answered ("Coming from a serving engine" below). Repeatable |
| `--max-tokens N` | `256` | with `--model`: the limit of new tokens of a request that names none of its own: stop after N new tokens unless an end token comes first. `--max-new-tokens N` is the same flag. The limit a request body names rules over it, and a run says so. Without `--model` the flag is an error: the seeded run decodes `--force-tokens` |
| `--max-model-len N` | `2051` | the positions a prompt and its new tokens share. A limit of new tokens somebody asked for that does not fit beside the prompt, or a prompt that leaves no room, is an `ERROR` before anything is planned or loaded; the default limit of 256 is cut to the room that is left, and the run says so. The default is the context length the kernel census of this family was recorded with. The flag bounds a request; the state of a run is sized for the run itself |
| `--temperature 0` | -- | accepted at 0 and nothing else: token choice is greedy |
| `--force-tokens N` | `32` without `--model`, off with it | decode exactly N tokens per sequence, ignoring the end token; not together with `--max-tokens`; it rules over the limit a request body names, and a run says so |
| `--text-out FILE` | -- | with `--model`: the generated text of sequence 0 goes to FILE instead of standard output; the file then holds what standard output would have held: one text per prompt, in order, each ending in a line break. An empty name (`--text-out "$OUT"` with the variable unset) is an `ERROR` before anything is planned, as it is for `--dump-tokens` and `--dump-census` |
| `--no-stream` | off | with `--model`: the text is written once, after the report of its prompt, instead of piece by piece as it is generated |
| `--sequences N[,N...]` | `1,4` without `--model`, `1` with it | sequences per run, fixed before the first step; a list repeats the run per count; with `--model` every sequence of a run gets the same prompt, and the text of a prompt is written once, with the first count of the list (a later count reports whether its sequence 0 produced the same text; `--dump-tokens` has the ids of every count) |
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

## Coming from a serving engine

This sample is one process with greedy token choice, not a serving engine: no socket, no queue, no sampling. But the
inputs of a request are the same, and the sample takes them as they are.

| what you have | what the sample takes | note |
|---|---|---|
| the model | `--model DIR`, a directory on local disk | a hub name is not fetched |
| the request body your chat client sends | `--messages request.json`, or `--messages -` to read it from standard input | every field of the body is answered: the table below |
| `max_tokens` or `max_completion_tokens` in the request | read from the body; `--max-tokens N` is the limit of a request that names none | default 256. A serving engine that is given no limit generates until the context is full; this sample stops at 256 |
| the context length (`--max-model-len`) | `--max-model-len N`, default 2051 | a prompt and its new tokens share it; a request beyond it is an `ERROR` before anything is loaded |
| `temperature`, `top_p`, the checkpoint's generation settings | nothing: token choice is always greedy. `"temperature": 0` and `--temperature 0` are accepted | a serving engine may apply the checkpoint's own generation settings to a request that names no temperature. This sample never does, and every run with a checkpoint says so in one line (below) |
| `"stream": true` | the text of sequence 0 goes to standard output as it is generated; `--no-stream` writes it once; `--text-out FILE` writes it to a file | every other line of the run goes to standard error |
| concurrent requests | `--sequences N` | a fixed set of sequences that start together, each with the same prompt; nothing is admitted while a step runs |
| which kernels serve the model | `--stack`, and `--provider BLOCK=KEY` for one block | a kernel choice is a row of the table a run prints before it loads anything |
| a host, a port, an API key, a served model name, several GPUs, a memory budget | not here | one process on one GPU; the plan prints the memory a run needs before it loads anything |

A request body, with a system turn, the reasoning switched off and what a client adds on its own:

```json
{"model": "nvidia/Qwen3.8-27B-NVFP4",
 "messages": [{"role": "system", "content": "Answer in two short sentences."},
              {"role": "user", "content": "Why is the sky blue?"}],
 "stream": true, "max_completion_tokens": 96, "seed": 7, "n": 1, "top_p": 1,
 "chat_template_kwargs": {"enable_thinking": false}}
```

```bash
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --messages request.json > answer.txt
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --messages - < request.json 2> run.log
```

A run prints one line for the body, which answers every field of it:

```text
[00] --messages -: a chat request body. Read: messages (2), chat_template_kwargs (enable_thinking), max_completion_tokens 96. model ".../Qwen3.8-27B-NVFP4": names the checkpoint directory. Not used: stream true (the text goes where the command line says, as it is generated unless --no-stream); seed 7 (token choice is greedy, nothing is drawn). At the value that changes nothing: n 1, top_p 1.
```

Every field of a body falls into one of these classes (`serving_options.py`, beside the script, holds them as data):

| class | fields | what the sample does |
|---|---|---|
| read | `messages`, `tools`, `chat_template_kwargs`, `max_tokens`, `max_completion_tokens` | the messages, tool schemas and template options go to the checkpoint's chat template; the limit is the request's own limit of new tokens. Two limits that differ are an `ERROR` |
| stated | `model` | its last part is printed and compared with the name of the checkpoint directory; what stands in front of it -- an organisation, or a path where a serving engine served a directory -- is not printed (`".../Qwen3.8-27B-NVFP4"`). Another name is not an error: the run decodes the directory it was given, and says so |
| not used | `seed`, `stream`, `stream_options`, `user`, `metadata`, `store`, `service_tier`, `parallel_tool_calls` | named with the reason: nothing is drawn under greedy decoding, the text goes where the command line says, one process has no account, no queue and nothing to store, and tool calls in the output are not parsed. The value is printed for `seed` and `stream` only: what somebody wrote into the other fields is not printed back |
| accepted at the value that changes nothing | the table below | accepted at that value and at `null`; at any other value the run ends in one `ERROR` line before anything is loaded, which names the field, its value, the reason and what is accepted |
| not known to this sample | any other field | printed by its name and ignored: the sample cannot say whether such a field would change the output. A name that is not a plain word is printed as a JSON string |

The fields that would change the output, and the values they are accepted at:

| field | accepted | another value is refused because |
|---|---|---|
| `temperature` | `0`, `null` | token choice is greedy: the token of the largest logit at every step |
| `top_p` | `1`, `null` | it shapes the distribution a token is drawn from, and nothing is drawn here: token choice is greedy |
| `top_k` | `0`, `-1`, `null` | it shapes the distribution a token is drawn from, and nothing is drawn here: token choice is greedy |
| `min_p` | `0`, `null` | it shapes the distribution a token is drawn from, and nothing is drawn here: token choice is greedy |
| `n` | `1`, `null` | a request gets one text; --sequences N decodes the same prompt in N sequences |
| `best_of` | `1`, `null` | a request gets one text: nothing more is generated to choose among |
| `use_beam_search` | `false`, `null` | there is no beam search: one sequence per prompt, greedy |
| `length_penalty` | `1`, `null` | there is no beam search: one sequence per prompt, greedy |
| `frequency_penalty` | `0`, `null` | no penalty is applied to the logits |
| `presence_penalty` | `0`, `null` | no penalty is applied to the logits |
| `repetition_penalty` | `1`, `null` | no penalty is applied to the logits |
| `logit_bias` | `{}`, `null` | no bias is applied to the logits |
| `logits_processors` | `[]`, `null` | nothing processes the logits: the token of the largest one is taken, over the whole vocabulary |
| `bad_words` | `[]`, `null` | nothing processes the logits: the token of the largest one is taken, over the whole vocabulary |
| `allowed_token_ids` | `null` | nothing processes the logits: the token of the largest one is taken, over the whole vocabulary |
| `watermarking` | `false`, `null` | nothing processes the logits: the token of the largest one is taken, over the whole vocabulary |
| `stop` | `[]`, `null` | stop strings are not served: a sequence ends at an end token of the checkpoint or at its limit of new tokens |
| `include_stop_str_in_output` | `false`, `null` | stop strings are not served: a sequence ends at an end token of the checkpoint or at its limit of new tokens |
| `stop_token_ids` | `[]`, `null` | a sequence ends at the end tokens of the checkpoint and at no other |
| `ignore_eos` | `false`, `null` | an end token ends a sequence; --force-tokens N decodes exactly N tokens instead |
| `min_tokens` | `0`, `null` | an end token ends a sequence whenever it comes |
| `repetition_detection` | `null` | a sequence ends at an end token of the checkpoint or at its limit of new tokens, whatever it repeats |
| `thinking_token_budget` | `null` | reasoning a model writes before its answer is part of the text, and no budget is put on it |
| `include_reasoning` | `true`, `null` | reasoning a model writes before its answer is part of the text: the output is not parsed |
| `logprobs` | `false`, `null` | a run writes text and, with --dump-tokens, token ids; no log-probabilities |
| `top_logprobs` | `0`, `null` | a run writes text and, with --dump-tokens, token ids; no log-probabilities |
| `prompt_logprobs` | `null` | a run writes text and, with --dump-tokens, token ids; no log-probabilities |
| `logprob_token_ids` | `null` | a run writes text and, with --dump-tokens, token ids; no log-probabilities |
| `echo` | `false`, `null` | the prompt is not written back with the text |
| `skip_special_tokens` | `true`, `null` | special tokens are left out of the text |
| `spaces_between_special_tokens` | `true`, `null` | special tokens are left out of the text |
| `response_format` | `{"type": "text"}`, `null` | constrained output is not served: the text is what the greedy tokens decode to |
| `structured_outputs` | `null` | constrained output is not served: the text is what the greedy tokens decode to |
| `structural_tag` | `null` | constrained output is not served: the text is what the greedy tokens decode to |
| `guided_json` | `null` | constrained output is not served: the text is what the greedy tokens decode to |
| `guided_regex` | `null` | constrained output is not served: the text is what the greedy tokens decode to |
| `guided_choice` | `null` | constrained output is not served: the text is what the greedy tokens decode to |
| `guided_grammar` | `null` | constrained output is not served: the text is what the greedy tokens decode to |
| `tool_choice` | `"auto"`, `null` | tool schemas are rendered into the prompt by the checkpoint's chat template; the output is neither constrained nor parsed |
| `function_call` | `null` | it is the older name of tool_choice, and the output is neither constrained nor parsed |
| `functions` | `null` | it is the older name of tools, the field tool schemas are read from |
| `add_generation_prompt` | `true`, `null` | the prompt ends where the checkpoint's chat template opens the reply |
| `continue_final_message` | `false`, `null` | the prompt ends where the checkpoint's chat template opens the reply |
| `add_special_tokens` | `false`, `null` | the prompt is what the checkpoint's chat template renders, encoded as it is: no special token is added |
| `chat_template` | `null` | the messages are rendered by the checkpoint's own chat template |
| `documents` | `[]`, `null` | the checkpoint's chat template is given messages and tool schemas, nothing else |
| `truncate_prompt_tokens` | `null` | a prompt is encoded whole |
| `reasoning_effort` | `null` | an option of the checkpoint's chat template goes into chat_template_kwargs |
| `extra_body` | `{}`, `null` | it is how a client library is handed the fields it adds to a request, not a field of the body it sends: its fields go to the top level of the body |

The table holds the fields of the public chat request and of the sampling parameters serving engines add to it that ask
for something this sample leaves out: sampling, stop strings, constrained output, a processor of the logits, parsing of
the output, another prompt than the checkpoint's chat template renders. A field under a name the table does not hold
is in the last class above: printed and ignored. `extra_body` is in the table because a request file written from
client code can carry it: a client library merges what it is handed under that name into the top level of the body
it sends, so a body that still holds the name would be decoded without the fields inside it.

So a body that asks for sampling ends like this, before the plan:

```text
[00] ERROR --messages request.json: "temperature": 0.6 would change the output and is not served: token choice is greedy: the token of the largest logit at every step. Accepted: 0, null, or the field left out
```

Which limit of new tokens applies to a request: `--force-tokens N` where it is given; else the limit the request body
names; else `--max-tokens N`; else 256. Each request of a run has its own. A limit that was given and does not apply is
said in a line (`its own "max_tokens": 64 applies, not --max-tokens 32: the flag is the limit of a request that names
none`).

Two lines of a run with a checkpoint are there for a reader who comes from a serving engine. Both are printed before
anything is loaded:

```text
[00] greedy decoding: the token of the largest logit at every step; the checkpoint's own generation settings (do_sample true, temperature 1.0, top_k 20, top_p 0.95 in generation_config.json) are not applied
[00] first token: a prompt goes through the decode step, one step per prompt token (no stack of this sample has a prefill kernel): 58 decode steps before the first new token of the default prompt, as long as 58 new tokens take. No text is written until then; the first token line of the report has the time
```

The first states the decoding rule beside what the checkpoint's `generation_config.json` names, so the difference
from an engine that applies those settings is never silent. The second states what a prompt costs here: no stack of
this sample prefills a prompt with a kernel of its own, so a prompt of N tokens is N decode steps and takes as long as
N new tokens before the first new token appears. It gives a count and no time, because before the load a run has no
time of its own to give; the `first token` line of the report has the measured one. In the runs of 2026-11-17 on
compute capability 10.0, a prompt of 29 tokens took 4.3 s before the first new token on the stack `fe-cublas` (0.15 s
per step) and the 58 tokens of the default prompt took 7.8 s on the stack `native` (0.13 s per step): orientation
values of single runs on a shared GPU, as every timing on this page.

What is not here, and is not planned for a sample: a network interface of any kind; a request queue, or admission of
work while a step runs; different prompts in one step; sampling; stop strings, constrained output, logit processors;
parsing of tool calls in the output (tool schemas are rendered into the prompt, the output is text); a download from a
hub; an automatic choice of the stack.

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
class, SM count and row count the launch shape that was recorded for it, with the layer widths of the model it was
recorded on. For the plain step of the 64-layer model that is 484 launches. A statement about this stack is three
numbers, never one: of those 484, 210 are the target's own compiled kernels (the fused linear-attention step of cuDNN
Frontend, library calls of FlashInfer and of cuBLASLt, a Triton kernel, PyTorch's gather), 273 are launches of kernels
whose home is cuDNN Frontend -- for each of them an entry that reproduces the target kernel's results byte for byte,
which is why a run counts them as "cuDNN Frontend entries with byte-equal results" -- and 1 is an operation that lives
inside a serving engine and gets a declared stand-in.

These are the numbers of the complete stack, not of today's. On compute capability 10.0 this stack launches all 273
today: the norm entry's 129, the projection entry's 64, the fused gate / up entry's 64 and the gate entry's 16. The
numbers are those of the recorded class, compute capability 10.0: the target launches the
projection kernel for the in-projections there and a library multiply on the other 10.x classes, so on those the
in-projection of this stack stays on its fallback row. And "byte-equal" is the entry's own record, not something this
sample measures: the comparison of an
entry's bytes with the target kernel's is made when the entry is written and needs the target's kernel. The census
file says for each entry a stack calls where that comparison stands and on which GPU class it was made (`tie_out`,
`differences`); for the gate entry it was made on compute capability 10.3, and none is recorded for 10.0; for the norm
entry it was made on compute capability 10.0; for the projection entry it was made on compute capability 10.0; for the
fused gate / up entry it was made on compute capability 10.0. What this
sample holds an entry to is its documented numerics contract (the band of its row) and its launch facts.

The stack is built block by block. A block whose target kernel is not served here yet stays on the fallback row it has in
`fe-cublas`, and a run prints it as `not yet (<fallback reason>)`. Today five entries are on their target forms:

| block | launches per step (64 layers) | what `native` launches | identity |
|---|---|---|---|
| token embedding lookup | 1 | PyTorch's gather | target: the target's own compiled kernel |
| residual add + RMSNorm + quantise | 129 | `cudnn.AddNormQuantDecode`, waiting for the launch before it and releasing the launch after it (`PDL True`, `PDL_RELEASE True`), in one of five forms by call site (the table below); compute capability 10.x | target: a cuDNN Frontend entry with byte-equal results |
| 8-bit in-projection | 64 | `cudnn.Fp8ProjectionDecode` on the stored 8-bit weight, the kernel stated whole: one K slice per weight tile of 128 rows, a ring of 8 stages, the weight stream ahead of the wait, 32 tensor-memory columns released late (the entry's own two fields: the target kernel allocates 256 and releases them early), waiting for the launch before it and releasing the launch after it (`SPLIT_K 1`, `MMA_M 128`, `STAGES 8`, `TMEM_COLS 32`, `PREWAIT_WEIGHTS True`, `HOLD_TMEM True`, `PDL True`, `PDL_RELEASE True`), at one of two weight shapes by call site (the table below); compute capability 10.0, 1 to 32 rows | target: a cuDNN Frontend entry with byte-equal results |
| linear-attention step | 48 | `cudnn.GdnDecodeFused` launched as a dependent of the launch before it and releasing the launch after it at entry (`PDL True`, `PDL_RELEASE True`) | target: the target's own compiled kernel |
| MLP gate / up + SiLU x up + quantise | 64 | `cudnn.SwigluFp4Decode` on the stored 4-bit weight in the entry's own row order, the kernel stated whole: the token tile of the row count (8 tokens up to 8 rows), two weight stages issued ahead of the wait, the weight stream on the evict-first L2 policy, a persistent grid of at most 148 thread blocks, waiting for the launch before it (`TOKEN_TILE 8`, `PREWAIT_STAGES 2`, `L2_EVICT_FIRST True`, `MAX_CLUSTERS 148`, `PDL True`); compute capability 10.x, 1 to 64 rows | target: a cuDNN Frontend entry with byte-equal results |
| gate + quantise | 16 | `cudnn.AttentionGateQuantDecode` with 256 threads per thread block, releasing the launch after it at entry (`THREADS 256`, `PDL_RELEASE True`); compute capability 10.x | target: a cuDNN Frontend entry with byte-equal results |
| every other block of the table below | 161 | the fallback rows of `fe-cublas` | not yet |
| the 4-bit quantise in front of the output head | 1 | nothing: this sample has no block for it yet | the target's declared stand-in, not yet |

The norm block is called at five kinds of site in a step, and the target calls its kernel in another form at each: the
first call of a step starts the residual stream from its operand, every later call adds to it in place; and each call
writes what the block after it reads. The provider row names the form of every call site (`NORM_FORMS` in
`model_samples/qwen_dense/configs.py`), the model's wiring names the site of every call (`QwenDense.sites()`), the
census names the form of every call, and a host test holds the three to each other. The in-projection is called in one
form at two weight shapes, and the same three places name the shape of each call (`PROJECTION_SITES`):

| block | call site | calls per step (64 layers) | form of the entry there |
|---|---|---|---|
| residual add + RMSNorm + quantise | `linear_attention.first` | 1 | [ADD False, STORE_RESIDUAL True, BF16_OUT True, QUANT e4m3_static] |
| residual add + RMSNorm + quantise | `full_attention.first` | 0 | [ADD False, STORE_RESIDUAL True, BF16_OUT False, QUANT e4m3_static] |
| residual add + RMSNorm + quantise | `linear_attention` | 47 | [ADD True, STORE_RESIDUAL False, BF16_OUT True, QUANT e4m3_static] |
| residual add + RMSNorm + quantise | `full_attention` | 16 | [ADD True, STORE_RESIDUAL False, BF16_OUT False, QUANT e4m3_static] |
| residual add + RMSNorm + quantise | `mlp` | 64 | [ADD True, STORE_RESIDUAL False, BF16_OUT False, QUANT nvfp4] |
| residual add + RMSNorm + quantise | `output_head` | 1 | [ADD True, STORE_RESIDUAL False, BF16_OUT True, QUANT None] |
| 8-bit in-projection (linear attention QKVZ / attention QKV) | `linear_attention.qkvz` | 48 | [OUT_FEATURES 16384, IN_FEATURES 5120] |
| 8-bit in-projection (linear attention QKVZ / attention QKV) | `full_attention.qkv` | 16 | [OUT_FEATURES 14336, IN_FEATURES 5120] |

A call site is named by what the normed rows feed: a layer's mixer (the layer's kind; `.first` for the first call of a
step, which has no residual stream yet), its MLP, the output head. A linear-attention step reads the normed bf16 rows
beside the 8-bit operand of its in-projection, an attention layer reads the 8-bit operand alone, an MLP reads 4-bit
codes, and the final norm writes bf16 rows only: the target quantises the output head's operand with a launch of its
own. Each form is one compiled kernel of the entry and one entry instance of the adapter, which serves a call with the
instance of its site and decides nothing. The forms cannot be told apart by a launch record -- one kernel id, grid
(1, 1, 1), block (640, 1, 1) and 80 B of shared memory at 1 row in every form -- so the kernel check holds them where
they do show: the form of each site against the entry instance's own form, the instance that served each call of the
traced steps (the call log), and the kernel's symbol, which carries the form's text. A run prints which instance served
how many calls of a step: 1, 47, 16, 64 and 1 on the 64-layer model.

The sixth form of the table, `full_attention.first`, is the first call of a step whose first layer is an attention
layer. The published model starts with a linear-attention layer, so a step of it never calls that form (0 above) and
the recorded target has no launch in it: the census derives the form, the one of an attention layer's input norm
with the residual mode of a first call (`first_layer`), and such a call counts as `target` by that rule, not by a
record. A reduced model that starts with an attention layer calls it -- `--layers 1`, 5, 9 and so on, because the
reduced model keeps the last layers of the published order -- and the run table below has that run.

The entry serves one hidden width, the published model's 5120, and the row says so (`widths` in `configs.py`). A
checkpoint of the family with another hidden width is not this row's: its norm block stays on the reference chain in
both stacks, on every GPU class, and its run states no kernel identity (below).

The in-projection of a layer multiplies the 8-bit operand its input norm wrote by the layer's merged 8-bit weight:
`qkv` and `z` of a linear-attention layer (16384 x 5120), the query with its gate half, the key and the value
projection of an attention layer (14336 x 5120). `cudnn.Fp8ProjectionDecode` is one weight shape per entry instance, so
the row names the shape of each of its two call sites and the adapter holds one instance per shape; the kernel's form
is the same at both and is stated whole in the row, so the entry never runs on its own pick. Unlike the norm's forms
the two shapes do show in a launch record: one thread block per 128 weight rows, grid (128, 1, 1) and (112, 1, 1),
block (256, 1, 1), 139,520 B of shared memory at 1 row. What a record does not show -- the ring depth is in it only
through the shared memory, the tensor-memory columns and their release, the wait and the release not at all -- the
kernel check holds on the objects and in the kernel's symbol, which carries the rendering
(`..._n16384k5120s1r1_m128g8t8c32_w1h1p1q1_...`). A run prints which instance served how many calls of a step: 48 and
16 on the 64-layer model.

Two fields of that form are the entry's own, and their values are not the target kernel's: the tensor-memory columns
and their release (`TMEM_COLS 32`, `HOLD_TMEM True`) are at the values the entry's byte comparison with the target
kernel ran, while the target kernel allocates 256 columns and releases them right after the accumulator read. The
census lists both under `differences`; both orders write the same bytes.

The kernel multiplies on the stored format, so the loader keeps the merged 8-bit codes of the in-projections for a plan
with this row (one byte per weight element: 4.8 GiB on the published model). It keeps the bf16 copy of a weight only
where a block of the run reads one: a run in which every sequence count is this row's holds no copy of the
in-projections (9.7 GiB on the published model), and a run with a sequence count on the fallback row, or with the row
selected away, holds both. The reference blocks of the checks read copies; for these weights the checks make theirs for
one call at a time ("What a run holds of a weight" below).
The scale the kernel reads is the fp32 product of the operand's static scale and the weight's scale, one element built
on the host once per layer. The row claims compute capability 10.0 and 1 to 32 rows -- where the target launches this
kernel for the in-projections; it has run at 1, 4, 9, 16, 17 and 32 rows (the run table below). On every other class and from 33 rows the block stays on its multiply over a copy
(`not yet (bf16 copy of 8-bit weights)`), as it does for a checkpoint of the family whose layer widths give other
weight shapes, and with a weight merged at one scale per row (`merge_rule="own_scale"` of a session) the plan declines
the row by name: the entry reads one scale per weight. The 8-bit out-projection is not this row's: the target runs a
library multiply for it in a plain step.

The MLP's first multiply is one launch in the target and one call of `cudnn.SwigluFp4Decode` here: the kernel
multiplies the 4-bit operand the MLP's norm wrote by the merged gate / up weight (34816 x 5120) on its stored 4-bit
codes, forms SiLU of the gate half times the up half and quantises the 17,408 values of a row to 4-bit codes with
their block scales, where the fallback row is a multiply over a copy followed by a reference chain. The row states the
kernel whole, so the entry never runs on its own pick. Its token tile follows the row count, which makes it four rows
of one key by row range (the sentence under the table of this stack names them); the other four fields are the same in
all four. A launch record shows the row: one launch per call, grid (1, 1, 148), block (192, 1, 1) and 231,424 B of
shared memory in the token tiles of 8, 16 and 32 rows, 230,400 B in the one of 64. The token tile shows in the
kernel's symbol (`..._h5120_i17408_t8_pw2_evf_c148_pdl_...`) and in the form, compared on the objects.

The kernel reads its weight in tiles of 128 rows -- 64 gate channels, then the up channels of the same 64 -- with the
block scales of those rows in the 128 x 4 layout, and a checkpoint stores the gate rows and the up rows as two tensors.
That layout is the provider's, not the loader's: the adapter says how the merged stored tensor is laid out
(`stored_layout`, which calls the entry's own helper `interleave_gate_up`: a permutation of bytes, nothing is
requantised), a session collects the layouts of its plans, and the load keeps the weight once, in that order, and no
bf16 copy of it: 6.0 GiB on the published model where the copies are 21.2 GiB. Two plans of one run that multiply on
one weight in two row orders -- the layouts of two providers, or a provider's layout in one plan and the checkpoint's
row order in another -- are a wrong table, an error with the plans of the run, before anything is built or read. The
reference blocks of the checks read copies in the checkpoint's row order, made for one call from the interleaved
bytes ("What a run holds of a weight" below).

The operand comes from the norm kernel as packed codes with their block scales in the 128 x 4 layout, and the entry
reads both as they are. The scalar it multiplies each sum by is the fp32 product of the operand's global scale and the
weight's, and the global scale it quantises with is the fp32 reciprocal of the scale its consumer reads the codes
with: one element each, built on the host once per value. A 4-bit operand that does not come from a kernel -- the norm row selected away with `--provider`, or one
row of a batch in a check -- has its block scales row by row, and the adapter brings them into the layout itself: a
launch of this block, never of its producer. The row claims compute capability 10.x, 1 to 64 rows and the published
widths; it has run at 1, 4, 8, 9, 16, 17, 32, 33 and 64 rows on compute capability 10.0 (the run table below). On
compute capability 12.x, and for a checkpoint of the family with another hidden or MLP width, the block stays on its
multiply over a copy (`not yet (bf16 copy of 4-bit weights)`).

What a consumer needs beyond a kernel's bytes is launched by the consumer, never inside the producer's call. The MLP's
down multiply is still on its fallback row: it reads the codes of the gate / up kernel with their block scales in the
layout the kernel writes (128 x 4), and the multiply over a copy gathers them when it dequantises. And where the norm's
form ends at the bf16 rows, the output head on its fallback row quantises them itself: that quantisation is the
target's declared stand-in, for which this sample has no block yet.

So a run with a checkpoint prints under its kernel table:

```text
the whole target kernel set of a step of this model (what a complete stack launches; this stack's share is the next line): 484 launches per step = 210 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results + 1 declared stand-in
kernel identity, as planned: 322 of 484 launches of a step are target kernels (49 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results); not yet: projection.fp8.out 64, mlp.down 64, attention.prepare 16, attention.core 16, output_head 1; 1 declared stand-in (not yet: output_head.quantize)
```

and with its checks the census line, `target census plain / cc 10.0 / 148 SMs (recorded 2026-11-12): NOT CLAIMED (fallback
rows present; ...); 322 launches of a step compared by launch shape (49 the target's own compiled kernels + 273 cuDNN
Frontend entries with byte-equal results + 0 declared stand-ins): each in the recorded one`. The default run on the
two-layer model counts 12 of 20 (2 the target's own compiled kernels + 10 cuDNN Frontend entries with byte-equal results;
the declared stand-in not yet). The line always says what was compared: `EQUAL`, once a class is claimed, carries the
number of launches that were held to a recorded launch shape and the number held by membership alone (a kernel a
library's tuner selects, a stand-in), so it never stands for a comparison that was not made.

**Nothing is claimed yet.** A GPU class is claimed only when its section of the census says so, and that needs every
block of a step on its target kernel, a census recorded on that compute capability and SM count, and the package versions
of the record. Until then the line reads `NOT CLAIMED` with every reason that applies: a row selected with `--provider`
(`not the target set: <block> selected by --provider`), fallback rows present, a kernel of the target the plan has no
block for, no recorded section for this compute capability and SM count (the line names both) or for this row count,
versions other than the recorded ones. The census file has one section today: compute capability 10.0 with 148 SMs, at
1 row, for the layer widths of the published 27B-class model, recorded on 2026-11-12. On another GPU class, another SM
count, another sequence count or a checkpoint of the family with other layer widths a run says `kernel identity: not
stated for this run` and names what differs: the kernels of the target are recorded per class, row count and model, and
are never copied from one to another. The reduced default model keeps the published widths, so the section holds for it;
the launch shape of the output head is recorded with the vocabulary as well and will hold for that vocabulary only.

**What is checked already**, in every run of this stack (its kernel check cannot be switched off), for every launch on a
target kernel: the provider row's form is the census's form of that kernel and the entry's own form, compared on the
objects; the launch carries the entry's own symbol and exactly the launch shape the entry states; and where the census
has a section for the run, the launch shape is the recorded one -- grid (96, 1, 1), block (256, 1, 1) and 42,112 B of
shared memory for the fused step, grid (3, 1, 1), block (256, 1, 1) and no shared memory for the gate, grid (1, 1, 1),
block (640, 1, 1) and 80 B for the norm in every form, grid (128, 1, 1) or (112, 1, 1) by the weight's shape, block
(256, 1, 1) and 139,520 B for the in-projection, grid (1, 1, 148), block (192, 1, 1) and 231,424 B for the fused
gate / up step, at 1 row on
compute capability 10.0 with 148 SMs. For the norm block and the in-projection all of it holds per call site. A
difference fails the run with a sentence that names the block, and for a form the call and its site.

**What this stack is not, today.** It is not a fast path: every block but its target kernels reads the bf16 copies
`fe-cublas` reads, and it issues every launch from Python. No timing of it means anything. And most target kernels do not run in the target's surroundings yet: the
fused step, the norm and the in-projection wait for the launch before them and release the launch after them, the
fused gate / up step waits for the launch before it, the
gate releases the launch after it, but around a target kernel that sits between fallback rows the launches are
PyTorch's and cuBLAS's, which neither release early nor start early, so there both knobs are exercised and change
nothing. Target kernels meet at the gather and the first norm of a step and, on compute capability 10.0, in the chain
of a layer's input: the norm releases at its own wait, the in-projection behind it starts on that release, streams
its weight and waits for the norm before it reads the 8-bit operand; in a linear-attention layer the fused step
starts on the in-projection's release and waits before it reads the projection's rows. They meet in every MLP as
well, on each 10.x class: the MLP's norm releases at its own wait, the gate / up kernel behind it starts on that
release, issues two stages of its weight and waits for the norm before it reads the 4-bit operand, its block scales
and the two scalars. The kernel signals the launch after it before its last output tile is written (it has no knob
for that); the down multiply behind it is a fallback row, launched without the attribute, and starts when the kernel
has completed. Each
row of `configs.py` says what the launch before it must not write and what the launch after it must wait for. With `--provider
linear_attention.step=fe.gdn_decode_fused` the stack runs the fused step in its `fe-cublas` form instead (no wait, no
release); the table then prints `not the target (...)` for the block and the census line says why the run is not the
target set.

`--stack fe-entries` is every cuDNN Frontend row of `native` with each block that a library or a temporary kernel will
serve there on its fallback row. It needs no package beyond cuDNN Frontend and PyTorch, and it is the stack the GPU test
of this area runs, so the rows of the entries stay verified wherever that test runs. Until `native` has a row a library
serves, the two stacks are the same rows.

On compute capability 12.x neither the gate entry nor the norm entry serves; both blocks then stay on their reference
chains (`not yet (reference chain)`) in both stacks, and the census has no section for that class. The row of the
projection entry claims compute capability 10.0 only, the rows of the fused gate / up entry compute capability 10.x:
on 12.x both multiplies are over copies.

The stack was last run on compute capability 10.0 (2026-11-17, a 148-SM part, PyTorch 2.13.0, CuTe DSL 4.8.0; the census
line of every run names the PyTorch version as other than the recorded 2.14.0). Where every sequence count of a run is
at most 32 and no row is selected away, the projection row and the gate / up row serve each count: the run holds the
in-projections and the gate / up weights on their stored codes alone, and the reference of its checks reads bf16
copies made on demand. From 33 sequences the in-projections have their copies, and from 65 the gate / up weights:

| run | result |
|---|---|
| the default run, `--stack native` | `[00] PASS` at 1 and at 4 sequences, with the ids hash of the default stack. A traced step of the two-layer model is 179 kernel launches and 13 copies (303 and 25 with the gate / up block on its fallback row); 12 are judged one by one. At 1 sequence the gather, the fused step, the gate, the five norm calls, the two in-projections and the two gate / up calls (2 the target's own compiled kernels + 10 cuDNN Frontend entries with byte-equal results) have the recorded launch shapes; at 4 no identity is stated. The norm entry was inside its band in the four forms this model calls -- over 285 calls at 1 sequence 583,680 normed elements, 4 of them on the other bf16 value around the normalised value, and 1,203,840 codes and block scales, each that of the nearest value; nothing outside -- and equal alone and among 4. The projection entry was inside the band of the reference multiply at both weight shapes (114 calls; the largest distance one bf16 unit, on fewer than 0.05 % of the elements, where the multiply over a copy it replaces is one unit off on 25.9 %), and equal alone and among 4. The fused gate / up entry was inside its band over 114 calls: at 1 sequence 124,032 block scales, each the byte of the fp64 chain, and 1,984,512 codes, 1,927 of them another code than the fp64 chain's, the farthest at 89 % of its bound; at 4 sequences 7,758 of 7,938,048 codes; nothing outside, and equal alone and among 4. The gate entry was inside its band (350,208 bytes at 1 sequence, each the code of the nearest bf16 value; 1,400,832 at 4, one of them the code of the other value around the gated value, inside the margin), and equal alone and among 4. At each count the reference read 460 copies of in-projection and gate / up weights made on demand (four per step and reader, 0.33 GiB the larger), and the four weights held the bytes of their load before and after the checks (`stored weights`) |
| the four-layer model, `--stack native --layers 4 --sequences 1` | `[00] PASS`: all five forms a 64-layer model calls (the input norm of a later linear-attention layer is the fifth), four in-projections, three at 16384 x 5120 from one instance and one at 14336 x 5120 from the other, and four gate / up calls; 22 launches of a step judged one by one (4 the target's own compiled kernels + 18 cuDNN Frontend entries with byte-equal results), each in the recorded launch shape. Its memory sentence asked for about 4.4 GiB; PyTorch reserved 4.4 GiB and the device grew by 4.6 GiB -- on the reduced models the sentence is an estimate (the last section has the runs) |
| the one-layer model, `--stack native --layers 1 --sequences 1` | `[00] PASS`: the sixth form, `full_attention.first` -- the model is one attention layer, so the first norm of a step is its input norm. 7 launches of a step judged one by one (the gather, three norm calls, the in-projection at the attention layer's weight shape, the gate / up call and the gate: 1 the target's own compiled kernel + 6 cuDNN Frontend entries with byte-equal results), each in the recorded launch shape; the three norm calls served by three instances, and the symbol of the first carries its form (`store`, `e4m3_static`, wait + release). The norm entry inside its band over 171 calls: 291,840 normed elements, 4 of them on the other bf16 value around the normalised value, and 601,920 codes and block scales, all those of the nearest value; the projection entry inside its band over 57 calls; the gate / up entry inside its band over 57 calls |
| the checkpoint, one sequence, the default prompt, 32 new tokens | `[00] PASS`: 322 launches of every one of the 32 traced steps on target kernels (49 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results; the declared stand-in has no block yet), in the wiring order, each in the recorded launch shape; 2,908 kernel launches and 200 copies per traced step in all (6,876 and 584 with the gate / up block on its fallback row). The call log of a step: the five forms of the norm entry served 1, 47, 16, 64 and 1 calls, each from its own instance, and the two instances of the projection entry 48 and 16. Replay held; 8 of 8 provider blocks inside their bands; the norm entry over 2,064 calls with 4,014,080 normed elements, 40 of them on the other bf16 value, and 10,813,440 codes and block scales, 3 of them not those of the nearest value, nothing outside; the projection entry over 1,024 calls, the largest distance one bf16 unit on fewer than 0.05 % of the elements; the fused gate / up entry over 1,024 calls with 1,114,112 block scales, each the byte of the fp64 chain, and 17,825,792 codes, 12,528 of them another code than the fp64 chain's, the farthest at 96 % of its bound, nothing outside; the gate entry over 256 calls with 1 of its 1,572,864 bytes the code of the other bf16 value around the gated value, inside the margin; the token line reported. The reference read 6,400 copies of in-projection and gate / up weights made on demand, and the 128 weights held the bytes of their load before and after the checks; the memory lines are in the section on checkpoints |
| the default run, `--stack fe-entries` | `[00] PASS`, the same lines; also as the L0 cell of the area's GPU test |
| `--stack native --provider linear_attention.step=fe.gdn_decode_fused`, 1 sequence | `[00] PASS`; `NOT CLAIMED (not the target set: linear_attention.step selected by --provider; ...)`; the 11 remaining target launches (1 the target's own compiled kernel + 10 cuDNN Frontend entries with byte-equal results) in the recorded shape |
| `--stack native --provider projection.fp8.in=cublas.bf16_dequantized`, 1 sequence | `[00] PASS`: the in-projection back on its multiply over a copy, so its weights have their copies and the reference's copies made on demand are those of the gate / up weights alone; `NOT CLAIMED (not the target set: projection.fp8.in selected by --provider; ...)`; the 10 remaining target launches (2 the target's own compiled kernels + 8 cuDNN Frontend entries with byte-equal results) in the recorded shape |
| `--stack native --provider mlp.gate_up=cublas.bf16_dequantized+torch_ref.silu_mul_quantize`, 1 sequence | `[00] PASS`: the gate / up block back on its multiply over a copy (303 launches a step again), so its weights have their copies and only the in-projections are made on demand; `NOT CLAIMED (not the target set: mlp.gate_up selected by --provider; ...)`; the 10 remaining target launches (2 the target's own compiled kernels + 8 cuDNN Frontend entries with byte-equal results) in the recorded shape |
| `--stack native --provider add_norm_quantize=torch_ref.add_norm_quantize` | `[00] PASS` at 1 and at 4 sequences: the norm block on its reference chain, so the gate / up entry is handed block scales row by row and brings them into its kernel's layout itself; inside its band at both counts and equal alone and among 4; `NOT CLAIMED (not the target set: add_norm_quantize selected by --provider; ...)`; at 1 sequence the 7 remaining target launches (2 the target's own compiled kernels + 5 cuDNN Frontend entries with byte-equal results) in the recorded shape |
| `--stack native --sequences 8,9` | `[00] PASS` at 8 and at 9 sequences: the gate / up entry in its 8-row token tile at 8 and in its 16-row tile at 9, in the same launch shape (the tile shows in the form and in the symbol, `_t8_` and `_t16_`); the projection entry in its 16-row token tile at 9 (147,712 B of shared memory per thread block); both inside their bands, and each sequence alone equal to itself among 8 and among 9; no identity is stated at these row counts |
| `--stack native --sequences 16,17` | `[00] PASS` at 16 and at 17 sequences: the gate / up entry in its 16-row and in its 32-row token tile, the projection entry in its 16-row and in its 32-row token tile (164,096 B of shared memory per thread block, the grids as at 1 row), each launch in the shape the entry states; inside their bands, and each sequence alone equal to itself among 16 and among 17. From 17 rows the embedding's gather is PyTorch's vectorized gather, held by name like the kernel of fewer rows. A traced step is 178 kernel launches and 13 copies, 12 judged one by one |
| `--stack native --sequences 32,33` | `[00] PASS` at both counts. 32 is the last row count of the projection row; at 33 the in-projection is on its multiply over a copy, so the load of the run keeps both forms of the in-projections -- the codes for the 32 sequences, the copies for the 33 -- and the copies made on demand at either count are those of the gate / up weights alone. The gate / up entry in its 32-row token tile at 32 and in its 64-row tile at 33 (230,400 B of shared memory per thread block); at 33 a traced step is 186 kernel launches, 10 judged one by one |
| `--stack native --sequences 64` | `[00] PASS` at the last row count of the gate / up rows: the 64-row token tile, grid (1, 1, 148), block (192, 1, 1) and 230,400 B of shared memory, as the entry states; inside its band over 114 calls -- 7,938,048 block scales, 7 of them another byte than the fp64 chain's, each inside its interval, and 127,008,768 codes, 122,754 of them another code than the fp64 chain's, the farthest at 91 % of its bound -- and each sequence alone equal to itself among 64. The memory sentence of this run was short (the last section) |
| `--stack native --sequences 64,65` | `[00] PASS` at both counts: at 65 the gate / up block is on its multiply over a copy (`not yet (bf16 copy of 4-bit weights)`; 308 launches a step), so the load of the run keeps both forms of the gate / up weights and no copy is made on demand at either count |

The fused step has the same launch shape in both of its forms, grid (96, 1, 1), block (256, 1, 1), 42,112 B at 1 row:
that it waits and releases shows in its symbol and in its form, not in a launch record. With the norm entry on its
target forms the greedy tokens of a checkpoint run of this stack are not those of the run before it had the row: the
kernel rounds the normed value once from fp32 and breaks an exact 4-bit tie by its own reciprocal, and on a model with
4-bit activations that is another token after a few steps, as it is between two reference stacks (the section on
checkpoints). The projection entry accumulates in fp32 in a fixed order and rounds once, where the multiply over a
copy rounds its unscaled sum and the scaled result; on the checkpoint run with that row the 32 produced tokens were
those of the run before it. The fused gate / up entry rounds the gate and the up product once each from an fp32 sum
and forms SiLU with approximate instructions, where the fallback row rounds a multiply over a copy and every step of
its reference chain: with this row the produced tokens of the checkpoint run moved again (another ids hash), and over
the 16 checked steps 14 greedy tokens were the reference's -- as many as a second reference with the other
accumulation precision has. The stack has not run on another GPU class.

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
| a request body, the text on standard output | 2026-11-17, compute capability 10.0 (x86, a 148-SM part, PyTorch 2.13.0, CuTe DSL 4.8.0). On the stack `fe-cublas`, the request body of the section "Coming from a serving engine" on standard input, with standard output and standard error on two pipes: `[00] PASS`; the answer arrived on standard output in 39 pieces, one per decode step, the first after the 29 steps of the prompt, and its line break after the last, before the first line of the report; standard output held the text and that line break, standard error every line of the run. On the stack `native`, the default prompt as a request body with its own limit of 32 new tokens, written with `--text-out` and `--no-stream`: `[00] PASS` with the ids hash and the kernel lines of the same prompt given with `--prompt`, the text in the file. On the stack `fe-cublas` again, the same request body at two sequence counts (`--sequences 1,2`): `[00] PASS`; standard output held the one text, written with the first count, and the second count reported the same text for its sequence 0 |

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
with exit code 0 before a tensor is read. As part of the plan every cuDNN Frontend entry of the stack builds its kernel
for the run's sequence counts on scratch operands: an entry asks what it needs of the installed toolchain where it
builds, so a CuTe DSL that cannot build for the GPU is a `SKIP` in the entry's own words here, not an error after the
weights are read. Then the weights are read, merged (query, key and value; `qkv` and `z`; the
beta and alpha rows, beta first; gate and up) and dequantised to bf16 copies, each entry's kernel is launched once, and each prompt
is fed through the decode step and decoded greedily to an end token or to its limit of new tokens. The text of sequence 0
goes to standard output as it is generated; reasoning the model writes before its answer is part of that text.

How long it takes, as orientation values of one run (compute capability 10.3, the default prompt, 256 new tokens, the
checkpoint in the page cache): about two and a half minutes. About 6 s until the first step (import, read, merge,
dequantise, compile, warm-up); 14 s for the 58 prompt tokens; 54 s for 225 new tokens, a quarter of a second per token on
this stack; 12 s for the traced pass; and about a minute for the checks. The text of sequence 0 is written while the
loop runs: each piece between the host read of one step and the launch of the next, flushed, so its write is part of
that step's turnaround (`--no-stream` writes the text once, after the report, and keeps the write out of every timed
step). On standard error a run prints one line before the prompt and the decode loop (`[00] decoding: ...`) and one
before the checks (`[00] checks on ...`), and nothing while either of the two runs.

Memory, in GiB (2^30 bytes), as the memory lines of a run print it: the stack `fe-cublas`
holds every weight as a bf16 copy, 50.1 GiB for this checkpoint, and its plan asks for about 56.7 GiB free with one
sequence and the default prompt: the copies, the state and working memory, which is an eighth of the copies for a model
of this size (2 GiB for the default run). A run with this checkpoint held 51.5 GiB in PyTorch at its peak, reserved
55.2 GiB from the device, and the device grew by 55.4 GiB. A GPU with less free memory than the 56.7 GiB is a `SKIP`
before a tensor is read. A 32 GB GPU cannot run this stack with a checkpoint of this size; it prints the
memory sentence and exits 0. On compute capability 10.0 the stack `native` keeps the in-projections on their stored
8-bit codes alone, 4.8 GiB where their copies are 9.7 GiB, and the gate / up weights on their stored 4-bit codes alone,
6.0 GiB where their copies are 21.2 GiB: it holds 30.0 GiB of weights, and its plan asks for about
37.1 GiB -- the weights, the state, working memory (an eighth of the weights held and a quarter of the stored tensors
kept) and 0.33 GiB for the one copy its checks make at a time. A run held 31.4 GiB in PyTorch at its peak, reserved
36.7 GiB, and the device grew by 36.9 GiB: inside the sentence by 0.2 GiB. On the other 10.x classes, where the
in-projections stay on their copies, its plan asks for about 41.3 GiB; that is arithmetic, not a run. The working
memory is an allowance taken from runs, not a bound: the room
a run took beyond its weights did not shrink with the copies it left out -- 5.3 GiB with every weight as a copy, 6.9 GiB
with the stored in-projections beside every copy and 6.6 GiB with the stored in-projections alone (two runs of
`native` before it had the gate / up row), 6.9 GiB with the in-projections and the gate / up weights on their stored
formats alone. The checks of the last run read 6,400 copies of in-projection and gate / up weights made on demand, 128
per step and reader, and printed what they cost. A stack on the stored formats alone,
which needs less than half, arrives with a later step of this area.

The prompt goes through the decode step, one step per token (the `first token` line says so), and a step issues 484 block
calls from Python: on the run above about twelve thousand kernel launches and a thousand small copies per step, with the
GPU busy for about a quarter of the period. That is what this stack is: it reads 51.6 GB per step where the model in its
stored formats needs 17.9 GB, and it is bound by the launch path, not by the GPU. The stack `native` on compute
capability 10.0 reads the in-projections on their stored codes, 5.2 GB where the copies are 10.4 GB, and the gate / up
weights on theirs, 6.4 GB where the copies are 22.8 GB: 30.0 GB per step,
and its `bytes per step` line says so. The timings a run prints are orientation
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
comparison made elsewhere, and greedy decoding is not the checkpoint's own default (its generation settings sample,
which every run with a checkpoint says in its `greedy decoding` line).

## The checkpoint's drafter

A checkpoint of this family can carry a second, small network beside the model: a drafter. In speculative decoding the
drafter proposes the next few tokens from the model's last hidden row, and the model then checks all of them in one step
instead of producing them one by one. `nvidia/Qwen3.8-27B-NVFP4` carries one.

**What is here today is its loader, and nothing else** (`QwenDenseDrafter`, `model_samples/qwen_dense/drafter.py`). No
script of this folder drafts yet: there is no draft step, no block and no provider row for the drafter, and
`00_decode.py` neither reads it nor has a flag for it. The loader is called from Python, from the root of this checkout:

```python
import sys

sys.path.insert(0, "samples/models")
from model_samples import Checkpoint
from model_samples.qwen_dense import QwenDense, QwenDenseDrafter

checkpoint = Checkpoint.describe("/data/Qwen3.8-27B-NVFP4")  # config.json and the shard headers: no tensor is read
model = QwenDense.describe(checkpoint)
drafter = QwenDenseDrafter.describe(checkpoint, model, draft_vocab="draft_vocab.json")
print(drafter.manifest_text())  # which tensors, and what a load makes of each
weights = drafter.load(checkpoint, "cpu")  # or a CUDA device
```

It needs the packages of `samples/models/requirements.txt` and the checkpoint on local disk, as `--model` does, and a
list of token ids, `draft_vocab.json`, which the recipe below makes.

Called as above, `load` also reads the model's embedding table, which the last line of the manifest below leaves out:
248,320 x 5,120 elements in bf16, 2.37 GiB for the published checkpoint. A caller that has loaded the model passes the
table it holds, `drafter.load(checkpoint, device, embedding=table)`: the drafter then reads none and keeps no second
copy.

The loader does three things, each from the checkpoint itself.

**It finds the drafter's tensors and says what a load makes of each.** They sit under one prefix of the checkpoint
(`mtp.` in the published one), all stored in bf16: two norm weights in front of a multiply twice the hidden width wide,
one attention layer with its MLP at the model's widths, and a final norm. The drafter has no embedding table and no
output head of its own; it reads the model's. The list is bound from `config.json` and the shard headers, beside the
model's list and before a tensor is read. A checkpoint without a drafter, or with one that is another network -- more
than one layer, an embedding table of its own, a tensor that is missing, stored in another format or of another shape,
a tensor under the prefix that is not on the list -- is declined there with a sentence that names what does not fit
(a `NotImplementedError`, which a script of this area prints as `SKIP`). For the published checkpoint the loader prints:

```text
drafter of checkpoint Qwen3.8-27B-NVFP4 (local): 15 tensors of its own under 'mtp.', 3 of the model's
  tensor                                        stored                at load
  mtp.pre_fc_norm_embedding.weight              bf16 [5120]           pre_fc_norm_embedding: as stored
  mtp.pre_fc_norm_hidden.weight                 bf16 [5120]           pre_fc_norm_hidden: as stored
  mtp.layers.0.input_layernorm.weight           bf16 [5120]           input_norm: as stored
  mtp.layers.0.self_attn.q_norm.weight          bf16 [256]            q_norm: as stored
  mtp.layers.0.self_attn.k_norm.weight          bf16 [256]            k_norm: as stored
  mtp.layers.0.post_attention_layernorm.weight  bf16 [5120]           post_norm: as stored
  mtp.norm.weight                               bf16 [5120]           final_norm: as stored
  mtp.fc.weight                                 bf16 [5120, 10240]    fc: 8-bit codes, one scale per row
  mtp.layers.0.self_attn.q_proj.weight          bf16 [12288, 5120]    rows 0 to 12,287 of qkv: 8-bit codes, one scale per row
  mtp.layers.0.self_attn.k_proj.weight          bf16 [1024, 5120]     rows 12,288 to 13,311 of qkv: 8-bit codes, one scale per row
  mtp.layers.0.self_attn.v_proj.weight          bf16 [1024, 5120]     rows 13,312 to 14,335 of qkv: 8-bit codes, one scale per row
  mtp.layers.0.self_attn.o_proj.weight          bf16 [5120, 6144]     o: 8-bit codes, one scale per row
  mtp.layers.0.mlp.gate_proj.weight             bf16 [17408, 5120]    rows 0 to 17,407 of gate_up: 8-bit codes, one scale per row
  mtp.layers.0.mlp.up_proj.weight               bf16 [17408, 5120]    rows 17,408 to 34,815 of gate_up: 8-bit codes, one scale per row
  mtp.layers.0.mlp.down_proj.weight             bf16 [5120, 17408]    down: 8-bit codes, one scale per row
  model.language_model.embed_tokens.weight      bf16 [248320, 5120]   embedding: the model's table
  lm_head.weight                                4-bit [248320, 5120]  head: its rows at the 65,536 ids of the draft vocabulary, as stored
  lm_head.input_scale                           fp32 scalar           head_quant: the static scale of the head's 4-bit operand
  draft vocabulary draft_vocab.json: 65,536 ids of the model's 248,320, in ascending order (65,536 entries listed, none dropped)
  loaded, besides the model's embedding table: 0.57 GiB (8-bit weights 0.40, norm weights 0.00, head rows 0.18)
```

**It quantises the weights of the drafter's five multiplies to 8 bits when it loads them**
(`weights.quantize_fp8_per_channel`): the checkpoint stores them in bf16, two bytes per element, and a loaded drafter
holds one byte per element and one scale per row. A row of a weight is one output channel of its multiply, and the
rule is per row:

- the scale is the row's largest magnitude divided by 448, the largest value of the 8-bit format (E4M3), formed in
  fp32. It is never below 1 / (448 x 512), so a row of zeros has a scale too;
- a code is the element divided by that scale, rounded to the nearest value of the format, to the even code on a tie,
  and saturating at +-448. The division is a division in fp32, not a multiplication by the reciprocal of the scale:
  where a quotient sits exactly between two codes the two give different codes.

A row is quantised by itself, so the weights that feed one multiply are concatenated first: query (with its gate half),
key and value; gate and up. A multiply over these codes applies the scale of each row to its result. The rule is fixed
this closely because the codes are an operand: a kernel that reads them computes with exactly these bytes. A host test
pins it, bytes and scales, against a quantiser written from the format's definition
(`test/python/misc/test_model_samples_drafter.py`).

**It cuts the drafter's output head out of the model's.** A drafter proposes from a part of the vocabulary, the *draft
vocabulary*: a JSON list of token ids. The loader takes each listed id once, in ascending order, and only ids the model
has, and says how many entries it dropped. The drafter's head is then the rows of the model's 4-bit output head at
those ids -- packed codes and block scales byte for byte as the checkpoint stores them, with the head's own global
scale and the static scale of its operand -- and `head_ids` gives the token id of every row. A file that is not a JSON
list of whole numbers, or that lists no id of the model, is refused with a sentence that names the file by its own
name (a `ValueError`, which a script of this area prints as `ERROR`); so is a checkpoint's drafter that is given no
list, and that sentence points to this section.

**A recipe for the list.** Any ids of the model can be listed. One way to choose them needs nothing but this sample:
let the model answer prompts of the kind you will give it, write the ids of every run with `--dump-tokens`, and list
what the model read and wrote in those runs, then the lowest ids that are not among them until the list is as long as
you want it.

```bash
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --prompt "Why is the sky blue?" --dump-tokens ids_1.json
python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --messages chat.json --dump-tokens ids_2.json
```

```python
import collections
import itertools
import json

dumps = ["ids_1.json", "ids_2.json"]  # the files --dump-tokens wrote, one or more
rows = 65536  # how many ids to list

seen = collections.Counter()
for path in dumps:
    with open(path) as f:
        for run in json.load(f)["runs"]:
            for sequence in run["prompt_ids"] + run["ids"]:
                seen.update(sequence)
listed = [token for token, _ in seen.most_common(rows)]  # what the model read and wrote, most frequent first
unseen = (token for token in itertools.count() if token not in seen)
listed += itertools.islice(unseen, rows - len(listed))  # then the lowest ids not among them
with open("draft_vocab.json", "w") as f:
    json.dump(sorted(listed), f)
```

**The runs behind this section** (2026-11-17; PyTorch 2.13.0, safetensors 0.8.0; the published checkpoint, manifest
hash `95049016a247`). On the CPU, without a GPU:

- the recipe, on six `--dump-tokens` files that runs of this sample with the checkpoint wrote (three prompts,
  2026-11-12, compute capability 10.0): they hold 2,341 tokens and 342 distinct ids; the list is those 342 and the
  65,194 lowest ids not among them, and the loader read it without dropping an entry;
- the manifest above: the checkpoint holds 15 tensors under `mtp.`, all in bf16, and the list names exactly those;
  nothing was read before the load. The call as it is written above read the model's embedding table too, 2,542,796,800
  bytes equal to the shard file's; a load that was handed a table read none;
- the head: all 65,536 rows of the loaded head equal the rows of the output head in the shard file at the listed ids,
  read there by the offsets of the shard's header -- 167,772,160 bytes of packed codes and 20,971,520 bytes of block
  scales -- and the two scalars are the file's;
- the 8-bit weights: every one of the 64,512 scales and of the 424,673,280 codes, checked a second way: the scale and
  the quotient formed in fp64 and rounded once to fp32, the code against the two midpoints around its value in a
  table written from the format's definition. No row is at the smallest scale. 2,021,676 quotients are exactly the
  midpoint of two codes, and each has the even code; a multiplication by the reciprocal of the scale would have given
  another code on 207,130 elements.

And the same load on a GPU of compute capability 10.0 (a 148-SM part): every byte of the five 8-bit weights, of the
norm weights and of the head rows equals the byte the CPU made, and every scale is the same bit for bit.

**What is not here.** No draft step and no speculative decoding: nothing runs the loaded drafter, so nothing compares
its output with anything, and no kernel has read these 8-bit weights. No script takes the list yet; the loader takes
it as `draft_vocab=`. The loader has run on the CPU and on one GPU class, compute capability 10.0; a class it has not
run on is uncovered, not passing. And the 8-bit weights are held to the rule above, not to the weights another
implementation makes of the same checkpoint: no such comparison is part of this repository.

## The verify step

Speculative decoding drafts a few tokens ahead and lets the model check them in one step: every sequence is fed the
token it kept last and the tokens drafted behind it -- `T` tokens per sequence at once -- and the step returns, for
each of them, the greedy token that follows it. A run keeps the drafted tokens up to the first one the model would
not have chosen, and goes on from there. This section is that step, the model's side of speculative decoding. Nothing
drafts yet (the section above loads the drafter and runs nothing), so no script runs a verify step: it is reached
from the helper layer, and the caller names the tokens it is fed and how many of them are kept.

```python
decode = Session(model, text, sequences=1, stack="native")                          # one token per sequence: the decode step
verify = Session(model, text, sequences=1, stack="native", tokens=6, share=decode)  # six: the verify step, on the same load
print(decode.plan_table([verify]), file=sys.stderr)   # both plans before the load: nothing declines after it
decode.load()
verify.load()
next_ids = verify.verify(ids, accept)                 # ids [sequences, 6] -> next ids [sequences, 6], on the GPU
```

`Session(tokens=T)` resolves the provider table for `T` tokens per sequence; the rows of a step are `sequences x T`,
sequence by sequence. `verify(ids, accept)` is one such step: `next_ids[i, t]` is the greedy token after token `t` of
sequence `i`, given the tokens in front of it. `accept` is how many of the tokens behind the first the run keeps,
`0` to `T - 1` -- one number for the fixed set of sequences, so that the attention caches keep one position; the
next step then goes on `accept + 1` positions later. Which tokens to keep is the caller's rule: the method compares
nothing, reads nothing to the host and copies nothing. A session of several tokens per sequence has no `prefill` and
no `step`: the decode step is a row of another plan, and a run that needs both has two sessions on one load, planned
together, as above.

**The state.** The recurrent state after EVERY token of a step is written to a slot of its own, so the next step can
start from whichever token the run kept, with nothing copied. The slots are numbered token by token -- sequence `i`,
token `t` owns slot `1 + t x sequences + i` -- so token 0 of a sequence is its plain slot `i + 1`, and the layout of
one token per sequence is this layout at `T = 1`. The conv history of a sequence holds `T + 2` columns in its plain
slot: a step reads its three taps where the last step left them and writes the last two of them followed by its own
`T` rows. Keeping `n` tokens moves one number on the host; the kernel reads it from a constant table. An attention
layer inserts `T` positions per sequence, and token `t` attends up to its own.

**Who serves it.** The block `linear_attention.step` is served by `fe.gdn_verify_fused`, `cudnn.GdnVerifyFused`: one
launch per layer for all tokens of all sequences. The step shape is a compile-time fact of that kernel, so the table
has one row per tokens count, 2 to 8, with `TOKENS` as a field of the form, for 1 to 16 sequences below six tokens
per sequence and 1 to 4 from six -- the step shapes the target runs on this kernel; eight tokens is what every form
of the entry holds on the classes it serves. A step shape beyond these is declined before the load, by name:
`cudnn.GdnVerifyFused serves 6 tokens per sequence of 2-8, sequences 1-4; this run has 5 sequences x 6 tokens per
sequence`. In the stacks `native` and `fe-entries` the row is the target's form, `[PDL True, PDL_RELEASE True, TOKENS
T]`: a dependent launch that waits for the in-projection only before it reads that launch's rows, and releases its
successor at entry. In `fe-cublas` it is `[PDL False, PDL_RELEASE False, TOKENS T]`. Neither the launch knobs nor the
tokens are in the kernel's name, which is cut in front of them: the form on the objects shows them, and the shared
memory of a thread block follows the tokens (126,512 B at six tokens and one sequence on compute capability 10.0).

Every other block is the row its own ranges give at `sequences x T` rows, as at any other row count. At one sequence
of six tokens the norm entry, the in-projection entry (compute capability 10.0), the fused gate / up entry (the
8-token tile) and the attention gate entry keep their rows; at 16 sequences of five tokens -- 80 rows -- the
in-projection (rows 1-32) and the gate / up step (rows 1-64) are on their fallback rows, printed with their reasons.
The attention preparation and the attention core are reference chains, called once per token. The out-projection of
a linear-attention layer is still the multiply over a copy: the target runs a kernel of cuDNN Frontend there in a
verify step (the 8-bit projection in a split of its own), which this stack does not serve yet -- it is counted as
`not yet`.

**What a verify plan says about the target** (stack `native`, the 64-layer model, compute capability 10.0 with 148
SMs, one sequence of six tokens: the one step shape the census has a recorded section for):

```text
the whole target kernel set of a step of this model (what a complete stack launches; this stack's share is the next line): 484 launches per step = 162 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal results + 1 declared stand-in
kernel identity, as planned: 322 of 484 launches of a step are target kernels (49 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results); not yet: projection.fp8.out 64, mlp.down 64, attention.prepare 16, attention.core 16, output_head 1; 1 declared stand-in (not yet: output_head.quantize)
```

The target's verify step has other kernels than its decode step in two places -- the fused step is the verify
kernel, and the out-projection of a linear-attention layer is a kernel of cuDNN Frontend where the decode step has a
library multiply -- so the whole is 162 + 321 + 1 where the decode step's is 210 + 273 + 1. The census section holds
for the tokens per sequence, the row count, the layer widths and the GPU class it was recorded with; any other verify
step prints `kernel identity: not stated for this run (no recorded section for ...)`. The drafter's own launches are
not part of it.

**The checks.** A verify session is checked by the checks of every run (`tie_out.run_checks`), on its own steps:
replay; every provider block against its reference, teacher-forced; the greedy tokens against the reference stack,
which runs the same verify steps; one sequence alone against the sequence among several, with its `T` rows; the
kernel check. The reference of the verify block is `T` reference decode steps, each started from the state the
kernel left for the token before it: the block's line then says, token by token, that the state the kernel wrote
for a token is one decode step on from the state it wrote for the token before -- which is what "the state of the
kept tokens" means for every number of tokens a run may keep -- in the band of the decode step (`gdn_step`: a cell is
one value head of one sequence over the tokens of its steps).

And it is held against the decode steps themselves (`tie_out.verify_checks(verify, decode)`): the token stream of a
run of decode steps goes through verify steps that keep 0, 1, .., `T - 1` tokens in turn, so every count is kept at
least once and every step after the first starts from a state a partial keep left. Every row of every step is then
compared with the decode step of the same token (`tokens`: the greedy tokens and the logits, by the family's logits
band), and after every step the state the verify run stands on is compared with the state of the decode steps of
the kept tokens (`kept state`). A linear-attention layer whose history taps are the same bytes in both runs was fed
the same rows by both: there the recurrent state is gated by the state rule of the band. A layer that was fed other
rows -- a block in front of it is another kernel at another row count -- is printed, not gated; with no layer gated
the line is `REPORTED`.

On compute capability 10.0 (a 148-SM part), stack `native`, the two-layer seeded model: at one sequence of six
tokens and at four, every greedy token of the verify steps was the decode steps' and the logits were the same bytes;
the recurrent state of the kept tokens was within the band everywhere (at one sequence the same bytes), the history
and the attention caches byte-equal; the kernel check passed with the verify kernel in the shape its entry states
and, at one sequence of six tokens, in the shape the census records. These are the runs of one GPU class.

**What the entry states about one sequence alone.** `cudnn.GdnVerifyFused` is replay-identical, and it states no
batch invariance between one sequence and several: one sequence takes another association of the gated norm's sum
than two and more, so the last bits of its output may differ by design (`docs/operations/determinism.md`: "Replay
only"). The rows therefore gate no batch invariance on this block; a run with more than one sequence prints whether
the bytes were equal and, beside it, this statement.

**What is not here.** No drafter runs, so nothing decides how many tokens to keep: the caller says it. One count
per step for all sequences: a step that keeps another number of tokens per sequence needs a position per sequence
in the attention blocks, which is not built. The verify out-projection on its target kernel, the attention blocks
and the MLP's down multiply on theirs. A captured verify step. A verify step of more than eight tokens per
sequence, of more than 16 sequences, or of more than 4 at six tokens and more: declined by name. The verify step
has run on compute capability 10.0 only; every other class is uncovered, not passing.

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

## Who serves each block (stacks `native` and `fe-entries`, compute capability 10.0)

| block | calls per step (64 layers) | provider key | entry [form] |
|---|---|---|---|
| token embedding lookup | 1 | `torch.index_select` | torch.index_select |
| residual add + RMSNorm + quantise | 129 | `fe.add_norm_quant_decode` | cudnn.AddNormQuantDecode [PDL True, PDL_RELEASE True] + one of 6 forms, by call site |
| 8-bit in-projection (linear attention QKVZ / attention QKV) | 48 / 16 | `fe.fp8_projection_decode` | cudnn.Fp8ProjectionDecode [SPLIT_K 1, MMA_M 128, STAGES 8, TMEM_COLS 32, PREWAIT_WEIGHTS True, HOLD_TMEM True, PDL True, PDL_RELEASE True] + one of 2 forms, by call site |
| linear-attention step | 48 | `fe.gdn_decode_fused` | cudnn.GdnDecodeFused [PDL True, PDL_RELEASE True] |
| 8-bit out-projection | 48 / 16 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 8-bit weights |
| MLP gate / up + SiLU x up + quantise | 64 | `fe.swiglu_fp4_decode` | cudnn.SwigluFp4Decode [TOKEN_TILE 8, PREWAIT_STAGES 2, L2_EVICT_FIRST True, MAX_CLUSTERS 148, PDL True] |
| MLP down | 64 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 4-bit weights |
| attention preparation | 16 | `torch_ref.attention_prepare` | torch_ref.attention_prepare, fallback: reference chain |
| attention core | 16 | `torch_ref.attention` | torch_ref.attention, fallback: reference chain over a plain per-layer cache |
| gate + quantise | 16 | `fe.attention_gate_quant_decode` | cudnn.AttentionGateQuantDecode [THREADS 256, PDL_RELEASE True] |
| output head | 1 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 4-bit weights |
| token choice | 1 | `torch.argmax` | torch.argmax |

By provider: cuDNN Frontend 5; fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2); PyTorch as the natural provider 2.
The two 8-bit projections count as one kind of block per provider, and on this class they have two: the in-projection
is cuDNN Frontend's, the out-projection a multiply over a copy. The six forms of the norm row and the two weight
shapes of the projection row, by call site, are in the section on the stack `native`. The row of the MLP's gate / up
multiply is the one of 1 to 8 rows; three more rows of the same key name the other token tiles: `TOKEN_TILE 16` from 9
to 16 rows; `TOKEN_TILE 32` from 17 to 32 rows; `TOKEN_TILE 64` from 33 to 64 rows. From 65 rows the block is the
multiply over a copy of `fe-cublas`. On the other 10.x classes the
in-projection is the multiply over a copy of `fe-cublas`: cuDNN Frontend 4; fallbacks 5 (cuBLAS over copies 3, PyTorch
chains 2). On compute capability 12.x the gate / up multiply is over a copy too, and the gate + quantise row and the
norm row are the reference chains of `fe-cublas` as well: cuDNN Frontend 1; fallbacks 8 (cuBLAS over copies 4, PyTorch
chains 4).

| GPU classes | sequences | regime | note |
|---|---|---|---|
| any | 1 and more | launch-bound | every block of these stacks is launched from Python, one launch or more per block; no byte-bound form is declared, and the sequence count at which one would win is not measured on any class |

## The checks of a run

Against the plain-PyTorch reference of every block, which keeps the chain's casts and accumulates in fp64
(`model_samples/tie_out.py` has the definitions):

| check | what must hold |
|---|---|
| replay | the same inputs from a fresh state give byte-identical ids and logits |
| blocks, teacher-forced | every value finite; every multiply within one bf16 rounding of the reference, element by element; the linear-attention step inside its band (below); in the stacks `native` and `fe-entries`, the gate + quantise entry and the add + RMSNorm + quantise entry each inside its band against its twin, and the fused gate / up entry inside the error bound of its contract around the fp64 chain (below) |
| tokens | the logits of the two stacks, each run end to end on its own state, are finite. On the reduced seeded model they are also within the declared logits band (`configs.py`: 4 bf16 units of the run's largest reference logit), and the greedy token equals the reference stack's at every step whose margin exceeds twice that band; steps below it are counted and printed as near-ties. On a checkpoint no band is declared and the line is `REPORTED` (the section on checkpoints says why). The largest logit distance and the number of equal greedy tokens are printed either way |
| batch invariance | one sequence alone gives the bytes it has among 4; the sequence that runs alone rotates with the step, so every row of the batch is covered: a gate for the linear-attention step on compute capability 10.0, 10.3 and 12.0 (in its `native` form on 10.0, the class that form has run on) and for the gate + quantise entry, the add + RMSNorm + quantise entry, the FP8 projection entry and the fused gate / up entry on 10.0, a printed fact for every other block and class |

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

The add + RMSNorm + quantise entry is compared with its twin too, the norm written at the kernel's rounding points
(`AddNormQuantizeBracket`), and its band (`add_norm_quant`) is the numerics contract on its page
(`docs/operations/normalizations/add_norm_quant_decode.md`), per output of a call. The kernel adds in fp32, normalises
in fp32, rounds the normed value once to bf16 and forms its codes from that bf16 value; the twin evaluates the same
value in fp64 and keeps the two bf16 values around it. The residual stream must be byte for byte `bf16(fp32(branch) +
fp32(residual))`. Every normed element must be one of the two bf16 values, and the nearest one wherever the value is
more than 20 fp32 units from their midpoint: the page states the kernel's fp32 value within 20 x 2^-24 of the fp64
one. 8-bit codes must be byte for byte the static quantisation of the normed rows the same call wrote; where a form
writes codes alone, each must be the code of one of the two bf16 values, the nearest one's beyond the margin. For 4-bit
codes the block scale must be `e4m3(global scale x (largest magnitude x fp32(1/6)))` of a block maximum the two values
admit, bit for bit, and under the block scale the kernel stored every code must lie in the bracket of the kernel's two
approximate reciprocals around the code of an admitted value -- two adjacent codes at most, and one wherever the
quotient is not within 5 x 2^-23 of a rounding midpoint, the bound the entry's own suite holds its codes to. Every
value finite. The line prints how many normed elements took the other bf16 value and how many codes are not those of
the nearest value; anything outside these sets fails the block.

The 4-bit codes of this kernel are not those of the reference chain's quantiser on every element, and the band does
not ask for that. Where a bf16 value over its block scale is exactly a 4-bit rounding midpoint, the kernel's code
follows its own reciprocal and the reference quantiser's follows the one it forms; on random rows at this model's
scales the two differ by one code on about 6 elements in 10,000. Both are quantisations of the same bf16 value. A
block that reads the kernel's codes is compared, teacher-forced, with its reference on those same codes.

The fused gate / up entry has a band of another kind (`swiglu_fp4`), again the numerics contract on its page
(`docs/operations/gemm_fusions/swiglu_fp4_decode.md`). The kernel accumulates the gate and the up product in fp32
over the 5,120 inputs, rounds each once to bf16, forms SiLU x up with an approximate exponential and division, rounds
the product to bf16 and writes 4-bit codes under block scales it forms from that value; nothing of the product is
stored. An fp32 sum of thousands of products is not within a few units of the fp64 one, so there are no two bf16
values to bracket it with. The twin (`GateUpFp4Bound`) evaluates the chain in fp64 on the exactly dequantised operands
and propagates an error bound to every element, with the terms the entry's own suite derives: the fp32 accumulation
over the input width, each bf16 rounding, the approximate instructions. Every block-scale byte must lie in the
interval of bytes that bound admits for the largest magnitude of its block, and every 4-bit code, read under the
block scale the kernel stored, must be within half a code step of the fp64 value plus that element's bound. Every
value finite. The line prints how many block scales and codes are not those of the fp64 chain rounded at the kernel's
rounding points, and the largest distance of a code from the fp64 value as a share of its bound; one byte outside
fails the block. The bound is wider than one bf16 rounding where the products cancel, which is the contract's own
statement, not a tolerance of this sample: on the published checkpoint no block scale was another byte than the fp64
chain's in the run of the default prompt and 3 of 1,114,112 were in a run of a chat request body, each inside its
interval; about 7 codes in 10,000 were another code in both.

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

The provider stack and the reference stack read the same weights through the same loader and the same dequantisation
-- the reference reads bf16 copies, kept by the load or made from the stored tensor for one call -- and use the same
quantisers, so the checks above cannot see an error in either. `test/python/misc/test_model_samples_weights.py` pins
them against a decode written from the format definitions.

### What a run holds of a weight

A checkpoint stores a multiplied weight as 8-bit codes with one scale (the projections) or as packed 4-bit codes with a
block scale per 16 elements and a global scale (the MLP's two multiplies, the output head). The loader merges the
weights of one multiply and then keeps each in the forms the blocks of the run read, decided from the plans of every
sequence count before the first load:

| a block of the run | what the load keeps of its weight |
|---|---|
| reads a bf16 copy (every fallback row; every block of `--stack reference`) | the copy, 2 bytes per element |
| multiplies on the stored format (an adapter that says so: `stored_weights`) | the merged stored tensor: 1 byte per element of an 8-bit weight, 0.5625 of a 4-bit one; in the row order the adapter names, where it names one (`stored_layout`) |
| both kinds, at different sequence counts or through `--provider` | both |

So a weight that every plan of the run multiplies on its stored format has no copy. Today two rows do that: the FP8
projection entry for the in-projections on compute capability 10.0, and the fused gate / up entry for the MLP's first
multiply on compute capability 10.x. The second keeps its weight in the layout its kernel multiplies on, gate and up
rows interleaved in tiles of 64, with the block scales of those rows in the kernel's 128 x 4 layout: the same bytes
in another order, made once per weight when it is loaded. The MLP's down multiply and the output head have no row on
their stored format yet, and the loader keeps those forms for a host test's own blocks only. A session that runs on
weights loaded for another plan and finds a form missing, or a stored weight in another row order than its block
multiplies on, ends in an error that names the block, before its first step.

The reference blocks of the checks multiply bf16 copies. For a weight without one the checks make the copy
themselves, for one call, and drop it after the call (a host test holds that on the tensors: none of an earlier call
is alive when the next is made). Such a copy is made from the stored bytes as they are at the call -- the bytes the
weight's own block multiplies on. For a weight kept in a kernel's row order it is made from that order, a band of rows
at a time, into the checkpoint's: no second stored tensor stands between the kernel's bytes and the reference's copy,
so the checksum of the load covers what both read. As long as no block has written into its weight it is bit for bit the copy a load
would have kept, and no check line depends on which of the two a run has (a host test loads the same plan both ways
and compares every line). A block that did write would move its reference with it and pass its band, so the checks
hold that condition themselves: the loader takes a checksum of every weight it keeps on its stored format alone, and
the checks take it again before their first pass and after their last. A weight whose stored bytes are not those of
the load fails the run, named by block and layer (`stored weights`); in a run that keeps its copies the same defect
fails the block's band, because the reference reads the copy of the load. The memory sentence names the checks' part
-- the largest such copy, one at a time -- and the checks print what the copies cost:

```text
  this stack needs about 3.5 GiB on this GPU for 1 and 4 sequences, one count after the other, 0.33 GiB of it for the checks (the reference's bf16 copy of one weight at a time, where the run keeps none); 177.7 GiB are free
  stored weights    4 of 4 weights this run keeps on their stored format alone hold the bytes of their load, before the checks and after them: the reference's copies were made from what was loaded    PASS
  on demand         the reference read 460 bf16 copies of weights this run keeps on their stored format alone (projection.fp8.in, mlp.gate_up): each made for one call and dropped after it; the largest 0.33 GiB, 110.1 GiB in all, 1.8 s of the checks (orientation only)
```

A run in which every weight has its copy prints the memory sentence without that part and neither of the two lines:
the default stack's output is what it was. Only the reference session of the checks makes copies per call: a session
whose own steps are timed, traced or checked is refused such a maker when it is built.

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
| token embedding lookup | `embedding_gather`, the kernel PyTorch's index-select launches: one of three, by the number of rows and the size of a table row (a run launched the small-index kernel up to 16 rows and the vectorized gather from 17) | one launch per step, first in the step, known by its name; no launch shape is declared for it |

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
PyTorch's. The 40 copies are the host read and copies inside the fallback rows. At 17 sequences (2026-11-16, the same
part) it is `[00] PASS` as well, with 469 launches a step: from 17 rows PyTorch's index-select launches its vectorized
gather, and the embedding row's pattern names that kernel beside the two of fewer rows. The check has not been run on
another GPU class, and not with a checkpoint (where a traced step of this stack has about twelve thousand launches
and the trace the check reads grows accordingly): a class or a model it has not run on is uncovered, not passing.

## Support per GPU class

`cudnn.GdnDecodeFused` serves compute capability 10.x and 12.x and declines elsewhere; the sample then prints
`[00] SKIP cudnn.GdnDecodeFused serves compute capability 10.x or 12.x; this device is <cc>` and exits 0. The entry is
validated on compute capability 10.0 and 12.0 (`docs/operations/linear_attention/gdn_decode_flashinfer_backend.md`).
`nvidia-smi --query-gpu=name,compute_cap --format=csv` prints the compute capability of a GPU.

| compute capability | the default run of this sample |
|---|---|
| 10.0 (data-center Blackwell, for example a GB200) | `[00] PASS`, verified at 1 and 4 sequences; batch invariance of the fused step gated and equal; the checkpoint run of the section above. With `--check-kernels`: `[00] PASS` at 1 and 4 sequences, the only class the kernel check has run on. The stacks `native` and `fe-entries`: `[00] PASS` at 1 and 4 sequences and with the checkpoint, batch invariance of the fused step, of the gate entry, of the norm entry, of the projection entry and of the fused gate / up entry gated and equal; the only class these stacks have run on |
| 10.3 (for example a GB300) | `[00] PASS`, verified at 1 and 4 sequences; batch invariance of the fused step gated and equal; a checkpoint run before the one of the section above |
| 10.7 (a VR200-class part) | `[00] PASS`; batch invariance of the fused step is printed on this class (`REPORTED`), not gated: every block was equal in the runs so far |
| 12.0 (for example an RTX 5090) | `[00] PASS`, verified at 1 and 4 sequences; batch invariance of the fused step gated and equal. The checkpoint does not fit a 32 GB card: on one, the sample printed the memory sentence (about 59.7 GiB needed) and exited 0 |
| any other | `[00] SKIP`, by design. `--stack reference` runs the reduced model there on PyTorch alone, with the whole report, and ends in `[00] RAN reference stack (no provider checked)`: nothing is verified (seen on compute capability 8.0) |

A class the current checks have not run on is uncovered, not passing. That holds for the stacks `native` and
`fe-entries` on every class but 10.0: the gate entry, the norm entry, the projection entry and the fused gate / up
entry inside this sample, the
fused step in its wait + release form, and the kernel patterns of the check have run on 10.0 only. The norm entry admits every compute
capability 10.x; its own page lists 10.3 and 10.7 as not run on a device. So does the page of the fused gate / up
entry, whose rows here claim every 10.x class as the entry does: on 10.3 and 10.7 the rows, the weight in the kernel's
row order and the memory sentence of that plan are unverified. So has the build of the entries' kernels while a run is
planned: what a plan prints on a GPU for which the installed CuTe DSL cannot build (a `SKIP` in the entry's own words) is
held by a host test over a stand-in entry and was seen on a real GPU only with the entry's own toolchain question made to
decline, not on such an installation.

Memory of the default run (2 layers, 1 and 4 sequences): 1.7 GiB of bf16 copies; PyTorch held 2.5 GiB at its peak,
reserved 3.5 GiB and grew the device by 3.7 GiB on a compute capability 10.0 GPU, so the sample asks for about 3.7 GiB
free there. On compute capability 12.x
it adds an allowance of 3 GiB and asks for about 6.7 GiB: the first launch of the fused step reserves device memory outside
PyTorch's pool on that class. The need is stated once, for every sequence count of the run, before anything is loaded: the
copies, the working memory and the first launch are counted once, the state per count. The run measures that launch and
prints what it took. Measured by this sample on a compute
capability 12.0 GPU of 32 GB: 3,124 MiB at the first launch of the 1-sequence form, nothing at the 4-sequence form, and
6.9 GiB of growth on the device over the whole run.
On the reduced models the memory sentence is an estimate: its working memory is a flat 2 GiB, and the runs of this page
miss it by a few tenths of a GiB in either direction, with the copies of the in-projections and of the gate / up
weights, with both of their forms and with their codes alone. Over these runs on compute capability 10.0 (one to four
layers, 1 to 33 sequences) PyTorch reserved between 0.5 GiB less and 0.1 GiB more than the sentence said, and the
device grew by about 0.2 GiB more than PyTorch reserved. The default `native` run, with the in-projections and the
gate / up weights on their codes alone, asked for about 3.5 GiB, reserved 3.6 GiB and grew the device by 3.8 GiB; the
four-layer run asked for about 4.4 GiB, reserved 4.4 GiB and grew it by 4.6 GiB; the one-layer run asked for about
3.1 GiB, reserved 2.6 GiB and grew it by 2.8 GiB; `--stack native --sequences 32,33`, which keeps both forms of the
in-projections, asked for about 4.0 GiB, reserved 4.1 GiB and grew it by 4.2 GiB. The room a run takes beyond its
weights does not follow the copies it leaves out: with every multiplied weight of the two-layer model on its stored
format alone (a driver's own blocks, 1.5 GiB of copies not made) a run asked for about 3.2 GiB, reserved 2.7 GiB and
grew the device by 2.9 GiB. So a GPU whose free memory is within half a GiB of the sentence can still fail to allocate
after the plan with these models; with the published checkpoint the working memory is an eighth of the weights held
and the runs stayed inside the sentence (the section on checkpoints).
Beyond 33 sequences the sentence is short, on every stack: what a run holds per sequence beyond the state the
sentence counts is not in its flat allowance. At 64 sequences the default stack asked for about 3.9 GiB, reserved
6.6 GiB and grew the device by 6.8 GiB; `--stack native` asked for about 3.8 GiB, reserved 6.0 GiB and grew it by
6.2 GiB; and `--stack native --sequences 64,65` asked for about 4.4 GiB, reserved 8.0 GiB and grew it by 8.2 GiB. At
such a count a GPU needs several GiB more than the sentence says.
The four-layer run (`--layers 4`) holds 3.1 GiB of copies. The whole default run takes well under a minute on a data-center
Blackwell GPU: a second or two to plan, which includes building the entries' kernels for both sequence counts (1.1 to
1.7 s in the runs since that build became part of the plan, 3.6 s in the first of them; built after the load, the fused
step's kernel took 0.5 to 2.4 s for the first sequence count and 0.3 to 0.4 s for the next), about 5 s to draw the
weights, the rest for the decode and its checks. These are orientation values of single runs, not measurements of record.
