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
| `02_speculative_decode.py` | speculative decoding with the checkpoint's drafter, one request in one process: the prompt through decode steps, then per step the drafter's passes on reference blocks, one verify step and one host read; on the reduced seeded model self-checking against decode steps of the same tokens, with `--model` text out and the tokens a step kept | the section "Speculative decoding" |

```bash
python samples/models/qwen_dense/00_decode.py                          # the stack native: 2 layers, 1 and 4 sequences, 32 tokens each, the step recorded and replayed
python samples/models/qwen_dense/00_decode.py --sequences 4 --layers 4
python samples/models/qwen_dense/00_decode.py --no-capture             # the same stack, every block launched from Python
python samples/models/qwen_dense/00_decode.py --stack fe-entries       # a functional-validation stack: no package beyond cuDNN Frontend and PyTorch
python samples/models/qwen_dense/00_decode.py --stack reference        # every block on the reference: RAN, never PASS

# A checkpoint is your own download: the sample fetches nothing. 22 GB of disk; a GPU of compute capability 10.x with
# about 28 GiB free on 10.0 and about 32 GiB on the other 10.x classes ("A checkpoint from local disk" has the details).
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
| `--messages FILE` | -- | a chat request as JSON, from a file or, with `-`, from standard input (once per run): a list of messages; an object with `messages` and optionally `tools` (JSON tool schemas) and `chat_template_kwargs` (options the checkpoint's template understands); or the whole request body a chat client sends, every field of which is answered ("A request body and its options" below). Repeatable |
| `--max-tokens N` | `256` | with `--model`: the limit of new tokens of a request that names none of its own: stop after N new tokens unless an end token comes first. `--max-new-tokens N` is the same flag. The limit a request body names rules over it, and a run says so. Without `--model` the flag is an error: the seeded run decodes `--force-tokens` |
| `--max-model-len N` | `2051` | the positions a prompt and its new tokens share. A limit of new tokens somebody asked for that does not fit beside the prompt, or a prompt that leaves no room, is an `ERROR` before anything is planned or loaded; the default limit of 256 is cut to the room that is left, and the run says so. The default is the context length the runs of this area were made at. The flag bounds a request; the state of a run is sized for the run itself |
| `--temperature 0` | -- | accepted at 0 and nothing else: token choice is greedy |
| `--force-tokens N` | `32` without `--model`, off with it | decode exactly N tokens per sequence, ignoring the end token; not together with `--max-tokens`; it rules over the limit a request body names, and a run says so |
| `--text-out FILE` | -- | with `--model`: the generated text of sequence 0 goes to FILE instead of standard output; the file then holds what standard output would have held: one text per prompt, in order, each ending in a line break. An empty name (`--text-out "$OUT"` with the variable unset) is an `ERROR` before anything is planned, as it is for `--dump-tokens` and `--dump-census` |
| `--no-stream` | off | with `--model`: the text is written once, after the report of its prompt, instead of piece by piece as it is generated |
| `--sequences N[,N...]` | `1,4` without `--model`, `1` with it | sequences per run, fixed before the first step; a list repeats the run per count; with `--model` every sequence of a run gets the same prompt, and the text of a prompt is written once, with the first count of the list (a later count reports whether its sequence 0 produced the same text; `--dump-tokens` has the ids of every count) |
| `--layers N` | `2` | layers of the reduced model: the last N of the published order, so the model ends in an attention layer; not with `--model` |
| `--stack fe-cublas\|native\|fe-entries\|reference` | `fe-cublas` | `fe-cublas`: cuDNN Frontend + cuBLAS + PyTorch, checked against the reference. `native`: every block on a kernel that works on the stored weights ("The stack `native`" below); the kernel check is always on. `fe-entries`: the cuDNN Frontend rows of `native`, everything else on fallback rows. `reference`: every block in PyTorch, nothing checked. Explicit, never chosen by what is installed |
| `--provider BLOCK=KEY` | -- | selects one declared provider row of a block; the key list is closed. Block names are in the table a run prints, keys in the table below; an unknown block or key is an `ERROR` that lists the declared ones |
| `--reference-precision fp64\|ieee-fp32` | `fp64` | accumulation of the reference blocks; printed in the header. `ieee-fp32` sets `torch.backends.cuda.matmul.fp32_precision` for the rest of the process, which ends with the run |
| `--dump-tokens FILE` | -- | per prompt, the prompt ids and the produced ids of every sequence as JSON, for use outside the run |
| `--check-kernels` | off; always on with `--stack native` and `--stack fe-entries` | adds the kernel check: prints the kernel table, keeps the profiler's record of every launch of the traced steps and holds them against the plan ("The kernel check" below). Not with `--stack reference`, which checks nothing |
| `--dump-census FILE` | -- | the launch records of the traced steps as JSON (per launch its kernel id, name, grid, block, shared memory, registers per thread, and its start and end in milliseconds on the profiler's clock), with the block calls of a step, what the entries state, and per call site the entry instance that served it and the form that instance states (`served`, beside the row's `form`), for use outside the run; implies `--check-kernels` |

A `--messages` file (the `chat.json` of the last command above), here with a system turn and the reasoning switched off:

```json
{"messages": [{"role": "system", "content": "Answer in one short sentence."},
              {"role": "user", "content": "What is 17 times 23?"}],
 "chat_template_kwargs": {"enable_thinking": false}}
```

A list of messages alone is also a valid file, and `tools` takes JSON tool schemas. The options are those of the
checkpoint's chat template: for `nvidia/Qwen3.8-27B-NVFP4`, `enable_thinking: false` leaves out the reasoning the model
otherwise writes before its answer.

## The performance mode

`--perf` runs the timed-request protocol on this script and prints the decode performance report of this GPU; the
protocol, every statistic with its definition, the labels and the report file are in
[../README.md](../README.md#the-performance-mode). Its flags:

| flag | default | meaning |
|---|---|---|
| `--perf` | off | run the protocol and print the report: one checked request with the reference checks on it (also at four positions of the timed steps), a warm-up request, the timed requests with nothing but the step inside, then one traced request and the kernel check on it. Implies one sequence count and one prompt, the kernel check (as `--check-kernels`), forced steps with the end token ignored, and the text written once, after the report (as `--no-stream`). Not with `--stack reference`, `--max-tokens`, `--force-tokens`, a list of sequence counts or several prompts: each is refused before the load |
| `--perf-warmup N` | `32` | forced decode steps of the warm-up request |
| `--perf-steps N` | `256` | timed decode steps of a request, behind its first new token: 257 new tokens and 256 intervals at the default. At least 2 |
| `--perf-requests N` | `3` | timed requests; the time per output token and the time to first token of a run are the median over them, the token-to-token intervals are pooled |
| `--perf-report FILE` | `report-<UTC time>.json` in the current directory | the report file: every number, its definition, the protocol, the environment. The default name carries a time stamp, so the same command can be run twice. A file that exists, a file that `--dump-tokens`, `--dump-census` or `--text-out` names too, an empty name or a folder that cannot be written is an `ERROR` before the load: a report file is never written over |
| `--perf-clock-lock MHZ` | none | the SM clock in MHz that an operator fixed outside the sample: a declaration. The report prints it beside the clock it observed and names it as a reason where a sample lies more than 3 % below or above it; the sample never sets a clock |
| `--perf-trace-records N` | `160000` | the most launch records the traced request may hold. Its first step is traced alone and sizes the rest (it is always traced, also where its records alone pass N); where the timed steps do not all fit, it traces the first steps that do, and the report says how many and at which positions |

One session serves every request of a run and is built for the longest of them: the checked request and the timed
requests run `--perf-steps` forced steps behind the first new token, the traced request at most as many, and the
warm-up request `--perf-warmup`, which may be the larger count. A prompt whose tokens, its first new token and the
forced steps of that longest request do not fit `--max-model-len` is an `ERROR` before the load, like every request
beyond the context: it says which request is the longest and names every count to take lower (`--perf-steps`,
`--perf-warmup`). No count is changed to make a run fit. On the stack `native` the requests of the mode are
replays of the recorded step, as a default run's are. A run of the mode that is `orientation`
says why on the second line of its report: its blocks are launched from Python (`--no-capture`, a
functional-validation stack), a block on a fallback row is a
reason of its own, and whatever the run saw of the GPU's other processes and of its clock that a controlled run may
not have is a reason too.

## A request body and its options

This sample is one process with greedy token choice: no socket, no queue, no sampling. It takes the inputs of a
request as they are.

| what you have | what the sample takes | note |
|---|---|---|
| the model | `--model DIR`, a directory on local disk | a hub name is not fetched |
| the request body your chat client sends | `--messages request.json`, or `--messages -` to read it from standard input | every field of the body is answered: the table below |
| `max_tokens` or `max_completion_tokens` in the request | read from the body; `--max-tokens N` is the limit of a request that names none | default 256: a request that names no limit stops at 256 new tokens |
| the context length (`--max-model-len`) | `--max-model-len N`, default 2051 | a prompt and its new tokens share it; a request beyond it is an `ERROR` before anything is loaded |
| `temperature`, `top_p`, the checkpoint's generation settings | nothing: token choice is always greedy. `"temperature": 0` and `--temperature 0` are accepted | the checkpoint's own generation settings are never applied, also to a request that names no temperature, and every run with a checkpoint says so in one line (below) |
| `"stream": true` | the text of sequence 0 goes to standard output as it is generated; `--no-stream` writes it once; `--text-out FILE` writes it to a file | every other line of the run goes to standard error |
| concurrent requests | `--sequences N` | a fixed set of sequences that start together, each with the same prompt; nothing is admitted while a step runs |
| which kernels serve the model | `--stack`, and `--provider BLOCK=KEY` for one block | a kernel choice is a row of the table a run prints before it loads anything |
| a speculative configuration: the JSON object that says how tokens are drafted | `02_speculative_decode.py --speculative-config JSON` | every key of it is answered: read, accepted at the value that changes nothing, or refused by name (the section "Speculative decoding"). `00_decode.py` refuses the flag with the name of that script |
| a draft vocabulary for the checkpoint's drafter | `02_speculative_decode.py --draft-vocab FILE` | a JSON list of token ids; the section "The checkpoint's drafter" has a recipe |
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
| stated | `model` | its last part is printed and compared with the name of the checkpoint directory; what stands in front of it -- an organisation, or the path of a directory -- is not printed (`".../Qwen3.8-27B-NVFP4"`). Another name is not an error: the run decodes the directory it was given, and says so |
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

The table holds the fields of the public chat request and of the sampling parameters commonly sent with it that ask
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

Two lines of a run with a checkpoint say what a request meets here. Both are printed before
anything is loaded:

```text
[00] greedy decoding: the token of the largest logit at every step; the checkpoint's own generation settings (do_sample true, temperature 1.0, top_k 20, top_p 0.95 in generation_config.json) are not applied
[00] first token: a prompt goes through the decode step, one step per prompt token (no stack of this sample has a prefill kernel): 58 decode steps before the first new token of the default prompt, as long as 58 new tokens take. No text is written until then; the first token line of the report has the time
```

The first states the decoding rule beside what the checkpoint's `generation_config.json` names, so a run
never differs from those settings silently. The second states what a prompt costs here: no stack of
this sample prefills a prompt with a kernel of its own, so a prompt of N tokens is N decode steps and takes as long as
N new tokens before the first new token appears. It gives a count and no time, because before the load a run has no
time of its own to give; the `first token` line of the report has the measured one. In the runs of 2026-11-17 on
compute capability 10.0, a prompt of 29 tokens took 4.3 s before the first new token on the stack `fe-cublas` (0.15 s
per step) and the 58 tokens of the default prompt took 7.8 s on the cuDNN Frontend rows of the stack `native`, the rows
`fe-entries` selects today without the attention preparation's row (0.13 s per step): orientation values of single runs
on a shared GPU, as every timing
on this page.

What is not here, and is not planned for a sample: a network interface of any kind; a request queue, or admission of
work while a step runs; different prompts in one step; sampling; stop strings, constrained output, logit processors;
parsing of tool calls in the output (tool schemas are rendered into the prompt, the output is text); a download from a
hub; an automatic choice of the stack.

## What the default run is

A run that names no stack is a run of the stack `native` ("The stack `native`: every block on a kernel" below): every
block of a decode step on a kernel entry of cuDNN Frontend or on a call of FlashInfer, on the stored weights, and the
step recorded once as a launch sequence and replayed for every token ("The recorded step" below). That holds with a
checkpoint and without one. A package the stack calls that is not installed stops the run before anything is loaded:
one `[00] ERROR` line that names the package and the command that installs it, exit code 1. No other stack takes its
place: the stacks `fe-entries`, `fe-cublas` and `reference` are for functional validation and run only where `--stack`
names one.

The reduced model keeps every layer width and reduces what the run prints as reductions: 2 of 64 layers (one linear-attention
and one attention layer), a vocabulary of 15,520 ids, seeded weights and a synthetic byte-level tokenizer. The weights are drawn
on the CPU generator and stored in the model's three formats, so the 8-bit and 4-bit paths are exercised. Two properties of
the seeded weights are choices, not accidents: the projections that write back into the residual stream are drawn at a quarter
of the unit scale, and the output head is a seeded permutation of the embedding table. The second makes the greedy token
decidable by a wide margin, and it means the produced ids are a walk through that permutation: with seeded weights the layers
move the logits, not the choice. The layers are verified block by block.

The stack `fe-cublas` is a functional-validation stack, not a fast path, taken by `--stack fe-cublas` alone. The
linear-attention step runs on cuDNN Frontend's fused kernel;
every multiply runs on cuBLAS over a bf16 copy of the quantised weights, which moves 2 bytes per weight element where the
stored formats hold 1 or 0.56; the rest runs in PyTorch; and every launch is issued from Python, one block after the other.
It exists to prove the loader, the loop, the state handling and the tie-out, and it is the correctness oracle of the
stack `native`. So is `fe-entries`, the cuDNN Frontend rows of `native` with every other block on a fallback row, and
`reference`, every block in PyTorch. A run of one of them says so in one line behind its load (`launched from Python:
fe-cublas is a functional-validation stack, taken by --stack alone ...`). Their timings are for orientation only.

### The recorded step

On the stack `native` the plain decode records its step once and replays it. The load ends in a stage of its own,
`capture`: one pass of the step's Python on PyTorch's capture stream, in which every launch of the step is recorded and
none runs -- every block's launch, the token choice, a copy of the chosen ids into the buffer the lookup reads, and
the adds that move the position rows on. From then on a token is one launch of that sequence and one host read of
4 bytes per sequence: no Python runs between two blocks. `--no-capture` launches every block from Python instead:
the same kernels, the same tokens.

What a recorded launch keeps is the address of every operand, so nothing a step reads may be selected by a host
integer. The token a step is fed and the rows of the position it stands at -- the position rows and slots the
attention preparation reads, the lengths the attention core reads -- stand at fixed addresses, set from the state's
tables and moved on in place by the step's own last launches. Four things a step launched from Python has for free
are code in front of every replay: the capacity test (a replay past the capacity is refused and launches nothing),
the ids the loop hands over (copied into the buffer the lookup reads, unless the loop hands over that buffer
itself), the position rows (one host integer says where they stand, and they are set again where it is not the
state's position), and the state (a sequence replays on the state it was recorded on, and a fresh state gets a
sequence of its own at its first replay).

The tensor a replayed step returns is that buffer: the step's own last copy has written the chosen ids into it, the
loop reads them from it, and handed back it is what the next step's lookup reads, as it then is. So a step is fed
the ids it is handed whatever they are: ids changed in place in the returned tensor are fed without a copy and
without a test that could miss the change, and any other tensor is copied in. The next step writes the buffer
again, so a caller that keeps the ids of a step past the next one copies them. What the state and the record keep
of a step are the ids as the token choice left them, in a tensor of the sequence's own.

Whether a step can be recorded is the plan's data, decided before anything is loaded: every block of the plan says
whether a recorded sequence can hold its calls -- an adapter to an entry or to a library call, the lookup and the
token choice can; a reference chain and a multiply over a copy cannot, because they form a table or a scalar on the
host per call or shape their launches by the position -- and only the plain decode step is recorded: one token per
sequence, no draft tokens, no drafter. A run prints exactly one line behind each load: `step recorded: ...`, with what
the recorded sequence holds, or `launched from Python: ...` with the reason -- a functional-validation stack,
`--no-capture`, a block of the plan on a fallback row, a step of speculative decoding, or a recording this
installation declined (the session is then left as it was, PyTorch's current stream put back, and the run goes on
from Python). The count of the first line is the CUDA runtime's where the installation states what the sequence
holds, and is then held against the plan kernel by kernel; elsewhere it is the plan's count, and worded so.

A replayed step is held to the same step launched from Python. The record of a run holds the ids the loop handed to
every step, and the checks launch those steps again from Python from a fresh state, each step behind the prompt on
the ids the step in front of it chose, as the loop feeds it: the ids fed, the ids chosen and the logits, byte for byte
(`recorded step`; with a checkpoint from position 0 through the first eight steps that feed on their own token and
across the first page boundary of the attention cache; the seeded run, every step). Behind the request the run reads
the step's own tail, which no recomputation sees: the ids buffer holds the last chosen ids and the position rows
stand where the request left the state. Under the kernel check a line holds two replayed steps of the traced pass to
the same steps launched from Python: the same kernels and copies, in their order and launch shapes, and behind them
the copy of the chosen ids and the position updates. A recorded step that does not compute what the steps launched
from Python compute is a failed check and a failed run, never a fall-back. The report prints the GPU-busy time of
both forms under the profiler, each with its launches, and states nothing about what bounds a step.

What a recorded sequence keeps: the tensors of the recorded pass stay in a pool of the sequence's own for as long as
it lives, in whole blocks of PyTorch's allocator. The memory sentence counts it (`QwenDense.recorded_step_bytes`: one
block of 2 MiB, and for the logits of a step a block of 20 MiB from 1 MiB on): 2 MiB at 1 sequence of the published
model and 22 MiB at 4 and at 8, 2 MiB at 1, 4 and 8 sequences of the two-layer model -- what PyTorch's reservation
grew by in runs at each of those counts on compute capability 10.0 (2026-11-24). It is an allowance from those runs:
at more sequences the activations of a step are not counted, and a run prints what its recordings took beside what
its plan counted (`recorded step, memory in a pool of its own`).

No CI lane of this repository runs a recorded step: the stack calls a library that the runners do not have. The cells
of the GPU test that hold it run where the library is installed.

## The stack `native`: every block on a kernel

`--stack native` is defined by its rows, not by what is installed: every block of the decode step runs on a kernel
that works on the stored weights. Seven are entries of cuDNN Frontend, four are calls of a library, FlashInfer, and
the token embedding lookup is PyTorch's gather. On compute capability 10.0 at 148 SMs, the class that ran, a step of
the 64-layer model is 484 launches on these kernels and no block is on a fallback row: 338 launches of cuDNN
Frontend -- the norm entry's 129, the projection entry's 64, the fused gate / up entry's 64, the fused step's 48, the
gate entry's 16, the attention preparation entry's 16 and the row quantise entry's 1 --, 145 calls of FlashInfer --
the MLP's 64 down multiplies, the 64 out-projections, the multiply of the output head and the attention core's 16 --
and PyTorch's gather. On the other 10.x classes the row of the in-projection does not serve, and that block stays on
its fallback row there. What this sample holds an entry to is its documented numerics contract (the band of its row)
and its launch facts.

The stack is built block by block. A block no kernel serves for a run -- on another GPU class, at another row count,
at other layer widths -- stays on a fallback row: the one it has in `fe-cublas`, or, for the attention core in
`fe-entries`, a row of these two stacks alone, the same reference chain with the reason that holds on a pool of
pages. Both tables of a run print such a row with its reason (`fallback: <reason>`).
Seven entries of cuDNN Frontend and four
calls of a library, FlashInfer, serve the stack:

| block | launches per step (64 layers) | what `native` launches | what the kernel check holds |
|---|---|---|---|
| token embedding lookup | 1 | PyTorch's gather | its name, its count and its place in the wiring order |
| residual add + RMSNorm + quantise | 129 | `cudnn.AddNormQuantDecode`, waiting for the launch before it and releasing the launch after it (`PDL True`, `PDL_RELEASE True`), in one of five forms by call site (the table below); compute capability 10.x | the entry's own symbol and launch shape, its count and its place in the wiring order |
| 8-bit in-projection | 64 | `cudnn.Fp8ProjectionDecode` on the stored 8-bit weight, the kernel stated whole: one K slice per weight tile of 128 rows, a ring of 8 stages, the weight stream ahead of the wait, 32 tensor-memory columns released late, waiting for the launch before it and releasing the launch after it (`SPLIT_K 1`, `MMA_M 128`, `STAGES 8`, `TMEM_COLS 32`, `PREWAIT_WEIGHTS True`, `HOLD_TMEM True`, `PDL True`, `PDL_RELEASE True`), at one of two weight shapes by call site (the table below); compute capability 10.0, 1 to 32 rows | the entry's own symbol and launch shape, its count and its place in the wiring order |
| linear-attention step | 48 | `cudnn.GdnDecodeFused` launched as a dependent of the launch before it and releasing the launch after it at entry (`PDL True`, `PDL_RELEASE True`) | the entry's own symbol and launch shape, its count and its place in the wiring order |
| MLP gate / up + SiLU x up + quantise | 64 | `cudnn.SwigluFp4Decode` on the stored 4-bit weight in the entry's own row order, the kernel stated whole: the token tile of the row count (8 tokens up to 8 rows), two weight stages issued ahead of the wait, the weight stream on the evict-first L2 policy, a persistent grid of at most 148 thread blocks, waiting for the launch before it (`TOKEN_TILE 8`, `PREWAIT_STAGES 2`, `L2_EVICT_FIRST True`, `MAX_CLUSTERS 148`, `PDL True`); compute capability 10.x, 1 to 64 rows | the entry's own symbol and launch shape, its count and its place in the wiring order |
| gate + quantise | 16 | `cudnn.AttentionGateQuantDecode` with 256 threads per thread block, releasing the launch after it at entry (`THREADS 256`, `PDL_RELEASE True`); compute capability 10.x | the entry's own symbol and launch shape, its count and its place in the wiring order |
| 8-bit out-projection | 64 | `flashinfer.gemm.bmm_fp8` on the stored 8-bit weight (`BACKEND auto`): the library chooses the kernel among its own; needs FlashInfer ("The rows a library serves" below); compute capability 10.x | its name, its count and its place in the wiring order; the kernel the library launched is printed, never gated |
| MLP down | 64 | `flashinfer.gemm.mm_fp4` on the stored 4-bit weight, with the library's CuTe DSL backend and outside any tuning context (`BACKEND cute-dsl`, `TUNED False`): the library's own pick for the shape, its split-K kernel class for this weight; needs FlashInfer; compute capability 10.x | its name, its count and its place in the wiring order; the kernel the library launched is printed, never gated |
| the 4-bit quantise in front of the output head | 1 | `cudnn.RowQuantizeFp4Decode` on the final norm's rows, at the model's hidden width, waiting for the launch before it and releasing the launch after it directly behind its own wait (`WIDTH 5120`, `PDL True`, `PDL_RELEASE True`); compute capability 10.x, rows 1-1,024, run on compute capability 10.0 at 148 SMs alone | the entry's own symbol and launch shape, its count and its place in the wiring order |
| output head | 1 | the same call of `flashinfer.gemm.mm_fp4` on the stored 4-bit weight, on the codes and the block scales the quantise in front of it wrote, handed over as they are: the library's persistent kernel class for this weight; needs FlashInfer; compute capability 10.x | its name, its count and its place in the wiring order; the kernel the library launched is printed, never gated |
| attention preparation | 16 | `cudnn.AttentionPrepareDecode` on the layer's cache as a pool of pages, the kernel stated whole: three position rows in the sections of the published rotary width, pages of 64 positions, 64-bit positions and slots, an ordinary launch (`POSITIONS 3d`, `SECTION (11, 11, 10)`, `PAGE_SIZE 64`, `POSITION_DTYPE int64`, `SLOT_DTYPE int64`, `PDL False`, `PDL_RELEASE False`); compute capability 10.x, 1 to 1,024 rows | the entry's own symbol and launch shape, its count and its place in the wiring order |
| attention core | 16 | `flashinfer.decode.trtllm_batch_decode_with_kv_cache` on the pool of pages the preparation wrote, through the page table and the lengths the state holds for the run's maximum length (`KV_LAYOUT HND`, with the library's own backend for these GPU classes): the library's paged decode attention, whose kernels are precompiled binaries; needs FlashInfer and its kernel-binary package ("The rows a library serves" below); compute capability 10.x, rows 1-1,024, run on compute capability 10.0 at 148 SMs alone. Not within one bf16 unit of the reference chain: its band is the bound of a stated model ("The attention core on the library" below). `--provider attention.core=torch_ref.attention` names the key that both chain rows have and selects, on each stack, that stack's own chain row: the same chain and the same bytes, printed with the reason of the stack that runs (an override takes the row of the run's stack where that row names the same call, and the first declared row of its key elsewhere) | its name, its count and its place in the wiring order; the kernel the library launched is printed, never gated |

The norm block is called at five kinds of site in a step, and its kernel is called in another form at each: the
first call of a step starts the residual stream from its operand, every later call adds to it in place; and each call
writes what the block after it reads. The provider row names the form of every call site (`NORM_FORMS` in
`model_samples/qwen_dense/configs.py`), the model's wiring names the site of every call (`QwenDense.sites()`), and a
host test holds the two to each other. The in-projection is called in one
form at two weight shapes, and the same two places name the shape of each call (`PROJECTION_SITES`):

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
codes, and the final norm writes bf16 rows only: the output head's operand is quantised by a launch of its
own. Each form is one compiled kernel of the entry and one entry instance of the adapter, which serves a call with the
instance of its site and decides nothing. The forms cannot be told apart by a launch record -- one kernel id, grid
(1, 1, 1), block (640, 1, 1) and 80 B of shared memory at 1 row in every form -- so the kernel check holds them where
they do show: the form of each site against the entry instance's own form, the instance that served each call of the
traced steps (the call log), and the kernel's symbol, which carries the form's text. A run prints which instance served
how many calls of a step: 1, 47, 16, 64 and 1 on the 64-layer model.

The sixth form of the table, `full_attention.first`, is the first call of a step whose first layer is an attention
layer. The published model starts with a linear-attention layer, so a step of it never calls that form (0 above): it is
the form of an attention layer's input norm with the residual mode of a first call. A reduced model that starts with an attention layer calls it -- `--layers 1`, 5, 9 and so on, because the
reduced model keeps the last layers of the published order -- and the run table below has that run.

The entry serves one hidden width, the published model's 5120, and the row says so (`widths` in `configs.py`). A
checkpoint of the family with another hidden width is not this row's: its norm block stays on the reference chain in
both stacks, on every GPU class.

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

Two fields of that form are performance fields of the entry: the tensor-memory columns and their release
(`TMEM_COLS 32`, `HOLD_TMEM True`). The row states them like every other field, so the entry never picks them.

The kernel multiplies on the stored format, so the loader keeps the merged 8-bit codes of the in-projections for a plan
with this row (one byte per weight element: 4.8 GiB on the published model). It keeps the bf16 copy of a weight only
where a block of the run reads one: a run in which every sequence count is this row's holds no copy of the
in-projections (9.7 GiB on the published model), and a run with a sequence count on the fallback row, or with the row
selected away, holds both. The reference blocks of the checks read copies; for these weights the checks make theirs for
one call at a time ("What a run holds of a weight" below).
The scale the kernel reads is the fp32 product of the operand's static scale and the weight's scale, one element built
on the host once per layer. The row claims compute capability 10.0 and 1 to 32 rows -- a limit that is kept as it is
until the row is run on another class; it has run at 1, 4, 9, 16, 17 and 32 rows (the run table below). On every other class and from 33 rows the block stays on its multiply over a copy
(`fallback: bf16 copy of 8-bit weights`), as it does for a checkpoint of the family whose layer widths give other
weight shapes, and with a weight merged at one scale per row (`merge_rule="own_scale"` of a session) the plan declines
the row by name: the entry reads one scale per weight. The 8-bit out-projection is not this row's: in a plain step a
library multiply serves it. In a step of several tokens per sequence the out-projection of a
linear-attention layer is this entry's as well, on rows of its own (the section "The verify step").

The MLP's first multiply is one call of `cudnn.SwigluFp4Decode`, one launch: the kernel
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
multiply over a copy (`fallback: bf16 copy of 4-bit weights`).

### The attention preparation: pages, slots and positions

The preparation of an attention layer is one call of `cudnn.AttentionPrepareDecode`, one launch.
Per row of a step the kernel normalises every query and key head row, rotates its rotary part by the row's position,
copies the gate rows beside the queries and writes the key and the value row into the layer's cache; the fallback row
is a chain of PyTorch operators, called once per token. The kernel writes a PAGED cache, and three words go with it.
A *page* is 64 positions of one sequence. The *pool* of a layer holds the pages of all sequences in one tensor, keys
in the first half of its last axis and values in the second; a sequence owns a run of pages for the whole run, because
the set of sequences is fixed. And a row is told where it goes by its *slot*, the flat index of its position's row in
the pool: sequence x pages of a sequence x 64 + position.

So a session whose plan has this row builds its attention cache as such a pool (`state.PagedKvCache`: the adapter says
which layout it needs), and with it the constant tables the kernel reads: the cos / sin row of every position, and for
the step that starts at a position the positions and the slots of its rows. The position of a step reaches the kernel
as a row of those tables, taken from the cache and from nowhere else. A text position is one number, so the three
position rows of the form hold the same value; and moving on by one position, or by the count a verify step kept,
launches and copies nothing. Every other stack, and a run this row does not serve, keeps the cache it had: two tensors
with one run of positions per sequence.

The adapter converts nothing: the entry reads four views of the in-projection's row (the queries with their gates, the
keys, the values), the layer's two norm weights, the table rows and the two halves of the pool, and writes the pool and
two output buffers of the call's own. Nothing is launched between the in-projection's kernel and this one. The
kernel check holds that on compute capability 10.0, where the in-projection is an exact row (up to 32 rows) and both
are judged one by one. Where the in-projection is on its fallback row -- on compute capability 10.3 and 10.7, and
past 32 rows -- its launches are attributed by family, and a PyTorch launch between the two would be counted among
them. The attention core behind it is the library's kernel in `native`, which reads the pool through the state's
page table and lengths ("The attention core on the library" below). In `fe-entries` it is still the reference chain: it
reads the pool through the cache's accessors -- the pages of a sequence as one run of positions, by views and host
integers alone -- as it reads the contiguous cache of the other stacks, and its launches are attributed to its
families.

The row states the kernel whole and claims compute capability 10.x and 1 to 1,024 rows -- a step of one token per
sequence or of several, one launch for all its rows -- at the published attention widths: 24 heads of width 256 on 4
key / value heads, a rotary width of 64 in the sections 11 + 11 + 10 (`widths` in `configs.py`; the sections are read
from a checkpoint's configuration). Past 1,024 rows, on compute capability 12.x and for a checkpoint of the family with another head width, rotary width or rotary section the block stays on the
reference chain, on a contiguous cache. The entry's own page lists compute capability 10.3 and 10.7
as not run on a device: on those classes a cell of this sample is its first run.

The kernel rounds at other points than the reference chain: its norm is fp32 arithmetic with an approximate root and
one rounding to bf16, and a rotated value is rounded once where the chain rounds it three times. Its band is therefore
the entry's own contract against a twin at the kernel's rounding points ("The checks of a run" below), and its bytes
are not the chain's: on the seeded model the ids hash is the stack `fe-cublas`'s, and a greedy run on a checkpoint follows
its first other token.

What ran with this row (2026-11-20; compute capability 10.0, 148 SMs; a shared GPU: correctness only; `fe-entries` in
PyTorch 2.13.0, `native` in PyTorch 2.14.0 with FlashInfer 0.7.2.post1). The default run of both scripts on the four
stacks, beside the same runs of the tree before the row. On `fe-cublas` and `reference` every line was the line it had
been but ONE, the environment line: its source hash now covers the entry's files, the contract of the paged pools and
the module a call of an entry runs in front of its launch, so that an edit of any of them shows in the line a run
prints. On `fe-entries` and `native` every ids
hash was the one it had been (`1509c0dd53ac`, and `11bfb9f9bcf5` for the speculative script), the logits were 1.00 bf16
unit from the reference's, inside the band of 4, and the kernel check passed, with 18 of the 20 block calls of a step on kernels on `native` and 13 of the 20 on `fe-entries`, and of a verify step 18 of the 20 and 14 of the 20. A traced step of the
two-layer model was 130 kernel launches on `fe-entries`, where it had been 179, and 69 on `native`, where it had been
118. The entry's line on `fe-entries` read 408,576 query and key elements at one sequence, none of them another byte
than the twin's, and 1,634,304 at four, 7 of them another byte and none outside its band. Verify sessions of 1 x 6, 16 x
2 and 4 x 8 rows, 80 plain steps at two sequences across the first page boundary, and seeded runs at 85 and at 1,024
sequences passed too: at 1,024 sequences one launch prepared the 1,024 rows of a step, 1,023 of 418,381,824 elements
were another byte than the twin's, none was outside its band, and one sequence alone wrote the bytes it wrote among the
1,024. The memory sentence of that day did not count what the checks of a run hold for every sequence: at 85 and at
1,024 sequences a run held more than the sentence named, and fit the GPU it ran on. The sentence counts that part since
2026-11-21 (the section "Support per GPU class" says what it counts and what it leaves out).

On the published checkpoint the performance mode passed on both stacks, its blocks judged at positions 0-15, 63, 64, 312
and 313 -- the two on either side of the first page boundary, on the pool -- and the kernel check of its traced steps
with 467 of the 484 block calls of a step on kernels on `native` and 338 of the 484 on `fe-entries`. Those two runs
were made before the mode observed its GPU. With the observation (2026-11-21, the same checkpoint, prompt and GPU
class, a shared GPU again) the mode passed on both stacks with the same judged positions and the same counts, and
its report carried the `GPU` lines; what a run observes of a shared GPU is no figure for this page. The greedy
text is not the text of the run before the row. On either stack the plain run of the default prompt takes another token
at its 12th new token, where the two highest logits are the same two tokens in the other order: on `fe-entries` they
were 2.0 apart before the row and are 0.25 apart with it, while the two row sets differ by 1.6 on average over the
twenty highest logits of that step and by 3.6 at most; on `native` 0.375 and 0.125, against 0.9 and 1.8. The reference
blocks, fed the same tokens, have them 0.625 apart. The speculative recipe of the section "Speculative decoding" ended
in `[02] PASS` at the end token, twice with the same figures on each stack: on `native` 230 new tokens in 70 steps (3.30
kept tokens a step; 12, 18, 12, 4, 13 and 11 steps kept one to six tokens), on `fe-entries` 254 in 76 (3.36; 16, 15, 13,
7, 8 and 17), where the runs before the row wrote 231 in 71 and 204 in 63. Its kernel check passed over 16 traced steps,
with 467 of the 484 block calls of the verify step on kernels on `native` and 386 of the 484 on `fe-entries`.

Seven runs with one thing wrong each, every one in a fresh interpreter (`fe-entries`, one sequence). A row that names one
position row passed its checks, with 12 of the 20 block calls of a step on kernels. A row that names
pages of 16 positions ended in `[00] ERROR`: the entry declined the pool at its first use. A slot table one row late, a
position table one row late, a twin with its rotary halves exchanged and a write accessor of the cache that exchanges
two pages each ended in `[00] FAIL` on the entry's line. The READ accessor with two pages exchanged -- what the
attention core reads -- ended in `[00] PASS`: a run's own checks hold what the kernel wrote; none of them compares
what that chain reads with anything, and the layer moves the seeded logits by less than the band. The read accessor
is held by tests, not by a run: by the host case of a session on stand-ins, whose tokens differ, by the host case of
the paged cache against the plain one, and by the two cells of the GPU test that read the pool by index arithmetic
of their own. With the defect planted, the host case of the paged cache failed and so did the cell of the reference
chains, on compute capability 8.0; the cell of 80 steps reads the pool the same way and was not run with it. A check
on a device of what the attention core reads is the row of the core's own kernel in `native`: that kernel reads the
pool through the page table and the lengths, and every call is judged against the reference chain ("The attention
core on the library" below).

What a consumer needs beyond a kernel's bytes is launched by the consumer, never inside the producer's call. The MLP's
down multiply reads the codes of the gate / up kernel with their block scales in the layout the kernel writes
(128 x 4): the library's multiply reads that layout and is handed them as they are (on its fallback row, in
`fe-entries`, the multiply over a copy gathers them when it dequantises). And the final norm's form ends at the bf16
rows: the 4-bit operand of the output head is formed from them by a block of its own, `output_head.quantize`, with a launch of its own -- on `native` and
`fe-entries` by the row quantise entry, whose codes and block
scales the head's multiply is handed as they are; everywhere else by the reference's quantiser.

So a run with a checkpoint prints under its kernel table:

```text
per step: 484 launches on 12 exact rows, checked one by one in the wiring order; the launches of 0 fallback calls and of the host path are attributed by family
```

On `--stack fe-entries`, the cuDNN Frontend rows alone:

```text
per step: 339 launches on 8 exact rows, checked one by one in the wiring order; the launches of 145 fallback calls and of the host path are attributed by family
```

and its kernel table names the four blocks on fallback rows, each with its reason. The default run on the two-layer
model prints 20 launches on 12 exact rows, and `fe-entries` 14 on 8 with 6 fallback calls.

**What is checked**, in every run of this stack (its kernel check cannot be switched off), for every launch of the
traced steps: the provider row's form is the entry's own form, compared on the objects; a launch of an entry carries
the entry's own symbol and exactly the launch shape the entry states -- at 1 row on compute capability 10.0 with 148
SMs grid (96, 1, 1), block (256, 1, 1) and 42,112 B of shared memory for the fused step, grid (3, 1, 1), block
(256, 1, 1) and no shared memory for the gate, grid (1, 1, 1), block (640, 1, 1) and 80 B for the norm in every form,
grid (128, 1, 1) or (112, 1, 1) by the weight's shape, block (256, 1, 1) and 139,520 B for the in-projection, grid
(1, 1, 148), block (192, 1, 1) and 231,424 B for the fused gate / up step --; every exact kernel is there once per
call, in the wiring order; and a kernel of a library, which no entry states, is held by its name, its count and its
place, with the kernel the library launched printed on the line of its row. For the norm block and the in-projection
all of it holds per call site. A
difference fails the run with a sentence that names the block, and for a form the call and its site.

**What this stack is not, today.** Its step is recorded and replayed on compute capability 10.0 alone, the class on
which every block of it is a kernel: where a block is on a fallback row the step is launched from Python, and the run
says so. And where a neighbour is on a fallback row a kernel's launch knobs change nothing: the
fused step, the norm, the in-projection and the quantise of the head's operand wait for the launch before them and
release the launch after them, the
fused gate / up step waits for the launch before it, the
gate releases the launch after it, but around a kernel that sits between fallback rows the launches are
PyTorch's and cuBLAS's, which neither release early nor start early, so there both knobs are exercised and change
nothing. Kernels of the stack meet at the gather and the first norm of a step and, on compute capability 10.0, in the chain
of a layer's input: the norm releases at its own wait, the in-projection behind it starts on that release, streams
its weight and waits for the norm before it reads the 8-bit operand; in a linear-attention layer the fused step
starts on the in-projection's release and waits before it reads the projection's rows. They meet in every MLP as
well, on each 10.x class: the MLP's norm releases at its own wait, the gate / up kernel behind it starts on that
release, issues two stages of its weight and waits for the norm before it reads the 4-bit operand, its block scales
and the two scalars. The kernel signals the launch after it before its last output tile is written (it has no knob
for that). In `fe-entries` the down multiply behind it is a fallback row, launched without the attribute, and starts
when the kernel has completed; in `native` it is the library's kernel, which the library launches with programmatic
dependent launch at its own default and which waits for the launch before it at its start. The library launches its
8-bit multiply the same way: in `native` it starts on the release of the fused step or of the gate, and waits. And it
launches its attention the same way behind the attention preparation's kernel, an ordinary launch that releases nothing
early: the kernel waits before it reads the queries and the pool; that it waits in front of its read of the page table
and the lengths is not established, so those two are constants of the state that no launch writes. Kernels of the stack
meet at the end of a step too, on each 10.x class: the final norm releases at its own wait, the quantise of the head's
operand behind it starts on that release, loads its one scale -- a constant the adapter formed on the host -- and
waits for the norm before it reads the rows; it releases the launch after it directly behind its own wait. In
`fe-entries` the head behind it is a fallback row and starts when the kernel has completed; in `native` it is the
library's multiply, which on compute capability 10.0 starts on that release and waits before it loads the codes and
the block scales. Each
row of `configs.py` says what the launch before it must not write and what the launch after it must wait for. With `--provider
linear_attention.step=fe.gdn_decode_fused` the stack runs the fused step in its `fe-cublas` form instead (no wait, no
release); both kernel tables then print the row in that form.

`--stack fe-entries` is every cuDNN Frontend row of `native` with each block that a library serves there on its
fallback row. It needs no package beyond cuDNN Frontend and PyTorch, and it is the stack the GPU test
of this area runs, so the rows of the entries stay verified wherever that test runs. The two stacks differ in four
rows: the MLP's down multiply, the output head, the 8-bit out-projection and the attention core, which `native` hands
to FlashInfer; `fe-entries` keeps the three multiplies on the multiplies over a copy of `fe-cublas` and the attention
core on its reference chain.

### The rows a library serves

Three blocks of the stack are multiplies of a library on the stored weights, and `native` has a row for each on compute capability 10.x: `flashinfer.mm_fp4` for the MLP's down multiply, `flashinfer.mm_fp4_head`
for the output head (the same call on the vocabulary's weight), and `flashinfer.bmm_fp8` for the out-projection (in a verify step behind the projection entry's
rows, which keep the linear-attention layers' call). A fourth is the attention core, `flashinfer.decode_attention`: the
library's paged decode attention over the pool of pages the attention preparation writes ("The attention core on the
library" below). FlashInfer is a package of this stack alone: the sample's own stack calls it, the `nvidia-cudnn-frontend` package does not depend on it. It is
imported inside the adapters of the rows that call it, on a device; no other stack and no plan on stated device facts
imports it. Such a plan is made through the helper layer (`model_samples.blocks.plan` with a `DeviceFacts` of your
own; the script has no flag for it) and shows these rows without the package -- as `unverified in CI`, because no
requirement file of this repository's tests names the package. Install it with

```bash
pip install -c samples/models/constraints-native.txt -r samples/models/requirements-flashinfer.txt
# the library's kernel-binary package, at the library's own version: published on the library's release index, not on PyPI
pip install --no-deps --index-url https://flashinfer.ai/whl -c samples/models/constraints-native.txt -r samples/models/requirements-flashinfer-cubin.txt
```

The second line installs the kernels of the attention core: precompiled binaries, which a run reads and never fetches
("What the first run does"). Each requirement file is a lower bound; the constraints file pins the versions the stack
was last run and checked with -- PyTorch 2.14.0, CuTe DSL 4.8.0, FlashInfer 0.7.2.post1 and apache-tvm-ffi 0.1.11 -- and
the kernel-binary package `flashinfer-cubin` 0.7.2.post1, which the library reads at its own version alone. So in an
environment that has another PyTorch the line replaces it, and its cuDNN with it: an environment of its own is the safe
place for `native`. FlashInfer requires `nvidia-cudnn-frontend` itself; run `pip install -e .` again if pip lists
that package among what it installed. The adapters hold the version of FlashInfer alone, to a validated one
(`VALIDATED` in
`model_samples/providers/flashinfer_calls.py`). Without either package a run on compute capability 10.x stops before
anything is loaded: `[00] ERROR the stack native cannot run:` and one sentence that names the package that is
missing, its command above and `--stack fe-entries`, the functional-validation stack that runs the cuDNN Frontend rows
without them; exit code 1, and no other stack takes the place of the sample's own. With the packages installed and
with another version of one it ends in `[00] SKIP` before anything is loaded, exit code 0, by a sentence that names the
version. On compute capability 12.x no row of the library serves, and the stack runs without the package.

The 4-bit multiply is held exactly. The library picks the kernel of a call itself, by the shape and the GPU -- one of
two kernel classes, in a grid of its own choice -- and that pick is a function of the call as long as nothing has
tuned the operation in the process. So the row says `TUNED False`, the kernel check holds that form on every call, and the line of the row names the
kernel the library launched -- its class and with it the shape the library picked: printed, never gated. The same call inside a
tuning context of the library (`flashinfer.autotune`), or after one has tuned the operation in the process, is not the
row's form: the call log reads `TUNED True`, and the kernel check fails with the form it found beside the form the row
names. `TUNED` is a statement about this process: the library's tuner is tuning, or has tuned the operation here. A
tuning store that a caller attached to the library is not seen by it (the sample attaches none); a tactic replayed from one shows in the kernel the run names on the row's line, and is not gated. The 8-bit
multiply is the other case, by design: which kernel serves `bmm_fp8` is the library's choice among its
own -- its tuner's where one has run, else its first candidate -- so the kernel check holds the launch as one of the kernels the call can launch, prints the kernel that was launched on the row's line, and never fails on which it is. A member is a cuBLASLt kernel of 8-bit operands or the library's own CUTLASS multiply of 8-bit operands; a long
kernel name is printed with its length and a digest, so that two such names are told apart. In every run of this page the library launched its own CUTLASS multiply there.

### The attention core on the library

In `native` on compute capability 10.x the attention core is one call of the library per attention layer and step,
`flashinfer.decode.trtllm_batch_decode_with_kv_cache`, the library's paged decode attention, with the library's own
backend for these GPU classes and `kv_layout="HND"`: one launch for all rows of the step. The adapter converts nothing and launches nothing. It hands the library the query rows as
the attention preparation wrote them, the two halves of the pool of pages, the rows of the state's page table and the
lengths of the step (`state.PagedKvCache.attended`), the run's maximum length, the chain's scale `256 ** -0.5` as a
number of the host, a workspace, a zero counter buffer of the library's own size and an output buffer of the call's
own. The page table is `[sequences, width]` of 32-bit entries: the width is the pages of the run's maximum length
(`--max-model-len`; 2,051 by default, 33 pages) rounded up to a multiple of four, a column past a sequence's own pages
names its last own page, and a slack row follows the last sequence, so that every entry a kernel could read names a
page of the pool. The lengths are what each sequence holds after the step. The workspace is 394 MiB, one per device
and process, and the memory sentence counts it once.

The library checks almost nothing of what it is handed and reads it all the same, so the adapter checks: lengths or a
page table that are not 32-bit words, a strided view of either, a table of another width than the maximum length's,
one off a 16-byte boundary or on another device, a strided output and a step past the maximum length are each a
`ValueError` that names the operand. A table of 64-bit words gives wrong bytes without an error at two sequences and
the right bytes at one sequence of one page, which is why the host case holds it at two sequences of two pages.

**The row is not within one bf16 unit of the reference chain.** The kernel documents no numerics contract, and its
rounding points are not established. Its band, `attention_bf16_weights`, is the bound of a stated model of the
kernel's arithmetic -- every softmax weight and the divisor carry one rounding at eight significant bits, the output
one rounding to bf16 -- declared before any run of the row and never taken from a measured distance. An element is
inside when it is finite and within `2**-8 x (spread + |chain|)` plus one spacing of bf16 values at the chain: `chain`
is the reference chain's value before its rounding, `spread` the weighted magnitude of the values it attends to. A run
prints the band once and, for the block, the largest share of its bound an element used (printed, not gated); an
element outside is a finding, never a wider band. The call writes nothing, and the pool is held byte for byte at every
call. What the band cannot see is one wrong, missing or extra row among thousands whose softmax weight is below the
band's term: a host case pins a value row off by one slot among 2,051 attended rows as inside.

**Which kernel is launched follows the run.** The library has several kernels for this attention and chooses among
them, and the grid, by the run's maximum length, the sequences, the tokens per sequence and the GPU's SM count; the
sample derives none of it. The kernel check holds the launch by the family's name, its count and its place, and the line of the row names the
kernel the library launched: for one sequence at a maximum length of 2,051 on compute capability 10.0 with 148 SMs, in
a decode step and in a verify step of six tokens, the variant `MultiCtasKvCgaVarSeqQ8Kv256StaticGroupedSwapsAbForGen`,
in grid (5 x T, 4, 1) at T tokens per sequence, block (512, 1, 1), with 178,456 B of shared memory. The row's `compute capability
10.x` says where the table selects it, not where it ran: every run of this row is on compute capability 10.0 with 148
SMs. On another 10.x class, or another SM count, the run holds that class's kernel binaries, judges the block in its
band and holds the launch by the family's name, as on every class. Every run prints the kernel the library launched on
the row's own line of the kernel check (`.. by name; the library launched <kernel> x <n> over the traced steps`):
printed, never gated. The order is held as for every exact row: nothing may be
launched between the preparation's kernel and this one, or between this one and the gate's.

The library states no batch invariance for this call and the sample states none: the bytes of one sequence alone can
differ from the same sequence among several, where another kernel or another split of the key rows serves the two. The
batch line counts the block among those that are equal or differ, gates nothing on it and prints that sentence
beside it.

### The output head's quantise on its entry

The 4-bit operand of the output head is formed by a launch of its own, behind the final norm.
This sample has a block for it, `output_head.quantize`, between the final norm and the head in every stack, and on
`native` and `fe-entries` its row is `cudnn.RowQuantizeFp4Decode` (compute
capability 10.x, 1 to 1,024 rows, hidden width 5120; run on compute capability 10.0 at 148 SMs alone). Above 1,024
rows, on another GPU class or with another hidden width the block is the reference's quantiser, as on `fe-cublas` and
`reference`: the table of such a run reads `torch_ref.quantize` with `fallback: reference chain` for it, and its
kernel table holds the block's launches by PyTorch's family.

What the row is and is not. The entry's kernel is one thread block per row, and the kernel check holds its launch to the launch shape the entry states. It writes the block scales of the live rows: the sample passes buffers of its own, in whole bands of 128 rows, so the padded rows of its scale buffer are not written, and nothing reads them. What a run holds is the entry's documented
contract, the band `row_quantize_fp4`: every block-scale byte that of the kernel's rule, every code one of the two
adjacent codes its two approximate reciprocals admit. The reference chain's own codes are not inside that band -- it
writes the code 0 for a negative value that rounds to zero, where the kernel keeps the sign -- so the block is
compared with a twin written at the kernel's rounding points, not with the chain.

What ran with this row (2026-11-23; compute capability 10.0, 148 SMs; a shared GPU: correctness only; `native` in
PyTorch 2.14.0 with FlashInfer 0.7.2.post1 and its kernel-binary package, every other stack in PyTorch 2.13.0). The
default run of both scripts on the four stacks, beside the same runs of the tree before the change that brought this
row. That change also put the attention core of `native` on the library's kernel and counts the checks of the
sequences beyond the first into the memory sentence, so lines of those two moved with this row's. Every ids hash was
the one it had been (`1509c0dd53ac`, and `11bfb9f9bcf5` for the speculative script). On `fe-cublas` and `reference`
every line was the line it had been but these: the environment line's source hash, which now covers the entry's files;
one row more in every served-by table, `output_head.quantize  1  PyTorch  torch_ref.quantize  any  fallback: reference
chain`; the by-provider line, `fallbacks 9 (cuBLAS over copies 4, PyTorch chains 5)` where it read `fallbacks 8
(cuBLAS over copies 4, PyTorch chains 4)` on `fe-cublas`, and `fallbacks 10 (PyTorch chains 10)` where it read
`fallbacks 9 (PyTorch chains 9)` on `reference`; the block calls of a step, 21 where they were 20 (111 where they were
110 in the speculative script); the batch-invariance line of `fe-cublas`, which counts one block more; and
the memory sentence of the plain script on `fe-cublas`, which names the part of the checks and asks for about 3.9 GiB
where it asked for about 3.7. On `native` the block read 18,240 block scales at one sequence, none another byte than
the twin's, and 291,840 codes, 581 of them the second code of their bracket and none outside (72,960 and 1,167,360 at
four sequences, 2,181 second codes); the head behind the entry was the bytes of its twin, and the kernel check passed:
a traced step of the two-layer model was 22 kernel launches and 1 copy, where it had been 69 and 4 with the attention
core and this block on their reference chains, and 20 of the 21 block calls of a step were on kernels,
where 18 of 20 had been. On
`fe-entries` the block read 556 and 2,326 second codes, none outside, and a traced step was 93 kernel launches, where
it had been 130, with 14 of the 21 block calls of a step on kernels. No logits line of the plain script moved on `native`; on `fe-entries` the
head's line read 23.2% of elements one bf16 unit from the reference's at four sequences, where it had read 23.3%, at
the same largest distance of one unit. In the speculative script the head's line of the verify steps read 23.2% on
`fe-entries` as well, where it had read 23.3%; and on `native` the largest logit distance of the kept tokens from the
decode steps of the same tokens read 0.52 bf16 units, where it had read 1.00.

The row's range, both sides of its boundary: at 1,024 sequences one launch of grid (1024, 1, 1) quantised the 1,024
rows of a step -- 298,844,160 codes, 559,695 of them the second code of their bracket on `native` and 595,654 on
`fe-entries`, none outside -- and both runs passed; at 1,025 sequences on `native` the block was the reference's
quantiser, the library's head multiply read its row-by-row block scales, and the run passed its bands and its kernel
check. A verify session of 16 sequences of five tokens quantised its 80 rows in one
launch of grid (80, 1, 1) on both stacks, none of 4,096,000 codes outside. With the two adapters of a `native`
session called block by block, at 1 row and at 129, the head's logits were the same bytes with the padded rows of the
scale buffer as the adapter returned them, set to zero, and set to a pattern that is no finite value. Two runs with
one thing wrong each failed: with the scale handed un-inverted, by the block's own line (18,240 of 18,240 block
scales not the twin's byte); with the scale buffer zero-filled in front of the launch, by the kernel check, which
named PyTorch's fill kernel between the final norm's launch and the entry's.

The checkpoint, one sequence, the default prompt. On `native` the run passed with 244 new tokens (ids hash
`7d767f68e358`, ended at the end token): 484 launches of every one of the 32 traced steps on exact rows, in the wiring order; 486 kernel launches and 2 copies per traced step in all, where
there had been 638 and 5 with 467 launches on exact rows; the block over 16 calls 5,120 block scales, none another byte than the
twin's, and 81,920 codes, 41 of them the second code of their bracket, none outside; the memory sentence asked for
about 28.3 GiB and the device grew by 24.48 GiB, where they had been 27.9 and 24.09. The tokens are not those of the
tree before the change, which decoded 216 to the end token (ids hash `265cd5861efc`): they moved with the attention
core's library row, whose kernel is inside its band and is not the reference chain's bytes (runs of 2026-11-22, in
which this row moved no token). On `fe-entries` the run passed with the 256 new tokens it had before (ids hash
`f0c662ba6d0b`): 339 launches of a step on exact rows; 35 second codes, none
outside. The performance mode's `kernels` lines read `484 launches per step on exact rows: cuDNN
Frontend 338, FlashInfer 145, PyTorch 1` on `native` and `339 launches per step on exact rows: cuDNN Frontend 338, PyTorch 1` on
`fe-entries`. The speculative recipe of this page (five drafted tokens, its draft vocabulary, 256 new tokens) passed
twice on `native` with equal figures -- 2.89 kept tokens per step over 80 steps, ids hash `e40d030b6c26` --, its
verify step 484 launches on exact rows, where the tree before the change kept 3.30
tokens per step over 70 steps of another text with 467 launches of a verify step on exact rows; and once on `fe-entries`, with the tokens
and the figures it had before -- 3.36 kept tokens per step over 76 steps, ids hash `bc7666f08a83` --, its verify step
387 launches on exact rows.

### What the first run does

FlashInfer builds what a call needs when the call is first made: the kernel of a 4-bit multiply is compiled in the
process, and the 8-bit multiply is a module the library compiles with the installed
CUDA compiler and ninja and keeps in its own cache directory (`.cache/flashinfer` under the home directory, or under
the directory `FLASHINFER_WORKSPACE_BASE` names) -- more than a minute on the machine these rows were
written on, once per machine. A run of `native` therefore makes each call of a library row on scratch operands while
it plans, after the memory sentence has held and before anything is loaded: at the run's row count and, for a run of
several sequences, at the rows of one sequence, which its batch-invariance check calls (the kernel of the 4-bit
multiply is compiled per range of row counts). It says so
in one line, `[00] planning: first use of FlashInfer <version>: ...`, and the `plan` figure of its stage line is what
that cost. A call the library refuses for the operands the adapter built ends the run in `[00] ERROR`; whatever else
keeps the library itself from importing, building or launching there ends it in `[00] SKIP`, with the type of the
exception and nothing loaded. A fault of the adapter's own code is neither: it is raised as it is.

Of that first use a run prints its own lines and none of the library's informational ones. FlashInfer logs through
loggers of its own, at the level the variable `FLASHINFER_LOGGING_LEVEL` names when the library is imported, and at
`info` where the variable names none: at that level it prints a line on standard error for every module it builds with
the CUDA compiler (the 8-bit multiply's and the launcher of the attention among them) and two for every kernel of the
4-bit multiply it compiles and keeps, the second with a path of its cache directory. Before it imports the library a run
sets that variable to `warning` for its process and for every process it starts, where the environment names no level
-- the variable unset, empty, or blanks alone --, so none of those lines is printed; the library's warnings and errors
still reach standard error. A level that is set in the environment stays: with `FLASHINFER_LOGGING_LEVEL=info` the
library prints its lines, some with a path of its cache directory. A text there that is no level name of Python's
logging module ends the run in `[00] ERROR` while it is planned, before the library is imported, in a line that names
the variable: the library's import would raise on it.

The kernels of the attention core are not built: they are precompiled binaries the library reads from files -- from
the directory `FLASHINFER_CUBIN_DIR` names, else from its kernel-binary package `flashinfer-cubin` where that has the
library's own version, else from its cache directory -- and a file it does not find it would download. A sample never
lets it. Before the library is first reached the run switches that download off for its process and for every process
it starts (`FLASHINFER_NO_DOWNLOAD`), and says so in a line of its own, `[00] planning: first use of the paged decode
attention of FlashInfer <version>: ...`: a kernel binary comes from the package or from a directory filled
beforehand, never from a fetch inside a run. The library does not always refuse a binary it lacks: where the missing
file is not the first one its kernel selection reads, it raises nothing, prints lines of its own with a path of its
installation and launches another kernel of the set. So the sample does not rely on the refusal. Before its first
call of the attention it reads the manifest and the header of the library's kernel set, holds the manifest to the
SHA-256 the library itself names for it, and holds every kernel binary
the header's table names for this call -- the data types, the head width, the paging and the page -- on the GPU's
class to the SHA-256 the manifest lists for it: 44 binaries on compute capability 10.0, more than a run reads, because
which of them a step reads is the library's selection and the sample models none of it. A binary that is missing or
is not the manifest's file, or a manifest that is not the library's, ends the run in `[00] SKIP` before the library's
attention is called, so a run that passes
cannot meet a missing binary at a later step. Then the library builds a launcher with the installed CUDA compiler and
ninja, once per machine and kernel-binary directory, and the plan makes one scratch call per step shape of the run --
its sequences and one sequence alone, at the tokens per sequence of each of its plans, at the run's maximum length.
What a run does by what the installation holds:

| the installation | the run |
|---|---|
| the package at the library's version, the directory variable unset | holds the package's files, then reads them |
| neither the package nor the variable, and the library's cache directory without a file of the set | `[00] SKIP`: the package is not installed, with its command |
| the variable set, no package, and the directory without a file of the set, or with a manifest that is not the library's | `[00] SKIP`: the directory does not hold them -- how many are missing or changed, or that its manifest is not there or is not the library's --; fill it, or install the package |
| the package at the library's version, with a binary of the set missing or changed, or a manifest that is not the library's | `[00] SKIP` that says how many of the set are missing or changed, or that the manifest is not the file the library names, and no command: the line above leaves an installed package as it is. Install it anew, with `--force-reinstall` added to that line |
| the package at another version than the library's | `[00] SKIP` at plan time with both versions and the command: the library would ignore the package |
| the variable set while the package is installed | `[00] SKIP` at plan time that names the variable and the package: take one of the two away |
| the package's metadata without its module, or its module without the metadata | `[00] SKIP` at plan time that says which |

Every such `SKIP` comes before anything is loaded, with exit code 0, and names no local path; in the last three the
library is not imported. Where the package cannot be installed, fill a directory beforehand and name it with
`FLASHINFER_CUBIN_DIR`, on a machine without the package: with the library's own command `flashinfer download-cubin`,
which fetches every file of every kernel set of the library (this page has not run it), or by copying such a directory
from a machine with the same GPU model -- the binaries are per architecture, and the choice among them follows the
SM count. The run then holds and reads that directory with the download switched off, and a file of the set that is
missing there is the `SKIP` of the table: a directory that holds only what earlier runs of the library fetched for
their own steps does not hold the set.

On compute capability 12.x neither the gate entry nor the norm entry serves; both blocks then stay on their reference
chains (`fallback: reference chain`) in both stacks. The row of the
projection entry claims compute capability 10.0 only, the rows of the fused gate / up entry compute capability 10.x:
on 12.x both multiplies are over copies.

The rows of the table below are runs made before the attention preparation had its row: in each of them that block ran
on its reference chain, over one contiguous cache, and the counts of a row are those of that plan (the runs with the row
are in the section on the attention preparation, above). With its library rows the stack ran on compute capability 10.0
on 2026-11-18 (a 148-SM part, PyTorch 2.14.0,
CuTe DSL 4.8.0, FlashInfer 0.7.2.post1: the versions `constraints-native.txt` pins): the first two rows of the table.
Every other row is a run of 2026-11-17 on the cuDNN Frontend rows of the stack, as `--stack native` selected them
then and `--stack fe-entries` selects them today without the attention preparation's row (PyTorch 2.13.0): the three library rows were fallback rows in those runs, and the counts
of launches, copies and weights in them are those of `fe-entries`. Where every sequence count of a run is
at most 32 and no row is selected away, the projection row and the gate / up row serve each count: the run holds the
in-projections and the gate / up weights on their stored codes alone, and the reference of its checks reads bf16
copies made on demand. From 33 sequences the in-projections have their copies, and from 65 the gate / up weights:

| run | result |
|---|---|
| the default run with the library rows, `--stack native` | `[00] PASS` at 1 and at 4 sequences, with the ids hash of the stack `fe-cublas`. A traced step of the two-layer model is 118 kernel launches and 4 copies; 17 are judged one by one: the ten launches of the cuDNN Frontend entries in the launch shape their entries state, the gather and the library's launches by name -- the two down multiplies in the library's kernel class `Sm100BlockScaledSplitKGemmKernel`, grid (40, 1, 2), block (192, 1, 1), 231,424 B of shared memory. The three library rows were inside the band of the reference multiply: the out-projection and the down multiply over 114 calls each, the largest distance one bf16 unit on fewer than 0.05 % of the elements; the output head over 57 calls, every element equal. The kernel the library chose for the out-projection was its own CUTLASS multiply at both call sites. The run kept all 9 weights of its multiplies on their stored format alone, and at 1 sequence the reference read 1,035 copies made on demand |
| the checkpoint with the library rows, `--stack native`, one sequence, the default prompt, 32 new tokens | `[00] PASS`: 451 launches of every one of the 32 traced steps on exact rows, in the wiring order; the library launched the 64 down multiplies in its split-K kernel class and the output head in its persistent one, grid (1, 1, 148). 1,422 kernel launches and 5 copies per traced step in all. Replay held; 8 of 8 provider blocks inside their bands; the 257 weights held the bytes of their load before and after the checks, and the reference read 12,850 copies made on demand, the largest 2.37 GiB; the token line reported. The memory lines are in the section on checkpoints |
| the default run, `--stack native` on 2026-11-17 | `[00] PASS` at 1 and at 4 sequences, with the ids hash of the stack `fe-cublas`. A traced step of the two-layer model is 179 kernel launches and 13 copies (303 and 25 with the gate / up block on its fallback row); 12 are judged one by one.  The norm entry was inside its band in the four forms this model calls -- over 285 calls at 1 sequence 583,680 normed elements, 4 of them on the other bf16 value around the normalised value, and 1,203,840 codes and block scales, each that of the nearest value; nothing outside -- and equal alone and among 4. The projection entry was inside the band of the reference multiply at both weight shapes (114 calls; the largest distance one bf16 unit, on fewer than 0.05 % of the elements, where the multiply over a copy it replaces is one unit off on 25.9 %), and equal alone and among 4. The fused gate / up entry was inside its band over 114 calls: at 1 sequence 124,032 block scales, each the byte of the fp64 chain, and 1,984,512 codes, 1,927 of them another code than the fp64 chain's, the farthest at 89 % of its bound; at 4 sequences 7,758 of 7,938,048 codes; nothing outside, and equal alone and among 4. The gate entry was inside its band (350,208 bytes at 1 sequence, each the code of the nearest bf16 value; 1,400,832 at 4, one of them the code of the other value around the gated value, inside the margin), and equal alone and among 4. At each count the reference read 460 copies of in-projection and gate / up weights made on demand (four per step and reader, 0.33 GiB the larger), and the four weights held the bytes of their load before and after the checks (`stored weights`) |
| the four-layer model, `--stack native --layers 4 --sequences 1` | `[00] PASS`: all five forms a 64-layer model calls (the input norm of a later linear-attention layer is the fifth), four in-projections, three at 16384 x 5120 from one instance and one at 14336 x 5120 from the other, and four gate / up calls; 22 launches of a step judged one by one, each entry's in the launch shape the entry states. Its memory sentence asked for about 4.4 GiB; PyTorch reserved 4.4 GiB and the device grew by 4.6 GiB -- on the reduced models the sentence is an estimate (the last section has the runs) |
| the one-layer model, `--stack native --layers 1 --sequences 1` | `[00] PASS`: the sixth form, `full_attention.first` -- the model is one attention layer, so the first norm of a step is its input norm. 7 launches of a step judged one by one (the gather, three norm calls, the in-projection at the attention layer's weight shape, the gate / up call and the gate), each entry's in the launch shape the entry states; the three norm calls served by three instances, and the symbol of the first carries its form (`store`, `e4m3_static`, wait + release). The norm entry inside its band over 171 calls: 291,840 normed elements, 4 of them on the other bf16 value around the normalised value, and 601,920 codes and block scales, all those of the nearest value; the projection entry inside its band over 57 calls; the gate / up entry inside its band over 57 calls |
| the checkpoint, one sequence, the default prompt, 32 new tokens | `[00] PASS`: 322 launches of every one of the 32 traced steps on exact rows, in the wiring order, each entry's in the launch shape the entry states; 2,908 kernel launches and 200 copies per traced step in all (6,876 and 584 with the gate / up block on its fallback row). The call log of a step: the five forms of the norm entry served 1, 47, 16, 64 and 1 calls, each from its own instance, and the two instances of the projection entry 48 and 16. Replay held; 8 of 8 provider blocks inside their bands; the norm entry over 2,064 calls with 4,014,080 normed elements, 40 of them on the other bf16 value, and 10,813,440 codes and block scales, 3 of them not those of the nearest value, nothing outside; the projection entry over 1,024 calls, the largest distance one bf16 unit on fewer than 0.05 % of the elements; the fused gate / up entry over 1,024 calls with 1,114,112 block scales, each the byte of the fp64 chain, and 17,825,792 codes, 12,528 of them another code than the fp64 chain's, the farthest at 96 % of its bound, nothing outside; the gate entry over 256 calls with 1 of its 1,572,864 bytes the code of the other bf16 value around the gated value, inside the margin; the token line reported. The reference read 6,400 copies of in-projection and gate / up weights made on demand, and the 128 weights held the bytes of their load before and after the checks; the memory lines are in the section on checkpoints |
| the default run, `--stack fe-entries` | `[00] PASS`, the lines of the default run of 2026-11-17 above; also as the L0 cell of the area's GPU test |
| `--stack native --provider linear_attention.step=fe.gdn_decode_fused`, 1 sequence | `[00] PASS` |
| `--stack native --provider projection.fp8.in=cublas.bf16_dequantized`, 1 sequence | `[00] PASS`: the in-projection back on its multiply over a copy, so its weights have their copies and the reference's copies made on demand are those of the gate / up weights alone |
| `--stack native --provider mlp.gate_up=cublas.bf16_dequantized+torch_ref.silu_mul_quantize`, 1 sequence | `[00] PASS`: the gate / up block back on its multiply over a copy (303 launches a step again), so its weights have their copies and only the in-projections are made on demand |
| `--stack native --provider add_norm_quantize=torch_ref.add_norm_quantize` | `[00] PASS` at 1 and at 4 sequences: the norm block on its reference chain, so the gate / up entry is handed block scales row by row and brings them into its kernel's layout itself; inside its band at both counts and equal alone and among 4 |
| `--stack native --sequences 8,9` | `[00] PASS` at 8 and at 9 sequences: the gate / up entry in its 8-row token tile at 8 and in its 16-row tile at 9, in the same launch shape (the tile shows in the form and in the symbol, `_t8_` and `_t16_`); the projection entry in its 16-row token tile at 9 (147,712 B of shared memory per thread block); both inside their bands, and each sequence alone equal to itself among 8 and among 9 |
| `--stack native --sequences 16,17` | `[00] PASS` at 16 and at 17 sequences: the gate / up entry in its 16-row and in its 32-row token tile, the projection entry in its 16-row and in its 32-row token tile (164,096 B of shared memory per thread block, the grids as at 1 row), each launch in the shape the entry states; inside their bands, and each sequence alone equal to itself among 16 and among 17. From 17 rows the embedding's gather is PyTorch's vectorized gather, held by name like the kernel of fewer rows. A traced step is 178 kernel launches and 13 copies, 12 judged one by one |
| `--stack native --sequences 32,33` | `[00] PASS` at both counts. 32 is the last row count of the projection row; at 33 the in-projection is on its multiply over a copy, so the load of the run keeps both forms of the in-projections -- the codes for the 32 sequences, the copies for the 33 -- and the copies made on demand at either count are those of the gate / up weights alone. The gate / up entry in its 32-row token tile at 32 and in its 64-row tile at 33 (230,400 B of shared memory per thread block); at 33 a traced step is 186 kernel launches, 10 judged one by one |
| `--stack native --sequences 64` | `[00] PASS` at the last row count of the gate / up rows: the 64-row token tile, grid (1, 1, 148), block (192, 1, 1) and 230,400 B of shared memory, as the entry states; inside its band over 114 calls -- 7,938,048 block scales, 7 of them another byte than the fp64 chain's, each inside its interval, and 127,008,768 codes, 122,754 of them another code than the fp64 chain's, the farthest at 91 % of its bound -- and each sequence alone equal to itself among 64. The memory sentence of this run was short (the last section) |
| `--stack native --sequences 64,65` | `[00] PASS` at both counts: at 65 the gate / up block is on its multiply over a copy (`fallback: bf16 copy of 4-bit weights`; 308 launches a step), so the load of the run keeps both forms of the gate / up weights and no copy is made on demand at either count |

The fused step has the same launch shape in both of its forms, grid (96, 1, 1), block (256, 1, 1), 42,112 B at 1 row:
that it waits and releases shows in its symbol and in its form, not in a launch record. With the norm block on its entry the greedy tokens of a checkpoint run of this stack are not those of the run before it had the row: the
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
| GPU | compute capability 10.x or 12.x **and**, for the 27B checkpoint, about 28 GiB of free memory on the sample's stack `native` on compute capability 10.0 (about 32 GiB on the other 10.x classes, by arithmetic); on the functional-validation stack `fe-cublas`, about 57 GiB (about 60 GiB on compute capability 12.x, where the first launch of the fused step takes about 3 GiB more): in practice a data-center Blackwell GPU. A 32 GB card prints the memory sentence and exits 0. The checkpoint has run on compute capability 10.0 on the stack `native`, and on 10.0 and 10.3 on the stack `fe-cublas` |
| disk | 22 GB for the checkpoint |
| packages | `pip install -r samples/models/requirements.txt`; on compute capability 10.x also the two packages of the sample's stack ("The rows a library serves") |
| time | on the stack `fe-cublas`, launched from Python: about two and a half minutes for the default prompt with 256 new tokens (the seconds are below) |

The sample was last run with:

| | |
|---|---|
| checkpoint | `nvidia/Qwen3.8-27B-NVFP4`: 3 shards, 2,194 tensors, 21.9 GB; licence field on its model card `apache-2.0`, not gated (card read 2026-11-10) |
| manifest hash | `95049016a247`, printed in the first line of a run: a SHA-256 over every shard's name, size and header (tensor names, formats, shapes, offsets), not over the values |
| packages | tokenizers 0.23.3, jinja2 3.1.6, safetensors 0.8.0; the lower bounds of `samples/models/requirements.txt` (0.21.1, 3.1.6, 0.4.3) pass the host tests of the text and weight readers |
| run | 2026-11-12, compute capability 10.0 (aarch64), in two environments: PyTorch 2.14.0 with CuTe DSL 4.8.0, and a fresh environment built by the lines of this file with PyTorch 2.15.0 and CuTe DSL 4.8.1. Three prompts at one sequence (two plain user turns, one with a tool schema and the reasoning switched off) and one prompt at two sequences; text out, `[00] PASS` in every run, the token line reported. The run before it was on compute capability 10.3 (2026-11-11) |
| provider rows | every row of the stack `fe-cublas` ran in that run: `cudnn.GdnDecodeFused` of cuDNN Frontend 1.33.0 from this checkout, cuBLAS through PyTorch 2.14.0 and 2.15.0, the PyTorch chains |
| a request body, the text on standard output | 2026-11-17, compute capability 10.0 (x86, a 148-SM part, PyTorch 2.13.0, CuTe DSL 4.8.0). On the stack `fe-cublas`, the request body of the section "A request body and its options" on standard input, with standard output and standard error on two pipes: `[00] PASS`; the answer arrived on standard output in 39 pieces, one per decode step, the first after the 29 steps of the prompt, and its line break after the last, before the first line of the report; standard output held the text and that line break, standard error every line of the run. On the stack `native`, the default prompt as a request body with its own limit of 32 new tokens, written with `--text-out` and `--no-stream`: `[00] PASS` with the ids hash and the kernel lines of the same prompt given with `--prompt`, the text in the file. On the stack `fe-cublas` again, the same request body at two sequence counts (`--sequences 1,2`): `[00] PASS`; standard output held the one text, written with the first count, and the second count reported the same text for its sequence 0 |

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
beta and alpha rows, beta first; gate and up) and dequantised to bf16 copies where the stack holds one (the stack `native` keeps a weight
its kernels multiply on its stored format), each entry's kernel is launched once, the step is recorded where the run records it, and each prompt
is fed through the decode step and decoded greedily to an end token or to its limit of new tokens. The text of sequence 0
goes to standard output as it is generated; reasoning the model writes before its answer is part of that text.

How long it takes on the stack `fe-cublas`, launched from Python, as orientation values of one run (compute capability 10.3, the default prompt, 256 new tokens, the
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
memory sentence and exits 0. On compute capability 10.0 the stack `fe-entries` keeps the in-projections on their stored
8-bit codes alone, 4.8 GiB where their copies are 9.7 GiB, and the gate / up weights on their stored 4-bit codes alone,
6.0 GiB where their copies are 21.2 GiB: it holds 30.0 GiB of weights, and its plan asks for about
37.1 GiB -- the weights, the state, working memory (an eighth of the weights held and a quarter of the stored tensors
kept) and 0.33 GiB for the one copy its checks make at a time. A run held 31.4 GiB in PyTorch at its peak, reserved
36.7 GiB, and the device grew by 36.9 GiB: inside the sentence by 0.2 GiB. On the other 10.x classes, where the
in-projections stay on their copies, its plan asks for about 41.3 GiB; that is arithmetic, not a run. The stack `native`
keeps the weights of its three library rows on their stored codes as well -- every weight a multiply reads: it holds
18.8 GiB of weights, and its plan asks for about 28.2 GiB, 2.37 GiB of it for the one copy its checks make at a time
(the output head's is the largest) and 0.38 GiB of it the workspace the library's attention is handed. A run before
that row, whose plan asked for 27.9 GiB, held 22.4 GiB in PyTorch at its peak, reserved 23.9 GiB, and the device grew
by 24.1 GiB. On the other 10.x classes its plan asks for about 32.5 GiB; that is arithmetic as well. The working
memory is an allowance taken from runs, not a bound: the room
a run took beyond its weights did not shrink with the copies it left out -- 5.3 GiB with every weight as a copy, 6.9 GiB
with the stored in-projections beside every copy and 6.6 GiB with the stored in-projections alone (two runs of
`native` before it had the gate / up row), 6.9 GiB with the in-projections and the gate / up weights on their stored
formats alone. The checks of the last run read 6,400 copies of in-projection and gate / up weights made on demand, 128
per step and reader, and printed what they cost. The checks of the run of `native` with its library rows read 12,850
copies made on demand.

The prompt goes through the decode step, one step per token (the `first token` line says so), and a step issues 484 block
calls from Python: on the run above about twelve thousand kernel launches and a thousand small copies per step, with the
GPU busy for about a quarter of the period. That is what this stack is: it reads 51.6 GB per step where the model in its
stored formats needs 17.9 GB, and it is bound by the launch path, not by the GPU. The stack `fe-entries` on compute
capability 10.0 reads the in-projections on their stored codes, 5.2 GB where the copies are 10.4 GB, and the gate / up
weights on theirs, 6.4 GB where the copies are 22.8 GB: 30.0 GB per step,
and its `bytes per step` line says so. The stack `native` reads every weight of a multiply on its stored codes there:
17.9 GB per step, the model's own bytes. The timings a run prints are orientation
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

`--dump-tokens` writes the produced ids for use outside the run. Greedy decoding is not the checkpoint's own default
(its generation settings sample, which every run with a checkpoint says in its `greedy decoding` line).

## The checkpoint's drafter

A checkpoint of this family can carry a second, small network beside the model: a drafter. In speculative decoding the
drafter proposes the next few tokens from the model's last hidden row, and the model then checks all of them in one step
instead of producing them one by one. `nvidia/Qwen3.8-27B-NVFP4` carries one.

**What is here today is its loader and its draft pass on reference blocks, both reached from the library**
(`QwenDenseDrafter`, `model_samples/qwen_dense/drafter.py`). A session that is handed the drafter runs its passes
in front of every verify step (`Session.speculate`, in the section "The verify step"), and `02_speculative_decode.py`
is the script that drafts with it (the section "Speculative decoding"). `00_decode.py` neither reads the drafter nor
drafts: it declares `--speculative-config` and `--draft-vocab` to refuse them with the name of that script. The loader
is called from Python, from the root of this checkout:

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
copy. The load also makes bf16 copies of the five weights and of the listed head rows, which the reference blocks of
the draft pass multiply on: 1.42 GiB with 65,536 ids, the last line of the manifest. `copies=False` leaves them out.

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
  and the bf16 copies its reference blocks multiply on (a load keeps them unless told copies=False): 1.42 GiB (of the five weights 0.79, of the head rows 0.62)
```

**It quantises the weights of the drafter's five multiplies to 8 bits when it loads them**
(`weights.quantize_fp8_per_channel`): the checkpoint stores them in bf16, two bytes per element, and a loaded drafter
holds one byte per element and one scale per row. A row of a weight is one output channel of its multiply, and the
rule is per row:

- the scale is the row's largest magnitude divided by 448, the largest value of the 8-bit format (E4M3), formed in
  fp32. It is never below 1 / (448 x 512), so a row of zeros has a scale too;
- a code is the element divided by that scale, rounded to the nearest value of the format, to the even code on a tie,
  and saturating at +-448. The division is a division in fp32, not a multiplication by the reciprocal of the scale:
  where a quotient sits exactly between two codes the two can give different codes.

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
you want it. The files of both scripts of this folder are read alike: `02_speculative_decode.py` writes the same
`prompt_ids` and `ids`, and the kept count of every step beside them.

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

**The draft pass.** A loaded drafter runs its pass on reference blocks -- plain PyTorch with fp64 accumulation, as
every block of the model that has no kernel yet -- in every stack (`QwenDenseDrafter.body`, `propose`, `draft`). One
pass is 18 block calls. The body is 15 of them: the embedding of a token and a hidden row of the model -- the row the
model's output head reads, behind its final norm -- are each normed and laid side by side, the embedding half first;
one multiply makes of them the residual stream of the pass; one attention layer with its MLP follows, at the model's
widths and on a cache of the drafter's own; a final norm gives the output rows. The head is 3 calls on one row per
sequence: the 4-bit quantisation of the head's operand, the multiply over the listed rows of the model's output head,
and the token of the largest logit -- the first of equal columns, then the token id of that column.

Every 8-bit operand of a pass is quantised per row of the step, at a scale formed from the row
(`weights.quantize_fp8_per_row`): the scale is the row's largest magnitude, taken over its bf16 values, divided by 448
and never below 1 / (448 x 512); a code is the element *multiplied by the reciprocal* of that scale; the division, the
reciprocal and the product are fp32 operations. This is a third rule beside the static scale of the model's
activations and the division of the weights above. Where a quotient sits exactly between two codes the two rules can
give two codes: the product then lands just above the midpoint and takes the code above, where the division takes
the even one. A host test holds a row with two such elements. A multiply then applies, to each element of its
result, the scale of the operand's row and the scale of the weight's row, the two multiplied in the accumulation
type. The first norm block quantises its two halves as one row, so both share one scale.

`draft` is the passes of one step of speculative decoding. The first pass runs over the rows it is handed: per
sequence the final-norm rows of the model's last step and, for each, the token that follows it -- one row per token of
a verify step, or one row behind a prompt. Of its output the row of the last kept token goes to the head: a view, by a
host integer. Every later pass runs at one row per sequence, on the pass's own output row and its own last proposal,
one position further. A pass writes the keys and values of its rows into the drafter's cache at their positions and
is told where they stand; it writes nothing of the model.

A draft step from Python, behind the call of the loader above (its `model`, `drafter` and `weights`):

```python
import torch
from model_samples.blocks import DeviceFacts, plan

# the blocks of the drafter's table.  Stated facts of a GPU class serve on the CPU; on a GPU, hand over the device
table, source = drafter.provider_table, drafter.table_source
blocks = plan(table, drafter.regimes, DeviceFacts((10, 0)), 1, 1, "reference", model=drafter, source=source)


def call(block, index, *operands):  # one block of pass `index`
    return blocks[block](*operands)


cache = drafter.state(1, 64, "cpu")  # the drafter's own cache: one sequence, up to 64 positions
hidden = torch.zeros((1, model.geometry.hidden), dtype=torch.bfloat16)  # [rows, hidden]: final-norm rows of the model
ids = torch.tensor([198], dtype=torch.int32)  # [rows]: the token that follows each of those rows
drafts = drafter.draft(call, weights, cache, hidden, ids, 0, 0, 5)  # position 0, row 0 kept, five passes
print(drafts.tolist())  # [sequences, passes]: the proposed token ids
```

These lines share the load of the call above -- the drafter's weights with their copies, and the model's embedding
table -- and the cache they make; they read nothing of the checkpoint. `plan` resolves the drafter's table as it
resolves a model's, for a GPU class, a number of sequences, the tokens per sequence and a stack; every row of this
table serves all of them, so one plan runs a pass over any number of rows. `call(block, pass, *operands)` runs one
block, as a family's step calls its blocks, with the pass, counted from 0, where a step has the layer. `hidden` and
`ids` are the rows of the step, sequence by sequence: per row the model's final-norm row -- the row its output head
reads -- and the token that follows it. `position` is where the first row of every sequence stands, `kept` the row
of a sequence the step goes on from (0 behind one row), and `passes` the number of tokens to propose. Here the
hidden rows are the caller's to supply; a session that is handed the drafter supplies its own -- the rows its
model's last step left -- and runs these passes in front of every verify step (`Session.speculate`, in the section
"The verify step").

| block | calls per pass | provider key | entry [form] |
|---|---|---|---|
| token embedding lookup | 1 | `torch.index_select` | torch.index_select |
| RMSNorm of the embedding rows and of the hidden rows + one quantisation per row over both | 1 | `torch_ref.pre_fc_norm_quantize` | torch_ref.pre_fc_norm_quantize, fallback: reference chain |
| 8-bit multiply, a scale per weight row and per row of the step | 5 | `torch_ref.linear` | torch_ref.linear, fallback: reference chain |
| (residual add +) RMSNorm + quantise per row | 2 | `torch_ref.add_norm_quantize` | torch_ref.add_norm_quantize, fallback: reference chain |
| attention preparation | 1 | `torch_ref.attention_prepare` | torch_ref.attention_prepare, fallback: reference chain |
| attention core | 1 | `torch_ref.attention` | torch_ref.attention, fallback: reference chain over a plain cache of the drafter's own |
| gate + quantise per row | 1 | `torch_ref.gate_row_quantize` | torch_ref.gate_row_quantize, fallback: reference chain |
| SiLU x up | 1 | `torch_ref.silu_mul` | torch_ref.silu_mul, fallback: reference chain |
| quantise per row | 1 | `torch_ref.quantize` | torch_ref.quantize, fallback: reference chain |
| residual add + RMSNorm | 1 | `torch_ref.add_norm_quantize` | torch_ref.add_norm_quantize, fallback: reference chain |
| 4-bit quantisation of the head's operand | 1 | `torch_ref.quantize` | torch_ref.quantize, fallback: reference chain |
| output head over the listed rows | 1 | `torch_ref.linear` | torch_ref.linear, fallback: reference chain |
| token choice over the listed rows | 1 | `torch.argmax+lookup` | torch.argmax + the token id of the chosen row, fallback: reference chain |

Every row of this table serves every stack, every GPU class and any row count. The multiply is called at five sites
and the norm + quantise at two, and the family hands each call the name of its site (`QwenDenseDrafter.sites()`). A
reference chain is one function at every site, so the rows name no form; what differs between the sites is data for
the kernels that are to serve these blocks (`configs.DRAFT_PROJECTION_SITES`, `configs.DRAFT_NORM_SITES`):

| block | call site | what differs there |
|---|---|---|
| 8-bit multiply, a scale per weight row and per row of the step | `fc` | weight 5120 x 10240 |
| 8-bit multiply, a scale per weight row and per row of the step | `qkv` | weight 14336 x 5120 |
| 8-bit multiply, a scale per weight row and per row of the step | `o` | weight 5120 x 6144 |
| 8-bit multiply, a scale per weight row and per row of the step | `gate_up` | weight 34816 x 5120 |
| 8-bit multiply, a scale per weight row and per row of the step | `down` | weight 5120 x 17408 |
| (residual add +) RMSNorm + quantise per row | `input` | no add: its rows are the residual stream |
| (residual add +) RMSNorm + quantise per row | `mlp` | the branch is added to the residual stream first |

`drafter.state(sequences, capacity, device)` is the cache: keys and values per position, zero at the start of a
prompt set, with no position of its own. The attention of a row reads every position of its sequence from 0 to its
own, so a position in front that no pass wrote is read as those zeros and nothing refuses it: filling the cache over
a prompt, one body pass per position, is the caller's. A session with the drafter does it in `prefill`. A step
whose passes would write past the cache is refused before its first pass. With the copies and that cache,
`drafter.memory_bytes(sequences, capacity)` is what a loaded drafter holds on the device besides the model's
embedding table: 2.00 GiB at the published widths
with 65,536 ids, one sequence and 2,051 positions. A pass on a load without its copies is an error that names
`copies=True`.

**What holds the pass.** Every block of it is its own reference chain, so the tie-out of a run judges none of them.
The drafter's blocks are held to their definitions and to each other at one row and at several, and their proposals
are reported against the model's own tokens on a checkpoint; they are not compared with another implementation of
the drafter in this repository (one second evaluation of its network was made by hand, outside this repository:
it is among the runs below).

- *To their definitions* (host tests, `test/python/misc/test_model_samples_drafter.py` and
  `test_model_samples_weights.py`): the row quantiser and the joint quantiser of the two normed halves against a
  quantiser written in the test from the 8-bit format, scales and codes; the gate with its one rounding and SiLU x up
  with its two against their formulas, element by element; the multiply with a scale per row of both operands
  against a loop over the code values; the token choice on stated logits with equal columns. Those are the
  functions. The blocks that call them are held one layer up: every block of the drafter's plan, as the plan builds
  it, against the function it stands for on stated operands, and one body pass through the plan's blocks against
  the same pass written out from the functions, rows and cache, then the head block by block.
- *To each other*: a pass over 2 and over 6 rows of a sequence gives the rows and the cache of that many passes of
  one row, byte for byte, at one sequence and at two; a recording stand-in for the blocks shows what every pass of
  a step is handed, for every kept count 0 to 5: the row of the last kept token, then the pass's own output row, its
  own last proposal and the next position; and every block of a pass over two sequences gives, for each sequence,
  the bytes that sequence gives alone, its operands cut to the rows the block says it has per row of the step.
- *Against the model's tokens* (reported, no gate): the checkpoint run below, with the same pass run on wrong inputs
  beside it.

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

**The runs of the draft pass** (2026-11-18; PyTorch 2.13.0; a GPU of compute capability 10.0, a 148-SM part shared
with other users: correctness only, by hand, with drivers that are not part of this repository).

*The seeded drafter of the reduced model, at the published widths.* A pass over six rows of a sequence gave the rows
and the cache of six passes of one row, byte for byte, at one sequence and at two. `draft` with every kept count 0 to
5 behind six rows, and behind one row, gave the five proposals and the cache of the same passes made one row at a
time. The row quantiser gave, for a stated tensor, the codes and the scales the CPU gives. Every kernel of those
calls is PyTorch's or cuBLAS's, by name: the embedding lookup is PyTorch's gather, and the id of the chosen row and
the concatenations launch other PyTorch kernels than the gather. A second run printed the same bytes.

*The published checkpoint, against the model's own tokens* (reported; no line of it is a gate of this sample). The
model decoded three prompts greedily on the cuDNN Frontend rows of the stack `native` (the rows `fe-entries` selects
today without the attention preparation's row), 70 new tokens each: the default prompt and two arithmetic word problems
whose text is not part of this
repository, 58, 117 and 78 prompt tokens. The drafter then ran over each
of them teacher-forced: one body pass per prompt position fills its cache, and from the last prompt position on every
position is one draft step of five passes behind one row, handed the model's final-norm row of that position and the
token the model chose there. The list is the one the recipe above made in the runs of 2026-11-17: the 342 ids of six
dump files, then the lowest ids; the default prompt is one of the three prompts behind those dumps, the other two are
not.

- The first proposal equals the model's next token at 167 of 207 positions (46, 56 and 65 of 69 by prompt), the
  second at 128 of 204, the third at 94 of 201, the fourth at 70 of 198, the fifth at 54 of 195. At 4 of the 207
  positions the model's token is not in the list. Had a step started at every position, the proposals would have
  kept 2.37 drafted tokens a step on average over 195 positions: none at 37 of them, all five at 42.
- The same pass with one input wrong, first proposal, same positions: the two halves in front of the first multiply
  exchanged, 0 of 207; the hidden row replaced by zeros, 8 of 207; the hidden row of the position before, 108 of
  207; the hidden row of the position after, 150 of 207. The hidden row carries the proposal and the pass reads the
  right one. A row that is one position old is still a row of the same text: half of the proposals survive it.
- The first pass over six rows -- the six rows that end at a position, the last of them kept -- gave the five
  proposals of the pass behind one row at 36 of 36 positions.
- A plain fp32 evaluation of the drafter's network on the checkpoint's bf16 tensors -- no 8-bit codes, no quantiser
  per row, written for this run from the checkpoint's layout -- gave the pass's first proposal at 200 of 207
  positions and the model's token at 166. It is not part of this repository.
- With a cache of 194 positions the loaded drafter held 2,135,219,200 bytes, which is `memory_bytes(1, 194)`.

**What is not here.** The drafter has a draft pass on reference blocks, reached from the library
(`QwenDenseDrafter.body`, `propose`, `draft`): every block of it is a reference chain in every stack, and its embedding
lookup is PyTorch's gather. A session that is handed the drafter runs its passes (`Session.speculate`, in the next
section); `02_speculative_decode.py` takes the list as `--draft-vocab FILE`, the loader as `draft_vocab=`. No kernel serves a block of the drafter and none has read these 8-bit weights. The loader and the pass have run on the CPU and on one GPU class, compute capability 10.0; a class they have not
run on is uncovered, not passing. And the 8-bit weights are held to the rule above, not to the weights another
implementation makes of the same checkpoint: no such comparison is part of this repository.

## The verify step

Speculative decoding drafts a few tokens ahead and lets the model check them in one step: every sequence is fed the
token it kept last and the tokens drafted behind it -- `T` tokens per sequence at once -- and the step returns, for
each of them, the greedy token that follows it. A run keeps the drafted tokens up to the first one the model would
not have chosen, and goes on from there. This section is that step, the model's side of speculative decoding, and
the step a session makes of it and the drafter's passes (**The speculative step**, below). `02_speculative_decode.py`
runs the second (the section "Speculative decoding"); both are reached from the helper layer as well, and for a
verify step alone the caller names the tokens it is fed and how many of them are kept.

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
nothing, reads nothing to the host and copies nothing. A session built with `tokens=` has no `prefill` and
no `step`: the decode step is a row of another plan, and a run that needs both has two sessions on one load, planned
together, as above.

The two sessions share the load and nothing else. Each has a state of its own -- recurrent state, history, caches,
position -- so a verify session starts at position 0 on a zero state, whatever the decode session has run: a prompt
prefilled through `decode` is no context of `verify`, and no error says so. What a verify session is to continue
reaches it through its own verify steps, `T` tokens at a time, keeping all of them.

**One session for a prompt and its verify steps.** A session built with `draft_tokens=k` does both: it feeds its prompt
through decode steps and then runs verify steps of `k + 1` tokens on the same state. Its plan table shows both plans,
and one memory sentence counts one state. The caller supplies the drafted tokens and says after the host read how many
were kept; handed the drafter, the session drafts them itself (**The speculative step**, below). Two sessions on one
load still share weights, never a state. How its prompt works on
that state follows the layout (**The prompt on that state**, below); how such a run is checked, and what ran, follows
the checks of a verify session (**The checks of a session that verifies behind its prompt**).

**The state.** The recurrent state after EVERY token of a step is written to a slot of its own, so the next step can
start from whichever token the run kept, with nothing copied. The slots are numbered token by token -- sequence `i`,
token `t` owns slot `1 + t x sequences + i` -- so token 0 of a sequence is its plain slot `i + 1`, and the layout of
one token per sequence is this layout at `T = 1`. The conv history of a sequence holds `T + 2` columns in its plain
slot: a step reads its three taps where the last step left them and writes the last two of them followed by its own
`T` rows. Keeping `n` tokens moves one number on the host; the kernel reads it from a constant table. An attention
layer inserts `T` positions per sequence, and token `t` attends up to its own.

**The prompt on that state.** The decode steps of the prompt of a session built with `draft_tokens=k` work on token
0 of its state of `k + 1` tokens per sequence: its slots are the plain slots, its history the columns 0 to 2, its
caches the same tensors, so the first verify step reads what the prompt left and nothing is copied.
`Session.prefill` feeds the prompt through the decode plan; `Session.verify(ids)` runs a step of the verify plan and
leaves its count open; the caller reads the greedy ids, counts the drafted tokens that equal the greedy token in
front of them (`sampling.accepted`: for several sequences the smallest count of the set, the one count that is
exactly greedy for every sequence) and says it with `Session.accept(count)`. A decode step behind a keep is not
built: `Session.step` is refused on such a session, and so is a second prompt behind its first verify step. The
capacity holds a whole window wherever a step starts: a step at position `p` needs `p + k + 1` positions, so a run
of a prompt and `n` new tokens is built with a capacity of `prompt + n + k - 1` -- its last step can start at the
last new token but one -- and a step that does not fit is a `ValueError` that names the capacity.

**Who serves it.** The block `linear_attention.step` is served by `fe.gdn_verify_fused`, `cudnn.GdnVerifyFused`: one
launch per layer for all tokens of all sequences. The step shape is a compile-time fact of that kernel, so the table
has one row per tokens count, 2 to 8, with `TOKENS` as a field of the form, for 1 to 16 sequences below six tokens per sequence and 1 to 4 from six -- the limits of the rows as they stand, kept until a wider step shape is run; eight tokens is what every form
of the entry holds on the classes it serves. A step shape beyond these is declined before the load, by name:
`cudnn.GdnVerifyFused serves 6 tokens per sequence of 2-8, sequences 1-4; this run has 5 sequences x 6 tokens per
sequence`. In the stacks `native` and `fe-entries` the row's form is `[PDL True, PDL_RELEASE True, TOKENS T]`: a dependent launch that waits for the in-projection only before it reads that launch's rows, and releases its
successor at entry. In `fe-cublas` it is `[PDL False, PDL_RELEASE False, TOKENS T]`. Neither the launch knobs nor the
tokens are in the kernel's name, which is cut in front of them: the form on the objects shows them, and the shared
memory of a thread block follows the tokens (126,512 B at six tokens and one sequence on compute capability 10.0).

Every other block is the row its own ranges give at `sequences x T` rows, as at any other row count. At one sequence
of six tokens the norm entry, the in-projection entry (compute capability 10.0), the fused gate / up entry (the
8-token tile) and the attention gate entry keep their rows; at 16 sequences of five tokens -- 80 rows -- the
in-projection (rows 1-32) and the gate / up step (rows 1-64) are on their fallback rows, printed with their reasons.
The attention preparation keeps its row as well, one launch for all `sequences x T` rows of the step, each row at its
own position and slot; the attention core is the library's kernel in `native`, one launch for all rows of the step,
and a reference chain in `fe-entries`, called once per token.

**The out-projection.** In a decode step a library multiply serves the out-projection of every layer. In a verify step the 8-bit projection kernel serves the out-projection of a linear-attention layer -- the kernel of
the in-projections, at the weight shape 5120 x 6144 -- and keeps the library multiply for an attention layer's. So the
block has rows of `fe.fp8_projection_decode`, `cudnn.Fp8ProjectionDecode`, for a step of several tokens per sequence
alone, in the stacks `native` and `fe-entries`, and such a row serves one of the block's two call sites: the
linear-attention layers' call (`linear_attention.out`). The attention layers' call (`full_attention.out`) it hands on
to the next row: the library's multiply in `native`, the multiply over a copy in `fe-entries`. A plan of a verify step
then prints the block once per call site (the 64-layer model, `native`):

```text
projection.fp8.out at linear_attention.out  48       cuDNN FE    cudnn.Fp8ProjectionDecode                    rows 2-8 of 2-16 of a step of several tokens per sequence, the linear-attention layers' weight 5120 x 6144  none
projection.fp8.out at full_attention.out    16       FlashInfer  flashinfer.gemm.bmm_fp8                      any row count, weight 5120 x 6144                                                                           none; unverified in CI
```

The split of this kernel follows the rows of the step (sequences x tokens per sequence), and a split is fixed per entry instance, so every range is a row of its own (`VERIFY_OUT_ROWS` in `configs.py`):

| GPU class | rows of a step | K slices, weight tile, ring stages, token tile | launch: grid, block, shared memory of a thread block |
|---|---|---|---|
| compute capability 10.0 | 2 to 8 | 2, 64, 11, 8 | (80, 2, 1), (256, 1, 1), 103,808 B |
| compute capability 10.0 | 9 to 16 | 4, 64, 7, 16 | (80, 4, 1), (256, 1, 1), 84,352 B |
| compute capability 10.0 | 17 and more | -- | the multiply over a copy at both call sites: no row of the entry holds that many rows on this class |
| compute capability 10.3 and 10.7 | 2 to 8, 9 to 16, 17 to 32 | 4, 64, 7, and 8, 16, 32 | (80, 4, 1), (256, 1, 1), and 71,040 B, 84,352 B, 110,976 B |

A row states the kernel whole, so the entry never runs on its own pick: `[SPLIT_K 2, MMA_M 64, STAGES 11, TOKEN_TILE
8, TMEM_COLS 32, PREWAIT_WEIGHTS True, HOLD_TMEM True, PDL True, PDL_RELEASE True]` for 2 to 8 rows, with the weight's
shape at its call site (`OUT_FEATURES 5120, IN_FEATURES 6144`). The token tile follows the row count in the entry -- the smallest of 8, 16 and 32 that holds the rows -- and the entry takes it as a floor of its
form. A row names the tile of its range, so that the call log of a step shows the row count a call was made with: an
instance that is handed twelve rows states the 16-row tile, and the kernel check names it against the row. The
tensor-memory columns and their release (`TMEM_COLS 32`, `HOLD_TMEM True`) are performance fields of the entry, which the row states like every other field. The kernel is launched as a dependent
of the fused verify step, which releases the launch after it at entry: it starts under that kernel, streams its weight
and waits for it right before it reads the 8-bit codes; it releases its own successor after its last weight issue, and
the norm in front of the MLP waits before it reads the rows.

The two ranges of compute capability 10.0 are two byte classes of the entry: the out-projection of a step of 8 rows
and of a step of 9 are not the same bytes, by the split of each range. Inside a range the entry writes a
row's bytes alone as among the others, and the batch-invariance check gates it there on compute capability 10.0 (one
sequence alone runs on the entry instance of its step's range). The rows of compute capability 10.3 and 10.7 claim
what the entry admits; the entry's page lists this weight shape as unverified on those classes, and no run of this
sample has been made there. By provider, a verify plan of `fe-entries` with such a row counts what the decode plan
counts on compute
capability 10.0 -- cuDNN Frontend 7; fallbacks 4 (cuBLAS over copies 3, PyTorch chains 1) -- because the two 8-bit
projections count as one kind of block per provider and both providers have one there already. On the other 10.x
classes the in-projection is a multiply over a copy, so the out-projection's row makes the verify plan cuDNN Frontend 7
where the decode plan is cuDNN Frontend 6, with the same fallbacks. A plan of `native` counts FlashInfer 4
beside them at either step shape, and one fallback row fewer per library row.

The kernel multiplies on the stored 8-bit weight, and the load keeps the forms of this block per call site: the
codes of the linear-attention layers' out-projections for a plan with such a row, the bf16 copies of the attention
layers'. A run whose every plan is such a verify plan holds no copy of the linear-attention layers' out-projections
(2.8 GiB on the published model; their codes are 1.4 GiB), and the reference of its checks reads copies made from the
stored bytes, one call at a time. A run in which a decode session shares the load -- the snippet above -- keeps what
both plans read: the codes for the verify plan and the copies for the decode plan. The codes kept beside the copies
add about 1.9 GiB to the memory sentence of the published model: their 1.4 GiB and their share of the working memory.

**What the kernel table of a verify plan counts** (stack `fe-entries`, the 64-layer model, compute capability 10.0 with
148 SMs, one sequence of six tokens):

```text
per step: 387 launches on 9 exact rows, checked one by one in the wiring order; the launches of 97 fallback calls and of the host path are attributed by family
```

A verify step has other kernels than a decode step in two places: the fused step is the verify kernel, and the
out-projection of a linear-attention layer is the projection entry's where the decode step has a library multiply. The
97 block calls on fallback rows are the 64 down multiplies, the 16 calls of the attention core, the 16 out-projections
of the attention layers and the output head. On `native` those are the library's rows, and a run prints:

```text
per step: 484 launches on 13 exact rows, checked one by one in the wiring order; the launches of 0 fallback calls and of the host path are attributed by family
```

The launches of the drafter's passes are not part of the table: a traced step of a session with a drafter holds them
in front of the model's, and the kernel check attributes them like every launch.

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

On compute capability 10.0 (a 148-SM part), on the cuDNN Frontend rows of the stack `native` -- the rows `fe-entries`
selects today without the attention preparation's row; these runs were made before the stack had its library rows --,
the two-layer seeded model: at every step
shape that ran up to 32 rows -- 2, 6, 8, 9, 12, 16, 18, 24, 28 and 32 rows -- every greedy token of the verify steps was
the decode
steps', and the recurrent state of the kept tokens was within the band (at one sequence the same bytes); the kernel
check passed with the verify kernel in the shape its entry states and, at one sequence of six tokens, with 13 launches of a step on exact rows. Up to 16 rows the verify session runs the out-projection of the linear-attention layer on its kernel and the decode
session on a multiply over a copy, which rounds its unscaled sum and the scaled result where the kernel rounds once:
there the logits of the two runs were one bf16 unit apart at most, inside the family's band of 4, and the keys and
values an attention layer cached were not the same bytes -- printed, not gated, because the layer was fed other rows.
From 17 rows both sessions multiply over a copy: at three sequences of six tokens and at four of six, seven and eight
the logits and the caches were the same bytes again, and at nine and 16 sequences of two tokens the logits were within
half a unit (a multiply over a copy is another library kernel at another row count). At 16 sequences of three, four
and five tokens -- 48, 64 and 80 rows -- every greedy token was the decode steps' as well, and the kept state was
printed, not gated: from 33 rows of a step the in-projection is a multiply over a copy in the verify session and its
entry in the decode session of 16 sequences, so the linear-attention layer was fed other rows. The entry was inside
the band of the reference multiply at the first and the last row count of both ranges --
one sequence of two tokens and of eight, three of three, two of eight -- with one bf16 unit on fewer than 0.05 % of
the elements, and each sequence alone wrote the bytes it wrote among the others. At two and three sequences, on a
part of this SM count, the verify entry takes another form than at one sequence and at four: one thread block per
value head and sequence, each on an SM of its own, with 224,672 B of shared memory at six tokens and 231,744 B at
eight, the largest step that form holds. These are the runs of one GPU class. With its library rows the stack ran
seven of these step shapes on 2026-11-18 (PyTorch 2.14.0, FlashInfer 0.7.2.post1; the attention preparation on its
reference chain) -- one sequence of two, six and eight tokens, two and four sequences of six, four of eight and 16 of
two --: every greedy token of the verify steps
was the decode steps', the recurrent state of the kept tokens was within the band, and the kernel check passed, at
one sequence of six tokens with 17 of the 20 block calls of a step on kernels.

**The checks of a session that verifies behind its prompt.** `tie_out.speculative_checks(session, decode)` holds a
run of a session built with `draft_tokens=` against decode steps of the same tokens, line by line. It replays what
the session recorded -- its prompt, the windows it verified, the counts it kept -- from a zero state, and feeds the
decode session the kept tokens beside it. `tokens`: every kept row against the decode step of its position, by the
family's logits band, and the greedy rule of the loop -- every window starts with the greedy token kept last, every
kept drafted token is the greedy token in front of it. `kept state`: as for a verify session, above. `position`:
after the prompt and after every step the session stands where the decode steps stand and on the token the step
kept, and the session's own run ended there -- integers. `prompt`: after the prompt, what every layer stands on and
the logits of every prompt step are the decode session's bytes. `recorded`: the ids and logits of every step the
session recorded of its own run are the replay's bytes, so what the other lines judge is the run the session's own
methods ran. `bytes`: the logits of every kept row and every layer after every step are the decode session's bytes.
`rejected rows`: a second replay with other tokens behind the kept ones leaves the kept rows and the standing state
byte-identical. `acceptance`, for a scheduled run -- one whose drafted tokens are built so that a correct step keeps
a stated number of them (`tie_out.forced_drafts`): every step kept the count it was scheduled to keep, and every
count was kept. `position`, `prompt`, `recorded` and `rejected rows` cannot fail by rounding and are gated on every
stack. `bytes` is gated on the stack `reference`, where a verify step is that many decode steps of one chain, and
`REPORTED` on a provider stack, where the two kinds of step launch other kernels for some blocks. On a checkpoint,
which declares no logits band, `tokens` is `REPORTED`: how many kept tokens equal the decode steps' and the first
that differs, with its sequence where there are several. The comparison is made once per run, on the session as its
run left it, and leaves it reset with its records. The session's own checks then replay the whole recorded run from
a zero state -- the prompt's decode steps under the block check of the decode plan, the verify steps under the block
check of the verify plan -- and its traced pass replays the prompt in front of the profiler and holds the verify
steps alone.

What ran (compute capability 10.0, 148 SMs; a shared GPU: correctness only). These are runs of 2026-11-17: `native`
selected the cuDNN Frontend rows then, the rows of `fe-entries` today without the attention preparation's row, and what
this passage states for `native` --
memory, launches, coverage -- is of those rows unless a sentence names the library rows. The seeded two-layer model, one
sequence,
five drafted tokens, a prompt of 8 tokens, 32 new tokens through 10 verify steps whose drafted tokens were scheduled to
be kept 0 to 5 in turn: on `native`, `fe-entries` and `fe-cublas` the 32 tokens were the plain session's; `tokens`
passed with 31 of 31 kept tokens equal to the decode steps' and the logits within 1.00 bf16 unit, inside the band of 4;
`kept state` was gated in the linear-attention layer; `position`, `prompt`, `recorded`, `rejected rows` and
`acceptance` passed; `bytes` was reported (on `native` 0 of 31 kept rows were the decode steps' bytes); the launches
and copies of every prompt step were those of a plain session's step in the same process; replay, the blocks of both
kinds of step and the kernel check passed, on `native` and `fe-entries` with 13 of the 20 block calls of a verify step on kernels: these were the runs
of 2026-11-17, in which both stacks selected the cuDNN Frontend rows. With its library rows, the attention preparation
on its reference chain as before, `native` passed the same lines on 2026-11-18 with 17 of the 20 block calls of a verify step on kernels. On `reference` every
line passed, `bytes` included, and the run ended in `RAN`. One sequence of two and of eight tokens, four layers, and
two sequences of six tokens (each with its own tokens, one count: the smallest) passed on `native` as well, and so did
the ends of the sequence range of such a session: four sequences of six tokens and 16 of two.

In one process on `fe-entries`, fourteen wrong loops each failed the line that is there for them. Three of them -- the
position moved by the kept count alone, another attention cache for the prompt, a row that attends over the whole
window -- passed `tokens` and `kept state` while they failed `position`, `prompt` and `rejected rows`. Four are wrong in
a method of the session itself, where the family's step is right -- `accept` does not tell the state which token was
kept, `accept` moves the position by the kept count alone, a step that is given its count does the same, the prompt is
fed to a state of its own -- so the comparison's replay held on `tokens`, `kept state`, `prompt` and `rejected rows`,
and each failed `position` or `recorded`. The forgotten count also ran on `fe-cublas`, where the kept state of no
layer was gated for it: `position` failed it there.

The run of `native` asked for about 3.6 GiB, reserved 3.6 GiB and grew the device by 3.8 GiB. At 16 sequences of two
tokens the run asked for about 3.6 GiB, reserved 4.3 GiB and grew the device by 4.4 GiB: the comparison keeps what the
session stands on after each step it replays -- 21 steps of 16 sequences in that run -- until its second replay has
ended, and the memory sentence does not count that.

On the published checkpoint (`native` on 2026-11-17, the rows `fe-entries` selects today without the attention
preparation's row; one sequence of six tokens,
the default prompt of 58 tokens through
`prefill`, then 10 verify steps fed a plain run's 32 new tokens with scheduled counts): `position`, `prompt`,
`recorded` and `rejected rows` passed; `kept state` was gated in 1 of 48 linear-attention layers; `tokens` was reported --
30 of 31 kept tokens were the decode steps' greedy tokens on the same tokens, the first that differs
kept token 13; replay, the blocks of both kinds of step and the kernel check passed, with 370 of the 484 block calls of a verify step on kernels. The sentence asked for about 40.0 GiB, PyTorch reserved 38.9 GiB and the device grew by
39.0 GiB, the GPU otherwise empty.

**The speculative step.** `Session.speculate` runs one step of speculative decoding: the drafter's passes on
reference blocks, one verify step, one host read. It is the step of `02_speculative_decode.py`, and is reached from
the library too. The drafter's
cache is filled by one pass of the drafter per prompt position, at one row. 

```python
from model_samples import Session, sampling

plain = Session(model, text, sequences=1, stack=stack, device=device, capacity=positions, checkpoint=checkpoint)
session = Session(model, text, sequences=1, stack=stack, device=device, capacity=positions, share=plain, draft_tokens=5, drafter=drafter)
print(plain.plan_table([session]), file=sys.stderr)  # every plan resolved; the plain table, one memory sentence for the run
plain.load()  # the model's weights are read here, once
session.load()  # the drafter is loaded here, with the model's embedding table
new = session.read(session.prefill([prompt]))  # the prompt: decode steps, and the drafter's passes over it; the first new token
while len(new) < limit:
    step = session.speculate()  # five passes of the drafter and one verify step of six tokens; nothing is read
    ((window, greedy),) = session.read(step)  # the ONE host read of the step
    kept = sampling.accepted(window, greedy)  # 0 to 5: the drafted tokens the model would have chosen itself
    session.accept(kept)  # the state stands at the last kept token; nothing is launched or copied
    new += greedy[: kept + 1]  # the tokens of this step: text can go out here, before the next launch
```

These lines share one load between the two sessions: the plain session reads the model's weights, once, and is what
the checks hold the loop against (`tie_out.speculative_checks`); the session with the drafter runs on them
(`share=plain`) and keeps the drafter's weights and its cache. `checkpoint`, `model` and `drafter` are those of the
loader's call in the section "The checkpoint's drafter". The session that loads is handed the checkpoint
(`checkpoint=`): a session that is handed none draws seeded weights, and the drafter of a checkpoint is refused
there, by name, where the session is built -- `checkpoint` is `None` for the seeded pair
(`QwenDenseDrafter.seeded(model)`). A drafter of another model's widths is refused there as well, by the first
width that differs. `text` is the tokenizer (`TextIO`), `stack` and
`device` a session's, `sys` the standard module, `prompt` the ids of one prompt, `limit` the new tokens asked for,
`positions` at least the prompt, the new tokens and four more: the last window reaches that far. The first
`plan_table` resolves every plan of the run and prints the plain session's table and one memory sentence for the
whole run, the drafter in it; `session.plan_table()` prints the session's own three tables. A session with a drafter
stands alone as well, without `share=`: it then loads the model itself. The loop reads the device once per step.
`drafter=` needs `draft_tokens=`, and a session with a drafter serves one sequence: one request, as the drafter rows
are for. On `native` the four library rows serve the decode steps of the prompt and the verify step of such a
session as they serve a plain run, and the drafter's passes call none of them: on compute capability 10.x the session
needs FlashInfer at a validated version, and without it `plan_table` declines before anything is loaded, by the
sentence of the section "The rows a library serves"; `fe-entries` runs the same session on the cuDNN Frontend rows.

A step, in order. `speculate` hands the drafter what the step before left on the state: every final-norm row of
that step, by reference (`state.hidden`), and the greedy token after each (`state.greedy`); the position of its first
row; and how many drafted tokens the run kept of it, one host integer. The first pass runs over all those rows and
writes the drafter's cache at their positions; the row of the last kept token goes to the head; every later pass runs
at one row on the drafter's own row and proposal. The window is the greedy token the run kept last and the proposals,
one concatenation; then one verify step; then the pair `[sequences, 2, tokens]` -- the window, and the greedy token
after each of its tokens -- which the caller reads. `speculate` compares nothing: the count is the caller's, from
that read (`sampling.accepted`, `accept`). The first step of a run has no verify step in front of it, so its first
pass runs at one row per sequence, over the last prompt position; from the second step on it runs at `S x T` rows.
A step that keeps fewer tokens than were drafted leaves the drafter's own rows in its cache behind the last kept
position; the first pass of the next step writes those positions again, from the model's rows, before a row reads
them. With `force=` (`[sequences, draft tokens]` int32 on the GPU) the step verifies drafted tokens of the caller's
in the place of the drafter's: the drafter runs all the same, and its own proposals are in the record
(`session.record.proposals`). On a session with a drafter `verify` is refused: a verify step without the passes in
front of it would leave the drafter's cache without its positions.

The plan table of such a session (`session.plan_table()`) has a third table under one line -- "the drafter: 18
block calls a pass; the first pass of a step behind a verify step runs at S x T rows, every other pass and every
pass over the prompt at S rows; every row of this table serves any row count" -- and the one memory sentence counts
the drafter: what its load and its cache hold (`memory_bytes`) and the working memory of a pass on reference blocks,
an allowance of twice the largest chunk of a weight a reference multiply converts to fp64 at a time
(`working_bytes`: 1.33 GiB at the published widths). At the
published widths with 65,536 ids and 2,051 positions the drafter adds 3.3 GiB: a run of a plain session and such a
session asks for about 32.8 GiB on `native` (compute capability 10.0, where its library rows keep the weights of the
down multiplies, the out-projections and the output head on their stored codes), about 43.6 GiB on `fe-entries` and
about 61.2 GiB on `fe-cublas`, where the same run without a drafter asks for about 29.5, 40.2 and 57.9 GiB. These are
arithmetic on stated facts; a measured run is below.

**The checks of a speculative step.** Every block of the drafter but its embedding lookup is a reference chain, so
the block check judges none of them. `tie_out.speculative_checks` holds a session with a drafter by two more lines.
Beside its replay it
writes the drafter's passes out a second time, at one row per call, from books of its own: the kept tokens of the
run in order and, per kept position, the model's final-norm row of it. A position gets its pass when the token that
follows it is known; the passes of a step are then the head on the last of those rows and, for every later draft,
the body on the row and the proposal of the pass before, one position further. `drafter cache`: after the passes of
every step, the session's drafter cache up to the last kept position is that second run's, byte for byte -- it sees
the rows, tokens and positions of the first pass of a step and of the passes over the prompt. `draft proposals`: the
proposals the session's own run recorded for every step are that second run's -- it sees the row a step goes on
from and what every later pass is handed. Both runs call the same blocks: the two lines hold what the passes of a
step are handed, not the arithmetic of a block, which the tests of the section above hold. The session's own
checks replay the run with the passes and hold the proposals beside the ids and logits (`replay`). Its traced pass
holds whole steps as `speculate` runs them, fed the recorded drafted tokens: the sites of such a step are, for every
pass, the drafter's wiring with the pass's number in the place of the layer, then the model's, and the kernel check
holds the embedding lookups in that order -- every pass's, then the model's -- attributes everything else a pass
launches to the drafter's fallback rows, and holds the model's part to the verify plan as before. The report of such a session has two acceptance lines: `kept tokens per step X.XX of 6 (1 + accepted drafted tokens /
steps), n steps; kept 1: .., .. 6: ..` and `accepted per draft: 1: x of n, 2: y of x, ..` -- per drafted token the
steps that kept it among those that kept the one before. A run whose steps verified drafted tokens of its caller's
prints `scheduled, not measured (reduction)` in their place. The rate line reads "not printed while the drafter runs
on reference blocks (kept tokens per step: the acceptance line)": a pass multiplies in fp64 on copies, and a rate
of such a loop would be read as a speed. The bytes of a step are stated: the verify step's arithmetic at its tokens
per sequence -- every weight once, six embedding rows, one recurrent state read and six written, three history
columns read and eight written, the caches up to the step's last position -- and a draft pass's, five times.

What ran (2026-11-18; compute capability 10.0, 148 SMs; a shared GPU: correctness only; `native` with its library
rows in PyTorch 2.14.0 with FlashInfer 0.7.2.post1, every other stack in PyTorch 2.13.0; the attention preparation on
its reference chain on every stack). The seeded
two-layer model with its seeded drafter, one sequence, five drafted tokens, a prompt of 8 tokens and 32 new tokens
through 10 steps of `speculate(force=)` whose drafted tokens were scheduled to be kept 0 to 5 in turn: on `native`,
`fe-entries` and `fe-cublas` the 32 tokens were the plain session's, and what the drafter proposed in those steps was
recorded beside the tokens they verified; every gated line of the comparison passed, `drafter cache` and
`draft proposals` in 10 of 10 steps; replay with the proposals and the blocks of both kinds of step passed; the
kernel check passed over 10 traced steps -- the embedding lookup of every pass and the model's in the wiring order,
everything else the passes launch attributed to the drafter's fallback rows -- with 17 of the 20 block calls of the verify step on kernels on `native` and 13 of the 20 on `fe-entries`; the acceptance line read "scheduled, not measured (reduction)"; and a second prompt set
behind `reset(fresh=False)` gave the ids, the kept counts and the proposals of the first. On `reference` every line
passed and the run ended in `RAN`. One, three and seven drafted tokens and the four-layer model passed on `native`
as well.

In one process on `native`, twenty-one runs with one thing wrong each, between two runs of the tree as it is. A
first pass one position off, the pass of one prompt position left out, a drafted token where the kept one belongs
and the rows of an older step each failed `drafter cache`; the row select at row 0, a later pass fed the model's row
again, fed the proposal of two passes before, or one position off each failed `draft proposals` and passed
`drafter cache`; in all eight the ids were the plain session's and `tokens`, `prompt` and `rejected rows` passed. A
second lookup in the model's table inside a pass, a copy between two exact rows of the model and the pair read
twice each failed the kernel check with its own sentence. A forgotten kept count, a position moved by the count
alone, a row that attends over the whole window and a count taken from the schedule failed the lines that are there
for them, as they do without a drafter. So did three ways of feeding the prompt to something the verify steps do
not stand on -- a state of its own for the model's layers, another attention cache, another view of the history
columns: each failed `prompt` and passed `drafter cache` and `draft proposals`. With the whole state apart, the rows
a drafter reads among it, the session was refused at its load, by name: its warm-up step found no row behind its
prompt step. A drafted token emitted by the caller failed no line of the comparison --
the window of the next step is the session's -- and showed in the ids against the plain run's. A replay without the
prompt was refused by name: such a session has no step in front of its prompt.

On the published checkpoint (the cuDNN Frontend rows -- the rows `native` selected when this run was made and
`fe-entries` selects today without the attention preparation's row; PyTorch 2.13.0 --, one sequence, five drafted
tokens; the default prompt, 58 tokens through
`prefill`; then the loop above on the drafter's own proposals until the end token or 256 new tokens; the draft
vocabulary of this README's recipe as the runs of the section above made it -- the 342 ids of six dump files, then
the lowest ids; the default prompt is one of the three prompts behind those dumps): the loop ended at the end token
after 204 new tokens in 63 steps and kept 3.22 tokens a step (1 + 140 accepted drafted tokens / 63 steps, every step
counted in full); 14, 11, 11, 9, 10 and 8 steps kept one to six tokens; the first drafted token was accepted in 49
of 63 steps, the second in 38 of those 49, the third in 27 of 38, the fourth in 18 of 27, the fifth in 8 of 18. It
is a figure of this list and this prompt, reported; no line of it is a gate of this sample. The same loop with
the passes handed something else (reported): zeros in the place of the hidden row kept 1.04 tokens a step, the
hidden rows of another text 1.05, the two halves in front of the first multiply exchanged 1.00 -- without the
model's row the drafter adds next to nothing --; a later pass fed the model's row again kept 2.48; a later pass one
position off kept 3.25, as many as the loop itself: acceptance does not see that shift, `draft proposals` does. On
the first 16 steps `drafter cache`, `draft proposals`, `position`, `prompt`, `recorded` and `rejected rows` passed;
`tokens` was reported -- 38 of 46 kept tokens were the decode steps' greedy tokens on the same tokens, the first
that differs kept token 13 -- and the loop's text is another text than a plain run's from its 14th new token on: a
verify step and a decode step launch other kernels for some blocks, and a greedy run follows its first other token.
Replay with the proposals, the blocks of both kinds of step and the kernel check over 6 traced steps passed, with
370 of the 484 block calls of the verify step on kernels. The sentence asked for about
43.3 GiB, PyTorch reserved 40.0 GiB and the device grew by 40.2 GiB, the GPU otherwise empty.

With its library rows `native` runs three multiplies on other kernels -- inside the band of the reference multiply,
not to the bytes of a multiply over a copy -- and a greedy run on a checkpoint follows its first other token: the
plain run of the default prompt is another text there, and so is the loop. The same loop on `native` (PyTorch 2.14.0,
FlashInfer 0.7.2.post1; the attention preparation on its reference chain) ended at the end token after 231 new tokens in
71 steps and kept 3.27 tokens a step (1 + 161
accepted drafted tokens / 71 steps); 9, 16, 20, 9, 7 and 10 steps kept one to six tokens; the first drafted token was
accepted in 62 of 71 steps, the second in 46 of those 62, the third in 26 of 46, the fourth in 17 of 26, the fifth in
10 of 17. With the passes handed something else it kept 1.05, 1.04 and 1.00 tokens a step, 2.58 with a later pass
fed the model's row again and 3.31 with a later pass one position off. On the first 16 steps `drafter cache`,
`draft proposals`, `position`, `prompt`, `recorded` and `rejected rows` passed; `tokens` was reported -- 39 of 45 kept
tokens were the decode steps' greedy tokens on the same tokens, the first that differs kept token 17. Replay with the
proposals, the blocks of both kinds of step and the kernel check over 6 traced steps passed, with 451 of the 484 block calls of the verify step on kernels, the 16 out-projections of the attention layers on the library's own CUTLASS multiply. The sentence asked for about
32.2 GiB, PyTorch reserved 28.8 GiB and the device grew by 28.9 GiB, the GPU otherwise empty. The drafter's blocks
are not compared with another implementation of the drafter in this repository.

**Memory.** The memory sentence of a verify plan counts the state of its step shape -- a recurrent state per
(sequence, token), `T + 2` history columns per sequence -- beside the flat working memory of every plan of the
reduced models. The figures of this paragraph are those of the runs on the cuDNN Frontend rows, above. Up to 32 rows of
a step PyTorch reserved between 0.5 GiB less and 0.1 GiB more than the sentence said: one sequence of six tokens, with
the decode session on the same load, asked for about 3.6 GiB, reserved 3.6 GiB
and grew the device by 3.8 GiB; the verify session alone, which keeps the out-projection of the linear-attention
layer on its codes without a copy, asked for about 3.5 GiB, reserved 3.6 GiB and grew the device by 3.8 GiB; three
sequences of three tokens, with the decode session on the same load, asked for about 3.6 GiB, reserved 3.1 GiB and
grew the device by 3.3 GiB -- half a GiB below its sentence, in each of its runs; every other step shape up to 32
rows reserved 3.6 GiB. Beyond 32 rows the sentence of that day was short, as it was
for a decode plan beyond 33 sequences: 16 sequences of three, four and five
tokens -- 48, 64 and 80 rows -- asked for about 4.0, 4.0 and 4.4 GiB, reserved 4.3, 5.6 and 7.0 GiB and grew the
device by 4.5, 5.8 and 7.1 GiB, 0.5 to 2.7 GiB more than the sentence said. Since 2026-11-21 the sentence counts
what the checks of a session hold for every sequence beyond the first, a session of several tokens by its rows (the
section "Support per GPU class" has the count, its reach and the runs); the figures of this paragraph are those of
their day.

**What the entry states about one sequence alone.** `cudnn.GdnVerifyFused` is replay-identical, and it states no
batch invariance between one sequence and several: one sequence takes another association of the gated norm's sum
than two and more, so the last bits of its output may differ by design (`docs/operations/determinism.md`: "Replay
only"). The rows therefore gate no batch invariance on this block; a run with more than one sequence prints whether
the bytes were equal and, beside it, this statement.

**What is not here.** Kernels for the
drafter: every block of its passes but the embedding lookup is a reference chain, so no rate is printed for such a run. A drafter at more than one sequence. Without a drafter nothing decides how many tokens to keep:
the caller says it. One count
per step for all sequences: a step that keeps another number of tokens per sequence needs a position per sequence
in the attention blocks, which is not built. A decode step behind a keep: a session built with `draft_tokens=`
feeds decode steps for its prompt alone, in front of its first verify step. The report of a verify session
(`Session.report()`) states its step timings and its traced pass
and says of three lines that they are not stated -- the first-token line (nothing is prefilled through verify
steps), the rate (a step is not one token), and the bytes of a step (the family's arithmetic is the decode step's:
one embedding row, one recurrent state and three history columns per sequence). A session built with `draft_tokens=`
states its first-token line and prints no rate -- its drafted tokens are its caller's -- and the bytes of a step of
such a session are not stated either. With a drafter it states the bytes of its step and its acceptance. The
kernel check and the coverage line of such a session are about its verify
steps: the decode steps of its prompt are replayed in front of the traced window and their kernels are not observed,
which the run says beside its kernel verdict; what they launch was compared with a plain session's step in the runs
above, by the driver of those runs. In `fe-entries`, the attention core, the attention
layers' out-projection and the MLP's down multiply, which `native` runs on the library's kernels. A recorded verify
step: only the plain decode step is recorded and replayed. A verify step of more than eight tokens per sequence, of more than 16 sequences, or of more than 4 at six tokens
and more: declined by name. The verify step
has run on compute capability 10.0 only; every other class is uncovered, not passing -- the rows of the
out-projection on compute capability 10.3 and 10.7 among them.

## Speculative decoding

`02_speculative_decode.py` runs speculative decoding with the checkpoint's drafter in one process, for one request. The
prompt goes through decode steps; then every step is the drafter's passes -- five by default -- ONE verify step of the
drafted tokens behind the token the run kept last, one host read, and the acceptance on the host: the loop keeps a
drafted token while it equals the greedy token of the row in front of it. The drafter is the one of the section "The
checkpoint's drafter", every block of it a reference chain in every stack but its embedding lookup; the step is
`Session.speculate` of the section "The verify step". The loop itself is in the script.

```bash
python samples/models/qwen_dense/02_speculative_decode.py                      # the stack native: 2 layers, seeded weights, a seeded drafter: self-checking
python samples/models/qwen_dense/02_speculative_decode.py --stack fe-entries   # a functional-validation stack: no package beyond cuDNN Frontend and PyTorch

# with a checkpoint: the packages and the download of the first command block of this page, and a draft vocabulary
python samples/models/qwen_dense/02_speculative_decode.py --model /data/Qwen3.8-27B-NVFP4 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 5, "use_local_argmax_reduction": true}' \
    --draft-vocab draft_vocab.json --messages request.json --max-tokens 256 > answer.txt
```

| you need | |
|---|---|
| for the default run | this checkout and a GPU of compute capability 10.x or 12.x: no checkpoint. The stack is `native`, as for `00_decode.py` |
| for the stack `native` on compute capability 10.x | FlashInfer at a validated version (the section "The rows a library serves"): the verify step and the decode steps call its multiplies on the stored weights. Without it the run stops before anything is loaded with `[02] ERROR the stack native cannot run:` and the package's name, exit code 1; `--stack fe-entries`, a functional-validation stack, runs the cuDNN Frontend rows without the package |
| with `--model` | the packages of `samples/models/requirements.txt`, the checkpoint on local disk (22 GB), and a draft vocabulary: a JSON list of token ids, `--draft-vocab FILE`. The recipe of the section "The checkpoint's drafter" makes one from the files `--dump-tokens` writes, of this script or of `00_decode.py`. Without a list the run ends in one `ERROR` line that names that recipe |
| GPU memory with the 27B checkpoint | with a list of 65,536 ids and the default context length a run asks for about 61 GiB free on the stack `fe-cublas` (about 64 GiB on compute capability 12.x); on the stack `fe-entries` for about 44 GiB on compute capability 10.0, about 48 GiB on the other 10.x classes and about 64 GiB elsewhere; on the stack `native`, whose library rows need FlashInfer and its kernel-binary package (the section "The rows a library serves"), for about 33 GiB and about 37 GiB there. That is the model as `00_decode.py` holds it, a state of six tokens per sequence, and the drafter with the bf16 copies its reference blocks multiply on. These are arithmetic on stated facts; the runs at the end of this section measured compute capability 10.0 |

**What a run is.** A prompt of N tokens is N decode steps and N - 1 passes of the drafter before the first token; then
every step is five draft passes and one verify step of six tokens, and keeps one to six of them. A run says so before it
loads anything, as a rule and counts, never as a time:

```text
[02] prompt of 58 tokens = 58 decode steps and 57 passes of the drafter before the first token; then every step is 5 draft passes and one verify step of 6 tokens, and keeps 1 to 6 of them
```

Every token of a speculative run after the first is a greedy token of a verify step; the first is the greedy token of
the prompt's last decode step. The drafter decides how many tokens a step keeps, never which. In the stacks `native` and
`fe-entries` on compute capability 10.0 a verify step and a decode step launch different kernels for two blocks (the linear-attention step, and the out-projection of the
linear-attention layers up to 16 rows of a step); in `fe-cublas` for one, the linear-attention step. The speculative run
is compared with decode steps of the same tokens at the steps the family's declared band decides. On a checkpoint, which
declares no band, the comparison is reported: there a verify step and a decode step of the same tokens do not choose the
same greedy token everywhere (this page has the run), so the text of a speculative run and the text of the plain script
part early; each is the greedy text of its own kind of step.

The generated text goes to standard output piece by piece: the tokens a step kept are written between its host read and
the next launch. Tokens of a step behind an end token or the limit of new tokens are not written. Every other line of
a run goes to standard error, as for `00_decode.py`.

**The default run** needs no checkpoint: the reduced seeded model with its seeded drafter (`QwenDenseDrafter.seeded`,
a draft vocabulary of a quarter of the ids drawn from the seed). Seeded weights draft nothing a model accepts, so the
run prints one more reduction -- "acceptance is scheduled, not measured: the drafted tokens of every step are replaced
by the plain run's tokens for 0, 1, .., 5 positions in turn and by another token behind them" -- and schedules: a plain
session of the same process decodes the token stream first, and every step verifies drafted tokens that are cut from
it (`speculate(force=)`, `tie_out.forced_drafts`). The drafter still runs every pass and its own proposals are
recorded; the count a step keeps is still the comparison's. With the default 32 new tokens that is 10 steps that keep
0, 1, 2, 3, 4, 5, 0, 1, 2, 3 drafted tokens. The acceptance line of such a run reads `scheduled, not measured
(reduction)`. A run of fewer new tokens than the schedule needs to keep every count once (17 at five drafted tokens)
is an `ERROR` before the plan; left to its default, the count follows the schedule.

| flag | default | meaning |
|---|---|---|
| `--speculative-config JSON` | `{"method": "mtp", "num_speculative_tokens": 5, "use_local_argmax_reduction": true}` | the speculative configuration, as the text of a JSON object; every key is answered (the table below), and a run prints one line for it |
| `--draft-vocab FILE` | -- | with `--model`, required: the JSON list of token ids the drafter proposes from. Without `--model` the flag is an error: the seeded run draws its list from the seed |
| `--sequences 1` | `1` | accepted at 1 only: this script serves one request. Another count is one `ERROR` line before the plan: "a session with a drafter serves one sequence: one request, as the drafter rows are for" |
| `--force-tokens N` | `32` without `--model`, or what the schedule needs if that is more; off with it | produce exactly N new tokens, ignoring the end token |
| `--stack`, `--provider BLOCK=KEY`, `--check-kernels` | as in `00_decode.py` | the stack and the rows of the MODEL's blocks, in both kinds of its steps: a row selected with `--provider` must serve the verify step and the decode step, or the plan declines by that row's sentence. A block of the drafter has one row; naming one in `--provider` is a usage error |
| `--dump-tokens FILE` | -- | the file of `00_decode.py` with `"sample": "qwen_dense/02_speculative_decode"`, the drafted tokens of a step (`draft_tokens`) and, per prompt, `kept`: the drafted tokens every step kept |
| `--model`, `--prompt`, `--messages`, `--max-tokens`, `--max-model-len`, `--temperature`, `--text-out`, `--no-stream`, `--layers`, `--reference-precision` | as in `00_decode.py` | the flag table at the top of this page; a request body is answered field by field ("A request body and its options"); under `--max-model-len` the limit of new tokens is held together with the rows the last verify step writes behind the last new token (one fewer than the drafted tokens), so the limit that fits is that many smaller than in `00_decode.py` |
| `--dump-census FILE` | -- | not in this script: a usage error. `00_decode.py` writes the launch records of a decode step |

**The speculative configuration.** `--speculative-config` takes the JSON object that says how tokens are drafted,
as a client writes it. A key of this object selects what is launched, so -- unlike a field of
a request body -- nothing of it is printed and ignored: a key is read, accepted at the value that changes nothing (and
at `null`, which is the key left out), or refused, each by name (`serving_options.py` holds the three tables as data):

| key | answer | accepted | another value is refused because |
|---|---|---|---|
| `method` | read; must be named | `"mtp"` | the drafter is the checkpoint's own, the method mtp; no other method is served |
| `num_speculative_tokens` | read; must be named | a whole number from 1 | it is the number of tokens a step drafts: a whole number from 1 |
| `use_local_argmax_reduction` | read; must be named | `true` | the draft token is the largest of the listed rows of the head: the one form of the drafter's head this sample runs |
| `rejection_sample_method` | accepted at the value that changes nothing | `"standard"`, `null` | a drafted token is kept while it equals the greedy token in front of it: nothing is drawn |
| `draft_sample_method` | accepted at the value that changes nothing | `"greedy"`, `null` | the drafter proposes the token of its largest logit |
| `enable_adaptive_verification` | accepted at the value that changes nothing | `false`, `null` | every step verifies every token it drafted |
| `disable_padded_drafter_batch` | accepted at the value that changes nothing | `false`, `null` | one request: there is no batch of drafts to pad |
| `parallel_drafting` | accepted at the value that changes nothing | `false`, `null` | the drafter's passes run one after the other, each on the proposal of the pass before |
| `use_heterogeneous_vocab` | accepted at the value that changes nothing | `false`, `null` | the drafter proposes ids of the model's own vocabulary |
| `model`, `quantization`, `revision`, `code_revision`, `draft_load_config`, `draft_lm_head_quantization` | refused at any value | `null` | the drafter is the checkpoint's own, loaded as its loader loads it |
| `index_share_for_mtp_iteration` | refused at any value | `null` | it overrides a field of the drafter's own configuration: the drafter runs as its checkpoint describes it |
| `max_model_len` | refused at any value | `null` | the drafter has no context length of its own: its cache has the positions of the model's (--max-model-len) |
| `synthetic_acceptance_rates`, `synthetic_acceptance_length`, `fly_window_size`, `fly_entropy_threshold`, `fly_entropy_top_k` | refused at any value | `null` | the tokens a step keeps are decided by the greedy comparison: an acceptance that is drawn, scheduled or loosened is not served |
| `num_speculative_tokens_per_batch_size` | refused at any value | `null` | one request: the number of drafted tokens is one number (num_speculative_tokens) |
| `enforce_eager`, `tensor_parallel_size`, `draft_tensor_parallel_size` | refused at any value | `null` | one process on one GPU, launched from Python |
| `moe_backend`, `attention_backend`, `kv_cache_dtype` | refused at any value | `null` | which kernel serves a block is a row of the table a run prints (--stack, --provider) |
| `disable_eagle_block_drop`, `disable_lookahead_block_hashes`, `prompt_lookup_max`, `prompt_lookup_min`, `suffix_decoding_max_tree_depth`, `suffix_decoding_max_cached_requests`, `suffix_decoding_max_spec_factor`, `suffix_decoding_min_token_prob`, `dspark_draft_topk`, `xpress_topc` | refused at any value | `null` | it is an option of another speculative method, or of a cache keyed by content, which a sample does not have |

A key no row holds is an `ERROR` that names it. The three keys that are read must be named: this sample has no
default for a key that selects the method or the launch set. The count of drafted tokens is any whole number from 1 to
the parser; which counts a stack serves is its rows' answer, at the plan: a provider stack holds a verify step of two to
eight tokens per sequence, so eight drafted tokens end in a `SKIP` with that row's sentence before anything is loaded,
and the stack `reference` serves any count (the seeded run as far as the new tokens its schedule needs fit the context
length). Another count runs the same way, its kernels checked against the plan of its own step shape. The key names and the values of
this table are those of the public JSON object of this name as it stood on 2026-10-31. A run prints one line for the
object, with the names of its keys and the values the table knows:

```text
[02] --speculative-config: read: method "mtp", num_speculative_tokens 5, use_local_argmax_reduction true. At the value that changes nothing: draft_sample_method "greedy".
[02] ERROR --speculative-config: "use_local_argmax_reduction" is not named: the draft token is the largest of the listed rows of the head: the one form of the drafter's head this sample runs. Accepted: true
```

**What a run reports.** The report of `00_decode.py`, and in the place of its rate two acceptance lines: `kept tokens
per step X.XX of 6 (1 + accepted drafted tokens / steps), n steps; kept 1: .., .. 6: ..` and `accepted per draft: 1: x
of n, 2: y of x, ..` -- per drafted token the steps that kept it among those that kept the one before. Every step counts
in full, the last one included, whose tokens behind an end token are not written. The acceptance length is a figure of
this run's prompts and of this draft-vocabulary list; it is not comparable with a figure made with another list. Nor
with a figure of another stack: a greedy run follows the first token its stack chooses otherwise (this section's last
runs have three). The rate line reads "not printed while the drafter runs on reference blocks (kept tokens per step: the
acceptance line)": a draft pass multiplies in fp64 on copies, and a rate of such a loop would be read as a speed. The
`bytes per step` line is the verify step's arithmetic at six tokens per sequence and a draft pass's, five times. The
`text out` line says how many tokens came out in how many steps. The report is printed before the checks.

**What a run proves, per stack.** The default run proves the loop on the seeded model and says per stack what it
gated. The lines are those of `tie_out.speculative_checks` (the section "The verify step" has each), one line of the
script's own, and the checks of every run.

| stack | what is gated |
|---|---|
| every stack | `tokens`: every kept token is the decode step's on the same tokens, inside the declared band, and the loop's greedy rule holds -- every window starts with the greedy token kept last, every kept drafted token is the greedy token of the row in front of it. `ids`, the script's own line: the new tokens are the plain session's of the same process (a drafted token that a loop emitted reaches no other line). `acceptance`: every step kept the count it was scheduled to keep, and every count from 0 to 5 was kept. `position`: the session stands where the decode steps stand, an integer. `prompt`: after the prompt, what every layer stands on and the logits of every prompt step are the decode session's bytes. `rejected rows`: a second replay with other tokens behind the kept ones is byte-identical -- a kept row does not read what a rejected draft wrote. `drafter cache`, `draft proposals`: what the drafter's passes are handed. `kept state`: the recurrent state of a linear-attention layer both runs fed the same rows, inside the fused step's band. `replay`, with the drafter's proposals, byte-identical |
| `reference` | also `bytes`: the logits of every kept row and what every layer stands on after every step are the decode steps' bytes. The run compares the reference with itself block by block and ends in `[02] RAN reference stack (no provider checked)`, or in `[02] FAIL` |
| `fe-cublas`, `native`, `fe-entries` | also the blocks of both kinds of step against the reference, teacher-forced; `bytes` is reported |
| `native` and `fe-entries`, and any stack with `--check-kernels` | also the kernel check of whole steps: the lookups of the drafter's passes and the verify plan's rows one by one, everything else a pass launches attributed to the drafter's fallback rows |

On `native`, at one sequence of six tokens on compute capability 10.0 with 148 SMs, a run proves what the verify step
proves -- on the 64-layer model 484 of the 485 block calls of a verify step on kernels, on the two-layer model of the default run 20 of 21; on `fe-entries`, without
the library rows, 387 of the 485 and 15 of 21 -- now
behind a real
prefill, in one session, with kept counts the comparison decides. No block of the drafter is on a kernel yet: the last line of a run before the verdict is "the
drafter ran on reference blocks: none of its blocks was compared with a provider (no provider serves them yet)".

On a checkpoint `position`, `prompt`, `recorded`, `rejected rows`, `drafter cache`, `draft proposals`, replay, the
blocks and the kernel check are gated as on the seeded model; `tokens` is reported, with the first kept token that
differs; nothing is scheduled and no plain run is made for the ids, so the tokens a checkpoint run writes are held by no
line of that run: the loop that writes them is the one the default run proves. The comparison covers the first 16 steps
behind the prompt, the block check the first 16 prompt steps and the first 8 steps, the traced pass 16 steps. The last
line of a run is the verdict of `00_decode.py` with the tag `[02]` ([../README.md](../README.md) has the table).

**What ran** (2026-11-19; PyTorch 2.13.0; compute capability 10.0, 148 SMs; a shared GPU: correctness only; `native` on
the cuDNN Frontend rows -- the rows it selected when these runs were made and `fe-entries` selects today without the
attention preparation's row). The
default run -- the seeded two-layer model with its seeded drafter, a prompt of 26 tokens, then 32 new tokens in 10
steps scheduled to keep 0, 1, 2, 3, 4, 5, 0, 1, 2 and 3 drafted tokens -- ended in `[02] PASS` on `fe-cublas`,
`native` and `fe-entries` and in `[02] RAN reference stack (no provider checked)` on `reference`. In each of the four
the 32 new tokens were the plain run's of the same process (ids hash `11bfb9f9bcf5`, which is also what `00_decode.py
--sequences 1 --force-tokens 32 --stack native` printed), nothing was written to standard output, and every gated line
of the table above passed, `acceptance` with 10 of 10 steps on their scheduled count. `bytes` passed on `reference`
and was reported on the other three. On `native` and `fe-entries` the kernel check passed over 10 traced steps, with
13 of the 20 block calls of the verify step on kernels. On `native`, three drafted tokens (`"num_speculative_tokens": 3`) passed in 13 steps; eight ended in `[02] SKIP cudnn.GdnVerifyFused serves 8 tokens per
sequence of 2-8, sequences 1-4; this run has 1 sequences x 9 tokens per sequence` before anything was loaded. The
four-layer model (`--layers 4`, on the stack `fe-cublas`) passed. The memory sentence of the default run asked for about
6.3 GiB on `fe-cublas` and `reference` and about 6.1 GiB on `native` and `fe-entries`; PyTorch reserved between 3.8
and 4.6 GiB in the four runs, and the device grew by about 0.2 GiB more in each.

In one process on `native`, three runs of the script with one thing wrong in its loop each, between two runs of the
tree as it is, which passed; each of the three ended in `[02] FAIL`. A drafted token written out where the verify step
chose another failed `ids` alone; one drafted token kept beyond the compared count failed `tokens`, `drafter cache`,
`draft proposals`, `acceptance` and `ids`; one kept below it failed `acceptance` alone.

With its library rows (2026-11-20; `native` in PyTorch 2.14.0 with FlashInfer 0.7.2.post1; the same GPU class; the
attention preparation on its reference chain) the
default run on `native` ended in `[02] PASS` with the same 32 new tokens: ids hash `11bfb9f9bcf5`, the plain run's of
its process. Every gated line of the table above passed, `acceptance` with 10 of 10 steps on their scheduled count,
and the kernel check passed over 10 traced steps, with 17 of the 20 block calls of the verify step on kernels. Three drafted tokens passed in 13 steps. The memory sentence of the default run asked for about
5.7 GiB. Without FlashInfer installed the same command ended in `[02] SKIP flashinfer.gemm.bmm_fp8 needs
flashinfer-python 0.7.2.post1, which is not installed: ...` before anything was loaded, exit code 0.

**The last runs of this recipe with a checkpoint** are those with the attention preparation on its entry, on `native`
and on `fe-entries` (2026-11-20; the same checkpoint, command, prompt and list as the two runs below; compute
capability 10.0, 148 SMs): the section "The attention preparation: pages, slots and positions" has their figures. The
two runs below were made before that row and are kept as the records of their row sets.

**The run with a checkpoint on the cuDNN Frontend rows, before the attention preparation's row** (2026-11-19; the
published checkpoint; cuDNN Frontend
1.34.0, CuTe DSL 4.8.0, PyTorch 2.13.0; compute capability 10.0, 148 SMs; a shared GPU: correctness only; `native`
selected those rows then, as `fe-entries` does today without the attention preparation's row; the third command of this
section as it is
written, and once more on the stack `fe-cublas`: the default prompt as a chat request body with `max_tokens`
256, 58 tokens through the prompt, five drafted tokens, the draft vocabulary of this README's recipe with 65,536 ids).
Both runs ended in `[02] PASS`, and both stopped at the end token. On `native` the loop wrote 204 new tokens in 63
steps: 3.22 kept tokens a step (1 + 140 accepted drafted tokens / 63 steps); 14, 11, 11, 9, 10 and 8 steps kept one to
six tokens; the first drafted token was accepted in 49 of 63 steps, the second in 38 of those 49, the third in 27 of 38,
the fourth in 18 of 27, the fifth in 8 of 18 -- the counts of the library loop in the section "The verify step", on the
same prompt and list. On `fe-cublas` it wrote 249 new tokens in 74 steps: 3.38 kept tokens a step (1 + 176 / 74; the
last step kept six tokens and the end token was its fourth, so the last two of them were not written); 13, 14,
14, 10, 11 and 12 steps kept one to six tokens; 61 of 74, 47 of 61, 33 of 47, 23 of 33 and 12 of 23. The stacks launch
other kernels for the same blocks and a greedy run follows its first other token: an acceptance length is a figure of
its stack as well as of its prompt and its list. In both runs `position`, `prompt`, `recorded`, `rejected rows`,
`drafter cache` and `draft proposals` passed on the first 16 steps, and replay with the proposals and the blocks of both
kinds of step passed; `tokens` was reported -- on `native` 38 of 46 kept tokens were the decode steps' on the same
tokens, the first that differs kept token 13; on `fe-cublas` 43 of 49, the first kept token 30. On `native` the kernel
check passed over 16 traced steps, with 370 of the 484 block calls of the verify step on kernels. The memory sentence asked for about 43.3 GiB on `native`, where PyTorch reserved 39.8 GiB and the device grew by
40.0 GiB, and for about 61.0 GiB on `fe-cublas`, where PyTorch reserved 58.5 GiB and the device grew by 58.7 GiB; the
GPU was otherwise empty at the start of each.

**The run with a checkpoint on `native` with its library rows, before the attention preparation's row** (2026-11-20;
PyTorch 2.14.0, FlashInfer
0.7.2.post1; the same checkpoint, command, prompt, list and GPU class; the attention preparation on its reference
chain), made twice with the same figures. Both runs
ended in `[02] PASS` at the end token. The loop wrote 231 new tokens in 71 steps: 3.27 kept tokens a step (1 + 161
accepted drafted tokens / 71 steps); 9, 16, 20, 9, 7 and 10 steps kept one to six tokens; the first drafted token was
accepted in 62 of 71 steps, the second in 46 of those 62, the third in 26 of 46, the fourth in 17 of 26, the fifth in
10 of 17 -- the counts of the library loop on `native` in the section "The verify step". `position`, `prompt`,
`recorded`, `rejected rows`, `drafter cache` and `draft proposals` passed on the first 16 steps, and replay with the
proposals and the blocks of both kinds of step passed; `tokens` was reported -- 39 of 45 kept tokens were the decode
steps' on the same tokens, the first that differs kept token 17. The kernel check passed over 16 traced steps, with 451 of the 484 block calls of the verify step on kernels, the 16 out-projections of the attention layers on the library's own CUTLASS multiply. The memory
sentence asked for about 32.2 GiB.

**What is not here.** The drafter on kernels: every block of it but its embedding lookup is a reference chain. More than one request: `--sequences` is
accepted at 1. A rate. A count or a position per sequence. A recorded step: every step of this script is launched from Python, and a run says so in one line. `--dump-census`. A text equal to the plain
script's on a checkpoint. Any GPU class but compute capability 10.0: the script has run there only, and a class it has
not run on is uncovered, not passing.

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
| 4-bit quantise of the output head's operand | 1 | `torch_ref.quantize` | torch_ref.quantize, fallback: reference chain |
| output head | 1 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 4-bit weights |
| token choice | 1 | `torch.argmax` | torch.argmax |

485 block calls per step on the full model, 21 on the two-layer model and 35 on the four-layer one. By provider: cuDNN
Frontend 1; fallbacks 9 (cuBLAS over copies 4, PyTorch chains 5); PyTorch as the natural provider 2.

## Who serves each block (stacks `native` and `fe-entries`, compute capability 10.0)

| block | calls per step (64 layers) | provider key | entry [form] |
|---|---|---|---|
| token embedding lookup | 1 | `torch.index_select` | torch.index_select |
| residual add + RMSNorm + quantise | 129 | `fe.add_norm_quant_decode` | cudnn.AddNormQuantDecode [PDL True, PDL_RELEASE True] + one of 6 forms, by call site |
| 8-bit in-projection (linear attention QKVZ / attention QKV) | 48 / 16 | `fe.fp8_projection_decode` | cudnn.Fp8ProjectionDecode [SPLIT_K 1, MMA_M 128, STAGES 8, TMEM_COLS 32, PREWAIT_WEIGHTS True, HOLD_TMEM True, PDL True, PDL_RELEASE True] + one of 2 forms, by call site |
| linear-attention step | 48 | `fe.gdn_decode_fused` | cudnn.GdnDecodeFused [PDL True, PDL_RELEASE True] |
| 8-bit out-projection | 48 / 16 | `flashinfer.bmm_fp8` | flashinfer.gemm.bmm_fp8 [BACKEND auto] |
| MLP gate / up + SiLU x up + quantise | 64 | `fe.swiglu_fp4_decode` | cudnn.SwigluFp4Decode [TOKEN_TILE 8, PREWAIT_STAGES 2, L2_EVICT_FIRST True, MAX_CLUSTERS 148, PDL True] |
| MLP down | 64 | `flashinfer.mm_fp4` | flashinfer.gemm.mm_fp4 [BACKEND cute-dsl, TUNED False] |
| attention preparation | 16 | `fe.attention_prepare_decode` | cudnn.AttentionPrepareDecode [POSITIONS 3d, SECTION (11, 11, 10), PAGE_SIZE 64, POSITION_DTYPE int64, SLOT_DTYPE int64, PDL False, PDL_RELEASE False] |
| attention core | 16 | `flashinfer.decode_attention` | flashinfer.decode.trtllm_batch_decode_with_kv_cache [KV_LAYOUT HND] |
| gate + quantise | 16 | `fe.attention_gate_quant_decode` | cudnn.AttentionGateQuantDecode [THREADS 256, PDL_RELEASE True] |
| 4-bit quantise of the output head's operand | 1 | `fe.row_quantize_fp4_decode` | cudnn.RowQuantizeFp4Decode [WIDTH 5120, PDL True, PDL_RELEASE True] |
| output head | 1 | `flashinfer.mm_fp4_head` | flashinfer.gemm.mm_fp4 [BACKEND cute-dsl, TUNED False] |
| token choice | 1 | `torch.argmax` | torch.argmax |

By provider: cuDNN Frontend 7; FlashInfer 4; PyTorch as the natural provider 2; 4 rows
unverified in CI.
The table is the one of `native`. `fe-entries` has the same rows but FlashInfer's four, which are the reference chain
of the attention core over the pool of pages and its
three multiplies over a copy of `fe-cublas`: cuDNN Frontend 7; fallbacks 4 (cuBLAS over copies 3, PyTorch chains 1);
PyTorch as the natural provider 2.
The two 8-bit projections count as one kind of block per provider, and on this class they have two: the in-projection
is cuDNN Frontend's, the out-projection FlashInfer's (a multiply over a copy in `fe-entries`). The six forms of the norm
row and the two weight
shapes of the projection row, by call site, are in the section on the stack `native`. The row of the MLP's gate / up
multiply is the one of 1 to 8 rows; three more rows of the same key name the other token tiles: `TOKEN_TILE 16` from 9
to 16 rows; `TOKEN_TILE 32` from 17 to 32 rows; `TOKEN_TILE 64` from 33 to 64 rows. From 65 rows the block is the
multiply over a copy of `fe-cublas`. On the other 10.x classes the
in-projection is the multiply over a copy of `fe-cublas`: cuDNN Frontend 6; FlashInfer 4; fallbacks 1 (cuBLAS over
copies 1); 4 rows unverified in CI. On compute capability 12.x no row of the library serves, so the
three multiplies are over copies again; the gate / up multiply is over a copy too, and the gate + quantise row, the
attention preparation row, the norm row and the row of the head's quantise are the reference chains of `fe-cublas` as
well: cuDNN Frontend 1; fallbacks 9 (cuBLAS over copies 4, PyTorch chains 5).

| GPU classes | sequences | regime | note |
|---|---|---|---|
| any | 1 and more | launch-bound | a step is one launch or more per block, issued from Python or replayed as one recorded sequence; no byte-bound form is declared, and the sequence count at which one would win is not measured on any class |

## The checks of a run

Against the plain-PyTorch reference of every block, which keeps the chain's casts and accumulates in fp64
(`model_samples/tie_out.py` has the definitions):

| check | what must hold |
|---|---|
| replay | the same inputs from a fresh state give byte-identical ids and logits |
| blocks, teacher-forced | every value finite; every multiply within one bf16 rounding of the reference, element by element; the linear-attention step inside its band (below); in the stacks `native` and `fe-entries`, the gate + quantise entry, the add + RMSNorm + quantise entry and the attention preparation entry each inside its band against its twin, and the fused gate / up entry inside the error bound of its contract around the fp64 chain (below) |
| tokens | the logits of the two stacks, each run end to end on its own state, are finite. On the reduced seeded model they are also within the declared logits band (`configs.py`: 4 bf16 units of the run's largest reference logit), and the greedy token equals the reference stack's at every step whose margin exceeds twice that band; steps below it are counted and printed as near-ties. On a checkpoint no band is declared and the line is `REPORTED` (the section on checkpoints says why). The largest logit distance and the number of equal greedy tokens are printed either way |
| batch invariance | one sequence alone gives the bytes it has among 4; the sequence that runs alone rotates with the step, so every row of the batch is covered: a gate for the linear-attention step on compute capability 10.0, 10.3 and 12.0 (in its `native` form on 10.0, the class that form has run on) and for the gate + quantise entry, the add + RMSNorm + quantise entry, the attention preparation entry, the FP8 projection entry and the fused gate / up entry on 10.0, a printed fact for every other block and class |

The band of the linear-attention step is judged twice. Over the whole run: more than 99.9 % of its 8-bit output bytes
equal, more than 99.9 % of its recurrent state within 1e-4 and all of it within two bf16 units of the state's largest
value, every value finite, the conv history byte-identical and the null slot untouched. And per cell, which is one value
head of one sequence in one layer (one block of the step's launch grid, so its natural unit of failure): over the run more
than 99 % of the cell's output bytes equal and more than 99.9 % of the cell's state within 1e-4.

The state band has a share and a cap because a state "within 1e-4 everywhere" does not hold against an fp64-accumulated
reference: the fused step rounds its gates and conv taps to bf16 as the reference chain does, and where its fp32 sum falls on the
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

The attention preparation entry is compared with its twin as well, the preparation written at the kernel's rounding
points (`AttentionPrepareRounded`), and its band (`attention_prepare`) is the numerics contract on its page
(`docs/operations/attention/attention_prepare_decode.md`). The twin takes the mean of the squares and its reciprocal
root in fp64 and rounds them to fp32 once, forms the two products of the norm in fp32 and rounds the normed value once
to bf16; a rotated value is one fp32 product and one fused sum, rounded once. Every query and key element must be within
one bf16 unit of the twin's value, taken at the larger of the two magnitudes, and a rotated element also within what its
two normed operands may move it by, each one unit off: the cosine times a unit of the first and the sine times a unit of
the second. The gate rows and the value rows must be the twin's bytes. The key and the value row of every row of the
step must stand at the page and the row of its position, which the judge forms from host integers -- the position of the
call, the sequences and the shape of the pool -- and from no table of the cache; and every other row of the pool must be
what it was before the call. Every value the call wrote must be finite: the queries, the gate rows, and the key and the
value rows of the step. The gate rows and the value rows are held finite beside their bytes, because the entry and the
twin copy them from the same operand: a NaN or an infinity in it is the same byte on both sides. The line prints how
many elements are not the twin's byte; one element outside its band, one gate or value bit, one copied value that is
not finite, or one row of the pool written where no row of the step belongs fails the block.

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
| both kinds, at different sequence counts, in the two kinds of step of one load, or through `--provider` | both |

So a weight that every plan of the run multiplies on its stored format has no copy. Today three kinds of row do that:
the FP8 projection entry for the in-projections on compute capability 10.0, the fused gate / up entry for the MLP's
first multiply on compute capability 10.x, and -- in a step of several tokens per sequence -- the FP8 projection entry
for the out-projection of the linear-attention layers. That last block is served by two rows, by call site, so its
forms are kept per call site: the linear-attention layers' weights as their row reads them, the attention layers' as
copies (the section "The verify step"). The second keeps its weight in the layout its kernel multiplies on, gate and up
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
-- the largest such copy, one at a time, beside what the checks hold for the sequences beyond the first ("Support per
GPU class") -- and the checks print what the copies cost:

```text
  this stack needs about 3.7 GiB on this GPU for 1 and 4 sequences, one count after the other, 0.47 GiB of it for the checks (the reference's bf16 copy of one weight at a time, where the run keeps none, and what they hold for the sequences beyond the first of a count); 177.7 GiB are free
  stored weights    4 of 4 weights this run keeps on their stored format alone hold the bytes of their load, before the checks and after them: the reference's copies were made from what was loaded    PASS
  on demand         the reference read 460 bf16 copies of weights this run keeps on their stored format alone (projection.fp8.in, mlp.gate_up): each made for one call and dropped after it; the largest 0.33 GiB, 110.1 GiB in all, 1.8 s of the checks (orientation only)
```

A run in which every weight has its copy prints the memory sentence without that part and neither of the two lines:
the output of the stack `fe-cublas` is what it was. Only the reference session of the checks makes copies per call: a session
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
why the form is compared on the objects and the name only against the entry's own symbol. Registers per thread, and
the start and the end of every launch, are recorded in `--dump-census` and compared with nothing. Nothing the check
says is a timing.

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

The seeded run of the stack `fe-cublas` with `--check-kernels` was last run on compute capability 10.0 (2026-11-13, a 148-SM part, PyTorch
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

| compute capability | what has run on it: the seeded run on the stack `fe-cublas`, the default run until the stack `native` became it, unless a stack is named |
|---|---|
| 10.0 (data-center Blackwell, for example a GB200) | `[00] PASS`, verified at 1 and 4 sequences; batch invariance of the fused step gated and equal; the checkpoint run of the section above. With `--check-kernels`: `[00] PASS` at 1 and 4 sequences, the only class the kernel check has run on. The stacks `native` and `fe-entries`: `[00] PASS` at 1 and 4 sequences and with the checkpoint, batch invariance of the fused step, of the gate entry, of the norm entry, of the projection entry and of the fused gate / up entry gated and equal; the only class these stacks have run on. The default run -- the stack `native`, its step recorded and replayed -- `[00] PASS` at 1 and 4 sequences and with the checkpoint (2026-11-24): the only class a recorded step has run on |
| 10.3 (for example a GB300) | `[00] PASS`, verified at 1 and 4 sequences; batch invariance of the fused step gated and equal; a checkpoint run before the one of the section above. The default run, on the stack `native`, has not run on this class |
| 10.7 (a VR200-class part) | `[00] PASS`; batch invariance of the fused step is printed on this class (`REPORTED`), not gated: every block was equal in the runs so far. The default run, on the stack `native`, has not run on this class |
| 12.0 (for example an RTX 5090) | `[00] PASS`, verified at 1 and 4 sequences; batch invariance of the fused step gated and equal. The checkpoint does not fit a 32 GB card: on one, the sample printed the memory sentence (about 59.7 GiB needed) and exited 0. The default run, on the stack `native`, has not run on this class |
| any other | `[00] SKIP`, by design. `--stack reference` runs the reduced model there on PyTorch alone, with the whole report, and ends in `[00] RAN reference stack (no provider checked)`: nothing is verified (seen on compute capability 8.0) |

A class the current checks have not run on is uncovered, not passing. That holds for the stacks `native` and
`fe-entries` on every class but 10.0: the gate entry, the norm entry, the projection entry, the fused gate / up entry
and the attention preparation entry inside this sample, the fused step in its wait + release form, and the kernel
patterns of the check have run on 10.0 only. The norm entry admits every compute
capability 10.x; its own page lists 10.3 and 10.7 as not run on a device. So does the page of the fused gate / up
entry, whose rows here claim every 10.x class as the entry does: on 10.3 and 10.7 the rows, the weight in the kernel's
row order and the memory sentence of that plan are unverified. The row of the attention preparation entry claims every
10.x class in the same way, and its entry's page lists 10.3 and 10.7 as not run on a device: there the row, the paged
cache of its plan and the reads of the attention core on that cache are unverified. The rows of the projection entry for
the
out-projection of a verify step claim 10.3 and 10.7 as well, in the split of their rows there, and are as unverified: the entry's page lists that weight shape as not run on those classes. The four rows a library serves claim
every 10.x class too and have run on 10.0 alone: on 10.3 and 10.7 the rows are unrun, and which kernel the library picks there for either multiply or for the attention core has not been seen. So has the build of the entries' kernels while
a run is
planned: what a plan prints on a GPU for which the installed CuTe DSL cannot build (a `SKIP` in the entry's own words) is
held by a host test over a stand-in entry and was seen on a real GPU only with the entry's own toolchain question made to
decline, not on such an installation.

Memory of the seeded run on the stack `fe-cublas` (2 layers, 1 and 4 sequences): 1.7 GiB of bf16 copies; PyTorch held 2.5 GiB at its peak,
reserved 3.5 GiB and grew the device by 3.7 GiB on a compute capability 10.0 GPU. The sample asks for about 3.9 GiB
free there: 3.7 GiB for the weights, the state and the working memory, and 0.2 GiB for what its checks hold for the
three sequences beyond the first of the larger count (below). On compute capability 12.x
it adds an allowance of 3 GiB and asks for about 6.9 GiB: the first launch of the fused step reserves device memory outside
PyTorch's pool on that class. The need is stated once, for every sequence count of the run, before anything is loaded: the
copies, the working memory and the first launch are counted once, the state per count, and of the checks the part of the
count that holds the most. The run measures that launch and
prints what it took. Measured by this sample on a compute
capability 12.0 GPU of 32 GB: 3,124 MiB at the first launch of the 1-sequence form, nothing at the 4-sequence form, and
6.9 GiB of growth on the device over the whole run.
On the reduced models the memory sentence is an estimate: its working memory is a flat 2 GiB, and the runs of this
paragraph missed the sentence of their day by a few tenths of a GiB in either direction, with the copies of the
in-projections and of the gate / up weights, with both of their forms and with their codes alone. Over these runs on
compute capability 10.0 (one to four layers, 1 to 33 sequences) PyTorch reserved between 0.5 GiB less and 0.1 GiB
more than that sentence said, and the device grew by about 0.2 GiB more than PyTorch reserved. That sentence did not
count what the checks hold for the sequences beyond the first: the sentence of today asks 0.2 GiB more for a default
run than the figures below, and more with every further sequence. The default run on the cuDNN Frontend rows (`fe-entries` today
without the attention preparation's row; `native` selected the same rows until 2026-11-17), with the in-projections and
the
gate / up weights on their codes alone, asked for about 3.5 GiB, reserved 3.6 GiB and grew the device by 3.8 GiB; the
four-layer run asked for about 4.4 GiB, reserved 4.4 GiB and grew it by 4.6 GiB; the one-layer run asked for about
3.1 GiB, reserved 2.6 GiB and grew it by 2.8 GiB; `--sequences 32,33` on those rows, which keeps both forms of the
in-projections, asked for about 4.0 GiB, reserved 4.1 GiB and grew it by 4.2 GiB. The default run of `native` with
its library rows, which keeps every multiplied weight on its codes alone, asked for about 3.2 GiB, reserved 2.7 GiB
and grew the device by 2.9 GiB; its four-layer run asked for about 3.8 GiB, reserved 3.2 GiB and grew it by 3.4 GiB.
The room a run takes beyond its
weights does not follow the copies it leaves out: with every multiplied weight of the two-layer model on its stored
format alone (a driver's own blocks, 1.5 GiB of copies not made) a run asked for about 3.2 GiB, reserved 2.7 GiB and
grew the device by 2.9 GiB. So a GPU whose free memory is within half a GiB of the sentence of a run of one sequence
can still fail to allocate after the plan with these models; with the published checkpoint the working memory is an
eighth of the weights held and the runs stayed inside the sentence (the section on checkpoints).
Beyond the first sequence the sentence counts what the checks of a run hold, by arithmetic on the model's widths and
with no term taken from a run (`check_row_bytes` in `model_samples/qwen_dense/model.py` names every term). At their
largest moment -- the judge of the fused linear-attention step inside the checked replay -- the checks hold four
records of the run's steps (the ids and the logits of every recorded step), the state of the reference session, three
saves of the judged layer's pool and what the judge forms over it: the distance of the recurrent states, and four
fp64 tensors of the pool for a ratio that is printed and not gated. For the two-layer model at the default request
that is 47 MiB a sequence, where the state is 3 MiB a sequence. The sentence adds it for every sequence beyond the
first -- of a run of several counts for the count that holds the most, and of every other count its record --,
names it as the checks' part, and adds four tenths of it more for PyTorch's pool, which keeps blocks beyond the
tensors that are alive. A step of several tokens per sequence is counted by its rows. Nothing is added at one
sequence, where the working memory holds the checks of the runs it was taken from, and nothing on the stack
`reference`, which runs no checks: no sentence of a run of one sequence moved.
The share for PyTorch's pool is from runs (2026-11-21; compute capability 10.0, 148 SMs; a shared GPU: correctness
only; `fe-entries`, the default request, the tree before the count). A driver capped the pool of the process in front
of the sample's `main()` (`torch.cuda.set_per_process_memory_fraction`). Capped at the counted figure alone, a run of
256 sequences (16.3 GiB) and a run of 1,024 sequences (54.1 GiB) each ended in PyTorch's out-of-memory error inside
the judge of the fused step, with 3.4 and 12.5 GiB of the pool reserved and not allocated: free blocks into which the
next tensor, of 1.5 and of 6.0 GiB, did not fit. With a tenth of the checks' part more, 256 sequences failed the same
way; with a quarter more they passed (19.2 GiB allowed, 18.4 GiB reserved) and 1,024 sequences failed (65.8 GiB
allowed, 18.5 GiB of them reserved and not allocated); with four tenths more both passed -- 256 sequences with
20.9 GiB allowed and 19.9 GiB reserved, 1,024 sequences with 72.9 GiB allowed, 70.9 GiB reserved and a peak of
52.4 GiB allocated. Four tenths is the smallest of the three shares with which both counts passed. It is an allowance
from these runs and not a bound, and it is no part of the figure the sentence names for the checks.
The sentence with the count, against runs (the same day, the same GPU, the same request; `fe-entries` where no stack is
named; every run passed its checks):

| sequences | the sentence asked for | PyTorch held at its peak | PyTorch reserved | the device grew by |
|---|---|---|---|---|
| 33 | 5.8 GiB | 2.94 GiB | 3.83 GiB | 4.01 GiB |
| 64 | 7.9 GiB | 4.49 GiB | 6.04 GiB | 6.22 GiB |
| 85 | 9.4 GiB | 5.99 GiB | 7.77 GiB | 7.94 GiB |
| 85, `fe-cublas` | 9.4 GiB | 5.97 GiB | 7.75 GiB | 7.93 GiB |
| 85, `native` | 9.2 GiB | 5.70 GiB | 7.47 GiB | 7.65 GiB |
| 256, the pool capped at the sentence's figure | 20.9 GiB | 14.44 GiB | 19.86 GiB | 20.04 GiB |
| 1,024 | 72.9 GiB | 52.43 GiB | 73.08 GiB | 73.26 GiB |
| 1,024, the pool capped at the sentence's figure | 72.9 GiB | 52.43 GiB | 70.90 GiB | 71.07 GiB |
| 16 sequences of five tokens, a verify session alone | 8.7 GiB | 5.45 GiB | 6.71 GiB | 6.89 GiB |

In the eight runs of one sequence count PyTorch's peak was 0.6 to 3.2 % above the weights held, the state and the
count; in the verify session it was 2.5 % below them. That comparison holds where the checks' moment is the peak of
the run -- at every measured count above one sequence (33 to 1,024) and in the verify session; the GPU test of 85
sequences holds it within 5 % --, and at one sequence and in the default run the peak is another moment (a load, a
multiply of the reference), as it was before the count and covered by the flat working memory: reported here, not
held to the count. From 33 to 85 sequences the sentence stands 1.5 to 1.8 GiB above what a run takes, and 0.9 GiB at
256 sequences -- the flat working memory and the pool's share are both added to the count, and the direction is the
one that only refuses a run. At 1,024 sequences a run that is given the memory takes 0.4 GiB more than the
sentence, and the same run with its pool held to the sentence's figure passes. A run of one sequence asked for about
3.5 GiB, as it did before the count, held 1.80 GiB at its peak, reserved 3.58 GiB and grew the device by 3.76 GiB:
a quarter of a GiB more than its sentence, the miss this section states for the reduced models. A run of the
published checkpoint at one sequence asked for about 37.1 GiB on `fe-entries`, the figure it asked for before the
count.
`--sequences 64,65` asked for about 8.6 GiB, reserved 8.04 GiB and grew the device by 8.22 GiB; PyTorch held 5.85 GiB
at its peak, 0.4 GiB more than the weights, the two states and the count. That run showed a defect, which is removed:
the reference session of the first count -- its state and its record -- was still alive when its checks had returned,
in a reference cycle of the session's own (its stage clock held a closure over the session), until Python collected
it. The stage clock is handed the device now, so the reference session of a count stands in no reference cycle and is
released when its checks return; a host test holds that on two counts with Python's collection of reference cycles
switched off. The same run then held 5.53 GiB at its peak, 1.6 % above the weights, the two states and the count,
reserved the same 8.04 GiB and grew the device by the same 8.22 GiB, and but for its clocks and that peak it printed
the lines it had printed.
What the sentence leaves out, by the code (no run of these kinds was made). The token check holds four fp32 tensors of
every checked logit at a time; it is the larger moment from about 150 checked steps of the two-layer model -- a
request of more positions than that at many sequences -- and from 10 checked steps at the published widths, and such
a run can fail to allocate inside its checks behind a sentence that let it start. With a checkpoint the checks replay
a second reference stack: the count holds its moment, by the code, and not the third reference state that is alive
while the second one is reset. The comparison of a verify session with its decode session is not counted (the
section "The verify step"), nor are the records of the prompt of a session that verifies behind its prompt. And a
context beyond about 2,000 positions at many sequences rests on the attention chain's term alone.
Until 2026-11-21 the sentence counted the state alone and was short from about 33 rows of a step -- 33 sequences of a
decode step, sequences x tokens per sequence of a verify step --, on every stack. At 64 sequences the stack `fe-cublas`
asked for about 3.9 GiB, reserved
6.6 GiB and grew the device by 6.8 GiB; the cuDNN Frontend rows (`fe-entries` today without the attention preparation's
row) asked for about 3.8 GiB,
reserved 6.0 GiB and grew it by 6.2 GiB, and with `--sequences 64,65` for about 4.4 GiB, reserved 8.0 GiB and grew it
by 8.2 GiB; `--stack native --sequences 64,65` with its library rows asked for about 4.2 GiB, reserved 7.1 GiB and
grew it by 7.3 GiB. A
verify plan of 48 to 80 rows was short by 0.3 to 2.6 GiB (the section "The verify step" has the runs).
The four-layer run (`--layers 4`) holds 3.1 GiB of copies. The whole default run takes well under a minute on a data-center
Blackwell GPU: a second or two to plan, which includes building the entries' kernels for both sequence counts (1.1 to
1.7 s in the runs since that build became part of the plan, 3.6 s in the first of them; built after the load, the fused
step's kernel took 0.5 to 2.4 s for the first sequence count and 0.3 to 0.4 s for the next), about 5 s to draw the
weights, the rest for the decode and its checks. These are orientation values of single runs, not measurements of record.
