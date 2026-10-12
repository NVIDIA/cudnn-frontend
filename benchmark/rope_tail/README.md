# BF16 tail RoPE stage benchmark

This benchmark calls `cudnn.TailRoPEForward(..., backend="frost")` with caller-owned
outputs. It measures a full BF16 tensor copy plus rotation of the last 64 channels,
using prepared FP32 tables. It does not measure table preparation, projection,
attention, backward, or model throughput.

Install this checkout and FlashInfer in an environment with PyTorch, CuTe DSL
4.7.0 or newer, a CUDA compiler and an NVIDIA B200. Confirm that
`cudnn.__file__` points at this checkout, especially with editable installations.
Supply `model.py` and `config.json` from the `inference` directory of
[DeepSeek-V4.1-Flash revision dba1be0](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277/inference).
The loader checks the complete file hashes before using the two reference
functions. No model weights are required.

```bash
python benchmark/rope_tail/bench.py \
    --deepseek-source /path/to/DeepSeek-V4.1-Flash/inference \
    --output /path/to/results/tail-rope.json
```

The output path must be new. The benchmark retains raw samples, profiler traces,
provider failures and numerical witnesses beside that file. Run on an idle GPU,
retain hardware and environment information, and repeat in an independent process
before drawing performance conclusions. `performance_admitted` stays false: a
successful script run is input to review, not an automatic speedup claim.

## Workloads and checks

Sixty-four cases cover `T=1,4,4096,16384`, both source base and compressed frequency
tables, and these head geometries per tensor-parallel rank:

| Role | TP | Heads per rank | Channels | Rotation |
| --- | ---: | ---: | ---: | --- |
| Query | 1 | 64 | 512 | Forward angle |
| Output | 1 | 64 | 512 | Opposite angle |
| Index query | 1 | 32 | 128 | Forward angle |
| Query | 8 | 8 | 512 | Forward angle |
| Output | 8 | 8 | 512 | Opposite angle |
| Index query | 8 | 4 | 128 | Forward angle |
| KV | 1 / 8 | 1 | 512 | Forward angle |
| Index key | 1 / 8 | 1 | 128 | Forward angle |

The single KV and index-key heads are measured once per token count and frequency
table. TP8 supplies the local query head geometry only; this benchmark does not
launch multiple ranks or measure collectives.

Activations are seeded synthetic BF16. The small-token cases expose decode
regressions; opposite-angle forward execution is not a backward implementation.
The head geometry and frequency parameters come from the pinned source config.
These are prepared operator workloads, not a reconstruction of a complete layer.

Frost and the source reference must match bit for bit on seven input generations,
including changed inputs, signed zeros, subnormals and large values. Inputs and
prepared tables must remain unchanged. Every replay starts with poisoned outputs.
CUDA profiler traces must show three captured Frost kernel launches for the three
independent input/output buffer pairs.

## Controls and timing

Controls include the unchanged DeepSeek complex RoPE reference, default FlashInfer,
FlashInfer built from the same sources with `--ftz=false`, and regular/tuned
`torch.compile` versions of an authored FP32 multiply/FMA expression. FlashInfer
kernels retain their upstream attribution. The no-FTZ wrapper is shared with the
sibling RoPE QDQ benchmark; it changes compiler flags and does not replace the
provider's kernel implementation.

Each control is checked on all seven generations. A control may be timed only
when it passes both timed distributions (random and changed input). Its failures
on other generations remain visible in `checks`, `full_contract_passed`, and the
saved witnesses; it must not be described as satisfying the full API contract.
Frost is required to pass every generation. Keeping a faster control that passes
the timed distributions makes the speed comparison conservative.

Timing uses eight paired blocks, alternating provider order, five retained
samples per block, and one discarded warmup. GPU event latency and Graph replay
wall latency are reported separately, amortized over three invocations. Input
restoration, output poisoning and optional L2 eviction are outside both clocks.
The hot-cache and evicted-cache regimes remain separate. The fastest measured
valid control is selected per case and metric; paired block medians and win
counts remain in the JSON. A case without a valid FlashInfer or compiled-Torch
control has insufficient baseline coverage for a speedup claim.

Graph replay wall latency includes replay submission and completion waiting. It
does not measure Python `plan.execute()` overhead on every invocation or the
convenience wrapper's allocation cost. Previous kernel measurements are not a
substitute for running this public-API benchmark on the exact checkout.
