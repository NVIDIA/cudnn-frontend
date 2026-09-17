# RoPE + microscaled QDQ on SM100

These standalone scripts compare the actual cuDNN FE `backend="frost"` API
against native complete RoPE + quantize/dequantize chains. They require
PyTorch, Triton >=3.7, FlashInfer with its CUDA and CuTe DSL MX quantizers,
NumPy, a CUDA toolkit supporting SM100, and cuDNN Frontend containing
`RopeQDQInplace`.

Provide the `inference` directory from DeepSeek-V4.1-Flash revision
`dba1be0a40aa45a94ad051997016db3960a90277`. The benchmark verifies SHA256 of
`model.py`, `kernel.py`, and `config.json` before loading reference functions.
It does not download code or model weights.

```bash
python benchmark/rope_qdq/bench_fp4.py \
  --deepseek-source /path/to/DeepSeek-V4.1-Flash/inference \
  --output /path/to/results/fp4.json
python benchmark/rope_qdq/bench_fp8.py \
  --deepseek-source /path/to/DeepSeek-V4.1-Flash/inference \
  --output /path/to/results/fp8.json
```

Run each in a fresh process on an otherwise idle GPU. Output files must not
already exist. The harness creates fresh compiler caches, records imported
implementation hashes, retains generated code and saves first-failure tensor
artifacts. Allow several GiB of result storage for stress failures in native
controls. The scripts leave caches and evidence available for inspection.

## Workloads and measured operation

- FP4: indexer queries at TP1/TP8 (32/4 heads) and shared keys at compression
  ratios1/2, for sequence lengths4096/16384. D=128; last64 channels use the
  compressed-layer RoPE configuration from the pinned model.
- FP8: window KV at B=1/4, sequence lengths4096/16384, using both base and
  compressed-layer RoPE configurations. D=512. DSpark main-token and five-token
  verification cases use source positions4096 and4097..4101.

Inputs are synthetic BF16 activations. The complete measured stage is tail
RoPE, BF16 rounding, then group32 UE8M0 quantization/dequantization into BF16
consumer storage. Native packed quantizers include their GPU unpack/writeback
cost. Projection, index scores/top-k, ring-cache writes, attention, training
gradients and full-model throughput are outside this measurement.

## Baselines and fairness

Both scripts retain the following families:

- DeepSeek source wrappers and preallocated in-place quantizer calls;
- FlashInfer RoPE followed by source quantization;
- unchanged FlashInfer RoPE with a separately compiled `--ftz=false` control;
- Torch eager, compiled and tuned fused chains, including explicit-FMA variants;
- FlashInfer RoPE plus tuned Torch QDQ;
- FlashInfer CUDA and CuTe DSL MX quantizers plus GPU dequantization.

The sole candidate is the actual prepared cuDNN FE API. All of its compile and
module initialization runs before timing. Native provider selector values are
the providers' existing API vocabulary. The no-FTZ control retains FlashInfer
implementation ownership; its wrapper changes only that compiler flag.

Every provider is checked bit-for-bit against the pinned source, including
signed zero, in eager execution and CUDA Graph replay. Seven generations cover
random inputs, changed negative inputs, signed zeros, small/subnormal/large
values and quantization midpoints. The candidate, source, no-FTZ and explicit-FMA
controls must pass all seven. Failure artifacts and metrics remain in the JSON.

Timing uses the two validated ordinary input distributions. A native control
that fails a stress generation still competes if it is exact on both timed
distributions. The reported baseline is the fastest such native chain for that
shape and timing regime. This avoids claiming a win by discarding a fast native
implementation whose failure is outside the actual timed inputs.

Each graph operates on three disjoint input buffers. Buffers and intermediates
are reset before every replay outside the measured interval. Eight paired
blocks alternate provider order, with five retained samples per block after
two warmups, for both hot-cache and L2-evicted regimes. GPU event time and host
wall time are separate metrics; CUDA Graph timing events are external events.
Routes are checked through profiler traces. Outputs are rechecked after timing.

The final `passed_pending_audit` status establishes the recorded numerical
matrix. Independently verify source identity, GPU isolation, clocks/thermal
state and timing summaries before making a performance claim. An exception
returns nonzero, retains failure context and removes completed timing sections
from admitted results.

## Source attribution

Mathematical and consumer contracts come from the pinned
[DeepSeek model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py)
and its companion `kernel.py`. Those implementations are loaded from the
provided checkout and remain attributed to DeepSeek. FlashInfer baselines
remain FlashInfer implementations. The FE kernels, Torch controls, GPU unpack
glue, benchmark harness and BF16-boundary fixtures are authored for this work.
