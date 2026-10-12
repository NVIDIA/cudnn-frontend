# Engram saved-state gate benchmark

Run from an environment containing the candidate cuDNN Frontend package,
CUDA-enabled PyTorch, CuTe DSL >=4.7.0 and Triton >=3.7.0 on an SM100 GPU:

```bash
python benchmark/engram/bench_saved_gate.py --tokens 4096 --output /tmp/engram.json
```

The benchmark includes the dense projection, four-stream gate, both dense GEMM
gradients, and the product rule for the two normalization weights. It reports
inference forward, training forward, and complete forward/backward separately.
All paths include projection and gradient work in their measured interval.
Inputs use BF16, normalization parameters and their gradients use FP32.

The default is S4096, microbatch 1, H5120, four streams and embedding width 6144:
a floating training surrogate using the DeepSeek-V4.1-Flash geometry and a
Megatron-style sequence/microbatch configuration. Inputs and weights are
synthetic. This does not measure the FP8 embedding table, quantized projection,
table-gradient accumulation, collectives, optimizer, or an entire training step.

Eight native controls include eager PyTorch, ordinary and autotuned Inductor,
compiled controls preserving eager FP32 reduction order, and a separate FP64-dot
control. Every control is checked for every phase on seven input generations,
including masked, zero and low-norm cases, in eager execution and changed-input
CUDA Graph replay. A failing native phase remains in the JSON with its failing
metrics and a tensor artifact; it receives no timing. A Frost or eager-reference
failure rejects the run. Thresholds are relative L2 / maximum error normalized
by reference magnitude: output 0.01/0.02, projection 0.006/0.012, gradients 0.03/0.06.

The same validated graphs are measured with eight alternating provider-order
blocks, five samples per block and three complete calls per sample. Compilation,
autotuning, input mutation, correctness checks and profiler work are outside
timing. Full eager wall/GPU time and CUDA Graph GPU time are distinct metrics.
Output/intermediate poisoning, stale/zero/NaN negative controls, kernel traces
and post-timing checks guard against empty or stale replay. All raw samples,
native failures, source hashes and per-phase comparisons are retained.

Performance comparisons select the fastest valid native control independently
for each phase and metric; slower phases must also be reported. These are
component-subgraph measurements, not model throughput. Independently repeat a
useful result before claiming a stable improvement.

Gate semantics follow [DeepSeek-V4.1-Flash inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py).
The benchmark reference and reduction controls use PyTorch. Projection GEMMs
use existing providers and are not new kernels contributed by this change.
