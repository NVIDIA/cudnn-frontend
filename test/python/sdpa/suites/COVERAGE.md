# SDPA suites — coverage master list

Generated from `registry.py` by `gen_coverage.py` — do not edit by hand.

Conventions:

- One suite = one deterministic seed sweep (`num_tests` configs from `rng_seed`);
  every config prints a `test_repro_suite.py::test_repro --repro ...` command.
- fp16/bf16 sibling suites share seed + knob set: identical geometry, both dtypes.
- Sliding-window/causal masks and bias are fuzz axes inside suites, never separate suites.
- THD suites fuzz first-class packed capacities: `total_q`/`total_kv` slack and
  declaring them on the graph (`sdpa(max_total_seq_len_q/kv=...)`).
- Model suites pin head/dim geometry of popular models (full/global attention only)
  and fuzz everything else through the same runner.

## Context (prefill forward)

| suite | dtype | level | N | fuzzed | pinned | gates / notes |
|---|---|---|---|---|---|---|
| context.fp16.dense | fp16 | L0 | 128 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, layout padded/full, sink, bias(1:5) | infer, impl AUTO |  |
| context.bf16.dense | bf16 | L0 | 128 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, layout padded/full, sink, bias(1:5) | infer, impl AUTO |  |
| context.fp16.unified | fp16 | L1 | 64 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, layout padded/cu_padded/full, sink, bias(1:3), unfuse_fma | infer, impl UNIFIED |  |
| context.bf16.unified | bf16 | L1 | 64 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, layout padded/cu_padded/full, sink, bias(1:3), unfuse_fma | infer, impl UNIFIED |  |
| context.fp16.thd | fp16 | L0 | 256 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, stats token/head-major, total_q/kv slack, declare totals on graph, ragged token gaps, sink | infer, layout THD (ragged), impl AUTO |  |
| context.bf16.thd | bf16 | L0 | 256 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, stats token/head-major, total_q/kv slack, declare totals on graph, ragged token gaps, sink | infer, layout THD (ragged), impl AUTO |  |
| context.fp16.thd_unified | fp16 | L1 | 128 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, stats token/head-major, total_q/kv slack, declare totals on graph, ragged token gaps, layout ragged/cu_ragged, sink | infer, no mask, diag TL, impl UNIFIED |  |
| context.fp16.thd_offset_mult | fp16 | L1 | 128 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, stats token/head-major, total_q/kv slack, declare totals on graph, ragged token gaps, layout ragged_mult/cu_ragged_mult, sink | infer, no mask, diag TL, impl UNIFIED | cuDNN>=92400; ragged offset multiplier: unified fwd engine only, cuDNN >= 9.24 |
| context.fp8.dense | fp8 | L0 | 384 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, e4m3/e5m2 in, out fp8/fp16, layout padded/full, sink, unfuse_fma, rescale 0/2/4 (SM100) | infer, impl AUTO |  |
| context.fp8.thd | fp8 | L0 | 384 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, e4m3/e5m2 in, out fp8/fp16, layout ragged/cu_ragged/cu_ragged_mult, total_q/kv slack, declare totals on graph, rescale 0/2/4 (SM100) | infer, no mask, diag TL |  |
| context.mxfp8.dense | mxfp8 | L0 | 384 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, e4m3/e5m2 in, out fp16/bf16, sink, unfuse_fma, rescale 0/2/4 (SM100) | infer, impl AUTO, SM100+, layout full (mxfp8 API has no seq-len args, #646) | SM>=100 |
| models.llama31.context.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=8, d_qk=128, d_v=128, sink=off | llama31 full/global attention layers |
| models.llama31.context.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=8, d_qk=128, d_v=128, sink=off | llama31 full/global attention layers |
| models.qwen35.context.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=32, h_kv=2, d_qk=256, d_v=256, sink=off | qwen35 full/global attention layers |
| models.qwen35.context.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=32, h_kv=2, d_qk=256, d_v=256, sink=off | qwen35 full/global attention layers |
| models.gpt_oss.context.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=8, d_qk=64, d_v=64, sink=fuzzed | gpt_oss full/global attention layers |
| models.gpt_oss.context.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=8, d_qk=64, d_v=64, sink=fuzzed | gpt_oss full/global attention layers |
| models.dsv3.context.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=128, h_kv=128, d_qk=192, d_v=128, sink=off | dsv3 full/global attention layers |
| models.dsv3.context.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=128, h_kv=128, d_qk=192, d_v=128, sink=off | dsv3 full/global attention layers |
| models.kimi_k26.context.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=64, d_qk=192, d_v=128, sink=off | kimi_k26 full/global attention layers |
| models.kimi_k26.context.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=64, d_qk=192, d_v=128, sink=off | kimi_k26 full/global attention layers |
| models.glm46.context.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=96, h_kv=8, d_qk=128, d_v=128, sink=off | glm46 full/global attention layers |
| models.glm46.context.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=96, h_kv=8, d_qk=128, d_v=128, sink=off | glm46 full/global attention layers |

## Generation (decode / small-s_q forward)

| suite | dtype | level | N | fuzzed | pinned | gates / notes |
|---|---|---|---|---|---|---|
| generation.fp16.decode | fp16 | L0 | 64 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, diag TL/BR | infer, s_q=1, no mask, layout full, impl AUTO |  |
| generation.bf16.decode | bf16 | L0 | 64 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, diag TL/BR | infer, s_q=1, no mask, layout full, impl AUTO |  |
| generation.fp16.decode_unified | fp16 | L1 | 32 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data | infer, s_q=1, no mask, diag TL, layout full, impl UNIFIED |  |
| generation.fp16.lean | fp16 | L0 | 64 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, diag TL/BR, layout padded/full | infer, s_q=1, s_kv 513..4096, no mask, impl AUTO |  |
| generation.bf16.lean | bf16 | L0 | 64 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, diag TL/BR, layout padded/full | infer, s_q=1, s_kv 513..4096, no mask, impl AUTO |  |
| generation.fp16.paged | fp16 | L0 | 128 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, block size 1..1024, sink | infer, s_q<=64, layout padded, paged KV, impl AUTO |  |
| generation.bf16.paged | bf16 | L0 | 128 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, block size 1..1024, sink | infer, s_q<=64, layout padded, paged KV, impl AUTO |  |
| generation.fp16.paged_unified | fp16 | L0 | 64 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, layout padded/cu_padded, block size 1..1024 | infer, s_q<=64, no mask, diag TL, paged KV, impl UNIFIED |  |
| generation.fp16.thd_chunked | fp16 | L0 | 96 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, stats token/head-major, total_q/kv slack, declare totals on graph, ragged token gaps | infer, s_q<=64, layout THD (ragged), impl AUTO | varlen chunked generation: packed THD chunks against long KV |
| generation.bf16.thd_chunked | bf16 | L0 | 96 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, stats token/head-major, total_q/kv slack, declare totals on graph, ragged token gaps | infer, s_q<=64, layout THD (ragged), impl AUTO | varlen chunked generation: packed THD chunks against long KV |
| generation.fp8.decode | fp8 | L0 | 128 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, e4m3/e5m2 in, out fp8/fp16, diag TL/BR, unfuse_fma, rescale 0/2/4 (SM100) | infer, s_q=1, no mask, layout full |  |
| generation.fp8.paged | fp8 | L0 | 96 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, e4m3/e5m2 in, out fp8/fp16, block size 16..128, rescale 0/2/4 (SM100) | infer, no mask, diag TL, layout padded, paged KV |  |
| models.llama31.generation.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=64, h_kv=8, d_qk=128, d_v=128, sink=off | llama31 full/global attention layers |
| models.llama31.generation.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=64, h_kv=8, d_qk=128, d_v=128, sink=off | llama31 full/global attention layers |
| models.qwen35.generation.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=32, h_kv=2, d_qk=256, d_v=256, sink=off | qwen35 full/global attention layers |
| models.qwen35.generation.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=32, h_kv=2, d_qk=256, d_v=256, sink=off | qwen35 full/global attention layers |
| models.gpt_oss.generation.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=64, h_kv=8, d_qk=64, d_v=64, sink=off | gpt_oss full/global attention layers |
| models.gpt_oss.generation.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=64, h_kv=8, d_qk=64, d_v=64, sink=off | gpt_oss full/global attention layers |
| models.dsv3.generation.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=128, h_kv=128, d_qk=192, d_v=128, sink=off | dsv3 full/global attention layers |
| models.dsv3.generation.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=128, h_kv=128, d_qk=192, d_v=128, sink=off | dsv3 full/global attention layers |
| models.kimi_k26.generation.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=64, h_kv=64, d_qk=192, d_v=128, sink=off | kimi_k26 full/global attention layers |
| models.kimi_k26.generation.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=64, h_kv=64, d_qk=192, d_v=128, sink=off | kimi_k26 full/global attention layers |
| models.glm46.generation.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=96, h_kv=8, d_qk=128, d_v=128, sink=off | glm46 full/global attention layers |
| models.glm46.generation.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=96, h_kv=8, d_qk=128, d_v=128, sink=off | glm46 full/global attention layers |

## Bprop (training fwd+bwd)

| suite | dtype | level | N | fuzzed | pinned | gates / notes |
|---|---|---|---|---|---|---|
| bprop.fp16.dense | fp16 | L0 | 192 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, layout padded/full, deterministic, sink, bias(1:7) | train, impl AUTO |  |
| bprop.bf16.dense | bf16 | L0 | 192 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, layout padded/full, deterministic, sink, bias(1:7) | train, impl AUTO |  |
| bprop.fp16.thd | fp16 | L0 | 256 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, stats token/head-major, total_q/kv slack, declare totals on graph, ragged token gaps, deterministic, sink | train, layout THD (ragged), impl AUTO |  |
| bprop.bf16.thd | bf16 | L0 | 256 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, stats token/head-major, total_q/kv slack, declare totals on graph, ragged token gaps, deterministic, sink | train, layout THD (ragged), impl AUTO |  |
| bprop.fp8.dense | fp8 | L0 | 256 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, out fp8/fp16, deterministic, sink | train, e4m3 in, layout full, rescale off |  |
| bprop.fp8.thd | fp8 | L0 | 256 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, out fp8/fp16, deterministic, sink, total_q/kv slack | train, e4m3 in, no mask, diag TL, layout THD (ragged), rescale off | cuDNN>=92101; ragged FP8 backward requires cuDNN > 9.21.0 |
| bprop.mxfp8.dense | mxfp8 | L0 | 256 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, out fp16/bf16, sink | train, e4m3 in, deterministic, layout full, rescale off, SM100+ | SM>=100 |
| models.llama31.bprop.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=8, d_qk=128, d_v=128, sink=off | llama31 full/global attention layers |
| models.llama31.bprop.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=8, d_qk=128, d_v=128, sink=off | llama31 full/global attention layers |
| models.qwen35.bprop.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=32, h_kv=2, d_qk=256, d_v=256, sink=off | qwen35 full/global attention layers |
| models.qwen35.bprop.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=32, h_kv=2, d_qk=256, d_v=256, sink=off | qwen35 full/global attention layers |
| models.gpt_oss.bprop.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=8, d_qk=64, d_v=64, sink=fuzzed | gpt_oss full/global attention layers |
| models.gpt_oss.bprop.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=8, d_qk=64, d_v=64, sink=fuzzed | gpt_oss full/global attention layers |
| models.dsv3.bprop.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=128, h_kv=128, d_qk=192, d_v=128, sink=off | dsv3 full/global attention layers |
| models.dsv3.bprop.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=128, h_kv=128, d_qk=192, d_v=128, sink=off | dsv3 full/global attention layers |
| models.kimi_k26.bprop.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=64, d_qk=192, d_v=128, sink=off | kimi_k26 full/global attention layers |
| models.kimi_k26.bprop.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=64, d_qk=192, d_v=128, sink=off | kimi_k26 full/global attention layers |
| models.glm46.bprop.fp16 | fp16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=96, h_kv=8, d_qk=128, d_v=128, sink=off | glm46 full/global attention layers |
| models.glm46.bprop.bf16 | bf16 | L0 | 4 | batch, seq lens, layout, mask flavor, data | h_q=96, h_kv=8, d_qk=128, d_v=128, sink=off | glm46 full/global attention layers |

**Total configs: 5136 across 66 suites.**
