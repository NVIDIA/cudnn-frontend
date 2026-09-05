# SDPA suites — coverage master list

Generated from `registry.py` by `gen_coverage.py` — do not edit by hand.

Conventions:

- One suite = one deterministic seed sweep (`num_tests` configs from `rng_seed`);
  every config prints a `test_repro_suite.py::test_repro --repro ...` command.
- 16-bit is one family: f16 suites draw fp16 or bf16 per config, like fp8 draws e4m3/e5m2.
- Sliding-window/causal masks and bias are fuzz axes inside suites, never separate suites.
- THD suites fuzz first-class packed capacities: `total_q`/`total_kv` slack and
  declaring them on the graph (`sdpa(max_total_seq_len_q/kv=...)`).
- Model suites pin head/dim geometry of popular models (full/global attention only)
  and fuzz everything else through the same runner.

## Context (prefill forward)

| suite | dtype | level | N | fuzzed | pinned | gates / notes |
|---|---|---|---|---|---|---|
| context.f16.dense | f16 | L0 | 512 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, layout padded/cu_padded/full, sink, bias(1:5) | infer |  |
| context.f16.thd | f16 | L0 | 768 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, stats token/head-major, total_q/kv slack, declare totals on graph, ragged token gaps, sink | infer, layout THD (ragged/cu_ragged) |  |
| context.fp8.dense | fp8 | L0 | 512 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, e4m3/e5m2 in, out fp8/fp16, layout padded/full, sink | infer |  |
| context.fp8.thd | fp8 | L0 | 512 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, e4m3/e5m2 in, out fp8/fp16, layout ragged/cu_ragged, sink, total_q/kv slack, declare totals on graph | infer | diag BR-weighted 2:1 — production context-phase alignment |
| context.mxfp8.dense | mxfp8 | L0 | 512 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, e4m3/e5m2 in, out fp16/bf16, sink | infer, SM100+, layout full (mxfp8 API has no seq-len args, #646) | SM>=100 |
| context.mxfp8.thd | mxfp8 | L0 | 192 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, e4m3/e5m2 in, out fp16/bf16, layout ragged/cu_ragged, sink, total_q/kv slack, declare totals on graph | infer, stats token-major TH1, d=128/128 (frost THD leg), SM100+ | SM>=100; diag BR-weighted 2:1 (production context alignment); fwd only (no THD mxfp8 bwd engine); needs opt-in FROST engine (CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1) — skips otherwise: the native backend check_support-accepts THD mxfp8 but cannot execute it |
| models.llama31.context.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16 | h_q=64, h_kv=8, d_qk=128, d_v=128 | llama31 full/global attention layers, fp8-trained flavor |
| models.llama31.context | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=8, d_qk=128, d_v=128, sink=off | llama31 full/global attention layers |
| models.qwen35.context.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16 | h_q=32, h_kv=2, d_qk=256, d_v=256 | qwen35 full/global attention layers, fp8-trained flavor |
| models.qwen35.context | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data | h_q=32, h_kv=2, d_qk=256, d_v=256, sink=off | qwen35 full/global attention layers |
| models.gpt_oss.context.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16 | h_q=64, h_kv=8, d_qk=64, d_v=64 | gpt_oss full/global attention layers, fp8-trained flavor |
| models.gpt_oss.context | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=8, d_qk=64, d_v=64, sink=fuzzed | gpt_oss full/global attention layers |
| models.dsv3.context.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16 | h_q=128, h_kv=128, d_qk=192, d_v=128 | dsv3 full/global attention layers, fp8-trained flavor |
| models.dsv3.context | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data | h_q=128, h_kv=128, d_qk=192, d_v=128, sink=off | dsv3 full/global attention layers |
| models.kimi_k3.context.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16 | h_q=96, h_kv=96, d_qk=192, d_v=128 | kimi_k3 full/global attention layers, fp8-trained flavor |
| models.kimi_k3.context | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data | h_q=96, h_kv=96, d_qk=192, d_v=128, sink=off | kimi_k3 full/global attention layers |
| models.llama31.context.mxfp8 | mxfp8 | L0 | 8 | batch, seq lens, mask flavor, data, e4m3/e5m2 in, out fp16/bf16 | h_q=64, h_kv=8, d_qk=128, d_v=128, layout full, SM100+ | SM>=100; llama31 mxfp8 flavor; no generation (no decode-shaped mxfp8 engine) |
| models.qwen35.context.mxfp8 | mxfp8 | L0 | 8 | batch, seq lens, mask flavor, data, e4m3/e5m2 in, out fp16/bf16 | h_q=32, h_kv=2, d_qk=256, d_v=256, layout full, SM100+ | SM>=100; qwen35 mxfp8 flavor; no generation (no decode-shaped mxfp8 engine) |
| models.gpt_oss.context.mxfp8 | mxfp8 | L0 | 8 | batch, seq lens, mask flavor, data, e4m3/e5m2 in, out fp16/bf16 | h_q=64, h_kv=8, d_qk=64, d_v=64, layout full, SM100+ | SM>=100; gpt_oss mxfp8 flavor; no generation (no decode-shaped mxfp8 engine) |
| models.dsv3.context.mxfp8 | mxfp8 | L0 | 8 | batch, seq lens, mask flavor, data, e4m3/e5m2 in, out fp16/bf16 | h_q=128, h_kv=128, d_qk=192, d_v=128, layout full, SM100+ | SM>=100; dsv3 mxfp8 flavor; no generation (no decode-shaped mxfp8 engine) |
| models.kimi_k3.context.mxfp8 | mxfp8 | L0 | 8 | batch, seq lens, mask flavor, data, e4m3/e5m2 in, out fp16/bf16 | h_q=96, h_kv=96, d_qk=192, d_v=128, layout full, SM100+ | SM>=100; kimi_k3 mxfp8 flavor; no generation (no decode-shaped mxfp8 engine) |

## Generation (decode / small-s_q forward)

| suite | dtype | level | N | fuzzed | pinned | gates / notes |
|---|---|---|---|---|---|---|
| generation.f16.decode | f16 | L0 | 256 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, diag TL/BR | infer, s_q=1, no mask, layout full |  |
| generation.f16.lean | f16 | L0 | 256 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, diag TL/BR, layout padded/full | infer, s_q=1, s_kv 513..8192, no mask |  |
| generation.f16.paged | f16 | L0 | 384 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, layout padded/cu_padded/ragged(THD Q + paged KV), block size 1..1024, sink | infer, s_q<=64, layout padded, paged KV |  |
| generation.f16.thd_chunked | f16 | L0 | 256 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, stats token/head-major, total_q/kv slack, declare totals on graph, ragged token gaps | infer, s_q<=64, layout THD (ragged) | varlen chunked generation: packed THD chunks against long KV |
| generation.fp8.decode | fp8 | L0 | 192 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, e4m3/e5m2 in, out fp8/fp16, diag TL/BR | infer, s_q=1, no mask, layout full |  |
| generation.fp8.paged | fp8 | L0 | 128 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, e4m3/e5m2 in, out fp8/fp16, block size 16..128 | infer, no mask, diag TL, layout padded, paged KV |  |
| models.llama31.generation.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16, paged 50% | h_q=64, h_kv=8, d_qk=128, d_v=128 | llama31 full/global attention layers, fp8-trained flavor |
| models.llama31.generation | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=64, h_kv=8, d_qk=128, d_v=128, sink=off | llama31 full/global attention layers |
| models.qwen35.generation.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16, paged 50% | h_q=32, h_kv=2, d_qk=256, d_v=256 | qwen35 full/global attention layers, fp8-trained flavor |
| models.qwen35.generation | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=32, h_kv=2, d_qk=256, d_v=256, sink=off | qwen35 full/global attention layers |
| models.gpt_oss.generation.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16, paged 50% | h_q=64, h_kv=8, d_qk=64, d_v=64 | gpt_oss full/global attention layers, fp8-trained flavor |
| models.gpt_oss.generation | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=64, h_kv=8, d_qk=64, d_v=64, sink=off | gpt_oss full/global attention layers |
| models.dsv3.generation.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16, paged 50% | h_q=128, h_kv=128, d_qk=192, d_v=128 | dsv3 full/global attention layers, fp8-trained flavor |
| models.dsv3.generation | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=128, h_kv=128, d_qk=192, d_v=128, sink=off | dsv3 full/global attention layers |
| models.kimi_k3.generation.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16, paged 50% | h_q=96, h_kv=96, d_qk=192, d_v=128 | kimi_k3 full/global attention layers, fp8-trained flavor |
| models.kimi_k3.generation | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data, paged 50% | h_q=96, h_kv=96, d_qk=192, d_v=128, sink=off | kimi_k3 full/global attention layers |

## Bprop (training fwd+bwd)

| suite | dtype | level | N | fuzzed | pinned | gates / notes |
|---|---|---|---|---|---|---|
| bprop.f16.dense | f16 | L0 | 512 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, layout padded/full, deterministic, sink, bias(1:7) | t, r, a, i, n |  |
| bprop.f16.thd | f16 | L0 | 768 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, stats token/head-major, total_q/kv slack, declare totals on graph, ragged token gaps, deterministic, sink | train, layout THD (ragged) |  |
| bprop.fp8.dense | fp8 | L0 | 384 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, out fp8/fp16, deterministic, sink | train, e4m3 in, layout full |  |
| bprop.fp8.thd | fp8 | L0 | 384 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, out fp8/fp16, deterministic, sink, total_q/kv slack | train, e4m3 in, layout THD (ragged) | ragged FP8 backward requires cuDNN > 9.21.0 |
| bprop.mxfp8.dense | mxfp8 | L0 | 384 | batch, s_q/s_kv, d_qk/d_v, heads (MHA/GQA/MQA), strides+gaps, data, mask: causal/left/right/band/none, diag TL/BR, out fp16/bf16, sink | train, e4m3 in, deterministic, layout full, SM100+ | SM>=100 |
| models.llama31.bprop.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16 | h_q=64, h_kv=8, d_qk=128, d_v=128 | llama31 full/global attention layers, fp8-trained flavor |
| models.llama31.bprop | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=8, d_qk=128, d_v=128, sink=off | llama31 full/global attention layers |
| models.qwen35.bprop.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16 | h_q=32, h_kv=2, d_qk=256, d_v=256 | qwen35 full/global attention layers, fp8-trained flavor |
| models.qwen35.bprop | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data | h_q=32, h_kv=2, d_qk=256, d_v=256, sink=off | qwen35 full/global attention layers |
| models.gpt_oss.bprop.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16 | h_q=64, h_kv=8, d_qk=64, d_v=64 | gpt_oss full/global attention layers, fp8-trained flavor |
| models.gpt_oss.bprop | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data | h_q=64, h_kv=8, d_qk=64, d_v=64, sink=fuzzed | gpt_oss full/global attention layers |
| models.dsv3.bprop.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16 | h_q=128, h_kv=128, d_qk=192, d_v=128 | dsv3 full/global attention layers, fp8-trained flavor |
| models.dsv3.bprop | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data | h_q=128, h_kv=128, d_qk=192, d_v=128, sink=off | dsv3 full/global attention layers |
| models.kimi_k3.bprop.fp8 | fp8 | L0 | 8 | batch, seq lens, layout, mask flavor, data, e4m3/e5m2 in, out fp8/fp16 | h_q=96, h_kv=96, d_qk=192, d_v=128 | kimi_k3 full/global attention layers, fp8-trained flavor |
| models.kimi_k3.bprop | f16 | L0 | 8 | batch, seq lens, layout, mask flavor, data | h_q=96, h_kv=96, d_qk=192, d_v=128, sink=off | kimi_k3 full/global attention layers |
| models.llama31.bprop.mxfp8 | mxfp8 | L0 | 8 | batch, seq lens, mask flavor, data, e4m3/e5m2 in, out fp16/bf16 | h_q=64, h_kv=8, d_qk=128, d_v=128, layout full, SM100+ | SM>=100; llama31 mxfp8 flavor; no generation (no decode-shaped mxfp8 engine) |
| models.qwen35.bprop.mxfp8 | mxfp8 | L0 | 8 | batch, seq lens, mask flavor, data, e4m3/e5m2 in, out fp16/bf16 | h_q=32, h_kv=2, d_qk=256, d_v=256, layout full, SM100+ | SM>=100; qwen35 mxfp8 flavor; no generation (no decode-shaped mxfp8 engine) |
| models.gpt_oss.bprop.mxfp8 | mxfp8 | L0 | 8 | batch, seq lens, mask flavor, data, e4m3/e5m2 in, out fp16/bf16 | h_q=64, h_kv=8, d_qk=64, d_v=64, layout full, SM100+ | SM>=100; gpt_oss mxfp8 flavor; no generation (no decode-shaped mxfp8 engine) |
| models.dsv3.bprop.mxfp8 | mxfp8 | L0 | 8 | batch, seq lens, mask flavor, data, e4m3/e5m2 in, out fp16/bf16 | h_q=128, h_kv=128, d_qk=192, d_v=128, layout full, SM100+ | SM>=100; dsv3 mxfp8 flavor; no generation (no decode-shaped mxfp8 engine) |
| models.kimi_k3.bprop.mxfp8 | mxfp8 | L0 | 8 | batch, seq lens, mask flavor, data, e4m3/e5m2 in, out fp16/bf16 | h_q=96, h_kv=96, d_qk=192, d_v=128, layout full, SM100+ | SM>=100; kimi_k3 mxfp8 flavor; no generation (no decode-shaped mxfp8 engine) |

**Total configs: 7232 across 57 suites.**
