# Attention Inference Benchmark

Benchmarks attention for **inference**, split into two phases (mirroring
forward/backward in `../attention_training`):

- **context** — prefill, in two kinds, both reported in **TFLOPS**:
  - *full*: `s_q == s_kv`, contiguous Q/K/V, compute-bound;
  - *chunked*: a small incoming chunk (512/1024 tokens) attends to a long
    cache (64k/128k) with bottom-right causal alignment.
- **generation** — decode: `q_tokens = 1 + MTP` new tokens (MTP 0–3) attend to
  a long cached KV. Bandwidth-bound; reported in **ms** and **GB/s** with
  **% of memory SOL** annotations (algorithmic-minimum bytes ÷ peak DRAM
  bandwidth).

## Backends

The configs sweep the two cuDNN frontend paths only:

| Backend | What it measures |
|---|---|
| `cudnn` | cuDNN frontend graph API, native backend engines (heur A + FALLBACK), contiguous KV |
| `cudnn_oss` | the same graph planned with `heur_mode.OPENSOURCE` and `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`: only the frontend's open-source engines (frost python engines + the backend's OSS candidates) may serve it; the winning plan is recorded per case |

Unsupported combinations are recorded (not hidden) and show up as blank slots
in the charts. Additional reference backends (`flashinfer`, `flash_mla`,
`b12x`, `flash_attention_4`) remain implemented in
`benchmark_single_attention.py` for ad-hoc runs via `--backend`.

## Model configs

| Config | Attention | Notes |
|---|---|---|
| `llama` | GQA d=128: 64/8 | bf16 + fp8 KV |
| `qwen35` | GQA d=256: 32/2 | wide-head GQA; bf16 + fp8 KV |
| `gpt_oss` | GQA d=64 + per-head sinks: SWA-128 and full-attention layer types | bf16 + fp8 KV, page 16 |
| `deepseek_v4` | shared-K=V MQA, 64 or 128 q-heads, d=512 | official HF configs; dense core only (CSA/HCA pools out of scope) |
| `kimi_k3` | absorbed MLA, 96 heads, 576/512, scale 1/√192, page 32 | context swept but structurally unabsorbed at prefill → blank slots (see training suite) |
| `auto_regressive_dit` | dense 9/9, d=128, bidirectional | chunked AR steps live in the context phase's chunked kind |

Model dimensions come from the official HuggingFace configs. Every LLM model
is swept across tensor-parallel shards (**TP 1/2/4/8**, framework rules: q
heads divide evenly, kv heads divide or replicate) — the `-tpN` presets are
the per-GPU shapes deployments actually execute: few local heads, large
batches. The 9-head video DiT is not head-shardable and runs whole-model
(deployments use sequence/context parallelism there). Generation
additionally sweeps the KV-cache dtype (`kv_cache_dtypes`): this corresponds
to the fp8-KV-cache serving configuration, realized on the cudnn paths as
the full fp8 attention graph (q/k/v/o e4m3 with unit descales). Sinks
(`has_sink`) add per-head attention-sink logits (gpt-oss-style; not wired
into the fp8 graph — those cases record as unsupported).

MLA models run **absorbed** in generation (`kind="mla_absorbed"`: K reads the
full record, V a leading slice of the *same* record, so KV bytes are counted
once) and **unabsorbed** in prefill — which is dense training-style attention
and lives in `../attention_training`.

## Usage

```bash
python -m benchmark.attention_inference.runner --list-configs
python -m benchmark.attention_inference.runner --config llama
python -m benchmark.attention_inference.runner --config kimi_k3 --dry-run
python -m benchmark.attention_inference.runner --config llama --backend cudnn_oss
python -m benchmark.attention_inference.runner --config llama --phase generation
```

Results are organized per architecture as `results/<config>/<gpu>/`, each
holding the CSV plus two charts: `<config>_context.png` (subplots stacked by
prefill kind) and `<config>_generation.png` (subplots stacked by MTP width).
Every expanded case owns an x slot whether or not it ran, so coverage gaps
are visible. Architectures are reported in isolation, never merged.

## Measurement notes

- Timing is CUDA events around the steady-state call, median of
  `num_iterations`, after warmup. All host-side planning (graph build,
  workspace, page tables) happens **outside** the timed region: this measures
  per-step device cost.
- Peak-bandwidth table for % SOL lives in `benchmark_single_attention.py`
  (`PEAK_BW_GBPS`); override with `CUDNN_BENCH_PEAK_BW_GBPS=<GB/s>`.
- fp8 KV on the cudnn paths runs the fp8 attention graph (q/k/v e4m3, unit
  descales).
- cuDNN `sink_token` does not support s_q==1; sink decode rows on cudnn
  record that limitation.
