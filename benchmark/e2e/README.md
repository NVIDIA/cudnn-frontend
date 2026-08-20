# End-to-end model perf-share benchmarks

One folder per model family. Each `<Model>/run_model.py` keeps the published
kernel-relevant layer dimensions and one minimal repeated layer period, swaps in
the cuDNN ops that exist today, runs a fwd+bwd training step, and prints where the
GPU kernel time goes — by **category**
(linear-attn / full-attn / gemm / norm / misc) and by **backend** (cuDNN / cuBLAS
/ torch). The point is the *support gap*: what fraction of a real training step
cuDNN already owns, and which un-owned op is next. Layer count and vocabulary may
be reduced to avoid repeating identical work; every such reduction or stand-in is
printed and documented rather than described as a full-model benchmark.

The current dense preset follows Qwen3.8-27B (and the kernel-equivalent
Qwen3.5/3.6-27B): H=5120, I=17408, GDN 16 QK / 48 V heads at head-dim 128, and a 3:1
GDN/full-attention period. It keeps four layers and scales the vocabulary by the
same 16x factor as the depth, retaining the LM-head/layer FLOP ratio. The MLP
can route through `cudnn.gemm.ops.swiglu_mlp` (`--accelerate_mlp`) and linear
attention through `cudnn.fla` (`--accelerate_attn`, PR #596).

The MLP op's forward fuses gate+up+SiLU+mul; its backward fuses the
`dh = dout @ Wd` dgrad GEMM with dSwiGLU into one FROST (cuTeDSL) kernel. The
benchmark can toggle this independently from the faster GDN shim, so their
end-to-end contributions are measured rather than multiplied from microbenchmarks.

On a full B200 with BF16, batch 4, sequence 2048, and the four-layer Qwen3.8
period, a 40-batch balanced round-robin measured:

| GDN path | MLP path | p50 step | paired ratio vs baseline | speedup |
|---|---|---:|---:|---:|
| FLA | FLA/Torch | 74.587 ms | 1.00000 | 1.000x |
| FLA | `swiglu_mlp` | 72.411 ms | 0.96572 | 1.035x |
| cuDNN shim | FLA/Torch | 64.107 ms | 0.85847 | 1.165x |
| cuDNN shim | `swiglu_mlp` | 62.804 ms | 0.84257 | 1.187x |

The combined result lowers paired elapsed time by 15.74% and won all 40
batches. The GDN arm includes a pending packed-QKV stride fix: FLA short-conv
produces strided views, which the shim compacts before the native kernel; that
copy is included in the timed region. These are proxy results, not full 64-layer
throughput.

The shared, model-agnostic harness lives in [`_perfshare.py`](_perfshare.py); a
model file only builds its model, applies the swaps, and calls `profile_and_report`.

## Models

| folder | proxy of | MLP swap | notes |
|---|---|---|---|
| [`Qwen3.8/`](Qwen3.8/) | Qwen3.8/3.6/3.5-27B dense hybrid Gated DeltaNet LM | `swiglu_mlp` | exact MLP/GDN dimensions; 4-layer period; FLA stand-in uses the cuDNN backend d256 GQA path at 20Q/4KV instead of gated 24Q/4KV |

Planned: Kimi Linear (KDA), DeepSeek-V3.

## Run

```bash
python benchmark/e2e/Qwen3.8/run_model.py --bs 4 --accelerate_mlp 1                  # MLP on cuDNN
python benchmark/e2e/Qwen3.8/run_model.py --bs 4 --accelerate_mlp 1 --accelerate_attn 1  # + linear-attn
python benchmark/e2e/Qwen3.8/run_model.py --bs 4 --accelerate_mlp 0                  # baseline
python benchmark/e2e/Qwen3.8/run_model.py --preset qwen3.5-27b --inspect      # equivalent older preset
```

Requires a cuDNN build with the fused GEMM engine and the cuDNN >= 9.23 backend
d256 SDPA path on an SM100 (Blackwell) device; `flash-linear-attention` provides
the model. The benchmark rejects the older OSS/CuteDSL d256 SDPA fallback.

## Add a model

1. `mkdir benchmark/e2e/<Model>` and add `run_model.py`.
2. Build the model, apply the cuDNN swaps, then call `profile_and_report(model, ids, ...)`.
3. Add a row to the table above.
