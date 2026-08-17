# End-to-end model perf-share benchmarks

One folder per model. Each `<Model>/run_model.py` builds a lightweight, real-shape
proxy of a hot model, swaps in the cuDNN ops that exist today, runs a fwd+bwd
training step, and prints where the GPU kernel time goes — by **category**
(linear-attn / full-attn / gemm / norm / misc) and by **backend** (cuDNN / cuBLAS
/ torch). The point is the *support gap*: what fraction of a real training step
cuDNN already owns, and which un-owned op is next.

At real model dims the **SwiGLU-MLP GEMMs are ~70% of a step** while linear
attention is only ~6%, so the dominant lever is the MLP. Each model can route the
MLP through `cudnn.gemm.ops.swiglu_mlp` (`--accelerate_mlp`) and, if the package is
installed, linear attention through `cudnn.fla` (`--accelerate_attn`, PR #596).

The MLP op's forward fuses gate+up+SiLU+mul (1.05-1.20x); its backward fuses the
`dh = dout @ Wd` dgrad GEMM with the dSwiGLU elementwise into one FROST (cuTeDSL)
kernel — ~1.5x the recompute+pointwise backward and ~1.25x a fair torch backward
at the Qwen3.5-27B MLP shape (CUDA-graph kernel time, SM100). Both directions now
win, so the MLP is a training-step win, not just an inference one.

The shared, model-agnostic harness lives in [`_perfshare.py`](_perfshare.py); a
model file only builds its model, applies the swaps, and calls `profile_and_report`.

## Models

| folder | proxy of | MLP swap | notes |
|---|---|---|---|
| [`Qwen3-Next/`](Qwen3-Next/) | Qwen3-Next hybrid Gated DeltaNet LM | `swiglu_mlp` | forward + FROST-fused backward both win (fwd 1.05-1.20x, bwd ~1.25x vs torch); training-step win |

Planned: Kimi Linear (KDA), DeepSeek-V3.

## Run

```bash
python benchmark/e2e/Qwen3-Next/run_model.py --accelerate_mlp 1                  # MLP on cuDNN
python benchmark/e2e/Qwen3-Next/run_model.py --accelerate_mlp 1 --accelerate_attn 1  # + linear-attn (needs PR #596)
python benchmark/e2e/Qwen3-Next/run_model.py --accelerate_mlp 0                  # baseline
python benchmark/e2e/Qwen3-Next/run_model.py --inspect                          # structure + GEMM sites
```

Requires a cuDNN build with the fused GEMM engine on an SM100 (Blackwell) device;
`flash-linear-attention` provides the model.

## Add a model

1. `mkdir benchmark/e2e/<Model>` and add `run_model.py`.
2. Build the model, apply the cuDNN swaps, then call `profile_and_report(model, ids, ...)`.
3. Add a row to the table above.
