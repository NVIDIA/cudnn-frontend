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
same 16x factor as the depth, retaining the LM-head/layer FLOP ratio.

The proxy exposes three independent axes:

- stock FLA versus `cudnn.fla` for linear GDN (`--accelerate_attn`, PR #596);
- stock FLA `GatedMLP` versus `cudnn.gemm.ops.swiglu_mlp`
  (`--accelerate_mlp`, PR #609); and
- vanilla `torch.nn.functional.scaled_dot_product_attention` versus FE's direct
  cuDNN-backend d256 op (`--full_attn_backend`, develop #335).

The model-level true-vanilla arm means stock FLA GDN + stock FLA MLP + vanilla
Torch SDPA; it is not an all-eager-Torch implementation. At the benchmark shape,
Torch 2.13 selects PyTorch FlashAttention rather than cuDNN. Only the MLP axis
belongs to PR #609.

## Qwen3.8 proxy result

On a full 148-SM B200 with BF16, batch 4, sequence 2048, and the four-layer
Qwen3.8 period, a 2^3 Williams-balanced experiment used 40 batches and three
repeats per arm. JIT/autotune, correctness, warmup, and gradient clearing were
outside the CUDA-event region.

| bits (G/M/A) | GDN | MLP | full-attn core | p50 step | paired ratio vs `000` | wins vs `000` |
|---|---|---|---|---:|---:|---:|
| `000` | stock FLA | stock FLA | Torch FlashAttention | 76.047 ms | 1.00000 | -- |
| `001` | stock FLA | stock FLA | cuDNN backend | 75.120 ms | 0.98555 | 25/40 |
| `010` | stock FLA | PR #609 | Torch FlashAttention | 74.619 ms | 0.97806 | 29/40 |
| `011` | stock FLA | PR #609 | cuDNN backend | 73.035 ms | 0.95995 | 38/40 |
| `100` | cuDNN shim | stock FLA | Torch FlashAttention | 66.075 ms | 0.86980 | 40/40 |
| `101` | cuDNN shim | stock FLA | cuDNN backend | 65.174 ms | 0.85522 | 40/40 |
| `110` | cuDNN shim | PR #609 | Torch FlashAttention | 65.071 ms | 0.85309 | 40/40 |
| `111` | cuDNN shim | PR #609 | cuDNN backend | 64.090 ms | **0.83686** | **40/40** |

The directly paired `111/000` result is 16.31% lower elapsed time, or 1.195x.
The three ratios below average each axis over all four contexts within every
batch, so interactions are measured instead of multiplying isolated speedups.
Shapley savings are mean per-batch attribution values and are not module-time
shares.

| axis | conditional paired ratio | conditional speedup | mean Shapley saving |
|---|---:|---:|---:|
| GDN | 0.86793 | 1.152x | 9.76 ms |
| MLP | 0.98154 | 1.019x | 1.26 ms |
| d256 full attention | 0.98315 | 1.017x | 1.16 ms |

An independently attributed true-vanilla CUDA profile gives the approximate,
mutually exclusive GPU active-time shares below. Generic GEMM kernels account
for about 70% of active work, but that view overlaps the module rows because it
includes MLP GEMMs, GDN and attention projections, and the LM head.

| module group | active time | share |
|---|---:|---:|
| four stock FLA MLP blocks | 37.18 ms | 51.6% |
| three stock FLA GDN blocks | 24.98 ms | 34.7% |
| one Torch full-attention block | 4.59 ms | 6.4% |
| LM head, norms, embedding, misc | 5.18 ms | 7.2% |

Focused A/Bs establish the smaller-axis signal outside model-level noise:

- Exact MLP fwd+bwd (`M=8192, H=5120, I=17408`): raw eager Torch 10.796 ms,
  stock FLA 0.5.2 `GatedMLP` 10.943 ms, and PR #609 9.848 ms. PR #609 is
  1.097x versus raw Torch and 1.111x versus stock FLA, winning all 40 batches.
- Exact d256 GQA SDPA core: PyTorch FlashAttention 2.619 ms versus cuDNN backend
  1.198 ms (2.188x). The full attention block including common QKV, RoPE, layout,
  O projection, and backward is 5.147 versus 3.432 ms (1.494x, 40/40).

The GDN arm includes a pending packed-QKV stride fix: FLA short-conv produces
strided views, which the shim compacts before the native kernel; that copy is
inside the timed region. These are single-job results for a four-layer shape
proxy, not full 64-layer Qwen throughput.

The shared, model-agnostic harness lives in [`_perfshare.py`](_perfshare.py); a
model file only builds its model, applies the swaps, and calls `profile_and_report`.

## Models

| folder | proxy of | MLP swap | notes |
|---|---|---|---|
| [`Qwen3.8/`](Qwen3.8/) | Qwen3.8/3.6/3.5-27B dense hybrid Gated DeltaNet LM | `swiglu_mlp` | exact MLP/GDN dimensions; 4-layer period; selectable Torch FlashAttention or cuDNN-backend d256 GQA at 20Q/4KV instead of gated 24Q/4KV |

Planned: Kimi Linear (KDA), DeepSeek-V3.

## Run

```bash
# Three-axis true vanilla: stock FLA GDN/MLP + vanilla Torch SDPA.
python benchmark/e2e/Qwen3.8/run_model.py --bs 4 --accelerate_mlp 0 --accelerate_attn 0 --full_attn_backend torch

# All three accelerated (requires the cudnn.fla shim).
python benchmark/e2e/Qwen3.8/run_model.py --bs 4 --accelerate_mlp 1 --accelerate_attn 1 --full_attn_backend cudnn

python benchmark/e2e/Qwen3.8/run_model.py --preset qwen3.5-27b --inspect  # equivalent older preset
```

Requires a cuDNN build with the fused GEMM engine and the cuDNN >= 9.23 backend
d256 SDPA path on an SM100 (Blackwell) device; `flash-linear-attention` provides
the model. The benchmark rejects the older OSS/CuteDSL d256 SDPA fallback.

## Add a model

1. `mkdir benchmark/e2e/<Model>` and add `run_model.py`.
2. Build the model, apply the cuDNN swaps, then call `profile_and_report(model, ids, ...)`.
3. Add a row to the table above.
