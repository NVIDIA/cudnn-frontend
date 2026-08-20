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
- stock FLA `GatedMLP` versus the opt-in `cudnn.fla` `gated_mlp` target backed
  by `cudnn.gemm.ops.swiglu_mlp` (`--accelerate_mlp`, PR #609); and
- vanilla `torch.nn.functional.scaled_dot_product_attention` versus FE's direct
  cuDNN-backend d256 op (`--full_attn_backend`, develop #335).

The model-level true-vanilla arm means stock FLA GDN + stock FLA MLP + vanilla
Torch SDPA; it is not an all-eager-Torch implementation. At the benchmark shape,
Torch 2.13 selects PyTorch FlashAttention rather than cuDNN. Only the MLP axis
belongs to PR #609.

The formal results below predate this public adapter and used an equivalent
direct call to `cudnn.gemm.ops.swiglu_mlp`. The adapter itself is validated by
the focused FLA compatibility suite and a native-route model smoke test.

## Qwen3.8 proxy result

On a full 148-SM B200 with BF16, batch 4, sequence 2048, and the four-layer
Qwen3.8 period, a 2^3 Williams-balanced experiment used 40 batches and three
repeats per arm. JIT/autotune, correctness, warmup, and gradient clearing were
outside the CUDA-event region. [`Qwen3.8/run_matrix.py`](Qwen3.8/run_matrix.py)
reproduces that design and emits both raw JSON and a generated Markdown report.
Each report includes the fully resolved shape, software/device provenance,
loaded source hashes, route/correctness results, a stable comparability
fingerprint, and a separate build/provenance fingerprint.

| bits (G/M/A) | GDN | MLP | full-attn core | p50 step | paired ratio vs `000` | wins vs `000` |
|---|---|---|---|---:|---:|---:|
| `000` | stock FLA | stock FLA | Torch FlashAttention | 75.075 ms | 1.00000 | -- |
| `001` | stock FLA | stock FLA | cuDNN backend | 73.326 ms | 0.98280 | 29/40 |
| `010` | stock FLA | PR #609 | Torch FlashAttention | 72.483 ms | 0.96579 | 34/40 |
| `011` | stock FLA | PR #609 | cuDNN backend | 71.180 ms | 0.94548 | 38/40 |
| `100` | cuDNN shim | stock FLA | Torch FlashAttention | 64.178 ms | 0.85348 | 40/40 |
| `101` | cuDNN shim | stock FLA | cuDNN backend | 62.788 ms | 0.83878 | 40/40 |
| `110` | cuDNN shim | PR #609 | Torch FlashAttention | 62.394 ms | 0.83025 | 40/40 |
| `111` | cuDNN shim | PR #609 | cuDNN backend | 61.230 ms | **0.81533** | **40/40** |

The directly paired `111/000` result is 18.47% lower elapsed time, or 1.226x.
The three ratios below average each axis over all four contexts within every
batch, so interactions are measured instead of multiplying isolated speedups.
Shapley savings are mean per-batch attribution values and are not module-time
shares.

| axis | conditional paired ratio | conditional speedup | mean Shapley saving |
|---|---:|---:|---:|
| GDN | 0.85468 | 1.170x | 10.54 ms |
| MLP | 0.97036 | 1.031x | 1.92 ms |
| d256 full attention | 0.97836 | 1.022x | 1.42 ms |

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

The recorded GDN arm included the packed-QKV stride fix from #685: FLA
short-conv produces strided views, which the shim compacts before the native
kernel; that copy is inside the timed region. #685 is not yet merged into
`develop`, so the default `short_conv=true` smoke and formal configurations
currently require a build containing that fix. The runner does not pin a
private checkout or require the fix by source hash. Instead it records the
loaded source hashes and requires exact successful native-GDN call counts plus
the end-to-end correctness gate; an incompatible shim fails explicitly. These
are single-job results for a four-layer shape proxy, not full 64-layer Qwen
throughput.

The shared, model-agnostic harness lives in [`_perfshare.py`](_perfshare.py); a
model file only builds its model, applies the swaps, and calls `profile_and_report`.

## Models

| folder | proxy of | MLP swap | notes |
|---|---|---|---|
| [`Qwen3.8/`](Qwen3.8/) | Qwen3.8/3.6/3.5-27B dense hybrid Gated DeltaNet LM | `swiglu_mlp` | exact MLP/GDN dimensions; 4-layer period; selectable Torch FlashAttention or cuDNN-backend d256 GQA at 20Q/4KV instead of gated 24Q/4KV |

Planned: Kimi Linear (KDA), DeepSeek-V3.

## Run

```bash
# The factorial math/reporting tests are CPU-only and do not import Torch.
python -m unittest discover -s benchmark/e2e/tests -v

# Validation-only smoke test. It preserves H/I/head dimensions and the four-layer
# 3:1 period, but M=bs*seq=128 (versus formal M=8192), so fixed overhead dominates.
# Do not use smoke timing for performance trends, headlines, or speedup claims.
python benchmark/e2e/Qwen3.8/run_matrix.py \
  --mode smoke --output-dir qwen3.8-factorial-results/smoke

# Formal 8-arm Williams run: full 148-SM B200, bs=4, seq=2048,
# 40 balanced batches, 3 repeats. Produces timestamped .json and .md artifacts.
python benchmark/e2e/Qwen3.8/run_matrix.py \
  --mode formal --output-dir qwen3.8-factorial-results/formal

# CI/job wrappers may request stable artifact names explicitly.
python benchmark/e2e/Qwen3.8/run_matrix.py \
  --mode formal \
  --raw-json qwen3.8-factorial-results/qwen38.json \
  --markdown qwen3.8-factorial-results/qwen38.md

# Compare a new formal run with an independently collected prior formal artifact.
# The command rejects mismatched comparability fingerprints. The report labels
# per-arm p50 and headline changes as cross-run, non-paired comparisons.
python benchmark/e2e/Qwen3.8/run_matrix.py \
  --mode formal \
  --compare qwen3.8-factorial-results/previous.json \
  --output-dir qwen3.8-factorial-results/comparison

# A single-arm profile remains available for support-share inspection.
# Three-axis true vanilla: stock FLA GDN/MLP + vanilla Torch SDPA.
python benchmark/e2e/Qwen3.8/run_model.py --bs 4 --accelerate_mlp 0 --accelerate_attn 0 --full_attn_backend torch

# All three accelerated (requires the cudnn.fla shim).
python benchmark/e2e/Qwen3.8/run_model.py --bs 4 --accelerate_mlp 1 --accelerate_attn 1 --full_attn_backend cudnn

python benchmark/e2e/Qwen3.8/run_model.py --preset qwen3.5-27b --inspect  # equivalent older preset
```

Requires a cuDNN build with the fused GEMM engine and the cuDNN >= 9.23 backend
d256 SDPA path on an SM100 (Blackwell) device; `flash-linear-attention` provides
the model. The MLP target requires FLA 0.5.2 and admits its validated plain,
local, bias-free BF16 `swish` module; unsupported runtime configurations fall
back to FLA. The default `short_conv=true` GDN axis uses the packed-QKV support
landed in #685. The benchmark rejects the older OSS/CuteDSL d256 SDPA fallback.
Formal mode additionally rejects anything other than a full 148-SM NVIDIA B200. Both modes verify exact
GDN/MLP/full-attention routes and explicit finite correctness before reporting.
Formal shape/timing overrides are allowed but are listed prominently and
included in the comparability fingerprint. The default
`qwen3.8-factorial-results/` output directory is gitignored wherever the runner
is launched inside the checkout.

## Add a model

1. `mkdir benchmark/e2e/<Model>` and add `run_model.py`.
2. Build the model, apply the cuDNN swaps, then call `profile_and_report(model, ids, ...)`.
3. Add a row to the table above.
