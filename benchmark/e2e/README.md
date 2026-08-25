# End-to-end model perf-share benchmarks

One folder per model family. Each `<Model>/run_model.py` keeps the published
kernel-relevant layer dimensions and one minimal repeated layer period, while a
separate precision leaf (for example `run_matrix.py` or `run_bf16.py`) declares
the workload and implementation treatments. Single-arm profiles can break GPU
kernel time down by **category** (linear-attn / full-attn / gemm / norm / misc)
and **backend** (cuDNN / cuBLAS / torch); matrix leaves report paired end-to-end
treatment effects. The point is the *support gap*: what fraction of a real
training step cuDNN already owns, and which un-owned op is next. Layer count and vocabulary may
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
- vanilla `torch.nn.functional.scaled_dot_product_attention` versus FE's public
  backend-graph d256 op (`--full_attn_backend`, #335; legacy standalone stacks
  removed by #682).

The model-level true-vanilla arm means stock FLA GDN + stock FLA MLP + vanilla
Torch SDPA; it is not an all-eager-Torch implementation. At the benchmark shape,
Torch 2.13 selects PyTorch FlashAttention rather than cuDNN. Only the MLP axis
belongs to PR #609.

The formal results below exercise the merged public `cudnn.fla` MLP adapter,
the merged packed-QKV GDN adapter, and FE's backend-only d256 SDPA path. Every
accelerated arm fails unless the requested native route is observed.

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
| `000` | stock FLA | stock FLA | Torch FlashAttention | 75.427 ms | 1.00000 | -- |
| `001` | stock FLA | stock FLA | cuDNN backend | 74.516 ms | 0.98939 | 27/40 |
| `010` | stock FLA | cuDNN shim (#609) | Torch FlashAttention | 73.673 ms | 0.96456 | 32/40 |
| `011` | stock FLA | cuDNN shim (#609) | cuDNN backend | 71.378 ms | 0.94218 | 38/40 |
| `100` | cuDNN shim | stock FLA | Torch FlashAttention | 65.016 ms | 0.86099 | 40/40 |
| `101` | cuDNN shim | stock FLA | cuDNN backend | 63.801 ms | 0.84558 | 40/40 |
| `110` | cuDNN shim | cuDNN shim (#609) | Torch FlashAttention | 63.729 ms | 0.84446 | 40/40 |
| `111` | cuDNN shim | cuDNN shim (#609) | cuDNN backend | 62.544 ms | **0.82759** | **40/40** |

The directly paired `111/000` result is 17.24% lower elapsed time, or 1.208x.
Raw artifact SHA-256: `65654ec5c55f7e900f1351c60d8f835b2d7e427613b8d1df18dd7f3d9b1b2757`.
The three ratios below average each axis over all four contexts within every
batch, so interactions are measured instead of multiplying isolated speedups.
Shapley savings are mean per-batch attribution values and are not module-time
shares.

| axis | conditional paired ratio | conditional speedup | mean Shapley saving |
|---|---:|---:|---:|
| GDN | 0.86005 | 1.163x | 10.13 ms |
| MLP | 0.96876 | 1.032x | 1.90 ms |
| d256 full attention | 0.98280 | 1.017x | 1.18 ms |

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

The GDN arm includes the packed-QKV stride fix from #685: FLA short-conv
produces strided views, which the shim compacts before the native kernel; that
copy is inside the timed region. The runner records the loaded source hashes and
requires exact successful native-GDN call counts plus the end-to-end correctness
gate; an incompatible shim fails explicitly. These are single-job results for a
four-layer shape proxy, not full 64-layer Qwen throughput.

## Qwen-Image BF16 proxy result

On the same full B200, the Qwen-Image proxy uses the published H=3072,
24x128-head and FFN=12288 dimensions, B=1, 4096 image plus 512 text tokens, and
four of the 60 repeated transformer blocks. Forty balanced batches with three
repeats compare an explicitly forced PyTorch FlashAttention treatment with the
FE public cuDNN backend graph; all projections, QK norm, RoPE, AdaLN, biased
GELU FFNs, residuals, and output work are common and remain inside the timed
transformer forward.

| SDPA treatment | p50 transformer forward | paired ratio | latency reduction | speedup | wins |
|---|---:|---:|---:|---:|---:|
| forced PyTorch FlashAttention | 9.943 ms | 1.00000 | -- | -- | -- |
| direct FE/cuDNN backend | 7.883 ms | **0.79640** | **20.36%** | **1.256x** | **40/40** |

The unforced public Torch call already selects `CUDNN_ATTENTION` at this d128
shape, so this is an implementation A/B rather than a claim of an additional
dispatcher-level user speedup. The complete four-block output matched within
0.141% relative L2. The focused B=2 unequal-text mask case matched within 0.300%
relative L2. Raw artifact SHA-256: `288ce0415c0cfd6564fde99debe0273a2304c07e7810547bd5da8b25cda0fbba`.

The shared, model-agnostic harness lives in [`_perfshare.py`](_perfshare.py); a
model file only builds its model, applies the swaps, and calls `profile_and_report`.

## Models

| folder | proxy of | current precision leaf | notes |
|---|---|---|---|
| [`Qwen3.8/`](Qwen3.8/) | Qwen3.8/3.6/3.5-27B dense hybrid Gated DeltaNet LM | BF16 fwd+CE+bwd, 2^3 GDN/MLP/SDPA | exact MLP/GDN dimensions; 4-layer period; selectable Torch FlashAttention or cuDNN-backend d256 GQA at 20Q/4KV instead of gated 24Q/4KV |
| [`Qwen-Image/`](Qwen-Image/) | Qwen-Image diffusion transformer | BF16 transformer forward, forced PyTorch Flash-vs-cuDNN joint SDPA | exact H=3072, 24x128 and FFN=12288; 4/60 repeated blocks; 4096 image + 512 text tokens |

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

# Qwen-Image uses the pinned benchmark-only Diffusers implementation.
python -m pip install -r benchmark/e2e/Qwen-Image/requirements.txt

# Validation-only reduced-token mask/route/correctness smoke.
python benchmark/e2e/Qwen-Image/run_bf16.py \
  --mode smoke --output-dir qwen-image-bf16-results/smoke

# Formal one-forward BF16 transformer proxy: B=1, 4096 image + 512 text tokens,
# four real-shape blocks, 40 balanced batches x 3 repeats on a full B200.
python benchmark/e2e/Qwen-Image/run_bf16.py \
  --mode formal --output-dir qwen-image-bf16-results/formal
```

The Qwen3.8 runner requires a cuDNN build with the fused GEMM engine and the
cuDNN >= 9.23 backend d256 SDPA path on an SM100 (Blackwell) device;
`flash-linear-attention` provides the model. The MLP target requires FLA 0.5.2
and admits its validated plain, local, bias-free BF16 `swish` module;
unsupported runtime configurations fall back to FLA. The default
`short_conv=true` GDN axis uses the packed-QKV support landed in #685. After
#682, the FE public d256 SDPA operator is backend-graph-only. Formal mode
additionally rejects anything other than a full 148-SM NVIDIA B200. Both modes verify exact
GDN/MLP/full-attention routes and explicit finite correctness before reporting.
Formal shape/timing overrides are allowed but are listed prominently and
included in the comparability fingerprint. The default
`qwen3.8-factorial-results/` output directory is gitignored wherever the runner
is launched inside the checkout.

### Qwen-Image joint mask

Qwen-Image performs non-causal joint attention in `[text, image]` order. Every
query can see every valid text token and every image token; only padded text key
columns are masked. Canonical batch-1 inference trims text padding and therefore
uses the dense no-mask path. For unequal right-padded prompts, the cuDNN adapter
temporarily permutes the joint sequence to `[image, text]`, where the padding is
a suffix representable by `seq_len_kv`, then restores output order. Every run
checks this path against the official boolean-mask semantics with a focused B=2
case. Arbitrary masks with holes fail closed.

The Qwen-Image result is one random-weight conditional transformer forward. It
does not include checkpoint loading, text encoding, VAE, scheduler work, or the
complete denoising loop and is not image-quality evidence. The model/config and
Diffusers implementation are pinned in the artifact. The A/B explicitly forces
PyTorch FlashAttention versus direct FE/cuDNN; the artifact also reports the
unforced public Torch dispatch choice (Torch 2.13 on B200 already selects cuDNN
for this d128 shape). This BF16 leaf has no
ModelOpt claim; a future ModelOpt-anchored NVFP4+FP8 experiment belongs in a
separate `run_nvfp4_fp8.py`.

## Add a model

1. `mkdir benchmark/e2e/<Model>` and put topology plus backend adapters in `run_model.py`.
2. Add one workload/precision leaf such as `run_bf16.py`; add FP8/FP4 as sibling files rather than dtype branches throughout the model.
3. Record immutable upstream model/implementation/recipe anchors and the resolved shape in every artifact.
4. Add CPU spec tests, a target-GPU route/correctness smoke, and a row above.
