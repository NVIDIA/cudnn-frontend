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

Every model leaf is a controlled **cuDNN-off versus cuDNN-on** experiment. The
off treatment uses the credible non-cuDNN framework route, even when the stock
framework dispatcher already selects cuDNN by default. The broader program has
five purposes:

1. expose cuDNN technology and its user-visible impact;
2. measure the return from fusions and specialized kernels;
3. surface integration gaps before users hit them;
4. identify the next high-value kernel opportunity; and
5. take missed opportunities back upstream through Megatron, vLLM, SGLang, and
   related integrations.

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
four of the 60 repeated transformer blocks. Each block has one image-stream and
one text-stream dense biased GELU MLP; this model has no MoE/router. Forty
Williams-balanced batches with three repeats measure two independent treatments:

- `M`: stock Diffusers GELU MLP versus public
  `cudnn.gemm.ops.gelu_mlp(x, w1, b1, w2, b2)`; and
- `A`: explicitly forced PyTorch FlashAttention versus the FE public cuDNN
  backend graph for joint SDPA.

| bits (M/A) | GELU MLP | joint SDPA | p50 transformer forward | paired ratio vs `00` | wins vs `00` |
|---|---|---|---:|---:|---:|
| `00` | Torch | forced PyTorch Flash | 9.978 ms | 1.00000 | -- |
| `01` | Torch | cuDNN backend | 7.921 ms | 0.79444 | 40/40 |
| `10` | cuDNN | forced PyTorch Flash | 9.785 ms | 0.98292 | 34/40 |
| `11` | cuDNN | cuDNN backend | 7.770 ms | **0.78121** | **40/40** |

The directly paired `11/00` result is 21.88% lower elapsed time, or **1.280x**.
The conditional attention ratio is 0.79285 (**1.261x**, 2.045 ms median
saving); the conditional GELU-MLP ratio is 0.98403 (**1.016x**, 0.142 ms median
saving). The MLP win is directionally consistent in both contexts: 34/40 wins
with Flash attention and 38/40 with cuDNN attention.

`gelu_mlp` implements the same published biased
`Linear -> GELU(approximate="tanh") -> Linear` FFN semantics and matches Torch
within the documented BF16 tolerance. Its forward fuses the first matmul, bias,
BF16 boundary, and GELU into one cuDNN graph launch, then runs the output linear
as a second graph; its autograd path also fuses
`dout @ w2` with GELU backward. It is not the SwiGLU op used by Qwen3.8.

A separate CUDA-event diagnostic explains where the E2E gain comes from. These
are mutually exclusive module regions from that diagnostic run, not the formal
factorial samples or FLOP shares:

Diagnostic artifact SHA-256: `fb3cb4d9381b400fec80d2635cf32c1d19926c3933a6c0356fe2120cd4aa3ef1`.

| four-block region | cuDNN-off time | off share | cuDNN-on time |
|---|---:|---:|---:|
| two GELU MLPs per block | 2.030 ms | 21.0% | 1.861 ms |
| joint SDPA core | 2.749 ms | 28.4% | 0.677 ms |
| attention projections, QK norm, RoPE, and surrounding work | 3.307 ms | 34.2% | 3.493 ms |
| AdaLN, residuals, output, and other work | 1.589 ms | 16.4% | 1.626 ms |

The stock unforced Torch call already selects `CUDNN_ATTENTION` at this d128
shape. That is successful cuDNN adoption, not a reason to discard the result:
the controlled experiment explicitly disables cuDNN SDPA by forcing Flash in
the off arm and quantifies the full-transformer impact of turning cuDNN back on.
It does not claim a further 1.280x from changing Torch's current dispatcher.

The complete four-block outputs matched the `00` baseline within 0.142%
relative L2; the focused B=2 unequal-text mask case matched within 0.300%.
Raw artifact SHA-256: `63274d0602fe0582088f5241e0dcddcaac244c1426c955bc8e979c4a09fb55d3`.

## Qwen-Image ModelOpt NVFP4 proxy result

The sibling [`Qwen-Image/run_nvfp4.py`](Qwen-Image/run_nvfp4.py) leaf anchors
its placement and quantization policy to NVIDIA ModelOpt 0.46.0 commit
`43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`: Qwen-Image's middle
transformer blocks use NVFP4 E2M1/block-16 Linears with E4M3 block scales and
`max` calibration. ModelOpt's Qwen invocation does not enable
`quantize_mha`, so joint attention remains BF16; this is not an all-FP4 model
or an MXFP8-attention result.

The four proxy blocks represent full-model blocks `[2, 20, 39, 57]` from the
official quantized range 2..57. Every block routes all 14 Linear roles through
NVFP4, including the two M=1 modulation projections. The proxy deliberately
uses BF16 as its high-precision dtype and one synthetic frozen max-calibration
pass instead of ModelOpt's default FP16/calibration workload. It therefore
claims recipe-policy alignment and kernel-plumbing/performance evidence, not
official calibration state or image quality. All 56 weights are prepacked once
during setup and excluded from event timing; this differs from the quoted bare
ModelOpt CLI's default `compress=false` execution state.

On the same full B200 and formal four-block shape, 42 position- and
carryover-balanced batches with three repeats measured:

| arm | Linear / FFN treatment | joint SDPA | p50 transformer forward | paired ratio vs A | wins vs A |
|---|---|---|---:|---:|---:|
| `A` | Torch BF16 / Diffusers GELU FFN | forced PyTorch Flash BF16 | 9.852 ms | 1.00000 | -- |
| `B` | Torch BF16 + cuDNN BF16 `gelu_mlp` | cuDNN BF16 | 7.782 ms | **0.78997** | **42/42** |
| `C` | cuDNN FROST NVFP4 for all 14 Linears | cuDNN BF16 | 7.646 ms | **0.77567** | **42/42** |

At fixed BF16, B/A is 21.00% lower elapsed time, or **1.266x**. The complete
cuDNN-enabled low-precision stack C is 22.43% lower than A, or **1.289x**. C is
also 1.73% lower than the already-optimized BF16 cuDNN arm B
(`C/B=0.98267`, **1.018x**, 37/42 wins). This incremental low-precision win is
modest: its paired p10--p90 ratio spans 0.97600--1.00452, so a few batches
slightly favor B.

The final C path caches typed views and resolved FROST bindings per logical
Linear, validates every stable buffer in `select("C")` outside the CUDA-event
region, and uses the public `run_resolved` entry point. Direct Linear outputs
remain fresh allocations whose temporary binding slot is always cleared. This
removes the repeated host binding work exposed by the earlier diagnostic while
retaining strict route and lifetime guards; it does not use a private lowered
launcher.

The run requires exact successful routes for all 56 NVFP4 Linears, with 56
logical activation quantizations reduced to 33 physical operations (25
standalone, eight fused, and 23 cache hits). A setup-only numerical gate
executes all seven distinct M/N/K/epilogue contracts against an independent
E2M1/F8_128x4 dequantized reference. The four-block C output differs from A by
0.852% relative L2, which is recorded only as a finite diagnostic because the
proxy uses random weights and synthetic calibration.

Raw artifact SHA-256: `7af126f91ea958a8912e611168136afc2241fbc79e9d74d4a26ace907648f7e6`.
The benchmark-private BF16-to-NVFP4 kernel is derived from FlashInfer commit
`f212ec8230486e3615502b8af75fe7022c60b2f3`, retaining its Apache-2.0 notice
and its TensorRT-LLM provenance; FROST folds dequantization into the MMA.

For Qwen3.8, the shared, model-agnostic harness lives in
[`_perfshare.py`](_perfshare.py); its model file only builds the model, applies
the swaps, and calls `profile_and_report`.

## Models

| folder | proxy of | current precision leaf | notes |
|---|---|---|---|
| [`Qwen3.8/`](Qwen3.8/) | Qwen3.8/3.6/3.5-27B dense hybrid Gated DeltaNet LM | BF16 fwd+CE+bwd, 2^3 GDN/MLP/SDPA | exact MLP/GDN dimensions; 4-layer period; selectable Torch FlashAttention or cuDNN-backend d256 GQA at 20Q/4KV instead of gated 24Q/4KV |
| [`Qwen-Image/`](Qwen-Image/) | Qwen-Image diffusion transformer | BF16 2^2 GELU-MLP/SDPA plus ModelOpt-anchored NVFP4 three-arm leaf | exact H=3072, 24x128 and FFN=12288; 4/60 repeated blocks; 4096 image + 512 text tokens; no MoE |

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

# Validation-only reduced-token mask/route/correctness smoke for all four
# Torch/cuDNN MLP x attention treatments.
python benchmark/e2e/Qwen-Image/run_bf16.py \
  --mode smoke --output-dir qwen-image-bf16-results/smoke

# Formal four-arm BF16 transformer proxy: B=1, 4096 image + 512 text tokens,
# four real-shape blocks, 40 Williams-balanced batches x 3 repeats on a full B200.
python benchmark/e2e/Qwen-Image/run_bf16.py \
  --mode formal --output-dir qwen-image-bf16-results/formal

# ModelOpt-anchored NVFP4 validation. The benchmark-private quantizer lazily
# compiles one SM100 CUDA extension, so CUDA_HOME, an sm_100a-capable nvcc,
# a host C++ compiler, Ninja, and a writable Torch extension cache are required.
python benchmark/e2e/Qwen-Image/run_nvfp4.py \
  --mode smoke --output-dir qwen-image-nvfp4-results/smoke

# Formal three-arm A/B/C run: BF16 off, BF16 cuDNN, then all-Linear NVFP4 cuDNN.
python benchmark/e2e/Qwen-Image/run_nvfp4.py \
  --mode formal --output-dir qwen-image-nvfp4-results/formal
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
for this d128 shape). The orthogonal MLP axis compares the stock Diffusers FFN
with the public cuDNN GELU-MLP op. The BF16 leaf itself has no ModelOpt claim;
the separate `run_nvfp4.py` leaf owns the pinned low-precision recipe, synthetic
calibration disclosure, route gates, and low-precision artifact.

## Add a model

1. `mkdir benchmark/e2e/<Model>` and put topology plus backend adapters in `run_model.py`.
2. Add one workload/precision leaf such as `run_bf16.py`; add FP8/FP4 as sibling files rather than dtype branches throughout the model.
3. Record immutable upstream model/implementation/recipe anchors and the resolved shape in every artifact.
4. Add CPU spec tests, a target-GPU route/correctness smoke, and a row above.
