# SM120 native blk128 Sage FP8 forward

## Scope and stopping point

This change adds native Q128/KV128 Sage FP8 forward to the existing BSA
wrapper. Public Q/K/V and output are BF16; QK/PV arithmetic is FP8 E4M3
with FP32 accumulation and online softmax. This is not a BF16-arithmetic
kernel, dense-attention speedup claim, or distributed communication change.

Optimization was stopped at the owner's request on 2026-09-20. The last
requested **additional 1.10x** against local revision `174f9b47` was
**not achieved**. This PR retains the already validated implementation and
its small, scoped compiler-scheduling increment. Rejected experiments and
raw profiler files are not included.

The PR's first commit, `5566716e`, preserves the exact attention source
from that fixed baseline (SHA-256
`5a8d0cf16b3e87a96c4f5cb509588ffe3601d572868a452b3bf06c91fed98f85`).
The final kernel source is unchanged from the retained research version
(SHA-256
`5b2c4c26a228055df1f8480ba5edd161a7a7369be0648becb28b2c86686f4ad6`).
The cleaned branch starts from upstream `697eddca`; it does not re-submit
the BF16 implementation already merged in #1070.

## Retained implementation

- A separate native kernel reads the original blk128 sparse indices and
  full Q128/KV128 tiles. It never expands metadata or launches blk64
  attention. Only low-level fragment/softmax helpers are shared.
- Eight compute warps use 240 registers each. An additional four-warp
  group donates registers, retaining 24 each; its first warp independently
  issues TMA loads. The 384-thread CTA uses 64,512 budgeted registers.
- Register-resident Q, shared Q/P storage reuse, native FP8 STSM/LDSM
  probability conversion, and K-major blocked V reduce data movement.
  V is quantized directly into `[B,H,ceil(Sk/128),D,128]`: this replaces
  the existing final quantization launch, without an extra repack kernel.
- Ordered FP32 softmax summation is preserved, with exponentials
  interleaved into that order. A warp-uniform vote skips output rescaling
  only when every owned scale is exactly one. Four compute-warp pairs
  have a one-time startup stagger; per-tile arithmetic order is unchanged.
- Compiler `--register-usage-level=2` is enabled only for blocked-V,
  fixed-count loops with at least 128 selected KV blocks. Short loops,
  variable counts, legacy BHSD V and blk64 keep their original options.
  Options participate in the compile-cache key. This beta tuning feature
  requires performance revalidation after toolchain changes.
- The quantizer's SM120 max reductions use supported shuffle operations;
  the existing SM100 reduction path is preserved. No dependency floor
  or numerical tolerance is raised.

## Validation and measurement

NVIDIA RTX PRO 6000 Blackwell **Server Edition**, 188 SMs, SM120;
PyTorch 2.13.0+cu130, CuTe DSL 4.7.1, cuDNN 9.20. Final tests after
integration onto upstream: **27 native FP8 tests passed**; BSA L0:
**81 passed, 28 skipped, 120 deselected**. Skips/deselections are not
passes; other GPU architectures were not executed here.

Coverage includes native-dispatch guards, fixed/variable/empty counts,
partial Q/KV tails, block-size metadata ranks 1/2/3, custom scales,
runtime metadata changes under CUDA-graph replay, both private V layouts,
FP8 midpoint/signed-zero cases, ordered-softmax byte checks, exactly
seven quantization launches, and tuned/untuned compile-cache separation.
Formatting and SPDX hooks pass. Compute Sanitizer was unavailable.

Target BHSD is `[1,8,142720,128]`, seed 120128. Density 15% selects
167/1115 blocks (14.9776% effective); 20% selects 223/1115.
`strided` is a unique modular-stride mask; `local` is a contiguous
wrapping block window. Neither is a production trace or causal mask.

Each table uses its own same-device alternating paired CUDA-graph
measurements: 10 warmups, 41 samples per path, median milliseconds.
Compilation, allocation and mask setup are excluded; inclusive replay
contains quantization plus attention. Timing and profiling are separate.
Clocks are not locked; do not form ratios across different runs.

## Current native blk128 versus matched-mask FP8 blk64

| Density | Pattern | Blk64 attention ms | Blk128 attention ms | Speedup | Blk64 + quant ms | Blk128 + quant ms | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| 15% | strided | 22.454 | 18.203 | 1.2335x | 24.585 | 20.279 | 1.2123x |
| 15% | local | 22.623 | 18.402 | 1.2294x | 24.750 | 20.353 | 1.2160x |
| 20% | strided | 30.279 | 24.502 | 1.2358x | 32.377 | 26.488 | 1.2223x |
| 20% | local | 30.386 | 24.587 | 1.2358x | 32.522 | 26.623 | 1.2216x |

The blk64 comparator attends to exactly the same tokens: each KV128 index
is expanded to its two KV64 children, and Q-block metadata is duplicated,
**only in baseline setup outside timing**. Native blk128 does neither.
These ratios include the whole native implementation, not just the final
compiler option. They are not dense-attention or BF16-kernel ratios.

Blk64 and blk128 have different softmax tiling, so they need not be
byte-identical. The benchmark's four-query-row FP32-reference diagnostic
has the following relative L2 errors (percent). This small sample is not
an exhaustive precision guarantee, and does not show universally lower
error than blk64:

| Density | Pattern | Blk64 sampled relative L2 | Blk128 sampled relative L2 |
|---|---|---:|---:|
| 15% | strided | 5.6040% | 5.6099% |
| 15% | local | 5.2997% | 5.2955% |
| 20% | strided | 5.5185% | 5.5366% |
| 20% | local | 5.6368% | 5.6198% |

## Final increment over the fixed native baseline

| Density | Pattern | Baseline attention ms | Final attention ms | Speedup | Baseline + quant ms | Final + quant ms | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| 15% | strided | 18.154 | 18.103 | 1.0028x | 20.063 | 19.982 | 1.0040x |
| 15% | local | 18.228 | 18.173 | 1.0030x | 20.146 | 20.078 | 1.0034x |
| 20% | strided | 24.364 | 24.263 | 1.0041x | 26.272 | 26.170 | 1.0039x |
| 20% | local | 24.479 | 24.376 | 1.0042x | 26.397 | 26.293 | 1.0040x |

Every case passes complete byte comparisons for all six quantized tensors
(after interpreting V's layout), attention O and centered-K LSE, and
public quantization-inclusive O. Thus the retained increment does not
change precision on these tested inputs. No comparison tolerance was
relaxed. Both sides use blocked V and the same unchanged quantizer;
baseline compiler options are empty, and only the candidate is tuned.
This small increment is **not** the unachieved additional 10% target.

## NCU evidence

Earlier local profiling of the same retained long-loop configuration,
20% strided, Nsight Compute 2026.3.0 full sections:

| Counter | Fixed native baseline | Retained compiler tuning |
|---|---:|---:|
| Launch registers/thread | 168 | 168 |
| Dynamic shared memory/CTA | 82.05 KB | 82.05 KB |
| Local/shared-memory spill requests | 0 / 0 | 0 / 0 |
| Executed instructions | 13,438,231,756 | 13,404,397,155 |
| Tensor-pipeline utilization | 84.2247% | 84.8281% |
| Scheduler cycles with no eligible warp | 64.81% | 64.67% |

The launch-reported 168 registers/thread is not the dynamic compute-warp
budget of 240. These counters support a minor scheduling/code-generation
change, not increased occupancy or a different arithmetic algorithm.
Both have approximately 8.99 active warps/SM and one resident CTA.
Observed clocks differed (2.15 versus 2.28 GHz); profiler duration is
**not** used to claim a speedup. The tables above are fresh, unprofiled,
paired measurements on the cleaned PR branch.

Raw NCU reports can contain hostnames, paths, process metadata and machine
identifiers. They remain local; only aggregate counters are published.
No card IP, SSH endpoint, device UUID or raw process command is included.

## Reproduce

Use a source checkout of this PR and install the documented CUDA/Python
dependencies. Confirm that the imported BSA interface and kernel resolve
to that checkout, not a different editable installation.

```bash
python -c 'import cudnn.block_sparse_attention._interface as m; print(m.__file__)'
(cd test/python && CUDA_VISIBLE_DEVICES=0 CUDNN_TEST_NO_ISOLATION=1 \
  python -m pytest -q fe_api/bsa -m L0)

mkdir -p agent/agent_space agent/agent_benchmark agent/agent_profiles

# Full native-versus-blk64 comparison.
CUDA_VISIBLE_DEVICES=0 python benchmark/bsa/benchmark_sm120_fp8_blk128.py \
  --sequence 142720 --heads 8 --densities 0.15 0.20 \
  --patterns strided local --warmup 10 --repeats 41 \
  --output agent/agent_benchmark/fp8_blk64_vs_blk128.json

# Exact fixed-baseline source preserved in this PR's first commit.
git show 5566716e:python/cudnn/block_sparse_attention/csrc/fwd/sm120_blk128/bsa_fwd_sm120_fp8.py \
  > agent/agent_space/fp8_fixed_baseline.py

CUDA_VISIBLE_DEVICES=0 python benchmark/bsa/benchmark_sm120_fp8_revision.py \
  --baseline-kernel agent/agent_space/fp8_fixed_baseline.py \
  --sequence 142720 --heads 8 --densities 0.15 0.20 \
  --patterns strided local --warmup 10 --repeats 41 --seed 120128 \
  --output agent/agent_benchmark/fp8_fixed_baseline_comparison.json

# Collect separately from latency timing. Use --profile baseline to profile
# the baseline into a different export path.
CUDA_VISIBLE_DEVICES=0 ncu --target-processes all \
  --kernel-name 'regex:.*BlockSparseAttnForwardFp8Sm120Blk128.*' \
  --launch-count 1 --set full --export agent/agent_profiles/fp8_candidate \
  python benchmark/bsa/benchmark_sm120_fp8_revision.py \
  --baseline-kernel agent/agent_space/fp8_fixed_baseline.py \
  --densities 0.20 --patterns strided --profile candidate
```

The revision benchmark executes the supplied Python source: use only a
trusted archived kernel. It records source hashes, compiler options and
raw timing samples, and rejects mismatched complete output bytes before
reporting speed. It uses the current quantizer on both sides; future
quantizer changes require archiving that baseline separately.

This PR adds no multi-GPU FP8 execution, communication overlap, backward,
new public prequantized-input API, or production-mask speed guarantee.
The earlier BF16 and distributed work is separate.
