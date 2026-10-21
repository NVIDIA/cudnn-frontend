# QSA measured results

This report covers the contiguous16 forward and optimized backward snapshot below. Native wins at 512 and 32768 tokens; packed 4096/16384 results are mixed. Ratios are Miles/native, so values above 1 favor native. All times are milliseconds.

BF16 packed THD, D256, block size 4, and seed 17. The synthetic selector keeps up to 512 complete causal blocks plus the partial tail. Packed sequence ends are unaligned. These are synthetic inputs, not captured model inputs.

Each implementation runs in a fresh process. Input and saved-state hashes match within every comparison. All six outputs pass the independent FP32 reference before timing; both backward calls use the same reference FP32 output and LSE. TF32 is disabled, including the environment override. Checks use atol=rtol=0.04 and relative L2 error <=0.025.

Full calls include metadata compaction, allocation, helper kernels and casts. B200/B300 results use the median of seven synchronized wall-time samples after two warmups. Compilation, reference evaluation and input generation are excluded. CUDA event times, raw samples and memory measurements are retained in JSON.

## B200

| T / packed sequences / Q:KV heads | Fwd native | Fwd Miles | Ratio | Bwd native | Bwd Miles | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| 129 / 3 / 12:1 * | 0.162 | 0.246 | 1.52 | 0.212 | 0.560 | 2.65 |
| 511 / 3 / 24:2 * | 0.202 | 0.276 | 1.37 | 0.275 | 0.646 | 2.35 |
| 512 / 1 / 12:1 | 0.179 | 0.235 | 1.31 | 0.331 | 0.815 | 2.46 |
| 4096 / 3 / 12:1 | 0.465 | 0.473 | 1.02 | 2.238 | 1.945 | 0.87 |
| 4096 / 3 / 24:2 | 0.774 | 0.676 | 0.87 | 2.515 | 2.872 | 1.14 |
| 16384 / 3 / 12:1 | 3.274 | 2.916 | 0.89 | 18.637 | 15.076 | 0.81 |
| 16384 / 3 / 24:2 | 6.266 | 5.425 | 0.87 | 21.368 | 26.693 | 1.25 |
| 32768 / 1 / 12:1 | 9.436 | 27.972 | 2.96 | 43.557 | 147.571 | 3.39 |

## B300

| T / packed sequences / Q:KV heads | Fwd native | Fwd Miles | Ratio | Bwd native | Bwd Miles | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| 129 / 3 / 12:1 * | 0.169 | 0.246 | 1.45 | 0.218 | 0.578 | 2.65 |
| 511 / 3 / 24:2 * | 0.189 | 0.268 | 1.42 | 0.265 | 0.647 | 2.44 |
| 512 / 1 / 12:1 | 0.181 | 0.244 | 1.35 | 0.325 | 0.804 | 2.48 |
| 4096 / 3 / 12:1 | 0.456 | 0.457 | 1.00 | 2.160 | 1.861 | 0.86 |
| 4096 / 3 / 24:2 | 0.754 | 0.651 | 0.86 | 2.413 | 2.738 | 1.13 |
| 16384 / 3 / 12:1 | 3.171 | 2.786 | 0.88 | 17.852 | 14.355 | 0.80 |
| 16384 / 3 / 24:2 | 6.063 | 5.202 | 0.86 | 20.481 | 25.500 | 1.25 |
| 32768 / 1 / 12:1 | 9.132 | 26.683 | 2.92 | 41.823 | 140.565 | 3.36 |

`*` includes empty attention rows.

## B200/B300 128K validation

All native outputs pass at 131072 tokens for both head layouts. The original Miles forward fails at this length on B200 and B300, so no 128K ratio is reported. A separate B300 control used identical inputs and changed only three bitmap row-offset multiplications to Int64: the original failed and the control passed. The original baseline remains unchanged.

| GPU | Q:KV heads | Fwd native | Bwd native | Peak allocated GiB, fwd / bwd |
|---|---:|---:|---:|---:|
| B200 | 12:1 | 45.254 | 190.920 | 25.42 / 25.80 |
| B200 | 24:2 | 82.965 | 387.816 | 30.81 / 31.56 |
| B300 | 12:1 | 43.836 | 182.220 | 25.42 / 25.80 |
| B300 | 24:2 | 80.245 | 376.798 | 30.81 / 31.56 |

The 128K bitmap uses 4 GiB; dense native int32 block IDs use 16 GiB. Peak allocated memory includes live inputs and reference saved state. Allocator reserved bytes are recorded separately in JSON.

## Environment and source

B200: SM100, 178.39 GiB CUDA-visible memory. B300 SXM6 AC: SM103, 267.73 GiB CUDA-visible memory. Both use one GPU and eight bound CPU threads, driver 615.71.09, CUDA 13.4, PyTorch 2.14.0a0+b2c75dd062.nv26.09, Triton 3.8.0+gitbd40f349.nv26.9, and CuTeDSL 4.8.1 in nvcr.io/nvidia/pytorch:26.09-py3. GPU clocks were not locked.

Both devices passed 38 focused source tests, masked-data poison checks and CUDA graph replay checks. The actual source package initializer ran with a matching native extension. These runs do not establish a clean Compute Sanitizer result.

Frontend base commit: `d8b0a7ed56cd8e62bc1e05f367f339624f6dbb4e`. The hashes below identify the measured, then-uncommitted sources. The installed distribution metadata reports frontend 1.28.0; the loaded Python package and native extension came from the source build.

| File | SHA256 |
|---|---|
| api.py | `971dfd9dcb08a458dde154fccb615c8df62b6a47e73fd679728c771ee3ff788e` |
| _contract.py | `0b0dda4fbafa19a7a52ab939e25779cc6445fb94a78912e71bc724e7bd91ecd9` |
| _metadata.py | `126882d50e94c5596ffa22cb32e7c733701ebe32c57619a54f656121e071dcb5` |
| kernels/fprop_sm100.py | `663ae53de2c847df1526a7d3d1bd983fd20b3e2cabb7f54276045718bcf9eeab` |
| kernels/bprop_sm100.py | `8169f09b5b5f3359af4b5f48778e60bd479a1606b1fd371778ee056973052523` |
| kernels/bprop_helpers.py | `22121e371c0456b8c48ac45e8e02358822c8407639d53314771210f897b44a05` |
| benchmark_qsa.py | `fd3831b3efc01b7fe4da59108960478609a223d2cc85a7c38280334bd3a0da48` |
| reference.py (B200/B300) | `a069f4314d9257540cd5f6172eaa49d7ba123da6c1bf0a6205a9d7e00b241ece` |
| Native extension (x86_64) | `04afbade3c56e96db84d3e6ac2201628b921c31d82dbf2ba919f51b63395edc3` |
| Original Miles block_sparse_attention.py | `18542596944a5c3f4a231ce53f9d660f65a23246072ff060223a23e2d1cfbae1` |

Miles commit: `0cc65167f991ab1f0d228f6c9b648d89bc60a203`. No Miles source is included in this repository. The diagnostic Int64 copy is not used in these ratios.

Raw JSON and logs are retained separately from the repository. Use [README.md](README.md) for the reproducible command and timing boundary.
