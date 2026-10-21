# QSA measured results

QSA forward and backward are faster than unchanged Miles in all nine regular cases on B200 and B300 SXM6 AC. All six output checks pass. The close packed 4096-token, 24-query-head case also wins when the timing order is reversed. Ratios are Miles/QSA; values above 1 favor QSA. Times are milliseconds.

BF16 packed THD, D256, block size 4, and seed 17. The synthetic selector keeps up to 512 complete causal blocks plus the partial tail. Packed sequence ends are unaligned. These are synthetic inputs, not captured model inputs.

Each implementation runs in a fresh process. Input and saved-state hashes match within each comparison. All six outputs pass an independent FP32 reference before timing; both timed backward calls use the same reference FP32 output and natural-log LSE. Separate untimed checks also validate each implementation with its own forward state. TF32 is disabled. Checks use atol=rtol=0.04 and relative L2 error <=0.025.

Full public calls include validation, allocation, metadata preparation, helper kernels and casts. Regular rows use seven synchronized wall-time samples after two warmups. The close 4096/3/24:2 row uses 31 samples after five warmups in each timing order. Compilation, reference evaluation and input generation are excluded. These timings do not use graph replay. CUDA-event times, raw samples and memory measurements are retained in JSON.

## B200

| T / packed sequences / Q:KV heads | Fwd QSA | Fwd Miles | Ratio | Bwd QSA | Bwd Miles | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| 129 / 3 / 12:1 * | 0.0742 | 0.2477 | 3.337 | 0.2021 | 0.5461 | 2.702 |
| 511 / 3 / 24:2 * | 0.1091 | 0.2594 | 2.377 | 0.2586 | 0.6352 | 2.456 |
| 512 / 1 / 12:1 | 0.0964 | 0.2309 | 2.395 | 0.2595 | 0.8021 | 3.091 |
| 4096 / 3 / 12:1 | 0.3602 | 0.4620 | 1.283 | 1.0739 | 1.9278 | 1.795 |
| 4096 / 3 / 24:2 | 0.6453 | 0.6492 | 1.006 | 2.6033 | 2.8530 | 1.096 |
| 16384 / 3 / 12:1 | 1.4158 | 2.9092 | 2.055 | 9.5492 | 15.0821 | 1.579 |
| 16384 / 3 / 24:2 | 2.6981 | 5.4218 | 2.009 | 21.8752 | 26.6915 | 1.220 |
| 32768 / 1 / 12:1 | 7.3593 | 27.9527 | 3.798 | 29.5521 | 147.7206 | 4.999 |
| 32768 / 1 / 24:2 | 14.1841 | 54.7745 | 3.862 | 59.4311 | 292.1574 | 4.916 |

## B300 SXM6 AC

| T / packed sequences / Q:KV heads | Fwd QSA | Fwd Miles | Ratio | Bwd QSA | Bwd Miles | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| 129 / 3 / 12:1 * | 0.1214 | 0.3781 | 3.115 | 0.3254 | 0.7591 | 2.333 |
| 511 / 3 / 24:2 * | 0.1412 | 0.4066 | 2.880 | 0.3626 | 0.8395 | 2.316 |
| 512 / 1 / 12:1 | 0.1293 | 0.3532 | 2.731 | 0.3687 | 0.9837 | 2.668 |
| 4096 / 3 / 12:1 | 0.3788 | 0.5722 | 1.511 | 1.1519 | 2.0486 | 1.779 |
| 4096 / 3 / 24:2 | 0.6483 | 0.7449 | 1.149 | 2.6045 | 2.8725 | 1.103 |
| 16384 / 3 / 12:1 | 1.3599 | 2.9141 | 2.143 | 9.3299 | 14.5925 | 1.564 |
| 16384 / 3 / 24:2 | 2.5517 | 5.3448 | 2.095 | 21.2922 | 25.8366 | 1.213 |
| 32768 / 1 / 12:1 | 7.1977 | 26.9634 | 3.746 | 28.7633 | 141.7526 | 4.928 |
| 32768 / 1 / 24:2 | 13.8263 | 52.8195 | 3.820 | 56.8805 | 281.4240 | 4.948 |

`*` includes empty attention rows. Tables above use QSA-first timing order.

## Reverse-order check

For the close packed 4096/3/24:2 case, Miles runs first in a separate pair. Both directions remain faster with the same inputs and reference saved state.

| GPU | Fwd QSA | Fwd Miles | Ratio | Bwd QSA | Bwd Miles | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| B200 | 0.6451 | 0.6483 | 1.005 | 2.5982 | 2.8445 | 1.095 |
| B300 SXM6 AC | 0.6500 | 0.7472 | 1.150 | 2.6089 | 2.8715 | 1.101 |

## 128K diagnostic comparison

All QSA outputs pass at 131072 tokens for both head layouts. Original Miles forward fails at this length because its bitmap row offsets exceed signed Int32. The comparisons below use a separate diagnostic copy with only three bitmap row-offset multiplications changed to Int64. These ratios are against that labeled control, not unchanged Miles. The original baseline source remains intact.

| GPU / Q:KV heads | Fwd QSA | Fwd Int64 control | Ratio | Bwd QSA | Bwd Int64 control | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| B200 / 12:1 | 37.9545 | 438.7797 | 11.561 | 137.4828 | 2535.7479 | 18.444 |
| B200 / 24:2 | 68.4137 | 870.0762 | 12.718 | 390.7777 | 5087.5230 | 13.019 |
| B300 SXM6 AC / 12:1 | 37.0255 | 422.9162 | 11.422 | 133.6314 | 2444.2462 | 18.291 |
| B300 SXM6 AC / 24:2 | 66.6133 | 838.8596 | 12.593 | 384.8628 | 4879.6132 | 12.679 |

| GPU / Q:KV heads | Peak allocated GiB, fwd / bwd |
|---|---:|
| B200 / 12:1 | 25.42 / 17.80 |
| B200 / 24:2 | 30.81 / 31.56 |
| B300 SXM6 AC / 12:1 | 25.42 / 17.80 |
| B300 SXM6 AC / 24:2 | 30.81 / 31.56 |

At 128K, the bitmap uses 4 GiB and forward compact int32 block IDs use 16 GiB. The paired 12:1 backward uses half as many compact-index rows. Peak allocated memory includes live inputs and reference saved state. Allocator reserved bytes are recorded separately.

## Validation and environment

The measured source passes strict checks on B200 and B300 SXM6 AC with actual CuTeDSL 4.8.1 and 4.7: 71 explicit cases, 36 changed-input/current-stream replay states, 12 packed cases, 50 analytic states, 49 backward oracle cases, 40 cancellation cases and 35 pair/stream states. Each passes 52 public and ancillary tests with 4.8.1. With actual 4.7, 51 tests pass and one unavailable ancillary helper test is skipped; all QSA tests pass. The final native runs on B200 and B300 SXM6 AC pass all 11 cases and the reverse-order close case. Each implementation also passes separate all-six-output checks using its own forward output and LSE before timing. These runs do not establish a clean Compute Sanitizer result.

| Measurement device | Runtime name | SM | CUDA-visible GiB | Driver | CUDA |
|---|---|---:|---:|---|---|
| B200 | NVIDIA B200 | 100 | 178.39 | 615.71.09 | 13.4 |
| B300 SXM6 AC | NVIDIA B300 SXM6 AC | 103 | 267.73 | 615.71.09 | 13.4 |

Each run uses one GPU and eight bound CPU threads in nvcr.io/nvidia/pytorch:26.09-py3. GPU clocks were not locked. The source package ran with a matching source-built native extension.

B200: PyTorch 2.14.0a0+b2c75dd062.nv26.09, Triton 3.8.0+gitbd40f349.nv26.9, CuTeDSL 4.8.1.

B300 SXM6 AC: PyTorch 2.14.0a0+b2c75dd062.nv26.09, Triton 3.8.0+gitbd40f349.nv26.9, CuTeDSL 4.8.1.

## Source

Frontend base commit: `e1abc1aeb035e28962915c785ccf79e1f16e5098`. The hashes below identify the measured, then-uncommitted sources. Frozen source archive SHA256: `84f4b2e66b6658e968e8a9649993457f6a197bd6add66b331d568228a6fd768d`. Installed distribution metadata reports frontend 1.28.0; the loaded Python package and native extension came from the matching source build.

| File | SHA256 |
|---|---|
| api.py | `5d1361c9fa2ffb19101b7e00cec024029252827c6c3e77f9e0b487de2e942cb8` |
| _contract.py | `7755ad31b469d3c5fdcd56c359cf8f0144960995a237b29e8784ce006b5cfa4f` |
| _forward.py | `32c91fa514a3f3587a6666f7a44c444da777833ee6e186024bf1c3fa1f8c4331` |
| _metadata.py | `97c37b5895f21cf0455a8a1b125ffabe3d211ade4e38e304eb087b2c1ce3777d` |
| kernels/fprop_dense.py | `624148e15e1ca138a95aa6d62438a98b022011a1406440233b329104c6c0fd25` |
| kernels/fprop_prepare.py | `f65f2b27f84aea4bcbdffb148614f400903ea6cfa6c7a838e36ee53c38baab9e` |
| kernels/fprop_sm100.py | `30b609cfa7941ae5cde409f2ab47c8af6eaacd4d9a2e069a068433bfc0d05889` |
| kernels/bprop_sm100.py | `98e3043ad4dd56514dd98c8388634167073c613cd2bc6b92b8dd1b3b517861b9` |
| kernels/bprop_sm100_base.py | `6d36fb785f67ef45ad200f4df73e8b1c53104ecb5bd6c95b68ab7cc18293259f` |
| kernels/bprop_helpers.py | `22121e371c0456b8c48ac45e8e02358822c8407639d53314771210f897b44a05` |
| benchmark_qsa.py | `fd4f00408e5a0a28dc8c138364f6549cf5ee1fbae00361e0d08de8a33a8deda2` |
| reference.py | `a069f4314d9257540cd5f6172eaa49d7ba123da6c1bf0a6205a9d7e00b241ece` |
| Native extension (x86_64) | `04afbade3c56e96db84d3e6ac2201628b921c31d82dbf2ba919f51b63395edc3` |
| Original Miles block_sparse_attention.py | `18542596944a5c3f4a231ce53f9d660f65a23246072ff060223a23e2d1cfbae1` |
| Miles Int64-only diagnostic copy | `7119ae432da78f265b3b37de783cbd5e45a2c4b919c59e3a6481edfbc9c4b8b8` |

Miles commit: `0cc65167f991ab1f0d228f6c9b648d89bc60a203`. No Miles source is included in this repository. The Int64 diagnostic copy is used only in the labeled 128K table.

Raw JSON, hashes and logs are retained outside the repository. See [README.md](README.md) for the command and timing boundary.
