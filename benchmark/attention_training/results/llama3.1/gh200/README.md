# GH200 attention training measurements

This directory records the Llama 3.1 405B-style attention configuration on one
NVIDIA GH200 480GB (SM90, 132 SMs, approximately 95 GiB visible HBM).
The model name describes attention dimensions; this is not a full model run.

The charts and measured CSV contain 68 successful rows: 66 from the original
210-case run below and two repaired 16k deterministic-backward rechecks.

| Backend / dtype | Measured | Skipped | Failed |
| --- | ---: | ---: | ---: |
| cuDNN BF16 | 26 | 0 | 4 |
| cuDNN FP8 | 30 | 0 | 0 |
| cuDNN MXFP8 | 0 | 0 | 30 |
| FROST BF16 | 0 | 30 | 0 |
| FROST FP8 | 0 | 30 | 0 |
| FROST MXFP8 | 0 | 0 | 30 |
| FlashAttention 4 BF16 | 10 | 0 | 20 |

`run-outcomes.json` retains every case's disposition and the raw CSV hashes.
Failed or skipped cases are not timing measurements.

## Source and environment

- Frontend and canonical benchmark: `ddf1dd36007633710c468f7fef1acb3897d7f6f4`.
- PyTorch: 2.14.1+cu130; CUDA toolkit: 13.2.
- Native cuDNN / FROST environment: cuDNN 9.27.0.42, CuTeDSL 4.8.1.
- FlashAttention 4 environment: 4.0.0b3, CuTeDSL 4.5.0, cuDNN 9.24.0.43
  installed alongside FA4. FA4 executes its own kernels; the CSV preserves this
  environment's distinct cuDNN version.
- Original runs selected `libcudart.so.12` when both CUDA runtime libraries were
  visible. The original warning is preserved in raw logs.
- Batch 2, Q/KV heads 64/8, head dimension 128; sequence lengths
  2048, 4096, 8192, 16384, 32768; causal and noncausal masks.
- Ten iterations and no explicit warmup, following the canonical configuration.
  The runner reports the median of the last five measurements when there are
  more than five. `peak_mma_tflops` uses sampled SM clocks.
- `skip_ref=True`: `max_diff=0` is the runner's placeholder and does not certify
  numerical accuracy. These measurements are not correctness tests.

## Negative results

The BF16 deterministic-backward failures at 32k request approximately 258 GiB
of workspace. The 16k failures request a second approximately 65 GiB workspace
while the first allocation is still live. The latter is a runner allocation
lifetime issue, not evidence that a single 65 GiB workspace exceeds this GPU's
capacity. The workspace reuse repair then passed both 16k masks (causal and noncausal),
adding two measurements without replacing the original failure records. These
rechecks explicitly select `libcudart.so.13` and use the repaired benchmark
source hash recorded in `run-outcomes.json`. There were 212 case executions
in total; the final CSV has 28 BF16, 30 FP8 and 10 FA4 measurements.

MXFP8 has no available engine for these SM90 cases. The FROST D128 BF16/FP8
cases are declined. FA4 backward fails at `get_smem_store_C` with a duplicate
`transpose` argument; only its forward measurements are present.
