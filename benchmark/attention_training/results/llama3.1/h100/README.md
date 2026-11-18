# H100 SXM attention training measurements

Llama 3.1 405B-style attention on NVIDIA H100 80GB HBM3 (SM90). This is an
attention microbenchmark, not a full model run. Two GPUs in one host ran
independent causal/noncausal mask shards; each timed case used one GPU.

The measured CSV contains 58 successful cases: 28 BF16 and 30 FP8 native cuDNN
rows. Two BF16 deterministic-backward cases at sequence length 32768 request
approximately 258 GiB of workspace and failed with OOM. `run-outcomes.json`
records all 60 outcomes and all 60 outcomes from the original runner.

The original runner measured 56 rows: its 16k deterministic-backward cases
also failed because each iteration allocated a new approximately 65 GiB
workspace before releasing the previous one. Reusing the workspace for the
fixed graph plans makes both 16k cases fit. No tensor dimensions, numerical
tolerances or timing regions changed. A separate allocation-budget probe
fails on the original runner and passes after the repair, for both passes.

## Reproduction and limits

- Base frontend / benchmark: `ddf1dd36007633710c468f7fef1acb3897d7f6f4` plus
  the workspace reuse change in this contribution. The modified benchmark
  file's SHA256 is recorded in `run-outcomes.json`.
- PyTorch 2.14.1+cu130, CUDA toolkit 13.0, cuDNN 9.27.0.42, CuTeDSL 4.8.1.
  `CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13` selected the runtime explicitly.
- Batch 2; Q/KV heads 64/8; head dimension 128; sequence lengths
  2048, 4096, 8192, 16384 and 32768. BF16 / FP8; forward, backward and
  deterministic backward; causal / noncausal masks.
- Ten iterations, zero explicit warmup. The canonical runner reports the
  median of the final five measurements. The SOL denominator uses sampled
  SM clocks. Concurrent shards can share host power/clock effects.
- `skip_ref=True`: `max_diff=0` is a placeholder, not numerical validation.
- This finite native-cuDNN sweep does not measure FROST D128, MXFP8 or FA4.
  Those combinations were excluded before rental based on the GH200
  compatibility results and the fixed dependency stack.

The PyTorch wheel metadata pins cuDNN 9.24.0.43; this environment deliberately
loads 9.27.0.42 to match the native GH200 measurements. The loaded version was
verified at runtime. Synthetic CP2 forward/backward correctness was tested
separately and is not inferred from these performance rows.
