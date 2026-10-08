# WAN 2.2 VAE encoder benchmark

`run_model.py` benchmarks complete WAN 2.2 VAE encoding. The baseline is the
unmodified Diffusers `AutoencoderKLWan` encoder compiled as a strict full graph.
The accelerated encoder uses the public cuDNN frontend Conv3D APIs for the input
convolution, all four residual stages, and the middle-block residual convolutions.
The existing standalone preparation API combines bias, optional residual,
RMSNorm, SiLU, causal padding, and cache updates between the split convolutions
and before the output-head convolution.

Attention, downsampling, stage shortcuts, and the output-head convolution remain
in Torch and are compiled around the direct API calls. This integration adds no
public kernels or APIs. The baseline compiles as one full graph; each direct cuDNN
call is an explicit graph boundary in the accelerated path.

Both paths must match eager Diffusers before timing. The runner alternates their
execution order and reports median CUDA-event latency and speedup, then calls
the unchanged [`_perfshare`](../_perfshare.py) harness for supplemental per-path
minimum timings and kernel shares. These are separate from the median A/B speedup;
kernel-name-based attribution is approximate, and fused kernels count as a whole.
Its fixed training/eager heading does not describe this forward-only benchmark.

Requires SM100/SM103, `nvidia-cutlass-dsl >= 4.9`, and the benchmark dependencies
in [`requirements.txt`](requirements.txt), which pins the Diffusers implementation:

```bash
python -m pip install -r benchmark/e2e/Wan2.2-VAE/requirements.txt
```

The default uses the production architecture with deterministic random weights:

```bash
python benchmark/e2e/Wan2.2-VAE/run_model.py
```

Use a local checkpoint for a model-faithful run:

```bash
python benchmark/e2e/Wan2.2-VAE/run_model.py \
    --model /path/to/Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

The benchmark never downloads weights. Random-weight results measure numerical
agreement and performance. Run `--help` for all shape, warmup, repeat, and
tolerance defaults.

The default input is batch 32, 17 frames at 480x832; it needs substantial GPU
memory. Reduce `--batch-size` for a smaller device; `--frames`, `--height`, and
`--width` also control the workload. Use `--check-only` for correctness without
timing, or `--profile-path torch-compile|cudnn-fused` for a warmed profiler range.
`run_model.py` sets the compiler recompile limit for the encoder's multiple chunk
shapes; callers using `wan_encoder.py` directly must configure this themselves.
