# Wan 2.1/2.2 VAE benchmark

`run_model.py` compares compiled Diffusers with an encoder or decoder using
public cuDNN Conv3D/post-operation APIs on SM100/SM103. Both compile the selected
component as a full graph and retain Diffusers temporal orchestration.
Both preserve precision casts and disable Inductor's automatic intermediate
padding, which miscompiles channels-last repeat/reshape shortcuts in PyTorch 2.13.

BF16 uses integrated convolution/normalization at C96/C192/C160/C320/C256.
TF32 fuses C160 normalization, uses separate C160 residual/C320 preparation,
and uses native C256/C640 convolution. Decoder C96/C192 TF32 uses separate residual
preparation. Strict FP32 uses native convolution throughout. First-frame convolutions use
Conv2D for TF32 and at BF16 C96/C192/C256; BF16 C160/C320/C640 retain fusion.
Strict FP32 runs spatial convolutions, including first-frame shortcuts, as
one-plane native Conv3D with channels-first operands and channels-last output.
This avoids oversized Conv2D FFT workspaces.
C384 uses custom TF32 and native BF16 because custom BF16 accumulation
differences amplify in Wan 2.1. Bias, residual, normalization, SiLU, padding and cache preparation
are fused around native convolutions too. Torch handles attention, resampling and output
convolutions. The library's causal input API accelerates the C12 encoder input;
decoder latent input uses native convolution.

Native BF16 convolutions with 512 or more output channels and Wan 2.1 latent
input (C16→C384) retain Conv3D: first-frame Conv2D rounding errors amplify
through the decoder.

Both decoders carry preparation across residual blocks within each stage. The library owns output storage; callers fill history regions required by its
fused-convolution contract. These copies and allocations are timed. Compilation,
validation and warmup are excluded. First-call time and peak additional allocated
memory are reported separately. Each path warms up immediately before its
consecutive CUDA-event timing samples to settle GPU clocks and power state.
Use `--profile-path` and CPU-correlated CUDA traces to attribute convolution time.

Install the benchmark dependencies and CUTLASS DSL 4.9.0.dev1 or later:

```bash
python -m pip install 'nvidia-cutlass-dsl[cu13]>=4.9.0.dev1' \
    -r benchmark/e2e/Wan2-VAE/requirements.txt
```

Supply a local Diffusers model directory or its `vae` subdirectory. The checkpoint
selects the architecture. Without `--model`, `--wan-version 2.1|2.2` selects seeded
synthetic weights.

```bash
python benchmark/e2e/Wan2-VAE/run_model.py \
    --model /path/to/Wan2.2-TI2V-5B-Diffusers --operation encode --dtype bf16
python benchmark/e2e/Wan2-VAE/run_model.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers --operation decode --dtype fp32 --allow-tf32
python benchmark/e2e/Wan2-VAE/run_model.py \
    --operation decode --dtype fp32 --no-allow-tf32 --height 128 --width 128
```

Defaults: N=1, 17 output/video frames, 480×832, three timing samples. Frames must
be 1 or 4n+1; spatial dimensions must be divisible by 8 for Wan 2.1 or 16 for
Wan 2.2. Use smaller resolutions for shorter strict-FP32 runs. The unmodified
reference's workspace size depends on cuDNN algorithm selection as well as resolution.
`--allow-tf32` permits convolution TF32 for both candidates; FP32 matrix multiplies
use strict arithmetic.

Validation uses eager Diffusers with the matching policy. Strict FP32 requires
`atol=rtol=1e-4`; BF16/TF32 encoders require mean and logvar NRMSE ≤2%/1%, and
decoders require PSNR ≥50 dB (range 2). Error metrics are printed before timing.
`--check-only` validates without timing; `--profile-path torch-compile|cudnn-fused`
captures a warmed profiler range. Validation checks numerical agreement and execution
with synthetic weights or the supplied model checkpoint.
