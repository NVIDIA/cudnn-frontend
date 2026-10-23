# Choose a normalization family

Select a family from the axes it reduces, the state it maintains, and the statistics its backward path needs.

## Before you begin

Complete the [normalization tutorial](/tutorials/normalization). Identify
the axes to reduce, whether running state changes, and which saved statistics
the backward graph consumes before choosing a family.

| Family | Typical reduction | State and outputs |
| --- | --- | --- |
| BatchNorm | Batch and spatial dimensions per channel | Saved mean/inverse variance; optional running mean/variance |
| BatchNorm inference | Uses established channel statistics | No training update |
| LayerNorm | One or more trailing feature dimensions | Saved mean/inverse variance for backward |
| RMSNorm | Trailing feature dimensions without mean subtraction | Saved inverse variance for backward |
| InstanceNorm | Spatial dimensions per sample and channel | Saved statistics for backward |
| Adaptive LayerNorm | LayerNorm plus learned conditioning | Family-specific scale/bias outputs and gradients |

## Configure training

Use `norm_forward_phase.TRAINING` where the API exposes it. Materialize saved statistics when backward runs in a separate graph. Keep epsilon's lifetime valid as a pass-by-value or supported graph tensor. Bind running-state outputs if the family updates them.

## Use generated statistics

`genstats` returns sum and squared sum. `bn_finalize` combines statistics with scale, bias, epsilon, accumulation count, and optional running state. Do not substitute one family's statistics layout for another.

Canonical examples: [LayerNorm forward](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/layernorm_basic.py) and [LayerNorm backward](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/layernorm_backward_basic.py). The BatchNorm API page records the missing standalone-sample gap.

The same LayerNorm flow is implemented in
[C++ forward](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/layernorm_basic.cc) and
[C++ backward](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/layernorm_backward_basic.cc).

## Expected result

The selected family produces the expected normalized output and saved state.
For training, its backward graph consumes the exact forward statistics and
matches input, scale, and bias gradients with an independent calculation.
