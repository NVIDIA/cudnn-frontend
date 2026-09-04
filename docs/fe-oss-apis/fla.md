# FLA Integration Shims

**The FLA integration APIs are experimental and subject to change.**

`cudnn.fla` can replace selected
[`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
(FLA) entry points with cuDNN Frontend implementations. The adapters are
process-wide monkeypatches: they cover modules that already exist as well as
modules created after activation. Call them before compiling or tracing a
model.

## Activate targets

```python
import cudnn.fla

# Backward-compatible default: Gated Delta Rule and KDA.
cudnn.fla.accelerate_fla()

# Incrementally opt the dense FLA GatedMLP into the fused cuDNN SwiGLU MLP.
cudnn.fla.accelerate_fla(targets="gated_mlp")

# Opt FLA 0.5.2 decode short convolution into the native update.
cudnn.fla.accelerate_fla(targets="short_conv")

# A string or iterable is accepted; "gdn", "mlp", and "shortconv" are aliases.
cudnn.fla.accelerate_fla(targets=("gdn", "gated_mlp", "shortconv"))
```

`accelerate_fla(verbose=True, *, targets=None)` is incremental and idempotent.
With `targets=None`, it retains the original best-effort behavior and enables
the `gated_delta_rule` and `kda` targets that exist in the installed FLA. An
explicit target selection is atomic: if any requested target cannot be
validated, no new requested target is installed and the raised `ImportError`
includes the rejection reason.

The `gated_mlp` target currently admits exactly FLA 0.5.2's plain, local,
bias-free `swish` `GatedMLP` with fused SwiGLU, contiguous BF16 CUDA inputs and
weights, and an SM100-family device. Unsupported runtime configurations such as
tensor parallelism or DTensor, quantization, LoRA, parametrizations, hooks,
custom linears, other dtypes/layouts/devices, or graph compilation execute the
original FLA method. Typed unsupported-kernel declines also fall back;
unexpected binding, allocation, or launch errors propagate.

The opt-in `short_conv` target patches
`fla.modules.conv.triton.ops.causal_conv1d_update` and preserves FLA 0.5.2's
public call and return contract:

```python
y, cache = causal_conv1d_update(
    x, cache, residual=None, weight=weight, bias=None, activation="silu"
)
```

The native route is deliberately restricted to inference on compute
capabilities 8.0, 8.6, 8.7, 8.9, 9.0, 10.0, 10.3, 11.0, 12.0, and 12.1 with BF16, a
contiguous `[N, D, 4]` cache, contiguous `[D, 4]` weights, no residual or bias,
and `silu`/`swish`. Every admitted architecture uses the same one-row
functional schedule. Input layouts `[N, D]`, `[N, 1, D]`, and `[1, N, D]` are
normalized to `[N, D]` with zero-copy views. Compact X rows accept every D;
padded `(ld, 1)` rows, including slices of wider fused projections, require `ld > D`
and `ld % 8 == 0`. Output shape and cache object identity are preserved.
Everything else executes the saved original FLA callable. Typed
unsupported-kernel declines fall back, while unexpected native binding,
allocation, and launch failures propagate.

The current native kernel requires CUTLASS DSL 4.7 or newer. If only the
package-wide `cutedsl>=4.5` minimum is present and the native import is
unavailable, the adapter catches that typed `ImportError` and executes FLA's
original path. Hardware correctness coverage is SM80, SM89, SM90, and SM100;
the native kernel is additionally runtime-validated on SM103 and SM120.
SM86, SM87, SM110, and SM121 are compile-validated only. The SM110 kernel
cross-compiles with CUTLASS DSL 4.7, but no SM110 hardware execution is claimed.

The cuDNN adapter and native kernel are independent NVIDIA implementations.
They use FLA's documented interface and observable depthwise causal-convolution
semantics for compatibility; no FLA Triton kernel source is incorporated or
translated.

Use `benchmark/fla_short_conv_shim_sm100.py` for an exact patched-callable
comparison. It runs on every functionally admitted target, records the actual
hardware and software metadata, reports CUDA-graph replay separately from
steady-state eager host enqueue time, and refuses to emit timings until route,
output, mutable-state, cache-identity, and restore gates pass.

## Inspect and restore

```python
import cudnn.fla

cudnn.fla.is_accelerated()             # any live cuDNN FLA target
cudnn.fla.is_accelerated("gated_mlp") # one target ("mlp" also works)

cudnn.fla.mlp_last_path()  # "native", "fallback:<reason>", or "error:<type>"
cudnn.fla.last_path()      # most recent Gated Delta Rule route
cudnn.fla.short_conv_last_path() # most recent short-convolution route

cudnn.fla.restore_fla(targets="gated_mlp") # restore only the MLP target
cudnn.fla.restore_fla(targets="shortconv") # alias for the short-conv target
cudnn.fla.restore_fla()                    # restore every target owned by cuDNN
```

`restore_fla(*, targets=None)` restores only patches still owned by
`cudnn.fla`; it does not overwrite a later third-party replacement. The route
helpers are diagnostics for tests and benchmarks, not synchronization or
per-thread state.

## Installation

Install FLA separately. The dense MLP and short-convolution adapters are
version-gated to the validated release:

```bash
pip install flash-linear-attention==0.5.2
pip install "nvidia-cudnn-frontend[cutedsl]"
```

The `cutedsl` extra supplies the optional CUTLASS DSL and CUDA Python
dependencies required by the native `gated_mlp` and `short_conv` targets. To
require the native `short_conv` route instead of its typed fallback, also
ensure CUTLASS DSL 4.7 or newer is installed using the package variant that
matches the CUDA Toolkit:

```bash
# CUDA Toolkit 12.9
pip install "nvidia-cutlass-dsl>=4.7"

# CUDA Toolkit 13.3
pip install "nvidia-cutlass-dsl[cu13]>=4.7"
```
