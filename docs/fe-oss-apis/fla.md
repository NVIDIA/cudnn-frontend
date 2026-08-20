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

# A string or iterable is accepted; "gdn" and "mlp" are aliases.
cudnn.fla.accelerate_fla(targets=("gdn", "gated_mlp"))
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

## Inspect and restore

```python
import cudnn.fla

cudnn.fla.is_accelerated()             # any live cuDNN FLA target
cudnn.fla.is_accelerated("gated_mlp") # one target ("mlp" also works)

cudnn.fla.mlp_last_path()  # "native", "fallback:<reason>", or "error:<type>"
cudnn.fla.last_path()      # most recent Gated Delta Rule route

cudnn.fla.restore_fla(targets="gated_mlp") # restore only the MLP target
cudnn.fla.restore_fla()                    # restore every target owned by cuDNN
```

`restore_fla(*, targets=None)` restores only patches still owned by
`cudnn.fla`; it does not overwrite a later third-party replacement. The route
helpers are diagnostics for tests and benchmarks, not synchronization or
per-thread state.

## Installation

Install FLA separately. The dense MLP adapter is version-gated to the validated
release:

```bash
pip install flash-linear-attention==0.5.2
pip install "nvidia-cudnn-frontend[cutedsl]"
```

The `cutedsl` extra supplies the optional dependencies required by the fused
GEMM path used by the native `gated_mlp` target.
