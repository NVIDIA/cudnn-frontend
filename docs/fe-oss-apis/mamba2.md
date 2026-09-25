# Mamba-2 SSD

The experimental Mamba-2 state-space duality (SSD) operation follows the
GDN/KDA graph architecture: `pygraph.mamba2` / `pygraph.mamba2_bwd`, a FROST
engine, and a PyTorch custom operator with autograd. The graph engine is
framework neutral. The PyTorch entry point is:

```python
from cudnn.linear_attention import mamba2

out, final_state = mamba2(
    x, dt, A, B, C,
    D=D, dt_bias=dt_bias, z=z, initial_state=initial_state,
    return_final_state=True,
    chunk_size=32,
    intermediate_dtype="float32",
    reuse_forward_states=False,
)
```

This is the SSD operation, including timestep preprocessing, skip connection
and optional SiLU gate. It does not include input/output projections, causal
convolution, or GatedRMSNorm. In a Mamba-2 block with a separate GatedRMSNorm,
leave `z=None` here and apply the block's normalization/gating afterward.

This specialization targets the N=128 SSD geometry in
[Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json)
and [Nemotron 3 Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/main/config.json):
head dimension 64, state size 128, and 8 B/C groups (64 or 128 heads).
It accepts only state size 128; this is not support for every Mamba-2 or
Nemotron-family configuration. The two native 64-column state tiles are an
internal decomposition of N=128 and do not expose an N=64 mode.

## Operation and supported tensors

For head `h`, group `g = h // (H // G)`, and token `t`:

```
delta[t,h] = softplus(dt[t,h] + dt_bias[h])
S[t,h,p,n] = exp(A[h] * delta[t,h]) * S[t-1,h,p,n]
             + delta[t,h] * x[t,h,p] * B[t,g,n]
y[t,h,p] = sum_n S[t,h,p,n] * C[t,g,n] + D[h] * x[t,h,p]
out[t,h,p] = y[t,h,p] * silu(z[t,h,p])  # when z is present
```

An absent initial state is zero; absent `D` and `dt_bias` are zero. Pass `A`
directly (typically `-exp(A_log)`), rather than `A_log`. Parameter transforms
outside this operation remain differentiable through ordinary PyTorch.

| Tensor | Shape | Dtype |
|---|---|---|
| `x`, optional `z`, `out` | `[B, L, H, 64]` | BF16 |
| `dt` | `[B, L, H]` | BF16 |
| `B`, `C` | `[B, L, G, 128]` | BF16 |
| `A`, optional `D`, optional `dt_bias` | `[H]` | FP32 |
| optional `initial_state`, `final_state` | `[B, H, 64, 128]` (value, state axes) | FP32 |

Requirements: SM100 (B200), CuTeDSL >= 4.7.0, positive dimensions, `H % G == 0`,
contiguous buffers with 16-byte aligned addresses, and `chunk_size=32`. The last
chunk may be partial. Each plan specializes its declared shape, dtype and
optional inputs. Other architectures, FP16, different head/state dimensions,
noncontiguous layouts, packed variable lengths, timestep clipping and per-channel
`D[H,P]` are not supported by this engine.

## Training and precision

First-order backward covers `x`, `dt`, `A`, `B`, `C`, `D`, `dt_bias`, `z` and
`initial_state`, including a gradient arriving from the final state. It also
supports losses depending only on the output or only on the final state.
Higher-order gradients are not supported.

`intermediate_dtype` is a numerical operation attribute, not an autotuning knob:

- `"float32"` (default): FP32 chunk checkpoints, adjoints and per-head B/C
  gradient partials. Gated backward also retains its cotangent in FP32.
- `"bfloat16"`: BF16 storage for those intermediates, with FP32 state
  accumulation and reductions. This reduces traffic and changes rounding.
  This mode requires z=None: BF16 state rounding with the optional SSD SiLU
  gate can amplify cancellation in parameter gradients. Use FP32 intermediates
  when supplying z; the engine rejects that combination with BF16 intermediates.

Both paths use BF16 tensor-core operands. Neither is an all-FP32 implementation.
The native gated forward keeps the raw output in FP32 until applying SiLU, and
saves a BF16 ungated output for backward, following the baseline's saved-output
contract. The gate backward accumulates the skip-weight gradient in FP32. With FP32
intermediates, the scan scales the unrounded gated cotangent and backward uses
high/low BF16 operands for the cotangent and state contractions. This avoids
losing the gate gradient in cancellation-sensitive parameter reductions. The
BF16 mode is supported only without this optional gate.

By default backward recomputes chunk-entry states. With
`reuse_forward_states=True`, forward saves `[B,H,ceil(L/32),64,128]` states instead.
At `B=2,L=2048,H=64` this retains 256 MiB in FP32 or 128 MiB in BF16. The choice
does not change the logical chunk size. Compare numerics for your training
distribution before selecting BF16 intermediates.

## Graph lifecycle

```python
import cudnn
import torch

graph = cudnn.pygraph()
# inputs is a dict of the tensors above, omitting absent optional inputs.
ports = {
    name: graph.tensor(list(t.shape), stride=list(t.stride()),
                       data_type=(cudnn.data_type.BFLOAT16
                                  if t.dtype == torch.bfloat16 else cudnn.data_type.FLOAT),
                       name=name)
    for name, t in inputs.items()
}
o, final, ungated, checkpoints = graph.mamba2(
    **ports, chunk_size=32, dt_softplus=True,
    output_final_state=True, save_state_checkpoints=False,
    intermediate_dtype="float32",
)
graph.build()  # selects mamba2_frost and compiles kernels from declarations
workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
out = torch.empty_like(inputs["x"])
state = torch.empty(o.dim[0], o.dim[2], 64, 128, device="cuda", dtype=torch.float32)
pack = {ports[name]: t for name, t in inputs.items()}
pack.update({o: out, final: state})
if ungated is not None:
    pack[ungated] = torch.empty_like(out)
graph.execute(pack, workspace=workspace)
```

The forward graph returns four slots; disabled optional outputs are `None`.
`ungated_out` exists when `z` is supplied. `state_checkpoints` exists when
`save_state_checkpoints=True`; set its declared dtype to BF16 explicitly if
using BF16 intermediates (its default inferred dtype is FP32).

The backward builder accepts forward inputs plus `dO`, optional `d_final_state`,
`ungated_out` (required exactly when `z` is present), and optional
`state_checkpoints`. It returns, in order:
`dX, dDt, dA, dB, dC, dD, d_dt_bias, dZ, d_initial_state`.
Gradients of absent optional inputs are `None`. Use the same `chunk_size` and
`intermediate_dtype` as forward.

The engine compiles at plan-build time, caches compiled artifacts through the
shared FROST cache, and carves all scratch from caller workspace. Execute does
not allocate, repack, copy to the host, or synchronize. Use separate workspace
and output buffers for overlapping executions, and build a plan per device.
The supplied cuDNN handle carries the launch stream.

The PyTorch adapter caches graphs by declarations per host thread and allocates
private output/workspace buffers per call. It supports `torch.compile` and CUDA
Graph capture after warming both forward and backward. Autograd materializes
broadcast/noncontiguous cotangents (such as those from `sum()`) before calling
the dense backward graph; direct graph calls reject noncontiguous buffers.

## Reproducible comparison

See [the benchmark](../../benchmark/linear_attention/benchmark_mamba2.py) for a
same-process comparison with an unmodified `state-spaces/mamba` checkout:

```bash
python benchmark/linear_attention/benchmark_mamba2.py \
  --mamba-repo /path/to/mamba \
  --intermediate-dtype bfloat16 --output mamba2_b200.json
```

The default shape is `B=2,L=2048,H=64,P=64,N=128,G=8`, BF16 I/O.
The native engine uses chunk size 32; Triton uses the Nemotron default of 128
(`--triton-chunk-size` can select another baseline). The script checks all
gradients and final states before timing and again from the captured buffers
after CUDA Graph replay. `--reuse-forward-states` measures the checkpointed
native route and records the choice. BF16-intermediate benchmarks cover the
supported ungated route; FP32-intermediate benchmarks also cover SiLU gating.
It measures CUDA Graph device time for complete forward, complete backward and
an actual forward-plus-backward pair, including timestep preprocessing and all
recomputations/reductions. Compilation, Python dispatch and allocation costs
are outside these device-time measurements. It records source hashes and
baseline commit/status; this is a component benchmark, not model throughput.
