# Causal Conv1d Decode Update

**This FE-OSS API is experimental and subject to change.**

`cudnn.ops.causal_conv1d_update` advances a mutable depthwise
causal-convolution cache in place and returns the output for one decode step.
The semantic tensor contract uses `weight[D, W]` and `conv_state[S, D, L]`,
where `L >= W - 1`. The first native implementation targets the width-four
short convolution used by GDN/KDA-style linear-attention blocks. Compilation,
architecture dispatch, output ownership, streams, and kernel schedules remain
private implementation details.

Without circular-buffer metadata, row `n`, channel `d`, and selected cache
slot `s` are updated as follows:

```text
history = concat(conv_state[s, d, :], x[n, d])
conv_state[s, d, :] = history[-L:]
acc[n, d] = sum_j history[-W + j] * weight[d, j] + bias[d]
output[n, d] = activation(acc[n, d])
```

The bias is zero when omitted. `activation=None` and `"identity"` return the
accumulator directly; `"silu"` and `"swish"` select the same fused SiLU
specialization. Identity and SiLU compile as separate kernels, so the unused
activation work is absent from the identity path.

When `conv_state_indices` is omitted, row `n` selects slot `n`. When it is
present, `s = conv_state_indices[n]`. A value of `-1` denotes a padding row:
its output is zero and it does not mutate `conv_state`. Other selected slots
must be in range and unique within the decode batch because `conv_state` is
mutated; repeated padding rows are allowed.

`cache_seqlens[n]`, when present, treats the selected state as a circular
buffer: the new token is written at `cache_seqlens[n] % L`, and the previous
`W - 1` tokens are read modulo `L`. `cache_seqlens` itself is not mutated.
The current native kernel declines this circular-buffer mode; it is part of
the semantic API so adding the implementation does not require another public
signature.

## Python API

```python
import torch
import cudnn

x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(2048, 4, device="cuda", dtype=torch.bfloat16)
conv_state = torch.randn(8, 2048, 4, device="cuda", dtype=torch.bfloat16)

output = cudnn.ops.causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation="silu",
)
```

The signature is:

```python
cudnn.ops.causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    *,
    cache_seqlens=None,
    conv_state_indices=None,
) -> torch.Tensor
```

`conv_state` is mutated in place and the return value is an ordinary newly
allocated Tensor, not a wrapper result. The operation registers the mutation
with `torch.library`, including its FakeTensor contract. It is inference-only;
autograd inputs are rejected.

The fourth and fifth positional arguments remain `bias` and `activation`.
State-routing metadata is keyword-only. The first call for a supported device,
shape, optional-input signature, and activation performs JIT compilation.
Warm the exact signature before latency measurement or CUDA Graph capture.

## Semantic tensor contract

- `x`: `[N, D]` for one decode token
- `weight`: `[D, W]`
- `conv_state`: `[S, D, L]`, updated in place, with `L >= W - 1`
- `bias`: optional `[D]`
- `cache_seqlens`: optional int32 `[N]` circular-buffer positions
- `conv_state_indices`: optional int32 `[N]` state-slot selection; `-1` is padding
- output: `[N, D]`

A future multi-token extension can admit `x[N, D, Tstep]` without changing
the state or weight meanings. The current public implementation rejects 3D
`x` rather than silently interpreting its layout.

## Current native implementation

- GPU: compute capabilities 8.0, 8.6, 8.7, 8.9, 9.0, 10.0, 10.3, 11.0,
  12.0, and 12.1; the portable one-row schedule is used on every admitted
  target
- performance-characterized GPU: B200 SM100
- `x`: contiguous BF16 `[N, D]`
- `weight`: contiguous BF16 `[D, 4]`
- `conv_state`: contiguous BF16 `[S, D, L]` with `L` equal to 3 or 4,
  updated in place
- `bias`: optional contiguous BF16 `[D]`
- `cache_seqlens`: must be omitted
- `conv_state_indices`: optional contiguous CUDA int32 `[N]`
- output: contiguous BF16 `[N, D]`
- activation: identity or SiLU
- autograd: unsupported
- pointer alignment: 16 bytes for BF16 tensors and 4 bytes for indices

The native kernel requires CUTLASS DSL 4.7 or newer for the inline-PTX
integration it imports. The package-wide `cutedsl` extra keeps its broader
`>=4.5` floor for unrelated APIs; installing only that minimum is not
sufficient for this operation. The optional FLA 0.5.2 adapter treats the
resulting `ImportError` as a typed decline and executes FLA's original path.

Runtime correctness was validated on A100 SM80, L40S SM89, H200 SM90, B200
SM100, a GB110 board reporting SM103, and RTX 5080 SM120. SM86, SM87, SM110,
and SM121 have compile-only validation. The SM110 path is functional support,
not a training-performance claim.

For width four, `L=3` is the standard `W - 1` final state handed off by
prefill; it uses a functionally correct scalar state-access specialization.
`L=4` retains the original vectorized fast path used by GDN/KDA decode. Other
semantically valid widths and state lengths, circular-buffer updates,
multi-token updates, speculative intermediate-state returns, prefill, and
training currently raise a clear unsupported-configuration error. The indexed
path is functional paged-state support; its duplicate-index validation is not
performance-characterized as a fast path. For the current native path, the
kernel checks indices on device and traps on values below `-1`, out-of-range
slots, or duplicate non-padding slots;
the resulting CUDA error is asynchronous and the failed update is not
transactional.

## Semantic provenance and benchmarking

Behavioral references are FLA 0.5.2 `ShortConvolution.step` (MIT) and the
public `causal_conv1d_update` contract from Dao-AILab/causal-conv1d at revision
`cd81f0413cad2fc1e6f17e785ac39f59aae690cd` (BSD-3-Clause). No source from
either project is included. The implementation uses CUTLASS/CuTe DSL, inline
PTX, and in-tree NVIDIA FROST primitives.

Use `benchmark/causal_conv1d_update_sm100.py` for route-proof, correctness, and
an interleaved comparison against FLA. It intentionally uses the private
preallocated plan so kernel timing does not include public output allocation
or custom-op dispatch. The benchmark records the actual GPU architecture and
software environment; Slurm metadata is optional.
