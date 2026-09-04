# Grouped weight-only NVFP4 projection (SM100/SM103)

**This is an experimental API and subject to change.** It requires the optional
CuTe DSL dependencies and a CUDA torch build:

```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

The domain-scoped `GroupedGemmWeightOnlyNvfp4` and
`grouped_gemm_weight_only_nvfp4` expose two measured grouped
W4A16 projections as a standalone operation. They deliberately are not new
top-level `cudnn.*` prepared-plan exports, and do not add a model-specific
matcher to the generic cuDNN graph compiler. The implementation is original to
this repository; its tensor shapes correspond to the NVFP4 checkpoint layout
used by NVIDIA Nemotron 3.5 Lightning at revision
`cc84af2fe71647d87f4486c064f320e1e7535243`.

## Operation

Expert `g` owns routed rows beginning at `first_token_offset[g]`; the next
expert's start is its end, and the final expert ends at `S`. Packed E2M1 weight
nibbles are dequantized in BF16 using one E4M3 scale per group of 16 K values.
The GEMM accumulates in FP32, multiplies by `factor[g]`, and rounds to BF16.

Two exact semantic variants are supported:

| `epilogue` | K | N | Result |
| --- | ---: | ---: | --- |
| `"linear"` | 1856 | 2688 | `bf16(factor[g] * A_g @ W_g.T)` |
| `"squared_relu"` | 2688 | 1856 | BF16 scale, ReLU, then BF16 self-square |

The shape-specific schedules and their tile sizes remain private implementation
details. Unsupported shapes or epilogues raise `NotImplementedError`; there is
no generic or graph-compiler fallback hidden behind this API.

## Tensor contract

For `E` experts and `S > 0` routed rows:

| Tensor | Physical shape | Dtype / layout |
| --- | --- | --- |
| `routed_tokens` | `(1, S, K)` | contiguous inner `(S,K)` plane, BF16 |
| `packed_weight` | `(E, N, K/2)` | contiguous `uint8`, low nibble first |
| `weight_scale` | `(E, N, K/16)` | contiguous E4M3 |
| `first_token_offset` | `(E, 1, 1)` | contiguous INT32 starts only |
| `factor` | `(E, 1, 1)` | contiguous FP32 |
| `output` | `(1, S, N)` | contiguous inner `(S,N)` plane, BF16 |

`E` must be in `[1, 128]`. All tensors must be on one CUDA device. Data tensors
must be 16-byte aligned; offsets and factors must be 4-byte aligned. The caller
must provide nondecreasing starts with `first_token_offset[0] == 0`, every start
in `[0, S]`, and may represent empty experts by repeated starts. These values
remain device-resident and are not copied to the host for validation.

## Wrapper API

```python
from cudnn.gemm.cutedsl.grouped.weight_only_nvfp4 import (
    grouped_gemm_weight_only_nvfp4,
)

result = grouped_gemm_weight_only_nvfp4(
    routed_tokens,
    packed_weight,
    weight_scale,
    first_token_offset,
    factor,
    epilogue="squared_relu",
)
output = result["output"]
```

The wrapper allocates only `output`. Its cached plan is specialized by device,
expert count, and epilogue; `S` is symbolic, so compatible routed-token extents
reuse the same compiled artifact.

## Class API

Use the class when output storage and compilation lifetime belong to the caller:

```python
from cudnn.gemm.cutedsl.grouped.weight_only_nvfp4 import (
    GroupedGemmWeightOnlyNvfp4,
)

op = GroupedGemmWeightOnlyNvfp4(
    routed_tokens,
    packed_weight,
    weight_scale,
    first_token_offset,
    factor,
    output,
    epilogue="linear",
)
op.check_support()
op.compile()
op.execute(
    routed_tokens,
    packed_weight,
    weight_scale,
    first_token_offset,
    factor,
    output,
)
```

`execute()` performs metadata and pointer validation, then launches exactly one
kernel. It does not allocate, convert, repack, synchronize, or read device data
to the host. The operation owns no workspace.
