# SwiGLU + MXFP8 quantization

Experimental forward/backward kernels fuse SwiGLU with MXFP8 quantization.
The forward computes `silu(gate) * up`. The backward computes both gate and
up gradients from a BF16 output gradient. Values round to BF16 before MXFP8
quantization, without storing that intermediate.

```python
from cudnn.ops import gated_act_mxfp8_forward, gated_act_mxfp8_backward

row, col, row_sf, col_sf = gated_act_mxfp8_forward(x, rowwise=True, colwise=True)
drow, dcol, drow_sf, dcol_sf = gated_act_mxfp8_backward(x, dy)
```

`x` is contiguous BF16 `[M, 2K]`; the first K columns are the gate.
`dy` is contiguous BF16 `[M, K]`. M and K must be positive multiples of 128,
and `M * 2K <= 2**31`. Inputs require 32-byte alignment and one CUDA device.

Both functions return a `TupleDict` in this order:

| Field | Format |
|---|---|
| `row_output` | E4M3 `[M, N]`, strides `(N, 1)` |
| `col_output` | E4M3 `[M, N]`, strides `(1, M)` |
| `row_scales` | uint8 `[M, N/32]`, physical swizzled E8M0 bytes |
| `col_scales` | uint8 `[N*M/32]`, physical swizzled E8M0 bytes |

N is K for forward and 2K for backward. At least one orientation must be
enabled. A disabled orientation returns zero-length value and scale tensors.
Scales use RCEIL rounding per 32 values and the GEMM 128-by-4 swizzle;
their logical tensor shape does not undo that swizzle.

These are explicit forward/backward calls; forward does not install an
autograd rule. Repeated calls produce the same bytes on a fixed build/device.

## Prepared execution

`cudnn.gemm.cake.gated_act_mxfp8.GatedActMxfp8Plan(shape, backward=False,
rowwise=True, colwise=False, device=None)` compiles one selected kernel.
`execute(x, (row, col, row_sf, col_sf), grad_output=None, stream=None)`
validates and writes caller-owned outputs. Set `backward=True` and supply
`grad_output` for backward. Outputs must not overlap each other or inputs.
Disabled output pairs must be zero-length tensors of their declared dtype.

The plan needs zero workspace and owns no device memory. Execution launches
one kernel and allocates no device storage. Input/output metadata views do
not copy data. The default is the current Torch stream; `stream=` accepts
the standard Frontend stream forms. Convenience calls allocate output
storage on that stream. Construct a plan before capture and use separate
output storage for concurrent calls.

## Requirements and source

SM100a, SM103a or SM107a; CUDA toolkit 13.4+; Torch and TVM-FFI.
No FlashInfer, Miles, or CuTeDSL import is required by this operation.
The CUDA source package is copied without changes from
[miles-ext-kernels, commit f55fb996](https://github.com/radixark/miles-ext-kernels/tree/f55fb99674be5edb0e1f6a0e4d6a911335da86ce/miles_ext_kernels/cake/_vendor/cake_gated_act_mxfp8_backend).
File hashes are recorded in `_cake/_vendor/VENDORED_GATED_ACT_MXFP8.json`.
