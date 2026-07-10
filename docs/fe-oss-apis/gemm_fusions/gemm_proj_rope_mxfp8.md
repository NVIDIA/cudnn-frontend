# GEMM + RoPE + MXFP8 Projection (SM100)

**This is an experimental API and subject to change.**

## Overview

**Fused projection GEMM + per-head YARN RoPE + dual-direction MXFP8 quantize**: a persistent dense GEMM on NVIDIA Blackwell GPUs (SM100+) that projects bf16 activations, applies the Megatron MLA-YARN rotary embedding to each attention head's trailing rotary features, and MXFP8 (E4M3, block=32) quantizes the result in **both** the rowwise (D-direction) and columnwise (S-direction) layouts. Implemented with CUTLASS/CUTE.

Emitting both scale directions makes the output directly consumable by block-scaled matmuls that need either operand orientation. For example, in DeepSeek-V3 the rowwise output feeds the forward `QK^T` and the columnwise output feeds the backward `dK = dS^T · Q` on the cuDNN `is_input_fp8` attention path.

- **Inputs**: bf16 activations `x`, bf16 projection weight `w`, and bf16 rotary tables `cos`/`sin`.
- **Outputs**: rowwise and columnwise MXFP8 data (`out_fp8_row`, `out_fp8_col`) and their E8M0 scale factors (`out_scales_row`, `out_scales_col`).

The kernel is tuned for the DeepSeek-V3 Q up-projection shapes: `NUM_HEADS=128`, `HEAD_DIM=192` (`QK_NOPE=128` + `QK_ROPE=64`), MXFP8 `BLOCK=32`, tile `TILE_M=128` (one head per CTA).

### Shapes

- **Inputs**
  - `x`: `(tokens, Q_LORA)` — `tokens % TILE_M == 0`
  - `w`: `(Q_LORA, NUM_HEADS·HEAD_DIM)` when `w_out_in=False`, or the transformer-engine-native transposed `(NUM_HEADS·HEAD_DIM, Q_LORA)` when `w_out_in=True`
  - `cos`, `sin`: `(tokens, QK_ROPE)`

- **Outputs**
  - `out_fp8_row`, `out_fp8_col`: `(tokens, NUM_HEADS, HEAD_DIM)`
  - `out_scales_row`: `(tokens, NUM_HEADS, HEAD_DIM // BLOCK)`
  - `out_scales_col`: `(tokens // BLOCK, NUM_HEADS, HEAD_DIM)`

### Equations

Project and reshape per head, then apply the interleaved-in / halves-out YARN RoPE to the trailing `QK_ROPE` features of each head:

$$
Y[t, h, :] = (x \, W)\;\text{reshaped to}\;[\text{tokens}, \text{NUM\_HEADS}, \text{HEAD\_DIM}]
$$

$$
Y_{\text{pe}} = \operatorname{RoPE}(Y[\ldots, \text{QK\_NOPE}:],\; \cos, \sin)
$$

MXFP8 quantize with block size `BLOCK=32`, independently for each direction (E8M0 per-block scale, E4M3 data):

$$
(\text{out\_fp8\_row}, \text{out\_scales\_row}) = \operatorname{MXFP8}_{\text{D}}(Y)\quad\text{(blocks along HEAD\_DIM)}
$$

$$
(\text{out\_fp8\_col}, \text{out\_scales\_col}) = \operatorname{MXFP8}_{\text{S}}(Y)\quad\text{(blocks along tokens)}
$$

### Diagram

```text
x (tokens x Q_LORA), w (Q_LORA x NUM_HEADS*HEAD_DIM)
     |  GEMM
     v
   Y (tokens x NUM_HEADS x HEAD_DIM)  ---- per-head YARN RoPE on trailing QK_ROPE
     |
     +--> MXFP8 rowwise (D)  -> out_fp8_row, out_scales_row
     +--> MXFP8 columnwise(S)-> out_fp8_col, out_scales_col
```

## API Usage

### High-level wrapper
```python
result = gemm_proj_rope_mxfp8_wrapper_sm100(
    x,
    w,
    cos,
    sin,
    w_out_in=False,
    stream=None,
)
out_fp8_row, out_scales_row, out_fp8_col, out_scales_col = result
# Key access: result["out_fp8_row"], result["out_scales_col"], ...
```

### Class API
```python
from cuda.bindings import driver as cuda

op = GemmProjRopeMxfp8Sm100(
    sample_x=x,
    sample_w=w,
    sample_cos=cos,
    sample_sin=sin,
    sample_out_fp8_row=out_fp8_row,
    sample_out_scales_row=out_scales_row,
    sample_out_fp8_col=out_fp8_col,
    sample_out_scales_col=out_scales_col,
    w_out_in=False,
)
assert op.check_support()
op.compile()
op.execute(x, w, cos, sin, out_fp8_row, out_scales_row, out_fp8_col, out_scales_col, current_stream=None)
```

---

## Parameters

### Input/Output tensors
- Input **x**: `x` (wrapper) or `sample_x`/`x` (class)
  - Shape: `(tokens, Q_LORA)`; Dtype: `bfloat16`
- Input **w**: `w` (wrapper) or `sample_w`/`w` (class)
  - Shape: `(Q_LORA, NUM_HEADS·HEAD_DIM)` (`w_out_in=False`) or `(NUM_HEADS·HEAD_DIM, Q_LORA)` (`w_out_in=True`); Dtype: `bfloat16`
- Input **cos**, **sin**: Shape `(tokens, QK_ROPE)`; Dtype: `bfloat16`
- Output **out_fp8_row** / **out_fp8_col**: Shape `(tokens, NUM_HEADS, HEAD_DIM)`; Dtype: `float8_e4m3fn`
- Output **out_scales_row**: Shape `(tokens, NUM_HEADS, HEAD_DIM // BLOCK)`; Dtype: `uint8` (E8M0)
- Output **out_scales_col**: Shape `(tokens // BLOCK, NUM_HEADS, HEAD_DIM)`; Dtype: `uint8` (E8M0)

### Common parameters
- `w_out_in: bool`
  - Whether `w` is stored `[out, in]` (`True`) or `[in, out]` (`False`). The kernel consumes both as the logical `[out, in]` B operand via the cutlass major mode (like cuBLAS `transb`); no transposed copy is materialized. Default: `False`
- CUDA stream (`current_stream` in class API, `stream` in wrapper). Defaults to the current torch stream (required for CUDA-graph capture).

### Wrapper return values

Returns a `TupleDict` with keys `out_fp8_row`, `out_scales_row`, `out_fp8_col`, `out_scales_col`. Tuple unpacking order is `(out_fp8_row, out_scales_row, out_fp8_col, out_scales_col)`.

---

## Support surface and constraints

### Dtypes
- `x`, `w`, `cos`, `sin` must be `bfloat16`.
- `out_fp8_row`, `out_fp8_col` must be `float8_e4m3fn`; `out_scales_row`, `out_scales_col` must be `uint8` (E8M0).

### Shapes and divisibility
- `tokens % TILE_M == 0` (`TILE_M = 128`); no tail handling.
- The projected weight dimension must equal `NUM_HEADS·HEAD_DIM`.

### Environment
- Requires CUDA with SM100+ compute capability.

---

## Source provenance

Integrated from the DeepSeek-V3 MLA fused Q up-projection kernel developed for Megatron-LM MXFP8 training (Blackwell / customte CUTLASS 4.4.1). The public entry point `run(...)` and the pure-PyTorch oracle `gemm_proj_rope_mxfp8_reference(...)` live in `python/cudnn/gemm_proj_rope_mxfp8/gemm_proj_rope_mxfp8.py`.

## Installation

Requires the optional CuTeDSL dependencies:

```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

## Usage examples

For usage examples, see test cases in `test/python/fe_api/gemm/test_gemm_proj_rope_mxfp8.py`.
